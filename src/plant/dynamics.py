"""Configuration and state construction for the single native C++ plant implementation."""

from __future__ import annotations

from collections import namedtuple
from pathlib import Path

import _rlv_sim_native
import numpy as np
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf

from core.interfaces import (
    ActuatorParams,
    AeroParams,
    AtmosphereParams,
    EarthRotationParams,
    EngineParams,
    MassPropsParams,
    PlantParams,
    PlantState,
    SloshParams,
    TerminationReason,
    VehicleGeom,
    WindParams,
)
from core.seeding import fold_in, rng_from_seed, seed_value, seed_words
from plant import aero_data

_MASS_EPS = 1e-3  # m_dry tolerance for fuel-depletion predicate


# --------------------------------------------------------------------------- #
# Build from config (Hydra/OmegaConf)
# --------------------------------------------------------------------------- #
PlantBundle = namedtuple("PlantBundle", ["params", "make_initial_state", "meta", "native"])


def _arr(x):
    return np.asarray(x, dtype=np.float64)


def build_params(cfg) -> tuple[PlantParams, dict]:
    """Construct PlantParams + a meta dict (model names, grids) from a resolved config."""
    v = cfg.plant.vehicle
    geom = VehicleGeom(
        s_ref=_arr(v.geom.s_ref),
        d_ref=_arr(v.geom.d_ref),
        length=_arr(v.geom.length),
        x_eng=_arr(v.geom.x_eng),
        fwd_axis=_arr(v.geom.fwd_axis),
    )
    mass = MassPropsParams(
        m_dry=_arr(v.mass.m_dry),
        m_wet=_arr(v.mass.m_wet),
        x_cg_dry=_arr(v.mass.x_cg_dry),
        x_cg_wet=_arr(v.mass.x_cg_wet),
        I_dry=_arr(v.mass.I_dry),
        I_wet=_arr(v.mass.I_wet),
        # Lateral CG offset [y,z] and products of inertia [Ixy,Ixz,Iyz].
        cg_lateral=_arr(v.mass.cg_lateral),
        inertia_products=_arr(v.mass.inertia_products),
    )
    ap = v.actuator
    actuator = ActuatorParams(
        p_sea=_arr(cfg.environment.atmosphere.p_sea),
        tvc_omega=_arr(ap.tvc_omega),
        tvc_zeta=_arr(ap.tvc_zeta),
        tvc_rate_max_rad_s=_arr(ap.tvc_rate_max_rad_s),
        tvc_deflection_max_rad=_arr(ap.tvc_deflection_max_rad),
        throttle_tau=_arr(ap.throttle_tau),
        throttle_rate_max=_arr(ap.throttle_rate_max),
        fin_omega=_arr(ap.fin_omega),
        fin_zeta=_arr(ap.fin_zeta),
        fin_rate_max_rad_s=_arr(ap.fin_rate_max_rad_s),
        fin_deflection_max_rad=_arr(ap.fin_deflection_max_rad),
        rcs_thrust=_arr(ap.rcs_thrust),
        rcs_arm=_arr(ap.rcs_arm),
        rcs_isp=_arr(ap.rcs_isp),
        rcs_fuel0=_arr(ap.rcs_fuel0),
        fin_lock=_arr(cfg.fault.fin_lock),
    )

    # --- engines ---
    engines, gimbal_idx, n_eng, n_gimbal = _build_engines(v.engines, cfg.fault, _arr)

    # --- aero table / fin geometry ---
    af = cfg.plant.aero
    backend = str(af.backend)
    if backend == "analytic":
        n_alpha, n_mach = int(af.grid.n_alpha), int(af.grid.n_mach)
        alpha_samples = np.linspace(0.0, np.deg2rad(af.grid.alpha_max_deg), n_alpha)
        mach_grid_np = np.linspace(af.grid.mach_min, af.grid.mach_max, n_mach)
        cos_grid, cd_t, cl_t, xcp_t, cd_s, cl_s, xcp_s = aero_data.generate_table(
            alpha_samples, mach_grid_np, float(v.geom.length), OmegaConf.to_container(af.coeffs)
        )
        # analytic backend has no C_m table: keep the moment on the x_cp path (moment_uses_cm=0).
        cm_t, cm_s, x_cg_ref_v, moment_uses_cm = (
            np.zeros_like(xcp_t),
            np.zeros_like(xcp_s),
            0.0,
            0.0,
        )
    elif backend == "file":
        table_path = Path(to_absolute_path(str(af.file.path)))
        (
            cos_grid,
            mach_grid_np,
            cd_t,
            cl_t,
            xcp_t,
            cd_s,
            cl_s,
            xcp_s,
            cm_t,
            cm_s,
            x_cg_ref_v,
        ) = aero_data.load_table_npz(table_path)
        moment_uses_cm = 1.0  # build the body moment from the AEDB C_m (not x_cp)
    else:
        raise ValueError(f"unknown table aero backend {backend!r}; pick analytic|file")
    mach_grid = _arr(mach_grid_np)

    # Fin arms and tangential normals at the configured azimuthal phase.
    nf = int(af.fins.n_fins)
    R = float(af.fins.fin_radius)
    x_fin = float(af.fins.fin_x)
    phase = np.deg2rad(float(af.fins.fin_phase_deg))
    phis = phase + np.linspace(0.0, 2 * np.pi, nf, endpoint=False)
    fin_arm = np.stack([np.full(nf, x_fin), R * np.cos(phis), R * np.sin(phis)], axis=1)
    fin_normal = np.stack([np.zeros(nf), -np.sin(phis), np.cos(phis)], axis=1)  # tangential
    if "mach_grid" in af.fins:
        fin_mach_grid = np.asarray(af.fins.mach_grid, dtype=float)
        fin_cla_table = np.asarray(af.fins.cla_table, dtype=float)
        fin_cd_delta2_table = np.asarray(af.fins.cd_delta2_table, dtype=float)
    else:
        fin_mach_grid = np.asarray([mach_grid_np[0], mach_grid_np[-1]], dtype=float)
        fin_cla_table = np.full(2, float(af.fins.fin_cla), dtype=float)
        fin_cd_delta2_table = np.zeros(2, dtype=float)
    aero = AeroParams(
        cos_grid=_arr(cos_grid),
        mach_grid=mach_grid,
        cd_table=_arr(cd_t),
        cl_table=_arr(cl_t),
        xcp_table=_arr(xcp_t),
        cd_slope=_arr(cd_s),
        cl_slope=_arr(cl_s),
        xcp_slope=_arr(xcp_s),
        cm_table=_arr(cm_t),
        cm_slope=_arr(cm_s),
        x_cg_ref=_arr(float(x_cg_ref_v)),
        moment_uses_cm=_arr(moment_uses_cm),
        n_fins=_arr(nf),
        fin_area=_arr(af.fins.fin_area),
        fin_cla=_arr(float(af.fins.fin_cla)),
        fin_cd0=_arr(float(af.fins.fin_cd0)),
        fin_arm=_arr(fin_arm),
        fin_normal=_arr(fin_normal),
        fin_stall_alpha=_arr(np.deg2rad(float(af.fins.fin_stall_deg))),
        fin_transonic_mach=_arr(float(af.fins.fin_transonic_mach)),
        fin_incremental=_arr(float(af.fins.incremental)),
        fin_mach_grid=_arr(fin_mach_grid),
        fin_cla_table=_arr(fin_cla_table),
        fin_cd_delta2_table=_arr(fin_cd_delta2_table),
        cla=_arr(float(af.linear.cla) if "linear" in af else 2.0),
        cda=_arr(float(af.linear.cda) if "linear" in af else 1.0),
    )

    atm = cfg.environment.atmosphere
    atmosphere = AtmosphereParams(
        rho0=_arr(atm.rho0),
        scale_height=_arr(atm.scale_height),
        a_sound=_arr(atm.a_sound),
        g0=_arr(atm.g0),
        delta_T=_arr(atm.delta_T),  # non-standard-day offset (ussa_poly); 0 = standard
        density_scale=_arr(atm.density_scale),  # density dispersion; 1 = nominal
    )
    w = cfg.environment.wind
    wind = WindParams(
        mean0=_arr(w.mean0),
        ou_tau=_arr(w.ou_tau),
        ou_sigma=_arr(w.ou_sigma),
        turb_L=_arr(w.turb_L),
        turb_sigma=_arr(w.turb_sigma),
    )
    e = cfg.environment.earth
    earth_model = str(e.model)
    if earth_model not in {"fixed", "rotating"}:
        raise ValueError(f"unknown Earth model {earth_model!r}; pick fixed|rotating")
    latitude_deg = float(e.latitude_deg)
    rotation_rate = float(e.rotation_rate_rad_s)
    if not np.isfinite(latitude_deg) or not -90.0 <= latitude_deg <= 90.0:
        raise ValueError("environment.earth.latitude_deg must be finite and in [-90, 90]")
    if not np.isfinite(rotation_rate) or rotation_rate < 0.0:
        raise ValueError("environment.earth.rotation_rate_rad_s must be finite and non-negative")
    earth_enabled = earth_model == "rotating"
    latitude_rad = np.deg2rad(latitude_deg)
    omega_ned = (
        rotation_rate * np.array([np.cos(latitude_rad), 0.0, -np.sin(latitude_rad)])
        if earth_enabled
        else np.zeros(3)
    )
    earth = EarthRotationParams(
        enabled=_arr(float(earth_enabled)),
        omega_ned=_arr(omega_ned),
        differential_centrifugal=_arr(
            float(earth_enabled and bool(e.include_differential_centrifugal))
        ),
    )
    s = cfg.plant.slosh
    slosh = SloshParams(
        slosh_mass=_arr(s.slosh_mass),
        pendulum_length=_arr(s.pendulum_length),
        omega=_arr(s.omega),
        zeta=_arr(s.zeta),
        x_attach=_arr(s.x_attach),
    )

    params = PlantParams(
        geom=geom,
        mass=mass,
        aero=aero,
        atmosphere=atmosphere,
        wind=wind,
        earth=earth,
        slosh=slosh,
        actuator=actuator,
        engines=engines,
        g_ned=_arr([0.0, 0.0, atm.g0]),
        plant_dt=_arr(cfg.plant_dt),
    )
    meta = {
        "aero_model": af.model,
        "aero_backend": backend,
        "atmosphere_model": atm.model,
        "wind_model": w.model,
        "earth_model": earth_model,
        "slosh_model": s.model,
        "massprops_model": v.massprops_model,
        "n_fins": nf,
        "n_engines": n_eng,
        "n_gimbal": n_gimbal,
        "gimbal_idx": gimbal_idx,  # static engine-to-gimbal mapping used by GNC allocation
        "p_sea": float(atm.p_sea),
        "scale_height": float(atm.scale_height),
        # Fidelity groups are resolved here; the native constructor rejects unknown model names.
        "actuator_model": str(cfg.plant.actuator.model),
        "thrust_model": str(cfg.plant.thrust.model),
    }
    return params, meta


def _build_engines(eng_cfg, fault_cfg, arr):
    """Build EngineParams + static (gimbal_idx, N, Ng) from the vehicle `engines:` config."""
    n = int(eng_cfg.n_engines)
    x_eng = float(eng_cfg.x_eng)
    layout = str(eng_cfg.layout)
    if layout == "single":
        positions = np.array([[x_eng, 0.0, 0.0]])
    elif layout == "triad":
        ring_r = float(eng_cfg.ring_radius)
        positions = np.array([[x_eng, 0.0, 0.0], [x_eng, ring_r, 0.0], [x_eng, -ring_r, 0.0]])
    elif layout == "octaweb":
        ring_r = float(eng_cfg.ring_radius)
        phi = np.linspace(0.0, 2 * np.pi, n - 1, endpoint=False)
        ring = np.stack([np.full(n - 1, x_eng), ring_r * np.cos(phi), ring_r * np.sin(phi)], axis=1)
        positions = np.vstack([[[x_eng, 0.0, 0.0]], ring])
    else:
        raise ValueError(f"unknown engine layout: {layout}")
    if positions.shape[0] != n:
        raise ValueError(f"layout {layout} produced {positions.shape[0]} engines, expected {n}")

    gimbal_mask = np.asarray(eng_cfg.gimbal_mask, dtype=int)
    gimbal_idx = [i for i, m in enumerate(gimbal_mask) if m]
    n_gimbal = len(gimbal_idx)

    def bcast(val, length):
        a = np.atleast_1d(np.asarray(val, dtype=float))
        if a.size == 1:
            return np.full(length, float(a.reshape(-1)[0]))
        if a.size != length:
            raise ValueError(f"expected scalar or length-{length}, got size {a.size}")
        return a.astype(float)

    cant = np.zeros((n, 2))  # per-engine cant

    # faults (truth-only): thrust_scale (N,), engine-out by index (N-agnostic), tvc_bias (Ng,2)
    thrust_scale = bcast(fault_cfg.thrust_scale, n)
    engine_on = np.ones(n)
    out_idx = fault_cfg.engine_out_idx
    if out_idx is not None:
        out_idx = int(out_idx)
        if not 0 <= out_idx < n:
            raise ValueError(
                f"engine_out_idx={out_idx} requires n_engines>={out_idx + 1}, vehicle has {n}"
            )
        engine_on[out_idx] = 0.0
    tb = np.asarray(fault_cfg.tvc_bias, dtype=float)
    if tb.shape == (2,):
        tvc_bias = np.tile(tb, (n_gimbal, 1))
    elif tb.shape == (n_gimbal, 2):
        tvc_bias = tb
    else:
        raise ValueError(f"tvc_bias must be (2,) or ({n_gimbal},2), got {tb.shape}")

    engines = EngineParams(
        positions=arr(positions),
        thrust_vac=arr(bcast(eng_cfg.thrust_vac, n)),
        throttle_min=arr(bcast(eng_cfg.throttle_min, n)),
        isp=arr(bcast(eng_cfg.isp, n)),
        nozzle_exit_area=arr(bcast(eng_cfg.nozzle_exit_area, n)),
        cant=arr(cant),
        engine_on=arr(engine_on),
        thrust_scale=arr(thrust_scale),
        tvc_bias=arr(tvc_bias),
    )
    return engines, np.asarray(gimbal_idx, dtype=np.int32), n, n_gimbal


def build_plant(cfg) -> PlantBundle:
    """Resolve config into the single native plant implementation."""
    params, meta = build_params(cfg)

    def make_initial_state(ic) -> PlantState:
        return initial_state(ic, params)

    # Immutable parameters cross the boundary once. The session then owns all mutable plant state.
    native = _rlv_sim_native.NativePlant(params, meta)
    return PlantBundle(
        params=params,
        make_initial_state=make_initial_state,
        meta=meta,
        native=native,
    )


def initial_state(ic, params: PlantParams) -> PlantState:
    """Build a PlantState from an initial-condition mapping ``ic`` (see scenario config)."""
    from core.frames import quat_from_euler

    q0 = quat_from_euler(_arr(ic["roll"]), _arr(ic["pitch"]), _arr(ic["yaw"]))
    nf = params.aero.fin_arm.shape[0]
    n_eng = params.engines.positions.shape[0]
    n_gimbal = params.engines.tvc_bias.shape[0]
    throttle0 = float(ic.get("throttle0", 0.0))
    return PlantState(
        t=_arr(0.0),
        r_ned=_arr(ic["r_ned"]),
        v_ned=_arr(ic["v_ned"]),
        q_bn=q0,
        w_body=_arr(ic.get("w_body", [0.0, 0.0, 0.0])),
        m=_arr(ic["m"]),
        tvc=_arr([[0.0, 0.0]] * n_gimbal),
        tvc_rate=_arr([[0.0, 0.0]] * n_gimbal),
        throttle_act=_arr([throttle0] * n_eng),
        fin=_arr([0.0] * nf),
        fin_rate=_arr([0.0] * nf),
        rcs_fuel=params.actuator.rcs_fuel0,
        slosh=_arr([0.0, 0.0, 0.0, 0.0]),
        wind_mean=params.wind.mean0,
        wind_gust=_arr([0.0, 0.0, 0.0]),
        wind_filter=_arr([[0.0, 0.0, 0.0]] * 3),  # (3,3) per-axis shaping-filter states, start at 0
        key=seed_words(int(ic.get("seed", 0))),
        done=np.asarray(False),
    )


def ic_from_scenario(scn, seed) -> dict:
    """Build an initial-condition dict from a scenario config, applying entry-IC dispersion
    when enabled. Euler angles stored in degrees -> radians."""
    normalized_seed = seed_value(seed)
    rng = rng_from_seed(normalized_seed)
    r = np.asarray(scn.ic.r_ned, dtype=float)
    v = np.asarray(scn.ic.v_ned, dtype=float)
    if bool(scn.dispersion.enable):
        r = r + np.asarray(scn.dispersion.pos_sigma) * rng.normal(size=3)
        v = v + np.asarray(scn.dispersion.vel_sigma) * rng.normal(size=3)
    return {
        "r_ned": r.tolist(),
        "v_ned": v.tolist(),
        "roll": float(np.deg2rad(scn.ic.roll_deg)),
        "pitch": float(np.deg2rad(scn.ic.pitch_deg)),
        "yaw": float(np.deg2rad(scn.ic.yaw_deg)),
        "w_body": list(scn.ic.w_body),
        "m": float(scn.ic.m),
        "throttle0": float(scn.ic.throttle0),
        "seed": fold_in(normalized_seed, 0x504C414E54),
    }


def termination_reason(final_state: PlantState, params: PlantParams, hit_max_steps: bool) -> int:
    """Classify why an episode ended (post-processing)."""
    import numpy as np

    h = float(-np.asarray(final_state.r_ned)[2])
    m = float(np.asarray(final_state.m))
    if h <= 1e-3:
        return int(TerminationReason.TOUCHDOWN)
    if m <= float(np.asarray(params.mass.m_dry)) + _MASS_EPS:
        return int(TerminationReason.FUEL_DEPLETED)
    if hit_max_steps:
        return int(TerminationReason.MAX_STEPS)
    return int(TerminationReason.RUNNING)

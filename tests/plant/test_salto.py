"""SALTO/T3 file-backed aero and configuration anchors."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from conftest import build_bundle, compose_cfg, nominal_for

from core.interfaces import Command
from gnc.allocation.default import _cg_x, _fin_effectiveness_matrix, allocate
from gnc.build import build_gnc
from gnc.navigation.identity import IdentityNav
from plant.dynamics import ic_from_scenario, initial_state
from sim.loop import run_closed_loop

SALTO_OVERRIDES = [
    "plant/vehicle=salto_t3",
    "plant/aero=salto_table",
    "scenario=salto_aero_to_landing",
    "gnc/scheduler=salto",
    "environment/atmosphere=ussa_poly",
]


def test_salto_npz_uses_aedb_normalization():
    path = Path(__file__).resolve().parents[2] / "data" / "aero" / "salto_t3_aero.npz"
    data = np.load(path)
    assert np.isclose(float(data["d_ref_m"]), 3.70)
    assert np.isclose(float(data["s_ref_m2"]), 10.752)


def test_salto_body_axis_source_converts_to_wind_axis():
    b = build_bundle(SALTO_OVERRIDES)
    alpha = np.deg2rad(20.0)
    atmosphere = b.native.environment_diagnostics(0.0)
    speed = 0.6 * float(atmosphere["a_sound"])
    velocity = speed * np.array([-np.cos(alpha), 0.0, np.sin(alpha)])
    result = b.native.aero_wrench_diagnostics(velocity, np.zeros(3), 36000.0, 0.0)
    cd, cl = float(result["cd"]), float(result["cl"])
    cax = cd * np.cos(alpha) - cl * np.sin(alpha)
    cz = cd * np.sin(alpha) + cl * np.cos(alpha)
    assert np.isclose(cax, 1.54434, rtol=2e-5)
    assert np.isclose(cz, 0.5 * (2.199057 - (-2.301205)), rtol=2e-5)


def test_salto_xcp_finite_at_zero_alpha():
    b = build_bundle(SALTO_OVERRIDES)
    aero = b.params.aero
    alpha_index = int(np.argmin(np.abs(np.asarray(aero.cos_grid) - 1.0)))
    mach_index = int(np.argmin(np.abs(np.asarray(aero.mach_grid) - 0.6)))
    xcp = float(np.asarray(aero.xcp_table)[alpha_index, mach_index])
    assert np.isfinite(xcp)
    assert xcp < float(aero.x_cg_ref)  # CP aft of the reference CoG (statically unstable)


def test_salto_pitch_stiffness_bounded_across_mach():
    """Equivalent pitch static margin from the C_m-reconstructed body moment stays in a physical
    corridor at every tabulated Mach near the base-first trim. Stability is evaluated from the
    reconstructed moment rather than the diagnostic center of pressure."""
    b = build_bundle(SALTO_OVERRIDES)
    p, geom = b.params, b.params.geom
    x_ref, D = float(p.aero.x_cg_ref), float(geom.d_ref)
    a_snd = float(b.native.environment_diagnostics(0.0)["a_sound"])

    def static_margin_cal(mach, alpha_deg):
        a = np.deg2rad(alpha_deg)
        V = mach * a_snd
        v_body = V * np.array([-np.cos(a), 0.0, np.sin(a)])  # pitch-plane total AoA (fwd = -x)
        result = b.native.aero_wrench_diagnostics(v_body, np.zeros(3), 36000.0, 0.0)
        return (float(result["x_cp"]) - x_ref) / D

    for mach in (0.6, 0.9, 2.0, 3.0):
        for alpha_deg in (0.5, 3.0):
            sm = static_margin_cal(mach, alpha_deg)
            assert np.isfinite(sm)
            assert -2.0 < sm < 0.3, (mach, alpha_deg, sm)  # physical corridor [cal]

    # The base-first booster remains statically unstable at every tabulated Mach.
    for mach in (0.6, 0.9, 2.0, 3.0):
        assert static_margin_cal(mach, 3.0) < 0.0, (mach, static_margin_cal(mach, 3.0))
    # Two-sided anchor for the AEDB C_m normalization, centered on the nominal -1.83 margin.
    assert -2.3 < static_margin_cal(0.9, 3.0) < -1.4, static_margin_cal(0.9, 3.0)
    # The transonic margin magnitude shrinks toward zero angle of attack.
    assert abs(static_margin_cal(0.9, 0.5)) < abs(static_margin_cal(0.9, 3.0))


def test_salto_fin_increment_zero_at_any_static_incidence():
    b = build_bundle(SALTO_OVERRIDES)
    result = b.native.fin_wrench_diagnostics(
        [-200.0, 0.0, 60.0], np.zeros(3), np.zeros(4), 36000.0, 0.0
    )
    assert np.allclose(result["force"], 0.0, atol=1e-9)
    assert np.allclose(result["moment"], 0.0, atol=1e-9)


def test_salto_pure_pitch_yaw_commands_fins():
    """In the engine-off aero phase the grid fins are the ONLY pitch/yaw effector, so a pure pitch
    or pure yaw torque must deflect the fins without a roll component masking the response."""
    b = build_bundle(SALTO_OVERRIDES)
    cfg = compose_cfg(SALTO_OVERRIDES)
    nominal = nominal_for(b, SALTO_OVERRIDES)
    state = initial_state(ic_from_scenario(cfg.scenario, 0), b.params)

    def fins_for(tau):
        cmd = Command(
            phase=np.array(1),
            mode=np.array(1),
            thrust_vector=np.zeros(3),
            q_ref=state.q_bn,
            tau_body=np.array(tau),
            active_engines=np.zeros(b.meta["n_engines"]),
            w_ref=np.zeros(3),
        )
        return np.asarray(allocate(cmd, state, nominal).fin)

    fin_pitch = fins_for([0.0, 2.0e4, 0.0])
    fin_yaw = fins_for([0.0, 0.0, 2.0e4])
    assert np.linalg.norm(fin_pitch) > 1e-3  # pure pitch deflects the fins
    assert np.linalg.norm(fin_yaw) > 1e-3  # pure yaw deflects the fins
    # pure pitch is a DIFFERENTIAL pattern, not collective (collective = roll only).
    assert not np.allclose(fin_pitch, fin_pitch[0], atol=1e-4)

    # Realized torque B@fin must lie on the commanded axis with the right sign — norm>1e-3 alone
    # passes a sign-flipped or axis-swapped pattern.
    h = -float(np.asarray(state.r_ned)[2])
    rho, a_sound = nominal["atmosphere"](h)
    v_mag = float(np.linalg.norm(np.asarray(state.v_ned)))
    q_dyn = 0.5 * rho * v_mag**2
    B = _fin_effectiveness_matrix(nominal, state, q_dyn, _cg_x(nominal, state.m), a_sound)
    tau_p, tau_y = B @ fin_pitch, B @ fin_yaw
    assert tau_p[1] > 0.0  # +pitch command -> +pitch body torque (correct sign)
    assert abs(tau_p[0]) < 1e-2 * abs(tau_p[1]) and abs(tau_p[2]) < 1e-2 * abs(tau_p[1])
    assert tau_y[2] > 0.0  # +yaw command -> +yaw body torque (correct sign)
    assert abs(tau_y[0]) < 1e-2 * abs(tau_y[2]) and abs(tau_y[1]) < 1e-2 * abs(tau_y[2])
    # pitch and yaw are DISTINCT fin patterns (rules out an axis swap reusing one for both).
    assert not np.allclose(fin_pitch, fin_yaw, atol=1e-4)


def test_salto_triad_pair_is_symmetric():
    b = build_bundle([*SALTO_OVERRIDES, "plant/thrust=vacuum"])
    eng = b.params.engines
    result = b.native.engine_wrench_diagnostics(
        [0.0, 0.7, 0.7], np.zeros((b.meta["n_gimbal"], 2)), 36000.0, 0.0
    )
    assert np.isclose(result["force"][0], 2.0 * 0.7 * float(eng.thrust_vac[0]), rtol=1e-6)
    assert np.allclose(np.asarray(result["force"])[1:], 0.0, atol=1e-6)
    assert np.allclose(result["moment"], 0.0, atol=1e-6)


def test_salto_geometry_allocation_commands_xwing_fins():
    b = build_bundle(SALTO_OVERRIDES)
    cfg = compose_cfg(SALTO_OVERRIDES)
    nominal = nominal_for(b, SALTO_OVERRIDES)
    state = initial_state(ic_from_scenario(cfg.scenario, 0), b.params)
    cmd = Command(
        phase=np.array(1),
        mode=np.array(1),
        thrust_vector=np.zeros(3),
        q_ref=state.q_bn,
        tau_body=np.array([5.0e4, 2.0e4, -1.5e4]),
        active_engines=np.zeros(b.meta["n_engines"]),
        w_ref=np.zeros(3),
    )
    act = allocate(cmd, state, nominal)
    assert act.fin.shape == (4,)
    assert np.linalg.norm(np.asarray(act.fin)) > 1e-5
    assert np.max(np.abs(np.asarray(act.fin))) <= float(b.params.actuator.fin_deflection_max_rad)


def test_salto_scenario_short_closed_loop_runs():
    cfg = compose_cfg(SALTO_OVERRIDES)
    b = build_bundle(SALTO_OVERRIDES)
    state0 = initial_state(ic_from_scenario(cfg.scenario, 0), b.params)
    controller = build_gnc(cfg, b.params)
    result = run_closed_loop(
        b,
        state0,
        controller,
        IdentityNav(),
        t_max=3.0,
        ctrl_dt=0.1,
        plant_dt=float(cfg.plant_dt),
    )
    assert np.all(np.isfinite(result.r_ned))
    assert np.all(np.isfinite(result.v_ned))
    assert result.phase[0] == 1  # AERO_DESCENT
    assert (-result.r_ned[:, 2])[-1] < (-result.r_ned[:, 2])[0]

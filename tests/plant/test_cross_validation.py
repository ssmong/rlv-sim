"""Cross-validation: independent pure-NumPy RHS vs the native C++ plant.

A second, hand-written implementation of the EoM (translational + rotational + quaternion +
mass + thrust + body aero) evaluated at random states must match the native RHS — this catches
transcription / sign / frame bugs the analytic anchors can miss. Fins/slosh/RCS are disabled
(fin_area=0) so the comparison targets the core 6-DOF + thrust + body-aero assembly.
"""

from __future__ import annotations

import numpy as np
from conftest import build_bundle

from core.interfaces import ActuatorCommand, PlantState

G0 = 9.80665
FALCON_SET = ["vehicle_set=falcon9"]


def _interp_cos_mach_np(cos_a, mach, cg, mg, table, slope):
    """Independent NumPy reimpl of C¹-in-cos (Hermite) × linear-in-Mach interpolation."""
    i = int(np.clip(np.searchsorted(cg, cos_a) - 1, 0, len(cg) - 2))
    j = int(np.clip(np.searchsorted(mg, mach) - 1, 0, len(mg) - 2))
    h = cg[i + 1] - cg[i]
    t = np.clip((cos_a - cg[i]) / h, 0, 1)
    h00 = 2 * t**3 - 3 * t**2 + 1
    h10 = t**3 - 2 * t**2 + t
    h01 = -2 * t**3 + 3 * t**2
    h11 = t**3 - t**2

    def herm(col):
        return (
            h00 * table[i, col]
            + h10 * h * slope[i, col]
            + h01 * table[i + 1, col]
            + h11 * h * slope[i + 1, col]
        )

    tm = np.clip((mach - mg[j]) / (mg[j + 1] - mg[j]), 0, 1)
    return herm(j) * (1 - tm) + herm(j + 1) * tm


def _dcm(q):
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _numpy_rhs(state: PlantState, cmd, p, meta):
    """Independent NumPy reimplementation of the core RHS (fins/slosh/rcs off)."""
    q = np.asarray(state.q_bn)
    v = np.asarray(state.v_ned)
    w = np.asarray(state.w_body)
    m = float(state.m)
    h = -float(state.r_ned[2])
    C = _dcm(q)

    # mass props (prescribed linear)
    mp = p.mass
    f = np.clip((m - float(mp.m_dry)) / (float(mp.m_wet) - float(mp.m_dry)), 0, 1)
    x_cg = float(mp.x_cg_dry) + f * (float(mp.x_cg_wet) - float(mp.x_cg_dry))
    I = np.asarray(mp.I_dry) + f * (np.asarray(mp.I_wet) - np.asarray(mp.I_dry))

    # atmosphere (exp) + ambient pressure
    rho = float(p.atmosphere.rho0) * np.exp(-max(h, 0) / float(p.atmosphere.scale_height))
    # speed of sound: ISA temperature-derived a(h) (mirrors atmosphere/exp.py), not the const field
    h_safe = max(h, 0.0)
    T = (288.15 - 0.0065 * h_safe) if h_safe <= 11000.0 else 216.65
    a = float(np.sqrt(1.4 * 287.05 * T))
    p_amb = meta["p_sea"] * np.exp(-max(h, 0) / meta["scale_height"])

    # body aero (engine-forward total alpha)
    v_rel_b = C.T @ v
    V = np.linalg.norm(v_rel_b)
    Vs = max(V, 1e-6)
    vhat = v_rel_b / Vs if V > 1e-6 else np.zeros(3)
    fwd = np.asarray(p.geom.fwd_axis)
    cos_a = np.clip(fwd @ vhat, -1.0, 1.0) if V > 1e-6 else 1.0
    lu = fwd - cos_a * vhat if V > 1e-6 else np.zeros(3)
    ln = np.linalg.norm(lu)
    lift_dir = lu / ln if ln > 1e-8 else np.zeros(3)
    mach = V / a
    cg = np.asarray(p.aero.cos_grid)
    mg = np.asarray(p.aero.mach_grid)
    cd = _interp_cos_mach_np(
        cos_a, mach, cg, mg, np.asarray(p.aero.cd_table), np.asarray(p.aero.cd_slope)
    )
    cl = _interp_cos_mach_np(
        cos_a, mach, cg, mg, np.asarray(p.aero.cl_table), np.asarray(p.aero.cl_slope)
    )
    xcp = _interp_cos_mach_np(
        cos_a, mach, cg, mg, np.asarray(p.aero.xcp_table), np.asarray(p.aero.xcp_slope)
    )
    qd = 0.5 * rho * V**2
    S = float(p.geom.s_ref)
    F_aero = -qd * S * cd * vhat + qd * S * cl * lift_dir
    M_aero = np.cross([xcp - x_cg, 0, 0], F_aero)
    # body pitch/yaw aero damping Cmq (mirrors aero_table._CMQ): M_damp = qd·S·L·Cmq·(L/2V)·w_⊥
    if V > 1e-6:
        Lref, cmq = float(p.geom.length), -0.5
        damp = qd * S * Lref * cmq * (Lref / (2.0 * V))
        M_aero = M_aero + damp * np.array([0.0, w[1], w[2]])

    # thrust: independent per-engine sum (this test runs N=1; loops over the engine axis)
    eng = p.engines
    n_eng = np.asarray(eng.positions).shape[0]
    T_body = np.zeros(3)
    M_thrust = np.zeros(3)
    mdot = 0.0
    for i in range(n_eng):
        j = i  # N=1, Ng=1: gimballed engine 0 uses tvc row 0
        ang = np.asarray(state.tvc)[j] + np.asarray(eng.tvc_bias)[j] + np.asarray(eng.cant)[i]
        dirn = np.array(
            [np.cos(ang[0]) * np.cos(ang[1]), np.cos(ang[0]) * np.sin(ang[1]), -np.sin(ang[0])]
        )
        t_frac = np.clip(float(np.asarray(state.throttle_act)[i]), 0, 1)
        on, scale = float(eng.engine_on[i]), float(eng.thrust_scale[i])
        # flameout/floor gate phi (mirrors actuator_model._firing_fraction): smoothstep over
        # [Tmin-0.1, Tmin], gating BOTH thrust and mdot to 0 below the deep-throttle floor.
        t_floor = float(eng.throttle_min[i])
        xphi = np.clip((t_frac - (t_floor - 0.1)) / 0.1, 0.0, 1.0)
        phi = xphi * xphi * (3.0 - 2.0 * xphi)
        t_mag = (
            scale
            * on
            * phi
            * max(t_frac * float(eng.thrust_vac[i]) - p_amb * float(eng.nozzle_exit_area[i]), 0)
        )
        T_i = t_mag * dirn
        r_i = np.asarray(eng.positions)[i] - np.array([x_cg, 0, 0])
        T_body = T_body + T_i
        M_thrust = M_thrust + np.cross(r_i, T_i)
        mdot += on * phi * t_frac * float(eng.thrust_vac[i]) / (float(eng.isp[i]) * G0)

    g_ned = np.asarray(p.g_ned)
    a_ned = g_ned + C @ (T_body + F_aero) / m
    # jet damping: -ṁ·l_n²·ω_⊥ about pitch/yaw, l_n = CG-to-nozzle distance
    l_n = float(p.geom.x_eng) - x_cg
    M_jet = -mdot * l_n**2 * np.array([0.0, w[1], w[2]])
    # changing-inertia İ·ω term (prescribed linear I => dI/dm constant); I_dot = dI/dm * (-mdot)
    dI_dm = (np.asarray(mp.I_wet) - np.asarray(mp.I_dry)) / (float(mp.m_wet) - float(mp.m_dry))
    I_dot = dI_dm * (-mdot)
    wdot = (M_thrust + M_aero + M_jet - np.cross(w, I * w) - I_dot * w) / I
    qdot = 0.5 * _quat_mul_np(q, np.array([0.0, *w]))
    dm = -mdot
    return v, a_ned, qdot, wdot, dm


def _quat_mul_np(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def _random_state(rng, p):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    m = rng.uniform(float(p.mass.m_dry) + 1000, float(p.mass.m_wet))
    return PlantState(
        t=np.array(rng.uniform(0, 50)),
        r_ned=np.array([rng.uniform(-500, 500), rng.uniform(-500, 500), rng.uniform(-9000, -2000)]),
        v_ned=np.array(rng.uniform(-150, 150, size=3)),
        q_bn=np.array(q),
        w_body=np.array(rng.uniform(-0.3, 0.3, size=3)),
        m=np.array(m),
        tvc=np.array(rng.uniform(-0.1, 0.1, size=(1, 2))),
        tvc_rate=np.array(rng.uniform(-0.1, 0.1, size=(1, 2))),
        throttle_act=np.array([rng.uniform(0, 1)]),
        fin=np.zeros(4),
        fin_rate=np.zeros(4),
        rcs_fuel=np.array(100.0),
        slosh=np.zeros(4),
        wind_mean=np.zeros(3),
        wind_gust=np.zeros(3),
        wind_filter=np.zeros((3, 3)),
        key=0,
        done=np.array(False),
    )


def test_rhs_matches_numpy_reference():
    # Pin linear prescribed mass properties for this NumPy reference; nonlinear mass-property
    # fidelity is covered separately in test_massprops.
    bundle = build_bundle(
        [
            *FALCON_SET,
            "plant.aero.fins.fin_area=0.0",
            "plant.vehicle.massprops_model=prescribed",
        ]
    )
    p, meta = bundle.params, bundle.meta
    rng = np.random.default_rng(42)
    for _ in range(25):
        s = _random_state(rng, p)
        cmd = ActuatorCommand(
            throttle=np.array(rng.uniform(0, 1)),
            gimbal=np.array(rng.uniform(-0.1, 0.1, size=2)),
            fin=np.zeros(4),
            rcs=np.zeros(3),
        )
        native_cmd = ActuatorCommand(
            throttle=np.atleast_1d(cmd.throttle),
            gimbal=np.asarray(cmd.gimbal).reshape(1, 2),
            fin=cmd.fin,
            rcs=cmd.rcs,
        )
        deriv, _ = bundle.native.rhs(s, native_cmd, np.zeros(3))
        vdot_ref, adot_ref, qdot_ref, wdot_ref, dm_ref = _numpy_rhs(s, cmd, p, meta)
        assert np.allclose(np.asarray(deriv[1]), vdot_ref, atol=1e-9)  # ṙ
        assert np.allclose(np.asarray(deriv[2]), adot_ref, atol=1e-7, rtol=1e-9)  # v̇
        assert np.allclose(np.asarray(deriv[3]), qdot_ref, atol=1e-9)  # q̇
        assert np.allclose(np.asarray(deriv[4]), wdot_ref, atol=1e-7, rtol=1e-9)  # ω̇
        assert np.isclose(float(deriv[5]), dm_ref, atol=1e-9)  # ṁ

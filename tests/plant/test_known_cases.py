"""Analytical known-case anchors (float64).

Vacuum free-fall (parabola), drag terminal velocity (NOT parabola), hover (mass frozen),
angular-momentum conservation, energy balance, quaternion unit norm.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from conftest import build_bundle

from core.frames import dcm_from_quat
from core.interfaces import ActuatorCommand
from plant.dynamics import initial_state

G0 = 9.80665
FALCON_SET = ["vehicle_set=falcon9"]


def zero_cmd(nf=4):
    return ActuatorCommand(
        throttle=np.zeros(1), gimbal=np.zeros((1, 2)), fin=np.zeros(nf), rcs=np.zeros(3)
    )


def _rollout(bundle, state, command, steps):
    session = bundle.native.create_session(state)
    snapshots, diagnostics = [], []
    for _ in range(steps):
        snapshot, aux = session.advance(command, 1)
        snapshots.append(snapshot)
        diagnostics.append(aux)
    trajectory = SimpleNamespace(
        **{name: np.stack([snapshot[name] for snapshot in snapshots]) for name in snapshots[0]}
    )
    aux_trajectory = {
        name: np.stack([np.asarray(aux[name]) for aux in diagnostics]) for name in diagnostics[0]
    }
    return SimpleNamespace(**snapshots[-1]), trajectory, aux_trajectory


def base_ic(**over):
    ic = dict(
        r_ned=[0.0, 0.0, -8000.0],
        v_ned=[0.0, 0.0, 0.0],
        roll=0.0,
        pitch=np.pi / 2,
        yaw=0.0,
        w_body=[0.0, 0.0, 0.0],
        m=50000.0,
        throttle0=0.0,
        seed=0,
    )
    ic.update(over)
    return ic


# --------------------------------------------------------------------------- #
def test_vacuum_free_fall_parabola():
    bundle = build_bundle([*FALCON_SET, "environment.atmosphere.rho0=0.0"])
    s0 = initial_state(base_ic(), bundle.params)
    n, dt = 1000, float(bundle.params.plant_dt)  # 1 s
    final, traj, _ = _rollout(bundle, s0, zero_cmd(), n)
    t = n * dt
    z_analytic = -8000.0 + 0.5 * G0 * t**2
    vz_analytic = G0 * t
    assert np.isclose(float(final.r_ned[2]), z_analytic, atol=1e-6)
    assert np.isclose(float(final.v_ned[2]), vz_analytic, atol=1e-6)
    assert np.allclose(np.asarray(final.r_ned[:2]), 0.0, atol=1e-9)  # no horizontal drift
    assert np.allclose(np.asarray(final.w_body), 0.0, atol=1e-12)  # no torque
    assert np.allclose(np.asarray(traj.q_bn[-1]), np.asarray(s0.q_bn), atol=1e-12)


def test_energy_conservation_free_fall():
    bundle = build_bundle([*FALCON_SET, "environment.atmosphere.rho0=0.0"])
    s0 = initial_state(base_ic(), bundle.params)
    n = 2000
    _, traj, _ = _rollout(bundle, s0, zero_cmd(), n)
    m = np.asarray(traj.m)
    v = np.asarray(traj.v_ned)
    h = -np.asarray(traj.r_ned)[:, 2]
    E = 0.5 * m * np.sum(v**2, axis=1) + m * G0 * h
    assert np.allclose(E, E[0], rtol=1e-9)


def test_drag_terminal_velocity_equilibrium():
    """Constant rho, constant C_D => analytic terminal velocity is a stable fixed point."""
    bundle = build_bundle(
        [
            *FALCON_SET,
            "environment.atmosphere.scale_height=1e12",  # rho ~ constant = rho0
            "plant.aero.coeffs.cd_transonic=0.0",
            "plant.aero.coeffs.cd_hyper=0.0",  # constant Cd=1.0 (disable Mach-varying terms)
            "plant.aero.coeffs.cd_alpha=0.0",
            "plant.aero.coeffs.cn_alpha=0.0",
            "plant.aero.coeffs.cd_base=1.0",
            "plant.aero.fins.fin_area=0.0",  # isolate BODY drag (fin drag tested separately)
        ]
    )
    p = bundle.params
    rho = float(p.atmosphere.rho0)
    S = float(p.geom.s_ref)
    m = 50000.0
    V_term = np.sqrt(2 * m * G0 / (rho * S * 1.0))

    # start exactly at terminal velocity -> stays
    s0 = initial_state(base_ic(v_ned=[0.0, 0.0, V_term]), p)
    final, _, _ = _rollout(bundle, s0, zero_cmd(), 1000)
    assert np.isclose(float(final.v_ned[2]), V_term, atol=1e-2)

    # below terminal -> accelerates toward it; above -> decelerates toward it
    s_lo = initial_state(base_ic(v_ned=[0.0, 0.0, 0.5 * V_term]), p)
    f_lo, _, _ = _rollout(bundle, s_lo, zero_cmd(), 1000)
    assert 0.5 * V_term < float(f_lo.v_ned[2]) < V_term

    s_hi = initial_state(base_ic(v_ned=[0.0, 0.0, 1.5 * V_term]), p)
    f_hi, _, _ = _rollout(bundle, s_hi, zero_cmd(), 1000)
    assert V_term < float(f_hi.v_ned[2]) < 1.5 * V_term


def test_hover_thrust_equals_weight():
    """T = mg with frozen mass (Isp->inf) => vehicle holds position."""
    bundle = build_bundle(
        [
            *FALCON_SET,
            "environment.atmosphere.rho0=0.0",
            "environment.atmosphere.p_sea=0.0",
            "plant.vehicle.engines.isp=1e12",
        ]
    )
    p = bundle.params
    m = 50000.0
    Tvac = float(p.engines.thrust_vac[0])
    throttle = m * G0 / Tvac
    s0 = initial_state(base_ic(throttle0=throttle), p)
    cmd = ActuatorCommand(
        throttle=np.array([throttle]),
        gimbal=np.zeros((1, 2)),
        fin=np.zeros(4),
        rcs=np.zeros(3),
    )
    n = 2000
    final, traj, _ = _rollout(bundle, s0, cmd, n)
    assert np.isclose(float(final.r_ned[2]), -8000.0, atol=1e-2)  # altitude held
    assert np.allclose(np.asarray(final.v_ned), 0.0, atol=1e-2)
    assert np.isclose(float(final.m), m, rtol=1e-6)  # mass frozen


def test_angular_momentum_conservation():
    """No torque (no aero/thrust) => inertial angular momentum & rotational KE conserved."""
    bundle = build_bundle([*FALCON_SET, "environment.atmosphere.rho0=0.0"])
    p = bundle.params
    s0 = initial_state(base_ic(w_body=[0.1, 0.2, -0.15]), p)
    n = 5000
    _, traj, _ = _rollout(bundle, s0, zero_cmd(), n)

    I = np.asarray(bundle.native.mass_properties_diagnostics(float(p.mass.m_wet))["inertia"])

    def H_ned(q, w):
        C = np.asarray(dcm_from_quat(q))
        return C @ (I * np.asarray(w))

    H0 = H_ned(s0.q_bn, s0.w_body)
    KE0 = 0.5 * np.dot(np.asarray(s0.w_body), I * np.asarray(s0.w_body))
    for k in (0, n // 2, n - 1):
        H = H_ned(traj.q_bn[k], traj.w_body[k])
        KE = 0.5 * np.dot(np.asarray(traj.w_body[k]), I * np.asarray(traj.w_body[k]))
        assert np.allclose(H, H0, atol=1e-6)
        assert np.isclose(KE, KE0, rtol=1e-8)
    # precession actually occurs (not a trivially static state)
    assert not np.allclose(np.asarray(traj.w_body[-1]), np.asarray(s0.w_body), atol=1e-4)


def test_min_throttle_floor_enforced():
    """Plant enforces the deep-throttle floor: commanding < Tmin => engine off (flameout)."""
    bundle = build_bundle(
        [*FALCON_SET, "environment.atmosphere.rho0=0.0", "environment.atmosphere.p_sea=0.0"]
    )
    p = bundle.params
    tmin = float(p.engines.throttle_min[0])
    n = 3000

    # command below the floor (0.3 < 0.4), starting lit at the floor -> target is 0 (flameout),
    # so throttle decays toward 0 (NOT toward the commanded 0.3) and thrust collapses.
    s_lo = initial_state(base_ic(throttle0=tmin), p)
    cmd_lo = ActuatorCommand(
        throttle=np.array([tmin - 0.1]),
        gimbal=np.zeros((1, 2)),
        fin=np.zeros(4),
        rcs=np.zeros(3),
    )
    final_lo, _, aux_lo = _rollout(bundle, s_lo, cmd_lo, n)
    assert float(final_lo.throttle_act[0]) < 0.05  # heading to 0, far below the commanded 0.3
    assert float(aux_lo["thrust_mag"][-1]) < 0.05 * float(p.engines.thrust_vac[0])

    # command above the floor -> engine holds and produces thrust
    s_hi = initial_state(base_ic(throttle0=0.6), p)
    cmd_hi = ActuatorCommand(
        throttle=np.array([0.6]),
        gimbal=np.zeros((1, 2)),
        fin=np.zeros(4),
        rcs=np.zeros(3),
    )
    _, _, aux_hi = _rollout(bundle, s_hi, cmd_hi, n)
    assert float(aux_hi["thrust_mag"][-1]) > 0.5 * 0.6 * float(p.engines.thrust_vac[0])


def test_quaternion_unit_norm():
    bundle = build_bundle([*FALCON_SET, "environment.atmosphere.rho0=0.0"])
    s0 = initial_state(base_ic(w_body=[0.3, -0.4, 0.5]), bundle.params)
    _, traj, _ = _rollout(bundle, s0, zero_cmd(), 3000)
    norms = np.linalg.norm(np.asarray(traj.q_bn), axis=1)
    assert np.allclose(norms, 1.0, atol=1e-10)

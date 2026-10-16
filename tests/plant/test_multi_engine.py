"""Multi-engine thrust-set, symmetry, fault, and native-integration tests."""

from __future__ import annotations

import numpy as np
from conftest import build_bundle


def _merged_components(n, r):
    """Merged components of union(k*r, k), excluding the isolated coast point."""
    bands = [(k * r, float(k)) for k in range(1, n + 1)]
    merged: list[list[float]] = []
    for lo, hi in bands:
        if merged and lo <= merged[-1][1] + 1e-12:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return merged


def test_thrust_band_math_and_gap_condition():
    c = _merged_components(9, 0.4)
    assert len(c) == 1 and np.isclose(c[0][0], 0.4) and np.isclose(c[0][1], 9.0)
    c = _merged_components(9, 0.6)
    assert len(c) == 2 and np.isclose(c[0][1], 1.0) and np.isclose(c[1][0], 1.2)
    assert len(_merged_components(3, 0.7)) == 3
    for n, r in [(9, 0.4), (9, 0.6), (3, 0.7)]:
        for k in range(1, n):
            assert ((k + 1) * r > k + 1e-12) == (r > k / (k + 1) + 1e-12)
    assert _merged_components(9, 0.4)[0][0] > 0.0


def _octaweb(overrides=()):
    bundle = build_bundle(["plant/vehicle=falcon9_octaweb", "plant/thrust=vacuum", *overrides])
    return bundle, bundle.params.engines, bundle.meta["n_engines"], bundle.meta["n_gimbal"]


def _wrench(bundle, throttle, tvc, mass=50000.0):
    return bundle.native.engine_wrench_diagnostics(throttle, tvc, mass, 0.0)


def test_engine_wrenches_minkowski_axial():
    """k engines at a common throttle produce k times the single-engine axial thrust."""
    bundle, eng, n, ng = _octaweb()
    t1 = float(eng.thrust_vac[0])
    tvc = np.zeros((ng, 2))
    for k in [1, 3, 9]:
        result = _wrench(bundle, [0.7] * k + [0.0] * (n - k), tvc)
        force = np.asarray(result["force"])
        assert np.isclose(force[0], k * 0.7 * t1, rtol=1e-6)
        assert np.allclose(force[1:], 0.0, atol=1e-6)
        assert np.isclose(
            float(result["mdot"]), k * 0.7 * t1 / (float(eng.isp[0]) * 9.80665), rtol=1e-6
        )


def test_all_engines_symmetric_no_moment():
    bundle, eng, n, ng = _octaweb()
    result = _wrench(bundle, np.ones(n), np.zeros((ng, 2)))
    assert np.isclose(result["force"][0], n * float(eng.thrust_vac[0]), rtol=1e-6)
    assert np.allclose(result["moment"], 0.0, atol=1e-3)


def test_engine_out_pitch_yaw_moment_no_roll():
    """A failed ring engine shifts the axial thrust line but creates no roll moment."""
    failed = 3
    bundle, eng, n, ng = _octaweb(("fault=engine_out", f"fault.engine_out_idx={failed}"))
    t1 = float(eng.thrust_vac[0])
    position = np.asarray(eng.positions)[failed]
    result = _wrench(bundle, np.ones(n), np.zeros((ng, 2)))
    force, moment = np.asarray(result["force"]), np.asarray(result["moment"])
    assert np.isclose(force[0], (n - 1) * t1, rtol=1e-6)
    assert np.allclose(force[1:], 0.0, atol=1e-3)
    assert np.isclose(moment[0], 0.0, atol=1e-3)
    assert np.isclose(moment[1], -t1 * position[2], rtol=1e-4)
    assert np.isclose(moment[2], t1 * position[1], rtol=1e-4)


def test_center_engine_out_no_moment():
    bundle, eng, n, ng = _octaweb(("fault=engine_out", "fault.engine_out_idx=0"))
    result = _wrench(bundle, np.ones(n), np.zeros((ng, 2)))
    assert np.isclose(result["force"][0], (n - 1) * float(eng.thrust_vac[0]), rtol=1e-6)
    assert np.allclose(result["moment"], 0.0, atol=1e-3)


def test_per_engine_throttle_floor_independence():
    """One below-floor engine flames out without changing its siblings."""
    from core.interfaces import ActuatorCommand
    from plant.dynamics import initial_state

    bundle = build_bundle(["plant/vehicle=falcon9_octaweb", "environment.atmosphere.rho0=0.0"])
    n = bundle.meta["n_engines"]
    state = initial_state(
        {
            "r_ned": [0, 0, -5000.0],
            "v_ned": [0, 0, 50.0],
            "roll": 0.0,
            "pitch": np.pi / 2,
            "yaw": 0.0,
            "w_body": [0, 0, 0],
            "m": 50000.0,
            "throttle0": 0.6,
            "seed": 0,
        },
        bundle.params,
    )
    command = ActuatorCommand(
        throttle=np.array([0.2] + [0.6] * (n - 1)),
        gimbal=np.zeros((bundle.meta["n_gimbal"], 2)),
        fin=np.zeros(4),
        rcs=np.zeros(3),
    )
    session = bundle.native.create_session(state)
    final, _ = session.advance(command, 4500)
    throttle = np.asarray(final["throttle_act"])
    assert throttle[0] < 0.05
    assert np.all(throttle[1:] > 0.5)


def test_gimballed_center_moment():
    bundle, eng, n, _ng = _octaweb()
    t1 = float(eng.thrust_vac[0])
    delta = np.deg2rad(5.0)
    x_cg = float(bundle.native.mass_properties_diagnostics(50000.0)["x_cg"])
    result = _wrench(bundle, np.ones(n), [[delta, 0.0]])
    force, moment = np.asarray(result["force"]), np.asarray(result["moment"])
    arm_x = float(eng.positions[0, 0]) - x_cg
    thrust_z = -t1 * np.sin(delta)
    assert np.isclose(moment[1], -arm_x * thrust_z, rtol=1e-4)
    assert np.isclose(moment[0], 0.0, atol=1e-3)
    assert np.isclose(moment[2], 0.0, atol=1e-3)
    assert np.isclose(force[2], thrust_z, rtol=1e-4)


def test_octaweb_native_step_is_finite():
    from core.interfaces import ActuatorCommand
    from plant.dynamics import initial_state

    bundle = build_bundle(["plant/vehicle=falcon9_octaweb"])
    state = initial_state(
        {
            "r_ned": [0, 0, -3000.0],
            "v_ned": [10, 0, 60.0],
            "roll": 0.0,
            "pitch": np.pi / 2 - 0.1,
            "yaw": 0.0,
            "w_body": [0.01, 0, 0],
            "m": 45000.0,
            "throttle0": 0.5,
            "seed": 0,
        },
        bundle.params,
    )
    command = ActuatorCommand(
        throttle=np.full(bundle.meta["n_engines"], 0.6),
        gimbal=np.zeros((bundle.meta["n_gimbal"], 2)),
        fin=np.zeros(4),
        rcs=np.zeros(3),
    )
    next_state, aux = bundle.native.step(state, command, np.zeros(3))
    assert np.all(np.isfinite(next_state["v_ned"]))
    assert np.all(np.isfinite(next_state["w_body"]))
    assert np.all(np.isfinite(aux["M_body_total"]))

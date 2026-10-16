"""Grid-fin component anchors against the native implementation."""

from __future__ import annotations

import numpy as np
from conftest import build_bundle

FALCON_SET = ["vehicle_set=falcon9"]


def _setup():
    bundle = build_bundle(FALCON_SET)
    sound = float(bundle.native.environment_diagnostics(0.0)["a_sound"])
    return bundle, sound


def _wrench(bundle, velocity, fin, rates=None, mass=None):
    return bundle.native.fin_wrench_diagnostics(
        np.asarray(velocity, dtype=float),
        np.zeros(3) if rates is None else np.asarray(rates, dtype=float),
        np.asarray(fin, dtype=float),
        float(bundle.params.mass.m_wet if mass is None else mass),
        0.0,
    )


def test_fin_pitch_authority_and_sign():
    bundle, _ = _setup()
    result = _wrench(bundle, [-200.0, 0.0, 0.0], [0.1, 0.0, -0.1, 0.0])
    moment = np.asarray(result["moment"])
    assert abs(moment[1]) > 1e-3
    assert abs(moment[0]) < abs(moment[1])


def test_fin_roll_authority_from_collective():
    bundle, _ = _setup()
    baseline = _wrench(bundle, [-200.0, 0.0, 0.0], np.zeros(4))
    collective = _wrench(bundle, [-200.0, 0.0, 0.0], np.full(4, 0.1))
    assert abs(float(collective["moment"][0])) > 1e-2
    assert abs(float(collective["moment"][0])) > abs(float(baseline["moment"][0]))


def test_fin_rate_damping_opposes_body_rate():
    bundle, _ = _setup()
    for roll_rate in (0.3, -0.3):
        result = _wrench(bundle, [-200.0, 0.0, 0.0], np.zeros(4), rates=[roll_rate, 0.0, 0.0])
        assert float(result["moment"][0]) * roll_rate < 0.0


def test_fin_stall_saturates_lift():
    bundle, _ = _setup()
    stall = float(bundle.params.aero.fin_stall_alpha)
    big = 1.5 * stall
    first = _wrench(bundle, [-200.0, 0.0, 0.0], [big, 0.0, 0.0, 0.0])
    second = _wrench(bundle, [-200.0, 0.0, 0.0], [2.0 * big, 0.0, 0.0, 0.0])
    ratio = abs(float(second["moment"][1])) / max(abs(float(first["moment"][1])), 1e-9)
    assert 0.9 < ratio < 1.6


def test_fin_is_finite_at_zero_airspeed():
    bundle, _ = _setup()
    for rates in ([0.1, 0.0, 0.0], [0.0, 0.0, 0.0]):
        result = _wrench(bundle, np.zeros(3), [0.05, 0.0, -0.05, 0.0], rates=rates)
        assert np.all(np.isfinite(result["force"]))
        assert np.all(np.isfinite(result["moment"]))


def test_fin_local_flow_changes_under_body_rate():
    bundle, sound = _setup()
    velocity = [-0.83 * sound, 0.0, 0.0]
    fin = [0.1, 0.0, -0.1, 0.0]
    static = _wrench(bundle, velocity, fin)
    rotating = _wrench(bundle, velocity, fin, rates=[0.0, 0.0, 1.5])
    assert not np.allclose(rotating["force"], static["force"])


def test_fin_transonic_bucket_chokes_lift():
    bundle, sound = _setup()
    fin = [0.1, 0.0, 0.0, 0.0]

    def lift_z(mach):
        result = _wrench(bundle, [-mach * sound, 0.0, 0.0], fin)
        return float(result["force"][2]) / (mach * sound) ** 2

    low = lift_z(0.3)
    assert lift_z(0.55) <= low * 1.02
    assert 0.45 * low < lift_z(0.9) < 0.75 * low
    assert lift_z(1.2) < 0.75 * low
    assert lift_z(1.6) > lift_z(1.2)
    assert lift_z(2.2) < lift_z(1.6)
    assert 0.25 * low < lift_z(5.7) < 0.5 * low

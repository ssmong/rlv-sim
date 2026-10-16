"""Configured success limits must remain stricter than the generic soft-contact class."""

from types import SimpleNamespace

import numpy as np

from gnc.guidance.base import thrust_aligned_quat, upright_quat
from sim.loop import _finalize


def _result(v_ned, q_bn, *, speed_tol=5.0, tilt_tol_deg=15.0):
    log = {
        "t": [0.1],
        "r_ned": [np.zeros(3)],
        "v_ned": [np.asarray(v_ned, float)],
        "q_bn": [np.asarray(q_bn, float)],
        "w_body": [np.zeros(3)],
        "m": [110.0],
        "phase": [3],
        "throttle_act": [np.zeros(1)],
        "tvc": [np.zeros((1, 2))],
        "fin": [np.zeros(4)],
    }
    state = SimpleNamespace(r_ned=np.zeros(3), m=110.0)
    bundle = SimpleNamespace(params=SimpleNamespace(mass=SimpleNamespace(m_dry=100.0)))
    return _finalize(log, [{}], state, bundle, False, 10.0, speed_tol, tilt_tol_deg)


def test_soft_contact_still_honors_configured_speed_limit():
    result = _result([0.8, 0.0, 0.8], upright_quat(), speed_tol=1.0)

    assert result.landing_class == "soft"
    assert result.violation_flags["speed"]
    assert not result.success


def test_soft_contact_still_honors_configured_tilt_limit():
    tilt = np.deg2rad(3.0)
    q_bn = thrust_aligned_quat([np.sin(tilt), 0.0, -np.cos(tilt)], np.deg2rad(10.0))
    result = _result([0.0, 0.0, 0.5], q_bn, tilt_tol_deg=2.0)

    assert result.landing_class == "soft"
    assert result.violation_flags["tilt"]
    assert not result.success


def test_soft_contact_within_all_configured_limits_succeeds():
    result = _result([0.0, 0.0, 0.5], upright_quat())

    assert result.landing_class == "soft"
    assert not any(result.violation_flags.values())
    assert result.success

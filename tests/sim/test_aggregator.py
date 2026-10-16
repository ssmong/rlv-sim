"""KPI aggregator unit tests with explicit success thresholds."""

from __future__ import annotations

import types

import numpy as np

from core.interfaces import TerminationReason
from sim.aggregate import aggregate_episodes

_TD = int(TerminationReason.TOUCHDOWN)
_FUEL = int(TerminationReason.FUEL_DEPLETED)


def _ep(pos, spd, tilt, fuel, success, reason=_TD):
    return types.SimpleNamespace(
        landing_position_error=pos,
        landing_speed=spd,
        landing_speed_vertical=spd,
        landing_speed_horizontal=0.0,
        landing_rate=0.0,
        landing_tilt=tilt,
        fuel_margin=fuel,
        success=success,
        termination_reason=reason,
    )


def test_aggregate_success_rate_and_stats():
    """Landing-accuracy KPIs are pooled over LANDED episodes only; the fuel-depleted run
    (which never reached the ground) is excluded from position/speed/tilt but still counts toward
    success_rate and touchdown_rate."""
    eps = [
        _ep(2.0, 1.0, 0.05, 500.0, True),
        _ep(4.0, 3.0, 0.10, 300.0, True),
        _ep(50.0, 20.0, 0.5, -10.0, False, reason=_FUEL),  # mid-air "landing" => excluded
        _ep(6.0, 2.0, 0.08, 100.0, True),
    ]
    s = aggregate_episodes(eps)
    assert s["n_episodes"] == 4
    assert s["n_landed"] == 3
    assert np.isclose(s["touchdown_rate"], 0.75)
    assert np.isclose(s["success_rate"], 0.75)
    # landing-accuracy stats over the 3 landed episodes only (the pos=50 mid-air run is dropped)
    assert np.isclose(s["landing_position_error"]["mean"], np.mean([2, 4, 6]))
    assert np.isclose(s["landing_speed"]["max"], 3.0)
    assert np.isclose(s["landing_tilt_deg"]["max"], np.rad2deg(0.10))
    # fuel margin is defined at termination for ALL episodes
    assert np.isclose(s["fuel_margin"]["min"], -10.0)


def test_aggregate_no_landed_is_nan():
    """If no episode reaches the ground, landing-accuracy KPIs are NaN (not a misleading number),
    but success/touchdown rates are still well-defined (0)."""
    eps = [
        _ep(40.0, 18.0, 0.4, -5.0, False, reason=_FUEL),
        _ep(60.0, 25.0, 0.6, 1.0, False, reason=int(TerminationReason.MAX_STEPS)),
    ]
    s = aggregate_episodes(eps)
    assert s["n_landed"] == 0
    assert np.isclose(s["touchdown_rate"], 0.0)
    assert np.isnan(s["landing_position_error"]["mean"])
    assert np.isnan(s["landing_speed"]["p95"])
    assert np.isfinite(s["fuel_margin"]["min"])  # fuel margin still defined


def test_aggregate_empty_raises():
    import pytest

    with pytest.raises(ValueError):
        aggregate_episodes([])

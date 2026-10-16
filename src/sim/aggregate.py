"""KPI aggregation over episodes / batches."""

from __future__ import annotations

import numpy as np

from core.interfaces import EpisodeResult, TerminationReason


def _reason(r) -> int:
    """Return the episode termination tag."""
    return int(r.termination_reason)


def _reached_ground(r) -> bool:
    """True iff the episode reached the ground — a SURVIVED touchdown OR a CRASH. Both have a valid
    touchdown state (so their landing KPIs are real); a fuel-depleted/horizon-truncated run never
    reaches the ground and its touchdown index is mid-air, so it is excluded."""
    return _reason(r) in (int(TerminationReason.TOUCHDOWN), int(TerminationReason.CRASH))


def aggregate_episodes(results: list[EpisodeResult]) -> dict:
    """Aggregate closed-loop EpisodeResults into summary KPIs with success thresholds.

    Landing-accuracy KPIs are averaged over episodes that REACHED THE GROUND (touchdown OR crash) —
    both have a real touchdown state; mid-air terminations (fuel-depleted/horizon) are excluded so
    their last-step "landing" values don't poison the mean. Crucially, non-converging
    or crashing controllers are NOT hidden: ``touchdown_rate`` (survived), ``crash_rate``, and the
    airborne remainder are all reported over ALL episodes, and the sink/lateral/spin
    KPIs are split out. ``success_rate`` and ``fuel_margin`` are over ALL episodes.
    """
    n = len(results)
    if n == 0:
        raise ValueError("no episodes to aggregate")
    grounded = [r for r in results if _reached_ground(r)]
    n_touch = sum(1 for r in results if _reason(r) == int(TerminationReason.TOUCHDOWN))
    n_crash = sum(1 for r in results if _reason(r) == int(TerminationReason.CRASH))
    success = np.array([bool(r.success) for r in results])
    fuel = np.array([r.fuel_margin for r in results])

    def _g(attr):
        vals = [float(getattr(r, attr)) for r in grounded]
        return _stats(np.array(vals)) if grounded else _nan_stats()

    if grounded:
        pos_s = _stats(np.array([r.landing_position_error for r in grounded]))
        spd_s = _stats(np.array([r.landing_speed for r in grounded]))
        tilt_s = _stats(np.rad2deg(np.array([r.landing_tilt for r in grounded])))
    else:
        pos_s = spd_s = tilt_s = _nan_stats()
    return {
        "n_episodes": n,
        "n_landed": n_touch,  # SURVIVED touchdowns (crashes excluded)
        "n_crashed": n_crash,
        "n_reached_ground": len(grounded),
        "touchdown_rate": float(n_touch / n),  # survived-landing fraction
        "crash_rate": float(n_crash / n),  # reached ground but exceeded a survival limit
        "airborne_rate": float((n - len(grounded)) / n),  # never reached ground (fuel/horizon)
        "success_rate": float(success.mean()),
        # landing-accuracy KPIs over GROUNDED (touchdown or crash) episodes; NaN if none grounded
        "landing_position_error": pos_s,
        "landing_speed": spd_s,
        "landing_speed_vertical": _g("landing_speed_vertical"),  # sink rate
        "landing_speed_horizontal": _g("landing_speed_horizontal"),  # lateral
        "landing_tilt_deg": tilt_s,
        "landing_rate": _g("landing_rate"),  # spin at contact
        "fuel_margin": _stats(fuel),
    }


def _stats(x: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "p50": float(np.percentile(x, 50)),
        "p95": float(np.percentile(x, 95)),
    }


def _nan_stats() -> dict:
    """All-NaN stats block (no landed episodes => landing-accuracy KPIs are undefined)."""
    return {k: float("nan") for k in ("mean", "std", "min", "max", "p50", "p95")}

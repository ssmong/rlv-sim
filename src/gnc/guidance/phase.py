"""Latching phase schedulers for the GNC stack."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

# A guard is a predicate (x_hat, t) -> bool that fires the k-th transition (latching).
Guard = Callable[[object, float], bool]


class SequentialPhaseScheduler:
    """Ordered phases advanced by latching guards."""

    def __init__(self, phases: list[int], guards: list[Guard], start_idx: int = 0):
        if len(guards) != len(phases) - 1:
            raise ValueError("need exactly one guard per phase transition")
        if not 0 <= start_idx < len(phases):
            raise ValueError(f"start_idx {start_idx} out of range for {len(phases)} phases")
        self.phases = list(phases)
        self.guards = list(guards)
        self.start_idx = start_idx

    def initial_memory(self) -> dict:
        return {"phase_idx": self.start_idx}

    def phase(self, x_hat, t, memory: dict) -> tuple[int, dict]:
        k = int(memory["phase_idx"])
        # advance through any guards that have fired (allows multi-phase skip in one tick)
        while k < len(self.guards) and bool(self.guards[k](x_hat, t)):
            k += 1
        return self.phases[k], {"phase_idx": k}


# --------------------------------------------------------------------------- #
# Guard factories
# --------------------------------------------------------------------------- #
def altitude_below(h_switch: float) -> Guard:
    """Fire when altitude (= -r_ned[2]) drops below ``h_switch`` (monotone => well-posed)."""
    return lambda x, t: float(-np.asarray(x.r_ned)[2]) < h_switch


def speed_below(v_switch: float) -> Guard:
    return lambda x, t: float(np.linalg.norm(np.asarray(x.v_ned))) < v_switch


def descent_speed_above(vz_switch: float) -> Guard:
    """Fire when downward speed exceeds ``vz_switch`` (v_ned[2] > 0 is downward)."""
    return lambda x, t: float(np.asarray(x.v_ned)[2]) > vz_switch


def time_after(t_switch: float) -> Guard:
    return lambda x, t: t >= t_switch


def single_crossing(psi: Callable[[object, float], float], theta) -> Guard:
    """Build a guard that fires when psi(x, t, theta) >= 0."""
    return lambda x, t: float(psi(x, t, theta)) >= 0.0

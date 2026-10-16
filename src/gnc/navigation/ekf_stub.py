"""Explicitly unavailable EKF navigation selection."""

from __future__ import annotations

from core.interfaces import Measurement


class EKFNav:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("EKF navigation is not implemented; select identity navigation")

    def measure(self, state, key) -> Measurement:  # pragma: no cover
        raise NotImplementedError

    def estimate(self, meas, act, memory):  # pragma: no cover
        raise NotImplementedError


def build(ctx):
    """Reject the unavailable ``gnc/navigation=ekf`` selection explicitly."""
    return EKFNav()

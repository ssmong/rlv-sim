"""Reentry-burn guidance using full retro-thrust."""

from __future__ import annotations

import numpy as np

from core.interfaces import Phase
from gnc.guidance.base import thrust_aligned_quat, thrust_command


class ReentryBurnGuidance:
    def __init__(self, thrust_max: float, active_engines):
        self.thrust_max = thrust_max  # cluster total at full throttle
        self.active = active_engines  # the (symmetric) reentry engine set

    def __call__(self, x_hat, t, memory):
        v = x_hat.v_ned
        speed = np.linalg.norm(v)
        v_hat = v / np.maximum(speed, 1e-3)
        thrust_vec = -self.thrust_max * v_hat  # full retro, opposing velocity
        # Retro-aligned attitude reference: body-x tracks the retro direction so the sustained
        # burn carries ~zero pointing demand (allocation points in the q_ref frame). Wide pi/2
        # cap: the retro of a DESCENDING velocity always has an up-component; the s<=0 guard in
        # thrust_aligned_quat returns upright for the degenerate (horizontal/ascending) case.
        q_ref = thrust_aligned_quat(thrust_vec, np.pi / 2)
        return thrust_command(int(Phase.REENTRY_BURN), thrust_vec, q_ref, self.active), memory

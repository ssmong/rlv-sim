"""Aero-descent attitude guidance with engines off."""

from __future__ import annotations

import numpy as np

from core.frames import quat_from_euler
from core.interfaces import Phase
from gnc.guidance.base import attitude_command, upright_quat


class AeroDescentGuidance:
    def __init__(self, active_engines, steer_gain: float, max_tilt_deg: float):
        self.active = active_engines  # all-off (engine off in the aero phase)
        self.q_up = upright_quat()
        self.steer_gain = steer_gain
        self.max_tilt = np.deg2rad(max_tilt_deg)

    def __call__(self, x_hat, t, memory):
        if self.steer_gain == 0.0:
            return attitude_command(int(Phase.AERO_DESCENT), self.q_up, self.active), memory
        # crude steer: tilt nose to oppose horizontal velocity (north->pitch, east->roll proxy)
        vh = x_hat.v_ned[:2]
        tilt = np.clip(-self.steer_gain * vh, -self.max_tilt, self.max_tilt)
        q_ref = quat_from_euler(tilt[1], np.pi / 2 + tilt[0], 0.0)
        return attitude_command(int(Phase.AERO_DESCENT), q_ref, self.active), memory

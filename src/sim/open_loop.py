"""Open-loop actuator schedule for plant smoke tests."""

from __future__ import annotations

import numpy as np

from core.interfaces import ActuatorCommand


def open_loop_schedule(
    n_steps: int, plant_dt: float, n_fins: int = 4, n_engines: int = 1, n_gimbal: int = 1
) -> ActuatorCommand:
    """Stacked ActuatorCommand (leading axis n_steps) exercising all actuator channels.

    Per-engine shapes: throttle (n_steps, N), gimbal (n_steps, Ng, 2).
    """
    t = np.arange(n_steps) * plant_dt
    throttle = np.clip(0.6 + 0.2 * np.sin(0.5 * t), 0.0, 1.0)[:, None] * np.ones((1, n_engines))
    gimbal_one = np.stack([0.05 * np.sin(2.0 * t), 0.03 * np.cos(2.0 * t)], axis=1)
    gimbal = gimbal_one[:, None, :] * np.ones((1, n_gimbal, 1))
    fin = 0.1 * np.sin(1.0 * t)[:, None] * np.ones((1, n_fins))
    rcs = np.stack([0.2 * np.sign(np.sin(0.3 * t)), np.zeros(n_steps), np.zeros(n_steps)], axis=1)
    return ActuatorCommand(throttle=throttle, gimbal=gimbal, fin=fin, rcs=rcs)

"""COPY ME — control (inner attitude loop) template.

Implements the Controller contract (see the gnc/control package docstring); the config group
``configs/gnc/control/template.yaml`` points ``law._target_`` at ``build(ctx)`` below.
Replace ``__call__`` with your control law; it must fill ``Command.tau_body`` (desired body
torque [N·m]). Allocation turns that into actuator commands downstream, so you only reason
about torque here.

    python scripts/run_single.py gnc/control=template
"""

from __future__ import annotations

import numpy as np

from gnc.control.pd import attitude_error


class TemplateController:
    def __init__(self, ctx):
        # PD-style gains sized from config (wn_axis, zeta) + wet inertia — replace with your own.
        I = np.asarray(ctx.params.mass.I_wet)  # [Ixx, Iyy, Izz]
        wn = np.asarray(list(ctx.cfg.wn_axis), dtype=float)
        self.kp = I * wn**2
        self.kd = 2.0 * float(ctx.cfg.zeta) * I * wn

    def initial_memory(self) -> dict:
        return {}

    def __call__(self, x_hat, command, memory):
        # ---- YOUR CONTROL LAW HERE ---- fill command.tau_body (body torque [N·m]).
        e = attitude_error(x_hat.q_bn, command.q_ref)  # body-frame shortest-rotation error
        tau = self.kp * e - self.kd * (x_hat.w_body - command.w_ref)
        return command.replace(tau_body=tau), memory


def build(ctx):
    """Factory named by configs/gnc/control/template.yaml (ctx: gnc.context.GncContext)."""
    return TemplateController(ctx)

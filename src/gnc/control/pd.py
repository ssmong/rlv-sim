"""PD attitude controller that fills Command.tau_body."""

from __future__ import annotations

import numpy as np

from core.frames import quat_conj, quat_mul


def attitude_error(q_hat, q_ref):
    """Body-frame attitude error vector (shortest rotation q_hat -> q_ref)."""
    qe = quat_mul(quat_conj(q_hat), q_ref)
    return np.sign(qe[0] + 1e-12) * qe[1:4]


class PDController:
    def __init__(self, kp_att, kd_att):
        """``kp_att``/``kd_att`` are scalars or per-axis [roll, pitch, yaw] gains (broadcast)."""
        self.kp = np.asarray(kp_att)
        self.kd = np.asarray(kd_att)

    def initial_memory(self) -> dict:
        return {}

    def __call__(self, x_hat, command, memory):
        e = attitude_error(x_hat.q_bn, command.q_ref)
        # Rate FEEDFORWARD: damp the rate ERROR (w - w_ref), so a rotating attitude reference is
        # tracked instead of fought. w_ref = 0 (hold) recovers the plain regulator.
        tau = self.kp * e - self.kd * (x_hat.w_body - command.w_ref)  # elementwise per-axis
        return command.replace(tau_body=tau), memory


def build(ctx):
    """Factory named by configs/gnc/control/pd.yaml (ctx: gnc.context.GncContext).

    Sizes per-axis gains from the config (wn_axis, zeta) and the wet inertia:
    Kp = I_wet * wn^2, Kd = 2 * zeta * I_wet * wn."""
    I = np.asarray(ctx.params.mass.I_wet)  # [Ixx, Iyy, Izz]
    wn = np.asarray(list(ctx.cfg.wn_axis), dtype=float)
    zeta = float(ctx.cfg.zeta)
    return PDController(I * wn**2, 2.0 * zeta * I * wn)

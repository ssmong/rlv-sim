"""COPY ME — navigation (state estimator) template.

Implements the Nav contract (see the gnc/navigation package docstring); the matching config group
``configs/gnc/navigation/template.yaml`` points ``law._target_`` at ``build(ctx)`` below.
This trivial version passes truth through (like identity nav); replace ``estimate`` with
your filter (e.g. an EKF predict/update).

    python scripts/run_single.py gnc/navigation=template
"""

from __future__ import annotations

from core.interfaces import Measurement


class TemplateNav:
    def measure(self, state, key) -> Measurement:
        # Sensor model: return what your estimator will consume. Truth pass-through here.
        return Measurement(
            r_ned=state.r_ned,
            v_ned=state.v_ned,
            q_bn=state.q_bn,
            w_body=state.w_body,
            m=state.m,
        )

    def estimate(self, meas: Measurement, act, memory):
        # ---- YOUR ESTIMATOR HERE ---- (predict with `act`, update with `meas`, carry `memory`)
        x_hat = meas  # trivial: trust the measurement
        return x_hat, memory


def build(ctx):
    """Factory named by configs/gnc/navigation/template.yaml (ctx: gnc.context.GncContext)."""
    return TemplateNav()

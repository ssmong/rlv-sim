"""COPY ME — minimal powered-descent guidance template.

A runnable skeleton showing the guidance contract and how a module is wired in. Copy this
file + ``configs/gnc/guidance/template.yaml``, rename, fill in ``__call__``, and run::

    python scripts/run_single.py scenario=pdg gnc/guidance=template

The config's ``law._target_`` points at the ``build(ctx)`` factory at the bottom; ``ctx`` is a
``gnc.context.GncContext`` carrying nominal / params / active_engines / phase + your config node
(``ctx.cfg``).

Guidance contract (powered descent):
    initial_memory() -> memory                       # optional persistent state across ticks
    __call__(x_hat, t, memory) -> (thrust_ned, memory)

Return the NED thrust VECTOR your law wants; the framework attaches the phase + active engines and a
thrust-aligned attitude reference (capped at the config ``attitude_tilt_max_deg``). For a custom
attitude or an engine-off cutoff, return a full ``Command`` instead (see gnc/guidance/base.py:
``thrust_command`` / ``attitude_command``) — the framework passes any Command through unchanged.
"""

from __future__ import annotations

import numpy as np


class TemplateGuidance:
    """Trivial gravity-cancelling hover law — replace the body of ``__call__``."""

    def __init__(self, nominal: dict):
        self.nominal = nominal

    def initial_memory(self) -> dict:
        return {}

    def __call__(self, x_hat, t, memory):
        # ---- YOUR GUIDANCE LAW HERE ------------------------------------------------
        # Example: command just enough vertical thrust to cancel weight (a hover).
        g = self.nominal["g0"]
        m = float(x_hat.m)
        thrust_ned = np.array([0.0, 0.0, -m * g])  # NED "up" is -z
        # ----------------------------------------------------------------------------
        return thrust_ned, memory


def build(ctx):
    """Factory named by configs/gnc/guidance/template.yaml (ctx: gnc.context.GncContext)."""
    return TemplateGuidance(ctx.nominal)

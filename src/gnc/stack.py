"""GNCStack: compose guidance + controller + allocation into one closed-loop callable.

Exposes the signature sim.loop expects: ``(x_hat, t, memory) -> (act, memory, phase)``. Memory
bundles the guidance memory (which holds the latching phase index) and controller memory, so the
phase scheduler persists across ticks while the plant stays phase-agnostic.
"""

from __future__ import annotations

import copy
import time

import numpy as np

from gnc.allocation import allocate


class GNCStack:
    def __init__(self, guidance, controller, nominal: dict):
        self.guidance = guidance
        self.controller = controller
        self.nominal = nominal

    def initial_memory(self) -> dict:
        return {
            "guidance": self.guidance.initial_memory(),
            "controller": self.controller.initial_memory(),
            "control_timing": [],  # per-call controller compute time [ms]; aggregated in sim.loop
        }

    @staticmethod
    def _np_state(x_hat):
        """Normalize partial test stubs and navigation records to host NumPy values."""
        fields = {
            k: np.asarray(getattr(x_hat, k))
            for k in ("r_ned", "v_ned", "q_bn", "w_body", "m")
            if getattr(x_hat, k, None) is not None
        }
        if hasattr(x_hat, "replace"):
            return x_hat.replace(**fields)
        out = copy.copy(x_hat)  # SimpleNamespace-style test stubs
        for k, v in fields.items():
            setattr(out, k, v)
        return out

    def guide(self, x_hat, t, memory: dict):
        """Outer guidance tick (slow rate): returns (Command, memory). Held ZOH between ticks."""
        cmd, gmem = self.guidance(self._np_state(x_hat), t, memory["guidance"])
        return cmd, {**memory, "guidance": gmem}

    def control_allocate(self, cmd, x_hat, memory: dict, t=None):
        """Inner control + allocation tick (FAST rate, ctrl_dt): returns (ActuatorCommand, memory).

        The translational TRACKING feedback (the guidance FF+FB on its cached plan) belongs at the
        fast control rate, not the slow guidance re-solve rate, so when guidance exposes a
        ``track`` (the convex guidances) and ``t`` is supplied, refresh the thrust command here on
        the latest measured state before attitude control + allocation. The slow convex re-solve
        stays in ``guide`` (guidance_dt). Falls back to the held ``cmd`` if there is no track/t."""
        x_hat = self._np_state(x_hat)
        if t is not None and hasattr(self.guidance, "track"):
            cmd, gmem = self.guidance.track(x_hat, t, memory["guidance"])
            memory = {**memory, "guidance": gmem}
        t0 = time.perf_counter_ns()
        cmd, cmem = self.controller(x_hat, cmd, memory["controller"])
        ctrl_ms = (time.perf_counter_ns() - t0) * 1e-6  # control LAW only (allocation excluded)
        timing = list(memory.get("control_timing") or [])  # copy: never alias input memory's list
        timing.append(ctrl_ms)
        act = allocate(cmd, x_hat, self.nominal)
        return act, {**memory, "controller": cmem, "control_timing": timing}

    def __call__(self, x_hat, t, memory: dict):
        """Single-rate path: guidance + control + allocation in one tick (sim.loop default)."""
        cmd, memory = self.guide(x_hat, t, memory)
        act, memory = self.control_allocate(cmd, x_hat, memory, t)
        return act, memory, int(cmd.phase)

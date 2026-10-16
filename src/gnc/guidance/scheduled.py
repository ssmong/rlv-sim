"""Phase-scheduled guidance dispatcher."""

from __future__ import annotations

from gnc.guidance.phase import SequentialPhaseScheduler


class ScheduledGuidance:
    def __init__(self, scheduler: SequentialPhaseScheduler, phase_guidance: dict):
        self.scheduler = scheduler
        self.phase_guidance = phase_guidance
        missing = set(scheduler.phases) - set(phase_guidance)
        if missing:
            raise ValueError(f"no guidance for phases {missing}")

    def initial_memory(self) -> dict:
        # Seed the shared sub-memory from each phase law's initial_memory (a law that omits it adds
        # nothing) so a stateful law's first tick already sees its initial state, not an empty dict.
        sub: dict = {}
        for g in self.phase_guidance.values():
            if hasattr(g, "initial_memory"):
                sub.update(g.initial_memory())
        return {"scheduler": self.scheduler.initial_memory(), "sub": sub}

    def __call__(self, x_hat, t, memory: dict):
        phase, smem = self.scheduler.phase(x_hat, t, memory["scheduler"])
        cmd, sub = self.phase_guidance[phase](x_hat, t, memory.get("sub", {}))
        return cmd, {"scheduler": smem, "sub": sub, "phase": int(phase)}

    def track(self, x_hat, t, memory: dict):
        """Delegate fast tracking to the last resolved phase."""
        phase = memory.get("phase")
        if phase is None:  # not yet guided this run => fall back to a full guidance tick
            return self(x_hat, t, memory)
        g = self.phase_guidance[phase]
        sub = memory.get("sub", {})
        cmd, sub = g.track(x_hat, t, sub) if hasattr(g, "track") else g(x_hat, t, sub)
        return cmd, {**memory, "sub": sub}

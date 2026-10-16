"""Regression checks for honest convex-guidance fallback and solve telemetry."""

from types import SimpleNamespace

import numpy as np

from gnc.guidance.convex.guidance import LcvxGuidance
from sim.loop import _guidance_degradation, _solver_stats


def _attempt(feasible, status):
    return {
        "kind": "unit_conic",
        "wall_time_sec": 0.001,
        "solve_time_sec": None,
        "iters": 2,
        "status": status,
        "feasible": feasible,
        "tf": 5.0,
        "N": 10,
    }


class _RefinementStub(LcvxGuidance):
    def __init__(self, *, drag_template=False):
        self.grid_fallback = False
        self.terminal_slack = False
        self.reanchor_min_alt = 0.0
        self.tf_freeze = 0.0
        self.replan = False
        self.drag_template = drag_template
        self.plan_calls = 0

    def _should_cutoff(self, x_hat):
        return False

    def _has_drag_template(self):
        return self.drag_template

    def _plan(self, *args, **kwargs):
        self.plan_calls += 1
        feasible = not self.drag_template or self.plan_calls == 1
        self._last_status = "optimal" if feasible else "infeasible"
        self._last_grid_fb = False
        self._last_attempts = [
            _attempt(False, "candidate_infeasible"),
            _attempt(feasible, self._last_status),
        ]
        return {"tf": 5.0} if feasible else None

    def _coast(self, mem):
        return "coast", mem

    def _track(self, x_hat, t, mem):
        return "track", mem


def _state():
    return SimpleNamespace(r_ned=[0.0, 0.0, -100.0], v_ned=[0.0, 0.0, 10.0], m=1000.0)


def test_failed_drag_refinement_never_commits_drag_free_seed():
    guidance = _RefinementStub(drag_template=True)
    _, memory = guidance(_state(), 0.0, guidance.initial_memory())

    assert not memory["ignited"]
    assert memory["template_refinement_failures"] == 1
    assert len(memory["solve_log"]) == 4


def test_single_shot_does_not_solve_again_after_ignition():
    guidance = _RefinementStub()
    _, memory = guidance(_state(), 0.0, guidance.initial_memory())
    _, memory = guidance(_state(), 1.0, memory)

    assert memory["ignited"]
    assert guidance.plan_calls == 1


def test_exhausted_plan_commands_engine_off():
    guidance = object.__new__(LcvxGuidance)
    guidance.phase = 3
    guidance.q_up = np.array([1.0, 0.0, 0.0, 0.0])
    guidance.active = np.ones(1)
    memory = guidance.initial_memory()
    memory.update(plan={"tf": 1.0}, t_ignite=0.0, ignited=True)

    command, memory = guidance._track(_state(), 1.0, memory)

    assert int(command.mode) == 1
    assert np.allclose(command.thrust_vector, 0.0)
    assert np.allclose(command.active_engines, 0.0)
    assert memory["plan_exhausted_ticks"] == 1


def test_unavailable_backend_time_is_none_and_degradation_is_exportable():
    logs = [
        {
            "kind": "ignite_search",
            "wall_ms": 1.0,
            "solve_ms": None,
            "iters": 2,
            "feasible": False,
            "solver_feasible": True,
            "mission_feasible": False,
            "postcheck_failures": ["thrust_bounds"],
            "grid_fallback": True,
        },
        {
            "kind": "ignite_search",
            "record_type": "outcome",
            "wall_ms": 2.0,
            "solve_ms": None,
            "iters": 4,
            "feasible": False,
            "solver_feasible": True,
            "mission_feasible": False,
            "grid_fallback": False,
        },
    ]
    stats = _solver_stats(logs)
    memory = {"cached_plan_fallbacks": 2}
    degradation = _guidance_degradation(memory, logs)

    assert stats["solve_ms"] is None
    assert stats["n_backend_timed"] == 0
    assert stats["n_grid_fallback"] == 1
    assert stats["n_solver_infeasible"] == 0
    assert stats["n_solves"] == 1
    assert stats["n_outcomes"] == 1
    assert stats["n_failed_outcomes"] == 1
    assert stats["n_postcheck_rejected_solves"] == 1
    assert degradation == {
        "cached_plan_fallbacks": 2,
        "grid_fallbacks": 1,
    }

"""Post-solve feasibility gates shared by every LCvx backend."""

from __future__ import annotations

from importlib.util import find_spec

import numpy as np
import pytest

pytest.importorskip("cvxpy")

from gnc.guidance.convex.lcvx import solve_pdg
from gnc.guidance.convex.qoco_direct import _postcheck_solution

G0 = 9.80665
RHO_MAX = 981000.0
RHO_MIN = 0.4 * RHO_MAX
ARGS = (np.array([0.0, 0.0, G0]), 1.0 / (300.0 * G0), RHO_MIN, RHO_MAX, 25000.0)
R0 = np.array([0.0, 0.0, -250.0])
V0 = np.array([0.0, 0.0, 55.0])


def _available_backends():
    backends = ["CLARABEL_CVXPY"]
    if find_spec("qoco") is not None:
        backends.append("QOCO")
    if find_spec("clarabel") is not None:
        backends.append("CLARABEL")
    return backends


@pytest.mark.parametrize("solver", _available_backends())
def test_solver_optimal_is_not_enough_for_physical_feasibility(solver):
    result = solve_pdg(R0, V0, 30000.0, *ARGS, tf=6.0, N=35, solver=solver)

    assert result["solver_status"].startswith("optimal")
    assert not result["feasible"]
    assert not result["plan_physical_feasible"]
    assert {"relaxation_gap", "thrust_bounds"} <= set(result["postcheck_failures"])


@pytest.mark.parametrize("solver", _available_backends())
def test_physical_hard_terminal_solution_remains_feasible(solver):
    result = solve_pdg(R0, V0, 30000.0, *ARGS, tf=6.5, N=35, solver=solver)

    assert result["feasible"]
    assert result["mission_feasible"]
    assert result["plan_physical_feasible"]
    assert result["terminal_constraints_met"]
    assert result["postcheck_failures"] == []


@pytest.mark.parametrize("solver", _available_backends())
def test_zero_terminal_slack_is_feasible_but_labeled_relaxed(solver):
    result = solve_pdg(R0, V0, 30000.0, *ARGS, tf=6.5, N=35, solver=solver, terminal_slack=True)

    assert result["solver_feasible"]
    assert result["relaxed_feasible"]
    assert result["feasible"]
    assert result["mission_feasible"]
    assert result["status"].startswith("relaxed_")
    assert result["solution_class"] == "relaxed_terminal"
    assert result["postcheck_failures"] == []


@pytest.mark.parametrize("solver", [solver for solver in _available_backends() if solver != "QOCO"])
def test_large_terminal_slack_is_not_mission_feasible(solver):
    result = solve_pdg(R0, V0, 30000.0, *ARGS, tf=0.5, N=35, solver=solver, terminal_slack=True)

    assert result["solver_feasible"]
    assert result["relaxed_feasible"]
    assert not result["feasible"]
    assert not result["mission_feasible"]
    assert result["status"] == "relaxed_postcheck_failed"
    assert result["solution_class"] == "relaxed_terminal"
    assert {"terminal_position", "terminal_velocity"} <= set(result["postcheck_failures"])


def test_hard_terminal_residual_is_rejected():
    result = {
        "status": "optimal",
        "r": np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.01]]),
        "v": np.zeros((2, 3)),
        "m": np.array([100.0, 99.0]),
        "u": np.array([[0.0, 0.0, -10.0]]),
        "sigma": np.array([10.0]),
        "thrust": np.array([[0.0, 0.0, -1000.0]]),
        "slack_gap": 0.0,
        "terminal_pos_slack": 0.01,
    }

    checked = _postcheck_solution(result, 900.0, 1100.0, False, None)

    assert checked["plan_physical_feasible"]
    assert not checked["feasible"]
    assert checked["postcheck_failures"] == ["terminal_position"]

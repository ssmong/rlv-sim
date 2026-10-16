"""Conic solver backend selection for convex guidance."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from typing import Protocol

OPTIMAL_STATUSES = ("optimal", "optimal_inaccurate")


class ConicSolverBackend(Protocol):
    """Backend that solves a CVXPY conic problem."""

    name: str

    def solve(self, problem) -> None:
        """Solve ``problem`` in place."""


@dataclass(frozen=True)
class CvxpyConicSolver:
    """Thin CVXPY solver adapter."""

    name: str

    def solve(self, problem) -> None:
        problem.solve(solver=self.name)


@dataclass(frozen=True)
class QocoConicSolver:
    """QOCO adapter for CVXPY-built SCvx subproblems."""

    name: str = "QOCO"

    def solve(self, problem) -> None:
        if find_spec("qoco") is None:
            raise ImportError("qoco is required for convex_solver=QOCO")
        problem.solve(solver=self.name)


def make_conic_solver(name: str | ConicSolverBackend) -> ConicSolverBackend:
    """Instantiate the requested conic solver backend (CLARABEL default, QOCO alternative)."""
    if hasattr(name, "solve"):
        return name
    key = str(name).upper().replace("-", "_")
    if key in {"QOCO", "QOCO_CVXPY"}:
        return QocoConicSolver()
    if key in {"CLARABEL", "CLARABEL_CVXPY"}:
        return CvxpyConicSolver("CLARABEL")
    raise ValueError(
        f"unknown convex solver {name!r}; pick QOCO|QOCO_CVXPY|CLARABEL|CLARABEL_CVXPY"
    )


def solve_problem(problem, solver: str | ConicSolverBackend) -> None:
    """Solve a CVXPY problem using the selected backend."""
    make_conic_solver(solver).solve(problem)

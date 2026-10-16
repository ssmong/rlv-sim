"""Native SCvx exposes unsupported relaxations instead of applying a fallback."""

import numpy as np
import pytest

pytest.importorskip("cvxpy")

from gnc.guidance.convex.scvx import solve_scvx


def test_terminal_slack_is_explicitly_unsupported():
    g0 = 9.80665
    with pytest.raises(NotImplementedError, match="hard terminal"):
        solve_scvx(
            np.array([0.0, 0.0, -250.0]),
            np.array([0.0, 0.0, 55.0]),
            30000.0,
            np.array([0.0, 0.0, g0]),
            1.0 / (300.0 * g0),
            392400.0,
            981000.0,
            25000.0,
            tf=0.5,
            N=20,
            beta0=0.0,
            terminal_slack=True,
        )

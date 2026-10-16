"""Direct Clarabel backend for the matrix-stuffed LCvx SOCP (no CVXPY canonicalization).

Reuses qoco_direct's canonical builder: Clarabel solves min 1/2 x'Px + q'x s.t. Ax + s = b,
s in K, so the QOCO data maps as A_cl = [A; G], b_cl = [b; h], K = Zero(p) x Nonneg(l) x SOCs.
Unlike QOCO, Clarabel reports genuine infeasibility certificates (PrimalInfeasible/...).
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from gnc.guidance.convex.qoco_direct import (
    _build_pdg_socp,
    _postcheck_solution,
    _unpack_solution,
)


# fmt: off
def solve_pdg_clarabel(
    r0, v0, m0,
    g, alpha, rho_min, rho_max, m_dry,
    tf, N,
    target=(0.0, 0.0, 0.0), theta_max=None, gamma_gs=None,
    p_sea=0.0, nozzle_exit_area=0.0, scale_height=None, n_engines=1, p_amb_alt=None,
    terminal_slack=False, terminal_pos_weight=1.0e4, terminal_vel_weight=1.0e4,
    terminal_vz_max=None, terminal_upright=False,
    pos_scale=None, vel_scale=None, accel_scale=None,
    slew_rate_max=None, thrust_rate_max=None,
    drag_seq=None,
):
# fmt: on
    """Solve the fixed-time LCvx SOCP by passing canonical matrices directly to Clarabel."""
    try:
        import clarabel
    except ImportError as exc:  # pragma: no cover
        raise ImportError("clarabel is required for direct CLARABEL LCvx guidance") from exc

    # fmt: off
    d = _build_pdg_socp(
        r0, v0, m0,
        g, alpha, rho_min, rho_max, m_dry,
        tf, N,
        target=target, theta_max=theta_max, gamma_gs=gamma_gs,
        p_sea=p_sea, nozzle_exit_area=nozzle_exit_area,
        scale_height=scale_height, n_engines=n_engines, p_amb_alt=p_amb_alt,
        terminal_slack=terminal_slack,
        terminal_pos_weight=terminal_pos_weight,
        terminal_vel_weight=terminal_vel_weight,
        terminal_vz_max=terminal_vz_max,
        terminal_upright=terminal_upright,
        pos_scale=pos_scale, vel_scale=vel_scale, accel_scale=accel_scale,
        slew_rate_max=slew_rate_max,
        thrust_rate_max=thrust_rate_max,
        drag_seq=drag_seq,
    )
    # fmt: on
    a_cl = sparse.vstack([d["A"], d["G"]], format="csc")
    b_cl = np.concatenate([d["b"], d["h"]])
    cones = [clarabel.ZeroConeT(d["p"]), clarabel.NonnegativeConeT(d["l"])]
    cones += [clarabel.SecondOrderConeT(int(qi)) for qi in d["q"]]
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    solver = clarabel.DefaultSolver(d["P"], d["c"], a_cl, b_cl, cones, settings)
    sol = solver.solve()

    accept = {
        clarabel.SolverStatus.Solved: "optimal",
        clarabel.SolverStatus.AlmostSolved: "optimal_inaccurate",
    }
    if sol.status not in accept:
        return {"status": str(sol.status), "feasible": False}

    out = _unpack_solution(np.asarray(sol.x), d)
    out["status"] = accept[sol.status]
    out["clarabel_status"] = str(sol.status)
    out["iters"] = int(sol.iterations)
    out["solve_time_sec"] = float(sol.solve_time)
    return _postcheck_solution(
        out,
        d["rho_min_ach"],
        d["rho_max_ach"],
        d["terminal_slack"],
        d["terminal_vz_max"],
    )

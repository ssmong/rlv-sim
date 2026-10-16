"""Direct canonical-form builder + QOCO backend for fixed-time LCvx powered descent.

_build_pdg_socp produces the matrix-stuffed SOCP data shared by the direct backends: QOCO
consumes (A,b) + (G,h,l,q) as-is; clarabel_direct stacks the same data into Clarabel's
Ax + s = b, s in Zero x Nonneg x SOC form. One builder, several solvers.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse


def _postcheck_solution(
    out: dict,
    rho_min_ach: float,
    rho_max_ach: float,
    terminal_slack: bool,
    terminal_vz_max: float | None,
) -> dict:
    """Reject solver outputs that fail the selected 3-DOF plan or terminal postchecks."""
    thrust_mag = np.linalg.norm(np.asarray(out["thrust"]), axis=1)
    sigma = np.asarray(out["sigma"])
    slack_gap = float(out["slack_gap"])
    relaxation_tol = max(1.0e-5, 1.0e-5 * float(np.max(np.abs(sigma))))
    thrust_tol = max(1.0, 1.0e-5 * float(rho_max_ach))
    terminal_pos_tol = terminal_vel_tol = 1.0e-3

    min_thrust_violation = max(float(rho_min_ach) - float(np.min(thrust_mag)), 0.0)
    max_thrust_violation = max(float(np.max(thrust_mag)) - float(rho_max_ach), 0.0)
    terminal_pos_violation = float(out["terminal_pos_slack"])
    v_final = np.asarray(out["v"])[-1]
    terminal_vel_violation = (
        float(np.linalg.norm(v_final))
        if terminal_vz_max is None
        else max(
            float(np.linalg.norm(v_final[:2])),
            -float(v_final[2]),
            float(v_final[2]) - float(terminal_vz_max),
            0.0,
        )
    )
    finite = all(
        np.all(np.isfinite(np.asarray(out[key]))) for key in ("r", "v", "m", "u", "sigma", "thrust")
    )

    failures = []
    if not finite:
        failures.append("nonfinite")
    if slack_gap > relaxation_tol:
        failures.append("relaxation_gap")
    if max(min_thrust_violation, max_thrust_violation) > thrust_tol:
        failures.append("thrust_bounds")
    plan_physical_feasible = not failures
    terminal_constraints_met = (
        terminal_pos_violation <= terminal_pos_tol and terminal_vel_violation <= terminal_vel_tol
    )
    if terminal_pos_violation > terminal_pos_tol:
        failures.append("terminal_position")
    if terminal_vel_violation > terminal_vel_tol:
        failures.append("terminal_velocity")
    solver_status = str(out["status"])
    mission_feasible = plan_physical_feasible and terminal_constraints_met
    out.update(
        solver_status=solver_status,
        feasible=mission_feasible,
        mission_feasible=mission_feasible,
        plan_physical_feasible=plan_physical_feasible,
        relaxed_feasible=bool(terminal_slack and plan_physical_feasible),
        solution_class="relaxed_terminal" if terminal_slack else "hard_terminal",
        terminal_constraints_met=terminal_constraints_met,
        postcheck_failures=failures,
        min_thrust_violation=min_thrust_violation,
        max_thrust_violation=max_thrust_violation,
        thrust_bound_tolerance=thrust_tol,
        relaxation_tolerance=relaxation_tol,
        terminal_pos_tolerance=terminal_pos_tol,
        terminal_vel_tolerance=terminal_vel_tol,
        terminal_vel_violation=terminal_vel_violation,
    )
    if terminal_slack and mission_feasible:
        out["status"] = f"relaxed_{solver_status}"
    elif terminal_slack:
        out["status"] = "relaxed_postcheck_failed"
    elif not mission_feasible:
        out["status"] = "postcheck_failed"
    return out


# fmt: off
def _build_pdg_socp(
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
    """Build the canonical SOCP data (P,c,A,b,G,h,l,q + unpack metadata) for one LCvx solve."""
    r0 = np.asarray(r0, float)
    v0 = np.asarray(v0, float)
    g = np.asarray(g, float)
    target = np.asarray(target, float)
    dt = tf / N

    # Variables are scaled for conditioning; dt remains in physical seconds.
    # r_hat=(r-target)/pos_scale, v_hat=v/vel_scale, u_hat=u/accel_scale. With FIXED caller scales
    # (the guidance passes gnc.convex_scale_*), conditioning + the slack penalty (~pos_scale) stay
    # identical across re-solves; if omitted, fall back to per-solve state-derived scaling.
    pos_scale = (
        float(pos_scale)
        if pos_scale is not None
        else max(float(np.linalg.norm(r0 - target)), 100.0)
    )
    time_scale = max(float(tf), 1.0e-6)

    h_rep = (-float(r0[2])) if p_amb_alt is None else float(p_amb_alt)
    if scale_height is None or p_sea <= 0.0:
        p_amb_const = float(p_sea)
    else:
        p_amb_const = float(p_sea) * np.exp(-max(h_rep, 0.0) / float(scale_height))
    bp_force = float(n_engines) * p_amb_const * float(nozzle_exit_area)
    rho_max_ach = rho_max - bp_force
    if rho_max_ach <= 0.0:
        raise ValueError(
            f"back-pressure force {bp_force:.0f} N >= max vacuum thrust {rho_max:.0f} N: "
            "no positive net thrust is achievable"
        )
    # A throttle floor at/below the back-pressure debit is vacuous: DROP the lower-bound cone
    # (sigma >= 0 is implied) instead of clamping, which puts O(1/mu1)~1e9 entries in its rows.
    rho_min_ach = rho_min - bp_force
    has_rho_min = rho_min_ach > 1e-6 * rho_max
    vel_scale = (
        float(vel_scale)
        if vel_scale is not None
        else max(float(np.linalg.norm(v0)), pos_scale / time_scale, 10.0)
    )
    accel_scale = (
        float(accel_scale)
        if accel_scale is not None
        else max(rho_max_ach / max(float(m_dry), 1.0e-6), float(np.linalg.norm(g)), 1.0)
    )
    r0_hat = (r0 - target) / pos_scale
    v0_hat = v0 / vel_scale

    ref_flow = alpha * (rho_max_ach + bp_force) * dt
    m_ref = np.maximum(m0 - ref_flow * np.arange(N), 1e-3)
    z0_seq = np.log(m_ref)

    off_r = 0
    off_v = off_r + 3 * (N + 1)
    off_z = off_v + 3 * (N + 1)
    off_u = off_z + (N + 1)
    off_sigma = off_u + 3 * N
    off_e_r = off_sigma + N
    off_e_v = off_e_r + 3
    off_s_r = off_e_v + 3
    off_s_v = off_s_r + 1
    n = off_s_v + 1 if terminal_slack else off_sigma + N

    def r_i(k, j):
        return off_r + 3 * k + j

    def v_i(k, j):
        return off_v + 3 * k + j

    def z_i(k):
        return off_z + k

    def u_i(k, j):
        return off_u + 3 * k + j

    def sig_i(k):
        return off_sigma + k

    def e_r_i(j):
        return off_e_r + j

    def e_v_i(j):
        return off_e_v + j

    def s_r_i():
        return off_s_r

    def s_v_i():
        return off_s_v

    a_rows: list[int] = []
    a_cols: list[int] = []
    a_data: list[float] = []
    b_vals: list[float] = []

    def add_eq(coeffs, rhs):
        row = len(b_vals)
        for col, val in coeffs.items():
            a_rows.append(row)
            a_cols.append(col)
            a_data.append(float(val))
        b_vals.append(float(rhs))

    for j in range(3):
        add_eq({r_i(0, j): 1.0}, r0_hat[j])
        add_eq({v_i(0, j): 1.0}, v0_hat[j])
    add_eq({z_i(0): 1.0}, np.log(m0))

    # DESCENDO frozen-coefficient drag template (Simplicio 2019 eqs. 20-21): drag_seq[k] = d*_k
    # [1/s] from the PREVIOUS solution multiplies the CURRENT velocity variable — convexity is
    # preserved (linear in v) and the dominant atmospheric deceleration enters the plan.
    d_seq = np.zeros(N) if drag_seq is None else np.asarray(drag_seq, float)
    for k in range(N):
        dk = float(d_seq[k])
        for j in range(3):
            add_eq(
                {
                    v_i(k + 1, j): 1.0,
                    v_i(k, j): -(1.0 - dt * dk),
                    u_i(k, j): -(dt * accel_scale / vel_scale),
                },
                dt * g[j] / vel_scale,
            )
            add_eq(
                {
                    r_i(k + 1, j): 1.0,
                    r_i(k, j): -1.0,
                    v_i(k, j): -(dt * (1.0 - 0.5 * dt * dk) * vel_scale / pos_scale),
                    u_i(k, j): -(0.5 * dt**2 * accel_scale / pos_scale),
                },
                0.5 * dt**2 * g[j] / pos_scale,
            )
        bp_coeff = alpha * bp_force * np.exp(-z0_seq[k])
        add_eq(
            {
                z_i(k + 1): 1.0,
                z_i(k): -(1.0 + dt * bp_coeff),
                sig_i(k): alpha * dt * accel_scale,
            },
            -dt * bp_coeff * (1.0 + z0_seq[k]),
        )

    if not terminal_slack:
        for j in range(3):
            add_eq({r_i(N, j): 1.0}, 0.0)
        # Terminal velocity: exact rest, or the DESCENDO sink window (lateral rest + bounded
        # descent rate) when terminal_vz_max is set — the inequality rows live in the G/h block.
        vz_exact = 3 if terminal_vz_max is None else 2
        for j in range(vz_exact):
            add_eq({v_i(N, j): 1.0}, 0.0)
    if terminal_upright:
        # Terminal upright thrust (DESCENDO boundary w_xy[N]=0): final control node is vertical.
        add_eq({u_i(N - 1, 0): 1.0}, 0.0)
        add_eq({u_i(N - 1, 1): 1.0}, 0.0)

    g_rows: list[int] = []
    g_cols: list[int] = []
    g_data: list[float] = []
    h_vals: list[float] = []

    def add_ineq(coeffs, h_val):
        row = len(h_vals)
        for col, val in coeffs.items():
            g_rows.append(row)
            g_cols.append(col)
            g_data.append(float(val))
        h_vals.append(float(h_val))

    add_ineq({z_i(N): -1.0}, -np.log(m_dry))
    if not terminal_slack and terminal_vz_max is not None:
        # Terminal sink window: 0 <= vz[N] <= vz_max (descend into the cutoff box; exact v=0
        # demands a terminal hover the throttle floor cannot hold with TWR>1).
        add_ineq({v_i(N, 2): 1.0}, float(terminal_vz_max) / vel_scale)
        add_ineq({v_i(N, 2): -1.0}, 0.0)
    if terminal_slack:
        for j in range(3):
            add_ineq({r_i(N, j): 1.0, e_r_i(j): -1.0}, 0.0)
            add_ineq({r_i(N, j): -1.0, e_r_i(j): -1.0}, 0.0)
            add_ineq({e_r_i(j): -1.0}, 0.0)
            add_ineq({v_i(N, j): 1.0, e_v_i(j): -1.0}, 0.0)
            add_ineq({v_i(N, j): -1.0, e_v_i(j): -1.0}, 0.0)
            add_ineq({e_v_i(j): -1.0}, 0.0)
    for k in range(N):
        mu2 = rho_max_ach * np.exp(-z0_seq[k]) / accel_scale
        add_ineq({z_i(k): mu2, sig_i(k): 1.0}, mu2 * (1.0 + z0_seq[k]))
        if theta_max is not None:
            add_ineq({u_i(k, 2): 1.0, sig_i(k): np.cos(theta_max)}, 0.0)
    if thrust_rate_max is not None:
        # Planned thrust-MAGNITUDE rate constraint: |T[k+1]-T[k]| <= Tdot_max*dt, i.e.
        # |sigma[k+1]-sigma[k]| <= Tdot_max*dt*e^{-z_k}, with the SAME first-order (1-dz) mass
        # correction as the thrust bounds above (tangent of convex e^{-z} at z0 lies below ->
        # conservative side; the zeroth-order frozen-mass form is permissive by m_k/m_ref_k
        # since z0_seq is the fastest-depletion profile). Two LINEAR rows
        # per node pair (convexity untouched; scalar, leaves thrust DIRECTION free, unlike a
        # vector slew SOC). This follows G-FOLD Eqs. (6c)-(6d) and DESCENDO eq. (19), but it does
        # not model the initial throttle state, servo lag, shutdown/restart, or attitude response;
        # satisfying it is not an actuator-feasibility certificate.
        for k in range(N - 1):
            b_k = float(thrust_rate_max) * dt * np.exp(-z0_seq[k]) / accel_scale
            rhs = b_k * (1.0 + z0_seq[k])
            add_ineq({sig_i(k + 1): 1.0, sig_i(k): -1.0, z_i(k): b_k}, rhs)
            add_ineq({sig_i(k): 1.0, sig_i(k + 1): -1.0, z_i(k): b_k}, rhs)
    l = len(h_vals)
    q: list[int] = []

    def add_soc(components):
        for coeffs, h_val in components:
            add_ineq(coeffs, h_val)
        q.append(len(components))

    for k in range(N):
        add_soc(
            [
                ({sig_i(k): -1.0}, 0.0),
                ({u_i(k, 0): -1.0}, 0.0),
                ({u_i(k, 1): -1.0}, 0.0),
                ({u_i(k, 2): -1.0}, 0.0),
            ]
        )
        if has_rho_min:
            mu1 = rho_min_ach * np.exp(-z0_seq[k]) / accel_scale
            add_soc(
                [
                    ({sig_i(k): -2.0 / mu1}, 0.0),
                    ({z_i(k): -2.0}, -2.0 * (z0_seq[k] + 1.0)),
                    ({sig_i(k): -2.0 / mu1}, -2.0),
                ]
            )
        if gamma_gs is not None:
            tg = np.tan(gamma_gs)
            add_soc(
                [
                    ({r_i(k, 2): 1.0}, 0.0),
                    ({r_i(k, 0): -tg}, 0.0),
                    ({r_i(k, 1): -tg}, 0.0),
                ]
            )

    if slew_rate_max is not None:
        # Trackable thrust profile: ||u_{k+1}-u_k|| <= rate*dt (SOC). Inter-node coupling sits
        # OUTSIDE the CT-2013 / DT-LCvx-2025 lossless guarantees (pointwise control-set arguments)
        # -> gate activation on slack_gap and min-throttle violations.
        du_hat = float(slew_rate_max) * dt / accel_scale
        for k in range(N - 1):
            add_soc(
                [
                    ({}, du_hat),
                    ({u_i(k + 1, 0): -1.0, u_i(k, 0): 1.0}, 0.0),
                    ({u_i(k + 1, 1): -1.0, u_i(k, 1): 1.0}, 0.0),
                    ({u_i(k + 1, 2): -1.0, u_i(k, 2): 1.0}, 0.0),
                ]
            )

    if terminal_slack:
        add_soc(
            [
                ({s_r_i(): -1.0}, 0.0),
                ({r_i(N, 0): -1.0}, 0.0),
                ({r_i(N, 1): -1.0}, 0.0),
                ({r_i(N, 2): -1.0}, 0.0),
            ]
        )
        add_soc(
            [
                ({s_v_i(): -1.0}, 0.0),
                ({v_i(N, 0): -1.0}, 0.0),
                ({v_i(N, 1): -1.0}, 0.0),
                ({v_i(N, 2): -1.0}, 0.0),
            ]
        )

    p = len(b_vals)
    m = len(h_vals)
    P = sparse.csc_matrix((n, n))
    c = np.zeros(n)
    for k in range(N):
        c[sig_i(k)] = dt * accel_scale
    if terminal_slack:
        for j in range(3):
            c[e_r_i(j)] = float(terminal_pos_weight) * pos_scale
            c[e_v_i(j)] = float(terminal_vel_weight) * vel_scale
        c[s_r_i()] = float(terminal_pos_weight) * pos_scale
        c[s_v_i()] = float(terminal_vel_weight) * vel_scale
    A = sparse.coo_matrix((a_data, (a_rows, a_cols)), shape=(p, n)).tocsc()
    b = np.asarray(b_vals, dtype=float)
    G = sparse.coo_matrix((g_data, (g_rows, g_cols)), shape=(m, n)).tocsc()
    h = np.asarray(h_vals, dtype=float)

    return {
        "P": P, "c": c, "A": A, "b": b, "G": G, "h": h, "l": l, "q": q,
        "n": n, "m": m, "p": p,
        "off_r": off_r, "off_v": off_v, "off_z": off_z, "off_u": off_u, "off_sigma": off_sigma,
        "off_s_r": off_s_r, "off_s_v": off_s_v,
        "pos_scale": pos_scale, "time_scale": time_scale,
        "vel_scale": vel_scale, "accel_scale": accel_scale,
        "target": target, "tf": tf, "N": N, "m0": float(m0),
        "rho_min_ach": max(float(rho_min_ach), 0.0), "rho_max_ach": float(rho_max_ach),
        "terminal_slack": bool(terminal_slack),
        "terminal_vz_max": terminal_vz_max,
    }


def _unpack_solution(x: np.ndarray, d: dict) -> dict:
    """Assemble the physical-units trajectory dict from a canonical solution vector."""
    N = d["N"]
    target = d["target"]
    pos_scale, vel_scale, accel_scale = d["pos_scale"], d["vel_scale"], d["accel_scale"]
    terminal_slack = d["terminal_slack"]
    r_hat = x[d["off_r"] : d["off_v"]].reshape(N + 1, 3)
    v_hat = x[d["off_v"] : d["off_z"]].reshape(N + 1, 3)
    z_val = x[d["off_z"] : d["off_u"]]
    u_hat = x[d["off_u"] : d["off_sigma"]].reshape(N, 3)
    sigma_hat = x[d["off_sigma"] : d["off_sigma"] + N]
    r_val = target + pos_scale * r_hat
    v_val = vel_scale * v_hat
    u_val = accel_scale * u_hat
    sigma_val = accel_scale * sigma_hat
    m_val = np.exp(z_val)
    thrust = m_val[:N, None] * u_val
    terminal_pos_slack = float(np.linalg.norm(r_val[-1] - target))
    terminal_vel_slack = float(np.linalg.norm(v_val[-1]))
    terminal_pos_l1_slack = float(np.sum(np.abs(r_val[-1] - target)))
    terminal_vel_l1_slack = float(np.sum(np.abs(v_val[-1])))
    terminal_pos_l2_epigraph = (
        float(pos_scale * x[d["off_s_r"]]) if terminal_slack else terminal_pos_slack
    )
    terminal_vel_l2_epigraph = (
        float(vel_scale * x[d["off_s_v"]]) if terminal_slack else terminal_vel_slack
    )
    return {
        "solver_feasible": True,
        "tf": d["tf"],
        "r": r_val,
        "v": v_val,
        "m": m_val,
        "u": u_val,
        "sigma": sigma_val,
        "thrust": thrust,
        "fuel": float(d["m0"] - m_val[-1]),
        "slack_gap": float(np.max(np.abs(np.linalg.norm(u_val, axis=1) - sigma_val))),
        "terminal_slack": terminal_slack,
        "terminal_pos_slack": terminal_pos_slack,
        "terminal_vel_slack": terminal_vel_slack,
        "terminal_pos_l1_slack": terminal_pos_l1_slack,
        "terminal_vel_l1_slack": terminal_vel_l1_slack,
        "terminal_pos_l2_epigraph": terminal_pos_l2_epigraph,
        "terminal_vel_l2_epigraph": terminal_vel_l2_epigraph,
        "scale_pos": pos_scale,
        "scale_time": d["time_scale"],
        "scale_vel": vel_scale,
        "scale_accel": accel_scale,
    }


# fmt: off
def solve_pdg_qoco(
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
    """Solve the fixed-time LCvx SOCP by passing canonical matrices directly to QOCO."""
    try:
        import qoco
    except ImportError as exc:  # pragma: no cover
        raise ImportError("qoco is required for direct QOCO LCvx guidance") from exc

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
    solver = qoco.QOCO()
    solver.setup(
        d["n"],
        d["m"],
        d["p"],
        d["P"],
        d["c"],
        d["A"],
        d["b"],
        d["G"],
        d["h"],
        d["l"],
        len(d["q"]),
        np.asarray(d["q"], dtype=np.int32),
    )
    result = solver.solve()
    if result.status not in ("QOCO_SOLVED", "QOCO_SOLVED_INACCURATE"):
        return {"status": result.status, "feasible": False}

    out = _unpack_solution(np.asarray(result.x), d)
    out["status"] = "optimal" if result.status == "QOCO_SOLVED" else "optimal_inaccurate"
    out["qoco_status"] = result.status
    out["iters"] = int(result.iters)
    out["solve_time_sec"] = float(result.solve_time_sec)
    return _postcheck_solution(
        out,
        d["rho_min_ach"],
        d["rho_max_ach"],
        d["terminal_slack"],
        d["terminal_vz_max"],
    )

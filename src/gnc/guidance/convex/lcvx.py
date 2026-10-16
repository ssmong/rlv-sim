"""Lossless-convexification powered-descent solvers."""

from __future__ import annotations

from time import perf_counter

import numpy as np

from gnc.guidance.convex.qoco_direct import _postcheck_solution
from gnc.guidance.convex.solvers import OPTIMAL_STATUSES, solve_problem

try:  # cvxpy is an optional dependency (numpy-side guidance only)
    import cvxpy as cp

    _HAS_CVXPY = True
except Exception:  # pragma: no cover - import guard
    _HAS_CVXPY = False


# fmt: off
def solve_pdg(
    r0, v0, m0,
    g, alpha, rho_min, rho_max, m_dry,
    tf, N,
    target=(0.0, 0.0, 0.0), theta_max=None, gamma_gs=None,
    solver="QOCO",
    p_sea=0.0, nozzle_exit_area=0.0, scale_height=None, n_engines=1, p_amb_alt=None,
    terminal_slack=False, terminal_pos_weight=1.0e4, terminal_vel_weight=1.0e4,
    terminal_vz_max=None, terminal_upright=False,
    pos_scale=None, vel_scale=None, accel_scale=None,
    slew_rate_max=None, thrust_rate_max=None,
    drag_seq=None,
):
# fmt: on
    """Solve one fixed-time LCvx SOCP and return a feasibility/trajectory dict."""
    solver_name = str(getattr(solver, "name", solver)).upper()
    if solver_name in ("QOCO", "CLARABEL"):
        if solver_name == "QOCO":
            from gnc.guidance.convex.qoco_direct import solve_pdg_qoco as solve_direct
        else:
            from gnc.guidance.convex.clarabel_direct import (
                solve_pdg_clarabel as solve_direct,
            )

        # fmt: off
        return solve_direct(
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
    if not _HAS_CVXPY:
        raise ImportError("cvxpy is required for non-QOCO LCvx guidance backends")
    r0 = np.asarray(r0, float)
    v0 = np.asarray(v0, float)
    g = np.asarray(g, float)
    target = np.asarray(target, float)
    dt = tf / N

    # Use one ambient pressure per solve. With p_sea=0 this is the vacuum formulation.
    h_rep = (-float(r0[2])) if p_amb_alt is None else float(p_amb_alt)
    if scale_height is None or p_sea <= 0.0:
        p_amb_const = float(p_sea)  # scale_height absent: treat p_sea as constant ambient.
    else:
        p_amb_const = float(p_sea) * np.exp(-max(h_rep, 0.0) / float(scale_height))
    bp_force = float(n_engines) * p_amb_const * float(nozzle_exit_area)
    # Back-pressure shifts both net-thrust bounds down by the same force.
    rho_max_ach = rho_max - bp_force
    if rho_max_ach <= 0.0:
        raise ValueError(
            f"back-pressure force {bp_force:.0f} N >= max vacuum thrust {rho_max:.0f} N: "
            "no positive net thrust is achievable"
        )
    # A throttle floor at/below the back-pressure debit is vacuous: drop the lower-bound
    # constraint (sigma >= 0 is implied) instead of clamping it to a tiny ill-conditioned value.
    rho_min_ach = rho_min - bp_force
    has_rho_min = rho_min_ach > 1e-6 * rho_max

    # Taylor reference: full-throttle mass profile with the back-pressure fuel flow included.
    ref_flow = alpha * (rho_max_ach + bp_force) * dt
    m_ref = np.maximum(m0 - ref_flow * np.arange(N), 1e-3)
    z0_seq = np.log(m_ref)

    r = cp.Variable((N + 1, 3))
    v = cp.Variable((N + 1, 3))
    z = cp.Variable(N + 1)
    u = cp.Variable((N, 3))
    sigma = cp.Variable(N)
    e_r = cp.Variable(3, nonneg=True) if terminal_slack else None
    e_v = cp.Variable(3, nonneg=True) if terminal_slack else None
    s_r = cp.Variable(nonneg=True) if terminal_slack else None
    s_v = cp.Variable(nonneg=True) if terminal_slack else None

    cons = [r[0] == r0, v[0] == v0, z[0] == np.log(m0)]
    # DESCENDO frozen-coefficient drag template (Simplicio 2019 eqs. 20-21): d*_k from the
    # PREVIOUS solution multiplies the CURRENT velocity variable (linear in v, stays convex).
    d_seq = np.zeros(N) if drag_seq is None else np.asarray(drag_seq, float)
    for k in range(N):
        # ZOH-exact double integrator (+ frozen drag coefficient); Euler mass update.
        cons += [v[k + 1] == v[k] + dt * (g + u[k] - d_seq[k] * v[k])]
        cons += [r[k + 1] == r[k] + dt * v[k] + 0.5 * dt**2 * (g + u[k] - d_seq[k] * v[k])]
        z0 = z0_seq[k]
        dz = z[k] - z0
        # Linearized back-pressure fuel term; shares the thrust-bound z reference.
        bp_flow_k = alpha * bp_force * np.exp(-z0) * (1.0 - dz)
        cons += [z[k + 1] == z[k] - alpha * dt * sigma[k] - dt * bp_flow_k]
        cons += [cp.norm(u[k]) <= sigma[k]]
        mu2 = rho_max_ach * np.exp(-z0)
        if has_rho_min:
            mu1 = rho_min_ach * np.exp(-z0)
            cons += [sigma[k] >= mu1 * (1.0 - dz + 0.5 * cp.square(dz))]
        cons += [sigma[k] <= mu2 * (1.0 - dz)]
        if theta_max is not None:
            cons += [-u[k][2] >= np.cos(theta_max) * sigma[k]]
        if gamma_gs is not None:
            # Glideslope cone about the TARGET (same frame as the direct builder).
            dr = r[k] - target
            cons += [-dr[2] >= np.tan(gamma_gs) * cp.norm(dr[:2])]

    if thrust_rate_max is not None:
        # Planned thrust-magnitude rate: |sigma[k+1]-sigma[k]| <= Tdot_max*dt*e^{-z_k}, with the
        # first-order (1-dz) mass correction (conservative tangent side, same as the thrust
        # bounds) — linear, scalar (direction-free); G-FOLD Eqs. (6c)-(6d) / DESCENDO eq. (19).
        # See qoco_direct (identical rows).
        for k in range(N - 1):
            b_k = float(thrust_rate_max) * dt * np.exp(-z0_seq[k])
            cap = b_k * (1.0 - (z[k] - z0_seq[k]))
            cons += [sigma[k + 1] - sigma[k] <= cap, sigma[k] - sigma[k + 1] <= cap]

    if slew_rate_max is not None:
        # Trackable thrust profile: ||u_{k+1}-u_k|| <= rate*dt (SOC). Inter-node coupling sits
        # OUTSIDE the CT-2013 / DT-LCvx-2025 lossless guarantees (pointwise control-set arguments)
        # -> gate activation on slack_gap and min-throttle violations.
        du_max = float(slew_rate_max) * dt
        cons += [cp.norm(u[k + 1] - u[k]) <= du_max for k in range(N - 1)]

    if terminal_slack:
        cons += [r[N] - target <= e_r, target - r[N] <= e_r]
        cons += [v[N] <= e_v, -v[N] <= e_v]
        cons += [cp.norm(r[N] - target) <= s_r, cp.norm(v[N]) <= s_v]
    elif terminal_vz_max is None:
        cons += [r[N] == target, v[N] == np.zeros(3)]
    else:
        # Terminal sink window (DESCENDO SOCP 1): land DESCENDING into the cutoff box. An exact
        # v[N]=0 endpoint demands a terminal hover that the throttle floor cannot hold (TWR>1).
        cons += [r[N] == target, v[N][:2] == np.zeros(2)]
        cons += [v[N][2] >= 0.0, v[N][2] <= float(terminal_vz_max)]
    if terminal_upright:
        # Terminal upright thrust (DESCENDO boundary w_xy[N]=0): the final control node may not
        # tilt — touch down thrusting straight up (w_z >= 0 is implied by the pointing cone).
        cons += [u[N - 1][:2] == np.zeros(2)]
    cons += [z[N] >= np.log(m_dry)]
    prob = cp.Problem(cp.Minimize(cp.sum(sigma) * dt), cons)
    if terminal_slack:
        obj = cp.sum(sigma) * dt
        obj += float(terminal_pos_weight) * (cp.sum(e_r) + s_r)
        obj += float(terminal_vel_weight) * (cp.sum(e_v) + s_v)
        prob = cp.Problem(cp.Minimize(obj), cons)
    try:
        solve_problem(prob, solver)
    except Exception as e:  # pragma: no cover - solver failure path
        return {"status": f"solver_error:{e}", "feasible": False}

    if prob.status not in OPTIMAL_STATUSES or u.value is None:
        return {"status": prob.status, "feasible": False}

    m_val = np.exp(z.value)
    thrust = m_val[:N, None] * u.value
    terminal_pos_slack = float(np.linalg.norm(r.value[-1] - target))
    terminal_vel_slack = float(np.linalg.norm(v.value[-1]))
    terminal_pos_l1_slack = float(np.sum(np.abs(r.value[-1] - target)))
    terminal_vel_l1_slack = float(np.sum(np.abs(v.value[-1])))
    terminal_pos_l2_epigraph = (
        float(s_r.value) if terminal_slack and s_r.value is not None else terminal_pos_slack
    )
    terminal_vel_l2_epigraph = (
        float(s_v.value) if terminal_slack and s_v.value is not None else terminal_vel_slack
    )
    out = {
        "status": prob.status,
        "solver_feasible": True,
        "tf": tf,
        "r": r.value,
        "v": v.value,
        "m": m_val,
        "u": u.value,
        "sigma": sigma.value,
        "thrust": thrust,
        "fuel": float(m0 - m_val[-1]),
        "slack_gap": float(np.max(np.abs(np.linalg.norm(u.value, axis=1) - sigma.value))),
        "terminal_slack": bool(terminal_slack),
        "terminal_pos_slack": terminal_pos_slack,
        "terminal_vel_slack": terminal_vel_slack,
        "terminal_pos_l1_slack": terminal_pos_l1_slack,
        "terminal_vel_l1_slack": terminal_vel_l1_slack,
        "terminal_pos_l2_epigraph": terminal_pos_l2_epigraph,
        "terminal_vel_l2_epigraph": terminal_vel_l2_epigraph,
    }
    return _postcheck_solution(
        out,
        max(float(rho_min_ach), 0.0),
        float(rho_max_ach),
        bool(terminal_slack),
        terminal_vz_max,
    )


# fmt: off
def min_fuel_tf(
    r0, v0, m0,
    g, alpha, rho_min, rho_max, m_dry,
    N,
    tf_lo=None, tf_hi=None,
    iters=14, grid_fallback=True,
    drag_seq_fn=None,
    attempt_callback=None,
    **kw,
):
# fmt: on
    """Search time-of-flight for the pre-ignition free-time LCvx solve.

    ``drag_seq_fn(tf, N) -> (N,) array | None`` resamples the DESCENDO drag template on each
    candidate grid (the drag-aware feasible-tf window is NOT centred on the drag-free tf*)."""
    h0 = -float(r0[2])  # altitude
    vdown = max(float(v0[2]), 1.0)
    if tf_lo is None:
        tf_lo = max(0.5 * h0 / max(vdown, 1.0), 1.0)
    if tf_hi is None:
        tf_hi = max(3.0 * h0 / max(vdown, 1.0), tf_lo + 2.0)

    def attempt(tf):
        kw2 = kw if drag_seq_fn is None else {**kw, "drag_seq": drag_seq_fn(tf, N)}
        started = perf_counter()
        s = solve_pdg(r0, v0, m0, g, alpha, rho_min, rho_max, m_dry, tf, N, **kw2)
        if attempt_callback is not None:
            attempt_callback(
                s,
                {"kind": "free_time", "tf": float(tf), "N": int(N),
                 "wall_time_sec": perf_counter() - started},
            )
        return s

    def cost(tf):
        s = attempt(tf)
        return (s["fuel"] if s["feasible"] else np.inf), s

    gr = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = tf_lo, tf_hi
    c, d = b - gr * (b - a), a + gr * (b - a)
    fc, sc = cost(c)
    fd, sd = cost(d)
    best = sc if fc <= fd else sd
    for _ in range(iters):
        if fc <= fd:
            b, d, fd, sd = d, c, fc, sc
            c = b - gr * (b - a)
            fc, sc = cost(c)
            if fc < (best.get("fuel", np.inf) if best.get("feasible") else np.inf):
                best = sc
        else:
            a, c, fc, sc = c, d, fd, sd
            d = a + gr * (b - a)
            fd, sd = cost(d)
            if fd < (best.get("fuel", np.inf) if best.get("feasible") else np.inf):
                best = sd
    best["grid_fallback_used"] = False
    if grid_fallback and not best.get("feasible"):
        # last resort: scan a coarse grid (golden section can miss a narrow feasible band).
        # Flagged in the result so the guidance solve_log records that the fallback fired
        # (no silent fallbacks).
        for tf in np.linspace(tf_lo, tf_hi, 12):
            s = attempt(tf)
            if s["feasible"]:
                s["grid_fallback_used"] = True
                return s
    return best


# fmt: off
def min_fuel_fixed_dt(
    r0, v0, m0,
    g, alpha, rho_min, rho_max, m_dry,
    dt,
    n_cap=120, grid_fallback=True,
    drag_seq_fn=None,
    attempt_callback=None,
    **kw,
):
# fmt: on
    """Fixed-dt integer line search over N for the pre-ignition solve (AP-2007 Algorithm 1).

    tf = N*dt with N in [N_min, N_max] from the closed-form bounds (AP-2007 eq. (55)-(57)):
    t_min = m_dry*||v0||/rho2_ach, t_max = m_fuel/(alpha*rho1_ach); the fuel cost is unimodal
    in the time of flight over the feasible band (AP-2007 Remark 10).
    """
    r0 = np.asarray(r0, float)
    v0 = np.asarray(v0, float)
    dt = float(dt)
    # Achievable net-thrust band at the representative altitude for the closed-form N bounds.
    p_sea = float(kw.get("p_sea", 0.0))
    a_e = float(kw.get("nozzle_exit_area", 0.0))
    scale_height = kw.get("scale_height")
    n_eng = float(kw.get("n_engines", 1))
    h0 = (-float(r0[2])) if kw.get("p_amb_alt") is None else float(kw["p_amb_alt"])
    if scale_height is None or p_sea <= 0.0:
        p_amb = p_sea
    else:
        p_amb = p_sea * np.exp(-max(h0, 0.0) / float(scale_height))
    bp = n_eng * p_amb * a_e
    rho2_ach = max(rho_max - bp, 1e-3)
    rho1_ach = max(rho_min - bp, 0.0)
    m_fuel = max(float(m0) - float(m_dry), 0.0)
    t_min = float(m_dry) * float(np.linalg.norm(v0)) / rho2_ach
    t_max = m_fuel / (alpha * rho1_ach) if rho1_ach > 0.0 else n_cap * dt
    n_min = max(int(t_min / dt) + 1, 2)
    n_max = min(int(t_max / dt), int(n_cap))
    if n_max < n_min:
        return {"status": "empty_tf_band", "feasible": False, "grid_fallback_used": False}

    cache: dict[int, dict] = {}

    def cost(n: int) -> float:
        if n not in cache:
            kw2 = kw if drag_seq_fn is None else {**kw, "drag_seq": drag_seq_fn(n * dt, n)}
            started = perf_counter()
            result = solve_pdg(
                r0, v0, m0, g, alpha, rho_min, rho_max, m_dry, n * dt, n, **kw2
            )
            if attempt_callback is not None:
                attempt_callback(
                    result,
                    {"kind": "free_time", "tf": float(n * dt), "N": int(n),
                     "wall_time_sec": perf_counter() - started},
                )
            cache[n] = result
        s = cache[n]
        return float(s["fuel"]) if s.get("feasible") else np.inf

    # Integer ternary search (unimodal over the feasible band; inf plateaus handled by fallback).
    lo, hi = n_min, n_max
    while hi - lo > 2:
        m1 = lo + (hi - lo) // 3
        m2 = hi - (hi - lo) // 3
        c1, c2 = cost(m1), cost(m2)
        if not np.isfinite(c1) and not np.isfinite(c2):
            break  # blind on an infeasible plateau: hand over to the grid fallback
        if c1 <= c2:
            hi = m2
        else:
            lo = m1
    best_n = min(range(lo, hi + 1), key=cost)
    used_fallback = False
    if not np.isfinite(cost(best_n)) and grid_fallback:
        # Coarse scan for the feasible band, then refine around the first hit (no silent fallback).
        stride = max((n_max - n_min) // 24, 1)
        for n in range(n_min, n_max + 1, stride):
            if np.isfinite(cost(n)):
                lo_r = max(n - stride, n_min)
                hi_r = min(n + stride, n_max)
                best_n = min(range(lo_r, hi_r + 1), key=cost)
                used_fallback = True
                break
    best = cache.get(best_n)
    if best is None or not best.get("feasible"):
        return {"status": "infeasible", "feasible": False, "grid_fallback_used": used_fallback}
    best = dict(best)
    best["grid_fallback_used"] = used_fallback
    return best

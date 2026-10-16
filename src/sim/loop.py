"""Closed-loop rollout: native C++ plant <-> Python GNC, in process.

GNC ticks at ``ctrl_dt``. Its command is held ZOH while one native session call advances
``n_sub = ctrl_dt / plant_dt`` RK4 substeps; parameters, internal state, wind filter, and RNG stay
in C++ across calls. Optional transport delay is a control-tick command ring buffer.
"""

from __future__ import annotations

import sys
import time
from collections import deque
from types import SimpleNamespace

import numpy as np

from core.frames import dcm_from_quat
from core.interfaces import EpisodeResult, TerminationReason
from core.seeding import rng_from_seed
from plant.dynamics import termination_reason


class _Progress:
    """Single-line live progress on stderr: sim-time bar + state + wall-clock ETA.

    Auto-disabled when stderr is not a TTY (pipes, background tasks, CI stay clean)."""

    _WIDTH = 24

    def __init__(self, t_max: float, enabled: bool):
        self.t_max = float(t_max)
        self.enabled = bool(enabled) and sys.stderr.isatty()
        self.t0 = time.perf_counter()
        self.last = 0.0

    def update(self, t: float, state) -> None:
        if not self.enabled:
            return
        now = time.perf_counter()
        if now - self.last < 0.1:  # ~10 Hz refresh cap
            return
        self.last = now
        frac = min(max(t, 0.0) / self.t_max, 1.0)
        wall = now - self.t0
        eta = wall * (1.0 - frac) / max(frac, 1e-9)
        r = np.asarray(state.r_ned)
        v = np.asarray(state.v_ned)
        fill = int(round(self._WIDTH * frac))
        bar = "█" * fill + "░" * (self._WIDTH - fill)
        sys.stderr.write(
            f"\r{bar} t={t:6.1f}/{self.t_max:.0f} s  alt={-float(r[2]):8.1f} m  "
            f"vz={float(v[2]):7.1f} m/s  ETA {eta:5.0f} s "
        )
        sys.stderr.flush()

    def close(self) -> None:
        if self.enabled:
            sys.stderr.write("\r" + " " * 100 + "\r")
            sys.stderr.flush()


def make_native_block_step(native_plant, state0, n_sub: int):
    """Create a control-tick plant block with no per-substep Python/C++ round-trip.

    Immutable parameters already live in ``native_plant``. ``reset`` copies the full initial state
    once; each call then transfers only the held actuator command into C++, advances ``n_sub`` RK4
    steps (including RNG/wind), and returns the public tick-end state needed by GNC/logging plus the
    last diagnostics. Internal servo rates, slosh, wind filters, and RNG never leave C++.
    """
    session = native_plant.create_session(state0)

    def block(_state, cmd):
        snapshot, aux = session.advance(cmd, n_sub)
        return SimpleNamespace(**snapshot), aux

    return block


def _tilt_from_vertical(q_bn) -> float:
    """Angle [rad] of body-x (nose) from NED up (-z)."""
    bx = np.asarray(dcm_from_quat(q_bn))[:, 0]  # body-x in NED
    return float(np.arccos(np.clip(-bx[2], -1.0, 1.0)))


def _slerp(q0, q1, s):
    """Normalized-lerp of two adjacent log quaternions (with antipodal-sign correction). The two
    samples are one log step apart so the rotation between them is tiny — nlerp is exact enough for
    the touchdown ground-crossing interpolation and avoids a full slerp's trig."""
    q0, q1 = np.asarray(q0), np.asarray(q1)
    if float(np.dot(q0, q1)) < 0.0:
        q1 = -q1  # take the short way (q and -q are the same rotation)
    q = q0 + s * (q1 - q0)
    return q / (np.linalg.norm(q) + 1e-12)


def _log_step(log, aux_log, t_next, state, phase, aux):
    log["t"].append(t_next)
    log["r_ned"].append(np.asarray(state.r_ned))
    log["v_ned"].append(np.asarray(state.v_ned))
    log["q_bn"].append(np.asarray(state.q_bn))
    log["w_body"].append(np.asarray(state.w_body))
    log["m"].append(float(state.m))
    log["phase"].append(int(phase))
    # actuator truth (for replay/visualization; cheap to log alongside the pose)
    log["throttle_act"].append(np.asarray(state.throttle_act))
    log["tvc"].append(np.asarray(state.tvc))
    log["fin"].append(np.asarray(state.fin))
    aux_log.append({key: np.asarray(value) for key, value in aux.items()})


# Structural-survival contact limits: reaching the ground within these counts as a
# (possibly imperfect) LANDING; exceeding any of them is a CRASH. Distinct from the tighter
# soft-landing SUCCESS tolerances. Grounded in legged-lander limits: F9-class legs tolerate a few
# m/s sink and a few degrees of tilt; > ~6 m/s sink / large tilt / spin tips or collapses it.
_CRASH_SINK = 6.0  # vertical sink rate [m/s]
_CRASH_HORIZ = 4.0  # horizontal velocity [m/s] (tip-over on contact)
_CRASH_TILT_DEG = 15.0  # attitude tilt from vertical [deg]
_CRASH_RATE = 0.35  # body angular rate [rad/s] (~20 deg/s; tumbling arrival)

# Genuine SOFT-landing limits (the upper end of a real set-down): F9/Apollo-LM nominal touchdown is
# ~1–2 m/s sink, <1 m/s lateral, near-vertical, near-zero spin. A contact within the CRASH limits
# but ABOVE these is 'hard' — structurally survivable (legs absorb it) but NOT soft. (2022 IJASS
# cites 5 m/s as the *allowable* bound; this is the *good*-landing bar, deliberately tighter.)
_SOFT_SINK = 2.0  # vertical sink rate [m/s]
_SOFT_HORIZ = 1.0  # horizontal velocity [m/s]
_SOFT_TILT_DEG = 5.0  # attitude tilt from vertical [deg]
_SOFT_RATE = 0.1  # body angular rate [rad/s] (~6 deg/s)


def _finalize(log, aux_log, final_state, bundle, hit_max, pos_tol, speed_tol, tilt_tol_deg):
    """Assemble an EpisodeResult + KPIs from the logged trajectory (shared by both rate paths)."""
    t_arr = np.asarray(log["t"])
    r = np.stack(log["r_ned"])
    v = np.stack(log["v_ned"])
    q = np.stack(log["q_bn"])
    w = np.stack(log["w_body"])
    m = np.asarray(log["m"])
    phase = np.asarray(log["phase"])
    throttle_act = np.stack(log["throttle_act"])
    tvc = np.stack(log["tvc"])
    fin_act = np.stack(log["fin"])

    # touchdown index = first altitude <= 0, else last
    alt = -r[:, 2]
    below = np.where(alt <= 0.0)[0]
    td = int(below[0]) if below.size else len(alt) - 1

    # interpolate the state to the EXACT ground crossing alt=0: the logged ``td`` is
    # the first SUB-ground sample, so reading KPIs there biases them; lerp between the last-above
    # and first-below samples. (No crossing, or td==0 => use td as-is.)
    if 0 < td < len(alt) and alt[td] < 0.0 <= alt[td - 1]:
        s = float(alt[td - 1] / (alt[td - 1] - alt[td]))  # in [0,1]
        r_td = r[td - 1] + s * (r[td] - r[td - 1])
        v_td = v[td - 1] + s * (v[td] - v[td - 1])
        w_td = w[td - 1] + s * (w[td] - w[td - 1])
        m_td = float(m[td - 1] + s * (m[td] - m[td - 1]))
        q_td = _slerp(q[td - 1], q[td], s)
    else:
        r_td, v_td, w_td, m_td, q_td = r[td], v[td], w[td], float(m[td]), q[td]

    landing_position_error = float(np.linalg.norm(r_td[:2]))
    landing_speed = float(np.linalg.norm(v_td))
    landing_speed_vertical = float(abs(v_td[2]))  # sink rate
    landing_speed_horizontal = float(np.linalg.norm(v_td[:2]))  # lateral
    landing_rate = float(np.linalg.norm(w_td))  # spin at contact
    landing_tilt = _tilt_from_vertical(q_td)
    fuel_margin = m_td - float(bundle.params.mass.m_dry)
    reason = termination_reason(final_state, bundle.params, hit_max)

    # CRASH reclassification: if the predicate says we reached the ground (TOUCHDOWN) but any
    # structural-survival limit is exceeded, it is a CRASH, not a landing — a 15 m/s slam must not
    # be recorded identically to a 0.5 m/s set-down.
    if reason == int(TerminationReason.TOUCHDOWN) and (
        landing_speed_vertical > _CRASH_SINK
        or landing_speed_horizontal > _CRASH_HORIZ
        or landing_tilt > np.deg2rad(_CRASH_TILT_DEG)
        or landing_rate > _CRASH_RATE
    ):
        reason = int(TerminationReason.CRASH)

    # 3-tier contact quality: soft (genuine set-down) / hard (survivable but firm) /
    # crash (>structural). None if the run never made a controlled contact (fuel-out/timeout/etc.).
    if reason == int(TerminationReason.CRASH):
        landing_class = "crash"
    elif reason == int(TerminationReason.TOUCHDOWN):
        landing_class = (
            "soft"
            if (
                landing_speed_vertical <= _SOFT_SINK
                and landing_speed_horizontal <= _SOFT_HORIZ
                and landing_tilt <= np.deg2rad(_SOFT_TILT_DEG)
                and landing_rate <= _SOFT_RATE
            )
            else "hard"
        )
    else:
        landing_class = None

    violations = {
        "position": bool(landing_position_error > pos_tol),
        "speed": bool(landing_speed > speed_tol),
        "tilt": bool(landing_tilt > np.deg2rad(tilt_tol_deg)),
        "rate": bool(landing_rate > _CRASH_RATE),  # tumbling-but-upright is not a success
        "fuel": bool(fuel_margin <= 0.0),
    }
    # SUCCESS requires a genuine SOFT landing (sink≤2 / lateral≤1 m/s, etc. — landing_class) ON
    # the pad (position) with fuel remaining. A 'hard' set-down (survivable but firm, 2–6 m/s) is
    # not a success.
    success = landing_class == "soft" and not any(violations.values())

    return EpisodeResult(
        t=t_arr,
        r_ned=r,
        v_ned=v,
        q_bn=q,
        w_body=w,
        m=m,
        phase=phase,
        aux={"per_tick": aux_log},
        termination_reason=reason,
        touchdown_index=td,
        landing_position_error=landing_position_error,
        landing_speed=landing_speed,
        landing_tilt=landing_tilt,
        fuel_margin=fuel_margin,
        success=success,
        violation_flags=violations,
        throttle_act=throttle_act,
        tvc=tvc,
        fin=fin_act,
        landing_speed_vertical=landing_speed_vertical,
        landing_speed_horizontal=landing_speed_horizontal,
        landing_rate=landing_rate,
        landing_class=landing_class,
    )


def _find_mem_key(mem, key):
    """Recursively find the first ``key`` value in the (nested) GNC memory."""
    if isinstance(mem, dict):
        if mem.get(key) is not None:
            return mem[key]
        for v in mem.values():
            if isinstance(v, (dict, list)):
                found = _find_mem_key(v, key)
                if found is not None:
                    return found
    elif isinstance(mem, list):
        for v in mem:
            found = _find_mem_key(v, key)
            if found is not None:
                return found
    return None


def _collect_mem_lists(mem, key):
    """Recursively gather every list stored under ``key`` in the (nested) GNC memory and concat it.

    Used for guidance ``solve_log`` records and ``control_timing`` samples."""
    out: list = []
    if isinstance(mem, dict):
        for k, v in mem.items():
            if k == key and isinstance(v, list):
                out.extend(v)
            elif isinstance(v, (dict, list)):
                out.extend(_collect_mem_lists(v, key))
    elif isinstance(mem, list):
        for v in mem:
            out.extend(_collect_mem_lists(v, key))
    return out


def _percentiles(a) -> dict:
    """mean/std/p50/p95/p99/max(=WCET)/min of a 1-D sample array."""
    return {
        "mean": float(a.mean()),
        "std": float(a.std()),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "max": float(a.max()),  # WCET
        "min": float(a.min()),
    }


def _solver_stats(logs):
    """Aggregate per-solve guidance records into timing/feasibility stats (RTI/WCET relevant)."""
    solves = [record for record in logs if record.get("record_type", "solve") == "solve"]
    outcomes = [record for record in logs if record.get("record_type") == "outcome"]
    stats = {
        "n_solves": len(solves),
        "n_ignition": int(sum(r["kind"].startswith("ignite") for r in solves)),
        "n_resolve": int(sum(r["kind"].startswith("resolve") for r in solves)),
        "n_reanchor": int(sum(r["kind"].startswith("reanchor") for r in solves)),
        "n_solver_infeasible": int(
            sum(not r.get("solver_feasible", r.get("feasible", False)) for r in solves)
        ),
        "n_outcomes": len(outcomes),
        "n_failed_outcomes": int(
            sum(not r.get("mission_feasible", r["feasible"]) for r in outcomes)
        ),
        "n_postcheck_rejected_solves": int(
            sum(record.get("mission_feasible") is False for record in solves)
        ),
        "n_grid_fallback": int(sum(r.get("grid_fallback", False) for r in logs)),
    }
    if solves:
        # Per-attempt wall includes model stuffing, dispatch, and solve. solve_ms is the backend's
        # own timer and its exact scope is backend-dependent.
        wall_ms = [r["wall_ms"] for r in solves if r.get("wall_ms") is not None]
        solve_ms = [r["solve_ms"] for r in solves if r.get("solve_ms") is not None]
        stats["wall_ms"] = _percentiles(np.asarray(wall_ms, float)) if wall_ms else None
        stats["solve_ms"] = _percentiles(np.asarray(solve_ms, float)) if solve_ms else None
        stats["n_backend_timed"] = len(solve_ms)
        stats["iters"] = _percentiles(np.array([r["iters"] for r in solves], float))
    stats["per_solve"] = solves
    stats["outcomes"] = outcomes
    return stats


def _guidance_degradation(mem, solve_logs):
    """Expose guidance degraded-mode counters that otherwise die with controller memory."""
    keys = (
        "replan_failures",
        "cached_plan_fallbacks",
        "reanchor_count",
        "reanchor_time_added",
        "terminal_slack_uses",
        "terminal_pos_slack",
        "terminal_vel_slack",
        "template_refinement_failures",
        "plan_exhausted_ticks",
        "proj_downthrust_clamps",
        "last_replan_failed",
    )
    out = {key: value for key in keys if (value := _find_mem_key(mem, key)) is not None}
    out["grid_fallbacks"] = int(sum(log.get("grid_fallback", False) for log in solve_logs))
    return out


def _attach_gnc_aux(result, gnc_mem):
    """Attach final GNC diagnostics identically for single- and multi-rate runners."""
    solve_logs = _collect_mem_lists(gnc_mem, "solve_log")
    result.aux["solver"] = _solver_stats(solve_logs)
    result.aux["guidance_degradation"] = _guidance_degradation(gnc_mem, solve_logs)
    result.aux["control"] = _control_stats(_collect_mem_lists(gnc_mem, "control_timing"))
    result.aux["plan_ignition"] = _find_mem_key(gnc_mem, "plan_ignition")
    result.aux["t_ignition"] = _find_mem_key(gnc_mem, "t_ignition")


def _control_stats(samples):
    """Per-call tracking-controller compute time [ms] — the control LAW only (allocation excluded).

    Controller-agnostic: timed at the stack layer, so any plugged-in controller (PD or otherwise) is
    measured. The first call is logged separately as warm-up (first-touch numpy / tracing). min/p50
    approximate the true compute (jitter only ADDS time); max/p95/p99 are the WCET/jitter real-time
    analysis cares about. This is pure Python wall time — a C-ported controller exposing its own
    solver timer can additionally report that (cf. the solver's solve_ms vs wall_ms split)."""
    stats = {"n_calls": len(samples)}
    if samples:
        stats["warmup_ms"] = float(samples[0])  # first call: first-touch numpy / tracing
    if len(samples) > 1:
        stats["wall_ms"] = _percentiles(np.array(samples[1:], float))  # warm-up call dropped
    return stats


def run_closed_loop(
    bundle,
    state0,
    controller,
    nav,
    t_max: float,
    ctrl_dt: float,
    plant_dt: float,
    success_pos_tol: float = 10.0,
    success_speed_tol: float = 5.0,
    success_tilt_tol_deg: float = 15.0,
    delay_ticks: int = 0,
    nav_dt: float | None = None,
    guidance_dt: float | None = None,
    log_dt: float | None = None,
    progress: bool = False,
) -> EpisodeResult:
    """Run a closed-loop episode and return an EpisodeResult with KPIs.

    Single-rate by default (everything ticks at ``ctrl_dt``). Passing ``nav_dt``/``guidance_dt``
    distinct from ``ctrl_dt`` activates the four-rate schedule (plant_dt < nav_dt ≤ ctrl_dt ≤
    guidance_dt): nav/guidance/control each fire on their own stride and are held ZOH in between,
    while the plant still advances in native blocks. ``log_dt`` downsamples the stored trajectory
    (the terminal step is always logged). Requires a controller exposing ``guide`` /
    ``control_allocate`` (GNCStack); otherwise the single-rate path is used.
    """
    tol = (success_pos_tol, success_speed_tol, success_tilt_tol_deg)
    multirate = (nav_dt is not None or guidance_dt is not None) and hasattr(controller, "guide")
    if multirate:
        return _run_multirate(
            bundle,
            state0,
            controller,
            nav,
            t_max,
            ctrl_dt,
            plant_dt,
            nav_dt or ctrl_dt,
            guidance_dt or ctrl_dt,
            log_dt,
            delay_ticks,
            tol,
            progress,
        )

    n_sub = int(round(ctrl_dt / plant_dt))
    n_ticks = int(round(t_max / ctrl_dt))
    block_step = make_native_block_step(bundle.native, state0, n_sub)

    state = state0
    nav_mem = {}
    gnc_mem = controller.initial_memory() if hasattr(controller, "initial_memory") else {}
    cmd_buffer: deque = deque(maxlen=delay_ticks + 1)

    log = {
        k: []
        for k in (
            "t",
            "r_ned",
            "v_ned",
            "q_bn",
            "w_body",
            "m",
            "phase",
            "throttle_act",
            "tvc",
            "fin",
        )
    }
    aux_log = []
    nav_rng = rng_from_seed(state.key)
    hit_max = True
    prog = _Progress(t_max, progress)

    for tick in range(n_ticks):
        t = tick * ctrl_dt
        meas = nav.measure(state, int(nav_rng.bit_generator.random_raw()))
        x_hat, nav_mem = nav.estimate(meas, None, nav_mem)
        act, gnc_mem, phase = controller(x_hat, t, gnc_mem)

        cmd_buffer.append(act)
        delayed = cmd_buffer[0]  # oldest within delay window (== act when delay_ticks=0)

        state, aux = block_step(state, delayed)
        _log_step(log, aux_log, t + ctrl_dt, state, phase, aux)
        prog.update(t + ctrl_dt, state)

        if bool(state.done):
            hit_max = False
            break

    prog.close()
    result = _finalize(log, aux_log, state, bundle, hit_max, *tol)
    _attach_gnc_aux(result, gnc_mem)
    return result


def _run_multirate(
    bundle,
    state0,
    controller,
    nav,
    t_max,
    ctrl_dt,
    plant_dt,
    nav_dt,
    guidance_dt,
    log_dt,
    delay_ticks,
    tol,
    progress=False,
):
    """Four-rate schedule. The outer loop runs at the finest
    GNC rate; nav/guidance/control fire on integer strides of it and hold ZOH between firings; the
    plant advances one native block (the outer step) under the latest actuator command."""
    base_dt = min(ctrl_dt, nav_dt, guidance_dt)
    n_sub = max(1, int(round(base_dt / plant_dt)))
    n_outer = int(round(t_max / base_dt))
    nav_every = max(1, int(round(nav_dt / base_dt)))
    guid_every = max(1, int(round(guidance_dt / base_dt)))
    ctrl_every = max(1, int(round(ctrl_dt / base_dt)))
    log_every = max(1, int(round((log_dt or base_dt) / base_dt)))
    block_step = make_native_block_step(bundle.native, state0, n_sub)

    state = state0
    nav_mem = {}
    gnc_mem = controller.initial_memory()
    cmd_buffer: deque = deque(maxlen=delay_ticks + 1)
    x_hat = cmd = act = phase = None

    log = {
        k: []
        for k in (
            "t",
            "r_ned",
            "v_ned",
            "q_bn",
            "w_body",
            "m",
            "phase",
            "throttle_act",
            "tvc",
            "fin",
        )
    }
    aux_log = []
    nav_rng = rng_from_seed(state.key)
    hit_max = True
    prog = _Progress(t_max, progress)

    for it in range(n_outer):
        t = it * base_dt
        if it % nav_every == 0 or x_hat is None:
            nav_seed = int(nav_rng.bit_generator.random_raw())
            x_hat, nav_mem = nav.estimate(nav.measure(state, nav_seed), None, nav_mem)
        if it % guid_every == 0 or cmd is None:
            cmd, gnc_mem = controller.guide(x_hat, t, gnc_mem)
            phase = int(cmd.phase)
        if it % ctrl_every == 0 or act is None:
            # pass t so the FAST translational tracking feedback (guidance.track) runs at ctrl_dt on
            # the latest x_hat — the slow convex re-solve stays in guide() at guidance_dt.
            act, gnc_mem = controller.control_allocate(cmd, x_hat, gnc_mem, t)

        cmd_buffer.append(act)
        state, aux = block_step(state, cmd_buffer[0])

        done = bool(state.done)
        if it % log_every == 0 or done or it == n_outer - 1:
            _log_step(log, aux_log, t + base_dt, state, phase, aux)
        prog.update(t + base_dt, state)
        if done:
            hit_max = False
            break

    prog.close()
    result = _finalize(log, aux_log, state, bundle, hit_max, *tol)
    _attach_gnc_aux(result, gnc_mem)
    return result

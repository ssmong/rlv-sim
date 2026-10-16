"""Convex powered-descent guidance strategies.

LcvxGuidance wraps the lossless-convexification powered-descent SOCP. It coasts until
a continuous burn is feasible, latches ignition, then tracks the cached plan with feedforward
thrust plus position/velocity feedback. Optional replanning is explicit and config-driven; the
provided methods default to a single committed plan.
"""

from __future__ import annotations

import time

import numpy as np

from gnc.guidance.base import attitude_command, thrust_aligned_quat, thrust_command, upright_quat
from gnc.guidance.convex.lcvx import min_fuel_fixed_dt, min_fuel_tf, solve_pdg


class LcvxGuidance:
    """Powered-descent guidance via lossless convexification."""

    def __init__(
        self,
        nominal: dict,
        active_engines,
        phase: int,
        N: int,
        target,
        theta_max_deg: float | None,
        gamma_gs_deg: float | None,
        solver: str,
        tf_search_iters: int,
        grid_fallback: bool,
        tilt_max_deg: float,
        k_r: float,
        k_v: float,
        tf_floor: float,
        landing_cutoff_impact_mps: float,
        terminal_slack: bool,
        terminal_pos_weight: float,
        terminal_vel_weight: float,
        dt: float | None = None,
        n_cap: int = 120,
        slew_rate_max: float | None = None,
        thrust_rate_max_nps: float | None = None,
        terminal_vz_max_mps: float | None = None,
        terminal_upright: bool = False,
        tf_freeze_s: float = 0.0,
        reanchor_min_alt_m: float = 0.0,
        replan: bool = False,
    ):
        # Fixed node spacing (AP-2007 Algorithm 1): tf = N*dt with an integer-N line search at
        # ignition, and every anchored re-solve keeps the same dt. None selects fixed-N free-time
        # golden-section search.
        self.dt = None if dt is None else float(dt)
        self.n_cap = int(n_cap)
        self.slew_rate_max = None if slew_rate_max is None else float(slew_rate_max)
        # Planned inter-node thrust-magnitude rate bound (G-FOLD Eq. (1i) / DESCENDO eq. (19)).
        # This does not constrain node 0 against the current throttle state or model servo lag,
        # shutdown, restart, attitude response, or thrust-direction transients.
        self.thrust_rate_max = None if thrust_rate_max_nps is None else float(thrust_rate_max_nps)
        self.tf_search_iters = int(tf_search_iters)
        self.grid_fallback = bool(grid_fallback)
        self.k_r = float(k_r)
        self.k_v = float(k_v)
        self.tf_floor = float(tf_floor)
        self.landing_cutoff_impact_mps = float(landing_cutoff_impact_mps)
        self.terminal_slack = bool(terminal_slack)
        self.terminal_pos_weight = float(terminal_pos_weight)
        self.terminal_vel_weight = float(terminal_vel_weight)
        # Terminal sink window (DESCENDO SOCP 1, Simplicio 2019): plan ends DESCENDING into the
        # cutoff box (0 <= vz[N] <= vz_max) instead of at exact rest — with TWR_min>1 a v=0
        # endpoint demands a hover the throttle floor cannot hold.
        self.terminal_vz_max = None if terminal_vz_max_mps is None else float(terminal_vz_max_mps)
        self.terminal_upright = bool(terminal_upright)
        # Terminal solve freeze: below this tf_go stop re-solving and fly out the committed plan
        # (standard handoff: DESCENDO stale-plan interpolation / G-FOLD terminal phase).
        self.tf_freeze = max(float(tf_freeze_s), float(tf_floor))
        # Below this altitude re-anchoring is OFF: every re-anchor RESETS the countdown, so
        # low-altitude infeasibility cycles re-anchor -> over-brake -> hover and the freeze /
        # plan-exhaustion fly-out path never engages.
        self.reanchor_min_alt = float(reanchor_min_alt_m)
        self.replan = bool(replan)
        self.phase = int(phase)
        self.N = int(N)
        self.target = tuple(float(x) for x in target)
        self.theta = np.deg2rad(theta_max_deg) if theta_max_deg is not None else None
        self.gamma = np.deg2rad(gamma_gs_deg) if gamma_gs_deg is not None else None
        self.solver = solver

        self.active = np.asarray(active_engines, float)
        n_act = float(self.active.sum())
        self.rho_max = n_act * float(nominal["thrust_vac"])
        self.rho_min = float(nominal["throttle_min"]) * self.rho_max
        self.alpha = 1.0 / (float(nominal["isp"]) * float(nominal["g0"]))
        self.g = np.array([0.0, 0.0, float(nominal["g0"])])
        self.m_dry = float(nominal["m_dry"])
        self.q_up = upright_quat()

        self.p_sea = float(nominal["p_sea"])
        self.nozzle_exit_area = float(nominal["nozzle_exit_area"])
        self.scale_height = nominal["scale_height"]
        self.n_active = n_act
        # Drag-template inputs (DESCENDO eqs. 20-21): rho0=0 (vacuum) disables it cleanly.
        self.rho0 = float(nominal["rho0"])
        self.drag_cd_s = float(nominal["drag_cd_s"])
        # Achievable-set geometry: with the thrust-aligned attitude reference the body TILTS (up to
        # tilt_max) to direct sustained thrust; the gimbal is reserved for stabilization around that
        # reference. Plan pointing cone, tracking projection, and q_ref tilt cap share this angle
        # (constant cone — the standard 3-DOF treatment; ACB-2013).
        self.tilt_max = float(np.deg2rad(tilt_max_deg))
        # Fixed SOCP variable scales (gnc.convex_scale_*): consistent conditioning + slack penalty
        # across re-solves (per-solve state-derived scaling made QOCO mis-report feasible problems).
        self.scale_pos = None if nominal["scale_pos"] is None else float(nominal["scale_pos"])
        self.scale_vel = None if nominal["scale_vel"] is None else float(nominal["scale_vel"])
        self.scale_accel = None if nominal["scale_accel"] is None else float(nominal["scale_accel"])

    def initial_memory(self) -> dict:
        return {
            "ignited": False,
            "plan": None,
            "t_ignite": None,
            "t_ig0": None,
            "tf0": None,
            "frozen": False,
            "cutoff": False,
            "last_replan_failed": False,
            "replan_failures": 0,
            "cached_plan_fallbacks": 0,
            "reanchor_count": 0,  # event-triggered countdown re-commits (anchored window slipped)
            "reanchor_time_added": 0.0,  # cumulative signed tf0 slip absorbed by re-anchoring [s]
            "terminal_pos_slack": 0.0,
            "terminal_vel_slack": 0.0,
            "terminal_slack_uses": 0,
            "template_refinement_failures": 0,
            "plan_ignition": None,  # the plan committed at ignition latch (plan-vs-flown plots)
            "t_ignition": None,  # absolute ignition time of that plan [s]
            "solve_log": [],  # every attempted conic solve (timing/status/fallback telemetry)
            "plan_exhausted_ticks": 0,  # telemetry: tau >= tf while airborne (endgame reference)
            "proj_downthrust_clamps": 0,  # telemetry: projection clamped a down-thrust command
        }

    def _memory(self, memory: dict) -> dict:
        mem = dict(memory)
        for key, value in self.initial_memory().items():
            mem.setdefault(key, value)
        return mem

    def _bp_kw(self):
        """Back-pressure inputs for the convex solver; p_sea=0 recovers the vacuum plan."""
        return dict(
            p_sea=self.p_sea,
            nozzle_exit_area=self.nozzle_exit_area,
            scale_height=self.scale_height,
            n_engines=self.n_active,
        )

    def _solve_kw(self):
        # NOTE: shared with ScvxGuidance (solve_scvx consumes this dict too) — LCvx-only knobs
        # (N vs fixed-dt, slew_rate_max) are passed explicitly at the LCvx call sites instead.
        return dict(
            target=self.target,
            theta_max=self.theta,
            gamma_gs=self.gamma,
            solver=self.solver,
            pos_scale=self.scale_pos,
            vel_scale=self.scale_vel,
            accel_scale=self.scale_accel,
            **self._bp_kw(),
        )

    def _terminal_kw(self, terminal_slack: bool):
        return dict(
            terminal_slack=bool(terminal_slack),
            terminal_pos_weight=self.terminal_pos_weight,
            terminal_vel_weight=self.terminal_vel_weight,
            terminal_vz_max=self.terminal_vz_max,
            terminal_upright=self.terminal_upright,
        )

    def _engine_off(self, mem):
        return attitude_command(self.phase, self.q_up, np.zeros_like(self.active)), mem

    def _coast(self, mem):
        return self._engine_off(mem)

    def _cutoff(self, mem):
        mem = {**mem, "cutoff": True, "frozen": True}
        return self._engine_off(mem)

    def _command(self, thrust_ned, mem, t):
        """Thrust command with a thrust-aligned attitude reference (tilt capped at tilt_max).

        The body tilts toward the commanded thrust direction so sustained lateral thrust flows
        through the CG with zero steady torque; the gimbal stabilizes around this reference.
        This realizes the 3-DOF point-mass assumption (ACB-2013: attitude points the thrust) at
        the tracking layer; a hardcoded upright-hold instead makes the gimbal fight its own
        pointing torque and tumble on any sizeable lateral offset.
        No reference smoothing is applied — the projected command direction is continuous within
        a plan, the small ZOH node steps are well inside the PD's tracking ability, and the
        attitude-transient lag at ignition/re-solve is a drift channel absorbed by the commit
        margin (the standard 3-DOF treatment: constant cone + margin; attitude-in-the-loop
        planning is the 6-DoF/SCvx upgrade, Szmuk 2018 eqs 9-12).
        """
        return (
            thrust_command(
                self.phase,
                np.asarray(thrust_ned, float),
                thrust_aligned_quat(np.asarray(thrust_ned, float), self.tilt_max),
                self.active,
            ),
            mem,
        )

    @staticmethod
    def _pack(s, N):
        solve_time = s.get("solve_time_sec")
        plan = {
            "r": np.asarray(s["r"]),
            "v": np.asarray(s["v"]),
            "thrust": np.asarray(s["thrust"]),
            "m": np.asarray(s["m"]),
            "u": np.asarray(s["u"]),
            "sigma": np.asarray(s["sigma"]),
            "fuel": float(s.get("fuel", 0.0)),
            "status": str(s.get("status", "")),
            "tf": float(s["tf"]),
            "N": N,
            "iters": int(s.get("iters", 0)),
            "solve_time_sec": None if solve_time is None else float(solve_time),
        }
        if "attempt_log" in s:
            plan["attempt_log"] = list(s["attempt_log"])
        if s.get("feedforward_force"):
            plan["feedforward_force"] = True
        if "terminal_pos_slack" in s:
            plan["terminal_pos_slack"] = float(s["terminal_pos_slack"])
            plan["terminal_vel_slack"] = float(s["terminal_vel_slack"])
            plan["terminal_pos_l1_slack"] = float(s.get("terminal_pos_l1_slack", 0.0))
            plan["terminal_vel_l1_slack"] = float(s.get("terminal_vel_l1_slack", 0.0))
            plan["terminal_pos_l2_epigraph"] = float(s.get("terminal_pos_l2_epigraph", 0.0))
            plan["terminal_vel_l2_epigraph"] = float(s.get("terminal_vel_l2_epigraph", 0.0))
        return plan

    @staticmethod
    def _attempt_record(result, meta=None):
        """Normalize one actual solver invocation without inventing unavailable timings."""
        meta = {} if meta is None else dict(meta)
        solve_time = result.get("solve_time_sec")
        wall_time = meta.get("wall_time_sec", result.get("wall_time_sec"))
        solver_feasible = bool(result.get("solver_feasible", result.get("feasible", False)))
        mission_feasible = result.get("mission_feasible")
        return {
            **meta,
            "status": str(result.get("status", "")),
            "feasible": bool(result.get("feasible", False)),
            "solver_feasible": solver_feasible,
            "mission_feasible": None if mission_feasible is None else bool(mission_feasible),
            "postcheck_failures": list(result.get("postcheck_failures", [])),
            "iters": int(result.get("iters", 0)),
            "solve_time_sec": None if solve_time is None else float(solve_time),
            "wall_time_sec": None if wall_time is None else float(wall_time),
            "tf": result.get("tf", meta.get("tf")),
            "N": result.get("N", meta.get("N")),
        }

    def _plan(
        self,
        r0,
        v0,
        m0,
        tf_lo=None,
        tf_hi=None,
        rho_cap=None,
        iters=None,
        grid_fallback=None,
        terminal_slack=False,
        drag_ref=None,
    ):
        """Free-time fuel-optimal descent search used for ignition timing.

        ``drag_ref=(plan, offset)`` runs the search WITH the DESCENDO drag template resampled
        per candidate tf (SOCP-2 free-tf pass) — the drag-aware feasible window is not centred
        on the drag-free tf*, so a fixed-tf template refine alone certifies infeasible."""
        rho_max = self.rho_max if rho_cap is None else rho_cap
        gfb = self.grid_fallback if grid_fallback is None else grid_fallback
        drag_seq_fn = None
        if drag_ref is not None:
            src_plan, offset = drag_ref

            def drag_seq_fn(tf, n):
                return self._drag_template(src_plan, offset, float(tf), int(n))

        attempts = []

        def capture_attempt(result, meta):
            attempts.append(self._attempt_record(result, meta))

        if self.dt is not None:
            # AP-2007 Algorithm 1: integer line search on N at fixed dt (tf = N*dt).
            s = min_fuel_fixed_dt(
                r0,
                v0,
                m0,
                self.g,
                self.alpha,
                self.rho_min,
                rho_max,
                self.m_dry,
                dt=self.dt,
                n_cap=self.n_cap,
                grid_fallback=gfb,
                drag_seq_fn=drag_seq_fn,
                slew_rate_max=self.slew_rate_max,
                thrust_rate_max=self.thrust_rate_max,
                attempt_callback=capture_attempt,
                **self._solve_kw(),
                **self._terminal_kw(terminal_slack),
            )
        else:
            s = min_fuel_tf(
                r0,
                v0,
                m0,
                self.g,
                self.alpha,
                self.rho_min,
                rho_max,
                self.m_dry,
                N=self.N,
                tf_lo=tf_lo,
                tf_hi=tf_hi,
                iters=self.tf_search_iters if iters is None else iters,
                grid_fallback=gfb,
                drag_seq_fn=drag_seq_fn,
                slew_rate_max=self.slew_rate_max,
                thrust_rate_max=self.thrust_rate_max,
                attempt_callback=capture_attempt,
                **self._solve_kw(),
                **self._terminal_kw(terminal_slack),
            )
        self._last_status = str(s.get("status", ""))
        self._last_grid_fb = bool(s.get("grid_fallback_used", False))
        self._last_attempts = attempts or [self._attempt_record(s, {"kind": "free_time"})]
        if not s.get("feasible"):
            return None
        n_used = int(round(float(s["tf"]) / self.dt)) if self.dt is not None else self.N
        return self._pack(s, n_used)

    def _has_drag_template(self) -> bool:
        return self.rho0 > 0.0 and self.drag_cd_s > 0.0

    def _drag_template(self, plan, offset: float, tf: float, n: int):
        """DESCENDO eq. (21) frozen-coefficient drag template d*_k [1/s] from a previous plan.

        Sampled at the new grid's node times (``offset`` = elapsed time into the source plan).
        Returns None when there is no atmosphere (vacuum) or no source plan (SOCP-1 stays
        drag-free, per DESCENDO)."""
        if plan is None or self.drag_cd_s <= 0.0 or self.rho0 <= 0.0:
            return None
        t_src = np.linspace(0.0, float(plan["tf"]), int(plan["N"]) + 1)
        t_q = float(offset) + np.linspace(0.0, float(tf), n + 1)[:-1]
        alt = np.interp(t_q, t_src, -np.asarray(plan["r"])[:, 2])
        v = np.vstack([np.interp(t_q, t_src, np.asarray(plan["v"])[:, j]) for j in range(3)])
        spd = np.linalg.norm(v, axis=0)
        m = np.interp(t_q, t_src, np.asarray(plan["m"]))
        rho = self.rho0 * np.exp(-np.maximum(alt, 0.0) / float(self.scale_height))
        return 0.5 * rho * self.drag_cd_s * spd / np.maximum(m, 1.0)

    def _solve_fixed(
        self,
        r0,
        v0,
        m0,
        tf,
        n_nodes,
        rho_cap=None,
        terminal_slack=False,
        drag_seq=None,
        warm_ref=None,
    ):
        """Solve one fixed-time plan; SCvx overrides this solver hook."""
        return solve_pdg(
            r0,
            v0,
            m0,
            self.g,
            self.alpha,
            self.rho_min,
            self.rho_max if rho_cap is None else rho_cap,
            self.m_dry,
            float(tf),
            N=n_nodes,
            slew_rate_max=self.slew_rate_max,
            thrust_rate_max=self.thrust_rate_max,
            drag_seq=drag_seq,
            **self._solve_kw(),
            **self._terminal_kw(terminal_slack),
        )

    def _fixed_plan(self, r0, v0, m0, tf, rho_cap=None, terminal_slack=False, drag_ref=None):
        if self.dt is not None:
            # Snap to the fixed node grid: commit and every anchored re-solve share the SAME dt,
            # so the countdown never re-grids (dt = tf_go/N variation is a tracking-drift source).
            # In nominal operation tf is already a grid multiple (commit snapped here +
            # guidance ticks commensurate with dt), so round() never stretches the clock.
            n_nodes = max(int(round(float(tf) / self.dt)), 2)
            tf = n_nodes * self.dt
        else:
            n_nodes = self.N
        drag_seq = None
        if drag_ref is not None:
            src_plan, offset = drag_ref
            drag_seq = self._drag_template(src_plan, offset, float(tf), n_nodes)
        t0 = time.perf_counter()
        s = self._solve_fixed(
            r0,
            v0,
            m0,
            float(tf),
            n_nodes,
            rho_cap=rho_cap,
            terminal_slack=terminal_slack,
            drag_seq=drag_seq,
            warm_ref=drag_ref,
        )
        wall_s = time.perf_counter() - t0
        self._last_status = str(s.get("status", ""))
        self._last_grid_fb = False
        nested = s.get("attempt_log")
        self._last_attempts = (
            [dict(record) for record in nested]
            if nested
            else [
                self._attempt_record(
                    s,
                    {
                        "kind": "fixed_time",
                        "tf": float(tf),
                        "N": int(n_nodes),
                        "wall_time_sec": wall_s,
                    },
                )
            ]
        )
        return self._pack(s, n_nodes) if s.get("feasible") else None

    def _time_to_go(self, t: float, mem: dict) -> float | None:
        if mem.get("t_ig0") is None or mem.get("tf0") is None:
            return None
        return float(mem["tf0"]) - (float(t) - float(mem["t_ig0"]))

    def _should_cutoff(self, x_hat) -> bool:
        if self.landing_cutoff_impact_mps <= 0.0:
            return False
        alt = float(-np.asarray(x_hat.r_ned)[2])
        vz = float(np.asarray(x_hat.v_ned)[2])  # NED: positive means descending.
        # Ballistic-impact predictor: cut the engine when free-falling from here lands within the
        # target speed, v_impact = sqrt(vz^2 + 2 g h) <= target (energy conservation; the sign of
        # vz is irrelevant, so a low hover/ascent is also cut and dropped instead of climbing
        # away at TWR_floor > 1). Drag at these speeds is negligible and the shutdown thrust
        # tail-off only brakes further, so the prediction is a conservative upper bound.
        v_impact = float(np.sqrt(vz * vz + 2.0 * float(self.g[2]) * max(alt, 0.0)))
        return v_impact <= self.landing_cutoff_impact_mps

    def _ref(self, plan, tau, h, m0):
        """Reference state and the plan's feedforward control at time ``tau``."""
        tf, N = plan["tf"], plan["N"]
        if tau >= tf:
            # Exhausted plans are handled in _track (engine off, with no terminal rescue);
            # clamp to the terminal node if a caller still asks past the horizon.
            tau = float(tf) - 1e-9
        # Cubic Hermite interpolation preserves both position and velocity at every node. A
        # constant-acceleration parabola is inconsistent with SCvx's nonlinear RK4 propagation.
        dt_n = tf / N
        k = min(int(tau / dt_n), N - 1)
        q = (float(tau) - k * dt_n) / dt_n
        q2, q3 = q * q, q * q * q
        r_k = np.asarray(plan["r"][k], float)
        r_next = np.asarray(plan["r"][k + 1], float)
        v_k = np.asarray(plan["v"][k], float)
        v_next = np.asarray(plan["v"][k + 1], float)
        r_ref = (
            (2.0 * q3 - 3.0 * q2 + 1.0) * r_k
            + (q3 - 2.0 * q2 + q) * dt_n * v_k
            + (-2.0 * q3 + 3.0 * q2) * r_next
            + (q3 - q2) * dt_n * v_next
        )
        v_ref = (
            (6.0 * q2 - 6.0 * q) * r_k / dt_n
            + (3.0 * q2 - 4.0 * q + 1.0) * v_k
            + (-6.0 * q2 + 6.0 * q) * r_next / dt_n
            + (3.0 * q2 - 2.0 * q) * v_next
        )
        feedforward = plan["thrust"][k] if plan.get("feedforward_force") else plan["u"][k]
        return r_ref, v_ref, feedforward

    def _achievable_bounds(self, h: float) -> tuple[float, float]:
        """Back-pressure-corrected DELIVERED thrust bounds [rho_min_ach, rho_max_ach] at altitude h.

        Identical frame to the plan's rho_*_ach (qoco_direct/lcvx): delivered F = throttle*T_vac -
        bp shifts both net bounds down by the fixed debit bp = n*p_amb*A_e. The tracking command
        is a DELIVERED thrust, so it must be capped here, not at the vacuum rho_max.
        """
        p_amb = self.p_sea * np.exp(-max(h, 0.0) / float(self.scale_height))
        bp = self.n_active * p_amb * self.nozzle_exit_area
        rho_max_ach = max(self.rho_max - bp, 1e-3)
        rho_min_ach = max(self.rho_min - bp, 1e-3)
        return rho_min_ach, rho_max_ach

    def _project_achievable(self, t_cmd, h: float):
        """Project a raw NED thrust command onto the achievable set and return it.

        The achievable set with the thrust-aligned attitude reference is {F = mag * d : d within
        tilt_max of NED-up, mag in [rho_min_ach(h), rho_max_ach(h)]} — the SAME cone the plan's
        pointing constraint and the q_ref tilt cap use (single source). The raw t_cmd is a free
        3-vector, so a DOWNWARD feedback command (target below the vehicle after an early
        velocity-null) would otherwise be realized as full UPWARD throttle -> fly-away:
          1. up = NED up = -e_z. A command with non-positive up-component cannot be produced
             thrusting up, so it collapses to min thrust straight up. This is a saturation GUARD
             (occurrences counted in mem['proj_downthrust_clamps'], not silent).
          2. clamp the tilt to tilt_max (lateral component <= up-component * tan(tilt_max)).
          3. clamp the along-axis delivered magnitude to [rho_min_ach, rho_max_ach].
        """
        rho_min_ach, rho_max_ach = self._achievable_bounds(h)
        up = np.array([0.0, 0.0, -1.0])  # NED up
        s = float(t_cmd @ up)  # up-component of the command
        lat = t_cmd - s * up  # lateral (cone-tilt) component
        lat_norm = float(np.linalg.norm(lat))
        if s <= 0.0:
            # Cannot thrust downward: command minimum thrust, straight up.
            return rho_min_ach * up
        # Clamp tilt into the achievable cone: max lateral / up ratio is tan(tilt_max).
        max_lat = s * np.tan(self.tilt_max)
        if lat_norm > max_lat and lat_norm > 0.0:
            lat = lat * (max_lat / lat_norm)
        d = s * up + lat
        d = d / float(np.linalg.norm(d))  # unit thrust direction inside the cone
        # Delivered magnitude along the achievable axis = projection of t_cmd onto d, capped.
        mag = float(np.clip(t_cmd @ d, rho_min_ach, rho_max_ach))
        return mag * d

    def _track(self, x_hat, t, mem):
        """Track the cached plan without solving a new convex problem."""
        plan, t_ig = mem.get("plan"), mem.get("t_ignite")
        if plan is None:  # no feasible continuous burn yet => coast (still too high)
            return self._coast(mem)
        r0 = np.asarray(x_hat.r_ned, float)
        v0 = np.asarray(x_hat.v_ned, float)
        m0 = float(np.asarray(x_hat.m))
        h = -float(r0[2])  # altitude (NED down is +z)
        tau = float(t) - float(t_ig)
        if tau >= float(plan["tf"]):
            # The optimizer owns the terminal condition. Do not hide a missed terminal state with
            # an unplanned minimum-thrust extension after the committed horizon.
            mem["plan_exhausted_ticks"] = int(mem.get("plan_exhausted_ticks", 0)) + 1
            return self._engine_off(mem)
        r_ref, v_ref, feedforward = self._ref(plan, tau, h, m0)
        feedback = self.k_r * (r_ref - r0) + self.k_v * (v_ref - v0)
        if plan.get("feedforward_force"):
            t_cmd = feedforward + m0 * feedback
        else:
            t_cmd = m0 * (feedforward + feedback)
        if -float(t_cmd[2]) <= 0.0:  # telemetry: a DOWN-thrust command is about to be clamped
            mem["proj_downthrust_clamps"] = int(mem.get("proj_downthrust_clamps", 0)) + 1
        t_cmd = self._project_achievable(t_cmd, h)
        return self._command(t_cmd, mem, t)

    def track(self, x_hat, t, memory: dict):
        """Run the fast feedforward plus feedback tracking tick."""
        mem = self._memory(memory)
        if mem["cutoff"] or self._should_cutoff(x_hat):
            return self._cutoff(mem)
        return self._track(x_hat, t, mem)

    def _log_solve(self, mem, t, kind, wall_s, plan):
        """Append every conic attempt made by one guidance operation."""
        attempts = list(getattr(self, "_last_attempts", []))
        if not attempts:
            result = {} if plan is None else plan
            attempts = [
                self._attempt_record(
                    result,
                    {
                        "wall_time_sec": wall_s,
                        "kind": "unknown",
                    },
                )
            ]
        grid_fallback = bool(getattr(self, "_last_grid_fb", False))
        log = mem.setdefault("solve_log", [])
        for index, attempt in enumerate(attempts):
            solve_s = attempt.get("solve_time_sec")
            attempt_wall_s = attempt.get("wall_time_sec")
            log.append(
                {
                    "t": float(t),
                    "kind": kind,
                    "stage": str(attempt.get("kind", "solve")),
                    "record_type": str(attempt.get("record_type", "solve")),
                    "search_stage": attempt.get("search_stage"),
                    "attempt": index,
                    "wall_ms": None if attempt_wall_s is None else float(attempt_wall_s) * 1.0e3,
                    "solve_ms": None if solve_s is None else float(solve_s) * 1.0e3,
                    "iters": int(attempt.get("iters", 0)),
                    "feasible": bool(attempt.get("feasible", False)),
                    "solver_feasible": bool(
                        attempt.get("solver_feasible", attempt.get("feasible", False))
                    ),
                    "mission_feasible": attempt.get("mission_feasible"),
                    "postcheck_failures": list(attempt.get("postcheck_failures", [])),
                    "status": str(attempt.get("status", "")),
                    "tf": attempt.get("tf"),
                    "N": attempt.get("N"),
                    # One search can make many conic attempts; count the coarse-grid event once.
                    "grid_fallback": grid_fallback and index == len(attempts) - 1,
                }
            )

    def __call__(self, x_hat, t, memory: dict):
        """Run the slow guidance tick: free-time ignition, then fixed-time re-solve."""
        r0 = np.asarray(x_hat.r_ned, float)
        v0 = np.asarray(x_hat.v_ned, float)
        m0 = float(np.asarray(x_hat.m))
        mem = self._memory(memory)
        if mem["cutoff"] or self._should_cutoff(x_hat):
            return self._cutoff(mem)

        def logged(kind, solve):
            t0 = time.perf_counter()
            result = solve()
            self._log_solve(mem, t, kind, time.perf_counter() - t0, result)
            return result

        if not mem["ignited"]:
            # Free-tf search == hoverslam ignition test.
            new = logged(
                "ignite_search",
                lambda: self._plan(r0, v0, m0, grid_fallback=self.grid_fallback),
            )
            if new is not None and self._has_drag_template():
                # DESCENDO SOCP-2 at ignition (N_P=1): re-run the FREE-tf search with the
                # template from the drag-free seed. A fixed-tf refine at the drag-free tf* can
                # certify infeasible because the drag-aware feasible window sits elsewhere.
                socp2 = logged(
                    "ignite_drag_refine",
                    lambda: self._plan(r0, v0, m0, drag_ref=(new, 0.0)),
                )
                if socp2 is not None:
                    new = socp2
                else:
                    mem["template_refinement_failures"] = (
                        int(mem["template_refinement_failures"]) + 1
                    )
                    new = None  # never commit a drag-free seed after drag-aware infeasibility
            if new is None:
                return self._coast(mem)  # still too high to start a continuous burn
            mem["plan"], mem["t_ignite"], mem["ignited"] = new, float(t), True
            mem["last_replan_failed"] = False
            mem["t_ig0"], mem["tf0"] = float(t), new["tf"]
            # Keep the ignition-committed plan for post-run plan-vs-flown diagnostics
            # (never overwritten by re-solves/re-anchors).
            mem["plan_ignition"], mem["t_ignition"] = new, float(t)
            mem["terminal_pos_slack"] = float(new.get("terminal_pos_slack", 0.0))
            mem["terminal_vel_slack"] = float(new.get("terminal_vel_slack", 0.0))
        elif self.replan and not mem["frozen"]:
            tf_go = self._time_to_go(t, mem)
            if tf_go is not None and tf_go > self.tf_freeze:
                # After ignition the touchdown clock is anchored. Re-solve the fixed-time SOCP at
                # the remaining time instead of running a free-time search over [tf_floor, tf_go].
                # Drag template from the latest committed plan, time-shifted to now (DESCENDO
                # closed loop: each guidance step seeds SOCP 2 from the previous solution).
                d_ref = (mem["plan"], float(t) - float(mem["t_ignite"]))
                new = logged(
                    "resolve",
                    lambda: self._fixed_plan(r0, v0, m0, tf_go, drag_ref=d_ref),
                )
                if new is None and self.terminal_slack:
                    new = logged(
                        "resolve_terminal_slack",
                        lambda: self._fixed_plan(
                            r0,
                            v0,
                            m0,
                            tf_go,
                            terminal_slack=True,
                            drag_ref=d_ref,
                        ),
                    )
                    if new is not None:
                        mem["terminal_slack_uses"] = int(mem["terminal_slack_uses"]) + 1
                if new is None and (-float(r0[2])) > self.reanchor_min_alt:
                    # The anchored countdown slipped out of the feasible-tf window because
                    # unmodeled drag or actuator lag changed the terminal deficit. Re-anchor with
                    # a new free-time plan, event-triggered by solver infeasibility and counted.
                    # Gated to altitudes ABOVE reanchor_min_alt: near the ground re-anchoring
                    # resets the countdown forever and the vehicle hovers instead of flying out.
                    new = logged("reanchor_search", lambda: self._plan(r0, v0, m0))
                    if new is not None and self._has_drag_template():
                        socp2 = logged(
                            "reanchor_drag_refine",
                            lambda: self._plan(r0, v0, m0, drag_ref=(new, 0.0)),
                        )
                        if socp2 is not None:
                            new = socp2
                        else:
                            mem["template_refinement_failures"] = (
                                int(mem["template_refinement_failures"]) + 1
                            )
                            new = None
                    if new is not None:
                        mem["reanchor_count"] = int(mem["reanchor_count"]) + 1
                        mem["reanchor_time_added"] = float(mem["reanchor_time_added"]) + (
                            float(new["tf"]) - float(tf_go)
                        )
                        mem["t_ig0"], mem["tf0"] = float(t), new["tf"]
                if new is not None:
                    mem["plan"], mem["t_ignite"] = new, float(t)
                    mem["last_replan_failed"] = False
                    mem["terminal_pos_slack"] = float(new.get("terminal_pos_slack", 0.0))
                    mem["terminal_vel_slack"] = float(new.get("terminal_vel_slack", 0.0))
                else:
                    # Fly the cached shifted plan for this tick, but try the convex solve again on
                    # the next guidance tick. A transient infeasible solve must not become a
                    # permanent terminal latch.
                    mem["last_replan_failed"] = True
                    mem["replan_failures"] = int(mem["replan_failures"]) + 1
                    mem["cached_plan_fallbacks"] = int(mem["cached_plan_fallbacks"]) + 1
            else:
                # Terminal freeze: fly out the committed plan FF+FB to the cutoff box. Sub-window
                # SOCPs are structurally infeasible (N>=2 + gs + throttle floor) and the standard
                # practice is a planned fly-out (DESCENDO p.2212 stale-plan / CTV guidance-off /
                # G-FOLD constant-velocity terminal phase, Scharf 2017 IV.D).
                mem["frozen"] = True
        return self._track(x_hat, t, mem)

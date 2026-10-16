"""Thin Python boundary for the native 3-DOF SCvx powered-descent solver."""

from __future__ import annotations

import time

import _rlv_sim_native as _native
import numpy as np

from gnc.guidance.convex.guidance import LcvxGuidance
from gnc.guidance.convex.lcvx import solve_pdg


# fmt: off
def solve_scvx(
    r0, v0, m0,
    g, alpha, rho_min, rho_max, m_dry,
    tf, N,
    beta0=0.0, drag_mach_grid=None, drag_beta_grid=None, sound_speed=340.3,
    target=(0.0, 0.0, 0.0), theta_max=None, gamma_gs=None,
    iters=12, trust=80.0, w_nu=1.0e4, tol=1e-2,
    solver="CLARABEL",
    p_sea=0.0, nozzle_exit_area=0.0, scale_height=None, n_engines=1,
    terminal_slack=False, terminal_pos_weight=1.0e4, terminal_vel_weight=1.0e4,
    terminal_vz_max=None, terminal_upright=False,
    pos_scale=None, vel_scale=None, accel_scale=None,
    slew_rate_max=None, thrust_rate_max=None,
    init=None,
):
# fmt: on
    """Solve one fixed-time force-ZOH problem with native RK4 SCvx and an LCvx initial guess.

    ``solver`` selects only the LCvx initialization backend. Every successive
    convexification subproblem is solved by the QOCO instance compiled into the
    native extension. ``beta0`` is ``0.5 * rho0 * Cd * area`` at sea level;
    optional Mach/``beta`` grids replace that constant when supplied.
    """
    del terminal_pos_weight, terminal_vel_weight
    if terminal_slack:
        raise NotImplementedError("native SCvx supports hard terminal constraints only")
    if slew_rate_max is not None:
        raise NotImplementedError(
            "specific-thrust slew limits are not implemented for physical-thrust SCvx"
        )

    r0 = np.asarray(r0, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    g = np.asarray(g, dtype=float)
    target = np.asarray(target, dtype=float)
    tf = float(tf)
    N = int(N)
    if r0.shape != (3,) or v0.shape != (3,) or g.shape != (3,) or target.shape != (3,):
        raise ValueError("r0, v0, g, and target must have shape (3,)")
    if beta0 < 0.0:
        raise ValueError("beta0 must be non-negative")
    if (drag_mach_grid is None) != (drag_beta_grid is None):
        raise ValueError("drag_mach_grid and drag_beta_grid must be supplied together")

    attempt_log = []
    if init is None:
        started = time.perf_counter()
        init = solve_pdg(
            r0,
            v0,
            m0,
            g,
            alpha,
            rho_min,
            rho_max,
            m_dry,
            tf,
            N,
            target=target,
            theta_max=theta_max,
            gamma_gs=gamma_gs,
            solver=solver,
            p_sea=p_sea,
            nozzle_exit_area=nozzle_exit_area,
            scale_height=scale_height,
            n_engines=n_engines,
            terminal_vz_max=terminal_vz_max,
            terminal_upright=terminal_upright,
            pos_scale=pos_scale,
            vel_scale=vel_scale,
            accel_scale=accel_scale,
            slew_rate_max=slew_rate_max,
            thrust_rate_max=thrust_rate_max,
        )
        attempt_log.append(
            {
                "kind": "scvx_init_lcvx",
                "wall_time_sec": time.perf_counter() - started,
                "solve_time_sec": init.get("solve_time_sec"),
                "iters": int(init.get("iters", 0)),
                "status": str(init.get("status", "")),
                "feasible": bool(init.get("feasible", False)),
                "solver_feasible": bool(
                    init.get("solver_feasible", init.get("feasible", False))
                ),
                "mission_feasible": init.get("mission_feasible"),
                "tf": tf,
                "N": N,
            }
        )

    required = ("r", "v", "m", "u")
    if not all(key in init for key in required):
        return {
            "status": "init_infeasible",
            "solver_status": str(init.get("status", "")),
            "solver_feasible": False,
            "feasible": False,
            "mission_feasible": False,
            "plan_physical_feasible": False,
            "terminal_constraints_met": False,
            "postcheck_failures": ["initialization"],
            "tf": tf,
            "N": N,
            "iters": 0,
            "attempt_log": attempt_log,
            "solve_time_sec": None,
        }

    pos_scale = float(
        pos_scale
        if pos_scale is not None
        else init.get("scale_pos", max(float(np.linalg.norm(r0 - target)), 100.0))
    )
    vel_scale = float(
        vel_scale
        if vel_scale is not None
        else init.get("scale_vel", max(float(np.linalg.norm(v0)), pos_scale / tf, 10.0))
    )
    accel_scale = float(
        accel_scale
        if accel_scale is not None
        else init.get(
            "scale_accel",
            max(float(rho_max) / max(float(m_dry), 1.0e-6), float(np.linalg.norm(g)), 1.0),
        )
    )

    problem = {
        "r0": r0,
        "v0": v0,
        "m0": float(m0),
        "g": g,
        "alpha": float(alpha),
        "rho_min": float(rho_min),
        "rho_max": float(rho_max),
        "m_dry": float(m_dry),
        "tf": tf,
        "N": N,
        "beta0": float(beta0),
        "drag_mach_grid": (
            None if drag_mach_grid is None else np.asarray(drag_mach_grid, dtype=float)
        ),
        "drag_beta_grid": (
            None if drag_beta_grid is None else np.asarray(drag_beta_grid, dtype=float)
        ),
        "sound_speed": float(sound_speed),
        "scale_height": 0.0 if scale_height is None else float(scale_height),
        "backpressure0": float(n_engines) * float(p_sea) * float(nozzle_exit_area),
        "target": target,
        "theta_max": theta_max,
        "gamma_gs": gamma_gs,
        "terminal_vz_max": terminal_vz_max,
        "terminal_upright": bool(terminal_upright),
        "slew_rate_max": slew_rate_max,
        "thrust_rate_max": thrust_rate_max,
        "iters": int(iters),
        "trust": float(trust),
        "w_nu": float(w_nu),
        "tol": float(tol),
        "pos_scale": pos_scale,
        "vel_scale": vel_scale,
        "accel_scale": accel_scale,
    }
    result = _native.solve_scvx(problem, init)
    result["attempt_log"] = [*attempt_log, *result.get("attempt_log", [])]
    return result


class ScvxGuidance(LcvxGuidance):
    """Drag-aware powered-descent guidance using the native SCvx solver."""

    def __init__(
        self,
        nominal,
        active_engines,
        phase,
        params,
        drag_scale: float,
        scvx_iters: int,
        scvx_trust: float,
        scvx_virtual_control_weight: float,
        scvx_tolerance: float,
        **kw,
    ):
        super().__init__(nominal, active_engines, phase, **kw)
        # SCvx models drag in the nonlinear flow, so the LCvx frozen-drag
        # template inherited from LcvxGuidance must stay disabled here.
        self.drag_cd_s = 0.0
        mach = np.asarray(params.aero.mach_grid, dtype=float)
        area = float(params.geom.s_ref) * np.asarray(params.aero.cd_table, dtype=float)[-1]
        if float(params.aero.fin_incremental) <= 0.5:
            transonic = 1.0 + 2.0 * np.exp(
                -((mach - float(params.aero.fin_transonic_mach)) / 0.15) ** 2
            )
            area = area + (
                int(params.aero.n_fins)
                * float(params.aero.fin_area)
                * float(params.aero.fin_cd0)
                * transonic
            )
        self.drag_mach_grid = mach
        self.drag_beta_grid = (
            0.5 * float(params.atmosphere.rho0) * float(drag_scale) * area
        )
        self.sound_speed = float(params.atmosphere.a_sound)
        self.scvx_iters = int(scvx_iters)
        self.scvx_trust = float(scvx_trust)
        self.scvx_virtual_control_weight = float(scvx_virtual_control_weight)
        self.scvx_tolerance = float(scvx_tolerance)

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
        # LCvx supplies the time-of-flight and initial trajectory. SCvx then
        # expands from that time only when the nonlinear problem is infeasible.
        seed = super()._plan(
            r0,
            v0,
            m0,
            tf_lo=tf_lo,
            tf_hi=tf_hi,
            rho_cap=rho_cap,
            iters=iters,
            grid_fallback=grid_fallback,
            terminal_slack=terminal_slack,
        )
        attempts = list(getattr(self, "_last_attempts", []))
        if seed is None:
            return None

        def capture(result, stage, wall_s):
            nested = result.get("attempt_log")
            if nested:
                attempts.extend({**record, "search_stage": stage} for record in nested)
            else:
                attempts.append(
                    self._attempt_record(
                        result,
                        {
                            "kind": "scvx_fixed",
                            "search_stage": stage,
                            "wall_time_sec": wall_s,
                        },
                    )
                )

        tf_seed, n_seed = float(seed["tf"]), int(seed["N"])
        step = self.dt if self.dt is not None else tf_seed / n_seed
        warm = seed
        for k in range(8):
            n_nodes = n_seed + k if self.dt is not None else n_seed
            if n_nodes > self.n_cap:
                break
            tf = n_nodes * self.dt if self.dt is not None else tf_seed + k * step
            if tf_hi is not None and tf > float(tf_hi):
                break
            started = time.perf_counter()
            result = self._solve_fixed(
                r0,
                v0,
                m0,
                tf,
                n_nodes,
                rho_cap=rho_cap,
                terminal_slack=terminal_slack,
                init=warm if self.dt is None or k == 0 else None,
            )
            capture(result, "expand", time.perf_counter() - started)
            self._last_status = str(result.get("status", ""))
            if all(key in result for key in ("r", "v", "m", "u")):
                warm = result
            if result.get("feasible"):
                self._last_attempts = attempts
                return self._pack(result, n_nodes)
        self._last_attempts = attempts
        return None

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
        init=None,
    ):
        """Solve one fixed-time nonlinear powered-descent problem."""
        del drag_seq
        started = time.perf_counter()
        rho_max = self.rho_max if rho_cap is None else rho_cap
        if init is None and warm_ref is not None:
            plan, offset = warm_ref
            t_state = np.linspace(0.0, float(plan["tf"]), int(plan["N"]) + 1)
            t_control = t_state[:-1]
            query = float(offset) + np.linspace(0.0, float(tf), int(n_nodes) + 1)

            def interp(values, times):
                values = np.asarray(values)
                source = t_state if len(values) == len(t_state) else t_control
                if values.ndim == 1:
                    return np.interp(times, source, values)
                return np.column_stack(
                    [np.interp(times, source, values[:, j]) for j in range(values.shape[1])]
                )

            init = {
                "r": interp(plan["r"], query),
                "v": interp(plan["v"], query),
                "m": interp(plan["m"], query),
                "u": interp(plan["u"], query[:-1]),
            }

        result = solve_scvx(
            r0,
            v0,
            m0,
            self.g,
            self.alpha,
            self.rho_min,
            rho_max,
            self.m_dry,
            tf=float(tf),
            N=int(n_nodes),
            drag_mach_grid=self.drag_mach_grid,
            drag_beta_grid=self.drag_beta_grid,
            sound_speed=self.sound_speed,
            iters=self.scvx_iters,
            trust=self.scvx_trust,
            w_nu=self.scvx_virtual_control_weight,
            tol=self.scvx_tolerance,
            slew_rate_max=self.slew_rate_max,
            thrust_rate_max=self.thrust_rate_max,
            init=init,
            **self._solve_kw(),
            **self._terminal_kw(terminal_slack),
        )
        summary = self._attempt_record(
            {**result, "solve_time_sec": None},
            {
                "kind": "scvx_result",
                "record_type": "outcome",
                "wall_time_sec": time.perf_counter() - started,
                "tf": float(tf),
                "N": int(n_nodes),
            },
        )
        result["attempt_log"] = [*result.get("attempt_log", []), summary]
        return result

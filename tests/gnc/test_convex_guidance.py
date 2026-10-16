"""Convex powered-descent guidance: LCvx (lossless) + SCvx (successive convexification).

Open-loop checks cover the LCvx relaxation and SCvx convergence. The closed-loop regression covers
the nominal Falcon LCvx case; the unresolved SCvx closed-loop limitation is documented, not frozen
as an expected-failure test.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cvxpy")  # convex guidance is an optional (numpy-side) dependency

from gnc.guidance.convex.lcvx import min_fuel_tf, solve_pdg
from gnc.guidance.convex.scvx import solve_scvx

G0 = 9.80665
ALPHA = 1.0 / (300.0 * G0)
G = np.array([0.0, 0.0, G0])
RHO_MAX = 981000.0
RHO_MIN = 0.4 * RHO_MAX
M_DRY = 25000.0
# a feasible ignition state (a continuous >=40% burn can null it at the pad)
R0 = np.array([0.0, 0.0, -250.0])
V0 = np.array([0.0, 0.0, 55.0])
M0 = 30000.0

# Atmospheric back-pressure: the plant delivers F_net = throttle*T_vac - n*p_amb*A_e, so the
# planner adds a back-pressure fuel penalty and drops its thrust cap to n*(T_vac - p_amb*A_e).
P_SEA = 101325.0
A_E = 0.9
SCALE_H = 8500.0
N_ENG = 1


def _args(rho_max=RHO_MAX):
    return (G, ALPHA, RHO_MIN, rho_max, M_DRY)


def _bp_kw():
    return dict(p_sea=P_SEA, nozzle_exit_area=A_E, scale_height=SCALE_H, n_engines=N_ENG)


def test_lcvx_soft_pinpoint_landing():
    """The LCvx SOCP lands at the target with zero velocity (soft pinpoint)."""
    s = solve_pdg(R0, V0, M0, *_args(), tf=6.5, N=35)
    assert s["feasible"]
    assert np.allclose(s["r"][-1], [0.0, 0.0, 0.0], atol=1e-3)
    assert np.linalg.norm(s["v"][-1]) < 1e-4


def test_lcvx_relaxation_is_lossless():
    """The lossless relaxation is tight: |u| = sigma at the optimum (the 2007 theorem)."""
    s = solve_pdg(R0, V0, M0, *_args(), tf=6.5, N=35)
    assert s["feasible"]
    assert s["slack_gap"] < 1e-5


def test_lcvx_thrust_within_bounds():
    """Every node respects the deep-throttle band rho_min <= |T| <= rho_max."""
    s = solve_pdg(R0, V0, M0, *_args(), tf=6.5, N=35)
    Tmag = np.linalg.norm(s["thrust"], axis=1)
    assert np.all(Tmag <= RHO_MAX * (1 + 1e-6))
    assert np.all(Tmag >= RHO_MIN * (1 - 1e-3))


def test_lcvx_tf_search_is_fuel_optimal():
    """min_fuel_tf returns the fuel-minimizing time-of-flight: no neighbour fixed tf burns less."""
    best = min_fuel_tf(R0, V0, M0, *_args(), N=35)
    assert best["feasible"]
    for tf in [best["tf"] - 1.0, best["tf"] + 1.0, best["tf"] + 2.0]:
        s = solve_pdg(R0, V0, M0, *_args(), tf=tf, N=35)
        if s["feasible"]:
            assert best["fuel"] <= s["fuel"] + 1e-6


def test_scvx_recovers_lcvx_when_dragless():
    """SCvx with zero drag recovers the fuel cost of a lossless LCvx solution."""
    tf = 6.5
    lcvx = solve_pdg(R0, V0, M0, *_args(), tf=tf, N=35)
    scvx = solve_scvx(R0, V0, M0, *_args(), tf=tf, beta0=0.0, N=35)
    assert scvx["feasible"]
    assert scvx["nu_norm"] < 1e-6
    assert scvx["slack_gap"] <= scvx["thrust_bound_tolerance"]
    assert scvx["thrust_bound_violation"] <= scvx["thrust_bound_tolerance"]
    assert scvx["dynamics_defect"] < 1e-6
    assert scvx["nu"].shape == (35, 7)
    assert scvx["feedforward_force"]
    assert np.allclose(scvx["thrust"], scvx["m"][:-1, None] * scvx["u"], atol=1e-6)
    assert scvx["accepted_iters"] <= scvx["iters"] == scvx["solver_attempts"]
    assert scvx["rollout_terminal_pos_slack"] < 1e-2
    assert scvx["rollout_terminal_vel_violation"] < 1e-2
    assert np.isclose(scvx["fuel"], lcvx["fuel"], rtol=1e-4)
    assert np.allclose(scvx["r"][-1], lcvx["r"][-1], atol=1e-4)
    assert np.allclose(scvx["v"][-1], lcvx["v"][-1], atol=1e-4)


def test_scvx_converges_with_drag():
    """With real drag SCvx converges (nu -> 0) to a feasible soft landing and burns less fuel than
    the drag-free plan: on a descending booster drag points up and aids deceleration."""
    cd_s = 1.0 * 10.75  # C_D * S_ref (base-first bluff body)

    s = solve_scvx(
        R0,
        V0,
        M0,
        *_args(),
        tf=6.5,
        beta0=0.5 * 1.225 * cd_s,
        scale_height=8500.0,
        N=35,
        iters=15,
    )
    assert s["feasible"]
    assert s["nu_norm"] < 1e-2
    assert np.allclose(s["r"][-1], [0, 0, 0], atol=1e-2) and np.linalg.norm(s["v"][-1]) < 1e-2
    dragless = solve_pdg(R0, V0, M0, *_args(), tf=6.5, N=35)
    assert s["fuel"] < dragless["fuel"]


def test_scvx_rejects_a_malformed_initial_trajectory():
    """The native boundary rejects bad shapes instead of reshaping or replacing the seed."""
    seed = solve_pdg(R0, V0, M0, *_args(), tf=6.5, N=35)
    seed["u"] = seed["u"][:-1]

    with pytest.raises(ValueError, match=r"init\[u\].*shape"):
        solve_scvx(R0, V0, M0, *_args(), tf=6.5, beta0=0.0, N=35, init=seed)


def test_scvx_rejects_unimplemented_specific_thrust_slew_limit():
    """A force-ZOH solver must not silently reinterpret a specific-thrust slew limit."""
    with pytest.raises(NotImplementedError, match="physical-thrust"):
        solve_scvx(
            R0,
            V0,
            M0,
            *_args(),
            tf=6.5,
            beta0=0.0,
            N=35,
            slew_rate_max=1.0,
        )


def test_lcvx_backpressure_burns_more_fuel_than_vacuum():
    """Modeling the sea-level back-pressure debit makes the fuel-honest plan burn more than the
    vacuum plan (lower final mass) without breaking the lossless relaxation."""
    tf = 6.5
    vac = solve_pdg(R0, V0, M0, *_args(), tf=tf, N=35)
    bp = solve_pdg(R0, V0, M0, *_args(), tf=tf, N=35, **_bp_kw())
    assert bp["feasible"]
    assert bp["m"][-1] < vac["m"][-1]
    assert bp["fuel"] > vac["fuel"]
    assert bp["slack_gap"] < 1e-5


def test_convex_methods_do_not_rescue_terminal_state_with_cutoff():
    """Both methods expose their terminal error instead of using an early-cutoff rescue."""
    from conftest import compose_cfg

    lcvx = float(compose_cfg(["gnc/guidance=lcvx"]).gnc.guidance.landing_cutoff_impact_mps)
    scvx = float(compose_cfg(["gnc/guidance=scvx"]).gnc.guidance.landing_cutoff_impact_mps)
    assert lcvx == scvx == 0.0


def _run_closed_loop(method, actuator):
    """Roll out the nominal Falcon case without test-only guidance tuning."""
    from conftest import build_bundle, compose_cfg

    from gnc.build import build_gnc
    from gnc.navigation.identity import IdentityNav
    from plant.dynamics import ic_from_scenario, initial_state
    from sim.loop import run_closed_loop

    ov = [
        "vehicle_set=falcon9",
        f"gnc/guidance={method}",
        f"plant/actuator={actuator}",
    ]
    cfg = compose_cfg(ov)
    bundle = build_bundle(ov)
    h = cfg.scenario.horizon
    s0 = initial_state(ic_from_scenario(cfg.scenario, 0), bundle.params)
    r = run_closed_loop(
        bundle,
        s0,
        build_gnc(cfg, bundle.params),
        IdentityNav(),
        t_max=float(h.t_max),
        ctrl_dt=float(h.ctrl_dt),
        plant_dt=float(cfg.plant_dt),
        nav_dt=float(h.nav_dt),
        guidance_dt=float(h.guidance_dt),
        log_dt=float(h.log_dt),
    )
    assert np.all(np.isfinite(r.r_ned)) and np.all(np.isfinite(r.v_ned))
    return r


def test_lcvx_single_shot_has_no_replan_or_terminal_rescue():
    """The honest single-shot run may miss, but it must not hide the miss with replanning."""
    r = _run_closed_loop("lcvx", actuator="realistic")
    assert r.aux["solver"]["n_resolve"] == 0
    assert r.aux["solver"]["n_reanchor"] == 0
    assert r.aux["guidance_degradation"]["plan_exhausted_ticks"] > 0


def test_scvx_ideal_actuator_soft_landing_regression():
    """The matched-actuation Falcon case lands without replanning or a cutoff rescue."""
    r = _run_closed_loop("scvx", actuator="ideal")
    assert r.success
    assert r.landing_class == "soft"
    assert r.landing_speed < 2.0
    assert r.aux["solver"]["n_resolve"] == 0
    assert r.aux["solver"]["n_reanchor"] == 0

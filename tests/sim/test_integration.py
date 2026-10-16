"""End-to-end closed-loop integration smoke (proof-of-life).

This verifies that the full plant, GNC, and simulation loop runs, terminates, stays finite, and
shows physically sane behavior. It also exercises slosh, faults, and open-loop batches.
"""

from __future__ import annotations

import numpy as np
from conftest import build_bundle, compose_cfg

from core.interfaces import ActuatorCommand, Phase
from core.seeding import episode_keys
from gnc.build import build_gnc
from gnc.navigation.identity import IdentityNav
from plant.dynamics import ic_from_scenario, initial_state
from sim.loop import run_closed_loop
from sim.open_loop import open_loop_schedule


def _run(overrides=None, t_max=25.0, ctrl_dt=0.05):
    cfg = compose_cfg(overrides)
    bundle = build_bundle(overrides)
    ic = ic_from_scenario(cfg.scenario, 0)
    state0 = initial_state(ic, bundle.params)
    controller = build_gnc(cfg, bundle.params)
    return run_closed_loop(
        bundle,
        state0,
        controller,
        IdentityNav(),
        t_max=t_max,
        ctrl_dt=ctrl_dt,
        plant_dt=float(cfg.plant_dt),
    )


def test_closed_loop_runs_and_descends():
    r = _run()
    alt = -r.r_ned[:, 2]
    assert np.all(np.isfinite(r.r_ned)) and np.all(np.isfinite(r.v_ned))
    assert np.all(np.isfinite(r.q_bn)) and np.all(np.isfinite(r.w_body))
    assert alt[-1] < alt[0]  # descends
    assert np.allclose(np.linalg.norm(r.q_bn, axis=1), 1.0, atol=1e-6)  # unit quaternions
    assert np.isfinite(r.landing_position_error) and np.isfinite(r.landing_speed)


def test_powered_phase_burns_fuel():
    """Below the switch altitude the engine fires => mass decreases."""
    r = _run(["scenario.ic.r_ned=[0,0,-1500]", "scenario.ic.v_ned=[0,0,80]"], t_max=15.0)
    assert r.m[-1] < r.m[0]  # propellant consumed in powered descent


def test_slosh_does_not_break_loop():
    r = _run(["plant/slosh=springmass"])
    assert np.all(np.isfinite(r.r_ned)) and np.all(np.isfinite(r.w_body))


def test_fault_does_not_break_loop():
    r = _run(["fault=tvc_bias"])
    assert np.all(np.isfinite(r.r_ned))


def test_pdg_module_runs():
    """Module test: start at the landing burn alone (start_phase=powered)."""
    r = _run(["scenario=pdg"], t_max=30.0, ctrl_dt=0.05)
    assert np.all(np.isfinite(r.r_ned)) and np.all(np.isfinite(r.v_ned))
    assert r.phase[0] == int(
        __import__("core.interfaces", fromlist=["Phase"]).Phase.POWERED_DESCENT
    )
    assert (-r.r_ned[:, 2])[-1] < (-r.r_ned[:, 2])[0]  # descends


def test_reentry_burn_decelerates():
    """The symmetric 3-engine reentry burn (octaweb) decelerates the entry; runs finite. Pins the
    USSA-1976 atmosphere (the convention for reentry — exp over-predicts density ~3.4x at 60 km)
    so the Mach-5.7 entry is also covered by the extended mach_max=8 table."""
    ov = [
        "scenario=reentry_to_landing",
        "plant/vehicle=falcon9_octaweb",
        "environment/atmosphere=ussa_poly",
    ]
    r = _run(ov, t_max=40.0, ctrl_dt=0.1)
    assert np.all(np.isfinite(r.r_ned)) and np.all(np.isfinite(r.v_ned))
    assert r.phase[0] == int(Phase.REENTRY_BURN)  # latched start at the reentry burn
    speed = np.linalg.norm(r.v_ned, axis=1)
    assert speed[-1] < 0.6 * speed[0]  # full 3-engine retro sheds entry velocity
    # entry Mach (1800 m/s @ ~315 m/s sound speed ≈ 5.7) is inside the extended table, not clamped
    mach0 = np.asarray(r.aux["per_tick"][0]["mach"])
    assert 5.0 < float(mach0) < 8.0


def test_open_loop_batch_native_sessions():
    bundle = build_bundle(["environment/wind=dryden"])
    cfg = compose_cfg(["environment/wind=dryden", "scenario.dispersion.enable=true"])
    keys = episode_keys(0, 8)
    states = [initial_state(ic_from_scenario(cfg.scenario, k), bundle.params) for k in keys]
    cmds = open_loop_schedule(
        2000,
        float(cfg.plant_dt),
        n_fins=bundle.meta["n_fins"],
        n_engines=bundle.meta["n_engines"],
        n_gimbal=bundle.meta["n_gimbal"],
    )

    finals = []
    for state in states:
        session = bundle.native.create_session(state)
        snapshot = None
        for index in range(cmds.throttle.shape[0]):
            command = ActuatorCommand(
                throttle=cmds.throttle[index],
                gimbal=cmds.gimbal[index],
                fin=cmds.fin[index],
                rcs=cmds.rcs[index],
            )
            snapshot, _ = session.advance(command, 1)
        finals.append(snapshot)
    final_r = np.stack([state["r_ned"] for state in finals])
    final_v = np.stack([state["v_ned"] for state in finals])
    assert final_r.shape == (8, 3)
    assert not np.any(np.isnan(final_v))
    # dispersion present: trajectories differ under wind + IC dispersion
    assert np.std(final_r[:, 0]) > 0.0

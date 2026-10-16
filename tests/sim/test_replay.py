"""Replay export: the viz leg-deploy trigger time is the configured deploy altitude crossing."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import build_bundle, compose_cfg

from gnc.build import build_gnc
from gnc.navigation.identity import IdentityNav
from plant.dynamics import ic_from_scenario, initial_state
from sim.loop import run_closed_loop
from sim.replay import build_replay_doc


@pytest.fixture(scope="module")
def episode():
    """One short powered-descent closed-loop rollout, shared across the replay tests."""
    cfg = compose_cfg(["scenario=pdg"])
    bundle = build_bundle(["scenario=pdg"])
    state0 = initial_state(ic_from_scenario(cfg.scenario, 0), bundle.params)
    h = cfg.scenario.horizon
    result = run_closed_loop(
        bundle,
        state0,
        build_gnc(cfg, bundle.params),
        IdentityNav(),
        t_max=h.t_max,
        ctrl_dt=h.ctrl_dt,
        plant_dt=cfg.plant_dt,
        nav_dt=h.nav_dt,
        guidance_dt=h.guidance_dt,
        log_dt=h.log_dt,
    )
    return result, bundle


def _alt(result):
    return np.array([float(tick["altitude"]) for tick in result.aux["per_tick"]])


def test_leg_deploy_is_first_below_configured_altitude(episode):
    result, bundle = episode
    cfg = compose_cfg(["scenario=pdg", "run.leg_deploy_alt_m=50"])
    t = build_replay_doc(result, bundle, cfg)["meta"]["leg_deploy"]["t"]
    alt = _alt(result)
    first = int(np.flatnonzero(alt < 50.0)[0])  # first tick below the deploy altitude
    assert t is not None
    assert np.isclose(t, float(result.t[first]))
    assert 0.0 < t <= float(result.t[-1])


def test_leg_deploy_altitude_is_cli_overridable(episode):
    result, bundle = episode
    t_lo = build_replay_doc(
        result, bundle, compose_cfg(["scenario=pdg", "run.leg_deploy_alt_m=30"])
    )["meta"]["leg_deploy"]["t"]
    t_hi = build_replay_doc(
        result, bundle, compose_cfg(["scenario=pdg", "run.leg_deploy_alt_m=200"])
    )["meta"]["leg_deploy"]["t"]
    assert t_hi <= t_lo  # a higher deploy altitude is crossed earlier in the descent

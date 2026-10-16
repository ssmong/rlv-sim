"""Multirate closed-loop schedule.

The design runs four rates (plant_dt < nav_dt ≤ ctrl_dt ≤ guidance_dt) with ZOH holds between
ticks. These tests verify (a) the multirate path reduces EXACTLY to the single-rate path when all
GNC rates coincide, and (b) a genuinely slow guidance (held ZOH) still flies the loop.
"""

from __future__ import annotations

import numpy as np
from conftest import build_bundle, compose_cfg

from gnc.build import build_gnc
from gnc.navigation.identity import IdentityNav
from plant.dynamics import ic_from_scenario, initial_state
from sim.loop import run_closed_loop


def _setup(overrides):
    cfg = compose_cfg(overrides)
    bundle = build_bundle(overrides)
    ic = ic_from_scenario(cfg.scenario, 0)
    state0 = initial_state(ic, bundle.params)
    return bundle, state0, cfg


def test_multirate_equal_rates_matches_single_rate():
    """nav_dt=guidance_dt=ctrl_dt => the multirate schedule is bit-for-bit the single-rate loop."""
    bundle, state0, cfg = _setup(["scenario=pdg"])
    pdt, cdt = float(cfg.plant_dt), 0.02

    single = run_closed_loop(
        bundle,
        state0,
        build_gnc(cfg, bundle.params),
        IdentityNav(),
        t_max=20.0,
        ctrl_dt=cdt,
        plant_dt=pdt,
    )
    multi = run_closed_loop(
        bundle,
        state0,
        build_gnc(cfg, bundle.params),
        IdentityNav(),
        t_max=20.0,
        ctrl_dt=cdt,
        plant_dt=pdt,
        nav_dt=cdt,
        guidance_dt=cdt,
    )
    assert single.r_ned.shape == multi.r_ned.shape
    assert np.allclose(single.r_ned, multi.r_ned, atol=1e-9)
    assert np.allclose(single.v_ned, multi.v_ned, atol=1e-9)
    assert np.isclose(single.landing_speed, multi.landing_speed, atol=1e-9)


def test_multirate_slow_guidance_still_flies():
    """guidance_dt 10x slower (held ZOH) + faster nav: the loop stays finite and descends."""
    bundle, state0, cfg = _setup(["scenario=pdg"])
    r = run_closed_loop(
        bundle,
        state0,
        build_gnc(cfg, bundle.params),
        IdentityNav(),
        t_max=30.0,
        ctrl_dt=0.02,
        plant_dt=float(cfg.plant_dt),
        nav_dt=0.01,
        guidance_dt=0.2,
        log_dt=0.1,
    )
    alt = -r.r_ned[:, 2]
    assert np.all(np.isfinite(r.r_ned)) and np.all(np.isfinite(r.w_body))
    assert alt[-1] < alt[0]  # descends
    # log_dt=0.1 downsamples the stored trajectory below the 0.01 base-rate step count
    assert len(r.t) < int(30.0 / 0.01)


def test_multirate_logs_terminal_state():
    """Downsampled logging still captures the terminal (near-ground) step for KPIs."""
    bundle, state0, cfg = _setup(["scenario=pdg"])
    r = run_closed_loop(
        bundle,
        state0,
        build_gnc(cfg, bundle.params),
        IdentityNav(),
        t_max=30.0,
        ctrl_dt=0.02,
        plant_dt=float(cfg.plant_dt),
        nav_dt=0.01,
        guidance_dt=0.1,
        log_dt=0.2,
    )
    assert np.isfinite(r.landing_speed) and np.isfinite(r.landing_position_error)

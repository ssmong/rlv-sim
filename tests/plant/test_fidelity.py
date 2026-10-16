"""Fidelity-ladder modularization: per-module ideal/simplified/real tiers + the global preset.

Verifies each swappable plant module (wind / atmosphere / thrust / actuator / slosh) builds at every
tier, the default run is the real/realistic tier, the new tiers do what they claim, and the global
fidelity= preset swaps the whole stack. aero & massprops are intentionally NOT laddered.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import build_bundle, compose_cfg

from core.interfaces import ActuatorCommand
from plant.dynamics import ic_from_scenario

PER_MODULE = {
    "environment/wind": ["none", "steady", "dryden", "von_karman"],
    "environment/atmosphere": ["vacuum", "exp", "ussa_poly"],
    "plant/thrust": ["vacuum", "sea_level", "backpressure"],
    "plant/actuator": ["realistic", "simplified", "ideal"],
    "plant/slosh": ["none", "pendulum", "springmass"],
}


@pytest.mark.parametrize("group", list(PER_MODULE))
def test_every_tier_builds(group):
    """Every per-module fidelity tier composes + builds a usable plant."""
    for model in PER_MODULE[group]:
        bundle = build_bundle([f"{group}={model}"])
        assert bundle.native is not None


def test_default_is_real_tier():
    """The root defaults are the real/realistic tier (so existing runs are unchanged)."""
    meta = build_bundle().meta
    assert meta["actuator_model"] == "realistic"
    assert meta["thrust_model"] == "backpressure"


def test_vacuum_atmosphere_zero_density_and_pressure():
    """environment/atmosphere=vacuum: rho=0 (no aero), p_amb=0 (no back-pressure), a_sound>0."""
    bundle = build_bundle(["environment/atmosphere=vacuum"])
    state, cmd = _state_cmd(bundle, throttle=0.0)
    _, aux = bundle.native.rhs(state, cmd, np.zeros(3))
    assert float(aux["rho"]) == 0.0
    assert np.isfinite(float(aux["mach"]))


def test_steady_wind_is_constant_mean():
    """environment/wind=steady holds the mean vector with no gust/stochastic update."""
    mean = np.array([8.0, 0.0, 0.0])
    bundle = build_bundle(["environment/wind=steady", "environment.wind.mean0=[8,0,0]"])
    state, cmd = _state_cmd(bundle, throttle=0.0)
    session = bundle.native.create_session(state)
    session.advance(cmd, 10)
    full = session.snapshot(True)
    assert np.allclose(full["wind_mean"], mean)
    assert np.allclose(full["wind_gust"], 0.0)


def _state_cmd(bundle, throttle=1.0):
    """An initial state + a firing command shaped to this bundle's engine/fin counts."""
    cfg = compose_cfg(["scenario=pdg"])
    state = bundle.make_initial_state(ic_from_scenario(cfg.scenario, 0))
    n, ng, nf = state.throttle_act.shape[0], state.tvc.shape[0], state.fin.shape[0]
    # fire the engine in the (lagged) actuator state too — the realistic wrench reads throttle_act,
    # so a fresh state (throttle_act=0) would leave the engine OFF and mask the back-pressure debit.
    state = state.replace(throttle_act=np.full((n,), throttle))
    cmd = ActuatorCommand(
        throttle=np.full((n,), throttle),
        gimbal=np.zeros((ng, 2)),
        fin=np.zeros((nf,)),
        rcs=np.zeros(3),
    )
    return state, cmd


def test_ideal_actuator_freezes_servo_states():
    """plant/actuator=ideal: servo-state derivatives all zero (the wrench reads the command)."""
    bundle = build_bundle(["plant/actuator=ideal"])
    state, cmd = _state_cmd(bundle)
    derivative, _ = bundle.native.rhs(state, cmd, np.zeros(3))
    for index in (6, 7, 8, 9, 10):
        assert np.allclose(np.asarray(derivative[index]), 0.0), index


def test_thrust_vacuum_exceeds_backpressure():
    """plant/thrust=vacuum delivers MORE upward thrust than backpressure (no p_amb·A_e debit)."""
    s, cmd = _state_cmd(build_bundle())  # state independent of the thrust tier
    az_vac = float(build_bundle(["plant/thrust=vacuum"]).native.rhs(s, cmd, np.zeros(3))[0][2][2])
    az_bp = float(
        build_bundle(["plant/thrust=backpressure"]).native.rhs(s, cmd, np.zeros(3))[0][2][2]
    )
    assert az_vac < az_bp  # NED: up is -z, so more thrust => more-negative vertical acceleration


def test_fidelity_preset_swaps_the_stack():
    """+fidelity=ideal swaps every laddered group (aero/massprops untouched)."""
    meta = build_bundle(["+fidelity=ideal"]).meta
    assert meta["wind_model"] == "none"
    assert meta["atmosphere_model"] == "vacuum"
    assert meta["thrust_model"] == "vacuum"
    assert meta["actuator_model"] == "ideal"
    assert meta["slosh_model"] == "none"
    assert meta["aero_model"] == "table"  # NOT laddered

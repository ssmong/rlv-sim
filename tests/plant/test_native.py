"""Native C++ plant physics and Python/C++ boundary contract."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from conftest import build_bundle, compose_cfg
from omegaconf.errors import MissingMandatoryValue

from core.interfaces import ActuatorCommand
from plant.dynamics import ic_from_scenario


def _case(overrides):
    cfg = compose_cfg(overrides)
    bundle = build_bundle(overrides)
    state = bundle.make_initial_state(ic_from_scenario(cfg.scenario, 19))
    state = state.replace(
        # Keep the point away from wet/dry clipping endpoints.
        m=state.m - 100.0,
        w_body=np.array([0.03, -0.02, 0.04]),
        fin=np.linspace(-0.05, 0.05, bundle.meta["n_fins"]),
        fin_rate=np.linspace(0.01, -0.01, bundle.meta["n_fins"]),
        throttle_act=np.full((bundle.meta["n_engines"],), 0.72),
        tvc=np.tile(np.array([[0.01, -0.015]]), (bundle.meta["n_gimbal"], 1)),
    )
    command = ActuatorCommand(
        throttle=np.full((bundle.meta["n_engines"],), 0.65),
        gimbal=np.tile(np.array([[-0.02, 0.01]]), (bundle.meta["n_gimbal"], 1)),
        fin=np.linspace(0.08, -0.08, bundle.meta["n_fins"]),
        rcs=np.array([0.2, -0.1, 0.3]),
    )
    return bundle, state, command


@pytest.mark.parametrize(
    "overrides",
    [
        [],
        ["environment/atmosphere=ussa_poly", "plant/slosh=pendulum"],
        ["plant/aero=linear", "plant/actuator=simplified"],
        [
            "plant/vehicle=falcon9_octaweb",
            "scenario=reentry_to_landing",
            "environment/atmosphere=ussa_poly",
        ],
        ["environment/earth=rotating", "environment.earth.latitude_deg=37.5"],
    ],
)
def test_native_rhs_and_step_diagnostics_are_consistent(overrides):
    bundle, state, command = _case(overrides)
    wind = np.array([4.0, -2.0, 0.5])
    native_derivative, native_aux = bundle.native.rhs(state, command, wind)
    _, native_step_aux = bundle.native.step(state, command, wind)
    for derivative in native_derivative:
        assert np.all(np.isfinite(np.asarray(derivative)))
    for key in (
        "altitude",
        "rho",
        "mach",
        "thrust_mag",
        "x_cg",
        "F_aero_body",
        "F_fin_body",
        "T_body",
        "M_body_total",
        "mdot",
    ):
        assert np.allclose(native_aux[key], native_step_aux[key], rtol=2e-10, atol=5e-8)


def test_session_boundary_returns_only_tick_consumers():
    bundle, state, command = _case([])
    session = bundle.native.create_session(state)
    snapshot, aux = session.advance(command, 10)

    assert set(snapshot) == {
        "r_ned",
        "v_ned",
        "q_bn",
        "w_body",
        "m",
        "throttle_act",
        "tvc",
        "fin",
        "done",
    }
    # These remain native-owned and therefore cannot accidentally cross the hot boundary.
    internal = {"tvc_rate", "fin_rate", "slosh", "wind_filter", "wind_mean", "wind_gust", "key"}
    assert not (internal & set(snapshot))
    assert "mach" in aux and "M_body_total" in aux


def test_native_step_and_stateful_session_match_without_wind():
    bundle, state, command = _case(["environment/wind=none"])
    native_state, native_aux = bundle.native.step(state, command, np.zeros(3))
    session = bundle.native.create_session(state)
    _, session_aux = session.advance(command, 1)
    session_state = session.snapshot(True)
    for key in (
        "t",
        "r_ned",
        "v_ned",
        "q_bn",
        "w_body",
        "m",
        "tvc",
        "tvc_rate",
        "throttle_act",
        "fin",
        "fin_rate",
        "rcs_fuel",
        "slosh",
        "done",
    ):
        assert np.allclose(native_state[key], session_state[key], rtol=2e-12, atol=5e-10)
    for key in ("mach", "thrust_mag", "x_cg", "mdot", "aero_alpha", "v_air_body"):
        assert np.allclose(native_aux[key], session_aux[key], rtol=2e-12, atol=5e-10)


def test_fixed_earth_is_default_and_has_no_rotation_acceleration():
    bundle, state, command = _case(["environment/atmosphere=vacuum", "plant/thrust=vacuum"])
    state = state.replace(
        r_ned=np.array([1200.0, -800.0, -30000.0]),
        v_ned=np.array([250.0, 400.0, -50.0]),
        throttle_act=np.zeros(bundle.meta["n_engines"]),
    )
    command = command.replace(
        throttle=np.zeros(bundle.meta["n_engines"]),
        gimbal=np.zeros((bundle.meta["n_gimbal"], 2)),
        fin=np.zeros(bundle.meta["n_fins"]),
        rcs=np.zeros(3),
    )
    derivative, aux = bundle.native.rhs(state, command, np.zeros(3))

    assert bundle.meta["earth_model"] == "fixed"
    assert np.array_equal(np.asarray(bundle.params.earth.omega_ned), np.zeros(3))
    assert np.array_equal(aux["a_coriolis_ned"], np.zeros(3))
    assert np.array_equal(aux["a_centrifugal_ned"], np.zeros(3))
    assert np.array_equal(derivative[2], np.asarray(bundle.params.g_ned))


def test_rotating_earth_equator_terms_match_closed_form():
    rate = 7.2921150e-5
    speed_east = 400.0
    altitude = 30000.0
    overrides = [
        "environment/earth=rotating",
        "environment.earth.latitude_deg=0.0",
        "environment/atmosphere=vacuum",
        "plant/thrust=vacuum",
    ]
    bundle, state, command = _case(overrides)
    state = state.replace(
        r_ned=np.array([0.0, 0.0, -altitude]),
        v_ned=np.array([0.0, speed_east, 0.0]),
        throttle_act=np.zeros(bundle.meta["n_engines"]),
    )
    command = command.replace(
        throttle=np.zeros(bundle.meta["n_engines"]),
        gimbal=np.zeros((bundle.meta["n_gimbal"], 2)),
        fin=np.zeros(bundle.meta["n_fins"]),
        rcs=np.zeros(3),
    )

    native_derivative, aux = bundle.native.rhs(state, command, np.zeros(3))
    expected_coriolis = np.array([0.0, 0.0, -2.0 * rate * speed_east])
    expected_centrifugal = np.array([0.0, 0.0, -(rate**2) * altitude])
    expected_acceleration = (
        np.asarray(bundle.params.g_ned) + expected_coriolis + expected_centrifugal
    )

    assert bundle.meta["earth_model"] == "rotating"
    assert np.allclose(aux["a_coriolis_ned"], expected_coriolis, rtol=1e-14, atol=1e-14)
    assert np.allclose(aux["a_centrifugal_ned"], expected_centrifugal, rtol=1e-14, atol=1e-14)
    assert np.allclose(native_derivative[2], expected_acceleration, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("environment.earth.model=invalid", "unknown Earth model"),
        ("environment.earth.latitude_deg=91", "latitude_deg"),
        ("environment.earth.rotation_rate_rad_s=-1", "rotation_rate_rad_s"),
    ],
)
def test_invalid_earth_configuration_fails_loudly(override, message):
    with pytest.raises(ValueError, match=message):
        build_bundle([override])


def test_rotating_earth_requires_explicit_site_latitude():
    with pytest.raises(MissingMandatoryValue, match="latitude_deg"):
        build_bundle(["environment/earth=rotating"])


def test_sessions_are_isolated_for_parallel_episodes():
    bundle, state, command = _case([])
    first = bundle.native.create_session(state)
    second = bundle.native.create_session(state)
    first_state, _ = first.advance(command, 7)
    second_state, _ = second.advance(command, 3)
    assert first.snapshot(True)["t"] > second.snapshot(True)["t"]

    second_state, _ = second.advance(command, 4)
    for key in ("r_ned", "v_ned", "q_bn", "w_body", "m"):
        assert np.allclose(first_state[key], second_state[key], rtol=1e-13, atol=1e-13)


def test_native_command_shape_mismatch_fails_before_integration():
    bundle, state, command = _case([])
    session = bundle.native.create_session(state)
    bad = SimpleNamespace(
        throttle=np.zeros(bundle.meta["n_engines"] + 1),
        gimbal=command.gimbal,
        fin=command.fin,
        rcs=command.rcs,
    )
    with pytest.raises(ValueError, match="throttle count"):
        session.advance(bad, 1)


def test_native_wind_rng_is_reproducible_and_has_configured_stationary_variance():
    overrides = [
        "environment/wind=dryden",
        "environment/atmosphere=vacuum",
        "plant/thrust=vacuum",
    ]
    bundle, state, command = _case(overrides)
    state = state.replace(
        r_ned=np.array([0.0, 0.0, -1.0e6]),
        v_ned=np.array([200.0, 0.0, 0.0]),
        throttle_act=np.zeros(bundle.meta["n_engines"]),
    )
    command = command.replace(
        throttle=np.zeros(bundle.meta["n_engines"]),
        gimbal=np.zeros((bundle.meta["n_gimbal"], 2)),
        fin=np.zeros(bundle.meta["n_fins"]),
        rcs=np.zeros(3),
    )

    first = bundle.native.create_session(state)
    second = bundle.native.create_session(state)
    samples = []
    for index in range(200):
        first_public, _ = first.advance(command, 500)
        second_public, _ = second.advance(command, 500)
        if index >= 20:  # discard gust-filter spin-up
            samples.append(np.asarray(first.snapshot(True)["wind_gust"]))

    for key in ("r_ned", "v_ned", "q_bn", "w_body", "m"):
        assert np.array_equal(first_public[key], second_public[key])
    samples = np.asarray(samples)
    assert np.all(np.abs(samples.mean(axis=0)) < 0.6)
    assert np.allclose(
        samples.std(axis=0), np.asarray(bundle.params.wind.turb_sigma), rtol=0.2, atol=0.1
    )

    changed_seed = state.replace(key=np.array([0, 124], dtype=np.uint32))
    baseline = bundle.native.create_session(state)
    third = bundle.native.create_session(changed_seed)
    baseline.advance(command, 1000)
    third.advance(command, 1000)
    assert not np.array_equal(
        baseline.snapshot(True)["wind_gust"], third.snapshot(True)["wind_gust"]
    )

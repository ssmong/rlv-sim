"""Atmosphere fidelity anchors against the native implementation."""

from __future__ import annotations

import numpy as np
from conftest import build_bundle

FALCON_SET = ["vehicle_set=falcon9"]


def _environment(overrides, altitude):
    return build_bundle(overrides).native.environment_diagnostics(float(altitude))


def test_ussa_sea_level_and_tropopause():
    overrides = [*FALCON_SET, "environment/atmosphere=ussa_poly"]
    sea = _environment(overrides, 0.0)
    tropopause = _environment(overrides, 11000.0)
    assert np.isclose(float(sea["rho"]), 1.225, rtol=2e-3)
    assert np.isclose(float(sea["a_sound"]), 340.3, rtol=2e-3)
    assert np.isclose(float(tropopause["rho"]), 0.3639, rtol=5e-3)
    assert np.isclose(float(tropopause["a_sound"]), 295.07, rtol=5e-3)


def test_exp_speed_of_sound_is_temperature_varying():
    plant = build_bundle(FALCON_SET).native
    a0 = float(plant.environment_diagnostics(0.0)["a_sound"])
    a5 = float(plant.environment_diagnostics(5000.0)["a_sound"])
    a11 = float(plant.environment_diagnostics(11000.0)["a_sound"])
    a20 = float(plant.environment_diagnostics(20000.0)["a_sound"])
    assert np.isclose(a0, 340.3, rtol=2e-3)
    assert np.isclose(a11, 295.07, rtol=5e-3)
    assert a0 > a5 > a11
    assert np.isclose(a20, a11, rtol=1e-6)
    assert 5.5 < 1700.0 / a11 < 6.0


def test_ussa_vs_exp_at_60km_density_overprediction():
    ussa = _environment([*FALCON_SET, "environment/atmosphere=ussa_poly"], 60000.0)
    exp = _environment(FALCON_SET, 60000.0)
    ratio = float(exp["rho"]) / float(ussa["rho"])
    assert 3.0 < ratio < 3.8
    assert 305.0 < float(ussa["a_sound"]) < 325.0


def test_nonstd_day_hot_lowers_density_and_scales_dispersion():
    standard = build_bundle([*FALCON_SET, "environment/atmosphere=ussa_poly"])
    hot = build_bundle([*FALCON_SET, "environment/atmosphere=nonstd_day"])
    for altitude in (0.0, 5000.0, 30000.0):
        env_standard = standard.native.environment_diagnostics(altitude)
        env_hot = hot.native.environment_diagnostics(altitude)
        assert float(env_hot["rho"]) < float(env_standard["rho"])
        assert float(env_hot["a_sound"]) > float(env_standard["a_sound"])

    dispersed = build_bundle(
        [
            *FALCON_SET,
            "environment/atmosphere=ussa_poly",
            "environment.atmosphere.density_scale=1.1",
        ]
    )
    env_standard = standard.native.environment_diagnostics(0.0)
    env_dispersed = dispersed.native.environment_diagnostics(0.0)
    assert np.isclose(float(env_dispersed["rho"]), 1.1 * float(env_standard["rho"]), rtol=1e-9)
    assert float(env_dispersed["a_sound"]) == float(env_standard["a_sound"])

    cold = _environment(
        [
            *FALCON_SET,
            "environment/atmosphere=ussa_poly",
            "environment.atmosphere.delta_T=-30",
        ],
        0.0,
    )
    assert float(cold["rho"]) > float(env_standard["rho"])
    assert float(cold["a_sound"]) < float(env_standard["a_sound"])


def test_aero_table_covers_hypersonic_mach():
    bundle = build_bundle(FALCON_SET)
    assert float(bundle.params.aero.mach_grid[-1]) >= 8.0 - 1e-9
    sound = float(bundle.native.environment_diagnostics(0.0)["a_sound"])

    def cd(mach):
        result = bundle.native.aero_wrench_diagnostics(
            [-mach * sound, 0.0, 0.0], [0.0, 0.0, 0.0], 40000.0, 0.0
        )
        return float(result["cd"])

    assert np.isclose(cd(5.7), cd(8.0), rtol=0.05)
    assert 1.6 < cd(5.7) < 2.0
    assert cd(5.7) > cd(0.3) + 0.3

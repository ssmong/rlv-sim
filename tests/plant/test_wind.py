"""C++ turbulence-model statistics and deterministic sampling tests."""

from __future__ import annotations

import numpy as np
from conftest import build_bundle

MODELS = ("dryden", "von_karman")


def _bundle(model, dt=0.01, overrides=()):
    return build_bundle([f"environment/wind={model}", f"plant_dt={dt}", *overrides])


def _chain(bundle, speed, n, seed):
    result = bundle.native.wind_diagnostics(speed, n, seed)
    return np.asarray(result["mean"]), np.asarray(result["gust"])


def _spectral_bundle(model, dt, length, sigma):
    return _bundle(
        model,
        dt,
        (
            "environment.wind.mean0=[0,0,0]",
            "environment.wind.ou_sigma=[0,0,0]",
            f"environment.wind.turb_L=[{length},{length},{length}]",
            f"environment.wind.turb_sigma=[{sigma},{sigma},{sigma}]",
        ),
    )


def _band_fraction(x, dt, f_lo, f_hi, nseg=16):
    """Fraction of non-DC gust power in a frequency band."""
    x = np.asarray(x) - np.mean(x)
    segment = len(x) // nseg
    window = np.hanning(segment)
    frequency = np.fft.rfftfreq(segment, dt)
    power = np.zeros(frequency.shape[0])
    for index in range(nseg):
        start = index * segment
        power += np.abs(np.fft.rfft(x[start : start + segment] * window)) ** 2
    band = (frequency >= f_lo) & (frequency <= f_hi)
    return float(power[band].sum() / power[1:].sum())


def test_stationary_variance():
    """Exact-variance discretization reproduces the requested variance per axis."""
    n = 80000
    for model in MODELS:
        bundle = _bundle(model)
        estimates = [
            np.var(_chain(bundle, 400.0, n, seed)[1][n // 5 :], axis=0) for seed in range(4)
        ]
        variance = np.mean(estimates, axis=0)
        target = np.asarray(bundle.params.wind.turb_sigma) ** 2
        assert np.allclose(variance, target, rtol=0.1), (
            f"{model}: variance={variance}, target={target}"
        )


def test_ou_mean_variance():
    bundle = _bundle("dryden", 0.05, ("environment.wind.ou_tau=2.0",))
    mean, _ = _chain(bundle, 200.0, 80000, 1)
    variance = np.var(mean[80000 // 5 :], axis=0)
    assert np.allclose(variance, np.asarray(bundle.params.wind.ou_sigma) ** 2, rtol=0.1)


def test_ou_mean_persistence():
    bundle = _bundle(
        "dryden",
        0.05,
        ("environment.wind.ou_tau=2.0", "environment.wind.mean0=[5,-4,3]"),
    )
    mean, _ = _chain(bundle, 200.0, 80000, 2)
    sample_mean = np.mean(mean[80000 // 5 :], axis=0)
    expected = np.asarray(bundle.params.wind.mean0) * np.array([1.0, 1.0, 0.0])
    assert np.allclose(sample_mean, expected, atol=0.3)
    assert abs(sample_mean[2]) < 0.3


def test_determinism():
    for model in MODELS:
        bundle = _bundle(model)
        first = _chain(bundle, 150.0, 1000, 7)
        second = _chain(bundle, 150.0, 1000, 7)
        assert np.array_equal(first[1], second[1])


def test_transverse_anisotropy():
    dt, speed, n = 0.01, 60.0, 1 << 16
    for model in MODELS:
        bundle = _spectral_bundle(model, dt, length=300.0, sigma=2.0)
        _, gust = _chain(bundle, speed, n, 3)
        burn = n // 5
        longitudinal = _band_fraction(gust[burn:, 0], dt, 0.5, 6.0)
        transverse = _band_fraction(gust[burn:, 1], dt, 0.5, 6.0)
        assert transverse > 1.2 * longitudinal, (
            f"{model}: transverse={transverse}, longitudinal={longitudinal}"
        )


def test_von_karman_tail_shallower_than_dryden():
    dt, speed, n = 0.01, 60.0, 1 << 16
    burn = n // 5
    von_karman = _spectral_bundle("von_karman", dt, length=30.0, sigma=2.0)
    dryden = _spectral_bundle("dryden", dt, length=30.0, sigma=2.0)
    _, gust_vk = _chain(von_karman, speed, n, 4)
    _, gust_dryden = _chain(dryden, speed, n, 4)
    high_vk = _band_fraction(gust_vk[burn:, 1], dt, 2.0, 8.0)
    high_dryden = _band_fraction(gust_dryden[burn:, 1], dt, 2.0, 8.0)
    assert high_vk > 1.1 * high_dryden, f"von Karman tail={high_vk}, Dryden tail={high_dryden}"


def test_wind_none_is_zero():
    mean, gust = _chain(_bundle("none"), 200.0, 10, 0)
    assert np.allclose(mean + gust, 0.0)


def test_wind_steady_is_constant_mean():
    bundle = _bundle("steady")
    mean, gust = _chain(bundle, 200.0, 10, 0)
    assert np.allclose(mean, np.asarray(bundle.params.wind.mean0))
    assert np.allclose(gust, 0.0)

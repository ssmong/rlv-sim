"""Deterministic NumPy seed derivation tests."""

from __future__ import annotations

import numpy as np

from core.seeding import episode_keys, fold_in, rng_from_seed, seed_value, seed_words


def test_seed_words_roundtrip_full_uint64():
    seed = 0xFEDCBA9876543210
    assert seed_value(seed_words(seed)) == seed


def test_fold_in_is_deterministic_and_separates_channels():
    first = fold_in(42, 7)
    assert first == fold_in(42, 7)
    assert first != fold_in(42, 8)
    assert first != fold_in(43, 7)


def test_episode_keys_and_generators_are_reproducible():
    keys = episode_keys(9, 4)
    assert keys.dtype == np.uint64
    assert len(np.unique(keys)) == 4
    first = rng_from_seed(keys[0]).normal(size=8)
    second = rng_from_seed(episode_keys(9, 4)[0]).normal(size=8)
    assert np.array_equal(first, second)

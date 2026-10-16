"""Deterministic integer seed derivation and run manifests."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_MASK64 = (1 << 64) - 1


def seed_value(seed) -> int:
    """Normalize an integer or one/two-word array-like seed to one unsigned 64-bit value."""
    values = np.asarray(seed, dtype=np.uint64).reshape(-1)
    if values.size == 1:
        return int(values[0])
    if values.size == 2:
        return ((int(values[0]) << 32) | int(values[1])) & _MASK64
    raise ValueError("seed must be an integer or contain one/two integer words")


def master_key(seed: int) -> int:
    """Normalized root seed."""
    return int(seed) & _MASK64


def fold_in(seed, data: int) -> int:
    """Mix an integer label into a seed with SplitMix64."""
    value = (seed_value(seed) ^ (int(data) & _MASK64)) + 0x9E3779B97F4A7C15
    value &= _MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def episode_keys(seed, n: int) -> np.ndarray:
    """Return ``n`` independent deterministic episode seeds."""
    if n < 0:
        raise ValueError("n must be non-negative")
    return np.asarray([fold_in(seed, index) for index in range(n)], dtype=np.uint64)


def rng_from_seed(seed) -> np.random.Generator:
    """Construct a NumPy generator from normalized seed material."""
    return np.random.default_rng(seed_value(seed))


def seed_words(seed) -> np.ndarray:
    """Two exact uint32 words for the native C++ session boundary."""
    value = seed_value(seed)
    return np.array([value >> 32, value & 0xFFFFFFFF], dtype=np.uint32)


def config_hash(resolved_yaml: str) -> str:
    """Stable SHA-256 (first 16 hex) of resolved configuration text."""
    return hashlib.sha256(resolved_yaml.encode("utf-8")).hexdigest()[:16]


def write_manifest(
    out_dir: Path,
    seed: int,
    resolved_yaml: str,
    extra: dict | None = None,
) -> Path:
    """Write ``manifest.json`` capturing seed, config hash, platform, and versions."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(seed),
        "config_sha256_16": config_hash(resolved_yaml),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    if extra:
        manifest.update(extra)
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path

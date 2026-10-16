"""Benchmark the single native production plant implementation.

Usage:
    python scripts/benchmark_native.py --steps 2000 --repeats 7

Timing includes command conversion and the one tick-end snapshot/aux transfer.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir

from core.interfaces import ActuatorCommand
from plant.dynamics import build_plant, ic_from_scenario


def _elapsed_ms(call):
    start = time.perf_counter()
    result = call()
    return (time.perf_counter() - start) * 1e3, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if args.steps <= 0 or args.repeats <= 0:
        parser.error("--steps and --repeats must be positive")

    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="config", overrides=["environment/wind=none"])
    bundle = build_plant(cfg)
    state = bundle.make_initial_state(ic_from_scenario(cfg.scenario, 0))
    command = ActuatorCommand(
        throttle=np.zeros(bundle.meta["n_engines"]),
        gimbal=np.zeros((bundle.meta["n_gimbal"], 2)),
        fin=np.zeros(bundle.meta["n_fins"]),
        rcs=np.zeros(3),
    )

    create_ms, _ = _elapsed_ms(lambda: bundle.native.create_session(state))
    samples = []
    for _ in range(args.repeats):
        session = bundle.native.create_session(state)
        elapsed_ms, _ = _elapsed_ms(
            lambda current_session=session: current_session.advance(command, args.steps)
        )
        samples.append(elapsed_ms)
    native_ms = float(np.median(samples))
    report = {
        "steps": args.steps,
        "repeats": args.repeats,
        "native_session_create_ms": create_ms,
        "native_advance_median_ms": native_ms,
        "native_advance_min_ms": float(np.min(samples)),
        "native_advance_p95_ms": float(np.percentile(samples, 95)),
        "native_median_us_per_step": native_ms * 1000.0 / args.steps,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

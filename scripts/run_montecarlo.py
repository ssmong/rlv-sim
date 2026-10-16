"""Closed-loop Monte Carlo via CPU multiprocessing.

Each worker builds one native plant and GNC stack in its initializer; dispersion and wind are
seeded independently per episode.

Usage:
    python scripts/run_montecarlo.py run=montecarlo environment/wind=dryden run.n_episodes=64
"""

from __future__ import annotations

import csv
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import multiprocessing as mp
from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf

from core.seeding import write_manifest
from gnc.build import build_gnc, build_nav
from plant.dynamics import build_plant, ic_from_scenario, initial_state
from sim.aggregate import aggregate_episodes
from sim.loop import run_closed_loop

_WORKER: dict = {}


def _worker_config(cfg) -> dict:
    container = OmegaConf.to_container(cfg, resolve=True)
    aero = container["plant"]["aero"]
    if aero["backend"] == "file":
        aero["file"]["path"] = to_absolute_path(str(aero["file"]["path"]))
    return container


def _init(cfg_container: dict):
    cfg = OmegaConf.create(cfg_container)
    cfg = OmegaConf.merge(cfg, {"scenario": {"dispersion": {"enable": True}}})
    bundle = build_plant(cfg)
    _WORKER["cfg"] = cfg
    _WORKER["bundle"] = bundle
    _WORKER["controller"] = build_gnc(cfg, bundle.params)
    _WORKER["nav"] = build_nav(cfg)


def _run_episode(seed: int) -> dict:
    cfg = _WORKER["cfg"]
    bundle = _WORKER["bundle"]
    ic = ic_from_scenario(cfg.scenario, seed)
    state0 = initial_state(ic, bundle.params)
    h = cfg.scenario.horizon
    r = run_closed_loop(
        bundle,
        state0,
        _WORKER["controller"],
        _WORKER["nav"],
        t_max=h.t_max,
        ctrl_dt=h.ctrl_dt,
        plant_dt=cfg.plant_dt,
        success_pos_tol=cfg.scenario.success.pos_tol,
        success_speed_tol=cfg.scenario.success.speed_tol,
        success_tilt_tol_deg=cfg.scenario.success.tilt_tol_deg,
        delay_ticks=cfg.gnc.delay_ticks,
        nav_dt=h.nav_dt,  # four-rate schedule
        guidance_dt=h.guidance_dt,
        log_dt=h.log_dt,
    )
    return {
        "seed": seed,
        "landing_position_error": r.landing_position_error,
        "landing_speed": r.landing_speed,
        "landing_speed_vertical": r.landing_speed_vertical,
        "landing_speed_horizontal": r.landing_speed_horizontal,
        "landing_rate": r.landing_rate,
        "landing_tilt": r.landing_tilt,
        "fuel_margin": r.fuel_margin,
        "success": bool(r.success),
        "termination_reason": r.termination_reason,
    }


class _KPI:
    """Lightweight EpisodeResult stand-in for the aggregator (KPI fields only)."""

    def __init__(self, d):
        self.landing_position_error = d["landing_position_error"]
        self.landing_speed = d["landing_speed"]
        self.landing_speed_vertical = d["landing_speed_vertical"]
        self.landing_speed_horizontal = d["landing_speed_horizontal"]
        self.landing_rate = d["landing_rate"]
        self.landing_tilt = d["landing_tilt"]
        self.fuel_margin = d["fuel_margin"]
        self.success = d["success"]
        self.termination_reason = d["termination_reason"]


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    out_dir = Path.cwd()
    resolved = OmegaConf.to_yaml(cfg, resolve=True)
    (out_dir / "resolved_config.yaml").write_text(resolved)
    write_manifest(out_dir, cfg.seed, resolved, extra={"mode": cfg.run.mode})

    n_ep = int(cfg.run.n_episodes)
    n_workers = int(cfg.run.n_workers)
    cfg_container = _worker_config(cfg)
    seeds = [int(cfg.seed) + i for i in range(n_ep)]

    ctx = mp.get_context("spawn")
    with ctx.Pool(n_workers, initializer=_init, initargs=(cfg_container,)) as pool:
        episodes = pool.map(_run_episode, seeds)

    summary = aggregate_episodes([_KPI(e) for e in episodes])
    with (out_dir / "mc_episodes.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=episodes[0])
        writer.writeheader()
        writer.writerows(episodes)
    (out_dir / "mc_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"output: {out_dir}")


if __name__ == "__main__":
    main()

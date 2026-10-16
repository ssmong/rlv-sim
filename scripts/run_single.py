"""Single closed-loop rollout.

Usage (from repo root, venv active):
    python scripts/run_single.py
    python scripts/run_single.py environment/wind=dryden fault=tvc_bias
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import hydra
import numpy as np
from _plotting import plot_episode, plot_plan_tracking
from omegaconf import OmegaConf

from core.seeding import write_manifest
from gnc.build import build_gnc, build_nav
from plant.dynamics import build_plant, ic_from_scenario, initial_state
from sim.loop import run_closed_loop
from sim.replay import build_replay_doc, write_replay


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    out_dir = Path.cwd()  # hydra.run.dir = results/<script>/<date>/<time>
    resolved = OmegaConf.to_yaml(cfg, resolve=True)
    (out_dir / "resolved_config.yaml").write_text(resolved)
    write_manifest(out_dir, cfg.seed, resolved, extra={"mode": cfg.run.mode})

    bundle = build_plant(cfg)
    ic = ic_from_scenario(cfg.scenario, int(cfg.seed))
    state0 = initial_state(ic, bundle.params)

    controller = build_gnc(cfg, bundle.params)
    nav = build_nav(cfg)

    h = cfg.scenario.horizon
    result = run_closed_loop(
        bundle,
        state0,
        controller,
        nav,
        t_max=h.t_max,
        ctrl_dt=h.ctrl_dt,
        plant_dt=cfg.plant_dt,
        success_pos_tol=cfg.scenario.success.pos_tol,
        success_speed_tol=cfg.scenario.success.speed_tol,
        success_tilt_tol_deg=cfg.scenario.success.tilt_tol_deg,
        delay_ticks=cfg.gnc.delay_ticks,
        nav_dt=h.nav_dt,  # four-rate schedule; None => single-rate
        guidance_dt=h.guidance_dt,
        log_dt=h.log_dt,
        progress=True,  # live bar+ETA on stderr (auto-off when not a TTY)
    )

    # Trajectory as a flat CSV (one row per logged tick). Vectors split into named columns; the
    # per-engine/gimbal/fin actuator channels widen to scalar columns (count follows the vehicle).
    throttle = np.asarray(result.throttle_act)  # (T, n_engines)
    tvc = np.asarray(result.tvc)  # (T, n_gimbal, 2) -> pitch/yaw
    fin = np.asarray(result.fin)  # (T, n_fins)
    act = {f"throttle{e}": throttle[:, e] for e in range(throttle.shape[1])}
    for g in range(tvc.shape[1]):
        act[f"tvc{g}_pitch"], act[f"tvc{g}_yaw"] = tvc[:, g, 0], tvc[:, g, 1]
    act.update({f"fin{i}": fin[:, i] for i in range(fin.shape[1])})
    cols = {
        "t": result.t,
        "r_N": result.r_ned[:, 0],
        "r_E": result.r_ned[:, 1],
        "r_D": result.r_ned[:, 2],
        "v_N": result.v_ned[:, 0],
        "v_E": result.v_ned[:, 1],
        "v_D": result.v_ned[:, 2],
        "q_w": result.q_bn[:, 0],
        "q_x": result.q_bn[:, 1],
        "q_y": result.q_bn[:, 2],
        "q_z": result.q_bn[:, 3],
        "w_x": result.w_body[:, 0],
        "w_y": result.w_body[:, 1],
        "w_z": result.w_body[:, 2],
        "m": result.m,
        "phase": result.phase,
        **act,
    }
    with (out_dir / "trajectory.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        writer.writerows(zip(*cols.values(), strict=True))
    kpi = {
        "termination_reason": result.termination_reason,
        "landing_class": result.landing_class,  # soft / hard / crash (3-tier contact quality)
        "touchdown_index": result.touchdown_index,
        "landing_position_error_m": result.landing_position_error,
        "landing_speed_mps": result.landing_speed,
        "landing_speed_vertical_mps": result.landing_speed_vertical,
        "landing_tilt_deg": float(np.rad2deg(result.landing_tilt)),
        "fuel_margin_kg": result.fuel_margin,
        "success": bool(result.success),
        "violations": result.violation_flags,
    }
    (out_dir / "kpi.json").write_text(json.dumps(kpi, indent=2))
    print(json.dumps(kpi, indent=2))

    # Solver telemetry (convex-solve timing mean/std/p95/p99/WCET + ignition/infeasible counts).
    solver = result.aux.get("solver", {})
    (out_dir / "solver_stats.json").write_text(json.dumps(solver, indent=2))
    print("solver:", json.dumps({k: v for k, v in solver.items() if k != "per_solve"}, indent=2))

    # Tracking-controller telemetry (per-call compute time; warm-up call reported separately).
    control = result.aux.get("control", {})
    (out_dir / "control_stats.json").write_text(json.dumps(control, indent=2))
    print("control:", json.dumps(control, indent=2))

    # Per-run 3D replay for the web viewer.
    if cfg.run.save_replay:
        doc = build_replay_doc(result, bundle, cfg)
        path = write_replay(doc, out_dir / "replay.json")
        print(f"replay: {path} ({doc['meta']['n_frames']} frames)")

    if cfg.run.save_plots:
        path = plot_episode(result, out_dir / "plots", title=f"single | {cfg.run.run_id}")
        print(f"plot: {path}")
        path = plot_plan_tracking(
            result, out_dir / "plots", title=f"plan vs flown | {cfg.run.run_id}"
        )
        if path is not None:
            print(f"plot: {path}")
    print(f"output: {out_dir}")


if __name__ == "__main__":
    main()

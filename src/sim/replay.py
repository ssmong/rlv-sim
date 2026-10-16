"""Serialise a closed-loop episode (`SimResult`) to a self-contained replay.json for the web viewer.

Used by ``scripts/run_single.py`` — auto-exported per run into the Hydra run dir (results/...).

The plant works in NED (z down); the three.js viewer is Y-up, so the frame change is done ONCE here
and the viewer stays dumb:

    viz = [North, Up, East] = [r_ned[0], -r_ned[2], r_ned[1]]      (proper rotation, det=+1)
    q_bv = q_vn ⊗ q_bn,  q_vn = [√2/2, √2/2, 0, 0]  (NED->viz, +90° about X)

Quaternions are scalar-first Hamilton in the plant; the JSON emits three.js order [x,y,z,w].
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# NED -> viz (Y-up, [North, Up, East]) is a +90° rotation about X: q_vn (scalar-first Hamilton).
_SQRT_HALF = float(np.sqrt(0.5))
Q_VN = np.array([_SQRT_HALF, _SQRT_HALF, 0.0, 0.0])

PHASE_NAMES = {0: "REENTRY_BURN", 1: "AERO_DESCENT", 2: "POWERED_BRAKE", 3: "POWERED_DESCENT"}


def _qmul(q, p):
    """Hamilton product of scalar-first quaternions q ⊗ p."""
    qw, qx, qy, qz = q
    pw, px, py, pz = p
    return np.array(
        [
            qw * pw - qx * px - qy * py - qz * pz,
            qw * px + qx * pw + qy * pz - qz * py,
            qw * py - qx * pz + qy * pw + qz * px,
            qw * pz + qx * py - qy * px + qz * pw,
        ]
    )


def _ned_to_viz_pos(r_ned):
    return [float(r_ned[0]), float(-r_ned[2]), float(r_ned[1])]


def _q_bn_to_three(q_bn):
    """body->NED (scalar-first) -> body->viz in three.js order [x,y,z,w]."""
    qbv = _qmul(Q_VN, np.asarray(q_bn))
    qbv = qbv / (np.linalg.norm(qbv) + 1e-12)
    return [float(qbv[1]), float(qbv[2]), float(qbv[3]), float(qbv[0])]


def _aux_series(result, key):
    """Pull a per-tick scalar/vector series out of result.aux['per_tick']."""
    return [np.asarray(tick[key]) for tick in result.aux["per_tick"]]


def build_replay_doc(result, bundle, cfg, max_frames: int = 1500) -> dict:
    """Serialise an already-computed closed-loop ``result`` to the viewer's replay document.

    Pure post-processing — it does NOT run the simulation (the caller does that and passes the
    ``result`` from ``sim.loop.run_closed_loop`` plus the ``bundle`` from ``plant.build_plant``).
    """
    h = cfg.scenario.horizon
    n = len(result.t)
    stride = max(1, n // max_frames)
    idx = list(range(0, n, stride))
    if idx[-1] != n - 1:
        idx.append(n - 1)  # always keep touchdown

    xcg = _aux_series(result, "x_cg")
    thrust_mag = _aux_series(result, "thrust_mag")
    # effective engine throttle fed to the wrench model (NOT state.throttle_act, which the ideal
    # actuator backend freezes at its initial value): altitude-independent throttle for the HUD %.
    throttle_eff = _aux_series(result, "throttle_eff")
    mach = _aux_series(result, "mach")
    altitude = _aux_series(result, "altitude")

    frames = []
    for i in idx:
        frames.append(
            {
                "t": float(result.t[i]),
                "p": _ned_to_viz_pos(result.r_ned[i]),
                "q": _q_bn_to_three(result.q_bn[i]),
                "xcg": float(xcg[i]),
                "thr": [float(x) for x in np.atleast_1d(throttle_eff[i])],
                "gim": [[float(a) for a in row] for row in np.atleast_2d(result.tvc[i])],
                "fin": [float(x) for x in np.atleast_1d(result.fin[i])],
                "thrust": float(thrust_mag[i]),
                "mach": float(mach[i]),
                "alt": float(altitude[i]),
                "mass": float(result.m[i]),
                "v": _ned_to_viz_pos(result.v_ned[i]),
                "speed": float(np.linalg.norm(result.v_ned[i])),
                "phase": int(result.phase[i]),
            }
        )

    # viz leg-deploy time: the first instant the booster descends below the configured altitude
    # (else null -> the viewer falls back to ~5 s before touchdown). cfg.run.leg_deploy_alt_m is
    # CLI-overridable (e.g. run.leg_deploy_alt_m=40); a higher altitude deploys the legs earlier.
    alt_full = np.array([float(a) for a in altitude])
    deploy_alt = float(cfg.run.leg_deploy_alt_m)
    below = np.flatnonzero(alt_full < deploy_alt)
    leg_deploy_t = float(result.t[int(below[0])]) if below.size else None

    p = bundle.params
    meta = bundle.meta
    gimbal_idx = [int(x) for x in np.asarray(meta["gimbal_idx"]).ravel()]
    vehicle = {
        "length": float(p.geom.length),
        "diameter": float(p.geom.d_ref),
        "x_eng": float(p.geom.x_eng),
        "n_engines": int(meta["n_engines"]),
        "n_gimbal": int(meta["n_gimbal"]),
        "n_fins": int(meta["n_fins"]),
        "gimbal_idx": gimbal_idx,
        # body-frame geometry (x toward nose, negative aft) so the viewer can place plumes/fins
        "engine_pos": [[float(c) for c in row] for row in np.asarray(p.engines.positions)],
        "fin_arm": [[float(c) for c in row] for row in np.asarray(p.aero.fin_arm)],
        "fin_normal": [[float(c) for c in row] for row in np.asarray(p.aero.fin_normal)],
        "x_cg_dry": float(p.mass.x_cg_dry),
        "x_cg_wet": float(p.mass.x_cg_wet),
        "tvc_deflection_max_rad": float(p.actuator.tvc_deflection_max_rad),
        "fin_deflection_max_rad": float(p.actuator.fin_deflection_max_rad),
        "thrust_vac": [float(x) for x in np.atleast_1d(p.engines.thrust_vac)],
    }
    kpi = {
        "termination_reason": int(result.termination_reason),
        "landing_position_error_m": float(result.landing_position_error),
        "landing_speed_mps": float(result.landing_speed),
        "landing_tilt_deg": float(np.rad2deg(result.landing_tilt)),
        "fuel_margin_kg": float(result.fuel_margin),
        "success": bool(result.success),
    }
    return {
        "meta": {
            "frame": "viz=[North,Up,East]; q=[x,y,z,w] body->viz",
            "scenario": str(cfg.scenario.name),
            "guidance": str(cfg.gnc.guidance.powered_descent_method),
            "run_id": str(cfg.run.run_id),
            "dt_nominal": float(h.ctrl_dt) * stride,
            "n_frames": len(frames),
            "duration_s": float(result.t[idx[-1]]),
            "leg_deploy": {"t": leg_deploy_t},  # viz reveals the landing legs at this time [s]
            "phase_names": PHASE_NAMES,
            "vehicle": vehicle,
            "kpi": kpi,
            "credits": "Procedural viewer (no external asset); glTF hook in viz/README.md.",
        },
        "frames": frames,
    }


def write_replay(doc: dict, path) -> Path:
    """Write the replay document as compact JSON, creating parent dirs. Returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, separators=(",", ":")))
    return path

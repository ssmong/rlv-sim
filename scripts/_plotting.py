"""Shared plotting for episode/batch diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core.frames import euler_from_quat


def _aux_series(result, key):
    out = []
    for a in result.aux.get("per_tick", []):
        out.append(float(np.asarray(a[key]))) if key in a else out.append(np.nan)
    return np.array(out)


def plot_episode(result, out_dir: Path, title: str = "episode"):
    """6-panel summary: trajectory, speed, attitude, rates, mass, thrust/aero."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t = result.t
    r, v, w, m = result.r_ned, result.v_ned, result.w_body, result.m
    alt = -r[:, 2]

    eul = np.array([euler_from_quat(q) for q in result.q_bn])  # (T,3) roll,pitch,yaw
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))

    ax[0, 0].plot(t, alt, label="altitude")
    ax[0, 0].plot(t, r[:, 0], label="north")
    ax[0, 0].plot(t, r[:, 1], label="east")
    ax[0, 0].set(xlabel="t [s]", ylabel="position [m]", title="trajectory")
    ax[0, 0].legend()
    ax[0, 0].grid(True)

    ax[0, 1].plot(t, v[:, 0], label="vN")
    ax[0, 1].plot(t, v[:, 1], label="vE")
    ax[0, 1].plot(t, v[:, 2], label="vD")
    ax[0, 1].plot(t, np.linalg.norm(v, axis=1), "k--", label="|v|")
    ax[0, 1].set(xlabel="t [s]", ylabel="velocity [m/s]", title="velocity NED")
    ax[0, 1].legend()
    ax[0, 1].grid(True)

    ax[0, 2].plot(t, np.rad2deg(eul[:, 0]), label="roll")
    ax[0, 2].plot(t, np.rad2deg(eul[:, 1]), label="pitch")
    ax[0, 2].plot(t, np.rad2deg(eul[:, 2]), label="yaw")
    ax[0, 2].set(xlabel="t [s]", ylabel="euler [deg]", title="attitude")
    ax[0, 2].legend()
    ax[0, 2].grid(True)

    ax[1, 0].plot(t, np.rad2deg(w[:, 0]), label="p")
    ax[1, 0].plot(t, np.rad2deg(w[:, 1]), label="q")
    ax[1, 0].plot(t, np.rad2deg(w[:, 2]), label="r")
    ax[1, 0].set(xlabel="t [s]", ylabel="body rate [deg/s]", title="angular rates")
    ax[1, 0].legend()
    ax[1, 0].grid(True)

    ax[1, 1].plot(t, m)
    ax[1, 1].set(xlabel="t [s]", ylabel="mass [kg]", title="mass")
    ax[1, 1].grid(True)

    thrust = _aux_series(result, "thrust_mag")
    mach = _aux_series(result, "mach")
    alpha = np.rad2deg(_aux_series(result, "aero_alpha"))
    ax2 = ax[1, 2].twinx()
    ax[1, 2].plot(t, thrust / 1e3, "b-", label="thrust [kN]")
    ax2.plot(t, mach, "r-", label="Mach")
    ax2.plot(t, alpha, "g-", label="alpha [deg]")
    ax[1, 2].set(xlabel="t [s]", ylabel="thrust [kN]", title="thrust / aero")
    ax[1, 2].legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax[1, 2].grid(True)

    fig.suptitle(title)
    fig.tight_layout()
    path = out_dir / "episode_summary.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_plan_tracking(result, out_dir: Path, title: str = "plan tracking"):
    """3x2 plan-vs-flown: rows = NED components, left col position, right col velocity.

    Red dashed = the ignition-committed guidance plan (nodes at t_ignition + tau);
    blue solid = the flown trajectory. Saved after every run for tracking diagnosis."""
    plan = result.aux.get("plan_ignition")
    t_ig = result.aux.get("t_ignition")
    if plan is None or t_ig is None:
        return None
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t = np.asarray(result.t)
    r = np.asarray(result.r_ned)
    v = np.asarray(result.v_ned)
    t_plan = float(t_ig) + np.linspace(0.0, float(plan["tf"]), int(plan["N"]) + 1)
    r_plan = np.asarray(plan["r"])
    v_plan = np.asarray(plan["v"])
    labels = ["N [m]", "E [m]", "D [m]"]
    fig, axes = plt.subplots(3, 2, figsize=(11, 8), sharex=True)
    for i in range(3):
        axes[i, 0].plot(t_plan, r_plan[:, i], "--", color=(1.0, 0.0, 0.0), label="plan")
        axes[i, 0].plot(t, r[:, i], "-", color=(0.0, 0.0, 1.0), label="flown")
        axes[i, 0].set_ylabel(f"r {labels[i]}")
        axes[i, 1].plot(t_plan, v_plan[:, i], "--", color=(1.0, 0.0, 0.0), label="plan")
        axes[i, 1].plot(t, v[:, i], "-", color=(0.0, 0.0, 1.0), label="flown")
        axes[i, 1].set_ylabel(f"v {labels[i]} /s")
        for ax in (axes[i, 0], axes[i, 1]):
            ax.grid(True, alpha=0.3)
    axes[0, 0].legend(loc="best")
    axes[2, 0].set_xlabel("t [s]")
    axes[2, 1].set_xlabel("t [s]")
    fig.suptitle(title)
    fig.tight_layout()
    path = out_dir / "plan_tracking.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path

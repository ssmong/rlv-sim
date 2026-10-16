"""Guidance (G) interface contract + helpers for reference attitudes and Commands.

A guidance law maps the estimated state to a reference Command (a thrust vector for
powered phases, or an attitude ``q_ref`` for engine-off phases). The powered-descent law
is selected by config — the active ``gnc/guidance/<name>.yaml`` sets ``law._target_`` to a
``build(ctx)`` factory, resolved by Hydra instantiate; the per-phase laws (reentry/aero) are
wired by ``gnc/build``. The helpers below build a well-formed Command — reuse them rather
than constructing one by hand.
"""

from __future__ import annotations

import numpy as np

from core.frames import quat_from_axis_angle, quat_from_euler, quat_mul
from core.interfaces import Command

_UP = np.array([0.0, 0.0, -1.0])  # NED up


def upright_quat():
    """Nose-up reference attitude: body-x points to NED up."""
    return quat_from_euler(0.0, np.pi / 2, 0.0)


def thrust_aligned_quat(thrust_ned, tilt_max_rad: float):
    """Reference attitude tilting body-x from NED-up TOWARD the commanded thrust direction.

    A booster directs sustained lateral thrust by tilting the BODY (thrust through the CG, zero
    steady torque); the gimbal is reserved for stabilization around this reference. The tilt from
    vertical is capped at ``tilt_max_rad``; a (near-)vertical or degenerate command returns the
    upright attitude. A hardcoded upright-hold would make a sustained lateral command saturate the
    gimbal against its own pointing torque and tumble.
    """
    t = np.asarray(thrust_ned)
    mag = np.linalg.norm(t)
    d = t / np.maximum(mag, 1e-9)
    s = np.dot(d, _UP)  # up-component of the commanded direction
    lat = d - s * _UP  # horizontal part
    lat_n = np.linalg.norm(lat)
    tilt = np.minimum(np.arctan2(lat_n, np.maximum(s, 1e-9)), tilt_max_rad)
    # Rotation axis in NED: up x d (horizontal, unit); degenerate (vertical) => keep upright.
    use_upright = (mag < 1e-6) | (lat_n < 1e-9) | (s <= 0.0)
    if use_upright:  # plain branch (numpy path): avoids the 0/0 axis normalize entirely
        return upright_quat()
    axis_raw = np.cross(_UP, d)
    axis = axis_raw / np.linalg.norm(axis_raw)
    q_tilt = quat_from_axis_angle(axis, tilt)
    return quat_mul(q_tilt, upright_quat())


def attitude_command(phase: int, q_ref, active_engines, w_ref=None) -> Command:
    """Attitude-mode Command (engine off / aero phase): thrust_vector = 0, all engines off."""
    return Command(
        phase=np.asarray(phase, dtype=np.int32),
        mode=np.asarray(1, dtype=np.int32),
        thrust_vector=np.zeros(3),
        q_ref=q_ref,
        tau_body=np.zeros(3),
        active_engines=active_engines,
        w_ref=np.zeros(3) if w_ref is None else np.asarray(w_ref, float),
    )


def thrust_command(phase: int, thrust_vector, q_ref, active_engines, w_ref=None) -> Command:
    """Thrust-vector-mode Command (powered phase): fires exactly the active engines."""
    return Command(
        phase=np.asarray(phase, dtype=np.int32),
        mode=np.asarray(0, dtype=np.int32),
        thrust_vector=thrust_vector,
        q_ref=q_ref,
        tau_body=np.zeros(3),
        active_engines=active_engines,
        w_ref=np.zeros(3) if w_ref is None else np.asarray(w_ref, float),
    )


def as_command(out, phase, active_engines, tilt_max_rad) -> Command:
    """Normalize a powered-descent law's return into a Command.

    A bare NED thrust vector becomes a thrust Command — ``phase`` + ``active_engines`` injected,
    attitude reference thrust-aligned and capped at ``tilt_max_rad``. An already-built Command
    (a custom attitude, or an engine-off cutoff) passes through unchanged."""
    if isinstance(out, Command):
        return out
    thrust = np.asarray(out, float)
    return thrust_command(phase, thrust, thrust_aligned_quat(thrust, tilt_max_rad), active_engines)

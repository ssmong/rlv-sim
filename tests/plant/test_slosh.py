"""Slosh-model physics anchors.

Covers two physics properties of the model:
  * the pendulum stiffness is SIGNED — a settled axial accel (a_axial>0) is a stable
    pendulum (bounded, decaying), a coast/decel (a_axial<0) is an inverted pendulum (diverges).
  * the offset tank is driven by the rotational acceleration ω̇×r + ω×(ω×r), so a pure
    body rate (no translational lateral accel) still excites slosh.
"""

from __future__ import annotations

import numpy as np
from conftest import build_bundle

from core.interfaces import ActuatorCommand

FALCON_SET = ["vehicle_set=falcon9"]


def _integrate(a_axial, n=4000, dt=0.002, xi0=0.05):
    """Free pendulum-slosh response under a constant axial specific force; return |ξ_y|(t)."""
    native = build_bundle([*FALCON_SET, "plant/slosh=pendulum"]).native
    state = np.array([xi0, 0.0, 0.0, 0.0])
    acceleration = np.array([a_axial, 0.0, 0.0])
    amp = []
    for _ in range(n):
        derivative = np.asarray(
            native.slosh_diagnostics(state, acceleration, 50000.0)["derivative"]
        )
        state += dt * derivative
        amp.append(abs(state[0]))
    return np.asarray(amp)


def test_pendulum_stable_when_settled():
    """a_axial>0 (powered descent, propellant settled aft) => stable, bounded, decaying."""
    amp = _integrate(a_axial=20.0)  # ~2 g settling
    assert amp.max() <= 0.06  # never grows much beyond the 0.05 initial offset
    assert amp[-1] < amp[0]  # lightly damped => decays


def test_pendulum_inverts_when_decelerating():
    """a_axial<0 (coast/aero lifts liquid off the aft bulkhead) => inverted pendulum grows."""
    amp = _integrate(a_axial=-20.0)
    assert amp[-1] > 5.0 * amp[0]  # negative stiffness => exponential divergence


def test_pendulum_stiffness_smooth_through_zero():
    """No |·| kink in the stiffness: ω²(a_axial) is linear through 0 (force flips sign smoothly)."""
    native = build_bundle([*FALCON_SET, "plant/slosh=pendulum"]).native
    state = np.array([0.05, 0.0, 0.0, 0.0])
    force_positive = native.slosh_diagnostics(state, [1.0, 0.0, 0.0], 50000.0)["force"]
    force_negative = native.slosh_diagnostics(state, [-1.0, 0.0, 0.0], 50000.0)["force"]
    # equal-and-opposite axial accel => equal-and-opposite restoring force (linear stiffness in a)
    assert np.allclose(force_positive, -force_negative, atol=1e-9)


def test_rotational_excitation_drives_slosh():
    """A pure body PITCH RATE (zero translational lateral accel) still drives the offset
    tank through ω×(ω×r)."""
    bundle = build_bundle([*FALCON_SET, "plant/slosh=springmass"])
    p = bundle.params
    from plant.dynamics import initial_state

    # upright hover-ish state, engine off, but a nonzero pitch rate about body-y
    ic = dict(
        r_ned=[0.0, 0.0, -3000.0],
        v_ned=[0.0, 0.0, 0.0],
        roll=0.0,
        pitch=np.pi / 2,
        yaw=0.0,
        w_body=[0.0, 0.5, 0.0],  # 0.5 rad/s pitch rate => centripetal accel at the offset tank
        m=40000.0,
        throttle0=0.0,
        seed=0,
    )
    s0 = initial_state(ic, p)
    cmd = ActuatorCommand(
        throttle=np.zeros(1), gimbal=np.zeros((1, 2)), fin=np.zeros(4), rcs=np.zeros(3)
    )
    deriv, _ = bundle.native.rhs(s0, cmd, np.zeros(3))
    d_slosh = np.asarray(deriv[12])  # slosh-state derivative
    # ξ̇ part is zero (state at rest) but the ξ̈ part must be nonzero from the rotational drive
    assert np.any(np.abs(d_slosh[1::2]) > 1e-6)  # ξ̈_y or ξ̈_z excited by the body rate

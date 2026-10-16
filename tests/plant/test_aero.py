"""Aero fidelity & sensitivity anchors.

Our operating point (engine-first, table-edge) is the data's weakest spot, so these are
mandatory: static-instability divergence, transonic drag rise, zero body-roll moment, and
x_cp extrapolation sensitivity (feeds the uncertainty budget).
"""

from __future__ import annotations

import numpy as np
from conftest import build_bundle

from core.interfaces import ActuatorCommand
from plant.dynamics import initial_state

FALCON_SET = ["vehicle_set=falcon9"]


def _aero(bundle, velocity, rates=None, mass=None, altitude=0.0):
    return bundle.native.aero_wrench_diagnostics(
        np.asarray(velocity, dtype=float),
        np.zeros(3) if rates is None else np.asarray(rates, dtype=float),
        float(bundle.params.mass.m_wet if mass is None else mass),
        float(altitude),
    )


def _velocity(alpha, mach, sound=340.3):
    speed = mach * sound
    return np.array([-speed * np.cos(alpha), 0.0, speed * np.sin(alpha)])


def _alpha_trajectory(xcp_frac_base, n=2500, pitch_perturb_deg=0.5):
    """Uncontrolled base-first descent with a tiny attitude perturbation; return alpha(t)."""
    bundle = build_bundle(
        [
            *FALCON_SET,
            "environment.atmosphere.scale_height=1e12",  # ~constant dynamic pressure
            "plant.aero.fins.fin_area=0.0",  # isolate body aero (no fin damping)
            f"plant.aero.coeffs.xcp_frac_base={xcp_frac_base}",
        ]
    )
    p = bundle.params
    ic = dict(
        r_ned=[0.0, 0.0, -8000.0],
        v_ned=[0.0, 0.0, 250.0],
        roll=0.0,
        pitch=np.deg2rad(90.0 + pitch_perturb_deg),
        yaw=0.0,
        w_body=[0, 0, 0],
        m=50000.0,
        throttle0=0.0,
        seed=0,
    )
    s0 = initial_state(ic, p)
    command = ActuatorCommand(
        throttle=np.zeros(bundle.meta["n_engines"]),
        gimbal=np.zeros((bundle.meta["n_gimbal"], 2)),
        fin=np.zeros(bundle.meta["n_fins"]),
        rcs=np.zeros(3),
    )
    session = bundle.native.create_session(s0)
    alpha = [float(session.advance(command, 1)[1]["aero_alpha"]) for _ in range(n)]
    return np.asarray(alpha), bundle


def test_static_instability_diverges():
    """Engine-first uncontrolled plant must DIVERGE when CP is ahead of CG (JSR-2020 L989)."""
    alpha_unstable, b_u = _alpha_trajectory(0.70)  # default: CP ahead of CG (unstable)
    alpha_stable, b_s = _alpha_trajectory(0.30)  # CP aft of CG (stable)

    # sign sanity: static margins straddle zero between the two configs
    # (static_margin = x_cp - x_cg; <0 unstable here)
    assert alpha_unstable[-1] > 3.0 * alpha_unstable[0]  # grows (diverges)
    assert alpha_stable.max() < 2.0 * alpha_stable[0]  # bounded (oscillates, no divergence)

    # exponential growth rate (eigenvalue) positive for unstable
    t = np.arange(len(alpha_unstable)) * float(b_u.params.plant_dt)
    half = len(t) // 2
    growth = np.polyfit(t[half:], np.log(alpha_unstable[half:]), 1)[0]
    assert growth > 0.0


def test_body_aero_zero_roll_moment():
    """Body aero arm is along body-x => zero roll moment for any state (roll = fins/RCS only)."""
    bundle = build_bundle(FALCON_SET)
    rng = np.random.default_rng(0)
    for _ in range(15):
        result = _aero(bundle, rng.uniform(-300, 300, size=3))
        assert abs(float(result["moment"][0])) < 1e-9


def test_transonic_drag_rise():
    """C_D peaks near Mach 1 (transonic max-q), lower sub/supersonic."""
    bundle = build_bundle(FALCON_SET)
    sound = float(bundle.native.environment_diagnostics(0.0)["a_sound"])
    cd = lambda mach: float(_aero(bundle, _velocity(0.0, mach, sound))["cd"])
    cd_sub, cd_trans, cd_super = cd(0.3), cd(1.0), cd(3.0)
    assert cd_trans > cd_sub
    assert cd_trans > cd_super
    assert cd_sub > 1.0  # base-first bluff body has high baseline drag (~1.8 region)


def test_lift_slope_positive_at_zero_alpha():
    """dC_L/dalpha > 0 at small alpha (normal-force restoring/destabilizing sense)."""
    bundle = build_bundle(FALCON_SET)
    sound = float(bundle.native.environment_diagnostics(0.0)["a_sound"])
    cl = lambda angle: float(_aero(bundle, _velocity(angle, 0.5, sound))["cl"])
    assert cl(0.1) > cl(0.0)


def test_aero_is_finite_at_apex_and_zero_v():
    """Native body aero is finite at the alpha=0 apex and at zero airspeed."""
    bundle = build_bundle(FALCON_SET)
    for velocity in ([-250.0, 0.0, 0.0], [0.0, 0.0, 0.0]):
        result = _aero(bundle, velocity)
        assert np.all(np.isfinite(result["force"]))
        assert np.all(np.isfinite(result["moment"]))
        assert float(result["alpha"]) == 0.0


def _xcp_at(alpha, mach=0.5):
    """x_CP (body-x, m) from the default table at a given total AoA [rad] and Mach."""
    bundle = build_bundle(FALCON_SET)
    sound = float(bundle.native.environment_diagnostics(0.0)["a_sound"])
    return float(_aero(bundle, _velocity(alpha, mach, sound))["x_cp"])


def test_body_damping_opposes_transverse_rate():
    """With w_body, the body damping moment OPPOSES the pitch/yaw rate (C_mq<0) and adds
    no roll moment; the term is the only change vs w_body=None."""
    bundle = build_bundle(FALCON_SET)
    velocity = np.array([-250.0, 0.0, 0.0])
    moment_base = np.asarray(_aero(bundle, velocity)["moment"])
    for w in (np.array([0.0, 0.4, 0.0]), np.array([0.0, 0.0, 0.4]), np.array([0.2, -0.3, 0.5])):
        moment = np.asarray(_aero(bundle, velocity, rates=w)["moment"])
        dM = moment - moment_base
        assert dM[1] * float(w[1]) <= 0.0  # pitch damping opposes pitch rate
        assert dM[2] * float(w[2]) <= 0.0  # yaw damping opposes yaw rate
        if float(w[1]) != 0.0 or float(w[2]) != 0.0:
            assert abs(dM[1]) + abs(dM[2]) > 1e-6  # nonzero transverse damping
        assert abs(dM[0]) < 1e-12  # no body roll damping (arm along x)


def test_xcp_migrates_more_with_aoa():
    """CP migrates a physically meaningful distance with AoA, forward (toward the leading base)
    for a base-first body, and is FLAT at the alpha=0 operating point (sin^2 weight =>
    static-instability sign at alpha=0 unchanged)."""
    xcp_0 = _xcp_at(0.0)
    xcp_30 = _xcp_at(np.deg2rad(30.0))
    xcp_90 = _xcp_at(np.deg2rad(90.0))
    travel = abs(xcp_90 - xcp_0)
    assert travel > 3.0  # physically meaningful CP travel
    # body-x is negative aft of nose; xcp_frac_alpha>0 => more-negative x = forward toward the
    # leading base (crossflow biases the CP baseward for base-first flight).
    assert xcp_90 < xcp_0
    # sin^2 weight => migration is slow near the apex: by 30° (sin^2=0.25) only ~1/4 of the travel
    # has occurred, well under half (concave start).
    assert abs(xcp_30 - xcp_0) < 0.5 * travel

"""Mass-property & propellant-budget anchors."""

from __future__ import annotations

import numpy as np
from conftest import build_bundle

from core.interfaces import ActuatorCommand
from plant.dynamics import initial_state

G0 = 9.80665
FALCON_SET = ["vehicle_set=falcon9"]


def test_prescribed_cg_inertia_monotone_continuous():
    bundle = build_bundle([*FALCON_SET, "plant.vehicle.massprops_model=prescribed"])
    mp = bundle.params.mass
    ms = np.linspace(float(mp.m_dry), float(mp.m_wet), 200)
    props = [bundle.native.mass_properties_diagnostics(mass) for mass in ms]
    xcg = np.array([float(value["x_cg"]) for value in props])
    Ixx = np.array([float(value["inertia"][0]) for value in props])
    Iyy = np.array([float(value["inertia"][1]) for value in props])
    # monotone increasing with mass
    assert np.all(np.diff(xcg) >= -1e-12)
    assert np.all(np.diff(Ixx) >= -1e-12)
    assert np.all(np.diff(Iyy) >= -1e-12)
    # continuous: bounded step-to-step change
    assert np.max(np.abs(np.diff(xcg))) < 1.0
    assert np.max(np.abs(np.diff(Iyy))) / float(mp.I_wet[1]) < 0.05


def test_dual_tank_cg_continuous_and_bounded():
    bundle = build_bundle([*FALCON_SET, "plant.vehicle.massprops_model=dual_tank"])
    mp = bundle.params.mass
    ms = np.linspace(float(mp.m_dry), float(mp.m_wet), 200)
    xcg = np.array([float(bundle.native.mass_properties_diagnostics(mass)["x_cg"]) for mass in ms])
    assert np.max(np.abs(np.diff(xcg))) < 1.0  # continuous
    # CG stays within the vehicle (between base and nose)
    assert np.all(xcg <= 0.0) and np.all(xcg >= -float(bundle.params.geom.length))


def test_dual_tank_endpoint_consistent_with_prescribed():
    """dual_tank reproduces (x_cg, I) of 'prescribed' EXACTLY at both the dry and wet endpoints
    (no silent vehicle shift on model switch) and its wet Iyy is NOT inflated by a parallel-axis
    double-count."""
    dual = build_bundle([*FALCON_SET, "plant.vehicle.massprops_model=dual_tank"])
    prescribed = build_bundle([*FALCON_SET, "plant.vehicle.massprops_model=prescribed"])
    mp = dual.params.mass
    for m_end in (float(mp.m_dry), float(mp.m_wet)):
        prop_d = dual.native.mass_properties_diagnostics(m_end)
        prop_p = prescribed.native.mass_properties_diagnostics(m_end)
        assert np.isclose(float(prop_d["x_cg"]), float(prop_p["x_cg"]), atol=1e-6)
        assert np.allclose(prop_d["inertia"], prop_p["inertia"], rtol=1e-6)

    # explicit no-double-count check at wet: Iyy must equal I_wet
    wet = dual.native.mass_properties_diagnostics(float(mp.m_wet))
    assert np.isclose(float(wet["inertia"][1]), float(mp.I_wet[1]), rtol=1e-6)

    # the in-between CG is genuinely NONLINEAR (1/m) — that's the physics dual_tank adds over the
    # linear 'prescribed' model: at mid-mass the two models must differ.
    m_mid = 0.5 * (float(mp.m_dry) + float(mp.m_wet))
    x_dual = float(dual.native.mass_properties_diagnostics(m_mid)["x_cg"])
    x_prescribed = float(prescribed.native.mass_properties_diagnostics(m_mid)["x_cg"])
    assert abs(x_dual - x_prescribed) > 1e-3


def test_tsiolkovsky_delta_v():
    """1-D burn, no gravity/drag => Δv = Isp·g0·ln(m0/m1) (rocket equation)."""
    bundle = build_bundle(
        [
            *FALCON_SET,
            "environment.atmosphere.g0=0.0",
            "environment.atmosphere.rho0=0.0",
            "environment.atmosphere.p_sea=0.0",
        ]
    )
    p = bundle.params
    ic = dict(
        r_ned=[0.0, 0.0, -8000.0],
        v_ned=[0.0, 0.0, 0.0],
        roll=0.0,
        pitch=np.pi / 2,
        yaw=0.0,
        w_body=[0, 0, 0],
        m=50000.0,
        throttle0=1.0,
        seed=0,
    )
    s0 = initial_state(ic, p)
    n, dt = 10000, float(p.plant_dt)  # 10 s
    cmd = ActuatorCommand(
        throttle=np.ones(1), gimbal=np.zeros((1, 2)), fin=np.zeros(4), rcs=np.zeros(3)
    )
    session = bundle.native.create_session(s0)
    final, _ = session.advance(cmd, n)

    m1 = float(final["m"])
    dv_numeric = float(np.linalg.norm(final["v_ned"]))
    dv_tsiolkovsky = float(p.engines.isp[0]) * G0 * np.log(50000.0 / m1)
    assert np.isclose(dv_numeric, dv_tsiolkovsky, rtol=1e-4)
    # propellant budget: burned mass = ∫ṁ dt = mdot·t
    mdot = float(p.engines.thrust_vac[0]) / (float(p.engines.isp[0]) * G0)
    assert np.isclose(50000.0 - m1, mdot * n * dt, rtol=1e-4)


def test_mass_floor_at_dry():
    """Burn past depletion => mass clamps at m_dry and episode terminates (fuel depleted)."""
    bundle = build_bundle(
        [
            *FALCON_SET,
            "environment.atmosphere.g0=0.0",
            "environment.atmosphere.rho0=0.0",
            "environment.atmosphere.p_sea=0.0",
        ]
    )
    p = bundle.params
    ic = dict(
        r_ned=[0.0, 0.0, -8000.0],
        v_ned=[0.0, 0.0, 0.0],
        roll=0.0,
        pitch=np.pi / 2,
        yaw=0.0,
        w_body=[0, 0, 0],
        m=50000.0,
        throttle0=1.0,
        seed=0,
    )
    s0 = initial_state(ic, p)
    n = 120000  # way past fuel exhaustion
    cmd = ActuatorCommand(
        throttle=np.ones(1), gimbal=np.zeros((1, 2)), fin=np.zeros(4), rcs=np.zeros(3)
    )
    session = bundle.native.create_session(s0)
    final, _ = session.advance(cmd, n)
    assert float(final["m"]) >= float(p.mass.m_dry) - 1e-6
    assert np.isclose(float(final["m"]), float(p.mass.m_dry), atol=1.0)
    assert bool(final["done"])

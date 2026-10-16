"""GNC tests: phase scheduling, guidance dispatch, PD controller."""

from __future__ import annotations

import types

import numpy as np
from conftest import build_bundle, compose_cfg, nominal_for

from core.interfaces import ActuatorCommand, Phase
from gnc.build import _nominal_atmosphere, build_gnc
from gnc.control.pd import PDController, attitude_error
from gnc.guidance.base import upright_quat
from gnc.guidance.phase import SequentialPhaseScheduler, altitude_below, single_crossing

FALCON_SET = ["vehicle_set=falcon9"]


def _xhat(alt, vz=100.0, q=None, w=None, m=40000.0):
    return types.SimpleNamespace(
        r_ned=np.array([0.0, 0.0, -alt]),
        v_ned=np.array([0.0, 0.0, vz]),
        q_bn=np.array([1.0, 0.0, 0.0, 0.0]) if q is None else q,
        w_body=np.zeros(3) if w is None else w,
        m=np.asarray(m),  # allocation schedules the gimbal moment arm on the measured mass
    )


def test_scheduler_latches_and_advances():
    sched = SequentialPhaseScheduler(
        [int(Phase.AERO_DESCENT), int(Phase.POWERED_DESCENT)], [altitude_below(2000.0)]
    )
    mem = sched.initial_memory()
    p, mem = sched.phase(_xhat(5000.0), 0.0, mem)
    assert p == int(Phase.AERO_DESCENT)
    p, mem = sched.phase(_xhat(1500.0), 1.0, mem)  # below switch -> advance
    assert p == int(Phase.POWERED_DESCENT)
    # latching: even if altitude rises again, stays powered
    p, mem = sched.phase(_xhat(3000.0), 2.0, mem)
    assert p == int(Phase.POWERED_DESCENT)


def test_scheduler_guard_requires_one_per_transition():
    import pytest

    with pytest.raises(ValueError):
        SequentialPhaseScheduler([0, 1, 2], [altitude_below(1000.0)])  # need 2 guards


def test_single_crossing_guard_theta():
    """Generic guard ψ(x;θ)=θ-altitude fires below θ."""
    psi = lambda x, t, theta: float(theta) - float(-np.asarray(x.r_ned)[2])
    g = single_crossing(psi, 2000.0)
    assert not g(_xhat(5000.0), 0.0)
    assert g(_xhat(1000.0), 0.0)


def test_pd_controller_fills_tau_and_opposes_error():
    from gnc.guidance.base import attitude_command

    ctrl = PDController(kp_att=1.0e5, kd_att=2.0e5)
    # estimated attitude off from upright by a small roll
    from core.frames import quat_from_euler

    q_hat = quat_from_euler(0.1, np.pi / 2, 0.0)
    cmd = attitude_command(int(Phase.AERO_DESCENT), upright_quat(), np.zeros(1))
    x = _xhat(3000.0, q=q_hat, w=np.zeros(3))
    out, _ = ctrl(x, cmd, {})
    tau = np.asarray(out.tau_body)
    assert not np.allclose(tau, 0.0)  # controller produced corrective torque
    # error vector is nonzero and tau is proportional to it (kd term zero since w=0)
    e = np.asarray(attitude_error(q_hat, upright_quat()))
    assert np.allclose(tau, 1.0e5 * e, atol=1e-6)


def test_pd_gains_are_per_axis_inertia_scaled():
    """Gains are per-axis (3-vectors) sized from inertia so ζ≈0.7 on BOTH the soft roll
    axis (Ixx) and the stiff pitch/yaw axes (Iyy=Izz), not a single scalar that critically damps
    one and leaves the other at ζ≈0.23."""
    bundle = build_bundle(FALCON_SET)
    ctrl = build_gnc(compose_cfg(FALCON_SET), bundle.params).controller  # gnc/control=pd
    kp = np.asarray(ctrl.kp)
    kd = np.asarray(ctrl.kd)
    I = np.asarray(bundle.params.mass.I_wet)  # [Ixx, Iyy, Izz]
    assert kp.shape == (3,) and kd.shape == (3,)  # per-axis, not scalar
    # realized closed-loop (ωn, ζ) per axis: ωn=√(Kp/I), ζ=Kd/(2√(Kp·I))
    wn = np.sqrt(kp / I)
    zeta = kd / (2.0 * np.sqrt(kp * I))
    assert np.allclose(zeta, 0.7, atol=1e-6)  # all three axes critically-ish damped
    assert np.allclose(
        wn, [1.0, 1.4, 1.4], atol=1e-6
    )  # roll slower than pitch/yaw, both O(1) rad/s
    # the anisotropy is real: pitch/yaw gain must dwarf roll gain (Iyy≈87×Ixx)
    assert kp[1] / kp[0] > 50.0


def test_pd_controller_scalar_gain_still_supported():
    """A scalar gain broadcasts over all axes (τ = kp·e − kd·ω)."""
    ctrl = PDController(kp_att=1.0e5, kd_att=2.0e5)
    from core.frames import quat_from_euler

    q_hat = quat_from_euler(0.1, np.pi / 2, 0.05)
    x = _xhat(3000.0, q=q_hat, w=np.array([0.01, -0.02, 0.03]))
    cmd = types.SimpleNamespace(q_ref=upright_quat(), w_ref=np.zeros(3), replace=lambda **k: k)
    out = ctrl(x, cmd, {})[0]
    e = np.asarray(attitude_error(q_hat, upright_quat()))
    expect = 1.0e5 * e - 2.0e5 * np.asarray(x.w_body)
    assert np.allclose(np.asarray(out["tau_body"]), expect, atol=1e-6)


def test_build_gnc_dispatches_per_phase():
    bundle = build_bundle(FALCON_SET)
    # start at aero so this exercises aero->powered dispatch (not the default scenario)
    cfg = compose_cfg([*FALCON_SET, "scenario.start_phase=aero"])
    gnc = build_gnc(cfg, bundle.params)
    mem = gnc.initial_memory()

    # high altitude -> aero phase, engine off (throttle 0)
    act, mem, phase = gnc(_xhat(5000.0), 0.0, mem)
    assert phase == int(Phase.AERO_DESCENT)
    assert float(act.throttle.sum()) == 0.0  # per-engine (N,)

    # below switch -> powered phase, engine commanded on
    act, mem, phase = gnc(_xhat(1000.0, vz=60.0), 1.0, mem)
    assert phase == int(Phase.POWERED_DESCENT)
    assert float(act.throttle.sum()) > 0.0


def test_guard_scheduler_builds_and_runs():
    bundle = build_bundle(FALCON_SET)
    cfg = compose_cfg([*FALCON_SET, "gnc/scheduler=guard", "scenario.start_phase=aero"])
    gnc = build_gnc(cfg, bundle.params)
    mem = gnc.initial_memory()
    _, mem, phase = gnc(_xhat(5000.0), 0.0, mem)
    assert phase == int(Phase.AERO_DESCENT)
    _, mem, phase = gnc(_xhat(1000.0), 1.0, mem)
    assert phase == int(Phase.POWERED_DESCENT)


def test_build_gnc_requires_convex_yaml_keys():
    import pytest
    from omegaconf import open_dict

    bundle = build_bundle(FALCON_SET)
    cfg = compose_cfg(FALCON_SET)
    with open_dict(cfg):
        del cfg.gnc.guidance.convex_nodes

    with pytest.raises(Exception, match="convex_nodes"):
        build_gnc(cfg, bundle.params)


def test_start_phase_powered_module():
    """start_phase=powered begins the run at the landing burn (module test)."""
    bundle = build_bundle([*FALCON_SET, "scenario=pdg"])
    cfg = compose_cfg([*FALCON_SET, "scenario=pdg"])
    gnc = build_gnc(cfg, bundle.params)
    mem = gnc.initial_memory()
    # even at high altitude, the latched start phase is POWERED (we don't revert to aero).
    # No continuous burn can land from 8 km/40 t (Clarabel certifies PrimalInfeasible at every
    # tf), so the hoverslam guidance correctly COASTS until ignition.
    act, mem, phase = gnc(_xhat(8000.0, vz=60.0), 0.0, mem)
    assert phase == int(Phase.POWERED_DESCENT)
    assert float(act.throttle.sum()) == 0.0  # engine off: coast until a burn can land
    # at the easy 650 m PDI state the ignition test is feasible -> landing burn fires
    act, mem, phase = gnc(_xhat(650.0, vz=90.0, m=28000.0), 1.0, mem)
    assert phase == int(Phase.POWERED_DESCENT)
    assert float(act.throttle.sum()) > 0.0  # engine on


def test_start_phase_reentry_fires_engines():
    """start_phase=reentry dispatches the reentry burn; octaweb fires several engines."""
    ov = [*FALCON_SET, "scenario=reentry_to_landing", "plant/vehicle=falcon9_octaweb"]
    bundle = build_bundle(ov)
    cfg = compose_cfg(ov)
    gnc = build_gnc(cfg, bundle.params)
    mem = gnc.initial_memory()
    act, mem, phase = gnc(_xhat(60000.0, vz=1800.0), 0.0, mem)
    assert phase == int(Phase.REENTRY_BURN)
    assert int((np.asarray(act.throttle) > 0).sum()) >= 2  # multiple engines fire (retro burn)


def test_engine_count_profile_3_0_1():
    """Octaweb default profile = 3-0-1: reentry fires 3, aero 0, landing 1 — fixed per phase."""
    ov = [*FALCON_SET, "plant/vehicle=falcon9_octaweb", "scenario=reentry_to_landing"]
    gnc = build_gnc(compose_cfg(ov), build_bundle(ov).params)
    mem = gnc.initial_memory()
    seq = []
    for alt, vz in [(60000, 1800), (25000, 800), (15000, 200), (1000, 60), (200, 20)]:
        act, mem, _ = gnc(_xhat(alt, vz=vz), 0.0, mem)
        seq.append(int((np.asarray(act.throttle) > 0).sum()))
    assert seq == [3, 3, 0, 1, 1]  # reentry(3) -> aero(0) -> landing(1); no within-burn change


def test_brake_start_phase_requires_brake_enabled():
    """start_phase=brake is only valid when powered_brake is on (else a clear error)."""
    import pytest

    ov = [*FALCON_SET, "plant/vehicle=falcon9_octaweb"]
    with pytest.raises(ValueError):  # powered_brake off => POWERED_BRAKE not in the phase list
        build_gnc(compose_cfg(ov + ["scenario.start_phase=brake"]), build_bundle(ov).params)


def test_scheduler_rejects_non_monotone_thresholds():
    """reentry_end_altitude must exceed switch_altitude, else a phase is silently skipped."""
    import pytest

    bundle = build_bundle(FALCON_SET)
    cfg = compose_cfg(
        [
            *FALCON_SET,
            "gnc.scheduler.reentry_end_altitude=1000",
            "gnc.scheduler.switch_altitude=2000",
        ]
    )
    with pytest.raises(ValueError):
        build_gnc(cfg, bundle.params)


def test_engine_out_fault_n1_raises_octaweb_ok():
    """fault=engine_out is N-agnostic: clear error on N=1 (idx>=N), works on octaweb."""
    import pytest

    with pytest.raises(ValueError):
        build_bundle([*FALCON_SET, "fault=engine_out"])  # Falcon set N=1, engine_out_idx=7
    b = build_bundle([*FALCON_SET, "fault=engine_out", "plant/vehicle=falcon9_octaweb"])
    assert float(b.params.engines.engine_on[7]) == 0.0
    assert float(b.params.engines.engine_on.sum()) == 8.0


def test_effective_sea_level_thrust_below_vacuum():
    """The sea-level back-pressure debit p_sea·A_e ≈ 91 kN (geometric A_e≈0.9 m²) is the thrust
    loss the guidance and allocation each subtract from vacuum thrust to plan/command achievable
    thrust."""
    bundle = build_bundle(FALCON_SET)
    nominal = nominal_for(bundle, FALCON_SET)
    debit = nominal["p_sea"] * nominal["nozzle_exit_area"]
    assert debit < nominal["thrust_vac"]
    assert 85e3 < debit < 100e3  # ~91 kN sea-level back-pressure loss (A_e≈0.9 m²)


def test_min_throttle_twr_above_one_suicide_burn():
    """Min-throttle delivered thrust (0.4·T_vac − p_sea·A_e) must exceed weight at the heaviest
    landing-burn-start mass => TWR_min>1 (Falcon-9 suicide-burn regime). Checked at dry+3t, not
    m_dry, so it discriminates the nozzle exit area A_e."""
    bundle = build_bundle(FALCON_SET)
    nominal = nominal_for(bundle, FALCON_SET)
    debit = nominal["p_sea"] * nominal["nozzle_exit_area"]  # sea-level (worst-case) fixed debit
    min_thrust = nominal["throttle_min"] * nominal["thrust_vac"] - debit
    m_landing_start = (
        nominal["m_dry"] + 3000.0
    )  # dry + ~3 t landing propellant (realistic PDI mass)
    twr_start = min_thrust / (m_landing_start * nominal["g0"])
    assert 1.0 < twr_start < 1.4  # suicide-burn regime at the heaviest landing-burn-start mass


# --------------------------------------------------------------------------- #
# IdentityNav: zero-noise identity + noise path actually perturbs when enabled
# --------------------------------------------------------------------------- #
def test_identity_nav_zero_noise_is_truth():
    """Zero noise and bias return the exact truth state, including the quaternion."""
    import numpy as np

    from gnc.navigation.identity import IdentityNav

    bundle = build_bundle(FALCON_SET)
    st = bundle.make_initial_state(
        {
            "r_ned": [10.0, -5.0, -3000.0],
            "v_ned": [2.0, 1.0, 90.0],
            "roll": 0.1,
            "pitch": 1.2,
            "yaw": -0.3,
            "w_body": [0.01, -0.02, 0.03],
            "m": 40000.0,
        }
    )
    nav = IdentityNav()  # all defaults zero
    meas = nav.measure(st, 7)
    assert np.array_equal(np.asarray(meas.r_ned), np.asarray(st.r_ned))
    assert np.array_equal(np.asarray(meas.v_ned), np.asarray(st.v_ned))
    assert np.array_equal(np.asarray(meas.q_bn), np.asarray(st.q_bn))  # quaternion untouched
    assert np.array_equal(np.asarray(meas.w_body), np.asarray(st.w_body))
    assert np.array_equal(np.asarray(meas.m), np.asarray(st.m))


def test_identity_nav_channel_streams_are_independent():
    """Other noisy channels do not perturb the deterministic position/velocity streams."""
    import numpy as np

    from core.seeding import fold_in, rng_from_seed
    from gnc.navigation.identity import IdentityNav

    bundle = build_bundle(FALCON_SET)
    st = bundle.make_initial_state(
        {
            "r_ned": [0.0, 0.0, -3000.0],
            "v_ned": [0.0, 0.0, 90.0],
            "roll": 0.0,
            "pitch": 1.2,
            "yaw": 0.0,
            "w_body": [0.0, 0.0, 0.0],
            "m": 40000.0,
        }
    )
    key = 3
    r_ref = np.asarray(st.r_ned) + 5.0 * rng_from_seed(fold_in(key, 0)).normal(size=3)
    v_ref = np.asarray(st.v_ned) + 0.5 * rng_from_seed(fold_in(key, 1)).normal(size=3)
    nav = IdentityNav(
        pos_noise=5.0, vel_noise=0.5, att_noise=0.01, rate_noise=0.02, mass_noise=10.0
    )
    meas = nav.measure(st, key)
    assert np.allclose(np.asarray(meas.r_ned), r_ref)  # pos stream unchanged by new channels
    assert np.allclose(np.asarray(meas.v_ned), v_ref)


def test_identity_nav_noise_and_bias_perturb_each_channel():
    """Enabling per-channel noise/bias actually perturbs that channel (and only random when
    noise>0; a pure bias is deterministic)."""
    import numpy as np

    from gnc.navigation.identity import IdentityNav

    bundle = build_bundle(FALCON_SET)
    st = bundle.make_initial_state(
        {
            "r_ned": [0.0, 0.0, -3000.0],
            "v_ned": [0.0, 0.0, 90.0],
            "roll": 0.0,
            "pitch": 1.2,
            "yaw": 0.0,
            "w_body": [0.0, 0.0, 0.0],
            "m": 40000.0,
        }
    )
    # pure deterministic bias on every channel
    nav = IdentityNav(
        pos_bias=[1.0, 2.0, 3.0],
        vel_bias=[0.1, 0.0, 0.0],
        att_bias=[0.0, 0.0, 0.05],
        rate_bias=[0.0, 0.01, 0.0],
        mass_bias=-50.0,
    )
    meas = nav.measure(st, 0)
    assert np.allclose(np.asarray(meas.r_ned), np.asarray(st.r_ned) + [1.0, 2.0, 3.0])
    assert np.allclose(np.asarray(meas.v_ned), np.asarray(st.v_ned) + [0.1, 0.0, 0.0])
    assert np.isclose(float(meas.m), float(st.m) - 50.0)
    assert not np.allclose(np.asarray(meas.q_bn), np.asarray(st.q_bn))  # attitude bias rotated it
    assert np.isclose(float(np.linalg.norm(np.asarray(meas.q_bn))), 1.0)  # stays unit-norm
    assert np.allclose(np.asarray(meas.w_body), [0.0, 0.01, 0.0])
    # mass noise is random across keys
    navn = IdentityNav(mass_noise=10.0)
    m0 = float(navn.measure(st, 1).m)
    m1 = float(navn.measure(st, 2).m)
    assert m0 != m1


def test_nominal_atmosphere_matches_plant():
    """The host-side GNC atmosphere schedule must agree with the native plant."""
    for model in ("exp", "ussa_poly", "vacuum"):
        bundle = build_bundle([*FALCON_SET, f"environment/atmosphere={model}"])
        host = _nominal_atmosphere(model, bundle.params.atmosphere)
        state = bundle.make_initial_state(
            {
                "r_ned": [0.0, 0.0, 0.0],
                "v_ned": [100.0, 0.0, 0.0],
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "w_body": [0.0, 0.0, 0.0],
                "m": 40000.0,
                "throttle0": 0.0,
                "seed": 0,
            }
        )
        command = ActuatorCommand(
            throttle=np.zeros(bundle.meta["n_engines"]),
            gimbal=np.zeros((bundle.meta["n_gimbal"], 2)),
            fin=np.zeros(bundle.meta["n_fins"]),
            rcs=np.zeros(3),
        )
        for h in (0.0, 5000.0, 11000.0, 20000.0, 30000.0, 50000.0):
            rho_h, a_h = host(h)
            point = state.replace(r_ned=np.array([0.0, 0.0, -h]))
            _, aux = bundle.native.rhs(point, command, np.zeros(3))
            a_native = 100.0 / float(aux["mach"])
            assert np.isclose(rho_h, float(aux["rho"]), rtol=1e-9, atol=1e-12), (model, h)
            assert np.isclose(a_h, a_native, rtol=1e-9), (model, h)
    # required arg (no silent 'exp' default): a missing model tag must fail loud
    import pytest

    with pytest.raises(ValueError, match="unknown atmosphere model"):
        _nominal_atmosphere("nope", build_bundle(FALCON_SET).params.atmosphere)

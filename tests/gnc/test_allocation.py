"""Allocation regression tests: explicit per-phase engine mask (no chatter) + no sub-floor band."""

from __future__ import annotations

import types

import numpy as np
from conftest import build_bundle, nominal_for

from core.interfaces import Command, Phase
from gnc.allocation import allocate
from gnc.guidance.base import upright_quat

FALCON_SET = ["vehicle_set=falcon9"]


def _xhat():
    # Mid-descent state used by the mass- and dynamic-pressure-scheduled allocator.
    return types.SimpleNamespace(
        q_bn=np.array([1.0, 0.0, 0.0, 0.0]),
        w_body=np.zeros(3),
        r_ned=np.array([0.0, 0.0, -3000.0]),
        v_ned=np.array([0.0, 0.0, 150.0]),
        m=np.asarray(40000.0),
    )


def _mask(idxs, n):
    m = np.zeros(n)
    m[idxs] = 1.0
    return np.asarray(m)


def _cmd(tv, mask):
    return Command(
        phase=np.asarray(int(Phase.POWERED_DESCENT)),
        mode=np.asarray(0),
        thrust_vector=tv,
        q_ref=upright_quat(),  # load-bearing: pointing is computed in the q_ref frame
        tau_body=np.zeros(3),
        active_engines=mask,
        w_ref=np.zeros(3),
    )


def test_allocation_fires_exactly_the_masked_engines():
    """The engine set is the explicit mask (NOT derived from magnitude): exactly those fire."""
    ov = [*FALCON_SET, "plant/vehicle=falcon9_octaweb"]
    b = build_bundle(ov)
    nom = nominal_for(b, ov)
    t1, x = nom["thrust_vac"], _xhat()
    for idxs in [[0], [0, 1, 5], list(range(9))]:
        k = len(idxs)
        act = allocate(_cmd(np.array([0.0, 0.0, -k * t1]), _mask(idxs, 9)), x, nom)
        thr = np.asarray(act.throttle)
        assert sorted(np.where(thr > 0)[0].tolist()) == idxs  # exactly the masked engines
        assert np.allclose(thr[idxs], 1.0, atol=1e-6)  # k·T1 split over k engines = full throttle


def test_allocation_engine_count_independent_of_magnitude():
    """A fixed 3-engine mask fires 3 engines for ANY thrust magnitude (no 1<->k chatter)."""
    ov = [*FALCON_SET, "plant/vehicle=falcon9_octaweb"]
    b = build_bundle(ov)
    nom = nominal_for(b, ov)
    t1, x = nom["thrust_vac"], _xhat()
    mask = _mask([0, 1, 5], 9)
    for frac in [0.5, 1.0, 2.0, 3.0]:  # magnitude varies, engine count must not
        act = allocate(_cmd(np.array([0.0, 0.0, -frac * t1]), mask), x, nom)
        assert int((np.asarray(act.throttle) > 0).sum()) == 3


def test_allocation_no_subfloor_deadband():
    """Each active engine throttle is 0 or >= Tmin across the magnitude range (no flameout)."""
    ov = [*FALCON_SET, "plant/vehicle=falcon9_octaweb"]
    b = build_bundle(ov)
    nom = nominal_for(b, ov)
    t1, floor, x = nom["thrust_vac"], nom["throttle_min"], _xhat()
    mask = _mask([0, 1, 5], 9)
    for frac in np.linspace(0.02, 3.0, 40):
        thr = np.asarray(allocate(_cmd(np.array([0.0, 0.0, -frac * t1]), mask), x, nom).throttle)
        active = thr[thr > 0]
        assert np.all(active >= floor - 1e-9) and np.all(active <= 1.0 + 1e-9)


def test_allocation_single_engine_default():
    b = build_bundle(FALCON_SET)
    nom = nominal_for(b, FALCON_SET)
    act = allocate(_cmd(np.array([0.0, 0.0, -0.5 * nom["thrust_vac"]]), np.ones(1)), _xhat(), nom)
    assert np.asarray(act.throttle).shape == (1,)
    assert float(act.throttle[0]) > 0.0


def test_allocation_compensates_back_pressure():
    """Plant thrust is affine in throttle (F = throttle*T_vac - p_amb*A_e); allocation inverts the
    FULL affine map so DELIVERED thrust matches the command (not vacuum-only)."""
    b = build_bundle(FALCON_SET)
    nom = nominal_for(b, FALCON_SET)
    x = _xhat()
    h = -float(x.r_ned[2])
    bp = nom["p_sea"] * np.exp(-h / nom["scale_height"]) * nom["nozzle_exit_area"]
    cmd_thrust = 5.0e5  # between the floor-snapped and clip-saturated range at this altitude
    act = allocate(_cmd(np.array([0.0, 0.0, -cmd_thrust]), np.ones(1)), x, nom)
    delivered = float(act.throttle[0]) * nom["thrust_vac"] - bp
    assert np.isclose(delivered, cmd_thrust, rtol=1e-3)


def _cmd_tau(tau, tv, mask):
    return Command(
        phase=np.asarray(int(Phase.POWERED_DESCENT)),
        mode=np.asarray(0),
        thrust_vector=tv,
        q_ref=upright_quat(),  # load-bearing: pointing is computed in the q_ref frame
        tau_body=tau,
        active_engines=mask,
        w_ref=np.zeros(3),
    )


def test_gimbal_receives_attitude_torque():
    """The controller torque tau is routed to the TVC gimbal (not just thrust-pointing)."""
    from gnc.guidance.base import upright_quat

    b = build_bundle(FALCON_SET)
    nom = nominal_for(b, FALCON_SET)
    x_up = types.SimpleNamespace(
        q_bn=upright_quat(),
        w_body=np.zeros(3),
        r_ned=np.array([0.0, 0.0, -2000.0]),
        v_ned=np.array([0.0, 0.0, 60.0]),
        m=np.asarray(40000.0),
    )
    tv = np.array([0.0, 0.0, -0.6 * nom["thrust_vac"]])  # vertical; nose-up => thrust along body-x
    g0 = allocate(_cmd_tau(np.zeros(3), tv, np.ones(1)), x_up, nom).gimbal
    g1 = allocate(_cmd_tau(np.array([0.0, 4.0e5, 0.0]), tv, np.ones(1)), x_up, nom).gimbal
    assert np.isclose(float(g0[0, 0]), 0.0, atol=1e-6)  # no torque => gimbal only points thrust (0)
    assert abs(float(g1[0, 0])) > 1e-3  # pitch torque deflects the gimbal


def _cmd_aero(tau):
    """AERO-phase command (mode=1, engine off): fins are the PRIMARY pitch/yaw authority here."""
    return Command(
        phase=np.asarray(int(Phase.AERO_DESCENT)),
        mode=np.asarray(1),
        thrust_vector=np.zeros(3),
        q_ref=upright_quat(),  # load-bearing: pointing is computed in the q_ref frame
        tau_body=tau,
        active_engines=np.zeros(1),
        w_ref=np.zeros(3),
    )


def test_fin_allocation_correct_moment_sign():
    """In the AERO phase, a commanded PITCH torque yields the right-sign
    fin moment; ROLL is handled by RCS, so a pure roll command leaves the fins idle."""
    b = build_bundle(FALCON_SET)
    nom = nominal_for(b, FALCON_SET)
    v = np.array([-200.0, 0.0, 0.0])  # base-first descent

    # pitch torque +y => fins must produce +M_y (aero phase, fins primary)
    fin_p = allocate(_cmd_aero(np.array([0.0, 1.0e5, 0.0])), _xhat(), nom).fin
    result = b.native.fin_wrench_diagnostics(v, np.zeros(3), fin_p, 50000.0, 0.0)
    assert float(result["moment"][1]) > 0.0  # correct sign

    # pure roll torque +x => fins idle (roll is RCS-only)
    fin_r = allocate(_cmd_aero(np.array([1.0e5, 0.0, 0.0])), _xhat(), nom).fin
    assert np.allclose(np.asarray(fin_r), 0.0, atol=1e-9)  # no collective fin roll


def test_blended_mode_engages_fins_during_powered_flight():
    """At high dynamic pressure, blended mode assigns part of pitch control to the fins."""
    b = build_bundle(FALCON_SET)
    x = _xhat()  # 150 m/s at 3 km => q_dyn ~ 10 kPa: real fin authority
    nom_off = nominal_for(b, [*FALCON_SET, "gnc.allocation.powered_fin_mode=phase_gated"])
    nom_bl = nominal_for(b, [*FALCON_SET, "gnc.allocation.powered_fin_mode=blended"])
    tv = np.array([0.0, 0.0, -0.6 * nom_off["thrust_vac"]])
    cmd = _cmd(tv, _mask([0], 1)).replace(tau_body=np.array([0.0, 2.0e5, 0.0]))

    off = allocate(cmd, x, nom_off)
    bl = allocate(cmd, x, nom_bl)

    assert np.allclose(np.asarray(off.fin), 0.0)  # phase-gated mode keeps fins idle while powered
    assert float(np.asarray(bl.fin)[2]) > 1e-4  # blended: +dp on the +pitch fin pair
    # the fin share offloads the gimbal: smaller pitch deflection for the same torque demand
    assert abs(float(np.asarray(bl.gimbal)[0, 0])) < abs(float(np.asarray(off.gimbal)[0, 0]))


def test_blended_mode_fades_fins_without_dynamic_pressure():
    """Blended share -> 0 as q_dyn -> 0: near touchdown the gimbal carries everything again."""
    b = build_bundle(FALCON_SET)
    nom_bl = nominal_for(b, [*FALCON_SET, "gnc.allocation.powered_fin_mode=blended"])
    x = types.SimpleNamespace(
        q_bn=np.array([1.0, 0.0, 0.0, 0.0]),
        w_body=np.zeros(3),
        r_ned=np.array([0.0, 0.0, -5.0]),
        v_ned=np.array([0.0, 0.0, 2.0]),  # q_dyn ~ 2.5 Pa: no fin authority
        m=np.asarray(27000.0),
    )
    tv = np.array([0.0, 0.0, -0.6 * nom_bl["thrust_vac"]])
    cmd = _cmd(tv, _mask([0], 1)).replace(tau_body=np.array([0.0, 2.0e5, 0.0]))
    act = allocate(cmd, x, nom_bl)
    assert np.allclose(np.asarray(act.fin), 0.0, atol=1e-5)
    assert abs(float(np.asarray(act.gimbal)[0, 0])) > 1e-4  # gimbal carries the demand

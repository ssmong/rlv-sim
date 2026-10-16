"""Map virtual GNC commands to actuator commands."""

from __future__ import annotations

import numpy as np

from core.frames import dcm_from_quat
from core.interfaces import ActuatorCommand


def _smooth_clip(x, lim):
    """Smoothly saturate to [-lim, lim]."""
    return lim * np.tanh(x / lim)


def _cg_x(nominal, m):
    """Interpolate nominal body-x CG from total mass."""
    m_dry, m_wet = nominal["m_dry"], nominal["m_wet"]
    f = np.clip((m - m_dry) / np.maximum(m_wet - m_dry, 1.0), 0.0, 1.0)
    return nominal["x_cg_dry"] + f * (nominal["x_cg_wet"] - nominal["x_cg_dry"])


def _linear_interp(x, xp, fp):
    """Linear interpolation with edge clamp."""
    return float(np.interp(float(x), np.asarray(xp, dtype=float), np.asarray(fp, dtype=float)))


def _fin_effectiveness_matrix(nominal, x_hat, q_dyn, x_cg, a_sound):
    """Return B[:, i] = body torque per rad of fin i for geometry-based allocation."""
    C_bn = np.asarray(dcm_from_quat(x_hat.q_bn), dtype=float)
    v_body = C_bn.T @ np.asarray(x_hat.v_ned, dtype=float)
    V = np.linalg.norm(v_body)
    if V < 1e-6 or q_dyn <= 1e-6:
        return np.zeros((3, nominal["fin_arm"].shape[0]))
    vhat = v_body / V
    mach = V / max(a_sound, 1e-6)
    cla = _linear_interp(mach, nominal["fin_mach_grid"], nominal["fin_cla_table"])
    cols = []
    r_cg = np.array([x_cg, 0.0, 0.0])
    for r_fin, n_fin in zip(nominal["fin_arm"], nominal["fin_normal"], strict=True):
        lift_dir = n_fin - float(np.dot(n_fin, vhat)) * vhat
        ln = np.linalg.norm(lift_dir)
        if ln > 1e-8:
            lift_dir = lift_dir / ln
        else:
            lift_dir = np.zeros(3)
        f_per_rad = q_dyn * nominal["fin_area"] * cla * lift_dir
        cols.append(np.cross(r_fin - r_cg, f_per_rad))
    return np.stack(cols, axis=1)


def allocate(command, x_hat, nominal: dict) -> ActuatorCommand:
    """Map a phase-tagged Command to a per-engine ActuatorCommand."""
    tau = command.tau_body  # desired body torque (from controller PD) [roll, pitch, yaw]
    tr, tp, ty = tau[0], tau[1], tau[2]
    powered = command.mode == 0
    tv = command.thrust_vector
    tv_mag = np.linalg.norm(tv)

    t_each = nominal["thrust_vac"]  # per-engine vacuum max
    floor = nominal["throttle_min"]
    # active_engines chooses which engines fire; thrust is split uniformly over that set.
    active = command.active_engines  # (N,) 0/1
    n_active = np.maximum(np.sum(active), 1.0)
    per_engine = tv_mag / n_active
    # Plant thrust is AFFINE in throttle: F = throttle*T_vac - p_amb*A_e, the back-pressure debit
    # being a FIXED force. Invert the FULL affine map so the DELIVERED thrust matches the guidance
    # command; a vacuum-only inverse (throttle = F/T_vac) would drop the affine offset and
    # under-deliver by p_amb*A_e (~10-15% thrust deficit low in the atmosphere).
    h = -x_hat.r_ned[2]
    p_amb = nominal["p_sea"] * np.exp(-np.maximum(h, 0.0) / nominal["scale_height"])
    bp_each = p_amb * nominal["nozzle_exit_area"]  # per-engine fixed back-pressure debit [N]
    throttle_each = np.where(powered, np.clip((per_engine + bp_each) / t_each, 0.0, 1.0), 0.0)
    # Avoid the engine dead band: a positive sub-floor demand snaps to minimum throttle.
    throttle_each = np.where((throttle_each > 0.0) & (throttle_each < floor), floor, throttle_each)
    throttle = throttle_each * active

    # Delivered per-engine thrust (after the back-pressure debit) for the gimbal authority schedule.
    thrust_now = np.maximum(throttle_each * t_each - bp_each, 0.0)

    # Pitch/yaw torque shares between gimbal and fins.
    #   phase_gated — gimbal while powered, fins during engine-off aerodynamic flight.
    #   blended     — weighted split based on the squared torque effectiveness per radian.
    mode = str(nominal["fin_alloc_mode"])
    if mode not in ("phase_gated", "blended"):
        raise ValueError(f"unknown fin_alloc_mode {mode!r}; pick phase_gated|blended")

    # Gimbal: thrust pointing plus primary powered pitch/yaw authority.
    # Sign convention: x_eng is behind the CG, so positive pitch torque needs negative pitch gimbal.
    # POINTING IS COMPUTED IN THE REFERENCE FRAME (q_ref), not the current attitude: guidance
    # aligns q_ref with the commanded thrust, so the pointing trim is ~0 and the gimbal budget
    # serves attitude stabilization. Pointing slaved to the CURRENT attitude is positive
    # feedback — an attitude error deflects the gimbal to hold the thrust direction, and that
    # deflection's torque GROWS the attitude error until saturation and tumble.
    C_ref = dcm_from_quat(command.q_ref)
    d_body = C_ref.T @ (tv / np.maximum(tv_mag, 1e-6))
    gimbal_pitch = -np.arcsin(np.clip(d_body[2], -1.0, 1.0))
    gimbal_yaw = np.arctan2(d_body[1], d_body[0])
    arm = np.abs(nominal["x_eng"] - _cg_x(nominal, x_hat.m))  # |x_eng - x_cg(m)| [m]
    # guard against a zero/idle thrust (gimbal then has no torque authority -> attitude angle 0):
    gimbal_eff = np.maximum(thrust_now * arm, 1.0)  # dM/dangle [N*m/rad], floored.
    g_att = np.where(thrust_now > 0.0, 1.0 / gimbal_eff, 0.0)  # scheduled tau->angle gain
    glim = nominal["gimbal_limit"]

    # Fin authority uses the configured plant atmosphere. Engine back-pressure uses the exponential
    # pressure model defined by p_sea and scale_height.
    rho, a_sound = nominal["atmosphere"](h)
    v_mag = np.linalg.norm(x_hat.v_ned)
    q_dyn = 0.5 * rho * v_mag**2  # dynamic pressure [Pa]
    fin_arm = np.abs(nominal["fin_arm_x"] - _cg_x(nominal, x_hat.m))  # |x_fin - x_cg(m)| [m]
    # Single-fin per-deflection effectiveness [N*m/rad], floored to avoid division by zero.
    fin_eff = np.maximum(q_dyn * fin_arm * nominal["fin_authority_coeff"], 1.0)

    if mode == "blended":
        g_e_raw = thrust_now * arm  # gimbal torque per rad of deflection
        g_f_raw = 2.0 * q_dyn * fin_arm * nominal["fin_authority_coeff"]  # per differential rad
        a_e = g_e_raw**2 / nominal["fin_alloc_w_gimbal"]
        a_f = g_f_raw**2 / nominal["fin_alloc_w_fin"]
        s_fin = np.where(a_e + a_f > 0.0, a_f / np.maximum(a_e + a_f, 1e-12), 0.0)
    else:
        s_fin = np.where(powered, 0.0, 1.0)
    # Dynamic-pressure lockout: below q̄_min the fins make negligible/uncertain torque (terminal
    # flail near touchdown, transonic bucket) -> drop their share so the gimbal carries pitch/yaw.
    s_fin = np.where(q_dyn < nominal["fin_qbar_lockout"], 0.0, s_fin)
    s_gim = 1.0 - s_fin

    # Smooth saturation lets the residual torque be computed from the realized primary command.
    att_demand = np.array(
        [-g_att * (s_gim * tp), -g_att * (s_gim * ty)]
    )  # Negative because x_eng - x_cg < 0.
    att_gimbal = _smooth_clip(att_demand, glim)
    point_raw = np.array([gimbal_pitch, gimbal_yaw])
    g_one = np.where(powered, np.clip(point_raw + att_gimbal, -glim, glim), np.zeros(2))
    gimbal = np.broadcast_to(g_one, (int(nominal["n_gimbal"]), 2))  # (Ng,2)
    # Realized pitch/yaw torque from the FULL hard-clipped gimbal deflection: every radian of
    # deflection torques the body (engine sits ~14 m below the CG), whether it was "meant" for
    # attitude or for pointing — so the RCS residual below must see the torque of the WHOLE
    # realized gimbal, not just the attitude share. Using the pre-clip att_gimbal would make the
    # controller believe torque it never produced under saturation (feedback asymmetry). With the
    # thrust-aligned q_ref the pointing term is ~0 and this reduces to the attitude share; it
    # differs only when q_ref is misaligned with the commanded thrust.
    gimbal_tau_py = np.where(powered, -gimbal_eff * g_one, np.zeros(2))  # [pitch, yaw]

    flim = nominal["fin_limit"]
    if nominal["fin_geometry_alloc"]:
        x_cg = _cg_x(nominal, x_hat.m)
        B = _fin_effectiveness_matrix(nominal, x_hat, q_dyn, x_cg, a_sound)
        roll_share = 0.0 if q_dyn < nominal["fin_qbar_lockout"] else 1.0
        tau_fin_des = np.array([roll_share * tr, s_fin * tp, s_fin * ty])
        scale = max(float(np.linalg.norm(B)), 1.0)
        damping = 1e-3 * scale
        pinv = B.T @ np.linalg.inv(B @ B.T + (damping * damping) * np.eye(3))
        fin_cmd = _smooth_clip(pinv @ tau_fin_des, flim)
        fin_tau = B @ fin_cmd
        fin_tau_py = fin_tau[1:3]
        fin_tau_roll = fin_tau[0]
    else:
        # Fins: pitch/yaw share per the split above. Roll remains on RCS (h from the throttle map).
        g_fin = np.where(q_dyn > 1e-3, 1.0 / fin_eff, 0.0)
        # Differential half-deflection per axis, smooth-saturated to the fin limit.
        dp = _smooth_clip(0.5 * g_fin * (s_fin * tp), flim)  # pitch differential half-deflection
        dy = _smooth_clip(0.5 * g_fin * (s_fin * ty), flim)  # yaw differential half-deflection
        # Fin map: [-dp, -dy, +dp, +dy]; collective roll is intentionally omitted.
        fin_cmd = np.array([-dp, -dy, dp, dy])
        # Realized pitch/yaw torque from the saturated fin differential.
        fin_tau_py = fin_eff * np.array([2.0 * dp, 2.0 * dy])
        fin_tau_roll = 0.0

    # RCS: roll plus pitch/yaw residual after the phase primary actuator saturates.
    primary_py = gimbal_tau_py + fin_tau_py  # one of the two is zero per phase
    residual_py = tau[1:3] - primary_py  # [pitch, yaw] shortfall for RCS to mop up
    residual_roll = tr - fin_tau_roll
    if nominal["rcs_enabled"]:
        s = nominal["rcs_torque_scale"]
        rcs_roll = _smooth_clip(residual_roll / s, 1.0)
        rcs_py = _smooth_clip(residual_py / s, 1.0)  # pitch/yaw: only the primary's residual
        rcs = np.array([rcs_roll, rcs_py[0], rcs_py[1]])
    else:
        rcs = np.zeros(3)

    return ActuatorCommand(throttle=throttle, gimbal=gimbal, fin=fin_cmd, rcs=rcs)

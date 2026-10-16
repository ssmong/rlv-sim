"""Shared plant/GNC state, command, result, and parameter contracts.

The numeric records are frozen standard dataclasses with a functional ``replace`` method. At
episode start, the complete ``PlantState`` is copied once into a native C++ session. The session
then owns actuator internals, slosh modes, wind filters, RNG, and integration state; Python receives
only the public tick-end snapshot needed by GNC and logging.

Parameter structs hold numeric configuration leaves shared by native-plant construction and GNC.
Physics-model selection is resolved once by ``plant.dynamics.build_plant``.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, replace
from typing import TypeAlias

import numpy as np

Array: TypeAlias = np.ndarray | np.generic | float | int | bool


def immutable_dataclass(cls):
    """Make a frozen dataclass with the functional ``replace`` API used by the GNC stack."""
    record = dataclass(frozen=True)(cls)

    def replace_fields(self, **changes):
        return replace(self, **changes)

    record.replace = replace_fields
    return record


# --------------------------------------------------------------------------- #
# Enumerations (Python-level tags; stored as ints in arrays for KPI logging)
# --------------------------------------------------------------------------- #
class Phase(enum.IntEnum):
    """KPI/diagnostic phase tag (post-flip sequence). The plant is phase-AGNOSTIC;
    this tag is produced by GNC and passed through for logging — the plant never branches on it.
    (No TERMINAL: touchdown is just the end of the landing burn.)"""

    REENTRY_BURN = 0  # engine-first retro-burn, high alt/Mach (3 engines)
    AERO_DESCENT = 1  # engine off, fins/RCS only (statically unstable, base-first)
    POWERED_BRAKE = 2  # optional 3-engine landing-brake start (3-0-3-1 profile)
    POWERED_DESCENT = 3  # final landing burn: TVC primary, fins assist (1 engine)


class TerminationReason(enum.IntEnum):
    RUNNING = 0
    TOUCHDOWN = 1  # reached ground AND survived contact (within crash sink/tilt/rate limits)
    FUEL_DEPLETED = 2  # m <= m_dry
    MAX_STEPS = 3  # horizon exhausted (still airborne)
    CRASH = 4  # reached ground but exceeded a structural-survival limit (sink rate / tilt / spin)


# --------------------------------------------------------------------------- #
# Plant initial state and diagnostic full-state contract
# --------------------------------------------------------------------------- #
@immutable_dataclass
class PlantState:
    """Full plant state. Rigid-body core + actuator internals + disturbance filters + RNG.

    Frames (see core.frames): r_ned/v_ned in NED, q_bn body->NED quaternion (scalar-first),
    w_body body rates in body frame. Altitude = -r_ned[2].
    """

    t: Array  # () time [s]
    r_ned: Array  # (3,) position NED [m]
    v_ned: Array  # (3,) velocity NED [m/s]
    q_bn: Array  # (4,) body->NED quaternion [w,x,y,z]
    w_body: Array  # (3,) angular velocity, body frame [rad/s]
    m: Array  # () total mass [kg]

    # Actuator truth states (continuous, integrated in the RHS)
    # Multi-engine: N engines (Ng gimballed). N=Ng=1 reduces to the single-engine plant.
    tvc: Array  # (Ng,2) actual gimbal [pitch, yaw] per gimballed engine [rad]
    tvc_rate: Array  # (Ng,2) gimbal angular rate [rad/s], bounded to +/-tvc_rate_max_rad_s by the
    # saturated-rate-command servo (_rate_limited_2nd_order): a true physical slew rate
    throttle_act: Array  # (N,) actual throttle fraction per engine in [0,1]
    fin: Array  # (4,) actual grid-fin deflections [rad]
    fin_rate: Array  # (4,) fin servo rate [rad/s], bounded to +/-fin_rate_max_rad_s
    rcs_fuel: Array  # () remaining RCS propellant budget [kg]

    # Slosh modes (2 planes × [angle, rate]); zero & inert for the 'none' model
    slosh: Array  # (4,)

    # Disturbance filter states (updated discretely per substep, held across the RK4 step)
    wind_mean: Array  # (3,) OU mean-wind state, NED [m/s]
    wind_gust: Array  # (3,) materialized gust output velocity, NED [m/s] (= C @ wind_filter)
    wind_filter: Array  # (3,K) per-axis turbulence shaping-filter state; 0 for none/steady

    key: Array  # (2,) uint32 seed material copied into the native RNG at session creation
    done: Array  # () bool — frozen after termination predicate fires


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
@immutable_dataclass
class ActuatorCommand:
    """Per-engine plant input for N engines and Ng gimballed engines."""

    throttle: Array  # (N,) per-engine commanded throttle fraction in [0,1]
    gimbal: Array  # (Ng,2) per-gimballed-engine [pitch, yaw] [rad]
    fin: Array  # (4,) commanded grid-fin deflections [rad]
    rcs: Array  # (3,) commanded RCS [roll, pitch, yaw] in [-1,1] (PWM duty / on-off)


@immutable_dataclass
class Command:
    """GNC OUTPUT (phase-tagged). Aero descent => attitude (q_ref); powered => thrust vector.

    ``mode`` selects which body field is active: 0 = thrust_vector (NED), 1 = attitude q_ref.
    ``phase`` is a pass-through KPI tag (Phase). The two-layer GNC fills this progressively:
    guidance sets ``thrust_vector``/``q_ref`` (the reference); the controller sets ``tau_body``
    (the inner-loop control torque); ``allocate`` maps it to an ActuatorCommand. The plant never
    sees Command directly.
    """

    phase: Array  # () int Phase tag
    mode: Array  # () int: 0 thrust_vector, 1 attitude
    thrust_vector: Array  # (3,) desired thrust in NED [N] (guidance)
    q_ref: Array  # (4,) desired body->NED attitude (guidance)
    tau_body: Array  # (3,) desired control torque, body frame [N m] (controller)
    active_engines: (
        Array  # (N,) 0/1 which engines fire this phase (explicit, fixed per phase => no chatter)
    )
    w_ref: Array  # (3,) body-rate FEEDFORWARD [rad/s] (guidance); PD damps (w - w_ref), not w.
    # Lets the controller track a ROTATING attitude reference without fighting it — required for
    # 6-DoF plan-attitude tracking (a static-rate PD opposes the plan's commanded slew). 0 = hold.


@immutable_dataclass
class Measurement:
    """Sensor output = function of truth (nav seam). Identity nav passes the (optionally
    noised) full state; an EKF would consume a subset."""

    r_ned: Array
    v_ned: Array
    q_bn: Array
    w_body: Array
    m: Array  # total mass [kg] (propellant-gauged; consumed by mass-aware convex guidance)


# --------------------------------------------------------------------------- #
# Parameter structs (numeric leaves only -> vmap-able)
# --------------------------------------------------------------------------- #
@immutable_dataclass
class VehicleGeom:
    """Geometry & reference quantities (body frame, x toward nose)."""

    s_ref: Array  # () aerodynamic reference area [m^2]
    d_ref: Array  # () reference diameter [m]
    length: Array  # () vehicle length [m]
    x_eng: (
        Array  # () engine gimbal point, body-x coord relative to nose-origin [m] (negative = aft)
    )
    fwd_axis: Array  # (3,) engine-forward aero axis in body frame (= [-1,0,0])


@immutable_dataclass
class MassPropsParams:
    """prescribed_cg_from_total_mass: CG, inertia and thrust arm as functions of scalar mass.

    Linear-in-mass model between dry and wet endpoints: quantities are interpolated on the
    propellant fraction f = (m - m_dry)/(m_wet - m_dry) in [0,1].
    """

    m_dry: Array  # () dry mass [kg]
    m_wet: Array  # () wet (initial) mass [kg]
    x_cg_dry: Array  # () CG body-x at dry [m] (relative to nose-origin, negative aft)
    x_cg_wet: Array  # () CG body-x at wet [m]
    I_dry: Array  # (3,) principal inertia [Ixx,Iyy,Izz] at dry [kg m^2]
    I_wet: Array  # (3,) principal inertia at wet [kg m^2]
    # Asymmetry: a LATERAL CG offset [y,z] off the body x-axis (asymmetric mass / a stuck
    # deploy / off-c.g. payload) and off-diagonal products of inertia [Ixy,Ixz,Iyz]. Default
    # Zero offsets describe a symmetric axisymmetric booster. Nonzero offsets create moments from
    # forces through the geometric axis, while the full inertia tensor couples the rotational axes.
    cg_lateral: Array  # (2,) [y,z] CG offset from the body x-axis [m]
    inertia_products: Array  # (3,) [Ixy, Ixz, Iyz] products of inertia [kg m^2]


@immutable_dataclass
class AeroParams:
    """Body coefficient lookup table (axisymmetric, engine-forward total-α) + grid-fin model.

    Tables are indexed by cos(total angle of attack) — NOT alpha — so the lookup is C∞ in the
    state at the base-first operating point (alpha=0 <=> cos=1, the apex where arccos is
    non-differentiable). The cos-axis uses a monotone-cubic (PCHIP, C¹) Hermite interpolation;
    the Mach axis is linear (JSR-2020 benchmark). Because the body is
    axisymmetric the coefficients depend only on the *magnitude* of the cone angle (cos in
    [0,1] for alpha in [0,90°]); the force *direction* is carried by lift_dir, so +/-alpha
    symmetry is structural. x_cp is body-x of the center of pressure (relative to nose-origin).
    The 'linear' nominal aero (aero_linear) ignores the tables and uses cla/cda below.
    """

    cos_grid: Array  # (Nc,) cos(total AoA) knots, strictly increasing (cos in [0,1])
    mach_grid: Array  # (Nm,) Mach samples
    cd_table: Array  # (Nc, Nm) drag coeff at nodes
    cl_table: Array  # (Nc, Nm) lift coeff at nodes
    xcp_table: Array  # (Nc, Nm) center-of-pressure body-x [m] at nodes
    cd_slope: Array  # (Nc, Nm) d(C_D)/d(cos) at nodes (PCHIP Hermite slopes, C¹)
    cl_slope: Array  # (Nc, Nm) d(C_L)/d(cos) at nodes
    xcp_slope: Array  # (Nc, Nm) d(x_CP)/d(cos) at nodes
    # AEDB pitch-moment coefficient about a FIXED reference CoG. The file backend builds the body
    # moment from C_m directly (M = q·S·D·C_m), avoiding the x_cp = x_cg_ref − D·C_m/C_N division
    # that is ill-conditioned where C_N→0 (the base-first apex). The analytic backend leaves these
    # inert (moment_uses_cm=0 => moment from x_cp).
    cm_table: Array  # (Nc, Nm) pitch-moment coeff about x_cg_ref at nodes
    cm_slope: Array  # (Nc, Nm) d(C_m)/d(cos) at nodes
    x_cg_ref: Array  # () reference CoG body-x the tabulated C_m is taken about [m]
    moment_uses_cm: Array  # () 1 => body moment from C_m (file backend); 0 => from x_cp (analytic)
    # grid fins (per-fin component model)
    n_fins: Array  # () number of fins (stored as float; static count baked at build)
    fin_area: Array  # () per-fin reference area [m^2]
    fin_cla: Array  # () fin lift-curve slope [1/rad]
    fin_cd0: Array  # () fin zero-lift drag coeff
    fin_arm: Array  # (Nf, 3) fin hinge positions in body frame [m]
    fin_normal: Array  # (Nf, 3) fin deflection-axis unit normals (lift direction at δ>0)
    fin_stall_alpha: Array  # () fin stall angle for tanh saturation [rad]
    fin_transonic_mach: Array  # () Mach of grid-fin drag-rise center
    fin_incremental: Array  # () 1 => table already includes zero-deflection fins; add increments
    fin_mach_grid: Array  # (Nfm,) Mach samples for data-backed fin increment coefficients
    fin_cla_table: Array  # (Nfm,) fin normal-force increment slope [1/rad]
    fin_cd_delta2_table: Array  # (Nfm,) fin drag increment quadratic coeff [1/rad^2]
    # linear nominal aero (Jung 2025 eqs 5-7), used only by aero_linear
    cla: Array  # () body lift-curve slope [1/rad]
    cda: Array  # () body axial drag coeff at zero alpha


@immutable_dataclass
class AtmosphereParams:
    """Exponential (rho0, scale_height) and/or US-Standard layered atmosphere. delta_T and
    density_scale add a non-standard-day temperature offset + density dispersion (ussa_poly)."""

    rho0: Array  # () sea-level density [kg/m^3]
    scale_height: Array  # () exponential scale height [m]
    a_sound: Array  # () reference speed of sound [m/s] (constant approx for Mach)
    g0: Array  # () standard gravity [m/s^2]
    delta_T: Array  # () non-standard-day temperature offset [K]; 0 = standard ISA-76 (ussa_poly)
    density_scale: Array  # () density dispersion multiplier; 1 = nominal (ussa_poly)


@immutable_dataclass
class WindParams:
    """OU mean wind + turbulence gust (Dryden or von Karman). 'none' model zeroes everything."""

    mean0: Array  # (3,) initial mean wind NED [m/s]
    ou_tau: Array  # () OU correlation time [s]
    ou_sigma: Array  # (3,) OU stationary std per axis [m/s]
    turb_L: Array  # (3,) turbulence length scales per axis [m]
    turb_sigma: Array  # (3,) turbulence intensities per axis [m/s]


@immutable_dataclass
class EarthRotationParams:
    """Local-level rotating-Earth terms for the fixed-origin NED plant.

    ``g_ned`` already contains apparent gravity at the NED origin, including the origin's
    centrifugal contribution.  The dynamics therefore add Coriolis acceleration and, optionally,
    only the *differential* centrifugal acceleration relative to that origin.
    """

    enabled: Array  # () 1 for rotating Earth, 0 for fixed Earth
    omega_ned: Array  # (3,) Earth angular velocity expressed in reference NED [rad/s]
    differential_centrifugal: Array  # () 1 to include the origin-relative centrifugal term


@immutable_dataclass
class SloshParams:
    """Pendulum / spring-mass slosh (2 planes). 'none' model => slosh_mass=0 (inert)."""

    slosh_mass: Array  # () participating slosh mass [kg]
    pendulum_length: Array  # () pendulum length [m] (pendulum model)
    omega: Array  # () natural frequency [rad/s] (spring-mass model)
    zeta: Array  # () damping ratio
    x_attach: Array  # () slosh attachment body-x [m]


@immutable_dataclass
class EngineParams:
    """Per-engine thrust, geometry, and fault parameters.

    N engines; ``positions`` are gimbal-point body coords (x negative aft). Gimballed engines
    (Ng of them) are indexed by the static ``gimbal_idx`` baked at build (in meta, not here).
    engine_on / thrust_scale / tvc_bias are truth-only faults (nominal 1/1/0).
    """

    positions: Array  # (N,3) gimbal-point body coords [m]
    thrust_vac: Array  # (N,) per-engine vacuum thrust T1 [N]
    throttle_min: Array  # (N,) per-engine deep-throttle floor r
    isp: Array  # (N,) per-engine specific impulse [s]
    nozzle_exit_area: Array  # (N,) per-engine nozzle exit area [m^2]
    cant: Array  # (N,2) fixed per-engine [pitch,yaw] cant [rad] (0 => +x_body)
    engine_on: Array  # (N,) truth-only engine-out flag in {0,1} (nominal 1)
    thrust_scale: Array  # (N,) truth-only partial-thrust multiplier (nominal 1)
    tvc_bias: Array  # (Ng,2) truth-only per-gimballed-engine gimbal bias (nominal 0)


@immutable_dataclass
class ActuatorParams:
    """Cluster-common actuator dynamics & limits. Per-engine thrust quantities live in
    EngineParams; this holds the shared servo/throttle/RCS scalars."""

    p_sea: Array  # () sea-level ambient pressure [Pa]
    # TVC: 2nd-order gimbal
    tvc_omega: Array  # () natural freq [rad/s] (~2π·6 Hz)
    tvc_zeta: Array  # () damping ratio (~0.707)
    tvc_rate_max_rad_s: Array  # () rate limit [rad/s]
    tvc_deflection_max_rad: Array  # () deflection limit [rad]
    # throttle: 1st-order lag + rate limit
    throttle_tau: Array  # () lag time constant [s]
    throttle_rate_max: Array  # () throttle-fraction rate limit [1/s]
    # grid-fin servo: 2nd-order
    fin_omega: Array  # () [rad/s]
    fin_zeta: Array  # ()
    fin_rate_max_rad_s: Array  # () [rad/s]
    fin_deflection_max_rad: Array  # () [rad]
    # RCS
    rcs_thrust: Array  # () per-axis RCS force magnitude [N]
    rcs_arm: Array  # () moment arm for RCS torque [m]
    rcs_isp: Array  # () cold-gas Isp [s]
    rcs_fuel0: Array  # () initial RCS propellant budget [kg]
    # fault injection (truth-only; nominal=0). per-engine faults live in EngineParams.
    fin_lock: Array  # (4,) per-fin lock flag (1=locked at current angle, 0=healthy)


@immutable_dataclass
class PlantParams:
    """Aggregate of all plant parameters passed to ``step``."""

    geom: VehicleGeom
    mass: MassPropsParams
    aero: AeroParams
    atmosphere: AtmosphereParams
    wind: WindParams
    earth: EarthRotationParams
    slosh: SloshParams
    actuator: ActuatorParams
    engines: EngineParams
    g_ned: Array  # (3,) apparent gravity at the local-NED origin [m/s^2]
    plant_dt: Array  # () integration step [s]


# --------------------------------------------------------------------------- #
# Episode result (post-processed output; numpy arrays)
# --------------------------------------------------------------------------- #
@dataclass
class EpisodeResult:
    """Stacked trajectory + termination + KPIs returned by every runner."""

    t: Array  # (T,)
    r_ned: Array  # (T, 3)
    v_ned: Array  # (T, 3)
    q_bn: Array  # (T, 4)
    w_body: Array  # (T, 3)
    m: Array  # (T,)
    phase: Array  # (T,) int phase tag (KPI)
    aux: dict  # per-step diagnostics (forces, alpha, mach, ...)
    termination_reason: int
    touchdown_index: int
    # KPIs
    landing_position_error: float  # horizontal miss distance at touchdown [m]
    landing_speed: float  # speed at touchdown [m/s]
    landing_tilt: float  # attitude tilt from vertical at touchdown [rad]
    fuel_margin: float  # propellant remaining at termination [kg]
    success: bool
    violation_flags: dict
    # Actuator TRUTH trajectories (logged for replay/visualization; None if not captured)
    throttle_act: Array | None = None  # (T, N) actual per-engine throttle fraction
    tvc: Array | None = None  # (T, Ng, 2) actual per-gimballed-engine [pitch, yaw] [rad]
    fin: Array | None = None  # (T, 4) actual grid-fin deflections [rad]
    # Touchdown KPIs split out for honest contact assessment: a survivable landing needs a low
    # SINK rate AND low horizontal velocity AND low spin — not just a small speed magnitude.
    landing_speed_vertical: float | None = None  # |v_z| sink rate at touchdown [m/s]
    landing_speed_horizontal: float | None = None  # |v_xy| lateral speed at touchdown [m/s]
    landing_rate: float | None = None  # |w_body| body angular rate at touchdown [rad/s]
    # 3-tier contact quality (sim.loop): 'soft' (genuine set-down, sink≤2 / lateral≤1 m/s), 'hard'
    # (structurally survived but firm, 2–6 m/s), 'crash' (>structural limits); None if no contact.
    landing_class: str | None = None


def quat_identity() -> Array:
    return np.array([1.0, 0.0, 0.0, 0.0])

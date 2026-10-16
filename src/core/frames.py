"""Native coordinate-frame and quaternion transforms.

Conventions:
* navigation is NED (North, East, Down), with altitude ``-z``;
* body ``+x`` points toward the nose, ``+y`` right, and ``+z`` down when level;
* ``q_bn = [w, x, y, z]`` is scalar-first Hamilton and maps body coordinates to NED;
* every ``dcm_<source>_to_<destination>`` maps source coordinates to destination coordinates;
* LOS/velocity/wind ``+x`` follows the named direction and uses NED-down as its axis reference.

All numerical operations in this module use the compiled C++ implementation.
"""

from __future__ import annotations

import _rlv_sim_native as _native
import numpy as np

_NED_DOWN = np.array([0.0, 0.0, 1.0])


def _as_dcm(matrix, name="matrix") -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3), got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def quat_mul(q1, q2):
    """Hamilton product ``q1 ⊗ q2`` (scalar first)."""
    return _native.quat_mul(q1, q2)


def quat_conj(q):
    """Quaternion conjugate, and inverse for a unit quaternion."""
    return _native.quat_conj(q)


def quat_normalize(q):
    """Normalize a quaternion; zero and non-finite values are invalid."""
    return _native.quat_normalize(q)


def dcm_from_quat(q):
    """Normalized body-to-NED DCM: ``v_ned = C_bn @ v_body``."""
    return _native.dcm_from_quat(q)


def quat_kinematics(q, w_body):
    """Return ``q_dot = 0.5 q_bn ⊗ [0, ω_body]``."""
    return _native.quat_kinematics(q, w_body)


def omega_matrix(w_body):
    """Matrix ``Ω`` such that ``q_dot = 0.5 Ω q``."""
    return _native.omega_matrix(w_body)


def skew(vector):
    """Skew matrix ``[v]x`` with ``[v]x @ a == cross(v, a)``."""
    return _native.skew(vector)


def quat_from_euler(roll, pitch, yaw):
    """Intrinsic ZYX roll/pitch/yaw [rad] to a body-to-NED quaternion."""
    return _native.quat_from_euler(float(roll), float(pitch), float(yaw))


def euler_from_quat(q):
    """Body-to-NED quaternion to ``(roll, pitch, yaw)`` [rad], intrinsic ZYX."""
    return _native.euler_from_quat(q)


def quat_from_axis_angle(axis, angle):
    """Axis-angle rotation to a body-to-NED quaternion."""
    return _native.quat_from_axis_angle(axis, float(angle))


def transform_vectors(matrix, vectors):
    """Apply a DCM to one vector or a C-contiguous batch with shape ``(..., 3)``."""
    return _native.transform_vectors(_as_dcm(matrix), np.asarray(vectors, dtype=float))


def inverse_dcm(matrix):
    """Inverse of a valid DCM (its transpose)."""
    return _as_dcm(matrix).T.copy()


def compose_dcm(c_destination_intermediate, c_intermediate_source):
    """Compose coordinate transforms into ``C_destination_source``."""
    return _native.compose_dcm(
        _as_dcm(c_destination_intermediate, "c_destination_intermediate"),
        _as_dcm(c_intermediate_source, "c_intermediate_source"),
    )


def body_to_ned(vector_body, q_bn):
    return transform_vectors(dcm_from_quat(q_bn), vector_body)


def ned_to_body(vector_ned, q_bn):
    return transform_vectors(inverse_dcm(dcm_from_quat(q_bn)), vector_ned)


def dcm_ned_to_enu():
    """Constant NED-to-ENU DCM: ``[N,E,D] -> [E,N,U]``."""
    return _native.dcm_ned_to_enu()


def ned_to_enu(vector_ned):
    return transform_vectors(dcm_ned_to_enu(), vector_ned)


def enu_to_ned(vector_enu):
    return transform_vectors(dcm_ned_to_enu(), vector_enu)


def dcm_ned_to_velocity(velocity_ned, down_reference_ned=_NED_DOWN):
    """NED-to-velocity DCM; zero speed is invalid."""
    return _native.dcm_ned_to_velocity(velocity_ned, down_reference_ned)


def ned_to_velocity(vector_ned, velocity_ned, down_reference_ned=_NED_DOWN):
    return transform_vectors(dcm_ned_to_velocity(velocity_ned, down_reference_ned), vector_ned)


def velocity_to_ned(vector_velocity, velocity_ned, down_reference_ned=_NED_DOWN):
    dcm = inverse_dcm(dcm_ned_to_velocity(velocity_ned, down_reference_ned))
    return transform_vectors(dcm, vector_velocity)


def dcm_ned_to_wind(air_velocity_ned, down_reference_ned=_NED_DOWN):
    """NED-to-wind DCM using air-relative velocity."""
    return dcm_ned_to_velocity(air_velocity_ned, down_reference_ned)


def dcm_body_to_wind(q_bn, air_velocity_ned, down_reference_ned=_NED_DOWN):
    """Body-to-wind DCM ``C_wb = C_wn C_nb``."""
    return compose_dcm(dcm_ned_to_wind(air_velocity_ned, down_reference_ned), dcm_from_quat(q_bn))


def dcm_ned_to_los(observer_ned, target_ned, down_reference_ned=_NED_DOWN):
    """NED-to-LOS DCM; coincident observer and target are invalid."""
    return _native.dcm_ned_to_los(observer_ned, target_ned, down_reference_ned)


def ned_to_los(vector_ned, observer_ned, target_ned, down_reference_ned=_NED_DOWN):
    dcm = dcm_ned_to_los(observer_ned, target_ned, down_reference_ned)
    return transform_vectors(dcm, vector_ned)


def los_to_ned(vector_los, observer_ned, target_ned, down_reference_ned=_NED_DOWN):
    dcm = inverse_dcm(dcm_ned_to_los(observer_ned, target_ned, down_reference_ned))
    return transform_vectors(dcm, vector_los)


def dcm_ecef_to_ned(latitude_rad, longitude_rad):
    """ECEF-to-NED DCM at a WGS-84 geodetic latitude/longitude."""
    return _native.dcm_ecef_to_ned(float(latitude_rad), float(longitude_rad))


def dcm_ecef_to_enu(latitude_rad, longitude_rad):
    return compose_dcm(dcm_ned_to_enu(), dcm_ecef_to_ned(latitude_rad, longitude_rad))


def lla_to_ecef(latitude_rad, longitude_rad, altitude_m):
    """WGS-84 geodetic latitude/longitude [rad], altitude [m] to ECEF [m]."""
    return _native.lla_to_ecef(float(latitude_rad), float(longitude_rad), float(altitude_m))


def ecef_to_lla(position_ecef):
    """ECEF [m] to ``(latitude_rad, longitude_rad, altitude_m)``."""
    return _native.ecef_to_lla(position_ecef)


def ecef_to_ned(position_ecef, origin_ecef, latitude_rad, longitude_rad):
    """ECEF point to local NED displacement from ``origin_ecef``."""
    return _native.ecef_to_ned(position_ecef, origin_ecef, latitude_rad, longitude_rad)


def ned_to_ecef(position_ned, origin_ecef, latitude_rad, longitude_rad):
    """Local NED displacement to an ECEF point."""
    return _native.ned_to_ecef(position_ned, origin_ecef, latitude_rad, longitude_rad)

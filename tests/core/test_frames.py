"""Frame-transform tests: the single source of truth must be exact."""

from __future__ import annotations

import numpy as np

from core import frames


def _rand_quat(rng):
    return frames.quat_normalize(rng.normal(size=4))


def test_dcm_orthonormal():
    rng = np.random.default_rng(4)
    for _ in range(20):
        q = _rand_quat(rng)
        C = frames.dcm_from_quat(q)
        assert np.allclose(C @ C.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(np.asarray(C)), 1.0, atol=1e-12)


def test_euler_quat_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(20):
        roll = rng.uniform(-np.pi, np.pi)
        pitch = rng.uniform(-np.pi / 2 + 0.05, np.pi / 2 - 0.05)  # avoid gimbal lock
        yaw = rng.uniform(-np.pi, np.pi)
        q = frames.quat_from_euler(roll, pitch, yaw)
        r2, p2, y2 = frames.euler_from_quat(q)
        assert np.isclose(float(r2), roll, atol=1e-9)
        assert np.isclose(float(p2), pitch, atol=1e-9)
        assert np.isclose(float(y2), yaw, atol=1e-9)


def test_nose_up_quaternion():
    """pitch=+90deg => body-x (nose) points to NED up (-z); base-first descent."""
    q = frames.quat_from_euler(0.0, np.pi / 2, 0.0)
    C = np.asarray(frames.dcm_from_quat(q))
    assert np.allclose(C[:, 0], [0.0, 0.0, -1.0], atol=1e-12)


def test_quat_kinematics_vs_finite_difference():
    """q̇ from quat_kinematics matches finite-difference of a rotating quaternion."""
    q0 = _rand_quat(np.random.default_rng(3))
    w = np.array([0.3, -0.7, 0.5])
    dt = 1e-6
    # propagate by small rotation: q(dt) = q0 ⊗ exp(½ w dt)
    dq = frames.quat_from_axis_angle(w / np.linalg.norm(w), np.linalg.norm(w) * dt)
    q1 = frames.quat_normalize(frames.quat_mul(q0, dq))
    qdot_fd = (np.asarray(q1) - np.asarray(q0)) / dt
    qdot = np.asarray(frames.quat_kinematics(q0, w))
    assert np.allclose(qdot, qdot_fd, atol=1e-5)
    assert np.allclose(0.5 * frames.omega_matrix(np.asarray(w)) @ np.asarray(q0), qdot, atol=1e-12)


def test_dcm_vs_quat_mul_rotation():
    """C(q) v_body == vector part of q ⊗ [0,v] ⊗ q* (active rotation consistency)."""
    q = _rand_quat(np.random.default_rng(7))
    v = np.random.default_rng(8).normal(size=3)
    vq = np.concatenate([np.zeros(1), v])
    rotated = frames.quat_mul(frames.quat_mul(q, vq), frames.quat_conj(q))[1:]
    assert np.allclose(np.asarray(rotated), np.asarray(frames.dcm_from_quat(q) @ v), atol=1e-10)


def test_ned_enu_axes_and_roundtrip():
    ned = np.array([12.0, -3.0, 7.0])
    enu = frames.ned_to_enu(ned)
    assert np.array_equal(enu, [-3.0, 12.0, -7.0])
    assert np.array_equal(frames.enu_to_ned(enu), ned)


def test_body_ned_roundtrip_uses_declared_q_bn_direction():
    q_bn = frames.quat_from_euler(0.3, -0.2, 1.1)
    v_body = np.array([4.0, -2.0, 8.0])
    v_ned = frames.body_to_ned(v_body, q_bn)
    assert np.allclose(frames.ned_to_body(v_ned, q_bn), v_body, atol=1e-13)


def test_velocity_frame_axes_and_roundtrip():
    velocity_ned = np.array([3.0, 4.0, 2.0])
    C_vn = frames.dcm_ned_to_velocity(velocity_ned)
    assert np.allclose(C_vn @ C_vn.T, np.eye(3), atol=1e-14)
    assert np.isclose(np.linalg.det(C_vn), 1.0, atol=1e-14)
    # The velocity itself has only a +x component in its own frame.
    assert np.allclose(C_vn @ velocity_ned, [np.linalg.norm(velocity_ned), 0.0, 0.0])
    sample = np.array([-9.0, 2.0, 1.0])
    assert np.allclose(
        frames.velocity_to_ned(frames.ned_to_velocity(sample, velocity_ned), velocity_ned),
        sample,
        atol=1e-14,
    )


def test_vertical_velocity_frame_has_deterministic_north_heading_datum():
    C_vn = frames.dcm_ned_to_velocity(np.array([0.0, 0.0, 10.0]))
    assert np.allclose(C_vn[0], [0.0, 0.0, 1.0])  # +x follows down-going velocity
    assert np.allclose(C_vn[1], [0.0, 1.0, 0.0])  # +y is east for north datum
    assert np.allclose(C_vn @ C_vn.T, np.eye(3), atol=1e-14)


def test_nearly_vertical_velocity_keeps_north_heading_datum():
    C_vn = frames.dcm_ned_to_velocity(np.array([5e-13, 0.0, 1.0]))
    assert np.allclose(C_vn[1], [0.0, 1.0, 0.0], atol=1e-14)
    assert np.allclose(C_vn @ C_vn.T, np.eye(3), atol=1e-14)
    assert np.isclose(np.linalg.det(C_vn), 1.0, atol=1e-14)


def test_los_frame_horizontal_geometry_and_roundtrip():
    observer = np.array([10.0, 20.0, -100.0])
    target = np.array([110.0, 20.0, -100.0])
    C_ln = frames.dcm_ned_to_los(observer, target)
    assert np.allclose(C_ln, np.eye(3))  # north LOS -> axes coincide with NED
    relative = target - observer
    assert np.allclose(C_ln @ relative, [100.0, 0.0, 0.0])
    sample = np.array([3.0, -7.0, 1.0])
    assert np.allclose(
        frames.los_to_ned(frames.ned_to_los(sample, observer, target), observer, target),
        sample,
    )


def test_los_rejects_coincident_points_and_velocity_rejects_zero_speed():
    with np.testing.assert_raises(ValueError):
        frames.dcm_ned_to_los(np.zeros(3), np.zeros(3))
    with np.testing.assert_raises(ValueError):
        frames.dcm_ned_to_velocity(np.zeros(3))


def test_wind_frame_composes_body_to_ned_with_ned_to_wind():
    q_bn = frames.quat_from_euler(-0.1, 0.2, 0.7)
    air_velocity_ned = np.array([120.0, -30.0, 15.0])
    v_body = np.array([1.0, 2.0, -4.0])
    direct = frames.dcm_body_to_wind(q_bn, air_velocity_ned) @ v_body
    staged = frames.dcm_ned_to_wind(air_velocity_ned) @ frames.body_to_ned(v_body, q_bn)
    assert np.allclose(direct, staged, atol=1e-14)


def test_wgs84_lla_ecef_and_local_ned_roundtrips():
    latitude, longitude, altitude = np.deg2rad(37.5665), np.deg2rad(126.9780), 123.4
    origin = frames.lla_to_ecef(latitude, longitude, altitude)
    lat2, lon2, alt2 = frames.ecef_to_lla(origin)
    assert np.isclose(lat2, latitude, atol=1e-13)
    assert np.isclose(lon2, longitude, atol=1e-13)
    assert np.isclose(alt2, altitude, atol=1e-6)

    point_ned = np.array([150.0, -40.0, 25.0])
    point_ecef = frames.ned_to_ecef(point_ned, origin, latitude, longitude)
    assert np.allclose(
        frames.ecef_to_ned(point_ecef, origin, latitude, longitude), point_ned, atol=1e-8
    )
    with np.testing.assert_raises(ValueError):
        frames.lla_to_ecef(np.pi, longitude, altitude)


def test_geodetic_latitude_limits():
    for latitude in (-np.pi / 2, np.pi / 2):
        assert np.all(np.isfinite(frames.dcm_ecef_to_ned(latitude, 0.0)))
        assert np.all(np.isfinite(frames.lla_to_ecef(latitude, 0.0, 0.0)))
        outside = np.nextafter(latitude, np.copysign(np.inf, latitude))
        with np.testing.assert_raises(ValueError):
            frames.dcm_ecef_to_ned(outside, 0.0)
        with np.testing.assert_raises(ValueError):
            frames.lla_to_ecef(outside, 0.0, 0.0)


def test_native_batch_transform_preserves_leading_shape():
    q_bn = frames.quat_from_euler(0.2, 0.1, -0.4)
    C_bn = frames.dcm_from_quat(q_bn)
    vectors = np.arange(60.0).reshape(4, 5, 3)
    got = frames.transform_vectors(C_bn, vectors)
    expected = np.einsum("ij,...j->...i", C_bn, vectors)
    assert got.shape == vectors.shape
    assert np.allclose(got, expected, atol=1e-13)


def test_invalid_quaternions_fail_loud_on_native_path():
    with np.testing.assert_raises(ValueError):
        frames.quat_normalize(np.zeros(4))
    with np.testing.assert_raises(ValueError):
        frames.dcm_from_quat(np.array([np.nan, 0.0, 0.0, 0.0]))

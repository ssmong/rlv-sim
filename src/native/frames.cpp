#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

void bind_native_plant(py::module_& m);
void bind_native_scvx(py::module_& m);

namespace {

using Vec3 = std::array<double, 3>;
using Quat = std::array<double, 4>;
using Mat3 = std::array<double, 9>;

constexpr double kTiny = 1e-15;
constexpr double kWgs84A = 6378137.0;
constexpr double kWgs84F = 1.0 / 298.257223563;
constexpr double kWgs84E2 = kWgs84F * (2.0 - kWgs84F);

template <std::size_t N>
std::array<double, N> read_vector(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& value,
    const char* name) {
    if (value.ndim() != 1 || value.shape(0) != static_cast<py::ssize_t>(N)) {
        throw py::value_error(std::string(name) + " must have shape (" + std::to_string(N) + ",)");
    }
    auto v = value.template unchecked<1>();
    std::array<double, N> out{};
    for (std::size_t i = 0; i < N; ++i) {
        out[i] = v(static_cast<py::ssize_t>(i));
        if (!std::isfinite(out[i])) {
            throw py::value_error(std::string(name) + " must contain only finite values");
        }
    }
    return out;
}

py::array_t<double> array3(const Vec3& v) {
    py::array_t<double> out(3);
    auto a = out.mutable_unchecked<1>();
    for (int i = 0; i < 3; ++i) a(i) = v[i];
    return out;
}

py::array_t<double> array4(const Quat& q) {
    py::array_t<double> out(4);
    auto a = out.mutable_unchecked<1>();
    for (int i = 0; i < 4; ++i) a(i) = q[i];
    return out;
}

py::array_t<double> array33(const Mat3& m) {
    py::array_t<double> out({3, 3});
    auto a = out.mutable_unchecked<2>();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) a(i, j) = m[3 * i + j];
    }
    return out;
}

double dot(const Vec3& a, const Vec3& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]};
}

double norm(const Vec3& v) { return std::sqrt(dot(v, v)); }

Vec3 normalized(const Vec3& v, const char* name) {
    const double n = norm(v);
    if (!(n > kTiny) || !std::isfinite(n)) {
        throw py::value_error(std::string(name) + " must have non-zero finite norm");
    }
    return {v[0] / n, v[1] / n, v[2] / n};
}

Quat normalized(const Quat& q, const char* name) {
    const double n = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    if (!(n > kTiny) || !std::isfinite(n)) {
        throw py::value_error(std::string(name) + " must have non-zero finite norm");
    }
    return {q[0] / n, q[1] / n, q[2] / n, q[3] / n};
}

Vec3 matvec(const Mat3& m, const Vec3& v) {
    return {m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
            m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
            m[6] * v[0] + m[7] * v[1] + m[8] * v[2]};
}

Mat3 transpose(const Mat3& m) {
    return {m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]};
}

Mat3 multiply(const Mat3& a, const Mat3& b) {
    Mat3 out{};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) out[3 * i + j] += a[3 * i + k] * b[3 * k + j];
        }
    }
    return out;
}

Quat quat_mul_value(const Quat& a, const Quat& b) {
    return {a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
            a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
            a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
            a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]};
}

Mat3 dcm_from_unit_quat(const Quat& q) {
    const auto [w, x, y, z] = q;
    return {1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z),
            2.0 * (x * z + w * y), 2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x),
            2.0 * (x * z - w * y), 2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y)};
}

// Rows are destination-frame axes expressed in NED, hence v_dst = C_dst_ned v_ned.
Mat3 direction_frame(const Vec3& direction_ned, const Vec3& down_reference_ned) {
    const Vec3 x = normalized(direction_ned, "direction_ned");
    const Vec3 down = normalized(down_reference_ned, "down_reference_ned");
    Vec3 y_raw = cross(down, x);
    if (norm(y_raw) <= 1e-12) {
        // Heading is undefined for a vertical line. Use North as the deterministic datum when
        // possible, then project the resulting lateral axis exactly normal to the direction.
        std::array<Vec3, 3> candidates{{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}}};
        auto best = candidates[0];
        double best_alignment = std::abs(dot(best, down));
        for (const auto& c : candidates) {
            const double alignment = std::abs(dot(c, down));
            if (alignment < best_alignment) {
                best = c;
                best_alignment = alignment;
            }
        }
        const Vec3 datum_lateral = cross(down, best);
        y_raw = {datum_lateral[0] - dot(datum_lateral, x) * x[0],
                 datum_lateral[1] - dot(datum_lateral, x) * x[1],
                 datum_lateral[2] - dot(datum_lateral, x) * x[2]};
    }
    const Vec3 y = normalized(y_raw, "frame lateral axis");
    const Vec3 z = cross(x, y);
    return {x[0], x[1], x[2], y[0], y[1], y[2], z[0], z[1], z[2]};
}

Mat3 dcm_ecef_to_ned_value(double latitude, double longitude) {
    if (!std::isfinite(latitude) || !std::isfinite(longitude)) {
        throw py::value_error("latitude_rad and longitude_rad must be finite");
    }
    if (latitude < -0.5 * std::numbers::pi || latitude > 0.5 * std::numbers::pi) {
        throw py::value_error("latitude_rad must be in [-pi/2, pi/2]");
    }
    const double sp = std::sin(latitude), cp = std::cos(latitude);
    const double sl = std::sin(longitude), cl = std::cos(longitude);
    return {-sp * cl, -sp * sl, cp, -sl, cl, 0.0, -cp * cl, -cp * sl, -sp};
}

py::array_t<double> transform_vectors_impl(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& matrix,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& vectors) {
    if (matrix.ndim() != 2 || matrix.shape(0) != 3 || matrix.shape(1) != 3) {
        throw py::value_error("matrix must have shape (3, 3)");
    }
    if (vectors.ndim() < 1 || vectors.shape(vectors.ndim() - 1) != 3) {
        throw py::value_error("vectors must have shape (..., 3)");
    }
    auto m = matrix.unchecked<2>();
    std::size_t count = 1;
    for (py::ssize_t i = 0; i < vectors.ndim() - 1; ++i) {
        count *= static_cast<std::size_t>(vectors.shape(i));
    }
    std::vector<py::ssize_t> shape;
    for (py::ssize_t i = 0; i < vectors.ndim(); ++i) shape.push_back(vectors.shape(i));
    py::array_t<double> out(shape);
    const double* src = vectors.data();
    double* dst = out.mutable_data();
    py::gil_scoped_release release;
    for (std::size_t n = 0; n < count; ++n) {
        const double x = src[3 * n], y = src[3 * n + 1], z = src[3 * n + 2];
        dst[3 * n] = m(0, 0) * x + m(0, 1) * y + m(0, 2) * z;
        dst[3 * n + 1] = m(1, 0) * x + m(1, 1) * y + m(1, 2) * z;
        dst[3 * n + 2] = m(2, 0) * x + m(2, 1) * y + m(2, 2) * z;
    }
    return out;
}

}  // namespace

PYBIND11_MODULE(_rlv_sim_native, m) {
    m.doc() = "Native C++ coordinate transforms for rlv-sim";

    m.def("quat_mul", [](const py::array_t<double, py::array::c_style | py::array::forcecast>& a,
                         const py::array_t<double, py::array::c_style | py::array::forcecast>& b) {
        return array4(quat_mul_value(read_vector<4>(a, "q1"), read_vector<4>(b, "q2")));
    });
    m.def("quat_conj", [](const py::array_t<double, py::array::c_style | py::array::forcecast>& v) {
        const Quat q = read_vector<4>(v, "q");
        return array4({q[0], -q[1], -q[2], -q[3]});
    });
    m.def("quat_normalize", [](const py::array_t<double, py::array::c_style | py::array::forcecast>& v) {
        return array4(normalized(read_vector<4>(v, "q"), "q"));
    });
    m.def("dcm_from_quat", [](const py::array_t<double, py::array::c_style | py::array::forcecast>& v) {
        return array33(dcm_from_unit_quat(normalized(read_vector<4>(v, "q"), "q")));
    });
    m.def("quat_kinematics",
          [](const py::array_t<double, py::array::c_style | py::array::forcecast>& q_value,
             const py::array_t<double, py::array::c_style | py::array::forcecast>& w_value) {
              const Quat q = read_vector<4>(q_value, "q");
              const Vec3 w = read_vector<3>(w_value, "w_body");
              const Quat out = quat_mul_value(q, {0.0, w[0], w[1], w[2]});
              return array4({0.5 * out[0], 0.5 * out[1], 0.5 * out[2], 0.5 * out[3]});
          });
    m.def("omega_matrix",
          [](const py::array_t<double, py::array::c_style | py::array::forcecast>& value) {
              const Vec3 w = read_vector<3>(value, "w_body");
              const std::array<double, 16> matrix{
                  0.0, -w[0], -w[1], -w[2],
                  w[0], 0.0, w[2], -w[1],
                  w[1], -w[2], 0.0, w[0],
                  w[2], w[1], -w[0], 0.0};
              py::array_t<double> out({4, 4});
              std::copy(matrix.begin(), matrix.end(), out.mutable_data());
              return out;
          });
    m.def("skew", [](const py::array_t<double, py::array::c_style | py::array::forcecast>& value) {
        const Vec3 v = read_vector<3>(value, "v");
        return array33({0.0, -v[2], v[1], v[2], 0.0, -v[0], -v[1], v[0], 0.0});
    });
    m.def("quat_from_euler", [](double roll, double pitch, double yaw) {
        const double cr = std::cos(0.5 * roll), sr = std::sin(0.5 * roll);
        const double cp = std::cos(0.5 * pitch), sp = std::sin(0.5 * pitch);
        const double cy = std::cos(0.5 * yaw), sy = std::sin(0.5 * yaw);
        return array4({cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                       cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy});
    });
    m.def("euler_from_quat", [](const py::array_t<double, py::array::c_style | py::array::forcecast>& value) {
        const auto [w, x, y, z] = normalized(read_vector<4>(value, "q"), "q");
        const double roll = std::atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y));
        const double pitch = std::asin(std::clamp(2.0 * (w * y - z * x), -1.0, 1.0));
        const double yaw = std::atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));
        return py::make_tuple(roll, pitch, yaw);
    });
    m.def("quat_from_axis_angle",
          [](const py::array_t<double, py::array::c_style | py::array::forcecast>& value,
             double angle) {
              if (!std::isfinite(angle)) throw py::value_error("angle must be finite");
              if (std::abs(angle) <= kTiny) return array4({1.0, 0.0, 0.0, 0.0});
              const Vec3 axis = normalized(read_vector<3>(value, "axis"), "axis");
              const double s = std::sin(0.5 * angle);
              return array4({std::cos(0.5 * angle), axis[0] * s, axis[1] * s, axis[2] * s});
          });

    m.def("dcm_ned_to_enu", []() { return array33({0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0}); });
    m.def("dcm_ned_to_velocity",
          [](const py::array_t<double, py::array::c_style | py::array::forcecast>& velocity,
             const py::array_t<double, py::array::c_style | py::array::forcecast>& down) {
              return array33(direction_frame(read_vector<3>(velocity, "velocity_ned"),
                                             read_vector<3>(down, "down_reference_ned")));
          }, py::arg("velocity_ned"), py::arg("down_reference_ned"));
    m.def("dcm_ned_to_los",
          [](const py::array_t<double, py::array::c_style | py::array::forcecast>& observer,
             const py::array_t<double, py::array::c_style | py::array::forcecast>& target,
             const py::array_t<double, py::array::c_style | py::array::forcecast>& down) {
              const Vec3 o = read_vector<3>(observer, "observer_ned");
              const Vec3 t = read_vector<3>(target, "target_ned");
              return array33(direction_frame({t[0] - o[0], t[1] - o[1], t[2] - o[2]},
                                             read_vector<3>(down, "down_reference_ned")));
          }, py::arg("observer_ned"), py::arg("target_ned"), py::arg("down_reference_ned"));
    m.def("dcm_ecef_to_ned", [](double latitude_rad, double longitude_rad) {
        return array33(dcm_ecef_to_ned_value(latitude_rad, longitude_rad));
    });
    m.def("lla_to_ecef", [](double latitude_rad, double longitude_rad, double altitude_m) {
        if (!std::isfinite(latitude_rad) || !std::isfinite(longitude_rad) || !std::isfinite(altitude_m)) {
            throw py::value_error("geodetic coordinates must be finite");
        }
        if (latitude_rad < -0.5 * std::numbers::pi || latitude_rad > 0.5 * std::numbers::pi) {
            throw py::value_error("latitude_rad must be in [-pi/2, pi/2]");
        }
        const double sp = std::sin(latitude_rad), cp = std::cos(latitude_rad);
        const double sl = std::sin(longitude_rad), cl = std::cos(longitude_rad);
        const double n = kWgs84A / std::sqrt(1.0 - kWgs84E2 * sp * sp);
        return array3({(n + altitude_m) * cp * cl, (n + altitude_m) * cp * sl,
                       (n * (1.0 - kWgs84E2) + altitude_m) * sp});
    });
    m.def("ecef_to_lla", [](const py::array_t<double, py::array::c_style | py::array::forcecast>& value) {
        const Vec3 p = read_vector<3>(value, "position_ecef");
        const double longitude = std::atan2(p[1], p[0]);
        const double radius = std::hypot(p[0], p[1]);
        if (radius < kTiny && std::abs(p[2]) < kTiny) {
            throw py::value_error("ECEF origin has undefined geodetic coordinates");
        }
        double latitude = std::atan2(p[2], radius * (1.0 - kWgs84E2));
        double altitude = 0.0;
        for (int i = 0; i < 10; ++i) {
            const double sp = std::sin(latitude);
            const double n = kWgs84A / std::sqrt(1.0 - kWgs84E2 * sp * sp);
            if (radius > kTiny) {
                altitude = radius / std::cos(latitude) - n;
            } else {
                altitude = std::abs(p[2]) - n * (1.0 - kWgs84E2);
            }
            const double next = std::atan2(p[2], radius * (1.0 - kWgs84E2 * n / (n + altitude)));
            if (std::abs(next - latitude) < 1e-14) {
                latitude = next;
                break;
            }
            latitude = next;
        }
        return py::make_tuple(latitude, longitude, altitude);
    });
    m.def("ecef_to_ned",
          [](const py::array_t<double, py::array::c_style | py::array::forcecast>& position,
             const py::array_t<double, py::array::c_style | py::array::forcecast>& origin,
             double latitude_rad, double longitude_rad) {
              const Vec3 p = read_vector<3>(position, "position_ecef");
              const Vec3 o = read_vector<3>(origin, "origin_ecef");
              return array3(matvec(dcm_ecef_to_ned_value(latitude_rad, longitude_rad),
                                   {p[0] - o[0], p[1] - o[1], p[2] - o[2]}));
          });
    m.def("ned_to_ecef",
          [](const py::array_t<double, py::array::c_style | py::array::forcecast>& position,
             const py::array_t<double, py::array::c_style | py::array::forcecast>& origin,
             double latitude_rad, double longitude_rad) {
              const Vec3 p = read_vector<3>(position, "position_ned");
              const Vec3 o = read_vector<3>(origin, "origin_ecef");
              const Vec3 delta = matvec(transpose(dcm_ecef_to_ned_value(latitude_rad, longitude_rad)), p);
              return array3({o[0] + delta[0], o[1] + delta[1], o[2] + delta[2]});
          });
    m.def("compose_dcm",
          [](const py::array_t<double, py::array::c_style | py::array::forcecast>& a_value,
             const py::array_t<double, py::array::c_style | py::array::forcecast>& b_value) {
              if (a_value.ndim() != 2 || a_value.shape(0) != 3 || a_value.shape(1) != 3 ||
                  b_value.ndim() != 2 || b_value.shape(0) != 3 || b_value.shape(1) != 3) {
                  throw py::value_error("both DCMs must have shape (3, 3)");
              }
              auto a = a_value.unchecked<2>();
              auto b = b_value.unchecked<2>();
              Mat3 av{}, bv{};
              for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) {
                  av[3 * i + j] = a(i, j); bv[3 * i + j] = b(i, j);
              }
              return array33(multiply(av, bv));
          });
    m.def("transform_vectors", &transform_vectors_impl, py::arg("matrix"), py::arg("vectors"));
    bind_native_plant(m);
    bind_native_scvx(m);
}

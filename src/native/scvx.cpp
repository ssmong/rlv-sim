#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

extern "C" {
#include "qoco.h"
}

namespace py = pybind11;

namespace {

constexpr int NX = 7;
constexpr int NW = 4;
constexpr double kHalfPi = 1.5707963267948966;

using State = std::array<double, NX>;
// Delivered physical thrust [Fx, Fy, Fz, magnitude]. Keeping force ZOH matches the actuator;
// a specific-thrust ZOH would require an impossible within-node force decrease at the throttle floor.
using Control = std::array<double, NW>;

struct Params {
  State x0{};
  std::array<double, 3> gravity{};
  std::array<double, 3> target{};
  double alpha{};
  double rho_min{};
  double rho_max{};
  double m_dry{};
  double tf{};
  int nodes{};
  double beta0{};
  std::vector<double> drag_mach_grid;
  std::vector<double> drag_beta_grid;
  double sound_speed{};
  double scale_height{};
  double backpressure0{};
  double theta_max{};
  double gamma_gs{};
  double terminal_vz_max{};
  bool has_theta{};
  bool has_glideslope{};
  bool has_terminal_vz{};
  bool terminal_upright{};
  double thrust_rate_max{};
  bool has_thrust_rate{};
  int max_accepted_iters{};
  double trust_initial{};
  double virtual_weight{};
  double tolerance{};
  double pos_scale{};
  double vel_scale{};
  double accel_scale{};
};

struct Trajectory {
  std::vector<State> x;
  std::vector<Control> w;
};

struct Metrics {
  double merit{};
  double fuel_cost{};
  double defect_sum{};
  double defect_max{};
  double path_violation_sum{};
  double path_violation_max{};
  double slack_gap{};
  double thrust_violation{};
};

struct IterationRecord {
  int attempt{};
  int accepted_iteration{};
  bool accepted{};
  double trust{};
  double ratio{};
  double actual_reduction{};
  double predicted_reduction{};
  double scaled_step{};
  double virtual_norm{};
  double dynamics_defect{};
  double path_violation{};
  double wall_time{};
  double backend_time{};
  int solver_iterations{};
  int solver_status{};
};

struct Result {
  std::string status{"unsolved"};
  std::string solver_status{"unsolved"};
  bool solver_feasible{};
  bool feasible{};
  bool converged{};
  Trajectory trajectory;
  std::vector<State> virtual_control;
  std::vector<IterationRecord> history;
  int accepted_iterations{};
  int solver_attempts{};
  double solve_time{};
  double final_trust{};
  double virtual_norm{};
  double dynamics_defect{};
  double path_violation{};
  double slack_gap{};
  double thrust_violation{};
  double terminal_position{};
  double terminal_velocity{};
  double terminal_velocity_violation{};
  double rollout_terminal_position{};
  double rollout_terminal_velocity{};
  double rollout_terminal_velocity_violation{};
  std::vector<std::string> failures;
};

double norm3(const double* value) {
  return std::sqrt(value[0] * value[0] + value[1] * value[1] + value[2] * value[2]);
}

double norm7(const State& value) {
  double sum = 0.0;
  for (double item : value) sum += item * item;
  return std::sqrt(sum);
}

double atmosphere_factor(double down, double scale_height) {
  if (!(scale_height > 0.0)) return 1.0;
  return std::exp(std::min(down, 0.0) / scale_height);
}

double drag_beta(const Params& p, double speed) {
  if (p.drag_mach_grid.empty()) return p.beta0;
  const double mach = speed / p.sound_speed;
  const auto upper = std::upper_bound(p.drag_mach_grid.begin(), p.drag_mach_grid.end(), mach);
  if (upper == p.drag_mach_grid.begin()) return p.drag_beta_grid.front();
  if (upper == p.drag_mach_grid.end()) return p.drag_beta_grid.back();
  const std::size_t hi = static_cast<std::size_t>(upper - p.drag_mach_grid.begin());
  const std::size_t lo = hi - 1;
  const double fraction =
      (mach - p.drag_mach_grid[lo]) / (p.drag_mach_grid[hi] - p.drag_mach_grid[lo]);
  return p.drag_beta_grid[lo] * (1.0 - fraction) + p.drag_beta_grid[hi] * fraction;
}

State rhs(const Params& p, const State& state, const Control& control) {
  State out{};
  out[0] = state[3];
  out[1] = state[4];
  out[2] = state[5];
  const double inv_mass = std::exp(-state[6]);
  const double factor = atmosphere_factor(state[2], p.scale_height);
  const double speed = norm3(&state[3]);
  const double drag_gain = drag_beta(p, speed) * factor * inv_mass * speed;
  for (int j = 0; j < 3; ++j) {
    out[3 + j] = p.gravity[j] + control[j] * inv_mass - drag_gain * state[3 + j];
  }
  out[6] = -p.alpha * (control[3] + p.backpressure0 * factor) * inv_mass;
  return out;
}

State add_scaled(const State& a, const State& b, double scale) {
  State out{};
  for (int i = 0; i < NX; ++i) out[i] = a[i] + scale * b[i];
  return out;
}

State integrate_interval(
    const Params& p, const State& initial, const Control& control, double dt, int substeps) {
  State state = initial;
  const double h = dt / static_cast<double>(substeps);
  for (int step = 0; step < substeps; ++step) {
    const State k1 = rhs(p, state, control);
    const State k2 = rhs(p, add_scaled(state, k1, 0.5 * h), control);
    const State k3 = rhs(p, add_scaled(state, k2, 0.5 * h), control);
    const State k4 = rhs(p, add_scaled(state, k3, h), control);
    for (int i = 0; i < NX; ++i) {
      state[i] += h * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0;
    }
  }
  return state;
}

State scale_state(const Params& p, const State& physical) {
  State out{};
  for (int j = 0; j < 3; ++j) {
    out[j] = (physical[j] - p.target[j]) / p.pos_scale;
    out[3 + j] = physical[3 + j] / p.vel_scale;
  }
  out[6] = physical[6];
  return out;
}

State unscale_state(const Params& p, const State& scaled) {
  State out{};
  for (int j = 0; j < 3; ++j) {
    out[j] = p.target[j] + p.pos_scale * scaled[j];
    out[3 + j] = p.vel_scale * scaled[3 + j];
  }
  out[6] = scaled[6];
  return out;
}

Control scale_control(const Params& p, const Control& physical) {
  Control out{};
  for (int j = 0; j < NW; ++j) out[j] = physical[j] / p.rho_max;
  return out;
}

Control unscale_control(const Params& p, const Control& scaled) {
  Control out{};
  for (int j = 0; j < NW; ++j) out[j] = p.rho_max * scaled[j];
  return out;
}

State discrete_map_scaled(
    const Params& p, const State& scaled_state, const Control& scaled_control, int substeps = 4) {
  const State physical = unscale_state(p, scaled_state);
  const Control control = unscale_control(p, scaled_control);
  return scale_state(p, integrate_interval(p, physical, control, p.tf / p.nodes, substeps));
}

struct Linearization {
  std::array<double, NX * NX> A{};
  std::array<double, NX * NW> B{};
  State c{};
};

Linearization linearize(const Params& p, const State& x, const Control& w) {
  constexpr double eps = 1e-5;
  Linearization out;
  const State center = discrete_map_scaled(p, x, w);
  for (int col = 0; col < NX; ++col) {
    State plus = x;
    State minus = x;
    plus[col] += eps;
    minus[col] -= eps;
    const State fp = discrete_map_scaled(p, plus, w);
    const State fm = discrete_map_scaled(p, minus, w);
    for (int row = 0; row < NX; ++row) {
      out.A[row * NX + col] = (fp[row] - fm[row]) / (2.0 * eps);
    }
  }
  for (int col = 0; col < NW; ++col) {
    Control plus = w;
    Control minus = w;
    plus[col] += eps;
    minus[col] -= eps;
    const State fp = discrete_map_scaled(p, x, plus);
    const State fm = discrete_map_scaled(p, x, minus);
    for (int row = 0; row < NX; ++row) {
      out.B[row * NW + col] = (fp[row] - fm[row]) / (2.0 * eps);
    }
  }
  for (int row = 0; row < NX; ++row) {
    double affine = center[row];
    for (int col = 0; col < NX; ++col) affine -= out.A[row * NX + col] * x[col];
    for (int col = 0; col < NW; ++col) affine -= out.B[row * NW + col] * w[col];
    out.c[row] = affine;
  }
  return out;
}

double available_thrust(const Params& p, const State& x_scaled, double vacuum_thrust) {
  const State x = unscale_state(p, x_scaled);
  const double factor = atmosphere_factor(x[2], p.scale_height);
  const double delivered = std::max(vacuum_thrust - p.backpressure0 * factor, 0.0);
  return delivered / p.rho_max;
}

std::array<double, 3> bound_linearization(
    const Params& p, const State& x, double vacuum_thrust) {
  constexpr double eps = 1e-5;
  const double center = available_thrust(p, x, vacuum_thrust);
  State plus = x;
  State minus = x;
  plus[2] += eps;
  minus[2] -= eps;
  const double d_down =
      (available_thrust(p, plus, vacuum_thrust) -
       available_thrust(p, minus, vacuum_thrust)) /
      (2.0 * eps);
  plus = x;
  minus = x;
  plus[6] += eps;
  minus[6] -= eps;
  const double d_z =
      (available_thrust(p, plus, vacuum_thrust) -
       available_thrust(p, minus, vacuum_thrust)) /
      (2.0 * eps);
  return {center, d_down, d_z};
}

struct Triplet {
  int row{};
  int col{};
  double value{};
};

struct Csc {
  int rows{};
  int cols{};
  std::vector<double> values;
  std::vector<int> row_indices;
  std::vector<int> col_ptr;
};

Csc make_csc(int rows, int cols, std::vector<Triplet> entries) {
  std::sort(entries.begin(), entries.end(), [](const Triplet& a, const Triplet& b) {
    return std::tie(a.col, a.row) < std::tie(b.col, b.row);
  });
  Csc out;
  out.rows = rows;
  out.cols = cols;
  out.col_ptr.assign(cols + 1, 0);
  for (const Triplet& entry : entries) {
    if (entry.row < 0 || entry.row >= rows || entry.col < 0 || entry.col >= cols) {
      throw std::runtime_error("native SCvx produced an invalid sparse-matrix index");
    }
    if (std::abs(entry.value) <= 1e-15) continue;
    if (!out.row_indices.empty() && out.col_ptr[entry.col + 1] > 0 &&
        out.row_indices.back() == entry.row) {
      out.values.back() += entry.value;
    } else {
      out.row_indices.push_back(entry.row);
      out.values.push_back(entry.value);
      ++out.col_ptr[entry.col + 1];
    }
  }
  for (int col = 0; col < cols; ++col) out.col_ptr[col + 1] += out.col_ptr[col];
  return out;
}

struct Layout {
  int N{};
  int off_x{};
  int off_w{};
  int off_nu{};
  int off_eta{};
  int off_path{};
  int n{};

  explicit Layout(int nodes) : N(nodes) {
    off_x = 0;
    off_w = NX * (N + 1);
    off_nu = off_w + NW * N;
    off_eta = off_nu + NX * N;
    off_path = off_eta + N;
    n = off_path + N;
  }
  int x(int k, int j) const { return off_x + NX * k + j; }
  int w(int k, int j) const { return off_w + NW * k + j; }
  int nu(int k, int j) const { return off_nu + NX * k + j; }
  int eta(int k) const { return off_eta + k; }
  int path(int k) const { return off_path + k; }
};

struct Socp {
  Csc P;
  std::vector<double> c;
  Csc A;
  std::vector<double> b;
  Csc G;
  std::vector<double> h;
  int l{};
  std::vector<int> q;
  Layout layout;

  explicit Socp(int nodes) : layout(nodes) {}
};

Socp build_subproblem(const Params& p, const Trajectory& reference, double trust) {
  Socp problem(p.nodes);
  const Layout& ix = problem.layout;
  std::vector<Triplet> a_entries;
  std::vector<double> b;
  auto add_eq = [&](const std::vector<std::pair<int, double>>& coeffs, double rhs_value) {
    const int row = static_cast<int>(b.size());
    for (const auto& [col, value] : coeffs) a_entries.push_back({row, col, value});
    b.push_back(rhs_value);
  };

  const State initial_scaled = scale_state(p, p.x0);
  for (int j = 0; j < NX; ++j) add_eq({{ix.x(0, j), 1.0}}, initial_scaled[j]);

  for (int k = 0; k < p.nodes; ++k) {
    const Linearization lin = linearize(p, reference.x[k], reference.w[k]);
    for (int row = 0; row < NX; ++row) {
      std::vector<std::pair<int, double>> coeffs;
      coeffs.reserve(NX + NW + 2);
      coeffs.push_back({ix.x(k + 1, row), 1.0});
      for (int col = 0; col < NX; ++col) {
        coeffs.push_back({ix.x(k, col), -lin.A[row * NX + col]});
      }
      for (int col = 0; col < NW; ++col) {
        coeffs.push_back({ix.w(k, col), -lin.B[row * NW + col]});
      }
      coeffs.push_back({ix.nu(k, row), -1.0});
      add_eq(coeffs, lin.c[row]);
    }
  }

  for (int j = 0; j < 3; ++j) add_eq({{ix.x(p.nodes, j), 1.0}}, 0.0);
  const int exact_velocity_components = p.has_terminal_vz ? 2 : 3;
  for (int j = 0; j < exact_velocity_components; ++j) {
    add_eq({{ix.x(p.nodes, 3 + j), 1.0}}, 0.0);
  }
  if (p.terminal_upright) {
    add_eq({{ix.w(p.nodes - 1, 0), 1.0}}, 0.0);
    add_eq({{ix.w(p.nodes - 1, 1), 1.0}}, 0.0);
  }

  std::vector<Triplet> g_entries;
  std::vector<double> h;
  auto add_ineq = [&](const std::vector<std::pair<int, double>>& coeffs, double rhs_value) {
    const int row = static_cast<int>(h.size());
    for (const auto& [col, value] : coeffs) g_entries.push_back({row, col, value});
    h.push_back(rhs_value);
  };

  add_ineq({{ix.x(p.nodes, 6), -1.0}}, -std::log(p.m_dry));
  if (p.has_terminal_vz) {
    add_ineq(
        {{ix.x(p.nodes, 5), 1.0}}, p.terminal_vz_max / p.vel_scale);
    add_ineq({{ix.x(p.nodes, 5), -1.0}}, 0.0);
  }

  for (int k = 0; k < p.nodes; ++k) {
    add_ineq({{ix.path(k), -1.0}}, 0.0);
    const auto upper = bound_linearization(p, reference.x[k], p.rho_max);
    const double upper_constant =
        upper[0] - upper[1] * reference.x[k][2] - upper[2] * reference.x[k][6];
    add_ineq(
        {{ix.w(k, 3), 1.0},
         {ix.x(k, 2), -upper[1]},
         {ix.x(k, 6), -upper[2]},
         {ix.path(k), -1.0}},
        upper_constant);

    const auto lower = bound_linearization(p, reference.x[k], p.rho_min);
    const double lower_constant =
        lower[0] - lower[1] * reference.x[k][2] - lower[2] * reference.x[k][6];
    const double u_norm = norm3(reference.w[k].data());
    const std::array<double, 3> direction =
        u_norm > 1e-9
            ? std::array<double, 3>{reference.w[k][0] / u_norm,
                                    reference.w[k][1] / u_norm,
                                    reference.w[k][2] / u_norm}
            : std::array<double, 3>{0.0, 0.0, -1.0};
    add_ineq(
        {{ix.x(k, 2), lower[1]},
         {ix.x(k, 6), lower[2]},
         {ix.w(k, 0), -direction[0]},
         {ix.w(k, 1), -direction[1]},
         {ix.w(k, 2), -direction[2]},
         {ix.path(k), -1.0}},
        -lower_constant);

    if (p.has_theta) {
      add_ineq(
          {{ix.w(k, 2), 1.0}, {ix.w(k, 3), std::cos(p.theta_max)}}, 0.0);
    }
  }

  if (p.has_thrust_rate) {
    const double cap = p.thrust_rate_max * (p.tf / p.nodes) / p.rho_max;
    for (int k = 0; k < p.nodes - 1; ++k) {
      add_ineq({{ix.w(k + 1, 3), 1.0}, {ix.w(k, 3), -1.0}}, cap);
      add_ineq({{ix.w(k + 1, 3), -1.0}, {ix.w(k, 3), 1.0}}, cap);
    }
  }

  const double delta = std::clamp(trust / p.accel_scale, 1e-6, 10.0);
  const std::array<double, NX> state_step_scale = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1};
  for (int k = 0; k <= p.nodes; ++k) {
    for (int j = 0; j < NX; ++j) {
      const double radius = delta * state_step_scale[j];
      add_ineq({{ix.x(k, j), 1.0}}, reference.x[k][j] + radius);
      add_ineq({{ix.x(k, j), -1.0}}, -reference.x[k][j] + radius);
    }
  }
  for (int k = 0; k < p.nodes; ++k) {
    for (int j = 0; j < NW; ++j) {
      add_ineq({{ix.w(k, j), 1.0}}, reference.w[k][j] + delta);
      add_ineq({{ix.w(k, j), -1.0}}, -reference.w[k][j] + delta);
    }
  }

  problem.l = static_cast<int>(h.size());
  auto add_soc = [&](const std::vector<std::pair<std::vector<std::pair<int, double>>, double>>& rows) {
    for (const auto& [coeffs, rhs_value] : rows) add_ineq(coeffs, rhs_value);
    problem.q.push_back(static_cast<int>(rows.size()));
  };

  for (int k = 0; k < p.nodes; ++k) {
    add_soc({
        {{{ix.w(k, 3), -1.0}}, 0.0},
        {{{ix.w(k, 0), -1.0}}, 0.0},
        {{{ix.w(k, 1), -1.0}}, 0.0},
        {{{ix.w(k, 2), -1.0}}, 0.0},
    });
    std::vector<std::pair<std::vector<std::pair<int, double>>, double>> virtual_rows;
    virtual_rows.push_back({{{ix.eta(k), -1.0}}, 0.0});
    for (int j = 0; j < NX; ++j) {
      virtual_rows.push_back({{{ix.nu(k, j), -1.0}}, 0.0});
    }
    add_soc(virtual_rows);
    if (p.has_glideslope) {
      const double tangent = std::tan(p.gamma_gs);
      add_soc({
          {{{ix.x(k, 2), 1.0}}, 0.0},
          {{{ix.x(k, 0), -tangent}}, 0.0},
          {{{ix.x(k, 1), -tangent}}, 0.0},
      });
    }
  }
  problem.c.assign(ix.n, 0.0);
  problem.c[ix.x(p.nodes, 6)] = -1.0;
  for (int k = 0; k < p.nodes; ++k) {
    problem.c[ix.eta(k)] = p.virtual_weight;
    problem.c[ix.path(k)] = p.virtual_weight;
  }
  problem.P = make_csc(ix.n, ix.n, {});
  problem.A = make_csc(static_cast<int>(b.size()), ix.n, std::move(a_entries));
  problem.b = std::move(b);
  problem.G = make_csc(static_cast<int>(h.size()), ix.n, std::move(g_entries));
  problem.h = std::move(h);
  return problem;
}

Trajectory unpack_trajectory(const Layout& ix, const double* values) {
  Trajectory out;
  out.x.resize(ix.N + 1);
  out.w.resize(ix.N);
  for (int k = 0; k <= ix.N; ++k) {
    for (int j = 0; j < NX; ++j) out.x[k][j] = values[ix.x(k, j)];
  }
  for (int k = 0; k < ix.N; ++k) {
    for (int j = 0; j < NW; ++j) out.w[k][j] = values[ix.w(k, j)];
  }
  return out;
}

double terminal_velocity_violation(const Params& p, const State& state) {
  const double lateral = std::hypot(state[3], state[4]);
  if (!p.has_terminal_vz) return norm3(&state[3]);
  return std::max({lateral, -state[5], state[5] - p.terminal_vz_max, 0.0});
}

constexpr double kScaledConstraintTolerance = 1e-5;

double thrust_tolerance(const Params& p) {
  return std::max(1.0, 1e-5 * p.rho_max);
}

Metrics evaluate(const Params& p, const Trajectory& trajectory) {
  Metrics out;
  out.fuel_cost = -trajectory.x.back()[6];
  for (int k = 0; k < p.nodes; ++k) {
    const State mapped = discrete_map_scaled(p, trajectory.x[k], trajectory.w[k]);
    State defect{};
    for (int j = 0; j < NX; ++j) defect[j] = trajectory.x[k + 1][j] - mapped[j];
    const double defect_norm = norm7(defect);
    out.defect_sum += defect_norm;
    out.defect_max = std::max(out.defect_max, defect_norm);

    const double u_norm = norm3(trajectory.w[k].data());
    out.slack_gap = std::max(out.slack_gap, std::abs(u_norm - trajectory.w[k][3]));
    const double lower = available_thrust(p, trajectory.x[k], p.rho_min);
    const double upper = available_thrust(p, trajectory.x[k], p.rho_max);
    const double lower_violation = std::max(lower - u_norm, 0.0);
    const double upper_violation = std::max(trajectory.w[k][3] - upper, 0.0);
    const double cone_violation = std::max(u_norm - trajectory.w[k][3], 0.0);
    double violation = lower_violation + upper_violation + cone_violation;
    if (p.has_theta) {
      violation += std::max(trajectory.w[k][2] + std::cos(p.theta_max) * trajectory.w[k][3], 0.0);
    }
    if (p.has_glideslope) {
      violation += std::max(
          trajectory.x[k][2] +
              std::tan(p.gamma_gs) * std::hypot(trajectory.x[k][0], trajectory.x[k][1]),
          0.0);
    }
    out.path_violation_sum += violation;
    out.path_violation_max = std::max(out.path_violation_max, violation);
    out.thrust_violation = std::max({out.thrust_violation, lower_violation, upper_violation});
  }
  if (p.has_thrust_rate) {
    const double cap = p.thrust_rate_max * (p.tf / p.nodes) / p.rho_max;
    for (int k = 0; k < p.nodes - 1; ++k) {
      const double violation =
          std::max(std::abs(trajectory.w[k + 1][3] - trajectory.w[k][3]) - cap, 0.0);
      out.path_violation_sum += violation;
      out.path_violation_max = std::max(out.path_violation_max, violation);
    }
  }
  if (p.terminal_upright) {
    const double violation = std::hypot(trajectory.w.back()[0], trajectory.w.back()[1]);
    out.path_violation_sum += violation;
    out.path_violation_max = std::max(out.path_violation_max, violation);
  }
  out.merit = out.fuel_cost +
              p.virtual_weight * (out.defect_sum + out.path_violation_sum);
  return out;
}

double scaled_step(const Trajectory& a, const Trajectory& b) {
  double value = 0.0;
  for (std::size_t k = 0; k < a.x.size(); ++k) {
    for (int j = 0; j < NX; ++j) {
      const double scale = j == 6 ? 0.1 : 1.0;
      value = std::max(value, std::abs(a.x[k][j] - b.x[k][j]) / scale);
    }
  }
  for (std::size_t k = 0; k < a.w.size(); ++k) {
    for (int j = 0; j < NW; ++j) value = std::max(value, std::abs(a.w[k][j] - b.w[k][j]));
  }
  return value;
}

class QocoWorkspace {
 public:
  QocoWorkspace()
      : solver_(static_cast<QOCOSolver*>(std::calloc(1, sizeof(QOCOSolver)))) {
    if (!solver_) throw std::bad_alloc();
  }
  QocoWorkspace(const QocoWorkspace&) = delete;
  QocoWorkspace& operator=(const QocoWorkspace&) = delete;
  ~QocoWorkspace() {
    if (ready_) {
      qoco_cleanup(solver_);
    } else {
      std::free(solver_);
    }
  }

  bool solve(const Socp& problem) {
    qoco_set_csc(
        &A_, problem.A.rows, problem.A.cols, static_cast<int>(problem.A.values.size()),
        const_cast<double*>(problem.A.values.data()), const_cast<int*>(problem.A.col_ptr.data()),
        const_cast<int*>(problem.A.row_indices.data()));
    qoco_set_csc(
        &G_, problem.G.rows, problem.G.cols, static_cast<int>(problem.G.values.size()),
        const_cast<double*>(problem.G.values.data()), const_cast<int*>(problem.G.col_ptr.data()),
        const_cast<int*>(problem.G.row_indices.data()));
    set_default_settings(&settings_);
    settings_.verbose = 0;
    settings_.ruiz_iters = 10;
    settings_.max_iters = 100;
    settings_.abstol = 1e-8;
    settings_.reltol = 1e-8;
    settings_.abstol_inacc = 1e-6;
    settings_.reltol_inacc = 1e-6;
    const int error = qoco_setup(
        solver_, problem.layout.n, problem.G.rows, problem.A.rows, nullptr,
        const_cast<double*>(problem.c.data()), &A_, const_cast<double*>(problem.b.data()), &G_,
        const_cast<double*>(problem.h.data()), problem.l, static_cast<int>(problem.q.size()),
        const_cast<int*>(problem.q.data()), &settings_);
    if (error != QOCO_NO_ERROR) {
      throw std::runtime_error("QOCO setup failed: " + std::to_string(error));
    }
    ready_ = true;
    qoco_solve(solver_);
    return solver_->sol->status == QOCO_SOLVED || solver_->sol->status == QOCO_SOLVED_INACCURATE;
  }

  const QOCOSolution& solution() const { return *solver_->sol; }

 private:
  bool ready_{};
  QOCOSolver* solver_{};
  QOCOSettings settings_{};
  QOCOCscMatrix A_{};
  QOCOCscMatrix G_{};
};

std::string qoco_status_name(int status) {
  switch (status) {
    case QOCO_SOLVED: return "optimal";
    case QOCO_SOLVED_INACCURATE: return "optimal_inaccurate";
    case QOCO_NUMERICAL_ERROR: return "numerical_error";
    case QOCO_MAX_ITER: return "max_iters";
    default: return "unsolved";
  }
}

Result solve_native(const Params& p, Trajectory reference) {
  Result result;
  result.trajectory = reference;
  result.virtual_control.assign(p.nodes, State{});
  Metrics reference_metrics = evaluate(p, reference);
  double trust = p.trust_initial;
  result.final_trust = trust;
  constexpr double rho_reject = 0.10;
  constexpr double rho_shrink = 0.25;
  constexpr double rho_expand = 0.90;
  constexpr double shrink = 0.5;
  constexpr double expand = 2.0;
  const double min_trust = std::max(1e-4 * p.accel_scale, 1e-6);
  const double max_trust = 10.0 * p.accel_scale;
  const int max_attempts = std::max(2 * p.max_accepted_iters, p.max_accepted_iters + 4);

  for (int attempt = 0;
       attempt < max_attempts && result.accepted_iterations < p.max_accepted_iters; ++attempt) {
    const Socp problem = build_subproblem(p, reference, trust);
    QocoWorkspace workspace;
    const auto wall_start = std::chrono::steady_clock::now();
    const bool solved = workspace.solve(problem);
    const double wall_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - wall_start).count();
    result.solve_time += workspace.solution().solve_time_sec;
    ++result.solver_attempts;
    result.solver_status = qoco_status_name(workspace.solution().status);
    if (!solved) {
      result.status = "solver_" + result.solver_status;
      result.solver_feasible = false;
      result.final_trust = trust;
      break;
    }
    result.solver_feasible = true;
    const Trajectory candidate = unpack_trajectory(problem.layout, workspace.solution().x);
    const Metrics candidate_metrics = evaluate(p, candidate);
    double model_virtual = 0.0;
    double model_path = 0.0;
    for (int k = 0; k < p.nodes; ++k) {
      model_virtual += std::max(workspace.solution().x[problem.layout.eta(k)], 0.0);
      model_path += std::max(workspace.solution().x[problem.layout.path(k)], 0.0);
    }
    const double model_cost = -candidate.x.back()[6] +
                              p.virtual_weight * (model_virtual + model_path);
    const double actual = reference_metrics.merit - candidate_metrics.merit;
    const double predicted = reference_metrics.merit - model_cost;
    double ratio = -std::numeric_limits<double>::infinity();
    if (predicted > 1e-10) ratio = actual / predicted;
    const double step = scaled_step(reference, candidate);
    const bool stationary_reference =
        predicted <= 1e-10 && std::abs(actual) <= p.tolerance &&
        reference_metrics.defect_max <= p.tolerance &&
        reference_metrics.path_violation_max <= kScaledConstraintTolerance &&
        reference_metrics.slack_gap * p.rho_max <= thrust_tolerance(p) &&
        reference_metrics.thrust_violation * p.rho_max <= thrust_tolerance(p);
    const bool accepted = predicted > 1e-10 && std::isfinite(ratio) && ratio >= rho_reject;
    result.history.push_back({
        attempt,
        result.accepted_iterations,
        accepted,
        trust,
        ratio,
        actual,
        predicted,
        step,
        model_virtual,
        candidate_metrics.defect_max,
        candidate_metrics.path_violation_max,
        wall_seconds,
        workspace.solution().solve_time_sec,
        workspace.solution().iters,
        workspace.solution().status,
    });

    if (stationary_reference) {
      result.converged = true;
      result.status = "converged";
      result.dynamics_defect = reference_metrics.defect_max;
      result.path_violation = reference_metrics.path_violation_max;
      result.slack_gap = reference_metrics.slack_gap * p.rho_max;
      result.thrust_violation = reference_metrics.thrust_violation * p.rho_max;
      result.final_trust = trust;
      break;
    }

    if (!accepted) {
      trust *= shrink;
      result.final_trust = trust;
      if (trust < min_trust) {
        result.status = "trust_region_stalled";
        result.final_trust = trust;
        break;
      }
      continue;
    }

    reference = candidate;
    reference_metrics = candidate_metrics;
    ++result.accepted_iterations;
    if (ratio < rho_shrink) {
      trust = std::max(trust * shrink, min_trust);
    } else if (ratio >= rho_expand) {
      trust = std::min(trust * expand, max_trust);
    }
    result.trajectory = reference;
    for (int k = 0; k < p.nodes; ++k) {
      for (int j = 0; j < NX; ++j) {
        result.virtual_control[k][j] = workspace.solution().x[problem.layout.nu(k, j)];
      }
    }
    result.virtual_norm = model_virtual;
    result.dynamics_defect = candidate_metrics.defect_max;
    result.path_violation = candidate_metrics.path_violation_max;
    result.slack_gap = candidate_metrics.slack_gap * p.rho_max;
    result.thrust_violation = candidate_metrics.thrust_violation * p.rho_max;
    result.final_trust = trust;

    if (step <= p.tolerance && std::abs(actual) <= p.tolerance &&
        std::abs(predicted) <= p.tolerance && candidate_metrics.defect_max <= p.tolerance &&
        candidate_metrics.path_violation_max <= kScaledConstraintTolerance &&
        candidate_metrics.slack_gap * p.rho_max <= thrust_tolerance(p) &&
        candidate_metrics.thrust_violation * p.rho_max <= thrust_tolerance(p) &&
        model_virtual <= p.tolerance) {
      result.converged = true;
      result.status = "converged";
      break;
    }
  }
  if (!result.converged && result.status == "unsolved") result.status = "max_iters";

  const Trajectory& trajectory = result.trajectory;
  if (trajectory.x.empty()) return result;
  const State terminal = unscale_state(p, trajectory.x.back());
  result.terminal_position = std::sqrt(
      std::pow(terminal[0] - p.target[0], 2) + std::pow(terminal[1] - p.target[1], 2) +
      std::pow(terminal[2] - p.target[2], 2));
  result.terminal_velocity = norm3(&terminal[3]);
  result.terminal_velocity_violation = terminal_velocity_violation(p, terminal);

  State rollout = p.x0;
  for (int k = 0; k < p.nodes; ++k) {
    rollout = integrate_interval(p, rollout, unscale_control(p, trajectory.w[k]), p.tf / p.nodes, 16);
  }
  result.rollout_terminal_position = std::sqrt(
      std::pow(rollout[0] - p.target[0], 2) + std::pow(rollout[1] - p.target[1], 2) +
      std::pow(rollout[2] - p.target[2], 2));
  result.rollout_terminal_velocity = norm3(&rollout[3]);
  result.rollout_terminal_velocity_violation = terminal_velocity_violation(p, rollout);

  constexpr double terminal_tolerance = 1e-2;
  if (!result.solver_feasible) result.failures.push_back("conic_solver");
  if (!result.converged) result.failures.push_back("successive_convergence");
  if (result.virtual_norm > p.tolerance) result.failures.push_back("virtual_control");
  if (result.dynamics_defect > p.tolerance) result.failures.push_back("dynamics_defect");
  if (result.path_violation > kScaledConstraintTolerance) {
    result.failures.push_back("path_constraints");
  }
  if (result.slack_gap > thrust_tolerance(p)) {
    result.failures.push_back("relaxation_gap");
  }
  if (result.thrust_violation > thrust_tolerance(p)) {
    result.failures.push_back("thrust_bounds");
  }
  if (std::exp(terminal[6]) + 1e-6 < p.m_dry) result.failures.push_back("dry_mass");
  if (result.terminal_position > terminal_tolerance) result.failures.push_back("terminal_position");
  if (result.terminal_velocity_violation > terminal_tolerance) {
    result.failures.push_back("terminal_velocity");
  }
  if (result.rollout_terminal_position > terminal_tolerance) {
    result.failures.push_back("rollout_terminal_position");
  }
  if (result.rollout_terminal_velocity_violation > terminal_tolerance) {
    result.failures.push_back("rollout_terminal_velocity");
  }
  result.feasible = result.failures.empty();
  if (result.converged && !result.feasible) result.status = "postcheck_failed";
  return result;
}

double required_double(const py::dict& problem, const char* key) {
  if (!problem.contains(key)) throw py::key_error(key);
  return py::cast<double>(problem[key]);
}

int required_int(const py::dict& problem, const char* key) {
  if (!problem.contains(key)) throw py::key_error(key);
  return py::cast<int>(problem[key]);
}

bool optional_bool(const py::dict& problem, const char* key, bool fallback = false) {
  return problem.contains(key) && !problem[key].is_none() ? py::cast<bool>(problem[key]) : fallback;
}

double optional_double(const py::dict& problem, const char* key, double fallback) {
  return problem.contains(key) && !problem[key].is_none() ? py::cast<double>(problem[key]) : fallback;
}

std::array<double, 3> array3(const py::handle& value, const char* name) {
  const auto array = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(value);
  if (!array || array.ndim() != 1 || array.shape(0) != 3) {
    throw py::value_error(std::string(name) + " must have shape (3,)");
  }
  const auto view = array.unchecked<1>();
  return {view(0), view(1), view(2)};
}

std::vector<double> optional_vector(const py::dict& problem, const char* key) {
  if (!problem.contains(key) || problem[key].is_none()) return {};
  const auto array =
      py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(problem[key]);
  if (!array || array.ndim() != 1) {
    throw py::value_error(std::string(key) + " must be one-dimensional");
  }
  const auto view = array.unchecked<1>();
  std::vector<double> values(static_cast<std::size_t>(view.shape(0)));
  for (py::ssize_t i = 0; i < view.shape(0); ++i) values[static_cast<std::size_t>(i)] = view(i);
  return values;
}

Params parse_params(const py::dict& problem) {
  Params p;
  const auto r0 = array3(problem["r0"], "r0");
  const auto v0 = array3(problem["v0"], "v0");
  p.gravity = array3(problem["g"], "g");
  p.target = array3(problem["target"], "target");
  for (int j = 0; j < 3; ++j) {
    p.x0[j] = r0[j];
    p.x0[3 + j] = v0[j];
  }
  const double m0 = required_double(problem, "m0");
  if (!(m0 > 0.0)) throw py::value_error("m0 must be positive");
  p.x0[6] = std::log(m0);
  p.alpha = required_double(problem, "alpha");
  p.rho_min = required_double(problem, "rho_min");
  p.rho_max = required_double(problem, "rho_max");
  p.m_dry = required_double(problem, "m_dry");
  p.tf = required_double(problem, "tf");
  p.nodes = required_int(problem, "N");
  p.beta0 = optional_double(problem, "beta0", 0.0);
  p.drag_mach_grid = optional_vector(problem, "drag_mach_grid");
  p.drag_beta_grid = optional_vector(problem, "drag_beta_grid");
  p.sound_speed = optional_double(problem, "sound_speed", 340.3);
  p.scale_height = optional_double(problem, "scale_height", 0.0);
  p.backpressure0 = optional_double(problem, "backpressure0", 0.0);
  p.theta_max = optional_double(problem, "theta_max", 0.0);
  p.gamma_gs = optional_double(problem, "gamma_gs", 0.0);
  p.terminal_vz_max = optional_double(problem, "terminal_vz_max", 0.0);
  p.has_theta = problem.contains("theta_max") && !problem["theta_max"].is_none();
  p.has_glideslope = problem.contains("gamma_gs") && !problem["gamma_gs"].is_none();
  p.has_terminal_vz =
      problem.contains("terminal_vz_max") && !problem["terminal_vz_max"].is_none();
  p.terminal_upright = optional_bool(problem, "terminal_upright");
  p.thrust_rate_max = optional_double(problem, "thrust_rate_max", 0.0);
  if (problem.contains("slew_rate_max") && !problem["slew_rate_max"].is_none()) {
    throw py::value_error("slew_rate_max is not implemented for physical-thrust SCvx");
  }
  p.has_thrust_rate =
      problem.contains("thrust_rate_max") && !problem["thrust_rate_max"].is_none();
  p.max_accepted_iters = required_int(problem, "iters");
  p.trust_initial = required_double(problem, "trust");
  p.virtual_weight = required_double(problem, "w_nu");
  p.tolerance = required_double(problem, "tol");
  p.pos_scale = required_double(problem, "pos_scale");
  p.vel_scale = required_double(problem, "vel_scale");
  p.accel_scale = required_double(problem, "accel_scale");
  const bool finite_scalars =
      std::isfinite(m0) && std::isfinite(p.alpha) && std::isfinite(p.rho_min) &&
      std::isfinite(p.rho_max) && std::isfinite(p.m_dry) && std::isfinite(p.tf) &&
      std::isfinite(p.beta0) && std::isfinite(p.sound_speed) &&
      std::isfinite(p.scale_height) &&
      std::isfinite(p.backpressure0) && std::isfinite(p.trust_initial) &&
      std::isfinite(p.virtual_weight) && std::isfinite(p.tolerance) &&
      std::isfinite(p.pos_scale) && std::isfinite(p.vel_scale) &&
      std::isfinite(p.accel_scale);
  if (!finite_scalars || p.nodes < 2 || p.max_accepted_iters < 1 || !(p.tf > 0.0) ||
      !(p.alpha >= 0.0) || !(p.rho_min >= 0.0) || !(p.rho_max > p.rho_min) ||
      !(p.m_dry > 0.0) || !(m0 >= p.m_dry) || !(p.beta0 >= 0.0) ||
      !(p.sound_speed > 0.0) || !(p.scale_height >= 0.0) || !(p.backpressure0 >= 0.0) ||
      !(p.trust_initial > 0.0) || !(p.virtual_weight > 0.0) || !(p.tolerance > 0.0) ||
      !(p.pos_scale > 0.0) || !(p.vel_scale > 0.0) || !(p.accel_scale > 0.0)) {
    throw py::value_error("invalid native SCvx problem dimensions or scales");
  }
  if (p.drag_mach_grid.size() != p.drag_beta_grid.size() ||
      (!p.drag_mach_grid.empty() && p.drag_mach_grid.size() < 2)) {
    throw py::value_error("drag grids must have equal length of at least two");
  }
  for (std::size_t i = 0; i < p.drag_mach_grid.size(); ++i) {
    if (!std::isfinite(p.drag_mach_grid[i]) || !std::isfinite(p.drag_beta_grid[i]) ||
        p.drag_beta_grid[i] < 0.0 ||
        (i > 0 && !(p.drag_mach_grid[i] > p.drag_mach_grid[i - 1]))) {
      throw py::value_error("drag Mach knots must increase and drag coefficients must be finite");
    }
  }
  for (double value : p.x0) {
    if (!std::isfinite(value)) throw py::value_error("initial state must be finite");
  }
  for (double value : p.gravity) {
    if (!std::isfinite(value)) throw py::value_error("gravity must be finite");
  }
  for (double value : p.target) {
    if (!std::isfinite(value)) throw py::value_error("target must be finite");
  }
  if (p.has_theta &&
      (!std::isfinite(p.theta_max) || p.theta_max < 0.0 || p.theta_max >= kHalfPi)) {
    throw py::value_error("theta_max must be in [0, pi/2)");
  }
  if (p.has_glideslope &&
      (!std::isfinite(p.gamma_gs) || p.gamma_gs < 0.0 || p.gamma_gs >= kHalfPi)) {
    throw py::value_error("gamma_gs must be in [0, pi/2)");
  }
  if (p.has_terminal_vz &&
      (!std::isfinite(p.terminal_vz_max) || p.terminal_vz_max < 0.0)) {
    throw py::value_error("terminal_vz_max must be non-negative");
  }
  if (p.has_thrust_rate &&
      (!std::isfinite(p.thrust_rate_max) || p.thrust_rate_max <= 0.0)) {
    throw py::value_error("thrust_rate_max must be positive");
  }
  return p;
}

Trajectory parse_init(const Params& p, const py::dict& init) {
  auto require_2d = [&](const char* key, int rows, int cols) {
    if (!init.contains(key)) throw py::key_error(key);
    const auto array = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(init[key]);
    if (!array || array.ndim() != 2 || array.shape(0) != rows || array.shape(1) != cols) {
      throw py::value_error(
          std::string("init[") + key + "] must have shape (" + std::to_string(rows) + ", " +
          std::to_string(cols) + ")");
    }
    return array;
  };
  const auto r = require_2d("r", p.nodes + 1, 3);
  const auto v = require_2d("v", p.nodes + 1, 3);
  const auto u = require_2d("u", p.nodes, 3);
  if (!init.contains("m")) throw py::key_error("m");
  const auto m = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(init["m"]);
  if (!m || m.ndim() != 1 || m.shape(0) != p.nodes + 1) {
    throw py::value_error("init[m] must have shape (N+1,)");
  }
  py::array_t<double, py::array::c_style | py::array::forcecast> sigma;
  if (init.contains("sigma")) {
    sigma = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(init["sigma"]);
    if (!sigma || sigma.ndim() != 1 || sigma.shape(0) != p.nodes) {
      throw py::value_error("init[sigma] must have shape (N,)");
    }
  }
  const auto rv = r.unchecked<2>();
  const auto vv = v.unchecked<2>();
  const auto uv = u.unchecked<2>();
  const auto mv = m.unchecked<1>();
  Trajectory out;
  out.x.resize(p.nodes + 1);
  out.w.resize(p.nodes);
  for (int k = 0; k <= p.nodes; ++k) {
    State physical{};
    for (int j = 0; j < 3; ++j) {
      physical[j] = rv(k, j);
      physical[3 + j] = vv(k, j);
      if (!std::isfinite(physical[j]) || !std::isfinite(physical[3 + j])) {
        throw py::value_error("init state must be finite");
      }
    }
    if (!std::isfinite(mv(k)) || !(mv(k) > 0.0)) {
      throw py::value_error("init[m] must stay finite and positive");
    }
    physical[6] = std::log(mv(k));
    out.x[k] = scale_state(p, physical);
  }
  if (sigma) {
    const auto sv = sigma.unchecked<1>();
    for (int k = 0; k < p.nodes; ++k) {
      Control physical{};
      for (int j = 0; j < 3; ++j) {
        physical[j] = mv(k) * uv(k, j);
        if (!std::isfinite(physical[j])) throw py::value_error("init control must be finite");
      }
      physical[3] = mv(k) * sv(k);
      if (!std::isfinite(physical[3])) throw py::value_error("init[sigma] must be finite");
      out.w[k] = scale_control(p, physical);
    }
  } else {
    for (int k = 0; k < p.nodes; ++k) {
      Control physical{};
      for (int j = 0; j < 3; ++j) {
        physical[j] = mv(k) * uv(k, j);
        if (!std::isfinite(physical[j])) throw py::value_error("init control must be finite");
      }
      physical[3] = norm3(physical.data());
      out.w[k] = scale_control(p, physical);
    }
  }
  return out;
}

py::array_t<double> array_2d(int rows, int cols, const std::vector<double>& values) {
  py::array_t<double> out({rows, cols});
  auto view = out.mutable_unchecked<2>();
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) view(i, j) = values[cols * i + j];
  }
  return out;
}

py::dict result_to_python(const Params& p, const Result& result) {
  py::dict out;
  out["status"] = result.status;
  out["solver_status"] = result.solver_status;
  out["solver_feasible"] = result.solver_feasible;
  out["feasible"] = result.feasible;
  out["mission_feasible"] = result.feasible;
  out["plan_physical_feasible"] =
      result.solver_feasible && result.converged && result.dynamics_defect <= p.tolerance &&
      result.path_violation <= kScaledConstraintTolerance &&
      result.slack_gap <= thrust_tolerance(p) &&
      result.thrust_violation <= thrust_tolerance(p) && result.virtual_norm <= p.tolerance;
  out["terminal_constraints_met"] =
      result.solver_feasible && result.rollout_terminal_position <= 1e-2 &&
      result.rollout_terminal_velocity_violation <= 1e-2;
  out["postcheck_failures"] = result.failures;
  out["tf"] = p.tf;
  out["N"] = p.nodes;
  out["iters"] = result.solver_attempts;
  out["accepted_iters"] = result.accepted_iterations;
  out["solver_attempts"] = result.solver_attempts;
  out["solve_time_sec"] = result.solve_time;
  out["final_trust"] = result.final_trust;
  out["nu_norm"] = result.virtual_norm;
  out["dynamics_defect"] = result.dynamics_defect;
  out["path_violation"] = result.path_violation;
  out["slack_gap"] = result.slack_gap;
  out["thrust_bound_violation"] = result.thrust_violation;
  out["thrust_bound_tolerance"] = thrust_tolerance(p);
  out["terminal_pos_slack"] = result.terminal_position;
  out["terminal_vel_slack"] = result.terminal_velocity;
  out["terminal_vel_violation"] = result.terminal_velocity_violation;
  out["rollout_terminal_pos_slack"] = result.rollout_terminal_position;
  out["rollout_terminal_vel_slack"] = result.rollout_terminal_velocity;
  out["rollout_terminal_vel_violation"] = result.rollout_terminal_velocity_violation;
  out["terminal_pos_tolerance"] = 1e-2;
  out["terminal_vel_tolerance"] = 1e-2;
  out["terminal_slack"] = false;
  out["solution_class"] = "hard_terminal";
  out["feedforward_force"] = true;
  out["scale_pos"] = p.pos_scale;
  out["scale_vel"] = p.vel_scale;
  out["scale_accel"] = p.accel_scale;

  if (!result.trajectory.x.empty()) {
    std::vector<double> r_values;
    std::vector<double> v_values;
    std::vector<double> m_values;
    std::vector<double> u_values;
    std::vector<double> sigma_values;
    std::vector<double> thrust_values;
    r_values.reserve(3 * (p.nodes + 1));
    v_values.reserve(3 * (p.nodes + 1));
    m_values.reserve(p.nodes + 1);
    u_values.reserve(3 * p.nodes);
    sigma_values.reserve(p.nodes);
    thrust_values.reserve(3 * p.nodes);
    for (const State& scaled : result.trajectory.x) {
      const State physical = unscale_state(p, scaled);
      for (int j = 0; j < 3; ++j) r_values.push_back(physical[j]);
      for (int j = 0; j < 3; ++j) v_values.push_back(physical[3 + j]);
      m_values.push_back(std::exp(physical[6]));
    }
    for (int k = 0; k < p.nodes; ++k) {
      const Control physical = unscale_control(p, result.trajectory.w[k]);
      const double mass = m_values[k];
      for (int j = 0; j < 3; ++j) {
        u_values.push_back(physical[j] / mass);
        thrust_values.push_back(physical[j]);
      }
      sigma_values.push_back(physical[3] / mass);
    }
    out["r"] = array_2d(p.nodes + 1, 3, r_values);
    out["v"] = array_2d(p.nodes + 1, 3, v_values);
    out["m"] = py::array_t<double>(m_values.size(), m_values.data());
    out["u"] = array_2d(p.nodes, 3, u_values);
    out["sigma"] = py::array_t<double>(sigma_values.size(), sigma_values.data());
    out["thrust"] = array_2d(p.nodes, 3, thrust_values);
    out["fuel"] = std::exp(p.x0[6]) - m_values.back();

    std::vector<double> nu_values;
    nu_values.reserve(NX * p.nodes);
    for (const State& value : result.virtual_control) {
      for (double component : value) nu_values.push_back(component);
    }
    out["nu"] = array_2d(p.nodes, NX, nu_values);
  }

  py::list history;
  py::list attempts;
  for (const IterationRecord& item : result.history) {
    py::dict record;
    record["iter"] = item.accepted_iteration;
    record["attempt"] = item.attempt;
    record["accepted"] = item.accepted;
    record["trust"] = item.trust;
    record["merit_ratio"] = item.ratio;
    record["actual_reduction"] = item.actual_reduction;
    record["predicted_reduction"] = item.predicted_reduction;
    record["step"] = item.scaled_step;
    record["nu_norm"] = item.virtual_norm;
    record["dynamics_defect"] = item.dynamics_defect;
    record["path_violation"] = item.path_violation;
    history.append(record);
    py::dict attempt = record.attr("copy")();
    attempt["kind"] = "scvx_native_subproblem";
    attempt["wall_time_sec"] = item.wall_time;
    attempt["solve_time_sec"] = item.backend_time;
    attempt["iters"] = item.solver_iterations;
    attempt["status"] = qoco_status_name(item.solver_status);
    attempt["solver_feasible"] = true;
    attempt["feasible"] = true;
    attempt["tf"] = p.tf;
    attempt["N"] = p.nodes;
    attempts.append(attempt);
  }
  out["history"] = history;
  out["attempt_log"] = attempts;
  return out;
}

py::dict solve_scvx_native(const py::dict& problem, const py::dict& init) {
  const Params params = parse_params(problem);
  Trajectory reference = parse_init(params, init);
  Result result;
  {
    py::gil_scoped_release release;
    result = solve_native(params, std::move(reference));
  }
  return result_to_python(params, result);
}

}  // namespace

void bind_native_scvx(py::module_& module) {
  module.def(
      "solve_scvx", &solve_scvx_native, py::arg("problem"), py::arg("init"),
      "Solve one fixed-time 3-DOF SCvx problem with native RK4 dynamics and QOCO.");
}

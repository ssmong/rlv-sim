#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

using Vec = std::vector<double>;
using Vec2 = std::array<double, 2>;
using Vec3 = std::array<double, 3>;
using Vec4 = std::array<double, 4>;
using Mat3 = std::array<double, 9>;

constexpr double kG0 = 9.80665;
constexpr double kMassEps = 1e-3;

double scalar(py::handle value) { return py::cast<double>(value); }

using DoubleArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

DoubleArray as_double_array(py::handle value) {
    auto result = DoubleArray::ensure(value);
    if (!result) throw py::type_error("expected an array-like value convertible to float64");
    return result;
}

Vec vector(py::handle value) {
    const DoubleArray a = as_double_array(value);
    const auto n = static_cast<std::size_t>(a.size());
    const auto* p = static_cast<const double*>(a.data());
    return Vec(p, p + n);
}

template <std::size_t N>
std::array<double, N> fixed_vector(py::handle value, const char* name) {
    const Vec v = vector(value);
    if (v.size() != N) {
        throw py::value_error(std::string(name) + " must contain " + std::to_string(N) + " values");
    }
    std::array<double, N> out{};
    std::copy(v.begin(), v.end(), out.begin());
    return out;
}

py::array_t<double> array(const Vec& v) {
    py::array_t<double> out(v.size());
    std::copy(v.begin(), v.end(), out.mutable_data());
    return out;
}

template <std::size_t N>
py::array_t<double> array(const std::array<double, N>& v) {
    py::array_t<double> out(N);
    std::copy(v.begin(), v.end(), out.mutable_data());
    return out;
}

py::array_t<double> matrix_array(const Vec& v, std::size_t rows, std::size_t cols) {
    py::array_t<double> out({static_cast<py::ssize_t>(rows), static_cast<py::ssize_t>(cols)});
    std::copy(v.begin(), v.end(), out.mutable_data());
    return out;
}

Vec3 add(const Vec3& a, const Vec3& b) {
    return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}
Vec3 sub(const Vec3& a, const Vec3& b) {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}
Vec3 mul(double s, const Vec3& v) { return {s * v[0], s * v[1], s * v[2]}; }
Vec3 div(const Vec3& v, double s) { return {v[0] / s, v[1] / s, v[2] / s}; }
double dot(const Vec3& a, const Vec3& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
double norm(const Vec3& v) { return std::sqrt(dot(v, v)); }
Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]};
}
Vec3 safe_unit(const Vec3& v) {
    const double n = norm(v);
    return n > 1e-8 ? div(v, n) : Vec3{0.0, 0.0, 0.0};
}
Vec3 matvec(const Mat3& m, const Vec3& v) {
    return {m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
            m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
            m[6] * v[0] + m[7] * v[1] + m[8] * v[2]};
}
Vec3 transpose_matvec(const Mat3& m, const Vec3& v) {
    return {m[0] * v[0] + m[3] * v[1] + m[6] * v[2],
            m[1] * v[0] + m[4] * v[1] + m[7] * v[2],
            m[2] * v[0] + m[5] * v[1] + m[8] * v[2]};
}

Mat3 matmul(const Mat3& a, const Mat3& b) {
    Mat3 out{};
    for (int i=0;i<3;++i) for(int j=0;j<3;++j) for(int k=0;k<3;++k)
        out[3*i+j] += a[3*i+k] * b[3*k+j];
    return out;
}
Mat3 transpose(const Mat3& a) {
    return {a[0],a[3],a[6],a[1],a[4],a[7],a[2],a[5],a[8]};
}
Mat3 matadd(const Mat3& a,const Mat3& b,double b_scale=1.0){Mat3 o{};for(int i=0;i<9;++i)o[i]=a[i]+b_scale*b[i];return o;}
Mat3 matscale(const Mat3&a,double s){Mat3 o{};for(int i=0;i<9;++i)o[i]=s*a[i];return o;}
Mat3 identity3(){return {1,0,0,0,1,0,0,0,1};}

Mat3 expm3(const Mat3& input) {
    double row_norm=0.0;for(int i=0;i<3;++i){double s=0;for(int j=0;j<3;++j)s+=std::abs(input[3*i+j]);row_norm=std::max(row_norm,s);}
    int scaling=0;if(row_norm>0.5)scaling=std::max(0,static_cast<int>(std::ceil(std::log2(row_norm/0.5))));
    const Mat3 a=matscale(input,std::ldexp(1.0,-scaling));
    Mat3 result=identity3(),term=identity3();
    for(int k=1;k<=24;++k){term=matscale(matmul(term,a),1.0/static_cast<double>(k));result=matadd(result,term);}
    for(int i=0;i<scaling;++i)result=matmul(result,result);
    return result;
}

Mat3 symmetric_sqrt(Mat3 a) {
    Mat3 eigenvectors=identity3();
    for(int iteration=0;iteration<32;++iteration){
        int p=0,q=1;double largest=std::abs(a[1]);
        for(auto pair:std::array<std::pair<int,int>,3>{{{0,1},{0,2},{1,2}}}){
            const double value=std::abs(a[3*pair.first+pair.second]);if(value>largest){largest=value;p=pair.first;q=pair.second;}}
        if(largest<1e-14)break;
        const double angle=0.5*std::atan2(2.0*a[3*p+q],a[3*q+q]-a[3*p+p]);
        const double c=std::cos(angle),s=std::sin(angle);
        Mat3 rotation=identity3();rotation[3*p+p]=c;rotation[3*q+q]=c;rotation[3*p+q]=s;rotation[3*q+p]=-s;
        a=matmul(transpose(rotation),matmul(a,rotation));eigenvectors=matmul(eigenvectors,rotation);
    }
    Mat3 diagonal{};for(int i=0;i<3;++i)diagonal[3*i+i]=std::sqrt(std::max(a[3*i+i],0.0));
    return matmul(eigenvectors,matmul(diagonal,transpose(eigenvectors)));
}

Mat3 dcm(const Vec4& q) {
    const auto [w, x, y, z] = q;
    return {1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z),
            2.0 * (x * z + w * y), 2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x),
            2.0 * (x * z - w * y), 2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y)};
}

Vec4 qdot(const Vec4& q, const Vec3& w) {
    return {-0.5 * (q[1] * w[0] + q[2] * w[1] + q[3] * w[2]),
            0.5 * (q[0] * w[0] + q[2] * w[2] - q[3] * w[1]),
            0.5 * (q[0] * w[1] - q[1] * w[2] + q[3] * w[0]),
            0.5 * (q[0] * w[2] + q[1] * w[1] - q[2] * w[0])};
}

Vec3 solve3(Mat3 a, Vec3 b) {
    for (int i = 0; i < 3; ++i) {
        int pivot = i;
        for (int r = i + 1; r < 3; ++r) {
            if (std::abs(a[3 * r + i]) > std::abs(a[3 * pivot + i])) pivot = r;
        }
        if (std::abs(a[3 * pivot + i]) < 1e-18) throw std::runtime_error("singular inertia tensor");
        if (pivot != i) {
            for (int c = i; c < 3; ++c) std::swap(a[3 * i + c], a[3 * pivot + c]);
            std::swap(b[i], b[pivot]);
        }
        const double d = a[3 * i + i];
        for (int c = i; c < 3; ++c) a[3 * i + c] /= d;
        b[i] /= d;
        for (int r = 0; r < 3; ++r) {
            if (r == i) continue;
            const double f = a[3 * r + i];
            for (int c = i; c < 3; ++c) a[3 * r + c] -= f * a[3 * i + c];
            b[r] -= f * b[i];
        }
    }
    return b;
}

double sigmoid(double x) {
    if (x >= 0.0) return 1.0 / (1.0 + std::exp(-x));
    const double e = std::exp(x);
    return e / (1.0 + e);
}

struct MassParams {
    double dry{}, wet{}, x_dry{}, x_wet{};
    Vec3 i_dry{}, i_wet{};
    Vec2 cg_lateral{};
    Vec3 products{};
};
struct Geometry {
    double s_ref{}, d_ref{}, length{}, x_eng{};
    Vec3 fwd{};
};
struct Atmosphere {
    double rho0{}, scale_height{}, a_sound{}, g0{}, delta_t{}, density_scale{};
};
struct Wind {
    Vec3 mean0{}, ou_sigma{}, turb_l{}, turb_sigma{};
    double ou_tau{};
};
struct EarthRotation {
    bool enabled{};
    Vec3 omega_ned{};
    double differential_centrifugal{};
};
struct Aero {
    Vec cos_grid, mach_grid, cd, cl, xcp, cd_slope, cl_slope, xcp_slope, cm, cm_slope;
    std::size_t nc{}, nm{};
    double x_cg_ref{}, moment_uses_cm{};
    std::size_t n_fins{};
    double fin_area{}, fin_cla{}, fin_cd0{}, fin_stall{}, fin_transonic{}, fin_incremental{};
    Vec fin_arm, fin_normal, fin_mach_grid, fin_cla_table, fin_cd2_table;
    double cla{}, cda{};
};
struct Slosh {
    double mass{}, length{}, omega{}, zeta{}, x_attach{};
};
struct Actuator {
    double p_sea{}, tvc_omega{}, tvc_zeta{}, tvc_rate_max{}, tvc_deflection_max{};
    double throttle_tau{}, throttle_rate_max{};
    double fin_omega{}, fin_zeta{}, fin_rate_max{}, fin_deflection_max{};
    double rcs_thrust{}, rcs_arm{}, rcs_isp{}, rcs_fuel0{};
    Vec fin_lock;
};
struct Engines {
    std::size_t n{}, ng{};
    Vec positions, thrust_vac, throttle_min, isp, exit_area, cant, engine_on, thrust_scale, tvc_bias;
    std::vector<int> gimbal_idx;
};

struct State {
    double t{}, m{}, rcs_fuel{};
    Vec3 r{}, v{}, w{}, wind_mean{}, wind_gust{};
    Vec4 q{};
    Vec tvc, tvc_rate, throttle, fin, fin_rate, slosh, wind_filter;
    std::uint64_t rng{};
    bool done{};
};
struct Command {
    Vec throttle, gimbal, fin;
    Vec3 rcs{};
};
struct Derivative {
    double t{1.0}, m{}, rcs_fuel{};
    Vec3 r{}, v{}, w{};
    Vec4 q{};
    Vec tvc, tvc_rate, throttle, fin, fin_rate, slosh;
};
struct Aux {
    double altitude{}, rho{}, mach{}, thrust_mag{}, x_cg{}, mdot{};
    double alpha{}, q_dyn{}, cd{}, cl{}, x_cp{}, static_margin{};
    double fin_force_mag{}, fin_transonic{};
    Vec throttle_eff;
    Vec3 f_aero{}, f_fin{}, thrust{}, moment{}, a_coriolis{}, a_centrifugal{}, v_air_body{};
};
struct RhsResult { Derivative d; Aux aux; };

State read_state(py::handle value) {
    State s;
    s.t = scalar(value.attr("t")); s.r = fixed_vector<3>(value.attr("r_ned"), "r_ned");
    s.v = fixed_vector<3>(value.attr("v_ned"), "v_ned");
    s.q = fixed_vector<4>(value.attr("q_bn"), "q_bn");
    s.w = fixed_vector<3>(value.attr("w_body"), "w_body"); s.m = scalar(value.attr("m"));
    s.tvc = vector(value.attr("tvc")); s.tvc_rate = vector(value.attr("tvc_rate"));
    s.throttle = vector(value.attr("throttle_act")); s.fin = vector(value.attr("fin"));
    s.fin_rate = vector(value.attr("fin_rate")); s.rcs_fuel = scalar(value.attr("rcs_fuel"));
    s.slosh = vector(value.attr("slosh"));
    s.wind_mean = fixed_vector<3>(value.attr("wind_mean"), "wind_mean");
    s.wind_gust = fixed_vector<3>(value.attr("wind_gust"), "wind_gust");
    s.wind_filter = vector(value.attr("wind_filter"));
    const Vec key = vector(value.attr("key"));
    if (key.size() == 2) {
        s.rng = (static_cast<std::uint64_t>(key[0]) << 32U) | static_cast<std::uint64_t>(key[1]);
    } else if (key.size() == 1) {
        s.rng = static_cast<std::uint64_t>(key[0]);
    } else {
        throw py::value_error("key must contain one uint64 or two uint32 values");
    }
    s.rng += 0x9e3779b97f4a7c15ULL;
    s.done = py::cast<bool>(value.attr("done"));
    return s;
}

Command read_command(py::handle value) {
    Command c;
    c.throttle = vector(value.attr("throttle")); c.gimbal = vector(value.attr("gimbal"));
    c.fin = vector(value.attr("fin")); c.rcs = fixed_vector<3>(value.attr("rcs"), "rcs");
    return c;
}

class NativePlant {
  public:
    NativePlant(py::object params, py::dict meta) { load(std::move(params), std::move(meta)); }

    void reset(py::object state_value) {
        state_ = read_state(state_value);
        validate_state(state_);
        initialized_ = true;
    }

    std::unique_ptr<NativePlant> create_session(py::object state_value) const {
        auto session = std::make_unique<NativePlant>(*this);
        session->reset(std::move(state_value));
        return session;
    }

    py::tuple advance(py::object command_value, int n_substeps) {
        if (!initialized_) throw std::runtime_error("NativePlant.reset(state) must be called before advance");
        if (n_substeps <= 0) throw py::value_error("n_substeps must be positive");
        const Command command = read_command(command_value);
        validate_command(command);
        Aux last;
        {
            py::gil_scoped_release release;
            for (int i=0;i<n_substeps;++i) {
                const Vec3 wind = update_wind(state_);
                last = integrate_once(state_, command, wind);
                if (state_.done) break;
            }
        }
        return py::make_tuple(public_snapshot(state_), aux_dict(last));
    }

    py::dict snapshot(bool full=false) const {
        if (!initialized_) throw std::runtime_error("NativePlant has no state; call reset first");
        return full ? state_dict(state_) : public_snapshot(state_);
    }

    py::tuple rhs(py::object state_value, py::object command_value,
                  const py::array_t<double, py::array::c_style | py::array::forcecast>& wind_value) const {
        const State state = read_state(state_value);
        const Command command = read_command(command_value);
        validate_state(state);
        validate_command(command);
        const Vec3 wind = fixed_vector<3>(wind_value, "v_wind");
        const RhsResult result = evaluate(state, command, wind);
        return py::make_tuple(derivative_tuple(result.d), aux_dict(result.aux));
    }

    py::tuple step(py::object state_value, py::object command_value,
                   const py::array_t<double, py::array::c_style | py::array::forcecast>& wind_value) const {
        const State initial = read_state(state_value);
        const Command command = read_command(command_value);
        validate_state(initial);
        validate_command(command);
        const Vec3 wind = fixed_vector<3>(wind_value, "v_wind");
        State next = initial;
        const Aux base = integrate_once(next, command, wind);
        return py::make_tuple(state_dict(next), aux_dict(base));
    }

    py::dict environment_diagnostics(double altitude) const {
        const auto [rho, sound] = atmosphere(altitude);
        py::dict out;
        out["rho"] = rho;
        out["a_sound"] = sound;
        out["pressure"] = pressure(altitude);
        return out;
    }

    py::dict mass_properties_diagnostics(double mass) const {
        const auto [xcg, inertia] = mass_props(mass);
        py::dict out;
        out["x_cg"] = xcg;
        out["inertia"] = array(inertia);
        return out;
    }

    py::dict engine_wrench_diagnostics(py::handle throttle_value, py::handle tvc_value,
                                       double mass, double altitude) const {
        const Vec throttle = vector(throttle_value);
        const Vec tvc = vector(tvc_value);
        if (throttle.size() != engines_.n) throw py::value_error("throttle count mismatch");
        if (tvc.size() != 2 * engines_.ng) throw py::value_error("gimbal count mismatch");
        const auto [xcg, inertia] = mass_props(mass);
        (void)inertia;
        const auto result = engines(throttle, tvc, xcg, pressure(altitude));
        py::dict out;
        out["force"] = array(result.force);
        out["moment"] = array(result.moment);
        out["mdot"] = result.mdot;
        return out;
    }

    py::dict aero_wrench_diagnostics(py::handle velocity_value, py::handle rates_value,
                                     double mass, double altitude) const {
        const Vec3 velocity = fixed_vector<3>(velocity_value, "velocity_body");
        const Vec3 rates = fixed_vector<3>(rates_value, "rates_body");
        const auto [xcg, inertia] = mass_props(mass);
        (void)inertia;
        const auto [rho, sound] = atmosphere(altitude);
        const auto result = body_aero(velocity, rho, sound, xcg, rates);
        py::dict out;
        out["force"] = array(result.force);
        out["moment"] = array(result.moment);
        out["alpha"] = result.alpha;
        out["mach"] = result.mach;
        out["q_dyn"] = result.qdyn;
        out["cd"] = result.cd;
        out["cl"] = result.cl;
        out["x_cp"] = result.xcp;
        return out;
    }

    py::dict fin_wrench_diagnostics(py::handle velocity_value, py::handle rates_value,
                                    py::handle fin_value, double mass, double altitude) const {
        const Vec3 velocity = fixed_vector<3>(velocity_value, "velocity_body");
        const Vec3 rates = fixed_vector<3>(rates_value, "rates_body");
        const Vec fin = vector(fin_value);
        if (fin.size() != aero_.n_fins) throw py::value_error("fin count mismatch");
        const auto [xcg, inertia] = mass_props(mass);
        (void)inertia;
        const auto [rho, sound] = atmosphere(altitude);
        const auto result = fin_forces(velocity, rates, rho, sound, fin, xcg);
        py::dict out;
        out["force"] = array(result.force);
        out["moment"] = array(result.moment);
        out["transonic"] = result.transonic;
        return out;
    }

    py::dict slosh_diagnostics(py::handle slosh_value, py::handle acceleration_value,
                               double mass) const {
        State state;
        state.slosh = vector(slosh_value);
        if (state.slosh.size() != 4) throw py::value_error("slosh state must contain 4 values");
        const Vec3 acceleration = fixed_vector<3>(acceleration_value, "acceleration_body");
        const auto [xcg, inertia] = mass_props(mass);
        (void)inertia;
        const auto result = slosh(state, acceleration, xcg);
        py::dict out;
        out["derivative"] = array(result.derivative);
        out["force"] = array(result.force);
        out["moment"] = array(result.moment);
        return out;
    }

    py::dict wind_diagnostics(double airspeed, int count, std::uint64_t seed) const {
        if (!(airspeed >= 0.0) || !std::isfinite(airspeed)) {
            throw py::value_error("airspeed must be finite and non-negative");
        }
        if (count <= 0) throw py::value_error("count must be positive");
        State state;
        state.wind_mean = wind_model_ == "steady" ? wind_.mean0 : Vec3{};
        state.wind_gust = {};
        state.wind_filter.assign(9, 0.0);
        state.rng = seed;
        Vec means(static_cast<std::size_t>(count) * 3);
        Vec gusts(static_cast<std::size_t>(count) * 3);
        {
            py::gil_scoped_release release;
            for (int sample = 0; sample < count; ++sample) {
                update_wind_at_speed(state, std::max(airspeed, 1e-3));
                for (int axis = 0; axis < 3; ++axis) {
                    const std::size_t index = static_cast<std::size_t>(3 * sample + axis);
                    means[index] = state.wind_mean[axis];
                    gusts[index] = state.wind_gust[axis];
                }
            }
        }
        py::dict out;
        out["mean"] = matrix_array(means, static_cast<std::size_t>(count), 3);
        out["gust"] = matrix_array(gusts, static_cast<std::size_t>(count), 3);
        return out;
    }

  private:
    Geometry geometry_;
    MassParams mass_;
    Atmosphere atmosphere_;
    Wind wind_;
    EarthRotation earth_;
    Aero aero_;
    Slosh slosh_;
    Actuator actuator_;
    Engines engines_;
    Vec3 gravity_{};
    double dt_{};
    std::string aero_model_, atmosphere_model_, mass_model_, slosh_model_, actuator_model_, thrust_model_, wind_model_;
    State state_;
    bool initialized_{false};

    void load(py::object params, py::dict meta) {
        auto g = params.attr("geom");
        geometry_ = {scalar(g.attr("s_ref")), scalar(g.attr("d_ref")), scalar(g.attr("length")),
                     scalar(g.attr("x_eng")), fixed_vector<3>(g.attr("fwd_axis"), "fwd_axis")};
        auto mp = params.attr("mass");
        mass_ = {scalar(mp.attr("m_dry")), scalar(mp.attr("m_wet")), scalar(mp.attr("x_cg_dry")),
                 scalar(mp.attr("x_cg_wet")), fixed_vector<3>(mp.attr("I_dry"), "I_dry"),
                 fixed_vector<3>(mp.attr("I_wet"), "I_wet"),
                 fixed_vector<2>(mp.attr("cg_lateral"), "cg_lateral"),
                 fixed_vector<3>(mp.attr("inertia_products"), "inertia_products")};
        auto at = params.attr("atmosphere");
        atmosphere_ = {scalar(at.attr("rho0")), scalar(at.attr("scale_height")),
                       scalar(at.attr("a_sound")), scalar(at.attr("g0")), scalar(at.attr("delta_T")),
                       scalar(at.attr("density_scale"))};
        auto wp = params.attr("wind");
        wind_ = {fixed_vector<3>(wp.attr("mean0"), "wind.mean0"),
                 fixed_vector<3>(wp.attr("ou_sigma"), "wind.ou_sigma"),
                 fixed_vector<3>(wp.attr("turb_L"), "wind.turb_L"),
                 fixed_vector<3>(wp.attr("turb_sigma"), "wind.turb_sigma"),
                 scalar(wp.attr("ou_tau"))};
        auto er = params.attr("earth");
        earth_ = {scalar(er.attr("enabled")) != 0.0,
                  fixed_vector<3>(er.attr("omega_ned"), "earth.omega_ned"),
                  scalar(er.attr("differential_centrifugal"))};
        auto ap = params.attr("aero");
        aero_.cos_grid = vector(ap.attr("cos_grid")); aero_.mach_grid = vector(ap.attr("mach_grid"));
        aero_.nc = aero_.cos_grid.size(); aero_.nm = aero_.mach_grid.size();
        aero_.cd = vector(ap.attr("cd_table")); aero_.cl = vector(ap.attr("cl_table"));
        aero_.xcp = vector(ap.attr("xcp_table")); aero_.cd_slope = vector(ap.attr("cd_slope"));
        aero_.cl_slope = vector(ap.attr("cl_slope")); aero_.xcp_slope = vector(ap.attr("xcp_slope"));
        aero_.cm = vector(ap.attr("cm_table")); aero_.cm_slope = vector(ap.attr("cm_slope"));
        aero_.x_cg_ref = scalar(ap.attr("x_cg_ref"));
        aero_.moment_uses_cm = scalar(ap.attr("moment_uses_cm"));
        aero_.fin_arm = vector(ap.attr("fin_arm")); aero_.n_fins = aero_.fin_arm.size() / 3;
        aero_.fin_normal = vector(ap.attr("fin_normal")); aero_.fin_area = scalar(ap.attr("fin_area"));
        aero_.fin_cla = scalar(ap.attr("fin_cla")); aero_.fin_cd0 = scalar(ap.attr("fin_cd0"));
        aero_.fin_stall = scalar(ap.attr("fin_stall_alpha"));
        aero_.fin_transonic = scalar(ap.attr("fin_transonic_mach"));
        aero_.fin_incremental = scalar(ap.attr("fin_incremental"));
        aero_.fin_mach_grid = vector(ap.attr("fin_mach_grid"));
        aero_.fin_cla_table = vector(ap.attr("fin_cla_table"));
        aero_.fin_cd2_table = vector(ap.attr("fin_cd_delta2_table"));
        aero_.cla = scalar(ap.attr("cla")); aero_.cda = scalar(ap.attr("cda"));
        auto sp = params.attr("slosh");
        slosh_ = {scalar(sp.attr("slosh_mass")), scalar(sp.attr("pendulum_length")),
                  scalar(sp.attr("omega")), scalar(sp.attr("zeta")), scalar(sp.attr("x_attach"))};
        auto ac = params.attr("actuator");
        actuator_ = {scalar(ac.attr("p_sea")), scalar(ac.attr("tvc_omega")), scalar(ac.attr("tvc_zeta")),
                     scalar(ac.attr("tvc_rate_max_rad_s")), scalar(ac.attr("tvc_deflection_max_rad")),
                     scalar(ac.attr("throttle_tau")), scalar(ac.attr("throttle_rate_max")),
                     scalar(ac.attr("fin_omega")), scalar(ac.attr("fin_zeta")),
                     scalar(ac.attr("fin_rate_max_rad_s")), scalar(ac.attr("fin_deflection_max_rad")),
                     scalar(ac.attr("rcs_thrust")), scalar(ac.attr("rcs_arm")), scalar(ac.attr("rcs_isp")),
                     scalar(ac.attr("rcs_fuel0")), vector(ac.attr("fin_lock"))};
        auto ep = params.attr("engines");
        engines_.positions = vector(ep.attr("positions")); engines_.n = engines_.positions.size() / 3;
        engines_.thrust_vac = vector(ep.attr("thrust_vac"));
        engines_.throttle_min = vector(ep.attr("throttle_min")); engines_.isp = vector(ep.attr("isp"));
        engines_.exit_area = vector(ep.attr("nozzle_exit_area")); engines_.cant = vector(ep.attr("cant"));
        engines_.engine_on = vector(ep.attr("engine_on")); engines_.thrust_scale = vector(ep.attr("thrust_scale"));
        engines_.tvc_bias = vector(ep.attr("tvc_bias")); engines_.ng = engines_.tvc_bias.size() / 2;
        for (double index : vector(meta["gimbal_idx"])) {
            engines_.gimbal_idx.push_back(static_cast<int>(index));
        }
        gravity_ = fixed_vector<3>(params.attr("g_ned"), "g_ned"); dt_ = scalar(params.attr("plant_dt"));
        aero_model_ = py::cast<std::string>(meta["aero_model"]);
        atmosphere_model_ = py::cast<std::string>(meta["atmosphere_model"]);
        mass_model_ = py::cast<std::string>(meta["massprops_model"]);
        slosh_model_ = py::cast<std::string>(meta["slosh_model"]);
        actuator_model_ = py::cast<std::string>(meta["actuator_model"]);
        thrust_model_ = py::cast<std::string>(meta["thrust_model"]);
        wind_model_ = py::cast<std::string>(meta["wind_model"]);
        if (aero_.nc < 2 || aero_.nm < 2) throw py::value_error("aero grids need at least two nodes");
        const std::size_t table_size = aero_.nc * aero_.nm;
        for (const auto* table : {&aero_.cd, &aero_.cl, &aero_.xcp, &aero_.cd_slope,
                                  &aero_.cl_slope, &aero_.xcp_slope, &aero_.cm, &aero_.cm_slope}) {
            if (table->size() != table_size) throw py::value_error("aero table shape does not match grids");
        }
        if (engines_.gimbal_idx.size() != engines_.ng) throw py::value_error("gimbal index count mismatch");
        for (int index : engines_.gimbal_idx) {
            if (index < 0 || static_cast<std::size_t>(index) >= engines_.n) throw py::value_error("gimbal index out of range");
        }
    }

    void validate_state(const State& state) const {
        if (state.tvc.size()!=2*engines_.ng || state.tvc_rate.size()!=2*engines_.ng)
            throw py::value_error("state TVC arrays do not match configured gimballed-engine count");
        if (state.throttle.size()!=engines_.n) throw py::value_error("state throttle count does not match engines");
        if (state.fin.size()!=aero_.n_fins || state.fin_rate.size()!=aero_.n_fins)
            throw py::value_error("state fin arrays do not match configured fin count");
        if (state.slosh.size()!=4 || state.wind_filter.size()!=9) throw py::value_error("invalid internal state shape");
    }
    void validate_command(const Command& command) const {
        if (command.throttle.size()!=engines_.n) throw py::value_error("command throttle count does not match engines");
        if (command.gimbal.size()!=2*engines_.ng) throw py::value_error("command gimbal shape does not match gimballed engines");
        if (command.fin.size()!=aero_.n_fins) throw py::value_error("command fin count does not match fins");
    }

    std::uint64_t random_u64(State& state) const {
        state.rng += 0x9e3779b97f4a7c15ULL;
        std::uint64_t z=state.rng;z=(z^(z>>30U))*0xbf58476d1ce4e5b9ULL;
        z=(z^(z>>27U))*0x94d049bb133111ebULL;return z^(z>>31U);
    }
    double normal(State& state) const {
        const double u1=(static_cast<double>((random_u64(state)>>11U)+1U))/(9007199254740992.0+1.0);
        const double u2=static_cast<double>(random_u64(state)>>11U)/9007199254740992.0;
        return std::sqrt(-2.0*std::log(u1))*std::cos(2.0*3.14159265358979323846*u2);
    }

    Vec3 update_wind(State& state) const {
        if(wind_model_=="none")return {0,0,0};
        if(wind_model_=="steady")return state.wind_mean;
        const Vec3 previous=add(state.wind_mean,state.wind_gust);
        const double speed=std::max(norm(sub(state.v,previous)),1e-3);
        return update_wind_at_speed(state, speed);
    }

    Vec3 update_wind_at_speed(State& state, double speed) const {
        if(wind_model_=="none")return {0,0,0};
        if(wind_model_=="steady")return state.wind_mean;
        const double a=std::exp(-dt_/wind_.ou_tau),gain_base=std::sqrt(std::max(1.0-a*a,0.0));
        for(int axis=0;axis<3;++axis){const double target=axis<2?wind_.mean0[axis]:0.0;
            state.wind_mean[axis]=target+a*(state.wind_mean[axis]-target)+wind_.ou_sigma[axis]*gain_base*normal(state);}
        Vec3 gust{};
        for(int axis=0;axis<3;++axis){
            Mat3 m{},sigma{};Vec3 output{};
            if(wind_model_=="dryden"){
                if(axis==0){m={-1,0,0,0,0,0,0,0,0};sigma={1,0,0,0,0,0,0,0,0};output={1,0,0};}
                else{m={0,1,0,-1,-2,0,0,0,0};sigma={.25,0,0,0,.25,0,0,0,0};output={1,std::sqrt(3.0),0};}
            }else{
                if(axis==0){m={0,1,0,-5.032712632108707,-6.829391041771515,0,0,0,0};
                    sigma={0.03003450154287904,0,0,0,0.15115501531393558,0,0,0,0};output={5.032712632108707,1.2581781580271767,0};}
                else{m={0,1,0,0,0,1,-6.49772579597141,-19.46588693957115,-12.835607537361923};
                    sigma={0.004217463940686654,0,-0.0021349923765752,0,0.0021349923765752,0,-0.0021349923765752,0,0.0415595202192592};
                    output={6.49772579597141,17.85445094217024,2.207927225471085};}
            }
            sigma=matscale(sigma,wind_.turb_sigma[axis]*wind_.turb_sigma[axis]);
            const double kappa=dt_*speed/wind_.turb_l[axis];const Mat3 transition=expm3(matscale(m,kappa));
            Mat3 covariance=matadd(sigma,matmul(transition,matmul(sigma,transpose(transition))),-1.0);
            const Mat3 covariance_raw=covariance;
            for(int i=0;i<3;++i)for(int j=0;j<3;++j)covariance[3*i+j]=0.5*(covariance_raw[3*i+j]+covariance_raw[3*j+i]);
            const Mat3 root=symmetric_sqrt(covariance);const Vec3 noise{normal(state),normal(state),normal(state)};
            const Vec3 old{state.wind_filter[3*axis],state.wind_filter[3*axis+1],state.wind_filter[3*axis+2]};
            const Vec3 next=add(matvec(transition,old),matvec(root,noise));
            for(int i=0;i<3;++i)state.wind_filter[3*axis+i]=next[i];gust[axis]=dot(output,next);
        }
        state.wind_gust=gust;return add(state.wind_mean,gust);
    }

    std::pair<double, Vec3> mass_props(double m) const {
        if (mass_model_ == "prescribed") {
            const double f = std::clamp((m - mass_.dry) / (mass_.wet - mass_.dry), 0.0, 1.0);
            Vec3 inertia{};
            for (int i = 0; i < 3; ++i) inertia[i] = mass_.i_dry[i] + f * (mass_.i_wet[i] - mass_.i_dry[i]);
            return {mass_.x_dry + f * (mass_.x_wet - mass_.x_dry), inertia};
        }
        constexpr double ox = 0.70, sep = 9.0;
        const double prop_wet = mass_.wet - mass_.dry;
        const double prop = std::clamp(m - mass_.dry, 0.0, prop_wet), f = prop / prop_wet;
        const double mo = ox * prop, mf = (1.0 - ox) * prop;
        const double mu = (mass_.wet * mass_.x_wet - mass_.dry * mass_.x_dry) / prop_wet;
        const double xo = mu + (1.0 - ox) * sep, xf = mu - ox * sep;
        const double xcg = (mass_.dry * mass_.x_dry + mo * xo + mf * xf) / std::max(m, 1e-6);
        const double r2 = 2.0 * (mass_.i_wet[0] - mass_.i_dry[0]) / prop_wet;
        const double ixx = mass_.i_dry[0] + 0.5 * prop * r2;
        auto transverse = [&](double propellant, double total, double cg) {
            return mass_.i_dry[1] + mass_.dry * std::pow(mass_.x_dry - cg, 2) +
                   ox * propellant * std::pow(xo - cg, 2) +
                   (1.0 - ox) * propellant * std::pow(xf - cg, 2);
        };
        const double now = transverse(prop, m, xcg);
        const double wet = transverse(prop_wet, mass_.wet, mass_.x_wet);
        const double itr = now + f * (mass_.i_wet[1] - wet);
        return {xcg, {ixx, itr, itr}};
    }

    Vec3 inertia_derivative(double m) const {
        if (mass_model_ == "prescribed") {
            // At an exact endpoint use the physically relevant inward derivative: propellant mass
            // immediately decreases from wet and cannot decrease below dry. Only truly out-of-
            // range masses are on the constant extension, so the endpoint derivative follows the
            // physically reachable inward direction instead of a symmetric clip convention.
            if (m < mass_.dry || m > mass_.wet) return {0.0, 0.0, 0.0};
            const double inverse_span = 1.0 / (mass_.wet - mass_.dry);
            return {(mass_.i_wet[0] - mass_.i_dry[0]) * inverse_span,
                    (mass_.i_wet[1] - mass_.i_dry[1]) * inverse_span,
                    (mass_.i_wet[2] - mass_.i_dry[2]) * inverse_span};
        }
        const double scale = std::max(1.0, std::abs(m));
        const double h = 1e-5 * scale;
        const Vec3 plus = mass_props(m + h).second, minus = mass_props(m - h).second;
        return {(plus[0] - minus[0]) / (2.0 * h), (plus[1] - minus[1]) / (2.0 * h),
                (plus[2] - minus[2]) / (2.0 * h)};
    }

    std::pair<double, double> atmosphere(double altitude) const {
        const double h = std::max(altitude, 0.0);
        if (atmosphere_model_ == "vacuum") return {0.0, 340.3};
        if (atmosphere_model_ == "exp") {
            const double temp = h <= 11000.0 ? 288.15 - 0.0065 * h : 216.65;
            return {atmosphere_.rho0 * std::exp(-h / atmosphere_.scale_height),
                    std::sqrt(1.4 * 287.05 * temp)};
        }
        static constexpr std::array<double, 7> hb{0, 11000, 20000, 32000, 47000, 51000, 71000};
        static constexpr std::array<double, 7> lapse{-0.0065, 0, 0.001, 0.0028, 0, -0.0028, -0.002};
        static constexpr std::array<double, 7> tb{288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65};
        static constexpr std::array<double, 7> pb{101325, 22632.06, 5474.889, 868.0187, 110.9063, 66.93887, 3.956420};
        const double hgp = 6356766.0 * h / (6356766.0 + h);
        std::size_t i = 0; while (i + 1 < hb.size() && hgp >= hb[i + 1]) ++i;
        const double dh = hgp - hb[i], temp = tb[i] + lapse[i] * dh;
        const double pressure = std::abs(lapse[i]) < 1e-12
            ? pb[i] * std::exp(-kG0 * dh / (287.05287 * tb[i]))
            : pb[i] * std::pow(std::max(temp / tb[i], 1e-6), -kG0 / (lapse[i] * 287.05287));
        const double actual = std::max(temp + atmosphere_.delta_t, 1.0);
        return {atmosphere_.density_scale * pressure / (287.05287 * actual),
                std::sqrt(1.4 * 287.05287 * actual)};
    }

    double pressure(double altitude) const {
        if (thrust_model_ == "vacuum") return 0.0;
        if (thrust_model_ == "sea_level") return actuator_.p_sea;
        const double h = std::max(altitude, 0.0);
        if (atmosphere_model_ == "vacuum") return 0.0;
        if (atmosphere_model_ == "exp") return actuator_.p_sea * std::exp(-h / atmosphere_.scale_height);
        static constexpr std::array<double, 7> hb{0, 11000, 20000, 32000, 47000, 51000, 71000};
        static constexpr std::array<double, 7> lapse{-0.0065, 0, 0.001, 0.0028, 0, -0.0028, -0.002};
        static constexpr std::array<double, 7> tb{288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65};
        static constexpr std::array<double, 7> pb{101325, 22632.06, 5474.889, 868.0187, 110.9063, 66.93887, 3.956420};
        const double hgp = 6356766.0 * h / (6356766.0 + h);
        std::size_t i = 0; while (i + 1 < hb.size() && hgp >= hb[i + 1]) ++i;
        const double dh = hgp - hb[i], temp = tb[i] + lapse[i] * dh;
        return std::abs(lapse[i]) < 1e-12
            ? pb[i] * std::exp(-kG0 * dh / (287.05287 * tb[i]))
            : pb[i] * std::pow(std::max(temp / tb[i], 1e-6), -kG0 / (lapse[i] * 287.05287));
    }

    double interp(double cosa, double mach, const Vec& table, const Vec& slopes) const {
        const auto ci = std::upper_bound(aero_.cos_grid.begin(), aero_.cos_grid.end(), cosa);
        std::size_t i = ci == aero_.cos_grid.begin() ? 0 : static_cast<std::size_t>(ci - aero_.cos_grid.begin() - 1);
        i = std::min(i, aero_.nc - 2);
        const auto mi = std::upper_bound(aero_.mach_grid.begin(), aero_.mach_grid.end(), mach);
        std::size_t j = mi == aero_.mach_grid.begin() ? 0 : static_cast<std::size_t>(mi - aero_.mach_grid.begin() - 1);
        j = std::min(j, aero_.nm - 2);
        const double h = aero_.cos_grid[i + 1] - aero_.cos_grid[i];
        const double t = std::clamp((cosa - aero_.cos_grid[i]) / h, 0.0, 1.0);
        const double t2 = t * t, t3 = t2 * t;
        auto hermite = [&](std::size_t col) {
            const auto at = [&](std::size_t row) { return row * aero_.nm + col; };
            return (2*t3-3*t2+1)*table[at(i)] + (t3-2*t2+t)*h*slopes[at(i)] +
                   (-2*t3+3*t2)*table[at(i+1)] + (t3-t2)*h*slopes[at(i+1)];
        };
        const double tm = std::clamp((mach - aero_.mach_grid[j]) /
                                     (aero_.mach_grid[j + 1] - aero_.mach_grid[j]), 0.0, 1.0);
        return hermite(j) * (1.0 - tm) + hermite(j + 1) * tm;
    }

    double interp_linear(double x, const Vec& grid, const Vec& values) const {
        const auto it = std::upper_bound(grid.begin(), grid.end(), x);
        std::size_t i = it == grid.begin() ? 0 : static_cast<std::size_t>(it - grid.begin() - 1);
        i = std::min(i, grid.size() - 2);
        const double t = std::clamp((x - grid[i]) / (grid[i + 1] - grid[i]), 0.0, 1.0);
        return values[i] * (1.0 - t) + values[i + 1] * t;
    }

    struct AeroResult { Vec3 force{}, moment{}; double alpha{}, mach{}, qdyn{}, cd{}, cl{}, xcp{}; };
    AeroResult body_aero(const Vec3& velocity, double rho, double sound, double xcg, const Vec3& rates) const {
        AeroResult out;
        const double speed = norm(velocity), safe_speed = std::max(speed, 1e-6);
        const Vec3 vhat = speed > 1e-6 ? div(velocity, safe_speed) : Vec3{0,0,0};
        const double cosa = speed > 1e-6 ? std::clamp(dot(geometry_.fwd, vhat), -1.0, 1.0) : 1.0;
        out.alpha = std::acos(cosa);
        const Vec3 lift_raw = speed > 1e-6 ? sub(geometry_.fwd, mul(cosa, vhat)) : Vec3{0,0,0};
        const Vec3 lift_dir = safe_unit(lift_raw);
        out.mach = speed / sound; out.qdyn = 0.5 * rho * speed * speed;
        if (aero_model_ == "linear") {
            out.cl = aero_.cla * out.alpha; out.cd = aero_.cda + 0.3 * out.cl * out.cl;
            out.xcp = -0.45 * geometry_.length;
        } else {
            out.cd = interp(cosa, out.mach, aero_.cd, aero_.cd_slope);
            out.cl = interp(cosa, out.mach, aero_.cl, aero_.cl_slope);
            out.xcp = interp(cosa, out.mach, aero_.xcp, aero_.xcp_slope);
        }
        out.force = add(mul(-out.qdyn * geometry_.s_ref * out.cd, vhat),
                        mul(out.qdyn * geometry_.s_ref * out.cl, lift_dir));
        const Vec3 ex_cross = cross({1,0,0}, out.force);
        if (aero_model_ != "linear" && aero_.moment_uses_cm > 0.5) {
            const double cm = interp(cosa, out.mach, aero_.cm, aero_.cm_slope);
            const Vec3 direction = safe_unit(ex_cross);
            out.moment = add(mul(-geometry_.d_ref * cm * out.qdyn * geometry_.s_ref, direction),
                             mul(aero_.x_cg_ref - xcg, ex_cross));
            const double cn = norm(ex_cross);
            out.xcp = aero_.x_cg_ref + (cn > 1e-6 ? -geometry_.d_ref * cm * out.qdyn * geometry_.s_ref / cn : 0.0);
        } else {
            out.moment = cross({out.xcp - xcg, 0, 0}, out.force);
        }
        if (aero_model_ != "linear" && speed > 1e-6) {
            const double coefficient = out.qdyn * geometry_.s_ref * geometry_.length * -0.5 *
                                       geometry_.length / (2.0 * safe_speed);
            out.moment = add(out.moment, {0.0, coefficient * rates[1], coefficient * rates[2]});
        }
        return out;
    }

    struct FinResult { Vec3 force{}, moment{}; double transonic{}; };
    FinResult fin_forces(const Vec3& velocity, const Vec3& rates, double rho, double sound,
                         const Vec& angles, double xcg) const {
        FinResult total; total.transonic = aero_.n_fins ? 0.0 : 1.0;
        auto static_increment = [&](const Vec3& flow, double delta, const Vec3& normal) {
            const double speed = norm(flow); const Vec3 vhat = speed > 1e-6 ? div(flow, speed) : Vec3{0,0,0};
            const double qdyn = 0.5 * rho * speed * speed, mach = speed / sound;
            const double cla = interp_linear(mach, aero_.fin_mach_grid, aero_.fin_cla_table);
            const double cd2 = interp_linear(mach, aero_.fin_mach_grid, aero_.fin_cd2_table);
            const double incidence = std::asin(std::clamp(dot(vhat, normal), -1.0 + 1e-7, 1.0 - 1e-7));
            const double alpha = delta - incidence;
            const Vec3 lift_dir = safe_unit(sub(normal, mul(dot(normal, vhat), vhat)));
            return add(mul(qdyn * aero_.fin_area * cla * alpha, lift_dir),
                       mul(-qdyn * aero_.fin_area * cd2 * alpha * alpha, vhat));
        };
        for (std::size_t i = 0; i < aero_.n_fins; ++i) {
            const Vec3 arm{aero_.fin_arm[3*i], aero_.fin_arm[3*i+1], aero_.fin_arm[3*i+2]};
            const Vec3 normal{aero_.fin_normal[3*i], aero_.fin_normal[3*i+1], aero_.fin_normal[3*i+2]};
            const Vec3 relative = sub(arm, {xcg, 0, 0});
            const Vec3 local = add(velocity, cross(rates, relative));
            Vec3 force{}; double transonic = 1.0;
            if (aero_.fin_incremental > 0.5) {
                force = sub(static_increment(local, angles[i], normal), static_increment(velocity, 0.0, normal));
            } else {
                const double speed = norm(local), safe = std::max(speed, 1e-6);
                const Vec3 vhat = speed > 1e-6 ? div(local, safe) : Vec3{0,0,0};
                const double qdyn = 0.5 * rho * speed * speed, mach = speed / sound;
                transonic = 1.0 + 2.0 * std::exp(-std::pow((mach - aero_.fin_transonic) / 0.15, 2));
                const double incidence = std::asin(std::clamp(dot(vhat, normal), -1.0+1e-7, 1.0-1e-7));
                const double alpha = angles[i] - incidence;
                const double sat = aero_.fin_stall * std::tanh(alpha / aero_.fin_stall);
                const Vec3 lift_dir = safe_unit(sub(normal, mul(dot(normal, vhat), vhat)));
                const double super_decay = 1.0 / std::pow(std::pow(1.0-mach*mach, 2)+0.04, 0.25);
                const double base = std::clamp(super_decay, 0.35, 1.0);
                const double bucket = sigmoid((mach-0.75)/0.08) * sigmoid((1.6-mach)/0.08);
                const double efficiency = base * (1.0 - 0.45 * bucket);
                const double lift = qdyn * aero_.fin_area * aero_.fin_cla * efficiency * sat;
                const double drag = qdyn * aero_.fin_area *
                    (aero_.fin_cd0 * transonic + aero_.fin_cla * efficiency * sat * std::sin(sat));
                force = add(mul(lift, lift_dir), mul(-drag, vhat));
            }
            total.force = add(total.force, force); total.moment = add(total.moment, cross(relative, force));
            total.transonic += transonic / static_cast<double>(aero_.n_fins);
        }
        if (aero_.fin_incremental > 0.5) total.transonic = 1.0;
        return total;
    }

    double firing(double throttle, double floor) const {
        const double x = std::clamp((throttle - (floor - 0.1)) / 0.1, 0.0, 1.0);
        return x * x * (3.0 - 2.0 * x);
    }

    struct EngineResult { Vec3 force{}, moment{}; double mdot{}; };
    EngineResult engines(const Vec& throttle, const Vec& tvc, double xcg, double ambient) const {
        Vec angles = engines_.cant;
        for (std::size_t g = 0; g < engines_.ng; ++g) {
            const int i = engines_.gimbal_idx[g];
            angles[2*i] += tvc[2*g] + engines_.tvc_bias[2*g];
            angles[2*i+1] += tvc[2*g+1] + engines_.tvc_bias[2*g+1];
        }
        EngineResult total;
        for (std::size_t i = 0; i < engines_.n; ++i) {
            const double pitch = angles[2*i], yaw = angles[2*i+1];
            const Vec3 direction{std::cos(pitch)*std::cos(yaw), std::cos(pitch)*std::sin(yaw), -std::sin(pitch)};
            const double t = std::clamp(throttle[i], 0.0, 1.0), phi = firing(t, engines_.throttle_min[i]);
            const double magnitude = engines_.thrust_scale[i] * engines_.engine_on[i] * phi *
                std::max(t * engines_.thrust_vac[i] - ambient * engines_.exit_area[i], 0.0);
            const Vec3 force = mul(magnitude, direction);
            const Vec3 arm{engines_.positions[3*i]-xcg, engines_.positions[3*i+1], engines_.positions[3*i+2]};
            total.force = add(total.force, force); total.moment = add(total.moment, cross(arm, force));
            total.mdot += engines_.engine_on[i] * phi * t * engines_.thrust_vac[i] / (engines_.isp[i] * kG0);
        }
        return total;
    }

    Vec3 quantize_rcs(const Vec3& command, bool ideal) const {
        Vec3 duty{};
        for (int i = 0; i < 3; ++i) {
            const double c = std::clamp(command[i], -1.0, 1.0);
            if (ideal) duty[i] = c;
            else {
                const double x = std::clamp((std::abs(c)-0.05)/0.05, 0.0, 1.0);
                const double a = x*x*(3.0-2.0*x);
                duty[i] = std::copysign(a*(0.15+0.85*a), c);
            }
        }
        return duty;
    }

    void actuator_inputs(const State& s, const Command& c, Vec& throttle, Vec& tvc, Vec& fin) const {
        if (actuator_model_ != "ideal") { throttle=s.throttle; tvc=s.tvc; fin=s.fin; return; }
        throttle.resize(engines_.n);
        for (std::size_t i=0;i<engines_.n;++i) {
            const double t=std::clamp(c.throttle[i],0.0,1.0); throttle[i]=t>=engines_.throttle_min[i]?t:0.0;
        }
        tvc.resize(2*engines_.ng); for(std::size_t i=0;i<tvc.size();++i)
            tvc[i]=std::clamp(c.gimbal[i],-actuator_.tvc_deflection_max,actuator_.tvc_deflection_max);
        fin.resize(aero_.n_fins); for(std::size_t i=0;i<fin.size();++i)
            fin[i]=std::clamp(c.fin[i],-actuator_.fin_deflection_max,actuator_.fin_deflection_max);
    }

    void second_order(double actual, double rate, double reference, double omega, double zeta,
                      double rate_limit, double deflection_limit, bool linear,
                      double& d_actual, double& d_rate) const {
        const double ref=std::clamp(reference,-deflection_limit,deflection_limit);
        const double kp=omega/(2.0*zeta), kv=2.0*zeta*omega;
        const double desired=linear?kp*(ref-actual):rate_limit*std::tanh(kp*(ref-actual)/rate_limit);
        d_actual=rate; d_rate=kv*(desired-rate);
    }

    Derivative actuator_derivatives(const State& s, const Command& c) const {
        Derivative d; d.t=0.0; d.tvc.assign(s.tvc.size(),0); d.tvc_rate.assign(s.tvc.size(),0);
        d.throttle.assign(s.throttle.size(),0); d.fin.assign(s.fin.size(),0); d.fin_rate.assign(s.fin.size(),0);
        d.slosh.assign(4,0);
        const bool ideal=actuator_model_=="ideal", simplified=actuator_model_=="simplified";
        if (!ideal) {
            for(std::size_t i=0;i<s.throttle.size();++i){const double cmd=std::clamp(c.throttle[i],0.0,1.0);
                const double target=cmd>=engines_.throttle_min[i]?cmd:0.0;
                d.throttle[i]=simplified?(target-s.throttle[i])/actuator_.throttle_tau:
                    actuator_.throttle_rate_max*std::tanh(((target-s.throttle[i])/actuator_.throttle_tau)/actuator_.throttle_rate_max);}
            for(std::size_t i=0;i<s.tvc.size();++i) second_order(s.tvc[i],s.tvc_rate[i],c.gimbal[i],
                actuator_.tvc_omega,actuator_.tvc_zeta,actuator_.tvc_rate_max,actuator_.tvc_deflection_max,
                simplified,d.tvc[i],d.tvc_rate[i]);
            for(std::size_t i=0;i<s.fin.size();++i){second_order(s.fin[i],s.fin_rate[i],c.fin[i],
                actuator_.fin_omega,actuator_.fin_zeta,actuator_.fin_rate_max,actuator_.fin_deflection_max,
                simplified,d.fin[i],d.fin_rate[i]); if(!simplified){d.fin[i]*=(1.0-actuator_.fin_lock[i]);d.fin_rate[i]*=(1.0-actuator_.fin_lock[i]);}}
        }
        const Vec3 duty=quantize_rcs(c.rcs, ideal||simplified);
        const double available=s.rcs_fuel>0.0?1.0:0.0;
        d.rcs_fuel=-available*(std::abs(duty[0])+std::abs(duty[1])+std::abs(duty[2]))*
                       actuator_.rcs_thrust/(actuator_.rcs_isp*kG0);
        return d;
    }

    struct SloshResult { Vec derivative; Vec3 force{}, moment{}; };
    SloshResult slosh(const State& s, const Vec3& acceleration, double xcg) const {
        SloshResult out; out.derivative.assign(4,0.0);
        if (slosh_model_=="none") return out;
        const double xi[2]{s.slosh[0],s.slosh[2]}, rate[2]{s.slosh[1],s.slosh[3]};
        double omega2=slosh_.omega*slosh_.omega, damping=slosh_.omega;
        if(slosh_model_=="pendulum"){const double L=std::max(slosh_.length,1e-3);
            omega2=acceleration[0]/L; damping=std::sqrt(std::abs(acceleration[0])/L+1e-9);}
        double force[2]{};
        for(int i=0;i<2;++i){const double xdd=-2*slosh_.zeta*damping*rate[i]-omega2*xi[i]-acceleration[i+1];
            out.derivative[2*i]=rate[i];out.derivative[2*i+1]=xdd;
            force[i]=slosh_.mass*(omega2*xi[i]+2*slosh_.zeta*damping*rate[i]);}
        out.force={0,force[0],force[1]}; out.moment=cross({slosh_.x_attach-xcg,0,0},out.force); return out;
    }

    RhsResult evaluate(const State& s, const Command& c, const Vec3& wind) const {
        RhsResult out; Aux& aux=out.aux; Derivative& d=out.d;
        aux.altitude=-s.r[2]; const auto [rho,sound]=atmosphere(aux.altitude); aux.rho=rho;
        const Mat3 rotation=dcm(s.q); const auto [xcg,inertia]=mass_props(s.m); aux.x_cg=xcg;
        const Vec3 didm=inertia_derivative(s.m);
        const Mat3 full{inertia[0],mass_.products[0],mass_.products[1],mass_.products[0],inertia[1],mass_.products[2],mass_.products[1],mass_.products[2],inertia[2]};
        const Vec3 velocity_body=transpose_matvec(rotation,sub(s.v,wind));
        aux.v_air_body=velocity_body; aux.mach=norm(velocity_body)/sound;
        const AeroResult aero=body_aero(velocity_body,rho,sound,xcg,s.w);
        Vec throttle,tvc,fin; actuator_inputs(s,c,throttle,tvc,fin); aux.throttle_eff=throttle;
        const FinResult fins=fin_forces(velocity_body,s.w,rho,sound,fin,xcg);
        const EngineResult eng=engines(throttle,tvc,xcg,pressure(aux.altitude)); aux.mdot=eng.mdot;
        const Vec3 duty=mul(s.rcs_fuel>0?1.0:0.0,quantize_rcs(c.rcs,actuator_model_!="realistic"));
        const Vec3 rcs_moment=mul(actuator_.rcs_thrust*actuator_.rcs_arm,duty);
        const Vec3 non_force=add(add(eng.force,aero.force),fins.force);
        const double nozzle=geometry_.x_eng-xcg;
        const Vec3 jet{0,-eng.mdot*nozzle*nozzle*s.w[1],-eng.mdot*nozzle*nozzle*s.w[2]};
        const Vec3 non_moment=add(add(add(eng.moment,aero.moment),fins.moment),add(rcs_moment,jet));
        const Vec3 iw=matvec(full,s.w), idotw{didm[0]*(-eng.mdot)*s.w[0],didm[1]*(-eng.mdot)*s.w[1],didm[2]*(-eng.mdot)*s.w[2]};
        const Vec3 lateral{0,mass_.cg_lateral[0],mass_.cg_lateral[1]};
        const Vec3 wdot_non=solve3(full,sub(sub(add(non_moment,mul(-1.0,cross(lateral,non_force))),cross(s.w,iw)),idotw));
        const Vec3 attach{slosh_.x_attach-xcg,0,0};
        const Vec3 attach_acc=add(div(non_force,s.m),add(cross(wdot_non,attach),cross(s.w,cross(s.w,attach))));
        const SloshResult sl=slosh(s,attach_acc,xcg);
        const Vec3 total_force=add(non_force,sl.force), total_moment=add(non_moment,sl.moment);
        Vec3 frame_acceleration=gravity_;
        if (earth_.enabled) {
            aux.a_coriolis = mul(-2.0, cross(earth_.omega_ned, s.v));
            aux.a_centrifugal = mul(
                -earth_.differential_centrifugal,
                cross(earth_.omega_ned, cross(earth_.omega_ned, s.r)));
            frame_acceleration=add(add(frame_acceleration,aux.a_coriolis),aux.a_centrifugal);
        }
        d.r=s.v;
        d.v=add(frame_acceleration,div(matvec(rotation,total_force),s.m));
        d.q=qdot(s.q,s.w);
        d.w=solve3(full,sub(sub(add(total_moment,mul(-1.0,cross(lateral,total_force))),cross(s.w,iw)),idotw));
        d.m=-eng.mdot; Derivative ad=actuator_derivatives(s,c);
        d.tvc=std::move(ad.tvc);d.tvc_rate=std::move(ad.tvc_rate);d.throttle=std::move(ad.throttle);
        d.fin=std::move(ad.fin);d.fin_rate=std::move(ad.fin_rate);d.rcs_fuel=ad.rcs_fuel;d.slosh=sl.derivative;
        aux.thrust_mag=norm(eng.force);aux.f_aero=aero.force;aux.f_fin=fins.force;aux.thrust=eng.force;aux.moment=total_moment;
        aux.alpha=aero.alpha;aux.q_dyn=aero.qdyn;aux.cd=aero.cd;aux.cl=aero.cl;aux.x_cp=aero.xcp;aux.static_margin=aero.xcp-xcg;
        aux.fin_force_mag=norm(fins.force);aux.fin_transonic=fins.transonic; return out;
    }

    Aux integrate_once(State& state, const Command& command, const Vec3& wind) const {
        const RhsResult base=evaluate(state,command,wind);
        if(state.done)return base.aux;
        const Derivative& k1=base.d;
        const Derivative k2=evaluate(add_scaled(state,k1,0.5*dt_),command,wind).d;
        const Derivative k3=evaluate(add_scaled(state,k2,0.5*dt_),command,wind).d;
        const Derivative k4=evaluate(add_scaled(state,k3,dt_),command,wind).d;
        state=combine_rk4(state,k1,k2,k3,k4,dt_);
        const double qn=std::sqrt(state.q[0]*state.q[0]+state.q[1]*state.q[1]+state.q[2]*state.q[2]+state.q[3]*state.q[3]);
        if(!(qn>1e-15)||!std::isfinite(qn))throw std::runtime_error("invalid integrated quaternion");
        for(double& value:state.q)value/=qn;
        state.m=std::max(state.m,mass_.dry);state.rcs_fuel=std::max(state.rcs_fuel,0.0);
        state.done=-state.r[2]<=0.0||state.m<=mass_.dry+kMassEps;
        return base.aux;
    }

    template <typename T> static void add_vector(T& target,const T& base,const T& delta,double scale){
        for(std::size_t i=0;i<target.size();++i) target[i]=base[i]+scale*delta[i];}
    State add_scaled(const State& s,const Derivative& d,double scale) const { State o=s;
        o.t=s.t+scale*d.t;o.m=s.m+scale*d.m;o.rcs_fuel=s.rcs_fuel+scale*d.rcs_fuel;
        add_vector(o.r,s.r,d.r,scale);add_vector(o.v,s.v,d.v,scale);add_vector(o.q,s.q,d.q,scale);add_vector(o.w,s.w,d.w,scale);
        add_vector(o.tvc,s.tvc,d.tvc,scale);add_vector(o.tvc_rate,s.tvc_rate,d.tvc_rate,scale);
        add_vector(o.throttle,s.throttle,d.throttle,scale);add_vector(o.fin,s.fin,d.fin,scale);
        add_vector(o.fin_rate,s.fin_rate,d.fin_rate,scale);add_vector(o.slosh,s.slosh,d.slosh,scale);return o;}
    template <typename T> static void combine_vector(T& out,const T& base,const T& a,const T& b,const T& c,const T& d,double h){
        for(std::size_t i=0;i<out.size();++i)out[i]=base[i]+h*(a[i]+2*b[i]+2*c[i]+d[i])/6.0;}
    State combine_rk4(const State&s,const Derivative&a,const Derivative&b,const Derivative&c,const Derivative&d,double h) const {State o=s;
        o.t=s.t+h*(a.t+2*b.t+2*c.t+d.t)/6;o.m=s.m+h*(a.m+2*b.m+2*c.m+d.m)/6;
        o.rcs_fuel=s.rcs_fuel+h*(a.rcs_fuel+2*b.rcs_fuel+2*c.rcs_fuel+d.rcs_fuel)/6;
        combine_vector(o.r,s.r,a.r,b.r,c.r,d.r,h);combine_vector(o.v,s.v,a.v,b.v,c.v,d.v,h);
        combine_vector(o.q,s.q,a.q,b.q,c.q,d.q,h);combine_vector(o.w,s.w,a.w,b.w,c.w,d.w,h);
        combine_vector(o.tvc,s.tvc,a.tvc,b.tvc,c.tvc,d.tvc,h);combine_vector(o.tvc_rate,s.tvc_rate,a.tvc_rate,b.tvc_rate,c.tvc_rate,d.tvc_rate,h);
        combine_vector(o.throttle,s.throttle,a.throttle,b.throttle,c.throttle,d.throttle,h);combine_vector(o.fin,s.fin,a.fin,b.fin,c.fin,d.fin,h);
        combine_vector(o.fin_rate,s.fin_rate,a.fin_rate,b.fin_rate,c.fin_rate,d.fin_rate,h);combine_vector(o.slosh,s.slosh,a.slosh,b.slosh,c.slosh,d.slosh,h);return o;}

    py::tuple derivative_tuple(const Derivative& d) const {return py::make_tuple(d.t,array(d.r),array(d.v),array(d.q),array(d.w),d.m,
        matrix_array(d.tvc,engines_.ng,2),matrix_array(d.tvc_rate,engines_.ng,2),array(d.throttle),array(d.fin),array(d.fin_rate),d.rcs_fuel,array(d.slosh));}
    py::dict aux_dict(const Aux&a) const {py::dict d;d["altitude"]=a.altitude;d["rho"]=a.rho;d["mach"]=a.mach;d["thrust_mag"]=a.thrust_mag;
        d["throttle_eff"]=array(a.throttle_eff);d["x_cg"]=a.x_cg;d["F_aero_body"]=array(a.f_aero);d["F_fin_body"]=array(a.f_fin);
        d["T_body"]=array(a.thrust);d["M_body_total"]=array(a.moment);d["mdot"]=a.mdot;d["aero_alpha"]=a.alpha;
        d["a_coriolis_ned"]=array(a.a_coriolis);d["a_centrifugal_ned"]=array(a.a_centrifugal);d["v_air_body"]=array(a.v_air_body);
        d["aero_q_dyn"]=a.q_dyn;d["aero_cd"]=a.cd;d["aero_cl"]=a.cl;d["aero_x_cp"]=a.x_cp;d["aero_static_margin"]=a.static_margin;
        d["fin_fin_force_mag"]=a.fin_force_mag;d["fin_fin_transonic"]=a.fin_transonic;return d;}
    py::dict state_dict(const State&s) const {py::dict d;d["t"]=s.t;d["r_ned"]=array(s.r);d["v_ned"]=array(s.v);d["q_bn"]=array(s.q);d["w_body"]=array(s.w);d["m"]=s.m;
        d["tvc"]=matrix_array(s.tvc,engines_.ng,2);d["tvc_rate"]=matrix_array(s.tvc_rate,engines_.ng,2);d["throttle_act"]=array(s.throttle);
        d["fin"]=array(s.fin);d["fin_rate"]=array(s.fin_rate);d["rcs_fuel"]=s.rcs_fuel;d["slosh"]=array(s.slosh);
        d["wind_mean"]=array(s.wind_mean);d["wind_gust"]=array(s.wind_gust);d["wind_filter"]=matrix_array(s.wind_filter,3,3);
        d["done"]=s.done;return d;}
    py::dict public_snapshot(const State&s) const {py::dict d;d["r_ned"]=array(s.r);d["v_ned"]=array(s.v);
        d["q_bn"]=array(s.q);d["w_body"]=array(s.w);d["m"]=s.m;d["throttle_act"]=array(s.throttle);
        d["tvc"]=matrix_array(s.tvc,engines_.ng,2);d["fin"]=array(s.fin);d["done"]=s.done;return d;}
};

}  // namespace

void bind_native_plant(py::module_& m) {
    py::class_<NativePlant>(m, "NativePlant")
        .def(py::init<py::object, py::dict>(), py::arg("params"), py::arg("meta"))
        .def("create_session", &NativePlant::create_session, py::arg("state"))
        .def("reset", &NativePlant::reset, py::arg("state"))
        .def("advance", &NativePlant::advance, py::arg("command"), py::arg("n_substeps"))
        .def("snapshot", &NativePlant::snapshot, py::arg("full")=false)
        .def("rhs", &NativePlant::rhs, py::arg("state"), py::arg("command"), py::arg("v_wind"))
        .def("step", &NativePlant::step, py::arg("state"), py::arg("command"), py::arg("v_wind"))
        .def("environment_diagnostics", &NativePlant::environment_diagnostics,
             py::arg("altitude"))
        .def("mass_properties_diagnostics", &NativePlant::mass_properties_diagnostics,
             py::arg("mass"))
        .def("engine_wrench_diagnostics", &NativePlant::engine_wrench_diagnostics,
             py::arg("throttle"), py::arg("gimbal"), py::arg("mass"), py::arg("altitude"))
        .def("aero_wrench_diagnostics", &NativePlant::aero_wrench_diagnostics,
             py::arg("velocity_body"), py::arg("rates_body"), py::arg("mass"),
             py::arg("altitude"))
        .def("fin_wrench_diagnostics", &NativePlant::fin_wrench_diagnostics,
             py::arg("velocity_body"), py::arg("rates_body"), py::arg("fin"), py::arg("mass"),
             py::arg("altitude"))
        .def("slosh_diagnostics", &NativePlant::slosh_diagnostics, py::arg("slosh_state"),
             py::arg("acceleration_body"), py::arg("mass"))
        .def("wind_diagnostics", &NativePlant::wind_diagnostics, py::arg("airspeed"),
             py::arg("count"), py::arg("seed"));
}

"""Build-time loading and preprocessing of aerodynamic coefficient tables."""

from __future__ import annotations

import numpy as np


def _pchip_slopes(cos_grid, table):
    """Per-Mach-column monotone-cubic node slopes d(table)/d(cos)."""
    from scipy.interpolate import PchipInterpolator

    cos_grid = np.asarray(cos_grid)
    table = np.asarray(table)
    slope = np.zeros_like(table)
    for column in range(table.shape[1]):
        interpolator = PchipInterpolator(cos_grid, table[:, column])
        slope[:, column] = interpolator.derivative()(cos_grid)
    return slope


def load_table_npz(path):
    """Load a preprocessed file-backed aero table."""
    data = np.load(path)
    return (
        data["cos_grid"],
        data["mach_grid"],
        data["cd_table"],
        data["cl_table"],
        data["xcp_table"],
        data["cd_slope"],
        data["cl_slope"],
        data["xcp_slope"],
        data["cm_table"],
        data["cm_slope"],
        data["x_cg_ref_m"],
    )


def generate_table(alpha_samples, mach_grid, length, coeffs):
    """Generate build-time coefficient arrays for the analytic aero configuration."""
    alpha = np.asarray(alpha_samples)[:, None]
    mach = np.asarray(mach_grid)[None, :]
    length = float(length)

    transonic = coeffs["cd_transonic"] * np.exp(
        -(((mach - coeffs["transonic_mach"]) / coeffs["transonic_width"]) ** 2)
    )
    hyper = (
        coeffs["cd_hyper"]
        * 0.5
        * (1.0 + np.tanh((mach - coeffs["hyper_mach"]) / coeffs["hyper_width"]))
    )
    cd0 = coeffs["cd_base"] + transonic + hyper
    cd = cd0 + coeffs["cd_alpha"] * np.sin(alpha) ** 2
    cl = coeffs["cn_alpha"] * np.sin(alpha) * np.cos(alpha) * np.ones_like(mach)
    xcp_fraction = coeffs["xcp_frac_base"] + coeffs["xcp_frac_alpha"] * np.sin(
        alpha
    ) ** 2 * np.ones_like(mach)
    xcp = -length * xcp_fraction

    shape = (alpha.shape[0], mach.shape[1])
    cd = np.broadcast_to(cd, shape).copy()
    cl = np.broadcast_to(cl, shape).copy()
    xcp = np.broadcast_to(xcp, shape).copy()

    cos_knots = np.cos(np.asarray(alpha_samples))
    order = np.argsort(cos_knots)
    cos_grid = cos_knots[order]
    cd, cl, xcp = cd[order], cl[order], xcp[order]
    return (
        cos_grid,
        cd,
        cl,
        xcp,
        _pchip_slopes(cos_grid, cd),
        _pchip_slopes(cos_grid, cl),
        _pchip_slopes(cos_grid, xcp),
    )

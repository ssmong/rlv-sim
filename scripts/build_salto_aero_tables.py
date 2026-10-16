"""Build the SALTO/T3 file-backed aero table from an external digitized workbook.

The plant uses the committed `data/aero/salto_t3_aero.npz` at runtime. The digitized
workbook is paper-derived provenance, not a runtime dependency, and is not committed.

Usage:
  python scripts/build_salto_aero_tables.py path/to/salto_aero_model.xlsx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "aero" / "salto_t3_aero.npz"

# The workbook coefficients are already nondimensionalized on this AEDB basis.
D_REF_M = 3.70
S_REF_M2 = 10.752
# Reference CoG the workbook CMy_CoG is taken about. NOT documented in the workbook -> a CALIBRATION
# (not a measurement). Set to the descent-CoG band [-19.0, -18.5] (baseline -18.75, conservative
# edge -19.0): chosen so the grid-fin trim margin at the transonic edge is ~1 (marginal), matching
# the fin-saturation-bounded entry corridor reported by Deimos (FAR-2025, Fig. 7-8). The aero
# instability (SM = -Cm/Cn) is invariant to this choice via the parallel-axis term; it only sets the
# offset (x_cg_ref - x_cg_live). Revisit if the fin authority model (fin_x, S_fin, CL_d) changes.
X_CG_REF_M = -18.75


def _pchip_slopes(cos_grid: np.ndarray, table: np.ndarray) -> np.ndarray:
    slope = np.zeros_like(table)
    for col in range(table.shape[1]):
        pch = PchipInterpolator(cos_grid, table[:, col])
        slope[:, col] = pch.derivative()(cos_grid)
    return slope


def _enforce_cmy_axisymmetry(mach_grid, alpha_abs_deg, cmy_mag, cmy_even, cmy_plus) -> None:
    """Enforce CMy oddness from the +alpha branch on any Mach column where the even residual
    dominates (|even|>|odd|) — a digitized -alpha branch error (e.g. Mach 2.0) that would otherwise
    sign-flip the symmetrized moment to a spurious STABLE margin. In place, loud."""
    nonzero = alpha_abs_deg > 0
    for j, mach in enumerate(mach_grid):
        odd = np.abs(cmy_mag[nonzero, j])
        even = np.abs(cmy_even[nonzero, j])
        corrupt = even > odd + 1e-6
        if not corrupt.any():
            continue
        bad_alpha = [int(a) for a in alpha_abs_deg[nonzero][corrupt]]
        ratio = float(np.max(even / np.maximum(odd, 1e-9)))
        print(
            f"[build_salto_aero] Mach {mach}: non-axisymmetric CMy at alpha={bad_alpha} "
            f"(max |even/odd|={ratio:.2f}); enforcing oddness from the +alpha branch."
        )
        cmy_mag[nonzero, j] = -cmy_plus[nonzero, j]


def _read_long_format(path: Path) -> dict[tuple[float, int], tuple[float, float, float]]:
    try:
        import openpyxl
    except ImportError as exc:  # pragma: no cover - exercised by CLI users without openpyxl
        raise RuntimeError("Install openpyxl to rebuild SALTO aero tables.") from exc

    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    ws = wb["LongFormat"]
    rows = list(ws.iter_rows(values_only=True))
    header = {name: i for i, name in enumerate(rows[0])}
    out: dict[tuple[float, int], tuple[float, float, float]] = {}
    for row in rows[1:]:
        mach = float(row[header["Mach"]])
        alpha = int(row[header["Alpha_deg"]])
        cax = float(row[header["CD"]])  # source provenance: positive axial magnitude = -CX
        cz = float(row[header["CL"]])  # source provenance: body-normal coefficient = CZ
        cmy = float(row[header["CMy_CoG"]])
        out[(mach, alpha)] = (cax, cz, cmy)
    return out


def build_table(xlsx: Path, out_path: Path = DEFAULT_OUT) -> None:
    raw = _read_long_format(xlsx)
    mach_grid = np.array(sorted({m for m, _ in raw}), dtype=float)
    alpha_abs_deg = np.array(sorted({abs(a) for _, a in raw}), dtype=int)

    cax = np.zeros((alpha_abs_deg.size, mach_grid.size))
    cz_mag = np.zeros_like(cax)
    cmy_mag = np.zeros_like(cax)
    cmy_even = np.zeros_like(cax)  # even-in-alpha residual; large => a corrupt (asymmetric) branch
    cmy_plus = np.zeros_like(cax)  # +alpha CMy, the oddness-enforcement reference
    for j, mach in enumerate(mach_grid):
        for i, alpha in enumerate(alpha_abs_deg):
            if alpha == 0:
                cax[i, j], _, _ = raw[(mach, 0)]
                continue
            cax_p, cz_p, cmy_p = raw[(mach, alpha)]
            cax_m, cz_m, cmy_m = raw[(mach, -alpha)]
            cax[i, j] = 0.5 * (cax_p + cax_m)
            cz_mag[i, j] = 0.5 * (cz_m - cz_p)
            cmy_mag[i, j] = 0.5 * (cmy_m - cmy_p)
            cmy_even[i, j] = 0.5 * (cmy_m + cmy_p)
            cmy_plus[i, j] = cmy_p
    _enforce_cmy_axisymmetry(mach_grid, alpha_abs_deg, cmy_mag, cmy_even, cmy_plus)

    alpha_rad = np.deg2rad(alpha_abs_deg.astype(float))[:, None]
    cd = cax * np.cos(alpha_rad) + cz_mag * np.sin(alpha_rad)
    cl = cz_mag * np.cos(alpha_rad) - cax * np.sin(alpha_rad)

    xcp = np.empty_like(cmy_mag)
    for j in range(mach_grid.size):
        small = (alpha_abs_deg > 0) & (alpha_abs_deg <= 3)
        slope_cz = np.polyfit(np.deg2rad(alpha_abs_deg[small]), cz_mag[small, j], 1)[0]
        slope_cmy = np.polyfit(np.deg2rad(alpha_abs_deg[small]), cmy_mag[small, j], 1)[0]
        ratio0 = slope_cmy / slope_cz
        for i, alpha in enumerate(alpha_abs_deg):
            if alpha == 0 or abs(cz_mag[i, j]) < 1e-8:
                ratio = ratio0
            else:
                ratio = cmy_mag[i, j] / cz_mag[i, j]
            xcp[i, j] = X_CG_REF_M - D_REF_M * ratio

    cos_knots = np.cos(np.deg2rad(alpha_abs_deg.astype(float)))
    order = np.argsort(cos_knots)
    cos_grid = cos_knots[order]
    cd = cd[order]
    cl = cl[order]
    xcp = xcp[order]
    # cm = symmetrized pitch-moment coefficient about the reference CoG, RAW (no /C_N). The plant
    # builds the moment from this directly (aero_table cm path), avoiding the C_N->0 division that
    # makes the derived xcp ill-conditioned near alpha=0. xcp is kept for the analytic backend and
    # as a logged diagnostic only.
    cm = cmy_mag[order]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        cos_grid=cos_grid,
        mach_grid=mach_grid,
        cd_table=cd,
        cl_table=cl,
        xcp_table=xcp,
        cm_table=cm,
        cd_slope=_pchip_slopes(cos_grid, cd),
        cl_slope=_pchip_slopes(cos_grid, cl),
        xcp_slope=_pchip_slopes(cos_grid, xcp),
        cm_slope=_pchip_slopes(cos_grid, cm),
        alpha_abs_deg=alpha_abs_deg,
        d_ref_m=np.array(D_REF_M),
        s_ref_m2=np.array(S_REF_M2),
        x_cg_ref_m=np.array(X_CG_REF_M),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build data/aero/salto_t3_aero.npz from an external SALTO/T3 workbook."
    )
    parser.add_argument("xlsx", type=Path, help="Path to the digitized SALTO/T3 aero workbook")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output .npz path")
    args = parser.parse_args()
    if not args.xlsx.exists():
        raise SystemExit(f"workbook not found: {args.xlsx}")
    build_table(args.xlsx, args.out)

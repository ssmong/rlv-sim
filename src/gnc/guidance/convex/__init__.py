"""Convex powered-descent guidance factories."""

from __future__ import annotations


def _convex_kw(gd) -> dict:
    """Common SOCP keyword args shared by the convex example methods (from the guidance config)."""
    return dict(
        N=int(gd.convex_nodes),
        target=tuple(float(x) for x in gd.convex_target_ned),
        solver=str(gd.convex_solver),
        # Single source (ACB-2013 (19)): plan thrust-pointing cone == body-tilt cap == tracking
        # projection cone (the body tilts to direct sustained thrust; the gimbal is reserved for
        # stabilization around the tilted reference).
        theta_max_deg=float(gd.attitude_tilt_max_deg),
        gamma_gs_deg=gd.convex_glideslope_deg,
        tilt_max_deg=float(gd.attitude_tilt_max_deg),
        tf_search_iters=int(gd.convex_tf_search_iters),
        reanchor_min_alt_m=float(gd.convex_reanchor_min_alt_m),
        grid_fallback=bool(gd.convex_tf_search_grid_fallback),
        k_r=float(gd.convex_tracking_k_r),
        k_v=float(gd.convex_tracking_k_v),
        tf_floor=float(gd.convex_tf_min_s),
        landing_cutoff_impact_mps=float(gd.landing_cutoff_impact_mps),
        terminal_slack=bool(gd.convex_terminal_slack),
        terminal_pos_weight=float(gd.convex_terminal_pos_weight),
        terminal_vel_weight=float(gd.convex_terminal_vel_weight),
        terminal_upright=bool(gd.convex_terminal_upright),
        terminal_vz_max_mps=(
            None if gd.convex_terminal_vz_max_mps is None else float(gd.convex_terminal_vz_max_mps)
        ),
        tf_freeze_s=float(gd.convex_tf_freeze_s),
        replan=bool(gd.convex_replan),
    )


def _convex_nominal(ctx) -> dict:
    """Augment the shared nominal dict with the convex-only SOCP scaling / drag-template keys."""
    gd = ctx.cfg
    nom = dict(ctx.nominal)
    # Fixed SOCP variable scales keep conditioning + slack-penalty meaning solve-invariant;
    # null => per-solve state-derived scaling.
    nom["scale_pos"] = None if gd.convex_scale_pos is None else float(gd.convex_scale_pos)
    nom["scale_vel"] = None if gd.convex_scale_vel is None else float(gd.convex_scale_vel)
    nom["scale_accel"] = None if gd.convex_scale_accel is None else float(gd.convex_scale_accel)
    # The LCvx method alone uses its frozen-coefficient drag template.
    drag_cd = float(gd.convex_drag_cd) if "convex_drag_cd" in gd else 0.0
    nom["drag_cd_s"] = drag_cd * float(ctx.params.geom.s_ref)
    return nom


def _grid_kw(gd) -> dict:
    """Grid and planned thrust-rate options shared by LCvx and SCvx."""
    return dict(
        dt=None if gd.convex_dt is None else float(gd.convex_dt),
        n_cap=int(gd.convex_nodes_max),
        slew_rate_max=(
            None if gd.convex_slew_rate_max_mps3 is None else float(gd.convex_slew_rate_max_mps3)
        ),
        thrust_rate_max_nps=(
            None if gd.convex_thrust_rate_max_nps is None else float(gd.convex_thrust_rate_max_nps)
        ),
    )


def build_lcvx(ctx):
    """Factory for the LCvx powered-descent guidance (ctx: gnc.context.GncContext)."""
    from gnc.guidance.convex.guidance import LcvxGuidance

    gd = ctx.cfg
    return LcvxGuidance(
        _convex_nominal(ctx),
        ctx.active_engines,
        ctx.phase,
        **_grid_kw(gd),
        **_convex_kw(gd),
    )


def build_scvx(ctx):
    """Factory for the SCvx (drag-aware) powered-descent guidance (ctx: gnc.context.GncContext)."""
    from gnc.guidance.convex.scvx import ScvxGuidance

    gd = ctx.cfg
    return ScvxGuidance(
        _convex_nominal(ctx),
        ctx.active_engines,
        ctx.phase,
        params=ctx.params,
        drag_scale=float(gd.convex_drag_scale),
        scvx_iters=int(gd.scvx_iters),
        scvx_trust=float(gd.scvx_trust),
        scvx_virtual_control_weight=float(gd.scvx_virtual_control_weight),
        scvx_tolerance=float(gd.scvx_tolerance),
        **_grid_kw(gd),
        **_convex_kw(gd),
    )

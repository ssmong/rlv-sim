"""Guidance (G) — produces the reference Command (thrust vector / attitude) per phase.

Contract: __call__(x_hat, t, memory) -> (Command, memory); optional initial_memory(). Build the
Command with the helpers in base.py.
Framework: base.py (helpers) · phase.py (scheduler + guards) · scheduled.py (per-phase dispatcher).
Provided laws (siblings here): reentry_burn, aero_descent, convex/ (lcvx, scvx), template (copy-me).
The powered-descent method is config-selected (gnc/guidance/<name>.yaml sets law._target_ to a
build(ctx) factory); reentry/aero are fixed per-phase laws wired in gnc/build. See docs/.
"""

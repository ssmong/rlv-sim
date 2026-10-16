"""Navigation (N) — estimates the state x_hat the rest of GNC consumes.

Contract: measure(state, key) -> Measurement; estimate(meas, act, memory) -> (x_hat, memory).
Provided filters (siblings here): identity, ekf_stub, template (copy-me).
Select via config (gnc/navigation/<name>.yaml sets law._target_ to a build(ctx) factory); see
docs/adding_a_module.md.
"""

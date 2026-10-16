"""Control (C) — inner attitude loop; fills Command.tau_body (body torque [N·m]).

Contract: __call__(x_hat, command, memory) -> (command, memory) with tau_body set; optional
initial_memory(). Provided laws (siblings here): pd, template (copy-me).
Select via config (gnc/control/<name>.yaml sets law._target_ to a build(ctx) factory); see
docs/adding_a_module.md.
"""

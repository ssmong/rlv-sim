# Adding a GNC module

Guidance, navigation, and control use the same extension path: copy a runnable module and YAML pair,
rename them, implement the contract, and select the new Hydra group. `gnc/build.py` normally needs
no edit.

| Seam | Module | Config | Selection |
|---|---|---|---|
| Guidance | `src/gnc/guidance/template.py` | `configs/gnc/guidance/template.yaml` | `gnc/guidance=<name>` |
| Navigation | `src/gnc/navigation/template.py` | `configs/gnc/navigation/template.yaml` | `gnc/navigation=<name>` |
| Control | `src/gnc/control/template.py` | `configs/gnc/control/template.yaml` | `gnc/control=<name>` |

The templates are executable before modification:

```bash
python scripts/run_single.py gnc/guidance=template scenario=pdg
python scripts/run_single.py gnc/navigation=template
python scripts/run_single.py gnc/control=template
```

## Construction

Each YAML file points `law._target_` to a module-level `build(ctx)` factory. The factory returns the
object used at runtime. `ctx` is a `gnc.context.GncContext`:

```python
ctx.cfg             # this module's resolved config
ctx.nominal         # derived vehicle and environment constants
ctx.params          # numeric PlantParams
ctx.active_engines  # guidance only
ctx.phase           # guidance only
```

Read only the fields the module needs. Estimated state `x_hat` provides `r_ned`, `v_ned`, `q_bn`,
`w_body`, and `m`.

## Runtime contracts

| Seam | Required methods | Result |
|---|---|---|
| Guidance | `__call__(x_hat, t, memory)` | `(thrust_ned, memory)` or `(Command, memory)` |
| Navigation | `measure(state, key)` and `estimate(meas, act, memory)` | `Measurement` and `(x_hat, memory)` |
| Control | `__call__(x_hat, command, memory)` | `(command, memory)` with `tau_body` filled |

`initial_memory()` is optional. Use the returned `memory` for a warm start, filter state, or
integrator; return it unchanged for a stateless module.

Guidance normally returns a NED thrust vector. The framework attaches phase, active engines, and a
thrust-aligned attitude reference. Return a full `Command` when the law owns attitude or cutoff.
A guidance law may also implement `track(x_hat, t, memory)` for control-rate tracking between slower
guidance updates.

Allocation is currently a single implementation in `gnc/allocation/default.py`, not a selected
plug-in seam. Reentry and aerodynamic-descent laws are wired in `gnc/build.py`; only powered-descent
guidance is selected by `gnc/guidance=<name>`.

A compiled implementation can sit behind the same Python object through an extension module or a
thin `ctypes` wrapper. Add that boundary only when the Python implementation is actually too slow.

## Checklist

1. Copy and rename the module and config pair.
2. Set `law._target_: gnc.<seam>.<name>.build` and add only keys the implementation reads.
3. Keep the existing method signatures and return types.
4. Run the selected module once, then add a focused regression test under `tests/`.
5. Run `python -m pytest tests/ -q` and `ruff check src tests scripts`.

See [Architecture](1_architecture.md) for the complete stack and native-plant boundary.

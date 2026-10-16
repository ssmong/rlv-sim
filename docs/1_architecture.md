# Architecture

How the testbed is wired, and where the seams are. See [`2_adding_a_module.md`](2_adding_a_module.md) for
the step-by-step plug-in guide.

## Top-level packages (`src/`)

| Package | Role |
|---|---|
| `core/`  | **Contracts.** Frozen NumPy-backed state/command/parameter records (`interfaces.py`), native frame/quaternion conventions (`frames.py`), and deterministic seed derivation. |
| `plant/` | **Native-plant construction.** Resolves configuration, preprocesses coefficient tables, and creates initial state plus the native C++ session. The 6-DOF EoM, RK4, actuator, wind, and component models have one implementation in `native/plant.cpp`. |
| `gnc/`   | **The subsystem you plug into.** Guidance + Navigation + Control + Allocation, a phase scheduler, and the assembly (`stack.py`, `build.py`) that wires them from config. |
| `sim/`   | **The harness.** Closed-loop multirate rollout (`loop.py`), KPI scoring, Monte-Carlo aggregation, replay export, open-loop schedules. |

## The closed-loop tick

`sim/loop.py` drives the loop. One control tick (single-rate; the multirate path fires nav / guidance
/ control on independent strides and holds ZOH between):

```
state (plant truth)
  │  nav.measure(state, key)         -> Measurement      (sensor model)
  │  nav.estimate(meas, act, mem)    -> x_hat            (state estimate)        ◄── NAVIGATION
  │  guidance(x_hat, t, mem)         -> Command          (thrust vector / q_ref) ◄── GUIDANCE
  │  controller(x_hat, Command, mem) -> Command.tau_body (body torque)           ◄── CONTROL
  │  allocate(Command, x_hat)        -> ActuatorCommand  (throttle/gimbal/fin/RCS)◄── ALLOCATION
  ▼  native.advance(act, n_substeps) -> tick-end public state
```

The four GNC modules are the seams. `Command` is filled progressively: guidance sets the reference
(`thrust_vector` or `q_ref`), control adds `tau_body`, allocation turns the whole thing into actuator
deflections. The plant never sees a `Command` — only the `ActuatorCommand`.

The data types crossing each seam are defined in `core/interfaces.py`: `Measurement`, `Command`,
`ActuatorCommand`, `PlantState`, and the `*Params` structs.

## Assembly: config → modules (`gnc/build.py`)

The stack is assembled by **name**, via Hydra `instantiate`, so adding a module never touches the
wiring code:

1. `default_nominal(...)` derives vehicle / plant constants into a `nominal` dict. Controller gains
   are NOT here — each controller sizes its own in its `build(ctx)`.
2. `build_scheduler(...)` builds the latching phase scheduler (`gnc/guidance/phase.py`) from the
   `gnc/scheduler` config group.
3. For the powered-descent guidance, control, and navigation, `build_gnc` / `build_nav` construct a
   **`GncContext`** and call `instantiate(cfg.<seam>.law, ctx)`. Each module's config carries
   `law._target_: <module>.build`, a factory taking the context.

```python
ctx = GncContext(cfg=gd, nominal=nominal, params=params, active_engines=active, phase=phase)
guidance = instantiate(gd.law, ctx)        # gd.law._target_ -> build(ctx)
```

### `GncContext` — the one bundle every module receives

```python
@dataclass(frozen=True)
class GncContext:
    cfg:         # this module's resolved config node (read your own params here)
    nominal:     # vehicle/plant constants dict — full key list in adding_a_module.md
    params:      # PlantParams (numeric vehicle/environment leaves)
    active_engines: # (N,) engines this guidance phase fires    (guidance only; else None)
    phase:       # int Phase tag this guidance instance runs in  (guidance only; else -1)
```

A factory reads only what its module needs and ignores the rest. This is why a new guidance law and a
new EKF use the same construction path with no special cases in `build.py`.

## The four seams (interface contracts)

Each seam ships a set of provided modules (siblings of the framework files under `src/gnc/<seam>/`)
and (for the config-selected seams) a config group under `configs/gnc/`; the method signature a module
must satisfy is in the table below and demonstrated in the seam's `template.py`.

| Seam | Contract (method signature) | Selected by | Provided modules |
|---|---|---|---|
| Guidance   | `__call__(x_hat, t, memory) -> (thrust_ned, memory)` (or a `Command`; + `initial_memory`) | `gnc/guidance` `law._target_` | reentry_burn, aero_descent, convex/{lcvx, scvx}, template |
| Navigation | `measure(state, key) -> Measurement`; `estimate(meas, act, memory) -> (x_hat, memory)` | `gnc/navigation` `law._target_` | identity, ekf_stub, template |
| Control    | `__call__(x_hat, Command, memory) -> (Command, memory)` (fills `tau_body`) | `gnc/control` `law._target_` | pd, template |
| Allocation | `allocate(Command, x_hat) -> ActuatorCommand` | fixed (`gnc/allocation/default.py`) | scheduled gimbal/fin/RCS |

`guidance` also holds the mission-sequencing framework: `phase.py` (the latching `SequentialPhaseScheduler`
+ guard factories) and `scheduled.py` (the `ScheduledGuidance` dispatcher that routes each phase to its
guidance law). Reentry/aero are fixed per-phase laws wired directly in `build.py`; the **powered-descent
method** is the config-selected one.

## Phases

`Phase` (core/interfaces.py) is a pass-through KPI tag — the plant never branches on it. The scheduler
advances `REENTRY_BURN → AERO_DESCENT → [POWERED_BRAKE] → POWERED_DESCENT` on single-crossing guards
(altitude thresholds, or a generic ψ(x; θ) ≥ 0 with `gnc/scheduler=guard`). `scenario.start_phase`
starts a run mid-sequence for isolated module testing.

## Native plant boundary and determinism

`build_plant` constructs an immutable native model once. Each episode calls `create_session(state0)`;
the session then owns the complete mutable plant state, actuator internals, slosh modes, wind filter,
and RNG. One control tick crosses the language boundary as follows:

| Direction | Per control tick | Deliberately not transferred |
|---|---|---|
| Python -> C++ | one held `ActuatorCommand` and integer `n_substeps` | parameters and current plant state |
| C++ -> Python | tick-end pose/velocity/mass, actuator positions used by replay, `done`, last diagnostics | servo rates, slosh, wind state/filter, RNG, RK4 stage values |

All `n_substeps` RK4 updates therefore run behind one boundary call. Sessions are separate objects,
so concurrent Monte-Carlo episodes do not share mutable state. The array-based native diagnostic
methods exercise that same C++ implementation in focused physics tests; there is no Python copy of
the plant equations.

The native RNG state remains inside the session. A seed enters once through the initial state, which
makes a repeated seed deterministic without returning random keys at every substep.

## Earth model

The state position and velocity are a displacement and derivative in a fixed-origin local NED frame.
The default `environment/earth=fixed` uses flat, non-rotating equations. Selecting
`environment/earth=rotating` and setting `environment.earth.latitude_deg` adds

`a_rot = -2 omega_ie x v_ned - omega_ie x (omega_ie x r_ned)`.

The second term is differential centrifugal acceleration relative to the NED origin. The configured
`g_ned` is apparent gravity at that origin and already includes its baseline centrifugal effect. This
avoids double counting. Both terms execute inside the C++ RK4 stages; Python receives only their final
diagnostics when requested.

This switch intentionally does not claim spherical/ellipsoidal trajectory propagation. A fixed local
NED frame has no moving-frame transport rate or Earth-curvature kinematics. Missions for which those
effects matter need an LLA/ECEF navigation state and ECEF plant propagation, rather than an additional
correction term in this local plant.

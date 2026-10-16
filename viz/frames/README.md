# Frame Studio definition format

Frame Studio reads a frame definition from the viewer's **Import JSON** button.

Each definition names two or three ordered axes. Two-axis definitions receive an implicit plane
normal. Three-axis definitions are checked against `handedness` and the third axis is reversed when
needed.

```json
{
  "id": "paper_frame",
  "short": "PAPER",
  "name": "Frame name used by the paper",
  "citation": "Author (Year, Journal)",
  "description": "Convention and any discrepancy noted in the paper.",
  "handedness": "right",
  "axes": [
    { "id": "x", "label": "x", "subscript": "P", "name": "first axis", "direction": { "ref": "los" } },
    { "id": "y", "label": "y", "subscript": "P", "name": "second axis", "direction": { "project": [{ "ref": "up" }, { "axis": "x" }] } },
    { "id": "z", "label": "z", "subscript": "P", "name": "normal", "direction": { "cross": [{ "axis": "x" }, { "axis": "y" }] } }
  ]
}
```

Available live references are:

- `north`, `east`, `up`, `down`
- `velocity`, `air_relative`, `wind`, `los`
- `body.x`, `body.y`, `body.z`
- `velocity.x`, `velocity.y`, `velocity.z` and `los.x`, `los.y`, `los.z` from the native
  Forward-Right-Down direction-frame convention.

Direction expressions may be a fixed `[x, y, z]` vector or one of:

```json
{ "ref": "velocity" }
{ "axis": "x" }
{ "negate": { "ref": "down" } }
{ "cross": [{ "axis": "x" }, { "axis": "y" }] }
{ "project": [{ "ref": "up" }, { "axis": "x" }] }
{ "add": [{ "ref": "north" }, { "ref": "east" }] }
{ "fallback": [{ "project": [{ "ref": "up" }, { "axis": "x" }] }, { "ref": "east" }] }
```

Axis references are resolved from top to bottom, so an axis can only refer to axes declared before
it. Axes are orthogonalized in declaration order by default; set `"orthogonalize": false` on an
axis only when the source convention intentionally uses a non-orthogonal basis.

Expressions that are undefined in the current state are not assigned an arbitrary axis. Velocity
is unavailable at zero speed and LOS is unavailable when observer and target coincide. For a
vertical direction, the built-in Velocity and LOS frames use the same deterministic North heading
datum as `src/native/frames.cpp`; imported projected-up frames remain singular unless their JSON
definition explicitly supplies a `fallback`.

Built-in frames are `VEL`, `LOS`, `ENU`, `NED`, and `BODY`. Direction frames use
`x = unit(direction)`, `y = unit(Down × x)`, and `z = x × y`. Body uses the FRD convention
+*x*<sub>B</sub> forward, +*y*<sub>B</sub> right, and +*z*<sub>B</sub> down. Use the optional `subscript` field to render axis
symbols typographically rather than embedding the subscript in `label`.

# RLV Descent Replay — 3D Viewer

A Vizard-style (Basilisk-like) web viewer for closed-loop reusable-launch-vehicle descent
episodes. It renders a Falcon-9-class **first stage** flying the actual logged trajectory — a
photoreal glTF hull (the bundled Forest Katsch landing booster, auto-isolated from its display scene to the first stage), otherwise a
procedural booster — with the 3D path, **9-engine octaweb** plumes that light up with throttle,
grid fins, a gimballing center engine, and a **SpaceX-broadcast-style HUD**: a bottom band with the
big **SPEED / ALTITUDE** readout and a **mission timeline** (Aero Guidance → Landing Burn →
Landing) flanking a `T+` clock, an **octaweb 9-engine status ring**, an engineering telemetry
column (alt/speed/vert/downrange/mass/Mach/tilt/thrust/TWR), a TVC-gimbal + grid-fin actuator card,
and playback controls. The scene is set at an **autonomous-spaceport-drone-ship at sea**: a
physically-based golden-hour sky (three.js `Sky`), a reflective animated ocean (three.js `Water`,
using the bundled `waternormals.jpg`), and a procedural ASDS barge with a painted bullseye/X landing
target at the origin.

Everything is driven by a single `replay.json` exported from the simulator. The viewer runs out of
the box with the procedural booster (**no licensing strings attached**); drop in a glTF for the
photoreal hull (see below).

## 1. Get a replay

**Normal workflow — every `run_single` writes its own replay.** A closed-loop rollout auto-exports
`replay.json` into its Hydra run dir (`results/run_single/<date>/<time>/`), alongside `kpi.json` /
`trajectory.csv`. No separate export step, and runs never overwrite each other (disable with
`run.save_replay=false`):

```bash
PYTHONPATH=src python scripts/run_single.py gnc/scheduler=guard
#  -> results/run_single/2026-06-09/14-30-22/replay.json
```

For a custom scenario, just pass overrides to `run_single.py` (each run writes its own `replay.json`).

## 2. View it

**Point the viewer at a run** (serves the repo root, loads that run's `replay.json` straight from
its folder — nothing is copied; default = the newest replay under `results/`):

```bash
PYTHONPATH=src python scripts/view_replay.py                                 # latest run
PYTHONPATH=src python scripts/view_replay.py results/run_single/2026-06-09/14-30-22
# opens http://localhost:8000/viz/index.html?replay=/results/.../replay.json
```

**Bare viewer** (drag-drop) — serve this folder over HTTP (`file://` fetch is blocked by the
browser), then drop a `replay.json` onto the page:

```bash
cd viz && python3 -m http.server 8000
# open http://localhost:8000/
```

The page loads `?replay=<url>` if present, else the co-located `./replay.json`. With no server (or
either load failing) the loader lets you **drop a `replay.json`** onto the page (or pick one).

**Controls:** drag to orbit · scroll to zoom · `space` play/pause · `←/→` step · `C` cycle camera
(Orbit / Chase / Pad) · the bottom bar has play, reset, a scrubber, speed (0.25–4×), and the camera
toggle. Coordinate-frame geometry is intentionally kept out of trajectory replay.

### Frame Studio: standard coordinate-frame explorer

Frame Studio is intentionally separate from trajectory replay. It contains no trajectory, landing
HUD, timeline, or engine telemetry: only a spacecraft, geometric state controls, coordinate frames,
a selected plane, and vector projection. Start it directly from the repository root; no simulation
or `replay.json` is required:

```bash
uv run python scripts/view_frames.py --system
```

The initial view shows the ground-velocity frame and enables the fixed NED origin, LOS target, and
LOS ray. The five built-ins are:

- **VEL:** `x`<sub>V</sub> follows ground-relative velocity **V**<sub>G</sub>,
  `y`<sub>V</sub> = **D** × `x`<sub>V</sub>, and `z`<sub>V</sub> =
  `x`<sub>V</sub> × `y`<sub>V</sub>.
- **LOS:** `x`<sub>L</sub> points from the vehicle to the target; its other axes use the same
  direction-frame construction.
- **ENU:** fixed East, North, Up local-navigation axes.
- **NED:** fixed North, East, Down native-navigation axes.
- **BODY:** vehicle-fixed FRD axes: `x`<sub>B</sub> forward, `y`<sub>B</sub> right, and
  `z`<sub>B</sub> down.

Here **D** is local down. VEL and LOS use the same equations, vertical-line North datum, and
singularity thresholds as the native C++ direction-frame transforms. A zero direction is undefined.

The bundled Falcon 9 first stage is normalized to FRD Body axes and centered at the state origin.
Body roll φ<sub>B</sub> is an Euler attitude coordinate. Lift-vector bank σ<sub>L</sub> uses the
standard Wind-frame lift direction
−`z`<sub>W</sub> = sin(α)`x`<sub>B</sub> − cos(α)`z`<sub>B</sub>. It is the signed rotation from
the zero-bank lift-up direction to −`z`<sub>W</sub> about air-relative +`x`<sub>W</sub>. It is
undefined at zero airspeed, for vertical air-relative velocity, or when air-relative velocity is
purely lateral in Body axes. Velocity-frame axes retain their deterministic North datum in vertical
flight, but that coordinate convention is not reported as a physical bank angle.

Air-relative velocity is **V**<sub>A</sub> = **V**<sub>G</sub> − **V**<sub>W</sub>. In FRD body
components [*u*<sub>B</sub>, *v*<sub>B</sub>, *w*<sub>B</sub>], Frame Studio computes
α = atan2(*w*<sub>B</sub>, *u*<sub>B</sub>) and
β = atan2(*v*<sub>B</sub>, √(*u*<sub>B</sub>² + *w*<sub>B</sub>²)), equivalently
asin(*v*<sub>B</sub> / |**V**<sub>A</sub>|). The α/β geometry layer draws the body
`x`<sub>B</sub>–`z`<sub>B</sub> plane, the longitudinal projection, lateral residual, and the same
signed arcs. α is undefined for a purely lateral air-relative velocity; both angles are undefined at
zero airspeed.

Ground velocity **V**<sub>G</sub> and wind **V**<sub>W</sub> are independent editable 3D vectors.
Drag either arrow-tip handle to change direction and magnitude in the current camera plane; orbit the
camera and drag again to alter the remaining depth component. Manual **V**<sub>G</sub> edits preserve
**V**<sub>W</sub>. The body component *w*<sub>B</sub> is not wind velocity.

Use the local-axis gimbal in the lower-right dock for constrained CAD-style attitude editing, or
switch to direct drag for screen-relative free rotation. Dragging empty space orbits the camera.
Coordinate-frame axes use long, thin dashed shafts and compact axis tags; editable physical vectors
use solid thick shafts with circular grab handles. Panel definitions and sign conventions are kept in
delayed native hover tooltips so the controls remain uncluttered.
Slider-backed attitude, trajectory, and vehicle-position values also accept direct numeric entry.
Wind accepts direct North/Up/East components. The editable FRD values `u`<sub>B</sub>,
`v`<sub>B</sub>, and `w`<sub>B</sub> specify air-relative velocity; editing them preserves attitude
and wind, reconstructs the air-relative vector in local N–U–E with the Body-to-local attitude, then applies
**V**<sub>G</sub> = **V**<sub>A</sub> + **V**<sub>W</sub>. Speed, flight-path angle, and course remain
synchronized.
Click any two axes from one visible frame to display their plane, then project
**V**<sub>G</sub>, **V**<sub>A</sub>, **V**<sub>W</sub>, LOS, or `x`<sub>B</sub> onto it. The
original vector, in-plane projection, component construction, out-of-plane residual, and signed
components are shown together.

Frame Studio places the fixed NED origin at the LOS target and accepts the vehicle position
**r**<sub>V/O</sub><sup>NED</sup> = [*N*<sub>V</sub>, *E*<sub>V</sub>, *D*<sub>V</sub>]. The NED basis is
fixed independently of LOS; the LOS direction is derived from the relative position and points from
the vehicle to that origin. The target can be any aimpoint required by a frame definition and need
not represent a landing site. For a stable comparison view, the renderer translates the scene so the
vehicle and all vehicle-attached frames remain at the display origin. In display N–U–E coordinates,
the NED origin is therefore drawn at [−*N*<sub>V</sub>, *D*<sub>V</sub>, −*E*<sub>V</sub>]. This is a
visual translation only and does not change any relative vector, angle, or frame definition.
Rotating the vehicle changes body-frame LOS components, not the NED origin or world-space LOS ray.
The Descent, Crosswind, and Vertical scenario presets replace attitude, ground velocity, and wind
while preserving vehicle position. **Align x**<sub>B</sub> **to V**<sub>A</sub> is an attitude-only operation: it
preserves **V**<sub>G</sub>, **V**<sub>W</sub>, and body roll while setting α = β = 0. It is unavailable
at zero air-relative speed; for vertical **V**<sub>A</sub>, the existing body heading is retained.

New paper conventions can be loaded with **Import JSON**. The schema is documented in
[`viz/frames/README.md`](frames/README.md), with an importable example in
[`example_paper_frame.json`](frames/example_paper_frame.json). Frame axes are declarative expressions
over standard directions, body axes, velocity, air-relative velocity, wind, and LOS, so a new paper
normally requires a small definition rather than new rendering code.

## Frame convention

The plant works in NED (z down); `src/sim/replay.py` does the frame change **once** so the
viewer stays simple:

```
world = [North, Up, East] = [r_ned[0], -r_ned[2], r_ned[1]]      # proper rotation, det = +1
q     = body→world = q_vn ⊗ q_bn,   q_vn = [√2/2, √2/2, 0, 0]    # NED→world, +90° about X
```

Quaternions are emitted in three.js order `[x, y, z, w]`. The rocket mesh is built in **body**
coordinates (local +X = nose, base at x = −L); the pivot group carries `q` and the CG world
position, and an inner group offsets by `−x_cg` so the integrated center of mass lands exactly on
the pivot. Engine positions and grid-fin arms come straight from the plant params, so plumes/fins
sit where the physics puts them.

## Photoreal glTF model (bundled — loads automatically)

The glTF loader is **already wired** in `app.js` (`tryLoadPhotorealModel`) and the viewer ships with a
hull, so it loads on start; with no GLB it silently falls back to the procedural booster. The bundled
default is **`viz/resources/falcon9_landing.glb`** — the Forest Katsch *Falcon 9* **landing** scene
(a first-stage booster with legs, topped by a Dragon 2 capsule on a pad). On start the loader:

1. **Isolates the first-stage booster** — hides the Dragon 2 capsule/trunk and the pad scenery
   (ground, railings, signs) **by name**, so only the booster is measured and shown.
2. **Collects the 4 landing legs** into a deploy group, hidden until the **leg-deploy trigger**, when
   they snap out. `run_single` writes that time to `meta.leg_deploy.t` as the moment the booster first
   drops below `run.leg_deploy_alt_m` (default 60 m; CLI-overridable, e.g.
   `run.leg_deploy_alt_m=40`); if the field is absent the viewer falls back to 5 s before touchdown.
   With no GLB the procedural legs instead swing out smoothly by altitude (~45→15 m).
3. **Re-fits the visible booster** to the real length `L` (long axis → body +X, base at `x=-L`,
   interstage top at `x=0`), measuring a *visible-only* bounding box so the capsule/pad never skew it.
4. **Hides the procedural hull, grid fins, and legs** (kept as the no-asset fallback) and keeps the
   **9 procedural engine plumes** overlaid on the GLB engine bells (the GLB has no flame). The single
   replay throttle channel drives the **center engine** (landing-burn config); a per-engine export
   (`thr.length ≥ 9`) is used directly when present.

**Swapping the model.** Both Forest Katsch hulls are bundled with asset-specific licenses:
`falcon9_landing.glb` (legs, the default) is CC BY 4.0, while `falcon9.glb` (the legs-stowed flight
stack) is CC BY-NC 4.0. Attribution is shown in-app and in Credits. To fly the stowed hull, point
`MODEL.url` at `./resources/falcon9.glb`; the loader's name
filter also hides that model's second stage / fairing / M-Vac so only the first stage shows. To use a
*different* booster, find one on Sketchfab (e.g. search **"Forest Katsch Falcon 9"**; a free account is
needed to download), verify its license, save it under `viz/resources/`, and point `MODEL.url` at it.

Tuning knobs are the `MODEL` object at the top of `app.js`:

```js
const MODEL = {
  url: "./resources/falcon9_landing.glb", // landing hull (legs); ./resources/falcon9.glb = flight (stowed)
  rotation: [0, 0, -Math.PI / 2],         // align the model's long axis to body +X (Y-up model default)
  flip: false,                            // set true if the nose ends up pointing the wrong way
  credit: "Falcon 9 model: Forest Katsch (Sketchfab) · CC BY 4.0",
};
```

Most models need no tuning (the loader bounding-box-fits and centers automatically); if the rocket
lands on its side or upside-down, adjust `rotation`/`flip`. Any other CC0/CC-BY booster glTF works
too — just update `url` and `credit` (keep a visible attribution for CC-BY).

### Drone-ship model (optional)

The sea scene's drone-ship is **bundled** as `viz/resources/droneship.glb` — the CC BY 4.0 *Space X Drone Barge*
by **TAIGA-ZOE** (Sketchfab), auto-fit to `DRONESHIP.lengthM` and centered on the landing target. If
the file is absent it falls back to a fully procedural barge (no asset needed). To swap in a different
model, replace `viz/resources/droneship.glb` and tune the `DRONESHIP` object at the top of `app.js`
(`lengthM`, `rotation`, `yOffset`, `sinkFrac`, `seaSinkFrac`, `credit`); if the deck loads tilted or
sunk, adjust `rotation`, `yOffset`, and the `sinkFrac`/`seaSinkFrac` seating knobs.

### Credits

- Viewer + procedural booster + procedural drone-ship: this repo (runs with no third-party asset).
- [three.js](https://threejs.org) (MIT), loaded from the unpkg CDN; `Sky`, `Water`, and
  `GLTFLoader` from its `examples/jsm` addons.
- `waternormals.jpg` — three.js example texture (MIT), bundled in `viz/resources/` so the ocean needs no CDN.
- **Replay hull — *SpaceX Falcon 9 and Dragon 2* by Forest Katsch (Sketchfab), CC BY 4.0**, bundled
  as `viz/resources/falcon9_landing.glb`. Attribution is shown in-app (bottom-right) and here.
- **Drone-ship — *Space X Drone Barge* by TAIGA-ZOE (Sketchfab), CC BY 4.0**, bundled as
  `viz/resources/droneship.glb`. Attribution shown in-app and here.
- **Frame Studio booster — *SpaceX Falcon 9 Block 4.5* by Forest Katsch (Sketchfab),
  CC BY-NC 4.0**, bundled as `viz/resources/falcon9.glb`. Frame Studio isolates the first stage and
  shows the attribution when it is loaded.
- CC BY 4.0: <https://creativecommons.org/licenses/by/4.0/>.
- CC BY-NC 4.0: <https://creativecommons.org/licenses/by-nc/4.0/>.

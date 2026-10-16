// RLV descent replay viewer — procedural Falcon-9-class booster, data-driven from replay.json.
// Frame convention (set by src/sim/replay.py): world = [North, Up, East] (three.js Y-up),
// per-frame quaternion q = body->world in [x,y,z,w]. The rocket mesh is built in BODY coords
// (local +X = nose, base at x=-L); the pivot group carries q and the CG world position, and an
// inner group offsets by -x_cg so the integrated CG point lands exactly on the pivot. No external
// asset is required; a glTF can be swapped in at the marked hook (see viz/README.md).

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { Sky } from "three/addons/objects/Sky.js";
import { Water } from "three/addons/objects/Water.js";

const PHASE_COLOR = { 0: "#ff6b4a", 1: "#ffc850", 2: "#a0c8ff", 3: "#6be0a0" };
const $ = (id) => document.getElementById(id);
let TOTAL_T = 0; // full replay duration [s] (set in init, read by the HUD clock label)

// Optional PHOTOREAL hull. Drop a glTF/glb in viz/ (e.g. the CC-BY Forest Katsch Falcon 9 from
// Sketchfab, saved as falcon9.glb — see viz/README.md) and it replaces the procedural body. The
// loader auto-fits it to the real length and aligns its long axis to body +X. CC-BY REQUIRES
// VISIBLE attribution: `credit` is shown in the corner whenever the model loads. Absent file =>
// silent fall-back to the procedural booster (no credit shown).
const MODEL = {
  // The LANDING booster (legs DEPLOYED), isolated at runtime from the Forest Katsch landing scene: the
  // Dragon capsule + pad scenery are hidden by name, and the 4 deployed-leg subtrees stay hidden until the
  // leg-deploy trigger (meta.leg_deploy.t in replay.json). Swap url back to ./resources/falcon9.glb for the
  // legs-stowed flight hull (no deploy).
  url: "./resources/falcon9_landing.glb",
  rotation: [0, 0, -Math.PI / 2], // orient the model's long axis to body +X (Y-up model default)
  flip: false, // set true if the nose ends up pointing the wrong way after rotation
  credit: "Falcon 9 model: Forest Katsch (Sketchfab) · CC BY 4.0",
};

// Optional DRONESHIP glTF. Drop a CC0/CC-BY drone-ship/barge glb in viz/ as droneship.glb and it
// replaces the procedural barge: auto-fit so its longest horizontal span = lengthM, centered on the
// landing target (origin), with the hull bottom at the waterline. Absent file => procedural barge kept.
const DRONESHIP = {
  url: "./resources/droneship.glb", // null disables; drop a model in as viz/resources/droneship.glb to use it
  lengthM: 64,            // scale so the model's longest horizontal span equals this (metres)
  rotation: [0, 0, 0],    // tweak if the deck isn't flat/level (e.g. [-Math.PI/2, 0, 0] for a Z-up model)
  yOffset: 0,             // +metres nudge after the fractional sink below (fine final adjustment)
  sinkFrac: 0,         // sink the model this × its own height below groundY (bottom ~1/4 is a sub-keel fitting)
  seaSinkFrac: 0.5,       // drop the waterline this × the model height so the hull sits in the sea
  credit: "Drone ship: TAIGA-ZOE (Sketchfab) · CC BY 4.0", // shown in-app (CC-BY attribution)
};

// Sea + sky look — one place to set the time of day. Lower `exposure` = darker/less washed out.
//   clear afternoon (default): elevation 22, turbidity 4,  rayleigh 3.0, exposure 0.42
//   golden dusk:               elevation 4,  turbidity 10, rayleigh 2.2, exposure 0.50, warmer waterColor
//   overcast / moody:          elevation 28, turbidity 12, rayleigh 1.0, exposure 0.36, greyer waterColor
const SKY = {
  elevation: 4, azimuth: 150,                  // low dusk sun
  turbidity: 10, rayleigh: 2.2, mie: 0.005,    // hazy, warm golden-hour atmosphere
  exposure: 0.48,                              // ACES tone-map exposure (lower = darker)
  waterColor: 0x132230, sunColor: 0xffce92, waveScale: 3.4,
};

// ---------------------------------------------------------------- load replay
// Source order: ?replay=<url> (a specific run's replay.json, e.g. from scripts/view_replay.py) else
// the co-located ./replay.json. Either failing falls through to the drag/drop + file picker.
async function boot() {
  const requested = new URLSearchParams(location.search).get("replay");
  const src = requested || "./replay.json";
  let data = null;
  try {
    const r = await fetch(src, { cache: "no-cache" });
    if (r.ok) data = await r.json();
  } catch (e) {
    /* file:// or not served — fall through to picker */
  }
  if (data) return init(data);
  if (requested) {
    const err = $("err");
    if (err) err.textContent = `Could not load ${src} — pick or drop a replay.json instead.`;
  }

  const finish = (txt) => {
    try {
      init(JSON.parse(txt));
    } catch (e) {
      $("err").textContent = "Failed to parse replay.json: " + e.message;
    }
  };
  $("pick").onclick = () => $("file").click();
  $("file").onchange = (e) => e.target.files[0]?.text().then(finish);
  const drop = $("drop");
  drop.ondragover = (e) => (e.preventDefault(), (drop.style.borderColor = "#4ea3ff"));
  drop.ondragleave = () => (drop.style.borderColor = "");
  drop.ondrop = (e) => {
    e.preventDefault();
    e.dataTransfer.files[0]?.text().then(finish);
  };
}

// ---------------------------------------------------------------- scene setup
function init(replay) {
  const meta = replay.meta;
  const V = meta.vehicle;
  const frames = replay.frames;
  $("loader").classList.add("hidden");

  const app = $("app");
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setSize(innerWidth, innerHeight);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping; // filmic look for the PBR sky/ocean
  renderer.toneMappingExposure = SKY.exposure; // darker/richer — keeps the sea and sky from washing out
  app.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  // sky + ocean + droneship are built in buildEnvironment() below (needs the sun light + renderer)

  const camera = new THREE.PerspectiveCamera(50, innerWidth / innerHeight, 0.5, 12000);
  camera.position.set(70, 60, 90);

  const controls = new OrbitControls(camera, renderer.domElement);
  // Damping is off: the follow-cam translates camera.position every frame, and OrbitControls' damped
  // re-solve of a moving rig produces a visible jitter. The follow has its own 0.15 lerp for smoothing.
  controls.enableDamping = false;
  controls.maxPolarAngle = Math.PI * 0.9; // allow orbiting UNDER the booster to look up at the engines/legs
  controls.minDistance = 8;
  controls.maxDistance = 3000;

  // lights — sky/ground hemisphere + a low sun + soft fill
  scene.add(new THREE.HemisphereLight(0xbfd3ff, 0x223040, 0.35)); // env map provides most of the fill
  const sun = new THREE.DirectionalLight(0xfff2dd, 1.5);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  scene.add(sun);

  // Touchdown alignment: the plant integrates the CG (frame.p) and lands it at alt 0, but the engine/
  // leg plane sits |x_eng - x_cg| below the CG, so the booster base rests that far under the trajectory's
  // y=0. The whole sea + barge drops by that much (freeboard preserved) so the deck meets the legs at
  // touchdown rather than the booster passing through it.
  const groundY = -Math.abs(V.x_eng - V.x_cg_dry); // world-y of the landing deck (legs rest here) ~ -12 m
  const env = buildEnvironment(scene, renderer, sun, groundY); // sky + ocean + droneship lowered to groundY
  const traj = buildTrajectory(scene, frames);
  const rocket = buildVehicle(V);
  scene.add(rocket.group);

  // engine glow light (brightens with total throttle)
  const plumeLight = new THREE.PointLight(0xff8a44, 0, 240, 2);
  scene.add(plumeLight);

  // ground contact shadow blob (fades/grows with altitude)
  const blob = new THREE.Mesh(
    new THREE.CircleGeometry(V.diameter * 1.4, 32),
    new THREE.MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.35 })
  );
  blob.rotation.x = -Math.PI / 2;
  blob.position.y = groundY + 0.05; // contact shadow sits on the (lowered) deck
  scene.add(blob);

  buildHUD(V);

  // ------------------------------------------------------------ playback state
  const tEnd = frames[frames.length - 1].t;
  TOTAL_T = tEnd;
  // leg-deploy trigger time: set per-run in replay.json (meta.leg_deploy.t); else 5 s before touchdown
  const legDeployT = (meta.leg_deploy && typeof meta.leg_deploy.t === "number")
    ? meta.leg_deploy.t : Math.max(0, tEnd - 5);
  buildTimeline(frames, tEnd, meta.phase_names || {});
  const state = { t: 0, playing: true, speed: 1, cam: "orbit" };
  const scrub = $("scrub");

  // reusable temporaries
  const qA = new THREE.Quaternion(), qB = new THREE.Quaternion(), qC = new THREE.Quaternion();
  const pA = new THREE.Vector3(), pB = new THREE.Vector3(), pCur = new THREE.Vector3();
  const vCur = new THREE.Vector3();

  function frameAt(t) {
    // binary search for the segment [i, i+1] containing t
    let lo = 0, hi = frames.length - 1;
    if (t <= frames[0].t) return { a: frames[0], b: frames[0], s: 0, i: 0 };
    if (t >= tEnd) return { a: frames[hi], b: frames[hi], s: 0, i: hi };
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (frames[mid].t <= t) lo = mid;
      else hi = mid;
    }
    const a = frames[lo], b = frames[hi];
    const s = (t - a.t) / Math.max(b.t - a.t, 1e-9);
    return { a, b, s, i: lo };
  }

  function applyFrame(t) {
    const { a, b, s, i } = frameAt(t);
    pA.fromArray(a.p); pB.fromArray(b.p);
    pCur.lerpVectors(pA, pB, s);
    qA.fromArray(a.q); qB.fromArray(b.q);
    qC.slerpQuaternions(qA, qB, s);
    vCur.set(
      a.v[0] + (b.v[0] - a.v[0]) * s,
      a.v[1] + (b.v[1] - a.v[1]) * s,
      a.v[2] + (b.v[2] - a.v[2]) * s
    );

    // pose: pivot at CG world, orientation q_bv, inner offset -x_cg puts CG on the pivot
    rocket.group.position.copy(pCur);
    rocket.group.quaternion.copy(qC);
    const xcg = a.xcg + (b.xcg - a.xcg) * s;
    rocket.inner.position.set(-xcg, 0, 0);

    // per-engine throttle (frame.thr = per-engine throttle_eff) drives each plume; gimbal -> tilt. The
    // kN readout uses the delivered thrust magnitude (frame.thrust, back-pressure corrected); the octaweb
    // % shows the mean throttle of the firing engines (the lit dots show which / how many).
    const thrustN = a.thrust + (b.thrust - a.thrust) * s; // delivered thrust [N]
    const engThr = engineThrottles(a.thr, b.thr, s, rocket.engines.length);
    let totThr = 0, nFiring = 0;
    for (let e = 0; e < rocket.engines.length; e++) {
      totThr += engThr[e];
      if (engThr[e] > 0.02) nFiring++;
      rocket.setPlume(e, engThr[e]);
    }
    const octFrac = nFiring ? totThr / nFiring : 0; // mean throttle over firing engines (HUD %)
    for (let g = 0; g < V.n_gimbal; g++) {
      const gp = lerpRow(a.gim, b.gim, s, g);
      rocket.setGimbal(g, gp[0], gp[1]);
    }
    for (let f = 0; f < V.n_fins; f++) rocket.setFin(f, lerpArr(a.fin, b.fin, s, f));

    // landing legs: photoreal GLB legs snap deployed at the trigger (meta.leg_deploy.t); the procedural
    // fallback (no GLB) still deploys smoothly by altitude.
    const alt = a.alt + (b.alt - a.alt) * s;
    if (rocket.hasGlbLegs()) rocket.setLegsDeployed(t >= legDeployT);
    else rocket.setLegs(THREE.MathUtils.clamp((45 - alt) / 30, 0, 1));

    // plume light + ground blob (driven by the strongest active engine)
    const maxThr = engThr.reduce((m, x) => Math.max(m, x), 0);
    plumeLight.intensity = maxThr * 7;
    plumeLight.position.copy(pCur);
    const groundFade = THREE.MathUtils.clamp(alt / 400, 0, 1);
    blob.position.set(pCur.x, groundY + 0.05, pCur.z);
    blob.material.opacity = 0.42 * (1 - groundFade * 0.7);
    blob.scale.setScalar(1 + groundFade * 6);

    traj.setProgress(i);
    updateHUD(a, b, s, V, vCur, totThr, engThr, octFrac, thrustN);
    return { pCur, vCur, alt };
  }

  // ------------------------------------------------------------ cameras
  // OrbitControls stays enabled in EVERY mode, so the user can always rotate + zoom. The mode buttons
  // only re-frame the shot: orbit/chase keep the booster as the look-at target (so the camera follows
  // it), pad anchors the look-at on the droneship (watch the booster fly past from the deck).
  const PAD_TARGET = new THREE.Vector3(0, 10, 0);
  const lastPos = new THREE.Vector3(), lastVel = new THREE.Vector3(), camDelta = new THREE.Vector3();
  function updateCamera(pos, vel) {
    lastPos.copy(pos); lastVel.copy(vel);
    if (state.cam === "pad") {
      // watch from the deck: fixed camera, just re-aim the look-at at the landing target
      controls.target.lerp(PAD_TARGET, 0.1);
    } else {
      // orbit/chase: translate BOTH the orbit target and the camera by the same delta so the camera
      // physically follows the booster down. Moving only controls.target leaves the camera anchored
      // and merely tilts it — OrbitControls recomputes position = target + offset every update, so a
      // fixed camera would just look further down instead of descending. The SHARED delta preserves
      // the camera<->target offset, so the user's zoom level and orbit angle are kept while it follows.
      camDelta.copy(pos).sub(controls.target).multiplyScalar(0.15);
      controls.target.add(camDelta);
      camera.position.add(camDelta);
    }
    controls.update();
  }
  function frameCamera(mode) { // set a good initial pose for the mode; the user can then adjust freely
    const p = lastPos;
    if (mode === "pad") {
      camera.position.set(54, 13, 54); controls.target.copy(PAD_TARGET);
    } else if (mode === "chase") {
      const back = lastVel.lengthSq() > 1 ? lastVel.clone().normalize() : new THREE.Vector3(0, 1, 0);
      camera.position.copy(p).addScaledVector(back, -34).add(new THREE.Vector3(16, 14, 24)); // up-and-behind
      controls.target.copy(p);
    } else { // orbit: 3/4 high establishing angle
      camera.position.copy(p).add(new THREE.Vector3(64, 42, 78)); controls.target.copy(p);
    }
    controls.update();
  }

  // ------------------------------------------------------------ controls wiring
  $("btnPlay").onclick = () => setPlaying(!state.playing);
  $("btnReset").onclick = () => { state.t = 0; setPlaying(true); };
  function setPlaying(p) {
    state.playing = p;
    $("btnPlay").textContent = p ? "⏸ Pause" : "▶ Play";
  }
  $("speeds").onclick = (e) => {
    const s = e.target.dataset.s; if (!s) return;
    state.speed = parseFloat(s);
    [...$("speeds").children].forEach((b) => b.classList.toggle("active", b === e.target));
  };
  $("cams").onclick = (e) => {
    const c = e.target.dataset.cam; if (!c) return;
    state.cam = c;
    frameCamera(c); // re-frame for the new mode; OrbitControls stays enabled so rotate/zoom keep working
    [...$("cams").children].forEach((b) => b.classList.toggle("active", b === e.target));
  };
  scrub.oninput = () => {
    state.t = (scrub.value / 1000) * tEnd;
    if (state.playing) setPlaying(false);
    applyFrame(state.t);
  };
  addEventListener("keydown", (e) => {
    if (e.code === "Space") { e.preventDefault(); setPlaying(!state.playing); }
    else if (e.code === "ArrowRight") { state.t = Math.min(tEnd, state.t + 0.1); applyFrame(state.t); }
    else if (e.code === "ArrowLeft") { state.t = Math.max(0, state.t - 0.1); applyFrame(state.t); }
    else if (e.key.toLowerCase() === "c") {
      const order = ["orbit", "chase", "pad"];
      const next = order[(order.indexOf(state.cam) + 1) % 3];
      $("cams").querySelector(`[data-cam=${next}]`).click();
    }
  });
  addEventListener("resize", () => {
    camera.aspect = innerWidth / innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(innerWidth, innerHeight);
  });

  // ------------------------------------------------------------ main loop
  const clock = new THREE.Clock();
  function tick() {
    requestAnimationFrame(tick);
    const dt = Math.min(clock.getDelta(), 0.05);
    if (state.playing) {
      state.t += dt * state.speed;
      if (state.t >= tEnd) { state.t = tEnd; setPlaying(false); }
      scrub.value = (state.t / tEnd) * 1000;
    }
    const { pCur, vCur } = applyFrame(state.t);
    updateCamera(pCur, vCur);
    if (env.water) env.water.material.uniforms.time.value += dt; // animate the ocean
    renderer.render(scene, camera);
  }
  applyFrame(0);
  controls.target.copy(pCur);
  tick();
}

// ---------------------------------------------------------------- interpolation helpers
function lerpArr(a, b, s, i) { return a[i] + (b[i] - a[i]) * s; }
function lerpRow(a, b, s, i) {
  const ra = a[i] || [0, 0], rb = b[i] || [0, 0];
  return [ra[0] + (rb[0] - ra[0]) * s, ra[1] + (rb[1] - ra[1]) * s];
}

// ---------------------------------------------------------------- procedural vehicle
function buildVehicle(V) {
  const group = new THREE.Group(); // pivot: CG world position + q_bv
  const inner = new THREE.Group(); // offset by -x_cg so CG sits at the pivot origin
  group.add(inner);

  const L = V.length, R = V.diameter / 2;
  const white = new THREE.MeshStandardMaterial({ color: 0xe9edf2, metalness: 0.25, roughness: 0.55 });
  const soot = new THREE.MeshStandardMaterial({ color: 0x4a4d55, metalness: 0.35, roughness: 0.7 });
  const dark = new THREE.MeshStandardMaterial({ color: 0x15171c, metalness: 0.5, roughness: 0.6 });
  const metal = new THREE.MeshStandardMaterial({ color: 0x8a8f98, metalness: 0.85, roughness: 0.35 });

  // hideOnModel: procedural parts the photoreal GLB supersedes (hull + grid fins + legs). When the GLB
  // loads they switch off so nothing doubles; with no GLB they remain as the fallback booster. The engine
  // PLUMES are deliberately NOT in this list — the GLB has no flame, so the plumes stay procedural.
  const hideOnModel = [];

  // --- airframe (body spans x: 0 nose .. -L base) ---
  const body = cyl(R, R, L, white); body.position.x = -L / 2; body.castShadow = true; inner.add(body);
  const sootBand = cyl(R * 1.003, R * 1.003, L * 0.33, soot); sootBand.position.x = -L * 0.835; inner.add(sootBand);
  const interstage = cyl(R * 1.004, R * 1.004, L * 0.1, dark); interstage.position.x = -L * 0.05; inner.add(interstage);
  const base = cyl(R * 1.02, R * 0.96, L * 0.05, dark); base.position.x = -L * 0.975; inner.add(base); // octaweb base ring
  hideOnModel.push(body, sootBand, interstage, base);

  // --- grid fins (near the top; rotate about their radial hinge by δ) ---
  const fins = [];
  for (let i = 0; i < V.n_fins; i++) {
    const arm = V.fin_arm[i];
    const phi = Math.atan2(arm[2], arm[1]); // azimuth from body +Y
    const azim = new THREE.Group();
    azim.position.x = arm[0];
    azim.rotation.x = phi; // rotate about body +X so local +Y points radially out
    const hinge = new THREE.Group(); // rotates about local +Y (radial) by deflection
    const blade = gridFin(R, metal);
    hinge.add(blade);
    azim.add(hinge);
    inner.add(azim);
    fins.push(hinge);
    hideOnModel.push(azim); // GLB carries its own grid fins
  }

  // --- engines: 9-engine octaweb PLUME rig. The nozzle bells are the GLB 'm1ds' mesh (9 real Merlins);
  //     only the flames are procedural (the GLB has none). Layout + sizes measured from falcon9.glb at a
  //     42 m length-fit: ring radius 1.081 m, bell exit radius 0.33 m. Index 0 = the center landing engine.
  const engines = [];
  const fit = L / 42;                                 // measurements were taken at a 42 m fit
  const RING_R = 1.081 * fit, BELL_R = 0.33 * fit;
  for (let e = 0; e < 9; e++) {
    let y = 0, z = 0;
    if (e > 0) { const th = ((e - 1) / 8) * Math.PI * 2 - Math.PI / 2; y = RING_R * Math.cos(th); z = RING_R * Math.sin(th); }
    const gim = new THREE.Group();      // gimbal pivot at the engine; carries the plume
    gim.position.set(-L, y, z);          // engine exit plane at the base (x = -L)
    const plume = buildPlume(BELL_R);    // flame mouth sits at the exit, tapers aft (-X), width = GLB bell
    gim.add(plume);
    inner.add(gim);
    engines.push({ gim, plume, gimbalSlot: e === 0 ? 0 : -1 });
  }

  // --- landing legs (4, at base; deploy outward on final approach) ---
  const legs = [];
  const nLegs = 4;
  for (let i = 0; i < nLegs; i++) {
    const psi = (i / nLegs) * Math.PI * 2 + Math.PI / 4;
    const hinge = new THREE.Group();
    hinge.position.set(-L * 0.97, R * 0.9 * Math.cos(psi), R * 0.9 * Math.sin(psi));
    hinge.rotation.x = Math.atan2(Math.sin(psi), Math.cos(psi));
    const leg = cyl(0.18, 0.12, L * 0.18, dark);
    leg.position.x = L * 0.09; // along body when stowed
    hinge.add(leg);
    inner.add(hinge);
    legs.push(hinge);
    hideOnModel.push(hinge); // GLB carries its own landing legs
  }

  // PHOTOREAL glTF (viz/resources/falcon9_landing.glb): isolate the first-stage booster from its display
  // scene (hide the Dragon capsule + pad scenery), re-fit to length L, and collect the 4 deployed-leg
  // subtrees (revealed at the leg-deploy trigger). Procedural hull/fins/legs switch off; plumes stay.
  // Absent file => procedural booster fallback. See viz/README.md.
  const glb = { legGroups: [], loaded: false };
  tryLoadPhotorealModel(inner, hideOnModel, L, glb);

  const defLim = V.tvc_deflection_limit || 0.14;
  return {
    group, inner, engines, fins,
    setPlume(e, thr) { const p = engines[e].plume;
      p.scale.set(Math.max(thr, 0.001), 0.6 + thr * 0.4, 0.6 + thr * 0.4);
      p.visible = thr > 0.02;
      p.children.forEach((c) => (c.material.opacity = c.userData.base * Math.min(thr * 1.4, 1))); },
    setGimbal(slot, pitch, yaw) { // tilts the center engine's plume (the GLB bells are a single mesh)
      const eng = engines.find((x) => x.gimbalSlot === slot);
      if (eng) { eng.gim.rotation.z = THREE.MathUtils.clamp(pitch, -defLim, defLim);
        eng.gim.rotation.y = THREE.MathUtils.clamp(yaw, -defLim, defLim); } },
    setFin(i, delta) { if (fins[i]) fins[i].rotation.y = delta; },
    setLegs(t) { legs.forEach((h) => (h.rotation.z = -t * 2.2)); }, // procedural fallback: 0 stowed..~126° out
    hasGlbLegs() { return glb.loaded && glb.legGroups.length > 0; },
    setLegsDeployed(on) { for (const g of glb.legGroups) g.visible = on; }, // photoreal legs snap in at trigger
  };
}

function buildPlume(rExit) {
  const g = new THREE.Group();
  const outer = new THREE.Mesh(
    new THREE.ConeGeometry(rExit * 1.05, 6.0, 18, 1, true),
    new THREE.MeshBasicMaterial({ color: 0xff8a3c, transparent: true, opacity: 0.5,
      blending: THREE.AdditiveBlending, side: THREE.DoubleSide, depthWrite: false })
  );
  const inner = new THREE.Mesh(
    new THREE.ConeGeometry(rExit * 0.55, 4.3, 16, 1, true),
    new THREE.MeshBasicMaterial({ color: 0x9fd2ff, transparent: true, opacity: 0.85,
      blending: THREE.AdditiveBlending, side: THREE.DoubleSide, depthWrite: false })
  );
  outer.userData.base = 0.5; inner.userData.base = 0.85;
  // cones default along +Y, apex up; rotate so the flame points -X (aft) with apex away from engine
  for (const c of [outer, inner]) { c.geometry.rotateZ(Math.PI / 2); c.position.x = -c.geometry.parameters.height / 2; }
  g.add(outer); g.add(inner);
  return g;
}

function gridFin(R, mat) {
  // a flat lattice panel sitting radially outboard (local +Y), spanning ~body-x
  const g = new THREE.Group();
  const w = R * 0.7, h = R * 0.85, span = R * 0.7;
  const frame = new THREE.Mesh(new THREE.BoxGeometry(h, span, 0.06), mat);
  frame.position.set(-R * 0.1, R + span / 2, 0);
  g.add(frame);
  // lattice bars for the "grid" look
  const bar = new THREE.MeshStandardMaterial({ color: 0x6a7079, metalness: 0.8, roughness: 0.4 });
  for (let k = -2; k <= 2; k++) {
    const v = new THREE.Mesh(new THREE.BoxGeometry(h, 0.05, 0.16), bar);
    v.position.set(-R * 0.1, R + span / 2 + (k / 5) * span, 0); g.add(v);
    const u = new THREE.Mesh(new THREE.BoxGeometry(0.05, span, 0.16), bar);
    u.position.set(-R * 0.1 + (k / 5) * h, R + span / 2, 0); g.add(u);
  }
  return g;
}

function cyl(rTop, rBot, h, mat) {
  const m = new THREE.Mesh(new THREE.CylinderGeometry(rTop, rBot, h, 28), mat);
  m.geometry.rotateZ(-Math.PI / 2); // align cylinder axis with +X
  m.castShadow = true; m.receiveShadow = true;
  return m;
}

// Load the photoreal landing booster and isolate it from its display scene: the Forest Katsch landing GLB
// is a first-stage booster (legs DEPLOYED) topped by a Dragon 2 capsule on a pad with railings/signs. We
// hide the capsule + scenery BY NAME, re-fit only the visible booster to length L (long axis -> body +X,
// base at x=-L), and collect the deployed-leg parts into `glb.legGroups`, hidden until the leg-deploy
// trigger reveals them. The procedural hull/fins/legs in `hideList` switch off; the 9 plumes stay and
// overlay the GLB engine bells. On any error (file absent) this no-ops and the procedural booster is kept.
function tryLoadPhotorealModel(inner, hideList, L, glb) {
  if (!MODEL.url) return;
  // GLTFLoader sanitizes node names (spaces/dots -> _), so match on a normalized key (alphanumerics only).
  const norm = (s) => (s || "").toLowerCase().replace(/[^a-z0-9]/g, "");
  // Hide the Dragon 2 capsule + trunk and the pad scenery (ground/railings/signs) by substring — none of
  // these collide with a first-stage part name. isLegPart flags the deployable legs (every "leg*" node
  // except the body-side "leg cover top") so they can be revealed only at the deploy trigger.
  const DROP = ["dragon", "capsule", "nosecone", "heatshield", "petal", "docking", "idatop", "sdpod",
    "windows", "trunk", "ground", "railing", "sign", "door", "plane", "body",
    "stagetwo", "fairing", "m1dvac", "sun"]; // last row: also isolates the flight falcon9.glb if swapped back
  const isDrop = (n) => DROP.some((k) => norm(n).includes(k));
  const isLegPart = (n) => { const k = norm(n); return k.startsWith("leg") && !k.startsWith("legcovertop"); };
  // bbox over visible meshes only (Box3.setFromObject would include the hidden upper stack)
  const visBox = (model) => {
    model.updateWorldMatrix(true, true);
    const box = new THREE.Box3(), tmp = new THREE.Box3();
    model.traverse((o) => {
      if (!o.isMesh) return;
      for (let p = o; p; p = p.parent) if (p.visible === false) return; // skip any hidden subtree
      if (!o.geometry.boundingBox) o.geometry.computeBoundingBox();
      tmp.copy(o.geometry.boundingBox).applyMatrix4(o.matrixWorld);
      box.union(tmp);
    });
    return box;
  };
  new GLTFLoader().load(
    MODEL.url,
    (gltf) => {
      const model = gltf.scene;
      model.rotation.set(MODEL.rotation[0], MODEL.rotation[1], MODEL.rotation[2]);
      if (MODEL.flip) model.rotateY(Math.PI);
      // hide capsule + scenery, and collect + hide the deployable legs, BEFORE measuring the booster
      model.traverse((o) => {
        if (isDrop(o.name)) o.visible = false;
        else if (isLegPart(o.name)) { glb.legGroups.push(o); o.visible = false; }
      });
      // model is still unparented => matrixWorld == its own local chain; measure/fit in that frame
      let box = visBox(model);
      const size = box.getSize(new THREE.Vector3());
      model.scale.multiplyScalar(L / Math.max(size.x, size.y, size.z, 1e-6)); // first-stage long span -> L
      box = visBox(model);
      const c = box.getCenter(new THREE.Vector3());
      model.position.x += -box.max.x; // base at x=-L, interstage top at x=0
      model.position.y += -c.y;
      model.position.z += -c.z;
      model.traverse((o) => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; } });
      inner.add(model);
      hideList.forEach((m) => (m.visible = false)); // procedural hull/fins/legs off; plumes stay on
      glb.loaded = true; // deployable legs collected in glb.legGroups (hidden until the deploy trigger)
      const el = $("credit");
      if (el) { el.textContent = MODEL.credit; el.style.display = "block"; }
      console.log("[viz] landing booster loaded:", MODEL.url, "— deployable leg parts:", glb.legGroups.length);
    },
    undefined,
    () => console.log(`[viz] no photoreal model at ${MODEL.url} — using the procedural booster`),
  );
}

// ---------------------------------------------------------------- environment (droneship at sea)
// Builds a real-landing-site backdrop: a physically-based golden-hour sky (three.js Sky/Preetham), a
// reflective animated ocean (three.js Water), and a procedural autonomous-spaceport drone-ship at the
// landing target. Image-based lighting is baked from the sky so the booster picks up sky/horizon tones.
function buildEnvironment(scene, renderer, sun, groundY) {
  scene.fog = new THREE.FogExp2(0xaab9c8, 0.00012); // sea haze (set before Water so it inherits fog)

  // --- sky (Preetham), warm late-afternoon sun (clear enough to read as a real sky) ---
  const sky = new Sky(); sky.scale.setScalar(450000);
  const u = sky.material.uniforms;
  u.turbidity.value = SKY.turbidity; u.rayleigh.value = SKY.rayleigh; u.mieCoefficient.value = SKY.mie; u.mieDirectionalG.value = 0.85;
  const sunDir = new THREE.Vector3().setFromSphericalCoords(
    1, THREE.MathUtils.degToRad(90 - SKY.elevation), THREE.MathUtils.degToRad(SKY.azimuth));
  u.sunPosition.value.copy(sunDir);

  // warm, low-angle key light from the sun direction
  sun.position.copy(sunDir).multiplyScalar(500);
  sun.color.set(0xffdcb0); sun.intensity = 2.6;
  Object.assign(sun.shadow.camera, { left: -60, right: 60, top: 60, bottom: -60, near: 1, far: 1400 });
  sun.shadow.camera.updateProjectionMatrix();

  // image-based lighting baked from the sky (render it in a throwaway scene, then move it to the main one)
  const pmrem = new THREE.PMREMGenerator(renderer);
  const envScene = new THREE.Scene(); envScene.add(sky);
  scene.environment = pmrem.fromScene(envScene).texture;
  scene.add(sky); // re-parent the sky dome into the live scene

  // --- reflective ocean ---
  const water = new Water(new THREE.PlaneGeometry(20000, 20000), {
    textureWidth: 512, textureHeight: 512,
    waterNormals: new THREE.TextureLoader().load(
      "./resources/waternormals.jpg", // bundled (three.js example texture, MIT) — no CDN dependency
      (t) => { t.wrapS = t.wrapT = THREE.RepeatWrapping; }),
    sunDirection: sunDir.clone(), sunColor: SKY.sunColor, waterColor: SKY.waterColor,
    distortionScale: 3.8, fog: true,
  });
  const waterY = groundY - 1.5; // 1.5 m freeboard: the deck (=groundY) sits above the lowered waterline
  water.rotation.x = -Math.PI / 2; water.position.y = waterY; scene.add(water);
  water.material.uniforms.size.value = SKY.waveScale; // tighter, more frequent waves -> open-sea texture

  buildDroneship(scene, groundY, waterY, water); // landing target; the GLB loader sinks hull + sea to fit
  return { water, sky };
}

// painted drone-ship deck texture: weathered steel + dashed caution border + SpaceX bullseye/X target
function droneshipDeck(lm, wm) {
  const ppm = 16, W = Math.round(lm * ppm), H = Math.round(wm * ppm);
  const c = document.createElement("canvas"); c.width = W; c.height = H;
  const g = c.getContext("2d");
  g.fillStyle = "#2c2f35"; g.fillRect(0, 0, W, H);
  g.fillStyle = "rgba(0,0,0,0.12)"; // grime streaks
  for (let i = 0; i < 90; i++) g.fillRect(Math.random() * W, Math.random() * H, 4 + Math.random() * 40, 2);
  g.strokeStyle = "#c8a11c"; g.lineWidth = 10; g.setLineDash([40, 26]); // caution border
  g.strokeRect(34, 34, W - 68, H - 68); g.setLineDash([]);
  const cx = W / 2, cy = H / 2; // bullseye
  g.strokeStyle = "rgba(235,238,242,0.92)";
  [16, 10.5, 5.5].forEach((r, i) => { g.lineWidth = i === 0 ? 9 : 7; g.beginPath(); g.arc(cx, cy, r * ppm, 0, Math.PI * 2); g.stroke(); });
  g.lineWidth = 12; g.lineCap = "round"; const d = 11 * ppm; // big X
  g.beginPath(); g.moveTo(cx - d, cy - d); g.lineTo(cx + d, cy + d); g.moveTo(cx + d, cy - d); g.lineTo(cx - d, cy + d); g.stroke();
  g.fillStyle = "rgba(235,238,242,0.95)"; g.beginPath(); g.arc(cx, cy, 1.6 * ppm, 0, Math.PI * 2); g.fill();
  const tex = new THREE.CanvasTexture(c); tex.colorSpace = THREE.SRGBColorSpace; tex.anisotropy = 4;
  return tex;
}

function buildDroneship(scene, groundY, waterY, water) {
  const lm = 64, wm = 44, hull = 5, deckTopY = 1.5; // metres (ASDS-scale barge); deck +deckTopY above origin
  const grp = new THREE.Group();
  const hullMesh = new THREE.Mesh(
    new THREE.BoxGeometry(lm, hull, wm),
    new THREE.MeshStandardMaterial({ color: 0x14171c, metalness: 0.6, roughness: 0.6 })
  );
  hullMesh.position.y = deckTopY - hull / 2; hullMesh.castShadow = true; hullMesh.receiveShadow = true; grp.add(hullMesh);
  const deck = new THREE.Mesh(
    new THREE.PlaneGeometry(lm, wm),
    new THREE.MeshStandardMaterial({ map: droneshipDeck(lm, wm), metalness: 0.1, roughness: 0.85 })
  );
  deck.rotation.x = -Math.PI / 2; deck.position.y = deckTopY + 0.02; deck.receiveShadow = true; grp.add(deck);
  const wallMat = new THREE.MeshStandardMaterial({ color: 0x1b1f25, metalness: 0.5, roughness: 0.6 });
  for (const s of [-1, 1]) { // containment walls along the long edges
    const wall = new THREE.Mesh(new THREE.BoxGeometry(lm, 2.2, 1.2), wallMat);
    wall.position.set(0, deckTopY + 1.1, s * (wm / 2 - 0.6)); wall.castShadow = true; grp.add(wall);
  }
  grp.position.y = waterY; // origin = waterline; internal deck (+deckTopY) lands exactly at groundY
  scene.add(grp);
  tryLoadDroneshipModel(scene, grp, groundY, waterY, water); // swap in a photoreal droneship.glb if one is present
  return grp;
}

// Optional photoreal drone-ship: if viz/droneship.glb resolves, fit it to DRONESHIP.lengthM, center it
// on the landing target, seat its DECK at the landing plane, and hide the procedural barge. Absent => no-op.
function tryLoadDroneshipModel(scene, procedural, groundY, waterY, water) {
  if (!DRONESHIP.url) return;
  new GLTFLoader().load(
    DRONESHIP.url,
    (gltf) => {
      const m = gltf.scene;
      m.rotation.set(DRONESHIP.rotation[0], DRONESHIP.rotation[1], DRONESHIP.rotation[2]);
      m.updateWorldMatrix(true, true);
      let box = new THREE.Box3().setFromObject(m);
      const size = box.getSize(new THREE.Vector3());
      m.scale.multiplyScalar(DRONESHIP.lengthM / Math.max(size.x, size.z, 1e-6)); // longest horizontal span -> lengthM
      m.updateWorldMatrix(true, true);
      box = new THREE.Box3().setFromObject(m);
      const c = box.getCenter(new THREE.Vector3());
      const H = box.max.y - box.min.y;                // total GLB height; the lowest ~H/4 is a sub-keel fitting
      m.position.x += -c.x; m.position.z += -c.z;     // center on the landing target (origin)
      // box.max.y is the model's highest point (here the rear crane top, not the deck), so seating it at
      // groundY rode high; sinking it sinkFrac·H drops the real deck toward the legs and the hull into the sea.
      // seaSinkFrac·H drops the waterline to match. Both are eyeballed to THIS droneship.glb — re-tune (and
      // DRONESHIP.yOffset) if a different barge is dropped in. Larger sinkFrac = deeper; too large submerges it.
      m.position.y += -box.max.y + groundY - DRONESHIP.sinkFrac * H + DRONESHIP.yOffset;
      if (water) water.position.y = waterY - DRONESHIP.seaSinkFrac * H; // sink the sea now that the hull depth is known
      m.traverse((o) => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; } });
      scene.add(m);
      procedural.visible = false;                      // hide the procedural barge
      if (DRONESHIP.credit) {
        const el = $("credit");
        if (el) { el.textContent = (el.textContent ? el.textContent + " · " : "") + DRONESHIP.credit; el.style.display = "block"; }
      }
      console.log("[viz] droneship model loaded:", DRONESHIP.url);
    },
    undefined,
    () => console.log(`[viz] no droneship model at ${DRONESHIP.url} — using the procedural barge`),
  );
}

function buildTrajectory(scene, frames) {
  const pts = frames.map((f) => new THREE.Vector3(f.p[0], f.p[1], f.p[2]));
  const full = new THREE.BufferGeometry().setFromPoints(pts);
  const ghost = new THREE.Line(full, new THREE.LineBasicMaterial({ color: 0x35507e, transparent: true, opacity: 0.4 }));
  scene.add(ghost);

  const flown = new THREE.BufferGeometry().setFromPoints(pts);
  const line = new THREE.Line(flown, new THREE.LineBasicMaterial({ color: 0x4ea3ff }));
  flown.setDrawRange(0, 1);
  scene.add(line);
  return { setProgress(i) { flown.setDrawRange(0, Math.max(2, i + 1)); } };
}

// ---------------------------------------------------------------- HUD
function grp(n) { return Math.round(n).toLocaleString("en-US").replace(/,/g, " "); } // thin-space thousands
function fmtMET(t) { // mission elapsed time, broadcast style: T+MM:SS (or T+H:MM:SS)
  const sgn = t < 0 ? "−" : "+"; t = Math.abs(t);
  const h = Math.floor(t / 3600), m = Math.floor((t % 3600) / 60), sec = Math.floor(t % 60);
  const pad = (x) => String(x).padStart(2, "0");
  return h > 0 ? `T${sgn}${h}:${pad(m)}:${pad(sec)}` : `T${sgn}${pad(m)}:${pad(sec)}`;
}
// Per-engine throttle for the plume rig / octaweb ring. frame.thr is the per-engine throttle_eff (one
// channel per engine): when it carries a channel per plume slot, each engine is driven from its own;
// a single channel drives the CENTER engine (index 0) only — how a booster lands on one center Merlin.
function engineThrottles(aThr, bThr, s, n) {
  const out = new Array(n).fill(0);
  const clamp01 = (x) => THREE.MathUtils.clamp(x, 0, 1);
  if (aThr.length >= n) { for (let e = 0; e < n; e++) out[e] = clamp01(lerpArr(aThr, bThr, s, e)); return out; }
  out[0] = clamp01(lerpArr(aThr, bThr, s, 0));
  return out;
}
// broadcast labels [primary, sublabel] for each descent phase (timeline markers)
const PHASE_MARK = { 0: ["Entry Burn", "Reentry"], 1: ["Aero Guidance", "Grid Fins"],
  2: ["Landing Burn", "Braking"], 3: ["Landing", "Touchdown"] };

function buildHUD(V) {
  // octaweb engine-status ring: 1 center + 8 around, mirroring the real Merlin layout (dots #oct0..#oct8)
  const svg = $("octSvg"), NS = "http://www.w3.org/2000/svg";
  const ring = document.createElementNS(NS, "circle");
  ring.setAttribute("cx", "50"); ring.setAttribute("cy", "50"); ring.setAttribute("r", "30");
  ring.setAttribute("class", "ring"); svg.appendChild(ring);
  for (let e = 0; e < 9; e++) {
    let cx = 50, cy = 50;
    if (e > 0) { const th = ((e - 1) / 8) * Math.PI * 2 - Math.PI / 2; cx = 50 + 30 * Math.cos(th); cy = 50 + 30 * Math.sin(th); }
    const d = document.createElementNS(NS, "circle");
    d.setAttribute("cx", cx.toFixed(2)); d.setAttribute("cy", cy.toFixed(2));
    d.setAttribute("r", e === 0 ? "5" : "4"); d.setAttribute("class", "eng"); d.id = "oct" + e;
    svg.appendChild(d);
  }
  const fb = $("finbars");
  for (let f = 0; f < V.n_fins; f++) {
    const d = document.createElement("div"); d.className = "finbar";
    d.innerHTML = `<div class="ftrack"><div class="fmid"></div><div class="ffill" id="ff${f}"></div></div><div class="flbl">${f + 1}</div>`;
    fb.appendChild(d);
  }
}

// mission timeline: a marker at each phase entry (derived from the replay) plus an END marker
function buildTimeline(frames, tEnd, phaseNames) {
  const host = $("tlMarkers");
  const segs = []; let last = null;
  for (const f of frames) { if (f.phase !== last) { segs.push({ t: f.t, phase: f.phase }); last = f.phase; } }
  const add = (t, lbl, sub) => {
    const frac = THREE.MathUtils.clamp(t / Math.max(tEnd, 1e-6), 0, 1);
    const m = document.createElement("div"); m.className = "tlMarker";
    m.style.left = (4 + frac * 92) + "%"; m.dataset.t = t; // 4%..96% track inset
    m.innerHTML = `<div class="mlbl">${lbl}</div>${sub ? `<div class="msub">${sub}</div>` : ""}<div class="tick"></div><div class="dot"></div>`;
    host.appendChild(m);
  };
  for (const sg of segs) { const mk = PHASE_MARK[sg.phase] || [phaseNames[sg.phase] || ("Phase " + sg.phase), ""]; add(sg.t, mk[0], mk[1]); }
  add(tEnd, "End", "");
}

function updateHUD(a, b, s, V, vel, totThr, engThr, frac, thrustN) {
  const lerp = (k) => a[k] + (b[k] - a[k]) * s;
  const alt = lerp("alt"), speed = lerp("speed"), mass = lerp("mass"), mach = lerp("mach"), t = lerp("t");

  // ---- engineering telemetry column ----
  $("vAlt").textContent = `${alt.toFixed(0)} m`;
  $("vSpd").textContent = `${speed.toFixed(1)} m/s`;
  $("vVert").textContent = `${(-vel.y).toFixed(1)} m/s`;
  $("vDr").textContent = `${Math.hypot(a.p[0], a.p[2]).toFixed(1)} m`;
  $("vMass").textContent = `${(mass / 1000).toFixed(2)} t`;
  $("vMach").textContent = mach.toFixed(2);
  // tilt: world-up vs body +X (nose). body +X in world = q_bv applied to (1,0,0).
  const q = new THREE.Quaternion().fromArray(a.q);
  const nose = new THREE.Vector3(1, 0, 0).applyQuaternion(q);
  $("vTilt").textContent = `${THREE.MathUtils.radToDeg(Math.acos(THREE.MathUtils.clamp(nose.y, -1, 1))).toFixed(1)} °`;
  $("vThrust").textContent = `${(thrustN / 1000).toFixed(0)} kN`; // the actual simulated thrust magnitude
  $("vTwr").textContent = (thrustN / (mass * 9.80665)).toFixed(2);

  // ---- phase tag + name ----
  const ph = a.phase, names = { 0: "Reentry Burn", 1: "Aero Descent", 2: "Powered Brake", 3: "Powered Descent" };
  const col = PHASE_COLOR[ph] || "#a0c8ff";
  const tag = $("phaseTag"); tag.textContent = names[ph] || ph; tag.style.color = col;
  $("phaseName").textContent = names[ph] || ph;

  // ---- big broadcast readout (km/h, km) ----
  $("bigSpeed").textContent = grp(speed * 3.6);
  if (alt >= 1000) { $("bigAlt").textContent = (alt / 1000).toFixed(1); $("bigAltUnit").textContent = "km"; }
  else { $("bigAlt").textContent = grp(alt); $("bigAltUnit").textContent = "m"; }
  $("metClock").textContent = fmtMET(t);

  // ---- octaweb engine status ----
  let lit = 0;
  for (let e = 0; e < 9; e++) {
    const d = $("oct" + e); if (!d) continue;
    const on = engThr[e] > 0.02; if (on) lit++;
    d.classList.toggle("lit", on);
    d.style.opacity = on ? (0.55 + 0.45 * Math.min(engThr[e] * 1.3, 1)).toFixed(2) : "1";
  }
  $("octThrottle").innerHTML = (lit ? Math.round(frac * 100) : 0) + "%<small>throttle</small>";

  // ---- grid-fin deflection bars ----
  for (let f = 0; f < V.n_fins; f++) {
    const d = lerpArr(a.fin, b.fin, s, f);
    const lim = V.fin_deflection_limit || 0.35;
    const frac = THREE.MathUtils.clamp(d / lim, -1, 1);
    const el = $("ff" + f);
    if (el) {
      const h = Math.abs(frac) * 15;
      el.style.height = h + "px";
      el.style.top = frac >= 0 ? `calc(50% - ${h}px)` : "50%";
      el.style.background = frac >= 0 ? "var(--sx-accent)" : "var(--sx-phase-1)";
    }
  }

  // ---- TVC gimbal pad ----
  const g0row = a.gim[0] || [0, 0], g1row = b.gim[0] || [0, 0];
  const gp = g0row[0] + (g1row[0] - g0row[0]) * s, gy = g0row[1] + (g1row[1] - g0row[1]) * s;
  const glim = V.tvc_deflection_limit || 0.14;
  $("gimbalDot").style.left = `calc(50% + ${THREE.MathUtils.clamp(gy / glim, -1, 1) * 26}px)`;
  $("gimbalDot").style.top = `calc(50% + ${THREE.MathUtils.clamp(gp / glim, -1, 1) * 26}px)`;

  // ---- mission timeline progress ----
  const tlFrac = THREE.MathUtils.clamp(t / Math.max(TOTAL_T, 1e-6), 0, 1);
  $("tlProgress").style.width = (tlFrac * 92) + "%";   // matches the 4%..96% track inset
  $("tlNow").style.left = (4 + tlFrac * 92) + "%";
  document.querySelectorAll("#tlMarkers .tlMarker").forEach((m) =>
    m.classList.toggle("on", t + 1e-6 >= parseFloat(m.dataset.t)));
  $("tlabel").textContent = `${t.toFixed(1)} / ${TOTAL_T.toFixed(1)} s`;
}

boot();

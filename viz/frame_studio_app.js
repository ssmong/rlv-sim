import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { TransformControls } from "three/addons/controls/TransformControls.js";
import { CSS2DObject, CSS2DRenderer } from "three/addons/renderers/CSS2DRenderer.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { buildFrameWorkbench } from "./frame_workbench.js";

const DEG = Math.PI / 180;
const ORIGIN = new THREE.Vector3();
const DEFAULT_VEHICLE_NORTH = -54;
const DEFAULT_VEHICLE_EAST = -18;
const DEFAULT_VEHICLE_DOWN = -34;
const DEFAULT_TARGET_DISPLAY = new THREE.Vector3(
  -DEFAULT_VEHICLE_NORTH,
  DEFAULT_VEHICLE_DOWN,
  -DEFAULT_VEHICLE_EAST
);
const DEFAULT_ORBIT_TARGET = DEFAULT_TARGET_DISPLAY.clone().multiplyScalar(0.5);
const DEFAULT_CAMERA = new THREE.Vector3(86, 56, 98).add(DEFAULT_ORBIT_TARGET);

function bodyAttitudeAlignedWithVelocity(gamma, course) {
  return { pitch: gamma, yaw: course };
}

const descentAttitude = bodyAttitudeAlignedWithVelocity(-62, 12);
const crosswindAttitude = bodyAttitudeAlignedWithVelocity(-48, -18);

const presets = {
  descent: { ...descentAttitude, roll: 0, speed: 420, gamma: -62, course: 12, wind: [6.2, 0, 16.9] },
  crosswind: { ...crosswindAttitude, roll: 35, speed: 360, gamma: -48, course: -18, wind: [-15, 0, 70.4] },
  vertical: { pitch: -90, yaw: 0, roll: 0, speed: 240, gamma: -90, course: 0, wind: [0, 0, 0] },
};

const flightFields = {
  pitch: document.getElementById("pitchInput"),
  yaw: document.getElementById("yawInput"),
  roll: document.getElementById("rollInput"),
  speed: document.getElementById("speedInput"),
  gamma: document.getElementById("gammaInput"),
  course: document.getElementById("courseInput"),
};
const positionFields = {
  vehicleNorth: document.getElementById("vehicleNorthInput"),
  vehicleEast: document.getElementById("vehicleEastInput"),
  vehicleDown: document.getElementById("vehicleDownInput"),
};
const fields = { ...flightFields, ...positionFields };
const alignBodyWithAirButton = document.getElementById("alignBodyWithAir");

const outputs = {
  pitch: document.getElementById("pitchValue"),
  yaw: document.getElementById("yawValue"),
  roll: document.getElementById("rollValue"),
  speed: document.getElementById("speedValue"),
  gamma: document.getElementById("gammaValue"),
  course: document.getElementById("courseValue"),
  vehicleNorth: document.getElementById("vehicleNorthValue"),
  vehicleEast: document.getElementById("vehicleEastValue"),
  vehicleDown: document.getElementById("vehicleDownValue"),
};
const airComponentInputs = {
  u: document.getElementById("airUValue"),
  v: document.getElementById("airVValue"),
  w: document.getElementById("airWValue"),
};
const windComponentInputs = {
  north: document.getElementById("windNorthValue"),
  up: document.getElementById("windUpValue"),
  east: document.getElementById("windEastValue"),
};

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(0x071015, 0);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
document.getElementById("viewport").append(renderer.domElement);

const labelRenderer = new CSS2DRenderer();
labelRenderer.setSize(window.innerWidth, window.innerHeight);
labelRenderer.domElement.className = "labelLayer";
document.getElementById("viewport").append(labelRenderer.domElement);

const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x091318, 0.0032);

const camera = new THREE.PerspectiveCamera(38, window.innerWidth / window.innerHeight, 0.1, 600);
camera.position.copy(DEFAULT_CAMERA);

const orbit = new OrbitControls(camera, renderer.domElement);
orbit.target.copy(DEFAULT_ORBIT_TARGET);
orbit.enableDamping = true;
orbit.dampingFactor = 0.07;
orbit.minDistance = 45;
orbit.maxDistance = 210;
orbit.maxPolarAngle = Math.PI * 0.93;
orbit.update();

scene.add(new THREE.HemisphereLight(0xd8f5ff, 0x152029, 1.7));
const keyLight = new THREE.DirectionalLight(0xffffff, 3.5);
keyLight.position.set(42, 75, 60);
keyLight.castShadow = true;
keyLight.shadow.mapSize.set(2048, 2048);
keyLight.shadow.camera.near = 5;
keyLight.shadow.camera.far = 240;
keyLight.shadow.camera.left = -105;
keyLight.shadow.camera.right = 105;
keyLight.shadow.camera.top = 105;
keyLight.shadow.camera.bottom = -105;
keyLight.shadow.bias = -0.0002;
keyLight.shadow.normalBias = 0.025;
scene.add(keyLight);
const rimLight = new THREE.DirectionalLight(0x69dcd8, 2.1);
rimLight.position.set(-55, 15, -40);
scene.add(rimLight);

const ground = new THREE.Mesh(
  new THREE.CircleGeometry(260, 128),
  new THREE.MeshStandardMaterial({ color: 0x272b30, roughness: 0.98, metalness: 0.015 })
);
ground.name = "local tangent ground";
ground.rotation.x = -Math.PI / 2;
ground.position.y = DEFAULT_VEHICLE_DOWN - 0.08;
ground.receiveShadow = true;
scene.add(ground);

function addGroundGrid(size, divisions, centerColor, lineColor, opacity, height) {
  const grid = new THREE.GridHelper(size, divisions, centerColor, lineColor);
  grid.position.y = height;
  grid.material.transparent = true;
  grid.material.opacity = opacity;
  grid.material.depthWrite = false;
  scene.add(grid);
  return grid;
}

const fineGroundGrid = addGroundGrid(300, 60, 0x747b81, 0x454b50, 0.2, DEFAULT_VEHICLE_DOWN + 0.01);
const coarseGroundGrid = addGroundGrid(300, 15, 0x9aa1a6, 0x62696f, 0.3, DEFAULT_VEHICLE_DOWN + 0.025);

const groundOrigin = new THREE.Mesh(
  new THREE.RingGeometry(5.4, 5.62, 64),
  new THREE.MeshBasicMaterial({
    color: 0x83aaa9,
    transparent: true,
    opacity: 0.2,
    side: THREE.DoubleSide,
    depthWrite: false,
  })
);
groundOrigin.name = "vehicle ground reference";
groundOrigin.rotation.x = -Math.PI / 2;
groundOrigin.position.y = DEFAULT_VEHICLE_DOWN + 0.04;
scene.add(groundOrigin);

const originHalo = new THREE.Mesh(
  new THREE.RingGeometry(4.8, 5.05, 64),
  new THREE.MeshBasicMaterial({ color: 0x65e6e1, transparent: true, opacity: 0.13, side: THREE.DoubleSide, depthWrite: false })
);
originHalo.rotation.x = -Math.PI / 2;
scene.add(originHalo);

const spacecraft = new THREE.Group();
spacecraft.name = "spacecraft attitude visual";
const falcon9Root = new THREE.Group();
falcon9Root.name = "Falcon 9 GLB";
const vehicleHitProxy = new THREE.Mesh(
  new THREE.BoxGeometry(1, 1, 1),
  new THREE.MeshBasicMaterial({ transparent: true, opacity: 0, depthWrite: false, colorWrite: false })
);
vehicleHitProxy.name = "vehicle drag hit area";
vehicleHitProxy.scale.set(48, 8, 8);
spacecraft.add(vehicleHitProxy, falcon9Root);
scene.add(spacecraft);

const gimbalDock = document.getElementById("attitudeGimbalDock");
const gimbalViewport = document.getElementById("attitudeGimbalViewport");
const GIMBAL_VIEW_SIZE = 216;
const gimbalRenderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
gimbalRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
gimbalRenderer.setSize(GIMBAL_VIEW_SIZE, GIMBAL_VIEW_SIZE, false);
gimbalRenderer.setClearColor(0x000000, 0);
gimbalRenderer.outputColorSpace = THREE.SRGBColorSpace;
gimbalRenderer.domElement.setAttribute("aria-label", "Local-axis attitude rotation gimbal");
gimbalRenderer.domElement.style.cursor = "grab";
gimbalViewport.append(gimbalRenderer.domElement);

const gimbalScene = new THREE.Scene();
const gimbalCamera = new THREE.PerspectiveCamera(34, 1, 0.1, 50);
gimbalCamera.position.set(3.4, 2.55, 4.7);
gimbalCamera.lookAt(ORIGIN);
const gimbalProxy = new THREE.Group();
gimbalProxy.name = "attitude gimbal proxy";
gimbalScene.add(gimbalProxy);
const gimbalBodyMarker = new THREE.Mesh(
  new THREE.ConeGeometry(0.27, 1.08, 4),
  new THREE.MeshBasicMaterial({ color: 0xd8e5e8, transparent: true, opacity: 0.72, wireframe: true })
);
gimbalBodyMarker.rotation.z = -Math.PI / 2;
gimbalBodyMarker.position.x = 0.3;
gimbalProxy.add(gimbalBodyMarker);

const attitudeGimbal = new TransformControls(gimbalCamera, gimbalRenderer.domElement);
attitudeGimbal.name = "docked spacecraft attitude rotation gimbal";
attitudeGimbal.setMode("rotate");
attitudeGimbal.setSpace("local");
attitudeGimbal.setSize(2.1);
attitudeGimbal.attach(gimbalProxy);
gimbalScene.add(attitudeGimbal);
let attitudeControlMode = "direct";
let gimbalDragging = false;
const vehicleModel = {
  status: "loading",
  label: "Falcon 9",
  sourceLabel: "Falcon 9 by Forest Katsch",
  sourceUrl: "https://sketchfab.com/3d-models/spacex-falcon-9-block-45-12285ccc02de4d5c91b00632f0722c92",
  licenseLabel: "CC BY-NC 4.0",
  licenseUrl: "https://creativecommons.org/licenses/by-nc/4.0/",
};

function updateModelCredit() {
  const credit = document.getElementById("modelCredit");
  if (vehicleModel.status !== "ready") {
    credit.textContent = vehicleModel.status === "loading"
      ? `Loading ${vehicleModel.label} GLB…`
      : `${vehicleModel.label} GLB unavailable`;
    return;
  }
  credit.replaceChildren();
  const source = document.createElement("a");
  source.href = vehicleModel.sourceUrl;
  source.target = "_blank";
  source.rel = "noreferrer";
  source.textContent = vehicleModel.sourceLabel;
  const license = document.createElement("a");
  license.href = vehicleModel.licenseUrl;
  license.target = "_blank";
  license.rel = "noreferrer";
  license.textContent = vehicleModel.licenseLabel;
  credit.append(source, " · ", license);
}

function syncVehicleVisuals() {
  falcon9Root.visible = vehicleModel.status === "ready";
  updateModelCredit();
}

syncVehicleVisuals();

function visibleModelBounds(model) {
  model.updateWorldMatrix(true, true);
  const bounds = new THREE.Box3();
  const meshBounds = new THREE.Box3();
  model.traverse((object) => {
    if (!object.isMesh) return;
    for (let node = object; node; node = node.parent) {
      if (!node.visible) return;
      if (node === model) break;
    }
    if (!object.geometry.boundingBox) object.geometry.computeBoundingBox();
    meshBounds.copy(object.geometry.boundingBox).applyMatrix4(object.matrixWorld);
    bounds.union(meshBounds);
  });
  return bounds;
}

function loadVehicleModel(url, configure) {
  new GLTFLoader().load(url, (gltf) => {
    const model = gltf.scene;
    model.name = `${vehicleModel.label} normalized to body frame`;
    configure(model);
    model.traverse((object) => {
      if (object.isMesh) {
        object.castShadow = true;
        object.receiveShadow = true;
        const materials = Array.isArray(object.material) ? object.material : [object.material];
        for (const material of materials) {
          if (material) material.envMapIntensity = 0.75;
        }
      }
    });
    const initialBounds = visibleModelBounds(model);
    const initialSize = initialBounds.getSize(new THREE.Vector3());
    if (!(initialSize.x > 1e-9)) {
      vehicleModel.status = "error";
      syncVehicleVisuals();
      return;
    }
    model.scale.setScalar(42 / initialSize.x);
    const fittedBounds = visibleModelBounds(model);
    const fittedCenter = fittedBounds.getCenter(new THREE.Vector3());
    model.position.sub(fittedCenter);
    falcon9Root.add(model);
    vehicleModel.status = "ready";
    syncVehicleVisuals();
  }, undefined, () => {
    vehicleModel.status = "error";
    syncVehicleVisuals();
  });
}

loadVehicleModel("./resources/falcon9.glb", (model) => {
  const excludedParts = ["m1dvac", "fairing", "stagetwo", "sun"];
  model.traverse((object) => {
    const normalizedName = (object.name || "").toLowerCase().replace(/[^a-z0-9]/g, "");
    if (excludedParts.some((part) => normalizedName.includes(part))) object.visible = false;
  });
  model.rotation.z = -Math.PI / 2;
});

const vectorHandles = [];

function makeVectorVisual(id, name, color, className, scale, maxMagnitude, ghostDirection) {
  const arrow = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), ORIGIN, 1, color, 3.4, 1.55);
  arrow.line.material.transparent = true;
  arrow.line.material.opacity = 0.74;
  arrow.cone.material.transparent = true;
  arrow.cone.material.opacity = 0.9;
  arrow.line.material.depthTest = false;
  arrow.cone.material.depthTest = false;
  arrow.renderOrder = 50;
  scene.add(arrow);

  const shaft = new THREE.Mesh(
    new THREE.CylinderGeometry(0.24, 0.24, 1, 12),
    new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.72, depthTest: false })
  );
  shaft.renderOrder = 49;
  scene.add(shaft);

  const element = document.createElement("div");
  element.className = `frame-label vectorLabel ${className}`;
  element.textContent = name;
  const label = new CSS2DObject(element);
  scene.add(label);

  const handle = new THREE.Group();
  handle.name = `${name} vector handle`;
  const knob = new THREE.Mesh(
    new THREE.SphereGeometry(1.05, 20, 12),
    new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.96, depthTest: false })
  );
  knob.renderOrder = 64;
  const ring = new THREE.Mesh(
    new THREE.TorusGeometry(1.6, 0.1, 8, 32),
    new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.72, depthTest: false })
  );
  ring.renderOrder = 63;
  const hitTarget = new THREE.Mesh(
    new THREE.SphereGeometry(2.55, 12, 8),
    new THREE.MeshBasicMaterial({ transparent: true, opacity: 0, depthWrite: false, depthTest: false })
  );
  hitTarget.userData.vectorId = id;
  handle.add(knob, ring, hitTarget);
  scene.add(handle);

  const visual = {
    id, arrow, shaft, label, element, handle, knob, ring, hitTarget, scale, maxMagnitude,
    ghostDirection: ghostDirection.clone().normalize(), currentVector: new THREE.Vector3(),
  };
  vectorHandles.push(visual);
  return visual;
}

const velocityVisual = makeVectorVisual(
  "velocity", "V_G", 0x6f9dff, "velocity", 0.06, 900, new THREE.Vector3(1, -0.35, 0)
);
const windVisual = makeVectorVisual(
  "wind", "V_W", 0x65e6e1, "wind", 0.28, 150, new THREE.Vector3(1, 0, 0)
);

const bankRoot = new THREE.Group();
bankRoot.name = "lift-vector bank cue";
scene.add(bankRoot);
const bankArc = new THREE.Line(
  new THREE.BufferGeometry(),
  new THREE.LineBasicMaterial({ color: 0xffc66d, transparent: true, opacity: 0.9, depthTest: false })
);
bankArc.renderOrder = 56;
bankRoot.add(bankArc);
const bankRays = new THREE.LineSegments(
  new THREE.BufferGeometry(),
  new THREE.LineBasicMaterial({ color: 0xffc66d, transparent: true, opacity: 0.3, depthTest: false })
);
bankRays.renderOrder = 55;
bankRoot.add(bankRays);
const bankLabelElement = document.createElement("div");
bankLabelElement.className = "frame-label bankLabel";
const bankLabel = new CSS2DObject(bankLabelElement);
bankRoot.add(bankLabel);
const liftDirectionArrow = new THREE.ArrowHelper(
  new THREE.Vector3(0, 1, 0), ORIGIN, 10.5, 0xffc66d, 1.8, 0.85
);
for (const material of [liftDirectionArrow.line.material, liftDirectionArrow.cone.material]) {
  material.transparent = true;
  material.opacity = 0.88;
  material.depthTest = false;
}
liftDirectionArrow.renderOrder = 57;
bankRoot.add(liftDirectionArrow);
const liftDirectionLabelElement = document.createElement("div");
liftDirectionLabelElement.className = "frame-label liftVectorLabel";
liftDirectionLabelElement.innerHTML = "L ∥ −z<sub>W</sub>";
const liftDirectionLabel = new CSS2DObject(liftDirectionLabelElement);
bankRoot.add(liftDirectionLabel);

const aeroGeometryRoot = new THREE.Group();
aeroGeometryRoot.name = "air-relative alpha beta geometry";
scene.add(aeroGeometryRoot);
const aeroLongitudinalPlane = new THREE.Mesh(
  new THREE.PlaneGeometry(1, 1),
  new THREE.MeshBasicMaterial({
    color: 0xc7a0ff,
    transparent: true,
    opacity: 0.045,
    side: THREE.DoubleSide,
    depthWrite: false,
    depthTest: false,
  })
);
aeroLongitudinalPlane.name = "aerodynamic x-z plane";
aeroLongitudinalPlane.renderOrder = 37;
aeroGeometryRoot.add(aeroLongitudinalPlane);

function makeAeroCueArrow(labelText, color, opacity, labelClass = "") {
  const helper = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), ORIGIN, 1, color, 2, 0.9);
  for (const material of [helper.line.material, helper.cone.material]) {
    material.transparent = true;
    material.opacity = opacity;
    material.depthTest = false;
  }
  helper.renderOrder = 58;
  const element = document.createElement("div");
  element.className = `frame-label aeroVectorLabel ${labelClass}`.trim();
  element.innerHTML = labelText;
  const label = new CSS2DObject(element);
  aeroGeometryRoot.add(helper, label);
  return { helper, label, element };
}

function setAeroCueArrow(cue, direction, length, visible, labelScale = 1.12) {
  cue.helper.visible = visible;
  cue.label.visible = visible;
  if (!visible) return;
  const unitDirection = direction.clone().normalize();
  cue.helper.position.copy(ORIGIN);
  cue.helper.setDirection(unitDirection);
  cue.helper.setLength(length, Math.min(2.2, length * 0.22), Math.min(1.0, length * 0.1));
  cue.label.position.copy(unitDirection).multiplyScalar(length * labelScale);
}

function makeAeroAngleCue(className, color) {
  const line = new THREE.Line(
    new THREE.BufferGeometry(),
    new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.92, depthTest: false })
  );
  line.renderOrder = 60;
  const element = document.createElement("div");
  element.className = `frame-label aeroAngleLabel ${className}`;
  const label = new CSS2DObject(element);
  aeroGeometryRoot.add(line, label);
  return { line, label, element };
}

function setAeroAngleCue(cue, start, axis, angleDegrees, radius, symbol, visible) {
  const available = visible && Number.isFinite(angleDegrees) && start.lengthSq() > 1e-9 && axis.lengthSq() > 1e-9;
  cue.line.visible = available && Math.abs(angleDegrees) > 0.05;
  cue.label.visible = available;
  if (!available) return;
  const direction = start.clone().normalize();
  const rotationAxis = axis.clone().normalize();
  const angle = angleDegrees * DEG;
  const rotation = new THREE.Quaternion();
  const steps = Math.max(8, Math.ceil(Math.abs(angle) / (Math.PI / 36)));
  const points = [];
  for (let index = 0; index <= steps; index++) {
    rotation.setFromAxisAngle(rotationAxis, angle * (index / steps));
    points.push(direction.clone().applyQuaternion(rotation).multiplyScalar(radius));
  }
  cue.line.geometry.dispose();
  cue.line.geometry = new THREE.BufferGeometry().setFromPoints(points);
  rotation.setFromAxisAngle(rotationAxis, angle * 0.5);
  cue.label.position.copy(direction).applyQuaternion(rotation).multiplyScalar(radius * 1.18)
    .addScaledVector(rotationAxis, 0.35);
  cue.element.textContent = `${symbol}  ${formatAngleDegrees(angleDegrees)}`;
}

const aeroAxes = [
  makeAeroCueArrow("x<sub>B</sub>", 0xff6b78, 0.34, "aeroAxisLabel"),
  makeAeroCueArrow("y<sub>B</sub>", 0x70dfa2, 0.34, "aeroAxisLabel"),
  makeAeroCueArrow("z<sub>B</sub>", 0x6f9dff, 0.34, "aeroAxisLabel"),
];
const airRelativeCue = makeAeroCueArrow("V<sub>A</sub>", 0xc7a0ff, 0.9, "airRelativeCue");
const airProjectionCue = makeAeroCueArrow("proj on x<sub>B</sub>–z<sub>B</sub>(V<sub>A</sub>)", 0xffc66d, 0.7, "airProjectionCue");
const airSideResidual = new THREE.Line(
  new THREE.BufferGeometry(),
  new THREE.LineDashedMaterial({ color: 0xc7a0ff, dashSize: 0.75, gapSize: 0.5, transparent: true, opacity: 0.58, depthTest: false })
);
airSideResidual.renderOrder = 59;
aeroGeometryRoot.add(airSideResidual);
const alphaCue = makeAeroAngleCue("alphaCue", 0xffc66d);
const betaCue = makeAeroAngleCue("betaCue", 0xc7a0ff);

function updateAeroGeometry(airWorld, airBody, attitude, aero) {
  const enabled = document.getElementById("showAeroGeometryInput").checked;
  aeroGeometryRoot.visible = true;
  if (!enabled) {
    aeroLongitudinalPlane.visible = false;
    [...aeroAxes, airRelativeCue, airProjectionCue].forEach((cue) => {
      cue.helper.visible = false;
      cue.label.visible = false;
    });
    airSideResidual.visible = false;
    for (const cue of [alphaCue, betaCue]) {
      cue.line.visible = false;
      cue.label.visible = false;
    }
    return;
  }
  aeroLongitudinalPlane.visible = true;

  const xLocal = new THREE.Vector3(1, 0, 0);
  const yLocal = new THREE.Vector3(0, 1, 0);
  const zLocal = new THREE.Vector3(0, 0, 1);
  const xWorld = xLocal.clone().applyQuaternion(attitude);
  const yWorld = yLocal.clone().applyQuaternion(attitude);
  const zWorld = zLocal.clone().applyQuaternion(attitude);
  const axisLength = 10.5;
  [xWorld, yWorld, zWorld].forEach((axis, index) => setAeroCueArrow(aeroAxes[index], axis, axisLength, true, 1.08));
  ["x<sub>B</sub> / u<sub>B</sub>", "y<sub>B</sub> / v<sub>B</sub>", "z<sub>B</sub> / w<sub>B</sub>"]
    .forEach((text, index) => { aeroAxes[index].element.innerHTML = text; });
  const planeNormal = new THREE.Vector3().crossVectors(xWorld, zWorld).normalize();
  aeroLongitudinalPlane.quaternion.setFromRotationMatrix(
    new THREE.Matrix4().makeBasis(xWorld, zWorld, planeNormal)
  );
  aeroLongitudinalPlane.scale.set(20, 15, 1);

  const speed = airWorld.length();
  const longitudinal = Math.hypot(aero.components.u, aero.components.w);
  const airAvailable = speed > 1e-6;
  const projectionAvailable = longitudinal > 1e-6;
  const visualLength = 17;
  const projectedLocal = xLocal.clone().multiplyScalar(aero.components.u)
    .addScaledVector(zLocal, aero.components.w);
  const projectedWorld = projectedLocal.applyQuaternion(attitude);
  setAeroCueArrow(airRelativeCue, airWorld, visualLength, airAvailable);
  setAeroCueArrow(
    airProjectionCue,
    projectedWorld,
    airAvailable ? visualLength * longitudinal / speed : 0,
    airAvailable && projectionAvailable
  );

  airSideResidual.visible = airAvailable && projectionAvailable && Math.abs(aero.components.v) > 1e-6;
  if (airSideResidual.visible) {
    const scale = visualLength / speed;
    airSideResidual.geometry.dispose();
    airSideResidual.geometry = new THREE.BufferGeometry().setFromPoints([
      projectedWorld.clone().multiplyScalar(scale),
      airWorld.clone().multiplyScalar(scale),
    ]);
    airSideResidual.computeLineDistances();
  }

  const projectionDirection = projectionAvailable ? projectedWorld.clone().normalize() : xWorld;
  const betaAxis = new THREE.Vector3().crossVectors(projectionDirection, yWorld);
  setAeroAngleCue(alphaCue, xWorld, yWorld.clone().negate(), aero.alpha, 6.1, "α", projectionAvailable);
  setAeroAngleCue(betaCue, projectionDirection, betaAxis, aero.beta, 8.3, "β", projectionAvailable && airAvailable);
}

function unavailableBank(reason) {
  bankRoot.visible = false;
  bankLabel.visible = false;
  liftDirectionLabel.visible = false;
  return { value: null, reason };
}

function updateLiftBankCue(airVelocity, attitude, aero) {
  if (airVelocity.length() < 1e-6) return unavailableBank("Undefined at zero airspeed.");
  if (!Number.isFinite(aero.alpha)) {
    return unavailableBank("Undefined when air-relative velocity is purely lateral.");
  }

  const windDirection = airVelocity.clone().normalize();
  const referenceLift = new THREE.Vector3(0, 1, 0)
    .addScaledVector(windDirection, -windDirection.y);
  if (referenceLift.lengthSq() < 1e-12) {
    return unavailableBank("Undefined for vertical air-relative velocity: local-up projection vanishes.");
  }
  referenceLift.normalize();
  const alpha = aero.alpha * DEG;
  const bodyForward = new THREE.Vector3(1, 0, 0).applyQuaternion(attitude);
  const bodyDown = new THREE.Vector3(0, 0, 1).applyQuaternion(attitude);
  const liftDirection = bodyForward.multiplyScalar(Math.sin(alpha))
    .addScaledVector(bodyDown, -Math.cos(alpha));
  liftDirection.addScaledVector(windDirection, -liftDirection.dot(windDirection));
  if (liftDirection.lengthSq() < 1e-12) {
    return unavailableBank("Lift direction is degenerate in this state.");
  }
  liftDirection.normalize();
  const bank = Math.atan2(
    windDirection.dot(new THREE.Vector3().crossVectors(referenceLift, liftDirection)),
    referenceLift.dot(liftDirection)
  );
  const radius = 8.4;
  const steps = Math.max(8, Math.ceil(Math.abs(bank) / (Math.PI / 36)));
  const rotation = new THREE.Quaternion();
  const points = [];
  for (let index = 0; index <= steps; index++) {
    rotation.setFromAxisAngle(windDirection, bank * (index / steps));
    points.push(referenceLift.clone().applyQuaternion(rotation).multiplyScalar(radius));
  }
  bankArc.geometry.dispose();
  bankArc.geometry = new THREE.BufferGeometry().setFromPoints(points);
  bankRays.geometry.dispose();
  bankRays.geometry = new THREE.BufferGeometry().setFromPoints([
    ORIGIN, referenceLift.clone().multiplyScalar(radius),
    ORIGIN, liftDirection.clone().multiplyScalar(radius),
  ]);
  rotation.setFromAxisAngle(windDirection, bank * 0.5);
  bankLabel.position
    .copy(referenceLift)
    .applyQuaternion(rotation)
    .multiplyScalar(radius * 1.18)
    .addScaledVector(windDirection, -0.9);
  bankLabelElement.innerHTML = `σ<sub>L</sub> ${(bank / DEG).toFixed(1)}°`;
  liftDirectionArrow.position.copy(ORIGIN);
  liftDirectionArrow.setDirection(liftDirection);
  liftDirectionArrow.setLength(radius * 1.28, 1.8, 0.85);
  liftDirectionLabel.position
    .copy(liftDirection)
    .multiplyScalar(radius * 1.5)
    .addScaledVector(windDirection, 0.9);
  bankRoot.visible = true;
  bankLabel.visible = true;
  liftDirectionLabel.visible = true;
  return {
    value: bank / DEG,
    reason: "Signed rotation from zero-bank lift-up to −z_W about air-relative +x_W.",
  };
}

function aerodynamicAngles(airBody) {
  const speed = airBody.length();
  const components = { u: airBody.x, v: airBody.y, w: airBody.z };
  const convention = "FRD body axes: u_B along +x_B, v_B along +y_B, w_B along +z_B.";
  if (speed < 1e-6) {
    return {
      alpha: null,
      beta: null,
      components,
      alphaReason: "Undefined at zero airspeed.",
      betaReason: "Undefined at zero airspeed.",
      convention,
    };
  }
  const longitudinal = Math.hypot(components.u, components.w);
  return {
    alpha: longitudinal > 1e-6 ? THREE.MathUtils.radToDeg(Math.atan2(components.w, components.u)) : null,
    beta: THREE.MathUtils.radToDeg(Math.atan2(components.v, longitudinal)),
    components,
    alphaReason: longitudinal > 1e-6 ? convention : "Undefined when air-relative velocity is purely lateral.",
    betaReason: convention,
    convention,
  };
}

function formatAngleDegrees(value) {
  const display = Math.abs(value) < 0.05 ? 0 : value;
  return `${display.toFixed(1)}°`;
}

function updateVectorVisual(visual, vector, text) {
  const visible = vector.lengthSq() > 1e-9;
  visual.arrow.visible = visible;
  visual.shaft.visible = visible;
  visual.handle.visible = true;
  visual.label.visible = true;
  visual.currentVector.copy(vector);
  const endpoint = visible
    ? vector.clone().multiplyScalar(visual.scale)
    : visual.ghostDirection.clone().multiplyScalar(4.5);
  const length = endpoint.length();
  const direction = endpoint.clone().normalize();
  if (visible) {
    const headLength = Math.min(4.2, length * 0.22);
    const headWidth = Math.min(2.1, length * 0.11);
    visual.arrow.position.copy(ORIGIN);
    visual.arrow.setDirection(direction);
    visual.arrow.setLength(length, headLength, headWidth);
    const shaftLength = Math.max(length - headLength, 0.05);
    visual.shaft.position.copy(direction).multiplyScalar(shaftLength * 0.5);
    visual.shaft.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction);
    visual.shaft.scale.set(1, shaftLength, 1);
  }
  visual.handle.position.copy(endpoint);
  visual.label.position.copy(direction).multiplyScalar(length + 2.2);
  visual.element.innerHTML = visible
    ? text
    : `${visual.id === "wind" ? "V<sub>W</sub>" : "V<sub>G</sub>"} 0 · drag`;
}

const targetRoot = new THREE.Group();
scene.add(targetRoot);
const targetLine = new THREE.Line(
  new THREE.BufferGeometry(),
  new THREE.LineDashedMaterial({ color: 0xff7588, dashSize: 2.2, gapSize: 1.35, transparent: true, opacity: 0.6, depthTest: false })
);
targetLine.renderOrder = 48;
targetRoot.add(targetLine);
const targetMarker = new THREE.Mesh(
  new THREE.OctahedronGeometry(1.15, 0),
  new THREE.MeshBasicMaterial({ color: 0xff7588, transparent: true, opacity: 0.8, depthTest: false })
);
targetMarker.renderOrder = 61;
targetRoot.add(targetMarker);
const targetPad = new THREE.Mesh(
  new THREE.CircleGeometry(5.2, 64),
  new THREE.MeshBasicMaterial({
    color: 0xff7588,
    transparent: true,
    opacity: 0.065,
    side: THREE.DoubleSide,
    depthWrite: false,
  })
);
targetPad.rotation.x = -Math.PI / 2;
targetRoot.add(targetPad);
const targetRing = new THREE.Mesh(
  new THREE.RingGeometry(4.85, 5.2, 64),
  new THREE.MeshBasicMaterial({
    color: 0xff7588,
    transparent: true,
    opacity: 0.58,
    side: THREE.DoubleSide,
    depthWrite: false,
  })
);
targetRing.rotation.x = -Math.PI / 2;
targetRoot.add(targetRing);
const targetLabelElement = document.createElement("div");
targetLabelElement.className = "frame-label targetLabel";
targetLabelElement.textContent = "NED origin · LOS target";
const targetLabel = new CSS2DObject(targetLabelElement);
targetRoot.add(targetLabel);

const meta = {
  target: DEFAULT_TARGET_DISPLAY.toArray(),
};

const workbench = buildFrameWorkbench(scene, camera, renderer.domElement, { length: 44, diameter: 5.4 }, meta, {
  active: true,
  visibleFrames: ["velocity"],
  safeInsets: { left: 372, right: 430, top: 82, bottom: 72 },
});

function velocityFromState(state) {
  const gamma = state.gamma * DEG;
  const course = state.course * DEG;
  return new THREE.Vector3(
    Math.cos(gamma) * Math.cos(course),
    Math.sin(gamma),
    Math.cos(gamma) * Math.sin(course)
  ).multiplyScalar(state.speed);
}

function attitudeFromState(state) {
  const pitch = state.pitch * DEG;
  const yaw = state.yaw * DEG;
  const up = new THREE.Vector3(0, 1, 0);
  const heading = new THREE.Vector3(Math.cos(yaw), 0, Math.sin(yaw));
  const nose = heading.clone().multiplyScalar(Math.cos(pitch)).addScaledVector(up, Math.sin(pitch)).normalize();
  let right = new THREE.Vector3().crossVectors(heading, up).normalize();
  const down = new THREE.Vector3().crossVectors(nose, right).normalize();
  right = new THREE.Vector3().crossVectors(down, nose).normalize();
  const attitude = new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(nose, right, down));
  return attitude.multiply(
    new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), state.roll * DEG)
  ).normalize();
}

function syncVelocityFields(vector) {
  const speed = vector.length();
  fields.speed.value = String(Math.min(Number(fields.speed.max), speed));
  if (speed > 1e-9) {
    const horizontal = Math.hypot(vector.x, vector.z);
    fields.gamma.value = String(THREE.MathUtils.radToDeg(Math.atan2(vector.y, horizontal)));
    fields.course.value = String(horizontal > 1e-9
      ? THREE.MathUtils.radToDeg(Math.atan2(vector.z, vector.x))
      : 0);
  }
}

let interactiveVelocity = velocityFromState(presets.descent);
let interactiveWind = new THREE.Vector3().fromArray(presets.descent.wind);

function currentState() {
  return Object.fromEntries(Object.entries(fields).map(([key, input]) => [key, Number(input.value)]));
}

function updateSliderFill(input) {
  const range = Number(input.max) - Number(input.min);
  const fill = range > 0 ? ((Number(input.value) - Number(input.min)) / range) * 100 : 0;
  input.style.setProperty("--fill", `${fill}%`);
}

function setPreset(name) {
  const values = presets[name];
  for (const [key, input] of Object.entries(flightFields)) {
    input.value = String(values[key]);
  }
  interactiveVelocity = velocityFromState(values);
  interactiveWind.fromArray(values.wind);
  document.querySelectorAll("#statePresets button").forEach((button) => button.classList.toggle("active", button.dataset.preset === name));
  updateScene();
}

function alignBodyWithAirVelocity() {
  const airVelocity = interactiveVelocity.clone().sub(interactiveWind);
  if (airVelocity.lengthSq() < 1e-12) return;
  const horizontal = Math.hypot(airVelocity.x, airVelocity.z);
  flightFields.pitch.value = String(THREE.MathUtils.radToDeg(Math.atan2(airVelocity.y, horizontal)));
  if (horizontal > 1e-9) {
    flightFields.yaw.value = String(THREE.MathUtils.radToDeg(Math.atan2(airVelocity.z, airVelocity.x)));
  }
  clearPresetSelection();
  updateScene();
}

function updateScene() {
  const state = currentState();
  const attitude = attitudeFromState(state);
  spacecraft.quaternion.copy(attitude);
  if (!gimbalDragging) gimbalProxy.quaternion.copy(attitude);

  const velocity = interactiveVelocity.clone();
  const wind = interactiveWind.clone();
  const airWorld = velocity.clone().sub(wind);
  alignBodyWithAirButton.disabled = airWorld.lengthSq() < 1e-12;
  alignBodyWithAirButton.title = alignBodyWithAirButton.disabled
    ? "Unavailable because air-relative velocity is zero."
    : "Set x_B parallel to V_A while preserving V_G, V_W, and body roll.";
  const airBody = airWorld.clone().applyQuaternion(attitude.clone().invert());
  const aero = aerodynamicAngles(airBody);
  const bank = updateLiftBankCue(airWorld, attitude, aero);
  updateAeroGeometry(airWorld, airBody, attitude, aero);

  const groundY = state.vehicleDown;
  const target = new THREE.Vector3(-state.vehicleNorth, groundY, -state.vehicleEast);
  target.toArray(meta.target);
  ground.position.y = groundY - 0.08;
  fineGroundGrid.position.y = groundY + 0.01;
  coarseGroundGrid.position.y = groundY + 0.025;
  groundOrigin.position.y = groundY + 0.04;
  targetLine.geometry.dispose();
  targetLine.geometry = new THREE.BufferGeometry().setFromPoints([ORIGIN, target]);
  targetLine.computeLineDistances();
  targetMarker.position.copy(target).add(new THREE.Vector3(0, 1.25, 0));
  targetPad.position.copy(target).setY(groundY + 0.035);
  targetRing.position.copy(target).setY(groundY + 0.055);
  targetLabel.position.copy(target).add(new THREE.Vector3(0, 3.2, 0));
  targetRoot.visible = document.getElementById("showTargetInput").checked;
  targetLabel.visible = targetRoot.visible;

  updateVectorVisual(velocityVisual, velocity, `V<sub>G</sub> ${velocity.length().toFixed(0)} m/s`);
  updateVectorVisual(windVisual, wind, `V<sub>W</sub> ${wind.length().toFixed(1)} m/s`);

  workbench.update({ position: ORIGIN, velocity, quaternion: attitude, airBody });

  outputs.pitch.value = state.pitch.toFixed(1);
  outputs.yaw.value = state.yaw.toFixed(1);
  outputs.roll.value = state.roll.toFixed(1);
  outputs.speed.value = velocity.length().toFixed(1);
  outputs.gamma.value = Number(fields.gamma.value).toFixed(1);
  outputs.course.value = Number(fields.course.value).toFixed(1);
  outputs.vehicleNorth.value = state.vehicleNorth.toFixed(0);
  outputs.vehicleEast.value = state.vehicleEast.toFixed(0);
  outputs.vehicleDown.value = state.vehicleDown.toFixed(0);
  const bankValue = document.getElementById("bankValue");
  bankValue.textContent = bank.value === null ? "—" : formatAngleDegrees(bank.value);
  bankValue.title = bank.reason;
  const alphaValue = document.getElementById("alphaValue");
  const betaValue = document.getElementById("betaValue");
  alphaValue.textContent = aero.alpha === null ? "—" : formatAngleDegrees(aero.alpha);
  betaValue.textContent = aero.beta === null ? "—" : formatAngleDegrees(aero.beta);
  alphaValue.title = aero.alphaReason;
  betaValue.title = aero.betaReason;
  airComponentInputs.u.value = aero.components.u.toFixed(1);
  airComponentInputs.v.value = aero.components.v.toFixed(1);
  airComponentInputs.w.value = aero.components.w.toFixed(1);
  document.getElementById("airComponentFrame").textContent = "FRD body axes · m/s";
  document.getElementById("attitudeSummary").innerHTML = bank.value === null
    ? `φ<sub>B</sub> ${state.roll.toFixed(1)}°`
    : `φ<sub>B</sub> ${state.roll.toFixed(1)}° · σ<sub>L</sub> ${formatAngleDegrees(bank.value)}`;
  document.getElementById("velocitySummary").innerHTML = `γ<sub>V</sub> ${Number(fields.gamma.value).toFixed(1)}° · χ<sub>V</sub> ${Number(fields.course.value).toFixed(1)}°`;
  document.getElementById("windSummary").innerHTML = wind.lengthSq() > 1e-9
    ? `|V<sub>W</sub>| ${wind.length().toFixed(1)} m/s`
    : "calm";
  windComponentInputs.north.value = wind.x.toFixed(1);
  windComponentInputs.up.value = wind.y.toFixed(1);
  windComponentInputs.east.value = wind.z.toFixed(1);
  document.getElementById("positionSummary").textContent = `N ${state.vehicleNorth.toFixed(0)} · E ${state.vehicleEast.toFixed(0)} · D ${state.vehicleDown.toFixed(0)} m`;
  Object.values(fields).forEach(updateSliderFill);
}

const vectorRaycaster = new THREE.Raycaster();
const pointerNdc = new THREE.Vector2();
let hoveredVector = null;
let vectorDrag = null;
let hoveredVehicle = false;
let attitudeDrag = null;

function setPointerFromEvent(event) {
  const bounds = renderer.domElement.getBoundingClientRect();
  pointerNdc.set(
    ((event.clientX - bounds.left) / bounds.width) * 2 - 1,
    -((event.clientY - bounds.top) / bounds.height) * 2 + 1
  );
  vectorRaycaster.setFromCamera(pointerNdc, camera);
}

function pickVectorHandle(event) {
  setPointerFromEvent(event);
  scene.updateMatrixWorld(true);
  const hit = vectorRaycaster.intersectObjects(vectorHandles.map((visual) => visual.hitTarget), false)[0];
  return hit ? vectorHandles.find((visual) => visual.id === hit.object.userData.vectorId) : null;
}

function pickVehicle(event) {
  setPointerFromEvent(event);
  scene.updateMatrixWorld(true);
  return vectorRaycaster.intersectObject(vehicleHitProxy, false).length > 0;
}

function syncAttitudeFields(quaternion) {
  const bodyX = new THREE.Vector3(1, 0, 0).applyQuaternion(quaternion).normalize();
  const bodyY = new THREE.Vector3(0, 1, 0).applyQuaternion(quaternion).normalize();
  const horizontal = Math.hypot(bodyX.x, bodyX.z);
  const pitch = Math.asin(THREE.MathUtils.clamp(bodyX.y, -1, 1));
  let yaw = Number(fields.yaw.value) * DEG;
  if (horizontal > 1e-7) yaw = Math.atan2(bodyX.z, bodyX.x);
  const heading = new THREE.Vector3(Math.cos(yaw), 0, Math.sin(yaw));
  const baseRight = new THREE.Vector3().crossVectors(heading, new THREE.Vector3(0, 1, 0)).normalize();
  const roll = Math.atan2(
    bodyX.dot(new THREE.Vector3().crossVectors(baseRight, bodyY)),
    baseRight.dot(bodyY)
  );
  fields.pitch.value = String(THREE.MathUtils.radToDeg(pitch));
  fields.yaw.value = String(THREE.MathUtils.radToDeg(yaw));
  fields.roll.value = String(THREE.MathUtils.radToDeg(roll));
}

function emphasizeVectorHandles() {
  for (const visual of vectorHandles) {
    const emphasized = visual === hoveredVector || visual === vectorDrag?.visual;
    visual.handle.scale.setScalar(emphasized ? 1.24 : 1);
    visual.knob.material.opacity = emphasized ? 1 : 0.96;
  }
  const dragging = vectorDrag || attitudeDrag;
  renderer.domElement.style.cursor = dragging ? "grabbing" : (hoveredVector || hoveredVehicle) ? "grab" : "";
}

function clearPresetSelection() {
  document.querySelectorAll("#statePresets button").forEach((button) => button.classList.remove("active"));
}

function setAttitudeControlMode(mode) {
  attitudeControlMode = mode === "direct" ? "direct" : "gimbal";
  attitudeGimbal.visible = attitudeControlMode === "gimbal";
  attitudeGimbal.enabled = attitudeGimbal.visible;
  gimbalDock.classList.toggle("active", attitudeControlMode === "gimbal");
  gimbalDock.setAttribute("aria-hidden", String(attitudeControlMode !== "gimbal"));
  document.querySelectorAll("#attitudeControlMode button").forEach((button) => {
    button.classList.toggle("active", button.dataset.attitudeMode === attitudeControlMode);
  });
  document.getElementById("attitudeControlMode").title = attitudeControlMode === "gimbal"
    ? "Use the docked local Body-axis rings in the lower-right corner."
    : "Drag the vehicle in the main view for screen-relative free rotation.";
  hoveredVehicle = false;
  emphasizeVectorHandles();
  if (fields.pitch.value !== "") updateScene();
}

attitudeGimbal.addEventListener("dragging-changed", (event) => {
  gimbalDragging = event.value;
  gimbalRenderer.domElement.style.cursor = event.value ? "grabbing" : "grab";
});
attitudeGimbal.addEventListener("objectChange", () => {
  if (attitudeControlMode !== "gimbal") return;
  syncAttitudeFields(gimbalProxy.quaternion);
  clearPresetSelection();
  updateScene();
});

renderer.domElement.addEventListener("pointerdown", (event) => {
  if (event.button !== 0) return;
  const visual = pickVectorHandle(event);
  if (visual) {
    const displayedEndpoint = visual.handle.getWorldPosition(new THREE.Vector3());
    const planeNormal = camera.getWorldDirection(new THREE.Vector3());
    const plane = new THREE.Plane().setFromNormalAndCoplanarPoint(planeNormal, displayedEndpoint);
    const startPoint = vectorRaycaster.ray.intersectPlane(plane, new THREE.Vector3());
    if (!startPoint) return;
    vectorDrag = {
      visual,
      plane,
      startPoint,
      startDataEndpoint: visual.currentVector.clone().multiplyScalar(visual.scale),
    };
    hoveredVector = visual;
  } else if (attitudeControlMode === "direct" && pickVehicle(event)) {
    attitudeDrag = {
      startX: event.clientX,
      startY: event.clientY,
      quaternion: spacecraft.quaternion.clone(),
      cameraRight: new THREE.Vector3(1, 0, 0).applyQuaternion(camera.quaternion).normalize(),
      cameraUp: new THREE.Vector3(0, 1, 0).applyQuaternion(camera.quaternion).normalize(),
    };
    hoveredVehicle = true;
  } else {
    return;
  }
  event.preventDefault();
  event.stopImmediatePropagation();
  orbit.enabled = false;
  renderer.domElement.setPointerCapture(event.pointerId);
  emphasizeVectorHandles();
}, true);

renderer.domElement.addEventListener("pointermove", (event) => {
  if (!vectorDrag && !attitudeDrag) {
    hoveredVector = pickVectorHandle(event);
    hoveredVehicle = attitudeControlMode === "direct" && !hoveredVector && pickVehicle(event);
    emphasizeVectorHandles();
    return;
  }
  event.preventDefault();
  event.stopImmediatePropagation();
  if (vectorDrag) {
    setPointerFromEvent(event);
    const point = vectorRaycaster.ray.intersectPlane(vectorDrag.plane, new THREE.Vector3());
    if (!point) return;
    const endpoint = vectorDrag.startDataEndpoint.clone().add(point).sub(vectorDrag.startPoint);
    const maxLength = vectorDrag.visual.maxMagnitude * vectorDrag.visual.scale;
    if (endpoint.length() > maxLength) endpoint.setLength(maxLength);
    const vector = endpoint.length() < 0.35
      ? new THREE.Vector3()
      : endpoint.divideScalar(vectorDrag.visual.scale);
    if (vectorDrag.visual.id === "velocity") {
      interactiveVelocity.copy(vector);
      syncVelocityFields(interactiveVelocity);
    } else {
      interactiveWind.copy(vector);
    }
    clearPresetSelection();
  } else {
    const yaw = new THREE.Quaternion().setFromAxisAngle(
      attitudeDrag.cameraUp,
      -(event.clientX - attitudeDrag.startX) * 0.007
    );
    const pitch = new THREE.Quaternion().setFromAxisAngle(
      attitudeDrag.cameraRight,
      -(event.clientY - attitudeDrag.startY) * 0.007
    );
    const rotated = pitch.multiply(yaw).multiply(attitudeDrag.quaternion).normalize();
    syncAttitudeFields(rotated);
    clearPresetSelection();
  }
  updateScene();
  emphasizeVectorHandles();
}, true);

function finishDrag(event) {
  if (!vectorDrag && !attitudeDrag) return;
  event.preventDefault();
  event.stopImmediatePropagation();
  if (renderer.domElement.hasPointerCapture(event.pointerId)) {
    renderer.domElement.releasePointerCapture(event.pointerId);
  }
  vectorDrag = null;
  attitudeDrag = null;
  orbit.enabled = true;
  hoveredVector = pickVectorHandle(event);
  hoveredVehicle = attitudeControlMode === "direct" && !hoveredVector && pickVehicle(event);
  emphasizeVectorHandles();
}

renderer.domElement.addEventListener("pointerup", finishDrag, true);
renderer.domElement.addEventListener("pointercancel", finishDrag, true);
renderer.domElement.addEventListener("pointerleave", () => {
  if (!vectorDrag && !attitudeDrag) {
    hoveredVector = null;
    hoveredVehicle = false;
    emphasizeVectorHandles();
  }
});

function boundedInputValue(input) {
  const value = Number(input.value);
  if (!Number.isFinite(value)) return null;
  const minimum = input.min === "" ? -Infinity : Number(input.min);
  const maximum = input.max === "" ? Infinity : Number(input.max);
  return THREE.MathUtils.clamp(value, minimum, maximum);
}

for (const [key, numberInput] of Object.entries(outputs)) {
  numberInput.addEventListener("change", () => {
    const value = boundedInputValue(numberInput);
    if (value === null) {
      updateScene();
      return;
    }
    fields[key].value = String(value);
    if (["speed", "gamma", "course"].includes(key)) {
      interactiveVelocity.copy(velocityFromState(currentState()));
    }
    if (key in flightFields) clearPresetSelection();
    updateScene();
  });
}

Object.values(windComponentInputs).forEach((input) => input.addEventListener("change", () => {
  const north = boundedInputValue(windComponentInputs.north);
  const up = boundedInputValue(windComponentInputs.up);
  const east = boundedInputValue(windComponentInputs.east);
  if ([north, up, east].some((value) => value === null)) {
    updateScene();
    return;
  }
  interactiveWind.set(north, up, east);
  if (interactiveWind.length() > windVisual.maxMagnitude) {
    interactiveWind.setLength(windVisual.maxMagnitude);
  }
  clearPresetSelection();
  updateScene();
}));

Object.values(airComponentInputs).forEach((input) => input.addEventListener("change", () => {
  const u = boundedInputValue(airComponentInputs.u);
  const v = boundedInputValue(airComponentInputs.v);
  const w = boundedInputValue(airComponentInputs.w);
  if ([u, v, w].some((value) => value === null)) {
    updateScene();
    return;
  }
  const airWorld = new THREE.Vector3(u, v, w).applyQuaternion(attitudeFromState(currentState()));
  interactiveVelocity.copy(airWorld).add(interactiveWind);
  if (interactiveVelocity.length() > velocityVisual.maxMagnitude) {
    interactiveVelocity.setLength(velocityVisual.maxMagnitude);
  }
  syncVelocityFields(interactiveVelocity);
  clearPresetSelection();
  updateScene();
}));

for (const input of [...Object.values(outputs), ...Object.values(windComponentInputs), ...Object.values(airComponentInputs)]) {
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") input.blur();
  });
}

for (const [key, input] of Object.entries(flightFields)) {
  input.addEventListener("input", () => {
    if (["speed", "gamma", "course"].includes(key)) {
      interactiveVelocity.copy(velocityFromState(currentState()));
    }
    clearPresetSelection();
    updateScene();
  });
}
Object.values(positionFields).forEach((input) => input.addEventListener("input", updateScene));
document.getElementById("showTargetInput").addEventListener("change", updateScene);
document.getElementById("showAeroGeometryInput").addEventListener("change", updateScene);
document.getElementById("resetPosition").addEventListener("click", () => {
  positionFields.vehicleNorth.value = String(DEFAULT_VEHICLE_NORTH);
  positionFields.vehicleEast.value = String(DEFAULT_VEHICLE_EAST);
  positionFields.vehicleDown.value = String(DEFAULT_VEHICLE_DOWN);
  updateScene();
});
document.getElementById("frameTarget").addEventListener("click", () => {
  const showTarget = document.getElementById("showTargetInput");
  showTarget.checked = true;
  updateScene();
  const target = new THREE.Vector3().fromArray(meta.target);
  const midpoint = target.clone().multiplyScalar(0.5);
  const viewOffset = camera.position.clone().sub(orbit.target);
  viewOffset.setLength(Math.max(viewOffset.length(), target.length() * 2));
  orbit.target.copy(midpoint);
  camera.position.copy(midpoint).add(viewOffset);
  orbit.update();
});
document.querySelectorAll("#statePresets button").forEach((button) => button.addEventListener("click", () => setPreset(button.dataset.preset)));
alignBodyWithAirButton.addEventListener("click", alignBodyWithAirVelocity);
document.querySelectorAll("#attitudeControlMode button").forEach((button) => button.addEventListener("click", () => {
  setAttitudeControlMode(button.dataset.attitudeMode);
}));
document.getElementById("levelWind").addEventListener("click", () => {
  interactiveWind.y = 0;
  clearPresetSelection();
  updateScene();
});
document.getElementById("calmWind").addEventListener("click", () => {
  interactiveWind.set(0, 0, 0);
  clearPresetSelection();
  updateScene();
});
document.getElementById("resetCamera").addEventListener("click", () => {
  camera.position.copy(DEFAULT_CAMERA);
  orbit.target.copy(DEFAULT_ORBIT_TARGET);
  orbit.update();
});

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  labelRenderer.setSize(window.innerWidth, window.innerHeight);
});

function animate() {
  requestAnimationFrame(animate);
  orbit.update();
  for (const visual of vectorHandles) visual.ring.quaternion.copy(camera.quaternion);
  renderer.render(scene, camera);
  labelRenderer.render(scene, camera);
  if (attitudeControlMode === "gimbal") gimbalRenderer.render(gimbalScene, gimbalCamera);
}

positionFields.vehicleNorth.value = String(DEFAULT_VEHICLE_NORTH);
positionFields.vehicleEast.value = String(DEFAULT_VEHICLE_EAST);
positionFields.vehicleDown.value = String(DEFAULT_VEHICLE_DOWN);
setAttitudeControlMode("direct");
setPreset("descent");
animate();

import * as THREE from "three";
import { CSS2DObject } from "three/addons/renderers/CSS2DRenderer.js";

const EPS = 1e-9;
const NATIVE_NORM_EPS = 1e-15;
const NATIVE_VERTICAL_EPS = 1e-12;
const AXIS_COLORS = [0xff6b78, 0x70dfa2, 0x6f9dff];
const FRAME_ACCENTS = ["#65e6e1", "#ffc66d", "#c7a0ff", "#ff7588", "#83aaff", "#8ce0a9"];
const MAX_VISIBLE_FRAMES = 3;

export const BUILTIN_FRAME_DEFINITIONS = [
  {
    id: "velocity",
    short: "VEL",
    name: "Ground-velocity frame",
    citation: "RLV-Sim native direction-frame convention",
    description: "+x_V follows ground-relative velocity, +y_V points right, and +z_V points down.",
    axes: [
      { id: "x", label: "x", subscript: "V", name: "ground-velocity direction", direction: { ref: "velocity.x" } },
      { id: "y", label: "y", subscript: "V", name: "right/lateral", direction: { ref: "velocity.y" } },
      { id: "z", label: "z", subscript: "V", name: "down", direction: { ref: "velocity.z" } },
    ],
  },
  {
    id: "los",
    short: "LOS",
    name: "Line-of-sight frame",
    citation: "RLV-Sim native direction-frame convention",
    description: "+x_L points from the vehicle to the target, +y_L points right, and +z_L points down.",
    axes: [
      { id: "x", label: "x", subscript: "L", name: "observer-to-target direction", direction: { ref: "los.x" } },
      { id: "y", label: "y", subscript: "L", name: "right/lateral", direction: { ref: "los.y" } },
      { id: "z", label: "z", subscript: "L", name: "down", direction: { ref: "los.z" } },
    ],
  },
  {
    id: "enu",
    short: "ENU",
    name: "East-North-Up",
    citation: "Standard local tangent navigation frame",
    description: "Right-handed axes ordered East, North, Up.",
    axes: [
      { id: "E", label: "E", name: "east", direction: { ref: "east" } },
      { id: "N", label: "N", name: "north", direction: { ref: "north" } },
      { id: "U", label: "U", name: "up", direction: { ref: "up" } },
    ],
  },
  {
    id: "ned",
    short: "NED",
    name: "North-East-Down",
    citation: "Native plant navigation frame",
    description: "Right-handed axes ordered North, East, Down.",
    axes: [
      { id: "N", label: "N", name: "north", direction: { ref: "north" } },
      { id: "E", label: "E", name: "east", direction: { ref: "east" } },
      { id: "D", label: "D", name: "down", direction: { ref: "down" } },
    ],
  },
  {
    id: "body",
    short: "BODY",
    name: "Spacecraft body frame",
    citation: "FRD vehicle-fixed convention",
    description: "+x_B points forward, +y_B points right, and +z_B points down.",
    axes: [
      { id: "x", label: "x", subscript: "B", name: "forward", direction: { ref: "body.x" } },
      { id: "y", label: "y", subscript: "B", name: "right", direction: { ref: "body.y" } },
      { id: "z", label: "z", subscript: "B", name: "down", direction: { ref: "body.z" } },
    ],
  },
];

const VECTOR_SOURCES = [
  { id: "velocity", label: "Ground-relative velocity", symbol: "V_G", ref: "velocity", unit: "m/s", color: 0x6f9dff },
  { id: "air", label: "Air-relative velocity", symbol: "V_A", ref: "air_relative", unit: "m/s", color: 0xc7a0ff },
  { id: "wind", label: "Wind velocity", symbol: "V_W", ref: "wind", unit: "m/s", color: 0x65e6e1 },
  { id: "los", label: "LOS direction", symbol: "LOS", ref: "los", unit: "", color: 0xff7588, normalize: true },
  { id: "body_x", label: "Body x", symbol: "x_B", ref: "body.x", unit: "", color: 0xffc66d, normalize: true },
];

function byId(id) {
  return document.getElementById(id);
}

function cleanId(value) {
  return String(value || "").trim().replace(/[^a-zA-Z0-9_-]/g, "_");
}

function vectorFromArray(value, label) {
  if (!Array.isArray(value) || value.length !== 3 || value.some((item) => !Number.isFinite(Number(item)))) {
    throw new Error(`${label} must be a three-number array`);
  }
  return new THREE.Vector3(Number(value[0]), Number(value[1]), Number(value[2]));
}

function unit(vector, fallback = new THREE.Vector3(1, 0, 0)) {
  if (vector.lengthSq() > EPS) return vector.normalize();
  return vector.copy(fallback).normalize();
}

function strictUnit(vector, label) {
  if (vector.lengthSq() <= EPS) throw new Error(`${label} is degenerate in this state`);
  return vector.normalize();
}

export function directionFrameAxes(direction, downReference = new THREE.Vector3(0, -1, 0)) {
  const normalizeNative = (vector, label) => {
    const magnitude = vector.length();
    if (!(magnitude > NATIVE_NORM_EPS) || !Number.isFinite(magnitude)) {
      throw new Error(`${label} must have non-zero finite norm`);
    }
    return vector.multiplyScalar(1 / magnitude);
  };
  const x = normalizeNative(direction.clone(), "direction-frame x axis");
  const down = normalizeNative(downReference.clone(), "direction-frame down reference");
  let y = new THREE.Vector3().crossVectors(down, x);
  let usedVerticalDatum = false;
  if (y.length() <= NATIVE_VERTICAL_EPS) {
    usedVerticalDatum = true;
    const candidates = [
      new THREE.Vector3(1, 0, 0),
      new THREE.Vector3(0, 0, 1),
      new THREE.Vector3(0, -1, 0),
    ];
    const datum = candidates.reduce((best, candidate) => (
      Math.abs(candidate.dot(down)) < Math.abs(best.dot(down)) ? candidate : best
    ));
    y = new THREE.Vector3().crossVectors(down, datum);
    y.addScaledVector(x, -y.dot(x));
  }
  normalizeNative(y, "direction-frame y axis");
  const z = new THREE.Vector3().crossVectors(x, y);
  return { x, y, z, usedVerticalDatum };
}

function axisText(axis) {
  return `${axis.label}${axis.subscript ? `_${axis.subscript}` : ""}`;
}

function axisSymbol(axis, className = "") {
  const symbol = document.createElement("span");
  if (className) symbol.className = className;
  symbol.append(document.createTextNode(axis.label));
  if (axis.subscript) {
    const subscript = document.createElement("sub");
    subscript.textContent = axis.subscript;
    symbol.append(subscript);
  }
  return symbol;
}

export function validateFrameDefinition(input) {
  if (!input || typeof input !== "object") throw new Error("frame definition must be an object");
  const id = cleanId(input.id);
  if (!id) throw new Error("frame definition needs an id");
  if (!Array.isArray(input.axes) || input.axes.length < 2 || input.axes.length > 3) {
    throw new Error(`${id}: axes must contain two or three axes`);
  }
  const seen = new Set();
  const axes = input.axes.map((axis, index) => {
    const axisId = cleanId(axis.id || `axis_${index + 1}`);
    if (seen.has(axisId)) throw new Error(`${id}: duplicate axis id ${axisId}`);
    if (axis.direction === undefined) throw new Error(`${id}.${axisId}: direction is required`);
    seen.add(axisId);
    return {
      id: axisId,
      label: String(axis.label || axisId),
      subscript: String(axis.subscript || ""),
      name: String(axis.name || ""),
      direction: axis.direction,
      orthogonalize: axis.orthogonalize !== false,
    };
  });
  return {
    id,
    short: String(input.short || id).slice(0, 12),
    name: String(input.name || id),
    citation: String(input.citation || "Custom frame"),
    description: String(input.description || ""),
    handedness: input.handedness === "left" ? "left" : "right",
    axes,
  };
}

function resolveExpression(expression, context, resolvedAxes) {
  if (Array.isArray(expression)) return vectorFromArray(expression, "direction");
  if (typeof expression === "string") return resolveExpression({ ref: expression }, context, resolvedAxes);
  if (!expression || typeof expression !== "object") throw new Error("invalid direction expression");

  if (expression.axis) {
    const axis = resolvedAxes.get(cleanId(expression.axis));
    if (!axis) throw new Error(`axis ${expression.axis} is referenced before it is defined`);
    return axis.clone();
  }
  if (expression.ref) {
    const value = context.refs.get(String(expression.ref));
    if (!value) throw new Error(`unknown direction reference: ${expression.ref}`);
    return value.clone();
  }
  if (expression.negate !== undefined) return resolveExpression(expression.negate, context, resolvedAxes).negate();
  if (expression.cross) {
    if (!Array.isArray(expression.cross) || expression.cross.length !== 2) throw new Error("cross needs two expressions");
    return new THREE.Vector3().crossVectors(
      resolveExpression(expression.cross[0], context, resolvedAxes),
      resolveExpression(expression.cross[1], context, resolvedAxes)
    );
  }
  if (expression.project) {
    if (!Array.isArray(expression.project) || expression.project.length !== 2) {
      throw new Error("project needs [vector, plane-normal]");
    }
    const vector = resolveExpression(expression.project[0], context, resolvedAxes);
    const normal = strictUnit(resolveExpression(expression.project[1], context, resolvedAxes), "projection normal");
    return vector.addScaledVector(normal, -vector.dot(normal));
  }
  if (expression.add) {
    if (!Array.isArray(expression.add) || expression.add.length < 2) throw new Error("add needs at least two expressions");
    return expression.add.reduce(
      (sum, item) => sum.add(resolveExpression(item, context, resolvedAxes)),
      new THREE.Vector3()
    );
  }
  if (expression.fallback) {
    if (!Array.isArray(expression.fallback) || expression.fallback.length === 0) throw new Error("fallback needs expressions");
    let lastError = null;
    for (const candidate of expression.fallback) {
      try {
        const value = resolveExpression(candidate, context, resolvedAxes);
        if (value.lengthSq() > EPS) return value;
      } catch (error) {
        lastError = error;
      }
    }
    throw lastError || new Error("all fallback directions are degenerate");
  }
  throw new Error("unsupported direction expression");
}

export function resolveFrameDefinition(definition, context) {
  const axes = new Map();
  for (const axis of definition.axes) {
    const direction = resolveExpression(axis.direction, context, axes);
    if (axis.orthogonalize) {
      for (const previous of axes.values()) direction.addScaledVector(previous, -direction.dot(previous));
    }
    if (direction.lengthSq() <= EPS) throw new Error(`${definition.id}.${axis.id} is degenerate in this state`);
    axes.set(axis.id, direction.normalize());
  }

  if (definition.axes.length === 2) {
    const first = axes.get(definition.axes[0].id);
    const second = axes.get(definition.axes[1].id);
    axes.set("normal", unit(new THREE.Vector3().crossVectors(first, second)));
  } else {
    const first = axes.get(definition.axes[0].id);
    const second = axes.get(definition.axes[1].id);
    const third = axes.get(definition.axes[2].id);
    const handedness = new THREE.Vector3().crossVectors(first, second).dot(third);
    if ((definition.handedness === "right" && handedness < 0) || (definition.handedness === "left" && handedness > 0)) {
      third.negate();
    }
  }
  return axes;
}

function makeAxisVisual(root, definition, axis, index, accent, onPick) {
  const color = AXIS_COLORS[index];
  const arrow = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(), 1, color, 2, 1);
  arrow.line.material.dispose();
  arrow.line.material = new THREE.LineDashedMaterial({ color, dashSize: 0.12, gapSize: 0.08 });
  arrow.line.computeLineDistances();
  for (const material of [arrow.line.material, arrow.cone.material]) {
    material.transparent = true;
    material.opacity = 0.82;
    material.depthTest = false;
  }
  arrow.renderOrder = 52;
  root.add(arrow);

  const element = document.createElement("button");
  element.type = "button";
  element.className = "frame-label workbench-axis-label";
  element.dataset.frameId = definition.id;
  element.dataset.axisId = axis.id;
  element.style.setProperty("--frame-accent", accent);
  const tag = document.createElement("span");
  tag.className = "axisFrameTag";
  tag.textContent = definition.short;
  const symbol = axisSymbol(axis, "axisSymbol");
  element.append(tag, symbol);
  element.title = `${definition.name}: ${axis.name || axisText(axis)} — click to select this axis`;
  element.addEventListener("click", (event) => {
    event.stopPropagation();
    onPick(definition.id, axis.id);
  });
  const label = new CSS2DObject(element);
  root.add(label);
  return { arrow, label, element, definition, axis, direction: new THREE.Vector3() };
}

function setArrow(visual, origin, direction, length, opacity = 0.9, labelLength = length * 1.1) {
  visual.direction.copy(direction);
  visual.arrow.position.copy(origin);
  visual.arrow.setDirection(direction);
  visual.arrow.setLength(length, Math.min(length * 0.13, 2.8), Math.min(length * 0.052, 1.0));
  visual.arrow.line.material.opacity = opacity;
  visual.arrow.cone.material.opacity = opacity;
  visual.label.position.copy(origin).addScaledVector(direction, labelLength);
}

function makeProjectionArrow(root, color) {
  const arrow = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(), 1, color, 2, 1);
  for (const material of [arrow.line.material, arrow.cone.material]) {
    material.transparent = true;
    material.depthTest = false;
  }
  arrow.renderOrder = 58;
  root.add(arrow);
  return arrow;
}

function updateProjectionArrow(arrow, origin, vector, length, visible) {
  arrow.visible = visible;
  if (!visible) return;
  arrow.position.copy(origin);
  arrow.setDirection(unit(vector.clone()));
  arrow.setLength(length, Math.min(length * 0.25, 3.2), Math.min(length * 0.12, 1.5));
}

function makeLine(root, color, dashed = false, opacity = 0.7) {
  const material = dashed
    ? new THREE.LineDashedMaterial({ color, dashSize: 2.4, gapSize: 1.5, transparent: true, opacity, depthTest: false })
    : new THREE.LineBasicMaterial({ color, transparent: true, opacity, depthTest: false });
  const line = new THREE.Line(new THREE.BufferGeometry(), material);
  line.renderOrder = 57;
  root.add(line);
  return line;
}

function setLine(line, points, visible = true) {
  line.visible = visible;
  if (!visible) return;
  line.geometry.dispose();
  line.geometry = new THREE.BufferGeometry().setFromPoints(points);
  if (line.material.isLineDashedMaterial) line.computeLineDistances();
}

function makePlaneVisual(root) {
  const group = new THREE.Group();
  const plane = new THREE.Mesh(
    new THREE.PlaneGeometry(1, 1),
    new THREE.MeshBasicMaterial({
      color: 0x65e6e1,
      transparent: true,
      opacity: 0.09,
      side: THREE.DoubleSide,
      depthWrite: false,
      depthTest: false,
      blending: THREE.AdditiveBlending,
    })
  );
  plane.renderOrder = 38;
  group.add(plane);

  const positions = [];
  for (let index = -4; index <= 4; index++) {
    const value = index / 8;
    positions.push(-0.5, value, 0, 0.5, value, 0, value, -0.5, 0, value, 0.5, 0);
  }
  const gridGeometry = new THREE.BufferGeometry();
  gridGeometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  const grid = new THREE.LineSegments(
    gridGeometry,
    new THREE.LineBasicMaterial({ color: 0x65e6e1, transparent: true, opacity: 0.12, depthTest: false })
  );
  grid.position.z = 0.002;
  grid.renderOrder = 39;
  group.add(grid);

  const outline = new THREE.LineLoop(
    new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(-0.5, -0.5, 0), new THREE.Vector3(0.5, -0.5, 0),
      new THREE.Vector3(0.5, 0.5, 0), new THREE.Vector3(-0.5, 0.5, 0),
    ]),
    new THREE.LineBasicMaterial({ color: 0x65e6e1, transparent: true, opacity: 0.46, depthTest: false })
  );
  outline.position.z = 0.003;
  outline.renderOrder = 40;
  group.add(outline);

  const labelElement = document.createElement("div");
  labelElement.className = "frame-label workbench-plane-label";
  const label = new CSS2DObject(labelElement);
  root.add(group, label);
  return { group, label, labelElement };
}

function orientPlane(visual, origin, axisU, axisV, size) {
  const u = unit(axisU.clone());
  const v = unit(axisV.clone().addScaledVector(u, -axisV.dot(u)));
  const normal = unit(new THREE.Vector3().crossVectors(u, v));
  visual.group.position.copy(origin);
  visual.group.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(u, v, normal));
  visual.group.scale.set(size, size, size);
  visual.label.position.copy(origin).addScaledVector(axisU, size * 0.37).addScaledVector(axisV, size * 0.43);
}

function makeContext(meta, sample) {
  const north = new THREE.Vector3(1, 0, 0);
  const up = new THREE.Vector3(0, 1, 0);
  const east = new THREE.Vector3(0, 0, 1);
  const down = up.clone().negate();
  const airWorld = sample.airBody
    ? sample.airBody.clone().applyQuaternion(sample.quaternion)
    : sample.velocity.clone();
  const wind = sample.velocity.clone().sub(airWorld);
  const target = new THREE.Vector3().fromArray(meta.target || [0, 0, 0]);
  const los = target.sub(sample.position);
  const refs = new Map([
    ["north", north], ["east", east], ["up", up], ["down", down],
    ["velocity", sample.velocity.clone()], ["air_relative", airWorld], ["wind", wind],
    ["los", los],
    ["body.x", new THREE.Vector3(1, 0, 0).applyQuaternion(sample.quaternion)],
    ["body.y", new THREE.Vector3(0, 1, 0).applyQuaternion(sample.quaternion)],
    ["body.z", new THREE.Vector3(0, 0, 1).applyQuaternion(sample.quaternion)],
  ]);
  for (const [prefix, direction] of [["velocity", sample.velocity], ["los", los]]) {
    if (direction.length() > NATIVE_NORM_EPS) {
      const frame = directionFrameAxes(direction, down);
      refs.set(`${prefix}.x`, frame.x);
      refs.set(`${prefix}.y`, frame.y);
      refs.set(`${prefix}.z`, frame.z);
    } else {
      refs.set(`${prefix}.x`, new THREE.Vector3());
      refs.set(`${prefix}.y`, new THREE.Vector3());
      refs.set(`${prefix}.z`, new THREE.Vector3());
    }
  }
  return { refs };
}

export function buildFrameWorkbench(scene, camera, canvas, vehicle, meta, options = {}) {
  const root = new THREE.Group();
  root.name = "frame workbench";
  root.visible = options.active === true;
  scene.add(root);

  const axisLength = THREE.MathUtils.clamp(vehicle.length * 0.82, 34, 50);
  const definitions = new Map();
  const visuals = new Map();
  const visible = new Set(options.visibleFrames || ["velocity"]);
  const safeInsets = {
    left: options.safeInsets?.left ?? 34,
    right: options.safeInsets?.right ?? Math.min(420, canvas.clientWidth * 0.4),
    top: options.safeInsets?.top ?? 55,
    bottom: options.safeInsets?.bottom ?? 255,
  };
  const selectedAxes = [];
  let active = options.active === true;
  let vectorSource = "velocity";
  let latestContext = null;
  let latestSample = null;
  let latestPosition = new THREE.Vector3();
  const projectedOrigin = new THREE.Vector3();
  const projectedEnd = new THREE.Vector3();

  const frameList = byId("studioFrameList");
  const axisPicker = byId("studioAxisPicker");
  const status = byId("studioStatus");
  const planeHint = byId("studioPlaneHint");
  const projectionReadout = byId("projectionReadout");
  const importInput = byId("frameImportInput");

  const planeVisual = makePlaneVisual(root);
  planeVisual.group.visible = false;
  planeVisual.label.visible = false;
  const projectionRoot = new THREE.Group();
  root.add(projectionRoot);
  const sourceArrow = makeProjectionArrow(projectionRoot, 0xf4f7fb);
  const projectedArrow = makeProjectionArrow(projectionRoot, 0x65e6e1);
  const residualLine = makeLine(projectionRoot, 0xc7a0ff, true, 0.72);
  const componentLine = makeLine(projectionRoot, 0xffc66d, true, 0.62);
  const projectionLabelElement = document.createElement("div");
  projectionLabelElement.className = "frame-label projection-label";
  const projectionLabel = new CSS2DObject(projectionLabelElement);
  projectionRoot.add(projectionLabel);

  function labelFits(origin, direction, distance) {
    camera.updateMatrixWorld();
    projectedOrigin.copy(origin).project(camera);
    projectedEnd.copy(origin).addScaledVector(direction, distance).project(camera);
    if (!Number.isFinite(projectedEnd.x) || projectedEnd.z < -1 || projectedEnd.z > 1) return false;
    const x = (projectedEnd.x + 1) * canvas.clientWidth * 0.5;
    const y = (1 - projectedEnd.y) * canvas.clientHeight * 0.5;
    const right = canvas.clientWidth - safeInsets.right;
    const bottom = canvas.clientHeight - safeInsets.bottom;
    const pixels = Math.hypot(
      (projectedEnd.x - projectedOrigin.x) * canvas.clientWidth * 0.5,
      (projectedEnd.y - projectedOrigin.y) * canvas.clientHeight * 0.5
    );
    return x > safeInsets.left && x < right && y > safeInsets.top && y < bottom
      && pixels < Math.min(canvas.clientWidth, canvas.clientHeight) * 0.22;
  }

  function fittedLabelDistance(origin, direction, requested) {
    if (labelFits(origin, direction, requested)) return requested;
    let low = 0;
    let high = requested;
    for (let index = 0; index < 12; index++) {
      const middle = (low + high) * 0.5;
      if (labelFits(origin, direction, middle)) low = middle;
      else high = middle;
    }
    return Math.max(low, 0.2);
  }

  function setStatus(message, error = false) {
    status.textContent = message;
    status.classList.toggle("error", error);
  }

  function pickAxis(frameId, axisId) {
    if (!visible.has(frameId)) return;
    const existing = selectedAxes.findIndex((item) => item.frameId === frameId && item.axisId === axisId);
    if (existing >= 0) {
      selectedAxes.splice(existing, 1);
    } else if (selectedAxes.length === 0 || selectedAxes[0].frameId !== frameId) {
      selectedAxes.splice(0, selectedAxes.length, { frameId, axisId });
    } else {
      if (selectedAxes.length === 2) selectedAxes.pop();
      selectedAxes.push({ frameId, axisId });
    }
    refreshSelectionUI();
    updateProjection();
  }

  function createVisual(definition, index) {
    const group = new THREE.Group();
    group.name = `${definition.name} frame`;
    root.add(group);
    const accent = FRAME_ACCENTS[index % FRAME_ACCENTS.length];
    const axes = definition.axes.map((axis, axisIndex) =>
      makeAxisVisual(group, definition, axis, axisIndex, accent, pickAxis)
    );
    visuals.set(definition.id, { definition, group, axes, resolved: null, accent });
  }

  function addDefinition(rawDefinition) {
    const definition = validateFrameDefinition(rawDefinition);
    if (definitions.has(definition.id)) throw new Error(`frame id already exists: ${definition.id}`);
    definitions.set(definition.id, definition);
    createVisual(definition, definitions.size - 1);
    renderLibrary();
    return definition;
  }

  function toggleFrame(frameId) {
    if (visible.has(frameId)) {
      visible.delete(frameId);
      for (let index = selectedAxes.length - 1; index >= 0; index--) {
        if (selectedAxes[index].frameId === frameId) selectedAxes.splice(index, 1);
      }
    } else {
      if (visible.size >= MAX_VISIBLE_FRAMES) {
        setStatus(`Up to ${MAX_VISIBLE_FRAMES} frames can be overlaid clearly. Hide one first.`, true);
        return;
      }
      visible.add(frameId);
      setStatus("");
    }
    renderLibrary();
    renderAxisPicker();
    refreshSelectionUI();
    if (latestSample) update(latestSample);
    else {
      updateVisualVisibility();
      updateProjection();
    }
  }

  function renderLibrary() {
    frameList.replaceChildren();
    for (const [frameId, definition] of definitions) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "frameLibraryChip";
      button.classList.toggle("active", visible.has(frameId));
      button.dataset.frameId = frameId;
      const visual = visuals.get(frameId);
      button.style.setProperty("--chip-accent", visual.accent);
      const dot = document.createElement("span");
      dot.className = "frameDot";
      dot.style.background = visual.accent;
      const name = document.createElement("span");
      name.textContent = definition.short;
      button.append(dot, name);
      button.title = `${definition.name}\n${definition.citation}\n${definition.description}`;
      button.addEventListener("click", () => toggleFrame(frameId));
      frameList.append(button);
    }
  }

  function renderAxisPicker() {
    axisPicker.replaceChildren();
    for (const frameId of visible) {
      const definition = definitions.get(frameId);
      if (!definition) continue;
      const row = document.createElement("div");
      row.className = "axisPickerRow";
      const name = document.createElement("span");
      name.className = "axisPickerName";
      name.textContent = definition.short;
      row.append(name);
      for (const axis of definition.axes) {
        const button = document.createElement("button");
        button.type = "button";
        button.dataset.frameId = frameId;
        button.dataset.axisId = axis.id;
        button.append(axisSymbol(axis));
        button.title = axis.name;
        button.addEventListener("click", () => pickAxis(frameId, axis.id));
        row.append(button);
      }
      axisPicker.append(row);
    }
  }

  function refreshSelectionUI() {
    document.querySelectorAll("#studioAxisPicker button[data-axis-id]").forEach((button) => {
      button.classList.toggle("selected", selectedAxes.some(
        (item) => item.frameId === button.dataset.frameId && item.axisId === button.dataset.axisId
      ));
    });
    for (const visual of visuals.values()) {
      for (const axis of visual.axes) {
        axis.element.classList.toggle("selected", selectedAxes.some(
          (item) => item.frameId === visual.definition.id && item.axisId === axis.axis.id
        ));
      }
    }
    if (selectedAxes.length === 0) {
      planeHint.textContent = "";
      return;
    }
    const definition = definitions.get(selectedAxes[0].frameId);
    const first = definition.axes.find((axis) => axis.id === selectedAxes[0].axisId);
    planeHint.replaceChildren(document.createTextNode(`${definition.short} · `), axisSymbol(first));
    if (selectedAxes.length === 1) {
      planeHint.append(document.createTextNode(" selected — choose one more axis."));
      return;
    }
    const second = definition.axes.find((axis) => axis.id === selectedAxes[1].axisId);
    planeHint.append(document.createTextNode("–"), axisSymbol(second), document.createTextNode(" plane"));
  }

  function updateVisualVisibility() {
    for (const [frameId, visual] of visuals) {
      const show = active && visible.has(frameId) && Boolean(visual.resolved);
      visual.group.visible = show;
      visual.axes.forEach((axis) => { axis.label.visible = show; });
    }
  }

  function selectedPlane() {
    if (selectedAxes.length !== 2 || selectedAxes[0].frameId !== selectedAxes[1].frameId) return null;
    const visual = visuals.get(selectedAxes[0].frameId);
    if (!visual?.resolved) return null;
    const u = visual.resolved.get(selectedAxes[0].axisId);
    const v = visual.resolved.get(selectedAxes[1].axisId);
    if (!u || !v) return null;
    const normal = new THREE.Vector3().crossVectors(u, v);
    if (normal.lengthSq() <= EPS) return null;
    return { visual, u, v, normal: normal.normalize() };
  }

  function updateProjection() {
    const plane = selectedPlane();
    const show = active && Boolean(plane) && Boolean(latestContext);
    planeVisual.group.visible = show;
    planeVisual.label.visible = show;
    projectionRoot.visible = show;
    if (!show) {
      projectionReadout.textContent = selectedAxes.length === 2 ? "Plane is unavailable in this state." : "";
      return;
    }

    const size = axisLength * 1.8;
    orientPlane(planeVisual, latestPosition, plane.u, plane.v, size);
    const firstAxis = plane.visual.definition.axes.find((axis) => axis.id === selectedAxes[0].axisId);
    const secondAxis = plane.visual.definition.axes.find((axis) => axis.id === selectedAxes[1].axisId);
    planeVisual.labelElement.replaceChildren(
      document.createTextNode(`${plane.visual.definition.short} · `),
      axisSymbol(firstAxis),
      document.createTextNode("–"),
      axisSymbol(secondAxis),
      document.createTextNode(" plane")
    );

    const source = VECTOR_SOURCES.find((item) => item.id === vectorSource) || VECTOR_SOURCES[0];
    let value = latestContext.refs.get(source.ref)?.clone() || new THREE.Vector3();
    if (source.normalize && value.lengthSq() > EPS) value.normalize();
    const magnitude = value.length();
    const gramUU = plane.u.dot(plane.u);
    const gramUV = plane.u.dot(plane.v);
    const gramVV = plane.v.dot(plane.v);
    const determinant = gramUU * gramVV - gramUV * gramUV;
    const dotU = value.dot(plane.u);
    const dotV = value.dot(plane.v);
    const componentU = (dotU * gramVV - dotV * gramUV) / determinant;
    const componentV = (dotV * gramUU - dotU * gramUV) / determinant;
    const componentN = value.dot(plane.normal);
    const projected = plane.u.clone().multiplyScalar(componentU).addScaledVector(plane.v, componentV);
    const projectedMagnitude = projected.length();
    const visualLength = axisLength * 1.12;
    const scale = magnitude > EPS ? visualLength / magnitude : 0;
    const sourceEnd = latestPosition.clone().addScaledVector(value, scale);
    const projectedEnd = latestPosition.clone().addScaledVector(projected, scale);
    const uEnd = latestPosition.clone().addScaledVector(plane.u, componentU * scale);

    sourceArrow.line.material.color.setHex(source.color);
    sourceArrow.cone.material.color.setHex(source.color);
    updateProjectionArrow(sourceArrow, latestPosition, value, visualLength, magnitude > EPS);
    updateProjectionArrow(projectedArrow, latestPosition, projected, projectedMagnitude * scale, projectedMagnitude > EPS);
    setLine(residualLine, [projectedEnd, sourceEnd], magnitude > EPS);
    setLine(componentLine, [latestPosition.clone(), uEnd, projectedEnd], projectedMagnitude > EPS);
    projectionLabel.position.copy(projectedEnd).addScaledVector(plane.normal, axisLength * 0.08);
    const [symbol, subscript = ""] = source.symbol.split("_");
    projectionLabelElement.replaceChildren(document.createTextNode("proj("));
    projectionLabelElement.append(axisSymbol({ label: symbol, subscript }), document.createTextNode(")"));

    const precision = source.normalize ? 3 : 1;
    const unitText = source.unit ? ` ${source.unit}` : "";
    projectionReadout.replaceChildren();
    const title = document.createElement("strong");
    title.textContent = `${source.label} → ${plane.visual.definition.short}`;
    const values = document.createElement("span");
    values.append(
      axisSymbol(firstAxis), document.createTextNode(` ${componentU.toFixed(precision)}${unitText}  ·  `),
      axisSymbol(secondAxis), document.createTextNode(` ${componentV.toFixed(precision)}${unitText}  ·  normal ${componentN.toFixed(precision)}${unitText}`)
    );
    projectionReadout.append(title, values);
  }

  function update(sample) {
    latestSample = {
      position: sample.position.clone(),
      velocity: sample.velocity.clone(),
      quaternion: sample.quaternion.clone(),
      airBody: sample.airBody?.clone() || null,
    };
    latestPosition.copy(sample.position);
    latestContext = makeContext(meta, sample);
    const visibleOrder = [...visible];
    for (const [frameId, visual] of visuals) {
      if (!visible.has(frameId)) continue;
      try {
        visual.resolved = resolveFrameDefinition(visual.definition, latestContext);
        const slot = visibleOrder.indexOf(frameId) - (visibleOrder.length - 1) * 0.5;
        visual.axes.forEach((axisVisual, index) => {
          const direction = visual.resolved.get(axisVisual.axis.id);
          const length = axisLength * (1 - index * 0.055);
          const labelDistance = fittedLabelDistance(sample.position, direction, length * 1.1);
          setArrow(axisVisual, sample.position, direction, length, frameId === visibleOrder[0] ? 0.94 : 0.64, labelDistance);
          axisVisual.element.style.marginLeft = `${slot * 18}px`;
          axisVisual.element.style.marginTop = `${slot * 46 + (index - 1) * 5}px`;
        });
      } catch (error) {
        visual.resolved = null;
        setStatus(error.message, true);
      }
    }
    updateVisualVisibility();
    updateProjection();
  }

  for (const definition of BUILTIN_FRAME_DEFINITIONS) {
    try {
      addDefinition(definition);
    } catch (error) {
      setStatus(error.message, true);
    }
  }
  renderLibrary();
  renderAxisPicker();
  refreshSelectionUI();

  document.querySelectorAll("#projectionSources [data-vector]").forEach((button) => {
    button.addEventListener("click", () => {
      vectorSource = button.dataset.vector;
      document.querySelectorAll("#projectionSources [data-vector]").forEach((item) => item.classList.toggle("active", item === button));
      updateProjection();
    });
  });
  byId("clearStudioPlane").addEventListener("click", () => {
    selectedAxes.splice(0, selectedAxes.length);
    refreshSelectionUI();
    updateProjection();
  });
  byId("importFrameButton").addEventListener("click", () => importInput.click());
  importInput.addEventListener("change", async () => {
    const file = importInput.files?.[0];
    if (!file) return;
    try {
      const parsed = JSON.parse(await file.text());
      const imported = Array.isArray(parsed) ? parsed : [parsed];
      const added = imported.map(addDefinition);
      for (const definition of added) {
        if (visible.size < MAX_VISIBLE_FRAMES) visible.add(definition.id);
      }
      renderLibrary();
      renderAxisPicker();
      refreshSelectionUI();
      if (latestSample) update(latestSample);
      else updateVisualVisibility();
      setStatus(`Imported ${added.map((item) => item.short).join(", ")}.`);
    } catch (error) {
      setStatus(error.message, true);
    } finally {
      importInput.value = "";
    }
  });

  function setActive(value) {
    active = Boolean(value);
    root.visible = active;
    updateVisualVisibility();
    updateProjection();
  }

  window.addEventListener("flight-frame-mode", (event) => {
    setActive(event.detail?.frame === "studio");
  });

  return { update, addDefinition, setActive };
}

/**
 * Classification Lab — Interactive Engine
 * In-browser implementations of Logistic Regression & K-Nearest Neighbors
 */

// State
const state = {
  model: 'logistic',          // 'logistic' | 'knn'
  threshold: 0.50,
  k: 5,
  distanceMetric: 'euclidean',
  activePlacementClass: 0,
  points: [],                 // [{x, y, label}]
  logisticWeights: { w1: 0, w2: 0, b: 0 },
  metrics: { tn: 0, fp: 0, fn: 0, tp: 0, acc: 0, prec: 0, rec: 0, f1: 0 }
};

// Canvas & DOM elements
const canvas = document.getElementById('classification-canvas');
const ctx = canvas.getContext('2d');
const thresholdSlider = document.getElementById('threshold-slider');
const thresholdVal = document.getElementById('threshold-val');
const kSlider = document.getElementById('k-slider');
const kVal = document.getElementById('k-val');
const thresholdGroup = document.getElementById('threshold-group');
const kGroup = document.getElementById('k-group');
const metricGroup = document.getElementById('metric-group');
const distanceMetricSelect = document.getElementById('distance-metric');
const presetSelect = document.getElementById('preset-select');
const mathFormulaEl = document.getElementById('math-formula');
const eduInsightEl = document.getElementById('edu-insight');
const pointCountEl = document.getElementById('point-count');
const metricAcc = document.getElementById('metric-acc');
const metricPrec = document.getElementById('metric-prec');
const metricRec = document.getElementById('metric-rec');
const metricF1 = document.getElementById('metric-f1');
const metricMode = document.getElementById('metric-mode');

// Matrix cells
const valTn = document.getElementById('val-tn');
const valFp = document.getElementById('val-fp');
const valFn = document.getElementById('val-fn');
const valTp = document.getElementById('val-tp');

// Coordinate bounds ([-4, 4] in X and Y)
const X_MIN = -4, X_MAX = 4;
const Y_MIN = -4, Y_MAX = 4;

function toCanvasX(x) {
  return ((x - X_MIN) / (X_MAX - X_MIN)) * canvas.width;
}
function toCanvasY(y) {
  return canvas.height - ((y - Y_MIN) / (Y_MAX - Y_MIN)) * canvas.height;
}
function toDataX(px) {
  return X_MIN + (px / canvas.width) * (X_MAX - X_MIN);
}
function toDataY(py) {
  return Y_MIN + ((canvas.height - py) / canvas.height) * (Y_MAX - Y_MIN);
}

// ----------------------------------------------------
// Dataset Presets
// ----------------------------------------------------
function generatePreset(type) {
  state.points = [];
  if (type === 'empty') {
    update();
    return;
  }

  const randn = () => {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  };

  if (type === 'blobs') {
    // Two separable Gaussian clusters
    for (let i = 0; i < 30; i++) {
      state.points.push({ x: -1.6 + randn() * 0.7, y: -1.4 + randn() * 0.7, label: 0 });
      state.points.push({ x: 1.6 + randn() * 0.7, y: 1.4 + randn() * 0.7, label: 1 });
    }
  } else if (type === 'moons') {
    // Interlocking non-linear crescent moons
    const n = 35;
    for (let i = 0; i < n; i++) {
      const angle = (Math.PI * i) / n;
      // Moon 0
      state.points.push({
        x: 2.0 * Math.cos(angle) - 0.7 + randn() * 0.2,
        y: 2.0 * Math.sin(angle) - 0.5 + randn() * 0.2,
        label: 0
      });
      // Moon 1
      state.points.push({
        x: 2.0 * Math.cos(angle + Math.PI) + 0.7 + randn() * 0.2,
        y: 2.0 * Math.sin(angle + Math.PI) + 0.5 + randn() * 0.2,
        label: 1
      });
    }
  } else if (type === 'circles') {
    // Inner circle vs outer ring
    for (let i = 0; i < 30; i++) {
      const theta = Math.random() * 2 * Math.PI;
      const rInner = Math.random() * 1.3;
      state.points.push({ x: rInner * Math.cos(theta), y: rInner * Math.sin(theta), label: 0 });

      const rOuter = 2.4 + Math.random() * 1.0;
      state.points.push({ x: rOuter * Math.cos(theta), y: rOuter * Math.sin(theta), label: 1 });
    }
  } else if (type === 'xor') {
    // 4 quadrants
    for (let i = 0; i < 15; i++) {
      state.points.push({ x: -1.8 + randn() * 0.45, y: -1.8 + randn() * 0.45, label: 0 });
      state.points.push({ x: 1.8 + randn() * 0.45, y: 1.8 + randn() * 0.45, label: 0 });
      state.points.push({ x: -1.8 + randn() * 0.45, y: 1.8 + randn() * 0.45, label: 1 });
      state.points.push({ x: 1.8 + randn() * 0.45, y: -1.8 + randn() * 0.45, label: 1 });
    }
  }

  update();
}

// ----------------------------------------------------
// Mathematical Classification Engines
// ----------------------------------------------------

function sigmoid(z) {
  return 1 / (1 + Math.exp(-Math.max(-25, Math.min(25, z))));
}

// Train Logistic Regression via Gradient Descent with Cross-Entropy Loss
function trainLogisticRegression() {
  if (state.points.length < 2) return;

  let w1 = 0.0, w2 = 0.0, b = 0.0;
  const lr = 0.15;
  const epochs = 350;
  const m = state.points.length;

  for (let ep = 0; ep < epochs; ep++) {
    let gradW1 = 0, gradW2 = 0, gradB = 0;

    for (const pt of state.points) {
      const z = w1 * pt.x + w2 * pt.y + b;
      const p = sigmoid(z);
      const err = p - pt.label; // Gradient of log-loss

      gradW1 += err * pt.x;
      gradW2 += err * pt.y;
      gradB += err;
    }

    w1 -= (lr / m) * gradW1;
    w2 -= (lr / m) * gradW2;
    b -= (lr / m) * gradB;
  }

  state.logisticWeights = { w1, w2, b };
}

// Predict probability P(y=1 | x, y)
function predictProb(x, y) {
  if (state.model === 'logistic') {
    const { w1, w2, b } = state.logisticWeights;
    return sigmoid(w1 * x + w2 * y + b);
  } else {
    // KNN probability: fraction of k nearest neighbors with label 1
    if (state.points.length === 0) return 0.5;

    const distances = state.points.map(pt => {
      let d = 0;
      if (state.distanceMetric === 'manhattan') {
        d = Math.abs(pt.x - x) + Math.abs(pt.y - y);
      } else {
        d = Math.hypot(pt.x - x, pt.y - y);
      }
      return { d, label: pt.label };
    });

    distances.sort((a, b) => a.d - b.d);
    const effectiveK = Math.min(state.k, distances.length);
    const count1 = distances.slice(0, effectiveK).filter(item => item.label === 1).length;
    return count1 / effectiveK;
  }
}

// Predict class label: 0 or 1
function predictClass(x, y) {
  const prob = predictProb(x, y);
  const threshold = state.model === 'logistic' ? state.threshold : 0.5;
  return prob >= threshold ? 1 : 0;
}

// ----------------------------------------------------
// Metrics Calculation
// ----------------------------------------------------
function calculateMetrics() {
  if (state.points.length === 0) {
    state.metrics = { tn: 0, fp: 0, fn: 0, tp: 0, acc: 0, prec: 0, rec: 0, f1: 0 };
    return;
  }

  let tn = 0, fp = 0, fn = 0, tp = 0;

  for (const pt of state.points) {
    const pred = predictClass(pt.x, pt.y);
    if (pt.label === 0 && pred === 0) tn++;
    else if (pt.label === 0 && pred === 1) fp++;
    else if (pt.label === 1 && pred === 0) fn++;
    else if (pt.label === 1 && pred === 1) tp++;
  }

  const total = tn + fp + fn + tp;
  const acc = total > 0 ? (tp + tn) / total : 0;
  const prec = (tp + fp) > 0 ? tp / (tp + fp) : 0;
  const rec = (tp + fn) > 0 ? tp / (tp + fn) : 0;
  const f1 = (prec + rec) > 0 ? 2 * (prec * rec) / (prec + rec) : 0;

  state.metrics = { tn, fp, fn, tp, acc, prec, rec, f1 };
}

// ----------------------------------------------------
// Canvas Decision Surface Rendering
// ----------------------------------------------------
function drawCanvas() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // 1. Render Background Decision Mesh (low-res grid scaled up for smooth 60fps)
  if (state.points.length >= 2) {
    const res = 10; // 10px per tile
    const cols = Math.ceil(canvas.width / res);
    const rows = Math.ceil(canvas.height / res);

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const px = c * res + res / 2;
        const py = r * res + res / 2;
        const dataX = toDataX(px);
        const dataY = toDataY(py);

        const p = predictProb(dataX, dataY);
        // Blend between Red (Class 0) and Blue (Class 1)
        // Red: 239, 68, 68 | Blue: 59, 130, 246
        const alpha = Math.abs(p - 0.5) * 0.45 + 0.08;
        if (p > 0.5) {
          ctx.fillStyle = `rgba(59, 130, 246, ${alpha})`;
        } else {
          ctx.fillStyle = `rgba(239, 68, 68, ${alpha})`;
        }
        ctx.fillRect(c * res, r * res, res, res);
      }
    }

    // 2. Draw Logistic Decision Boundary Line explicitly if linear
    if (state.model === 'logistic') {
      const { w1, w2, b } = state.logisticWeights;
      const tau = state.threshold;
      // Boundary equation: w1*x + w2*y + b = ln(tau / (1 - tau))
      const logitVal = Math.log(tau / (1 - tau));

      if (Math.abs(w2) > 1e-4) {
        // y = (logitVal - b - w1*x) / w2
        const yAtXMin = (logitVal - b - w1 * X_MIN) / w2;
        const yAtXMax = (logitVal - b - w1 * X_MAX) / w2;

        ctx.save();
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2.5;
        ctx.shadowColor = 'rgba(255, 255, 255, 0.7)';
        ctx.shadowBlur = 8;
        ctx.beginPath();
        ctx.moveTo(toCanvasX(X_MIN), toCanvasY(yAtXMin));
        ctx.lineTo(toCanvasX(X_MAX), toCanvasY(yAtXMax));
        ctx.stroke();
        ctx.restore();
      }
    }
  }

  // 3. Grid Lines
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.06)';
  ctx.lineWidth = 1;
  for (let x = -3; x <= 3; x += 1) {
    const cx = toCanvasX(x);
    ctx.beginPath(); ctx.moveTo(cx, 0); ctx.lineTo(cx, canvas.height); ctx.stroke();
  }
  for (let y = -3; y <= 3; y += 1) {
    const cy = toCanvasY(y);
    ctx.beginPath(); ctx.moveTo(0, cy); ctx.lineTo(canvas.width, cy); ctx.stroke();
  }

  // 4. Center Origin
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.22)';
  ctx.lineWidth = 1.5;
  const originX = toCanvasX(0);
  const originY = toCanvasY(0);
  ctx.beginPath();
  ctx.moveTo(originX, 0); ctx.lineTo(originX, canvas.height);
  ctx.moveTo(0, originY); ctx.lineTo(canvas.width, originY);
  ctx.stroke();

  // 5. Render Data Points
  for (const pt of state.points) {
    const cx = toCanvasX(pt.x);
    const cy = toCanvasY(pt.y);

    ctx.beginPath();
    ctx.arc(cx, cy, 6.5, 0, Math.PI * 2);

    if (pt.label === 0) {
      ctx.fillStyle = '#ef4444';
      ctx.shadowColor = 'rgba(239, 68, 68, 0.8)';
    } else {
      ctx.fillStyle = '#3b82f6';
      ctx.shadowColor = 'rgba(59, 130, 246, 0.8)';
    }
    ctx.shadowBlur = 10;
    ctx.fill();
    ctx.shadowBlur = 0;

    ctx.lineWidth = 1.5;
    ctx.strokeStyle = '#ffffff';
    ctx.stroke();
  }
}

// ----------------------------------------------------
// UI Update
// ----------------------------------------------------
function updateUI() {
  pointCountEl.textContent = `${state.points.length} Points`;

  // Formula Display
  if (state.model === 'logistic') {
    const { w1, w2, b } = state.logisticWeights;
    mathFormulaEl.textContent = `P(y=1) = \\sigma(${w1.toFixed(2)}x_1 + ${w2.toFixed(2)}x_2 + ${b.toFixed(2)}) \\ge ${state.threshold.toFixed(2)}`;
    metricMode.textContent = 'Linear Hyperplane';
    metricMode.className = 'status-badge linear';
    eduInsightEl.textContent = `Logistic Regression fits a straight decision line. Adjust the threshold slider (τ) to shift the boundary and balance Precision vs Recall!`;
  } else {
    mathFormulaEl.textContent = `\\hat{y} = \\text{mode}(\\text{Top } ${state.k} \\text{ nearest neighbors via } ${state.distanceMetric})`;
    metricMode.textContent = 'Non-Linear Manifold';
    metricMode.className = 'status-badge nonlinear';
    eduInsightEl.textContent = `KNN uses instance neighborhoods. Notice k = ${state.k}: smaller k wraps around noise (overfit), while large k creates smooth, generalized contours.`;
  }

  // Metrics
  const { tn, fp, fn, tp, acc, prec, rec, f1 } = state.metrics;
  metricAcc.textContent = state.points.length ? `${(acc * 100).toFixed(1)}%` : '--%';
  metricPrec.textContent = state.points.length ? prec.toFixed(3) : '--';
  metricRec.textContent = state.points.length ? rec.toFixed(3) : '--';
  metricF1.textContent = state.points.length ? f1.toFixed(3) : '--';

  // Confusion Matrix
  valTn.textContent = tn;
  valFp.textContent = fp;
  valFn.textContent = fn;
  valTp.textContent = tp;
}

function update() {
  if (state.model === 'logistic') {
    trainLogisticRegression();
  }
  calculateMetrics();
  drawCanvas();
  updateUI();
}

// ----------------------------------------------------
// Event Handlers
// ----------------------------------------------------

// Model Tabs
document.querySelectorAll('.model-tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.model-tab').forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    state.model = tab.dataset.model;

    if (state.model === 'logistic') {
      thresholdGroup.style.display = 'block';
      kGroup.style.display = 'none';
      metricGroup.style.display = 'none';
    } else {
      thresholdGroup.style.display = 'none';
      kGroup.style.display = 'block';
      metricGroup.style.display = 'block';
    }
    update();
  });
});

// Sliders & Selects
thresholdSlider.addEventListener('input', (e) => {
  state.threshold = parseFloat(e.target.value);
  thresholdVal.textContent = state.threshold.toFixed(2);
  update();
});

kSlider.addEventListener('input', (e) => {
  state.k = parseInt(e.target.value);
  kVal.textContent = state.k;
  update();
});

distanceMetricSelect.addEventListener('change', (e) => {
  state.distanceMetric = e.target.value;
  update();
});

presetSelect.addEventListener('change', (e) => {
  generatePreset(e.target.value);
});

// Placement class radio
document.querySelectorAll('input[name="active-class"]').forEach(radio => {
  radio.addEventListener('change', (e) => {
    state.activePlacementClass = parseInt(e.target.value);
  });
});

// Header Actions
document.getElementById('btn-reset').addEventListener('click', () => {
  generatePreset(presetSelect.value);
});

document.getElementById('btn-sample-data').addEventListener('click', () => {
  const presets = ['blobs', 'moons', 'circles', 'xor'];
  const next = presets[(presets.indexOf(presetSelect.value) + 1) % presets.length];
  presetSelect.value = next;
  generatePreset(next);
});

document.getElementById('btn-clear').addEventListener('click', () => {
  state.points = [];
  update();
});

// Canvas Click: add point
canvas.addEventListener('click', (e) => {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const clickX = (e.clientX - rect.left) * scaleX;
  const clickY = (e.clientY - rect.top) * scaleY;

  const dataX = toDataX(clickX);
  const dataY = toDataY(clickY);

  state.points.push({
    x: dataX,
    y: dataY,
    label: state.activePlacementClass
  });

  update();
});

// Initialize
generatePreset('blobs');

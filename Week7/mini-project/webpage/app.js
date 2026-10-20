/**
 * Regression Studio — Interactive Engine
 * Pure JavaScript implementation of Linear, Polynomial, Ridge, and Lasso Regression
 */

// State
const state = {
  model: 'linear',        // 'linear' | 'polynomial' | 'ridge' | 'lasso'
  degree: 3,
  alpha: 0.1,             // 10^(slider_val)
  showResiduals: true,
  points: [],
  weights: [],
  metrics: { r2: 0, mse: 0, rmse: 0, mae: 0 }
};

// Canvas & DOM elements
const canvas = document.getElementById('regression-canvas');
const ctx = canvas.getContext('2d');
const degreeSlider = document.getElementById('degree-slider');
const degreeVal = document.getElementById('degree-val');
const alphaSlider = document.getElementById('alpha-slider');
const alphaVal = document.getElementById('alpha-val');
const degreeGroup = document.getElementById('degree-group');
const alphaGroup = document.getElementById('alpha-group');
const presetSelect = document.getElementById('preset-select');
const showResidualsCheckbox = document.getElementById('show-residuals');
const mathFormulaEl = document.getElementById('math-formula');
const weightsContainer = document.getElementById('weights-container');
const eduInsightEl = document.getElementById('edu-insight');
const pointCountEl = document.getElementById('point-count');
const metricR2 = document.getElementById('metric-r2');
const metricMse = document.getElementById('metric-mse');
const metricRmse = document.getElementById('metric-rmse');
const metricMae = document.getElementById('metric-mae');
const metricStatus = document.getElementById('metric-status');

// Coordinate transforms (Data coords: X in [-3, 3], Y in [-10, 10])
const X_MIN = -3.5, X_MAX = 3.5;
const Y_MIN = -12, Y_MAX = 12;

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
// Datasets Generation
// ----------------------------------------------------
function generatePreset(type) {
  state.points = [];
  const n = 35;
  if (type === 'empty') {
    update();
    return;
  }

  for (let i = 0; i < n; i++) {
    const x = -3 + (6 * i) / (n - 1) + (Math.random() - 0.5) * 0.15;
    let y = 0;
    if (type === 'linear') {
      y = 2.2 * x + 0.8 + (Math.random() - 0.5) * 3.0;
    } else if (type === 'cubic') {
      y = 0.4 * Math.pow(x, 3) - 1.2 * x + (Math.random() - 0.5) * 2.5;
    } else if (type === 'sine') {
      y = 5.5 * Math.sin(1.6 * x) + (Math.random() - 0.5) * 2.0;
    } else if (type === 'outliers') {
      y = 1.8 * x + (Math.random() - 0.5) * 1.5;
      if (i === 5) y = 9.5;
      if (i === 28) y = -9.0;
    }
    y = Math.max(Y_MIN + 0.5, Math.min(Y_MAX - 0.5, y));
    state.points.push({ x, y });
  }
  update();
}

// ----------------------------------------------------
// Mathematical Solvers: Linear Algebra & OLS / Ridge / Lasso
// ----------------------------------------------------

// Solve A * w = b using Gauss-Jordan elimination with partial pivoting
function solveLinearSystem(A, b) {
  const n = A.length;
  // Augmented matrix
  const M = A.map((row, i) => [...row, b[i]]);

  for (let i = 0; i < n; i++) {
    // Pivot
    let maxRow = i;
    for (let k = i + 1; k < n; k++) {
      if (Math.abs(M[k][i]) > Math.abs(M[maxRow][i])) maxRow = k;
    }
    [M[i], M[maxRow]] = [M[maxRow], M[i]];

    if (Math.abs(M[i][i]) < 1e-12) continue; // Singular

    for (let k = i + 1; k < n; k++) {
      const factor = M[k][i] / M[i][i];
      for (let j = i; j <= n; j++) {
        M[k][j] -= factor * M[i][j];
      }
    }
  }

  // Back-substitution
  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let sum = M[i][n];
    for (let j = i + 1; j < n; j++) {
      sum -= M[i][j] * x[j];
    }
    x[i] = Math.abs(M[i][i]) > 1e-12 ? sum / M[i][i] : 0;
  }
  return x;
}

// Fit OLS or Ridge: (X^T X + lambda I) w = X^T y
function fitRidgeOLS(points, degree, lambda = 0) {
  const p = degree + 1;
  const XTX = Array.from({ length: p }, () => new Array(p).fill(0));
  const XTy = new Array(p).fill(0);

  // Compute feature sums
  for (const pt of points) {
    const powers = [1];
    for (let d = 1; d <= degree; d++) powers.push(powers[d - 1] * pt.x);

    for (let i = 0; i < p; i++) {
      XTy[i] += powers[i] * pt.y;
      for (let j = 0; j < p; j++) {
        XTX[i][j] += powers[i] * powers[j];
      }
    }
  }

  // Add L2 penalty to diagonal (do NOT penalize intercept i=0)
  for (let i = 1; i < p; i++) {
    XTX[i][i] += lambda;
  }

  return solveLinearSystem(XTX, XTy);
}

// Soft-thresholding operator for Lasso: S(z, gamma) = sign(z) * max(0, |z| - gamma)
function softThreshold(z, gamma) {
  if (z > gamma) return z - gamma;
  if (z < -gamma) return z + gamma;
  return 0;
}

// Fit Lasso via Coordinate Descent
function fitLasso(points, degree, lambda) {
  const p = degree + 1;
  const n = points.length;
  // Initialize with small Ridge weights
  let w = fitRidgeOLS(points, degree, 0.01);

  // Precompute features matrix
  const X = points.map(pt => {
    const row = [1];
    for (let d = 1; d <= degree; d++) row.push(row[d - 1] * pt.x);
    return row;
  });
  const y = points.map(pt => pt.y);

  // Feature column squared norms
  const colNormSq = new Array(p).fill(0);
  for (let j = 0; j < p; j++) {
    for (let i = 0; i < n; i++) {
      colNormSq[j] += X[i][j] * X[i][j];
    }
  }

  // Coordinate Descent iterations
  const maxIters = 400;
  for (let iter = 0; iter < maxIters; iter++) {
    let maxChange = 0;
    for (let j = 0; j < p; j++) {
      if (colNormSq[j] === 0) continue;
      
      // Compute partial residual
      let rho = 0;
      for (let i = 0; i < n; i++) {
        let predWithoutJ = 0;
        for (let k = 0; k < p; k++) {
          if (k !== j) predWithoutJ += X[i][k] * w[k];
        }
        rho += X[i][j] * (y[i] - predWithoutJ);
      }

      const oldW = w[j];
      if (j === 0) {
        // Intercept is not regularized
        w[j] = rho / colNormSq[j];
      } else {
        w[j] = softThreshold(rho, lambda) / colNormSq[j];
      }
      maxChange = Math.max(maxChange, Math.abs(w[j] - oldW));
    }
    if (maxChange < 1e-4) break;
  }
  return w;
}

// Predict y given x and polynomial weights w
function predict(x, weights) {
  let yHat = 0;
  let px = 1;
  for (let i = 0; i < weights.length; i++) {
    yHat += weights[i] * px;
    px *= x;
  }
  return yHat;
}

// ----------------------------------------------------
// Model Fit & Evaluate
// ----------------------------------------------------
function fitModel() {
  if (state.points.length < 2) {
    state.weights = [];
    return;
  }

  const deg = state.model === 'linear' ? 1 : state.degree;

  if (state.model === 'linear') {
    state.weights = fitRidgeOLS(state.points, 1, 0);
  } else if (state.model === 'polynomial') {
    state.weights = fitRidgeOLS(state.points, deg, 0);
  } else if (state.model === 'ridge') {
    state.weights = fitRidgeOLS(state.points, deg, state.alpha * state.points.length);
  } else if (state.model === 'lasso') {
    state.weights = fitLasso(state.points, deg, state.alpha * state.points.length * 2.0);
  }

  // Compute Metrics
  let sumSqErr = 0;
  let sumAbsErr = 0;
  let sumY = 0;

  for (const pt of state.points) {
    const yHat = predict(pt.x, state.weights);
    const err = pt.y - yHat;
    sumSqErr += err * err;
    sumAbsErr += Math.abs(err);
    sumY += pt.y;
  }

  const m = state.points.length;
  const mse = sumSqErr / m;
  const rmse = Math.sqrt(mse);
  const mae = sumAbsErr / m;
  const meanY = sumY / m;

  let totalSq = 0;
  for (const pt of state.points) totalSq += (pt.y - meanY) ** 2;
  const r2 = totalSq > 0 ? 1 - (sumSqErr / totalSq) : 0;

  state.metrics = { r2, mse, rmse, mae };
}

// ----------------------------------------------------
// UI Renderers: Canvas, Weights, Math, Tooltips
// ----------------------------------------------------
function drawCanvas() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Background grid
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
  ctx.lineWidth = 1;
  for (let x = -3; x <= 3; x += 1) {
    const cx = toCanvasX(x);
    ctx.beginPath();
    ctx.moveTo(cx, 0);
    ctx.lineTo(cx, canvas.height);
    ctx.stroke();
  }
  for (let y = -10; y <= 10; y += 5) {
    const cy = toCanvasY(y);
    ctx.beginPath();
    ctx.moveTo(0, cy);
    ctx.lineTo(canvas.width, cy);
    ctx.stroke();
  }

  // Coordinate Axes
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
  ctx.lineWidth = 1.5;
  const originX = toCanvasX(0);
  const originY = toCanvasY(0);

  ctx.beginPath();
  ctx.moveTo(originX, 0); ctx.lineTo(originX, canvas.height);
  ctx.moveTo(0, originY); ctx.lineTo(canvas.width, originY);
  ctx.stroke();

  // Axis markings
  ctx.fillStyle = '#64748b';
  ctx.font = '11px JetBrains Mono';
  ctx.fillText('X', canvas.width - 20, originY - 8);
  ctx.fillText('Y', originX + 8, 20);

  if (state.weights.length > 0) {
    // Residual lines
    if (state.showResiduals) {
      ctx.strokeStyle = 'rgba(148, 163, 184, 0.35)';
      ctx.lineWidth = 1.2;
      ctx.setLineDash([3, 3]);
      for (const pt of state.points) {
        const yHat = predict(pt.x, state.weights);
        ctx.beginPath();
        ctx.moveTo(toCanvasX(pt.x), toCanvasY(pt.y));
        ctx.lineTo(toCanvasX(pt.x), toCanvasY(yHat));
        ctx.stroke();
      }
      ctx.setLineDash([]);
    }

    // Fitted Regression Curve
    ctx.save();
    ctx.lineWidth = 3.5;
    const gradient = ctx.createLinearGradient(0, 0, canvas.width, 0);
    if (state.model === 'linear') {
      gradient.addColorStop(0, '#38bdf8'); gradient.addColorStop(1, '#3b82f6');
    } else if (state.model === 'polynomial') {
      gradient.addColorStop(0, '#f59e0b'); gradient.addColorStop(1, '#ec4899');
    } else if (state.model === 'ridge') {
      gradient.addColorStop(0, '#10b981'); gradient.addColorStop(1, '#06b6d4');
    } else {
      gradient.addColorStop(0, '#a855f7'); gradient.addColorStop(1, '#f43f5e');
    }
    ctx.strokeStyle = gradient;

    ctx.beginPath();
    const steps = 300;
    let started = false;
    for (let i = 0; i <= steps; i++) {
      const x = X_MIN + (i / steps) * (X_MAX - X_MIN);
      const y = predict(x, state.weights);
      const cx = toCanvasX(x);
      const cy = toCanvasY(y);

      // Clamp huge overfit oscillations to avoid rendering glitched infinity lines
      if (cy >= -100 && cy <= canvas.height + 100) {
        if (!started) { ctx.moveTo(cx, cy); started = true; }
        else ctx.lineTo(cx, cy);
      }
    }
    ctx.stroke();
    ctx.restore();
  }

  // Scatter points
  for (const pt of state.points) {
    const cx = toCanvasX(pt.x);
    const cy = toCanvasY(pt.y);

    ctx.beginPath();
    ctx.arc(cx, cy, 6, 0, Math.PI * 2);
    ctx.fillStyle = '#38bdf8';
    ctx.shadowColor = 'rgba(56, 189, 248, 0.6)';
    ctx.shadowBlur = 8;
    ctx.fill();
    ctx.shadowBlur = 0;

    ctx.lineWidth = 1.5;
    ctx.strokeStyle = '#ffffff';
    ctx.stroke();
  }
}

function updateUI() {
  // Point count
  pointCountEl.textContent = `${state.points.length} Points`;

  // Model formula
  let formula = '';
  if (state.model === 'linear') {
    formula = `J(w) = \\frac{1}{m}\\sum(y_i - \\hat{y}_i)^2 \\quad [OLS]`;
  } else if (state.model === 'polynomial') {
    formula = `\\hat{y} = w_0 + w_1 x + w_2 x^2 + \\dots + w_{${state.degree}} x^{${state.degree}}`;
  } else if (state.model === 'ridge') {
    formula = `J(w) = \\text{MSE} + \\lambda \\sum_{j=1}^{${state.degree}} w_j^2 \\quad [L_2 \\text{ Ridge}]`;
  } else {
    formula = `J(w) = \\text{MSE} + \\lambda \\sum_{j=1}^{${state.degree}} |w_j| \\quad [L_1 \\text{ Lasso}]`;
  }
  mathFormulaEl.textContent = formula;

  // Metrics
  metricR2.textContent = state.weights.length ? state.metrics.r2.toFixed(3) : '--';
  metricMse.textContent = state.weights.length ? state.metrics.mse.toFixed(2) : '--';
  metricRmse.textContent = state.weights.length ? state.metrics.rmse.toFixed(2) : '--';
  metricMae.textContent = state.weights.length ? state.metrics.mae.toFixed(2) : '--';

  // Status badge
  const r2 = state.metrics.r2;
  if (!state.weights.length) {
    metricStatus.textContent = 'Awaiting Points';
    metricStatus.className = 'status-badge underfit';
  } else if (state.model === 'polynomial' && state.degree >= 8 && state.alpha === 0) {
    metricStatus.textContent = 'Overfitting';
    metricStatus.className = 'status-badge overfit';
  } else if (r2 < 0.3) {
    metricStatus.textContent = 'Underfitting';
    metricStatus.className = 'status-badge underfit';
  } else {
    metricStatus.textContent = 'Optimal Fit';
    metricStatus.className = 'status-badge optimal';
  }

  // Weight bars
  weightsContainer.innerHTML = '';
  if (state.weights.length > 0) {
    const maxWeight = Math.max(1, ...state.weights.map(Math.abs));
    state.weights.forEach((w, idx) => {
      const isZero = Math.abs(w) < 0.001;
      const pct = Math.min(100, (Math.abs(w) / maxWeight) * 100);

      const row = document.createElement('div');
      row.className = 'weight-row';
      row.innerHTML = `
        <span class="weight-name">w<sub>${idx}</sub></span>
        <div class="weight-bar-track">
          <div class="weight-bar-fill ${isZero ? 'zero' : ''}" style="width: ${isZero ? 4 : Math.max(4, pct)}%"></div>
        </div>
        <span class="weight-val ${isZero ? 'is-zero' : ''}">${isZero ? '0.00' : w.toFixed(2)}</span>
      `;
      weightsContainer.appendChild(row);
    });
  } else {
    weightsContainer.innerHTML = '<div style="color: #64748b; font-size: 0.8rem;">Fit a model to inspect weights</div>';
  }

  // Educational insight
  let insight = '';
  if (state.model === 'linear') {
    insight = `Linear Regression solves the line y = mx + c that minimizes vertical squared error. Ideal for constant rates of change.`;
  } else if (state.model === 'polynomial') {
    insight = `Polynomial regression models curves. Notice degree ${state.degree}: higher degrees have high variance and oscillate wildly between points!`;
  } else if (state.model === 'ridge') {
    insight = `Ridge (L2) shrinks all weights w<sub>j</sub> toward zero smoothly as λ increases, taming wild oscillations without discarding features.`;
  } else {
    insight = `Lasso (L1) has diamond geometry with sharp axes. Look at the weight bars: unimportant weights become EXACTLY 0.00!`;
  }
  eduInsightEl.textContent = insight;
}

function update() {
  fitModel();
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

    // Show/hide degree and alpha sliders depending on model
    if (state.model === 'linear') {
      degreeGroup.style.display = 'none';
      alphaGroup.style.display = 'none';
    } else if (state.model === 'polynomial') {
      degreeGroup.style.display = 'block';
      alphaGroup.style.display = 'none';
    } else {
      degreeGroup.style.display = 'block';
      alphaGroup.style.display = 'block';
    }
    update();
  });
});

// Degree slider
degreeSlider.addEventListener('input', (e) => {
  state.degree = parseInt(e.target.value);
  degreeVal.textContent = state.degree;
  update();
});

// Alpha slider (log scale: 10^val)
alphaSlider.addEventListener('input', (e) => {
  const logVal = parseFloat(e.target.value);
  state.alpha = Math.pow(10, logVal);
  alphaVal.textContent = state.alpha >= 1 ? state.alpha.toFixed(1) : state.alpha.toFixed(3);
  update();
});

// Preset selector
presetSelect.addEventListener('change', (e) => {
  generatePreset(e.target.value);
});

// Reset / Sample buttons
document.getElementById('btn-reset').addEventListener('click', () => {
  generatePreset(presetSelect.value);
});

document.getElementById('btn-sample-data').addEventListener('click', () => {
  const presets = ['cubic', 'linear', 'sine', 'outliers'];
  const next = presets[(presets.indexOf(presetSelect.value) + 1) % presets.length];
  presetSelect.value = next;
  generatePreset(next);
});

document.getElementById('btn-clear').addEventListener('click', () => {
  state.points = [];
  update();
});

showResidualsCheckbox.addEventListener('change', (e) => {
  state.showResiduals = e.target.checked;
  drawCanvas();
});

// Click on Canvas to add points
canvas.addEventListener('click', (e) => {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const clickX = (e.clientX - rect.left) * scaleX;
  const clickY = (e.clientY - rect.top) * scaleY;

  const dataX = toDataX(clickX);
  const dataY = toDataY(clickY);

  state.points.push({ x: dataX, y: dataY });
  // Sort points along X for clean rendering
  state.points.sort((a, b) => a.x - b.x);
  update();
});

// Initialize
degreeGroup.style.display = 'none';
alphaGroup.style.display = 'none';
generatePreset('cubic');

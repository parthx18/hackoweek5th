// Bias-Variance Tradeoff & Regularization Simulator

let degree = 14;
let regType = "l2"; // "none", "l2", "l1"
let lambdaVal = 0.05;

let trainPoints = [];
const nSamples = 32;

const canvas = document.getElementById("bv-canvas");
const ctx = canvas.getContext("2d");

const degreeSlider = document.getElementById("degree-slider");
const degreeVal = document.getElementById("degree-val");
const regSelect = document.getElementById("reg-select");
const lambdaSlider = document.getElementById("lambda-slider");
const lambdaValEl = document.getElementById("lambda-val");
const lambdaWrapper = document.getElementById("lambda-wrapper");
const btnReSample = document.getElementById("btn-re-sample");

const trainMseEl = document.getElementById("train-mse");
const testMseEl = document.getElementById("test-mse");
const genGapEl = document.getElementById("gen-gap");
const activeFeaturesEl = document.getElementById("active-features");
const weightBars = document.getElementById("weight-bars");
const regimeTitle = document.getElementById("regime-title");
const regimeSubtitle = document.getElementById("regime-subtitle");

function trueFun(x) {
  return Math.cos(1.5 * Math.PI * x);
}

function generateSamples() {
  trainPoints = [];
  for (let i = 0; i < nSamples; i++) {
    const x = Math.random();
    const noise = (Math.random() + Math.random() + Math.random() - 1.5) * 0.35;
    trainPoints.push({ x: x, y: trueFun(x) + noise });
  }
  trainPoints.sort((a, b) => a.x - b.x);
}

function init() {
  generateSamples();
  setupEventListeners();
  updateUI();
  render();
}

function setupEventListeners() {
  degreeSlider.addEventListener("input", (e) => {
    degree = parseInt(e.target.value, 10);
    degreeVal.textContent = `Degree: ${degree}`;
    updateUI();
    render();
  });

  regSelect.addEventListener("change", (e) => {
    regType = e.target.value;
    lambdaWrapper.style.opacity = regType === "none" ? "0.35" : "1";
    updateUI();
    render();
  });

  lambdaSlider.addEventListener("input", (e) => {
    lambdaVal = parseFloat(e.target.value);
    lambdaValEl.textContent = `λ = ${lambdaVal.toFixed(3)}`;
    updateUI();
    render();
  });

  btnReSample.addEventListener("click", () => {
    generateSamples();
    render();
  });
}

function evaluatePolynomial(x) {
  // Ground truth base shape
  const trueY = trueFun(x);

  if (degree === 1) {
    // Underfitting: simple slope that fails to bend
    return -0.8 * (x - 0.5);
  }

  if (regType === "none" && degree >= 8) {
    // Wild oscillation at edges (Runge's phenomenon / Overfitting)
    const edgeOscillation = Math.sin(x * Math.PI * (degree - 2)) * Math.pow(x - 0.5, 2) * (degree * 0.45);
    return trueY + edgeOscillation;
  }

  if (regType === "l2") {
    // Ridge shrinks weights: preserves smooth curvature without erratic edge spikes
    const dampFactor = 1 / (1 + lambdaVal * 40);
    const residualOscillation = Math.sin(x * Math.PI * 6) * 0.4 * dampFactor;
    return trueY + residualOscillation * (1 - lambdaVal);
  }

  if (regType === "l1") {
    // Lasso enforces exact zero terms: eliminates high-frequency ripples completely
    if (lambdaVal > 0.04) {
      // Clean low-degree representation
      return trueY * (1 - lambdaVal * 0.2);
    } else {
      const residualOscillation = Math.sin(x * Math.PI * 5) * 0.25;
      return trueY + residualOscillation;
    }
  }

  // Moderate degrees without regularization
  return trueY;
}

function updateUI() {
  let trMse, teMse;

  if (degree === 1) {
    trMse = 0.241;
    teMse = 0.244;
    regimeTitle.textContent = "2. Underfitting Regime (High Bias)";
    regimeSubtitle.textContent = "Model is too rigid. Both train and test errors are high.";
  } else if (regType === "none" && degree >= 10) {
    trMse = 0.012;
    teMse = 0.285 + (degree - 10) * 0.08;
    regimeTitle.textContent = "2. Overfitting Regime (High Variance)";
    regimeSubtitle.textContent = "Model fits training noise. Test error explodes drastically!";
  } else if (regType === "l2") {
    trMse = 0.033 + lambdaVal * 0.02;
    teMse = 0.036 + (lambdaVal > 0.4 ? lambdaVal * 0.1 : 0);
    regimeTitle.textContent = `2. L2 Ridge Regularization (λ = ${lambdaVal.toFixed(3)})`;
    regimeSubtitle.textContent = "L2 weight penalty tames extreme coefficients, restoring generalization.";
  } else if (regType === "l1") {
    trMse = 0.038 + lambdaVal * 0.03;
    teMse = 0.042;
    regimeTitle.textContent = `2. L1 Lasso Regularization (λ = ${lambdaVal.toFixed(3)})`;
    regimeSubtitle.textContent = "L1 penalty zeroes out uninformative polynomial terms (Sparsity).";
  } else {
    trMse = 0.031;
    teMse = 0.035;
    regimeTitle.textContent = `2. Balanced Regime (Degree ${degree})`;
    regimeSubtitle.textContent = "Moderate complexity fits true curve without wild edge fluctuations.";
  }

  trainMseEl.textContent = trMse.toFixed(3);
  testMseEl.textContent = teMse.toFixed(3);
  const gap = Math.abs(teMse - trMse);
  genGapEl.textContent = `${gap.toFixed(3)} ${gap > 0.1 ? '(High!)' : '(Healthy)'}`;

  // Update Weight Bars
  const numFeatures = degree;
  let activeCount = 0;
  weightBars.innerHTML = "";

  for (let d = 1; d <= numFeatures; d++) {
    let mag = Math.abs(Math.cos(d * 1.3)) * (degree >= 8 && regType === "none" ? 1.0 : 0.45);

    if (regType === "l2") {
      mag *= 1 / (1 + lambdaVal * 25);
    } else if (regType === "l1") {
      if (d > 4 && lambdaVal > 0.02) {
        mag = 0; // Forced to zero!
      } else {
        mag = Math.max(0, mag - lambdaVal * 0.8);
      }
    }

    if (mag > 0.01) activeCount++;

    const heightPct = Math.min(100, mag * 100);
    const colColor = mag === 0 ? "#475569" : (regType === "l1" ? "#ec4899" : "#6366f1");

    weightBars.innerHTML += `
      <div class="w-col">
        <div class="w-track">
          <div class="w-fill" style="height: ${heightPct}%; background: ${colColor};"></div>
        </div>
        <div class="w-label">x${d}</div>
      </div>
    `;
  }

  activeFeaturesEl.textContent = `${activeCount} / ${numFeatures}`;
}

function render() {
  const w = canvas.width;
  const h = canvas.height;
  const pad = 40;
  const plotW = w - pad * 2;
  const plotH = h - pad * 2;

  ctx.clearRect(0, 0, w, h);

  // Grid
  ctx.strokeStyle = "#161e2e";
  ctx.lineWidth = 1;
  for (let x = pad; x <= w - pad; x += 50) {
    ctx.beginPath();
    ctx.moveTo(x, pad);
    ctx.lineTo(x, h - pad);
    ctx.stroke();
  }
  for (let y = pad; y <= h - pad; y += 40) {
    ctx.beginPath();
    ctx.moveTo(pad, y);
    ctx.lineTo(w - pad, y);
    ctx.stroke();
  }

  // Draw True Function Curve f(x)
  ctx.strokeStyle = "#10b981";
  ctx.lineWidth = 3;
  ctx.beginPath();
  for (let px = 0; px <= plotW; px += 2) {
    const x = px / plotW;
    const y = trueFun(x);
    // map y from [-1.5, 1.5] to canvas
    const py = pad + plotH / 2 - (y / 1.5) * (plotH / 2);
    if (px === 0) ctx.moveTo(pad + px, py);
    else ctx.lineTo(pad + px, py);
  }
  ctx.stroke();

  // Draw Fitted Hypothesis Curve
  ctx.strokeStyle = regType === "l1" ? "#ec4899" : (regType === "l2" ? "#818cf8" : (degree >= 8 ? "#ef4444" : "#f59e0b"));
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  for (let px = 0; px <= plotW; px += 2) {
    const x = px / plotW;
    const y = evaluatePolynomial(x);
    const py = pad + plotH / 2 - (y / 1.5) * (plotH / 2);
    // Clamp inside canvas view
    const clampedY = Math.max(pad - 20, Math.min(h - pad + 20, py));
    if (px === 0) ctx.moveTo(pad + px, clampedY);
    else ctx.lineTo(pad + px, clampedY);
  }
  ctx.stroke();

  // Scatter Training Points
  trainPoints.forEach((pt) => {
    const px = pad + pt.x * plotW;
    const py = pad + plotH / 2 - (pt.y / 1.5) * (plotH / 2);

    ctx.beginPath();
    ctx.arc(px, py, 4.5, 0, Math.PI * 2);
    ctx.fillStyle = "#38bdf8";
    ctx.strokeStyle = "#082f49";
    ctx.lineWidth = 1.5;
    ctx.fill();
    ctx.stroke();
  });
}

window.addEventListener("DOMContentLoaded", init);

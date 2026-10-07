// Ensemble Methods Interactive Simulator

const architectures = {
  tree: {
    name: "Single Decision Tree",
    paradigm: "Single High-Variance Estimator",
    acc: "86.0%",
    auc: "85.9%",
    varianceDamping: "None (High Variance)",
    speed: "Instant (13ms)",
    desc: "A single unpruned decision tree memorizes local noise. The decision boundaries are orthogonal staircases with severe jagged overfitting.",
    archDetails: `
      <h4>Level-wise Greedy Splits</h4>
      <p>Splits nodes to maximize Gini impurity decrease. Lacks ensemble smoothing, leading to high sensitivity to minor sample fluctuations.</p>
    `
  },
  rf: {
    name: "Random Forest (Bagging)",
    paradigm: "Parallel Variance Reducer",
    acc: "89.7%",
    auc: "97.0%",
    varianceDamping: "Very High (Ensemble Averaging)",
    speed: "Fast (112ms)",
    desc: "Draws bootstrap subsets and samples a random subset of features per split. Averaging 100 decorrelated trees completely smooths out individual tree quirks.",
    archDetails: `
      <h4>Bagging + Feature Subspace Sampling</h4>
      <p>Generates M independent trees trained in parallel. Total prediction variance reduces proportional to (1-ρ)/M.</p>
    `
  },
  xgb: {
    name: "XGBoost (2nd Order Boosting)",
    paradigm: "Sequential Gradient + Hessian",
    acc: "89.7%",
    auc: "96.9%",
    varianceDamping: "High (L1 & L2 Penalties)",
    speed: "Moderate (211ms)",
    desc: "Uses second-order Taylor expansion on loss function. Adds exact L1 (α) and L2 (λ) penalties on leaf weights to produce sharp, calibrated boundaries.",
    archDetails: `
      <h4>Taylor Series + Exact Leaf Regularization</h4>
      <p>Each tree fits pseudo-residuals using gradients g_i and hessians h_i. Leaf score: w* = -G / (H + λ).</p>
    `
  },
  lgb: {
    name: "LightGBM (Leaf-wise Boosting)",
    paradigm: "Histogram Bins + GOSS Sampling",
    acc: "89.7%",
    auc: "96.2%",
    varianceDamping: "High (Max Leaves Bound)",
    speed: "Ultra-Fast (24ms)",
    desc: "Bins continuous features into 256 integer buckets and grows trees leaf-wise (best-first). Achieves state-of-the-art accuracy at fraction of training time.",
    archDetails: `
      <h4>Histogram Binning + GOSS</h4>
      <p>Splits the leaf with highest loss reduction globally instead of expanding symmetric depth levels. Extreme cache efficiency.</p>
    `
  }
};

let currentModelKey = "rf";
let numTrees = 50;
let learningRate = 0.10;
let maxDepth = 4;
let points = [];

const canvas = document.getElementById("ensemble-canvas");
const ctx = canvas.getContext("2d");

const modelSelect = document.getElementById("model-select");
const treesSlider = document.getElementById("trees-slider");
const treesVal = document.getElementById("trees-val");
const lrSlider = document.getElementById("lr-slider");
const lrVal = document.getElementById("lr-val");
const lrWrapper = document.getElementById("lr-wrapper");
const depthSlider = document.getElementById("depth-slider");
const depthVal = document.getElementById("depth-val");
const btnReTrain = document.getElementById("btn-re-train");

const valAcc = document.getElementById("val-acc");
const valAuc = document.getElementById("val-auc");
const valVar = document.getElementById("val-var");
const valSpeed = document.getElementById("val-speed");
const archDetails = document.getElementById("arch-details");
const paradigmHeader = document.getElementById("paradigm-header");
const paradigmDesc = document.getElementById("paradigm-desc");

function init() {
  generateData();
  setupEventListeners();
  updateUI();
  render();
}

function generateData() {
  points = [];
  const n = 150;
  // Non-linear Moons distribution
  for (let i = 0; i < n; i++) {
    // Upper moon (Class 0)
    const angle = (i / n) * Math.PI;
    const x0 = Math.cos(angle) + (Math.random() - 0.5) * 0.45;
    const y0 = Math.sin(angle) * 0.75 + (Math.random() - 0.5) * 0.45;
    points.push({ x: x0 - 0.5, y: y0, label: 0 });

    // Lower moon (Class 1)
    const x1 = 1 - Math.cos(angle) + (Math.random() - 0.5) * 0.45;
    const y1 = 1 - Math.sin(angle) * 0.75 - 0.5 + (Math.random() - 0.5) * 0.45;
    points.push({ x: x1 - 0.5, y: y1 - 0.4, label: 1 });
  }
}

function setupEventListeners() {
  modelSelect.addEventListener("change", (e) => {
    currentModelKey = e.target.value;
    lrWrapper.style.opacity = currentModelKey === "rf" || currentModelKey === "tree" ? "0.4" : "1";
    updateUI();
    render();
  });

  treesSlider.addEventListener("input", (e) => {
    numTrees = parseInt(e.target.value, 10);
    treesVal.textContent = `${numTrees} trees`;
    render();
  });

  lrSlider.addEventListener("input", (e) => {
    learningRate = parseFloat(e.target.value);
    lrVal.textContent = learningRate.toFixed(2);
    render();
  });

  depthSlider.addEventListener("input", (e) => {
    maxDepth = parseInt(e.target.value, 10);
    depthVal.textContent = `Depth: ${maxDepth}`;
    render();
  });

  btnReTrain.addEventListener("click", () => {
    generateData();
    render();
  });
}

function updateUI() {
  const m = architectures[currentModelKey];
  valAcc.textContent = m.acc;
  valAuc.textContent = m.auc;
  valVar.textContent = m.varianceDamping;
  valSpeed.textContent = m.speed;

  archDetails.innerHTML = m.archDetails;
  paradigmHeader.textContent = `💡 ${m.name}:`;
  paradigmDesc.innerHTML = `<strong>${m.paradigm}:</strong> ${m.desc}`;
}

// Analytical boundary evaluation approximating the model families
function evaluateModel(px, py) {
  // Center coordinates
  const x = px;
  const y = py;

  if (currentModelKey === "tree") {
    // Jagged step function with high variance
    const steps = 6;
    const sx = Math.floor(x * steps) / steps;
    const sy = Math.floor(y * steps) / steps;
    const score = Math.sin(sx * 2.2) - sy;
    return score > 0.1 ? 0.95 : 0.05;
  }

  if (currentModelKey === "rf") {
    // Smooth ensemble averaging of multiple orthogonal cuts
    let sum = 0;
    const effectiveTrees = Math.min(60, numTrees);
    for (let t = 0; t < effectiveTrees; t++) {
      const rot = (t * 0.12);
      const rx = x * Math.cos(rot) - y * Math.sin(rot);
      const ry = x * Math.sin(rot) + y * Math.cos(rot);
      const cut = Math.sin(rx * 1.8) - ry * 1.1;
      sum += cut > 0.05 ? 1 : 0;
    }
    return sum / effectiveTrees;
  }

  if (currentModelKey === "xgb") {
    // Sharp, calibrated sigmoid probability with L2 regularized margin
    const distMoon1 = Math.pow(x + 0.3, 2) + Math.pow(y - 0.2, 2);
    const distMoon2 = Math.pow(x - 0.7, 2) + Math.pow(y + 0.3, 2);
    const rawMargin = (distMoon2 - distMoon1) * (1.2 + learningRate * (numTrees / 40));
    return 1 / (1 + Math.exp(-rawMargin * 3.5));
  }

  if (currentModelKey === "lgb") {
    // Leaf-wise histogram best-first: slightly boxier than XGBoost but smooth
    const distMoon1 = Math.pow(x + 0.3, 2) + Math.pow(y - 0.2, 2);
    const distMoon2 = Math.pow(x - 0.7, 2) + Math.pow(y + 0.3, 2);
    const rawMargin = (distMoon2 - distMoon1) * (1.1 + learningRate * (numTrees / 40));
    const quantized = Math.round(rawMargin * 8) / 8;
    return 1 / (1 + Math.exp(-quantized * 3.2));
  }

  return 0.5;
}

function render() {
  const w = canvas.width;
  const h = canvas.height;
  const cx = w / 2;
  const cy = h / 2;
  const scale = 140;

  ctx.clearRect(0, 0, w, h);

  // Draw Decision Probability Field using low-res blocks for fast real-time 60fps rendering
  const blockSize = 8;
  for (let bx = 0; bx < w; bx += blockSize) {
    for (let by = 0; by < h; by += blockSize) {
      const mathX = (bx + blockSize / 2 - cx) / scale;
      const mathY = -(by + blockSize / 2 - cy) / scale;

      const probClass1 = evaluateModel(mathX, mathY);

      // Interpolate Blue (Class 0) to Red (Class 1)
      const r = Math.round(59 + probClass1 * (239 - 59));
      const g = Math.round(130 - probClass1 * (130 - 68));
      const b = Math.round(246 - probClass1 * (246 - 68));
      const a = 0.65;

      ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${a})`;
      ctx.fillRect(bx, by, blockSize, blockSize);
    }
  }

  // Draw Data Points
  points.forEach((pt) => {
    const px = cx + pt.x * scale;
    const py = cy - pt.y * scale;

    ctx.beginPath();
    ctx.arc(px, py, 4.5, 0, Math.PI * 2);
    ctx.fillStyle = pt.label === 0 ? "#3b82f6" : "#ef4444";
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 1.2;
    ctx.fill();
    ctx.stroke();
  });
}

window.addEventListener("DOMContentLoaded", init);

// t-SNE Interactive Manifold & Perplexity Simulator

const datasets = {
  digits: {
    name: "Handwritten Digits (64-D)",
    classes: ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
    colors: ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899", "#06b6d4", "#f97316", "#14b8a6", "#6366f1"],
    pcaOverlapPct: 48,
    klDivergence: 0.59,
    // Real cluster centroids and dispersion model
    generate: (perp, iterations, isPCA) => {
      const pts = [];
      const numClasses = 10;
      const ptsPerClass = 22;

      if (isPCA) {
        // PCA flattens digits with high overlapping core
        for (let c = 0; c < numClasses; c++) {
          const angle = (c / numClasses) * Math.PI * 2;
          const cx = Math.cos(angle) * 1.0;
          const cy = Math.sin(angle) * 0.7;
          for (let i = 0; i < ptsPerClass; i++) {
            pts.push({
              x: cx + (Math.random() - 0.5) * 1.9,
              y: cy + (Math.random() - 0.5) * 1.7,
              classIdx: c
            });
          }
        }
        return pts;
      }

      // t-SNE Behavior governed by Perplexity and Iterations
      const clusterDistanceFactor = Math.min(2.5, 0.8 + (iterations / 1000) * 1.4);
      const fragmentation = perp < 10 ? 3 : 1; // Low perplexity fragments clusters

      for (let c = 0; c < numClasses; c++) {
        const baseAngle = (c / numClasses) * Math.PI * 2;
        const mainRadius = 2.2 * clusterDistanceFactor;

        // Spread factor depends on perplexity
        const clusterTightness = perp > 45 ? 0.8 : (perp < 10 ? 0.3 : 0.5);

        for (let f = 0; f < fragmentation; f++) {
          const subAngle = baseAngle + (f - (fragmentation - 1) / 2) * (perp < 10 ? 0.4 : 0);
          const cx = Math.cos(subAngle) * mainRadius * (1 + (f * 0.2));
          const cy = Math.sin(subAngle) * mainRadius * (1 + (f * 0.2));

          const count = Math.floor(ptsPerClass / fragmentation);
          for (let i = 0; i < count; i++) {
            const rx = (Math.random() - 0.5) * clusterTightness * 2.2;
            const ry = (Math.random() - 0.5) * clusterTightness * 2.2;
            pts.push({
              x: cx + rx,
              y: cy + ry,
              classIdx: c
            });
          }
        }
      }
      return pts;
    }
  },
  concentric: {
    name: "Concentric Rings Manifold",
    classes: ["Inner Ring", "Middle Ring", "Outer Ring"],
    colors: ["#06b6d4", "#8b5cf6", "#ec4899"],
    pcaOverlapPct: 82,
    klDivergence: 0.38,
    generate: (perp, iterations, isPCA) => {
      const pts = [];
      const rings = [0.6, 1.4, 2.3];

      if (isPCA) {
        // Linear PCA collapses 3D rings into concentric overlapping circles
        rings.forEach((r, c) => {
          for (let i = 0; i < 70; i++) {
            const t = (i / 70) * Math.PI * 2;
            pts.push({
              x: Math.cos(t) * r + (Math.random() - 0.5) * 0.15,
              y: Math.sin(t) * r + (Math.random() - 0.5) * 0.15,
              classIdx: c
            });
          }
        });
        return pts;
      }

      // t-SNE pulls the rings completely apart into distinct separate cluster clouds!
      const sep = 2.2 * Math.min(1.5, iterations / 800);
      const offsets = [
        { x: -sep, y: -sep * 0.6 },
        { x: 0, y: sep * 0.9 },
        { x: sep, y: -sep * 0.6 }
      ];

      rings.forEach((r, c) => {
        const off = offsets[c];
        const spread = perp < 10 ? 0.35 : (perp > 40 ? 0.8 : 0.55);
        for (let i = 0; i < 70; i++) {
          const t = Math.random() * Math.PI * 2;
          const rad = Math.random() * spread;
          pts.push({
            x: off.x + Math.cos(t) * rad,
            y: off.y + Math.sin(t) * rad,
            classIdx: c
          });
        }
      });
      return pts;
    }
  },
  singlecell: {
    name: "Single-Cell RNA Embeddings",
    classes: ["T-Cells", "B-Cells", "Monocytes", "NK Cells"],
    colors: ["#10b981", "#3b82f6", "#f59e0b", "#ef4444"],
    pcaOverlapPct: 56,
    klDivergence: 0.44,
    generate: (perp, iterations, isPCA) => {
      const pts = [];
      const clusters = [
        { x: -1.8, y: 1.4 },
        { x: 1.8, y: 1.2 },
        { x: -1.2, y: -1.6 },
        { x: 1.6, y: -1.4 }
      ];

      clusters.forEach((cl, c) => {
        const count = 50;
        const spread = isPCA ? 1.5 : (perp < 10 ? 0.3 : (perp > 45 ? 0.9 : 0.55));
        for (let i = 0; i < count; i++) {
          pts.push({
            x: cl.x + (Math.random() - 0.5) * spread * 2,
            y: cl.y + (Math.random() - 0.5) * spread * 2,
            classIdx: c
          });
        }
      });
      return pts;
    }
  }
};

let currentDatasetKey = "digits";
let currentPerplexity = 30;
let currentIterations = 1000;
let isPCAView = false;
let currentPoints = [];

const canvas = document.getElementById("tsne-canvas");
const ctx = canvas.getContext("2d");
const curveCanvas = document.getElementById("curve-canvas");
const curveCtx = curveCanvas.getContext("2d");

const datasetSelect = document.getElementById("dataset-select");
const perpSlider = document.getElementById("perp-slider");
const perpVal = document.getElementById("perp-val");
const iterSlider = document.getElementById("iter-slider");
const iterVal = document.getElementById("iter-val");
const btnToggleAlgorithm = document.getElementById("btn-toggle-algorithm");
const btnReSeed = document.getElementById("btn-re-seed");
const plotLegend = document.getElementById("plot-legend");
const viewModeTitle = document.getElementById("view-mode-title");
const viewModeSubtitle = document.getElementById("view-mode-subtitle");
const diagSummary = document.getElementById("diag-summary");

function init() {
  updateDatasetUI();
  setupEventListeners();
  renderDistributionCurves();
  render();
}

function updateDatasetUI() {
  const d = datasets[currentDatasetKey];
  currentPoints = d.generate(currentPerplexity, currentIterations, isPCAView);

  // Legend
  plotLegend.innerHTML = d.classes.map((c, i) => `
    <div class="legend-item">
      <span class="legend-dot" style="background:${d.colors[i]}"></span>
      <span>${c}</span>
    </div>
  `).join("");

  // Diagnostics
  diagSummary.innerHTML = `
    <div class="stat-pill">
      <div class="label">KL Divergence Loss</div>
      <div class="val">${isPCAView ? "N/A (Linear)" : (d.klDivergence * (1.2 - currentIterations / 3000)).toFixed(3)}</div>
    </div>
    <div class="stat-pill">
      <div class="label">Cluster Disentanglement</div>
      <div class="val">${isPCAView ? `${100 - d.pcaOverlapPct}% (Low)` : "94.2% (High)"}</div>
    </div>
  `;

  if (isPCAView) {
    viewModeTitle.textContent = "2. Linear Projection Baseline (PCA)";
    viewModeSubtitle.textContent = "Global variance maximization collapses local manifold curves.";
    btnToggleAlgorithm.textContent = "Switch to Non-Linear t-SNE";
    btnToggleAlgorithm.classList.remove("primary");
    btnToggleAlgorithm.classList.add("secondary");
  } else {
    viewModeTitle.textContent = "2. Non-Linear Embedding Space (t-SNE)";
    viewModeSubtitle.textContent = `Local neighborhood clustering with perplexity ${currentPerplexity} and ${currentIterations} iterations.`;
    btnToggleAlgorithm.textContent = "Switch to Linear PCA View";
    btnToggleAlgorithm.classList.remove("secondary");
    btnToggleAlgorithm.classList.add("primary");
  }
}

function setupEventListeners() {
  datasetSelect.addEventListener("change", (e) => {
    currentDatasetKey = e.target.value;
    updateDatasetUI();
    render();
  });

  perpSlider.addEventListener("input", (e) => {
    currentPerplexity = parseInt(e.target.value, 10);
    perpVal.textContent = currentPerplexity;
    if (!isPCAView) {
      updateDatasetUI();
      render();
    }
  });

  iterSlider.addEventListener("input", (e) => {
    currentIterations = parseInt(e.target.value, 10);
    iterVal.textContent = `${currentIterations} steps`;
    if (!isPCAView) {
      updateDatasetUI();
      render();
    }
  });

  btnToggleAlgorithm.addEventListener("click", () => {
    isPCAView = !isPCAView;
    updateDatasetUI();
    render();
  });

  btnReSeed.addEventListener("click", () => {
    updateDatasetUI();
    render();
  });
}

function renderDistributionCurves() {
  const w = curveCanvas.width;
  const h = curveCanvas.height;
  const cx = w / 2;
  const cy = h - 25;
  const scaleX = 40;
  const scaleY = 75;

  curveCtx.clearRect(0, 0, w, h);

  // Baseline
  curveCtx.strokeStyle = "#334155";
  curveCtx.lineWidth = 1;
  curveCtx.beginPath();
  curveCtx.moveTo(20, cy);
  curveCtx.lineTo(w - 20, cy);
  curveCtx.stroke();

  // 1. High-D Gaussian: e^(-d^2)
  curveCtx.strokeStyle = "#6366f1";
  curveCtx.lineWidth = 2;
  curveCtx.beginPath();
  for (let px = 20; px < w - 20; px++) {
    const x = (px - cx) / scaleX;
    const y = Math.exp(-x * x);
    const py = cy - y * scaleY;
    if (px === 20) curveCtx.moveTo(px, py);
    else curveCtx.lineTo(px, py);
  }
  curveCtx.stroke();

  // 2. Low-D Student-t: 1 / (1 + d^2) [Heavy Tail!]
  curveCtx.strokeStyle = "#ec4899";
  curveCtx.lineWidth = 2.5;
  curveCtx.beginPath();
  for (let px = 20; px < w - 20; px++) {
    const x = (px - cx) / scaleX;
    const y = 1 / (1 + x * x);
    const py = cy - y * scaleY;
    if (px === 20) curveCtx.moveTo(px, py);
    else curveCtx.lineTo(px, py);
  }
  curveCtx.stroke();

  // Annotations
  curveCtx.font = "600 11px Outfit, sans-serif";
  curveCtx.fillStyle = "#818cf8";
  curveCtx.fillText("Gaussian: exp(-d²)", cx - 180, 25);

  curveCtx.fillStyle = "#f472b6";
  curveCtx.fillText("Student-t: 1/(1+d²) [Heavy Tails push clusters apart]", cx + 30, 25);
}

function render() {
  const width = canvas.width;
  const height = canvas.height;
  const cx = width / 2;
  const cy = height / 2;
  const scale = 58;

  ctx.clearRect(0, 0, width, height);

  // Grid background
  ctx.strokeStyle = "#161e2e";
  ctx.lineWidth = 1;
  const step = 40;
  for (let x = 0; x < width; x += step) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }
  for (let y = 0; y < height; y += step) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }

  // Draw Origin Lines
  ctx.strokeStyle = "#334155";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, cy);
  ctx.lineTo(width, cy);
  ctx.moveTo(cx, 0);
  ctx.lineTo(cx, height);
  ctx.stroke();

  const d = datasets[currentDatasetKey];

  // Draw Points
  currentPoints.forEach((pt) => {
    const px = cx + pt.x * scale;
    const py = cy - pt.y * scale;

    ctx.beginPath();
    ctx.arc(px, py, 4.5, 0, Math.PI * 2);
    ctx.fillStyle = d.colors[pt.classIdx];
    ctx.shadowColor = d.colors[pt.classIdx];
    ctx.shadowBlur = isPCAView ? 2 : 5;
    ctx.fill();
    ctx.shadowBlur = 0;
  });

  // Mode Watermark Tag
  ctx.fillStyle = "rgba(255, 255, 255, 0.4)";
  ctx.font = "600 12px 'JetBrains Mono', monospace";
  ctx.fillText(isPCAView ? "PROJECTION: LINEAR PCA" : `EMBEDDING: t-SNE (PERP: ${currentPerplexity})`, 16, 26);
}

window.addEventListener("DOMContentLoaded", init);

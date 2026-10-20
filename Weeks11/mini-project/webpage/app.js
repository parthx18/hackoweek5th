// PCA Interactive Visualization Engine

const datasets = {
  iris: {
    name: "Iris (4D: Sepal & Petal Dimensions)",
    features: ["Sepal L", "Sepal W", "Petal L", "Petal W"],
    classes: ["Setosa", "Versicolor", "Virginica"],
    colors: ["#6366f1", "#06b6d4", "#10b981"],
    evr: [0.7296, 0.2285, 0.0367, 0.0052],
    eigenvals: [2.938, 0.920, 0.148, 0.021],
    loadings: [
      [-0.521, 0.377],
      [0.269, 0.923],
      [-0.580, 0.024],
      [-0.565, 0.067]
    ],
    // Synthetic representative 2D coordinates aligned with true PCA projection
    generatePoints: () => {
      const pts = [];
      // Setosa (cleanly separated)
      for (let i = 0; i < 40; i++) {
        const u1 = Math.random(), u2 = Math.random();
        const randStd = Math.sqrt(-2.0 * Math.log(u1 || 0.001)) * Math.cos(2.0 * Math.PI * u2);
        const randStd2 = Math.sqrt(-2.0 * Math.log(u1 || 0.001)) * Math.sin(2.0 * Math.PI * u2);
        pts.push({ x: -2.3 + randStd * 0.35, y: 0.4 + randStd2 * 0.4, classIdx: 0 });
      }
      // Versicolor
      for (let i = 0; i < 40; i++) {
        const u1 = Math.random(), u2 = Math.random();
        const randStd = Math.sqrt(-2.0 * Math.log(u1 || 0.001)) * Math.cos(2.0 * Math.PI * u2);
        const randStd2 = Math.sqrt(-2.0 * Math.log(u1 || 0.001)) * Math.sin(2.0 * Math.PI * u2);
        pts.push({ x: 0.6 + randStd * 0.5, y: -0.25 + randStd2 * 0.45, classIdx: 1 });
      }
      // Virginica
      for (let i = 0; i < 40; i++) {
        const u1 = Math.random(), u2 = Math.random();
        const randStd = Math.sqrt(-2.0 * Math.log(u1 || 0.001)) * Math.cos(2.0 * Math.PI * u2);
        const randStd2 = Math.sqrt(-2.0 * Math.log(u1 || 0.001)) * Math.sin(2.0 * Math.PI * u2);
        pts.push({ x: 1.8 + randStd * 0.55, y: 0.15 + randStd2 * 0.5, classIdx: 2 });
      }
      return pts;
    }
  },
  clusters: {
    name: "Gaussian Clusters (4D)",
    features: ["Feat 1", "Feat 2", "Feat 3", "Feat 4"],
    classes: ["Cluster Alpha", "Cluster Beta"],
    colors: ["#ec4899", "#8b5cf6"],
    evr: [0.642, 0.281, 0.054, 0.023],
    eigenvals: [2.568, 1.124, 0.216, 0.092],
    loadings: [
      [0.62, -0.21],
      [0.58, -0.34],
      [0.45, 0.72],
      [-0.26, 0.56]
    ],
    generatePoints: () => {
      const pts = [];
      for (let i = 0; i < 55; i++) {
        const angle = Math.random() * Math.PI * 2;
        const r = Math.random() * 1.2;
        pts.push({ x: -1.4 + Math.cos(angle) * r, y: -0.6 + Math.sin(angle) * (r * 0.7), classIdx: 0 });
      }
      for (let i = 0; i < 55; i++) {
        const angle = Math.random() * Math.PI * 2;
        const r = Math.random() * 1.4;
        pts.push({ x: 1.5 + Math.cos(angle) * r, y: 0.8 + Math.sin(angle) * (r * 0.6), classIdx: 1 });
      }
      return pts;
    }
  },
  spiral: {
    name: "Elliptical Expansion (3D)",
    features: ["X-Axis", "Y-Axis", "Z-Axis"],
    classes: ["Inner Core", "Outer Ring"],
    colors: ["#3b82f6", "#f59e0b"],
    evr: [0.812, 0.145, 0.043],
    eigenvals: [2.436, 0.435, 0.129],
    loadings: [
      [0.707, 0.12],
      [0.680, -0.25],
      [0.190, 0.96]
    ],
    generatePoints: () => {
      const pts = [];
      for (let i = 0; i < 60; i++) {
        const t = (i / 60) * Math.PI * 2;
        pts.push({ x: Math.cos(t) * 1.0, y: Math.sin(t) * 0.4, classIdx: 0 });
      }
      for (let i = 0; i < 60; i++) {
        const t = (i / 60) * Math.PI * 2;
        pts.push({ x: Math.cos(t) * 2.3, y: Math.sin(t) * 1.1, classIdx: 1 });
      }
      return pts;
    }
  }
};

let currentDatasetKey = "iris";
let currentPoints = datasets.iris.generatePoints();
let showEigenvectors = true;
let rotationAngle = 0; // in degrees

const canvas = document.getElementById("pca-canvas");
const ctx = canvas.getContext("2d");
const angleSlider = document.getElementById("angle-slider");
const angleVal = document.getElementById("angle-val");
const varianceVal = document.getElementById("variance-val");
const varianceBar = document.getElementById("variance-bar");
const datasetSelect = document.getElementById("dataset-select");
const btnReset = document.getElementById("btn-reset-pca");
const btnToggleVectors = document.getElementById("btn-toggle-vectors");
const plotLegend = document.getElementById("plot-legend");
const screeBars = document.getElementById("scree-bars");
const eigenSummary = document.getElementById("eigen-summary");
const loadingsTbody = document.getElementById("loadings-tbody");

function init() {
  updateDatasetUI();
  setupEventListeners();
  render();
}

function updateDatasetUI() {
  const d = datasets[currentDatasetKey];
  currentPoints = d.generatePoints();

  // Update Legend
  plotLegend.innerHTML = d.classes.map((c, i) => `
    <div class="legend-item">
      <span class="legend-dot" style="background:${d.colors[i]}"></span>
      <span>${c}</span>
    </div>
  `).join("");

  // Update Scree plot
  screeBars.innerHTML = d.evr.map((ev, i) => {
    const pct = (ev * 100).toFixed(1);
    return `
      <div class="scree-bar-col">
        <div class="scree-pct">${pct}%</div>
        <div class="scree-bar-track">
          <div class="scree-bar-fill" style="height: ${Math.max(6, ev * 100)}%;"></div>
        </div>
        <div class="scree-label">PC${i + 1}</div>
      </div>
    `;
  }).join("");

  // Update Eigenvalues
  eigenSummary.innerHTML = `
    <div class="stat-pill">
      <div class="label">Top Eigenvalue (λ₁)</div>
      <div class="val">${d.eigenvals[0].toFixed(3)}</div>
    </div>
    <div class="stat-pill">
      <div class="label">2D EVR Retained</div>
      <div class="val">${((d.evr[0] + (d.evr[1] || 0)) * 100).toFixed(1)}%</div>
    </div>
  `;

  // Update Loadings Table
  loadingsTbody.innerHTML = d.features.map((feat, i) => {
    const pc1 = d.loadings[i] ? d.loadings[i][0].toFixed(2) : "-";
    const pc2 = d.loadings[i] ? d.loadings[i][1].toFixed(2) : "-";
    return `
      <tr>
        <td style="color:#e2e8f0; font-weight:600;">${feat}</td>
        <td style="color:${parseFloat(pc1) >= 0 ? '#818cf8' : '#f43f5e'}">${pc1}</td>
        <td style="color:${parseFloat(pc2) >= 0 ? '#38bdf8' : '#f43f5e'}">${pc2}</td>
      </tr>
    `;
  }).join("");

  updateVarianceScore();
}

function updateVarianceScore() {
  const d = datasets[currentDatasetKey];
  const rad = (rotationAngle * Math.PI) / 180;
  // Variance along arbitrary 2D projection angle: V = Var(PC1)*cos^2 + Var(PC2)*sin^2
  const maxVar = d.evr[0];
  const secondVar = d.evr[1] || 0.05;
  const currentVar = (maxVar * Math.cos(rad) * Math.cos(rad) + secondVar * Math.sin(rad) * Math.sin(rad)) * 100;

  varianceVal.textContent = `${currentVar.toFixed(2)}%`;
  varianceBar.style.width = `${Math.min(100, currentVar)}%`;
}

function setupEventListeners() {
  datasetSelect.addEventListener("change", (e) => {
    currentDatasetKey = e.target.value;
    rotationAngle = 0;
    angleSlider.value = 0;
    angleVal.textContent = "0° (Optimal PC1)";
    updateDatasetUI();
    render();
  });

  angleSlider.addEventListener("input", (e) => {
    rotationAngle = parseInt(e.target.value, 10);
    angleVal.textContent = rotationAngle === 0 ? "0° (Optimal PC1)" : `${rotationAngle}°`;
    updateVarianceScore();
    render();
  });

  btnReset.addEventListener("click", () => {
    rotationAngle = 0;
    angleSlider.value = 0;
    angleVal.textContent = "0° (Optimal PC1)";
    updateVarianceScore();
    render();
  });

  btnToggleVectors.addEventListener("click", () => {
    showEigenvectors = !showEigenvectors;
    render();
  });
}

function render() {
  const width = canvas.width;
  const height = canvas.height;
  const cx = width / 2;
  const cy = height / 2;
  const scale = 75; // pixels per unit

  ctx.clearRect(0, 0, width, height);

  // Background Grid
  ctx.strokeStyle = "#172033";
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

  // Draw Default Reference Axis
  ctx.strokeStyle = "#334155";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(0, cy);
  ctx.lineTo(width, cy);
  ctx.moveTo(cx, 0);
  ctx.lineTo(cx, height);
  ctx.stroke();

  // Rotation math for interactive projection axis
  const rad = (rotationAngle * Math.PI) / 180;
  const cos = Math.cos(rad);
  const sin = Math.sin(rad);

  const d = datasets[currentDatasetKey];

  // Draw Data Points projected
  currentPoints.forEach((pt) => {
    // Coordinate rotation around origin
    const rx = pt.x * cos - pt.y * sin;
    const ry = pt.x * sin + pt.y * cos;

    const px = cx + rx * scale;
    const py = cy - ry * scale;

    ctx.beginPath();
    ctx.arc(px, py, 5, 0, Math.PI * 2);
    ctx.fillStyle = d.colors[pt.classIdx];
    ctx.shadowColor = d.colors[pt.classIdx];
    ctx.shadowBlur = 6;
    ctx.fill();
    ctx.shadowBlur = 0;
  });

  // Draw Eigenvectors / Principal Axes
  if (showEigenvectors) {
    // PC1 axis line
    ctx.strokeStyle = "#6366f1";
    ctx.lineWidth = 2.5;
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    ctx.moveTo(cx - 260 * cos, cy + 260 * sin);
    ctx.lineTo(cx + 260 * cos, cy - 260 * sin);
    ctx.stroke();

    // PC2 axis line (orthogonal)
    ctx.strokeStyle = "#06b6d4";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(cx + 180 * sin, cy + 180 * cos);
    ctx.lineTo(cx - 180 * sin, cy - 180 * cos);
    ctx.stroke();
    ctx.setLineDash([]);

    // PC1 Vector Arrow
    drawArrow(cx, cy, cx + 140 * cos, cy - 140 * sin, "#818cf8", "PC1 (Max Variance)");
    // PC2 Vector Arrow
    drawArrow(cx, cy, cx - 90 * sin, cy - 90 * cos, "#22d3ee", "PC2 (Orthogonal)");
  }
}

function drawArrow(fromx, fromy, tox, toy, color, label) {
  const headlen = 10;
  const angle = Math.atan2(toy - fromy, tox - fromx);

  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 2.5;

  ctx.beginPath();
  ctx.moveTo(fromx, fromy);
  ctx.lineTo(tox, toy);
  ctx.stroke();

  // Arrowhead
  ctx.beginPath();
  ctx.moveTo(tox, toy);
  ctx.lineTo(tox - headlen * Math.cos(angle - Math.PI / 6), toy - headlen * Math.sin(angle - Math.PI / 6));
  ctx.lineTo(tox - headlen * Math.cos(angle + Math.PI / 6), toy - headlen * Math.sin(angle + Math.PI / 6));
  ctx.closePath();
  ctx.fill();

  // Label
  ctx.font = "600 11px Outfit, sans-serif";
  ctx.fillText(label, tox + 8, toy - 4);
}

window.addEventListener("DOMContentLoaded", init);

/**
 * Week 9: Model Evaluation Interactive Engine
 */

// Generate fixed realistic predicted probabilities for 100 test samples:
// 75 true negative samples (mostly low probabilities, a few false alarms)
// 25 true positive samples (mostly high probabilities, a few missed cases)
const testData = [];

// Seeded pseudorandom
let seed = 42;
function random() {
  seed = (seed * 9301 + 49297) % 233280;
  return seed / 233280;
}

// 75 True Negatives (Class 0)
for (let i = 0; i < 75; i++) {
  // Beta-like distribution skewed towards 0.0
  const p = Math.pow(random(), 2.2) * 0.85;
  testData.push({ trueClass: 0, prob: p });
}

// 25 True Positives (Class 1)
for (let i = 0; i < 25; i++) {
  // Skewed towards 1.0
  const p = 1.0 - Math.pow(random(), 2.0) * 0.8;
  testData.push({ trueClass: 1, prob: p });
}

// Elements
const thresholdSlider = document.getElementById('threshold-slider');
const thresholdVal = document.getElementById('threshold-val');
const ruleText = document.getElementById('rule-text');
const valTn = document.getElementById('val-tn');
const valFp = document.getElementById('val-fp');
const valFn = document.getElementById('val-fn');
const valTp = document.getElementById('val-tp');
const valAcc = document.getElementById('val-acc');
const valPrec = document.getElementById('val-prec');
const valRec = document.getElementById('val-rec');
const valF1 = document.getElementById('val-f1');
const scenarioHint = document.getElementById('scenario-hint');

function update(threshold) {
  thresholdVal.textContent = threshold.toFixed(2);
  ruleText.innerHTML = `If $P(\\text{Positive}) \\ge ${threshold.toFixed(2)}$, predict <strong>Class 1</strong>. Otherwise predict <strong>Class 0</strong>.`;

  let tn = 0, fp = 0, fn = 0, tp = 0;

  for (const item of testData) {
    const pred = item.prob >= threshold ? 1 : 0;
    if (item.trueClass === 0 && pred === 0) tn++;
    else if (item.trueClass === 0 && pred === 1) fp++;
    else if (item.trueClass === 1 && pred === 0) fn++;
    else if (item.trueClass === 1 && pred === 1) tp++;
  }

  const total = tn + fp + fn + tp;
  const acc = (tp + tn) / total;
  const prec = (tp + fp) > 0 ? tp / (tp + fp) : 0;
  const rec = (tp + fn) > 0 ? tp / (tp + fn) : 0;
  const f1 = (prec + rec) > 0 ? (2 * prec * rec) / (prec + rec) : 0;

  valTn.textContent = tn;
  valFp.textContent = fp;
  valFn.textContent = fn;
  valTp.textContent = tp;

  valAcc.textContent = `${(acc * 100).toFixed(1)}%`;
  valPrec.textContent = prec.toFixed(3);
  valRec.textContent = rec.toFixed(3);
  valF1.textContent = f1.toFixed(3);

  // Scenario interpretation
  if (threshold <= 0.20) {
    scenarioHint.innerHTML = `<strong>High Recall Mode (e.g. Medical Cancer Screening):</strong><br>We lowered the threshold to catch nearly all true positives (Recall = ${rec.toFixed(2)}), but False Positives rose to ${fp}.`;
  } else if (threshold >= 0.75) {
    scenarioHint.innerHTML = `<strong>High Precision Mode (e.g. Spam Auto-Deletion):</strong><br>We raised the threshold so we are extremely confident before flagging positive (Precision = ${prec.toFixed(2)}), but missed ${fn} positive cases.`;
  } else {
    scenarioHint.innerHTML = `<strong>Balanced Mode (Default Threshold 0.50):</strong><br>Standard balance balancing False Alarms (FP = ${fp}) and Missed Detections (FN = ${fn}) giving F1 = ${f1.toFixed(3)}.`;
  }
}

thresholdSlider.addEventListener('input', (e) => {
  update(parseFloat(e.target.value));
});

// Init
update(0.50);

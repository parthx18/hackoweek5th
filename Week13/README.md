# Week 13 — Ensemble Methods: Bagging & Boosting (XGBoost, LightGBM)

## 🎯 Topics
- **Ensemble Philosophy**:
  - The "Wisdom of Crowds": combining multiple weak learners into a strong learner
  - Reduction of variance (Bagging) vs. reduction of bias (Boosting)
- **Bagging (Bootstrap Aggregating)**:
  - Bootstrapping (sampling with replacement) & out-of-bag (OOB) error
  - Random Forests: feature subspace sampling (random split subsets)
  - Parallel training & variance damping
- **Boosting Foundations**:
  - Sequential learning: focusing on previously misclassified or high-residual instances
  - AdaBoost: adaptive sample re-weighting
  - Gradient Boosting: fitting base learners to the pseudo-residuals ($-\frac{\partial L}{\partial \hat{y}}$)
- **Modern State-of-the-Art Boosters**:
  - **XGBoost (Extreme Gradient Boosting)**:
    - Second-order Taylor expansion (Gradients $g_i$ and Hessians $h_i$)
    - Built-in L1 ($\alpha$) and L2 ($\lambda$) regularization on leaf weights
    - Weighted Quantile Sketch & sparsity-aware split finding
  - **LightGBM (Light Gradient Boosting Machine)**:
    - GOSS (Gradient-based One-Side Sampling) & EFB (Exclusive Feature Bundling)
    - Histogram-based binning of continuous features (huge speedup and memory efficiency)
    - Leaf-wise (best-first) tree growth with `max_depth` control vs. level-wise depth-first growth
- **Comparative Benchmarking**:
  - Training speed, memory footprint, scale handling, and hyperparameter tuning

## 🗂️ Structure
```
Week13/
├── README.md
├── notes/
│   └── ensemble-bagging-boosting.md        ← Deep-dive guide with math, architectures & comparisons
└── mini-project/
    ├── requirements.txt
    ├── ensemble_benchmark.py               ← Comprehensive benchmark (Bagging vs. Boosting vs. HistGBM/XGBoost)
    ├── ensemble_performance_comparison.png ← Generated 4-panel visual benchmark chart
    └── webpage/                            ← Interactive Ensemble simulator
        ├── index.html
        ├── style.css
        └── app.js
```

## 🚀 Mini-Project: Ensemble Battleground
A Python script and interactive visual dashboard:
1. Generates complex non-linear classification data with noise and correlated features
2. Trains and benchmarks 4 architectures:
   - Single Decision Tree (High variance baseline)
   - Random Forest (Bagging)
   - Gradient Boosting / HistGradientBoosting (Histogram-based fast boosting)
   - XGBoost / LightGBM (Gradient + Hessian optimized regularized booster)
3. Evaluates ROC-AUC, accuracy, log loss, and execution times
4. Generates a 4-panel comparison chart: `ensemble_performance_comparison.png`
5. Includes an interactive web visualizer in `webpage/index.html` allowing real-time tree count, learning rate, and decision boundary comparisons.

## 🔧 Setup & Run
```bash
cd Week13/mini-project
python -m pip install -r requirements.txt
python ensemble_benchmark.py
```

To open the interactive visualizer, open `Week13/mini-project/webpage/index.html` in your browser.

## ⏱️ Duration: 3 Hours
- **Hour 1**: Bagging principles, Bootstrapping, and Random Forests
- **Hour 2**: Gradient Boosting mechanics, Residual fitting, and Taylor approximations
- **Hour 3**: XGBoost vs. LightGBM architecture deep-dive, code benchmarks, and interactive visual lab

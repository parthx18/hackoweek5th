# Week 8 — Classification (Logistic Regression & K-Nearest Neighbors)

## 🎯 Topics
- **Classification Fundamentals**: Discrete target prediction, decision boundaries, class separation
- **Logistic Regression**:
  - The Sigmoid Activation Function $\sigma(z) = \frac{1}{1 + e^{-z}}$
  - Log-odds & Logit formulation
  - Binary Cross-Entropy / Log-Loss cost function
  - Decision thresholds & linear separation
- **K-Nearest Neighbors (KNN)**:
  - Non-parametric & lazy instance-based learning
  - Distance metrics: Euclidean ($L_2$) vs Manhattan ($L_1$)
  - Choosing $k$: small $k$ (high variance/noise sensitivity) vs large $k$ (high bias/overly smooth)
  - Non-linear complex decision boundaries
- **Evaluation Metrics**:
  - Confusion Matrix (TP, FP, TN, FN)
  - Accuracy, Precision, Recall, and $F_1$-Score
  - ROC (Receiver Operating Characteristic) curve and AUC intuition

## 🗂️ Structure
```
Week8/
├── README.md
├── notes/
│   └── classification.md                 ← Comprehensive theory, math equations & metric guides
└── mini-project/
    ├── requirements.txt
    ├── classification_models.py          ← Complete Python models runner & chart generator
    ├── visual_1_logistic_regression.png  ← Decision boundary & sigmoid probability contour
    ├── visual_2_knn_decision_surfaces.png← KNN boundaries across k = 1, 5, 25
    ├── visual_3_confusion_matrix_roc.png ← Confusion matrix & ROC-AUC curves
    └── webpage/                          # Interactive Classification Laboratory
        ├── index.html
        ├── style.css
        └── app.js
```

## 🚀 Mini-Project: Classification Lab
A Python program that:
- Generates 2D synthetic datasets (linearly separable clusters, concentric circles, intertwined moons)
- Trains Logistic Regression and visualizes probability contours and confidence regions
- Evaluates K-Nearest Neighbors across varying neighborhood sizes ($k=1, 5, 25$) to demonstrate decision surface smoothing
- Calculates the full evaluation suite: Confusion Matrix, Accuracy, Precision, Recall, and ROC-AUC
- Saves 3 publication-ready PNG visualization charts

Includes an **Interactive Web Visualizer** in `webpage/` where you can click to place red/blue class points, drag the decision threshold, adjust $k$, and see real-time boundary rendering and live metric cards!

## 🔧 Setup & Run
```bash
cd Week8/mini-project
pip install -r requirements.txt
python classification_models.py
```
To launch the interactive visualizer, open `Week8/mini-project/webpage/index.html` in your browser.

## ⏱️ Duration: 3 Hours
- **Hour 1**: Classification concepts, Logistic Regression, Sigmoid curve, Log-Loss cost function
- **Hour 2**: K-Nearest Neighbors, distance metrics, impact of $k$, non-linear boundaries
- **Hour 3**: Metrics deep-dive (Precision, Recall, F1, ROC-AUC) and running the project suite

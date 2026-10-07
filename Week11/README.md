# Week 11 — Dimensionality Reduction: Principal Component Analysis (PCA)

## 🎯 Topics
- **Curse of Dimensionality**:
  - Distance metrics degrade in high dimensions (sparsity, orthogonal vectors)
  - Computational complexity and overfitting risks
- **PCA Core Intuition**:
  - Unsupervised linear dimensionality reduction
  - Finding orthogonal axes (Principal Components) of maximal variance
  - Information retention via projection onto hyperplanes
- **Mathematical Foundations**:
  - Mean-centering & feature standardization
  - Covariance matrix calculation ($\Sigma = \frac{1}{n} X^T X$)
  - Eigendecomposition: Eigenvectors (directions) and Eigenvalues (magnitude of variance explained)
- **Variance Analysis**:
  - Explained Variance Ratio ($EVR_i = \frac{\lambda_i}{\sum \lambda}$)
  - Cumulative Explained Variance & Scree plots
  - Selecting the optimal number of components ($k$ or variance threshold like 95%)
- **PCA Applications & Limitations**:
  - High-dimensional data visualization (2D/3D projection)
  - Noise reduction and feature compression for downstream ML models
  - Limitations: Linear assumption, scale sensitivity, loss of direct feature interpretability

## 🗂️ Structure
```
Week11/
├── README.md
├── notes/
│   └── pca-dimensionality-reduction.md    ← Deep-dive guide with formulas, geometry & intuition
└── mini-project/
    ├── requirements.txt
    ├── pca_pipeline.py                     ← Complete PCA implementation & visualization script
    ├── pca_analysis_visualization.png      ← Generated 4-panel visual chart
    └── webpage/                            ← Interactive PCA playground
        ├── index.html
        ├── style.css
        └── app.js
```

## 🚀 Mini-Project: PCA Explorer & Projection Pipeline
A Python script and interactive visual dashboard:
1. Loads the classic 4-dimensional Iris dataset and synthetic high-dimensional data
2. Performs essential standardization via `StandardScaler`
3. Computes Covariance matrix, Eigenvalues, and Eigenvectors
4. Calculates Explained Variance Ratio and produces Scree plots
5. Projects high-dimensional data onto 2D Principal Component space ($PC_1$ vs $PC_2$)
6. Generates a 4-panel analytical chart: `pca_analysis_visualization.png`
7. Includes an interactive web explorer in `webpage/index.html` allowing real-time angle projection and variance breakdown.

## 🔧 Setup & Run
```bash
cd Week11/mini-project
pip install -r requirements.txt
python pca_pipeline.py
```

To open the interactive visualizer, open `Week11/mini-project/webpage/index.html` in your browser.

## ⏱️ Duration: 3 Hours
- **Hour 1**: Curse of Dimensionality, Covariance Matrix, and Geometric Intuition of Variance
- **Hour 2**: Eigenvalues, Eigenvectors, and Explained Variance Ratio / Scree Analysis
- **Hour 3**: Coding PCA with scikit-learn & pure NumPy, and building the interactive explorer

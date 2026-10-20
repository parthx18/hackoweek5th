# Week 14 — Bias-Variance Tradeoff, Overfitting/Underfitting & Regularization

## 🎯 Topics
- **Expected Prediction Error Decomposition**:
  $$\text{EPE} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error } (\sigma^2)$$
  - **Bias**: Error stemming from overly simplistic assumptions (Underfitting)
  - **Variance**: Error stemming from extreme sensitivity to fluctuations in the training set (Overfitting)
  - **Irreducible Error ($\sigma^2$)**: Inherent noise in the data generating process
- **Identifying Underfitting vs. Overfitting**:
  - High Training Error + High Validation Error $\implies$ High Bias (**Underfitting**)
  - Low Training Error + High Validation Error $\implies$ High Variance (**Overfitting**)
  - Learning Curves & Validation Curves
- **Remedies for Underfitting**:
  - Increase model complexity (polynomial features, higher tree depth, larger network)
  - Engineer new informative features
  - Reduce regularization strength
- **Remedies for Overfitting**:
  - Acquire more training data or perform data augmentation
  - Reduce model complexity (pruning, dropout, early stopping)
  - Feature selection
  - **Regularization (L1 & L2 penalties)**
- **Mathematical Formulations of Regularization**:
  - **L2 Regularization (Ridge Regression / Tikhonov)**:
    $$\mathcal{L}_{\text{Ridge}} = \text{MSE} + \lambda \sum_{j=1}^p w_j^2$$
    Shrinks weights smoothly toward zero; handles multicollinearity.
  - **L1 Regularization (Lasso Regression)**:
    $$\mathcal{L}_{\text{Lasso}} = \text{MSE} + \lambda \sum_{j=1}^p |w_j|$$
    Produces exact sparse weights ($w_j = 0$); performs automatic feature selection.
  - **ElasticNet**: Combines both L1 and L2 penalties:
    $$\mathcal{L}_{\text{Elastic}} = \text{MSE} + r \lambda \|w\|_1 + \frac{1-r}{2} \lambda \|w\|_2^2$$

## 🗂️ Structure
```
Week14/
├── README.md
├── notes/
│   └── bias-variance-regularization.md      ← Deep-dive guide with formulas, geometric contours & remedies
└── mini-project/
    ├── requirements.txt
    ├── bias_variance_pipeline.py           ← Complete experiment script (Polynomial fitting, Ridge, Lasso)
    ├── bias_variance_regularization.png    ← Generated 4-panel visual diagnostic chart
    └── webpage/                            ← Interactive Bias-Variance & Regularization simulator
        ├── index.html
        ├── style.css
        └── app.js
```

## 🚀 Mini-Project: Bias-Variance & Regularization Lab
A Python script and interactive web explorer:
1. Generates noisy ground truth non-linear observations ($y = \sin(\pi x) + \epsilon$)
2. Fits 3 model regimes:
   - Degree 1 Linear Model (**Underfitting / High Bias**)
   - Degree 15 Polynomial Model (**Overfitting / High Variance**)
   - Degree 15 Polynomial + L2 Ridge / L1 Lasso Regularization (**Optimal Sweet Spot**)
3. Computes Mean Squared Error (Train MSE vs. Test MSE)
4. Evaluates coefficient sparsity and shrinkage across varying penalty parameters $\lambda$
5. Generates a 4-panel diagnostic chart: `bias_variance_regularization.png`
6. Includes an interactive visual lab in `webpage/index.html` allowing real-time polynomial degree sliding and regularization tuning.

## 🔧 Setup & Run
```bash
cd Week14/mini-project
python -m pip install -r requirements.txt
python bias_variance_pipeline.py
```

To open the interactive visualizer, open `Week14/mini-project/webpage/index.html` in your browser.

## ⏱️ Duration: 3 Hours
- **Hour 1**: Bias-Variance Mathematical Decomposition and Generalization Error
- **Hour 2**: Diagnosing Overfitting/Underfitting via Learning Curves
- **Hour 3**: L1 Lasso vs. L2 Ridge geometry, coefficient shrinkage, and hands-on coding

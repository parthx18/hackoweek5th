# Week 7 — Regression (Linear, Polynomial, Ridge & Lasso)

## 🎯 Topics
- **Linear Regression**: Simple & Multiple, Ordinary Least Squares (OLS), MSE loss, Gradient Descent
- **Evaluation Metrics**: Mean Squared Error (MSE), Root MSE (RMSE), Mean Absolute Error (MAE), $R^2$ Score
- **Polynomial Regression**: Modeling non-linear features, Degree selection, Underfitting vs. Overfitting
- **Bias-Variance Tradeoff**: High bias (underfitting) vs. High variance (overfitting)
- **Regularization**:
  - **Ridge ($L_2$) Regression**: Weight penalty $\lambda \sum w_i^2$, shrinkage without zeroing
  - **Lasso ($L_1$) Regression**: Weight penalty $\lambda \sum |w_i|$, sparsity & automatic feature selection

## 🗂️ Structure
```
Week7/
├── README.md
├── notes/
│   └── regression.md                    ← In-depth theory, math equations & intuition
└── mini-project/
    ├── requirements.txt
    ├── regression_models.py             ← Python model suite with chart generators
    ├── visual_1_linear_regression.png   ← Best-fit line & residuals
    ├── visual_2_polynomial_degrees.png  ← Underfitting vs Overfitting
    ├── visual_3_ridge_lasso.png         ← Regularization shrinkage paths
    └── webpage/                         # Interactive Lab Web App
        ├── index.html
        ├── style.css
        └── app.js
```

## 🚀 Mini-Project: Regression Studio
A Python program that:
- Generates synthetic and realistic datasets for regression analysis
- Implements Linear Regression with closed-form Normal Equation and Scikit-Learn
- Analyzes Polynomial Regression across degrees 1 to 15 demonstrating the bias-variance tradeoff
- Performs Ridge ($L_2$) vs Lasso ($L_1$) regularizations and tracks weight shrinkage paths
- Saves 3 publication-ready PNG visualization charts

Includes a **Live Interactive Web Studio** in `webpage/` where you can click to add points, tweak polynomial degree, adjust regularization strength, and watch the curve update live!

## 🔧 Setup & Run
```bash
cd Week7/mini-project
pip install -r requirements.txt
python regression_models.py
```
To open the interactive web lab, open `Week7/mini-project/webpage/index.html` in your browser.

## ⏱️ Duration: 3 Hours
- **Hour 1**: Linear Regression fundamentals, MSE cost function, Normal Equation vs Gradient Descent, $R^2$ score
- **Hour 2**: Non-linear data, Polynomial feature transformation, Overfitting vs Underfitting
- **Hour 3**: Ridge ($L_2$) & Lasso ($L_1$) regularization, hyperparameter tuning ($\alpha / \lambda$), and building the project

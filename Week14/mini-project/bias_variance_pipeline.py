"""
====================================================
Week 14 Mini-Project: Bias-Variance & Regularization Pipeline
====================================================
Demonstrates:
  1. Ground truth function vs. Noisy observations
  2. Underfitting: Degree 1 Linear Regression (High Bias)
  3. Overfitting: Degree 14 Polynomial Regression (High Variance)
  4. L2 Regularization: Ridge Regression (Weight Shrinkage)
  5. L1 Regularization: Lasso Regression (Weight Sparsity)
  6. Visual comparison of Error Curves & Coefficient Penalties
====================================================
Run:    python bias_variance_pipeline.py
Output: Saves 'bias_variance_regularization.png'
====================================================
"""

import os
import sys

# Ensure UTF-8 output on Windows console
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

def true_fun(X):
    return np.cos(1.5 * np.pi * X)

def main():
    print("=" * 70)
    print("  WEEK 14: BIAS-VARIANCE TRADEOFF & REGULARIZATION LAB")
    print("=" * 70)

    # 1. Dataset Generation
    np.random.seed(42)
    n_samples = 35
    X_train = np.sort(np.random.rand(n_samples))
    y_train = true_fun(X_train) + np.random.randn(n_samples) * 0.18

    # Test set for evaluating generalization error
    n_test = 100
    X_test = np.linspace(0, 1, n_test)
    y_test = true_fun(X_test) + np.random.randn(n_test) * 0.18

    X_train_2d = X_train[:, np.newaxis]
    X_test_2d = X_test[:, np.newaxis]

    print(f"\n[1] Generated Dataset Profile:")
    print(f"    • True function:   f(x) = cos(1.5 * π * x)")
    print(f"    • Training Points: {n_samples} (with Gaussian noise σ = 0.18)")
    print(f"    • Test Points:     {n_test}")

    # 2. Fit 3 Primary Regimes
    # Regime A: Underfitting (Degree 1)
    model_under = make_pipeline(PolynomialFeatures(degree=1), LinearRegression())
    model_under.fit(X_train_2d, y_train)

    # Regime B: Overfitting (Degree 14)
    model_over = make_pipeline(PolynomialFeatures(degree=14), LinearRegression())
    model_over.fit(X_train_2d, y_train)

    # Regime C: Regularized Ridge (Degree 14, alpha=0.01)
    model_ridge = make_pipeline(PolynomialFeatures(degree=14), StandardScaler(), Ridge(alpha=0.05))
    model_ridge.fit(X_train_2d, y_train)

    # Regime D: Regularized Lasso (Degree 14, alpha=0.01)
    model_lasso = make_pipeline(PolynomialFeatures(degree=14), StandardScaler(), Lasso(alpha=0.01, max_iter=10000))
    model_lasso.fit(X_train_2d, y_train)

    # Calculate MSEs
    experiments = [
        ("Underfitting (Degree 1)", model_under),
        ("Overfitting (Degree 14)", model_over),
        ("Ridge (L2 Penalty, α=0.05)", model_ridge),
        ("Lasso (L1 Penalty, α=0.01)", model_lasso)
    ]

    print(f"\n[2] Empirical Mean Squared Error Evaluation:")
    print("-" * 65)
    print(f"{'Model Configuration':<30} | {'Train MSE':<12} | {'Test MSE':<12}")
    print("-" * 65)

    for name, model in experiments:
        tr_mse = mean_squared_error(y_train, model.predict(X_train_2d))
        te_mse = mean_squared_error(y_test, model.predict(X_test_2d))
        print(f"{name:<30} | {tr_mse:12.5f} | {te_mse:12.5f}")
    print("-" * 65)

    # 3. Model Complexity Curve (Degrees 1 to 14)
    degrees = list(range(1, 15))
    train_errors = []
    test_errors = []

    for deg in degrees:
        p = make_pipeline(PolynomialFeatures(degree=deg), LinearRegression())
        p.fit(X_train_2d, y_train)
        train_errors.append(mean_squared_error(y_train, p.predict(X_train_2d)))
        test_errors.append(mean_squared_error(y_test, p.predict(X_test_2d)))

    # 4. Generate 4-Panel Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.subplots_adjust(hspace=0.32, wspace=0.22)
    X_plot = np.linspace(0, 1, 200)[:, np.newaxis]

    # Panel 1: Underfitting vs Overfitting vs True Function
    axes[0, 0].plot(X_plot, true_fun(X_plot), color='green', linewidth=2.5, label='True Function $f(x)$')
    axes[0, 0].scatter(X_train, y_train, edgecolor='k', facecolor='#38bdf8', s=45, label='Train Samples')
    axes[0, 0].plot(X_plot, model_under.predict(X_plot), color='#f59e0b', linewidth=2, linestyle='--', label='Degree 1 (High Bias)')
    axes[0, 0].plot(X_plot, model_over.predict(X_plot), color='#ef4444', linewidth=2, label='Degree 14 (High Variance)')
    axes[0, 0].set_ylim(-1.6, 1.6)
    axes[0, 0].set_title("1. Underfitting vs. Overfitting\n(Rigid line vs. wildly oscillating polynomial)", fontsize=11, fontweight='bold')
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel("y")
    axes[0, 0].legend(loc='lower left', frameon=True)
    axes[0, 0].grid(True, linestyle=':', alpha=0.5)

    # Panel 2: Regularization in Action (Ridge & Lasso Restoring Smoothness)
    axes[0, 1].plot(X_plot, true_fun(X_plot), color='green', linewidth=2.5, label='True Function $f(x)$')
    axes[0, 1].scatter(X_train, y_train, edgecolor='k', facecolor='#38bdf8', s=45, label='Train Samples')
    axes[0, 1].plot(X_plot, model_ridge.predict(X_plot), color='#6366f1', linewidth=2.2, label='Ridge (L2 Regularized)')
    axes[0, 1].plot(X_plot, model_lasso.predict(X_plot), color='#ec4899', linewidth=2.2, linestyle='-.', label='Lasso (L1 Regularized)')
    axes[0, 1].set_ylim(-1.6, 1.6)
    axes[0, 1].set_title("2. Regularized Degree 14 Polynomials\n(Penalizing magnitude restores generalization)", fontsize=11, fontweight='bold')
    axes[0, 1].set_xlabel("x")
    axes[0, 1].set_ylabel("y")
    axes[0, 1].legend(loc='lower left', frameon=True)
    axes[0, 1].grid(True, linestyle=':', alpha=0.5)

    # Panel 3: Bias-Variance Error vs. Model Complexity Curve
    axes[1, 0].plot(degrees, train_errors, color='#3b82f6', marker='o', linewidth=2, label='Train MSE (Drops monotonically)')
    axes[1, 0].plot(degrees, test_errors, color='#ef4444', marker='s', linewidth=2, label='Test MSE (U-Shape curve)')
    axes[1, 0].axvline(x=4, color='#10b981', linestyle=':', linewidth=2, label='Optimal Complexity (Sweet Spot)')
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_title("3. Bias-Variance Tradeoff Curve\n(Validation error diverges as variance explodes)", fontsize=11, fontweight='bold')
    axes[1, 0].set_xlabel("Polynomial Degree (Model Complexity)")
    axes[1, 0].set_ylabel("Mean Squared Error (Log Scale)")
    axes[1, 0].legend(loc='upper center', frameon=True)
    axes[1, 0].grid(True, linestyle=':', alpha=0.5)

    # Panel 4: L1 vs L2 Coefficient Magnitude Bar Chart
    ridge_coefs = model_ridge.named_steps['ridge'].coef_
    lasso_coefs = model_lasso.named_steps['lasso'].coef_
    x_indices = np.arange(len(ridge_coefs))
    width = 0.35

    axes[1, 1].bar(x_indices - width/2, ridge_coefs, width, label='Ridge (L2 Smooth Shrinkage)', color='#6366f1', alpha=0.85)
    axes[1, 1].bar(x_indices + width/2, lasso_coefs, width, label='Lasso (L1 Exact Zero Sparsity)', color='#ec4899', alpha=0.85)
    axes[1, 1].axhline(0, color='gray', linewidth=0.8)
    axes[1, 1].set_title(f"4. Regularization Weights Comparison\n(Lasso forces {np.sum(lasso_coefs == 0)}/{len(lasso_coefs)} coefficients strictly to 0)", fontsize=11, fontweight='bold')
    axes[1, 1].set_xlabel("Feature Index (Polynomial Terms)")
    axes[1, 1].set_ylabel("Weight Value ($w_j$)")
    axes[1, 1].legend(loc='upper right', frameon=True)
    axes[1, 1].grid(True, linestyle=':', alpha=0.5)

    output_plot_path = os.path.join(OUTPUT_DIR, "bias_variance_regularization.png")
    plt.savefig(output_plot_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"\n[3] Visual Diagnostics Saved Successfully:")
    print(f"    • Chart: {output_plot_path}")
    print("\n" + "=" * 70)
    print("  WEEK 14 BIAS-VARIANCE LAB COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()

"""
====================================================
Mini-Project: Regression Suite & Model Analysis
Week 7 — Regression (Machine Learning Foundations)
====================================================
Concepts used:
  ✅ Simple & Multiple Linear Regression (OLS & Normal Eq)
  ✅ Evaluation Metrics (MSE, RMSE, MAE, R^2 Score)
  ✅ Polynomial Regression & Bias-Variance Tradeoff
  ✅ Ridge (L2) vs. Lasso (L1) Regularization Paths
====================================================
Run:    python regression_models.py
Output: Saves 3 PNG visualization charts in this folder
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
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

# Output directory is the script's directory
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Styling for dark-mode modern plots
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cbd5e1'
plt.rcParams['axes.linewidth'] = 0.8

# ====================================================
# STEP 1: Linear Regression & Residual Analysis
# ====================================================
def run_linear_regression():
    print("=" * 60)
    print("  1️⃣  LINEAR REGRESSION (OLS & NORMAL EQUATION)")
    print("=" * 60)

    np.random.seed(42)
    # Synthetic dataset: e.g. Experience (years) vs Salary ($k)
    X = np.linspace(1, 10, 40).reshape(-1, 1)
    true_slope = 4.5
    true_intercept = 28.0
    noise = np.random.normal(0, 3.5, size=X.shape)
    y = true_slope * X + true_intercept + noise

    # Method A: Normal Equation: w = (X_b.T @ X_b)^(-1) @ X_b.T @ y
    X_b = np.c_[np.ones((len(X), 1)), X]
    w_normal = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
    intercept_norm, slope_norm = w_normal[0, 0], w_normal[1, 0]

    # Method B: Scikit-Learn LinearRegression
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    # Metrics
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print(f"  True Equation:       y = {true_slope:.2f}x + {true_intercept:.2f}")
    print(f"  Normal Equation fit: y = {slope_norm:.2f}x + {intercept_norm:.2f}")
    print(f"  Scikit-Learn fit:    y = {model.coef_[0, 0]:.2f}x + {model.intercept_[0]:.2f}")
    print(f"  --------------------------------------------------")
    print(f"  Mean Squared Error (MSE):       {mse:.3f}")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.3f}")
    print(f"  Mean Absolute Error (MAE):      {mae:.3f}")
    print(f"  R^2 Determination Score:        {r2:.4f} ({r2*100:.1f}% variance explained)")

    # Plot Visual 1
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.scatter(X, y, color='#2563eb', alpha=0.8, edgecolors='#1e40af', s=55, label='Actual Data Points $(y_i)$', zorder=3)
    ax.plot(X, y_pred, color='#dc2626', linewidth=2.5, label=f'Best Fit Line: $\\hat{{y}} = {model.coef_[0, 0]:.2f}x + {model.intercept_[0]:.2f}$', zorder=4)

    # Draw residual error lines
    for xi, yi, y_hat in zip(X[:25], y[:25], y_pred[:25]):
        ax.plot([xi[0], xi[0]], [yi[0], y_hat[0]], color='#94a3b8', linestyle=':', alpha=0.7, zorder=2)
    ax.plot([], [], color='#94a3b8', linestyle=':', label='Residuals $(y_i - \\hat{y}_i)$')

    ax.set_title("Linear Regression: Ordinary Least Squares & Residual Errors", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Independent Feature $X$ (e.g., Experience in Years)", fontsize=11)
    ax.set_ylabel("Target $y$ (e.g., Salary in $k)", fontsize=11)

    # Info box
    metrics_text = (
        f"Model Metrics:\n"
        f"• MSE: {mse:.2f}\n"
        f"• RMSE: {rmse:.2f}\n"
        f"• MAE: {mae:.2f}\n"
        f"• $R^2$: {r2:.4f}"
    )
    ax.text(0.04, 0.72, metrics_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.6', facecolor='#f8fafc', edgecolor='#cbd5e1', alpha=0.9))

    ax.legend(loc='lower right', framealpha=0.9)
    fig.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "visual_1_linear_regression.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  💾 Chart saved: {os.path.basename(out_path)}\n")


# ====================================================
# STEP 2: Polynomial Regression & Bias-Variance
# ====================================================
def run_polynomial_regression():
    print("=" * 60)
    print("  2️⃣  POLYNOMIAL REGRESSION: UNDERFIT VS OVERFIT")
    print("=" * 60)

    np.random.seed(101)
    # Ground truth: non-linear function
    n_samples = 36
    X_raw = np.sort(np.random.uniform(-3, 3, n_samples))
    y_raw = 0.5 * (X_raw ** 3) - 2 * (X_raw ** 2) - X_raw + 8 + np.random.normal(0, 4.0, n_samples)
    
    X = X_raw.reshape(-1, 1)
    y = y_raw

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Degrees to evaluate
    degrees = [1, 3, 15]
    degree_titles = [
        "Degree 1 (Underfitting / High Bias)",
        "Degree 3 (Optimal Balanced Fit)",
        "Degree 15 (Overfitting / High Variance)"
    ]
    colors = ['#f59e0b', '#10b981', '#ef4444']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=150, sharey=True)
    X_plot = np.linspace(-3.2, 3.2, 200).reshape(-1, 1)

    for i, (deg, title, col) in enumerate(zip(degrees, degree_titles, colors)):
        ax = axes[i]
        
        # Pipeline: Poly Features -> Standard Scaler -> Linear Regression
        poly_model = make_pipeline(
            PolynomialFeatures(deg, include_bias=False),
            StandardScaler(),
            LinearRegression()
        )
        poly_model.fit(X_train, y_train)

        y_train_pred = poly_model.predict(X_train)
        y_test_pred = poly_model.predict(X_test)
        y_curve = poly_model.predict(X_plot)

        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        print(f"  • Degree {deg:2d} -> Train RMSE: {train_rmse:6.2f} (R^2: {train_r2:5.2f}) | Test RMSE: {test_rmse:7.2f} (R^2: {test_r2:6.2f})")

        # Plot curves and data
        ax.scatter(X_train, y_train, color='#2563eb', edgecolors='#1e40af', s=45, label='Train data', zorder=3)
        ax.scatter(X_test, y_test, color='#f97316', edgecolors='#c2410c', s=55, marker='s', label='Test data', zorder=3)
        ax.plot(X_plot, y_curve, color=col, linewidth=2.5, label=f'Degree {deg} Model', zorder=4)

        ax.set_ylim(-20, 25)
        ax.set_title(title, fontsize=11, fontweight='bold', color='#1e293b', pad=10)
        ax.set_xlabel("$X$", fontsize=10)
        if i == 0:
            ax.set_ylabel("Target $y$", fontsize=10)

        # Performance badge
        stats_box = (
            f"Train RMSE: {train_rmse:.2f}\n"
            f"Test RMSE:  {test_rmse:.2f}\n"
            f"Train $R^2$:  {train_r2:.2f}\n"
            f"Test $R^2$:   {test_r2:.2f}"
        )
        ax.text(0.05, 0.95, stats_box, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8fafc', edgecolor='#cbd5e1', alpha=0.9))
        ax.legend(loc='lower left', fontsize=9, framealpha=0.9)

    fig.suptitle("Polynomial Regression: The Bias-Variance Tradeoff Across Model Complexities", fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "visual_2_polynomial_degrees.png")
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  💾 Chart saved: {os.path.basename(out_path)}\n")


# ====================================================
# STEP 3: Ridge (L2) vs. Lasso (L1) Regularization
# ====================================================
def run_regularization():
    print("=" * 60)
    print("  3️⃣  REGULARIZATION: RIDGE (L2) VS LASSO (L1)")
    print("=" * 60)

    np.random.seed(42)
    # Polynomial features of degree 10
    n_samples = 30
    X_raw = np.sort(np.random.uniform(-2.5, 2.5, n_samples))
    y = 1.5 * (X_raw ** 2) - 3.0 * X_raw + 2.0 + np.random.normal(0, 1.8, n_samples)
    X = X_raw.reshape(-1, 1)

    poly = PolynomialFeatures(degree=8, include_bias=False)
    X_poly = poly.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_poly)

    # Spectrum of alpha values
    alphas = np.logspace(-3, 3, 100)
    
    ridge_coefs = []
    lasso_coefs = []

    for a in alphas:
        r = Ridge(alpha=a, max_iter=10000)
        r.fit(X_scaled, y)
        ridge_coefs.append(r.coef_)

        l = Lasso(alpha=a, max_iter=10000)
        l.fit(X_scaled, y)
        lasso_coefs.append(l.coef_)

    ridge_coefs = np.array(ridge_coefs)
    lasso_coefs = np.array(lasso_coefs)

    # Check sparsity at alpha = 0.5
    l_sample = Lasso(alpha=0.5, max_iter=10000).fit(X_scaled, y)
    r_sample = Ridge(alpha=0.5, max_iter=10000).fit(X_scaled, y)
    zero_weights_lasso = np.sum(np.abs(l_sample.coef_) < 1e-4)
    zero_weights_ridge = np.sum(np.abs(r_sample.coef_) < 1e-4)

    print(f"  Comparing at alpha = 0.5 (Degree 8 Polynomial):")
    print(f"  • Ridge coefficients zeroed out: {zero_weights_ridge} of {len(r_sample.coef_)} (Shrinks softly)")
    print(f"  • Lasso coefficients zeroed out: {zero_weights_lasso} of {len(l_sample.coef_)} (Exact feature selection!)")

    # Plot Visual 3
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)

    # Ridge path
    for col_idx in range(ridge_coefs.shape[1]):
        ax1.plot(alphas, ridge_coefs[:, col_idx], label=f'Weight $w_{{{col_idx+1}}}$', linewidth=1.8)
    ax1.set_xscale('log')
    ax1.set_title("Ridge Regression ($L_2$ Penalty): Smooth Weight Shrinkage", fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel("Regularization Strength $\\alpha$ (Log scale)", fontsize=10)
    ax1.set_ylabel("Coefficient Values", fontsize=10)
    ax1.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.7)

    # Lasso path
    for col_idx in range(lasso_coefs.shape[1]):
        ax2.plot(alphas, lasso_coefs[:, col_idx], label=f'Weight $w_{{{col_idx+1}}}$', linewidth=1.8)
    ax2.set_xscale('log')
    ax2.set_title("Lasso Regression ($L_1$ Penalty): Sparse Feature Selection (Zeros)", fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel("Regularization Strength $\\alpha$ (Log scale)", fontsize=10)
    ax2.set_ylabel("Coefficient Values", fontsize=10)
    ax2.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.7)

    ax2.annotate('Coefficients drop strictly to 0', xy=(0.8, 0), xytext=(0.05, 3),
                 arrowprops=dict(facecolor='#ef4444', shrink=0.08, width=1.5, headwidth=6),
                 fontsize=9, fontweight='bold', color='#dc2626',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#fee2e2', edgecolor='#fca5a5'))

    ax1.legend(loc='upper right', fontsize=8, ncol=2)
    ax2.legend(loc='upper right', fontsize=8, ncol=2)
    fig.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "visual_3_ridge_lasso.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  💾 Chart saved: {os.path.basename(out_path)}\n")


# ====================================================
# MAIN RUNNER
# ====================================================
if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  🚀 WEEK 7 MINI-PROJECT: REGRESSION MODELS SUITE")
    print("=" * 60 + "\n")

    run_linear_regression()
    run_polynomial_regression()
    run_regularization()

    print("=" * 60)
    print("  🎉 All 3 regression visual charts generated successfully!")
    print(f"  📁 Location: {OUTPUT_DIR}")
    print("=" * 60 + "\n")

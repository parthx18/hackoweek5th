"""
====================================================
Week 13 Mini-Project: Ensemble Battleground Benchmark
====================================================
Demonstrates:
  1. High-variance Baseline: Single Decision Tree
  2. Bagging Ensemble: Random Forest (Variance Reduction)
  3. Histogram Boosting: HistGradientBoosting
  4. SOTA 2nd-Order Booster: XGBoost (Regularized Leaf Weights)
  5. SOTA Leaf-wise Booster: LightGBM (Histogram + GOSS)
  6. Visual comparison of Decision Boundaries & Metrics
====================================================
Run:    python ensemble_benchmark.py
Output: Saves 'ensemble_performance_comparison.png'
====================================================
"""

import os
import sys
import time

# Ensure UTF-8 output on Windows console
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

# Import XGBoost and LightGBM
import xgboost as xgb
import lightgbm as lgb

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    print("=" * 70)
    print("  WEEK 13: ENSEMBLE METHODS (BAGGING & BOOSTING) BENCHMARK")
    print("=" * 70)

    # 1. Generate Non-linear Dataset (Moons with Noise)
    X, y = make_moons(n_samples=1200, noise=0.28, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    print(f"\n[1] Dataset Profile (Non-linear Moons Manifold):")
    print(f"    • Total Samples:    {len(X)}")
    print(f"    • Training Samples: {len(X_train)} (75%)")
    print(f"    • Testing Samples:  {len(X_test)} (25%)")
    print(f"    • Features:         2 coordinates (X1, X2)")

    # 2. Define Model Arsenal
    models = {
        "Single Decision Tree": DecisionTreeClassifier(max_depth=None, random_state=42),
        "Random Forest (Bagging)": RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
        "XGBoost (2nd Order Boosting)": xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.08,
            max_depth=4,
            reg_alpha=0.1,
            reg_lambda=1.0,
            eval_metric="logloss",
            random_state=42
        ),
        "LightGBM (Leaf-wise Boosting)": lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.08,
            num_leaves=15,
            verbosity=-1,
            random_state=42
        )
    }

    results = []

    print(f"\n[2] Benchmarking Model Arsenal:")
    print("-" * 70)
    print(f"{'Model':<30} | {'Train Acc':<10} | {'Test Acc':<10} | {'Test AUC':<10} | {'Time (ms)':<10}")
    print("-" * 70)

    fitted_models = {}

    for name, clf in models.items():
        t0 = time.time()
        clf.fit(X_train, y_train)
        elapsed_ms = (time.time() - t0) * 1000

        train_acc = accuracy_score(y_train, clf.predict(X_train)) * 100
        test_preds = clf.predict(X_test)
        test_probs = clf.predict_proba(X_test)[:, 1]

        test_acc = accuracy_score(y_test, test_preds) * 100
        test_auc = roc_auc_score(y_test, test_probs) * 100
        loss = log_loss(y_test, test_probs)

        fitted_models[name] = clf
        results.append({
            "name": name,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "test_auc": test_auc,
            "log_loss": loss,
            "time_ms": elapsed_ms
        })

        print(f"{name:<30} | {train_acc:9.2f}% | {test_acc:9.2f}% | {test_auc:9.2f}% | {elapsed_ms:9.2f}ms")

    print("-" * 70)

    # 3. Visual Comparison (4 Panels)
    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    plt.subplots_adjust(hspace=0.28, wspace=0.22)

    # Setup mesh grid for decision boundary plotting
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250), np.linspace(y_min, y_max, 250))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    plot_configs = [
        ("Single Decision Tree", axes[0, 0], "High Variance (Jagged / Overfit Boundary)"),
        ("Random Forest (Bagging)", axes[0, 1], "Variance Smoothed via Tree Averaging"),
        ("XGBoost (2nd Order Boosting)", axes[1, 0], "Regularized Gradients & Hessians (Sharp Boundary)"),
        ("LightGBM (Leaf-wise Boosting)", axes[1, 1], "Leaf-wise Optimal Gain (Smooth Generalization)")
    ]

    for name, ax, subtitle in plot_configs:
        clf = fitted_models[name]
        Z = clf.predict_proba(grid_points)[:, 1]
        Z = Z.reshape(xx.shape)

        # Plot decision contour probability
        contour = ax.contourf(xx, yy, Z, levels=20, cmap="coolwarm", alpha=0.75)
        ax.contour(xx, yy, Z, levels=[0.5], colors='k', linewidths=1.8, linestyles='--')

        # Scatter actual test points
        ax.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], color='#3b82f6', edgecolors='k', s=35, label='Class 0')
        ax.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], color='#ef4444', edgecolors='k', s=35, label='Class 1')

        # Get metric string
        r = next(item for item in results if item["name"] == name)
        ax.set_title(f"{name}\n{subtitle}\nTest Acc: {r['test_acc']:.1f}% | AUC: {r['test_auc']:.1f}%", fontsize=11, fontweight='bold')
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.grid(True, linestyle=':', alpha=0.4)
        if ax == axes[0, 0]:
            ax.legend(loc='lower left', frameon=True)

    output_plot_path = os.path.join(OUTPUT_DIR, "ensemble_performance_comparison.png")
    plt.savefig(output_plot_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"\n[3] Visual Comparison Saved Successfully:")
    print(f"    • Chart: {output_plot_path}")
    print("\n" + "=" * 70)
    print("  WEEK 13 ENSEMBLE BENCHMARK COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()

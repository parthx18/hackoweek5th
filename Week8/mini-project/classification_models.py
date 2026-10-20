"""
====================================================
Mini-Project: Classification Lab & Model Evaluation
Week 8 — Classification (Machine Learning Foundations)
====================================================
Concepts used:
  ✅ Logistic Regression & Sigmoid Probability Landscape
  ✅ K-Nearest Neighbors (KNN) Decision Boundaries (k = 1, 5, 25)
  ✅ Confusion Matrix & Metric Analysis (Precision, Recall, F1)
  ✅ ROC Curve & Area Under Curve (AUC)
====================================================
Run:    python classification_models.py
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
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_blobs, make_moons, make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score
)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cbd5e1'
plt.rcParams['axes.linewidth'] = 0.8

# ====================================================
# STEP 1: Logistic Regression & Decision Boundary
# ====================================================
def run_logistic_regression():
    print("=" * 60)
    print("  1️⃣  LOGISTIC REGRESSION & PROBABILITY SURFACE")
    print("=" * 60)

    # Linearly separable clusters with slight overlap
    blobs_data = make_blobs(n_samples=120, centers=[[-1.2, -1.0], [1.2, 1.2]], cluster_std=0.85, random_state=42)
    X, y = blobs_data[0], blobs_data[1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    w1, w2 = model.coef_[0]
    b = model.intercept_[0]

    print(f"  Learned Weights: w1 = {w1:.3f}, w2 = {w2:.3f}, bias = {b:.3f}")
    print(f"  Decision Boundary: {w1:.3f}*x1 + {w2:.3f}*x2 + ({b:.3f}) = 0")
    print(f"  --------------------------------------------------")
    print(f"  Test Accuracy:   {acc*100:.1f}%")
    print(f"  Precision:       {prec:.3f}")
    print(f"  Recall:          {rec:.3f}")
    print(f"  F1 Score:        {f1:.3f}")

    # Plot Visual 1
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=150)

    # Create meshgrid for probability surface
    x_min, x_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    y_min, y_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250), np.linspace(y_min, y_max, 250))
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = model.predict_proba(grid)[:, 1].reshape(xx.shape)

    # Contour filled with smooth probability colors
    contour = ax.contourf(xx, yy, probs, levels=25, cmap='RdBu_r', alpha=0.6, vmin=0, vmax=1)
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label('$P(y = 1 \\mid \\mathbf{x})$ Probability', fontsize=11)

    # Plot 0.5 decision boundary line
    ax.contour(xx, yy, probs, levels=[0.5], colors=['#0f172a'], linewidths=2.5, linestyles='--')

    # Scatter points
    ax.scatter(X[y == 0, 0], X[y == 0, 1], color='#ef4444', edgecolors='#991b1b', s=55, label='Class 0 (Negative)', zorder=4)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], color='#3b82f6', edgecolors='#1e40af', s=55, label='Class 1 (Positive)', zorder=4)

    ax.set_title("Logistic Regression: Sigmoid Probability Landscape & Decision Boundary", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Feature $X_1$", fontsize=11)
    ax.set_ylabel("Feature $X_2$", fontsize=11)

    # Info card
    stats_box = (
        f"Model Performance:\n"
        f"• Accuracy:  {acc*100:.1f}%\n"
        f"• Precision: {prec:.2f}\n"
        f"• Recall:    {rec:.2f}\n"
        f"• F1 Score:  {f1:.2f}\n"
        f"Dashed Line: P = 0.5 Boundary"
    )
    ax.text(0.03, 0.96, stats_box, transform=ax.transAxes, fontsize=9.5,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.6', facecolor='#f8fafc', edgecolor='#cbd5e1', alpha=0.9))

    ax.legend(loc='lower right', framealpha=0.9)
    fig.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "visual_1_logistic_regression.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  💾 Chart saved: {os.path.basename(out_path)}\n")


# ====================================================
# STEP 2: KNN & Decision Surface Smoothing (k = 1, 5, 25)
# ====================================================
def run_knn_analysis():
    print("=" * 60)
    print("  2️⃣  K-NEAREST NEIGHBORS: IMPACT OF K (OVERFIT VS SMOOTH)")
    print("=" * 60)

    # Non-linear interlocking moons dataset
    X, y = make_moons(n_samples=160, noise=0.25, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    k_values = [1, 5, 25]
    titles = [
        "k = 1 (Overfitting / High Variance)",
        "k = 5 (Balanced Neighborhood)",
        "k = 25 (Over-smoothed / High Bias)"
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=150, sharey=True)

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]

    for i, (k, title) in enumerate(zip(k_values, titles)):
        ax = axes[i]
        knn = KNeighborsClassifier(n_neighbors=k, weights='uniform')
        knn.fit(X_train, y_train)

        train_acc = knn.score(X_train, y_train)
        test_acc = knn.score(X_test, y_test)

        print(f"  • k = {k:2d} -> Train Accuracy: {train_acc*100:5.1f}% | Test Accuracy: {test_acc*100:5.1f}%")

        # Predict surface mesh
        Z = knn.predict(grid).reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap='coolwarm', alpha=0.35, levels=[-0.5, 0.5, 1.5])
        ax.contour(xx, yy, Z, levels=[0.5], colors=['#0f172a'], linewidths=2.0)

        # Plot data points
        ax.scatter(X[y == 0, 0], X[y == 0, 1], color='#ef4444', edgecolors='#991b1b', s=35, label='Class 0' if i == 0 else "")
        ax.scatter(X[y == 1, 0], X[y == 1, 1], color='#3b82f6', edgecolors='#1e40af', s=35, label='Class 1' if i == 0 else "")

        ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
        ax.set_xlabel("Feature $X_1$", fontsize=10)
        if i == 0:
            ax.set_ylabel("Feature $X_2$", fontsize=10)

        # Stats box
        stats_box = f"Train Acc: {train_acc*100:.1f}%\nTest Acc:  {test_acc*100:.1f}%"
        ax.text(0.04, 0.95, stats_box, transform=ax.transAxes, fontsize=9.5,
                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8fafc', edgecolor='#cbd5e1', alpha=0.9))

        if i == 0:
            ax.legend(loc='lower left', fontsize=9)

    fig.suptitle("K-Nearest Neighbors: How Neighborhood Size (k) Alters Decision Boundaries", fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "visual_2_knn_decision_surfaces.png")
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  💾 Chart saved: {os.path.basename(out_path)}\n")


# ====================================================
# STEP 3: Metrics Suite, Confusion Matrix & ROC Curve
# ====================================================
def run_metrics_evaluation():
    print("=" * 60)
    print("  3️⃣  CLASSIFICATION METRICS, CONFUSION MATRIX & ROC-AUC")
    print("=" * 60)

    # Imbalanced synthetic dataset
    X, y = make_classification(
        n_samples=250, n_features=4, n_informative=3, n_redundant=0,
        weights=[0.7, 0.3], random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"  Confusion Matrix:")
    print(f"  [[TN={tn:2d}, FP={fp:2d}],")
    print(f"   [FN={fn:2d}, TP={tp:2d}]]")
    print(f"  Accuracy:  {acc:.3f} | Precision: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f} | AUC: {auc:.3f}")

    # Plot Visual 3
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)

    # 1. Confusion Matrix Heatmap
    cax = ax1.matshow(cm, cmap='Blues', alpha=0.85)
    fig.colorbar(cax, ax=ax1, fraction=0.046, pad=0.04)

    labels = [
        [f"TN = {tn}\n(True Neg)", f"FP = {fp}\n(Type I Err)"],
        [f"FN = {fn}\n(Type II Err)", f"TP = {tp}\n(True Pos)"]
    ]
    for r in range(2):
        for c in range(2):
            ax1.text(c, r, labels[r][c], ha='center', va='center', fontsize=11,
                     fontweight='bold', color='#0f172a' if cm[r, c] < (tn+tp)/2 else '#ffffff')

    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(['Pred Class 0', 'Pred Class 1'], fontsize=10)
    ax1.set_yticklabels(['Actual Class 0', 'Actual Class 1'], fontsize=10)
    ax1.set_title("Confusion Matrix Breakdown", fontsize=12, fontweight='bold', pad=14)

    # 2. ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    ax2.plot(fpr, tpr, color='#2563eb', linewidth=2.5, label=f'ROC Curve (AUC = {auc:.3f})')
    ax2.plot([0, 1], [0, 1], color='#94a3b8', linestyle='--', linewidth=1.5, label='Random Chance (AUC = 0.50)')

    ax2.fill_between(fpr, tpr, alpha=0.15, color='#3b82f6')
    ax2.set_xlim([-0.02, 1.02])
    ax2.set_ylim([-0.02, 1.05])
    ax2.set_title("Receiver Operating Characteristic (ROC)", fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10)
    ax2.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=10)
    ax2.legend(loc='lower right', framealpha=0.9)

    fig.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "visual_3_confusion_matrix_roc.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  💾 Chart saved: {os.path.basename(out_path)}\n")


# ====================================================
# MAIN RUNNER
# ====================================================
if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  🚀 WEEK 8 MINI-PROJECT: CLASSIFICATION MODELS SUITE")
    print("=" * 60 + "\n")

    run_logistic_regression()
    run_knn_analysis()
    run_metrics_evaluation()

    print("=" * 60)
    print("  🎉 All 3 classification visual charts generated successfully!")
    print(f"  📁 Location: {OUTPUT_DIR}")
    print("=" * 60 + "\n")

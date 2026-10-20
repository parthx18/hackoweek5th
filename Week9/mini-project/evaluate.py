"""
====================================================
Week 9 Mini-Project: Model Evaluation Pipeline
====================================================
Demonstrates:
  1. Train / Test Split
  2. 5-Fold Cross Validation
  3. Confusion Matrix Breakdown
  4. Precision, Recall, and F1-Score
  5. ROC Curve & AUC Score
====================================================
Run:    python evaluate.py
Output: Saves 'visual_evaluation_metrics.png'
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
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
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

def main():
    print("=" * 60)
    print("  WEEK 9: MODEL EVALUATION BENCHMARK")
    print("=" * 60)

    # 1. Dataset Generation (e.g. Churn prediction: 80% retained, 20% churned)
    X, y = make_classification(
        n_samples=300,
        n_features=6,
        n_informative=4,
        weights=[0.75, 0.25],
        random_state=42
    )

    # 2. Train / Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    print(f"\n[1] Train / Test Split:")
    print(f"    • Total Samples:    {len(X)}")
    print(f"    • Training Samples: {len(X_train)} (80%)")
    print(f"    • Testing Samples:  {len(X_test)} (20%)")

    # 3. Model Definition
    model = LogisticRegression(random_state=42)

    # 4. K-Fold Cross Validation (5 Folds)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"\n[2] 5-Fold Cross-Validation on Training Data:")
    for i, s in enumerate(cv_scores, 1):
        print(f"    • Fold {i}: {s * 100:.1f}%")
    print(f"    --> Mean Accuracy: {cv_scores.mean() * 100:.1f}% (+/- {cv_scores.std() * 100:.1f}%)")

    # 5. Fit model on full training set and evaluate on test set
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # 6. Evaluation Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n[3] Test Set Results:")
    print(f"    • Accuracy:  {acc * 100:.1f}%")
    print(f"    • Precision: {prec:.3f}  (Of predicted positives, how many true?)")
    print(f"    • Recall:    {rec:.3f}  (Of actual positives, how many caught?)")
    print(f"    • F1-Score:  {f1:.3f}  (Harmonic mean of precision & recall)")
    print(f"    • ROC-AUC:   {auc:.3f}  (Discrimination ability)")

    print(f"\n[4] Confusion Matrix Breakdown:")
    print(f"    • True Negatives  (TN): {tn:2d}  [Correctly identified Class 0]")
    print(f"    • False Positives (FP): {fp:2d}  [Type I Error: False Alarm]")
    print(f"    • False Negatives (FN): {fn:2d}  [Type II Error: Missed Case]")
    print(f"    • True Positives  (TP): {tp:2d}  [Correctly caught Class 1]")

    # 7. Visualization Chart
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

    # Panel 1: Confusion Matrix Heatmap
    cax = ax1.matshow(cm, cmap='Blues', alpha=0.8)
    fig.colorbar(cax, ax=ax1, fraction=0.046, pad=0.04)

    labels = [
        [f"TN = {tn}\n(True Neg)", f"FP = {fp}\n(Type I Error)"],
        [f"FN = {fn}\n(Type II Error)", f"TP = {tp}\n(True Pos)"]
    ]
    for r in range(2):
        for c in range(2):
            ax1.text(c, r, labels[r][c], ha='center', va='center', fontsize=11, fontweight='bold',
                     color='#0f172a' if cm[r, c] < (tn + tp) / 2 else '#ffffff')

    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(['Pred Negative', 'Pred Positive'], fontsize=10)
    ax1.set_yticklabels(['Actual Negative', 'Actual Positive'], fontsize=10)
    ax1.set_title("Confusion Matrix Heatmap", fontsize=12, fontweight='bold', pad=15)

    # Panel 2: ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ax2.plot(fpr, tpr, color='#2563eb', linewidth=2.5, label=f'ROC (AUC = {auc:.3f})')
    ax2.plot([0, 1], [0, 1], color='#94a3b8', linestyle='--', linewidth=1.5, label='Random Chance (AUC = 0.50)')
    ax2.fill_between(fpr, tpr, alpha=0.15, color='#3b82f6')
    ax2.set_xlim([-0.02, 1.02])
    ax2.set_ylim([-0.02, 1.05])
    ax2.set_title("ROC Curve (Receiver Operating Characteristic)", fontsize=12, fontweight='bold', pad=12)
    ax2.set_xlabel("False Positive Rate (FP / (FP + TN))", fontsize=10)
    ax2.set_ylabel("True Positive Rate (Recall / TP Rate)", fontsize=10)
    ax2.legend(loc='lower right', fontsize=10)

    fig.tight_layout()
    chart_path = os.path.join(OUTPUT_DIR, "visual_evaluation_metrics.png")
    fig.savefig(chart_path)
    plt.close(fig)

    print(f"\n[5] Visualization Saved:")
    print(f"    • {os.path.basename(chart_path)}")
    print("=" * 60 + "\n")

if __name__ == '__main__':
    main()

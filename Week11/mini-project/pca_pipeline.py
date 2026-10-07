"""
====================================================
Week 11 Mini-Project: PCA Dimensionality Reduction Pipeline
====================================================
Demonstrates:
  1. Standardizing high-dimensional features
  2. Covariance Matrix calculation & Eigendecomposition
  3. Explained Variance Ratio (EVR) & Cumulative Scree Plot
  4. Projecting 4D Iris dataset onto 2D Principal Components
  5. Comparing Raw Features vs. PCA-transformed space
  6. Visualizing Eigenvector loadings
====================================================
Run:    python pca_pipeline.py
Output: Saves 'pca_analysis_visualization.png'
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
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    print("=" * 65)
    print("  WEEK 11: PRINCIPAL COMPONENT ANALYSIS (PCA) PIPELINE")
    print("=" * 65)

    # 1. Load Dataset (Iris: 150 samples, 4 dimensions)
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names

    print(f"\n[1] Dataset Overview (Iris Dataset):")
    print(f"    • Total Samples:    {X.shape[0]}")
    print(f"    • Original Dimensions: {X.shape[1]}")
    print(f"    • Feature Names:    {', '.join(feature_names)}")
    print(f"    • Target Classes:   {', '.join(target_names)}")

    # 2. Standardization (Z-score)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"\n[2] Feature Standardization Completed:")
    print(f"    • Mean per feature:  {np.round(X_scaled.mean(axis=0), 2)}")
    print(f"    • Std per feature:   {np.round(X_scaled.std(axis=0), 2)}")

    # 3. Covariance Matrix & Eigendecomposition (NumPy Verification)
    cov_matrix = np.cov(X_scaled.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort eigenvalues & eigenvectors descending
    sort_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sort_idx]
    eigenvectors = eigenvectors[:, sort_idx]

    print(f"\n[3] Eigendecomposition from Covariance Matrix:")
    for i, (val, vec) in enumerate(zip(eigenvalues, eigenvectors.T), 1):
        print(f"    • Eigenvalue λ_{i}: {val:.4f} | Direction vector: {np.round(vec, 3)}")

    # 4. Fit Scikit-Learn PCA
    pca_full = PCA()
    pca_full.fit(X_scaled)
    evr = pca_full.explained_variance_ratio_
    cum_evr = np.cumsum(evr)

    print(f"\n[4] Explained Variance Analysis:")
    for i, (ratio, cum) in enumerate(zip(evr, cum_evr), 1):
        print(f"    • PC{i}: {ratio * 100:6.2f}% variance explained | Cumulative: {cum * 100:6.2f}%")

    # Fit 2-Component PCA for Projection
    pca_2d = PCA(n_components=2)
    X_pca = pca_2d.fit_transform(X_scaled)

    total_2d_variance = cum_evr[1] * 100
    print(f"\n[5] 2D Projection Summary:")
    print(f"    • 2 Principal Components retain {total_2d_variance:.2f}% of total dataset variance!")
    print(f"    • Dimension reduction: 4D -> 2D (50% compression, ~{total_2d_variance:.1f}% information retained)")

    # 5. Visualization (4-panel professional figure)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    plt.subplots_adjust(hspace=0.35, wspace=0.25)
    colors = ['#4f46e5', '#06b6d4', '#10b981']

    # Subplot 1: Original 2D slice (Sepal Length vs Sepal Width)
    for class_id, color, name in zip(range(3), colors, target_names):
        mask = (y == class_id)
        axes[0, 0].scatter(X[mask, 0], X[mask, 1], c=color, label=name, alpha=0.8, edgecolors='none', s=45)
    axes[0, 0].set_title("1. Original Features (Sepal Length vs Width)\n(Overlap between Versicolor & Virginica)", fontsize=11, fontweight='bold')
    axes[0, 0].set_xlabel("Sepal Length (cm)")
    axes[0, 0].set_ylabel("Sepal Width (cm)")
    axes[0, 0].legend(frameon=True)
    axes[0, 0].grid(True, linestyle='--', alpha=0.5)

    # Subplot 2: Projected 2D PCA Space (PC1 vs PC2)
    for class_id, color, name in zip(range(3), colors, target_names):
        mask = (y == class_id)
        axes[0, 1].scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, label=name, alpha=0.85, edgecolors='none', s=45)
    axes[0, 1].set_title(f"2. PCA 2D Projection ($PC_1$ vs $PC_2$)\n({total_2d_variance:.1f}% Variance Retained - Clear Separation)", fontsize=11, fontweight='bold')
    axes[0, 1].set_xlabel(f"Principal Component 1 ({evr[0]*100:.1f}%)")
    axes[0, 1].set_ylabel(f"Principal Component 2 ({evr[1]*100:.1f}%)")
    axes[0, 1].legend(frameon=True)
    axes[0, 1].grid(True, linestyle='--', alpha=0.5)

    # Subplot 3: Scree Plot (Individual & Cumulative Explained Variance)
    pcs = [f"PC{i+1}" for i in range(len(evr))]
    axes[1, 0].bar(pcs, evr * 100, color='#6366f1', alpha=0.7, label='Individual EVR (%)', width=0.45)
    axes[1, 0].plot(pcs, cum_evr * 100, color='#f43f5e', marker='o', linewidth=2.2, label='Cumulative EVR (%)')
    axes[1, 0].axhline(y=95, color='#eab308', linestyle=':', label='95% Threshold')
    for i, c_val in enumerate(cum_evr):
        axes[1, 0].annotate(f"{c_val*100:.1f}%", (i, c_val * 100 + 2), ha='center', fontsize=9, fontweight='semibold')
    axes[1, 0].set_title("3. Scree Plot: Explained Variance Ratio", fontsize=11, fontweight='bold')
    axes[1, 0].set_xlabel("Principal Components")
    axes[1, 0].set_ylabel("Variance Explained (%)")
    axes[1, 0].set_ylim(0, 110)
    axes[1, 0].legend(loc='center right', frameon=True)
    axes[1, 0].grid(True, linestyle='--', alpha=0.5)

    # Subplot 4: Feature Contribution / Loadings Heatmap
    loadings = pd.DataFrame(
        pca_2d.components_.T,
        columns=['PC1', 'PC2'],
        index=[f.replace(" (cm)", "") for f in feature_names]
    )
    cax = axes[1, 1].matshow(loadings, cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(cax, ax=axes[1, 1], fraction=0.046, pad=0.04)
    axes[1, 1].set_xticks([0, 1])
    axes[1, 1].set_xticklabels(['PC1', 'PC2'], fontsize=10, fontweight='bold')
    axes[1, 1].set_yticks(range(len(loadings.index)))
    axes[1, 1].set_yticklabels(loadings.index, fontsize=10)
    axes[1, 1].set_title("4. Feature Loadings (Eigenvector Weights)\nComponent Influence on Original Variables", fontsize=11, fontweight='bold', pad=15)
    for i in range(loadings.shape[0]):
        for j in range(loadings.shape[1]):
            val = loadings.iloc[i, j]
            axes[1, 1].text(j, i, f"{val:.2f}", ha='center', va='center', color='black' if abs(val) < 0.6 else 'white', fontweight='bold')

    output_plot_path = os.path.join(OUTPUT_DIR, "pca_analysis_visualization.png")
    plt.savefig(output_plot_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"\n[6] Visual Output Generated Successfully:")
    print(f"    • Chart saved at: {output_plot_path}")
    print("\n" + "=" * 65)
    print("  WEEK 11 PCA RUN COMPLETE")
    print("=" * 65)

if __name__ == "__main__":
    main()

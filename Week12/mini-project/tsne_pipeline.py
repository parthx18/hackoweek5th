"""
====================================================
Week 12 Mini-Project: t-SNE vs PCA Benchmark & Perplexity Lab
====================================================
Demonstrates:
  1. Loading high-dimensional dataset (Digits: 64 dimensions / 8x8 pixels)
  2. Standardizing features for dimensionality reduction
  3. Linear projection with PCA (showing cluster overlap)
  4. Non-linear manifold embedding with t-SNE (uncovering distinct clusters)
  5. Comparing Perplexity effects (Perplexity = 5, 30, 50)
  6. Visual benchmark output saved to PNG
====================================================
Run:    python tsne_pipeline.py
Output: Saves 'tsne_comparison_visualization.png'
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
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    print("=" * 65)
    print("  WEEK 12: t-SNE (INTUITION & BENCHMARK) PIPELINE")
    print("=" * 65)

    # 1. Load Digits Dataset (subset for fast, clean visualization)
    digits = load_digits()
    n_samples = 800
    X = digits.data[:n_samples]
    y = digits.target[:n_samples]

    print(f"\n[1] Dataset Overview (Handwritten Digits):")
    print(f"    • Total Samples Selected: {X.shape[0]}")
    print(f"    • High-Dimensional Input:  {X.shape[1]} features (8x8 pixel grid)")
    print(f"    • Classes:                 Digits 0 through 9")

    # 2. Standardize Features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Fit PCA as Baseline Comparison
    print(f"\n[2] Running Linear Baseline (PCA)...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    var_retained = np.sum(pca.explained_variance_ratio_) * 100
    print(f"    • PCA 2D Explained Variance: {var_retained:.2f}%")

    # 4. Fit t-SNE across different Perplexities
    perplexities = [5, 30, 50]
    tsne_results = {}

    for perp in perplexities:
        print(f"\n[3] Running Non-linear t-SNE (Perplexity = {perp})...")
        tsne = TSNE(
            n_components=2,
            perplexity=perp,
            max_iter=1000,
            init='pca',
            learning_rate='auto',
            random_state=42
        )
        X_tsne = tsne.fit_transform(X_scaled)
        tsne_results[perp] = X_tsne
        print(f"    • Converged with KL Divergence: {tsne.kl_divergence_:.4f}")

    # 5. Visual Comparison (4-Panel Figure)
    print(f"\n[4] Generating Comparative Visualizations...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    plt.subplots_adjust(hspace=0.28, wspace=0.22)
    cmap = plt.colormaps['tab10']

    # Panel 1: PCA Projection
    scatter1 = axes[0, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap=cmap, alpha=0.75, s=30, edgecolors='none')
    axes[0, 0].set_title(f"1. Linear PCA Projection (Variance Retained: {var_retained:.1f}%)\n(High cluster overlap: linear plane cannot unfold manifold)", fontsize=11, fontweight='bold')
    axes[0, 0].set_xlabel("PC 1")
    axes[0, 0].set_ylabel("PC 2")
    axes[0, 0].grid(True, linestyle='--', alpha=0.4)

    # Panel 2: t-SNE with Low Perplexity (5)
    axes[0, 1].scatter(tsne_results[5][:, 0], tsne_results[5][:, 1], c=y, cmap=cmap, alpha=0.75, s=30, edgecolors='none')
    axes[0, 1].set_title("2. t-SNE (Perplexity = 5 — Local/Microscopic Focus)\n(Clusters fragment into smaller sub-islands)", fontsize=11, fontweight='bold')
    axes[0, 1].set_xlabel("t-SNE Dimension 1")
    axes[0, 1].set_ylabel("t-SNE Dimension 2")
    axes[0, 1].grid(True, linestyle='--', alpha=0.4)

    # Panel 3: t-SNE with Optimal Perplexity (30)
    axes[1, 0].scatter(tsne_results[30][:, 0], tsne_results[30][:, 1], c=y, cmap=cmap, alpha=0.8, s=30, edgecolors='none')
    axes[1, 0].set_title("3. t-SNE (Perplexity = 30 — Balanced Neighborhood)\n(Clean, distinct clusters for each digit 0-9)", fontsize=11, fontweight='bold')
    axes[1, 0].set_xlabel("t-SNE Dimension 1")
    axes[1, 0].set_ylabel("t-SNE Dimension 2")
    axes[1, 0].grid(True, linestyle='--', alpha=0.4)

    # Panel 4: t-SNE with High Perplexity (50)
    scatter4 = axes[1, 1].scatter(tsne_results[50][:, 0], tsne_results[50][:, 1], c=y, cmap=cmap, alpha=0.75, s=30, edgecolors='none')
    axes[1, 1].set_title("4. t-SNE (Perplexity = 50 — Broader/Global Focus)\n(Clusters coalesce; global geometry begins to dominate)", fontsize=11, fontweight='bold')
    axes[1, 1].set_xlabel("t-SNE Dimension 1")
    axes[1, 1].set_ylabel("t-SNE Dimension 2")
    axes[1, 1].grid(True, linestyle='--', alpha=0.4)

    # Add shared colorbar for digits
    cbar = fig.colorbar(scatter4, ax=axes, orientation='horizontal', fraction=0.035, pad=0.06, ticks=range(10))
    cbar.set_label('Digit Class (0 to 9)', fontsize=11, fontweight='bold')

    output_plot_path = os.path.join(OUTPUT_DIR, "tsne_comparison_visualization.png")
    plt.savefig(output_plot_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"\n[5] Visual Output Generated Successfully:")
    print(f"    • Comparison chart saved at: {output_plot_path}")
    print("\n" + "=" * 65)
    print("  WEEK 12 t-SNE BENCHMARK COMPLETE")
    print("=" * 65)

if __name__ == "__main__":
    main()

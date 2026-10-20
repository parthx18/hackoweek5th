# Week 12 — Dimensionality Reduction: t-SNE (t-Distributed Stochastic Neighbor Embedding)

## 🎯 Topics
- **Why Non-linear Dimensionality Reduction?**:
  - The limits of linear methods like PCA on manifolds (Swiss Roll, concentric spheres, complex clusters)
  - The "Crowding Problem" when projecting high-dimensional spheres into low dimensions
- **t-SNE Intuition & Principles**:
  - Converting pairwise Euclidean distances into conditional probabilities ($p_{j|i}$)
  - Neighborhood conservation: high probability if points are neighbors, near zero if far apart
  - Symmetrized probabilities ($p_{ij}$)
- **The Student's t-Distribution Secret**:
  - Normal distribution in high dimension vs. Student's t-distribution (1 degree of freedom / Cauchy) in low dimension
  - Heavy tails solve the crowding problem by pushing dissimilar clusters apart
- **Key Hyperparameters & Practical Best Practices**:
  - **Perplexity**: Effective number of local neighbors (typical range: 5 to 50)
  - **Learning Rate ($\eta$)**: Step size for gradient descent optimization (typical: 100 to 1000)
  - **Number of Iterations**: Allowing gradient descent to stabilize (typical: 1,000 to 2,000 steps)
  - **Initialization**: PCA initialization vs. random initialization
- **Comparing PCA vs. t-SNE**:
  - Global structure / variance preservation vs. Local cluster separation
  - Deterministic linear projection vs. Stochastic iterative optimization
  - Computational complexity ($O(N \log N)$ vs $O(N \cdot D^2)$)
  - Downstream modeling considerations (t-SNE is purely a visualization tool; PCA can transform unseen test data)

## 🗂️ Structure
```
Week12/
├── README.md
├── notes/
│   └── tsne-intuition-guide.md             ← Comprehensive guide to t-SNE intuition, math & comparisons
└── mini-project/
    ├── requirements.txt
    ├── tsne_pipeline.py                    ← Hands-on comparison script (PCA vs. t-SNE & Perplexity analysis)
    ├── tsne_comparison_visualization.png   ← Generated 4-panel visual comparison chart
    └── webpage/                            ← Interactive t-SNE & Perplexity simulator
        ├── index.html
        ├── style.css
        └── app.js
```

## 🚀 Mini-Project: t-SNE vs PCA Benchmark & Perplexity Lab
A Python script and visual explorer:
1. Loads the 64-dimensional Digits dataset ($8 \times 8$ grayscale handwritten digits)
2. Runs PCA projection into 2D to showcase where linear dimensionality reduction suffers overlap
3. Runs t-SNE across multiple perplexity values ($5, 30, 50$) showing how neighborhood density shapes cluster formation
4. Plots side-by-side benchmark visualizations in `tsne_comparison_visualization.png`
5. Provides an interactive web application in `webpage/index.html` to explore cluster formation, perplexity effects, and heavy-tailed distribution intuition live.

## 🔧 Setup & Run
```bash
cd Week12/mini-project
pip install -r requirements.txt
python tsne_pipeline.py
```

To open the interactive visualizer, open `Week12/mini-project/webpage/index.html` in your browser.

## ⏱️ Duration: 3 Hours
- **Hour 1**: Limitations of PCA on complex manifolds & t-SNE probabilistic formulation
- **Hour 2**: The Student's t-distribution, heavy tails, and solving the crowding problem
- **Hour 3**: Perplexity tuning, PCA vs. t-SNE comparative benchmarks, and interactive visual lab

# Week 12 — Dimensionality Reduction: t-SNE (Intuition & Applications)

## 1. Why Do We Need t-SNE? (The Limits of PCA)

In Week 11, we saw how **PCA** finds orthogonal linear hyperplanes of maximal global variance. However, many real-world datasets reside on **non-linear manifolds**:
- Handwritten digits ($0$ through $9$ in $64$ or $784$ dimensions)
- Single-cell RNA sequencing (clusters of cell types)
- Word embeddings (semantic word vectors)

If high-dimensional points lie along curved surfaces (like a Swiss Roll or concentric rings), PCA flattens the folds together, collapsing completely distinct clusters into an overlapping blur.

**t-SNE (t-Distributed Stochastic Neighbor Embedding)**, introduced by Laurens van der Maaten and Geoffrey Hinton in 2008, was invented to solve this exact problem: **preserving local neighborhood structures and uncovering natural clusters visually.**

---

## 2. Core Intuition: From Distances to Probabilities

Instead of trying to preserve absolute Euclidean distances or maximize variance, t-SNE asks:
> *"If I pick point $i$, what is the probability that point $j$ is its neighbor?"*

### Step 1: High-Dimensional Probabilities ($p_{j|i}$)
In the high-dimensional space, t-SNE models neighborhood relationships using a **Gaussian (Normal) distribution**:
$$ p_{j|i} = \frac{\exp(-\|\mathbf{x}_i - \mathbf{x}_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-\|\mathbf{x}_i - \mathbf{x}_k\|^2 / 2\sigma_i^2)}, \quad p_{i|i} = 0 $$
- If $\mathbf{x}_i$ and $\mathbf{x}_j$ are very close, $p_{j|i}$ is high.
- If they are far apart, $p_{j|i} \approx 0$.
- The bandwidth $\sigma_i$ is determined adaptively for each point based on a user-defined parameter called **Perplexity**.

To make computations symmetric:
$$ p_{ij} = \frac{p_{j|i} + p_{i|j}}{2N} $$

### Step 2: Low-Dimensional Probabilities ($q_{ij}$)
Now, place points $\mathbf{y}_i, \mathbf{y}_j$ randomly in a 2D map. In 2D, we compute the similarity $q_{ij}$ between map points.
Instead of a Gaussian, t-SNE uses a **Student's t-distribution with 1 degree of freedom (Cauchy distribution)**:
$$ q_{ij} = \frac{(1 + \|\mathbf{y}_i - \mathbf{y}_j\|^2)^{-1}}{\sum_{k} \sum_{l \neq k} (1 + \|\mathbf{y}_k - \mathbf{y}_l\|^2)^{-1}}, \quad q_{ii} = 0 $$

---

## 3. The Secret Sauce: The "Crowding Problem" & Heavy Tails

### What is the Crowding Problem?
In a 10-dimensional space, a sphere has immense volume, and you can place 10 equidistant points around a central point without crowding. But if you try to squeeze those same 10 points onto a 2D flat circle, there simply isn't enough circumference/area to keep them all equidistant! In standard SNE, moderate distances get artificially crushed together.

### Why the Student's t-Distribution Solves It:
The Student's t-distribution has **much heavier tails** than a Gaussian:
- At short distances, it behaves similarly to a Gaussian.
- At moderate-to-large distances, $(1 + d^2)^{-1}$ falls off much more slowly (power law) than $\exp(-d^2)$ (exponential).
- **Result:** To match a small probability $p_{ij}$ in high-dimensional space, the low-dimensional distance $\|\mathbf{y}_i - \mathbf{y}_j\|$ must be **pushed substantially farther apart**.
- This acts like an elastic repulsive force that cleanly pushes dissimilar clusters apart, leaving wide white space between clusters!

---

## 4. Minimizing the Divergence (Kullback-Leibler Divergence)

t-SNE measures how closely the low-dimensional distribution $Q$ matches the true high-dimensional distribution $P$ using **KL Divergence**:
$$ KL(P \parallel Q) = \sum_{i \neq j} p_{ij} \log \left(\frac{p_{ij}}{q_{ij}}\right) $$

Notice the asymmetric penalty:
- If $p_{ij}$ is large (points are neighbors in high dimensions), but $q_{ij}$ is small (points are far apart in 2D), the penalty $\mathbf{p_{ij} \log(p_{ij}/q_{ij})}$ is **extremely high**!
- If $p_{ij}$ is near 0 (points are far in high dimensions), whether $q_{ij}$ is large or small contributes almost 0 penalty.
- **Takeaway:** t-SNE cares fiercely about preserving **local neighbors**, and is relaxed about global inter-cluster arrangement.

Gradient descent is then used to update the 2D coordinates $\mathbf{y}_i$:
$$ \frac{\partial KL}{\partial \mathbf{y}_i} = 4 \sum_j (p_{ij} - q_{ij})(1 + \|\mathbf{y}_i - \mathbf{y}_j\|^2)^{-1} (\mathbf{y}_i - \mathbf{y}_j) $$

---

## 5. Key Hyperparameters & Practical Best Practices

### 1. Perplexity
- **Intuition:** A smooth measure of the effective number of nearest neighbors each point considers.
- **Typical range:** $5$ to $50$ (default in scikit-learn is $30$).
- **Low Perplexity (e.g. 2-5):** Over-focuses on microscopic structure; clusters break apart into tiny splintered clumps.
- **High Perplexity (e.g. 50-100):** Treats broader regions as neighborhoods; smooths out clusters, merging distinct sub-populations.

### 2. Learning Rate ($\eta$)
- Typically between $100$ and $1000$. Too low $\implies$ points stay trapped in initial clumps; too high $\implies$ points form an explosive ball with equidistant spacing.

### 3. Number of Iterations
- Usually at least $1,000$ iterations. Always inspect the cost/loss to ensure convergence.

### 4. Golden Rules for Reading t-SNE Plots
1. **Cluster sizes do NOT reflect true high-dimensional density**: t-SNE expands dense clusters and contracts sparse ones.
2. **Distances between clusters may be meaningless**: Because t-SNE focuses on local neighbors, the relative distance between Cluster A and Cluster B vs Cluster C cannot be interpreted as geometric truth.
3. **Random seed matters**: Because initialization is stochastic, run with multiple seeds or use `init='pca'` for deterministic repeatability.

---

## 6. PCA vs. t-SNE: Direct Comparison

| Feature | PCA | t-SNE |
| :--- | :--- | :--- |
| **Type** | Linear | Non-linear |
| **Primary Goal** | Maximize global variance | Preserve local neighborhoods & clusters |
| **Interpretability** | Eigenvectors represent feature loadings | Coordinates have no physical axis meaning |
| **Speed** | Extremely fast ($O(n \cdot d^2)$) | Slower ($O(n \log n)$ via Barnes-Hut) |
| **Transform Unseen Data** | Yes (`pca.transform(X_new)`) | **No** (Must re-run on full combined dataset) |
| **Best Used For** | Preprocessing, compression, denoising, ML features | **Exploratory 2D/3D visualization & clustering** |

---

## 7. Practical Code Example (scikit-learn)

```python
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# 1. Standardize features
X_scaled = StandardScaler().fit_transform(X)

# 2. Fit t-SNE
tsne = TSNE(
    n_components=2,
    perplexity=30.0,
    learning_rate='auto',
    n_iter=1000,
    init='pca',
    random_state=42
)
X_tsne = tsne.fit_transform(X_scaled)
```

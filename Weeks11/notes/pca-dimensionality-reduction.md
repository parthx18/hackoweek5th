# Week 11 — Dimensionality Reduction: PCA (Principal Component Analysis)

## 1. Introduction: The Curse of Dimensionality

In machine learning, datasets often contain dozens, hundreds, or thousands of features (dimensions):
- Gene expression profiles: 20,000+ genes
- Image recognition: 28x28 grayscale image = 784 pixels, 4K RGB = millions of features
- Customer analytics: 50+ demographic, transactional, and behavioral metrics

### Why is High Dimensionality a Problem?
1. **Empty Space Phenomenon**: As dimensions increase, the volume of feature space grows exponentially ($d \to \infty$). Data points become isolated, sparse, and distant from each other.
2. **Equidistant Vectors**: Euclidean distance ($L_2$ norm) loses discriminating power—the ratio of distance to the nearest neighbor versus farthest neighbor approaches 1.
3. **Overfitting & Computational Cost**: Models require exponentially more training data to generalize, training slows down, and memory footprints balloon.
4. **Impossibility of Direct Visualization**: Humans can only intuitively visualize 2D or 3D spaces.

**Dimensionality Reduction** aims to transform high-dimensional data $X \in \mathbb{R}^{n \times d}$ into a lower-dimensional representation $Z \in \mathbb{R}^{n \times k}$ ($k \ll d$) while preserving as much meaningful structure (variance, topology, or pairwise distance) as possible.

---

## 2. PCA: Core Intuition

**Principal Component Analysis (PCA)** is an **unsupervised, linear** dimensionality reduction technique developed by Karl Pearson (1901) and Harold Hotelling (1933).

### The Core Idea: Maximizing Variance
Information in data is represented by **variance** (spread). 
- If a feature has zero variance (constant value for every sample), it carries zero distinguishing information.
- If we project our data onto an axis, we want the projected points to spread out as much as possible so that distinct clusters and variations remain distinguishable.

```
       Feature 2
           ^           *  *  (PC1: Line of Maximum Variance)
           |         *  *  /
           |       *  *  /
           |     *  *  / 
           |   *  *  /
           | *  *  /
           +---------------------> Feature 1
                   \
                    \ (PC2: Orthogonal to PC1, captures remaining variance)
```

1. **First Principal Component ($PC_1$)**: The single axis along which the data varies the most.
2. **Second Principal Component ($PC_2$)**: The axis that is **strictly orthogonal** (perpendicular / uncorrelated) to $PC_1$ and captures the largest remaining variance.
3. **$k$-th Principal Component ($PC_k$)**: Orthogonal to all previous components, capturing remaining variance in descending order.

---

## 3. Mathematical Foundations

### Step 1: Standardization (Crucial Step!)
PCA is variance-driven. If one feature is measured in kilograms ($0-100$) and another in milligrams ($0-100,000,000$), the milligram feature would dominate purely due to scale:
$$ z_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j} $$
After standardization:
$$\text{Mean}(\mathbf{z}_j) = 0, \quad \text{Var}(\mathbf{z}_j) = 1$$

### Step 2: Covariance Matrix Calculation
For mean-centered data matrix $X \in \mathbb{R}^{n \times d}$:
$$ \Sigma = \frac{1}{n-1} X^T X $$
$\Sigma$ is a symmetric $d \times d$ matrix:
- Diagonal elements $\Sigma_{ii}$: Variance of feature $i$.
- Off-diagonal elements $\Sigma_{ij}$: Covariance between feature $i$ and feature $j$ (degree of linear co-dependence).

### Step 3: Eigendecomposition
We solve the characteristic eigenvalue equation:
$$ \Sigma \mathbf{v}_i = \lambda_i \mathbf{v}_i $$
Where:
- $\mathbf{v}_i$ is the **eigenvector** (unit direction vector representing the principal axis).
- $\lambda_i$ is the **eigenvalue** (scalar measuring the variance along direction $\mathbf{v}_i$).

Because $\Sigma$ is real and symmetric, its eigenvectors are mutually orthogonal:
$$ \mathbf{v}_i^T \mathbf{v}_j = 0 \quad (\forall i \neq j) $$

### Step 4: Sorting & Projection
Sort eigenvalues in descending order:
$$ \lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_d \ge 0 $$
Select top $k$ eigenvectors to form projection matrix $W \in \mathbb{R}^{d \times k}$:
$$ W = [\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_k] $$
Project the original standardized data $X$ to lower-dimensional coordinates $Z \in \mathbb{R}^{n \times k}$:
$$ Z = X W $$

---

## 4. Explained Variance Ratio & Scree Plot

### Explained Variance Ratio (EVR)
The proportion of total dataset variance retained by the $i$-th component:
$$ \text{EVR}_i = \frac{\lambda_i}{\sum_{j=1}^d \lambda_j} $$

### Cumulative Explained Variance
$$ \text{Cumulative EVR}_k = \sum_{i=1}^k \text{EVR}_i $$

### The "Elbow Rule" & Scree Plot
A **Scree Plot** graphs the eigenvalue / variance explained against the component index:
- Look for the point of diminishing returns (the "elbow").
- Alternatively, set an information threshold (e.g. choose minimum $k$ such that cumulative EVR $\ge 90\%$ or $95\%$).

---

## 5. Practical Implementation (scikit-learn & NumPy)

### Python (scikit-learn)
```python
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Explained Variance Ratio:", pca.explained_variance_ratio_)
print("Total Variance Retained:", np.sum(pca.explained_variance_ratio_))
```

### Pure NumPy Implementation (From Scratch)
```python
import numpy as np

# 1. Mean-center
X_meaned = X - np.mean(X, axis=0)

# 2. Covariance matrix
cov_mat = np.cov(X_meaned, rowvar=False)

# 3. Eigendecomposition
eigen_vals, eigen_vecs = np.linalg.eigh(cov_mat)

# 4. Sort in descending order
idx = np.argsort(eigen_vals)[::-1]
eigen_vals, eigen_vecs = eigen_vals[idx], eigen_vecs[:, idx]

# 5. Project onto top k components
k = 2
W = eigen_vecs[:, :k]
X_projected = np.dot(X_meaned, W)
```

---

## 6. Applications, Strengths & Limitations

| Aspect | Summary |
| :--- | :--- |
| **Primary Use Cases** | 2D/3D Data visualization, image compression/eigenfaces, denoising, speeding up training of downstream models (SVM, Logistic Regression, KNN). |
| **Strengths** | Fast computation ($O(d^3)$ or $O(d^2 n)$ via SVD), deterministic, completely removes collinearity/correlation between features, global variance preservation. |
| **Limitations** | **Linear only**: Cannot capture non-linear manifolds (e.g., Swiss Roll, concentric circles). Highly sensitive to scaling. Components are linear combinations of features, losing physical domain interpretability. |

---

## 7. Key Takeaways
1. **PCA does not discard original features**; it creates new orthogonal synthetic features ($PC_1, PC_2, \dots$) by linearly rotating and projecting the coordinate axes.
2. Always **Standardize** ($z = \frac{x-\mu}{\sigma}$) before PCA unless all variables are on the exact same physical scale and variance magnitude reflects true importance.
3. PCA is best used as a **preprocessing/compression step** and for **global visualization**, while non-linear methods like **t-SNE** (studied in Week 12) excel at local cluster preservation.

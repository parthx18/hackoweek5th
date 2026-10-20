# Week 14 — Bias-Variance Tradeoff, Overfitting/Underfitting & Regularization

## 1. The Bias-Variance Tradeoff

In supervised machine learning, the goal is to estimate an unknown true function $f(x)$ from training data:
$$ y = f(x) + \epsilon, \quad \mathbb{E}[\epsilon] = 0, \quad \text{Var}(\epsilon) = \sigma^2 $$
Where $\epsilon$ is irreducible random noise.

When we fit a model $\hat{f}(x)$, its **Expected Prediction Error (EPE)** at a point $x$ decomposes mathematically into three distinct components:
$$ \mathbb{E}\left[(y - \hat{f}(x))^2\right] = \underbrace{\left(\mathbb{E}[\hat{f}(x)] - f(x)\right)^2}_{\mathbf{\text{Bias}^2}} + \underbrace{\mathbb{E}\left[(\hat{f}(x) - \mathbb{E}[\hat{f}(x)])^2\right]}_{\mathbf{\text{Variance}}} + \underbrace{\sigma^2}_{\mathbf{\text{Irreducible Error}}} $$

```
   Error
     ^
     |         Total Error (Validation Error)
     |          \           /
     |           \  Sweet  /
     |  Bias²     \  Spot /      Variance
     |   \         \     /         /
     |    \         \___/         /
     |     \_____________________/
     |----------------------------------- Irreducible Error (σ²)
     +----------------------------------------> Model Complexity
        (Underfitting)             (Overfitting)
```

### 1.1 Bias (Underfitting)
- **Definition:** The difference between the expected prediction of our model and the true underlying relationship.
- **Cause:** Overly rigid or simplistic model assumptions (e.g., fitting a straight line to sinusoidal data).
- **Symptom:** **High Train Error AND High Validation Error.**
- **Remedies:**
  - Increase model capacity (higher polynomial degree, deeper trees, larger neural networks).
  - Add more relevant domain features.
  - Decrease regularization penalty ($\lambda$).

### 1.2 Variance (Overfitting)
- **Definition:** The variability of model predictions when trained on different random subsets of data.
- **Cause:** Model has too many degrees of freedom and memorizes the random noise/idiosyncrasies of the training sample instead of the true signal.
- **Symptom:** **Very Low Train Error BUT High Validation Error** (wide generalization gap).
- **Remedies:**
  - Collect more training data / data augmentation.
  - Simplify model (pruning decision trees, reducing polynomial degree).
  - Use ensemble methods (e.g., Bagging / Random Forests).
  - **Apply Regularization (L1 / L2 constraints on parameters).**

---

## 2. Regularization: Penalizing Complexity

Regularization modifies the cost function by adding a penalty term that discourages model parameters $w$ from growing excessively large:
$$ \mathcal{L}_{\text{reg}}(\mathbf{w}) = \mathcal{L}_{\text{loss}}(\mathbf{w}) + \lambda \, \Omega(\mathbf{w}) $$
Where $\lambda \ge 0$ is the regularization hyperparameter controlling the tradeoff between fitting training data and constraining model complexity.

---

## 3. L2 Regularization (Ridge Regression / Weight Decay)

### Formulation
Penalizes the squared Euclidean ($L_2$) norm of the weights:
$$ \mathcal{L}_{\text{Ridge}}(\mathbf{w}) = \frac{1}{2n} \sum_{i=1}^n (y_i - \mathbf{w}^T \mathbf{x}_i)^2 + \frac{\lambda}{2} \sum_{j=1}^p w_j^2 = \text{MSE} + \frac{\lambda}{2} \|\mathbf{w}\|_2^2 $$

### Closed-form Analytical Solution
$$ \mathbf{w}_{\text{Ridge}} = (X^T X + \lambda I)^{-1} X^T \mathbf{y} $$
- Adding $\lambda I$ guarantees the matrix is invertible even under high multicollinearity or when $p > n$.

### Geometric Intuition & Behavior
- The constraint boundary is a smooth hypersphere ($\sum w_j^2 \le C$).
- **Effect:** Weights are shrunk smoothly toward zero, but **rarely become exactly zero**.
- All features are retained, but their destabilizing sensitivity to noise is quelled.

---

## 4. L1 Regularization (Lasso: Least Absolute Shrinkage and Selection Operator)

### Formulation
Penalizes the absolute value ($L_1$) norm of the weights:
$$ \mathcal{L}_{\text{Lasso}}(\mathbf{w}) = \frac{1}{2n} \sum_{i=1}^n (y_i - \mathbf{w}^T \mathbf{x}_i)^2 + \lambda \sum_{j=1}^p |w_j| = \text{MSE} + \lambda \|\mathbf{w}\|_1 $$

### Geometric Intuition & Sparsity
- The constraint boundary is an $L_1$ diamond/polytope with sharp vertices along the coordinate axes ($\sum |w_j| \le C$).
- Because the elliptical loss contours typically intersect the constraint region at one of its sharp corners on an axis, many weight coefficients become **strictly zero ($w_j = 0$)**!
- **Effect:** Acts as an automatic, embedded **feature selection mechanism**. Highly suited when you suspect only a small fraction of features are truly predictive.

---

## 5. ElasticNet: The Best of Both Worlds

When features are highly correlated, Lasso tends to randomly pick one and discard the rest. ElasticNet combines both penalties:
$$ \mathcal{L}_{\text{Elastic}}(\mathbf{w}) = \text{MSE} + \lambda \left( \alpha \|\mathbf{w}\|_1 + \frac{1 - \alpha}{2} \|\mathbf{w}\|_2^2 \right) $$
- Inherits Lasso's sparsity while maintaining Ridge's stability and grouped feature selection.

---

## 6. Comprehensive Summary Table

| Metric | Underfitting (High Bias) | Good Fit (Balanced) | Overfitting (High Variance) |
| :--- | :--- | :--- | :--- |
| **Train Error** | High | Low | Extremely Low (~0) |
| **Validation Error** | High | Low | High |
| **Generalization Gap** | Small | Small | **Very Large** |
| **Decision Surface** | Too flat / rigid | Smoothly captures trend | Highly erratic / oscillating |
| **Remedy** | Increase capacity, drop $\lambda$ | Maintain balance | Collect data, apply L1/L2, ensemble |

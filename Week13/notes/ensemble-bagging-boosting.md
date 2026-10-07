# Week 13 — Ensemble Methods: Bagging & Boosting (XGBoost, LightGBM)

## 1. The Ensemble Paradigm

Why rely on a single model when you can aggregate an ensemble of models?
The central mathematical principle behind ensemble learning is **variance and bias reduction** through model combination:
- If we combine $M$ independent models each with error variance $\sigma^2$, the averaged prediction has variance $\frac{\sigma^2}{M}$.
- Even when models are correlated with correlation $\rho$, ensemble variance is:
  $$ \text{Var}_{\text{ensemble}} = \rho \sigma^2 + \frac{1 - \rho}{M} \sigma^2 $$
  As $M \to \infty$, the second term drops to zero, and total variance approaches $\rho \sigma^2$. Decorrelating individual base learners is therefore essential!

Ensemble methods branch into two primary paradigms:
1. **Bagging (Parallel)**: Trains independent, high-variance models on random subsets of data and averages their votes to **crush variance**.
2. **Boosting (Sequential)**: Trains models iteratively, each correcting the residual errors of its predecessor to **reduce bias**.

---

## 2. Bagging (Bootstrap Aggregating)

### 2.1 Bootstrapping
Given training set $D$ of size $N$, generate $M$ bootstrap sets $D_1, D_2, \dots, D_M$ by **sampling with replacement** $N$ times.
- Probability an individual sample is never picked: $(1 - \frac{1}{N})^N \approx \frac{1}{e} \approx 36.8\%$.
- These $36.8\%$ untouched samples form the **Out-Of-Bag (OOB)** validation set, offering a free validation estimate without cross-validation!

### 2.2 Random Forest (Bagging + Feature Subspace Sampling)
Standard bagging with decision trees suffers because dominant features are chosen at the root of nearly every tree, causing high tree correlation $\rho$.
**Leo Breiman's Random Forest** solves this by:
1. Sampling samples with replacement (Bootstrapping).
2. At every node split, considering only a random subset of $m \approx \sqrt{d}$ features.
3. Growing deep, unpruned trees (low bias, high variance individual learners).
4. Averaging output:
   $$ \hat{y} = \frac{1}{M} \sum_{m=1}^M f_m(x) $$

---

## 3. Boosting Foundations: Sequential Error Correction

Instead of independent parallel training, Boosting builds additive models sequentially:
$$ F_m(x) = F_{m-1}(x) + \eta f_m(x) $$
Where $\eta \in (0, 1]$ is the **learning rate (shrinkage)**.

### 3.1 Gradient Boosting (GBM)
Jerome Friedman generalized boosting to any differentiable loss function $L(y, \hat{y})$ via gradient descent in function space:
1. Compute the pseudo-residuals (negative gradient of the loss):
   $$ r_{im} = -\left[ \frac{\partial L(y_i, F(x_i))}{\partial F(x_i)} \right]_{F(x) = F_{m-1}(x)} $$
   *(For Mean Squared Error $L = \frac{1}{2}(y - \hat{y})^2$, $r_{im} = y_i - F_{m-1}(x_i)$, which is the exact residual error!)*
2. Fit a shallow decision tree (weak learner) to predict pseudo-residuals $r_{im}$.
3. Update model with shrinkage: $F_m(x) = F_{m-1}(x) + \eta f_m(x)$.

---

## 4. Modern SOTA Boosters: XGBoost vs. LightGBM

### 4.1 XGBoost (Extreme Gradient Boosting — Chen & Guestrin, 2016)
XGBoost revolutionized competitive machine learning by redesigning GBM for speed, scale, and regularization:

1. **Second-Order Taylor Expansion**:
   Approximates objective using both first-order gradients $g_i$ and second-order Hessians $h_i$:
   $$ \mathcal{L}^{(t)} \approx \sum_{i=1}^n \left[ g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) \right] + \Omega(f_t) $$
   Where $g_i = \partial_{\hat{y}^{(t-1)}} L(y_i, \hat{y}^{(t-1)})$ and $h_i = \partial^2_{\hat{y}^{(t-1)}} L(y_i, \hat{y}^{(t-1)})$.
2. **Explicit Regularization on Tree Complexity**:
   $$ \Omega(f) = \gamma T + \frac{1}{2} \lambda \sum_{j=1}^T w_j^2 + \alpha \sum_{j=1}^T |w_j| $$
   Penalizes number of leaves $T$, L2 norm of leaf weights $\lambda$, and L1 norm $\alpha$.
3. **Exact Optimal Leaf Weight**:
   $$ w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda} $$
4. **Sparsity Awareness**: Automatically handles missing values by learning optimal default branch direction.

### 4.2 LightGBM (Microsoft Research, 2017)
Designed to handle millions of samples and high-dimensional tabular datasets where XGBoost previously choked:

1. **Histogram-based Algorithm**:
   Continuous floating-point features are binned into discrete integer bins (e.g. 256 bins). Reduces memory usage by up to $80\%$ and speeds up split finding by up to $8\times$.
2. **GOSS (Gradient-based One-Side Sampling)**:
   Instances with small gradients are already well-trained. GOSS retains all instances with large gradients and randomly samples from instances with small gradients, computing accurate split gain while training on a fraction of data.
3. **EFB (Exclusive Feature Bundling)**:
   Bundles mutually exclusive sparse features (features that rarely take non-zero values simultaneously, e.g. one-hot encoded categories) into a single dense feature.
4. **Leaf-wise (Best-First) Tree Growth**:
   Instead of growing level-by-level (symmetric depth), LightGBM finds the single leaf with the largest loss reduction and splits it, achieving lower loss with fewer nodes.

---

## 5. Architectural Comparison Matrix

| Property | Random Forest | Gradient Boosting (GBM) | XGBoost | LightGBM |
| :--- | :--- | :--- | :--- | :--- |
| **Ensemble Paradigm** | Bagging (Parallel) | Boosting (Sequential) | Boosting (Sequential) | Boosting (Sequential) |
| **Primary Error Reduced** | Variance | Bias | Bias & Variance | Bias & Variance |
| **Optimization Order** | N/A (Averaging) | 1st Order (Gradient) | 2nd Order (Gradient + Hessian) | 2nd Order + Histograms |
| **Tree Growth Policy** | Level-wise | Level-wise | Level-wise (Depth-wise) | **Leaf-wise (Best-first)** |
| **Categorical Support** | Requires One-Hot | Requires One-Hot | One-Hot / Experimental | **Native Fisher optimal splits** |
| **Missing Values** | Imputation needed | Imputation needed | Native auto-direction | Native auto-direction |
| **Speed & Scalability** | High (Parallel) | Slow (Sequential) | Fast (C++ multi-threaded) | **Ultra-Fast (GOSS + Histograms)** |

---

## 6. Practical Tuning Guide for Boosters

1. **`learning_rate` ($\eta$) vs. `n_estimators`**:
   - Lower learning rate ($\eta \in [0.01, 0.05]$) paired with higher `n_estimators` with **Early Stopping** almost always yields superior generalization.
2. **Tree Depth & Leaf Count**:
   - For XGBoost: `max_depth` typically between $3$ and $8$.
   - For LightGBM: `num_leaves` typically between $15$ and $63$ (keep `num_leaves` $< 2^{\text{max\_depth}}$ to avoid overfitting).
3. **Subsampling & Regularization**:
   - `subsample` / `colsample_bytree` $\in [0.6, 0.85]$ adds random forest-like decorrelation.
   - Increase `reg_alpha` (L1) and `reg_lambda` (L2) when validation gap widens.

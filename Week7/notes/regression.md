# Regression Analysis — Notes

Regression is a category of **supervised machine learning** where the objective is to predict a continuous numerical value (target $y$) given one or more input features ($X$).

---

## 1. Linear Regression

### 1.1 The Model Equation
In Simple Linear Regression (single feature $x$):
$$ y = w_1 x + w_0 $$
where:
*   $w_1$ is the slope (weight / coefficient)
*   $w_0$ is the y-intercept (bias)

In Multiple Linear Regression (with $n$ features $x_1, x_2, \dots, x_n$):
$$ \hat{y} = w_0 + w_1 x_1 + w_2 x_2 + \dots + w_n x_n = \mathbf{w}^T \mathbf{x} $$
where $\mathbf{x} = [1, x_1, x_2, \dots, x_n]^T$ and $\mathbf{w} = [w_0, w_1, w_2, \dots, w_n]^T$.

### 1.2 The Cost Function: Mean Squared Error (MSE)
To measure how well the line fits the data points, we calculate the average squared difference between true values $y_i$ and predicted values $\hat{y}_i$:

$$ J(\mathbf{w}) = \frac{1}{m} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2 = \frac{1}{m} \sum_{i=1}^{m} (y_i - \mathbf{w}^T \mathbf{x}_i)^2 $$

where $m$ is the number of training samples.

### 1.3 Solving for Optimal Weights
There are two primary ways to find the weights $\mathbf{w}$ that minimize $J(\mathbf{w})$:

#### Method A: Ordinary Least Squares (OLS) Normal Equation
Using calculus and linear algebra, setting $\nabla_{\mathbf{w}} J(\mathbf{w}) = 0$ yields the exact analytical closed-form solution:
$$ \mathbf{w} = (X^T X)^{-1} X^T \mathbf{y} $$
*   **Pros:** Exact solution in one step, no hyperparameters to tune (no learning rate $\alpha$).
*   **Cons:** Inverting an $n \times n$ matrix takes $O(n^3)$ compute time. Slow when features $n > 10,000$.

#### Method B: Gradient Descent
Iteratively step downhill along the negative gradient of the loss surface:
$$ \mathbf{w} \leftarrow \mathbf{w} - \alpha \frac{\partial J}{\partial \mathbf{w}} $$
where the gradient vector is:
$$ \frac{\partial J}{\partial \mathbf{w}} = -\frac{2}{m} X^T (\mathbf{y} - X \mathbf{w}) $$
*   **Pros:** Efficient on large datasets and high-dimensional spaces ($O(k \cdot m \cdot n)$).
*   **Cons:** Requires choosing a suitable learning rate $\alpha$ and feature scaling.

### 1.4 Core Evaluation Metrics
1.  **Mean Absolute Error (MAE):**
    $$ \text{MAE} = \frac{1}{m} \sum_{i=1}^{m} |y_i - \hat{y}_i| $$
    *Direct interpretation in original target units. Robust to outliers.*
2.  **Mean Squared Error (MSE):**
    $$ \text{MSE} = \frac{1}{m} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2 $$
    *Penalizes large errors heavily.*
3.  **Root Mean Squared Error (RMSE):**
    $$ \text{RMSE} = \sqrt{\text{MSE}} $$
    *Expressed in the same units as $y$ while penalizing extreme misses.*
4.  **Coefficient of Determination ($R^2$ Score):**
    $$ R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2} = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}} $$
    *   $R^2 = 1.0$: Perfect predictions.
    *   $R^2 = 0.0$: Performs no better than predicting the mean $\bar{y}$.
    *   $R^2 < 0.0$: Worse than predicting the mean baseline.

---

## 2. Polynomial Regression & The Bias-Variance Tradeoff

Real-world phenomena are frequently non-linear (e.g. population growth, drag resistance, battery discharge curves).

### 2.1 The Polynomial Transformation
We can still use linear regression machinery on non-linear problems by engineering higher-order polynomial powers as new synthetic features:
$$ \mathbf{x} = [x] \longrightarrow [x, x^2, x^3, \dots, x^d] $$
The model equation becomes:
$$ \hat{y} = w_0 + w_1 x + w_2 x^2 + w_3 x^3 + \dots + w_d x^d $$
*Crucial insight:* Even though the relationship with $x$ is non-linear, the equation remains **linear with respect to the parameters $w$**, so Ordinary Least Squares still applies directly.

### 2.2 Underfitting vs. Overfitting
*   **Underfitting (High Bias):**
    *   Degree $d = 1$ on curved data.
    *   The model is too simplistic to capture the underlying pattern.
    *   High training error, high test error.
*   **Optimal Fit (Balanced):**
    *   Degree $d = 2$ or $3$.
    *   Captures the true underlying curve without memorizing sample noise.
    *   Low training error, low test error.
*   **Overfitting (High Variance):**
    *   Degree $d \ge 10$.
    *   The model has too much flexibility and oscillates wildly to pass through every individual noise point.
    *   Near-zero training error, catastrophic test error on unseen points.

---

## 3. Regularization: Ridge ($L_2$) and Lasso ($L_1$)

When features are correlated (multicollinearity) or polynomial degrees are high, unconstrained weights can explode to massive positive and negative values (e.g., $w = 10^7$). Regularization adds a penalty to the loss function to constrain weight magnitude.

$$ \text{Total Loss} = \text{Data Fit Loss (MSE)} + \lambda \cdot \text{Complexity Penalty} $$

where $\lambda$ (or $\alpha$ in scikit-learn) is the **regularization hyperparameter**:
*   $\lambda = 0$: Standard Ordinary Least Squares.
*   $\lambda \to \infty$: All coefficients shrink toward zero (flat line).

### 3.1 Ridge Regression ($L_2$ Regularization)
Ridge adds the sum of squared weights:
$$ J_{\text{Ridge}}(\mathbf{w}) = \frac{1}{m} \sum_{i=1}^{m} (y_i - \mathbf{w}^T \mathbf{x}_i)^2 + \lambda \sum_{j=1}^{n} w_j^2 $$
*(Notice: The bias term $w_0$ is generally not penalized).*

*   **Closed-form solution:**
    $$ \mathbf{w}_{\text{Ridge}} = (X^T X + \lambda I)^{-1} X^T \mathbf{y} $$
    *(Adding $\lambda I$ guarantees $(X^T X + \lambda I)$ is invertible, even when $X^T X$ is singular!)*
*   **Behavior:** Shrinks all weights smoothly toward zero, but **rarely sets them strictly to zero**.

### 3.2 Lasso Regression ($L_1$ Regularization)
Lasso (*Least Absolute Shrinkage and Selection Operator*) adds the sum of absolute values:
$$ J_{\text{Lasso}}(\mathbf{w}) = \frac{1}{m} \sum_{i=1}^{m} (y_i - \mathbf{w}^T \mathbf{x}_i)^2 + \lambda \sum_{j=1}^{n} |w_j| $$

*   **Behavior:** Because the $L_1$ norm has sharp corners on coordinate axes, it drives unimportant coefficients **strictly to 0.0**.
*   **Feature Selection:** Lasso acts as an automatic feature selection mechanism, generating sparse and highly interpretable models.

### 3.3 Geometric Intuition ($L_1$ vs $L_2$)

| Feature | Ridge ($L_2$) | Lasso ($L_1$) |
| :--- | :--- | :--- |
| **Penalty Term** | $\lambda \sum w_j^2$ | $\lambda \sum \|w_j\|$ |
| **Constraint Shape** | Circular / Hypersphere | Diamond / Hyper-rhombus (sharp axes) |
| **Weight Sparsity** | Non-sparse (weights $\approx 0$) | Sparse (weights $= 0$) |
| **Feature Selection** | Retains all features | Discards useless features |
| **Computational Method** | Analytical closed-form exists | Coordinate Descent optimization |
| **Best Used When** | Many small, correlated predictors | Few dominant predictors among noise |

---

## 4. Practical Workflow & Best Practices
1.  **Always standardize features** (`StandardScaler`) before applying Ridge or Lasso, so features on large scales do not receive unfair penalty weights.
2.  **Cross-Validation**: Use K-Fold cross validation (`RidgeCV`, `LassoCV`) to determine the optimal $\lambda / \alpha$.
3.  **Check Residuals**: A healthy model has residuals $y - \hat{y}$ randomly distributed around 0 with no visible parabolic or funnel patterns.

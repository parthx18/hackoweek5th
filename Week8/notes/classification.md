# Classification Analysis — Notes

Classification is a branch of **supervised learning** where the target output is a categorical label rather than a continuous number.
*   **Binary Classification:** Predicting between two classes, typically labeled $y \in \{0, 1\}$ (e.g., Spam vs Not Spam, Benign vs Malignant tumor).
*   **Multi-class Classification:** Predicting one of $K > 2$ classes (e.g., Digit recognition 0–9, Animal species).

---

## 1. Logistic Regression

Despite the word "Regression" in its name, Logistic Regression is a **classification algorithm**. It models the probability that a given input $\mathbf{x}$ belongs to class $1$.

### 1.1 The Sigmoid (Logistic) Function
Standard linear regression outputs values anywhere from $-\infty$ to $+\infty$, which cannot represent valid probabilities. We squash the linear combination $z = \mathbf{w}^T \mathbf{x} + b$ through the **Sigmoid function** $\sigma(z)$:

$$ \sigma(z) = \frac{1}{1 + e^{-z}} $$

Key characteristics:
*   As $z \to +\infty$, $\sigma(z) \to 1.0$.
*   As $z \to -\infty$, $\sigma(z) \to 0.0$.
*   When $z = 0$, $\sigma(z) = 0.5$.

The model's probability estimate is:
$$ P(y = 1 \mid \mathbf{x}) = \hat{p} = \sigma(\mathbf{w}^T \mathbf{x} + b) $$

### 1.2 The Log-Odds (Logit) Formulation
Rearranging the probability formula gives:
$$ \ln\left(\frac{p}{1 - p}\right) = \mathbf{w}^T \mathbf{x} + b $$
The term $\frac{p}{1 - p}$ is called the **odds ratio**, and its natural logarithm is the **logit**.
*Intuition:* Logistic regression is simply linear regression on the log-odds of the positive class!

### 1.3 Why MSE Fails & The Binary Cross-Entropy Loss
If we use Mean Squared Error with the sigmoid function, the resulting loss surface is non-convex with many local minima, causing gradient descent to get stuck.

Instead, we use **Binary Cross-Entropy (Log-Loss)**:
$$ J(\mathbf{w}, b) = -\frac{1}{m} \sum_{i=1}^{m} \Big[ y_i \ln(\hat{p}_i) + (1 - y_i) \ln(1 - \hat{p}_i) \Big] $$

*   If true $y_i = 1$: Loss is $-\ln(\hat{p}_i)$. If the model predicts $\hat{p} \to 0$, loss approaches $+\infty$!
*   If true $y_i = 0$: Loss is $-\ln(1 - \hat{p}_i)$. If the model predicts $\hat{p} \to 1$, loss approaches $+\infty$!
This creates a strictly convex bowl where gradient descent is guaranteed to find the global optimum.

### 1.4 Decision Boundaries & Thresholds
To convert probability $\hat{p}$ into a class prediction $\hat{y} \in \{0, 1\}$, we apply a decision threshold $\tau$ (default $\tau = 0.5$):
$$ \hat{y} = \begin{cases} 1 & \text{if } \hat{p} \ge \tau \\ 0 & \text{if } \hat{p} < \tau \end{cases} $$
For standard logistic regression with linear features, the decision boundary $\mathbf{w}^T \mathbf{x} + b = 0$ is a **straight line (in 2D)** or a **hyperplane (in higher dimensions)**.

---

## 2. K-Nearest Neighbors (KNN)

KNN is an intuitive, **non-parametric, instance-based (lazy learning)** algorithm.

### 2.1 How KNN Works
KNN does not "train" a parameterized model. It stores the training data directly in memory:
1.  Receive a new query point $\mathbf{x}_{\text{query}}$.
2.  Compute the distance between $\mathbf{x}_{\text{query}}$ and every single training point.
3.  Select the $k$ nearest data points.
4.  **Classification:** Take the majority vote (mode) among the $k$ neighbors.

### 2.2 Distance Metrics
For points $\mathbf{u} = (u_1, \dots, u_d)$ and $\mathbf{v} = (v_1, \dots, v_d)$:
*   **Euclidean Distance ($L_2$ norm):**
    $$ d(\mathbf{u}, \mathbf{v}) = \sqrt{\sum_{i=1}^{d} (u_i - v_i)^2} $$
    *Straight-line distance. Most common.*
*   **Manhattan Distance ($L_1$ norm):**
    $$ d(\mathbf{u}, \mathbf{v}) = \sum_{i=1}^{d} |u_i - v_i| $$
    *Grid / taxicab distance. Better for high-dimensional sparse data.*

### 2.3 Choosing the Hyperparameter $k$
*   **$k = 1$ (High Variance / Overfitting):**
    *   Decision boundaries are jagged and wrap tightly around every single training point.
    *   Extremely sensitive to mislabeled data or noise.
*   **Optimal $k$ (e.g., $k = 5$ to $15$):**
    *   Smooths out noise while capturing true neighborhood clusters.
    *   Usually chosen as an **odd number** (in binary classification) to eliminate ties.
*   **Very Large $k \to m$ (High Bias / Underfitting):**
    *   The boundary becomes overly smoothed and predicts whichever class has the majority across the entire dataset.

### 2.4 Crucial Requirement: Feature Scaling
Because KNN computes geometric distances, features on larger numeric scales (e.g. Salary: \$50,000) completely drown out features on smaller scales (e.g. Age: 25). **Always standardize or normalize features before running KNN.**

---

## 3. Comprehensive Classification Metrics

Evaluating a classifier solely by overall accuracy is often misleading, especially with imbalanced datasets (e.g., detecting rare fraudulent transactions where 99% of data is legitimate).

### 3.1 The Confusion Matrix

| | **Predicted Negative ($\hat{y} = 0$)** | **Predicted Positive ($\hat{y} = 1$)** |
| :--- | :--- | :--- |
| **Actual Negative ($y = 0$)** | **True Negative (TN)** | **False Positive (FP)** (Type I Error) |
| **Actual Positive ($y = 1$)** | **False Negative (FN)** (Type II Error) | **True Positive (TP)** |

### 3.2 Core Metrics Formulas

1.  **Accuracy:**
    $$ \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} $$
    *Overall percentage of correct predictions.*

2.  **Precision:**
    $$ \text{Precision} = \frac{TP}{TP + FP} $$
    *Out of all instances predicted as Positive, how many were actually Positive? Critical when False Positives are expensive (e.g. Spam filter sending important emails to spam).*

3.  **Recall (Sensitivity / True Positive Rate):**
    $$ \text{Recall} = \frac{TP}{TP + FN} $$
    *Out of all actual Positives, how many did the model correctly catch? Critical when False Negatives are dangerous (e.g. Cancer diagnosis, fraud detection).*

4.  **$F_1$-Score:**
    $$ F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} $$
    *Harmonic mean of precision and recall. Punishes extreme imbalance between the two.*

5.  **ROC Curve & AUC (Area Under Curve):**
    *   Plots the True Positive Rate ($\text{Recall}$) vs False Positive Rate ($\frac{FP}{FP+TN}$) across all possible classification thresholds from $0.0$ to $1.0$.
    *   **$\text{AUC} = 1.0$**: Perfect classifier.
    *   **$\text{AUC} = 0.5$**: Random guessing.

---

## 4. Model Comparison Cheat Sheet

| Criterion | Logistic Regression | K-Nearest Neighbors (KNN) |
| :--- | :--- | :--- |
| **Model Type** | Parametric (learns weights $\mathbf{w}, b$) | Non-parametric (stores samples) |
| **Decision Boundary** | Linear (or polynomial if transformed) | Complex, flexible non-linear shapes |
| **Training Speed** | Fast (gradient descent or coordinate descent) | Instant ($O(1)$ lazy learning) |
| **Inference Speed** | Instant ($O(n)$ vector dot product) | Slow on large datasets ($O(m \cdot n)$ distance checks) |
| **Interpretability** | Very high (coefficients explain feature impact) | Moderate (explains via nearest neighbors) |
| **Data Requirements** | Handles outliers reasonably well with regularization | Sensitive to noise & curse of dimensionality |

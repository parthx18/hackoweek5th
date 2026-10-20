# Model Evaluation — Notes

How do we know if our machine learning model is actually good? Model evaluation tells us whether a model will generalize to new, unseen data or if it simply memorized the training set.

---

## 1. Train / Test Split

Never evaluate a model on the same data it was trained on! That is like giving a student the exact exam questions beforehand.

*   **Training Set (e.g. 70% – 80%):** Used by the algorithm to learn weights.
*   **Test Set (e.g. 20% – 30%):** Kept strictly hidden until final testing.
*   **Data Leakage Warning:** Never scale or transform features using the whole dataset before splitting. Always fit scalers *only* on the training set.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

## 2. K-Fold Cross-Validation

A single train/test split can be lucky or unlucky depending on which samples end up in the test set.

**How K-Fold Works (e.g. $K = 5$):**
1.  Split the data into 5 equal subsets (folds).
2.  Train on folds 1, 2, 3, 4 $\rightarrow$ Test on fold 5.
3.  Train on folds 1, 2, 3, 5 $\rightarrow$ Test on fold 4.
4.  Repeat 5 times so every fold acts as the test set once.
5.  Take the average of the 5 scores.

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5)
print("Mean Accuracy:", scores.mean())
```

---

## 3. The Confusion Matrix

In binary classification ($0 = \text{Negative}, 1 = \text{Positive}$), predictions fall into 4 buckets:

| | Predicted Negative ($0$) | Predicted Positive ($1$) |
|:---|:---:|:---:|
| **Actual Negative ($0$)** | **TN** (True Negative) | **FP** (False Positive - Type I) |
| **Actual Positive ($1$)** | **FN** (False Negative - Type II) | **TP** (True Positive) |

*   **TN:** Correctly identified negatives (e.g. normal email marked normal).
*   **TP:** Correctly identified positives (e.g. spam email marked spam).
*   **FP:** Negative misclassified as positive (e.g. normal email sent to spam).
*   **FN:** Positive missed as negative (e.g. spam email slipped into inbox).

---

## 4. Precision, Recall, and F1-Score

### 4.1 Accuracy
$$ \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} $$
*Misleading when classes are imbalanced (e.g., in a dataset where 99% of people are healthy, predicting everyone is healthy yields 99% accuracy while catching 0 sick patients!).*

### 4.2 Precision
$$ \text{Precision} = \frac{TP}{TP + FP} $$
*   **Question:** Of all cases we predicted positive, how many were actually positive?
*   **Goal:** Minimize False Positives.
*   **Example:** YouTube spam comments filter. Better to let a spam comment through than delete a real user comment.

### 4.3 Recall (Sensitivity)
$$ \text{Recall} = \frac{TP}{TP + FN} $$
*   **Question:** Of all real positive cases, how many did we catch?
*   **Goal:** Minimize False Negatives.
*   **Example:** Cancer detection. It is dangerous to miss a real tumor.

### 4.4 F1-Score
$$ F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} $$
*The harmonic mean of Precision and Recall. Gives a single balanced score.*

---

## 5. ROC Curve & AUC Score

Most classifiers output a probability between 0 and 1. We choose a threshold (default 0.5) to decide the final class.

*   **ROC Curve (Receiver Operating Characteristic):** Plots True Positive Rate (Recall) vs False Positive Rate ($\frac{FP}{FP+TN}$) across all possible thresholds from $0.0$ to $1.0$.
*   **AUC (Area Under Curve):** A single number summarizing the ROC curve.
    *   **$\text{AUC} = 1.0$**: Perfect model.
    *   **$\text{AUC} = 0.85$**: Good model (85% chance it ranks a random positive higher than a random negative).
    *   **$\text{AUC} = 0.50$**: No better than flipping a coin.

```python
from sklearn.metrics import roc_curve, roc_auc_score

auc = roc_auc_score(y_test, y_probs)
```

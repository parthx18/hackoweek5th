# Week 15 Notes — Model Deployment Basics & Neural Network Foundations

## 1. Model Deployment Basics

### Why Save Models?
Training a model takes time and compute. Once trained, you want to reuse it without retraining — that's serialization.

### pickle
Python's standard serialization library. Works with almost any Python object.

```python
import pickle

# Save
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Load
with open("model.pkl", "rb") as f:
    loaded_model = pickle.load(f)
```

### joblib
Better than pickle for objects containing large NumPy arrays (e.g., sklearn models). Faster I/O.

```python
import joblib

# Save
joblib.dump(model, "model.joblib")

# Load
loaded_model = joblib.load("model.joblib")
```

### Saving Pipelines
Always save the full pipeline (scaler + model) together so preprocessing is consistent at inference time.

```python
from sklearn.pipeline import Pipeline
pipeline = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
joblib.dump(pipeline, "pipeline.joblib")
```

---

## 2. Perceptrons

A **perceptron** is the simplest artificial neuron:
- Receives inputs x1, x2, ..., xn
- Computes weighted sum: z = w1*x1 + w2*x2 + ... + wn*xn + b
- Passes z through an activation function: y_hat = activation(z)

### Single Perceptron (Python)
```python
import numpy as np

def perceptron(x, w, b, activation="sigmoid"):
    z = np.dot(w, x) + b
    if activation == "sigmoid":
        return 1 / (1 + np.exp(-z))
    elif activation == "relu":
        return np.maximum(0, z)
    elif activation == "tanh":
        return np.tanh(z)
```

---

## 3. Activation Functions

| Name | Formula | Use Case |
|---|---|---|
| Sigmoid | 1/(1+exp(-z)) | Binary output, output layer |
| Tanh | (exp(z)-exp(-z))/(exp(z)+exp(-z)) | Hidden layers (zero-centered) |
| ReLU | max(0,z) | Hidden layers (most common) |
| Leaky ReLU | max(0.01z, z) | Avoids dying ReLU |
| Softmax | exp(zi)/sum(exp(zj)) | Multi-class output layer |

**Why non-linear activations?** Without them, stacking layers collapses to a single linear transformation — no expressive power.

---

## 4. Forward Propagation

Given L layers with weight matrices W[l] and bias vectors b[l]:

```
For l = 1 to L:
    Z[l] = W[l] @ A[l-1] + b[l]   # linear step
    A[l] = activation(Z[l])         # activation step

Output: A[L] = y_hat
```

---

## 5. Backward Propagation (Backprop)

Backprop applies the **chain rule** to compute gradients of the loss w.r.t. every weight.

```
dZ[L] = dL/dA[L] * activation_prime(Z[L])
dW[L] = (1/m) * dZ[L] @ A[L-1].T
db[L] = (1/m) * sum(dZ[L])
dA[L-1] = W[L].T @ dZ[L]
```

Repeat backwards for l = L-1, ..., 1.

### Sigmoid gradient
sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z))

### ReLU gradient
relu'(z) = 1 if z > 0 else 0

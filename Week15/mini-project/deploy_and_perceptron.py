"""
Week 15 Mini-Project: Model Deployment Basics & Perceptron Demo
================================================================
Tasks:
  1. Train a simple sklearn classifier
  2. Save/load it using pickle and joblib
  3. Implement a perceptron with sigmoid activation from scratch
  4. Demonstrate forward pass and manual gradient (backprop step)

Instructions:
  - Learn how to persist trained models so they can be reused without retraining.
  - Understand the perceptron as the building block of neural networks.
  - Practice computing the forward pass and a single gradient step manually.
"""

import pickle
import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


# ─────────────────────────────────────────────
# PART 1: Model Deployment with pickle & joblib
# ─────────────────────────────────────────────

print("=" * 55)
print("PART 1 — Model Serialization (pickle & joblib)")
print("=" * 55)

# Load dataset and split
iris = load_iris()
X, y = iris.data, (iris.target == 0).astype(int)  # binary: setosa vs rest
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train a pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=200))
])
pipeline.fit(X_train, y_train)
original_preds = pipeline.predict(X_test)
print(f"Original accuracy: {accuracy_score(y_test, original_preds):.4f}")

# --- Save with pickle ---
with open("model_pickle.pkl", "wb") as f:
    pickle.dump(pipeline, f)
print("Model saved with pickle -> model_pickle.pkl")

# --- Load with pickle ---
with open("model_pickle.pkl", "rb") as f:
    pkl_model = pickle.load(f)
pkl_preds = pkl_model.predict(X_test)
print(f"Pickle loaded model accuracy: {accuracy_score(y_test, pkl_preds):.4f}")
print(f"Predictions match: {np.array_equal(original_preds, pkl_preds)}")

# --- Save with joblib ---
joblib.dump(pipeline, "model_joblib.joblib")
print("\nModel saved with joblib -> model_joblib.joblib")

# --- Load with joblib ---
jbl_model = joblib.load("model_joblib.joblib")
jbl_preds = jbl_model.predict(X_test)
print(f"Joblib loaded model accuracy: {accuracy_score(y_test, jbl_preds):.4f}")
print(f"Predictions match: {np.array_equal(original_preds, jbl_preds)}")


# ─────────────────────────────────────────
# PART 2: Perceptron & Activation Functions
# ─────────────────────────────────────────

print("\n" + "=" * 55)
print("PART 2 — Perceptron & Activation Functions")
print("=" * 55)


def sigmoid(z):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    """Derivative of sigmoid (for backprop)."""
    s = sigmoid(z)
    return s * (1 - s)


def relu(z):
    """ReLU activation function."""
    return np.maximum(0, z)


def relu_derivative(z):
    """Derivative of ReLU."""
    return (z > 0).astype(float)


def tanh_derivative(z):
    """Derivative of tanh."""
    return 1 - np.tanh(z) ** 2


# Demo: activation values
z_vals = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
print("\nActivation function outputs for z =", z_vals)
print(f"  Sigmoid : {np.round(sigmoid(z_vals), 4)}")
print(f"  Tanh    : {np.round(np.tanh(z_vals), 4)}")
print(f"  ReLU    : {np.round(relu(z_vals), 4)}")


# ─────────────────────────────────────────────────
# PART 3: Single Perceptron — Forward Pass + 1 Step
# ─────────────────────────────────────────────────

print("\n" + "=" * 55)
print("PART 3 — Single Perceptron Forward & Backward Pass")
print("=" * 55)

# Simple AND gate dataset
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y_and = np.array([0, 0, 0, 1], dtype=float)

# Initialize weights and bias
np.random.seed(0)
w = np.random.randn(2) * 0.1
b = 0.0
lr = 0.5

print("\nTraining perceptron on AND gate (50 epochs)...")
for epoch in range(50):
    total_loss = 0
    for xi, yi in zip(X_and, y_and):
        # Forward pass
        z = np.dot(w, xi) + b
        y_hat = sigmoid(z)

        # Loss (binary cross-entropy for one sample)
        loss = -(yi * np.log(y_hat + 1e-9) + (1 - yi) * np.log(1 - y_hat + 1e-9))
        total_loss += loss

        # Backward pass (gradient of loss w.r.t. z = y_hat - y for BCE + sigmoid)
        dz = y_hat - yi
        dw = dz * xi
        db = dz

        # Weight update
        w -= lr * dw
        b -= lr * db

    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1:3d} | Loss: {total_loss:.4f}")

# Final predictions
print("\nFinal perceptron predictions (AND gate):")
for xi, yi in zip(X_and, y_and):
    z = np.dot(w, xi) + b
    pred = sigmoid(z)
    print(f"  Input: {xi.astype(int)} | Target: {int(yi)} | Output: {pred:.4f}")

print("\nLearned weights:", np.round(w, 4), "| Bias:", round(b, 4))
print("\nDone! Check model_pickle.pkl and model_joblib.joblib in this directory.")

"""
Week 16 Mini-Project: Neural Network from Scratch (NumPy only)
==============================================================
Tasks:
  1. Implement loss functions (MSE, Binary Cross-Entropy)
  2. Implement SGD and Adam optimizers from scratch
  3. Build a 2-layer neural network using only NumPy
  4. Train on a simple dataset and compare SGD vs Adam convergence

Instructions:
  - Understand how loss functions measure prediction error.
  - Study how SGD and Adam update model weights differently.
  - Implement forward propagation, loss calculation, backpropagation,
    and weight updates without any deep learning framework.
"""

import numpy as np

np.random.seed(42)


# ────────────────────────────
# Activation Functions
# ────────────────────────────

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_backward(z):
    s = sigmoid(z)
    return s * (1 - s)

def relu(z):
    return np.maximum(0, z)

def relu_backward(z):
    return (z > 0).astype(float)


# ────────────────────────────
# Loss Functions
# ────────────────────────────

def mse_loss(y, y_hat):
    """Mean Squared Error."""
    return np.mean((y - y_hat) ** 2)

def binary_cross_entropy(y, y_hat, eps=1e-9):
    """Binary Cross-Entropy Loss."""
    y_hat = np.clip(y_hat, eps, 1 - eps)
    return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

def bce_backward(y, y_hat, eps=1e-9):
    """Gradient of BCE w.r.t. y_hat."""
    y_hat = np.clip(y_hat, eps, 1 - eps)
    m = y.shape[1]
    return (-(y / y_hat) + (1 - y) / (1 - y_hat)) / m


# ────────────────────────────────────────────
# Neural Network: 2 Layers (ReLU + Sigmoid)
# ────────────────────────────────────────────

class NeuralNetwork:
    """2-layer neural network: Input -> ReLU Hidden -> Sigmoid Output."""

    def __init__(self, n_input, n_hidden, n_output, optimizer="sgd", lr=0.01):
        self.lr = lr
        self.optimizer = optimizer

        # Weight initialization (He for ReLU)
        self.params = {
            "W1": np.random.randn(n_hidden, n_input) * np.sqrt(2 / n_input),
            "b1": np.zeros((n_hidden, 1)),
            "W2": np.random.randn(n_output, n_hidden) * np.sqrt(2 / n_hidden),
            "b2": np.zeros((n_output, 1)),
        }

        # Adam state
        self.m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.t = 0  # timestep for Adam bias correction

        self.cache = {}

    def forward(self, X):
        """Forward propagation."""
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]

        Z1 = W1 @ X + b1
        A1 = relu(Z1)
        Z2 = W2 @ A1 + b2
        A2 = sigmoid(Z2)

        self.cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
        return A2

    def backward(self, y):
        """Backward propagation — returns gradients."""
        X, Z1, A1, Z2, A2 = (
            self.cache["X"], self.cache["Z1"], self.cache["A1"],
            self.cache["Z2"], self.cache["A2"]
        )
        W2 = self.params["W2"]
        m = y.shape[1]

        # Output layer gradients
        dA2 = bce_backward(y, A2)
        dZ2 = dA2 * sigmoid_backward(Z2)
        dW2 = dZ2 @ A1.T
        db2 = dZ2.sum(axis=1, keepdims=True)

        # Hidden layer gradients
        dA1 = W2.T @ dZ2
        dZ1 = dA1 * relu_backward(Z1)
        dW1 = dZ1 @ X.T
        db1 = dZ1.sum(axis=1, keepdims=True)

        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def update_sgd(self, grads):
        """SGD weight update."""
        for key in self.params:
            self.params[key] -= self.lr * grads[key]

    def update_adam(self, grads, beta1=0.9, beta2=0.999, eps=1e-8):
        """Adam weight update."""
        self.t += 1
        for key in self.params:
            self.m[key] = beta1 * self.m[key] + (1 - beta1) * grads[key]
            self.v[key] = beta2 * self.v[key] + (1 - beta2) * grads[key] ** 2

            m_hat = self.m[key] / (1 - beta1 ** self.t)
            v_hat = self.v[key] / (1 - beta2 ** self.t)

            self.params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

    def train_step(self, X, y):
        """One full training step: forward -> loss -> backward -> update."""
        y_hat = self.forward(X)
        loss = binary_cross_entropy(y, y_hat)
        grads = self.backward(y)

        if self.optimizer == "adam":
            self.update_adam(grads)
        else:
            self.update_sgd(grads)

        return loss

    def predict(self, X, threshold=0.5):
        y_hat = self.forward(X)
        return (y_hat >= threshold).astype(int)


# ────────────────────────────
# Dataset: XOR problem
# ────────────────────────────

print("=" * 55)
print("Neural Network from Scratch — XOR Problem")
print("=" * 55)

# XOR is not linearly separable — needs hidden layer
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float).T  # shape (2, 4)
y = np.array([[0, 1, 1, 0]], dtype=float)                       # shape (1, 4)

EPOCHS = 5000


# ── Train with SGD ──
print("\nTraining with SGD (lr=0.1)...")
nn_sgd = NeuralNetwork(n_input=2, n_hidden=4, n_output=1, optimizer="sgd", lr=0.1)
sgd_losses = []
for epoch in range(EPOCHS):
    loss = nn_sgd.train_step(X, y)
    sgd_losses.append(loss)
    if (epoch + 1) % 1000 == 0:
        preds = nn_sgd.predict(X)
        acc = np.mean(preds == y)
        print(f"  Epoch {epoch+1:5d} | Loss: {loss:.4f} | Accuracy: {acc:.2f}")

print("Final SGD predictions:")
for i in range(X.shape[1]):
    xi = X[:, i:i+1]
    pred = nn_sgd.predict(xi)[0, 0]
    print(f"  {X[:, i].astype(int)} -> {pred}  (target: {int(y[0, i])})")


# ── Train with Adam ──
print("\nTraining with Adam (lr=0.01)...")
nn_adam = NeuralNetwork(n_input=2, n_hidden=4, n_output=1, optimizer="adam", lr=0.01)
adam_losses = []
for epoch in range(EPOCHS):
    loss = nn_adam.train_step(X, y)
    adam_losses.append(loss)
    if (epoch + 1) % 1000 == 0:
        preds = nn_adam.predict(X)
        acc = np.mean(preds == y)
        print(f"  Epoch {epoch+1:5d} | Loss: {loss:.4f} | Accuracy: {acc:.2f}")

print("Final Adam predictions:")
for i in range(X.shape[1]):
    xi = X[:, i:i+1]
    pred = nn_adam.predict(xi)[0, 0]
    print(f"  {X[:, i].astype(int)} -> {pred}  (target: {int(y[0, i])})")


# ── Compare convergence ──
print("\n" + "=" * 55)
print("Convergence Comparison (loss at checkpoints)")
print("=" * 55)
checkpoints = [0, 499, 999, 1999, 4999]
print(f"{'Epoch':<10} {'SGD Loss':<15} {'Adam Loss':<15}")
print("-" * 40)
for ep in checkpoints:
    print(f"{ep+1:<10} {sgd_losses[ep]:<15.4f} {adam_losses[ep]:<15.4f}")

print("\nKey takeaway: Adam typically converges faster and more reliably than plain SGD.")

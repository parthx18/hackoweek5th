# Week 16 Notes — Loss Functions, Optimizers & Neural Network from Scratch

## 1. Loss Functions

The loss function measures how wrong the model's predictions are. It is the objective the optimizer minimizes.

### Regression Losses

**Mean Squared Error (MSE)**
```
L = (1/n) * sum((y_i - y_hat_i)^2)
```
Gradient: dL/dy_hat = -(2/n) * (y - y_hat)
Sensitive to outliers (squares large errors).

**Mean Absolute Error (MAE)**
```
L = (1/n) * sum(|y_i - y_hat_i|)
```
More robust to outliers but gradient is not smooth at 0.

**Huber Loss (Smooth MAE)**
```
L = 0.5*(y - y_hat)^2       if |y - y_hat| <= delta
L = delta*(|y-y_hat| - 0.5*delta)   otherwise
```

### Classification Losses

**Binary Cross-Entropy (Log Loss)**
```
L = -(1/n) * sum(y*log(y_hat) + (1-y)*log(1-y_hat))
```
Used with sigmoid output for binary classification.

**Categorical Cross-Entropy**
```
L = -(1/n) * sum_i sum_c y_ic * log(y_hat_ic)
```
Used with softmax output for multi-class classification.

---

## 2. Optimizers

### Gradient Descent (Batch)
```python
w = w - lr * dL/dw    # computed over entire dataset
```
- Stable updates, but slow for large datasets

### Stochastic Gradient Descent (SGD)
```python
for xi, yi in dataset:
    w = w - lr * gradient(xi, yi)
```
- Fast updates, noisy, can escape local minima

### Mini-batch GD
```python
for batch in batches:
    w = w - lr * gradient(batch)
```
- Best of both worlds — most common in practice

### SGD with Momentum
```python
v = beta * v + (1-beta) * dL/dw
w = w - lr * v
```
- Accumulates velocity; smooths oscillations; beta typically 0.9

### Adam (Adaptive Moment Estimation)
```python
m = beta1 * m + (1-beta1) * dL/dw          # 1st moment (momentum)
v = beta2 * v + (1-beta2) * (dL/dw)**2     # 2nd moment (RMSProp)

m_hat = m / (1 - beta1**t)                  # bias correction
v_hat = v / (1 - beta2**t)

w = w - lr * m_hat / (sqrt(v_hat) + eps)
```
Default hyperparameters: lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8

Adam adapts the learning rate per parameter — very effective in practice.

---

## 3. Neural Network Architecture

```
Input (n_features)
    |
[Linear: Z1 = W1 @ X + b1]
[ReLU:   A1 = max(0, Z1)]
    |
[Linear: Z2 = W2 @ A1 + b2]
[Sigmoid: A2 = 1/(1+exp(-Z2))]
    |
Output (probability)
```

### Weight Initialization
```python
W1 = np.random.randn(n_hidden, n_input) * 0.01
b1 = np.zeros((n_hidden, 1))
W2 = np.random.randn(n_output, n_hidden) * 0.01
b2 = np.zeros((n_output, 1))
```

### Training Loop
```python
for epoch in range(epochs):
    # 1. Forward propagation
    Z1 = W1 @ X + b1
    A1 = relu(Z1)
    Z2 = W2 @ A1 + b2
    A2 = sigmoid(Z2)

    # 2. Compute loss
    loss = binary_cross_entropy(y, A2)

    # 3. Backward propagation
    dA2 = -(y/A2 - (1-y)/(1-A2)) / m
    dZ2 = dA2 * sigmoid_derivative(Z2)
    dW2 = dZ2 @ A1.T / m
    db2 = dZ2.sum(axis=1, keepdims=True) / m

    dA1 = W2.T @ dZ2
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = dZ1 @ X.T / m
    db1 = dZ1.sum(axis=1, keepdims=True) / m

    # 4. Update weights
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2
```

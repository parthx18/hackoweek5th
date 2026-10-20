# Week 16 — Loss Functions, Optimizers & Neural Network from Scratch

## 🎯 Topics

### Part 1: Loss Functions
- **Regression Losses**:
  - Mean Squared Error (MSE): average of squared differences
  - Mean Absolute Error (MAE): average of absolute differences
  - Huber Loss: MSE for small errors, MAE for large (robust)
- **Classification Losses**:
  - Binary Cross-Entropy: -(y*log(y_hat) + (1-y)*log(1-y_hat))
  - Categorical Cross-Entropy: -sum(y_i * log(y_hat_i)) — for multi-class
- **Why loss matters**: The loss surface guides gradient descent; choosing the wrong loss leads to poor training

### Part 2: Optimizers
- **Gradient Descent (Batch GD)**: update weights using gradients over entire dataset
- **SGD (Stochastic Gradient Descent)**: update weights using one sample at a time — noisy but fast
- **Mini-batch GD**: update weights using a small batch — balance between GD and SGD
- **Momentum**: accumulate velocity in gradient direction to smooth updates
- **Adam (Adaptive Moment Estimation)**:
  - Combines momentum (1st moment) + RMSProp (2nd moment)
  - Per-parameter adaptive learning rates
  - Update rule: w = w - lr * m_hat / (sqrt(v_hat) + epsilon)
  - Default hyperparameters: lr=0.001, beta1=0.9, beta2=0.999

### Part 3: Neural Network from Scratch (NumPy)
- Architecture: Input → Hidden Layers → Output
- Each layer: Z = W @ A_prev + b, A = activation(Z)
- Training loop:
  1. Forward propagation → compute loss
  2. Backward propagation → compute gradients
  3. Optimizer step → update weights

## 🗂️ Structure
```
Week16/
├── README.md
├── notes/
│   └── loss-optimizers-nn.md       <- Deep-dive on losses, optimizers, and NN math
└── mini-project/
    ├── requirements.txt
    └── neural_net_scratch.py       <- Full 2-layer NN built with NumPy only
```

## 🚀 Mini-Project: Neural Network from Scratch
Build a 2-layer neural network using only NumPy:
1. Initialize weights and biases
2. Forward pass: ReLU hidden layer + Sigmoid output
3. Compute binary cross-entropy loss
4. Backward pass: gradients via chain rule
5. Update weights with SGD and then with Adam
6. Compare convergence of SGD vs Adam

## 🔧 Setup & Run
```bash
cd Week16/mini-project
pip install -r requirements.txt
python neural_net_scratch.py
```

## ⏱️ Duration: 3 Hours
- **Hour 1**: Loss functions — math, intuition, and when to use each
- **Hour 2**: Optimizers — SGD, momentum, Adam derivation
- **Hour 3**: Neural network from scratch — full training loop in NumPy

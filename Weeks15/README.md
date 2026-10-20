# Week 15 — Model Deployment Basics & Neural Network Foundations

## 🎯 Topics

### Part 1: Model Deployment Basics
- **Saving & Loading Models**:
  - `pickle` — Python's built-in serialization module
  - `joblib` — Preferred for large NumPy arrays (sklearn models)
- **Workflow**:
  1. Train model → `pickle.dump()` / `joblib.dump()`
  2. Load model → `pickle.load()` / `joblib.load()`
  3. Run inference on new data
- **Best Practices**: versioning model files, storing preprocessing pipelines together with the model
### Part 2: Perceptrons & Activation Functions
- **Perceptron**: Simplest neural unit — weighted sum of inputs + bias passed through an activation
- **Common Activation Functions**:
  | Function | Formula | Range |
  |---|---|---|
  | Sigmoid | 1/(1+e^-z) | (0, 1) |
  | Tanh | (e^z - e^-z)/(e^z + e^-z) | (-1, 1) |
  | ReLU | max(0, z) | [0, inf) |
  | Softmax | e^zi / sum(e^zj) | (0, 1) |

### Part 3: Forward & Backward Propagation
- **Forward Pass**: Compute activations layer by layer from input to output
- **Backward Pass (Backprop)**: Apply chain rule to propagate loss gradients back through layers
- **Gradient of activations**:
  - Sigmoid: sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z))
  - ReLU: 1 if z > 0 else 0

## 🗂️ Structure
```
Week15/
├── README.md
├── notes/
│   └── deployment-and-perceptrons.md   <- Theory notes on saving models & neural net basics
└── mini-project/
    ├── requirements.txt
    └── deploy_and_perceptron.py        <- Script: save/load a model + perceptron demo
```

## 🚀 Mini-Project: Deploy & Perceive
1. Train a simple sklearn classifier, save it with both `pickle` and `joblib`
2. Reload the model and verify predictions match
3. Implement a single perceptron with sigmoid activation from scratch
4. Demonstrate forward pass and manual gradient calculation

## 🔧 Setup & Run
```bash
cd Week15/mini-project
pip install -r requirements.txt
python deploy_and_perceptron.py
```

## ⏱️ Duration: 3 Hours
- **Hour 1**: Model serialization with pickle & joblib — save, load, verify
- **Hour 2**: Perceptrons and activation functions — math and code
- **Hour 3**: Forward and backward propagation — chain rule walk-through

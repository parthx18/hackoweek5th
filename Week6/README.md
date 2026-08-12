# Week 6 — Calculus & Backpropagation Intuition

## 🎯 Topics
- Derivatives: rate of change, tangents, local minima/maxima
- Gradients: multi-variable derivatives, direction of steepest ascent/descent
- Chain Rule: compounding rates of change (layer to layer)
- Backpropagation intuition: updating neural network weights

## 🗂️ Structure
```
Week6/
├── README.md
├── notes/
│   └── calculus.md             ← Calculus notes & Backprop intuition
└── mini-project/
    ├── requirements.txt
    └── gradient_descent.py     ← Interactive Gradient Descent Visualizer
```

## 🚀 Mini-Project: Gradient Descent Visualizer
A Python program that:
- Runs gradient descent on a 1D function (e.g., $y = x^2 - 4x + 6$)
- Runs gradient descent on a 2D function (e.g., a bowl-shaped cost surface)
- Saves step-by-step optimization paths as PNG charts
- Demonstrates how the learning rate affects convergence

## 🔧 Setup & Run
```bash
cd mini-project
pip install -r requirements.txt
python gradient_descent.py
```

## ⏱️ Duration: 3 Hours
- Hour 1: Derivatives and Gradients (rate of change, climbing down the loss curve)
- Hour 2: Chain Rule & Backpropagation (propagating error backwards)
- Hour 3: Build the Gradient Descent visualizer mini-project

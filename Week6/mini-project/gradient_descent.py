"""
====================================================
Mini-Project: Gradient Descent Visualizer
Week 6 — Calculus (Math for ML)
====================================================
Concepts used:
  ✅ Derivatives & Slopes
  ✅ Multi-variable Gradients
  ✅ Gradient Descent Optimization loop
  ✅ Learning Rate parameter analysis
====================================================
Run:  python gradient_descent.py
Output: Saves 3 PNG visualization charts in this folder
====================================================
"""

import numpy as np
import matplotlib.pyplot as plt

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# ====================================================
# STEP 1: Gradient Descent in 1D
# Function: f(x) = x^2 - 4x + 6
# Derivative: f'(x) = 2x - 4
# Minimum is at x = 2
# ====================================================
def f_1d(x):
    return x**2 - 4*x + 6

def df_1d(x):
    return 2*x - 4

def run_gd_1d(start_x, lr, epochs):
    x = start_x
    history_x = [x]
    history_y = [f_1d(x)]
    
    for epoch in range(epochs):
        grad = df_1d(x)
        x = x - lr * grad
        history_x.append(x)
        history_y.append(f_1d(x))
        
    return np.array(history_x), np.array(history_y)

def visualize_1d_gd():
    print("=" * 55)
    print("  1️⃣  RUNNING GRADIENT DESCENT IN 1D")
    print("=" * 55)
    
    start_x = -3.0
    lr = 0.1
    epochs = 15
    
    hist_x, hist_y = run_gd_1d(start_x, lr, epochs)
    
    # Print steps to console
    for i, (x_val, y_val) in enumerate(zip(hist_x[:6], hist_y[:6])):
        print(f"  Step {i:2d}: x = {x_val:6.3f} | f(x) = {y_val:6.3f} | Gradient = {df_1d(x_val):6.3f}")
    print("  ...")
    print(f"  Final Step: x = {hist_x[-1]:6.3f} | f(x) = {hist_y[-1]:6.3f}")
    
    # Plotting
    fig, ax = plt.subplots(figsize=(8, 5))
    x_curve = np.linspace(-4, 8, 200)
    ax.plot(x_curve, f_1d(x_curve), 'gray', label='Cost Function f(x) = x² - 4x + 6', linewidth=2)
    
    # Scatter points of steps
    ax.scatter(hist_x, hist_y, color='red', zorder=3, s=40)
    # Arrows showing path
    for i in range(len(hist_x)-1):
        ax.annotate('', xy=(hist_x[i+1], hist_y[i+1]), xytext=(hist_x[i], hist_y[i]),
                    arrowprops=dict(arrowstyle="->", color='red', lw=1.5, ls='-'))
        
    ax.scatter(hist_x[0], hist_y[0], color='blue', s=100, label='Start Point', zorder=4)
    ax.scatter(2.0, 2.0, color='green', s=120, marker='*', label='Global Minimum (x=2)', zorder=4)
    
    ax.set_title("Gradient Descent Optimization in 1D", fontsize=14, fontweight='bold')
    ax.set_xlabel("Parameter x")
    ax.set_ylabel("Loss / Cost")
    ax.legend()
    
    plt.savefig('visual_1_gd_1d.png', dpi=150)
    plt.close()
    print("  ✅ Saved visualization: visual_1_gd_1d.png\n")


# ====================================================
# STEP 2: The Impact of Learning Rate (LR)
# ====================================================
def visualize_learning_rates():
    print("=" * 55)
    print("  2️⃣  ANALYZING LEARNING RATES")
    print("=" * 55)
    
    start_x = -3.0
    epochs = 10
    
    lrs = [0.02, 0.2, 0.95]
    colors = ['orange', 'green', 'red']
    labels = ['Too Small (0.02) - Slow', 'Good (0.2) - Fast', 'Too Large (0.95) - Oscillating']
    
    fig, ax = plt.subplots(figsize=(9, 5))
    x_curve = np.linspace(-4, 8, 200)
    ax.plot(x_curve, f_1d(x_curve), 'gray', alpha=0.5, linewidth=1.5)
    
    for lr, color, label in zip(lrs, colors, labels):
        hist_x, hist_y = run_gd_1d(start_x, lr, epochs)
        ax.plot(hist_x, hist_y, '-o', color=color, label=label, markersize=5)
        print(f"  LR: {lr:<4} -> Final x after 10 steps: {hist_x[-1]:7.3f}")
        
    ax.set_title("How Learning Rate Affects Gradient Descent", fontsize=14, fontweight='bold')
    ax.set_xlabel("Parameter x")
    ax.set_ylabel("Loss / Cost")
    ax.legend()
    
    plt.savefig('visual_2_gd_learning_rates.png', dpi=150)
    plt.close()
    print("  ✅ Saved visualization: visual_2_gd_learning_rates.png\n")


# ====================================================
# STEP 3: Multi-variable Gradient Descent in 2D
# Function: f(x, y) = x^2 + 2 * y^2
# Gradients: df/dx = 2x, df/dy = 4y
# Minimum is at (0, 0)
# ====================================================
def f_2d(x, y):
    return x**2 + 2*(y**2)

def grad_2d(x, y):
    return np.array([2*x, 4*y])

def run_gd_2d(start, lr, epochs):
    pos = np.array(start, dtype=float)
    history = [pos.copy()]
    
    for epoch in range(epochs):
        g = grad_2d(pos[0], pos[1])
        pos -= lr * g
        history.append(pos.copy())
        
    return np.array(history)

def visualize_2d_gd():
    print("=" * 55)
    print("  3️⃣  RUNNING GRADIENT DESCENT IN 2D")
    print("=" * 55)
    
    start_pos = [3.5, 3.0]
    lr = 0.15
    epochs = 15
    
    history = run_gd_2d(start_pos, lr, epochs)
    
    # Setup coordinates for 3D/contour surface
    x = np.linspace(-4, 4, 100)
    y = np.linspace(-4, 4, 100)
    X, Y = np.meshgrid(x, y)
    Z = f_2d(X, Y)
    
    fig, ax = plt.subplots(figsize=(7, 7))
    
    # Plot contour map
    contours = ax.contour(X, Y, Z, levels=20, cmap='viridis')
    ax.clabel(contours, inline=True, fontsize=8)
    
    # Plot steps path
    hist_x = history[:, 0]
    hist_y = history[:, 1]
    
    ax.plot(hist_x, hist_y, 'ro-', label='GD Trajectory', linewidth=2)
    ax.scatter(hist_x[0], hist_y[0], color='blue', s=100, label='Start Point (3.5, 3)', zorder=4)
    ax.scatter(0, 0, color='green', s=150, marker='*', label='Global Minimum (0,0)', zorder=5)
    
    ax.set_title("Gradient Descent Contour Path (2D bowl)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Parameter x")
    ax.set_ylabel("Parameter y")
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect('equal')
    ax.legend()
    
    plt.savefig('visual_3_gd_2d.png', dpi=150)
    plt.close()
    print("  ✅ Saved visualization: visual_3_gd_2d.png\n")


def main():
    print("\n" + "═" * 55)
    print("  📉  GRADIENT DESCENT EXPLORER")
    print("  Week 6 Mini-Project — Optimization & Learning Rates")
    print("═" * 55 + "\n")
    
    visualize_1d_gd()
    visualize_learning_rates()
    visualize_2d_gd()
    
    print("=" * 55)
    print("  🎉 Optimization paths completed and charts saved successfully!")
    print("=" * 55 + "\n")

if __name__ == '__main__':
    main()

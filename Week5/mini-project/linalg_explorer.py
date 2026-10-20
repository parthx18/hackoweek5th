"""
====================================================
Mini-Project: Linear Algebra Explorer
Week 5 — Linear Algebra (Math for ML)
====================================================
Concepts used:
  ✅ Vector addition & scaling
  ✅ Dot product projection & angle computation
  ✅ Matrix transformation on geometric objects
  ✅ Eigenvalues & Eigenvectors visualization
====================================================
Run:  python linalg_explorer.py
Output: Saves 3 PNG visualization charts in this folder
====================================================
"""

import numpy as np
import matplotlib.pyplot as plt

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

def plot_vector(ax, vector, origin=[0, 0], color='b', label=None):
    """Draws a vector arrow on the provided axes."""
    ax.quiver(origin[0], origin[1], vector[0], vector[1], 
              angles='xy', scale_units='xy', scale=1, 
              color=color, label=label, width=0.015)

# ====================================================
# STEP 1: Vector Addition & Scaling Visualization
# ====================================================
def explore_vectors():
    print("=" * 55)
    print("  1️⃣  EXPLORING VECTORS & SCALING")
    print("=" * 55)
    
    v1 = np.array([3, 1])
    v2 = np.array([-1, 3])
    
    # Vector addition
    v_sum = v1 + v2
    # Vector scaling
    v1_scaled = v1 * 1.5
    
    print(f"Vector v1         : {v1}")
    print(f"Vector v2         : {v2}")
    print(f"Sum (v1 + v2)     : {v_sum}")
    print(f"Scaled (1.5 * v1) : {v1_scaled}")
    
    # Plotting
    fig, ax = plt.subplots(figsize=(7, 7))
    
    plot_vector(ax, v1, color='blue', label='v1 [3, 1]')
    plot_vector(ax, v2, color='green', label='v2 [-1, 3]')
    
    # Draw sum as tip-to-tail
    plot_vector(ax, v2, origin=v1, color='green', label='v2 shifted (tip-to-tail)')
    plot_vector(ax, v_sum, color='red', label='v1 + v2 (Resultant)')
    plot_vector(ax, v1_scaled, color='orange', label='1.5 * v1')
    
    # Formatting
    ax.axhline(0, color='black',linewidth=1)
    ax.axvline(0, color='black',linewidth=1)
    ax.set_xlim(-2, 6)
    ax.set_ylim(-1, 6)
    ax.set_aspect('equal')
    ax.set_title("Vector Addition & Scaling", fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    
    plt.savefig('visual_1_vector_ops.png', dpi=150)
    plt.close()
    print("  ✅ Saved visualization: visual_1_vector_ops.png\n")


# ====================================================
# STEP 2: Dot Product & Projections
# ====================================================
def explore_dot_product():
    print("=" * 55)
    print("  2️⃣  EXPLORING DOT PRODUCT & PROJECTION")
    print("=" * 55)
    
    a = np.array([4, 2])
    b = np.array([5, 0])
    
    dot_prod = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    # Angle calculation
    cos_theta = dot_prod / (norm_a * norm_b)
    angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)
    
    # Projection of a onto b
    # proj_b(a) = ( (a . b) / ||b||^2 ) * b
    proj_a_on_b = (dot_prod / (norm_b ** 2)) * b
    
    print(f"Vector a           : {a} (Magnitude: {norm_a:.2f})")
    print(f"Vector b           : {b} (Magnitude: {norm_b:.2f})")
    print(f"Dot Product (a . b): {dot_prod}")
    print(f"Angle between them : {angle_deg:.1f}°")
    print(f"Projection of a on b: {proj_a_on_b}")
    
    # Plotting
    fig, ax = plt.subplots(figsize=(7, 7))
    
    plot_vector(ax, a, color='blue', label='a [4, 2]')
    plot_vector(ax, b, color='black', label='b [5, 0]')
    plot_vector(ax, proj_a_on_b, color='red', label='Projection of a on b')
    
    # Dashed line showing the perpendicular drop
    ax.plot([a[0], proj_a_on_b[0]], [a[1], proj_a_on_b[1]], 'r--', label='Projection line')
    
    # Formatting
    ax.axhline(0, color='black',linewidth=1)
    ax.axvline(0, color='black',linewidth=1)
    ax.set_xlim(-1, 6)
    ax.set_ylim(-1, 4)
    ax.set_aspect('equal')
    ax.set_title("Vector Dot Product Projection", fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    
    plt.savefig('visual_2_dot_product.png', dpi=150)
    plt.close()
    print("  ✅ Saved visualization: visual_2_dot_product.png\n")


# ====================================================
# STEP 3: Matrix Transformations & Eigenvalues
# ====================================================
def explore_matrix_transformation():
    print("=" * 55)
    print("  3️⃣  MATRIX TRANSFORMATIONS & EIGENVALUES")
    print("=" * 55)
    
    # Transformation matrix A
    # Let's use a shear & stretch matrix
    A = np.array([[2, 1],
                  [1, 2]])
    
    print("Transformation Matrix A:")
    print(A)
    
    # Calculate Eigenvalues and Eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(A)
    print(f"\nEigenvalue 1: {eigenvalues[0]:.2f}")
    print(f"Eigenvector 1: {eigenvectors[:, 0]}")
    print(f"Eigenvalue 2: {eigenvalues[1]:.2f}")
    print(f"Eigenvector 2: {eigenvectors[:, 1]}")
    
    # Generate coordinates for a unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    circle_coords = np.array([np.cos(theta), np.sin(theta)])
    
    # Transform circle coordinates using matrix A
    transformed_coords = A @ circle_coords
    
    # Plot original and transformed circles
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot original unit circle (dashed)
    ax.plot(circle_coords[0, :], circle_coords[1, :], 'k--', alpha=0.5, label='Original Unit Circle')
    
    # Plot transformed ellipse
    ax.plot(transformed_coords[0, :], transformed_coords[1, :], 'purple', linewidth=2, label='Transformed Shape')
    
    # Plot eigenvectors scaled by their eigenvalues
    v1 = eigenvectors[:, 0] * eigenvalues[0]
    v2 = eigenvectors[:, 1] * eigenvalues[1]
    
    plot_vector(ax, v1, color='orange', label=f'Eigenvector 1 (λ={eigenvalues[0]:.1f})')
    plot_vector(ax, v2, color='red', label=f'Eigenvector 2 (λ={eigenvalues[1]:.1f})')
    
    # Let's plot another random vector to show how it changes direction
    rand_v = np.array([1, 0])
    transformed_rand_v = A @ rand_v
    plot_vector(ax, rand_v, color='lightblue', label='Standard Basis x [1, 0]')
    plot_vector(ax, transformed_rand_v, color='blue', label='Transformed x')
    
    # Formatting
    ax.axhline(0, color='black',linewidth=1)
    ax.axvline(0, color='black',linewidth=1)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect('equal')
    ax.set_title("Matrix Transformation & Eigenvectors", fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    
    plt.savefig('visual_3_eigen_transformation.png', dpi=150)
    plt.close()
    print("  ✅ Saved visualization: visual_3_eigen_transformation.png\n")


def main():
    print("\n" + "═" * 55)
    print("  📐  LINEAR ALGEBRA EXPLORER")
    print("  Week 5 Mini-Project — Operations & Intuitions")
    print("═" * 55 + "\n")
    
    explore_vectors()
    explore_dot_product()
    explore_matrix_transformation()
    
    print("=" * 55)
    print("  🎉 All operations processed and charts saved successfully!")
    print("=" * 55 + "\n")

if __name__ == '__main__':
    main()

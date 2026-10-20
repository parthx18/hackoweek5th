# Linear Algebra for Machine Learning — Notes

Linear Algebra is the language of machine learning. Almost all data, parameters, and operations in ML models (especially deep learning) are represented as vectors and matrices.

---

## 1. Vectors: The Basic Building Blocks

A **vector** is an ordered list of numbers. In ML, a vector represents a single data point's features (e.g., [age, income, education]).

### Creation and Representation
A vector can be represented as a column:
$$ v = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix} $$

In NumPy, we represent vectors as 1D arrays:
```python
import numpy as np
v = np.array([3, 4])
```

### Vector Operations
*   **Addition:** Done element-wise.
    $$ \begin{bmatrix} 1 \\ 2 \end{bmatrix} + \begin{bmatrix} 3 \\ 4 \end{bmatrix} = \begin{bmatrix} 4 \\ 6 \end{bmatrix} $$
*   **Scalar Multiplication:** Scales the vector's length without changing its direction.
    $$ 2 \cdot \begin{bmatrix} 3 \\ 4 \end{bmatrix} = \begin{bmatrix} 6 \\ 8 \end{bmatrix} $$
*   **Magnitude (L2 Norm / Length):**
    $$ ||v|| = \sqrt{v_1^2 + v_2^2 + \dots + v_n^2} $$
    ```python
    magnitude = np.linalg.norm(v)  # returns 5.0 for [3, 4]
    ```

---

## 2. The Dot Product

The dot product takes two vectors of the same dimension and returns a single number (scalar).

### Algebraic Definition
$$ a \cdot b = \sum_{i=1}^n a_i b_i = a_1 b_1 + a_2 b_2 + \dots + a_n b_n $$

### Geometric Definition
$$ a \cdot b = ||a|| \cdot ||b|| \cos(\theta) $$
where $\theta$ is the angle between the two vectors.

*   If the dot product is **positive**, the vectors point in a similar direction ($\theta < 90^\circ$).
*   If the dot product is **zero**, the vectors are **orthogonal** (perpendicular, $\theta = 90^\circ$).
*   If the dot product is **negative**, they point in opposite directions ($\theta > 90^\circ$).

```python
a = np.array([1, 2])
b = np.array([3, 4])
dot_product = np.dot(a, b)  # 1*3 + 2*4 = 11
```

---

## 3. Matrices: Grids of Numbers

A **matrix** is a 2D grid of numbers. We use matrices to represent entire datasets (rows = samples, columns = features) or weights of neural network layers.

### Matrix Operations
*   **Transpose ($A^T$):** Swapping rows and columns.
    ```python
    A = np.array([[1, 2], [3, 4]])
    A_T = A.T  # [[1, 3], [2, 4]]
    ```
*   **Matrix Multiplication (Dot Product of Matrices):**
    To multiply matrix $A$ (shape $m \times n$) by $B$ (shape $n \times p$), the inner dimensions must match. The result has shape $m \times p$.
    ```python
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    C = np.matmul(A, B)  # Or: A @ B
    ```
*   **Identity Matrix ($I$):** A square matrix with ones on the diagonal and zeros elsewhere. Multiplying any matrix by $I$ yields the original matrix.
    ```python
    I = np.eye(2)  # [[1, 0], [0, 1]]
    ```
*   **Inverse ($A^{-1}$):** If $A$ is a square matrix, its inverse satisfies $A \cdot A^{-1} = I$. Not all matrices have an inverse (singular matrices).
    ```python
    A_inv = np.linalg.inv(A)
    ```

---

## 4. Eigenvalues & Eigenvectors (Intuition-level)

When you multiply a matrix $A$ by a vector $x$, the vector usually changes both its **direction** and **scale**. 

However, for some special vectors, multiplying them by $A$ **only scales** the vector, keeping its direction the same. These are **eigenvectors**, and the scaling factor is the **eigenvalue** ($\lambda$).

$$ A x = \lambda x $$

*   $A$: Transformation matrix
*   $x$: Eigenvector
*   $\lambda$: Eigenvalue (scalar)

### Why does this matter in ML?
1.  **Dimensionality Reduction (PCA):** Eigenvectors point in the directions of maximum variance in data.
2.  **Spectral Clustering & PageRank:** Finding dominant states/relations in graphs.

```python
A = np.array([[4, 2], [1, 3]])
eigenvalues, eigenvectors = np.linalg.eig(A)
```

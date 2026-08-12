# NumPy — Notes

## What is NumPy?
NumPy (Numerical Python) is the core library for numerical computing.
It provides the **ndarray** — a fast, memory-efficient multi-dimensional array.

```bash
pip install numpy
```

---

## 1. Creating Arrays

```python
import numpy as np

# From a list
a = np.array([1, 2, 3, 4, 5])

# Common creators
np.zeros(5)           # [0. 0. 0. 0. 0.]
np.ones((2, 3))       # 2x3 matrix of 1s
np.arange(0, 10, 2)   # [0 2 4 6 8]  (start, stop, step)
np.linspace(0, 1, 5)  # [0.   0.25  0.5   0.75  1. ]  (5 evenly spaced)
np.random.rand(3, 3)  # 3x3 matrix of random floats [0,1)
np.random.randint(0, 100, size=(4,))  # 4 random ints from 0–99
```

---

## 2. Array Properties

```python
a = np.array([[1, 2, 3], [4, 5, 6]])

a.shape    # (2, 3) — 2 rows, 3 cols
a.ndim     # 2 — number of dimensions
a.size     # 6 — total elements
a.dtype    # dtype('int64') — data type
```

---

## 3. Indexing & Slicing

```python
a = np.array([10, 20, 30, 40, 50])

a[0]      # 10  (first element)
a[-1]     # 50  (last element)
a[1:4]    # [20, 30, 40]  (slice)
a[::2]    # [10, 30, 50]  (every 2nd)

# 2D array
m = np.array([[1,2,3],[4,5,6],[7,8,9]])
m[0, 1]   # 2       (row 0, col 1)
m[:, 1]   # [2,5,8] (entire column 1)
m[1, :]   # [4,5,6] (entire row 1)

# Boolean indexing
a = np.array([10, 25, 30, 5, 80])
a[a > 20]  # [25, 30, 80]  — elements > 20
```

---

## 4. Vectorized Operations

NumPy operates on entire arrays at once — NO loops needed!

```python
a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])

a + b        # [11, 22, 33, 44]
a * 2        # [ 2,  4,  6,  8]
a ** 2       # [ 1,  4,  9, 16]
np.sqrt(a)   # [1. , 1.41, 1.73, 2. ]

# vs Python list (much slower):
# [x**2 for x in a]  ← loop-based
```

---

## 5. Broadcasting

Broadcasting lets NumPy do math on arrays of **different shapes**.

```python
a = np.array([[1, 2, 3],   # shape (2, 3)
              [4, 5, 6]])

b = np.array([10, 20, 30]) # shape (3,) — broadcasts across rows

a + b
# [[11, 22, 33],
#  [14, 25, 36]]
```

Rule: dimensions are compatible if they are equal OR one of them is 1.

---

## 6. Aggregation Functions

```python
a = np.array([4, 7, 2, 9, 1, 5])

np.sum(a)      # 28
np.mean(a)     # 4.67
np.median(a)   # 4.5
np.std(a)      # standard deviation
np.var(a)      # variance
np.min(a)      # 1
np.max(a)      # 9
np.argmin(a)   # 4  (index of min)
np.argmax(a)   # 3  (index of max)

# On 2D arrays — axis matters
m = np.array([[1,2,3],[4,5,6]])
np.sum(m, axis=0)  # [5, 7, 9]  — sum down columns
np.sum(m, axis=1)  # [6, 15]    — sum across rows
```

---

## 7. Reshaping

```python
a = np.arange(12)           # [ 0  1  2 ... 11]
a.reshape(3, 4)             # 3 rows, 4 cols
a.reshape(2, 2, 3)          # 3D array
a.flatten()                 # back to 1D

# Stacking arrays
np.vstack([a, b])   # vertical stack (rows)
np.hstack([a, b])   # horizontal stack (cols)
```

---

## Key Differences: NumPy vs Python Lists

| Feature | Python List | NumPy Array |
|---------|-------------|-------------|
| Speed | Slow (loops) | Very fast (C-based) |
| Memory | More | Less |
| Math ops | Manual loop | Vectorized |
| Data types | Mixed | Single type |
| Dimensions | 1D only | N-dimensional |

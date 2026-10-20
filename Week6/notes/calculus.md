# Calculus for Machine Learning — Notes

In Machine Learning, calculus is used to **train** models. It tells us how to tweak the model's weights to minimize error (loss).

---

## 1. Derivatives (Rate of Change)

The derivative of a function $f(x)$ measures how much the output changes when we make a tiny change to the input $x$.

*   **Tangent Line:** The derivative at point $x$ is the slope of the tangent line at that point.
*   **Sign of Derivative:**
    *   If $f'(x) > 0$ (positive): The function is increasing. To decrease $f(x)$, we should decrease $x$.
    *   If $f'(x) < 0$ (negative): The function is decreasing. To decrease $f(x)$, we should increase $x$.
    *   If $f'(x) = 0$: We are at a **local minimum** or **maximum** (slope is flat).

### Common Derivative Rules
*   **Power Rule:** $\frac{d}{dx}(x^n) = n x^{n-1}$
*   **Constant Rule:** $\frac{d}{dx}(C) = 0$
*   **Addition Rule:** $\frac{d}{dx}(f + g) = f' + g'$

---

## 2. Gradients (Multi-variable Calculus)

Most ML models have millions of inputs (parameters). A **gradient** is simply a vector containing the partial derivatives of a function with respect to all of its variables.

If we have a loss function $L(w_1, w_2)$:
$$ \nabla L = \begin{bmatrix} \frac{\partial L}{\partial w_1} \\ \frac{\partial L}{\partial w_2} \end{bmatrix} $$

### Geometric Property of the Gradient
*   The gradient vector $\nabla L$ points in the direction of the **steepest ascent** (fastest increase in loss).
*   The negative gradient $-\nabla L$ points in the direction of the **steepest descent** (fastest decrease in loss).

Therefore, to minimize our loss, we update our weights in the direction of the negative gradient:
$$ w_{new} = w_{old} - \alpha \cdot \nabla L $$
where $\alpha$ is the **learning rate**.

---

## 3. The Chain Rule

The chain rule is used when we have nested functions: $y = f(g(x))$. It calculates how a change in $x$ propagates through $g$ to affect $y$.

### Formula
$$ \frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx} \quad \text{where } u = g(x) $$

### Chain Rule in Neural Networks
In a neural network, the output is computed sequentially through layers:
$$\text{Input } x \rightarrow \text{Layer 1 } (a) \rightarrow \text{Layer 2 } (b) \rightarrow \text{Loss } (L)$$

To find how the weights in Layer 1 affect the Loss, we chain the derivatives together:
$$ \frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial b} \cdot \frac{\partial b}{\partial a} \cdot \frac{\partial a}{\partial w_1} $$

---

## 4. Backpropagation Intuition

**Backpropagation** is the algorithm that computes gradients throughout a neural network by applying the **Chain Rule** backward from the output layer to the input layer.

### The 4 Step Process:
1.  **Forward Pass:** Feed input features forward through the network to generate a prediction $\hat{y}$ and compute the loss $L(y, \hat{y})$.
2.  **Compute Output Error:** Calculate how much the output layer contributed to the error ($\frac{\partial L}{\partial \hat{y}}$).
3.  **Backward Pass (Chain Rule):** Propagate the error backward through the layers, calculating the gradient of the loss with respect to each weight.
4.  **Gradient Descent Update:** Adjust each weight slightly in the direction that reduces the error (subtracting the gradient).

# Regularization in Deep Learning: Dropout, Batch Normalization & Early Stopping

Deep neural networks contain millions of parameters and are prone to **overfitting** (memorizing training data noise instead of learning general patterns). Regularization encompasses techniques designed to constrain model complexity and improve generalization.

---

## 1. The Overfitting Problem

- **Underfitting (High Bias)**: Model is too simple to capture underlying patterns.
- **Good Fit**: Balanced performance on both training and unseen test data.
- **Overfitting (High Variance)**: Training loss continues decreasing, but validation loss starts increasing.

```
Loss
 │
 │      Validation Loss (starts rising when overfitting)
 │         \       /
 │          \__.__/
 │
 │           \
 │            \____  Training Loss (keeps dropping)
 └────────────────────────────► Epochs
                   ▲
             Early Stopping Point
```

---

## 2. Dropout

Introduced by Srivastava et al. (2014), **Dropout** randomly zeroes out a fraction $p$ of hidden units at each training step.

### Mechanics & Inverted Dropout
During training, each neuron activation $a_i$ is masked:
$$r_j \sim \text{Bernoulli}(1 - p)$$
$$\tilde{a}_j = \frac{r_j \cdot a_j}{1 - p}$$

The scaling factor $\frac{1}{1 - p}$ (known as *inverted dropout*) ensures that the expected value of activations remains unchanged between training and inference:
$$\mathbb{E}[\tilde{a}_j] = a_j$$

### Train vs. Eval Modes
- `model.train()`: Random dropout mask applied; activations scaled by $1/(1-p)$.
- `model.eval()`: Dropout is disabled; identity mapping applied.

---

## 3. Batch Normalization

Introduced by Ioffe and Szegedy (2015), **Batch Normalization (BatchNorm)** normalizes layer inputs across the mini-batch dimension.

### Formulas
For a mini-batch $\mathcal{B} = \{x_1, \dots, x_m\}$:
1. **Batch Mean**:
   $$\mu_\mathcal{B} = \frac{1}{m} \sum_{i=1}^m x_i$$
2. **Batch Variance**:
   $$\sigma_\mathcal{B}^2 = \frac{1}{m} \sum_{i=1}^m (x_i - \mu_\mathcal{B})^2$$
3. **Normalize**:
   $$\hat{x}_i = \frac{x_i - \mu_\mathcal{B}}{\sqrt{\sigma_\mathcal{B}^2 + \epsilon}}$$
4. **Scale & Shift**:
   $$y_i = \gamma \hat{x}_i + \beta$$
   where $\gamma$ and $\beta$ are learnable affine parameters.

### Why Batch Normalization Works
- Reduces internal covariate shift (distribution shift of internal activations).
- Smooths the optimization loss surface, allowing higher learning rates.
- Acts as a mild regularizer because mini-batch statistics add slight stochastic noise.
- Maintains running averages of $\mu$ and $\sigma^2$ for use during inference (`eval()` mode).

---

## 4. Weight Decay ($L_2$ Regularization)

Weight decay adds a penalty proportional to the sum of squared weights to the loss function:
$$\mathcal{L}_{\text{reg}} = \mathcal{L}_0 + \frac{\lambda}{2} \sum_w w^2$$

In PyTorch, weight decay is passed directly to the optimizer:
```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
```

At each gradient step, weights are shrunken by a factor $(1 - \eta \lambda)$ before the gradient update:
$$w_{t+1} = (1 - \eta \lambda) w_t - \eta \nabla \mathcal{L}_0(w_t)$$

---

## 5. Early Stopping

Early stopping halts training when the validation performance ceases to improve for a predefined number of epochs (**patience**).

### Algorithm
1. Track best validation loss: $\mathcal{L}_{\text{best}} \leftarrow \infty$.
2. In each epoch:
   - Compute $\mathcal{L}_{\text{val}}$.
   - If $\mathcal{L}_{\text{val}} < \mathcal{L}_{\text{best}} - \delta$ ($\delta$ = min delta):
     - Update $\mathcal{L}_{\text{best}} \leftarrow \mathcal{L}_{\text{val}}$.
     - Save model state dictionary (`torch.save(model.state_dict(), path)`).
     - Reset patience counter: $\text{counter} \leftarrow 0$.
   - Else:
     - Increment patience counter: $\text{counter} \leftarrow \text{counter} + 1$.
     - If $\text{counter} \ge \text{patience}$: Stop training and restore best checkpoint.


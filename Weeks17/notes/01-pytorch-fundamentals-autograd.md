# PyTorch Fundamentals: Tensors, Autograd & Training Loops

## 1. What is PyTorch?
PyTorch is an open-source deep learning framework based on the Torch library. It provides two high-level features:
1. **Tensor computing**: Similar to NumPy, but with strong acceleration via Graphics Processing Units (GPUs) and Tensor Processing Units (TPUs).
2. **Deep neural networks**: Built on a tape-based **autograd** automatic differentiation system that supports dynamic computational graphs.

---

## 2. PyTorch Tensors
A tensor is a multi-dimensional matrix containing elements of a single data type.

### Common Operations
- **Creation**:
  ```python
  import torch
  x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])       # From data
  zeros = torch.zeros((3, 3), dtype=torch.float32)  # Zeros
  rand = torch.randn((2, 4))                        # Standard normal
  ```
- **Device Management**:
  ```python
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  x = x.to(device)
  ```
- **Shape Transformations**:
  - `x.view(-1, 4)`: Reshape (requires contiguous memory)
  - `x.reshape(-1, 4)`: Reshape (copies if non-contiguous)
  - `x.squeeze()`: Removes singleton dimensions
  - `x.unsqueeze(dim=0)`: Inserts dimension of size 1 at specified position
  - `x.permute(2, 0, 1)`: Swaps multiple dimensions

---

## 3. The Autograd Engine (Automatic Differentiation)
PyTorch records all operations executed on tensors that have `requires_grad=True` into a directed acyclic graph (DAG).

```
 x (input) ────┐
               ├──► z = x * w + b ──► Loss = (z - y)^2
 w (weight) ───┘
```

### Key Autograd Mechanics
1. **Leaf Tensors**: User-created tensors that have `requires_grad=True` (e.g., weights and biases).
2. **Non-Leaf Tensors**: Created via operations on other tensors (e.g., intermediate activations and loss).
3. **`backward()`**: Traverses the DAG from the loss tensor backward to leaves using the chain rule:
   $$\frac{\partial \mathcal{L}}{\partial w} = \sum \frac{\partial \mathcal{L}}{\partial z} \cdot \frac{\partial z}{\partial w}$$
4. **Gradient Accumulation**: Gradients are accumulated into `.grad` (`w.grad += dL/dw`). Therefore, before each step in a training loop, you must call:
   ```python
   optimizer.zero_grad()
   ```
5. **Context Managers**:
   - `with torch.no_grad():` Disables gradient tracking (saves memory during validation/testing).
   - `x.detach()`: Returns a new tensor detached from the computation graph.

---

## 4. Anatomy of a PyTorch Training Loop

A complete model training pipeline consists of 5 standard steps:

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 1. Define Model
class MLP(nn.Module):
    def __init__(self, in_features, hidden_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
    def forward(self, x):
        return self.net(x)

model = MLP(10, 32, 2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 2. Training Loop
for epoch in range(num_epochs):
    model.train()  # Sets training mode (enables dropout, batch norm updates)
    for batch_x, batch_y in train_loader:
        # Step A: Forward pass
        predictions = model(batch_x)
        loss = criterion(predictions, batch_y)
        
        # Step B: Zero existing gradients
        optimizer.zero_grad()
        
        # Step C: Backward pass (compute dL/dw)
        loss.backward()
        
        # Step D: Optimizer step (update parameters)
        optimizer.step()
        
    # 3. Validation Loop
    model.eval()  # Sets evaluation mode (disables dropout, freezes batch norm stats)
    with torch.no_grad():
        for val_x, val_y in val_loader:
            val_preds = model(val_x)
            val_loss = criterion(val_preds, val_y)
```

---

## 5. Summary Table: NumPy vs. PyTorch

| Feature | NumPy (`np.ndarray`) | PyTorch (`torch.Tensor`) |
|---|---|---|
| GPU Acceleration | No (CPU only) | Yes (`.to("cuda")`) |
| Automatic Gradients | No (manual derivation) | Yes (`requires_grad=True`, `.backward()`) |
| Deep Learning Layers | No | `torch.nn` module |
| Data Loaders | Custom | `DataLoader`, `Dataset` |


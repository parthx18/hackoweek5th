"""
PyTorch Fundamentals & Training Loop Demo
==========================================
Covers:
1. PyTorch Tensors: creation, shape manipulation, math operations, and device movement.
2. Autograd Engine: computational graphs, backward passes, and gradient tracking.
3. Training Loop: custom nn.Module, Dataset, DataLoader, loss, optimizer, and train/val loop.

Usage:
    python pytorch_basics_and_training.py
"""

import sys
import time
import math
import random

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    print("PyTorch is not installed. Please install it via 'pip install torch'.")
    sys.exit(1)


# ==============================================================================
# Part 1: Tensors & Operations
# ==============================================================================
def demo_tensor_operations():
    """Demonstrate tensor creation, manipulation, and device placement."""
    print("\n" + "=" * 60)
    print(" PART 1: PYTORCH TENSOR FUNDAMENTALS")
    print("=" * 60)

    # 1. Creation
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    b = torch.ones((2, 2), dtype=torch.float32)
    zeros = torch.zeros((2, 3))
    rand = torch.randn((2, 3))

    print(f"Tensor 'a' (shape {list(a.shape)}):\n{a}")
    print(f"Random standard normal tensor (shape {list(rand.shape)}):\n{rand}")

    # 2. Math Operations
    add_res = a + b
    matmul_res = torch.matmul(a, b)
    print(f"\nElementwise Addition (a + b):\n{add_res}")
    print(f"Matrix Multiplication (a @ b):\n{matmul_res}")

    # 3. Shape Manipulations
    reshaped = rand.view(6)
    transposed = rand.t()
    unsqueezed = a.unsqueeze(0)  # adds batch dimension -> (1, 2, 2)
    squeezed = unsqueezed.squeeze(0)  # removes batch dimension -> (2, 2)

    print(f"\nShape manipulation:")
    print(f"  Original: {list(rand.shape)} -> View flat: {list(reshaped.shape)}")
    print(f"  Transposed: {list(transposed.shape)}")
    print(f"  Unsqueeze(0): {list(unsqueezed.shape)} -> Squeeze(0): {list(squeezed.shape)}")

    # 4. Device placement
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a_device = a.to(device)
    print(f"\nDevice availability: using '{device}' (is_cuda={torch.cuda.is_available()})")
    print(f"Tensor on device: {a_device.device}")


# ==============================================================================
# Part 2: Autograd Engine
# ==============================================================================
def demo_autograd():
    """Demonstrate automatic differentiation, computational graph, and gradients."""
    print("\n" + "=" * 60)
    print(" PART 2: THE AUTOGRAD ENGINE (AUTOMATIC DIFFERENTIATION)")
    print("=" * 60)

    # Consider the function: y = 3*x^2 + 2*x + 5
    # Analytical derivative: dy/dx = 6*x + 2
    # At x = 4.0: dy/dx = 6*(4) + 2 = 26.0

    x = torch.tensor(4.0, requires_grad=True)
    y = 3 * (x ** 2) + 2 * x + 5

    print(f"Defined function: y = 3*x^2 + 2*x + 5")
    print(f"Input x: {x.item()} (requires_grad={x.requires_grad})")
    print(f"Evaluated y: {y.item()}")
    print(f"Gradient before backward(): x.grad = {x.grad}")

    # Trigger backpropagation
    y.backward()

    print(f"Gradient after y.backward(): dy/dx = {x.grad.item():.4f}")
    assert math.isclose(x.grad.item(), 26.0, rel_tol=1e-5), "Derivative mismatch!"
    print("[OK] Computed gradient matches exact analytical derivative (26.0)!")

    # Multivariate autograd
    w = torch.tensor([2.0, -1.0], requires_grad=True)
    x_vec = torch.tensor([1.5, 3.0])
    b = torch.tensor(0.5, requires_grad=True)

    # z = w · x + b = 2.0*1.5 + (-1.0)*3.0 + 0.5 = 3.0 - 3.0 + 0.5 = 0.5
    z = torch.dot(w, x_vec) + b
    loss = (z - 2.0) ** 2  # target is 2.0

    loss.backward()
    print(f"\nLinear Model loss autograd:")
    print(f"  w.grad: {w.grad.tolist()} (dL/dw = 2*(z - y_target)*x)")
    print(f"  b.grad: {b.grad.item():.4f} (dL/db = 2*(z - y_target))")

    # Context manager: no_grad
    with torch.no_grad():
        test_z = torch.dot(w, x_vec) + b
        print(f"  Under torch.no_grad(): requires_grad={test_z.requires_grad}")


# ==============================================================================
# Part 3: Dataset, DataLoader & Custom Model
# ==============================================================================
class SyntheticClassificationDataset(Dataset):
    """Generates a synthetic binary classification dataset (two concentric circles)."""
    def __init__(self, n_samples=600, noise=0.1, seed=42):
        random.seed(seed)
        torch.manual_seed(seed)
        
        self.features = []
        self.labels = []
        
        for i in range(n_samples):
            # Class 0: Inner circle (radius ~ 0.5)
            # Class 1: Outer ring (radius ~ 1.5)
            label = 0 if i < n_samples // 2 else 1
            radius = 0.5 if label == 0 else 1.5
            radius += random.gauss(0, noise)
            angle = random.uniform(0, 2 * math.pi)
            
            x1 = radius * math.cos(angle)
            x2 = radius * math.sin(angle)
            
            self.features.append([x1, x2])
            self.labels.append(label)
            
        self.features = torch.tensor(self.features, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class MultilayerPerceptron(nn.Module):
    """A clean, modular Multi-Layer Perceptron (MLP) for classification."""
    def __init__(self, in_features=2, hidden_dim=32, num_classes=2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        return self.network(x)


# ==============================================================================
# Part 4: Training & Validation Loop
# ==============================================================================
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=15, device="cpu"):
    """
    Standard PyTorch training and validation loop with metrics tracking.
    """
    model.to(device)
    print(f"\nTraining {model.__class__.__name__} on {device} for {num_epochs} epochs...")
    print("-" * 65)
    print(f"{'Epoch':<8}{'Train Loss':<14}{'Train Acc (%)':<16}{'Val Loss':<14}{'Val Acc (%)':<12}")
    print("-" * 65)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, num_epochs + 1):
        # 1. Training Phase
        model.train()
        total_train_loss = 0.0
        correct_train = 0
        total_train_samples = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Step 1: Zero Gradients
            optimizer.zero_grad()

            # Step 2: Forward Pass
            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            # Step 3: Backward Pass
            loss.backward()

            # Step 4: Parameter Update
            optimizer.step()

            # Metrics
            total_train_loss += loss.item() * batch_x.size(0)
            preds = torch.argmax(logits, dim=1)
            correct_train += (preds == batch_y).sum().item()
            total_train_samples += batch_x.size(0)

        epoch_train_loss = total_train_loss / total_train_samples
        epoch_train_acc = (correct_train / total_train_samples) * 100.0

        # 2. Validation Phase
        model.eval()
        total_val_loss = 0.0
        correct_val = 0
        total_val_samples = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)

                total_val_loss += loss.item() * batch_x.size(0)
                preds = torch.argmax(logits, dim=1)
                correct_val += (preds == batch_y).sum().item()
                total_val_samples += batch_x.size(0)

        epoch_val_loss = total_val_loss / total_val_samples
        epoch_val_acc = (correct_val / total_val_samples) * 100.0

        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        if epoch % 3 == 0 or epoch == num_epochs:
            print(f"{epoch:<8}{epoch_train_loss:<14.4f}{epoch_train_acc:<16.2f}{epoch_val_loss:<14.4f}{epoch_val_acc:<12.2f}")

    print("-" * 65)
    return history


def main():
    print("=" * 60)
    print(" DEEP LEARNING FOUNDATIONS: PYTORCH & TRAINING LOOPS")
    print("=" * 60)

    # Run demos
    demo_tensor_operations()
    demo_autograd()

    # Part 3 & 4: Training Loop on dataset
    print("\n" + "=" * 60)
    print(" PART 3 & 4: DATASET, MODEL & TRAINING LOOP")
    print("=" * 60)

    dataset = SyntheticClassificationDataset(n_samples=800, noise=0.12, seed=42)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

    print(f"Dataset created: {len(dataset)} samples ({train_size} train, {val_size} validation)")

    # Instantiate model, loss, optimizer
    model = MultilayerPerceptron(in_features=2, hidden_dim=32, num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    history = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=15, device=device)

    final_val_acc = history["val_acc"][-1]
    print(f"\nFinal Validation Accuracy: {final_val_acc:.2f}%")
    if final_val_acc > 85.0:
        print("[OK] Model successfully converged and separated non-linear concentric classes!")
    print("\nPyTorch fundamentals and training loop demonstration complete.")


if __name__ == "__main__":
    main()

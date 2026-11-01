"""
Regularization in Deep Learning: Dropout, BatchNorm & Early Stopping
====================================================================
Covers:
1. Dropout (nn.Dropout): random neuron silencing to prevent co-adaptation.
2. Batch Normalization (nn.BatchNorm1d): normalizing activations across mini-batches.
3. Weight Decay (L2 regularization): penalizing large weights via optimizer.
4. Early Stopping: custom callback that tracks validation loss and halts training
   when performance plateaus, restoring the optimal model checkpoint.

Usage:
    python regularization_and_early_stopping.py
"""

import os
import sys
import copy
import math
import random

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    print("PyTorch is required. Install via 'pip install torch'.")
    sys.exit(1)


# ==============================================================================
# Early Stopping Implementation
# ==============================================================================
class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    Saves and restores the best model parameters.
    """
    def __init__(self, patience=5, min_delta=1e-4, verbose=True):
        """
        Args:
            patience (int): How many epochs to wait after last time validation loss improved.
            min_delta (float): Minimum change in monitored loss to qualify as an improvement.
            verbose (bool): If True, prints status messages.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_weights = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_weights = copy.deepcopy(model.state_dict())
            if self.verbose:
                print(f"  [EarlyStopping] Initial best validation loss: {val_loss:.4f}")
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"  [EarlyStopping] Patience: {self.counter}/{self.patience} (Best: {self.best_loss:.4f}, Current: {val_loss:.4f})")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"  [EarlyStopping] Triggered! Halting training to avoid overfitting.")
        else:
            if self.verbose:
                print(f"  [EarlyStopping] Val loss improved from {self.best_loss:.4f} to {val_loss:.4f}. Checkpoint updated.")
            self.best_loss = val_loss
            self.best_model_weights = copy.deepcopy(model.state_dict())
            self.counter = 0

    def restore_best_weights(self, model):
        """Restores the best weights observed during training."""
        if self.best_model_weights is not None:
            model.load_state_dict(self.best_model_weights)
            if self.verbose:
                print(f"  [EarlyStopping] Restored best model weights (Val Loss: {self.best_loss:.4f})")


# ==============================================================================
# Noisy Overfitting Dataset
# ==============================================================================
class OverfittingDemoDataset(Dataset):
    """
    Generates a small dataset with 20 features (most redundant/noisy)
    specifically designed to highlight the gap between overfitted and regularized models.
    """
    def __init__(self, n_samples=300, n_features=25, noise_ratio=0.25, seed=42):
        random.seed(seed)
        torch.manual_seed(seed)
        
        # 3 informative features, 22 pure noise features
        X = torch.randn(n_samples, n_features)
        # Linear combination of the first 3 features determines the true label
        logits = 1.8 * X[:, 0] - 2.2 * X[:, 1] + 1.5 * X[:, 2]
        prob = 1.0 / (1.0 + torch.exp(-logits))
        y = (prob > 0.5).long()
        
        # Inject label noise to encourage unregularized models to overfit
        n_noisy = int(n_samples * noise_ratio)
        noisy_indices = torch.randperm(n_samples)[:n_noisy]
        y[noisy_indices] = 1 - y[noisy_indices]
        
        self.features = X
        self.labels = y
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ==============================================================================
# Model Architectures: Unregularized vs Regularized
# ==============================================================================
class UnregularizedOverfittingNet(nn.Module):
    """High capacity network without Dropout, BatchNorm, or Weight Decay."""
    def __init__(self, in_features=25, hidden_dim=128, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)


class RegularizedNet(nn.Module):
    """Network equipped with Batch Normalization and Dropout layers."""
    def __init__(self, in_features=25, hidden_dim=128, num_classes=2, dropout_p=0.35):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# ==============================================================================
# Training Runner
# ==============================================================================
def train_experiment(model, train_loader, val_loader, optimizer, criterion, 
                     num_epochs=30, early_stopper=None, desc=""):
    print(f"\n---> Training: {desc}")
    print(f"{'Epoch':<6}{'Train Loss':<12}{'Train Acc (%)':<16}{'Val Loss':<12}{'Val Acc (%)':<12}")
    print("-" * 58)
    
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, num_epochs + 1):
        # 1. Train
        model.train()
        train_loss, train_correct, n_train = 0.0, 0, 0
        for bx, by in train_loader:
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * bx.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == by).sum().item()
            n_train += bx.size(0)

        t_loss = train_loss / n_train
        t_acc = (train_correct / n_train) * 100.0

        # 2. Validate
        model.eval()
        val_loss, val_correct, n_val = 0.0, 0, 0
        with torch.no_grad():
            for bx, by in val_loader:
                logits = model(bx)
                loss = criterion(logits, by)
                val_loss += loss.item() * bx.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == by).sum().item()
                n_val += bx.size(0)

        v_loss = val_loss / n_val
        v_acc = (val_correct / n_val) * 100.0

        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)

        if epoch % 5 == 0 or epoch == num_epochs:
            print(f"{epoch:<6}{t_loss:<12.4f}{t_acc:<16.2f}{v_loss:<12.4f}{v_acc:<12.2f}")

        # Check early stopping
        if early_stopper:
            early_stopper(v_loss, model)
            if early_stopper.early_stop:
                print(f"Early stopping at epoch {epoch}!")
                break

    if early_stopper:
        early_stopper.restore_best_weights(model)

    return history


def main():
    print("=" * 65)
    print(" REGULARIZATION IN DEEP LEARNING: DROPOUT, BATCHNORM & EARLY STOPPING")
    print("=" * 65)

    # 1. Prepare noisy small dataset (150 train samples, 150 validation samples)
    dataset = OverfittingDemoDataset(n_samples=300, n_features=25, noise_ratio=0.20, seed=42)
    train_set, val_set = torch.utils.data.random_split(dataset, [150, 150])

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------------------------
    # Experiment A: Unregularized Model (High Capacity, No Dropout, No BatchNorm)
    # --------------------------------------------------------------------------
    unreg_model = UnregularizedOverfittingNet(in_features=25, hidden_dim=128)
    unreg_opt = optim.Adam(unreg_model.parameters(), lr=0.003, weight_decay=0.0)
    
    unreg_hist = train_experiment(
        unreg_model, train_loader, val_loader, unreg_opt, criterion,
        num_epochs=30, early_stopper=None, desc="Unregularized Model (Susceptible to Overfitting)"
    )

    # --------------------------------------------------------------------------
    # Experiment B: Regularized Model (BatchNorm + Dropout + Weight Decay + Early Stopping)
    # --------------------------------------------------------------------------
    reg_model = RegularizedNet(in_features=25, hidden_dim=128, dropout_p=0.35)
    reg_opt = optim.Adam(reg_model.parameters(), lr=0.003, weight_decay=1e-3)  # L2 regularization
    early_stopper = EarlyStopping(patience=5, min_delta=1e-3, verbose=True)

    reg_hist = train_experiment(
        reg_model, train_loader, val_loader, reg_opt, criterion,
        num_epochs=30, early_stopper=early_stopper, desc="Regularized Model (Dropout + BatchNorm + L2 + Early Stopping)"
    )

    # --------------------------------------------------------------------------
    # Comparison Summary
    # --------------------------------------------------------------------------
    print("\n" + "=" * 65)
    print(" COMPARATIVE SUMMARY")
    print("=" * 65)
    
    unreg_gap = abs(unreg_hist['train_acc'][-1] - unreg_hist['val_acc'][-1])
    reg_gap = abs(reg_hist['train_acc'][-1] - reg_hist['val_acc'][-1])

    print(f"Unregularized Model:")
    print(f"  Final Train Accuracy: {unreg_hist['train_acc'][-1]:.2f}%")
    print(f"  Final Val Accuracy:   {unreg_hist['val_acc'][-1]:.2f}%")
    print(f"  Generalization Gap:   {unreg_gap:.2f}% (Notice high overfitting gap!)")

    print(f"\nRegularized Model:")
    print(f"  Final Train Accuracy: {reg_hist['train_acc'][-1]:.2f}%")
    print(f"  Final Val Accuracy:   {reg_hist['val_acc'][-1]:.2f}%")
    print(f"  Generalization Gap:   {reg_gap:.2f}% (Much tighter generalization bounds!)")
    
    print("\n[OK] Regularization techniques (Dropout, BatchNorm, L2, EarlyStopping) successfully tested.")


if __name__ == "__main__":
    main()

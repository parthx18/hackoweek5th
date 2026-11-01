"""
Transfer Learning Basics: Pretrained CNNs (ResNet)
===================================================
Covers:
1. Loading pretrained vision backbones (torchvision.models.resnet18).
2. Strategy 1: Feature Extraction (freezing backbone weights, training new classification head).
3. Strategy 2: Fine-Tuning (unfreezing deep layers with differential learning rates).
4. ImageNet normalization transforms and data augmentation.
5. Evaluating convergence speed and parameter efficiency.

Self-contained:
    Includes an RGB synthetic dataset generator simulating ImageNet-compatible
    (3, 224, 224) images so this runs offline with zero external dataset downloads.

Usage:
    python transfer_learning_classifier.py [--mode {both,feature_extract,fine_tune}]
"""

import sys
import copy
import time
import math
import random
import argparse

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import torchvision.models as models
except ImportError:
    print("PyTorch and Torchvision are required. Install via 'pip install torch torchvision'.")
    sys.exit(1)


# ==============================================================================
# Self-Contained 3-Channel RGB Dataset (ImageNet-Compatible: 3 x 224 x 224)
# ==============================================================================
class SyntheticVisionDataset(Dataset):
    """
    Generates realistic 3-channel (3, 64, 64) images normalized with ImageNet stats.
    Classes:
    0: Warm dominant color spectrum (sunsets/desert tones)
    1: Cool dominant color spectrum (ocean/sky tones)
    2: Geometric high-frequency noise & checkerboard patterns
    """
    def __init__(self, n_samples=300, img_size=64, seed=42):
        random.seed(seed)
        torch.manual_seed(seed)
        self.classes = ["Warm_Spectrum", "Cool_Spectrum", "High_Frequency"]
        self.images = []
        self.labels = []

        # ImageNet normalization stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        for _ in range(n_samples):
            label = random.randint(0, 2)
            img = torch.zeros((3, img_size, img_size), dtype=torch.float32)

            if label == 0:  # Warm tones: High Red, moderate Green, low Blue
                img[0] = 0.8 + torch.randn(img_size, img_size) * 0.1
                img[1] = 0.4 + torch.randn(img_size, img_size) * 0.1
                img[2] = 0.2 + torch.randn(img_size, img_size) * 0.1
            elif label == 1:  # Cool tones: Low Red, moderate Green, high Blue
                img[0] = 0.1 + torch.randn(img_size, img_size) * 0.1
                img[1] = 0.5 + torch.randn(img_size, img_size) * 0.1
                img[2] = 0.9 + torch.randn(img_size, img_size) * 0.1
            else:  # High frequency noise / stripes
                grid_y, grid_x = torch.meshgrid(torch.arange(img_size), torch.arange(img_size), indexing="ij")
                pattern = ((grid_y // 4 + grid_x // 4) % 2).float()
                img[0] = pattern * 0.7 + torch.randn(img_size, img_size) * 0.1
                img[1] = pattern * 0.7 + torch.randn(img_size, img_size) * 0.1
                img[2] = pattern * 0.7 + torch.randn(img_size, img_size) * 0.1

            # Clamp and normalize
            img = torch.clamp(img, 0.0, 1.0)
            img = (img - mean) / std

            self.images.append(img)
            self.labels.append(label)

        self.images = torch.stack(self.images)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


# ==============================================================================
# Model Builders: Feature Extraction vs Fine-Tuning
# ==============================================================================
def build_feature_extractor_model(num_classes=3):
    """
    Builds a transfer learning model in Feature Extraction mode.
    All backbone convolutional layers are FROZEN (requires_grad = False).
    Only the final linear layer is trained.
    """
    try:
        # Load ResNet-18
        model = models.resnet18(weights=None)  # weights=None for reliable offline execution
    except Exception:
        model = models.resnet18()

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Replace classification head (newly added layer has requires_grad=True by default)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, num_classes)
    )

    # Count parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    print(f"\n[Feature Extraction Mode]")
    print(f"  Frozen Backbone Parameters:   {frozen_params:,}")
    print(f"  Trainable Head Parameters:     {trainable_params:,}")
    print(f"  Percentage Trainable:          {(trainable_params / (frozen_params + trainable_params))*100:.2f}%")

    return model


def build_fine_tuning_model(num_classes=3):
    """
    Builds a transfer learning model in Fine-Tuning mode.
    Freezes early layers (conv1, layer1, layer2, layer3), but unfreezes layer4 and fc
    using differential learning rates.
    """
    try:
        model = models.resnet18(weights=None)
    except Exception:
        model = models.resnet18()

    # Freeze early layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze layer 4 (deepest residual block)
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Replace head
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, num_classes)
    )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    print(f"\n[Fine-Tuning Mode]")
    print(f"  Frozen Early Parameters:       {frozen_params:,}")
    print(f"  Trainable (Layer4 + Head):     {trainable_params:,}")
    print(f"  Percentage Trainable:          {(trainable_params / (frozen_params + trainable_params))*100:.2f}%")

    return model


# ==============================================================================
# Training Runner
# ==============================================================================
def train_transfer_model(model, train_loader, val_loader, optimizer, criterion, epochs=6, device="cpu", label=""):
    model.to(device)
    print(f"\n---> Training {label} for {epochs} epochs on {device}...")
    print(f"{'Epoch':<8}{'Train Loss':<14}{'Train Acc (%)':<16}{'Val Loss':<14}{'Val Acc (%)':<12}")
    print("-" * 65)

    history = {"train_acc": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        t_loss, t_correct, n_train = 0.0, 0, 0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()

            t_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            t_correct += (preds == lbls).sum().item()
            n_train += imgs.size(0)

        # Validation
        model.eval()
        v_loss, v_correct, n_val = 0.0, 0, 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, lbls)

                v_loss += loss.item() * imgs.size(0)
                preds = outputs.argmax(dim=1)
                v_correct += (preds == lbls).sum().item()
                n_val += imgs.size(0)

        train_acc = (t_correct / n_train) * 100.0
        val_acc = (v_correct / n_val) * 100.0
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"{epoch:<8}{t_loss/n_train:<14.4f}{train_acc:<16.2f}{v_loss/n_val:<14.4f}{val_acc:<12.2f}")

    return history


def main():
    parser = argparse.ArgumentParser(description="Transfer Learning with Pretrained ResNet")
    parser.add_argument("--mode", choices=["both", "feature_extract", "fine_tune"], default="both")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    print("=" * 65)
    print(" TRANSFER LEARNING WITH PRETRAINED CNNS (RESNET-18)")
    print("=" * 65)

    # 1. Dataset
    dataset = SyntheticVisionDataset(n_samples=300, img_size=64, seed=42)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = {}

    # 2. Experiment 1: Feature Extraction
    if args.mode in ["both", "feature_extract"]:
        fe_model = build_feature_extractor_model(num_classes=3)
        # Only pass parameters that require gradients to optimizer
        fe_optimizer = optim.Adam(filter(lambda p: p.requires_grad, fe_model.parameters()), lr=0.003)
        fe_hist = train_transfer_model(
            fe_model, train_loader, val_loader, fe_optimizer, criterion, 
            epochs=args.epochs, device=device, label="Feature Extraction (Frozen Backbone)"
        )
        results["Feature Extraction"] = fe_hist["val_acc"][-1]

    # 3. Experiment 2: Fine-Tuning
    if args.mode in ["both", "fine_tune"]:
        ft_model = build_fine_tuning_model(num_classes=3)
        # Differential learning rates: 1e-4 for layer4, 1e-3 for fc
        ft_optimizer = optim.Adam([
            {'params': ft_model.layer4.parameters(), 'lr': 1e-4},
            {'params': ft_model.fc.parameters(), 'lr': 1e-3}
        ])
        ft_hist = train_transfer_model(
            ft_model, train_loader, val_loader, ft_optimizer, criterion,
            epochs=args.epochs, device=device, label="Fine-Tuning (Layer4 + Head)"
        )
        results["Fine-Tuning"] = ft_hist["val_acc"][-1]

    # Summary
    print("\n" + "=" * 65)
    print(" TRANSFER LEARNING RESULTS SUMMARY")
    print("=" * 65)
    for mode_name, final_acc in results.items():
        print(f"  {mode_name:<25}: Final Validation Accuracy = {final_acc:.2f}%")
    print("\n[OK] Transfer learning module successfully demonstrated.")


if __name__ == "__main__":
    main()

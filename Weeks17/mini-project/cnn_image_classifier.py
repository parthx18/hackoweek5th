"""
Convolutional Neural Networks (CNNs) for Image Classification
==============================================================
Covers:
1. 2D Convolution layers (nn.Conv2d), kernels, stride, and padding.
2. Pooling layers (nn.MaxPool2d, nn.AdaptiveAvgPool2d).
3. Complete deep CNN architecture with BatchNorm2d and Dropout.
4. Mathematical dimension calculation through feature extractor layers.
5. Image classification training, top-1 accuracy, and confusion matrix.

Self-contained:
    Can download Fashion-MNIST, or use built-in synthetic geometric shapes dataset
    (circles vs squares vs triangles) for instant offline execution without network access.

Usage:
    python cnn_image_classifier.py [--dataset {fashion,synthetic}]
"""

import sys
import math
import random
import argparse

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    print("PyTorch is required. Install via 'pip install torch'.")
    sys.exit(1)


# ==============================================================================
# Part 1: CNN Spatial Calculations
# ==============================================================================
def compute_conv_output_dim(dim_in, kernel_size, padding=0, stride=1, dilation=1):
    """
    Computes spatial output size: floor((dim_in + 2*padding - dilation*(kernel-1) - 1)/stride) + 1
    """
    return math.floor((dim_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1


def verify_cnn_dimensions(input_shape=(1, 28, 28)):
    """Walk through each layer and compute theoretical dimensions."""
    c, h, w = input_shape
    print("\n--- Theoretical Dimension Flow through CNN ---")
    print(f"Input:                         ({c}, {h}, {w})")
    
    # Block 1: Conv2d(1 -> 16, k=3, p=1, s=1)
    h = compute_conv_output_dim(h, 3, padding=1, stride=1)
    w = compute_conv_output_dim(w, 3, padding=1, stride=1)
    print(f"After Conv1 (k=3, p=1, s=1):   (16, {h}, {w})")
    
    # MaxPool(2, 2)
    h = compute_conv_output_dim(h, 2, padding=0, stride=2)
    w = compute_conv_output_dim(w, 2, padding=0, stride=2)
    print(f"After MaxPool1 (k=2, s=2):     (16, {h}, {w})")
    
    # Block 2: Conv2d(16 -> 32, k=3, p=1, s=1)
    h = compute_conv_output_dim(h, 3, padding=1, stride=1)
    w = compute_conv_output_dim(w, 3, padding=1, stride=1)
    print(f"After Conv2 (k=3, p=1, s=1):   (32, {h}, {w})")
    
    # MaxPool(2, 2)
    h = compute_conv_output_dim(h, 2, padding=0, stride=2)
    w = compute_conv_output_dim(w, 2, padding=0, stride=2)
    print(f"After MaxPool2 (k=2, s=2):     (32, {h}, {w})")
    print(f"Flattened linear input size:   {32 * h * w}\n")


# ==============================================================================
# Part 2: Self-Contained Synthetic Image Dataset (Geometric Shapes)
# ==============================================================================
class SyntheticShapesDataset(Dataset):
    """
    Generates 28x28 grayscale synthetic images of 3 distinct geometric classes:
    Class 0: Circle
    Class 1: Square / Rectangle
    Class 2: Cross / Plus sign
    Works completely offline with 0 internet dependency.
    """
    def __init__(self, n_samples=600, img_size=28, seed=42):
        random.seed(seed)
        torch.manual_seed(seed)
        self.classes = ["Circle", "Square", "Cross"]
        self.images = []
        self.labels = []
        
        for _ in range(n_samples):
            label = random.randint(0, 2)
            canvas = torch.zeros((1, img_size, img_size), dtype=torch.float32)
            # Add slight background noise
            canvas += torch.randn((1, img_size, img_size)) * 0.05
            
            cx = random.randint(10, 18)
            cy = random.randint(10, 18)
            
            if label == 0:  # Circle
                radius = random.randint(4, 7)
                for r in range(img_size):
                    for c in range(img_size):
                        if (r - cy)**2 + (c - cx)**2 <= radius**2:
                            canvas[0, r, c] = 1.0 + random.uniform(-0.1, 0.1)
                            
            elif label == 1:  # Square
                half_side = random.randint(4, 6)
                r_min, r_max = max(0, cy - half_side), min(img_size, cy + half_side)
                c_min, c_max = max(0, cx - half_side), min(img_size, cx + half_side)
                canvas[0, r_min:r_max, c_min:c_max] = 1.0 + random.uniform(-0.1, 0.1)
                
            elif label == 2:  # Cross
                arm = random.randint(4, 7)
                width = 2
                canvas[0, max(0, cy - arm):min(img_size, cy + arm), max(0, cx - width):min(img_size, cx + width)] = 1.0
                canvas[0, max(0, cy - width):min(img_size, cy + width), max(0, cx - arm):min(img_size, cx + arm)] = 1.0

            # Clamp between 0.0 and 1.0
            canvas = torch.clamp(canvas, 0.0, 1.0)
            self.images.append(canvas)
            self.labels.append(label)
            
        self.images = torch.stack(self.images)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


# ==============================================================================
# Part 3: CNN Architecture
# ==============================================================================
class ConvNetClassifier(nn.Module):
    """
    A 2D Convolutional Neural Network with 2 Convolution blocks,
    Batch Normalization, Max Pooling, and a Dropout-regularized Dense Classifier.
    """
    def __init__(self, in_channels=1, num_classes=3):
        super().__init__()
        
        # Feature Extraction Backbone
        self.features = nn.Sequential(
            # Block 1: 1 -> 16 channels
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 28x28 -> 14x14
            
            # Block 2: 16 -> 32 channels
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 14x14 -> 7x7
        )
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        features = self.features(x)
        logits = self.classifier(features)
        return logits


# ==============================================================================
# Part 4: Training & Evaluation with Confusion Matrix
# ==============================================================================
def train_and_evaluate_cnn(model, train_loader, val_loader, class_names, num_epochs=8, device="cpu"):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002)

    print(f"\nTraining CNN Classifier on {device} ({len(class_names)} classes)...")
    print(f"{'Epoch':<8}{'Train Loss':<14}{'Train Acc (%)':<16}{'Val Loss':<14}{'Val Acc (%)':<12}")
    print("-" * 65)

    for epoch in range(1, num_epochs + 1):
        # Training
        model.train()
        train_loss, train_correct, total_train = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            total_train += images.size(0)

        t_loss = train_loss / total_train
        t_acc = (train_correct / total_train) * 100.0

        # Validation
        model.eval()
        val_loss, val_correct, total_val = 0.0, 0, 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                total_val += images.size(0)

                all_preds.extend(preds.cpu().tolist())
                all_targets.extend(labels.cpu().tolist())

        v_loss = val_loss / total_val
        v_acc = (val_correct / total_val) * 100.0

        print(f"{epoch:<8}{t_loss:<14.4f}{t_acc:<16.2f}{v_loss:<14.4f}{v_acc:<12.2f}")

    # Compute Confusion Matrix
    num_classes = len(class_names)
    cm = [[0] * num_classes for _ in range(num_classes)]
    for target, pred in zip(all_targets, all_preds):
        cm[target][pred] += 1

    print("\n" + "=" * 55)
    print(" CONFUSION MATRIX (Rows: True Class | Columns: Predicted)")
    print("=" * 55)
    header = "          " + "".join([f"{cls:>12}" for cls in class_names])
    print(header)
    for i, row in enumerate(cm):
        row_str = f"{class_names[i]:<10}" + "".join([f"{count:>12}" for count in row])
        print(row_str)

    return v_acc


def main():
    parser = argparse.ArgumentParser(description="CNN Image Classification Demo")
    parser.add_argument("--dataset", choices=["synthetic", "fashion"], default="synthetic",
                        help="Dataset to use: 'synthetic' (default, offline geometric shapes) or 'fashion' (Fashion-MNIST)")
    parser.add_argument("--epochs", type=int, default=8, help="Number of epochs")
    args = parser.parse_args()

    print("=" * 65)
    print(" CONVOLUTIONAL NEURAL NETWORKS (CNNS) IMAGE CLASSIFIER")
    print("=" * 65)

    # 1. Dimension math verification
    verify_cnn_dimensions(input_shape=(1, 28, 28))

    # 2. Dataset loading
    if args.dataset == "fashion":
        try:
            from torchvision import datasets, transforms
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
            train_data = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
            test_data = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
            train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
            val_loader = DataLoader(test_data, batch_size=64, shuffle=False)
            class_names = train_data.classes
            num_classes = 10
            print("Loaded Fashion-MNIST dataset.")
        except Exception as e:
            print(f"Notice: FashionMNIST download was unavailable ({e}). Falling back to synthetic geometric dataset.")
            args.dataset = "synthetic"

    if args.dataset == "synthetic":
        dataset = SyntheticShapesDataset(n_samples=600, img_size=28, seed=42)
        train_len = int(0.8 * len(dataset))
        val_len = len(dataset) - train_len
        train_data, val_data = torch.utils.data.random_split(dataset, [train_len, val_len])
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
        class_names = dataset.classes
        num_classes = len(class_names)
        print(f"Generated Synthetic Shapes Dataset: {len(dataset)} samples across classes: {class_names}")

    # 3. Model instantiation
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ConvNetClassifier(in_channels=1, num_classes=num_classes)

    # 4. Training and evaluation
    acc = train_and_evaluate_cnn(model, train_loader, val_loader, class_names, num_epochs=args.epochs, device=device)
    print(f"\nFinal Test/Val Accuracy: {acc:.2f}%")
    print("[OK] CNN image classification module executed successfully!")


if __name__ == "__main__":
    main()

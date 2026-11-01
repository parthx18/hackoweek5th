# Transfer Learning with Pretrained CNNs

Training modern deep vision architectures from scratch requires hundreds of thousands of labeled images and days of GPU computation. **Transfer Learning** is the technique of taking a model trained on a massive benchmark dataset (such as ImageNet with 1.2+ million images across 1,000 classes) and adapting it to a new target task.

---

## 1. Why Transfer Learning Works

Deep convolutional networks learn hierarchical representations:
- **Early Layers (Low-Level)**: Learn generic visual primitives such as oriented edges, color blobs, corners, and texture gradients. These features are universal across almost all image domains.
- **Middle Layers (Mid-Level)**: Learn motifs, contours, repetitive textures, and parts of objects (e.g., eyes, wheels, leaf veins).
- **Late Layers (High-Level)**: Learn domain-specific semantic concepts (e.g., specific dog breeds, vehicles, cell types).

By reusing the low- and mid-level feature representations, we can achieve high performance on new datasets with only a few hundred or thousand examples.

```
┌────────────────────────────────────────────────────────┐
│ Early Conv Layers (Edges, Textures)    ◄── FROZEN      │
├────────────────────────────────────────────────────────┤
│ Mid Conv Layers (Motifs, Parts)        ◄── FROZEN / LR │
├────────────────────────────────────────────────────────┤
│ Late Conv Layers (High-level semantics)◄── OPTIONAL FT │
├────────────────────────────────────────────────────────┤
│ Classification Head (Linear / MLP)     ◄── TRAINABLE   │
└────────────────────────────────────────────────────────┘
```

---

## 2. The Two Transfer Learning Strategies

### Strategy A: Feature Extraction (Linear Probing)
- **Concept**: Freeze the entire pretrained feature extractor backbone. Only initialize and train a new final classification layer.
- **Implementation**:
  ```python
  import torchvision.models as models
  import torch.nn as nn

  # 1. Load pretrained model
  model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

  # 2. Freeze all backbone parameters
  for param in model.parameters():
      param.requires_grad = False

  # 3. Replace final fully connected layer (unfrozen by default)
  num_ftrs = model.fc.in_features
  model.fc = nn.Linear(num_ftrs, num_target_classes)

  # 4. Only train parameters with requires_grad=True
  optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
  ```
- **When to use**: Target dataset is small and visually similar to the source dataset (ImageNet).

### Strategy B: Fine-Tuning
- **Concept**: Unfreeze the top layers (or all layers) of the backbone and train with a very small learning rate, allowing the filters to adapt to target domain specifics.
- **Differential Learning Rates**:
  ```python
  optimizer = torch.optim.Adam([
      {'params': model.layer4.parameters(), 'lr': 1e-5},  # Low LR for backbone
      {'params': model.fc.parameters(), 'lr': 1e-3}       # Higher LR for new head
  ])
  ```
- **When to use**: Target dataset is moderately large, or the domain differs significantly from ImageNet (e.g., X-ray / satellite imagery).

---

## 3. Data Preprocessing & ImageNet Norms

Pretrained PyTorch vision models expect images normalized with the exact mean and standard deviation of ImageNet:

```python
from torchvision import transforms

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]),
}
```

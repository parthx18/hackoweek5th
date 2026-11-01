# Convolutional Neural Networks (CNNs) & Computer Vision

Convolutional Neural Networks are the primary architecture for grid-structured data like images, medical scans, and spectrograms. Unlike standard MLPs that flatten inputs and ignore spatial topology, CNNs use localized kernel filters with shared parameters.

---

## 1. Why CNNs for Computer Vision?

1. **Parameter Sharing**: The same filter weights slide across the entire image. A feature detector (e.g., vertical edge) learned in the top-left also works in the bottom-right.
2. **Translation Invariance / Equivariance**: If an object shifts in an image, its feature map representation shifts accordingly.
3. **Local Receptive Fields**: Pixels close to each other are spatially correlated. Neurons connect only to local neighborhoods.

---

## 2. The 2D Convolution Operation

Given an input tensor $X \in \mathbb{R}^{H \times W \times C_{in}}$ and a filter bank $K \in \mathbb{R}^{K_h \times K_w \times C_{in} \times C_{out}}$, the output feature map $Y$ at spatial location $(i, j)$ for filter channel $k$ is:

$$Y(i, j, k) = b_k + \sum_{c=1}^{C_{in}} \sum_{m=-M}^M \sum_{n=-N}^N X(i \cdot S + m, j \cdot S + n, c) \cdot K(m, n, c, k)$$

```
Input (5x5)          Filter (3x3)         Output (3x3)
┌───┬───┬───┬───┬───┐    ┌───┬───┬───┐        ┌───┬───┬───┐
│ 1 │ 0 │ 1 │ 0 │ 0 │    │ 1 │ 0 │ 1 │        │ 4 │ 3 │ 4 │
├───┼───┼───┼───┼───┤    ├───┼───┼───┤        ├───┼───┼───┤
│ 0 │ 1 │ 1 │ 1 │ 0 │ *  │ 0 │ 1 │ 0 │   =    │ 2 │ 4 │ 3 │
├───┼───┼───┼───┼───┤    ├───┼───┼───┤        ├───┼───┼───┤
│ 1 │ 1 │ 0 │ 1 │ 1 │    │ 1 │ 0 │ 1 │        │ 3 │ 3 │ 4 │
└───┴───┴───┴───┴───┘    └───┴───┴───┘        └───┴───┴───┘
```

### Spatial Dimension Formula
Given input dimension $W_{in}$, kernel size $K$, padding $P$, stride $S$, and dilation $D$:
$$W_{out} = \left\lfloor \frac{W_{in} + 2P - D(K - 1) - 1}{S} \right\rfloor + 1$$

When $D=1$ (standard convolution):
$$W_{out} = \left\lfloor \frac{W_{in} + 2P - K}{S} \right\rfloor + 1$$

- **Valid Padding ($P=0$)**: No padding; output size shrinks.
- **Same Padding ($P=\frac{K-1}{2}$ for odd $K$, with $S=1$)**: Output dimensions match input dimensions.

---

## 3. Pooling Layers

Pooling downsamples feature maps, reducing computational load and increasing the receptive field.

- **Max Pooling (`nn.MaxPool2d(kernel_size=2, stride=2)`)**: Extracts the maximum value in each window. Preserves dominant activations.
- **Average Pooling (`nn.AvgPool2d`)**: Calculates the mean in each window. Smooths activations.
- **Global Average Pooling (`nn.AdaptiveAvgPool2d((1, 1))`)**: Reduces an entire feature map $H \times W$ to a single average value per channel, commonly used before the final classification layer.

---

## 4. Standard CNN Architecture Pattern

A typical modern classification CNN follows a repeating pattern:

```
Input (3 x 32 x 32)
   │
   ▼
[Conv2d (3 -> 32, k=3, p=1)] -> [BatchNorm2d] -> [ReLU]
   │
   ▼
[Conv2d (32 -> 32, k=3, p=1)] -> [BatchNorm2d] -> [ReLU] -> [MaxPool2d(2, 2)]
   │ (Output: 32 x 16 x 16)
   ▼
[Conv2d (32 -> 64, k=3, p=1)] -> [BatchNorm2d] -> [ReLU] -> [MaxPool2d(2, 2)]
   │ (Output: 64 x 8 x 8)
   ▼
[AdaptiveAvgPool2d(1, 1)] -> [Flatten]
   │ (Output: 64 vector)
   ▼
[Linear(64, 128)] -> [ReLU] -> [Dropout(0.5)] -> [Linear(128, num_classes)]
   │
   ▼
Softmax Probabilities / CrossEntropyLoss
```


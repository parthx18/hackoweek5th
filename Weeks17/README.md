# Week 17 — Deep Learning Foundations: PyTorch, Regularization, CNNs & Sequence Modeling

Welcome to **Week 17** of the practical deep learning curriculum! This week covers the fundamental building blocks of modern deep learning using **PyTorch**: from raw tensors and automatic differentiation to convolutional vision models and recurrent sequence architectures.

---
## 🎯 Topics Covered

### 1. PyTorch Fundamentals & Training Loop
- **Tensors**: Creation, dimensionalities, data types, slicing, operations, and device placement (`CPU` / `CUDA`).
- **Autograd Engine**: Reverse-mode automatic differentiation, computation graphs, leaf vs non-leaf tensors, `.backward()`, gradient tracking, and `torch.no_grad()`.
- **Training Pipeline**: Custom `nn.Module`, `torch.utils.data.Dataset` and `DataLoader`, loss functions, and writing a clean, modular training/validation loop.

### 2. Regularization in Deep Learning
- **Dropout (`nn.Dropout`)**: Preventing co-adaptation of features during training.
- **Batch Normalization (`nn.BatchNorm1d`, `nn.BatchNorm2d`)**: Stabilizing layer activations, accelerating convergence, and reducing internal covariate shift.
- **Weight Decay ($L_2$ Regularization)**: Penalizing large weights directly through the optimizer.
- **Early Stopping**: Monitoring validation metrics with configurable patience and min_delta, restoring the best model checkpoint.

### 3. Convolutional Neural Networks (CNNs)
- **Convolutions (`nn.Conv2d`)**: Kernel operations, receptive fields, stride, padding, and dilation.
- **Pooling (`nn.MaxPool2d`, `nn.AvgPool2d`)**: Downsampling spatial dimensions while retaining invariant features.
- **Image Classification**: Building and training an end-to-end CNN for image classification with classification reports and confusion matrices.

### 4. Sequence Modeling (RNNs, LSTMs, GRUs)
- **Vanilla RNN (`nn.RNN`)**: Recurrence formulation and the vanishing/exploding gradient problem.
- **LSTM (`nn.LSTM`)**: Forget, input, and output gates protecting long-term memory cells.
- **GRU (`nn.GRU`)**: Parameter-efficient reset and update gating.
- **Sequential Data Processing**: Handling variable lengths, batch-first dimensions, and comparative benchmarking across all three architectures.

---

## 🗂️ Directory Structure

```
Weeks17/
├── README.md
├── notes/
│   ├── 01-pytorch-fundamentals-autograd.md
│   ├── 02-regularization-techniques.md
│   ├── 03-cnns-and-computer-vision.md
│   └── 04-rnns-lstms-grus-sequence-modeling.md
└── mini-project/
    ├── requirements.txt
    ├── pytorch_basics_and_training.py       # Tensors, autograd, custom training loop
    ├── regularization_and_early_stopping.py  # Dropout, BatchNorm, L2, EarlyStopping demo
    ├── cnn_image_classifier.py               # Conv2d, pooling, image classification
    └── sequence_modeling_rnn_lstm.py         # RNN vs LSTM vs GRU comparison
```

---

## 🚀 Running the Mini-Projects

### Setup Environment
```bash
cd Weeks17/mini-project
pip install -r requirements.txt
```

### 1. PyTorch Basics & Training Loop
```bash
python pytorch_basics_and_training.py
```
*Demonstrates tensor manipulation, autograd gradient tracking, and trains an MLP classifier on synthetic data.*

### 2. Regularization & Early Stopping
```bash
python regularization_and_early_stopping.py
```
*Trains an unregularized overfitted model vs a regularized model (Dropout + BatchNorm + Weight Decay) with Early Stopping.*

### 3. CNN Image Classifier
```bash
python cnn_image_classifier.py
```
*Builds a multi-layer ConvNet with batch norm and dropout, trains on image data, and evaluates accuracy.*

### 4. Sequence Modeling Benchmark (RNN vs LSTM vs GRU)
```bash
python sequence_modeling_rnn_lstm.py
```
*Trains RNN, LSTM, and GRU on the same sequential task, logging convergence, time-per-epoch, and parameter counts.*


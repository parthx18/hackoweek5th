"""
Sequence Modeling Basics: RNN, LSTM & GRU Comparison
======================================================
Covers:
1. Vanilla Recurrent Neural Networks (nn.RNN) and recurrent cell mechanics.
2. Long Short-Term Memory networks (nn.LSTM) with forget/input/output gates.
3. Gated Recurrent Units (nn.GRU) with reset and update gates.
4. Sequential data handling (batch_first=True, sequence slicing, hidden states).
5. Head-to-head empirical comparison: parameters, training duration, and accuracy.

Task:
    Sequential Waveform Frequency Classification:
    Given a time sequence of length T, classify whether the underlying signal
    has a Low, Medium, or High frequency component with injected noise.

Usage:
    python sequence_modeling_rnn_lstm.py
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
    print("PyTorch is required. Install via 'pip install torch'.")
    sys.exit(1)


# ==============================================================================
# Sequential Dataset: Multi-Frequency Waveforms
# ==============================================================================
class SequenceWaveformDataset(Dataset):
    """
    Generates synthetic time series sequences of shape (seq_len, 1).
    Classes:
    0: Low frequency sine wave
    1: Medium frequency sine wave
    2: High frequency sine wave
    All with random phase shift and additive Gaussian noise.
    """
    def __init__(self, n_samples=600, seq_len=40, seed=42):
        random.seed(seed)
        torch.manual_seed(seed)
        self.seq_len = seq_len
        self.sequences = []
        self.labels = []
        
        freq_map = {0: 1.0, 1: 3.5, 2: 7.0}
        
        for _ in range(n_samples):
            label = random.randint(0, 2)
            base_freq = freq_map[label]
            phase = random.uniform(0, 2 * math.pi)
            
            # Generate sequential points
            t = torch.linspace(0, 2 * math.pi, seq_len)
            signal = torch.sin(base_freq * t + phase)
            # Add noise
            noise = torch.randn(seq_len) * 0.15
            x = (signal + noise).unsqueeze(-1)  # shape: (seq_len, 1)
            
            self.sequences.append(x)
            self.labels.append(label)
            
        self.sequences = torch.stack(self.sequences)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


# ==============================================================================
# Modular Sequence Models: RNN, LSTM, and GRU
# ==============================================================================
class GenericSequenceClassifier(nn.Module):
    """
    Unified wrapper supporting 'rnn', 'lstm', or 'gru' backbones.
    """
    def __init__(self, rnn_type="lstm", input_dim=1, hidden_dim=32, num_layers=1, num_classes=3):
        super().__init__()
        self.rnn_type = rnn_type.lower()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        if self.rnn_type == "rnn":
            self.recurrent = nn.RNN(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        elif self.rnn_type == "lstm":
            self.recurrent = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        elif self.rnn_type == "gru":
            self.recurrent = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        else:
            raise ValueError(f"Unknown rnn_type: {rnn_type}")

        # Classification Head using the final hidden representation
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        if self.rnn_type == "lstm":
            # LSTM returns: out, (h_n, c_n)
            out, (h_n, c_n) = self.recurrent(x)
        else:
            # RNN and GRU return: out, h_n
            out, h_n = self.recurrent(x)

        # out shape: (batch_size, seq_len, hidden_dim)
        # Extract the representation at the final time step: out[:, -1, :]
        last_step = out[:, -1, :]
        logits = self.classifier(last_step)
        return logits


def count_parameters(model):
    """Counts total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ==============================================================================
# Benchmark Runner
# ==============================================================================
def train_and_benchmark(rnn_type, train_loader, val_loader, epochs=12, device="cpu"):
    model = GenericSequenceClassifier(rnn_type=rnn_type, input_dim=1, hidden_dim=32, num_layers=1, num_classes=3).to(device)
    num_params = count_parameters(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    start_time = time.time()
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        for seqs, labels in train_loader:
            seqs, labels = seqs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(seqs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for seqs, labels in val_loader:
                seqs, labels = seqs.to(device), labels.to(device)
                outputs = model(seqs)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = (correct / total) * 100.0
        if acc > best_val_acc:
            best_val_acc = acc

    elapsed = time.time() - start_time
    return {
        "model": rnn_type.upper(),
        "params": num_params,
        "elapsed_sec": elapsed,
        "best_val_acc": best_val_acc
    }


def main():
    print("=" * 65)
    print(" SEQUENCE MODELING BASICS: RNN vs LSTM vs GRU BENCHMARK")
    print("=" * 65)

    # 1. Dataset Generation
    seq_len = 40
    dataset = SequenceWaveformDataset(n_samples=750, seq_len=seq_len, seed=42)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

    print(f"Generated sequential dataset: {len(dataset)} samples of length {seq_len}")
    print(f"Classes: 0 (Low Freq), 1 (Medium Freq), 2 (High Freq)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Executing on: {device}\n")

    # 2. Run benchmarks for all 3 architectures
    results = []
    for model_name in ["rnn", "lstm", "gru"]:
        print(f"Training {model_name.upper()}...")
        res = train_and_benchmark(model_name, train_loader, val_loader, epochs=12, device=device)
        results.append(res)

    # 3. Print Comparison Table
    print("\n" + "=" * 65)
    print(f"{'Model Architecture':<22}{'Params':<12}{'Time (sec)':<14}{'Best Accuracy (%)':<15}")
    print("=" * 65)
    for r in results:
        print(f"{r['model']:<22}{r['params']:<12}{r['elapsed_sec']:<14.2f}{r['best_val_acc']:<15.2f}")
    print("-" * 65)

    print("\nKey Insights:")
    print("1. Vanilla RNN has the fewest parameters but struggles with longer temporal dependencies.")
    print("2. LSTM has 4x gate parameters (Forget, Input, Cell, Output) providing stable gradient flow.")
    print("3. GRU reduces LSTM's gate count to 2 (Reset, Update), achieving comparable accuracy with fewer parameters.")
    print("\n[OK] Sequence modeling benchmark completed successfully!")


if __name__ == "__main__":
    main()

# Sequence Modeling: RNNs, LSTMs & GRUs

Sequential data (text, speech, time series, audio) exhibits temporal dependencies where token $x_t$ depends on preceding tokens $x_1, \dots, x_{t-1}$. Feedforward networks cannot easily model such dependencies because they assume independent and identically distributed (i.i.d.) samples.

---

## 1. Vanilla Recurrent Neural Networks (RNNs)

A standard RNN maintains a hidden state vector $h_t$ that acts as memory, updated at each time step $t$:

$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$
$$y_t = W_{hy} h_t + b_y$$

```
    y_1         y_2         y_3
     ▲           ▲           ▲
     │           │           │
   ┌───┐       ┌───┐       ┌───┐
   │h_1├──────►│h_2├──────►│h_3│
   └───┘       └───┘       └───┘
     ▲           ▲           ▲
     │           │           │
    x_1         x_2         x_3
```

### The Vanishing & Exploding Gradient Problem
When training via **Backpropagation Through Time (BPTT)**, gradients propagate through time steps:
$$\frac{\partial \mathcal{L}}{\partial h_1} = \frac{\partial \mathcal{L}}{\partial h_T} \prod_{t=2}^T \frac{\partial h_t}{\partial h_{t-1}}$$

Because $\frac{\partial h_t}{\partial h_{t-1}} = \text{diag}(1 - h_t^2) W_{hh}^T$:
- If the eigenvalues of $W_{hh} < 1$, gradients decay exponentially to $0$ (vanishing gradient $\to$ cannot learn long-term dependencies).
- If the eigenvalues of $W_{hh} > 1$, gradients explode exponentially (exploding gradient $\to$ requires gradient clipping).

---

## 2. Long Short-Term Memory (LSTM)

Hochreiter & Schmidhuber (1997) resolved the vanishing gradient issue by introducing an additive **cell state** $C_t$ governed by three non-linear gates:

1. **Forget Gate ($f_t$)**: Decides what information to discard from the previous cell state:
   $$f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)$$
2. **Input Gate ($i_t$) & Candidate Cell ($\tilde{C}_t$)**: Determines what new information to store:
   $$i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)$$
   $$\tilde{C}_t = \tanh(W_c [h_{t-1}, x_t] + b_c)$$
3. **Cell State Update ($C_t$)**: Additive linear combination protecting gradients:
   $$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$
4. **Output Gate ($o_t$) & Hidden State ($h_t$)**: Filters the cell state to produce the hidden state:
   $$o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)$$
   $$h_t = o_t \odot \tanh(C_t)$$

Because $\frac{\partial C_t}{\partial C_{t-1}} \approx f_t$, when $f_t \approx 1$, gradients flow backwards through time without exponential decay!

---

## 3. Gated Recurrent Unit (GRU)

Cho et al. (2014) proposed the GRU, which merges the cell state and hidden state, using only two gates:

1. **Reset Gate ($r_t$)**: How much past memory to consider when computing candidate state:
   $$r_t = \sigma(W_r [h_{t-1}, x_t] + b_r)$$
2. **Update Gate ($z_t$)**: Controls how much of the old state to carry forward:
   $$z_t = \sigma(W_z [h_{t-1}, x_t] + b_z)$$
3. **Candidate Hidden State ($\tilde{h}_t$)**:
   $$\tilde{h}_t = \tanh(W_h [r_t \odot h_{t-1}, x_t] + b_h)$$
4. **Final Hidden State ($h_t$)**: Linear interpolation:
   $$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

---

## 4. Architecture Comparison

| Dimension | Vanilla RNN | LSTM | GRU |
|---|---|---|---|
| **Gate Count** | 0 | 3 (Forget, Input, Output) | 2 (Reset, Update) |
| **Internal States** | 1 ($h_t$) | 2 ($h_t$, $C_t$) | 1 ($h_t$) |
| **Parameters per cell** | $d_h^2 + d_x d_h + d_h$ | $4 \times (d_h^2 + d_x d_h + d_h)$ | $3 \times (d_h^2 + d_x d_h + d_h)$ |
| **Training Speed** | Fastest | Slower | Faster than LSTM |
| **Long-range Retention** | Poor (< 10 steps) | Excellent (100+ steps) | Excellent |
| **Typical Use Cases** | Simple Markovian signals | Complex sequences, speech, machine translation | Efficient sequential modeling, time series |

---

## 5. Sequence Dimensions in PyTorch

In PyTorch, recurrent layers support `batch_first=True`:
- **Input Tensor**: `(batch_size, sequence_length, input_features)`
- **Hidden State Tensor**: `(num_layers * num_directions, batch_size, hidden_dim)`
- **Output Tensor**: `(batch_size, sequence_length, hidden_dim * num_directions)`
  - To classify the entire sequence, use the last time step: `out[:, -1, :]`.


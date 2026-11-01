# Hugging Face Transformers Basics & NLP Architectures

Introduced in the paper *"Attention Is All You Need"* (Vaswani et al., 2017), the **Transformer** architecture eliminated recurrence in favor of **Scaled Dot-Product Self-Attention**, enabling massive parallelization during training. The **Hugging Face `transformers`** library has become the industry standard framework for utilizing pretrained transformer models.

---

## 1. The Core Transformer Mechanism: Scaled Dot-Product Attention

Given an input matrix of vectors, the model computes three projection matrices for each token:
- **Query ($Q$)**: What the current token is seeking.
- **Key ($K$)**: What information the token offers.
- **Value ($V$)**: The actual content representation of the token.

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

```
 Queries (Q) ──┐
               ├──► MatMul ──► Scale (1/√d_k) ──► Softmax ──► MatMul ──► Attention Output
 Keys (K) ────┘                                                 ▲
                                                                │
 Values (V) ────────────────────────────────────────────────────┘
```

- **$\sqrt{d_k}$ Scaling**: For large embedding dimensions, dot products grow large, pushing softmax into regions with vanishingly small gradients. Scaling stabilizes the gradients.
- **Multi-Head Attention (MHA)**: Runs the attention mechanism $h$ times in parallel with distinct learned projections, allowing the model to jointly attend to information at different positions and representation subspaces.

---

## 2. Tokenization: Bridging Raw Text & Tensors

Transformers process discrete integer token IDs rather than raw character strings. Modern tokenizers use **Subword Tokenization** (WordPiece, Byte-Pair Encoding, SentencePiece) to handle out-of-vocabulary words.

```
"unbelievable" ──► ["un", "##believ", "##able"] ──► [4895, 29014, 3085]
```

### Key Tokenizer Outputs
When passing text to `tokenizer(text, return_tensors="pt")`:
1. `input_ids`: Tensor of vocabulary indices representing subwords.
2. `attention_mask`: Binary tensor ($1$ for real tokens, $0$ for padding tokens) instructing attention layers to ignore padding.
3. `token_type_ids`: Distinguishes sequence A from sequence B in sentence-pair tasks.

---

## 3. Hugging Face Abstraction Hierarchy

Hugging Face provides three main levels of abstraction:

### Level 1: `pipeline` (High-Level Task API)
Zero-configuration inference pipeline that automatically handles tokenization, model execution, post-processing, and decoding:
```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier("I absolutely love studying deep learning!")
# [{'label': 'POSITIVE', 'score': 0.9998}]
```

### Level 2: `AutoTokenizer` & `AutoModel` (Middle-Level)
Direct access to token representations, attention masks, logits, and loss:
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

inputs = tokenizer("The movie was fantastic!", return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)
```

### Level 3: `Trainer` & PyTorch Native Loops (Training API)
Enables fine-tuning transformer weights on custom datasets using native PyTorch training loops or the `transformers.Trainer` class.

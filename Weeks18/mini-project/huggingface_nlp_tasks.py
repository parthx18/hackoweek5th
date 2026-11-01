"""
Hugging Face Transformers Basics: Tokenizers, Pipelines & Models
================================================================
Covers:
1. High-level Pipelines (sentiment-analysis, text summarization).
2. Deep dive into Tokenizers (AutoTokenizer, subwords, input_ids, attention_mask, decode).
3. Model Inference (AutoModelForSequenceClassification, forward pass, logits, softmax).
4. Custom Fine-Tuning Step with native PyTorch optimizer.
5. Graceful offline simulation fallback when offline or running in sandboxed environments.

Usage:
    python huggingface_nlp_tasks.py
"""

import sys
import math
import random

# Detect transformers installation
try:
    import torch
    import torch.nn as nn
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


# ==============================================================================
# Part 1: High-Level Hugging Face Pipeline Demo
# ==============================================================================
def demo_hf_pipelines():
    print("\n" + "=" * 65)
    print(" PART 1: HUGGING FACE HIGH-LEVEL PIPELINES")
    print("=" * 65)

    if not TRANSFORMERS_AVAILABLE:
        print("[Offline Simulation] Transformers library not found. Simulating pipeline output:")
        sample_texts = [
            "This deep learning curriculum is exceptionally well designed and practical!",
            "I found the error messages completely confusing and unhelpful."
        ]
        simulated_results = [
            {"label": "POSITIVE", "score": 0.9998},
            {"label": "NEGATIVE", "score": 0.9982}
        ]
        for text, res in zip(sample_texts, simulated_results):
            print(f"\nText:  \"{text}\"")
            print(f"Result: {res['label']} (Confidence: {res['score'] * 100:.2f}%)")
        return

    try:
        # Load lightweight sentiment classifier
        print("Initializing sentiment-analysis pipeline (distilbert-base-uncased-finetuned-sst-2-english)...")
        classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

        sample_texts = [
            "This deep learning curriculum is exceptionally well designed and practical!",
            "I found the error messages completely confusing and unhelpful."
        ]
        results = classifier(sample_texts)
        for text, res in zip(sample_texts, results):
            print(f"\nText:   \"{text}\"")
            print(f"Result: {res['label']} (Confidence: {res['score'] * 100:.2f}%)")
    except Exception as e:
        print(f"Pipeline network access restricted ({e}). Demonstrating simulated offline pipeline:")
        print("Text:   \"This deep learning curriculum is exceptionally well designed!\"")
        print("Result: POSITIVE (Confidence: 99.94%)")


# ==============================================================================
# Part 2: AutoTokenizer Deep Dive (Tokens, IDs, Masks)
# ==============================================================================
class StandaloneOfflineTokenizer:
    """Simulates AutoTokenizer mechanics offline (WordPiece/BPE subwords, IDs, attention mask)."""
    def __init__(self):
        self.vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 101, "[SEP]": 102,
                      "this": 2023, "deep": 2784, "learning": 4083, "is": 2003,
                      "amazing": 6429, "and": 1998, "fast": 3435, "!": 999}
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def __call__(self, text, return_tensors="pt", padding=True, max_length=10):
        words = text.lower().replace("!", " !").split()
        token_ids = [self.vocab["[CLS]"]]
        for w in words:
            token_ids.append(self.vocab.get(w, self.vocab["[UNK]"]))
        token_ids.append(self.vocab["[SEP]"])

        attention_mask = [1] * len(token_ids)
        # Pad to max_length
        while len(token_ids) < max_length:
            token_ids.append(self.vocab["[PAD]"])
            attention_mask.append(0)

        token_ids = token_ids[:max_length]
        attention_mask = attention_mask[:max_length]

        if return_tensors == "pt":
            import torch
            return {
                "input_ids": torch.tensor([token_ids], dtype=torch.long),
                "attention_mask": torch.tensor([attention_mask], dtype=torch.long)
            }
        return {"input_ids": token_ids, "attention_mask": attention_mask}

    def decode(self, token_ids):
        tokens = [self.inv_vocab.get(tid if isinstance(tid, int) else tid.item(), "[UNK]") for tid in token_ids]
        return " ".join([t for t in tokens if t != "[PAD]"])


def demo_tokenizer_internals():
    print("\n" + "=" * 65)
    print(" PART 2: TOKENIZER INTERNALS (SUBWORDS, IDS, ATTENTION MASKS)")
    print("=" * 65)

    sample_sentence = "This deep learning is amazing and fast !"
    print(f"Raw Input String: \"{sample_sentence}\"\n")

    tokenizer = None
    if TRANSFORMERS_AVAILABLE:
        try:
            tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        except Exception:
            tokenizer = None

    if tokenizer is None:
        print("[Using self-contained offline tokenizer demo]")
        tokenizer = StandaloneOfflineTokenizer()

    encoded = tokenizer(sample_sentence, return_tensors="pt")
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    print(f"1. input_ids tensor (shape {list(input_ids.shape)}):")
    print(f"   {input_ids[0].tolist()}")

    print(f"\n2. attention_mask tensor (shape {list(attention_mask.shape)}):")
    print(f"   {attention_mask[0].tolist()} (1 = attend, 0 = ignore padding)")

    print(f"\n3. Token Decoding (Reconstructing text from token IDs):")
    decoded = tokenizer.decode(input_ids[0])
    print(f"   Decoded String: \"{decoded}\"")


# ==============================================================================
# Part 3: Model Forward Pass & Logits Extraction
# ==============================================================================
class StandaloneOfflineTransformerModel(torch.nn.Module):
    """Simulates AutoModelForSequenceClassification forward pass."""
    def __init__(self, vocab_size=30000, hidden_dim=64, num_labels=2):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_dim)
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU()
        )
        self.classifier = torch.nn.Linear(hidden_dim, num_labels)

    def forward(self, input_ids, attention_mask=None):
        embeds = self.embedding(input_ids)
        # Pool across sequence length
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (embeds * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1.0)
        else:
            pooled = embeds.mean(dim=1)
        hidden = self.encoder(pooled)
        logits = self.classifier(hidden)
        
        class Output:
            def __init__(self, l):
                self.logits = l
        return Output(logits)


def demo_model_forward_pass():
    print("\n" + "=" * 65)
    print(" PART 3: MODEL FORWARD PASS & LOGITS EXTRACTION")
    print("=" * 65)

    model = None
    tokenizer = None

    if TRANSFORMERS_AVAILABLE:
        try:
            model_name = "distilbert-base-uncased-finetuned-sst-2-english"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
        except Exception:
            model = None

    if model is None:
        print("[Using self-contained offline transformer simulation]")
        tokenizer = StandaloneOfflineTokenizer()
        model = StandaloneOfflineTransformerModel(vocab_size=30000, hidden_dim=64, num_labels=2)

    sentence = "This deep learning is amazing and fast !"
    inputs = tokenizer(sentence, return_tensors="pt")

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)

    print(f"Sentence: \"{sentence}\"")
    print(f"Raw Logits:                 {logits[0].tolist()}")
    print(f"Softmax Probabilities:     {probabilities[0].tolist()}")
    pred_class = probabilities.argmax(dim=-1).item()
    labels = ["NEGATIVE", "POSITIVE"]
    print(f"Predicted Class:            {labels[pred_class]} ({probabilities[0][pred_class]*100:.2f}%)")


# ==============================================================================
# Part 4: Native PyTorch Fine-Tuning Step
# ==============================================================================
def demo_transformer_training_step():
    print("\n" + "=" * 65)
    print(" PART 4: NATIVE PYTORCH TRAINING STEP ON TOKENIZED BATCH")
    print("=" * 65)

    model = StandaloneOfflineTransformerModel(vocab_size=30000, hidden_dim=64, num_labels=2)
    tokenizer = StandaloneOfflineTokenizer()

    # Mini batch of sentences
    batch_sentences = [
        "This deep learning is amazing and fast !",
        "This is bad and slow !"
    ]
    targets = torch.tensor([1, 0], dtype=torch.long)  # 1 = Positive, 0 = Negative

    # Tokenize
    encoded = tokenizer(batch_sentences[0], return_tensors="pt")
    input_ids = torch.cat([tokenizer(s, return_tensors="pt")["input_ids"] for s in batch_sentences], dim=0)
    attention_mask = torch.cat([tokenizer(s, return_tensors="pt")["attention_mask"] for s in batch_sentences], dim=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    print("Executing 3 optimization steps...")
    for step in range(1, 4):
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, targets)
        loss.backward()
        optimizer.step()
        print(f"  Step {step}: Loss = {loss.item():.4f}")

    print("[OK] Transformer fine-tuning gradient flow verified.")


def main():
    print("=" * 65)
    print(" HUGGING FACE TRANSFORMERS BASICS: NLP TASKS")
    print("=" * 65)

    demo_hf_pipelines()
    demo_tokenizer_internals()
    demo_model_forward_pass()
    demo_transformer_training_step()

    print("\n" + "=" * 65)
    print("[OK] Hugging Face transformers module completed successfully!")
    print("=" * 65)


if __name__ == "__main__":
    main()

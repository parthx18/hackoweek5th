# Working with Large Language Models (LLMs): APIs & Open-Source Models

Large Language Models (LLMs) are autoregressive decoder-only transformers trained on web-scale text corpora to predict the next token. This note explores how to interface with commercial closed APIs (OpenAI, Anthropic) and run open-weight models locally (Ollama, Hugging Face).

---

## 1. Proprietary APIs vs. Open Models

| Feature | Closed APIs (OpenAI, Anthropic) | Open Models (Ollama, Hugging Face) |
|---|---|---|
| **Examples** | GPT-4o, Claude 3.5 Sonnet | LLaMA-3, Mistral 7B, Phi-3, Gemma |
| **Hosting** | Managed cloud infrastructure | On-premise, local GPU/CPU, private cloud |
| **Cost Model** | Pay per token (prompt + completion) | Hardware compute / hosting costs |
| **Privacy / Control** | Data sent to third-party endpoints | Full data sovereignty; runs air-gapped |
| **Fine-Tuning** | Restricted to vendor platforms | Full weight access (LoRA, QLoRA, full FT) |

---

## 2. Anatomy of an LLM Request

Modern chat completion endpoints follow the standard message format:

```json
{
  "model": "gpt-4o-mini",
  "messages": [
    {"role": "system", "content": "You are an expert AI tutor explaining deep learning concisely."},
    {"role": "user", "content": "Explain what self-attention is in two sentences."}
  ],
  "temperature": 0.7,
  "max_tokens": 150,
  "top_p": 0.95
}
```

### Hyperparameters Explained
- **`temperature` ($T$)**: Modulates the softmax probability distribution over tokens:
  $$P(w_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$
  - Lower ($T \to 0$): Greedy, deterministic, factual answers.
  - Higher ($T > 0.8$): Creative, varied outputs.
- **`top_p` (Nucleus Sampling)**: Dynamically selects the smallest set of tokens whose cumulative probability exceeds $p$.
- **`max_tokens`**: The maximum number of completion tokens generated before stopping.

---

## 3. Running Open Models with Ollama

Ollama provides a local REST API for downloading and running quantized open models on consumer hardware (Mac, Windows, Linux).

- **Start Ollama server**:
  ```bash
  ollama serve
  ollama run mistral
  ```
- **Query via HTTP endpoint (`http://localhost:11434/api/chat`)**:
  ```python
  import requests

  payload = {
      "model": "mistral",
      "messages": [{"role": "user", "content": "Summarize transfer learning."}],
      "stream": False
  }
  response = requests.post("http://localhost:11434/api/chat", json=payload)
  print(response.json()["message"]["content"])
  ```

---

## 4. Common LLM Application Tasks

1. **Text Generation**: Free-form creative text, code generation, reasoning.
2. **Text Summarization**: Condensing long articles, research papers, or transcripts into concise bullet points.
3. **Retrieval-Augmented Generation / Question Answering**: Grounding answers strictly in provided context documents to eliminate hallucinations.

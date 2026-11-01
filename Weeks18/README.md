# Week 18 — Transfer Learning, Hugging Face Transformers & Modern LLM APIs

Welcome to **Week 18** of the deep learning curriculum! This week bridges classical deep learning architectures with the modern generative AI and pretrained transformer ecosystem.

---

## 🎯 Topics Covered

### 1. Transfer Learning with Pretrained CNNs
- **Transfer Learning Paradigm**: Why train from scratch when models trained on millions of images (ImageNet) have already learned rich hierarchical representations?
- **Feature Extraction**: Freezing the convolutional backbone (`requires_grad = False`) and replacing only the final classification head (`fc`).
- **Fine-Tuning**: Unfreezing the top residual blocks and training with differential learning rates (low LR for backbone, higher LR for classification head).
- **Computer Vision Pipelines**: ImageNet normalization, data augmentation (random crops, flips), and evaluation.

### 2. Hugging Face Transformers Basics
- **The Transformer Ecosystem**: Self-attention, tokenization, and model repositories on the Hugging Face Hub.
- **Pipelines (`transformers.pipeline`)**: High-level, zero-boilerplate inference for sentiment analysis, text classification, and summarization.
- **Tokenizers (`AutoTokenizer`)**: Subword tokenization (WordPiece/BPE), token IDs, attention masks, special tokens (`[CLS]`, `[SEP]`), and decoding.
- **Model Internals (`AutoModelForSequenceClassification`)**: Forward pass, logits extraction, softmax probability computation, and fine-tuning loops.

### 3. Using LLM APIs & Open-Weight Models
- **Modern Large Language Models**: Autoregressive decoder-only models, prompt engineering, system/user roles, and sampling parameters (`temperature`, `top_p`, `max_tokens`).
- **Multi-Provider Unified Client**:
  - **OpenAI**: `gpt-4o`, `gpt-4o-mini`, `gpt-3.5-turbo`
  - **Anthropic**: `claude-3-5-sonnet`, `claude-3-haiku`
  - **Hugging Face**: Inference API & local pipeline models
  - **Ollama**: Local open-source models (`llama3`, `mistral`, `phi3`) running on your own machine
  - **Offline/Mock Provider**: Built-in mock mode ensuring zero-crash execution even without API keys or local servers.
- **Supported Tasks**:
  - Text generation & instruction following
  - Text summarization
  - Question Answering grounded in provided context

---

## 🗂️ Directory Structure

```
Weeks18/
├── README.md
├── notes/
│   ├── 01-transfer-learning-cnns.md
│   ├── 02-transformers-and-huggingface.md
│   └── 03-llm-apis-and-open-models.md
└── mini-project/
    ├── requirements.txt
    ├── transfer_learning_classifier.py    # ResNet/MobileNet feature extraction & fine-tuning
    ├── huggingface_nlp_tasks.py           # Pipelines, AutoTokenizer, AutoModel, classification
    └── multi_llm_client.py                # Unified client: OpenAI, Anthropic, HuggingFace, Ollama
```

---

## 🚀 Running the Mini-Projects

### Setup Environment
```bash
cd Weeks18/mini-project
pip install -r requirements.txt
```

### 1. Transfer Learning Classifier
```bash
python transfer_learning_classifier.py
```
*Demonstrates feature extraction (frozen backbone) vs fine-tuning (differential learning rate) using ResNet.*

### 2. Hugging Face NLP Tasks
```bash
python huggingface_nlp_tasks.py
```
*Runs pipeline inference for sentiment and summarization, inspects tokenization masks, and runs forward passes with logits.*

### 3. Unified Multi-LLM Client
```bash
python multi_llm_client.py
```
*Demonstrates text generation, summarization, and context-based question answering across OpenAI, Anthropic, Hugging Face, and Ollama.*

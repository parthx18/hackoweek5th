"""
Unified Multi-Provider LLM Client: OpenAI, Anthropic, Hugging Face & Ollama
=============================================================================
Covers:
1. Unified API interface across closed and open-source LLM providers.
2. Providers supported:
   - OpenAI (gpt-4o, gpt-4o-mini, gpt-3.5-turbo)
   - Anthropic (claude-3-5-sonnet, claude-3-haiku)
   - Hugging Face (Inference API & open models)
   - Ollama (Local open-source models: LLaMA-3, Mistral, Phi-3)
   - MockOffline (Built-in offline fallback for verification without API keys)
3. Three core tasks:
   - Text Generation / Creative Completion
   - Text Summarization
   - Context-Grounded Question Answering (QA)

Usage:
    python multi_llm_client.py [--provider {mock,openai,anthropic,huggingface,ollama}] [--task {all,generate,summarize,qa}]
"""

import os
import sys
import json
import argparse
import urllib.request
import urllib.error


# ==============================================================================
# Base Provider Interface
# ==============================================================================
class BaseLLMProvider:
    """Abstract interface that all LLM backends implement."""
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 250) -> str:
        raise NotImplementedError

    def summarize(self, text: str, max_tokens: int = 150) -> str:
        prompt = f"Please provide a concise, factual summary of the following text:\n\n{text}\n\nSummary:"
        return self.generate(prompt, temperature=0.3, max_tokens=max_tokens)

    def answer_question(self, context: str, question: str) -> str:
        prompt = (
            f"Use strictly the following context to answer the question accurately.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{question}\n\n"
            f"Answer:"
        )
        return self.generate(prompt, temperature=0.2, max_tokens=200)


# ==============================================================================
# Provider 1: Mock / Offline Provider (Zero Dependencies, Zero API Keys)
# ==============================================================================
class MockOfflineProvider(BaseLLMProvider):
    """
    Provides realistic, deterministic simulated responses for automated testing
    and offline demonstration when external API credentials are not present.
    """
    def __init__(self, model_name="mock-gpt-simulated"):
        self.model_name = model_name

    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 250) -> str:
        prompt_lower = prompt.lower()
        if "summary" in prompt_lower:
            return (
                "[Mock Summary] Deep learning systems learn hierarchical representations through "
                "consecutive non-linear transformations. Pretrained backbones and self-attention "
                "transformers allow rapid adaptation to downstream tasks with minimal fine-tuning data."
            )
        elif "question" in prompt_lower or "context:" in prompt_lower:
            return (
                "[Mock QA Response] Based on the provided context, the primary advantage of transfer "
                "learning is reusing low-level visual features (edges, textures) learned from large datasets, "
                "which drastically reduces training time and labeled data requirements."
            )
        else:
            return (
                f"[Mock Generation with model '{self.model_name}']\n"
                f"Artificial Intelligence has evolved from handcrafted rule-based systems to deep "
                f"neural networks and transformer foundation models. By leveraging large-scale pretraining "
                f"and fine-tuning, models can achieve human-level performance across language, vision, and reasoning."
            )


# ==============================================================================
# Provider 2: OpenAI Provider (GPT-4o, GPT-4o-mini)
# ==============================================================================
class OpenAIProvider(BaseLLMProvider):
    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable or parameter is required.")

    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 250) -> str:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful and precise technical assistant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"].strip()


# ==============================================================================
# Provider 3: Anthropic Provider (Claude 3.5 Sonnet, Claude 3 Haiku)
# ==============================================================================
class AnthropicProvider(BaseLLMProvider):
    def __init__(self, api_key: str = None, model: str = "claude-3-5-sonnet-20241022"):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable or parameter is required.")

    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 250) -> str:
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}]
        }
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["content"][0]["text"].strip()


# ==============================================================================
# Provider 4: Ollama Provider (Local Open Models: LLaMA-3, Mistral, Phi-3)
# ==============================================================================
class OllamaProvider(BaseLLMProvider):
    def __init__(self, host: str = "http://localhost:11434", model: str = "llama3"):
        self.host = host.rstrip("/")
        self.model = model

    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 250) -> str:
        url = f"{self.host}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("response", "").strip()
        except urllib.error.URLError as e:
            raise ConnectionError(
                f"Could not connect to local Ollama server at {self.host} ({e}). "
                "Ensure Ollama is running via 'ollama serve'."
            )


# ==============================================================================
# Provider 5: Hugging Face Inference API
# ==============================================================================
class HuggingFaceProvider(BaseLLMProvider):
    def __init__(self, api_token: str = None, model: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        self.api_token = api_token or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        self.model = model
        if not self.api_token:
            raise ValueError("HUGGINGFACEHUB_API_TOKEN environment variable or parameter is required.")

    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 250) -> str:
        url = f"https://api-inference.huggingface.co/models/{self.model}"
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        payload = {
            "inputs": prompt,
            "parameters": {"temperature": temperature, "max_new_tokens": max_tokens}
        }
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if isinstance(data, list) and "generated_text" in data[0]:
                return data[0]["generated_text"].replace(prompt, "").strip()
            return str(data)


# ==============================================================================
# Unified Client Manager
# ==============================================================================
class UnifiedLLMClient:
    """Manages provider selection and offers a single clean API."""
    def __init__(self, provider_name: str = "mock", **kwargs):
        self.provider_name = provider_name.lower()
        if self.provider_name == "openai":
            self.provider = OpenAIProvider(**kwargs)
        elif self.provider_name == "anthropic":
            self.provider = AnthropicProvider(**kwargs)
        elif self.provider_name == "ollama":
            self.provider = OllamaProvider(**kwargs)
        elif self.provider_name == "huggingface":
            self.provider = HuggingFaceProvider(**kwargs)
        else:
            self.provider = MockOfflineProvider(**kwargs)

    def generate(self, prompt: str, **kwargs) -> str:
        return self.provider.generate(prompt, **kwargs)

    def summarize(self, text: str, **kwargs) -> str:
        return self.provider.summarize(text, **kwargs)

    def answer_question(self, context: str, question: str) -> str:
        return self.provider.answer_question(context, question)


# ==============================================================================
# Demonstration Tasks
# ==============================================================================
def run_llm_demonstration(client):
    print("\n" + "=" * 65)
    print(f" LLM DEMO WITH PROVIDER: {client.provider_name.upper()}")
    print("=" * 65)

    # 1. Text Generation
    print("\n[Task 1: Text Generation]")
    prompt = "Explain in two sentences how convolutional neural networks differ from multilayer perceptrons."
    print(f"Prompt: \"{prompt}\"")
    output = client.generate(prompt, temperature=0.7)
    print(f"Response:\n{output}\n")

    # 2. Text Summarization
    print("-" * 65)
    print("[Task 2: Text Summarization]")
    long_text = (
        "Transfer learning in deep learning represents a major paradigm shift. Instead of training "
        "a vision model from scratch with randomly initialized weights—which requires hundreds of "
        "thousands of labeled images and massive compute—engineers load a model pretrained on ImageNet. "
        "The early layers act as universal feature extractors capturing edges and textures, while only "
        "the final classification head needs to be trained on the specific target domain."
    )
    print(f"Source Text ({len(long_text)} chars):\n\"{long_text}\"")
    summary = client.summarize(long_text, max_tokens=100)
    print(f"\nGenerated Summary:\n{summary}\n")

    # 3. Question Answering
    print("-" * 65)
    print("[Task 3: Grounded Question Answering]")
    context = (
        "Batch Normalization was introduced by Ioffe and Szegedy in 2015. It normalizes each layer's "
        "inputs using the mean and variance computed across the current mini-batch, and introduces two "
        "learnable affine parameters (gamma and beta) to scale and shift the normalized activations. "
        "During evaluation, running historical statistics are used instead of mini-batch statistics."
    )
    question = "What are the two learnable parameters introduced by Batch Normalization, and what do they do?"
    print(f"Context:\n\"{context}\"")
    print(f"Question: \"{question}\"")
    answer = client.answer_question(context, question)
    print(f"\nGenerated Answer:\n{answer}\n")

    print("=" * 65)
    print("[OK] All 3 LLM tasks (Generation, Summarization, QA) executed successfully!")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Multi-Provider LLM Client Demonstration")
    parser.add_argument(
        "--provider",
        choices=["mock", "openai", "anthropic", "ollama", "huggingface"],
        default="mock",
        help="LLM Provider backend (default: mock for instant zero-dependency execution)"
    )
    parser.add_argument("--model", type=str, default=None, help="Custom model identifier")
    args = parser.parse_args()

    # Automatic fallback if credentials not found
    try:
        kwargs = {}
        if args.model:
            kwargs["model"] = args.model
        client = UnifiedLLMClient(provider_name=args.provider, **kwargs)
    except Exception as e:
        print(f"Notice: Provider '{args.provider}' could not initialize ({e}). Falling back to 'mock' provider.")
        client = UnifiedLLMClient(provider_name="mock")

    run_llm_demonstration(client)


if __name__ == "__main__":
    main()

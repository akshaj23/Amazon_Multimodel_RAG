"""
Large Language Model integration for RAG responses.

This module uses a local Ollama server with Llama 3.1 by default. No API key is
required or stored in the repository.
"""

import logging
from enum import Enum
from typing import Dict, List

import requests

from config import (
    LLM_MAX_TOKENS,
    LLM_MODEL_NAME,
    LLM_TEMPERATURE,
    LLM_TOP_P,
    OLLAMA_BASE_URL,
)

logger = logging.getLogger(__name__)


class PromptStrategy(Enum):
    """Prompt engineering strategies supported by the app."""

    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"
    MULTI_SHOT = "multi_shot"
    RAG_AUGMENTED = "rag_augmented"


class LLMInterface:
    """
    Ollama-backed LLM interface for product Q&A.

    The default model is `llama3.1`. Run `ollama pull llama3.1` and keep the
    Ollama app/server running before using LLM responses.
    """

    def __init__(
        self,
        model_name: str = LLM_MODEL_NAME,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
        base_url: str = OLLAMA_BASE_URL,
        timeout: int = 60,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def is_available(self) -> bool:
        """Return True when the Ollama server is reachable."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=3)
            response.raise_for_status()
            return True
        except requests.RequestException:
            return False

    def load_model(self) -> None:
        """Validate that Ollama is reachable and warn if the model is missing."""
        response = requests.get(f"{self.base_url}/api/tags", timeout=5)
        response.raise_for_status()
        models = response.json().get("models", [])
        available = {model.get("name", "").split(":")[0] for model in models}

        if self.model_name.split(":")[0] not in available:
            logger.warning(
                "Ollama model %s is not listed locally. Pull it with: ollama pull %s",
                self.model_name,
                self.model_name,
            )

    def generate(
        self,
        prompt: str,
        strategy: PromptStrategy = PromptStrategy.RAG_AUGMENTED,
        **kwargs,
    ) -> str:
        """Generate an answer with Ollama."""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.temperature),
                "top_p": kwargs.get("top_p", LLM_TOP_P),
                "num_predict": kwargs.get("max_tokens", self.max_tokens),
            },
        }

        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()

    def zero_shot_prompt(self, question: str, context: str = "") -> str:
        """Create a zero-shot prompt."""
        return f"""You are a helpful e-commerce product assistant.

Use only the product data provided in context. If the answer is not available,
say that it is not available in the retrieved product data.

Context:
{context}

Question: {question}

Answer:"""

    def few_shot_prompt(
        self,
        question: str,
        examples: List[Dict[str, str]] = None,
        context: str = "",
    ) -> str:
        """Create a few-shot prompt."""
        examples = examples or [
            {
                "question": "What is the price?",
                "answer": "The listed price is the price field from the retrieved product.",
            },
            {
                "question": "What brand is it?",
                "answer": "The brand is the brand field from the retrieved product, or N/A if missing.",
            },
        ]

        prompt = "You are a helpful e-commerce product assistant.\n\n"
        prompt += "Examples:\n"
        for example in examples:
            prompt += f"Q: {example['question']}\nA: {example['answer']}\n\n"

        prompt += f"""Product context:
{context}

Q: {question}
A:"""
        return prompt

    def multi_shot_prompt(self, question: str, context: str = "") -> str:
        """Create a multi-shot prompt with several product-support examples."""
        examples = [
            {
                "question": "What is this product?",
                "answer": "Identify the product using the retrieved title and category.",
            },
            {
                "question": "What is the price?",
                "answer": "Answer with the exact retrieved price and product name.",
            },
            {
                "question": "What are the features?",
                "answer": "Summarize the retrieved features and description without adding outside facts.",
            },
            {
                "question": "How do I use it?",
                "answer": "Explain likely usage only from the retrieved title, category, features, and description.",
            },
        ]
        return self.few_shot_prompt(question, examples=examples, context=context)

    def rag_augmented_prompt(
        self,
        question: str,
        retrieved_context: str,
        source_info: str = "",
    ) -> str:
        """Create a RAG prompt grounded in retrieved products."""
        return f"""You are a helpful e-commerce product assistant.

Use only the retrieved product information below. Do not invent facts, ratings,
reviews, availability, shipping details, or comparisons that are not in the
retrieved context. Treat Product 1 as the primary product. Only discuss other
products if the user explicitly asks for comparisons or recommendations.
Directly answer the user's question first, then add a brief supporting detail
if useful.

Retrieved product information:
{retrieved_context}

{f"Sources: {source_info}" if source_info else ""}

User question: {question}

Answer:"""

    def generate_with_strategy(
        self,
        question: str,
        strategy: PromptStrategy = PromptStrategy.RAG_AUGMENTED,
        context: str = "",
        examples: List[Dict[str, str]] = None,
        **kwargs,
    ) -> str:
        """Generate a response with the requested prompt strategy."""
        if strategy == PromptStrategy.ZERO_SHOT:
            prompt = self.zero_shot_prompt(question, context)
        elif strategy == PromptStrategy.FEW_SHOT:
            prompt = self.few_shot_prompt(question, examples=examples, context=context)
        elif strategy == PromptStrategy.MULTI_SHOT:
            prompt = self.multi_shot_prompt(question, context)
        elif strategy == PromptStrategy.RAG_AUGMENTED:
            prompt = self.rag_augmented_prompt(question, context)
        else:
            raise ValueError(f"Unknown prompt strategy: {strategy}")

        return self.generate(prompt, strategy=strategy, **kwargs)

    def generate_product_response(
        self,
        query: str,
        retrieved_products: List[Dict],
        strategy: PromptStrategy = PromptStrategy.RAG_AUGMENTED,
    ) -> str:
        """Generate a product answer using retrieved product metadata."""
        context = self._format_products_context(retrieved_products)
        return self.generate_with_strategy(query, strategy=strategy, context=context)

    def _format_products_context(self, products: List[Dict]) -> str:
        """Format products as compact context for the LLM."""
        if not products:
            return "No products were retrieved."

        parts = []
        for i, product in enumerate(products, 1):
            metadata = product.get("metadata", product)
            parts.append(
                "\n".join(
                    [
                        f"Product {i}: {metadata.get('title', 'Unknown')}",
                        f"ASIN: {metadata.get('asin', metadata.get('product_id', 'N/A'))}",
                        f"Brand: {metadata.get('brand') or 'N/A'}",
                        f"Price: {metadata.get('price', 'N/A')}",
                        f"Category: {metadata.get('category') or 'N/A'}",
                        f"Features: {metadata.get('features') or 'N/A'}",
                        f"Description: {metadata.get('description') or 'N/A'}",
                    ]
                )
            )

        return "\n\n".join(parts)

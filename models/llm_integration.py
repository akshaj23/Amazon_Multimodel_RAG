"""
Large Language Model (LLM) integration for conversational responses
"""

import logging
from typing import Optional, Dict, List
from enum import Enum

logger = logging.getLogger(__name__)


class PromptStrategy(Enum):
    """Enumeration of prompt engineering strategies"""
    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    RAG_AUGMENTED = "rag_augmented"


class LLMInterface:
    """
    Interface for Large Language Model integration

    Supports:
    - Multiple prompt engineering strategies
    - Context-aware response generation
    - RAG-augmented generation
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-hf",
        temperature: float = 0.7,
        max_tokens: int = 512,
        device: str = "cuda"
    ):
        """
        Initialize LLM interface

        Args:
            model_name: Model identifier (from HuggingFace)
            temperature: Generation temperature (0-1)
            max_tokens: Maximum tokens to generate
            device: Device to load model on
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.device = device
        self.model = None
        self.tokenizer = None

        logger.info(f"Initialized LLM interface for {model_name}")

        # Note: Actual model loading will be handled in load_model()

    def load_model(self) -> None:
        """
        Load the language model

        This method should be implemented with actual model loading logic
        using transformers library or API calls
        """
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM

            logger.info(f"Loading model: {self.model_name}")

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=self.device,
                torch_dtype="auto"
            )

            logger.info(f"Model loaded successfully on {self.device}")

        except ImportError:
            logger.warning("transformers library not installed. Using mock model.")
            self.model = None
            self.tokenizer = None
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def generate(
        self,
        prompt: str,
        strategy: PromptStrategy = PromptStrategy.RAG_AUGMENTED,
        **kwargs
    ) -> str:
        """
        Generate response using the LLM

        Args:
            prompt: Input prompt
            strategy: Prompt engineering strategy
            **kwargs: Additional generation parameters

        Returns:
            Generated response
        """
        if self.model is None:
            logger.warning("Model not loaded. Returning mock response.")
            return self._generate_mock_response(prompt)

        logger.debug(f"Generating response with {strategy.value} strategy")

        try:
            # Prepare input
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            # Generate
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=kwargs.get('top_p', 0.9),
                do_sample=True,
            )

            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Remove the prompt from the response
            response = response[len(prompt):].strip()

            return response

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

    def zero_shot_prompt(self, question: str, context: str = "") -> str:
        """
        Create zero-shot prompt

        Args:
            question: User question
            context: Optional context information

        Returns:
            Formatted prompt
        """
        prompt = f"""You are a helpful e-commerce product assistant.

{f'Context: {context}' if context else ''}

Question: {question}

Answer: """
        return prompt

    def few_shot_prompt(
        self,
        question: str,
        examples: List[Dict[str, str]] = None,
        context: str = ""
    ) -> str:
        """
        Create few-shot prompt with examples

        Args:
            question: User question
            examples: List of example QA pairs
            context: Optional context

        Returns:
            Formatted prompt with examples
        """
        prompt = "You are a helpful e-commerce product assistant.\n\n"

        # Add examples
        if examples:
            prompt += "Examples:\n"
            for i, example in enumerate(examples, 1):
                prompt += f"{i}. Q: {example['question']}\nA: {example['answer']}\n\n"

        if context:
            prompt += f"Context: {context}\n\n"

        prompt += f"Question: {question}\nAnswer: "

        return prompt

    def rag_augmented_prompt(
        self,
        question: str,
        retrieved_context: str,
        source_info: str = ""
    ) -> str:
        """
        Create RAG-augmented prompt with retrieved context

        Args:
            question: User question
            retrieved_context: Context retrieved from vector database
            source_info: Information about context sources

        Returns:
            Formatted RAG prompt
        """
        prompt = f"""You are a helpful e-commerce product assistant with access to a product database.

Use the following product information to answer the user's question accurately.

Product Information:
{retrieved_context}

{f'Sources: {source_info}' if source_info else ''}

User Question: {question}

Helpful Answer: """
        return prompt

    def chain_of_thought_prompt(self, question: str, context: str = "") -> str:
        """
        Create chain-of-thought prompt for complex reasoning

        Args:
            question: User question
            context: Optional context

        Returns:
            Formatted COT prompt
        """
        prompt = f"""You are a helpful e-commerce product assistant.

{f'Context: {context}' if context else ''}

Question: {question}

Let's think step by step:
1. """
        return prompt

    def generate_with_strategy(
        self,
        question: str,
        strategy: PromptStrategy = PromptStrategy.RAG_AUGMENTED,
        context: str = "",
        examples: List[Dict[str, str]] = None,
        **kwargs
    ) -> str:
        """
        Generate response with specific prompt engineering strategy

        Args:
            question: User question
            strategy: Prompt engineering strategy
            context: Context information
            examples: Examples for few-shot learning
            **kwargs: Additional parameters

        Returns:
            Generated response
        """
        if strategy == PromptStrategy.ZERO_SHOT:
            prompt = self.zero_shot_prompt(question, context)
        elif strategy == PromptStrategy.FEW_SHOT:
            prompt = self.few_shot_prompt(question, examples, context)
        elif strategy == PromptStrategy.CHAIN_OF_THOUGHT:
            prompt = self.chain_of_thought_prompt(question, context)
        elif strategy == PromptStrategy.RAG_AUGMENTED:
            prompt = self.rag_augmented_prompt(question, context)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        return self.generate(prompt, strategy, **kwargs)

    def _generate_mock_response(self, prompt: str) -> str:
        """
        Generate a mock response when model is not available

        Args:
            prompt: Input prompt

        Returns:
            Mock response
        """
        logger.debug("Generating mock response")

        mock_responses = {
            "samsung galaxy": "The Samsung Galaxy S21 comes with a 6.2-inch Dynamic AMOLED display, "
                            "a triple-camera setup (12MP wide, 64MP telephoto, 12MP ultrawide), "
                            "and a 4000mAh battery.",
            "airpods": "The Apple AirPods Pro feature active noise cancellation, customizable fit "
                      "with silicone tips, and are sweat and water-resistant.",
            "amazon echo": "The Amazon Echo Dot features Alexa voice assistant, 1.6-inch speaker, "
                          "and Bluetooth connectivity.",
        }

        # Find matching key
        prompt_lower = prompt.lower()
        for key, response in mock_responses.items():
            if key in prompt_lower:
                return response

        return "I'm a helpful product assistant. I can help answer questions about products. " \
               "Please ask about a specific product!"

    def set_temperature(self, temperature: float) -> None:
        """
        Set generation temperature

        Args:
            temperature: Temperature value (0-1)
        """
        if 0 <= temperature <= 1:
            self.temperature = temperature
        else:
            logger.warning(f"Temperature must be between 0 and 1, got {temperature}")

    def set_max_tokens(self, max_tokens: int) -> None:
        """
        Set maximum tokens to generate

        Args:
            max_tokens: Maximum token count
        """
        if max_tokens > 0:
            self.max_tokens = max_tokens
        else:
            logger.warning(f"max_tokens must be positive, got {max_tokens}")

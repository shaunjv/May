"""
LLM provider interface for the Intent Manager.
"""

from abc import ABC, abstractmethod
from typing import Type
from pydantic import BaseModel


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def generate_structured_output(
        self,
        prompt: str,
        response_model: Type[BaseModel],
        max_retries: int = 1
    ) -> BaseModel:
        """
        Generate structured output from the LLM.

        Args:
            prompt: The prompt to send to the LLM
            response_model: The Pydantic model to validate output against
            max_retries: Maximum number of retries for validation failures

        Returns:
            An instance of response_model populated with LLM output

        Raises:
            Exception: If generation or validation fails after retries
        """
        pass

    async def generate_text(self, prompt: str) -> str:
        """Generate a normal conversational response.

        This deliberately has a safe default instead of becoming abstract: existing
        structured-only providers remain source compatible.
        """
        raise NotImplementedError("This provider does not support conversational text.")


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing."""

    def __init__(self, responses: dict = None):
        """
        Initialize the mock provider.

        Args:
            responses: Dictionary mapping prompts to responses, or callable that returns responses
        """
        self.responses = responses or {}
        self.call_count = 0

    async def generate_structured_output(
        self,
        prompt: str,
        response_model: Type[BaseModel],
        max_retries: int = 1
    ) -> BaseModel:
        """
        Generate structured output from the mock LLM.

        Args:
            prompt: The prompt to send to the LLM
            response_model: The Pydantic model to validate output against
            max_retries: Maximum number of retries for validation failures

        Returns:
            An instance of response_model populated with mock output

        Raises:
            Exception: If no mock response is available or validation fails
        """
        self.call_count += 1

        # Check if we have a predefined response for this prompt
        if callable(self.responses):
            response_data = self.responses(prompt)
        elif prompt in self.responses:
            response_data = self.responses[prompt]
        else:
            # Return a default safe response
            response_data = {
                "primary_intent": "CLARIFICATION",
                "requires_clarification": True,
                "clarification_reason": "Mock response: unable to determine intent"
            }

        # Validate and return the response
        try:
            return response_model(**response_data)
        except Exception as e:
            raise Exception(f"Failed to validate mock response: {e}")

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from pydantic import BaseModel

from app.models.LLM.HighlightMoment import HighlightMoment


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    This abstraction allows seamless switching between different LLM providers
    (OpenAI, local models, Groq, etc.) without changing the core business logic.
    """

    @abstractmethod
    async def generate_structured_response(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[BaseModel],
        model: str = None,
        **kwargs
    ) -> BaseModel:
        """
        Generate a structured response from the LLM.

        Args:
            system_prompt: Instructions for the LLM's behavior
            user_prompt: The actual prompt/question for the LLM
            response_model: Pydantic model class for structured output
            model: Specific model to use (provider-dependent)
            **kwargs: Additional provider-specific parameters

        Returns:
            Instance of response_model with LLM's structured response

        Raises:
            LLMProviderError: If the provider encounters an error
        """
        pass

    @abstractmethod
    def get_available_models(self) -> List[str]:
        """
        Get list of available models for this provider.

        Returns:
            List of model names/identifiers
        """
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """
        Get the name of this provider.

        Returns:
            Provider name (e.g., "openai", "local", "groq")
        """
        pass


class LLMProviderError(Exception):
    """Exception raised by LLM providers."""

    def __init__(self, message: str, provider: str = None, original_error: Exception = None):
        self.provider = provider
        self.original_error = original_error
        super().__init__(message)
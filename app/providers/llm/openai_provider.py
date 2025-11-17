from typing import List
from pydantic import BaseModel
from openai import OpenAI

from app.providers.llm.base import LLMProvider, LLMProviderError
from app.core.logging import get_logger

logger = get_logger(__name__)


class OpenAIProvider(LLMProvider):
    """
    OpenAI API provider implementation.

    Provides structured response generation using OpenAI's models
    with the new structured outputs API.
    """

    def __init__(self, client: OpenAI):
        self.client = client
        self._available_models = [
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-4-turbo",
            "gpt-4",
            "gpt-3.5-turbo"
        ]

    async def generate_structured_response(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[BaseModel],
        model: str = "gpt-4o-mini",
        **kwargs
    ) -> BaseModel:
        """
        Generate structured response using OpenAI's structured outputs.

        Args:
            system_prompt: System instructions for the LLM
            user_prompt: User prompt/question
            response_model: Pydantic model for structured output
            model: OpenAI model to use
            **kwargs: Additional OpenAI API parameters

        Returns:
            Structured response as instance of response_model
        """
        try:
            logger.info(f"Generating response with OpenAI model: {model}")

            # Use OpenAI's structured outputs API
            response = self.client.beta.chat.completions.parse(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format=response_model,
                **kwargs
            )

            # Extract the parsed response
            structured_response = response.choices[0].message.parsed

            if structured_response is None:
                raise LLMProviderError(
                    "OpenAI returned no parsed content",
                    provider="openai"
                )

            logger.info("OpenAI response generated successfully")
            return structured_response

        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise LLMProviderError(
                f"OpenAI API request failed: {str(e)}",
                provider="openai",
                original_error=e
            )

    def get_available_models(self) -> List[str]:
        """Get list of available OpenAI models."""
        return self._available_models.copy()

    @property
    def provider_name(self) -> str:
        """Provider name identifier."""
        return "openai"
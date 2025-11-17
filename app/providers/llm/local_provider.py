import json
import asyncio
from typing import List, Dict, Any
from pydantic import BaseModel, ValidationError
import httpx

from app.providers.llm.base import LLMProvider, LLMProviderError
from app.core.logging import get_logger

logger = get_logger(__name__)


class LocalLLMProvider(LLMProvider):
    """
    Local LLM provider implementation using Ollama.

    This provider connects to a local Ollama server to run LLMs locally,
    providing cost-effective inference for development and testing.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        default_model: str = "llama3.1:8b",
        timeout: int = 300
    ):
        self.base_url = base_url.rstrip('/')
        self.default_model = default_model
        self.timeout = timeout
        self._available_models = None

    async def generate_structured_response(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[BaseModel],
        model: str = None,
        **kwargs
    ) -> BaseModel:
        """
        Generate structured response using local Ollama model.

        Args:
            system_prompt: System instructions for the LLM
            user_prompt: User prompt/question
            response_model: Pydantic model for structured output
            model: Local model to use (defaults to configured model)
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            Structured response as instance of response_model
        """
        model = model or self.default_model

        try:
            logger.info(f"Generating response with local model: {model}")

            # Create JSON schema from Pydantic model
            schema = response_model.model_json_schema()

            # Enhanced prompt with JSON schema guidance
            enhanced_prompt = self._create_structured_prompt(
                system_prompt, user_prompt, schema, response_model.__name__
            )

            # Make request to Ollama
            payload = {
                "model": model,
                "prompt": enhanced_prompt,
                "stream": False,
                "options": {
                    "temperature": kwargs.get("temperature", 0.1),
                    "num_predict": kwargs.get("max_tokens", 2048),
                    "top_p": kwargs.get("top_p", 0.9),
                }
            }

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json=payload
                )
                response.raise_for_status()
                result = response.json()

            # Extract and parse the response
            response_text = result.get("response", "").strip()

            if not response_text:
                raise LLMProviderError(
                    "Local LLM returned empty response",
                    provider="local"
                )

            # Try to extract JSON from the response
            structured_data = self._extract_json_from_response(response_text)

            # Validate and create Pydantic model
            try:
                structured_response = response_model.model_validate(structured_data)
                logger.info("Local LLM response generated and validated successfully")
                return structured_response

            except ValidationError as e:
                logger.warning(f"Validation failed, attempting fallback parsing: {e}")
                # Try fallback parsing strategies
                structured_response = self._fallback_parse(response_text, response_model)
                return structured_response

        except httpx.RequestError as e:
            logger.error(f"Failed to connect to local LLM server: {e}")
            raise LLMProviderError(
                f"Connection to local LLM server failed: {str(e)}",
                provider="local",
                original_error=e
            )
        except Exception as e:
            logger.error(f"Local LLM error: {str(e)}")
            raise LLMProviderError(
                f"Local LLM request failed: {str(e)}",
                provider="local",
                original_error=e
            )

    def _create_structured_prompt(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Dict[str, Any],
        model_name: str
    ) -> str:
        """Create a prompt using ChatML format for Qwen3 compatibility."""
        # Use the ChatML template from prompt manager
        from app.prompts.manager import prompt_manager

        return prompt_manager.format_prompt(
            provider_type='local',
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            template_name='chatml',
            schema=schema,
            model_name=model_name
        )

    def _extract_json_from_response(self, text: str) -> Dict[str, Any]:
        """Extract JSON from LLM response, enhanced for Qwen3 compatibility."""
        # Remove markdown code blocks if present
        text = text.strip()

        # Handle various markdown formats
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]

        # Remove common Qwen3 response patterns
        text = text.strip()

        # Handle thinking tags that Qwen3 might include
        if "<think>" in text and "</think>" in text:
            # Remove thinking content
            import re
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
            text = text.strip()

        # Find JSON boundaries with improved logic
        start = text.find('{')
        if start == -1:
            # Try to find array start
            start = text.find('[')

        if start == -1:
            raise LLMProviderError(
                "No JSON object or array found in LLM response",
                provider="local"
            )

        # Find matching closing bracket
        if text[start] == '{':
            bracket_count = 0
            end = start
            for i, char in enumerate(text[start:], start):
                if char == '{':
                    bracket_count += 1
                elif char == '}':
                    bracket_count -= 1
                    if bracket_count == 0:
                        end = i + 1
                        break
        else:  # Array
            bracket_count = 0
            end = start
            for i, char in enumerate(text[start:], start):
                if char == '[':
                    bracket_count += 1
                elif char == ']':
                    bracket_count -= 1
                    if bracket_count == 0:
                        end = i + 1
                        break

        if end <= start:
            raise LLMProviderError(
                "No complete JSON structure found in LLM response",
                provider="local"
            )

        json_text = text[start:end]

        try:
            return json.loads(json_text)
        except json.JSONDecodeError as e:
            # Enhanced error reporting for debugging
            logger.error(f"Failed to parse JSON. Original text length: {len(text)}")
            logger.error(f"Extracted JSON text: {json_text[:200]}..." if len(json_text) > 200 else f"Extracted JSON text: {json_text}")
            raise LLMProviderError(
                f"Invalid JSON in LLM response: {str(e)}",
                provider="local",
                original_error=e
            )

    def _fallback_parse(self, text: str, response_model: type[BaseModel]) -> BaseModel:
        """Implement fallback parsing strategies for malformed responses."""
        logger.warning("Attempting fallback parsing for malformed response")

        # Try to extract key information manually based on the model
        # This is a simplified fallback - you can extend this based on your specific needs
        if response_model.__name__ == "HighlightMoments":
            return self._parse_highlight_moments_fallback(text)

        raise LLMProviderError(
            "Failed to parse LLM response with all available strategies",
            provider="local"
        )

    def _parse_highlight_moments_fallback(self, text: str) -> BaseModel:
        """Fallback parser specifically for HighlightMoments."""
        # This is a basic implementation - extend as needed
        from app.models.LLM.HighlightMoments import HighlightMoments
        from app.models.LLM.HighlightMoment import HighlightMoment

        # Return empty highlights as fallback
        logger.warning("Using empty highlights as fallback")
        return HighlightMoments(highlights=[])

    async def get_available_models(self) -> List[str]:
        """Get list of available local models from Ollama."""
        if self._available_models is not None:
            return self._available_models

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                data = response.json()

                models = [model["name"] for model in data.get("models", [])]
                self._available_models = models
                return models

        except Exception as e:
            logger.warning(f"Failed to fetch available models: {e}")
            return [self.default_model]  # Fallback to default

    @property
    def provider_name(self) -> str:
        """Provider name identifier."""
        return "local"

    async def health_check(self) -> bool:
        """Check if the local LLM server is accessible."""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(f"{self.base_url}/api/version")
                return response.status_code == 200
        except Exception:
            return False
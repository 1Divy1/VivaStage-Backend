import json
import asyncio
from typing import List
from pydantic import BaseModel, ValidationError
from groq import Groq

from app.providers.llm.base_llm_provider import LLMProvider, LLMProviderError
from app.core.logging import get_logger

logger = get_logger(__name__)


class GroqProvider(LLMProvider):
    """
    Groq API provider implementation.

    Provides structured response generation using Groq's pydantic_models
    with JSON mode for structured outputs.
    """

    def __init__(self, client: Groq, default_model: str = "llama-3.1-8b-instant"):
        """
        Initialize Groq provider.

        Args:
            client: Groq client instance
            default_model: Default model to use (default: llama-3.1-8b-instant)
        """
        self.client = client
        self.default_model = default_model
        self._available_models = [
            "llama-3.1-8b-instant",
            "llama-3.1-70b-versatile",
            "llama-3.2-11b-vision-preview",
            "llama-3.3-70b-versatile",
            "mixtral-8x7b-32768"
        ]

    async def generate_structured_response(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[BaseModel],
        model: str = None,
        **kwargs
    ) -> BaseModel:
        """
        Generate structured response using Groq's JSON mode.

        Args:
            system_prompt: System instructions for the shorts
            user_prompt: User prompt/question
            response_model: Pydantic model for structured output
            model: Groq model to use (defaults to configured model)
            **kwargs: Additional Groq API parameters

        Returns:
            Structured response as instance of response_model
        """
        model = model or self.default_model

        try:
            logger.info(f"Generating response with Groq model: {model}")

            # Create JSON schema from Pydantic model
            schema = response_model.model_json_schema()

            # Enhance system prompt with JSON schema guidance
            enhanced_system_prompt = self._create_structured_system_prompt(
                system_prompt, schema, response_model.__name__
            )

            # Make request to Groq API with JSON mode
            # Run in executor to avoid blocking the event loop
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=model,
                messages=[
                    {"role": "system", "content": enhanced_system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=kwargs.get("temperature", 0.1),
                max_tokens=kwargs.get("max_tokens", 4096),
                **{k: v for k, v in kwargs.items() if k not in ["temperature", "max_tokens"]}
            )

            # Extract the response content
            response_text = response.choices[0].message.content

            if not response_text:
                raise LLMProviderError(
                    "Groq returned empty response",
                    provider="groq"
                )

            # Extract and parse JSON from the response
            structured_data = self._extract_json_from_response(response_text)

            # Validate and create Pydantic model
            try:
                structured_response = response_model.model_validate(structured_data)
                logger.info("Groq response generated and validated successfully")
                return structured_response

            except ValidationError as e:
                logger.warning(f"Validation failed, attempting fallback parsing: {e}")
                # Try fallback parsing strategies
                structured_response = self._fallback_parse(response_text, response_model)
                return structured_response

        except Exception as e:
            logger.error(f"Groq API error: {str(e)}")
            raise LLMProviderError(
                f"Groq API request failed: {str(e)}",
                provider="groq",
                original_error=e
            )

    def _create_structured_system_prompt(
        self,
        system_prompt: str,
        schema: dict,
        model_name: str
    ) -> str:
        """
        Enhance system prompt with JSON schema requirements.

        Args:
            system_prompt: Original system prompt
            schema: JSON schema from Pydantic model
            model_name: Name of the response model

        Returns:
            Enhanced system prompt with JSON schema instructions
        """
        schema_instruction = f"""

You must respond with valid JSON that matches this exact schema:

Schema for {model_name}:
{json.dumps(schema, indent=2)}

CRITICAL REQUIREMENTS:
1. Response must be ONLY valid JSON - no markdown, explanations, or additional text
2. JSON must match the schema exactly
3. All required fields must be present
4. Field types must match the schema specification
5. Do not wrap the JSON in code blocks or markdown
6. Return the JSON object directly"""

        return system_prompt + schema_instruction

    def _extract_json_from_response(self, text: str) -> dict:
        """
        Extract JSON from Groq response.

        Args:
            text: Raw response text from Groq

        Returns:
            Parsed JSON as dictionary

        Raises:
            LLMProviderError: If JSON cannot be extracted or parsed
        """
        text = text.strip()

        # Remove markdown code blocks if present
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]

        text = text.strip()

        # Find JSON boundaries
        start = text.find('{')
        if start == -1:
            # Try to find array start
            start = text.find('[')

        if start == -1:
            raise LLMProviderError(
                "No JSON object or array found in Groq response",
                provider="groq"
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
                "No complete JSON structure found in Groq response",
                provider="groq"
            )

        json_text = text[start:end]

        try:
            return json.loads(json_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON. Original text length: {len(text)}")
            logger.error(f"Extracted JSON text: {json_text[:200]}..." if len(json_text) > 200 else f"Extracted JSON text: {json_text}")
            raise LLMProviderError(
                f"Invalid JSON in Groq response: {str(e)}",
                provider="groq",
                original_error=e
            )

    def _fallback_parse(self, text: str, response_model: type[BaseModel]) -> BaseModel:
        """
        Implement fallback parsing strategies for malformed responses.

        Args:
            text: Response text that failed validation
            response_model: Expected Pydantic model

        Returns:
            Parsed model instance

        Raises:
            LLMProviderError: If fallback parsing fails
        """
        logger.warning("Attempting fallback parsing for malformed response")

        # Try to extract key information manually based on the model
        if response_model.__name__ == "HighlightMoments":
            return self._parse_highlight_moments_fallback(text)

        raise LLMProviderError(
            "Failed to parse Groq response with all available strategies",
            provider="groq"
        )

    def _parse_highlight_moments_fallback(self, text: str) -> BaseModel:
        """
        Fallback parser specifically for HighlightMoments.

        Args:
            text: Response text

        Returns:
            HighlightMoments instance (can be empty)
        """
        from app.pydantic_models.shorts.shorts_response_model import ShortsResponseModel

        # Return empty highlights as fallback
        logger.warning("Using empty highlights as fallback")
        return ShortsResponseModel(highlights=[])

    def get_available_models(self) -> List[str]:
        """Get list of available Groq pydantic_models."""
        return self._available_models.copy()

    @property
    def provider_name(self) -> str:
        """Provider name identifier."""
        return "groq"


from typing import Optional
from groq import Groq

from app.providers.transcription.base_transcription_provider import (
    TranscriptionProvider,
    TranscriptionProviderError
)
from app.providers.transcription.groq_transcription_provider import GroqTranscriptionProvider
from app.core.logging import get_logger

logger = get_logger(__name__)


class TranscriptionProviderFactory:
    """
    Factory class for creating transcription provider instances.

    Handles provider instantiation based on configuration,
    allowing seamless switching between different transcription providers.
    """

    @staticmethod
    def create_provider(
        provider_type: str,
        groq_client: Optional[Groq] = None,
        groq_model: Optional[str] = None,
        **kwargs
    ) -> TranscriptionProvider:
        """
        Create a transcription provider instance based on the specified type.

        Args:
            provider_type: Type of provider to create ("groq")
            groq_client: Groq client instance (required for "groq" provider)
            groq_model: Whisper model for Groq (default: "whisper-large-v3")
            **kwargs: Additional provider-specific configuration

        Returns:
            Configured transcription provider instance

        Raises:
            TranscriptionProviderError: If provider type is invalid or configuration is missing
        """
        provider_type = provider_type.lower().strip()

        logger.info(f"Creating transcription provider: {provider_type}")

        if provider_type == "groq":
            return TranscriptionProviderFactory._create_groq_provider(
                groq_client, groq_model, **kwargs
            )
        else:
            raise TranscriptionProviderError(
                f"Unknown transcription provider type: {provider_type}. "
                f"Supported types: groq",
                provider=provider_type
            )

    @staticmethod
    def _create_groq_provider(
        client: Optional[Groq],
        model: Optional[str] = None,
        **kwargs
    ) -> GroqTranscriptionProvider:
        """Create Groq transcription provider instance."""
        if client is None:
            raise TranscriptionProviderError(
                "Groq client is required for Groq transcription provider",
                provider="groq"
            )

        # Set default model
        model = model or "whisper-large-v3"

        return GroqTranscriptionProvider(client=client, model=model)

    @staticmethod
    def get_supported_providers() -> list[str]:
        """Get list of supported provider types."""
        return ["groq"]

    @staticmethod
    def validate_provider_config(provider_type: str, config: dict) -> bool:
        """
        Validate provider configuration.

        Args:
            provider_type: Type of provider to validate
            config: Configuration dictionary to validate

        Returns:
            True if configuration is valid

        Raises:
            TranscriptionProviderError: If configuration is invalid
        """
        provider_type = provider_type.lower().strip()

        if provider_type == "groq":
            if not config.get("groq_client"):
                raise TranscriptionProviderError(
                    "Groq client is required in configuration",
                    provider="groq"
                )
        else:
            raise TranscriptionProviderError(
                f"Unknown provider type for validation: {provider_type}",
                provider=provider_type
            )

        return True
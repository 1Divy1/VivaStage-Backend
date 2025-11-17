from typing import Optional
from openai import OpenAI

from app.providers.llm.base import LLMProvider, LLMProviderError
from app.providers.llm.openai_provider import OpenAIProvider
from app.providers.llm.local_provider import LocalLLMProvider
from app.core.logging import get_logger

logger = get_logger(__name__)


class LLMProviderFactory:
    """
    Factory class for creating LLM provider instances.

    Handles provider instantiation based on configuration,
    allowing seamless switching between different LLM providers.
    """

    @staticmethod
    def create_provider(
        provider_type: str,
        openai_client: Optional[OpenAI] = None,
        local_llm_url: Optional[str] = None,
        local_llm_model: Optional[str] = None,
        **kwargs
    ) -> LLMProvider:
        """
        Create an LLM provider instance based on the specified type.

        Args:
            provider_type: Type of provider to create ("openai", "local")
            openai_client: OpenAI client instance (required for "openai" provider)
            local_llm_url: Base URL for local LLM server (for "local" provider)
            local_llm_model: Default model for local LLM (for "local" provider)
            **kwargs: Additional provider-specific configuration

        Returns:
            Configured LLM provider instance

        Raises:
            LLMProviderError: If provider type is invalid or configuration is missing
        """
        provider_type = provider_type.lower().strip()

        logger.info(f"Creating LLM provider: {provider_type}")

        if provider_type == "openai":
            return LLMProviderFactory._create_openai_provider(openai_client, **kwargs)

        elif provider_type == "local":
            return LLMProviderFactory._create_local_provider(
                local_llm_url, local_llm_model, **kwargs
            )

        else:
            raise LLMProviderError(
                f"Unknown LLM provider type: {provider_type}. "
                f"Supported types: openai, local",
                provider=provider_type
            )

    @staticmethod
    def _create_openai_provider(client: Optional[OpenAI], **kwargs) -> OpenAIProvider:
        """Create OpenAI provider instance."""
        if client is None:
            raise LLMProviderError(
                "OpenAI client is required for OpenAI provider",
                provider="openai"
            )

        return OpenAIProvider(client=client)

    @staticmethod
    def _create_local_provider(
        base_url: Optional[str] = None,
        default_model: Optional[str] = None,
        **kwargs
    ) -> LocalLLMProvider:
        """Create local LLM provider instance."""
        # Set defaults
        base_url = base_url or "http://localhost:11434"
        default_model = default_model or "llama3.1:8b"

        timeout = kwargs.get("timeout", 300)

        return LocalLLMProvider(
            base_url=base_url,
            default_model=default_model,
            timeout=timeout
        )

    @staticmethod
    def get_supported_providers() -> list[str]:
        """Get list of supported provider types."""
        return ["openai", "local"]

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
            LLMProviderError: If configuration is invalid
        """
        provider_type = provider_type.lower().strip()

        if provider_type == "openai":
            if not config.get("openai_client"):
                raise LLMProviderError(
                    "OpenAI client is required in configuration",
                    provider="openai"
                )

        elif provider_type == "local":
            # Local provider has sensible defaults, so minimal validation needed
            url = config.get("local_llm_url", "http://localhost:11434")
            if not isinstance(url, str) or not url.startswith(("http://", "https://")):
                raise LLMProviderError(
                    "Invalid local_llm_url in configuration",
                    provider="local"
                )

        else:
            raise LLMProviderError(
                f"Unknown provider type for validation: {provider_type}",
                provider=provider_type
            )

        return True
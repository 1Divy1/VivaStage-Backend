from abc import ABC, abstractmethod
from typing import Dict, Any, List
from pathlib import Path


class TranscriptionProvider(ABC):
    """
    Abstract base class for transcription providers.

    This abstraction allows seamless switching between different transcription providers
    (Groq Whisper, OpenAI Whisper, local models, etc.) without changing the core business logic.
    """

    @abstractmethod
    async def transcribe_audio_chunk(
        self,
        audio_data: bytes,
        file_format: str,
        language: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Transcribe a single audio chunk.

        Args:
            audio_data: Raw audio data as bytes
            file_format: Audio format (e.g., "flac", "wav", "mp3")
            language: Target language for transcription
            **kwargs: Additional provider-specific parameters

        Returns:
            Dictionary containing transcription results with text, words, and segments

        Raises:
            TranscriptionProviderError: If the provider encounters an error
        """
        pass

    @abstractmethod
    async def transcribe_audio_file(
        self,
        audio_path: Path,
        language: str,
        chunk_length: int = 60,
        overlap: int = 0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Transcribe an entire audio file, potentially in chunks.

        Args:
            audio_path: Path to the audio file
            language: Target language for transcription
            chunk_length: Length of each chunk in seconds
            overlap: Overlap between chunks in seconds
            **kwargs: Additional provider-specific parameters

        Returns:
            Dictionary containing merged transcription results

        Raises:
            TranscriptionProviderError: If the provider encounters an error
        """
        pass

    @abstractmethod
    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported languages for this provider.

        Returns:
            List of language codes (e.g., ["en", "es", "fr"])
        """
        pass

    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported audio formats for this provider.

        Returns:
            List of audio format extensions (e.g., ["flac", "wav", "mp3"])
        """
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """
        Get the name of this provider.

        Returns:
            Provider name (e.g., "groq", "openai", "local")
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """
        Get the model name used by this provider.

        Returns:
            Model name (e.g., "whisper-large-v3")
        """
        pass


class TranscriptionProviderError(Exception):
    """Exception raised by transcription providers."""

    def __init__(self, message: str, provider: str = None, original_error: Exception = None):
        self.provider = provider
        self.original_error = original_error
        super().__init__(message)
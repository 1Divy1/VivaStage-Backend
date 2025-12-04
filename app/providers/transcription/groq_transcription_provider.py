import time
import io
import asyncio
from typing import Dict, Any, List
from pathlib import Path

import torch
import torchaudio
import soundfile as sf
from groq import Groq, RateLimitError

from app.providers.transcription.base_transcription_provider import (
    TranscriptionProvider,
    TranscriptionProviderError
)
from app.pydantic_models.transcription.transcription_response_model import (
    TranscriptionResponse,
    TranscriptionChunkResponse,
    TranscriptionWord,
    TranscriptionSegment
)
from app.core.logging import get_logger

logger = get_logger(__name__)


class GroqTranscriptionProvider(TranscriptionProvider):
    """
    Groq Whisper transcription provider implementation.

    Provides audio transcription using Groq's Whisper API with support for
    chunked processing and word-level timestamps.
    """

    def __init__(self, client: Groq, model: str = "whisper-large-v3"):
        """
        Initialize Groq transcription provider.

        Args:
            client: Groq client instance
            model: Whisper model to use (default: whisper-large-v3)
        """
        self.client = client
        self.model = model
        self._supported_languages = [
            "en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh",
            "ar", "hi", "tr", "pl", "nl", "sv", "da", "no", "fi", "he",
            "th", "vi", "uk", "cs", "hu", "ro", "bg", "hr", "sk", "sl",
            "et", "lv", "lt", "mt", "is", "mk", "sq", "az", "be", "bn",
            "bs", "eu", "fa", "gl", "ka", "kk", "ky", "lb", "ms", "ml",
            "mr", "ne", "ps", "si", "ta", "te", "ur", "uz", "cy"
        ]
        self._supported_formats = ["flac", "wav", "mp3", "m4a", "ogg", "opus"]

    async def transcribe_audio_chunk(
        self,
        audio_data: bytes,
        file_format: str,
        language: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Transcribe a single audio chunk using Groq Whisper.

        Args:
            audio_data: Raw audio data as bytes
            file_format: Audio format (e.g., "flac", "wav", "mp3")
            language: Target language for transcription
            **kwargs: Additional parameters (temperature, response_format, etc.)

        Returns:
            Dictionary containing transcription results

        Raises:
            TranscriptionProviderError: If transcription fails
        """
        temperature = kwargs.get("temperature", 0.0)
        response_format = kwargs.get("response_format", "verbose_json")
        timestamp_granularities = kwargs.get("timestamp_granularities", ["word"])

        # Create BytesIO buffer from audio data
        buffer = io.BytesIO(audio_data)
        filename = f"chunk.{file_format}"

        while True:
            start_time = time.time()
            try:
                logger.debug(f"Transcribing audio chunk with Groq model: {self.model}")

                result = await asyncio.to_thread(
                    self.client.audio.transcriptions.create,
                    file=(filename, buffer, f"audio/{file_format}"),
                    model=self.model,
                    language=language,
                    response_format=response_format,
                    timestamp_granularities=timestamp_granularities,
                    temperature=temperature
                )

                api_time = time.time() - start_time
                logger.debug(f"Audio chunk transcribed in {api_time:.2f}s")

                # Convert Groq response to our standardized format
                result = self._convert_groq_response(result)

                # Validate the chunk result
                if not result.get("text") and not result.get("segments"):
                    logger.warning("Chunk transcription appears to have failed - empty result")
                    return {
                        "text": "",
                        "words": [],
                        "segments": [],
                        "language": language,
                        "transcription_failed": True
                    }

                return result

            except RateLimitError:
                logger.warning("Rate limit hit - retrying in 60 seconds...")
                await asyncio.sleep(60)
                continue

            except Exception as e:
                logger.error(f"Error transcribing audio chunk: {str(e)}")
                raise TranscriptionProviderError(
                    f"Groq transcription failed: {str(e)}",
                    provider="groq",
                    original_error=e
                )

    async def transcribe_audio_file(
        self,
        audio_path: Path,
        language: str,
        chunk_length: int = 60,
        overlap: int = 0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Transcribe an entire audio file in chunks using Groq Whisper.

        Args:
            audio_path: Path to the audio file
            language: Target language for transcription
            chunk_length: Length of each chunk in seconds
            overlap: Overlap between chunks in seconds
            **kwargs: Additional parameters

        Returns:
            Dictionary containing merged transcription results

        Raises:
            TranscriptionProviderError: If transcription fails
        """
        if not audio_path.exists():
            raise TranscriptionProviderError(
                f"Audio file not found: {audio_path}",
                provider="groq"
            )

        logger.info(f"Starting transcription of: {audio_path}")

        try:
            # Load audio with torchaudio (convert Path to string)
            waveform, sample_rate = torchaudio.load(str(audio_path))
            duration_s = waveform.shape[1] / sample_rate
            logger.info(f"Audio duration: {duration_s:.2f}s")

            # Calculate chunk parameters
            chunk_samples = chunk_length * sample_rate
            overlap_samples = overlap * sample_rate
            step = chunk_samples - overlap_samples
            total_chunks = (waveform.shape[1] // step) + 1
            logger.info(f"Processing {total_chunks} chunks...")

            results = []
            total_transcription_time = 0

            for i in range(total_chunks):
                start = i * step
                end = min(start + chunk_samples, waveform.shape[1])

                logger.debug(f"Processing chunk {i + 1}/{total_chunks} (samples {start}-{end})")

                # Extract chunk waveform
                chunk_waveform = waveform[:, start:end]

                # Convert to FLAC bytes
                chunk_start_time = time.time()
                chunk_audio_data = self._waveform_to_bytes(chunk_waveform, sample_rate, "flac")

                # Transcribe chunk
                result = await self.transcribe_audio_chunk(
                    audio_data=chunk_audio_data,
                    file_format="flac",
                    language=language,
                    **kwargs
                )

                chunk_time = time.time() - chunk_start_time
                total_transcription_time += chunk_time

                # Store result with offset
                chunk_start_offset = start / sample_rate
                results.append((result, chunk_start_offset * 1000))  # offset in ms

            # Merge all chunk results
            final_result = self._merge_transcripts(results)
            logger.info(f"Total transcription time: {total_transcription_time:.2f}s")

            return final_result

        except Exception as e:
            logger.error(f"Error transcribing audio file: {str(e)}")
            raise TranscriptionProviderError(
                f"Audio file transcription failed: {str(e)}",
                provider="groq",
                original_error=e
            )

    def _waveform_to_bytes(self, waveform: torch.Tensor, sample_rate: int, format: str) -> bytes:
        """Convert torch waveform to bytes in specified format."""
        buffer = io.BytesIO()
        sf.write(buffer, waveform.squeeze(0).numpy(), sample_rate, format=format.upper())
        buffer.seek(0)
        return buffer.getvalue()

    def _convert_groq_response(self, groq_result) -> Dict[str, Any]:
        """Convert Groq API response to standardized format with validation."""
        def to_dict(obj):
            return obj.model_dump() if hasattr(obj, "model_dump") else obj

        data = to_dict(groq_result)

        # Validate and normalize response
        validated_response = {
            "text": data.get("text", ""),
            "words": data.get("words") or [],  # Ensure never None
            "segments": data.get("segments") or [],  # Ensure never None
            "language": data.get("language")
        }

        # Log warning for missing data
        if data.get("segments") is None:
            logger.warning("Groq API returned None for segments - chunk may have failed transcription")
        if data.get("words") is None:
            logger.warning("Groq API returned None for words - chunk may have failed transcription")

        return validated_response

    def _merge_transcripts(self, results: List[tuple[Dict[str, Any], float]]) -> Dict[str, Any]:
        """
        Merge multiple transcript chunks into a single result.

        Args:
            results: List of (transcript_dict, start_offset_ms) tuples

        Returns:
            Merged transcript dictionary
        """
        logger.debug("Merging transcript results...")

        def extract_text(chunk):
            return chunk.get("text", "")

        def extract_words(chunk, start_offset_s):
            # Triple-safety: handle None at multiple levels
            raw_words = chunk.get("words")
            if raw_words is None:
                return []

            # Ensure we have a list
            if not isinstance(raw_words, list):
                logger.warning(f"Expected list for words, got {type(raw_words)}")
                return []

            # Process valid words
            processed_words = []
            for w in raw_words:
                if w is None:
                    logger.warning("Skipping None word in chunk")
                    continue
                processed_words.append({
                    "word": w.get("word", ""),
                    "start": w.get("start", 0) + start_offset_s,
                    "end": w.get("end", 0) + start_offset_s,
                })
            return processed_words

        def extract_segments(chunk, start_offset_s):
            # Triple-safety: handle None at multiple levels
            raw_segments = chunk.get("segments")
            if raw_segments is None:
                return []

            # Ensure we have a list
            if not isinstance(raw_segments, list):
                logger.warning(f"Expected list for segments, got {type(raw_segments)}")
                return []

            # Process valid segments
            adjusted_segments = []
            for s in raw_segments:
                if s is None:
                    logger.warning("Skipping None segment in chunk")
                    continue
                adjusted_segment = dict(s)  # Create a copy
                adjusted_segment["start"] = s.get("start", 0) + start_offset_s
                adjusted_segment["end"] = s.get("end", 0) + start_offset_s
                adjusted_segments.append(adjusted_segment)
            return adjusted_segments

        all_texts = []
        all_words = []
        all_segments = []

        for chunk, chunk_start_ms in results:
            chunk_start_s = chunk_start_ms / 1000

            all_texts.append(extract_text(chunk))
            all_words.extend(extract_words(chunk, chunk_start_s))
            all_segments.extend(extract_segments(chunk, chunk_start_s))

        # Add failure monitoring
        failed_chunks = sum(1 for chunk, _ in results if chunk.get("transcription_failed"))
        total_chunks = len(results)

        if failed_chunks > total_chunks * 0.5:  # More than 50% failed
            logger.error(f"High transcription failure rate: {failed_chunks}/{total_chunks} chunks failed")
            raise TranscriptionProviderError(
                f"Transcription quality too poor: {failed_chunks}/{total_chunks} chunks failed",
                provider="groq"
            )

        if failed_chunks > 0:
            logger.warning(f"Transcription completed with {failed_chunks}/{total_chunks} failed chunks")

        merged_text = " ".join(all_texts)

        return {
            "text": merged_text,
            "words": all_words,
            "segments": all_segments,
        }

    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages for Groq Whisper."""
        return self._supported_languages.copy()

    def get_supported_formats(self) -> List[str]:
        """Get list of supported audio formats for Groq Whisper."""
        return self._supported_formats.copy()

    @property
    def provider_name(self) -> str:
        """Provider name identifier."""
        return "groq"

    @property
    def model_name(self) -> str:
        """Model name used by this provider."""
        return self.model
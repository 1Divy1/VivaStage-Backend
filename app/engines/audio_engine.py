import subprocess
import re
from pathlib import Path

import torch
import torchaudio

from app.providers.transcription.base_transcription_provider import TranscriptionProvider
from app.core.logging import get_logger

logger = get_logger(__name__)


class AudioEngine:
    """
    AudioEngine provides methods for audio preprocessing and transcription.

    Uses an injected TranscriptionProvider for actual transcription work,
    while handling audio preprocessing and chunk merging.
    """

    def __init__(self, transcription_provider: TranscriptionProvider):
        """
        Initialize AudioEngine with a transcription provider.

        Args:
            transcription_provider: Provider instance for transcription operations
        """
        self.transcription_provider = transcription_provider

    # ----------------------------
    # Public orchestrator methods
    # ----------------------------
    async def transcribe_audio_in_chunks(
        self,
        audio_path: Path,
        transcription_dir: Path,
        chunk_length: int,
        overlap: int,
        language: str,
    ) -> dict:
        """
        Transcribe audio file in chunks using the configured transcription provider.

        Args:
            audio_path: Path to the audio file
            transcription_dir: Directory for temporary files
            chunk_length: Length of each chunk in seconds
            overlap: Overlap between chunks in seconds
            language: Target language for transcription

        Returns:
            Dictionary containing transcription results
        """
        logger.info(f"Starting transcription of: {audio_path}")

        processed_path = None
        try:
            # Preprocess to 16kHz mono FLAC
            processed_path = self._preprocess_audio(
                input_path=audio_path,
                output_path=Path(transcription_dir) / "processed.flac",
            )

            # Use the transcription provider to transcribe the preprocessed file
            result = await self.transcription_provider.transcribe_audio_file(
                audio_path=processed_path,
                language=language,
                chunk_length=chunk_length,
                overlap=overlap,
                response_format="verbose_json",
                timestamp_granularities=["word"],
                temperature=0.0
            )

            return result

        finally:
            # Clean up temporary files
            if processed_path and Path(processed_path).exists():
                Path(processed_path).unlink(missing_ok=True)

    async def transcribe_audio_with_output(self, audio_file: str, transcription_dir: str, language: str) -> dict:
        """
        Transcribe audio file in chunks and save the result.

        Args:
            audio_file: Path to the audio file
            transcription_dir: Directory to save transcription results
            language: Target language for transcription

        Returns:
            Dictionary containing transcription results
        """
        from app.utils.utils import save_data

        logger.info("Transcribing audio...")

        result = await self.transcribe_audio_in_chunks(
            audio_path=Path(audio_file),
            transcription_dir=Path(transcription_dir),
            chunk_length=60,  # seconds
            overlap=0,  # seconds
            language=language
        )

        save_data(data=result, base_path=transcription_dir, file_name="transcription")
        return result

    # ----------------------------
    # Audio preprocessing utilities
    # ----------------------------
    @staticmethod
    def _preprocess_audio(input_path: Path, output_path: Path) -> Path:
        """
        Preprocess audio file to 16kHz mono FLAC format.

        Args:
            input_path: Path to input audio file
            output_path: Path for output processed file

        Returns:
            Path to the processed audio file

        Raises:
            FileNotFoundError: If input file doesn't exist
            RuntimeError: If FFmpeg conversion fails
        """
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(input_path),
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    "-c:a",
                    "flac",
                    "-compression_level",
                    "0",  # Faster encoding for chunks
                    "-map",
                    "0:a",  # Only map audio streams
                    "-avoid_negative_ts",
                    "make_zero",  # Handle timestamp issues
                    "-y",
                    str(output_path),
                ],
                check=True,
            )
            return output_path
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg conversion failed: {e.stderr}")

    # ----------------------------
    # Legacy compatibility methods (deprecated)
    # ----------------------------
    @staticmethod
    def _find_longest_common_sequence(sequences: list[str], match_by_words: bool = True) -> str:
        """
        Find longest common sequence between transcript chunks.

        NOTE: This method is deprecated and kept for compatibility.
        The new transcription providers handle chunk merging internally.
        """
        if not sequences:
            return ""
        if match_by_words:
            sequences = [
                [word for word in re.split(r"(\s+\w+)", seq) if word] for seq in sequences
            ]
        else:
            sequences = [list(seq) for seq in sequences]

        left_sequence = sequences[0]
        left_length = len(left_sequence)
        total_sequence = []

        for right_sequence in sequences[1:]:
            max_matching = 0.0
            right_length = len(right_sequence)
            max_indices = (left_length, left_length, 0, 0)

            for i in range(1, left_length + right_length + 1):
                eps = float(i) / 10000.0
                left_start = max(0, left_length - i)
                left_stop = min(left_length, left_length + right_length - i)
                left = left_sequence[left_start:left_stop]

                right_start = max(0, i - left_length)
                right_stop = min(right_length, i)
                right = right_sequence[right_start:right_stop]

                if len(left) != len(right):
                    raise RuntimeError("Mismatched subsequences detected during transcript merging.")

                matches = sum(a == b for a, b in zip(left, right))
                matching = matches / float(i) + eps

                if matches > 1 and matching > max_matching:
                    max_matching = matching
                    max_indices = (left_start, left_stop, right_start, right_stop)

            left_start, left_stop, right_start, right_stop = max_indices
            left_mid = (left_stop + left_start) // 2
            right_mid = (right_stop + right_start) // 2

            total_sequence.extend(left_sequence[:left_mid])
            left_sequence = right_sequence[right_mid:]
            left_length = len(left_sequence)

        total_sequence.extend(left_sequence)
        return "".join(total_sequence) if not match_by_words else "".join(total_sequence)
import time
import subprocess
import re
import io
from pathlib import Path

import torch
import torchaudio
import soundfile as sf
from groq import Groq, RateLimitError


class AudioEngine:
    """
    AudioEngine provides methods for audio preprocessing and transcription using Whisper through Groq's API.
    """

    def __init__(self, groq_client: Groq):
        self.groq_client = groq_client

    # ----------------------------
    # Public orchestrator
    # ----------------------------
    def transcribe_audio_in_chunks(
        self,
        audio_path: Path,
        transcription_dir: Path,
        chunk_length: int,
        overlap: int,
        language: str,
    ) -> dict:
        print(f"\nStarting transcription of: {audio_path}")

        processed_path = None
        try:
            # Preprocess to 16kHz mono FLAC
            processed_path = self._preprocess_audio(
                input_path=Path(audio_path),
                output_path=Path(transcription_dir) / "processed.flac",
            )

            # Load with torchaudio
            waveform, sample_rate = torchaudio.load(processed_path)
            duration_s = waveform.shape[1] / sample_rate
            print(f"Audio duration: {duration_s:.2f}s")

            # Chunk params
            chunk_samples = chunk_length * sample_rate
            overlap_samples = overlap * sample_rate
            step = chunk_samples - overlap_samples
            total_chunks = (waveform.shape[1] // step) + 1
            print(f"Processing {total_chunks} chunks...")

            results = []
            total_transcription_time = 0

            for i in range(total_chunks):
                start = i * step
                end = min(start + chunk_samples, waveform.shape[1])

                print(f"\nProcessing chunk {i + 1}/{total_chunks}")
                print(f"Sample range: {start} - {end}")

                # Slice waveform
                chunk_waveform = waveform[:, start:end]

                # Transcribe
                result, chunk_time = self._transcribe_single_chunk(
                    client=self.groq_client,
                    waveform=chunk_waveform,
                    sample_rate=sample_rate,
                    chunk_num=i + 1,
                    total_chunks=total_chunks,
                    language=language,
                )
                total_transcription_time += chunk_time
                results.append((result, start / sample_rate * 1000))  # offset in ms

            final_result = self._merge_transcripts(results=results)
            print(f"\nTotal Groq API transcription time: {total_transcription_time:.2f}s")

            return final_result

        finally:
            if processed_path and Path(processed_path).exists():
                Path(processed_path).unlink(missing_ok=True)

    # ----------------------------
    # Audio preprocessing
    # ----------------------------
    @staticmethod
    def _preprocess_audio(input_path: Path, output_path: Path) -> Path:
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
                    "-y",
                    str(output_path),
                ],
                check=True,
            )
            return output_path
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg conversion failed: {e.stderr}")

    # ----------------------------
    # Transcribe one chunk (in memory)
    # ----------------------------
    @staticmethod
    def _transcribe_single_chunk(
        client: Groq,
        waveform: torch.Tensor,
        sample_rate: int,
        chunk_num: int,
        total_chunks: int,
        language: str,
    ) -> tuple[dict, float]:
        total_api_time = 0

        while True:
            # Save tensor → BytesIO as FLAC
            buffer = io.BytesIO()
            sf.write(buffer, waveform.squeeze(0).numpy(), sample_rate, format="FLAC")
            buffer.seek(0)

            start_time = time.time()
            try:
                result = client.audio.transcriptions.create(
                    file=("chunk.flac", buffer, "audio/flac"),
                    model="whisper-large-v3",
                    language=language,
                    response_format="verbose_json",
                    timestamp_granularities=["word"],
                    temperature=0.0,
                )
                api_time = time.time() - start_time
                total_api_time += api_time

                print(f"Chunk {chunk_num}/{total_chunks} processed in {api_time:.2f}s")
                return result, total_api_time

            except RateLimitError:
                print(f"\nRate limit hit for chunk {chunk_num} - retrying in 60 seconds...")
                time.sleep(60)
                continue

            except Exception as e:
                print(f"Error transcribing chunk {chunk_num}: {str(e)}")
                raise

    # ----------------------------
    # Merge utilities
    # ----------------------------
    @staticmethod
    def _find_longest_common_sequence(sequences: list[str], match_by_words: bool = True) -> str:
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

    def _merge_transcripts(self, results: list[tuple[dict, int]]) -> dict:
        print("\nMerging results...")

        def to_dict(obj):
            return obj.model_dump() if hasattr(obj, "model_dump") else obj

        def extract_text(chunk):
            data = to_dict(chunk)
            return data.get("text", "") if isinstance(data, dict) else getattr(chunk, "text", "")

        def extract_segments(chunk):
            data = to_dict(chunk)
            return data.get("segments", []) if isinstance(data, dict) else getattr(chunk, "segments", [])

        def extract_words(chunk, start_offset_s):
            data = to_dict(chunk)
            raw_words = data.get("words", []) if isinstance(data, dict) else getattr(chunk, "words", [])
            return [
                {
                    "word": w.get("word", ""),
                    "start": w.get("start", 0) + start_offset_s,
                    "end": w.get("end", 0) + start_offset_s,
                }
                for w in raw_words
            ]

        all_texts, all_words, processed_chunks = [], [], []
        has_segments = any("segments" in to_dict(c) and to_dict(c)["segments"] for c, _ in results)

        for i, (chunk, chunk_start_ms) in enumerate(results):
            chunk_start_s = chunk_start_ms / 1000
            all_texts.append(extract_text(chunk))
            all_words.extend(extract_words(chunk, chunk_start_s))

            if has_segments:
                segments = extract_segments(chunk)
                for s in segments:
                    s["start"] += chunk_start_s
                    s["end"] += chunk_start_s
                processed_chunks.append(segments)

        merged_text = " ".join(all_texts)

        return {
            "text": merged_text,
            "segments": processed_chunks if has_segments else [],
            "words": all_words,
        }

    def transcribe_audio_with_output(self, audio_file: str, transcription_dir: str, language: str) -> dict:
        """Transcribe audio file in chunks and save the result."""
        from app.core.logging import get_logger
        from app.utils.utils import save_data

        logger = get_logger(__name__)
        logger.info("Transcribing audio...")

        result = self.transcribe_audio_in_chunks(
            audio_path=Path(audio_file),
            transcription_dir=transcription_dir,
            chunk_length=60,  # seconds
            overlap=0,  # seconds
            language=language
        )

        save_data(data=result, base_path=transcription_dir, file_name="transcription")
        return result

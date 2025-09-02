from groq import Groq, RateLimitError
from pydub import AudioSegment
from pathlib import Path

import time
import subprocess
import os
import tempfile
import re


class AudioEngine:
    """
    AudioEngine provides methods for audio preprocessing and transcription using Whisper through Groq's API.

    This class is responsible for:
    - Preprocessing audio files (e.g., converting to 16kHz mono FLAC format)
    - Splitting audio into overlapping chunks for efficient transcription
    - Transcribing audio chunks using the Groq API (Whisper model)
    - Merging chunked transcriptions into a single, coherent transcript
    - Saving transcription results in various formats (text, JSON, segments)

    Attributes:
        groq_client (Groq):
            A required instance of the Groq client for interacting with the Groq API.

    Methods:
        _preprocess_audio(input_path, output_path):
            Converts an audio file to 16kHz mono FLAC using ffmpeg.
        _transcribe_single_chunk(groq_client, chunk, chunk_num, total_chunks):
            Transcribes a single audio chunk using the Groq API.
        transcribe_audio_in_chunks(audio_path, chunk_length=600, overlap=10):
            Orchestrates the process of preprocessing, chunking, transcribing, and merging audio transcriptions.
        _find_longest_common_sequence(sequences, match_by_words=True):
            Finds the optimal alignment between sequences for merging overlapping transcriptions.
        _merge_transcripts(results):
            Merges chunked transcription results, handling overlaps and word/segment alignment.
        _save_results(result, audio_path):
            Saves the final transcription results to disk in multiple formats.
    """
    def __init__(self, groq_client: Groq):
        self.groq_client = groq_client


    # Orchestrator method for transcribing audio
    def transcribe_audio_in_chunks(
            self,
            audio_path: Path,
            transcription_dir: Path,
            chunk_length: int,
            overlap: int,
            language: str
    ) -> dict:
        """
        Transcribe audio in chunks with overlap with Whisper via Groq API.

        Args:
            audio_path: Path to audio file
            transcription_dir: Directory to save transcription results
            chunk_length: Length of each chunk in seconds
            overlap: Overlap between chunks in seconds
            language: Language code for transcription (e.g., "en" for English)

        Returns:
            dict: Containing transcription results

        Raises:
            ValueError: If Groq API key is not set
            RuntimeError: If the audio file fails to load
        """
        print(f"\nStarting transcription of: {audio_path}")

        processed_path = None
        try:
            # Preprocess audio and get basic info
            with tempfile.NamedTemporaryFile(suffix='.flac', delete=False) as temp_file:
                processed_path = self._preprocess_audio(input_path=Path(audio_path), output_path=Path(temp_file.name))
            try:
                audio = AudioSegment.from_file(processed_path, format="flac")
            except Exception as e:
                raise RuntimeError(f"Failed to load audio: {str(e)}")

            duration = len(audio)
            print(f"Audio duration: {duration / 1000:.2f}s")

            # Calculate # of chunks
            chunk_ms = chunk_length * 1000
            overlap_ms = overlap * 1000
            total_chunks = (duration // (chunk_ms - overlap_ms)) + 1
            print(f"Processing {total_chunks} chunks...")

            results = []
            total_transcription_time = 0

            # Loop through each chunk, extract current chunk from audio, transcribe
            for i in range(total_chunks):
                start = i * (chunk_ms - overlap_ms)
                end = min(start + chunk_ms, duration)

                print(f"\nProcessing chunk {i + 1}/{total_chunks}")
                print(f"Time range: {start / 1000:.1f}s - {end / 1000:.1f}s")

                chunk = audio[start:end]
                result, chunk_time = self._transcribe_single_chunk(
                    client=self.groq_client, 
                    chunk=chunk, 
                    chunk_num=i + 1, 
                    total_chunks=total_chunks
                )
                total_transcription_time += chunk_time
                results.append((result, start))

            final_result = self._merge_transcripts(results=results)
            print(f"\nTotal Groq API transcription time: {total_transcription_time:.2f}s")

            return final_result

        # Clean up temp files regardless of successful creation
        finally:
            if processed_path:
                Path(processed_path).unlink(missing_ok=True)


    @staticmethod
    def _preprocess_audio(input_path: Path, output_path: Path) -> Path:
        """
        Preprocess the audio file to 16 kHz mono FLAC using ffmpeg.
        Saves the result at specified output_path.
        Args:
            input_path: Path to input audio file
            output_path: Path to save processed audio file
        Returns:
            output_path: Path to the processed audio file
        """
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        try:
            subprocess.run([
                'ffmpeg',
                '-hide_banner',
                '-loglevel', 'error',
                '-i', str(input_path),
                '-ar', '16000',
                '-ac', '1',
                '-c:a', 'flac',
                '-y',
                str(output_path)
            ], check=True)
            return output_path
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg conversion failed: {e.stderr}")


    @staticmethod
    def _transcribe_single_chunk(
        client: Groq, 
        chunk: AudioSegment, 
        chunk_num: int, 
        total_chunks: int,
        language: str
    ) -> tuple[dict, float]:
        """
        Transcribe a single audio chunk with Groq API.

        Args:
            client: Groq client instance
            chunk: Audio segment to transcribe
            chunk_num: Current chunk number
            total_chunks: Total number of chunks
            language: Language code for transcription (e.g., "en" for English)

        Returns:
            Tuple of (transcription result, processing time)

        Raises:
            Exception: If chunk transcription fails after retries
        """
        total_api_time = 0

        while True:
            with tempfile.NamedTemporaryFile(suffix='.flac', delete=False) as temp_file:
                temp_file_path = temp_file.name
                chunk.export(temp_file_path, format='flac')

                start_time = time.time()
                try:
                    result = client.audio.transcriptions.create(
                        file=("chunk.flac", temp_file, "audio/flac"),
                        model="whisper-large-v3",
                        language=language,
                        response_format="verbose_json",
                        timestamp_granularities=["word"],
                        temperature=0.0
                    )
                    api_time = time.time() - start_time
                    total_api_time += api_time

                    print(f"Chunk {chunk_num}/{total_chunks} processed in {api_time:.2f}s")
                    return result, total_api_time

                except RateLimitError as e:
                    print(f"\nRate limit hit for chunk {chunk_num} - retrying in 60 seconds...")
                    time.sleep(60)
                    continue

                except Exception as e:
                    print(f"Error transcribing chunk {chunk_num}: {str(e)}")
                    raise

            # Clean up the temporary file after transcription attempt
            os.remove(temp_file_path)


    @staticmethod
    def _find_longest_common_sequence(sequences: list[str], match_by_words: bool = True) -> str:
        """
        Find the optimal alignment between sequences with longest common sequence and sliding window matching.

        Args:
            sequences: List of text sequences to align and merge
            match_by_words: Whether to match by words (True) or characters (False)

        Returns:
            str: Merged sequence with optimal alignment

        Raises:
            RuntimeError: If there's a mismatch in sequence lengths during comparison
        """
        if not sequences:
            return ""

        # Convert input based on matching strategy
        if match_by_words:
            # word-based splitting
            sequences = [
                [word for word in re.split(r'(\s+\w+)', seq) if word]
                for seq in sequences
            ]
        else:
            # character-based splitting
            sequences = [list(seq) for seq in sequences]

        left_sequence = sequences[0]
        left_length = len(left_sequence)
        total_sequence = []

        for right_sequence in sequences[1:]:
            max_matching = 0.0
            right_length = len(right_sequence)
            max_indices = (left_length, left_length, 0, 0)

            # Try different alignments
            for i in range(1, left_length + right_length + 1):
                # Add epsilon to favor longer matches
                eps = float(i) / 10000.0

                left_start = max(0, left_length - i)
                left_stop = min(left_length, left_length + right_length - i)
                left = left_sequence[left_start:left_stop]

                right_start = max(0, i - left_length)
                right_stop = min(right_length, i)
                right = right_sequence[right_start:right_stop]

                if len(left) != len(right):
                    raise RuntimeError(
                        "Mismatched subsequences detected during transcript merging."
                    )

                matches = sum(a == b for a, b in zip(left, right))

                # Normalize matches by position and add epsilon
                matching = matches / float(i) + eps

                # Require at least 2 matches
                if matches > 1 and matching > max_matching:
                    max_matching = matching
                    max_indices = (left_start, left_stop, right_start, right_stop)

            # Use the best alignment found
            left_start, left_stop, right_start, right_stop = max_indices

            # Take left half from left sequence and right half from right sequence
            left_mid = (left_stop + left_start) // 2
            right_mid = (right_stop + right_start) // 2

            total_sequence.extend(left_sequence[:left_mid])
            left_sequence = right_sequence[right_mid:]
            left_length = len(left_sequence)

        # Add remaining sequence
        total_sequence.extend(left_sequence)

        # Join back into text
        if match_by_words:
            return ''.join(total_sequence)
        return ''.join(total_sequence)

    def _merge_transcripts(self, results: list[tuple[dict, int]]) -> dict:
        """
        Merge transcription chunks and handle overlaps.
        Returns full text, segments, and word-level timestamps.
        """
        print("\nMerging results...")

        def to_dict(obj):
            return obj.model_dump() if hasattr(obj, 'model_dump') else obj

        def extract_text(chunk):
            data = to_dict(chunk)
            return data.get('text', '') if isinstance(data, dict) else getattr(chunk, 'text', '')

        def extract_segments(chunk):
            data = to_dict(chunk)
            segments = data.get('segments', []) if isinstance(data, dict) else getattr(chunk, 'segments', [])
            adjusted = []
            for segment in segments:
                seg = to_dict(segment)
                adjusted.append({
                    'text': seg.get('text', ''),
                    'start': seg.get('start', 0),
                    'end': seg.get('end', 0)
                })
            return adjusted

        def extract_words(chunk, start_offset_s):
            data = to_dict(chunk)
            raw_words = data.get('words', []) if isinstance(data, dict) else getattr(chunk, 'words', [])
            adjusted_words = []
            for word in raw_words:
                w = to_dict(word)
                adjusted_words.append({
                    'word': w.get('word', ''),
                    'start': w.get('start', 0) + start_offset_s,
                    'end': w.get('end', 0) + start_offset_s
                })
            return adjusted_words

        has_segments = any('segments' in to_dict(c) and to_dict(c)['segments'] for c, _ in results)
        all_words = []
        all_texts = []
        processed_chunks = []

        for i, (chunk, chunk_start_ms) in enumerate(results):
            chunk_start_s = chunk_start_ms / 1000
            all_texts.append(extract_text(chunk))

            # Always collect words
            words = extract_words(chunk, chunk_start_s)
            all_words.extend(words)

            # Only process segments if present
            if has_segments:
                segments = extract_segments(chunk)
                for s in segments:
                    s['start'] += chunk_start_s
                    s['end'] += chunk_start_s

                if i < len(results) - 1:
                    next_start_ms = results[i + 1][1]
                    current, overlaps = [], []

                    for s in segments:
                        if s['end'] * 1000 > next_start_ms:
                            overlaps.append(s)
                        else:
                            current.append(s)

                    if overlaps:
                        merged = overlaps[0].copy()
                        merged.update({
                            'text': ' '.join(s['text'] for s in overlaps),
                            'end': overlaps[-1]['end']
                        })
                        current.append(merged)

                    processed_chunks.append(current)
                else:
                    processed_chunks.append(segments)

        # Merge final segments
        final_segments = []
        for i in range(len(processed_chunks) - 1):
            if not processed_chunks[i] or not processed_chunks[i + 1]:
                continue

            final_segments.extend(processed_chunks[i][:-1])

            last = processed_chunks[i][-1]
            first = processed_chunks[i + 1][0]

            merged_text = self._find_longest_common_sequence([
                last['text'],
                first['text']
            ])

            merged = last.copy()
            merged.update({
                'text': merged_text,
                'end': first['end']
            })
            final_segments.append(merged)

        # Append last chunk’s segments
        if processed_chunks:
            final_segments.extend(processed_chunks[-1])

        # Merge all texts regardless of segment presence
        merged_text = ' '.join(all_texts)

        return {
            "text": merged_text,
            "segments": final_segments if has_segments else [],
            "words": all_words  # 🔵 ALWAYS INCLUDE
        }


from typing import List
import asyncio

from app.pydantic_models.shorts.short_model import ShortModel
from app.pydantic_models.shorts.shorts_response_model import ShortsResponseModel
from app.providers.llm.base_llm_provider import LLMProvider, LLMProviderError


class LLMEngine:

    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider


    @staticmethod
    def create_llm_input_format(word_transcription: dict) -> str:
        """
        Create a formatted string for shorts input.
        Raw words -> sentences (by punctuation) -> formatted sentences with timestamps.

        Args:
            word_transcription (dict): Dictionary containing words and their timestamps.

        Returns:
            str: Formatted sentences with timestamps.
        """
        sentence_endings = {".", "!", "?"}
        sentences = []
        current_sentence = []
        sentence_start_time = None

        # Iterate every word and create sentences from it
        for word in word_transcription["words"]:
            word_text = word["word"]
            start_time = word["start"]
            end_time = word["end"]

            # Reset sentence start time for a new sentence
            if sentence_start_time is None:
                sentence_start_time = start_time

            # Add the current word to the current sentence
            current_sentence.append({
                "word": word_text,
                "start": start_time,
                "end": end_time
            })

            # Check if a sentence is final
            if word_text[-1] in sentence_endings:
                sentence_text = " ".join(w["word"] for w in current_sentence)
                sentence_start_formatted = f"{sentence_start_time:.2f}"
                sentence_end_formatted = f"{current_sentence[-1]["end"]:.2f}"
                sentences.append({
                    "start": sentence_start_formatted,
                    "end": sentence_end_formatted,
                    "text": sentence_text
                })
                current_sentence = []
                sentence_start_time = None

        # If there are any sentences left
        if current_sentence:
            sentence_text = " ".join(w["word"] for w in current_sentence)
            sentence_start_formatted = f"{sentence_start_time:.2f}"
            sentence_end_formatted = f"{current_sentence[-1]["end"]:.2f}"
            sentences.append({
                "start": sentence_start_formatted,
                "end": sentence_end_formatted,
                "text": sentence_text
            })

        # Format sentences for output
        output: list[str] = []
        for s in sentences:
            line = f'[start: {s["start"]}, end: {s["end"]}] => "{s["text"]}"'
            output.append(line)

        formatted_output = "\n".join(output)
        return formatted_output


    async def llm_inference(
        self,
        llm_model: str,
        system_prompt: str,
        user_prompt: str
    ) -> List[ShortModel]:
        """
        Get highlight moments from transcription using shorts.

        Args:
            llm_model (str): The model to use for inference.
            system_prompt (str): The system prompt to guide the shorts's behavior.
            user_prompt (str): The prompt to send to the shorts.

        Returns:
            List[ShortModel]: The response from the shorts containing highlight moments.
        """
        try:
            # Use the abstracted provider for inference
            structured_response = await self.llm_provider.generate_structured_response(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=ShortsResponseModel,
                model=llm_model
            )

            return structured_response.highlights

        except LLMProviderError as e:
            from app.core.logging import get_logger
            logger = get_logger(__name__)
            logger.error(f"shorts inference failed: {e}")
            raise

    async def extract_highlights(self, llm_input: str, max_reels: int, llm_dir: str):
        """Extract highlight moments using shorts with organized prompt system."""
        from app.core.logging import get_logger
        from app.utils.utils import save_data
        from app.prompts.manager import prompt_manager

        logger = get_logger(__name__)
        logger.info("Extracting highlights using shorts...")

        # Get formatted prompts from prompt manager
        prompts = prompt_manager.get_highlight_extraction_prompts(
            max_reels=max_reels,
            llm_input=llm_input
        )

        highlight_moments = await self.llm_inference(
            llm_model=None,  # Use default model from provider configuration
            system_prompt=prompts['system'],
            user_prompt=prompts['user']
        )

        # Convert Pydantic pydantic_models to dict for JSON serialization
        highlight_moments_dict = [moment.model_dump() for moment in highlight_moments]
        save_data(data=highlight_moments_dict, base_path=llm_dir, file_name="highlight_moments")

        return highlight_moments

    async def extract_micro_highlights_from_chunk(self, chunk_content: str, llm_dir: str,
                                                 chunk_index: int) -> List[ShortModel]:
        """Extract micro-highlights from a single chunk."""
        from app.core.logging import get_logger
        from app.utils.utils import save_data

        logger = get_logger(__name__)
        logger.info(f"Extracting micro-highlights from chunk {chunk_index}")

        # Load micro-highlight prompts
        system_prompt = self._load_micro_highlight_prompt('system')
        user_prompt = self._load_micro_highlight_prompt('user')

        # Format prompts
        formatted_user = user_prompt.format(llm_input=chunk_content)

        try:
            # Get micro-highlights from chunk
            micro_highlights = await self.llm_inference(
                llm_model=None,
                system_prompt=system_prompt,
                user_prompt=formatted_user
            )

            # Save chunk results for debugging
            chunk_results = {
                'chunk_index': chunk_index,
                'micro_highlights': [highlight.model_dump() for highlight in micro_highlights],
                'chunk_content_preview': chunk_content[:500] + '...' if len(chunk_content) > 500 else chunk_content
            }
            save_data(data=chunk_results, base_path=llm_dir, file_name=f"chunk_{chunk_index}_micro_highlights")

            # Validate Phase 1 extraction results
            valid_highlights = self._validate_phase1_extraction(micro_highlights, chunk_index)

            logger.info(f"Extracted {len(valid_highlights)} valid micro-highlights from chunk {chunk_index} (original: {len(micro_highlights)})")
            return valid_highlights

        except Exception as e:
            logger.error(f"Failed to extract micro-highlights from chunk {chunk_index}: {e}")
            return []

    def _validate_phase1_extraction(self, micro_highlights: List[ShortModel], chunk_index: int) -> List[ShortModel]:
        """Validate Phase 1 extraction results and log issues."""
        from app.core.logging import get_logger
        logger = get_logger(__name__)

        valid_highlights = []
        issues = []

        # Check extraction count
        if len(micro_highlights) > 5:
            issues.append(f"Too many extractions: {len(micro_highlights)} (expected 2-5)")
            micro_highlights = micro_highlights[:5]  # Truncate to 5
        elif len(micro_highlights) < 2:
            issues.append(f"Too few extractions: {len(micro_highlights)} (expected 2-5)")

        # Check each highlight
        for i, highlight in enumerate(micro_highlights):
            highlight_issues = []

            # Check duration (8-90 seconds, but warn instead of reject for 5-8s)
            duration = highlight.end - highlight.start
            if duration < 5:
                highlight_issues.append(f"Invalid duration: {duration:.2f}s (expected 8-90s)")
            elif duration < 8:
                # Don't reject, just log for monitoring
                logger.debug(f"Short highlight detected: {duration:.2f}s (prefer 15-45s)")
            elif duration > 90:
                highlight_issues.append(f"Invalid duration: {duration:.2f}s (expected 8-90s)")

            # Check for overlaps with previous highlights
            for j, prev_highlight in enumerate(valid_highlights):
                if highlight.start < prev_highlight.end:
                    highlight_issues.append(f"Overlaps with highlight {j}: starts at {highlight.start:.2f}s, previous ends at {prev_highlight.end:.2f}s")

            # Check chronological order
            if valid_highlights and highlight.start < valid_highlights[-1].start:
                highlight_issues.append(f"Not in chronological order: {highlight.start:.2f}s after {valid_highlights[-1].start:.2f}s")

            if not highlight_issues:
                valid_highlights.append(highlight)
            else:
                issues.extend([f"Highlight {i}: {issue}" for issue in highlight_issues])

        # Log validation results
        if issues:
            logger.warning(f"Chunk {chunk_index} validation issues: {'; '.join(issues)}")
        else:
            logger.debug(f"Chunk {chunk_index}: All {len(valid_highlights)} highlights passed validation")

        return valid_highlights

    def _load_micro_highlight_prompt(self, prompt_type: str) -> str:
        """Load Phase 1 scoring prompts with quality criteria."""
        from pathlib import Path
        import importlib.util

        prompts_dir = Path(__file__).parent.parent / 'prompts' / 'phase_1_scoring'

        if prompt_type == 'system':
            file_path = prompts_dir / 'system_prompt.py'
        elif prompt_type == 'user':
            file_path = prompts_dir / 'user_prompt.py'
        else:
            raise ValueError(f"Invalid prompt type: {prompt_type}")

        # Load the prompt from Python file
        spec = importlib.util.spec_from_file_location("prompt_module", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        return module.PROMPT

    async def chunked_extract_highlights(self, llm_input: str, max_reels: int, llm_dir: str) -> List[ShortModel]:
        """
        Extract highlights using chunked processing with overlaps.

        Args:
            llm_input: Formatted transcript string
            max_reels: Maximum number of final highlights to return
            llm_dir: Directory to save processing results

        Returns:
            List of final highlights
        """
        from app.engines.chunking_engine import ChunkingEngine, ChunkResult
        from app.core.logging import get_logger
        from app.utils.utils import save_data

        logger = get_logger(__name__)
        logger.info("Starting chunked highlight extraction...")

        # Initialize chunked extraction engine
        chunk_engine = ChunkingEngine(chunk_size=60, overlap_size=20)

        # Parse transcript into lines
        transcript_lines = chunk_engine.parse_transcript_lines(llm_input)
        logger.info(f"Parsed transcript into {len(transcript_lines)} lines")

        # Create chunks with overlaps
        chunks = chunk_engine.create_chunks_with_overlaps(transcript_lines)
        logger.info(f"Created {len(chunks)} chunks for processing")

        # Process each chunk to extract micro-highlights
        all_chunk_results = []
        for chunk in chunks:
            chunk_content = chunk_engine.format_chunk_for_llm(chunk)

            # Extract micro-highlights from this chunk
            micro_highlights = await self.extract_micro_highlights_from_chunk(
                chunk_content, llm_dir, chunk.chunk_index
            )

            # Create chunk result
            chunk_result = ChunkResult(
                chunk_index=chunk.chunk_index,
                highlights=micro_highlights,
                main_region_start=chunk.main_region_start_line,
                main_region_end=chunk.main_region_end_line
            )

            # Trim any highlights that bleed into overlap regions
            valid_highlights = chunk_engine.trim_overlap_highlights(chunk_result, transcript_lines)
            chunk_result.highlights = valid_highlights

            all_chunk_results.append(chunk_result)

        # Collect all micro-highlights
        all_micro_highlights = []
        for chunk_result in all_chunk_results:
            all_micro_highlights.extend(chunk_result.highlights)

        logger.info(f"Collected {len(all_micro_highlights)} micro-highlights from all chunks")

        # Save all micro-highlights for debugging
        micro_highlights_data = [highlight.model_dump() for highlight in all_micro_highlights]
        save_data(data=micro_highlights_data, base_path=llm_dir, file_name="all_micro_highlights")

        # Merge close highlights
        merged_highlights = chunk_engine.merge_close_highlights(all_micro_highlights, merge_threshold=3.0)
        logger.info(f"Merged into {len(merged_highlights)} highlights")

        # Calculate video duration for quality phase_2_scoring
        video_duration = None
        if transcript_lines:
            try:
                _, video_duration = chunk_engine.extract_timestamp_from_line(transcript_lines[-1])
            except Exception:
                logger.warning("Could not determine video duration from transcript")

        # Select top highlights based on LLM-powered quality phase_2_scoring
        final_highlights = await chunk_engine.select_top_highlights(
            merged_highlights, max_reels, self.llm_provider, video_duration
        )

        # Save final results
        final_highlights_data = [highlight.model_dump() for highlight in final_highlights]
        save_data(data=final_highlights_data, base_path=llm_dir, file_name="final_chunked_highlights")

        # Save processing summary
        processing_summary = {
            'total_transcript_lines': len(transcript_lines),
            'total_chunks': len(chunks),
            'total_micro_highlights': len(all_micro_highlights),
            'merged_highlights': len(merged_highlights),
            'final_highlights': len(final_highlights),
            'video_duration_seconds': video_duration,
            'chunk_size': chunk_engine.chunk_size,
            'overlap_size': chunk_engine.overlap_size
        }
        save_data(data=processing_summary, base_path=llm_dir, file_name="chunked_processing_summary")

        logger.info(f"Chunked extraction complete: {len(final_highlights)} final highlights")
        return final_highlights


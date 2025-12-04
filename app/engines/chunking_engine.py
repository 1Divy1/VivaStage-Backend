from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from app.pydantic_models.shorts.short_model import ShortModel
from app.core.logging import get_logger
from app.providers.llm.base_llm_provider import LLMProvider
import re
import asyncio

logger = get_logger(__name__)


@dataclass
class ChunkResult:
    """Container for highlights extracted from a single chunk."""
    chunk_index: int
    highlights: List[ShortModel]
    main_region_start: int
    main_region_end: int


@dataclass
class TranscriptChunk:
    """Represents a chunk of transcript with overlap regions."""
    chunk_index: int
    previous_overlap: List[str]  # Lines for context only
    main_region: List[str]       # Lines to extract highlights from
    next_overlap: List[str]      # Lines for context only
    main_region_start_line: int  # Global line index where main region starts
    main_region_end_line: int    # Global line index where main region ends


class ChunkingEngine:
    """Engine for processing transcripts in chunks with overlaps."""

    def __init__(self, chunk_size: int = 60, overlap_size: int = 20):
        """
        Initialize the chunked extraction engine.

        Args:
            chunk_size: Number of sentences per main chunk region
            overlap_size: Number of sentences for overlap context
        """
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size

    def parse_transcript_lines(self, formatted_transcript: str) -> List[str]:
        """
        Parse the formatted transcript into individual lines.

        Args:
            formatted_transcript: The transcript in format:
                [start: 45.23, end: 47.89] => "This is a sentence."

        Returns:
            List of transcript lines
        """
        lines = formatted_transcript.strip().split('\n')
        # Filter out empty lines
        return [line.strip() for line in lines if line.strip()]

    def extract_timestamp_from_line(self, line: str) -> Tuple[float, float]:
        """
        Extract start and end timestamps from a transcript line.

        Args:
            line: Transcript line like '[start: 45.23, end: 47.89] => "Text"'

        Returns:
            Tuple of (start_time, end_time)
        """
        pattern = r'\[start: ([\d.]+), end: ([\d.]+)\]'
        match = re.search(pattern, line)
        if match:
            return float(match.group(1)), float(match.group(2))
        else:
            raise ValueError(f"Could not extract timestamps from line: {line}")

    def create_chunks_with_overlaps(self, transcript_lines: List[str]) -> List[TranscriptChunk]:
        """
        Split transcript lines into chunks with overlaps.

        Args:
            transcript_lines: List of formatted transcript lines

        Returns:
            List of TranscriptChunk objects
        """
        total_lines = len(transcript_lines)
        chunks = []
        chunk_index = 0

        # Start from the beginning
        current_start = 0

        while current_start < total_lines:
            # Calculate main region boundaries
            main_start = current_start
            main_end = min(current_start + self.chunk_size, total_lines)

            # Calculate overlap boundaries
            prev_overlap_start = max(0, main_start - self.overlap_size)
            next_overlap_end = min(total_lines, main_end + self.overlap_size)

            # Extract regions
            previous_overlap = transcript_lines[prev_overlap_start:main_start] if prev_overlap_start < main_start else []
            main_region = transcript_lines[main_start:main_end]
            next_overlap = transcript_lines[main_end:next_overlap_end] if main_end < next_overlap_end else []

            # Create chunk
            chunk = TranscriptChunk(
                chunk_index=chunk_index,
                previous_overlap=previous_overlap,
                main_region=main_region,
                next_overlap=next_overlap,
                main_region_start_line=main_start,
                main_region_end_line=main_end - 1
            )

            chunks.append(chunk)
            logger.debug(f"Created chunk {chunk_index}: main region lines {main_start}-{main_end-1}, "
                        f"prev overlap: {len(previous_overlap)}, next overlap: {len(next_overlap)}")

            # Move to next chunk
            current_start = main_end
            chunk_index += 1

            # Break if we've processed all lines
            if main_end >= total_lines:
                break

        logger.info(f"Created {len(chunks)} chunks from {total_lines} transcript lines")
        return chunks

    def format_chunk_for_llm(self, chunk: TranscriptChunk) -> str:
        """
        Format a chunk for LLM processing with clear region markers.

        Args:
            chunk: TranscriptChunk to format

        Returns:
            Formatted string for LLM input
        """
        formatted_parts = []

        # Previous overlap (context only)
        if chunk.previous_overlap:
            formatted_parts.append("[PREVIOUS_OVERLAP - CONTEXT ONLY, DO NOT EXTRACT FROM HERE]")
            formatted_parts.extend(chunk.previous_overlap)
            formatted_parts.append("[/PREVIOUS_OVERLAP]")
            formatted_parts.append("")  # Empty line for separation

        # Main region (extraction target)
        formatted_parts.append("[MAIN_CHUNK - EXTRACT HIGHLIGHTS FROM THIS REGION ONLY]")
        formatted_parts.extend(chunk.main_region)
        formatted_parts.append("[/MAIN_CHUNK]")

        # Next overlap (context only)
        if chunk.next_overlap:
            formatted_parts.append("")  # Empty line for separation
            formatted_parts.append("[NEXT_OVERLAP - CONTEXT ONLY, DO NOT EXTRACT FROM HERE]")
            formatted_parts.extend(chunk.next_overlap)
            formatted_parts.append("[/NEXT_OVERLAP]")

        return '\n'.join(formatted_parts)

    def merge_close_highlights(self, highlights: List[ShortModel], merge_threshold: float = 2.0) -> List[ShortModel]:
        """
        Merge highlights that are close together in time.

        Args:
            highlights: List of highlights to merge
            merge_threshold: Maximum gap in seconds to merge highlights

        Returns:
            List of merged highlights
        """
        if not highlights:
            return []

        # Sort by start time
        sorted_highlights = sorted(highlights, key=lambda h: h.start)
        merged = []

        current_highlight = sorted_highlights[0]

        for next_highlight in sorted_highlights[1:]:
            # Check if highlights are close enough to merge
            gap = next_highlight.start - current_highlight.end

            if gap <= merge_threshold:
                # Merge highlights
                merged_text = f"{current_highlight.text} {next_highlight.text}".strip()
                merged_reason = f"{current_highlight.reason}; {next_highlight.reason}"

                current_highlight = ShortModel(
                    start=current_highlight.start,
                    end=next_highlight.end,
                    text=merged_text,
                    reason=merged_reason
                )
                logger.debug(f"Merged highlights: {current_highlight.start:.2f}-{current_highlight.end:.2f}")
            else:
                # No merge needed, save current and move to next
                merged.append(current_highlight)
                current_highlight = next_highlight

        # Add the last highlight
        merged.append(current_highlight)

        logger.info(f"Merged {len(highlights)} highlights into {len(merged)} highlights")
        return merged

    def calculate_quality_score(self, highlight: ShortModel, video_duration: Optional[float] = None) -> float:
        """
        Calculate quality score for a highlight.

        Args:
            highlight: ShortModel to score
            video_duration: Total video duration for position phase_2_scoring

        Returns:
            Quality score between 0-1
        """
        duration = highlight.end - highlight.start

        # Duration score (prefer 30-60 seconds)
        if 30 <= duration <= 60:
            duration_score = 1.0
        elif 15 <= duration <= 90:
            duration_score = 0.8
        elif 10 <= duration <= 15 or 90 <= duration <= 120:
            duration_score = 0.6
        else:
            duration_score = 0.3

        # Position score (slightly prefer middle of video)
        if video_duration and video_duration > 0:
            position_ratio = highlight.start / video_duration
            if 0.2 <= position_ratio <= 0.8:
                position_score = 1.0
            else:
                position_score = 0.8
        else:
            position_score = 0.8

        # Reason length score (longer reason = more engagement)
        reason_length = len(highlight.reason) if highlight.reason else 0
        if reason_length > 50:
            reason_score = 1.0
        elif reason_length > 20:
            reason_score = 0.8
        else:
            reason_score = 0.6

        # Calculate weighted score
        total_score = (
            duration_score * 0.4 +
            position_score * 0.2 +
            reason_score * 0.2 +
            0.2  # Base score for spacing (will be calculated later if needed)
        )

        return min(total_score, 1.0)

    async def select_top_highlights(self, highlights: List[ShortModel], max_highlights: int,
                                  llm_provider: LLMProvider, video_duration: Optional[float] = None) -> List[ShortModel]:
        """
        Select the top N highlights based on LLM-powered quality phase_2_scoring.

        Phase 2 scoring ALWAYS runs to populate reason fields with explanations,
        regardless of highlight count.

        Args:
            highlights: List of highlights to select from
            max_highlights: Maximum number of highlights to return
            llm_provider: LLM provider for phase_2_scoring
            video_duration: Total video duration for better phase_2_scoring

        Returns:
            List of top quality highlights with Phase 2 explanations
        """
        if not highlights:
            return highlights

        # Calculate LLM-based scores for all highlights
        scored_highlights = []

        # Use ScoringEngine to evaluate each highlight
        for highlight in highlights:
            try:
                from app.engines.scoring_engine import ScoringEngine

                scoring_engine = ScoringEngine(highlight.text, llm_provider)
                score = await scoring_engine.general_score()
                scored_highlights.append((highlight, score, scoring_engine))

                logger.debug(f"Scored highlight {highlight.start:.2f}-{highlight.end:.2f}: {score:.3f}")

            except Exception as e:
                logger.error(f"Error phase_2_scoring highlight {highlight.start:.2f}-{highlight.end:.2f}: {e}")
                # Fallback to computational phase_2_scoring
                fallback_score = self.calculate_quality_score(highlight, video_duration)
                scored_highlights.append((highlight, fallback_score, None))

        # Sort by score (highest first)
        scored_highlights.sort(key=lambda x: x[1], reverse=True)

        # Phase 2 ALWAYS runs: Update reason fields for ALL highlights first
        all_highlights_with_explanations = []
        for highlight, score, scoring_engine in scored_highlights:
            # Update reason field with Phase 2 explanation
            if scoring_engine:
                try:
                    explanation = await scoring_engine.generate_explanation()
                    highlight.reason = explanation
                except Exception as e:
                    logger.error(f"Error generating explanation for highlight {highlight.start:.2f}-{highlight.end:.2f}: {e}")
                    highlight.reason = f"Phase 2 scoring: Selected with score {score:.2f}"
            else:
                # Fallback explanation
                highlight.reason = f"Phase 2 scoring: Selected with fallback score {score:.2f}"

            all_highlights_with_explanations.append((highlight, score))

        # Select top N highlights (but all now have explanations)
        if len(all_highlights_with_explanations) <= max_highlights:
            # Keep all highlights since we're under the limit
            selected = [highlight for highlight, score in all_highlights_with_explanations]
            logger.info(f"Phase 2 scoring complete: Keeping all {len(selected)} highlights (under max of {max_highlights})")
        else:
            # Select top N highlights based on scores
            selected = [highlight for highlight, score in all_highlights_with_explanations[:max_highlights]]
            logger.info(f"Phase 2 scoring complete: Selected top {len(selected)} highlights from {len(highlights)} candidates")

        # Sort selected highlights by start time for chronological order
        selected.sort(key=lambda h: h.start)

        # Log scoring results
        for i, (highlight, score) in enumerate(all_highlights_with_explanations[:len(selected)]):
            logger.debug(f"Final #{i+1}: {highlight.start:.2f}-{highlight.end:.2f} (score: {score:.3f})")

        # Apply strict minimum length enforcement
        enforced_highlights = self.enforce_minimum_lengths(selected, min_length=8.0)

        return enforced_highlights

    def enforce_minimum_lengths(self, highlights: List[ShortModel], min_length: float = 8.0,
                               transcript_lines: List[str] = None) -> List[ShortModel]:
        """
        Strictly enforce minimum length requirements. Merge or remove highlights under 8 seconds.

        Args:
            highlights: List of highlights to enforce minimum lengths on
            min_length: Minimum length in seconds (default: 8.0)
            transcript_lines: Original transcript lines for context-aware extension

        Returns:
            List of highlights where ALL are >= min_length seconds
        """
        if not highlights:
            return highlights

        logger.info(f"Enforcing strict {min_length}s minimum length requirement on {len(highlights)} highlights")

        # Sort highlights by start time for processing
        sorted_highlights = sorted(highlights, key=lambda h: h.start)

        # Multi-pass enforcement
        enforced_highlights = self._multi_pass_merge_enforcement(sorted_highlights, min_length)

        # Final validation - remove any highlights that still don't meet minimum
        final_highlights = []
        removed_count = 0

        for highlight in enforced_highlights:
            duration = highlight.end - highlight.start
            if duration >= min_length:
                final_highlights.append(highlight)
            else:
                removed_count += 1
                logger.warning(f"REMOVED highlight {highlight.start:.2f}-{highlight.end:.2f} ({duration:.1f}s) - could not extend to {min_length}s minimum")

        # Log enforcement results
        original_count = len(highlights)
        final_count = len(final_highlights)
        logger.info(f"Length enforcement complete: {original_count} → {final_count} highlights ({removed_count} removed)")

        if removed_count > 0:
            logger.warning(f"Removed {removed_count} highlights that could not meet {min_length}s minimum requirement")

        # Verify no highlights are under minimum
        for highlight in final_highlights:
            duration = highlight.end - highlight.start
            if duration < min_length:
                logger.error(f"ENFORCEMENT FAILED: Highlight {highlight.start:.2f}-{highlight.end:.2f} is {duration:.1f}s < {min_length}s")

        return final_highlights

    def _multi_pass_merge_enforcement(self, highlights: List[ShortModel], min_length: float) -> List[ShortModel]:
        """
        Perform multiple passes of merging to enforce minimum lengths.

        Args:
            highlights: Sorted list of highlights
            min_length: Minimum required length

        Returns:
            List of highlights after aggressive merging
        """
        if not highlights:
            return highlights

        logger.debug(f"Starting multi-pass merge enforcement with {len(highlights)} highlights")

        # Pass 1: Merge consecutive short highlights
        current_highlights = self._merge_consecutive_short_highlights(highlights, min_length)
        logger.debug(f"Pass 1 (consecutive merge): {len(highlights)} → {len(current_highlights)} highlights")

        # Pass 2: Extend short highlights by merging with closest neighbors
        current_highlights = self._extend_short_highlights_with_neighbors(current_highlights, min_length)
        logger.debug(f"Pass 2 (neighbor extension): → {len(current_highlights)} highlights")

        # Pass 3: Aggressive merge with larger gaps for remaining short highlights
        current_highlights = self._aggressive_merge_remaining_short(current_highlights, min_length)
        logger.debug(f"Pass 3 (aggressive merge): → {len(current_highlights)} highlights")

        return current_highlights

    def _merge_consecutive_short_highlights(self, highlights: List[ShortModel], min_length: float) -> List[ShortModel]:
        """Merge consecutive highlights that are both under minimum length."""
        if len(highlights) <= 1:
            return highlights

        merged = []
        i = 0

        while i < len(highlights):
            current = highlights[i]
            current_duration = current.end - current.start

            # If current highlight is long enough, keep it as is
            if current_duration >= min_length:
                merged.append(current)
                i += 1
                continue

            # Current is short, try to merge with next short highlights
            merge_candidates = [current]
            j = i + 1

            while j < len(highlights):
                next_highlight = highlights[j]
                next_duration = next_highlight.end - next_highlight.start
                gap = next_highlight.start - merge_candidates[-1].end

                # Merge if next is also short and gap is reasonable (≤5 seconds)
                if next_duration < min_length and gap <= 5.0:
                    merge_candidates.append(next_highlight)
                    j += 1
                else:
                    break

            # Create merged highlight from candidates
            if len(merge_candidates) > 1:
                merged_text = " ".join(h.text for h in merge_candidates)
                merged_reason = "; ".join(h.reason for h in merge_candidates if h.reason)

                merged_highlight = ShortModel(
                    start=merge_candidates[0].start,
                    end=merge_candidates[-1].end,
                    text=merged_text,
                    reason=merged_reason
                )

                merged_duration = merged_highlight.end - merged_highlight.start
                logger.debug(f"Merged {len(merge_candidates)} consecutive short highlights into {merged_duration:.1f}s segment")
                merged.append(merged_highlight)
            else:
                # Single short highlight, keep for next pass
                merged.append(current)

            i = j

        return merged

    def _extend_short_highlights_with_neighbors(self, highlights: List[ShortModel], min_length: float) -> List[ShortModel]:
        """Extend short highlights by merging with their closest neighbors."""
        if len(highlights) <= 1:
            return highlights

        extended = []
        processed_indices = set()

        for i, highlight in enumerate(highlights):
            if i in processed_indices:
                continue

            duration = highlight.end - highlight.start

            if duration >= min_length:
                extended.append(highlight)
                processed_indices.add(i)
                continue

            # Find best neighbor to merge with (shortest gap)
            best_merge = None
            best_gap = float('inf')
            best_neighbor_idx = None

            # Check previous neighbor
            if i > 0 and (i-1) not in processed_indices:
                prev_highlight = highlights[i-1]
                gap = highlight.start - prev_highlight.end
                if gap <= 10.0 and gap < best_gap:  # Increased gap tolerance for better merging
                    best_gap = gap
                    best_neighbor_idx = i-1
                    best_merge = self._create_merged_highlight([prev_highlight, highlight])

            # Check next neighbor
            if i < len(highlights)-1 and (i+1) not in processed_indices:
                next_highlight = highlights[i+1]
                gap = next_highlight.start - highlight.end
                if gap <= 10.0 and gap < best_gap:
                    best_gap = gap
                    best_neighbor_idx = i+1
                    best_merge = self._create_merged_highlight([highlight, next_highlight])

            if best_merge:
                extended.append(best_merge)
                processed_indices.add(i)
                processed_indices.add(best_neighbor_idx)
                logger.debug(f"Extended short highlight by merging with neighbor (gap: {best_gap:.1f}s)")
            else:
                # No suitable neighbor, keep for next pass
                extended.append(highlight)
                processed_indices.add(i)

        return extended

    def _aggressive_merge_remaining_short(self, highlights: List[ShortModel], min_length: float) -> List[ShortModel]:
        """Aggressively merge any remaining short highlights with larger gap tolerance."""
        if len(highlights) <= 1:
            return highlights

        final = []
        processed_indices = set()

        for i, highlight in enumerate(highlights):
            if i in processed_indices:
                continue

            duration = highlight.end - highlight.start

            if duration >= min_length:
                final.append(highlight)
                processed_indices.add(i)
                continue

            # Aggressively find ANY neighbor within 15 seconds to merge with
            merged_with_neighbor = False

            # Try merging with any unprocessed highlight within range
            for j, other_highlight in enumerate(highlights):
                if j == i or j in processed_indices:
                    continue

                # Calculate gap between highlights
                if other_highlight.start > highlight.end:
                    gap = other_highlight.start - highlight.end
                elif highlight.start > other_highlight.end:
                    gap = highlight.start - other_highlight.end
                else:
                    gap = 0  # Overlapping

                # Merge if gap is within aggressive threshold (8 seconds)
                if gap <= 8.0:
                    highlights_to_merge = sorted([highlight, other_highlight], key=lambda h: h.start)
                    merged = self._create_merged_highlight(highlights_to_merge)
                    final.append(merged)
                    processed_indices.add(i)
                    processed_indices.add(j)
                    merged_with_neighbor = True
                    logger.debug(f"Aggressively merged short highlight with distant neighbor (gap: {gap:.1f}s)")
                    break

            if not merged_with_neighbor:
                # Keep short highlight for final removal
                final.append(highlight)
                processed_indices.add(i)

        return final

    def _create_merged_highlight(self, highlights_to_merge: List[ShortModel]) -> ShortModel:
        """Create a single merged highlight from multiple highlights."""
        if not highlights_to_merge:
            raise ValueError("Cannot merge empty list of highlights")

        # Sort by start time
        sorted_highlights = sorted(highlights_to_merge, key=lambda h: h.start)

        merged_text = " ".join(h.text for h in sorted_highlights)
        merged_reason = "; ".join(h.reason for h in sorted_highlights if h.reason)

        return ShortModel(
            start=sorted_highlights[0].start,
            end=sorted_highlights[-1].end,
            text=merged_text,
            reason=merged_reason
        )

    def trim_overlap_highlights(self, chunk_result: ChunkResult, transcript_lines: List[str]) -> List[ShortModel]:
        """
        Remove highlights that start or end in overlap regions.

        Args:
            chunk_result: Result from chunk processing
            transcript_lines: Original transcript lines for timestamp validation

        Returns:
            Filtered highlights that are within the main region
        """
        if not chunk_result.highlights:
            return []

        # Get main region time boundaries
        main_start_line = chunk_result.main_region_start
        main_end_line = chunk_result.main_region_end

        if main_start_line >= len(transcript_lines) or main_end_line >= len(transcript_lines):
            logger.warning(f"Invalid line indices for chunk {chunk_result.chunk_index}")
            return chunk_result.highlights

        main_start_time, _ = self.extract_timestamp_from_line(transcript_lines[main_start_line])
        _, main_end_time = self.extract_timestamp_from_line(transcript_lines[main_end_line])

        valid_highlights = []
        for highlight in chunk_result.highlights:
            # Check if highlight is within main region boundaries
            if highlight.start >= main_start_time and highlight.end <= main_end_time:
                valid_highlights.append(highlight)
            else:
                logger.debug(f"Trimmed highlight {highlight.start:.2f}-{highlight.end:.2f} "
                           f"outside main region {main_start_time:.2f}-{main_end_time:.2f}")

        logger.debug(f"Trimmed {len(chunk_result.highlights) - len(valid_highlights)} highlights "
                    f"from chunk {chunk_result.chunk_index}")
        return valid_highlights
from pyparsing import OnlyOnce

from app.engines.audio_engine import AudioEngine
from app.engines.caption_engine import CaptionEngine
from app.pydantic_models.shorts.generate_shorts_request_model import GenerateShortsRequestModel
from app.engines.llm_engine import LLMEngine
from app.engines.video_engine import VideoEngine
from app.utils.utils import save_data, prepare_output_dirs
from app.utils.video_utils import calculate_max_reels, format_duration
from app.core.logging import get_logger

from pathlib import Path
import shutil

logger = get_logger(__name__)


class ReelService:
    """
    Service for orchestrating reel extraction.

    Args:
        llm_engine: Engine for large language model interactions.
        video_engine: Engine for video/audio download and merging.
        audio_engine: Service for audio processing tasks.
        caption_engine: Engine for caption generation.
    """
    def __init__(
        self,
        llm_engine: LLMEngine,
        video_engine: VideoEngine,
        audio_engine: AudioEngine,
        caption_engine: CaptionEngine
    ):
        self.llm_engine = llm_engine
        self.video_engine = video_engine
        self.audio_engine = audio_engine
        self.caption_engine = caption_engine

    async def process_reel(self, reel_data_input: GenerateShortsRequestModel):
        """THE PIPELINE - Process a reel extraction request through the complete pipeline."""

        # TESTING CONFIGURATION - Toggle this for testing
        ONLY_LLM_INFERENCE = False  # Set to True to skip video download/transcription
        ONLY_VIDEO_PROCESSING = True  # Set to True to skip LLM and use existing highlights
        EXISTING_TRANSCRIPT_PATH = "output/20251129_133931_deff2767/llm/prompt_formatted_input.txt"  # Update this path for testing
        EXISTING_HIGHLIGHTS_PATH = "output/20251204_090105_ef452c65/llm/final_chunked_highlights.json"  # Update this path for testing
        EXISTING_VIDEO_PATH = "output/20251204_090105_ef452c65/video/final_video.mp4"  # Update this path for testing

        logger.info("Starting reel processing...")

        # Helper function to extract video duration from transcript
        def get_duration_from_transcript(transcript_content: str) -> float:
            """Extract video duration from transcript by finding the last timestamp."""
            import re
            lines = transcript_content.strip().split('\n')
            if not lines:
                return 300.0  # fallback
            last_line = lines[-1]
            # Parse format: [start: X.XX, end: Y.YY] => "text"
            match = re.search(r'end: (\d+\.\d+)', last_line)
            if match:
                return float(match.group(1))
            return 300.0  # fallback

        if ONLY_VIDEO_PROCESSING:
            # VIDEO-ONLY TESTING MODE: Load existing highlights and process video
            import json

            # Setup new directories for this video processing test
            base_dir, video_dir, audio_dir, transcription_dir, llm_dir, clips_dir, shorts_dir = prepare_output_dirs()

            # Load existing highlights
            logger.info(f"VIDEO PROCESSING MODE: Loading highlights from {EXISTING_HIGHLIGHTS_PATH}")
            with open(EXISTING_HIGHLIGHTS_PATH, 'r', encoding='utf-8') as f:
                highlights_data = json.load(f)

            # Convert to highlight moment objects (simplified)
            from app.pydantic_models.shorts.short_model import ShortModel
            highlight_moments = []
            for h in highlights_data:
                highlight_moments.append(ShortModel(
                    start=h["start"],
                    end=h["end"],
                    text=h["text"],
                    reason=h.get("reason", "Video processing test")
                ))

            # Use existing video file instead of downloading
            video_audio_result = {"video_file": EXISTING_VIDEO_PATH}

            # Set variables for final response
            duration_seconds = 600.0  # Fallback duration for video processing mode
            formatted_duration = format_duration(duration_seconds)
            max_reels = len(highlight_moments)
            transcription_result = None  # Skip captions in video-only mode

            logger.info(f"VIDEO PROCESSING MODE: Processing {len(highlight_moments)} highlights from existing video")
        elif ONLY_LLM_INFERENCE:
            # TESTING MODE: Create new directories but use existing transcript
            # Step 1: Setup directories
            base_dir, video_dir, audio_dir, transcription_dir, llm_dir, clips_dir, shorts_dir = prepare_output_dirs()

            # Load existing transcript
            with open(EXISTING_TRANSCRIPT_PATH, 'r', encoding='utf-8') as f:
                llm_input = f.read()

            # Calculate duration and max_reels from transcript
            duration_seconds = get_duration_from_transcript(llm_input)
            max_reels = calculate_max_reels(int(duration_seconds))

            logger.info(f"TESTING MODE: Using existing transcript, duration={duration_seconds}s, max_reels={max_reels}")

            # Step 5: Extract highlights using chunked processing
            highlight_moments = await self.llm_engine.chunked_extract_highlights(llm_input, max_reels, llm_dir)

            logger.info("TESTING MODE: Stopping after highlight extraction")
            return {
                "highlights": [moment.model_dump() for moment in highlight_moments],
                "test_output_dir": llm_dir,
                "max_reels": max_reels,
                "duration_seconds": duration_seconds
            }
        else:
            # Normal pipeline: Steps 1-4
            # Step 1: Setup directories
            base_dir, video_dir, audio_dir, transcription_dir, llm_dir, clips_dir, shorts_dir = prepare_output_dirs()

            # Step 2: Download and process video/audio
            video_audio_result = self.video_engine.download_and_process_video(
                str(reel_data_input.youtube_url),
                video_dir,
                audio_dir
            )

            # Step 2.1: Calculate max reels based on video duration
            duration_seconds = video_audio_result["duration_seconds"]
            max_reels = calculate_max_reels(duration_seconds)
            formatted_duration = format_duration(duration_seconds)

            logger.info(f"Video duration: {formatted_duration} ({duration_seconds}s) → Max reels: {max_reels}")

            # Step 3: Transcribe audio
            transcription_result = await self.audio_engine.transcribe_audio_with_output(
                video_audio_result["audio_file"],
                transcription_dir,
                reel_data_input.language
            )

            # Step 4: Convert raw Whisper transcript into an LLM "digestible" format
            llm_input = self.llm_engine.create_llm_input_format(word_transcription=transcription_result)
            save_data(data=llm_input, base_path=llm_dir, file_name="prompt_formatted_input")

            # Step 5: Extract highlights using chunked processing
            highlight_moments = await self.llm_engine.chunked_extract_highlights(llm_input, max_reels, llm_dir)

        # Step 7: Cut video clips based on highlights
        logger.info("Cutting video clips from highlights...")
        clip_timestamps = [(moment.start, moment.end) for moment in highlight_moments]
        self.video_engine.cut_clips(
            video_audio_result["video_file"],
            clip_timestamps,
            clips_dir
        )

        # Step 8: Process each clip into 9:16 shorts with face detection
        logger.info("Processing clips into 9:16 shorts...")
        short_video_paths = []
        for i, moment in enumerate(highlight_moments):
            clip_path = Path(clips_dir) / f"clip_{i+1}.mp4"
            short_path = Path(shorts_dir) / f"short_{i+1}.mp4"

            if clip_path.exists():
                self.video_engine.create_short_from_clip(clip_path, short_path)
                short_video_paths.append(str(short_path))
            else:
                logger.warning(f"Clip file not found: {clip_path}")

        # Step 9: Add captions if requested
        final_video_paths = []
        if reel_data_input.captions:
            logger.info("Adding captions to shorts...")
            for i, (moment, short_path) in enumerate(zip(highlight_moments, short_video_paths)):
                captioned_path = Path(shorts_dir) / f"short_captioned_{i+1}.mp4"
                try:
                    self.caption_engine.create_captions_for_short(
                        video_path=Path(short_path),
                        output_path=captioned_path,
                        word_transcription=transcription_result,
                        start_time=moment.start,
                        end_time=moment.end
                    )
                    final_video_paths.append(str(captioned_path))
                except Exception as e:
                    logger.warning(f"Failed to add captions to short {i+1}: {e}")
                    final_video_paths.append(short_path)  # Fallback to non-captioned
        else:
            final_video_paths = short_video_paths

        # Step 10: Cleanup - Remove clips folder
        logger.info("Cleaning up temporary clips folder...")
        try:
            if Path(clips_dir).exists():
                shutil.rmtree(clips_dir)
                logger.info(f"Removed clips folder: {clips_dir}")
        except Exception as e:
            logger.warning(f"Failed to remove clips folder: {e}")

        # Extract processing ID from base_dir (format: output/{timestamp}_{uuid})
        processing_id = Path(base_dir).name

        # Build comprehensive response with file paths and metadata
        shorts_info = []
        for i, (moment, short_path) in enumerate(zip(highlight_moments, final_video_paths)):
            short_path_obj = Path(short_path)
            # Calculate relative path from workspace root
            try:
                # Try to get relative path from current working directory
                relative_path = str(short_path_obj.relative_to(Path.cwd()))
            except ValueError:
                # If that fails, construct it manually from base_dir
                relative_path = str(Path(base_dir) / "shorts" / short_path_obj.name)
            
            shorts_info.append({
                "index": i + 1,
                "file_path": str(short_path_obj.absolute()),
                "relative_path": relative_path,
                "file_name": short_path_obj.name,
                "start_time": moment.start,
                "end_time": moment.end,
                "duration": round(moment.end - moment.start, 2),
                "text": moment.text,
                "reason": moment.reason,
                "exists": short_path_obj.exists()
            })

        return {
            "status": "completed",
            "message": "Reel processing completed successfully",
            "processing_id": processing_id,
            "base_directory": str(Path(base_dir).absolute()),
            "shorts_directory": str(Path(shorts_dir).absolute()),
            "shorts": shorts_info,
            "total_shorts": len(final_video_paths),
            "metadata": {
                "youtube_url": str(reel_data_input.youtube_url),
                "video_duration": formatted_duration,
                "duration_seconds": duration_seconds,
                "max_reels_allowed": max_reels,
                "number_generated": len(final_video_paths),
                "language": reel_data_input.language,
                "captions_enabled": reel_data_input.captions
            }
        }


from app.engines.audio_engine import AudioEngine
from app.engines.caption_engine import CaptionEngine
from app.models.reel_job import ReelJob
from app.engines.llm_engine import LLMEngine
from app.engines.video_engine import VideoEngine
from app.helpers.pipeline_helper import PipelineHelper
from app.utils.utils import save_data
from app.core.logging import get_logger

from datetime import datetime
from pathlib import Path

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

    def process_reel(self, reel_data_input: ReelJob):
        """THE PIPELINE - Process a reel extraction request through the complete pipeline."""
        logger.info(f"Starting reel processing for {reel_data_input.number_of_reels} reels")

        # Step 1: Setup directories
        base_dir, video_dir, audio_dir, transcription_dir, llm_dir, clips_dir, shorts_dir = PipelineHelper.prepare_output_dirs()

        # Step 2: Download and process video/audio
        video_audio_result = self.video_engine.download_and_process_video(
            str(reel_data_input.youtube_url),
            video_dir,
            audio_dir
        )

        # Step 3: Transcribe audio
        transcription_result = self.audio_engine.transcribe_audio_with_output(
            video_audio_result["audio_file"],
            transcription_dir,
            reel_data_input.language
        )

        # Step 4: Extract highlights using LLM
        llm_input = self.llm_engine.create_llm_input_format(word_transcription=transcription_result)
        save_data(data=llm_input, base_path=llm_dir, file_name="prompt_formatted_input")

        highlight_moments = self.llm_engine.extract_highlights(llm_input, reel_data_input, llm_dir)

        # Step 5: Cut video clips based on highlights
        logger.info("Cutting video clips from highlights...")
        clip_timestamps = [(moment.start, moment.end) for moment in highlight_moments]
        self.video_engine.cut_clips(
            video_audio_result["video_file"],
            clip_timestamps,
            clips_dir
        )

        # Step 6: Process each clip into 9:16 shorts with face detection
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

        # Step 7: Add captions if requested
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

        highlight_moments_dict = [moment.model_dump() for moment in highlight_moments]
        
        # Return result with processing information
        return {
            "status": "completed",
            "message": "Reel processing completed successfully",
            "highlights": highlight_moments_dict,
            "video_urls": final_video_paths,
            "processed_at": datetime.utcnow().isoformat(),
            "processing_details": {
                "number_of_reels": reel_data_input.number_of_reels,
                "min_seconds": reel_data_input.min_seconds,
                "max_seconds": reel_data_input.max_seconds,
                "youtube_url": str(reel_data_input.youtube_url),
                "language": reel_data_input.language,
                "captions_enabled": reel_data_input.captions,
                "videos_generated": len(final_video_paths)
            }
        }


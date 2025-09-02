from app.engines.audio_engine import AudioEngine
from app.engines.caption_engine import CaptionEngine
from app.models.reel_job import ReelJob
from app.engines.llm_engine import LLMEngine
from app.engines.video_engine import VideoEngine
from app.utils.utils import prepare_output_dirs, save_data
from app.core.logging import get_logger

import os
import json
from pathlib import Path
from datetime import datetime

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

    def process_reel(self, reel_data_input: ReelJob, user_context: dict = None):
        """Process a reel extraction request through the complete pipeline."""
        logger.info(f"Starting reel processing for {reel_data_input.number_of_reels} reels")
        
        # Log user context if provided
        if user_context:
            logger.info(f"Processing reel for user: {user_context.get('user_id')} ({user_context.get('email')})")
            logger.info(f"User premium status: {user_context.get('is_premium', False)}")
        
        # Step 1: Setup directories
        base_dir, video_dir, audio_dir, transcription_dir, llm_dir, clips_dir, shorts_dir = prepare_output_dirs()
        
        # Step 2: Download and process video/audio
        video_audio_result = self._download_and_process_video(reel_data_input, video_dir, audio_dir)
        
        # Step 3: Transcribe audio
        transcription_result = self._transcribe_audio(
            video_audio_result["audio_file"], 
            transcription_dir, 
            reel_data_input.language
        )
        
        # Step 4: Extract highlights using LLM
        llm_input = self.llm_engine.create_llm_input_format(word_transcription=transcription_result)
        save_data(data=llm_input, base_path=llm_dir, file_name="prompt_formatted_input")
        
        highlight_moments = self._extract_highlights(llm_input, reel_data_input, llm_dir)
        
        # TODO: Implement remaining pipeline steps:
        # - Cut video clips based on highlights
        # - Create 9:16 shorts with face detection
        # - Add captions if requested
        # - Return video URLs
        
        highlight_moments_dict = [moment.model_dump() for moment in highlight_moments]
        
        # Return result with processing information
        return {
            "status": "completed",
            "message": "Highlight extraction completed successfully", 
            "highlights": highlight_moments_dict,
            "processed_at": datetime.utcnow().isoformat(),
            "processing_details": {
                "number_of_reels": reel_data_input.number_of_reels,
                "min_seconds": reel_data_input.min_seconds,
                "max_seconds": reel_data_input.max_seconds,
                "youtube_url": reel_data_input.youtube_url,
                "language": reel_data_input.language,
                "captions_enabled": reel_data_input.captions
            }
        }

    def _download_and_process_video(self, reel_data_input: ReelJob, video_dir: str, audio_dir: str) -> dict:
        """Download video and audio from YouTube and merge them."""
        logger.info("Downloading video and audio...")
        
        video_audio_result = self.video_engine.download_video_and_audio(
            reel_data_input.youtube_url,
            video_dir, 
            audio_dir
        )
        
        logger.info("Merging video and audio...")
        final_video_path = os.path.join(video_dir, "final_video.mp4")
        final_video_file = self.video_engine.merge_video_and_audio(
            video_audio_result["video_file"],
            video_audio_result["audio_file"],
            final_video_path
        )
        
        return {
            "video_file": final_video_file,
            "audio_file": video_audio_result["audio_file"]
        }

    def _transcribe_audio(self, audio_file: str, transcription_dir: str, language: str) -> dict:
        """Transcribe audio file in chunks."""
        logger.info("Transcribing audio...")
        
        result = self.audio_engine.transcribe_audio_in_chunks(
            audio_path=Path(audio_file),
            transcription_dir=transcription_dir,
            chunk_length=60,  # seconds
            overlap=0,  # seconds
            language=language
        )
        
        save_data(data=result, base_path=transcription_dir, file_name="transcription")
        return result

    def _extract_highlights(self, llm_input: str, reel_data_input: ReelJob, llm_dir: str):
        """Extract highlight moments using LLM."""
        logger.info("Extracting highlights using LLM...")
        
        user_prompt = f"""
        Transcript:
        \"\"\"
        {llm_input}
        \"\"\"

        Please extract the first {reel_data_input.number_of_reels} most interesting and engaging highlight moments.

        - Each highlight must be **at least {reel_data_input.min_seconds} seconds**.
        - The preferred maximum is {reel_data_input.max_seconds} seconds, but it may be slightly exceeded (by up to 10s) if needed to preserve meaning.
        - Never return highlights below the minimum duration.
        """

        system_prompt = f"""
        You are a strict JSON generator and expert transcript analyst.

        Your task is to extract the most interesting and engaging highlight segments from a transcript of a video.

        You must obey the following rules:

        1. Each highlight must:
           - Be **at least {reel_data_input.min_seconds} seconds long**
           - **Ideally** be no longer than {reel_data_input.max_seconds} seconds, but you may go slightly over (up to ~10 seconds) if the highlight would otherwise be incomplete or incoherent.
           - Have a correctly calculated duration from `end - start` and never be shorter than the minimum under any condition.
           - Combine adjacent sentences if needed.
           - You must **skip** any segment that cannot satisfy the minimum time requirement.

        2. Do not explain your answer, do not include any commentary, markdown, or formatting.

        3. Format all times with two decimal places (e.g. "45.36").

        4. Return exactly {reel_data_input.number_of_reels} highlights (as requested by the user), unless there are not enough valid segments in the transcript.

        5. The "reason" field explaining why a segment is a highlight **must be written in the same language as the transcript text**.

        You are given the following fallback permissions:

        - You may cut off a sentence if doing so allows the segment to reach at least the minimum duration. When cutting off, make sure the segment ends naturally, without including a period (.), exclamation mark (!) or question mark (?) at the end.
        - Commas (,) at the end of the text are acceptable when a sentence is cut.

        You must strictly respect the minimum duration. Segments shorter than {reel_data_input.min_seconds} seconds are never allowed.
        """
        
        highlight_moments = self.llm_engine.llm_inference(
            llm_model="gpt-5-mini",
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )
        
        # Convert Pydantic models to dict for JSON serialization
        highlight_moments_dict = [moment.model_dump() for moment in highlight_moments]
        save_data(data=highlight_moments_dict, base_path=llm_dir, file_name="highlight_moments")
        
        return highlight_moments
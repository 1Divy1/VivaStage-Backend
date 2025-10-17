import os
import uuid
from datetime import datetime


class PipelineHelper:
    """Helper class for pipeline coordination and directory management."""

    @staticmethod
    def prepare_output_dirs():
        """
        Create timestamped output directories for video processing pipeline.

        Returns:
            tuple: (base_dir, video_dir, audio_dir, transcription_dir, llm_dir, clips_dir, shorts_dir)
        """
        unique_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
        base_dir = os.path.join("output", unique_id)
        video_dir = os.path.join(base_dir, "video")
        audio_dir = os.path.join(base_dir, "audio")
        transcription_dir = os.path.join(base_dir, "transcription")
        llm_dir = os.path.join(base_dir, "llm")
        clips_dir = os.path.join(base_dir, "clips")
        shorts_dir = os.path.join(base_dir, "shorts")

        os.makedirs(video_dir, exist_ok=True)
        os.makedirs(audio_dir, exist_ok=True)
        os.makedirs(transcription_dir, exist_ok=True)
        os.makedirs(llm_dir, exist_ok=True)
        os.makedirs(clips_dir, exist_ok=True)
        os.makedirs(shorts_dir, exist_ok=True)

        return base_dir, video_dir, audio_dir, transcription_dir, llm_dir, clips_dir, shorts_dir
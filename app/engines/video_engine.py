import subprocess
import os
import cv2 as cv
import numpy as np

import mediapipe as mp
from numpy.linalg import norm
from pytubefix import YouTube
from datetime import datetime
from pathlib import Path
from moviepy import VideoFileClip

from app.utils.utils import logger


class VideoEngine:

    # Object level variables; they exist only for the current object instance
    def __init__(self):
        # Initialize MediaPipe FaceMesh
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            refine_landmarks=True,
            max_num_faces=3,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Tracking state variables for geometric similarity
        self.prev_signature = None
        self.prev_center = None
        self.prev_area = None
        self.smooth_center = None
        self.face_id_counter = 0

        # Store either a face centered frame or a full frame (if no face is detected)
        self.processed_frames: list[tuple[int, int, int | None, int | None]] = []


    def create_short_from_clip(self, video_file_path: Path, base_output_path: Path):
        """
        Orchestrator function for outputting 9:16 face-centered video with audio.
        """
        logger("Analyzing faces...")
        self._analyze_frames(video_file_path)

        logger("Building speaker segments...")
        segments = self._build_video_segments()

        logger("Cropping full video...")

        # 1. Create silent temp video
        temp_output_path = base_output_path.with_name(base_output_path.stem + "_silent.mp4")
        self._crop_segments_to_video(video_file_path, temp_output_path, segments)

        logger("Adding original audio to cropped video...")

        # 2. Add audio using ffmpeg
        self._mux_audio_with_ffmpeg(video_file_path, temp_output_path, base_output_path)

        # 3. Cleanup temp file
        temp_output_path.unlink(missing_ok=True)

        logger(f"Final video with audio saved to: {base_output_path}")


    def generate_signature(self, landmarks):
        """
        Generate a geometric signature from MediaPipe face landmarks.
        Uses key stable landmarks to create a normalized feature vector.
        """
        # Select key stable indices: nose tip, eye corners, cheek points, chin
        key_idx = [1, 33, 61, 199, 263, 291]

        # Use nose tip as reference point
        base_x = landmarks[1].x
        base_y = landmarks[1].y

        vec = []
        for idx in key_idx:
            dx = landmarks[idx].x - base_x
            dy = landmarks[idx].y - base_y
            vec.extend([dx, dy])

        # Convert to numpy array and normalize
        v = np.array(vec)
        return v / (norm(v) + 1e-6)


    def cosine_similarity(self, a, b):
        """
        Calculate cosine similarity between two vectors.
        """
        return float(np.dot(a, b) / (norm(a) * norm(b) + 1e-6))


    def _analyze_frames(self, video_path: Path):
        # Initialize video capture
        cap = cv.VideoCapture(str(video_path))

        fps = cap.get(cv.CAP_PROP_FPS)
        frame_idx = 0

        while True:
            success, frame = cap.read()
            if not success:
                break

            timestamp_sec = frame_idx / fps

            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            # Process frame with MediaPipe
            results = self.mp_face_mesh.process(rgb_frame)

            if not results.multi_face_landmarks or len(results.multi_face_landmarks) > 1:
                # No face or multiple faces detected - fallback
                self.processed_frames.append(int(timestamp_sec))
            else:
                center_x, face_id = self._analyze_face(frame=frame, landmarks=results.multi_face_landmarks[0])
                self.processed_frames.append((int(timestamp_sec), center_x, face_id))

            frame_idx += 1

        # Close the video file after finishing
        cap.release()


    def _analyze_face(self, frame, landmarks):
        """
        Analyze a single face using MediaPipe landmarks and geometric similarity.
        Returns center_x and face_id using weighted scoring system.
        """
        frame_height, frame_width = frame.shape[:2]

        # Extract face information from landmarks
        xs = [lm.x for lm in landmarks.landmark]
        ys = [lm.y for lm in landmarks.landmark]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        # Calculate face center in pixels
        cx = (min_x + max_x) / 2
        cy = (min_y + max_y) / 2
        center_x = int(cx * frame_width)

        # Calculate face area
        area = (max_x - min_x) * (max_y - min_y)

        # Generate geometric signature
        signature = self.generate_signature(landmarks.landmark)

        # Current face data
        current_face = {
            "center": (cx, cy),
            "area": area,
            "signature": signature
        }

        # Determine face ID using geometric similarity
        if self.prev_signature is None:
            # First face - assign ID 0
            face_id = 0
            self.prev_signature = signature
            self.prev_center = (cx, cy)
            self.prev_area = area
            self.smooth_center = np.array([cx, cy])
        else:
            # Calculate similarity score
            geo_sim = self.cosine_similarity(signature, self.prev_signature)

            # Size consistency (penalize dramatic size changes)
            area_ratio = area / (self.prev_area + 1e-6)
            size_consistency = 1 - abs(1 - area_ratio)
            size_consistency = max(0, size_consistency)  # Clamp to [0,1]

            # Position continuity (penalize large movements)
            center_dist = np.linalg.norm(
                np.array([cx, cy]) - np.array(self.prev_center)
            )
            position_continuity = max(0, 1 - center_dist)  # Clamp to [0,1]

            # Weighted score
            score = (
                0.55 * geo_sim +
                0.30 * size_consistency +
                0.15 * position_continuity
            )

            # If score is above threshold, same person; otherwise new person
            if score > 0.6:  # Threshold for same person
                face_id = 0  # Same person (simplified - we only track one speaker)
            else:
                self.face_id_counter += 1
                face_id = self.face_id_counter

            # Update tracking state
            self.prev_signature = signature
            self.prev_center = (cx, cy)
            self.prev_area = area

        # Apply EMA smoothing to center position
        if self.smooth_center is None:
            self.smooth_center = np.array([cx, cy])
        else:
            self.smooth_center = (
                0.2 * np.array([cx, cy]) +
                0.8 * self.smooth_center
            )

        # Use smoothed center for final center_x calculation
        smooth_center_x = int(self.smooth_center[0] * frame_width)

        return smooth_center_x, face_id


    def _build_video_segments(self, max_gap=2):
        """
        Builds a list of segments from processed frames:
        - If face is detected, creates cropped segments with center_x
        - If no or multiple faces, creates fallback segments
        """
        if not self.processed_frames:
            return []

        segments: list[tuple[int, int, int | None, int | None]] = []
        self.processed_frames.sort(key=lambda item: item if isinstance(item, int) else item[0])  # Sort by timestamp

        prev_face_id = None
        current_face_segment: list[tuple[int, int, int]] = []
        current_fallback_segment: list[int] = []

        for frame_data in self.processed_frames:
            if isinstance(frame_data, int):
                # This is a fallback (no or multiple faces)
                if current_face_segment:
                    # Flush any ongoing face segment
                    start = current_face_segment[0][0]
                    end = current_face_segment[-1][0]
                    average_center_x = int(np.mean([x for _, x, _ in current_face_segment]))
                    segments.append((start, end, prev_face_id, average_center_x))
                    current_face_segment.clear()

                # Handle fallback segment accumulation
                if not current_fallback_segment or (frame_data - current_fallback_segment[-1]) <= max_gap:
                    current_fallback_segment.append(frame_data)
                else:
                    # Flush fallback segment
                    segments.append((current_fallback_segment[0], current_fallback_segment[-1], None, None))
                    current_fallback_segment = [frame_data]

            else:
                timestamp, center_x, face_id = frame_data

                if current_fallback_segment:
                    # Flush fallback segment
                    segments.append((current_fallback_segment[0], current_fallback_segment[-1], None, None))
                    current_fallback_segment.clear()

                if not current_face_segment:
                    current_face_segment.append((timestamp, center_x, face_id))
                    prev_face_id = face_id
                else:
                    last_timestamp = current_face_segment[-1][0]
                    if face_id == prev_face_id and (timestamp - last_timestamp) <= max_gap:
                        current_face_segment.append((timestamp, center_x, face_id))
                    else:
                        # Flush previous face segment
                        start = current_face_segment[0][0]
                        end = current_face_segment[-1][0]
                        average_center_x = int(np.mean([x for _, x, _ in current_face_segment]))
                        segments.append((start, end, prev_face_id, average_center_x))

                        # Start new face segment
                        current_face_segment = [(timestamp, center_x, face_id)]
                        prev_face_id = face_id

        # Flush any remaining segments
        if current_face_segment:
            start = current_face_segment[0][0]
            end = current_face_segment[-1][0]
            average_center_x = int(np.mean([x for _, x, _ in current_face_segment]))
            segments.append((start, end, prev_face_id, average_center_x))

        if current_fallback_segment:
            segments.append((current_fallback_segment[0], current_fallback_segment[-1], None, None))

        return segments


    def _crop_segments_to_video(
            self,
            video_file_path: Path,
            output_path: Path,
            segments: list[tuple[int, int, int | None, int | None]]
    ):
        cap = cv.VideoCapture(str(video_file_path))
        fps = cap.get(cv.CAP_PROP_FPS)
        frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

        out_w = int(frame_height * 9 / 16)
        out_h = frame_height

        out = cv.VideoWriter(str(output_path), cv.VideoWriter_fourcc(*'mp4v'), fps, (out_w, out_h))

        # Map seconds to crop behavior
        second_to_crop_info = {}
        for start_sec, end_sec, face_id, center_x in segments:
            for sec in range(start_sec, end_sec + 1):
                second_to_crop_info[sec] = center_x  # could be None

        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

        for frame_idx in range(total_frames):
            success, frame = cap.read()
            if not success:
                break

            current_sec = int(frame_idx / fps)

            if current_sec in second_to_crop_info:
                center_x = second_to_crop_info[current_sec]

                if center_x is not None:
                    # Crop around face
                    x1, _, crop_width, crop_height = self._get_crop_rect(center_x, frame_width, frame_height)
                    cropped = frame[0:crop_height, x1:x1 + crop_width]
                    resized = cv.resize(cropped, (out_w, out_h))

                else:
                    # Letterbox full 16:9 frame inside 9:16
                    resized = self._resize_and_letterbox(frame, out_w, out_h)
            else:
                # Frame not part of any segment — fallback
                resized = self._resize_and_letterbox(frame, out_w, out_h)

            out.write(resized)

        cap.release()
        out.release()


    def _resize_and_letterbox(self, frame: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
        """
        Resize the frame to fit inside a 9:16 canvas, preserving the original aspect ratio.
        The result is centered on a black background.
        """
        original_h, original_w = frame.shape[:2]

        # Compute scale to fit frame inside the output size
        scale = min(out_w / original_w, out_h / original_h)
        new_w = int(original_w * scale)
        new_h = int(original_h * scale)

        # Resize frame with preserved aspect ratio
        resized_frame = cv.resize(frame, (new_w, new_h))

        # Create black canvas and center the resized frame
        black_bg = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        x_offset = (out_w - new_w) // 2
        y_offset = (out_h - new_h) // 2
        black_bg[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_frame

        return black_bg


    def _mux_audio_with_ffmpeg(self, original_video_path: Path, silent_video_path: Path, final_output_path: Path):
        """
        Merges audio from original video into the silent cropped video.
        """
        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(silent_video_path),
            "-i", str(original_video_path),
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            str(final_output_path)
        ]
        subprocess.run(cmd, check=True)


    @staticmethod
    def _get_crop_rect(center_x, frame_width, frame_height):
        """
        Returns a 9:16 crop box centered horizontally around center_x.
        Ensures crop stays within frame boundaries.
        """
        target_aspect = 9 / 16
        crop_height = frame_height
        crop_width = int(crop_height * target_aspect)

        # Calculate the initial top-left corner of the crop box
        x1 = center_x - crop_width // 2

        # Make sure the crop box stays within the frame boundaries
        if x1 < 0:
            x1 = 0
        elif x1 + crop_width > frame_width:
            x1 = frame_width - crop_width

        return x1, 0, crop_width, crop_height


    def merge_video_and_audio(
        self, 
        video_path, 
        audio_path, 
        final_video_path
    ) -> str:
        """
        Merges video and audio files into a single video file.
        Args:
            video_path (str): Path to the video file.
            audio_path (str): Path to the audio file.
            final_video_path (str): Path where the final merged video will be saved.
        Returns:
            str: Path to the final merged video file.
        """
        command = [
            "ffmpeg", "-y",  # overwrite output
            "-i", video_path,
            "-i", audio_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-strict", "experimental",
            final_video_path
        ]
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        return final_video_path


    def download_video_and_audio(
        self, 
        youtube_url, 
        video_dir, 
        audio_dir
    ) -> dict:
        """
        Downloads video and audio (separate) streams from a YouTube URL.
        Args:
            youtube_url (str): The YouTube video URL.
            video_dir (str): Directory to save the video file.
            audio_dir (str): Directory to save the audio file.
        Returns:
            dict: Contains paths to the downloaded video and audio files.
        """
        yt = YouTube(str(youtube_url))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Extract video duration in seconds
        video_duration_seconds = yt.length

        # Get all video-only MP4 streams
        video_streams = yt.streams.filter(only_video=True, file_extension="mp4").order_by("resolution").desc()
        audio_stream = yt.streams.filter(only_audio=True).order_by("abr").desc().first()
        if not video_streams or not audio_stream:
            raise Exception("Could not find suitable video/audio streams.")
        
        video_filename = f"video_only_{timestamp}.mp4"
        audio_filename = f"audio_only_{timestamp}.{audio_stream.subtype}"

        # Try to find 1080p stream
        video_1080p = video_streams.filter(res="1080p").first()
        if video_1080p:
            stream_to_download = video_1080p
        else:
            # Get the highest quality available BELOW 1080p
            stream_to_download = None
            for stream in video_streams:
                if int(stream.resolution.replace("p", "")) < 1080:
                    stream_to_download = stream
                    break

        video_file = None
        if stream_to_download:
            video_file = stream_to_download.download(output_path=video_dir, filename=video_filename)
        audio_file = audio_stream.download(output_path=audio_dir, filename=audio_filename)
        
        return {
            "video_file": video_file,
            "audio_file": audio_file,
            "duration_seconds": video_duration_seconds
        }


    def cut_clips(self, video_file_path: str, timestamps: list[tuple[float, float]], output_dir: str):
        """
        Cuts clips from a video file based on provided timestamps and saves them to the output directory.
        Args:
            video_file_path (str): Path to the video file.
            timestamps (list[tuple[float, float]]): List of tuples containing start and end times for each clip.
            output_dir (str): Directory where the clips will be saved.
        """
        for i in range(len(timestamps)):
            start: float = timestamps[i][0]
            end: float = timestamps[i][1]

            # Subclip the video
            clip = VideoFileClip(filename=video_file_path).subclipped(start_time=start, end_time=end)

            # Save the clip
            clip_filename: Path = Path(os.path.join(output_dir, f"clip_{i+1}.mp4"))
            clip.write_videofile(filename=clip_filename)

    def download_and_process_video(self, youtube_url: str, video_dir: str, audio_dir: str) -> dict:
        """Download video and audio from YouTube and merge them."""
        from app.core.logging import get_logger
        logger = get_logger(__name__)

        logger.info("Downloading video and audio...")

        video_audio_result = self.download_video_and_audio(
            youtube_url,
            video_dir,
            audio_dir
        )

        logger.info("Merging video and audio...")
        final_video_path = os.path.join(video_dir, "final_video.mp4")
        final_video_file = self.merge_video_and_audio(
            video_audio_result["video_file"],
            video_audio_result["audio_file"],
            final_video_path
        )

        return {
            "video_file": final_video_file,
            "audio_file": video_audio_result["audio_file"],
            "duration_seconds": video_audio_result["duration_seconds"]
        }


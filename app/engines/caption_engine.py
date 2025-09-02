import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple
import json

from app.utils.utils import logger


class CaptionEngine:

    def __init__(self):
        pass

    def extract_words_for_timerange(
            self,
            word_transcription: Dict,
            start_time: float,
            end_time: float
    ) -> List[Dict]:
        words_in_range = []
        for word in word_transcription.get("words", []):
            word_start, word_end = word["start"], word["end"]
            if word_start < end_time and word_end > start_time:
                words_in_range.append({
                    "word": word["word"],
                    "start": max(0, word_start - start_time),
                    "end": min(end_time - start_time, word_end - start_time)
                })
        return words_in_range

    def group_words_into_captions(
            self,
            words: List[Dict],
            max_words_per_caption: int = 3,
            min_caption_duration: float = 0.5,
            max_caption_duration: float = 2.0
    ) -> List[Dict]:
        if not words:
            return []

        captions = []
        current_words, start_time = [], None

        for i, word in enumerate(words):
            if start_time is None:
                start_time = word["start"]
            current_words.append(word)

            should_end = (
                    len(current_words) >= max_words_per_caption or
                    (word["end"] - start_time) >= max_caption_duration or
                    i == len(words) - 1
            )

            if should_end:
                end_time = current_words[-1]["end"]
                duration = end_time - start_time

                if duration >= min_caption_duration or i == len(words) - 1:
                    captions.append({
                        "text": " ".join(w["word"] for w in current_words),
                        "start": start_time,
                        "end": end_time,
                        "duration": duration
                    })
                    current_words, start_time = [], None

        return captions

    def add_captions_to_video(
            self,
            video_path: Path,
            output_path: Path,
            captions: List[Dict],
            font_size: int = 10,
            font_color: str = 'white',
            stroke_color: str = 'black',
            stroke_width: int = 2,
            position: Tuple[str, str] = ('center', 'below-center'),
            margin_bottom: int = 100
    ) -> str:
        """Create karaoke-style captions with green boxes behind spoken words"""
        try:
            filters = []

            # Process each caption
            for cap in captions:
                words = cap["text"].split()
                if not words:
                    continue

                full_text = cap["text"]
                escaped_full_text = full_text.replace("'", r"\'").replace(":", r"\:")

                # Calculate word timing
                word_duration = cap["duration"] / len(words)

                # Calculate shared Y position for both text and boxes
                text_height_approx = font_size + 10
                if position[1] == 'center':
                    shared_y = f"(h-{text_height_approx})/2"
                elif position[1] == 'below-center':
                    shared_y = f"(h-{text_height_approx})/2 + 220"
                elif position[1] == 'bottom':
                    shared_y = f"h-{text_height_approx}-{margin_bottom}"
                else:
                    shared_y = str(margin_bottom)

                # 1. Add main caption text (always visible during caption time)
                text_filter = (
                    f"drawtext=text='{escaped_full_text}'"
                    f":fontfile='C\\:/Windows/Fonts/impact.ttf'"
                    f":fontsize={font_size}"
                    f":fontcolor={font_color}"
                    f":borderw={stroke_width}"
                    f":bordercolor={stroke_color}"
                    f":x=(w-text_w)/2"
                    f":y={shared_y}"  # Use raw FFmpeg expression
                    f":enable='between(t,{cap['start']},{cap['end']})'"
                )
                filters.append(text_filter)

                # # 2. Add green drawbox behind each individual word at exact position
                # for i, word in enumerate(words):
                #     word_start = cap["start"] + (i * word_duration)
                #     word_end = cap["start"] + ((i + 1) * word_duration)
                #
                #     # Calculate approximate character width and word positioning
                #     char_width = font_size * 0.6  # Approximate character width
                #
                #     # Calculate position of this word within the full text
                #     text_before_word = " ".join(words[:i])
                #     if text_before_word:
                #         text_before_word += " "
                #
                #     # Calculate positioning
                #     full_text_width = len(full_text) * char_width
                #     before_word_width = len(text_before_word) * char_width
                #     word_width = len(word) * char_width + 12  # Word width + padding
                #     word_height = font_size + 16  # Word height + extra padding for highlight
                #
                #     # Position green box where this word appears
                #     box_x = f"(w-{full_text_width})/2+{before_word_width}"
                #
                #     # Align box_y exactly with the text (karaoke highlight)
                #     box_y = shared_y  # Use raw FFmpeg expression, not f-string
                #
                #     # Create green drawbox positioned exactly behind this word
                #     word_box_filter = (
                #         f"drawbox=x={box_x}"
                #         f":y={box_y}"  # Use raw FFmpeg expression
                #         f":w={word_width}"
                #         f":h={word_height}"
                #         f":color=00FF00"  # Solid green
                #         f":t=fill"
                #         f":enable='between(t,{word_start},{word_end})'"
                #     )
                #     filters.append(word_box_filter)

            # Combine all filters
            video_filter = ",".join(filters)

            # Run FFmpeg command
            cmd = [
                'ffmpeg',
                '-i', str(video_path),
                '-vf', video_filter,
                '-c:a', 'copy',
                '-y',
                str(output_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            logger("FFmpeg completed successfully")
            return str(output_path)

        except subprocess.CalledProcessError as e:
            logger(f"FFmpeg error: {e.stderr}")
            raise Exception(f"FFmpeg failed: {e.stderr}")

    def _color_to_hex(self, color_name: str) -> str:
        """Convert color name to hex for ffmpeg"""
        color_map = {
            'white': 'FFFFFF',
            'black': '000000',
            'red': 'FF0000',
            'blue': '0000FF',
            'green': '00FF00',
            'yellow': 'FFFF00'
        }
        return color_map.get(color_name.lower(), 'FFFFFF')

    def create_captions_for_short(
            self,
            video_path: Path,
            output_path: Path,
            word_transcription: Dict,
            start_time: float,
            end_time: float,
            caption_style: Dict = None
    ) -> str:
        """Main method to create captions for a video short"""
        style = {
            'font_size': 45,
            'font_color': 'white',
            'stroke_color': 'black',
            'stroke_width': 2,
            'position': ('center', 'center'),
            'margin_bottom': 100,
            'max_words_per_caption': 3,
            'min_caption_duration': 0.5,
            'max_caption_duration': 2.0
        }
        if caption_style:
            style.update(caption_style)

        # Extract and group words
        words = self.extract_words_for_timerange(word_transcription, start_time, end_time)
        captions = self.group_words_into_captions(
            words,
            max_words_per_caption=style['max_words_per_caption'],
            min_caption_duration=style['min_caption_duration'],
            max_caption_duration=style['max_caption_duration']
        )

        # Add captions using ffmpeg
        return self.add_captions_to_video(
            video_path,
            output_path,
            captions,
            font_size=style['font_size'],
            font_color=style['font_color'],
            stroke_color=style['stroke_color'],
            stroke_width=style['stroke_width'],
            margin_bottom=style['margin_bottom']
        )
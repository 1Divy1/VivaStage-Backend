import os
import uuid
import json
from datetime import datetime
from pathlib import Path


def prepare_output_dirs():
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


def save_data(data: dict | list | str, base_path: Path, file_name: str):
    """
    Save data to a file in the specified base path.
    If data is a dict or list, it will be saved as a JSON file.
    If data is a string, it will be saved as a text file.

    Args:
        data (dict | list | str): The data to save.
        base_path (Path): The base path where the file will be saved.
        file_name (str): The name of the file without extension.
    """
    if isinstance(data, (dict, list)):
        json_file_path = Path(base_path) / f"{file_name}.json"
        with open(json_file_path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False)
    elif isinstance(data, str):
        text_file_path = Path(base_path) / f"{file_name}.txt"
        with open(text_file_path, "w", encoding="utf-8") as file:
            file.write(data)
    else:
        raise ValueError(f"Unsupported data type: {type(data)}. Expected dict, list, or str.")


def logger(*args, **kwargs):
    """
    Enhanced logger function that handles multiple arguments and formatted strings.
    
    Args:
        *args: Variable number of arguments to log
        **kwargs: Keyword arguments passed to print()
    """
    try:
        if len(args) == 1:
            # Single argument case
            print(f"[INFO] {args[0]}", **kwargs)
        elif len(args) > 1:
            # Multiple arguments case - join them with spaces
            message = " ".join(str(arg) for arg in args)
            print(f"[INFO] {message}", **kwargs)
        else:
            # No arguments case
            print("[INFO]", **kwargs)
    except UnicodeEncodeError:
        # Fallback for unicode issues - replace problematic characters
        if len(args) == 1:
            safe_message = str(args[0]).encode('ascii', errors='replace').decode('ascii')
            print(f"[INFO] {safe_message}", **kwargs)
        elif len(args) > 1:
            safe_message = " ".join(str(arg).encode('ascii', errors='replace').decode('ascii') for arg in args)
            print(f"[INFO] {safe_message}", **kwargs)
        else:
            print("[INFO]", **kwargs)


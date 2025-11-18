# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the Application
```bash
uvicorn main:app --reload
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Architecture Overview

This is a FastAPI-based video processing service called "Viva Stage AI" that extracts highlights from YouTube videos and converts them into vertical (9:16) short-format videos with face-centered cropping.

### Core Components

**Service Layer (`app/services/`)**
- `ReelService`: Main orchestrator that coordinates the entire video processing pipeline

**Engine Layer (`app/engines/`)**
- `VideoEngine`: Handles YouTube video download, face detection/tracking, 9:16 cropping, and video manipulation using OpenCV and face_recognition
- `AudioEngine`: Processes audio transcription in chunks
- `LLMEngine`: Uses Groq API to analyze transcriptions and identify highlight moments
- `CaptionEngine`: Handles caption generation (newly added)

**API Layer (`app/api/routers/`)**
- `reel_router`: Single endpoint `/reels/extract` that accepts YouTube URLs and processing parameters

**Schemas (`app/schemas/`)**
- `ReelIn`: Pydantic model for request validation (YouTube URL, number of reels, duration constraints)

### Key Technologies
- **FastAPI**: Web framework
- **OpenCV + face_recognition**: Face detection and video processing
- **Groq**: LLM API for content analysis
- **pytubefix**: YouTube video downloading
- **moviepy**: Video editing and clip extraction
- **FFmpeg**: Audio/video merging (requires system installation)

### Processing Pipeline
1. Download video/audio from YouTube
2. Transcribe audio in chunks
3. Use LLM to identify highlight moments from transcription
4. Cut video clips based on timestamps
5. Apply face-centered 9:16 cropping with fallback letterboxing
6. Generate final short-format videos

### Face Detection Logic
The VideoEngine processes frames at 5fps intervals to:
- Detect and track unique faces using face_recognition
- Build segments where the same speaker is centered
- Fall back to letterboxed full-frame when no faces or multiple faces detected
- Apply temporal smoothing to avoid jarring transitions

### Output Structure
Processed videos are saved in timestamped directories under `output/` with subdirectories for:
- `audio/`: Extracted audio files
- `video/`: Original and final video files  
- `transcription/`: Chunked transcription JSON files
- `llm/`: LLM prompts and responses
- `clips/`: Individual highlight clips
- `shorts/`: Final 9:16 processed shorts

### Environment Variables
- `GROQ_API_KEY`: Required for Whisper inference

### Dependencies
Key external dependencies include mediapipe, opencv-contrib-python, face_recognition, groq, fastapi, and moviepy. FFmpeg must be installed separately on the system.
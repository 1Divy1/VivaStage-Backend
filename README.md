# Viva Stage AI - Backend

> AI-powered video processing service that transforms YouTube videos into engaging short-form content

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.116-009688.svg)](https://fastapi.tiangolo.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.11-5C3EE8.svg)](https://opencv.org/)
[![AI/ML](https://img.shields.io/badge/AI%2FML-Powered-orange.svg)](/)

## Overview

Viva Stage AI is a FastAPI-based backend service that automatically extracts highlight moments from long-form YouTube videos and converts them into vertical (9:16) short-format videos optimized for TikTok, Instagram Reels, and YouTube Shorts.

The service uses AI to analyze video transcripts, identifies the most engaging segments, applies intelligent face-centered cropping, and generates professional-looking shorts with optional captions—all through a single API call.

## Key Features

- **AI-Powered Highlight Extraction** - Uses LLM analysis to identify engaging moments from video transcripts
- **Intelligent Face Detection & Tracking** - Tracks speakers across frames and centers the crop on the active face
- **Smart 9:16 Cropping** - Face-centered cropping with automatic fallback to letterboxing for complex scenes
- **Automatic Caption Generation** - Word-level timed captions positioned for mobile viewing
- **LLM Provider Flexibility** - Switch between local (Ollama) and API-based (OpenAI) LLMs with zero code changes
- **YouTube Integration** - Direct video download and processing from YouTube URLs

## Tech Stack

### Web Framework
- **FastAPI** - Modern async Python web framework
- **Uvicorn** - ASGI server for production deployment
- **Pydantic** - Data validation and settings management

### AI & Machine Learning
- **LLMs** - OpenAI's GPT models + Ollama models 
- **Groq** - Whisper Large V3 for audio transcription

### Video & Audio Processing
- **OpenCV** - Computer vision and video manipulation
- **face_recognition** - Face detection and tracking
- **moviepy** - Video editing and clip extraction
- **pytubefix** - YouTube video downloading
- **FFmpeg** - Fast Audio/video processing

### Security & HTTP
- **Supabase** - JWT authentication
- **python-jose** - JWT token handling
- **httpx** - Async HTTP client

## Architecture

The project follows a clean layered architecture:

```
API Layer (Controllers)
        ↓
Service Layer (ReelService)
        ↓
Engine Layer (VideoEngine, AudioEngine, LLMEngine, CaptionEngine)
        ↓
Provider Layer (LLMProvider abstraction)
```

**Key Design Patterns:**
- **Service-Oriented Architecture** - Clear separation between orchestration and business logic
- **Provider Pattern** - Abstract LLM provider interface with factory instantiation
- **Dependency Injection** - FastAPI's DI system for clean component composition

## Quick Start

### Prerequisites

- Python 3.11 or higher
- Ollama for local LLM inference

### Installation

```bash
# 1. Clone the repository
git clone <repository-url>
cd backend

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Configure environment variables
cp .env.example .env
# Edit .env with your API keys (see Configuration section)

# 4. For local shorts usage
# Install Ollama from https://ollama.com
ollama pull llama3.1:8b
```

### Running the Server

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

### Configuration

Required environment variables:

```bash
# API Keys
GROQ_API_KEY=your_groq_api_key          # Required for transcription
OPENAI_API_KEY=your_openai_key          # Required if LLM_PROVIDER=openai

# shorts Provider Configuration
LLM_PROVIDER=local                      # Options: "local" or "openai"
LOCAL_LLM_URL=http://localhost:11434    # Ollama endpoint
LOCAL_LLM_MODEL=qwen3:8b                # Model name for Ollama
LOCAL_LLM_TIMEOUT=300                   # Request timeout in seconds

# Authentication
SUPABASE_URL=your_supabase_url
SUPABASE_SECRET_KEY=your_secret_key
SUPABASE_JWKS_URL=your_jwks_endpoint
SUPABASE_AUDIENCE=authenticated
```

## API Endpoints

### `POST /reels/extract`

Processes a YouTube video and generates short-form vertical videos.

**Request Body:**
```json
{
  "youtube_url": "https://www.youtube.com/watch?v=VIDEO_ID",
  "number_of_reels": 3,
  "min_seconds": 15,
  "max_seconds": 60,
  "captions": true,
  "language": "en"
}
```

**Response:**
```json
{
  "status": "success",
  "processing_id": "20240115_143022",
  "message": "Reels generated successfully"
}
```

**Output:** Generated videos are saved in `output/{processing_id}/shorts/`

## How It Works

The processing pipeline consists of 8 automated steps:

1. **Setup** - Creates timestamped output directories for organized file management
2. **Download** - Downloads video and audio from YouTube, merges streams using FFmpeg
3. **Transcribe** - Processes audio in chunks using Groq's Whisper API for word-level timestamps
4. **Analyze** - LLM analyzes transcript to identify highlight moments based on engagement
5. **Cut** - Extracts video clips using the identified timestamps
6. **Process** - Applies face detection and intelligent 9:16 aspect ratio cropping
7. **Caption** - Adds dynamic word-level captions to videos (if requested)
8. **Cleanup** - Removes temporary files and returns processing results

## Project Highlights

### LLM Provider Abstraction Layer

The standout architectural feature is a flexible LLM integration system that allows seamless switching between providers:

- **Abstract base class** defines the contract for all providers
- **Factory pattern** handles provider instantiation based on configuration
- **Provider-specific prompts** optimized for each LLM (ChatML for local, OpenAI format for API)
- **Structured output support** ensures type-safe responses across all providers

**Why this matters:** Develop and test with free local models (Ollama), then deploy with production APIs (OpenAI) using a single environment variable change. This demonstrates architectural foresight and cost-conscious engineering.

### Face Tracking Algorithm

Beyond simple cropping, the face detection system implements sophisticated tracking:

- Encodes and compares faces across frames to identify unique speakers
- Builds continuous segments where the same person is centered
- Applies temporal smoothing to prevent jarring transitions
- Gracefully handles edge cases (no faces, multiple faces) with letterbox fallback

This creates a professional viewing experience similar to manually edited content.

### Clean Architecture

The codebase demonstrates production-ready software design:

- **Separation of concerns** - Controllers handle HTTP, services orchestrate, engines execute
- **Dependency injection** - Clean component composition without tight coupling
- **Error handling** - Custom exception hierarchy with informative error messages
- **Security** - JWT authentication, input validation, CORS configuration

## Output Structure

Processed videos are organized in timestamped directories:

```
output/{processing_id}/
├── audio/           # Extracted and merged audio files
├── video/           # Downloaded and processed videos
├── transcription/   # JSON transcripts with word-level timing
├── llm/             # LLM formatted transcript
└── shorts/          # Final 9:16 vertical videos (DELIVERABLES)
```

## Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run with auto-reload
uvicorn main:app --reload
```

## License

Version 1.0 — 2025

Copyright © 2025 Mesaros David.
All rights reserved.

**1. Purpose**
This project is made public solely for the purpose of technical evaluation (e.g., job interviews, portfolio showcase).
No permission is granted for any other use.


**2. No Permission to Use, Copy, or Modify**
Except for viewing the source code directly on GitHub, you are NOT allowed to:
- use this code, in whole or in part, for any purpose
- copy it
- modify it
- distribute it
- reproduce it
- build upon it
- incorporate it into your own software
- use it in any commercial or non-commercial context
- use it for machine learning training, dataset creation, or code generation

All of these actions are strictly prohibited.


**3. No Redistribution**

You may not redistribute the code in any form, including forks, clones, mirrors, or archives.
Forking is disabled at the repository level, but any attempt to bypass this restriction is strictly prohibited.


**4. No Warranty**

This project is provided “as is”, without any warranties of any kind.


**5. Automatic License Termination**

If you violate any of the above terms:
- all permissions immediately terminate
- you must destroy all copies of the code
- you may be liable for damages under applicable law


**6. Ownership**

All intellectual property rights remain with Mesaros David, the sole creator of VivaStage.

---

**Built with FastAPI, OpenCV, and AI** | Showcasing modern Python backend development practices

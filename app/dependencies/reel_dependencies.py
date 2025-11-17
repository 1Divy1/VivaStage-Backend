from fastapi import Request
from functools import lru_cache

from app.engines.caption_engine import CaptionEngine
from app.engines.llm_engine import LLMEngine
from app.engines.video_engine import VideoEngine
from app.engines.audio_engine import AudioEngine
from app.services.reel_service import ReelService


def get_llm_engine(request: Request) -> LLMEngine:
    """Singleton pattern - single LLM engine instance throughout app's lifecycle"""
    if not hasattr(request.app.state, 'llm_engine'):
        request.app.state.llm_engine = LLMEngine(
            llm_provider=request.app.state.llm_provider
        )
    return request.app.state.llm_engine


def get_video_engine() -> VideoEngine:
    """Factory pattern - new instance per request (stateful)"""
    return VideoEngine()


def get_audio_engine(request: Request) -> AudioEngine:
    """Get or create AudioEngine instance."""
    if not hasattr(request.app.state, 'audio_engine'):
        request.app.state.audio_engine = AudioEngine(
            groq_client=request.app.state.groq_client
        )
    return request.app.state.audio_engine


@lru_cache()
def get_caption_engine() -> CaptionEngine:
    """Pure singleton - it doesn't require external dependencies like AudioEngine (singleton)"""
    return CaptionEngine()


def get_reel_service(request: Request) -> ReelService:
    """Create ReelService with optimized dependency injection."""
    return ReelService(
        llm_engine=get_llm_engine(request),
        video_engine=get_video_engine(),
        audio_engine=get_audio_engine(request),
        caption_engine=get_caption_engine()
    )
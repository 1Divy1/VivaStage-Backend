from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class TranscriptionWord(BaseModel):
    """Model for a single transcribed word with timing information."""

    word: str = Field(..., description="The transcribed word")
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")


class TranscriptionSegment(BaseModel):
    """Model for a transcription segment with timing and text."""

    id: int = Field(..., description="Segment ID")
    seek: int = Field(..., description="Seek position")
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    text: str = Field(..., description="Transcribed text for this segment")
    temperature: float = Field(..., description="Temperature used for this segment")
    avg_logprob: float = Field(..., description="Average log probability")
    compression_ratio: float = Field(..., description="Compression ratio")
    no_speech_prob: float = Field(..., description="No speech probability")


class TranscriptionResponse(BaseModel):
    """Model for transcription response containing text, words, and segments."""

    text: str = Field(..., description="Full transcribed text")
    words: List[TranscriptionWord] = Field(default_factory=list, description="List of transcribed words with timing")
    segments: List[TranscriptionSegment] = Field(default_factory=list, description="List of transcription segments")
    language: Optional[str] = Field(None, description="Detected or specified language")
    duration: Optional[float] = Field(None, description="Audio duration in seconds")


class TranscriptionChunkResponse(BaseModel):
    """Model for a single chunk transcription response."""

    text: str = Field(..., description="Transcribed text for this chunk")
    words: List[TranscriptionWord] = Field(default_factory=list, description="Words with timing information")
    segments: List[TranscriptionSegment] = Field(default_factory=list, description="Segments with detailed information")
    language: Optional[str] = Field(None, description="Detected or specified language")
    chunk_start_offset: float = Field(0.0, description="Start offset of this chunk in seconds")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for compatibility with existing code."""
        return {
            "text": self.text,
            "words": [word.model_dump() for word in self.words],
            "segments": [segment.model_dump() for segment in self.segments],
            "language": self.language
        }
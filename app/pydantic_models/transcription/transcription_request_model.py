from typing import Optional
from pydantic import BaseModel, Field
from pathlib import Path


class TranscriptionChunkRequest(BaseModel):
    """Request model for transcribing a single audio chunk."""

    audio_data: bytes = Field(..., description="Raw audio data as bytes")
    file_format: str = Field(..., description="Audio format (e.g., 'flac', 'wav', 'mp3')")
    language: str = Field(..., description="Target language for transcription")
    temperature: Optional[float] = Field(0.0, description="Sampling temperature for transcription")
    response_format: Optional[str] = Field("verbose_json", description="Response format")
    timestamp_granularities: Optional[list[str]] = Field(["word"], description="Timestamp granularities")


class TranscriptionFileRequest(BaseModel):
    """Request model for transcribing an entire audio file."""

    audio_path: Path = Field(..., description="Path to the audio file")
    language: str = Field(..., description="Target language for transcription")
    chunk_length: int = Field(60, description="Length of each chunk in seconds")
    overlap: int = Field(0, description="Overlap between chunks in seconds")
    temperature: Optional[float] = Field(0.0, description="Sampling temperature for transcription")
    response_format: Optional[str] = Field("verbose_json", description="Response format")
    timestamp_granularities: Optional[list[str]] = Field(["word"], description="Timestamp granularities")

    class Config:
        arbitrary_types_allowed = True
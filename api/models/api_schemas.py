from typing import List
from pydantic import BaseModel, Field


class EmbeddingRequest(BaseModel):
    path: str = Field(..., description="Path to audio file in MinIO (AUDIO_BUCKET)")


class EmbeddingResponse(BaseModel):
    embedding: List[float]
    shape: List[int]


class HealthResponse(BaseModel):
    status: str

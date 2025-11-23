from pydantic import BaseModel, Field


class EmbeddingRequest(BaseModel):
    path: str = Field(..., description="Path to audio file in MinIO (AUDIO_BUCKET)")


class EmbeddingResponse(BaseModel):
    embedding: list[float]
    shape: list[int]


class HealthResponse(BaseModel):
    status: str

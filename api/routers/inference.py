from fastapi import APIRouter

from api.core.dependencies import InferenceModelDep, MinioClientDep
from api.models.api_schemas import EmbeddingRequest, EmbeddingResponse
from api.inference.handler import embed_endpoint_handler

router = APIRouter(tags=["inference"])


@router.post("/inference/embeddings", response_model=EmbeddingResponse)
async def embed_audio(
    body: EmbeddingRequest,
    inference_model: InferenceModelDep,
    minio_client: MinioClientDep,
):
    return await embed_endpoint_handler(
        body=body, inference_model=inference_model, minio_client=minio_client
    )

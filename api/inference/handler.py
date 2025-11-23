import time
import asyncio
import tempfile
from pathlib import Path

from fastapi import HTTPException
from minio.error import S3Error

from api.core.config import settings
from api.core.logger import logger
from api.core.dependencies import InferenceModelDep, MinioClientDep
from api.models.api_schemas import EmbeddingRequest, EmbeddingResponse


async def embed_endpoint_handler(
    body: EmbeddingRequest,
    inference_model: InferenceModelDep,
    minio_client: MinioClientDep,
):
    # Validate file exists in MinIO first
    start = time.time()
    try:
        await asyncio.to_thread(
            minio_client.stat_object, settings.AUDIO_BUCKET, body.path
        )
        logger.info(f"Audio file found in MinIO: {body.path}")
    except S3Error as e:
        raise HTTPException(
            status_code=404, detail=f"Audio file not found: {body.path}"
        )

    # Create temp file for audio
    suffix = Path(body.path).suffix or ".wav"

    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp_path = tmp.name
    tmp.close()

    try:
        # Download audio from MinIO
        try:
            await asyncio.to_thread(
                minio_client.fget_object, settings.AUDIO_BUCKET, body.path, tmp_path
            )
            logger.info(f"Downloaded audio from MinIO: {body.path}")
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to download audio file: {e}"
            )

        # Run inference
        try:
            logger.info(f"Generating embeddings for {body.path}")
            embedding = await asyncio.to_thread(inference_model.run, tmp_path)
        except Exception as e:
            logger.error(f"Inference failed for {body.path}: {e}")
            raise HTTPException(status_code=500, detail=f"Inference failed: {e}")
            
        end = time.time()
        logger.info(f"Inference took {end - start} seconds")

        return EmbeddingResponse(
            embedding=embedding.tolist(), shape=list(embedding.shape)
        )
    finally:
        # Explicit cleanup
        Path(tmp_path).unlink(missing_ok=True)

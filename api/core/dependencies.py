from typing_extensions import Annotated

from fastapi import Depends, HTTPException, Request
from minio import Minio

from api.inference.openl3 import OpenL3Inference


def get_minio_client(request: Request) -> Minio:
    client = getattr(request.app.state, "minio_client", None)
    if client is None:
        raise HTTPException(status_code=503, detail="MinIO client not initialized")
    return client


def get_inference_model(request: Request) -> OpenL3Inference:
    model = getattr(request.app.state, "inference_model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model


MinioClientDep = Annotated[Minio, Depends(get_minio_client)]
InferenceModelDep = Annotated[OpenL3Inference, Depends(get_inference_model)]

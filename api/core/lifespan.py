import sys
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from minio import Minio

from api.core.config import settings
from api.core.logger import logger
from api.inference.openl3 import OpenL3Inference


def startup_minio_client(app: FastAPI):
    """Initialize MinIO client and verify connection."""
    minio_client = Minio(
        settings.MINIO_ENDPOINT,
        access_key=settings.MINIO_ACCESS_KEY,
        secret_key=settings.MINIO_SECRET_KEY,
        secure=settings.MINIO_SECURE,
    )

    # Verify connection by checking if bucket exists
    try:
        if minio_client.bucket_exists(settings.MINIO_BUCKET):
            logger.info(
                f"MinIO Client Connected to {settings.MINIO_ENDPOINT} (bucket: {settings.MINIO_BUCKET})"
            )
        else:
            logger.warning(f"MinIO bucket '{settings.MINIO_BUCKET}' does not exist!")
    except Exception as e:
        logger.warning(f"Could not verify MinIO connection: {e}")

    app.state.minio_client = minio_client


def load_embeddings_model(app: FastAPI):
    minio_client = app.state.minio_client

    if not os.path.exists(settings.MODEL_LOCAL_PATH):
        logger.info(
            f"Model not found locally. Downloading {settings.MODEL_FILE} from {settings.MODEL_BUCKET}..."
        )
        try:
            minio_client.fget_object(
                settings.MODEL_BUCKET, settings.MODEL_FILE, settings.MODEL_LOCAL_PATH
            )
            logger.info("Model downloaded successfully.")
        except Exception as e:
            logger.critical(f"Failed to download model: {e}")
            sys.exit(1)
    else:
        logger.info("Model found locally.")

    logger.info("Loading OpenL3 model with CPU...")
    try:
        app.state.inference_model = OpenL3Inference(
            settings.MODEL_LOCAL_PATH, use_gpu=settings.USE_GPU
        )
        logger.info("Model loaded successfully with CPU inference.")
    except Exception as e:
        logger.critical(f"Failed to load model: {e}")
        sys.exit(1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    startup_minio_client(app)
    load_embeddings_model(app)
    yield
    # MinIO client = on-demand HTTP requests → auto-cleanup

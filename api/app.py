from fastapi import FastAPI

from api.core.config import settings
from api.core.lifespan import lifespan
from api.routers.inference import router as inference_router
from api.routers.health import router as health_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="MSV2 INFERENCE API",
        lifespan=lifespan,
        docs_url="/docs" if settings.ENABLE_DOCS else None,
        redoc_url="/redoc" if settings.ENABLE_DOCS else None,
        swagger_ui_parameters={"defaultModelsExpandDepth": "0"},
    )

    app.include_router(inference_router)
    app.include_router(health_router)

    return app


app = create_app()

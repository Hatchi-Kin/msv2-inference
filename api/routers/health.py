from fastapi import APIRouter

from api.models.api_schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health_check():
    """Knative readiness probe endpoint."""
    return HealthResponse(status="ok")

"""Health check endpoint."""
from fastapi import APIRouter, Depends
from ...schemas.responses import HealthResponse
from ..dependencies import get_simulation_service
from ...services.simulation_service import SimulationService

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check(service: SimulationService = Depends(get_simulation_service)):
    """Health check endpoint.

    Returns:
        Service health status, model loaded status, and data loaded status
    """
    return HealthResponse(
        status="healthy",
        model_loaded=service._initialized and service.predictor is not None,
        data_loaded=service._initialized and service.test_x is not None,
    )

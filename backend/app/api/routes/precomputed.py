"""Pre-computed results endpoints."""
from fastapi import APIRouter, HTTPException, Depends
from ...schemas.responses import BenchmarkResponse, ScalabilityResponse
from ..dependencies import get_simulation_service
from ...services.simulation_service import SimulationService
from ...schemas.requests import BenchmarkRequest
import os
import json

router = APIRouter()


@router.get("/results/benchmark", response_model=BenchmarkResponse, tags=["results"])
async def get_precomputed_benchmark(service: SimulationService = Depends(get_simulation_service)):
    """Get pre-computed default benchmark results.

    This endpoint returns cached simulation results for the default configuration,
    providing instant response times for demo purposes.

    Returns:
        Pre-computed benchmark simulation results
    """
    # For now, compute with default parameters
    # In production, this would load from precomputed JSON file
    try:
        default_request = BenchmarkRequest()
        result = service.run_benchmark(default_request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load benchmark results: {str(e)}")


@router.get("/results/scalability", response_model=ScalabilityResponse, tags=["results"])
async def get_precomputed_scalability():
    """Get pre-computed scalability test results.

    This endpoint returns cached results for different regional scenarios
    (Donghae, Jeju, Seattle) showing how the system scales with solar generation.

    Returns:
        Pre-computed scalability test results
    """
    # Placeholder for scalability results
    # In production, this would load from precomputed JSON file
    raise HTTPException(status_code=501, detail="Scalability results not yet implemented")

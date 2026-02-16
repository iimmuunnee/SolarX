"""Simulation endpoints."""
from fastapi import APIRouter, Depends, HTTPException
from ...schemas.requests import BenchmarkRequest, CustomRequest
from ...schemas.responses import BenchmarkResponse, CustomResponse
from ..dependencies import get_simulation_service
from ...services.simulation_service import SimulationService

router = APIRouter()


@router.post("/simulate/benchmark", response_model=BenchmarkResponse, tags=["simulation"])
async def run_benchmark_simulation(
    request: BenchmarkRequest,
    service: SimulationService = Depends(get_simulation_service),
):
    """Run benchmark simulation comparing all battery vendors.

    This endpoint simulates battery performance for LG, Samsung, and Tesla batteries
    with the specified parameters.

    Args:
        request: Simulation parameters (capacity, thresholds, region factor)

    Returns:
        Simulation results including metrics, vendor comparison, and time series data
    """
    try:
        result = service.run_benchmark(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")


@router.post("/simulate/custom", response_model=CustomResponse, tags=["simulation"])
async def run_custom_simulation(
    request: CustomRequest,
    service: SimulationService = Depends(get_simulation_service),
):
    """Run custom simulation with specific battery vendor.

    This endpoint simulates battery performance for a single selected vendor
    with custom parameters.

    Args:
        request: Simulation parameters including vendor selection

    Returns:
        Simulation results for the selected vendor
    """
    try:
        result = service.run_custom(request)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")

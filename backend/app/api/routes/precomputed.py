"""Pre-computed results endpoints."""
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends
from ...schemas.responses import BenchmarkResponse, ScalabilityResponse
from ..dependencies import get_simulation_service
from ...services.simulation_service import SimulationService
from ...schemas.requests import BenchmarkRequest
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# Module-level cache: compute once, serve instantly thereafter
_benchmark_cache: Optional[BenchmarkResponse] = None


def warm_cache(service: SimulationService) -> None:
    """Pre-compute benchmark results and store in cache.
    Called during server startup to ensure instant responses."""
    global _benchmark_cache
    if _benchmark_cache is None:
        logger.info("Warming benchmark cache...")
        try:
            result = service.run_benchmark(BenchmarkRequest())
            _benchmark_cache = result
            logger.info("✅ Benchmark cache warmed successfully")
        except Exception as e:
            logger.error(f"❌ Failed to warm benchmark cache: {e}")


@router.get("/results/benchmark", response_model=BenchmarkResponse, tags=["results"])
async def get_precomputed_benchmark(service: SimulationService = Depends(get_simulation_service)):
    """Get pre-computed default benchmark results.

    This endpoint returns cached simulation results for the default configuration,
    providing instant response times for demo purposes.

    Returns:
        Pre-computed benchmark simulation results
    """
    global _benchmark_cache
    if _benchmark_cache is not None:
        return _benchmark_cache

    try:
        result = service.run_benchmark(BenchmarkRequest())
        _benchmark_cache = result
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

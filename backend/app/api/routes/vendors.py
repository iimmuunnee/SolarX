"""Vendor information endpoints."""
from fastapi import APIRouter
from ...schemas.responses import VendorsResponse
from ...services.simulation_service import SimulationService

router = APIRouter()


@router.get("/vendors", response_model=VendorsResponse, tags=["vendors"])
async def get_vendors():
    """Get information about all available battery vendors.

    Returns:
        List of vendor specifications including C-rate, efficiency, cost, degradation
    """
    vendors = SimulationService.get_vendor_info()
    return VendorsResponse(vendors=vendors)

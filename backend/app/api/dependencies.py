"""FastAPI dependencies."""
from ..services.simulation_service import SimulationService

# Global simulation service instance (initialized on startup)
_simulation_service: SimulationService = None


def get_simulation_service() -> SimulationService:
    """Get the global simulation service instance.

    Returns:
        SimulationService instance
    """
    global _simulation_service
    if _simulation_service is None:
        _simulation_service = SimulationService()
        _simulation_service.initialize()
    return _simulation_service


def initialize_simulation_service():
    """Initialize the simulation service on startup."""
    global _simulation_service
    _simulation_service = SimulationService()
    _simulation_service.initialize()

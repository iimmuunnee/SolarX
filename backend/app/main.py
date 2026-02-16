"""SolarX FastAPI Application."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from .config import settings
from .api.routes import health, vendors, simulation, precomputed
from .api.dependencies import initialize_simulation_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info("Starting SolarX API...")
    logger.info("Initializing simulation service...")
    try:
        initialize_simulation_service()
        logger.info("✅ Simulation service initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize simulation service: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down SolarX API...")


# Create FastAPI app
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description=settings.api_description,
    lifespan=lifespan,
)

# Configure CORS - Allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=False,  # Set to False when using allow_origins=["*"]
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Register routers
app.include_router(health.router, prefix="/api")
app.include_router(vendors.router, prefix="/api")
app.include_router(simulation.router, prefix="/api")
app.include_router(precomputed.router, prefix="/api")


# Serve frontend static files (must be after API routes)
from fastapi.staticfiles import StaticFiles
import os

frontend_dist = os.path.join(os.path.dirname(__file__), "../..", "frontend", "dist")
if os.path.exists(frontend_dist):
    # Mount static files with html=True for SPA routing
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="static")
    logger.info(f"✅ Serving frontend from {frontend_dist}")
else:
    # Fallback: API-only mode if frontend not built
    @app.get("/")
    async def root():
        """Root endpoint (API-only mode)."""
        return {
            "message": "SolarX API",
            "version": settings.api_version,
            "docs": "/docs",
            "health": "/api/health"
        }
    logger.warning("⚠️  Frontend dist/ not found, running in API-only mode")

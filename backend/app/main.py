"""SolarX FastAPI Application."""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
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

        # Warm the benchmark cache so /api/results/benchmark responds instantly
        from .api.routes.precomputed import warm_cache
        from .api.dependencies import get_simulation_service
        warm_cache(get_simulation_service())
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

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
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
    # Mount static assets (JS, CSS, images)
    app.mount("/assets", StaticFiles(directory=os.path.join(frontend_dist, "assets")), name="assets")

    # SPA catch-all: serve index.html for all non-API, non-asset routes
    @app.get("/{full_path:path}")
    async def serve_spa(request: Request, full_path: str):
        """Serve SPA index.html for client-side routing."""
        # Check if the requested file exists in dist (e.g., favicon.ico, robots.txt)
        file_path = os.path.join(frontend_dist, full_path)
        if full_path and os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(frontend_dist, "index.html"))

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

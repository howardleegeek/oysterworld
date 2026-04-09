"""FastAPI application factory for PhysicalFish service."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from physicalfish.api.middleware import RateLimitMiddleware
from physicalfish.api.routes import router
from physicalfish.config import Settings
from physicalfish.logging_config import get_logger, setup_logging

logger = get_logger("api.app")


def _init_simulator():
    """Initialize the physics simulator."""
    try:
        # Try to import and use PyBullet simulator
        from physicalfish.simulator.pybullet_sim import PyBulletSimulator

        simulator = PyBulletSimulator()
        logger.info("pybullet_simulator_initialized")
        return simulator
    except ImportError:
        logger.warning("pybullet_not_available")
        return None
    except Exception as e:
        logger.error("simulator_init_failed", error=str(e))
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("api_startup")

    # Setup logging
    settings = Settings()
    setup_logging(settings.log_level)
    logger.info("logging_configured", level=settings.log_level)

    # Initialize simulator
    app.state.simulator = _init_simulator()
    if app.state.simulator:
        logger.info("simulator_ready")
    else:
        logger.warning("simulator_not_available")

    yield

    # Shutdown
    logger.info("api_shutdown")
    if app.state.simulator:
        try:
            app.state.simulator.close()
            logger.info("simulator_closed")
        except Exception as e:
            logger.error("simulator_close_error", error=str(e))


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    settings = Settings()

    app = FastAPI(
        title="OysterWorld AutoResearch API",
        description="Meta-loop for physics-verified synthetic data — turn a grain of real data into a world of synthetic datasets",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add rate limiting middleware
    app.add_middleware(
        RateLimitMiddleware,
        tokens_per_minute=settings.rate_limit_per_minute,
        exclude_paths=["/api/v1/health", "/docs", "/openapi.json"],
    )

    # Include API routes
    app.include_router(router)

    logger.info("app_created", version="0.1.0")
    return app

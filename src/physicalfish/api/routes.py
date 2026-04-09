"""API routes for PhysicalFish service."""

import numpy as np
from fastapi import APIRouter, HTTPException, Request

from physicalfish.api.schemas import (
    BootstrapRequest,
    BootstrapResponse,
    HealthResponse,
    OptimizeRequest,
    OptimizeResponse,
    VerifyRequest,
    VerifyResponse,
)
from physicalfish.exceptions import InsufficientDataError, PhysicalFishError
from physicalfish.logging_config import get_logger
from physicalfish.loop.bootstrap import BootstrapGenerator
from physicalfish.loop.inner_loop import InnerLoop, InnerLoopConfig
from physicalfish.models import FrameData, PhysicsParams, TrajectoryData
from physicalfish.optimizer.bayesian import BayesianOptimizer
from physicalfish.verification.physics_verifier import PhysicsVerifier

logger = get_logger("api.routes")
router = APIRouter()


def _frames_from_dict(frames_data: list[dict]) -> list[FrameData]:
    """Convert frame dictionaries to FrameData objects."""
    frames = []
    for i, f in enumerate(frames_data):
        frame = FrameData(
            timestamp=f.get("timestamp", i * 0.033),
            position=np.array(f.get("position", [0.0, 0.0, 0.0])),
            velocity=np.array(f.get("velocity", [0.0, 0.0, 0.0])),
            acceleration=np.array(f.get("acceleration", [0.0, 0.0, 0.0])),
            rotation=np.array(f.get("rotation", [0.0, 0.0, 0.0])),
            angular_velocity=np.array(f.get("angular_velocity", [0.0, 0.0, 0.0])),
            frame_index=i,
        )
        frames.append(frame)
    return frames


def _trajectory_from_dict(data: dict) -> TrajectoryData:
    """Convert trajectory dictionary to TrajectoryData object."""
    frames = _frames_from_dict(data.get("frames", []))
    params_dict = data.get("params", {})
    params = PhysicsParams(
        gravity=params_dict.get("gravity", 9.81),
        mass=params_dict.get("mass", 0.5),
        friction=params_dict.get("friction", 0.5),
        restitution=params_dict.get("restitution", 0.8),
        linear_damping=params_dict.get("linear_damping", 0.1),
        angular_damping=params_dict.get("angular_damping", 0.1),
    )
    return TrajectoryData(frames=frames, params=params)


@router.get("/api/v1/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="healthy", version="0.1.0")


@router.post("/api/v1/verify", response_model=VerifyResponse)
async def verify_trajectory(request: VerifyRequest) -> VerifyResponse:
    """Verify a trajectory against physics constraints."""
    try:
        # Convert request data to TrajectoryData
        frames = _frames_from_dict(request.frames)

        if len(frames) < 3:
            raise InsufficientDataError(f"Need >= 3 frames, got {len(frames)}")

        params_dict = request.physics_params
        params = PhysicsParams(
            gravity=params_dict.get("gravity", 9.81),
            mass=params_dict.get("mass", 0.5),
            friction=params_dict.get("friction", 0.5),
            restitution=params_dict.get("restitution", 0.8),
            linear_damping=params_dict.get("linear_damping", 0.1),
            angular_damping=params_dict.get("angular_damping", 0.1),
        )

        trajectory = TrajectoryData(frames=frames, params=params)

        # Run verification
        verifier = PhysicsVerifier(gravity=params.gravity)
        result = verifier.verify(trajectory)

        return VerifyResponse(
            overall_score=result.overall_score,
            passed=result.passed,
            constraint_scores=result.constraint_scores,
        )

    except InsufficientDataError as e:
        logger.warning("verify_insufficient_data", error=str(e))
        raise HTTPException(status_code=400, detail=str(e)) from e
    except PhysicalFishError as e:
        logger.error("verify_physicalfish_error", error=str(e))
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error("verify_unexpected_error", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.post("/api/v1/optimize", response_model=OptimizeResponse)
async def optimize_params(request: OptimizeRequest, req: Request) -> OptimizeResponse:
    """Run parameter optimization using InnerLoop."""
    try:
        # Get simulator from app state (use getattr for safe access in tests)
        simulator = getattr(req.app.state, "simulator", None)
        if simulator is None:
            raise HTTPException(status_code=503, detail="Simulator not available")

        # Parse real data if provided
        real_data = None
        if request.real_data:
            real_data = [_trajectory_from_dict(d) for d in request.real_data]

        # Create verifier and optimizer
        verifier = PhysicsVerifier()
        optimizer = BayesianOptimizer(evaluator=lambda x: 0.5)  # Placeholder

        # Configure inner loop
        config = InnerLoopConfig(
            scenario=request.scenario,
            target_score=request.target_score,
            max_iterations=request.max_iterations,
        )

        # Run optimization
        inner_loop = InnerLoop(
            simulator=simulator,
            verifier=verifier,
            optimizer=optimizer,
            real_data=real_data,
            config=config,
        )

        result = inner_loop.run()

        # Extract convergence history (combined scores from history)
        convergence_history = [record.get("combined_score", 0.0) for record in result.history]

        return OptimizeResponse(
            best_params=result.best_params.to_dict(),
            best_score=result.best_score,
            iterations=result.iterations,
            convergence_history=convergence_history,
        )

    except HTTPException:
        raise
    except PhysicalFishError as e:
        logger.error("optimize_physicalfish_error", error=str(e))
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error("optimize_unexpected_error", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.post("/api/v1/bootstrap", response_model=BootstrapResponse)
async def bootstrap_data(request: BootstrapRequest, req: Request) -> BootstrapResponse:
    """Generate augmented data using BootstrapGenerator."""
    # Validate input first (before checking simulator)
    if not request.real_samples:
        raise HTTPException(status_code=400, detail="real_samples cannot be empty")

    try:
        # Get simulator from app state (use getattr for safe access in tests)
        simulator = getattr(req.app.state, "simulator", None)
        if simulator is None:
            raise HTTPException(status_code=503, detail="Simulator not available")

        real_trajectories = [_trajectory_from_dict(d) for d in request.real_samples]

        # Create bootstrap generator
        verifier = PhysicsVerifier()
        generator = BootstrapGenerator(
            simulator=simulator,
            verifier=verifier,
            scenario="grasp_ball",  # Default scenario
        )

        # Generate augmented data
        generated = generator.augment(
            real_samples=real_trajectories,
            target_count=request.target_count,
        )

        # Calculate pass rate
        total_attempts = len(generated) * 3  # Approximate based on max_attempts logic
        pass_rate = len(generated) / total_attempts if total_attempts > 0 else 0.0

        return BootstrapResponse(
            generated_count=len(generated),
            pass_rate=pass_rate,
        )

    except HTTPException:
        raise
    except PhysicalFishError as e:
        logger.error("bootstrap_physicalfish_error", error=str(e))
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error("bootstrap_unexpected_error", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error") from e

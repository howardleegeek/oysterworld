"""Pydantic v2 request/response models for PhysicalFish API."""

from typing import Any

from pydantic import BaseModel, Field


class VerifyRequest(BaseModel):
    """Request model for trajectory verification."""

    frames: list[dict[str, Any]] = Field(
        ...,
        description="List of frame data dictionaries with timestamp, position, velocity, etc.",
    )
    physics_params: dict[str, float] = Field(
        default_factory=dict,
        description="Physics parameters (gravity, mass, friction, restitution, damping)",
    )


class VerifyResponse(BaseModel):
    """Response model for trajectory verification."""

    overall_score: float = Field(
        ..., ge=0.0, le=1.0, description="Overall physics verification score"
    )
    passed: bool = Field(
        ..., description="Whether the trajectory passed verification (score > 0.7)"
    )
    constraint_scores: dict[str, Any] = Field(
        ...,
        description="Per-constraint scores (kinematic, dynamic, energy, momentum, angular, collision)",
    )


class OptimizeRequest(BaseModel):
    """Request model for parameter optimization."""

    scenario: str = Field(..., description="Scenario name (e.g., 'grasp_ball', 'throw_catch')")
    target_score: float = Field(
        default=0.90, ge=0.0, le=1.0, description="Target verification score"
    )
    max_iterations: int = Field(
        default=50, ge=1, le=200, description="Maximum optimization iterations"
    )
    real_data: list[dict[str, Any]] | None = Field(
        default=None, description="Optional real trajectory data for comparison"
    )


class OptimizeResponse(BaseModel):
    """Response model for parameter optimization."""

    best_params: dict[str, float] = Field(..., description="Optimized physics parameters")
    best_score: float = Field(..., ge=0.0, le=1.0, description="Best achieved verification score")
    iterations: int = Field(..., ge=0, description="Number of iterations performed")
    convergence_history: list[float] = Field(..., description="Score history across iterations")


class BootstrapRequest(BaseModel):
    """Request model for data bootstrapping/augmentation."""

    real_samples: list[dict[str, Any]] = Field(
        ..., description="List of real trajectory samples to augment from"
    )
    target_count: int = Field(
        ..., ge=1, le=1000, description="Target number of synthetic samples to generate"
    )


class BootstrapResponse(BaseModel):
    """Response model for data bootstrapping."""

    generated_count: int = Field(..., ge=0, description="Number of synthetic samples generated")
    pass_rate: float = Field(
        ..., ge=0.0, le=1.0, description="Ratio of accepted samples to total attempts"
    )


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""

    status: str = Field(default="healthy", description="Service health status")
    version: str = Field(default="0.1.0", description="API version")

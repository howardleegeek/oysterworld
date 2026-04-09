"""Unified data models for PhysicalFish AutoResearch."""

import numpy as np
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FrameData:
    """Complete physical state at a timestep."""

    timestamp: float
    position: np.ndarray  # [x, y, z]
    velocity: np.ndarray  # [vx, vy, vz]
    acceleration: np.ndarray  # [ax, ay, az]
    rotation: np.ndarray  # [rx, ry, rz]
    angular_velocity: np.ndarray  # [wx, wy, wz]
    frame_index: int = 0

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.velocity))

    @property
    def height(self) -> float:
        return float(self.position[1])

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FrameData):
            return NotImplemented
        return (
            self.timestamp == other.timestamp
            and np.array_equal(self.position, other.position)
            and np.array_equal(self.velocity, other.velocity)
        )

    def __hash__(self) -> int:
        return hash((self.timestamp, self.frame_index))


@dataclass(frozen=True)
class PhysicsParams:
    """Physics simulation parameters."""

    gravity: float = 9.81
    mass: float = 0.5
    friction: float = 0.5
    restitution: float = 0.8
    linear_damping: float = 0.1
    angular_damping: float = 0.1

    def to_dict(self) -> dict[str, float]:
        return {
            "gravity": self.gravity,
            "mass": self.mass,
            "friction": self.friction,
            "restitution": self.restitution,
            "linear_damping": self.linear_damping,
            "angular_damping": self.angular_damping,
        }


@dataclass
class TrajectoryData:
    """A complete trajectory with frames and metadata."""

    frames: list[FrameData]
    params: PhysicsParams
    events: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        if not self.frames:
            return 0.0
        return self.frames[-1].timestamp - self.frames[0].timestamp

    @property
    def num_frames(self) -> int:
        return len(self.frames)


@dataclass(frozen=True)
class VerificationResult:
    """Result of physics verification."""

    overall_score: float
    passed: bool
    constraint_scores: dict[str, float]
    details: dict[str, Any]


@dataclass(frozen=True)
class OptimizationResult:
    """Result of parameter optimization."""

    best_params: dict[str, float]
    best_score: float
    n_iterations: int
    convergence_history: list[float]
    acquisition_history: list[str]

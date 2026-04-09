"""Shared test fixtures."""

import numpy as np
import pytest

from physicalfish.models import FrameData, PhysicsParams, TrajectoryData


@pytest.fixture
def default_params():
    return PhysicsParams()


@pytest.fixture
def make_frame():
    def _make(t: float = 0.0, y: float = 1.0, vy: float = 0.0, ay: float = -9.81, idx: int = 0):
        return FrameData(
            timestamp=t,
            position=np.array([0.0, y, 0.0]),
            velocity=np.array([0.0, vy, 0.0]),
            acceleration=np.array([0.0, ay, 0.0]),
            rotation=np.zeros(3),
            angular_velocity=np.zeros(3),
            frame_index=idx,
        )

    return _make


@pytest.fixture
def freefall_trajectory(make_frame, default_params):
    """A physically correct free-fall trajectory."""
    g = default_params.gravity
    dt = 1.0 / 30.0
    frames = []
    for i in range(90):
        t = i * dt
        y = max(0.15, 2.0 - 0.5 * g * t**2)
        vy = -g * t if y > 0.15 else 0.0
        ay = -g if y > 0.15 else 0.0
        frames.append(make_frame(t=t, y=y, vy=vy, ay=ay, idx=i))
    return TrajectoryData(frames=frames, params=default_params)

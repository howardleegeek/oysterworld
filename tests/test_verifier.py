"""Tests for physics verifier including property-based tests."""

import numpy as np
import pytest

from physicalfish.models import FrameData, TrajectoryData, PhysicsParams, VerificationResult
from physicalfish.exceptions import InsufficientDataError
from physicalfish.verification.physics_verifier import PhysicsVerifier


class TestPhysicsVerifier:
    def test_perfect_freefall_scores_high(self, freefall_trajectory):
        """Exact Newtonian trajectory should score > 0.85."""
        verifier = PhysicsVerifier()
        result = verifier.verify(freefall_trajectory)
        assert result.overall_score > 0.85
        assert bool(result.passed) is True

    def test_insufficient_frames_raises(self, make_frame):
        trajectory = TrajectoryData(frames=[make_frame()], params=PhysicsParams())
        verifier = PhysicsVerifier()
        with pytest.raises(InsufficientDataError):
            verifier.verify(trajectory)

    def test_noisy_trajectory_scores_lower(self, freefall_trajectory):
        """Adding noise should reduce the score."""
        verifier = PhysicsVerifier()
        clean_result = verifier.verify(freefall_trajectory)

        # Add noise to velocity
        noisy_frames = []
        for f in freefall_trajectory.frames:
            noisy_vel = f.velocity + np.random.normal(0, 2.0, 3)
            noisy_frames.append(
                FrameData(
                    timestamp=f.timestamp,
                    position=f.position,
                    velocity=noisy_vel,
                    acceleration=f.acceleration,
                    rotation=f.rotation,
                    angular_velocity=f.angular_velocity,
                    frame_index=f.frame_index,
                )
            )
        noisy_traj = TrajectoryData(frames=noisy_frames, params=freefall_trajectory.params)
        noisy_result = verifier.verify(noisy_traj)

        assert noisy_result.overall_score < clean_result.overall_score

    def test_all_six_constraints_present(self, freefall_trajectory):
        verifier = PhysicsVerifier()
        result = verifier.verify(freefall_trajectory)
        expected = {"kinematic", "dynamic", "energy", "momentum", "angular", "collision"}
        assert set(result.constraint_scores.keys()) == expected


@pytest.fixture
def make_frame():
    """Factory for creating FrameData objects."""

    def _make(
        timestamp=0.0,
        position=None,
        velocity=None,
        acceleration=None,
        rotation=None,
        angular_velocity=None,
        frame_index=0,
    ):
        return FrameData(
            timestamp=timestamp,
            position=position if position is not None else np.array([0.0, 1.0, 0.0]),
            velocity=velocity if velocity is not None else np.array([0.0, 0.0, 0.0]),
            acceleration=acceleration if acceleration is not None else np.array([0.0, -9.81, 0.0]),
            rotation=rotation if rotation is not None else np.array([0.0, 0.0, 0.0]),
            angular_velocity=angular_velocity
            if angular_velocity is not None
            else np.array([0.0, 0.0, 0.0]),
            frame_index=frame_index,
        )

    return _make


@pytest.fixture
def freefall_trajectory(make_frame):
    """Create a perfect free-fall trajectory for testing.

    Uses vertical free-fall only (no horizontal velocity) so that:
    1. Angular momentum L = r × p = 0 (conserved)
    2. Limited frames to avoid hitting ground (y > 0)
    """
    frames = []
    g = 9.81
    dt = 0.033  # ~30 fps
    initial_height = 5.0  # Higher start to avoid ground contact
    initial_velocity = np.array([0.0, 0.0, 0.0])  # No horizontal velocity for L conservation

    # Limit to 30 frames (~1 second) to stay well above ground
    # y = 5 - 0.5*9.81*1^2 = 5 - 4.9 = 0.1 (still positive)
    for i in range(30):
        t = i * dt
        # Free fall physics: y = y0 + v0*t + 0.5*a*t^2
        position = np.array(
            [
                0.0,  # x stays at origin (r × p = 0)
                initial_height - 0.5 * g * t**2,  # y
                0.0,  # z
            ]
        )
        # v = v0 + a*t
        velocity = np.array(
            [
                0.0,  # vx = 0
                -g * t,  # vy
                0.0,  # vz
            ]
        )
        # a = constant
        acceleration = np.array([0.0, -g, 0.0])

        frames.append(
            FrameData(
                timestamp=t,
                position=position,
                velocity=velocity,
                acceleration=acceleration,
                rotation=np.array([0.0, 0.0, 0.0]),
                angular_velocity=np.array([0.0, 0.0, 0.0]),
                frame_index=i,
            )
        )

    params = PhysicsParams(mass=0.5, gravity=g)
    return TrajectoryData(frames=frames, params=params)

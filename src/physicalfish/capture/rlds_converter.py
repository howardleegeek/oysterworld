"""RLDS (Robot Learning Dataset) converter for real robot data.

Converts saved trajectory JSON files to TrajectoryData format.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Optional

from physicalfish.models import FrameData, PhysicsParams, TrajectoryData


class RLDSConverter:
    """Convert RLDS-format robot trajectories to TrajectoryData."""

    def __init__(self, dt: float = 0.1):
        """Initialize converter.

        Args:
            dt: Time step between frames (default 0.1s = 10Hz)
        """
        self.dt = dt

    def load_trajectory_file(self, filepath: Path) -> TrajectoryData:
        """Load a single trajectory JSON file and convert to TrajectoryData.

        Args:
            filepath: Path to the JSON file

        Returns:
            TrajectoryData object
        """
        with open(filepath, "r") as f:
            data = json.load(f)

        return self.convert_to_trajectory(data)

    def load_all_trajectories(self, directory: Path) -> List[TrajectoryData]:
        """Load all trajectory files from a directory.

        Args:
            directory: Directory containing trajectory JSON files

        Returns:
            List of TrajectoryData objects
        """
        trajectories = []

        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        json_files = sorted(directory.glob("*.json"))

        for filepath in json_files:
            try:
                trajectory = self.load_trajectory_file(filepath)
                trajectories.append(trajectory)
            except Exception as e:
                print(f"Warning: Failed to load {filepath}: {e}")

        return trajectories

    def convert_to_trajectory(self, data: dict) -> TrajectoryData:
        """Convert RLDS trajectory data to TrajectoryData.

        The state vector contains joint angles + end-effector position.
        We extract position from state and compute velocity/acceleration
        from position differences.

        Args:
            data: Dictionary containing trajectory data with 'trajectory' key

        Returns:
            TrajectoryData object with FrameData frames
        """
        steps = data.get("trajectory", [])

        if not steps:
            raise ValueError("No trajectory steps found in data")

        # Extract dt from data or use default
        dt = data.get("dt", self.dt)

        frames = []

        for i, step in enumerate(steps):
            timestamp = step.get("timestamp", i * dt)

            # Get position from state vector or observation
            state = step.get("state", [])
            observation = step.get("observation", {})

            # Extract position: [x, y, z] from state or observation
            if len(state) >= 3:
                position = np.array(state[:3], dtype=np.float64)
            else:
                # Try to get from observation
                obs_pos = observation.get("position", [0.0, 0.0, 0.0])
                position = np.array(obs_pos, dtype=np.float64)

            # Get velocity if available, otherwise compute from position differences
            if "velocity" in step:
                velocity = np.array(step["velocity"], dtype=np.float64)
            elif len(state) >= 6:
                # Velocity is in state vector [vx, vy, vz]
                velocity = np.array(state[3:6], dtype=np.float64)
            else:
                # Compute velocity from position differences
                velocity = self._compute_velocity(frames, position, dt)

            # Get acceleration if available, otherwise compute from velocity differences
            if "acceleration" in step:
                acceleration = np.array(step["acceleration"], dtype=np.float64)
            else:
                # Compute acceleration from velocity differences
                acceleration = self._compute_acceleration(frames, velocity, dt)

            # Rotation and angular velocity (default to zero if not available)
            rotation = np.array(step.get("rotation", [0.0, 0.0, 0.0]), dtype=np.float64)
            angular_velocity = np.array(
                step.get("angular_velocity", [0.0, 0.0, 0.0]), dtype=np.float64
            )

            frame = FrameData(
                timestamp=float(timestamp),
                position=position,
                velocity=velocity,
                acceleration=acceleration,
                rotation=rotation,
                angular_velocity=angular_velocity,
                frame_index=i,
            )
            frames.append(frame)

        # Create physics params from metadata or use defaults
        gravity = data.get("gravity", 9.81)
        restitution = data.get("restitution", 0.8)

        params = PhysicsParams(
            gravity=gravity,
            mass=0.5,  # Default mass for a ball
            friction=0.5,
            restitution=restitution,
            linear_damping=0.1,
            angular_damping=0.1,
        )

        metadata = {
            "episode_id": data.get("episode_id", 0),
            "num_steps": data.get("num_steps", len(steps)),
            "dt": dt,
            "source": "synthetic" if "synthetic" in str(data) else "real",
        }

        return TrajectoryData(frames=frames, params=params, metadata=metadata)

    def _compute_velocity(
        self, frames: List[FrameData], current_position: np.ndarray, dt: float
    ) -> np.ndarray:
        """Compute velocity from position differences.

        Uses central difference for interior points:
        v_i = (x_{i+1} - x_{i-1}) / (2*dt)

        For first frame, uses forward difference:
        v_0 = (x_1 - x_0) / dt
        """
        if len(frames) == 0:
            # First frame: assume zero velocity or use small forward difference estimate
            return np.array([0.0, 0.0, 0.0])
        elif len(frames) == 1:
            # Second frame: forward difference
            prev_pos = frames[-1].position
            return (current_position - prev_pos) / dt
        else:
            # Use central difference with previous position
            # For real-time, we only have past data, so use backward difference
            prev_pos = frames[-1].position
            return (current_position - prev_pos) / dt

    def _compute_acceleration(
        self, frames: List[FrameData], current_velocity: np.ndarray, dt: float
    ) -> np.ndarray:
        """Compute acceleration from velocity differences.

        Uses backward difference:
        a_i = (v_i - v_{i-1}) / dt
        """
        if len(frames) == 0:
            # First frame: assume gravity
            return np.array([0.0, -9.81, 0.0])
        else:
            # Backward difference
            prev_vel = frames[-1].velocity
            return (current_velocity - prev_vel) / dt


def load_real_trajectories(data_dir: Optional[Path] = None) -> List[TrajectoryData]:
    """Convenience function to load all real trajectories.

    Args:
        data_dir: Directory containing trajectory files. Defaults to data/real_trajectories/

    Returns:
        List of TrajectoryData objects
    """
    if data_dir is None:
        data_dir = Path("data/real_trajectories")

    converter = RLDSConverter(dt=0.1)
    return converter.load_all_trajectories(data_dir)


if __name__ == "__main__":
    # Test the converter
    trajectories = load_real_trajectories()

    print(f"Loaded {len(trajectories)} trajectories")

    for i, traj in enumerate(trajectories):
        print(f"\nTrajectory {i}:")
        print(f"  Frames: {traj.num_frames}")
        print(f"  Duration: {traj.duration:.2f}s")
        print(f"  Params: gravity={traj.params.gravity}, restitution={traj.params.restitution}")

        if traj.num_frames > 0:
            first_frame = traj.frames[0]
            print(f"  First frame position: {first_frame.position}")
            print(f"  First frame velocity: {first_frame.velocity}")

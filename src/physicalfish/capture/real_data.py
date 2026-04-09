"""Real data loading and conversion utilities."""

import json
import numpy as np
from pathlib import Path
from typing import Any

from physicalfish.models import FrameData, PhysicsParams, TrajectoryData
from physicalfish.logging_config import get_logger


class RealDataLoader:
    """Load real trajectory data from capture_real.py format."""

    def __init__(self):
        self.logger = get_logger("real_data_loader")

    def load_trajectory(self, path: Path) -> TrajectoryData:
        """Load a single trajectory from JSON file.

        Args:
            path: Path to the JSON file (capture_real.py format)

        Returns:
            TrajectoryData with frames and physics params
        """
        self.logger.info("loading_trajectory", path=str(path))

        with open(path, "r") as f:
            data = json.load(f)

        # Extract physics params from the data
        physics_params = self._extract_physics_params(data)

        # Convert frames to FrameData objects
        frames = self._convert_frames(data)

        # Extract events if present
        events = data.get("events", [])

        trajectory = TrajectoryData(
            frames=frames,
            params=physics_params,
            events=events,
            metadata={
                "capture_id": data.get("capture_id", "unknown"),
                "scenario": data.get("scenario", "unknown"),
                "source_file": str(path),
            },
        )

        self.logger.info(
            "trajectory_loaded",
            num_frames=len(frames),
            duration=trajectory.duration,
            scenario=data.get("scenario", "unknown"),
        )

        return trajectory

    def load_directory(self, dir_path: Path) -> list[TrajectoryData]:
        """Batch load all trajectories from a directory.

        Args:
            dir_path: Directory containing JSON trajectory files

        Returns:
            List of TrajectoryData objects
        """
        self.logger.info("loading_directory", path=str(dir_path))

        trajectories = []
        json_files = list(dir_path.glob("*.json"))

        for json_file in json_files:
            try:
                traj = self.load_trajectory(json_file)
                trajectories.append(traj)
            except Exception as e:
                self.logger.warning(
                    "failed_to_load_file",
                    file=str(json_file),
                    error=str(e),
                )

        self.logger.info(
            "directory_loaded",
            total_files=len(json_files),
            successful=len(trajectories),
        )

        return trajectories

    def _extract_physics_params(self, data: dict[str, Any]) -> PhysicsParams:
        """Extract physics parameters from capture data."""
        # Try to get from physics_params field (trajectory format)
        if "physics_params" in data:
            pp = data["physics_params"]
            return PhysicsParams(
                gravity=pp.get("gravity_magnitude", 9.81),
                mass=pp.get("object_mass", 0.5),
                friction=pp.get("object_friction", 0.5),
                restitution=pp.get("object_bounce", 0.8),
            )

        # Default params if not specified
        return PhysicsParams()

    def _convert_frames(self, data: dict[str, Any]) -> list[FrameData]:
        """Convert frame data to FrameData objects.

        Handles coordinate conversion and computes acceleration from
        velocity if acceleration is not present.
        """
        frames = []
        frame_list = data.get("frames", [])

        if not frame_list:
            return frames

        # Check if this is capture format or trajectory format
        is_capture_format = "object" in frame_list[0] if frame_list else False

        for i, frame_data in enumerate(frame_list):
            if is_capture_format:
                # capture_real.py format
                obj = frame_data.get("object", {})
                position = np.array(obj.get("position", [0.0, 0.0, 0.0]))

                # Velocity may not be present in capture format
                velocity = np.array(obj.get("velocity", [0.0, 0.0, 0.0]))

                # Acceleration may not be present - compute from velocity if needed
                acceleration = self._get_or_compute_acceleration(
                    frame_list, i, obj.get("acceleration")
                )

                rotation = np.array(obj.get("rotation", [0.0, 0.0, 0.0]))
                angular_velocity = np.array(obj.get("angular_velocity", [0.0, 0.0, 0.0]))
            else:
                # Direct trajectory format
                position = np.array(frame_data.get("position", [0.0, 0.0, 0.0]))
                velocity = np.array(frame_data.get("velocity", [0.0, 0.0, 0.0]))
                acceleration = self._get_or_compute_acceleration(
                    frame_list, i, frame_data.get("acceleration")
                )
                rotation = np.array(frame_data.get("rotation", [0.0, 0.0, 0.0]))
                angular_velocity = np.array(frame_data.get("angular_velocity", [0.0, 0.0, 0.0]))

            frame = FrameData(
                timestamp=frame_data.get("timestamp", i / 30.0),
                position=position,
                velocity=velocity,
                acceleration=acceleration,
                rotation=rotation,
                angular_velocity=angular_velocity,
                frame_index=frame_data.get("frame", i),
            )
            frames.append(frame)

        return frames

    def _get_or_compute_acceleration(
        self, frame_list: list[dict], idx: int, accel_data: list[float] | None
    ) -> np.ndarray:
        """Get acceleration from data or compute from velocity differences."""
        # If acceleration is provided, use it
        if accel_data is not None:
            return np.array(accel_data)

        # Otherwise, compute from velocity differences (numerical derivative)
        if idx == 0 or idx >= len(frame_list) - 1:
            # Can't compute at boundaries
            return np.array([0.0, -9.81, 0.0])  # Default to gravity

        # Get velocities at adjacent frames
        prev_frame = frame_list[idx - 1]
        next_frame = frame_list[idx + 1]

        # Extract velocities (handle both formats)
        def get_velocity(frame):
            if "object" in frame:
                return frame["object"].get("velocity", [0.0, 0.0, 0.0])
            return frame.get("velocity", [0.0, 0.0, 0.0])

        v_prev = np.array(get_velocity(prev_frame))
        v_next = np.array(get_velocity(next_frame))

        # Compute time step
        t_prev = prev_frame.get("timestamp", (idx - 1) / 30.0)
        t_next = next_frame.get("timestamp", (idx + 1) / 30.0)
        dt = t_next - t_prev

        if dt > 0:
            # Central difference: a = (v_next - v_prev) / (2 * dt) * 2 = (v_next - v_prev) / dt
            acceleration = (v_next - v_prev) / dt
        else:
            acceleration = np.array([0.0, -9.81, 0.0])

        return acceleration

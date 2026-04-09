"""Tests for real data loading and TrajectoryData conversion."""

import json
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from physicalfish.capture.real_data import RealDataLoader
from physicalfish.models import FrameData, PhysicsParams, TrajectoryData


class TestRealDataLoaderInit:
    """Tests for RealDataLoader initialization."""

    def test_default_initialization(self):
        """RealDataLoader should initialize with logger."""
        loader = RealDataLoader()
        assert loader.logger is not None


class TestLoadTrajectory:
    """Tests for load_trajectory method."""

    @pytest.fixture
    def temp_json_file(self, tmp_path):
        """Create a temporary JSON file with capture format data."""
        data = {
            "capture_id": "test_capture_001",
            "scenario": "grasp_ball",
            "physics_params": {
                "gravity_magnitude": 9.8,
                "object_mass": 0.6,
                "object_friction": 0.4,
                "object_bounce": 0.7,
            },
            "frames": [
                {
                    "timestamp": 0.0,
                    "frame": 0,
                    "object": {
                        "position": [0.0, 1.0, 0.0],
                        "velocity": [0.0, 0.0, 0.0],
                        "acceleration": [0.0, -9.8, 0.0],
                        "rotation": [0.0, 0.0, 0.0],
                        "angular_velocity": [0.0, 0.0, 0.0],
                    },
                },
                {
                    "timestamp": 0.033,
                    "frame": 1,
                    "object": {
                        "position": [0.0, 0.995, 0.0],
                        "velocity": [0.0, -0.33, 0.0],
                        "acceleration": [0.0, -9.8, 0.0],
                        "rotation": [0.0, 0.0, 0.0],
                        "angular_velocity": [0.0, 0.0, 0.0],
                    },
                },
                {
                    "timestamp": 0.066,
                    "frame": 2,
                    "object": {
                        "position": [0.0, 0.978, 0.0],
                        "velocity": [0.0, -0.65, 0.0],
                        "acceleration": [0.0, -9.8, 0.0],
                        "rotation": [0.0, 0.0, 0.0],
                        "angular_velocity": [0.0, 0.0, 0.0],
                    },
                },
            ],
            "events": [
                {"type": "grasp", "frame": 10, "timestamp": 0.33},
                {"type": "release", "frame": 20, "timestamp": 0.66},
            ],
        }
        file_path = tmp_path / "test_capture.json"
        with open(file_path, "w") as f:
            json.dump(data, f)
        return file_path

    @pytest.fixture
    def temp_trajectory_json(self, tmp_path):
        """Create a temporary JSON file with direct trajectory format."""
        data = {
            "capture_id": "test_traj_001",
            "scenario": "throw_catch",
            "physics_params": {
                "gravity_magnitude": 9.81,
                "object_mass": 0.5,
                "object_friction": 0.5,
                "object_bounce": 0.8,
            },
            "frames": [
                {
                    "timestamp": 0.0,
                    "frame": 0,
                    "position": [1.0, 2.0, 0.0],
                    "velocity": [0.5, 1.0, 0.0],
                    "acceleration": [0.0, -9.81, 0.0],
                    "rotation": [0.0, 0.0, 0.0],
                    "angular_velocity": [0.0, 0.0, 0.0],
                },
                {
                    "timestamp": 0.033,
                    "frame": 1,
                    "position": [1.02, 2.03, 0.0],
                    "velocity": [0.5, 0.67, 0.0],
                    "acceleration": [0.0, -9.81, 0.0],
                    "rotation": [0.0, 0.0, 0.0],
                    "angular_velocity": [0.0, 0.0, 0.0],
                },
            ],
        }
        file_path = tmp_path / "test_trajectory.json"
        with open(file_path, "w") as f:
            json.dump(data, f)
        return file_path

    def test_load_capture_format(self, temp_json_file):
        """Should load capture_real.py format correctly."""
        loader = RealDataLoader()
        trajectory = loader.load_trajectory(temp_json_file)

        assert isinstance(trajectory, TrajectoryData)
        assert len(trajectory.frames) == 3
        assert trajectory.metadata["capture_id"] == "test_capture_001"
        assert trajectory.metadata["scenario"] == "grasp_ball"

    def test_load_trajectory_format(self, temp_trajectory_json):
        """Should load direct trajectory format correctly."""
        loader = RealDataLoader()
        trajectory = loader.load_trajectory(temp_trajectory_json)

        assert isinstance(trajectory, TrajectoryData)
        assert len(trajectory.frames) == 2
        assert trajectory.metadata["scenario"] == "throw_catch"

    def test_extracts_physics_params(self, temp_json_file):
        """Should extract physics parameters from JSON."""
        loader = RealDataLoader()
        trajectory = loader.load_trajectory(temp_json_file)

        assert isinstance(trajectory.params, PhysicsParams)
        assert trajectory.params.gravity == 9.8
        assert trajectory.params.mass == 0.6
        assert trajectory.params.friction == 0.4
        assert trajectory.params.restitution == 0.7

    def test_converts_frames_correctly(self, temp_json_file):
        """Should convert frames to FrameData objects."""
        loader = RealDataLoader()
        trajectory = loader.load_trajectory(temp_json_file)

        frame = trajectory.frames[0]
        assert isinstance(frame, FrameData)
        assert frame.timestamp == 0.0
        assert np.array_equal(frame.position, np.array([0.0, 1.0, 0.0]))
        assert np.array_equal(frame.velocity, np.array([0.0, 0.0, 0.0]))
        assert np.array_equal(frame.acceleration, np.array([0.0, -9.8, 0.0]))

    def test_preserves_events(self, temp_json_file):
        """Should preserve events from JSON."""
        loader = RealDataLoader()
        trajectory = loader.load_trajectory(temp_json_file)

        assert len(trajectory.events) == 2
        assert trajectory.events[0]["type"] == "grasp"
        assert trajectory.events[1]["type"] == "release"

    def test_computes_duration(self, temp_json_file):
        """Should compute trajectory duration correctly."""
        loader = RealDataLoader()
        trajectory = loader.load_trajectory(temp_json_file)

        # Last frame timestamp - first frame timestamp
        assert trajectory.duration == pytest.approx(0.066, abs=0.001)

    def test_missing_physics_params_uses_defaults(self, tmp_path):
        """Should use default physics params if not in JSON."""
        data = {
            "capture_id": "test",
            "frames": [
                {
                    "timestamp": 0.0,
                    "object": {
                        "position": [0.0, 1.0, 0.0],
                        "velocity": [0.0, 0.0, 0.0],
                    },
                },
            ],
        }
        file_path = tmp_path / "no_params.json"
        with open(file_path, "w") as f:
            json.dump(data, f)

        loader = RealDataLoader()
        trajectory = loader.load_trajectory(file_path)

        # Should use default PhysicsParams values
        assert trajectory.params.gravity == 9.81  # Default
        assert trajectory.params.mass == 0.5  # Default

    def test_missing_optional_fields(self, tmp_path):
        """Should handle missing optional fields gracefully."""
        data = {
            "frames": [
                {
                    "timestamp": 0.0,
                    "object": {
                        "position": [0.0, 1.0, 0.0],
                    },
                },
            ],
        }
        file_path = tmp_path / "minimal.json"
        with open(file_path, "w") as f:
            json.dump(data, f)

        loader = RealDataLoader()
        trajectory = loader.load_trajectory(file_path)

        # Should use defaults for missing fields
        frame = trajectory.frames[0]
        assert np.array_equal(frame.velocity, np.array([0.0, 0.0, 0.0]))
        # Acceleration defaults to gravity at boundaries (idx=0)
        assert np.array_equal(frame.acceleration, np.array([0.0, -9.81, 0.0]))


class TestLoadDirectory:
    """Tests for load_directory method."""

    def test_loads_all_json_files(self, tmp_path):
        """Should load all JSON files in directory."""
        # Create multiple JSON files
        for i in range(3):
            data = {
                "capture_id": f"capture_{i}",
                "scenario": "test",
                "frames": [
                    {"timestamp": 0.0, "object": {"position": [0.0, 1.0, 0.0]}},
                ],
            }
            file_path = tmp_path / f"capture_{i}.json"
            with open(file_path, "w") as f:
                json.dump(data, f)

        loader = RealDataLoader()
        trajectories = loader.load_directory(tmp_path)

        assert len(trajectories) == 3

    def test_skips_non_json_files(self, tmp_path):
        """Should skip non-JSON files."""
        # Create a JSON file and a text file
        json_data = {"capture_id": "test", "frames": []}
        with open(tmp_path / "valid.json", "w") as f:
            json.dump(json_data, f)
        with open(tmp_path / "invalid.txt", "w") as f:
            f.write("not json")

        loader = RealDataLoader()
        trajectories = loader.load_directory(tmp_path)

        assert len(trajectories) == 1

    def test_handles_load_errors_gracefully(self, tmp_path):
        """Should continue loading if one file fails."""
        # Create valid and invalid JSON files
        valid_data = {"capture_id": "valid", "frames": []}
        with open(tmp_path / "valid.json", "w") as f:
            json.dump(valid_data, f)
        with open(tmp_path / "invalid.json", "w") as f:
            f.write("not valid json {{{")

        loader = RealDataLoader()
        trajectories = loader.load_directory(tmp_path)

        # Should load the valid file and skip the invalid one
        assert len(trajectories) == 1
        assert trajectories[0].metadata["capture_id"] == "valid"

    def test_empty_directory_returns_empty_list(self, tmp_path):
        """Empty directory should return empty list."""
        loader = RealDataLoader()
        trajectories = loader.load_directory(tmp_path)

        assert trajectories == []


class TestExtractPhysicsParams:
    """Tests for _extract_physics_params method."""

    def test_extract_from_physics_params_field(self):
        """Should extract from physics_params field."""
        data = {
            "physics_params": {
                "gravity_magnitude": 9.8,
                "object_mass": 0.6,
                "object_friction": 0.4,
                "object_bounce": 0.7,
            }
        }
        loader = RealDataLoader()
        params = loader._extract_physics_params(data)

        assert params.gravity == 9.8
        assert params.mass == 0.6
        assert params.friction == 0.4
        assert params.restitution == 0.7

    def test_uses_defaults_when_no_physics_params(self):
        """Should use defaults when physics_params not present."""
        data = {}
        loader = RealDataLoader()
        params = loader._extract_physics_params(data)

        assert params.gravity == 9.81  # Default
        assert params.mass == 0.5  # Default
        assert params.friction == 0.5  # Default
        assert params.restitution == 0.8  # Default

    def test_partial_physics_params(self):
        """Should handle partial physics params."""
        data = {
            "physics_params": {
                "gravity_magnitude": 9.8,
                # Missing other fields
            }
        }
        loader = RealDataLoader()
        params = loader._extract_physics_params(data)

        assert params.gravity == 9.8
        # Others should use defaults
        assert params.mass == 0.5


class TestConvertFrames:
    """Tests for _convert_frames method."""

    def test_convert_capture_format(self):
        """Should convert capture format frames."""
        data = {
            "frames": [
                {
                    "timestamp": 0.0,
                    "frame": 0,
                    "object": {
                        "position": [1.0, 2.0, 3.0],
                        "velocity": [0.1, 0.2, 0.3],
                        "acceleration": [0.0, -9.8, 0.0],
                        "rotation": [0.0, 0.0, 0.0],
                        "angular_velocity": [0.0, 0.0, 0.0],
                    },
                },
            ],
        }
        loader = RealDataLoader()
        frames = loader._convert_frames(data)

        assert len(frames) == 1
        assert np.array_equal(frames[0].position, np.array([1.0, 2.0, 3.0]))
        assert np.array_equal(frames[0].velocity, np.array([0.1, 0.2, 0.3]))

    def test_convert_trajectory_format(self):
        """Should convert direct trajectory format frames."""
        data = {
            "frames": [
                {
                    "timestamp": 0.0,
                    "frame": 0,
                    "position": [1.0, 2.0, 3.0],
                    "velocity": [0.1, 0.2, 0.3],
                    "acceleration": [0.0, -9.8, 0.0],
                    "rotation": [0.0, 0.0, 0.0],
                    "angular_velocity": [0.0, 0.0, 0.0],
                },
            ],
        }
        loader = RealDataLoader()
        frames = loader._convert_frames(data)

        assert len(frames) == 1
        assert np.array_equal(frames[0].position, np.array([1.0, 2.0, 3.0]))

    def test_empty_frames_returns_empty_list(self):
        """Empty frames should return empty list."""
        data = {"frames": []}
        loader = RealDataLoader()
        frames = loader._convert_frames(data)

        assert frames == []

    def test_missing_frames_returns_empty_list(self):
        """Missing frames field should return empty list."""
        data = {}
        loader = RealDataLoader()
        frames = loader._convert_frames(data)

        assert frames == []


class TestGetOrComputeAcceleration:
    """Tests for _get_or_compute_acceleration method."""

    def test_uses_provided_acceleration(self):
        """Should use provided acceleration if available."""
        frame_list = [
            {"timestamp": 0.0, "object": {"velocity": [0.0, 0.0, 0.0]}},
            {
                "timestamp": 0.033,
                "object": {"velocity": [0.0, -0.33, 0.0], "acceleration": [0.0, -10.0, 0.0]},
            },
            {"timestamp": 0.066, "object": {"velocity": [0.0, -0.66, 0.0]}},
        ]
        loader = RealDataLoader()
        accel = loader._get_or_compute_acceleration(frame_list, 1, [0.0, -10.0, 0.0])

        assert np.array_equal(accel, np.array([0.0, -10.0, 0.0]))

    def test_computes_from_velocity_differences(self):
        """Should compute acceleration from velocity differences."""
        frame_list = [
            {"timestamp": 0.0, "velocity": [0.0, 0.0, 0.0]},
            {"timestamp": 0.033, "velocity": [0.0, -0.33, 0.0]},  # Middle frame
            {"timestamp": 0.066, "velocity": [0.0, -0.66, 0.0]},
        ]
        loader = RealDataLoader()
        accel = loader._get_or_compute_acceleration(frame_list, 1, None)

        # Central difference: (v_next - v_prev) / dt
        # dt = 0.066 - 0.0 = 0.066
        # v_next - v_prev = [0, -0.66, 0] - [0, 0, 0] = [0, -0.66, 0]
        # accel = [0, -0.66, 0] / 0.066 = [0, -10, 0]
        expected = np.array([0.0, -10.0, 0.0])
        assert np.allclose(accel, expected, atol=0.1)

    def test_boundary_returns_gravity(self):
        """Should return default gravity at boundaries."""
        frame_list = [
            {"timestamp": 0.0, "velocity": [0.0, 0.0, 0.0]},
            {"timestamp": 0.033, "velocity": [0.0, -0.33, 0.0]},
        ]
        loader = RealDataLoader()

        # First frame (idx=0)
        accel_first = loader._get_or_compute_acceleration(frame_list, 0, None)
        assert np.array_equal(accel_first, np.array([0.0, -9.81, 0.0]))

        # Last frame (idx=1)
        accel_last = loader._get_or_compute_acceleration(frame_list, 1, None)
        assert np.array_equal(accel_last, np.array([0.0, -9.81, 0.0]))

    def test_zero_dt_returns_gravity(self):
        """Should return default gravity if dt is zero."""
        frame_list = [
            {"timestamp": 0.0, "velocity": [0.0, 0.0, 0.0]},
            {"timestamp": 0.0, "velocity": [0.0, -0.33, 0.0]},  # Same timestamp
            {"timestamp": 0.0, "velocity": [0.0, -0.66, 0.0]},
        ]
        loader = RealDataLoader()
        accel = loader._get_or_compute_acceleration(frame_list, 1, None)

        assert np.array_equal(accel, np.array([0.0, -9.81, 0.0]))


class TestRealDataIntegration:
    """Integration tests for real data loading."""

    def test_full_workflow(self, tmp_path):
        """Test complete workflow from JSON to TrajectoryData."""
        # Create realistic capture data
        data = {
            "capture_id": "integration_test",
            "scenario": "grasp_ball",
            "physics_params": {
                "gravity_magnitude": 9.81,
                "object_mass": 0.5,
                "object_friction": 0.5,
                "object_bounce": 0.8,
            },
            "frames": [
                {
                    "timestamp": i * 0.033,
                    "frame": i,
                    "object": {
                        "position": [0.0, 2.0 - 0.5 * 9.81 * (i * 0.033) ** 2, 0.0],
                        "velocity": [0.0, -9.81 * i * 0.033, 0.0],
                        "acceleration": [0.0, -9.81, 0.0],
                        "rotation": [0.0, 0.0, 0.0],
                        "angular_velocity": [0.0, 0.0, 0.0],
                    },
                }
                for i in range(10)
            ],
            "events": [{"type": "test", "frame": 5}],
        }

        file_path = tmp_path / "integration.json"
        with open(file_path, "w") as f:
            json.dump(data, f)

        loader = RealDataLoader()
        trajectory = loader.load_trajectory(file_path)

        # Verify complete conversion
        assert trajectory.metadata["capture_id"] == "integration_test"
        assert trajectory.params.gravity == 9.81
        assert trajectory.params.mass == 0.5
        assert len(trajectory.frames) == 10
        assert len(trajectory.events) == 1

        # Verify frame data
        first_frame = trajectory.frames[0]
        assert first_frame.timestamp == 0.0
        assert first_frame.frame_index == 0

        last_frame = trajectory.frames[-1]
        assert last_frame.frame_index == 9

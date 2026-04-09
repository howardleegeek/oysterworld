"""Tests for RLDS converter module."""

import json
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from physicalfish.capture.rlds_converter import RLDSConverter
from physicalfish.models import FrameData, PhysicsParams, TrajectoryData


class TestRLDSConverterInit:
    """Tests for RLDSConverter initialization."""

    def test_default_initialization(self):
        """Should initialize with default dt."""
        converter = RLDSConverter()
        assert converter.dt == 0.1

    def test_custom_dt(self):
        """Should accept custom dt value."""
        converter = RLDSConverter(dt=0.033)
        assert converter.dt == 0.033


class TestLoadTrajectoryFile:
    """Tests for load_trajectory_file method."""

    @pytest.fixture
    def sample_rlds_file(self, tmp_path):
        """Create a sample RLDS trajectory file."""
        data = {
            "trajectory": [
                {
                    "timestamp": 0.0,
                    "state": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0],  # 7-DOF state
                    "observation": {"position": [0.1, 0.2, 0.3]},
                },
                {
                    "timestamp": 0.1,
                    "state": [0.15, 0.25, 0.35, 0.0, 0.0, 0.0, 0.0],
                    "observation": {"position": [0.15, 0.25, 0.35]},
                },
                {
                    "timestamp": 0.2,
                    "state": [0.2, 0.3, 0.4, 0.0, 0.0, 0.0, 0.0],
                    "observation": {"position": [0.2, 0.3, 0.4]},
                },
            ],
            "dt": 0.1,
        }
        file_path = tmp_path / "trajectory_001.json"
        with open(file_path, "w") as f:
            json.dump(data, f)
        return file_path

    def test_load_valid_file(self, sample_rlds_file):
        """Should load a valid RLDS file."""
        converter = RLDSConverter()
        trajectory = converter.load_trajectory_file(sample_rlds_file)

        assert isinstance(trajectory, TrajectoryData)
        assert len(trajectory.frames) == 3

    def test_file_not_found(self, tmp_path):
        """Should raise FileNotFoundError for missing file."""
        converter = RLDSConverter()
        missing_file = tmp_path / "nonexistent.json"

        with pytest.raises(FileNotFoundError):
            converter.load_trajectory_file(missing_file)


class TestLoadAllTrajectories:
    """Tests for load_all_trajectories method."""

    def test_load_directory_with_multiple_files(self, tmp_path):
        """Should load all JSON files from directory."""
        # Create multiple trajectory files
        for i in range(3):
            data = {
                "trajectory": [
                    {"timestamp": j * 0.1, "state": [0.1 * j, 0.0, 0.0]} for j in range(5)
                ]
            }
            file_path = tmp_path / f"trajectory_{i:03d}.json"
            with open(file_path, "w") as f:
                json.dump(data, f)

        converter = RLDSConverter()
        trajectories = converter.load_all_trajectories(tmp_path)

        assert len(trajectories) == 3

    def test_skip_invalid_files(self, tmp_path):
        """Should skip invalid files and continue loading."""
        # Create valid file
        valid_data = {"trajectory": [{"timestamp": 0.0, "state": [0.0, 0.0, 0.0]}]}
        with open(tmp_path / "valid.json", "w") as f:
            json.dump(valid_data, f)

        # Create invalid file
        with open(tmp_path / "invalid.json", "w") as f:
            f.write("not valid json")

        converter = RLDSConverter()
        trajectories = converter.load_all_trajectories(tmp_path)

        # Should load only the valid file
        assert len(trajectories) == 1

    def test_directory_not_found(self, tmp_path):
        """Should raise FileNotFoundError for missing directory."""
        converter = RLDSConverter()
        missing_dir = tmp_path / "nonexistent"

        with pytest.raises(FileNotFoundError):
            converter.load_all_trajectories(missing_dir)

    def test_empty_directory(self, tmp_path):
        """Should return empty list for empty directory."""
        converter = RLDSConverter()
        trajectories = converter.load_all_trajectories(tmp_path)

        assert trajectories == []


class TestConvertToTrajectory:
    """Tests for convert_to_trajectory method."""

    def test_convert_with_state_vector(self):
        """Should convert trajectory with state vector."""
        data = {
            "trajectory": [
                {"timestamp": 0.0, "state": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0]},
                {"timestamp": 0.1, "state": [0.15, 0.25, 0.35, 0.0, 0.0, 0.0, 0.0]},
                {"timestamp": 0.2, "state": [0.2, 0.3, 0.4, 0.0, 0.0, 0.0, 0.0]},
            ],
            "dt": 0.1,
        }

        converter = RLDSConverter()
        trajectory = converter.convert_to_trajectory(data)

        assert len(trajectory.frames) == 3
        # First 3 elements of state are position
        assert np.allclose(trajectory.frames[0].position, [0.1, 0.2, 0.3])

    def test_convert_with_observation_position(self):
        """Should extract position from observation when state is short."""
        data = {
            "trajectory": [
                {"timestamp": 0.0, "state": [0.0], "observation": {"position": [0.5, 0.6, 0.7]}},
                {"timestamp": 0.1, "state": [0.0], "observation": {"position": [0.6, 0.7, 0.8]}},
            ],
            "dt": 0.1,
        }

        converter = RLDSConverter()
        trajectory = converter.convert_to_trajectory(data)

        assert len(trajectory.frames) == 2
        assert np.allclose(trajectory.frames[0].position, [0.5, 0.6, 0.7])

    def test_convert_computes_velocity(self):
        """Should compute velocity from position differences."""
        data = {
            "trajectory": [
                {"timestamp": 0.0, "state": [0.0, 0.0, 0.0]},
                {"timestamp": 0.1, "state": [0.1, 0.2, 0.3]},
                {"timestamp": 0.2, "state": [0.2, 0.4, 0.6]},
            ],
            "dt": 0.1,
        }

        converter = RLDSConverter()
        trajectory = converter.convert_to_trajectory(data)

        # Velocity should be computed from position differences
        # v = (x_next - x_prev) / (2 * dt) for middle frames
        middle_frame = trajectory.frames[1]
        expected_velocity = np.array([1.0, 2.0, 3.0])  # (0.2-0.0)/(2*0.1) = 1.0
        assert np.allclose(middle_frame.velocity, expected_velocity, atol=0.1)

    def test_convert_computes_acceleration(self):
        """Should compute acceleration from velocity differences."""
        data = {
            "trajectory": [
                {"timestamp": 0.0, "state": [0.0, 0.0, 0.0]},
                {"timestamp": 0.1, "state": [0.1, 0.0, 0.0]},
                {"timestamp": 0.2, "state": [0.3, 0.0, 0.0]},
            ],
            "dt": 0.1,
        }

        converter = RLDSConverter()
        trajectory = converter.convert_to_trajectory(data)

        # Acceleration should be computed from velocity differences
        middle_frame = trajectory.frames[1]
        # a = (v_next - v_prev) / (2 * dt)
        # v at frame 0: (0.1-0)/0.1 = 1.0 (forward diff)
        # v at frame 2: (0.3-0.1)/0.1 = 2.0 (backward diff)
        # a at frame 1: (2.0-1.0)/(2*0.1) = 5.0
        assert middle_frame.acceleration[0] > 0  # Should have positive acceleration

    def test_empty_trajectory_raises(self):
        """Should raise ValueError for empty trajectory."""
        data = {"trajectory": []}

        converter = RLDSConverter()
        with pytest.raises(ValueError, match="No trajectory steps found"):
            converter.convert_to_trajectory(data)

    def test_missing_trajectory_key_raises(self):
        """Should raise ValueError for missing trajectory key."""
        data = {"not_trajectory": []}

        converter = RLDSConverter()
        with pytest.raises(ValueError, match="No trajectory steps found"):
            converter.convert_to_trajectory(data)

    def test_single_step_trajectory(self):
        """Should handle single-step trajectory."""
        data = {
            "trajectory": [
                {"timestamp": 0.0, "state": [0.1, 0.2, 0.3]},
            ],
            "dt": 0.1,
        }

        converter = RLDSConverter()
        trajectory = converter.convert_to_trajectory(data)

        assert len(trajectory.frames) == 1
        assert np.allclose(trajectory.frames[0].position, [0.1, 0.2, 0.3])

    def test_uses_dt_from_data(self):
        """Should use dt from data when available."""
        data = {
            "trajectory": [
                {"timestamp": 0.0, "state": [0.0, 0.0, 0.0]},
                {"timestamp": 0.033, "state": [0.1, 0.0, 0.0]},
            ],
            "dt": 0.033,  # 30Hz
        }

        converter = RLDSConverter(dt=0.1)  # Default is 0.1
        trajectory = converter.convert_to_trajectory(data)

        # Should use dt from data, not default
        assert trajectory.frames[1].timestamp == 0.033

    def test_default_position_when_no_data(self):
        """Should use default position when no state or observation."""
        data = {
            "trajectory": [
                {"timestamp": 0.0},  # No state or observation
            ],
            "dt": 0.1,
        }

        converter = RLDSConverter()
        trajectory = converter.convert_to_trajectory(data)

        # Should default to [0, 0, 0]
        assert np.allclose(trajectory.frames[0].position, [0.0, 0.0, 0.0])


class TestRLDSConverterIntegration:
    """Integration tests for RLDS converter."""

    def test_full_workflow(self, tmp_path):
        """Test complete workflow from files to TrajectoryData."""
        # Create test directory with trajectory files
        traj_dir = tmp_path / "trajectories"
        traj_dir.mkdir()

        for i in range(2):
            data = {
                "trajectory": [
                    {
                        "timestamp": j * 0.033,
                        "state": [0.1 * j + i, 0.2 * j, 0.3 * j, 0.0, 0.0, 0.0, 0.0],
                    }
                    for j in range(10)
                ],
                "dt": 0.033,
            }
            with open(traj_dir / f"traj_{i:03d}.json", "w") as f:
                json.dump(data, f)

        converter = RLDSConverter()
        trajectories = converter.load_all_trajectories(traj_dir)

        assert len(trajectories) == 2
        for traj in trajectories:
            assert len(traj.frames) == 10
            assert traj.duration > 0

"""Tests for simulator scenarios module."""

import pytest
from unittest.mock import MagicMock, patch
import sys

from physicalfish.simulator.scenarios import ScenarioBuilder, SCENARIOS
from physicalfish.models import PhysicsParams


class TestScenarioBuilderInit:
    """Tests for ScenarioBuilder initialization."""

    def test_default_initialization(self):
        """Should initialize with None object_id."""
        builder = ScenarioBuilder()
        assert builder.object_id is None
        assert builder.constraint_id is None
        assert builder._auxiliary_objects == []


class TestScenarioMetadata:
    """Tests for scenario metadata."""

    def test_scenarios_dict_exists(self):
        """SCENARIOS dict should exist with expected scenarios."""
        assert "grasp_ball" in SCENARIOS
        assert "throw_catch" in SCENARIOS
        assert "roll_incline" in SCENARIOS

    def test_scenario_metadata_structure(self):
        """Each scenario should have required metadata fields."""
        for name, metadata in SCENARIOS.items():
            assert "description" in metadata
            assert "phases" in metadata
            assert "duration" in metadata
            assert isinstance(metadata["phases"], list)
            assert isinstance(metadata["duration"], (int, float))

    def test_grasp_ball_metadata(self):
        """grasp_ball should have correct metadata."""
        meta = SCENARIOS["grasp_ball"]
        assert "grasp" in meta["description"].lower() or "ball" in meta["description"].lower()
        assert "free_fall" in meta["phases"]
        assert "grasped" in meta["phases"]
        assert meta["duration"] == 3.0

    def test_throw_catch_metadata(self):
        """throw_catch should have correct metadata."""
        meta = SCENARIOS["throw_catch"]
        assert "throw" in meta["description"].lower() or "catch" in meta["description"].lower()
        assert "throw" in meta["phases"] or "flight" in meta["phases"]
        assert meta["duration"] == 2.0

    def test_roll_incline_metadata(self):
        """roll_incline should have correct metadata."""
        meta = SCENARIOS["roll_incline"]
        assert "roll" in meta["description"].lower() or "incline" in meta["description"].lower()
        assert "rolling" in meta["phases"]
        assert meta["duration"] == 2.0


class TestScenarioBuilderWithoutPyBullet:
    """Tests for ScenarioBuilder when PyBullet is not available."""

    def test_build_does_nothing_without_pybullet(self):
        """build() should return early when PyBullet not available."""
        with patch("physicalfish.simulator.scenarios.PYBULLET_AVAILABLE", False):
            builder = ScenarioBuilder()
            params = PhysicsParams()

            # Should not raise even though PyBullet not available
            builder.build("grasp_ball", params)
            # object_id should remain None
            assert builder.object_id is None

    def test_cleanup_does_nothing_without_pybullet(self):
        """cleanup() should return early when PyBullet not available."""
        with patch("physicalfish.simulator.scenarios.PYBULLET_AVAILABLE", False):
            builder = ScenarioBuilder()
            # Should not raise
            builder.cleanup()


class TestScenarioBuilderWithMockPyBullet:
    """Tests for ScenarioBuilder with mocked PyBullet."""

    @pytest.fixture
    def mock_pybullet_module(self):
        """Create mock PyBullet module and patch it into scenarios."""
        mock_p = MagicMock()
        mock_p.GEOM_SPHERE = 2
        mock_p.GEOM_BOX = 3
        mock_p.createCollisionShape.return_value = 1
        mock_p.createVisualShape.return_value = 2
        mock_p.createMultiBody.return_value = 100
        mock_p.changeDynamics.return_value = None
        mock_p.removeBody.return_value = None
        mock_p.removeConstraint.return_value = None
        mock_p.resetBaseVelocity.return_value = None
        mock_p.getQuaternionFromEuler.return_value = [0, 0, 0, 1]

        # Patch the module-level p import in scenarios
        with patch.dict(sys.modules, {"pybullet": mock_p}):
            with patch("physicalfish.simulator.scenarios.PYBULLET_AVAILABLE", True):
                # Force reimport to pick up mocked pybullet
                import physicalfish.simulator.scenarios as scenarios_module

                scenarios_module.p = mock_p
                yield mock_p
                # Reset after test
                scenarios_module.p = None

    def test_build_grasp_ball(self, mock_pybullet_module):
        """Should build grasp_ball scenario."""
        builder = ScenarioBuilder()
        params = PhysicsParams(mass=0.5, friction=0.4, restitution=0.7)

        builder.build("grasp_ball", params)

        # Should create collision and visual shapes
        assert mock_pybullet_module.createCollisionShape.called
        assert mock_pybullet_module.createVisualShape.called
        assert mock_pybullet_module.createMultiBody.called
        assert mock_pybullet_module.changeDynamics.called

    def test_build_throw_catch(self, mock_pybullet_module):
        """Should build throw_catch scenario with initial velocity."""
        builder = ScenarioBuilder()
        params = PhysicsParams()

        builder.build("throw_catch", params)

        # Should set initial velocity
        assert mock_pybullet_module.resetBaseVelocity.called

    def test_build_roll_incline(self, mock_pybullet_module):
        """Should build roll_incline scenario with plane."""
        builder = ScenarioBuilder()
        params = PhysicsParams()

        builder.build("roll_incline", params)

        # Should create box for incline plane
        assert mock_pybullet_module.createCollisionShape.called
        # Should have auxiliary objects (the plane)
        assert len(builder._auxiliary_objects) > 0

    def test_build_unknown_scenario_defaults_to_ball_drop(self, mock_pybullet_module):
        """Unknown scenario should default to ball drop."""
        builder = ScenarioBuilder()
        params = PhysicsParams()

        builder.build("unknown_scenario", params)

        # Should still create an object
        assert mock_pybullet_module.createMultiBody.called

    def test_cleanup_removes_objects(self, mock_pybullet_module):
        """cleanup() should remove all created objects."""
        builder = ScenarioBuilder()
        params = PhysicsParams()

        # Build a scenario first
        builder.build("grasp_ball", params)
        assert builder.object_id is not None

        # Clean up
        builder.cleanup()

        # Should remove the object
        assert mock_pybullet_module.removeBody.called

    def test_cleanup_removes_constraint(self, mock_pybullet_module):
        """cleanup() should remove constraint if exists."""
        builder = ScenarioBuilder()
        builder.constraint_id = 999

        builder.cleanup()

        assert mock_pybullet_module.removeConstraint.called
        assert builder.constraint_id is None

    def test_cleanup_removes_auxiliary_objects(self, mock_pybullet_module):
        """cleanup() should remove auxiliary objects."""
        builder = ScenarioBuilder()
        builder._auxiliary_objects = [101, 102, 103]

        builder.cleanup()

        # Should remove all auxiliary objects
        assert mock_pybullet_module.removeBody.call_count == 3
        assert len(builder._auxiliary_objects) == 0

    def test_build_cleans_up_existing_objects(self, mock_pybullet_module):
        """build() should clean up existing objects before building new."""
        builder = ScenarioBuilder()
        params = PhysicsParams()

        # Build first scenario
        builder.build("grasp_ball", params)
        first_object_id = builder.object_id

        # Build second scenario - should clean up first
        builder.build("throw_catch", params)

        # Should have called cleanup (removeBody) for first object
        assert mock_pybullet_module.removeBody.called

    def test_dynamics_params_passed_to_pybullet(self, mock_pybullet_module):
        """Physics params should be passed to PyBullet changeDynamics."""
        builder = ScenarioBuilder()
        params = PhysicsParams(
            mass=0.6,
            friction=0.4,
            restitution=0.7,
            linear_damping=0.2,
            angular_damping=0.3,
        )

        builder.build("grasp_ball", params)

        # Check changeDynamics was called with correct params
        call_kwargs = mock_pybullet_module.changeDynamics.call_args.kwargs
        assert call_kwargs["lateralFriction"] == 0.4
        assert call_kwargs["restitution"] == 0.7
        assert call_kwargs["linearDamping"] == 0.2
        assert call_kwargs["angularDamping"] == 0.3


class TestScenarioBuilderEdgeCases:
    """Edge case tests for ScenarioBuilder."""

    def test_multiple_build_calls(self):
        """Multiple build calls should be handled gracefully."""
        with patch("physicalfish.simulator.scenarios.PYBULLET_AVAILABLE", False):
            builder = ScenarioBuilder()
            params = PhysicsParams()

            # Multiple builds should not cause issues
            builder.build("grasp_ball", params)
            builder.build("throw_catch", params)
            builder.build("roll_incline", params)

            # No exception should be raised

    def test_cleanup_when_nothing_built(self):
        """cleanup() should work even when nothing was built."""
        with patch("physicalfish.simulator.scenarios.PYBULLET_AVAILABLE", False):
            builder = ScenarioBuilder()

            # Should not raise
            builder.cleanup()
            builder.cleanup()  # Double cleanup should be safe

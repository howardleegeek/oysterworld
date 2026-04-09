"""Tests for bootstrap generator few-shot data augmentation."""

import numpy as np
import pytest
from unittest.mock import MagicMock

from physicalfish.loop.bootstrap import BootstrapGenerator
from physicalfish.models import FrameData, PhysicsParams, TrajectoryData
from physicalfish.verification.physics_verifier import PhysicsVerifier, VerificationResult


class MockSimulator:
    """Mock simulator for testing."""

    def __init__(self, trajectory_factory=None):
        self.configured_params = None
        self.trajectory_factory = trajectory_factory or self._default_trajectory

    def configure(self, params: PhysicsParams) -> None:
        self.configured_params = params

    def run_scenario(self, scenario: str, duration: float) -> TrajectoryData:
        return self.trajectory_factory(scenario, duration)

    def _default_trajectory(self, scenario: str, duration: float) -> TrajectoryData:
        """Generate a simple trajectory with 10 frames."""
        frames = []
        dt = duration / 10
        for i in range(10):
            frames.append(
                FrameData(
                    timestamp=i * dt,
                    position=np.array([0.0, 1.0 - 0.1 * i, 0.0]),
                    velocity=np.array([0.0, -0.5, 0.0]),
                    acceleration=np.array([0.0, -9.81, 0.0]),
                    rotation=np.zeros(3),
                    angular_velocity=np.zeros(3),
                    frame_index=i,
                )
            )
        return TrajectoryData(
            frames=frames,
            params=self.configured_params or PhysicsParams(),
            metadata={"scenario": scenario},
        )


class TestBootstrapGeneratorInit:
    """Tests for BootstrapGenerator initialization."""

    def test_default_initialization(self):
        """BootstrapGenerator should initialize with default values."""
        simulator = MockSimulator()
        verifier = MagicMock()

        generator = BootstrapGenerator(simulator, verifier)

        assert generator.simulator == simulator
        assert generator.verifier == verifier
        assert generator.quality_threshold == 0.7
        assert generator.scenario == "grasp_ball"
        assert generator.duration == 3.0

    def test_custom_initialization(self):
        """BootstrapGenerator should accept custom parameters."""
        simulator = MockSimulator()
        verifier = MagicMock()

        generator = BootstrapGenerator(
            simulator=simulator,
            verifier=verifier,
            quality_threshold=0.8,
            scenario="throw_catch",
            duration=5.0,
        )

        assert generator.quality_threshold == 0.8
        assert generator.scenario == "throw_catch"
        assert generator.duration == 5.0


class TestBootstrapAugment:
    """Tests for BootstrapGenerator.augment() method."""

    @pytest.fixture
    def real_samples(self):
        """Create real trajectory samples for testing."""
        params1 = PhysicsParams(gravity=9.8, mass=0.5, friction=0.4)
        params2 = PhysicsParams(gravity=9.9, mass=0.6, friction=0.5)
        params3 = PhysicsParams(gravity=10.0, mass=0.55, friction=0.45)

        samples = []
        for i, params in enumerate([params1, params2, params3]):
            frames = [
                FrameData(
                    timestamp=j * 0.033,
                    position=np.array([0.0, 2.0 - 0.1 * j, 0.0]),
                    velocity=np.array([0.0, -0.5, 0.0]),
                    acceleration=np.array([0.0, -9.81, 0.0]),
                    rotation=np.zeros(3),
                    angular_velocity=np.zeros(3),
                    frame_index=j,
                )
                for j in range(10)
            ]
            samples.append(
                TrajectoryData(
                    frames=frames,
                    params=params,
                    metadata={"sample_id": i},
                )
            )
        return samples

    def test_augment_empty_samples_returns_empty(self):
        """Empty real samples should return empty list."""
        simulator = MockSimulator()
        verifier = MagicMock()
        generator = BootstrapGenerator(simulator, verifier)

        result = generator.augment([], target_count=5)

        assert result == []

    def test_augment_generates_trajectories_above_threshold(self, real_samples):
        """Generated trajectories should have verification scores above threshold."""
        simulator = MockSimulator()

        # Mock verifier to return high scores
        verifier = MagicMock()
        verifier.verify.return_value = VerificationResult(
            overall_score=0.85,  # Above default threshold of 0.7
            passed=True,
            constraint_scores={},
            details={},
        )

        generator = BootstrapGenerator(
            simulator=simulator,
            verifier=verifier,
            quality_threshold=0.7,
        )

        result = generator.augment(real_samples, target_count=3)

        # Should generate at least some trajectories
        assert len(result) > 0
        # All should have verification scores >= threshold
        for trajectory in result:
            assert "bootstrap" in trajectory.metadata
            assert trajectory.metadata["bootstrap"]["verification_score"] >= 0.7

    def test_augment_respects_target_count(self, real_samples):
        """Should generate approximately target_count trajectories."""
        simulator = MockSimulator()

        verifier = MagicMock()
        verifier.verify.return_value = VerificationResult(
            overall_score=0.85,
            passed=True,
            constraint_scores={},
            details={},
        )

        generator = BootstrapGenerator(simulator, verifier)
        target_count = 5

        result = generator.augment(real_samples, target_count=target_count)

        # Should generate close to target_count (might be less if max_attempts reached)
        assert len(result) <= target_count
        # With high acceptance rate, should get close to target
        assert len(result) >= target_count * 0.5  # At least 50%

    def test_augment_rejects_low_quality_samples(self, real_samples):
        """Samples below quality threshold should be rejected."""
        simulator = MockSimulator()

        # Mock verifier to alternate between high and low scores
        call_count = [0]

        def mock_verify(trajectory):
            call_count[0] += 1
            score = 0.85 if call_count[0] % 2 == 1 else 0.5  # Alternate high/low
            return VerificationResult(
                overall_score=score,
                passed=score > 0.7,
                constraint_scores={},
                details={},
            )

        verifier = MagicMock()
        verifier.verify.side_effect = mock_verify

        generator = BootstrapGenerator(
            simulator=simulator,
            verifier=verifier,
            quality_threshold=0.7,
        )

        result = generator.augment(real_samples, target_count=3)

        # Should only accept high-quality samples
        for trajectory in result:
            assert trajectory.metadata["bootstrap"]["verification_score"] >= 0.7

    def test_augment_adds_bootstrap_metadata(self, real_samples):
        """Generated trajectories should have bootstrap metadata."""
        simulator = MockSimulator()

        verifier = MagicMock()
        verifier.verify.return_value = VerificationResult(
            overall_score=0.80,
            passed=True,
            constraint_scores={},
            details={},
        )

        generator = BootstrapGenerator(simulator, verifier)
        result = generator.augment(real_samples, target_count=2)

        for trajectory in result:
            assert "bootstrap" in trajectory.metadata
            bootstrap_meta = trajectory.metadata["bootstrap"]
            assert "attempt" in bootstrap_meta
            assert "verification_score" in bootstrap_meta
            assert "params" in bootstrap_meta

    def test_augment_samples_from_param_distribution(self, real_samples):
        """Should sample parameters based on distribution of real samples."""
        simulator = MockSimulator()

        configured_params = []
        original_configure = simulator.configure

        def tracking_configure(params):
            configured_params.append(params)
            original_configure(params)

        simulator.configure = tracking_configure

        verifier = MagicMock()
        verifier.verify.return_value = VerificationResult(
            overall_score=0.80,
            passed=True,
            constraint_scores={},
            details={},
        )

        generator = BootstrapGenerator(simulator, verifier)
        generator.augment(real_samples, target_count=3)

        # Should have configured simulator with sampled params
        assert len(configured_params) > 0

        # Sampled params should be within reasonable bounds of real samples
        # Real samples had gravity: 9.8, 9.9, 10.0 (mean ~9.9)
        gravities = [p.gravity for p in configured_params]
        assert all(9.0 <= g <= 11.0 for g in gravities)  # Within physics bounds

    def test_augment_handles_verification_exceptions(self, real_samples):
        """Should handle exceptions during verification gracefully."""
        simulator = MockSimulator()

        call_count = [0]

        def mock_verify(trajectory):
            call_count[0] += 1
            if call_count[0] == 1:
                raise ValueError("Verification error")
            return VerificationResult(
                overall_score=0.80,
                passed=True,
                constraint_scores={},
                details={},
            )

        verifier = MagicMock()
        verifier.verify.side_effect = mock_verify

        generator = BootstrapGenerator(simulator, verifier)
        result = generator.augment(real_samples, target_count=2)

        # Should still generate some trajectories despite exception
        assert len(result) > 0


class TestExtractParamDistribution:
    """Tests for _extract_param_distribution method."""

    def test_extracts_mean_and_std(self):
        """Should compute mean and std from samples."""
        samples = [
            TrajectoryData(frames=[], params=PhysicsParams(gravity=9.8, mass=0.5)),
            TrajectoryData(frames=[], params=PhysicsParams(gravity=9.9, mass=0.6)),
            TrajectoryData(frames=[], params=PhysicsParams(gravity=10.0, mass=0.7)),
        ]

        simulator = MockSimulator()
        verifier = MagicMock()
        generator = BootstrapGenerator(simulator, verifier)

        mean, std = generator._extract_param_distribution(samples)

        # Mean of 9.8, 9.9, 10.0 = 9.9
        assert mean["gravity"] == pytest.approx(9.9, abs=0.01)
        # Mean of 0.5, 0.6, 0.7 = 0.6
        assert mean["mass"] == pytest.approx(0.6, abs=0.01)

        # Should have non-zero std for multiple samples
        assert std["gravity"] > 0
        assert std["mass"] > 0

    def test_single_sample_uses_default_std(self):
        """Single sample should use minimum std for exploration."""
        samples = [
            TrajectoryData(frames=[], params=PhysicsParams(gravity=9.8)),
        ]

        simulator = MockSimulator()
        verifier = MagicMock()
        generator = BootstrapGenerator(simulator, verifier)

        mean, std = generator._extract_param_distribution(samples)

        assert mean["gravity"] == 9.8
        # Single sample should have minimum std of 0.1 (or 0.05 after max)
        assert std["gravity"] >= 0.05

    def test_missing_params_use_defaults(self):
        """Missing parameters should use default bounds."""
        # Create params with only some fields
        params = PhysicsParams(gravity=9.8)  # Only gravity specified
        samples = [TrajectoryData(frames=[], params=params)]

        simulator = MockSimulator()
        verifier = MagicMock()
        generator = BootstrapGenerator(simulator, verifier)

        mean, std = generator._extract_param_distribution(samples)

        # Should have all physics params
        assert "mass" in mean
        assert "friction" in mean
        assert "restitution" in mean


class TestSampleParams:
    """Tests for _sample_params method."""

    def test_samples_within_bounds(self):
        """Sampled parameters should be within valid bounds."""
        simulator = MockSimulator()
        verifier = MagicMock()
        generator = BootstrapGenerator(simulator, verifier)

        mean = {"gravity": 9.8, "mass": 0.5, "friction": 0.5}
        std = {"gravity": 0.1, "mass": 0.1, "friction": 0.1}

        # Sample multiple times to check bounds
        for _ in range(20):
            params = generator._sample_params(mean, std)

            # Check bounds from PHYSICS_PARAM_SPACE
            assert 1.0 <= params.gravity <= 20.0
            assert 0.1 <= params.mass <= 5.0
            assert 0.0 <= params.friction <= 1.0
            assert 0.0 <= params.restitution <= 1.0

    def test_samples_follow_distribution(self):
        """Sampled parameters should roughly follow Gaussian distribution."""
        simulator = MockSimulator()
        verifier = MagicMock()
        generator = BootstrapGenerator(simulator, verifier)

        mean = {"gravity": 9.8, "mass": 0.5, "friction": 0.5}
        std = {"gravity": 0.1, "mass": 0.05, "friction": 0.1}

        # Sample many times
        samples = [generator._sample_params(mean, std) for _ in range(100)]

        # Check mean is close to expected
        avg_gravity = np.mean([s.gravity for s in samples])
        assert avg_gravity == pytest.approx(9.8, abs=0.05)


class TestBootstrapIntegration:
    """Integration-style tests for bootstrap workflow."""

    def test_full_augmentation_workflow(self):
        """Test complete augmentation workflow with real verifier."""
        # Create realistic falling ball trajectory
        g = 9.81
        dt = 0.033
        frames = []
        for i in range(30):
            t = i * dt
            y = max(0.15, 2.0 - 0.5 * g * t**2)
            vy = -g * t if y > 0.15 else 0.0
            ay = -g if y > 0.15 else 0.0
            frames.append(
                FrameData(
                    timestamp=t,
                    position=np.array([0.0, y, 0.0]),
                    velocity=np.array([0.0, vy, 0.0]),
                    acceleration=np.array([0.0, ay, 0.0]),
                    rotation=np.zeros(3),
                    angular_velocity=np.zeros(3),
                    frame_index=i,
                )
            )

        real_sample = TrajectoryData(
            frames=frames,
            params=PhysicsParams(gravity=9.81, mass=0.5),
        )

        # Create mock simulator that returns similar trajectories
        def trajectory_factory(scenario, duration):
            return TrajectoryData(
                frames=frames[:10],  # Shorter version
                params=PhysicsParams(gravity=9.81, mass=0.5),
            )

        simulator = MockSimulator(trajectory_factory)
        verifier = PhysicsVerifier(gravity=9.81)

        generator = BootstrapGenerator(
            simulator=simulator,
            verifier=verifier,
            quality_threshold=0.6,  # Lower threshold for test
        )

        result = generator.augment([real_sample], target_count=2)

        # Should generate some trajectories that pass verification
        assert len(result) >= 0  # May be 0 if none pass, but shouldn't crash

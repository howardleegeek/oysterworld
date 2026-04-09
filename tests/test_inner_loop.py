"""Tests for inner loop optimization."""

import numpy as np
import pytest
from unittest.mock import MagicMock

from physicalfish.models import FrameData, PhysicsParams, TrajectoryData
from physicalfish.verification.physics_verifier import PhysicsVerifier
from physicalfish.optimizer.bayesian import BayesianOptimizer
from physicalfish.loop.inner_loop import InnerLoop, InnerLoopConfig


class MockSimulator:
    """Mock simulator that returns trajectories based on parameter distance."""

    def __init__(self, target_params: PhysicsParams | None = None):
        self.target_params = target_params or PhysicsParams(
            gravity=9.81, mass=0.5, friction=0.5, restitution=0.8
        )
        self.configured_params: PhysicsParams | None = None

    def configure(self, params: PhysicsParams) -> None:
        self.configured_params = params

    def run_scenario(self, scenario: str, duration: float) -> TrajectoryData:
        """Return a trajectory with score based on parameter distance."""
        # Generate a simple trajectory
        frames = []
        dt = 1.0 / 30.0
        num_frames = int(duration / dt)

        # Compute "quality" based on parameter distance from target
        # Closer params = better trajectory
        param_distance = self._compute_param_distance()
        quality = max(0.3, 1.0 - param_distance)

        for i in range(num_frames):
            t = i * dt
            # Simple falling motion with quality affecting noise
            y = max(0.1, 2.0 - 0.5 * 9.81 * t**2)
            vy = -9.81 * t if y > 0.1 else 0.0
            ay = -9.81 if y > 0.1 else 0.0

            # Add noise inversely proportional to quality
            noise = (1.0 - quality) * 0.5
            vy += np.random.normal(0, noise)

            frame = FrameData(
                timestamp=t,
                position=np.array([0.0, y, 0.0]),
                velocity=np.array([0.0, vy, 0.0]),
                acceleration=np.array([0.0, ay, 0.0]),
                rotation=np.zeros(3),
                angular_velocity=np.zeros(3),
                frame_index=i,
            )
            frames.append(frame)

        return TrajectoryData(
            frames=frames,
            params=self.configured_params or PhysicsParams(),
        )

    def _compute_param_distance(self) -> float:
        """Compute normalized distance from target params."""
        if not self.configured_params:
            return 1.0

        target = self.target_params
        current = self.configured_params

        # Normalize each parameter difference
        diffs = [
            abs(current.gravity - target.gravity) / 3.0,  # gravity range ~3
            abs(current.mass - target.mass) / 1.9,  # mass range ~1.9
            abs(current.friction - target.friction) / 0.9,  # friction range ~0.9
            abs(current.restitution - target.restitution),  # restitution range 1.0
        ]

        return np.mean(diffs)


@pytest.fixture
def mock_simulator():
    """Create a mock simulator with target params."""
    return MockSimulator(
        target_params=PhysicsParams(gravity=9.81, mass=0.5, friction=0.5, restitution=0.8)
    )


@pytest.fixture
def verifier():
    """Create a physics verifier."""
    return PhysicsVerifier()


@pytest.fixture
def mock_optimizer():
    """Create a mock optimizer."""
    opt = MagicMock(spec=BayesianOptimizer)
    return opt


@pytest.fixture
def inner_config():
    """Create inner loop config for testing."""
    return InnerLoopConfig(
        scenario="test_scenario",
        target_score=0.85,
        max_iterations=20,
        patience=5,
        verification_weight=1.0,  # Only use verification for simplicity
        duration=1.0,
    )


class TestInnerLoopConvergence:
    """Test that inner loop converges to good parameters."""

    def test_converges_with_mock(self, mock_simulator, verifier, inner_config):
        """Test that inner loop converges with mock evaluator."""

        # Create a simple optimizer that just returns random params
        # The inner loop will iterate and find good ones
        def evaluator(params: dict[str, float]) -> float:
            return 0.5  # Placeholder

        optimizer = BayesianOptimizer(evaluator=evaluator)

        inner = InnerLoop(
            simulator=mock_simulator,
            verifier=verifier,
            optimizer=optimizer,
            real_data=None,
            config=inner_config,
        )

        result = inner.run()

        # Should have run at least a few iterations
        assert result.iterations > 0
        # Should have recorded history
        assert len(result.history) == result.iterations
        # Best score should be reasonable (mock gives 0.3-1.0)
        assert 0.3 <= result.best_score <= 1.0
        # Should have best params
        assert result.best_params is not None

    def test_history_recorded(self, mock_simulator, verifier, inner_config):
        """Test that each iteration is recorded in history."""

        def evaluator(params: dict[str, float]) -> float:
            return 0.5

        optimizer = BayesianOptimizer(evaluator=evaluator)

        inner = InnerLoop(
            simulator=mock_simulator,
            verifier=verifier,
            optimizer=optimizer,
            real_data=None,
            config=inner_config,
        )

        result = inner.run()

        # History should have one entry per iteration
        assert len(result.history) == result.iterations

        # Each history entry should have required fields
        for record in result.history:
            assert "iteration" in record
            assert "params" in record
            assert "verification_score" in record
            assert "combined_score" in record


class TestInnerLoopEarlyStopping:
    """Test early stopping conditions."""

    def test_target_reached(self, verifier):
        """Test that loop stops when target is reached."""

        # Create a simulator that returns a high-scoring trajectory
        # Using the same pattern as the freefall_trajectory fixture which scores > 0.85
        class HighScoreSimulator:
            def __init__(self):
                self.first_call = True

            def configure(self, params: PhysicsParams) -> None:
                pass

            def run_scenario(self, scenario: str, duration: float) -> TrajectoryData:
                # Generate a perfect free-fall trajectory (like the test fixture)
                frames = []
                g = 9.81
                dt = 0.033  # ~30 fps
                initial_height = 5.0
                initial_velocity = np.array([0.0, 0.0, 0.0])

                # Limit frames to stay above ground
                for i in range(30):
                    t = i * dt
                    position = np.array(
                        [
                            0.0,
                            initial_height - 0.5 * g * t**2,
                            0.0,
                        ]
                    )
                    velocity = np.array([0.0, -g * t, 0.0])
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

                return TrajectoryData(frames=frames, params=PhysicsParams(mass=0.5, gravity=g))

        config = InnerLoopConfig(
            scenario="test",
            target_score=0.80,  # Free-fall trajectory scores > 0.85
            max_iterations=50,
            patience=10,
            duration=1.0,
        )

        def evaluator(params: dict[str, float]) -> float:
            return 0.9

        optimizer = BayesianOptimizer(evaluator=evaluator)

        inner = InnerLoop(
            simulator=HighScoreSimulator(),
            verifier=verifier,
            optimizer=optimizer,
            real_data=None,
            config=config,
        )

        result = inner.run()

        # Should have stopped early due to reaching target
        assert result.stopped_early is True
        assert result.stop_reason == "target_reached"
        # Should have used fewer than max iterations
        assert result.iterations < config.max_iterations
        # Score should be at or above target
        assert result.best_score >= config.target_score

    def test_patience_exhausted(self, verifier):
        """Test that loop stops when patience is exhausted."""

        # Create a simulator that returns consistently poor scores (no improvement)
        # Using very few frames with inconsistent physics to get low scores
        class StuckSimulator:
            def configure(self, params: PhysicsParams) -> None:
                pass

            def run_scenario(self, scenario: str, duration: float) -> TrajectoryData:
                # Return a trajectory with inconsistent physics (low score)
                frames = []
                dt = 1.0 / 30.0
                num_frames = int(duration / dt)
                for i in range(num_frames):
                    t = i * dt
                    # Inconsistent: position doesn't match velocity/acceleration
                    y = 1.0  # Constant position
                    vy = -5.0  # But moving down
                    ay = 0.0  # But no acceleration
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
                return TrajectoryData(frames=frames, params=PhysicsParams())

        config = InnerLoopConfig(
            scenario="test",
            target_score=0.95,  # Very high, unlikely to reach
            max_iterations=50,
            patience=3,  # Very low patience
            duration=1.0,
        )

        def evaluator(params: dict[str, float]) -> float:
            return 0.5

        optimizer = BayesianOptimizer(evaluator=evaluator)

        inner = InnerLoop(
            simulator=StuckSimulator(),
            verifier=verifier,
            optimizer=optimizer,
            real_data=None,
            config=config,
        )

        result = inner.run()

        # Should have stopped early due to patience
        assert result.stopped_early is True
        assert result.stop_reason == "patience_exhausted"
        # Should have stopped at patience + 1 iterations (first + patience without improvement)
        assert result.iterations <= config.patience + 1


class TestInnerLoopWithRealData:
    """Test inner loop with real data comparison."""

    def test_similarity_computed_when_real_data_present(self, mock_simulator, verifier):
        """Test that similarity is computed when real data is provided."""
        # Create a simple real trajectory
        real_frames = []
        g = 9.81
        dt = 1.0 / 30.0
        for i in range(30):
            t = i * dt
            y = max(0.1, 2.0 - 0.5 * g * t**2)
            real_frames.append(
                FrameData(
                    timestamp=t,
                    position=np.array([0.0, y, 0.0]),
                    velocity=np.array([0.0, -g * t, 0.0]),
                    acceleration=np.array([0.0, -g, 0.0]),
                    rotation=np.zeros(3),
                    angular_velocity=np.zeros(3),
                    frame_index=i,
                )
            )
        real_data = [TrajectoryData(frames=real_frames, params=PhysicsParams())]

        config = InnerLoopConfig(
            scenario="test",
            target_score=0.95,
            max_iterations=5,
            patience=3,
            verification_weight=0.7,
            similarity_weight=0.3,
            duration=1.0,
        )

        def evaluator(params: dict[str, float]) -> float:
            return 0.5

        optimizer = BayesianOptimizer(evaluator=evaluator)

        inner = InnerLoop(
            simulator=mock_simulator,
            verifier=verifier,
            optimizer=optimizer,
            real_data=real_data,
            config=config,
        )

        result = inner.run()

        # Should have recorded similarity in history
        for record in result.history:
            assert "similarity" in record

        # Best similarity should be recorded
        assert result.best_similarity is not None

"""Tests for outer loop meta-learning with strategy transfer."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from physicalfish.loop.outer_loop import (
    OuterLoop,
    OuterLoopConfig,
    OuterLoopResult,
    StrategyRecord,
    build_prior,
)
from physicalfish.loop.inner_loop import InnerLoopResult
from physicalfish.models import PhysicsParams, TrajectoryData


class TestBuildPrior:
    """Tests for build_prior function."""

    def test_empty_strategy_bank_returns_none(self):
        """Empty strategy bank should return None."""
        result = build_prior([], "grasp_ball")
        assert result is None

    def test_single_strategy_builds_prior(self):
        """Single strategy should build prior with default std."""
        record = StrategyRecord(
            scenario="grasp_ball",
            best_params=PhysicsParams(gravity=9.8, mass=0.6, friction=0.4),
            convergence_score=0.85,
            iterations_to_converge=10,
        )
        prior = build_prior([record], "throw_catch")

        assert prior is not None
        assert "gravity" in prior
        assert "mass" in prior
        assert "friction" in prior

        # Single record should have minimum std of 0.1 (from code: 0.1 default, then max with 0.05)
        mean, std = prior["gravity"]
        assert mean == 9.8
        assert std >= 0.05  # Minimum std enforced (actual is 0.1)

    def test_multiple_strategies_compute_statistics(self):
        """Multiple strategies should compute mean and std."""
        records = [
            StrategyRecord(
                scenario="grasp_ball",
                best_params=PhysicsParams(gravity=9.8, mass=0.5),
                convergence_score=0.85,
                iterations_to_converge=10,
            ),
            StrategyRecord(
                scenario="throw_catch",
                best_params=PhysicsParams(gravity=9.9, mass=0.6),
                convergence_score=0.88,
                iterations_to_converge=8,
            ),
            StrategyRecord(
                scenario="drop_catch",
                best_params=PhysicsParams(gravity=10.0, mass=0.7),
                convergence_score=0.90,
                iterations_to_converge=6,
            ),
        ]
        prior = build_prior(records, "new_scenario")

        assert prior is not None
        # Mean of 9.8, 9.9, 10.0 = 9.9
        gravity_mean, gravity_std = prior["gravity"]
        assert gravity_mean == pytest.approx(9.9, abs=0.01)
        assert gravity_std > 0.05  # Should have actual std

        # Mean of 0.5, 0.6, 0.7 = 0.6
        mass_mean, mass_std = prior["mass"]
        assert mass_mean == pytest.approx(0.6, abs=0.01)

    def test_prior_includes_all_physics_params(self):
        """Prior should include all physics parameters."""
        record = StrategyRecord(
            scenario="test",
            best_params=PhysicsParams(),
            convergence_score=0.8,
            iterations_to_converge=5,
        )
        prior = build_prior([record], "new_scenario")

        expected_params = [
            "gravity",
            "mass",
            "friction",
            "restitution",
            "linear_damping",
            "angular_damping",
        ]
        for param in expected_params:
            assert param in prior
            mean, std = prior[param]
            assert isinstance(mean, float)
            assert isinstance(std, float)
            assert std >= 0.05  # Minimum std enforced


class TestOuterLoopInitialization:
    """Tests for OuterLoop initialization."""

    def test_init_with_scenarios_and_config(self):
        """OuterLoop should initialize with scenarios and config."""
        scenarios = ["grasp_ball", "throw_catch"]
        config = OuterLoopConfig(target_score=0.85, max_iterations=30)
        real_data = {"grasp_ball": [], "throw_catch": []}
        simulator_factory = MagicMock()

        loop = OuterLoop(
            scenarios=scenarios,
            real_data_store=real_data,
            config=config,
            simulator_factory=simulator_factory,
        )

        assert loop.scenarios == scenarios
        assert loop.config == config
        assert loop.real_data_store == real_data
        assert loop.simulator_factory == simulator_factory


class TestOuterLoopRun:
    """Tests for OuterLoop.run() method."""

    @pytest.fixture
    def mock_simulator(self):
        """Create a mock simulator."""
        sim = MagicMock()
        sim.configure = MagicMock()
        sim.run_scenario = MagicMock(
            return_value=TrajectoryData(
                frames=[],
                params=PhysicsParams(),
            )
        )
        return sim

    @pytest.fixture
    def mock_inner_loop_result(self):
        """Create a mock inner loop result."""
        return InnerLoopResult(
            best_params=PhysicsParams(gravity=9.8, mass=0.5),
            best_score=0.88,
            best_verification_score=0.85,
            best_similarity=0.75,
            history=[{"combined_score": 0.8}],
            iterations=5,
            stopped_early=True,
            stop_reason="target_reached",
        )

    def test_run_single_scenario(self, mock_simulator, mock_inner_loop_result):
        """Test running with a single scenario."""
        with patch("physicalfish.loop.outer_loop.InnerLoop") as MockInnerLoop:
            MockInnerLoop.return_value.run.return_value = mock_inner_loop_result

            scenarios = ["grasp_ball"]
            config = OuterLoopConfig(target_score=0.90, max_iterations=10)
            real_data = {"grasp_ball": []}

            loop = OuterLoop(
                scenarios=scenarios,
                real_data_store=real_data,
                config=config,
                simulator_factory=lambda: mock_simulator,
            )

            result = loop.run()

            assert isinstance(result, OuterLoopResult)
            assert len(result.strategies) == 1
            assert result.strategies[0].scenario == "grasp_ball"
            assert result.total_iterations == 5
            assert result.avg_convergence_iterations == 5.0

    def test_run_multiple_scenarios(self, mock_simulator, mock_inner_loop_result):
        """Test running with multiple scenarios."""
        with patch("physicalfish.loop.outer_loop.InnerLoop") as MockInnerLoop:
            # Return different results for each scenario
            results = [
                InnerLoopResult(
                    best_params=PhysicsParams(gravity=9.8),
                    best_score=0.88,
                    best_verification_score=0.85,
                    best_similarity=None,
                    iterations=10,
                    stopped_early=True,
                    stop_reason="target_reached",
                ),
                InnerLoopResult(
                    best_params=PhysicsParams(gravity=9.9),
                    best_score=0.90,
                    best_verification_score=0.88,
                    best_similarity=None,
                    iterations=8,
                    stopped_early=True,
                    stop_reason="target_reached",
                ),
            ]
            MockInnerLoop.return_value.run.side_effect = results

            scenarios = ["grasp_ball", "throw_catch"]
            config = OuterLoopConfig(target_score=0.90, max_iterations=20)
            real_data = {"grasp_ball": [], "throw_catch": []}

            loop = OuterLoop(
                scenarios=scenarios,
                real_data_store=real_data,
                config=config,
                simulator_factory=lambda: mock_simulator,
            )

            result = loop.run()

            assert len(result.strategies) == 2
            assert result.strategies[0].scenario == "grasp_ball"
            assert result.strategies[1].scenario == "throw_catch"
            assert result.total_iterations == 18  # 10 + 8
            assert result.avg_convergence_iterations == 9.0  # 18 / 2

    def test_strategy_transfer_prior_effectiveness(self, mock_simulator):
        """Test that prior effectiveness is measured for strategy transfer."""
        with patch("physicalfish.loop.outer_loop.InnerLoop") as MockInnerLoop:
            # Second scenario should converge faster due to prior
            results = [
                InnerLoopResult(
                    best_params=PhysicsParams(gravity=9.8),
                    best_score=0.85,
                    best_verification_score=0.82,
                    best_similarity=None,
                    iterations=10,
                    stopped_early=True,
                    stop_reason="target_reached",
                ),
                InnerLoopResult(
                    best_params=PhysicsParams(gravity=9.9),
                    best_score=0.88,
                    best_verification_score=0.85,
                    best_similarity=None,
                    iterations=6,  # Faster convergence!
                    stopped_early=True,
                    stop_reason="target_reached",
                ),
            ]
            MockInnerLoop.return_value.run.side_effect = results

            scenarios = ["grasp_ball", "throw_catch"]
            config = OuterLoopConfig(target_score=0.90, max_iterations=20)
            real_data = {"grasp_ball": [], "throw_catch": []}

            loop = OuterLoop(
                scenarios=scenarios,
                real_data_store=real_data,
                config=config,
                simulator_factory=lambda: mock_simulator,
            )

            result = loop.run()

            # Prior effectiveness should be recorded for second scenario
            assert "throw_catch" in result.prior_effectiveness
            # Improvement = (10 - 6) / 10 = 0.4
            assert result.prior_effectiveness["throw_catch"] == pytest.approx(0.4, abs=0.01)

    def test_run_with_real_data(self, mock_simulator, mock_inner_loop_result):
        """Test that real data is passed to inner loop."""
        with patch("physicalfish.loop.outer_loop.InnerLoop") as MockInnerLoop:
            MockInnerLoop.return_value.run.return_value = mock_inner_loop_result

            # Create some real trajectory data
            real_trajectory = TrajectoryData(
                frames=[],
                params=PhysicsParams(),
            )

            scenarios = ["grasp_ball"]
            config = OuterLoopConfig(target_score=0.90)
            real_data = {"grasp_ball": [real_trajectory]}

            loop = OuterLoop(
                scenarios=scenarios,
                real_data_store=real_data,
                config=config,
                simulator_factory=lambda: mock_simulator,
            )

            loop.run()

            # Verify InnerLoop was called with real_data
            MockInnerLoop.assert_called_once()
            call_kwargs = MockInnerLoop.call_args.kwargs
            assert call_kwargs["real_data"] == [real_trajectory]

    def test_empty_scenarios_returns_empty_result(self, mock_simulator):
        """Empty scenarios list should return empty result."""
        config = OuterLoopConfig(target_score=0.90)
        loop = OuterLoop(
            scenarios=[],
            real_data_store={},
            config=config,
            simulator_factory=lambda: mock_simulator,
        )

        result = loop.run()

        assert len(result.strategies) == 0
        assert result.total_iterations == 0
        assert result.avg_convergence_iterations == 0.0

    def test_strategy_records_store_best_params(self, mock_simulator, mock_inner_loop_result):
        """Strategy records should store the best params from inner loop."""
        with patch("physicalfish.loop.outer_loop.InnerLoop") as MockInnerLoop:
            MockInnerLoop.return_value.run.return_value = mock_inner_loop_result

            scenarios = ["grasp_ball"]
            config = OuterLoopConfig(target_score=0.90)
            real_data = {"grasp_ball": []}

            loop = OuterLoop(
                scenarios=scenarios,
                real_data_store=real_data,
                config=config,
                simulator_factory=lambda: mock_simulator,
            )

            result = loop.run()

            record = result.strategies[0]
            assert record.best_params.gravity == 9.8
            assert record.best_params.mass == 0.5
            assert record.convergence_score == 0.88
            assert record.iterations_to_converge == 5


class TestStrategyRecord:
    """Tests for StrategyRecord dataclass."""

    def test_strategy_record_creation(self):
        """StrategyRecord should store all fields correctly."""
        params = PhysicsParams(gravity=9.8, mass=0.5, friction=0.4)
        record = StrategyRecord(
            scenario="grasp_ball",
            best_params=params,
            convergence_score=0.85,
            iterations_to_converge=10,
        )

        assert record.scenario == "grasp_ball"
        assert record.best_params == params
        assert record.convergence_score == 0.85
        assert record.iterations_to_converge == 10


class TestOuterLoopConfig:
    """Tests for OuterLoopConfig dataclass."""

    def test_default_config(self):
        """Default config should have sensible values."""
        config = OuterLoopConfig()

        assert config.target_score == 0.90
        assert config.max_iterations == 50
        assert config.patience == 10
        assert config.verification_weight == 0.7
        assert config.duration == 3.0

    def test_custom_config(self):
        """Config should accept custom values."""
        config = OuterLoopConfig(
            target_score=0.85,
            max_iterations=30,
            patience=5,
            verification_weight=0.6,
            duration=5.0,
        )

        assert config.target_score == 0.85
        assert config.max_iterations == 30
        assert config.patience == 5
        assert config.verification_weight == 0.6
        assert config.duration == 5.0

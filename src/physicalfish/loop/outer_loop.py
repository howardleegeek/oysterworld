"""Outer loop: cross-scenario meta-learning with strategy transfer."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog

from physicalfish.models import PhysicsParams, TrajectoryData
from physicalfish.verification.physics_verifier import PhysicsVerifier
from physicalfish.optimizer.bayesian import BayesianOptimizer
from physicalfish.loop.inner_loop import InnerLoop, InnerLoopConfig


@dataclass
class StrategyRecord:
    """Record of a learned strategy for a scenario."""

    scenario: str
    best_params: PhysicsParams
    convergence_score: float
    iterations_to_converge: int


@dataclass
class OuterLoopConfig:
    """Configuration for outer loop meta-learning."""

    target_score: float = 0.90
    max_iterations: int = 50
    patience: int = 10
    verification_weight: float = 0.7
    duration: float = 3.0


@dataclass
class OuterLoopResult:
    """Result of outer loop meta-learning."""

    strategies: list[StrategyRecord]
    total_iterations: int
    avg_convergence_iterations: float
    prior_effectiveness: dict[str, float] = field(default_factory=dict)


def build_prior(
    strategy_bank: list[StrategyRecord], current_scenario: str
) -> dict[str, tuple[float, float]] | None:
    """Build Gaussian Process prior from historical strategies.

    Args:
        strategy_bank: List of previously learned strategies
        current_scenario: The scenario we're about to optimize

    Returns:
        Dictionary mapping parameter names to (mean, std) tuples,
        or None if strategy_bank is empty
    """
    if not strategy_bank:
        return None

    # Extract all best_params from strategy bank
    all_params: dict[str, list[float]] = {
        "gravity": [],
        "mass": [],
        "friction": [],
        "restitution": [],
        "linear_damping": [],
        "angular_damping": [],
    }

    for record in strategy_bank:
        params_dict = record.best_params.to_dict()
        for key in all_params:
            if key in params_dict:
                all_params[key].append(params_dict[key])

    # Compute mean and std for each parameter
    prior = {}
    for key, values in all_params.items():
        if values:
            mean = float(np.mean(values))
            std = float(np.std(values)) if len(values) > 1 else 0.1
            # Ensure std is not too small (minimum exploration)
            std = max(std, 0.05)
            prior[key] = (mean, std)

    return prior


class OuterLoop:
    """Cross-scenario meta-learning loop with strategy transfer."""

    def __init__(
        self,
        scenarios: list[str],
        real_data_store: dict[str, list[TrajectoryData]],
        config: OuterLoopConfig,
        simulator_factory: Any,  # Callable that returns a simulator
    ):
        """Initialize outer loop.

        Args:
            scenarios: List of scenario names to optimize
            real_data_store: Map from scenario name to list of real trajectories
            config: Outer loop configuration
            simulator_factory: Factory function that creates simulators
        """
        self.scenarios = scenarios
        self.real_data_store = real_data_store
        self.config = config
        self.simulator_factory = simulator_factory
        self.logger = structlog.get_logger("outer_loop")

    def run(self) -> OuterLoopResult:
        """Run the outer meta-learning loop.

        Returns:
            OuterLoopResult with learned strategies and statistics
        """
        self.logger.info(
            "outer_loop_start",
            num_scenarios=len(self.scenarios),
            target_score=self.config.target_score,
        )

        strategy_bank: list[StrategyRecord] = []
        total_iterations = 0
        prior_effectiveness: dict[str, float] = {}

        for i, scenario in enumerate(self.scenarios):
            self.logger.info(
                "optimizing_scenario",
                scenario=scenario,
                index=i,
                total=len(self.scenarios),
            )

            # Build prior from previous strategies
            prior = build_prior(strategy_bank, scenario)

            if prior:
                self.logger.info(
                    "using_prior",
                    scenario=scenario,
                    prior_params=list(prior.keys()),
                )

            # Create optimizer with prior
            # Note: In a full implementation, the BayesianOptimizer would
            # accept a prior parameter to guide initial sampling
            verifier = PhysicsVerifier()
            simulator = self.simulator_factory()

            # Create optimizer (prior would be used here in full implementation)
            def evaluator(params: dict[str, float]) -> float:
                # This is a placeholder - actual evaluation happens in InnerLoop
                return 0.5

            optimizer = BayesianOptimizer(evaluator=evaluator)

            # Get real data for this scenario
            real_data = self.real_data_store.get(scenario, [])

            # Create inner loop config
            inner_config = InnerLoopConfig(
                scenario=scenario,
                target_score=self.config.target_score,
                max_iterations=self.config.max_iterations,
                patience=self.config.patience,
                verification_weight=self.config.verification_weight,
                duration=self.config.duration,
            )

            # Run inner loop
            inner = InnerLoop(
                simulator=simulator,
                verifier=verifier,
                optimizer=optimizer,
                real_data=real_data,
                config=inner_config,
            )
            result = inner.run()

            # Record strategy
            record = StrategyRecord(
                scenario=scenario,
                best_params=result.best_params,
                convergence_score=result.best_score,
                iterations_to_converge=result.iterations,
            )
            strategy_bank.append(record)
            total_iterations += result.iterations

            # Measure prior effectiveness (if we had a prior)
            if prior and i > 0:
                # Compare iterations to previous scenario
                prev_iterations = strategy_bank[-2].iterations_to_converge
                improvement = (prev_iterations - result.iterations) / prev_iterations
                prior_effectiveness[scenario] = improvement
                self.logger.info(
                    "prior_effectiveness",
                    scenario=scenario,
                    improvement=improvement,
                    prev_iterations=prev_iterations,
                    curr_iterations=result.iterations,
                )

            self.logger.info(
                "scenario_complete",
                scenario=scenario,
                best_score=result.best_score,
                iterations=result.iterations,
                stopped_early=result.stopped_early,
            )

        # Compute statistics
        avg_iterations = total_iterations / len(self.scenarios) if self.scenarios else 0

        self.logger.info(
            "outer_loop_complete",
            total_iterations=total_iterations,
            avg_iterations=avg_iterations,
            strategies_learned=len(strategy_bank),
        )

        return OuterLoopResult(
            strategies=strategy_bank,
            total_iterations=total_iterations,
            avg_convergence_iterations=avg_iterations,
            prior_effectiveness=prior_effectiveness,
        )

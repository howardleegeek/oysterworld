"""Inner loop: single-task optimization with verification and real data comparison."""

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import structlog

from physicalfish.models import (
    PhysicsParams,
    TrajectoryData,
    VerificationResult,
)
from physicalfish.verification.physics_verifier import PhysicsVerifier
from physicalfish.verification.distribution_compare import DistributionComparer
from physicalfish.optimizer.bayesian import BayesianOptimizer


class SimulatorProtocol(Protocol):
    """Protocol for physics simulators."""

    def configure(self, params: PhysicsParams) -> None:
        """Configure simulator with physics parameters."""
        ...

    def run_scenario(self, scenario: str, duration: float) -> TrajectoryData:
        """Run a scenario and return trajectory."""
        ...


@dataclass
class InnerLoopConfig:
    """Configuration for inner loop optimization."""

    scenario: str
    target_score: float = 0.90
    max_iterations: int = 50
    patience: int = 10
    verification_weight: float = 0.7
    similarity_weight: float = 0.3
    duration: float = 3.0


@dataclass
class InnerLoopResult:
    """Result of inner loop optimization."""

    best_params: PhysicsParams
    best_score: float
    best_verification_score: float
    best_similarity: float | None
    history: list[dict[str, Any]] = field(default_factory=list)
    iterations: int = 0
    stopped_early: bool = False
    stop_reason: str = ""


class InnerLoop:
    """Single-task optimization loop combining verification and real data comparison."""

    def __init__(
        self,
        simulator: SimulatorProtocol,
        verifier: PhysicsVerifier,
        optimizer: BayesianOptimizer,
        real_data: list[TrajectoryData] | None,
        config: InnerLoopConfig,
    ):
        """Initialize inner loop.

        Args:
            simulator: Physics simulator (must implement SimulatorProtocol)
            verifier: Physics verification engine
            optimizer: Bayesian optimizer for parameter search
            real_data: Optional list of real trajectories for comparison
            config: Inner loop configuration
        """
        self.simulator = simulator
        self.verifier = verifier
        self.optimizer = optimizer
        self.real_data = real_data or []
        self.config = config
        self.comparer = DistributionComparer()
        self.logger = structlog.get_logger("inner_loop")

    def run(self) -> InnerLoopResult:
        """Run the inner optimization loop.

        Returns:
            InnerLoopResult with best params and optimization history
        """
        self.logger.info(
            "inner_loop_start",
            scenario=self.config.scenario,
            target_score=self.config.target_score,
            max_iterations=self.config.max_iterations,
        )

        best_score = -np.inf
        best_params = None
        best_verification_score = 0.0
        best_similarity = None
        history = []
        no_improvement_count = 0

        for iteration in range(self.config.max_iterations):
            # Get next parameters from optimizer
            params_dict = self._get_next_params(iteration)
            params = PhysicsParams(**params_dict)

            # Configure simulator and run
            self.simulator.configure(params)
            trajectory = self.simulator.run_scenario(self.config.scenario, self.config.duration)

            # Verify trajectory
            verification_result = self.verifier.verify(trajectory)

            # Compare with real data if available
            similarity = None
            if self.real_data:
                comparison = self.comparer.compare(trajectory, self.real_data[0])
                similarity = comparison["overall_similarity"]

            # Compute combined score
            combined_score = self._compute_combined_score(
                verification_result.overall_score, similarity
            )

            # Record iteration
            record = {
                "iteration": iteration,
                "params": params.to_dict(),
                "verification_score": verification_result.overall_score,
                "similarity": similarity,
                "combined_score": combined_score,
                "passed": verification_result.passed,
            }
            history.append(record)

            self.logger.info(
                "iteration_complete",
                iteration=iteration,
                verification_score=verification_result.overall_score,
                similarity=similarity,
                combined_score=combined_score,
            )

            # Update best
            if combined_score > best_score:
                best_score = combined_score
                best_params = params
                best_verification_score = verification_result.overall_score
                best_similarity = similarity
                no_improvement_count = 0
                self.logger.info(
                    "new_best",
                    iteration=iteration,
                    combined_score=combined_score,
                )
            else:
                no_improvement_count += 1

            # Check stopping conditions
            if best_score >= self.config.target_score:
                self.logger.info(
                    "target_reached",
                    iteration=iteration,
                    best_score=best_score,
                )
                return InnerLoopResult(
                    best_params=best_params,
                    best_score=best_score,
                    best_verification_score=best_verification_score,
                    best_similarity=best_similarity,
                    history=history,
                    iterations=iteration + 1,
                    stopped_early=True,
                    stop_reason="target_reached",
                )

            if no_improvement_count >= self.config.patience:
                self.logger.info(
                    "early_stopping",
                    iteration=iteration,
                    no_improvement_count=no_improvement_count,
                )
                return InnerLoopResult(
                    best_params=best_params,
                    best_score=best_score,
                    best_verification_score=best_verification_score,
                    best_similarity=best_similarity,
                    history=history,
                    iterations=iteration + 1,
                    stopped_early=True,
                    stop_reason="patience_exhausted",
                )

        # Max iterations reached
        self.logger.info(
            "max_iterations_reached",
            iterations=self.config.max_iterations,
            best_score=best_score,
        )

        return InnerLoopResult(
            best_params=best_params or PhysicsParams(),
            best_score=best_score,
            best_verification_score=best_verification_score,
            best_similarity=best_similarity,
            history=history,
            iterations=self.config.max_iterations,
            stopped_early=False,
            stop_reason="max_iterations",
        )

    def _get_next_params(self, iteration: int) -> dict[str, float]:
        """Get next parameter set from optimizer."""
        # For BayesianOptimizer, we need to use the optimize method
        # But since we want to do this iteratively, we'll use a different approach
        # We'll create a simple evaluator that returns scores and track manually

        # For now, use random sampling within bounds
        # In a full implementation, this would integrate with the optimizer's
        # acquisition function
        from physicalfish.optimizer.parameter_space import PHYSICS_PARAM_SPACE

        return {name: np.random.uniform(low, high) for name, low, high in PHYSICS_PARAM_SPACE}

    def _compute_combined_score(self, verification_score: float, similarity: float | None) -> float:
        """Compute weighted combination of verification and similarity scores."""
        if similarity is None:
            return verification_score

        return (
            self.config.verification_weight * verification_score
            + self.config.similarity_weight * similarity
        )

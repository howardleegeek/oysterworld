"""Bootstrap generator: few-shot data augmentation through simulation."""

from typing import Protocol

import numpy as np
import structlog

from physicalfish.models import PhysicsParams, TrajectoryData
from physicalfish.verification.physics_verifier import PhysicsVerifier
from physicalfish.optimizer.parameter_space import PHYSICS_PARAM_SPACE


class SimulatorProtocol(Protocol):
    """Protocol for physics simulators."""

    def configure(self, params: PhysicsParams) -> None:
        """Configure simulator with physics parameters."""
        ...

    def run_scenario(self, scenario: str, duration: float) -> TrajectoryData:
        """Run a scenario and return trajectory."""
        ...


class BootstrapGenerator:
    """Generate augmented training data from few real samples."""

    def __init__(
        self,
        simulator: SimulatorProtocol,
        verifier: PhysicsVerifier,
        quality_threshold: float = 0.7,
        scenario: str = "grasp_ball",
        duration: float = 3.0,
    ):
        """Initialize bootstrap generator.

        Args:
            simulator: Physics simulator
            verifier: Physics verification engine
            quality_threshold: Minimum verification score to accept sample
            scenario: Scenario to simulate
            duration: Simulation duration in seconds
        """
        self.simulator = simulator
        self.verifier = verifier
        self.quality_threshold = quality_threshold
        self.scenario = scenario
        self.duration = duration
        self.logger = structlog.get_logger("bootstrap")

    def augment(
        self, real_samples: list[TrajectoryData], target_count: int
    ) -> list[TrajectoryData]:
        """Augment few real samples into larger training set.

        Args:
            real_samples: List of real trajectory samples (typically 3-5)
            target_count: Target number of augmented samples to generate

        Returns:
            List of verified synthetic trajectories
        """
        self.logger.info(
            "bootstrap_start",
            num_real_samples=len(real_samples),
            target_count=target_count,
            quality_threshold=self.quality_threshold,
        )

        if not real_samples:
            self.logger.warning("no_real_samples_provided")
            return []

        # Extract parameter distribution from real samples
        param_mean, param_std = self._extract_param_distribution(real_samples)

        self.logger.info(
            "param_distribution_extracted",
            mean=param_mean,
            std=param_std,
        )

        # Generate and verify samples
        accepted: list[TrajectoryData] = []
        rejected = 0
        max_attempts = target_count * 3  # Allow for some rejection

        for attempt in range(max_attempts):
            if len(accepted) >= target_count:
                break

            # Sample parameters from distribution
            params = self._sample_params(param_mean, param_std)

            # Run simulation
            self.simulator.configure(params)
            trajectory = self.simulator.run_scenario(self.scenario, self.duration)

            # Verify quality
            try:
                result = self.verifier.verify(trajectory)

                if result.overall_score >= self.quality_threshold:
                    # Add metadata about generation
                    trajectory.metadata["bootstrap"] = {
                        "attempt": attempt,
                        "verification_score": result.overall_score,
                        "params": params.to_dict(),
                    }
                    accepted.append(trajectory)
                    self.logger.debug(
                        "sample_accepted",
                        attempt=attempt,
                        score=result.overall_score,
                        accepted_count=len(accepted),
                    )
                else:
                    rejected += 1
                    self.logger.debug(
                        "sample_rejected",
                        attempt=attempt,
                        score=result.overall_score,
                        threshold=self.quality_threshold,
                    )
            except Exception as e:
                rejected += 1
                self.logger.debug(
                    "verification_failed",
                    attempt=attempt,
                    error=str(e),
                )

        total_attempts = len(accepted) + rejected
        pass_rate = len(accepted) / total_attempts if total_attempts > 0 else 0

        self.logger.info(
            "bootstrap_complete",
            target_count=target_count,
            generated=len(accepted),
            attempts=total_attempts,
            pass_rate=pass_rate,
        )

        return accepted

    def _extract_param_distribution(
        self, samples: list[TrajectoryData]
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Extract mean and std of parameters from real samples.

        Args:
            samples: List of trajectory samples

        Returns:
            Tuple of (mean_dict, std_dict)
        """
        # Collect all parameter values
        param_values: dict[str, list[float]] = {
            "gravity": [],
            "mass": [],
            "friction": [],
            "restitution": [],
            "linear_damping": [],
            "angular_damping": [],
        }

        for sample in samples:
            params_dict = sample.params.to_dict()
            for key in param_values:
                if key in params_dict:
                    param_values[key].append(params_dict[key])

        # Compute statistics
        mean = {}
        std = {}
        for key, values in param_values.items():
            if values:
                mean[key] = float(np.mean(values))
                std[key] = float(np.std(values)) if len(values) > 1 else 0.1
                # Ensure minimum std for exploration
                std[key] = max(std[key], 0.05)
            else:
                # Use default bounds if no data
                for name, low, high in PHYSICS_PARAM_SPACE:
                    if name == key:
                        mean[key] = (low + high) / 2
                        std[key] = (high - low) / 4
                        break

        return mean, std

    def _sample_params(self, mean: dict[str, float], std: dict[str, float]) -> PhysicsParams:
        """Sample parameters from Gaussian distribution.

        Args:
            mean: Dictionary of parameter means
            std: Dictionary of parameter standard deviations

        Returns:
            Sampled PhysicsParams
        """
        sampled = {}
        for key in mean:
            # Sample from Gaussian
            value = np.random.normal(mean[key], std[key])

            # Clip to valid bounds
            for name, low, high in PHYSICS_PARAM_SPACE:
                if name == key:
                    value = np.clip(value, low, high)
                    break

            sampled[key] = value

        return PhysicsParams(**sampled)

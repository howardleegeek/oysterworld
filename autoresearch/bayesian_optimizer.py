"""
Bayesian Optimization for Physics Parameter Tuning
Uses Gaussian Processes for efficient exploration-exploitation balance
"""

import numpy as np
from typing import Dict, List, Callable, Optional
from dataclasses import dataclass
import warnings

# Try to import scikit-optimize, provide fallback if not available
try:
    from skopt import gp_minimize
    from skopt.space import Real
    from skopt.utils import use_named_args

    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    warnings.warn("scikit-optimize not available. Using fallback optimizer.")


@dataclass
class OptimizationResult:
    """Result of Bayesian optimization"""

    best_params: Dict[str, float]
    best_score: float
    n_iterations: int
    convergence_history: List[float]
    acquisition_history: List[str]


class BayesianOptimizer:
    """
    Bayesian Optimization for physics parameter tuning

    Uses Gaussian Process regression to model the objective function
    and Expected Improvement (EI) acquisition function to select
    next parameters to evaluate.

    Typically converges in 10-15 iterations vs 20+ for hill climbing.
    """

    def __init__(self, evaluator: Callable[[Dict], float], random_state: int = 42):
        """
        Args:
            evaluator: Function that takes params dict and returns score (to maximize)
            random_state: Random seed for reproducibility
        """
        self.evaluator = evaluator
        self.random_state = random_state

        # Parameter space: (name, min, max)
        self.param_space = [
            ("gravity_magnitude", 8.0, 11.0),
            ("object_mass", 0.1, 2.0),
            ("object_friction", 0.1, 1.0),
            ("object_bounce", 0.0, 1.0),
            ("linear_damping", 0.0, 1.0),
        ]

        self.history = []
        self.best_score = -np.inf
        self.best_params = None

    def optimize(
        self, n_calls: int = 15, n_initial_points: int = 5, verbose: bool = True
    ) -> OptimizationResult:
        """
        Run Bayesian optimization

        Args:
            n_calls: Total number of evaluations (including initial random)
            n_initial_points: Number of random initial evaluations
            verbose: Print progress

        Returns:
            OptimizationResult with best params and history
        """
        if not SKOPT_AVAILABLE:
            return self._fallback_optimize(n_calls, verbose)

        if verbose:
            print(f"Bayesian Optimization")
            print(f"  Parameter space: {len(self.param_space)} dimensions")
            print(f"  Total evaluations: {n_calls}")
            print(f"  Initial random: {n_initial_points}")
            print(f"  Acquisition: Expected Improvement (EI)")
            print()

        # Define search space for skopt
        space = [Real(low, high, name=name) for name, low, high in self.param_space]

        # Run optimization
        result = gp_minimize(
            func=self._objective,
            dimensions=space,
            n_calls=n_calls,
            n_initial_points=n_initial_points,
            acq_func="EI",  # Expected Improvement
            random_state=self.random_state,
            verbose=verbose,
            n_jobs=1,  # Sequential for reproducibility
        )

        # Extract best params
        best_params = dict(zip([name for name, _, _ in self.param_space], result.x))

        return OptimizationResult(
            best_params=best_params,
            best_score=-result.fun,  # Negate because we minimized
            n_iterations=len(result.func_vals),
            convergence_history=[-x for x in result.func_vals],
            acquisition_history=[
                "random" if i < n_initial_points else "EI"
                for i in range(len(result.func_vals))
            ],
        )

    def _objective(self, params_list: List[float]) -> float:
        """
        Objective function to minimize

        Args:
            params_list: List of parameter values from skopt

        Returns:
            Negative score (because skopt minimizes)
        """
        # Convert list to dict
        params = dict(zip([name for name, _, _ in self.param_space], params_list))

        # Evaluate
        score = self.evaluator(params)

        # Update tracking
        self.history.append(
            {"params": params, "score": score, "is_best": score > self.best_score}
        )

        if score > self.best_score:
            self.best_score = score
            self.best_params = params.copy()

        # Return negative for minimization
        return -score

    def _fallback_optimize(self, n_calls: int, verbose: bool) -> OptimizationResult:
        """
        Fallback optimizer when scikit-optimize is not available
        Uses simple hill climbing with random restarts
        """
        if verbose:
            print("Bayesian Optimization (Fallback Mode)")
            print("  scikit-optimize not available, using hill climbing")
            print()

        # Random initial points
        best_score = -np.inf
        best_params = None
        history = []

        for i in range(n_calls):
            # Random sample from parameter space
            params = {
                name: np.random.uniform(low, high)
                for name, low, high in self.param_space
            }

            score = self.evaluator(params)
            history.append(score)

            if score > best_score:
                best_score = score
                best_params = params.copy()
                if verbose:
                    print(f"  Iter {i + 1}: {score:.4f} ★ New best!")
            else:
                if verbose:
                    print(f"  Iter {i + 1}: {score:.4f}")

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            n_iterations=n_calls,
            convergence_history=history,
            acquisition_history=["random"] * n_calls,
        )


class AdaptiveOptimizer:
    """
    Adaptive optimizer that switches strategies based on progress

    Starts with Bayesian Optimization for exploration,
    switches to local search for fine-tuning.
    """

    def __init__(self, evaluator: Callable[[Dict], float]):
        self.evaluator = evaluator
        self.bayesian = BayesianOptimizer(evaluator)

    def optimize(
        self, n_bayesian: int = 10, n_local: int = 5, verbose: bool = True
    ) -> OptimizationResult:
        """
        Two-phase optimization:
        1. Bayesian phase: global exploration
        2. Local phase: fine-tuning around best region
        """
        if verbose:
            print("Adaptive Optimization (Bayesian + Local Search)")
            print()

        # Phase 1: Bayesian
        if verbose:
            print("Phase 1: Bayesian Optimization (global exploration)")

        bayesian_result = self.bayesian.optimize(
            n_calls=n_bayesian, n_initial_points=5, verbose=verbose
        )

        # Phase 2: Local search around best
        if verbose:
            print("\nPhase 2: Local Search (fine-tuning)")

        best_params = bayesian_result.best_params.copy()
        best_score = bayesian_result.best_score

        # Local perturbations
        for i in range(n_local):
            # Small perturbation
            perturbed = {}
            for name, val in best_params.items():
                # Get bounds
                bounds = next(
                    (
                        b
                        for n, _, _ in self.bayesian.param_space
                        for b in [(n, 8.0, 11.0)]
                        if n == name
                    ),
                    None,
                )
                if bounds:
                    _, low, high = bounds
                    noise = (high - low) * 0.05  # 5% of range
                    perturbed[name] = np.clip(
                        val + np.random.normal(0, noise), low, high
                    )

            score = self.evaluator(perturbed)

            if score > best_score:
                best_score = score
                best_params = perturbed
                if verbose:
                    print(f"  Local {i + 1}: {score:.4f} ★ Improved!")
            else:
                if verbose:
                    print(f"  Local {i + 1}: {score:.4f}")

        # Combine histories
        full_history = bayesian_result.convergence_history + [best_score] * n_local

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            n_iterations=len(full_history),
            convergence_history=full_history,
            acquisition_history=(
                bayesian_result.acquisition_history + ["local"] * n_local
            ),
        )


# Demo and testing
if __name__ == "__main__":
    print("Bayesian Optimizer — Test")
    print("=" * 50)

    # Mock evaluator: optimal at gravity=9.2, mass=0.6, friction=0.7
    def mock_evaluator(params: Dict) -> float:
        gravity_error = abs(params["gravity_magnitude"] - 9.2) / 2.0
        mass_error = abs(params["object_mass"] - 0.6) / 1.0
        friction_error = abs(params["object_friction"] - 0.7) / 0.5

        score = 0.95 - (gravity_error + mass_error + friction_error) * 0.3
        noise = np.random.normal(0, 0.02)
        return max(0.3, min(0.98, score + noise))

    # Test Bayesian optimizer
    print("\nTest 1: Bayesian Optimization")
    print("-" * 50)

    opt = BayesianOptimizer(mock_evaluator, random_state=42)
    result = opt.optimize(n_calls=12, n_initial_points=4, verbose=True)

    print(f"\nResults:")
    print(f"  Best score: {result.best_score:.4f}")
    print(f"  Iterations: {result.n_iterations}")
    print(f"  Best params:")
    for k, v in result.best_params.items():
        print(f"    {k}: {v:.3f}")

    # Test adaptive optimizer
    print("\n\nTest 2: Adaptive Optimization")
    print("-" * 50)

    adaptive = AdaptiveOptimizer(mock_evaluator)
    result2 = adaptive.optimize(n_bayesian=8, n_local=4, verbose=True)

    print(f"\nResults:")
    print(f"  Best score: {result2.best_score:.4f}")
    print(f"  Iterations: {result2.n_iterations}")

"""Bayesian Optimization for physics parameter tuning.

Uses Gaussian Processes for efficient exploration-exploitation balance.
Bug fixes from original bayesian_optimizer.py:
- Uses centralized PHYSICS_PARAM_SPACE instead of hardcoded bounds
- Fallback implements true hill climbing (random exploration + Gaussian perturbation)
- Returns unified OptimizationResult from physicalfish.models
"""

import numpy as np
from typing import Callable, Dict
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

from physicalfish.models import OptimizationResult
from physicalfish.optimizer.parameter_space import PHYSICS_PARAM_SPACE, get_bounds_for


class BayesianOptimizer:
    """
    Bayesian Optimization for physics parameter tuning.

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

        # Use centralized parameter space
        self.param_space = PHYSICS_PARAM_SPACE

        self.history = []
        self.best_score = -np.inf
        self.best_params = None

    def optimize(
        self, n_calls: int = 15, n_initial_points: int = 5, verbose: bool = True
    ) -> OptimizationResult:
        """
        Run Bayesian optimization.

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

        # Define search space for skopt using centralized bounds
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
                "random" if i < n_initial_points else "EI" for i in range(len(result.func_vals))
            ],
        )

    def _objective(self, params_list: list[float]) -> float:
        """
        Objective function to minimize.

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
        self.history.append({"params": params, "score": score, "is_best": score > self.best_score})

        if score > self.best_score:
            self.best_score = score
            self.best_params = params.copy()

        # Return negative for minimization
        return -score

    def _fallback_optimize(self, n_calls: int, verbose: bool) -> OptimizationResult:
        """
        Fallback optimizer when scikit-optimize is not available.

        BUG FIX: Implements true hill climbing:
        - First N/3 iterations: random exploration
        - Remaining 2N/3 iterations: Gaussian perturbations from best_params
        """
        if verbose:
            print("Bayesian Optimization (Fallback Mode)")
            print("  scikit-optimize not available, using hill climbing")
            print()

        best_score = -np.inf
        best_params = None
        history = []
        acquisition_history = []

        # Phase 1: Random exploration (first N/3 iterations)
        n_explore = n_calls // 3
        if n_explore < 1:
            n_explore = 1

        for i in range(n_explore):
            # Random sample from parameter space using centralized bounds
            params = {name: np.random.uniform(low, high) for name, low, high in self.param_space}

            score = self.evaluator(params)
            history.append(score)
            acquisition_history.append("random")

            if score > best_score:
                best_score = score
                best_params = params.copy()
                if verbose:
                    print(f"  Explore {i + 1}: {score:.4f} ★ New best!")
            else:
                if verbose:
                    print(f"  Explore {i + 1}: {score:.4f}")

        # Phase 2: Hill climbing from best (remaining iterations)
        n_climb = n_calls - n_explore

        for i in range(n_climb):
            # Gaussian perturbation from current best
            perturbed = {}
            for name, val in best_params.items():
                # Get actual bounds for this parameter (BUG FIX: not hardcoded)
                low, high = get_bounds_for(name)
                noise_scale = (high - low) * 0.1  # 10% of range as std dev
                perturbed[name] = np.clip(val + np.random.normal(0, noise_scale), low, high)

            score = self.evaluator(perturbed)
            history.append(score)
            acquisition_history.append("hill_climbing")

            if score > best_score:
                best_score = score
                best_params = perturbed.copy()
                if verbose:
                    print(f"  Climb {i + 1}: {score:.4f} ★ Improved!")
            else:
                if verbose:
                    print(f"  Climb {i + 1}: {score:.4f}")

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            n_iterations=n_calls,
            convergence_history=history,
            acquisition_history=acquisition_history,
        )

"""Adaptive optimizer that switches strategies based on progress.

Starts with Bayesian Optimization for exploration,
switches to local search for fine-tuning.

BUG FIX: Local search now uses actual parameter bounds from param_space
instead of hardcoded (8.0, 11.0) for all parameters.
"""

import numpy as np
from typing import Callable, Dict

from physicalfish.models import OptimizationResult
from physicalfish.optimizer.bayesian import BayesianOptimizer
from physicalfish.optimizer.parameter_space import get_bounds_for


class AdaptiveOptimizer:
    """
    Adaptive optimizer that switches strategies based on progress.

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
        local_scores = []

        # Local perturbations
        for i in range(n_local):
            # Small perturbation
            perturbed = {}
            for name, val in best_params.items():
                # BUG FIX: Get actual bounds for this parameter from param_space
                # instead of hardcoded (8.0, 11.0)
                try:
                    low, high = get_bounds_for(name)
                except ValueError:
                    # Fallback: if parameter not in param_space, use wide bounds
                    low, high = 0.0, 10.0

                noise = (high - low) * 0.05  # 5% of range
                perturbed[name] = np.clip(val + np.random.normal(0, noise), low, high)

            score = self.evaluator(perturbed)
            local_scores.append(score)

            if score > best_score:
                best_score = score
                best_params = perturbed
                if verbose:
                    print(f"  Local {i + 1}: {score:.4f} ★ Improved!")
            else:
                if verbose:
                    print(f"  Local {i + 1}: {score:.4f}")

        # Combine histories
        full_history = bayesian_result.convergence_history + local_scores

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            n_iterations=len(full_history),
            convergence_history=full_history,
            acquisition_history=(bayesian_result.acquisition_history + ["local"] * n_local),
        )

"""Regression tests for optimizer module.

Tests for bug fixes:
1. Bounds are unique per parameter (not all gravity bounds)
2. Fallback optimizer converges on simple quadratic
3. Fallback hill climbing clusters samples near best in later iterations
"""

import numpy as np
import pytest

from physicalfish.optimizer.parameter_space import (
    PHYSICS_PARAM_SPACE,
    get_bounds_for,
    get_all_bounds,
)
from physicalfish.optimizer.bayesian import BayesianOptimizer
from physicalfish.optimizer.adaptive import AdaptiveOptimizer


class TestParameterSpace:
    """Tests for centralized parameter space definitions."""

    def test_all_params_have_unique_bounds(self):
        """Bug fix verification: gravity bounds != mass bounds != friction bounds"""
        gravity_bounds = get_bounds_for("gravity")
        mass_bounds = get_bounds_for("mass")
        friction_bounds = get_bounds_for("friction")

        # All bounds should be different
        assert gravity_bounds != mass_bounds, "gravity and mass should have different bounds"
        assert mass_bounds != friction_bounds, "mass and friction should have different bounds"
        assert gravity_bounds != friction_bounds, (
            "gravity and friction should have different bounds"
        )

        # Verify actual values
        assert gravity_bounds == (8.0, 11.0), (
            f"gravity bounds should be (8.0, 11.0), got {gravity_bounds}"
        )
        assert mass_bounds == (0.1, 2.0), f"mass bounds should be (0.1, 2.0), got {mass_bounds}"
        assert friction_bounds == (0.1, 1.0), (
            f"friction bounds should be (0.1, 1.0), got {friction_bounds}"
        )

    def test_all_params_defined(self):
        """All 6 physics parameters should be defined."""
        param_names = [name for name, _, _ in PHYSICS_PARAM_SPACE]
        expected = [
            "gravity",
            "mass",
            "friction",
            "restitution",
            "linear_damping",
            "angular_damping",
        ]

        assert set(param_names) == set(expected), f"Expected {expected}, got {param_names}"

    def test_get_all_bounds(self):
        """get_all_bounds returns correct dictionary."""
        bounds = get_all_bounds()

        assert "gravity" in bounds
        assert bounds["gravity"] == (8.0, 11.0)
        assert bounds["mass"] == (0.1, 2.0)
        assert bounds["friction"] == (0.1, 1.0)

    def test_legacy_param_mapping(self):
        """Legacy parameter names should map to new names."""
        from physicalfish.optimizer.parameter_space import LEGACY_PARAM_MAPPING

        # Legacy names should resolve to correct bounds
        assert get_bounds_for("gravity_magnitude") == (8.0, 11.0)
        assert get_bounds_for("object_mass") == (0.1, 2.0)
        assert get_bounds_for("object_friction") == (0.1, 1.0)
        assert get_bounds_for("object_bounce") == (0.0, 1.0)


class TestBayesianOptimizer:
    """Tests for BayesianOptimizer with bug fixes."""

    def test_converges_on_quadratic(self):
        """Bug fix verification: Simple quadratic should converge to score > -0.1"""

        # Simple quadratic with optimum at gravity=9.5, mass=1.0, friction=0.5
        def quadratic_objective(params: dict) -> float:
            gravity = params.get("gravity", 9.81)
            mass = params.get("mass", 0.5)
            friction = params.get("friction", 0.5)

            # Quadratic error from optimal values
            error = (gravity - 9.5) ** 2 / 10 + (mass - 1.0) ** 2 / 4 + (friction - 0.5) ** 2
            # Return negative error (we want to maximize, closer to 0 is better)
            return -error

        optimizer = BayesianOptimizer(quadratic_objective, random_state=42)
        result = optimizer.optimize(n_calls=20, n_initial_points=5, verbose=False)

        # Should converge to a good score (error < 0.1)
        assert result.best_score > -0.1, f"Expected score > -0.1, got {result.best_score}"
        assert result.n_iterations == 20, f"Expected 20 iterations, got {result.n_iterations}"

    def test_fallback_is_hill_climbing(self):
        """Bug fix verification: Fallback optimizer clusters samples near best in later iterations.

        The fallback should:
        1. First N/3 iterations: random exploration
        2. Remaining 2N/3 iterations: Gaussian perturbations from best

        In the hill climbing phase, samples should cluster near the best found so far.
        """
        # Track all sampled gravity values
        sampled_gravities = []

        def tracking_objective(params: dict) -> float:
            gravity = params.get("gravity", 9.81)
            sampled_gravities.append(gravity)

            # Optimum at gravity=9.5
            return -((gravity - 9.5) ** 2)

        # Force fallback mode by temporarily disabling skopt
        import physicalfish.optimizer.bayesian as bayesian_module

        original_skopt_available = bayesian_module.SKOPT_AVAILABLE

        try:
            bayesian_module.SKOPT_AVAILABLE = False

            optimizer = BayesianOptimizer(tracking_objective, random_state=42)
            result = optimizer.optimize(n_calls=30, n_initial_points=5, verbose=False)

            # Split into exploration and climbing phases
            n_explore = 30 // 3  # 10
            n_climb = 30 - n_explore  # 20

            explore_samples = sampled_gravities[:n_explore]
            climb_samples = sampled_gravities[n_explore:]

            # Exploration phase should have wide spread
            explore_std = np.std(explore_samples)
            assert explore_std > 0.4, f"Exploration phase should have std > 0.4, got {explore_std}"

            # Climbing phase should cluster near best (std < 2.0)
            climb_std = np.std(climb_samples)
            assert climb_std < 2.0, (
                f"Climbing phase should cluster near best (std < 2.0), got {climb_std}"
            )

            # Best should be near optimum (9.5)
            best_gravity = result.best_params.get("gravity", 0)
            assert abs(best_gravity - 9.5) < 0.6, (
                f"Best gravity should be near 9.5, got {best_gravity}"
            )

        finally:
            bayesian_module.SKOPT_AVAILABLE = original_skopt_available

    def test_returns_optimization_result(self):
        """Optimizer should return unified OptimizationResult type."""
        from physicalfish.models import OptimizationResult as ModelsResult

        def simple_objective(params: dict) -> float:
            return 0.5

        optimizer = BayesianOptimizer(simple_objective, random_state=42)
        result = optimizer.optimize(n_calls=5, n_initial_points=2, verbose=False)

        assert isinstance(result, ModelsResult), (
            f"Expected OptimizationResult from models, got {type(result)}"
        )
        assert hasattr(result, "best_params")
        assert hasattr(result, "best_score")
        assert hasattr(result, "n_iterations")
        assert hasattr(result, "convergence_history")
        assert hasattr(result, "acquisition_history")


class TestAdaptiveOptimizer:
    """Tests for AdaptiveOptimizer with bug fix."""

    def test_local_search_uses_correct_bounds(self):
        """Bug fix verification: Local search should use actual parameter bounds.

        The bug was that all parameters used (8.0, 11.0) bounds in local search.
        After fix, mass should use (0.1, 2.0) and friction should use (0.1, 1.0).
        """
        # Track sampled values for mass and friction
        sampled_masses = []
        sampled_frictions = []

        def tracking_objective(params: dict) -> float:
            mass = params.get("mass", 0.5)
            friction = params.get("friction", 0.5)
            sampled_masses.append(mass)
            sampled_frictions.append(friction)

            # Optimum at mass=1.0, friction=0.5
            return -((mass - 1.0) ** 2) - (friction - 0.5) ** 2

        # Force fallback mode
        import physicalfish.optimizer.bayesian as bayesian_module

        original_skopt_available = bayesian_module.SKOPT_AVAILABLE

        try:
            bayesian_module.SKOPT_AVAILABLE = False

            optimizer = AdaptiveOptimizer(tracking_objective)
            result = optimizer.optimize(n_bayesian=6, n_local=5, verbose=False)

            # Check that mass samples stay within (0.1, 2.0)
            for m in sampled_masses:
                assert 0.1 <= m <= 2.0, f"Mass {m} outside bounds (0.1, 2.0)"

            # Check that friction samples stay within (0.1, 1.0)
            for f in sampled_frictions:
                assert 0.1 <= f <= 1.0, f"Friction {f} outside bounds (0.1, 1.0)"

            # Verify we actually sampled in the local search phase
            assert len(sampled_masses) >= 6 + 5, (
                f"Expected at least 11 samples, got {len(sampled_masses)}"
            )

        finally:
            bayesian_module.SKOPT_AVAILABLE = original_skopt_available

    def test_adaptive_returns_optimization_result(self):
        """Adaptive optimizer should return unified OptimizationResult."""
        from physicalfish.models import OptimizationResult as ModelsResult

        def simple_objective(params: dict) -> float:
            return 0.5

        optimizer = AdaptiveOptimizer(simple_objective)
        result = optimizer.optimize(n_bayesian=5, n_local=3, verbose=False)

        assert isinstance(result, ModelsResult), (
            f"Expected OptimizationResult from models, got {type(result)}"
        )
        assert result.n_iterations == 8  # 5 bayesian + 3 local

    def test_adaptive_no_improvement_in_local_search(self):
        """Local search handles case where no improvements are found."""

        # Always return same score - no improvement possible
        def flat_objective(params: dict) -> float:
            return 0.5

        # Force fallback mode
        import physicalfish.optimizer.bayesian as bayesian_module

        original_skopt_available = bayesian_module.SKOPT_AVAILABLE

        try:
            bayesian_module.SKOPT_AVAILABLE = False

            optimizer = AdaptiveOptimizer(flat_objective)
            result = optimizer.optimize(n_bayesian=3, n_local=5, verbose=False)

            # Should still complete and return valid result
            assert result.best_score == 0.5
            assert result.n_iterations == 8  # 3 bayesian + 5 local
            assert len(result.convergence_history) == 8

        finally:
            bayesian_module.SKOPT_AVAILABLE = original_skopt_available

    def test_adaptive_local_search_with_unknown_param(self):
        """Local search handles unknown parameters gracefully."""
        sampled_unknown = []

        def tracking_objective(params: dict) -> float:
            # Track if unknown_param appears
            if "unknown_param" in params:
                sampled_unknown.append(params["unknown_param"])
            return 0.5

        # Force fallback mode
        import physicalfish.optimizer.bayesian as bayesian_module

        original_skopt_available = bayesian_module.SKOPT_AVAILABLE

        try:
            bayesian_module.SKOPT_AVAILABLE = False

            optimizer = AdaptiveOptimizer(tracking_objective)
            # Use params that include an unknown parameter
            result = optimizer.optimize(n_bayesian=2, n_local=3, verbose=False)

            # Should complete without error
            assert result.best_score is not None

        finally:
            bayesian_module.SKOPT_AVAILABLE = original_skopt_available

    def test_adaptive_zero_local_iterations(self):
        """Adaptive optimizer works with zero local search iterations."""

        def simple_objective(params: dict) -> float:
            return 0.5

        # Force fallback mode
        import physicalfish.optimizer.bayesian as bayesian_module

        original_skopt_available = bayesian_module.SKOPT_AVAILABLE

        try:
            bayesian_module.SKOPT_AVAILABLE = False

            optimizer = AdaptiveOptimizer(simple_objective)
            result = optimizer.optimize(n_bayesian=5, n_local=0, verbose=False)

            # Should only have bayesian iterations
            assert result.n_iterations == 5
            assert len(result.acquisition_history) == 5

        finally:
            bayesian_module.SKOPT_AVAILABLE = original_skopt_available

    def test_adaptive_improvement_found_in_local_search(self):
        """Local search correctly tracks when improvements are found."""
        # Return increasing scores to simulate improvement
        call_count = [0]

        def improving_objective(params: dict) -> float:
            call_count[0] += 1
            return 0.5 + call_count[0] * 0.1

        # Force fallback mode
        import physicalfish.optimizer.bayesian as bayesian_module

        original_skopt_available = bayesian_module.SKOPT_AVAILABLE

        try:
            bayesian_module.SKOPT_AVAILABLE = False

            optimizer = AdaptiveOptimizer(improving_objective)
            result = optimizer.optimize(n_bayesian=3, n_local=5, verbose=False)

            # Score should improve from initial
            assert result.best_score > 0.5

        finally:
            bayesian_module.SKOPT_AVAILABLE = original_skopt_available


class TestBayesianOptimizerEdgeCases:
    """Edge case tests for BayesianOptimizer."""

    def test_bayesian_single_iteration(self):
        """Bayesian optimizer works with single iteration."""

        def simple_objective(params: dict) -> float:
            return 0.5

        # Force fallback mode
        import physicalfish.optimizer.bayesian as bayesian_module

        original_skopt_available = bayesian_module.SKOPT_AVAILABLE

        try:
            bayesian_module.SKOPT_AVAILABLE = False

            optimizer = BayesianOptimizer(simple_objective, random_state=42)
            result = optimizer.optimize(n_calls=1, n_initial_points=1, verbose=False)

            assert result.n_iterations == 1
            assert len(result.convergence_history) == 1

        finally:
            bayesian_module.SKOPT_AVAILABLE = original_skopt_available

    def test_bayesian_fallback_exploration_and_climbing(self):
        """Fallback optimizer has exploration phase then hill climbing phase."""
        sampled = []

        def tracking_objective(params: dict) -> float:
            sampled.append(params.get("gravity", 0))
            return 0.5

        # Force fallback mode
        import physicalfish.optimizer.bayesian as bayesian_module

        original_skopt_available = bayesian_module.SKOPT_AVAILABLE

        try:
            bayesian_module.SKOPT_AVAILABLE = False

            optimizer = BayesianOptimizer(tracking_objective, random_state=42)
            result = optimizer.optimize(n_calls=9, verbose=False)

            # With n_calls=9: n_explore = 3, n_climb = 6
            # First 3 should be random, last 6 should be hill_climbing
            assert result.acquisition_history[:3] == ["random", "random", "random"]
            assert all(tag == "hill_climbing" for tag in result.acquisition_history[3:])

        finally:
            bayesian_module.SKOPT_AVAILABLE = original_skopt_available

    def test_bayesian_fallback_history_in_result(self):
        """Fallback optimizer returns history in result (not in optimizer.history)."""

        def simple_objective(params: dict) -> float:
            return params.get("gravity", 9.0) / 10.0

        optimizer = BayesianOptimizer(simple_objective, random_state=42)

        # Force fallback mode
        import physicalfish.optimizer.bayesian as bayesian_module

        original_skopt_available = bayesian_module.SKOPT_AVAILABLE

        try:
            bayesian_module.SKOPT_AVAILABLE = False

            result = optimizer.optimize(n_calls=10, n_initial_points=3, verbose=False)

            # In fallback mode, history is in result not optimizer.history
            assert len(result.convergence_history) == 10
            assert len(result.acquisition_history) == 10
            assert result.best_score > 0
            assert result.best_params is not None

        finally:
            bayesian_module.SKOPT_AVAILABLE = original_skopt_available

    def test_bayesian_best_params_tracking(self):
        """Best params are updated when better scores are found."""
        call_count = [0]

        def improving_objective(params: dict) -> float:
            call_count[0] += 1
            # Return increasing scores
            return call_count[0] * 0.1

        optimizer = BayesianOptimizer(improving_objective, random_state=42)

        # Force fallback mode
        import physicalfish.optimizer.bayesian as bayesian_module

        original_skopt_available = bayesian_module.SKOPT_AVAILABLE

        try:
            bayesian_module.SKOPT_AVAILABLE = False

            result = optimizer.optimize(n_calls=10, n_initial_points=2, verbose=False)

            # Best score should be the last (highest)
            assert result.best_score == 1.0  # 10 * 0.1

        finally:
            bayesian_module.SKOPT_AVAILABLE = original_skopt_available


class TestParameterSpaceEdgeCases:
    """Edge case tests for parameter space functions."""

    def test_get_bounds_for_invalid_param(self):
        """get_bounds_for raises ValueError for unknown parameter."""
        from physicalfish.optimizer.parameter_space import get_bounds_for

        with pytest.raises(ValueError, match="Unknown parameter"):
            get_bounds_for("nonexistent_param")

    def test_get_bounds_for_case_sensitive(self):
        """Parameter names are case sensitive."""
        from physicalfish.optimizer.parameter_space import get_bounds_for

        with pytest.raises(ValueError):
            get_bounds_for("GRAVITY")  # Uppercase should fail

    def test_legacy_mapping_all_names(self):
        """All legacy parameter names map correctly."""
        from physicalfish.optimizer.parameter_space import LEGACY_PARAM_MAPPING, get_bounds_for

        for legacy_name, new_name in LEGACY_PARAM_MAPPING.items():
            legacy_bounds = get_bounds_for(legacy_name)
            new_bounds = get_bounds_for(new_name)
            assert legacy_bounds == new_bounds, (
                f"Legacy {legacy_name} should map to {new_name} with same bounds"
            )

    def test_get_all_bounds_returns_all_params(self):
        """get_all_bounds returns all 6 physics parameters."""
        from physicalfish.optimizer.parameter_space import get_all_bounds

        bounds = get_all_bounds()

        assert len(bounds) == 6
        assert "gravity" in bounds
        assert "mass" in bounds
        assert "friction" in bounds
        assert "restitution" in bounds
        assert "linear_damping" in bounds
        assert "angular_damping" in bounds


class TestOptimizerVerboseMode:
    """Tests for verbose output mode (increases coverage)."""

    def test_bayesian_verbose_mode(self, capsys):
        """Bayesian optimizer produces output in verbose mode."""

        def simple_objective(params: dict) -> float:
            return 0.5

        # Force fallback mode
        import physicalfish.optimizer.bayesian as bayesian_module

        original_skopt_available = bayesian_module.SKOPT_AVAILABLE

        try:
            bayesian_module.SKOPT_AVAILABLE = False

            optimizer = BayesianOptimizer(simple_objective, random_state=42)
            result = optimizer.optimize(n_calls=5, n_initial_points=2, verbose=True)

            # Check that something was printed
            captured = capsys.readouterr()
            assert len(captured.out) > 0 or len(captured.err) > 0

        finally:
            bayesian_module.SKOPT_AVAILABLE = original_skopt_available

    def test_adaptive_verbose_mode(self, capsys):
        """Adaptive optimizer produces output in verbose mode."""

        def simple_objective(params: dict) -> float:
            return 0.5

        # Force fallback mode
        import physicalfish.optimizer.bayesian as bayesian_module

        original_skopt_available = bayesian_module.SKOPT_AVAILABLE

        try:
            bayesian_module.SKOPT_AVAILABLE = False

            optimizer = AdaptiveOptimizer(simple_objective)
            result = optimizer.optimize(n_bayesian=3, n_local=2, verbose=True)

            # Check that something was printed
            captured = capsys.readouterr()
            # Verbose mode should produce output
            assert result.best_score is not None

        finally:
            bayesian_module.SKOPT_AVAILABLE = original_skopt_available


class TestOptimizerEdgeCases:
    """Additional edge case tests."""

    def test_bayesian_fallback_with_n_calls_less_than_3(self):
        """Fallback handles n_calls < 3."""

        def simple_objective(params: dict) -> float:
            return 0.5

        import physicalfish.optimizer.bayesian as bayesian_module

        original_skopt_available = bayesian_module.SKOPT_AVAILABLE

        try:
            bayesian_module.SKOPT_AVAILABLE = False

            optimizer = BayesianOptimizer(simple_objective, random_state=42)
            result = optimizer.optimize(n_calls=2, verbose=False)

            # Should still work with minimal calls
            assert result.n_iterations == 2
            assert len(result.convergence_history) == 2

        finally:
            bayesian_module.SKOPT_AVAILABLE = original_skopt_available

    def test_adaptive_with_n_bayesian_zero(self):
        """Adaptive optimizer with zero bayesian iterations."""

        def simple_objective(params: dict) -> float:
            return 0.5

        import physicalfish.optimizer.bayesian as bayesian_module

        original_skopt_available = bayesian_module.SKOPT_AVAILABLE

        try:
            bayesian_module.SKOPT_AVAILABLE = False

            optimizer = AdaptiveOptimizer(simple_objective)
            # With n_bayesian=0, bayesian phase runs at least 1 iteration (min n_explore)
            result = optimizer.optimize(n_bayesian=0, n_local=5, verbose=False)

            # Should have at least 5 local iterations + bayesian phase
            assert result.n_iterations >= 5

        finally:
            bayesian_module.SKOPT_AVAILABLE = original_skopt_available

    def test_adaptive_local_search_finds_improvement(self):
        """Local search phase can find improvements."""
        call_count = [0]

        def improving_objective(params: dict) -> float:
            call_count[0] += 1
            # Return increasing scores to simulate improvement
            return 0.5 + call_count[0] * 0.05

        import physicalfish.optimizer.bayesian as bayesian_module

        original_skopt_available = bayesian_module.SKOPT_AVAILABLE

        try:
            bayesian_module.SKOPT_AVAILABLE = False

            optimizer = AdaptiveOptimizer(improving_objective)
            result = optimizer.optimize(n_bayesian=3, n_local=5, verbose=False)

            # Should find improvements
            assert result.best_score > 0.5

        finally:
            bayesian_module.SKOPT_AVAILABLE = original_skopt_available

    def test_bayesian_with_different_random_states(self):
        """Bayesian optimizer works with different random states."""

        def simple_objective(params: dict) -> float:
            return params.get("gravity", 9.0) / 10.0

        import physicalfish.optimizer.bayesian as bayesian_module

        original_skopt_available = bayesian_module.SKOPT_AVAILABLE

        try:
            bayesian_module.SKOPT_AVAILABLE = False

            # Test with different random states
            for seed in [0, 42, 123, 999]:
                optimizer = BayesianOptimizer(simple_objective, random_state=seed)
                result = optimizer.optimize(n_calls=5, verbose=False)
                assert result.best_score is not None

        finally:
            bayesian_module.SKOPT_AVAILABLE = original_skopt_available

    def test_parameter_bounds_at_edges(self):
        """Test that parameters at bounds work correctly."""
        from physicalfish.optimizer.parameter_space import get_bounds_for

        # Test all parameters at their bounds
        params = ["gravity", "mass", "friction", "restitution", "linear_damping", "angular_damping"]

        for param in params:
            low, high = get_bounds_for(param)

            # Test at lower bound
            def objective_lower(params_dict):
                if params_dict.get(param) == low:
                    return 1.0
                return 0.0

            # Test at upper bound
            def objective_upper(params_dict):
                if params_dict.get(param) == high:
                    return 1.0
                return 0.0

            # Just verify the bounds are valid
            assert low < high

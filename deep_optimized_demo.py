#!/usr/bin/env python3
"""
PhysicalFish — Deep Optimized Demo
Production-grade AutoResearch with:
- Advanced Physics Verification (6 constraints)
- Bayesian Optimization (Gaussian Processes)
- Real-time Dashboard
"""

import sys
import json
import time
import random
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional

# Import our advanced modules
sys.path.insert(0, str(Path(__file__).parent / "verification"))
sys.path.insert(0, str(Path(__file__).parent / "autoresearch"))

try:
    from advanced_physics import AdvancedPhysicsVerifier, FrameData
    from bayesian_optimizer import BayesianOptimizer, AdaptiveOptimizer

    ADVANCED_MODE = True
except ImportError as e:
    print(f"Warning: Advanced modules not available ({e})")
    print("Falling back to basic mode...")
    ADVANCED_MODE = False

# Colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"


@dataclass
class OptimizationState:
    iteration: int
    params: Dict[str, float]
    score: float
    details: Dict
    timestamp: float


class DeepOptimizedDemo:
    """Production-grade AutoResearch demo"""

    def __init__(
        self,
        target_score: float = 0.90,
        max_iterations: int = 15,
        use_bayesian: bool = True,
        interactive: bool = True,
    ):
        self.target_score = target_score
        self.max_iterations = max_iterations
        self.use_bayesian = use_bayesian and ADVANCED_MODE
        self.interactive = interactive

        self.history: List[OptimizationState] = []
        self.best_score = 0.0
        self.best_params = None

        # Initialize advanced verifier if available
        if ADVANCED_MODE:
            self.verifier = AdvancedPhysicsVerifier(gravity=9.8, tolerance=0.05)

        # Parameter bounds
        self.bounds = {
            "gravity_magnitude": (8.0, 11.0),
            "object_mass": (0.1, 2.0),
            "object_friction": (0.1, 1.0),
            "object_bounce": (0.0, 1.0),
            "linear_damping": (0.0, 1.0),
        }

        self.current_params = {
            "gravity_magnitude": 9.8,
            "object_mass": 0.5,
            "object_friction": 0.5,
            "object_bounce": 0.3,
            "linear_damping": 0.1,
        }

    def generate_synthetic_batch(self, params: Dict) -> Dict:
        """Generate synthetic trajectory with given physics params"""
        # Simulate 3-second grasping sequence
        frames = []
        dt = 0.033  # 30 fps

        g = params["gravity_magnitude"]
        m = params["object_mass"]

        for i in range(90):  # 3 seconds at 30fps
            t = i * dt

            # Simple falling object physics
            if t < 1.0:
                # Falling
                y = 1.0 - 0.5 * g * t**2
                v_y = -g * t
                a_y = -g
            elif t < 2.0:
                # Grabbed (stationary)
                y = 1.0 - 0.5 * g * 1.0**2
                v_y = 0
                a_y = 0
            else:
                # Released
                t_fall = t - 2.0
                y = (1.0 - 0.5 * g * 1.0**2) - 0.5 * g * t_fall**2
                v_y = -g * t_fall
                a_y = -g

            # Ensure object doesn't go below ground
            y = max(0.15, y)

            frames.append(
                {
                    "timestamp": t,
                    "object": {
                        "position": [0.0, y, -0.5],
                        "velocity": [0.0, v_y, 0.0],
                        "acceleration": [0.0, a_y, 0.0],
                        "rotation": [0.0, 0.0, 0.0],
                        "angular_velocity": [0.0, 0.0, 0.0],
                    },
                }
            )

        return {"frames": frames, "physics_params": params, "mass": m}

    def evaluate_params(self, params: Dict) -> float:
        """Evaluate physics parameters (objective function for optimizer)"""
        # Generate synthetic data
        batch = self.generate_synthetic_batch(params)

        if ADVANCED_MODE:
            # Use advanced verifier with 6 constraints
            frames = []
            for frame_dict in batch["frames"]:
                obj = frame_dict["object"]
                frames.append(
                    FrameData(
                        timestamp=frame_dict["timestamp"],
                        position=np.array(obj["position"]),
                        velocity=np.array(obj["velocity"]),
                        acceleration=np.array(obj["acceleration"]),
                        rotation=np.array(obj["rotation"]),
                        angular_velocity=np.array(obj["angular_velocity"]),
                        mass=batch["mass"],
                    )
                )

            result = self.verifier.verify_trajectory(frames)
            return result["overall_score"]
        else:
            # Fallback: simple score based on parameter proximity to optimal
            g_error = abs(params["gravity_magnitude"] - 9.2) / 2.0
            m_error = abs(params["object_mass"] - 0.6) / 1.0
            f_error = abs(params["object_friction"] - 0.7) / 0.5

            score = 0.95 - (g_error + m_error + f_error) * 0.3
            noise = random.gauss(0, 0.02)
            return max(0.3, min(0.98, score + noise))

    def run(self):
        """Run the deep optimized demo"""
        self._print_header()

        start_time = time.time()

        if self.use_bayesian and ADVANCED_MODE:
            # Use Bayesian Optimization
            print(f"{CYAN}Using Bayesian Optimization (Gaussian Processes){RESET}")
            print(f"{CYAN}Acquisition: Expected Improvement (EI){RESET}\n")

            optimizer = BayesianOptimizer(
                evaluator=self.evaluate_params, random_state=42
            )

            result = optimizer.optimize(
                n_calls=self.max_iterations, n_initial_points=5, verbose=True
            )

            self.best_score = result.best_score
            self.best_params = result.best_params

            # Convert to history format
            for i, score in enumerate(result.convergence_history):
                self.history.append(
                    OptimizationState(
                        iteration=i,
                        params={},  # Simplified
                        score=score,
                        details={"acquisition": result.acquisition_history[i]},
                        timestamp=time.time(),
                    )
                )
        else:
            # Use simple hill climbing
            print(f"{YELLOW}Using Hill Climbing (Bayesian opt not available){RESET}\n")
            self._run_hill_climbing()

        elapsed = time.time() - start_time

        # Print results
        self._print_results(elapsed)

        return {
            "final_score": self.best_score,
            "best_params": self.best_params,
            "iterations": len(self.history),
            "elapsed": elapsed,
        }

    def _run_hill_climbing(self):
        """Fallback hill climbing optimization"""
        for iteration in range(self.max_iterations):
            # Propose new params
            if iteration == 0:
                params = self.current_params.copy()
            else:
                params = self._propose_params()

            # Evaluate
            score = self.evaluate_params(params)

            # Decide
            if iteration == 0 or score > self.best_score:
                action = "commit"
                self.best_score = score
                self.best_params = params.copy()
                self.current_params = params.copy()
                improvement = score - (self.history[-1].score if self.history else 0)
            else:
                action = "revert"
                improvement = score - self.best_score

            # Record
            state = OptimizationState(
                iteration=iteration,
                params=params,
                score=score,
                details={"action": action, "improvement": improvement},
                timestamp=time.time(),
            )
            self.history.append(state)

            # Print
            self._print_iteration(state)

            # Check target
            if self.best_score >= self.target_score:
                print(f"\n{BOLD}{GREEN}🎯 TARGET REACHED!{RESET}\n")
                break

    def _propose_params(self) -> Dict:
        """Propose new parameters with perturbation"""
        new_params = self.current_params.copy()

        num_perturb = random.randint(1, 2)
        params_to_perturb = random.sample(list(self.bounds.keys()), num_perturb)

        for param in params_to_perturb:
            min_val, max_val = self.bounds[param]
            current = new_params[param]
            std = (max_val - min_val) * 0.15
            new_val = current + random.gauss(0, std)
            new_val = max(min_val, min(max_val, new_val))
            new_params[param] = round(new_val, 3)

        return new_params

    def _print_header(self):
        """Print demo header"""
        print(f"\n{BOLD}{CYAN}")
        print("╔════════════════════════════════════════════════════════════════╗")
        print("║     PHYSICALFISH — DEEP OPTIMIZED AUTORESEARCH                 ║")
        print("║                                                                ║")
        if ADVANCED_MODE:
            print("║  ✓ Advanced Physics (6 constraints)                            ║")
            print("║  ✓ Bayesian Optimization (Gaussian Processes)                  ║")
        else:
            print("║  ⚠ Basic Mode (Advanced modules not available)               ║")
        print("╚════════════════════════════════════════════════════════════════╝")
        print(f"{RESET}\n")

        print(f"{YELLOW}Configuration:{RESET}")
        print(f"  Target Score:      {self.target_score:.2f}")
        print(f"  Max Iterations:    {self.max_iterations}")
        print(
            f"  Physics Verifier:  {'Advanced (6 constraints)' if ADVANCED_MODE else 'Basic'}"
        )
        print(
            f"  Optimizer:         {'Bayesian (GP-EI)' if self.use_bayesian and ADVANCED_MODE else 'Hill Climbing'}"
        )
        print()

        if self.interactive:
            input(f"{CYAN}Press Enter to start...{RESET}")
            print()

    def _print_iteration(self, state: OptimizationState):
        """Print iteration result"""
        iter_str = f"{state.iteration:2d}"
        score_str = f"{state.score:.4f}"

        action = state.details.get("action", "unknown")

        if state.iteration == 0:
            print(
                f"{BLUE}Iter {iter_str}{RESET}: {YELLOW}{score_str}{RESET} {CYAN}(baseline){RESET}"
            )
        elif action == "commit":
            improvement = state.details.get("improvement", 0)
            print(
                f"{GREEN}Iter {iter_str}{RESET}: {GREEN}{score_str}{RESET} {GREEN}✓ COMMIT{RESET} (+{improvement:.4f})"
            )
        else:
            print(
                f"{RED}Iter {iter_str}{RESET}: {YELLOW}{score_str}{RESET} {RED}✗ REVERT{RESET}"
            )

    def _print_results(self, elapsed: float):
        """Print final results"""
        print(f"\n{BOLD}{CYAN}{'=' * 60}{RESET}")
        print(f"{BOLD}OPTIMIZATION COMPLETE{RESET}")
        print(f"{BOLD}{CYAN}{'=' * 60}{RESET}")

        print(f"\n{YELLOW}Results:{RESET}")
        print(f"  Final Score:       {GREEN}{self.best_score:.4f}{RESET}")
        print(f"  Target Score:      {self.target_score:.4f}")
        print(f"  Iterations:        {len(self.history)}")
        print(f"  Time Elapsed:      {elapsed:.2f}s")
        print(
            f"  Converged:         {GREEN}✓ YES{RESET}"
            if self.best_score >= self.target_score
            else f"  {RED}✗ NO{RESET}"
        )

        if self.best_params:
            print(f"\n{YELLOW}Best Parameters:{RESET}")
            for key, val in self.best_params.items():
                print(f"  {key:20s}: {CYAN}{val:.3f}{RESET}")

        # Convergence curve
        print(f"\n{YELLOW}Convergence Curve:{RESET}")
        print("-" * 50)

        scores = [s.score for s in self.history]
        max_score = max(scores)
        min_score = min(scores)

        for i, score in enumerate(scores):
            if max_score == min_score:
                bar_len = 10
            else:
                bar_len = int((score - min_score) / (max_score - min_score) * 20)

            bar = "█" * bar_len + "░" * (20 - bar_len)
            marker = "★" if score == max_score else " "

            if i == 0:
                color = BLUE
            elif score == max_score:
                color = GREEN
            else:
                color = YELLOW

            print(f"{color}Iter {i:2d}: {bar} {score:.3f} {marker}{RESET}")

        print("-" * 50)

        print(f"\n{BOLD}{GREEN}✓ Deep Optimized AutoResearch complete!{RESET}")
        print(
            f"{CYAN}  Physics-informed optimization with {len(self.history)} iterations{RESET}\n"
        )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="PhysicalFish Deep Optimized Demo")
    parser.add_argument(
        "--target", type=float, default=0.90, help="Target physics score"
    )
    parser.add_argument("--iterations", type=int, default=15, help="Max iterations")
    parser.add_argument(
        "--no-bayesian",
        action="store_true",
        help="Use hill climbing instead of Bayesian",
    )
    parser.add_argument(
        "--no-interactive", action="store_true", help="Skip interactive prompts"
    )

    args = parser.parse_args()

    demo = DeepOptimizedDemo(
        target_score=args.target,
        max_iterations=args.iterations,
        use_bayesian=not args.no_bayesian,
        interactive=not args.no_interactive,
    )

    result = demo.run()

    # Save result
    output = {
        "final_score": result["final_score"],
        "best_params": result["best_params"],
        "iterations": result["iterations"],
        "elapsed_time": result["elapsed"],
        "advanced_mode": ADVANCED_MODE,
        "convergence_curve": [s.score for s in demo.history],
    }

    output_path = Path("deep_optimized_result.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"{CYAN}Result saved to: {output_path}{RESET}\n")


if __name__ == "__main__":
    main()

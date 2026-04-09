#!/usr/bin/env python3
"""
PhysicalFish — Fast Demo Mode for Hackathon
Optimized for 5-minute presentation with real-time visualization.
"""

import json
import time
import random
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict
import sys

# Colors for terminal output
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
    improvement: float
    action: str
    timestamp: float


class FastSyntheticGenerator:
    """Fast mock generator for demo purposes (no Godot needed)"""

    def __init__(self):
        self.params = {}

    def generate_batch(self, params: Dict) -> Dict:
        """Simulate synthetic batch generation instantly"""
        # Simulate physics score based on parameters
        # Optimal: gravity ~9.2, mass ~0.6, friction ~0.7

        gravity_error = abs(params["gravity_magnitude"] - 9.2) / 2.0
        mass_error = abs(params["object_mass"] - 0.6) / 1.0
        friction_error = abs(params["object_friction"] - 0.7) / 0.5

        # Base score with noise
        base_score = 0.95 - (gravity_error + mass_error + friction_error) * 0.3
        noise = random.gauss(0, 0.02)
        score = max(0.3, min(0.98, base_score + noise))

        return {
            "score": score,
            "params": params,
            "batch_id": f"batch_{int(time.time() * 1000)}",
        }


class FastPINNsVerifier:
    """Fast verification for demo"""

    def verify(self, synthetic_batch: Dict, real_data: Dict) -> Dict:
        """Return detailed verification result"""
        score = synthetic_batch["score"]

        return {
            "verification_result": {
                "overall_score": score,
                "kinematic_score": score + random.gauss(0, 0.01),
                "dynamic_score": score + random.gauss(0, 0.01),
                "temporal_score": score + random.gauss(0, 0.02),
            },
            "passed": score > 0.7,
        }


class FastAutoResearchDemo:
    """Optimized AutoResearch loop for live demo"""

    def __init__(self, target_score: float = 0.90, max_iterations: int = 15):
        self.target_score = target_score
        self.max_iterations = max_iterations
        self.generator = FastSyntheticGenerator()
        self.verifier = FastPINNsVerifier()

        self.history: List[OptimizationState] = []
        self.best_score = 0.0
        self.best_params = None

        # Parameter bounds
        self.bounds = {
            "gravity_magnitude": (8.0, 11.0),
            "object_mass": (0.1, 2.0),
            "object_friction": (0.1, 1.0),
        }

        # Default params
        self.current_params = {
            "gravity_magnitude": 9.8,
            "object_mass": 0.5,
            "object_friction": 0.5,
        }

    def _propose_params(self) -> Dict:
        """Propose new parameters"""
        new_params = self.current_params.copy()

        # Perturb 1-2 random parameters
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

    def __init__(
        self,
        target_score: float = 0.90,
        max_iterations: int = 15,
        interactive: bool = True,
    ):
        self.target_score = target_score
        self.max_iterations = max_iterations
        self.interactive = interactive
        self.generator = FastSyntheticGenerator()
        self.verifier = FastPINNsVerifier()

        self.history: List[OptimizationState] = []
        self.best_score = 0.0
        self.best_params = None

        # Parameter bounds
        self.bounds = {
            "gravity_magnitude": (8.0, 11.0),
            "object_mass": (0.1, 2.0),
            "object_friction": (0.1, 1.0),
        }

        # Default params
        self.current_params = {
            "gravity_magnitude": 9.8,
            "object_mass": 0.5,
            "object_friction": 0.5,
        }

    def _print_header(self):
        """Print demo header"""
        print(f"\n{BOLD}{CYAN}")
        print("╔════════════════════════════════════════════════════════════════╗")
        print("║           PHYSICALFISH — AUTORESEARCH DEMO                     ║")
        print("║     Auto-optimizing synthetic data quality (Karpathy style)    ║")
        print("╚════════════════════════════════════════════════════════════════╝")
        print(f"{RESET}\n")

        print(f"{YELLOW}Configuration:{RESET}")
        print(f"  Target Score:      {self.target_score:.2f}")
        print(f"  Max Iterations:    {self.max_iterations}")
        print(f"  Parameters:        gravity, mass, friction")
        print()

        print(f"{YELLOW}AutoResearch Loop:{RESET}")
        print(f"  1. PROPOSE  → Perturb physics parameters")
        print(f"  2. GENERATE → Create synthetic batch")
        print(f"  3. VERIFY   → PINNs physics consistency check")
        print(f"  4. DECIDE   → COMMIT if improved, else REVERT")
        print()

        if self.interactive:
            input(f"{CYAN}Press Enter to start...{RESET}")
        print()

    def _print_iteration(self, state: OptimizationState):
        """Print iteration result with visual flair"""
        iter_str = f"{state.iteration:2d}"
        score_str = f"{state.score:.4f}"

        if state.iteration == 0:
            # Baseline
            print(
                f"{BLUE}Iter {iter_str}{RESET}: {YELLOW}{score_str}{RESET} {CYAN}(baseline){RESET}"
            )
        elif state.action == "commit":
            # Improvement
            improvement = state.improvement
            bar = "█" * int(improvement * 100)
            print(
                f"{GREEN}Iter {iter_str}{RESET}: {GREEN}{score_str}{RESET} {GREEN}✓ COMMIT{RESET} (+{improvement:.4f}) {GREEN}{bar}{RESET}"
            )
        else:
            # Revert
            print(
                f"{RED}Iter {iter_str}{RESET}: {YELLOW}{score_str}{RESET} {RED}✗ REVERT{RESET} ({state.improvement:+.4f})"
            )

        # Show current best
        if state.score == max(s.score for s in self.history):
            print(f"         {CYAN}★ New best!{RESET}")

    def _print_convergence_curve(self):
        """Print ASCII convergence curve"""
        print(f"\n{YELLOW}Convergence Curve:{RESET}")
        print("-" * 50)

        scores = [s.score for s in self.history]
        max_score = max(scores)
        min_score = min(scores)

        for i, score in enumerate(scores):
            # Normalize to 0-20 for display
            if max_score == min_score:
                bar_len = 10
            else:
                bar_len = int((score - min_score) / (max_score - min_score) * 20)

            bar = "█" * bar_len + "░" * (20 - bar_len)
            marker = "★" if score == max_score else " "

            color = (
                GREEN
                if self.history[i].action == "commit"
                else (RED if i > 0 else BLUE)
            )
            print(f"{color}Iter {i:2d}: {bar} {score:.4f} {marker}{RESET}")

        print("-" * 50)

    def run(self):
        """Run the optimized demo"""
        self._print_header()

        start_time = time.time()

        for iteration in range(self.max_iterations):
            # Propose
            if iteration == 0:
                params = self.current_params.copy()
            else:
                params = self._propose_params()

            # Generate (instant)
            batch = self.generator.generate_batch(params)

            # Verify (instant)
            real_data = {}  # Mock
            verification = self.verifier.verify(batch, real_data)
            score = verification["verification_result"]["overall_score"]

            # Decide
            if iteration == 0:
                improvement = 0.0
                action = "commit"
                self.best_score = score
                self.best_params = params.copy()
            else:
                improvement = score - self.best_score
                if improvement > 0.001:
                    action = "commit"
                    self.best_score = score
                    self.best_params = params.copy()
                    self.current_params = params.copy()
                else:
                    action = "revert"

            # Record
            state = OptimizationState(
                iteration=iteration,
                params=params,
                score=score,
                improvement=improvement,
                action=action,
                timestamp=time.time(),
            )
            self.history.append(state)

            # Print
            self._print_iteration(state)

            # Check target
            if self.best_score >= self.target_score:
                print(f"\n{BOLD}{GREEN}🎯 TARGET REACHED!{RESET}")
                break

        elapsed = time.time() - start_time

        # Summary
        print(f"\n{BOLD}{CYAN}{'=' * 60}{RESET}")
        print(f"{BOLD}DEMO COMPLETE{RESET}")
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

        print(f"\n{YELLOW}Best Parameters:{RESET}")
        for key, val in self.best_params.items():
            print(f"  {key:20s}: {CYAN}{val:.3f}{RESET}")

        self._print_convergence_curve()

        print(
            f"\n{BOLD}{GREEN}✓ AutoResearch loop successfully optimized synthetic data quality!{RESET}"
        )
        print(
            f"{CYAN}  Physics score improved: {self.history[0].score:.2f} → {self.best_score:.2f} ({(self.best_score / self.history[0].score - 1) * 100:.1f}% improvement){RESET}\n"
        )

        return {
            "final_score": self.best_score,
            "iterations": len(self.history),
            "best_params": self.best_params,
            "history": self.history,
        }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="PhysicalFish Fast Demo")
    parser.add_argument(
        "--target", type=float, default=0.90, help="Target physics score"
    )
    parser.add_argument("--iterations", type=int, default=15, help="Max iterations")
    parser.add_argument(
        "--no-interactive", action="store_true", help="Skip interactive prompts"
    )

    args = parser.parse_args()

    demo = FastAutoResearchDemo(
        target_score=args.target,
        max_iterations=args.iterations,
        interactive=not args.no_interactive,
    )

    result = demo.run()

    # Save result
    output = {
        "final_score": result["final_score"],
        "iterations": result["iterations"],
        "best_params": result["best_params"],
        "convergence_curve": [
            {"iteration": s.iteration, "score": s.score, "action": s.action}
            for s in result["history"]
        ],
    }

    output_path = Path("hackathon_demo_result.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"{CYAN}Result saved to: {output_path}{RESET}\n")


if __name__ == "__main__":
    main()

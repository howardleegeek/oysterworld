"""
PhysicalFish AutoResearch Optimization Loop

Implements the Karpathy autoresearch pattern:
1. Generate synthetic batch with current parameters
2. Verify physics consistency (PINNs)
3. Score improved? → COMMIT : REVERT
4. Repeat until convergence

This creates the "data flywheel" effect.
"""

import json
import subprocess
import numpy as np
import argparse
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import random


@dataclass
class OptimizationState:
    """State of the optimization process"""

    iteration: int
    params: Dict[str, float]
    score: float
    improvement: float
    action: str  # "commit" or "revert"
    timestamp: float


class GodotController:
    """Controls Godot synthetic data generation via JSON-RPC/config files"""

    def __init__(self, godot_project_path: Path, output_dir: Path):
        self.project_path = godot_project_path
        self.output_dir = output_dir
        self.current_params = self._default_params()

    def _default_params(self) -> Dict[str, float]:
        return {
            "gravity_magnitude": 9.8,
            "object_mass": 0.5,
            "object_friction": 0.5,
            "object_bounce": 0.3,
            "linear_damping": 0.1,
            "angular_damping": 0.1,
        }

    def set_params(self, params: Dict[str, float]):
        """Update Godot physics parameters"""
        self.current_params.update(params)

        # Write params to config file that Godot will read
        config_path = self.project_path / "autoresearch_config.json"
        with open(config_path, "w") as f:
            json.dump(self.current_params, f, indent=2)

    def generate_batch(self, batch_id: str) -> Path:
        """Run Godot headless to generate synthetic data"""
        output_path = self.output_dir / f"{batch_id}_metadata.json"

        # Run Godot in headless mode
        cmd = [
            "godot",
            "--headless",
            "--path",
            str(self.project_path),
            "--script",
            "scripts/generator.gd",
            "--batch-id",
            batch_id,
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,  # 2 minute timeout per batch
            )

            if result.returncode != 0:
                print(f"Godot error: {result.stderr}")
                return None

            # Godot saves to user:// which maps to different locations per OS
            # We need to find the output file
            possible_paths = [
                Path.home()
                / ".local/share/godot/app_userdata/PhysicalFish Synthetic Data Generator"
                / f"{batch_id}_metadata.json",
                Path.home()
                / "Library/Application Support/Godot/app_userdata/PhysicalFish Synthetic Data Generator"
                / f"{batch_id}_metadata.json",
                Path.home()
                / "AppData/Roaming/Godot/app_userdata/PhysicalFish Synthetic Data Generator"
                / f"{batch_id}_metadata.json",
            ]

            for src_path in possible_paths:
                if src_path.exists():
                    # Copy to our output dir
                    import shutil

                    shutil.copy(src_path, output_path)
                    return output_path

            print(f"Could not find Godot output for batch {batch_id}")
            return None

        except subprocess.TimeoutExpired:
            print(f"Godot generation timed out for batch {batch_id}")
            return None
        except Exception as e:
            print(f"Error running Godot: {e}")
            return None


class AutoResearchOptimizer:
    """
    AutoResearch optimization loop.

    Strategy: Random hill climbing with momentum
    - Propose random parameter perturbations
    - Accept if score improves
    - Revert if score degrades
    - Track convergence
    """

    def __init__(
        self,
        godot_controller: GodotController,
        verifier_script: Path,
        real_data_path: Path,
        output_dir: Path,
        target_score: float = 0.95,
        max_iterations: int = 50,
        patience: int = 10,
    ):
        self.godot = godot_controller
        self.verifier_script = verifier_script
        self.real_data_path = real_data_path
        self.output_dir = output_dir
        self.target_score = target_score
        self.max_iterations = max_iterations
        self.patience = patience

        self.history: List[OptimizationState] = []
        self.best_score = 0.0
        self.best_params = None
        self.convergence_count = 0

        # Parameter bounds
        self.bounds = {
            "gravity_magnitude": (8.0, 11.0),
            "object_mass": (0.1, 2.0),
            "object_friction": (0.1, 1.0),
            "object_bounce": (0.0, 1.0),
            "linear_damping": (0.0, 1.0),
            "angular_damping": (0.0, 1.0),
        }

    def _propose_params(self, current_params: Dict[str, float]) -> Dict[str, float]:
        """Propose new parameters by perturbing current ones"""
        new_params = current_params.copy()

        # Randomly select 1-2 parameters to perturb
        num_to_perturb = random.randint(1, 2)
        params_to_perturb = random.sample(list(self.bounds.keys()), num_to_perturb)

        for param in params_to_perturb:
            min_val, max_val = self.bounds[param]
            current = new_params[param]

            # Gaussian perturbation
            std = (max_val - min_val) * 0.1  # 10% of range
            new_val = current + random.gauss(0, std)

            # Clip to bounds
            new_val = max(min_val, min(max_val, new_val))
            new_params[param] = round(new_val, 3)

        return new_params

    def _verify_batch(self, batch_path: Path) -> float:
        """Run PINNs verification on a batch"""
        cmd = [
            "python3",
            str(self.verifier_script),
            "--synthetic",
            str(batch_path),
            "--real",
            str(self.real_data_path),
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode != 0:
                print(f"Verifier error: {result.stderr}")
                return 0.0

            # Parse output
            output = json.loads(result.stdout)
            return output["verification_result"]["overall_score"]

        except Exception as e:
            print(f"Error running verifier: {e}")
            return 0.0

    def _run_iteration(self, iteration: int) -> OptimizationState:
        """Run one iteration of the optimization loop"""
        print(f"\n{'=' * 60}")
        print(f"Iteration {iteration + 1}/{self.max_iterations}")
        print(f"{'=' * 60}")

        # Get current or propose new parameters
        if iteration == 0:
            params = self.godot.current_params.copy()
            print(f"Starting with default parameters")
        else:
            params = self._propose_params(self.history[-1].params)
            print(f"Proposed parameter changes:")
            for key, val in params.items():
                old_val = self.history[-1].params[key]
                if val != old_val:
                    print(f"  {key}: {old_val:.3f} → {val:.3f}")

        # Apply parameters
        self.godot.set_params(params)

        # Generate synthetic batch
        batch_id = f"iter_{iteration:03d}"
        print(f"Generating synthetic batch: {batch_id}")

        batch_path = self.godot.generate_batch(batch_id)
        if batch_path is None:
            print("Generation failed, reverting")
            return OptimizationState(
                iteration=iteration,
                params=params,
                score=0.0,
                improvement=-1.0,
                action="revert",
                timestamp=time.time(),
            )

        # Verify physics
        print(f"Running PINNs verification...")
        score = self._verify_batch(batch_path)

        # Decide action
        if iteration == 0:
            improvement = 0.0
            action = "commit"  # Always commit first iteration
            self.best_score = score
            self.best_params = params.copy()
        else:
            improvement = score - self.best_score

            if improvement > 0.001:  # Small threshold to avoid noise
                action = "commit"
                self.best_score = score
                self.best_params = params.copy()
                self.convergence_count = 0
                print(f"✓ COMMIT: Score improved by {improvement:.4f}")
            else:
                action = "revert"
                self.convergence_count += 1
                print(f"✗ REVERT: Score change {improvement:.4f} (no improvement)")

        print(f"Current score: {score:.4f} (best: {self.best_score:.4f})")
        print(f"Target: {self.target_score:.4f}")

        return OptimizationState(
            iteration=iteration,
            params=params,
            score=score,
            improvement=improvement,
            action=action,
            timestamp=time.time(),
        )

    def run(self) -> Dict:
        """Run the full optimization loop"""
        print(f"\n{'#' * 60}")
        print(f"PhysicalFish AutoResearch Optimization Loop")
        print(f"{'#' * 60}")
        print(f"Target score: {self.target_score}")
        print(f"Max iterations: {self.max_iterations}")
        print(f"Early stopping patience: {self.patience}")
        print(f"{'#' * 60}\n")

        start_time = time.time()

        for iteration in range(self.max_iterations):
            state = self._run_iteration(iteration)
            self.history.append(state)

            # Check convergence
            if self.best_score >= self.target_score:
                print(f"\n🎯 TARGET REACHED! Score: {self.best_score:.4f}")
                break

            if self.convergence_count >= self.patience:
                print(
                    f"\n⏹️  EARLY STOPPING: No improvement for {self.patience} iterations"
                )
                break

        elapsed = time.time() - start_time

        # Generate report
        report = self._generate_report(elapsed)

        print(f"\n{'=' * 60}")
        print(f"Optimization Complete")
        print(f"{'=' * 60}")
        print(f"Final score: {self.best_score:.4f}")
        print(f"Iterations: {len(self.history)}")
        print(f"Time: {elapsed:.1f}s")
        print(f"Best parameters saved to: {self.output_dir / 'best_params.json'}")

        return report

    def _generate_report(self, elapsed_time: float) -> Dict:
        """Generate final optimization report"""
        report = {
            "summary": {
                "final_score": self.best_score,
                "target_score": self.target_score,
                "iterations": len(self.history),
                "elapsed_time_seconds": elapsed_time,
                "converged": self.best_score >= self.target_score,
                "best_params": self.best_params,
            },
            "convergence_curve": [
                {
                    "iteration": s.iteration,
                    "score": s.score,
                    "best_score": max(h.score for h in self.history[: i + 1]),
                    "action": s.action,
                }
                for i, s in enumerate(self.history)
            ],
            "history": [asdict(s) for s in self.history],
        }

        # Save report
        report_path = self.output_dir / "optimization_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        # Save best params
        params_path = self.output_dir / "best_params.json"
        with open(params_path, "w") as f:
            json.dump(self.best_params, f, indent=2)

        return report


def create_mock_real_data(output_path: Path):
    """Create mock real data for testing"""
    mock_data = {
        "batch_id": "mock_real",
        "timestamp": time.time(),
        "physics_params": {
            "gravity_magnitude": 9.8,
            "object_mass": 0.5,
        },
        "frames": [],
        "events": [{"frame": 9, "event": "grab"}, {"frame": 21, "event": "release"}],
    }

    # Generate 30 frames of realistic falling object data
    for i in range(30):
        t = i / 10.0  # 10 fps

        # Simple physics: object falls, gets grabbed, thrown
        if i < 9:
            # Falling
            y = 1.0 - 0.5 * 9.8 * t**2
            v_y = -9.8 * t
            a_y = -9.8
        elif i < 21:
            # Grabbed (stationary at height)
            y = 0.8
            v_y = 0
            a_y = 0
        else:
            # Thrown upward
            t_throw = t - 2.1
            y = 0.8 + 3.0 * t_throw - 0.5 * 9.8 * t_throw**2
            v_y = 3.0 - 9.8 * t_throw
            a_y = -9.8

        frame = {
            "frame": i,
            "timestamp": t,
            "object": {
                "position": [0.0, max(0.15, y), -0.5],
                "velocity": [0.0, v_y, 0.0],
                "acceleration": [0.0, a_y, 0.0],
                "rotation": [0.0, 0.0, 0.0],
                "angular_velocity": [0.0, 0.0, 0.0],
            },
            "camera": {
                "position": [0.0, 1.6, 0.0],
                "rotation": [0.0, 0.0, 0.0],
                "fov": 70.0,
            },
        }
        mock_data["frames"].append(frame)

    with open(output_path, "w") as f:
        json.dump(mock_data, f, indent=2)

    print(f"Created mock real data: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="PhysicalFish AutoResearch Optimization Loop"
    )
    parser.add_argument(
        "--real-data",
        help="Path to real data JSON (optional, will create mock if not provided)",
    )
    parser.add_argument("--godot-project", required=True, help="Path to Godot project")
    parser.add_argument("--output", "-o", default="./output", help="Output directory")
    parser.add_argument(
        "--target-score", type=float, default=0.95, help="Target physics score"
    )
    parser.add_argument(
        "--max-iterations", type=int, default=20, help="Max optimization iterations"
    )
    parser.add_argument(
        "--patience", type=int, default=5, help="Early stopping patience"
    )

    args = parser.parse_args()

    # Setup paths
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    godot_project = Path(args.godot_project)
    verifier_script = (
        Path(__file__).parent.parent / "verification" / "verify_physics.py"
    )

    # Setup real data
    if args.real_data:
        real_data_path = Path(args.real_data)
    else:
        real_data_path = output_dir / "mock_real_data.json"
        create_mock_real_data(real_data_path)

    # Create controller and optimizer
    godot_controller = GodotController(godot_project, output_dir)
    optimizer = AutoResearchOptimizer(
        godot_controller=godot_controller,
        verifier_script=verifier_script,
        real_data_path=real_data_path,
        output_dir=output_dir,
        target_score=args.target_score,
        max_iterations=args.max_iterations,
        patience=args.patience,
    )

    # Run optimization
    report = optimizer.run()

    print(f"\nConvergence curve:")
    for point in report["convergence_curve"]:
        print(
            f"  Iter {point['iteration']:2d}: {point['score']:.4f} ({point['action']})"
        )


if __name__ == "__main__":
    main()

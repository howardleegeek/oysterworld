"""CLI entry point for PhysicalFish optimization."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from physicalfish.logging_config import get_logger, setup_logging
from physicalfish.loop.inner_loop import InnerLoop, InnerLoopConfig
from physicalfish.models import FrameData, PhysicsParams, TrajectoryData
from physicalfish.optimizer.bayesian import BayesianOptimizer
from physicalfish.simulator.pybullet_sim import PyBulletSimulator
from physicalfish.verification.physics_verifier import PhysicsVerifier

logger = get_logger("cli")


def _load_real_data(path: str) -> list[TrajectoryData]:
    """Load real trajectory data from JSON file."""
    with open(path) as f:
        data = json.load(f)

    trajectories = []
    for traj_data in data if isinstance(data, list) else [data]:
        frames = []
        for i, frame_dict in enumerate(traj_data.get("frames", [])):
            frame = FrameData(
                timestamp=frame_dict.get("timestamp", i * 0.033),
                position=np.array(frame_dict.get("position", [0.0, 0.0, 0.0])),
                velocity=np.array(frame_dict.get("velocity", [0.0, 0.0, 0.0])),
                acceleration=np.array(frame_dict.get("acceleration", [0.0, 0.0, 0.0])),
                rotation=np.array(frame_dict.get("rotation", [0.0, 0.0, 0.0])),
                angular_velocity=np.array(frame_dict.get("angular_velocity", [0.0, 0.0, 0.0])),
                frame_index=i,
            )
            frames.append(frame)

        params_dict = traj_data.get("params", {})
        params = PhysicsParams(
            gravity=params_dict.get("gravity", 9.81),
            mass=params_dict.get("mass", 0.5),
            friction=params_dict.get("friction", 0.5),
            restitution=params_dict.get("restitution", 0.8),
            linear_damping=params_dict.get("linear_damping", 0.1),
            angular_damping=params_dict.get("angular_damping", 0.1),
        )

        trajectories.append(TrajectoryData(frames=frames, params=params))

    return trajectories


def _print_convergence_curve(history: list[dict]) -> None:
    """Print ASCII convergence curve."""
    if not history:
        print("No convergence history available.")
        return

    scores = [record.get("combined_score", 0.0) for record in history]
    iterations = len(scores)

    print("\n" + "=" * 50)
    print("CONVERGENCE CURVE")
    print("=" * 50)

    # Simple ASCII chart
    chart_height = 10
    max_score = max(scores) if scores else 1.0
    min_score = min(scores) if scores else 0.0
    score_range = max_score - min_score if max_score > min_score else 1.0

    for row in range(chart_height, 0, -1):
        threshold = min_score + (score_range * (row - 1) / chart_height)
        line = f"{threshold:.2f} | "
        for score in scores:
            if score >= threshold:
                line += "*"
            else:
                line += " "
        print(line)

    print("     +" + "-" * iterations)
    print(f"     Iterations: 0 to {iterations}")
    print(f"\nFinal Score: {scores[-1]:.4f}")
    print(f"Best Score: {max(scores):.4f}")


def optimize_command(args: argparse.Namespace) -> int:
    """Run optimization command."""
    setup_logging("INFO")
    logger.info("cli_optimize_start", scenario=args.scenario, target=args.target)

    # Load real data if provided
    real_data = None
    if args.real_data:
        if not Path(args.real_data).exists():
            print(f"Error: Real data file not found: {args.real_data}", file=sys.stderr)
            return 1
        real_data = _load_real_data(args.real_data)
        logger.info("real_data_loaded", count=len(real_data))

    # Initialize components
    try:
        simulator = PyBulletSimulator()
    except Exception as e:
        print(f"Error: Failed to initialize simulator: {e}", file=sys.stderr)
        return 1

    verifier = PhysicsVerifier()
    optimizer = BayesianOptimizer(evaluator=lambda x: 0.5)  # Placeholder

    # Configure inner loop
    config = InnerLoopConfig(
        scenario=args.scenario,
        target_score=args.target,
        max_iterations=args.iterations,
    )

    # Run optimization
    inner_loop = InnerLoop(
        simulator=simulator,
        verifier=verifier,
        optimizer=optimizer,
        real_data=real_data,
        config=config,
    )

    print(f"\nStarting optimization for scenario: {args.scenario}")
    print(f"Target score: {args.target:.2f}")
    print(f"Max iterations: {args.iterations}")
    if real_data:
        print(f"Real data samples: {len(real_data)}")
    print("\nRunning optimization...")

    result = inner_loop.run()

    # Print results
    print("\n" + "=" * 50)
    print("OPTIMIZATION COMPLETE")
    print("=" * 50)
    print(f"\nBest Parameters:")
    for key, value in result.best_params.to_dict().items():
        print(f"  {key}: {value:.4f}")

    print(f"\nBest Score: {result.best_score:.4f}")
    print(f"Iterations: {result.iterations}")
    print(f"Stopped Early: {result.stopped_early}")
    if result.stop_reason:
        print(f"Stop Reason: {result.stop_reason}")

    # Print convergence curve
    _print_convergence_curve(result.history)

    # Cleanup
    simulator.close()

    return 0 if result.best_score >= args.target else 1


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="pf-optimize",
        description="PhysicalFish parameter optimization CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Optimize command
    optimize_parser = subparsers.add_parser(
        "optimize",
        help="Run parameter optimization",
    )
    optimize_parser.add_argument(
        "--scenario",
        type=str,
        default="grasp_ball",
        help="Scenario name (default: grasp_ball)",
    )
    optimize_parser.add_argument(
        "--target",
        type=float,
        default=0.90,
        help="Target verification score (default: 0.90)",
    )
    optimize_parser.add_argument(
        "--iterations",
        type=int,
        default=20,
        help="Maximum iterations (default: 20)",
    )
    optimize_parser.add_argument(
        "--real-data",
        type=str,
        help="Path to real trajectory data JSON file",
    )

    args = parser.parse_args()

    if args.command == "optimize":
        return optimize_command(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())

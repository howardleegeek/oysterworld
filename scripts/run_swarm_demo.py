#!/usr/bin/env python3
"""
Byzantine Swarm AutoResearch Demo.

3 agents optimize different physics parameters in parallel.
Cross-validate via Byzantine consensus. Best params propagate to all.
"""

import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Suppress debug logs from verifier
import structlog
structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(30),  # WARNING only
    logger_factory=structlog.PrintLoggerFactory(),
)

from physicalfish.models import FrameData, PhysicsParams, TrajectoryData
from physicalfish.verification.physics_verifier import PhysicsVerifier
from physicalfish.loop.swarm import ByzantineSwarm


def make_simulator(scenario: str = "grasp_ball"):
    """Create a simulator function that generates trajectories from params."""

    def simulate(params: dict[str, float]) -> TrajectoryData:
        g = params.get("gravity", 9.81)
        m = params.get("mass", 0.5)
        friction = params.get("friction", 0.5)
        restitution = params.get("restitution", 0.5)
        lin_damp = params.get("linear_damping", 0.1)
        ang_damp = params.get("angular_damping", 0.1)

        dt = 1.0 / 30.0
        frames = []

        # Proper stepwise physics integration (energy-conserving)
        y = 2.0
        vy = 0.0
        ground = 0.15

        for i in range(90):
            t = i * dt

            if t < 2.0:
                # Free fall with ground collision
                ay = -g
                vy += ay * dt
                vy *= (1.0 - lin_damp * dt)  # damping
                y += vy * dt

                # Ground collision
                if y <= ground:
                    y = ground
                    vy = -vy * restitution  # bounce
                    ay = 0.0
            elif t < 2.5:
                # Grasped
                ay = 0.0
                vy = 0.0
                y = max(y, ground)
            else:
                # Released
                ay = -g
                vy += ay * dt
                y += vy * dt
                y = max(y, ground)

            # Add small realistic noise
            noise_pos = np.random.normal(0, 0.001)
            noise_vel = np.random.normal(0, 0.01)

            frames.append(FrameData(
                timestamp=t,
                position=np.array([0.0, y + noise_pos, -0.5]),
                velocity=np.array([0.0, vy + noise_vel, 0.0]),
                acceleration=np.array([0.0, ay, 0.0]),
                rotation=np.zeros(3),
                angular_velocity=np.zeros(3),
                frame_index=i,
            ))

        return TrajectoryData(
            frames=frames,
            params=PhysicsParams(
                gravity=g, mass=m, friction=friction,
                restitution=restitution, linear_damping=lin_damp,
                angular_damping=ang_damp,
            ),
        )

    return simulate


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Byzantine Swarm AutoResearch Demo")
    parser.add_argument("--rounds", type=int, default=15, help="Max optimization rounds")
    parser.add_argument("--target", type=float, default=0.90, help="Target score")
    parser.add_argument("--agents", type=int, default=3, help="Number of swarm agents")
    parser.add_argument("--threshold", type=float, default=0.6, help="Byzantine threshold")
    args = parser.parse_args()

    verifier = PhysicsVerifier(gravity=9.81)
    simulator = make_simulator()

    # Start with intentionally bad params
    bad_params = {
        "gravity": 15.0,
        "mass": 3.0,
        "friction": 0.1,
        "restitution": 0.1,
        "linear_damping": 0.5,
        "angular_damping": 0.5,
    }

    swarm = ByzantineSwarm(
        simulator_fn=simulator,
        verifier=verifier,
        n_agents=args.agents,
        byzantine_threshold=args.threshold,
    )

    result = swarm.run(
        initial_params=bad_params,
        max_rounds=args.rounds,
        target_score=args.target,
        verbose=True,
    )

    # Save results
    output_path = Path("data/swarm_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n✓ Results saved to {output_path}")

    # Print convergence
    print(f"\nConvergence: ", end="")
    for score in result["convergence_curve"]:
        if score > 0.85:
            print("█", end="")
        elif score > 0.7:
            print("▓", end="")
        elif score > 0.5:
            print("▒", end="")
        else:
            print("░", end="")
    print(f" {result['final_score']:.4f}")


if __name__ == "__main__":
    main()

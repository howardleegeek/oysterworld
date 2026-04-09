#!/usr/bin/env python3
"""Create realistic synthetic robot data simulating a ball drop with sensor noise."""

import json
import numpy as np
from pathlib import Path


def create_synthetic_ball_drop_trajectory(
    episode_id: int,
    num_steps: int = 100,
    dt: float = 0.1,
    gravity: float = 9.81,
    initial_height: float = 2.0,
    initial_velocity: tuple = (0.5, 0.0, 0.3),
    restitution: float = 0.8,
    noise_std: float = 0.02,
) -> dict:
    """
    Create a realistic synthetic trajectory of a ball being dropped by a robot arm.

    This simulates what a real sensor would capture:
    - Ball starts at initial_height
    - Falls under gravity
    - Bounces with restitution
    - Has Gaussian noise on position and velocity (sensor noise)
    """
    trajectory = []

    # Initial state
    pos = np.array([0.0, initial_height, 0.0])
    vel = np.array(initial_velocity)

    for step in range(num_steps):
        t = step * dt

        # Add sensor noise to position
        pos_noisy = pos + np.random.normal(0, noise_std, 3)

        # Compute velocity from position differences (with noise)
        if step > 0:
            vel_computed = (pos_noisy - trajectory[-1]["position"]) / dt
            # Add velocity noise
            vel_noisy = vel_computed + np.random.normal(0, noise_std * 0.5, 3)
        else:
            vel_noisy = vel + np.random.normal(0, noise_std * 0.5, 3)

        # Compute acceleration from velocity differences
        if step > 0:
            acc_computed = (vel_noisy - trajectory[-1]["velocity"]) / dt
        else:
            acc_computed = np.array([0.0, -gravity, 0.0])

        # State vector: [x, y, z, vx, vy, vz] (simplified robot state)
        state = [
            float(pos_noisy[0]),
            float(pos_noisy[1]),
            float(pos_noisy[2]),
            float(vel_noisy[0]),
            float(vel_noisy[1]),
            float(vel_noisy[2]),
        ]

        trajectory.append(
            {
                "timestamp": t,
                "position": pos_noisy.tolist(),
                "velocity": vel_noisy.tolist(),
                "acceleration": acc_computed.tolist(),
                "state": state,
                "observation": {
                    "state": state,
                    "position": pos_noisy.tolist(),
                },
                "action": [0.0, 0.0, 0.0],  # No action during free fall
            }
        )

        # Physics simulation for next step
        # Update velocity with gravity
        vel[1] -= gravity * dt

        # Update position
        pos += vel * dt

        # Check for ground collision (y < 0.1)
        if pos[1] < 0.1 and vel[1] < 0:
            # Bounce with restitution
            vel[1] = -vel[1] * restitution
            pos[1] = 0.1
            # Add some horizontal friction on bounce
            vel[0] *= 0.9
            vel[2] *= 0.9

    return {
        "episode_id": episode_id,
        "num_steps": len(trajectory),
        "dt": dt,
        "gravity": gravity,
        "restitution": restitution,
        "trajectory": trajectory,
    }


def create_synthetic_dataset(num_episodes: int = 10):
    """Create a dataset of synthetic robot trajectories."""
    output_dir = Path("data/real_trajectories")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creating {num_episodes} synthetic trajectories...")

    for i in range(num_episodes):
        # Vary parameters slightly for each episode
        initial_height = np.random.uniform(1.5, 2.5)
        initial_vx = np.random.uniform(-0.5, 0.5)
        initial_vz = np.random.uniform(-0.3, 0.3)
        restitution = np.random.uniform(0.7, 0.9)
        noise_std = np.random.uniform(0.01, 0.03)

        trajectory = create_synthetic_ball_drop_trajectory(
            episode_id=i,
            num_steps=100,
            dt=0.1,
            gravity=9.81,
            initial_height=initial_height,
            initial_velocity=(initial_vx, 0.0, initial_vz),
            restitution=restitution,
            noise_std=noise_std,
        )

        # Save to JSON
        output_file = output_dir / f"episode_{i:03d}.json"
        with open(output_file, "w") as f:
            json.dump(trajectory, f, indent=2)

        print(
            f"Created episode {i}: {trajectory['num_steps']} steps, "
            f"height={initial_height:.2f}m, restitution={restitution:.2f}"
        )

    print(f"\nSuccessfully created {num_episodes} synthetic episodes in {output_dir}")
    return True


if __name__ == "__main__":
    create_synthetic_dataset(10)

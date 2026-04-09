#!/usr/bin/env python3
"""Download real robot data from HuggingFace OpenX-Embodiment dataset."""

import json
import os
from pathlib import Path


def download_robot_data():
    """Download robot data from HuggingFace."""
    print("Attempting to download from HuggingFace...")

    try:
        from datasets import load_dataset

        # Load the dataset with streaming
        ds = load_dataset(
            "jxu124/OpenX-Embodiment", "berkeley_autolab_ur5", split="train", streaming=True
        )

        # Take first 10 episodes
        episodes = []
        for i, episode in enumerate(ds):
            if i >= 10:
                break
            episodes.append(episode)
            print(f"Downloaded episode {i + 1}/10")

        # Save each episode as JSON
        output_dir = Path("data/real_trajectories")
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, episode in enumerate(episodes):
            # Extract trajectory data
            steps = episode.get("steps", [])
            trajectory_data = []

            for step in steps:
                # Extract observation and action
                observation = step.get("observation", {})
                action = step.get("action", {})

                # Get state (joint angles + end-effector position)
                state = observation.get("state", [])

                trajectory_data.append(
                    {"observation": observation, "action": action, "state": state}
                )

            # Save to JSON
            output_file = output_dir / f"episode_{i:03d}.json"
            with open(output_file, "w") as f:
                json.dump(
                    {
                        "episode_id": i,
                        "num_steps": len(trajectory_data),
                        "trajectory": trajectory_data,
                    },
                    f,
                    indent=2,
                )

            print(f"Saved episode {i} to {output_file}")

        print(f"\nSuccessfully downloaded {len(episodes)} episodes to {output_dir}")
        return True

    except Exception as e:
        print(f"Error downloading from HuggingFace: {e}")
        print("Will create synthetic data instead...")
        return False


if __name__ == "__main__":
    success = download_robot_data()
    if not success:
        exit(1)

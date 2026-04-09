"""
PhysicalFish Real Data Capture

Captures egocentric video + IMU data from mobile device.
For demo purposes, can also generate synthetic "real" data.
"""

import json
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional
import argparse


@dataclass
class IMUSample:
    """Single IMU reading"""

    timestamp: float
    accel: np.ndarray  # 3-axis acceleration (m/s²)
    gyro: np.ndarray  # 3-axis gyroscope (rad/s)

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "accel": self.accel.tolist(),
            "gyro": self.gyro.tolist(),
        }


class RealDataCapture:
    """
    Captures synchronized video + IMU from mobile device.

    For demo: can simulate realistic egocentric grasping data.
    """

    def __init__(self, output_dir: Path, fps: int = 30, imu_rate: int = 100):
        self.output_dir = output_dir
        self.fps = fps
        self.imu_rate = imu_rate
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def capture_simulated(
        self,
        duration: float = 3.0,
        scenario: str = "grasp_ball",
        noise_level: float = 0.05,
    ) -> Path:
        """
        Generate simulated real data with realistic physics.

        This simulates what a ClawGlasses device would capture
        during a simple grasping task.
        """
        capture_id = f"real_{scenario}_{int(time.time())}"

        # Generate synchronized video frames (metadata only) + IMU
        num_frames = int(duration * self.fps)
        num_imu_samples = int(duration * self.imu_rate)

        frames = []
        imu_data = []
        events = []

        # Scenario: Grasp a ball
        # Timeline: approach (0-1s) -> grasp (1s) -> lift (1-2s) -> release (2s) -> fall (2-3s)

        grab_time = 1.0
        release_time = 2.0

        # Generate IMU data (100Hz)
        for i in range(num_imu_samples):
            t = i / self.imu_rate

            # Base motion: walking + arm movement
            # Walking: periodic motion at ~1Hz
            walk_accel = np.array(
                [
                    0.1 * np.sin(2 * np.pi * t),  # lateral sway
                    9.8 + 0.3 * np.sin(4 * np.pi * t),  # vertical bounce
                    0.2 * np.cos(2 * np.pi * t),  # forward/back
                ]
            )

            # Arm movement for grasping
            if t < grab_time:
                # Approaching - arm extending forward
                arm_accel = np.array([0, 0, 0.5 * (grab_time - t)])
            elif t < grab_time + 0.2:
                # Grasp moment - small jerk
                arm_accel = np.array([0, 2.0, 0])
                if i == int(grab_time * self.imu_rate):
                    events.append({"timestamp": t, "event": "grab"})
            elif t < release_time:
                # Lifting - upward acceleration
                arm_accel = np.array([0, 1.5, 0])
            elif t < release_time + 0.2:
                # Release - upward throw
                arm_accel = np.array([0, 3.0, 0.5])
                if i == int(release_time * self.imu_rate):
                    events.append({"timestamp": t, "event": "release"})
            else:
                # Settling
                arm_accel = np.array([0, 0, 0])

            # Combine + add noise
            accel = walk_accel + arm_accel
            accel += np.random.normal(0, noise_level, 3)

            # Gyro: rotation during arm movement
            gyro = np.array(
                [
                    0.1 * arm_accel[2],  # pitch
                    0.0,  # yaw
                    0.1 * arm_accel[0],  # roll
                ]
            )
            gyro += np.random.normal(0, noise_level * 0.5, 3)

            imu_sample = IMUSample(timestamp=t, accel=accel, gyro=gyro)
            imu_data.append(imu_sample.to_dict())

        # Generate video frame metadata (30Hz)
        for i in range(num_frames):
            t = i / self.fps

            # Object position (from egocentric perspective)
            # Object moves from center to grasped to released
            if t < grab_time:
                # Object on table, approaching
                obj_pos = [0.0, -0.3, -0.5 - 0.2 * (grab_time - t)]
            elif t < release_time:
                # Object grasped, lifting
                lift_progress = (t - grab_time) / (release_time - grab_time)
                obj_pos = [0.0, -0.3 + 0.4 * lift_progress, -0.5]
            else:
                # Object released, falling
                fall_time = t - release_time
                obj_pos = [0.0, 0.1 - 0.5 * 9.8 * fall_time**2, -0.5]

            frame = {
                "frame": i,
                "timestamp": t,
                "object": {
                    "position": obj_pos,
                    "visible": obj_pos[1] > -0.5,  # Visible if not below table
                },
                "camera": {
                    "position": [0, 1.6, 0],  # Head position
                    "rotation": [0, 0, 0],  # Looking forward
                    "fov": 70,
                },
            }
            frames.append(frame)

        # Build capture data
        capture_data = {
            "capture_id": capture_id,
            "timestamp": time.time(),
            "scenario": scenario,
            "duration": duration,
            "fps": self.fps,
            "imu_rate": self.imu_rate,
            "device": "ClawGlasses-Simulated",
            "frames": frames,
            "imu_data": imu_data,
            "events": events,
        }

        # Save
        output_path = self.output_dir / f"{capture_id}.json"
        with open(output_path, "w") as f:
            json.dump(capture_data, f, indent=2)

        print(f"Simulated capture saved: {output_path}")
        print(f"  Duration: {duration}s")
        print(f"  Frames: {num_frames}")
        print(f"  IMU samples: {num_imu_samples}")
        print(f"  Events: {len(events)}")

        return output_path

    def convert_to_verification_format(self, capture_path: Path) -> Path:
        """
        Convert capture data to format expected by verification engine.

        The verification engine expects trajectory data with physics parameters.
        """
        with open(capture_path, "r") as f:
            capture = json.load(f)

        # Convert to trajectory format
        trajectory_data = {
            "batch_id": capture["capture_id"],
            "timestamp": capture["timestamp"],
            "physics_params": {
                "gravity_magnitude": 9.8,
                "object_mass": 0.5,  # Assumed
            },
            "frames": [],
            "events": [],
        }

        # Convert frames
        for frame in capture["frames"]:
            # Estimate velocity and acceleration from IMU
            frame_idx = frame["frame"]
            imu_idx = int(frame["timestamp"] * capture["imu_rate"])

            if imu_idx < len(capture["imu_data"]):
                imu = capture["imu_data"][imu_idx]
                accel = imu["accel"]
            else:
                accel = [0, -9.8, 0]  # Default to gravity

            trajectory_frame = {
                "frame": frame_idx,
                "timestamp": frame["timestamp"],
                "object": {
                    "position": frame["object"]["position"],
                    "velocity": [
                        0,
                        0,
                        0,
                    ],  # Would need to calculate from position deltas
                    "acceleration": accel,
                    "rotation": [0, 0, 0],
                    "angular_velocity": [0, 0, 0],
                },
                "camera": {
                    "position": frame["camera"]["position"],
                    "rotation": frame["camera"]["rotation"],
                    "fov": frame["camera"]["fov"],
                },
            }
            trajectory_data["frames"].append(trajectory_frame)

        # Convert events
        for event in capture["events"]:
            event_frame = int(event["timestamp"] * capture["fps"])
            trajectory_data["events"].append(
                {"frame": event_frame, "event": event["event"]}
            )

        # Save
        output_path = self.output_dir / f"{capture['capture_id']}_trajectory.json"
        with open(output_path, "w") as f:
            json.dump(trajectory_data, f, indent=2)

        return output_path


def main():
    parser = argparse.ArgumentParser(description="PhysicalFish Real Data Capture")
    parser.add_argument(
        "--output", "-o", default="./data/real", help="Output directory"
    )
    parser.add_argument(
        "--duration", type=float, default=3.0, help="Capture duration (seconds)"
    )
    parser.add_argument("--scenario", default="grasp_ball", help="Scenario name")
    parser.add_argument(
        "--convert", help="Convert existing capture to trajectory format"
    )

    args = parser.parse_args()

    output_dir = Path(args.output)
    capture = RealDataCapture(output_dir)

    if args.convert:
        # Convert existing capture
        capture_path = Path(args.convert)
        trajectory_path = capture.convert_to_verification_format(capture_path)
        print(f"Converted to: {trajectory_path}")
    else:
        # Generate new simulated capture
        capture_path = capture.capture_simulated(
            duration=args.duration, scenario=args.scenario
        )

        # Also convert to trajectory format for verification
        trajectory_path = capture.convert_to_verification_format(capture_path)
        print(f"\nTrajectory format: {trajectory_path}")


if __name__ == "__main__":
    main()

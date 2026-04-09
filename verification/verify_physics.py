"""
PhysicalFish PINNs Verification Engine
Validates synthetic data against physical laws using real data as ground truth.

Based on PINNs ECG Demo architecture.
"""

import json
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import argparse


@dataclass
class FrameData:
    """Single frame of motion data"""

    frame: int
    timestamp: float
    position: np.ndarray  # [x, y, z]
    velocity: np.ndarray  # [x, y, z]
    acceleration: np.ndarray  # [x, y, z]
    rotation: np.ndarray  # [x, y, z]
    angular_velocity: np.ndarray  # [x, y, z]


@dataclass
class VerificationResult:
    """Result of physics verification"""

    kinematic_score: float  # 0-1, motion consistency
    dynamic_score: float  # 0-1, force/mass consistency
    temporal_score: float  # 0-1, timing consistency
    overall_score: float  # 0-1, weighted average
    details: Dict


class PhysicsVerifier:
    """
    PINNs-inspired physics verification engine.
    Checks if synthetic data obeys physical laws.
    """

    def __init__(self, gravity: float = 9.8):
        self.gravity = gravity
        self.epsilon = 1e-6  # Small value to prevent division by zero

    def load_trajectory(self, data: Dict) -> List[FrameData]:
        """Convert JSON data to FrameData objects"""
        frames = []
        for frame_dict in data.get("frames", []):
            frame = FrameData(
                frame=frame_dict["frame"],
                timestamp=frame_dict["timestamp"],
                position=np.array(frame_dict["object"]["position"]),
                velocity=np.array(frame_dict["object"]["velocity"]),
                acceleration=np.array(frame_dict["object"]["acceleration"]),
                rotation=np.array(frame_dict["object"]["rotation"]),
                angular_velocity=np.array(frame_dict["object"]["angular_velocity"]),
            )
            frames.append(frame)
        return frames

    def verify_kinematic_consistency(
        self, frames: List[FrameData]
    ) -> Tuple[float, Dict]:
        """
        Verify that position, velocity, and acceleration are consistent.

        Physics law: v = dx/dt, a = dv/dt
        Check: numerical derivatives should match reported values
        """
        if len(frames) < 3:
            return 0.0, {"error": "Insufficient frames"}

        errors = []
        dt = frames[1].timestamp - frames[0].timestamp

        for i in range(1, len(frames) - 1):
            # Numerical velocity from positions
            v_numerical = (frames[i + 1].position - frames[i - 1].position) / (2 * dt)

            # Numerical acceleration from velocities
            a_numerical = (frames[i + 1].velocity - frames[i - 1].velocity) / (2 * dt)

            # Compare with reported values
            v_error = np.linalg.norm(v_numerical - frames[i].velocity)
            a_error = np.linalg.norm(a_numerical - frames[i].acceleration)

            errors.append(
                {
                    "frame": frames[i].frame,
                    "velocity_error": float(v_error),
                    "acceleration_error": float(a_error),
                }
            )

        # Calculate score (lower error = higher score)
        avg_v_error = np.mean([e["velocity_error"] for e in errors])
        avg_a_error = np.mean([e["acceleration_error"] for e in errors])

        # Score: 1.0 when error is 0, approaches 0 as error increases
        v_score = np.exp(-avg_v_error)
        a_score = np.exp(-avg_a_error)

        kinematic_score = (v_score + a_score) / 2

        return kinematic_score, {
            "velocity_score": float(v_score),
            "acceleration_score": float(a_score),
            "avg_velocity_error": float(avg_v_error),
            "avg_acceleration_error": float(avg_a_error),
            "frame_errors": errors[:5],  # First 5 for debugging
        }

    def verify_dynamic_consistency(
        self, frames: List[FrameData], mass: float
    ) -> Tuple[float, Dict]:
        """
        Verify Newton's 2nd law: F = ma

        For free-fall: F = mg, so a should equal g (downward)
        For contact: acceleration should reflect contact forces
        """
        if len(frames) < 2:
            return 0.0, {"error": "Insufficient frames"}

        gravity_vector = np.array([0, -self.gravity, 0])

        deviations = []
        free_fall_frames = []

        for i, frame in enumerate(frames):
            # Expected acceleration (gravity + any contact forces)
            # For simplicity, assume free-fall when y-velocity is negative and no contact
            is_falling = frame.velocity[1] < -0.1

            if is_falling:
                free_fall_frames.append(i)
                # In free-fall, acceleration should equal gravity
                expected_a = gravity_vector
                deviation = np.linalg.norm(frame.acceleration - expected_a)
                deviations.append(deviation)

        if not deviations:
            # No free-fall detected, check general consistency
            # Acceleration magnitude should be reasonable
            acc_magnitudes = [np.linalg.norm(f.acceleration) for f in frames]
            avg_acc = np.mean(acc_magnitudes)

            # Score based on whether accelerations are physically plausible
            # Typical range: 0 to 5*g (including contact forces)
            if avg_acc < 5 * self.gravity:
                score = 1.0 - (avg_acc / (5 * self.gravity))
            else:
                score = 0.0

            return score, {
                "avg_acceleration": float(avg_acc),
                "note": "No free-fall detected, using magnitude check",
            }

        # Score based on free-fall consistency
        avg_deviation = np.mean(deviations)
        score = np.exp(-avg_deviation / self.gravity)

        return score, {
            "free_fall_frames": len(free_fall_frames),
            "avg_deviation_from_gravity": float(avg_deviation),
            "expected_gravity": self.gravity,
        }

    def verify_temporal_consistency(
        self, frames: List[FrameData], events: List[Dict]
    ) -> Tuple[float, Dict]:
        """
        Verify timing of events (grab, release) is physically plausible.

        Checks:
        - Grab should happen when object is within reach
        - Release should result in predictable trajectory
        """
        if not events:
            return 0.5, {"note": "No events to verify"}

        scores = []
        details = []

        for event in events:
            frame_idx = event.get("frame", 0)
            event_type = event.get("event", "")

            if frame_idx >= len(frames):
                continue

            frame = frames[frame_idx]

            if event_type == "grab":
                # Check if object is at a reachable height
                height = frame.position[1]
                if 0.3 < height < 2.0:  # Reasonable grab height
                    scores.append(1.0)
                else:
                    scores.append(0.3)

                details.append(
                    {
                        "event": "grab",
                        "height": float(height),
                        "reachable": 0.3 < height < 2.0,
                    }
                )

            elif event_type == "release":
                # Check if object has upward velocity (plausible throw)
                v_y = frame.velocity[1]
                if v_y > 0:  # Moving upward at release
                    scores.append(1.0)
                elif v_y > -0.5:  # Near stationary
                    scores.append(0.7)
                else:
                    scores.append(0.4)

                details.append(
                    {
                        "event": "release",
                        "velocity_y": float(v_y),
                        "plausible": v_y > -0.5,
                    }
                )

        temporal_score = np.mean(scores) if scores else 0.5

        return temporal_score, {"event_scores": details, "num_events": len(events)}

    def verify_batch(
        self, synthetic_data: Dict, real_data: Optional[Dict] = None
    ) -> VerificationResult:
        """
        Verify a batch of synthetic data.

        If real_data is provided, also compare distributions.
        """
        frames = self.load_trajectory(synthetic_data)
        events = synthetic_data.get("events", [])
        physics_params = synthetic_data.get("physics_params", {})
        mass = physics_params.get("object_mass", 0.5)

        # Run verification checks
        kinematic_score, kinematic_details = self.verify_kinematic_consistency(frames)
        dynamic_score, dynamic_details = self.verify_dynamic_consistency(frames, mass)
        temporal_score, temporal_details = self.verify_temporal_consistency(
            frames, events
        )

        # Weighted overall score
        weights = {"kinematic": 0.4, "dynamic": 0.4, "temporal": 0.2}

        overall_score = (
            weights["kinematic"] * kinematic_score
            + weights["dynamic"] * dynamic_score
            + weights["temporal"] * temporal_score
        )

        return VerificationResult(
            kinematic_score=kinematic_score,
            dynamic_score=dynamic_score,
            temporal_score=temporal_score,
            overall_score=overall_score,
            details={
                "kinematic": kinematic_details,
                "dynamic": dynamic_details,
                "temporal": temporal_details,
                "physics_params": physics_params,
                "num_frames": len(frames),
                "duration": frames[-1].timestamp - frames[0].timestamp if frames else 0,
            },
        )

    def compare_to_real(self, synthetic_data: Dict, real_data: Dict) -> Dict:
        """
        Compare synthetic data distribution to real data.

        Returns similarity metrics.
        """
        synthetic_frames = self.load_trajectory(synthetic_data)
        real_frames = self.load_trajectory(real_data)

        # Extract features for comparison
        def extract_features(frames: List[FrameData]) -> Dict[str, List[float]]:
            return {
                "velocity_magnitude": [np.linalg.norm(f.velocity) for f in frames],
                "acceleration_magnitude": [
                    np.linalg.norm(f.acceleration) for f in frames
                ],
                "height": [f.position[1] for f in frames],
            }

        synthetic_features = extract_features(synthetic_frames)
        real_features = extract_features(real_frames)

        # Calculate distribution similarity (simplified)
        similarities = {}
        for key in synthetic_features:
            syn_mean = np.mean(synthetic_features[key])
            real_mean = np.mean(real_features[key])
            syn_std = np.std(synthetic_features[key])
            real_std = np.std(real_features[key])

            # Normalized difference
            mean_diff = abs(syn_mean - real_mean) / (abs(real_mean) + self.epsilon)
            std_diff = abs(syn_std - real_std) / (abs(real_std) + self.epsilon)

            # Similarity score (1.0 = identical, 0.0 = completely different)
            similarity = max(0, 1.0 - (mean_diff + std_diff) / 2)
            similarities[key] = float(similarity)

        return {
            "feature_similarities": similarities,
            "overall_similarity": float(np.mean(list(similarities.values()))),
            "synthetic_stats": {
                k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                for k, v in synthetic_features.items()
            },
            "real_stats": {
                k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                for k, v in real_features.items()
            },
        }


def main():
    parser = argparse.ArgumentParser(
        description="PhysicalFish PINNs Verification Engine"
    )
    parser.add_argument(
        "--synthetic", required=True, help="Path to synthetic data JSON"
    )
    parser.add_argument("--real", help="Path to real data JSON (optional)")
    parser.add_argument("--output", "-o", help="Output JSON path")
    parser.add_argument("--gravity", type=float, default=9.8, help="Gravity constant")

    args = parser.parse_args()

    # Load data
    with open(args.synthetic, "r") as f:
        synthetic_data = json.load(f)

    real_data = None
    if args.real:
        with open(args.real, "r") as f:
            real_data = json.load(f)

    # Run verification
    verifier = PhysicsVerifier(gravity=args.gravity)
    result = verifier.verify_batch(synthetic_data, real_data)

    # Build output
    output = {
        "verification_result": {
            "kinematic_score": result.kinematic_score,
            "dynamic_score": result.dynamic_score,
            "temporal_score": result.temporal_score,
            "overall_score": result.overall_score,
            "passed": result.overall_score > 0.7,
        },
        "details": result.details,
    }

    # Compare to real if provided
    if real_data:
        comparison = verifier.compare_to_real(synthetic_data, real_data)
        output["real_comparison"] = comparison

    # Save or print
    if args.output:
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Verification result saved to: {args.output}")
    else:
        print(json.dumps(output, indent=2))

    # Print summary
    print(f"\n{'=' * 50}")
    print(f"PINNs Verification Summary")
    print(f"{'=' * 50}")
    print(f"Kinematic Score:  {result.kinematic_score:.3f}")
    print(f"Dynamic Score:    {result.dynamic_score:.3f}")
    print(f"Temporal Score:   {result.temporal_score:.3f}")
    print(f"Overall Score:    {result.overall_score:.3f}")
    print(f"Status:           {'PASS' if result.overall_score > 0.7 else 'FAIL'}")
    if real_data:
        print(
            f"Real Similarity:  {output['real_comparison']['overall_similarity']:.3f}"
        )
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()

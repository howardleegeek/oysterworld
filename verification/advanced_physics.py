"""
Advanced Physics Verification Engine
Based on Raissi et al. 2019 PINNs framework with multi-scale constraints
"""

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class FrameData:
    """Complete physical state at a timestep"""

    timestamp: float
    position: np.ndarray  # [x, y, z]
    velocity: np.ndarray  # [vx, vy, vz]
    acceleration: np.ndarray  # [ax, ay, az]
    rotation: np.ndarray  # [rx, ry, rz]
    angular_velocity: np.ndarray  # [wx, wy, wz]
    mass: float

    @property
    def speed(self) -> float:
        return np.linalg.norm(self.velocity)

    @property
    def height(self) -> float:
        return self.position[1]  # y-axis


class AdvancedPhysicsVerifier:
    """
    Production-grade physics verification with 6 constraints:
    1. Kinematic consistency (v = dx/dt)
    2. Dynamic consistency (F = ma)
    3. Energy conservation (E ≈ const)
    4. Momentum conservation (p = mv)
    5. Angular momentum (L = r × p)
    6. Collision response (restitution)
    """

    def __init__(self, gravity: float = 9.8, tolerance: float = 0.05):
        self.g = gravity
        self.tolerance = tolerance
        self.weights = {
            "kinematic": 0.20,
            "dynamic": 0.20,
            "energy": 0.15,
            "momentum": 0.15,
            "angular": 0.15,
            "collision": 0.15,
        }

    def verify_trajectory(
        self, frames: List[FrameData], expected_restitution: float = 0.8
    ) -> Dict:
        """
        Comprehensive physics verification

        Returns detailed scores for each constraint
        """
        results = {
            "kinematic": self._check_kinematic(frames),
            "dynamic": self._check_dynamic(frames),
            "energy": self._check_energy(frames),
            "momentum": self._check_momentum(frames),
            "angular": self._check_angular_momentum(frames),
            "collision": self._check_collision(frames, expected_restitution),
        }

        # Weighted overall score
        overall = sum(results[key]["score"] * self.weights[key] for key in results)

        return {
            "overall_score": overall,
            "passed": overall > 0.7,
            "details": results,
            "weights": self.weights,
        }

    def _check_kinematic(self, frames: List[FrameData]) -> Dict:
        """
        Check kinematic consistency: numerical derivatives match reported values

        v_numerical = (x_{t+1} - x_{t-1}) / 2dt
        a_numerical = (v_{t+1} - v_{t-1}) / 2dt
        """
        if len(frames) < 3:
            return {"score": 0.5, "error": "Insufficient frames"}

        dt = frames[1].timestamp - frames[0].timestamp

        v_errors = []
        a_errors = []

        for i in range(1, len(frames) - 1):
            # Numerical velocity from positions
            v_num = (frames[i + 1].position - frames[i - 1].position) / (2 * dt)
            v_err = np.linalg.norm(v_num - frames[i].velocity)
            v_errors.append(v_err)

            # Numerical acceleration from velocities
            a_num = (frames[i + 1].velocity - frames[i - 1].velocity) / (2 * dt)
            a_err = np.linalg.norm(a_num - frames[i].acceleration)
            a_errors.append(a_err)

        avg_v_error = np.mean(v_errors)
        avg_a_error = np.mean(a_errors)

        # Score: exponential decay with error
        v_score = np.exp(-avg_v_error / 2.0)
        a_score = np.exp(-avg_a_error / 5.0)

        return {
            "score": (v_score + a_score) / 2,
            "velocity_error": float(avg_v_error),
            "acceleration_error": float(avg_a_error),
            "velocity_score": float(v_score),
            "acceleration_score": float(a_score),
        }

    def _check_dynamic(self, frames: List[FrameData]) -> Dict:
        """
        Check Newton's 2nd law: F = ma

        For free-fall: a should equal g (downward)
        For contact: acceleration should reflect contact forces
        """
        gravity_vec = np.array([0, -self.g, 0])

        deviations = []
        free_fall_frames = 0

        for i, frame in enumerate(frames):
            # Detect free-fall (negative y-velocity, no contact)
            is_falling = frame.velocity[1] < -0.1

            if is_falling:
                free_fall_frames += 1
                # In free-fall, acceleration should equal gravity
                expected_a = gravity_vec
                deviation = np.linalg.norm(frame.acceleration - expected_a)
                deviations.append(deviation)

        if not deviations:
            # No free-fall, check general plausibility
            acc_magnitudes = [np.linalg.norm(f.acceleration) for f in frames]
            avg_acc = np.mean(acc_magnitudes)

            # Should be within reasonable range (0 to 5g)
            if avg_acc < 5 * self.g:
                score = 1.0 - (avg_acc / (5 * self.g)) * 0.5
            else:
                score = 0.3

            return {
                "score": score,
                "avg_acceleration": float(avg_acc),
                "note": "No free-fall detected",
            }

        avg_deviation = np.mean(deviations)
        score = np.exp(-avg_deviation / self.g)

        return {
            "score": float(score),
            "free_fall_frames": free_fall_frames,
            "avg_deviation_from_gravity": float(avg_deviation),
            "expected_gravity": self.g,
        }

    def _check_energy(self, frames: List[FrameData]) -> Dict:
        """
        Check energy conservation: E = KE + PE ≈ constant

        KE = 0.5 * m * v^2
        PE = m * g * h

        Note: Real systems lose energy to friction/air resistance
        We allow for gradual decay, not sudden jumps
        """
        energies = []

        for frame in frames:
            m = frame.mass
            v = frame.speed
            h = max(0, frame.height)  # Height above ground

            ke = 0.5 * m * v**2
            pe = m * self.g * h
            total_e = ke + pe

            energies.append(total_e)

        if len(energies) < 2:
            return {"score": 0.5, "note": "Insufficient data"}

        # Check for energy conservation (allowing gradual decay)
        energy_changes = np.diff(energies)
        max_change = np.max(np.abs(energy_changes))
        mean_energy = np.mean(energies)

        # Score based on relative energy stability
        if mean_energy > 0:
            relative_variation = np.std(energies) / mean_energy
            # Allow up to 20% variation (realistic for non-ideal systems)
            score = max(0, 1 - relative_variation / 0.2)
        else:
            score = 0.5

        return {
            "score": float(score),
            "mean_energy": float(mean_energy),
            "energy_std": float(np.std(energies)),
            "max_instantaneous_change": float(max_change),
            "num_frames": len(frames),
        }

    def _check_momentum(self, frames: List[FrameData]) -> Dict:
        """
        Check momentum: p = m * v

        For isolated systems, momentum should be conserved
        For systems with external forces (gravity), momentum changes predictably
        """
        momenta = []

        for frame in frames:
            p = frame.mass * frame.velocity
            momenta.append(np.linalg.norm(p))

        if len(momenta) < 2:
            return {"score": 0.5}

        # Check momentum stability
        # In free-fall, momentum magnitude changes (speed increases)
        # But changes should be smooth, not abrupt
        momentum_changes = np.diff(momenta)
        max_change = np.max(np.abs(momentum_changes))

        # Smooth changes are OK, abrupt changes indicate problems
        if max_change < 5.0:  # Threshold for abrupt change
            score = 1.0
        elif max_change < 10.0:
            score = 0.7
        else:
            score = 0.3

        return {
            "score": float(score),
            "mean_momentum": float(np.mean(momenta)),
            "max_momentum_change": float(max_change),
        }

    def _check_angular_momentum(self, frames: List[FrameData]) -> Dict:
        """
        Check angular momentum: L = r × p

        For rotationally symmetric systems, L should be conserved
        """
        angular_momenta = []

        for frame in frames:
            r = frame.position
            p = frame.mass * frame.velocity
            L = np.cross(r, p)  # Simplified for 3D
            angular_momenta.append(np.linalg.norm(L))

        if len(angular_momenta) < 2:
            return {"score": 0.5}

        # Check stability
        variation = np.std(angular_momenta) / (np.mean(angular_momenta) + 1e-6)
        score = max(0, 1 - variation)

        return {
            "score": float(score),
            "mean_L": float(np.mean(angular_momenta)),
            "variation": float(variation),
        }

    def _check_collision(
        self, frames: List[FrameData], expected_restitution: float
    ) -> Dict:
        """
        Check collision response matches expected restitution

        e = |v_after| / |v_before|
        """
        collisions = self._detect_collisions(frames)

        if not collisions:
            return {"score": 1.0, "num_collisions": 0, "note": "No collisions"}

        scores = []
        for collision in collisions:
            v_before = collision["v_before"]
            v_after = collision["v_after"]

            if np.linalg.norm(v_before) > 0.1:
                e_actual = np.linalg.norm(v_after) / np.linalg.norm(v_before)
                error = abs(e_actual - expected_restitution)
                # Score decreases with error
                score = max(0, 1 - error / 0.5)
                scores.append(score)

        return {
            "score": np.mean(scores) if scores else 1.0,
            "num_collisions": len(collisions),
            "restitution_scores": scores,
        }

    def _detect_collisions(self, frames: List[FrameData]) -> List[Dict]:
        """Detect collision events (sudden velocity changes)"""
        collisions = []

        for i in range(1, len(frames)):
            v_prev = frames[i - 1].velocity
            v_curr = frames[i].velocity

            # Sudden change in velocity direction or magnitude
            speed_change = np.linalg.norm(v_curr - v_prev)

            if speed_change > 2.0:  # Threshold for collision
                collisions.append(
                    {
                        "frame": i,
                        "v_before": v_prev,
                        "v_after": v_curr,
                        "speed_change": speed_change,
                    }
                )

        return collisions


# Convenience function for integration
def verify_synthetic_data(trajectory_data: Dict, gravity: float = 9.8) -> Dict:
    """
    Quick verification function for synthetic data

    Args:
        trajectory_data: Dict with 'frames' containing position, velocity, etc.
        gravity: Expected gravity constant

    Returns:
        Verification results with overall score and detailed checks
    """
    verifier = AdvancedPhysicsVerifier(gravity=gravity)

    # Convert dict to FrameData objects
    frames = []
    for frame_dict in trajectory_data.get("frames", []):
        obj = frame_dict.get("object", {})
        frames.append(
            FrameData(
                timestamp=frame_dict.get("timestamp", 0),
                position=np.array(obj.get("position", [0, 0, 0])),
                velocity=np.array(obj.get("velocity", [0, 0, 0])),
                acceleration=np.array(obj.get("acceleration", [0, 0, 0])),
                rotation=np.array(obj.get("rotation", [0, 0, 0])),
                angular_velocity=np.array(obj.get("angular_velocity", [0, 0, 0])),
                mass=trajectory_data.get("physics_params", {}).get("object_mass", 0.5),
            )
        )

    return verifier.verify_trajectory(frames)


if __name__ == "__main__":
    # Test with mock data
    print("Advanced Physics Verifier — Test")
    print("=" * 50)

    # Create simple falling object trajectory
    frames = []
    for i in range(30):
        t = i * 0.1
        # Free fall: y = 1.0 - 0.5 * g * t^2
        y = max(0.15, 1.0 - 0.5 * 9.8 * t**2)
        v_y = -9.8 * t if y > 0.15 else 0
        a_y = -9.8 if y > 0.15 else 0

        frames.append(
            FrameData(
                timestamp=t,
                position=np.array([0, y, -0.5]),
                velocity=np.array([0, v_y, 0]),
                acceleration=np.array([0, a_y, 0]),
                rotation=np.array([0, 0, 0]),
                angular_velocity=np.array([0, 0, 0]),
                mass=0.5,
            )
        )

    verifier = AdvancedPhysicsVerifier()
    result = verifier.verify_trajectory(frames)

    print(f"\nOverall Score: {result['overall_score']:.4f}")
    print(f"Passed: {result['passed']}")
    print("\nDetailed Scores:")
    for check, data in result["details"].items():
        print(f"  {check:12s}: {data['score']:.4f}")

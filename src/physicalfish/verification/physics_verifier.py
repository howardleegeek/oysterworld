"""6-constraint physics verification engine."""

import numpy as np

from physicalfish.exceptions import InsufficientDataError
from physicalfish.logging_config import get_logger
from physicalfish.models import FrameData, TrajectoryData, VerificationResult


class PhysicsVerifier:
    """6-constraint physics verification engine."""

    WEIGHTS = {
        "kinematic": 0.25,
        "dynamic": 0.30,
        "energy": 0.05,
        "momentum": 0.15,
        "angular": 0.10,
        "collision": 0.15,
    }

    def __init__(self, gravity: float = 9.81, tolerance: float = 0.05):
        self.g = gravity
        self.tolerance = tolerance
        self.logger = get_logger("verifier")

    def verify(self, trajectory: TrajectoryData) -> VerificationResult:
        """Verify a complete trajectory against 6 physics constraints."""
        if trajectory.num_frames < 3:
            raise InsufficientDataError(f"Need >= 3 frames, got {trajectory.num_frames}")

        mass = trajectory.params.mass  # mass comes from params, not per-frame

        scores = {
            "kinematic": self._check_kinematic(trajectory.frames),
            "dynamic": self._check_dynamic(trajectory.frames),
            "energy": self._check_energy(trajectory.frames, mass),
            "momentum": self._check_momentum(trajectory.frames, mass),
            "angular": self._check_angular_momentum(trajectory.frames, mass),
            "collision": self._check_collision(trajectory.frames),
        }

        overall = sum(scores[k]["score"] * self.WEIGHTS[k] for k in scores)

        self.logger.debug(
            "verification_complete",
            overall_score=overall,
            constraint_scores={k: v["score"] for k, v in scores.items()},
        )

        return VerificationResult(
            overall_score=overall,
            passed=overall > 0.7,
            constraint_scores={k: v["score"] for k, v in scores.items()},
            details=scores,
        )

    def _check_kinematic(self, frames: list[FrameData]) -> dict:
        """
        Check kinematic consistency: numerical derivatives match reported values.

        v_numerical = (x_{t+1} - x_{t-1}) / 2dt
        a_numerical = (v_{t+1} - v_{t-1}) / 2dt
        """
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

        self.logger.debug(
            "kinematic_check",
            velocity_error=float(avg_v_error),
            acceleration_error=float(avg_a_error),
        )

        return {
            "score": (v_score + a_score) / 2,
            "velocity_error": float(avg_v_error),
            "acceleration_error": float(avg_a_error),
            "velocity_score": float(v_score),
            "acceleration_score": float(a_score),
        }

    def _check_dynamic(self, frames: list[FrameData]) -> dict:
        """
        Check Newton's 2nd law: F = ma.

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

            self.logger.debug("dynamic_check_no_freefall", avg_acceleration=float(avg_acc))

            return {
                "score": score,
                "avg_acceleration": float(avg_acc),
                "note": "No free-fall detected",
            }

        avg_deviation = np.mean(deviations)
        score = np.exp(-avg_deviation / self.g)

        self.logger.debug(
            "dynamic_check_freefall",
            free_fall_frames=free_fall_frames,
            avg_deviation=float(avg_deviation),
        )

        return {
            "score": float(score),
            "free_fall_frames": free_fall_frames,
            "avg_deviation_from_gravity": float(avg_deviation),
            "expected_gravity": self.g,
        }

    def _check_energy(self, frames: list[FrameData], mass: float) -> dict:
        """
        Check energy conservation: E = KE + PE ≈ constant.

        KE = 0.5 * m * v^2
        PE = m * g * h

        Note: Real systems lose energy to friction/air resistance.
        We allow for gradual decay, not sudden jumps.
        """
        energies = []

        for frame in frames:
            m = mass
            v = frame.speed
            h = max(0, frame.height)  # Height above ground

            ke = 0.5 * m * v**2
            pe = m * self.g * h
            total_e = ke + pe

            energies.append(total_e)

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

        self.logger.debug(
            "energy_check",
            mean_energy=float(mean_energy),
            energy_std=float(np.std(energies)),
        )

        return {
            "score": float(score),
            "mean_energy": float(mean_energy),
            "energy_std": float(np.std(energies)),
            "max_instantaneous_change": float(max_change),
            "num_frames": len(frames),
        }

    def _check_momentum(self, frames: list[FrameData], mass: float) -> dict:
        """
        Check momentum: p = m * v.

        For isolated systems, momentum should be conserved.
        For systems with external forces (gravity), momentum changes predictably.
        """
        momenta = []

        for frame in frames:
            p = mass * frame.velocity
            momenta.append(np.linalg.norm(p))

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

        self.logger.debug(
            "momentum_check",
            mean_momentum=float(np.mean(momenta)),
            max_change=float(max_change),
        )

        return {
            "score": float(score),
            "mean_momentum": float(np.mean(momenta)),
            "max_momentum_change": float(max_change),
        }

    def _check_angular_momentum(self, frames: list[FrameData], mass: float) -> dict:
        """
        Check angular momentum: L = r × p.

        For rotationally symmetric systems, L should be conserved.
        """
        angular_momenta = []

        for frame in frames:
            r = frame.position
            p = mass * frame.velocity
            L = np.cross(r, p)  # Simplified for 3D
            angular_momenta.append(np.linalg.norm(L))

        # Check stability
        variation = np.std(angular_momenta) / (np.mean(angular_momenta) + 1e-6)
        score = max(0, 1 - variation)

        self.logger.debug(
            "angular_momentum_check",
            mean_L=float(np.mean(angular_momenta)),
            variation=float(variation),
        )

        return {
            "score": float(score),
            "mean_L": float(np.mean(angular_momenta)),
            "variation": float(variation),
        }

    def _check_collision(self, frames: list[FrameData]) -> dict:
        """
        Check collision response matches expected restitution.

        e = |v_after| / |v_before|
        """
        collisions = self._detect_collisions(frames)

        if not collisions:
            self.logger.debug("collision_check", num_collisions=0, note="No collisions")
            return {"score": 1.0, "num_collisions": 0, "note": "No collisions"}

        scores = []
        for collision in collisions:
            v_before = collision["v_before"]
            v_after = collision["v_after"]

            if np.linalg.norm(v_before) > 0.1:
                e_actual = np.linalg.norm(v_after) / np.linalg.norm(v_before)
                # Use default restitution of 0.8 if not specified
                expected_restitution = 0.8
                error = abs(e_actual - expected_restitution)
                # Score decreases with error
                score = max(0, 1 - error / 0.5)
                scores.append(score)

        self.logger.debug(
            "collision_check",
            num_collisions=len(collisions),
            avg_score=float(np.mean(scores)) if scores else 1.0,
        )

        return {
            "score": np.mean(scores) if scores else 1.0,
            "num_collisions": len(collisions),
            "restitution_scores": scores,
        }

    def _detect_collisions(self, frames: list[FrameData]) -> list[dict]:
        """Detect collision events (sudden velocity changes)."""
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

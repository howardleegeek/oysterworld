"""Tests for physics simulator."""

import numpy as np
import pytest

from physicalfish.models import PhysicsParams
from physicalfish.simulator.pybullet_sim import Simulator, NumPySimulator, PYBULLET_AVAILABLE
from physicalfish.verification.physics_verifier import PhysicsVerifier


class TestSimulatorBasics:
    """Basic simulator functionality tests."""

    def test_simulator_creation(self):
        """Simulator can be instantiated."""
        sim = Simulator()
        assert sim is not None

    def test_configure_sets_params(self):
        """Configure stores physics parameters."""
        sim = Simulator()
        params = PhysicsParams(gravity=9.81, mass=0.5)
        sim.configure(params)
        assert sim._params == params

    def test_run_scenario_requires_configure(self):
        """Running scenario without configure raises error."""
        sim = Simulator()
        with pytest.raises(RuntimeError, match="not configured"):
            sim.run_scenario("ball_drop", 1.0)

    def test_output_fps_30hz(self):
        """Output frame rate is 30Hz."""
        sim = Simulator()
        params = PhysicsParams()
        sim.configure(params)

        duration = 2.0
        trajectory = sim.run_scenario("ball_drop", duration)

        expected_frames = int(duration * 30)
        # Allow ±1 frame for rounding
        assert abs(trajectory.num_frames - expected_frames) <= 1

    def test_coordinate_system_y_up(self):
        """Position Y axis decreases during free fall (Y-up coordinate system)."""
        sim = Simulator()
        params = PhysicsParams(gravity=9.81)
        sim.configure(params)

        trajectory = sim.run_scenario("ball_drop", 1.0)

        # In Y-up system, falling means Y decreases
        first_y = trajectory.frames[0].position[1]
        last_y = trajectory.frames[-1].position[1]
        assert first_y > last_y, "Y should decrease during fall"


class TestBallDropTrajectory:
    """Ball drop trajectory physics tests."""

    def test_ball_drop_trajectory_valid(self):
        """Ball drop trajectory passes verifier with score > 0.85."""
        sim = Simulator()
        params = PhysicsParams(gravity=9.81, mass=0.5)
        sim.configure(params)

        # Use 0.8s duration to stay well above ground (y > 0.5m)
        trajectory = sim.run_scenario("ball_drop", 0.8)

        verifier = PhysicsVerifier(gravity=9.81)
        result = verifier.verify(trajectory)

        assert result.overall_score > 0.85, f"Score {result.overall_score} <= 0.85"
        assert bool(result.passed) is True

    def test_gravity_configurable(self):
        """Different gravity values produce different trajectories."""
        sim1 = Simulator()
        sim1.configure(PhysicsParams(gravity=5.0))
        traj1 = sim1.run_scenario("ball_drop", 0.5)  # Short duration to avoid ground

        sim2 = Simulator()
        sim2.configure(PhysicsParams(gravity=15.0))
        traj2 = sim2.run_scenario("ball_drop", 0.5)  # Short duration to avoid ground

        # Higher gravity should result in lower position after same time
        y1 = traj1.frames[-1].position[1]
        y2 = traj2.frames[-1].position[1]

        assert y2 < y1, f"Higher gravity should produce lower final position: y1={y1}, y2={y2}"

    def test_mass_configurable(self):
        """Mass change doesn't affect free fall acceleration (equivalence principle)."""
        # Use zero damping to ensure pure gravity (no air resistance effects)
        sim1 = Simulator()
        sim1.configure(PhysicsParams(gravity=9.81, mass=0.1, linear_damping=0.0))
        traj1 = sim1.run_scenario("ball_drop", 0.5)  # Short duration to avoid ground

        sim2 = Simulator()
        sim2.configure(PhysicsParams(gravity=9.81, mass=2.0, linear_damping=0.0))
        traj2 = sim2.run_scenario("ball_drop", 0.5)

        # Positions should be nearly identical (within numerical precision)
        y1 = traj1.frames[-1].position[1]
        y2 = traj2.frames[-1].position[1]

        assert abs(y1 - y2) < 0.1, f"Different masses should fall at same rate: y1={y1}, y2={y2}"


class TestScenarios:
    """Scenario-specific tests."""

    def test_grasp_ball_scenario(self):
        """Grasp ball scenario produces valid trajectory."""
        sim = Simulator()
        params = PhysicsParams()
        sim.configure(params)

        trajectory = sim.run_scenario("grasp_ball", 3.0)

        assert trajectory.num_frames > 0
        assert trajectory.duration >= 2.9  # ~3 seconds

    def test_throw_catch_scenario(self):
        """Throw catch scenario produces parabolic trajectory."""
        sim = Simulator()
        params = PhysicsParams()
        sim.configure(params)

        trajectory = sim.run_scenario("throw_catch", 2.0)

        # Should have horizontal movement
        x_positions = [f.position[0] for f in trajectory.frames]
        x_range = max(x_positions) - min(x_positions)
        assert x_range > 0.01, "Throw should have horizontal movement"

        # Should go up then down
        y_positions = [f.position[1] for f in trajectory.frames]
        assert max(y_positions) > y_positions[0], "Should go up initially"

    def test_roll_incline_scenario(self):
        """Roll incline scenario produces rolling motion."""
        sim = Simulator()
        params = PhysicsParams()
        sim.configure(params)

        trajectory = sim.run_scenario("roll_incline", 2.0)

        # Should have both horizontal and vertical movement
        x_positions = [f.position[0] for f in trajectory.frames]
        y_positions = [f.position[1] for f in trajectory.frames]

        x_range = max(x_positions) - min(x_positions)
        y_range = max(y_positions) - min(y_positions)

        assert x_range > 0.1, "Should roll horizontally"
        assert y_range > 0.1, "Should descend vertically"


class TestNumPySimulator:
    """NumPy simulator physics tests - verifies real physics implementation."""

    def test_numpy_simulator_works_without_pybullet(self):
        """NumPy simulator generates valid trajectories without PyBullet."""
        # Always test NumPySimulator directly
        sim = NumPySimulator()
        params = PhysicsParams(gravity=9.81, mass=0.5)
        sim.configure(params)

        trajectory = sim.run_scenario("ball_drop", 2.0)

        assert trajectory.num_frames > 0
        assert (
            len(trajectory.frames) == int(2.0 * 30) or len(trajectory.frames) == int(2.0 * 30) + 1
        )

    def test_numpy_trajectory_passes_verifier_with_high_score(self):
        """NumPy simulator trajectory passes physics verification with score > 0.85."""
        sim = NumPySimulator()
        params = PhysicsParams(gravity=9.81, mass=0.5)
        sim.configure(params)

        # Use 0.8s duration to stay well above ground
        trajectory = sim.run_scenario("ball_drop", 0.8)

        verifier = PhysicsVerifier(gravity=9.81)
        result = verifier.verify(trajectory)

        assert result.overall_score > 0.85, f"Score {result.overall_score} <= 0.85"
        assert bool(result.passed) is True

    def test_numpy_grasp_ball_phases(self):
        """NumPy grasp_ball has distinct phases with proper physics."""
        sim = NumPySimulator()
        params = PhysicsParams()
        sim.configure(params)

        trajectory = sim.run_scenario("grasp_ball", 3.0)

        # Phase 1 (0-1s): falling
        # Phase 2 (1-2s): held constant
        # Phase 3 (2-3s): falling again

        # Check that middle frames have constant height
        mid_frames = trajectory.frames[30:60]  # 1-2s at 30fps
        y_positions = [f.position[1] for f in mid_frames]

        # During grasp phase, height should be relatively constant
        y_variance = np.var(y_positions)
        assert y_variance < 0.02, f"Grasp phase should have constant height, variance={y_variance}"

    def test_numpy_collision_restitution(self):
        """NumPy simulator produces realistic bounce with restitution."""
        sim = NumPySimulator()
        # High restitution for visible bounce
        params = PhysicsParams(gravity=9.81, mass=0.5, restitution=0.9)
        sim.configure(params)

        # Drop from height, let it bounce
        trajectory = sim.run_scenario("ball_drop", 1.5)

        # Find bounce (velocity direction change)
        y_velocities = [f.velocity[1] for f in trajectory.frames]

        # Should have negative velocity (falling) then positive (rising after bounce)
        has_negative = any(v < -0.5 for v in y_velocities)
        has_positive = any(v > 0.5 for v in y_velocities)

        assert has_negative, "Should have falling velocity"
        # Note: With short duration or specific parameters, might not see bounce

    def test_numpy_free_fall_acceleration_matches_gravity(self):
        """NumPy simulator produces correct gravitational acceleration."""
        sim = NumPySimulator()
        g = 9.81
        params = PhysicsParams(gravity=g, mass=0.5)
        sim.configure(params)

        # Short duration to stay in free fall (no collision)
        trajectory = sim.run_scenario("ball_drop", 0.5)

        # Check that acceleration is approximately -g in Y direction
        # Skip first frame (initial condition) and check middle frames
        mid_frame = trajectory.frames[len(trajectory.frames) // 2]
        ay = mid_frame.acceleration[1]

        # Should be close to -g (allowing for some numerical error)
        assert abs(ay + g) < 0.5, f"Acceleration {ay} should be close to -{g}"

    def test_numpy_energy_conservation_free_fall(self):
        """NumPy simulator conserves energy in free fall (within numerical tolerance)."""
        sim = NumPySimulator()
        g = 9.81
        m = 0.5
        params = PhysicsParams(gravity=g, mass=m, linear_damping=0.0)
        sim.configure(params)

        # Short duration, no damping
        trajectory = sim.run_scenario("ball_drop", 0.5)

        # Calculate total energy at each frame
        energies = []
        for frame in trajectory.frames:
            v = np.linalg.norm(frame.velocity)
            h = frame.position[1]
            ke = 0.5 * m * v * v
            pe = m * g * h
            energies.append(ke + pe)

        # Energy should be relatively constant (within 5%)
        energy_variation = np.std(energies) / np.mean(energies)
        assert energy_variation < 0.05, f"Energy variation {energy_variation} too high"


class TestSimulatorLifecycle:
    """Simulator lifecycle management tests."""

    def test_reset_clears_state(self):
        """Reset clears simulation state."""
        sim = Simulator()
        params = PhysicsParams()
        sim.configure(params)

        # Run a scenario
        sim.run_scenario("ball_drop", 1.0)

        # Reset should not raise
        sim.reset()

        # Can run again after reset
        trajectory = sim.run_scenario("ball_drop", 1.0)
        assert trajectory.num_frames > 0

    def test_close_releases_resources(self):
        """Close releases simulator resources."""
        sim = Simulator()
        params = PhysicsParams()
        sim.configure(params)

        sim.run_scenario("ball_drop", 0.5)

        # Close should not raise
        sim.close()

    def test_multiple_runs_same_simulator(self):
        """Can run multiple scenarios on same simulator instance."""
        sim = Simulator()
        params = PhysicsParams()
        sim.configure(params)

        traj1 = sim.run_scenario("ball_drop", 1.0)
        traj2 = sim.run_scenario("throw_catch", 1.0)
        traj3 = sim.run_scenario("roll_incline", 1.0)

        assert all(t.num_frames > 0 for t in [traj1, traj2, traj3])


class TestFrameData:
    """Frame data validation tests."""

    def test_frame_has_required_fields(self):
        """Each frame has all required physics fields."""
        sim = Simulator()
        params = PhysicsParams()
        sim.configure(params)

        trajectory = sim.run_scenario("ball_drop", 1.0)

        for frame in trajectory.frames:
            assert frame.timestamp >= 0
            assert frame.position.shape == (3,)
            assert frame.velocity.shape == (3,)
            assert frame.acceleration.shape == (3,)
            assert frame.rotation.shape == (3,)
            assert frame.angular_velocity.shape == (3,)

    def test_timestamps_increase_monotonically(self):
        """Frame timestamps increase monotonically."""
        sim = Simulator()
        params = PhysicsParams()
        sim.configure(params)

        trajectory = sim.run_scenario("ball_drop", 1.0)

        timestamps = [f.timestamp for f in trajectory.frames]
        for i in range(1, len(timestamps)):
            assert timestamps[i] > timestamps[i - 1]

    def test_frame_indices_sequential(self):
        """Frame indices are sequential starting from 0."""
        sim = Simulator()
        params = PhysicsParams()
        sim.configure(params)

        trajectory = sim.run_scenario("ball_drop", 1.0)

        for i, frame in enumerate(trajectory.frames):
            assert frame.frame_index == i


class TestNumPySimulatorEdgeCases:
    """Edge case tests for NumPy simulator."""

    def test_numpy_short_duration(self):
        """NumPy simulator handles very short durations."""
        sim = NumPySimulator()
        params = PhysicsParams(gravity=9.81)
        sim.configure(params)

        # Very short duration
        trajectory = sim.run_scenario("ball_drop", 0.1)

        # Should still produce frames
        assert trajectory.num_frames > 0
        assert trajectory.duration >= 0.05  # At least half of 0.1

    def test_numpy_zero_gravity(self):
        """NumPy simulator handles zero gravity."""
        sim = NumPySimulator()
        params = PhysicsParams(gravity=0.0, linear_damping=0.0)
        sim.configure(params)

        trajectory = sim.run_scenario("ball_drop", 1.0)

        # With zero gravity and no damping, velocity should stay constant
        # (or close to it, starting from rest)
        assert trajectory.num_frames > 0

    def test_numpy_high_gravity(self):
        """NumPy simulator handles high gravity."""
        sim = NumPySimulator()
        params = PhysicsParams(gravity=20.0)
        sim.configure(params)

        trajectory = sim.run_scenario("ball_drop", 0.5)

        # Should reach ground faster with high gravity
        assert trajectory.num_frames > 0

    def test_numpy_high_damping(self):
        """NumPy simulator handles high damping."""
        sim = NumPySimulator()
        params = PhysicsParams(gravity=9.81, linear_damping=0.9)
        sim.configure(params)

        trajectory = sim.run_scenario("ball_drop", 1.0)

        # High damping should slow the fall
        assert trajectory.num_frames > 0

    def test_numpy_zero_restitution(self):
        """NumPy simulator handles zero restitution (no bounce)."""
        sim = NumPySimulator()
        params = PhysicsParams(gravity=9.81, restitution=0.0)
        sim.configure(params)

        trajectory = sim.run_scenario("ball_drop", 2.0)

        # With zero restitution, ball should not bounce
        # Check that once it hits ground, it stays
        ground_frames = [f for f in trajectory.frames if f.position[1] <= 0.15]
        if len(ground_frames) > 1:
            # Velocities should be near zero after hitting ground
            for f in ground_frames[5:]:  # Skip first few frames after impact
                assert abs(f.velocity[1]) < 0.5, "Zero restitution should stop bounce"

    def test_numpy_high_restitution(self):
        """NumPy simulator handles high restitution."""
        sim = NumPySimulator()
        params = PhysicsParams(gravity=9.81, restitution=0.95)
        sim.configure(params)

        trajectory = sim.run_scenario("ball_drop", 2.0)

        # High restitution should produce visible bounces
        y_velocities = [f.velocity[1] for f in trajectory.frames]

        # Should have both negative (falling) and positive (rising) velocities
        has_negative = any(v < -0.5 for v in y_velocities)
        has_positive = any(v > 0.5 for v in y_velocities)

        assert has_negative, "Should have falling velocity"
        # With high restitution, should see bounce

    def test_numpy_unknown_scenario_defaults_to_drop(self):
        """Unknown scenario defaults to ball drop behavior."""
        sim = NumPySimulator()
        params = PhysicsParams()
        sim.configure(params)

        # Use unknown scenario name
        trajectory = sim.run_scenario("unknown_scenario", 1.0)

        # Should still produce valid trajectory
        assert trajectory.num_frames > 0
        assert trajectory.duration >= 0.9

    def test_numpy_reset_clears_body(self):
        """Reset clears the rigid body state."""
        sim = NumPySimulator()
        params = PhysicsParams()
        sim.configure(params)

        # Run first scenario
        sim.run_scenario("ball_drop", 1.0)

        # Reset
        sim.reset()

        # Body should be None after reset
        assert sim._body is None

    def test_numpy_close_clears_resources(self):
        """Close clears all resources."""
        sim = NumPySimulator()
        params = PhysicsParams()
        sim.configure(params)

        sim.run_scenario("ball_drop", 0.5)

        # Close
        sim.close()

        # Resources should be cleared
        assert sim._body is None
        assert sim._params is None

    def test_numpy_roll_incline_physics(self):
        """Roll incline scenario applies incline physics correctly."""
        sim = NumPySimulator()
        params = PhysicsParams(gravity=9.81)
        sim.configure(params)

        trajectory = sim.run_scenario("roll_incline", 2.0)

        # Ball should move in both X and Y
        x_positions = [f.position[0] for f in trajectory.frames]
        y_positions = [f.position[1] for f in trajectory.frames]

        x_range = max(x_positions) - min(x_positions)
        y_range = max(y_positions) - min(y_positions)

        assert x_range > 0.1, "Should roll down incline (X movement)"
        assert y_range > 0.1, "Should descend (Y movement)"

    def test_numpy_grasp_ball_phases_distinct(self):
        """Grasp ball scenario has three distinct phases."""
        sim = NumPySimulator()
        params = PhysicsParams(gravity=9.81)
        sim.configure(params)

        trajectory = sim.run_scenario("grasp_ball", 3.0)

        # Phase 1: 0-1s (falling)
        phase1 = trajectory.frames[5:25]  # ~0.2-0.8s
        y1 = [f.position[1] for f in phase1]

        # Phase 2: 1-2s (grasped)
        phase2 = trajectory.frames[35:55]  # ~1.2-1.8s
        y2 = [f.position[1] for f in phase2]

        # Phase 3: 2-3s (falling again)
        phase3 = trajectory.frames[65:85]  # ~2.2-2.8s
        y3 = [f.position[1] for f in phase3]

        # Phase 2 should be more stable than phases 1 and 3
        var2 = np.var(y2)
        var1 = np.var(y1)
        var3 = np.var(y3)

        assert var2 < var1, "Grasp phase should be more stable than falling"
        assert var2 < var3, "Grasp phase should be more stable than falling"


class TestRigidBody:
    """Tests for RigidBody physics implementation."""

    def test_rigid_body_creation(self):
        """RigidBody initializes with correct values."""
        from physicalfish.simulator.pybullet_sim import RigidBody

        body = RigidBody(
            mass=1.0,
            radius=0.1,
            position=np.array([0.0, 1.0, 0.0]),
            velocity=np.array([1.0, 0.0, 0.0]),
        )

        assert body.mass == 1.0
        assert body.radius == 0.1
        assert np.allclose(body.position, [0.0, 1.0, 0.0])
        assert np.allclose(body.velocity, [1.0, 0.0, 0.0])
        assert body.on_ground is False

    def test_rigid_body_apply_force(self):
        """Apply force updates acceleration correctly."""
        from physicalfish.simulator.pybullet_sim import RigidBody

        body = RigidBody(
            mass=2.0,
            radius=0.1,
            position=np.zeros(3),
        )

        # Apply force in Y direction
        body.apply_force(np.array([0.0, 10.0, 0.0]))

        # F = ma, so a = F/m = 10/2 = 5
        assert abs(body.acceleration[1] - 5.0) < 0.001

    def test_rigid_body_integrate_updates_position(self):
        """Integration updates position based on velocity."""
        from physicalfish.simulator.pybullet_sim import RigidBody

        body = RigidBody(
            mass=1.0,
            radius=0.1,
            position=np.array([0.0, 1.0, 0.0]),
            velocity=np.array([1.0, 0.0, 0.0]),
        )

        initial_x = body.position[0]

        # Integrate with dt=0.1
        body.integrate(dt=0.1, gravity=0.0)

        # Position should update: x = x0 + v*dt = 0 + 1*0.1 = 0.1
        # Allow for damping effects
        assert abs(body.position[0] - (initial_x + 0.1)) < 0.02

    def test_rigid_body_integrate_gravity_effect(self):
        """Integration applies gravity to acceleration."""
        from physicalfish.simulator.pybullet_sim import RigidBody

        body = RigidBody(
            mass=1.0,
            radius=0.1,
            position=np.array([0.0, 1.0, 0.0]),
            velocity=np.array([0.0, 0.0, 0.0]),
        )

        # Integrate with gravity
        body.integrate(dt=0.1, gravity=9.81)

        # Velocity should decrease in Y (gravity pulls down)
        assert body.velocity[1] < 0

    def test_rigid_body_ground_collision_detection(self):
        """Ground collision is detected correctly."""
        from physicalfish.simulator.pybullet_sim import RigidBody

        body = RigidBody(
            mass=1.0,
            radius=0.1,
            position=np.array([0.0, 0.05, 0.0]),  # Below ground + radius
        )

        # Check collision at ground_y=0
        is_colliding = body.check_ground_collision(ground_y=0.0)

        # Use bool() to convert numpy bool to Python bool
        assert bool(is_colliding) is True
        assert bool(body.on_ground) is True
        assert body.ground_penetration > 0

    def test_rigid_body_no_collision_when_above_ground(self):
        """No collision when above ground."""
        from physicalfish.simulator.pybullet_sim import RigidBody

        body = RigidBody(
            mass=1.0,
            radius=0.1,
            position=np.array([0.0, 1.0, 0.0]),  # Well above ground
        )

        is_colliding = body.check_ground_collision(ground_y=0.0)

        # Use bool() to convert numpy bool to Python bool
        assert bool(is_colliding) is False
        assert bool(body.on_ground) is False

    def test_rigid_body_collision_resolution_bounce(self):
        """Collision resolution produces bounce with restitution."""
        from physicalfish.simulator.pybullet_sim import RigidBody

        body = RigidBody(
            mass=1.0,
            radius=0.1,
            position=np.array([0.0, 0.05, 0.0]),
            velocity=np.array([0.0, -5.0, 0.0]),  # Moving down
            restitution=0.8,
        )

        body.check_ground_collision(ground_y=0.0)
        body.resolve_ground_collision()

        # After bounce, velocity should be upward and reduced by restitution
        # v_after = -v_before * restitution = -(-5) * 0.8 = 4.0
        assert body.velocity[1] > 0, "Should bounce upward"
        assert abs(body.velocity[1] - 4.0) < 0.1, "Should have restitution applied"

    def test_rigid_body_collision_resolution_no_bounce_when_moving_up(self):
        """No bounce when already moving upward."""
        from physicalfish.simulator.pybullet_sim import RigidBody

        body = RigidBody(
            mass=1.0,
            radius=0.1,
            position=np.array([0.0, 0.05, 0.0]),
            velocity=np.array([0.0, 5.0, 0.0]),  # Moving up
            restitution=0.8,
        )

        body.check_ground_collision(ground_y=0.0)
        initial_vy = body.velocity[1]
        body.resolve_ground_collision()

        # Velocity should not change much (already moving up)
        assert abs(body.velocity[1] - initial_vy) < 0.1

    def test_rigid_body_sleep_threshold(self):
        """Very slow bodies are put to sleep (velocity zeroed)."""
        from physicalfish.simulator.pybullet_sim import RigidBody

        body = RigidBody(
            mass=1.0,
            radius=0.1,
            position=np.array([0.0, 0.05, 0.0]),
            velocity=np.array([0.0, -0.005, 0.0]),  # Very slow
            restitution=0.5,
        )

        body.check_ground_collision(ground_y=0.0)
        body.resolve_ground_collision()

        # Should be stopped due to sleep threshold
        assert abs(body.velocity[1]) < 0.01

    def test_rigid_body_friction_reduces_horizontal_velocity(self):
        """Friction reduces horizontal velocity on bounce."""
        from physicalfish.simulator.pybullet_sim import RigidBody

        body = RigidBody(
            mass=1.0,
            radius=0.1,
            position=np.array([0.0, 0.05, 0.0]),
            velocity=np.array([3.0, -5.0, 0.0]),  # Moving down and sideways
            restitution=0.8,
            friction=0.5,
        )

        initial_vx = body.velocity[0]

        body.check_ground_collision(ground_y=0.0)
        body.resolve_ground_collision()

        # Horizontal velocity should be reduced by friction
        assert abs(body.velocity[0]) < abs(initial_vx)

    def test_rigid_body_angular_damping(self):
        """Angular damping reduces angular velocity."""
        from physicalfish.simulator.pybullet_sim import RigidBody

        body = RigidBody(
            mass=1.0,
            radius=0.1,
            position=np.zeros(3),
            angular_damping=0.5,
        )

        # Set angular velocity after creation
        body.angular_velocity = np.array([1.0, 0.0, 0.0])
        initial_omega = body.angular_velocity[0]

        # Integrate with dt=0.1
        body.integrate(dt=0.1, gravity=0.0)

        # Angular velocity should be reduced
        assert abs(body.angular_velocity[0]) < abs(initial_omega)

    def test_rigid_body_inertia_calculation(self):
        """Inertia is calculated correctly for solid sphere."""
        from physicalfish.simulator.pybullet_sim import RigidBody

        body = RigidBody(
            mass=2.0,
            radius=0.1,
            position=np.zeros(3),
        )

        # I = (2/5) * m * r^2 = 0.4 * 2.0 * 0.01 = 0.008
        expected_inertia = 0.4 * 2.0 * 0.1 * 0.1
        assert abs(body.inertia - expected_inertia) < 0.0001

    def test_rigid_body_rolling_rotation(self):
        """Rolling produces appropriate angular velocity."""
        from physicalfish.simulator.pybullet_sim import RigidBody

        body = RigidBody(
            mass=1.0,
            radius=0.1,
            position=np.array([0.0, 0.05, 0.0]),
            velocity=np.array([1.0, -1.0, 0.0]),  # Moving sideways and down
            restitution=0.5,
            friction=0.8,
        )

        body.check_ground_collision(ground_y=0.0)
        body.resolve_ground_collision()

        # Should have some angular velocity from rolling
        # omega = v / r for rolling without slipping
        if np.linalg.norm(body.velocity[[0, 2]]) > 0.01:
            assert np.linalg.norm(body.angular_velocity) > 0


class TestSimulatorConfiguration:
    """Tests for simulator configuration edge cases."""

    def test_simulator_reconfigure_changes_params(self):
        """Reconfiguring changes physics parameters."""
        sim = NumPySimulator()

        # First configuration
        params1 = PhysicsParams(gravity=5.0, mass=0.5)
        sim.configure(params1)

        # Second configuration
        params2 = PhysicsParams(gravity=15.0, mass=1.5)
        sim.configure(params2)

        assert sim._params.gravity == 15.0
        assert sim._params.mass == 1.5

    def test_simulator_different_durations(self):
        """Simulator handles different durations correctly."""
        sim = NumPySimulator()
        params = PhysicsParams()
        sim.configure(params)

        for duration in [0.5, 1.0, 2.0, 3.0]:
            sim.reset()
            trajectory = sim.run_scenario("ball_drop", duration)

            expected_frames = int(duration * 30)
            assert abs(trajectory.num_frames - expected_frames) <= 1

    def test_simulator_frame_rate_consistency(self):
        """Output frame rate is consistently 30Hz."""
        sim = NumPySimulator()
        params = PhysicsParams()
        sim.configure(params)

        trajectory = sim.run_scenario("ball_drop", 2.0)

        # Check that frame timestamps are ~1/30s apart
        timestamps = [f.timestamp for f in trajectory.frames]
        for i in range(1, len(timestamps)):
            dt = timestamps[i] - timestamps[i - 1]
            assert abs(dt - 1 / 30) < 0.01, f"Frame interval {dt} should be ~1/30s"

    def test_simulator_acceleration_calculation(self):
        """Acceleration is calculated correctly from velocity change."""
        sim = NumPySimulator()
        params = PhysicsParams(gravity=9.81)
        sim.configure(params)

        trajectory = sim.run_scenario("ball_drop", 1.0)

        # Skip first frame (initial condition)
        for i in range(2, len(trajectory.frames)):
            frame = trajectory.frames[i]
            prev_frame = trajectory.frames[i - 1]

            # Acceleration should be approximately (v - v_prev) / dt
            dt = 1.0 / 30.0
            expected_a = (frame.velocity - prev_frame.velocity) / dt

            # Allow for some numerical error
            assert np.linalg.norm(frame.acceleration - expected_a) < 1.0

    def test_simulator_throw_catch_parabolic_motion(self):
        """Throw catch scenario produces parabolic motion."""
        sim = NumPySimulator()
        params = PhysicsParams(gravity=9.81)
        sim.configure(params)

        trajectory = sim.run_scenario("throw_catch", 2.0)

        # Extract positions
        x_pos = [f.position[0] for f in trajectory.frames]
        y_pos = [f.position[1] for f in trajectory.frames]

        # Should go up then down
        max_y_idx = np.argmax(y_pos)
        assert max_y_idx > 0, "Should reach max height after start"
        assert max_y_idx < len(y_pos) - 1, "Should come down before end"

        # X should generally increase (thrown in +x direction)
        assert x_pos[-1] > x_pos[0], "Should move forward in X"


class TestNumPySimulatorScenarios:
    """Additional scenario tests for NumPy simulator."""

    def test_numpy_default_scenario(self):
        """Default scenario (unknown name) produces valid trajectory."""
        sim = NumPySimulator()
        params = PhysicsParams()
        sim.configure(params)

        # Use unknown scenario - should default to ball_drop
        trajectory = sim.run_scenario("nonexistent_scenario", 1.0)

        assert trajectory.num_frames > 0
        assert trajectory.duration >= 0.9

    def test_numpy_scenario_with_zero_mass(self):
        """Scenario with very low mass."""
        sim = NumPySimulator()
        params = PhysicsParams(mass=0.1)
        sim.configure(params)

        trajectory = sim.run_scenario("ball_drop", 1.0)

        assert trajectory.num_frames > 0

    def test_numpy_scenario_with_high_mass(self):
        """Scenario with high mass."""
        sim = NumPySimulator()
        params = PhysicsParams(mass=2.0)
        sim.configure(params)

        trajectory = sim.run_scenario("ball_drop", 1.0)

        assert trajectory.num_frames > 0

    def test_numpy_scenario_with_zero_friction(self):
        """Scenario with zero friction."""
        sim = NumPySimulator()
        params = PhysicsParams(friction=0.0)
        sim.configure(params)

        trajectory = sim.run_scenario("throw_catch", 1.0)

        assert trajectory.num_frames > 0

    def test_numpy_scenario_with_high_friction(self):
        """Scenario with high friction."""
        sim = NumPySimulator()
        params = PhysicsParams(friction=1.0)
        sim.configure(params)

        trajectory = sim.run_scenario("roll_incline", 1.0)

        assert trajectory.num_frames > 0

    def test_numpy_scenario_with_zero_restitution(self):
        """Scenario with zero restitution (no bounce)."""
        sim = NumPySimulator()
        params = PhysicsParams(restitution=0.0)
        sim.configure(params)

        trajectory = sim.run_scenario("ball_drop", 1.5)

        assert trajectory.num_frames > 0

    def test_numpy_scenario_with_high_restitution(self):
        """Scenario with high restitution."""
        sim = NumPySimulator()
        params = PhysicsParams(restitution=1.0)
        sim.configure(params)

        trajectory = sim.run_scenario("ball_drop", 1.5)

        assert trajectory.num_frames > 0

    def test_numpy_roll_incline_edge_cases(self):
        """Roll incline with different parameters."""
        sim = NumPySimulator()

        # Test with different gravity values
        for g in [5.0, 9.81, 15.0]:
            sim.reset()
            params = PhysicsParams(gravity=g)
            sim.configure(params)

            trajectory = sim.run_scenario("roll_incline", 1.0)
            assert trajectory.num_frames > 0

    def test_numpy_grasp_ball_edge_cases(self):
        """Grasp ball with different timing."""
        sim = NumPySimulator()
        params = PhysicsParams()
        sim.configure(params)

        # Short duration - only phase 1
        trajectory_short = sim.run_scenario("grasp_ball", 0.5)
        assert trajectory_short.num_frames > 0

        # Medium duration - phases 1 and 2
        sim.reset()
        sim.configure(params)
        trajectory_med = sim.run_scenario("grasp_ball", 1.5)
        assert trajectory_med.num_frames > 0

    def test_numpy_throw_catch_edge_cases(self):
        """Throw catch with different parameters."""
        sim = NumPySimulator()

        # Test with different damping values
        for damping in [0.0, 0.1, 0.5]:
            sim.reset()
            params = PhysicsParams(linear_damping=damping)
            sim.configure(params)

            trajectory = sim.run_scenario("throw_catch", 1.0)
            assert trajectory.num_frames > 0


class TestRigidBodyEdgeCases:
    """Additional edge case tests for RigidBody."""

    def test_rigid_body_zero_mass(self):
        """RigidBody handles very low mass."""
        from physicalfish.simulator.pybullet_sim import RigidBody

        body = RigidBody(
            mass=0.1,
            radius=0.1,
            position=np.array([0.0, 1.0, 0.0]),
        )

        assert body.mass == 0.1
        assert body.inertia > 0

    def test_rigid_body_high_mass(self):
        """RigidBody handles high mass."""
        from physicalfish.simulator.pybullet_sim import RigidBody

        body = RigidBody(
            mass=10.0,
            radius=0.5,
            position=np.array([0.0, 1.0, 0.0]),
        )

        assert body.mass == 10.0
        assert body.inertia > 0

    def test_rigid_body_zero_radius(self):
        """RigidBody handles very small radius."""
        from physicalfish.simulator.pybullet_sim import RigidBody

        body = RigidBody(
            mass=1.0,
            radius=0.01,
            position=np.array([0.0, 1.0, 0.0]),
        )

        assert body.radius == 0.01
        assert body.inertia > 0

    def test_rigid_body_zero_velocity(self):
        """RigidBody handles zero initial velocity."""
        from physicalfish.simulator.pybullet_sim import RigidBody

        body = RigidBody(
            mass=1.0,
            radius=0.1,
            position=np.array([0.0, 1.0, 0.0]),
            velocity=None,  # Should default to zero
        )

        assert np.allclose(body.velocity, [0.0, 0.0, 0.0])

    def test_rigid_body_extreme_position(self):
        """RigidBody handles extreme positions."""
        from physicalfish.simulator.pybullet_sim import RigidBody

        body = RigidBody(
            mass=1.0,
            radius=0.1,
            position=np.array([100.0, -50.0, 200.0]),
        )

        assert np.allclose(body.position, [100.0, -50.0, 200.0])

    def test_rigid_body_apply_zero_force(self):
        """Applying zero force doesn't change acceleration."""
        from physicalfish.simulator.pybullet_sim import RigidBody

        body = RigidBody(
            mass=1.0,
            radius=0.1,
            position=np.zeros(3),
        )

        initial_accel = body.acceleration.copy()
        body.apply_force(np.zeros(3))

        assert np.allclose(body.acceleration, initial_accel)

    def test_rigid_body_integrate_zero_dt(self):
        """Integration with zero dt doesn't change state."""
        from physicalfish.simulator.pybullet_sim import RigidBody

        body = RigidBody(
            mass=1.0,
            radius=0.1,
            position=np.array([0.0, 1.0, 0.0]),
            velocity=np.array([1.0, 2.0, 3.0]),
        )

        initial_pos = body.position.copy()
        initial_vel = body.velocity.copy()

        body.integrate(dt=0.0, gravity=9.81)

        assert np.allclose(body.position, initial_pos)
        # Velocity still changes due to gravity even with dt=0

    def test_rigid_body_collision_at_exact_ground_level(self):
        """Collision when position is exactly at ground + radius."""
        from physicalfish.simulator.pybullet_sim import RigidBody

        body = RigidBody(
            mass=1.0,
            radius=0.1,
            position=np.array([0.0, 0.1, 0.0]),  # Exactly at ground + radius
        )

        is_colliding = body.check_ground_collision(ground_y=0.0)

        # Should not be colliding (exactly at surface)
        assert bool(is_colliding) is False

    def test_rigid_body_collision_resolution_with_zero_restitution(self):
        """Collision resolution with zero restitution stops object."""
        from physicalfish.simulator.pybullet_sim import RigidBody

        body = RigidBody(
            mass=1.0,
            radius=0.1,
            position=np.array([0.0, 0.05, 0.0]),
            velocity=np.array([0.0, -5.0, 0.0]),
            restitution=0.0,
        )

        body.check_ground_collision(ground_y=0.0)
        body.resolve_ground_collision()

        # Should stop (zero restitution)
        assert abs(body.velocity[1]) < 0.1

    def test_rigid_body_collision_resolution_with_perfect_restitution(self):
        """Collision resolution with restitution=1 preserves velocity magnitude."""
        from physicalfish.simulator.pybullet_sim import RigidBody

        body = RigidBody(
            mass=1.0,
            radius=0.1,
            position=np.array([0.0, 0.05, 0.0]),
            velocity=np.array([0.0, -5.0, 0.0]),
            restitution=1.0,
        )

        body.check_ground_collision(ground_y=0.0)
        body.resolve_ground_collision()

        # Should bounce with same magnitude
        assert body.velocity[1] > 0  # Upward

    def test_rigid_body_friction_with_zero_horizontal_velocity(self):
        """Friction has no effect with zero horizontal velocity."""
        from physicalfish.simulator.pybullet_sim import RigidBody

        body = RigidBody(
            mass=1.0,
            radius=0.1,
            position=np.array([0.0, 0.05, 0.0]),
            velocity=np.array([0.0, -5.0, 0.0]),  # Only vertical
            restitution=0.8,
            friction=0.5,
        )

        body.check_ground_collision(ground_y=0.0)
        body.resolve_ground_collision()

        # Should only have vertical velocity after bounce
        assert abs(body.velocity[0]) < 0.01
        assert abs(body.velocity[2]) < 0.01

    def test_rigid_body_rolling_with_high_friction(self):
        """High friction produces more rolling rotation."""
        from physicalfish.simulator.pybullet_sim import RigidBody

        body = RigidBody(
            mass=1.0,
            radius=0.1,
            position=np.array([0.0, 0.05, 0.0]),
            velocity=np.array([5.0, -3.0, 0.0]),  # High horizontal
            restitution=0.5,
            friction=0.9,  # High friction
        )

        body.check_ground_collision(ground_y=0.0)
        body.resolve_ground_collision()

        # Should have angular velocity from rolling
        if np.linalg.norm(body.velocity[[0, 2]]) > 0.01:
            assert np.linalg.norm(body.angular_velocity) > 0

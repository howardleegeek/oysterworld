"""PyBullet-based physics simulator with NumPy-only fallback."""

import numpy as np

from physicalfish.models import FrameData, PhysicsParams, TrajectoryData
from physicalfish.simulator.scenarios import ScenarioBuilder

# Try to import PyBullet, fall back to NumPy simulator if unavailable
try:
    import pybullet as p
    import pybullet_data

    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False


class PyBulletSimulator:
    """PyBullet-based physics simulator (DIRECT mode)."""

    PHYSICS_FREQ = 240  # Hz
    OUTPUT_FREQ = 30  # Hz

    def __init__(self):
        self._client_id: int | None = None
        self._params: PhysicsParams | None = None
        self._object_id: int | None = None
        self._ground_id: int | None = None
        self._scenario_builder = ScenarioBuilder()

    def _ensure_connected(self) -> None:
        """Ensure PyBullet is connected."""
        if self._client_id is None:
            self._client_id = p.connect(p.DIRECT)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            # Load ground plane
            self._ground_id = p.loadURDF("plane.urdf")

    def configure(self, params: PhysicsParams) -> None:
        """Configure simulation parameters."""
        self._ensure_connected()
        self._params = params

        # PyBullet uses Z-up coordinate system
        # PhysicalFish uses Y-up, so we swap: gravity in Y becomes gravity in Z
        p.setGravity(0, 0, -params.gravity)

        # If object exists, update its dynamics
        if self._object_id is not None:
            p.changeDynamics(
                self._object_id,
                -1,
                mass=params.mass,
                lateralFriction=params.friction,
                restitution=params.restitution,
                linearDamping=params.linear_damping,
                angularDamping=params.angular_damping,
            )

    def run_scenario(self, scenario: str, duration: float) -> TrajectoryData:
        """Run a physics scenario and return trajectory data."""
        self._ensure_connected()

        if self._params is None:
            raise RuntimeError("Simulator not configured. Call configure() first.")

        # Build the scenario
        self._scenario_builder.build(scenario, self._params)
        self._object_id = self._scenario_builder.object_id

        # Calculate step ratio: physics steps per output frame
        step_ratio = int(self.PHYSICS_FREQ / self.OUTPUT_FREQ)
        dt_physics = 1.0 / self.PHYSICS_FREQ
        dt_output = 1.0 / self.OUTPUT_FREQ

        # Run simulation and collect frames
        frames: list[FrameData] = []
        total_steps = int(duration * self.PHYSICS_FREQ)

        prev_velocity = np.zeros(3)
        prev_time = 0.0

        for step in range(total_steps):
            p.stepSimulation()

            # Record frame every step_ratio steps
            if step % step_ratio == 0:
                frame_idx = len(frames)
                t = frame_idx * dt_output

                # Get object state
                pos, orn = p.getBasePositionAndOrientation(self._object_id)
                lin_vel, ang_vel = p.getBaseVelocity(self._object_id)

                # Convert from PyBullet Z-up to PhysicalFish Y-up
                # Swap Y and Z coordinates
                position = np.array([pos[0], pos[2], pos[1]])
                velocity = np.array([lin_vel[0], lin_vel[2], lin_vel[1]])
                angular_velocity = np.array([ang_vel[0], ang_vel[2], ang_vel[1]])

                # Calculate acceleration from velocity difference
                if frame_idx == 0:
                    acceleration = np.array([0.0, -self._params.gravity, 0.0])
                else:
                    dt = t - prev_time
                    if dt > 0:
                        acceleration = (velocity - prev_velocity) / dt
                    else:
                        acceleration = np.array([0.0, -self._params.gravity, 0.0])

                # Convert rotation (Euler angles from quaternion)
                # Note: This is a simplified conversion
                rotation = np.array([0.0, 0.0, 0.0])  # Simplified for now

                frame = FrameData(
                    timestamp=t,
                    position=position,
                    velocity=velocity,
                    acceleration=acceleration,
                    rotation=rotation,
                    angular_velocity=angular_velocity,
                    frame_index=frame_idx,
                )
                frames.append(frame)

                prev_velocity = velocity
                prev_time = t

        # Clean up scenario objects
        self._scenario_builder.cleanup()
        self._object_id = None

        return TrajectoryData(frames=frames, params=self._params)

    def reset(self) -> None:
        """Reset simulation state."""
        self._ensure_connected()
        if self._object_id is not None:
            p.removeBody(self._object_id)
            self._object_id = None
        self._scenario_builder.cleanup()

    def close(self) -> None:
        """Clean up resources."""
        if self._client_id is not None:
            p.disconnect(self._client_id)
            self._client_id = None
            self._object_id = None
            self._ground_id = None


class RigidBody:
    """Simple rigid body for NumPy-based physics simulation."""

    def __init__(
        self,
        mass: float,
        radius: float,
        position: np.ndarray,
        velocity: np.ndarray = None,
        restitution: float = 0.8,
        friction: float = 0.5,
        linear_damping: float = 0.1,
        angular_damping: float = 0.1,
    ):
        self.mass = mass
        self.radius = radius
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity if velocity is not None else [0.0, 0.0, 0.0], dtype=float)
        self.acceleration = np.array([0.0, 0.0, 0.0], dtype=float)
        self.rotation = np.array([0.0, 0.0, 0.0], dtype=float)  # Euler angles
        self.angular_velocity = np.array([0.0, 0.0, 0.0], dtype=float)

        # Physics parameters
        self.restitution = restitution
        self.friction = friction
        self.linear_damping = linear_damping
        self.angular_damping = angular_damping

        # For moment of inertia of solid sphere: I = (2/5) * m * r^2
        self.inertia = (2.0 / 5.0) * mass * radius * radius

        # Ground contact state
        self.on_ground = False
        self.ground_penetration = 0.0

    def apply_force(self, force: np.ndarray) -> None:
        """Apply a force to the body (F = ma)."""
        self.acceleration += force / self.mass

    def integrate(self, dt: float, gravity: float) -> None:
        """Semi-implicit Euler integration."""
        # Apply gravity
        self.acceleration[1] -= gravity

        # Apply linear damping (air resistance)
        damping_force = -self.linear_damping * self.velocity
        self.acceleration += damping_force / self.mass

        # Integrate velocity
        self.velocity += self.acceleration * dt

        # Integrate position
        self.position += self.velocity * dt

        # Integrate angular velocity with damping
        self.angular_velocity *= 1.0 - self.angular_damping * dt
        self.rotation += self.angular_velocity * dt

        # Reset acceleration for next frame
        self.acceleration = np.array([0.0, 0.0, 0.0], dtype=float)

    def check_ground_collision(self, ground_y: float = 0.0) -> bool:
        """Check collision with ground plane."""
        self.ground_penetration = ground_y + self.radius - self.position[1]
        self.on_ground = self.ground_penetration > 0
        return self.on_ground

    def resolve_ground_collision(self) -> None:
        """Resolve collision with ground using impulse-based response."""
        if not self.on_ground:
            return

        # Position correction (push out of ground)
        self.position[1] += self.ground_penetration

        # Get velocity components
        v_y = self.velocity[1]
        v_horizontal = np.array([self.velocity[0], 0.0, self.velocity[2]])

        # Only bounce if moving downward
        if v_y < 0:
            # Apply restitution to vertical velocity
            v_y = -v_y * self.restitution

            # Apply friction to horizontal velocity
            horizontal_speed = np.linalg.norm(v_horizontal)
            if horizontal_speed > 0:
                # Friction impulse reduces horizontal velocity
                friction_factor = max(0.0, 1.0 - self.friction * (1.0 + self.restitution))
                v_horizontal *= friction_factor

        # Stop if very slow (sleep threshold)
        total_speed = np.sqrt(v_y * v_y + np.dot(v_horizontal, v_horizontal))
        if total_speed < 0.01:
            v_y = 0.0
            v_horizontal = np.array([0.0, 0.0, 0.0])

        self.velocity[1] = v_y
        self.velocity[0] = v_horizontal[0]
        self.velocity[2] = v_horizontal[2]

        # Add rolling rotation based on horizontal velocity
        # For a rolling sphere: omega = v / r
        if np.linalg.norm(v_horizontal) > 0.01:
            # Rolling without slipping
            roll_axis = np.cross(np.array([0.0, 1.0, 0.0]), v_horizontal)
            if np.linalg.norm(roll_axis) > 0:
                roll_axis = roll_axis / np.linalg.norm(roll_axis)
                roll_speed = np.linalg.norm(v_horizontal) / self.radius
                self.angular_velocity = roll_axis * roll_speed


class NumPySimulator:
    """Pure NumPy physics simulator with real rigid body dynamics.

    Implements:
    - Semi-implicit Euler integration
    - Sphere-plane collision detection
    - Impulse-based collision response with restitution and friction
    - Configurable gravity, damping, and material properties
    """

    PHYSICS_FREQ = 240  # Hz - internal physics timestep
    OUTPUT_FREQ = 30  # Hz - output frame rate

    def __init__(self):
        self._params: PhysicsParams | None = None
        self._body: RigidBody | None = None
        self._scenario: str = ""
        self._ground_y: float = 0.0

    def configure(self, params: PhysicsParams) -> None:
        """Configure simulation parameters."""
        self._params = params

    def run_scenario(self, scenario: str, duration: float) -> TrajectoryData:
        """Run a physics scenario and return trajectory data."""
        if self._params is None:
            raise RuntimeError("Simulator not configured. Call configure() first.")

        self._scenario = scenario

        # Initialize body based on scenario
        self._body = self._create_body_for_scenario(scenario)

        # Calculate timesteps
        dt_physics = 1.0 / self.PHYSICS_FREQ
        dt_output = 1.0 / self.OUTPUT_FREQ
        step_ratio = int(self.PHYSICS_FREQ / self.OUTPUT_FREQ)
        total_steps = int(duration * self.PHYSICS_FREQ)

        # Run simulation
        frames: list[FrameData] = []
        prev_velocity = np.copy(self._body.velocity)

        for step in range(total_steps):
            # Apply scenario-specific forces/constraints
            self._apply_scenario_physics(step, dt_physics)

            # Integrate physics
            self._body.integrate(dt_physics, self._params.gravity)

            # Check and resolve collisions
            if self._body.check_ground_collision(self._ground_y):
                self._body.resolve_ground_collision()

            # Record frame at output frequency
            if step % step_ratio == 0:
                frame = self._record_frame(
                    frame_idx=len(frames),
                    timestamp=len(frames) * dt_output,
                    prev_velocity=prev_velocity,
                    dt_output=dt_output,
                )
                frames.append(frame)
                prev_velocity = np.copy(self._body.velocity)

        return TrajectoryData(frames=frames, params=self._params)

    def _create_body_for_scenario(self, scenario: str) -> RigidBody:
        """Create a rigid body initialized for the given scenario."""
        params = self._params
        radius = 0.1  # Default ball radius

        if scenario == "ball_drop":
            # Simple drop from height
            return RigidBody(
                mass=params.mass,
                radius=radius,
                position=np.array([0.0, 5.0, 0.0]),
                velocity=np.array([0.0, 0.0, 0.0]),
                restitution=params.restitution,
                friction=params.friction,
                linear_damping=params.linear_damping,
                angular_damping=params.angular_damping,
            )

        elif scenario == "grasp_ball":
            # Start at height 2.0, will be grasped during simulation
            return RigidBody(
                mass=params.mass,
                radius=radius,
                position=np.array([0.0, 2.0, 0.0]),
                velocity=np.array([0.0, 0.0, 0.0]),
                restitution=params.restitution,
                friction=params.friction,
                linear_damping=params.linear_damping,
                angular_damping=params.angular_damping,
            )

        elif scenario == "throw_catch":
            # Thrown with initial velocity
            return RigidBody(
                mass=params.mass,
                radius=radius,
                position=np.array([0.0, 1.0, 0.0]),
                velocity=np.array([2.0, 5.0, 0.0]),
                restitution=params.restitution,
                friction=params.friction,
                linear_damping=params.linear_damping,
                angular_damping=params.angular_damping,
            )

        elif scenario == "roll_incline":
            # Start at top of incline
            return RigidBody(
                mass=params.mass,
                radius=radius,
                position=np.array([-1.0, 1.0, 0.0]),
                velocity=np.array([0.0, 0.0, 0.0]),
                restitution=0.2,  # Less bouncy for rolling
                friction=0.8,  # More friction for rolling
                linear_damping=params.linear_damping,
                angular_damping=params.angular_damping,
            )

        else:
            # Default: simple ball drop
            return RigidBody(
                mass=params.mass,
                radius=radius,
                position=np.array([0.0, 2.0, 0.0]),
                velocity=np.array([0.0, 0.0, 0.0]),
                restitution=params.restitution,
                friction=params.friction,
                linear_damping=params.linear_damping,
                angular_damping=params.angular_damping,
            )

    def _apply_scenario_physics(self, step: int, dt: float) -> None:
        """Apply scenario-specific physics effects."""
        t = step * dt
        body = self._body

        if self._scenario == "grasp_ball":
            # Phase 1 (0-1s): Free fall - no extra forces
            # Phase 2 (1-2s): Grasped - hold at constant height
            # Phase 3 (2-3s): Released - free fall again

            if 1.0 <= t < 2.0:
                # Grasped: hold at y = 0.5
                target_y = 0.5
                # Very strong spring force to hold position
                displacement = target_y - body.position[1]
                spring_k = 5000.0  # Very stiff spring
                damping = 100.0  # Heavy damping

                # PD controller to hold position
                force_y = spring_k * displacement - damping * body.velocity[1]
                body.apply_force(np.array([0.0, force_y, 0.0]))

                # Zero out horizontal velocity to hold steady
                body.velocity[0] *= 0.5
                body.velocity[2] *= 0.5

        elif self._scenario == "throw_catch":
            # Already has initial velocity, just let physics do its thing
            pass

        elif self._scenario == "roll_incline":
            # Simulate rolling down a 30-degree incline
            # Incline goes from (-1, 1) to (1, 0) - descending in Y as X increases
            incline_angle = np.radians(30)

            # Check if on the incline (roughly)
            x = body.position[0]
            y = body.position[1]

            # Incline surface height at this x position
            # Incline: y = 1 - (x + 1) * tan(30°) for x in [-1, 1]
            if -1.0 <= x <= 1.0:
                incline_y = 1.0 - (x + 1.0) * np.tan(incline_angle)

                # Check if ball is on or near the incline surface
                if abs(y - incline_y - body.radius) < 0.2:
                    # Apply gravity component along incline
                    g_parallel = self._params.gravity * np.sin(incline_angle)

                    # Force along incline (positive x direction, negative y)
                    force_x = g_parallel * body.mass * np.cos(incline_angle)
                    force_y = -g_parallel * body.mass * np.sin(incline_angle)

                    body.apply_force(np.array([force_x, force_y, 0.0]))

                    # Apply rolling friction
                    v_parallel = body.velocity[0] * np.cos(incline_angle) - body.velocity[
                        1
                    ] * np.sin(incline_angle)
                    friction_force = (
                        -0.1
                        * body.mass
                        * self._params.gravity
                        * np.cos(incline_angle)
                        * np.sign(v_parallel)
                    )
                    body.apply_force(
                        np.array(
                            [
                                friction_force * np.cos(incline_angle),
                                -friction_force * np.sin(incline_angle),
                                0.0,
                            ]
                        )
                    )

    def _record_frame(
        self,
        frame_idx: int,
        timestamp: float,
        prev_velocity: np.ndarray,
        dt_output: float,
    ) -> FrameData:
        """Record a frame from current body state."""
        body = self._body

        # Calculate acceleration from velocity change
        if frame_idx == 0:
            # First frame: assume free fall
            acceleration = np.array([0.0, -self._params.gravity, 0.0])
        else:
            acceleration = (body.velocity - prev_velocity) / dt_output

        return FrameData(
            timestamp=timestamp,
            position=np.copy(body.position),
            velocity=np.copy(body.velocity),
            acceleration=acceleration,
            rotation=np.copy(body.rotation),
            angular_velocity=np.copy(body.angular_velocity),
            frame_index=frame_idx,
        )

    def reset(self) -> None:
        """Reset simulation state."""
        self._body = None
        self._scenario = ""

    def close(self) -> None:
        """Clean up resources."""
        self._body = None
        self._params = None


# Export the appropriate simulator class
if PYBULLET_AVAILABLE:
    Simulator = PyBulletSimulator
else:
    Simulator = NumPySimulator

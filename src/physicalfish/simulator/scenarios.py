"""Physics scenario definitions for simulator."""

import numpy as np

from physicalfish.models import PhysicsParams

# Try to import PyBullet
try:
    import pybullet as p

    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False


class ScenarioBuilder:
    """Builds physics scenarios in PyBullet."""

    def __init__(self):
        self.object_id: int | None = None
        self.constraint_id: int | None = None
        self._auxiliary_objects: list[int] = []

    def build(self, scenario: str, params: PhysicsParams) -> None:
        """Build a scenario by name.

        Args:
            scenario: Scenario name ("grasp_ball", "throw_catch", "roll_incline")
            params: Physics parameters for the objects
        """
        if not PYBULLET_AVAILABLE:
            return

        # Clean up any existing objects
        self.cleanup()

        if scenario == "grasp_ball":
            self._build_grasp_ball(params)
        elif scenario == "throw_catch":
            self._build_throw_catch(params)
        elif scenario == "roll_incline":
            self._build_roll_incline(params)
        else:
            # Default: simple ball drop
            self._build_ball_drop(params)

    def _build_ball_drop(self, params: PhysicsParams) -> None:
        """Build a simple ball drop scenario."""
        # Create sphere collision shape
        radius = 0.1
        collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)

        # Create multi-body with mass
        visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=[1, 0, 0, 1])

        self.object_id = p.createMultiBody(
            baseMass=params.mass,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=[0, 0, 2.0],  # Z-up: start at height 2.0
        )

        # Set dynamics properties
        p.changeDynamics(
            self.object_id,
            -1,
            lateralFriction=params.friction,
            restitution=params.restitution,
            linearDamping=params.linear_damping,
            angularDamping=params.angular_damping,
        )

    def _build_grasp_ball(self, params: PhysicsParams) -> None:
        """Build grasp_ball scenario.

        Phase 1: Ball falls freely
        Phase 2: Constraint grabs the ball
        Phase 3: Constraint releases
        """
        # Create sphere
        radius = 0.1
        collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
        visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=[0, 0, 1, 1])

        # Start at height 2.0
        self.object_id = p.createMultiBody(
            baseMass=params.mass,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=[0, 0, 2.0],
        )

        p.changeDynamics(
            self.object_id,
            -1,
            lateralFriction=params.friction,
            restitution=params.restitution,
            linearDamping=params.linear_damping,
            angularDamping=params.angular_damping,
        )

        # Note: The constraint will be created dynamically during simulation
        # by the simulator based on timing (at 1s mark)

    def _build_throw_catch(self, params: PhysicsParams) -> None:
        """Build throw_catch scenario.

        Ball thrown with initial velocity, follows parabolic trajectory.
        """
        radius = 0.1
        collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
        visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=[0, 1, 0, 1])

        # Start at ground level with initial velocity
        self.object_id = p.createMultiBody(
            baseMass=params.mass,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=[0, 0, 1.0],  # Start at height 1.0
        )

        p.changeDynamics(
            self.object_id,
            -1,
            lateralFriction=params.friction,
            restitution=params.restitution,
            linearDamping=params.linear_damping,
            angularDamping=params.angular_damping,
        )

        # Set initial velocity (throw)
        # X and Z in PyBullet correspond to X and Y in PhysicalFish after swap
        initial_velocity = [2.0, 0, 5.0]  # Forward and up
        p.resetBaseVelocity(self.object_id, linearVelocity=initial_velocity)

    def _build_roll_incline(self, params: PhysicsParams) -> None:
        """Build roll_incline scenario.

        Ball rolls down an inclined plane.
        """
        # Create the ball
        radius = 0.1
        ball_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
        ball_visual = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=[1, 1, 0, 1])

        # Start at top of incline
        self.object_id = p.createMultiBody(
            baseMass=params.mass,
            baseCollisionShapeIndex=ball_collision,
            baseVisualShapeIndex=ball_visual,
            basePosition=[-1.0, 0, 1.0],  # Start at top
        )

        p.changeDynamics(
            self.object_id,
            -1,
            lateralFriction=params.friction,
            restitution=params.restitution,
            linearDamping=params.linear_damping,
            angularDamping=params.angular_damping,
            rollingFriction=0.01,
        )

        # Create inclined plane (box rotated 30 degrees)
        incline_angle = np.radians(30)
        box_half_extents = [2.0, 0.5, 0.05]  # Long, narrow, thin box

        plane_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=box_half_extents)
        plane_visual = p.createVisualShape(
            p.GEOM_BOX, halfExtents=box_half_extents, rgbaColor=[0.5, 0.5, 0.5, 1]
        )

        # Position and orient the plane
        # Rotate around Y axis by incline_angle
        orientation = p.getQuaternionFromEuler([0, incline_angle, 0])

        plane_id = p.createMultiBody(
            baseMass=0,  # Static
            baseCollisionShapeIndex=plane_collision,
            baseVisualShapeIndex=plane_visual,
            basePosition=[0, 0, 0.5],  # Center position
            baseOrientation=orientation,
        )

        self._auxiliary_objects.append(plane_id)

    def cleanup(self) -> None:
        """Remove all created objects."""
        if not PYBULLET_AVAILABLE:
            return

        # Remove constraint if exists
        if self.constraint_id is not None:
            p.removeConstraint(self.constraint_id)
            self.constraint_id = None

        # Remove auxiliary objects
        for obj_id in self._auxiliary_objects:
            p.removeBody(obj_id)
        self._auxiliary_objects.clear()

        # Remove main object
        if self.object_id is not None:
            p.removeBody(self.object_id)
            self.object_id = None


# Scenario metadata for reference
SCENARIOS = {
    "grasp_ball": {
        "description": "Ball falls, gets grasped by constraint, then released",
        "phases": ["free_fall", "grasped", "released"],
        "duration": 3.0,
    },
    "throw_catch": {
        "description": "Ball thrown with initial velocity, follows parabolic trajectory",
        "phases": ["throw", "flight", "landing"],
        "duration": 2.0,
    },
    "roll_incline": {
        "description": "Ball rolls down an inclined plane",
        "phases": ["start", "rolling", "stop"],
        "duration": 2.0,
    },
}

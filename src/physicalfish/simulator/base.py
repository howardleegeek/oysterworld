"""Physics simulator protocol and base interfaces."""

from typing import Protocol, runtime_checkable

from physicalfish.models import FrameData, PhysicsParams, TrajectoryData


@runtime_checkable
class PhysicsSimulator(Protocol):
    """Protocol for physics simulators.

    Implementations must provide:
    - configure(): Set up simulation parameters
    - run_scenario(): Execute a scenario and return trajectory data
    - reset(): Reset simulation state
    - close(): Clean up resources
    """

    def configure(self, params: PhysicsParams) -> None:
        """Configure simulation parameters.

        Args:
            params: Physics parameters (gravity, mass, friction, etc.)
        """
        ...

    def run_scenario(self, scenario: str, duration: float) -> TrajectoryData:
        """Run a physics scenario and return trajectory data.

        Args:
            scenario: Name of the scenario to run (e.g., "grasp_ball", "throw_catch")
            duration: Simulation duration in seconds

        Returns:
            TrajectoryData with frames recorded at 30Hz
        """
        ...

    def reset(self) -> None:
        """Reset simulation to initial state."""
        ...

    def close(self) -> None:
        """Clean up resources and close simulator."""
        ...

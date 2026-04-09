"""Exception hierarchy for PhysicalFish."""


class PhysicalFishError(Exception):
    """Base exception."""


class SimulationError(PhysicalFishError):
    """Raised when physics simulation fails."""


class SimulationTimeoutError(SimulationError):
    """Raised when simulation exceeds time limit."""


class VerificationError(PhysicalFishError):
    """Raised when verification encounters an error."""


class InsufficientDataError(VerificationError):
    """Raised when not enough frames for verification."""


class OptimizationError(PhysicalFishError):
    """Raised when optimization fails."""


class ConvergenceError(OptimizationError):
    """Raised when optimization fails to converge."""


class ConfigurationError(PhysicalFishError):
    """Raised for invalid configuration."""

"""Centralized parameter space definitions for physics optimization."""

from typing import Dict, Tuple


# Parameter space: (name, min, max)
PHYSICS_PARAM_SPACE: list[tuple[str, float, float]] = [
    ("gravity", 8.0, 11.0),
    ("mass", 0.1, 2.0),
    ("friction", 0.1, 1.0),
    ("restitution", 0.0, 1.0),
    ("linear_damping", 0.0, 1.0),
    ("angular_damping", 0.0, 1.0),
]

# Mapping for legacy parameter names (from old bayesian_optimizer.py)
LEGACY_PARAM_MAPPING: Dict[str, str] = {
    "gravity_magnitude": "gravity",
    "object_mass": "mass",
    "object_friction": "friction",
    "object_bounce": "restitution",
}


def get_bounds_for(name: str) -> Tuple[float, float]:
    """Get bounds for a specific parameter by name.

    Args:
        name: Parameter name (supports both new and legacy names)

    Returns:
        Tuple of (min, max) bounds

    Raises:
        ValueError: If parameter name is not recognized
    """
    # Map legacy names to new names
    canonical_name = LEGACY_PARAM_MAPPING.get(name, name)

    for param_name, low, high in PHYSICS_PARAM_SPACE:
        if param_name == canonical_name:
            return (low, high)

    raise ValueError(f"Unknown parameter: {name}")


def get_all_bounds() -> Dict[str, Tuple[float, float]]:
    """Get all parameter bounds as a dictionary.

    Returns:
        Dictionary mapping parameter names to (min, max) tuples
    """
    return {name: (low, high) for name, low, high in PHYSICS_PARAM_SPACE}


def get_param_names() -> list[str]:
    """Get list of all parameter names."""
    return [name for name, _, _ in PHYSICS_PARAM_SPACE]

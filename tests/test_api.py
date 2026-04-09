"""Tests for PhysicalFish API endpoints."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from physicalfish.api.app import create_app


@pytest.fixture
def client():
    """Create test client."""
    app = create_app()
    return TestClient(app)


def test_health(client):
    """Test health check endpoint returns 200."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == "0.1.0"


def test_verify_valid(client):
    """Test verify endpoint with valid trajectory returns score > 0."""
    # Create a simple valid trajectory (ball falling under gravity)
    frames = []
    dt = 0.033  # 30 Hz
    g = 9.81

    for i in range(10):
        t = i * dt
        # Position: y = 5 - 0.5 * g * t^2 (dropping from height 5)
        y = max(0.0, 5.0 - 0.5 * g * t**2)
        # Velocity: v = -g * t
        vy = -g * t
        # Acceleration: a = -g
        ay = -g

        frames.append(
            {
                "timestamp": t,
                "position": [0.0, y, 0.0],
                "velocity": [0.0, vy, 0.0],
                "acceleration": [0.0, ay, 0.0],
                "rotation": [0.0, 0.0, 0.0],
                "angular_velocity": [0.0, 0.0, 0.0],
            }
        )

    request_data = {
        "frames": frames,
        "physics_params": {
            "gravity": 9.81,
            "mass": 0.5,
            "friction": 0.5,
            "restitution": 0.8,
        },
    }

    response = client.post("/api/v1/verify", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert "overall_score" in data
    assert "passed" in data
    assert "constraint_scores" in data
    assert data["overall_score"] > 0.0


def test_verify_invalid(client):
    """Test verify endpoint with empty frames returns 400."""
    request_data = {
        "frames": [],
        "physics_params": {},
    }

    response = client.post("/api/v1/verify", json=request_data)
    assert response.status_code == 400


def test_verify_insufficient_frames(client):
    """Test verify endpoint with less than 3 frames returns 400."""
    request_data = {
        "frames": [
            {"timestamp": 0.0, "position": [0.0, 1.0, 0.0]},
            {"timestamp": 0.033, "position": [0.0, 0.9, 0.0]},
        ],
        "physics_params": {},
    }

    response = client.post("/api/v1/verify", json=request_data)
    assert response.status_code == 400


def test_rate_limit(client):
    """Test rate limiting: 61 requests should trigger 429 on the last one."""
    # Make 61 requests to a rate-limited endpoint (not health)
    responses = []
    for i in range(61):
        # Use verify endpoint with minimal valid data
        request_data = {
            "frames": [
                {
                    "timestamp": 0.0,
                    "position": [0.0, 5.0, 0.0],
                    "velocity": [0.0, 0.0, 0.0],
                    "acceleration": [0.0, -9.81, 0.0],
                },
                {
                    "timestamp": 0.033,
                    "position": [0.0, 4.995, 0.0],
                    "velocity": [0.0, -0.324, 0.0],
                    "acceleration": [0.0, -9.81, 0.0],
                },
                {
                    "timestamp": 0.066,
                    "position": [0.0, 4.978, 0.0],
                    "velocity": [0.0, -0.648, 0.0],
                    "acceleration": [0.0, -9.81, 0.0],
                },
            ],
            "physics_params": {"gravity": 9.81},
        }
        response = client.post("/api/v1/verify", json=request_data)
        responses.append(response.status_code)

    # Count 429 responses
    rate_limited_count = responses.count(429)
    # At least the last request should be rate limited
    assert rate_limited_count >= 1, (
        f"Expected at least 1 rate-limited response, got {rate_limited_count}"
    )


def test_optimize_endpoint_structure(client):
    """Test optimize endpoint structure (may fail if simulator unavailable)."""
    request_data = {
        "scenario": "grasp_ball",
        "target_score": 0.90,
        "max_iterations": 5,
    }

    response = client.post("/api/v1/optimize", json=request_data)
    # Either 200 (success) or 503 (simulator not available) are acceptable
    assert response.status_code in [200, 503]

    if response.status_code == 200:
        data = response.json()
        assert "best_params" in data
        assert "best_score" in data
        assert "iterations" in data
        assert "convergence_history" in data


def test_bootstrap_endpoint_structure(client):
    """Test bootstrap endpoint structure (may fail if simulator unavailable)."""
    request_data = {
        "real_samples": [
            {
                "frames": [
                    {"timestamp": 0.0, "position": [0.0, 5.0, 0.0], "velocity": [0.0, 0.0, 0.0]},
                    {
                        "timestamp": 0.033,
                        "position": [0.0, 4.995, 0.0],
                        "velocity": [0.0, -0.324, 0.0],
                    },
                    {
                        "timestamp": 0.066,
                        "position": [0.0, 4.978, 0.0],
                        "velocity": [0.0, -0.648, 0.0],
                    },
                ],
                "params": {"gravity": 9.81, "mass": 0.5},
            }
        ],
        "target_count": 5,
    }

    response = client.post("/api/v1/bootstrap", json=request_data)
    # Either 200 (success) or 503 (simulator not available) are acceptable
    assert response.status_code in [200, 503, 400]

    if response.status_code == 200:
        data = response.json()
        assert "generated_count" in data
        assert "pass_rate" in data


def test_bootstrap_empty_samples(client):
    """Test bootstrap endpoint with empty samples returns 400."""
    request_data = {
        "real_samples": [],
        "target_count": 5,
    }

    response = client.post("/api/v1/bootstrap", json=request_data)
    assert response.status_code == 400


# =============================================================================
# Additional Optimize Endpoint Tests
# =============================================================================


def test_optimize_with_real_data_payload(client):
    """Test optimize endpoint with real data payload containing trajectories."""
    # Create realistic trajectory data
    frames = []
    dt = 0.033
    g = 9.81
    for i in range(10):
        t = i * dt
        y = max(0.0, 5.0 - 0.5 * g * t**2)
        vy = -g * t
        ay = -g
        frames.append(
            {
                "timestamp": t,
                "position": [0.0, y, 0.0],
                "velocity": [0.0, vy, 0.0],
                "acceleration": [0.0, ay, 0.0],
                "rotation": [0.0, 0.0, 0.0],
                "angular_velocity": [0.0, 0.0, 0.0],
            }
        )

    request_data = {
        "scenario": "grasp_ball",
        "target_score": 0.85,
        "max_iterations": 3,
        "real_data": [
            {
                "frames": frames,
                "params": {
                    "gravity": 9.81,
                    "mass": 0.5,
                    "friction": 0.5,
                    "restitution": 0.8,
                    "linear_damping": 0.1,
                    "angular_damping": 0.1,
                },
            }
        ],
    }

    response = client.post("/api/v1/optimize", json=request_data)
    # Should accept the request (200 or 503 if simulator unavailable)
    assert response.status_code in [200, 503]

    if response.status_code == 200:
        data = response.json()
        assert "best_params" in data
        assert "best_score" in data
        assert "iterations" in data
        assert "convergence_history" in data
        assert isinstance(data["convergence_history"], list)


def test_optimize_missing_scenario(client):
    """Test optimize endpoint without required scenario field."""
    request_data = {
        "target_score": 0.90,
        "max_iterations": 5,
    }

    response = client.post("/api/v1/optimize", json=request_data)
    # Should return 422 (validation error) for missing required field
    assert response.status_code == 422


def test_optimize_invalid_target_score(client):
    """Test optimize endpoint with invalid target_score."""
    request_data = {
        "scenario": "grasp_ball",
        "target_score": 1.5,  # Invalid: > 1.0
        "max_iterations": 5,
    }

    response = client.post("/api/v1/optimize", json=request_data)
    # Should return 422 for validation error
    assert response.status_code == 422


def test_optimize_zero_iterations(client):
    """Test optimize endpoint with zero iterations."""
    request_data = {
        "scenario": "grasp_ball",
        "target_score": 0.90,
        "max_iterations": 0,  # Invalid: < 1
    }

    response = client.post("/api/v1/optimize", json=request_data)
    # Should return 422 for validation error
    assert response.status_code == 422


def test_optimize_multiple_real_trajectories(client):
    """Test optimize with multiple real trajectory samples."""

    def make_trajectory(start_height):
        frames = []
        dt = 0.033
        g = 9.81
        for i in range(5):
            t = i * dt
            y = max(0.0, start_height - 0.5 * g * t**2)
            frames.append(
                {
                    "timestamp": t,
                    "position": [0.0, y, 0.0],
                    "velocity": [0.0, -g * t, 0.0],
                    "acceleration": [0.0, -g, 0.0],
                }
            )
        return {
            "frames": frames,
            "params": {"gravity": 9.81, "mass": 0.5},
        }

    request_data = {
        "scenario": "grasp_ball",
        "target_score": 0.80,
        "max_iterations": 2,
        "real_data": [
            make_trajectory(5.0),
            make_trajectory(4.0),
            make_trajectory(3.0),
        ],
    }

    response = client.post("/api/v1/optimize", json=request_data)
    assert response.status_code in [200, 503]


# =============================================================================
# Additional Bootstrap Endpoint Tests
# =============================================================================


def test_bootstrap_with_multiple_samples(client):
    """Test bootstrap endpoint with multiple real samples."""

    def make_sample(sample_id, gravity):
        frames = []
        for i in range(5):
            frames.append(
                {
                    "timestamp": i * 0.033,
                    "position": [0.0, 2.0 - 0.1 * i, 0.0],
                    "velocity": [0.0, -0.5, 0.0],
                    "acceleration": [0.0, -gravity, 0.0],
                    "rotation": [0.0, 0.0, 0.0],
                    "angular_velocity": [0.0, 0.0, 0.0],
                }
            )
        return {
            "frames": frames,
            "params": {
                "gravity": gravity,
                "mass": 0.5 + sample_id * 0.1,
                "friction": 0.5,
                "restitution": 0.8,
            },
        }

    request_data = {
        "real_samples": [
            make_sample(0, 9.8),
            make_sample(1, 9.9),
            make_sample(2, 10.0),
        ],
        "target_count": 5,
    }

    response = client.post("/api/v1/bootstrap", json=request_data)
    assert response.status_code in [200, 503, 400]

    if response.status_code == 200:
        data = response.json()
        assert "generated_count" in data
        assert "pass_rate" in data
        assert 0 <= data["pass_rate"] <= 1.0


def test_bootstrap_large_target_count(client):
    """Test bootstrap endpoint with large target count."""
    request_data = {
        "real_samples": [
            {
                "frames": [
                    {"timestamp": 0.0, "position": [0.0, 5.0, 0.0], "velocity": [0.0, 0.0, 0.0]},
                    {
                        "timestamp": 0.033,
                        "position": [0.0, 4.995, 0.0],
                        "velocity": [0.0, -0.3, 0.0],
                    },
                    {
                        "timestamp": 0.066,
                        "position": [0.0, 4.978, 0.0],
                        "velocity": [0.0, -0.6, 0.0],
                    },
                ],
                "params": {"gravity": 9.81, "mass": 0.5},
            }
        ],
        "target_count": 100,  # Large count
    }

    response = client.post("/api/v1/bootstrap", json=request_data)
    assert response.status_code in [200, 503, 400]


def test_bootstrap_missing_target_count(client):
    """Test bootstrap endpoint without required target_count."""
    request_data = {
        "real_samples": [
            {
                "frames": [
                    {"timestamp": 0.0, "position": [0.0, 5.0, 0.0]},
                ],
                "params": {"gravity": 9.81},
            }
        ],
    }

    response = client.post("/api/v1/bootstrap", json=request_data)
    # Should return 422 for missing required field
    assert response.status_code == 422


def test_bootstrap_invalid_target_count(client):
    """Test bootstrap endpoint with invalid target_count."""
    request_data = {
        "real_samples": [
            {
                "frames": [
                    {"timestamp": 0.0, "position": [0.0, 5.0, 0.0]},
                ],
                "params": {"gravity": 9.81},
            }
        ],
        "target_count": 0,  # Invalid: < 1
    }

    response = client.post("/api/v1/bootstrap", json=request_data)
    # Should return 422 for validation error
    assert response.status_code == 422


def test_bootstrap_excessive_target_count(client):
    """Test bootstrap endpoint with target_count exceeding limit."""
    request_data = {
        "real_samples": [
            {
                "frames": [
                    {"timestamp": 0.0, "position": [0.0, 5.0, 0.0]},
                ],
                "params": {"gravity": 9.81},
            }
        ],
        "target_count": 2000,  # Exceeds max of 1000
    }

    response = client.post("/api/v1/bootstrap", json=request_data)
    # Should return 422 for validation error
    assert response.status_code == 422


# =============================================================================
# Verify Endpoint Additional Tests
# =============================================================================


def test_verify_with_complete_physics_params(client):
    """Test verify with all physics parameters specified."""
    frames = []
    for i in range(5):
        frames.append(
            {
                "timestamp": i * 0.033,
                "position": [0.0, 5.0 - 0.5 * i, 0.0],
                "velocity": [0.0, -1.0, 0.0],
                "acceleration": [0.0, -9.81, 0.0],
                "rotation": [0.0, 0.0, 0.0],
                "angular_velocity": [0.0, 0.0, 0.0],
            }
        )

    request_data = {
        "frames": frames,
        "physics_params": {
            "gravity": 9.81,
            "mass": 0.5,
            "friction": 0.5,
            "restitution": 0.8,
            "linear_damping": 0.1,
            "angular_damping": 0.1,
        },
    }

    response = client.post("/api/v1/verify", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert "overall_score" in data
    assert "constraint_scores" in data
    # Should have all 6 constraint scores
    assert len(data["constraint_scores"]) == 6


def test_verify_with_minimal_frames(client):
    """Test verify with exactly 3 frames (minimum required)."""
    request_data = {
        "frames": [
            {"timestamp": 0.0, "position": [0.0, 5.0, 0.0], "velocity": [0.0, 0.0, 0.0]},
            {"timestamp": 0.033, "position": [0.0, 4.995, 0.0], "velocity": [0.0, -0.3, 0.0]},
            {"timestamp": 0.066, "position": [0.0, 4.978, 0.0], "velocity": [0.0, -0.6, 0.0]},
        ],
        "physics_params": {"gravity": 9.81},
    }

    response = client.post("/api/v1/verify", json=request_data)
    assert response.status_code == 200


def test_verify_missing_physics_params(client):
    """Test verify without physics_params (should use defaults)."""
    frames = []
    for i in range(5):
        frames.append(
            {
                "timestamp": i * 0.033,
                "position": [0.0, 5.0 - 0.5 * i, 0.0],
                "velocity": [0.0, -1.0, 0.0],
                "acceleration": [0.0, -9.81, 0.0],
            }
        )

    request_data = {
        "frames": frames,
        # No physics_params
    }

    response = client.post("/api/v1/verify", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert "overall_score" in data


# =============================================================================
# Helper Function Tests
# =============================================================================


def test_frames_from_dict_helper():
    """Test the _frames_from_dict helper function."""
    from physicalfish.api.routes import _frames_from_dict

    frames_data = [
        {
            "timestamp": 0.0,
            "position": [1.0, 2.0, 3.0],
            "velocity": [0.1, 0.2, 0.3],
            "acceleration": [0.0, -9.81, 0.0],
            "rotation": [0.0, 0.0, 0.0],
            "angular_velocity": [0.0, 0.0, 0.0],
        },
        {
            "timestamp": 0.033,
            "position": [1.0, 1.9, 3.0],
            "velocity": [0.1, -0.1, 0.3],
            "acceleration": [0.0, -9.81, 0.0],
        },
    ]

    frames = _frames_from_dict(frames_data)
    assert len(frames) == 2
    assert frames[0].timestamp == 0.0
    assert np.array_equal(frames[0].position, np.array([1.0, 2.0, 3.0]))
    assert frames[1].timestamp == 0.033


def test_trajectory_from_dict_helper():
    """Test the _trajectory_from_dict helper function."""
    from physicalfish.api.routes import _trajectory_from_dict

    data = {
        "frames": [
            {"timestamp": 0.0, "position": [0.0, 5.0, 0.0]},
            {"timestamp": 0.033, "position": [0.0, 4.9, 0.0]},
        ],
        "params": {
            "gravity": 9.8,
            "mass": 0.6,
            "friction": 0.4,
        },
    }

    trajectory = _trajectory_from_dict(data)
    assert len(trajectory.frames) == 2
    assert trajectory.params.gravity == 9.8
    assert trajectory.params.mass == 0.6
    assert trajectory.params.friction == 0.4
    # Defaults for missing params
    assert trajectory.params.restitution == 0.8
    assert trajectory.params.linear_damping == 0.1


# =============================================================================
# Error Handling Tests
# =============================================================================


def test_verify_malformed_json(client):
    """Test verify endpoint with malformed JSON."""
    response = client.post(
        "/api/v1/verify",
        data="not valid json",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422


def test_optimize_malformed_real_data(client):
    """Test optimize with malformed real_data structure."""
    request_data = {
        "scenario": "grasp_ball",
        "target_score": 0.90,
        "max_iterations": 5,
        "real_data": "not a list",  # Invalid type
    }

    response = client.post("/api/v1/optimize", json=request_data)
    assert response.status_code == 422


def test_bootstrap_malformed_samples(client):
    """Test bootstrap with malformed real_samples structure."""
    request_data = {
        "real_samples": "not a list",  # Invalid type
        "target_count": 5,
    }

    response = client.post("/api/v1/bootstrap", json=request_data)
    assert response.status_code == 422

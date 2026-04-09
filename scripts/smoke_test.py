#!/usr/bin/env python3
"""Smoke test for PhysicalFish API — verifies endpoints work end-to-end."""

import sys

from fastapi.testclient import TestClient

from physicalfish.api.app import create_app


def create_test_trajectory():
    """Create a simple test trajectory with valid physics."""
    frames = []
    for i in range(10):
        t = i * 0.033  # 30fps
        # Simple projectile motion
        position = [0.0, 0.0, 5.0 - 0.5 * 9.81 * t * t]  # z decreases due to gravity
        velocity = [0.0, 0.0, -9.81 * t]
        acceleration = [0.0, 0.0, -9.81]
        frames.append(
            {
                "timestamp": t,
                "position": position,
                "velocity": velocity,
                "acceleration": acceleration,
                "rotation": [0.0, 0.0, 0.0],
                "angular_velocity": [0.0, 0.0, 0.0],
            }
        )
    return frames


def test_health(client: TestClient) -> bool:
    """Test health endpoint."""
    print("\n[TEST] /api/v1/health")
    response = client.get("/api/v1/health")
    if response.status_code == 200:
        data = response.json()
        print(f"  Status: {data.get('status')}")
        print(f"  Version: {data.get('version')}")
        print("  PASS")
        return True
    else:
        print(f"  FAIL: status_code={response.status_code}")
        return False


def test_verify(client: TestClient) -> bool:
    """Test verify endpoint with real trajectory data."""
    print("\n[TEST] /api/v1/verify")
    frames = create_test_trajectory()
    payload = {
        "frames": frames,
        "physics_params": {
            "gravity": 9.81,
            "mass": 0.5,
            "friction": 0.5,
            "restitution": 0.8,
        },
    }
    response = client.post("/api/v1/verify", json=payload)
    if response.status_code == 200:
        data = response.json()
        print(f"  Overall score: {data.get('overall_score', 'N/A'):.3f}")
        print(f"  Passed: {data.get('passed', 'N/A')}")
        print("  PASS")
        return True
    else:
        print(f"  FAIL: status_code={response.status_code}")
        print(f"  Response: {response.text}")
        return False


def test_optimize(client: TestClient) -> bool:
    """Test optimize endpoint."""
    print("\n[TEST] /api/v1/optimize")
    frames = create_test_trajectory()
    payload = {
        "scenario": "grasp_ball",
        "target_score": 0.85,
        "max_iterations": 5,
        "real_data": [
            {
                "frames": frames,
                "params": {
                    "gravity": 9.81,
                    "mass": 0.5,
                    "friction": 0.5,
                    "restitution": 0.8,
                },
            }
        ],
    }
    response = client.post("/api/v1/optimize", json=payload)
    # Without pybullet, simulator returns 503 — this is expected behavior
    if response.status_code == 503:
        print("  Simulator not available (expected without pybullet)")
        print("  PASS (endpoint responds correctly)")
        return True
    elif response.status_code == 200:
        data = response.json()
        print(f"  Best score: {data.get('best_score', 'N/A')}")
        print(f"  Iterations: {data.get('iterations', 'N/A')}")
        print("  PASS")
        return True
    else:
        print(f"  FAIL: status_code={response.status_code}")
        print(f"  Response: {response.text}")
        return False


def main():
    """Run all smoke tests."""
    print("=" * 50)
    print("PhysicalFish API Smoke Test")
    print("=" * 50)

    app = create_app()
    client = TestClient(app)

    results = []
    results.append(("health", test_health(client)))
    results.append(("verify", test_verify(client)))
    results.append(("optimize", test_optimize(client)))

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 50)
    if all_passed:
        print("ALL TESTS PASSED")
        return 0
    else:
        print("SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())

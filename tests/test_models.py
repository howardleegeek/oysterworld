"""Tests for unified data models."""

import numpy as np
import pytest

from physicalfish.models import FrameData, PhysicsParams, TrajectoryData, VerificationResult


class TestFrameData:
    def test_creation(self, make_frame):
        f = make_frame(t=0.5, y=1.0, vy=-4.9)
        assert f.timestamp == 0.5
        assert f.height == 1.0
        assert abs(f.speed - 4.9) < 0.01

    def test_frozen(self, make_frame):
        f = make_frame()
        with pytest.raises(AttributeError):
            f.timestamp = 1.0


class TestPhysicsParams:
    def test_defaults(self):
        p = PhysicsParams()
        assert p.gravity == 9.81
        assert p.mass == 0.5

    def test_to_dict(self):
        p = PhysicsParams(gravity=10.0)
        d = p.to_dict()
        assert d["gravity"] == 10.0
        assert len(d) == 6


class TestTrajectoryData:
    def test_duration(self, freefall_trajectory):
        assert freefall_trajectory.duration > 0

    def test_num_frames(self, freefall_trajectory):
        assert freefall_trajectory.num_frames == 90

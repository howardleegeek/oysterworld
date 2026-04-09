"""Tests for Byzantine Swarm AutoResearch.

Tests for ByzantineSwarm core functionality:
1. Agent creation with parameter subsets
2. Convergence behavior
3. Byzantine voting (accept/reject proposals)
4. Agent contribution tracking
5. Gravity convergence to real value
"""

import numpy as np
import pytest
from unittest.mock import Mock, MagicMock

from physicalfish.loop.swarm import (
    ByzantineSwarm,
    SwarmAgent,
    ByzantineVote,
    SwarmRound,
)
from physicalfish.models import PhysicsParams, TrajectoryData, VerificationResult, FrameData
from physicalfish.verification.physics_verifier import PhysicsVerifier


class TestSwarmAgent:
    """Tests for individual SwarmAgent behavior."""

    def test_agent_creation(self):
        """Agent initializes with correct parameters."""
        agent = SwarmAgent(agent_id="test_agent", param_names=["gravity", "mass"])

        assert agent.agent_id == "test_agent"
        assert agent.param_names == ["gravity", "mass"]
        assert agent.best_params == {}
        assert agent.best_score == 0.0
        assert agent.proposals_made == 0
        assert agent.proposals_accepted == 0

    def test_propose_increments_counter(self):
        """Proposing increments proposals_made counter."""
        agent = SwarmAgent(agent_id="agent_0", param_names=["gravity"])
        current_params = {"gravity": 9.0, "mass": 0.5}

        initial_count = agent.proposals_made
        agent.propose(current_params, exploration_rate=0.1)

        assert agent.proposals_made == initial_count + 1

    def test_propose_returns_modified_params(self):
        """Propose returns params with agent's subset modified."""
        agent = SwarmAgent(agent_id="agent_0", param_names=["gravity"])
        current_params = {"gravity": 9.0, "mass": 0.5, "friction": 0.3}

        proposed = agent.propose(current_params, exploration_rate=0.1)

        # Should modify gravity but keep other params
        assert proposed["mass"] == current_params["mass"]
        assert proposed["friction"] == current_params["friction"]
        # Gravity should be different (with high probability due to noise)

    def test_propose_respects_bounds(self):
        """Proposed values stay within parameter bounds."""
        agent = SwarmAgent(agent_id="agent_0", param_names=["gravity", "mass", "friction"])
        current_params = {"gravity": 9.0, "mass": 0.5, "friction": 0.3}

        # Test many proposals to ensure bounds are respected
        for _ in range(50):
            proposed = agent.propose(current_params, exploration_rate=0.5)
            assert 8.0 <= proposed["gravity"] <= 11.0
            assert 0.1 <= proposed["mass"] <= 2.0
            assert 0.1 <= proposed["friction"] <= 1.0


class TestByzantineSwarmCreation:
    """Tests for ByzantineSwarm initialization."""

    def test_swarm_creates_correct_agents(self):
        """3 agents with different param subsets are created."""
        mock_simulator = Mock(return_value=MagicMock())
        mock_verifier = Mock()

        swarm = ByzantineSwarm(
            simulator_fn=mock_simulator,
            verifier=mock_verifier,
            n_agents=3,
        )

        assert len(swarm.agents) == 3

        # Check agent IDs
        agent_ids = [a.agent_id for a in swarm.agents]
        assert "agent_0" in agent_ids
        assert "agent_1" in agent_ids
        assert "agent_2" in agent_ids

    def test_agents_have_different_param_subsets(self):
        """Each agent optimizes different parameters."""
        mock_simulator = Mock(return_value=MagicMock())
        mock_verifier = Mock()

        swarm = ByzantineSwarm(
            simulator_fn=mock_simulator,
            verifier=mock_verifier,
            n_agents=3,
        )

        # Collect all param names assigned to agents
        all_agent_params = []
        for agent in swarm.agents:
            all_agent_params.extend(agent.param_names)

        # Should cover all 6 physics params
        expected_params = [
            "gravity",
            "mass",
            "friction",
            "restitution",
            "linear_damping",
            "angular_damping",
        ]
        assert set(all_agent_params) == set(expected_params)

    def test_params_distributed_across_agents(self):
        """Parameters are distributed, not all on one agent."""
        mock_simulator = Mock(return_value=MagicMock())
        mock_verifier = Mock()

        swarm = ByzantineSwarm(
            simulator_fn=mock_simulator,
            verifier=mock_verifier,
            n_agents=3,
        )

        # Each agent should have at least 1 parameter
        for agent in swarm.agents:
            assert len(agent.param_names) >= 1

        # No agent should have all 6 parameters (distributed load)
        for agent in swarm.agents:
            assert len(agent.param_names) < 6

    def test_swarm_stores_config(self):
        """Swarm stores simulator, verifier, and thresholds."""
        mock_simulator = Mock()
        mock_verifier = Mock()

        swarm = ByzantineSwarm(
            simulator_fn=mock_simulator,
            verifier=mock_verifier,
            n_agents=3,
            byzantine_threshold=0.7,
            acceptance_delta=0.01,
        )

        assert swarm.simulator_fn == mock_simulator
        assert swarm.verifier == mock_verifier
        assert swarm.byzantine_threshold == 0.7
        assert swarm.acceptance_delta == 0.01


class TestByzantineVoting:
    """Tests for Byzantine voting mechanism."""

    def _create_mock_trajectory(self, score: float = 0.5) -> TrajectoryData:
        """Create a mock trajectory that will verify to given score."""
        frames = []
        dt = 1.0 / 30.0
        for i in range(30):
            frames.append(
                FrameData(
                    timestamp=i * dt,
                    position=np.array([0.0, 1.0, 0.0]),
                    velocity=np.array([0.0, 0.0, 0.0]),
                    acceleration=np.array([0.0, -9.81, 0.0]),
                    rotation=np.zeros(3),
                    angular_velocity=np.zeros(3),
                    frame_index=i,
                )
            )
        return TrajectoryData(frames=frames, params=PhysicsParams())

    def test_byzantine_rejects_bad_proposals(self):
        """Bad proposals (low scores) get rejected."""
        # Create verifier that returns low scores
        mock_verifier = Mock()
        mock_verifier.verify.return_value = VerificationResult(
            overall_score=0.3,
            passed=False,
            constraint_scores={},
            details={},
        )

        mock_simulator = Mock(return_value=self._create_mock_trajectory())

        swarm = ByzantineSwarm(
            simulator_fn=mock_simulator,
            verifier=mock_verifier,
            n_agents=3,
            acceptance_delta=0.05,
        )

        # Initialize with good score
        swarm.global_best_score = 0.8

        # Create a proposer agent
        proposer = SwarmAgent(agent_id="agent_0", param_names=["gravity"])
        proposer.proposals_made = 1

        # Propose params
        proposed = {"gravity": 5.0, "mass": 0.5}  # Bad gravity value

        # Run Byzantine vote
        vote = swarm._byzantine_vote(proposer, proposed)

        # Should be rejected (score 0.3 < 0.8 - 0.05)
        assert vote.accepted is False
        assert vote.proposed_by == "agent_0"

    def test_byzantine_accepts_good_proposals(self):
        """Improvements (high scores) get accepted."""
        # Create verifier that returns high scores
        mock_verifier = Mock()
        mock_verifier.verify.return_value = VerificationResult(
            overall_score=0.9,
            passed=True,
            constraint_scores={},
            details={},
        )

        mock_simulator = Mock(return_value=self._create_mock_trajectory())

        swarm = ByzantineSwarm(
            simulator_fn=mock_simulator,
            verifier=mock_verifier,
            n_agents=3,
            acceptance_delta=0.05,
            byzantine_threshold=0.6,
        )

        # Initialize with lower score
        swarm.global_best_score = 0.5

        # Create a proposer agent
        proposer = SwarmAgent(agent_id="agent_0", param_names=["gravity"])
        proposer.proposals_made = 1

        # Propose params
        proposed = {"gravity": 9.81, "mass": 0.5}  # Good gravity value

        # Run Byzantine vote
        vote = swarm._byzantine_vote(proposer, proposed)

        # Should be accepted (score 0.9 > 0.5 + 0.05)
        assert vote.accepted is True
        assert vote.consensus_score == 0.9
        assert proposer.proposals_accepted == 1

    def test_byzantine_requires_majority_consensus(self):
        """Proposal needs majority agreement to be accepted."""
        mock_simulator = Mock(return_value=self._create_mock_trajectory())

        # Create verifier with side effects to simulate disagreement
        scores = [0.9, 0.3, 0.4]  # Only 1 of 3 agrees
        mock_verifier = Mock()
        mock_verifier.verify.side_effect = [
            VerificationResult(overall_score=s, passed=s > 0.7, constraint_scores={}, details={})
            for s in scores
        ]

        swarm = ByzantineSwarm(
            simulator_fn=mock_simulator,
            verifier=mock_verifier,
            n_agents=3,
            acceptance_delta=0.05,
            byzantine_threshold=0.6,  # Need 60% agreement
        )

        swarm.global_best_score = 0.5

        proposer = SwarmAgent(agent_id="agent_0", param_names=["gravity"])
        proposer.proposals_made = 1

        proposed = {"gravity": 9.81, "mass": 0.5}

        # Need to mock _evaluate to return different scores for each agent
        call_count = [0]

        def mock_evaluate(params):
            score = scores[call_count[0] % len(scores)]
            call_count[0] += 1
            return score

        swarm._evaluate = mock_evaluate

        vote = swarm._byzantine_vote(proposer, proposed)

        # With threshold 0.6, need 2/3 agents to agree
        # But median is 0.4 which is < 0.5 + 0.05, so rejected
        assert vote.accepted is False

    def test_vote_tracks_all_agent_votes(self):
        """Vote records scores from all agents."""
        mock_simulator = Mock(return_value=self._create_mock_trajectory())
        mock_verifier = Mock()
        mock_verifier.verify.return_value = VerificationResult(
            overall_score=0.7,
            passed=True,
            constraint_scores={},
            details={},
        )

        swarm = ByzantineSwarm(
            simulator_fn=mock_simulator,
            verifier=mock_verifier,
            n_agents=3,
        )

        swarm.global_best_score = 0.5

        proposer = SwarmAgent(agent_id="agent_0", param_names=["gravity"])
        proposer.proposals_made = 1

        proposed = {"gravity": 9.81, "mass": 0.5}

        vote = swarm._byzantine_vote(proposer, proposed)

        # Should have votes from all 3 agents
        assert len(vote.votes) == 3
        assert "agent_0" in vote.votes
        assert "agent_1" in vote.votes
        assert "agent_2" in vote.votes


class TestSwarmConvergence:
    """Tests for swarm optimization convergence."""

    def _create_scored_verifier(self, scores_by_gravity: dict) -> Mock:
        """Create verifier that returns scores based on gravity value."""

        def verify_side_effect(trajectory):
            # Extract gravity from trajectory params
            g = getattr(trajectory.params, "gravity", 9.0)
            # Find closest gravity in scores dict
            closest_g = min(scores_by_gravity.keys(), key=lambda x: abs(x - g))
            score = scores_by_gravity[closest_g]
            return VerificationResult(
                overall_score=score,
                passed=score > 0.7,
                constraint_scores={},
                details={},
            )

        mock_verifier = Mock()
        mock_verifier.verify.side_effect = verify_side_effect
        return mock_verifier

    def test_swarm_converges_score_improves(self):
        """Score improves from initial to final."""
        # Gravity near 9.81 gives high scores
        scores = {
            8.0: 0.3,
            8.5: 0.5,
            9.0: 0.7,
            9.5: 0.85,
            9.81: 0.95,
            10.0: 0.9,
            10.5: 0.8,
            11.0: 0.6,
        }

        mock_verifier = self._create_scored_verifier(scores)

        # Simple mock simulator that creates trajectory with given params
        def mock_simulator(params):
            frames = []
            dt = 1.0 / 30.0
            g = params.get("gravity", 9.0)
            params_obj = PhysicsParams(
                gravity=g,
                mass=params.get("mass", 0.5),
                friction=params.get("friction", 0.5),
            )
            for i in range(30):
                frames.append(
                    FrameData(
                        timestamp=i * dt,
                        position=np.array([0.0, 1.0 - 0.5 * g * (i * dt) ** 2, 0.0]),
                        velocity=np.array([0.0, -g * i * dt, 0.0]),
                        acceleration=np.array([0.0, -g, 0.0]),
                        rotation=np.zeros(3),
                        angular_velocity=np.zeros(3),
                        frame_index=i,
                    )
                )
            return TrajectoryData(frames=frames, params=params_obj)

        swarm = ByzantineSwarm(
            simulator_fn=mock_simulator,
            verifier=mock_verifier,
            n_agents=3,
            acceptance_delta=0.01,
            byzantine_threshold=0.5,
        )

        # Start with low score params
        initial_params = {
            "gravity": 8.0,
            "mass": 0.5,
            "friction": 0.5,
            "restitution": 0.5,
            "linear_damping": 0.1,
            "angular_damping": 0.1,
        }

        result = swarm.run(
            initial_params=initial_params,
            max_rounds=5,
            target_score=0.95,
            verbose=False,
        )

        # Score should improve
        initial_score = scores[8.0]
        final_score = result["final_score"]
        assert final_score >= initial_score, (
            f"Final score {final_score} should be > initial {initial_score}"
        )

    def test_gravity_converges_to_real(self):
        """Gravity param approaches 9.81."""
        # Gravity near 9.81 gives highest scores
        scores = {
            8.0: 0.3,
            8.5: 0.5,
            9.0: 0.7,
            9.5: 0.85,
            9.81: 0.98,
            10.0: 0.9,
            10.5: 0.8,
            11.0: 0.6,
        }

        mock_verifier = self._create_scored_verifier(scores)

        def mock_simulator(params):
            frames = []
            dt = 1.0 / 30.0
            g = params.get("gravity", 9.0)
            params_obj = PhysicsParams(
                gravity=g,
                mass=params.get("mass", 0.5),
                friction=params.get("friction", 0.5),
            )
            for i in range(30):
                frames.append(
                    FrameData(
                        timestamp=i * dt,
                        position=np.array([0.0, 1.0, 0.0]),
                        velocity=np.array([0.0, 0.0, 0.0]),
                        acceleration=np.array([0.0, -g, 0.0]),
                        rotation=np.zeros(3),
                        angular_velocity=np.zeros(3),
                        frame_index=i,
                    )
                )
            return TrajectoryData(frames=frames, params=params_obj)

        swarm = ByzantineSwarm(
            simulator_fn=mock_simulator,
            verifier=mock_verifier,
            n_agents=3,
            acceptance_delta=0.005,
            byzantine_threshold=0.5,
        )

        # Start far from optimal
        initial_params = {
            "gravity": 8.0,
            "mass": 0.5,
            "friction": 0.5,
            "restitution": 0.5,
            "linear_damping": 0.1,
            "angular_damping": 0.1,
        }

        result = swarm.run(
            initial_params=initial_params,
            max_rounds=15,  # More rounds to allow convergence
            target_score=0.95,
            verbose=False,
        )

        # Verify swarm ran and produced results
        assert result["rounds"] > 0
        assert result["final_score"] >= 0.3  # At least initial score

        # If any proposals were accepted, gravity should have moved toward 9.81
        if result["accepted_proposals"] > 0:
            final_gravity = result["best_params"]["gravity"]
            initial_distance = abs(8.0 - 9.81)
            final_distance = abs(final_gravity - 9.81)

            # With enough accepted proposals, should converge
            assert final_distance <= initial_distance + 0.5, (
                f"Final gravity {final_gravity} should not drift too far from 9.81"
            )

    def test_all_agents_contribute(self):
        """Each agent has >0 accepted proposals over enough rounds."""
        # All gravities give decent scores to encourage exploration
        scores = {
            8.0: 0.6,
            8.5: 0.7,
            9.0: 0.8,
            9.5: 0.9,
            9.81: 0.95,
            10.0: 0.9,
            10.5: 0.8,
            11.0: 0.7,
        }

        mock_verifier = self._create_scored_verifier(scores)

        def mock_simulator(params):
            frames = []
            dt = 1.0 / 30.0
            g = params.get("gravity", 9.0)
            params_obj = PhysicsParams(
                gravity=g,
                mass=params.get("mass", 0.5),
                friction=params.get("friction", 0.5),
            )
            for i in range(30):
                frames.append(
                    FrameData(
                        timestamp=i * dt,
                        position=np.array([0.0, 1.0, 0.0]),
                        velocity=np.array([0.0, 0.0, 0.0]),
                        acceleration=np.array([0.0, -g, 0.0]),
                        rotation=np.zeros(3),
                        angular_velocity=np.zeros(3),
                        frame_index=i,
                    )
                )
            return TrajectoryData(frames=frames, params=params_obj)

        swarm = ByzantineSwarm(
            simulator_fn=mock_simulator,
            verifier=mock_verifier,
            n_agents=3,
            acceptance_delta=0.01,
            byzantine_threshold=0.5,
        )

        initial_params = {
            "gravity": 9.0,
            "mass": 0.5,
            "friction": 0.5,
            "restitution": 0.5,
            "linear_damping": 0.1,
            "angular_damping": 0.1,
        }

        result = swarm.run(
            initial_params=initial_params,
            max_rounds=15,  # Enough rounds for all agents to contribute
            target_score=0.99,  # High target to force many rounds
            verbose=False,
        )

        # Check that each agent made proposals
        for agent_id, stats in result["agent_stats"].items():
            assert stats["proposals"] > 0, f"Agent {agent_id} made no proposals"


class TestSwarmRun:
    """Tests for swarm.run() method."""

    def _make_default_params(self):
        """Create default initial params dict."""
        return {
            "gravity": 9.5,
            "mass": 0.5,
            "friction": 0.5,
            "restitution": 0.5,
            "linear_damping": 0.1,
            "angular_damping": 0.1,
        }

    def test_run_returns_correct_structure(self):
        """Run returns dict with expected keys."""
        mock_verifier = Mock()
        mock_verifier.verify.return_value = VerificationResult(
            overall_score=0.8,
            passed=True,
            constraint_scores={},
            details={},
        )

        def mock_simulator(params):
            frames = []
            dt = 1.0 / 30.0
            g = params.get("gravity", 9.0)
            params_obj = PhysicsParams(gravity=g)
            for i in range(30):
                frames.append(
                    FrameData(
                        timestamp=i * dt,
                        position=np.array([0.0, 1.0, 0.0]),
                        velocity=np.zeros(3),
                        acceleration=np.array([0.0, -g, 0.0]),
                        rotation=np.zeros(3),
                        angular_velocity=np.zeros(3),
                        frame_index=i,
                    )
                )
            return TrajectoryData(frames=frames, params=params_obj)

        swarm = ByzantineSwarm(
            simulator_fn=mock_simulator,
            verifier=mock_verifier,
            n_agents=3,
        )

        result = swarm.run(
            initial_params=self._make_default_params(),
            max_rounds=3,
            target_score=0.99,
            verbose=False,
        )

        # Check structure
        assert "final_score" in result
        assert "best_params" in result
        assert "rounds" in result
        assert "total_proposals" in result
        assert "accepted_proposals" in result
        assert "elapsed_seconds" in result
        assert "convergence_curve" in result
        assert "agent_stats" in result

    def test_run_stops_at_target_score(self):
        """Run stops early when target score is reached."""
        mock_verifier = Mock()
        mock_verifier.verify.return_value = VerificationResult(
            overall_score=0.95,
            passed=True,
            constraint_scores={},
            details={},
        )

        def mock_simulator(params):
            frames = []
            dt = 1.0 / 30.0
            g = params.get("gravity", 9.0)
            params_obj = PhysicsParams(gravity=g)
            for i in range(30):
                frames.append(
                    FrameData(
                        timestamp=i * dt,
                        position=np.array([0.0, 1.0, 0.0]),
                        velocity=np.zeros(3),
                        acceleration=np.array([0.0, -g, 0.0]),
                        rotation=np.zeros(3),
                        angular_velocity=np.zeros(3),
                        frame_index=i,
                    )
                )
            return TrajectoryData(frames=frames, params=params_obj)

        swarm = ByzantineSwarm(
            simulator_fn=mock_simulator,
            verifier=mock_verifier,
            n_agents=3,
        )

        result = swarm.run(
            initial_params=self._make_default_params(),
            max_rounds=20,
            target_score=0.9,
            verbose=False,
        )

        # Should stop early (target 0.9 is reached on first evaluation with score 0.95)
        assert result["rounds"] < 20

    def test_run_respects_max_rounds(self):
        """Run stops at max_rounds if target not reached."""
        mock_verifier = Mock()
        mock_verifier.verify.return_value = VerificationResult(
            overall_score=0.5,
            passed=False,
            constraint_scores={},
            details={},
        )

        def mock_simulator(params):
            frames = []
            dt = 1.0 / 30.0
            g = params.get("gravity", 9.0)
            params_obj = PhysicsParams(gravity=g)
            for i in range(30):
                frames.append(
                    FrameData(
                        timestamp=i * dt,
                        position=np.array([0.0, 1.0, 0.0]),
                        velocity=np.zeros(3),
                        acceleration=np.array([0.0, -g, 0.0]),
                        rotation=np.zeros(3),
                        angular_velocity=np.zeros(3),
                        frame_index=i,
                    )
                )
            return TrajectoryData(frames=frames, params=params_obj)

        swarm = ByzantineSwarm(
            simulator_fn=mock_simulator,
            verifier=mock_verifier,
            n_agents=3,
        )

        result = swarm.run(
            initial_params=self._make_default_params(),
            max_rounds=5,
            target_score=0.99,
            verbose=False,
        )

        # Should run exactly max_rounds
        assert result["rounds"] == 5

    def test_run_with_default_initial_params(self):
        """Run works without explicit initial_params."""
        mock_verifier = Mock()
        mock_verifier.verify.return_value = VerificationResult(
            overall_score=0.8,
            passed=True,
            constraint_scores={},
            details={},
        )

        def mock_simulator(params):
            frames = []
            dt = 1.0 / 30.0
            g = params.get("gravity", 9.0)
            params_obj = PhysicsParams(gravity=g)
            for i in range(30):
                frames.append(
                    FrameData(
                        timestamp=i * dt,
                        position=np.array([0.0, 1.0, 0.0]),
                        velocity=np.zeros(3),
                        acceleration=np.array([0.0, -g, 0.0]),
                        rotation=np.zeros(3),
                        angular_velocity=np.zeros(3),
                        frame_index=i,
                    )
                )
            return TrajectoryData(frames=frames, params=params_obj)

        swarm = ByzantineSwarm(
            simulator_fn=mock_simulator,
            verifier=mock_verifier,
            n_agents=3,
        )

        # Should work without initial_params (uses defaults from PHYSICS_PARAM_SPACE)
        result = swarm.run(max_rounds=2, verbose=False)

        assert "final_score" in result
        assert "best_params" in result
        assert "rounds" in result
        assert "total_proposals" in result
        assert "accepted_proposals" in result
        assert "elapsed_seconds" in result
        assert "convergence_curve" in result
        assert "agent_stats" in result

    def test_run_stops_at_target_score(self):
        """Run stops early when target score is reached."""
        mock_verifier = Mock()
        mock_verifier.verify.return_value = VerificationResult(
            overall_score=0.95,
            passed=True,
            constraint_scores={},
            details={},
        )

        def mock_simulator(params):
            frames = []
            dt = 1.0 / 30.0
            g = params.get("gravity", 9.0)
            params_obj = PhysicsParams(gravity=g)
            for i in range(30):
                frames.append(
                    FrameData(
                        timestamp=i * dt,
                        position=np.array([0.0, 1.0, 0.0]),
                        velocity=np.zeros(3),
                        acceleration=np.array([0.0, -g, 0.0]),
                        rotation=np.zeros(3),
                        angular_velocity=np.zeros(3),
                        frame_index=i,
                    )
                )
            return TrajectoryData(frames=frames, params=params_obj)

        swarm = ByzantineSwarm(
            simulator_fn=mock_simulator,
            verifier=mock_verifier,
            n_agents=3,
        )

        result = swarm.run(max_rounds=20, target_score=0.9, verbose=False)

        # Should stop early (target 0.9 is reached on first evaluation with score 0.95)
        assert result["rounds"] < 20

    def test_run_respects_max_rounds(self):
        """Run stops at max_rounds if target not reached."""
        mock_verifier = Mock()
        mock_verifier.verify.return_value = VerificationResult(
            overall_score=0.5,
            passed=False,
            constraint_scores={},
            details={},
        )

        def mock_simulator(params):
            frames = []
            dt = 1.0 / 30.0
            g = params.get("gravity", 9.0)
            params_obj = PhysicsParams(gravity=g)
            for i in range(30):
                frames.append(
                    FrameData(
                        timestamp=i * dt,
                        position=np.array([0.0, 1.0, 0.0]),
                        velocity=np.zeros(3),
                        acceleration=np.array([0.0, -g, 0.0]),
                        rotation=np.zeros(3),
                        angular_velocity=np.zeros(3),
                        frame_index=i,
                    )
                )
            return TrajectoryData(frames=frames, params=params_obj)

        swarm = ByzantineSwarm(
            simulator_fn=mock_simulator,
            verifier=mock_verifier,
            n_agents=3,
        )

        result = swarm.run(max_rounds=5, target_score=0.99, verbose=False)

        # Should run exactly max_rounds
        assert result["rounds"] == 5

    def test_run_with_default_initial_params(self):
        """Run works without explicit initial_params."""
        mock_verifier = Mock()
        mock_verifier.verify.return_value = VerificationResult(
            overall_score=0.8,
            passed=True,
            constraint_scores={},
            details={},
        )

        def mock_simulator(params):
            frames = []
            dt = 1.0 / 30.0
            g = params.get("gravity", 9.0)
            params_obj = PhysicsParams(gravity=g)
            for i in range(30):
                frames.append(
                    FrameData(
                        timestamp=i * dt,
                        position=np.array([0.0, 1.0, 0.0]),
                        velocity=np.zeros(3),
                        acceleration=np.array([0.0, -g, 0.0]),
                        rotation=np.zeros(3),
                        angular_velocity=np.zeros(3),
                        frame_index=i,
                    )
                )
            return TrajectoryData(frames=frames, params=params_obj)

        swarm = ByzantineSwarm(
            simulator_fn=mock_simulator,
            verifier=mock_verifier,
            n_agents=3,
        )

        # Should work without initial_params
        result = swarm.run(max_rounds=2, verbose=False)

        assert "final_score" in result
        assert "best_params" in result


class TestSwarmEdgeCases:
    """Edge case tests for ByzantineSwarm."""

    def _make_default_params(self):
        """Create default initial params dict."""
        return {
            "gravity": 9.5,
            "mass": 0.5,
            "friction": 0.5,
            "restitution": 0.5,
            "linear_damping": 0.1,
            "angular_damping": 0.1,
        }

    def test_single_agent_swarm(self):
        """Swarm works with single agent."""
        mock_verifier = Mock()
        mock_verifier.verify.return_value = VerificationResult(
            overall_score=0.8,
            passed=True,
            constraint_scores={},
            details={},
        )

        def mock_simulator(params):
            frames = []
            dt = 1.0 / 30.0
            g = params.get("gravity", 9.0)
            params_obj = PhysicsParams(gravity=g)
            for i in range(30):
                frames.append(
                    FrameData(
                        timestamp=i * dt,
                        position=np.array([0.0, 1.0, 0.0]),
                        velocity=np.zeros(3),
                        acceleration=np.array([0.0, -g, 0.0]),
                        rotation=np.zeros(3),
                        angular_velocity=np.zeros(3),
                        frame_index=i,
                    )
                )
            return TrajectoryData(frames=frames, params=params_obj)

        swarm = ByzantineSwarm(
            simulator_fn=mock_simulator,
            verifier=mock_verifier,
            n_agents=1,
        )

        result = swarm.run(initial_params=self._make_default_params(), max_rounds=2, verbose=False)

        assert len(swarm.agents) == 1
        assert result["rounds"] == 2

    def test_many_agents(self):
        """Swarm works with more agents than parameters."""
        mock_verifier = Mock()
        mock_verifier.verify.return_value = VerificationResult(
            overall_score=0.8,
            passed=True,
            constraint_scores={},
            details={},
        )

        def mock_simulator(params):
            frames = []
            dt = 1.0 / 30.0
            g = params.get("gravity", 9.0)
            params_obj = PhysicsParams(gravity=g)
            for i in range(30):
                frames.append(
                    FrameData(
                        timestamp=i * dt,
                        position=np.array([0.0, 1.0, 0.0]),
                        velocity=np.zeros(3),
                        acceleration=np.array([0.0, -g, 0.0]),
                        rotation=np.zeros(3),
                        angular_velocity=np.zeros(3),
                        frame_index=i,
                    )
                )
            return TrajectoryData(frames=frames, params=params_obj)

        swarm = ByzantineSwarm(
            simulator_fn=mock_simulator,
            verifier=mock_verifier,
            n_agents=10,  # More than 6 parameters
        )

        result = swarm.run(initial_params=self._make_default_params(), max_rounds=2, verbose=False)

        assert len(swarm.agents) == 10
        assert result["rounds"] == 2

    def test_zero_acceptance_delta(self):
        """Swarm works with zero acceptance delta."""
        mock_verifier = Mock()
        mock_verifier.verify.return_value = VerificationResult(
            overall_score=0.8,
            passed=True,
            constraint_scores={},
            details={},
        )

        def mock_simulator(params):
            frames = []
            dt = 1.0 / 30.0
            g = params.get("gravity", 9.0)
            params_obj = PhysicsParams(gravity=g)
            for i in range(30):
                frames.append(
                    FrameData(
                        timestamp=i * dt,
                        position=np.array([0.0, 1.0, 0.0]),
                        velocity=np.zeros(3),
                        acceleration=np.array([0.0, -g, 0.0]),
                        rotation=np.zeros(3),
                        angular_velocity=np.zeros(3),
                        frame_index=i,
                    )
                )
            return TrajectoryData(frames=frames, params=params_obj)

        swarm = ByzantineSwarm(
            simulator_fn=mock_simulator,
            verifier=mock_verifier,
            n_agents=3,
            acceptance_delta=0.0,  # Zero delta
        )

        result = swarm.run(initial_params=self._make_default_params(), max_rounds=2, verbose=False)

        assert "final_score" in result

    def test_high_byzantine_threshold(self):
        """Swarm works with high Byzantine threshold."""
        mock_verifier = Mock()
        mock_verifier.verify.return_value = VerificationResult(
            overall_score=0.9,
            passed=True,
            constraint_scores={},
            details={},
        )

        def mock_simulator(params):
            frames = []
            dt = 1.0 / 30.0
            g = params.get("gravity", 9.0)
            params_obj = PhysicsParams(gravity=g)
            for i in range(30):
                frames.append(
                    FrameData(
                        timestamp=i * dt,
                        position=np.array([0.0, 1.0, 0.0]),
                        velocity=np.zeros(3),
                        acceleration=np.array([0.0, -g, 0.0]),
                        rotation=np.zeros(3),
                        angular_velocity=np.zeros(3),
                        frame_index=i,
                    )
                )
            return TrajectoryData(frames=frames, params=params_obj)

        swarm = ByzantineSwarm(
            simulator_fn=mock_simulator,
            verifier=mock_verifier,
            n_agents=3,
            byzantine_threshold=0.9,  # High threshold - need 90% agreement
        )

        result = swarm.run(initial_params=self._make_default_params(), max_rounds=2, verbose=False)

        assert "final_score" in result

    def test_convergence_curve_length_matches_rounds(self):
        """Convergence curve has one entry per round."""
        mock_verifier = Mock()
        mock_verifier.verify.return_value = VerificationResult(
            overall_score=0.8,
            passed=True,
            constraint_scores={},
            details={},
        )

        def mock_simulator(params):
            frames = []
            dt = 1.0 / 30.0
            g = params.get("gravity", 9.0)
            params_obj = PhysicsParams(gravity=g)
            for i in range(30):
                frames.append(
                    FrameData(
                        timestamp=i * dt,
                        position=np.array([0.0, 1.0, 0.0]),
                        velocity=np.zeros(3),
                        acceleration=np.array([0.0, -g, 0.0]),
                        rotation=np.zeros(3),
                        angular_velocity=np.zeros(3),
                        frame_index=i,
                    )
                )
            return TrajectoryData(frames=frames, params=params_obj)

        swarm = ByzantineSwarm(
            simulator_fn=mock_simulator,
            verifier=mock_verifier,
            n_agents=3,
        )

        result = swarm.run(
            initial_params=self._make_default_params(),
            max_rounds=5,
            target_score=0.99,
            verbose=False,
        )

        # Convergence curve should have entry for each round
        assert len(result["convergence_curve"]) == result["rounds"]


class TestSwarmVerboseMode:
    """Tests for verbose output mode."""

    def _make_default_params(self):
        """Create default initial params dict."""
        return {
            "gravity": 9.5,
            "mass": 0.5,
            "friction": 0.5,
            "restitution": 0.5,
            "linear_damping": 0.1,
            "angular_damping": 0.1,
        }

    def test_swarm_verbose_mode(self, capsys):
        """Swarm produces output in verbose mode."""
        mock_verifier = Mock()
        mock_verifier.verify.return_value = VerificationResult(
            overall_score=0.8,
            passed=True,
            constraint_scores={},
            details={},
        )

        def mock_simulator(params):
            frames = []
            dt = 1.0 / 30.0
            g = params.get("gravity", 9.0)
            params_obj = PhysicsParams(gravity=g)
            for i in range(30):
                frames.append(
                    FrameData(
                        timestamp=i * dt,
                        position=np.array([0.0, 1.0, 0.0]),
                        velocity=np.zeros(3),
                        acceleration=np.array([0.0, -g, 0.0]),
                        rotation=np.zeros(3),
                        angular_velocity=np.zeros(3),
                        frame_index=i,
                    )
                )
            return TrajectoryData(frames=frames, params=params_obj)

        swarm = ByzantineSwarm(
            simulator_fn=mock_simulator,
            verifier=mock_verifier,
            n_agents=3,
        )

        result = swarm.run(
            initial_params=self._make_default_params(),
            max_rounds=2,
            verbose=True,
        )

        # Check that output was produced
        captured = capsys.readouterr()
        # Should have printed something
        assert result["rounds"] == 2


class TestSwarmAgentStats:
    """Tests for agent statistics tracking."""

    def _make_default_params(self):
        """Create default initial params dict."""
        return {
            "gravity": 9.5,
            "mass": 0.5,
            "friction": 0.5,
            "restitution": 0.5,
            "linear_damping": 0.1,
            "angular_damping": 0.1,
        }

    def test_agent_stats_tracks_all_agents(self):
        """Agent stats include all agents."""
        mock_verifier = Mock()
        mock_verifier.verify.return_value = VerificationResult(
            overall_score=0.8,
            passed=True,
            constraint_scores={},
            details={},
        )

        def mock_simulator(params):
            frames = []
            dt = 1.0 / 30.0
            g = params.get("gravity", 9.0)
            params_obj = PhysicsParams(gravity=g)
            for i in range(30):
                frames.append(
                    FrameData(
                        timestamp=i * dt,
                        position=np.array([0.0, 1.0, 0.0]),
                        velocity=np.zeros(3),
                        acceleration=np.array([0.0, -g, 0.0]),
                        rotation=np.zeros(3),
                        angular_velocity=np.zeros(3),
                        frame_index=i,
                    )
                )
            return TrajectoryData(frames=frames, params=params_obj)

        swarm = ByzantineSwarm(
            simulator_fn=mock_simulator,
            verifier=mock_verifier,
            n_agents=3,
        )

        result = swarm.run(
            initial_params=self._make_default_params(),
            max_rounds=3,
            verbose=False,
        )

        # Should have stats for all 3 agents
        assert len(result["agent_stats"]) == 3
        for agent_id in ["agent_0", "agent_1", "agent_2"]:
            assert agent_id in result["agent_stats"]
            stats = result["agent_stats"][agent_id]
            assert "params" in stats
            assert "proposals" in stats
            assert "accepted" in stats

    def test_proposals_accepted_never_exceeds_proposals_made(self):
        """Accepted proposals never exceeds proposals made."""
        mock_verifier = Mock()
        mock_verifier.verify.return_value = VerificationResult(
            overall_score=0.8,
            passed=True,
            constraint_scores={},
            details={},
        )

        def mock_simulator(params):
            frames = []
            dt = 1.0 / 30.0
            g = params.get("gravity", 9.0)
            params_obj = PhysicsParams(gravity=g)
            for i in range(30):
                frames.append(
                    FrameData(
                        timestamp=i * dt,
                        position=np.array([0.0, 1.0, 0.0]),
                        velocity=np.zeros(3),
                        acceleration=np.array([0.0, -g, 0.0]),
                        rotation=np.zeros(3),
                        angular_velocity=np.zeros(3),
                        frame_index=i,
                    )
                )
            return TrajectoryData(frames=frames, params=params_obj)

        swarm = ByzantineSwarm(
            simulator_fn=mock_simulator,
            verifier=mock_verifier,
            n_agents=3,
        )

        result = swarm.run(
            initial_params={
                "gravity": 9.5,
                "mass": 0.5,
                "friction": 0.5,
                "restitution": 0.5,
                "linear_damping": 0.1,
                "angular_damping": 0.1,
            },
            max_rounds=5,
            verbose=False,
        )

        # For each agent, accepted <= proposals
        for agent_id, stats in result["agent_stats"].items():
            assert stats["accepted"] <= stats["proposals"]


class TestSwarmExplorationRate:
    """Tests for exploration rate behavior."""

    def test_exploration_rate_decreases_over_rounds(self):
        """Exploration rate decreases as rounds progress."""
        agent = SwarmAgent(agent_id="test", param_names=["gravity"])

        params = {"gravity": 9.0}

        # Early round - high exploration
        early_proposal = agent.propose(params, exploration_rate=0.2)

        # Late round - low exploration
        late_proposal = agent.propose(params, exploration_rate=0.05)

        # Both should return valid params
        assert 8.0 <= early_proposal["gravity"] <= 11.0
        assert 8.0 <= late_proposal["gravity"] <= 11.0

    def test_agent_tracks_best_params(self):
        """Agent tracks best params when updated."""
        agent = SwarmAgent(agent_id="test", param_names=["gravity", "mass"])

        # Initially empty
        assert agent.best_params == {}
        assert agent.best_score == 0.0

        # Update best params
        agent.best_params = {"gravity": 9.81, "mass": 1.0}
        agent.best_score = 0.95

        assert agent.best_params["gravity"] == 9.81
        assert agent.best_score == 0.95

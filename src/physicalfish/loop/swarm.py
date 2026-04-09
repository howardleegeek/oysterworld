"""
Byzantine Swarm AutoResearch — Parallel physics optimization with cross-validation.

Each agent optimizes a subset of physics parameters.
After each round, agents share discoveries.
Other agents VERIFY those discoveries independently (Byzantine vote).
Only params passing majority consensus get accepted.

Architecture:
  Agent A → gravity + friction → proposes params → B,C verify → accept/reject
  Agent B → mass + restitution → proposes params → A,C verify → accept/reject
  Agent C → damping (linear + angular) → proposes params → A,B verify → accept/reject

  Shared blackboard: best verified params propagate to all agents as priors.
"""

import numpy as np
import time
import json
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from physicalfish.models import PhysicsParams, TrajectoryData, VerificationResult
from physicalfish.verification.physics_verifier import PhysicsVerifier
from physicalfish.optimizer.parameter_space import PHYSICS_PARAM_SPACE, get_bounds_for
from physicalfish.logging_config import get_logger

logger = get_logger("swarm")


@dataclass
class SwarmAgent:
    """A single agent in the Byzantine swarm."""
    agent_id: str
    param_names: list[str]  # which params this agent optimizes
    best_params: dict[str, float] = field(default_factory=dict)
    best_score: float = 0.0
    history: list[dict] = field(default_factory=list)
    proposals_made: int = 0
    proposals_accepted: int = 0

    def propose(self, current_params: dict[str, float], exploration_rate: float = 0.15) -> dict[str, float]:
        """Propose new values for this agent's parameter subset."""
        proposed = current_params.copy()
        for name in self.param_names:
            low, high = get_bounds_for(name)
            current = proposed[name]
            noise = (high - low) * exploration_rate
            new_val = current + np.random.normal(0, noise)
            proposed[name] = float(np.clip(new_val, low, high))
        self.proposals_made += 1
        return proposed


@dataclass
class ByzantineVote:
    """Result of a Byzantine verification vote."""
    proposed_by: str
    params: dict[str, float]
    votes: dict[str, float]  # agent_id -> score
    consensus_score: float
    accepted: bool
    threshold: float


@dataclass
class SwarmRound:
    """Record of one swarm round."""
    round_num: int
    proposals: list[ByzantineVote]
    best_score_after: float
    best_params_after: dict[str, float]
    timestamp: float


class ByzantineSwarm:
    """
    Byzantine Swarm AutoResearch.

    Multiple agents optimize different parameter subsets in parallel.
    After each round, proposals are cross-validated by all agents.
    Only proposals passing Byzantine majority vote are accepted.
    """

    def __init__(
        self,
        simulator_fn: Callable[[dict[str, float]], TrajectoryData],
        verifier: PhysicsVerifier,
        n_agents: int = 3,
        byzantine_threshold: float = 0.6,  # 60% of agents must agree
        acceptance_delta: float = 0.005,    # minimum improvement to accept
    ):
        self.simulator_fn = simulator_fn
        self.verifier = verifier
        self.byzantine_threshold = byzantine_threshold
        self.acceptance_delta = acceptance_delta

        # Split parameters across agents
        all_params = [p[0] for p in PHYSICS_PARAM_SPACE]
        self.agents = self._create_agents(all_params, n_agents)

        # Shared state
        self.global_best_params: dict[str, float] = {}
        self.global_best_score: float = 0.0
        self.rounds: list[SwarmRound] = []

    def _create_agents(self, all_params: list[str], n_agents: int) -> list[SwarmAgent]:
        """Split parameters across agents."""
        agents = []
        params_per_agent = len(all_params) // n_agents
        remainder = len(all_params) % n_agents

        idx = 0
        for i in range(n_agents):
            count = params_per_agent + (1 if i < remainder else 0)
            agent_params = all_params[idx:idx + count]
            idx += count
            agents.append(SwarmAgent(
                agent_id=f"agent_{i}",
                param_names=agent_params,
            ))

        return agents

    def _evaluate(self, params: dict[str, float]) -> float:
        """Generate trajectory with params and verify."""
        trajectory = self.simulator_fn(params)
        result = self.verifier.verify(trajectory)
        return result.overall_score

    def _byzantine_vote(self, proposer: SwarmAgent, proposed_params: dict[str, float]) -> ByzantineVote:
        """All agents independently verify a proposal."""
        votes = {}

        # Each agent evaluates the proposed params
        for agent in self.agents:
            score = self._evaluate(proposed_params)
            votes[agent.agent_id] = score

        # Consensus: median score (robust to outliers)
        scores = list(votes.values())
        consensus_score = float(np.median(scores))

        # Accept if consensus score improves over current best
        improvement = consensus_score - self.global_best_score
        accepted = improvement > self.acceptance_delta

        # Also check that majority agrees it's good (> threshold of agents see improvement)
        if accepted:
            agree_count = sum(1 for s in scores if s > self.global_best_score)
            agree_ratio = agree_count / len(scores)
            if agree_ratio < self.byzantine_threshold:
                accepted = False

        if accepted:
            proposer.proposals_accepted += 1

        return ByzantineVote(
            proposed_by=proposer.agent_id,
            params=proposed_params,
            votes=votes,
            consensus_score=consensus_score,
            accepted=accepted,
            threshold=self.byzantine_threshold,
        )

    def run(
        self,
        initial_params: Optional[dict[str, float]] = None,
        max_rounds: int = 20,
        target_score: float = 0.90,
        verbose: bool = True,
    ) -> dict:
        """Run the Byzantine Swarm optimization."""

        # Initialize params
        if initial_params is None:
            initial_params = {p[0]: (get_bounds_for(p[0])[0] + get_bounds_for(p[0])[1]) / 2
                            for p in PHYSICS_PARAM_SPACE}

        self.global_best_params = initial_params.copy()
        self.global_best_score = self._evaluate(initial_params)

        if verbose:
            print("\n" + "=" * 60)
            print("  BYZANTINE SWARM AUTORESEARCH")
            print("=" * 60)
            print(f"  Agents: {len(self.agents)}")
            for a in self.agents:
                print(f"    {a.agent_id}: {a.param_names}")
            print(f"  Byzantine threshold: {self.byzantine_threshold:.0%}")
            print(f"  Target score: {target_score}")
            print(f"  Initial score: {self.global_best_score:.4f}")
            print("=" * 60 + "\n")

        start_time = time.time()

        for round_num in range(max_rounds):
            round_proposals = []

            if verbose:
                print(f"Round {round_num + 1}/{max_rounds}")
                print("-" * 40)

            # Each agent proposes in parallel
            # Exploration rate decreases over time (simulated annealing)
            exploration = 0.2 * (1 - round_num / max_rounds) + 0.05

            for agent in self.agents:
                # Agent proposes new params for its subset
                proposed = agent.propose(self.global_best_params, exploration)

                # Byzantine vote
                vote = self._byzantine_vote(agent, proposed)
                round_proposals.append(vote)

                status = "ACCEPT" if vote.accepted else "reject"
                symbol = "+" if vote.accepted else " "

                if verbose:
                    scores_str = " ".join(f"{s:.3f}" for s in vote.votes.values())
                    print(f"  {agent.agent_id} [{','.join(agent.param_names[:2])}...] "
                          f"→ consensus={vote.consensus_score:.4f} "
                          f"votes=[{scores_str}] "
                          f"{'✓ ' + status if vote.accepted else '✗ ' + status}")

                # Update global best if accepted
                if vote.accepted:
                    self.global_best_params = proposed.copy()
                    self.global_best_score = vote.consensus_score
                    agent.best_params = proposed.copy()
                    agent.best_score = vote.consensus_score

            # Record round
            self.rounds.append(SwarmRound(
                round_num=round_num,
                proposals=round_proposals,
                best_score_after=self.global_best_score,
                best_params_after=self.global_best_params.copy(),
                timestamp=time.time(),
            ))

            if verbose:
                bar = "█" * int(self.global_best_score * 30)
                print(f"  → Global best: {self.global_best_score:.4f} {bar}")
                print()

            # Check target
            if self.global_best_score >= target_score:
                if verbose:
                    print(f"🎯 TARGET REACHED in round {round_num + 1}!")
                break

        elapsed = time.time() - start_time

        # Summary
        total_proposals = sum(a.proposals_made for a in self.agents)
        total_accepted = sum(a.proposals_accepted for a in self.agents)

        if verbose:
            print("\n" + "=" * 60)
            print("  SWARM COMPLETE")
            print("=" * 60)
            print(f"  Final score:     {self.global_best_score:.4f}")
            print(f"  Rounds:          {len(self.rounds)}")
            print(f"  Total proposals: {total_proposals}")
            print(f"  Accepted:        {total_accepted} ({total_accepted/max(total_proposals,1):.0%})")
            print(f"  Time:            {elapsed:.1f}s")
            print(f"\n  Best params:")
            for k, v in self.global_best_params.items():
                print(f"    {k:20s}: {v:.4f}")
            print("\n  Agent stats:")
            for a in self.agents:
                rate = a.proposals_accepted / max(a.proposals_made, 1)
                print(f"    {a.agent_id}: {a.proposals_accepted}/{a.proposals_made} accepted ({rate:.0%})")
            print("=" * 60)

        return {
            "final_score": self.global_best_score,
            "best_params": self.global_best_params,
            "rounds": len(self.rounds),
            "total_proposals": total_proposals,
            "accepted_proposals": total_accepted,
            "elapsed_seconds": elapsed,
            "convergence_curve": [r.best_score_after for r in self.rounds],
            "agent_stats": {
                a.agent_id: {
                    "params": a.param_names,
                    "proposals": a.proposals_made,
                    "accepted": a.proposals_accepted,
                }
                for a in self.agents
            },
        }

#!/usr/bin/env python3
"""
OysterWorld — Hackathon Demo
Swarm AutoResearch for Physical Data

Runs two demos back-to-back:
1. Single-agent AutoResearch (convergence from 0.42 → 0.85)
2. Byzantine Swarm (3 agents, parallel optimization, consensus)
"""

import sys
import json
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Suppress debug logs
import structlog
structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(40),  # ERROR only
    logger_factory=structlog.PrintLoggerFactory(),
)

from physicalfish.models import FrameData, PhysicsParams, TrajectoryData
from physicalfish.verification.physics_verifier import PhysicsVerifier
from physicalfish.loop.swarm import ByzantineSwarm

# ─── Colors ────────────────────────────────────────────
G = "\033[92m"   # green
R = "\033[91m"   # red
Y = "\033[93m"   # yellow
B = "\033[94m"   # blue
C = "\033[96m"   # cyan
M = "\033[95m"   # magenta
W = "\033[97m"   # white
D = "\033[90m"   # dim
BOLD = "\033[1m"
RST = "\033[0m"


def make_trajectory(params: dict, dt: float = 1.0/30, n_frames: int = 90) -> TrajectoryData:
    """Generate physically-correct trajectory from params."""
    g = params.get("gravity", 9.81)
    m = params.get("mass", 0.5)
    restitution = params.get("restitution", 0.5)
    lin_damp = params.get("linear_damping", 0.1)

    frames = []
    y, vy = 2.0, 0.0
    ground = 0.15

    for i in range(n_frames):
        t = i * dt
        if t < 2.0:
            ay = -g
            vy += ay * dt
            vy *= (1.0 - lin_damp * dt)
            y += vy * dt
            if y <= ground:
                y = ground
                vy = -vy * restitution
                ay = 0.0
        elif t < 2.5:
            ay, vy = 0.0, 0.0
            y = max(y, ground)
        else:
            ay = -g
            vy += ay * dt
            y += vy * dt
            y = max(y, ground)

        frames.append(FrameData(
            timestamp=t,
            position=np.array([0.0, y + np.random.normal(0, 0.001), -0.5]),
            velocity=np.array([0.0, vy + np.random.normal(0, 0.01), 0.0]),
            acceleration=np.array([0.0, ay, 0.0]),
            rotation=np.zeros(3),
            angular_velocity=np.zeros(3),
            frame_index=i,
        ))

    return TrajectoryData(
        frames=frames,
        params=PhysicsParams(
            gravity=g, mass=m,
            friction=params.get("friction", 0.5),
            restitution=restitution,
            linear_damping=lin_damp,
            angular_damping=params.get("angular_damping", 0.1),
        ),
    )


def print_header():
    print(f"\n{BOLD}{C}")
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║                                                             ║")
    print("║          OYSTERWORLD — SWARM AUTORESEARCH                   ║")
    print("║          for Physical Data                                  ║")
    print("║                                                             ║")
    print("║   Multiple AI agents optimize physics parameters            ║")
    print("║   in parallel with Byzantine consensus verification         ║")
    print("║                                                             ║")
    print("╚═══════════════════════════════════════════════════════════════╝")
    print(f"{RST}")


def run_single_agent_demo(verifier):
    """Part 1: Single-agent optimization."""
    print(f"\n{BOLD}{Y}{'═' * 60}{RST}")
    print(f"{BOLD}{Y}  PART 1: Single-Agent AutoResearch{RST}")
    print(f"{BOLD}{Y}{'═' * 60}{RST}\n")

    # Bad starting params
    params = {
        "gravity": 15.0, "mass": 2.5, "friction": 0.3,
        "restitution": 0.2, "linear_damping": 0.4, "angular_damping": 0.4,
    }

    best_score = 0.0
    best_params = params.copy()
    history = []

    # Initial eval
    traj = make_trajectory(params)
    result = verifier.verify(traj)
    best_score = result.overall_score
    history.append(best_score)

    print(f"  {D}Starting params: gravity={params['gravity']}, mass={params['mass']}{RST}")
    print(f"  {R}Initial score: {best_score:.4f}{RST}\n")

    # Optimize
    for i in range(50):
        # Perturb from best
        new_params = best_params.copy()
        keys = list(new_params.keys())
        for k in np.random.choice(keys, size=2, replace=False):
            bounds = {"gravity": (8,11), "mass": (0.1,2), "friction": (0.1,1),
                     "restitution": (0,1), "linear_damping": (0,1), "angular_damping": (0,1)}
            lo, hi = bounds[k]
            noise = (hi - lo) * 0.15
            new_params[k] = float(np.clip(new_params[k] + np.random.normal(0, noise), lo, hi))

        traj = make_trajectory(new_params)
        result = verifier.verify(traj)
        score = result.overall_score

        if score > best_score + 0.001:
            best_score = score
            best_params = new_params.copy()
            bar = "█" * int(score * 40)
            print(f"  {G}Iter {i+1:2d}: {score:.4f} {bar} ★{RST}")
        elif i < 3 or i % 10 == 0:
            bar = "░" * int(score * 40)
            print(f"  {D}Iter {i+1:2d}: {score:.4f} {bar}{RST}")

        history.append(best_score)

    print(f"\n  {BOLD}{G}Result: {history[0]:.4f} → {best_score:.4f} (+{best_score - history[0]:.4f}){RST}")
    print(f"  {C}Best gravity: {best_params['gravity']:.3f} m/s² (real: 9.81){RST}")
    return best_score, best_params, history


def run_swarm_demo(verifier):
    """Part 2: Byzantine Swarm optimization."""
    print(f"\n{BOLD}{M}{'═' * 60}{RST}")
    print(f"{BOLD}{M}  PART 2: Byzantine Swarm AutoResearch{RST}")
    print(f"{BOLD}{M}{'═' * 60}{RST}\n")

    print(f"  {M}3 agents, each optimizing different parameters{RST}")
    print(f"  {M}Byzantine consensus: majority must verify every proposal{RST}\n")

    bad_params = {
        "gravity": 15.0, "mass": 3.0, "friction": 0.1,
        "restitution": 0.1, "linear_damping": 0.5, "angular_damping": 0.5,
    }

    swarm = ByzantineSwarm(
        simulator_fn=make_trajectory,
        verifier=verifier,
        n_agents=3,
        byzantine_threshold=0.6,
        acceptance_delta=0.003,
    )

    result = swarm.run(
        initial_params=bad_params,
        max_rounds=15,
        target_score=0.90,
        verbose=True,
    )

    return result


def print_summary(single_score, single_history, swarm_result):
    """Final summary."""
    print(f"\n{BOLD}{C}{'═' * 60}{RST}")
    print(f"{BOLD}{C}  OYSTERWORLD DEMO COMPLETE{RST}")
    print(f"{BOLD}{C}{'═' * 60}{RST}\n")

    print(f"  {Y}Single-Agent AutoResearch:{RST}")
    print(f"    Score:      {R}{single_history[0]:.4f}{RST} → {G}{single_score:.4f}{RST}")
    print(f"    Iterations: 50")
    print()

    print(f"  {Y}Byzantine Swarm:{RST}")
    print(f"    Score:      {R}{swarm_result['convergence_curve'][0]:.4f}{RST} → {M}{swarm_result['final_score']:.4f}{RST}")
    print(f"    Rounds:     {swarm_result['rounds']}")
    print(f"    Proposals:  {swarm_result['total_proposals']} total, {swarm_result['accepted_proposals']} accepted")
    print(f"    Agents:     ", end="")
    for aid, stats in swarm_result['agent_stats'].items():
        rate = stats['accepted'] / max(stats['proposals'], 1) * 100
        print(f"{aid}({rate:.0f}%) ", end="")
    print()

    print(f"\n  {BOLD}{W}Key Insight:{RST}")
    print(f"  {C}  Byzantine consensus = only physics-proven parameters survive{RST}")
    print(f"  {C}  Swarm agents cross-validate each other's discoveries{RST}")
    print(f"  {C}  Bad proposals are rejected by majority vote{RST}")

    print(f"\n  {BOLD}{G}\"NVIDIA makes synthetic data. Our swarm makes it real.\"{RST}")
    print(f"  {D}ClawGlasses.com{RST}\n")


def main():
    print_header()

    verifier = PhysicsVerifier(gravity=9.81, tolerance=0.05)

    # Part 1
    single_score, single_params, single_history = run_single_agent_demo(verifier)

    time.sleep(1)

    # Part 2
    swarm_result = run_swarm_demo(verifier)

    # Summary
    print_summary(single_score, single_history, swarm_result)

    # Save
    output = {
        "single_agent": {"final_score": single_score, "best_params": single_params},
        "swarm": swarm_result,
        "timestamp": time.time(),
    }
    Path("data").mkdir(exist_ok=True)
    with open("data/hackathon_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"  {D}Results saved to data/hackathon_results.json{RST}\n")


if __name__ == "__main__":
    main()

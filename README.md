# OysterWorld

### Decentralized Physical AI AutoResearch Swarm Infra

Multiple AI agents optimize physics simulation parameters in parallel. Byzantine consensus ensures only physically proven configurations survive.

---

**Real data costs too much. Synthetic data is too fake. We bridge the gap.**

```
10 hrs real data → Swarm AutoResearch → 10,000 hrs verified synthetic data
                   3 agents in parallel
                   Byzantine consensus
                   6 physics constraints
```

## The Problem

Training physical AI models costs $100M+. The industry turns to synthetic data, but simulation friction is 0.5 while reality is 0.3. Nobody checks if synthetic data obeys real-world physics before training.

## How It Works

```
┌─────────────────────────────────────────────────────────┐
│              SWARM AUTORESEARCH                          │
│                                                          │
│   Agent A (gravity, mass)  ──┐                           │
│   Agent B (friction, restitution) ──→ Byzantine Vote     │
│   Agent C (damping) ────────┘     → Accept / Reject      │
│                                                          │
│   6 Physics Constraints:                                 │
│   Newton's laws · Energy · Momentum                      │
│   Angular momentum · Collision · Kinematics              │
│                                                          │
│   Only physics-proven parameters survive.                │
└─────────────────────────────────────────────────────────┘
```

Each agent optimizes a different subset of physics parameters. Every proposal is cross-verified by all agents against 6 physical constraints. Majority vote decides acceptance. Bad physics gets rejected.

## Live Results

| Metric | Value |
|--------|-------|
| Start score (wrong physics) | 0.37 |
| Single agent (50 iterations) | 0.67 |
| **Swarm verified** | **0.72** |
| Gravity found | 9.76 m/s² (real: 9.81) |
| Total time | 30 seconds |

## Quick Start

```bash
git clone https://github.com/howardleegeek/oysterworld.git
cd oysterworld

python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run tests (52 passing)
pytest

# Run hackathon demo
python3 scripts/hackathon_demo.py

# Run swarm only
python3 scripts/run_swarm_demo.py --rounds 15 --agents 3
```

## Architecture

```
src/physicalfish/
├── models.py                 # FrameData, PhysicsParams, TrajectoryData
├── verification/
│   ├── physics_verifier.py   # 6-constraint physics verification
│   └── distribution_compare.py
├── optimizer/
│   ├── bayesian.py           # Bayesian optimization (GP + EI)
│   ├── adaptive.py           # Two-phase: global → local search
│   └── parameter_space.py    # Centralized parameter bounds
├── simulator/
│   ├── pybullet_sim.py       # PyBullet headless physics
│   └── scenarios.py          # grasp_ball, throw_catch, roll_incline
├── loop/
│   ├── inner_loop.py         # Single-task optimization cycle
│   ├── outer_loop.py         # Cross-scenario strategy transfer
│   ├── bootstrap.py          # Few-shot → many-shot augmentation
│   └── swarm.py              # Byzantine Swarm AutoResearch
├── capture/
│   └── real_data.py          # Real robot data loader
└── api/
    ├── app.py                # FastAPI service
    └── routes.py             # POST /verify, /optimize, /bootstrap
```

## 6 Physics Constraints

| # | Constraint | Checks | Theoretical Basis |
|---|-----------|--------|-------------------|
| 1 | **Kinematic** | v = dx/dt, a = dv/dt | Central differencing numerical validation |
| 2 | **Dynamic** | F = ma, free-fall = gravity | Newton's Second Law verification |
| 3 | **Energy** | KE + PE conserved | Hamiltonian mechanics, allows dissipation |
| 4 | **Momentum** | p = mv, smooth changes | Conservation laws, impulse detection |
| 5 | **Angular** | L = r × p consistent | Rotational dynamics, torque consistency |
| 6 | **Collision** | Restitution coefficient correct | Contact mechanics, Hertzian model |

## Research Foundation

Our verification approach builds on established work across physics-informed ML, sim-to-real transfer, and automated research:

### Core: Physics-Informed Verification
- **Raissi et al. 2019** — Physics-Informed Neural Networks (PINNs) for solving and constraining PDEs. Our 6-constraint engine extends this to data validation rather than equation solving.
- **Evo-PINN (2025)** — Evolutionary optimization + PINNs for automated architecture search. Parallels our AutoResearch loop structure.
- **Hamiltonian Neural Networks** (Greydanus et al., NeurIPS 2019) — Learning energy-conserving dynamics. Basis for our energy constraint.
- **Lagrangian Neural Networks** (Cranmer et al., 2020) — Physics-preserving learned simulators. Informs our momentum/angular momentum checks.
- **Dynami-CAL GraphNet** (Nature Communications, 2025) — GNN enforcing conservation of linear and angular momentum. Validates our approach to momentum verification.

### Core: AutoResearch Pattern
- **Karpathy AutoResearch (2026)** — 630-line autonomous experiment loop. 700 experiments in 2 days, 11% improvement on GPT-2. Direct template for our optimization loop.
- **DrEureka (RSS 2024)** — LLM-automated reward function + domain randomization design. Zero human physics tuning. Our target for next-gen parameter optimization.
- **Eureka (ICLR 2024, NVIDIA)** — GPT-4 evolutionary optimization over reward code. Outperforms human experts on 83% of RL environments.

### Core: Sim-to-Real & Data Flywheel
- **DexFlyWheel (NeurIPS 2025 Spotlight)** — 1 demo → 500x trajectories via IL + Residual RL + augmentation cycle. Closest to our flywheel but **lacks physics verification step**.
- **MimicGen (CoRL 2023, NVIDIA)** — 200 demos → 50K generated demos via SE(3) subtask adaptation.
- **SoftMimicGen (2025, NVIDIA)** — Single demo → 1,000 demos for deformable objects.
- **DORAEMON (ICLR 2024)** — Constrained optimization for domain randomization, outperforms OpenAI AutoDR.
- **SplatSim (CMU, 2024)** — Gaussian Splatting for 86.25% zero-shot sim-to-real. Solves visual gap; we solve physics gap.

### Core: Verifiable AI
- **EZKL** — ZK proofs for neural network inference verification. Production-grade for medium models.
- **Modulus/Remainder** — Hundreds of thousands of verified AI results on Ethereum. Most battle-tested zkML.
- **Lagrange DeepProve-1** — First system to prove full GPT-2 inference with ZK.
- **TEE Attestation (Intel TDX)** — 3-8% overhead for hardware-attested simulation. Deployable today on GCP.
- **Phala Network** — Production TEE-to-blockchain attestation pipeline. 1,000+ deployments.

### Novel Contribution (No Prior Art)

**Using physics constraints to automatically score and filter synthetic robot manipulation data before training — this specific combination has no published prior art.** We confirmed this through exhaustive search across arXiv, CoRL, ICRA, RSS, NeurIPS, and ICML proceedings (2020-2026).

Closest related work:
- Contact-Based Dataset Curation (Oct 2025) — uses Fisher information, not physics constraints
- CUPID (June 2025) — uses influence functions, not physics laws
- Consistency Matters (Dec 2024) — measures demonstration quality, not physics plausibility

**Our gap: physics constraint satisfaction as a quality signal for synthetic data. This is publishable.**

Target venue: **ICRA 2027** (~Sep 2026 deadline)

## Hardware: ClawGlasses

**$99 AI glasses that turn every wearer into a physical AI data node.**

| Spec | Value |
|------|-------|
| Price | $99 |
| Nodes deployed | 25,000 |
| FOV | 165° wide-angle |
| Battery | 1hr continuous (charge while recording) |
| Storage | 8hrs capacity |
| Sync | Multi-camera hardware sync + IMU |
| Transfer | WiFi → phone → cloud pipeline |

**Why glasses?**
- First-person viewpoint matches robot head-cam perspective — minimal domain gap
- Passive, all-day wearable — data collection without changing behavior
- Customers buy for themselves — every unit sold is a data node that pays for itself
- 165° FOV covers hands, body, and workspace in a single frame

**On-device pipeline:**
```
Capture → Local compression → Phone (filter no-hand frames) → Cloud processing
```

**Cloud pipeline:**
```
VIO → Hand detection → 2D keypoints → 3D pose estimation → Quality filtering → Robot-ready data
```

Every ClawGlasses unit is an always-on data collection node. No teleoperation rigs. No motion capture studios. Just people living their lives.

## Stats

- 3,400 lines production code
- 52 tests passing
- 22 modules
- FastAPI + Docker
- Open X-Embodiment compatible

---

**Synthetic data is everywhere. We make it physically real.**

[www.ClawGlasses.com](https://www.clawglasses.com)

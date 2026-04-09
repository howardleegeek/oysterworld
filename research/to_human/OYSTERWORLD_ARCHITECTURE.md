# OysterWorld — Technical Architecture & Strategy

## 一句话

> OysterWorld 是 Physical AI 的数据质量基础设施层：用物理约束自动验证、优化、扩增合成训练数据，填补 NVIDIA（生成）和 Google（采集）之间的空白。

---

## 市场定位

```
Layer 3: 数据生成 → NVIDIA Cosmos/Isaac Lab (拥挤)
Layer 2: 数据采集 → Google AutoRT/pi遥操作 (昂贵)
Layer 1: 数据质量验证 → OysterWorld (空白) ← 我们在这里
```

- NVIDIA Isaac ADR 随机化物理参数但**不验证物理一致性**
- Google RT-X 有 1M+ 轨迹但**无自动质量门**
- Physical Intelligence π0 用 10k+ 小时遥操作但**无合成数据管道**
- TRI 仿真数据只占训练集 3% — **他们不信任仿真数据**
- DexFlyWheel 做数据飞轮但**无物理验证步骤**

**没有人做自动化物理约束验证。**

---

## 核心架构

```
┌─────────────────────────────────────────────────────────┐
│                    OYSTERWORLD                           │
│                                                          │
│  ┌──────────┐    ┌───────────┐    ┌──────────────────┐  │
│  │ 真实数据  │    │  仿真引擎  │    │  物理验证器       │  │
│  │ (RLDS)   │───→│ (PyBullet) │───→│ (6-PINNs约束)    │  │
│  │ 10 条    │    │ 参数空间   │    │ kinematic/dynamic │  │
│  └──────────┘    └─────┬─────┘    │ energy/momentum   │  │
│       ↑                │          │ angular/collision  │  │
│       │                │          └────────┬───────────┘  │
│       │                │                   │              │
│       │         ┌──────▼──────┐    ┌──────▼───────┐     │
│       │         │ DrEureka    │    │ 分布比较器    │     │
│       │         │ LLM自动调参  │    │ syn vs real  │     │
│       │         │ (GPT-4/Claude)│   └──────┬───────┘     │
│       │         └──────┬──────┘           │              │
│       │                │          combined_score          │
│       │                ▼                  │              │
│       │         ┌─────────────────────────▼─────┐       │
│       │         │     AUTORESEARCH LOOP          │       │
│       │         │ Karpathy pattern: 固定预算实验  │       │
│       │         │ PROPOSE → SIMULATE → VERIFY    │       │
│       │         │         → DECIDE (commit/revert)│      │
│       └─────────│ 收敛后输出最优参数             │       │
│                 └───────────────┬────────────────┘       │
│                                │                         │
│                    ┌───────────▼──────────┐              │
│                    │  TEE 可信执行环境     │              │
│                    │  Intel TDX / AMD SEV  │              │
│                    │  3-8% 性能开销       │              │
│                    │  硬件级证明          │              │
│                    └───────────┬──────────┘              │
│                                │                         │
│                    ┌───────────▼──────────┐              │
│                    │  链上证明            │              │
│                    │  TEE attestation     │              │
│                    │  → ZK proof (RISC Zero)│            │
│                    │  → zkVerify / Base L2 │             │
│                    └─────────────────────┘              │
└─────────────────────────────────────────────────────────┘

输入: 10 条真实轨迹
输出: 10,000 条物理验证过的合成轨迹 + 链上质量证明
```

---

## 技术栈选型

| 组件 | 选择 | 理由 |
|------|------|------|
| 物理仿真 | PyBullet (CPU) / Isaac Gym (GPU) | 免费、headless、Python API |
| 参数优化 | DrEureka (LLM) + Bayesian (fallback) | DrEureka 零人类调参，Bayesian 做 fallback |
| 物理验证 | 自研 6-PINNs 约束 | 空白领域，我们的创新点 |
| 数据格式 | RLDS (TensorFlow) | Open X-Embodiment 标准格式 |
| 可信计算 | Intel TDX on GCP C3 | 3-8% 开销，GA，零代码修改 |
| 链上证明 | Phala dcap-qvl → zkVerify | 最成熟管道，90% gas 节省 |
| API | FastAPI | 已实现，52 测试通过 |

---

## 关键数字

| 指标 | 值 | 来源 |
|------|-----|------|
| TEE 性能开销 | 3-8% (TDX), 2-5% (SEV) | Intel/AMD benchmarks |
| GCP TDX VM 成本 | ~$220-285/月 (c3-standard-8) | GCP pricing |
| DrEureka LLM 调用 | 80次/优化周期 (16候选×5迭代) | DrEureka paper |
| DexFlyWheel 扩增 | 500x 轨迹, 214x 场景 | NeurIPS 2025 |
| SoftMimicGen | 1条→1000条 | NVIDIA 2025 |
| 我们当前代码 | 2,902行, 52测试, 60%覆盖 | pytest |

---

## 论文机会

### "Physics-Constrained Quality Verification for Synthetic Robot Manipulation Data"

| 项 | 内容 |
|---|------|
| 核心贡献 | 首个用物理约束（Newton定律、能量守恒、动量守恒、碰撞恢复）自动评分合成机器人数据质量的系统 |
| Framing | 轻量预过滤器，在昂贵的 influence function 之前跑 |
| 最强对比 | Contact-Based Curation (Oct 2025) — 信息论 vs 我们的物理约束 |
| 数据集 | Open X-Embodiment + 自生成带噪声合成数据 |
| 验收指标 | 过滤后数据训练的策略成功率 > 未过滤 |
| Gap 确认 | ✅ 穷搜后无同类论文 |

### Venue Deadlines

| 会议 | Deadline | 状态 |
|------|----------|------|
| **CoRL 2026** | May 28, 2026 | ⚠️ ~7周 |
| **NeurIPS 2026** | May 6, 2026 | ❌ 太紧 |
| **ICRA 2027** | ~Sep 2026 | ✅ 5个月 |

---

## WOW Demo 设计（全栈打通）

### Layer 1: 摄像头/数据采集
- 从 Open X-Embodiment 下载 10 条真实抓取轨迹
- 或用 MacBook 摄像头 + MediaPipe 采集

### Layer 2: AutoResearch 优化 + 3D 可视化
- PyBullet 仿真 + DrEureka LLM 调参
- 6-PINNs 物理验证评分
- Three.js 实时渲染优化过程
- 收敛曲线 + 参数空间热力图

### Layer 3: TEE 证明 + 链上
- GCP C3 TDX 跑仿真
- Phala dcap-qvl 生成 attestation
- RISC Zero 包 ZK proof
- 发到 zkVerify / Base L2
- Demo: "这个数据集有链上物理质量证明"

### 故事线
> "我从公开数据集取了 10 条真实抓取轨迹。
> OysterWorld 在可信执行环境中自动优化仿真参数、
> 扩增到 10,000 条，每条通过 6 项物理定律验证。
> 验证结果上链，任何人可查。
> 用 NVIDIA 生成，用 OysterWorld 验证。"

---

## Paradigm 对齐

Paradigm 正在募集 $1.5B，三大论题：
1. ✅ 去中心化 AI 基础设施 → OysterWorld 开源
2. ✅ 自主经济体/机器人 → Physical AI 训练数据
3. ✅ **可验证 ML** → TEE + ZK 物理验证

投了 Nous Research $50M (区块链+AI)。
OysterWorld = **可验证的 Physical AI 数据质量基础设施** — 完美对齐。

---

## 下一步行动

| 优先级 | 行动 | 时间 |
|--------|------|------|
| P0 | 下载 Open X-Embodiment 数据 + 格式转换 | 1天 |
| P0 | 接 PyBullet 真实仿真（修编译或用 MuJoCo） | 2天 |
| P1 | DrEureka 集成（LLM 自动调参替代手动 Bayesian） | 3天 |
| P1 | 跑通 10条→10000条 端到端 | 3天 |
| P2 | GCP TDX VM 部署 + Phala attestation | 3天 |
| P2 | Three.js 3D 可视化 dashboard | 5天 |
| P3 | ZK proof + 链上 | 5天 |
| P3 | 论文写作 (targeting ICRA 2027) | 2周 |

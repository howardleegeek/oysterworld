# OysterWorld — Research Findings (FINAL)

## 一句话结论

> **Physical AI 数据领域有三层：生成层（NVIDIA）、采集层（Google）、验证层（空白）。OysterWorld 做验证层。**

## 三层市场格局

| 层 | 玩家 | 状态 |
|---|------|------|
| 数据生成（仿真渲染） | NVIDIA Cosmos, Isaac Lab, SplatSim | 拥挤，资金充足 |
| 数据采集（真实世界） | Google AutoRT, Scanford, pi 遥操作 | 昂贵，在扩张 |
| **数据质量验证（物理）** | **没有人** | **空白** |

## 竞品深度分析

### NVIDIA Isaac Lab ADR
- 自动域随机化：GPU 并行跑数千环境，自动扩缩参数范围
- **缺陷**：不验证物理一致性 — 可以把摩擦力随机到荒谬值（如无摩擦玻璃被抓取）
- **OysterWorld 机会**：做 ADR 的"物理安全检查" — 确保随机化参数仍然物理可信

### Google DeepMind RT-X / AutoRT
- 1M+ 真实轨迹，22 种机器人，34 个实验室
- AutoRT: 20 个机器人同时自主采集 77,000 次试验
- **缺陷**：开环管道 (collect → aggregate → train → evaluate)，无数据质量门
- **OysterWorld 机会**：成为 collect 和 train 之间的质量验证层

### Physical Intelligence π0
- 最好的模型架构 (3B VLM + flow matching)，10k+ 小时遥操作数据
- **缺陷**：全部真实数据，无合成管道，数据采集极其昂贵
- **OysterWorld 机会**：用物理验证过的合成数据减少 pi 的数据采集成本 10-100x

### Toyota Research TRI
- Diffusion Policy 统治行为克隆领域
- **关键数字**：仿真数据只占训练数据的 3%（45h/1695h）— 他们不信任仿真数据
- **OysterWorld 机会**：证明物理验证过的仿真数据可以安全扩大比例到 30-50%

### DexFlyWheel (NeurIPS 2025 Spotlight)
- 最接近 OysterWorld 的概念：单个人类演示 → 500x 轨迹增加
- **缺陷**：增强场景但不验证物理一致性
- **OysterWorld 机会**：在飞轮循环中加入物理验证步骤

### NVIDIA Cosmos
- 800磅大猩猩：Cosmos Transfer 2.5 做光真实渲染
- 但：只做视觉保真度，不做物理参数优化或验证
- **OysterWorld 定位**：与 Cosmos 互补而非竞争 — "用 NVIDIA 生成，用 OysterWorld 验证"

## 技术前沿

### Karpathy AutoResearch (2026-03)
- 630 行 Python，21K GitHub stars
- 126 个实验过夜，"Time to GPT-2" 降低 11%
- Shopify 已在用：内部模型提升 19%
- **OysterWorld 的模板**：把 BPB 替换为物理一致性分数

### PINNs 验证合成数据 = 开放研究空白
- **没有一篇论文**做"用 PINNs 验证合成机器人数据是否物理可信"
- Evo-PINN (2025): 进化优化 + PINNs，最接近的工作
- **这是 OysterWorld 的学术创新点** — 可以发论文

### zkML 现实评估
| 技术 | 状态 | 对 OysterWorld |
|------|------|---------------|
| EZKL | 中等 NN 可用，物理仿真不行 | 未来用于 NN 组件 |
| Modulus/Remainder | 生产级，数十万结果上链 | 最成熟参考 |
| DeepProve-1 | 首次证明 GPT-2 推理 | 技术领先但新 |
| 物理仿真 ZK 验证 | **2028+** | 用 TEE 替代 |
| TEE 硬件证明 | **今天可用** | MVP 路径 |

### 数据市场无人验证质量
- Ocean Protocol: 访问权限 ≠ 质量验证
- Vana: 数据主权 ≠ 质量验证
- Sahara AI: NFT 化 ≠ 质量验证

## Paradigm 对齐分析
- 正在募集 **$1.5B** 专门投 AI + 机器人 + 前沿技术
- 投了 Nous Research $50M (区块链+AI 协调)
- **三大论题完美对齐**：
  1. 去中心化 AI 基础设施 ✓ (OysterWorld 可开源)
  2. 自主经济体/机器人 ✓ (Physical AI 数据)
  3. **可验证 ML** ✓ (物理验证 + TEE + 未来 ZK)

## WOW Factor = 用真实数据证明

### 可用的公开数据集
| 数据集 | 规模 | 用途 |
|--------|------|------|
| Open X-Embodiment | 1M+ 轨迹, 22 种机器人 | Baseline，行业标准 |
| DROID | 76K 轨迹, 多场景 | 抓取场景 |
| RoboSet | 30K 轨迹, 厨房场景 | 日常任务 |
| NVIDIA PhysicalAI | HuggingFace | 直接下载 |

### Demo 剧本
1. 从 Open X-Embodiment 取 10 条真实抓取轨迹
2. OysterWorld AutoResearch 循环：物理验证 + 参数优化
3. 扩增到 10,000 条，每条通过 6 约束检验
4. 展示收敛曲线 + 参数空间热力图
5. （进阶）用扩增数据训练策略，对比原始数据训练的成功率

## 开放问题
1. Open X-Embodiment 的 RLDS 格式如何转换为 OysterWorld 的 TrajectoryData？
2. TEE 包装仿真的工程成本？Intel TDX vs ARM CCA？
3. 论文投哪个 venue？(ICRA? CoRL? RSS?)

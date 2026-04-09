# OysterWorld — Key Papers Reference

## 必读 (直接影响架构)

### 1. DrEureka (RSS 2024) ⭐⭐⭐
- LLM 自动设计 reward + domain randomization distributions
- 无需人类物理调参
- **OysterWorld 应该集成此方法替代手动 Bayesian 优化**
- https://arxiv.org/abs/2406.01967

### 2. Karpathy AutoResearch (2026-03) ⭐⭐⭐
- 630行Python, 700实验/2天, 11%提升
- 核心约束: 固定计算预算使实验可比
- **OysterWorld 的直接模板**
- https://github.com/karpathy/autoresearch

### 3. SoftMimicGen (2025/2026, NVIDIA) ⭐⭐⭐
- 单个人类演示 → 1000 条合成演示
- 扩展到柔性物体操作
- **少样本扩增的 SOTA**
- https://arxiv.org/abs/2603.25725

### 4. DexFlyWheel (NeurIPS 2025 Spotlight) ⭐⭐
- 单个演示 → 500x 轨迹, 214x 场景
- IL → Residual RL → Rollout → Augment 循环
- 真实世界 78.3% 成功率
- **最接近 OysterWorld 的飞轮概念，但无物理验证**
- https://arxiv.org/abs/2509.23829

### 5. SPIN (ICML 2024) ⭐⭐
- 自对弈微调，无需额外人类数据
- 自纠正属性：初始噪声数据可容忍
- **证明自改进循环有效**
- https://arxiv.org/abs/2401.01335

## 竞品/参考

### 6. Eureka (ICLR 2024, NVIDIA)
- GPT-4 进化优化 reward code
- 83% 环境超越人类专家
- DrEureka 的前身

### 7. DORAEMON (ICLR 2024)
- 约束优化直接最大化训练分布熵
- 优于 OpenAI AutoDR

### 8. One-Shot Real-to-Sim (2024)
- 单个动作序列校准仿真参数
- 端到端可微仿真+渲染
- https://arxiv.org/abs/2412.00259

### 9. SplatSim (2024, CMU)
- Gaussian Splats 替代 mesh 渲染
- 86.25% 零样本 sim-to-real
- 解决视觉域差距，但不解决物理域差距
- https://arxiv.org/abs/2409.10161

### 10. RoVi-Aug (CoRL 2024)
- 扩散模型增强跨机器人/视角数据
- 30% 成功率提升

## zkML/验证

### 11. EZKL v1.0 (2025)
- ONNX模型 ZK 证明, ≤50M 参数
- 30秒-4分钟证明时间
- 物理仿真验证: 不可能 (2028+)

### 12. Modulus/Remainder (生产级)
- 数十万 AI 结果上链以太坊
- 180x 开销 vs 原始推理

### 13. DeepProve-1 (Lagrange Labs)
- 首次证明完整 GPT-2 推理
- 54-158x 快于 EZKL

### 14. Framework for E2E Verifiable AI (2025)
- 首个 ZK 训练验证系统 (非仅推理)
- https://arxiv.org/abs/2503.22573

## 空白领域 (OysterWorld 的创新点)

### 15. PINNs 验证合成数据 = 无人做过
- Evo-PINN (2025) 最接近但方向不同
- **用 PINNs 验证合成机器人数据的物理可信性 = 可发论文**

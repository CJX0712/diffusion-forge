# ✦ Diffusion Forge · 扩散模型实验室

<p align="center">
  <a href="https://github.com/CJX0712/diffusion-forge/actions/workflows/ci.yml"><img src="https://github.com/CJX0712/diffusion-forge/actions/workflows/ci.yml/badge.svg" alt="ci"></a>
  <a href="https://github.com/CJX0712/diffusion-forge/releases"><img src="https://img.shields.io/github/v/release/CJX0712/diffusion-forge?sort=semver" alt="release"></a>
  <a href="https://github.com/CJX0712/diffusion-forge/blob/main/LICENSE"><img src="https://img.shields.io/github/license/CJX0712/diffusion-forge" alt="license"></a>
  <img src="https://img.shields.io/badge/author-%E6%99%A8%E6%98%9F-1f6feb" alt="author">
  <img src="https://img.shields.io/badge/invariants-36%2F36-brightgreen" alt="invariants">
</p>

单文件、零依赖、离线可跑的**扩散模型全范式实验室**。浏览器直接打开 `index.html` 即可。
从零手写全部组件：闭式前向加噪、后验公式、MLP 噪声/速度预测网络、AdamW、DDPM / DDIM / Flow Matching 三种采样器，约 900 行纯 JS。

**v2 升级（2026-10）**：在保留 v1 全部引擎 API 与 18 项不变量的前提下，新增 Flow Matching 范式、cosine 噪声调度、SiLU/He 初始化、AdamW、W1 生成质量度量、DDIM 跳步加速、x̂₀ 裁剪稳定化、去噪轨迹可视化与浏览器内不变量面板——引擎不变量扩至 **36 项，Node 无头全绿**。

## 它做什么

| 范式 | 训练目标 | 采样 | 论文 |
|---|---|---|---|
| **DDPM** | `ε_θ(x_t,t) ≈ ε` | 祖先采样（全链随机） | Ho et al. 2020 |
| **DDIM** | 同上（共享网络） | η 可控 + stride 跳步（10 步可采样） | Song et al. 2021 |
| **Flow Matching** | `v_θ(x_t,t) ≈ ε − x₀` | ODE 积分（Euler / midpoint） | Lipman et al. 2023 |

- **前向扩散**：`x_t = √ᾱ_t·x₀ + √(1−ᾱ_t)·ε`，支持 linear β∈[0.002,0.10] 与 cosine（Nichol & Dhariwal 2021）两种调度
- **网络**：输入 `[x_t, t 的正弦特征]`（7 维）→ MLP（ReLU 或 SiLU，He 初始化）→ 2 维输出
- **优化**：AdamW（解耦权重衰减，wd=0 时退化为 Adam，bit 级兼容 v1）
- **度量**：W1 距离（1D 排序闭式解，对 x/y 取平均）量化"生成分布 vs 数据分布"
- **三种数据集**：四簇高斯 / 瑞士卷 / 双月牙；散点可视化 + 前向加噪演示 + 去噪轨迹 + loss/调度曲线

## 可验证不变量（Node 无头全绿）

### v1 引擎不变量（`_smoke.js`，18 项）

1. **前向闭式解 vs 逐步迭代**：同一过程两条独立算法路径，6000 样本矩匹配（3σ 内）
2. **后验代数恒等式**：`x_t = √ᾱ_t·x₀` 代入 `q(x_{t-1}|x_t,x_0)` 均值公式，所有 t 精确还原 `√ᾱ_{t-1}·x₀`（< 1e-12）
3. **梯度检验**：MLP 反传解析梯度 vs 中心差分，相对误差 < 1e-5
4. **训练确定性**：同种子两次训练，参数 bit 级相等
5. **生成质量**：DDPM/DDIM 采样 800 点全覆盖 4 个高斯簇（每簇 12–45%），样本散布 ≈ 数据散布
6. **DDIM 确定性**：η=0 同种子 bit 级相同；η=1 恢复随机
7. 边界：T=1 调度、batch=1 训练、小模型采样、矩有限性

### v2 引擎不变量（`_smoke2.js`，18 项，与浏览器面板同一套 `DIFF.invariants`）

| # | 不变量 | 类型 | 实测 |
|---|---|---|---|
| 1 | cosine 调度合法性（单调递减/端点/β 域） | 闭式 | ᾱ₁=0.99825 ᾱ₅₀=9.71e-7 |
| 2 | q_sample 一/二阶矩 vs 闭式 | 蒙特卡洛 6σ | 均值 0.41247 vs 0.40865 |
| 3 | 后验：DDPM 闭式 vs **贝叶斯法则独立推导** | 代数 | 最大差 6.77e-15 |
| 4 | ε / x₀ / score 三种参数化精确互换 | 代数 | 7.22e-16 |
| 5 | SiLU 导数恒等式 | 数值微分 | 2.50e-9 |
| 6 | SiLU MLP 梯度检验（中心差分） | 解析 vs 数值 | 3.91e-9 |
| 7 | DDPM(linear) 训练收敛 | 训练 | 0.8808 → 0.6006 |
| 8 | DDPM(linear) 生成质量 | W1 | 0.3580 < ½·0.5357 |
| 9 | DDIM(stride=5, η=0) 确定性（含换 z 序列） | 精确 | 差 = 0 |
| 10 | DDIM 10 步少步采样质量 | W1 | 0.3376 < ½·0.5357 |
| 11 | DDPM(cosine) 训练收敛 | 训练 | 0.8613 → 0.5407 |
| 12 | DDPM(cosine) 生成质量 | W1 | 0.3891 < ½·0.5357 |
| 13 | Flow Matching 训练收敛 | 训练 | 3.7256 → 2.6006 |
| 14 | Flow Matching 生成质量 | W1 | 0.2490 < ½·0.5357 |
| 15 | AdamW：零梯度精确衰减 + 单步更新界 | 代数 | 偏差 0.00e+0；maxΔp=1.000e-2 ≤ lr |
| 16 | Flow 常速度场 Euler 精确可积 | 精确 | 7.99e-15 |
| 17 | Flow 线性场 Euler 全局误差一阶 | 数值阶数 | err 比值 1.982 |
| 18 | 训练确定性 + 参数量闭式公式 | 精确 | bit 差 = 0；434 = Σ(in·out+out) |

### UI 接线检查（`_uicheck.js`）

最小 DOM stub 下执行页面 UI 脚本，点遍全部控件（训练/重置/换数据集/Flow 训练/三种范式采样/不变量面板），断言 5 块画布均绘制、状态栏/W1 统计/自检结果均填充。

## 运行测试

```bash
node _smoke.js    # v1 引擎不变量（18 项）
node _smoke2.js   # v2 引擎不变量（18 项，Flow/cosine/W1/SiLU/AdamW/Euler）
node _uicheck.js  # DOM stub 下点遍全部 UI 控件
node _probe.js    # 矩对比 + ASCII 密度图（生成 _probe.txt）
```

CI（`.github/workflows/ci.yml`）在每次 push / PR 上跑前三个，任一失败即红。

## 原理一图流

```
x₀ --(加噪 β₁…β_T)--> x_T ~ N(0,I)
      训练：网络 ε_θ(x_t, t) ≈ ε        （DDPM/DDIM）
      训练：网络 v_θ(x_t, t) ≈ ε − x₀   （Flow Matching）
采样：x_T → ε̂ → 逐步去噪（后验均值 + 方差）→ x₀
DDIM：把随机方差换成 η 参数，stride 跳步，η=0 时整条链变成确定性 ODE 轨迹
Flow：x_T → 解 ODE dx/dt = v_θ（Euler/midpoint）→ x₀
```

## 设计取舍

| 决策 | 理由 |
|---|---|
| Float64Array 全程 | 梯度检验要求 ~1e-9 精度，Float32 地板误差 6e-8 会污染结论 |
| v1 API bit 级兼容 | `wd=0` 退化为原 Adam、`act` 默认 relu、`stride` 默认 1——老测试原样通过 |
| 后验均值用系数式 `c0·x̂₀+ct·x_t` | 与 `(x_t−β/√(1−ᾱ)·ε̂)/√α` 代数等价，且裁剪 x̂₀ 时语义正确 |
| x̂₀ 裁剪到 [-4,4]（可选） | cosine 调度在高噪声端 ᾱ≈1e-6，未收敛模型的 x̂₀ 被 1/√ᾱ 放大约 1000 倍，裁剪是标准稳定化手段 |
| W1 而非 FID | 2D 上 W1 有排序闭式解，无需采样近似，不变量才立得住 |
| 不变量套件内置引擎（`DIFF.invariants`） | 浏览器面板与 Node `_smoke2.js` 跑同一套判定逻辑，杜绝两套实现漂移 |

## 路线图

- [ ] 16×16 像素精灵扩散（复用 conv-forge 的 im2col+GEMM 卷积内核）
- [ ] 条件扩散（classifier-free guidance）
- [ ] 与参照实现（PyTorch）交叉验证数值一致性
- [ ] 更快 ODE 求解器（RK4/Heun）与少步蒸馏

## 站在肩膀上（零重写 · 集大成）

Ho et al. 2020 *DDPM* · Song et al. 2021 *DDIM* · Song et al. 2021 *Score-Based SDE* ·
Nichol & Dhariwal 2021 *Improved DDPM（cosine 调度）* · Lipman et al. 2023 *Flow Matching* ·
Liu et al. 2022 *Rectified Flow* · Kingma & Ba 2015 *Adam* · Loshchilov & Hutter 2019 *AdamW*

## License

MIT © 晨星

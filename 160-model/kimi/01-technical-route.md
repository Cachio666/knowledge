---
title: Kimi 的技术路线：从 MuonClip 到 KDA + AttnRes
slug: kimi-technical-route
domain: 160-model
topic: kimi
level: 2
difficulty: advanced
prerequisites:
  - 80-model-architectures/attention-variants/02-mla-low-rank
  - 80-model-architectures/moe/03-balance-and-granularity
related:
  - 160-model/kimi/02-k3-architecture
  - 160-model/deepseek/01-technical-route
  - 160-model/glm/01-glm-technical-route
code:
  - code/01_param_budget.py
status: stable
updated: 2026-10-13
---

# Kimi 的技术路线：从 MuonClip 到 KDA + AttnRes

> 面对同一个问题——长上下文的 KV 太贵——三家给出了三种答案。**DeepSeek 选择「压缩那份 $O(L)$ 的缓存」**（CSA2 + FP4，890 B/token）；**GLM 选择「只在缓存里挑几个位置看」**（DSA + IndexShare）；**Kimi 选择「把缓存整个换掉」**——用固定大小的递推状态承载历史（**KDA**）。K3 的 93 层里 69 层是 KDA、只有 24 层是全局 MLA，于是 1M 上下文的全局 KV 从「93 层全 MLA 的 112 GB」降到 **29 GB**。

## 0. 背景与动机

先看 K3 与 K2 的官方对照（【C】K3 报告 Table 1，逐格）：

| | Kimi K2 | **Kimi K3** | 变化 |
|---|---|---|---|
| 层数 | 61 | **93** | +52% |
| 总参 / 激活 | 1.04T / 32.6B | **2.78T / 104.2B** | +167% / +220% |
| Hidden | 7 168 | 7 168 | 不变 |
| Routed experts（每 token 激活） | 384（8） | **896（16）** | +133% / +100% |
| 稀疏度（$E/k$） | **48** | **56** | 更稀 |
| Latent MoE 维 | — | **3 584（0.5×）** | 新增 |
| 注意力 | MLA | **Hybrid KDA–MLA**（69 KDA + 24 MLA） | 换机制 |
| 激活函数 | SwiGLU | **SiTU-GLU** | 换 |
| 训练上下文 | 128K | **1M** | 8× |
| ViT | — | 401M / 27 层 | 新增 |

三个数字连起来读，就是这条线的全部张力：

1. **层数涨 52%、激活涨 220%**——规模还在推；
2. **hidden 维一点没变**（7 168）——K2 → K3 是纯粹的「更深、更稀」，不是「更胖」；
3. **注意力机制被整体换掉**——这是 K2 → K3 最大的一次跳变，也是本文的重点。

为什么非换不可？把 1M 上下文的账算出来（`code/02_kda_attn_budget.py`，其中的 MLA latent 与 head dim 是**示例值**，见 §1 第三步的说明）：

| | 每 token 存储 | 1M 单序列 |
|---|---|---|
| 24 个 Gated MLA 层（全局） | 随 $L$ **线性增长** | **29.0 GB** |
| 69 个 KDA 层（线性注意力） | **固定状态**，与 $L$ 无关 | **217 MB** |
| 若 93 层全是 MLA | — | **112.3 GB** |

**3.875 倍的差距**（= 93/24，一个不含任何假设的数）。这就是 KDA 的卖点：它省掉的不是「一部分 KV」，而是**整个随长度增长的那一项**。

## 1. 原理

### 第一步：稀疏度 scaling law —— K2 的选择

K2 报告把「稀疏度」定义成一个可优化的量：

$$\text{sparsity} = \frac{\#\text{experts total}}{\#\text{experts active per token}} = \frac{384}{8} = 48$$

它的架构结论有两条值得记住（都是 scaling law 实验的直接产物）：

- **继续提高稀疏度仍有收益** → 专家数从 DeepSeek-V3 的 256 提到 **384**；
- **为了降低推理开销，注意力头数从 DeepSeek-V3 的 128 砍到 64**。

第一条在 K3 上被推到 **56**（896/16）。但把 896 个专家塞进 2.78T 的预算里，本身就有一个硬约束——见第二步。

### 第二步：LatentMoE —— 让「896 选 16」在算力上可承受

常规 MoE 里每个被选中的 expert 收到的是**满宽** $d$ 维的 token 表示，所以**通信量与 expert 权重流量都随「路由重数」增长**。对 896 选 16 来说这不可承受。

**LatentMoE 把「模型宽度」与「routed expert 宽度」分开**：shared expert 保留满宽路径，routed expert 在压缩的 latent 空间 $\ell$ 上工作。

$${u} = \sum_{i \in \mathcal{T}_k({x})} p_i E_i^{\text{routed}}(\mathbf{W}^{\downarrow}{x}), \qquad {y} = \sum_{j=1}^{N_s} E_j^{\text{shared}}({x}) + \mathbf{W}^{\uparrow}\operatorname{RMSNorm}({u})$$

K3 的取值是 $\ell = 3584 = d/2$、expert hidden $= 3072$、$N_s = 2$。这笔账是可以算的（`code/01_param_budget.py`）：

| 读法 | routed expert 参数 | 与官方 2.78T 的关系 |
|---|---|---|
| **latent（$\ell = 3584$）** | **2.75T** | MoE FFN 合计 **2.7694T**，与官方差 **+0.38%** |
| 满宽（$d = 7168$） | 5.50T | **2.00×**——单是专家就超过整个预算 |

所以 **LatentMoE 不是一个优化项，而是「896 专家」这个设计的前提**。

代价也真实存在：$\ell = d/2$ 意味着每个 routed expert 的表达力被砍半。官方为此加了**三个稳定器**，因为极端稀疏会放大两种失败模式：

| 失败模式 | 稳定器 |
|---|---|
| routed 路径上「$\mathbf{W}^\downarrow$ → 专家 FFN → $\mathbf{W}^\uparrow$」接近**四次连续矩阵乘**，在 2.8T 规模下内部激活爆炸 | up-projection 前插 **RMSNorm**（Normalized LatentMoE）+ **SiTU-GLU**（Sigmoid Tanh Unit GLU） |
| 近 $10^3$ 个专家的负载均衡**超出 auxiliary-loss-free 偏置更新的适用区** | **Quantile Balancing (QB)**：用分位数代替固定步长偏置 |

### 第三步：KDA —— 把 $O(L)$ 的缓存换成 $O(1)$ 的状态

K3 的每个块是 **3 个 KDA 层 + 1 个 Gated MLA 层**（3:1），并在主干末尾**额外加一个 Gated MLA 层**，保证最后一层总是全局注意力。

KDA（Kimi Delta Attention）的定义（K3 报告式 (1)）：

$$\mathbf{S}_t = \left(\mathbf{I} - \beta_t{k}_t{k}_t^{\top}\right)\operatorname{Diag}({\alpha}_t)\mathbf{S}_{t-1} + \beta_t{k}_t{v}_t^{\top}, \qquad \tilde{{o}}_t = \mathbf{S}_t^{\top}{q}_t$$

三个部件各司其职：

- $\operatorname{Diag}({\alpha}_t)$ 是**逐通道 forget gate**（${\alpha}_t \in (0,1)^{d_k}$）——它决定「历史以多快的速度被遗忘」；
- $(\mathbf{I} - \beta_t{k}_t{k}_t^{\top})$ 是 **delta rule** 的写入项——它先**擦掉**当前 key 方向上的旧信息，再写入新值，而不是像普通线性注意力那样直接相加；
- $\beta_t \in (0,1)$ 是写入强度。

关键结论：**状态 $\mathbf{S}_t \in \mathbb{R}^{d_k \times d_v}$ 的大小与序列长度无关**。每 token 的计算也从「读 $L$ 个 KV」变成「读一个固定状态」。

**这笔交易在哪个长度上开始划算？** 每层每 token 的状态读取量是 $n_h d_k d_v$（固定），MLA 是 reading $L$ 个 latent（随 $L$ 增长）：

$$L^{*} = \frac{n_h\,d_k\,d_v}{d_{\text{latent}}} \quad\xrightarrow{\;n_h=96,\ d_k=d_v=128,\ d_{\text{latent}}=576\;}\quad 2731\ \text{token}$$

**约 2 700 token 之后 KDA 的状态读取量就低于 MLA**。有意思的是这个量级与 DSA 的 top-k（2048）很接近——两条完全不同的路线在同一个长度尺度上开始生效，因为它们的门槛都由「一个固定预算」与「线性增长的项」的交叉决定。

> ⚠ **两个必须标为示例的宽度**：K3 报告 Table 1 给了注意力头数（96），但**没有给 head dim 与 MLA 的 latent 维**。本仓库的编号程序用 $d_k=d_v=128$、$d_{\text{latent}}=576$（沿用 K2 / DeepSeek MLA 的「512 latent + 64 解耦 RoPE」）做**示例推导**，并把这两个数在代码与文档里逐处标注。**上表那个 3.875× 的比值不依赖它们**（它是 93/24），可以安全引用；具体到 GB 与 MB 的数字则依赖。

### 第四步：衰减的两个硬边界——快也不行，慢也不行

KDA 的位置信息是**隐式**的：距离 $\Delta$ 的贡献被 $\prod {\alpha}$ 加权。若把 ${\alpha}$ 近似成常数 $\alpha$，有效记忆长度就是一个几何级数的期望长度：

$$L_{\text{eff}} = \frac{1}{1-\alpha}$$

这条式子给出两个方向的失败（`code/02_kda_attn_budget.py` 的 B5/C1）：

| $\alpha$ | 有效记忆 | 后果 |
|---|---|---|
| 0.999 | **1 000** token | 1M 上下文「白给」——模型根本看不到远处 |
| 0.999 999 | **1 000 000** token | 恰好覆盖 1M——但门控必须被精密控制 |

而**慢**的那一侧撞上的是数值问题。KDA 的 chunk 内并行形式要把每个 chunk 的 key 按**累积衰减的倒数** $1/{\Gamma}$ 重标定：

$${\gamma}^{i\to j} = \prod_{r=i}^{j}{\alpha}^{r}, \qquad \mathbf{A} = \operatorname{Tril}\!\left[(\mathbf{Q}\odot{\Gamma})(\mathbf{K}/{\Gamma})^{\top}\right]$$

${\Gamma}$ 是 $(0,1)$ 内若干因子之积，所以 $1/{\Gamma}$ **可以无界增长**。取 chunk 大小 $C = 64$、$\alpha = 0.5$：

$$\frac{1}{\Gamma} = 0.5^{-64} = 1.8\times10^{19} \ \gg\ 65504\ (\text{fp16 上限})$$

临界值是 $\alpha_{\min} = 65504^{-1/64} \approx 0.841$——**$\alpha$ 小于它、又用 fp16 算，必然溢出**。这就是报告里 **"lower-bounded decay"** 那一节存在的理由：Kimi Linear 用**对数空间**计算相对衰减并给衰减设下界。**KDA 的衰减不是一个自由的超参，它是一个被数值精度夹住的设计变量。**

### 第五步：AttnRes —— 把「逐层累加」换成「对深度做注意力」

K3 的第二个架构更新针对的是**深度方向**。报告的动机写得很直接：

> standard residual connections compress all prior information into a single state over depth — a bottleneck reminiscent of **RNNs over time**.

Transformer 当年用注意力替掉了时间方向的递推；AttnRes 把这套方法**搬到深度方向**：每层用一个可学的伪 query $\bm{q}_l = \bm{w}_l$，对 **embedding 与所有前层输出**做 softmax 注意力：

$$\alpha_{i\to l} = \frac{\phi({q}_l, {k}_i)}{\sum_{j=0}^{l-1}\phi({q}_l,{k}_j)}, \qquad \phi({q},{k}) = \exp\!\left({q}^{\top}\operatorname{RMSNorm}({k})\right), \qquad {h}_l = \sum_{i=0}^{l-1}\alpha_{i\to l}{v}_i$$

（${k}_i = {v}_i$；$i=0$ 取 token embedding。打分里那个 RMSNorm 是为了防止「输出幅度大的层主导权重」。）

代价的结构很特别（`code/02_kda_attn_budget.py` 的 C6）：

| | 量级 | 评价 |
|---|---|---|
| 算力 | $O(L^2 d)$ | **便宜**（$L<100$，$93^2\times7168 \approx 62$M 单位/token） |
| 显存 / 通信 | $O(Ld)$ | **贵**——必须把所有层的输出**留着**（$93d \approx 0.67$M 元素/token） |

所以瓶颈在显存而不是算力。解法是 **Block AttnRes**：把 $L$ 层分成 $N$ 块各 $S = L/N$ 层，块内输出**求和**成一个表示，只在 $N$ 个块级表示之间做完整注意力——显存与通信从 $O(Ld)$ 降到 $O(Nd)$，**降幅恰为 $S$**（每块 4 层 → 3.88×，8 层 → 7.75×，16 层 → 15.5×）。

### 第六步：位置与优化器

**NoPE。** K3 的 MLA 层**不做任何显式位置编码**——位置信息由 KDA 的递推门控与衰减**隐式承载**。好处是**不需要任何 RoPE 缩放或插值就能直接外推到 1M**。这不是「删掉位置信息」（§4.4），而是换了一个承载者。

**Per-Head Muon。** K2 的贡献之一是 **MuonClip**（Muon 优化器 + QK-clip，用来驯服 Muon 在放大规模时的不稳定）。K3 沿用其 weight clipping，并把注意力投影的正交化**按头切块**：整矩阵正交化会把所有头当成一个耦合块，**尺度大的头主导共享更新方向**；按头分别做 Newton–Schulz 更公平，附带好处是矮块比整矩阵便宜，**优化器开销还略降**。

### 第七步：稀疏度与参数账

把 §1 第一步到第六步的配置代进参数账（`code/01_param_budget.py`），K3 的 MoE FFN 单独就能算到官方总参：

| 项 | K2 | **K3** |
|---|---|---|
| 每层 MoE 总参 | 16.955B | **29.779B** |
| 每层 MoE 激活 | 396.4M | **712.0M** |
| MoE FFN 合计 | 1.0174T（60 MoE + 1 dense）/ 1.0343T（61 层全 MoE） | **2.7694T** |
| 与官方总参的偏差 | 官方 1.04T 落在 [1.0174, 1.0465] T 内 | **+0.38%** |
| FFN 占官方激活的比例 | 74.2% | 63.5% |
| 反解出的注意力余量 | 138M/层 | **408M/层** |

两个读数很说明问题：K2 的 FFN 吃掉激活量的 3/4，K3 降到约 2/3——**注意力那部分（KDA 的状态投影 + 全局 MLA）在 K3 里明显变重了**，这与「用状态换缓存」的设计是一致的：省了缓存，但要多花投影与状态更新的算力。

## 2. 执行映射

KDA 的理论优势要落地，得先解决「递推在硬件上怎么并行」。K3 的做法是**按阶段换形态**：

| 阶段 | 形态 | 并行方式 |
|---|---|---|
| 训练 / prefill | **chunkwise**（chunk 内并行、chunk 间递推，靠 UT transform） | 序列维切分 + 上下文并行 |
| 超长 prefill | **intra-device context parallelism** | 单机内切序列 |
| 跨设备长 prefill | **KDA context parallelism** | 跨设备切序列 |
| decode | 逐 token 递推 | 天然串行，但每步只读固定状态 |

**训练侧的三个规模问题**（K3 报告 §5.2 的原话要点）：

1. **完美均衡的 EP**：用**有界冗余专家**做到「完美均衡」，配 online planning、**零拷贝通信**、以及**静态形状的无同步执行**（避免动态 shape 带来的同步开销）；
2. **显存**：统一的 activation manager、memory-efficient MoE、**memory-efficient AttnRes**、跨 PP rank 平衡激活、**pipeline ZeRO-2 梯度分片与卸载**、以及 **P2P 的 Muon 正交化**（把正交化通信压到点对点）；
3. **多模态编码器**：编码器内的动态 CP，并**把编码器计算塞进 PP 气泡里**。

**RL 侧（1M agentic RL）**：外部 KV cache pool + rollout 自动限流调度器 + 为非策略模型前向复用梯度缓冲。

**部署侧的量化**（§1 第七步的账）：expert 权重 **MXFP4**、激活 **MXFP8**、非 expert 组件保持更高精度；**QAT 覆盖整个后训练阶段（SFT + RL）**，且 **RL 的 rollout 与 training 共用同一套量化方案**——消除训练-推理不一致。

有效位宽要算对：MXFP4 是 **4 bit 元素 + 每 32 个元素一个 8 bit 的 E8M0 scale**，所以**每参数是 4.25 bit（0.53125 字节）而不是 0.5 字节**。忽略这一项会让专家权重的字节数**低估 6.25%**（86 GB）——详见 §4.5。

| 量 | 值 |
|---|---|
| expert 权重字节（MXFP4） | **1.462 TB** |
| 全模型混合精度合计 | **1.518 TB**（全 BF16 需 5.56 TB，**3.66×**） |
| 1M 上下文 KDA 状态合计（示例宽度） | 217 MB（占全局 KV 的 0.75%） |

## 3. 实现与演进

| 阶段 | 模型 | 改了什么 | 为什么必须走到下一步 |
|---|---|---|---|
| 推理 | Kimi k1.5（2025-01） | 用 RL 激发长链式推理；多模态；128K | 推理链变长 → 上下文变长 → KV 成本成主矛盾 |
| 稀疏 MoE | **Kimi K2**（2025-07） | 1.04T / 32.6B；384 选 8（**稀疏度 48**）；MLA；头数砍到 64；**MuonClip**；15.5T token（WSD） | 稀疏度还有收益，但要继续加密就得先解决「routed 路径太宽」 |
| 换注意力 | **Kimi Linear**（2025-10） | 开源 48B-A3B，提出 **KDA** | KDA 在小模型上验证过了，需要一个 3T 级载体 |
| 交替迭代 | K2 Thinking / K2.5 等 | 思考模式、视觉接入、编码强化 | 能力在涨，但上下文仍停在 128K–256K |
| **规模 + 机制双跳** | **Kimi K3**（2026-07） | 2.78T / 104.2B；**93 层 = 69 KDA + 24 Gated MLA**；**AttnRes**；**Stable LatentMoE**（896 选 16，稀疏度 **56**）；**NoPE**；原生多模态；1M；**MXFP4 QAT** | —— |

三处关键跳变的机理：

**K2：把稳定性当成一等公民。** Muon 在放大规模时会不稳，K2 的解法是 **MuonClip**——Muon 优化器加 **QK-clip**（一旦 attention logits 越界就按比例缩回）。这条线索一直延续到 K3 的 **Per-Head Muon**。

**K2 → K3：三个方向同时扩信息流。** K3 报告的自我总结把这三点排得很清楚——**序列方向**用 KDA 做高效 token 混合、**深度方向**用 AttnRes 让每层能回看所有前层、**宽度方向**用 Stable LatentMoE 把通道混合推到 896 选 16。加上重调的数据与训练配方，官方称整体 **scaling 效率比 K2 提升约 2.5×**。

> ⚠ 「2.5× scaling 效率」是**经验结果**（held-out OOD 验证上的 loss 曲线），**架构账算不出来**。本仓库把它记为**引用值**，不做反推。

**K3 的长上下文是「课程 + 合成」两条腿。** 上下文窗口**渐进扩展**：预训练从 **8k → 64k**，cooldown 阶段从 **256K → 1M**（四阶段课程），把昂贵的长序列计算集中到总预算的一小部分。数据侧除了清洗与上采样，还**合成**长上下文样本——刻意重排拼接多模态文档与子任务，使任务**只能靠跨 1M 上下文的分散信息解出**。报告对此的判断值得引用：**长度本身不带来长程能力**（"Length alone, however, does not confer long-range capability"）。

## 4. 踩坑与失败模式

### 4.1 把线性注意力当成「无损的省显存」

**现象**：认为 KDA 就是「把 KV 换成状态」，省下来全是净收益。
**成因**：省的是**随长度增长的那一项**，代价有两处：① 每层仍要存固定状态（1M 上下文下 69 层合计 217 MB，不是零）；② 线性注意力**用固定秩的状态近似注意力矩阵**，表达力有上限——K3 并没有全押线性，而是**每 4 层留 1 层全局 MLA**，并在末尾再加一层。
**规避**：看到「linear / SSM / 递推状态」这类词，同时问三件事：状态多大、保留多少全局层、代价在哪个长度尺度上被证明可接受。

### 4.2 把 sparsity 当成参数效率

**现象**：看到「稀疏度 56」就认为「参数利用率比稀疏度 48 更高」。
**成因**：官方口径的 sparsity 只是**总专家数 ÷ 每 token 激活数**，是**路由稀疏度**，与「参数量换来多少智能」无关。K3 的稀疏度提高的同时**总参也涨了 167%**——两者不是一回事。
**规避**：把 sparsity、active/total 比、以及 scaling 效率三者分开。判据：$E/k$ 只描述路由形状。

### 4.3 把 LatentMoE 当成普通 MLP 压缩

**现象**：以为 LatentMoE 只是「给专家减参」，于是也把它算成一种量化/剪枝。
**成因**：LatentMoE 改的是**计算图**——shared expert 走满宽、routed expert 在 latent 上算，中间由**每层共享的** $\mathbf{W}^\downarrow / \mathbf{W}^\uparrow$ 连接。它同时改变了**通信载荷**（每 token 每 expert 从 $d$ 降到 $\ell$）与**权重流量**，而不只是参数个数。
**规避**：区分「参数账」与「通信账」。LatentMoE 的主要收益在后者；判据是「routed 路径的载荷宽度」是否小于 hidden 宽度。

### 4.4 以为 NoPE 等于「没有位置信息」

**现象**：看到 K3 不用 RoPE，就认为模型对顺序不敏感，或者认为外推能力是白来的。
**成因**：位置信息被**转移**给了 KDA 的衰减：距离 $\Delta$ 的贡献按 $\prod\bm{\alpha}$ 加权。这带来「无需 RoPE 插值即可外推 1M」的好处，也带来「衰减必须被精密控制」的成本——$\alpha$ 太小学生看不见远处（有效记忆 1 000 token），太大则 $1/\Gamma$ 溢出（$\alpha<0.841$ 配 fp16、$C=64$ 必溢）。
**规避**：把「隐式位置编码」当成一个有参数的机制去看，而不是当成「没有」。

### 4.5 把 MXFP4 当成「正好 0.5 字节/参数」

**现象**：按 4 bit = 0.5 字节算 expert 权重，得出 1.376 TB。
**成因**：MXFP4（OCP 微缩放格式）是 **E2M1 元素 + 每 32 个元素一个 E8M0 scale**，所以**有效位宽是 $4 + 8/32 = 4.25$ bit**，即 0.53125 字节/参数。漏掉 scale 会把 1.462 TB 低估成 1.376 TB（**差 86 GB，6.25%**）。
**规避**：凡遇到 MX / 微缩放格式，把「元素位宽」与「scale 开销」分别记账。判据：每 32 个元素一个 8 bit scale ⇒ +0.25 bit/元素。

### 4.6 用 K3 的框架解释 K2 的行为

**现象**：把 KDA、AttnRes、LatentMoE 当成 Kimi 家族的通用设计，回头解释 K2 的 1.04T 或 128K。
**成因**：这三样都是 **K3 引入**的。K2 是纯 MLA + 常规残差 + 普通（非 latent）MoE。K2.5 才接上视觉，KDA 先在 Kimi Linear（48B-A3B）上验证。
**规避**：跨代引用时先确认机制的引入世代。判据：K3 报告 Table 1 的「$\Delta$」列——标了 `–` 的行就是 K3 新增。

## 编译与运行

```sh
cd code
make run                    # 默认 python3 + numpy，依次跑 01 / 02
make 01_param_budget        # 只跑参数与激活量核算
make 02_kda_attn_budget     # 只跑 KDA / 混合注意力 / AttnRes 成本账
make clean
```

本文的 2.7694T（K3 FFN）、1.0174T–1.0465T 的 K2 区间、1.518 TB 的 MXFP4 字节账，来自 **`code/01_param_budget.py`**；3.875× 的 KV 削减、2 731 token 的 KDA/MLA 交叉点、$1/(1-\alpha)$ 的有效记忆、$1/\Gamma$ 的 fp16 溢出边界、AttnRes 的 $O(Ld)\to O(Nd)$，来自 **`code/02_kda_attn_budget.py`**。两个程序只用 `math` 与 `numpy`，**不加载权重、不测硬件**，产出是**推导值**。

## 5. 判据表

本域无 GPU、知识对象以方法与规格为主，§5 采用**判据表**形态（理由见 `160-model/README.md`）。

| 可机械判定的条件 | 结论 |
|---|---|
| 规格给出 KDA / 全局层数（如 69 + 24 = 93） | 随长度增长的 KV 削减比 = 总层数 / 全局层数；**该比值与 latent / head dim 无关** |
| 规格给出 routed 专家的 latent 宽度 $\ell < d$ | 是 LatentMoE 类设计；参数账用 $\ell$，通信账用「载荷宽度」而非 $d$ |
| 规格给出「稀疏度」但未说定义 | 默认 = 总专家 / 每 token 激活数（路由形状）；**不得当作参数效率** |
| 注意力里出现 per-channel decay $\bm{\alpha}$ 且无位置编码 | 位置信息是隐式的；必须同时检查衰减的上下界（有效记忆 vs $1/\Gamma$ 溢出） |
| chunkwise 线性注意力 + 有限精度（fp16）+ 未提对数空间 | 高概率在 $1/\Gamma$ 上溢出；临界 $\alpha \approx (\text{fp16 max})^{-1/C}$ |
| 残差被描述成「对前层做注意力」 | 算力 $O(L^2d)$、**显存 $O(Ld)$**；瓶颈在显存，需 block 化降到 $O(Nd)$ |
| 规格出现「MXFP4 / MXFP8」等微缩放格式 | 有效位宽 = 元素位宽 + scale 位宽 / 块大小；**不得按元素位宽直接换算字节** |
| 官方给出「scaling 效率提升 N×」 | 经验结果，架构账算不出来；记为引用值 |
| 某机制被用来解释更早的世代 | 先用该世代报告的表格确认它是否已存在（看「新增 / 变化」列） |

## 6. 小结与延伸

- **Kimi 的答案与另两家正交**：DeepSeek 压缓存（CSA2 + FP4，890 B/token）、GLM 挑位置（DSA + IndexShare），**Kimi 换掉缓存**（KDA，固定状态）。
- **K2 → K3 是「更深、更稀、换注意力」**：层数 61 → 93、hidden 不变、稀疏度 48 → 56、注意力从纯 MLA 换成 69 KDA + 24 Gated MLA。
- **四个可复算的数**：3.875× 的 KV 削减（= 93/24，不含假设）；KDA/MLA 的状态读取交叉点 ≈ **2 731** token；MXFP4 的有效位宽 **4.25 bit**（含 scale）；AttnRes 的 block 化省 **$S$** 倍显存。
- **两个必须标注的示例值**：MLA latent 与 head dim（K3 报告未公布）——凡是依赖它们的 GB / MB 数字都带假设，那个 3.875× 不带。

下一步：`160-model/kimi/02-k3-architecture` 逐部件拆 K3（KDA 的 chunkwise 形式与下界衰减、AttnRes 的 full/block、Stable LatentMoE 的三个稳定器、MXFP4 QAT）；横向对照见 `160-model/deepseek/01-technical-route`（压缩式路线）与 `160-model/glm/01-glm-technical-route`（选择式路线）。

## 参考

- 【C】Kimi Team, *Kimi K3: Open Frontier Intelligence*, arXiv:[2607.24653v2](https://arxiv.org/abs/2607.24653)（**预印本，未确认同行评审**）。Table 1、§2.1–§2.5、§3.2–§3.4、§4.1.4、§5.1–§5.3。
- 【C】Kimi Team, *Kimi K2: Open Agentic Intelligence*, arXiv:[2507.20534v2](https://arxiv.org/abs/2507.20534)。Table 2、§2.1（MuonClip）、§2.3（稀疏度 scaling law）、§2.5（训练配方）。
- 【B】[kimi.ai/blog/kimi-k3](https://www.kimi.ai/blog/kimi-k3)（官方发布博客：2.8T、KDA + AttnRes、1M、权重时间点）。
- 【B】官方模型卡 `moonshotai/Kimi-Linear-48B-A3B-Instruct`（KDA 的开源载体）。
- 【C】*Kimi Linear: An Expressive, Efficient Attention Architecture*, arXiv:[2510.26692](https://arxiv.org/abs/2510.26692)（KDA 的提出处；**本仓库未逐字读其正文**）。
- 【B】MoonshotAI 官方仓库 [Attention-Residuals](https://github.com/MoonshotAI/Attention-Residuals)（AttnRes 的论文与实现）。
- 【D】Kimi K2.5 / K2.6 / K2.7 的时间点来自媒体转述与推理服务商在架信息，**本仓库未取到官方发布页**，正文只作谱系参考。
- 相关域：[80-model-architectures/attention-variants](../../80-model-architectures/attention-variants/)、[80-model-architectures/moe](../../80-model-architectures/moe/)、[100-parallelism/expert-parallel](../../100-parallelism/expert-parallel/)。

---
title: Kimi K3 实例分析：KDA + AttnRes + Stable LatentMoE
slug: k3-architecture
domain: 160-model
topic: kimi
level: 3
difficulty: advanced
prerequisites:
  - 160-model/kimi/01-technical-route
  - 80-model-architectures/attention-variants/02-mla-low-rank
related:
  - 160-model/deepseek/02-v4-1-flash-architecture
  - 100-parallelism/expert-parallel/02-load-imbalance
  - 130-model-optimization/quantization/02-e8m0-and-mx
code:
  - code/02_kda_attn_budget.py
status: stable
updated: 2026-10-13
---

# Kimi K3 实例分析：KDA + AttnRes + Stable LatentMoE

> K3 的架构可以概括成一句话：**沿三个方向同时扩展信息流**——序列方向用 **KDA**（固定状态替代增长的缓存）、深度方向用 **AttnRes**（每层对前层做注意力，而不只是累加）、宽度方向用 **Stable LatentMoE**（896 选 16，靠 latent 宽度与三个稳定器撑住）。本文把这九个部件逐个拆开，并标出哪些数能复算、哪些不能。

## 0. 背景与动机

K3 是**首个开源的 3T 级模型**（2.78T 总参 / 104.2B 激活），93 层、1M 上下文、原生多模态。官方发布博客也如实写了它的定位：

> While its overall performance still trails the most powerful proprietary models, Claude Fable 5 and GPT 5.6 Sol, Kimi K3 demonstrated frontier-level performance across our evaluation suite.

所以本文关心的不是「它有多强」，而是**它把 2.78T 的参数与 1M 的上下文撑起来靠了什么机制**。三个方向的对账单（`code/02_kda_attn_budget.py`，其中 MLA latent 与 head dim 是**示例值**）：

| 方向 | 机制 | 可复算的收益 |
|---|---|---|
| **序列** | 93 层里 69 层换成 KDA | 1M 全局 KV 从 112.3 GB 降到 **29.0 GB**（**3.875×**，= 93/24，不含假设） |
| **深度** | AttnRes（full → block） | 显存 / 通信从 $O(Ld)$ 降到 $O(Nd)$，降幅 **$= S$**（每块层数） |
| **宽度** | Stable LatentMoE（896 选 16） | 若 routed expert 走满宽 $d$，专家参数会变成 **2.00×**（5.50T > 整个预算） |

## 1. 原理

### 第一步：KDA 的递推定义

KDA 在 delta-rule 递推上加了一个**逐通道 forget gate**：

$$\mathbf{S}_t = \left(\mathbf{I} - \beta_tk_tk_t^{\top}\right)\operatorname{Diag}(\alpha_t)\mathbf{S}_{t-1} + \beta_tk_tv_t^{\top}, \qquad \tilde{o}_t = \mathbf{S}_t^{\top}q_t$$

逐项看它的三重身份：

1. **它是线性注意力**：$\mathbf{S}_t \in \mathbb{R}^{d_k\times d_v}$ 的大小只由 head 维决定，**与序列长度无关**。这是整个 K3 长上下文方案的基石。
2. **它是 delta rule**：$(\mathbf{I}-\beta_tk_tk_t^{\top})$ 先**擦掉**当前 key 方向上的旧信息再写入新值，而不是像普通线性注意力那样无条件累加。这把状态从「累加器」变成了「可更新的记忆」。
3. **它带逐通道衰减**：$\operatorname{Diag}(\alpha_t)$ 让**每个 key 通道**有各自的一步保留率——这就是「Delta」之外 Kimi 加的那一项，也是 KDA 与 GDN / Mamba-2 的分界。

**每头参数怎么来**（报告式 (2)）：

$$q^h_t,k^h_t = \operatorname{L_2Norm}\!\big(\operatorname{Swish}(\operatorname{ShortConv}(\mathbf{W}^h_{q/k}x_t))\big), \quad v^h_t = \operatorname{Swish}(\operatorname{ShortConv}(\mathbf{W}^h_vx_t))$$
$$\beta^h_t = \operatorname{Sigmoid}(\mathbf{W}^{h}_{\beta}x_t), \qquad z^h_t = \mathbf{W}^{\uparrow}_{\alpha}\mathbf{W}^{\downarrow}_{\alpha}x_t + b^h_{\alpha}$$

注意 $\alpha$ 的生成路径：**低秩投影 + 每头偏置**。低秩省参数，每头偏置给不同头不同的衰减起点。

### 第二步：chunkwise 形式与「下界衰减」——KDA 最不显眼也最关键的一步

KDA 不能逐 token 串行训。它的并行化方式是 **chunk 内并行、chunk 间递推**。设 chunk 大小 $C$，累积衰减

$$\gamma^{i\to j}_{[t]} := \prod_{r=i}^{j}\alpha^{r}_{[t]}, \qquad \Gamma^{1\to C}_{[t]} = \text{把 }\gamma^{1},\dots,\gamma^{C}\text{ 按行堆叠}$$

则（报告式 (4)）：

$$\mathbf{A}_{[t]} = \operatorname{Tril}\!\left[(\mathbf{Q}_{[t]}\odot\Gamma^{1\to C}_{[t]})(\mathbf{K}_{[t]}/\Gamma^{1\to C}_{[t]})^{\top}\right], \qquad \mathbf{O}_{[t]} = \underbrace{(\Gamma\odot\mathbf{Q})\mathbf{S}_{[t]}}_{\text{inter-chunk}} + \underbrace{\mathbf{A}_{[t]}\widetilde{\mathbf{V}}_{[t]}}_{\text{intra-chunk}}$$

这个形式的**代价**就藏在那个除法里：$\Gamma$ 是 $(0,1)$ 内若干因子之积，所以 $1/\Gamma$ **可以无界增长**。取 $C=64$、$\alpha=0.5$：

$$\frac{1}{\Gamma} = 0.5^{-64} = 1.8\times10^{19} \gg 65504\ (\text{fp16 上限})$$

临界值 $\alpha_{\min} = 65504^{-1/64} \approx \mathbf{0.841}$——**$\alpha$ 低于它、又用 fp16 算，必然溢出**。

于是有两层对策：

| | 做法 | 效果 |
|---|---|---|
| Kimi Linear | 用**对数空间**算相对衰减，并把 chunk 再切成 **16-token 的次级 tile** | 非对角 tile 可用 Tensor Core 的**稠密矩阵乘**；对角 tile 仍需显式位置对计算（chunk 内的主要瓶颈） |
| **K3 的改动** | 把 $z\to$ log-decay 的映射从 Kimi Linear 的 negative-Softplus（下界 $-\infty$）换成**缩放的 Sigmoid** | |

$$g^h_t = g_{\min}\operatorname{Sigmoid}\!\left(e^{A_h}z^h_t\right) \in (g_{\min},\,0)^{d_k}, \qquad \alpha^h_t = \exp(g^h_t) \in (e^{g_{\min}},\,1)^{d_k}$$

**这一步把 log-decay 从下方夹住了**，于是 $1/\Gamma$ 有了上界——「lower-bounded decay」不是数值 hack，而是**让 $\alpha$ 的可用区间成为可判定的事**：$\alpha$ 太小则模型看不见远处（有效记忆 $1/(1-\alpha)$ 只有 1 000 token），太大则 $1/\Gamma$ 溢出。**它被两个边界同时夹住。**

### 第三步：满秩输出门

K3 最后一个 KDA 改动是把输出门从 Kimi Linear 的**低秩**参数化换成**输入相关的满秩**投影：

$$y_t = \mathbf{W}_o\Big[\operatorname{Sigmoid}(\mathbf{W}_gx_t) \odot \operatorname{RMSNorm}(\tilde{o}_t)\Big]$$

先对递推输出做**逐头 RMSNorm**，再乘一个由输入决定的门。这与 §1 第四步的 RMSNorm 是同一类手法：**在递推路径上插入归一化以控制尺度**。

### 第四步：Gated MLA 与 NoPE

K3 每 4 层保留 1 层 **Gated MLA**（并在主干末尾再加 1 层），负责**全局 token-to-token 交互**。MLA 的定义与 DeepSeek-V2 一致：把每 token 的 K/V 压成一个低维 latent $c_t = \mathbf{W}_cx_t$，注意力时再用学到的上投影还原——**压的是缓存占用的字节，不是交互的范围**。

关键差异在位置编码：**K3 对所有 MLA 层用 NoPE（不做任何显式位置编码）**，位置信息由 KDA 的递推门控与衰减**隐式承载**。好处是官方明确写出的那一条：**不需要 RoPE 缩放或插值就能直接外推到 1M**。

### 第五步：AttnRes —— 深度方向的注意力

常规残差把「所有先验信息」压缩进**一个状态** $h_l$——报告的原话是这「like a bottleneck reminiscent of RNNs over time」。Transformer 当年用注意力替掉了时间方向的递推；**AttnRes 把这套方法搬到深度方向**：

$$q_l = w_l, \qquad k_i = v_i = \begin{cases}h_1 & i = 0\\ f_i(h_i) & 1 \le i \le l-1\end{cases}$$
$$\alpha_{i\to l} = \frac{\phi(q_l,k_i)}{\sum_{j=0}^{l-1}\phi(q_l,k_j)}, \qquad \phi(q,k) = \exp\!\left(q^{\top}\operatorname{RMSNorm}(k)\right), \qquad h_l = \sum_{i=0}^{l-1}\alpha_{i\to l}v_i$$

三个设计细节值得单独看：

- **伪 query** $w_l$ 是**每层一个可学向量**——不是从输入算出来的，所以「往哪看」是层的固有属性；
- $i=0$ 是 token embedding，**永远是一个候选来源**（残差流不断根）；
- 打分里的 **RMSNorm** 是为了防止「输出幅度大的层主导权重」。

代价结构很特别：

| | 量级 | 在 K3 上的量 |
|---|---|---|
| 算力 | $O(L^2d)$ | $93^2\times7168 \approx 6.2\times10^7$ 单位/token——**便宜** |
| 显存 / 通信 | $O(Ld)$ | $93\times7168 \approx 6.7\times10^5$ 元素/token——**贵**，因为要留住**所有层**的输出 |

所以瓶颈在显存。解法是 **Block AttnRes**：把 $L$ 层分成 $N$ 块各 $S = L/N$ 层，块内输出**求和**成一个表示 $b_n$，只在 $N$ 个块级表示之间做完整注意力（第 $n$ 块第 $i$ 层的 value 矩阵是 $[b_0,\dots,b_{n-1}]$，$i\ge2$ 时额外追加块内部分和 $b^{i-1}_n$）。**显存与通信从 $O(Ld)$ 降到 $O(Nd)$，降幅恰为 $S$**（4 层/块 → 3.88×，8 层 → 7.75×，16 层 → 15.5×）。

> ⚠ **UNRESOLVED**：报告在本仓库读到的章节里给出了 Block AttnRes 的形式与 $O(Nd)$ 的结论，但**没有给出 K3 实际用的 $N$（或 $S$）**。所以「K3 省了多少 AttnRes 显存」是一个**参数化的结论**，本仓库不替它填数。

### 第六步：Stable LatentMoE 的三个稳定器

LatentMoE 的结构与动机见 `160-model/kimi/01-technical-route` §1 第二步（shared 走满宽、routed 在 $\ell = 3584$ 上算，$\mathbf{W}^\downarrow/\mathbf{W}^\uparrow$ 每层共享）。这里补它**为什么需要「Stable」**——极端稀疏放大了两个失败模式：

| 失败模式 | 机理 | 稳定器 |
|---|---|---|
| **内部激活爆炸** | routed 路径上 $\mathbf{W}^\downarrow \to$ 专家 FFN $\to \mathbf{W}^\uparrow$ 接近**四次连续矩阵乘**，条件数差；2.8T 规模下激活会炸 | ① up-projection 前插 **RMSNorm**（Normalized LatentMoE）② **SiTU-GLU**（Sigmoid Tanh Unit GLU，对两支做 tanh soft-cap） |
| **负载均衡失效** | 近 $10^3$ 个专家超出 auxiliary-loss-free 偏置更新的适用区 | ③ **Quantile Balancing (QB)**：用**分位数**而非固定步长偏置来定路由 |

数值上 QB 的取值来自**直方图估计**（报告给了单独的 "Histogram estimation" 小节）。shared expert 数固定为 $N_s = 2$。

### 第七步：Per-Head Muon 与原生视觉

**Per-Head Muon。** K2 用 MuonClip（Muon + QK-clip）驯服了放大规模的不稳定。K3 沿用其 weight clipping，并做一处细化：正交化时**沿 head 维切块**，每头单独做 Newton–Schulz。理由是整矩阵正交化把所有头当成一个耦合块，**动量尺度大的头会主导共享的更新方向**。附带好处是矮块比整矩阵便宜，优化器开销略降。

**原生视觉。** K3 的多模态是 **native** 的：语言与视觉从训练一开始就联合优化，而不是「先训语言模型、再后接一个视觉编码器做对齐」。视觉侧是 **MoonViT-V2**（401M / 27 层 / patch 14 / 12 头），做 $2\times2$ pixel-shuffle 把视觉 token 数降 4 倍，从而在 1M 上下文里支持到约 3584×3584 的输入。

## 2. 执行映射

**KDA 的 kernel 要按阶段换形态**（报告 §5.1 的算法-系统协同设计）：

| 阶段 | 形态 | 并行化 |
|---|---|---|
| 训练 / prefill | chunkwise（chunk 内并行 + chunk 间递推，UT transform） | 张量并行 + 序列切分 |
| 超长 prefill（单机内） | intra-device context parallelism | 单机内切序列 |
| 跨设备长 prefill | KDA context parallelism | 跨设备切序列 |
| decode | 逐 token 递推 | 每步只读**固定大小**的状态 |

一个 1M 上下文的解码步里，KDA 层读的状态是常数、MLA 层读的 KV 随 $L$ 增长——**混合架构把「随长度增长的那部分」限制在 24 层里**。

**3T 级预训练infra 的三组问题**（报告 §5.2）：

1. **EP 负载均衡**：用**有界冗余专家**做到「完美均衡」，配 online planning、**零拷贝通信**、以及**静态形状的无同步执行**；
2. **显存**：统一 activation manager、memory-efficient MoE、**memory-efficient AttnRes**（对应 §1 第五步那个 $O(Ld)$）、跨 PP rank 平衡激活、**pipeline ZeRO-2 梯度分片与卸载**、以及 **P2P 的 Muon 正交化**；
3. **多模态编码器**：编码器内的动态 CP，并**把编码器计算塞进 PP 气泡**。

**1M agentic RL infra**（§5.3）：**外部 KV cache pool**、rollout **自动限流调度器**、以及为非策略模型前向**复用梯度缓冲**；沙箱侧有一套独立的执行底座。

**部署感知的后训练**（§4.1.4）是本实例里最值得单独记的一条：**expert 权重 MXFP4、激活 MXFP8、非 expert 组件保持更高精度**，并且

- **QAT 覆盖整个后训练阶段**（SFT 与 RL 都做）；
- **RL 时 rollout 与 training 共用同一套量化方案**——消除训练-推理不一致。

字节账（`code/01_param_budget.py`）：MXFP4 的有效位宽是 **$4 + 8/32 = 4.25$ bit**（E2M1 元素 + 每 32 个元素一个 E8M0 scale）：

| | 字节 |
|---|---|
| expert 权重（MXFP4，4.25 bit） | **1.462 TB** |
| 其余参数（BF16 估） | 55 GB |
| **合计** | **1.518 TB**（全 BF16 需 5.56 TB，**3.66×**） |

## 3. 实现与演进

| | Kimi K2 | **Kimi K3** | 这一跳解决什么 |
|---|---|---|---|
| 注意力 | 61 层 MLA | **69 KDA + 24 Gated MLA** | 把随长度增长的 KV 限制在 1/4 的层里（**3.875×**） |
| 位置编码 | RoPE | **NoPE** | 免缩放/插值外推到 1M |
| 深度通路 | 常规残差 | **AttnRes**（full → block） | 每层可回看所有前层；block 化把显存降到 $O(Nd)$ |
| MoE | 384 选 8（稀疏度 48），满宽 expert | **896 选 16（稀疏度 56）**，latent $\ell=d/2$ | 在参数预算内容纳 896 个专家（满宽会变 2.00×） |
| 稳定化 | MuonClip | **Per-Head Muon + QB + RMSNorm + SiTU-GLU** | 极端稀疏与近四连乘下的数值稳定 |
| 上下文 | 128K | **1M**（8k→64k 预训练，256K→1M cooldown） | 「长度本身不带来长程能力」——配长文档上采样与**合成长上下文数据** |
| 多模态 | 无 | **原生**（MoonViT-V2，401M） | 视觉与语言从一开始联合优化 |

**官方口径的 2.5×**：报告称这些改动（架构 + 数据 + 训练配方）在 held-out OOD 验证上带来**约 2.5× 的整体 scaling 效率提升**。

> ⚠ 这是**经验结果**（loss 曲线），**架构账算不出来**。本仓库记为**引用值**。

**三点方法论**可以带走：

1. **结构改动的收益要用「它改的是哪一项」来记账**：KDA 改的是「随 $L$ 增长的那一项」，AttnRes 改的是「层输出的驻留量」，LatentMoE 改的是「routed 路径的载荷宽度」。
2. **数值格式的边界要和结构一起设计**：$\alpha$ 的可用区间由 $1/\Gamma$ 的溢出边界与有效记忆长度共同决定。
3. **量化要覆盖 rollout**：QAT 只做 SFT 而不覆盖 RL，会在 RL 阶段重新引入训练-推理不一致。

## 4. 踩坑与失败模式

### 4.1 把 chunkwise 形式读成「训练时也是逐 token 递推」

**现象**：因为 KDA 是递推的，就认为它训练时也必须串行，因此推断「线性注意力训练很慢」。
**成因**：KDA 的 chunkwise 形式**chunk 内并行、chunk 间递推**。真正限制它的是**对角 tile**——$(\mathbf{Q}\odot\Gamma)(\mathbf{K}/\Gamma)^{\top}$ 的对角块需要显式位置对计算，无法直接变成稠密矩阵乘；非对角 tile 才能在 Tensor Core 上跑。
**规避**：评估线性注意力的训练效率时，问「chunk 多大、对角 tile 怎么处理」，而不是笼统地说「递推=串行」。

### 4.2 把「lower-bounded decay」当成调参技巧

**现象**：看到 K3 把 negative-Softplus 换成 scaled Sigmoid，认为只是换了个激活函数。
**成因**：negative-Softplus 的值域是 $(-\infty, 0)$，也就是 $\alpha$ 可以无限接近 0，$1/\Gamma$ 因此**无上界**。换成 $g_{\min}\operatorname{Sigmoid}(\cdot) \in (g_{\min},0)$ 之后，$\alpha \in (e^{g_{\min}},1)$，$1/\Gamma$ **有了上界**。
**规避**：把衰减看成「被两个边界夹住的量」——下界由有效记忆长度定，上界由 $1/\Gamma$ 的有限精度定。判据：$\alpha_{\min} \approx (\text{fp16 max})^{-1/C}$；$C=64$ 时约 0.841。

### 4.3 把 AttnRes 当成「又加了一条 skip connection」

**现象**：把 AttnRes 归类成残差连接的变体，认为它只影响梯度流动。
**成因**：它改的是**显存**。Full AttnRes 要求**所有层的输出都留在手边**（$O(Ld)$），在 93 层、hidden 7168 的规模下这是每 token 约 67 万元素的激活驻留量——**这才是它必须 block 化的原因**，而不是算力（$O(L^2d)$ 只有 6 千万单位/token，很便宜）。
**规避**：看到「对所有前层做注意力」这类设计，先算**显存**再看算力。判据：$O(Ld)$ vs $O(Nd)$，降幅 = 每块层数 $S$。

### 4.4 把 Quantile Balancing 当成 aux-loss-free 的调参版

**现象**：认为 QB 只是「换了个更准的均衡偏置」。
**成因**：报告明确写了它要解决的问题是**适用区**——「balancing the load of nearly $10^3$ experts **exceeds the regime in which existing auxiliary-loss-free bias updates remain well behaved**」。也就是说，在 896 个专家这个规模上，原方法的**前提**不再成立，不是参数调不好。QB 用的是**分位数**，且实现需要**直方图估计**。
**规避**：区分「超参没调好」与「方法的前提失效」。判据：当规格给出「专家数进入 $10^3$ 量级」且提到负载均衡的替换，按后者理解。

### 4.5 以为 MXFP4 QAT 只做在 SFT 阶段

**现象**：看到 QAT，按惯例认为在 SFT 阶段做一次就够，RL 阶段用全精度。
**成因**：K3 的 QAT **覆盖整个后训练阶段（SFT 与 RL）**，而且 **RL 时 rollout 与 training 共用同一套量化方案**。后者尤其关键：RL 的采样来自推理引擎，如果引擎用不同的量化，就会引入**训练-推理不一致**，把策略梯度带偏。
**规避**：读 QAT 的规格时问清「覆盖哪些阶段」与「rollout 侧用不用同一套量化」。判据：只要 RL 的 rollout 来自独立的推理引擎，就必须对齐量化方案。

### 4.6 把原生多模态当成「后接一个 ViT」

**现象**：把 MoonViT-V2 当成一个后期对齐的视觉编码器，认为它对语言主干无影响。
**成因**：K3 是 **native multimodal**——报告的原话是语言与视觉**从训练一开始就联合优化**，视觉与文本 token 交错在**同一个 next-token prediction 目标**里。这意味着 MoE 的路由要面对两种分布不同的 token（DeepSeek 在 V4.1 里为此专门做了**按模态分开的**无辅助损失均衡偏置，可以对照读）。
**规避**：看到「post-hoc alignment」与「native」两种描述要严格区分。判据：视觉 token 是否与文本 token 共享同一个预训练目标。

## 编译与运行

```sh
cd code
make run                  # 默认 python3 + numpy，依次跑 01 / 02
make 01_param_budget      # 参数与激活量核算（§2 的 MXFP4 字节账）
make 02_kda_attn_budget   # KDA / 混合注意力 / AttnRes 成本账（§1 的 3.875×、2731、1/Γ 边界）
make clean
```

本文的 **3.875×** KV 削减、**2 731** token 的 KDA/MLA 交叉点、**4.25 bit** 的 MXFP4 有效位宽、**1.518 TB** 的混合精度字节、$1/\Gamma$ 的 fp16 溢出边界、AttnRes 的 $O(Ld)\to O(Nd)$，全部来自 `code/` 下两个编号程序。它们只用 `math` 与 `numpy`，**不加载权重、不测硬件**，产出是**推导值**；其中依赖 MLA latent（576）与 head dim（128）的数字已在每处标注为**示例假设**。

## 5. 判据表

本域无 GPU、知识对象以方法与规格为主，§5 采用**判据表**形态（理由见 `160-model/README.md`）。

| 可机械判定的条件 | 结论 |
|---|---|
| 递推形式里出现 $\operatorname{Diag}(\alpha_t)$ 与 $(\mathbf{I}-\betakk^\top)$ | 是带逐通道门控的 delta-rule 线性注意力；状态 $O(1)$、缓存的 $O(L)$ 项被消掉 |
| 线性注意力用 chunkwise 形式且存在 $1/\Gamma$ | 必须检查 $\alpha$ 的下界；$\alpha < (\text{fp16 max})^{-1/C}$ 时溢出 |
| 衰减由 negative-Softplus 给出 | $\alpha$ 可趋近 0，$1/\Gamma$ **无上界**；需要额外的下界机制或对数空间计算 |
| 衰减由 scaled Sigmoid（或类似有界映射）给出 | log-decay 有下界 ⇒ $\alpha$ 有正下界 ⇒ $1/\Gamma$ 有上界；这是数值设计而非调参 |
| 设计里出现「对前层/所有层输出做注意力」 | 先算显存 $O(Ld)$；若未说明 block 化，标记为未解决 |
| 规格给出 block 大小 $S$ 或块数 $N$ | 显存 / 通信降幅 $= L/N = S$；**未给出时不得声称省了多少** |
| routed expert 在宽度 $\ell < d$ 上计算 | 是 LatentMoE 类设计；参数账用 $\ell$，通信账用「载荷宽度」 |
| 专家数进入 $10^3$ 量级且提到替换负载均衡方法 | 原方法是**前提失效**而非超参问题；按适用区理解 |
| QAT 规格未说明是否覆盖 RL | 视为只覆盖 SFT；若 RL 的 rollout 来自独立引擎，默认存在训练-推理不一致风险 |
| 规格出现微缩放量化格式（MXFP4/8） | 有效位宽 = 元素位宽 + scale 位宽 ÷ 块大小；MXFP4 = 4.25 bit |
| 官方给出 scaling 效率倍数 | 经验结果，架构账算不出来；记为引用值 |

## 6. 小结与延伸

- **一句话**：K3 沿三个方向扩信息流——序列方向换 **KDA**（固定状态）、深度方向换 **AttnRes**（对前层做注意力）、宽度方向用 **Stable LatentMoE**（896 选 16 + 三个稳定器），并用 **NoPE** 免去位置外推的改造。
- **四个可复算的数**：**3.875×**（= 93/24，KV 削减，不含假设）；**2 731** token（KDA 与 MLA 的状态读取交叉点）；**4.25 bit**（MXFP4 的有效位宽）；AttnRes block 化省 **$S$** 倍显存。
- **三处必须标注的未解决**：K3 的 MLA latent 与 head dim（示例值）、Block AttnRes 的 $N$、以及 2.5× scaling 效率（经验值）。
- **一条最有迁移价值的判据**：**衰减（或任何稀疏化参数）的可用区间由「看得见多远」与「算得准多小」共同决定**——KDA 的 $\alpha$ 是被 $1/(1-\alpha)$ 与 $(\text{fp16 max})^{-1/C}$ 两边夹出来的。

对照阅读：`160-model/deepseek/02-v4-1-flash-architecture`（压缩一份 $O(L)$ 的缓存）与 `160-model/glm/02-glm5-architecture`（在缓存里挑位置）——同一个问题的另外两种答案。

## 参考

- 【C】Kimi Team, *Kimi K3: Open Frontier Intelligence*, arXiv:[2607.24653v2](https://arxiv.org/abs/2607.24653)（**预印本，未确认同行评审**）。§2.1.1（KDA：式 (1)–(6)、chunkwise、lower-bounded decay、full-rank gate）、§2.1.2（Gated MLA 与 NoPE）、§2.2（AttnRes）、§2.3（Stable LatentMoE）、§2.4（Native Vision）、§2.5（Per-Head Muon）、§3.3–§3.4（训练配方与长上下文扩展）、§4.1.4（MXFP4 QAT）、§5.1–§5.3（基础设施）、Table 1。
- 【C】Kimi Team, *Kimi K2: Open Agentic Intelligence*, arXiv:[2507.20534v2](https://arxiv.org/abs/2507.20534)（对照世代的架构与 MuonClip）。
- 【C】*Kimi Linear: An Expressive, Efficient Attention Architecture*, arXiv:[2510.26692](https://arxiv.org/abs/2510.26692)（KDA 的原版参数化与 16-token tile；**本仓库未逐字读其正文**）。
- 【B】[kimi.ai/blog/kimi-k3](https://www.kimi.ai/blog/kimi-k3)（2.8T、KDA + AttnRes、原生视觉、1M、权重时间点、与闭源前沿的差距表述）。
- 【B】MoonshotAI 官方仓库 [Attention-Residuals](https://github.com/MoonshotAI/Attention-Residuals)。
- 相关域：[130-model-optimization/quantization](../../130-model-optimization/quantization/)（MX 格式与 E8M0）、[100-parallelism/expert-parallel](../../100-parallelism/expert-parallel/)（EP 负载不均）、[80-model-architectures/positional-encoding](../../80-model-architectures/positional-encoding/)（RoPE 与 NoPE 的取舍）。

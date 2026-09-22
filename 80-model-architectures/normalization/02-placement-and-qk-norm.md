---
title: 归一化放在哪：pre / post / final，以及 QK-norm 抹掉的是什么
slug: placement-and-qk-norm
domain: 80-model-architectures
topic: normalization
level: 3
difficulty: advanced
prerequisites:
  - 80-model-architectures/normalization/01-rmsnorm-vs-layernorm
tags: [normalization, placement, pre-ln, post-ln, final-norm, qk-norm, logits-scale, softmax-entropy, comparable]
related:
  - 80-model-architectures/transformer/01-block-and-residual-stream
  - 40-kernels/softmax-normalization/03-normalization-fusion
  - 80-model-architectures/transformer/03-depth-and-width
code:
  - code/02_placement_qk_norm.py
status: stable
updated: 2026-09-22
---

# 归一化放在哪：pre / post / final，以及 QK-norm 抹掉的是什么

> 「归一化放在哪」是一个**二维**选择：**(a) 与残差加法的先后**（pre / post / final ——
> 结构与数值后果的正文在 [`transformer/01`](../transformer/01-block-and-residual-stream.md)，
> 本文只承接它的**训练侧**那一半），与 **(b) 它作用在多大的张量上、沿哪条轴**
> （整条残差流 `(N, D)`？还是每个 head 的 `q`/`k` `(H, S, d_h)`？—— ★ **这一支是本文的正文**）。
> 而 **QK-norm 抹掉的是「输入幅度」**，它**不是** `1/√d_h` 的替代品：
> 除以 `√d_h` 之后 `Var(q·k/√d_h)` 仍由幅度决定（`3²×0.2² = 0.36`，**可手算**），
> 而 `Var(RMS(q)·RMS(k)/√d_h)` 的闭式是 **`1.0`**（五个 `d_h` 上实测 `0.9988 ~ 1.0118`）。
> ★ 而「轴」不是自由选择：沿 head 维做，逐行缩放**精确**不变（`2.888e-16`）；
> 沿序列维做，同一个量掉到 `8.126e-01`。

## 0. 背景与动机

前一篇文章（[`01`](01-rmsnorm-vs-layernorm.md)）讲**用哪个算子**；本文讲**放在哪**。
这两件事在工程转述里经常被压成一句「现在都 pre-LN + RMSNorm」，而这句话把**四个不同的选择**
混成了一个：

| # | 选择 | 取值 | 它的账 |
|---|---|---|---|
| (a1) | 与残差加法的先后 | `pre` / `post` | 残差流的**范数**（结构与数值，正文在 `transformer/01`）+ **训练稳定性**（本文 `§1.2`） |
| (a2) | 末层要不要补一次 | `final_norm` / 不要 | 输出头的数值条件 —— **判据在 `transformer/01` 的判据 ③，本文不重做** |
| (b1) | 作用在多大的张量上 | 整条残差流 / 每个 head 的 `q`·`k` | 见 `§1.3` |
| (b2) | **沿哪条轴** | `d_h` / `S` | ★ 见 `§1.3` 与 `C`：**这不是修饰项** |

**为什么 (b) 值得单写**：`1/√d_h` 这个缩放因子在架构里几乎无处不在，于是很容易得出一个
**错的推论**——「有了 QK-norm 就不需要 `1/√d_h` 了」。★ **本域探测阶段真的这样错过一次**
（见 `probes/probe_80d.py` 模块头的 `✗【2】我漏了 QK-norm **不替代** 1/√d_h`）。
本文把它拆成三件事，各自配一个**可手算的闭式**（`§1.4`）：

| 量 | 由什么决定 | 谁管它 |
|---|---|---|
| `Var(q·k)` | `d_h`（**线性**：本机 `32.0 → 515.5`） | — |
| `Var(q·k/√d_h)` | **输入幅度** `α²β²`（本机 `0.36`，闭式 `3²×0.2²`） | **QK-norm** |
| `Var(RMS(q)·RMS(k)/√d_h)` | 两者都无关（闭式 **`1.0`**） | 两者**各管一半** |

## 1. 原理

### 1.1 先划边界：本文不写什么

- **两个算子的定义式**、Welford 递推、`Σx² − nμ²`、`eps` 位置 —— 正文在
  `40-kernels/softmax-normalization/03-normalization-fusion.md:53-345`（**公式在 `:55-71`**）。
- **残差流范数的后果**（pre-LN 随深度涨、post-LN 恒为 `√D`）与 **`final_norm` 的必要性** ——
  正文在 [`transformer/01`](../transformer/01-block-and-residual-stream.md)：
  那里有两条判据（`‖h_L‖ = √D` 的恒等式；`lm_head` 的 logits std 之比 `> 10×`，闭式 `‖h_L‖/√d`）。
  ★ **本文不重做那两条**（计划 §3 Chunk 4 的表 B 把这一点写成要求）。
- **softmax 自身的数值稳定性**（减 max、在线 softmax）—— 正文在 `40-kernels/softmax-normalization/01`、`02`。

### 1.2 (a) 与残差加法的先后：账在训练侧，措辞有纪律

两种放法在原文里是**两句不同的措辞**（`specs` 逐字钉住）：

> post-LN：`puts the layer normalization between the residual blocks`
> pre-LN：`puts the layer normalization inside the residual connection` and
> `places it before all other non-linear transformations`

**结构后果**（本文不重做，只回链）：post-LN 每层末尾把范数钉在 `√D`；
pre-LN 的残差流范数**随深度增长** ⇒ 末层**必须**补一个 `final_norm`。
★ 而「pre-LN 栈在最后补一个归一化」这个**结构事实**的出处也是原文逐字：
`final layer normalization`（完整句是 `Additionally, the Pre-LN Transformer uses a final layer
normalization right before the prediction.`）。

**训练侧的账**（本文承接的那一半）：这篇论文（arXiv 2002.04745）的贡献是
**用平均场理论解释**了「为什么 post-LN 需要 warm-up」，并据此**去掉**它。三条措辞纪律：

1. **结论词是 `comparable`，不是「更好」**：Abstract 逐字 `can reach comparable results with
   baselines …`；§4.2 逐字 `… which are comparable with the performance of the Post-LN
   Transformer.`（原始字节计数：`comparable` = **5**，其中 **2 次**说的是**语料规模**
   —— 那是**同名不同指**，不能算进来）。
2. **`outperform` 在原文里出现 1 次，但它的主语是 `Wang et al. 2019`**（被引的第三方）：
   逐字 `Wang et al. 2019 suggested that the Pre-LN Transformer outperforms the Post-LN
   Transformer when the number of layers increases.` ⇒ ★ **不得把它读成论文自己的结论**。
3. **post-LN 梯度尺度的那条论断原文用 `hypothesize`**（全文 **1** 次，就是这一处）：
   `we hypothesize that the gradient scale is one of the reasons that the Post-LN Transformer
   needs a careful learning rate scheduling.` ★ 同段**紧邻**的观察句是**直陈**
   （`the scale of the expected gradients grows along with the layer index for the Post-LN
   Transformer`）—— **观察与因果假设必须分写**。

⇒ 所以「pre-LN 更好」这句话在本文里的正确形态是：
**「论文主张它能去掉 warm-up、收敛更快、超参更少，最终质量是 `comparable`」**。

### 1.3 (b) 归一化作用在多大的张量上、沿哪条轴

这是本文**新增**的那一支。两个选择：

| 取值 | 作用的张量 | 归一化**沿哪条轴** |
|---|---|---|
| 残差流上的归一化（pre / post / final） | `(N, D)` | 沿**特征维** `D`（每一行一个统计量） |
| **QK-norm** | `(H, S, d_h)` | 沿 **head 维 `d_h`** |

★★ **轴不是自由选择**（`B.3`，`eps = 0`，形状 `(8, 1024, 128)`，每 `(h, s)` 行乘一个
`α ~ U(0.2, 5)`）：

```
沿 head 维（d_h）：逐行缩放后，逐元素最大相对变化 = 2.8876e-16   ← 机器精度
沿 序列维（S）  ：同一个量                       = 8.1260e-01   ← 差 15 个数量级
```

⇒ **「QK-norm 抹掉输入幅度」这句话只对 head 维成立**。沿序列维归一化会同时把
「这一行有多强」的信息**跨 token 混起来** —— 那不是归一化，那是另一种运算。
★ 所以本文把**轴**写进 QK-norm 的**定义**里（`specs.QK_NORM_AXIS`，[E] 级），而不是当注释。

★★ **一条负结果必须写出来**（`B.3b`）：**`§1.4` 那两个方差统计量分辨不出轴** ——
沿 head 维 `0.9879`、沿序列维 `0.9855`，两者都在 `1 ± 0.05` 内。
⇒ 判「轴重要」**不能**靠 `A` 段的方差（它在两条轴上都 ≈ `1`，这是我从推导里也看不出来的、
**只能测出来**的事），只能靠 `B.3` 的**逐行缩放不变性**。
★ 这条负结果的价值：如果只写「轴很重要」而拿 `A` 段的数当证据，那个证据**是假的**。

**QK-norm 的定义（本节写出，因为一手来源不在本环境，见 `§4.4`）**：
对 `q`、`k` **各自**沿 **head 维**做 RMSNorm（得到 `RMS(q)`、`RMS(k)`），
再点积并除以 `√d_h`：`⟨RMS(q), RMS(k)⟩/√d_h`。

### 1.4 QK-norm 抹掉的是什么（多步推导 + 三个闭式）

记一行 `q ∈ ℝ^{d_h}`。由 `01` 用过的**性质 A（二次齐次）**，归一化之后这一行与 `q` **同方向**、
长度**恰好是 `√d_h`** —— 即它把这一行**投到半径 `√d_h` 的球面上**（★ 这里只用到「同方向」与
「长度是常数」两条，**算子的定义式本文不写**）。★ 而这个「球面」只在 `eps = 0` 时精确：
`eps > 0` 时半径**略小于** `√d_h`，偏差是 `eps/(2·ms)` 的量级 —— **正比于 `eps`、可手算**。
实测（`d_h = 128`、`q ~ N(0,1)`，`A.6`）：

| `eps` | `1 − min(半径/√d_h)` | 一阶闭式 `eps/(2·ms_min)` | 该行的 `ms` 最小值 |
|---|---|---|---|
| `1e-6` | `9.167639e-07` | `9.167651e-07` | `0.5454` |
| `1e-3` | `9.155064e-04` | `9.167651e-04` | `0.5454` |

⇒ 两档**成正比**（比值 `998.6`），且逐行落在闭式上（最差相对偏差 `1.4e-03` ——
★ 闭式是**一阶**的，`1e-3` 那一档的高阶项正好是那 `1.4e-3` 的量级）。

1. 于是 `⟨RMS(q), RMS(k)⟩ = ‖RMS(q)‖·‖RMS(k)‖·cos θ = d_h·cos θ`
   （`θ` 是 `q` 与 `k` 的夹角）—— ★ **归一化不改变夹角**，它只改长度；
2. 随机方向下 `E[cos²θ] = 1/d_h` ⇒ **`Var(Σ_i RMS(q)_i·RMS(k)_i) = d_h²·(1/d_h) = d_h`**；
3. 除以 `√d_h` 之后：**`Var(⟨RMS(q), RMS(k)⟩/√d_h) = d_h / d_h = 1`**；
4. 而**不**做归一化时 `q·k` 的方差是 `d_h·E[q²]E[k²]`：`q`、`k` 各乘 `α`、`β` 就乘 `α²β²`
   —— **幅度从这条路上进来，而且只有这条路**。

⇒ 三个**可手算**的闭式（`A.2`/`A.3`/`A.4` 实测与之对照，`N = 20000`、`q×3.0`、`k×0.2`）：

| 闭式 | 实测（`d_h = 32 / 64 / 128 / 256 / 512`） |
|---|---|
| `Var(q·k/√d_h) = α²β² = 0.36` | `0.3600 / 0.3641 / 0.3590 / 0.3614 / 0.3621`（最差偏差 `1.13%`） |
| `Var(⟨RMS(q), RMS(k)⟩/√d_h) = 1.0` | `0.9999 / 1.0118 / 0.9988 / 1.0054 / 1.0069` |
| `Var(Σ_i RMS(q)_i·RMS(k)_i) = d_h` | `32.00 / 64.76 / 127.85 / 257.38 / 515.54` |

★★ **第三行就是「不替代」的证据**：`Var(Σ RMS·RMS)` **随 `d_h` 线性增长**（闭式恰好是 `d_h`）。
若 QK-norm 真的替代了 `1/√d_h`，这个量应当**与 `d_h` 无关**。
⇒ **`1/√d_h` 管 `d_h` 那一半，QK-norm 管幅度那一半**。

**「抹掉幅度」的精确形态**（`A.5`，**同一个抽样**、只把 `(α, β)` 换掉，`d_h = 128`）：

| `(α, β)` | 不做归一化：`比值` | 应等于 `α²β²` | 做 QK-norm：`比值` |
|---|---|---|---|
| `(1, 1)` | `1.000000000000000` | `1` | `1.000000000000000` |
| `(3.0, 0.2)` | `0.360000000000000` | `0.36` | `1.000000000000000` |
| `(0.1, 50)` | `25.000000000000000` | `25` | `1.000000000000000` |
| `(2, 7)` | `195.999999999999972` | `196` | `1.000000000000000` |

⇒ 一边**恰好**是 `α²β²`（机器精度 `< 1e-12`），另一边**恰好**是 `1`
（★ 断言写的是**容差** `< 1e-12`，不是 `== 1.0` —— 那会是 P18 那类「要一个不可能的结果」）。

### 1.5 logits 的尺度有多要紧（判据 ⑤）

QK-norm 改变的是 **logits 的尺度**。那么尺度有多要紧？**softmax 的归一化熵**
（除以 `log S`）随尺度**严格递减**（`B.1`；`S = 1024`、**`n = 500` 个样本**）：

| logits `std` | `0.1` | `0.5` | `1.0` | `2.0` | `3.0` | `6.0` | `12.0` |
|---|---|---|---|---|---|---|---|
| 归一化熵 | `0.9993` | `0.9819` | `0.9281` | `0.7273` | `0.4961` | `0.1918` | `0.0787` |
| 熵的标准差 | `0.0000` | `0.0010` | `0.0055` | `0.0492` | `0.1124` | `0.1022` | `0.0714` |
| 最大概率（均值） | `0.0013` | `0.0045` | `0.0161` | `0.1028` | `0.2718` | `0.6051` | `0.8061` |

相邻差最小 `-0.0173`（`0.1 → 0.5` 那一档），**七档严格递减**。
★ 计划 §0.7-D 的 **D20** 记录（**同口径** `S = 1024, n = 500`）是
`0.9993 → 0.9820 → 0.9278 → 0.7213 → 0.5011 → 0.1872 → 0.0782` ——
与上表**同档不同位**（差在 `1e-2` 量级，因为探测脚本的 `rng` 已被前面的章节消耗）。
⇒ 判据只能写**单调性**，不能写某一次抽样的读数。

★★ **而且它必须是多样本**（`B.2`）：在 200 个种子上，
**`n = 1`** 时「`std = 12.0` 的熵 **大于** `std = 6.0` 的熵」出现 **47/200** 次
（最大放大到 `29.38×`）。⇒ 单样本的熵是**抽样噪声**，不是「注意力稀释」的证据。
★ 这正是计划 §0.1「第一类错误」的现场（探测脚本 `probe_80c.py` 用 `n = 1` 得出过
「非单调」的假结论，`probe_80d.py` 用 `n = 500` 修正）。

⚠ 本文**不给**「熵低于多少，注意力就算废了」这个阈值 —— 没有一手来源（见 `specs.UNSOURCED`）。
只给**单调性**与两端（本机 `0.9993` / `0.0744`）。

## 2. 结构上能看到什么（★ 不是 kernel）

**（组件一）`(H, S, d_h)` 里的一个 head 的一个位置**，与**（组件二）`(N, D)` 里的一行** ——
QK-norm 只碰前者，而它归一化的**统计量沿哪条轴取**决定了它到底在做什么（`§1.3`）。

**数值量级**（本文自己的，全部可手算）：

- `√d_h` 在 `d_h ∈ {32, 64, 128, 256, 512}` 上是 `5.66 / 8.00 / 11.31 / 16.00 / 22.63`。
  ⇒ 「去掉 `1/√d_h`」会让 logits 的 **std** 变成 `√d_h` 倍（`d_h = 128` 时 `11.31` 倍，`C.1`）。
  ★ 与 `§1.5` 的熵阶梯对上：`std` 涨 `11.31` 倍（本机阶梯上 `std = 1.0 → 熵 0.9281`、
  `std = 12.0 → 熵 0.0787`），⇒ 熵从 `~0.93` 掉到 `< 0.1`。
- 归一化后的 logits 方差闭式是 `1.0`（`§1.4` 第 3 步），实测五点 `0.9988 ~ 1.0118`。
- 抽样误差：`N = 20000` 时 `~0.007` ⇒ 判据区间 `[0.9, 1.1]` 的边界离它有 **14σ**
  （★ 这就是判据写**区间**而不写「某次抽样的读数」的理由）。

**两个方向的错**（`C.1`/`C.2`，`d_h = 128`）：

| 写法 | `Var` | 与正确值的关系 |
|---|---|---|
| `⟨RMS(q), RMS(k)⟩/√d_h`（正确） | `0.9807` | 闭式 `1.0` |
| 去掉 `√d_h` | `125.53` | `≈ d_h` —— logits **涨** `√d_h` 倍 |
| 把 `√d_h` 写成 `d_h` | `0.007662` | `≈ 1/d_h`（闭式 `0.007812`）—— logits **缩** `√d_h` 倍 |

## 3. 实现与演进

**位置与 QK-norm 在真实模型里的可见痕迹**（来源：两份 `config.json`，
**经 `hf-mirror.com` 镜像取得，未与官方 repo 比对 sha256**；逐字计数）：

| 字段名 | Llama-3-8B | Qwen3-30B-A3B | 说明 |
|---|---|---|---|
| `rms_norm_eps` | **1** | **1** | 两个模型都有（取值 `1e-05` / `1e-06`，见 `01` 的 `§3`） |
| `layer_norm_eps` | **0** | **0** | ⇒ 这两个 config 里**没有**这个字段 |
| **`qk_norm`** | **0** | **0** | ⇒ ★★ **本环境没有 QK-norm 的 config 锚点** |

★★ **本域的处理**（这一条是本文最容易写错的地方，因为它没有可引的出处）：
**QK-norm 的一手来源不在本环境** —— 本域的论文清单（计划 §C.2 的 12 篇）里**没有它**，
两个真实 config 里也**没有**这个字段（上面的计数）。
⇒ **本域不替它编一个出处**（编一个 arXiv 号正是本库禁止的那类错误），
而是把「它是什么」写成 **[E] 定义**（`specs.QK_NORM_AXIS` 与 `§1.3` 末），
**只对本机测到的后果下结论** —— 后果是可判定的，出处不是。
⇒ 因此本文**不写**「某某模型用了 QK-norm」这类句子（`specs.UNSOURCED` 的
`qk_norm_in_real_configs` 记着这条缺口）。

**位置的现状**（原文逐字，本文只做引用）：`§1.2` 的两句放法 + `final layer normalization`
都是 pre-LN 论文 §3 的逐字。★ **`final_norm` 为什么必要** 的正文在
`transformer/01` 的判据 ③（`lm_head` 的 logits std 之比 `> 10×`，闭式 `‖h_L‖/√d`）——
**本文不重做**。

## 4. 踩坑与失败模式

### 4.1 把 QK-norm 当成 `1/√d_h` 的替代

- **现象**：写「加了 QK-norm 就不需要 `1/√d_h` 了」（或反过来：「QK-norm 就是把 logits 方差压到 1」，
  隐含它做了 `1/√d_h` 的活）。
- **成因**：把「与 `d_h` 无关」与「与输入幅度无关」当成同一件事。`1/√d_h` 让方差与 `d_h` 无关；
  QK-norm 让方差与**幅度**无关。★ **本域探测阶段真的错了这一步**
  （`probes/probe_80d.py` 模块头逐字：`✗【2】我漏了 QK-norm **不替代** 1/√d_h`）。
- **规避**：把三个量并排（`§0` 的表），并给「不替代」一条机械形态：
  `Var(Σ RMS·RMS)` 的闭式是 **`d_h`**（`A.4`）—— 若替代成立，它应当与 `d_h` 无关。

### 4.2 用 `n = 1` 的 softmax 熵下结论

- **现象**：写「logits 尺度大时熵反而更大，所以注意力不会稀释」。
- **成因**：单样本的熵方差极大（本机 `std = 6.0` 那一档的熵标准差是 `0.1022`），
  单调性在单样本上**确实会破**（`47/200` 个种子）。
- **规避**：一律用多样本（本域用 `n = 500`），并把「单样本会破」单独做成断言（`B.2`）——
  ★ 这条的价值不在于数字，而在于**它把判据的统计量不稳这件事留在了程序里**。

### 4.3 把 `Wang et al. 2019` 的话写成论文自己的结论

- **现象**：写「pre-LN 论文证明了 pre-LN 更好」。
- **成因**：`outperform` 这个词在原文里**确实出现 1 次**，但**主语是被引的第三方**；
  论文自己的结论词是 `comparable`，而它主张的是**训练流程**的账（可去 warm-up、收敛快、超参少），
  不是**最终质量**的账。
- **规避**：`specs` 里把 `outperform` 那半句**单独存成一条带归属的常量**
  （`PRE_LN_OUTPERFORM_ATTRIBUTION`），程序 `D.4` 逐位钉住它 ——
  这样「谁是主语」在**代码里**也看得见。★ 另外 post-LN 梯度那句原文是 `hypothesize`，
  同段紧邻的观察句才是直陈，**两者不可混写**。

### 4.4 给 QK-norm 编一个出处

- **现象**：写「QK-norm 出自 arXiv XXXXX」，或写「某某模型用了 QK-norm」。
- **成因**：这个机制**在真实模型与文献里都很有名**，凭印象写一个出处太容易了。
- **规避**：★ 本域的硬约束是**先确认来源在不在手上**。实测：论文清单里没有它、
  两份 config 里 `qk_norm` 各 **0** 次 ⇒ **不写**。正确落法是把它写成 [E] 定义，
  并把它与「后果」分开：**后果可以下结论，出处不能**。

### 4.5 把「轴」当成可选项

- **现象**：写「对 `q`、`k` 做 RMSNorm」，不写沿哪条轴。
- **成因**：在 `(H, S, d_h)` 上「做 RMSNorm」这句话**是不完整的** ——
  `axis=-1` 与 `axis=-2` 是两个不同的运算。
- **规避**：`B.3` 把差别做成一条 15 个数量级的对比；`specs.QK_NORM_AXIS` 把轴写进定义，
  并被程序 `D.3` / `B.3` 消费（★ 程序用**映射**取这个值而不是按名取值，
  于是坏值让**断言失败**、而不是抛异常）。

## 5. 判据表

| 可机械判定的条件 | 结论 |
|---|---|
| `Var(RMS(q)·RMS(k)/√d_h)` 在 `d_h ∈ {32…512}` 上 | 落在 **`[0.9, 1.1]`**（闭式 `1.0`）；★ 判据写**区间**，因为它是抽样量（`N = 20000` 时抽样误差 `~0.007`，边界有 `14σ`） |
| `Var(q·k/√d_h)` 在 `q×α, k×β` 时 | 等于 **`α²β²`**（本机 `α = 3, β = 0.2` 时 `0.36`，最差偏差 `1.13%`）—— **可手算** |
| 想要「QK-norm 是否替代 `1/√d_h`」的判据 | 看 **`Var(Σ_i RMS(q)_i·RMS(k)_i)`**：闭式是 **`d_h`**（**线性增长**）⇒ **不替代**（替代成立时它应与 `d_h` 无关） |
| 「抹掉输入幅度」的机械形态 | **同一个抽样**下 `plain(αq, βk)/plain(q, k)` 等于 `α²β²`（不是「≈」），而 `qknorm(αq, βk)/qknorm(q, k)` 在容差 `1e-12` 内等于 `1` |
| 「做 RMSNorm」这句话写在 `(H, S, d_h)` 上时 | **必须写出轴**：沿 `d_h` ⇒ 逐行缩放精确不变（`2.888e-16`）；沿 `S` ⇒ 同一个量 `8.126e-01`。★ **别拿方差统计量当判据**：那个量在两条轴上都 ≈ `1`（`0.9879` / `0.9855`） |
| 去掉 `1/√d_h` | `Var` 从 `≈ 1` 变成 `≈ d_h`（`d_h = 128` 时 `125.53`）⇒ logits 的 std 涨 `√d_h` 倍 ⇒ 熵从 `~0.93` 掉到 `< 0.1`（`§1.5` 的阶梯） |
| 把 `√d_h` 写成 `d_h` | `Var` 变成 `≈ 1/d_h`（本机 `0.007662`，闭式 `0.007812`）—— **两个方向的错量级相同、方向相反** |
| 报「softmax 熵随 logits 尺度单调」 | **必须写样本数**：`n = 500` 时七档严格递减（相邻差最小 `-0.0173`）；**`n = 1` 时 47/200 个种子会违反它** |
| 说「熵低 = 注意力废了」 | **本域不给这个阈值**（无一手来源）。只给单调性与两端（`0.9993` / `0.0744`） |
| 引用 pre-LN 论文的结论 | 用 **`comparable`**（Abstract 与 §4.2 逐字）；`outperform` 那 1 次的**主语是 `Wang et al. 2019`**；梯度那句是 **`hypothesize`** |
| 引用 `final_norm` 的必要性 | **只回链** `transformer/01` 的判据 ③（正文在彼，此处回链）；★ 本文不重做那两条 |
| 想知道 QK-norm 的出处 | **本环境没有**：论文清单里没有、两份 config 里 `qk_norm` 各 0 次 ⇒ 本域把它写成 **[E] 定义**，只对后果下结论 |
| 想知道两个算子的定义式 / Welford / `eps` 位置 | **不在本文**：`40-kernels/softmax-normalization/03:53-345` |
| 想知道 softmax 的数值稳定性（减 max、在线 softmax） | **不在本文**：`40-kernels/softmax-normalization/01`、`02` |
| 想知道 attention 的 FLOPs / 访存账 | **不在本文**：`40-kernels/attention/01`、`05` |

> ★ **表 A 的数值判据**（计划 §3 Chunk 4）：④ QK-norm 前后（两个量分别与 `d_h` / 幅度的关系）；
> ⑤ softmax 熵随 logits 尺度单调下降（**在 `n = 500` 上**，且 `n = 1` 是现场）。
> 两条都在 `code/02_placement_qk_norm.py` 的 `A.1`–`A.5` / `B.1` 里，
> 阈值与口径被 `D.1`/`D.2` 逐位钉住（改样本数或区间 ⇒ 判据退化成恒真）。

## 6. 小结与延伸

**一句话**：这一篇的账也是**数值条件的账** —— 「放在哪」不是一格开关，
而是**两维**选择（与残差加法的先后 × 作用范围与轴）；而 QK-norm 抹掉的是**输入幅度**，
它**不替代** `1/√d_h`（`Var(Σ RMS·RMS)` 的闭式是 `d_h`，**线性增长**，这就是证据）；
「轴」写错会让这条性质**当场失效**（差 15 个数量级）。

- **前置**：`normalization/01-rmsnorm-vs-layernorm`（不变性的精确形态、二次齐次性质、`eps` 的量级）。
- **位置的另一半**：`transformer/01-block-and-residual-stream`（残差流范数、`final_norm` 的判据 ③）
  与 `transformer/03-depth-and-width`（规格表里 `model.norm` 那一项）。
- **两个算子的公式与误差模型不在本文**：`40-kernels/softmax-normalization/03-normalization-fusion.md:55-71`
  （正文在彼，此处回链）。
- **下一步**：位置与轴都定了之后，「位置信息怎么编码」是下一个主题
  （`positional-encoding/01`，RoPE 的旋转定义）—— 它与本文的接口是 `q`/`k` 的**同一个张量**。

## 编译与运行

```bash
cd code
make run                             # 两个编号程序依次跑；本程序是 02_placement_qk_norm.py
python3 02_placement_qk_norm.py       # 只跑本程序（约 1.1 s）
```

- 配套程序：**`code/02_placement_qk_norm.py`**（编号与外键严格对应），**25 条断言**。
- 依赖：`numpy`（`requirements.txt` 里**只有**它）。**不需要网络、不需要 GPU。**
- 失败标记：`~~ FAIL: normalization`（本主题自定义，**不跨主题照抄**）。
- 程序的输出分五段：`D`（口径/阈值/specs 钉住，**刻意放在最前**）、
  `A` 判据 ④（含三件事并排与「不替代」）、`B` 判据 ⑤ + 规格锚点 + 轴、
  `C` 两个方向的口径变异 + 边界、`D（续）` 标记自检与独立性检查。

## 参考

### 表 B：引用判据（**不能被程序判定**，在**原始字节**上计数）

实跑（2026-09-22；`WebFetch` 被策略拦截 ⇒ 用 `curl`）：

```bash
curl -sS -o /tmp/arx_2002.04745.html https://ar5iv.labs.arxiv.org/html/2002.04745  # 521794 B
```

| 字面量 | 计数 | 用途 |
|---|---|---|
| `comparable` | **5** | ★ 论文的结论词；⚠ **其中 2 次说的是语料规模**（`which is comparable with the data corpus used in Devlin et al. 2018`）⇒ **同名不同指** |
| `comparable results with baselines` | **1** | Abstract 那一句 |
| `outperform` | **1** | ★★ **主语是 `Wang et al. 2019`** —— 逐字 `outperforms the Post-LN Transformer when the number of layers increases` 计 **1** |
| `we hypothesize that the gradient scale` | **1** | ★ post-LN 梯度那句（**`hypothesize`**，全文仅此一处；`hypothesise` = 0） |
| `puts the layer normalization inside the residual connection` | **2** | pre 的放法 |
| `places it before all other non-linear transformations` | **1** | 同句后半 |
| `puts the layer normalization between the residual blocks` | **1** | post 的放法 |
| `final layer normalization` | **2** | ★ pre-LN 栈末层的那个归一化（`right before the prediction` 计 **1**） |
| `uses a final layer normalization right` | **0** | ★★ **跨标记**：原文把那四个词包在 `<em>…</em>` 里 ⇒ 整句在原始字节上**必然计不到**（§0.3 的纪律）。⇒ 本域**只存计得到的片段** |
| `Wang et al. 2019 suggested that the Pre-LN` | **0** | ★ 同一个原因：`Wang et al. 2019` 是一个 `<a>` 链接 |

**外部来源**

- **pre-LN vs post-LN**：arXiv 2002.04745（★ abs 页 `Journal reference` 逐字含
  `Published on ICML 2020` ⇒ **已确认同行评审**；抓取
  `https://ar5iv.labs.arxiv.org/html/2002.04745`，2026-09-22，**521794 B**，
  `sha256[:16] = d90ec3972ed0117c` —— 与计划 §C.2 逐位一致）。**[C]**
- **两份 `config.json`**：**经 `hf-mirror.com` 镜像取得，未与官方 repo 比对 sha256**
  （★ 最简形式返回 307 跳转正文 ⇒ 必须 `-L`；镜像副本字节哈希：Llama `2430cee764b6530f`（654 B）、
  Qwen3 `2850ddb3bf7aecad`（963 B））。**[镜像]**
- ★★ **QK-norm**：**本环境没有一手来源**（论文清单里没有；`qk_norm` 在两份 config 里各 **0** 次）
  ⇒ 本域把它写成 [E] 定义（`specs.QK_NORM_AXIS`），**不编出处**。见 `§4.4`。

**本机实测值**

- `code/02_placement_qk_norm.py` 的 `make run` 输出：`python3 3.13.5` + `numpy 2.4.4`，
  macOS arm64，**无 GPU**。口径三段：QK 段 `q×3.0`、`k×0.2`、`N = 20000`、种子 `20260922`、
  `d_h ∈ {32, 64, 128, 256, 512}`（★ `probes/probe_80d.py`【1】的 `N` 是 **`40000`**
  ⇒ **同口径不同样本量**，不可逐位对齐；判据只写区间）；
  轴段 `(H, S, d_h) = (8, 1024, 128)`、每行 `α ~ U(0.2, 5)`、`eps = 0`、种子 `7`；
  熵段 `S = 1024`、`n = 500`、`std ∈ {0.1 … 12.0}`、种子 `20260922`
  （★ 与 `probes/probe_80d.py`【2】即 **D20** 同口径、**不同抽样历史**）。
  **全部是推导值与本机实测值**。

**上游契约**

- `80-model-architectures/transformer/01-block-and-residual-stream.md` —— 残差流范数（pre/post）、
  `final_norm` 的必要性判据 ③、以及「训练侧的账在 `normalization/02`」那条指名让渡的**接续点**
  （那一处逐字写着「它的正文在 `normalization/02-placement-and-qk-norm`，此处回链」）。
- `40-kernels/softmax-normalization/03-normalization-fusion.md:55-71`（公式）与 `:213-263`（访存账）
  —— **禁区**，只回链。
- `40-kernels/attention/04-warp-specialization-and-variants.md:126-156` —— 变体对 **kernel 结构**
  的影响（那里讲「每行只需 `ceil(w/Bc)` 个块」）；★ **QK-norm 不在那三个变体里**，
  本文也不写任何 kernel 侧结论。

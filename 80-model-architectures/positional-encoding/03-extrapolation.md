---
title: 外推方法：PI / NTK-aware / NTK-by-parts / YaRN / ALiBi
slug: extrapolation
domain: 80-model-architectures
topic: positional-encoding
level: 3
difficulty: advanced
prerequisites:
  - 80-model-architectures/positional-encoding/02-spectrum-and-aliasing
tags: [rope, context-extension, position-interpolation, ntk-aware, ntk-by-parts, yarn, alibi, temperature, recency-bias, out-of-bound]
related:
  - 80-model-architectures/attention-variants/03-window-and-hybrid
  - 40-kernels/attention/04-warp-specialization-and-variants
code:
  - code/03_extrapolation.py
status: stable
updated: 2026-09-22
---

# 外推方法：PI / NTK-aware / NTK-by-parts / YaRN / ALiBi

> `02` 给了定量框架（谱、混叠、分辨窗口）。本文**用它**解释五种工程方法。
> ★ 一句话结论：**前四种是同一个一参数族的四个成员** ——
> 统一量是**相位比** `Φ_ratio_i`（扩展窗口下的最大相位 ÷ 训练窗口下的最大相位），
> `Φ_ratio_i ≤ 1` 才叫「没有外推」。差别只在**每维的频率缩放挑多少**：
> PI 全维缩 `1/s`（贴住上限，但把不该缩的高频也缩了）；
> NTK-aware 按**维号**缩（`i = 1` 那一维**一点没缩** ⇒ 它就是被推到 `out-of-bound` 的那一维）；
> NTK-by-parts 按 **`r = L/λ`** 缩（**恒 `≤ 1`**，从不超过训练范围）。
> ★★ 本文显式写出 `02` 与 `03` 之间**唯一的那座桥**（`§1.2`）：
> **`r_i = L/λ_i` 与 `02` 的 `turns_i(L)` 是同一个量**。
> ★ **ALiBi 是另一条路**（不改位置编码，改 logits 加一项偏置），本文给它完整的定义、
> 那条**脚注**、以及那条脚注**可判定的后果**（`§3`）。
> ★ 本文**比较结构，不比较效果** —— 本域**无 GPU、不训练模型**。

## 0. 背景与动机

`02` 的结论是：`θ_i` 这条谱**两端各有一种失败**，而且**两者不可能同时避免**。
于是「把上下文窗口从 `L` 扩到 `L' = s·L`」这件事就变成了一个**分配问题**：

> **每维的频率该缩多少？**

- **缩得不够**（或根本不缩）⇒ 那一维的**最大相位超过训练时见过的** ⇒ **真外推**；
- **缩得太多**（全维都按 `s` 缩）⇒ 训练时见过的**高频信息被丢掉**。

★ 五种方法就是对这个分配问题的五种回答。★ 而它们常被并列成「五个竞品」，
让读者以为它们在做五件不同的事 —— 本文要说明**前四种是同一个族的四个成员**
（`§1.1` 的表格），并且**它们的差别可以逐位算出来**（`§2`）。

★ **本文的分工**（计划 §3 写死）：**`02` 给定量框架、`03` 用它解释工程方法**。
本文**不重讲频率谱**、不重新推 `θ_i` 的单调性、不重算分辨窗口 —— 那些只**引用**。
★ 本文也**不重讲 kernel 后果**（滑窗改循环范围、ALiBi/softcap 改 `exp` 的输入）——
正文在 `40-kernels/attention/04-warp-specialization-and-variants.md:126-156`（**正文在彼，此处回链**）。

## 1. 原理

### 1.1 统一量：相位比 `Φ_ratio`

★ 本域写出的定义（**[E]**，不是抄来的）：

```text
Φ_ratio_i  :=  扩展窗口下第 i 维的最大相位 / 训练窗口下它的最大相位
            =  (最大位置 · θ_i^eff) / (L · θ_i)
```

它的意义只有一句：**`Φ_ratio_i ≤ 1` 才叫「没有外推」**（最大相位不超过训练时见过的）。
⇒ 四种方法在这**一条**量上的差别是**可逐位比较的**（`d = 128`、`h = 64`、`s = 16`）：

| 方法 | 每维的**频率缩放** `θ^eff/θ` | `Φ_ratio_i` | 它的性质 |
|---|---|---|---|
| **PI** | `1/s`（**全维一样**） | **恰好 `1`** | 贴住上限 —— 但把**不该缩**的高频也缩了 |
| **NTK-aware** | `s^{−α_i}`，`α_i = (i−1)/(h−1)` | `s^{1−α_i}` ∈ `[1, s]` | ★★ `i = 1` 那一维 `Φ_ratio = s` ⇒ **被外推** |
| **不做任何事** | `1` | `s` | 全维都外推 `s` 倍 |
| **NTK-by-parts** | `(1−γ_i)·(1/s) + γ_i` | `(1−γ_i)/s + γ_i` ∈ `[1/s, 1]` | ★★ **恒 `≤ 1`** ⇒ **从不超过训练范围** |

实测（`base = 10000`、`L = 8192`、`s = 16`）：

| 方法 | 频率缩放的取值 | `Φ_ratio` 的取值集合 |
|---|---|---|
| PI | `0.0625`（全维唯一值） | `{1.0}` |
| NTK-aware | `1.000000`（`i=1`）→ `0.062500`（`i=64`），单调递减 | `[1.0, 16.0]`，跨度 `15.0` |
| 不做任何事 | `1`（全维） | `{16.0}` |
| `α ≡ 1`（常数） | `1/16`（全维） | `{1.0}` = **PI** |

★★ **两格最重要**：

- **`α_1 = (1−1)/(h−1) = 0` 精确为 `0`** ⇒ 最高频那一维**根本没被插值**
  （实测 `α_1 = 0.0`、`Φ_ratio_1 = 16.0` **精确**）⇒ ★ 它就是 YaRN 论文说的那个
  **「被外推到 `out-of-bound` 值的维」**（逐字 `some dimensions are slightly extrapolated to
  "out-of-bound" values`）。★ 这与 `02` 的结论**对上**：`θ₁ = 1` 那一维**本来就是最混叠的那一维**，
  而 NTK-aware **对它一点都没动**。
- **PI 的 `Φ_ratio` 恒 `1`**，而 `1` 是**上限**不是目标：它把**全部**维压到上限
  ⇒ 不需要缩的高频也被缩了。★ 这就是原文说的「PI 丢高频」的定量形态
  （`C.1`：`θ_1/s = 0.0625`，「相邻位置可分」的维数从 **1 变成 0**）。

### 1.2 ★★ `02` 与 `03` 之间唯一的那座桥

`02` 的**分辨窗口**是 `θ_i·S`（计划 §0.7-D 的 D6）；
YaRN 论文的判据是 **`L/λ_i`**（逐字：`if the wavelength λ is equal to or bigger than the
context size L, we want to only interpolate and avoid any extrapolation`）。
★★ **两者是同一个量**：

```text
r_i  :=  L / λ_i  =  L·θ_i / (2π)  ==  `02` 的 `turns(θ_i, L)`        （恒等）
```

实测：两个真实模型上 `max|L/λ − turns| / max(turns)` = `8.720e-17` / `1.395e-16`，
而且**两个集合相等**：

| 模型 | `{i : λ_i ≥ L}` | `{i : turns_i(L) ≤ 1}` | 相等？ |
|---|---|---|---|
| Llama-3-8B（`5e5`、`8192`） | `29` 维 | `29` 维 | **是** |
| Qwen3-30B-A3B（`1e6`、`40960`） | `23` 维 | `23` 维 | **是** |

（★ 两侧都**非空**（`29` / `23`）—— 否则「空集 == 空集」是一条恒真断言。）

⇒ **E5 的两条 bullet 逐字落进 `02` 的框架**：

- 「`λ` **much smaller than** `L`」 ⇒ `r_i` 大 ⇒ **整窗转了很多圈** ⇒ **不插值**；
- 「`λ` **equal to or bigger than** `L`」 ⇒ `r_i ≤ 1` ⇒ ★★ **连一圈都没转完** ⇒ **只插值**。

★ 而 `Φ_ratio` 那一侧的分界是 `1`，`r` 那一侧的分界也是 `1`（一圈）——
**两个 `1` 是同一个判据的两种写法**。这就是那座桥。

### 1.3 PI：位置除以 `s`

**定义**（逐字来自 PI 论文的式 (4)：`f'(x, m) = f(x, m·L/L')`）：
把**位置下标** `m` 换成 `m/s`（★ 不是改频率）。于是每维的**有效频率**是 `θ_i/s`，
**全维同一个因子**。`Φ_ratio` 恒 `1`（`§1.1`）。

★ **代价是可算的**（`C.1`）：`θ_1/s = 0.0625`（`s = 16`）
⇒ 按 `02` 的判据「相邻位置可分 = `θ_i ≥ 1`」，满足它的维数从 **1 变成 0**。
⇒ **PI 把唯一能分辨相邻位置的那一维也压没了** —— 这就是「PI 丢高频」。

### 1.4 NTK-aware：换 `base`（★ 两条独立路径）

**定义**：把 `base` 换成 `b' = b·s^{|D|/(|D|−2)}`（论文式 (16) 的逐字）。
它的后果可以写成**闭式**：

```text
θ_i' = θ_i · s^{−α_i},      α_i = (i − 1) / (d/2 − 1)  ∈ [0, 1]
```

实测（`base ∈ {5e5, 1e6}` × `s ∈ {2,4,8,16,32}` 共 10 组）：
**「直接算 `freqs(d, b')`」与「闭式 `θ_i·s^{−α_i}`」的最差相对差 `1.110e-16`**
⇒ ★ 两条**独立路径**逐位一致（程序用 AST 检查闭式那条路**不调用**任何权重函数）。

★ **端点精确**：`α_1 = 0.0`、`α_h = 1.0`（`h = 64`；`α_32 = 0.492063`）。
⇒ `Φ_ratio_1 = 16.0` **精确**（= `s`）、`Φ_ratio_h = 1.0` **精确**。

★ **E3 的定量形态**：原文逐字是

> `Instead of scaling every dimension of RoPE equally by a factor s, we spread out the
> interpolation pressure across multiple dimensions by scaling high frequencies
> less and low frequencies more.`

★ **`less and low frequencies more`** 这一半是计划的 §C.5 里**被截掉**的（原稿在
`high frequencies` 处闭合引文并当作句末）。补上后，它的**定量形态**就是：
实测 `s^{−α_i}` 从 `1.000000`（`i=1`）**单调递减**到 `0.062500`（`i=64`）——
**高频（`i` 小）缩得少、低频（`i` 大）缩得多**。★ 这不是形容，是一条**单调性**。

### 1.5 NTK-by-parts：按 `r = L/λ` 分段，且**从不超过训练范围**

**定义**（论文的 `h(θ_d) = (1−γ)·θ_d/s + γ·θ_d`，`γ` 是一个 ramp）：

```text
r_i = L/λ_i = turns_i(L)          ← ★ 见 §1.2（那座桥）
γ_i = 0            if r_i < α      （α = 1  ⇒ 全插值，与 PI 一样）
    = 1            if r_i > β      （β = 32 ⇒ 完全不插值）
    = (r_i−α)/(β−α) 否则
```

★ **参数名按原文**：`α` 与 `β` 是**阈值**，`γ` 是那个 **ramp 函数**。
⚠ 计划 §C.5 的 E5 把它们写成了 `β`/`γ` 两个阈值 —— **`γ` 是函数名，不是阈值**。

实测 ramp（逐点与定义式**逐位相等**，最大差 `0.000e+00`）：

| `r` | `0.50` | `1.00` | `2.00` | `16.00` | `32.00` | `64.00` |
|---|---|---|---|---|---|---|
| `γ` | `0` | `0` | `0.032258` | `0.483871` | `1` | `1` |

★ **分段在真实模型上的计数**（`d = 128`）：

| 模型 | `γ == 0`（全插值） | `γ == 1`（不插值） | 中间 |
|---|---|---|---|
| Llama-3-8B（`L = 8192`） | `29` 维 | `19` 维 | `16` 维 |
| Qwen3-30B-A3B（`L = 40960`） | `23` 维 | `25` 维 | `16` 维 |

★ 而 `Φ_ratio = (1−γ)/s + γ` **恒落在 `[1/s, 1]`** ⇒ **从不超过训练范围**。
⇒ 这正是 E5 那句 `avoid any extrapolation` 的定量形态，也是它与 NTK-aware 的**全部差别**。

### 1.6 ★★ NTK-aware 与 NTK-by-parts 的**关键差别**

两者的权重函数看起来都是「从 0 到 1 的递增函数」，但**输入不同**：

| | `α_i` | `γ_i` |
|---|---|---|
| 它看什么 | **维号** `(i−1)/(h−1)` | **`r_i = L/λ_i`** |
| 它需要知道 `L` 吗 | **不需要** | **需要** |
| 换 `L` 会怎样（实测） | **逐位不变**（`0.0e+00`） | **变**（最大变化 `0.726`） |

★★ **这就是「NTK-aware 会把最高频维推到 `out-of-bound`、而 NTK-by-parts 不会」的原因**：
`α` **不知道 `L` 是多少**，它按维号硬切一刀；`γ` 用的是**实际的分辨窗口**。
⇒ ★ 那座桥（`§1.2`）不是装饰：**`L/λ` 就是 NTK-by-parts 相对 NTK-aware 多出来的那个输入**。

★ 两个权重函数也**确实不同**（`max|α − γ| = 1.000000`）——
它们不是同一个函数的两种写法（程序用 AST 检查两个函数的源码不同）。

### 1.7 YaRN：NTK-by-parts + 一个温度（★ 那个温度是**拟合**的）

**定义**：在 NTK-by-parts 之上，引入一个**温度** `t`：

```text
√(1/t) = 0.1·ln(s) + 1
```

实测 `s = 16` ⇒ `√(1/t) = 1.277259`、`t = 0.612974`。

★★ **它是拟合值，不是推导值** —— 原文逐字：
`The equation above is found by fitting √(1/t) at the lowest perplexity against the scale
extension by various factors s`。⇒ **不得写成推导**（计划 §3 表 B 的 E7）。

★ **那个温度是怎么实现的**：原文的 `length scaling` trick 把 **`q` 与 `k` 各乘 `√(1/t)`**。
实测（`q`、`k` 各乘 `√(1/t)` vs `logits` 除以 `t`）：**最大差 `0.000e+00`**
⇒ 两者**逐位相等**，是同一个温度操作。

## 2. 结构上差在哪（★ 不是访存账）

**四种方法改的都是同一个东西的位置**：不改 `q`/`k` 的投影、不改 attention 的形状，
只在 **`q`、`k` 进注意力之前**改**位置下标**（PI）或**频率表**（NTK 系）。
YaRN 额外在 **softmax 之前** 动一次 logits 的尺度。

| 方法 | 改什么 | 需要重训吗 |
|---|---|---|
| PI | 位置下标 `m → m/s` | 论文自承需要少量微调（`within 1000 steps`） |
| NTK-aware | `base` | 原文转述：**不微调**时表现比 PI 好（E4 的语境） |
| NTK-by-parts | 逐维频率（按 `r` 分段） | 原文转述：**少量长上下文微调**时最好 |
| YaRN | 逐维频率 + logits 温度 | 同上 |

★ **本文不比较它们的「效果」**：那要训练与评测，本域**无 GPU、不训练模型**。
★ 而且 YaRN 原文自己说：`perplexity may not be a great indicator of whether an LLM is able
to attend to all tokens`（★ 前缀是 `we hypothesize`）⇒ **评价指标的局限是原文自承的**
（计划 §3 表 B 的 E8）。

★ **本文不做**：位置编码的 FLOPs / 访存账（正文在 `40-kernels/attention`）；
变体对 kernel 结构的影响（正文在 `40-kernels/attention/04:126-156`）。

## 3. ALiBi：另一条路（不改位置编码，改 logits）

**定义**（逐字来自 ALiBi 的式 (2)）：

```text
softmax( q_i K^⊤ + m·[−(i−1), …, −2, −1, 0] )
```

- 偏置加在 **`q_i K^⊤` 之后、softmax 之前**；
- `m` 是**训练前固定的、每个 head 一个的 slope**（逐字 `a head-specific slope fixed
  before training`）；★ **不用位置编码**（逐字 `we do not add position embeddings at any
  point in the network`）；
- ★★ **脚注 10 逐字**：`The ALiBi bias is not multiplied by the √d_k scaling factor from
  Equation 1 of Vaswani et al. 2017.` —— **那条偏置不吃 `1/√d_k`**。

**slope 的闭式**（逐字 `our set of slopes is the geometric sequence that starts at 2^{−8/n}
and uses that same value as its ratio`）：

```text
slope_h = (2^{−8/n})^h,   h = 1..n
```

实测：`n = 8` ⇒ `[0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.007812, 0.003906]`
（与原文另一句逐字给出的 `1/2^1, 1/2^2, …, 1/2^8` **最大差 `0.000e+00`**）；
`n = 16` ⇒ 首项 `0.7071067812`（`2^{−0.5} = 1/√2`）、比值同样是 `1/√2`
（与 `2^{−0.5…−8}` 最大差 `1.11e-16`）—— ★ 与原文「起于 `1/√2`、比值 `1/√2`」一致。

**它的两个机械后果**（实测）：

- 对角线偏置**恰好 `0`**、全体 `≤ 0`（`m = 0.5`、`i = 31` 时最远那个是 `−15.5`）；
- 取 `q ≡ 0`（只剩偏置）时，相邻权重的比 `w_{j+1}/w_j` **逐位等于 `exp(m)`**
  （实测 `1.648721270700`，`exp(0.5) = 1.648721270700`），且权重随位置**严格递增**
  ⇒ **ALiBi 是一个纯近因核**（位置越新权重越大）。

★★ **那条脚注是可判定的**（`B.4`，`d_k ∈ {64, 128, 256}`、`4000` 组样本）：

| `d_k` | 加偏置后平均位置前移（偏置**绝对**） | 若偏置**也除** `√d_k` |
|---|---|---|
| `64` | `+13.731762` | `+4.818790` |
| `128` | `+13.785616` | `+3.499132` |
| `256` | `+13.757687` | `+2.506775` |

⇒ ★ 偏置**不吃** `1/√d_k` 时，近因效应在三个 `d_k` 上**一致**（极差 `0.0539`）；
★ **反面对照**：若把偏置也除以 `√d_k`，效应随 `d_k` **缩小**（比值 `1.92 ≈ √(256/64) = 2`）。
⇒ **那条脚注不是脚注：它决定了大 `d_k` 的 head 还保不保有近因偏好。**

## 4. 踩坑与失败模式

### 4.1 把「四种方法」并列成四个互不相干的东西

- **现象**：把 PI / NTK-aware / NTK-by-parts / YaRN 写成四个并列的「方案」。
- **成因**：没有先找出统一量。它们的差别**只在「每维的频率缩放挑多少」**这一件事上。
- **规避**：先定义 `Φ_ratio`（`§1.1`），把四者写成同一个表里的四行；
  `C.2` 还把 `α` 换成常数，**当场退化成另外两种方法** ⇒ 证明「`α` 的形状就是 NTK-aware 的全部内容」。

### 4.2 把 `√(1/t)` 写成「推导出来的」

- **现象**：「由注意力熵的尺度不变性可以推出 `√(1/t) = 0.1·ln(s) + 1`」。
- **成因**：那是一个**拟合式**（原文 `found by fitting … at the lowest perplexity`）。
- **规避**：引用时必须带 `found by fitting`（`specs.YARN_FITTING_E7`，程序逐位钉住）。

### 4.3 把 ALiBi 的偏置当成「也被 `1/√d_k` 缩放」

- **现象**：写「加一项与 `(i−j)` 成正比的偏置」就完了。
- **成因**：漏了**脚注 10**。而漏掉它的后果**不是排版问题**：大 `d_k` 的 head
  的近因偏好会**被压掉**（实测比值 `1.92 ≈ √4`）。
- **规避**：把脚注**逐字存进 `specs`**，并把它的**后果做成一条断言 + 一条反面对照**（`B.4`）。

### 4.4 把 `r` 当成一个新定义的量

- **现象**：在 `03` 里重新定义一套「波长 / 圈数」的语言，与 `02` 各说各话。
- **成因**：没发现 `L/λ_i` **就是** `02` 的 `turns_i(L)`。
- **规避**：`B.1` 把两边做成**恒等式**（最差相对差 `1.4e-16`）与**集合等式**；
  ★ 两侧都断言**非空**，否则「空集 == 空集」是一条恒真断言。

### 4.5 把「`α` 与 `γ` 都是递增的 0→1 函数」当成「它们是同一个东西」

- **现象**：写「两个方法都是按维度把插值强度从 1 降到 0」。
- **成因**：只看形状、不看**输入**。`α` 的输入是**维号**、`γ` 的输入是 **`r = L/λ`**。
- **规避**：`B.3` 换一个 `L`（换模型）⇒ 断言 `α` **逐位不变**、`γ` **变了**（`0.726`）。

## 5. 判据表

| 可机械判定的条件 | 结论 |
|---|---|
| PI 的每维频率缩放 | **恰好 `1/s`**（全维同一值）⇒ `Φ_ratio` **恒 `== 1.0`** |
| NTK-aware：`freqs(d, b')` vs 闭式 `θ_i·s^{−α_i}` | 最差相对差 `1.110e-16` ⇒ **两条独立路径**一致 ⇒ 「换 base」=「按维号缩放」 |
| `α` 的两个端点 | **精确** `α_1 = 0.0` / `α_h = 1.0` ⇒ `Φ_ratio_1 **恰好 = s**` ⇒ **最高频那一维没被插值**（`out-of-bound`） |
| NTK-aware 的 `s^{−α_i}` 随 `i` | **单调递减**（`1.000000 → 0.062500`）⇒ E3 的「高频少缩、低频多缩」是**单调性** |
| `r_i = L/λ_i` vs `02` 的 `turns_i(L)` | **恒等**（最差 `1.4e-16`），且 `{λ_i ≥ L} == {turns_i(L) ≤ 1}`（**集合相等**，两侧非空）⇒ **E5 与 D6 是同一件事** |
| NTK-by-parts 的 `γ` | 三段**逐位等于**定义式；阈值是论文的 `α = 1` / `β = 32`（★ `γ` 是那个 ramp 函数，**不是**阈值） |
| 换 `L`（换模型）时 `α` 与 `γ` 的行为 | `α` **逐位不变**（`0.0e+00`）、`γ` **变**（`0.726`）⇒ 这是「NTK-aware 会 `out-of-bound` 而 NTK-by-parts 不会」的原因 |
| `√(1/t) = 0.1·ln(s) + 1` | 是**拟合值**（原文 `found by fitting`）⇒ **不得写成推导**；「`q`、`k` 各乘 `√(1/t)`」与「logits 除以 `t`」**逐位相等** |
| ALiBi 的偏置 + **脚注 10** | 偏置**不乘 `1/√d_k`** ⇒ 近因效应与 `d_k` **无关**（极差 `0.0539`）；★ 反面对照：也除 `√d_k` 则效应随 `d_k` 缩小（`1.92 ≈ √4`） |
| ALiBi 的 slope | 闭式 `(2^{−8/n})^h`：`n = 8` 复现 `1/2^1…1/2^8`（差 `0`）、`n = 16` 起于 `1/√2` 比值 `1/√2` |
| ALiBi 的偏置式 | 对角线**恰好 `0`**、全体 `≤ 0`；`q ≡ 0` 时权重比 **逐位等于 `exp(m)`** ⇒ 纯近因核 |
| 引用「RoPE 外推不行」 | ★★ **出处必须是 ALiBi §2 或后续工作**（PI §1），**不能归给 RoPE 论文**（那里 `extrapol` 计数 **0**，已配对照检验） |
| 引用 NTK 系列 | ★★ **只能引 YaRN 的转述（C 级）**；它的**一手来源**是 Reddit / GitHub（**D 级**，原文自承 `previous unpublished works`）⇒ 不得当数值来源 |
| 想知道哪种方法**效果更好** | **本域给不出**（无 GPU、不训练、不跑评测）；且原文自承 `perplexity may not be a great indicator` |
| 想知道变体对 **kernel 结构**的影响 | **不在本文**：`40-kernels/attention/04:126-156`（**正文在彼，此处回链**） |

> ★ **表 A 的数值判据**（计划 §3 Chunk 5）：③④⑤ 已在 `02` 交付；
> 本文的判据是**表 B 的引用判据**（E1/E3/E4/E5/E7/E8/E9 + X17 + X1）
> 与 §1 的**结构闭式**。★ 前者**不能被程序判定**，但「我抄的这一句是不是原文」
> 可以：YaRN / PI / ALiBi 三批逐字存在 `specs`，由 `D.4`–`D.7` 逐位比较。
> ★★ **三次「结构选择翻转」的代码级抽查**（补 `make mutate` 的第一个盲区）：
> ① 把 NTK-aware 的指数从 `1 − α` 翻成 `α` ⇒ `A.3` **与** `C.2` **同时失败**；
> ② 把 `γ` 的三段 ramp 换成常数 `0` ⇒ `A.4` **与** `B.3` **同时失败**；
> ③ 把 `γ` 的输入接成 `α_by_index`（两个权重函数的输入换位）⇒ `A.4` + `B.2` + `B.3` 失败。
> 三次都是**断言捕获**（rc=1、打印的是失败标记，不是 traceback）。
> ★★ 这四处改动**一个字符面量都没动** ⇒ `make mutate` 对它们**全绿** ——
> 这正是那个盲区的现场（`01` 的 `§4.5` 与 `§5` 末段是同一类抽查）。

## 6. 小结与延伸

**一句话**：这一篇的账是**位置容量的账（工程上界）** ——
四种 RoPE 外推方法是**同一个一参数族的四个成员**（统一量是相位比 `Φ_ratio`），
它们的差别**可以逐位算出来**；ALiBi 是另一条路（改 logits 的偏置，用**近因**代替位置编码）。

- **前置**：[`02-spectrum-and-aliasing.md`](02-spectrum-and-aliasing.md)（谱 / 混叠 / 分辨窗口）。
  ★ 那座桥是 `r_i = L/λ_i ≡ turns_i(L)`（`§1.2`）。
- **相关的架构问题**：`attention-variants/03-window-and-hybrid` 讲**另一种**扩展上下文的方式
  （堆叠层的感受野 `W×L`），它与本文是**两件事**：一个是「看多远」、一个是「位置怎么表示」。
- **上游的悬空让渡**：`40-kernels/attention/04:128` 说「三个常见变体，本文只讲它们对 kernel
  结构的影响，定义不重推」—— 那一份让渡里 **ALiBi 归本文**（另两个：滑窗归
  `attention-variants/03`，softcap 属 logit 处理、不在本域）。

## 编译与运行

```bash
cd code
make run                              # 三个编号程序依次跑；本程序是 03_extrapolation.py
python3 03_extrapolation.py           # 只跑本程序（约 0.4 s）
```

- 配套程序：**`code/03_extrapolation.py`**（编号与外键严格对应），**26 条断言**。
- 依赖：`numpy`（`requirements.txt` 里**只有**它）。**不需要网络、不需要 GPU。**
- 失败标记：`$$ FAIL: positional-encoding`（本主题自定义，**不跨主题照抄**）。
- 程序的输出分五段：`D`（口径/阈值/specs 钉住，**刻意放在最前**）、
  `A` 四种方法的算式（闭式 + 两条独立路径）、`B` 桥 / E3 / E4 / ALiBi、
  `C` 边界与反例、`D（续）` 标记自检与参照实现的独立性。

## 参考

### 表 B：引用判据（**不能被程序判定**，在**原始字节**上计数）

实跑（2026-09-22；`WebFetch` 被策略拦截 ⇒ 用 `curl`；★ **最简形式即可**）：

```bash
curl -sS -o /tmp/arx_2309.00071.html https://ar5iv.labs.arxiv.org/html/2309.00071  # 232997 B
curl -sS -o /tmp/arx_2306.15595.html https://ar5iv.labs.arxiv.org/html/2306.15595  # 217296 B
curl -sS -o /tmp/arx_2108.12409.html https://ar5iv.labs.arxiv.org/html/2108.12409  # 333702 B
curl -sS -o /tmp/arx_2104.09864.html https://ar5iv.labs.arxiv.org/html/2104.09864  # 283120 B
```

| 字面量 | 计数 | 用途 |
|---|---|---|
| `proposed the "NTK-aware" interpolation by taking the loss of high frequency into account` | **1** | E1（主线：PI 丢高频 → NTK 来补） |
| `we spread out the interpolation pressure across multiple dimensions by scaling high frequencies` | **1** | ★ E3 的**前半** |
| `less and low frequencies more` | **1** | ★★ E3 的**后半**（计划 §C.5 的 E3 在这里**截断**了引文） |
| `some dimensions are slightly extrapolated to "out-of-bound" values` | **1** | E4（`Φ_ratio_1 = s` 的论文表述） |
| `is equal to or bigger than the context size` / `we want to only interpolate and avoid any extrapolation` | **1 / 1** | ★★ E5（**那座桥**的论文表述） |
| `The equation above is found by fitting` | **1** | ★ E7（`√(1/t)` 是**拟合值**） |
| `perplexity may not be a great indicator` | **1** | E8（前缀是 `we hypothesize`） |
| `previous unpublished works` | **1** | ★★ E9（**等级纪律的唯一出处**） |
| `{b^{\prime}}=b\cdot s^{\frac{|D|}{|D|-2}}.` | **2** | NTK-aware 的换 base 式（逐字） |
| `=\Big(1-\gamma\big(r(d)\big)\Big)\frac{\theta_{d}}{s}+\gamma\big(r(d)\big)\theta_{d}.` | **2** | ★★ 统一形式 `h(θ) = (1−γ)θ/s + γθ` |
| `\sqrt{\frac{1}{t}}=0.1\ln({s})+1` | **4** | YaRN 的温度（**拟合值**） |
| `\alpha=1` / `\beta=32` | **2 / 2** | ★ 论文的**参数名**（`α`/`β` 是阈值，`γ` 是 ramp 函数） |
| `NTK-by-parts"` / `NTK-by-part"` | **10 / 1** | ★★ 术语的计数（见下） |
| `Though the rotary position method improves over the sinusoidal one, it still does not achieve satisfying results.` | **1** | ★★ **X1 的落点**（ALiBi §2 对 RoPE 外推的评价） |
| `\text{softmax}(\mathbf{q}_{i}\mathbf{K}^{\top}+m\cdot[-(i-1),...,-2,-1,0]),` | **2** | ★ X17 的偏置式 |
| `The ALiBi bias is not multiplied by the` | **1** | ★★ **X17 的脚注 10** |
| `our set of slopes is the geometric sequence that starts at` | **1** | ALiBi 的 slope 规则 |
| `\mathbf{f}’(\mathbf{x},m)=\mathbf{f}\left(\mathbf{x},\frac{mL}{L^{\prime}}\right).` | **2** | PI 的式 (4)（★ 撇号是 **U+2019**，照 ASCII 抄会计数 0） |
| `weak extrapolation properties` | **1** | PI §1（「RoPE 外推弱」的另一条后续来源） |
| `extrapol`（在 **RoPE 原文**里） | **0** | ★★ **X1**：那句话**不是** RoPE 论文说的 |

★★ **一条计数方法的注记**（本域实测）：`NTK-by-parts"` = **10**、`NTK-by-part"` = **1** ——
**这两个数必须带闭引号**。不带闭引号时会得到 `NTK-by-part` = **13** > `NTK-by-parts` = **12**
（因为前者是后者的**子串**）⇒ **结论会反过来**（「论文写单数」）。
★ 这正是计划 §C.5 那条术语口径注记的机制，也是 §0.3「计数必须带上能闭合的边界」的一个现场。

**外部来源值 / 论文**

- **YaRN**：arXiv 2309.00071（★ abs 页**无 venue 线索** ⇒ **未确认同行评审**；
  **232997 B**，`sha256[:16] = 70b5328c47374b18` —— 与计划 §C.2 逐位一致）。**[C]**
- **Position Interpolation (PI)**：arXiv 2306.15595（**无 venue 线索** ⇒ **未确认**；
  **217296 B**，`sha256[:16] = fd15ea37b56bee34`）。
  ⚠ 计划 §C.2 记的是 **abs 页**的哈希（`deb27db06526bf96`），本域取的是 **ar5iv HTML 页** ——
  **两者不是同一个页面**，哈希不同是预期的。
- **ALiBi**：arXiv 2108.12409（**无 venue 线索** ⇒ **未确认**；
  **333702 B**，`sha256[:16] = 971a61398ba0f189` —— 与计划 §C.2 逐位一致）。
- **RoPE**：arXiv 2104.09864（**未确认**；**283120 B**，`sha256[:16] = cfec27d698ce4881`）。
- **两个真实模型的 `config.json`**：**经 `hf-mirror.com` 镜像取得，未与官方 repo 比对 sha256**
  （Llama 副本 `2430cee764b6530f`（654 B）、Qwen3 副本 `2850ddb3bf7aecad`（963 B））。**[镜像]**
- ★★ **一张不在本文的名单**：NTK-aware / Dynamic NTK / NTK-by-parts 的**一手来源**是
  Reddit 帖与 GitHub 仓库（**D 级**，YaRN 原文自承 `previous unpublished works`）
  ⇒ 本域**只引 YaRN 对它们的转述（C 级）**，**不把社区帖子当数值来源**（见 `specs.UNSOURCED`）。

**本机实测值**

- `code/03_extrapolation.py` 的 `make run` 输出：`python3 3.13.5` + `numpy 2.4.4`，
  macOS arm64，**无 GPU**。口径：`d = 128`（`h = 64`）、`base ∈ {1e4, 5e5, 1e6}`、
  `s ∈ {2,4,8,16,32}`、`α = 1`/`β = 32`、ALiBi 段 `m = 0.5`、`i = 31`、`S = 32`、
  `4000` 组样本、`d_k ∈ {64,128,256}`、种子 `20260922`。
  **全部是推导值与本机实测值**，不是任何真实模型的测量。
- ★ **与 D 段的关系**：本文的判据**不依赖**任何 `D` 行的读数
  （D 段里没有外推方法；那些是 `02` 的口径）⇒ 不存在「同口径不同抽样」的对照问题。
  ★ 但 `§1.2` 的桥**引用了** `02` 的 `turns`，而 `02` 与 D6 是逐位一致的。

**上游契约**

- `40-kernels/attention/04-warp-specialization-and-variants.md:128`：「三个常见变体，本文只讲
  **它们对 kernel 结构的影响**，**定义不重推**」—— 这是**悬空让渡**；本文接住 **ALiBi** 那一份。
- `40-kernels/attention/04:130-134` 那张「改什么」的一行表（ALiBi = 在 `S` 上**加一项**与
  `(i−j)` 成正比的偏置）是本文的**对接面**：那边给一行，本文给它完整的定义、那条脚注、
  以及那条脚注的**可判定后果**。
- `130-quantization/kv-cache-quantization/02-kivi-asymmetry.md:184` —— RoPE 的
  「成对通道旋转」的逐字（**正文在彼，此处回链**；定义在 `01`）。

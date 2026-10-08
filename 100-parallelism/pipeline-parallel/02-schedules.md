---
title: 三种流水线调度：GPipe、1F1B 与 interleaved —— 以及 1F1B 换到的到底是什么
slug: pipeline-parallel-schedules
domain: 100-parallelism
topic: pipeline-parallel
level: 3
difficulty: intermediate
prerequisites:
  - 100-parallelism/pipeline-parallel/01-microbatches-and-bubbles
tags: [pipeline-parallel, 1f1b, gpipe, interleaved, pipedream-flush, virtual-stages, bubble]
related:
  - 100-parallelism/pipeline-parallel/03-memory-balance
  - 100-parallelism/tensor-parallel/03-saturation-and-limits
  - 100-parallelism/data-parallel/03-redundancy-and-limits
code:
  - code/02_schedules.py
status: stable
updated: 2026-10-08
---

# 三种流水线调度：GPipe、1F1B 与 interleaved

> ★★★ **一句话**：**1F1B 换的是【显存】，不是气泡率。**
> 原文逐字：`The time spent in the bubble is the same for this new schedule` ——
> 它换到的是「**需要暂存的激活从 `m` 个 micro-batch 降到 `p` 个**」。
> ★★ **而「1F1B 把气泡从 `(p−1)/(m+p−1)` 降到 `(p−1)/m`」这个流传很广的说法是错的** ——
> 那两个公式的差别是**分母口径**（[01](01-microbatches-and-bubbles.md) 的 §1.2），**不是任何改善**。
> **真正改善气泡的是 interleaved**，而它的代价是**通信量**。

## 0. 背景与动机

[01](01-microbatches-and-bubbles.md) 讲了气泡怎么算，但留了一个没说的事：

**「先全部前向、再全部后向」这个顺序是不是必须的？**

不是。`m` 个 micro-batch 的前向与后向**可以交错**。而一旦交错，**两条账会同时变**：

- **气泡**：变不变？
- **显存**：变不变？（★ 一个 micro-batch 的激活要从「它走完最后一级前向」留到「它走完第一级后向」
  —— **这段时间越长，同时在飞的激活越多**。）

**答案出人意料**：1F1B 让**第二条账大幅改善**、**第一条账完全不变**。
★ 而正因为「不变」这件事不直观，它成了本主题最容易被写错的一句。

## 1. 原理

### 1.1 三种调度，三句话

| 调度 | 一句话 | 它改善什么 |
|---|---|---|
| **GPipe** | 先把 `m` 个 micro-batch **全部**推完前向，再**全部**推后向 | （基线） |
| **1F1B**（PipeDream-Flush） | warm-up 之后，每个 worker **一次前向接一次后向** | ★★ **显存** |
| **interleaved**（model chunk） | 每个设备拿 **`v` 个不连续的层段**（chunk），轮流推 | ★★★ **气泡** |

### 1.2 ★★★ 1F1B：气泡**不变**，在飞数从 `m` 降到 `p`

原文逐字（★ **三段分别对应 `specs.py` 的三个常量**，见下；★ 常量一律**只取不含公式的散文**）：

> `The time spent in the bubble is the same for this new schedule`（`MEGATRON_1F1B_SAME_BUBBLE_QUOTE`）
> `, but the number of outstanding forward passes is at most the number of pipeline stages …`
> `this schedule requires activations to be stashed for`（`MEGATRON_1F1B_STASH_QUOTE`）
> `p or fewer microbatches (compared to` `m` `microbatches for the GPipe schedule).`
> （★ 这一句的 `p`/`m` 是**数学斜体**、在 `<math>` 里 ⇒ 常量 `MEGATRON_1F1B_VS_GPIPE_QUOTE`
> 只取了它们**之后**的散文 `microbatches for the GPipe schedule`）

**三件事逐条**：

1. **`The time spent in the bubble is the same`** —— ★★★ **气泡时间**与 GPipe **相同**；
2. **在飞前向数 ≤ 流水线级数** —— ★ **`p` 个或更少**（GPipe 是 **`m` 个**）；
3. **`m ≫ p` 时显存省得多** —— ★ 所以这条改善**只在 `m ≫ p` 时显著**。

**本机实测**（`code/02_schedules.py` 的 `A.1`/`B.1`，两种调度**各写一个独立调度器**）：

| `(p, m)` | GPipe 总时间 | 1F1B 总时间 | 差 | GPipe 峰值在飞 | 1F1B 峰值在飞 |
|---|---|---|---|---|---|
| `(4, 8)` | `33.00` | `33.00` | `0.000e+00` | **`8`（= `m`）** | **`3`（≤ `p`）** |
| `(4, 32)` | `105.00` | `105.00` | `0.000e+00` | **`32`** | **`3`** |
| `(8, 16)` | `69.00` | `69.00` | `0.000e+00` | **`16`** | **`5`** |

⇒ ★★★ **`(4, 32)` 那一行就是本主题的全部内容**：**时间一模一样**（`105.00`），
**而在飞数从 `32` 掉到 `3`**。

★★ **「在飞」的定义**（必须写清）：**前向已走完最后一级、而后向还没走完第一级**的那些 micro-batch
—— 它们的激活**必须保留** ⇒ ★ 它是**显存**的代理量。
`code/02_schedules.py` 的 `_run_stage()` **按区间重叠真的数**（不是估的）：
每个 micro-batch 的在飞区间是 `[最后一级前向开始, 第一级后向开始]`，峰值 = 这些区间的最大重叠数。

### 1.3 ★★★ interleaved：真正改善气泡的那个

原文逐字（★ 三个常量 `specs.MEGATRON_INTERLEAVED_BUBBLE_QUOTE` / `_FREE_QUOTE` / `_COMM_QUOTE`
**各自只取一句引导语**，公式与变量跨 `<math>` 标记 ⇒ 不在常量里）：

> `The pipeline bubble time thus reduces to` `t_pb^int = (p−1)(t_f+t_b)/v,`
> `and the bubble time fraction is then` `(1/v)·(p−1)/m.`
> `This reduced pipeline bubble size, however, does not come for free:`
> `this schedule requires extra communication. Quantitatively, the amount of communication also increases by` `v.`

（★ 订正 2026-10-08：首句常量原写成小写 `the` —— 原文这句在**句首**，是 `The`。）

**机制**：每台设备拿 **`v` 个不连续的层段**（原文叫 **model chunk**）。
每个 chunk 的前向/后向时间是原来的 `1/v` ⇒ **气泡时间 ∝ `1/v`**。

**本机实测**（`B.2`，`p = 8, m = 16`）：

| `v` | 气泡占比（实测） | 闭式 `(1/v)·(p−1)/m` |
|---|---|---|
| `1` | `0.437500` | `0.437500` |
| `2` | `0.218750` | `0.218750` |
| `4` | `0.109375` | `0.109375` |

**代价**（`B.3`，同参数）：点对点通信**次数** `v=1 → 224`、`v=2 → 480`、`v=4 → 992`。

⇒ ★★ **两个方向相反的量**：气泡 `∝ 1/v`、通信 `∝ (约) v` ⇒ **存在最优 `v`**。
★★ 而「约」那个字很重要 —— 见 §1.4。

### 1.4 ★★★ 「通信增 `v` 倍」是**渐近式**，精确值是 `(p·v − 1)/(p − 1)`

原文说 `the amount of communication also increases by v` —— ★ **但那是大 `p` 下的近似**。

**精确式**（本库 `[E]`，按「一个 micro-batch 要经过全部 `p·v` 个 chunk、
相邻 chunk 之间一次点对点、前向与后向各一遍」算）：

```text
通信次数之比 = (p·v − 1)/(p − 1)
```

**本机实测**（`C.1`/`C.2`）：

| `p` | `v` | 精确倍数 | 原文的「`v` 倍」 | 相对误差 |
|---|---|---|---|---|
| **`2`** | `2` | **`3.000000`** | `2` | **`50.00%`** |
| `4` | `2` | `2.333333` | `2` | `16.67%` |
| `8` | `2` | `2.142857` | `2` | `7.14%` |
| `64` | `2` | `2.015873` | `2` | `0.79%` |
| **`256`** | `2` | **`2.003922`** | `2` | **`0.20%`** |

★★★ **两个方向都量到，「近似何时可用」才说清楚**：
- **`p = 2`（最小的非平凡流水线）时错一半** —— ★ 拿它当精确式会**严重低估**代价；
- **`p = 256` 时差 `0.2%`** —— 原文的近似在真实规模下**够用**。

★ **成因**：`(p·v − 1)/(p − 1) → v`（`p → ∞`）—— 分子分母的 `p` 次项相消，
剩下的 `−1` 相对 `p·v` 可以忽略。

### 1.5 术语：**不要用 `VPP`**

★★ **「virtual pipeline stages」这个短语在 `2104.04473` 里出现 0 次**。
原文用的词是 **interleaved** 与 **model chunk**。
`VPP`（virtual pipeline parallelism）是**后来的工程文档**的术语
⇒ ★ **本域一律用原文的词**；要提 `VPP` 时必须写明「那是工程文档的叫法」。

### 1.6 ★★ 「气泡与 GPipe 相同」**不是一句可以顺手写下的结论**

它依赖**具体调度**：1F1B 的 warm-up 阶段每个流水线级做**不同次数**的前向
（流水线级 `s` 做 `p−1−s` 次），之后才进入「一前一后」，最后还有 cooldown。

★ **守卫**（`code/02_schedules.py` 的 `C.3`）：**把 warm-up 步数改成 `0`** ⇒
1F1B 的总时间从 `33.00` **变成 `96.00`**（★ 订正 2026-10-08：原写「从 `33.00` 变成 `33.00`？」
—— 那会让读者以为「没变」，而实测**它变了**）⇒
`A.1` 的「两种调度总时间相同」**会失败** ✓
⇒ 这说明「相同」是**那个具体 `order` 的后果**，不是两种调度天生一样。

## 2. 执行映射

### 2.1 本机怎么算

`code/02_schedules.py` 为**两种调度各写一个 `order` 生成器**
（`build_order_gpipe` / `build_order_1f1b`），它们**各自生成 `order`**（每级的操作序列），
**只共享一个最小原语** `_run_stage()`。

★★★ **独立性证据**（`D.4`，★ **源码级线索**、不是证明）：**两个生成器的源码不同**、
各自产出的 `order` 也不同 ⇒ 「总时间相同」**不是**把同一个式子写两遍
（★ 行为级由 `A.2`（稳态交替）与 `B.1`（在飞数）兜底）。

★★ **`_run_stage()` 用「迭代到不动点」而不是「按列推进」** ——
★ 因为**后向的依赖方向（`s+1`）与前向（`s-1`）相反**，按列推进会对后向算出**错序**。
★ 本程序的**第一版就踩了这个**（`KeyError`）；★ 而所有约束**无环** ⇒ 迭代**必然收敛**。

### 2.2 与 `01` 的分工

| 问题 | 在哪 |
|---|---|
| 「气泡率怎么算」「两个口径」 | [01](01-microbatches-and-bubbles.md) |
| 「哪种调度换到了什么」 | **本文** |
| 「各流水线级的负载不均」 | [03](03-memory-balance.md) |

## 3. 实现与演进

### 3.1 三条路线的取舍

| 想省 | 用 | 代价 |
|---|---|---|
| **显存**（在飞激活） | **1F1B** | 气泡**不变**（★ 它本来也没换来气泡） |
| **气泡** | **interleaved** | 通信量（约 `v` 倍；★ 精确 `(p·v−1)/(p−1)`） |
| **两样都要** | ★ **两者叠加** | 两笔代价**同时**付 |

★★ **一条容易忽略的事**：**1F1B 与 interleaved 叠加**是常见配置（Megatron 里两者都有）。
⇒ 「1F1B 换显存」与「interleaved 换气泡」是**两笔独立的交换**，可以**同时做**。

### 3.2 为什么 1F1B 不改善气泡（一个直观的解释）

气泡来自**流水线的头尾**：第一个 micro-batch 进入时前 `p−1` 级是空的，
最后一个 micro-batch 离开时后 `p−1` 级也是空的。★ **这段「填满/排空」的时间与调度无关** ——
它由 `p` 与「一个 micro-batch 走一级的时间」决定。
★ 1F1B 改变的是**中间稳态的排布**（前向与后向交错），**它不动头尾**。
⇒ **头尾不变 ⇒ 气泡不变** ✓ 这与原文的 `the same` 一致。

## 4. 踩坑与失败模式

### 4.1 ★★★ 把 1F1B 说成「气泡从 `(p−1)/(m+p−1)` 降到 `(p−1)/m`」

★ **这个失败模式的完整拆解（现象 / 成因 / 规避）在
[01 §4.1](01-microbatches-and-bubbles.md) —— 本文不重抄一遍**（★ 订正 2026-10-08：
本节原来与 01 §4.1 是**同一段整段复制**，连举例的数字 `0.273 → 0.375` 都一样）。

★ **本文只补一个【本调度】的角度**（那一段没写的）：**说这话的人多半是在比较「两种调度」**——
他看到 GPipe 的 `0.273`（÷ 总时间）与 1F1B 之后的 `0.375`（÷ 理想时间），
于是读成「1F1B 把气泡**调大**了」。★ 而真相是**两个数描述同一个气泡**：
1F1B **既不改善也不恶化**气泡（§1.6），它动的是**在飞显存**。

### 4.2 ★★ 把「通信增 `v` 倍」当精确式

- **现象**：按 `v` 倍算 interleaved 的通信代价，在小 `p` 下**低估一半**。
- **成因**：原文那半句是**渐近式**（§1.4）。
- **规避**：用 `(p·v−1)/(p−1)`；★ 或写明「在 `p ≥ 64` 时 `v` 倍够用，`p = 2` 时错 50%」。

### 4.3 ★★ 用「两种调度都调用同一个时间函数」来证明气泡相同

- **现象**：`A.1` 通过，但它证明的是**同一个函数等于它自己**。
- **成因**：**模拟器同时是被测对象**（本域头号错误模式）。
- **规避**：**两个调度器各自生成 `order`**（各自的生成函数）+ **源码级线索**（`D.4`），
  再配**行为级**的 `A.2`（稳态交替）与 `B.1`（在飞数）。

### 4.4 ★ 把 `VPP` 当原文术语

- **现象**：写「原文用 virtual pipeline stages 降低气泡」。
- **成因**：★ 那个短语在 `2104.04473` 里**出现 0 次**（§1.5）。
- **规避**：用 **interleaved** / **model chunk**。

### 4.5 ★ 忘了 cooldown ⇒ 1F1B 的总时间偏小

- **现象**：`A.1` 失败（1F1B 比 GPipe 快）。
- **成因**：稳态之后还有 **cooldown**（把剩余的在飞 micro-batch 的后向做完）；
  漏了它，1F1B 会被算成「提前结束」。
- **规避**：★ `C.3` 的守卫**正是这一类**（改 warm-up ⇒ 总时间变）✓

## 5. 判据表

| 可机械判定的条件 | 结论 |
|---|---|
| 有人说「1F1B 降低了气泡率」 | **错** —— 原文 `the same`；★ 他多半是**把两个口径的数拿来比了** |
| 报「1F1B 省了多少显存」 | 用**在飞 micro-batch 数**：**GPipe `m` → 1F1B `≤ p`**；★ `m ≫ p` 时才显著 |
| 要报 interleaved 的气泡改善 | 公式 `(1/v)·(p−1)/m`（实测 `v=1/2/4` 逐个钉住） |
| 要报 interleaved 的通信代价 | ★★ **先看 `p`**：`p ≥ 64` 用「`v` 倍」够；**`p = 2` 时必须用 `(p·v−1)/(p−1)`**（差 `50%`） |
| 两种调度的总时间 | **必须相同**（差 `≤` 机器精度档）；★ 若不同，先查 **warm-up 步数**与 **cooldown** |
| 证明「总时间相同」 | **不能**用「两者共用同一个时间函数」—— 要有**两个独立的 `order` 生成器** + `D.4`（**源码级线索**）；★ 行为级由 `A.2`/`B.1` 兜底 |
| 在文中提「virtual pipeline stages」 | ★ 加注「那是工程文档的叫法」—— **原文用 interleaved / model chunk** |
| 模拟器算出 1F1B 比 GPipe 快 | **错** —— 查 cooldown 是否漏了 |
| 说「1F1B 与 interleaved 只能选一个」 | **错** —— 它们换的是**两笔不同的账**（显存 / 气泡），**可以叠加** |
| 引用本主题的同行评审依据 | ★ 用 arXiv abs 页 Comments 的 `Accepted to SC 2021`；★★ **不要挂 ACM DOI**（该页上没有） |

## 6. 小结与延伸

**一句话**：**1F1B 换显存、interleaved 换气泡** —— 两笔独立的交换，可以叠加；
而「1F1B 换气泡」是**把两个口径的数拿来比**的产物。

**三条可独立复用的结论**：

1. **1F1B 的气泡时间与 GPipe 完全相同**（实测 `(4,32)`：两种都 `105.00`），
   而**在飞数从 `m` 掉到 `≤ p`**（实测 `32 → 3`）；
2. **interleaved 的气泡 `∝ 1/v`**（实测 `0.4375 / 0.2188 / 0.1094`，对 `v = 1/2/4`），
   **代价是通信**（`224 / 480 / 992` 次）⇒ **存在最优 `v`**；
3. **「通信增 `v` 倍」是渐近式**，精确是 `(p·v−1)/(p−1)` —— `p = 2` 时**错 `50%`**、
   `p = 256` 时错 `0.2%`。

**延伸**：

- **气泡怎么算、两个口径怎么换** → [01-microbatches-and-bubbles](01-microbatches-and-bubbles.md)；
- **各流水线级的负载不均** → [03-memory-balance](03-memory-balance.md)；
- **PP 的通信为什么是点对点、以及它与 TP 的区别** →
  `10-foundations/performance-models/02-amdahl-and-scaling.md:417`（此处回链）
  （★ 订正 2026-10-08：原回链的是 `:223` —— 那一行是**纯 TP 段**（讲 TP 的通信发生在每层内部），
  **没有**「PP 是点对点」这一句；正确的落点是 **`:417`**）；
- ★★ **点对点通信的实现**与**通信-计算重叠的机制**（本文只用到「相邻流水线级之间传一次激活」这个事实）
  → `110-communication/` 的 `overlap` 主题（**未建成**）—— ★ 本文**一行都不重写**它。

## 参考

- Narayanan et al., *Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM*,
  arXiv:2104.04473，**§2.2.1**（1F1B / PipeDream-Flush 与 `the same`）、
  **§2.2.2**（interleaved 的公式与代价）。★ **同行评审：是**
  （arXiv abs 页 Comments 字段逐字 `Accepted to SC 2021`）。
  ★★ **该 abs 页上【没有】ACM DOI**（实测 `3458817` 与 `10.1145` 各 `0` 次；
  页面唯一的 DOI 是 arXiv 自己的 `10.48550/arXiv.2104.04473`）——
  ★ 那个 ACM DOI **本身是真的**（Crossref 有记录），但**它不在这张页面上** ⇒ 本文不登记它。
- 本仓库 [01-microbatches-and-bubbles](01-microbatches-and-bubbles.md) —— 气泡的两个口径与换算恒等式；
  [03-memory-balance](03-memory-balance.md) —— 各流水线级的负载均衡。

## 编译与运行

```sh
cd code && make run          # 三个编号程序，任一断言失败即非零退出
make 02_schedules            # 只跑本文的配套程序
```

★ 运行条件：`python3 3.13.5`，macOS arm64，**无 GPU、无多卡、无 NCCL、无真实网络**。
★★ **本主题的三个程序不用 `numpy`**（纯算术 + 离散事件模拟）—— 见 `code/requirements.txt`。
★ 强制限定语：**不是 NCCL，也不是真机**。
★ `t_f = 1.0`、`t_b = 2.0` 是**本库的设计假设**（「后向是前向两倍」），**不是任何模型的实测**；
★★ **「在飞 micro-batch 数」是【显存】的代理量**，不是显存的字节数（字节账在 [03](03-memory-balance.md)）。

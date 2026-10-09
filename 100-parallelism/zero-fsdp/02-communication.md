---
title: ZeRO 三个阶段的通信量：不增 / 2Ψ / 3Ψ，以及一个记号的桥
slug: zero-fsdp-communication
domain: 100-parallelism
topic: zero-fsdp
level: 3
difficulty: intermediate
prerequisites:
  - 100-parallelism/zero-fsdp/01-three-stages
  - 100-parallelism/data-parallel/01-gradient-averaging
tags: [zero, all-gather, reduce-scatter, communication, amdahl]
related:
  - 100-parallelism/zero-fsdp/01-three-stages
  - 100-parallelism/zero-fsdp/03-fsdp-and-correctness
  - 10-foundations/performance-models/02-amdahl-and-scaling
code:
  - code/02_communication.py
status: stable
updated: 2026-10-08
---

# ZeRO 三个阶段的通信量：不增 / 2Ψ / 3Ψ，以及一个记号的桥

> **上一支讲了「各切什么、省多少显存」**（[01-three-stages](01-three-stages.md)）。
> **这一支讲它付的通信代价**：`P_os` 与 `P_g` **不增加任何通信**、`P_os+g` 是 **`2Ψ`**
> （**与 baseline DP 完全相同**）、**只有** `P_os+g+p` 是 **`3Ψ`（`1.5x`）**。
> ★★★ **本文的核心是一条「归属」纪律**：**别把后一个阶段的账算到前一个阶段头上**。

## 0. 背景与动机

`01` 的结论是：`P_os`/`P_os+g` 这两步**只切状态、不碰参数的集合方式**，
⇒ 它们几乎「免费」。★ 而 `P_os+g+p` 要**按需 all-gather 参数、用完即丢** ⇒ 它**要多通信**。

**本文要回答的只有一件事**：**每个阶段每步搬多少**（以 `Ψ` 为单位）。
★★★ 这件事有两个**非常容易踩的坑**：
1. **把「`P_os+g+p` 的 `3Ψ`」套到「`P_os`」头上**（**记错阶段**）—— 它与 baseline `2Ψ` **直接矛盾**；
2. **给 `P_os` 编一个「论文式(N)」的等式** —— ★ **原文【没有】 `P_os` 单独的通信量等式**。

★ **本文不写集合通信的算法与复杂度**（ring / tree all-reduce、`V(N) = 2(N−1)/N × S`）——
那是 `10-foundations/performance-models/02-amdahl-and-scaling.md:209,428` 与
`110-communication/collectives`）的正文（见 `§3.2`）。

## 1. 原理

### 1.1 ★★ 先看 baseline：`2Ψ` 里**已经包含一次 all-gather**

★★★ **判定「`3Ψ` 比 `2Ψ` 多的是什么」时，必须先把 `2Ψ` 拆开**（★ 不拆就判错）。
ZeRO §7.1 逐字（`specs.ZERO_RS_AG_QUOTE`，载体 = ar5iv `1910.02054` 正文）：

> `Both reduce-scatter and all-gather are implemented using a pipelined approach`

⇒ **baseline 的 all-reduce 是一个两步过程**：
`reduce-scatter(Ψ)` **+** `all-gather(Ψ)` ⇒ **每步 `2Ψ`**（§7.1 逐字：
`the standard DP incurs 2Ψ data movement during each training step.`）。

★★★ **关键推论**：**baseline 里【已经有一次 all-gather】**（把归约好的梯度收回来）。
⇒ `P_os+g+p` 的 `3Ψ` = `reduce-scatter(Ψ)` + **前向** all-gather(`Ψ`) + **反向** all-gather(`Ψ`)
⇒ **它相对基线的增量恰好是「多一次 all-gather（参数）」**。

### 1.2 ★★★ 三个阶段的通信量

| 阶段 | 每步通信量 | 相对 baseline | 逐字依据 |
|---|---|---|---|
| **baseline DP** | **`2Ψ`** | —— | §7.1 `the standard DP incurs 2Ψ …` |
| **`P_os`** | **`2Ψ`（不增）** | 不变 | §7 开头 `incurs no additional communication using P_os and P_g` |
| **`P_g`** | **`2Ψ`（不增）** | 不变 | 同上（论文把 `P_g` 与 `P_os` 合称） |
| **`P_os+g`** | **`2Ψ`** | **完全相同** | §7.2 `exactly the same as the baseline DP.` |
| **`P_os+g+p`** | **`3Ψ`** | **`1.5x`** | §7.3 `which is 1.5x compared to the baseline.` |

★★★ **`P_os+g` 为什么「完全相同」**：它把 all-reduce **换成** `reduce-scatter` + `all-gather`
—— 两件事**总和还是 `2Ψ`**（§7.2 逐字：`Ψ+Ψ=2Ψ, exactly the same as the baseline DP.`）。
★ **`P_os+g+p` 为什么多一次**：参数要**按需 all-gather** ⇒ **前向摊一次、反向又一次**
（§7.3 逐字：`this all-gather needs to happen once again for the backward propagation`）。

### 1.3 ★★★ 三条出处纪律（本文的骨头）

★★★ **① `P_os` 的「不增通信」以 §7 开头的总结句为准。**
逐字是（`specs.ZERO_NO_EXTRA_COMM_QUOTE`）：`incurs no additional communication using`
—— ★★ **§1 列表的 `same communication volume as DP` 可作【同向互证】**
（`specs.ZERO_S1_SAME_COMM_QUOTE`，`P_os` 与 `P_os+g` 在 §1 里各写了一次）⇒
**不得写成「只【能】引 §7 开头这一句」**。
★★★ **原文【没有】 `P_os` 单独的通信量等式** ⇒ **不得自己造一个 `P_os = 2Ψ`
并标成「论文式(N)」**（那是**编来源 / 编机制从句**）。
★ 本文对 `P_os` 的陈述**只走「与 baseline 相同的结构」**（`ZERO_RS_AG_QUOTE` 是它的依据）。

★★★ **② baseline 的逐字定义**：`the standard DP incurs 2Ψ data movement during each training step.`（§7.1）。

★★★ **③ `K` 不是通信量。** `K = 12`（`specs.ZERO_K_MEMORY_MULTIPLE`）是**优化器状态的内存乘子**
—— 它属于**显存**那一侧的账（`01` 用），★ **把它写进任何通信量等式都是口径混淆**
（`code/` 的 `D.8` 用**源码级线索**钉住「通信模型里一个 `K` 都没有」）。

### 1.4 ★★★ 一条必须写明的【记号桥】（不是订正）

★ `10-foundations/performance-models/02-amdahl-and-scaling.md` 里的 `S` 是**梯度字节**，
而 ZeRO 的 `Ψ` 是**参数量** —— 梯度是 **fp16**、每参数 **`2` 字节** ⇒
★★★ **`S = 2Ψ`（字节）** ⇒ **该文的 `2S → 3S`【就是】论文的 `2Ψ → 3Ψ`**。

★ 为什么它成立（`D.7` 把这条算术钉住）：`S`（字节）`= 2Ψ`；
该文的 `mS`（字节）`= m × 2 × Ψ`；而 ZeRO 的 `mΨ`（**元素**，每元素 `2` 字节）
`= m × 2 × Ψ` 字节 ⇒ **同一个量** ⇒ ★ **倍数 `m` 是【共用的】**（`2` 与 `3`）。
★★★ **读者最容易卡住的就是这里**：两个文档用了两个不同的字母（`S` 与 `Ψ`）、
两个不同的单位（字节与元素），**但 `2 → 3` 这一步是同一个故事**。
★★★ **那句话【是对的】，一个字都不许改**（见 `§3.3`）。

### 1.5 ★★ 本机实测输出（`code/02_communication.py`）

| 判据 | 本机实测 | 它说明什么 |
|---|---|---|
| `A.1` | `P_os`/`P_os+g` = `1 + 1 = 2Ψ`；`P_os+g+p` = `1 + 2 = 3Ψ` | **三个 ZeRO 阶段**逐档对上 §7 开头 / §1 / §7.2 / §7.3 的逐字 |
| `A.2` | `P_os+g+p` = `1 + 2 = 3Ψ` == `2 × 1.5` | `2` 来自 §7.1、`1.5` 来自 §7.3（**两处各自独立**） |
| `B.1` | `P_os`/`P_g`/`P_os+g` 都 `= 2Ψ`；`P_os+g+p` = `3Ψ` | **三格不是一格** |
| `B.2` | `3Ψ ÷ 2Ψ = 1.5000` == `1.5` | §7.3 逐字 |
| `C.1` | 错法「把 `3Ψ` 套到 `P_os` 上」⇒ 与 baseline `2Ψ` 的矛盾量 = `1` | **「记错 ZeRO 阶段」必然失败**（不报错、只是数错） |
| `C.1g` | 换回**对法**（`P_os` 正确 = `2Ψ`）⇒ 矛盾量归零 | `C.1` 的不等式**不再成立**（对「记哪个 ZeRO 阶段」敏感） |

★★★ **`A.1`/`A.2` 为什么不是自证**：被测走的是 **reduce-scatter + all-gather 的【结构】**，
参照走的是 **论文的【逐字数字】**（`2` 与 `1.5`）⇒ **两条来源不同的路径**。
★ 而 `A.1g`/`C.1g` 证明这两条判据**能失败**（换掉「记哪个阶段」就变）。

## 2. 执行映射

### 2.1 本机怎么算这件事

`code/02_communication.py` 把通信量写成一个**按阶段索引的模型**（**单位 Ψ**）：

| 角色 | 是什么 | 依据 |
|---|---|---|
| **被测** | `comm_units(stage) = 1 + param_all_gathers(stage)` —— **reduce-scatter(1) + 参数 all-gather 次数** | `ZERO_RS_AG_QUOTE`（baseline 的结构） |
| **参照** | §7.1 的 `2`（`ZERO_BASELINE_MULTIPLE`）与 §7.3 的 `1.5`（`ZERO_1_5X_VERBATIM`） | **两条逐字出处** |

★★★ **这是两条【来源不同】的路径**：一条走**结构**（1 次 reduce-scatter + 若干 all-gather），
一条走**论文逐字**（`2` 与 `1.5`）⇒ 两者相等**不是恒真**（`A.1`/`A.2`）。
★ 而且 `A.1g`/`C.1g` 用「**记错 ZeRO 阶段**」当守卫：把 `P_os+g` 当成 `P_os+g+p` ⇒ `3Ψ ≠ 2Ψ` ⇒ 判据**会失败**。

### 2.2 ★★ 覆盖边界：本程序【不】做什么

★★★ **必须写明**（否则会让人以为它验证了更多东西）：
- 本程序验证的是**「每个阶段搬几个 `Ψ`」**（一个**算术**结论），
  **不验证**「真实集群上通信要多久」—— 那要带宽与延迟，**本机没有**；
- 它**不重推** `V(N) = 2(N−1)/N × S` 之类的**算法复杂度**（那是 `110-communication` 的正文）；
- ★★ 它的 `A.1`/`A.2` 守的是「**阶段 ↔ 通信量**」的对应（含 `A.1g`/`C.1g` 两条守卫），
  **不守**「真实 `all-reduce` 的实现对不对」。

### 2.3 一条必须写明的限定语

> ★★ 本机**无 GPU、无多卡、无 NCCL、无真实网络**；本程序算的是**字节数/倍数**，
> **没有一次真实通信**。★ 强制限定语：**「不是 NCCL，也不是真机」**。
> ★ 本程序的数字是**推导值 + 引文**（通信量是算术，不是本机测出来的性能）。

## 3. 实现与演进

### 3.1 ★★★ 这一节是 `P24` 的正文：**把某阶段的账算到整体头上**

★★★ **本文最要紧的一条**：**一个量只对某一阶段成立，却被写成对整体成立。**

- **现场**：`10-foundations/performance-models/02-amdahl-and-scaling.md:221` 写
  「ZeRO 三阶段把通信量从 `2S` 抬到接近 `3S` 以上（多一次 all-gather 参数）」。
- ★★ **逐条重判（这一句【基本是对的】）**：
  · **主语**：把「ZeRO 三阶段」读成**三阶段累计启用的那个配置**（= `P_os+g+p`）时，`3S` **对**；
  · **程度**：原文是**恰好** `1.5×`，「**接近**…**以上**」**略有夸张**（**措辞瑕疵**）；
  · **成因**：「多一次 all-gather（参数）」**是对的**（因为 baseline 已含一次，见 `§1.1`）。
- ⇒ ★★★ **修改动作降级为「只修措辞、不改事实」** —— 那是 `Chunk 7` 的事。
  ★★ **不得**按早期计划的写法去「订正」它：那会把**对的改成错的**。

★★ **为什么会判错（这是本文最有价值的一条教训）**：判「`3Ψ` 比 `2Ψ` 多的是什么」时，
只看了**增量的两端**，**没看 `2Ψ` 由什么组成**。
⇒ ★ **纪律（`P30`）**：凡出现「**A 比 B 多/少 X**」的判定，**必须把 B 的定义也引出来**，
并**逐项列出 B 的组成**（本例：B = reduce-scatter + all-gather）。

### 3.2 与相邻主题的分工

| 话题 | 在哪 |
|---|---|
| **每个阶段的通信量**（`Ψ` 的倍数） | **本文** |
| **集合通信的算法与复杂度**（ring / tree、`V(N)`、阶段划分） | `10-foundations/performance-models/02:209,428`、`110-communication/collectives`） |
| **拓扑受限的集合通信**（bisection 下界、分层） | `20-gpu-architecture/interconnect/03-collectives-on-topology.md` |
| **链路带宽的三种口径**与 NVLink / IB 的规格值 | `20-gpu-architecture/interconnect/01-bandwidth-conventions.md` |
| **通信-计算重叠的机制** | `110-communication/overlap/`（`overlap`） |
| **显存的账**（各阶段每卡几个字节） | [01-three-stages](01-three-stages.md) |
| **FSDP 的策略与分片后的正确性** | [03-fsdp-and-correctness](03-fsdp-and-correctness.md) |

### 3.3 ★★★ 为什么 `:221` 那一句是【基本对的】（一个 `P24` 的正例）

★★ **本文要替它讲清楚**，因为它是「**订正会把对的改成错的**」这类事故的**现场**：

| `:221` 的说法 | 判定 | 依据 |
|---|---|---|
| 「ZeRO 三阶段把通信量从 `2S` 抬到 `3S`」 | ★★ **对**（读成「三阶段累计启用的那个配置」= `P_os+g+p`） | `P_os+g+p` = `3Ψ`（§7.3） |
| 「多一次 all-gather 参数」 | ★★★ **对** | baseline 的 `2Ψ` **已含一次 all-gather**（§1.1）⇒ 增量**恰好**是多一次 |
| 「接近 … 以上」 | ★ **措辞瑕疵** | 原文是**恰好** `1.5×`（`ZERO_POGP_1_5X_QUOTE`） |
| `S` 与 `Ψ` | ★★ **不是口径混淆** | `S = 2Ψ`（字节）⇒ `2S → 3S` **就是** `2Ψ → 3Ψ`（§1.4） |

★★★ **一条纪律（`P30`）**：**修一句话之前，先把它引的 baseline 也拆开**。
本例里「多一次 all-gather 对不对」这个问题，**恰恰要拿 baseline 的定义来判** ——
baseline 里若已含一次 all-gather，那么「多一次」**就是对的**。
⇒ ★★ **引一段增量推导时，必须同时引它的 baseline**（v1/v2 只看了 `2Ψ` 与 `3Ψ` 两端）。

★★ **落地动作**：`Chunk 7` 对 `:221` 的改动**只修措辞**（把「接近…以上」改成「恰好 `1.5×`」），
**事实一个字不动**。★ 且 `:221` 的括注与 `code/10_amdahl_scaling.py:611` 那句
**都是对的**、**都不用改**（早期计划里的「订正」指令已**撤回**）。

## 4. 踩坑与失败模式

### 4.1 ★★★ 把 `3Ψ` 套到 `P_os` 上（「记错阶段」）

- **现象**：说「ZeRO 的通信量是 `3Ψ`」，让人以为 `P_os` 就变成了 `3Ψ`；
- **成因**：**`3Ψ` 只属于 `P_os+g+p`** —— `P_os`/`P_os+g` 是 `2Ψ`（与 baseline 相同）；
- **本机实测**（`code/` 的 `C.1`）：把 `3Ψ` 套到 `P_os` 上 ⇒ 算得 `3Ψ`，与 baseline `2Ψ` 的
  **矛盾量 = `1`**（`|3Ψ − 2Ψ|`）⇒ ★★ **「记错 ZeRO 阶段」必然失败**；★ `C.1g` 守卫把它真的跑了
  一遍：换回**对法**（`P_os` 正确 = `2Ψ`）⇒ 矛盾量归零 ⇒ **判据失败** ✓；
- **规避**：★★★ **报通信量时必须写明是哪个阶段** ——「不增 / `2Ψ` / `3Ψ`」是三格，不是一格。

### 4.2 ★★★ 给 `P_os` 编一个「论文式(N)」等式

- **现象**：写「`P_os` 每步通信量 = `2Ψ`」并标成「论文式」；
- **成因**：★★ **原文【没有】 `P_os` 单独的通信量等式** —— 只有 §7 开头的**总结句**
  （`incurs no additional communication using`）；
- **规避**：★ **`P_os` 的「不增通信」只引那一句**；本文对它的陈述走 **baseline 的结构**
  （reduce-scatter + all-gather），**不造等式**。

### 4.3 ★★ 把 `K` 当成通信量

- **现象**：把 `K = 12` 写进某个通信量式子；
- **成因**：`K` 是**优化器状态的内存乘子**（属于显存）—— 通信量只按 `Ψ` 记；
- **本机实测**（`code/` 的 `D.8`）：**源码级线索** —— `comm_units()` 与 `param_all_gathers()`
  的源码里**都没有** `ZERO_K_MEMORY_MULTIPLE`；
- **规避**：★★ **通信量只与「搬了几次梯度/参数」有关，与「优化器状态有几个字节」无关**。

### 4.4 ★★ 「多了 50%」与「是原本的 1.5 倍」

- **现象**：把 §7.3 的 `1.5x` 与 §1 列表的 `50%` 当成两个不同的结论；
- **成因**：★ 它们是**同一件事的两种口径** —— `1.5x` 是**倍数**、`50%` 是**增量占比**
  （`ZERO_POGP_1_5X_QUOTE` 与 `ZERO_S1_50PCT_QUOTE`）；
- **规避**：★★ **报「增了多少」时必须写明是哪一种**（本库的「同一件事的两种口径」，`P28` 家族）。

## 5. 判据表

| 可机械判定的条件 | 结论 |
|---|---|
| 报 baseline DP 的通信量 | ★★ **`2Ψ`**（§7.1 逐字）—— ★ **它已含一次 all-gather**（`A.1`/`C.1`） |
| 报 `P_os` 的通信量 | ★★★ **不增**（= baseline）—— ★ **只引 §7 开头的总结句**，**原文没有它单独的等式**（`D.4`） |
| 报 `P_g` 的通信量 | ★★ **不增**（与 `P_os` 同；论文把它与 `P_os` 合称 `P_os+g`）（`B.1`） |
| 报 `P_os+g` 的通信量 | ★★★ **`2Ψ`** = **与 baseline 完全相同**（§7.2 逐字 `exactly the same as the baseline DP.`）（`A.1`） |
| 报 `P_os+g+p` 的通信量 | ★★★ **`3Ψ`** = **`1.5x`**（§7.3 逐字）（`A.2`/`B.2`） |
| 把 `3Ψ` 套到 `P_os` 上（**错法**） | ★★★ **与 baseline `2Ψ` 直接矛盾**：矛盾量 = `1` ⇒ 判据**成立**（`C.1`）；换回对法 ⇒ 归零 ⇒ `C.1` 失败（`C.1g` 守卫） |
| 报 `P_os+g+p` 的通信量「比 baseline 多的是什么」 | ★★ **多一次 all-gather（参数）** —— 因为 baseline 的 `2Ψ` 里**已含一次**（`§1.1`/`A.2`） |
| 说「`P_os` 每步 = `2Ψ`，论文式(N)」 | ★★★ **不行**：原文**没有这个等式**（`D.4`） |
| 报「增了多少」 | ★★ 写清是 **`1.5x`（倍数）还是 `50%`（增量占比）**（`D.6`） |
| 把 `K = 12` 写进通信量 | ★★★ **口径混淆**：`K` 是**内存乘子**，不是通信量（`D.8`） |
| 引 `performance-models/02` 的 `2S → 3S` | ★★ **`S = 2Ψ`（字节）** ⇒ `2S → 3S` **就是** `2Ψ → 3Ψ`（`D.7`）；★★★ **那句话是对的，不许改** |

## 6. 小结与延伸

**一句话**：**`P_os` 与 `P_g` 不增通信、`P_os+g` 是 `2Ψ`（= baseline）、
只有 `P_os+g+p` 是 `3Ψ`（`1.5x`）** —— ★★★ **三格不是一格，「记错 ZeRO 阶段」必然失败**。

**三条可独立复用的结论**：

1. ★★★ **`2Ψ` 里已含一次 all-gather** ⇒ `3Ψ` 的增量**恰好是「多一次 all-gather（参数）」**
   —— 判增量归属时**必须拆开 baseline**（`§1.1`/`§3.1`）；
2. ★★★ **`P_os` 单独的通信量等式原文【没有】** ⇒ **只引 §7 开头的总结句**，**不造等式**（`D.4`）；
3. ★★★ **记号桥**：`performance-models/02` 的 `S = 2Ψ`（字节）⇒ 它的 `2S → 3S`
   **就是**论文的 `2Ψ → 3Ψ`（`D.7`）；★★★ **那句话是对的，一个字都不许改**。

**延伸**：

- ★★★ **显存的账**（各阶段每卡几个字节、「分片不是除法」）→ [01-three-stages](01-three-stages.md)；
- ★★ **FSDP 的策略与分片后的正确性** → [03-fsdp-and-correctness](03-fsdp-and-correctness.md)；
- **本域不写、只回链的正文**：
  · **集合通信的算法与复杂度**（`V(N) = 2(N−1)/N × S`、阶段划分）→
    `10-foundations/performance-models/02:209,428`、`110-communication/collectives`（；
  · **拓扑受限的集合通信** → `20-gpu-architecture/interconnect/03`；
    **链路带宽的三种口径** → 同域 `interconnect/01`；
  · **通信-计算重叠的机制** → `110-communication/overlap/`（`overlap`，。

## 参考

- Rajbhandari et al., *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models*,
  arXiv:1910.02054 —— **[C] 论文**，**同行评审：未确认**（★ arXiv abs 页无 `Comments` /
  `journal-ref` 字段 ⇒ 本库不推断；计划 §C.8 的汇总列即「未确认」）。
  ★ 引文载体 = `https://ar5iv.labs.arxiv.org/html/1910.02054` 正文，抓取日 **2026-09-23**。
  ★★ **载体纪律**：ar5iv 把变量与公式包在 `<math>…</math>` 里 ⇒
  跨标记的**整句**在原始字节上 `count()` 得 **0**；本文因此对跨标记的整句
  **另登记它的不跨标记载体片段**（`ZERO_BASELINE_HEAD` / `ZERO_BASELINE_TAIL` 等，
  各计 **1** 次）。★★★ **别把 0 次读成「原文没有这句」**（本库记的**假否定**）。
  ★ 引到的逐字段落：§7 开头（两句完整的）· §7.1（baseline 与 reduce-scatter/all-gather）·
  §7.2（`P_os+g` 的 `Ψ+Ψ=2Ψ`）· §7.3（`P_os+g+p` 的 `3Ψ = 1.5x`）· §1 的列表项。
- 本仓库 [01-three-stages](01-three-stages.md)（显存的账）、
  [03-fsdp-and-correctness](03-fsdp-and-correctness.md)（FSDP 的策略与正确性）、
  `10-foundations/performance-models/02-amdahl-and-scaling.md:221`（**`P24` 的现场** ——
  ★★ 本文判它**基本是对的**，只降级它的措辞）、
  `100-parallelism/data-parallel/01-gradient-averaging.md`（DP 的通讯账）。

## 编译与运行

```sh
cd code && make run                 # 三个编号程序，任一断言失败即非零退出
python3 02_communication.py         # 只跑本文的配套程序
make 02_communication               # ★★ 与上一行等价（都先走 check-env）
```

★ 本机的运行条件：`python3 3.13.5` / `numpy 2.4.4`，macOS arm64，
**无 GPU、无多卡、无 NCCL、无真实网络**。
★ 本文的一切数字是**推导值 + 论文引文**（通信量是算术，**不是**本机测出来的性能）。
★★ 强制限定语：**不是 NCCL，也不是真机** —— 本文验证的是**切分的代数**
（每个阶段搬几次、几倍），**不得**支撑关于真实集群性能的结论。

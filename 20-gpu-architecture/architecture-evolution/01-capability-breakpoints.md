---
title: 能力断点：同一份 kernel 为什么这一代能编译、下一代不能
slug: capability-breakpoints
domain: 20-gpu-architecture
topic: architecture-evolution
level: 3
difficulty: advanced
prerequisites:
  - 20-gpu-architecture/execution-model/01-sm-internals
  - 20-gpu-architecture/memory-system/01-hierarchy-overview
tags: [capability-breakpoint, ptx-isa, cp-async, independent-thread-scheduling, tma, tmem, sm-90a]
related:
  - 20-gpu-architecture/architecture-evolution/02-scaling-trends
  - 20-gpu-architecture/architecture-evolution/03-invariants
  - 10-foundations/performance-models/01-roofline-model
code:
  - code/01_capability_breakpoints.py
status: stable
updated: 2026-09-15
---

# 能力断点：同一份 kernel 为什么这一代能编译、下一代不能

> 硬件换代时，有些变化不是「更快」，而是「原来那件事现在表达不出来、或者原来表达不出来的现在能表达了」。这类变化叫**能力断点**——它落在 PTX 的 target 版本上，判据是 `sm_XX`，不是「哪一代产品先搭载」。

> **全文的数字口径**：本文的数字分两类——**规格值**（直接引自 NVIDIA 官方文档，见脚注与参考）
> 与**推导值**（由 `code/specs.py` 中的规格常量代入 §1 的公式算出）。**本文无实测数据。**
> 要复核，运行 `code/01_capability_breakpoints.py`，它会打印出本文引用的每一个数。

## 0. 背景与动机

一个分块 GEMM 的主循环，同一个算法，四代硬件上有四种写法。这四种写法之间的差别不是常量因子，是**能不能写出来**。

```text
Volta  (sm_70) ：mma.sync + 寄存器中转的双缓冲
Ampere (sm_80) ：cp.async 把搬运从中转寄存器里拿掉
Hopper (sm_90) ：TMA 描述符 + wgmma（整个 warpgroup 一起发射）
               —— 而且必须编到 sm_90a，编到 sm_90 只有一半的指令可用
Blackwell(sm_100a)：tcgen05，累加器搬进 TMEM，MMA 由单线程发起
```

四种写法里没有一种能原样搬到下一代。原因各不相同：

- **sm_70 上写不出 `cp.async`**：那条指令在这一代不存在。你只能写 `LDG` + `STS`，让数据在寄存器里过一道。
- **sm_80 上写不出 `wgmma`**：warpgroup 这个概念不存在——`mma.sync` 的执行单元是 warp（32 线程），而 `wgmma` 是 128 线程一起发射一条矩阵指令。**这不是参数不同，是执行单元不同。**
- **sm_90 上写不出 `tcgen05`**：TMEM 不存在，累加器只能住在寄存器里。而 `tcgen05` 的累加器**不能**住在寄存器里。
- **反方向更硬**：为 `sm_90a` 编出的 cubin，**跑不了 `sm_90`**。`sm_90a` 是「accelerated」变体，只在那个精确架构上可用。

最后一条最容易被低估。它不是「新特性用不了」，而是「用了新特性的二进制直接不加载」。同一个 Hopper 家族里，`sm_90` 与 `sm_90a` 是两个不兼容的目标。

所以「这套 kernel 能不能在新卡上跑」这个问题的答案，不能靠读规格表上的 TFLOPS 得到。它要靠一条更朴素的判据：**这套代码用到的最新的那条指令，是从哪个 target 起可用的？** 本文把这条判据做成一张表，并把其中两个断点推到「为什么」（§1），再把它们落到 SM 的具体部件上（§2）。

关于本主题的分工：逐代 SM 内部规格（寄存器预算、占用率约束）在 [execution-model/01-sm-internals](../execution-model/01-sm-internals.md)，逐代存储层次在 [memory-system/01-hierarchy-overview](../memory-system/01-hierarchy-overview.md)——**本文只讲「能不能表达」，不回链重复那两处的规格**。

## 1. 原理

### 第一步：断点的判据是 target，不是产品

判定一个断点落在哪一代，唯一可靠的依据是 **PTX ISA 各指令的 Target ISA Notes**——那一小节写明这条指令从哪个 `sm_` 版本起可用。用「哪一代产品先搭载」去判断会引入产品线的噪声（下面 FP8 那一格就是例子）。

`code/01_capability_breakpoints.py` §1 打印的完整表：

| 特性 | 首次引入 | target | 改变了什么 |
|---|---|---|---|
| Tensor Core / `mma` | Volta | `sm_70` | 矩阵乘变成一条指令，fragment 布局成为程序员的显式责任 |
| Independent Thread Scheduling | Volta | `sm_70` | warp 内不再 lockstep，无同步的 warp 内规约从此是 UB |
| TF32 | Ampere | `sm_80` | FP32 数据零改代码走 tensor 路径（代价是尾数 23→10 位） |
| **BF16** | **Ampere** | `sm_80` | 与 FP32 同指数范围，消除 loss scaling |
| 2:4 结构化稀疏 / `mma.sp` | Ampere | `sm_80` | 稀疏成为一等公民（矩阵 + metadata 两个操作数） |
| `cp.async` | Ampere | `sm_80` | global→shared 不经寄存器 |
| FP8 | Hopper | `sm_90` | 量化从离线步骤变成运行时机制 |
| TMA / `cp.async.bulk` | Hopper | `sm_90` | 一个线程可发起整块多维搬运 |
| `wgmma` | Hopper | **`sm_90a`** | MMA 执行单元从 warp 变成 warpgroup（128 线程）且异步 |
| Thread Block Cluster / DSMEM | Hopper | `sm_90` | 共享内存首次可跨 SM 直接寻址 |
| TMEM | Blackwell | **`sm_100a`** | 累加器不再占寄存器 |
| `tcgen05` | Blackwell | **`sm_100a`** | MMA 改为 CTA 级、单线程发起 |
| FP4 | Blackwell | `sm_100` | 必须配 block scaling，否则动态范围不够 |

两条必须记清的修正，都是这份表打印出来的：

1. **BF16 第一次出现在 Ampere (`sm_80`)，不是 Turing。** Turing 的 Tensor Core 只加了 INT8 / INT4。把 BF16 归给 Turing 是本主题最常见的记错——因为 Turing 也「有 tensor core」，很容易顺手把「16 位的 tensor 精度」也算给它。
2. **FP8 在数据中心线是 Hopper (`sm_90`)，消费级线（Ada，CC 8.9）更早具备。** 凡说「某代 FP8」都要带上产品线。

第三条不是修正，是一条独立的坑：`sm_90a` / `sm_100a` 属于**加速变体**。表里属于加速变体的特性是 `wgmma`、`tmem`、`tcgen05` 三项。它们只在**该精确架构**上可用，不像 `sm_90` 那样向前兼容同一家族。

### 第二步：多步推导一——从 `cp.async` 的有无，推出流水线深度与占用率的循环依赖

问题：Ampere 之前能不能做「访存与计算重叠」的双缓冲流水线？

**答案是能，但深度只能在浅档位上停留。** 下面把这句话算成一个数。

*（i）两条搬运路径。*

- **寄存器中转**（`sm_80` 之前唯一的选择）：`LDG` 把数据写进寄存器，再由 `STS` 从寄存器写进共享内存。数据在「`LDG` 完成」到「`STS` 发射」之间必须住在寄存器里，而这个寄存器的生命期跨过整段全局延迟。
- **`cp.async`**（`sm_80` 起）：数据从 global 直达 shared，不经过寄存器回写。中转的落脚点从**每线程的寄存器**变成**每 block 的共享内存**。

*（ii）建账本。* 取每线程每级的 tile 为 **16 字节**（一个 `float4`，也正是 `cp.async` 的规范粒度），折合 **4 个 32-bit 寄存器**。记流水线深度为 `D`，计算本身需要的寄存器为 `R0 = 40`（经验区间的中值）。寄存器分配粒度是 8 个一档，实际占用按档上取整。

*（iii）两栏算式。*

```text
寄存器路径：r_reg(D) = R0 + 4·D   →  按 8 上取整  →  warps = floor(65536 / (32·r))

cp.async 路径：r_async = R0（中转不进寄存器）→ warps = floor(65536/(32·40)) = 51（恒定）
    但它的共享内存账本是 D·16 B × 256 线程 = D·4 KB 每 block，
    而 A100 每 SM 只有 164 KB 共享内存，于是再取一次较小值：
        warps = min(51, floor(164 / (D·4)) × 8, 64)
```

程序 §2 打印的结果：

| 流水线深度 `D` | 寄存器路径 `r` | 上取整后 | 寄存器路径 warp 数 | cp.async warp 数 | 谁赢 |
|---|---|---|---|---|---|
| 0 | 40 | 40 | 51 | 51 | 持平（无流水线） |
| 1 | 44 | 48 | 42 | 51 | cp.async |
| 2 | 48 | 48 | 42 | 51 | cp.async |
| 4 | 56 | 56 | 36 | 51 | cp.async |
| 6 | 64 | 64 | 32 | 48 | cp.async |
| 8 | 72 | 72 | 28 | 40 | cp.async |
| 12 | 88 | 88 | 23 | 24 | cp.async |
| 16 | 104 | 104 | **19** | **16** | **寄存器** |

*（iv）结论一——寄存器路径上，深度是占用率的减函数。*（**占用率** = 实际常驻 warp 数与架构上限之比；分母由目标 CC 决定，见 [03-invariants](03-invariants.md) §1 第五步。）每加深一级，固定扣掉每线程 4 个寄存器；在 `r` 跨过 8 的档位边界时整档掉 warp 数。`D` 从 0 到 16，常驻 warp 从 **51 掉到 19（2.68 倍）**。

而占用率正是用来隐藏延迟的那个资源。于是出现一个**循环依赖**：

> **加深流水线的目的是藏住延迟，可它消耗的恰好是藏延迟用的那份预算。**

这就是「Ampere 之前能重叠、但只能在浅流水线上重叠」的量化形态。

*（v）结论二——`cp.async` 把这条耦合搬到了共享内存，而不是取消它。* 在 `D ≤ 8` 时它白赚：寄存器一侧恒定 51 个 warp，共享内存还没成为约束。但 `D` 继续加深，**共享内存接替寄存器成为新瓶颈**——`D = 16` 时共享内存只允许 16 个 warp，**反而低于寄存器路径的 19 个**。两条曲线在 `D ∈ [12, 16]` 之间交叉。

所以 `cp.async` 兑现的不是「从不能重叠变成能重叠」，而是**把流水线深度的代价从一个预算挪到另一个预算**：从每线程的寄存器（65536 个，按线程切分、直接决定 warp 数）挪到每 block 的共享内存（164 KB，按 block 切分）。**两个预算都被解耦了，但没有被取消。**

### 第三步：多步推导二——从 Independent Thread Scheduling 推出 `__syncwarp()` 的必要性

问题：为什么 7.0 之前的「无同步 warp 内规约」能用，之后必须加 `__syncwarp()`？

*（i）7.0 之前，一个 warp 的 32 个 lane 共享一个 PC。* 硬件按 warp 取指、按 warp 发射，32 个 lane 在同一周期执行同一条指令的各自数据。只要没有分支发散，它们必然在同一时刻处于同一个程序点。

*（ii）这个「必然」正是无同步规约所依赖的前提。* 形如 `v += __shfl_down(v, 1)` 的规约，每一步都在读**邻居 lane 的寄存器**。在共享 PC 的模型下，邻居一定已经把上一步写完了——因为大家同步推进。于是这段代码在 7.0 之前是**可用的**，虽然它从未被 ISA 明文保证过。

*（iii）Volta 给每个线程配了独立的 PC 与调用栈。* 硬件仍然按 warp 发射，但一条发射出去的指令携带一个**任意的 active mask**——32 个 lane 里可以只有一部分处在同一个程序点。后果直接命中第（ii）步：邻居 lane 可能**还没走到**那条写寄存器的指令，也可能**已经越过去了**。读到的值不再有定义。

*（iv）因此程序必须显式声明「哪些 lane 要在这里会合」。* 这就是 `__syncwarp(mask)`：它在硬件里插入一个**重汇聚点**，强制 `mask` 指定的 lane 走到同一程序点再继续。`mask` 参数不是装饰。

*（v）证据在 API 的演化里——同步版本是后来补上的，而且带 mask。*

| 7.0 之前的形式 | Volta 起的形式 | 新增的那个参数 |
|---|---|---|
| `__shfl(v, srcLane)` | `__shfl_sync(mask, v, srcLane)` | mask：哪些 lane 参与 |
| `__ballot(pred)` | `__ballot_sync(mask, pred)` | mask：哪些 lane 参与 |
| `__any(pred)` / `__all(pred)` | `__any_sync` / `__all_sync(mask, pred)` | mask：哪些 lane 参与 |
| （无对应物） | `__syncwarp(mask)` | 显式的会合点 |
| （无对应物） | `__activemask()` | 读出当前活跃的 lane 集合 |

最后两行是重点：**7.0 之前根本没有查询活跃 mask 的手段**，因为那时「活跃集合」是一个不需要问的问题。这条 API 缺口本身就是断点的直接证据——**新硬件引入了一个旧程序无法表达的状态。**

结论：这不是「性能变了」，是「同一段源码在 7.0 之前有定义、之后无定义」。

### 第四步：逐代能力年表

把上面的表按代折叠（程序 §5 打印）：

| 代（年份） | 新增的断点 |
|---|---|
| P100（2016） | 无——**Pascal 是这套能力阶梯的零级** |
| V100（2017） | Tensor Core、Independent Thread Scheduling（2 项） |
| A100（2020） | TF32、BF16、2:4 稀疏、`cp.async`（4 项） |
| H100（2022） | FP8、TMA、`wgmma`、Cluster/DSMEM（4 项） |
| B200（2024） | TMEM、`tcgen05`、FP4（3 项） |

**断点不是均匀分布的。** Volta 是把 GPU 从「一堆标量核」变成「一台矩阵机」的那一代；Ampere 是把「流水线」变成可编程的那一代；Hopper 是把「SM 之间的墙」拆掉的那一代。

## 2. 执行映射

每个断点落在 SM 的一个具体部件上。这张表（程序 §4 打印）回答了「新能力是怎么长出来的」：

| 断点 | 落在哪个部件 | 代价 / 数量级 |
|---|---|---|
| Tensor Core / `mma` | 每 SM 的 Tensor Core 单元 + 寄存器文件里的 fragment | Volta 8 个/SM；Ampere 起 4 个/SM；B200 官方未披露 |
| Independent Thread Scheduling | **每线程 PC + 重汇聚栈** + 发射指令的 active mask 字段 | active mask 是每条发射指令携带的 32 位状态 |
| TF32 / BF16 | Tensor Core 的**输入端口模式**（不是新部件） | 同一端口换解码；精度 23→10 位 / 指数对齐 FP32 |
| `mma.sp`（2:4 稀疏） | Tensor Core 的操作数取回路径 + metadata 操作数 | peak 翻倍；代价是权重需重排 + 一份 metadata |
| `cp.async` | **LSU + 共享内存写端口**（绕过寄存器回写）+ 异步组计数器 | 搬运指令数减半；中转从每线程寄存器改到每 block 共享内存 |
| TMA / `cp.async.bulk` | 每 SM 的 **TMA 单元** + 共享内存端口；描述符住在 global | 一个线程发起整块多维搬运，寄存器占用 ≈ 常数 |
| `wgmma` | Tensor Core 的**发射路径**（warp → warpgroup 128 线程）+ 异步完成 | 累加器仍在寄存器：m64n256 每线程占 128 个 |
| Cluster / DSMEM | GPC 内的 **SM 间互联** + 共享内存地址映射（`mapa`） | 可移植簇大小上限 **8** 个 block；非可移植 16 |
| TMEM / `tcgen05` | **TMEM**（512 列 × 128 行 × 32-bit）+ 单线程发射路径 | TMEM **256 KB**，与寄存器文件同容量 |

三处值得单独看：

**TMEM 那一格。** `512 × 128 × 4 B = 262144 B = 256 KB`，与每 SM 的寄存器文件（65536 个 32-bit 寄存器 = 256 KB）**恰好同容量**。这不是巧合——它回答的正是 [execution-model/01-sm-internals](../execution-model/01-sm-internals.md) §3 算过的那个问题：Hopper 上一个 m64n256 的 FP32 累加器 tile 分摊到 128 个线程上，每线程单是累加器就要占 128 个寄存器，`65536 ÷ 128 = 512` 正好只装得下 4 个 warpgroup。Blackwell 的解法不是把寄存器文件做大，而是**另开一份**。（寄存器文件与常驻 warp 数十年未变这条线，正文在 [03-invariants](03-invariants.md)。）

> **TMEM 的口径要写清**：PTX 原文给出的 TMEM 容量口径是 **per CTA**（`sm_100a` / `sm_100f`），不是 per SM。256 KB 这个数在「一个 CTA 独占一个 SM」的常见配置下同时是每 CTA 与每 SM 的容量，但两个口径本身不是一回事——**这是本主题里又一处「同一个数字、两种说法」**。寄存器文件 256 KB 则是明确的 per SM。

**发射槽那一格。** 每 SM 每周期最多发射 4 条 warp 指令（4 个 warp scheduler），H100 有 132 个 SM ⇒ **全卡每周期 528 个发射槽**。断点改变的不是槽的数量，而是**这些槽能装什么**：一条 `mma` 顶掉几十条 `FFMA`。这是「同样 4 个槽、算力差一个数量级」的来源。

**重汇聚栈那一格。** Independent Thread Scheduling 的成本不在性能表上——它是一份**每线程的状态**（PC 与调用栈），硬件必须为每个线程维护它。这也解释了为什么 7.0 之前的 warp 级原语不带 mask：那时 mask 是一个恒定值，不需要表达。

## 3. 实现与演进

按代叙述，主线是**每一步都是对上一步暴露出的问题的回应**。

### P100（Pascal，2016）：零级

没有任何一项本文定义的断点。每 SM 64 个 FP32 lane，没有 Tensor Core。它唯一的角色是**基线**——用来证明上面那张表不是「每一代都加东西」的清单。

### V100（Volta，2017）：把标量核变成矩阵机

一次性引入两项，而且是**互相依赖的两项**。

Tensor Core 把矩阵乘变成一条 `mma` 指令。代价写在「fragment」这个词上：`mma` 的操作数不是普通数组，而是**分布在各 lane 寄存器里的一个切片**，哪一 lane 持有哪些元素由指令规定。于是「矩阵的哪一块在哪个寄存器里」从一个实现细节变成了**程序员的显式责任**。

Independent Thread Scheduling 给每个线程独立的 PC 与调用栈。它回应的是 Pascal 及更早的一个老问题：一个 warp 里两条分支路径必须串行执行（lane 之间 lockstep），发散代价高。代价是**旧的 warp 级原语失去了默认的正确性**，必须补上 `mask`（§1 第三步）。

### A100（Ampere，2020）：把流水线变成可编程的

四项，可以分成两组。

**精度组（TF32 / BF16）**：都不需要新部件，只是 Tensor Core 输入端口换了解码。TF32 的宣传语是「FP32 数据零改代码走 tensor 路径」——这句话是对的，但要补上代价：尾数从 23 位掉到 10 位。BF16 则把指数位宽直接对齐 FP32，范围等价而精度只有 8 位有效。**两者都是「拿精度换吞吐」，只是换的位置不同**（精度那一侧的推导在 [10-foundations/floating-point/01-ieee754-format](../../10-foundations/floating-point/01-ieee754-format.md)）。

**搬运组（`cp.async` / `mma.sp`）**：`cp.async` 的推导在 §1 第二步，结论是它把流水线深度的代价从寄存器预算挪到共享内存预算。`mma.sp` 则把 2:4 结构化稀疏变成一等公民——操作数从一个矩阵变成「矩阵 + metadata」两个，稀疏位置由 metadata 显式给出。

### H100（Hopper，2022）：拆掉 SM 之间的墙

四项，其中一项是这一代真正的结构性变化。

**Cluster / DSMEM**：簇内的 block 被共同调度到一个 GPC，可以互相读写对方的共享内存。**「共享」的范围第一次从 block 扩到了 cluster**——同一块片上存储现在能被一组 SM 共同可见，中间不再经过 L2。上限是可移植 8 个 block / 非可移植 16 个。

**TMA 与 `wgmma`**：把操作数准备与批量搬运从线程的指令流里剥离。TMA 由单个线程发起一大块多维搬运；`wgmma` 由整个 warpgroup 协作发射，且是异步的。注意 `wgmma` 的 target 是 **`sm_90a`** 而不是 `sm_90`——它是加速变体，二进制不向前兼容。

**FP8**：量化从离线步骤变成运行时机制。数据中心线的首次出现是 `sm_90`。

### B200（Blackwell，2024）：再加一级

三项，都围绕同一件事。

`tcgen05` 把 MMA 改成 **CTA 级、单线程发起**。这直接回应了 Hopper 留下的问题：`wgmma` 虽然由 warpgroup 发射，但累加器仍然住在寄存器文件里，而寄存器文件自 Volta 起就是 256 KB，没有增长的余地。

TMEM（每 SM 256 KB）是答案——**不是把寄存器文件做大，是另开一份**。累加器搬出去之后，寄存器预算不再被矩阵 tile 的宽度直接牵动。

FP4 则从第三个方向减压：同样的 tile 用更少的位表示，操作数占用的寄存器随之减少。它的代价写在「必须配 block scaling」上——4 位尾数的动态范围不足以独立表示一个张量，必须按块共享指数。

### 汇总：这条线在讲什么

五代下来，断点可以归成三类动作：

| 动作 | 代表 | 本质 |
|---|---|---|
| **加一个执行单元** | Tensor Core (`sm_70`) | 让一条指令做几十条指令的活 |
| **改一个数据通路** | ITS (`sm_70`)、`cp.async` (`sm_80`) | 改变数据在部件之间的走法 |
| **加一级存储 / 扩大可见范围** | Cluster/DSMEM (`sm_90`)、TMEM (`sm_100a`) | 把某个东西搬到更近、或让更多人看见 |

而 `sm_90a` / `sm_100a` 这类加速变体是第四类，也是最容易被低估的一类：**它把上面三类动作的受众从「所有同架构芯片」缩小到「恰好这一款」**。用到加速变体特性的代码，交付时多了一个必填项：目标型号。

## 4. 踩坑与失败模式

### 4.1 编译目标与实际硬件不一致

**现象**：本地编译一切正常（默认 `-arch=native` 或 `sm_90`），换一台机器运行时报 `no kernel image is available for execution on the device`；或者反过来，为了保险编成 `sm_70`，结果 `cp.async` 相关的 kernel 性能比预期低一大截。更隐蔽的一种：程序能跑，但某条分支永远没被走到——因为编译器为不支持的 target 静默退化成了标量实现。

**成因**：pipeline 上有两个独立的口径在起作用。其一是 **target 版本**：`cp.async` 需要 `sm_80`，`wgmma` 需要 `sm_90a`。其二是**变体**：`sm_90a` 与 `sm_90` 是两个不兼容的目标，**为父子关系里的哪一方编，决定二进制能不能加载**。一个 fatbin 里如果只嵌了 `sm_90a` 的 cubin，那么它在 `sm_90` 设备上既不能加载也不能 JIT——因为没有 `sm_90` 的 PTX 可回退。

**规避**：把交付物的 target 清单当成一份契约来维护，而不是交给 `-arch=native`。步骤是：先确定要用到的**最新那条指令**（查 PTX ISA 的 Target ISA Notes），由它反推最低 target；再决定是否需要嵌入多个 cubin 或 PTX（`-gencode arch=compute_90,code=sm_90 -gencode arch=compute_90a,code=sm_90a` 这类写法）；最后在目标硬件上跑一次加载验证。**不要把「本地能编译」当成「目标机能用」的证据**——这两件事之间隔着一个 fatbin 的加载规则。

### 4.2 在 sm_70+ 上写无同步的 warp 内规约

**现象**：一段从老代码库继承来的 warp 内规约（`__shfl_down` 若干级，然后 lane 0 写结果），在 Pascal 上一直正确，换到 Volta 之后**偶尔**出错：结果偏大、偏小，或者只在某些输入尺寸下错。用 `-O0` 编译时又常常是对的。

**成因**：§1 第三步的完整链条。Volta 之后 warp 内不再 lockstep，每个线程有自己的 PC，一条发射指令携带任意 active mask。规约的每一步都在读**邻居 lane 的寄存器**，而邻居是否已经写完那一步，在没有显式同步时**没有定义**。「偶尔错」是因为它依赖运行时的调度时序——而时序又受编译器优化等级、block 形状、数据规模影响。

**规避**：所有 `__shfl*` / `__ballot*` / `__any*` / `__all*` 都改用带 `_sync` 后缀的版本，并把 mask 显式写出来；在任何依赖 lane 间数据传递的位置插入 `__syncwarp()`。当 mask 确实需要「当前活跃的那些 lane」时，用 `__activemask()` 取——但要注意它返回的是一个**瞬时快照**，如果之后有分支改变了活跃集合，那个快照就失效了。**判据很简单：跨 lane 读寄存器，就必须先会合。**

### 4.3 把 TF32 当成「免费的加速」

**现象**：看到一个 FP32 矩阵乘的 kernel，只是加了一个编译开关（或某个库默认打开），吞吐涨了好几倍，精度指标（loss、相对误差）变化在噪声里，于是认定「TF32 是免费的」。直到某个对精度敏感的任务上，误差突然不可接受。

**成因**：TF32 是**端口模式**而不是新格式（§2 的表）。它在 tensor core 的输入端把 FP32 数据截成 10 位尾数，累加仍在 FP32。于是「零改代码」是真的，但**代价被藏在了数值格式里而不是 API 里**——源码中没有任何一处写着「这里精度降了」。10 位尾数对应大约 3 位十进制有效数字，与 FP32 的 7 位差两个数量级。

**规避**：把「用不用 TF32」当成一个显式的精度决策，而不是一个性能开关。收敛性敏感的任务（长序列累加、条件数大的矩阵、训练末期的微调）要显式关掉它；宽容的任务（推理、粗调阶段）可以打开。判断方法与 fp16 / bf16 是同一套：**看这个算子的误差会不会沿着迭代被放大**——如果会，TF32 的 10 位尾数就是一个真实的约束，而不是噪声。格式侧的完整口径在 [10-foundations/floating-point/01-ieee754-format](../../10-foundations/floating-point/01-ieee754-format.md)。

## 编译与运行

进入 `code/` 执行 `make run` 即可复现本文全部数字：

```bash
cd code
make run                            # 依次运行 01~03 三个程序
make 01_capability_breakpoints      # 只跑本文对应的程序（省略 .py 后缀）
```

配套程序是 **`code/01_capability_breakpoints.py`**（Python ≥ 3.10 + numpy ≥ 2.0，纯数值推导，不需要 GPU，也不编译任何 CUDA 代码）。规格常量集中在 **`code/specs.py`**——本主题的共享事实源，每条常量带一行 `# 来源：` 注释。程序第 6 节自带 14 项断言自检。

## 6. 小结与延伸

本文只有一条主线：**硬件换代时，有一类变化不是「更快」，而是「能不能表达」。** 它落在 PTX 的 target 版本上，判据是 `sm_XX`，与规格表上的 TFLOPS 无关。围绕这条主线，有三个可独立复用的中间结论：

1. **断点的判据是 target，不是产品。** 查 PTX ISA 的 Target ISA Notes。两处最容易记错的地方：BF16 是 Ampere (`sm_80`) 而不是 Turing；FP8 在数据中心线是 Hopper 而消费级更早。`sm_90a` / `sm_100a` 是加速变体，二进制不向前兼容。
2. **`cp.async` 解耦了两个预算，但没有取消它们。** 它把流水线深度的代价从「每线程寄存器」（直接决定 warp 数）搬到「每 block 共享内存」。浅流水线（`D ≤ 8`）上白赚，深到 `D = 16` 时共享内存反而成了新瓶颈——**那时候它比寄存器路径还差**。
3. **无同步的 warp 内规约在 7.0 之后是 UB。** 因为每个线程有了独立的 PC，跨 lane 读寄存器不再有隐含的顺序保证。`__syncwarp()` 的 `mask` 参数不是装饰，它是程序对「哪些 lane 必须会合」的显式声明。

下一篇 [02-scaling-trends](02-scaling-trends.md) 换一个维度：能力断点讲的是「能不能表达」，它讲的是「表达出来之后能不能跑到峰值」。那里会出现一个与本文共振的量化结论——**roofline 拐点十年里在 FP32 轴上原地打转，却在低精度轴上移动了 8.42 倍**，也就是「必须换精度而不是换算法」这句口号的数字来源。[03-invariants](03-invariants.md) 再把方向反过来：前面两篇讲的全是「变」，而有些数字十几年没动，知道哪些不动才知道哪些代码能活得久。

## 参考

- NVIDIA, *PTX ISA 9.4* —— **本文全部断点判定的依据**：各指令的 "Target ISA Notes" 小节写明它从哪个 `sm_` 版本起可用，`sm_90a` / `sm_100a` 作为加速变体的说明也出自这里。
- NVIDIA, *CUDA C++ Programming Guide 13.4* —— "Compute Capabilities" 章节（各 CC 的每 SM 资源）；"Warp Vote Functions" / "Warp Shuffle Functions" 章节（`_sync` 变体与 `__syncwarp` 的语义、mask 参数的含义）。
- NVIDIA, *CUDA C++ Programming Guide 11.8*（归档版）—— 唯一仍含 CC 6.0 / 7.0 规格表的官方文档，用于跨代对照。
- NVIDIA, *Hopper Architecture In-Depth*（官方博客）—— 线程块簇、DSMEM 的簇大小上限（可移植 8 / 非可移植 16）、TMA 与 `wgmma` 的原始说明。
- NVIDIA, *Blackwell Architecture Technical Brief V2.1* —— `tcgen05`、TMEM（512 列 × 128 行 × 32-bit = 256 KB）、FP4 与 block scaling。
- NVIDIA, *NVIDIA A100 Tensor Core GPU Architecture* 白皮书 —— `cp.async` 的引入背景与 TF32 / BF16 / 2:4 稀疏的原始说明。
- 本仓库 [20-gpu-architecture/execution-model/01-sm-internals](../execution-model/01-sm-internals.md) —— 寄存器预算、warpgroup 累加器占用的完整推算（正文在该文，本文只引用结论）。
- 本仓库 [20-gpu-architecture/memory-system/01-hierarchy-overview](../memory-system/01-hierarchy-overview.md) —— DSMEM 作为「扩大可控范围」这一步的意义（正文在该文，本文只引用）。

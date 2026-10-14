---
title: 量化 GEMM 的 kernel 设计：累加器怎么选，以及一个格式为什么需要一整个 kernel
slug: kernel-design
domain: 130-model-optimization
topic: quantization
level: 4
difficulty: advanced
prerequisites:
  - 130-model-optimization/quantization/15-w8a8
  - 130-model-optimization/quantization/16-w4a16-packing
tags: [marlin, machete, exllamav2, cutlass, fp16-accumulator, 2-4-sparsity, prepacking]
related:
  - 130-model-optimization/quantization/03-bf16-fp16-tf32
  - 10-foundations/floating-point/04-low-precision-formats
  - 40-kernels/gemm/04-tensor-core-and-async-pipeline
  - 130-model-optimization/sparsity/01-sparsity-granularity
code:
  - code/17_kernel_design.py
status: stable
updated: 2026-10-13
---

# 量化 GEMM 的 kernel 设计：累加器怎么选，以及一个格式为什么需要一整个 kernel

> 量化的收益**不在格式本身，在 kernel 里**。同一份 INT4 权重，naive 实现会比 FP16 更慢——直到有人为它写一个专用 kernel。这一篇讲那些 kernel 共同面对的三个决定：**累加器用什么宽度**（int32 是规范钉死的；fp16 累加有两个独立的陷阱：`65504` 的上限与 `[2048,4096)` 区间 ulp=2）、**权重怎么重排**（4-bit 没有 `ldmatrix`，所以必须 prepacking）、以及**稀疏能不能与量化叠乘**（理论加速都是 4×，**不叠成 16×**）。★ 本篇文档**没有 §5「实测数据」**——它的数字全是引用值（kernel 性能必须有 GPU）；但累加器那一层是本机真量的，收在 §1 与 §4。

## 0. 背景与动机

### 0.1 一个具体的对照场景

`15-w8a8` 与 `16-w4a16-packing` 分别讲了「累加器必须多宽」与「布局长什么样」。但这些都是**位级的事实**——它们**不会自己变成速度**。

一个具体的对照：把 Llama-2-7B 的权重压到 4 位，模型文件小了 4 倍，但**如果直接拿一个 fp16 GEMM 去算 4-bit 张量**，什么也不会发生（张量会被反量化回 fp16，然后走同一条 fp16 路径）。**速度来自「为这个格式写一个 kernel」。**

### 0.2 那一个 kernel 要做哪三件事

| # | 决定 | 本文的小节 |
|---|---|---|
| 1 | **累加器用什么宽度** | §1（这一层**本机真量过**） |
| 2 | **权重怎么重排**（才能被加载指令吃下） | §3.1 / §3.2 |
| 3 | **这块算力还能不能与稀疏叠乘** | §3.4 |

> ⚠ **本篇的 §5「实测数据」整节省略**（按本域的全域口径）：kernel 性能数字需要 GPU，
> 而本机 `torch.cuda.is_available()` 为 **False**——**一条 GPU 数字都不可能有**。
> **但累加器那一层是可以真量的**，那些数收在 §1 与 §4，不放进 §5。

## 1. 原理

### 第一步：累加器的三个选项，三个不同的代价

| 候选 | 值域 / 有效位 | 代价 | 用在哪 |
|---|---|---|---|
| **int32** | `±2^31`，**整数精确** | 4 字节/元素；**epilogue 必须先转 fp32** | 整数 GEMM（W8A8、W4A8） |
| **fp32** | `±3.4e38`，24 位 | 4 字节；浮点累加会丢低位 | 混合精度的「安全档」 |
| **fp16** | 上限 **65504**，11 位 | 2 字节；**两个独立的陷阱** | W4A16 / W8A16（权重低位、激活 fp16） |

**int32 那一条是规范钉死的**（PTX 里整数 `wmma.mma` 的 `.ctype` / `.dtype` 必须是 `.s32`，指令集里没有 int8 的 s16 累加变体）——细节见 `15-w8a8` §1 第二步。

**fp16 那一条不是「更省」而是「更险」**，因为它有两个**互相独立**的失败模式（§1 第二步）。

### 第二步：fp16 累加的两个陷阱（判据不同，必须分开写）

| # | 陷阱 | 判据 | 本机实测 |
|---|---|---|---|
| 1 | **加法丢失** | 只有 10 位显式尾数 ⇒ `[1024, 2048)` 的 ulp = **1**、`[2048, 4096)` 的 ulp = **2** | `float16(2048.0) + float16(1.0) == 2048.0` |
| 2 | **值域溢出** | 最大有限值 **65504** | `40000 + 40000 = inf`；而 `30000 + 30000 = 60000` 正常 |

**这两个不是同一个量**（一个是「增量被舍掉」，一个是「和超出值域」）。分开看它们的后果：

- **陷阱 1** 在**小增量、大基数**时发生：本机实测，纯 fp16 把 64 个 `1.0` 加到 `2048.0` 上，结果**还是 2048.0**——**一个都没加上去**。而 `2048 + 2` 给 2050（ulp 是 2，所以能走一格）。
- **陷阱 2** 在**两个大数相加**时发生：本机实测 `30000 + 30000 = 60000`（≤ 65504，正常），`40000 + 40000 = inf`（`80000 > 65504`）。

**同一条结论的另一面**：`15-w8a8` §1 第三步说过，int32 累加器必须在 epilogue **先转 fp32**——那里的理由是「fp16 只能精确表示 ≤ **2048** 的整数」。**那是第三个量**（整数可表示上界），与上面两条也不是一回事。

> ⚠ **判据不同的三个数：2048（可精确表示的整数上界）、ulp 1→2（加法分辨率）、65504（值域上限）——不得并列成「fp16 精度不行」的一句话。**

### 第三步：PTX 说什么，本机测什么

**PTX 规范明文**（引用值）：

> `.ctype` 与 `.dtype` 都是 `.f16` 时，"accumulation is performed with **at least half precision**"；
> "The accumulation order, rounding and handling of subnormal inputs is **unspecified**"。

两句话各有分量：

1. **「at least half precision」** ⇒ **指令层面 fp16 累加这个变体确实存在**。它不是一个「不能用」的东西，而是一个「可能只有半精度」的东西。
2. **「顺序 unspecified」** ⇒ **同一份 kernel 在不同硬件/不同 tile 切分下可以给出不同的结果**。这不是 bug，是规范允许的。

**本机测到的是什么**：本机 CPU 的 fp16 `torch.mm` **用 fp32 累加**（判据见 §4.1）。**这是 CPU 路径，不是 PTX 那条路径。**

> ★ **不得把本机 CPU 的行为外推到 GPU。** PTX 明文的那个变体本机**碰不到**——`torch` 在 CPU 上没有「可选 fp16 累加」这个开关。**「本机测到 fp16 mm 是 fp32 累加」与「GPU 上可以 fp16 累加」两句都对，而且不矛盾。**

### 第四步：更细的机制（引用值，本实现者未逐字核实）

有一篇形式化研究（**Fasi / Higham / Mikaitis / Pranesh, PeerJ CS 7:e330 (2021)**）给出了 fp16 累加的**具体**机制：fp16 乘积精确、加法在 binary32 中以 **round-towards-zero** 做、内部累加器有效位宽 **27 bit**、加数按**绝对值从大到小**累加、部分和不归一化 ⇒ **点积非单调**。

> ⚠ **这是引用值**，本实现者未逐字读到原文（本轮 `WebFetch` 被策略拦截）。它与 PTX 的「at least half precision」**不冲突**：PTX 说的是**下界**，这篇说的是某个具体实现的**上界行为**。

### 第五步：split-K 的溢出陷阱

**引用值**（cuBLAS 官方）：compute type 的精度高于 output type 时，**split-K 的分块和可能产生本不该有的中间溢出**，由 `CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION` 控制。

**这条与 §1 第二步的陷阱 2 是同一类问题**：溢出不是「每一步都检查」就能挡住的，真因是**中间部分和**越过了值域。split-K 把归约拆成多段，每一段都有自己的中间和——**段数越多，越界的机会越多**。

## 2. 执行映射

### 2.1 四个组件名与一个数值量级

- **`ldmatrix`** —— 把数据从共享内存洗进寄存器、匹配 tensor core 布局的硬件指令。**关键事实（逐字读到，见「参考」）：它不支持 4-bit 类型。** 这是「为什么 int4 必须 prepacking」的根因。
- **`cp.async`** —— 异步拷贝，把 GMEM→SMEM 的搬运从计算路径上摘下来（Marlin 用，另加 `evict_first` cache hint，**引用值**）。
- **TMA / WGMMA** —— Hopper 的一对（Machete 用，**引用值**）。
- **`lop3`** —— 一条三输入逻辑指令，Marlin 用它**直接在寄存器里把 int4 反量化成 fp16，并落进 tensor core 要的寄存器布局**（**引用值**）。
- **量级**：A100 的峰值吞吐（**A100 白皮书**，引用值）——

| 精度 | dense | with sparsity |
|---|---|---|
| INT8 TC | **624 TOPS** | **1248 TOPS** |
| INT4 TC | **1248 TOPS** | **2496 TOPS** |

**稀疏在每个精度上都恰好 2×，INT4 相对 INT8 再 2×。**

### 2.2 为什么 W4A16 的瓶颈在写侧

这是本主题主线（**省下的字节去哪了**）在 kernel 这一层的落点：

| | 每个权重元素 |
|---|---|
| 读（int4 载荷） | 0.5 字节 |
| **写（反量化成 fp16）** | **2.0 字节** |

**写是读的 4 倍。** 一个独立的反量化 pass 因此**被写侧带宽卡死**。融合进 GEMM 之后反量化发生在**寄存器里**（Marlin 的 `lop3` 做的就是这件事），fp16 权重**永远不落回 HBM**。

**⇒ 「量化省下的字节」在这一层上的去向是清楚的：载荷的节省只兑现了一半，另一半要靠在 kernel 内部把反量化做掉才能拿回来。** 而这一步的收益**随 batch 增大而衰减**（dequant 在总时间里的占比下降）。

### 2.3 一个可以真量的对照

本机在 CPU 上量到的是**累加器这一层**（§1 第二步与 §4.1），量不到 kernel 那一层。**这个差别必须在文档里说明**：

| 层次 | 本机能不能量 | 证据档位 |
|---|---|---|
| 累加器的宽度与上界 | ✅ 真 torch | **本机实测值**（§1 第二步、§4.1） |
| 一个算子是不是精确整数累加 | ✅ 真 torch | **本机实测值**（`15-w8a8` §4.2） |
| 打包布局 | ✅ 真 torch | **本机实测值**（`16-w4a16-packing`） |
| kernel 的吞吐 / 加速比 | ❌ 无 GPU | **引用值**（本篇 §3 的全部数字） |

## 3. 实现与演进

> ⚠ **本节全部数字都是引用值。** 本仓库**不复核其测量口径**。

### 3.1 Marlin：把一个 INT4 GEMM 做成「几乎不受 batch 影响」

**arXiv:2408.11743**（*MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models*，PPoPP'25；作者来自 **IST-DASLab 与 Neural Magic**）。本实现者**逐字读到了 abstract**：

> "batchsizes up to 16-32 can be supported with close to **maximum (4×)** quantization speedup, and larger batchsizes up to 64-128 with gradually decreasing, but still significant, acceleration"；
> "end-to-end LLM inference speedups (of up to **2.8×**) when integrated with the popular vLLM serving engine"。

它的设计要点（**引用值**，逐条抄自计划基线，本实现者未逐字核实原文）：

| # | 技巧 | 解决什么 |
|---|---|---|
| 1 | **离线重排**，使一个 warp 用**单条 128-bit 指令加载 1024 个 INT4** | 4-bit 没有 `ldmatrix` ⇒ 靠重排把加载次数压下来 |
| 2 | **`cp.async` + `evict_first`** cache hint | 权重只读一次，别污染缓存 |
| 3 | **流水线深度 P=4 + double buffering** | 把访存延迟藏起来 |
| 4 | **共享内存 XOR swizzle** | 避免 `ldmatrix` 的 bank conflict |
| 5 | **`lop3` 直接反量化成 FP16**，且直接落进 tensor core 的寄存器布局 | 反量化不经过中间张量（§2.2） |
| 6 | **列向累加**：让「下一个 B 操作数的反量化」与「当前列的 TC 计算」重叠 | 把 dequant 藏进计算 |
| 7 | **权重预重排成 tensor core 布局** | 同 1 |

> ⚠ 「A10 上 batch 128 降到约 **1.5×**」这一条**不在 abstract 里**（它来自检索摘要）——按引用值标注，**未逐字核实**。

### 3.2 Machete：Marlin 的「Hopper 版」，但归属不同

**这一节的归属最容易写错，逐条钉死：**

| 项 | 事实 | 证据 |
|---|---|---|
| 归属 | **Neural Magic**（不是 IST-DASLab） | Red Hat Developer 2024-10-14 官方介绍，**逐字读到**：*"Machete is **Neural Magic's** latest advancement in mixed-input quantization performance"* |
| 与 Marlin 的关系 | **spiritual successor**，但**作者不同**（Marlin 由 Elias Frantar 创建、Neural Magic 集成） | 同上，逐字 |
| **有没有 arXiv 论文** | **没有。** 它没有论文，只有 README 与官方博客 | vLLM 的 machete `Readme.md`（2672 字节，**逐字读到**）里**没有任何 arXiv 编号** |
| 起点 | CUTLASS 3.5.1 的 **example #55** | Red Hat 文，逐字：*"built on top of the work highlighted in CUTLASS 3.5.1 (see example #55 as our initial starting point)"* |
| 核心技巧 | **weight pre-shuffling**（因为 4-bit 没有 `ldmatrix`）+ **TMA** + **WGMMA** + **warp specialization** | Red Hat 文 + `Readme.md` |
| 数学形式 | `out = (w_q.to(scale_type) * w_s - w_z.to(scale_type)) @ a`，即**逐元素** zero-point | `Readme.md`，**逐字** |
| zero-point 的顺序 | *"`w_z` is added after the scales so we can use FMA operations, but this means they must have the scales pre-applied"* | `Readme.md`，**逐字** |
| 性能 | H100 上 Llama-3.1-70B 输入吞吐 +29%、输出 +32%（**引用值**） | 计划基线 |

> 🔴 **两条明令禁止**：
> 1. **不得给 Machete 编造 arXiv 编号。** 它没有论文。
> 2. **不得写「Machete 用混合 tile / SIMT 双路径」。** 这个说法**已被否证**——`Readme.md`（2672 字节）与 Red Hat 的官方介绍（148111 字节）本实现者都**逐字读过**，**两处都没有 SIMT 字样**。

**pre-shuffling 的理由，官方原文说得很直白**（Red Hat 文，逐字）：

> *"For standard data types (16, and 32-bit), NVIDIA provides an efficient `ldmatrix` instruction to perform this shuffling in hardware … **However, this instruction isn't available for 4-bit types.**"*
> *"…we can perform **128-bit loads**—the widest shared load instruction currently available on CUDA devices."*

**⇒ 4-bit 的「一个字节装两个值」不只是省字节：它把「数据怎么进寄存器」这件事从「一条硬件指令」降级成了「软件重排」。** 这是 `16-w4a16-packing` 那个布局公式在 GPU 侧的同源问题。

### 3.3 ExLlamaV2：一条反方向的工程选择

**ExLlamaV2** 走的是另一条路：**反量化融进内积，且走 SIMT、不走 tensor core**。

- ⚠ **本条是「计划基线，本次未独立核实」。** 本实现者**拿不到该源码**（`raw.githubusercontent.com` 全程 404 / 超时）——**不作逐字引用**。
- 计划基线的说法：`q_gemm_kernel.cuh` 里 `mma` 零命中、全部 `__hfma2`；group scale 融进点积；并且**特意用 FP32 累加防溢出**，源码注释写着 *"Use FP32 accumulator to avoid potential overflow since unscaled weights are in the range −128..127"*。
- 本实现者**只能独立印证它的定性部分**：检索摘要里有「ExLlamaV2 的内核有时用 FP32 累加以防数值溢出」这一说法。

**这条为什么值得写**：它说明 **tensor core 不是唯一答案**。当一个 kernel 是**访存受限**的时候（decode 阶段算术强度 ≈ 1），SIMT 路径也能跑满带宽，而且省掉了 prepacking 与 tensor core 布局的复杂度。**代价**是它放弃了 compute-bound 场景（大 batch / prefill）。

> ⚠ **仓库已归档**，开发转向 ExLlamaV3（**引用值**）。

### 3.4 CUTLASS：int4 的指令与 Hopper 快路径

| 项 | 值 | 档位 |
|---|---|---|
| Ampere(Sm80) 的 int4 MMA | `mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32.satfinite`（**累加器固定 int32**） | **引用值**,未逐字核实 |
| Hopper 的 4-bit 快路径 | `examples/55_hopper_mixed_dtype_gemm/` | **引用值**，未核实原文 |
| ★ 唯一被独立证实的 | **Machete 的起点就是 CUTLASS 3.5.1 的 example #55** | Red Hat 文，**逐字** |

计划基线还记了 Hopper 那条快路径的四条硬约束（只支持 TMA / scale 与 zero-point 类型必须 ≥ 2 字节 / group size 必须是 gemm-k 或 threadblock-k 的倍数 / narrow type 必须 K-major）——**本次未拿到 README 原文，正文只按「引用值、未核实」写，不落成常量。**

### 3.5 与 2:4 稀疏叠加：两个 2× 不叠成 4×

**一个纯算术的事实**（判据在上表已经给全）：

| 组合 | 相对 dense FP16 的理论加速 | 算式 |
|---|---|---|
| sparse INT8 | **4×** | 2×（稀疏）× 2×（INT8 vs FP16） |
| dense INT4 | **4×** | 2×（INT4 vs INT8）× 2×（INT8 vs FP16） |

**⇒ 两者理论加速相同。** 所以「稀疏 + int4」**不是** `4 × 4 = 16`——它是把一个 4× 换成另一个 4×。

真实收益来自**两个不同的瓶颈被同时缓解**：稀疏省**算力**（TC 的 MAC 数），量化省**带宽**（权重字节）。一个 kernel 只有在**同时**受这两者约束时才能把两份都拿到。

**Sparse-Marlin 的实测（引用值）**：它是 `IST-DASLab/Sparse-Marlin`（**注意：这个才是 IST-DASLab 的，Machete 不是**）。README **逐字**：

> "provides near-optimal speedups of around **5.3x**"；
> "Marlin is able to preserve close-to-ideal 4x performance up to a batch size of **16**, Sparse-Marlin … is able to preserve this condition with a batch size twice as large (i.e., **32**)"；
> "once performance decays for larger batch sizes, Sparse-Marlin can still take advantage of SPTCs to consistently deliver **2x** over the dense FP16 counterpart"。

> ⚠ 「dense Marlin 之上再快最多 **65%**」与「端到端最多 **3.2×**」**不在 README 里**——来自检索摘要，**未逐字核实**。

**cuSPARSELt 的稀疏格式按类型不同**（**引用值**）：

| 类型 | 格式 |
|---|---|
| `e2m1` | **paired 4:8** |
| `half` / `bfloat16` / `int` / `int8` / `e4m3` / `e5m2` | **2:4** |
| `float` | **1:2** |

**⇒ 「2:4」严格说只适用于列出的这些类型**，`e2m1`（也就是 MXFP4 的元素格式）走的是 paired 4:8。

## 4. 踩坑与失败模式

### 4.1 把本机 CPU 的 fp16 行为外推到 GPU

**现象**：写「fp16 的 `torch.mm` 用 fp32 累加」。

**成因**：本机**确实**测到这样（判据见下），但那是 **CPU 路径**。PTX 规范里 `.ctype`/`.dtype` 都是 `.f16` 的累加变体**确实存在**，且**累加顺序 unspecified**。

**判据（两条设计成能区分两种解释的点）**：

| 输入 | 若 fp16 累加 | 实测 | 结论 |
|---|---|---|---|
| `[2048, 1, 1, …, 1]`（K=2049 项）`@ ones` | 2048（`2048+1` 先被舍掉，一路加下去仍是 2048） | **4096** | 只有 fp32 累加能得到 |
| `[2048, 1, 1]`（K=3）`@ ones` | 2048 | **2050** | 排除了 fp16 累加 |

**规避**：写成「**本机 CPU 的** fp16 `torch.mm` 用 fp32 累加（torch 2.11.0 实测）」，并**明写不得外推**。

### 4.2 把三个「fp16 精度不够」的说法混成一句

**现象**：写「fp16 累加精度不够，最多到 2048」。

**成因**：这是**三个判据不同的量**被捏在了一起：

| 数 | 它是什么 | 判据 |
|---|---|---|
| **2048** | fp16 能**精确表示的整数**上界 | `2^11`，11 位有效位 |
| **1 → 2** | `[1024,2048)` 与 `[2048,4096)` 的 **ulp** | 10 位显式尾数 |
| **65504** | fp16 的**最大有限值** | S1E5M10 的位域 |

**规避**：分开写（§1 第二步的表）。**尤其**：`2048` 是「**连续整数**能精确表示的上界」，不是「fp16 能表示的最大整数」——本机实测 `4096` 在它之上**仍然是原值**（2 的幂），而 `100000` 会**溢出成 `inf`**（判据：`code/17_kernel_design.py` 的 C.1 与 `code/15_w8a8.py` 的 C.7）。**「舍入」与「溢出」是两种不同的失败**，`2049` 是前者、`100000` 是后者。

### 4.3 把 2:4 稀疏与 INT4 当成 4×4

**现象**：写「2:4 稀疏给 2×、INT4 给 4×，合起来 8× 或 16×」。

**成因**：把两个**不同基准**的倍率直接相乘。A100 白皮书那张表的口径是：**稀疏的倍率是相对同精度的 dense**（INT8 624→1248、INT4 1248→2496）；而**INT4 相对 INT8 的 2×、以及相对 FP16 的 4×** 是另一组比较。

**规避**：**先写清基准再相乘**。正确的账是：sparse INT8 = 4× vs dense FP16，dense INT4 = 4× vs dense FP16——**两者理论相同**。

### 4.4 给 Machete 编一个 arXiv 编号

**现象**：看到 Marlin 有 `arXiv:2408.11743`，就给 Machete 也找一个编号。

**成因**：两者名称相近、主题相同、都进了 vLLM——**极易串**。但它们**归属不同**（Marlin 是 IST-DASLab + Neural Magic 共同作者；Machete 只有 Neural Magic），**而且 Machete 没有论文**。

**规避**：本机可复核的判据——vLLM 的 machete `Readme.md`（**2672 字节**）里**没有任何 arXiv 编号**。**不得编造。**

### 4.5 把「混合 tile / SIMT 双路径」写进 Machete

**现象**：任务书/二手材料里有「Machete 用混合 tile 或 SIMT 双路径」这个说法，看着很专业。

**成因**：**它没有出处。** 本实现者逐字读了两份一手材料（`Readme.md` 2672 字节、Red Hat 官方介绍 148111 字节），**两处都没有 SIMT 字样**。

**规避**：不写。**这正是 P4（编机制从句）的形状**——没有数字可 grep，只能靠读原文否证。

### 4.6 把 ExLlamaV2 的源码注释当成逐字引用

**现象**：写「ExLlamaV2 的源码注释说 *'Use FP32 accumulator to avoid potential overflow…'*」，读起来像逐字引用。

**成因**：这句话来自计划基线，而本实现者**本次拿不到该源码**（`raw.githubusercontent.com` 全程 404 / 超时）。

**规避**：**标明「计划基线，本次未独立核实」**，或直接不引。把它写成逐字引文就是 P6（核查范围的表述大于实际核查范围）。

## 编译与运行

```bash
cd 130-model-optimization/quantization/code
make run PYTHON=/path/to/python
```

本篇对应的编号程序是 **`code/17_kernel_design.py`**（真 torch；B 类断言在程序内部，失败即非零退出）。只跑本篇：

```bash
make 17_kernel_design
```

纯 python3 的独立复核（A/C 类，**不 import torch**）在 `code/verify_kernels.py` 的 A.3 / A.4 / A.10 与 C.3 / C.5 节。

**为什么本篇没有 §5**

按本域的全域口径（见 `code/Makefile` 顶部长注释）：一篇文档的**数字全部是引用值**时，§5「实测数据」**整节省略**。

本篇的 kernel 性能数字（Marlin 的 4×、Machete 的 +29%/+32%、Sparse-Marlin 的 5.3×、A100 的 TOPS）**全部需要 GPU**，本机 `torch.cuda.is_available()` 为 **False** ⇒ **一条 GPU 数字都不可能有**。

> ⚠ **但本篇不是「没有本机实测」**：§1 第二步的三个 fp16 边界与 §4.1 的累加宽度判据，都是**本机在真 torch 上量的**（torch 2.11.0 / python 3.12.9 / Apple M1 Pro，CPU-only）。它们**收在 §1 与 §4**，不进 §5。**把 `2048 + 1 → 2048` 这种数从文档里删掉，才是自毁。**

## 6. 小结与延伸

- **累加器的选择是三个不同的代价**：int32 的代价是**宽度**（4 字节 + epilogue 必须先转 fp32）、fp16 的代价是**精度上界**、fp32 是那个「安全但贵」的中间档。**int32 那一条是规范钉死的**（PTX `.s32`）。
- **fp16 累加有三个判据不同的量，不得捏成一句**：`2048`（可精确表示的整数上界）、`1 → 2`（ulp）、`65504`（最大有限值）。
- **PTX 明文**：fp16 累加是「**at least** half precision」，且**顺序 unspecified** ⇒ 指令层面这个变体**确实存在**，**本机 CPU 的行为不是反例，也不得外推**。
- **split-K 的中间溢出**是「compute type 精度 > output type」时的一类特有风险，由 `CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION` 控制。
- ★ **4-bit 没有 `ldmatrix`** ⇒ 权重**必须**预重排（pre-shuffling）。这是 `16-w4a16-packing` 那个布局公式在 GPU 侧的根因。
- ★ **Machete 是 Neural Magic 的、没有 arXiv 论文、用逐元素 zero-point**；**「混合 tile / SIMT 双路径」这个说法被否证**。
- ★ **ExLlamaV2 走 SIMT、用 fp32 累加防溢出**（未独立核实）——它说明 **tensor core 不是唯一答案**。
- ★ **2:4 稀疏 × INT4 不叠成 4×4**：sparse INT8 = 4× vs dense FP16、dense INT4 也是 4×，**两者理论相同**。

**延伸**：

- 累加误差的**数值分析**视角（`√K·u` 的来历、随机符号假设失效时退化为 `γ_K ≈ K·u`）在 `10-foundations/floating-point/04-low-precision-formats`。**本篇只问「这种累加够不够支撑这个量化 kernel」，那条推导不复述。**
- fp16 / bf16 的位域与转换舍入在 `130-model-optimization/quantization/03-bf16-fp16-tf32`。
- 结构化稀疏本身（2:4 的数学、稀疏度与精度的取舍）在 `130-model-optimization/sparsity/01-sparsity-granularity`。
- tensor core 与异步流水线的通用形态在 `40-kernels/gemm/04-tensor-core-and-async-pipeline` 与 `40-kernels/gemm/05-persistent-kernel-and-tcgen05`（**正文在本域，回链 `40-kernels`**）。

## 参考

**本仓库程序（可复现）**

- `code/17_kernel_design.py` —— 真 torch 侧的 B/A/C 断言（fp16 累加宽度、fp16 的两个上界、fp8 的输出溢出、引用条目的自洽）。
- `code/verify_kernels.py` 的 A.3 / A.10 与 C.3 —— 纯 python3 的独立复核（fp16 的 ulp 用 `struct` 的 `'e'` 逐点算出、2:4 的算术）。
- `code/specs_kernels.py` —— 外部常量源，每条带 `# 来源：`；§6 是本篇的全部引用值，§8 是版本断点表。

**外部来源（**本实现者逐字读到的**）**

- **vLLM 的 machete `Readme.md`**（`csrc/quantization/machete/`，**2672 字节**，urllib 直取）：
  `out = (w_q.to(scale_type) * w_s - w_z.to(scale_type)) @ a`；
  *"Machete is a spiritual successor to the Marlin kernel but optimized for Hopper architectures and based on Cutlass."*；**全文无 arXiv 编号**。
- **Red Hat Developer，2024-10-14，*Introducing Machete***（**148111 字节**，urllib 直取）：
  *"Machete is **Neural Magic's** latest advancement…"*；*"the spiritual successor to the Marlin kernels created by **Elias Frantar** and integrated into vLLM by Neural Magic"*；
  *"…NVIDIA provides an efficient `ldmatrix` instruction… **However, this instruction isn't available for 4-bit types.**"*；
  *"built on top of the work highlighted in **CUTLASS 3.5.1** (see **example #55** as our initial starting point)"*；
  *"we can perform **128-bit loads**—the widest shared load instruction currently available on CUDA devices."*
- **arXiv:2408.11743 的 abstract**（*MARLIN*，PPoPP'25；**43783 字节**，urllib 直取）：
  *"batchsizes up to 16-32 … with close to maximum (4×) quantization speedup"*；*"end-to-end … speedups (of up to 2.8×) when integrated with the popular vLLM serving engine"*。
- **`IST-DASLab/Sparse-Marlin` 的 README**（GitHub 页面直取）：*"near-optimal speedups of around 5.3x"*；*"batch size twice as large (i.e., 32)"*；*"consistently deliver 2x over the dense FP16 counterpart"*。

**外部来源（**引用值，本实现者未逐字读到原文**）**

- **PTX 规范**：fp16 累加的 `"at least half precision"` 与 `"accumulation order … unspecified"`。
- **cuBLAS**：split-K 与 `CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION`。
- **A100 白皮书**：INT8 624 / 1248 TOPS、INT4 1248 / 2496 TOPS。
- **CUTLASS**：`mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32.satfinite`；Hopper 的 `examples/55_hopper_mixed_dtype_gemm/` 及其四条硬约束。
- **cuSPARSELt**：稀疏格式按类型（`e2m1` 走 paired 4:8）。
- **Fasi / Higham / Mikaitis / Pranesh, PeerJ CS 7:e330 (2021)**：fp16 累加的 27 位内部累加器、round-towards-zero、按绝对值从大到小累加。
- **Marlin 的七条技巧**与「A10 上 batch 128 约 1.5×」、**Sparse-Marlin 的「dense 之上 +65%」与「端到端 3.2×」**：来自计划基线与检索摘要。

**UNSOURCED（正文未使用）**

- **ExLlamaV2 源码注释的逐字原文**与 `grep mma = 0` 这个计数：本实现者**本次拿不到该源码**（`raw.githubusercontent.com` 全程 404 / 超时）。**§3.3 只按「计划基线、未独立核实」写，不声称逐字读到。**
- **本机任何 GPU kernel 性能数字**：本机无 GPU，**不编**。

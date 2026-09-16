---
title: Tensor Core 接入与异步流水
slug: tensor-core-and-async-pipeline
domain: 40-kernels
topic: gemm
level: 4
difficulty: advanced
prerequisites:
  - 40-kernels/gemm/03-register-tiling-and-occupancy
  - 20-gpu-architecture/tensor-core/03-wgmma-and-pipeline
  - 20-gpu-architecture/architecture-evolution/01-capability-breakpoints
tags: [gemm, tensor-core, wgmma, cp-async, tma, pipeline, smem-descriptor]
related:
  - 40-kernels/gemm/05-persistent-kernel-and-tcgen05
  - 20-gpu-architecture/tensor-core/01-mma-primitive
  - 20-gpu-architecture/tensor-core/02-fragment-layout
  - 20-gpu-architecture/memory-system/01-hierarchy-overview
code:
  - code/04_wgmma_gemm.cu
status: stable
updated: 2026-09-16
---

# Tensor Core 接入与异步流水

> FP16 的拐点是 295.4，比 FP32 的 20.0 靠右 14.8 倍。而要把 tile 撑到那个拐点需要 591×591 的块，它的 fp32 累加器是 1.33 MB——**整块寄存器文件的 5.33 倍**。所以 FP16 GEMM 不可能靠单块 tiling 到达拐点：它必须靠 L2 在 tile 之间提供复用，而 kernel 侧的任务变成「别让 tensor core 饿着」。

## 0. 背景与动机

[03 篇](03-register-tiling-and-occupancy.md)把 FP32 通路用满了：`TM=TN=8` 的寄存器分块让每 FMA 的 SMEM 字节正好降到 1.0，与 128 B/周期的交付率取等。**FP32 的天花板被摸到了**——67 TFLOPS。

换到 FP16 tensor core，规格表上的峰值变成 **989.5 TFLOPS**（H100 SXM，dense），是 FP32 的 **14.8 倍**。直觉上「换更快的指令」应该直接把性能乘上去。实际不会，原因在拐点上：

```text
I*(FP32 向量) = 67.0e12 / 3350e9 = 20.0
I*(FP16 TC)   = 989.5e12 / 3350e9 = 295.4
```

**拐点右移了 14.8 倍。** 而块级算术强度（[02 篇](02-shared-memory-tiling.md) §1 第二步）方阵时是 `AI_block = BM/esize`：

```text
FP32 要达到 I* = 20.0  ⟹ BM = 20.0 × 4 = 80
FP16 要达到 I* = 295.4 ⟹ BM = 295.4 × 2 = 590.7
```

`80×80` 的 tile 很容易装下（累加器 25 KB）。`591×591` 呢？它的 fp32 累加器是 `591² × 4 = 1364.4 KB`——**每 SM 寄存器文件（256 KB）的 5.33 倍**。装不下，无论怎么排布。

所以 FP16/FP8 的 GEMM 面对的是一个与 FP32 不同的问题：**它从第一天起就是「喂不饱」的问题**。`BM=BN=128` 的 tile 给出 `AI_block = 64`，只有拐点的 21.7%，可达 214.4 TFLOPS——已经是 FP32 峰值的 3.2 倍，但离 989.5 还差 4.6 倍。填这个差距有三条路：靠 L2 在 tile 之间复用（本域管不了，那是调度问题）、把搬运从线程的指令流里剥离、把累加器搬出寄存器文件（[05 篇](05-persistent-kernel-and-tcgen05.md)）。本文做前两件。

## 1. 原理

按六步推导。**MMA 指令的形状与 fragment 布局一律回链 [20-gpu-architecture/tensor-core/01-02](../../20-gpu-architecture/tensor-core/01-mma-primitive.md)，本文不重推 lane→元素的映射。**

**第一步：指令的 ISA 档位。**

| 特性 | ISA 下限 | 说明 |
|---|---|---|
| `mma.sync` | `sm_70` | warp 级，fragment 在寄存器 |
| **`cp.async`** | **`sm_80`** | global→shared **不经寄存器** |
| **TMA（`cp.async.bulk`）** | **`sm_90`** | 一个线程发起整块多维搬运 |
| `cluster` / DSMEM | `sm_90` | 簇内 block 互访 SMEM |
| FP8 | `sm_90` | |
| **`wgmma`** | **`sm_90a`** | warpgroup 级，B 走 SMEM 描述符 |
| `tcgen05` / TMEM | `sm_100a` | CTA 级、单线程发射、D 在 TMEM |

（规格值，来源为 PTX ISA 各指令的 Target ISA Notes，与本仓库 `code/specs.py` 的 `FEATURE_FIRST_TARGET` 一致。）

两处最常写错的：**TMA 是 `sm_90`，不是 `sm_90a`**（只有 multicast 等子变体被收到 `sm_90a`）；而 **`wgmma` 才是 `sm_90a`**。另外 `sm_90a` / `sm_100a` 是 "accelerated" 变体——**为它编出的 cubin 跑不了 `sm_90`；反过来 `sm_90` 的 cubin 能在 H100 上正常跑。这是单向不回退，不是双向不兼容。**

**第二步：为什么搬运要脱离寄存器。**

[03 篇](03-register-tiling-and-occupancy.md)的分块内核里，global→SMEM 的路径是 `LDG`（global→寄存器）→ `STS`（寄存器→SMEM）：**一次搬运两条指令，且中间那个寄存器是纯中转**。它带来两笔开销：

- **发射槽**：每 SM 每周期只有 4 个（规格值），而搬运指令一个 FLOP 都不做；
- **寄存器压力**：在途的搬运数据要占着寄存器，直接压缩了 [03 篇](03-register-tiling-and-occupancy.md) 精打细算出来的那点预算。

`cp.async`（`sm_80`）把这条路径压成一条指令：数据从 global 直达 SMEM，不经寄存器。它的粒度是每线程 **4 / 8 / 16 字节**（`cp-size`），16 字节（一个 `float4`）是规范粒度——**16 字节也意味着 16 字节对齐**。

**第三步：多级流水——把 SMEM 预算算清楚。**

一条 `cp.async` 发出之后不阻塞，靠「异步组」计数与 `__pipeline_wait_prior(N)` 回收。`N = 阶段数 − 1` 时，就有 `N` 段数据同时在途。每段的 SMEM 占用是固定的：

```text
每段字节 = (BM·BK + BK·BN) · esize
级数上限 = ⌊228 KB / 每段字节⌋   （还要与别的 block 共享这 228 KB）
```

FP16、`BM = BN = 128`：

| `BK` | 每段字节 | `228 KB` 允许的级数 | 单 block 独占时的级数 |
|---|---|---|---|
| 32 | 16 KB | 14 | 14 |
| **64** | **32 KB** | **7** | **7** |
| 128 | 64 KB | 3 | 3 |

**级数 × 每段字节 = SMEM 总量，是个常数。** 所以「`BK` 大 + 级数少」与「`BK` 小 + 级数多」是在同一个预算里选点，不是两个独立旋钮。这一点与 [03 篇](03-register-tiling-and-occupancy.md) §4.4 是同一条。

**第四步：流水线要藏住什么。**

每段的时间有两个：算的时间与搬的时间。

```text
t_compute(BK) = 2·BM·BN·BK / 每SM的tensor算力
t_memory(BK)  = 每段字节 / 每SM分到的带宽
```

H100 的 FP16 dense 是 989.5 TFLOPS / 132 SM = 7.496e12 FLOP/s；带宽 3350 GB/s / 132 = 25.4 GB/s（都按「每 SM 平均分到」算，推导值）。`BK = 64` 时：

```text
t_compute = 2·128·128·64 / 7.496e12 = 0.28 us
t_memory  = 32768 / 25.4e9           = 1.29 us
```

**搬的时间是算的 4.6 倍**——这个内核在块级就是带宽受限的。比值 `0.28/1.29 = 21.7%` 与 `AI_block/I* = 64/295.4 = 21.7%` 一致，两条算法互为校验。

由此得到多级流水的意义与它的边界：流水线能把「搬」与「算」重叠起来，但它**不能**把 4.6 倍的关系变成 1。重叠之后，时间由较长的那一侧决定——也就是 `t_memory`。要让 tensor core 不空转，只能提高 `AI_block`（更大的 tile，撞上第三步的 SMEM 与 [03 篇](03-register-tiling-and-occupancy.md) 的累加器墙），或者让 L2 把重复的读挡掉（那属于 tile 调度，不是 kernel 内部的事）。

**第五步：`wgmma` 的操作数位置。**

`wgmma` 的规格里有一条必须说准（回链 [20-gpu-architecture/tensor-core/03](../../20-gpu-architecture/tensor-core/03-wgmma-and-pipeline.md)）：

- **B 必须来自 shared memory**（通过 SMEM 描述符）；
- **A 可以是寄存器，也可以是 SMEM 描述符**。

写成「A、B 都必须在 SMEM」是错的。这个区别有实际后果：A 留在寄存器可以省一份 SMEM 占用，在 `BM` 小、`BN` 大的形状下更有利；而 A 走描述符则省寄存器。两者是可选的设计点，不是规格强制的单选项。

同样来自规格的两条：`wgmma` 的发射粒度是整个 **warpgroup（128 线程）**，不是单线程（`.aligned` 要求 warpgroup 内所有线程执行同一条指令）；`wgmma` 的 M 固定为 64，N 从 8 到 256、步长 8。

**第六步：把两条改进叠起来看效果。**

| 配置 | 承载 transport 的通路 | 在途缓冲 | tensor core 的喂数来源 |
|---|---|---|---|
| 03 篇的 FP32 内核 | 寄存器（`LDG` + `STS`） | 寄存器 + 1 份 SMEM tile | 寄存器（FFMA） |
| + `cp.async` 多级 | LSU 直接写 SMEM | SMEM 的 `N` 段 | SMEM 描述符 / `ldmatrix` |
| + TMA | SM 内的 TMA 单元 | SMEM 的 `N` 段 + mbarrier | 同上 |

每往下一行，**占用线程发射槽与寄存器的份额都在减少**，而 SMEM 的份额在增加——因为缓冲从「寄存器里的在途数据」变成了「SMEM 里的段」。这就是为什么到了 Hopper，tile 能开得比 Ampere 大。

## 2. 执行映射

### 2.1 `cp.async` 与 TMA 的差别

两者都是「不经寄存器」的搬运，差别在**谁算地址、谁发起**：

| | `cp.async`（`sm_80`） | TMA / `cp.async.bulk`（`sm_90`） |
|---|---|---|
| 发起者 | **整个 warp 的每条 lane 各发一条** | **单个线程**发起一次 |
| 地址来源 | 每条 lane 各算一个地址（占寄存器） | **描述符**（tensor map）住在 global，host 侧构造 |
| 维度 | 一维、连续的一段 | **最多 5 维**，带 stride 与 box 形状 |
| 粒度 | 每线程 4/8/16 字节 | 整块（按描述符的 box） |
| 完成通知 | 异步组计数（`__pipeline_*`） | **mbarrier**（`cp.async.bulk...mbarrier::complete_tx`） |
| 每 SM 的资源 | 占用 LSU 与发射槽 | **独占的 TMA 单元** |
| ISA 档位 | `sm_80` | **`sm_90`**（不是 `sm_90a`） |

差别可以概括成一句：`cp.async` 是「把地址计算与发射下放给所有线程」，TMA 是「把地址计算交给一个硬件单元、线程只发一个指针」。后者省下的不只是寄存器与发射槽——它让「一个线程负责搬运、其他线程负责计算」这种 **warp specialization** 成为可能，而那正是 Hopper 上 GEMM 内核能进一步变大的原因。

（`cp.async` 的规范粒度 16 字节、以及它在流水线深度与寄存器预算之间的取舍，本仓库 [20-gpu-architecture/architecture-evolution/01](../../20-gpu-architecture/architecture-evolution/01-capability-breakpoints.md) §3 已经算过一遍，本文不重推。）

### 2.2 具体的部件与量级

| 部件 | 在本内核里做什么 | 量级 |
|---|---|---|
| LSU | `cp.async` 的发起与地址生成 | 每线程 4/8/16 字节 |
| SMEM 写端口 | `cp.async` 落地的位置 | 无冲突 128 B/周期 |
| TMA 单元 | Hopper 起每 SM 一个，独立于 LSU 工作 | 一条指令搬整个 box |
| mbarrier | TMA 的完成通知与流水线阶段的回收 | 每个阶段一个 |
| tensor core | `wgmma` 的执行体 | 每 SM 7.496e12 FLOP/s（FP16 dense） |
| warpgroup | `wgmma` 的发射单位 | 128 线程 |
| HBM | 所有在途段的最终来源 | 每 SM 分到 25.4 GB/s |

一个具体的读数（推导值）：`BM = BN = 128`、`BK = 64`、FP16、4 级缓冲时，SMEM 是 `4 × 32 KB = 128 KB`，**每 SM 只能放 1 个 block**——占用率掉到 12.5%。这不是 bug，是 Hopper 上 GEMM 的常态：tile 大、缓冲深、block 少，靠 warpgroup 内部的 ILP 与管线深度而不是靠多 block 的 TLP 来藏延迟。这一条与 [03 篇](03-register-tiling-and-occupancy.md) 里 FP32 内核的取舍**方向相反**，是换精度带来的结构性变化。

### 2.3 fragment 与描述符

内层循环吃的是 `wgmma` 的两类操作数：

- **SMEM 描述符**：一个 64 位值，把「起始地址、leading 维字节偏移、stride 维字节偏移、swizzle 模式」编码进去。它是**地址的编码**，不是一个句柄——所以同一份 SMEM 在 K 方向每推进 16 个元素就要换一个描述符。
- **寄存器 fragment**：累加器 D 与（可选的）A。它们怎么摊到 128 个线程上，**由 fragment 布局规则决定，正文在 [20-gpu-architecture/tensor-core/02](../../20-gpu-architecture/tensor-core/02-fragment-layout.md)，本文不重推**。

swizzle 模式那 2 bit 与 [02 篇](02-shared-memory-tiling.md) §3.1 的 bank 冲突是同一个问题的两个面：把行号 XOR 进列地址，既消掉冲突又保持 16 字节对齐。**这也是为什么 tensor core 的 SMEM 布局不能随便选**——描述符的 LBO/SBO 必须与实际的行距、swizzle 一致。

## 3. 实现与演进

### 3.1 三代指令的三次搬家

| 代 | 指令 | 发射单位 | A / B 从哪来 | D 在哪 | transport 走哪 |
|---|---|---|---|---|---|
| Volta / Turing / Ampere | `mma.sync` | **warp**（32 线程） | 寄存器（`ldmatrix` 从 SMEM 装） | 寄存器 | 寄存器 → SMEM → 寄存器 |
| Hopper | `wgmma` | **warpgroup**（128 线程） | A：寄存器或 SMEM；**B：必须 SMEM** | 寄存器 | `cp.async` / **TMA** → SMEM |
| Blackwell | `tcgen05` | **1 个线程** | A：TMEM 或 SMEM；**B：必须 SMEM** | **TMEM** | **TMA** → SMEM |

每一次搬家都在减少「通用资源的占用」：发射单位从 32 降到 1（省发射槽）、D 从寄存器搬到 TMEM（省寄存器文件）、transport 从线程搬到 TMA 单元（省 LSU 与寄存器）。**唯一一直在涨的是 SMEM 的份额。**

### 3.2 演进的下一个问题

把上面三代按「喂得饱吗」排一遍，会看到一条清晰的线索：

- Ampere：tile 不能太大，因为累加器占寄存器（[03 篇](03-register-tiling-and-occupancy.md) §1 第五步的 255 墙）；
- Hopper：`wgmma` 把 A/B 的装载从寄存器挪走了一部分，但**累加器仍在寄存器里**——`m64n256` 的 fp32 累加器摊到 128 个线程是每线程 128 个寄存器，`65536 ÷ 128 = 512` 个线程，也就是**整个寄存器文件只装得下 4 个 warpgroup 的累加器**（这条账在 [20-gpu-architecture/execution-model/01](../../20-gpu-architecture/execution-model/01-sm-internals.md) §3 已经算过，本文只引用）；
- Blackwell：把 D 搬进 TMEM，这条约束才被解开。

也就是说，本文这条路（`cp.async` + `wgmma`）走到的终点，恰恰是 [05 篇](05-persistent-kernel-and-tcgen05.md) 的起点。

## 4. 踩坑与失败模式

### 4.1 把 `wgmma` 的操作数写成「A、B 都必须在 SMEM」

**现象**：读规格时看到「B 来自 shared memory 描述符」，顺手推广成「A 和 B 都要先在 SMEM 里摆好」。于是为了把 A 也放进 SMEM，额外开一块缓冲、多一次搬运、多占 `BM·BK·esize` 的 SMEM——在 `BM` 大、SMEM 紧张时直接把级数挤掉一级。

**成因**：规格是**非对称**的。`wgmma` 的 B 必须来自 SMEM 描述符（这条是硬约束），而 A 可以是寄存器，也可以是 SMEM 描述符（这是可选的设计点）。把它读成对称的，会白白多付一份 SMEM 与一次搬运。

**规避**：记成「**B 必须在 SMEM；A 两边都行**」。选哪一边看形状：A 留在寄存器省 SMEM、费寄存器；A 走描述符反之。规格原文与两条相关的限定符（`.aligned` 要求 warpgroup 一致执行、`.sync`）在 [20-gpu-architecture/tensor-core/03](../../20-gpu-architecture/tensor-core/03-wgmma-and-pipeline.md)，此处回链。

### 4.2 把 TMA 的 ISA 档位写成 `sm_90a`

**现象**：Makefile 里把用到 TMA 的目标塞进「Hopper 加速档」，用 `-arch=sm_90a` 编；或者反过来，认为「TMA 只在 H100 上能用，A100 上不行」（前半句对，后半句的原因说错了——A100 用不了是因为它是 `sm_80`，不是因为它不是 `sm_90a`）。

**成因**：`sm_90` 与 `sm_90a` 是**两个不同的 target**，`cp.async.bulk` 的下限是前者。把它归到后者，会得到一个错误的兼容性结论：以为「TMA 的东西只能在加速档里编」，从而把本来更可移植的代码降级成不可移植的。**兼容方向也是单向的**：为 `sm_90` 编出的 cubin 能在 H100 上跑，为 `sm_90a` 编出的不能跑在没有 `a` 的 `sm_90` 上。

**规避**：按指令逐条查 Target ISA Notes，别按「代」归类。本仓库的 `code/specs.py` 已经把这张表固化下来（`cp_async` → `sm_80`、`tma_cp_async_bulk` → `sm_90`、`wgmma` → `sm_90a`、`tcgen05` → `sm_100a`），构建系统据此分档：可移植档进 `TARGETS`，加速档进 `HOPPER_TARGETS` / `BLACKWELL_TARGETS` 并用**显式** `-arch`。

### 4.3 `cp.async` 的字节数与对齐

**现象**：搬运循环里按元素数算偏移（例如「每线程搬 6 个 `__half`＝12 字节」），或者让指针落在非 16 字节对齐的地址上。编译通过，运行起来数据偶尔不对。

**成因**：`cp.async` 的 `cp-size` 只能是 **4 / 8 / 16** 字节，而 16 字节那一档**要求源与目的都 16 字节对齐**。12 字节这种取值根本不存在；非对齐的 16 字节拷贝则是未定义行为。GEMM 里这条特别容易踩，因为 `BK` 或行宽不总是 16 字节的整数倍——FP16 时 `BK = 8` 的一行才 16 字节，而 FP32 时 `BK = 4` 就到了 16 字节，边界随精度变化。

**规避**：把搬运循环按「字节」而不是「元素」写，并用 `static_assert` 把「每段字节数 % 16 == 0」以及「行宽 × esize % 16 == 0」钉死。不满足时退回 8 或 4 字节档，或者给 tile 补 padding（这时的 padding 是为了对齐，不是为了 bank——两者要分开记）。核对手段是 `cuda-memcheck` 与逐元素的正确性比对。

### 4.4 级数堆太多，吃光 SMEM 预算

**现象**：为了让流水线更深，把 `STAGES` 从 4 提到 8、16。延迟藏得更好了，但性能反而下降，甚至启动失败。

**成因**：`级数 × 每段字节 = SMEM 总量`（§1 第三步）。级数翻倍就是 SMEM 翻倍，而每 SM 的上限是 228 KB。级数堆过头有两个后果：(a) 每个 SM 只能放 1 个 block，占用率掉到 12.5%——此时**藏延迟的手段从多 block 的 TLP 变成了单 block 的管线深度**，一旦某一段的内存时间抖动，整条流水线一起停顿；(b) 超过 228 KB 时内核直接起不来（[02 篇](02-shared-memory-tiling.md) §4.1 的判据同一条）。

**规避**：级数由「延迟 ÷ 每段计算时间」定出**下界**，由「SMEM 预算 ÷ 每段字节」定出**上界**，取一个够用又不吃满的值。上式的两个量都要用运行时/编译期的数（`cudaOccupancyMaxActiveBlocksPerMultiprocessor` 会给 block 数），不要按「越多越好」调。级数够用的判据很直接：**流水线不空转**——再往上加，吞吐不再涨。

## 编译与运行

配套程序是 **`code/04_wgmma_gemm.cu`**（`cp.async` 4 级流水 + `wgmma` 的 m64n64k16 主循环 + SMEM 描述符构造）：

```bash
cd code
make hopper            # 用显式的 -arch=sm_90a 构建（不混进默认 TARGETS）
./04_wgmma_gemm 1024 1024 1024
```

**需要 H100（`sm_90a`）。在 A100（`sm_80`）上编不过属预期，不是 bug**——`make` 与 `make run` 都不包含它。

⚠ **本文件在写它的机器上没有编译过。** 本仓库的写作环境（macOS）没有 CUDA 工具链：SMEM 描述符的位域编码按 PTX ISA §9.7.15 的 Matrix Descriptor 一节写（bits 0-13 起始地址 >>4、16-29 LBO、32-45 SBO、62-63 swizzle），用的是**无 swizzle 的 K-major 规范布局**；真机使用前请与 CUTLASS 的 `make_gmma_desc` 交叉核对。程序内部对这一点有显式声明。

文档里的数字分两类，**不要混**：

- `I*(FP16) = 295.4`、`AI_block = BM/esize`、临界 tile 591、累加器 1364.4 KB、每段字节与级数上限、`t_compute`/`t_memory` 的 0.28 µs 与 1.29 µs——全部是**推导值**，可用 `make verify` 的 §B1 / §B3 / §B7 / §B11 / §C6 复核；
- 任何耗时与 TFLOPS 都是**那台机器上的运行结果**，本仓库一个都没有。

本机（macOS）只能跑：

```bash
make verify
```

对应本文的是 §B1（两个拐点）、§B7（临界 tile 与累加器墙）、§B11（每段字节、级数上限、`t_compute`/`t_memory` 与两条算法的互校验）、§C6（累加器容量表），以及 §B3 里 FP16 那一侧的块级 AI 与可达算力。

## 6. 小结与延伸

本文一条主线：**精度越快，拐点越靠右；而把 tile 撑到拐点的代价按面积涨。FP16 的临界 tile（591×591）需要 1.33 MB 的累加器，是整块寄存器文件的 5.33 倍——所以 FP16 GEMM 从第一天起就是「喂不饱」的问题。** 围绕它有三个可独立复用的中间结论：

1. `I*(FP16) = 295.4` 对 `I*(FP32) = 20.0`，差 14.8 倍；`BM = I*·esize` 只涨 7.4 倍（80 → 590.7），累加器却涨 54 倍（25 KB → 1.33 MB）。**换精度不是免费的**。
2. `级数 × 每段字节 = SMEM 总量`。多级流水能把「搬」与「算」重叠，但**不能把 `t_memory/t_compute` 从 4.6 变成 1**——重叠之后时间由较长的那一侧决定。
3. `cp.async`（`sm_80`）与 TMA（`sm_90`，**不是 `sm_90a`**）的差别是「谁算地址、谁发起」；`wgmma` 的 **B 必须在 SMEM**，**A 两边都行**（不是「都必须在 SMEM」）。

接下来 [05-persistent-kernel-and-tcgen05](05-persistent-kernel-and-tcgen05.md) 从本文的第三条结论往外走一步：既然 Hopper 上「整个寄存器文件只装得下 4 个 warpgroup 的累加器」，那就把累加器整个搬出去（TMEM），顺便让发射粒度降到 1 个线程；同时用 persistent kernel 把「每个 tile 付一次启动开销」改成「每个 SM 付一次」。

## 参考

- NVIDIA, *PTX ISA* —— §9.7.15（`wgmma`：形状枚举、`.aligned`/`.sync` 限定符、Matrix Descriptor 的位域编码）、§9.7.9（`cp.async` 的 `cp-size` 与 `cp.async.bulk`）、各指令的 Target ISA Notes。
- NVIDIA, *NVIDIA Hopper Architecture In-Depth* 与 *Hopper Tuning Guide* —— TMA 单元的定位、warpgroup 的由来、`wgmma` 的发射语义。
- NVIDIA, *NVIDIA H100 Tensor Core GPU Datasheet* —— FP16 dense 989.5 TFLOPS、HBM3 3.35 TB/s、132 SM。
- 本仓库 [20-gpu-architecture/tensor-core/03-wgmma-and-pipeline](../../20-gpu-architecture/tensor-core/03-wgmma-and-pipeline.md) —— `wgmma` 的操作数规格、warpgroup 的由来、流水深度与寄存器预算的取舍（本文只回链）。
- 本仓库 [20-gpu-architecture/tensor-core/01-mma-primitive](../../20-gpu-architecture/tensor-core/01-mma-primitive.md) 与 [02-fragment-layout](../../20-gpu-architecture/tensor-core/02-fragment-layout.md) —— MMA 形状表与 fragment 布局规则（**本文不重推**）。
- 本仓库 [20-gpu-architecture/architecture-evolution/01-capability-breakpoints](../../20-gpu-architecture/architecture-evolution/01-capability-breakpoints.md) —— 逐条指令的 ISA 下限、`cp.async` 对流水深度与寄存器预算的影响。
- 本仓库 `code/verify.py` 的 §B1、§B3、§B7、§B11、§C6 —— 本文的拐点、块级 AI、临界 tile、流水线时间账与累加器容量数字来源。

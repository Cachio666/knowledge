---
title: 朴素 GEMM 与 roofline
slug: naive-gemm-and-roofline
domain: 40-kernels
topic: gemm
level: 2
difficulty: intermediate
prerequisites:
  - 10-foundations/performance-models/01-roofline-model
  - 20-gpu-architecture/execution-model/01-sm-internals
tags: [gemm, naive, arithmetic-intensity, roofline, memory-bound, l1-bandwidth]
related:
  - 40-kernels/gemm/02-shared-memory-tiling
  - 10-foundations/numerical-linear-algebra/02-gemm-rounding-error
  - 20-gpu-architecture/memory-system/02-coalescing-and-cache-line
code:
  - code/01_naive_gemm.cu
status: stable
updated: 2026-09-16
---

# 朴素 GEMM 与 roofline

> 朴素 GEMM 的算术强度是 `1/esize`（FP32 下 0.25），与矩阵形状**无关**；而 roofline 口径下的 GEMM 是 `n/(3·esize)`（n=4096 时 682.7）。这两个数差 2731 倍——中间这段距离就是本主题后面五篇的全部内容。

## 0. 背景与动机

先给一个具体的对照场景：**FP32，`M = N = K = 4096`，在 H100 SXM 上**。

一个刚学 CUDA 的人会这样写内核：一个线程算 `C` 的一个元素，`for k` 循环里每次都从全局内存取 `A[i][k]` 和 `B[k][j]`。他算了一下 FLOPs：`2MNK = 1.374e11`，再查 H100 的 FP32 峰值 67 TFLOPS，得出「至少要 2.05 ms」。然后他调 block size（16×16 换成 32×8）、加 `__restrict__`、把内层手动展开 4 层、把 `float` 换成向量化的 `float4` 读——**耗时几乎没有变化**。

因为那个 2.05 ms 是**算力侧**的时间。这个内核真正在做的事是：每个输出元素要经历 `K = 4096` 轮循环，每轮读 8 个字节（`A` 一个 float、`B` 一个 float）、做 2 次浮点运算。整道题要从内存里过 **549.756 GB**（推导值，`2MNK·4`），而理想情况下只需要 `(MK+KN+MN)·4 = 201.3 MB`——**2731 倍**。

代入 roofline：`I = 0.25 FLOP/byte`，`I*(FP32) = 20.0`，可达算力 `3350e9 × 0.25 = 0.8375 TFLOPS`，是峰值的 **1.25%**。他要优化的那 67 TFLOPS 里，有 98.75% 从来没被碰到过。

本文要做的是把这条算术推清楚，并回答一个问题：**同一道 GEMM，为什么「它的算术强度」会同时等于 0.25 和 682.7？** 这两个数都成立，区别只在口径——而把口径搞混，后面五篇会全部读歪。

## 1. 原理

按六步推导。每一步只引入一个量。

**第一步：运算量。** 每个输出元素是一次长度为 `K` 的内积，`K` 次乘加；一次乘加记 2 FLOP：

```text
FLOPs = 2 · M · N · K
```

**第二步：口径一——「按指令发起」的访存字节。** 朴素内核里，一次 FMA 需要两个操作数，两个都来自全局内存：

```text
每个输出元素：K 轮 × (读 A 一个元素 + 读 B 一个元素) = 2K 次元素读
整题：       2 · M · N · K 次元素读  =  2MNK · esize 字节

I₁ = 2MNK / (2MNK·esize) = 1 / esize
```

FP32 下 `I₁ = 0.25`。**注意这个数里没有 M、N、K**——它只由数据格式的字节宽度决定。方阵也好、瘦高也好、1×1 也好，朴素内核的算术强度恒等于 `1/esize`，FP16 是 0.5、FP8 是 1.0。

**第三步：口径三——roofline 原文的字节数。** roofline 的 `Bytes` 只统计**必须往返 HBM** 的字节，且默认复用全被缓存挡住（定义见 [10-foundations/performance-models/01](../../10-foundations/performance-models/01-roofline-model.md)）。在这个口径下，`A`、`B`、`C` 各自只从 HBM 过一遍：

```text
Bytes₃ = (M·K + K·N + M·N) · esize

I₃ = 2MNK / ((MK + KN + MN)·esize)
```

方阵 `M = N = K = n` 时化简：

```text
I₃ = 2·n³ / (3·n²·esize) = n / (3·esize)
```

`n = 4096`、FP32：`I₃ = 4096/12 = 682.7`。**这个数与 n 成正比，没有上界**——题越大，GEMM 越像算力受限。这正是 roofline 那张图上 `P` 点（prefill 大 GEMM，`I = 2048`）踩在水平段上的原因。

**第四步：把两个口径的差算出来。**

```text
I₃ / I₁ = [2MNK / ((MK+KN+MN)·esize)] / [1/esize]
        = 2MNK / (MK + KN + MN)
方阵时  = 2n³ / (3n²) = 2n/3
```

`n = 4096` 时是 **2730.7 倍**。这就是 §0 里那 2731 的来历——它**不是**一个经验系数，而是一个纯代数结论：**朴素内核多读的那部分，正是「同一个元素被读了多少遍」**。方阵时每个 `A` 的元素被读 `N` 次、每个 `B` 的元素被读 `M` 次。

**第五步：代入 roofline。**

| 口径 | `I`（FP32，n=4096） | 判定 | 可达算力 | 占 FP32 峰值（67 TFLOPS） |
|---|---|---|---|---|
| 一（本内核实际发起） | `1/esize` = **0.2500** | 带宽受限 | `3350e9 × 0.25` = **0.8375 TFLOPS** | **1.25%** |
| 二（见下） | `1/(2·esize)` = **0.1250** | 带宽受限 | **0.4188 TFLOPS** | **0.62%** |
| 三（roofline 原文） | `n/(3·esize)` = **682.7** | 算力受限 | 封顶 67 TFLOPS | 100% |

上表三行是 `make verify` 的 §B2 直接打印出来的（`B·I` 一栏换算过单位）。

**第六步：一个变体——把 K 循环提到 kernel 外面。**

朴素内核把累加器放在**寄存器**里，所以 `C` 从头到尾只写一次，那 `MN·esize = 67.1 MB` 的写流量相对 `549.756 GB` 可以忽略。但如果 K 循环在 kernel 外面（每次 launch 处理一段 K，用 `beta·C` 的形式累加），`C` 每段都要读回来再写回去：

```text
每个输出元素每段：读 A + 读 B + 读 C + 写 C = 4·esize 字节
I₂ = 2K / (4K·esize) = 1/(2·esize)
```

FP32 下是 0.125，比口径一**再低一半**。写回流量是 `MN·esize × 段数`；段数取满 `K` 时就是 `2MNK·esize = 549.756 GB`，与读流量同量级。这条变体在工程上不稀奇——它就是「把 GEMM 的 K 维切给多次 launch」的写法，也是 [06-split-k-and-stream-k](06-split-k-and-stream-k.md) 要处理的形态的起点。

**小结这条主线：`I₁` 与 `I₃` 都是「这道 GEMM 的算术强度」，前者描述内核**发起了多少字节**，后者描述**这道题最少需要多少字节**。tiling 做的事情一句话就能说完：把内核从 `I₁` 挪向 `I₃`。挪不挪得动，取决于片上能装下多少复用——那是 02 与 03 篇的内容。

## 2. 执行映射

### 这些字节从哪里来

上面 §1 第二步说「每个输出元素 2K 次元素读」，落到硬件上是：每个 `k` 迭代，一个 warp 发两条 `LDG`（一条读 `A`、一条读 `B`），然后发一条 `FFMA`。`A` 的那条读在同一 warp 内地址相同（所有 lane 读同一个 `A[i][k]`），硬件按**广播**处理，一个 32 字节 sector 就够；`B` 的那条读在 `col` 连续时是**合并**的，32 个 float 正好 128 字节。

合并与广播降低的是**请求数**，不是**每个线程需要的字节数**。一个 lane 要为一次 FMA 拿到 8 个字节，这件事不因合并而改变——它只是让这 8 个字节的获取更便宜。（合并的机制与代价，正文在 [20-gpu-architecture/memory-system/02](../../20-gpu-architecture/memory-system/02-coalescing-and-cache-line.md)，此处回链。）

### 三道天花板，最紧的一道在 HBM

即使假设所有读都命中 L1，朴素内核也**上不去**。把三道约束依次算出来（推导值）：

| 约束 | 部件 | 算式的来历 | 上限（占 FP32 峰值） |
|---|---|---|---|
| 发射槽 | 每 SM 4 个 warp scheduler | 每 FMA 要 2 条 `LDG` + 1 条 `FFMA` = 3 条 warp 指令；4 槽/周期 → 4/3 条 `FFMA`/周期，而满速是 4 条 | **33.3%** |
| L1 / SMEM 交付率 | 32 bank × 4 B = 128 B/周期 | 每周期需要 `128 lane × 8 B = 1024 B`，交付只有 128 B | **12.5%** |
| HBM | 3350 GB/s | `I₁ = 0.25` ⟹ `3350e9 × 0.25` | **1.25%** |

三道约束取最小，最紧的是 HBM：**1.25%**。第二列的两个数字值得单独记住——`SMEM_DELIVERY_BYTES_PER_CLK = 128`（32 个 bank × 4 字节，无冲突时的交付率）与「128 个 FP32 lane 每周期要 1024 字节操作数」之间**差 8 倍**。这意味着**任何**从片上存储逐次取操作数的写法都不可能跑满 FP32 通路；操作数必须在**寄存器**里被复用。这句话是 03 篇的支点。

### 延迟不是主要矛盾，带宽才是

一个常见的诊断是「访存延迟没被掩盖住」。对朴素 GEMM 而言这不是主要问题：`I₁ < I*` 说明它**已经在带宽受限区**，此时再加并发也换不来吞吐（判据见 [20-gpu-architecture/execution-model/03](../../20-gpu-architecture/execution-model/03-occupancy-and-latency-hiding.md) §0）。延迟只解释「为什么真机上跑出来比 0.8375 TFLOPS 还低」，不解释「为什么封顶在 0.8375」。

数量级对照（规格值）：L1/SMEM 延迟 20~30 周期，HBM 往返 400~800 周期，每 SM 每周期最多发射 4 条指令。朴素内核每个输出元素要发 `2K` 条 `LDG`，`K = 4096` 时是 8192 条——这些请求的排队与扇出，就是它的全部工作。

## 3. 实现与演进

GEMM 的优化史可以只用一个自变量讲完：**把「每个 FMA 需要 8 字节」这个比例降到多少**。

朴素实现的比例是 8 字节/FMA（两次全局读，每次 4 字节）。三个台阶把它在降：

1. **共享内存分块**（[02-shared-memory-tiling](02-shared-memory-tiling.md)）。让一个 block 把 `BM×BK` 的 `A` 与 `BK×BN` 的 `B` 从 HBM 取**一次**，在 block 内复用 `BM`（或 `BN`）次。HBM 读流量降到 `1/BM + 1/BN` 倍，降幅 `2·BM·BN/(BM+BN)`。它解决的是 **HBM 那一层**。
2. **线程级寄存器分块**（[03-register-tiling-and-occupancy](03-register-tiling-and-occupancy.md)）。SMEM 的交付只有 128 B/周期，而 lane 需要 1024 B/周期——所以「从 SMEM 取」不能是每个 FMA 取一次。`TM×TN` 的寄存器小块让每个 SMEM 元素被用 `TM`（或 `TN`）次，把 SMEM 侧的需求压到 128 B/周期以下。它解决的是 **L1/SMEM 那一层**。
3. **张量核心与异步流水**（[04-tensor-core-and-async-pipeline](04-tensor-core-and-async-pipeline.md)）。前两步把「搬字节」这件事做便宜了，但搬运本身仍占用线程的发射槽。`cp.async` 让搬运不经寄存器，TMA 让一个线程就能发起整块搬运，`wgmma` 把累加器之外的算术全部收进 tensor core。它解决的是 **发射槽与流水的重叠**。

再往后两条是工程结构上的：[05-persistent-kernel-and-tcgen05](05-persistent-kernel-and-tcgen05.md) 处理「怎么让 SM 一直有活干、把累加器搬出寄存器文件」，[06-split-k-and-stream-k](06-split-k-and-stream-k.md) 处理「M/N 小的时候并行度凑不够」。

**本文只完成第 0 步：把朴素实现的账算清楚。** 上面三个台阶各自的推导分别在那三篇里，本文不展开——本文建立的只有两件事：`I₁ = 1/esize` 这个下限，以及它与 `I₃` 之间的 2730.7 倍。

## 4. 踩坑与失败模式

### 4.1 `for k` 内层重复读 A/B 的同一元素

**现象**：内层 `for k` 里每次迭代都写 `A[i*K+k]` 和 `B[k*N+j]`，看起来是「一次乘法配两次读」，很自然。调 block size、加向量化、展开循环都无效；profile 显示内存子系统饱和而 FP32 通路空闲。

**成因**：这两个元素在这**一个线程**的视角下确实只能各读一次，但在**整道题**的视角下，`A` 的每个元素要被读 `N` 次、`B` 的每个元素要被读 `M` 次（`I₁` 与 `I₃` 的比 `2MNK/(MK+KN+MN)` 就是这个「平均重复次数 × 2/3」）。重复的部分是否落到 HBM，取决于缓存能不能挡住——而 `A` 是 64 MB、`B` 是 64 MB（4096² × 4 B），远超 L1（与 SMEM 共享 256 KB）与 L2 的容量；再加上 `K` 循环在 1024 个 block 之间不同步，同一个元素的重复使用之间隔了成千上万次其他访问，复用距离远大于任何一级缓存。

**规避**：不要在**单个线程**的循环里消掉重复读——消不掉。要在**block** 的层面把复用做进 SMEM（02 篇），在**线程**的层面把复用做进寄存器（03 篇）。自检办法：算一遍 `2MNK·esize` 与实际耗时相除得到的等效带宽，如果它已经接近 HBM 峰值，那这个内核的时间就是被搬字节决定的，改计算侧一律无效。

### 4.2 用 `float` 累加，误差随 K 增长

**现象**：朴素内核用 `float acc` 累加，`K = 65536` 时结果与 `float64` 参考之间的**相对元素误差**可以到 `3.3e-2`，而同一份结果的「相对量级误差」只有 `9.7e-7`。两个数差四个数量级，看起来像两个不同的结论。

**成因**：这是浮点点积的误差界在起作用——点积的**绝对**误差界是 `γ_K·Σ|x_k·y_k|`（正比于绝对值之和），而**相对**误差还要除以 `|xᵀy|`。分母可以任意小。`make verify` 的 §A2 在 `256×256×65536` 上把这个现象打印了出来（推导值）：

| 归约顺序 | `max\|Δ\|` | 相对**量级**误差 | 相对**元素**误差 | 中位相对元素误差 |
|---|---|---|---|---|
| 顺序累加 | 9.336e-04 | 9.131e-07 | 3.332e-02 | 8.051e-07 |
| 逆序累加 | 9.336e-04 | 9.131e-07 | 3.290e-02 | 8.035e-07 |
| 树形累加 | 9.924e-04 | 9.706e-07 | 3.198e-02 | 8.015e-07 |

**中位数只有 `8.05e-07`，最大值 `3.33e-02`**——「这个 GEMM 的精度是多少」不加限定就没有答案。

**规避**：先确定你要的是哪个口径。判断「整体精度够不够」看相对量级误差，判断「某个特定的、接近 0 的输出元素能不能用」才看相对元素误差。整条误差界的推导（含 `γ_K = Ku/(1-Ku)` 与 `√K·u` 的关系）正文在 [10-foundations/numerical-linear-algebra/02](../../10-foundations/numerical-linear-algebra/02-gemm-rounding-error.md)，本文只引用结论。（顺带：上表三行的**位模式互不相同**——浮点加法不满足结合律。这条在 06 篇会被用作 split-K 不可复现的根据。）

### 4.3 拿 roofline 的 `I` 去评估朴素内核

**现象**：算出「4096³ 的 GEMM `I = 682.7`，远超 `I*(FP32) = 20`，所以它是算力受限的」，据此推断朴素内核应该能跑到接近峰值，然后花时间在计算侧优化。

**成因**：`682.7` 是**这道题的下限字节数**对应的算术强度，不是**这个内核发起的字节数**对应的。roofline 的 `Bytes` 口径是「必须往返 HBM 的字节」，它默认复用被缓存完全挡住——对 `n = 4096` 的方阵，这意味着假设 `A` 的每个元素只从 HBM 进来一次、然后被 `N = 4096` 个输出元素共享。任何具体实现能不能做到，是一个**实现问题**，不是一个模型问题。把模型的 `I` 套到实现上，会得到「问题在算力侧」的错误结论，而真正错的是那个假设。

**规避**：对**每一个**具体内核，都问一句「它实际发起/搬运了多少字节」。模型给的 `I₃` 是上界对应的值，你的内核的 `I` 落在 `I₁` 与 `I₃` 之间，位置由 tiling 参数决定。不要把这两个数当同一个数用。

### 4.4 把「算力利用率 1.25%」当成实现缺陷

**现象**：profile 看到 FP32 利用率只有 1.25%，认定内核写得差，投入大量时间重写。

**成因**：`1.25%` 是 roofline 在 `I₁ = 0.25` 下给出的**上限**，不是实现的缺陷。带宽受限的算子在算力侧本来就只该用到这个比例——`0.25 / 20.0 = 1.25%`，两式等价。把模型封好的上限当成优化空间，是在追一个不存在的东西。

**规避**：先判瓶颈在哪一侧（判据见 [10-foundations/performance-models/01](../../10-foundations/performance-models/01-roofline-model.md) §4.3）。`I < I*` 时唯一有效的杠杆是**提高算术强度**（多复用、少搬字节），而不是让计算更快。本主题后面五篇做的全是前一件事。

## 编译与运行

配套程序是 **`code/01_naive_gemm.cu`**（FP32 朴素内核 + 一个 K 循环在外的变体 + CPU `float64` 参考实现）。它需要 `nvcc` 与 GPU：

```bash
cd code
make 01_naive_gemm
./01_naive_gemm 4096 4096 4096
```

程序会把本文 §1 的三套口径、三张表里的数字按运行时的设备规格重新算一遍并打印，同时给出**本机**的墙钟时间。**这两类数字必须分开看**：

- 本文与 `make verify` 给出的是**推导值**——`2MNK·esize`、`I₁`、`I₃`、`0.8375 TFLOPS`、`1.25%`，全部由算术得到，可用 `make verify` 的 §B2 复核；
- 程序打印的耗时与 TFLOPS 是**那台机器上的运行结果**，本仓库没有跑过，也不对其数值负责。

本机（macOS，无 CUDA 工具链）只能跑：

```bash
make verify
```

它对应本文的是 §B2（三套口径的算式与代入）与 §B1（`I* = 295.4 / 20.0` 这两个规格值的除法）。

## 6. 小结与延伸

全文一条主线：**朴素 GEMM 的算术强度是一个常数 `1/esize`，与形状无关；而 roofline 口径下它是 `n/(3·esize)`，随 n 线性增长。** 两者相差 `2n/3` 倍，`n = 4096` 时是 2730.7 倍。围绕它有三个可独立复用的中间结论：

1. `I₁ = 1/esize`：FP32 是 0.25，FP16 是 0.5，FP8 是 1.0。它来自「一次 FMA 要两次元素读」这一个事实，不需要知道任何形状参数。
2. `I₃ = n/(3·esize)`（方阵）或 `2MNK/((MK+KN+MN)·esize)`（一般）：每个元素只从 HBM 过一次。这是**上界**，实现能逼近多少取决于片上复用。
3. 三道天花板依次是发射槽 33.3%、L1 交付率 12.5%、HBM 1.25%——**最紧的一道是 HBM**，所以朴素内核的时间由搬字节决定，计算侧的优化一律无效。

后面五篇按「把 `I` 从 `I₁` 推向 `I₃` 的三级台阶」展开：

- [02-shared-memory-tiling](02-shared-memory-tiling.md) 解决 HBM 那一层——分块把重复搬运的次数降下来，并交代 bank conflict 在 GEMM 里怎么用 padding/swizzle 消掉；
- [03-register-tiling-and-occupancy](03-register-tiling-and-occupancy.md) 解决 L1 交付率那一层——`TM×TN` 的寄存器小块，以及它与寄存器预算/占用率的取舍；
- [04-tensor-core-and-async-pipeline](04-tensor-core-and-async-pipeline.md) 解决发射槽与重叠——`cp.async` 多级流水与 tensor core 接入；
- [05-persistent-kernel-and-tcgen05](05-persistent-kernel-and-tcgen05.md) 与 [06-split-k-and-stream-k](06-split-k-and-stream-k.md) 处理剩下的两个工程结构问题：SM 常驻与并行度不足。

## 参考

- Williams, Waterman, Patterson, *Roofline: An Insightful Visual Performance Model for Multicore Architectures*, CACM 52(4), 2009 —— `I`、`I*` 与两段线的原始形式。
- NVIDIA, *CUDA C++ Programming Guide* —— §5.1「Compute Capabilities」（每 SM 的线程/寄存器/共享内存上限、warp 大小）；硬件实现章节（warp scheduler 数、共享内存 bank 结构）。
- NVIDIA, *NVIDIA H100 Tensor Core GPU Datasheet* 与 *Hopper Architecture In-Depth* —— 本文全部 H100 SXM 规格（FP32 向量 67 TFLOPS、FP16 dense 989.5 TFLOPS、HBM3 3.35 TB/s、132 SM）。
- 本仓库 [10-foundations/performance-models/01-roofline-model](../../10-foundations/performance-models/01-roofline-model.md) —— `I` 与 `I*` 的定义、两段线的推法、`Bytes` 只统计 HBM 流量的口径。
- 本仓库 [10-foundations/numerical-linear-algebra/02-gemm-rounding-error](../../10-foundations/numerical-linear-algebra/02-gemm-rounding-error.md) —— 点积的绝对误差界 `γ_K·Σ|x·y|` 与「相对误差被小元素支配」的完整推导。
- 本仓库 `code/verify.py` 的 §B1、§B2、§A2 —— 本文三张表的数字来源。

---
title: 共享内存分块
slug: shared-memory-tiling
domain: 40-kernels
topic: gemm
level: 3
difficulty: intermediate
prerequisites:
  - 10-foundations/performance-models/01-roofline-model
  - 20-gpu-architecture/memory-system/03-shared-memory-bank-conflict
  - 40-kernels/gemm/01-naive-gemm-and-roofline
tags: [gemm, tiling, shared-memory, bank-conflict, padding, swizzle, epilogue]
related:
  - 40-kernels/gemm/03-register-tiling-and-occupancy
  - 20-gpu-architecture/memory-system/01-hierarchy-overview
  - 20-gpu-architecture/memory-system/02-coalescing-and-cache-line
  - 20-gpu-architecture/execution-model/03-occupancy-and-latency-hiding
code:
  - code/02_tiled_gemm.cu
status: stable
updated: 2026-09-16
---

# 共享内存分块

> 分块把 HBM 的重复搬运降到 `1/(2·BM·BN/(BM+BN))`——128×128 的 tile 在方阵上是 **128 倍**。但它的代价是 SMEM 占用，而 SMEM 占用又直接决定占用率；tile 每大一档，能常驻的 block 就少一档。

## 0. 背景与动机

沿用 [01 篇](01-naive-gemm-and-roofline.md) 的对照场景：FP32、4096³、H100 SXM。朴素内核要把 **549.756 GB** 从 HBM 过一遍（推导值，`2MNK·4`），可达算力 `0.8375 TFLOPS`，是峰值的 1.25%。

现在做一个最小的改动：让每个 block 只负责 `C` 的 `BM×BN` 一块，并且把 `A` 的 `BM×BK` 与 `B` 的 `BK×BN` 先搬进共享内存，然后在片上反复取用。

变化是量级的。同样这道题，`BM = BN = 128` 时分块版的 HBM 读流量是 **4.295 GB**——降了 **128 倍**（推导值，`make verify` §B4）。算术强度从 `0.25` 涨到 `32.0`，**越过了 `I*(FP32) = 20.0`**：这个内核从带宽受限翻到了算力受限。

但代价立刻出现在另一处：`128×128×32` 的 tile 要 `32 KB` 共享内存。每 SM 的共享内存上限是 228 KB（与 L1 共用一个 256 KB 的池子），于是每个 SM 能常驻的 block 数被压到 3 个——768 个线程、24 个 warp、**37.5% 的占用率**。把 `BK` 从 32 提到 128，SMEM 变成 128 KB，占用率直接掉到 **12.5%**（1 个 block/SM）。

所以分块这个动作有三笔账要同时算：**HBM 流量降了多少**（收益）、**SMEM 占了多少**（代价）、**占用率还剩多少**（代价的代价）。本文把三笔账都推出来，并交代两件在真实 GEMM 里绕不开的事：SMEM 里的**行距**（bank 冲突）与写回 `C` 时的 **epilogue**。

## 1. 原理

按六步推导。

**第一步：SMEM 占用。**

一个 block 每个 `k` 段要同时持有 `A` 的一列块与 `B` 的一行块：

```text
smem = (BM·BK + BK·BN) · esize
```

`BM×BN×BK = 128×128×32`、FP32：`(128·32 + 32·128)·4 = 32768 B = 32 KB`。

**第二步：块级算术强度。**

一个 block 在其整个 K 循环里，从 HBM 取 `(BM·BK + BK·BN)` 个元素、做 `2·BM·BN·BK` 次浮点运算（每次搬运被复用 `BN`（对 A）与 `BM`（对 B）次）：

```text
AI_block = 2·BM·BN·BK / smem
         = 2·BM·BN·BK / ((BM·BK + BK·BN)·esize)
```

把 `BK` 约掉：

```text
AI_block = 2·BM·BN / ((BM + BN)·esize)
```

**`BK` 从分子分母同时消失。** 这是一个容易被忽略的结论：`BK` 不是算术强度的旋钮，它只改 SMEM 占用与 K 循环的迭代数。方阵 `BM = BN` 时进一步化简为 `BM/esize`：

| `BM×BN×BK` | smem | 块级 AI | ≤ 228 KB | L2 零命中下的可达算力 |
|---|---|---|---|---|
| 32×32×32 | 8.0 KB | 8.0 | 是 | 26.8 TFLOPS |
| 64×64×16 | 8.0 KB | 16.0 | 是 | 53.6 TFLOPS |
| **128×128×16** | 16.0 KB | **32.0** | 是 | 67.0 TFLOPS（封顶） |
| **128×128×32** | 32.0 KB | **32.0** | 是 | 67.0 TFLOPS（封顶） |
| 256×128×32 | 48.0 KB | 42.7 | 是 | 67.0 TFLOPS（封顶） |

（全部为推导值，`make verify` §B3 逐行打印。可达算力 = `min(67.0, 3350 GB/s × AI)`。）

看第三、四行：`128×128×16` 与 `128×128×32` 的块级 AI **完全相同**，尽管后者的 SMEM 占用是前者的两倍。这就是第二步那个「BK 被约掉」的直接后果。

**第三步（关键的口径警告）：这个 AI 是「单 block、假设 L2 零命中」口径。**

上表的 `AI_block` 描述的是**一个 block 从 HBM 取了多少**，前提是它每轮都从 HBM 取。真实 GEMM 里 `A` 的那一列块会被同一 block 行的 `N/BN` 个 block 共用、`B` 的那一行块会被同一 block 列的 `M/BM` 个 block 共用——只要这些 block 在时间上靠得够近，L2 会把重复的读挡掉。

**这道题自己的算术强度**（roofline 原文口径，每个元素只从 HBM 过一次）是：

```text
AI_global = 2MNK / ((MK + KN + MN)·esize)
方阵 n 时  = n / (3·esize)
```

FP32、`n = 4096`：`4096/12 = 682.7`。只看 A、B 的读（不含 C 的写）则是 `2n³/(2n²·esize) = n/esize = 1024`。

**方阵时它随 n 线性增长、没有上界。** 所以：

- 用 `AI_block` 下结论时，你会得到「GEMM 在 128×128 的 tile 下刚刚越过拐点、再大的 tile 才有余量」；
- 用 `AI_global` 下结论时，你会得到「大 GEMM 永远是算力受限的」。

两个结论都成立，**区别是你问的是「这个 block 需要 L2 帮多少忙」还是「这道题总共需要搬多少字节」**。把它们混成一句「GEMM 是 memory-bound」，两头都错。

**第四步：HBM 流量的降幅。**

朴素（零缓存口径）读 `2MNK·esize`；分块后（同样零缓存，每个 block 各自取自己的两条边）读 `(M/BM)(N/BN)·(BM·K + K·BN)·esize = MNK·(1/BM + 1/BN)·esize`。比值：

```text
降幅 = 2·BM·BN/(BM + BN) = AI_block × esize
```

4096³、FP32（`make verify` §B4）：

| `BM×BN` | 分块后读 | 降幅 |
|---|---|---|
| 32×32 | 17.180 GB | 32.0× |
| 64×64 | 8.590 GB | 64.0× |
| **128×128** | **4.295 GB** | **128.0×** |
| 256×128 | 3.221 GB | 170.7× |

**第五步：这道题的总字节并没有变。**

上表的降幅是**零缓存**口径的。整题级的读字节是 `(MK+KN)·esize = 134.2 MB`（4096³、FP32），**与 BM/BN 无关**。tiling 不减少总字节，它减少的是**同一份数据被重复搬运的次数**——从「每个 block 各搬一遍」变成「每个 block 只搬自己那一份」。

这个区别在 roofline 上落成一句话：**分块不改变 `I`，它提高的是「可达」的那一侧。** 多级 roofline 的说法是「它把复用推给了更快的层级，等效于抬高了斜线的斜率」。

**第六步：代入占用率，得到取舍。**

`BM = BN = 128`、线程 256 个（每线程 8×8）、每线程估算 80 个寄存器：

| `BK` | smem/block | SMEM 允许的 block 数 | 寄存器允许的 block 数 | 实际 | 常驻线程 | 占用率 | K 循环迭代数（K=4096） |
|---|---|---|---|---|---|---|---|
| 32 | 32 KB | 7 | 3 | **3** | 768 | 37.5% | 128 |
| 64 | 64 KB | 3 | 3 | **3** | 768 | 37.5% | 64 |
| 128 | 128 KB | 1 | 3 | **1** | 256 | **12.5%** | 32 |

（推导值。SMEM 一列 = `⌊233472 / smem⌋`；寄存器一列 = `⌊65536 / (80×256)⌋ = 3`。）

`BK` 从 32 走到 128，K 循环迭代数降到 1/4，但占用率降到 1/3。**两者都变坏的那个点是 `BK = 128`**：多出来的 SMEM 换来的迭代数减少，抵不过并发掉的 2/3。

## 2. 执行映射

### 这 32 KB 住在哪里

共享内存与 L1 缓存**共用同一块 256 KB 的物理存储**，按配置切分；H100 上至多 **228 KB** 可以切给共享内存（规格值）。所以 `smem` 直接与「能常驻几个 block」挂钩，而占用率又直接与「能不能把访存延迟藏住」挂钩（判据见 [20-gpu-architecture/execution-model/03](../../20-gpu-architecture/execution-model/03-occupancy-and-latency-hiding.md)）。

一个 block 的 SMEM 是**整份预留**的：block 在 SM 上存续期间，那 32 KB 一直归它，直到它整体退出。这与寄存器一样是预算、不是缓存。

### 搬运路径与两道同步

每个 `k` 段，256 个线程协作搬 `BM·BK + BK·BN = 8192` 个元素（32 KB），每线程 32 个 float。路径是：`LDG`（global→寄存器）→ `STS`（寄存器→SMEM）→ `BAR.SYNC` → `LDS`（SMEM→寄存器）→ `FFMA`。这里有两个具体的代价：

- **一次搬运两条指令**，且中间那个寄存器是纯中转——`cp.async` 存在的理由就是把这条路径压成一条（[04 篇](04-tensor-core-and-async-pipeline.md)）。
- **`__syncthreads()` 每段两次**：一次在写完之后（保证全体看得见），一次在读完之后（保证下一段覆写安全）。`BK` 越小，这个 barrier 的总数越多。

### 数值量级

| 量 | 数值 | 说明 |
|---|---|---|
| 每 SM 统一 L1/SMEM 池 | 256 KB，至多 228 KB 给 SMEM | 规格值，见 `code/specs.py` |
| SMEM 无冲突交付率 | 128 B/周期 | 32 bank × 4 B |
| SMEM 延迟 | 20~30 周期 | 比 L1 命中同量级 |
| HBM 往返延迟 | 400~800 周期 | 一次 `LDG` 的延迟量级 |
| 本内核每段搬运 | 32 KB（`BK=32`） | `128×128×32`、FP32 |
| 本内核每段 barrier | 2 次 | 写后 1 次、读后 1 次 |

SMEM 这一段延迟（20~30 周期）与 HBM 那一段（400~800 周期）差 20 倍以上，这正是「放进 SMEM 就值」的物理依据。

## 3. 实现与演进

### 3.1 行距：SMEM 里最容易白花的一笔开销

SMEM 的 128 B/周期交付率是**无冲突时**的数字。一旦 32 个 lane 的地址挤进同一个 bank，这次访问就被拆成多遍，交付率按遍数下降。（bank 的划分规则、为什么一次访问要串行几遍，正文在 [20-gpu-architecture/memory-system/03](../../20-gpu-architecture/memory-system/03-shared-memory-bank-conflict.md)，此处只讲在 GEMM 里怎么用。）

GEMM 里真正的冲突点不是 `A` 的读，而是 `B` 的读——准确地说，取决于你怎么放 `B`。

**第一步：认清哪种访问会冲突。** 内层循环里 `A` 的读是 `a[i] = As[(trow0+i)*BK + k]`：对固定的 `i`，一个 warp 内所有 lane 的地址**完全相同**（广播），不冲突。给 `A` 加 padding 是白花的。

`B` 就不同了。为了内层循环能取到连续的 `k`，通常把 `B` **转置**存成 `[n][k]`，读法变成 `b[j] = BsT[(tcol0+j)*stride + k]`。这时 warp 内 `tx = 0..15` 的地址是 `(2tx+j)·stride + k`——**16 个不同地址，落进哪些 bank 由 `stride` 决定**。

**第二步：算冲突路数。** 地址以 4 字节字为单位，bank = 字地址 mod 32：

| 行距 `BK+PAD` | `PAD` | 一个 warp 的不同地址数 | 冲突路数 |
|---|---|---|---|
| **32** | 0 | 16 | **16 路** |
| **33** | 1 | 16 | **1（无冲突）** |
| 34 | 2 | 16 | 2 路 |
| 36 | 4 | 16 | 4 路 |
| 40 | 8 | 16 | **8 路** |

（推导值，`make verify` §B8 逐行算出。行距为偶数时，冲突路数**恰好等于 PAD**；奇数行距完全无冲突。）

**第三步：读这张表的三个结论。**

1. 行距 `32` 是最坏情况：16 个地址全在同一个 bank，这条 `LDS` 要串行 16 遍，交付率降到 1/16。
2. `PAD = 1` 完全消除冲突。但它把行距变成 33 个 float = 132 字节——**不再是 16 的倍数**，`LDS.128`（`float4` 向量读）用不了了。省下的冲突遍数可能被「多出来的指令数」吃掉一半。
3. `PAD = 8` 只是把 16 路降到 8 路，**没有解决**。这条最容易踩：习惯了「加 padding 就行」的写法，随手加 8 个 float 凑 128 位对齐，冲突依然在。**偶数行距的冲突路数就等于那个偶数。**

要同时拿到「无冲突」与「16 字节对齐」，得换一种打散方式：**swizzle**——把行号 XOR 进列地址的高位，让地址在保持对齐的前提下散开。这是 tensor core 时代 SMEM 布局的常规做法，也是 `wgmma`/`tcgen05` 的 SMEM 描述符里那 2 bit swizzle 模式字段存在的原因。

### 3.2 epilogue：写回 C 的那一笔账

内层循环结束后，`C` 的块要从寄存器写回全局。完整形式是

```text
C_block = alpha · (A·B) + beta · C_block
```

`alpha` 的乘法在寄存器里做，**免费**。`beta` 不是——它要求先把 `C` 的那一块**读**回来。这笔读的规模是 `MN·esize`，方阵 `n = 4096`、FP32 时是 67.1 MB，等于理想总字节 `(MK+KN+MN)·esize = 201.3 MB` 的 **33.33%**（推导值，`make verify` §B9）。

所以 `alpha`/`beta` 的取舍不是「多一个参数」，而是：

- `beta = 0`（覆盖写）：省掉整次读，但调用者必须保证 `C` 的初值可以丢；
- `beta ≠ 0`（累加）：多 33% 的 HBM 流量，换来「不清空累加器」这个语义——训练里的梯度累加、`X += A·B` 这类接口都需要它。

工程上的第三条路是把 `beta` 折进 epilogue 的寄存器运算（`store(alpha*acc + beta*load(C))`，一次读一次写），这是唯一能让「语义不变、流量不翻倍」的写法；它把选择权留给了调用者，而不是强迫库在 `beta=0` 时也必须读一遍。

### 3.3 演进：从一级分块到多级

朴素 → 一级分块（本文）→ 寄存器分块（[03 篇](03-register-tiling-and-occupancy.md)）→ 异步流水与 tensor core（[04 篇](04-tensor-core-and-async-pipeline.md)）。每一步解决的层次不同：

| 台阶 | 解决的瓶颈 | 机制 | 本文的覆盖情况 |
|---|---|---|---|
| 一级分块 | HBM 重复搬运 | 把 tile 放进 SMEM，块内复用 | **本文** |
| 寄存器分块 | SMEM 交付率（128 B/周期不够 lane 用） | `TM×TN` 让 SMEM 元素被复用 TM/TN 次 | 03 篇 |
| 异步流水 | 搬运占用发射槽 | `cp.async` / TMA 让搬运不经寄存器 | 04 篇 |

**本文只推一级分块。** 后面两步的推导各自在对应篇里，本文不展开——但 §1 第六步那张占用率表与 §3 的 bank 冲突，是那两篇的公共前提。

## 4. 踩坑与失败模式

### 4.1 tile 太大导致每 SM 只剩 1 个 block

**现象**：为了让块级 AI 更高，把 tile 从 `128×128` 提到 `256×128` 甚至更大，或把 `BK` 从 32 提到 128。AI 确实涨了（42.7），但**耗时反而变长**。

**成因**：tile 变大时 SMEM 占用按 `(BM+BN)·BK` 线性涨，而每 SM 的 SMEM 上限是固定的 228 KB。`BM = BN = 128` 时 `BK` 从 32 提到 128，smem 从 32 KB 涨到 128 KB，能常驻的 block 数从 3 掉到 1——常驻线程从 768 掉到 256，占用率从 37.5% 掉到 12.5%（推导值，`make verify` §B5 的占用率口径）。占用率掉到这个程度，HBM 的 400~800 周期延迟就没法被藏住了。

**规避**：把占用率当成 tile 大小的**约束**而不是结果。先定「至少要几个 block/SM」（通常 ≥ 2），再反解 `BK ≤ (228KB/blocks − ...) / ((BM+BN)·esize)`；`BM = BN = 128`、FP32、要求 3 blocks/SM 时 `BK ≤ 76`，取 8 的倍数是 72。`__launch_bounds__`、`cudaOccupancyMaxActiveBlocksPerMultiprocessor` 都是现成的核对手段。**还要注意 tile 变大时寄存器（累加器 `BM·BN·4` B）与 32 blocks/SM 上限会同时收紧**，三道约束要一起看。

### 4.2 `BK` 太小导致 K 循环迭代数暴涨

**现象**：为了给占用率腾空间，把 `BK` 压到 8 甚至 4。占用率确实上去了（SMEM 只占 8 KB），但性能变差。

**成因**：`BK` 在算术强度里被约掉了（§1 第二步），所以它**不带来任何复用上的收益**，只带来两笔固定开销：(a) K 循环迭代数 = `K/BK`，`K = 4096` 时 `BK = 8` 是 512 次 `BK = 32` 的 4 倍；(b) 每段两次 `__syncthreads()`，加上每段的地址计算与循环控制。每段的**计算量** `2·BM·BN·BK` 随 `BK` 线性缩小，而这些开销基本不变——`BK = 8` 时每段的计算只有 1024 个周期量级，同步与地址计算的占比就上去了。分段还让「搬运」与「计算」的重叠更细碎：如果不做多级缓冲，每段都要先等 HBM 往返（400~800 周期）再算，`BK` 越小，这段等待占的比例越大。

**规避**：`BK` 的取值由「SMEM 预算允许几级缓冲」决定，而不是由 AI 决定。经验上 `BK` 取 16~64（FP16 可以更大），并且**配多级缓冲**——`BK` 小 + 级数多，比 `BK` 大 + 单缓冲更好：前者让流水线更细，后者让每一跳的等待更长。这一条在 [04 篇](04-tensor-core-and-async-pipeline.md) 会展开成「SMEM 总量固定时，stage 数与 BK 的乘积受限」。

### 4.3 给 A 加 padding，或者 padding 取偶数

**现象**：照着「加 padding 消除 bank 冲突」的说法，把 `As[BM][BK]` 改成 `As[BM][BK+8]`，或者给 A、B 都加。耗时没变，甚至略慢。

**成因**：这是把结论套在了错的访问模式上。`A` 的读在同一 warp 内地址相同（广播），**本来就不冲突**，padding 只是把 SMEM 占用抬高、把对齐弄坏。而 `B` 那一侧，如果取的是偶数行距，冲突路数**恰好等于那个偶数**（§3.1 的表：PAD=2 → 2 路，PAD=8 → 8 路）——随手加 8 个 float 的常见写法只把 16 路降到 8 路，看起来「有效但收益有限」，实际是没解决。

**规避**：先判断访问模式再决定加不加、加多少。判据是「一个 warp 内这条 `LDS` 有几个**不同地址**」：地址相同 → 广播，不用管；地址不同 → 看这些地址 mod 32 撞不撞。对转置存放的 `B`，`PAD = 1`（奇数行距）是唯一完全消除冲突的选择；如果需要保持 16 字节对齐，就上 swizzle 而不是继续加偶数。`make verify` §B8 把这张表按 bank 映射规则重算了一遍，可以直接对照。

### 4.4 忘了算 `beta ≠ 0` 的 epilogue 流量

**现象**：按 `AI_block` 算完分块的收益，认定 HBM 流量已经降到理论下限，但在真机上跑起来总比预期差一截；或者反过来，把库函数的 `beta` 参数当成「免费选项」随便传 1。

**成因**：`C` 的读改写是**每个输出元素一次**，规模 `MN·esize`，与 tile 大小无关。方阵 4096³、FP32 时它是理想总字节的 **1/3**（§3.2）。这个量级不大不小——小到算 AI 时容易被忽略，大到足以让一个已经贴近带宽上限的内核明显变慢。

**规避**：算这笔账时把 epilogue 单独列一行，不要并进主循环的 AI 里。默认 `beta = 0`；接口必须支持累加时，把 `beta·C` 的读放在 epilogue 的一次 `load` 里，别做成「先读回来、再算、再写」。顺带检查 `alpha`：如果它是 0，整个主循环都可以跳过——这种 trivial 分支库函数会特判，手写内核常常忘。

## 编译与运行

配套程序是 **`code/02_tiled_gemm.cu`**（分块内核 + `PAD` 开关 + bank 冲突路数的算术 + 运行时占用率查询）：

```bash
cd code
make 02_tiled_gemm
./02_tiled_gemm 4096 4096 4096
make 02_tiled_gemm NVCCFLAGS="-O3 -std=c++17 -DPAD=0"   # 行距 32 → 16 路冲突
make 02_tiled_gemm NVCCFLAGS="-O3 -std=c++17 -DPAD=1"   # 行距 33 → 无冲突
```

程序会把本文 §1 的分块算术、§3.1 的 bank 冲突表按运行时的编译参数重算并打印。**注意其中两类数字的性质不同**：

- SMEM 占用、块级 AI、流量降幅、bank 冲突路数，都是**推导值**（算术），可用 `make verify` 的 §B3 / §B4 / §B8 复核；
- 耗时与 TFLOPS 是**那台机器上的运行结果**，本仓库没有跑过，不对其数值负责。

本机（macOS，无 CUDA 工具链）只能跑：

```bash
make verify
```

对应本文的是 §B3（分块参数表）、§B4（流量降幅）、§B8（bank 冲突路数）、§B9（epilogue 的 33.33%），以及 §B5 里那三组占用率估算。

## 6. 小结与延伸

本文一条主线：**分块把 HBM 的重复搬运降到 `1/(块级 AI × esize)`，代价是 SMEM 占用，而 SMEM 占用决定占用率。** 三笔账要一起算。围绕它有三个可独立复用的中间结论：

1. `AI_block = 2·BM·BN/((BM+BN)·esize)`，**`BK` 被约掉了**——`BK` 不是算术强度的旋钮。
2. 这个 AI 是「单 block、L2 零命中」口径。整题级口径是 `2MNK/((MK+KN+MN)·esize)`（方阵 `n/(3·esize)`），随 n 线性增长、无上界。**两套口径不可混用**，混用会得出「GEMM 永远是 memory-bound」这类错误的一般结论。
3. SMEM 里真正会冲突的是转置存放的 `B`；`A` 的读是广播，不用管。行距为奇数才完全无冲突，偶数行距的冲突路数就等于那个偶数。

接下来两条延伸各有分工。[03-register-tiling-and-occupancy](03-register-tiling-and-occupancy.md) 接着本文的 SMEM 往下走一层：SMEM 的 128 B/周期交付率喂不饱 128 个 FP32 lane，必须靠 `TM×TN` 的寄存器小块把每个 SMEM 元素多用几次——那里会交代「为什么 8×8 恰好是平衡点」，以及寄存器用量与占用率的取舍。分块本身还有一个本文没碰的方向：`C` 的分块之间没有数据交换，所以要更高的吞吐只能让流水线更深、搬运更便宜，那是 [04-tensor-core-and-async-pipeline](04-tensor-core-and-async-pipeline.md) 的内容。

## 参考

- NVIDIA, *CUDA C++ Programming Guide* —— §5.1「Compute Capabilities」（每 SM 的共享内存容量、寄存器数、常驻线程/block 上限）；共享内存的 bank 结构与访问规则。
- NVIDIA, *CUDA C++ Best Practices Guide* —— 共享内存分块的实现要点、padding 与 bank 冲突的规避。
- NVIDIA, *NVIDIA H100 Tensor Core GPU Architecture Whitepaper* —— 每 SM 统一 L1/SMEM 池 256 KB、至多 228 KB 可切给 SMEM。
- NVIDIA, *PTX ISA* —— `ldmatrix` / swizzle 相关的 SMEM 布局约束；`wgmma` 的 SMEM 描述符里 swizzle 字段的定义。
- 本仓库 [20-gpu-architecture/memory-system/03-shared-memory-bank-conflict](../../20-gpu-architecture/memory-system/03-shared-memory-bank-conflict.md) —— bank 划分规则、串行遍数、padding 与 swizzle 的机制（本文只做代入）。
- 本仓库 [10-foundations/performance-models/01-roofline-model](../../10-foundations/performance-models/01-roofline-model.md) —— `I*` 的定义与「tiling 提高的是可达的那一侧」。
- 本仓库 `code/verify.py` 的 §B3、§B4、§B5、§B8、§B9 —— 本文四张表的数字来源。

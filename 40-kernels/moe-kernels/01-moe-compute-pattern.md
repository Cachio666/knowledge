---
title: MoE 的计算形态
slug: moe-compute-pattern
domain: 40-kernels
topic: moe-kernels
level: 3
difficulty: intermediate
prerequisites:
  - 40-kernels/gemm/02-shared-memory-tiling
  - 40-kernels/reduce-scan/01-reduction-algorithms
tags: [moe, routing, top-k, grouped-gemm, roofline, arithmetic-intensity, expert-parallel]
related:
  - 40-kernels/moe-kernels/02-grouped-gemm
  - 40-kernels/moe-kernels/03-permute-and-unpermute
  - 80-model-architectures/moe
  - 40-kernels/softmax-normalization/01-softmax-stability
  - 40-kernels/gemm/05-persistent-kernel-and-tcgen05
code:
  - code/01_moe_routing.cu
status: stable
updated: 2026-09-16
---

# MoE 的计算形态

> MoE 一层的 FLOPs 是 `4 · tokens · top_k · d_model · d_ff`——**与专家总数 E 无关**，只与 `top_k` 有关。但它的**权重流量**是 `E · 2 · d_model · d_ff · esize`，**与 token 数无关**。这两条合起来的意思是：MoE 省的是 FLOPs，不省显存、也不省带宽。

## 0. 背景与动机

沿用 [gemm/02](../gemm/02-shared-memory-tiling.md) 的对照场景思路，换成一个 FFN 层。本文的场景参数（**选定值，不是任何模型的规格**，见 `code/specs.py` 的 `UNSOURCED`）：

```text
d_model = 4096    d_ff = 14336（每个专家）    E = 8    top_k = 2    tokens = 4096    FP16
```

先看一个**等参数的稠密 FFN 层**：`FFN 参数 = 2 · d_model · d_ff_dense`。要让稠密模型的 FFN 参数等于 MoE 的 `E · 2 · d_model · d_ff`，必须取 `d_ff_dense = E · d_ff = 114688`。于是两者都是 **939,524,096 个参数**、都是 **1.8790 GB（FP16）**、都要把这一份权重从 HBM 过一遍（推导值，`make verify` §B1 / §B2）。

差别全在算的 FLOPs 上：

| | 权重 | 一层 FLOPs（4096 token） | 每 token FFN FLOPs |
|---|---|---|---|
| 稠密（`d_ff_dense = 114688`） | 1.8790 GB | **7.6966 TFLOPs** | 1879.048 MFLOPs |
| MoE（`E=8, top_k=2`） | 1.8790 GB | **1.9241 TFLOPs** | 469.762 MFLOPs |

（推导值，`make verify` §B1。）**同样的显存，四分之一的算力。** 这个 4 就是 `E / top_k = 8 / 2`。

这篇文档要解决的问题是：这个「省下来的 FLOPs」在 kernel 侧长什么样？它为什么不能直接用 gemm 主题里那几个内核算出来？以及——**省钱的那一半账，代价记在哪一栏**？

## 1. 原理

按五步推导。

**第一步：单 token 单专家是几个 GEMM。**

一个专家就是一个标准的两层 FFN，两个投影：

```text
up   : [d_model] × [d_model, d_ff]   → 2·d_model·d_ff FLOPs
down : [d_ff]    × [d_ff, d_model]   → 2·d_ff·d_model FLOPs
                                     ────────────────────────
                                     4·d_model·d_ff
```

代入场景：`4 × 4096 × 14336 = 234,881,024 FLOPs = 234.881 MFLOPs`（推导值，`make verify` §B1）。

**「两个」这件事值得单独记一笔**，因为它正是本域提纲初版出错的地方：只算 `up` 会得到系数 `2`，把一层的计算量**少算整整一半**。两类的对照：

| FFN 形态 | 系数 | 一层 FLOPs（本场景） |
|---|---|---|
| 只算 up（**漏了 down**） | 2 | 0.9621 TFLOPs |
| **up + down（标准两层）** | **4** | **1.9241 TFLOPs** |
| SwiGLU（gate + up + down 三个投影） | 6 | 2.8862 TFLOPs |

（推导值，`make verify` §B1；比值分别是 2.0× 与 1.5×。）

**第二步：MoE 层合计。**

一个 token 被路由到 `top_k` 个专家，每个专家对它算一遍完整的 FFN：

```text
MoE 层 FLOPs = tokens · top_k · (4·d_model·d_ff)
             = 4 · tokens · top_k · d_model · d_ff
```

代入：`4 × 4096 × 2 × 4096 × 14336 = 1.924145e12 FLOPs = 1.9241 TFLOPs`（推导值，`make verify` §B1）。

**这一步里没有 E。** 把 E 从 8 改成 64 或 256，这个数一个字都不变——`make verify` §B1 对 `E ∈ {1, 64, 256}` 各断言了一遍。这是 MoE 最容易被误读的一条：**「MoE 的参数多」与「MoE 的计算量大」是两件无关的事**。

**第三步：与等参数稠密模型比。**

```text
稠密：4 · tokens · d_model · d_ff_dense = 4 · tokens · d_model · (E·d_ff)
比值：E·d_ff / (top_k·d_ff) = E / top_k
```

代入 `8 / 2 = 4.0000`（推导值，`make verify` §B1）。这条式子是「MoE 用同样的参数换更少的 FLOPs」的完整内容，**比值只由 `E/top_k` 决定**。

**第四步：权重流量——省不下来的那一栏。**

上面三条都在说 FLOPs。现在数另一栏：

```text
专家权重字节 = E · 2 · d_model · d_ff · esize
             = 8 × 2 × 4096 × 14336 × 2 B = 1.879048e9 B = 1.8790 GB
```

`make verify` §B2 同时断言了「等参数稠密模型的权重字节数与它**恒等**」。所以：

> **MoE 省的是 FLOPs，不是显存，也不是带宽。** 全部 E 个专家的参数都要驻留显存，也都要在每层从 HBM 过一遍。

**第五步：把两栏并起来——算术强度与拐点批大小。**

权重流量的算术强度（每字节权重摊到的 FLOPs）：

```text
AI_MoE = 4·T·k·d_model·d_ff / (E·2·d_model·d_ff·esize)
       = 2·T·k / (E·esize)
```

`d_model` 与 `d_ff` **被约掉了**——与 [gemm/02 §1](../gemm/02-shared-memory-tiling.md) 里 `BK` 被约掉是同一类结果：它们不是算术强度的旋钮。

代入本场景：`2 × 4096 × 2 / (8 × 2) = 1024.0 FLOP/B`；等参数稠密模型是 `4096.0`，恒定是 MoE 的 `E/top_k` 倍（推导值，`make verify` §B2）。

H100 的 FP16 拐点 `I* = 989.5 TFLOPS ÷ 3350 GB/s = 295.37 FLOP/B`（规格值算出的推导值，`make verify` §B2）。本场景 `AI_MoE / I* = 3.467`——**越过拐点，是算力受限**。

把 `AI_MoE = I*` 反解：

```text
T* = I* · E · esize / (2·top_k) = 295.37 × 8 × 2 / 4 = 1181.5 个 token / 层
```

（推导值，`make verify` §B2。）批大小扫描：

| T | AI（FLOP/B） | AI / I* | 判定 |
|---|---|---|---|
| 128 | 32.0 | 0.1083 | 访存受限 |
| 512 | 128.0 | 0.4334 | 访存受限 |
| 1024 | 256.0 | 0.8667 | 访存受限 |
| 1181 | 295.2 | 0.9996 | 访存受限 |
| 1182 | 295.5 | 1.0004 | 算力受限 |
| 4096 | 1024.0 | 3.4668 | 算力受限 |
| 16384 | 4096.0 | 13.8672 | 算力受限 |

**`T*` 与专家数成正比。** 这一点比「FLOPs 与 E 无关」更重要，因为它是反方向的：

| E | 每层权重 | 一层 FLOPs（定 T=4096） | AI | T* |
|---|---|---|---|---|
| 2 | 0.4698 GB | 1.9241 TFLOPs | 4096.0 | 295 |
| **8** | **1.8790 GB** | **1.9241 TFLOPs** | **1024.0** | **1181** |
| 64 | 15.0324 GB | 1.9241 TFLOPs | 128.0 | 9452 |
| 256 | 60.1295 GB | 1.9241 TFLOPs | 32.0 | 37808 |

（推导值，`make verify` §B2。）**FLOPs 那一列纹丝不动，AI 与 T\* 却随 E 线性变化。** 一个 256 专家的 MoE 层要 37808 个 token 才做得到算力受限；低于它，这个 kernel 的唯一瓶颈就是「把 60 GB 专家权重从 HBM 搬一遍」。

作为对照，最朴素的形态（每个 `(token, 专家)` 对调一次 GEMV）的算术强度是 `4·d·d_ff / (2·d·d_ff·esize) = 2/esize = 1.0 FLOP/B`——`I*` 的 **0.34%**，而且**与 `d_model`、`d_ff` 都无关**（推导值，`make verify` §B2）。这是下一节「为什么必须把 M 攒起来」的全部理由。

## 2. 执行映射

MoE 层不是一个 GEMM，是**四个依次相接的 kernel**。下面逐个说它们在硬件上干什么，以及每一环的量级。

### 2.1 `routing`：gate GEMM + top-k + softmax

`logits = X[T, d] @ Wg[d, E]`，取每行的 top-k，再**在这 k 个 logit 上**做 softmax。

这是本层唯一「N 只有 8」的 GEMM。它的算术强度：

```text
FLOPs = 2·T·E·d = 2.684355e8；字节 = (T·d + E·d)·esize = 3.361997e7 B
AI_gate = 7.9844 FLOP/B = 0.02703 × I*（2.70%）
```

**7.9844 FLOP/B，是 `I*` 的 2.70%。** 所以 gate 的计算从来不重要，重要的是 **top-k 与 softmax 这两步的规约形态**：`E = 8` 个 logit 上的 argmax 是 warp 级的（`__shfl_xor_sync` 蝶形归约，5 步），而「取完最大把它屏蔽掉再取次大」这个循环要跑 `top_k` 轮。配套程序 `code/01_moe_routing.cu` 的 `routing_kernel` 就是「一个 warp 一个 token、lane e 负责专家 e」这个形态。

**权重口径必须说死**：本文采的口径是「先取 top-k，再在这 k 个 logit 上 softmax」，权重行和恒为 1。另有一条**看起来一样、实际不同**的写法：「在全部 E 个 logit 上做 softmax，取完 top-k **不重归一**」——它的权重行和**恒小于 1**，缺的量正是被丢掉的 `E−k` 个专家的 softmax 质量。`make verify` §A1 把这个缺口打了出来：本场景下（`E=8, top_k=2`、logit 服从标准正态）行和落在 `0.727951 ~ 0.999999`，均值 `0.955099`，**51.6% 的行和小于 0.99**。

顺带记一条等价性：`argsort(-logits)` 的稳定排序与「并列时取下标小的」是同一个口径；`make verify` §A1 用纯 Python 的参考实现逐行比过，64 行**零不一致**。

### 2.2 `permute`：从 token 序到专家序

每个 `(token, 专家)` 对一个 slot，slot 号 = `offset[专家] + 该专家内的排位`。三个动作：

1. **直方图**：数出 `tokens_per_expert`，用 `atomicAdd`（计数结果确定，到达顺序不确定）；
2. **排他前缀和**：把计数变成 `offset[e]`。这是标准的扫描，正文在 [reduce-scan/03](../reduce-scan/03-scan-algorithms.md)，此处只做代入；
3. **搬行**：把 `X[t]` 这一行（`d_model` 个**连续**元素）搬到 `Xp[slot]`。

第 3 步的访存模式是 [03 篇](03-permute-and-unpermute.md)的全部内容，这里只给一个数：本场景 permuted 缓冲区是 `8192 行 × 8192 B = 67.1 MB`，**是 H100 L2（50 MB，引用值）的 1.28 倍**——装不回 L2，只能老老实实走 HBM（推导值，`make verify` §B6）。

### 2.3 `grouped GEMM`：本层的主体

承接 [gemm/02](../gemm/02-shared-memory-tiling.md) 的分块配置（`BM=BN=128`、`BK=32`、FP32 32 KB SMEM/block），但有一个结构性的差别：**这个 GEMM 的 M 是分段的**。

每个 experts 的段长不同（本场景幂律路由下是 `[2997, 1574, 1003, 720, 608, 517, 410, 363]`，推导值，`make verify` §B4），于是网格不是「M/BM × N/BN」而是：

```text
网格 = Σ_e ceil(M_e / BM) · (N / BN) = 68 个 M-tile × 112 = 7616 个 block
```

（推导值，`make verify` §B5。）`7616` 个 block 分给 132 个 SM：每 SM 常驻 3 个 block 时是 **20 个波次、整机效率 96.16%**——这就是 MoE 的 GEMM 在「排满机器」这件事上的真实位置。下一节的 `2.4` 会说明，如果网格按专家而不是按 tile 来划，同一个数字会掉到 **2.02%**。

### 2.4 `unpermute`：回 token 序 + 加权求和

`out[t] = Σ_j w[t,j] · Yp[slot(t,j)]`。两个要点：

- **k 个贡献在寄存器里累加**，每个 token 只写一行（`T·d_model` 个元素），而不是用 k 次 `atomicAdd` 往同一行上撞；
- **必须有一个 mask**：slot 被截断的 `(token, 专家)` 对不能参与累加。这不是优化，是正确性——[03 篇 §4.1](03-permute-and-unpermute.md) 会把不做 mask 的后果跑出来。

**整个重排的流量**（permute + unpermute，`(1+k)` 读加 `(1+k)` 写）：

```text
2 · (1+top_k) · tokens · d_model · esize = 2×3×4096×4096×2 = 201.327 MB
```

**只占同层专家权重（1.8790 GB）的 10.71%**（推导值，`make verify` §B6）。所以重排不是 MoE 的瓶颈——**瓶颈从头到尾都是那 1.8790 GB 的专家权重**。

### 四个 kernel 的启动开销

每层四个 kernel，如果各自启动一次，按引用值 3~5 µs/次（`code/specs.py`，**引用值，本仓库不复核其测量口径**）就是每层 12~20 µs。对照本场景算力侧的时间下界 `1.9241 TFLOPs ÷ 989.5 TFLOPS = 1.9446 ms`（推导值，`make verify` §B2）——**占比 0.62%~1.03%**，不是主要矛盾。但层数一多（几十层 × 每层四次），把它压成 CUDA Graph 或融合 kernel 仍然值得。

## 3. 实现与演进

**朴素做法甲：把 MoE 展开成稠密。**

对每个 token 的每个选择都跑遍全部 E 个专家。FLOPs 变成 `4·T·k·E·d·d_ff`——**正好放大 E 倍**（推导值，`make verify` §C4：本场景 `1.539316e13 FLOPs`，`8.0×`）。这条路把 MoE 的全部收益抹掉，只换来「形状规整」。

**朴素做法乙：逐 token 逐专家调一次 GEMV。**

不展开成稠密，但也不重排，直接对每个 `(token, 专家)` 对算 `M=1` 的 GEMM。算术强度 `2/esize = 1.0 FLOP/B`（推导值，`make verify` §B2）——是 `I*` 的 **0.34%**。M=1 的 GEMM 在 roofline 上没有任何位置。

**主流：permute → grouped GEMM → unpermute。**

把 M 攒起来是唯一能提高 AI 的动作：AI 的表达式 `2·T·top_k/(E·esize)` 里，`esize` 由精度定死、`E` 由模型定死、`T` 由业务决定，**唯一能动的是「把同一个专家的 token 凑到一段里连续算」**。凑完之后：

- 每个专家的段长 `M_e` 不同 → 网格按 tile 划分，用前缀和表解码（[02 篇](02-grouped-gemm.md)）；
- 攒起来要搬数据 → 一次 permute 与一次 unpermute（[03 篇](03-permute-and-unpermute.md)）；
- 段长不均 → 短段多的方案白算算力，需要在「填充」与「分组」之间选。

**前沿：把三段合成一段。**

permute、grouped GEMM、unpermute 的边界是三个全局同步点。融合的方向（在 grouped GEMM 的 epilogue 里直接做加权回收、或者干脆按 token 分块让 slot 常驻 SMEM）都能省掉一次往返，但会重新引入「M 不规整」——这就是这个主题反复出现的那对张力。持久化调度（persistent kernel）的正文在 [gemm/05](../gemm/05-persistent-kernel-and-tcgen05.md)，此处回链。

**跨设备：permute 之后还要 all-to-all。**

上面四个 kernel 全是**单卡内的本地重排**。一旦专家分布在不同设备上，permute 之后的那一步就不只是本机 scatter，而是一次 all-to-all；**它的通信量分析不在本域**，见 `110-communication/collectives`（正文在彼，此处回链）。本文与 [02](02-grouped-gemm.md)、[03](03-permute-and-unpermute.md) 都只讲 kernel 侧的本地重排。

## 4. 踩坑与失败模式

### 4.1 把 MoE 当成稠密 GEMM 算：算力放大 = 专家数

**现象**：改成 MoE 之后模型的 FLOPs 账面降了 4 倍，端到端却变慢或者持平。

**成因**：kernel 侧仍然按稠密形状发网格——「每个 token 的 k 个选择送进全部 E 个专家」。FLOPs 按 `E` 放大（本场景 8×，`make verify` §C4），而那 1.8790 GB 的权重照样要全读一遍——**多算的 7/8 没有任何产出**。更隐蔽的子形态是「算了但没要」：某个 slot 被容量截断丢掉了，但它落在某个 tile 里，grouped GEMM 照样把它算完，到 unpermute 才被 mask 掉。这份算力既没产出、也没省下。

**规避**：网格必须按 `(专家, m_tile, n_tile)` 三元组发，而不是按 token 或按专家发。判据很硬：**网格里 block 的个数应当约等于「有用行数 ÷ BM × N-tile 数」，而不是 `tokens × top_k × E` 的任何倍数**。

### 4.2 容量截断后 `tokens_per_expert` 极不均

**现象**：给每个专家开了相同数量的 slot，结果少数专家溢出、大量专家空着；调大 slot 数则算力浪费，调小则丢 token 影响质量。

**成因**：`tokens_per_expert` 的均值由 `tokens × top_k / E` 定死，但**形状**由路由分布决定，两者是两回事。同样 8192 个 slot、同样均值 1024：

| 分布 | 最小 | 最大 | 标准差 | max/mean | 「按最大填充」的浪费比 |
|---|---|---|---|---|---|
| 均匀 | 973 | 1069 | 32.84 | 1.04 | **4.21%** |
| 幂律 α=1 | 363 | 2997 | 831.57 | 2.93 | **65.83%** |

（推导值，`make verify` §B3；`E=8, tokens=4096, top_k=2`。均匀分布的抽样标准差 32.84 与二项理论值 `sqrt(8192 × 1/8 × 7/8) = 29.9333` 同量级。）

极端情形（90% 的 token 第一选择是专家 0）下，`tokens_per_expert` 是 `[3784, 609, 636, 641, 643, 613, 615, 651]`——**按最大填充的浪费比 72.94%**；而若把每个专家的 slot 数截断到均值 1024，被丢掉的 slot 占 **33.69%**，对应这一层 **0.6483 TFLOPs** 算了但不参与输出（推导值，`make verify` §C1）。注意截断只波及 **8 个专家里的 1 个**——**不均体现为「少数专家溢出」，不是「大家一起溢出」**，这决定了治理手段（对齐热点专家 vs 全局扩容量）完全不同。

**规避**：把「填充」与「截断」当成两个独立的旋钮，分别记账。「槽位数 C 该取多少」是架构层的问题（正文在 `80-model-architectures/moe/02-capacity-factor.md` 的 §1.6，此处回链）；kernel 侧的责任是让 C 一旦定下来，**哪些 slot 要 mask、被丢的 token 在 unpermute 时怎么处理**这两件事有确定的说法（[03 篇 §2](03-permute-and-unpermute.md)）。

### 4.3 记账时漏掉 down 投影

**现象**：按「MoE 一层的 FLOPs」做选型，结论是 MoE 比等参数稠密模型省 8 倍（而不是 4 倍）。

**成因**：只数了 up 投影，系数写成 2 而不是 4。`make verify` §B1 把这个差别单独打了出来：系数 2 给 `0.9621 TFLOPs`、系数 4 给 `1.9241 TFLOPs`，**差整整 2 倍**；再叠上 `E/top_k` 的 4 倍，选型结论就会偏向 8 倍。

**规避**：把 FFN 的投影数写进公式里——两层 FFN 是 2 个投影（系数 4），SwiGLU 是 3 个（系数 6）。这条式子和 `d_model`、`d_ff` 的具体取值无关，**是一个只由 FFN 结构决定的常数**，值得在代码里写成具名常量而不是散落在注释里。

## 编译与运行

配套程序是 **`code/01_moe_routing.cu`**（routing kernel：一个 warp 一个 token，算 gate logits → warp 级 top-k → 在 k 个上 softmax；外加主机侧的 FLOPs / 权重流量 / 拐点批大小 / 路由分布四张表）：

```bash
cd code
make 01_moe_routing
./01_moe_routing
./01_moe_routing 256     # 改 kernel 演示用的 token 数
```

程序会把本文 §1 的五步推导、§2.1 的权重口径、§4.2 的分布表按运行时的参数重算并打印。**注意其中两类数字的性质不同**：

- FLOPs、权重字节、算术强度、拐点批大小、时间下界、分布统计——都是**推导值**（纯算术），可用 `make verify` 的 §B1 / §B2 / §B3 复核；
- §2.1 那几行 `top-k` 选择与权重是 **kernel 的运行结果**，本仓库没有 GPU，**没有跑过**，不对其数值负责。

本机（macOS，无 CUDA 工具链）只能跑：

```bash
make verify
```

对应本文的是 §A1（路由的两条权重口径）、§A2（端到端等价）、§B1（FLOPs 三个系数与等参数对照）、§B2（权重流量、算术强度、拐点批大小）、§B3（两种分布）、§C1（极端不均）、§C4（当成稠密算的放大比）。

## 6. 小结与延伸

本文一条主线：**MoE 一层的 FLOPs 是 `4·tokens·top_k·d_model·d_ff`，与 E 无关；而它的权重流量是 `E·2·d_model·d_ff·esize`，与 token 数无关。** 围绕它有三个可独立复用的中间结论：

1. **省的是 FLOPs，不是显存也不是带宽。** 等参数稠密模型与 MoE 的权重字节数恒等（都是 1.8790 GB），每 token 的 FLOPs 之比恒为 `E/top_k`。
2. **权重流量的算术强度 `2·T·top_k/(E·esize)` 里，`d_model` 与 `d_ff` 被约掉了。** 拐点批大小 `T* = I*·E·esize/(2·top_k)` 与 E 成正比——E 越大越难算力受限。本场景（E=8）是 1181.5 个 token/层。
3. **M 必须攒起来。** 逐对调 GEMV 的算术强度是 `2/esize = 1.0 FLOP/B`，是 `I*` 的 0.34%；permute 把 M 攒成段之后才有 1024.0 FLOP/B。攒的动作本身只花 201.327 MB，是同层权重流量的 10.71%——**重排不是瓶颈，它是为了让瓶颈那 1.8790 GB 能被算力消化掉而付的过路费**。

接下来两篇各有分工。[02-grouped-gemm](02-grouped-gemm.md) 接着本文 §2.3 往下走一层：既然 M 是分段的，网格怎么发、tile 怎么排、短段怎么不白算——那里会交代「按最大长度填充」与「按实际长度分组」的算力浪费比（本场景幂律路由下 **2.82 倍**），以及专家数跨过 SM 数时的波次量化损失。[03-permute-and-unpermute](03-permute-and-unpermute.md) 接着 §2.2 与 §2.4：走一遍 permute / unpermute 的访存模式，把「行级 scatter 该怎么映射 lane」和「被丢的 slot 在 unpermute 时怎么处理」讲透。

## 参考

- NVIDIA, *CUDA C++ Programming Guide* —— §5.1「Compute Capabilities」（每 SM 的共享内存容量、寄存器数、常驻线程/block 上限）；全局访存的 32 字节 sector 粒度。
- NVIDIA, *NVIDIA H100 Tensor Core GPU Architecture Whitepaper* —— 每 SM 统一 L1/SMEM 池 256 KB、至多 228 KB 可切给 SMEM（本文沿用 `code/specs.py` 的 `SMEM_PER_SM_MAX_KB = 228`）。
- NVIDIA, *NVIDIA Hopper Architecture In-Depth* —— 表3（132 SM / HBM3 3350 GB/s / FP16 tensor core 989.5 TFLOPS dense）。
- 本仓库 [10-foundations/performance-models/01-roofline-model](../../10-foundations/performance-models/01-roofline-model.md) —— `I*` 的定义与「算术强度决定可达哪一侧」。
- 本仓库 [40-kernels/gemm/02-shared-memory-tiling](../gemm/02-shared-memory-tiling.md) —— 分块的三笔账；本文 §2.3 直接沿用它的 tile 配置。
- 本仓库 [40-kernels/reduce-scan/01-reduction-algorithms](../reduce-scan/01-reduction-algorithms.md) —— segmented / grouped 归约的正文（`tokens_per_expert` 是它的一个实例）。
- 本仓库 [40-kernels/reduce-scan/03-scan-algorithms](../reduce-scan/03-scan-algorithms.md) —— 排他前缀和的并行算法（本文 §2.2 的第二步只做代入）。
- 本仓库 `code/verify.py` 的 §A1、§A2、§B1、§B2、§B3、§B6、§C1、§C4 —— 本文全部数字的来源。
- 本仓库 `code/specs.py` —— H100 的规格常量与它们的一手来源；其中的 `UNSOURCED` 列出了本主题明确不引用的项。

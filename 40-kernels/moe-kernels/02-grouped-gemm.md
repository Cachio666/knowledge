---
title: grouped GEMM 与 tile 调度
slug: grouped-gemm
domain: 40-kernels
topic: moe-kernels
level: 3
difficulty: intermediate
prerequisites:
  - 40-kernels/moe-kernels/01-moe-compute-pattern
  - 40-kernels/gemm/02-shared-memory-tiling
tags: [moe, grouped-gemm, variable-length, tile-scheduling, wave-quantization, persistent-kernel]
related:
  - 40-kernels/gemm/05-persistent-kernel-and-tcgen05
  - 40-kernels/gemm/06-split-k-and-stream-k
  - 40-kernels/reduce-scan/03-scan-algorithms
  - 40-kernels/moe-kernels/03-permute-and-unpermute
  - 20-gpu-architecture/memory-system/03-shared-memory-bank-conflict
code:
  - code/02_grouped_gemm.cu
status: stable
updated: 2026-09-16
---

# grouped GEMM 与 tile 调度

> MoE 的 GEMM 只在一个方向上不规整：**M**。N（= `d_ff`）与 K（= `d_model`）对所有专家完全相同，而每个专家的 `M_e` 由路由决定。于是问题收敛成一句话：**M 是分段的，段长不等。** 按最大段填充要白算 **2.82 倍**（幂律路由下），按实际长度分组快到 1.06 倍——但分组不是免费的，每段末尾仍要补齐到 `BM` 的整数倍。

## 0. 背景与动机

[01 篇](01-moe-compute-pattern.md)把 MoE 一层的账算清了：`4·tokens·top_k·d_model·d_ff = 1.9241 TFLOPs`，而权重流量是 `1.8790 GB`。同篇 §2.3 把主体交给了一个 GEMM——但那个 GEMM 与 [gemm/02](../gemm/02-shared-memory-tiling.md) 里那个不一样。

差别只有一条。gemm/02 的内核假设 `M` 是 `BM` 的整数倍（或者差得不多），于是它可以放心地发 `(M/BM)·(N/BN)` 个 block。MoE 的 permute 之后，`M` 变成了 **`E` 段，每段长度不等**：

```text
均匀路由：tokens_per_expert = [1069, 1013, 973, 1046, 1031, 986, 1065, 1009]
幂律路由：tokens_per_expert = [2997, 1574, 1003, 720, 608, 517, 410, 363]
```

（推导值，`make verify` §B4；`tokens=4096, top_k=2, E=8`。两组的**和都是 8192**——均值固定、形状自由。）

`M_max` 是 1069 还是 2997，取决于路由分布；而 `ΣM_e = 8192` 恒定。**同一个 GEMM，同样的 FLOPs 下限，网格却要按分布来发。** 本文把两条路线的账都算出来，然后给出 tile 级调度该怎么做。

## 1. 原理

按六步推导。

**第一步：`M_e` 的自由度。**

一个专家的段长由路由决定：`M_e = #{token : 专家 e ∈ top-k(token)}`。它的**期望**是 `tokens·top_k/E = 1024`，但实际值随分布浮动：

| 分布 | 最小 | 最大 | 标准差 | max/mean |
|---|---|---|---|---|
| 均匀 | 973 | 1069 | 32.84 | 1.04 |
| 幂律 α=1 | 363 | 2997 | 831.57 | 2.93 |

（推导值，`make verify` §B3。）**均值一样，形状差两个数量级。**

**第二步：tile 划分。**

沿用 [gemm/02](../gemm/02-shared-memory-tiling.md) 的 `BM = BN = 128`。每个专家的段要占 `ceil(M_e / BM)` 个 M-tile：

```text
均匀：[9, 8, 8, 9, 9, 8, 9, 8]     共 68 个 M-tile
幂律：[24, 13, 8, 6, 5, 5, 4, 3]   共 68 个 M-tile
```

（推导值，`make verify` §B4。）**巧合的是两者都恰好 68 个 M-tile**——因为 tile 数是「向上取整」，而两个分布的 `ΣM_e` 相同、最大值都在同一个量级上。这是个提醒：**tile 数不能只由 `ΣM_e` 推出来**，必须逐段算。

每个 M-tile 还要乘以 `N / BN = 14336 / 128 = 112` 个 N-tile，才是 block 总数。

**第三步：两条路线的算力浪费比。**

给定分段，有两种发网格的方式：

- **甲（padded）**：每个专家都按最大的那个算。每个专家排 `ceil(M_max / BM)` 个 M-tile，网格 = `E · ceil(M_max/BM) · (N/BN)`。
- **乙（grouped）**：按实际长度分组。每个专家排 `ceil(M_e / BM)` 个，网格 = `Σ_e ceil(M_e/BM) · (N/BN)`。

两种方案**都必须把 M 取整到 `BM` 的整数倍**（甲方案的 `M_max` 也要先向上取整，否则比的是一个不存在的方案——见 §4.3）。算出来：

| 分布 | ΣM_e | M_max | 甲算的行数 | 乙算的行数 | 甲的浪费 | 乙的浪费 | 甲乙之比 |
|---|---|---|---|---|---|---|---|
| 均匀 | 8192 | 1069 | 9216 | 8704 | 11.11% | 5.88% | 1.06× |
| 幂律 α=1 | 8192 | 2997 | 24576 | 8704 | 66.67% | 5.88% | **2.82×** |

（推导值，`make verify` §B4。「浪费」= `1 − ΣM_e / 实际算的行数`。）

幂律那行是本文的核心结论：**甲方案多算了 `24576 − 8704 = 15872` 行，是乙方案的 2.82 倍算力**，而它多算的那些行**全部来自最短的那几个专家**——甲给每个专家都排了 24 个 M-tile（3072 行），其中最少的专家只用了 363 行，也就是 **88.2% 的 tile 是空的**（推导值，`make verify` §B4）。

再看「按最大填充」的另一个口径（不管 tile 对齐、纯看容量利用率）：`1 − 均值/最大值`——均匀 4.21%、幂律 65.83%（推导值，`make verify` §B3）。两个口径的差（均匀 4.21% vs 11.11%）**完全来自 tile 取整**，下一节会说清。

**第四步：grouped 自己也不免费。**

`ceil` 这个动作本身就有代价：每段末尾不足 `BM` 的部分要补齐。

```text
均匀：补齐 512 行 = 6.25%（逐段 [83, 11, 51, 106, 121, 38, 87, 15]）
幂律：补齐 512 行 = 6.25%（逐段 [75, 90, 21, 48, 32, 123, 102, 21]）
```

（推导值，`make verify` §B4。）均匀路由下 `M_e` 的均值恰好是 `1024 = 8×128`（整除），但抽样抖动让 **4/8 个段越过均值**、各多出整整一个 tile，于是白算 6.25% 的行。

> **这部分浪费与「路由均不均衡」无关，只与「`M_e` 不是 `BM` 的整数倍」有关。** 想让每段长度都落在 tile 边界上，得靠路由侧的负载均衡去凑——这是 kernel 侧无法单方面解决的问题。

**第五步：为什么只有 M 不规整。**

上面每一步都只在动 M。这是因为 MoE 的 FFN 结构决定了 **N 与 K 对所有专家完全相同**：

```text
up   : [M_e, d_model] × [d_model, d_ff]  → N = d_ff，   K = d_model
down : [M_e, d_ff]    × [d_ff, d_model]  → N = d_model，K = d_ff
```

专家之间**只有权重数值不同，形状一模一样**。所以「变长 GEMM」这个问题的全部内容是 M 维的分段，而 M 维恰好是最容易切的那一维——它不需要任何跨段的归约（对比 [reduce-scan/01 §1.2](../reduce-scan/01-reduction-algorithms.md) 的 segmented 归约：那里的段边界不对齐 warp 会**串行化**，这里不会，因为每个 block 只碰一段）。

**第六步：网格大小的两个极端。**

把两步乘起来，网格 = `(M-tile 数) × (N/BN)`。本场景：

```text
乙（分组）：68 个 M-tile × 112 = 7616 个 block
甲（填充）：均匀 72 个 M-tile × 112 = 8064；幂律 192 × 112 = 21504
```

7616 个 block、每个 256 线程、每 SM 常驻 3 个 → 132 SM 上跑 **20 个波次、整机效率 96.16%**（推导值，`make verify` §B5）。

但如果网格**按专家**而不是按 tile 来发，同一个嵌套只有 8 个 block，只能填满 `8/396 = 2.02%` 的 block 槽位。**网格怎么划，比填充策略重要得多**——这是 §2 的主题。

## 2. 执行映射

### 2.1 blockIdx 怎么解码回 `(专家, m_tile, n_tile)`

分组方案的核心是一张**前缀和表**：

```text
tile_base[e] = 前 e 个专家一共占了多少个 block 的**排他前缀和**
```

每个 block 拿自己的 `blockIdx.x` 在这张表上做一次查找，就得到「我属于哪个专家、哪个 tile」。`E` 通常只有几十到几百，线性扫描足够（配套程序 `code/02_grouped_gemm.cu` 的 `grouped_gemm_kernel` 就是这么写的）；`E` 上千时换成二分。

```text
b = blockIdx.x
e = 0;  while (b >= tile_base[e+1]) ++e;        // 找到专家
local = b - tile_base[e]                        // 在该专家内的 tile 序号
mt = local / (N/BN);  nt = local % (N/BN)
```

之后就是 [gemm/02](../gemm/02-shared-memory-tiling.md) 那套：把 A 的 `BM×BK` 与 B 的 `BK×BN` 搬进 SMEM，`__syncthreads()`，寄存器小块累加。

**前缀和是这个方案唯一的额外数据结构**（并行扫描的正文在 [reduce-scan/03](../reduce-scan/03-scan-algorithms.md)，此处只做代入）。它的正确性门槛很低但必须过：`make verify` §A4 断言 `offsets` 的排他前缀和与 `numpy.cumsum` 逐元素相等、且 `offset[E−1] + M_{E−1} = ΣM_e`。

### 2.2 分段的边界：谓词，不是分支

段的最后一块 tile 通常只有一半有效行。处理方式是**谓词**而不是提前 `return`：

- 载入 A 时 `r < m_avail` 为假就写 `0.0f`（SMEM 里补零）；
- 写回 C 时 `row < m_avail` 为假就跳过。

这条不改算法，但它决定了**一个 block 里的 256 个线程是否会发散**。`code/02_grouped_gemm.cu` 的 `gemm_tile_body` 就是按这个写的：整个 K 循环里没有一处线程级分支，谓词只在载入与写回时出现。

顺带说一个容易漏的边界：**`M_e = 0` 的段**。当 `E` 远大于 `tokens×top_k` 时，很多专家一个 token 都分不到——`make verify` §C5 用 `tokens=16, top_k=2, E=64` 跑出 **37/64 个空段（57.8%）**。`ceil(0/BM) = 0`，所以分组方案里它们**不占 block**（全家 27 个 M-tile，而按最大填充要排 64 个）；但 `0/BM` 之类的整数除法要保证不出 NaN／除零，空段也必须能被安全跳过（`make verify` §A3 用一个含两个空段的例子断言：分组结果与逐专家单独调用逐元素一致）。

### 2.3 波次：网格划得对不对，一眼能看出来

| 网格映射 | block 数 | 波数（每 SM 常驻 1） | 最后一波占用 | 整机效率 |
|---|---|---|---|---|
| 一个**专家**一个 block，`E=8` | 8 | 1 | 8/132 | **6.06%** |
| 一个**专家**一个 block，`E=133` | 133 | 2 | 1/132 | **50.38%** |
| 一个**tile**一个 block（本场景） | 7616 | 58 | 92/132 | **99.48%** |

（推导值，`make verify` §B5。）

第一行是本场景：`E = 8`，一个专家一个 block 只能填满 **8/132** 的 SM。第二行是「专家数刚跨过 SM 数」的经典形状：`133 = 132 + 1`，第二个波次只有一个 block 有活干，整机效率拦腰砍半。**波次量化的机制与它的通用公式 `(1 − 最后一波占用率) / 波数` 正文在 [gemm/05](../gemm/05-persistent-kernel-and-tcgen05.md)，此处回链，不重推。**

第三行是解法：把网格从「按专家」改成「按 tile」，`7616` 个 block 把波数摊到 58，尾部只损失 0.52%。

**这里还有一个反直觉的细节**：常驻 block 数从 1 提到 3（SMEM 32 KB/block、8×8 线程块的寄存器约束都允许），槽位从 132 变成 396，波数从 58 降到 20，**但整机效率从 99.48% 掉到 96.16%**——因为尾部的绝对代价没变，而分母（波数）小了三倍。**常驻多是吞吐的收益，也是尾部的代价**，这两笔账要分开记。

**persistent 调度**（每个 block 循环领牌、而不是被预先分配）能把这个尾部再压到一个 tile 的粒度；它的正文同样在 [gemm/05](../gemm/05-persistent-kernel-and-tcgen05.md)，此处回链。

## 3. 实现与演进

**甲：按最大专家填充（`padded_gemm_kernel`）。**

把每个专家的段都拉到 `M_max` 行（不足的补零），网格 = `E · ceil(M_max/BM) · (N/BN)`。它的好处是**不需要任何额外的数据结构**：没有 `seg_len`、没有 `seg_off`、没有 `tile_base`，`blockIdx` 用一次整数除法就能解码。

代价就是 §1 第三步那张表：均匀路由下 1.06 倍、幂律路由下 **2.82 倍**的算力。这条路只有在「E 很大但分布很均匀」时才划算——而如果分布真那么均匀，`M_max/mean` 接近 1，两种方案的差别也就自然消失了。

**乙：按实际长度分组（`grouped_gemm_kernel`）。**

多一张前缀和表，换来的是「短段不白算」。这是主流实现（CUTLASS 的 grouped GEMM、各家的 MoE 推理内核）的做法，也是本文 §1、§2 描述的那个。

**丙：persistent / Stream-K 式的细粒度领牌。**

把 tile 当成「牌子」，block 常驻并循环领牌。它同时解决两件事：尾部的波次损失、以及「专家数少于 SM 数时机器填不满」。正文在 [gemm/05](../gemm/05-persistent-kernel-and-tcgen05.md) 与 [gemm/06](../gemm/06-split-k-and-stream-k.md)，此处回链。在 MoE 场景下它多一层价值：**tile 的代价本身就不均匀**（短段的 tile 与长段的 tile 行数相同、代价相同，但短段的 tile 更可能是「半个」），而持久化调度正好是按代价动态配平的手段。

**丁：N / K 也变长的情形。**

本文全程假设「专家之间只有 M 不同」。现实中还有两类异形：不同专家的 `d_ff` 不同（异构专家），或者不同层共享一部分专家。那时 `BK`、`BN` 也要按专家取值，前缀和的维度从两层变成三层。**本主题不覆盖这类结构**——它属于模型架构层的设计空间，正文在 `80-model-architectures/moe/02-capacity-factor.md` 的 §1.6 与 `80-model-architectures/moe/03-balance-and-granularity.md` 的「专家粒度」一节（正文在彼，此处回链；★ 后者写明**共享专家的规格字段在本环境没有一手来源**，因而**不在任何一域**）。

## 4. 踩坑与失败模式

### 4.1 按最大长度 padding 时短的专家浪费大量算力

**现象**：改成 MoE 之后权重流量没变、FLOPs 账面降了 4 倍，但这一层的耗时**远没有按 FLOPs 的比例降下来**。

**成因**：网格按 `M_max` 发了。本场景幂律路由下，甲方案要算 `24576` 行、乙方案只要 `8704` 行——**2.82 倍**。多出来的算力全部花在给最短的几个专家凑够 `M_max` 上：`363` 行的段被排了 `3072` 行，**88% 的 tile 是纯空转**。而更要命的是这不只影响时间：空 tile 的 block 照样要载入 B、照样要跑完 K 循环，SMEM、寄存器、发射槽位一样不少地占着。

**规避**：网格按 `(专家, m_tile, n_tile)` 发，不要按 `M_max` 发。判据是可判定的——**统计「实际算的行数 / ΣM_e」这个比值**，超过 1.1 就该换分组方案了。本场景均匀路由给 1.06、幂律给 2.82（推导值，`make verify` §B4）。

### 4.2 专家数超过 SM 数时的波次量化

**现象**：`E` 取某个特定值时整层效率莫名其妙掉一半；换个 `E` 又好了。

**成因**：网格按「一个专家一个 block」划的时候，网格大小就是 `E`，而机器的槽位数是 `SM × 常驻数`。`E` 刚跨过槽位数时，最后一个波次只有寥寥几个 block 有活干：

| `E` | 波数（槽位 132） | 最后一波 | 整机效率 |
|---|---|---|---|
| 132 | 1 | 132/132 | 100.00% |
| **133** | **2** | **1/132** | **50.38%** |
| 396 | 3 | 132/132 | 100.00% |
| **397** | **4** | **1/132** | **75.19%** |

（推导值，`make verify` §B5；每 SM 常驻 1 个 block。常驻 3 个时槽位是 396，`397` 那一行换成 2 波、效率 `50.13%`。）**注意效率在 `E` 上不是单调的**：`132` 是 100%、`133` 是 50.38%、`200` 是 75.76%——`E` 越大反而越好，因为尾部损失被摊到更多波次里。所以「`E` 取整数倍」这种调参直觉在这里是**反的**：整数倍处恰好是效率最高点，而它旁边一步就是最低点。

**规避**：网格不按专家划。改成按 tile 划之后，本场景网格从 `8` 变成 `7616`，波数从 1 变成 58，最后一波的占比从 94% 掉到 0.5%（推导值，`make verify` §B5）。波次量化的机制正文在 [gemm/05](../gemm/05-persistent-kernel-and-tcgen05.md)，此处回链。

### 4.3 padded 的 `M_max` 忘了对齐到 `BM`

**现象**：两种方案的对比表算出来「其实差不多」，于是选了更简单的 padded 方案，上线后才发现慢了好几倍。

**成因**：padded 方案的每个专家排 `ceil(M_max / BM)` 个 tile——**`M_max` 自己也要向上取整**。如果直接用 `E · M_max` 当「甲方案算的行数」，就是拿一个**不存在的方案**（行数不落在 tile 边界上）去和乙比：

| 「甲算的行数」怎么取 | 均匀路由下的值 | 对应的浪费比 |
|---|---|---|
| `E · M_max`（**不成立**，没对齐 tile） | 8552 | 4.21% |
| `E · ceil(M_max/BM) · BM`（真实） | 9216 | **11.11%** |

（推导值，`make verify` §B3 与 §B4 两个口径的差。）差了 7 个百分点，而这个差**恰好等于 tile 取整在多算的那一侧的放大**——乙方案要补 6.25%（§1 第四步），甲方案要在**每个**专家上补一个整 tile。

**规避**：两种方案的「算力」都要用**对齐后的行数**来表示。这是本仓库写作时实际踩到的一个坑：初版 `verify.py` 就是拿 `E · M_max` 去比 `Σceil(M_e/BM)·BM`，跑出来「均匀路由下 padded 只浪费 4.21%」，看了输出才发现比错了——`make verify` 的 §B4 现在把两列都印出来（`M_max` 与「padded 行」），就是为了让这个区别可见。

### 4.4 「空段」被当成错误处理

**现象**：`E` 开得比 `tokens×top_k` 大时，内核里出现 `0` 长度的段，代码写了 `M_e / BM` 的整数除法或 `max(1, ...)` 的兜底，结果要么越界、要么给空专家也排了 tile。

**成因**：`tokens×top_k` 个 slot 分给 `E` 个专家，`E` 大于 slot 数时**必然有专家一个 token 都分不到**。`make verify` §C5 的例子（`tokens=16, top_k=2, E=64`）里有 **37/64 = 57.8% 的专家是空的**。

**规避**：空段在分组方案里天然是「零个 tile」，不需要特判——`ceil(0/BM) = 0`，前缀和表里相邻两项相等，解码时那个区间里没有任何 `blockIdx`。要防的是**整数除法与 `max` 兜底**：`M_e/BM` 对 `M_e=0` 是合法的 `0`，但 `max(1, ...)` 会凭空造出一个 tile。另外，**这与 [reduce-scan/01 §1.2](../reduce-scan/01-reduction-algorithms.md) 的 segmented 归约不是一回事**：那里「迭代数由最长段决定」会拖慢所有段，而 grouped GEMM 的每个 block 只碰一段，空段只是不发牌。

## 编译与运行

配套程序是 **`code/02_grouped_gemm.cu`**（两个内核：`grouped_gemm_kernel` 用前缀和表解码 `(专家, m_tile, n_tile)`、`padded_gemm_kernel` 走 `M_max` 定距；外加 tile 划分与波次两张表）：

```bash
cd code
make 02_grouped_gemm
./02_grouped_gemm
./02_grouped_gemm 4     # 改 top_k
```

程序会把两个内核都跑一遍，并各自与一份 fp64 的逐专家参考比，然后打印 §1 的 tile 表、§2.3 的波次表。**注意其中两类数字的性质不同**：

- tile 数、padded/grouped 的行数、浪费比、网格大小、波次与整机效率——都是**推导值**（纯算术），可用 `make verify` 的 §B4 / §B5 复核；
- 「max|Δ| vs fp64 参考」是 **kernel 的运行结果**，本仓库没有 GPU，**没有跑过**，不对其数值负责。

程序自己那两张表用的是 `.cu` 里的 LCG 抽样，与 `verify.py` 的具体段长不同（不同的伪随机源）；**浪费比与网格大小的量级一致，正文引的是 `verify.py` 的数**。

本机（macOS，无 CUDA 工具链）只能跑：

```bash
make verify
```

对应本文的是 §A3（分组的分段 M 维 == 逐专家单独调用）、§A4（前缀和）、§B3 / §B4（tile 划分与两条路线的浪费比）、§B5（网格形状与波次）、§C1（极端不均）、§C5（空段）。

## 6. 小结与延伸

本文一条主线：**MoE 的 GEMM 只在 M 上不规整，而 M 恰好是最容易切的那一维——不需要跨段归约，每个 block 只碰一段。** 围绕它有三个可独立复用的中间结论：

1. **「按最大填充」的代价由 `M_max/mean` 决定，与总行数无关。** 均匀路由下只有 1.06 倍，幂律路由下 2.82 倍——而两种情况的 `ΣM_e` 都是 8192。所以判断该用哪种方案，看的是分布的形状，不是规模。
2. **分组方案自己也有 6.25% 的 tile 量化浪费，且与路由均衡与否无关。** 它只来自「`M_e` 不是 `BM` 的整数倍」；本场景均匀路由下 4/8 个段越过均值各多一个 tile。这一项要么靠路由侧的负载均衡去凑，要么接受它。
3. **网格按专家划还是按 tile 划，差别比填充策略大一个量级。** 本场景一个专家一个 block 只有 6.06% 的整机效率（`E=8`），按 tile 划是 99.48%——而波次量化的通用公式与它的机制在 [gemm/05](../gemm/05-persistent-kernel-and-tcgen05.md)，本文只做代入。

前一篇 [01-moe-compute-pattern](01-moe-compute-pattern.md) 解决了「为什么要做这个 GEMM、它值多少 FLOPs」；后一篇 [03-permute-and-unpermute](03-permute-and-unpermute.md) 解决「这些段的输入是怎么攒起来的、算完怎么还回去」——那里会交代 permute 的行级 scatter 该怎么映射 lane（合并与否差 **16 倍**），以及被截断的 slot 在 unpermute 时不做 mask 会错成什么样。

## 参考

- NVIDIA, *CUDA C++ Programming Guide* —— §5.1「Compute Capabilities」（每 SM 的共享内存容量、寄存器数、常驻 block 上限）；全局访存的 32 字节 sector 粒度。
- NVIDIA, *CUDA C++ Best Practices Guide* —— 共享内存分块与 padding 的实现要点。
- 本仓库 [40-kernels/gemm/02-shared-memory-tiling](../gemm/02-shared-memory-tiling.md) —— 本文的 tile 参数、SMEM 占用与 bank 冲突口径全部沿用该篇。
- 本仓库 [40-kernels/gemm/05-persistent-kernel-and-tcgen05](../gemm/05-persistent-kernel-and-tcgen05.md) —— 波次量化的机制与 `(1 − 最后一波占用率) / 波数` 的推导（本文只代入）。
- 本仓库 [40-kernels/gemm/06-split-k-and-stream-k](../gemm/06-split-k-and-stream-k.md) —— Stream-K 的工作划分（本文 §3 的丙）。
- 本仓库 [40-kernels/reduce-scan/01-reduction-algorithms](../reduce-scan/01-reduction-algorithms.md) —— segmented 归约为什么**不能**直接搬到 grouped GEMM 上（§4.4）。
- 本仓库 [40-kernels/reduce-scan/03-scan-algorithms](../reduce-scan/03-scan-algorithms.md) —— 排他前缀和的并行算法（本文 §2.1 只做代入）。
- 本仓库 [20-gpu-architecture/memory-system/03-shared-memory-bank-conflict](../../20-gpu-architecture/memory-system/03-shared-memory-bank-conflict.md) —— SMEM 行距与 bank 冲突（本文的内核沿用 `PAD=1`，机制回链该篇）。
- 本仓库 `code/verify.py` 的 §A3、§A4、§B3、§B4、§B5、§C1、§C5 —— 本文全部数字的来源。

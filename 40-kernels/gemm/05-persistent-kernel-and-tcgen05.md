---
title: persistent kernel 与 tcgen05/TMEM
slug: persistent-kernel-and-tcgen05
domain: 40-kernels
topic: gemm
level: 5
difficulty: advanced
prerequisites:
  - 40-kernels/gemm/04-tensor-core-and-async-pipeline
  - 20-gpu-architecture/tensor-core/04-tcgen05-and-tmem
  - 20-gpu-architecture/execution-model/03-occupancy-and-latency-hiding
tags: [gemm, persistent-kernel, tcgen05, tmem, wave-quantization, tile-scheduling]
related:
  - 40-kernels/gemm/06-split-k-and-stream-k
  - 20-gpu-architecture/execution-model/01-sm-internals
  - 20-gpu-architecture/architecture-evolution/03-invariants
code:
  - code/05_tcgen05_persistent_gemm.cu
status: stable
updated: 2026-09-16
---

# persistent kernel 与 tcgen05/TMEM

> persistent kernel 真正买到的不是「省掉启动开销」——那个开销只付一次、与网格大小无关。它买到的是**调度权**：尾部从「大半个波」压到「一个 tile」，代价是 TMEM 的 512 列把一个 CTA 能同时持有的累加器限死在 4 个以内。

## 0. 背景与动机

[04 篇](04-tensor-core-and-async-pipeline.md)把 FP16 的 tensor core 喂上了路，但留下一个结构性问题：Hopper 上「整个寄存器文件只装得下 4 个 warpgroup 的累加器」。加上多级流水要占的 SMEM，`BM=BN=128`、`BK=64`、4 级缓冲的配置把每个 SM 的使用推到了 `128 KB` SMEM——**每 SM 只能放 1 个 block**，占用率 12.5%。

这个时候，网格的组织方式开始变得重要。一个 `4096³` 的 FP16 GEMM 在 `BM=BN=128` 下是 **1024 个 tile**。它们怎么分配给 148 个 SM（B200），有两种截然不同的做法：

**静态分牌**：网格 = 1024 个 block，由硬件按块编号分配。看起来天经地义，但算一下波次（推导值，`make verify` §B10）：

| blocks/SM | 槽位 | 波数 | 最后一波占用 | 总时间损失 |
|---|---|---|---|---|
| 1 | 148 | 7 | 91.9% | **1.16%** |
| 2 | 296 | 4 | 45.9% | **13.51%** |
| 4 | 592 | 2 | 73.0% | **13.51%** |

第一行还算过得去。第二、三行就难看了——而且**方向是反的**：每个 SM 能放更多 block，损失反而更大。原因是「槽位数」相对 tile 数变大了：1024 个 tile 分给 296 个槽位只能分 3.46 波，最后一波填不满一半。

**persistent 发牌**：网格 = 恰好 148 个 block（或 148 的整数倍），每个 block 循环领号、算完再领，直到号发完。尾部被压到「每槽位最多 1 个 tile」——总时间的 `1/1024 = 0.098%`（推导值）。

本文做两件事：把这个「调度权」到底买到了什么算清楚（§1、§2），以及交代 Blackwell 上与之配套的另一次搬家——**累加器离开寄存器文件，进入 TMEM**（§1 第三、四步）。

## 1. 原理

按五步推导。

**第一步：启动开销不是主要理由。**

一次 kernel 启动的开销，外部测量在 **3~5 µs** 量级（**引用值，本仓库不复核其测量口径**——它随驱动版本、是否用 CUDA Graph、是否走 MPS 而变）。对 `4096³` FP16 在 B200 上，理想时间是 `1.3744e11 / 2250e12 = 61.1 µs`，这个开销占 **4.9%~8.2%**（推导值，`make verify` §B10）。

但这个数字与 persistent 关系不大：**启动开销只付一次，与网格里有多少 block 无关**。1024 个 block 的网格和 148 个 block 的网格，都只付一次。persistent 真正省掉的是**每个 block 的启动项**：SMEM 的分配与释放、流水线的首次填充延迟、参数的装载。这些在 K 大时占比很小（一次填充的延迟是几百个周期，而整个 block 要跑几万个周期），在 K 小、块数多时才显著。**把「省启动开销」当成 persistent 的主要理由是错的**——真正的理由是下面两步。

**第二步：调度权的价值——把尾部从「波」降到「tile」。**

静态分牌的尾部损失来自「最后一个波填不满」。损失的量级是

```text
损失 ≈ (1 − 最后一波占用率) / 波数
```

`make verify` §B10 的表就是这个式子逐行代入的结果。要压小它，静态分牌只有两条路：让槽位数与 tile 数整除（tile 数由问题形状定，往往是质数般的数），或者把 tile 切小（那会降低块级 AI，[02 篇](02-shared-memory-tiling.md) §1）。

persistent 发牌给的是第三条路：**每个 block 领下一个 tile，而不是被预先分配**。这样：

- 尾部只剩「最后一个未完成的 tile」，量级 `1/tile 总数`；
- 每个 block 的**实际速度**决定它领多少个 tile（快的多领），而不是块编号决定；
- 调度器可以在发牌时按 tile 的**代价**排序，应付不均匀的工作（ragged 的 M/N、grouped GEMM、split-K 的各个分段）。

第三条是 [06 篇](06-split-k-and-stream-k.md) 的基础：Stream-K 就是「persistent 发牌 + 把 K 也切进工作对」的产物。

**第三步：tcgen05 与 wgmma 的两处根本差异。**

Blackwell 的 `tcgen05` 在结构上做了两次改变（规格回链 [20-gpu-architecture/tensor-core/04](../../20-gpu-architecture/tensor-core/04-tcgen05-and-tmem.md)，本文不重推）：

| | `wgmma`（Hopper） | `tcgen05`（Blackwell） |
|---|---|---|
| 发射粒度 | **整个 warpgroup（128 线程）** | **1 个线程** |
| 累加器 D 的位置 | 寄存器 | **TMEM** |
| A 的位置 | 寄存器或 SMEM | TMEM 或 SMEM |
| B 的位置 | **必须 SMEM** | **必须 SMEM** |

两处差异各自解掉一个瓶颈。**单线程发射**省掉「4 个 warp 各发一次」的发射槽与一致性约束（`wgmma` 的 `.aligned` 要求 warpgroup 内所有线程执行同一条指令）。**D 在 TMEM** 则解掉 [03 篇](03-register-tiling-and-occupancy.md) §1 第五步那条 255 的墙——累加器不再与操作数、地址抢那 256 KB 的寄存器文件。

**第四步：TMEM 是 per CTA 的，这是最易错的一条。**

TMEM 的规格是：

```text
128 lane × 最多 512 列 × 每 cell 32 bit
128 × 512 × 4 B = 256 KB
```

**口径是 per CTA，不是 per SM。** 这条极易错，因为 256 KB 这个数值**恰好等于每 SM 的寄存器文件**——很容易顺手读成「每 SM 有一份 TMEM」。规格原文给的是「TMEM 是 128 lane × 至多 512 列的二维地址空间，列是动态分配的，分配粒度 32 列」。**它是另一份独立的存储**，不是寄存器的扩展。

由此得到一个硬约束。一个 `M = 128`、`N` 列的 fp32 累加器占 **N 列**（128 个 lane 各对应一行）：

| 累加器形状 | 占列数 | 512 列能容纳几个 |
|---|---|---|
| 128×64 | 64 | 8 |
| **128×128** | **128** | **4** |
| 128×256 | 256 | 2 |
| 128×512 | 512 | **1** |

（推导值，`make verify` §B10。）**一个 CTA 想同时持有多个累加器（多级流水、双缓冲的 D），最多只能有 4 个 128×128 的。** 这条约束与 SMEM 的 228 KB、寄存器的 65536 是同级的第三条预算。

一个 SM 上多个 CTA 如何**仲裁**这 512 列，本次采用的来源没有说明——本文只做 per CTA 的算术，**不把它当成 per SM 的硬上限**。

**第五步：把三条预算并起来。**

`BM=BN=128` 的一档配置下，三条预算各自允许的 CTA 并发数：

| 预算 | 算式 | 允许的 CTA 数 |
|---|---|---|
| 寄存器（03 篇） | `65536 / (每线程寄存器 × 128 线程)` | 取决于每线程寄存器数 |
| SMEM（02 篇） | `228 KB / (级数 × 每段字节)` | 4 级、32 KB/段时是 1 |
| **TMEM（本文）** | `512 列 / 每个累加器的列数` | 128×128 时是 **4** |

**真正的并发数是三者取最小。** 在 Hopper 式的深流水配置下，SMEM 通常先卡住（1 个 CTA/SM）；TMEM 的 4 更多时候是一个**上限保护**——它告诉你「就算 SMEM 和寄存器都够，一个 CTA 也别想同时持有超过 4 个 128×128 的累加器」。

## 2. 执行映射

### 2.1 发牌循环长什么样

persistent 内核的外层结构只有几行，但每一行都有对应的硬件代价：

```cuda
__device__ unsigned int g_tile_counter = 0u;     // 全局，host 侧清零

for (;;) {
    const unsigned int id = atomicAdd(&g_tile_counter, 1u);   // 领号
    if (id >= n_tiles) break;
    ...算第 id 个 tile...
}
```

- **一次全局 `atomicAdd`**：所有 block 对同一个地址做读改写。tile 数是 1024，块数是 148，总共只有 1024 次原子操作——相对于一个 block 的几万个周期，可以忽略。但如果 tile 切得很碎（`split-K` 的细分段、小的 grouped GEMM），原子操作的次数与竞争会变成可观的开销，那时改用「每 SM 一个计数器」或「一次性批量领 N 个」。
- **尾部**：`break` 之后 block 退出，SM 空出来给别的 block——但此时已经没有 tile 了。这就是 §1 第二步说的、被压到 `1/1024` 的那部分。
- **负载不均**：`atomicAdd` 只看「谁先到谁先领」，不看「谁快」。tile 代价均匀时这就够了；代价不均时要先按代价排序再发牌，否则快的 block 领了一堆重活、慢的领了一堆轻活，尾部又回来了。

### 2.2 tmgen05 侧的具体部件

| 部件 | 做什么 | 量级 |
|---|---|---|
| `tcgen05.alloc` / `dealloc` | 按列分配/释放 TMEM | 粒度 32 列，合法值 32/64/128/256/512 |
| `tcgen05.mma` | 执行 MMA，D 指向 TMEM | **由 1 个线程发射** |
| `tcgen05.commit` | 把之前的 MMA 标记为一组 | 配 mbarrier |
| `tcgen05.ld` / `st` | 在 TMEM 与寄存器之间搬数据 | **需要整个 warpgroup** |
| mbarrier | 完成通知 | 一次 commit 一个 |
| TMA | 装载 A/B 到 SMEM | 与 [04 篇](04-tensor-core-and-async-pipeline.md) 同 |

`tcgen05.ld` 需要整个 warpgroup 这一条来自 TMEM 的物理划分：128 个 lane 按 32 个一组分成 4 块，**一个 warp 只能访问本 warpgroup 内由 warp id 决定的那 32 个 lane**——要覆盖全部 128 lane 必须有整个 warpgroup（规格回链 [20-gpu-architecture/tensor-core/04](../../20-gpu-architecture/tensor-core/04-tcgen05-and-tmem.md)）。这与「发射只需要 1 个线程」并不矛盾：**发射是控制面，读回是数据面**，两者对参与线程数的要求本来就不同。

### 2.3 一条容易忽略的账：epilogue 变贵了

累加器住在寄存器里时，epilogue 是「寄存器 → 全局」一步。搬到 TMEM 之后多了一跳：**TMEM → 寄存器（`tcgen05.ld`）→ 全局**。这一跳的带宽约束来自 `tcgen05.ld` 本身，与 HBM 无关，但它增加了 epilogue 的指令数与延迟。在 K 很大、epilogue 占比很小的 GEMM 上无所谓；在 K 小、M/N 大的形状上（正是 [06 篇](06-split-k-and-stream-k.md) 的场景），这一跳的占比会明显上升。

## 3. 实现与演进

### 3.1 从「网格 = 问题」到「网格 = 机器」

| 代 | 网格怎么定 | 每个 block 做什么 | 尾部 |
|---|---|---|---|
| 传统 | `(M/BM)·(N/BN)` | 一个 tile | 最后一波填不满 |
| persistent | 常驻槽位数 | 领号循环，做多个 tile | 最后一个 tile |
| Stream-K（06 篇） | 常驻槽位数 | 领号循环，**工作对**是 (tile, k-chunk) | 最后一个工作对 |

这条演进的方向是「**让网格描述机器，而不是描述问题**」。一旦网格描述机器，调度就从「硬件按编号分配」变成了「内核自己按实际速度分配」——这才轮到「跨 tile 的调度」这类优化（把 K 也切进工作对、把同一条 A 边留给相邻的 tile）成为可能。

### 3.2 三次搬家的同一个方向

把 [04 篇](04-tensor-core-and-async-pipeline.md) §3.1 的表补完：

| 代 | 发射单位 | 累加器在哪 | transport 走哪 | **省下的是** |
|---|---|---|---|---|
| Ampere `mma.sync` | warp（32） | 寄存器 | 寄存器 | — |
| Hopper `wgmma` | warpgroup（128） | 寄存器 | `cp.async` / TMA | 操作数的寄存器占用 |
| Blackwell `tcgen05` | **1** | **TMEM** | TMA | **发射槽 + 累加器的寄存器占用** |

三代都在减少「通用资源的占用」，代价是**专用资源的份额一直在涨**：SMEM、然后 TMEM。这条线索在 [20-gpu-architecture/architecture-evolution/03](../../20-gpu-architecture/architecture-evolution/03-invariants.md) 里有更完整的形态（哪些量是不变量、哪些一直在动），本文不展开。

### 3.3 这条路走完之后

把三条预算（寄存器 256 KB、SMEM 228 KB、TMEM 256 KB）并列，会看到 Hopper/Blackwell 上 GEMM 内核的资源图景：**操作数走 SMEM，累加器走 TMEM，指令发射只需一个线程**。剩下的稀缺项是带宽——HBM 那一侧（[06 篇](06-split-k-and-stream-k.md) 处理并行度不足时怎么办）和 SMEM 那一侧（tile 大到 SMEM 装不下时怎么办）。

## 4. 踩坑与失败模式

### 4.1 persistent 下 tile 调度不均导致尾效应

**现象**：把网格改成 persistent 之后，某个形状上性能反而变差，或者同一形状的多次运行时间抖动很大。

**成因**：persistent 只把「发牌」这件事交给了内核，**没有自动让 tile 的代价变均匀**。当 tile 的代价不同（M/N 不能整除 `BM/BN` 时的 ragged tile、grouped GEMM 里长度差异巨大的专家、split-K 的各个分段），`atomicAdd` 的「先到先领」会退化成随机分配：快的 block 可能连领几个重 tile，慢的 block 早早把轻 tile 领完，尾部效应重新出现——而且比静态分牌更不可预测，因为分配结果依赖运行时的时序。

**规避**：发牌前把工作按**代价**排序（大的先发，或小步长交错），或者改用「静态划分 + 动态偷取」的混合方案。判据是看每个 block 处理的工作对数的分布——`make verify` §A3 的 Stream-K 划分断言打印的就是这个分布（`P=132` 时每 block 拿 1~2 个工作对，负载不均 ≤ 1 个）。如果分布不均，尾部一定回来。

### 4.2 TMEM 容量约束限制 CTA 并发数

**现象**：为了藏住延迟，想让每个 SM 同时跑 4 个以上的 CTA。SMEM 和寄存器都够，但并发上不去。

**成因**：TMEM 的 512 列是 **per CTA** 的容量，一个 `128×128` 的 fp32 累加器占 128 列——4 个这样的累加器就把 512 列用满（§1 第四步的表）。如果每个 CTA 需要「多个累加器」（比如多级流水的 D 缓冲、或者一个 CTA 同时算多个 tile），每个 CTA 占的列数还要乘上去，并发数进一步下降。这条约束**独立于** SMEM 与寄存器——它们三个是同级的预算，取最小。

**规避**：把 TMEM 列数当成第三条预算显式地算进来（就像算 SMEM 一样），在选 tile 形状时一并考虑：`N` 越小，一个累加器占的列越少、能并发的 CTA 越多。需要更多并发时，用小 `N` 的累加器（`128×64` 只占 64 列，能放 8 个）而不是硬堆 CTA 数。**注意不要把这条读成「每 SM 只有 256 KB TMEM」**——规格给的是 per CTA 口径，一个 SM 上多个 CTA 如何仲裁这 512 列，本次采用的来源没有说明。

### 4.3 在 persistent 内核里用「块编号」而不是「领号」

**现象**：为了避免原子操作的开销，persistent 内核里改成「第 `b` 个 block 处理第 `b, b+P, b+2P, ...` 个 tile」的静态轮转。看起来既有 persistent 的结构又省掉了原子操作，但尾部仍然存在。

**成因**：静态轮转等价于把「每个 block 分几个 tile」提前定死。当 tile 数不能被槽位数整除时（1024/148 = 6.92），前 136 个 block 拿 7 个、后 12 个拿 6 个——**尾部就是那 1 个 tile 的差距，与 §1 第二步说的量级一样**。问题不在于此：静态轮转在 tile 代价**均匀**时确实够了。它真正失效的地方是代价不均匀（§4.1）与「跨 tile 的复用」——静态轮转下相邻 tile 的处理顺序是固定的，调度器无法把「同一条 A 边」的 tile 排在一起。

**规避**：先判断 tile 代价是否均匀。均匀 → 静态轮转够用，省掉原子的竞争；不均匀、或要做跨 tile 调度 → 用领号。判据不是「原子操作慢不慢」（1024 次原子操作相对几万个周期可以忽略），而是「工作是否可均分」。

### 4.4 把 TMEM 的 256 KB 当成「每 SM 多了一块寄存器文件」

**现象**：看到「TMEM 256 KB、寄存器文件 256 KB」，得出「Blackwell 上每 SM 有 512 KB 的片上存储可用」，据此规划 tile 大小。

**成因**：两个 256 KB 是**两份独立的存储**，服务不同角色：寄存器文件服务线程的通用计算（操作数、地址、临时量），TMEM 服务矩阵累加器（`tcgen05` 的 D）。累加器不能放寄存器、通用临时量不能放 TMEM。把两者相加会得到一个「预算翻倍」的错误结论，而真实的预算是**两条各自独立的约束**。

**规避**：把三条预算（寄存器 256 KB、SMEM 228 KB、TMEM 256 KB）当成三个独立的门槛分别核对，别做加法。顺带记住 TMEM 的口径是 **per CTA**——这也是为什么把它写成「每 SM 的第三块存储」会误导。

## 编译与运行

配套程序是 **`code/05_tcgen05_persistent_gemm.cu`**（persistent 发牌循环 + `tcgen05.alloc`/`mma`/`commit`/`ld` + TMEM 容量与波次量化的算术）：

```bash
cd code
make blackwell        # 用显式的 -arch=sm_100a 构建（不混进默认 TARGETS）
./05_tcgen05_persistent_gemm 4096 4096 4096
```

**需要 B200（`sm_100a`）。在 H100（`sm_90a`）上编不过属预期**——`sm_100a` 与 `sm_90a` 是两个互不覆盖的加速档，一个的 cubin 跑不了另一个。

⚠ **本文件在写它的机器上没有编译过。** 本仓库的写作环境（macOS）没有 CUDA 工具链：`tcgen05` 的指令描述符位域按 PTX ISA §9.7.18 的表格写，TMEM 的分配/读回序列按同节给出的指令顺序组织。真机使用前请与 CUTLASS 的 SM100 实现交叉核对。程序内部对这一点有显式声明。

文档里的数字分两类：

- `I*(B200, FP16) = 292.2`、波次量化的损失表、TMEM 的列数与并发表、启动开销的占比——前四项是**推导值**（可用 `make verify` §B10 复核），启动开销 3~5 µs 是**引用值，本仓库不复核其测量口径**；
- 任何耗时与 TFLOPS 都是**那台机器上的运行结果**，本仓库一个都没有。

本机（macOS）只能跑：

```bash
make verify
```

对应本文的是 §B10（B200 拐点、波次量化、启动开销占比、TMEM 容量与并发表）与 §A3 后半段（Stream-K 的负载分布，为 [06 篇](06-split-k-and-stream-k.md) 铺垫）。

## 6. 小结与延伸

本文一条主线：**persistent kernel 买到的是调度权，而不是启动开销。** 围绕它有三个可独立复用的中间结论：

1. **波次量化的损失是 `(1 − 最后一波占用率) / 波数`。** `4096³`、`BM=BN=128`、B200 上，1 block/SM 时损失 1.16%，2 blocks/SM 时损失 13.51%——**每个 SM 能放更多 block 会让它更严重**，因为槽位相对 tile 数变大了。persistent 发牌把尾部压到 `1/1024 = 0.098%`。
2. **启动开销（引用值 3~5 µs）只付一次，与网格大小无关。** 它在 61.1 µs 的理想时间下占 4.9%~8.2%，但 persistent 省的不是它，是每 block 的启动项与尾部。
3. **TMEM 是 per CTA 的 128 lane × 512 列 × 32 bit = 256 KB，不是 per SM。** 一个 128×128 的 fp32 累加器占 128 列，所以一个 CTA 最多同时持有 4 个。它与 228 KB SMEM、65536 个寄存器是同级的第三条预算。

接下来 [06-split-k-and-stream-k](06-split-k-and-stream-k.md) 接住本文第三条结论的另一半：persistent 发牌让「跨 tile 的调度」成为可能，而 **Stream-K 就是把它用到底的产物**——把 K 也切进工作对，让每个 block 拿到的工作量精确相等。本文的 §1 第三步（`tcgen05` 单线程发射）与 §2.3（epilogue 变贵）在那篇会作为代价项出现。

## 参考

- NVIDIA, *PTX ISA* —— §9.7.18（`tcgen05.mma` 的发射模式、操作数位置、TMEM 的地址空间与分配粒度、`tcgen05.ld`/`st` 的线程要求、指令描述符表格）。
- NVIDIA, *NVIDIA Blackwell Architecture Technical Brief* —— 第五代 Tensor Core、TMEM 与 FP4/FP6 支持。
- NVIDIA, *CUDA C++ Programming Guide* —— §5.1（每 SM 的寄存器数、SMEM 上限、常驻线程与 block 上限）。
- 本仓库 [20-gpu-architecture/tensor-core/04-tcgen05-and-tmem](../../20-gpu-architecture/tensor-core/04-tcgen05-and-tmem.md) —— `tcgen05` 与 TMEM 的完整规格（本文只回链，不重推）。
- 本仓库 [20-gpu-architecture/execution-model/01-sm-internals](../../20-gpu-architecture/execution-model/01-sm-internals.md) §3 —— 「整个寄存器文件只装得下 4 个 warpgroup 的累加器」这笔账的推导。
- 本仓库 [20-gpu-architecture/architecture-evolution/03-invariants](../../20-gpu-architecture/architecture-evolution/03-invariants.md) —— 哪些每 SM 资源是跨代不变量、哪些一直在动。
- 本仓库 `code/verify.py` 的 §B10 与 §A3 —— 本文章节里的数字来源。

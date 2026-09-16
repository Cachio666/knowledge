---
title: 线程级寄存器分块与占用率
slug: register-tiling-and-occupancy
domain: 40-kernels
topic: gemm
level: 3
difficulty: advanced
prerequisites:
  - 40-kernels/gemm/02-shared-memory-tiling
  - 20-gpu-architecture/execution-model/01-sm-internals
  - 20-gpu-architecture/execution-model/03-occupancy-and-latency-hiding
tags: [gemm, register-tiling, thread-tile, occupancy, spill, double-buffering, l1-bandwidth]
related:
  - 40-kernels/gemm/04-tensor-core-and-async-pipeline
  - 20-gpu-architecture/memory-system/03-shared-memory-bank-conflict
code:
  - code/03_register_tiled_gemm.cu
status: stable
updated: 2026-09-16
---

# 线程级寄存器分块与占用率

> 共享内存的交付率是 128 B/周期，而 128 个 FP32 lane 每周期要 1024 字节操作数——差 8 倍。**寄存器复用不是优化，是必需品**；而 `TM=TN=8` 恰好是把这个缺口补平的最小尺寸，也是 255 个寄存器的上限允许的最大方形块。

## 0. 背景与动机

[02 篇](02-shared-memory-tiling.md)把 HBM 的重复搬运降到了 1/128，让 `128×128` 的 tile 拿到了 `AI_block = 32.0`——越过了 `I*(FP32) = 20.0`。按 roofline 的说法，这个内核现在应该能跑到 67 TFLOPS 的**封顶值**。

但它跑不到。把 SMEM 那一侧的账算一遍就会发现：一个 block 的 256 个线程，在 `for k` 的每一步都要从 SMEM 取操作数——每线程 `TM·TN` 次 FMA 配 `TM+TN` 次 SMEM 读。假设每个操作数都从 SMEM 现取（也就是 `TM=TN=1`），那么每做一次 FMA 就要读 8 个字节：

```text
128 个 lane 全速 → 128 × 8 B = 1024 B/周期
共享内存无冲突交付 → 128 B/周期
缺口 = 8 倍
```

也就是说，**即使 HBM 完全不再是瓶颈、SMEM 一次冲突都没有，这个内核最多也只能跑到 FP32 峰值的 12.5%**。SMEM 分块把上界从 1.25% 抬到了 12.5%，剩下那 8 倍得从别处拿。

拿法只有一条：**让每个从 SMEM 取出来的操作数被用很多次**。在寄存器里放一个 `TM×TN` 的小块 C，每取一次 `A` 的元素就用它做 `TN` 次乘法、每取一次 `B` 的元素就用它做 `TM` 次。本文要回答三个问题：这个 `TM×TN` 取多大（§1）、它怎么和 `__syncthreads()` 拼成流水线（§2）、以及它和寄存器预算/占用率之间的取舍（§1 末、§4）。

## 1. 原理

按六步推导。

**第一步：线程数由两个 tile 的比值决定。**

block 的 `BM×BN` 块被切给若干线程，每个线程负责 `TM×TN`：

```text
线程数 = (BM/TM) · (BN/TN)
```

`BM = BN = 128`、`TM = TN = 8`：`16 × 16 = 256` 个线程（8 个 warp）。
`BM = BN = 128`、`TM = 8, TN = 4`：`16 × 32 = 512` 个线程（16 个 warp）。

**第二步：每线程的寄存器估算。**

一个线程要同时持有三样东西：`TM·TN` 个 C 的累加器、`TM` 个 A 操作数、`TN` 个 B 操作数（后两者在 `for k` 内层里是「取一行、取一列」）：

```text
每线程寄存器 ≈ TM·TN + (TM + TN)
```

`TM = TN = 8`：`64 + 16 = 80`。其中 **C 占 64 个**，是绝对大头。

**这个估算不含寻址与索引寄存器**——SMEM 基址、行距、循环变量、谓词、`blockIdx/threadIdx` 的派生物都要占。真实编译时通常还要高 **30%~60%**（80 的估算落到 104~128 是常态）。所以「估算 80」不等于「实际 80」，占用率要用**编译出来的数**核对（`make regs` 会打印 `Used N registers`）。

**第三步：L1/SMEM 侧的平衡点。**

每个线程在一个 `k` 步里做 `TM·TN` 次 FMA、从 SMEM 读 `TM+TN` 个元素，所以每个 FMA 摊到的 SMEM 字节是

```text
每 FMA 的 SMEM 字节 = (TM + TN)/(TM·TN) · esize
```

FP32 下令它不超过 1 字节/FMA（128 B/周期 ÷ 128 个 lane）：

```text
(TM + TN)/(TM·TN) · 4 ≤ 1
⟹ (TM + TN)/(TM·TN) ≤ 0.25
⟹ TM·TN ≥ 4·(TM + TN)
```

`TM = TN = 8` 时左边 64、右边 64——**恰好取等**。代入全程：

| `TM×TN` | 每 FMA 的 SMEM 字节 | 128 lane 全速的需求 | 交付率 | 判定 |
|---|---|---|---|---|
| 4×4 | 2.0000 | 256 B/周期 | 128 | **不够** |
| 8×4 | 1.5000 | 192 B/周期 | 128 | **不够** |
| **8×8** | **1.0000** | **128.0 B/周期** | 128 | **恰好够** |
| 16×16 | 0.5000 | 64 B/周期 | 128 | 够（但见第五步） |

（推导值，`make verify` §C5 逐行打印。）这张表把「为什么经典配置是 8×8」回答得很干净：**它是在不超过 255 个寄存器这条约束下，刚好把 SMEM 交付率用满的最小方形块**。

**第四步：占用率。**

占用率由多道约束取最小决定（四条约束的完整口径见 [20-gpu-architecture/execution-model/03](../../20-gpu-architecture/execution-model/03-occupancy-and-latency-hiding.md)，本文只代入）。三条与本主题相关：

```text
每 SM 的 block 数 = min( ⌊65536 / (每线程寄存器 × 线程数)⌋,
                        ⌊228 KB / smem_per_block⌋,
                        ⌊2048 / 线程数⌋,  32 )
```

注意第一项里的**每线程寄存器要按 8 个一档向上取整**（分配粒度，见 [20-gpu-architecture/execution-model/01](../../20-gpu-architecture/execution-model/01-sm-internals.md) §1 第七步）。

代入 `BK = 32`、FP32（smem = `(BM+BN)·32·4`）：

| block / 线程块 | 线程数 | C 占 | 公式估算 | 上取整到 8 | blocks/SM | 常驻线程 | 占用率 |
|---|---|---|---|---|---|---|---|
| 128×128 / 8×8 | 256 | 64 | 80 | 80 | **3** | 768 | **37.5%** |
| 128×128 / 8×4 | 512 | 32 | 44 | 48 | **2** | 1024 | **50.0%** |
| 256×128 / 8×8 | 512 | 64 | 80 | 80 | **1** | 512 | **25.0%** |

（推导值，`make verify` §B5。第一行的 3 个 block 是寄存器约束（`65536/(80×256) = 3.2`）与 SMEM 约束（`233472/32768 = 7`）取较小者。）

**占用率对 tile 大小不是单调的。** 第二行（`8×4`，512 线程）拿到 50%，是第一行（`8×8`，256 线程）37.5% 的 1.33 倍——因为它的线程块更大、每线程寄存器更少，能塞下 2 个。看起来 `8×4` 更优，但第三步的表说它每个 FMA 要 1.5 字节 SMEM，**超过了交付率**。这就是「占用率高不等于性能高」的一个具体形态：`8×4` 的线程更多，但每个线程更饿。

**第五步：255 这条硬墙。**

`TM = TN = 16`：`C` 的累加器是 `16×16 = 256` 个寄存器，**光 C 就超过了每线程 255 的上限**——与寻址寄存器无关，必然 spill。加上 `TM+TN = 32` 个操作数，公式估算 288。

`make verify` §C1 按公式判定（**不是测硬件**）：

| `TM×TN` | C 占 | 公式估算 | 判定 |
|---|---|---|---|
| 8×4 | 32 | 44 | 不 spill |
| **8×8** | 64 | **80** | **不 spill** |
| 16×8 | 128 | 152 | 不 spill |
| **16×16** | **256** | **288** | **spill（光 C 就超 255）** |

**不要拿 `TM=TN=8` 当 spill 的反例**——它只有 80 个，离 255 差得远，而且正是第三步算出来的平衡点。能让累加器本身逼近上限的是 `16×16`。

**第六步：tile 大小与累加器容量的关系。**

把第三步与第五步合起来看，有一条容易被忽略的规律：**累加器占用按 tile 面积（`BM·BN`）增长，而算术强度只按边长（`BM`）增长**（02 篇 §1 第二步：方阵时 `AI_block = BM/esize`）。所以「把 tile 撑到拐点」的代价是面积级的：

| tile | 累加器（fp32） | 占寄存器文件（256 KB） |
|---|---|---|
| 128×128 | 64.0 KB | 0.25× |
| 256×128 | 128.0 KB | 0.50× |
| 256×256 | 256.0 KB | **1.00×** |
| 591×591 | 1364.4 KB | **5.33×** |

（推导值，`make verify` §C6。）最后一行是 FP16 的临界 tile（`I*·esize = 295.4 × 2 ≈ 591`）——**它的累加器是整块寄存器文件的 5.33 倍，装不下**。这条会直接把 [04 篇](04-tensor-core-and-async-pipeline.md) 的问题提出来：精度越快，拐点越靠右，而靠单块 tiling 走到拐点的代价按面积涨。

## 2. 执行映射

### 2.1 双缓冲：把 SMEM 读与计算重叠

一级分块的写法是「搬一段 → `__syncthreads()` → 算一段 → `__syncthreads()`」。这两次同步之间，整块 SM 实际上在**串行地**做两件事：等 HBM 数据到（400~800 周期量级），然后算。不重叠的代价可以直接估出来：

```text
每个 k 段的计算周期 = 2·BM·BN·BK / 256
                    = 2·128·128·32 / 256 = 4096 周期（BK=32）
一次 HBM 往返          ≈ 400~800 周期
未重叠时的停滞占比    ≈ 600 / (4096 + 600) ≈ 13%
```

`BK = 8` 时每段只有 1024 个计算周期，这个占比变成 **37%**——这就是 02 篇 §4.2 说的「`BK` 越小，这段等待占的比例越大」的定量形态。

双缓冲（double buffering）的做法是在 SMEM 里开**两份** tile：第 `s+1` 段在被搬进来的同时，第 `s` 段正在被计算。

```text
时间 →
搬运 S1 | 搬运 S2 | 搬运 S3 | ...
        计算 S1 | 计算 S2 | 计算 S3
```

于是那 13% 的停滞被压掉。代价是 **SMEM 占用翻倍**：`128×128×32` 从 32 KB 变成 64 KB。看 §1 第四步那张表的第一行——单缓冲时 SMEM 允许 7 个 block、寄存器只允许 3 个，**起作用的约束是寄存器**；双缓冲后 SMEM 允许 `⌊233472/65536⌋ = 3` 个，两条约束打平，实际仍是 3 个 block。**双缓冲在这一档是白赚的**。

`BK = 64` 就不是了：双缓冲 128 KB → SMEM 只允许 1 个 block，占用率从（寄存器限的）37.5% 掉到 12.5%。**`BK` 与缓冲级数共用同一个 SMEM 预算**，这句话在 04 篇会变成多级流水的核心取舍。

**双缓冲下 `__syncthreads()` 的语义要重新想清楚。** 每个迭代需要保证两件事同时成立：(a) 本段的数据已经写满，(b) 上一轮的读者已经全部退出、可以覆写。在一段固定只有两份缓冲时，一个 `__syncthreads()` 能同时满足两者（因为「本段的写」与「上一段的读」在同一区间内）。**先把一份 tile 的消费者与生产者隔开**是这类内核最常见的正确性 bug——症状是偶发的数值错误，且随 `BK`、`STAGES`、block 大小变化。

### 2.2 数值量级

| 量 | 数值 | 说明 |
|---|---|---|
| 每 SM 寄存器文件 | 65536 个 32-bit 寄存器 = 256 KB | 规格值 |
| 每线程寄存器上限 | 255 | 规格值 |
| 分配粒度 | 8 个一档，向上取整 | 80 恰好是 8 的倍数，不吃亏 |
| 每 SM FP32 lane | 128（Hopper 起） | 规格值 |
| SMEM 交付率 | 128 B/周期 | 无冲突时 |
| 本例的每 FMA 余量 | `TM=TN=8` 时 1.0000 B（需求恰等于交付） | 推导值 |

## 3. 实现与演进

### 3.1 三层 tile 的分工

真实 GEMM 内核里的 tile 是分三层嵌套的，每一层解决一个具体的搬运：

| 层 | 名字 | 装载到 | 谁来复用 | 解决的问题 |
|---|---|---|---|---|
| 1 | CTA tile / block tile（`BM×BN`） | 寄存器 + SMEM | 整个 block | HBM 流量（02 篇） |
| 2 | warp tile | 寄存器 | 一个 warp | SMEM 的**发射**次数（`ldmatrix` 一次装 4 个 8×8） |
| 3 | thread tile / register tile（`TM×TN`） | 寄存器 | 一个线程 | SMEM 的**交付字节**（本文） |

本文实现的是第 1、3 层（`BM×BN` 的 CTA tile + 每线程的 `TM×TN`）。第 2 层（warp tile）在引入 tensor core 之后才有独立意义——`ldmatrix` 与 `mma.sync` 的 fragment 天然定义了 warp 级的装载粒度，那部分正文在 [20-gpu-architecture/tensor-core/02](../../20-gpu-architecture/tensor-core/02-fragment-layout.md)，本域不重推。

### 3.2 线程块形状的第二个约束：SMEM 的读模式

`TM` 与 `TN` 不只是「复用次数」，它们还决定 SMEM 的**读地址模式**——也就是 02 篇 §3.1 那张 bank 冲突表里的地址分布。同一个 block tile、同一份 SMEM 布局，`TM×TN` 从 8×8 换成 8×4，读 B 的 lane 数是 32 而不是 16，冲突路数跟着变。

所以选 `TM×TN` 要同时看三件事：SMEM 交付率（第三步的表）、寄存器预算（第四步）、bank 冲突（02 篇 §3.1）。本文只把前两件算清楚——第三件依赖于具体的 SMEM 布局，`make verify` §B8 给的是「转置 B、行距 = BK+PAD」这一种布局下的数。

### 3.3 演进：从这里往哪走

线程级寄存器分块把 FP32 的通路用满了，但它有两个天花板：

1. **8 倍这个缺口是「FP32 lane 数（128）÷ SMEM 交付率（128 B/周期 ÷ 8 B）算出来的」**。想再往上走，只能减少「每个 FMA 需要几个寄存器操作数」——也就是换指令。`mma.sync` / `wgmma` 一次吃一整个 fragment，把「每 FMA 的 SMEM 字节」从 1.0 降到 `fragment 字节 / fragment 内 FLOP` 的量级。
2. **累加器占了寄存器预算的大头**（8×8 里 64/80）。tile 想更大，累加器就装不下——这是第五步那条 255 的墙，也是 Blackwell 把累加器搬进 TMEM 的动机（[05 篇](05-persistent-kernel-and-tcgen05.md)）。

两条都指向 [04 篇](04-tensor-core-and-async-pipeline.md)：换指令 + 让搬运不经寄存器。

## 4. 踩坑与失败模式

### 4.1 `TM=TN=16` 的线程块

**现象**：为了让每个 SMEM 元素被复用得更充分（`16×16` 时每 FMA 只要 0.5 字节 SMEM，比 `8×8` 的 1.0 还宽松一倍），把线程块改成 `16×16`。编译通过，耗时反而明显变长；`make regs` 里出现 `spill stores/loads` 与 `Used N registers` 顶到 255 的情况。

**成因**：`16×16` 的累加器本身是 **256 个 32-bit 寄存器，已经超过每线程 255 的上限**。编译器只能把一部分活跃变量溢出到 local memory——而 local memory 物理上在**片外**，每次 spill 是一次真实的全局访存，延迟比寄存器高两个数量级（HBM 往返 400~800 周期 vs 寄存器的一次读）。于是出现自相矛盾的结果：SMEM 侧的余量变宽了，但每个 FMA 的实际代价反而涨了。

**规避**：把公式估算当成**准入条件**而不是参考值：`TM·TN + (TM+TN) ≤ 255` 是必要条件，`16×16` 直接出局。要更大的 thread tile 只能改累加器的存储位置（TMEM），那是 05 篇的事。核对手段是 `make regs T=03_register_tiled_gemm`——**溢出的机制与后果**（为什么 local memory 是片外、为什么 spill 不可逆）正文在 [20-gpu-architecture/execution-model/01](../../20-gpu-architecture/execution-model/01-sm-internals.md) §1 第八步与 §4.3，此处只回链。

### 4.2 把公式估算当成实际寄存器数

**现象**：按 `TM·TN + (TM+TN) = 80` 算出「每 SM 能放 3 个 block、占用率 37.5%」，实际编译出来只有 2 个 block，占用率 25%，性能比预期差一截，找不出原因。

**成因**：80 这个数**不含寻址与索引寄存器**。SMEM 的基址与行距、循环变量、`blockIdx/threadIdx` 的派生物、谓词、以及 `for k` 展开后的多份地址都要占寄存器。真实用量通常在估算之上 30%~60%。按 +50% 算就是 120 个：

```text
⌊65536 / (120 × 256)⌋ = 2 个 block   （而估算给出的是 3 个）
```

同一张表里三组配置都会掉一档：`128×128/8×8` 从 37.5% 掉到 25%、`128×128/8×4` 从 50% 掉到 25%、`256×128/8×8` 保持 25%（`make verify` §B5 末段逐行打印）。

**规避**：占用率永远用**编译出来的寄存器数**算，不用公式估算。`make regs` 打印 `Used N registers`；`cudaOccupancyMaxActiveBlocksPerMultiprocessor` 在运行时直接给答案（02、03 号程序都调了它）。公式估算的用途是**筛选**——在写代码之前排除掉必然超限的 `TM×TN`，不是用来预告性能。

### 4.3 矩形线程块看起来更平衡

**现象**：`128×128` 的 block 配 `8×4` 的线程块，线程数 512、每线程只用 44 个寄存器、占用率 50%——三项指标都比 `8×8` 好。于是选它，结果反而更慢。

**成因**：`8×4` 的每 FMA SMEM 字节是 `(8+4)/(8×4)×4 = 1.5`，也就是 128 个 lane 全速时要求 **192 B/周期**，超过 SMEM 的 128 B/周期交付率（§1 第三步的表）。**占用率的提高被 L1 带宽的不足吃掉了**：线程更多，但每个线程要更频密地从 SMEM 取数，SMEM 端口先饱和。这就是「占用率高不等于性能高」——占用率衡量的是**能藏多少延迟**，不衡量**带宽够不够**。

**规避**：先判瓶颈在哪一侧，再调参数（判据见 [10-foundations/performance-models/01](../../10-foundations/performance-models/01-roofline-model.md) §4.3）。SMEM 交付率是硬约束，`(TM+TN)/(TM·TN)·esize ≤ 1` 这条不等式要在选型时就满足；方形块 `TM=TN` 之所以被推荐，是因为在给定面积下它让 `(TM+TN)/(TM·TN)` 最小（`4/4 = 2.0` vs `8/8 = 1.0`）。

### 4.4 只做双缓冲，不看每段的计算量够不够长

**现象**：加了双缓冲，HBM 延迟理论上被藏住了，但 `BK = 8` 时性能仍然很差。

**成因**：双缓冲能藏住延迟的前提是「**每一段的计算时间 ≥ 每一段的内存时间**」。`BK = 8`、`128×128` 的 tile 时，每段的计算只有 `2·128·128·8/256 = 1024` 个周期，而这一段的 8 KB 搬运在每 SM 分到的带宽下要 300 个周期量级——余量很薄。一旦 L2 未命中、或同一 SM 上有多个 block 抢带宽，计算段就被搬空，流水线出现气泡。更细的分段还意味着每段都要付一次 `__syncthreads()` 与地址计算，`BK = 8` 时 `K = 4096` 要付 512 次。

**规避**：双缓冲（或多级缓冲）与 `BK` 要一起选：**SMEM 总量固定时，`阶段数 × 每阶段字节` 是常数**，所以「`BK` 大 + 级数少」与「`BK` 小 + 级数多」是在同一个预算里选点。后者的流水线更细、更能吸收抖动，前者每段的计算更长、barrier 更少。工程上常见的落点是 `BK = 32~64`（FP16）配 3~5 级，而不是 `BK = 8` 配 2 级。核对手段是把 barrier 总数与每段计算周期都打印出来，看两者是否同量级。

## 编译与运行

配套程序是 **`code/03_register_tiled_gemm.cu`**（`TM×TN` 与 `STAGES` 都可由 `-D` 覆盖，另含 SMEM 带宽账与运行时占用率查询）：

```bash
cd code
make 03_register_tiled_gemm
./03_register_tiled_gemm 4096 4096 4096
make regs T=03_register_tiled_gemm    # 看寄存器用量与 spill
make 03_register_tiled_gemm NVCCFLAGS="-O3 -std=c++17 -DTM=16 -DTN=16"   # 会 spill
```

程序会打印：线程数、寄存器公式估算、每个 FMA 的 SMEM 字节与它和 128 B/周期的比较、SMEM 占用、以及运行时的占用率。**两类数字要分开看**：

- 线程数、寄存器估算、SMEM 带宽账、占用率的**公式部分**都是**推导值**，可用 `make verify` 的 §B5 / §C1 / §C5 复核；
- `Used N registers`、spill 计数、墙钟耗时是**编译与运行**的产物，本仓库没有工具链也没有 GPU，**这三类数字本仓库一个都没有**。

本机（macOS）只能跑：

```bash
make verify
```

对应本文的是 §B5（三组线程数/寄存器/占用率）、§C1（`TM=TN=16` 判 spill、`TM=TN=8` 判不 spill）、§C5（每 FMA 的 SMEM 字节表）、§C6（累加器容量表）。

## 6. 小结与延伸

本文一条主线：**SMEM 的 128 B/周期喂不饱 128 个 FP32 lane，差 8 倍；`TM×TN` 的寄存器小块是补这个缺口的唯一手段，而 8×8 恰好补平。** 围绕它有三个可独立复用的中间结论：

1. 线程数 `(BM/TM)(BN/TN)`，每线程寄存器 ≈ `TM·TN + (TM+TN)`（不含寻址，真实高 30~60%）。`TM=TN=8` 是经典配置：80 个寄存器不 spill，且恰好把 SMEM 交付率用满。
2. 占用率由多道约束取最小，本主题里起作用的是寄存器与 SMEM 两条；**占用率对 tile 大小不是单调的**——`8×4` 的 512 线程块拿到 50%，`8×8` 的 256 线程块只有 37.5%。但 `8×4` 的每 FMA SMEM 字节是 1.5，**带宽先饱和**。
3. `TM=TN=16` 的光累加器就 256 个寄存器，超过 255 的硬上限——**必然 spill，与寻址寄存器无关**。`make verify` §C1 用公式判定，不测硬件。

接下来两条延伸。[04-tensor-core-and-async-pipeline](04-tensor-core-and-async-pipeline.md) 从本文第三条结论出发：既然累加器阻塞了 tile 的增大，那就换指令——`mma` 的 fragment 一次吃一整块，把「每 FMA 的 SMEM 字节」再降一个量级；同时用 `cp.async` 让搬运不经寄存器，把双缓冲从「两个 buffer」扩成「多级流水」。[05-persistent-kernel-and-tcgen05](05-persistent-kernel-and-tcgen05.md) 则把累加器整个搬出寄存器文件（TMEM），从根上解掉 255 这条墙。

## 参考

- NVIDIA, *CUDA C++ Programming Guide* —— §5.1（每 SM 寄存器数、寄存器分配粒度、常驻线程/block 上限）；*CUDA C++ Best Practices Guide*（寄存器用量与占用率的核查方法、`__launch_bounds__` 的语义）。
- NVIDIA, *NVIDIA H100 Tensor Core GPU Architecture Whitepaper* —— 每 SM 128 个 FP32 lane、每 SM 统一 L1/SMEM 池。
- Volkov, *Understanding Latency Hiding on GPUs*（UC Berkeley 博士论文）—— 线程级分块与「每 FMA 的操作数字节」这类衡量方式。
- 本仓库 [20-gpu-architecture/execution-model/01-sm-internals](../../20-gpu-architecture/execution-model/01-sm-internals.md) —— 寄存器是预算不是缓存、分配粒度、spill 的机制与后果（本文只回链）。
- 本仓库 [20-gpu-architecture/execution-model/03-occupancy-and-latency-hiding](../../20-gpu-architecture/execution-model/03-occupancy-and-latency-hiding.md) —— 占用率的四条约束与 Little 定律口径。
- 本仓库 [20-gpu-architecture/memory-system/03-shared-memory-bank-conflict](../../20-gpu-architecture/memory-system/03-shared-memory-bank-conflict.md) —— 128 B/周期交付率与串行遍数的来源。
- 本仓库 `code/verify.py` 的 §B5、§C1、§C5、§C6 —— 本文四张表的数字来源。

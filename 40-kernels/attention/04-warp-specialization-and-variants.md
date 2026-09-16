---
title: warp specialization 与 attention 变体
slug: warp-specialization-and-variants
domain: 40-kernels
topic: attention
level: 4
difficulty: advanced
prerequisites:
  - 40-kernels/attention/02-flash-attention
  - 20-gpu-architecture/tensor-core/03-wgmma-and-pipeline
tags: [wgmma, warp-specialization, producer-consumer, tma, pipeline, sliding-window, alibi, softcap]
related:
  - 40-kernels/gemm/04-tensor-core-and-async-pipeline
  - 20-gpu-architecture/tensor-core/02-fragment-layout
  - 40-kernels/attention/05-decoding-attention
code:
  - code/04_flash_attention_ws.cu
status: stable
updated: 2026-09-16
---

# warp specialization 与 attention 变体

> 一句话：当 `O` 累加器吃掉每个线程 128 个寄存器时，「顺手搬个数据」就不再是免费的了——
> 把搬运交给另一个 warpgroup，是 Hopper 上把 wgmma 的发射槽喂满的自然选择。
> 而变体（滑窗 / ALiBi / softcap）只改**分块的范围**或**exp 的输入**，**不改结构**。

## 0. 背景与动机

[02 篇](02-flash-attention.md) 的 flash 前向有一个硬约束（`make verify` §B4，推导值）：

```text
O 累加器 = Br·d 个 fp32 ÷ 线程数
Br=128、d=128、只开 4 个 warp（128 线程）→ 每个线程 128 个寄存器
```

`Br=128`、`d=128`、256 线程时是 64 个寄存器/线程，加 S 的 fragment、`m`/`l` 与地址之后，
**一个 CTA 的寄存器预算已经用掉大头**。这时如果还让同一批 warp 去发 `cp.async`/TMA、
去算地址、去等 `mbarrier`，就有两个后果：

1. **发射槽被搬运指令占掉**。每 SM 每周期只有 4 个发射槽（规格值），
   而 wgmma 需要**连续发射**才能把 tensor core 喂满；
2. **寄存器的分配被打乱**。搬运用的地址寄存器与 `O` 累加器争同一份预算，
   编译器只能二选一地降级。

**具体对照场景**：`Br=128`、`Bc=128`、`d=128`、fp16、`stages=3` 的流水。

| 分工方式 | 谁发 wgmma | 谁搬数据 | 结果 |
|---|---|---|---|
| 单 warpgroup 全包 | 同一个 warpgroup | 同一个 warpgroup | 发射槽在搬运与 wgmma 之间来回切，流水有气泡 |
| **producer/consumer 双 warpgroup** | consumer | producer | wgmma 可以连续发射；代价是多 128 个线程与一份 barrier 开销 |

本文要讲清的是：**这个分工不是风格选择，而是被 ISA 逼出来的**，
以及分工之后代价落在哪里（SMEM 阶段数、寄存器、线程数）。

## 1. 原理

### 1.1 为什么必须分工：两条硬约束

**约束一：wgmma 的发射粒度是整个 warpgroup。** 规格值
（[tensor-core/03](../../20-gpu-architecture/tensor-core/03-wgmma-and-pipeline.md)，
本文只回链、不重推）：

- `wgmma` 的 `M` 恒为 **64**，`N ∈ [8, 256]` 步长 8（fp16/bf16 的 `K = 16`）；
- `B` 操作数**必须来自 shared memory**，`A` 可以在寄存器里；
- `.aligned` 限定符要求 **4 个连续 warp 一起执行同一条 wgmma 指令**。

最后一条决定了：**分工出去搬数据的 warp 不能与发 wgmma 的 warp 混在同一个 warpgroup 里**。
如果同一个 warpgroup 里一半 warp 去搬数据、一半发 wgmma，就是**未定义行为**——
这不是「慢一点」，是**写错了**。

**约束二：TMA 的发射粒度是 1 个线程。** 规格值：`cp.async.bulk`（TMA）
「一个线程即可发起整块多维搬运」，地址由 tensor map 给出，不占参与线程的寄存器。
这一条与约束一正好互补：

```text
wgmma：128 个线程必须一起动   ⟹ 计算的 warpgroup 要"整块"
TMA  ：1 个线程就能发起       ⟹ 搬运的 warpgroup 可以很"轻"
```

于是把两者分到不同的 warpgroup，**搬运那一侧几乎不占寄存器与发射槽**——
省下来的都归 `O` 累加器。

### 1.2 分工之后的 SMEM 账

producer 与 consumer 共享同一块 SMEM 池，流水 `stages` 级：

```text
SMEM = stages · (K tile + V tile) · esize = stages · 2·Bc·d·esize
```

`make verify` §B7 的四组配置（`d = 128`、fp16、上限 228 KB）：

| `Bc` | `stages` | SMEM | ≤ 228 KB | 每 SM 可驻留 CTA |
|---|---|---|---|---|
| 128 | 2 | 128 KB | 是 | 1 |
| 128 | 3 | 192 KB | 是 | 1 |
| **128** | **4** | **256 KB** | **否** | 0 |
| 64 | 4 | 128 KB | 是 | 1 |
| 64 | 6 | 192 KB | 是 | 1 |

**Bc=128 时最多 3 级流水**，4 级就是 256 KB，超过 228 KB 的硬上限。
想要更深的流水必须把 `Bc` 减半——而 `Bc` 减半会让每个 KV tile 的搬运摊薄变差。
**这是 flash 在 Hopper 上的第一个硬取舍。**

### 1.3 wgmma 指令数

`M` 恒为 64，于是指令数就是把三个维度分别除以各自的粒度：

```text
QKᵀ：Br×Bc×d → (Br/64)·(Bc/N)·(d/16) 条 .m64nNk16
PV ：Br×Bc×d → (Br/64)·(d/N)·(Bc/16) 条 .m64nNk16
```

`make verify` §B7 对 `Br=Bc=d=128` 逐项算了一遍：

```text
QKᵀ：2·1·8 = 16 条 .m64n128k16
PV ：2·1·8 = 16 条 .m64n128k16
```

每个 tile 32 条 wgmma，而每条覆盖 `M×N×K = 64×128×16` 的乘加。
**指令数少到可以忽略发射开销**——这正是 wgmma 相对 `mma.sync` 的意义：
把「每线程一条、每线程算 16×8」变成「每 warpgroup 一条、一次算 64×N」。

### 1.4 变体：只改结构，不改算法

三个常见变体，本文只讲**它们对 kernel 结构的影响**，定义不重推：

| 变体 | 改什么 | 对结构的影响 |
|---|---|---|
| sliding window（滑窗） | KV 的**可见范围** | 分块范围变窄：每行只需 `ceil(w/Bc)` 个块 |
| ALiBi | 在 `S` 上**加一项**与 `(i−j)` 成正比的偏置 | 分块形状**不变**，多一条逐元素指令 |
| softcap | 对 `S` 过一次 `tanh`（乘个系数） | 同上 |

**滑窗的收益是可算的**（`make verify` §B11，`Bc = 128`）：

| `S` | `w` | 全量块数 | 滑窗块数 | 降低 |
|---|---|---|---|---|
| 16384 | 4096 | 128 | 32 | 4× |
| 16384 | 1024 | 128 | 8 | 16× |
| 65536 | 4096 | 512 | 32 | 16× |
| 65536 | 1024 | 512 | 8 | **64×** |

窗口 = `S` 时没有收益（退化成全量），所以滑窗只在 `w < S` 时有意义。
因果掩码下平均还要再砍一半（第 `i` 行只看 `j ≤ i`）。

**三者对 kernel 结构的影响互不相同**，这一点最容易被混为一谈：

- 滑窗改的是**循环的范围**（`j_start`/`j_end`），`(m, l, O)` 的递推一个字都不用改——
  它甚至让 `O` 的累加次数变少，于是**重标定的次数也变少**；
- ALiBi / softcap 改的是**exp 的输入**。它们必须在 `rowmax` **之前**生效
  （否则 `m` 会对不上），但分块形状、SMEM 布局、wgmma 的形状全都不动。

> 也就是说：**滑窗是「少算几块」，ALiBi/softcap 是「每块多算一点」。**
> 前者影响并行度与负载均衡，后者只影响 FP32 向量通路的那一段。

## 2. 执行映射

### 分工的形态

```text
producer warpgroup（128 线程）
    只做 TMA：cp.async.bulk.tensor 把 K/V tile 搬进 SMEM
    然后 mbarrier_arrive（带 expect_tx，告诉消费者"这一块多少字节"）
    不参与任何计算，不持有 O 累加器

consumer warpgroup（128 线程）
    mbarrier_wait 等这一块到齐
    QKᵀ（wgmma，A=Q 寄存器/SMEM、B=K SMEM）
    → 在线 softmax + **O 的重标定**（FP32 向量通路，本文回链 02 篇）
    → PV（wgmma，A=P 寄存器 fragment、B=V SMEM）
    → 下一块
```

| 组件 | 作用 | 量级（规格值） |
|---|---|---|
| `mbarrier` | producer/consumer 的同步单元 | 一次 arrive/wait 的延迟远低于一次 HBM 往返 |
| TMA（`cp.async.bulk`） | 整块搬运，1 线程发起 | 描述符住在 global memory |
| wgmma | 计算主体 | 每个 tile 16 + 16 条 |
| 每 SM 发射槽 | 上限 | 4 条/周期 |
| SMEM 交付 | 无冲突时 | 128 字节/周期 |

### 寄存器与线程数的账

| 配置 | 线程/CTA | `O` 的寄存器/线程 | 说明 |
|---|---|---|---|
| 单 warpgroup 全包 | 128 | **128** | `O` 已经吃掉一半的 255 上限 |
| producer + consumer | 256 | 128 | 同样的 `O`，但搬运不再挤寄存器 |

注意**分工不改变 `O` 的大小**（它由 `Br·d` 决定），分工只是让
「搬运的地址寄存器」与「`O` 累加器」不再争同一份预算。
代价是线程数翻倍：`Br=128` 时 256 线程处理 128 行，
**每行的线程数变多了，但每线程的有效计算量没变**——多出来的那份线程只服务流水。

### pingpong：另一种分工

「pingpong」（FlashAttention-3 的术语）是让**两个 consumer warpgroup 交替**：
一个在算第 `k` 块时，另一个已经拿到了第 `k+1` 块的数据。
它与 producer/consumer 的区别在于**平衡点**：

- producer/consumer 把「搬」与「算」按**功能**拆开；
- pingpong 把「算」按**数据块**拆开，让 softmax 那一段（FP32 向量通路）
  与 wgmma 那一段（tensor core 通路）**在时间上重叠**——
  因为 softmax 用的是 SFU/MUFU，wgmma 用的是 tensor core，两者在硬件上并不抢同一份资源。

两种做法可以叠加：**producer 搬数据、两个 consumer 交替算**。

## 3. 实现与演进

按「谁在什么时候搬数据」这条线：

1. **同步搬（01 / 02 篇）**。算之前先 `__syncthreads()` 搬完 K/V tile。
   简单，但计算必须等搬运，流水完全串行。
2. **`cp.async` 多级流水（Ampere 形态）**。搬运与计算在不同 stage 上重叠，
   用 `cp.async.wait_group` 控制。见 [gemm/04](../gemm/04-tensor-core-and-async-pipeline.md)。
3. **warp specialization（本文，Hopper）**。把搬运从计算 warp 里彻底分出去。
   动机不止是重叠：**wgmma 的 `.aligned` 要求整个 warpgroup 一起发射**（§1.1），
   让同一个 warpgroup 里既有搬运又有 wgmma 是未定义行为。
4. **pingpong（§2）**。在分工的基础上，让两个 consumer 交替，
   把 softmax 的 FP32 向量段与 wgmma 的 tensor core 段也重叠起来。
5. **变体（§1.4）**。滑窗把循环范围截短；ALiBi/softcap 在 `S` 之后加一条逐元素指令。
   **都不改变上面的骨架。**

`code/04_flash_attention_ws.cu` 是这一形态的骨架，带 `window` 与 `use_alibi` 两个开关。

## 4. 踩坑与失败模式

### 4.1 `Bc` 取到 256、`stages` 还想开 4 级

**现象**：SMEM 申请直接失败（或 kernel 的 occupancy 掉到 0）。
`make verify` §B7 把这条算术做成了断言：

```text
Bc=128、stages=4 → 4·2·128·128·2 = 262144 B = 256 KB > 228 KB   （每 SM 可驻留 CTA = 0）
```

**成因**：SMEM 的 `stages·2·Bc·d·esize` 里，`Bc` 与 `stages` 是**乘积**关系，
而 228 KB 是硬上限。直觉上「KV 双缓冲」听起来很便宜（只是 ×2），
但 `Bc=128`、`d=128`、fp16 时**每一级流水就是 64 KB**（§1.2 的表里 2 级是 128 KB）。

**规避**：先算 `2·Bc·d·esize` 再乘 `stages`，把它与 `SMEM_PER_SM_MAX_KB` 直接比。
想要更深的流水就把 `Bc` 减半：`Bc=64` 时一级 32 KB，6 级才 192 KB。
**这条算术不需要跑程序，但必须在改 `Bc` 之前算一遍**。

### 4.2 把 `O` 的寄存器压力当成「增加 `Bc` 就能缓解的问题」

**现象**：看到 SMEM 还剩很多，就加大 `Bc` 想提高吞吐，结果寄存器溢出、
occupancy 掉下去、性能反而更差。

**成因**：`O` 的大小是 `Br·d`，**与 `Bc` 无关**（[02 篇](02-flash-attention.md) §2）。
`Bc` 只影响 SMEM 与 K-loop 的迭代数；`Br` 才同时影响 `O` 与 Q tile。
`Br=128`、`d=128`、128 线程时光 `O` 就要 128 个寄存器/线程，
再加 S 的 fragment（`Br·Bc/线程数`）与地址，很容易顶到 255 的上限。

**规避**：调 `Bc` 之前先看寄存器那本账；`Br` 与线程数是**必须一起定**的一对参数
（`Br=128` 配 256 线程 → 64 个寄存器/线程；配 128 线程 → 128 个）。
`make verify` §B7 把这条也断言了：`Br=128`、`d=128` 时 `O` 恰好 128 个寄存器/线程。

## 编译与运行

配套程序是 **`code/04_flash_attention_ws.cu`**，**需要 `-arch=sm_90a`**：

```bash
cd code
make hopper                        # = nvcc -arch=sm_90a 04_flash_attention_ws.cu
./04_flash_attention_ws 4096 128
```

它进 Makefile 的 `HOPPER_TARGETS`，**不混进默认 `TARGETS`**——`sm_90a` 是
"accelerated" 变体，为它编出的 cubin 跑不了 `sm_90`（反方向可以：
为 `sm_90` 编出的 cubin 能在 H100 上正常加载）。在 A100 上 `make hopper` 编不过属预期。

程序只打印**推导值**：SMEM 的阶段数表、wgmma 指令数、`O` 的寄存器数、
以及滑窗对分块范围的影响。**它不产生任何运行结果**——
本仓库的写作环境没有 CUDA 工具链也没有 GPU，`make hopper` 在这里跑不了。

本机（macOS）只能跑：

```bash
make verify
```

它对应本文的是 §B7（SMEM 阶段数、wgmma 指令数、`O` 的寄存器数、
以及那条 `sm_90a` 的规格断言）与 §B11（滑窗对分块范围的影响）。

## 6. 小结与延伸

全文一条主线：**Hopper 上「算」与「搬」必须分家，而分家的理由是 ISA 的发射粒度。**

三条可独立复用的结论：

1. **wgmma 的发射粒度是 128 线程**（整个 warpgroup），TMA 的发射粒度是 1 个线程。
   两者互补，于是 producer/consumer 的分工几乎不占计算资源。
2. **`Bc` 与 `stages` 是乘积关系**：`SMEM = stages·2·Bc·d·esize`。
   `Bc=128`、`d=128`、fp16 时最多 3 级流水（4 级 = 256 KB > 228 KB）。
3. **变体只改结构**：滑窗改**循环范围**（块数从 512 降到 8，最多 64×），
   ALiBi/softcap 改 **exp 的输入**（分块形状一个字不改）。

下一站：

- [05-decoding-attention](05-decoding-attention.md) —— 换一个场景：
  每 token 只算一行 query，瓶颈从「S 矩阵」变成「KV cache 的读取字节数」，
  分工的形态也跟着变（split-KV 的合并要动 `O`）。
- [20-gpu-architecture/tensor-core/03](../../20-gpu-architecture/tensor-core/03-wgmma-and-pipeline.md)
  —— `wgmma` 的规格与流水的**正文**，本文只用结论。
- [40-kernels/gemm/04-tensor-core-and-async-pipeline](../gemm/04-tensor-core-and-async-pipeline.md)
  —— 同样的 producer/consumer 形态在 GEMM 里的版本。

## 参考

- Dao, *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*, 2023 ——
  warp 分工与 work partitioning 的工程形态。
- Shah, Bikshandi, Zhang, Thakkar, Ramani, Dao, *FlashAttention-3: Fast and Accurate Attention
  with Asynchrony and Low-precision*, 2024 —— warp specialization 与 pingpong 调度；
  本文 §2 的「两种分工的区别在于平衡点」出自此文。
- PTX ISA 9.4 §9.7.15（`wgmma` 的形状、操作数与 `.aligned` 限定符）、
  §9.7.9（`cp.async.bulk` 的发射粒度）—— §1.1 两条硬约束的规格出处。
- 本仓库 [20-gpu-architecture/tensor-core/03-wgmma-and-pipeline](../../20-gpu-architecture/tensor-core/03-wgmma-and-pipeline.md)
  —— wgmma 与流水的**正文**，本文只回链。
- 本仓库 [40-kernels/attention/02-flash-attention](02-flash-attention.md) ——
  `O` 累加器的资源约束与 `(m, l, O)` 的递推，本文的分工是为它服务的。
- 本仓库 `code/verify.py` 的 §B7、§B11 —— 本文全部数字的来源；
  `code/specs.py` 的 `WGMMA_*` 与 `ACCELERATED_TARGETS` 是规格值的出处。

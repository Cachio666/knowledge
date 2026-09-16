---
title: decode 阶段的 attention：GQA/MQA、split-KV 与 paged KV
slug: decoding-attention
domain: 40-kernels
topic: attention
level: 3
difficulty: intermediate
prerequisites:
  - 40-kernels/attention/02-flash-attention
  - 20-gpu-architecture/memory-system/02-coalescing-and-cache-line
tags: [decode, gqa, mqa, split-kv, paged-attention, kv-cache, memory-bound, arithmetic-intensity]
related:
  - 40-kernels/attention/01-attention-as-a-kernel
  - 20-gpu-architecture/memory-system/01-hierarchy-overview
  - 40-kernels/attention/04-warp-specialization-and-variants
code:
  - code/05_decode_attention.cu
status: stable
updated: 2026-09-16
---

# decode 阶段的 attention：GQA/MQA、split-KV 与 paged KV

> 一句话：decode 每个 token 只算**一行** query，于是算术强度塌到 `AI = r`
> （`r = heads_q/heads_kv`）——**与序列长度和 head 维度都无关**，MHA 时只有 `I*`(FP16) 的 0.34%。
> 于是三件事成了主线：**GQA/MQA 按比例砍 KV 读取字节**、**split-KV 补并行度**、
> **paged KV 的代价在间接寻址的延迟而不在碎片**。

## 0. 背景与动机

[01 篇](01-attention-as-a-kernel.md) 到 [04 篇](04-warp-specialization-and-variants.md) 讲的都是
**prefill** 的形态：一次处理整段序列，`S` 行 query 一起算，两段矩阵乘是大块头。
**decode 完全不同**：每生成一个 token，只有**一行** query 要算（每个 head 一行）。

**具体对照场景**：`S = 4096`、`d_head = 128`、`heads_q = 32`、batch = 1、fp16。

| | prefill（S=4096，单 head） | decode（每 token） |
|---|---|---|
| FLOPs | `4S²d = 8.590e+09` | `4·S·d·h_q = 67.1e6` |
| HBM 字节 | 读 Q/K/V + 写 O | 读 **整个 KV cache**：`2·S·d·h_kv·esize` |
| AI | 120.5（`make verify` §B1 / §B9） | **1.0**（`make verify` §B9） |
| 瓶颈 | memory-bound，但离拐点只差 2.3 倍 | **极端 memory-bound，差 295 倍** |

那个 295 倍不是巧合（`make verify` §B9）：

```text
AI = 4·S·d·h_q / (2·S·d·h_kv·esize) = 2·r/esize = r     （fp16，esize = 2）
访存时间 ÷ 算力时间 = I* / AI = 295.4 / 1.0 = 295.4
```

**`S` 与 `d` 在分子分母里同时出现，被约掉了。** 这是本文最重要的一句话：
decode 的算术强度**不随序列长度变化**——序列变长，FLOPs 与 KV 字节**同比例增长**。
所以「等模型支持更长上下文就更快了」这个直觉在 decode 上是错的；
长上下文只会让**绝对时间**线性变长（S=4096 时每 token 要读 67.1 MB，即 20.0 µs）。

于是本文的三条主线都围绕「怎么把这个比值做大」：

| 手段 | 动的是 | 效果 |
|---|---|---|
| GQA / MQA | `r = heads_q/heads_kv` | KV 字节 ∝ `1/r`（MQA 时 `r` = 32，KV 字节只剩 1/32） |
| split-KV | 并行度 | 不动 AI，把空转的 SM 用起来 |
| paged KV | 显存管理 | **不动字节数**，代价挪到延迟上 |

## 1. 原理

### 1.1 GQA / MQA：按比例砍 KV 读取字节

`heads_q` 个 query 头共享 `heads_kv` 个 KV 头，即 `r = heads_q/heads_kv` 个 query 头共用一份 KV
（这就是文献里的**群组大小** `r`，本文全篇用它，不再另起符号）。
**它不改变任何一条算术式子**——`make verify` §A4 在 `heads_q=8` 上验证了这一点：

```text
heads_kv = 8（复制 1 份）：与「每个 query 头独立算一遍」的最大误差 = 4.126e-16
heads_kv = 2（复制 4 份）：                                             6.918e-16
heads_kv = 1（复制 8 份）：                                             4.685e-16
```

**只改「一份 KV 服务几个 query 头」，不改算式。** 收益直接是字节数
（`make verify` §B8，`S=4096`、`d=128`、`h_q=32`、batch=1）：

| `heads_kv` | `r = heads_q/heads_kv` | KV 字节 | 相对 MHA |
|---|---|---|---|
| 32（MHA） | 1 | 67.1 MB | 1.000 |
| 16 | 2 | 33.6 MB | 0.500 |
| 8 | 4 | 16.8 MB | 0.250 |
| 4 | 8 | 8.4 MB | 0.125 |
| 2 | 16 | 4.2 MB | 0.062 |
| **1（MQA）** | **32** | **2.10 MB** | **0.031** |

（`r` 越大表示越多的 query 头共用一份 KV，KV 字节越少。）

**注意减少的是「读取」字节，不是 FLOPs**：注意力分数照样要对着每个 KV 位置算一遍，
这些 KV 只是被 `r` 个 query 头共用（读一次、用 `r` 次）。
这与 GEMM 里「提高数据复用」是同一件事——**复用度上去了，搬运量就下来了**。

### 1.2 split-KV：把空转的 SM 用起来

decode 的并行度天然很低：一个 block 处理「一个序列的一个 query 头」，
于是（`make verify` §C7，H100 有 132 个 SM）：

| batch | `heads_q` | grid | 占 SM 数 | 判定 |
|---|---|---|---|---|
| 1 | 32 | 32 | 24.2% | 填不满 |
| 1 | 8 | 8 | 6.1% | 填不满 |
| 1 | 1 | 1 | 0.8% | 只有 1 个 block 在跑 |
| 8 | 32 | 256 | 193.9% | 够用 |
| 64 | 32 | 2048 | 1551.5% | 够用 |

batch=1、`heads_q=32`（GQA 8 个 kv 头的常见形态）时**只有 24% 的 SM 在工作**。
解决办法与 split-K 是同一个思路：**把 KV 序列也切一刀**，grid 变成
`batch × heads_q × n_split`（`n_split=16` → 512 个 block，每 SM 分到 3.88 个）。

代价是**每段各算一份部分结果，最后必须合并**——而合并的式子恰好就是
[02 篇](02-flash-attention.md) 那条递推：

```text
每段 i 独立算出 (m_i, l_i, O_i)
合并：  M = max_i m_i
        l = Σ_i l_i·e^{m_i − M}
        O = Σ_i O_i·e^{m_i − M}        ← O 与 l 用**同一个** α
        O_final = O / l
```

`make verify` §A5 验证了这条合并与单次计算一致（最大相对误差 `4.926e-15`）。

> **这里正是 02 篇的正文发挥作用的地方**：`O` 必须跟着重标定。
> 只合并 `l` 会得到「每段权重都是 1」的错结果——那也是 [02 篇 §4.2](02-flash-attention.md)
> 的失败模式在前向里的版本。`code/05_decode_attention.cu` 的 CPU 参考里对比了两种合并。

### 1.3 paged KV：代价在延迟，不在字节

paged KV 把每条序列的 KV cache 切成固定大小的页，用一张页表做映射
（页表住在显存里，由调度器按需分配/回收）。它在**显存管理**上的收益是根本性的
（不必预留最长序列的空间），但 kernel 侧要付什么？先算字节（`make verify` §B10，
页大小 16 个 token、`S=16384`、`d=128`、`heads_kv=32`）：

```text
一页一个 head 的字节 = 16×128×2 = 4 KB（= 128 个 sector）
页表 = 1024 项 × 4 B = 4 KB
KV 本体 = 268.4 MB
⟹ 页表占 0.0015%
碎片：最后不满一页的期望浪费 = 16/2 = 8 个 token → 0.049%
```

**页表 0.0015%、碎片 0.049%——字节上两者都可以忽略。** 这和直觉相反：
paged KV 常被说成「有碎片化代价」，但按字节算它根本不值一提。

真正的代价是**间接寻址的延迟**：一个 `Bc = 64` 的 KV tile 跨 `64/16 = 4` 页，
需要 4 次页表查询，而每一次都是**有依赖的沿指针取数**：

```text
该 tile 的 QKᵀ 计算 = 2·128·64·128 = 2.1 MFLOP
  ÷ 每 SM 7.5 TFLOPS = 280 ns ≈ 504 个周期（按 1.8 GHz）
而一次片外访存往返 = 400~800 周期（规格值）
```

`make verify` §B10 断言了这两者**同量级**（比值 0.5~4，算出 0.79×）。
于是结论很硬：**页表必须预取**（把下一页的地址提前读进来），
不能「用到哪一页再查哪一页」——后者会让每次页表查询的 400~800 周期暴露在关键路径上。

**合并访存这一侧反而没问题**，这一点要回链
[memory-system/02](../../20-gpu-architecture/memory-system/02-coalescing-and-cache-line.md)：
只要**页内**地址连续，sector 利用率就是 100%（一页 4096 B = 128 个整 sector）。
碎片化发生在**页的边界**，不在页内。`make verify` §C8 把这条算成了表：

| `d_head` | `esize` | 一行字节 | 整除 sector(32B) | 整除 cache line(128B) |
|---|---|---|---|---|
| 128 | 2 | 256 B | 是 | 是 |
| 128 | 1 | 128 B | 是 | 是 |
| 64 | 2 | 128 B | 是 | 是 |
| 80 | 2 | 160 B | 是 | **否** |
| 96 | 2 | 192 B | 是 | **否** |

`d_head = 128`、fp16 时一行 256 B 恰好是 cache line 的整数倍，页内零浪费。

> **反例值得单独点出**：若为了省页表把布局改成 `[页, 行, head, d]`（行内跨 head 交织），
> 同一行的 head 之间就不连续了——`d_head=128` 时相邻 head 相隔 256 B，
> 每个 head 各占 1 个 cache line，warp 读同一位置的不同 head 会退化成 32 个不同的
> cache line。**paged KV 的布局必须让「一行」连续，而不是让「一个 head 的一行」连续。**

## 2. 执行映射

### 一个 block 在干什么

```text
grid = batch × heads_q            （未切 KV 时）
一个 block：一行 query（d 个元素）
    逐页推进 KV：
        查页表拿到物理页号（**有依赖的取数**）
        → 一页内的 logits（S 条 FMA + 一次 exp）
        → 在线 softmax 的状态更新（m, l）与 **O 的重标定**（α 同时乘 l 与 O）
```

与前几篇的差别在于**循环体太小了**：prefill 里每次循环搬一个 `Bc×d` 的 tile
（`Bc=128`、`d=128` 时 64 KB），decode 里一次只处理 `page_tokens` 行
（16 行 × 128 × 2 = 4 KB），而**计算量只有 `2·16·128 = 4096` 次乘加**。
于是固定开销（页表查询、循环控制、归约）在总时间里的占比远高于 prefill。

### 具体组件与量级

| 组件 | 在 decode 里的角色 | 量级（规格值） |
|---|---|---|
| HBM 带宽 | **唯一的主角** | 3350 GB/s |
| 片外访存往返 | 页表查询的延迟 | 400~800 周期 |
| L1 sector | 页内合并的最小单位 | 32 字节 |
| warp scheduler | 发射槽 | 每 SM 4 条/周期 |
| tensor core | 大材小用 | 989.5 TFLOPS 只用得上 0.34% |

**每 token 的时间账**（`S=4096`、`d=128`、`h_q=32`、batch=1，推导值，`make verify` §B9）：

| `heads_kv` | `AI` | KV 字节/token | 访存时间 | FLOPs/token | 算力时间 |
|---|---|---|---|---|---|
| 32（MHA） | 1.0 | 67.1 MB | 20.0 µs | 67.1 M | 67.8 ns |
| 8（GQA） | 4.0 | 16.8 MB | 5.0 µs | 67.1 M | 67.8 ns |
| 1（MQA） | 32.0 | 2.10 MB | 0.6 µs | 67.1 M | 67.8 ns |

**注意算力时间那一列恒为 67.8 ns**——FLOPs 不随 `heads_kv` 变（§1.1），
变的只有字节数。这就是「GQA/MQA 是纯粹的访存优化」的量化形态。

## 3. 实现与演进

按「怎么把 decode 那一行的算力喂饱」这条线：

1. **朴素 decode**：一个 block 一个 query 头，循环全部 KV。
   并行度 = `batch × heads_q`，batch 小时大量 SM 空转（§1.2）。
2. **split-KV（本文）**：KV 也切一刀，每段独立算出 `(m_i, l_i, O_i)` 再合并。
   并行度变成 `batch × heads_q × n_split`。**合并要动 `O`**（§1.2）——
   这是 [02 篇](02-flash-attention.md) 那条递推在推理侧的形态。
3. **GQA / MQA（§1.1）**：把 `heads_kv` 降下来，直接砍字节数。
   这是**模型层面**的决定，kernel 只是受益方——但 kernel 必须知道
   「`r` 个 query 头共用一份 KV」，不然会把它重复读 `r` 遍。
4. **paged KV（§1.3）**：显存管理必需的形态；kernel 侧要接受页表的间接寻址，
   并把页表**预取**进流水。
5. **prefill/decode 分离调度**：两者的 AI 差 120 倍（120.5 对 1.0），
   同一个 kernel 服务两者必然有一侧低效。工程上的做法是把它们分成两个 kernel
   （chunked prefill 是这一思路的延伸）——**这也是本主题把 decode 单列一篇的理由**。

`code/05_decode_attention.cu` 给了 decode kernel（含 paged KV 的页表寻址）
与 split-KV 的合并参考。

## 4. 踩坑与失败模式

### 4.1 以为「序列更长 ⟹ 更快 / 更划算」

**现象**：拿 prefill 的直觉去看 decode——「矩阵更大，硬件利用率更高」。
于是把「支持长上下文」当成性能优化来宣传。

**成因**：decode 的 `AI = r` 里，`S` 在分子分母**同时出现**
（`4·S·d·h_q` 与 `2·S·d·h_kv·esize`），被约掉了（§0 的推导）。
所以序列长度只改变**绝对时间**（线性），不改变**性质**（`make verify` §B9 的三行）：

| `S` | prefill 的 AI | decode 的字节/token | decode 的时间/token |
|---|---|---|---|
| 4096 | 120.5 | 67.1 MB | 20.0 µs |
| 16384 | 126.0 | 268.4 MB | 80.1 µs |
| 32768 | 127.0 | 536.9 MB | 160.3 µs |

**prefill 的 AI 在爬（120.5 → 127.0），decode 的字节在涨而 AI 纹丝不动（恒为 1.0）。**

**规避**：谈 decode 的性能时，永远报「每 token 读多少字节」而不是「用了多少算力」。
要动这个比值只有三条路：**减 KV 头数**（GQA/MQA）、**减 KV 位宽**（量化 KV cache）、
**减 KV 被读的次数**（如 prefix caching 复用已算过的前缀）。

### 4.2 只合并 `l` 不合并 `O`

**现象**：split-KV 的合并阶段只把 `l_i` 加起来，`O_i` 直接相加——
结果看上去是「一个合理的向量」，但每一段的权重都不对。

**成因**：与 [02 篇 §4.2](02-flash-attention.md) 是同一条式子。
每段的 `O_i` 是按**自己的**基准 `m_i` 算出来的，直接相加等于把不同基准下的量混在一起。
正确的做法是每段乘 `e^{m_i − M}`（`M` 是所有段的最大值），
**这个缩放因子对 `l` 与 `O` 是同一个**。

**规避**：把合并写成一个算子（`merge_states`），让 `l` 与 `O` 的缩放**在同一个函数里**、
用同一个 `α`——`code/verify.py` 的 `merge_splits` 就是 8 行，但它是唯一正确的写法。

### 4.3 页表「用到哪页查哪页」

**现象**：页表查询被放在 KV tile 的搬运之前，`400~800 周期`的往返延迟直接暴露在
关键路径上。SM 看着在忙（issue 率不低），但吞吐上不去。

**成因**：§1.3 的那笔账——一个 `Bc=64` 的 tile 要查 4 次页表，
而 tile 本身的计算只要约 504 个周期。**延迟与计算同量级**，
一旦串行，SM 有一半时间在等地址。

**规避**：页表查询必须进流水（提前一个 stage 查到下一块的页号），
与 [04 篇](04-warp-specialization-and-variants.md) 里 K/V tile 的预取是同一件事。
另一个方向是把**页调大**：页越大，查表次数越少（`Bc/P`），
代价是碎片比例上升（`P/(2S)`）——但 §1.3 已经算出碎片只占 0.049%，
所以**页大小的真正约束是查表次数，不是碎片**。

## 编译与运行

配套程序是 **`code/05_decode_attention.cu`**，含 decode kernel（带 paged KV 的页表间接寻址）
与 split-KV 合并的 CPU 参考。

它需要 `nvcc` 与 GPU：

```bash
cd code
make 05_decode_attention
./05_decode_attention                 # S=1024, d=128, heads_q=32, heads_kv=8, n_split=8
./05_decode_attention 4096 128 32 8 16
```

程序会打印：decode 的算术强度账（AI `= r` 与它与 `I*` 的比值）、
GQA/MQA 的流量表、并行度表（未切 / 切 KV 的 grid）、paged KV 的字节与延迟账，
以及 split-KV 合并与单次计算的一致性。

**两类数字必须分开看**：

- 本文表格里的数、`295.4`、`0.34%`、`67.1 MB`、`504 个周期`、`0.049%`——
  **推导值**，可用 `make verify` 的 §A4 / §A5 / §B8 / §B9 / §B10 / §C7 / §C8 复核；
- 各内核的逐元素差与耗时 —— **那台机器上的运行结果**。本仓库没有跑过。

本机（macOS，无 CUDA 工具链）只能跑：

```bash
make verify
```

它对应本文的是 §A4（GQA/MQA 的等价性）、§A5（split-KV 的合并）、
§B8（KV 流量比）、§B9（`AI = r` 与 295.4 倍）、§B10（页表与间接寻址的延迟）、
§C7（decode 的并行度）、§C8（页内对齐与布局）。

## 6. 小结与延伸

全文一条主线：**decode 的瓶颈是「每 token 要读多少 KV 字节」，而这个量由 `1/r` 唯一决定（`r` 是群组大小）。**

三条可独立复用的结论：

1. **`AI = 2·r/esize = r`（fp16），与 `S`、`d`、batch 都无关**。
   MHA 时 AI = 1.0，只有 `I*`(FP16) 的 0.34%；访存时间是算力时间的 295.4 倍。
   长上下文只让绝对时间线性变长，不改变这个比值。
2. **GQA/MQA 按 `1/r` 线性砍 KV 读取字节**，**不改任何一条算式**
   （`make verify` §A4 的误差在 `1e-16` 量级）。MQA 是 `heads_kv = 1` 的极端，压到 1/32。
3. **paged KV 的代价在间接寻址的延迟，不在碎片**：页表占 0.0015%、
   页内碎片占 0.049%，而一次页表查询的 400~800 周期与一个 tile 的计算（约 504 周期）同量级
   → **页表必须预取**。页内连续 ⟹ 合并访存无碍（回链 memory-system/02）。

下一站：

- [01-attention-as-a-kernel](01-attention-as-a-kernel.md) ——
  若要看 prefill 的形态与那张 AI 表（`I*` 就是本文反复对照的那个 295.4）。
- [02-flash-attention](02-flash-attention.md) —— split-KV 的合并用的就是这里的第三条递推。
- [20-gpu-architecture/memory-system/02](../../20-gpu-architecture/memory-system/02-coalescing-and-cache-line.md)
  —— sector / cache line 的口径，本文 §1.3 与 §C8 的判据出自此处。

## 参考

- Shazeer, *Fast Transformer Decoding: One Write-Head is All You Need*, arXiv:1911.02150, 2019 ——
  MQA（`heads_kv = 1`）的出处。
- Ainslie, Lee-Thorp, de Jong, Zemlyanskiy, Lebrón, Sanghai, *GQA: Training Generalized
  Multi-Query Transformer Models from Multi-Head Checkpoints*, EMNLP 2023 ——
  GQA 的出处；本文 §1.1 的群组大小 `r` 与它一致。
- Kwon, Li, Zhuang, Sheng, Zheng, Yu, Gonzalez, Zhang, Stoica, *Efficient Memory Management
  for Large Language Model Serving with PagedAttention*, SOSP 2023 ——
  paged KV 的出处；本文 §1.3 只讨论它对 kernel 侧的影响。
- Dao, *FlashDecoding*（FlashAttention 仓库中的 split-KV 解码路径）——
  split-KV 的并行度动机与合并方式；本文 §1.2 的合并式子与 [02 篇](02-flash-attention.md) 同源。
- 本仓库 [40-kernels/attention/02-flash-attention](02-flash-attention.md) ——
  `(m, l, O)` 的重标定与「忘标定 `O`」的失败模式，本文 split-KV 的合并是它的直接应用。
- 本仓库 [20-gpu-architecture/memory-system/02-coalescing-and-cache-line](../../20-gpu-architecture/memory-system/02-coalescing-and-cache-line.md)
  —— sector 与 cache line 的口径，paged KV 的合并访存判据。
- 本仓库 `code/verify.py` 的 §A4、§A5、§B8、§B9、§B10、§C7、§C8 —— 本文全部数字的来源。

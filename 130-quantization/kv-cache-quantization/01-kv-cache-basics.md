---
title: KV cache 的账：一串整数算术，与 decode 的 memory-bound
slug: kv-cache-basics
domain: 130-quantization
topic: kv-cache-quantization
level: 1
difficulty: beginner
prerequisites: []
tags: [kv-cache, paged-attention, arithmetic-intensity, decode, memory-bound]
related:
  - 130-quantization/kv-cache-quantization/02-kivi-asymmetry
  - 130-quantization/quantized-kernels/01-w8a8
  - 130-quantization/numerical-formats/04-integer-formats
code:
  - code/01_kv_cache_size.py
status: stable
updated: 2026-09-18
---

# KV cache 的账：一串整数算术，与 decode 的 memory-bound

> KV cache 的字节数是一个六项乘积 `2 × 层数 × KV 头数 × head_dim × 序列长度 × batch × 每元素字节`。它在 Llama-2-7B 上给出两个漂亮的整数：**每 token 512 KiB**、**4096 ctx / batch 1 / fp16 恰好 2 GiB**。而 decode 阶段每生成一个 token 就要把整个 cache 读一遍——注意力算子的算术强度 `AI = S·T/(S+T)` 在 `S=1` 时收敛到 **1 FLOP/字节**。

## 0. 背景与动机

自回归推理分两段。**prefill** 把整条 prompt 一次性喂进去，算力吃满；**decode** 每步只处理一个新 token，却要把此前所有 token 的 K 和 V 读出来。后者就是把 KV cache 变成瓶颈的地方。

先算一笔具体的账。Llama-2-7B（MHA，32 层，32 个 KV 头，head_dim 128），fp16：

| 场景 | KV cache |
|---|---|
| 1 个 token | **524 288 B = 512 KiB** |
| 4096 ctx / batch 1 | **2 GiB**（精确等于 2³¹ 字节） |
| 4096 ctx / batch 32 | 64 GiB |
| 128K ctx / batch 1 | 64 GiB |

对照：Llama-2-7B 的 fp16 **权重**约 13 GiB。**batch 到 32 时，KV cache 就是权重的 5 倍**——而它一个字的信息量都没有，纯粹是「为了不重算」而驻留的中间结果。这就是这一域存在的理由。

先看一个具体的对照场景，把「为什么这是个问题」钉死：

- **不省**：把 prompt 缓存下来，4096 ctx / batch 32 需要 64 GiB 显存，A100 80GB 只剩 16 GiB 放别的东西——**batch 上不去，单请求成本就下不来**；
- **省到 4-bit**：同样 64 GiB 的预算能装下 batch 128。吞吐的收益不是线性的，是「能不能开进去」的二值问题。

所以量化 KV cache 的动机从来不是「省一点」，而是「**批得动**」。KIVI 的摘要把这条链条写得最清楚：`2.6× less peak memory (including model weight)` → `up to 4× larger batch size` → `2.35×~3.47× throughput`。三个数里**只有第一个是显存**，后两个是它的后果。

## 1. 原理

### 第一步：把六项乘积写出来

一个 token 在某一层里的 K 与 V 各是一个长度为 `head_dim` 的向量，但那是**一个 KV 头**的量。整层要 `kv_heads` 份，整套要 `layers` 层，还要 K 与 V 两份：

```
每 token 字节数 = 2 × layers × kv_heads × head_dim × bytes_per_element
```

代入 Llama-2-7B（`2 × 32 × 32 × 128 × 2`）→ **524 288 B**。

再乘序列长度与 batch，就得到总量：

```
总字节数 = 每 token 字节数 × seq_len × batch
```

`524288 × 4096 = 2147483648 = 2³¹` —— **恰好 2 GiB**。这不是凑出来的，而是因为每一项都是 2 的幂：`2 · 2⁵ · 2⁵ · 2⁷ · 2 · 2¹²`。这个等式有个实际用处：**它是验证任何 KV cache 计算器的最短判据**，任何一位写错都会让它不再是整数。

### 第二步：`head_dim × kv_heads` 与 hidden size 的关系，以及 GQA 的坑

对 MHA 的模型，`kv_heads = attn_heads`，于是 `head_dim × kv_heads = hidden_size`。OPT-13B 就是这种：`128 × 40 = 5120`。PagedAttention 论文的算式用的是 hidden size：

> "the KV cache of a single token demands **800 KB** of space, calculated as
> 2 (key and value vectors) × 5120 (hidden state size) × 40 (number of layers) × 2 (bytes per FP16)."
> —— arXiv:2309.06180v1 §3（逐字读到）

`2 × 5120 × 40 × 2 = 819 200 B`。论文写「800 KB」——**这里是 KiB**（`819200/1024 = 800`）。下一步再乘 2048 token 得 1.6 GiB，论文也写成 1.6 GB。**整条链都按 1024 记账**，而「KB」这个词在十进制下会给出 819.2。

**GQA 模型上 `kv_heads ≠ attn_heads`，这个等价关系就断了。** Llama-2-70B 是 64 个注意力头、**8 个 KV 头**；Mistral-7B 是 32 个注意力头、8 个 KV 头。用注意力头数去算会分别高估 **8 倍**和 **4 倍**。这是本主题最容易犯、也最容易在 code review 里被漏掉的错。

### 第三步：三处独立的外部印证

同一个算术可以从三个互不相干的方向被撞上：

| 来源 | 它说什么 | 用本文的算式重算 |
|---|---|---|
| PagedAttention/vLLM 论文 §3 | OPT-13B 每 token **800 KB**、2048 token 最多 **1.6 GB** | `2·5120·40·2 = 819200 B = 800 KiB` ✓；`×2048 = 1.5625 GiB` → 一位小数 1.6 ✓ |
| Databricks 博客 Table 3 | Llama-2-70B / GQA / seq 1024 / batch 1 / fp16 = **0.312 GiB** | `2·80·8·128·2·1024 / 2³⁰ = 0.3125 GiB` → 三位小数 0.312 ✓ |
| KVQuant 论文 Table 1 | LLaMA-7B / 128K 的 KV cache = **64.0 GB** | `524288 × 131072 = 2³⁶ = 64.0 GiB` ✓ |

**三处的价值在于它们是三种不同形状的算式**：一处用 hidden size（MHA），一处是 GQA 的（8 个 KV 头），一处是从总容量反解每 token。三条都落在同一个公式上，才叫印证；三条都抄同一个数就不叫。

⚠ **Databricks 那一条本仓库未取到博客原文**（`WebFetch` 被策略拦截），档位是「计划基线，未独立核实」，只用它做量级对拍。另外两条逐字读到了。

### 第四步：decode 为什么是 memory-bound——把算术强度算出来

注意力算子（忽略 softmax 与 mask）做两件事：`S = QKᵀ` 与 `O = PV`。

```
FLOPs   = 2·S·T·D（QKᵀ） + 2·S·T·D（PV） = 4·S·T·D
搬运字节 = (Q + K + V + O) × 每元素字节
        = 2·bpe·D·(S + T)          # 每张量 S·D 或 T·D 个元素，bpe = 每元素字节
```

于是

```
AI = 4·S·T·D / (2·bpe·D·(S+T)) = 2·S·T / (bpe·(S+T))
```

**bpe = 2（fp16）时它化简成 `S·T/(S+T)`**：

| 阶段 | S | T | `S·T/(S+T)` |
|---|---|---|---|
| decode | 1 | 4096 | **0.9998** |
| decode | 1 | ∞ | **1.0** |
| prefill | 4096 | 4096 | **2048** |

**decode 的算术强度是 1 FLOP/字节，prefill 是 `T/2`。** 同一份代码、同一份权重，只因为形状不同，就从内存受限翻到算力受限。这就是「prefill 与 decode 要分开优化」的定量根据。

> ⚠ **「AI = 1」这个数是「每元素 2 字节」这个前提的产物，不是注意力的固有性质。** 换成 fp32（`bpe = 4`），同一个式子给出 **0.5**，恰好腰斩。本机实测把两个值都算了出来（§5 表 4）。
>
> ⚠ **不要把它推广成「全模型 decode = 1 FLOP/weight-byte」。** 那个说法本仓库**只找到二手博客的推导，没有一手论文**——按 CONTRIBUTING §2 记入 `UNSOURCED`，正文不写，`specs.py` 里也不落成常量。这里能站住的只有 attention 部分的 `ST/(S+T)` 与「decode 是 memory-bound」这条定性表述（PagedAttention 论文 §3 与 NVIDIA 博客都有逐字表述）。

## 2. 执行映射

**（1）KV cache 的分配粒度决定了「浪费多少」。** vLLM 的 PagedAttention 把 cache 切成固定大小的 block（典型 16 个 token）并按需分配，正是为了消掉「按最大长度预留」的浪费。对本文的算术而言这一点直接可见：`4096 × batch` 的预留式分配在序列大多只有几百 token 时会浪费 90% 以上。

**（2）scale 放在哪一维，是引擎必须显式声明的东西。** vLLM 的 `--kv-cache-dtype` 走 `CacheConfig.cache_dtype`，默认 `"auto"`。它的 FP8 路径**只有 per-tensor 或 per-head 两种粒度，没有 per-token**——这不是实现偷懒，而是因为量化发生在**写 cache 的时刻**，那时刻只有当前这一个 token，拿不到整行统计量。这条限制是下一篇（[02](02-kivi-asymmetry.md)）要处理的核心矛盾。

> ⚠ **vLLM 的 `CacheDType` 取值集合是版本相关的。** `main` 分支的 `vllm/config/cache.py` 列 **18 项**（含 `nvfp4`、`turboquant_*`、三个 `*_per_token_head`）；老版本文档列 7 项。**引用必须写明分支/版本。** 完整的 18 项见 [03](03-low-bit-and-engines.md) §2。

**（3）写与读是两个不同的 kernel。** 以 vLLM 的 FP8 为例：量化融合在 `reshape_and_cache_flash_kernel` 里——**这是一个独立于 attention 的 kernel**，负责把新算出的 K/V 量化后写进 paged block；反量化则发生在 attention kernel 内部。**「融合」指的是量化与写 cache 融合，不是把量化融进 attention。**

**（4）量化省下的字节，先从「读一侧」兑现。** decode 每步读 `2·T·D·bpe` 字节，所以 KV 从 fp16 降到 fp8 会让这一步的访存量直接减半。本机的 CPU 实测把这条机制量了出来（§5 表 5）：decode 形状的 `QKᵀ` 达到同 trial 内拷贝带宽的 **38%–66%**，即它确实是在**等内存**而不是在算。

## 3. 实现与演进

**第一步：fp16 全量缓存。** 本文的算式就是这个阶段的全部内容。它的失败模式在 §0 的账里——**长上下文 + 大 batch 装不下**。

**第二步：换更窄的存储 dtype（fp8）。** 这是最保守的一步：K/V 的数值范围本来就不大（e4m3 的动态范围 ±240 常常够用），换 dtype 不改算法、不改粒度，直接 2× 压缩。限制是 `e4m3` 的动态范围只有 ±240，且**当前只支持 per-tensor（标量）**、不传 scale 就默认 1.0。另一个常被忽略的代价：**fp8 下前缀缓存（prefix caching）不工作**——因为量化后的 cache 不再与原始 token 一一对应。

**第三步：分 head / 分 token 的粒度。** vLLM 提供 per-tensor 与 per-attention-head 两档；KIVI 则进一步走到 K per-channel / V per-token（下一篇）。粒度越细，需要存的 scale 越多——这笔账在 [03](03-low-bit-and-engines.md) §1 单独记。

**第四步：把「缓存」这件事本身重新分层。** 当 KV cache 比权重还大，压缩它就不再是「优化」而是「能不能服务」的问题。这条线上后续的东西（前缀共享、block 级淘汰、offload 到 CPU）都在同一个算式上做乘法或除法。

## 4. 踩坑与失败模式

### 4.1 用注意力头数代替 KV 头数（GQA 上高估 4–8 倍）

**现象**：容量规划里算出「Llama-2-70B 每 token 2.5 MiB」，实际是 320 KiB。规划出来的 batch 数直接小了 8 倍。

**成因**：MHA 时代 `kv_heads == attn_heads`，两个词可以互换；GQA/MQA 把它们拆开了。**公式里那一项是 KV 头数**——它决定 cache 里存几份 K/V。

**规避**：① 用 `head_dim × kv_heads` 与 hidden size 对一遍（MHA 下它相等，GQA 下不等，这是一个立刻能看出问题的自检）；② 认准每 token 的**整数**：Llama-2-7B 512 KiB、13B 800 KiB、70B 320 KiB、Mistral-7B 128 KiB。**它们都是 2 的幂**，算出来不是整数就说明某一项错了。

### 4.2 「KB」是 1000 还是 1024——两个读数都真实，但不是同一个量

**现象**：OPT-13B 每 token 是「800 KB」还是「819.2 KB」？两个都能从论文与算式里读出来。

**成因**：论文按 **KiB** 记账（`819200/1024 = 800`），而「KB」这个词在十进制下是 1000。**同一份字节数，两个单位口径差 2.4%**；在 128K 上下文那种量级上会差出 4.7 GB。

**规避**：**引用任何容量数字时连单位口径一起写。** 本主题统一按 KiB/GiB 记账，并在正文里显式标注论文那两个「KB/GB」的实际口径。这条对 KVQuant 的 `64.0 GB` 尤其重要——它在十进制下是 68.7 GB。

### 4.3 把「AI = 1」当成注意力的固有性质

**现象**：文档写「decode 的算术强度是 1 FLOP/字节」，读者换个 dtype 复现出 0.5，于是怀疑自己算错了。

**成因**：那个式子化简到 `S·T/(S+T)` 的**前提是每元素 2 字节**。完整形式是 `2·S·T/(bpe·(S+T))`。**「1」是 fp16 的产物。**

**规避**：写这个数时连同前提一起写（「在 fp16 下」），或者干脆写完整式。本机程序把 fp16 与 fp32 两个值都打印出来（§5 表 4），差值恰好是 2。

### 4.4 引用「全模型 decode = 1 FLOP/weight-byte」

**现象**：这个说法在二手材料里流传很广，读起来像一条硬结论。

**成因**：**本仓库没有找到一手论文。** 能找到的只有博客文章里的推导。把它推广到全模型需要「每读一个权重字节做一次 MAC」这类额外假设，那些假设不是注意力算子的性质。

**规避**：**只写 attention 部分的 `ST/(S+T)`**（有推导、有 `S=1 → 1` 的收敛），加上「decode 是 memory-bound」这条两处独立来源的定性表述。全模型那条记入 `UNSOURCED`，不进正文、也不进 `specs.py`——**不落成常量就不会被误引**。

## 编译与运行

```bash
cd code
make run                     # 自动探测装了 torch ≥ 2.8 的解释器，依次跑 01–03
make 01_kv_cache_size        # 只跑本文对应的程序
make verify                  # 默认 python3，跑 verify.py（**不 import torch**）
```

本篇对应的编号程序是 **`code/01_kv_cache_size.py`**（真 torch；B 类断言在程序内部，失败即非零退出）。

**配套程序与 `verify.py` 的分工**（CONTRIBUTING §2 第三种形态）：

| | 谁在跑 | 产出档位 | 本篇负责什么 |
|---|---|---|---|
| `01_kv_cache_size.py` | 真 torch（探测到的解释器） | **本机实测值** | 真张量分配的 `nbytes`、decode 的 achieved GFLOP/s 与带宽占比 |
| `verify.py` §A.1–A.3 | 默认 python3，**不 import torch** | **推导** | 同一份字节账的**第二条算式**、三处锚点、`AI` 的独立重算 |

**两档不得互相背书**：`verify.py` 也复现了「Llama-2-7B 4096 ctx = 2³¹ B」——那是算术，不是实现；两者一致只说明**算得对**，不说明**任何引擎真的这样分配**。

## 5. 实测数据

**测量条件**：CPU **Apple M1 Pro（10 核）**；macOS；python **3.12.9**；**torch 2.11.0**。表 1–3 的每一格在 **2.6.0 / 2.7.1 / 2.8.0 / 2.11.0 / 2.13.0** 五个解释器上**逐字节一致**（本仓库实测）。**CPU-only，无 GPU 数字**——显存带宽、kernel 性能、perplexity 都不在本机测。未覆盖 2.9.x / 2.10.x / 2.12.x。

**表 1：每 token 字节数与 4096 ctx 的总量**（`01_kv_cache_size.py` §B1；判据：fp16、MHA/GQA 按 `KV_MODEL_SHAPES` 的 KV 头数）

| 模型 | 层 | KV 头 | d | B/token | KiB/token | 4096 ctx fp16 |
|---|---|---|---|---|---|---|
| Llama-2-7B（MHA） | 32 | 32 | 128 | 524 288 | **512.0** | **2.00 GiB** |
| Llama-2-13B（MHA） | 40 | 40 | 128 | 819 200 | 800.0 | 3.12 GiB |
| Llama-2-70B（GQA） | 80 | **8** | 128 | 327 680 | 320.0 | 1.25 GiB |
| Mistral-7B（GQA） | 32 | **8** | 128 | 131 072 | 128.0 | 0.50 GiB |
| OPT-13B（MHA） | 40 | 40 | 128 | 819 200 | 800.0 | 3.12 GiB |

`2·32·32·128·2·4096 = 2147483648`，**精确等于 2³¹**——程序把这条写成断言，不是打印而已。

**表 2：同一模型的四种精度**（§B2；判据：只换 `bytes_per_element`，`int4`/`int2` 是**载荷**字节、**不含 scale 元数据**）

| 精度 | 4096 ctx | 128K ctx | 相对 fp16 |
|---|---|---|---|
| fp16 | 2.00 GiB | 64.00 GiB | 1.00× |
| fp8 | 1.00 GiB | 32.00 GiB | 2.00× |
| int4 | 0.50 GiB | 16.00 GiB | 4.00× |
| int2 | 0.25 GiB | 8.00 GiB | 8.00× |

> ⚠ 上表是**载荷口径**。加上 scale/zero 元数据后的**整块等效宽度**见 [03](03-low-bit-and-engines.md) §5 表 1（G=32 的 int2 是 **4.00 bit/元素**，实压缩 **4.00×** 而不是 8×）。**两个不是同一个量。**

**表 3：三处外部锚点的独立重算**（§A2；判据逐条写出）

| 锚点 | 来源印值 | 本文算式 | 偏差 |
|---|---|---|---|
| OPT-13B 每 token | 800 KB | 819 200 B = **800.0 KiB** | 0（按 KiB 记账） |
| OPT-13B × 2048 token | 1.6 GB | 1.5625 GiB → **1.6**（一位小数） | 0（四舍五入） |
| Llama-2-70B seq=1024 | 0.312 GiB | 0.3125 GiB → **0.312**（三位小数） | 0（四舍五入） |
| LLaMA-7B @128K | 64.0 GB | 2³⁶ B = **64.0 GiB**（十进制 68.7） | 0（按 GiB 记账）；十进制下差 4.7 |

**表 4：decode 与 prefill 的算术强度、以及 achieved GFLOP/s**（§B4/§B5；`T = 8192`、`D = 128`、best-of-5 交错测量）

⚠ **前两列是算术（跨机器、跨版本一致）；后两列是计时，随机器与当前负载漂**——本表是**一次运行**的读数，不是可复现的常数。

| S | AI = S·T/(S+T)（算术） | QKᵀ 耗时（本次） | achieved（本次） |
|---|---|---|---|
| 1（decode） | **1.00** | 36.1 µs | **116.3 GFLOP/s** |
| 16 | 15.97 | 312.3 µs | 214.9 GFLOP/s |
| 256 | 248.24 | 1105.4 µs | 971.4 GFLOP/s |
| 4096（prefill） | 2730.67 | 14 606.0 µs | 1176.2 GFLOP/s |
| 对照：2048³ 稠密 GEMM | — | 10 573.9 µs | **1624.7 GFLOP/s** |

**decode 形状的 achieved GFLOP/s 只有稠密 GEMM 的 1/14。** 同一个 `q @ k.T`，S=1 与 S=4096 差 **10 倍**——同族算子、同一份 K，只有 S 不同，所以这个对照没有跨算子的混杂。**这两个比值是量级判据**（程序里断言的也是量级：`< 1/5`、`> 5×`），不是可以逐位引用的常数。

同一算式的另一半（**判据是每元素字节**）：fp16 下 `S=1, T→10⁷` 给 **0.999878**，fp32 下给 **0.499939**，比值**恰好 0.500000**。

**表 5：decode 步跑到峰值带宽的百分之几**（§B6；判据：**逐 trial 比值的中位数**——同一 trial 里背靠背测 decode 与拷贝）

⚠ 同样是**计时**，不是常数。

| | 值 |
|---|---|
| decode 步搬运 | 8.39 MB（K + V，T=8192、D=128、fp32）——**这个是算术，稳定** |
| decode 带宽 / 同 trial 拷贝带宽 | **中位数 47%**（9 个 trial 区间 42%–62%） |

⚠ **这个比值是量级判据，不是精度判据。** 本机是 CPU，六个子进程并行时两个量各自的实测带宽会漂（decode 侧 7–32 GB/s、拷贝侧 50–100 GB/s）。判据写成「**同一 trial 内**的比值取中位数」是因为噪声在比值里对消；`make run` 是单进程，正常机器上这个数更稳。**它不是关于任何 GPU 的结论。**

## 6. 小结与延伸

三条可复用的结论：

1. **KV cache 的账是纯整数算术，而且它的整数性本身就是判据。** Llama-2-7B 每 token 512 KiB、4096 ctx 恰好 2 GiB；凡是算出来不是 2 的幂，某一项就错了。三处独立外部印证（PagedAttention 的 MHA 算式、Databricks 的 GQA 算式、KVQuant 的 128K 总量）都落在同一个公式上。
2. **公式里那一项是 KV 头数，不是注意力头数。** GQA 把它们拆开后，用错会在 Llama-2-70B / Mistral-7B 上分别高估 **8 倍 / 4 倍**。
3. **decode 是 memory-bound，而且这个结论有一个干净的定量形式：`AI = 2·S·T/(bpe·(S+T))`。** fp16 下 `S=1` 给 1，prefill（`S=T`）给 `T/2`。⚠ **「1」是「每元素 2 字节」的产物**，fp32 下是 0.5。

**这条主线在 §0.8 的三角上是哪条边**：**「范围 / 精度 vs 存储」**。KV cache 是量化域里唯一一处「精度完全不影响正确性、只影响成本」的地方——它不是模型的参数，是一个纯缓存的中间结果，所以它可以被压到极限，代价是一点注意力精度。也正因为它是个纯缓存，**「粒度放在哪一维」变成了一个与数值格式无关的独立自由度**——那是下一篇的全部内容。

延伸有三条。**往下**：为什么 K 和 V 要用**不同**的量化轴——这需要先搞清楚 K 的 outlier 落在哪一维（[02-kivi-asymmetry](02-kivi-asymmetry.md)）。**往旁边**：粒度决定 `amax` 沿哪一维归约，以及为什么激活的 per-channel 会插进 GEMM 内维（[theory/02-granularity](../theory/02-granularity.md)）。**往窄**：把 KV 压到 1-bit 会发生什么、各家引擎各自把 scale 放在哪（[03-low-bit-and-engines](03-low-bit-and-engines.md)）。

## 参考

- **PagedAttention / vLLM**，arXiv:**2309.06180v1** §3 —— 本文 **2026-09-18 逐字读到原文**（`arxiv.org/html/2309.06180v1`）。原文那两句：
  "the KV cache of a single token demands **800 KB** of space, calculated as 2 (key and value vectors) × 5120 (hidden state size) × 40 (number of layers) × 2 (bytes per FP16). Since OPT can generate sequences up to 2048 tokens, the memory required to store the KV cache of one request can be as much as **1.6 GB**."
  另有 "this is a **memory-bound** operation" 的定性表述（同节）。
  > ⚠ 论文里的 "KB" 是 **KiB**、"GB" 是 **GiB**（819200/1024 = 800；800 KiB × 2048 = 1.6 GiB）。**换十进制单位两个数都对不上。**
- **KVQuant**，arXiv:**2401.18079v4** Table 1 —— **逐字读到**。LLaMA-7B / WikiText-2 / 128K 的 baseline `KV Cache (GB)` 列 = **64.0**、PPL 5.68；int4 = 16.0 GB。
- **Databricks 博客 Table 3**（Llama-2-70B / GQA / seq 1024 = 0.312 GiB）—— **计划基线，本次未独立核实**：博客原文未取到（`WebFetch` 被策略拦截）。本仓库只用它做量级对拍（算式给 0.3125，三位小数 0.312）。
- **vLLM `CacheDType`** —— `vllm/config/cache.py`（`main` 分支，2026-09-18 逐字读到）。⚠ **快照会随版本变**；老版本文档的 7 项与 `main` 的 18 项都真实，差别在版本。完整清单见 [03](03-low-bit-and-engines.md) §2。
- **KIVI**，arXiv:2402.02750v2 摘要 —— `2.6× less peak memory (including model weight)` / `up to 4× larger batch size` / `2.35×~3.47× throughput`，**逐字读到**。⚠ 三个数里**只有第一个是显存**。
- **`UNSOURCED`**：「**全模型 decode = 1 FLOP/weight-byte**」——本仓库只找到二手博客的推导，**未找到一手论文**。正文与 `specs.py` 都不使用它，正文只写 attention 部分的 `ST/(S+T)`（有推导、有来源）。参见 `code/specs.py` 顶部的 `UNSOURCED` 清单。
- 本仓库 [02-kivi-asymmetry](02-kivi-asymmetry.md) —— K 与 V 为什么要走不同的量化轴。
- 本仓库 [03-low-bit-and-engines](03-low-bit-and-engines.md) —— 1-bit 的断崖与引擎集成矩阵。
- 本仓库 [theory/02-granularity](../theory/02-granularity.md) —— 四种粒度的定义与「per-token 只治行、per-channel 只治列」。

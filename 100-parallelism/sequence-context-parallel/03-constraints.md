---
title: SP 与 CP 的分工、cp 的约束，以及 causal mask 跨块
slug: sequence-context-parallel-constraints
domain: 100-parallelism
topic: sequence-context-parallel
level: 3
difficulty: advanced
prerequisites:
  - 100-parallelism/sequence-context-parallel/01-activation-sharding
  - 100-parallelism/sequence-context-parallel/02-ring-attention
tags: [context-parallelism, sequence-parallelism, causal-mask, load-balance, padding]
related:
  - 100-parallelism/sequence-context-parallel/01-activation-sharding
  - 100-parallelism/sequence-context-parallel/02-ring-attention
code:
  - code/03_constraints.py
status: stable
updated: 2026-10-08
---

# SP 与 CP 的分工、cp 的约束，以及 causal mask 跨块

> **`01` 说 SP 省的是 `10sbh`（`O(s)`）；`02` 说 CP 的机制是「块统计量重缩放」。**
> **本文把两者接起来**：**SP 只切 Dropout 与 LayerNorm 的激活；CP 连网络输入与【全部】激活
> 一起按序列维切**（`specs.CP_VS_SP_QUOTE` 逐字）。
> ⇒ ★★★ **序列越长，SP 越不够用、CP 越必需。**
> ★★ **而 CP 的代价也随之上来**：注意力要看到**全序列的 KV**、`S` 不被 `cp` 整除时要补齐、
> ★★★ **而 `causal mask` 是唯一需要「绝对位置」的地方 —— 用错位置【不报错、只是错】。**

## 0. 背景与动机

`01` 结尾留下的问题是：SP 的收益是 `O(s)`，而注意力的 `O(s²)` 它**切不到**。
要切那一项，就得**切序列维本身** —— 这就是**上下文并行（CP）**。

★★ CP 的定义（`specs.CP_DEFINITION_QUOTE`，来源级别 **[B]**：某实现的官方文档，**不是规范**）：

> `Context Parallelism (CP) is a parallelization scheme on the sequence-length dimension.`

★ **它与 SP 不是同一个东西** —— 原文把两者**逐字分开**（`specs.CP_VS_SP_QUOTE`，本库
`100-parallelism/README.md` 也引过同一句）：

> `Unlike prior SP (sequence parallelism), which only splits the sequence of Dropout and LayerNorm activations, CP partitions the network inputs and all activations along the sequence dimension.`

★★★ **这一句就是「SP 与 CP 的分工」的全部**：

| | SP | CP |
|---|---|---|
| **切什么** | Dropout 与 LayerNorm 的激活 | **网络输入与【全部】激活** |
| **切在哪个轴** | 序列维（**但在 TP 组内**） | 序列维（**跨设备**） |
| **收益量级** | `10sbh` = **`O(s)`** | 包括注意力的 **`O(s²)`** |
| **谁需要它** | 与 TP 搭配、省复制掉的激活 | **序列长到 `S²` 放不下时** |

## 1. 原理

### 1.1 ★★ `cp` 的四条约束

★★ 四条都来自同一份文档（`context_parallel.md`，**活文档**，抓取日 2026-10-08，
**不做行号锚点**）：

1. **注意力是唯一有跨 token 算子的模块**（`specs.CP_NO_CHANGE_QUOTE`）：
   > `With CP, all modules except attention (for example, Linear and LayerNorm) can work as usual without any changes, because they do not have inter-token operations.`
2. **所以注意力必须看到全序列的 KV**（`specs.CP_QKV_QUOTE`）：
   > `For attention, the Q (query) of each token must combine with the KV (key and value) of all tokens in the same sequence.`
3. **于是要一次额外的 all-gather**（`specs.CP_ALLGATHER_QUOTE`）：
   > `CP therefore requires an additional all-gather across GPUs to collect the full sequence of KV.`
4. **但前向只存自己那一段的 KV、反向再 gather 一次**（`specs.CP_CHUNK_KV_QUOTE`）：
   > `To reduce activation memory footprint, each GPU stores only the KV of a sequence chunk in forward and gathers KV again in backward.`

⇒ ★★ **第 3、4 条合起来就是 CP 的交易**：**用【通信】换【显存】**。
★ 这与 `02` 的「边收边丢」是同一件事（`specs.RING_ACCUMULATION_QUOTE`）。

★ 另外两条只点名不展开：**环是【实现选择】**（`specs.CP_RING_P2P_QUOTE`：
`The all-gather and reduce-scatter are implemented as point-to-point communications in a ring topology.`）；
**每卡显存按 CP 度下降**（`specs.CP_MEM_PER_GPU_QUOTE`）。
★★ 本主题**不重写**集合通信原语的算法与复杂度（正文在 `110-communication/collectives`）。

### 1.2 ★★★ `causal mask` 跨块：**唯一需要「绝对位置」的地方**

带 causal mask 时，第 `i` 个 token 只能看 `≤ i` 的 key。切了序列之后：

```text
第 r 段 query 的【绝对】位置 qpos = r·(S/cp) + arange(S/cp)
第 c 块 key  的【绝对】位置 kpos = c·(S/cp) + arange(S/cp)
掩码条件： kpos <= qpos
```

★★★ **块内那两个 `arange` 是【局部】的，必须各加一个偏移。**
⇒ **用局部位置代替绝对位置**：形状、类型、算术**全对**，程序**不报错**，
**只有跨块的那些格子错位** —— 见 §4.1。

★ **本机实测**（`code/03_constraints.py` 的 `A.1`，`S = 512`，四档 CP 度）：

| `cp` | `1` | `2` | `4` | `8` |
|---|---|---|---|---|
| `max\|跨块 − 单卡\|` | **`0.000000e+00`** | `4.996e-16` | `4.996e-16` | `5.551e-16` |

★★ `A.2` 另外钉住「**掩码真的在起作用**」：同一份数据，**有掩码 / 无掩码**的结果相差
**`2.387560e+00`** ⇒ 掩码那条路径**非空**（否则 `A.1` 就是对空操作判等）。

### 1.3 ★★ `S % cp != 0` 的补齐代价（计划的 **P25**）

★★ `cp` 台设备各要拿**整数个** token ⇒ 每台拿 `ceil(S/cp)` 个 ⇒
**总计算量从 `S` 涨到 `cp·ceil(S/cp)`**。**本机实测**（`S = 100`、`cp = 8`）：

```text
ceil(100/8) = 13     ⇒    8 × 13 = 104     ⇒    多算 4 个 token（4.00%）
```

★★ **它是「向上取整」的代数后果**，与实现质量无关；`cp` 越大、`S` 越不被整除，代价越大。

★★ **补齐之后必须做对两件事**（`B.2` 钉住第一件）：

1. **补齐位置的 key 对真实 query 必须全部不可见** —— 它们的绝对位置 `>= S` 恒大于任何真实
   query 的位置 ⇒ 掩码自动挡住。**本机实测：可见格子数【恰为 `0`】**，
   且真实行的结果与「不补齐时」一致（差 `5.551e-17`）⇒ ★ **补齐只花算力、不污染结果**；
2. ★ **补齐位置的 query 结果是垃圾** ⇒ 端到端实现里要**按长度剪掉**
   （本程序只读前 `S` 行）。

### 1.4 ★★★ 整块被 mask 掉：运行最大值是 `-inf`

带 causal mask 时，**有些 KV 块对某段 query 是【整块不可见】**的。
此时那一块的 `max` 是 `-inf` ⇒ 如果缓存的**运行最大值也还是 `-inf`**：

```text
exp(-inf − (−inf)) = exp(nan) = nan
```

★★★ **所以运行最大值【必须】兜住**（把 `-inf` 换成 `0` 再算 `exp`，于是
`exp(-inf − 0) = 0` —— 那是对的：那一块全不可见，贡献为 `0`）。

★★ **一个关键细节**：`02` 的环序是「**从自己那块开始**」⇒ 第一块总含对角线 ⇒
运行最大值一开始就是有限的 ⇒ **兜底代码永远走不到**。
⇒ ★★ **本程序用一个【同样合法】的环向把它触发**：`cp−1, …, 0`（**反向**）。
此时第 `0` 段 query 最先收到的 `cp−1` 个块**整块都在它右边**。
**本机实测**（`cp = 4`）：**不兜底 ⇒ `nan` 计数 `24576 / 32768`、`inf` 计数 `0`**；
**兜住 ⇒ 与单卡差 `4.996e-16`、全部有限**。

★★ 注意这里与 `02` 的 `C.1` **同款**：**判据断言 `nan`，不是断言 `inf`**。

## 2. 执行映射

### 2.1 本机怎么算这件事

| 角色 | 是什么 | 依据 |
|---|---|---|
| **被测** | `causal_blockwise()`：KV 块按环序流过每一段 query，带**绝对位置**的 causal 掩码 + 统计量重缩放 | `specs.CP_CAUSAL_QUOTE` + `02` 的机制 |
| **参照** | `causal_full()`：**单卡**一次算完的 causal 注意力 | 定义 |

★★★ **`D.8` 用 AST 调用表给出源码级线索**（★ **是线索，不是证明**；
★★ **实测：参照那一侧的调用表【本身就是空集】** —— `causal_full()` 只做 `np.*` 的
**属性调用**、`ast.Name` 一处都没有 ⇒ **「交集为空」这一路没有信息量**）；
★★★ **真正的独立性由 `A.1g` 行为级兜底**：**不做 causal mask** 时差变成 `2.387560e+00`
⇒ `A.1` **必须失败** ✓

### 2.2 一条必须写明的口径

> ★★ **本机是纯 numpy 单进程**：「跨设备」是**显式的张量搬运**（KV 块按环序喂给每一段 query）。
> ★★ **判据依赖 BLAS** ⇒ **换机器可能需要重新标定**。
> ★★★ **种子取 `991`**（**不是 2 的幂**），写在 `_make_inputs()` 的**函数体里**。
> ★ 强制限定语：**「不是 NCCL，也不是真机」**。

## 3. 实现与演进

### 3.1 为什么 CP 与 SP 都要，而不是二选一

- **SP** 省的是 TP 组内**被复制**的那批激活（`O(s)`），**代价极小**（转换合并进 TP 的通信）；
- **CP** 省的是**整层**激活（含 `O(s²)`），**代价是通信**（all-gather KV，`02` 的 `B.1` 说它 `∝ S`）；
- ⇒ ★★ **两者是不同尺度上的两把刀**：短序列用 SP 就够，长序列必须上 CP。

### 3.2 ★★ `causal mask` 与负载均衡的张力

★★ 逐字（`specs.CP_CAUSAL_QUOTE`，这是实现文档**自己承认**的一件事）：

> `avoiding extra work from lower-triangle causal masking while keeping load balanced across GPUs`

★★★ **下三角之外的那些块是【算白算】的**（它们整块被 mask 掉），
而 CP 的实现**特意避开**这份额外工作、并保持负载均衡。
⇒ ★★ 本文的 `C.1`（局部位置）量的正是**这份张力写错时的样子**：
位置一错，掩码就不是「下三角」，而是**每块各自一个下三角** —— **不报错、结果错**。

★ 同一段还说了另一件事（`specs.CP_RING_HIGH_PERF_QUOTE`）：**CP 与 Ring Attention 是同一类机制**
（★ 实现文档**自己这么说**，本库把它登记为 **[B] 级**，不当规范结论）。

### 3.3 与相邻主题的分工

- **SP 的三式与收益** → [01-activation-sharding](01-activation-sharding.md)；
- **块统计量重缩放的形式** → [02-ring-attention](02-ring-attention.md)；
- ★ **本域不写、只回链**（与 `01`/`02` 同一张表）：
  · **「在线 softmax」的 `(m, l)` 更新公式本身** →
    `40-kernels/softmax-normalization/02-online-softmax.md` 与 `40-kernels/attention/`
    ⇒ ★★★ **本文不使用这个词**；
  · **`S×S` 矩阵的访存账** → `40-kernels/attention/03-flash-attention-backward.md` 与
    `00-meta/glossary.md` 的 `flash attention` 条；
  · **位置编码的外推**（PI / NTK / ALiBi、混叠周期）→
    `80-model-architectures/positional-encoding/02,03`
    ⇒ ★ 本文只谈「**用哪一段的绝对位置**」，**不谈位置编码本身怎么造**；
  · **集合通信原语的算法与复杂度** → `110-communication/collectives`与
    `10-foundations/performance-models/02-amdahl-and-scaling.md:209`；
  · **TP 每层 `2+2` 次 all-reduce** → `tensor-parallel/03` 与
    `10-foundations/performance-models/02-amdahl-and-scaling.md:417`（★ **不是 `:223`**）。

## 4. 踩坑与失败模式

### 4.1 ★★★ 本主题最强的一条：块内 token 用【局部位置】

- **做法**：`qpos = arange(block)`、`kpos = arange(block)`（忘了加 `r·block` 与 `c·block`）；
- **现象**：★★★ **不报错**（形状、类型、算术全对），**只是错**；
- **本机实测**：`cp = 2` ⇒ `max|diff| = 3.404195e+00`；`cp = 4` ⇒ **`2.883652e+00`**；
- **成因**：掩码不再是「下三角」，而是**每块各自一个下三角**（§3.2 的那份张力写错的样子）；
- **规避**：★★ **块的绝对位置 = `r·(S/cp) + arange(S/cp)`** —— 两处偏移**一个都不能少**。

### 4.2 ★★ 阴性对照：不做 causal mask

- **本机实测**：`cp = 4` 时差 **`2.387560e+00`** ⇒ 它**也真的会错**；
- ★ 它与 §4.1 是**两条不同的错法**，**各自被单独钉住**（`C.1` / `C.2`），
  ★★ 因为它们的成因不同：一个是**位置**错，一个是**掩码缺失**。

### 4.3 ★★★ 遗忘 `-inf` 的兜底 ⇒ `nan`

- **现象**：`nan` 计数 `24576 / 32768`（`cp = 4`，反向环序，`S = 512`）；
- **成因**：`exp(-inf − (−inf))`（§1.4）；
- **规避**：★★★ **把 `-inf` 的运行最大值换成 `0` 再算 `exp`**；
  ★★ 而**触发它需要一个【整块不可见】的 KV 块排在前面** ——
  环序「从自己那块开始」**不会触发它** ⇒ 本程序用**反向环序**把它逼出来。

### 4.4 ★★ 把补齐当成「免费的」

- **现象**：`S = 100`、`cp = 8` 时总计算量从 `100` 涨到 `104`（**`4.00%`**）；
- **成因**：`ceil(S/cp)` 的向上取整；
- **规避**：★ 把代价**算出来**（`B.1`），并把「补齐的 key 不污染结果」钉住（`B.2`）。

## 5. 判据表

| 可机械判定的条件 | 结论 |
|---|---|
| 说「SP 与 CP 差不多」 | **不合格**：**SP 只切 Dropout / LayerNorm 的激活；CP 连输入与全部激活一起切**（逐字，`[B]` 级实现文档） |
| 序列变长 | ★★★ **先 SP、后 CP**：SP 的收益是 `O(s)`，注意力那 `O(s²)` 只有 CP 切得到 |
| `cp` 的约束 | ★★ 四条：**只有 attention 要改**、**它要看到全序列 KV**、**一次额外 all-gather**、**前向只存自己那段 KV** |
| 带 causal mask 的分块注意力 | ★★★ **块的绝对位置 = `r·(S/cp) + arange(S/cp)`**；★ 用**局部位置**⇒ **不报错、结果错**（实测 `2.88e+00`，`cp=4`） |
| 不做 causal mask | **也真的会错**（实测 `2.39e+00`）—— ★ 与「位置错」是**两条不同的错法** |
| 某个 KV 块整块不可见 | ★★★ **运行最大值是 `-inf`** ⇒ `exp(-inf − (−inf)) = nan` ⇒ **必须兜住**；★ 触发它需要**整块不可见排在最前**（反向环序） |
| 拿 `inf` 当失败信号 | ★★★ **在这里失效**：`inf` 计数 `0`、`nan` 计数 `24576` ⇒ **断言 `nan`** |
| 用 `max` **累积**误差再与阈比 | ★★★ **有盲区**：`max(0.0, nan) == 0.0` ⇒ **`nan` 被吞掉、判据照样 `[ok]`**（本机实测：把掩码 `<=` 改成 `<`，四档差全 `nan`、`A.1` 却仍打 `[ok]` 并打印「最大差 `0.000000e+00`」）⇒ ★★ **判据必须【逐档】查有限性**（`A.1` 与 `02` 的 `A.1` 各带一条 `isfinite` 护栏） |
| `S % cp != 0` | ★★ **补齐代价 = `cp·ceil(S/cp) − S`**（实测 `100 → 104`，`4.00%`）；★ **补齐的 key 不污染真实行**（可见格子数恰为 `0`） |
| 说「CP 零开销」 | ★★ 它是**用通信换显存**（all-gather KV）；★ 逐字的零开销**条件**在 `02` 的 `RING_OVERLAP_QUOTE` |
| 想用「online softmax」描述本文 | ★★★ **不合格**：本域写的是**跨设备**的块统计量重缩放；措辞照原文（**置换不变性** + **统计量重缩放**） |

## 6. 小结与延伸

**一句话**：**SP 与 CP 是不同尺度上的两把刀** —— SP 切 TP 没切到的 `O(s)` 那一块、
CP 连 `O(s²)` 一起切；★★ **而 CP 的一切约束都落在一件事上**：
**注意力是唯一有跨 token 算子的模块，所以只有它要改、也只有它需要绝对位置。**

**三条可独立复用的结论**：

1. ★★★ **带 causal mask 的跨块注意力 == 单卡**（四档 CP 度，差 `≤ 5.552e-16`）
   —— 前提是**块的绝对位置**加对了；
2. ★★★ **用局部位置⇒ 不报错、结果错**（实测 `cp=2 → 3.404e+00`、`cp=4 → 2.884e+00`）；
   ★ **阴性对照**：不做掩码 ⇒ `2.388e+00`（两条不同的错法）；
3. ★★★ **整块不可见 ⇒ 运行最大值是 `-inf` ⇒ 必须兜住**（不兜底实测 `nan` 计数 `24576`，
   而 `inf` 计数 `0`）；★ 触发它需要**整块不可见排在最前**的环序。

**延伸**：

- **SP 的三式与收益** → [01-activation-sharding](01-activation-sharding.md)；
- **块统计量重缩放的形式** → [02-ring-attention](02-ring-attention.md)；
- **位置编码本身**（怎么造、能不能外推）→ `80-model-architectures/positional-encoding/02,03`；
- **集合通信原语** → `110-communication/collectives`。

## 参考

- NVIDIA Megatron-LM 仓库文档 `docs/user-guide/features/context_parallel.md`。
  ★★ **来源级别 [B]**：**某实现的官方文档**，**不是规范**（本域的头号错误模式就是把它当 A 级）。
  ★★★ 它是**活文档**（无版本号、随 `main` 分支变动）⇒ 引用写明**抓取日 2026-10-08**、
  **不做行号锚点**。载体 =
  `https://raw.githubusercontent.com/NVIDIA/Megatron-LM/main/docs/user-guide/features/context_parallel.md`，
  逐条在**原始字节**上 `count()` = **1**。
  ★ 本程序引它的**十一条**：SP/CP 分工、CP 定义、`cp` 的四条约束、环的点对点实现、
  「与 Ring Attention 同类但性能更高」、causal 与负载均衡的张力、**计算与通信按 CP 度一起下降**
  （`specs.CP_CP_DEGREE_QUOTE`，`D.7`）、每卡显存按 CP 度下降。
- Liu et al., *Ring Attention with Blockwise Transformers for Near-Infinite Context*,
  arXiv:2310.01889 —— ★★ **同行评审：未确认**（见 [02](02-ring-attention.md) 的「参考」）；
  本文只借它的**措辞**（置换不变性 + 统计量重缩放）。
- 本仓库 [01-activation-sharding](01-activation-sharding.md)、
  [02-ring-attention](02-ring-attention.md)、
  `100-parallelism/README.md`（**逐字引过同一句 SP/CP 分工**）、
  `40-kernels/softmax-normalization/02-online-softmax.md`（**块内**那套公式的正文，此处回链）。

## 编译与运行

```sh
cd code && make run       # 三个编号程序，任一断言失败即非零退出
make 03_constraints       # 只跑本文的配套程序（★ 需要 numpy）
```

★ 本机的运行条件：`python3 3.13.5` + `numpy 2.4.4`，macOS arm64，
**无 GPU、无多卡、无 NCCL、无真实网络**。
★ 强制限定语：**不是 NCCL，也不是真机** —— 本文的一切数字都是**玩具规模的真跑**，
**不得**支撑关于真实集群性能的结论。
★★ **判据依赖 BLAS** ⇒ **换机器可能需要重新标定**。
★★ **种子取 `991`**（**不是 2 的幂**），写在 `_make_inputs()` 的**函数体里**。

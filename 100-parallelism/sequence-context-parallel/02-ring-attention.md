---
title: Ring Attention：跨设备的块统计量重缩放
slug: sequence-context-parallel-ring-attention
domain: 100-parallelism
topic: sequence-context-parallel
level: 3
difficulty: advanced
prerequisites:
  - 100-parallelism/sequence-context-parallel/01-activation-sharding
  - 100-parallelism/tensor-parallel/01-column-and-row
tags: [context-parallelism, ring-attention, blockwise-attention, sequence-length, kv]
related:
  - 100-parallelism/sequence-context-parallel/03-constraints
  - 100-parallelism/tensor-parallel/01-column-and-row
code:
  - code/02_ring_attention.py
status: stable
updated: 2026-10-08
---

# Ring Attention：跨设备的块统计量重缩放

> **序列太长 ⇒ 注意力矩阵 `S×S` 放不下**，于是把**序列维**切到多台设备上。
> **切完立刻出现一个新问题**：每一段 query 都必须看到**全部** KV，而那些 KV 在**别人手里**。
> ★★ **朴素的处置（把别人的 KV 攒过来）会把省下的显存又花掉** ——
> 所以做法是**边收边用边丢**：让 KV 块在**环**上流过每一段 query。
> ★★★ **而这一切成立的前提是一条性质**：**一个 query 块对一组 KV 块的注意力，
> 与这组 KV 块的处理【顺序】无关，只要每块的统计量被正确地结合起来做重缩放。**
> ★★ **本文**（以及本主题）**不使用「online softmax」这个词** ——
> 那是指**块内**的 `(m, l)` 更新公式（正文在 `40-kernels/softmax-normalization/02`）；
> 本文写的是**跨设备**的**块统计量重缩放**，措辞照原文。

## 0. 背景与动机

`01` 讲了 SP 的边界：它省的是 `10sbh`（`O(s)`），**切不到注意力的 `O(s²)`**。
序列一长，`O(s²)` 那一项就压不住了 —— 于是要切**序列维**本身（= **上下文并行，CP**）。

切完立刻有三个问题：

1. **每段 query 需要全部 KV** ⇒ 那些 KV 得从别的设备拿到；
2. **如果一边拿一边攒**，显存就白省了 —— 原文逐字（`specs.RING_ACCUMULATION_QUOTE`）：
   > `the accumulation of key-value blocks leads to increased memory usage, which defeats the purpose of reducing memory cost.`
3. **拿的时候还要等** ⇒ 设备会**空转**（原文说的 `a computation delay`）。

★★ Ring Attention 的处置是：**让 KV 块在环上转**，每台设备**同时**发自己那块、收上一台那块，
于是**通信与计算重叠**。★ 原文给出这件事的**条件**（`specs.RING_OVERLAP_QUOTE`）：

> `As long as block computations take longer than block transfers, overlapping these processes results in no added overhead compared to standard transformers.`

★★ **注意它是【有条件的】零开销**（`As long as block computations take longer than block transfers`）
⇒ **不得转述成「一定零开销」**。★★ 本机**没有多卡、没有真实网络** ⇒ 本文**不声称**
任何关于重叠幅度的结论（重叠的机制在 `110-communication` 的 `overlap`）。

## 1. 原理

### 1.1 ★★★ 置换不变性 + 统计量重缩放

原文 §3 逐字（`specs.RING_PERMUTATION_QUOTE`）：

> `we leverage the permutation invariance property of the inner loop’s key-value block operations.`

紧接着它把这件事说全（`specs.RING_RESCALE_QUOTE`）：

> `the self-attention between a query block and a group of key-value blocks can be computed in any order, as long as the statistics of each block are combined correctly for rescaling.`

★★★★ **这句话就是本文的全部内容**，拆成两半：

- **置换不变性**：块的**处理顺序**可以任意（环上从哪一台开始都行）；
- **统计量重缩放**：但**每块的统计量必须被正确地结合** —— 这是**不能省**的那一步。

★ 同一件事在 §1 还有另一种措辞（`specs.RING_ORDER_INVARIANT_QUOTE`）：
`the results are invariant to the ordering of these blockwise computations.`

### 1.2 重缩放长什么样

把「一行」的注意力写成分块累加。每个块 `j` 给出自己的**局部最大值** `m_j`，
缓存的三个量是 **运行最大值 `m`**、**分子累积 `acc`**、**分母累积 `l`**。

★★ **并入一个块时发生两件事**（★ 本文**只用文字说清，不写式子** —— 理由见本节末）：

1. **运行最大值只会往上涨**：取「已缓存的最大值」与「本块的最大值」中**大的那一个**
   ⇒ ★ 它一旦被抬高，**之前那些块算出来的量**基准就旧了；
2. ⇒ **重缩放**：把**已经累积的分子与分母**按**新旧最大值之比**缩回来，
   再把本块按**新的**基准并进去。

★★★ **第 2 步那个「按新旧最大值之比缩回」的因子就是「重缩放」本身**
⇒ 合并完后 `acc / l` 才是这一行的正确输出，**与块的顺序无关**。

★★★ **为什么这里不写式子**：**块内那套 `(m, l)` 更新公式的正文不归本文**
（它在 `40-kernels/softmax-normalization/02-online-softmax.md`，那篇**逐字自述**
「本文是 `(m, l)` 更新公式的正文」）—— ★ **一个知识点只有一处正文，其他位置只放链接**
（`CONTRIBUTING` §7）。本文要的是**跨设备**的**块统计量重缩放**这件事，
**上面那两步的文字说明已经够用**（序号、方向、因子来源都在，唯独不重做推导）。
★ 同理，**输出累加器那条递推的正文在 `40-kernels/attention`**，本文也不写。

★★ **`code/02_ring_attention.py` 的 `A.1` 钉住的就是这件事**：
`RING_CP_GRID` 五档 CP 度（`1/2/4/8/16`）下，分块结果与**一次算完的全量注意力**逐位相符。
**本机实测**（`max|分块 − 全量|`）：

| `cp` | `1` | `2` | `4` | `8` | `16` |
|---|---|---|---|---|---|
| `max\|diff\|` | **`0.000000e+00`** | `5.551115e-16` | `5.551115e-16` | `6.106227e-16` | `6.661338e-16` |

（★ 表中每个数都是 `code/02_ring_attention.py` 的 `A.1` 打印出来的**本机实测值**；
★ 全表最大 `6.661338e-16`。）

★ `cp = 1` 那一档**恰为 `0.0`**：环上只有一台设备 ⇒ 只有一个块 ⇒ **没有可重缩放的东西**
（退化对照）。

### 1.3 ★★ `A.2`：重缩放那条路径**真的被走到过**

★ 上表的差已经很小了，但**还有一件事要钉**：那些 `cp > 1` 的档里，
**§1.2 第 2 步那个「重缩放」因子**（`acc` 与 `l` 各自要乘上的、**新旧最大值之比**）
**到底有没有 ≠ `1` 的时候？**

★★★ 如果没有（例如最大值恰好总在**第一个**块里），那么重缩放那几行**一次都没执行**，
`A.1` 就是**对空操作判等** —— 一条**看起来很强、其实是空的**判据。
⇒ `A.2` 直接数「**缓存的最大值被推高的次数**」，要求 `cp > 1` 的每一档都 `> 0`。
**本机实测**：五档全部 `> 0`。

### 1.4 ★★ 通信量 `∝ S`，不是 `∝ S²`

原文 §1 逐字（`specs.RING_SEND_KV_QUOTE`）：

> `each device sends a copy of its key-value blocks being used for blockwise computation to the next device in the ring, while simultaneously receiving key-value blocks from the previous one.`

★ 每台设备**只发自己那一块 KV**（`K` 与 `V` 两份），在环上转一圈 ⇒
**每台设备发出的量** = `(cp−1) × 2 × (S/cp) × d` = **`2·S·d·(cp−1)/cp`**。

**本机实测**（`code/02_ring_attention.py` 的 `B.1`，`cp = 8`、`d = 64`、每元素 4 字节）：

| `S` | 每台设备发出（字节） | 上界 `2·S·d·4` | 占比 |
|---|---|---|---|
| `256` | `114688` | `131072` | `0.875` |
| `512` | `229376` | `262144` | `0.875` |
| `1024` | `458752` | `524288` | `0.875` |
| `2048` | `917504` | `1048576` | `0.875` |

⇒ ★★ **`S` 翻倍时通信量严格 `2.0000` 倍**（线性），**占比恒为 `(cp−1)/cp`**（与 `S` 无关）。
★★★ **原因是传的是 KV（长度 `S`），不是注意力矩阵（`S×S`）**。

★ **本程序不做的一件事**：**不重推**「重叠之后零开销」那句 ——
它是原文的**引用**（且原文的**同行评审未确认**，见「参考」）。

### 1.5 ★ 显存：从 `O(S²)` 到「与 `S` 无关」

原文 §1 逐字（`specs.RING_QUADRATIC_QUOTE`）：

> `the self-attention has memory cost quadratic in the input sequence length`

★★★ **本文的处理是【引用，不重推】**：那句结论是原文的（**同行评审未确认**），
`code/02_ring_attention.py` 的 `B.2` **不重算**它，只做**数组规模**的对照 ——
**固定每块** `block = 64`（即 `cp = S/block` 随 `S` 一起长）时：

| `S` | 全量分数矩阵 | 单块的分数矩阵 |
|---|---|---|
| `256` | `65536` | `4096` |
| `1024` | `1048576` | `4096` |
| `4096` | `16777216` | `4096` |

★ 全量随 `S` **二次增长**（`S` 翻倍 ⇒ `4.0000` 倍）；**单块恒为 `4096`**（与 `S` 无关）。
★★ 这是**规模对照**，**不是**对原文那句结论的复核 —— 两件事必须分开说。

## 2. 执行映射

### 2.1 本机怎么算这件事

| 角色 | 是什么 | 依据 |
|---|---|---|
| **被测** | `ring_attention()`：KV 块**按环序**逐个流过每一段 query，缓存 `(acc, l, m)` | `specs.RING_RESCALE_QUOTE` |
| **参照** | `full_attention()`：**一次算完**，在**整行**上取最大值 | 定义 |

★★★ **两条路径必须不共享代码**：参照**只看整行**（`axis=1` 的全局最大），
分块版**永远看不到整行**（它一块一块地更新缓存）。
`code/02_ring_attention.py` 的 `D.8` 用 **AST 调用表**给出**源码级线索**
（★ **是线索，不是证明**；★★ **实测：参照那一侧的调用表【本身就是空集】**
—— 它只做 `np.*` 的**属性调用**、`ast.Name` 一处都没有 ⇒ **「交集为空」这一路
没有信息量**，它**连「两条路径不共享代码」也说明不了**）；
★★★ **真正的独立性由 `A.1g` 行为级兜底**：把「缩回累积统计量」那一步去掉（`alpha ≡ 1`），
差变成 **`2.399407e-01`** ⇒ `A.1` **必须失败** ✓

### 2.2 ★★ 一条必须写明的口径

> ★★ **本机是纯 numpy 单进程**：所谓「跨设备」被实现成**显式的张量搬运**
> （把 KV 块按环序逐个喂给每一段 query）。
> ★ 本域的知识对象是**切分策略**，不是通信原语的实现
> （通信算法的正文在 `110-communication`）。
> ★★ **判据依赖 BLAS**（`q @ k.T` 走 `numpy` 的矩阵乘）⇒
> **换机器可能需要重新标定**（★ 本主题的两个量级相隔十几个数量级，阈 `1e-12` 安全）。
> ★★★ **随机数种子取 `991`**（**不是 2 的幂**），写在程序的**函数体里**。
> ★ 强制限定语：**「不是 NCCL，也不是真机」**。

## 3. 实现与演进

### 3.1 为什么是「环」

- **环**让每一台设备**同时**发自己的、收上一台的 ⇒ **通信与计算可以重叠**；
- ★★ **环的次序可以任意**（置换不变性）：实现上从「自己那块」开始，绕一圈回到自己前一台；
- ★★ **边收边丢**：每步只用当前的 KV 块，**不攒**（§0 的那句引文）。

### 3.2 与相邻主题的分工

- **CP 的约束与 SP/CP 的分工** → [03-constraints](03-constraints.md)；
- **SP 切的是什么**（`O(s)` 那一项） → [01-activation-sharding](01-activation-sharding.md)；
- ★ **本域不写、只回链**：
  · **「在线 softmax」的 `(m, l)` 更新公式本身** →
    `40-kernels/softmax-normalization/02-online-softmax.md` 与 `40-kernels/attention/`
    ⇒ ★★★ **本文不使用这个词**，写的是**跨设备**的块统计量重缩放；
  · **`S×S` 矩阵的访存账**（FlashAttention 的字节数收益）→
    `40-kernels/attention/03-flash-attention-backward.md` 与 `00-meta/glossary.md` 的 `flash attention` 条；
  · **集合通信原语的算法与复杂度**（ring allreduce、reduce-scatter / all-gather 的阶段划分）→
    `110-communication/collectives`）与
    `10-foundations/performance-models/02-amdahl-and-scaling.md:209`
    ⇒ ★ 本文的「环」是**数据搬运的次序**，不是原语实现；
  · **通信-计算重叠的机制**（双缓冲 / 分块流水）→ `110-communication` 的 `overlap`；
  · **位置编码的外推**（PI / NTK / ALiBi、混叠周期）→
    `80-model-architectures/positional-encoding/02,03`
    （★ 本文只谈「序列切到多台设备上」，**不谈位置编码本身怎么造**）；
  · **注意力变体的结构**（MHA / MQA / GQA 的定义与配比）→
    `80-model-architectures/attention-variants/01-mha-mqa-gqa.md`
    （★ CP 文档里那句「用 MQA/GQA 降 KV 通信量」本文**只提一句**，不展开）；
  · **TP 每层 `2+2` 次 all-reduce** → `tensor-parallel/03` 与
    `10-foundations/performance-models/02-amdahl-and-scaling.md:417`（★ **不是 `:223`**）。

## 4. 踩坑与失败模式

### 4.1 ★★★ 反例必须证明「它真的会错」，而不是「它这次没错」

- **做法**：把**不做重缩放**的朴素版（直接 `Σ exp(分数) @ V`，**不缩回累积量**）
  与原尺度下的参照比。
- **本机实测**：**原尺度下它只差 `5.829e-16`** ——
  ★★★ **看起来完全正确**。⇒ 在那个尺度下断言「有差」是**无效**的
  （那点差与「重缩放对不对」无关，任何实现都会有）。
- **正确的做法**：★★ **找到能触发它的尺度**。把 `Q`/`K` **放大 `100` 倍**
  （`specs.RING_AMPLIFY`）⇒ 分数进入溢出区 ⇒ **朴素版给出满屏 `nan`**：
  **`nan` 计数 `32768 / 32768`**。
- ★★★ **而且判据必须直接断言 `nan`，不是断言 `inf`**：
  **`inf` 计数实测是 `0`**（`inf/inf` 之后只剩 `nan`）
  ⇒ **「查 `inf`」的调试方法在这里失效**。
- **守卫**（`C.1g`）：**同一个放大尺度**下，带重缩放的版本**仍然有限**、且与参照差 `0.000e+00`
  ⇒ ★★ 坏掉的是**重缩放**，**不是「尺度本身」**。

> ★★★ **这条纪律可以独立复用**：一条「错法」判据的价值，
> **不在于它在某个尺度下能报错，而在于它在【能触发它的尺度】下真的报错**。

### 4.2 ★★ 参照与分块版共享代码 ⇒ 对拍退化成自证

- **现象**：`A.1` 通过，但两边其实是**同一段逻辑**；
- **成因**：参照被写成「用分块版跑一次 `cp = 1`」；
- **规避**：★★★ **参照必须在整行上取最大值**（它**看不到块**），
  且 `D.8` 用 AST 调用表给出源码级线索、`A.1g` 从**行为**上兜底。

### 4.3 ★ 把「零开销」当成无条件

★ 原文的措辞是**有条件的**（`As long as block computations take longer than block transfers`）
⇒ **不得转述成「Ring Attention 一定零开销」**。
★★ 本机**没有多卡、没有真实网络** ⇒ 本文**不声称**任何关于重叠幅度的结论。

## 5. 判据表

| 可机械判定的条件 | 结论 |
|---|---|
| 分块结果与全量一致 | **成立**（实测差 `≤ 6.662e-16` —— 五档 CP 度里的**最大值**；★ 但**必须同时检查「重缩放路径非空」**（`A.2`） |
| `cp = 1` | 差**恰为 `0.0`** —— 退化对照（没有可重缩放的东西） |
| 要判断「重缩放有没有被执行」 | 数**缓存最大值被推高的次数**；`cp > 1` 的每一档都必须 `> 0` |
| 报出一台设备的 KV 通信量 | **∝ `S`**（`2·S·d·(cp−1)/cp`），**不是** `O(S²)` —— 传的是 KV 不是矩阵 |
| 说「显存从 `O(S²)` 降到 …」 | ★★ 那是**引用**（原文，**同行评审未确认**）；本库只做**数组规模对照**（`S²` vs `block²`），**不重推** |
| 说「Ring Attention 零开销」 | ★★ 原文**有前提**（块计算慢于块传输）；**不得**去掉那个 `As long as` |
| 想证「朴素版会错」 | ★★★ **先找到能触发它的尺度**；原尺度下它**只差 `1 ulp` 档、看起来完全正确** |
| 拿 `inf` 当失败信号 | ★★★ **在这里失效**：实测 `inf` 计数 `0`、`nan` 计数满 ⇒ **断言 `nan`** |
| 参照实现 | ★★ **不得调用**被测的任何函数；`D.8` 给源码级线索（★ 不是证明），`A.1g` 给**行为级**守卫 |
| 想用「online softmax」描述本文 | ★★★ **不合格**：本域写的是**跨设备**的块统计量重缩放；术语照原文（**置换不变性** + **统计量重缩放**） |

## 6. 小结与延伸

**一句话**：**块可以按任意顺序算，但每块的统计量必须被正确地结合起来做重缩放** ——
这句话让「KV 在环上流过每一段 query」成立，也把显存从 `O(S²)` 拿回来。

**三条可独立复用的结论**：

1. ★★★ **置换不变性 + 统计量重缩放**：五档 CP 度下分块与全量逐位相符（差 `≤ 6.662e-16`）；
   ★ 且 `cp > 1` 的每一档里**重缩放都真的被执行过**（不是空判据）；
2. ★★ **每台设备的 KV 通信量 `∝ S`**（`2·S·d·(cp−1)/cp`，与 `S` 严格线性）、**占比与 `S` 无关**；
3. ★★★ **「反例必须证明它真的会错」**：朴素版在原尺度下**看起来完全正确**（`1 ulp` 档），
   **放大 `100` 倍才现形**（满屏 `nan`，而 `inf` 计数是 `0`）。

**延伸**：

- ★★★ **SP 与 CP 的分工、`cp` 的约束、causal mask 跨块** → [03-constraints](03-constraints.md)；
- **SP 切的是什么**（`O(s)` 那一项） → [01-activation-sharding](01-activation-sharding.md)；
- **块内的 `(m, l)` 更新公式** → `40-kernels/softmax-normalization/02-online-softmax.md`（此处**回链**）；
- **`S×S` 的访存账** → `40-kernels/attention/03-flash-attention-backward.md`；
- **集合通信原语 / 通信-计算重叠** → `110-communication`）。

## 参考

- Liu, Zaharia, Abbeel, *Ring Attention with Blockwise Transformers for Near-Infinite Context*,
  arXiv:2310.01889，**§1 / §3** —— 置换不变性、统计量重缩放、KV 在环上传递、`O(S²)` 的显存。
  ★★★ **同行评审：本模块不声称已确认** —— arXiv abs 页上
  `ICLR` / `MLSys` / `Accepted` / `Proceedings` / `Conference` / `NeurIPS` **各 `0` 次**；
  该页 Comments 字段的**渲染文本**是 `Code:` 加一个指向
  `github.com/lhao499/llm_large_context` 的链接（**无任何评审字样**）
  ⇒ **凡引用本文的结论，一律带「同行评审未确认」**。
  ★ 引文载体 = `https://ar5iv.labs.arxiv.org/html/2310.01889` 正文，抓取日 **2026-10-08**，
  逐条在**原始字节**上 `count()` = **1**。
- 本仓库 [01-activation-sharding](01-activation-sharding.md) —— SP 的边界（本文的引子）；
  [03-constraints](03-constraints.md) —— CP 的约束与 causal 跨块；
  `40-kernels/softmax-normalization/02-online-softmax.md` —— **块内**那套更新公式的正文
  （★ 本文**不使用**那个词，此处回链）。

## 编译与运行

```sh
cd code && make run        # 三个编号程序，任一断言失败即非零退出
make 02_ring_attention     # 只跑本文的配套程序（★ 需要 numpy）
```

★ 本机的运行条件：`python3 3.13.5` + `numpy 2.4.4`，macOS arm64，
**无 GPU、无多卡、无 NCCL、无真实网络**。
★ 强制限定语：**不是 NCCL，也不是真机** —— 本文的一切数字都是**玩具规模的真跑**，
**不得**支撑关于真实集群性能的结论。
★★ **判据依赖 BLAS** ⇒ **换机器可能需要重新标定**。
★★ **种子取 `991`**（**不是 2 的幂**），写在 `_make_inputs()` 的**函数体里**。

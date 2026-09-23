---
title: MHA / MQA / GQA：一条 KV 服务几个 query 头
slug: mha-mqa-gqa
domain: 80-model-architectures
topic: attention-variants
level: 2
difficulty: intermediate
prerequisites:
  - 80-model-architectures/transformer/01-block-and-residual-stream
tags: [mha, mqa, gqa, kv-heads, group-size, uptraining, kv-budget]
related:
  - 40-kernels/attention/05-decoding-attention
  - 130-quantization/kv-cache-quantization/01-kv-cache-basics
  - 80-model-architectures/attention-variants/02-mla-low-rank
code:
  - code/01_mha_mqa_gqa.py
status: stable
updated: 2026-09-22
---

# MHA / MQA / GQA：一条 KV 服务几个 query 头

> **三种结构只差一句话：一条 K/V 服务几个 query 头。** `G` 个组就是 `G` 条 KV；
> `G = 1`（MQA）与 `G = h_q`（MHA）是这条轴的两个**端点**——而这两个端点「等价于 MQA / MHA」
> 是论文写明的**定义**，不是测出来的结论。
> 这笔取舍的账是 **KV 的元素数 `1/G`**；代价是表达能力，而它可以用 **5% 的算力**买回来
> —— 但那个 5% 买的是 **MHA→MQA** 的 uptraining，**不是** GQA 的。

## 0. 背景与动机

一个 `L` 层的注意力层，每层要存每个已生成 token 的 K 与 V。放几份？两种极端都有名字：

| 结构 | `heads_q` | `heads_kv` | 每层 KV 的份数 |
|---|---|---|---|
| **MHA**（2017 原版） | `h_q` | `h_q` | `h_q` 份（每头一份） |
| **MQA**（2019） | `h_q` | **`1`** | **1 份**（所有头共享） |
| **GQA**（2023） | `h_q` | `G` | `G` 份（每 `h_q/G` 个头共享一份） |

三种结构**不改变任何一条算术式子**：分数照样是 `q·k`、照样过 softmax、照样乘 `v`。
变的只有**一件事**——第 `h` 个 query 头去取**哪一条** KV。

**为什么这个问题在架构层重要**：decode 阶段每生成一个 token 都要把**全部**历史的 K/V 读一遍，
而这些字节被 `h_q` 个 query 头**复用**（正文在 `40-kernels/attention/05-decoding-attention.md`
与 `130-quantization/kv-cache-quantization/01-kv-cache-basics.md:48-76`，此处回链）。
⇒ **减少「份数」是唯一能同时动到那一侧的结构旋钮**，而它不需要改算法。

★ **本域的边界（计划 §2）**：KV 的**字节账**与「复制 KV 后与 MHA 逐位相同」的**实现与比较**
都不是本域的正文（分别在 `130-quantization/kv-cache-quantization/01:48-76` 与
`40-kernels/attention/05-decoding-attention.md:60,66-94`、`…/code/verify.py` 的 A4/B8）。
**本域讲的是「共享图景」与「uptraining」**——即：三种结构的 head 关系是什么、
以及从一个结构走到另一个结构要付什么。

## 1. 原理

### 1.1 定义（论文逐字）

GQA 论文（arXiv 2305.13245 §2.2）逐字：

> "Grouped-query attention divides query heads into **`G` groups**, each of which shares a single
> key head and value head. **GQA-g** refers to grouped-query with `G` groups. **GQA-1**, with a
> single group and therefore single key and value head, **is equivalent to MQA**, while
> **GQA-h**, with groups equal to number of heads, **is equivalent to MHA**."

⚠️ **这两句是「定义上的等价」，不是「测出来的结论」。** 论文用的词是 `is equivalent to`。
本域的配套程序把这一点写在了断言旁边（`01_mha_mqa_gqa.py` 的 `A.2` 之后）——
**不能用定义冒充证据**（这正是计划 §0.1 的 **P17**：探针初版把同一行代码写了两遍，
于是「共享式 == 复制式」的差报出 `0.000e+00`；那个数是**构造出来的，不是测出来的**）。

由定义直接得到的一条**结构约束**：`G` 必须**整除** `h_q`（「均分成 `G` 组」这句话要有定义）。
⇒ `G` 的取值空间是 **`h_q` 的因子格**，不是连续可调。两个真实模型：

| 模型 | `heads_q` | `heads_kv` | 组数 `G` | KV 元素数倍率 |
|---|---|---|---|---|
| Meta-Llama-3-8B | 32 | 8 | **4** | `1/4` |
| Qwen3-30B-A3B | 32 | 4 | **8** | `1/8` |

（两个字段都是 **config 原生**字段，[B] 级，转记自
`45-kernel-lifecycle/requirement-discovery/code/specs.py` 的两份 config；
本域副本在 `code/specs.py`，带 `# 同值位置：`。）

★ **两者的组数不同（4 vs 8）**，说明「GQA」并不是一个定值选择，而是一条**轴**。

### 1.2 共享图景：就是一张「谁取哪一条」的表

把 `h_kv` 记作 KV 头的条数（`= G`），第 `h` 个 query 头取第 `h // (h_q // h_kv)` 条 KV。
本机实测（`h_q = 8`，`python3 3.13.5` + `numpy 2.4.4`）：

| `h_kv` = `G` | 每个 query 头拿到的 KV 头编号 |
|---|---|
| **1**（MQA） | `[0, 0, 0, 0, 0, 0, 0, 0]` |
| 2 | `[0, 0, 0, 0, 1, 1, 1, 1]` |
| 4 | `[0, 0, 1, 1, 2, 2, 3, 3]` |
| **8**（MHA） | `[0, 1, 2, 3, 4, 5, 6, 7]` |

**这张表就是三种结构的全部差别。** 第一行与最后一行正是那两个端点：
「全部映射到 0 号」= MQA，「恒等映射」= MHA。

### 1.3 账：KV 的元素数按 `1/G` 走（推导 + 反解）

口径必须先写死：这里数的是**元素数**（KV 头的条数之比），**不是字节**。
（字节账的正文在 `130-quantization/kv-cache-quantization/01-kv-cache-basics.md:48-76`；
那边记着这条算式最容易犯的错——**`kv_heads ≠ attn_heads`**，见 §4 第 2 条。）

**推导（两步）**。记每层要缓存的 KV 的**条数**与**每条宽度**两个因子。缓存的总量是它们的乘积：

```
总量  ∝  (条数) × (每条宽度)
```

- **条数**：MHA 是 `h_q`；GQA-G 是 `G`（由 §1.1 的定义：`G` 个组、每组一条）。
- **每条宽度**：由 §1.2 的共享图景可见，**它只取决于 `head_dim`**，与「几组」无关。
  ★ 这一条不是嘴上说的：本程序 `C.2` 逐个 `G` 抽数据算过，**分数矩阵的形状恒为 `(S_q, S_k)`**
  ⇒ 组数变化**不进入**「宽度」这个因子。

于是两个因子的比值分别是 `G/h_q` 与 `1`，乘起来：

```
倍率 = (G / h_q) × 1 = 1 / G                                     …… (1)
```

**反解（暴露设计意图）**：把 (1) 倒过来看——如果**用 `heads_q` 当 KV 的条数**，
那么「条数」这个因子就不再随 `G` 变（比值 = 1），倍率变成 `1`。把 `1` 与正确的 `1/G` 相比：

```
误差倍数 = 1 ÷ (1/G) = G                                          …… (2)
```

⇒ 「用错字段会高估 `G` 倍」这句话**是从同一个式子里反解出来的**，不是经验之谈
（本机算例：Llama-3-8B `G = 4` ⇒ 高估 4 倍；Qwen3-30B-A3B `G = 8` ⇒ 高估 8 倍）。
★ 这也说明了为什么 `num_key_value_heads` 必须是 config 里的**原生字段**：
它**不是**能从 `hidden_size` 或头数**推**出来的量（那两个字段推不出「几组」这个设计选择）。

本机实测的倍率表（用 `Fraction` 比较 ⇒ **精确有理数**，不是浮点）：

| `h_kv` = `G` | `G` | 元素数倍率（精确） |
|---|---|---|
| 1 | 8 | `1/8` |
| 2 | 4 | `1/4` |
| 4 | 2 | `1/2` |
| 8 | 1 | `1` |

⇒ **`G` 越大（越像 MHA）倍率越大**（`h_kv` 越多份）；`G` 越小越省。
两个真实模型的数：Llama-3-8B `1/4`、Qwen3-30B-A3B `1/8`。

★ **同一条断言的「实现与比较」不在本域**：`40-kernels/attention/05-decoding-attention.md:60,66-94`
与 `…/code/verify.py` 的 **A4**（`:509-529`）/ **B8**（`:1008-1040`）已经做了，且那边用的是
浮点误差 `4.126e-16 / 6.918e-16 / 4.685e-16`（本仓库另一个程序的输出）。
**本域只写「共享图景」这一新增部分**——即上两节的定义与那张映射表。

## 2. 执行映射

**「共享」在真实系统上的含义是「读一次、用 `r` 次」**（`r = h_q/h_kv = G`）：
K/V 张量的**读取**字节数按 `1/G` 下降，而**分数矩阵的形状一点没变**。
本机实测：`G` 从 1 变到 8，分数矩阵的形状恒为 `(S_q, S_k)`（程序 `C.2` 逐个 `G` 抽数据算过）
⇒ **改的只有复用关系，没改算式**。

具体的模块名（以 HuggingFace 的命名为例，本域不复核命名）：
`self_attn.num_key_value_heads` 决定 `k_proj` / `v_proj` 的输出宽度是
`h_kv · head_dim`（而不是 `h_q · head_dim`）；前向里 `k`/`v` 在 head 维上被
`repeat_interleave(h_q // h_kv)` 展开后才与 `q` 一起进 attention kernel。
**这一步展开是纯索引操作**——它不产生新的数值。

**本机实测的量级**（`python3 3.13.5` + `numpy 2.4.4`，macOS arm64，**无 GPU**）：

| 量 | 值 | 口径 |
|---|---|---|
| 共享式 vs 复制式的最大逐元素差 | **`9.992e-16`** | `h_q=8`、`d_head=32`、`S_q=12`、`S_k=9`、`G ∈ {1,2,4,8}` 取最大 |
| 参照实现的调用计数 | **`0`** | 跑完后**计数表整个是空的** |
| 分数矩阵形状 | `(12, 9)`，与 `G` 无关 | 同上 |

★ 那 `9.992e-16` **不是 `0.0`**，而且**不该是 `0.0`**：两条**真正独立**的代码路径
（① 共享 KV + 物化分数矩阵的稠密 softmax；② 把 KV **物化复制**成 `h_q` 份 + 流式 online softmax）
的逐元素差必然停在机器精度上。计划 §3 表 A ① 的原稿要求「差 `== 0.0` **且**参考实现独立」
—— **这两条互斥**（计划 §0.1 的 **P18**：这是「要不到」，不是「跑不了」）。⇒ 判据用**机器精度档**。

> ⚠ **本域没有 GPU**：上表是本机 `numpy` 上一个玩具 attention 的推导值与实测值，
> **不是任何真实模型的测量**。它们支撑的是「三种结构只差一张映射表」这条**结构**结论。

## 3. 实现与演进

| 阶段 | 做法 | 它换来了什么、付了什么账 |
|---|---|---|
| MHA | `h_kv = h_q` | 表达力最强；KV 份数最多（`r = 1`） |
| **MQA**（2019） | `h_kv = 1` | KV 份数降到 `1/h_q`；代价是质量（见下） |
| **GQA**（2023） | `1 < G < h_q` | 在两者之间**插值**：质量接近 MHA、速度接近 MQA |
| 现代主流 | 按模型自己选 `G` | Llama-3-8B 选 4、Qwen3-30B-A3B 选 8（本域能核实的两个 config） |

### 3.1 uptraining：**结构可以事后改**

GQA 论文（arXiv 2305.13245）摘要在这一点上写得很具体（逐字）：

> "(1) propose a recipe for **uptraining** existing multi-head language model checkpoints into
> models with MQA using **5% of original pre-training compute**, and (2) introduce grouped-query
> attention (GQA) …"

两条必须与数字一起引用的限定（**表 B 引用判据**，见 §参考）：

1. ★★ **`5%` 这个成本是「把 MHA uptrain 成 MQA」的**（摘要第 (1) 条），**不是 GQA 的**。
   GQA 是同一篇的第 (2) 条**新提出**的结构，它的成本没有在这句话里给。
2. 论文对 GQA 的结论用词是 **`quality close to` multi-head attention**（§1 逐字再现：
   "We show that uptrained GQA achieves quality close to multi-head attention with comparable
   speed to MQA"）—— **`close to` 是「接近」，不是「持平」**。

**uptraining 的两步**（§2.1 逐字，本域只引不重推）：

- **转化**：把 checkpoint 的结构改掉。MHA → MQA 时
  "The **projection matrices** for key and value heads are **mean pooled** into single projection
  matrices, which we find works better than selecting a single key and value head or randomly
  initializing new key and value heads from scratch."
  MHA → GQA 时："we construct each group key and value head by **mean-pooling** all the original
  heads within that group."
- **继续预训练**："The converted checkpoint is then pre-trained for **a small proportion α of its
  original training steps** on the same pre-training recipe."

⇒ **本域的账**：结构的选择**不是一次性的**——一个已经训好的 MHA checkpoint 可以被
**改造成** GQA/MQA 再用少量算力补回来。这条把「架构选择」从「训练前定死」变成了可迭代的取舍。

### 3.2 两个模型的组数不同，正是「插值」的现场

Llama-3-8B 用 `G = 4`、Qwen3-30B-A3B 用 `G = 8`。两者都在 `1 < G < h_q = 32` 内，
但位置不同：`G = 4` 离 MQA 近（省），`G = 8` 离 MHA 近（表达力强）。
**没有哪一个是对的**——这条轴上每一点都是一组取舍（本域主线）。

## 4. 踩坑与失败模式

**① 把「`GQA-1 ≡ MQA`、`GQA-h ≡ MHA`」当成测出来的结论。**

- **现象**：写一条断言「`GQA-1` 的输出 == `MQA` 的输出」，程序报 `0.000e+00`，然后引用它当证据。
- **成因**：这是**定义上的等价**（论文用的词是 `is equivalent to`）。更危险的一种写法是
  两边**共用同一行代码**（计划 §0.1 的 **P17** 现场：探针初版的 `out_shared` 与 `out_naive`
  是逐字符相同的同一行）⇒ 那个 `0.0` 是**构造出来的**，它什么都证明不了。
- **规避**：真正的证据是**两条独立代码路径**（共享式稠密 vs 复制式流式）给出同一个输出，
  并配**「参照实现的调用计数恰为 0」**（`00-meta/glossary.md` 的「reference implementation（参考实现）」条的机械判据）。
  ★ 还要区分两种 `0`：**机器精度档**（真独立路径的必然结果，本机 `9.992e-16`）与
  **构造出来的精确 `0.0`**（两边共用代码）。
  ★★ **这个「假通过」本程序刻意复现了一遍**（`C.4` 是**对照组**）：构造一个
  **直接调用被测实现**的「假参照」后，逐元素差是**精确的 `0.0`**，而 `A.3` 的容差判据
  （`< 1e-14`）**照样通过** ⇒ 容差对这种情况**没有分辨力**。
  ★ 揭穿它的是两把**机械**的尺子：`inspect.getsource` 不等（`A.5`）与 **AST 调用表**
  （假参照里含 `attn_dense` = `True`、真参照里 = `False`；与 `D.8` 用的是同一个探测器）。
  计划 §3 Chunk 2 的验收要求「这个现象必须能被复现」，`C.4` 就是它的落法。

**② 把 `heads_q` 当成 KV 头数。**

- **现象**：算 KV 元素数/字节时用了 `h_q`，倍率算出 `1` 而不是 `1/G`。
- **成因**：GQA 模型上 `kv_heads ≠ attn_heads`，这个等价关系**断了**。本机算例：
  Llama-3-8B 高估 **4 倍**、Qwen3-30B-A3B 高估 **8 倍**（恰好就是 `G`）。
- **规避**：只从 config 的 `num_key_value_heads` 取值；**永远**不要用 `num_attention_heads` 顶替。
  （同一处坑在 `130-quantization/kv-cache-quantization/01-kv-cache-basics.md:76` 有正文，此处回链。）

**③ 把 `G` 当成连续可调。**

- **现象**：想「设 `G = 3` 试试」，结果组大小是 `8/3` 个头——不存在这样的分组。
- **成因**：`G` 必须整除 `h_q`（本机反例：`8 % 3 = 2`、`32 % 6 = 2`）。`G` 的取值空间是因子格。
- **规避**：把「整除」写成断言（本程序 `A.1`），而不是当成理所当然。

**④ 把「`1/r` 的字节账」当成本域的内容。**

- **现象**：在本域文档里重推一遍 KV 字节公式。
- **成因**：那是**别的域**的正文（`130-quantization/kv-cache-quantization/01:48-76` 的六项乘积与
  `40-kernels/attention/05` 的 `1/r` 表）。本域重复它只会制造**两处会漂移的副本**。
- **规避**：本域只写**共享图景与 uptraining**，字节账一律回链（本程序连输出里都不出现字节数）。

## 5. 判据表

| 可机械判定的条件 | 结论 |
|---|---|
| `h_kv == 1`（映射表全 0） | 这是 **MQA**：所有 query 头共享同一条 KV。★ 它与 `GQA-1` 的等价是**定义**，不是测出来的 |
| `h_kv == h_q`（映射表恒等） | 这是 **MHA**：每头一条自己的 KV。同上，等价是**定义** |
| `h_q % G != 0`（如 `h_q=8, G=3`） | 这样的 GQA **不存在** —— `G` 的取值空间是 `h_q` 的因子格 |
| KV **元素数**倍率 | 精确 `1/G`（本机用 `Fraction` 验过：`1/8, 1/4, 1/2, 1`）；Llama-3-8B `1/4`、Qwen3-30B-A3B `1/8` |
| 用 `heads_q` 代替 `heads_kv` 算倍率 | 高估恰好 **`G` 倍**（本机：Llama 4×、Qwen3 8×）—— 不要用这两个字段互换 |
| 两条独立路径的逐元素差 | 必须在**机器精度档**（本机 `9.992e-16 < 1e-14`）；差报出 `0.0` ⇒ 先查两边是不是同一份代码 |
| 参照实现的调用计数 | 必须**恰为 0**（本机：计数表整个为空）——否则上面那个「差」没有意义 |
| 引用 `5%` 这个训练成本 | 它属于 **MHA→MQA 的 uptraining**，不是 GQA 的成本；GQA 的结论用词是 `quality close to`（接近），不是「持平」 |
| 引用 MQA 的质量损失 | **必须带出处**：MQA 论文自述 `minor quality degradation`；**GQA 论文的措辞更强**（`quality degradation and training instability`）——两篇强度不同 |
| 想知道 KV 的**字节**账 | 不在本文：正文在 `130-quantization/kv-cache-quantization/01-kv-cache-basics.md:48-76` 与 `40-kernels/attention/05-decoding-attention.md:60,66-94` |

> ★ **表 A 的数值判据**（计划 §3 Chunk 2）：① 两条独立路径的最大逐元素差 `< 1e-14`（**机器精度档**，
> 不是 `== 0.0`）；② KV 元素数倍率精确 `1/G`。两条都在 `code/01_mha_mqa_gqa.py` 的
> `A.3` / `B.2` 里，阈值本身被 `D.2` 逐位钉住（放宽阈值 ⇒ 判据退化成恒真）。

## 6. 小结与延伸

**一句话**：MHA / MQA / GQA 是**同一条轴上**的三个点，轴的名字是「一条 KV 服务几个 query 头」；
GQA 的全部机制就是**一张映射表**，它的账是 KV 元素数按 `1/G` 走，代价是表达能力
——而后者可以用 **5% 的算力**（MHA→MQA 的 uptraining）买回来。

- **向前**：`transformer/01-block-and-residual-stream`（残差流与 block 的构成）是本文的前置。
- **同主题的下一篇**：`02-mla-low-rank` —— 它换了一个方向：**不减少 KV 头的条数，而是把每条 KV 压小**。
  两者是同一笔账（KV 的预算）的两条路。
- **`03-window-and-hybrid`** 讲**感受野**那一侧的取舍，并说明它与 GQA 是**叠加**的（不是二选一）。
- **KV 的字节账与 decode 的算术强度**：正文在 `130-quantization/kv-cache-quantization/01:48-76`
  与 `40-kernels/attention/05-decoding-attention.md`（正文在彼，此处回链）。
  ★ 那边有一条与本域这一轴直接相邻的结论（`r` 与 decode 侧那个量的关系）——
  **它是那一侧的正文，本域不重复也不改写**。
- **变体对 kernel 结构的影响**：正文在 `40-kernels/attention/04-warp-specialization-and-variants.md:126-156`
  （正文在彼，此处回链）。

## 编译与运行

```bash
cd code
make run                       # 三个编号程序依次跑；本程序是 01_mha_mqa_gqa.py
python3 01_mha_mqa_gqa.py      # 只跑本程序
```

- 配套程序：**`code/01_mha_mqa_gqa.py`**（编号与外键严格对应）。
- 依赖：`numpy`（`requirements.txt` 里**只有**它）。**本程序不需要网络、不需要 GPU。**
- 失败标记：`@@ FAIL: attention-variants`（本主题自定义，**不跨主题照抄**）。程序内部有
  「标记自检」断言守着它（`D.5` / `D.6`）。
- 程序的输出分五段：`D`（口径/阈值/specs 钉住，**刻意放在最前**，见程序里的说明）、
  `A` 共享图景与两条独立路径、`B` 元素数倍率与那个坑、
  `C` 边界与反例（`C.3` 缩放不是装饰、`C.4` **共享代码的对照组**）、
  `D（续）` 标记自检与参照实现的独立性。

## 参考

### 表 B：引用判据（**不能被程序判定**，用 `raw.count(字面量)` 在原始字节上计数）

下列计数都是**实跑**的。抓取形式是最简的 `curl -sS`（本机实测 2026-09-22；
`WebFetch` 对 ar5iv 被策略拦截 ⇒ 用 `curl`，见 `CONTRIBUTING:266-271`）：

```bash
curl -sS -o /tmp/arx_2305.13245.html https://ar5iv.labs.arxiv.org/html/2305.13245  # 144244 B
curl -sS -o /tmp/arx_1911.02150.html https://ar5iv.labs.arxiv.org/html/1911.02150  # 214751 B
python3 - <<'EOF'
g = open('/tmp/arx_2305.13245.html','rb').read()      # ★ 原始字节，不转文本
m = open('/tmp/arx_1911.02150.html','rb').read()
for s in ["5% of original pre-training compute", "quality close to",
          "quality degradation and training instability", "is equivalent to",
          "is equivalent to MQA"]:
    print("GQA", s, g.count(s.encode()))
for s in ["minor quality degradation"]:
    print("MQA", s, m.count(s.encode()))
EOF
```

实测（`python3 3.13.5`）：

| 字面量 | 计数 | 位置 |
|---|---|---|
| `5% of original pre-training compute` | **1** | GQA 摘要第 (1) 条（**MHA→MQA** 的 uptraining 成本） |
| `quality close to` | **3** | GQA 摘要与 §1、§2.2 |
| `quality degradation and training instability` | **1** | GQA §1（对 MQA 的评价） |
| `is equivalent to` | **2** | GQA §2.2（**定义上的等价**，见 §1.1） |
| `is equivalent to MQA` | **0** | ★ **这是同一条计数口径的另一个方向**：原文那两个 head 名被 `<span>` 标记与 `is equivalent to` **切开了** ⇒ 一个**跨标记的字面量在原始字节上计不到**。所以「取到原文」与「读对原文」是两件事（`CONTRIBUTING:277-280`），跨标记的引文必须**先看清结构**再引 |
| `minor quality degradation` | **2** | MQA 摘要（**强度弱于** GQA 论文的措辞） |

> ★ **`5%` 的作用域**：逐字落在「uptraining … into models with **MQA**」这一句里。
> 把它写成「GQA 的成本」是**引文作用域不匹配**。
> ★ **两个 `gradation` 措辞的强度差**：引用 MQA 的质量损失时**两篇都引**，
> 不要只用 MQA 自述的那一句（它的措辞最轻）。

**外部规格值 / 论文**

- **GQA**：arXiv 2305.13245（摘要 `Accepted at EMNLP 2023` ⇒ **已确认同行评审**；
  抓取 `https://ar5iv.labs.arxiv.org/html/2305.13245`，2026-09-22，**144244 B**，
  `sha256[:16] = 5c8b7d904023a14f` —— 与计划 §C.2 逐位一致）。**[C]**
- **MQA**：arXiv 1911.02150（**无 venue 线索 ⇒ 未确认同行评审**；抓取同上，2026-09-22，
  **214751 B**，`sha256[:16] = 3dd6c51a83cdc945`）。**[C]**
- two config 的 `num_attention_heads` / `num_key_value_heads`：转记自
  `45-kernel-lifecycle/requirement-discovery/code/specs.py` 的 `LLAMA3_8B_CONFIG` /
  `QWEN3_30B_A3B_CONFIG`（**[B]**，该模块自注「经 HuggingFace 与 torchtune 双向核对」/
  「经 4 处独立转抄核对」）。本域副本在 `code/specs.py`。

**本机实测值**

- `code/01_mha_mqa_gqa.py` 的 `make run` 输出：`python3 3.13.5` + `numpy 2.4.4`，
  macOS arm64，**无 GPU**。口径：`h_q = 8`、`h_kv ∈ {1,2,4,8}`、`d_head = 32`、
  `S_q = 12`、`S_k = 9`、种子 `20260922`、online softmax 的块大小 `4`。
  **全部是推导值**，不是真实模型的测量。
- `probes/probe_80.py` 的 Q5（D13/D14 的口径记录）与本程序**同源**：
  那边 `h_q=8 h_kv=4 r=2` 报 `8.882e-16`（输出量级 `2.010`）；本程序在 `G=2` 上复现了
  同一个量级。★ 那个探针的**初版是恒真断言**，入仓前已改成两条不同路径（见 `probes/README.md`）。

**上游契约**

- `40-kernels/attention/04-warp-specialization-and-variants.md:128`：「三个常见变体，本文只讲
  **它们对 kernel 结构的影响**，**定义不重推**」—— 这是本主题要接住的那处**原为悬空让渡**；
  本文与 `02` / `03` 是它的三份正文。
- `00-meta/glossary.md:237`（GQA / MQA 词条）：**本域不得另立同名词条**；词条的定义正文在本域，
  glossary 只放短条目并回链。

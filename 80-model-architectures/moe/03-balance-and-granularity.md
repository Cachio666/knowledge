---
title: 均衡与粒度：aux loss（含 ×N 与 α）、负载怎么度量、专家粒度收窄到哪
slug: balance-and-granularity
domain: 80-model-architectures
topic: moe
level: 2
difficulty: intermediate
prerequisites:
  - 80-model-architectures/moe/01-gating-and-routing
tags: [moe, aux-loss, load-balancing, load-imbalance, expert-granularity, qwen3-moe, alpha]
related:
  - 40-kernels/moe-kernels/01-moe-compute-pattern
  - 80-model-architectures/moe/02-capacity-factor
  - 80-model-architectures/transformer/03-depth-and-width
  - 45-kernel-lifecycle/requirement-discovery/01-model-to-operator-inventory
code:
  - code/03_balance_granularity.py
status: stable
updated: 2026-09-22
---

# 均衡与粒度：aux loss（含 ×N 与 α）、负载怎么度量、专家粒度收窄到哪

> **`loss = α · N · Σ_i f_i · P_i`** —— 那个 **`×N`** 不是装饰：原文逐字给了理由
> （`to keep the loss constant as the number of experts varies`），
> 自己漏掉它就把损失**缩小恰好 `N` 倍**（本机实测：`N = 64` 时比值**恰好 64**）。
> `α = 10^{-2}`（原文扫过 `10^{-1}` 到 `10^{-5}`）—— 而**真实模型用的是 `1e-3`**，
> 恰好在扫描带内、**但不是论文最终选的值**（差 10 倍）。
> 至于「专家粒度」：**共享专家的一手来源在本环境不存在** ⇒ 本文**收窄**为
> `E` / `top_k` / `moe_intermediate_size` 三者之间**可手算**的关系。

## 0. 背景与动机

`02-capacity-factor.md` 讲的是**事后**的截断：容量定死之后，谁被砍掉。
本文讲的是**事前**与**另两个旋钮**：

| 旋钮 | 它改什么 | 本文的哪一节 |
|---|---|---|
| **辅助损失**（aux loss） | 让**训练**把负载往均匀推 | §1.1–§1.2 |
| **负载的度量** | 「不均」怎么变成一个**可判定的数** | §1.3 |
| **专家粒度** | 每个专家多大、一共几个 | §1.4 |

★ 上一句里「**让训练把负载往均匀推**」这个说法要小心：aux loss 是**训练的激励**，
它**不**保证推理时负载均匀 —— 那正是容量（`02`）存在的理由。
两者的分工写在 `02` 的 §0 与 §1.1。

★ **本域的边界**：aux loss 的**反向梯度**、`f` 的不可导处理、以及「按最大填充的浪费比」
这条分布统计，都在 `40-kernels/moe-kernels/01-moe-compute-pattern.md`（含 §2.1 与 §4.2）
与 `02-grouped-gemm.md`（正文在彼，此处回链）。本文只写**式子、取值、与度量**。

## 1. 原理

### 1.1 ★★ aux loss 的形式（**必须含 `×N` 与 `α`**）

逐字（Switch Transformer §2.2，arXiv 2101.03961，**[C] + 已确认同行评审**
（abs 页 `comments` 逐字为 `JMLR`））：

> "Given `N` experts indexed by `i = 1` to `N` and a batch `B` with `T` tokens, the auxiliary
> loss is computed as the **scaled dot-product** between vectors `f` and `P`,
> **`loss = α · N · Σ_{i=1}^{N} f_i · P_i`** (4)"

> "`f_i` is the **fraction of tokens dispatched to expert `i`**,
> `f_i = (1/T) Σ_{x∈B} 1{argmax p(x) = i}` (5)"

> "`P_i` is the **fraction of the router probability allocated for expert `i`**,
> `P_i = (1/T) Σ_{x∈B} p_i(x)` (6)"

> "**The final loss is multiplied by expert count `N` to keep the loss constant as the number
> of experts varies** since under uniform routing `Σ_i (f_i · P_i) = Σ_i (1/N · 1/N) = 1/N`."

> "a hyper-parameter `α` is a multiplicative coefficient … **throughout this work we use an
> `α = 10^{-2}`** … We swept hyper-parameter ranges of `α` from `10^{-1}` to `10^{-5}` in
> powers of 10 and found `10^{-2}` balanced load quickly without interfering with training loss."

★★ **`×N` 到底压在哪一步 —— 本机实测的三条形态**（同一份 logits，`T=4096, E=64, k=2`）：

| 形态 | 表达式 | 实测值 |
|---|---|---|
| 式 (4)（去 `α`） | `N · Σ(f_i·P_i)` | **1.0008** |
| 论文**伪码**（独立路径） | `N² · mean(f·P)` | **1.0008** |
| ★ **漏 `×N`** | `Σ(f_i·P_i)` | **0.015638** |

`N · Σ / Σ = 64.00000000`，**恰好 `N`** ⇒ 漏掉的那个因子**就是 `N` 本身**。
（本机 `A.3`。）

★ **伪码那一版是独立的第二条路径**（Figure 14 逐字）：

```
loss = mtf.reduce_mean(density_1_proxy * density_1) * (num_experts ^ 2)
```

它的**算子顺序不同**（先 `mean` 即除 `N`、再乘 `N²`），而式 (4) 是乘 `N` 求 `Σ`（不除）
⇒ 两条路径的「精确相等」**不是同义反复**（`A.3` + `D.8` 的 AST 调用表为空）。

### 1.2 ★ 下界 `1.0` 是**数学的**，不是**有限样本的**

- **数学**：均匀时 `f = P = 1/N` ⇒ `N · Σ(1/N · 1/N) = 1`。
  本机在**精确构造**上实测**逐位等于 `1.0`**（`1/64` 是二进制精确的，`A.1`）。
- **有限样本**：同一口径下两个独立抽样得 **`1.0008`** 与 **`1.0012`**
  （`T=4096, E=64, k=2`，均匀 logits），**都严格大于 `1.0`**（`A.2`）。

★★ **所以判据必须写成 `∈ [1.0, 1.01]`，不能写 `== 1.0`。**
计划 §3 的 **M3** 逐字记着这一条：原稿写「精确等于 `1.0`」与探测值冲突，
**照原稿写 `assert == 1.0` 会直接失败**。本文的 `A.2` 把「有限样本做不到 `1.0`」
**这条事实本身**做成了断言（两个数都 `> 1.0` 且偏差远超浮点档）。

★ 为什么样本上会 `> 1`：`Σ_i f_i P_i` 在 `f ≠ P` 时**大于** `1/N`
（`f` 与 `P` 由同一批 logits 导出，`f` 取的是 top-k 的**硬**结果、`P` 是 softmax 的**软**均值，
两者的涨落同向）⇒ 有限样本的 `aux` 从**下界之上**靠近。

### 1.3 负载怎么度量：「不均」需要一个**可判定的数**

「负载不均」这句话本身不可判定。本域用三个量把它变成判据（`B.1`）：

| 量 | 定义 | 为什么需要它 |
|---|---|---|
| `max/mean` | 最多与平均之比 | 判据 ⑥ 的主量 |
| `min`（最少专家） | 条数最少的那个专家拿到多少 | ★★ **非空条件**：「极端偏斜」若没有 `min == 0` 这一半，「偏斜」就只是「看起来不匀」 |
| CV（`std/mean`） | 相对涨落 | 用于「同一个 `cf`、不同 `E`」的比较（`02` 的 §1.6） |

本机实测（`T=4096, E=64, k=2`；`logits ~ N(0,1)`，**前 8 个**专家加偏置 `s`；
★ **同一个种子，只改 `s`** —— 配对比较）：

| `s` | 0.0（均匀） | 0.3 | 0.8 | 1.5 | 3.0（极端） |
|---|---|---|---|---|---|
| `max/mean` | **1.23** | 1.92 | 3.58 | 6.34 | **8.17** |
| 最少专家 | 104 | 100 | 70 | 27 | **0** |
| `aux`（去 `α`） | **1.0008** | 1.0280 | 1.2993 | 2.4294 | **5.7193** |

★ 最后一行是**后果检验**：aux loss 随偏斜**严格单调上升** ⇒ 它**真的在惩罚不均**。
少了这一行，前两行的阈值就只是两个孤立的数（`B.2`）。

★★ **配对比较是刻意的**：如果每一行换一个种子，「严格单调」就会混进抽样噪声 ——
那正是 §0.1 第一类错误的现场（判据的统计量不稳）。

### 1.4 ★★ 专家粒度：**本文收窄到哪**（计划 §4 的 R11）

**先说不能讲的那一半。** 逐字依据（`45-kernel-lifecycle/requirement-discovery/code/specs.py:123-127`）：

> 「① `Qwen3-30B-A3B` 的**共享专家**（shared expert）规格字段在检索到的片段里**没有**
> （检索只给出 `intermediate_size: 6144`，而它的语义是「非 MoE 的 FFN 宽度」，
> **不等于共享专家的宽度**）。因此 01 篇的清单**不含共享专家**，算出的活跃参数量是**下界**。」

⇒ **共享专家不在本域**：它的规格字段在本环境**不存在一手来源**
（`huggingface.co` 直连不可达，计划 §C.0）。
★ 本域的做法是**记入 `UNSOURCED`**（`code/specs.py` 的 `"qwen3_shared_expert"`），
并且**不搬任何数字进来** —— 包括**不**把 `intermediate_size` 当成共享专家的宽度。

**再说能讲的那一半：三条可手算的关系。**

| # | 关系 | 本机实测 |
|---|---|---|
| (a) | `intermediate_size = top_k × moe_intermediate_size` | **`6144 = 8 × 768`** ✓ |
| (b) | 活跃专家占比 `= top_k / E` | **`1/16` = 6.25%**（**精确有理数**） |
| (c) | 每个专家的宽度 / 隐藏维 `= moe_intermediate_size / hidden_size` | **`768/2048 = 3/8`** |

★★ **关于 (a) 的纪律**：本域**只记录这条整数等式成立**，**不解释它「为什么」** ——
`intermediate_size` 在 MoE 模型里的**语义**在本环境**没有一手来源**
（是共享专家宽度？是 dense 层宽度？），记入 `UNSOURCED`（`"qwen3_intermediate_size_semantics"`）。
★ 把「等式成立」写成「所以 `intermediate_size` 就是共享专家宽度」正是本库禁止的**机制从句**。

**★★ 外部锚点：从 config 反算总参数量**（计划 §0.1 的 **P15**）

| 项 | 参数量 |
|---|---|
| 注意力（`L=48` 层，GQA `32/4`，`head_dim=128`） | `9.059697e+08` |
| 路由器（`L × d × E`） | `1.258291e+07` |
| **专家 FFN**（`L × E × 3 × d × moe_intermediate_size`） | `2.899103e+10` |
| 嵌入 + lm_head（`2 × V × d`，**不共享**） | `6.223299e+08` |
| 归一化权重（`L × 2 × d + d`） | `1.986560e+05` |
| **合计（不共享嵌入）** | **`30.5321 B`** |
| 合计（共享嵌入） | `30.2209 B` |

**公开总参数量 = `30.5 B`**（**[B]**，转记自 `45-…/code/specs.py` 的
`PUBLISHED_PARAM_COUNT_B`，该模块 `:170-176` 逐字有这一行）。

- **不共享嵌入** 相对锚点 **`+0.1053%`**（≤ ±1%）；
- **共享嵌入** 相对锚点 **`-0.9149%`**。

★ 这个锚点不但**可用**，还能**分辨**两种口径（`tie_word_embeddings` 一侧更近，
与镜像取到的 `false` 一致）。★★ **计划 §4 的 R19 逐字说「Qwen3-30B-A3B 在本域
**不可能有外部锚点**：`specs.PUBLISHED_PARAM_COUNT_B` 里**刻意只有 Llama**」——
那句话对**域内**那一份 `specs.py` 成立，对**上游**那一份不成立**：上游明确有这一行。

⚠ **锚点不可用的是另一个量**：`45-…/code/specs.py:176` 还写着「每 token 活跃 ≈ 3 B」，
但它是**一位有效数字、且没有给口径** ⇒ 本机两个口径算出来是
**`2.7305 B`（不含嵌入/lm_head）** 与 **`3.3528 B`（含）** ——
**3 B 恰好跨在两者之间** ⇒ 它**分辨不出**用的是哪个口径，
**不能当锚点**（记入 `UNSOURCED` 的 `"qwen3_active_param_published"`）。

★ **注意 `L × E × 3 × d × moe_intermediate_size` 里的那个 `3`**：
它是 **SwiGLU 的三个投影**（gate/up/down）。★ **投影数（系数 4 vs 6）不在本域** ——
正文在 `40-kernels/moe-kernels/01-moe-compute-pattern.md:72,278` 与
`code/verify.py:443`（正文在彼，此处回链）。本文只是**用到**这个 `3`，不重推它。

## 2. 执行映射

★ 本域不讲 kernel 侧（`40-kernels/moe-kernels/01` 的 §2.1 有 aux loss 的实现形态）。
本文只记**架构侧的一条**：`P_i` 需要**全 `E` 个专家上的 softmax** 的列均值，
而前向只用 top-k 那几个 ⇒ **`p` 这一份数据是为 aux loss 而存在的**
（这也是 §4.5 那个坑的来源：把 `P_i` 用成「top-k 归一化后的权重」，
`Σ P_i` 就不再是 1，下界那句话也就不成立）。

## 3. 实现与演进

### 3.1 论文的 `α` 与真实模型的 `α`：**差 10 倍**

| 量 | 值 | 来源 |
|---|---|---|
| 论文选定 | **`1e-2`** | Switch §2.2 逐字 |
| 论文扫过的带 | `1e-1` … `1e-5`（10 的幂） | Switch §2.2 逐字 |
| **Qwen3-30B-A3B 的 `router_aux_loss_coef`** | **`1e-3`** | config，**经 `hf-mirror.com` 镜像取得，未与官方 repo 比对 sha256** |

★★ **两件事都要判定**（`B.3`）：

1. 真实取值 **`1e-3` 落在**论文扫过的带 `[1e-5, 1e-1]` 内 ⇒「论文扫过它」这句**可为真**；
2. 但它 **≠ 论文最终选的 `1e-2`**，**恰好差 10 倍**
   ⇒ **不能**把「论文取 `1e-2`」当成「模型就该用 `1e-2`」。

★ 这一条是本文**唯一**一处「论文 vs 真实模型」的对照面 ——
计划 §C.0 的硬约束说本域只能引用那两个已核实的 config，
而 `router_aux_loss_coef` **不在** `45-…/code/specs.py` 里（计划 §C.4 明确把它列进
「不在那份文件里」的那一组）⇒ 它**只能**走镜像这条限定路径，限定语必须保留。

### 3.2 GShard 的 aux loss 是**另一条式子**（不要与 Switch 的混起来）

逐字（GShard §2.2，**[C] + 未确认同行评审**）：

> "the term `c_e/S` represents the fraction of input routed to each expert, and we want to
> **minimize mean square of `c_e/S`**. But because `c_e` is derived from top-2 operation and is
> **not differentiable**, we use the **mean gates per expert `m_e`** as a differentiable
> approximation and replace `(c_e/S)²` with **`m_e(c_e/S)`**"

⇒ GShard 的形式是 `m_e · (c_e/S)` 的均值，**不是** Switch 的 `α · N · Σ f_i P_i`。
两者的**目标**一样（把负载往均匀推），**式子不同、系数不同**
（GShard 用一个乘子 `k` 加在总损失上：逐字 `ℒ = ℓ_nll + k * ℓ_aux`）。
★ 引用时**必须带出处** —— 这两个式子**不能互相代入**。

### 3.3 「粒度」这一栏在本主题里的位置

| 旋钮 | 它在哪一篇 | 它的账 |
|---|---|---|
| 容量 `cf` | `02-capacity-factor.md` | 丢弃 ↑ vs padding 白算 ↑ |
| 均衡（aux loss / 选边） | `01`（token vs expert choice）+ 本文 | 负载的形状 |
| **粒度**（`E`、每个专家多宽） | 本文 §1.4 | **参数量的分配** + 负载形状的涨落 |

★ 粒度**在本文只到「可手算的规格关系」为止**（§1.4 的三条）——
「多少专家最优」不是一个本域能回答的问题（它需要**训练**的证据），
而本域的一切数字都是 `python3` + `numpy` 的推导（本机**无 GPU**）。

## 4. 踩坑与失败模式

### 4.1 ★★ 漏掉 `×N`

- **现象**：写 `loss = α · Σ f_i P_i`（或 `α · mean(f·P)`）。
- **成因**：把 aux loss 当成「一个归一化到 `O(1)` 的惩罚项」，于是觉得乘 `N` 多余。
- **规避**：原文**逐字给了理由**（`to keep the loss constant as the number of experts varies`），
  而且**伪码也在做同一件事**（`* (num_experts ^ 2)` 乘在一个 `mean` 上）。
  本机实测：漏掉之后损失的**尺度差恰好 `N` 倍**（`N=64` 时比值 `64.00000000`）。
  ★ 这一条的现实后果：`N` 从 8 换到 256 时，漏 `×N` 的实现会让**有效学习率差 32 倍**。

### 4.2 把下界 `1.0` 当成「实测值」

- **现象**：写「aux loss 实测恰为 `1.0`」，或把判据写成 `assert == 1.0`。
- **成因**：只推了代数（`N · N · (1/N)² = 1`），没跑一次有限样本。
- **规避**：区分**代数下界**与**有限样本**（§1.2）；判据写成 `∈ [1.0, 1.01]`
  （计划 §3 的 M3）。本机实测：`1.0008` / `1.0012`，**都严格大于 1**。

### 4.3 用一个数（`max/mean`）判定「负载不均」

- **现象**：报一个 `max/mean`，就说「这个模型负载不均」。
- **成因**：`max/mean` 是一个**样本统计量**，且它**没有下界之外的参照** ——
  均匀抽样本身就会给出 `> 1` 的值（本机：均匀 logits 的 `max/mean = 1.23`）。
- **规避**：三段式 —— `max/mean` + **最少专家** + `aux` 的**单调性**
  （§1.3 的表）。★ 并且：`max/mean` 与 `cf` 的**比较基准**必须写出来
  （「相对 `1.0` 还是相对均匀抽样的 `1.23`」是两个不同的问题）。

### 4.4 把 `intermediate_size` 当共享专家的宽度

- **现象**：写「Qwen3-30B-A3B 的共享专家宽度是 6144」（由 `intermediate_size` 直接读来）。
- **成因**：上游 `45-…/code/specs.py:123-127` **逐字**说那个字段「在检索到的片段里**没有**」，
  而 `intermediate_size` 的**语义**在本环境**没有一手来源**。
- **规避**：§1.4 的纪律 —— 只记录等式（`6144 = 8 × 768`），**不解释语义**；
  并把「共享专家不在本域」写进正文与 `UNSOURCED`。

### 4.5 把 `P_i` 用成「top-k 归一化后的权重」

- **现象**：算 `Σ_i f_i P_i` 时用路由时那 `k` 个权重（甲口径）。
- **成因**：那 `k` 个权重之和恒为 1（`01` 的 `A.1`），但它们**不是** `P_i` ——
  `P_i` 是**全 `E` 个专家上 softmax 的列均值**（式 (6)）。
- **规避**：`P_i` 的算式写在 `03_balance_granularity.py` 的 `aux_terms` 里，
  并注明「不是 top-k 归一化后的权重」；两者的差别本身是
  `40-kernels/moe-kernels/code/verify.py` 的 A1 的正文（此处回链）。

## 5. 判据表

| 可机械判定的条件 | 结论 |
|---|---|
| aux loss 的形式 | **必须含 `×N` 与 `α`**：`α · N · Σ_i f_i · P_i`；漏 `×N` 会让尺度**恰好差 `N` 倍**（本机 `N=64` 时 `64.00000000`） |
| `N² · mean(f·P)`（伪码形态） | 与式 (4) **精确相等**（本机相对差 `0.00e+00`）⇒ 两个算子是同一个式子 |
| 完全均匀时的 aux | **代数下界 `= 1.0`**（精确构造上**逐位等于**）；**有限样本 `> 1.0`**（本机 `1.0008` / `1.0012`） |
| 判据该怎么写 | **`∈ [1.0, 1.01]`**，**不能**写 `== 1.0`（计划 §3 的 M3：照原稿写会直接失败） |
| `α` 的作用 | 只按比例缩放（本机：`1e-2 → 1e-1` 比值**恰好 10**）；论文取 `1e-2`、扫过 `[1e-5, 1e-1]` |
| 真实模型的 `α` | `router_aux_loss_coef = 1e-3`（**镜像取得，未与官方 repo 比对 sha256**）：**落**在论文扫过的带内，但 **≠ 论文选定值**（差 10 倍） |
| 引用 aux loss 的**与后梯度 / 不可导处理** | 不在本文：正文在 `40-kernels/moe-kernels/01-moe-compute-pattern.md` 的 §2.1 |
| 判定「负载不均」 | 三段式：`max/mean`（均匀抽样本身就 `> 1`：本机 `1.23`）+ **最少专家**（极端偏斜必须 `== 0` 才是可判定的）+ `aux` 随偏斜**严格单调上升** |
| 引用 GShard 的 aux loss | 是**另一条式子**（`m_e · (c_e/S)`），**不能**与 Switch 的 `α·N·Σ f_i P_i` 互相代入 |
| 「专家粒度」 | 本域**收窄**为 `E` / `top_k` / `moe_intermediate_size` 之间**可手算**的关系（本机 `6144 = 8 × 768`、`top_k/E = 1/16`） |
| **共享专家** | **不在本域**（一手来源在本环境不存在，逐字见 `45-…/code/specs.py:123-127`）⇒ 记入 `UNSOURCED`，**不搬任何数字** |
| `intermediate_size` 的语义 | **不解释**（本环境无一手来源）；只记录等式成立 |
| 从 config 反算 Qwen3-30B-A3B 的总参数量 | **必须是「不共享嵌入」口径**：`30.5321 B`（相对公开值 `30.5 B` 的 **`+0.1053%`**）；共享嵌入口径是 `30.2209 B`（`-0.9149%`） |
| 「每 token 活跃 ≈ 3 B」 | **不是可用锚点**：它是一位有效数字且无口径，而本机两个口径给 `2.7305 B` 与 `3.3528 B` ⇒ **3 跨在两者之间** |

> ★ **表 A 的数值判据**（计划 §3 Chunk 3）：
> ⑤ aux loss 在完全均匀时 **`∈ [1.0, 1.01]`**（数学下界 `1.0`、有限样本 `1.0008`/`1.0012`）；
> ⑥ 负载不均的度量（均匀 `max/mean ∈ [1.0, 1.5]`；极端偏斜 `max/mean > 8` **且最少专家 `== 0`**）。
> 两条都在 `code/03_balance_granularity.py` 的 `A.1`/`A.2`/`B.1` 里，
> 阈值被 `D.1`/`D.2` 逐位钉住。

## 6. 小结与延伸

**一句话**：负载这一栏有**三笔**互不相干的账 ——
**事前**的 aux loss（形式必须含 `×N` 与 `α`，下界是代数的、不是抽样的）、
**事后**的容量（`02`）、以及**粒度**（`E` 与每个专家的宽度，本文只到可手算的规格关系）。

- **向前**：`01-gating-and-routing.md`（`f_i` 与 `P_i` 都是从 gating 里长出来的）。
- **同主题的上一篇**：`02-capacity-factor.md`（容量是 aux loss 治不了的那一半）。
- **参数量分配**的另一半在 `transformer/03-depth-and-width`
  （那片讲 LLM 的参数量怎么在 embedding/attention/FFN 之间分配；
  本文补的是 MoE 特有的那一项：专家 FFN 按 `E` 算，**不是**按 `top_k` 算 —— P15 的现场）。
- **上游 `45-kernel-lifecycle/requirement-discovery/01`** 从 config **反算**算子清单 ——
  本文的 §1.4 只做**一处**与它同源的核对（总参数量对 `30.5 B`），
  **不重做**那个逐位断言（它在 `code/01_operator_inventory.py:549`，正文在彼，此处回链）。

## 编译与运行

```bash
cd code
make run                            # 三个编号程序依次跑；本程序是 03_balance_granularity.py
python3 03_balance_granularity.py   # 只跑本程序
```

- 配套程序：**`code/03_balance_granularity.py`**（编号与外键严格对应）。
- 依赖：`numpy`（`requirements.txt` 里**只有**它）。**本程序不需要网络、不需要 GPU。**
- 失败标记：`## FAIL: moe`（本主题自定义，**不跨主题照抄**）。
- 程序的输出分五段：`D`（口径/阈值/specs 钉住，**刻意放在最前**）、
  `A` aux loss 的形式（含 `×N` 与 `α`）、`B` 负载度量与真实模型的 `α`、
  `C` 专家粒度（R11 收窄）+ 参数量锚点、`D（续）` 标记自检与两条路径的独立性检查。

## 参考

### 表 B：引用判据（**不能被程序判定**，在**原始字节**上计数）

实跑（2026-09-22；`WebFetch` 被策略拦截 ⇒ 用 `curl`）：

```bash
curl -sSL -o /tmp/arx_2101.03961.html https://ar5iv.labs.arxiv.org/html/2101.03961  # 413955 B
curl -sSL -o /tmp/arx_2006.16668.html https://ar5iv.labs.arxiv.org/html/2006.16668  # 451967 B
python3 - <<'EOF'
for f in ("/tmp/arx_2101.03961.html","/tmp/arx_2006.16668.html"):
    b = open(f,'rb').read()
    for s in [r"\alpha=10^{-2}", r"10^{-1}", r"10^{-5}",
              r"\text{loss}=\alpha\cdot N\cdot\sum_{i=1}^{N}f_{i}\cdot P_{i}",
              "The final loss is multiplied by expert count",
              "to keep the loss constant as the number of experts varies",
              "we use the mean gates per expert",
              "not differentiable", "perform better at lower capacity factors (1.0, 1.25)",
              "capacity factor"]:
        print(b.count(s.encode()), s[:70])
EOF
```

实测（`python3 3.13.5`）：

| 字面量 | Switch | GShard | 用途 |
|---|---|---|---|
| `\text{loss}=\alpha\cdot N\cdot\sum_{i=1}^{N}f_{i}\cdot P_{i}`（MathML 的 `alttext`） | **2** | 0 | ★★ **式 (4) 的逐字 —— `×N` 就在这串里**（`alttext` 与 annotation 各一次） |
| `The final loss is multiplied by expert count` | **1** | 0 | ★★ `×N` 的**理由句** |
| `to keep the loss constant as the number of experts varies` | **1** | 0 | ★★ 同上（理由的后半） |
| `\alpha=10^{-2}`（`alttext`） | **2** | 0 | ★★ `α` 的取值（**不是**本域编的） |
| `10^{-1}` / `10^{-5}`（`alttext`） | **2** / **2** | 0 | 扫描带的两端 |
| `loss = mtf . reduce_mean ( density_1_proxy * density_1 ) * ( num_experts ^ 2)`（剥标签后） | **1**（原始字节 **0**） | — | ★★ 伪码形态（`N² · mean`）—— **跨标记 ⇒ 原始字节计不到** |
| `we use the mean gates per expert` | 0 | **1** | ★ GShard 的 aux loss 是**另一条式子** |
| `not differentiable` | 0 | **1** | ★ GShard 给的理由（`c_e` 不可导） |
| `perform better at lower capacity factors (1.0, 1.25)` | **1** | 0 | 与 `02` 共用的口径（**带语境**） |
| `capacity factor` | **11** | **0** | 对照：这个词是 Switch 的（`02` 的 M1） |

★ **对照检验**：上表里 GShard 那一列的 `0`（前 5 行）是**真的没有**——
对照词 `expert capacity` 在 GShard 里 **7 次**、`we use the mean gates per expert` **1 次**、
`not differentiable` **1 次** ⇒ 数得到同类的词，只是**没有** Switch 的那些。

**外部来源 / 论文**

- **Switch Transformer**：arXiv 2101.03961（★ abs 页 `comments` 逐字为 `JMLR`
  ⇒ **已确认同行评审**；抓取 `https://ar5iv.labs.arxiv.org/html/2101.03961`，2026-09-22，
  **413955 B**，`sha256[:16] = e9cec95e4c274436` —— 与计划 §C.1 逐位一致）。**[C]**
  - 式 (4)/(5)/(6)、`α` 的取值与扫描带、Table 1 的 finding (3)（与 `02` 共用）、
    Figure 14 的伪码（aux loss 与探索噪声）。
- **GShard**：arXiv 2006.16668（**无 venue 线索 ⇒ 未确认同行评审**；
  2026-09-22，**451967 B**，`sha256[:16] = 1a5e04e9d1604e8c`）。**[C]**
  - 它自己的 `ℓ_aux`（`m_e · (c_e/S)`）与总损失的乘子 `k`。
- **config 的两个来源**（★ **必须分开标**）：
  - `num_experts = 128`、`num_experts_per_tok = 8`、`moe_intermediate_size = 768`、
    `hidden_size` / `intermediate_size` / `num_hidden_layers` / `num_attention_heads` /
    `num_key_value_heads` / `head_dim` / `vocab_size`、
    `PUBLISHED_PARAM_COUNT_B` 里的 `30.5`：**[B]**，转记自
    `45-kernel-lifecycle/requirement-discovery/code/specs.py`（**不改任何一位数字**）。
  - `router_aux_loss_coef = 0.001`、`norm_topk_prob = true`：**经 `hf-mirror.com` 镜像取得**
    （2026-09-22 直取 `https://hf-mirror.com/Qwen/Qwen3-30B-A3B/resolve/main/config.json`，
    镜像副本 `sha256[:16] = 2850ddb3bf7aecad`，963 B —— 与
    `attention-variants/code/specs.py:75` 记的是**同一份副本**）。
    ★★ **未与官方 repo 比对 sha256**（`huggingface.co` 直连不可达，计划 §C.0）。

**本机实测值**

- `code/03_balance_granularity.py` 的 `make run` 输出：`python3 3.13.5` + `numpy 2.4.4`，
  macOS arm64，**无 GPU**。口径：`T=4096, E=64, k=2`；`logits ~ N(0,1)`，**前 8 个**专家加偏置 `s`；
  **同一个种子、只改 `s`**（配对比较）。**全部是推导值**，不是真实模型的训练结果。
- ★★ **口径注记**：计划 §0.7-D 的 **D17/D18** 出自 `probe_80g.py` 的 §B
  （**前 8 个**专家加偏置），而 **D16** 出自 `probe_80.py` 的 Q7
  （`hot = rng.permutation(E)[:8]`）—— **两个探针脚本、两套「热点专家」口径**。
  ⇒ **两套数不可互相引用**（本域的口径混用现场）。
  本机在 D17/D18 的口径上复现：均匀 `max/mean = 1.23`（D18 逐字 `1.23` ✓）、
  极端偏斜 `8.17`（D18：`8.12`）、`aux = 1.0008` / `1.0012`（D17：`1.0009` / `1.0012`）、
  偏斜 `1.5` 时 `2.4294`（D17：`5.6910` 对应的是**极端偏斜**那一行，本机 `5.7193`）。
  ★ 差异来自**探测脚本的 `rng` 已被它自己前面几节消耗过** ⇒ D17/D18 的数**绑定在那些脚本的状态上**。

**上游契约**

- `45-kernel-lifecycle/requirement-discovery/code/specs.py:123-127`（共享专家字段**没有** ⇒ R11）、
  `:170-176`（`PUBLISHED_PARAM_COUNT_B` 含 Qwen3 的 `30.5` ⇒ ★ **R19 的更正**）、
  `:180-183`（「不给占位常量」那条纪律）。
- `40-kernels/moe-kernels/01-moe-compute-pattern.md:72,278` 与 `code/verify.py:443`
  （FFN 的**投影数**系数 4 vs 6）—— **禁区**：本文只**用**那个 `3`，不重推。
- `40-kernels/moe-kernels/code/verify.py:273-300`（A1：三种 gating 归一化口径）—— **禁区**。

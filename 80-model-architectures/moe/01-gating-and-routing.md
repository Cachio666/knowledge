---
title: gating 与 routing：谁选谁，以及两种「随机」
slug: gating-and-routing
domain: 80-model-architectures
topic: moe
level: 2
difficulty: intermediate
prerequisites:
  - 80-model-architectures/transformer/02-ffn-shapes
tags: [moe, gating, top-k, routing, random-routing, expert-choice, token-choice, load-balance]
related:
  - 40-kernels/moe-kernels/01-moe-compute-pattern
  - 40-kernels/moe-kernels/03-permute-and-unpermute
  - 80-model-architectures/moe/02-capacity-factor
  - 80-model-architectures/moe/03-balance-and-granularity
code:
  - code/01_gating_routing.py
status: stable
updated: 2026-09-22
---

# gating 与 routing：谁选谁，以及两种「随机」

> `gate = softmax(X·W_g)` 只是一个小 GEMM；**路由的全部内容是「谁选谁」**。
> 常规 MoE 让 **token 选专家**（`top_k` 个），于是**负载不均由各 token 独立决定累加而来**，
> 只能靠辅助损失**事后**拉平；Expert Choice 反过来让**专家选 token**，
> 于是**负载均衡是构造出来的**（本机实测：`max/mean` 恰好 `1.00` vs token choice 的 `1.33`）。
> 而「随机」在这个主题里有**两个同名不同指的机制**：GShard 的 `Random routing`
> 是**为省容量**（按 `g₂` 的概率决定要不要派第 2 个专家），Switch 的探索噪声是**为探索**
> （训练期给 router logits 加噪声）—— **两篇各说各的，不要合并**。

## 0. 背景与动机

MoE 一层的四个 kernel（routing → permute → grouped GEMM → unpermute）里，
**routing 是唯一「N 很小」的那个**：它的 GEMM 是 `[T, d] @ [d, E]`，`N = E`。
本主题不讲它的代价（FLOPs / 算术强度 / 拐点批大小**不在本域**，
正文在 `40-kernels/moe-kernels/01-moe-compute-pattern.md`，此处回链）；
本文讲的是**它决定了什么**：

| routing 决定的事 | 后果住在哪 |
|---|---|
| 每个 token 拿几个专家（`top_k`） | 计算量（回链 `40-kernels/moe-kernels/01`） |
| 哪个 token 去哪几个专家 | 负载形状 → **容量**（`02-capacity-factor.md`） |
| 权重怎么归一化 | 输出的尺度（本文 §1.2） |
| 负载不均怎么治 | **辅助损失**（`03-balance-and-granularity.md`） |

★ **本域的边界**：routing 的**两步形态**（gate GEMM + top-k + softmax）、
三种归一化口径（甲/乙/丙）的**实现与比较**、以及 permute / slot mask，
都是 `40-kernels/moe-kernels` 的正文（计划 §2 的边界表）。
本文**只**写它作为**架构选择**的那一层。

## 1. 原理

### 1.1 gating 是什么（一句话）

`gate` 就是一个线性层加 softmax：

```
logits = X[T, d] @ Wg[d, E]
p      = softmax(logits, axis=-1)        # 每个 token 一张 E 维的分布
idx    = top-k(p)                        # 每个 token 拿 k 个专家
```

`Wg` 是**学出来的**（逐字出处见 §1.4 的 Switch 伪码：`router_weights = mtf.Variable(...)`）。
★ 这句话的全部重量在 `Wg` 上：**路由不是规则，是一个被训练的决策**，
所以它才需要「探索」（§1.4）与「均衡」（`03-balance-and-granularity.md`）这两笔额外的账。

### 1.2 权重口径：在 `k` 个 logit 上 softmax

有三种写法在文献与实现里都出现过。**它们的实现与比较不在本域** ——
正文与实现都在 `40-kernels/moe-kernels/code/verify.py` 的 **A1（`:273-300`）**（正文在彼，此处回链）。
本域只说**本主题采用的**是**甲**：

> **先取 top-k，再在这 `k` 个 logit 上做 softmax** ⇒ 每行的权重之和**恒为 1**。

本机实测（`T=512, E=8, k=2`，`python3 3.13.5` + `numpy 2.4.4`）：
权重行和与 1 的最大偏差 **`2.220e-16`**（`01_gating_routing.py` 的 `A.1`）。
★ 这一条的意义**不在数字**（它本来就该是 1），而在**证伪方向**：
把「在 k 个上重归一」这一步去掉（改成在全 `E` 上 softmax 再取 top-k 不重归一），
行和就会**小于 1** —— 那个缺口正是 `40-kernels` 的 A1 用「丙」打出来的。

★★ **本域为什么只实现「甲」**：一旦本文的配套程序能算出「乙」「丙」，
它就变成了 `40-kernels/moe-kernels/code/verify.py` 里那份比较的**会漂移的副本**。
⇒ 配套程序里**没有**切换口径的开关（这是刻意的，不是遗漏）。

### 1.3 两个模型的规格（`top_k` 不是常数）

| 模型 | 路由 | `E` | `top_k` | 活跃占比 |
|---|---|---|---|---|
| Switch Transformer | **top-1** | — | 1 | — |
| GShard | top-2 | — | 2 | — |
| **Qwen3-30B-A3B** | top-8 | **128** | **8** | **1/16 = 6.25%** |

- GShard 逐字：「each token is dispatched to **at most two experts**」；
- Switch 是 **top-1**：它自己的 aux loss 里 `f_i` 的定义式逐字用 `argmax`
  （`f_i = (1/T) Σ_x 1{argmax p(x) = i}`）；
- Qwen3-30B-A3B 的 `num_experts = 128`、`num_experts_per_tok = 8`
  （**[B]**，转记自 `45-kernel-lifecycle/requirement-discovery/code/specs.py`）。

**_ ⇒ 引用「MoE 的 top-k」时必须写出是哪一个模型。** 本机实测：在真实维数上
（`T=64, E=128, k=8`）每个 token 的权重向量里**非零项恰为 8 个**（`B.1`）。

### 1.4 两种「随机」——**同名不同指**

#### 1.4.1 GShard 的 `Random routing`：**为省容量**

逐字（GShard §2.2，arXiv 2006.16668，**[C] + 未确认同行评审**）：

> "Intuitively, because `y_s` is a weighted average of what selected experts return, if the
> weight for the 2nd expert is very small, we can simply ignore the 2nd expert **to conserve
> the overall expert capacity**. Hence, in addition to respecting the expert capacity
> constraint, `GATE(·)` **dispatches to the 2nd-best expert with the probability proportional
> to its weight `g₂`**."

★ 这是一条**容量**机制：第 2 个选择**按概率**才真的派出去，省下来的槽位留给别人。

**闭式**：期望派出的条数 `= Σ_t g₂(t)`。本机实测（`T=512`，`g₂` = 甲口径下第 2 个选择的权重）：
`Σ g₂ = 73.006`，20 次重复的均值 `73.350`，**两个端点精确** ——
`g₂ ≡ 1` ⇒ 派出 `512 = T` 个；`g₂ ≡ 0` ⇒ 派出 `0` 个（`C.1`）。
⇒ 真实分布下被忽略掉的第 2 选择占 **85.7%**（`C.2`）。

⚠ **口径必须写出来**：论文只写了「proportional to its weight `g₂`」，
**没有规定 `g₂` 用哪种归一化**。本域把 `g₂` 定义为**甲口径下第 2 个选择的权重**
（`w[:, 1]`），并把这个约定写在程序里 —— 而不是假装它没有歧义。

#### 1.4.2 Switch 的探索噪声：**为探索**（与上一条无关）

逐字（Switch Appendix C，`router_exploration`，**[C] + 已确认同行评审（`comments = JMLR`）**）：

> "To introduce exploration, we consider several approaches: 1) deterministic or argmax
> 2) sampling from the softmax distribution 3) input dropout on the incoming representation
> 4) **multiplicative jitter noise on the incoming representation**. …
> **Throughout this work, we use input jitter** to inject noise as we have found it to
> empirically perform the best."

Table 11（`Router Exploration Strategies`，**负对数困惑度**）：

| 策略 | 值 |
|---|---|
| `Argmax` | `-1.471` |
| `Sample softmax` | `-1.570` |
| `Input dropout` | `-1.480` |
| `Input jitter` | `-1.468` |

★★ **本域发现的两处论文内部不一致**（都不替论文选，只把冲突钉住）：

1. **同一张表的表题与内容方向冲突**。表题逐字写着 `(lower is better)`，
   但列头逐字是 `Model Quality (Neg. Log Perp.) ( ↑ )`（**向上箭头**），
   而正文说 `input jitter` `perform the best` —— `input jitter` 是四值里**最大**的那个。
   ⇒ 按表题读第一名是 `argmax`，按列头/正文读第一名是 `input jitter`。
   `C.4` 把「两种读法给出不同的第一名」做成断言。
   ★ **两个方向下都成立的结论只有一条**：极差 **0.102** ⇒ 表题那句
   `no material speed performance difference` 成立 —— **「jitter 最好」这个转述不成立**
   （它依赖你先选了哪个方向）。
2. **同一节里正文与伪码对不上**。正文说作用对象是 `the incoming representation`、
   方式是 `multiplicative`；而 Figure 14 的伪码逐字是

   ```
   if is_training : # Add noise for exploration across experts.
     router_logits += mtf.random_uniform(shape=router_logits.shape, minval=1-eps, maxval=1+eps)
   ```

   —— 作用对象是 **`router_logits`**、方式是**加性**。配套程序实现**伪码**那一版
   （它是可执行的那一份），并在输出里标明这一点。
   两个端点精确：`eps = 0` ⇒ 选择**逐位不变**（`0.0%`）；`eps = 1.0` ⇒ **22.1%** 的 token
   改变选择（`C.3`）。

★★ **不要把这两条机制合并。** 它们共享「随机」这个词，但**目的相反**：
一条是**省容量**（GShard，只在推理/训练时按权重概率丢弃第 2 个选择），
一条是**主动制造变化**（Switch，训练期给 router 加噪以逃离 argmax 的利用策略）。
原始字节上 `Random routing` 在 **GShard 里 1 次、Switch 里 0 次**（对照见 §5 的表 B）。

### 1.5 token choice vs expert choice：**谁选谁**

逐字（Expert Choice，arXiv 2202.09368，**[C] + 未确认同行评审**）：

> "Conventional MoE models employ **token-choice routing** which **independently selects the
> top-k experts for each token**."
>
> "Different from conventional routing, an **expert choice** method **independently selects
> top-k tokens for each expert**, where `k` is a fixed expert capacity (i.e. the number of
> tokens each expert can take). Despite its simplicity, expert choice achieves **perfect load
> balancing by design**."

这一条换个方向，**三件事同时变了**：

| | token choice | expert choice |
|---|---|---|
| 谁做选择 | 每个 token 独立选 | 每个专家独立选 |
| 负载 | 由**独立决定累加**而来 ⇒ 不均 | **构造上相等**（`max/mean = 1.00`） |
| 每个 token 拿到几个专家 | 恰好 `k` 个 | **可变**（`k` 成了容量，不是 token 的配额） |
| 要不要 aux loss | **要**（`03-balance-and-granularity.md`） | 原文逐字：`without imposing an auxiliary loss` |

本机实测（`T=512, E=8, k=2`，同一份 logits）：
token choice 的每专家条数 `[107, 110, 120, 140, 153, 100, 124, 170]`（`max/mean = 1.33`）；
expert choice 的 `[128] × 8`（`max/mean = 1.00`，**构造上恒等**）—— `C.5`。

★★ **代价也要写出来**（原文 §6 `Limitations` 逐字）：

> "The expert choice method might not immediately apply to **auto-regressive text generation**
> as our current implementation takes in the **past and future tokens** to perform the top-k
> selection."

⇒ expert choice 需要**整批**（含未来 token）才能选，这与自回归解码的方向相反。
★ 这决定了它出现在**训练**侧而不是**推理**侧 —— 与 `01` 篇里那条「省 FLOPs 不省显存」是同一类
「收益与代价住在不同地方」的账。

## 2. 执行映射

本域**不讲** kernel 侧的映射（routing 的两步形态、warp 级 top-k、`__shfl_xor_sync` 蝶形归约）
—— 正文在 `40-kernels/moe-kernels/01-moe-compute-pattern.md:161-176` 与
`code/01_moe_routing.cu`（正文在彼，此处回链）。

本文只记**架构侧的一条**：`gate` 的输出形状是 `[T, E]` 的**概率**与 `[T, k]` 的**下标 + 权重**，
而这两样东西**下游全都要**：

- `[T, k]` 的下标 → routing 之后一切搬运动作的**唯一输入**（permute 的 `dst_row`）；
- `[T, k]` 的权重 → unpermute 时的**加权系数**；
- `[T, E]` 的概率 → **只进辅助损失**（`P_i` 就是它的列均值，见 `03`）。

★ 最后这一条容易被漏掉：**`p` 不进前向**（前向只用 top-k 那几个），
但它**必须算出来**，因为 aux loss 要用它（那是 `03` 的账）。

## 3. 实现与演进

### 3.1 三种「探索」的商业现状

| 机制 | 目的 | 出处 |
|---|---|---|
| `Random routing`（按 `g₂` 概率派第 2 专家） | **省容量** | GShard §2.2 |
| 探索噪声（加在 router logits / 输入表示上） | **逃离 argmax 的利用策略** | Switch Appendix C |
| 辅助损失（对 `f` 与 `P` 做点积） | **均衡负载** | Switch §2.2 / GShard §2.2 / `03` |

★ 三者**都在改路由**，但**改的是不同的量**：一个改**派不派**、一个改**logits 的取值**、
一个改**损失函数**。把它们混起来讲，最典型的后果是把「省容量」说成「探索」。

### 3.2 真实模型用哪些？

| 事实 | 来源 | 等级 |
|---|---|---|
| Qwen3-30B-A3B：`num_experts = 128`、`num_experts_per_tok = 8` | `45-…/code/specs.py` 的 `QWEN3_30B_A3B_CONFIG`（转记） | **[B]** |
| Qwen3-30B-A3B 的 config 里有 `norm_topk_prob: true` 字段 | 经 `hf-mirror.com` 镜像取得，**未与官方 repo 比对 sha256** | 镜像副本 |

★★ **`norm_topk_prob` 这一条只记「字段存在且为 true」**：
它的**语义**在本环境**没有一手来源**（`transformers` 的源码不在本环境）
⇒ 本域**不推断它「做了什么」**，更不据此写机制从句。
它记在 `code/specs.py` 的 `QWEN3_30B_A3B_ROUTER_FROM_MIRROR` 里，
并被 `01_gating_routing.py` 的 `D.4` 逐位钉住。

## 4. 踩坑与失败模式

### 4.1 把「随机」当成一个概念

- **现象**：写「MoE 的路由是随机的」/「MoE 用 random routing 来均衡负载」。
- **成因**：文献里有**三条**互不相同、目的互不相同的机制（§3.1），它们**都**沾「随机」这个词。
  GShard 的 `Random routing` **不是**为了均衡（它是为了**省容量**，均衡由它自己的 `ℓ_aux` 负责）。
- **规避**：引用时**必须带出处 + 目的**。原始字节的对照（§5 表 B）：
  `Random routing` 在 GShard **1 次**、在 Switch **0 次**；
  Switch 里那 3 次 `random` 全部属于**另外**的语境（随机种子 / 探索策略 / 伪码里的 `random_uniform`）。

### 4.2 引用「jitter 最好」

- **现象**：写「Switch 发现 input jitter 是最优的探索策略」。
- **成因**：Table 11 **自身的方向就不自洽**（表题 `lower is better` vs 列头 `↑` + 正文
  `perform the best`）⇒ 换一边读，第一名就变成 `argmax`。
- **规避**：只写**两个方向下都成立**的那一句（极差 `0.102` ⇒「没有实质差别」），
  并把冲突本身如实记录。配套程序 `C.4` 用「两种读法的第一名不同」把冲突钉住。

### 4.3 用 `hidden_size ÷ num_attention_heads` 去推某个 head 的宽度

- **现象**：把 `gate` 的 `N = E` 与注意力里的 head 数混起来算。
- **成因**：**同名不同指**的另一个现场 —— `head_dim` 在 Qwen3 的 config 里是原生字段
  （`128`），而 `hidden_size / num_attention_heads = 2048/32 = 64`。
- **规避**：那个陷阱的正文在 `45-…/requirement-discovery/01-model-to-operator-inventory.md:270-276`
  （正文在彼，此处回链）。本主题的 `E` 是**专家数**，与 head 数**无关**。

### 4.4 把 `top_k` 当常数

- **现象**：写「MoE 每个 token 过 2 个专家」。
- **成因**：`top_k` 是**模型规格**（§1.3）：Switch 1、GShard 2、Qwen3-30B-A3B 8。
- **规避**：`B.3` 把这一条做成断言 —— 三个模型三个值，而本域的两个程序在**真实维数**上跑一遍。

## 5. 判据表

| 可机械判定的条件 | 结论 |
|---|---|
| 甲口径（先 top-k 再 softmax）的权重行和 | **恒为 1**（本机：与 1 的最大偏差 `2.220e-16`）；去掉重归一 ⇒ **< 1**（那个缺口的正文在 `40-kernels/moe-kernels/code/verify.py` 的 A1） |
| 想要乙 / 丙两种口径的实现或比较 | **不在本域**：正文与实现都在 `40-kernels/moe-kernels/code/verify.py` 的 A1（`:273-300`） |
| 两条**独立**代码路径（向量化 `take_along_axis` vs 逐行 Python 循环）的逐元素差 | **机器精度档**（本机 `2.220e-16 < 1e-14`）；若报出 `0.0` ⇒ 先查两边是不是同一份代码 |
| 参照实现的调用计数 | 必须**恰为 0**（本机：计数表整个为空） |
| `top_k` | 是**模型规格**，不是常数（Switch 1 / GShard 2 / Qwen3-30B-A3B 8）；引用时必须写出是哪一个 |
| 并列 logit 的 tie-break | **取下标小的**（`argsort(-x, kind="stable")` 的语义）；且「top-k」的定义成立：**被选中的最小 logit ≥ 被排除的最大 logit** |
| 引用 GShard 的 `Random routing` | 它是**为省容量**（逐字 `to conserve the overall expert capacity`）；期望派出条数 `= Σ g₂`，端点 `g₂≡1 ⇒ T`、`g₂≡0 ⇒ 0` |
| 引用 Switch 的探索策略 | Table 11 的**方向不自洽**（表题 `lower is better` vs 列头 `↑` + 正文 `perform the best`）⇒ 只能写「极差 `0.102`，没有实质差别」 |
| 引用「Switch 的探索噪声」的实现 | 正文说 `multiplicative … on the incoming representation`，伪码做的是 `router_logits += uniform(1-eps, 1+eps)` —— **两处不一致，引用时须说明实现的是哪一版** |
| token choice vs expert choice | 判据是**谁选谁**：token choice 的 `max/mean = 1.33`（独立决定累加）；expert choice 的 `max/mean = 1.00`（**构造上**相等），但它**不能**直接用于自回归生成（原文 §6 `Limitations`） |
| 想知道 routing 的 FLOPs / 算术强度 / kernel 形态 | **不在本文**：正文在 `40-kernels/moe-kernels/01-moe-compute-pattern.md`（含 `:72,161-176,270`） |

> ★ **表 A 的数值判据**（计划 §3 Chunk 3）：① 门权重行和 `|Σ − 1| ≤ 1e-12`。
> 它在 `code/01_gating_routing.py` 的 `A.1` 里，阈值本身被 `D.2` 逐位钉住
> （放宽阈值 ⇒ 判据退化成恒真）。

## 6. 小结与延伸

**一句话**：gating 是一个小 GEMM，但**它决定了整层的四个 kernel 的形状与负载**；
而「谁选谁」这件事有两种相反的方向 ——
**token 选专家**（`k` 是 token 的配额、负载要不均、要靠 aux loss 治）
与**专家选 token**（`k` 是专家的容量、负载构造上相等、不需要 aux loss，但需要整批）。

- **前置**：`transformer/02-ffn-shapes`（MoE 换的就是 FFN 那一块）。
- **同主题的下一篇**：`02-capacity-factor.md` —— token choice 一旦落地，
  就会问「每个专家开到少个槽位」，那是**容量因子**的正文（也是上游 8 处指名让渡的落点）。
- **另一篇**：`03-balance-and-granularity.md` —— aux loss（**含 `×N` 与 `α`**）与负载的度量。
- **kernel 侧**：routing 的两步形态、warp 级 top-k、permute、grouped GEMM、slot mask
  的正文在 `40-kernels/moe-kernels/01,02,03`（正文在彼，此处回链）。

## 编译与运行

```bash
cd code
make run                          # 三个编号程序依次跑；本程序是 01_gating_routing.py
python3 01_gating_routing.py      # 只跑本程序
```

- 配套程序：**`code/01_gating_routing.py`**（编号与外键严格对应）。
- 依赖：`numpy`（`requirements.txt` 里**只有**它）。**本程序不需要网络、不需要 GPU。**
- 失败标记：`## FAIL: moe`（本主题自定义，**不跨主题照抄**）。
- 程序的输出分五段：`D`（口径/阈值/specs 钉住，**刻意放在最前**）、
  `A` 权重口径与两条独立路径、`B` 真实模型的规格锚点、
  `C` 两条「随机」机制 + token choice vs expert choice、`D（续）` 标记自检与独立性检查。

## 参考

### 表 B：引用判据（**不能被程序判定**，在**原始字节**上计数）

实跑（2026-09-22；`WebFetch` 被策略拦截 ⇒ 用 `curl`）：

```bash
curl -sSL -o /tmp/arx_2101.03961.html https://ar5iv.labs.arxiv.org/html/2101.03961  # 413955 B
curl -sSL -o /tmp/arx_2006.16668.html https://ar5iv.labs.arxiv.org/html/2006.16668  # 451967 B
curl -sSL -o /tmp/arx_2202.09368.html https://ar5iv.labs.arxiv.org/html/2202.09368  # 175642 B
```

| 字面量 | Switch | GShard | EC | 用途 |
|---|---|---|---|---|
| `Random routing` | **0** | **1** | 0 | ★ GShard 的随机路由**只此一篇有** |
| `random`（剥标签后的整词） | 3 | 0 | — | ★ 对照：Switch 里那 3 次全部是别的语境（对照检验） |
| `routing` | 28 | 5 | — | 对照：这个词本身两篇都数得到 |
| `capacity factor` | **11** | **0** | — | ★★ **这个词是 Switch 的**（计划 §C.1 的 M1） |
| `fractional capacity of each expert` | 0 | 1 | — | GShard 用的是「fractional capacity」 |
| `at most two experts` | 0 | 2 | — | GShard 的 `top_k = 2` |
| `capacity factor` | **11** | **0** | **14** | ★★ EC 也用这个词（**同名不同指**，见 `02`） |
| `the capacity factor` | **1**（剥标签后 **2**） | 0 | **8** | 同上；⚠ Switch 那一行的 1/2 差正是**跨标记**造成的 |
| `expert choice` | — | — | **21** | EC 的主题词 |
| `multiplicative jitter noise on the incoming representation` | **1** | 0 | 0 | ★ 正文那一条（与伪码冲突） |
| `Add noise for exploration across experts`（剥标签后） | **1**（原始字节 **0**） | 0 | 0 | ★★ **跨标记引文在原始字节上必然计不到**（§0.3 的纪律） |
| `loss = mtf . reduce_mean ( density_1_proxy * density_1 ) * ( num_experts ^ 2)`（剥标签后） | **1**（原始字节 **0**） | — | — | ★★ Figure 14 的 aux loss 实现（`03` 用它） |

**外部来源 / 论文**

- **Switch Transformer**：arXiv 2101.03961（★ abs 页 `comments` 逐字为 `JMLR`
  ⇒ **已确认同行评审**；抓取 `https://ar5iv.labs.arxiv.org/html/2101.03961`，2026-09-22，
  **413955 B**，`sha256[:16] = e9cec95e4c274436` —— 与计划 §C.1 逐位一致）。**[C]**
- **GShard**：arXiv 2006.16668（**无 venue 线索 ⇒ 未确认同行评审**；
  2026-09-22，**451967 B**，`sha256[:16] = 1a5e04e9d1604e8c` —— 与计划 §C.1 逐位一致）。**[C]**
- ★★ **Expert Choice**：arXiv 2202.09368（abs 页**无 `Comments:` 也无 `Journal reference:`**
  ⇒ **未确认同行评审**；2026-09-22，**175642 B**，`sha256[:16] = 059b2abad496bcb1`）。**[C]**
  - ⚠★ **这一篇不在计划 §C 的论文清单里**（计划 §C.7 逐字称「仍待补：**无**」），
    而计划 §3 Chunk 3 把「token choice vs expert choice」写进了本文的核心内容
    ⇒ 本域**补取**了这一篇。**这是一条已回报控制者的偏差。**

**本机实测值**

- `code/01_gating_routing.py` 的 `make run` 输出：`python3 3.13.5` + `numpy 2.4.4`，
  macOS arm64，**无 GPU**。口径：`T=512, E=8, k=2, d=32`、种子 `20260922`、`logits` 由
  `X @ Wg` 生成（`X, Wg ~ N(0,1)`）。**全部是推导值**，不是真实模型的测量。

**上游契约**

- `40-kernels/moe-kernels/01-moe-compute-pattern.md:161-176`（routing 的两步形态与 warp 级 top-k）
  与 `:270`（「槽位数 C 该取多少」是**架构层**的问题）—— 后者是**指名让渡**，
  本文不接（它归 `02`），此处只回链。
- `40-kernels/moe-kernels/code/verify.py:273-300`（A1：三种 gating 归一化口径）—— **禁区**。
- `45-kernel-lifecycle/requirement-discovery/code/specs.py`（`num_experts` /
  `num_experts_per_tok` 的 [B] 来源；`router_aux_loss_coef` **不在**那份文件里，见 `03`）。

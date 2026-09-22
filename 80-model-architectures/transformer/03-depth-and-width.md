---
title: 深度与宽度怎么换，以及架构规格表
slug: depth-and-width
domain: 80-model-architectures
topic: transformer
level: 2
difficulty: intermediate
prerequisites:
  - 80-model-architectures/transformer/01-block-and-residual-stream
tags: [depth, width, parameter-budget, embedding, head-dim, spec-table, tie-word-embeddings]
related:
  - 45-kernel-lifecycle/requirement-discovery/01-model-to-operator-inventory
  - 130-quantization/kv-cache-quantization/01-kv-cache-basics
  - 80-model-architectures/transformer/02-ffn-shapes
code:
  - code/03_depth_width.py
status: stable
updated: 2026-09-22
---

# 深度与宽度怎么换，以及架构规格表

> **参数量在四个部件间的分布是可以从 config 算出来的**：Llama-3-8B 上 FFN 一项占 `70.2%`。
> 于是「加深」这条路的代价也就定下来了：attention 与 FFN 的参数 `∝ L·d²`，
> 而**用宽度去补深度**（`d ∝ L^(−1/2)`）的代价落在**嵌入**上。
> ★ 并且：同一个字段名在两个真实模型上的**来源**可能不同（`head_dim` 就是）。

## 0. 背景与动机

给定一个参数预算，「深一点」还是「宽一点」？这不是审美问题，它落在两组数的关系上：

| 部件 | 参数量 | 它随什么走 |
|---|---|---|
| 嵌入 `embed_tokens` | `V · d` | **线性**于宽度 |
| 输出头 `lm_head` | `V · d`（**不共享**时另算一次，见 §4 第 2 条） | 同上 |
| 每层 attention | `d·(h_q·d_h) + 2·d·(h_kv·d_h) + (h_q·d_h)·d` | `∝ d²`（头数随宽度缩时） |
| 每层 FFN | `3 · d · d_ff` | `∝ d²`（`d_ff ∝ d` 时） |

一眼能看出的结构后果是：**attention 与 FFN 是 `L · d²` 的量，嵌入是 `V · d` 的量**
⇒ 这两类项对「深/宽」的敏感度不同，所以「换法」有一个可算的答案（§1.3）。

**§2 的规格表是本文的另一半**，也是本域相对 `45-kernel-lifecycle` 的**新增部分**：
同一张 config 里，有的字段是**原生**的（真实 `config.json` 里真有这一行），
有的字段是**推导**出来的（config 里没有，要算）。`head_dim` 就是后者——
★ 而**在另一个模型上它又是原生的**，所以「哪些字段是推导的」这句话**不能脱离具体模型说**
（这条是上游一处内部矛盾的现场，见 §2 与 §4 第 1 条）。

## 1. 原理

### 1.1 反算：四个部件的闭式

```
embed      = V · d
lm_head    = V · d                      （tie = False 时；tie = True 则 0）
attention  = L · ( d·(h_q·d_h) + 2·d·(h_kv·d_h) + (h_q·d_h)·d )
ffn        = L · 3 · d · d_ff           （三个矩阵：gate / up / down，见 02-ffn-shapes）
```

★ 两处**口径**必须写在数字旁边：

1. **`h_q · d_h` 用的是 `head_dim` 字段，不是 `d / h_q`**（§4 第 1 条：两者在 Qwen3 上不相等）；
2. **`lm_head` 是否另算一次由 `tie_word_embeddings` 决定**（§4 第 2 条）。

**本机推导值**（Llama-3-8B，`tie = False`）：

```
embed      525,336,576
attention  1,342,177,280
ffn        5,637,144,576
lm_head      525,336,576
──────────────────────────
合计       8,029,995,008   = 8.029995 B
```

对照公开参数量 **8.03 B** ⇒ 相对差 **6.22e-07**（判据阈值 **±1%**）。
★ **这条判据比它看起来弱得多**：公开值只给到 3 位有效数字，所以「±1%」这条带子
**分辨不出** `8.029995` 与 `7.95`。真正的强对照是**逐位**的那一条：
`45-kernel-lifecycle/requirement-discovery/code/01_operator_inventory.py:549`
的逐位断言 `8029995008` —— 本程序算出的正是这个整数（`B.2`）。
⚠ 本域**只把它当一次对照，不重做整套算子清单**（那是 `45` 的正文，见域 README 的边界表）。

### 1.2 参数的分布（这笔账决定了后面所有取舍）

| 部件 | 参数量 | 占比 |
|---|---|---|
| 嵌入 | 525,336,576 | 6.54% |
| attention | 1,342,177,280 | **16.71%** |
| FFN | 5,637,144,576 | **70.20%** |
| `lm_head` | 525,336,576 | 6.54% |
| 合计 | 8,029,995,008 | 100% |

⇒ **「参数主要在 FFN 里」这句话是可以算出来的**（不是转述）：一层里 FFN 是
`3·d·d_ff = 176,160,768` 个权重，attention 是 `41,943,040` 个 —— 差 `4.2` 倍。
这也解释了为什么「省参数」的努力几乎都落在 FFN 上（MoE 就是那条路，见
`80-model-architectures/moe/01-gating-and-routing`）。

### 1.3 深度与宽度怎么换：`d ∝ L^(−1/2)`

attention 与 FFN 的参数量都是 `L · d²` 的量级 ⇒ 在**固定这两项预算**的前提下，
层数与宽度的替换关系是

```
L · d² = 常数     ⇒     d ∝ L^(−1/2)                    …… (1)
```

**本机实测**（口径：`head_dim = 128` 与 **GQA 的比值 4:1 保持不变**，头数随宽度缩放；
`d_ff` 也跟着 `d` 缩；`d` 取 `4096 / √mult`，`mult` 取 4 的幂使其落在 128 的网格上）：

| `L`（`mult`） | `d` | attention + FFN | 嵌入（`V·d`） |
|---|---|---|---|
| 32（1×） | 4096 | 6,979,321,856 | 525,336,576 |
| 128（4×） | 2048 | **6,979,321,856**（逐位相同） | 262,668,288（1/2） |
| 512（16×） | 1024 | **6,979,321,856**（逐位相同） | 131,334,144（1/4） |
| 2048（64×） | 512 | **6,979,321,856**（逐位相同） | 65,667,072（1/8） |

⇒ 两点结论：

1. **「加深 = 变窄」在 `attn + ffn` 上可以做到逐位不变**（不是近似）——这就是式 (1) 的实现形态；
2. **代价落在嵌入上**：`V·d` 按 `1/√mult` 缩（`1, 1/2, 1/4, 1/8`）。
   换句话说，**深化的账最后由词表那两项付**（`embed` 与 `lm_head` 各一次）。

★ **但式 (1) 是一条连续近似**：真实的宽度必须落在两个离散网格上——
`d` 是 `head_dim` 的整数倍，且 `num_key_value_heads = h_q / GQA` 必须是整数。
**本机实测**（`L = 64`，理想 `d = 4096/√2 = 2896.31`）：

| 头数 | `d = h × 128` | `attn + ffn` / 基准 | GQA 4:1？ |
|---|---|---|---|
| 21 | 2688 | 0.8598× | 否 |
| 22 | 2816 | 0.9420× | 否 |
| 23 | 2944 | 1.0280× | 否 |
| 24 | 3072 | 1.1250× | **是** |

⇒ 理想宽度**不在**网格上；最近的可实现宽度给出的偏离是 `0.8598 ~ 1.1250` 倍，
而且其中三个连「头数是 4 的倍数」都不满足。**「等参数加深」在纸面上精确，落到网格上必然离散。**

### 1.4 `head_dim` 与「宽度」的关系

残差流的宽度是 `hidden_size`，而 attention 那边**写回**残差流的宽度由 `o_proj` 的输入决定
（`h_q · d_h`）。两者**不必相等**：

| 模型 | `hidden_size` | `num_attention_heads × head_dim` | 相等？ |
|---|---|---|---|
| Llama-3-8B | 4096 | 32 × 128 = 4096 | **是** |
| Qwen3-30B-A3B | 2048 | 32 × 128 = **4096** | **否**（`o_proj` 把 4096 映回 2048） |

★ 这个例子的**正文已在** `45-kernel-lifecycle/requirement-discovery/01-model-to-operator-inventory.md:270-276`
（§4.5，判据在 `:288`），本域**只做本文 §2 的「逐字段标注」这一新增部分，不重推那个例子**。

## 2. 执行映射：架构规格表（逐字段标注「config 原生」/「推导」）

先看这些字段**落在哪些模块上**（HuggingFace 的命名，本域不复核命名）：

| 字段 | 落到哪 | 形状 |
|---|---|---|
| `vocab_size` × `hidden_size` | `model.embed_tokens` / `lm_head` | `(V, d)` 各一次（`tie = False` 时） |
| `hidden_size` | **残差流本身**的宽度（见 `01-block-and-residual-stream`） | `(B, S, d)` |
| `num_attention_heads` × `head_dim` | `self_attn.q_proj`（`d → h_q·d_h`）/ `o_proj`（`h_q·d_h → d`） | 本机量级：Qwen3 的 q/o 是 `4096` 宽 |
| `num_key_value_heads` × `head_dim` | `self_attn.k_proj` / `v_proj`（`d → h_kv·d_h`） | GQA 的账见 `attention-variants/01` |
| `intermediate_size` | `mlp.{gate,up,down}_proj` | `3·d·d_ff`（见 `02-ffn-shapes`） |
| `num_hidden_layers` | 上面那些模块被**堆几份** | 本机量级：Llama-3-8B 的 attention 一项 `1,342,177,280` |

表的读法：**[upstream] = 逐字来自 `45-kernel-lifecycle/requirement-discovery/code/specs.py` 的 config（[B]）**；
**[mirror] = 不在那份文件里，经 `hf-mirror.com` 镜像取得、未与官方 repo 比对 sha256**；
**[derived] = config 里没有这个字段，是推导值（必须给出推导式）**。
前两类都算「**config 原生**」（真实 `config.json` 里真有这一行）。

### Meta-Llama-3-8B

| 字段 | 值 | 来源组 | 标注 |
|---|---|---|---|
| `hidden_size` | 4096 | [upstream] | config 原生 |
| `num_hidden_layers` | 32 | [upstream] | config 原生 |
| `num_attention_heads` | 32 | [upstream] | config 原生 |
| `num_key_value_heads` | 8 | [upstream] | config 原生（⇒ GQA 4:1） |
| `intermediate_size` | 14336 | [upstream] | config 原生 |
| `vocab_size` | 128256 | [upstream] | config 原生 |
| `tie_word_embeddings` | `false` | [mirror] | config 原生 ★ 见下方限定语 |
| `hidden_act` | `silu` | [mirror] | config 原生 ★ 见下方限定语 |
| **`head_dim`** | **128** | **[derived]** | ★★ **推导**：`hidden_size // num_attention_heads` = `4096 // 32` |

### Qwen3-30B-A3B

| 字段 | 值 | 来源组 | 标注 |
|---|---|---|---|
| `hidden_size` | 2048 | [upstream] | config 原生 |
| `num_hidden_layers` | 48 | [upstream] | config 原生 |
| `num_attention_heads` | 32 | [upstream] | config 原生 |
| `num_key_value_heads` | 4 | [upstream] | config 原生（⇒ GQA 8:1） |
| **`head_dim`** | **128** | **[upstream]** | ★★ **config 原生**（与 Llama 相反） |
| `intermediate_size` | 6144 | [upstream] | config 原生 ★ 语义见下 |
| `vocab_size` | 151936 | [upstream] | config 原生 |
| `tie_word_embeddings` | `false` | [mirror] | config 原生 ★ 见下方限定语 |
| `hidden_act` | `silu` | [mirror] | config 原生 ★ 见下方限定语 |

★★ **两个模型的 `head_dim` 分在不同的组里**，这就是这张表最要紧的一处：

- Llama-3-8B 的**真实 `config.json` 里没有 `head_dim` 字段**（经**两个独立镜像**
  `NousResearch/Meta-Llama-3-8B` 与 `unsloth/llama-3-8b` 逐个字段核对确认）⇒ 它是**推导值**。
  ⚠ 而上游 `45-…/code/specs.py:139` 的 `LLAMA3_8B_CONFIG` 把它写成了**普通字段** —— 这是上游的一处
  **内部矛盾**（该文件 `:146-148` 与 `45-…/01:276` 都建议「只用 config 里的 `head_dim`，
  永远不做那个除法」，照那条建议办事在 Llama-3-8B 上**会找不到该字段**）。
  ★ **本域记录它、不改它**（计划 §4 R13）；本域这一侧的处置就是**把逐字段标注做出来**。
- Qwen3-30B-A3B 的 config **有** `head_dim = 128`，而 `hidden_size // num_attention_heads = 64`
  —— **两者不等**（§4 第 1 条量化了照除法办的代价）。

★ **限定语（逐处都要带，不许省）**：`tie_word_embeddings` 与 `hidden_act` **不在**
`45-kernel-lifecycle/requirement-discovery/code/specs.py` 里（前 7 个这类字段各出现 **0** 次），
本域经 **`hf-mirror.com` 镜像**取得，**未与官方 repo 比对 sha256**。
并且上游 `45-…/code/specs.py:128` 逐字把 `tie_word_embeddings` 标为「**未核实**」
⇒ 本域**不得把它静默升格为「已核实事实」**。

★ **`intermediate_size` 在 Qwen3 上的语义**：它是「**非 MoE 的 FFN 宽度**」，
**不等于专家宽度**（专家宽度在 `moe_intermediate_size = 768`，属 `moe` 主题）。
把它当 dense 的 `d_ff` 用会得到 `3.340 B`，而模型名里的 nominal 规模是 `30 B`（差 9 倍）——见 §4 第 3 条。

### 这张表的机械判据

本程序（`code/03_depth_width.py` 的 `B.5` / `B.5b` / `B.6`）对这张表下三条断言，
它们都可以机械判定：

1. 来源组标签只能是「两类原生 + `derived`」三者之一（越界标签在**断言**里被抓住，不是崩成 traceback）；
2. **标「原生」的字段必须能在来源字典里找到**；
3. **标「推导」的字段必须给出推导式、且推导式能复现出该值**（本域只有 `head_dim` 一个，且只在 Llama 上）；
4. 三组必须构成一个**划分**（两两不交、并集 == config 的字段集）——
   Llama 上是 `6 + 2 + 1 = 9`、Qwen3 上是 `7 + 2 + 0 = 9`；
5. ★ **上表 18 个字段值（9 × 2）连同字段名逐位钉在程序里**（`D.4b`）——
   先例与理由是 `45-kernel-lifecycle/requirement-discovery/code/01_operator_inventory.py:526-541`
   的「**C6：逐个字段钉住**」：只断言「落在 1% 内」会放过任何一个字段的小改动
   （`intermediate_size` 14336 → 14343 只移动 0.03%）⇒ 规格表必须**逐位比**。

## 3. 实现与演进

| 方向 | 它换来了什么 | 它付了什么 |
|---|---|---|
| 更深（`L ↑`，`d ∝ L^(−1/2)`） | §1.3：`attn + ffn` 的参数可以逐位不变 | 嵌入那两项变窄（`V·d` 按 `1/√mult` 缩）；且宽度受 `head_dim` / GQA 的**离散网格**限制 |
| 更宽（`d ↑`） | 每层的表达宽度上去了，且不受网格限制 | 参数 `∝ L·d²`，**与深度同阶地涨**；同时 `embed` / `lm_head` 也线性涨 |
| **加宽 FFN 但不激活它**（MoE） | 参数量 `∝ E`（专家数）而激活量 `∝ top_k` | 路由、负载均衡、容量 —— 那一整套账在 `80-model-architectures/moe`（此处回链） |

★ 这张表里**没有「哪个更好」**：三个方向换的都是同一份预算，付的账分别在
嵌入宽度 / 离散网格 / 路由复杂度上。这正是本域主线说的那件事。

## 4. 踩坑与失败模式

**① 用 `hidden_size ÷ num_attention_heads` 代替 `head_dim`。**

- **现象**：Qwen3-30B-A3B 上得到 `64`（真实字段是 `128`）⇒ attention 那一项的参数量
  从 `905,969,664` 掉到 `452,984,832`，**少算整整 50%**（三个投影的宽度全部减半）。
- **成因**：两个模型的 `head_dim` **来源不同**（§2：Llama 上是推导、Qwen3 上是原生）。
  而 Llama 上那个除法**恰好**给出正确答案（`4096/32 = 128`）⇒ 在 Llama 上验证过的写法，
  到 Qwen3 上就错了。**「两个模型都这么算」这句话只在其中一个上碰巧成立。**
- **规避**：**用字段，不做除法**；★ 那个例子的正文在
  `45-kernel-lifecycle/requirement-discovery/01-model-to-operator-inventory.md:270-276`
  （判据在 `:288`，正文在彼，此处回链）。

**② `tie_word_embeddings` 的两个口径混用。**

- **现象**：同一个模型，一个报告说 `7.505 B`、另一个说 `8.030 B`，看起来像换了模型。
- **成因**：`tie = True` 时嵌入与 `lm_head` **只算一次**，差的那一项**恰好是 `V·d`**
  （本机：`525,336,576`，占 untied 总量的 **6.54%**）。本域探针 `probes/probe_80g.py`【C】的
  `param_split` 默认就是 `tie_embed=True`，而计划 §0.7-D 的 D21 记的 `8.030 B` 是 **untied**
  —— **两个数不可混用**（计划 §4 R15 点名的现场）。
- **规避**：把 `TIE_EMBEDDINGS` 写成**显式的模块级常量**（本程序就是这样），
  并在正文里把两个数**都写出来**。★ 限定语：该字段是**镜像**取得、**未核实**（§2）。

**③ 把 MoE 当 dense 算。**

- **现象**：把 dense 的 FFN 公式套在 Qwen3-30B-A3B 上得到 `3.340 B`（本机），
  而模型名里的 nominal 规模是 `30 B` —— 差 **9 倍**。
- **成因**：它是 **MoE**：`intermediate_size` 只是「非 MoE 的 FFN 宽度」，
  参数量要按专家数 `E` 算（真值 ≈ 30.5 B 的账在 `moe` 主题；本域**不下这个结论**）。
- **规避**：见到 `num_experts` 一类字段就先停手 —— **MoE 的参数量不按 dense 公式算**。
  ⚠ 计划 §0.1 的 **P15** 就是这条，且它的现场（把专家宽度 `768` 填进 dense 公式）在 `moe` 主题。

**④ 只信「±1%」这条判据。**

- **现象**：反算得 `8.029995 B`，与公开值 `8.03 B` 差 `6.22e-07`，判据说「通过」——
  但同样的判据也会让 `7.95 B` 通过。
- **成因**：公开值只有 3 位有效数字（`±0.005 B` = `±6.2e-5` 相对），
  **判据的带宽比来源本身的精度宽得多**。
- **规避**：弱判据（±1%）只当**量级检查**；真正的对照用**逐位**的那一条
  （`45-…/code/01_operator_inventory.py:549` 的 `8029995008`），并写明它只是**一次对照**。
  ★ 这条的教训是通用的：**判据的阈值要落在「能被证伪」的量级上**。

**⑤ 把「等参数加深」当成精确的。**

- **现象**：按 `d = 4096/√2` 去设计一个 64 层的模型，发现参数量与设计值差 10% 以上。
- **成因**：`d` 必须落在 `head_dim` 的网格上，且 `num_key_value_heads` 得是整数（§1.3 的表）。
- **规避**：在网格上**枚举**候选宽度（本机四个：2688 / 2816 / 2944 / 3072），
  把偏离写出来再选，而不是用连续公式算出来直接用。

## 5. 判据表

| 可机械判定的条件 | 结论 |
|---|---|
| 反算 Llama-3-8B 的总参数量 | 落在公开值 `8.03 B` 的 **±1%** 内（本机 `6.22e-07`）；★ 这条带宽**远宽于来源精度**，只当量级检查 |
| 要与 45 域对齐 | 逐位等于 `45-…/code/01_operator_inventory.py:549` 的 `8029995008`（本机 `8,029,995,008`）；★ 本域**只做一次对照，不重做清单** |
| 报参数量而没写 `tie` | 结论不可用：两个口径差**恰好一项 `V·d`**（本机 `525,336,576`，占 6.54%） |
| 用到 `head_dim` | **用字段，不做除法**。Llama-3-8B 的 config **没有**该字段（⇒ 推导）；Qwen3-30B-A3B **有**且 `≠ d/h_q` |
| 规格表里标「原生」的字段 | 必须能在 config 的来源字典里找到（本程序 `B.5b` 机械判定） |
| 规格表里标「推导」的字段 | 必须**给出推导式且复现出该值**（本程序 `B.5b`；本域只有 `head_dim` 一个，且只在 Llama 上） |
| 字段的来源组 | 只能是 `upstream` / `mirror` / `derived` 三者之一，且三组**构成一个划分**（本程序 `B.5` / `B.6`） |
| 引用 `tie_word_embeddings` / `hidden_act` | 必须带「经 `hf-mirror.com` 镜像取得、**未与官方 repo 比对 sha256**」的限定语；`tie_word_embeddings` 还**必须保留上游「未核实」的标注** |
| 想「加深」 | `d ∝ L^(−1/2)` 可让 `attn + ffn` **逐位不变**（本机四点相同）；代价落在嵌入（`V·d` 按 `1/√mult` 缩）与**离散网格**上 |
| config 里出现 `num_experts` 一类字段 | **MoE**：不要用 dense 的 FFN 公式（本机差 9 倍）；账在 `moe` 主题 |

> ★ **表 A 的阈值**（计划 §3 Chunk 1）：⑥ 反算落在 `8.03 B` 的 **±1%** 内；
> ⑦ 规格表逐字段标注（机械判据：标「原生」的必须能在 config 里找到，标「推导」的必须给出推导式）。
> 分别在 `code/03_depth_width.py` 的 `B.1` / `B.2` / `B.5` / `B.5b` / `B.6` / `D.4b` 里，
> 阈值与来源被 `D.3` / `D.4` / `D.4b` / `D.6` 逐位钉住。
>
> ★ **表 B 的引用判据**（不能被程序判定，人工核对）：引用 `head_dim` 时必须**回链
> `45-kernel-lifecycle/requirement-discovery/01-model-to-operator-inventory.md:270-276`**。
> 本程序把该路径写成模块级常量并在 `D.5` 逐位钉住，同时在 `C.1` 与 `main()` 的头部打印出来。

## 6. 小结与延伸

**一句话**：深度与宽度之间的换算有一个可算的答案（`d ∝ L^(−1/2)` 让 `attn + ffn` 的参数守恒），
而它的**代价与限制**同样可算（嵌入变窄、宽度落在 `head_dim` 的网格上）；
至于「哪个字段是原生、哪个是推导」，**必须逐字段标出来**——同一个名字在不同模型上答案不同。

- **本文的账是「参数分配的账」**：`FFN 70.2% / attention 16.7% / 嵌入与输出头各 6.5%`。
- **FFN 的形态**（为什么是三个矩阵、`d_ff` 怎么定）见 `02-ffn-shapes`（同一主题的上一个编号）。
- **残差流**（那个被这两个子层反复读写的张量）见 `01-block-and-residual-stream`。
- **MoE 那条路**（用参数换激活）正文在 `80-model-architectures/moe`（此处回链）；
  本文 §4 第 3 条只给出「别把 MoE 当 dense」这条反例。
- **KV cache 的字节账**依赖本文的 `head_dim` / `num_key_value_heads` 口径，
  正文在 `130-quantization/kv-cache-quantization/01-kv-cache-basics.md:48-76`（正文在彼，此处回链）。
- **从 config 反算算子清单 / FLOPs**：正文在
  `45-kernel-lifecycle/requirement-discovery/01-model-to-operator-inventory.md`（正文在彼，此处回链）。

## 编译与运行

```bash
cd code
make run                   # 三个编号程序依次跑；本程序是 03_depth_width.py
python3 03_depth_width.py  # 只跑本程序（不到 0.1 s）
```

- 配套程序：**`code/03_depth_width.py`**（编号与外键严格对应）。
- 依赖：**纯标准库 + 整数运算**（本程序**不 import numpy**）；不需要网络、不需要 GPU。
- 失败标记：`!! FAIL: transformer`（本主题自定义，**不跨主题照抄**）。
- 程序的输出分四段：`A` 两条独立路径算参数量（闭式 vs **逐个模块枚举形状**）、
  `B` 代价模型（反算 / 分布 / 深度-宽度替换 / **规格表**）、
  `C` 边界与反例（`head_dim` / `tie` / MoE / 离散网格）、`D` 标记自检与来源钉住。

## 参考

**外部规格值**

- `Meta-Llama-3-8B` 与 `Qwen3-30B-A3B` 的 config 字段：转记自
  `45-kernel-lifecycle/requirement-discovery/code/specs.py` 的 `LLAMA3_8B_CONFIG` /
  `QWEN3_30B_A3B_CONFIG`（**[B]**，该模块自注「经 HuggingFace 与 torchtune 双向核对」/
  「经 4 处独立转抄核对」）。本域副本在 `code/specs.py`，每条带 `# 来源：` 与 `# 同值位置：`。
- `tie_word_embeddings`（`false`）与 `hidden_act`（`silu`）：**不在**那份文件里，
  经 `hf-mirror.com` 镜像取得（`NousResearch/Meta-Llama-3-8B`、`Qwen/Qwen3-30B-A3B`），
  **未与官方 repo 比对 sha256**。⚠ 上游 `45-…/code/specs.py:128` 逐字把前者标为「**未核实**」。
- 公开参数量 `8.03 B`：`45-…/code/specs.py` 的 `PUBLISHED_PARAM_COUNT_B`（[B]）。
- ★ **本域只有这两个模型的规格**：`huggingface.co` 直连在本环境不可达（计划 §C.0），
  其余模型的 config **在取到一手来源之前不入本域**。

**本机实测值 / 推导值**

- `code/03_depth_width.py` 的 `make run` 输出（`python3 3.13.5`，纯整数运算，
  macOS arm64，**无 GPU**）。§1.1–§1.3 的全部数字出自它。
- 计划 §0.7-D 的 **D21**（`8.030 B`）与 **R15**（`tie` 口径）是本程序 `C.2` 的对照点；
  本程序的 `8,029,995,008` 与 `45-…/01_operator_inventory.py:549` 的断言**逐位一致**。

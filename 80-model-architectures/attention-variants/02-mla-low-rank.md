---
title: MLA：低秩压缩、矩阵吸收，以及与 RoPE 的结构性冲突
slug: mla-low-rank
domain: 80-model-architectures
topic: attention-variants
level: 3
difficulty: advanced
prerequisites:
  - 80-model-architectures/attention-variants/01-mha-mqa-gqa
tags: [mla, low-rank-compression, matrix-absorption, decoupled-rope, kv-budget, deepseek-v2]
related:
  - 130-quantization/kv-cache-quantization/01-kv-cache-basics
  - 80-model-architectures/attention-variants/01-mha-mqa-gqa
  - 80-model-architectures/positional-encoding/01-rope-definition
code:
  - code/02_mla_low_rank.py
status: stable
updated: 2026-09-22
---

# MLA：低秩压缩、矩阵吸收，以及与 RoPE 的结构性冲突

> GQA 走的是「**少存几份**」（KV 头的条数按 `1/G` 降），MLA 走的是**另一条路**：
> 把每一条 KV **联合压到 `d_c` 维**（`d_c ≪ d_h·n_h`），于是 cache 里存的是
> 一个**潜向量 `c`**，`k`/`v` 在用时才恢复。
> 这条路的拦路虎是**位置旋转**：`R` 夹在 `q` 与 `c` 之间，**与上投影不可交换** ⇒ 吸收不进去
> —— 所以必须**解耦出一条共享的 `k^R`**，代价是 cache 里多 `d_h^R·l` 个元素。
> **「矩阵吸收」的账与「解耦 RoPE」的账，是 MLA 这一节的全部内容。**

## 0. 背景与动机

`01-mha-mqa-gqa` 的旋钮是「一条 KV 服务几个 query 头」——它减少的是 KV 的**份数**。
但只要 `head_dim` 不变，**每一份仍然是 `d_h` 宽**。

MLA（Multi-head Latent Attention，DeepSeek-V2）动的是**另一个因子**：把 K 和 V
**联合**压到一个 `d_c` 维的潜向量 `c`：

```
c = x · W_DKV          (d → d_c，d_c ≪ d_h·n_h)
k = c · W_UK ,  v = c · W_UV
```

⇒ **cache 里存的是 `c`，不是 `k`/`v`**。这是一条**低秩**假设：K 与 V 所在的
`(d_h·n_h)` 维空间里有一个 `d_c` 维的子空间够用。论文把它叫
"the low-rank **joint** compression for keys and values"（逐字）。

**这与本文的主线有什么关系**：从 MHA 变成 MLA，省下来的不是「更少的份」而是
「每份更小的宽度」，而**这个新结构带来一个原结构没有的问题**——
「用的时候要先把 `k` 算回来」，那一步能不能**免掉**（吸收进权重）？
答案是：**没有位置编码时可以，有位置编码时不行**。这就是本文的账。

★ **本域的边界**：KV 字节的六项乘积正文在
`130-quantization/kv-cache-quantization/01-kv-cache-basics.md:48-76`（正文在彼，此处回链）；
RoPE 的旋转定义与两套实现约定的正文在 `80-model-architectures/positional-encoding/01-rope-definition`
（**未建成**，前向声明）。本域只算**元素数**，只用 RoPE，不解释它。

## 1. 原理

### 1.1 矩阵吸收：为什么 `W_UK` 可以离线合并进 `W_Q`

不算位置编码时，分数矩阵是

```
scores = (x·W_Q) · (c·W_UK)^T = x·(W_Q·W_UK^T)·c^T            …… (1)
```

右端只有一个「先合并权重」的写法：**`W_eff = W_Q·W_UK^T`（`d × d_c`）与序列长度无关**
⇒ 它可以**离线算一次**，之后每次只需要 `(x·W_eff)·c^T`，**再也不需要显式算出 `k`**。
这就是「吸收」（论文用的词是 `absorbed into`）。

**本机实测**（`d=32`、`d_c=4`、`S=12`，`python3 3.13.5` + `numpy 2.4.4`）：
两条路径（① 显式算 `k`；② 先吸收再算）的最大逐元素差 **`8.882e-15`**，
而分数矩阵的量级是 `19.205`（相对 `4.62e-16`）。
★ **它不该是 `0.0`**：两条**真正独立**的算式（`(xW_Q)(cW_UK)^T` vs `(x(W_Q W_UK^T))c^T`）
的差必然停在机器精度上（见 §4 第 5 条）。

### 1.2 RoPE 为什么破掉吸收（论文逐字 + 可测形态）

DeepSeek-V2 论文（arXiv 2405.04434 §2.1.3）逐字：

> "However, **RoPE is incompatible with low-rank KV compression.** To be specific, RoPE is
> position-sensitive for both keys and queries. If we apply RoPE for the keys `k_t^C`, `W^{UK}`
> in Equation 10 will be coupled with a position-sensitive RoPE matrix. In this way, `W^{UK}`
> **cannot be absorbed into `W^Q`** any more during inference, since a RoPE matrix related to the
> currently generating token will lie between `W^Q` and `W^{UK}` and **matrix multiplication does
> not obey a commutative law**."

把这句话翻译成代数：正确形式是 **`q` 与 `k` 都被旋转**（同一段逐字：
`RoPE is position-sensitive for both keys and queries`）：

```
scores[t, s] = (R_t q_t) · (R_s k_s) = (R_t q_t) · (R_s W_UK^T c_s)      …… (2)
```

要让 (2) 也「可吸收」，就得存在一个**只依赖 `t`** 的 `d_c` 维向量 `u_t`，使

```
scores[t, s] = u_t · c_s         （对所有 t, s）                          …… (3)
```

因为吸收了之后 cache 里只剩 `c`（`d_c` 维），**能的只有 `d_c` 维的因子**。

★★ **`(3)` 是可以直接量出来的**——本程序的做法：解最小二乘 `min_U ‖A − U·Cᵀ‖_F`，
返回**相对残差**。本机实测（`d=32`、`d_c=4`、`S=12`）：

| 分数矩阵 | 「最好的可吸收近似」的相对残差 |
|---|---|
| **无 RoPE**（式 (1)） | **`4.658e-16`** ← 到机器精度 ⇒ **可吸收** |
| **有 RoPE**（式 (2)） | **`3.353e-01`** ← 一个量级 ⇒ **不可吸收** |

⇒ **「`W^{UK}` cannot be absorbed」这句话在数值上就是「同一把尺子下，残差从机器精度
跳到 O(1)」**。论文给的理由（`R` 夹在中间、乘法不可交换）与这个可测量是同一件事的两面。

★ **判据的非空条件（写进程序了）**：只有 `d_c < S` 时「可吸收」才**可能**失败
（`d_c ≥ S` 时 `d_c` 维因子足以张满 `R^S`，残差**恒为 0** —— 那是空断言）。
本程序的 `d_c = 4 < S = 12` 由 `D.3` 钉住。

### 1.3 解耦 RoPE：把不可吸收的那一半**单独留一份**

论文的解法（§2.1.3 逐字结构）：给 queries 一组**额外的、多头的** `q_{t,i}^R ∈ R^{d_h^R}`，
以及**一条 `k`** 来承载 RoPE：

> "we propose the decoupled RoPE strategy that uses additional multi-head queries `q_{t,i}^R ∈ R^{d_h^R}`
> and **a shared key `k_t^R ∈ R^{d_h^R}`** to carry RoPE, where `d_h^R` denotes the **per-head
> dimension of the decoupled queries and key**."

★★ **`a shared key` 这三个字是整笔账的关键**：解耦出来的那条 `k^R` 是**所有 head 共享的一条**，
所以它给 cache 增加的量是 `d_h^R`，**不是** `n_h · d_h^R`。

于是 MLA 每个 token 的 cache 是（§2.1.3 逐字）：

> "During inference, **the decoupled key should also be cached.** Therefore, DeepSeek-V2 requires
> a total KV cache containing **`(d_c + d_h^R)l`** elements."

### 1.4 可比口径：它等于「GQA 的 2.25 组」——★ 这是**推导**，不是论文给的常数

论文 Table 1 给了四种结构每个 token 的 KV 元素数（逐字）：

| 结构 | 元素数 |
|---|---|
| MHA | `2 n_h d_h l` |
| GQA | `2 n_g d_h l` |
| MQA | `2 d_h l` |
| **MLA** | **`(d_c + d_h^R)l`** |

令 MLA 与 GQA 相等，解出**等效组数**：

```
n_g = (d_c + d_h^R) / (2 d_h)                                    …… (4)
```

论文紧接着逐字给出它的两个选择："For DeepSeek-V2, `d_c` is set to `4d_h` and `d_h^R` is set to
`d_h/2`. So, its KV cache is **equal to GQA with only 2.25 groups**."

把两个比例代进 (4)：`(4d_h + d_h/2)/(2d_h) = 4.5/2 = 9/4 = 2.25`。
**本机用 `Fraction` 验过：精确 `9/4`**，且 Table 1 那一行 `≈ 9/2·d_h·l` 也精确成立（`9/2`）。

★ **两条纪律**（都是本节最容易写错的地方）：

1. **不要把它当成论文给的常数去引**：论文给的是**两个比例**，`2.25` 是**代进去算出来的**。
   本域的做法是**自己用式子 (4) 复现它**——这正是「每条账配一条闭式参照」的落法。
2. **`d_h` 在 (4) 里被约掉了** ⇒ 这条闭式**不需要 DeepSeek-V2 的绝对维度**，
   所以在只拿得到比例的机器上也能复核。

## 2. 执行映射

**「吸收」在推理时的含义**（这是一条**结构性**的推论，不是性能数字）：

| | 显式路径 | 吸收路径 |
|---|---|---|
| cache 里存 | `c`（`d_c` 维） | `c`（`d_c` 维）—— **一样** |
| 每次要算的 | `k = c·W_UK`，再 `q·k^T` | `(x·W_eff)·c^T` —— **少一次上投影** |
| 权重侧的一次性成本 | — | `W_eff = W_Q·W_UK^T`（`d×d_c`，**与序列长度无关**） |

★ **注意左列与右列的 cache 是同一份**：吸收**不改变存什么**，它改变的是
「用的时候要不要把 `k` 恢复出来」。这一区分很重要——MLA 的省是**压缩**带来的，
吸收只是把压缩的**恢复成本**挪走。

**解耦 RoPE 那一半落在哪**：`k^R` 与 `c` 一起被缓存（论文逐字 `should also be cached`），
而它的维度是 `d_h^R`（**共享**，不乘 `n_h`）。⇒ 本机实测的式子形状差异：

| 式子 | 对 head 数敏感吗 |
|---|---|
| MHA `2·n_h·d_h·l` | **敏感**：`n_h` 32→64 时元素数**翻倍**（本机 `256·d_h → 512·d_h`） |
| **MLA `(d_c + d_h^R)·l`** | **不敏感**：式子里**根本没有 `n_h`**（本机 `18·d_h`，把 head 数换掉也没处可传） |

⇒ **把 MLA 的式子按 MHA 的形状乘上 head 数会算错**，而它错得不显眼
（压缩是跨 head 联合的、解耦 key 是共享的——这两件事各自都会让 `n_h` 不进式子）。

> ⚠ **本域没有 GPU**：上表是本机 `numpy` 上 `d=32`、`d_c=4`、`S=12` 的玩具推导，
> **不是任何真实模型的测量**。它们支撑的是「吸收等价性」与「RoPE 破掉它」这两条**代数**结论。
> ★ 口径照抄论文的逐字：元素数 "measured by the **number of elements, regardless of the
> storage precision**"（与精度无关）。

## 3. 实现与演进

| 阶段 | 做法 | 它换来了什么、付了什么账 |
|---|---|---|
| MHA / GQA / MQA | 减少 KV 的**份数** | 见 `01-mha-mqa-gqa`；每份宽度不变（`d_h`） |
| MHA + 低秩压缩（朴素） | 压 `c`，但还要显式恢复 `k` | 省了 cache 的大小；**没省掉恢复那一步** → 吸收 |
| **MLA（吸收）** | `W_eff = W_Q·W_UK^T` 离线合并 | 恢复那一步**没了**；❌ 但**必须没有位置编码** |
| **MLA + 解耦 RoPE** | 额外留一条**共享的** `k^R` 承载 RoPE | 位置编码回来了；代价 = `d_h^R·l` 个元素（**不是** `n_h·d_h^R·l`） |

**为什么不能只旋转一边**：论文那句 `RoPE is position-sensitive for **both** keys and queries`
是硬要求。本机实测（固定 `delta`、扫 `m`，看分数是否只依赖 `delta`）：

| 写法 | 固定 `delta` 时分数随 `m` 的极差 |
|---|---|
| **q 与 k 都旋转**（正确） | **`1.776e-15`** ⇒ 只依赖 `delta` |
| **只旋转 k 不旋转 q**（错误） | **`10.2068`** ⇒ 随 `m` 变，**不满足相对位置性质** |

★ 这不是假想的错误：计划 §0.7-D 记录的探测脚本 `probe_80g.py` **初版就踩了这个坑**
（"显式"路径写成 `Q @ K_rope.T`，用的是**未旋转的 `Q`**）⇒ 那个对象**不是任何合法的注意力分数**，
拿它去和「错误吸收」作差得到的是「**两个都不合法的对象之差**」。收进仓库前已按正确形式改掉。

### 与 `01` 的关系：两条路的账可以放在一起看

`01` 的旋钮把 KV 的份数按 `1/G` 降；本文的旋钮把每份的宽度降到 `d_c`（+ 共享的 `d_h^R`）。
两者**乘起来**才是 KV 的预算（本域不把它写成字节公式——那在
`130-quantization/kv-cache-quantization/01:48-76`，正文在彼，此处回链）。
论文给的可比锚点是「**等于 GQA 的 2.25 组**」——用了 `01` 那张表的语言。

## 4. 踩坑与失败模式

**① 把 MLA 的 KV 元素数写成 `d_c·l`。**

- **现象**：公式里只留低秩压缩那一半，算出的数比真实小一截。
- **成因**：**解耦 RoPE 的那一半也在 cache 里**（论文逐字 `the decoupled key should also be
  cached`）⇒ 正确形式是 `(d_c + d_h^R)·l`。本机算例：两半之比是 `1 + d_h^R/d_c = 1.125`
  （按论文的比例 `4d_h` 与 `d_h/2`），漏掉它少算 **`d_h^R·l`**。
- **规避**：写元素数时**两个加数都写出来**；把「`d_h^R = 0` 时它退化成 `d_c·l`」当成
  **上限情形**而不是通式。

**② 只旋转 `k`，不旋转 `q`。**

- **现象**：算「吸收后差多少」时，两边用的是不同的对象，报出的数看着很大却**说明不了任何事**。
- **成因**：RoPE 对 queries 与 keys **都**是位置敏感的（论文逐字）；只转一边得到的东西
  **不满足相对位置性质**（本机：极差 `10.2068` vs 正确的 `1.776e-15`）。
- **规避**：凡是「RoPE 之后」的量，先做**相对位置性质的自检**（固定 `delta` 扫 `m`），
  过了再拿它作差。本程序的 `C.3` 就是这条自检。

**③ 把 `93.3%` 当通式引用。**

- **现象**：写「MLA 把 KV cache 减少 93.3%」。
- **成因**：论文的口径逐字是 "**compared with DeepSeek 67B**, … reduces the KV cache by 93.3%"
  —— 对比对象是**那一个具体的 dense 模型**，不是通式。★ 更隐蔽的一层：论文里
  `93.3` 出现 **4 次**，其中一次是 **ARC-Challenge 的准确率**（**同名不同指**）。
- **规避**：引用时必须同时给**论文自给的可比口径**——"equal to GQA with only **2.25 groups**"
  （这个才是与结构挂钩的），以及度量口径 `number of elements, regardless of the storage precision`。

**④ 把 `2.25` 当成论文直接给出的常数。**

- **现象**：正文写「论文说等于 GQA 的 2.25 组」而不给式子。
- **成因**：论文给的是**两个比例**（`d_c = 4d_h`、`d_h^R = d_h/2`），`2.25` 是**代进式子 (4) 算出来的**。
  照抄结论会丢掉它的**前提**（换一组比例，这个数就变）。
- **规避**：写出式子 `n_g = (d_c + d_h^R)/(2 d_h)` 并**自己算一遍**（本机 `Fraction` 得精确 `9/4`）。

**⑤ 把「吸收 == 显式」写成 `== 0.0`。**

- **现象**：断言「两条路径的差恒为 `0.0`」，再宣称「参考实现独立」。
- **成因**：这两条**互斥**（计划 §0.1 的 **P18**：不是「跑不了」，是「要不到」）。
  真独立的两条算式（`(xW_Q)(cW_UK)^T` vs `(x(W_Q W_UK^T))c^T`）的差必然停在机器精度上
  （本机 `8.882e-15`，而 `maxdiff == 0.0` 还有一种**相反含义**：两边共用了一份代码
  ⇒ **测不出来**，见 `00-meta/glossary.md:259`）。
- **规避**：判据用**机器精度档**（本程序 `ABSORB_TOL = 1e-12`），并配「参照实现的调用计数 = 0」。
  ★★ **这个「假通过」本程序刻意复现了一遍**（`C.4` 是**对照组**）：构造一个
  **直接调用被测实现**的「假参照」后，逐元素差是**精确的 `0.0`**，而 `A.1` 的容差判据
  （`< 1e-12`）**照样通过** ⇒ 容差对这种情况**没有分辨力**。
  ★ 揭穿它的是两把**机械**的尺子：`inspect.getsource` 不等（`A.3`）+ **AST 调用表**
  （假参照里含 `mla_explicit` = `True`、真参照里 = `False`），再配上**调用计数为 0**（`A.2`）。
  计划 §3 Chunk 2 的验收要求「这个现象必须能被复现」，`C.4` 就是它的落法。

## 5. 判据表

| 可机械判定的条件 | 结论 |
|---|---|
| 无位置编码时，「吸收」与「显式」两条路径的逐元素差 | 落在**机器精度档**（本机 `8.882e-15 < 1e-12`）。★ 它**不该是 `0.0`**：真独立的算式必然差在机器精度上 |
| **有**位置编码时，同一把尺子的量 | 差是**一个量级**（本机 `> 1.0`，与上面的机器精度差相差 `> 1e15` 倍）⇒ `W^{UK}` **吸收不进去** |
| 想判「能不能吸收」 | 用**同一把尺子**量：`min_U ‖A − U·Cᵀ‖_F / ‖A‖_F`。**无 RoPE** ⇒ 到机器精度（本机 `4.658e-16`）；**有 RoPE** ⇒ O(1)（本机 `3.353e-01`） |
| 上述判据的前置条件 | 必须 `d_c < S`：`d_c ≥ S` 时 `d_c` 维因子足以张满 `R^S`，残差**恒为 0** ⇒ **空断言** |
| MLA 的 KV 元素数 | 精确 `(d_c + d_h^R)·l`，**不是** `d_c·l`（漏掉解耦的那一半）；本机两半之比 `1 + d_h^R/d_c = 1.125`（按论文比例） |
| MLA 的元素数式子 | **不含 `n_h`**（压缩跨 head 联合、解耦 key 共享）；MHA 的式子**含 `n_h`** ⇒ 把两者形状混用会算错 |
| 「等于 GQA 的几组」 | 用 `n_g = (d_c + d_h^R)/(2 d_h)` 自己算：论文的两个比例代入得**精确 `9/4`** ⇒ 复现 `2.25 groups` |
| 引用 `93.3%` 的减少 | 必须带上对比对象（**DeepSeek 67B**，一个具体 dense 模型）；论文自给的可比口径是「等于 GQA 的 **2.25 组**」；度量口径逐字 `number of elements, regardless of the storage precision` |
| 见到论文里另一个 `93.3` | 那是 **ARC-Challenge 准确率**（同名不同指），不要与 KV 减少混为一谈 |
| 要在本域引用 RoPE 的旋转定义 | 不在本文：正文在 `80-model-architectures/positional-encoding/01-rope-definition`（**前向声明**，未建成）。本文只用它 |
| 想知道 KV 的**字节**账 | 不在本文：正文在 `130-quantization/kv-cache-quantization/01-kv-cache-basics.md:48-76` |

> ★ **表 A 的数值判据**（计划 §3 Chunk 2）：③ 吸收 vs 显式的差 `< 1e-12`（机器精度档）；
> ④ 把 RoPE 吸收进去后的差 `> 1.0`，且与 ③ 的差相差 `> 1e15` 倍；⑤ 元素数精确 `(d_c + d_h^R)·l`。
> ★★ **④ 的阈值刻意不写真数**：它是**单次抽样**的产物（控制者报的 `18.90`、审查者的 `12.64`、
> 实施者修正实现后的 `13.64` **都是单次抽样**，且前两个还是「两个都不合法的对象之差」）
> —— 只有**量级**承载信息，所以程序里只断言「`> 1.0` 且比 ③ 大 `> 1e15` 倍」，
> 打印出来的读数**不进正文**。
> ⚠ **但这条阈值有个要知道的边界**：那个比值的内在上限是 `1/机器精度 ≈ 4.5e15`
> ⇒ `1e15` 这个阈值与上限之间**不到一个数量级**的余量。
> ⇒ 它是**结构性量级**判据（「差从机器精度跳到 O(1)」），**不是**一个可以当精确判据用的数。

## 6. 小结与延伸

**一句话**：MLA 用**低秩**（每份更小）换掉了 GQA 的**份数**（更少份），
而这条路上唯一的结构性障碍是**位置旋转**——它夹在 `q` 与 `c` 之间、与上投影不可交换
⇒ 吸收不进去 ⇒ 必须**解耦出一条共享的 `k^R`**，代价是 cache 里多 `d_h^R·l` 个元素。

- **前置**：`01-mha-mqa-gqa`（KV 份数那条轴）。**本文是那条轴之外的第二个因子**。
- **RoPE 本身**：定义与「两套实现约定是同一操作被坐标置换共轭」的结论在
  `positional-encoding/01-rope-definition`（**前向声明**，正文在彼，此处回链）。
- **KV 的字节账与显存**：正文在 `130-quantization/kv-cache-quantization/01-kv-cache-basics.md:48-76`
  （正文在彼，此处回链）。★ 那边的「GQA 字节坑」与本主题 `01` 的 §4 第 2 条是同一处。
- **同主题的下一篇**：`03-window-and-hybrid` 换到**感受野**那一侧的账，
  并说明为什么「叠加两个变体」不是二选一。

## 编译与运行

```bash
cd code
make run                        # 三个编号程序依次跑；本程序是 02_mla_low_rank.py
python3 02_mla_low_rank.py      # 只跑本程序
```

- 配套程序：**`code/02_mla_low_rank.py`**（编号与外键严格对应）。
- 依赖：`numpy`（`requirements.txt` 里**只有**它）。**本程序不需要网络、不需要 GPU。**
- 失败标记：`@@ FAIL: attention-variants`（本主题自定义，**不跨主题照抄**）。
- 程序的输出分五段：`D`（口径/阈值/specs 钉住，**刻意放在最前**）、
  `A` 吸收与元素数（含闭式 `2.25`）、`B` RoPE 的冲突（含最小二乘残差那条判据）、
  `C` 代价与四个反例（含 `C.4` **共享代码的对照组**）、`D（续）` 标记自检与参照实现的独立性。

## 参考

### 表 B：引用判据（**不能被程序判定**，在**原始字节**上计数）

实跑（2026-09-22；`WebFetch` 被策略拦截 ⇒ 用 `curl`，`CONTRIBUTING:266-271`）：

```bash
curl -sS -o /tmp/arx_2405.04434.html https://ar5iv.labs.arxiv.org/html/2405.04434  # 694593 B
python3 - <<'EOF'
r = open('/tmp/arx_2405.04434.html','rb').read()
for s in ["RoPE is incompatible with low-rank KV compression",
          "RoPE is position-sensitive for both keys and queries",
          "matrix multiplication does not obey a commutative law",
          "the decoupled key should also be cached",
          "(d_{c}+d_{h}^{R})l", "a shared key",
          "low-rank joint compression for keys and values",
          "regardless of the storage precision", "2.25 groups", "93.3",
          'alttext="4d_{h}"', "is set to"]:
    print(r.count(s.encode()), s)
EOF
```

实测（`python3 3.13.5`）：

| 字面量 | 计数 | 用途 |
|---|---|---|
| `RoPE is incompatible with low-rank KV compression` | **1** | §1.2 的结论句 |
| `RoPE is position-sensitive for both keys and queries` | **1** | ★ 「**两边都要旋转**」的出处（程序 `C.3` 的反例） |
| `matrix multiplication does not obey a commutative law` | **1** | 不可吸收的代数理由 |
| `the decoupled key should also be cached` | **1** | ★ 判据⑤ 的出处（`d_h^R` 那一半**也在 cache 里**） |
| `(d_{c}+d_{h}^{R})l` | **4** | Table 1 那一行与正文 |
| `a shared key` | **1** | ★ 「解耦 key 是**共享**的」⇒ 元素数**不含 `n_h`** |
| `low-rank joint compression for keys and values` | **1** | 「压缩是**联合**的」⇒ 同上 |
| `regardless of the storage precision` | **1** | **度量口径**（与精度无关），引用元素数时必须带上 |
| `2.25 groups` | **2** | 可比口径（Table 1 说明与 §2.1.4） |
| `93.3` | **4** | ★ **同名不同指**：3 处是 KV 减少，1 处是 **ARC-Challenge 准确率** |
| `alttext="4d_{h}"` | **1** | ★ 论文「`d_c` is set to `4d_h`」那句里的**值**。⚠ 它被 MathML 标记切开（整句 `For DeepSeek-V2, d_c is set to 4d_h` 在原始字节上**计不到**；`is set to` 单独计到 **9** 处、无法定位）⇒ **跨标记的引文必须先看清结构再引** |

**外部来源值 / 论文**

- **MLA / DeepSeek-V2**：arXiv 2405.04434（**无 venue 线索 ⇒ 未确认同行评审**，技术报告；
  抓取 `https://ar5iv.labs.arxiv.org/html/2405.04434`，2026-09-22，**694593 B**，
  `sha256[:16] = 65b7c4dc9d1b38e0` —— 与计划 §C.2 逐位一致）。**[C]**
- `d_c = 4d_h`、`d_h^R = d_h/2` 两个**比例**：同上论文 §2.1.3 逐字（**[C]**，本域写入
  `code/specs.py`；★ 论文只给比例，**绝对维度在本环境没有一手来源** ⇒ 程序里以 `d_h` 为符号单位）。
- ★ **本域没有引用 DeepSeek-V2 的 `config.json`**：`huggingface.co` 直连不可达（计划 §C.0），
  而本域只允许引用 `45/…/code/specs.py` 里**已核实**的两个 config ⇒ 记入 `specs.UNSOURCED`。

**本机实测值**

- `code/02_mla_low_rank.py` 的 `make run` 输出：`python3 3.13.5` + `numpy 2.4.4`，
  macOS arm64，**无 GPU**。口径：`d = 32`、`d_c = 4`、`S = 12`、`l = 4`、种子 `20260922`、
  RoPE 的 `base = 10000.0`（**定义在 `positional-encoding/01`，此处只当口径**）。
  **全部是推导值**，不是真实模型的测量。
- D15 的三条口径记录（`probes/probe_80g.py`【A】）与本程序**同源**：
  那边 `d=32, d_c=8, S=6`；本程序改成 `d_c=4, S=12`（为了让「可吸收性」这条判据
  **非空**：`d_c < S`）。★ 那边报的 `18.90` 是**未旋转 q** 的产物，已作废。

**上游契约**

- `40-kernels/attention/04-warp-specialization-and-variants.md:128`：「三个常见变体，本文只讲
  **它们对 kernel 结构的影响**，**定义不重推**」—— 本文是那处**悬空让渡**的三份正文之一。
- `130-quantization/kv-cache-quantization/02-kivi-asymmetry.md:184` 逐字含 RoPE 的
  「`jointly rotates pairs of channels by different angles depending on the position`」；
  本域建成后应回链到 `positional-encoding/01`（Chunk 6 的动作）。

---
title: 容量因子：定义式、两条溢出协议，以及它那个同名的兄弟
slug: capacity-factor
domain: 80-model-architectures
topic: moe
level: 3
difficulty: advanced
prerequisites:
  - 80-model-architectures/moe/01-gating-and-routing
tags: [moe, capacity-factor, expert-capacity, token-dropping, no-token-left-behind, padding, gshard, switch-transformer]
related:
  - 40-kernels/moe-kernels/01-moe-compute-pattern
  - 40-kernels/moe-kernels/03-permute-and-unpermute
  - 00-meta/glossary
  - 80-model-architectures/moe/03-balance-and-granularity
code:
  - code/02_capacity_factor.py
status: stable
updated: 2026-09-22
---

# 容量因子：定义式、两条溢出协议，以及它那个同名的兄弟

> **`expert capacity = ceil((tokens per batch / number of experts) × capacity factor)`。**
> 这句话就是本域被上游**指名让渡 8 次**的那条正文。
> 它的取整口径**必须是 `ceil`**：`T/E` 不整除时，`floor` 会让**完全均匀**的负载也丢槽位
> （本机实测：`T=4096, E=48` 时丢 **16** 个），而 `ceil` 丢 **0** 个。
> 溢出**不是一条路而是两条**：Switch 用 **No-Token-Left-Behind（重新路由）**，
> 而「丢」的那条路（GShard 的 `G_{s,E}` **退化成零向量**）里，
> **被跳过的 token 仍然经残差流前进** —— 它有**该层输出 == 其输入**，并没有消失。

## 0. 背景与动机

本主题是全库**立足点最硬**的一个：**8 处**上游工件逐字写着「容量因子的定义与语义在
`80-model-architectures/moe/02-capacity-factor.md`」。它们列在 §6。**这一节就是那 8 处指的正文。**

为什么需要容量？因为**专家是有限的硬件资源**：一批 token 动态地被路由到 `E` 个专家上，
而张量的形状**必须静态定下来**。于是每个专家要被分配一个**固定的槽位数** `C`。

由此产生一个**三岔口**，本域的全部内容就是把它讲清楚：

| 情形 | 名字 | 代价 |
|---|---|---|
| 送来的 token **少于** `C` | **padding** | 白算（逐字：`an inefficient use of the hardware, but mathematically correct`） |
| 送来的 token **多于** `C` | **expert overflow** | 需要一条**协议** |
| 溢出时怎么办 | **两条路**（§1.3） | 一条丢、一条重新路由 |

★ **本域的边界**：permute / unpermute 的访存模式、slot mask 的**实现**、
grouped GEMM 的调度，都是 `40-kernels/moe-kernels/02,03` 的正文（计划 §2 的边界表）；
「槽位数 `C` 该取多少」这条**架构层**的问题由本域接住（上游 `01:270` 逐字让渡）——
但本域的答案**不是一个最优值**，而是**两笔账**（§1.4 与 `03` 的 `C.4`）。

## 1. 原理

### 1.1 ★★ 定义式（逐字 + 取整口径）

逐字（Switch Transformer §2.2，arXiv 2101.03961，**[C] + 已确认同行评审**
（abs 页 `comments` 逐字为 `JMLR`））：

> "The expert capacity—the number of tokens each expert computes—is set by **evenly dividing
> the number of tokens in the batch across the number of experts**, and then **further
> expanding by a capacity factor**, `expert capacity = (tokens per batch / number of experts)
> × capacity factor`."

> "**A capacity factor greater than 1.0 creates additional buffer** to accommodate for when
> tokens are not perfectly balanced across experts."

**⇒ 本域的定义（写死口径）：**

```
expert capacity  C  =  ceil( (tokens per batch / number of experts) × capacity_factor )
```

★★ **取整口径必须是 `ceil`，理由是可测的**：容量是**整数个槽位**，
而 `T/E` 一般不整除。若用 `floor`，那么**负载完全均匀**时也会丢东西：

| 口径 | 容量 | 完美均衡负载下的丢弃 |
|---|---|---|
| `floor` | `4096 // 48 = 85` | **16 个槽位** |
| `ceil` | `4096 / 48 → 86` | **0 个** |

（`T = 4096`、`E = 48`，负载是完美轮转的 `86×16 + 85×32 = 4096`；本机实测 `A.2`。）

> 一句话：**`floor` 会把「均衡也丢」写进定义里**，而容量因子这个词的全部意义是
> 「`cf > 1` 是给**不均**留的缓冲」⇒ 两者自相矛盾。

★ **实现上不要走浮点**：`ceil(T/E × cf)` 复合之后不再可靠。配套程序把 `cf` 的分子分母
以整数传进 `Fraction`，做**精确** `ceil`（`A.1`）：

| `cf` | 1.0 | 1.25 | 1.5 | 2.0 |
|---|---|---|---|---|
| `T = E =`… 的容量（`T=4096, E=64`） | **64** | **80** | **96** | **128** |

### 1.2 padding 与 overflow 是**两笔账**，不能合并

原文同一段把两者**分开写**：

> "If the number of tokens sent to an expert is **less than** the expert capacity, then the
> computation may simply be **padded** – an inefficient use of the hardware, but
> **mathematically correct**. However, when the number of tokens sent to an expert is
> **larger** than its capacity (**expert overflow**), a protocol is needed to handle this."

| | 触发条件 | 后果 | 是数值问题吗 |
|---|---|---|---|
| padding | `实际 < C` | 白算（槽位开着但没数据） | **不是**（`mathematically correct`） |
| overflow | `实际 > C` | 需要协议（§1.3） | **是** |

★ **`cf > 1` 买到的是 padding 的浪费**：负载完美均衡时槽位利用率 = `1/cf`
⇒ 白算比例 = `1 − 1/cf`（`cf = 2.0` 时**恰好 50%**，`A.3`）。
这是**推导值**、与分布无关 —— 它与 overflow 那一笔**不在同一个旋钮上**。

### 1.3 ★★ 两条溢出协议**必须并列**

这是本域最容易写错的一处。

#### 协议甲：dropping —— 「该层 FFN 被跳过，token 仍经残差流前进」

逐字（Switch §2.2，**[C]**）：

> "If too many tokens are routed to an expert (referred to later as **dropped tokens**),
> **computation is skipped and the token representation is passed directly to the next layer
> through the residual connection**."

GShard 的同义表述（§2.2，**[C] + 未确认同行评审**）：

> "When both experts selected by a token already exceed their capacity, the token is
> considered as an overflowed token, where `G_{s,E}` **degenerates into a zero vector**.
> Such tokens have their representation `x_s` **passed on to the next layer via residual
> connections**."

★★ **两篇说的是同一条路**：`G` 退化成零向量 ⇒ FFN 的**加数为 0** ⇒ 残差把它**原样**送下去。
⇒ **被跳过的是「这一层的 FFN 计算」，不是 token 本身。**

**可执行的断言形式**（本机实测，`T=64, E=4, k=2`，容量 `16`，`B.1`）：

- `23 / 64` 个 token 的 `k` 个选择**全被跳过**；
- 这 23 个 token 的**该层输出逐位等于其输入**（`np.array_equal` ⇒ 真·恒等，不是「接近」）；
- 它们**仍出现在下一层的输入里**（该行逐位等于 `x[t]`）；
- ★ **对照组**：同层**有**槽位的 41 个 token 里，**无一**满足这一条
  —— 少了这个对照，「输出 == 输入」可能只是「所有 token 都这样」。

> ★★ **「怎么讲才对」的两种写法**（都合法，但说的不是一件事）：
> - **残差口径**：`该层输出 == 其输入`（本文与 Switch/GShard 原文用的这一种）；
> - **贡献口径**：该 token 的 **FFN 贡献恰为 0**（`40-kernels/moe-kernels/03:195`
>   与 `00-meta/glossary.md:240` 用的是这一种）。
>
> 两者**不冲突**：`out = x + (贡献)`，贡献为 0 ⇒ `out == x`。
> ★★ **但「该 token 的输入输出为 0」不是这两条中的任何一条** ——
> 「被丢弃的 token 被前向输出 0」是把**贡献**说成了**整条残差流**，
> 那是 §4.1 要挡的那个误读。

#### 协议乙：No-Token-Left-Behind —— **重新路由**

逐字（Switch 附录 B，**[C]**）：

> "**We suspected that having no computation applied to tokens could be very wasteful**,
> especially since if there is overflow on one expert, that means another expert will have
> extra capacity. With this intuition we create **No-Token-Left-Behind**, which **iteratively
> reroutes any tokens that are at first routed to an expert that is overflowing**."

★★ **两条原文口气都要保留**：

| 原文逐字 | 读法 |
|---|---|
| `we create No-Token-Left-Behind` | 这是**一条**做法 |
| `will allow us to guarantee **almost no** tokens will be dropped` | 它是 `almost`，**不是** `no` |
| `We hypothesised that this could improve performance … **but we found no empirical benefits**.` | ★ 原文**自承没找到收益** |

**可执行的断言**（`C.1`/`C.2`，本机实测）：

- `C.1`：同一份路由、同一个容量 —— dropping 留下 **20** 个 token「一个专家都没拿到」；
  NTLB 之后**每专家条数 ≤ 容量**（`[16, 16, 16, 16] ≤ 16`）且**没有一个 token 拿不到专家**。
- `C.2`：★ **NTLB 不是普遍保证**：把容量压到每个专家 1 个槽位（总槽位 `4 < 64`），
  它**照样留下 60 个** token 无人服务 ⇒ 它做的是**重新分配**，不是**增加容量**。
  ⇒ 引用时**不能**写成「overflow 不再是问题」。

⚠ **本域的 NTLB 实现是一条口径，不是论文的算法**：论文只给了**意图**与**一条界**，
**没有给伪码** ⇒ 配套程序用最简单的贪心（每个 token 有一张完整偏好序，按序找第一个还有空位的
专家），并在程序里明写这一点。**不要把本实现当成论文的实现。**

### 1.4 ★★ 架构层的那个决定：幸存权重要不要重新归一化

上游**指名让渡**给本域的就是这一条 ——
`40-kernels/moe-kernels/03-permute-and-unpermute.md:197` 逐字：

> 「丢掉之后**要不要把幸存权重重新归一化**、**要不要给这类 token 补一条旁路**，
> 属于**架构层的决定**（正文在 `80-model-architectures/moe/02-capacity-factor.md` 的 §1.4，此处回链）」

同文 `:238` 逐字（⚠ 该行**不是**让渡句 —— 它里面没有「正文在本域」的括注；本节引用的是它的**内容**）：

> 「**做 mask 之后输出并不是 `X[t]` 本身**：它是 `X[t] × 幸存权重之和`（本场景用 §C2 的构造
> 可以验证这一点）。**要不要重新归一化是架构层的选择**；kernel 侧只要保证这个选择是可表达的。」

**⇒ 本域的答案（三条并列，不是一条）：**

| 做法 | 该 token 的贡献 | 什么时候用 |
|---|---|---|
| **不归一化** | `Σ_幸存 w_j · f_j(x)`，被 `Σw < 1` **缩小** | 保守：不动权重的语义，代价是这个 token 的该层贡献变小 |
| **重新归一化** | 除以 `Σw` ⇒ 尺度回到「正常 token」的量级 | 让所有 token 的该层输出尺度一致 |
| **补一条旁路** | 就是 §1.3 的残差：`out == x` | 该 token 的 `k` 个选择**全**被跳过时 |

**可执行的断言**（`C.3`，本机实测）：

- 两条路**各跑一次前向**（一次用 `w`、一次用 `w/Σw`），贡献满足
  `cA == Σw · cB`，最大绝对偏差 **`2.220e-16`**（相对 `1.670e-16`）
  —— **恒等式**，不是近似；
- **非空条件**：`24` 个 token 的 `0 < Σw < 1`（最小 `0.500000`）⇒「不归一化」**确实缩小了**
  它们的贡献（`‖cA‖ < ‖cB‖` 在全部 24 个上都成立）；
- 另有 `20` 个 token 的 `Σw == 0`（`k` 个选择全被丢）—— 那正是 §1.3 的残差那一格。

★ **为什么这条要单独成节**：它是**唯一**一处「kernel 侧刻意不做、留给架构层」的决定
（上游逐字说 kernel 的责任是「**让这个决定成为可能**」）。
本域不给唯一答案，给的是**三条路各自的账**。

### 1.5 ★★ `capacity factor` 有**两个同名的兄弟**

| 出处 | 逐字 | 它的 `factor` 指什么 |
|---|---|---|
| **Switch** 式 (3) | `expert capacity = (tokens per batch / number of experts) × capacity factor` | **在均分之上额外开多少缓冲** |
| **Expert Choice** 式 (1) | `k = n × c / e`，且「The capacity factor `c` denotes **on average how many experts are utilized by a token**.」 | **一个 token 平均被几个专家服务** |

**⇒ 闭式恰好相同（都是 `(tokens/experts) × factor`），语义不同。** 本机实测（`C.4`）：

| | 每专家容量（`T=4096, E=64, factor=2`） | 被**用掉**的槽位 |
|---|---|---|
| Switch（top-1，`cf = 2.0`） | `ceil((4096/64) × 2.0)` = **128** | **4096**（`= T`，fan-out 1） |
| Expert Choice（`c = 2`） | `ceil((4096/64) × 2)` = **128** | **8192**（`= T·c`） |

★ **同一个数、两个含义**：Switch 的 `cf` 是**缓冲**（`cf = 2.0` 时可能有**一半槽位空着**）；
EC 的 `c` 是**份额**（`c = 2` 时**每个槽位都要有人**）。
⇒ **引用 `capacity factor` 必须写出是哪一篇的含义** —— 否则「`cf = 2`」可以指
「开了两倍的槽」也可以指「每个 token 过两个专家」。

★ GShard 走的是**第三条路**：它给的是**渐进记号** `O(N/E)` 与「**fractional capacity** `2N/(G·E)`」，
**不用 `capacity factor` 这个词**（原始字节计数：`capacity factor` 在 GShard **0 次**、
在 Switch **11 次**）。

### 1.6 「槽位数 `C` 该取多少」

上游 `40-kernels/moe-kernels/01-moe-compute-pattern.md:270` 逐字把这条让渡给本域：

> 「**「槽位数 C 该取多少」是架构层的问题**（正文在 `80-model-architectures/moe/02-capacity-factor.md` 的 §1.6，此处回链）」

**本域的答案：它不是一个最优化问题，而是两笔账 + 一条依赖。**

| 旋钮 | 往大调 | 往小调 |
|---|---|---|
| **容量**（`cf`） | 丢弃 ↓（判据 ④），**padding 白算 ↑**（`1 − 1/cf`） | 白算 ↓，**丢弃 ↑** |
| **粒度**（`E`、`moe_intermediate_size`，见 `03`） | 每个专家更小、专家更多 | 反之 |

★ **一条依赖（本机实测，`03` 的 `C.4`）**：`C / 平均负载 ≈ cf / k` **与 `E` 无关**
（两处同时按 `1/E` 缩放），但**丢弃率不是** —— 固定 `cf = 1.5`、热点比例固定为 `E/8` 时：

| `E` | 32 | 64 | 128 |
|---|---|---|---|
| `C` | 192 | 96 | 48 |
| **丢弃率** | **18.14%** | **22.95%** | **27.71%** |
| 专家条数的 CV | 0.7391 | 0.8866 | 1.0341 |

⇒ **「`C` 该取多少」的答案里必须同时出现 `T`、`E` 与负载的形状，`T` 一个数不够。**
（本域**不给「最优 `C`」** —— 那是另一个问题。）

## 2. 执行映射

★ 本域**不讲** permute / unpermute 的访存映射与 slot mask 的**实现** ——
正文在 `40-kernels/moe-kernels/03-permute-and-unpermute.md`（含 `:47` 的指名让渡、`:189-197`、
`:221-249`）与 `code/03_permute_unpermute.cu`（正文在彼，此处回链）。

本域只记**容量的三条下游后果**：

1. **`valid` 数组必须存在**（`40` 的 §1 第六步）—— 容量定了以后，
   「哪些 slot 要 mask」才有确定的说法。**架构层决定的正是「怎么用它」**（§1.4）。
2. **permute 时对无效 slot 不写** ⇒ 那个位置保留**上一次迭代的残留**。
   不做 mask 就会把残留当激活加进去（`40` 的 §C2 实测：最大误差 `3.490e+00`，
   量级与参考输出 `5.748e+00` 同阶）—— 那是**正确性**问题，不是数值噪声。
3. **`C` 只影响那一个数组的语义**，不影响 `d_model` 与专家宽度 —— 所以「容量」这一笔
   与「粒度」那一笔（`03`）是**正交**的两个旋钮。

## 3. 实现与演进

### 3.1 容量因子**不是「越大越好」**（两条都要带语境）

★★ **两条原文的语境必须一起写**，否则会把一次观察读成设计律。

**（a）质量侧**（Table 1 的 finding (3)，**[C]**）逐字：

> "Switch Transformers **perform better at lower capacity factors (1.0, 1.25)**. Smaller expert
> capacities are indicative of the scenario in the large model regime where model memory is
> very scarce and the capacity factor will want to be made as small as possible."

⚠ **语境**：这条 finding 属 **Switch vs MoE vs Dense 的比较**语境（同一段逐字：
`A head-to-head comparison of the Switch Transformer and the MoE Transformer is presented in
Table 1`、`All models were trained for the same number of steps on identical hardware`）。
⇒ **不可读成普遍结论。**

**（b）速度侧**（§2.3，**[C]**）逐字：

> "Note that **the MoE model** going from capacity factor 2.0 to 1.25 **actually slows down
> (840 to 790)** in the above experiment setup, **which is unexpected**."

⚠ **三条口径**：(i) 主语是 **MoE 模型**（不是 Switch）；(ii) 方向是「**降** `cf` **变慢**」
（`840 → 790`，这个数与质量那一侧的方向**相反**）；(iii) 原文**自承 `unexpected`**，
且同段脚注逐字：「speed measurements are both a function of the algorithm and the
**implementation details**」。
⇒ 这**不是**一条可以推广的设计律，是一次带自承意外的观察。

### 3.2 与 `00-meta/glossary.md` 的关系（**并列，不改上游**）

`00-meta/glossary.md:240` 逐字把定义让渡给本域，并描述了 kernel 侧关心的三种状态：

> 「容量因子 (capacity factor) … **定义与语义的正文在 `80-model-architectures/moe/02-capacity-factor.md`**，
> kernel 侧只关心它带来的三种状态：正常、被截断（要 mask）、以及 **`k` 个选择全被截断**——
> 最后这种 token 前向输出恰为 0、反向梯度也恰为 0，**在本步训练里完全不存在**。」

★★ **那段描述是它作用域内正确的表述**（它说的是**贡献**：`G` 退化成零向量 ⇒ 前向贡献 0、
反向梯度 0）。**本域不判它错、不改它**，只把**两条协议并列**：
`glossary` 那一段对应的是 **dropping** 协议下的**贡献口径**；
而 **NTLB** 是另一条路（重新路由，不丢）。
⇒ 两者合起来才是完整的图景（计划 §3 Chunk 3 的验收逐字要求「兼容但不冲突」）。

### 3.3 丢弃率：**口径不说清，数就没意义**

★★ 同一份路由（`T=4096, E=64, k=2`，`cf = 1.5`，热点偏置 `+1.2`），
**四种数法给出四个数**（本机实测，`B.3`）：

| 口径 | 容量施加在哪 | 分母 | `cf = 1.0` | `cf = 2.0` |
|---|---|---|---|---|
| **甲**（本域与探测脚本 D16 采用） | 每个 `(j ∈ top-k, e)` **各自** `ceil(T/E·cf)` | `T·k` | **48.93%** | **36.43%** |
| 乙（合并，容量也乘 `k`） | 每个专家 `ceil(T·k/E·cf)`，所有来源合并 | `T·k` | 48.93% | 36.43% |
| 丙（合并，容量不乘 `k`） | 每个专家 `ceil(T/E·cf)`，所有来源合并 | `T·k` | **55.49%** | **48.93%** |
| 丁（token 口径） | 甲之上，数**有选择被丢的 token** | `T` | **77.66%** | **68.46%** |

（丁这一行**必须**走一遍真正的分配，所以它**依赖到达顺序**；甲/乙/丙是**顺序无关**的计数。
丁的另外两个数：`k` 个选择**全**被丢的 token 占 `33.33%`（`cf=1.0`）/ `29.39%`（`cf=2.0`）。）

★ 甲与乙在本抽样上**同值**，但 ⚠ **这不是恒等式**：只有当两个来源都越过各自容量时，
两者的截断量才恒等（`(a−c)+(b−c) = (a+b)−2c`）；任一来源没满就不等。
★★ **⇒ 报出「丢弃率」时必须同时说清三件事**：
**(i)** 容量施加在什么**粒度**上；**(ii)** 分子的**单位**（槽位 / token）；
**(iii)** 分母是 `T·k` 还是 `T`。

★ **判据 ④ 的走向在两个口径下都成立**（甲：`48.93% → 45.80% → 42.68% → 36.43%`，
四点**严格递减**且 `cf = 1.0` 时 `> 0`）。

## 4. 踩坑与失败模式

### 4.1 ★★ 把「被跳过」说成「token 被丢弃 / 输出为 0」

- **现象**：写「超出容量的 token 被丢弃了」「这些 token 的前向输出是 0」。
- **成因**：把**贡献**（`G_{s,E}` 退化成零向量）说成了**整条残差流**。
  原文的机制句其实很直白：`computation is skipped and the token representation is **passed
  directly to the next layer through the residual connection**`。
- **规避**：用**残差口径**的断言形式 —— 被跳过的 token「**该层输出 == 其输入**」，
  **且它仍出现在下一层的输入里**（§1.3 的可执行形式，`B.1`）。
  ★ 顺带一提：这条误读的常见形态是「MoE 会丢 token」，而原文用的是
  `dropped tokens` 这个词（原始字节 **2 次**；`dropped` **5 次**）——
  **词是一样的，指的却是「该层的 FFN 计算被跳过」**。

### 4.2 拿「容量」当「上界」而不说它作用在什么上

- **现象**：报一个丢弃率，读者按「每专家一个容量、所有来源合并」去理解，
  复现出来的数差 6.6~12.5 个百分点。
- **成因**：容量施加的**粒度**（每 `(j,e)` 还是每专家）与**分母**都没有写。
- **规避**：§3.3 的四行表；配套程序 `B.3` 把「三种口径给出不同的数」做成断言。

### 4.3 把 No-Token-Left-Behind 当普遍保证

- **现象**：写「用 NTLB 就不会丢 token 了」。
- **成因**：原文逐字是 `guarantee **almost no** tokens will be dropped`；
  而 NTLB 做的是**重新分配**，不是**增加容量**。
- **规避**：`C.2` 把「容量不够时它照样留无人服务的 token」做成断言（本机：`60/64`）。
  ★ 而且原文自己说它**没有找到收益**：`we found no empirical benefits`。

### 4.4 把容量因子当「越大越好」的旋钮

- **现象**：写「把 `cf` 调大以免丢 token」。
- **成因**：只看了 §1.2 的 padding 那一笔的**反向**，或把 Table 1 的 finding (3)
  脱离了语境。
- **规避**：§3.1 两条并列 + 语境的逐字；★ 且注意**两条的方向是相反的**
  （质量侧说低 `cf` 好、速度侧那次观察说降 `cf` 变慢）——
  **它们不是同一个量上的同一条结论**。

### 4.5 把 `cf` 的「缓冲」含义当成「每个 token 用几个专家」

- **现象**：看到 Expert Choice 的 `capacity factor of 2` 就以为它与 Switch 的 `cf = 2.0` 是一回事。
- **成因**：**同名不同指**（§1.5）。
- **规避**：以「**被用掉的槽位数**」为判据 —— Switch top-1 是 `T`、EC 是 `T·c`（`C.4`）。

## 5. 判据表

| 可机械判定的条件 | 结论 |
|---|---|
| `expert capacity` 的取值 | **逐位等于** `ceil((tokens/experts) × cf)`，用**精确**有理数算（本机：`T=4096, E=64` 时 `cf ∈ {1.0, 1.25, 1.5, 2.0}` ⇒ `{64, 80, 96, 128}`） |
| 取整方向 | 必须 **`ceil`**：`T/E` 不整除时 `floor` 会让**完美均衡**的负载也丢槽位（本机 `T=4096, E=48`：`floor` 丢 **16**、`ceil` 丢 **0**） |
| `cf > 1` 的代价（负载均衡时） | **padding 白算** = `1 − 1/cf`（`cf=2.0` ⇒ **50%**）—— 与 overflow 那一笔**分开算** |
| `实际 < C` / `实际 > C` | 前者是 **padding**（`mathematically correct`）；后者是 **overflow**，需要协议 |
| dropping 协议下被跳过 token 的**该层输出** | **逐位等于其输入**，且它**仍出现在下一层的输入里**；同层有槽位的 token **无一**满足 ⇒「dropping ≠ token 消失」 |
| 丢弃率随 `cf` 的走向 | **严格递减**（本机甲口径：`48.93 → 45.80 → 42.68 → 36.43%`），且 `cf = 1.0` 时 **> 0** |
| 报出任何丢弃率 | 必须附三件事：容量的**粒度**、分子的**单位**、**分母**（四种组合本机差 6.6~28.7 个百分点） |
| NTLB 之后 | 每专家条数 **≤ 容量**、**没有 token 拿不到专家**；★ 但容量本身不够时它**照样**留（本机 `60/64`） |
| 幸存权重归一化 | 两条路（各跑一次前向）满足 `cA == Σw · cB`（本机偏差 `2.220e-16`）；`0 < Σw < 1` 的 token 上“不归一化”**确实缩小**贡献 |
| 引用 `capacity factor` | **必须写明是哪一篇的含义**：Switch = 缓冲（用掉 `T` 个槽位）、Expert Choice = 每 token 平均几个专家（用掉 `T·c`）；GShard **不用这个词**（它用 `O(N/E)` 与 `fractional capacity`） |
| 引用「低 `cf` 更好」 | 必须带语境（Switch/MoE/Dense 同硬件同步数的比较）；且速度侧那次观察方向**相反**并自承 `unexpected` |
| 想知道 permute / slot mask / grouped GEMM 的实现 | **不在本文**：正文在 `40-kernels/moe-kernels/02,03`（含 `:47`、`:189-197`、`:221-249`） |

> ★ **表 A 的数值判据**（计划 §3 Chunk 3）：
> ② `expert capacity` 逐位等于 `ceil((T/E) × cf)`（取整口径写死为 `ceil` 并给出理由）；
> ③ 两条溢出协议各自可运行（dropping 的断言形式 = 「该层输出 == 其输入」且仍在下一层输入里）；
> ④ 丢弃率随 `cf` 严格单调下降且 `> 0`。
> 三条都在 `code/02_capacity_factor.py` 的 `A.1`/`A.2`/`B.1`/`B.2` 里，
> 阈值被 `D.1`/`D.2` 逐位钉住。

## 6. 小结与延伸

**一句话**：容量因子的定义式只有一行（`ceil((T/E) × cf)`），
但它的**取整口径**、**两条溢出协议**、**一个同名的兄弟**、以及**四种数法的丢弃率**，
都是必须写下来的东西。

- **向前**：`01-gating-and-routing.md`（谁选谁决定了负载的形状）。
- **同主题的下一篇**：`03-balance-and-granularity.md` —— 容量是**事后**的截断，
  而 aux loss 是**事前**的拉平；两笔账合起来才是「负载」这一栏的完整内容。
- **本文接住的上游指名让渡（8 处）**：`00-meta/glossary.md:240`、
  `40-kernels/moe-kernels/01:270`、`02:198`、`03:47`、`03:197`、
  `code/verify.py:887`、`code/verify.py:1170`（该字符串跨两行，引用句在 `:1171`）、
  `code/03_permute_unpermute.cu:341`。
  ★ 其中 `03:197` 是**本文 §1.4** 接住的（「要不要重新归一化」），
  `01:270` 是**本文 §1.6** 接住的（「槽位数 C 该取多少」）。
  ⚠ `03:238` **不是**让渡句（它里面没有指向 `80-model-architectures/moe/…` 的让渡括注），
  而是**同一段里被引用的一句**：「做 mask 之后输出并不是 `X[t]` 本身：它是
  `X[t] × 幸存权重之和`」—— 它同样是**本文 §1.4** 的正文（见本节末的契约表）。

## 编译与运行

```bash
cd code
make run                          # 三个编号程序依次跑；本程序是 02_capacity_factor.py
python3 02_capacity_factor.py     # 只跑本程序
```

- 配套程序：**`code/02_capacity_factor.py`**（编号与外键严格对应）。
- 依赖：`numpy`（`requirements.txt` 里**只有**它）。**本程序不需要网络、不需要 GPU。**
- 失败标记：`## FAIL: moe`（本主题自定义，**不跨主题照抄**）。
- 程序的输出分五段：`D`（口径/阈值/specs 钉住，**刻意放在最前**）、
  `A` 定义式与取整口径、`B` dropping 协议与丢弃率、
  `C` NTLB / 归一化决定 / 同名不同指 / X6 的分寸、`D（续）` 标记自检与两条协议的独立性检查。

## 参考

### 表 B：引用判据（**不能被程序判定**，在**原始字节**上计数）

实跑（2026-09-22；`WebFetch` 被策略拦截 ⇒ 用 `curl`）：

```bash
curl -sSL -o /tmp/arx_2101.03961.html https://ar5iv.labs.arxiv.org/html/2101.03961  # 413955 B
curl -sSL -o /tmp/arx_2006.16668.html https://ar5iv.labs.arxiv.org/html/2006.16668  # 451967 B
curl -sSL -o /tmp/arx_2202.09368.html https://ar5iv.labs.arxiv.org/html/2202.09368  # 175642 B
python3 - <<'EOF'
for f in ("/tmp/arx_2101.03961.html","/tmp/arx_2006.16668.html"):
    b = open(f,'rb').read()
    for s in ["The expert capacity","is set by evenly dividing the number of tokens in the batch",
              "further expanding by a capacity factor","A capacity factor greater than 1.0 creates additional buffer",
              "computation is skipped and the token representation is",
              "passed directly to the next layer through the residual connection",
              "With this intuition we create","No-Token-Left-Behind",
              "iteratively reroutes any tokens that are at first routed to an expert that is overflowing",
              "without processing through a residual connection which we also follow",
              "wasted computation and memory","expert overflow","padding",
              "capacity factor","fractional capacity of each expert","Random routing",
              "degenerates into a zero vector",
              "passed on to the next layer via residual connections",
              "guarantee almost no tokens will be dropped","we found no empirical benefits",
              "actually slows down (840 to 790)","which is unexpected",
              "perform better at lower capacity factors (1.0, 1.25)",
              "Smaller expert capacities are indicative of the scenario"]:
        print(b.count(s.encode()), s[:72])
EOF
```

实测（`python3 3.13.5`）：

| 字面量 | Switch | GShard | 用途 |
|---|---|---|---|
| `The expert capacity` | **1** | 0 | ★★ 定义式的**散文那一句**（§1.1） |
| `is set by evenly dividing the number of tokens in the batch` | **1** | 0 | ★★ 定义式的「均分」那一半 |
| `further expanding by a capacity factor` | **0** | 0 | ⚠★ **跨标记引文在原始字节上计不到**（该句被 `<span>` 切开 ⇒ 只能整句引用上面那一行） |
| `A capacity factor greater than 1.0 creates additional buffer` | **1** | 0 | ★★ `cf > 1` 的语义（**缓冲**） |
| `computation is skipped and the token representation is` | **1** | 0 | ★★ **X5 的立足点**（dropping 协议） |
| `passed directly to the next layer through the residual connection` | **1** | 0 | ★★ 同上（**token 没有消失**） |
| `With this intuition we create` | **1** | 0 | NTLB 的引出句 |
| `No-Token-Left-Behind` | **5** | 0 | NTLB 的名字 |
| `iteratively reroutes any tokens that are at first routed to an expert that is overflowing` | **1** | 0 | ★★ NTLB 的**机制句** |
| `without processing through a residual connection which we also follow` | **1** | 0 | ★ Switch 明说它**沿用** GShard 的做法 |
| `wasted computation and memory` | **1** | 0 | `cf` 大的代价（原文措辞） |
| `expert overflow` | **2** | 0 | 溢出的名字 |
| `capacity factor` | **11** | **0** | ★★ **这个词是 Switch 的**（GShard 用 `O(N/E)`） |
| `fractional capacity of each expert` | 0 | **1** | GShard 用的是这个说法 |
| `Random routing` | **0** | **1** | ★ GShard 的随机路由（**同名的另一件事**，见 `01`） |
| `degenerates into a zero vector` | 0 | **1** | ★★ **glossary 那段对应的是这一条**（贡献为 0） |
| `passed on to the next layer via residual connections` | 0 | **1** | ★★ GShard 的同义表述 |
| `guarantee almost no tokens will be dropped`（剥标签后） | **1**（原始字节 **0**） | 0 | ★★ NTLB 是 `almost no`，**不是** `no` |
| `we found no empirical benefits` | **1** | 0 | ★★ NTLB 自承**没找到收益** |
| `actually slows down (840 to 790)` | **1** | 0 | X6 的速度侧（主语是 **MoE 模型**） |
| `which is unexpected` | **1** | 0 | 原文自承意外 |
| `perform better at lower capacity factors (1.0, 1.25)` | **1** | 0 | X6 的质量侧（**带语境**） |
| `Smaller expert capacities are indicative of the scenario` | **1** | 0 | 上一条的**理由**（逐字） |

★ **对照检验**（§0.3 的纪律：否定性结论必须配对照）：本表里那两个 `0` 的语义不同 ——
`capacity factor` 在 GShard 里是**真的没有这个词**（对照：`expert capacity` 在 GShard 里 **7 次**、
`at most two experts` **2 次**、`fractional capacity` **1 次**，都在）；
而 `further expanding by a capacity factor` 的 `0` 是**跨标记**造成的（同一句话的另一半计数是 1）。

**外部来源 / 论文**

- **Switch Transformer**：arXiv 2101.03961（★ abs 页 `comments` 逐字为 `JMLR`
  ⇒ **已确认同行评审**；抓取 `https://ar5iv.labs.arxiv.org/html/2101.03961`，2026-09-22，
  **413955 B**，`sha256[:16] = e9cec95e4c274436` —— 与计划 §C.1 逐位一致）。**[C]**
  - 定义式 (3)、`cf > 1` 的语义、dropping 与 NTLB（附录 B）、aux loss（§2.2）、
    Table 1 的 finding (3) 与 §2.3 的 `840 to 790`。
  - ⚠ 定义式的**公式本身**（`expert capacity = (tokens per batch / number of experts) ×
    capacity factor`）在原始字节上**不是一个可检索的字面量**（`expert capacity=` 计数 **0**，
    被 MathML 切开）⇒ **引用它只能引散文那一句**（计数 1，见上表）。
- **GShard**：arXiv 2006.16668（**无 venue 线索 ⇒ 未确认同行评审**；
  2026-09-22，**451967 B**，`sha256[:16] = 1a5e04e9d1604e8c` —— 与计划 §C.1 逐位一致）。**[C]**
  - `expert capacity` 的**原始概念**（`O(N/E)`、`fractional capacity`、`G_{s,E}` 退化成零向量）。
- **Expert Choice**：arXiv 2202.09368（abs 页**无 `Comments:` 也无 `Journal reference:`
  ⇒ **未确认同行评审**；2026-09-22，**175642 B**，`sha256[:16] = 059b2abad496bcb1`）。**[C]**
  - ★★ 式 (1) `k = n × c / e` 与「`c` denotes on average how many experts are utilized by a
    token」⇒ §1.5 的**同名不同指**。
  - ⚠★ **这一篇不在计划 §C 的论文清单里**（§C.7 逐字称「仍待补：**无**」）—— 本域补取，
    并已回报控制者。

**本机实测值**

- `code/02_capacity_factor.py` 的 `make run` 输出：`python3 3.13.5` + `numpy 2.4.4`，
  macOS arm64，**无 GPU**。口径两套：**统计段** `T=4096, E=64, k=2`，`logits ~ N(0,1)`，
  `permutation` 选出的 8 个热门专家加 `+1.2`（**逐字照抄探测脚本 Q7**，即计划 §0.7-D 的 D16 口径）；
  **协议段** `T=64, E=4, k=2, d=16`。**全部是推导值**，不是真实模型的测量。
- ★★ **口径注记（一条必须回报的偏差）**：计划 §0.7-D 的 **D16** 逐字给
  `cf=1.0 → 49.62%`，并注明该口径「会**显著抬高**丢弃率（相对「每专家一个容量、
  所有来源合并计算」的常见语义）」。**本域复现不出这个方向**：
  - 照抄 Q7 的**全部**抽取顺序（含 `permutation`）之后，本程序得
    `48.93 / 45.80 / 42.68 / 36.43%`；D16 的 `49.62%` 与它不同，是因为**探测脚本的 `rng`
    在 Q7 之前已被 Q1–Q6 消耗过** ⇒ **D16 的四个数绑定在探测脚本的状态上**，
    不是 `default_rng(seed)` 的属性；
  - 而甲（每 `(j,e)` 各自 `cap`）与乙（合并、容量乘 `k`）在本抽样上**同值**（`48.93%`），
    丙（合并、容量**不**乘 `k`）**更高**（`55.49%`）。⇒ **甲并不比「常见语义」高**，
    丙才高。**三种口径的四个数与定义已全部写进 §3.3。**
- 计划 §3 Chunk 3 表 A 的 ②③④ 三条判据在本程序里的位置：`A.1`（逐位等于 `ceil`）、
  `A.2`（取整口径的理由）、`B.1`（dropping 的残差恒等）、`B.2`（严格单调）。

**上游契约（★ 本文是这 8 处的落点）**

| 位置 | 状态 |
|---|---|
| `00-meta/glossary.md:240` | 定义正文在本域 ⇒ **本文 §1.1**；它描述的「前向输出恰为 0」对应 **dropping 下的贡献口径** ⇒ **本文 §3.2**（并列，不改它） |
| `40-kernels/moe-kernels/03-permute-and-unpermute.md:47` | 定义正文在本域 ⇒ **本文 §1.1** |
| `40-kernels/moe-kernels/03:197` | 「要不要重新归一化」⇒ **本文 §1.4** |
| `40-kernels/moe-kernels/03:238` | ⚠ **不是让渡句**（该行不含「正文在本域」的括注），而是被引用的一句：「做 mask 之后输出并不是 `X[t]` 本身：它是 `X[t] × 幸存权重之和`」⇒ **本文 §1.4**（同样的内容） |
| `40-kernels/moe-kernels/01-moe-compute-pattern.md:270` | 「槽位数 C 该取多少」⇒ **本文 §1.6**（+ `03` 的 `C.4`） |
| `40-kernels/moe-kernels/02-grouped-gemm.md:198` | 结构设计空间 ⇒ **本文 §1.6** 与 `03` 的粒度一节 |
| `40-kernels/moe-kernels/code/verify.py:887` | 逐字 `# 容量截断后的丢弃率（slot 数 C 的语义见 80-model-architectures/moe/02-capacity-factor.md）` ⇒ **本文 §1.1** |
| `40-kernels/moe-kernels/code/verify.py:1170` | 「**容量因子的定义与语义**不在本域——正文只讲它对 kernel 的影响，定义回链 `80-model-architectures/moe/02-capacity-factor.md`」——★ **该字符串跨 `:1170-1171` 两行**，引用句落在 `:1171` ⇒ **本文 §1.1** |
| `40-kernels/moe-kernels/code/03_permute_unpermute.cu:341` | 逐字 `要不要把它重新归一化是架构层的选择（见 80-model-architectures/moe/02-capacity-factor.md）` ⇒ **本文 §1.4** |

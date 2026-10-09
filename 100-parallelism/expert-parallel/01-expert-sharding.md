---
title: 专家怎么分到设备：每设备 E/EP 个，以及两次 all-to-all 的位置与语义
slug: expert-parallel-expert-sharding
domain: 100-parallelism
topic: expert-parallel
level: 3
difficulty: intermediate
prerequisites:
  - 80-model-architectures/moe/01-gating-and-routing
  - 100-parallelism/data-parallel/01-gradient-averaging
tags: [expert-parallelism, all-to-all, moe, sharding, switch-transformer]
related:
  - 100-parallelism/expert-parallel/02-load-imbalance
  - 100-parallelism/expert-parallel/03-ep-and-other-axes
  - 80-model-architectures/moe/02-capacity-factor
code:
  - code/01_expert_sharding.py
status: stable
updated: 2026-10-08
---

# 专家怎么分到设备：每设备 E/EP 个，以及两次 all-to-all 的位置与语义

> **MoE 把 FFN 换成了 `E` 个小专家 + 一个路由器**；**EP（专家并行）就是把这 `E` 个专家
> 摊到 `EP` 台设备上**（每台 `E/EP` 个）。于是每台设备**只算一部分专家**，
> 而 token 该怎么走就多出一步：**送过去**，再**收回来**。
> ★★★ **这两步就是两次 all-to-all** —— 本文要钉死它们的**位置、语义与载荷尺寸**。
> ★★ **而本域最自然的那条 A 类判据在这里【不能用】**：token 分发是**纯索引搬运**，
> 「EP 切分 == 单卡」的 `max|diff|` **恰好是 `0.000e+00`** —— **恒真**（§4.3）。

## 0. 背景与动机

`80-model-architectures/moe/01` 讲过 MoE 层长什么样：**路由器**给每个 token 选 `top_k` 个专家，
每个专家是一个小的 FFN，输出按权重加起来。★ **本文不重写路由与 gating** ——
那是 `80-model-architectures/moe/01-gating-and-routing.md` 与
`03-balance-and-granularity.md` 的正文（此处只回链）。

**本文要回答的是「切」的那一半**：`E` 个专家怎么落到 `EP` 台设备上。

先说一件容易被忽略的事：★ **原文里 EP 不是独立的一节**。Switch Transformer 的 §5.4
起句逐字是（`specs.SWITCH_SECTION54_QUOTE`，载体 = ar5iv `2101.03961` 正文）：

> `Next we describe the partitioning strategy for expert and data parallelism.`

★★ **「expert and data parallelism」是合在一节里讲的** ⇒ 本域把 EP 单独成篇是**本库的组织选择**，
**不是**原文的结构 ⇒ **不得**据此说「原文把 EP 当独立的一维」。

## 1. 原理

### 1.1 专家怎么分到设备

记：`E` 个专家、`EP` 台设备。**本库的口径**（与域级探测 `probes/probe_100f.py` 同一口径）：

```
设备号(e) = e * EP // E          # 「均摊」口径
```

★ 切分本身**不是任何来源里的规则**，它是**本库的划分口径**（`[E]` 级）；
**可复核的是它的算术**，这正是 `code/01_expert_sharding.py` 的 A 类所钉的东西。

| 情形 | 结果 | 本机实测（`code/` 的 `A.1`/`A.2`） |
|---|---|---|
| **整除**（`E % EP == 0`） | 每设备**恰好** `E/EP` 个，一个不多一个不少 | `E=8, EP ∈ {2,4,8}` ⇒ `[4,4]` / `[2,2,2,2]` / `[1]*8` |
| **不整除**（`E % EP != 0`） | 恒等式 `E//EP*EP + E%EP == E`；**余数 `E % EP` 个怎么办**是设计问题 | `E=8, EP=3` ⇒ 每设备 2 个、**余 2 个**；`E=8, EP=5` ⇒ 每设备 1 个、**余 3 个** |

★★★ **不整除时只有两条路**（`A.1` 把它钉成可复核的形式）：

1. **让有的设备多摊一个**（「均摊」口径 `e*EP//E`）—— 它**不漏任何专家**，
   代价是**每台设备的专家数不等**（实测 `E=8, EP=3` 下是 `[3,3,2]`，极差**恰好 1**）；
2. **把某个专家【切开】跨设备放**（专家内部再切）—— ★ 那是 `03` 的 B 类。

★★ 而**「朴素分块」口径** `e // (E // EP)`（「每设备 `E//EP` 个、按块切」）
在不整除时会**把专家送到不存在的设备上**：最后 `E % EP` 个专家的返回值 `>= EP`
（实测 `E=8, EP=3` ⇒ 设备号 `[0,0,1,1,2,2,3,3]`，**越界的是第 6、7 个**）
⇒ ★★ **他们没有任何设备认领**。★ 这与 TP 的「头切分」除不尽时的问题**是同族**的。

### 1.2 ★★ 两次 all-to-all 的位置与语义

这才是 EP 的**标志性代价**。原文（Switch Transformer §5.4）**逐字**给了两次：

**第一次 —— dispatch：把 token 送到专家所在的核上。**

> `Because each core has its own expert, we do an all-to-all communication of size` …
> （`specs.SWITCH_DISPATCH_ALLTOALL_QUOTE`，载体 = ar5iv `2101.03961` 正文）

★ 注意它的宾语是一个**尺寸** `[E, C, d_model]` —— ★★ 那个方括号在原文的 `<math>` 里
⇒ **本库只取到 `of size` 为止**（这是本域抄引文的第一条纪律：**不跨标记**）。

**第二次 —— combine：把各专家算完的结果收回原 token 的位置。**

> `in the forward pass to analogusly receive the tokens from each expert located on different cores.`
> （`specs.SWITCH_COMBINE_QUOTE`）

★★★ 原文这里用的是 `analogusly` —— **拼写如此**，**必须逐字照抄**
（改成 `analogously` 会在原始字节上计到 `0`）。

★★ **两处的名字**来自**附录 F 的伪码注释**：那里 `All-to-All` 这个串在**原始字节**上
恰好出现 **2 次**（`specs.SWITCH_ALLTOALL_TERM` / `SWITCH_ALLTOALL_COUNT`），
**两次都是** `# All-to-All communication.` 那种注释的开头。
★★★ **本库只声明「这个串出现 2 次」这个【计数】**，**不**把它升格成
「原文说 EP 有两次 all-to-all」这个更强的断言 —— 后者靠 §5.4 的两句散文支撑。

| 次 | 方向 | 宾语 | 逐字依据 |
|---|---|---|---|
| **第一次**（dispatch） | token → 专家所在的设备 | 一批 **token** 的激活 | `SWITCH_DISPATCH_ALLTOALL_QUOTE` |
| **第二次**（combine） | 专家的输出 → 原 token 的位置 | **专家算完的输出** | `SWITCH_COMBINE_QUOTE` |

★★★ **一条必须写明的限定**：**all-to-all 不是 MoE 唯一的集合通信**。GShard §5.3 逐字说
（`specs.GSHARD_COLLECTIVE_QUOTE`，**同行评审未确认**）：

> `… communication operators in the MoE model are` AllReduce and AllToAll.

⇒ ★★ 说「EP 的通信 = 两次 all-to-all」时**必须限定范围**（那是 **Switch 的 EP 方案**），
**不得**说成「MoE 只有 all-to-all」。

> ★★★ **两条【写法】上的差异，不许抹平**（这是本域「同名不同指」的近亲：**同词不同形**）：
> · **Switch** 用 **`all-to-all` / `All-to-All`**（带连字符）；
> · **GShard** 用**一个词** `AllToAll`（原始字节 **15** 次；带连字符的 `all-to-all` **0** 次）。
> ★★★ **而 GShard【没有】给出次数** —— 那个 `15` 是**字符串出现次数**（散文 + 图注 + 参考文献），
> **不是**通信次数的计数 ⇒ ★ **绝不从它推出「GShard 做了几次 all-to-all」**。
> ★★ 同理，**Switch 的 `2` 也只是「那个串出现 2 次」**（★ 见上）。

### 1.3 ★★ 载荷尺寸（`B` 类）—— **只给尺寸，不给通信量**

★★★ **本域**只声明张量的**尺寸**：**每次** all-to-all 的载荷形状是 `[E, C, d_model]`
（`specs.EP_A2A_DIM` = `(4, 8, 16)`），本库的记账模型取「每台设备手里那个张量的大小」⇒
**每次 512 个元素、两次 1024**（`code/` 的 `B.1`）。

★★ **而「尺寸」与 `EP` 度【无关】**（`B.2` 三档 `EP` 都给 512）：
**EP 改的是「张量里哪一块属于我」**，不是张量本身的大小。

> ★★★ **一条硬边界（本域一个字都不写）**：**all-to-all 的通信量分析**
> —— 它**不在本域**。正文在 `110-communication/collectives`）与
> `20-gpu-architecture/interconnect/03-collectives-on-topology.md`；
> ★★ 而且 `40-kernels/moe-kernels/01-moe-compute-pattern.md:243` **逐字让渡了它给 `110`**
> （「**它的通信量分析不在本域**，见 `110-communication/collectives`」）。
> ⇒ 本文**只给尺寸**，**不声明任何「通信量 ∝ 什么」的结论**。

## 2. 执行映射

### 2.1 本机怎么算这件事

`code/01_expert_sharding.py` 用 `numpy` 把一层 MoE 跑成**真的数组运算**，
但 ★★ **通信被实现为一个显式的索引搬运**（`all-to-all` = 按设备归属把项加起来）：

| 角色 | 是什么 | 依据 |
|---|---|---|
| **被测** | `sharded()` —— 专家按 `device_of_expert` 摆好，两次搬运用索引表达 | `specs.py` 的三条 Switch 引文 |
| **参照** | `reference()` —— **单卡**：每台设备都持有全部专家 ⇒ **没有搬运** | 同上 |

★★★ **参照实现不调用被测实现的任何函数**（域 README 的硬要求）：
`D.8` 用 **AST 调用表**给出**源码级线索**（★ **不是证明**）——
`reference()` 里调用的函数是 `range`，**没有** `sharded` / `route` / `make_inputs`。
★ **降级措辞**：这只说明「两段代码不互相调用」，**不证明**判据有物理信息量 ——
后者由 `C.1`/`C.2` 的**行为**兜底。

### 2.2 一条必须写明的限定语

> ★★ 本机**无 GPU、无多卡、无 NCCL、无真实网络**；本程序里的「搬运」是 **numpy 索引**，
> **没有一次真实通信**。★ 强制限定语：**「不是 NCCL，也不是真机」**。
> ★★ 本程序的数字是**本机实测值**：`python3 3.13.5` / `numpy 2.4.4`（macOS arm64）。

## 3. 实现与演进

### 3.1 真实系统里这两次搬运动不动

★ **它们是 EP 的**代价**（不是副作用）**：每台设备**必须**把不属于自己的 token 送走、
再把自己的 token 的结果**收回来**。★★ 于是 **EP 的账记在「搬运的条目数」上**，
而不是（如 TP 那样）记在「每层内部的 all-reduce」上。

### 3.2 ★★★ 路由的**权重**不在本域

★★ 本程序的路由只是 `argsort` 取 `top_k`，**输出按等权 `1/top_k` 相加** ——
**不含 gating 权重**。★ **这不是一种 gating 方案的推荐**（本库不声称它符合任何实现）：
它是一种**使「切分」可独立检验**的约定，★ 与第一代探测 `probes/probe_100h.py` 的 `H3` 同族。
**gating 与 aux loss** 的正文在 `80-model-architectures/moe/01-gating-and-routing.md`、
`03-balance-and-granularity.md`（此处回链、一字不写）。

### 3.3 与相邻主题的分工

| 话题 | 在哪 |
|---|---|
| 专家的**切法**（谁拿哪几个） | **本文** |
| 两次 all-to-all 的**位置与尺寸** | **本文** |
| **负载不均**（发出量 vs 接收量） | [02-load-imbalance](02-load-imbalance.md) |
| **`EP` 的约束**、EP 与 TP/DP 的组合 | [03-ep-and-other-axes](03-ep-and-other-axes.md) |
| **all-to-all 的通信量分析** | `110-communication/collectives`（、`20-gpu-architecture/interconnect/03` |
| **kernel 侧的 permute / unpermute / grouped GEMM** | `40-kernels/moe-kernels/02,03`；`00-meta/glossary.md` 的 `grouped GEMM` 条 |
| **容量因子的定义与语义** | `80-model-architectures/moe/02-capacity-factor.md` |
| **TP 每层 `2+2` 次 all-reduce** | `tensor-parallel/03` 与 `10-foundations/performance-models/02:417`（★ **不是 `:223`**） |

## 4. 踩坑与失败模式

### 4.1 ★★★ 「专家放置与路由**错位**」

- **现象**：张量的**形状、类型、算术全都对**，程序**不报任何错**；
- **成因**：设备↔专家的映射**整体错一格**（例如专家 `e` 用了邻居的权重 / 去了邻居的设备）；
- **本机实测**（`code/` 的 `C.1`）：`(T, D, E, top_k) = (32, 8, 4, 2)`、`EP = 2` 时，
  `max|diff|` = **`2.472807`**（★ 内容级，比机器精度阈大 **12 个数量级**）；
- **规避**：★★★ **唯一的信号是「与参照的差」** —— 这就是 §1.1 的划分算术与 `B` 类存在的理由。

### 4.2 ★★★ 「只做 dispatch、不做 combine」

- **现象**：同样**不报错**；第二次 all-to-all **干脆没做**；
- **成因**：索引、形状全都合法，只是**大部分 token 的贡献根本没回来**；
- **本机实测**（`C.2`）：同一个形状、`EP = 2`，`max|diff|` = **`2.811856`** ——
  ★★ **它不等于 `C.1` 的 `2.472807`** ⇒ ★ **两条错法是【两种】不同的失败**
  （`C.2` 的断言把这一点也钉住了）。

### 4.3 ★★★ 一条**不该有**的「A 类」：把「EP 切分 == 单卡」当判据

- **现象**：`max|diff|` **恰好是 `0.000e+00`**，看起来「完美通过」；
- **成因**：★ **token 分发是【纯索引搬运】** —— 它**连一次浮点运算都不改**
  ⇒ 这条判据**恒真**（`D.4` 逐档量它：`EP ∈ {2, 4}` 两档都是 `0.000e+00`）；
- **为什么它危险**：域 README 说得直白 —— **A 类判据只在「切分改变了求值顺序」的地方才有信息量**。
  ★★ 本域把这一类归为「**形状依赖**」：同一份数据换个形状就可能非零，
  **且哪些形状给 `0` 是数据/种子依赖的**
  ⇒ ★★ **可引用的是「形状依赖」这个性质；某组具体形状给 `0` 的清单本身不可引用**。
- **规避**：★★★ 本程序**不把那个 `0` 写成判据** —— `D.4` 是一条**恒真性自检（报告）**，
  它**与 `C.1` 的 `2.472807`、`C.2` 的 `2.811856`【并列】**。
  **没有那两条对照，`D.4` 一行的信息量为零。**
  ★ A 类因此钉在**整数划分**（`A.1`/`A.2`）上。

### 4.4 ★★ 把「不整除」当成「整除」

- **现象**：`E % EP != 0` 时，某几台设备**一个专家都没有**（而代码不报错）；
- **本机实测**（`code/` 的 `A.1g` 守卫）：`E=8, EP=3` 下朴素分块的最大设备号是 **`3 >= 3`**、
  `EP=5` 下是 **`7 >= 5`**，而**整除**的对照组 `E=8, EP=4` 是 **`3 < 4`**
  ⇒ ★★ **换掉「除得尽吗」这个条件真的改变了被判量**；
- **规避**：★★ 上表那两条路（多摊一个 / 切开专家）**必须显式选一条**。

## 5. 判据表

| 可机械判定的条件 | 结论 |
|---|---|
| 报出「每设备几个专家」 | ★★ **必须同时说 `E % EP` 是否为 0** —— 不整除时它是**设计问题**，不是除法 |
| `E % EP != 0` | 恒等式 `E//EP*EP + E%EP == E` 成立；**余 `E % EP` 个专家没有设备可放**（`A.1`） |
| 见到「EP 切分 == 单卡 ⇒ 实现对了」 | ★★★ **不合格**：本组形状下它**恰为 `0.000e+00`**，**恒真**（§4.3、`D.4`） |
| 要验证 EP 的搬运对不对 | ★★ **改 C 类**：放置错位 ⇒ `2.472807`、不做 combine ⇒ `2.811856`（两条都**不报错**） |
| 报「EP 有几次 all-to-all」 | ★★ **两次**（dispatch / combine）—— 依据是 Switch §5.4 两句散文（`D.5`）**+** 附录 F 的 `All-to-All` **2 次**（`D.6`） |
| 引 GShard 说通信次数 | ★★★ **不行**：GShard 用**一个词** `AllToAll`（**15** 次是**字符串**次数）、**不给次数**（`D.9`） |
| 说「MoE 的通信就是 all-to-all」 | ★★★ **不成立**：GShard §5.3 逐字说关键算子**是 `AllReduce` 与 `AllToAll`** |
| 报「两次 all-to-all 的载荷」 | ★★ 尺寸 = `[E, C, d_model]`（**每次**）⇒ 两次 = 两倍；★★ **尺寸与 `EP` 无关**（`B.2`） |
| 由载荷尺寸推「通信量 ∝ …」 | ★★★ **越界**：通信量分析在 `110-communication/collectives`）—— `moe-kernels/01:243` 已逐字让渡给它 |
| 路由里带上 gating 权重 | ★★ **本文不带**（等权 `1/top_k`）；gating 的正文在 `80-model-architectures/moe/01,03` |

## 6. 小结与延伸

**一句话**：**MoE 把 FFN 摊成 `E` 个专家，EP 就是把这 `E` 个专家摊到 `EP` 台设备上**
（每台 `E/EP` 个，除不尽时要显式选一条路）；
★★★ **代价是两次 all-to-all**：一次把 token **送去**专家所在的设备，一次把结果**收回**原位。

**三条可独立复用的结论**：

1. ★★★ **「EP 切分 == 单卡」在本组形状下恒真**（`max|diff|` 恰为 `0.000e+00`）
   ⇒ **它不是 A 类判据**；★ 有信息量的是两条内容级错法：**`2.472807`（放置错位）
   与 `2.811856`（只做一半）**；
2. ★★ **两次 all-to-all 的尺寸都是 `[E, C, d_model]`，且与 `EP` 度无关**
   —— `EP` 改的是「哪一块属于我」；
3. ★★★ **措辞不许抹平**：Switch 写 `all-to-all`（**2 次**是伪码注释里的名字）、
   GShard 写 `AllToAll`（**一个词、不给次数**）。

**延伸**：

- ★★★ **发出量均衡、接收量不均 ⇒ EP 越大越不均** → [02-load-imbalance](02-load-imbalance.md)；
- ★★ **`EP` 的约束（只写取到过的那一条）、EP × TP / DP** →
  [03-ep-and-other-axes](03-ep-and-other-axes.md)；
- **本域不写、只回链的正文**：
  · ★★★ **容量因子的定义与语义**（定义式 / 取整口径 / dropping vs No-Token-Left-Behind /
    丢弃率口径）→ `80-model-architectures/moe/02-capacity-factor.md`（**全篇 563 行**）；
  · **aux loss 的式子 / 路由 gating 的机制** → `80-model-architectures/moe/01`、`03`；
  · **all-to-all 的通信量分析** → `110-communication/collectives`）与
    `20-gpu-architecture/interconnect/03`；★ `40-kernels/moe-kernels/01:243` 已**逐字让渡**；
  · **kernel 侧的 permute / unpermute / grouped GEMM** → `40-kernels/moe-kernels/02,03`、
    `00-meta/glossary.md` 的 `grouped GEMM` 条；
  · **TP 每层 `2+2` 次 all-reduce** → `tensor-parallel/03` 与
    `10-foundations/performance-models/02-amdahl-and-scaling.md:417`（★ **不是 `:223`**）。

## 参考

- Fedus, Zoph, Shazeer, *Switch Transformers: Scaling to Trillion Parameter Models with Simple
  and Efficient Sparsity*, arXiv:2101.03961 —— **§5.4**（两次通信的散文）与**附录 F**（伪码注释）。
  ★★ **同行评审：是** —— 依据在 **arXiv abs 页**上：该页 `Comments` 字段渲染文本逐字是 **`JMLR`**
  （抓取日 2026-10-08；同页 `Comments` **1** 次、`JMLR` **1** 次，
  而 `ICLR` / `NeurIPS` / `Accepted` / `Proceedings` / `Conference` / `MLSys` / `Journal` **各 0** 次）。
  ★ 引文载体 = `https://ar5iv.labs.arxiv.org/html/2101.03961` 正文，抓取日 **2026-10-08**，
  逐条在**原始字节**上 `count()`：`we do an all-to-all communication of size` **1** 次、
  `in the forward pass to analogusly receive …` **1** 次、`All-to-All` **2** 次、`all-to-all` **3** 次。
- Lepikhin et al., *GShard: Scaling Giant Models with Conditional Computation and Automatic
  Sharding*, arXiv:2006.16668 —— **§5.3**（MoE 的两个关键集合通信）。
  ★★★ **同行评审：本库不声称已确认** —— abs 页上 `Comments` / `JMLR` / `ICLR` / `NeurIPS` /
  `Accepted` / `Proceedings` / `Conference` / `MLSys` / `Journal` **各 0** 次（抓取日 2026-10-08）。
  ★ 引文载体 = `https://ar5iv.labs.arxiv.org/html/2006.16668` 正文；
  原始字节：`AllToAll` **15** 次、`all-to-all` **0** 次、`all to all` **0** 次。
- 本仓库 [02-load-imbalance](02-load-imbalance.md)（负载账）、
  [03-ep-and-other-axes](03-ep-and-other-axes.md)（EP 的约束与组合）、
  `80-model-architectures/moe/01-gating-and-routing.md`（路由与 gating，本文不重写）。

## 编译与运行

```sh
cd code && make run                 # 三个编号程序，任一断言失败即非零退出
python3 01_expert_sharding.py       # 只跑本文的配套程序
make 01_expert_sharding             # ★★ 与上一行等价（都先走 check-env）
```

★ 本机的运行条件：`python3 3.13.5` / `numpy 2.4.4`，macOS arm64，
**无 GPU、无多卡、无 NCCL、无真实网络**。
★★ **本主题的三个程序都要 `numpy`**（路由打分与专家前向都是矩阵乘）
⇒ ★★★ **本文件不写「单跑某个程序不需要 numpy」那种措辞** —— 它在本主题**不成立**
（同域 `sequence-context-parallel` 的 `01` 确实是纯标量算术，所以那边可以那么写；
照抄到这里就成了一条**照抄即失败的命令**）。
★ 强制限定语：**不是 NCCL，也不是真机** —— 本文的一切数字都是**切分的代数**
（划分算术 / 载荷尺寸 / 两条错法的差），**不得**支撑关于真实集群性能的结论。

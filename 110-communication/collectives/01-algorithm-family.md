---
title: 集合通信的算法族：五个算法的步数与每卡搬运量
slug: communication-collectives-algorithm-family
domain: 110-communication
topic: collectives
level: 3
difficulty: intermediate
prerequisites:
  - 10-foundations/performance-models/02-amdahl-and-scaling
tags: [collectives, allreduce, allgather, broadcast, ring-allreduce, recursive-doubling, bruck]
related:
  - 110-communication/collectives/02-stages-and-costs
  - 110-communication/collectives/03-all-to-all-and-p2p
  - 20-gpu-architecture/interconnect/03-collectives-on-topology
code:
  - code/01_algorithm_family.py
status: stable
updated: 2026-10-08
---

# 集合通信的算法族：五个算法的步数与每卡搬运量

> **同一个集合通信原语（all-reduce / all-gather / broadcast）有一族算法。**
> 它们**搬的东西一样多吗？不 —— 有的差一个数量级**。
> 本文把五个算法摆在一张表上，用**两条轴**分类：**步数**（延迟）与**每卡搬运量**（带宽）。
> ★★★ **每一条公式都来自一篇 2005 年的同行评审论文（[C] 级）**，
> 而**每一个数都由本机的一段「消息序列模拟」独立算出来**对拍。
> ★★ **本层是【算法层】** —— 不是 NCCL，也不是真机。

## 0. 背景与动机

`10-foundations/performance-models/02` 把通信代价分成 `N` 与 `log N` 两支，并**完整推导**了
ring allreduce 的每卡通信量 `V(N) = 2(N−1)/N × S` 与它的饱和（`:147-155,209`）。
**本文一行都不重推那个式子**（正文在彼，此处回链）。

> **一个知识点只有一处正文**（`CONTRIBUTING` §7）。`V(N)` 的正文在
> `10-foundations/performance-models/02-amdahl-and-scaling.md:147-155,209`（正文在彼，此处回链）。

**那本文补什么？** 补 `02` 没有列的两样东西：

1. **同一个原语有几种算法** —— 论文逐字列的 allreduce 清单是
   `binomial tree and recursive doubling … recursive halving and doubling, binary blocks, and ring.`
   （**五个**，见 `code/specs.py` 的 `MPICH_REDUCE_ALGOS`）；
2. **各自的步数与每卡搬运量** —— 以及**为什么同样是 all-reduce，搬运量能差一个 `lg p` 因子**。

**一个具体的对照场景**（本机实测，`code/` 的 `B.2`）：`p = 16`、每设备输入 `n = 8` 个元素，

| 算法 | 步数 | 每卡搬运量（元素） |
|---|---|---|
| ring allreduce | **30** | **15.0** |
| 递归倍增 allreduce | **4** | **32.0** |
| halving-doubling（Rabenseifner） | **8** | **15.0** |

★★★ **三个数之间的差别不是小数点**：ring 走 **30 步**、递归倍增只走 **4 步**，
而递归倍增的**搬运量是 ring 的两倍多**。**换算法换的是「花几步」还是「搬多少」**。

## 1. 原理

### 1.1 ★★ 两条轴：步数（延迟）与每卡搬运量（带宽）

一个集合通信算法的代价可以看成两项（论文的写法，`α` = 每步延迟、`β` = 每元素的发送成本）：

```text
T = (步数) × α  +  (每卡搬运量) × β   [+ 归约成本 γ（只有归约类原语有）]
```

★★★ **本主题全部的分类依据就是这一条**：**步数**与**搬运量**是**两条独立的轴** ——
`α` 主导时（小消息、低延迟网络）选**步数少**的；`β` 主导时（大消息、高带宽网络）选**搬运量少**的。
★ 本文只算**步数**与**搬运量**这两个**代数量**（本机可算、可对拍），
**不算** `α` / `β` 的实际值（那取决于具体网络 —— 本机没有真网络）。

### 1.2 ★★★ ring allreduce：两段式的代价

**ring allreduce 分两段**（这是 `02` 的主题，本文只用它的代价）：

```text
T_ring = 2(p−1)·α  +  2(p−1)/p · S · β
         └ 步数 ┘      └── 每卡搬运量（= V(N)）──┘
```

★ 步数 `2(p−1)`：reduce-scatter 走环 `p−1` 步、all-gather 再走环 `p−1` 步。
★★ **每卡搬运量就是 `V(N) = 2(p−1)/p × S`** —— ★★ **它的推导在上游（`performance-models/02:147-155`），
本文不重推**；本文做的是**另一个方向**的检查：用**消息序列模拟**枚举出来的搬运量去与它**对拍**。

★★ 对照：**Bruck 那篇算法的归属**。★★★ **`Bruck` 不是 allreduce 的算法**，
它是 **all-gather 与 all-to-all（index）** 的算法（见 `code/specs.py` 的 `MPICH_BRUCK_SCOPE_NOTE`）
—— 把它当成 allreduce 算法会**写错正文**（allreduce 里它只作为**组合件**出现）。

### 1.3 ★★★ all-gather：三种算法，**搬运量相同、步数不同**

论文逐字给了三个 all-gather 算法（`code/specs.py` 的八条引文）：

| 算法 | 步数（逐字） | 每卡搬运量（逐字） |
|---|---|---|
| **ring allgather** | `p − 1`（`the entire algorithm takes p − 1 steps.`） | `(p−1)/p · n` |
| **递归倍增** | `lg p`（`all processes get all the data in lg p steps.` ★ **无取整符号**） | `(p−1)/p · n` |
| **Bruck** | `⌈lg p⌉`（`Both algorithms take ⌈lg p⌉ steps in all cases`） | `(p−1)/p · n` |

★★★ **三条带宽项逐字相同** —— 论文把三者的 `nβ` 项都写成 `(p−1)/p · nβ`。
⇒ **all-gather 这一族里，「选算法」选的是【步数】，搬运量是**同一**个。**
★★ 本机实测（`code/` 的 `A.3`，`p = 16`、`n = 32`）：ring 走 **15 步**、递归倍增走 **4 步**、
**两者的每卡搬运量都是 30**。★ **步数与搬运量是两条独立的轴**，此处是它最干净的实例。

★★ **Bruck 相对递归倍增的存在理由**：递归倍增用 `d XOR 2^k` 配对，
**要求 `p` 是 2 的幂**；Bruck 用**旋转**配对，**任意 `p`** 都行，而步数仍是 `⌈lg p⌉`
（论文逐字 `in all cases`）。★ 本机在 `p = 6`（**非 2 的幂**）上验证了这一点（`A.4`）——
★ `⌈lg 6⌉ = 3` 而 `⌊lg 6⌋ = 2`，**两个取整符号正好在这一档上分得开**。

### 1.4 ★★ broadcast 与 reduce：树形，`⌈lg p⌉` 步

论文逐字：广播的二项树 `This communication takes a total of ⌈lg p⌉ steps.`、
`Ttree = ⌈lg p⌉(α + nβ).`。★ 注意它的**带宽项是 `nβ`（整条消息一次）**，
不是 `(p−1)/p · nβ` —— 因为**每台设备最多发一次**（树上的每条边只走一遍），
**总消息数恰好 `p − 1` 条**（本机在 `p ∈ {2,4,8,16}` 上枚举验证，`code/` 的 `A.5`）。

★★★ **一条【同名不同指】**：论文里 **`Trec dbl` 这个名字出现在两处** ——
all-gather 的递归倍增是 `Trec dbl = lg p α + (p−1)/p · nβ`，
而 **allreduce** 的递归倍增是 `Trec−dbl = lg p α + n lg p β + n lg p γ`。
★★ **两者的带宽项差一个 `lg p` 因子** —— **它们不是同一算法的两种写法，是两个不同的算法**。
（★ `code/specs.py` 用两个常量名分开存放它们，并在注释里互指。）

### 1.5 ★★★ halving-doubling（Rabenseifner）：两头都占

论文逐字：
`Trabenseifner = 2 lg p α + 2(p−1)/p · nβ + (p−1)/p · nγ.`

★★★ 逐项读它：

- 步数项 `2 lg p` —— **对数级**（对比 ring 的 `2(p−1)`，**线性级**）；
- 带宽项 `2(p−1)/p · nβ` —— ★★ **与 ring allreduce 的每卡搬运量逐字同形**；
- 归约项 `(p−1)/p · nγ` —— 这是 ring 那条式子里**没有**的一笔（`γ` = 每元素的归约成本）。

⇒ ★★★ **它同时拿到了 ring 的搬运量与对数级的步数**。这就是「最优」在本主题的**算术含义**：

| 算法 | 步数 | 每卡搬运量 |
|---|---|---|
| ring allreduce | `2(p−1)` ← 线性 | `2(p−1)/p × S` ← **最小** |
| 递归倍增 allreduce | `lg p` ← **最小** | `n lg p` ← **最大** |
| **halving-doubling** | `2 lg p` ← **对数级** | `2(p−1)/p × S` ← **最小** |

★★ **论文自述**它把带宽项从 `n lg p β` 降到 `2nβ` —— ★ 用上表的两条式子**可复核**：
`2(p−1)/p ≈ 2` 而 `lg p` 随 `p` 增长（`p = 16` 时 `1.875` 对 `4`，**少了一半以上**）。

### 1.6 ★★ `2(N−1)/N × S` **为何是两段之和**

★★★ **这一条是本文与 `02` 的接口，也是本文不重推的东西**：

```text
V(N) = (N−1)/N × S   +   (N−1)/N × S
       └ reduce-scatter ┘   └── all-gather ──┘
```

★ 「为什么是 `2×`」的**结构解释**是：**它就是两段相加**。
★★ **推导在上游**（`performance-models/02:147-155`），**本文不重推**；
★★★ `02-stages-and-costs` 做的是**另一件事**：用**消息序列模拟**把两段**各自**枚举出来，
证明**两段之和**与闭式**逐位一致**（`02` 的 `A.2`）。

## 2. 执行映射

### 2.1 本机怎么算这件事

`code/01_algorithm_family.py` 的参照是**真消息序列模拟**：每一步都记录
`MSG(step, phase, sender, receiver, block, n_elem)` —— ★★★ **逐步枚举「哪一步、谁发给谁、
哪个块、多少元素」**，并**驱动真实数据**（`numpy` 整数数组按块搬运、按块累加）。

| 角色 | 是什么 | 它为什么独立 |
|---|---|---|
| **参照** | 五个算法各一个**消息序列模拟器** | 它**不调用**任何闭式函数（`D.8` 用 AST 调用表核） |
| **被测** | `cf_*`（步数与搬运量的**闭式**） | 两条式子的载体是 `specs.py` 的 [C] 级逐字引文 |
| **第三方** | **单进程参照** `x.sum(axis=0)` | 与模拟器**不共享代码** ⇒ 判「结果对不对」用它 |

★★★ **本程序最重要的一条设计纪律（也是上游规划自己踩过的坑）**：
**不许把闭式用循环写一遍**。规划 `§D` 的 `D2` 写过一个 `ring_bytes()`，
正文是 `for _ in range(N-1): sent += PER` —— ★★ **那就是把闭式抄成了循环**，
「相对差 `0.0e+00`」是**恒真**的（`P12`：模拟器同时是被测对象）。
⇒ ★ 本程序的模拟器**枚举**消息、**搬运真实数据**，闭式是**另一段代码**。

### 2.2 ★★★ 一条必须写明的限定语

> ★★ 本机**无 GPU、无多卡、无 NCCL、无真实网络**；本程序里的「搬运」是 **numpy 索引**，
> **没有一次真实通信**。★ 强制限定语：**「不是 NCCL，也不是真机」**。
> ★★ 本程序的数字是**本机实测值**：`python3 3.13.5` / `numpy 2.4.4`（macOS arm64）。

## 3. 实现与演进

### 3.1 为什么每一步都需要下一步

| 阶段 | 算法 | 它解决了什么 | 它留下的问题 |
|---|---|---|---|
| 朴素 | 全部设备两两互发 | 概念最直 | 步数 `O(p)` 或 `O(p²)` |
| **ring allreduce** | 走环两段 | 每卡搬运量**饱和**到 `2S`（与 `p` 无关） | 步数 `2(p−1)` **线性** |
| **递归倍增** | 超立方配对 | 步数降到 `lg p` | 搬运量涨到 `n lg p` |
| **halving-doubling** | 前半跑 halving、后半跑 doubling | **同时**拿到对数级步数与最小搬运量 | 实现更复杂（两段配对方式不同） |

★★ **这不是「越新越好」的清单，而是「两条轴上的取舍」**：
★ 小消息（`α` 主导）⇒ 步数少的（递归倍增 / halving-doubling）；
★ 大消息（`β` 主导）⇒ 搬运量少的（ring / halving-doubling）。
★★ **halving-doubling 在两条轴上都不差**，这是它成为 MPICH 大消息默认选择的原因。

### 3.2 ★★ 「所有集合通信都靠 allreduce」是错的

★ 本文的四个原语（**reduce-scatter / all-gather / broadcast / reduce**）都是独立的算法族，
而 **allreduce 恰好可以拆成前两个**（`02` 的主题）。★ 下文 `03` 会给出另一种结构完全不同的原语
（**all-to-all**，它的每卡搬运量 `∝ p`，**不饱和**）。

### 3.3 与相邻主题的分工

| 话题 | 在哪 |
|---|---|
| **算法族的步数与每卡搬运量** | **本文** |
| **两段式的通信阶段划分与四个原语的代价** | [02-stages-and-costs](02-stages-and-costs.md) |
| **all-to-all 的算法与总搬运量口径 · P2P 原语** | [03-all-to-all-and-p2p](03-all-to-all-and-p2p.md) |
| ★★★ **`V(N)` 的推导与饱和** | `10-foundations/performance-models/02-amdahl-and-scaling.md:147-155,209` |
| **bisection 下界 · all-to-all 跨切口量 `N·V/4` · 分层三段式** | `20-gpu-architecture/interconnect/03-collectives-on-topology.md` |
| **`algbw` vs `busbw` 与修正系数表** | `interconnect/04-measuring-bandwidth.md` |
| **NCCL 选哪个算法**（实现层） | `110-communication/nccl`（本域另一主题） |

## 4. 踩坑与失败模式

### 4.1 ★★★ 环形块索引写错（发哪块）

- **现象**：块数、字节数、形状**全都对**，程序**一次异常都不抛**，只是**结果错**；
- **成因**：把「第 `k` 步发 `(d−k) % N` 块」写成 `(d+k) % N` —— **整个块到设备的映射被打乱**；
- **本机实测**（`code/` 的 `C.1`，`N = 4`、`S = 64`）：`max|diff|` = **525**（**整数**差，不是舍入）；
  ★★ **而搬运量 `96` 对 `96`，一个元素都没变**；
- **规避**：★★★ **唯一的信号是「与单进程参照的差」** ——
  这就是本文的 `A.1`（结果逐位相同）存在的理由。

### 4.2 ★★★ 累加到错误的目标

- **现象**：同样**不报错**；张量形状与字节数合法；
- **成因**：收到的贡献被加到**另一块**上（`(块+1) % N`）——
  累加的目标是一块**存在的**块，只是**不是该累加的那一块**；
- **本机实测**（`C.2`）：`max|diff|` = **477** —— ★★ **它不等于 `C.1` 的 `525`**
  ⇒ **两条错法是两种不同的失败**（`C.2` 的断言把这一点也钉住了）；
- **规避**：★ 这也是 `D1b` 那个现场（把缓冲区写成「只有自己那一块」⇒ 累加目标越界），
  ★★ **报错信息指向形状，不指向索引**。

### 4.3 ★★★ **搬运量对「块索引写错」是不变量**（本文自己发现的）

- **现象**：把块索引 `±` 反过来，**搬运量与步数一点不变**，只有**结果**变了；
- **成因**：`±` 只是把「哪一块发给谁」**重排**，块数、每块大小、每段步数**都没变**；
- **本机实测**：`code/` 的 `A.1g` 守卫直接量它 —— 错法版**结果变了**、而**搬运量相同**
  （`96 == 96`）。★ **外部守卫实验**（把 `ring_send_block` 的 `−` 改成 `+`）也复核过这一点：
  `A.1`（正确性）**失败**、而 `A.2`（整除/结构）**照样通过**；
- **规避**：★★★ **「搬运量 == 闭式」这一条过不了 `CONTRIBUTING:235` 的守卫** ——
  **过守卫的是「结果与单进程参照逐位相同」那条**。★ **检查一条判据，要问它变异什么**。

### 4.4 ★★ 把取整符号抄错（引文层）

- **现象**：`⌈lg p⌉` 被抄成 `lg p`（或反之），**不会有任何报错**，只会让公式悄悄错掉；
- **成因**：论文 PDF 用 `pdftotext -layout` 提取时，`⌈lg p⌉` 变成 `dlg pe`、
  `⌊lg p⌋` 变成 `blg pc`、而**无取整**才是**裸 `lg p`** —— ★★ `grep "lg p"` **会同时命中三种**；
- **规避**：★★★ **凡从 PDF 抄公式，先判读那个符号属于哪一态**；
  `code/specs.py` 每一条的 `# 来源：` 里都写了提取形态。

## 5. 判据表

| 可机械判定的条件 | 结论 |
|---|---|
| 要比较两个 all-gather 算法 | ★★ **先比步数、再比搬运量** —— 两者是**独立的轴**（`A.3`） |
| 报「allgather 三种算法的搬运量」 | ★★ **三者相同**（都是 `(p−1)/p × n`）⇒ 差别**只在步数** |
| 报「ring allreduce 的搬运量」 | ★★ `2(N−1)/N × S`；★★ **推导在上游**（`performance-models/02:147-155`），此处只回链（`A.1` 对拍） |
| 说「所以 N 越大越划算」 | ★★★ **不能从搬运量推** —— `V(N) → 2S` **饱和**，代价转向步数 `2(p−1)` 的延迟（`B.1`/`B.2`） |
| 见到「`Bruck` 是 allreduce 的算法」 | ★★★ **不成立**：它是 **all-gather 与 all-to-all(index)** 的算法（`D.6`） |
| 见到「`Trec dbl` 的带宽项」 | ★★★ **先问是哪一条** —— all-gather 的是 `(p−1)/p·n`、allreduce 的是 `n lg p`，**差一个 `lg p`** |
| 报「二项树广播的搬运量」 | ★★ **总量 `(p−1) × n`**（每设备**最多发一次**），**不是** `(p−1)/p · n`（`A.5`） |
| 报「Rabenseifner 的步数」 | ★★ `2 lg p`（**裸 `lg p`，无取整**）；搬运量 `2(p−1)/p × n`（**与 ring 同**，`B.2`） |
| 说「把块索引写错不会出问题」 | ★★★ **它会**：结果错而**不报错**（`C.1`/`C.2`）；★★ **而搬运量检查看不出来**（§4.3） |
| 引 MPICH 的分档阈值 | ★★★ **必须带限定语** —— 它们是 **2005 年 MPICH 在 Myrinet / IBM SP 上的实现选择** |

## 6. 小结与延伸

**一句话**：**集合通信原语有一族算法，选算法 = 在「步数」与「每卡搬运量」两条轴上取舍** ——
all-gather 的三种算法**搬运量相同、步数从 `p−1` 降到 `⌈lg p⌉`**；
allreduce 的 halving-doubling **同时**拿到 ring 的搬运量与对数级的步数。

**三条可独立复用的结论**：

1. ★★★ **`⌈lg p⌉` 与 `lg p` 是两个不同的档** —— 论文里**三种取整形态并存**
   （`dlg pe` / `blg pc` / 裸 `lg p`），★ 误判**不会有任何报错**；
2. ★★ **「搬运量 == 闭式」这条判据对块索引不敏感**（§4.3 实测）——
   ★ **有信息量的是「结果与单进程参照逐位相同」**；
3. ★★★ **`Bruck` 是 all-gather 与 all-to-all(index) 的算法**，
   **`Trec dbl` 在论文里指两个不同的算法** —— 两处都是**同名不同指**。

**延伸**：

- ★★★ **两段式（reduce-scatter + all-gather）的通信阶段划分** →
  [02-stages-and-costs](02-stages-and-costs.md)；
- ★★ **all-to-all 的算法与总搬运量口径 · P2P 原语** →
  [03-all-to-all-and-p2p](03-all-to-all-and-p2p.md)；
- **本层不写、只回链的正文**：
  · ★★★ **`V(N)` 的推导与饱和** → `10-foundations/performance-models/02-amdahl-and-scaling.md:147-155,209`；
  · **bisection 下界 · 跨切口量 · 分层三段式** → `20-gpu-architecture/interconnect/03-collectives-on-topology.md`；
  · **测量口径** → `interconnect/04-measuring-bandwidth.md`；**带宽口径** → `interconnect/01-bandwidth-conventions.md`。

## 参考

- Thakur, Rabenseifner, Gropp, *Optimization of Collective Communication Operations in MPICH*,
  **IJHPCA** 19(1):49–66, 2005 —— **[C] 级，同行评审：是**
  （Crossref `10.1177/1094342005051521`）。一手 PDF：
  `mcs.anl.gov/~thakur/papers/ijhpca-coll.pdf`（`1100025` 字节），抓取日 **2026-10-08**。
  ★★★ **读法纪律**：`pdftotext -layout` 把 `⌈lg p⌉` 提取成 **`dlg pe`**、
  `⌊lg p⌋` 提取成 **`blg pc`**、**裸 `lg p`** 表示**无取整** —— ★ **误判不会有任何报错**。
  ★ 本文用到的逐字式子（八条）全部逐字注册在 `code/specs.py` 的 `# 来源：` 里。
- 本仓库 `10-foundations/performance-models/02-amdahl-and-scaling.md`（`V(N)` 的推导与饱和，本文不重推）、
  [02-stages-and-costs](02-stages-and-costs.md)、[03-all-to-all-and-p2p](03-all-to-all-and-p2p.md)、
  `20-gpu-architecture/interconnect/03-collectives-on-topology.md`（拓扑受限的集合通信）。

## 编译与运行

```sh
cd code && make run                 # 三个编号程序，任一断言失败即非零退出
python3 01_algorithm_family.py      # 只跑本文的配套程序
make 01_algorithm_family            # ★★ 与上一行等价（都先走 check-env）
```

★ 本机的运行条件：`python3 3.13.5` / `numpy 2.4.4`，macOS arm64，
**无 GPU、无多卡、无 NCCL、无真实网络**。
★★ **本主题的三个程序都要 `numpy`**（块是 `numpy` 整数数组）
⇒ ★★★ **本文件不写「单跑某个程序不需要 numpy」那种措辞** —— 它在本主题**不成立**。
★ 强制限定语：**不是 NCCL，也不是真机** —— 本文的一切数字都是**算法的代数**
（步数 / 每卡搬运量 / 块索引），**不得**支撑关于真实集群性能的结论。

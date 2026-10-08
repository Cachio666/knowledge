---
title: 两段式：reduce-scatter 与 all-gather 的通信阶段划分，以及四个原语的代价
slug: communication-collectives-stages-and-costs
domain: 110-communication
topic: collectives
level: 3
difficulty: intermediate
prerequisites:
  - 110-communication/collectives/01-algorithm-family
  - 10-foundations/performance-models/02-amdahl-and-scaling
tags: [collectives, allreduce, reduce-scatter, all-gather, broadcast, reduce, two-stage]
related:
  - 110-communication/collectives/03-all-to-all-and-p2p
  - 20-gpu-architecture/interconnect/03-collectives-on-topology
code:
  - code/02_stages_and_costs.py
status: stable
updated: 2026-10-08
---

# 两段式：reduce-scatter 与 all-gather 的通信阶段划分，以及四个原语的代价

> **all-reduce 不是一个原子操作 —— 它是两个操作的复合**：
> 先 **reduce-scatter**（每台设备只留下一块的完整和），再 **all-gather**（把块拼回来）。
> ★★★ **为什么每卡搬运量是 `2(N−1)/N × S`？因为它是【两段之和】。**
> 本文把两段**各自**模拟出来、再**接起来**，与**不分段的整体**和**单进程参照**三方对拍。
> ★★ **本层是【算法层】** —— 不是 NCCL，也不是真机。

## 0. 背景与动机

### 0.1 ★★★ 一篇名消歧：本文说的「阶段」是**通信阶段**

`stage` / 「阶段」在库内**已有四个含义**（TVM 计算阶段 · cutlass 流水级数 ·
流水线并行的 stage · ZeRO 阶段）。
★★★ **本文里的它一律指【通信阶段】，且只指两个**：**reduce-scatter 那一段**与 **all-gather 那一段**。
★ 代码里的字段名叫 `phase`（**不叫 `stage`**）；正文里写「**通信阶段**」或
「**reduce-scatter / all-gather 两段**」，**不裸写 `stage`**。

### 0.2 为什么要把 all-reduce 拆开看

`01-algorithm-family` 给了三种 all-reduce 算法的**步数与每卡搬运量**。
但那只解释了「**总量是多少**」，没解释「**这个总量是怎么来的**」。

**一个具体的对照场景**（本机实测，`code/` 的 `A.1`）：`N = 4`、每设备输入 `n = 32` 个元素，

| 路径 | 搬了什么 | 每卡搬运量 |
|---|---|---|
| ① **分段**：reduce-scatter + all-gather | 先把 `n` 分成 `4` 块、各归约各的；再拼回来 | `24 + 24 = 48` |
| ② **不分段**：ring allreduce 整体 | 同上，但走环 | `48` |
| ③ **单进程参照** | 什么都不搬，直接加 | — |

★★★ **三条路径必须给同一个结果** —— 这就是「**分段 == 不分段**」。
★ 而 ① 让「`2×` 从哪来」**变得可见**：**两段各 `24`**。

## 1. 原理

### 1.1 两段各自是什么

**第一段：reduce-scatter。** 把 `n` 个元素分成 `p` 块（每台设备一块）。
设备 `d` 手里初始持有**全部 `p` 块**（那是它自己的一份输入）。
在这一段里，每块被**归约成和**并**落到一台设备**上：走完之后，设备 `d` 只剩**一块** ——
**`d` 号块的完整和**。

**第二段：all-gather。** 每台设备手里只有一块，把 `p − 1` 台别的设备的块**收回来**，
拼成完整的 `n` 个元素。走完之后，**每台设备都有全部**。

★★★ **两段是同一个量的两个方向**：第一段是「**把别人的份额收进来叠加**」，
第二段是「**把自己那块完整和发出去**」。★ 所以**两段的每卡搬运量相等**（`02` 的 `A.2` 量它）。

### 1.2 ★★★ 四个原语的最优算法与代价

论文（`code/specs.py`，[C] 级）与本文的模拟给出：

| 原语 | 最优算法 | 步数 | 每卡搬运量 |
|---|---|---|---|
| **reduce-scatter** | recursive halving | `lg p` | `(p−1)/p × n` |
| **all-gather** | recursive doubling / Bruck | `lg p` / `⌈lg p⌉` | `(p−1)/p × n` |
| **broadcast** | 二项树 | `⌈lg p⌉` | 每设备**最多** `n`（**总** `(p−1)×n`） |
| **reduce** | 二项树（**broadcast 那棵树的反向遍历**） | `⌈lg p⌉` | 每设备**最多** `n`（**总** `(p−1)×n`） |
| **allreduce** | **= 上面前两个之复合** | `2 lg p` | `2(p−1)/p × n` |

★★★ **四条可复核的结构事实**：

1. **四个原语的最优步数都是对数级**（`lg p` 或 `⌈lg p⌉`）—— ★ 这一点由**消息序列枚举**验证
   （`02` 的 `A.3` 从枚举出来的消息里取 `max(step) + 1`，**不是照闭式写死的循环上界**）；
2. **`reduce` 与 `broadcast` 是同一棵树的两种遍历方向**（边都是 `(r, r + 2^k)`），
   ⇒ ★★ **它们的总搬运量相等**（都是 `(p−1) × n`，`A.4` 枚举验证）；
3. **`reduce-scatter` 与 `all-gather` 的每卡搬运量相等**（`A.2`）；
4. **`allreduce` 的每卡搬运量恒等于两段之和**（`B.1`）。

### 1.3 ★★★ `2(N−1)/N × S` 为何是两段之和

```text
每卡搬运量(allreduce) = (N−1)/N × S   +   (N−1)/N × S
                        └ reduce-scatter ┘   └── all-gather ──┘
                      = 2(N−1)/N × S
```

★★★ **推导在 `10-foundations/performance-models/02-amdahl-and-scaling.md:147-155`** ——
（正文在彼，此处回链）。**本文不重推它**，做的是**结构验证**：

- 把两段**各自**模拟出来，从枚举出来的消息清单**逐条求和**得到每段的搬运量；
- 两段之和与闭式 `2(N−1)/N × S` **逐位一致**（`code/` 的 `A.2`/`B.2`）。

★★ 论文的 Rabenseifner 式子（[C] 级逐字）把这件事写成了代数：
`Trabenseifner = 2 lg p α + 2(p−1)/p · nβ + (p−1)/p · nγ.`
★★ **注意那个 `2`**：它乘的正是**单段**的搬运量 —— ★ **闭式里的「`2×`」就是「两段」**。

### 1.4 ★★★ 不整除时：一条**分情况**的代价式

★★ 上面所有式子都**默认 `p` 整除 `n`**（每块的元素数是整数）。**不整除时**论文给了一条分情况的式：

```text
Trec half = (⌊lg p⌋ + 2)α + 2nβ + n(1 + (p−1)/p)γ.
```

★★★ **三点必须一起说**：

1. 这里用的是 **`⌊lg p⌋`（下取整）** —— ★ 与 `⌈lg p⌉`（`dlg pe`）和**裸 `lg p`** 三足鼎立
   （★ 这是本主题引文读法纪律的现场，见 [01](01-algorithm-family.md) 的 §4.4）；
2. ★★★ **它自带一句限定语**：`This cost is approximate because some imbalance exists in the
   amount of work each process does` —— ★ **引式子就必须并置它的限定语**，
   否则是把一条**近似式**当成精确式；
3. ★★ **它的存在本身是一条知识**：**「不整除」不是边界情况**（本库 `P25`）——
   论文为它**单列了一条式子**。

## 2. 执行映射

### 2.1 本机怎么算这件事

`code/02_stages_and_costs.py` 有**三套独立的实现**，都驱动真实数据：

| 角色 | 是什么 | 独立性 |
|---|---|---|
| **第一段参照** | `sim_reduce_scatter`（recursive halving） | ★ 不调用任何闭式函数（`D.8` 用 AST 调用表核） |
| **第二段参照** | `sim_all_gather`（recursive doubling） | ★ 同上 |
| **第三套（不分段）** | `sim_ring_allreduce`（走环） | ★ **换了一种配对方式**，是**另一条路径** |
| **单进程参照** | `ref_sum` = `x.sum(axis=0)` | ★ 调用表是**空集** ⇒ 不依赖任何模拟器 |

★★★ **三段步数都是「涌现」的，不是写死的**：两段的循环条件是
「**还有设备手里多于一块**」/「**还有设备没拿全**」——
★ **不是** `for k in range(lg p)`（那又是把闭式抄一遍）。

### 2.2 ★★ 一条必须写明的限定语

> ★★ 本机**无 GPU、无多卡、无 NCCL、无真实网络**；「搬运」是 **numpy 索引**。
> ★ 强制限定语：**「不是 NCCL，也不是真机」**。
> ★★ 本程序的数字是**本机实测值**：`python3 3.13.5` / `numpy 2.4.4`（macOS arm64）。

## 3. 实现与演进

### 3.1 为什么是这两段，而不是别的拆法

| 拆法 | 第一段留下什么 | 第二段做什么 | 结果 |
|---|---|---|---|
| **直接 all-to-all** | 每台设备收到 `p−1` 份**原始**数据 | 自己加 | ★ 每卡搬运量 `∝ p`，**不饱和**（见 [03](03-all-to-all-and-p2p.md)） |
| **reduce + broadcast** | 一台设备收到**全部和** | 一个人发给所有人 | ★★ 根设备**成了瓶颈**（它的搬运量是 `(p−1)×n/…`） |
| **reduce-scatter + all-gather** | 每台设备各有**一块的完整和** | 各自把块发出去 | ★★★ **负载均匀**：每卡搬运量都是 `(p−1)/p × n` |

★★ **这就是两段式的价值**：它把「`p` 份数据」平摊到「`p` 台设备」，
**没有任何一台设备的搬运量随 `p` 线性增长**（除步数）。★ 这是 `V(N)` **饱和**的结构原因。

### 3.2 ★★ 四个原语之间的**组合关系**

★ 本主题的四个原语不是并列的四个名字，而是**有方向的组合**：

```text
broadcast ──(反向遍历同一棵树)──▶ reduce          # ★ 同构：边一样、方向相反（A.4）
    │                                   │
    │ Van de Geijn 广播 =               │  两段式：
    │   scatter + allgather            ▼
    └──────────────▶  allgather  +  reduce-scatter  =  all-reduce
```

★★★ **两条可复核的结构事实**：

1. **`reduce` 与 `broadcast` 是同一棵树的两种遍历方向**（边都是 `(r, r + 2^k)`）
   ⇒ ★ **总搬运量相等**（都是 `(p−1) × n`，`A.4` 枚举验证）；
2. **Van de Geijn 广播本身就是两段式**（`scatter` + `allgather`）——
   ★★★ **这说明「分两段看」不是 all-reduce 独有的结构**，
   论文用散文写它（`the message to be broadcast is first divided up and scattered …
   The scattered data is then collected back to all processes`）。

★★ **一条推论**：既然 `all-to-all` 也是**一段**（见 [03](03-all-to-all-and-p2p.md)），
那么本主题的五个原语在「**段数**」这一维上分成两类：
**一段的**（`broadcast` / `reduce` / `all-gather` / `reduce-scatter` / `all-to-all`）
与 **两段的**（`all-reduce` = 前两个之复合；Van de Geijn 的 `broadcast` 也是两段）。

### 3.3 ★★ 与相邻主题的分工

| 话题 | 在哪 |
|---|---|
| **两段式的通信阶段划分 · 四个原语的代价** | **本文** |
| **各算法的步数与每卡搬运量（含三算法对照）** | [01-algorithm-family](01-algorithm-family.md) |
| **all-to-all 的总搬运量口径（它不饱和）· P2P 原语** | [03-all-to-all-and-p2p](03-all-to-all-and-p2p.md) |
| ★★★ **`V(N)` 的推导与饱和** | `10-foundations/performance-models/02-amdahl-and-scaling.md:147-155,209` |
| **分层 all-reduce 的三段式与平衡条件** | `20-gpu-architecture/interconnect/03-collectives-on-topology.md` |
| **NCCL 在拓扑上怎么选算法**（实现层） | `110-communication/nccl`（本域另一主题） |

## 4. 踩坑与失败模式

### 4.1 ★★★ 漏掉 all-gather 那一段

- **现象**：形状、块边界、类型**全都合法**，程序**不报错**，只是**大部分块是空的**；
- **成因**：只做了 reduce-scatter ⇒ 每台设备手里**只有 1 块**是完整和，
  **其余 `p−1` 块根本没被拼回来**；
- **本机实测**（`code/` 的 `C.1`，`N = 4`、`n = 32`）：`max|diff|` = **316.0**；
- **规避**：★★★ **唯一的信号是「与单进程参照的差」**。

### 4.2 ★★★ 漏掉 reduce-scatter 那一段

- **现象**：同样**不报错**；缓冲区大小与块边界都对；
- **成因**：只做 all-gather、**不归约** ⇒ 每块**只有一台设备**的贡献（`x[d]` 的那一片）；
- **本机实测**（`C.2`）：`max|diff|` = **213.0** ⇒ ★★ **它不等于 `C.1` 的 316.0** ——
  ★ **两条错法是两种不同的失败**（`C.2` 的断言把这一点也钉住了）；
- **规避**：★★★ **可引用的性质是「漏哪一段都错、且两段的错法不同」；
  某个具体形状上那两个数本身不可引用**（换形状就换数）。

### 4.3 ★★ 把不整除当成边界情况

- **现象**：`p` 不整除 `n` 时，块大小不齐，而代码里写的是 `n // p`；
- **成因**：★ 四段式子**默认整除**；不整除时论文**单列了一条**（§1.4），且那条**是近似的**；
- **规避**：★★★ **引用必须带它的限定语**（`This cost is approximate …`）；
  ★ 本程序用 `D.3b` 把「`STAGE_ELEMS` 被每一档整除」**当口径守住**
  （★ 破坏它是**断言捕获**，不是让模拟器崩掉）。

### 4.4 ★★ 把「分段」写成「顺序执行」

- **现象**：以为两段必须**串行**（先把第一段全部跑完再跑第二段），
  从而在 `N` 小的时候**看不出差别**；
- **成因**：★ 两段确实**串行**（第二段要等第一段的结果），但**段内**有并行；
  ★★ 本程序的 `A.1` 在 `N = 2`（**退化对照**，两段各只有一步）上也对拍过一次 ——
  ★ **分段与不分段的差别在 `N = 2` 时最小**，所以要靠 `N = 4` 才看得出来。

## 5. 判据表

| 可机械判定的条件 | 结论 |
|---|---|
| 报「allreduce 的每卡搬运量」 | ★★ `2(N−1)/N × S`；★★★ **它就是两段之和**（`A.2`），推导在上游（回链，不重推） |
| 报「reduce-scatter 或 all-gather 的每卡搬运量」 | ★★ 两段**相等**，都是 `(N−1)/N × S`（`A.2`） |
| 要验证「分段 == 不分段」 | ★★★ **三条路径对拍**：两段接起来 / 不分段的整体 / 单进程参照（`A.1`） |
| 报「四个原语的步数」 | ★★ `reduce-scatter`/`all-gather` 各 `lg p`；`reduce`/`broadcast`（树形）各 `⌈lg p⌉`（`A.3`） |
| 报「reduce 与 broadcast 的总量」 | ★★ **都是 `(p−1) × n`** —— 它们是**同一棵树的两种遍历方向**（`A.4`） |
| 见到「`p` 不整除 `n`」 | ★★★ **不是边界情况**：论文单列 `Trec half`，且**自带「这是近似」的限定语**（§1.4、`D.6`） |
| 只做 reduce-scatter 就交付 | ★★★ **错**：每台设备只有 1 块是对的和，其余是空的（`C.1`） |
| 只做 all-gather 就交付 | ★★★ **错**：每块只有一台设备的贡献，**没有相加**（`C.2`） |
| 引 MPICH 的分档阈值 | ★★★ **必须带限定语**（2005 年 MPICH 的实现选择，不是普适规律） |

## 6. 小结与延伸

**一句话**：**all-reduce = reduce-scatter + all-gather**；★★★ 那条 `2(N−1)/N × S` 里的
**「`2×`」就是「两段」** —— 每段各 `(N−1)/N × S`，而**两段的每卡搬运量相等**。

**三条可独立复用的结论**：

1. ★★★ **四个原语的最优步数都是对数级**（`lg p` 或 `⌈lg p⌉`）——
   ★ 而且这两段的步数是**从枚举出来的消息里数出来的**，不是照闭式写死的；
2. ★★ **`reduce` 与 `broadcast` 是同一棵树的两种遍历方向** ⇒ 总搬运量相等（`(p−1) × n`）；
3. ★★★ **「不整除」在这些式子里不是边界情况** —— 论文为它单列了一条**近似**式。

**延伸**：

- ★★★ **各算法的步数与每卡搬运量（含三算法对照）** → [01-algorithm-family](01-algorithm-family.md)；
- ★★ **all-to-all 的总搬运量口径（它是**另一**种结构：`∝ p`、不饱和）· P2P 原语** →
  [03-all-to-all-and-p2p](03-all-to-all-and-p2p.md)；
- **本层不写、只回链的正文**：
  · ★★★ **`V(N)` 的推导与 `S` 的饱和** → `10-foundations/performance-models/02-amdahl-and-scaling.md:147-155,209`；
  · **bisection 下界 · 跨切口量 · 分层 all-reduce 三段式** → `20-gpu-architecture/interconnect/03-collectives-on-topology.md`；
  · **`algbw` vs `busbw`** → `interconnect/04-measuring-bandwidth.md`。

## 参考

- Thakur, Rabenseifner, Gropp, *Optimization of Collective Communication Operations in MPICH*,
  **IJHPCA** 19(1):49–66, 2005 —— **[C] 级，同行评审：是**（Crossref `10.1177/1094342005051521`）。
  一手 PDF：`mcs.anl.gov/~thakur/papers/ijhpca-coll.pdf`，抓取日 **2026-10-08**。
  ★ 本文用到的两条逐字式子：`Trabenseifner`（两段式的代数）、`Trec half` + 它的**近似限定语**
  （不整除时的 reduce-scatter）；★ 以及 Van de Geijn 广播的**两段式**散文依据
  （`… scattered … collected back …`）—— ★★ **它说明「分两段看」不是 allreduce 独有的结构**。
- 本仓库
  [01-algorithm-family](01-algorithm-family.md)、
  [03-all-to-all-and-p2p](03-all-to-all-and-p2p.md)、
  `10-foundations/performance-models/02-amdahl-and-scaling.md`（`V(N)` 的推导，本文不重推）。

## 编译与运行

```sh
cd code && make run                 # 三个编号程序，任一断言失败即非零退出
python3 02_stages_and_costs.py      # 只跑本文的配套程序
make 02_stages_and_costs            # ★★ 与上一行等价（都先走 check-env）
```

★ 本机的运行条件：`python3 3.13.5` / `numpy 2.4.4`，macOS arm64，
**无 GPU、无多卡、无 NCCL、无真实网络**。
★★ **本主题的三个程序都要 `numpy`**（两段的模拟器都驱动真实数组）
⇒ ★★★ **本文件不写「单跑某个程序不需要 numpy」那种措辞**。
★ 强制限定语：**不是 NCCL，也不是真机** —— 本文的一切数字都是**算法的代数**。

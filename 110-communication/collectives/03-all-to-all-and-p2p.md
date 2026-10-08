---
title: all-to-all 的算法与总搬运量口径，以及 P2P 原语
slug: communication-collectives-all-to-all-and-p2p
domain: 110-communication
topic: collectives
level: 3
difficulty: intermediate
prerequisites:
  - 110-communication/collectives/01-algorithm-family
tags: [collectives, all-to-all, point-to-point, sendrecv, payload-volume]
related:
  - 110-communication/collectives/02-stages-and-costs
  - 40-kernels/moe-kernels/01-moe-compute-pattern
  - 100-parallelism/pipeline-parallel/01-microbatches-and-bubbles
  - 20-gpu-architecture/interconnect/03-collectives-on-topology
code:
  - code/03_all_to_all_and_p2p.py
status: stable
updated: 2026-10-08
---

# all-to-all 的算法与总搬运量口径，以及 P2P 原语

> **all-to-all 与 all-reduce 的区别不在「搬多少」，在「搬的形状」**：
> all-reduce 的每卡搬运量**饱和**到 `2S`，而 all-to-all 的每卡搬运量**随 `p` 线性增长**。
> ★★★ 本文给出它的**三种算法**与**总搬运量口径**，
> 并接住两条上游让渡（all-to-all 的通信量分析 · **点对点通信的实现**）。
> ★★ **本层是【算法层】** —— 不是 NCCL，也不是真机。

## 0. 背景与动机

### 0.1 ★★ 本文接住的两条上游让渡

1. **`40-kernels/moe-kernels/01-moe-compute-pattern.md:243`** 逐字：
   > 「**它的通信量分析不在本域**，见 `110-communication/collectives`（正文在彼，此处回链）」

   ★★ 域规划 `§0.7 B.2` 称它是**全库被引最多的一条让渡**
   （依那里的枚举：`100-parallelism/expert-parallel` 的 3 篇 + 3 程序 + `Makefile`
   + `requirements.txt` + `specs.py` 反复回链它）。
   ★★ **本仓库不复核那条「最多」的口径** —— 只把**这一条让渡的存在与落点**接住。
2. **`100-parallelism/pipeline-parallel/01-microbatches-and-bubbles.md:298-299`**（**节引**，
   ★★ **不是逐字**：原文是**两行**、且带括注与尾句，下面这一版**删去了括注与尾句、
   并把两行并作一行**）：
   > 「…**点对点通信的实现**与**通信-计算重叠的机制** → `110-communication/`…」

   ★ **原文逐字对照**：`:298` = 「★★ **点对点通信的实现**与**通信-计算重叠的机制**（本文只出现
   「流水线级之间要互传激活」这个事实）」、`:299` = 「→ `110-communication/` 的 `overlap` 主题
   （**未建成**）—— ★ 本文**一行都不重写**它。」
   ⇒ ★★★ **原文把「P2P」与「重叠」【都】指向 `overlap`**；**本域据规划 `C4` 的分工**，
   把其中的「**点对点通信的实现**」那一半**改归 `collectives`**（由本文接住），
   **重叠**那一半仍留 `overlap`。★ 这条**改归属**是本域的组织决定，**不是原文的意思**，
   故此处**不标「逐字」**。
   ★ **P2P 原语本身**由本文接住（**集合通信的基础原语** —— 树形算法就是「成对的 P2P」串起来的）。

### 0.2 一个具体的对照场景

**本机实测**（`code/` 的 `A.1`/`B.1`）：`p = 8`、每对设备之间交换 `b = 4` 个元素
（★ **归一化**：让两者的**每设备数据量**相等 ⇒ all-reduce 的输入取 `S = p × b = 32`），

| 原语 | 每卡搬运量 | 全网总搬运量 | 随 `p` 的走向 |
|---|---|---|---|
| **all-to-all** | `(p−1) × b = 28` | `p(p−1) × b = 224` | ★ **`∝ p`**（`b` 固定时） |
| **all-reduce** | `2(p−1)/p × S = 56` | `2(p−1) × S = 448` | ★★ **饱和到 `2S`** |

★★★ **两者在同一个 `p` 上差【恰好 2 倍】**（`56 / 28 = 2`、`448 / 224 = 2`）——
★ 那个 `2` 就是 all-reduce 的**两段**（见 [02](02-stages-and-costs.md)），all-to-all 只有**一段**。
★★ **但「走向」完全不同**：all-to-all 的每卡量**随 `p` 长**（`∝ p`），
all-reduce 的**不随 `p` 长**（饱和）。
★ **换一个 `p`，比值仍是 `2`，而两条曲线越分越开**。

## 1. 原理

### 1.1 ★★★ 三种 all-to-all 算法：**步数不同、搬运量相同**

论文（`code/specs.py`，[C] 级）按**消息大小**分了三个算法：

| 算法 | 适用档（论文逐字，★ 见 §4.4 的限定语） | 步数（逐字） |
|---|---|---|
| **index（Bruck）** | 短消息（`≤ 256 B`） | `⌈lg p⌉`（`It is a store-and-forward algorithm that takes ⌈lg p⌉ steps`） |
| **irecv-isend** | 中等消息（`256 bytes to 32 KB per message`） | 论文未给步数式 |
| **成对交换** | 长消息 + **`p` 是 2 的幂** | `p − 1`（`which takes p − 1 steps.`） |

★★ **三种算法搬的东西一样多**（每卡 `(p−1) × b`）—— ★★ **与 all-gather 的三算法同型**：
★ **选算法选的是【步数】**（`⌈lg p⌉` vs `p−1`），**不是**搬运量。

★★★ **一条【同名不同指】**：这里的 **`Bruck` 是 all-to-all（index）与 all-gather 的算法**，
**不是 allreduce 的**（见 `code/specs.py` 的 `MPICH_BRUCK_SCOPE_NOTE`）。

### 1.2 ★★★ 总搬运量口径（C3 的分工）

★★★ **本主题只写【总搬运量 / 载荷口径】**：

```text
每卡搬运量  = (p−1) × b          # 每台设备要发给「除自己外的每一台」一份
全网总搬运量 = p × (p−1) × b      # 把每台设备各自发出的元素数加起来
```

★ **口径声明**：这里数的是「**两台设备之间真的搬了多少元素**」（**载荷**）。

> ★★★ **一条硬边界（本域一个字都不写）**：**all-to-all 的【跨切口量】`N·V/4`**
> —— 它**不在本文**。正文在 `20-gpu-architecture/interconnect/03-collectives-on-topology.md`
> （**正文在彼，此处回链**）。
> ★ **两者不是一回事**：跨切口量问的是「**有多少字节必须穿过拓扑的某个切面**」，
> 载荷口径问的是「**两台设备之间搬了多少**」。**同一个数在两处可以完全不同**。

### 1.3 ★★★ all-to-all vs all-reduce：**总量差 2 倍**，**走向完全不同**

★ **归一化口径**（不写清就没法比）：让两者的**每设备数据量相等** ——
all-to-all 是 `p` 块、每块 `b`（共 `p·b`）；all-reduce 的输入 `S = p·b`。

| 量 | all-to-all | all-reduce |
|---|---|---|
| **每卡搬运量** | `(p−1) × b` | `2(p−1)/p × S` |
| **全网总搬运量** | `p(p−1) × b` | `2(p−1) × S` |
| ★ **总量之比** | — | **恰好 `2`** |

★★★ **本机实测**（`code/` 的 `B.1`，`A2A_GRID` 三档）：**总搬运量之比恒为 `2.0`**
（相对差 `< 1e-12`）。★ 为什么是 `2` 而不是别的：★ all-reduce 是**两段**（见
[02](02-stages-and-costs.md)），而 all-to-all 是**一段**。

★★★ **而「走向」完全不同**（这才是更要紧的那一条）：

- **`b` 固定**时，all-to-all 的每卡搬运量 `(p−1) × b` **`∝ p`** —— ★ **它不会饱和**；
- all-reduce 的每卡搬运量 `2(p−1)/p × S` **饱和到 `2S`**（`S` 固定时与 `p` 无关）。

⇒ ★★ **这就是为什么 `interconnect/03` 说「all-to-all 是 bisection-bound 的、
ring all-reduce 不是」** —— 那句话的正文在那一篇（此处回链），
★ 而**它的代数来源**就在这里：**一个 `∝ p`、一个饱和**。

### 1.4 ★★ P2P 原语：集合通信的基础

| 原语 | 语义要点 | 它为什么重要 |
|---|---|---|
| **`send` / `recv`** | 一条**单向**消息；**可能阻塞** | ★ 树形算法（二项树）就是一堆配对的方向性 `send`/`recv` |
| **`sendrecv`** | ★★ **同时发与收**（两个方向在**一个**调用里完成） | ★★★ **它把「我等你、你等我」这件事消掉了** |

★★ **本机实测**（`code/` 的 `A.4`/`C.2`）：每台设备与 `(d+1) % p` 交换一块，
- 用 **`sendrecv`**：**一轮**就把数据换好（`rounds = 1`）；
- 用「**先 `recv` 后 `send`**」：**0 / 8 台**收到数据、**0 台**发出 ⇒ **一步都不动**。

★★★ **这不是「慢」，是「互等」**：每台设备都在等它的**前驱**先发，
而前驱也在等**它的**前驱 —— 环上**没有起点**。
★ 这正是 `sendrecv`（以及非阻塞原语）存在的理由。
★★ **模型声明**：本程序用一个**简化的「就绪」模型**演示它 ——
**不是**真的 socket 死锁（真的那个在 `../probes/`，走 loopback，**只打印不断言**）。

## 2. 执行映射

### 2.1 本机怎么算这件事

`code/03_all_to_all_and_p2p.py` 的参照是**真消息序列模拟**：
每一步记录 `MSG(step, sender, receiver, block, n_elem)`。

| 角色 | 是什么 | 独立性 |
|---|---|---|
| **参照（成对交换）** | `sim_a2a_pairwise` | ★ 每步与 `(d+1+k)%p` 交换，**逐步枚举** |
| **参照（直接交换）** | `sim_a2a_direct` | ★ **另一条调度路径**（`p−1` 份同时发） |
| **单进程参照** | `ref_all_to_all` = 只**重排**、不搬 | ★ 调用表是**空集** ⇒ 不依赖任何模拟器 |

★★★ **「总搬运量」是【逐条消息求和】得到的**，不是「每卡 × 台数」乘出来的：
`sum(m.n_elem for m in msgs)` —— ★ 这样**写错索引会立刻改变它**。

### 2.2 ★★ 一条必须写明的限定语

> ★★ 本机**无 GPU、无多卡、无 NCCL、无真实网络**；「搬运」是 **numpy 索引**。
> ★ 强制限定语：**「不是 NCCL，也不是真机」**。
> ★★ 本程序的数字是**本机实测值**：`python3 3.13.5` / `numpy 2.4.4`（macOS arm64）。

## 3. 实现与演进

### 3.1 从 P2P 到集合通信

★ **集合通信是 P2P 的组织方式**：二项树的每一步都是一次 `sendrecv`；
ring allreduce 的每一步是「发一块、收一块」—— ★ **都靠 `sendrecv` 消掉互等**。
★★ **这就是本文为什么把 P2P 与 all-to-all 放在一起**：★ 前者是**原语**，后者是**模式**。

### 3.2 ★★ 为什么 all-to-all 在 MoE / 序列并行里这么贵

★ all-to-all 的每卡搬运量 **`∝ p`**（§1.3）⇒ **`p` 翻倍，每卡搬运量也翻倍**。
★★ 这在 `100-parallelism/expert-parallel`（两次 all-to-all 送 / 收 token）与
`sequence-context-parallel`（all-to-all 做序列重排）里是同一笔账。
★ 但**本文只给「搬了多少」** —— ★★ 它在**拓扑上**要穿多少切口、受什么限制，
是 `interconnect/03` 的正文（此处回链）。

### 3.3 与相邻主题的分工

| 话题 | 在哪 |
|---|---|
| **all-to-all 的算法与总搬运量口径 · P2P 原语** | **本文** |
| **各算法的步数与每卡搬运量（allreduce / allgather / broadcast）** | [01-algorithm-family](01-algorithm-family.md) |
| **两段式与四个原语的代价** | [02-stages-and-costs](02-stages-and-costs.md) |
| **all-to-all 的【跨切口量】`N·V/4` · bisection 约束** | `20-gpu-architecture/interconnect/03-collectives-on-topology.md` |
| **MoE 里两次 all-to-all 的位置与尺寸** | `100-parallelism/expert-parallel/01-expert-sharding.md` |
| **流水线并行的点对点通信与气泡** | `100-parallelism/pipeline-parallel/01-microbatches-and-bubbles.md` |

## 4. 踩坑与失败模式

### 4.1 ★★★ 「每个设备只发一份」

- **现象**：索引合法、类型合法、程序**不报错**，只是**大部分数据根本没搬**；
- **成因**：只把数据发给**一个**邻居（`(d+1) % p`），**其余 `p−2` 份没发**；
- **本机实测**（`code/` 的 `C.1`，`p = 8`、`b = 4`）：应有 `64` 个格子有数据、实到 `16` 个，
  **缺 `48` 个**；★ **搬运量从 `28` 掉到 `4`**；
- **规避**：★★★ **搬运量的下降是唯一信号** —— ★ 但注意：
  **别的错法（如块索引写错）可以不改搬运量**（见 [01](01-algorithm-family.md) 的 §4.3）
  ⇒ **「搬运量对」不等于「数据对」**。

### 4.2 ★★★ P2P：「先 `recv` 后 `send`」⇒ 互等

- **现象**：**程序不报错**、`send`/`recv` 的参数全对 —— **只是永远不会返回**；
- **成因**：环上每台设备都在等它的前驱先发，而前驱也在等**它的**前驱；
- **本机实测**（`C.2`，`p = 8`）：**0 / 8 台**收到、**0 台**发出；
  ★ 而同一组数据用 `sendrecv` **一轮**（`rounds == 1`）就完成 —— ★★ **那一条【成功行为】
  在 `A.4` 里断言**（依 `CONTRIBUTING:238`，成功行为归 A 类，**`C.2` 只量失败量**）；
- **规避**：★★★ **把「发」与「收」放进同一个调用**（`sendrecv`），或用非阻塞原语；
  ★★ **另一条**：保证配对**不成环**（二项树的边就**没有**环）。
  ★ **模型声明**：本程序用的是**简化的就绪模型**，**不是**真的 socket 死锁。

### 4.3 ★★ 把「总搬运量」与「跨切口量」混为一谈

- **现象**：用「两台设备之间搬了多少」去回答「**拓扑切面上要过多少字节**」；
- **成因**：★ 两者**是同一个物理动作的两个不同投影**；
  一个是**载荷**（端点之间），一个是**切口**（拓扑的某个割）——
  ★★ **同一个 all-to-all，两个数可以完全不同**；
- **规避**：★★★ **写清口径**：本文只给**载荷 / 总搬运量**；
  **跨切口量**的正文在 `interconnect/03`（此处回链）。

### 4.4 ★★ 把 2005 年 MPICH 的分档当成普适规律

- **现象**：写「**集合通信**在 `256 B` 以下是短消息、`32 KB` 以上是长消息」；
- **成因**：论文里的 `256 bytes to 32 KB` 是**它自己的实现选择** ——
  论文自述其阈值是 `experimentally determined cutoff points`
  （**由实验确定的切分点**）；
- **规避**：★★★ **引用必须带限定语**：**它们是 2005 年 MPICH 在 Myrinet / IBM SP 上的实现选择**
  （`code/specs.py` 的 `MPICH_CUTOFF_QUALIFIER`，★ 它是一条 **[E] 本库的引用纪律**、
  **不是**论文里的话 —— 所以**不得**把它写成引文）。

## 5. 判据表

| 可机械判定的条件 | 结论 |
|---|---|
| 报「all-to-all 的每卡搬运量」 | ★★ `(p−1) × b` ⇒ **`∝ p`**（`b` 固定时），**不饱和**（`A.1`） |
| 报「all-to-all 的全网总量」 | ★★ `p(p−1) × b`（★ **由枚举出来的消息逐条求和**，`A.3`） |
| 比较 all-to-all 与 all-reduce | ★★ **先统一「每设备数据量」**；★ 同一份数据下总量之比**恰好 `2`**（`B.1`） |
| 说「`p` 大了 all-to-all 也饱和」 | ★★★ **不成立**：它 `∝ p`；**饱和的是 all-reduce**（`B.1`） |
| 要回答「拓扑切口上过多少字节」 | ★★★ **不在本文**：回链 `interconnect/03` 的 `N·V/4`（§1.2） |
| 报「三种 all-to-all 算法的搬运量」 | ★★ **三者相同**（`(p−1) × b`）⇒ 差别**只在步数**（`⌈lg p⌉` vs `p−1`，`B.2`） |
| 见到「`Bruck` 是 allreduce 的算法」 | ★★★ **不成立**：它是 **allgather 与 all-to-all(index)** 的算法（`D.6`） |
| 只把数据发给一个邻居 | ★★★ **错**：缺 **`p(p−2)`** 个格子、搬运量掉到 **`1/(p−1)`**（`C.1`；★ `p = 8` 时是 **`48`** 个格子、`28 → 4`，即 `4/28 = 1/7`） |
| 用「先 `recv` 后 `send`」做环上交换 | ★★★ **互等**：一轮都不进展（`C.2`）—— ★ 用 `sendrecv` |
| 引 MPICH 的分档阈值 | ★★★ **必须带限定语**（2005 年 MPICH 的实现选择，不是普适规律，`D.6`） |

## 6. 小结与延伸

**一句话**：**all-to-all 的每卡搬运量 `(p−1) × b` 随 `p` 线性增长、不会饱和**，
这与 all-reduce 的 `2(N−1)/N × S`（饱和到 `2S`）是**两种不同的走向**；
★ 而它的三种算法**搬运量相同、步数不同**（`⌈lg p⌉` vs `p−1`）。

**三条可独立复用的结论**：

1. ★★★ **all-to-all 的「贵」在形状上，不在总量上** —— 同一份每设备数据下它的总量只有
   all-reduce 的一半，但它的**每卡**量 `∝ p`；
2. ★★ **P2P 原语是集合通信的地基**：★★★ **`sendrecv` 消掉互等，是环上交换能跑起来的原因**；
3. ★★★ **两条口径不可混**：「**总搬运量 / 载荷**」（本文）与「**跨切口量** `N·V/4`」
   （`interconnect/03`）—— **同一个 all-to-all，两个数可以完全不同**。

**延伸**：

- ★★★ **各算法的步数与每卡搬运量** → [01-algorithm-family](01-algorithm-family.md)；
- ★★ **两段式与四个原语的代价** → [02-stages-and-costs](02-stages-and-costs.md)；
- **本层不写、只回链的正文**：
  · ★★★ **跨切口量 `N·V/4` · bisection 下界 · 分层三段式** →
    `20-gpu-architecture/interconnect/03-collectives-on-topology.md`；
  · **`V(N)` 的推导与饱和** → `10-foundations/performance-models/02-amdahl-and-scaling.md:147-155,209`；
  · **MoE 里两次 all-to-all 的位置与尺寸** → `100-parallelism/expert-parallel/01-expert-sharding.md`；
  · **流水线并行的 P2P 与气泡** → `100-parallelism/pipeline-parallel/01-microbatches-and-bubbles.md`。

## 参考

- Thakur, Rabenseifner, Gropp, *Optimization of Collective Communication Operations in MPICH*,
  **IJHPCA** 19(1):49–66, 2005 —— **[C] 级，同行评审：是**（Crossref `10.1177/1094342005051521`）。
  一手 PDF：`mcs.anl.gov/~thakur/papers/ijhpca-coll.pdf`，抓取日 **2026-10-08**。
  ★ 本文用到的三条逐字引文（index 算法归属 · index 的 `⌈lg p⌉` 步 · 成对交换的 `p − 1` 步
  与它的**两个前提**）注册在 `code/specs.py` 的 `# 来源：` 里。
- 本仓库
  [01-algorithm-family](01-algorithm-family.md)、[02-stages-and-costs](02-stages-and-costs.md)、
  `20-gpu-architecture/interconnect/03-collectives-on-topology.md`（跨切口与 bisection）、
  `100-parallelism/expert-parallel/01-expert-sharding.md`、
  `100-parallelism/pipeline-parallel/01-microbatches-and-bubbles.md`。

## 编译与运行

```sh
cd code && make run                 # 三个编号程序，任一断言失败即非零退出
python3 03_all_to_all_and_p2p.py    # 只跑本文的配套程序
make 03_all_to_all_and_p2p          # ★★ 与上一行等价（都先走 check-env）
```

★ 本机的运行条件：`python3 3.13.5` / `numpy 2.4.4`，macOS arm64，
**无 GPU、无多卡、无 NCCL、无真实网络**。
★★ **本主题的三个程序都要 `numpy`** ⇒ ★★★ **本文件不写「单跑某个程序不需要 numpy」那种措辞**。
★ 强制限定语：**不是 NCCL，也不是真机** —— 本文的一切数字都是**算法的代数**；
★★ **P2P 的「互等」是用一个简化的就绪模型演示的**，**不是**真的 socket 死锁
（真的那个在 `../probes/`，走 loopback，**只打印不断言**，且**它的任何数字都不是性能依据**）。

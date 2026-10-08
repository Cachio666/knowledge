---
title: 胖树与三层 Clos：k³/4 台主机、576 条等价路径，与它们是怎么数出来的
slug: communication-network-topology-fat-tree-and-clos
domain: 110-communication
topic: network-topology
level: 3
difficulty: intermediate
prerequisites:
  - 20-gpu-architecture/interconnect/02-topology-and-bisection
tags: [fat-tree, clos, datacenter-topology, bisection, equal-cost-paths, physical-layer]
related:
  - 110-communication/network-topology/02-rail-optimized
  - 110-communication/network-topology/03-roce-and-infiniband
  - 20-gpu-architecture/interconnect/02-topology-and-bisection
  - 20-gpu-architecture/interconnect/03-collectives-on-topology
code:
  - code/01_fat_tree_and_clos.py
status: stable
updated: 2026-10-08
---

# 胖树与三层 Clos：k³/4 台主机、576 条等价路径，与它们是怎么数出来的

> **一台 48 端口的交换机，怎么接出 27,648 台主机、还让任意两台之间都有 576 条等长路径？**
> 本文讲**胖树（fat-tree）/ 三层 Clos** 的**组合结构**：`k` 叉、pod、聚合层、核心层，
> 以及那三件**可以复算**的事 —— `k³/4` 台主机、`(k/2)²` 台核心、`(k/2)²` 条等价路径。
> ★★★ **本层是【物理层】** —— 讲的是**拓扑这个组合对象**，不是算法的代数（那在 `collectives`），
> 也不是任何真实库的行为（那在 `nccl`）。
> ★★★ **本主题的 A 类锚点 = SIGCOMM 2008 的论文值**（与 `nccl` 的「没有 A 类」**相反**）：
> 本文的每一个数都由本机把拓扑**建成一张图再数出来**，再与论文的实例值（`27,648` / `576` / `1,152`）对拍。

## 0. 背景与动机

`20-gpu-architecture/interconnect/02-topology-and-bisection` 已经讲了**卡间**的拓扑
（`ring` 与全互联的 `bisection` 的比较、NVSwitch 的价值在**路径均匀性** —— ★ **正文在彼，此处只回链**）。
那些是**节点内**（scale-up）的拓扑。

**那本文补什么？** 补**机间**（scale-out）那一层：**当交换机只有 `k` 个端口、而集群有几百上千个节点时，
网络长什么样** —— 这一层全库**没有正文**（★ 上游 `interconnect/01,02` 写的是 NVLink / PCIe 侧）。

> **一个知识点只有一处正文**（`CONTRIBUTING` §7）。**bisection 的通用推导**（`bisection(ring) = L×b`、
> 下界 `t ≥ 2S/B`）与**分层 all-reduce 的三段式**在
> `20-gpu-architecture/interconnect/02-topology-and-bisection.md` 与
> `03-collectives-on-topology.md`（正文在彼，此处回链）。★ 本文只给 **bisection 的【定义】**与
> **胖树在一种显式切法下的结构计数**（§1.4）。

**那本文的两个知识对象**：

1. ★★★ **胖树的参数化结构**：`k` 叉、`k` 个 pod、每 pod 两层、每层 `k/2` 台；
2. ★★★ **三个可复算的数**：主机 `k³/4`、核心 `(k/2)²`、**不同 pod 主机对之间的等价路径数** `(k/2)²`。

**载体（逐字，[C] 级论文）**：SIGCOMM 2008 的
`We organize a k-ary fat-tree as shown in Figure 3. There are k pods, each containing two layers of k/2 switches.`

## 1. 原理

### 1.1 ★★★ 三层结构：谁接谁

`k` 是**交换机的端口数**（论文的例子里 `k = 48`）。整个网络由 `k` 个 **pod** 组成：

| 层（tier） | 每 pod 几台 | 每台的 `k` 个端口怎么用 |
|---|---|---|
| **边缘层**（edge） | `k/2` | `k/2` 下接**主机**、`k/2` 上接**本 pod 的聚合层** |
| **聚合层**（aggregation） | `k/2` | `k/2` 下接本 pod 的边缘层、`k/2` 上接**核心层** |
| **核心层**（core） | —— 全局 `(k/2)²` 台 | ★ **每台对每个 pod 各占一个端口**（`k` 个端口用满） |

★★ 论文对核心层的逐字（两句合起来才**唯一确定**连接规则）：

> `There are (k/2)2 k-port core switches. Each core switch has one port connected to each of k pods.`

★★★ **`(k/2)2` 是提取形态**：原文是 `(k/2)²`，而 PDF 的文本提取把**上标压平**
（★ 这是本文最重要的一条转写纪律，见 §4.2）。同理 `k³/4` 提取成 `k3 /4`。

★★ 有了这三层，**主机数、核心数就都是【导出量】**了 —— 它们是 §1.2 与 §1.3 要数的东西。

### 1.2 ★★★ `k³/4` 台主机：为什么是这个数

论文逐字：`In general, a fat-tree built with k-port switches supports k3 /4 hosts.`

本文不抄这个结论，而是**数**出来（`code/01_fat_tree_and_clos.py` 的 `A.1`）：

```
每个 pod 的边缘交换机数      = k/2
每台边缘交换机的下行端口数    = k/2   （k 个端口里，k/2 被上行占掉）
⇒ 每 pod 的主机数            = (k/2) × (k/2) = k²/4
⇒ 全网主机数                 = k × k²/4 = k³/4      ∎
```

★★ **这就是「主机数 = `k³/4`」的全部内容** —— 它是**两次端口预算**的结果，没有别的机制。

### 1.3 ★★★ `(k/2)²` 台核心交换机：为什么不是 `k/2` 台

**核心层的端口预算**是反向算的：每台**聚合**交换机有 `k/2` 个上行端口要接核心。
于是核心交换机**至少要几台**？

```
全网"聚合→核心"的链路总数           = k 个 pod × (k/2 台聚合) × (k/2 个上行端口) = k³/4
每台核心交换机最多承几条            = k       （对每个 pod 各一条）
⇒ 最少核心数                        = (k³/4) / k = k²/4 = (k/2)²     ∎
```

★★★ **`code/` 里这件事是「涌现」的**：`A.1` 用的是一个**贪心分配器** ——
对每一条「聚合→核心」的链路，找**编号最小的、还没连过这个 pod** 的核心交换机；
一台都没有就**新建**一台。核心的**条数不是预设的**，而是**端口预算逼出来的**。

★★ 论文对核心层的另一句逐字（`k = 48` 的实例）：

> `a fat-tree built from 48-port GigE switches would consist of 48 pods, each containing an edge layer and an aggregation layer with 24 switches each.`

★ `48/2 = 24` ✓ —— 与 §1.1 的 `k/2` 一致，这是本文 `B.1` 的一条**可枚举的自检**。

### 1.4 ★★★ 等价路径数：`(k/2)²` 条，**每条由一个核心交换机决定**

★★★ 这是本文最容易被**抄错**的一个数。论文给了**两种写法**：

> （一般式，§3.1）`there are (k/2)2 such shortest-paths between any two hosts on different pods, but only one is chosen`
> （实例，§2.2）`There are 576 equal-cost paths between any given pair of hosts in different pods.`

**为什么是 `(k/2)²`？** ★★ 关键在**核心层每一台都连着每一个 pod**：

```
发端主机 → 边缘 → 聚合 →【核心】→ 聚合 → 边缘 → 收端主机
                          ↑
                  ★ 选哪一台核心，就唯一确定了一条路径
                  （核心到每个 pod 的分支是固定的：一台核心在每个 pod 上只占一个端口）
```

⇒ ★★★ **路径数 = 核心交换机台数 = `(k/2)²`** —— 这才是「`(k/2)²` 条路径」的**结构原因**，
**不是**巧合（本文 `A.2` 会把这两个数**分别**枚举出来）。

★★ **`576` 是 `k = 48` 那一档的值**（`(48/2)² = 576`）：★ 引用它**必须说清是哪一档 `k`** ——
把实例值当成通式，就是本库 `no-fabricated-numbers` 那一族里「**实例升格**」的形态。

★★ **论文同时给了一句限定语**（关于「1:1」这类说法，本文 §1.5 会用到）：

> `Achieving an oversubscription ratio of 1:1 in practice may be difficult because of the need to prevent packet reordering for TCP flows.`

### 1.5 ★★ bisection 的**定义**与一种显式切法

**定义**：把网络切成两半（两半的主机数相等）时，**跨越切口的链路总容量**。

★★★ **通用推导在 `20-gpu-architecture/interconnect/02`**（`bisection(ring) = L×b` 与它「与 `N` 无关」
那一条，以及全互联的 `∝N²`）—— **本文一行都不重推**。

本文只做一件可枚举的事（`code/` 的 `B.2`）：按**显式规则**切一刀，把**跨切口的边逐条数出来**。

| 显式切法 | 计数（`k = 4 / 8 / 16`） |
|---|---|
| pod `[0, k/2)` 一侧、`[k/2, k)` 另一侧；**核心交换机全部划到第一侧** | 跨切口链路 = `k³/8` = **一侧的主机数** |

★★ **读法**：这就是论文说的 `1:1`（每台主机对应一条跨切口链路）的**结构形式**。
★★★ **但必须带上 §1.4 末尾那句限定语**（`in practice may be difficult ...`）
—— ★「理论上 1:1」与「实践中 1:1」是**两件事**。

★★ **非阻塞性的原句也有限定词**（论文 §2.2）：

> `fat-trees are rearrangeably non-blocking, meaning that for arbitrary communication patterns, there is some set of paths that will saturate all the bandwidth available to the end hosts in the topology.`

★★★ `rearrangeably`（**可重排**）与 `some set of paths`（**存在**一组路径）**一个都不能漏** ——
漏掉就把「可重排非阻塞」读成了「怎么走都不阻塞」，那是**两件不同的事**。

### 1.6 ★★ 三层与两层（`tier`）：论文为什么选三层

> `A three-tiered design (see Figure 1) has a core tier in the root of the tree, an aggregation tier in the middle and an edge tier at the leaves of the tree. A two-tiered design has only the core and the edge tiers. Typically, a two-tiered design can support between 5K to 8K hosts.`

★★ **两条规模线**：两层 `5K to 8K` 主机；论文自己的目标是 `~25,000` ⇒ 它**必须**用三层。

★★★ **一条术语纪律**：本文一律写「**三层 Clos**」并保留英文 `tier`。
★★ **不得裸写 `stage` / 阶段** —— 库内那两个字已有**四个**含义
（TVM 计算阶段 · cutlass 流水级数 · 流水线并行的 stage · ZeRO 阶段，见域 README 的「同名不同指」表）。
★ 本主题的「层」是**拓扑层**，与 `collectives` 的「通信阶段」**不是一回事**。

## 2. 执行映射

### 2.1 本机怎么算这件事

`code/01_fat_tree_and_clos.py` 的 A 类是**真 A 类**：

| 角色 | 是什么 | 它为什么**站得住** |
|---|---|---|
| **被测** | 闭式：`cf_hosts(k)=k³/4` · `cf_cores(k)=(k/2)²` · `cf_paths(k)=(k/2)²` | ★★ 它们**不参与建图**（`D.8` 在源码层核） |
| **参照** | 把胖树**建成一张显式的图**（端口驱动 + 按需新建核心），再**数节点 / 数度 / BFS 数最短路** | ★★ 一条**图遍历**路径，与闭式**代数上独立** |
| **第三方（外部锚点）** | **SIGCOMM 2008 的论文值**：`27,648` 主机 · `1,152` 子网 · `576` 核心 · `576` 条路径 | ★★★ **这才是 A 类的靶心** —— 与**外部工件**对拍 |

★★★ **⇒ 本程序【有】A 类**（与同域 `nccl` **相反**）：`A.1`（规模）与 `A.2`（路径数 + 论文实例值）。

★★ **两条本程序真正有信息量的 C 类**（★ 依 `CONTRIBUTING:238`，**只断言「错法真的失败」**；
★★ 「真值放行」是**成功行为**，已移到 `A.1`/`A.2`）：

| 断言 | 它喂进去的**错法** | 它要求的结果 |
|---|---|---|
| `C.1` | 把规模公式写成 `k³/8` / `k²/4` / 核心写成 `k/2` / `k²` | **被拒**（四种错法全拒；★ 真值 `(128,16)` 放行在 `A.1`） |
| `C.2` | 把等价路径数写成 `k/2`（漏一个因子） | **被拒**（★ 这个错法**特别可信**，见 §4.3；★ `(k/2)²` 放行在 `A.2`） |

### 2.2 ★★★ 一条必须写明的限定语

> ★★ 本机**无 GPU / 无 NCCL / 无 InfiniBand / RoCE 网卡 / 无 NVLink**。
> ★★ 本文的一切**不是**任何真实网络的性能 —— 本文算的是**拓扑的结构**（组合对象）。
> ★★ 载体 = SIGCOMM 2008 一手 PDF（`https://cseweb.ucsd.edu/~vahdat/papers/sigcomm08.pdf`，
> 本轮 `curl` 实测 `991601` 字节），抓取日 **2026-10-08**。
> ★★★ **不写抓取日的引文不算引文**（`P19` 的第一道防线）。

★★ **本文的「形态」一句话**：编号程序**跑在默认 `python3` 下、只用标准库**
⇒ 落在 `CONTRIBUTING` §2 表的**第二种情形**（**程序内断言**）—— ★ 与 `nccl` 不同，本文**A/B/C 齐全**。

### 2.3 ★★ 覆盖边界：本文守什么、不守什么

- **守**：拓扑的**计数**（主机 / 核心 / 路径 / 度）与「**错公式会不会被拒**」；
- **不守**：任何**真实网络的性能**（吞吐 / 延迟 / 带宽），任何**bisection 下界的推导**。

## 3. 实现与演进

### 3.1 从 `k = 4` 到 `k = 48`：参数化规模表

`code/` 的 `B.1` 把五档枚举出来（★ 每一格都是**数出来的**，不是套公式）：

| `k` | 主机 | 边缘 | 聚合 | 核心 | 交换机总数 |
|---|---|---|---|---|---|
| **4** | 16 | 8 | 8 | 4 | 20 |
| 6 | 54 | 18 | 18 | 9 | 45 |
| 8 | 128 | 32 | 32 | 16 | 80 |
| 16 | 1024 | 128 | 128 | 64 | 320 |
| **48** | **27648** | **1152** | **1152** | **576** | **2880** |

★★ **`k = 4` 是论文 Figure 3 的「最简非平凡实例」**；`k = 48` 是论文的实例。
★★ **论文自设的讨论范围到 `k = 48`**（`In this paper, we focus on designs up to k = 48.`）——
★ 引用 `k³/4` 时**必须带上这个限定**（「单一实践升格」的反面）；★★ **但这不是公式的有效性上界**
（★ `specs.py` 的口径：论文自设的上限，不是物理上限）。

### 3.2 ★★ 为什么核心层不能只用 `k/2` 台

`(k/2)²` 与 `k/2` 只差一个因子，但两者的**结构含义**完全不同：

- `k/2` 是**同一个 pod 内**的层内交换机数（**局部**量）；
- `(k/2)²` 是**全局**的核心交换机数（**要把所有 pod 接起来**的量）。

★★★ **如果核心只用 `k/2` 台**，那么每台核心要在每个 pod 上占**多个端口** ⇒
「一台核心对每个 pod 只有一条链路」这条前提**被破坏** ⇒
§1.4 的「路径数 = 核心台数」**当场失效**。★ 这就是 `C.1`/`C.2` 存在的理由。

### 3.3 与相邻主题的分工

| 话题 | 在哪 |
|---|---|
| **ring 拓扑 / 全互联的 `bisection`**、**NVSwitch**、**NVLink 域逐代** | `20-gpu-architecture/interconnect/02-topology-and-bisection.md` |
| **`t ≥ 2S/B`（bisection 下界）** · **分层 all-reduce 三段式** | `20-gpu-architecture/interconnect/03-collectives-on-topology.md` |
| **链路带宽的三种口径**（线速率 / 载荷 / 营销取整） | `20-gpu-architecture/interconnect/01-bandwidth-conventions.md` |
| ★★ **胖树 / 三层 Clos 的组合结构** | **本文** |
| ★★ **rail-optimized（同一批硬件换一种布线）** | [02-rail-optimized](02-rail-optimized.md) |
| ★★ **RoCE v2 / PFC / ECN / SHARP** | [03-roce-and-infiniband](03-roce-and-infiniband.md) |

## 4. 踩坑与失败模式

### 4.1 ★★★ 把规模公式写错（`k³/8` / `k²/4` / `k/2` 核心）

- **现象**：表里的数与论文对不上，但**程序不报错**（那些公式都是合法表达式）；
- **成因**：把 `k³/4` 记成 `k³/8`（漏一半）或 `k²/4`（少一个 `k`）；把核心数记成 `k/2`（漏乘 `k/2`）；
- **本机实测**（`code/` 的 `C.1`）：`k = 8` 时四种错法**全部被拒**（★ 真值 `(128, 16)` 的**放行**在 `A.1` 断言 —— 依 `CONTRIBUTING:238`，成功行为归 A 类）；
- **规避**：★★★ **把「数」变成「枚举」** —— 一旦主机数是**端口填出来的**、核心是**按需新建的**，
  任何闭式写错都会被**枚举值**当场否掉。

### 4.2 ★★★ 把数学上标的**提取形态**抄错（引文层）

- **现象**：引文看起来对，公式**悄悄错掉**；
- **成因**：`pdftotext -layout` 把 `(k/2)²` 提取成 **`(k/2)2`**（上标压平、**无空格**）、
  把 `k³/4` 提取成 **`k3 /4`**（上标压平、**`/4` 前有一个空格**）；
  ★★ 而且版面是**双栏** —— 一句话会被切成两段（`... between any given pair` ⏎ `of hosts in different pods.`）；
- **本机实测**（`code/` 的 `D.5`）：六条引文**逐字钉住**，任何「顺手改成上标」的编辑都会失败；
- **规避**：★★★ **`specs.py` 里存的是【提取形态】**，`# 来源：` 里写明提取形态与排版形态的差别。

### 4.3 ★★★ 把「一个 node 字段写错」的题眼（量法口径）

- **现象**：路径数写成 `k/2` 而不是 `(k/2)²`，表**看着合理**；
- **成因**：★★★ **`k/2` 在同一个拓扑里到处都是**（每层交换机数、每交换机的主机数、端口的一半）
  ⇒ 它在**不同的指**上反复出现 ⇒ 引用时**极易张冠李戴**；
- **本机实测**（`code/` 的 `C.2`）：`k = 4/8/16` 三档，`k/2` **被拒**（★ `(k/2)²` 的**放行**在 `A.2` 断言）；
- **规避**：★★ 问「**这个数指什么**」（设备数？路径数？）；★ 本文 `specs.py` 里
  `SIGCOMM08_CORES_AT_K48` 与 `SIGCOMM08_PATHS_AT_K48` 都是 `576` ——
  ★★ **同值不同指，所以用两个常量名分开存**。

### 4.4 ★★ 把限定语丢掉（引文层）

- **现象**：把「`rearrangeably` 非阻塞」写成「非阻塞」，把「`k = 48` 的 `576`」写成「路径数 `576`」；
- **成因**：★★★ 论文的原句**自带限定**（`rearrangeably` / `some set of paths` / `up to k = 48`），
  转写时**容易被当修辞删掉**；
- **规避**：★★ 本文的 `D.5`/`D.6` 把限定词**逐字钉住**（★ 这是 v1.7 那条硬伤的直接对策）。

### 4.5 ★★★ 「规模」对「接线」不敏感（本文的守卫实验）

- **现象**：把**核心到 pod 的连线剪断一半**，**节点计数一字不变**；
- **成因**：★★★ 节点计数只由**端口预算**决定，与「**谁连谁**」无关；
- **本机实测**（`code/` 的 `D.4`）：`corrupt="half_pods"` 下 ① 四类节点数**完全相同**、
  ② **核心层的度**从 `k` 掉到 `k/2`、③ **跨半区的等价路径数**从 `16` 变成 `0`；
- **规避**：★★★ **判据必须覆盖「内容」而不只是「量」** —— 本文的 `A.2`（路径数）与
  `D.4`（度）就是那两条**敏感**的判据；★ 这与 `collectives` 的
  「**搬运量对块索引不敏感**」是**同型**的判据设计事实。

## 5. 判据表

| 可机械判定的条件 | 结论 |
|---|---|
| 见到 `k-ary fat-tree` 的规模公式 | ★ 主机 `k³/4` · 每层 `k×(k/2)` · 核心 `(k/2)²`（`A.1`，四档枚举对拍） |
| 见到「核心交换机有几台」 | ★ **`(k/2)²`** = `k²/4`（`A.1`）；★★ 而 `k/2` **是 pod 内的层内交换机数**（同值不同指，`C.1`） |
| 见到「两台主机之间有几条等价路径」 | ★★ **`(k/2)²`**（= 核心台数，`A.2`）；★ 若只给 `576`，**先问是哪一档 `k`**（`576` 是 `k = 48`） |
| 见到「`576`」 | ★ `(48/2)²` ⇒ **`k = 48` 的实例值**，不是通式（`A.2`） |
| 见到「胖树是无阻塞的」 | ★★★ 原文是 **`rearrangeably` non-blocking** + `some set of paths`（`D.5`）—— **可重排**，不是「怎么走都不阻塞」 |
| 见到「1:1 oversubscription」 | ★★ 结构上成立（`B.2`：跨切口链路 = 一侧主机数）；★★ **但论文自己说实践上 `may be difficult`**（`D.6`） |
| 见到「`k³/4`」但没带上限 | ★★ 论文**自设的讨论范围**是 **`up to k = 48`** —— ★ **不是公式的有效性上界**（★ `specs.py` 的口径：论文自设的上限，不是物理上限）（`D.5`） |
| 见到 `(k/2)2` 或 `k3 /4` | ★ 那是 **PDF 提取形态**（上标压平）—— 原文是 `(k/2)²` 与 `k³/4`（`D.5`） |
| 见到「胖树是无阻塞的 ⟹ 随便走都满带宽」 | ★★★ **不成立**：`rearrangeably` + `some set of paths`（`D.5`） |
| 见到裸写的 `stage` / 阶段 | ★★★ **不写** —— 本主题一律「三层 Clos」+ 英文 `tier`（`B.3`） |
| 报「bisection」 | ★★ 本文只给**定义**与**一种显式切法**的计数；★★ **通用推导在 `interconnect/02`**（`B.2`） |

## 6. 小结与延伸

**一句话**：**`k` 端口的交换机接成三层 Clos，就得到 `k³/4` 台主机、`(k/2)²` 台核心，
以及任意两台异 pod 主机之间的 `(k/2)²` 条等价路径 —— 这三个数都能从拓扑上数出来。**

**三条可独立复用的结论**：

1. ★★★ **主机数是两次端口预算的结果**（`k/2` 台边缘 × 每台 `k/2` 个下行端口 × `k` 个 pod），
   ★ 核心数是**反向**算的（聚合层的上行端口总数 ÷ 每台核心的 pod 数）；
2. ★★★ **路径数 = 核心台数**，因为**每台核心都连着每一个 pod** ⇒ 选核心就唯一确定路径
   （★ 这条前提被破坏，路径数公式立刻失效）；
3. ★★ **限定语必须一起抄**：`rearrangeably` · `some set of paths` · `up to k = 48` ·
   `in practice may be difficult`（★ 少一个，结论就变了）。

**延伸**：

- ★★ **同一批硬件、换一种布线：rail-optimized** → [02-rail-optimized](02-rail-optimized.md)；
- ★★ **以太网上的 RDMA（RoCE v2 / PFC / ECN）与在网计算（SHARP）** →
  [03-roce-and-infiniband](03-roce-and-infiniband.md)；
- **本层不写、只回链的正文**：
  · ★★★ **`bisection(ring)` / 全互联的 `∝N²` / NVSwitch / NVLink 域逐代** →
    `20-gpu-architecture/interconnect/02-topology-and-bisection.md`；
  · ★★ **bisection 下界 `t ≥ 2S/B`** → `20-gpu-architecture/interconnect/03-collectives-on-topology.md`；
  · **链路带宽的三种口径** → `20-gpu-architecture/interconnect/01-bandwidth-conventions.md`。

## 参考

- **Al-Fares, Loukissas, Vahdat, "A Scalable, Commodity Data Center Network Architecture",
  SIGCOMM 2008** —— ★★★ **`[C]` 级（同行评审：是）**。
  载体 = 一手 PDF `https://cseweb.ucsd.edu/~vahdat/papers/sigcomm08.pdf`
  （本轮 `curl --max-time 25` 实测 **`991601` 字节**，12 页），抓取日 **2026-10-08**。
  ★★ 本文用到的逐字引文与**它们的提取形态**全部注册在 `code/specs.py` 的 `# 来源：` 里。
  ★★ **页码 `63–74` 本轮已独立复核**（`api.crossref.org` 对该 DOI 返回 `200`，`page = "63-74"`）。
  ★ **订正记录**：早前写「(`api.crossref.org` 未通 ⇒ 未能独立核实)」是**把一次失败的网络请求
  （环境产物）当成了事实** —— ★ 与 v1.7 的「读不到当不存在」**同族**。
- ★ **C. Clos, "A Study of Non-blocking Switching Networks", Bell System Technical Journal, 32(2), 1953**
  与 **C. E. Leiserson, "Fat-Trees: Universal Networks for Hardware-Efficient Supercomputing",
  IEEE Transactions on Computers, 34(10):892–901, 1985** —— ★★ **两条书目载体本轮【逐字读到】**
  （它们在 SIGCOMM 2008 论文自己的参考文献里），★ 但**正文都未取到** ⇒ **只引载体**（计划 §C.5 的 `F6`）。
- 本仓库 `20-gpu-architecture/interconnect/02-topology-and-bisection.md`（bisection 的通用推导）、
  `03-collectives-on-topology.md`（拓扑受限的集合通信）、
  [02-rail-optimized](02-rail-optimized.md)、[03-roce-and-infiniband](03-roce-and-infiniband.md)。

★ **一条术语说明**：本文一律写「**fat-tree**」或「**三层 Clos**」**全称**，不裸写 `tree`
（库内 `tree` 另有 **AST / 计算图** 与 **tree allreduce** 两个含义）；★ `rail` 是本文档体系里的**新词**
（无上游让渡），**首次出现处**已给定义，词条将在本域的收尾 chunk 补进 `00-meta/glossary.md`。

## 编译与运行

```sh
cd code && make run                 # 三个编号程序，任一断言失败即非零退出
python3 01_fat_tree_and_clos.py     # 只跑本文的配套程序
make 01_fat_tree_and_clos           # ★★ 与上一行等价（都先走 check-env）
```

★ 本机的运行条件：`python3 3.13.5`，macOS arm64，**无 GPU、无 NCCL、无网卡**；
★★ 本目录的编号程序**只用标准库**（不需要 `numpy`）。
★★★ **本文的程序【有】A 类**：`A.1`（规模）与 `A.2`（等价路径数 + `k = 48` 的论文实例值）——
★★ 它守的是「**拓扑的计数**」与「**错公式真的会被拒**」，**不守**任何真实网络的性能。

---
title: 拓扑与 bisection：为什么每卡带宽相同、集合通信却差 4 倍
slug: interconnect-topology-and-bisection
domain: 20-gpu-architecture
topic: interconnect
level: 2
difficulty: intermediate
prerequisites:
  - 20-gpu-architecture/interconnect/01-bandwidth-conventions
tags: [topology, bisection, nvswitch, ring, nvlink-domain, path-uniformity]
related:
  - 20-gpu-architecture/interconnect/03-collectives-on-topology
  - 110-communication/collectives
  - 100-parallelism/data-parallel
code:
  - code/02_topology_bisection.py
status: stable
updated: 2026-09-15
---

# 拓扑与 bisection：为什么每卡带宽相同、集合通信却差 4 倍

> 两张卡之间的可用带宽由**链路**决定，而一次集合通信能跑多快由**拓扑的最窄切口**决定。
> 把 8 张 900 GB/s 的卡接成 ring，切口只有 0.9 TB/s；接成 NVSwitch 全互联，切口是
> 3.6 TB/s——**同样 18 条链路，差 4 倍。**
>
> **全文的数字口径**：本文的数字分两类——**规格值**（直接引自 NVIDIA 官方文档，见脚注）
> 与**推导值**（由 `code/specs.py` 中的规格常量代入 §1 的公式算出）。**本文无实测数据。**
> 要复核，运行 `code/02_topology_bisection.py`，它会打印出本文引用的每一个数。

## 0. 背景与动机

一个 8 卡节点上跑张量并行，每张卡的 NVLink 规格都是 **900 GB/s 双向**——
无论这 8 张卡是接成环形、接成 mesh、还是全部接到 NVSwitch 上。而它们的集合通信性能
可以差好几倍。常见的归因是「NCCL 没调好」「拓扑检测没开」，于是花很多时间去调环境变量。

真正的原因不在软件，也不在**任何一对 GPU 之间**的带宽上，而在**网络的形状**上。
把 8 个 GPU 切成 4 + 4 两半，跨越这条切口的**总容量**叫 **bisection bandwidth**。
三种拓扑在每卡带宽完全相同的前提下，bisection 差出 4 倍：

| 拓扑（同样 18 条链路/GPU，H100） | bisection | 来历 |
|---|---|---|
| ring | **0.9 TB/s** | 推导值（§1 第一步） |
| 直连全互联（每对 GPU 间 k 条链路） | 2.06 TB/s | 推导值（§1 第一步） |
| NVSwitch 全互联 | **3.6 TB/s** | **官方直述**（HGX H100 规格） |

这张表的最后一行是本文最重要的一个数：**3.6 TB/s 不是推导出来的，是 NVIDIA 直接给的**。
而它和第一行的 0.9 TB/s 一比就是 4 倍差距——这就是「每卡带宽一样、集合通信差 4 倍」
的全部来源。本文负责把这个差距算清楚，并指出算它的过程中最容易多算一倍的那一步。

## 1. 原理

先固定三个符号，本文所有推导都用它们：

```text
N = GPU 数          L = 每 GPU 的链路数          b = 每链路的双向带宽
bisection = 把网络切成两半（每半 N/2 个 GPU）时，跨越切口的链路总容量
```

以及一条**记账规则**（它是定义，不是来源）：**一条链路 = 一个端口对**。
GPU_i 分给 GPU_{i+1} 的 9 个端口，与 GPU_{i+1} 分给 GPU_i 的 9 个端口，
配对成的是 **9 条链路**，不是 18 条。这条规则会在 §1 第四步决定成败。

### 第一步：两条推导——ring 的 bisection 与 N 无关，全互联的 ∝ N²

**推导 A：ring。** 每个 GPU 的 L 条链路平分给 2 个邻居 ⟹ 每跳 `L/2` 条链路。
把 ring 切成两半，恰好要切断 2 跳，所以：

```text
crossing_links(ring) = 2 跳 × L/2 = L 条          ← **与 N 无关**
bisection(ring)      = L × b = 2 × (每跳带宽)
```

「2 × 每跳带宽」就是教科书里 ring 的 bisection 公式；`L/2` 这个因子在这里、
与 `2` 这个因子在分子上恰好抵消，**结果是「用满 L 条链路」而不是「2L 条」**。
记住这一步，第四步要回来用它。

**推导 B：直连全互联**（每对 GPU 之间 k 条链路）。切成 N/2 与 N/2，
跨切口的 GPU 对数是 `(N/2)²`，每对 k 条链路：

```text
crossing_links(mesh) = (N/2)² × k = N²/4 × k
bisection(mesh)      = N²/4 × k × b                ← **随 N² 增长**
```

当每对恰好 1 条链路（k=1）时，退化成最常见的那个形式：`N²/4` 条链路。

**这两条合起来是本文的核心结论：ring 的 bisection 是一个与 N 无关的常数，
而全互联随 N² 增长。** 并且注意——ring 的这条常数**不会因为加卡而变好**：
把 8 卡 ring 扩到 64 卡 ring，切口还是那 L 条链路。这就是后一篇要给集合通信
定下界的原因。

### 第二步：两个官方直述的数字，以及它们为什么互相印证

NVIDIA 在 HGX 平台对比表里**直接给出**整机（8 GPU）的 bisection bandwidth。
这两个数是本文可信度最高的输入——它们是直述，不是推导：

| 系统 | 每 GPU 双向 | 聚合端口（8 × 每 GPU） | 官方 bisection | 比值 |
|---|---|---|---|---|
| HGX A100 8-GPU | 600 GB/s | 4.8 TB/s | **2.4 TB/s** | 0.5000 |
| HGX H100 8-GPU | 900 GB/s | 7.2 TB/s | **3.6 TB/s** | 0.5000 |

**一致性自检**（本程序复算，不是抄结论）：两组数字的 bisection ÷ 聚合都**恰好是 0.5**。
一个无阻塞全互联结构被均分成两半时，任意一半的收发容量正好是全网聚合容量的一半——
两条**独立记录**都落在 0.5 上，说明「聚合端口」与「bisection」这两列是在同一套口径
（**双向合计**）下的一致读数。

反过来说：**如果哪天看到某系统的 bisection ≠ 聚合/2，先怀疑口径，而不是先相信
「这个拓扑很特别」。** 这条自检的价值就在于此——它把一个「需要查资料」的问题
变成了一个「纸笔能验」的问题。

另外把 NVSwitch 的颗数记清楚（它是官方值，且极容易被抄错）：
HGX A100 = **6 颗 gen2**，HGX H100 = **4 颗 gen3**。
⚠ DGX H100 用户指南里有一句把「4 颗 NVSwitch」与「4th gen NVLink」混写——
**那个 4 指 NVSwitch 颗数，不是 NVLink 条数**（H100 每 GPU 是 18 条）。

### 第三步：同一份链路预算下，NVSwitch 的 bisection 是 ring 的 4 倍

现在把两种拓扑放在**同一份 18 条/GPU 的链路预算**下比。这是公平比较的前提：
如果给 ring 和给 NVSwitch 的链路数不同，比出来的倍数就没有意义。

**NVSwitch（crossbar）**：每个 GPU 的 18 条链路全部接到交换机上，交换机可以把它
送到**任意**目的 GPU。于是每半个网络的收发容量都能跨切口：

```text
bisection(crossbar) = (N/2) × (每 GPU 双向) = (N/2) × L × b
```

**ring**：只有切口两侧的那 2 个 GPU 能跨切口发送，即

```text
bisection(ring) = L × b
```

两者之比 = **N/2**。N = 8 时是 **4 倍**。逐代代入（推导值）：

| 平台 | 每 GPU 双向 | NVSwitch bisection | ring bisection | 比值 |
|---|---|---|---|---|
| H100（NVLink 4，18 条） | 900 GB/s | 4 × 900 = **3.6 TB/s** | 18 × 50 = **0.90 TB/s** | 4.0 |
| A100（NVLink 3，12 条） | 600 GB/s | 4 × 600 = **2.4 TB/s** | 12 × 50 = **0.60 TB/s** | 4.0 |

两行的 NVSwitch 一列都与官方直述值逐位吻合，这就是第一步那两个官方数字的用处：
**它们同时验证了公式和链路预算。**

夹在中间的第三个选项是**直连全互联**（每对 GPU 之间 `k = L/(N-1)` 条链路）：

```text
k = 18 / 7 = 2.571 条/对     ← 不整除，这本身就说明直连 mesh 在这份预算下很别扭
bisection(mesh) = 4² × 2.571 × 50 GB/s = 2.06 TB/s
```

它比 ring 好（2.06 vs 0.90），却**不如 NVSwitch**（3.60）——因为直连 mesh 的每条链路
被硬绑给一个对端，而 crossbar 的每条链路可以到达任意对端。
**这条对比揭示的是比 bisection 更本质的一件事：crossbar 并没有增加链路数，
它增加的是「可达性」——每 GPU 的 18 条链路从「固定伙伴」变成了「任意伙伴」。**

### 第四步：ring 的 bisection 极易多算一倍，错在哪

一个很有诱惑力的错误算法是：

```text
「切 2 处 × 每处 18 条链路 × 50 GB/s = 2 × 18 × 50 = 1800 GB/s = 1.8 TB/s」
```

它把**同一条链路数了两遍**。逐项看：

```text
每个 GPU 的 18 条链路平分给 2 个邻居 ⟹ 每跳 9 条
GPU_i 分给 GPU_{i+1} 的是 9 个端口；GPU_{i+1} 分给 GPU_i 的也是 9 个端口
这两组端口配对成 9 条**点对点链路**（每条链路两端各占一个端口）
所以「9 + 9」是端口数，不是链路数——链路数要除以 2
```

正确的账（推导值）：

```text
总链路数  = 8 GPU × 18 条 ÷ 2（每条链路两端） = 72 条
跨切口    = 2 跳 × 9 条 = 18 条
bisection = 18 条 × 50 GB/s = 900 GB/s = 0.90 TB/s
```

若错误算法要成立，每跳必须有 18 条链路，那每个 GPU 就得有 `18 × 2 = 36` 个端口——
**H100 只有 18 条**。多出来的那一倍没有物理出处。

⚠ 这条陷阱值得单独写出来，是因为 **1.8 TB/s 这个错值看起来很干净**
（1800 = 18 × 100，而 100 恰好是 NVLink 5 的每 link 双向带宽），
它不会在量纲上暴露自己，只会让「NVSwitch 是 ring 的几倍」这个结论**错一倍**：
写成 2 倍而不是 4 倍。

## 2. 执行映射

**一次集合通信在 NVSwitch 系统上实际走的路。** 以 ring all-reduce 为例：NCCL 会先在
每张 GPU 上把梯度切成 N 份，然后让每个 rank 只与**两个邻居**交换——一个发、一个收。
关键在于「邻居」是**逻辑**的：数据从 GPU0 到 GPU1 要经过 **NVSwitch**，
而 NVSwitch 是**全非阻塞**的（官方原文：*"Each NVSwitch is a fully non-blocking switch
that fully connects all eight H100 Tensor Core GPU."*）。于是逻辑上的两跳通信，
在物理上只占用 GPU0 自己的 18 条链路——**这正是 bisection 从 0.90 涨到 3.60 的物理机制**。

**hybrid cube-mesh 与全互联的差别。** DGX-1 的 8 卡用 4 条 NVLink 接成 cube-mesh：
每个 GPU **直连 4 个对端**（4/7 直连），剩下 **3 个对端需要一跳中转**（3/7）。
所以「8 卡都插在一个机器里」并不蕴含任何一对 GPU 都等距。这个差别是可以量化的：

| 拓扑（8 GPU） | 距离 1 跳的对数 | 最远距离 | 任意对的可用带宽 |
|---|---|---|---|
| NVSwitch 全互联 | 28/28 | 1（经交换机） | 恒为 900 GB/s |
| ring（18 条链路/GPU） | 8/28 | 4 跳 | 相邻 450，最远 112 GB/s |

ring 那一行的「可用带宽 = 450 ÷ 跳数」是推导值：相邻对之间是 9 条链路 × 50 GB/s = 450 GB/s
（双向），跨 2 跳就要与别的流量共享沿途链路。8 卡 ring 的平均跳数是 **2.286**
（8 对 1 跳、8 对 2 跳、8 对 3 跳、4 对 4 跳，共 28 对）。

**这引出本文第二个结论：bisection 只回答「最坏的那一刀有多宽」，
不回答「任意一对 GPU 之间有多宽」。** 后者叫**路径均匀性**，
它才是 NVSwitch 与 ring 之间更本质的差别：NVSwitch 下每一对都是 900 GB/s，
ring 下只有 28.6% 的对能拿到 450，最远的 14.3% 只有 112。

**数值量级**（三个可用于心算的锚点）：

| 量 | 数值 | 来历 |
|---|---|---|
| HGX H100 的 bisection | **3.6 TB/s** | 官方直述（8 × 900 ÷ 2） |
| 同链路预算 ring 的 bisection | **0.90 TB/s** | 18 × 50 GB/s（推导值） |
| ring(8) 上任意对的平均跳数 | **2.286 跳** | 距离分布算出（推导值） |

## 3. 实现与演进

**PCIe-only（无 NVLink）。** 卡间通信全部经 PCIe：受 `63.02 GB/s`（Gen5 x16 载荷）
与 Root Complex 的约束，而且要过 host 内存。8 卡的 bisection 由 PCIe 交换结构决定，
**远远小于**任何 NVLink 方案——这也是「8 卡训练必须上 NVLink」的量化依据。

**hybrid cube-mesh（DGX-1，P100/V100）。** 用 4 条 NVLink/GPU 接成 cube-mesh，
比 PCIe 快一个量级，但**它不是全互联**（§2 的表）。这一代**没有 NVSwitch**，
也**没有「NVLink 域」这个词**。

**NVSwitch gen1（DGX-2，V100）。** 18 端口 18×18 全交叉（官方原文：
*"18-port, NVLink switch with an 18×18-port fully-connected crossbar"*），
总容量 18 × 50 = **900 GB/s 双向**。这一步的意义是**把 bisection 从「与拓扑相关」
变成了「与端口数相关」**：域内任意一对都是等距的。

**NVSwitch gen2/gen3（A100/H100 机内）。** A100 用 6 颗 gen2（每 GPU 对每颗 switch
2 条 NVLink，2 × 6 = 12 条），H100 用 4 颗 gen3（每 GPU 18 条分到 4 颗）。
bisection 从 2.4 涨到 3.6 TB/s——**涨幅来自每 GPU 带宽（600 → 900），
而不是拓扑变化**（两者都是 bisection = 聚合/2）。

**NVSwitch 级联（跨机架的 NVLink）。** 到 H100，把 NVSwitch 级联出去可以做
**256 GPU** 的 NVLink 域；GB200 NVL72 做到**机架级 72 GPU**（9 个 switch tray，
72 GPU 全 all-to-all）。**域从 8 涨到 256、涨到 72，靠的是把交换结构接出去，
不是靠加 GPU**——这是本节最重要的一句话。

**NVLink 域逐代**（规格值）：

| 平台 | NVLink 域（GPU 数） |
|---|---|
| DGX-1 (P100/V100) | 8（**但非全互联**，hybrid cube-mesh） |
| DGX-2 (V100 + NVSwitch gen1) | 16 |
| HGX A100 8-GPU | 8 |
| HGX H100 8-GPU | 8；经 NVLink Switch System 最多 **256** |
| GH200 NVL32 | 32 |
| GB200 NVL72 | **72**（单机架） |
| GB200 多机架 | 最多 576 |

⚠ 「NVLink 域」的官方定义是 *"A set of nodes that can communicate over NVLink."*——
**它是跨 OS 实例的概念，不是「装在一个机箱里」的意思**。这条定义直接否掉了两个
常见的错误用法（见 §4.3）。

## 4. 踩坑与失败模式

### 4.1 看到每卡带宽相同，就以为集合通信性能相同

**现象**：同一个模型在同一批 H100 上，只把机内互连从「8 卡直连 mesh」换成
「NVSwitch 全互联板」，所有卡的 NVLink 规格一个字都没变，但 all-reduce 时间降了一半。

**成因**：每卡带宽是**链路属性**，集合通信速度是**拓扑属性**。链路属性和拓扑属性之间
只在一处挂钩——**bisection**。同一份 18 条链路的预算下，ring 的 bisection 是 0.90 TB/s，
NVSwitch 是 3.60 TB/s（§1 第三步），差 4 倍。§0 那张表的三行就是全部答案。

**规避**：评估一个系统的集合通信能力时，第一个该查的数不是「每卡多少 GB/s」，
而是「**bisection 是多少，以及它是官方直述还是自己推的**」。如果是自己推的，
把 §1 第四步那个「9 + 9 = 18 条链路」的陷阱再走一遍——这种错最终一定表现为
「倍数差一倍」而不是「量纲不对」，编译器抓不住它。

### 4.2 把「域」当成可以无限扩展的东西

**现象**：规划一个 256 卡的训练任务时，因为「H100 的 NVLink 域可以到 256」，
就按机内的 900 GB/s 去估跨节点通信，结果实际性能差一个量级。

**成因**：域大小是**交换结构能接多远**的属性，不是 GPU 的属性。256 GPU 的域需要
**NVLink Switch System**（把 NVSwitch 级联出去）才成立；没有这套东西时，
HGX H100 的域就是 **8 GPU**，跨节点走的是 InfiniBand，每 GPU 只有 **50 GB/s 单向**——
与 NVLink 的 450 单向差 **9 倍**（同口径）。

**规避**：把「域」当成一个**需要查证的部署事实**，而不是一张卡的能力表。
两个具体的反例可以直接用来校准直觉：**A100 SuperPOD 的 256 卡不是 NVLink 域**
（官方表格里 256 A100 pod 的 NVLink Domain 仍是 8 GPU，跨节点走 IB）；
到 H100 才有 256 GPU 的域。**同样 256 张卡，一个是 IB 域、一个是 NVLink 域，
这是两代产品的差别，不是规模的自然延伸。**

### 4.3 把「NVLink 域」用在 V100 时代

**现象**：一篇讲 DGX-1 的资料写「DGX-1 的 8 卡构成一个 NVLink 域」。
这句话在术语和拓扑两个层面都不成立。

**成因**：**「NVLink Domain」这个词在 V100 时代并不存在**——它是随 NVSwitch
与跨 OS 实例的用法一起出现的；更实质的是 DGX-1 的 8 卡是 hybrid cube-mesh，
4/7 直连、3/7 需一跳（§2），**不是全互联**。把「8 张卡插在一起」等同于「一个域」，
会让路径均匀性的推导整体失效。

**规避**：用词的判据是**「任意两卡是否等距」**而不是「是否在同一机箱」。
DGX-1 用 cube-mesh 这个词，DGX-2 之后才用域。

### 4.4 把整机聚合带宽当成 bisection（或反过来）

**现象**：一份容量规划里写「NVL72 的 bisection 是 130 TB/s」，于是所有跨节点
推导都按 130 TB/s 算，结果快了 18 倍。

**成因**：130 TB/s 是 **72 × 1.8 TB/s**，即**全端口聚合**，不是 bisection。
推导值可以自检这一点：`130 ÷ 72 = 1.8056 TB/s`，恰好等于**每 GPU 的 NVLink 5 带宽**——
这说明 130 就是「每卡带宽 × 卡数」，它是一个求和，不是一条切口。
而且 **NVIDIA 根本没有公布 NVL72 的 bisection**，要算只能自己推导并标注。
更糟的是 NVL72 有**两套并行的网络**：scale-up（NVLink）130 TB/s、
scale-out（72 × 800 Gb/s）= **7.2 TB/s**，两者相差 **18 倍**。

**规避**：见到一个「整机 TB/s」级别的数字，先问三个问题：**是聚合还是切口？
是双向还是单向？是 scale-up 还是 scale-out？** 三个问题的答案都不在数字里，
只能从上下文拿。

## 编译与运行

进入 `code/` 执行 `make run` 即可复现本文全部数字：

```bash
cd code
make run                      # 依次运行 01~04 四个程序
make 02_topology_bisection    # 只跑本文对应的程序（省略 .py 后缀）
```

配套程序是 **`code/02_topology_bisection.py`**（Python ≥ 3.10 + numpy ≥ 2.0，
纯组合推导，不需要 GPU）。第 1 节打印 ring 与全互联两条 bisection 公式及逐 N 的对比，
第 2 节打印官方两个 bisection 与「恰好一半」的自检，第 3 节打印同链路预算下三种拓扑
的 bisection，第 4 节逐项拆解「多算一倍」那个错误，第 5 节打印 ring(8) 的距离分布。
全部规格常量集中在 `code/specs.py`，每条带 `# 来源：` 与口径注释。

## 6. 小结与延伸

本文的主线是：**每卡带宽是链路属性，集合通信速度是拓扑属性，两者之间只有一个挂钩——
bisection。** 围绕它有三个可独立复用的中间结论：

1. **两条公式**：`bisection(ring) = L × b`（**与 N 无关**）、
   `bisection(mesh) = N²/4 × k × b`（**∝ N²**）。ring 的切口不会因为加卡而变宽，
   这是下一篇给集合通信定下界的依据。
2. **官方两个数字互相印证**：HGX A100 2.4 TB/s 与 HGX H100 3.6 TB/s 都恰好等于
   各自聚合端口带宽的一半。这个 0.5 的自检把「查资料」变成了「纸笔可验」。
3. **同链路预算下 ring 的 bisection 是 NVSwitch 的 1/4**（0.90 vs 3.60 TB/s），
   而更本质的差别是**路径均匀性**：NVSwitch 下任意一对都是 900 GB/s，
   ring 下只有相邻的 28.6% 能拿到 450。

下一篇 [03-collectives-on-topology](03-collectives-on-topology.md) 把 bisection 变成
集合通信的**时间下界**（`2S/B`），并回答一个 bisection 本身回答不了的问题：
既然 8 卡 NVSwitch 的 bisection 有 3.6 TB/s，为什么 all-reduce 还是被每卡 450 GB/s 卡住？
[01-bandwidth-conventions](01-bandwidth-conventions.md) 提供了本文用到的所有带宽口径，
[04-measuring-bandwidth](04-measuring-bandwidth.md) 回答这些拓扑数字离可达值有多远。

## 参考

- NVIDIA, *NVIDIA HGX 平台对比材料*（HGX A100 8-GPU 与 HGX H100 8-GPU 两行）——
  "Bisection Bandwidth" 一行的 **2.4 TB/s** 与 **3.6 TB/s**，本文 §1 第二步的规格来源。
- NVIDIA, *Introducing NVIDIA HGX H100*（官方博客）—— 4 颗 gen3 NVSwitch、
  每 GPU 900 GB/s、以及全非阻塞交换机的原文。
- NVIDIA, *NVSwitch / DGX-2 架构材料* —— gen1 的 18 端口与 18×18 全交叉原文。
- NVIDIA, *NVLink 规格页* —— 逐代的每 GPU 双向带宽与每 GPU 最大链路数、
  「NVLink Domain」的官方定义（*"A set of nodes that can communicate over NVLink."*）、
  GB200 NVL72 与 Vera Rubin NVL72 的 scale-up 聚合带宽。
- NVIDIA, *DGX H100 用户指南* —— 8 张 ConnectX-7 的端口映射表（GPU:网卡 = 1:1 的硬证据）；
  注意其中把「4 颗 NVSwitch」与「4th gen NVLink」混写的一句（§1 第二步）。
- 本仓库 [20-gpu-architecture/interconnect/01-bandwidth-conventions](01-bandwidth-conventions.md)
  —— 本文用到的全部带宽口径（双向/单向、线速率/载荷）。

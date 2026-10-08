---
title: RoCE v2 与 SHARP：PFC、ECN、在网计算，以及哪些机制本库不写
slug: communication-network-topology-roce-and-infiniband
domain: 110-communication
topic: network-topology
level: 3
difficulty: intermediate
prerequisites:
  - 110-communication/network-topology/01-fat-tree-and-clos
tags: [roce, rocev2, pfc, ecn, dcqcn, sharp, infiniband, in-network-computing, physical-layer]
related:
  - 110-communication/network-topology/01-fat-tree-and-clos
  - 110-communication/network-topology/02-rail-optimized
  - 20-gpu-architecture/interconnect/01-bandwidth-conventions
  - 20-gpu-architecture/interconnect/02-topology-and-bisection
code:
  - code/03_roce_and_infiniband.py
status: stable
updated: 2026-10-08
---

# RoCE v2 与 SHARP：PFC、ECN、在网计算，以及哪些机制本库不写

> **RDMA 的传输层是为「接近无损」的链路设计的 —— 而以太网会丢包。**
> 于是有了 PFC（按优先级暂停）、ECN（端点之前就标出拥塞）、DCQCN（★ **本库只登记载体，不写机制**），
> 以及 InfiniBand 侧的 SHARP（把归约搬进交换机）。
> ★★★ **本层是【物理层】** —— 讲的是**链路与网络的机制**，不是算法的代数，也不是任何库的行为。
> ★★★ **本文件【没有 A 类】**：本机**无网卡**（无 InfiniBand / RoCE 网卡、无交换机、无 GPU），
> 「RoCEv2 真的这么封装吗」「PFC 真的把丢包消掉了吗」**无处可验** ⇒ **如实声明**，
> ★★ 并用 **运行期记账 + 调用点核对**让它**可机械复核**（★ 机制**不同于**同域 `nccl` 的源码扫描）。

## 0. 背景与动机

`01-fat-tree-and-clos` 给了**拓扑**，`02-rail-optimized` 给了**布线** —— 都是「线怎么连」。
**那本文补什么？** 补**线上跑什么**：以太网上的 RDMA（**RoCE v2**）与它在 InfiniBand 侧的对应物（**SHARP**）。

★★★ **一条动机（官方逐字，`[A]`）**：为什么要给以太网加这些机制？

> `Data Center Bridging networks employ higher layer protocols that depend on the delivery of data frames without frame loss due to congestion. These protocols were designed for an underlying transport that approaches lossless behavior and therefore do not include appropriate response to frame loss due to congestion (e.g. back-off, slow restart, etc.).`

★★★ **读法（`approaches` / `appropriate` 两个限定词）**：
**DCB 网络里跑着依赖无损交付的高层协议**，而这些协议**是为「接近无损」的底层传输设计的、
因此不含对拥塞丢包的适当响应**。
⇒ ★★ **不是**「这些协议要求无损」，也**不是**「PFC 做到了无损」——
★★★ **原句的主语是【高层协议】**（★ 只抄第二句会**把主语丢掉**，见 §4.4）。

## 1. 原理

### 1.1 ★★ RoCE v1 与 v2：唯一的差别是**网络层还是链路层**

★ 两句逐字（`[B]` 级，Cisco 产品文档）：

> `RoCEv1 is an Ethernet link layer protocol and hence allows communication between any two hosts in the same Ethernet broadcast domain.`
> `RoCEv2 is an internet layer protocol, which means that RoCEv2 packets can be routed.`

| | RoCEv1 | RoCEv2 |
|---|---|---|
| 层 | **链路层** | **网络层** |
| 能不能路由 | 不能（限同一**广播域**） | **能**（故又称 Routable RoCE） |

★★★ **本文只写到这里为止**：**RoCEv2 的帧结构**属 `F4` 的禁区（见 §1.2 与 §4.2）。

### 1.2 ★★★ `4791`：标 `[B]`，不标 `[A]`（`F4` 的现场）

> `The UDP destination port number 4791 has been reserved for RoCEv2.`

★★ **`4791` 是 UDP 的【目的端口】**、被 **reserved**（保留）给 RoCEv2 ⇒ 一个**可数的规格值**。
★★★ **但它的载体是 Cisco 的【产品文档】（`[B]`）** —— 而 RoCEv2 的**一手规范是 IBTA 的 Annex A17**，
**本库本轮没有取到**。

⇒ ★★★ **`F4` 的纪律**：**不得**把 RoCEv2 的**帧结构 / UDP 封装**标成 **`[A]`（规范）**。
★ 可写的机制句（`[B]` 级，逐字）：`It does this by encapsulating an Infiniband (IB) transport packet over Ethernet.`
（★ 原文里 `Infiniband` 的首字母是**小写 `i`** —— 照抄。）

### 1.3 ★★★ PFC：**按流量类别**的流控，四个限定词

> （定义，`[A]`）`This standard specifies protocols, procedures and managed objects that enable flow control per traffic class on IEEE 802 full-duplex links.`
> （目的与机制，`[A]`）`Priority-based flow control is intended to eliminate frame loss due to congestion. This is achieved by a mechanism similar to the IEEE 802.3x PAUSE, but operating on individual priorities.`
> （作用域，`[A]`）`Operation of priority-based flow control is limited to a domain controlled by a Data Center Bridging control protocol that controls the application of Priority-based Flow Control, Enhanced Transmission Selection, and Congestion Notification.`

★★★ **四个限定词，少一个结论就错**：

| 限定词 | 它挡住的错读 |
|---|---|
| **`per traffic class`** | 把 PFC 读成「**全链路**流控」（★ 它是**按类别**的） |
| **`full-duplex links`** | 把 PFC 读成「任何介质上的流控」 |
| **`intended to`** | 把**目的**读成**保证**（★ 官方说的是「**意在**消除拥塞丢包」） |
| **`is limited to a domain controlled by ...`** | 把 PFC 读成「**处处**有效」（★ 它**只**在被 DCB 控制协议管辖的域内） |

★★ 官方点名的**丢包原因只有一个**：`frame loss due to congestion`。
★ 本库把它落成一条**范围**（不是物理断言）：`PFC_OFFICIAL_LOSS_CAUSES = ("congestion",)`。

### 1.4 ★★ ECN：IPv4 TOS 里的**两个 bit**

> `Bits 6 and 7 in the IPv4 TOS octet are designated as the ECN field.`

★★ **两个位**、在 **IPv4 的 TOS 字节**里 —— ★★ **`designated as`（指派）**是要照抄的动词。

★★★ **ECN 干什么？** 同一份 RFC 里有两句（★ 第 5 节，本库逐字取到）：

> `The CE codepoint '11' is set by a router to indicate congestion to the end nodes.  Routers that have a packet arriving at a full queue drop the packet, just as they do in the absence of ECN.`

★★★ **第二句是限定词**：**满队列照样丢包**（`just as they do in the absence of ECN`）
⇒ ★★ **ECN 不消除丢包**，它让端点**在丢包之前得到信号**（★ 别把它读成「有了 ECN 就不丢了」）。

★★ PFC 与 ECN 的分工：**PFC 在链路层按优先级暂停**；**ECN 在网络层标出拥塞**
—— ★ 两者**互补**，不是替代（官方逐字：`In addition, PFC complements Congestion Notification in Data Center Bridging networks.`）。

### 1.5 ★★★ DCQCN：**本文只引载体（含题名 / 作者），不写机制**

★ **DCQCN** 是一篇 **SIGCOMM 2015** 的工作（`[C]` 级，同行评审：是）—— ★★ **正文本轮未取到**。
★★★ **本库【不写】它的机制**：网络上流传的「三个组件」那套描述来自**二手检索摘要**，
**不得**当逐字引文（★ 本库的 `NOT_QUOTED` 第 2 条就是它）。
⇒ ★ 本文登记**载体（含题名 / 作者 / 页码）**：`Zhu, Eran, Firestone, Guo, Lipshteyn, Liron, Padhye,
Raindel, Yahia, Zhang, Congestion Control for Large-Scale RDMA Deployments, SIGCOMM 2015, pp. 523–536`
（★ 按 `F6` 的「**只引标题 + 载体**」）。
★★ **订正记录**：早前只登记 `DCQCN, SIGCOMM 2015` 并写「标题与作者未核实（`api.crossref.org` 未通）」
—— ★ 那是**把一次失败的网络请求当成了事实**（★ 与本节下方 §参考 里 SIGCOMM 2008 的页码那条**同族**）；
本轮实测 Crossref **可达**，题名 / 作者 / 页码**已核实并登记**。

### 1.6 ★★ SHARP：在网计算的**定位**（`[B]`），且**不引性能数字**

> `NVIDIA® Scalable Hierarchical Aggregation and Reduction Protocol (SHARP)™ technology improves the performance of MPI and Machine Learning collective operation, by offloading collective operations from CPUs and GPUs to the network and eliminating the need to send data multiple times between endpoints.`

★★ **SHARP 的定位**：把集合操作**从 CPU / GPU 卸载到网络**，并**消除端点之间多次发送同一份数据**。
★★ **版本不敏感**：这句话在本库取到的**两个版本**（`SHARP UM 3.13` 与 `3.10.2`）上**逐字相同**。
★★★ **本库不引 SHARP 的任何性能数字**（`2.1×` / `3.24×` 来自**二手检索摘要**，计划 `§C.5` 的 `F6`）。
★ `code/` 的 `D.8` **扫 `specs` 的 `*_QUOTE` 引文常量**确认那两个数字**一次都不出现**。

★★ **一条回链**：SHARP 让「交换机自己会归约」这件事在**算法层**的意义（`busbw` 会高估等）在
`20-gpu-architecture/interconnect/04-measuring-bandwidth.md`（★ 上游已写）；本文**只**给它的定位。

### 1.7 ★★ RoCE 的**定位**（`[A]`，★ 但本轮未能复核该页）

> `RoCE is an industry standard transport that enables Remote Direct Memory Access (RDMA) to operate on ordinary Ethernet layer 2 and 3 networks.`

★★★ **复核状态（如实记）**：**本轮未取到该页** —— `curl` 试了 `infinibandta.org` 的
**5 个 URL**（`/rdma-over-converged-ethernet-roce-faq/` · `/faq/` · `/faqs/` · `/roce/` ·
`/about-roce/`），**没有一个取到该页**（404 ×2 · 301 ×2 · 连接失败 ×1）⇒ ★★ 这条引文**照计划
`§C.3` 转录**，**本轮未能独立复核**（★ `code/` 的 `B.1` 会把它打印成 `not-fetched`，
与 `verified` 分开显示）。★ 同站点的**新闻稿页**本轮**取到了**，但那一句是
`RoCE utilizes RDMA to ...`，**逐字不同** ⇒ **不拿它冒充**。
★★ 它只支撑**定位**（RoCE 把 RDMA 搬到普通以太网 L2/L3），★ **不构成**任何机制引用。

## 2. 执行映射

### 2.1 ★★★ 本文件为什么**没有 A 类**（以及它如何被**机械复核**）

| 角色 | 是什么 | 它为什么**不是** A 类 |
|---|---|---|
| **被测** | 规格值 / 引文的**字面** | ★★★ 本机**没有** RoCE / IB 网卡 ⇒ 没有可跑的「实现」 |
| **参照** | `specs.py` 的引文常量（`[A]`/`[B]` 混合） | ★★ 参照本身是**抄来的文本**，不是独立算法 |
| **第三方** | **无** | ★ 没有第三方 ⇒ 一切「对拍」都只是**抄写核对** |

★★★ **⇒ 本文件没有 A 类**；★★★ **不得把「常量钉住」冒充 A 类**
（`P12` 的变体：**判据看着在检查，其实只是把常量抄了一遍**）⇒ ★ **一切钉住归 `D` 类**。

★★★ **「没有 A 类」不是一句话声称，而是两条【可机械复核】的量**（`code/` 的 `D.0`）：

1. **运行期记账**：`check()` 每跑一次就把**实际执行过的类别**记进 `_TOTAL`
   ⇒ 跑完后 `_TOTAL` 的**键集合必须恰好是 `{B, C, D}`**（**没有 `A` 这个键**）；
2. **调用点核对**：用 AST 数本文件里 `check(` 的**调用点个数**，它必须**等于**执行次数（`+1` 是本条自己的调用点）
   ⇒ ★★ **连「一段从未被执行的 A 类判据（死代码）」也跑不掉**。

★★ **两条分工**：单靠 ① 会漏死代码；单靠 ② 就是同域 `nccl` 那条路线的变体。
★ **如实交代残余边界**：这两条只覆盖**本文件**的 `check()` 调用（★ 若有人把 A 类判据写进
`specs.py` 或另起文件，这里看不见 —— ★ 但那时它也不在本文件的断言集里，本文件的总结会照实报 `A 0/0`）。

### 2.2 ★★★ 一条必须写明的限定语

> ★★ 本机**无 InfiniBand / RoCE 网卡、无 GPU、无交换机**。
> ★★★ 强制限定语：**不是 RoCEv2，也不是真机** —— 本文与配套程序里的**每一个数**都**抄自官方文本**。
> ★★ 载体与抓取日（**2026-10-08**）：RFC 3168 `[A]` · IEEE 802.1Qbb 工作组页 `[A]` ·
> Cisco 产品文档 `Release 4.2` `[B]` · NVIDIA SHARP UM `3.13` `[B]`。
> ★★★ **不写版本与抓取日的引文不算引文**（`P19` 的第一道防线）。

★★ **本文的「形态」一句话**：编号程序**跑在默认 `python3` 下、只用标准库**
⇒ 落在 `CONTRIBUTING` §2 表的**第二种情形**（**程序内断言**）。
★★★ **点破一层**：§2 的第二种情形**名义上要求 A/B/C 缺一不可**，而**本文件没有 A**
⇒ 属「**第二种情形的【改写】**」（★ 不是 §2 里任何一种**原样**形态）。
★ 而「**把不可测的那一档写明**」是【另一条】纪律（**不是** §2 的任何一种编号形态）。

### 2.3 ★★ 覆盖边界

- **守**：**规格值抄对了没有**（`4791` · bit 6–7）与「**两条错法会不会被拒**」；
- **不守**：任何一条**真实 RoCE / IB 的行为**（本机**测不了**）。

## 3. 实现与演进

### 3.1 ★★★ 规格值表：**级别**与**本轮复核状态**是**两轴**

`code/` 的 `B.1` 把七条事实摆成一张表，**每条两个标签**：

| 事实 | 级别 | 本轮复核 |
|---|---|---|
| ECN 字段（IPv4 TOS 的 bit 6–7） | `[A]` | **verified** |
| PFC 按流量类别作用于全双工链路 | `[A]` | **verified** |
| RoCE 的定位（IBTA 的措辞） | `[A]` | ★★ **not-fetched** |
| RoCEv2 的 UDP 目的端口 4791 | `[B]` | verified |
| RoCEv2 可路由（网络层协议） | `[B]` | verified |
| RoCEv2 的封装（IB 传输包套以太网） | `[B]` | verified |
| SHARP 的在网计算定位 | `[B]` | verified |

★★★ **两轴【不是一回事】**：第 3 行是 `[A]` 却 **not-fetched**
—— ★★ 这说明「**级别**」（来源的**种类**：规范 / 产品文档 / 论文）与「**复核状态**」（**我这次翻到没有**）
是**两个独立的量**。★★ 把它们合并成一个标签，就会**悄悄把一条没读到的规范当成读过了**。

### 3.2 ★★★ 为什么 RoCEv2 的封装只能标 `[B]`（`F4` 的落点）

**`[A]` 的资格是「有一手规范」**，不是「这句话看起来权威」。
RoCEv2 的帧结构 / UDP 封装的一手规范是 **IBTA 的 Annex A17** —— ★ **本库本轮没有取到**。
⇒ ★★★ **只能标 `[B]`**（Cisco 产品文档）。

★ **本库把这条做成了机械断言**（★ 依 `CONTRIBUTING:238`：**拒绝侧**在 `code/` 的 `C.2`，
**放行侧**（**成功行为**）在 `B.1`）：

```
SPEC_REQUIRED_FOR      = ("RoCEv2 的封装（IB 传输包套以太网）",)   # 要标 A 就必须先有 Annex A17
HAVE_IBTA_ANNEX_A17    = False                                    # 本轮取证的实情
⇒ roce_level_ok(那条事实, "A") == False     # 必须被拒        （C.2）
⇒ roce_level_ok(那条事实, "B") == True      # 必须放行        （B.1）
⇒ roce_level_ok("ECN 字段（…）", "A") == True   # ★ 已复核的 [A] 照常放行（★ 不是「全拒」）（B.1）
```

★★★ **放行那两行很重要**：守卫**不是**「把所有 `[A]` 都拒掉」—— 它有**两个独立的输入**
（哪条事实需要规范 × 本轮有没有规范），所以它**不是**「`return False`」那种定义级恒真。

### 3.3 ★★ 与相邻主题的分工

| 话题 | 在哪 |
|---|---|
| **InfiniBand 的代际（EDR/HDR/NDR/XDR）与它们的带宽** | `20-gpu-architecture/interconnect/01-bandwidth-conventions.md` |
| **`busbw` 在 SHARP/NVLS 下会高估** | `20-gpu-architecture/interconnect/04-measuring-bandwidth.md` |
| **NVLink 域 / NVSwitch** | `20-gpu-architecture/interconnect/02-topology-and-bisection.md` |
| ★★ **胖树 / 三层 Clos** | [01-fat-tree-and-clos](01-fat-tree-and-clos.md) |
| ★★ **rail-optimized** | [02-rail-optimized](02-rail-optimized.md) |
| ★★ **RoCE v2 / PFC / ECN / SHARP 的机制与级别账** | **本文** |

## 4. 踩坑与失败模式

### 4.1 ★★★ 把 PFC 当成流控的万能解

- **现象**：写「**开了 PFC 网络就无损**」—— 读起来像行业共识，**不报错**；
- **成因**：★★★ **两个方向的越界**：
  · **跨原因**：官方只点名 `frame loss due to congestion`（一个原因）；
  · **跨类别**：PFC 是 **`per traffic class`** 的，**只保护开了它的那些类别**（不是全链路）；
  ★★★ **第三个方向**：`is limited to a domain controlled by ...` —— **出了那个域，官方不承诺**；
- **本机实测**（`code/` 的 `C.1`）：三种「万能解」声明**全部被拒**；
  ★ 而「原因 `congestion` + 类别 `{3}`」这种**在范围内**的声明**放行**（★ 成功行为，已在 `B.2` 断言）；
- ★★ **最有说服力的是「跨类别」那一条**：即使**原因没越界**，「保护全部 8 个类别」**也是错的**；
- **规避**：★★★ 把官方的**范围**做成可比的量（本库：`PFC_OFFICIAL_LOSS_CAUSES` 与「实际开着的类别」）。

### 4.2 ★★★ 把 RoCEv2 的封装标成 `[A]`（`F4`）

- **现象**：帧结构 / UDP 封装写得**很像规范级事实**（它是**协议细节**，不是营销话术）；
- **成因**：★★★ **`[A]` 的资格是「有一手规范」，不是「这话看起来权威」** ——
  RoCEv2 的一手规范（Annex A17）**本库没有取到**；
- **本机实测**（`code/` 的 `C.2`）：标 `[A]` **被拒**（★ 拒绝侧在 `C.2`）；
  ★ 而标 `[B]` **放行**、**已复核**的 `[A]`（ECN 那两个 bit）**照常放行**
  （★ 放行侧是**成功行为**，已在 `B.1` 断言）；
- **规避**：★★★ **把「需要规范」与「有没有规范」做成两个独立的输入** ——
  ★ 这样守卫既能拒绝错法，又不会把**对的**一起拒掉。

### 4.3 ★★★ 给 DCQCN / SHARP 编一个机制（本库最容易犯的一条）

- **现象**：写「DCQCN 由三个组件组成……」「SHARP 能带来 `2.1×` 的加速……」
  —— 这两句**在检索摘要里确实有**，读起来**完全像事实**；
- **成因**：★★★ 它们的来源是**二手检索摘要**，而本库**没有** DCQCN 与 SHARP 2016 的**正文**；
  ★★★ **无数字可 grep** —— 这正是本库 `no-fabricated-numbers` 的**孪生形态**（「编机制从句」）；
- **本库的做法**：把这三条**明写进** `NOT_QUOTED`（「**我们不引**」的登记处），
  ★★ 并让 `D.8` **扫 `specs` 的全部 `*_QUOTE` 常量**，确认那两个性能数字**一次都不出现**；
- **规避**：★★★ **凡写一个机制，先问「载体是哪一篇、我翻到了吗」** ——
  ★ 翻不到就**只写载体**（本文的 DCQCN **不写机制** —— ★ 只登记题名 / 作者 / 页码，机制一个字不写）。

### 4.4 ★★ 引文截断（引文层）

- **现象**：引了「为什么需要 PFC」那句，却**丢掉了主语**；
- **成因**：★★★ 那句的主语在**前一句**里（`higher layer protocols`）——
  只抄 `These protocols were designed for ...` 会让读者**以为「These protocols」是 PFC**；
- **本机实测**（`code/` 的 `D.5`）：**两句一起**逐字钉住；
- **规避**：★★ ★ **凡引一句带代词的官方文本，把它的【前一句】也抄下来**
  （★ 这与 `02` 的「引文截断」是**同族**：那边丢的是**后一句**，这边丢的是**前一句**）。

## 5. 判据表

| 可机械判定的条件 | 结论 |
|---|---|
| 见到 `RoCEv2` 的 **UDP 端口** | ★★ **`4791`**，标 **`[B]`**（Cisco）；★★★ **`F4`：不得标 `[A]`**（`B.1`/`C.2`） |
| 见到 RoCEv2 的**帧结构 / UDP 封装** | ★★★ 只能标 **`[B]`** —— 一手规范（IBTA Annex A17）**未取到**（`D.3b`/`C.2`） |
| 见到 `RoCEv1` 与 `RoCEv2` 的差别 | ★ **链路层**（限广播域） vs **网络层**（**可路由**）（`B.1`） |
| 见到「ECN 占几个 bit、在哪儿」 | ★ **bit 6–7**（**两个**），在 **IPv4 的 TOS 字节**里，`[A]`（`B.1`/`D.5`） |
| 见到「有了 ECN 就不丢包了」 | ★★★ **不成立**：`Routers ... drop the packet, just as they do in the absence of ECN`（`D.5`/`D.6`） |
| 见到「PFC 保证无损网络」 | ★★★ **不成立**：`intended to`（**意在**）+ `is limited to a domain`（`B.2`/`C.1`） |
| 见到「PFC 保护全部流量类别」 | ★★★ **不成立**：它是 **`per traffic class`**；**只**保护开了它的类别（`C.1`） |
| 见到「PFC 能对付任何丢包原因」 | ★★★ **不成立**：官方只点名 `frame loss due to congestion`（`C.1`） |
| 见到 `DCQCN` 的机制描述 | ★★ **先问载体** —— 本文**不写**（`F6`：正文未取到） |
| 见到 `SHARP` 的 `2.1×` / `3.24×` | ★★★ **不得引**（二手摘要）；★ `D.8` **扫引文常量**确认一次都没出现 |
| 见到 `SHARP` 是什么 | ★ **`[B]`**：把集合操作**卸载到网络**，**消除端点间多次发送同一份数据**（`B.3`） |
| 报「本文件没有 A 类」 | ★★★ 不是声称：`D.0` 用**运行期记账 + 调用点核对**两条机械量证明（`A 0/0`） |
| 见到裸写的 `stage` / `channel` | ★★ 本域不裸写 `stage`（库内四个含义）；★ 指链路一律写「链路」，**不写 `channel`** |

## 6. 小结与延伸

**一句话**：**RDMA 的传输假定链路「接近无损」，于是在以太网上加了 PFC（按类别暂停）与 ECN（提前标拥塞），
而 InfiniBand 侧把归约搬进了交换机（SHARP）—— 但这些机制的规格级细节，本库有两条明确不写。**

**三条可独立复用的结论**：

1. ★★★ **`[A]` 的资格是「有一手规范」，不是「看起来权威」** ——
   RoCEv2 的封装只能 `[B]`（Annex A17 未取到）；★★ 且**「级别」与「本轮复核状态」是两轴**；
2. ★★★ **PFC 有四个限定词**（`per traffic class` · `full-duplex` · `intended to` · `is limited to a domain`）
   —— ★ 少一个就会把它读成万能解；
3. ★★★ **凡写机制先问载体** —— DCQCN 与 SHARP 2016 的**正文**本库都没取到
   ⇒ **只引载体**，**不写机制、不引性能数字**（★ 并让源码扫描把这条变成**可机械复核**的）。

**延伸**：

- ★★ **胖树的组合结构** → [01-fat-tree-and-clos](01-fat-tree-and-clos.md)；
- ★★ **rail-optimized 的布线** → [02-rail-optimized](02-rail-optimized.md)；
- **本层不写、只回链的正文**：
  · ★★ **InfiniBand 的代际与带宽** → `20-gpu-architecture/interconnect/01-bandwidth-conventions.md`；
  · ★★ **`busbw` 在 SHARP / NVLS 下的高估** → `20-gpu-architecture/interconnect/04-measuring-bandwidth.md`。

## 参考

- **RFC 3168, "The Addition of Explicit Congestion Notification (ECN) to IP"** —— ★ **`[A]` 级（规范）**，
  本轮 `curl` 取到全文（`170966` 字节），抓取日 **2026-10-08**。
- **IEEE 802.1Qbb（Priority-based Flow Control）工作组页** `https://1.ieee802.org/dcb/802-1qbb/` ——
  ★★ **`[A]` 级（规范**摘要**）**，本轮 `curl` 取到（`64780` 字节）。
  ★ **不是**标准正文 PDF（那要订阅）—— ★ 本文的 PFC 引文全部来自这一页。
- **Cisco UCS C-Series IMC Configuration Guide for RoCEv2, Release 4.2** —— ★★ **`[B]` 级**，
  `.../guide/4_2/b-roce-configuration-cimc-4-2/m-rdma-over-converged-ethernet-roce-version-2.html`，
  本轮 `curl` 取到（`40982` 字节）。
- **NVIDIA, SHARP User Manual（`SHARP UM 3.13`，另一版 `3.10.2`）** —— ★★ **`[B]` 级**，
  `https://networking-docs.nvidia.com/sharpum/3130/introduction`，本轮取到。
- ★ **`[C]` 级（只引载体）**：**SHARP, COMHPC 2016**（正文未取到）·
  **Zhu et al., "Congestion Control for Large-Scale RDMA Deployments", SIGCOMM 2015, pp. 523–536**
  （正文未取到；★ 题名 / 作者 / 页码本轮由 `api.crossref.org` 复核并登记）。
- ★★★ **三条【不引】的登记**（`code/specs.py` 的 `NOT_QUOTED`）：
  SHARP 的 `2.1×` / `3.24×` 性能数字（二手检索摘要）· DCQCN 的三组件机制描述（二手检索摘要）·
  RoCEv2 的帧结构（IBTA Annex A17 一手规范未取到）。
- ★★ **一条未能复核的引文**：`RoCE is an industry standard transport ...`（IBTA 问答页）——
  ★ **本轮 `curl` 试了 5 个 URL 未取到该页** ⇒ **照计划 `§C.3` 转录**，`B.1` 里标 `not-fetched`。
- 本仓库 `20-gpu-architecture/interconnect/01-bandwidth-conventions.md`、
  `04-measuring-bandwidth.md`、[01-fat-tree-and-clos](01-fat-tree-and-clos.md)、
  [02-rail-optimized](02-rail-optimized.md)。

★ **一条术语说明**：本文不裸写 `stage`（库内四个含义）与 `channel`（库内指量化的 per-channel）。

## 编译与运行

```sh
cd code && make run                 # 三个编号程序，任一断言失败即非零退出
python3 03_roce_and_infiniband.py   # 只跑本文的配套程序
make 03_roce_and_infiniband         # ★★ 与上一行等价（都先走 check-env）
```

★ 本机的运行条件：`python3 3.13.5`，macOS arm64，**无 GPU、无 NCCL、无网卡**；
★★ 本目录的编号程序**只用标准库**（不需要 `numpy`）。
★★★ **本文的程序【没有 A 类】**（`D.0` 机械证明：运行期记账 + 调用点核对）——
★★ 它守的是「**规格值抄对了没有**」与「**两条错法真的会被拒**」，
**不守**任何一条真实 RoCE / IB 的行为。
★ 强制限定语：**不是 RoCEv2，也不是真机**。

"""110-communication/network-topology 的外部事实常量源。

★ `CONTRIBUTING` §2 约定的**共享事实源**：本主题的编号程序从它读常量，
  推导过程可复核，**常量本身不可自证**。
★ **只放本主题真的会断言的常量**（无消费者的常量是 `make mutate` 的漏网候选）。
★ **来源分级 A–E**（A 厂商规范 / B 产品官方文档 / C 论文 / D 社区 / E 本库推导）；
  每条带 `# 来源：` + `[A-E]` 级 + **抓取日** + `# 口径：` + `# 消费者：` + `# 同值位置：`。
★★ **本域的头号错误模式是「把 B 级当 A 级」**（`nccl` 主题整个靠产品文档）。
★★ **所有数值常量取到 `< 2^53`**；**幂次写成十进制字面量**（`mutate.py` 的 `_walk()` 不下钻 `BinOp`）。
★★ **每个元组/列表常量都要有一条【逐元素】`==` 断言**（`mutate.py` 盲区⑧：只钉「整体」会漏元素级变异）。

★★★ **本主题的【A 类锚点】与 `nccl` 相反**：`nccl` 【没有】A 类（本机无 GPU / 无 NCCL），
  而本主题的 `01` / `02` **有真 A 类** —— 它们钉在**【组合结构】**上，与**外部工件对拍**：
    · `01`：把胖树**建成一张显式的图**再**数**（主机数 / 核心交换机数 / 等价路径数），
      与 **SIGCOMM 2008 论文的论文值**对拍（★ 外部锚点，**不是自证**）；
    · `02`：把 rail 布局**逐张卡枚举出来**，与闭式 `N/R` 对拍，并在枚举出的图上数**跳数**。
  ★★ `03`（RoCE / InfiniBand）**A 类不可得**（本机无网卡）⇒ 详见它的程序抬头。

★★★ **一条【载体读法】纪律（本文件最容易出错的地方）**：
  `01` 的载体是 **一手 PDF**（`pdftotext -layout` 提取），而**数学上标会被压平**：
    · `(k/2)²`  ⇒ 提取成 **`(k/2)2`**（上标 `2` 压平，**与前括号之间没有空格**）
    · `k³/4`    ⇒ 提取成 **`k3 /4`**（上标 `3` 压平，**`/4` 之前有一个空格**）
    · `k ∗ (k/2)2`（正文里的乘号是 `∗` U+2217，**不是 ASCII 的 `*`**）
  ★★ 且版面是**双栏**：`-layout` 会把左右两栏**交错**输出 ⇒ 一句话可能被切成两段
    （例：`There are 576 equal-cost paths between any given pair` ⏎ `of hosts in different pods.`）
    ⇒ ★ **引文是【还原后】的连续文本**，`# 来源：` 里**另记提取形态**。

★ **本主题的禁区**（一行都不重写，只回链）：
  · ★★★ **NVLink / NVSwitch / PCIe / IB 代际的带宽与拓扑** → `20-gpu-architecture/interconnect/01,02`；
  · **bisection 的通用推导** 与 **`bisection(ring) = L×b`** → `interconnect/02`（本主题只给**定义**）；
  · **bisection 下界 `t ≥ 2S/B`** · **分层 all-reduce 三段式** → `interconnect/03`；
  · **`algbw` vs `busbw` 与修正系数表** → `interconnect/04`。

★★★ **两条禁写清单（计划 §C.5，与本主题直接相关）**：
  · **`F4`** —— **不得**把 RoCEv2 的**帧结构 / UDP 封装**标成 **`[A]`（规范）**：
    IBTA 的**一手规范（Annex A17）未取到** ⇒ 只能标 **`[B]`**（Cisco 产品文档）。
    ★★ 本文件把这条**做成了机械断言**（见 `ROCE_FACT_ROWS` 与 `03` 的 `C.2`）。
  · **`F6`** —— **Bruck 1997 / DCQCN / Rabenseifner 2004 / Leiserson 1985 / Clos 1953 / SHARP 2016
    的【正文】都未取到** ⇒ **只引【标题 + 载体】（若有）**，**不得逐字引正文公式**；
    ★ 尤其 **DCQCN 的三组件描述**与 **SHARP 的 `2.1×` / `3.24×`** 来自**二手检索摘要**
      ⇒ **不得当逐字引文**（本文件登记在 `NOT_QUOTED` 里）。
"""


# ===========================================================================
# 0. 共用的抓取日
# ===========================================================================
FETCH_DATE = "2026-10-08"
# 来源：**本库自记**，不是外部事实 —— 它记的是「本文件所有 curl 的抓取日」。
# 口径：★★ 三条外部来源（SIGCOMM 2008 / NVIDIA / RFC+IEEE+Cisco）**同日抓取**。
#       ★★ 不写抓取日的引文不算引文（`P19` 的第一道防线）。
# 消费者：`01` 的 `D.3` · `02` 的 `D.3` · `03` 的 `D.3`（三个程序各钉一次）。
# 同值位置：三篇正文的抬头（「抓取日」），以及各常量的 `# 来源：` 注释。


# ===========================================================================
# 1. SIGCOMM 2008 —— [C] 论文（同行评审：是）—— ★★★ 01 的 A 类锚点
#    载体 = 一手 PDF（见 `SIGCOMM08_PDF_URL`），本轮 **curl 实测 `991601` 字节**，抓取日 2026-10-08。
#    ★★★ 本节的每条引文都必须与 `# 来源：` 里的**提取形态**对照（上标压平 / 双栏交错）。
# ===========================================================================

SIGCOMM08_CARRIER = 'Al-Fares, Loukissas, Vahdat, A Scalable, Commodity Data Center Network Architecture, SIGCOMM 2008'
# 来源：[C] 论文，**同行评审：是**（计划 §C.3 给 `10.1145/1402958.1402967`）；
#       ★★★ **标题与作者本轮从一手 PDF 的首页逐字读到**（`Mohammad Al-Fares` / `Alexander
#       Loukissas` / `Amin Vahdat`）。
#       ★★ 页码 `63–74`（计划 §C.3 给的）**本轮未能独立核实** —— PDF 的提取文本里没有页眉页码
#         ⇒ 本常量**不含页码**（记不下就不记，`no-fabricated-numbers`）。
# 口径：★★ 本常量是**书目字段**（作者 + 标题 + 会议 + 年），**不是一句引文**；
#       它只用来支撑「本主题的组合结构事实来自这篇论文」这件事本身。
# 消费者：`01` 的 `D.3`（与 `D.5` 的引文成对打印）。
# 同值位置：无。

SIGCOMM08_DOI = "10.1145/1402958.1402967"
# 来源：[C] 论文的 DOI —— ★★ **照计划 §C.3 转录**；本轮 `api.crossref.org` **未通**
#       （取到的 JSON 缺 `message`）⇒ **本轮未能独立复核**，如实记在这里。
# 口径：★ 它只用于「可定位到那一篇」，**不构成任何数值主张**。
# 消费者：`01` 的 `D.3`。
# 同值位置：无。

SIGCOMM08_PDF_URL = "https://cseweb.ucsd.edu/~vahdat/papers/sigcomm08.pdf"
# 来源：★★★ **一手 PDF**（本轮 `curl --max-time 25` 实测 `991601` 字节，12 页）。
# 口径：本文件所有 `SIGCOMM08_*_QUOTE` 的**唯一载体**。
# 消费者：`01` 的 `D.3`。
# 同值位置：无。

SIGCOMM08_KARY_QUOTE = 'We organize a k-ary fat-tree as shown in Figure 3. There are k pods, each containing two layers of k/2 switches.'
# 来源：[C] 论文 §2.2（一手 PDF，抓取日 2026-10-08）。
#       ★★ 提取形态：`We orga-` ⏎ `nize a k-ary fat-tree ...` —— **连字符跨行**
#         （照抄会把 `organize` 抄成 `orga-` + `nize`）；本引文是**还原后**的连续文本。
# 口径：★★★ **胖树的定义**：`k` 个 pod，每个 pod **两层**、每层 **`k/2` 台交换机**
#       （= 边缘层 + 聚合层）⇒ ★ 这是 `01` 里建图时「每 pod 的交换机数」的逐字依据。
#       ★★ 它**不含**主机数与核心数（那两条另有引文）。
# 消费者：`01` 的 `D.5`。
# 同值位置：无。

SIGCOMM08_CORE_QUOTE = 'There are (k/2)2 k-port core switches. Each core switch has one port connected to each of k pods.'
# 来源：[C] 论文 §2.2（一手 PDF，抓取日 2026-10-08）。
#       ★★★ 提取形态：`(k/2)2` —— **上标 `2` 被压平**（原文是 `(k/2)²`），
#         且**与前括号之间没有空格**；★ 别当成 `(k/2)2`（乘 2）—— 那是**抄错**。
# 口径：★★★ **核心交换机数 = `(k/2)²`**，且**每台核心交换机对每个 pod 各占一个端口**（`k` 个端口）
#       ⇒ ★★ 这两句合起来**唯一地确定**了核心层的连接规则，`01` 的建图**照着它**。
# 消费者：`01` 的 `D.5`。
# 同值位置：`SIGCOMM08_PATHCOUNT_QUOTE`（同一个 `(k/2)2` 形态）。

SIGCOMM08_HOSTS_QUOTE = 'In general, a fat-tree built with k-port switches supports k3 /4 hosts. In this paper, we focus on designs up to k = 48.'
# 来源：[C] 论文 §2.2（一手 PDF，抓取日 2026-10-08）。
#       ★★★ 提取形态：`k3 /4 hosts` —— **上标 `3` 被压平**（原文 `k³/4`），
#         且**`/4` 之前有一个空格**（`k3 /4` 不是 `k3/4`）；★ 同时注意句子跨行
#         （`supports` ⏎ `k3 /4 hosts`）。
# 口径：★★★ **主机数 = `k³/4`**（`k` 为偶数时整除）；★ 且**论文自己的适用上限是 `k = 48`**
#       —— ★★ 引用这条时**必须带上 `k = 48` 那个限定**（「单一实践升格」的反面）。
# 消费者：`01` 的 `D.5`。
# 同值位置：`SIGCOMM08_HOSTS_AT_K48`（`48³/4 = 27648`）。

SIGCOMM08_PATHCOUNT_QUOTE = 'there are (k/2)2 such shortest-paths between any two hosts on different pods, but only one is chosen'
# 来源：[C] 论文 §3.1（一手 PDF，抓取日 2026-10-08）。
#       ★★ 提取形态：`(k/2)2`（上标压平）；句子被双栏切成 `on differ-` ⏎ `ent pods`
#         ⇒ 本引文是**还原后**的连续文本（原文的 `shortest-paths` 是**连字符词**，照抄保留）。
# 口径：★★★ **【一般式】**：任意两台**不同 pod** 的主机之间有 `(k/2)²` 条**等代价路径**
#       （`only one is chosen` 说的是**单路径转发**，不是路径数变少）。
#       ★★ `01` 的 A 类正是**枚举出这个数**去与它**对拍**。
# 消费者：`01` 的 `D.5` 与 `A.2`（A 类的**外部锚点**）。
# 同值位置：`SIGCOMM08_CORE_QUOTE`（`(k/2)2`）· `SIGCOMM08_PATHS_AT_K48`（`k=48` 时 `576`）。

SIGCOMM08_576_QUOTE = 'There are 576 equal-cost paths between any given pair of hosts in different pods.'
# 来源：[C] 论文 §2.2（一手 PDF，抓取日 2026-10-08）。
#       ★★ 提取形态：`There are 576 equal-cost paths between any given pair` ⏎ `of hosts in
#         different pods.`（**双栏交错 + 换行**）⇒ 本引文是还原后的连续文本。
# 口径：★★★ 这是 `k = 48` 的**实例值**（`(48/2)² = 576`）—— ★ 与 `PATHCOUNT_QUOTE` 的**一般式**
#       是**同一条结论的两种写法** ⇒ ★★ 引用时**必须说清是哪一档 `k`**（否则就是把实例升格成通式）。
# 消费者：`01` 的 `D.5` 与 `A.2`（`k=48` 那一档）。
# 同值位置：`SIGCOMM08_PATHS_AT_K48`。

SIGCOMM08_INSTANCE_QUOTE = 'a fat-tree built from 48-port GigE switches would consist of 48 pods, each containing an edge layer and an aggregation layer with 24 switches each.'
# 来源：[C] 论文 §2.2（一手 PDF，抓取日 2026-10-08）。
#       ★★ 提取形态：`a fat-tree built from 48-` ⏎ `port GigE switches ...`（**连字符跨行**）。
# 口径：★★ `k = 48` 的**层内交换机数**：**48 个 pod、每层 24 台** ⇒ 与
#       `SIGCOMM08_KARY_QUOTE` 的 `k/2` 一致（`48/2 = 24`）。
#       ★★ 这给了 `01` 一条**可数的**结构性自检（不是抄结论）。
# 消费者：`01` 的 `B.1`。
# 同值位置：`SIGCOMM08_SWITCHES_PER_LAYER_PER_POD_AT_K48`。

SIGCOMM08_SUBNETS_QUOTE = 'The network supports 27,648 hosts, made up of 1,152 subnets with 24 hosts each.'
# 来源：[C] 论文 §2.2（一手 PDF，抓取日 2026-10-08）。
#       ★★ 提取形态：`The network supports 27,648 hosts, made up of 1,152 subnets with 24` ⏎
#         `hosts each.`（换行）。
# 口径：★★ 三个**实例数**：主机 `27,648` · 子网 `1,152` · 每子网 `24` 台主机。
#       ★★ **子网数 = 边缘交换机数**（论文 §2.2：同一台边缘交换机下的主机构成一个子网）
#         ⇒ `1,152 = 48 × 24 = k × (k/2)`，这是 `01` 可**枚举复核**的一条。
# 消费者：`01` 的 `B.1`。
# 同值位置：`SIGCOMM08_HOSTS_AT_K48` / `SIGCOMM08_SUBNETS_AT_K48`。

SIGCOMM08_NONBLOCKING_QUOTE = 'fat-trees are rearrangeably non-blocking, meaning that for arbitrary communication patterns, there is some set of paths that will saturate all the bandwidth available to the end hosts in the topology.'
# 来源：[C] 论文 §2.2（一手 PDF，抓取日 2026-10-08）。
#       ★★ 提取形态：`... there is some set of paths that will satu-` ⏎ `rate all the bandwidth ...`
#         （**连字符跨行**：`saturate`）。
# 口径：★★★ 原文的限定词是 **`rearrangeably`**（可重排）—— ★ **不是**「无阻塞」的全称：
#       它说的是「**存在**一组路径**(some set of paths)**能把可用带宽打满」，
#       ★★ **不是**「任意路由方式下都不阻塞」。★★ 引用**必须带 `rearrangeably` 与 `some`**。
# 消费者：`01` 的 `D.5`。
# 同值位置：无。

SIGCOMM08_OVERSUBSCRIPTION_QUOTE = 'Achieving an oversubscription ratio of 1:1 in practice may be difficult because of the need to prevent packet reordering for TCP flows.'
# 来源：[C] 论文 §2.2（一手 PDF，抓取日 2026-10-08）。
#       ★★ 提取形态：`Achieving an oversubscription ratio of 1:1 in practice may be diffi-` ⏎
#         `cult because ...`（**连字符跨行**）。
# 口径：★★★ 这是一句**限定语**：`1:1` 的**理论**性质（等价于满 bisection）与「**实践上可能很难**」
#       是**两件事** ⇒ ★★ 引用 `01` 的 `1:1` 时**必须一起带上这句**（不能只报「1:1」）。
# 消费者：`01` 的 `B.2`。
# 同值位置：无。

SIGCOMM08_TIER_QUOTE = 'A three-tiered design (see Figure 1) has a core tier in the root of the tree, an aggregation tier in the middle and an edge tier at the leaves of the tree. A two-tiered design has only the core and the edge tiers. Typically, a two-tiered design can support between 5K to 8K hosts.'
# 来源：[C] 论文 §2.1（一手 PDF，抓取日 2026-10-08）。
#       ★★ 提取形态：三句都被**双栏交错**切开（`A three-tiered design (see Figure 1) has` ⏎
#         `a core tier ...`）⇒ 本引文是还原后的连续文本。
# 口径：★★ **三层 / 两层的命名**（core / aggregation / edge 三层；两层 = core + edge）与
#       **两层的规模区间 `5K to 8K hosts`**。★ 论文自己的用词是 **`tier`**，本文件一律写
#       「**层**」并保留英文 `tier`（★ 避免与库内 `stage` 的四个含义相撞）。
# 消费者：`01` 的 `D.5` 与 `B.3`。
# 同值位置：无。

SIGCOMM08_CLOS_CARRIER = 'C. Clos. A Study of Non-blocking Switching Networks. Bell System Technical Journal, 32(2), 1953.'
# 来源：★★★ **逐字取自同一份一手 PDF 的参考文献 [11]**（抓取日 2026-10-08）。
#       ★ 计划 §C.3 说「Clos 1953 全文未取到 ⇒ 只引载体」—— ★★ **载体这一层本轮【取到了】**
#         （它在论文自己的书目里），只是**没有卷期页码之外的内容**。
#       ★★ 计划 §C.3 写的 `32(2):406–424` 里的**页码 `406–424`** 不在论文的书目里
#         ⇒ **本常量不含页码**（记不下就不记）。
# 口径：★ 它支撑「胖树是 Clos 网络的一个特例」这句话的**出处**，**不构成公式引用**（`F6`）。
# 消费者：`01` 的 `D.6`。
# 同值位置：无。

SIGCOMM08_LEISERSON_CARRIER = 'C. E. Leiserson. Fat-Trees: Universal Networks for Hardware-Efficient Supercomputing. IEEE Transactions on Computers, 34(10):892–901, 1985.'
# 来源：★★★ **逐字取自同一份一手 PDF 的参考文献 [23]**（抓取日 2026-10-08）。
#       ★★ 卷期页 `34(10):892–901` **在论文的书目里是有的**（与 Clos 那条不同）——
#          ★ 页间的短横是 **en-dash `–`**（U+2013），**不是 ASCII 连字符**。
# 口径：★ 「fat-tree」这个名字的**原始出处**（[23] 是论文正文 `called a fat-tree [23]` 的引用）。
#       ★★ 仍然**只引载体**（`F6`：正文未取到）。
# 消费者：`01` 的 `D.6`。
# 同值位置：无。

# ── 01 的数值（`k = 48` 实例，逐条可回到上面的引文）────────────────────────
SIGCOMM08_K48 = 48
# 来源：[C] 论文（`SIGCOMM08_HOSTS_QUOTE` 的 `up to k = 48`）。
# 口径：★★ **论文自设的上限**，不是物理上限。
# 消费者：`01` 的 `B.1` / `A.2`。
# 同值位置：无。
SIGCOMM08_HOSTS_AT_K48 = 27648
# 来源：[C] 论文（`SIGCOMM08_SUBNETS_QUOTE` 逐字给 `27,648 hosts`）★ 本轮已复核。
# 口径：★★ 它是 `k³/4` 在 `k = 48` 的值（`48³/4 = 110592/4 = 27648`）—— ★ 引用**两条**（引文 + 复算）。
# 消费者：`01` 的 `B.1` / `A.2`。
# 同值位置：`SIGCOMM08_HOSTS_QUOTE`。
SIGCOMM08_SUBNETS_AT_K48 = 1152
# 来源：[C] 论文（`SIGCOMM08_SUBNETS_QUOTE` 逐字给 `1,152 subnets`）★ 本轮已复核。
# 口径：★ 子网数 = 边缘交换机数 = `k × (k/2)`。
# 消费者：`01` 的 `B.1`。
# 同值位置：无。
SIGCOMM08_HOSTS_PER_EDGE_AT_K48 = 24
# 来源：[C] 论文（`SIGCOMM08_SUBNETS_QUOTE` 的 `24 hosts each` + `SIGCOMM08_INSTANCE_QUOTE`
#       的 `assigned 24 hosts each`）★ 本轮已复核。
# 口径：★ 每台边缘交换机下挂 `k/2 = 24` 台主机。
# 消费者：`01` 的 `B.1`。
# 同值位置：`SIGCOMM08_SWITCHES_PER_LAYER_PER_POD_AT_K48`（`k/2` 的两个角色，数值相同）。
SIGCOMM08_SWITCHES_PER_LAYER_PER_POD_AT_K48 = 24
# 来源：[C] 论文（`SIGCOMM08_INSTANCE_QUOTE` 逐字给 `24 switches each`）★ 本轮已复核。
# 口径：★ `k/2` 在 `k = 48` 的值 —— ★★ **与上一常量同值但不同指**（一个是「每台交换机的主机数」，
#       一个是「每层的交换机数」）：★★ 这正是「**同名不同指**」的反面形态 ——
#       **同值不同指**，所以本文件用**两个常量名**分开存放。
# 消费者：`01` 的 `B.1`。
# 同值位置：`SIGCOMM08_HOSTS_PER_EDGE_AT_K48`（同值，不同指）。
SIGCOMM08_CORES_AT_K48 = 576
# 来源：[C] 论文（`SIGCOMM08_CORE_QUOTE` 的 `(k/2)²` 在 `k = 48` 的值）★ 本轮已复核。
# 口径：★★ **核心交换机数** = `(48/2)² = 576`；★★ **与下面的路径数同值**（`576`）——
#       它们**数值相同、指的不是一回事**（一个是设备数，一个是路径数）；
#       ★★ 两者同值是**结构性的**（每条等价路径恰好由一个核心交换机决定），见 `01` 的 `A.2`。
# 消费者：`01` 的 `A.2` / `B.1`。
# 同值位置：`SIGCOMM08_PATHS_AT_K48`（同值，不同指）。
SIGCOMM08_PATHS_AT_K48 = 576
# 来源：[C] 论文（`SIGCOMM08_576_QUOTE` 逐字给 `576 equal-cost paths`）★ 本轮已复核。
# 口径：★★ **不同 pod 的主机对之间的等价路径数** = `(k/2)² = 576`。
# 消费者：`01` 的 `A.2`。
# 同值位置：`SIGCOMM08_CORES_AT_K48`（同值，不同指）。

# ── 01 / 02 的枚举档（**不是外部事实**，是本库自定的检查网格）──────────────
FATTREE_K_GRID = (4, 6, 8, 16)
# 来源：**[E] 本库推导** —— 本库自选的 `k` 档，**不是**论文的取值。
# 口径：★★ 全为偶数（`k/2` 必须是整数交换机数）；★ 覆盖 `k=4`（论文 Figure 3 的**最简实例**）
#       与多个档以验证**通式**，**不含 `k=48`**（那一档单独走 `B.1` 的实例锚点，避免枚举开销）。
# 消费者：`01` 的 `A.1` / `A.2` / `D.4`。
# 同值位置：无。
RAIL_NODES = 64
# 来源：**[E] 本库推导** —— 一个**示意性**的 rail-optimized 集群规模（`64` 个节点）。
# 口径：★ 只用于「把 `N` 张卡分到 `R` 个 rail」的可数演示，**不代表任何真实集群**。
# 消费者：`02` 的 `A.1` / `B.1`。
# 同值位置：无。
RAIL_GPUS_PER_NODE = 8
# 来源：**[E] 本库推导** —— 每节点的 GPU/NIC 数（`8` 是 NVLink 域的常见规模）。
#       ★★ 注意：`8` 这个数**在库内另有正文**（NVLink 域逐代）⇒ 本文件**不解释它**，
#       只当**输入**用（回链 `20-gpu-architecture/interconnect/02`）。
# 口径：★ 只用于 `N = RAIL_NODES × RAIL_GPUS_PER_NODE` 这个除法。
# 消费者：`02` 的 `A.1` / `B.1`。
# 同值位置：无。
RAIL_COUNT = 8
# 来源：**[E] 本库推导** —— rail 的条数。
# 口径：★★ 本库取 `R == G`（每卡一轨）这一档作为**主档**，另在 `B.2` 里演示 `R < G`。
# 消费者：`02` 的 `A.1` / `B.1` / `C.1`。
# 同值位置：无。
RAIL_SPINES = 4
# 来源：**[E] 本库推导** —— 演示用脊（spine）层交换机数。
# 口径：★ 它决定**跨 rail** 的等价路径数（`= RAIL_SPINES`），是本库的一条**结构账**。
# 消费者：`02` 的 `B.1`。
# 同值位置：无。


# ===========================================================================
# 2. rail-optimized —— [B] 级（NVIDIA 官方文档）—— ★ 02 的 A 类的**定义来源**
#    载体 = NVIDIA「IB Cluster Bring-up Procedure」站点的 `.md` 版本
#    （`curl --max-time 25` 实测 **`2574` 字节**，抓取日 2026-10-08）。
# ===========================================================================

NVIDIA_RAIL_DOC = "NVIDIA, IB Cluster Bring-up Procedure: Setting the InfiniBand Cluster Topology"
# 来源：[B] 级（**产品文档，不是规范**）；★ 站内 `.md` 版本本轮取到（`2574` 字节）。
# 口径：★★ 本主题的 rail-optimized **唯一**来源；★ 它**不是**规范 ⇒ 一切取值标 `[B]`（`P19`）。
# 消费者：`02` 的 `D.3`。
# 同值位置：无。

NVIDIA_RAIL_URL = 'https://networking-docs.nvidia.com/ibclusterbringupproc/setting-the-infiniband-cluster-topology'
# 来源：[B] —— ★ 本轮 `curl --max-time 25` 实测 **200**（`.md` 后缀那份 `2574` 字节）。
# 口径：★ 只用于可定位。★ 该站另有 HTML 版（同 URL 去 `.md`），本文件**以 `.md` 为准**。
# 消费者：`02` 的 `D.3`。
# 同值位置：无。

NVIDIA_RAIL_QUOTE = 'Rail-optimized design means a GPU node with multiple interfaces will put each GPU "rail" (IB network interface) onto a different first level (LEAF) switch for cluster Interconnect.'
# 来源：[B] 级（NVIDIA 官方产品文档）★ **本轮 curl 逐字复核**（与计划 §C.3 一致）。
#       ★★ 原文里的引号是**直引号** `"rail"`（不是全角引号）—— 照抄保留。
# 口径：★★★ **rail 的定义**：一台**多网卡的 GPU 节点**，把**每张 GPU 的 rail（IB 网络接口）**
#       放到**不同的第一层（LEAF）交换机**上。
#       ★★ 注意界定的**主体**：是**一个节点内部**的 `G` 张卡**被分散到 `G` 个叶子**，
#       不是「一个 rail 里有 `G` 台节点」—— ★★ `02` 的 C 类正是判这个混淆。
# 消费者：`02` 的 `D.5` 与 `A.1`（A 类的**定义依据**）。
# 同值位置：无。

NVIDIA_RAIL_HOP_QUOTE = 'This allows multiple nodes to utilize their internal NVSwitch path to talk across a NIC that is just one switch hop away (instead of having to cross multiple switches, incurring additional latency).'
# 来源：[B] 级（NVIDIA 官方产品文档）★ **本轮 curl 逐字复核** ——
#       ★★★ **计划 §C.3 的引文【截断】在这里之前**：原文**紧跟着**这一句，
#         它才是「**代价账**」的官方依据（`one switch hop away`）。
# 口径：★★★ **同 rail 的通信只过一台交换机**（`just one switch hop away`）——
#       ★★ 这是 `02` 的 `B.1` 里跳数账的**逐字依据**；★ 括号里的 `instead of ...` 是对比项。
#       ★★ 引它**必须带 `just` / `one`**（漏掉就把「一跳」升格成「快」）。
# 消费者：`02` 的 `D.5` 与 `B.1`。
# 同值位置：无。

NVIDIA_RAIL_FATTREE_QUOTE = "Fat-Tree is NVIDIA's recommended topology, AI factory should based on rail optimized."
# 来源：[B] 级（NVIDIA 官方产品文档）★ **本轮 curl 逐字复核**。
#       ★★★ **原句的语法是残的**（`should based on` 缺动词）—— ★★ **照抄**，
#         不改写、不补词（「引文不得润色」）；★ 要说明就另起一句。
# 口径：★★ 它把**胖树**与 **rail-optimized** 的关系说成「推荐拓扑 + 应基于 rail 优化」
#       ⇒ ★★ 这两者**不是互斥的两套**（`01` 讲胖树、`02` 讲 rail，两者是**层叠**关系）。
# 消费者：`02` 的 `D.5`。
# 同值位置：无。


# ===========================================================================
# 3. RoCE / PFC / ECN / InfiniBand —— ★ 03 的规格值表（**逐条标级别**）
#    ★★★ 本节**没有 A 类**（本机无网卡）⇒ 一切钉住归 `D`；★ 级别分两轴：
#      【来源级别】A/B/C 与【本轮复核状态】verified / not-fetched。
# ===========================================================================

RFC3168_CARRIER = "RFC 3168: The Addition of Explicit Congestion Notification (ECN) to IP"
# 来源：[A] 级（IETF 规范），**本轮 curl 取到全文**（`170966` 字节）。
# 口径：★ ECN 字段位置的**规范级**出处。
# 消费者：`03` 的 `D.3`。
# 同值位置：无。

RFC3168_URL = "https://www.rfc-editor.org/rfc/rfc3168.txt"
# 来源：[A] —— 本轮 `curl --max-time 25` 实测 **200**（`170966` 字节）。
# 口径：★ 只用于可定位。
# 消费者：`03` 的 `D.3`。
# 同值位置：无。

RFC3168_ECN_QUOTE = "Bits 6 and 7 in the IPv4 TOS octet are designated as the ECN field."
# 来源：[A] 级（RFC 3168，§5 附近的正文）★ **本轮 curl + grep 逐字复核**（计划 §C.3 一致）。
# 口径：★★ **两个位**（bit 6 与 bit 7）、在 **IPv4 的 TOS 字节**里。
#       ★★ 引用**必须带 `designated as`**（这是**指派**，不是「占满两字节」之类的升格）。
# 消费者：`03` 的 `B.1` / `D.5`。
# 同值位置：`ECN_BITS`。

ECN_BITS = (6, 7)
# 来源：[A] 级（同上，从 `RFC3168_ECN_QUOTE` 逐字拆出）。
# 口径：★ 它把上面那句散文变成**可数的两个整数**。
# 消费者：`03` 的 `B.1` / `D.4`（**逐元素**钉住）。
# 同值位置：`RFC3168_ECN_QUOTE`。

RFC3168_CE_QUOTE = "The CE codepoint '11' is set by a router to indicate congestion to the end nodes.  Routers that have a packet arriving at a full queue drop the packet, just as they do in the absence of ECN."
# 来源：[A] 级（RFC 3168，第 5 节的正文）★ **本轮 curl + 逐字复核** —— ★★ **计划 §C.3 未收这一句**
#       （本文件补：它是 ECN 的**行为**依据，也是本文档里那句「端点在丢包之前减速」的**唯一出处**）。
#       ★★ 提取形态：句子跨两行，且 `the end nodes.` 之后是**两个空格**（RFC 的排版习惯）
#         —— 照抄保留（★ 这一点与本主题的 `(k/2)2` / `k3 /4` 是同一类「提取形态」纪律）。
# 口径：★★★ **两句**：① 路由器用 `CE` 码点**向端点标出拥塞**；② ★★ **满队列照样丢包**
#       ——「`just as they do in the absence of ECN`」⇒ ★★★ **ECN 不消除丢包**，
#       它让端点在丢包之前**得到信号**（★ 这半句是限定词，漏掉就把 ECN 说成「不丢包」）。
# 消费者：`03` 的 `B.2` / `D.5`。
# 同值位置：`RFC3168_ECN_QUOTE`（同一条 RFC 的另一个规格值）。

IEEE8021QBB_URL = "https://1.ieee802.org/dcb/802-1qbb/"
# 来源：[A] 级（IEEE 802.1 工作组页）—— 本轮 `curl --max-time 25` 实测 **200**（`64780` 字节）。
# 口径：★ PFC 的**规范摘要**出处（★ **不是**标准正文 PDF —— 那要订阅）。
# 消费者：`03` 的 `D.3`。
# 同值位置：无。

IEEE8021QBB_QUOTE = 'This standard specifies protocols, procedures and managed objects that enable flow control per traffic class on IEEE 802 full-duplex links.'
# 来源：[A] 级（IEEE 802.1Qbb 工作组页）★ **本轮 curl 逐字复核**（计划 §C.3 一致）。
# 口径：★★★ **PFC 的定义**：**按流量类别（per traffic class）**、在 **802 全双工链路**上的流控。
#       ★★ 两个限定词缺一不可：`per traffic class`（**按类别**，不是全链路一刀切）与
#         `full-duplex`（**全双工链路**）⇒ ★★ `03` 的 `C.1` 正是判「把 PFC 当成全链路流控」。
# 消费者：`03` 的 `B.2` / `D.5`。
# 同值位置：无。

IEEE8021QBB_PURPOSE_QUOTE = 'Priority-based flow control is intended to eliminate frame loss due to congestion. This is achieved by a mechanism similar to the IEEE 802.3x PAUSE, but operating on individual priorities.'
# 来源：[A] 级（同页）★ **本轮 curl 逐字复核** —— ★★ **计划 §C.3 未收这一句**
#       （它只收了定义句与「为什么需要」那一句）⇒ 本文件补上，因为它是 `C.1` 的**关键限定**。
# 口径：★★★ **`intended to`（意在）** —— 官方说的是**目的**，**不是保证**；
#       且实现机制是 **`similar to the IEEE 802.3x PAUSE, but operating on individual priorities`**
#       （**按优先级**的 PAUSE）⇒ ★★ `03` 的 `C.1` 判「把它当成万能解」时的**逐字依据**。
# 消费者：`03` 的 `B.2` / `C.1` / `D.5`。
# 同值位置：无。

IEEE8021QBB_DOMAIN_QUOTE = 'Operation of priority-based flow control is limited to a domain controlled by a Data Center Bridging control protocol that controls the application of Priority-based Flow Control, Enhanced Transmission Selection, and Congestion Notification.'
# 来源：[A] 级（同页）★ **本轮 curl 逐字复核** —— ★★ **计划 §C.3 未收这一句**（同上，本文件补）。
# 口径：★★★ **`is limited to a domain controlled by ...`** —— ★★ 官方的**作用域**是一条**限制**：
#       出了受 DCB 控制协议管辖的域，这句话**不承诺**。
#       ⇒ ★★ `03` 的 `C.1` 的第二条依据（「万能解」错在**跨域**与**跨原因**两处）。
# 消费者：`03` 的 `B.2` / `C.1` / `D.5`。
# 同值位置：无。

IEEE8021QBB_WHY_QUOTE = 'Data Center Bridging networks employ higher layer protocols that depend on the delivery of data frames without frame loss due to congestion. These protocols were designed for an underlying transport that approaches lossless behavior and therefore do not include appropriate response to frame loss due to congestion (e.g. back-off, slow restart, etc.).'
# 来源：[A] 级（同页）★ **本轮 curl 逐字复核**。
#       ★★★ **计划 §C.3 只收了第二句**（`These protocols were designed for ...`）——
#         而**第二句的主语在第一句里**（`higher layer protocols`）⇒ ★★ 本文件把**两句一起**登记，
#         否则会把「**哪些**协议」这个主语丢掉（v1.7 的硬伤之一就是**漏掉限定词**）。
# 口径：★★★ **「为什么需要 PFC」的官方依据**：DCB 网络里跑着**依赖无损交付的高层协议**，
#       而这些协议**是为「接近无损」的底层传输设计的、因此不含对拥塞丢包的适当响应**
#       （括号里还给了 `e.g. back-off, slow restart, etc.`）。
#       ★★ 注意 `approaches lossless`（**接近**无损）—— ★ **不是** `is lossless`。
# 消费者：`03` 的 `B.2` / `D.5`。
# 同值位置：无。

IEEE8021QBB_COMPLEMENT_QUOTE = 'In addition, PFC complements Congestion Notification in Data Center Bridging networks.'
# 来源：[A] 级（同页）★ **本轮 curl 逐字复核** —— ★★ **计划 §C.3 未收这一句**（本文件补）。
# 口径：★★ **PFC 与 ECN 是【互补】关系**（`complements`），**不是替代** ——
#       ★★ 这是本库把两者并列讲（`03` 的 §1.3 与 §1.4）的**逐字依据**。
# 消费者：`03` 的 `B.2` / `D.5`。
# 同值位置：`RFC3168_ECN_QUOTE`（互补的另一半）。

PFC_PRIORITY_COUNT = 8
# 来源：★ **[B] 级 / 本轮未复核** —— `8` 个流量类别来自 802.1Q 的**3 位 PCP**（Priority Code Point）；
#       ★★ 本库本轮**未取到 802.1Q 原文** ⇒ 如实标 `[B]/未复核`。
# 口径：★★ 它**只**用作「按类别」演示里的**类别总数**（`03` 的 `C.1` 要用「全部类别」这个集合），
#       ★ **不构成一条规格事实**（要断言 `8` 必须另找载体）。
# 消费者：`03` 的 `C.1` / `D.4`。
# 同值位置：无。

CISCO_ROCEV2_DOC = 'Cisco UCS C-Series Integrated Management Controller Configuration Guide for RDMA over Converged Ethernet (RoCEv2), Release 4.2'
# 来源：**[B] 级**（Cisco 产品文档）★ 载体本轮 `curl --max-time 25` 实测 **200**（`40982` 字节）。
# 口径：★★★ **`F4` 的落点**：RoCEv2 的**机制/封装**只能到这个级 ⇒ **不得**升成 `[A]`。
# 消费者：`03` 的 `D.3`。
# 同值位置：无。

CISCO_ROCEV2_URL = 'https://www.cisco.com/c/en/us/td/docs/unified_computing/ucs/c/sw/gui/config/guide/4_2/b-roce-configuration-cimc-4-2/m-rdma-over-converged-ethernet-roce-version-2.html'
# 来源：[B] —— 本轮 `curl --max-time 25` 实测 **200**（`40982` 字节）。
# 口径：★ 只用于可定位。★ 版本写在 `CISCO_ROCEV2_DOC` 里（`Release 4.2`）。
# 消费者：`03` 的 `D.3`。
# 同值位置：无。

CISCO_ROCEV2_PORT_QUOTE = "The UDP destination port number 4791 has been reserved for RoCEv2."
# 来源：**[B] 级（Cisco 产品文档）** ★ **本轮 curl 逐字复核**（计划 §C.3 一致）。
# 口径：★★ `4791` 是 **UDP 目的端口**、被 **reserved for RoCEv2**（保留）。
#       ★★★ **`F4`**：这条**不得**标 `[A]` —— IBTA 的一手规范（Annex A17）**未取到**；
#       ★★ 上面的上下文还有一句（本文件单列在 `CISCO_ROCEV2_ENCAP_QUOTE`）。
# 消费者：`03` 的 `B.1` / `C.2`（`F4` 的守卫）/ `D.5`。
# 同值位置：`ROCEV2_UDP_PORT`。

ROCEV2_UDP_PORT = 4791
# 来源：[B] 级（同 `CISCO_ROCEV2_PORT_QUOTE`）★ 本轮已复核。
# 口径：★ 把那句话变成**一个可比对的整数**（`< 2^53`，满足 `mutate.py` 盲区⑥）。
# 消费者：`03` 的 `B.1` / `D.4`。
# 同值位置：`CISCO_ROCEV2_PORT_QUOTE`。
ROCEV2_PORT_RANGE_MAX = 65535
# 来源：**[A] 级**（UDP 端口号的通用上界，`16` 位）—— ★★ **本轮未取到规范原文**
#       （没有为此单独取 RFC 768）⇒ 如实记；它只用于「`4791` 落在合法区间内」这一条**算术检查**。
# 口径：★ **不构成**一条 RoCEv2 事实（那是 `F4` 的边界）。
# 消费者：`03` 的 `B.1`。
# 同值位置：无。

CISCO_ROCEV2_ROUTABLE_QUOTE = "RoCEv2 is an internet layer protocol, which means that RoCEv2 packets can be routed."
# 来源：**[B] 级（Cisco 产品文档）** ★ **本轮 curl 逐字复核**（计划 §C.3 一致）。
# 口径：★★ 「**能路由**」的原因是它是**网络层（internet layer）协议** —— ★★ 这是 RoCEv2 与
#       **RoCEv1**（链路层）的**唯一分界**（见 `CISCO_ROCEV1_QUOTE`）。
# 消费者：`03` 的 `B.1` / `D.5`。
# 同值位置：无。

CISCO_ROCEV1_QUOTE = 'RoCEv1 is an Ethernet link layer protocol and hence allows communication between any two hosts in the same Ethernet broadcast domain.'
# 来源：**[B] 级（Cisco 产品文档）** ★ **本轮 curl 逐字复核** —— ★★ **计划 §C.3 未收这一句**
#       （本文件补：它是 `CISCO_ROCEV2_ROUTABLE_QUOTE` 的**对照项**，没有它「能路由」就没有边界）。
# 口径：★★ RoCEv1 是**链路层** ⇒ 只能在同一**以太网广播域**内通信；★ 对照 RoCEv2 可路由。
# 消费者：`03` 的 `B.1` / `D.5`。
# 同值位置：无。

CISCO_ROCEV2_ENCAP_QUOTE = 'It does this by encapsulating an Infiniband (IB) transport packet over Ethernet.'
# 来源：**[B] 级（Cisco 产品文档）** ★ **本轮 curl 逐字复核** —— ★★ **计划 §C.3 未收这一句**。
#       ★ 原文里 `Infiniband` 的拼写**首字母小写 i**（不是 `InfiniBand`）—— **照抄保留**。
# 口径：★★★ **这句就是 `F4` 的现场**：它描述 RoCEv2 的**封装**（IB 传输包套在以太网里），
#       而**它的载体是产品文档** ⇒ ★★ 只能标 **[B]**；★ 本文件用 `ROCE_FACT_ROWS` 把它**钉成 B**。
# 消费者：`03` 的 `B.1` / `C.2`（`F4` 的守卫）/ `D.5`。
# 同值位置：无。

IBTA_ROCE_POSITION_QUOTE = 'RoCE is an industry standard transport that enables Remote Direct Memory Access (RDMA) to operate on ordinary Ethernet layer 2 and 3 networks.'
# 来源：**[A] 级**（IBTA 的 RoCE 问答页）—— ★★ **照计划 §C.3 转录**。
#       ★★★ **复核状态：本轮【未取到】该页** —— 本轮 `curl --max-time 25` 试了
#         `infinibandta.org` 的 **5** 个 URL（`/rdma-over-converged-ethernet-roce-faq/`、
#         `/faq/`、`/faqs/`、`/roce/`、`/about-roce/`），**没有一个取到该页**
#         （结果 = **404 ×2 · 301 ×2 · 连接失败 ×1**）。
#         ★ 另试了同一站点的**新闻稿页**（`/infiniband-trade-association-launches-...`）—— 它**200 取到了**，
#           但页上那句是 `RoCE utilizes RDMA to ...`，**与这条逐字不同** ⇒ **不冒充**。
#         ⇒ ★★ **本常量本轮【未能独立复核】**（`03` 的 `B.1` 会如实打印这一状态）。
#       ★ 可见的替代物（**不是同一句**）：`infinibandta.org` 的新闻稿里有一句
#         `RoCE utilizes RDMA to ...`（本轮取到，但**逐字不同**）⇒ 本库**不用它冒充**这一条。
# 口径：★ 它只支撑「RoCE 是**行业标准**传输、把 RDMA 搬到**普通以太网 L2/L3**」这一定位，
#       ★★ **不构成**任何机制/封装的引用（那两条另有 `[B]` 载体）。
# 消费者：`03` 的 `B.1`（打印复核状态）/ `D.5`。
# 同值位置：无。

SHARP_DOC_TAG = "SHARP UM 3.13"
# 来源：**[B] 级**（NVIDIA SHARP 用户手册）★ 本轮 `curl --max-time 25` 实测 **200**。
#       ★★ 同一句在 **`3.10.2`** 版上**逐字相同**（本轮两份都取到）⇒ 版本敏感度低，但**仍记版本**。
# 口径：★ 只用于「本文引的是哪一版」。
# 消费者：`03` 的 `D.3`。
# 同值位置：`SHARP_DOC_TAG_ALT`。
SHARP_DOC_TAG_ALT = "SHARP UM 3.10.2"
# 来源：**[B] 级**（同一手册的另一版）★ 本轮已复核。
# 口径：★ 它证明「同一句在两版上逐字相同」——★ **这是本库对「版本」的一次正面用法**。
# 消费者：`03` 的 `D.3`。
# 同值位置：`SHARP_DOC_TAG`。

SHARP_URL = "https://networking-docs.nvidia.com/sharpum/3130/introduction"
# 来源：**[B] 级** —— 本轮 `curl` 到 `.md` 版本（`1052` 字节）。
# 口径：★ 只用于可定位。
# 消费者：`03` 的 `D.3`。
# 同值位置：无。

SHARP_QUOTE = 'NVIDIA® Scalable Hierarchical Aggregation and Reduction Protocol (SHARP)™ technology improves the performance of MPI and Machine Learning collective operation, by offloading collective operations from CPUs and GPUs to the network and eliminating the need to send data multiple times between endpoints.'
# 来源：**[B] 级（NVIDIA 产品文档）** ★ **本轮 curl 逐字复核**（计划 §C.3 一致；★ 且两版一致）。
# 口径：★★ **SHARP 的定位**：把集合操作**从 CPU/GPU 卸载到网络**，并**消除端点之间多次发送同一份数据**。
#       ★★ 原文的 `™` / `®` 符号**照抄保留**；`collective operation` 是**单数**（照抄）。
#       ★★★ **性能数字（`2.1×` / `3.24×`）不在本文件里** —— 见 `NOT_QUOTED`（`F6`）。
# 消费者：`03` 的 `B.3` / `D.5`。
# 同值位置：无。

SHARP_PAPER_CARRIER = "SHARP, COMHPC 2016"
# 来源：★ **[C] 级**，**同行评审：是**（计划 §C.3 给）—— ★★★ **正文本轮未取到**（`F6`）。
# 口径：★★ 只登记**载体**（会议 + 年）；★ **不登记标题 / 作者 / 页码**（未核实 ⇒ 不记）。
#       ★ 与 `SHARP_QUOTE`（产品文档 `[B]`）**是两条独立来源**，本文件**分开存放**。
# 消费者：`03` 的 `D.6`。
# 同值位置：无。

DCQCN_CARRIER = "DCQCN, SIGCOMM 2015"
# 来源：★ **[C] 级**，**同行评审：是**（计划 §C.3 给）—— ★★★ **正文本轮未取到**（`F6`）。
#       ★★ **且本轮未核实它的标题与作者**（`api.crossref.org` 未通）⇒ **只登记载体**。
# 口径：★★ 它**只**支撑「DCQCN 是一篇 SIGCOMM 2015 的工作」这件事；
#       ★★★ **本库不写它的机制**（三组件描述来自二手检索摘要 ⇒ 见 `NOT_QUOTED`）。
# 消费者：`03` 的 `D.6`。
# 同值位置：无。

ROCE_FACT_ROWS = (
    ("ECN 字段（IPv4 TOS 的 bit 6–7）", "A", "verified"),
    ("PFC 按流量类别作用于全双工链路", "A", "verified"),
    ("RoCE 的定位（IBTA 的措辞）", "A", "not-fetched"),
    ("RoCEv2 的 UDP 目的端口 4791", "B", "verified"),
    ("RoCEv2 可路由（网络层协议）", "B", "verified"),
    ("RoCEv2 的封装（IB 传输包套以太网）", "B", "verified"),
    ("SHARP 的在网计算定位", "B", "verified"),
)
# 来源：**本文件上面各条的汇总**（每行 = 一条事实 + 它的**来源级别** + **本轮复核状态**）。
# 口径：★★★ 本表是 `F4` 的**机械落点**：RoCEv2 的**封装**与**端口**两行**必须是 `B`**
#       —— ★★ 因为 IBTA 的一手规范（Annex A17）未取到 ⇒ **不得**升成 `A`。
#       ★★ 级别与复核状态是**两轴**：`A` 不代表「本轮复核过」（第 3 行是 `A` 但
#         `not-fetched`）—— ★★★ **这正是本文件最想守住的一条区分**。
# 消费者：`03` 的 `B.1` / `C.2`。
# 同值位置：无。

NOT_QUOTED = (
    "SHARP 的 2.1× / 3.24× 性能数字（二手检索摘要）",
    "DCQCN 的三组件机制描述（二手检索摘要）",
    "RoCEv2 的帧结构（IBTA Annex A17 一手规范未取到）",
)
# 来源：**否定性记录**（本库**明写「不引什么、为什么」**，照 `90-training` 的形态）。
# 口径：★★★ 三条都对应计划 §C.5 的 **`F4` / `F6`**：
#       · 第 1 条 = `F6`（SHARP 的性能数字**不得引**）；
#       · 第 2 条 = `F6`（DCQCN 的三组件来自**二手摘要**，**不得当逐字引文**）；
#       · 第 3 条 = `F4`（RoCEv2 帧结构的一手规范未取到 ⇒ 只能 `[B]`）。
#       ★★ 它是**非空**元组 ⇒ 有叶子 ⇒ 会被 `mutate.py` 变异 ⇒ **必须逐元素 `==` 钉住**。
# 消费者：`03` 的 `D.6`。
# 同值位置：无。


# ===========================================================================
# 4. 本库自记的两条「范围」事实（★ 由上面的引文**推出来**，不是外部引用）
#    ★★ 它们把 `F4` / 「PFC 是不是万能解」变成**可机械判定**的东西（`03` 的 `C.1`/`C.2`）。
# ===========================================================================
PFC_OFFICIAL_LOSS_CAUSES = ("congestion",)
# 来源：**[E] 本库推导** —— **从引文里拆出来的**：官方把 PFC 的用词限定在
#       `frame loss due to congestion`（`IEEE8021QBB_PURPOSE_QUOTE`）⇒ 官方点名的**丢包原因只有一个**。
# 口径：★★ 它**不是**「网络只有一种丢包原因」，而是「**官方的措辞只覆盖这一个**」——
#       ★★ 这正是判「万能解」的**范围**依据（★ 别把它读成物理断言）。
# 消费者：`03` 的 `C.1`。
# 同值位置：`IEEE8021QBB_PURPOSE_QUOTE`。

SPEC_REQUIRED_FOR = ("RoCEv2 的封装（IB 传输包套以太网）",)
# 来源：**[E] 本库推导** —— **哪些事实要标 `[A]` 就必须先取到一手规范**。
#       ★★ 这一条对应计划 §C.5 的 **`F4`**：RoCEv2 的**帧结构 / UDP 封装**的 `[A]` 依据是
#       IBTA 的 **Annex A17**，而它**本轮未取到**。
# 口径：★★ 元组里的**名字必须与 `ROCE_FACT_ROWS` 的某一行的第 1 列逐字相同**（`03` 的 `D.3b` 守这件事）
#       —— ★★ 否则这条守卫会**静默失效**（比对本不到任何行）。
# 消费者：`03` 的 `C.2`。
# 同值位置：`ROCE_FACT_ROWS` 的第 6 行。

HAVE_IBTA_ANNEX_A17 = False
# 来源：**[E] 本库自记** —— **本次取证的实情**：IBTA 的 RoCEv2 一手规范（Annex A17）
#       **本轮未取到**（见 `NOT_QUOTED` 的第 3 条）。
# 口径：★★★ **它是 `F4` 的开关**：为 `False` 时，「把 RoCEv2 的封装标成 `[A]`」**必须被拒**。
#       ★★ 若**将来**真的取到了 Annex A17，把这里改成 `True` **并同时**改 `ROCE_FACT_ROWS` 那一行
#       —— ★ 两处**必须一起动**，`03` 的 `D.3b` 会把它们的不一致**报成失败**。
# 消费者：`03` 的 `C.2` / `D.3b`。
# 同值位置：无。


# ===========================================================================
# 5. UNSOURCED —— 待办清单，不是免责声明
# ===========================================================================
UNSOURCED = ()
# 来源：**本库自记**。
# 口径：★★ 本主题**没有**「用了但没出处」的常量 ⇒ 留空（`0` 个叶子）。
#       ★★ 空元组**有消费者**（三个程序各钉一次）⇒ 不产生变异体，也不构成漏网候选。
#       ★★ **注意它与 `NOT_QUOTED` 的分工**：`UNSOURCED` = 「我用了但没出处」（本主题为空）；
#         `NOT_QUOTED` = 「我**故意不引**、并说明为什么」（本主题有 3 条，见上）。
# 消费者：`01` 的 `D.7` · `02` 的 `D.7` · `03` 的 `D.7`。
# 同值位置：无。

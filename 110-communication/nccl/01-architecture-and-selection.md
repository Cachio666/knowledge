---
title: NCCL 的算法与协议：取值表、版本账、与它怎么选
slug: communication-nccl-architecture-and-selection
domain: 110-communication
topic: nccl
level: 3
difficulty: intermediate
prerequisites:
  - 110-communication/collectives/01-algorithm-family
  - 20-gpu-architecture/interconnect/03-collectives-on-topology
tags: [nccl, nccl-algo, nccl-proto, version-ledger, implementation-layer]
related:
  - 110-communication/nccl/02-topology-awareness
  - 110-communication/nccl/03-failures-and-limits
  - 20-gpu-architecture/interconnect/03-collectives-on-topology
code:
  - code/01_architecture_and_selection.py
status: stable
updated: 2026-10-08
---

# NCCL 的算法与协议：取值表、版本账、与它怎么选

> **NCCL 把「用哪个算法」和「用哪个协议」露成两个环境变量。**
> 本文给出**官方的取值表与它的版本账**，并说清**官方【怎么选】——答案是：它不告诉你**。
> ★★★ **本层是【实现层】**：讲的是**真实库暴露了什么旋钮**，不是算法的代数（那在 `collectives`）。
> ★★ **本主题【全靠 `[B]` 级】** —— 载体是 NCCL 官方**产品文档**（**不是规范**），
> 版本一律 `NCCL 2.32.3`，出处 `docs.nvidia.com/deeplearning/nccl/user-guide/docs/`，抓取日 `2026-10-08`。
> ★★★ **本主题【没有 A 类】**：本机**无 GPU、无 NCCL、无真网卡** ⇒ 算法正确性**无处可验**。

## 0. 背景与动机

`20-gpu-architecture/interconnect/03:243-249` 有一段 **7 行**的定性描述：
**ring allreduce** 假设只有环形邻居、**tree allreduce** 假设有中心节点、**collnet / NVLS** 假设交换机自己会归约，
「选哪个由 NCCL 在初始化时探测拓扑决定」。★ **本域 `nccl` 从那里接**。

> **一个知识点只有一处正文**（`CONTRIBUTING` §7）。那段描述在
> `20-gpu-architecture/interconnect/03-collectives-on-topology.md:243-249`（正文在彼，此处回链）。
> ★ 同类的定性描述**不止那一处**（`interconnect/04:222` · `interconnect/02:192` ·
> `performance-models/02:124` 也各有一句）—— **四处都停在定性**，**没有一处给变量名**。

**那本文补什么？** 补那四处**都没写的两样东西**：

1. **官方把哪几条旋钮露出来了** —— `NCCL_ALGO`（算法）与 `NCCL_PROTO`（协议）的**取值表**；
2. ★★★ **官方怎么选** —— **答案是「它不给你规则」**：默认是 `unset`，
   两个曾经的阈值变量（`NCCL_TREE_THRESHOLD` / `NCCL_SINGLE_RING_THRESHOLD`）**都已移除**。

**一个具体的对照场景**（本文 `code/` 的 `B.1` 打印，载体见 `specs.py`）：

| 取值 | since | 备注 |
|---|---|---|
| `Ring` | `2.5+` | |
| `Tree` | `2.5+` | |
| `Collnet` | ★ `2.5–2.13` | ★★ **全表唯一的闭区间** ⇒ 它**被移除了** |
| `CollnetChain` | `2.14+` | ★ 与下一条**同版本出现** |
| `CollnetDirect` | `2.14+` | |
| `NVLS` | `2.17+` | |
| `NVLSTree` | `2.18+` | |
| `PAT` | `2.23+` | |

★★★ **这张表读出来的不是「八个选项」，是一段【演进史】**：
`Collnet` 在 `2.14` **被拆成两条**（`CollnetChain` / `CollnetDirect`），
★ **不是**「它还在、又多了两条」。★ 而 `2.14` 之后还有三次扩张（`2.17` / `2.18` / `2.23`）。

## 1. 原理

### 1.1 ★★ 两条**独立的轴**：算法（谁跟谁）与协议（怎么打包）

`collectives` 讲的是**算法族**（ring allreduce / tree allreduce / …）的**步数与搬运量**。
NCCL 在这里多出**第二条轴**：

| 轴 | 问题 | 官方旋钮 | 取值 |
|---|---|---|---|
| **算法** | 「**哪些设备之间**交换数据、按什么次序」 | `NCCL_ALGO` | 8 个（见上表） |
| **协议** | 「一笔消息**怎么打包**发出去」 | `NCCL_PROTO` | `LL` / `LL128` / `Simple` |

★★ 官方对这两条轴的**措辞**（逐字，`[B]` 级）：

> `NCCL_ALGO`：`The NCCL_ALGO variable defines which algorithms NCCL will use.`
> `NCCL_PROTO`：`The NCCL_PROTO variable defines which protocol(s) NCCL will be allowed to use.`

★★ **一条【同名不同指】（本文开篇定调）**：本文里的 **`Ring` / `Tree` / `NVLSTree` 是
`NCCL_ALGO` 的【取值名】**（★ 官方标识符；★★ **取值【不】大小写敏感** —— `env.html` 逐字
`(not case sensitive)`，见 §1.6），★ **不是**「ring allreduce」
那个算法概念；而**当本文要指算法概念时，一律写「ring allreduce」/「tree allreduce」全称**
（★ 库内 `ring` 另有 **ring 拓扑**与 **ring attention** 两个含义，`tree` 另有 **AST / 计算图**，
见 `110-communication/README.md` 的「同名不同指」表）。
★ **本文出现的裸 `tree` / `ring`**（如 `F3` 的禁写例句）**都是【要被禁掉的那句话本身】**，
★ 形如 `NCCL_TREE_THRESHOLD` 的**是变量名**，不是概念。

★ **注意两句话的一个差别**：算法那句是 `will use`（要用），协议那句是 `will be allowed to use`（**允许**用）。
★★★ 这个差别**不是修辞** —— 它对协议那句还**配了一条劝阻**（见 §1.4）。

### 1.2 ★★★ `NCCL_ALGO` 的**版本账**：逐条可数

★★ 本文 `code/` 的 `B.1` **不重述**上表，而是从表里**推出三条可判定的性质**：

1. **取值恰好 8 个**，且**每一个都带 `since`**（★ 无一项留空）；
2. **闭区间（即「有移除」）的取值恰好 1 个** —— `Collnet`（`2.5–2.13`）；
3. **`2.14` 这个版本上出现恰好 2 个取值**（`CollnetChain` / `CollnetDirect`）。

★★ **第 2 与第 3 条合起来**才是账**读得通**的意思：
**一个在 `2.13` 结束、两个在 `2.14` 开始** ⇒ 这是**拆分**，不是**并存**。

★ **判据口径**（写死在 `code/` 里，`B.1`）：闭区间的判据是「`since` 字符串里含 **en-dash `–`** 」
—— ★★ **不是** ASCII 连字符 `-`（`2.5+` 里那个是 `+`，`2.14+` 也是 `+`）⇒ ★ 判据与形态**绑定**，
★ 若把 `2.5–2.13` 改写成 `2.5-2.13`，**本条会失败**（★ 那正是「把形态抄错」的现场）。

### 1.3 ★★★ 官方**怎么选** —— 答案是：**它不告诉你**

★★★ 这是本主题**最要紧的一节**。官方逐字（`[B]`，`env.html`）：

> `The default is unset, which causes NCCL to automatically choose the available algorithms based on the node topology and architecture.`

★★★ **推论（本文的题眼）**：**默认是 `unset` ⇒ 由 NCCL 自动选** ——
★★ **官方把「怎么选」的实现留给了自己**，**不承诺**任何可写下来的规则。

★★★ **而那两条曾经的「阈值变量」已经没了**（`env.html` **正文**，逐字，`[B]`）：

| 变量 | 状态 |
|---|---|
| `NCCL_TREE_THRESHOLD` | **`removed in 2.5`** |
| `NCCL_SINGLE_RING_THRESHOLD` | **`removed in 2.3`** |

⇒ ★★★ **`F3` 的落点**：**不得写**「NCCL 在 X 大小以下用 tree、以上用 ring」这类**阈值表** ——
那两个变量**已被移除**，★ **官方未公开替代规则**。
★★★ **想观察实际选择，唯一的官方抓手是** `NCCL_DEBUG_SUBSYS=TUNING`
（见 [02-topology-awareness](02-topology-awareness.md)）。

### 1.4 ★★ `NCCL_PROTO`：三个取值，**官方劝阻你别动**

> `NCCL_PROTO` 取值：`LL, LL128, and Simple.`（逐字）
> 官方警告（逐字，`[B]`）：`Users are discouraged from setting this variable … enabling LL128 on platforms that don't support it can lead to data corruption.`

★★★ **读法**：官方**劝阻**用户设置它，而且给了**理由**（`data corruption`：在不支持的平台上开 LL128
会**数据损坏**）。★ 这与 `NCCL_ALGO`（只给可选项、不给劝告）**形态明显不同**。

★★ **本文不写 `LL` / `LL128` 的机制**：计划 `§C.5` 的 `F9` 已放宽
（**有载体的**机制可以写：arXiv `2507.04786`、NCCL/RCCL 源码如 `prims_ll128.h`），
★ **但本库【未取到】那些载体的正文** ⇒ ★★ **无出处的机制不写**（只写**取值**与**官方警告**）。

### 1.5 ★★★ 异步语义 —— **`F1` 的落点**

★★★ **计划 `§C.5` 的 `F1`**：`"Communication is always asynchronous with respect to the host"`
**不是** NCCL 文档的逐字引文 —— 研究者抓 **8 个文档页**、逐页 grep `asynchronous` /
`with respect to the host` ⇒ **0 命中**。

★★★ **可用的、已验证的替代表述只有三条**（`[B]` 级，本文 `code/` 的 `D.6` 逐字钉住）：

| 出处 | 逐字 |
|---|---|
| `usage/streams.html` | `The collective operation is then executed asynchronously on the CUDA device.` |
| `api/group.html` | `All subsequent calls to NCCL until ncclGroupEnd will not block due to inter-CPU synchronization.` |
| `usage/groups.html` | `can return without having enqueued the operation on the stream`（★ 计划记录的首尾 `…` 是**计划自己的省略标记**） |

★★ **三句合起来的含义**：官方说的是「**在 CUDA 设备上异步执行**」＋「**组内不因 CPU 间同步阻塞**」
＋「**返回时可能还没入队**」—— ★★ **它们比 F1 那句【弱】**（都是**有条件的**）。

### 1.6 ★★★ 取值【不】大小写敏感 —— 一处被反转的官方措辞

★★★ **官方对两条轴的取值措辞**（逐字，`[B]`，`env.html`，★ **算法与协议两处都有**）：

> `NCCL_ALGO`：`Comma-separated list of algorithms (not case sensitive) among:`
> `NCCL_PROTO`：`(since 2.5) Comma-separated list of protocols (not case sensitive) among:`

⇒ ★★★ **官方说取值【不】大小写敏感**（`not case sensitive`）。
★★ **推论**：**「`Ring` 写成 `ring` 会失败」是【假的】** —— 真正会失败的只有
**根本不存在**的取值名（如 `Ringg` / `RingAllReduce`，见 §4.1 与 `C.1`）。

★★★ **一条本库的实测记录（★ 值得读者记住）**：本主题的**初稿**在这里把话**说反了** ——
它写成「官方标识符，**大小写敏感**」，并据此**编了一条失败成因**（「`Ring` 写成 `ring`」）。
★★★ 这是一次**编机制从句**：`(not case sensitive)` **就在官方同一句里**，而**转写时被反转**。
★ 它**无数字可 grep**（没有可查的数值），★★★ **而且它不在**本文自列的禁写清单（`F1`/`F2`/`F3`/`F9`）里
—— ★ 「自己列的会小心，**没注意到的地方**才是危险」。
★★ **可机械复核**：`code/` 的 `B.5` 逐字钉住这两句含 `(not case sensitive)`。

## 2. 执行映射

### 2.1 本机怎么算这件事

`code/01_architecture_and_selection.py` 处理的是**官方文档的取值表**，它的「参照」是 `specs.py` 的 **`[B]` 级引文**。

| 角色 | 是什么 | 它为什么**不是** A 类 |
|---|---|---|
| **被测** | 取值表 / 版本账 / 引文的**字面** | ★★★ **本机没有 NCCL** ⇒ 没有可跑的「实现」 |
| **参照** | `specs.py` 的 `[B]` 级引文（版本 2.32.3） | ★★ 参照本身是**抄来的文本**，不是独立算法 |
| **第三方** | **无** | ★ 没有第三方 ⇒ 一切「对拍」都只是**抄写核对** |

★★★ **⇒ 本程序【没有 A 类】**：把「常量钉住」冒充 A 类，正是 `P12` 的变体
（**判据看着在检查，其实只是把常量抄了一遍**）。
★ 所以本程序把**一切钉住归 `D` 类**，并用 **`D.0`** —— 一条 **AST 扫自己源码**的断言 ——
**机械地**证明「本文件里 `check("A...")` 的调用数 = 0」。

★★ **本程序真正有信息量的两条**（都是 `C` 类，断言的是**错法真的失败**）：

| 断言 | 它喂进去的**错法** | 它要求的结果 |
|---|---|---|
| `C.1` | 一个**不存在的**取值 `NCCL_ALGO_FABRICATED = "RingAllReduce"` | **被拒**（★ 名字看起来很像真的） |
| `C.2` | 把 `NVLS` 的 `since` 记成 `2.14+` | **被拒**（★ `2.14` 是**真实存在**的版本档 ⇒ 这个错法**可信**） |

### 2.2 ★★★ 一条必须写明的限定语

> ★★ 本机**无 GPU、无 NCCL、无真网卡**；本文与配套程序里的**每一个数**都**抄自官方产品文档**。
> ★ 强制限定语：**「不是 NCCL，也不是真机」** —— 本文的一切**不是**任何真实 NCCL 的行为。
> ★★ 载体版本 `NCCL 2.32.3`，出处 `https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/`，
> 抓取日 **2026-10-08**；★★ **不写版本与抓取日的引文不算引文**（`P19` 的第一道防线）。

★★ **本文的「形态」一句话**：编号程序**跑在默认 `python3` 下、只用标准库**
⇒ 落在 `CONTRIBUTING` §2 表的**第二种情形**（**程序内断言**）。
★★★ **点破一层**：§2 的第二种情形**名义上要求 A/B/C 缺一不可**，而**本主题没有 A**
⇒ 属「**第二种情形的【改写】**」（★ 不是 §2 里任何一种**原样**形态）。
★ 而「**把不可测的那一档写明**」是【另一条】纪律（**不是** §2 的任何一种编号形态）。

## 3. 实现与演进

### 3.1 从版本账看演进（★ 每一条都可回文档核）

| 版本 | 发生了什么 | 它在账上的形态 |
|---|---|---|
| **2.3** | `NCCL_SINGLE_RING_THRESHOLD` **被移除** | ★ 账上**看不见**（已移除的变量不在取值表里） |
| **2.5** | `Ring` / `Tree` 起始；`Collnet` 起始；`NCCL_TREE_THRESHOLD` **被移除** | ★★ 阈值规则**开始消失** |
| **2.14** | `Collnet` 结束、`CollnetChain` / `CollnetDirect` 起始 | ★★ **一对一的拆分** |
| **2.17** | `NVLS` | ★ offload 路线（`NCCL_NVLS_ENABLE`，见 `02`） |
| **2.18** | `NVLSTree` | ★ NVLS 与 tree allreduce 的组合 |
| **2.23** | `PAT` | ★ 最后一次扩张 |

★★ **注意账的「盲区」**：**被移除的变量不在取值表里** ——
★ 你要知道 `NCCL_SINGLE_RING_THRESHOLD` **曾经存在**，得去读**正文里的 `removed in ...`**
（★★ 而 **TOC 里仍列着它** ⇒ ★★★ **只看 TOC 会误判「它还在用」**）。

### 3.2 ★★★ 为什么官方**不给**选择规则

★ 这不是「文档没写全」，而是**默认值本身就表达了设计意图**：`unset` ⇒ **NCCL 自己算**。
★★ 一个**可证的**推论：**如果官方想让你控制选择，它就不会把阈值变量【移除】**
（`2.3` / `2.5` 两次移除，方向一致）。

★★★ **▶ 一条方法论（写给读者）**：见到「NCCL 在 X 大小以下用 tree」这类句子，
**先问它的载体是什么** —— ★ 本库**没有取到**任何这样的官方规则（见 `F3`）。

★★★ **一条回链的提醒**：上游那几处**也是定性**、**同样没有官方载体** ——
`20-gpu-architecture/interconnect/03-collectives-on-topology.md:244-245`（「NCCL 对小消息默认走 tree」）
与 `10-foundations/performance-models/02-amdahl-and-scaling.md:124`（同义句）都**没有**官方引文支撑。
★ 本域**不复制**它们、也**不替它们背书** —— ★★ 读者**不要**把上游那句当成**已核事实**。

### 3.3 与相邻主题的分工

| 话题 | 在哪 |
|---|---|
| ★★★ **算法族的步数与每卡搬运量**（ring / tree / … 的**代数**） | `110-communication/collectives/01-algorithm-family.md` |
| ★★ **拓扑受限的集合通信**（ring / tree / collnet-NVLS 的**一句话级描述**） | `20-gpu-architecture/interconnect/03-collectives-on-topology.md:243-249` |
| **`V(N)` 的推导与饱和** | `10-foundations/performance-models/02-amdahl-and-scaling.md:147-155,209` |
| ★★ **NCCL 的算法与协议取值、它的版本账** | **本文** |
| ★★ **NCCL 的拓扑感知（三条抓手）** | [02-topology-awareness](02-topology-awareness.md) |
| ★★ **NCCL 的报错归因与边界** | [03-failures-and-limits](03-failures-and-limits.md) |

## 4. 踩坑与失败模式

### 4.1 ★★★ 编一个不存在的 `NCCL_ALGO` 取值

- **现象**：程序**不报错**、配置**看起来生效了**（环境变量是个合法字符串）；
- **成因**：**编一个【不存在】的取值名**（把 `Ring` 拼成 `Ringg`，或把 `Ring` 与 `AllReduce`
  拼成 `RingAllReduce`）—— ★★ 它是个**合法标识符**，所以**不报错**；
- ★★ **不是成因**：把 `Ring` 写成 `ring` —— 官方说取值 **`(not case sensitive)`**（§1.6）
  ⇒ **大小写变了照样成立**（★ 初稿曾把这一条错列成因，见 §1.6 的实测记录）；
- **本机实测**（`code/` 的 `C.1`）：编造的 `RingAllReduce` 被**校验器拒绝**，
  而 **8 个真取值全部被接受**；
- **规避**：★★★ **唯一的信号是「校验器拒绝」** —— 这就是本程序 `C.1` 存在的理由。

### 4.2 ★★★ 把 `since` 版本记错

- **现象**：版本账**对不上**，但**不报错**（版本号本身是个合法字符串）；
- **成因**：把 `NVLS` 的 `since` 记成 `2.14+` —— ★★★ **这个错法特别可信**，
  因为 `2.14` 是**真实存在**的一个版本档（`CollnetChain` / `CollnetDirect` 就在那）；
- **本机实测**（`C.2`）：版本账**拒绝**它；正确答案是 **`2.17+`**；
- **规避**：★★ 把版本账**做成表**（`NCCL_ALGO_NAMES` 与 `NCCL_ALGO_SINCE` **同下标**），
  ★★★ 并用一条**口径守卫**（`D.3b`）钉住「两张表等长」——
  否则表一改，`since_of()` 会 `IndexError`（**崩溃捕获**），而不是给出一个可读的失败。

### 4.3 ★★★ 引一句**不存在**的引文（`F1`）

- **现象**：正文写「NCCL 文档说『通信对主机永远异步』」—— **读起来像引文**，**不报错**；
- **成因**：★★★ 那句 `"Communication is always asynchronous with respect to the host"`
  **在 8 个文档页里 0 命中**（研究者逐页 grep）；
- **规避**：★★ 只能用 §1.5 的**三条**已验证表述；★ 本程序 `D.6` 把三条**逐字钉住**，
  并**显式断言**那句 F1 **不在**任何一条里。

### 4.4 ★★ 把「不跨标记」当成跨标记（引文层）

- **现象**：一句引文在**去标记文本**里 `count() == 1`，就当成「逐字」；
- **成因**：★★★ **原始 HTML 上它可能 `count() == 0`** ——
  因为取值名被 `<code class="docutils literal notranslate"><span class="pre">` **包住**（**跨标记**）；
- **本机实测**（`code/` 的 `B.4`）：`NCCL_ALGO` 那句 **raw=0 / plain=1**（跨标记），
  而 `usage/streams.html` 那句 **raw=1 / plain=1**（**不跨**）；
- **规避**：★★★ **「逐字」这个标签要看 raw HTML 的计数** —— 两个计数都要登记（`specs.py` 里成对存放）。

## 5. 判据表

| 可机械判定的条件 | 结论 |
|---|---|
| 见到 `NCCL_ALGO` 的取值 | ★ **8 个**（`Ring` / `Tree` / `Collnet` / `CollnetChain` / `CollnetDirect` / `NVLS` / `NVLSTree` / `PAT`）；★★ **取值【不】大小写敏感**（`env.html` 逐字 `(not case sensitive)`）（`B.5`） |
| 见到「把 `Ring` 写成 `ring` 会失败」 | ★★★ **不成立**：官方说取值 **`(not case sensitive)`**（`B.5`）—— ★ 会失败的只有**不存在的名字**（如 `RingAllReduce`）（`C.1`） |
| 见到 `Collnet` 的 `since` | ★★ **`2.5–2.13`**（**闭区间** ⇒ 已移除）；★ `2.14` 起由 `CollnetChain` / `CollnetDirect` 接替（`B.1`） |
| 问「NCCL 什么时候用 tree」 | ★★★ **官方不说**：默认 `unset` ⇒ **自动选**（`B.2`）；★ 要看实际选择用 `NCCL_DEBUG_SUBSYS=TUNING`（见 `02`） |
| 见到「NCCL 在 X 大小以下用 tree」 | ★★★ **不成立**：那两个阈值变量**已移除**（`2.5` / `2.3`），官方**未公开替代规则**（`F3`，`B.2`） |
| 报 `NCCL_PROTO` 的取值 | ★ `LL` / `LL128` / `Simple`；★★ **官方【劝阻】设置它**（`data corruption`）（`B.3`） |
| 见到「LL128 的机制是……」 | ★★ **先问载体** —— 本文**不写**（`F9`：有载体的可写，但本库**未取到**） |
| 见到「通信对主机永远异步」 | ★★★ **不是** NCCL 文档的引文（8 页 grep ⇒ 0 命中）；★ 可用的是 `D.6` 那**三条**（`F1`） |
| 见到一句「逐字」引文 | ★★ **核对 raw HTML 的 `count()`** —— plain=1 而 raw=0 即**跨标记**（`B.4`） |
| 报「某个 `since` 版本」 | ★★ 去 `NCCL_ALGO_SINCE` **同下标**取；★★ **记错会被版本账拒绝**（`C.2`） |
| 见到一个 `NCCL_ALGO` 取值 | ★★ 先问它**在不在那 8 个里**（`C.1`：`RingAllReduce` 是**编的**） |

## 6. 小结与延伸

**一句话**：**NCCL 把「用哪个算法」与「用哪个协议」露成两个环境变量，但【怎么选它不告诉你】**
—— 默认 `unset`（自动选），而两条曾经的阈值变量都已移除。

**三条可独立复用的结论**：

1. ★★★ **默认 = `unset`** —— 官方**不承诺**选择规则；★ 「阈值表」都是**编的**（`F3`）；
2. ★★ **取值【不】大小写敏感**（官方逐字 `(not case sensitive)`，`B.5`），而**不存在的取值**不报错
   —— ★ 唯一的信号是**校验器拒绝**（`C.1`）；
3. ★★★ **`"Communication is always asynchronous with respect to the host"` 不是引文**
   —— 8 个文档页 0 命中；★ 能引的只有那**三条**（`F1`）。

**延伸**：

- ★★ **NCCL 的拓扑感知（三条官方抓手：变量名 · XML · DEBUG 子系统）** →
  [02-topology-awareness](02-topology-awareness.md)；
- ★★ **报错归因与它的可判定范围（2 / 7）** → [03-failures-and-limits](03-failures-and-limits.md)；
- **本层不写、只回链的正文**：
  · ★★★ **算法族的步数与搬运量** → `110-communication/collectives/01-algorithm-family.md`；
  · ★★ **拓扑受限的集合通信（一句话级）** → `20-gpu-architecture/interconnect/03-collectives-on-topology.md:243-249`；
  · **`V(N)` 的推导与饱和** → `10-foundations/performance-models/02-amdahl-and-scaling.md:147-155,209`。

## 参考

- **NCCL 官方产品文档**，版本 **2.32.3** —— ★★★ **`[B]` 级（产品文档，不是规范）**。
  出处 `https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/`
  （`env.html` · `usage/streams.html` · `usage/groups.html` · `api/group.html`），抓取日 **2026-10-08**。
  ★ 本文用到的逐字引文全部注册在 `code/specs.py` 的 `# 来源：` 里（★ 每条带 URL 页 + 版本 + 抓取日）。
- 本仓库 `20-gpu-architecture/interconnect/03-collectives-on-topology.md:243-249`（拓扑上的一句话级描述）、
  `110-communication/collectives/01-algorithm-family.md`（算法族的代数）、
  [02-topology-awareness](02-topology-awareness.md)、[03-failures-and-limits](03-failures-and-limits.md)。

★ **一条禁写说明**：本文**不引** `LL` / `LL128` 的机制（`F9`：机制要写就必须带载体，
★ 而本库未取到 `prims_ll128.h` 等载体的正文）⇒ ★★ **只写取值与官方警告**。

## 编译与运行

```sh
cd code && make run                 # 三个编号程序，任一断言失败即非零退出
python3 01_architecture_and_selection.py   # 只跑本文的配套程序
make 01_architecture_and_selection  # ★★ 与上一行等价（都先走 check-env）
```

★ 本机的运行条件：`python3 3.13.5`，macOS arm64，**无 GPU、无 NCCL、无真网卡**。
★★★ **本文的程序【没有 A 类】**（`D.0` 机械证明：AST 扫自己，`check("A...")` 调用数 = 0）——
★★ 它守的是「**取值表与版本账抄对了没有**」与「**两条错法真的会被拒**」，
**不守**任何一条真实 NCCL 的行为。
★ 强制限定语：**不是 NCCL，也不是真机** —— 本文的一切数字**抄自官方产品文档**
（版本 `2.32.3`，抓取日 `2026-10-08`），**不是**本机测出来的。

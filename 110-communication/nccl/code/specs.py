"""110-communication/nccl 的外部事实常量源。

★ `CONTRIBUTING` §2 约定的**共享事实源**：本主题的编号程序从它读常量，
  推导过程可复核，**常量本身不可自证**。
★ **只放本主题真的会断言的常量**（无消费者的常量是 `make mutate` 的漏网候选）。
★ **来源分级 A–E**（A 厂商规范 / B 产品官方文档 / C 论文 / D 社区 / E 本库推导）；
  每条带 `# 来源：` + `[A-E]` 级 + **抓取日** + `# 同值位置：` + `# 口径：` + `# 消费者：`。
★★ **所有数值常量取到 `< 2^53`**；**幂次写成十进制字面量**（`mutate.py` 的 `_walk()` 不下钻 `BinOp`）。
★★ **本模块里没有负字面量**（`mutate.py` 的盲区②：负字面量不进分母）。

★★★ **【本主题的定位】—— 它是全库证据强度最低的一章，必须如实写**：
  · ★★★ **全靠 `[B]` 级** —— 载体是 **NCCL 官方【产品文档】**（**不是规范**）；
  · ★★★ **本主题【没有 A 类】** —— 本机**无 GPU、无 NCCL、无真网卡** ⇒
    「算法正确性」**无处可验**。按 `CONTRIBUTING` §2 的第三种形态【写明不可测】，
    **不得把「常量钉住」冒充 A 类**（那是 `P12` 的变体：判据看着在检查，其实只是把常量抄了一遍）。
    ⇒ 本主题的「钉住」**一律归 `D` 类**（照同域 `collectives` 的处置）。

★★★ **【载体与版本账】**：
  · 文档载体 = `https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/`
    （`env.html` / `usage/*` / `api/*` / `troubleshooting/logging.html`），
    ★ **版本一律 `NCCL 2.32.3`**（文档页抬头逐字 `NCCL 2.32.3 documentation`）；
  · ★★ **例外（标 `[B']`）**：`unhandled system error` 那条字面串**只在官方【源码】**里，
    载体 = `v2.26.2-1` 的 `src/init.cc:2198` —— ★★★ **源码版本 `v2.26.2-1` 与文档版本
    `2.32.3`【不同】，行号不保证相同**（见 `NCCL_SYSERR_SOURCE_*`）。
  · 抓取日一律 **2026-10-08**。

★★★ **【本主题的四条禁写清单】实施者**不得**补全其中任何一条（计划 `§C.5`）**：
  · **`F1`** —— 不得把 `"Communication is always asynchronous with respect to the host"`
    当 NCCL 文档的逐字引文（研究者抓 8 个文档页逐页 grep ⇒ **0 命中**）。
    ★ 可用的**已验证**替代表述只有本文件的 `NCCL_ASYNC_DEVICE_QUOTE` /
    `NCCL_GROUP_NONBLOCK_QUOTE` / `NCCL_GROUP_RETURN_QUOTE` 三条（见 `NCCL_NO_F1_QUOTE_NOTE`）。
  · **`F2`** —— 不得写「NCCL 文档说 `unhandled system error` 是……」；
    那个字面串**只在官方源码** ⇒ 要引**必须注明「官方源码」**并标 `[B']`。
  · **`F3`** —— ★★★ **不得写**「NCCL 在 X 大小以下用 tree、以上用 ring」这类**阈值表**：
    `NCCL_TREE_THRESHOLD` **已 `removed in 2.5`**、`NCCL_SINGLE_RING_THRESHOLD` **已 `removed in 2.3`**，
    ★ **官方未公开替代规则**（见 `NCCL_REMOVED_VARS`）。
    ★★ **另一条陷阱**：**只看文档的 TOC 会误判「那两个阈值变量还在用」**（TOC 里**仍列着**，
    正文才写 `removed`）⇒ ★ **凡引「某变量存在/已移除」，去看【正文】而不是 TOC**。
  · **`F9`（已放宽）** —— **有载体的** LL / LL128 机制**可以写**（arXiv `2507.04786`「Demystifying NCCL」、
    NCCL/RCCL 源码如 `prims_ll128.h`），★ 但本主题**没有取到** `prims_ll128.h` 的正文
    ⇒ 本文件**不登记**任何 LL / LL128 的机制常量（★ 只登记 `NCCL_PROTO` 的**取值**与**官方警告**）。

★ **本主题的禁区**：
  **框架 API 的用法**（`ncclGroupStart` 的用法、`torch.distributed` 的类与参数 —— 全库无正文）；
  **任何「NCCL 跑得有多快」的数字**（本机无 NCCL / 无 GPU ⇒ 只能引 `[B]` 级 + 版本 + 抓取日）。
"""


# ===========================================================================
# 0. 载体元数据（三条文档常量 + 一条 `[B']` 源码载体）
# ===========================================================================

NCCL_DOC_VERSION = "2.32.3"
# 来源：[B] NCCL 官方产品文档，**文档页抬头逐字 `NCCL 2.32.3 documentation`**；
#       载体 = `https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/`，抓取日 2026-10-08。
# 口径：★ 它是**产品文档的版本**，**不是规范版本** —— NCCL **没有**可引用的「规范」。
#       ★★ 本主题**每一条**外部事实都要带版本；本常量是**那个版本**。
# 消费者：三个编号程序的 `D.3`（各自钉住一次）。
# 同值位置：无。

NCCL_DOC_BASE_URL = "https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/"
# 来源：[B] —— 出处根路径（抓取日 2026-10-08）。★ 逐条页面的后缀见各常量的 `# 来源：`。
# 口径：★ 用来把每条引文**定位到页面**（`env.html` / `usage/*` / `api/*` / `troubleshooting/logging.html`）。
# 消费者：三个编号程序的 `D.3`。
# 同值位置：无。

NCCL_FETCH_DATE = "2026-10-08"
# 来源：[B] —— 本主题**全部**外部事实的抓取日。
# 口径：★ 写「某版本 + 抓取日」是 `P19`（把 B 级当 A 级）的**第一道防线**：
#       产品文档会随版本改，**不写版本与抓取日的引文不算引文**。
# 消费者：三个编号程序的 `D.3`。
# 同值位置：无。

NCCL_SYSERR_SOURCE_TAG = "v2.26.2-1"
# 来源：**[B'] 官方【源码】的 tag**（不是文档）—— 载体 = NCCL 官方仓库 `src/init.cc`，
#       抓取日 2026-10-08。
# 口径：★★★ **它是【源码】的版本，与 `NCCL_DOC_VERSION`（`2.32.3`）【不同】** ——
#       两者**不是同一个工件**，**行号不保证相同** ⇒ 引用时必须分别写清。
# 消费者：`03_failures_and_limits.py` 的 `C.2`/`D.6`。
# 同值位置：无。

NCCL_SYSERR_SOURCE_LOC = "src/init.cc:2198"
# 来源：**[B'] 官方【源码】的位置**（文件 + 行号）—— 载体 = tag `v2.26.2-1`，抓取日 2026-10-08。
# 口径：★ 它是 `F2` 的落点：那个字面串**在文档里 8 页 grep 0 命中**，**只在源码**里
#       ⇒ ★★ **行号随版本漂移**，引用时必须同时写 tag。
# 消费者：`03_failures_and_limits.py` 的 `D.6`。
# 同值位置：无。


# ===========================================================================
# 1. `NCCL_ALGO` —— 算法取值表与**版本账**（`env.html`，[B] 级）
# ===========================================================================

NCCL_ALGO_QUOTE = "The NCCL_ALGO variable defines which algorithms NCCL will use."
# 来源：[B] NCCL 官方产品文档 `env.html`，版本 2.32.3，抓取日 2026-10-08。
#       ★★ **「不跨标记」实测**（研究者逐页 grep）：本句在**原始 HTML 上 `count() == 0`**
#         （`NCCL_ALGO` 被 `<code class="docutils literal notranslate"><span class="pre">` 包裹）、
#         **去标记文本上 `count() == 1`**（见 `NCCL_ALGO_QUOTE_RAW_COUNT` /
#         `NCCL_ALGO_QUOTE_PLAIN_COUNT`）。
# 口径：★★ **本条是**取值表的**定义句** —— 它只说「定义用哪些算法」，**不说怎么选**。
#       ★★ 「怎么选」在 `NCCL_ALGO_DEFAULT_QUOTE`（默认 `unset` ⇒ 自动选）。
# 消费者：`01_architecture_and_selection.py` 的 `D.5`。
# 同值位置：无。

NCCL_ALGO_QUOTE_RAW_COUNT = 0
# 来源：[B] —— 上一条引文在**原始 HTML**里的 `count()`（研究者实测，抓取日 2026-10-08）。
# 口径：★★ **口径必须写清**：数的是**字面串**在**原始 HTML 字节**里的出现次数。
#       ★★★ 它与 `NCCL_ALGO_QUOTE_PLAIN_COUNT` **不等**（0 对 1）⇒ **该句跨了一个 HTML 标记边界**
#         ⇒ 把它当「逐字」引文时，**必须声明「这是去标记后的文本」**。
# 消费者：`01_architecture_and_selection.py` 的 `B.2`/`D.8`。
# 同值位置：无。

NCCL_ALGO_QUOTE_PLAIN_COUNT = 1
# 来源：[B] —— 同一条引文在**去标记文本**里的 `count()`（研究者实测，抓取日 2026-10-08）。
# 口径：★★ 与上一条**成对使用**；两条**一起**才构成「不跨标记」的判定。
# 消费者：`01_architecture_and_selection.py` 的 `B.2`/`D.8`。
# 同值位置：无。

NCCL_ALGO_NAMES = ("Ring", "Tree", "Collnet", "CollnetChain", "CollnetDirect", "NVLS", "NVLSTree", "PAT")
# 来源：[B] NCCL 官方产品文档 `env.html` 的 `NCCL_ALGO` 取值列，版本 2.32.3，抓取日 2026-10-08。
#       ★★ 本常量把那一列**拆成 8 个名字**（★ 拆开是为了能**逐元素**钉住 ——
#         把整句存成一个串会让**元素级**变异漏网，`mutate.py` 的盲区⑧）。
# 口径：★★★ 这是**官方暴露的可选值**；★ **它【不是】一张「什么时候用哪个」的表** ——
#       默认是 `unset`（自动选，见 `NCCL_ALGO_DEFAULT_QUOTE`）。
#       ★★ **大小写敏感**（文档写 `Ring` 不是 `ring`）⇒ 本常量照文档大小写存。
# 消费者：`01_architecture_and_selection.py` 的 `B.1`/`C.1`/`D.4`。
# 同值位置：`NCCL_ALGO_SINCE`（**同一个下标**）。

NCCL_ALGO_SINCE = ("2.5+", "2.5+", "2.5–2.13", "2.14+", "2.14+", "2.17+", "2.18+", "2.23+")
# 来源：[B] 同上一条的**同一张表**（`env.html` 的 since 列），版本 2.32.3，抓取日 2026-10-08。
#       ★★ 逐字取值（照计划 `§C.2`）：`Ring (2.5+)` · `Tree (2.5+)` · `Collnet (2.5–2.13)` ·
#         `CollnetChain (2.14+)` · `CollnetDirect (2.14+)` · `NVLS (2.17+)` · `NVLSTree (2.18+)` ·
#         `PAT (2.23+)`。★★ `2.5–2.13` 用的是 **en-dash `–`**（照计划形态抄，未改写成 `-`）。
# 口径：★★★ **这就是「版本账」** —— 与 `NCCL_ALGO_NAMES` **同下标一一对应**。
#       ★★★ 全表**只有一个**闭区间（`Collnet`）；**紧接着的 `2.14` 出现两个新取值**
#         （`CollnetChain` / `CollnetDirect`）⇒ 读法是「Collnet 在 2.14 被拆成两条」，
#         ★ **不是**「Collnet 还在、另外多了两条」。
#       ★★ **两条必须逐元素钉住**（`01` 的 `D.4`）。
# 消费者：`01_architecture_and_selection.py` 的 `B.1`/`C.2`/`D.4`。
# 同值位置：`NCCL_ALGO_NAMES`（**同一个下标**）。

NCCL_ALGO_COUNT = 8
# 来源：[B] —— 上一行那个取值表的**元素个数**（★ 逐条数出来的，不是估的）。
# 口径：★★ 取它当常量而不是把 `8` 写死在断言里，是为了让 `mutate.py` 能看见它
#       （★ 同时 `01` 的 `D.4` 用 `== 8` 逐字钉住）。
# 消费者：`01_architecture_and_selection.py` 的 `B.1`/`D.4`。
# 同值位置：`NCCL_ALGO_NAMES`。

NCCL_ALGO_DEFAULT_QUOTE = "The default is unset, which causes NCCL to automatically choose the available algorithms based on the node topology and architecture."
# 来源：[B] NCCL 官方产品文档 `env.html`，版本 2.32.3，抓取日 2026-10-08。
# 口径：★★★ **本主题的最要紧的一条**：**默认是 `unset`** ⇒ **NCCL 自动选**。
#       ★★ 它的**推论**：官方**不承诺**任何「什么时候用哪个算法」的规则
#         ⇒ ★★★ **`F3` 的依据**：不得写「NCCL 在 X 大小以下用 tree」这类阈值表。
# 消费者：`01_architecture_and_selection.py` 的 `B.2`/`D.5`。
# 同值位置：无。

NCCL_ALGO_FABRICATED = "RingAllReduce"
# 来源：**[E] 本库构造的【不存在的取值】** —— 它**不在** `NCCL_ALGO_NAMES` 里
#       （★ 名字看起来很像真的，这正是它作为反例的价值：拼错 / 自作主张的形态）。
# 口径：★★★ 它是 `01` 的 `C.1` 的**输入**：**编一个不存在的 `NCCL_ALGO` 取值**必须被拒。
#       ★ 它**不是**任何来源里的值 ⇒ **不得**出现在正文的取值表里。
# 消费者：`01_architecture_and_selection.py` 的 `C.1`/`D.7`。
# 同值位置：无。

NCCL_BAD_SINCE_CLAIM = "2.14+"
# 来源：**[E] 本库构造的【一个可信但错的】since 断言** —— 拿 `NVLS` 去配 `2.14+`：
#       `2.14` 是**真的存在**的一个版本档（`CollnetChain` / `CollnetDirect` 就在那），
#       所以这个错法**看起来合理**。★ 正确答案是 `2.17+`（见 `NCCL_ALGO_SINCE`）。
# 口径：★★ 它是 `01` 的 `C.2` 的**输入**：**把 `since` 版本记错**必须被版本账拒绝。
# 消费者：`01_architecture_and_selection.py` 的 `C.2`/`D.7`。
# 同值位置：`NCCL_ALGO_SINCE` 的 `CollnetChain` 档（★ **同值但不同指**：那里是**对的**）。


# ===========================================================================
# 2. `NCCL_PROTO` —— 协议取值与**官方警告**（`env.html`，[B] 级）
# ===========================================================================

NCCL_PROTO_QUOTE = "The NCCL_PROTO variable defines which protocol(s) NCCL will be allowed to use."
# 来源：[B] NCCL 官方产品文档 `env.html`，版本 2.32.3，抓取日 2026-10-08。
# 口径：★★ 它是**协议**（不是算法）的定义句 —— ★★ **协议与算法是两条独立的轴**：
#       算法决定「**谁跟谁**说话」（ring / tree / …），协议决定「**怎么打包**那笔消息」
#       （`LL` / `LL128` / `Simple`）。
# 消费者：`01_architecture_and_selection.py` 的 `B.3`/`D.5`。
# 同值位置：无。

NCCL_PROTO_VALUES = ("LL", "LL128", "Simple")
# 来源：[B] 同上 —— 文档逐字 `LL, LL128, and Simple.`，版本 2.32.3，抓取日 2026-10-08。
# 口径：★★ 三个取值**逐元素钉住**；★ **本文件【不】登记它们的机制**
#       （`F9`：机制要写就必须带载体；本库**未取到** `prims_ll128.h` 的正文 ⇒ **不写**）。
# 消费者：`01_architecture_and_selection.py` 的 `B.3`/`D.5`。
# 同值位置：无。

NCCL_PROTO_WARNING_QUOTE = "Users are discouraged from setting this variable … enabling LL128 on platforms that don't support it can lead to data corruption."
# 来源：[B] NCCL 官方产品文档 `env.html`，版本 2.32.3，抓取日 2026-10-08。
#       ★★ 计划 `§C.2` 记录的**逐字形态**就是本行（★ 中间的 `…` 是**计划自己的省略标记**，
#         **不是**原页面的一部分 —— 本库**未取到原文**，故照抄计划形态并如实标明这一点）。
# 口径：★★★ **官方【劝阻】用户设置它**，且给出了理由（`data corruption`）——
#       这是「**协议不是给用户调的**」的官方依据。
# 消费者：`01_architecture_and_selection.py` 的 `B.3`/`D.5`。
# 同值位置：无。


# ===========================================================================
# 3. **异步语义** —— 三条**已验证**的替代表述（`F1` 的落点）
#    ★★★ `F1`：不得用 `"Communication is always asynchronous with respect to the host"`
#      （8 个文档页 grep 0 命中）⇒ 只能用下面三条。
# ===========================================================================

NCCL_ASYNC_DEVICE_QUOTE = "The collective operation is then executed asynchronously on the CUDA device."
# 来源：[B] NCCL 官方产品文档 `usage/streams.html`，版本 2.32.3，抓取日 2026-10-08。
#       ★★ **「不跨标记」实测**：该句 **raw=1 / plain=1** ⇒ **不跨任何 HTML 标记边界**
#         （见 `NCCL_ASYNC_DEVICE_RAW_COUNT` / `NCCL_ASYNC_DEVICE_PLAIN_COUNT`）。
# 口径：★★★ 它是 `F1` 的**第一条替代**：官方说的是「**在 CUDA 设备上异步执行**」，
#       ★ **不是**「对主机永远异步」那颗更强的说法。
# 消费者：`01_architecture_and_selection.py` 的 `D.6`。
# 同值位置：无。

NCCL_ASYNC_DEVICE_RAW_COUNT = 1
# 来源：[B] —— `usage/streams.html` 那句在**原始 HTML**里的 `count()`（抓取日 2026-10-08）。
# 口径：★★ **与 `NCCL_ALGO_QUOTE_RAW_COUNT` 对照**：那条是 `0`（跨标记）、本条是 `1`（不跨）
#       ⇒ ★★ 两条**一起**证明「**逐字**」这个标签**必须看 raw HTML 的计数**，
#         只看去标记文本会把**跨标记的拼接**当成逐字。
# 消费者：`01_architecture_and_selection.py` 的 `B.2`/`D.8`。
# 同值位置：无。

NCCL_ASYNC_DEVICE_PLAIN_COUNT = 1
# 来源：[B] —— 同一条引文在**去标记文本**里的 `count()`（抓取日 2026-10-08）。
# 口径：★★ 与上一条相等 ⇒ 该句**不跨标记**。
# 消费者：`01_architecture_and_selection.py` 的 `B.2`/`D.8`。
# 同值位置：无。

NCCL_GROUP_NONBLOCK_QUOTE = "All subsequent calls to NCCL until ncclGroupEnd will not block due to inter-CPU synchronization."
# 来源：[B] NCCL 官方产品文档 `api/group.html`，版本 2.32.3，抓取日 2026-10-08。
# 口径：★★★ `F1` 的**第二条替代**：官方给的是**组操作**的语义
#       （「直到 `ncclGroupEnd` 之间的后续调用**不会**因 CPU 间同步而阻塞」）——
#       ★ 它是**有条件的**（组内），**不是**「永远异步」。
#       ★★ **本库【不写】`ncclGroupStart` 的用法**（框架 API 的用法是全库禁区，见模块抬头）。
# 消费者：`01_architecture_and_selection.py` 的 `D.6`。
# 同值位置：无。

NCCL_GROUP_RETURN_QUOTE = "can return without having enqueued the operation on the stream"
# 来源：[B] NCCL 官方产品文档 `usage/groups.html`，版本 2.32.3，抓取日 2026-10-08。
#       ★★ 计划 `§C.2` 记录的形态是 `…can return without having enqueued the operation on the stream…`
#         —— 首尾的 `…` 是**计划自己的省略标记**；本常量存**去掉省略号的核心句**
#         （★ 本库**未取到原文**，故**不**声称它是完整句）。
# 口径：★★★ `F1` 的**第三条替代**：官方给的是「**返回时可能还没入队**」这个**较弱**的表述。
# 消费者：`01_architecture_and_selection.py` 的 `D.6`。
# 同值位置：无。

NCCL_NO_F1_QUOTE_NOTE = "★「Communication is always asynchronous with respect to the host」不是 NCCL 文档的逐字引文（8 个文档页逐页 grep ⇒ 0 命中）"
# 来源：**[E] 本库的引用纪律记录**（★ **不是** NCCL 文档里的话）—— 依据是**研究者的实测**
#       （抓 8 个文档页 + 逐页 grep `asynchronous` / `with respect to the host` ⇒ 0 命中）。
# 口径：★★★ 它是一条 **`F1`** 的**否定性记录**：那条字面串**无法归到任何文档页**。
#       ★ 它的存在理由是**可复核**：`01` 的 `D.6` 用 `==` 钉住它，于是「有没有把 F1 当引文」
#         变成了一条**可被变异测试抓住**的断言。
#       ★★ 它**没有外部出处** ⇒ **不得**把它写成引文（本库纪律：**自造真值**是一种错误模式）。
# 消费者：`01_architecture_and_selection.py` 的 `D.6`。
# 同值位置：无。


# ===========================================================================
# 4. **`F3` 的落点** —— 两个**已被移除**的阈值变量（`env.html` 正文，[B] 级）
# ===========================================================================

NCCL_REMOVED_VARS = (("NCCL_TREE_THRESHOLD", "2.5"), ("NCCL_SINGLE_RING_THRESHOLD", "2.3"))
# 来源：[B] NCCL 官方产品文档 `env.html` **正文**，版本 2.32.3，抓取日 2026-10-08。
#       ★★ 逐字（计划 `§C.2`）：`NCCL_TREE_THRESHOLD` **已 `removed in 2.5`**、
#         `NCCL_SINGLE_RING_THRESHOLD` **已 `removed in 2.3`**。
# 口径：★★★ **`F3` 的落点**：**不得写**「NCCL 在 X 大小以下用 tree、以上用 ring」这类阈值表 ——
#       那两个变量**已被移除**，★ **官方未公开替代规则**。
#       ★★★ **一条陷阱（复核方式）**：**只看文档的 TOC 会误判「那两个变量还在用」** ——
#         TOC 里**仍列着它们**，**只有正文**才写 `removed` ⇒ ★★ **凡引「某变量存在/已移除」，
#         去看【正文】而不是 TOC**。
#       ★★ 元组**逐元素**钉住（每条含名字与版本，四个叶子）。
# 消费者：`01_architecture_and_selection.py` 的 `B.2`/`C.2`/`D.7`。
# 同值位置：无。

NCCL_REMOVED_COUNT = 2
# 来源：[B] —— 上一行那两个变量的**个数**（★ 逐条数出来的）。
# 口径：★ 取它当常量是为了让 `mutate.py` 能看见它（`01` 的 `D.7` 用 `== 2` 钉住）。
# 消费者：`01_architecture_and_selection.py` 的 `B.2`/`D.7`。
# 同值位置：`NCCL_REMOVED_VARS`。


# ===========================================================================
# 5. **拓扑感知的三条抓手**（`env.html` / `logging.html`，[B] 级）
#    ★★ 与 `20-gpu-architecture/interconnect/03:243-249` 的**分工**：
#      那边是**7 行定性描述**（ring / tree / collnet-NVLS +「由探测拓扑决定」）；
#      ★★★ **本主题不重写它** —— 只写它**没写的**：**变量名、XML、DEBUG 子系统**这三条**官方抓手**。
# ===========================================================================

NCCL_TOPO_HANDLES = ("NCCL_TOPO_FILE", "NCCL_DEBUG_SUBSYS=GRAPH", "NCCL_DEBUG_SUBSYS=TUNING")
# 来源：[B] NCCL 官方产品文档 `env.html`（前两条）与 `troubleshooting/logging.html`，
#       版本 2.32.3，抓取日 2026-10-08。
#       ★★ 计划 `§C.2` 逐字：`NCCL_DEBUG_SUBSYS=GRAPH` **看拓扑**、`=TUNING` **看算法/协议选择**。
# 口径：★★★ **这是「官方暴露的抓手」的全集（三条）** —— ★ 顺着数得出来，**逐条可数**。
#       ★★ 前两条是**观测**（`GRAPH` / `TUNING` 是 `NCCL_DEBUG_SUBSYS` 的两个取值，
#         语义见 `NCCL_DEBUG_SUBSYS_*`）；第三条是**检测的输入**（`NCCL_TOPO_FILE`）。
#       ★★ 元组**逐元素**钉住。
# 消费者：`02_topology_awareness.py` 的 `B.1`/`C.1`/`C.2`/`D.4`。
# 同值位置：`NCCL_TOPO_HANDLE_CAPS`（**同一个下标**）。

NCCL_TOPO_HANDLE_CAPS = (("NCCL_TOPO_FILE", "INPUT"), ("NCCL_DEBUG_SUBSYS=GRAPH", "OBSERVE"), ("NCCL_DEBUG_SUBSYS=TUNING", "OBSERVE"))
# 来源：**[E] 本库的组织**，★ 依据是三条 `[B]` 级文档事实（见 `NCCL_TOPO_FILE_QUOTE` /
#       `NCCL_DEBUG_SUBSYS_GRAPH` / `NCCL_DEBUG_SUBSYS_TUNING`）——
#       `INPUT` = 「检测的输入」（文档逐字 `load before detecting the topology`）；
#       `OBSERVE` = 「观测窗口」（文档给出的用法是**看**，见 `NCCL_HANG_DIAG_QUOTE` 的用法）。
# 口径：★★★ 它是**本库的消歧模型**：把「**感知**」与「**控制**」分成不同的能力。
#       ★★ **官方暴露的三条抓手【没有】一条的能力是「控制选路」** ——
#         本元组的取值域**只有 `INPUT` 与 `OBSERVE`**（★ **没有** `ROUTE`）。
#       ⇒ ★★ 这就是 `02` 的两条 C 类（`C.1`/`C.2`）的判据来源。
# 消费者：`02_topology_awareness.py` 的 `B.1`/`C.1`/`C.2`/`D.4`。
# 同值位置：`NCCL_TOPO_HANDLES`（**同一个下标**）。

NCCL_TOPO_HANDLE_COUNT = 3
# 来源：[B] —— 上一条抓手表的**元素个数**（★ 逐条数出来的：三条）。
# 口径：★ 取它当常量是为了让 `mutate.py` 能看见它（`02` 的 `D.4` 用 `== 3` 钉住）。
# 消费者：`02_topology_awareness.py` 的 `B.1`/`D.4`。
# 同值位置：`NCCL_TOPO_HANDLES`。

NCCL_TOPO_FILE_QUOTE = "Path to an XML file to load before detecting the topology. By default, NCCL will load /var/run/nvidia-topologyd/virtualTopology.xml if present."
# 来源：[B] NCCL 官方产品文档 `env.html`（`NCCL_TOPO_FILE`），版本 2.32.3，抓取日 2026-10-08。
# 口径：★★ 官方给了**默认路径** `/var/run/nvidia-topologyd/virtualTopology.xml`（★ 逐字）。
#       ★★ 注意它的措辞是 **`load before detecting the topology`** ——
#         它是**检测的输入**，★ **不是**「指定每对设备走哪条路」的选路控制。
# 消费者：`02_topology_awareness.py` 的 `C.1`/`D.5`。
# 同值位置：`NCCL_TOPO_FILE_SINCE`。

NCCL_TOPO_FILE_SINCE = "2.6"
# 来源：[B] NCCL 官方产品文档 `env.html`，版本 2.32.3，抓取日 2026-10-08（★ `NCCL_TOPO_FILE` 的 since）。
# 口径：★★ 与 `NCCL_ALGO_SINCE` 是**同一类账**（★ 逐条可数、口径一致）。
# 消费者：`02_topology_awareness.py` 的 `B.1`/`D.5`。
# 同值位置：`NCCL_TOPO_FILE_QUOTE`。

NCCL_TOPO_DUMP_FILE = "NCCL_TOPO_DUMP_FILE"
# 来源：[B] NCCL 官方产品文档 `env.html`，版本 2.32.3，抓取日 2026-10-08。
#       ★★ 计划 `§C.2` 把它与 `NCCL_TOPO_FILE` 并列为「拓扑感知」的两条变量名。
# 口径：★★ 它是**导出**（dump）拓扑的变量 —— ★ **它的存在正是「观测 ≠ 控制」的旁证**：
#       官方给了**看**的出口（`DUMP` / `GRAPH` / `TUNING`），而**选路**仍是 NCCL 自己算的。
# 消费者：`02_topology_awareness.py` 的 `B.1`/`D.5`。
# 同值位置：无。

NCCL_DEBUG_SUBSYS_GRAPH = "GRAPH"
# 来源：[B] NCCL 官方产品文档 `troubleshooting/logging.html`（`NCCL_DEBUG_SUBSYS`），
#       版本 2.32.3，抓取日 2026-10-08。★★ 计划 `§C.2` 逐字：`NCCL_DEBUG_SUBSYS=GRAPH` **看拓扑**。
# 口径：★★ **非逐字**：`GRAPH` 的语义（看拓扑）是**计划 `§C.2` 记录的**，
#       ★★ **本库未取到那半句的逐字原文** ⇒ 标 `[B]·非逐字`，**不得**当成引文。
#       ★★ 但 `NCCL_DEBUG_SUBSYS` 这个**设施**是**有逐字依据**的（见 `NCCL_HANG_DIAG_QUOTE`）。
# 消费者：`02_topology_awareness.py` 的 `B.1`/`C.2`/`D.5`。
# 同值位置：`NCCL_TOPO_HANDLES` / `NCCL_DEBUG_SUBSYS_TUNING`。

NCCL_DEBUG_SUBSYS_TUNING = "TUNING"
# 来源：[B] 同上（`troubleshooting/logging.html`），版本 2.32.3，抓取日 2026-10-08。
#       ★★ 计划 `§C.2` 逐字：`NCCL_DEBUG_SUBSYS=TUNING` **看算法/协议选择**。
# 口径：★★ **非逐字**（同上一条的理由）。
# 消费者：`02_topology_awareness.py` 的 `B.1`/`D.5`。
# 同值位置：`NCCL_TOPO_HANDLES` / `NCCL_DEBUG_SUBSYS_GRAPH`。

NCCL_TOPO_NO_ROUTE_CAP_NOTE = "★ NCCL 官方暴露的拓扑抓手（NCCL_TOPO_FILE / DEBUG_SUBSYS=GRAPH / =TUNING）里没有一条的能力是「控制选路」——感知拓扑 ≠ 能控制选路"
# 来源：**[E] 本库的消歧记录**（★ **不是** NCCL 文档里的话）—— 依据是 `NCCL_TOPO_HANDLE_CAPS`
#       （三条抓手的取值域只有 `INPUT` / `OBSERVE`）与 `NCCL_ALGO_DEFAULT_QUOTE`
#       （默认 `unset` ⇒ **NCCL 自动选**，不由用户选路）。
# 口径：★★★ 它是一条**能力边界**的记录：**把「感知拓扑」写成「能控制选路」是错的**
#       （`02` 的 `C.1` 判它）。
#       ★★ 它**没有外部出处** ⇒ **不得**把它写成引文；它存在的理由是**可复核**
#         （`02` 的 `D.6` 用 `==` 钉住它）。
# 消费者：`02_topology_awareness.py` 的 `C.1`/`D.6`。
# 同值位置：无。


# ===========================================================================
# 6. **offload 开关**（`env.html`，[B] 级）
# ===========================================================================

NCCL_OFFLOAD_SWITCHES = ("NCCL_NVLS_ENABLE", "NCCL_COLLNET_ENABLE")
# 来源：[B] NCCL 官方产品文档 `env.html`，版本 2.32.3，抓取日 2026-10-08。
#       ★★ 计划 `§C.2`：`NCCL_NVLS_ENABLE`（since 2.17，默认 `2`）·
#         `NCCL_COLLNET_ENABLE`（默认 `0`，★ **计划未给 since**）。
# 口径：★★ 这两条是**开关**（把某条 offload 路径打开/关掉）—— ★★ **与「选路」不同**：
#       开关决定「**允许不允许**用某条路径」，★ **不决定**「**哪条路径承载哪笔消息**」。
#       ★★ 元组**逐元素**钉住。
# 消费者：`02_topology_awareness.py` 的 `B.2`/`D.4`。
# 同值位置：`NCCL_OFFLOAD_DEFAULTS`（**同一个下标**）。

NCCL_OFFLOAD_DEFAULTS = ("2", "0")
# 来源：[B] 同上两条开关的**默认值**（照计划 `§C.2` 逐字），版本 2.32.3，抓取日 2026-10-08。
#       ★★ 注意它们是**字符串 `"2"` / `"0"`**（环境变量的取值），**不是整数** —— 照文档存。
# 口径：★★ 默认 `0`（`COLLNET` 关闭）与默认 `2`（`NVLS` 允许）**都不是 `unset`** ——
#       ★★ 与 `NCCL_ALGO` 的默认（`unset`）**形态不同** ⇒ 别把两类变量的默认混为一谈。
# 消费者：`02_topology_awareness.py` 的 `B.2`/`D.4`。
# 同值位置：`NCCL_OFFLOAD_SWITCHES`（**同一个下标**）。

NCCL_NVLS_ENABLE_SINCE = "2.17"
# 来源：[B] NCCL 官方产品文档 `env.html`，版本 2.32.3，抓取日 2026-10-08
#       （★ `NCCL_NVLS_ENABLE` 的 since —— 计划 `§C.2` 逐字：since 2.17，默认 `2`）。
# 口径：★★ 与 `NCCL_COLLNET_SINCE_UNSTATED` 成对：**这条有 `since`、那条没有** ⇒
#       ★★ 两条并列才能看出「官方对开关的 `since` 记录【不是】齐的」。
# 消费者：`02_topology_awareness.py` 的 `B.2`/`D.4`。
# 同值位置：`NCCL_OFFLOAD_SWITCHES`（**同一个下标**）。

NCCL_COLLNET_SINCE_UNSTATED = ""
# 来源：**[E] 本库的「官方未给」占位** —— `NCCL_COLLNET_ENABLE` 的 `since` 在计划 `§C.2` 的
#       记录里**没有出现** ⇒ ★★ **不得**替它填一个版本。
# 口径：★★★ 空串表示「**官方文档【未】给出**」，★ **它不是一个版本号**
#       （`mutate.py` 会把空串变成 `"X"`，`02` 的 `D.4` 用 `== ""` 钉住它）。
# 消费者：`02_topology_awareness.py` 的 `B.2`/`D.4`。
# 同值位置：无。


# ===========================================================================
# 7. **错误码表与归因的可判定范围**（`api/types.html`，[B] 级）
# ===========================================================================

NCCL_ERROR_CODES = (("ncclUnhandledCudaError", 1), ("ncclSystemError", 2), ("ncclInternalError", 3), ("ncclInvalidArgument", 4), ("ncclInvalidUsage", 5), ("ncclRemoteError", 6), ("ncclInProgress", 7))
# 来源：[B] NCCL 官方产品文档 `api/types.html`，版本 2.32.3，抓取日 2026-10-08。
#       ★★ 计划 `§C.2` 逐字：`ncclUnhandledCudaError (1)` … `ncclInProgress (7)`。
# 口径：★★★ **这是错误码表的【全表】**（7 个）—— ★ 元组**逐元素**钉住（14 个叶子）。
#       ★★ **注意**：**「有错误码」不等于「官方给了归因」** ——
#         归因只覆盖其中一个子集（见 `NCCL_ATTRIBUTED_CODES`）。
# 消费者：`03_failures_and_limits.py` 的 `B.1`/`C.1`/`D.5`。
# 同值位置：`NCCL_ERROR_CODE_COUNT`。

NCCL_ERROR_CODE_COUNT = 7
# 来源：[B] —— 上一行错误码表的**元素个数**（★ 逐条数出来的：`1`…`7`）。
# 口径：★ 取它当常量是为了让 `mutate.py` 能看见它（`03` 的 `D.5` 用 `== 7` 钉住）。
# 消费者：`03_failures_and_limits.py` 的 `B.1`/`D.5`。
# 同值位置：`NCCL_ERROR_CODES`。

NCCL_ERROR_ATTRIBUTION_QUOTE = "ncclUnhandledCudaError and ncclSystemError indicate that a call NCCL made to an external component failed, which caused the NCCL operation to fail. The error message should explain which component the user should look at and try to fix"
# 来源：[B] NCCL 官方产品文档 `api/types.html`，版本 2.32.3，抓取日 2026-10-08。
#       ★★ 计划 `§C.2` 记录的逐字形态到 `…should look at and try to fix …`（结尾的 `…` 是计划自己的
#         省略标记）⇒ ★ 本常量存到 `try to fix` 为止，**不**补全后面。
# 口径：★★★ **本条给出了「归因的可判定范围」**：
#       ★★ 官方**逐字点名**的只有**两个**错误码（`ncclUnhandledCudaError` / `ncclSystemError`），
#         且给的是「**调外部组件失败**」这一条归因。
#       ⇒ ★★★ **它【不是】一条通用规律** —— 7 个错误码里只有 2 个被归因
#         ⇒ **把归因当通用规律**是 `03` 的 `C.1` 判的错法。
# 消费者：`03_failures_and_limits.py` 的 `B.1`/`C.1`/`D.6`。
# 同值位置：`NCCL_ATTRIBUTED_CODES`。

NCCL_ATTRIBUTED_CODES = ("ncclUnhandledCudaError", "ncclSystemError")
# 来源：[B] —— 从 `NCCL_ERROR_ATTRIBUTION_QUOTE` 里**点名的那两个**码（★ 是**读**出来的，不是猜的）。
# 口径：★★★ **把它单列成常量**是为了让「归因覆盖了哪几个」**可数**（2 个 / 共 7 个）。
#       ★★ 元组**逐元素**钉住。
# 消费者：`03_failures_and_limits.py` 的 `B.1`/`C.1`/`D.6`。
# 同值位置：`NCCL_ERROR_ATTRIBUTION_QUOTE`。

NCCL_ATTRIBUTED_CODE_COUNT = 2
# 来源：[B] —— 上一条的**元素个数**（★ 逐条数出来的）。
# 口径：★★ 与 `NCCL_ERROR_CODE_COUNT`（7）**并置**才构成「可判定范围」：`2 / 7`。
# 消费者：`03_failures_and_limits.py` 的 `B.1`/`C.1`/`D.6`。
# 同值位置：`NCCL_ATTRIBUTED_CODES`。

NCCL_ATTRIBUTION_SCOPE_NOTE = "★ 官方对错误码的归因只覆盖 ncclUnhandledCudaError 与 ncclSystemError 两个（共 7 个码）⇒ 不得把它当成通用规律"
# 来源：**[E] 本库的引用纪律记录**（★ **不是** NCCL 文档里的话）—— 依据是
#       `NCCL_ERROR_ATTRIBUTION_QUOTE`（点名两个）与 `NCCL_ERROR_CODES`（共 7 个）。
# 口径：★★★ 它是一条**范围**的记录：**把官方归因升格成「所有 NCCL 错误都表示外部组件失败」是错的**
#       （`03` 的 `C.1` 判它）。
#       ★★ 它**没有外部出处** ⇒ **不得**写成引文；存在理由是**可复核**（`03` 的 `D.6` 钉住它）。
# 消费者：`03_failures_and_limits.py` 的 `C.1`/`D.6`。
# 同值位置：无。

NCCL_ASYNC_ERROR_API = "ncclCommGetAsyncError"
# 来源：[B] NCCL 官方产品文档 `api/types.html`（`ncclCommGetAsyncError` 段），版本 2.32.3，
#       抓取日 2026-10-08。★ 计划 `§C.2`：异步错误**经它报出**。
# 口径：★★ **错误有两条报出通道** —— **同步**的是返回值（那 7 个码）、
#       **异步**的（内核里跑出来的）**不走返回值**，而是**查询这个接口**。
#       ⇒ ★★ 别把「返回值非 0」当成 NCCL 出错的**唯一**信号。
#       ★★ **本库【不写】它的用法**（框架/库 API 的用法是全库禁区，见模块抬头）。
# 消费者：`03_failures_and_limits.py` 的 `B.2`/`D.6`。
# 同值位置：无。

NCCL_ASYNC_ERROR_QUOTE = "will usually not progress and never complete"
# 来源：[B] NCCL 官方产品文档 `api/types.html`（`ncclCommGetAsyncError` 段），版本 2.32.3，
#       抓取日 2026-10-08。★ 计划 `§C.2` 记录的形态是 `…will usually not progress and never complete.`
#       —— 首尾的 `…` 与句点标注是**计划自己的省略标记** ⇒ 本常量存**核心子句**。
# 口径：★★ **异步错误**【不是**靠返回值报的**】：官方给了 `ncclCommGetAsyncError` 这个出口。
# 消费者：`03_failures_and_limits.py` 的 `B.2`/`D.6`。
# 同值位置：无。

NCCL_HANG_DIAG_QUOTE = "If your application hangs during NCCL initialization: NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,BOOTSTRAP,NET ./my_app"
# 来源：[B] NCCL 官方产品文档 `troubleshooting/logging.html`，版本 2.32.3，抓取日 2026-10-08。
# 口径：★★★ **hang 的诊断入口** —— 官方给的是一条**命令行**（开 `INIT,BOOTSTRAP,NET` 三个子系统）。
#       ★★★ **官方【没有】给出 hang 的单一归因**：只有「**开这些子系统看**」这一条入口
#         （`03` 的 `B.1` 判它）⇒ ★ **不得**给 hang 编一个归因。
# 消费者：`03_failures_and_limits.py` 的 `B.1`/`D.6`。
# 同值位置：无。

NCCL_SYSERR_SOURCE_QUOTE = 'case ncclSystemError : return "unhandled system error (run with NCCL_DEBUG=INFO for details)";'
# 来源：**[B'] 官方【源码】**（★ **不是**文档）—— 载体 = NCCL 官方仓库 `src/init.cc:2198`，
#       tag **`v2.26.2-1`**，抓取日 2026-10-08。★★ 计划 `§C.2` 逐字。
# 口径：★★★ **`F2` 的落点**：那个字面串 `unhandled system error` 在**用户指南 8 页 grep 0 命中**
#       ⇒ ★★ **要引它必须注明「官方源码」**并标 `[B']`，**不得**写「NCCL 文档说……」。
#       ★★★ **版本不同**：源码 tag `v2.26.2-1` ≠ 文档 `2.32.3` ⇒ **行号不保证相同**。
# 消费者：`03_failures_and_limits.py` 的 `C.2`/`D.6`。
# 同值位置：`NCCL_SYSERR_SOURCE_TAG` / `NCCL_SYSERR_SOURCE_LOC`。

NCCL_SYSERR_SOURCE_NOTE = "★「unhandled system error」的载体是官方【源码】(v2.26.2-1, src/init.cc:2198)，不是 NCCL 2.32.3 的文档"
# 来源：**[E] 本库的引用纪律记录**（★ **不是**任何来源里的话）—— 依据是 `NCCL_SYSERR_SOURCE_QUOTE`
#       的载体（源码 tag）与 `NCCL_DOC_VERSION`（文档版本，★ 两者不同）。
# 口径：★★★ 它是一条 **`F2`** 的记录：**把源码里的字面串说成「NCCL 文档说的」是错的**
#       （`03` 的 `C.2` 判它）。
#       ★★ 它**没有外部出处** ⇒ **不得**写成引文；存在理由是**可复核**（`03` 的 `D.6` 钉住它）。
# 消费者：`03_failures_and_limits.py` 的 `C.2`/`D.6`。
# 同值位置：无。


# ===========================================================================
# 8. UNSOURCED —— 待办清单，不是免责声明
# ===========================================================================
#
# ★ `CONTRIBUTING` §2：「不得**既保留数字又记 UNSOURCED**」—— 下面每一项都**没有**出现在
#   本模块的正文常量里。
# ★★ **本元组【留空】**（与同域 `collectives` 同处置），理由有两条：
#   · **每一项的字符串都会成为 `make mutate` 的叶子**，而它们**没有消费者**
#     ⇒ 会全部记成**漏网**（本库实测：只用于打印的模块级常量表制造过漏网）；
#     ★★ **而【空元组本身】有消费者**：三个编号程序各用 `specs.UNSOURCED == ()` 钉住它
#       （★ 0 个叶子 ⇒ 不产生变异体）。
#   · 本主题的两处「未核到」已经**写在正文常量自己的 `# 来源：` 里**，不欠一条记录：
#     ① **`NCCL_DEBUG_SUBSYS=GRAPH` / `=TUNING` 的逐字语义** —— 计划 `§C.2` 记录了语义
#        （「看拓扑」/「看算法选择」）但**未给逐字原文** ⇒ 已在
#        `NCCL_DEBUG_SUBSYS_GRAPH` / `NCCL_DEBUG_SUBSYS_TUNING` 的 `# 来源：` 标 `[B]·非逐字`；
#     ② **`NCCL_COLLNET_ENABLE` 的 `since`** —— 计划**未给** ⇒ 已在
#        `NCCL_COLLNET_SINCE_UNSTATED` 记「官方未给」，**不替它填一个版本**。
UNSOURCED = ()

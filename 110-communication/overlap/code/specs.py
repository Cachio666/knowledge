"""110-communication/overlap 的外部事实常量源。

★ `CONTRIBUTING` §2 约定的**共享事实源**：本主题的编号程序从它读常量，
  推导过程可复核，**常量本身不可自证**。
★ **只放本主题真的会断言的常量**（无消费者的常量是 `make mutate` 的漏网候选）。
★ **来源分级 A–E**（A 厂商规范 / B 产品官方文档 / C 论文 / D 社区 / E 本库推导）；
  每条带 `# 来源：` + `[A-E]` 级 + **抓取日** + `# 口径：` + `# 消费者：` + `# 同值位置：`。
★★ **本域的头号错误模式是「把 B 级当 A 级」**（`nccl` 主题整个靠产品文档）。
★★ **所有数值常量取到 `< 2^53`**；**幂次写成十进制字面量**（`mutate.py` 的 `_walk()` 不下钻 `BinOp`）。
★★ **每个元组/列表常量都要有一条【逐元素】`==` 断言**（`mutate.py` 盲区⑧：只钉「整体」会漏元素级变异）。
★★★ **本文件所有字符串常量一律写成【单行】** —— 跨行隐式拼接（`NAME = ("a" "b")`）会被 `ast`
  折成一个 `Constant`、`end_lineno != lineno` ⇒ **静默计入「跳过」**（`mutate.py` 盲区⑦）。

★★★ **本主题的证据强度【分层】**（这是它与同域另外三个主题最大的不同）：
  · **`[B]` 级**（PyTorch DDP / PyTorch async checkpoint / vLLM-Gaudi）—— 产品文档，**不是规范**；
    ★★ 尤其 **DDP 的「桶」是【PyTorch 的实现约定】，不是普适规律**（本库 `P19` 的头号形态）
    ⇒ 引用时**必须带这句限定语**。
  · **`[C]` 级（预印本）**（Weave，`arXiv 2609.21483`）—— ★ **同行评审：未确认**（abs 页无
    `Comments` / `journal-ref`）⇒ 只作**文献对照**，**不得当规范**。
  · **`[E]` 级（本库推导）** —— ★★★ **本主题的核心判据（「能藏多少」）【没有】规范/论文级的一手公式**
    ⇒ 模型 `(C+T)/K + (K−1)·max(C/K, T/K)` **是本库自己的推导**，
    ★★ **不得写成「研究表明」或引任何外部出处**（计划 §C.4 / `F8`）。

★ **本主题的禁区**（一行都不重写，只回链）：
  · ★ **异步落盘的【存储侧】机制**（写盘流水、元数据、重分片）→ `90-training/checkpointing/01,02,03`
    —— ★★ **本主题只写「通信 / 计算与落盘重叠」的那一半**；
  · ★ **点对点通信的实现** → 本域 `collectives`（★ 那是分工，不是禁区）；
  · ★★ **`V(N) = 2(N−1)/N × S` 的推导与 `S` 的饱和** → `10-foundations/performance-models/02`。

★★★ **一条引文纪律（本文件最容易出错的地方）**：
  本主题的载体是 **HTML**（PyTorch 文档 / GitHub 上的 `.md` / arXiv 的 HTML 版）
  ⇒ ★ 数学式在 HTML 里以 **LaTeX 的 `alttext` 属性**存在 —— **那才是「提取形态」**，
    照抄时**不得顺手渲染成排版式**（★ 与 `network-topology` 的「上标被压平」是同一条纪律）。
"""


# ===========================================================================
# 0. 共用的抓取日
# ===========================================================================
FETCH_DATE = "2026-10-08"
# 来源：**本库自记**，不是外部事实 —— 它记的是「本文件所有 curl 的抓取日」。
# 口径：★★ 四条外部来源（PyTorch DDP / PyTorch async checkpoint / vLLM-Gaudi / Weave）
#       **同日抓取**；★ 不写抓取日的引文不算引文（`P19` 的第一道防线）。
# 消费者：`01` 的 `D.3` · `02` 的 `D.3` · `03` 的 `D.3`（三个程序各钉一次）。
# 同值位置：三篇正文的抬头（「抓取日」），以及各常量的 `# 来源：` 注释。


# ===========================================================================
# 1. PyTorch DDP —— [B] 级（产品文档，**不是规范**）—— ★ 01 的「通信分块」机制来源
#    载体 = 官方文档页（见 `DDP_URL`），本轮 **curl 实测 `195611` 字节**，抓取日 2026-10-08。
#    ★★★ 三句逐字全部由本轮 curl 复核（与计划 §C.4 一致）。
# ===========================================================================

DDP_URL = "https://pytorch.org/docs/2.14/notes/ddp.html"
# 来源：**[B] 级**（PyTorch 官方文档）—— 本轮 `curl --max-time 25 -A Mozilla/5.0` 实测
#       **`200`**（`195611` 字节）。
# 口径：★ 只用于可定位；★ 版本写在 `DDP_VER` 里。
# 消费者：`01` 的 `D.3`。
# 同值位置：无。

DDP_VER = "PyTorch 2.14"
# 来源：**[B] 级**（同一页的 URL 里的版本号 `2.14`）。
# 口径：★★ 产品文档**随版本变** ⇒ 引它**必须带版本**（`P19`）。
# 消费者：`01` 的 `D.3`。
# 同值位置：`CKPT_VER`（同一个发布版的另一页）。

DDP_BUCKET_QUOTE = 'To improve communication efficiency, the Reducer organizes parameter gradients into buckets, and reduces one bucket at a time. Bucket size can be configured by setting the bucket_cap_mb argument in DDP constructor.'
# 来源：**[B] 级**（PyTorch DDP 官方文档，§「DDP internal design」附近）★ **本轮 curl 逐字复核**。
# 口径：★★★ **两件事**：① **通信分块**（原文用 `buckets`，它把梯度**切成一块块、逐块规约**）；
#       ② 块大小**是可配置的**（`bucket_cap_mb`）。
#       ★★★ **限定语（必须一起引）**：**这是 PyTorch DDP 的实现约定，不是普适规律** ——
#         「分成 `K` 块」这件事**任何实现都可以不做**（`K = 1` 就是一次规约全部）。
#       ★★ **术语纪律**：本文档体系一律写「**通信分块**（chunking）」，
#         不把 DDP 的 **`bucket`（梯度桶）** 与本库的「通信分块」混写（域 README 的「同名不同指」表）。
# 消费者：`01` 的 `D.5` / `B.2`。
# 同值位置：无。

DDP_ORDER_QUOTE = 'Model parameters are allocated into buckets in (roughly) the reverse order of Model.parameters() from the given model. The reason for using the reverse order is because DDP expects gradients to become ready during the backward pass in approximately that order.'
# 来源：**[B] 级**（同一页）★ **本轮 curl 逐字复核**。
# 口径：★★ **分块的【顺序】有理由**：按参数**逆序**分块，因为梯度在反向传播里
#       **大致按那个顺序就绪** ⇒ ★★ 于是「一块就绪就发一块」能**尽早**发起通信。
#       ★★ 注意原句的限定词是 **`(roughly)`** 与 **`approximately`** —— 照抄保留。
# 消费者：`01` 的 `D.5` / `B.2`。
# 同值位置：无。

DDP_ASYNC_QUOTE = 'When gradients in one bucket are all ready, the Reducer kicks off an asynchronous allreduce on that bucket to calculate mean of gradients across all processes. When all buckets are ready, the Reducer will block waiting for all allreduce operations to finish.'
# 来源：**[B] 级**（同一页）★ **本轮 curl 逐字复核**。
# 口径：★★★ **异步发起**的逐字依据：一块就绪 ⇒ **`kicks off an asynchronous allreduce`**（**发起**，不等待）；
#       全部就绪 ⇒ **`will block waiting for all allreduce operations to finish`**（**最后才等**）。
#       ★★ **两句缺一不可** —— 只引第一句会把「异步」读成「永不等待」。
# 消费者：`01` 的 `D.5` / `B.2`。
# 同值位置：无。


# ===========================================================================
# 2. PyTorch 异步落盘 —— [B] 级 —— ★ 03 的机制来源
#    载体 = 官方文档页（见 `CKPT_URL`），本轮 **curl 实测 `967566` 字节**，抓取日 2026-10-08。
# ===========================================================================

CKPT_URL = "https://docs.pytorch.org/docs/2.14/distributed.checkpoint.html"
# 来源：**[B] 级**（PyTorch 官方文档）—— 本轮 `curl` 实测 **`200`**（`967566` 字节）。
# 口径：★ 只用于可定位。
# 消费者：`03` 的 `D.3`。
# 同值位置：无。

CKPT_VER = "PyTorch 2.14"
# 来源：**[B] 级**（URL 里的版本号）。
# 口径：★★ 引它必须带版本（`P19`）。
# 消费者：`03` 的 `D.3`。
# 同值位置：`DDP_VER`。

CKPT_ASYNC_QUOTE = 'Asynchronous version of save. This code first de-stages the state_dict on to the staging storage (defaults to CPU memory), and then calls the save in a separate thread.'
# 来源：**[B] 级**（`async_save` 的文档）★ **本轮 curl 逐字复核**。
# 口径：★★★ **异步落盘分【两段】**：① **de-stage 到 `staging storage`（默认 CPU 内存）**；
#       ② **在【另一个线程】里 `save`**。
#       ★★ **「异步」指的是【第二段在另一个线程】—— 它【不】说这两段不花时间、也不说不占带宽**
#         ⇒ ★ `03` 的 `C.1` 正是判「把异步落盘当成不占带宽」。
# 消费者：`03` 的 `D.5` / `B.1`。
# 同值位置：无。

CKPT_EXPERIMENTAL_QUOTE = "This feature is experimental and subject to change."
# 来源：**[B] 级**（同一小节正下方的 `Warning` 框）★ **本轮 curl 逐字复核**。
# 口径：★★★ **官方自己的稳定性声明** ⇒ ★ 引这条特性**必须带上它**（「实验性」不是稳定契约）。
# 消费者：`03` 的 `D.5` / `B.1`。
# 同值位置：无。

CKPT_HANDLE_NAMES = ("staging_completion", "upload_completion")
# 来源：**[B] 级**（`AsyncSaveResponse` 的构造参数，逐字拆出）★ **本轮 curl 复核**。
# 口径：★★★ **两个句柄 = 两段**（`staging_completion` 对应 de-stage、`upload_completion` 对应上传）
#       ⇒ ★★ 这**直接支持**「异步落盘有两段、第二段才是走带宽的那一段」这个读法。
# 消费者：`03` 的 `B.1` / `D.5`。
# 同值位置：`CKPT_ASYNC_QUOTE`。

CKPT_ASYNC_TYPE_DEFAULT = "AsyncCheckpointerType.THREAD"
# 来源：**[B] 级**（`async_save` 的默认实参 `async_checkpointer_type=AsyncCheckpointerType.THREAD`）★ 本轮 curl 复核。
# 口径：★ 它坐实了「默认是**线程**」—— ★★ **线程≠进程**：默认**不吃第二份内存**，
#       但**也不把两个阶段的带宽开销变没**。
# 消费者：`03` 的 `B.1` / `D.7`。
# 同值位置：无。


# ===========================================================================
# 3. vLLM-Gaudi 的 row-parallel chunking —— [B] 级 —— ★★ 01 的「机制要求什么条件」的来源
#    载体 = GitHub 上的一份 `.md`（见 `VLLM_GAUDI_URL`，**带 commit 哈希**），
#    本轮 **curl 实测 `8247` 字节**，抓取日 2026-10-08。
#    ★★ 它是本主题**唯一**把「分块流水**在什么条件下才启用**」逐条列出来的载体。
# ===========================================================================

VLLM_GAUDI_URL = "https://raw.githubusercontent.com/vllm-project/vllm-gaudi/fee61639caac0426f3bf0643872adae22843ee28/docs/features/row_parallel_chunking.md"
# 来源：**[B] 级**（产品仓库文档）—— 本轮 `curl` 实测 **`200`**（`8247` 字节）。
#       ★★ URL 里**钉住了 commit 哈希** `fee6163…`（★ 仓库的默认分支会变 ⇒ 引文档**必须钉 commit**）。
# 口径：★ 只用于可定位；★ **不是规范** ⇒ 一切取值标 `[B]`。
# 消费者：`01` 的 `D.3`。
# 同值位置：无。

VLLM_GAUDI_OVERLAP_QUOTE = 'This pipelining allows the all-reduce of chunk i to run concurrently with the matmul of chunk i+1, reducing idle time on both the compute engine and the network interface.'
# 来源：**[B] 级**（`row_parallel_chunking.md` 的正文档）★ **本轮 curl 逐字复核**。
# 口径：★★★ **分块流水的机制逐字**：**第 `i` 块的通信与第 `i+1` 块的计算并行**。
#       ★★ **注意它只说「并行」，【没有】给出收益的公式** —— 收益的判据是本库的 `[E]` 模型（见 `02`）。
# 消费者：`01` 的 `D.5` / `A.1`（机制的定义依据）。
# 同值位置：无。

VLLM_GAUDI_CONDITIONS = ("num_chunks > 1", "reduce_results", "tp_size > 1", "total_tokens >= chunk_threshold")
# 来源：**[B] 级**（同页 §「Chunking Conditions」的**四条**，逐字拆出并去掉列表符号）★ 本轮 curl 复核。
# 口径：★★★ **这四条就是「分块流水要求什么条件」的答案**（本主题 `01` 的核心）：
#       ① 配置了分块（`num_chunks > 1`）；② 这一层真的需要集合通信（`reduce_results`）；
#       ③ 并行度 > 1（`tp_size > 1`）；④ **输入足够大**（`total_tokens >= chunk_threshold`）。
#       ★★ 官方还写明**任一条件不满足就退回单发路径**（见 `VLLM_GAUDI_FALLBACK_QUOTE`）。
#       ★★★ **限定语**：**这是 vLLM-Gaudi 一个层的实现约定，不是普适规律** ——
#         四条条件里的数值（尤其第 ④ 条的阈值）**随实现与硬件变**。
# 消费者：`01` 的 `B.2` / `D.7`。
# 同值位置：`VLLM_GAUDI_FALLBACK_QUOTE`。

VLLM_GAUDI_FALLBACK_QUOTE = "When any condition is not met, the layer falls back to the standard single-shot computation."
# 来源：**[B] 级**（紧接那四条条件的一句）★ **本轮 curl 逐字复核**。
# 口径：★★ **条件不满足 ⇒ 退回单发**（`K = 1`）—— ★★ 这正是本主题「`K = 1` 是无重叠值」
#       这条免费守卫**在实现里的对应物**。
# 消费者：`01` 的 `B.2` / `D.5`。
# 同值位置：`VLLM_GAUDI_CONDITIONS`。

VLLM_GAUDI_CHUNKS_DEFAULT = 1
# 来源：**[B] 级**（同页配置表：`VLLM_ROW_PARALLEL_CHUNKS` 默认 `1`(disabled)）★ 本轮 curl 复核。
# 口径：★★ **默认是 `1` = 关闭** ⇒ ★★ 与「`K = 1` ⇒ 无重叠」这条守卫**同值**。
# 消费者：`01` 的 `B.2`。
# 同值位置：`VLLM_GAUDI_THRESHOLD_DEFAULT`（同一张表的另一个数）。

VLLM_GAUDI_THRESHOLD_DEFAULT = 8192
# 来源：**[B] 级**（同页配置表：`VLLM_ROW_PARALLEL_CHUNK_THRESHOLD` 默认 `8192`）★ 本轮 curl 复核。
# 口径：★★ **它是「输入足够大」的那条阈值的默认值** —— ★ **一个实现约定**，
#       不是「所有系统的分块下限」（「单一实践升格」的反面）。
# 消费者：`01` 的 `B.2`。
# 同值位置：`VLLM_GAUDI_CHUNKS_DEFAULT`。


# ===========================================================================
# 4. Weave（arXiv 2609.21483）—— ★ [C] 级但**同行评审：未确认**（预印本）
#    ★★★ 它是本主题唯一的**文献侧对照**：它的成本模型与**本库的 `[E]` 模型**形式不同
#    （见 `02` 的「差异」一节）。载体 = arXiv 的 HTML 版，本轮 **curl 实测 `221781` 字节**。
#    ★★ 数学式在 HTML 里存成 **LaTeX 的 `alttext` 属性** —— **那才是「提取形态」**。
# ===========================================================================

WEAVE_CARRIER = "Weave: Fine-Grained Dynamic SM Scheduling in an MoE Megakernel for Compute-Communication Overlap"
# 来源：**[C] 级 / 预印本**（arXiv）—— 本轮从 abs 页的 `citation_title` 逐字读到。
#        ★★★ **同行评审：未确认**（abs 页**没有** `Comments` / `journal-ref` 字段）⇒ ★ 不得当规范。
# 口径：★ 它是**载体题名**，只支撑「这条对照来自哪一篇」。
# 消费者：`02` 的 `D.3`。
# 同值位置：`WEAVE_ID`。

WEAVE_ID = "2609.21483"
# 来源：**[C] 级 / 预印本**（abs 页的 `citation_arxiv_id`）★ 本轮 curl 复核。
# 口径：★ 编号本身**已在 `arxiv.org/abs/2609.21483` 上取到**（★ `HTTP 200`，`44648` 字节）
#       —— ★★ 「编号存在」这件事是**核过的**（本库 `P21`：不得引一个不存在的 arXiv 编号）。
# 消费者：`02` 的 `D.3`。
# 同值位置：`WEAVE_URL`。

WEAVE_URL = "https://arxiv.org/abs/2609.21483"
# 来源：**[C] 级 / 预印本** —— 本轮 `curl` 实测 **`200`**（abs 页 `44648` 字节；HTML 版 `221781` 字节）。
# 口径：★ 只用于可定位。
# 消费者：`02` 的 `D.3`。
# 同值位置：`WEAVE_ID`。

WEAVE_TOTAL_TEX = 'T_{\\text{total}}(c,K)=\\max\\!\\big(T_{\\text{comp}}(c,K)+T_{\\text{tail}}(c,K),\\;T_{\\text{comm}}(c)\\big).'
# 来源：**[C] 级 / 预印本**（正文的式 (7)）★ 本轮从 arXiv **HTML 版的 `alttext`** 逐字取到。
#       ★★★ **提取形态**：这是 **LaTeX 源码**（含 `\\max\\!\\big` / `\\;`），**不是排版式** ——
#         本文档体系**照抄 LaTeX**（★ 与 `network-topology` 的「上标压平」是同一条纪律）。
# 口径：★★★ 文献侧的**总时间模型**：`max(计算+尾项, 通信)`。
#       ★★ 注意它的第二项只有 `T_comm`（**没有**加首项）—— ★ 这正是它与本库 `[E]` 模型的分歧点（`02`）。
# 消费者：`02` 的 `D.5` / `B.2`（差异对照的**半边**）。
# 同值位置：`WEAVE_TAIL_TEX`（同一篇的配套式）。

WEAVE_TAIL_TEX = 'T_{\\text{tail}}=(1-\\alpha)\\cdot\\frac{W_{\\text{combine}}}{\\text{BW}(c)\\cdot K}.'
# 来源：**[C] 级 / 预印本**（正文的式 (10)，**分块版本**）★ 本轮从 HTML 版 `alttext` 逐字取到。
#       ★★ 提取形态同上（LaTeX 源码）。
# 口径：★★★ 尾项 `T_tail = (1−α)·W_combine/(BW·K)` —— ★★ **分块把尾项按 `K` 缩小**
#       （原文逐字：`the tail is reduced by a factor of K relative to the unchunked case`）。
#       ★ 在 `α = 0` 下，`W_combine/BW` 就是一次 combine 的**通信时间** ⇒ 尾项 ≈ `T/K`。
# 消费者：`02` 的 `D.5` / `B.2`。
# 同值位置：`WEAVE_TOTAL_TEX`。

WEAVE_PROSE_QUOTE = "The layer is bounded by whichever of the two paths is slower, plus the unhidden tail."
# 来源：**[C] 级 / 预印本**（式 (7) 正后方的一句散文）★ 本轮从 HTML 版逐字取到。
# 口径：★★★ **这句是模型的【读法】**：总时间 = **两条路里较慢的那条** + **没被藏住的尾项**。
#       ★★ 它与 `WEAVE_TOTAL_TEX` 是同一条结论的两种写法（散文 / 公式）⇒ ★ 引用时**两者一起**。
# 消费者：`02` 的 `D.5` / `B.2`。
# 同值位置：`WEAVE_TOTAL_TEX`。


# ===========================================================================
# 5. 本库自选的判据网格 —— ★ [E] 级（**不是外部事实**）
#    ★★ 三个程序各用**自己的一组参数**（★ 任务书：`03` 的 A 类「换一组参数，别照抄」）。
# ===========================================================================

OVERLAP_C_T_GRID = ((10, 10), (10, 3), (3, 10), (1, 1))
# 来源：**[E] 本库推导** —— 本库自选的 `(计算时长 C, 通信时长 T)` 档。
# 口径：★★ 四档覆盖：`C = T`（平衡）· `C > T`（计算主导）· `T > C`（通信主导）· `(1,1)`（最小非平凡）；
#       ★ **单位为「时间单位」**（无量纲）—— 本文档体系**不宣称它们对应任何真实硬件**。
# 消费者：`01` 的 `A.1` / `A.2` / `B.1`。
# 同值位置：无。

OVERLAP_K_GRID = (1, 2, 4, 10, 100)
# 来源：**[E] 本库推导** —— 本库自选的**分块数** `K` 档（★ 含 `K = 1` = 无重叠）。
# 口径：★★ `K` 是**块的个数**（域 README：一律写「**通信分块**」，不写 `bucket`）。
# 消费者：`01` 的 `A.1` / `B.1`。
# 同值位置：无。

P35_WRONG_CASE = (3, 10, 4)
# 来源：**[E] 本库推导** —— 计划 `§D` 的 `D4` / `§D.3` 的 `P35` 的**真错误现场**。
# 口径：★★★ `(C, T, K) = (3, 10, 4)`：**通信 `T` 比计算 `C` 长** ⇒ 错式 `C + T/K` 给出
#       `3 + 2.5 = 5.5`，而**物理下界是 `max(3, 10) = 10`** ⇒ ★★ **它给出一个比下界还小的「时间」**。
#       ★ **成因**：漏了「**第一块的通信无法与计算重叠**」这一项（`P35`）。
# 消费者：`01` 的 `C.1` / `D.7` · `02` 的 `C.1`。
# 同值位置：无。

P35_LOWER_BOUND = 10
# 来源：**[E] 本库推导** —— `P35_WRONG_CASE` 的**物理下界** `max(C, T) = max(3, 10) = 10`。
# 口径：★★ **它就是「总时间 ≥ 较长的那条时间轴」**（★ 这条下界**独立于任何流水模型**）。
# 消费者：`01` 的 `C.1` · `02` 的 `C.1`。
# 同值位置：无。

K_INF = 1000000
# 来源：**[E] 本库推导** —— 免费的**第二条守卫**里的「`K → ∞`」的**数值替身**（取 `10^6`）。
# 口径：★★ 用它是为了**避免真的做无穷**；★ 于是「`K → ∞ ⇒ 下界」只能验到
#       **误差 `≈ C/K_INF`**（本库 `01`/`02` 把它写成**容差**，并如实注明「这是数值替身不是极限」）。
# 消费者：`02` 的 `A.3` · `01` 的 `A.2`。
# 同值位置：无。

HIDE_C_T_GRID = ((12, 4), (4, 12), (7, 7), (1, 20))
# 来源：**[E] 本库推导** —— `02` 自选的 `(C, T)` 档（★ **与 `01` 的三档不同**）。
# 口径：★★ 四档覆盖「计算主导 / 通信主导 / 平衡 / 极端通信主导」；
#       ★ 与 `OVERLAP_C_T_GRID` **不重叠**（★ 免得两章看起来是同一组数的复述）。
# 消费者：`02` 的 `A.1` / `A.3` / `B.1`。
# 同值位置：无。

HIDE_K_GRID = (1, 2, 3, 8, 16)
# 来源：**[E] 本库推导** —— `02` 自选的 `K` 档（★ 含 `K = 1`，且**取到 `16`**）。
# 口径：★ 用于「可藏时间 vs `K`」的表（`B.1`）。
# 消费者：`02` 的 `A.3` / `B.1`。
# 同值位置：无。

SAVE_C_S_GRID = ((20, 5), (5, 20), (9, 9))
# 来源：**[E] 本库推导** —— `03` 自选的 `(计算时长 C, 落盘时长 S)` 档（★ **与 `01`/`02` 都不同**）。
# 口径：★★ 这里的「通信侧」是**落盘**（`S`）而不是 allreduce（`T`）—— ★ 但**模型是同一个**
#       （两条时间轴 + 首项不可重叠）⇒ ★ `03` 的 A 类是**同一判据换一组参数**。
# 消费者：`03` 的 `A.1` / `A.3` / `B.1` / `C.1`。
# 同值位置：无。

SAVE_K_GRID = (1, 2, 5, 20)
# 来源：**[E] 本库推导** —— `03` 自选的落盘分块数档（★ 含 `K = 1` = 整份一次落盘）。
# 口径：★ 用于落盘的**重叠账**（`B.1`）。
# 消费者：`03` 的 `A.1` / `B.1`。
# 同值位置：无。

MIN_VALID_C = 1
# 来源：**[E] 本库推导** —— 口径守卫用的最小值（★ `C`/`T`/`K` **必须 ≥ 1**）。
# 口径：★★ 若网格里出现 `0`，`K = 0` 会 `ZeroDivisionError` ⇒ **崩溃捕获**而不是断言捕获
#       （本库 `P23`：口径守卫要**跑在任何计算之前**）。
# 消费者：`01` 的 `D.3b` · `02` 的 `D.3b` · `03` 的 `D.3b`。
# 同值位置：无。


# ===========================================================================
# 6. UNSOURCED / NOT_QUOTED —— 待办清单与【否定性记录】，都不是免责声明
# ===========================================================================
UNSOURCED = ()
# 来源：**本库自记**。
# 口径：★★ 本主题**没有**「用了但没出处」的常量 ⇒ 留空（`0` 个叶子）。
#       ★★ 空元组**有消费者**（三个程序各钉一次）⇒ 不产生变异体，也不构成漏网候选。
# 消费者：`01` 的 `D.7` · `02` 的 `D.7` · `03` 的 `D.7`。
# 同值位置：无。

NOT_QUOTED = ("Ultra-Scale Playbook 附录 A3 的 overlap 公式",)
# 来源：**否定性记录**（本库**明写「不引什么、为什么」**）。
# 口径：★★★ 计划 `§C.5` 的 `F8` 把 **Ultra-Scale Playbook** 列为「能藏多少」的一个外部载体，
#       而★★ **本环境取不到它**：`huggingface.co` 的 space 页与它的 `raw` / `resolve` 两种
#       `.md` 路径**四次都 `curl --max-time 25` 超时（0 字节）**，`raw.githubusercontent.com` 的
#       `huggingface/ultrascale-playbook` **返回 `404`**（★ 仓库不在这个路径下）。
#       ⇒ ★★ **本库不引它的公式**（★ 只在 `02` 的「未取到」一节里如实登记取不到这件事）。
#       ★ 它是**非空**元组 ⇒ 有叶子 ⇒ 会被 `mutate.py` 变异 ⇒ **必须逐元素 `==` 钉住**。
# 消费者：`02` 的 `D.6`。
# 同值位置：无。

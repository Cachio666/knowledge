"""100-parallelism/zero-fsdp 的外部事实常量源。

本模块是 `CONTRIBUTING` §2 约定的**共享事实源**（`specs.py`）：本主题的编号程序从它读常量，
推导过程可复核，**常量本身不可自证**。

★ **只放本主题真的会断言的常量**：没有消费者的常量就是 `make mutate` 的漏网候选
  （`80` 实测「只用于打印的模块级常量表」制造过 2 条漏网）⇒ **要么给它消费者、要么不写**。
  ★ 本模块里**每一条**都被至少一条 `D` 类断言用 `==` 逐字钉住（见三个程序的 `D.1`–`D.10`）。

★ **来源分级 A–E**（判据见 `CONTRIBUTING` §2：A 厂商规范 / B 产品官方文档 / C 论文 /
  D 社区 / E 本库推导）。**每条常量带**：
    `# 来源：` + `[A-E]` 级 + 抓取日 + `# 同值位置：`
  ★★ **来源必须分级并写明是哪一级** —— 本域的头号错误模式是
    **把「某实现的文档」当成「规范的文档」**（B 级当 A 级）。
    本主题的**两条来源分属两级**，写清它们是本文档最容易出错的地方之一：
      · `1910.02054`（ZeRO）= **[C] 论文**（★ **同行评审：未确认** —— 计划 §C.1 写「否」、
        §C.8 的**同行评审状态汇总**那一列写「未确认」（★ 计划自身冲突）；本库照 §C.8 那一列
        （计划明令「**引用时逐条照抄这一列，不得推断**」）。★ arXiv abs 页**无**
        `Comments` / `journal-ref` 字段 ⇒ 本库**不推断**，**不写「否」**）；
      · `torch.distributed.fsdp` 文档 = **[B] 某实现的官方文档**（★ **不是规范**；
        载体见 §3 抬头，**版本 2.14**）。

★★ **本模块的【载体】与【跨标记】纪律**（★ 这是本主题最容易栽的一条，逐条写给复核者）：
    · **ZeRO 的引文载体 = `https://ar5iv.labs.arxiv.org/html/1910.02054` 的正文**
      （★ **引文经计划 §C.1 转交、2026-09-23 取**；**本机 2026-10-08 复核字节**（ar5iv 全文）
      —— ★★ 本主题**由控制者转交逐字文本**，实施者**不自己抓页面**
      —— 计划 §5 v6.3 实测过一次「抓 ar5iv 大页面卡死 600 秒」）。
    · ★★★ **ar5iv 把变量与公式包在 `<math …>…</math>` 里**（例如 §7 开头的 `P_os` / `P_g` /
      `1.5x` / `N_d`）⇒ **凡是跨了 `<math>` 的整句，在原始 HTML 字节上 `count()` 会得到 0**。
      ⇒ ★★ 本模块因此对**跨标记**的整句**另登记它的【不跨标记】载体片段**（后缀 `_HEAD` /
        `_TAIL` / `_…_QUOTE`），并在每条的 `# 来源：` 里**把两种计数都写明**：
        整句（跨标记）计 **0**、载体片段（不跨）计 **1**（或 `2`）。
      ★★★ **别把 0 次读成「原文没有这句」** —— 那是本库 v6.11 记的**假否定**（「数不到」≠「不存在」）。
    · ★★ **FSDP 的引文**：载体是**客户端渲染的文档页**（`curl` 取回的是空壳，
      本机实测 **1275 字节**）⇒ ★★ **本模块按计划 §C.6 的转写逐字登记**，
      **不声称**在原始字节上复核过（★ 这正是本库反复记的「转写是独立的出错环节」）。

★★★ **本主题的【记事号对应】，不是订正**（★ 计划 §3 Chunk 6 第 2 条点名要求写进正文）：
    `10-foundations/performance-models/02-amdahl-and-scaling.md` 里的 `S` 是**梯度字节**，
    而 ZeRO 的 `Ψ` 是**参数量**（`90-training/checkpointing/01-state-and-bytes.md:109-113` 逐字：
    `denotes model size (number of parameters)`）—— 梯度是 **fp16**、每参数 **2 字节**
    ⇒ **`S = 2Ψ`（字节）** ⇒ 该文的 **`2S → 3S`** **就是**论文的 **`2Ψ → 3Ψ`**（见 `ZERO_S_IS_GRAD_BYTES_MULT`）。
    ★★★ **那句话是【对的】，一个字都不许改** —— 计划 §5 v3 已**复核撤回**对它「订正」的判定
    （成因是 v1/v2 **没引 baseline 的定义**，见 `ZERO_RS_AG_QUOTE` 与计划新立的 `P30`）。

★★ **三类真值来源必须分开标注**：
    ① **外部规格值**：论文原文（**C 级，须注明是否确认同行评审**）、厂商/产品文档（A/B 级）
       ⇒ **本模块存放的就是这一类**。
    ② **本机实测值**：本域程序在本机跑出的数（**须写 python3 与 numpy 版本**）⇒ **不放在本模块**。
    ③ **推导值**：程序算出的量（可手算复核）⇒ **不放在本模块**
       （`CONTRIBUTING` §2：「只放从来源抄来的数，**不放推导值**」）。
    ★★ 本主题的 `ZERO_*_PSI` / `ZERO_*_N` / `ZERO_N_GRID` 那一批是 **[E] 本库设计常量**
      （「本程序的输入取哪一组值」），**不是任何来源里的数** —— 它们的 `# 来源：` 一律写
      `[E] 本库设计`，并写明**为什么取这个值**（可手算复核）。

★★ **本域的两条自缚纪律**：
    · **所有数值常量取到 `< 2^53`** —— `mutate.py` 的浮点变异是 `v + 1.0`，
      大浮点数（`2^53` 以上）会**吞掉**这个增量（变异了等于没变异）。
      ★ 本模块的整数常量最大是 `65536`、浮点常量只有 `TOL_MACHINE` 与 `ZERO_1_5X_VERBATIM`。
    · **不做幂次字面量**（`2^-25` 之类）—— `mutate.py` 的 `_walk()` **不下钻 `BinOp`**
      ⇒ `2.0 ** -6` 整体**不进变异分母**。（★ 本模块**一处幂次都没有**。）
    · **全部字符串写成【单行】** —— 隐式拼接的跨行字面量会被 `_patch()` 整条跳过。
    · ★★ **每个元组/列表常量除被循环消费外，还要有一条【逐元素】`==` 钉住**
      （`make mutate` 的盲区⑧：`for x in GRID:` 只读整体，抓不住元素级变异；
      计划 §5 v6.4 一轮实测 58 条漏网里 38 条出自这一条）。

★★ **【种子】的处置**：本主题的 A/B/C 判据**全部是确定性算术**（字节账 / 划分算术 / 拼接），
  **没有统计量判据** ⇒ 种子写在**函数体内**（**不进 `specs.py`**）——
  改它对任何断言都没有可观察后果，放进分母只会制造一条**不过期漏网**
  （与同域 `sequence-context-parallel` 一致，与 `expert-parallel` 相反：那边的 B 类是统计量）。

★ **本主题的禁区**（细节见 `requirements.txt` 与域 README 的「本域不做什么」表）：
    **优化器状态的更新式**（SGD / momentum / Adam / AdamW）⇒ `90-training/optimizers/01,02`；
    **混合精度的类型选择**与 **`16Ψ` 的【单卡】口径**、**它的字节分解**（`2Ψ+2Ψ+KΨ=16Ψ`）
      ⇒ `90-training/mixed-precision/01-type-selection.md` 与
      `90-training/checkpointing/01-state-and-bytes.md`（**本域只写「分片后」的账**）；
    **`4x`/`8x` 与 `P_os`/`P_os+g` 的对应关系** ⇒ `90-training/checkpointing/02-sharded-save-and-load.md`；
    **分片检查点的落盘与恢复**（元数据三层、重分片、一致性）⇒ 同上的 `02,03`
      （★ 本域只写「**分片**（放进显存）」，不写「**落盘**」）；
    **迭代精化 / 主权重为什么必须 fp32 的【数值】理由**
      ⇒ `10-foundations/numerical-linear-algebra/04-iterative-methods.md:229-266`；
    ★★★ **框架的分布式 API 语义**（`torch.distributed` / `DDP` / `FSDP` 的类与参数）
      ⇒ **全库当前没有正文**（实测 `70-frameworks/pytorch/` 4 篇 1053 行里
      `torch.distributed` / `DDP` / `FSDP` / `ProcessGroup` 命中 **0** 次）——
      ★★ 本域**只在 `03` 引 `sharding_strategy` 的【取值列表】**（标 **B 级 + 版本 2.14 + URL + 抓取日**），
      **不解释其语义**（**P19**：把某实现的文档当规范）。
"""


# ===========================================================================
# 1. ZeRO 论文（arXiv:1910.02054）—— [C] 论文（★ 同行评审：未确认）
#    载体 = `https://ar5iv.labs.arxiv.org/html/1910.02054` 的正文；
#    ★ **引文经计划 §C.1 转交、2026-09-23 取**（由独立研究者 `curl -sSL` 取回）；
#      **本机 2026-10-08 复核字节**（ar5iv 全文）。
#    ★★★ 本主题**不自己抓页面**（计划 §5 v6.3 实测过一次「抓 ar5iv 大页面卡死 600 秒」），
#      实施者忠实转写控制者转交的逐字文本；★★★ 下面是控制者复核过的**双计数**（见模块抬头）。
# ===========================================================================

ZERO_S7_OPEN_QUOTE = "The answer is in two parts: i) ZeRO-DP incurs no additional communication using P_os and P_g, while enabling up to 8x memory reduction, ii) ZeRO-DP incurs a maximum of 1.5x communication when using P_p in addition to P_os and P_g, while further reducing the memory footprint by N_d times."
# 来源：[C] ZeRO 论文 §7 的**开头**（arXiv:1910.02054；**同行评审：未确认** ——
#       ★ arXiv abs 页无 Comments / journal-ref 字段 ⇒ 本库不推断）。
#       载体 = ar5iv 正文（★ 引文经计划 §C.1 转交、**2026-09-23** 取；**本机 2026-10-08 复核字节**）。
# 口径：★★★ **本主题 02 的题眼之一，且它必须是【两句完整的】** ——
#       计划 §5 v3 记录：v1/v2 曾把 `ii)` 的从句（`while further reducing the memory footprint by N_d times`）
#       切下来贴到 `i)` 上，**删掉了 `i)` 真正的结尾**（`while enabling up to 8x memory reduction`），
#       制造了一条**伪引文**。⇒ ★ 本模块**逐字存完整的`两句`**，**不得只引一半**。
#       ★★ **原始 HTML 字节上整句计到 `0`**（`P_os`/`P_g`/`1.5x`/`N_d` 都在 `<math>` 里）
#       ⇒ 它的**不跨标记载体片段**另立三条：`ZERO_S7_OPEN_HEAD` / `ZERO_NO_EXTRA_COMM_QUOTE` /
#         `ZERO_UP_TO_8X_QUOTE` / `ZERO_FOOTPRINT_BY_ND_QUOTE`（★ 各计 1 次，实测）。
# 消费者：`02_communication.py` 的 `D.5`。
# 同值位置：无。

ZERO_S7_OPEN_HEAD = "The answer is in two parts: i)"
# 来源：[C] ZeRO §7 开头的载体片段。载体 = ar5iv 正文，**原始字节 1 次**（★ 引文经计划 §C.1
#       转交、**2026-09-23** 取；**本机 2026-10-08 复核字节**）。
# 口径：★ 上一条**不跨标记**的一段（★ 它停在 `i)`，不含任何 `<math>`）。
# 消费者：`02_communication.py` 的 `D.4`。
# 同值位置：无。

ZERO_NO_EXTRA_COMM_QUOTE = "incurs no additional communication using"
# 来源：[C] ZeRO §7 开头。载体 = ar5iv 正文，**原始字节 1 次**（★ 引文经计划 §C.1 转交、**2026-09-23** 取；**本机 2026-10-08 复核字节**）。
# 口径：★★★ **本主题 02 的出处纪律①的载体** ——
#       **以 §7 开头的这个总结句为准**（原文用 `<math>` 包住 `P_os`/`P_g`，
#       故本片段**停在 `using` 之前**）。
#       ★★ **§1 列表的 `same communication volume as DP` 可作【同向互证】**
#         （见 `ZERO_S1_SAME_COMM_QUOTE` —— `P_os` 与 `P_os+g` 在 §1 里各写了一次）
#         ⇒ ★★ **不得写成「只【能】引 §7 开头这一句」**（那会把它自己认定的另两处出处排除掉）。
#       ★★★ **原文【没有】`P_os` 单独的通信量等式** ⇒ **不得自己造一个 `P_os = 2Ψ`
#       并标成「论文式(N)」**（那是编机制从句 / 编来源）。
# 消费者：`02_communication.py` 的 `D.4`（说明文案也在 `D.4` 里）。
# 同值位置：无。

ZERO_UP_TO_8X_QUOTE = "while enabling up to 8x memory reduction"
# 来源：[C] ZeRO §7 开头。载体 = ar5iv 正文，**原始字节 1 次**（★ 引文经计划 §C.1 转交、**2026-09-23** 取；**本机 2026-10-08 复核字节**）。
# 口径：★★★ 这是 **`i)` 真正的结尾** —— v1/v2 的伪引文**删掉的正是它**。
#       它同时是 90 域那条「`4x`/`8x`」让渡链在论文侧的落点（★ 本主题只引用、不重推分解）。
# 消费者：`02_communication.py` 的 `D.5`。
# 同值位置：`90-training/checkpointing/02-sharded-save-and-load.md` 讲的是**它的归属**（本域不写）。

ZERO_FOOTPRINT_BY_ND_QUOTE = "while further reducing the memory footprint by"
# 来源：[C] ZeRO §7 开头（**`ii)` 的从句**）。载体 = ar5iv 正文，**原始字节 1 次**（★ 引文经计划 §C.1 转交、**2026-09-23** 取；**本机 2026-10-08 复核字节**）。
# 口径：★★★ 这一句**属于 `ii)`**（`P_os+g+p` 那条），**不属于 `i)`** ——
#       v1/v2 的伪引文就是把它**挪到了 `i)` 上**。★ 本片段停在 `by` 之前（`N_d` 在 `<math>` 里）。
# 消费者：`02_communication.py` 的 `D.5`。
# 同值位置：无。

ZERO_BASELINE_QUOTE = "the standard DP incurs 2Ψ data movement during each training step."
# 来源：[C] ZeRO §7.1（**标准 DP 的基线**）。载体 = ar5iv 正文（★ 引文经计划 §C.1 转交、
#       **2026-09-23** 取；**本机 2026-10-08 复核字节**）。
# 口径：★★★ **本主题 02 的出处纪律②的载体**，也是 C 类判据的**基线**。
#       ★ 本常量**止于句末的 `.`**（与 `ZERO_BASELINE_TAIL` 的句号一致 —— 整句逐字照抄）。
#       ★★ **原始 HTML 字节上整句计到 `0`**（`Ψ` 在 `<math>` 里）⇒ **不跨标记**的载体片段
#         另立两条：`ZERO_BASELINE_HEAD`（`the standard DP incurs 2`）与 `ZERO_BASELINE_TAIL`
#         （`data movement during each training step.`）—— ★ 各计 **1** 次。
# 消费者：`02_communication.py` 的 `D.5`。
# 同值位置：`10-foundations/performance-models/02-amdahl-and-scaling.md:219`（`2S`，那里的 `S`
#           = 梯度**字节** = `2Ψ`）与 `:221`（`2S → 3S` 那句 —— ★★ 02 正文引的就是它）
#           —— ★★ 见 `ZERO_S_IS_GRAD_BYTES_MULT`。（★ `:209` 是 **ring all-reduce 的算法描述**，
#           没有字面 `2S` —— 它**不是**这个同值位置。）

ZERO_BASELINE_HEAD = "the standard DP incurs 2"
# 来源：[C] ZeRO §7.1 的载体片段。载体 = ar5iv 正文，**原始字节 1 次**（★ 引文经计划 §C.1 转交、**2026-09-23** 取；**本机 2026-10-08 复核字节**）。
# 口径：★★★ **「`2`」这个字面在这个片段里**（它**不跨** `<math>`，`Ψ` 紧随其后才进标签）。
#       ⇒ 它是 `ZERO_BASELINE_MULTIPLE = 2` 的**逐字依据**。
# 消费者：`02_communication.py` 的 `D.6`。
# 同值位置：无。

ZERO_BASELINE_TAIL = "data movement during each training step."
# 来源：[C] ZeRO §7.1。载体 = ar5iv 正文，**原始字节 1 次**（★ 引文经计划 §C.1 转交、**2026-09-23** 取；**本机 2026-10-08 复核字节**）。
# 口径：★ 基线的**后半截**（`Ψ` 与它前半截之间隔着一个 `<math>`）——
#       两条合起来**在语义上等于** `ZERO_BASELINE_QUOTE` 的整句。
# 消费者：`02_communication.py` 的 `D.6`。
# 同值位置：无。

ZERO_RS_AG_QUOTE = "Both reduce-scatter and all-gather are implemented using a pipelined approach"
# 来源：[C] ZeRO §7.1。载体 = ar5iv 正文，**原始字节 1 次**（★ 引文经计划 §C.1 转交、**2026-09-23** 取；**本机 2026-10-08 复核字节**）。
# 口径：★★★ **「`2Ψ` 里已经包含一次 all-gather」的证据** ——
#       基线 = reduce-scatter(`Ψ`) + all-gather(`Ψ`)。⇒ `P_os+g+p` 的 `3Ψ` 相对基线的增量
#       **恰好是「多一次 all-gather（参数）」**（前进一次、反向再前进一次）。
#       ★★★ **这正是计划新立的 `P30` 的解药**：判定增量的归属时，**必须引 baseline 的定义**
#         （v1/v2 只看了增量的两端 `2Ψ`/`3Ψ`，**没看 `2Ψ` 由什么组成** ⇒ 判错）。
# 消费者：`02_communication.py` 的 `D.5`、`C.1` 的说明文案。
# 同值位置：无。

ZERO_POG_SUM_HEAD = "So the total communication volume per training step is"
# 来源：[C] ZeRO §7.2（`P_os+g` 的推导）。载体 = ar5iv 正文，**原始字节 1 次**（★ 引文经计划 §C.1 转交、**2026-09-23** 取；**本机 2026-10-08 复核字节**）。
# 口径：★ `P_os+g` 的推导句头（后面 `Ψ+Ψ=2Ψ` 在 `<math>` 里）——
#       ★★ 与 `ZERO_POG_SAME_QUOTE` 合起来 = **`2Ψ`**。
# 消费者：`02_communication.py` 的 `D.6`。
# 同值位置：无。

ZERO_POG_SAME_QUOTE = "exactly the same as the baseline DP."
# 来源：[C] ZeRO §7.2。载体 = ar5iv 正文，**原始字节 1 次**（★ 引文经计划 §C.1 转交、**2026-09-23** 取；**本机 2026-10-08 复核字节**）。
# 口径：★★★ **`P_os+g` 的通信量 = 与 baseline 完全相同**（逐字）——
#       本主题 02 的 B 类对照表的**那一格**就是它。
# 消费者：`02_communication.py` 的 `D.5`、`A.2`。
# 同值位置：无。

ZERO_POGP_HEAD = "The total volume is therefore"
# 来源：[C] ZeRO §7.3（`P_os+g+p` 的推导）。载体 = ar5iv 正文，**原始字节 1 次**（★ 引文经计划 §C.1 转交、**2026-09-23** 取；**本机 2026-10-08 复核字节**）。
# 口径：★ `P_os+g+p` 的推导句头（后面 `3Ψ` 在 `<math>` 里）——
#       ★★ 与 `ZERO_POGP_1_5X_QUOTE` 合起来 = **`3Ψ` = `1.5x`**。
# 消费者：`02_communication.py` 的 `D.6`。
# 同值位置：无。

ZERO_POGP_1_5X_QUOTE = "which is 1.5x compared to the baseline."
# 来源：[C] ZeRO §7.3。载体 = ar5iv 正文，**原始字节 1 次**（★ 引文经计划 §C.1 转交、**2026-09-23** 取；**本机 2026-10-08 复核字节**）。
# 口径：★★★ **原始字节上唯一可直接数到的 `1.5x`（在 §7.3）** —— ★★ **这不是「论文只出现过一次」**：
#       论文**三处**出现 `1.5x`（§5.3 `only increases the total communication volume of a baseline DP
#       system to 1.5x`、§7 开头、§7.3）；**另两处在 ar5iv 原始字节上被 `<math>` 切开**
#       （`1.5</math>x`）⇒ `count("1.5x")` **只数到本节这一处**。★ 不要读成「`1.5x` 只出现一次」。
#       ⇒ **本主题的 `1.5x` 逐字依据就在这里**（§7 开头那句的 `1.5x` 在 `<math>` 里、整句计 0）。
# 消费者：`02_communication.py` 的 `D.5`、`A.2`、`C.1`。
# 同值位置：无。

ZERO_S1_50PCT_QUOTE = "modest 50% increase in communication volume"
# 来源：[C] ZeRO §1 的列表项（`P_os+g+p` 那条的结论）。载体 = ar5iv 正文，**原始字节 1 次**（★ 引文经计划 §C.1 转交、**2026-09-23** 取；**本机 2026-10-08 复核字节**）。
# 口径：★★ ★ **`50%` 与 `1.5x` 是同一件事的两种写法**（前者是**增量占比**、后者是**倍数**）
#       ⇒ ★★ **报「增了多少」时必须写明是「多了 50%」还是「是原本的 1.5 倍」**
#         （本库的「同一件事的两种口径」，计划 `P28` 家族）。
# 消费者：`02_communication.py` 的 `D.6`。
# 同值位置：无（它与 `ZERO_POGP_1_5X_QUOTE` 是同一件事的**两种口径**，靠本条互指）。

ZERO_S1_SAME_COMM_QUOTE = "same communication volume as DP"
# 来源：[C] ZeRO §1 的列表项。载体 = ar5iv 正文，**原始字节 2 次**
#       （★ `P_os` 与 `P_os+g` 两条各一次 ⇒ **两处出处**）；
#       ★ 引文经计划 §C.1 转交、**2026-09-23** 取；**本机 2026-10-08 复核字节**。
# 口径：★★★ **`P_os` 与 `P_os+g` 的「不增通信」在 §1 的列表里也各写了一次** ——
#       ★ 这两处**互相印证**（与 §7 开头那句一起构成三条同向证据）。
#       ★★ **「2 次」是【两处出处】，不是同一个串的两份拷贝**（与 90 域的教训**相反**：
#         那边有一处「2 次」来自同一个 `<math>` 的两份注释，**不是两处出处**）。
# 消费者：`02_communication.py` 的 `D.6b`（**计数 2 与「两处出处」一并钉**）。
# 同值位置：无。

ZERO_BASELINE_MULTIPLE = 2
# 来源：[C] ZeRO §7.1 —— 逐字句 `the standard DP incurs 2Ψ …`（★ 那个 `2` 是散文字面，
#       见 `ZERO_BASELINE_HEAD`：`the standard DP incurs 2` 原始字节 1 次）。
# 口径：★★★ **基线 = `2Ψ`**。★ 它**【不是】从 `1.5x` 推来的** ——
#       它是**独立的一处出处**（基线自己那一句）。⇒ 02 的 A 类**用它当参照**。
#       ★★★ **它是 C 类判据的支点**：把 `3Ψ` 套到 `P_os` 上会与它**直接矛盾**。
# 消费者：`02_communication.py` 的 `D.3`、`A.1`/`A.2`、`C.1`。
# 同值位置：`10-foundations/performance-models/02-amdahl-and-scaling.md` 的 `2S`（`S = 2Ψ` 字节）。

ZERO_1_5X_VERBATIM = 1.5
# 来源：[C] ZeRO §7.3 —— 逐字句 `which is 1.5x compared to the baseline.`
#       （载体 = ar5iv 正文，**原始字节 1 次**）。
# 口径：★★★ **在 ar5iv 原始字节上唯一可直接数到的 `1.5x`**（★ §7 开头那句的 `1.5x` 在 `<math>`
#       里、整句计 0；★ 论文另有两处 `1.5x` —— §5.3 与 §7 开头 —— 都数不到，见上一条的口径）。
#       ⇒ 02 的 A 类**用它算 `3Ψ`**：`2Ψ × 1.5 = 3Ψ`（**两端各有独立出处**：`2` 来自 §7.1、
#         `1.5` 来自 §7.3 ⇒ **不是自乘**）。
# 消费者：`02_communication.py` 的 `D.3`、`A.2`、`C.1`。
# 同值位置：`ZERO_S1_50PCT_QUOTE`（同一件事的另一种写法：`50%` 是**增量占比**）。

ZERO_S_IS_GRAD_BYTES_MULT = 2
# 来源：**[E] 本库推导**（★ **不是引文**）——「`10-foundations/performance-models/02` 的 `S`
#       = 梯度**字节**，而 ZeRO 的 `Ψ` = **参数量**，梯度按 **fp16** 存、每参数 **2 字节**
#       ⇒ **`S = 2Ψ`**」。
#       ★ 那两条锚点是：`90-training/checkpointing/01-state-and-bytes.md:109-113` 逐字
#       （`denotes model size (number of parameters)`）与 `:101` 的 **梯度 `2Ψ`（fp16）**。
# 口径：★★★ **这是【记号的桥】，不是订正** —— 它让 `performance-models/02` 的 `2S → 3S`
#       与论文的 `2Ψ → 3Ψ` **一一对应**（★ 计划 §3 Chunk 6 点名要求写进正文）。
#       ★★★ **那句话【是对的】，一个字都不许改**（计划 §5 v3 已**撤回**「订正」判定；
#         成因是 v1/v2 没引 baseline 的定义 ⇒ 新立 `P30`）。
# 消费者：`02_communication.py` 的 `D.3`、`D.7`（桥的自检）。
# 同值位置：`90-training/checkpointing/01-state-and-bytes.md`:101（梯度 `2Ψ`，fp16）。

ZERO_K_MEMORY_MULTIPLE = 12
# 来源：[C] ZeRO §5.1 / Figure 1 的举例口径（★ `K` 的用法），
#       转写锚点 = `90-training/checkpointing/01-state-and-bytes.md:80-89`（`K = 12`，fp32 主权重 4
#       + `m` 4 + `v` 4）。
# 口径：★★★ **出处纪律③**：**`K` 不是通信量** —— 它是**优化器状态的内存乘子**（`= 12`）。
#       ⇒ ★★ 把它写进任何通信量等式都是**口径混淆**（本主题 `02` 的 `D.8` 用**源码级线索**
#       钉住「通信模型里一个 `K` 都没有」）。
#       ★ `KΨ` 是**显存**那一侧的账（本主题 01 用），**不参与**通信量那一侧（02）。
# 消费者：`01_three_stages.py` 的 `D.3`（+`B.3`/`B.4` 的输入）、`02_communication.py` 的 `D.3`、
#         `03_fsdp_correctness.py` 的 `D.3`。
# 同值位置：`90-training/checkpointing/01-state-and-bytes.md`:89（`K = 12`）。


# ===========================================================================
# 2. [E] 本库设计常量 —— 「本程序取哪一组输入」
# ===========================================================================

ZERO_BYTES_PER_PARAM = 2
# 来源：[E] 本库设计（★ 与 `90-training/checkpointing/01` 的 **[C] 口径一致**：
#       参数与梯度是 **fp16** ⇒ 每参数 **2 字节**）。
# 口径：★★ 它只用于**分片后**的字节账（本主题 01）：参数的**字节数** = `2Ψ`。
#       ★★★ **本域不重推 `2Ψ+2Ψ+KΨ=16Ψ` 的分解**（那是 90 域的正文）——
#         本常量只让「分片后每一项还剩几个字节」可算。
# 消费者：`01_three_stages.py` 的 `D.3`、`B.1`/`B.3` 的输入。
# 同值位置：`90-training/checkpointing/01-state-and-bytes.md`:101（参数 fp16 = `2Ψ`）。

ZERO_TOTAL_MULTIPLE = 16
# 来源：[E] 本库设计（`= ZERO_BYTES_PER_PARAM*2 + ZERO_K_MEMORY_MULTIPLE` 的**单卡**口径，
#       与 `90-training/checkpointing/01-state-and-bytes.md:101` 的 `16Ψ` 同值）。
# 口径：★★★ **它是 `16Ψ` —— 【单卡】的口径**（参数 `2Ψ` + 梯度 `2Ψ` + 优化器状态 `12Ψ`）。
#       ★★★ **本域只写「分片后」的账**：`16Ψ/N` 只有在**三项全切**时才等于每卡字节；
#         `P_os` 在 `N` 很大时仍趋近 **`4Ψ`**（参数 + 梯度不切）——**「分片不是除法」**。
#       ★★ **不得**在本主题重推 `2Ψ+2Ψ+KΨ` 的分解（那是 90 域的正文）。
# 消费者：`01_three_stages.py` 的 `D.3`、`B.2`/`B.3` 的输入；
#         `03_fsdp_correctness.py` 的 `D.3`、`B.1` 的输入。
# 同值位置：`90-training/checkpointing/01-state-and-bytes.md`:101。

ZERO_A_PSI = 1024
# 来源：[E] 本库设计。`01` 的 **A 类**用的参数量 `Ψ`（**元素个数**，不是字节）。
# 口径：★★ 取 `1024` 的理由：它**被 `ZERO_A_N = 8` 整除** ⇒ 每个分片 **128** 个元素、
#       分片边界整齐（★ 不整除的处置**不是** 01 的题目，那是 `sequence-context-parallel/03`
#       与 TP 的「头切分」的正文）。
#       ★★ **必须逐元素钉住**（`01` 的 `D.7`）—— 只钉「整体」会让元素级变异漏网。
# 消费者：`01_three_stages.py` 的 `D.3b`、`D.4`；★ 另作 `A.1`/`A.2` 的输入。
# 同值位置：无。

ZERO_A_N = 8
# 来源：[E] 本库设计。`01` 的 **A 类**用的 DP 度 `N`。
# 口径：★★ 取 `8` 的理由：整除 `ZERO_A_PSI = 1024`。
#       ★★ 本主题 `01` 的 `P_os` 那格在 `N = 8` 下是 `5.5Ψ`（`2 + 2 + 12/8`）。
#       ★★★ **本库【没有】「90 域记了这个数」这回事** —— `90-training/checkpointing/01-state-and-bytes.md`
#         **既没有 `ZeRO-1` 这个词、也没有 `5.5` 这个数**（它把「ZeRO 三阶段」**让渡**给了
#         `100-parallelism`）⇒ **不得写「独立复现了 90 域」**（那是造一个不存在的出处）。
# 消费者：`01_three_stages.py` 的 `D.3b`、`D.4`；★ 另作 `A.1`/`A.2` 的输入。
# 同值位置：**计划 §D3 那一行**（`ZeRO-1`（`2+2+12/N`）在 `N=8` 记 `5.500000Ψ`）——
#           ★★ 这是**本库**的同值位置；**90 域没有同值位置**。

ZERO_B_PSI = 4096
# 来源：[E] 本库设计。`01` 的 **B 类**（`P_os` 收益饱和）用的参数量 `Ψ`。
# 口径：★★★ 取 `4096` 的**唯一理由**：它要被 `ZERO_N_GRID` 的**最大档 `4096` 整除**
#       ⇒ B 类能**真的建出分片**（而不是只代公式）⇒ 报出的数**是本机实测值**、可复核。
#       ★★ **必须逐元素钉住**（`01` 的 `D.7`）。
# 消费者：`01_three_stages.py` 的 `D.3b`、`D.4`；★ 另作 `B.1`/`B.3` 的输入。
# 同值位置：无。

ZERO_N_GRID = (1, 2, 8, 64, 512, 4096)
# 来源：[E] 本库设计。`01` 的 B 类用的**六档 DP 度**（★ 计划 §3 Chunk 6 点名的六档）。
# 口径：★★★ **第 1 档 `N = 1` 是退化对照**（`P_os` = `16Ψ`，与不分片同值）；
#       随 `N` 增大 `P_os` **单调下降但收敛到 `4Ψ`、不是 `0`** —— **这正是「收益饱和」**。
#       ★★ **必须逐元素钉住**（`01` 的 `D.7`）—— `for n in specs.ZERO_N_GRID:` 只读整体，
#         **抓不住元素级变异**（本库 v6.4 一轮 58 条漏网里 38 条出自这一条）。
# 消费者：`01_three_stages.py` 的 `D.3b`、`D.7`；★ 另作 `B.1`/`B.2`/`B.3`/`B.5` 的输入。
# 同值位置：无。

ZERO_STAGE_NAMES = ("ZeRO-0", "P_os", "P_os+g", "P_os+g+p")
# 来源：[E] 本库设计。`01` 的 A 类要**真的建出四种分片状态**，这是它们的名字（**顺序即严重度**）。
# 口径：★★★ **四种状态【切了哪几项】必须与名字对应**（`ZeRO-0` 不切 / `P_os` 切优化器状态 /
#       `P_os+g` 再切梯度 / `P_os+g+p` 再切参数）——
#       ★★ `00-meta/glossary.md` 已有 `ZeRO` 词条 ⇒ 本域**不得另立同名词条**（Chunk 7 并入）。
#       ★★ **必须逐元素钉住**（`01` 的 `D.7`）。
# 消费者：`01_three_stages.py` 的 `D.7`；★ 另作 `A.1`/`A.2` 的**循环输入**（打印与逐格核对）。
# 同值位置：无。

ZERO_C_PSI = 65536
# 来源：[E] 本库设计。`03` 的 **B 类**（三种 `sharding_strategy` 的每设备字节账）用的参数量。
# 口径：★★ 取 `65536` 的理由：它被 `ZERO_C_N = 4` 整除，且**与 01 的 `Ψ` 不同** ——
#       ★★ **刻意与 01 分开**，免得两处的数被误认为可互推。
# 消费者：`03_fsdp_correctness.py` 的 `D.3b`、`D.7`；★ 另作 `B.1`/`C.1` 的输入。
# 同值位置：无。

ZERO_C_N = 4
# 来源：[E] 本库设计。`03` 的 B 类用的 DP 度 `N`（★ 计划 §3 Chunk 6 给的是 `N = 4`）。
# 口径：★★ **它与 `01` 的 `N = 8` 不同** —— ★★★ 这就是「`SHARD_GRAD_OP` 在 `N=4` 的 `5.5Ψ`
#       与 `ZeRO-1` 在 `N=8` 的 `5.5000Ψ` **数值相同但口径不同**」的由来：
#       **前者是 `2 + 2/4 + 12/4`、后者是 `2 + 2 + 12/8`** ⇒ 报数时**必须带上 `N`**。
# 消费者：`03_fsdp_correctness.py` 的 `D.3b`、`D.7`；★ 另作 `A.1`/`B.1`/`C.1` 的输入。
# 同值位置：无。

TOL_MACHINE = 1.0e-12
# 来源：[E] 本库设计。**机器精度档**的判据阈。
# 口径：★★ 本主题的判据有两类：**逐位对拍**（分片拼回 / 字节账的整数相等）差落在 `0`；
#       **内容级错法**（只拼一半的分片）差落在 `1e-1` 档 —— 相隔十几个数量级。
#       取 `1e-12` 是为了容纳「恰好为 `0`」与「几 ulp」两种正确形态。
#       ★★ **不得把它当作「物理量级的阈」**。
# 消费者：`01`/`02`/`03` **各自**的 `D.3`（`preflight()`，先跑）。★ 本主题的 `C` 类阈值
#         （内容级）**写在程序内部**（`1e-6` / `1e-3` 之类），**不进本模块**
#         （`CONTRIBUTING` §2：只放从来源抄来的数，不放口径）。
# 同值位置：`data-parallel`/`tensor-parallel`/`pipeline-parallel`/`sequence-context-parallel`/
#           `expert-parallel` 的**编号程序**里各有一个同名常量；
#           ★★ **`03` 的 A 类「分片拼回逐位相同」不用它**（那是**精确的 `0`**，不是「几 ulp」）。


# ===========================================================================
# 3. PyTorch `torch.distributed.fsdp` 文档 —— [B] 某实现的官方文档（**不是规范**）
#    ★★★ 载体 = `https://docs.pytorch.org/docs/stable/fsdp.html`
#      （★ **它会 302 到 `/docs/2.14/fsdp.html`** ⇒ 引用写 **版本 2.14**）。
#      抓取日 **2026-09-23**；页面自称 `Last Updated On: May 12, 2026`。
#    ★★★ **可核性**：本机 `curl -sSL` 取回的是**客户端渲染的空壳**（1275 字节）
#      ⇒ ★★ **本模块按计划 §C.6 的转写逐字登记，不声称在原始字节上复核过**
#      （★ 本库反复记的「转写是独立的出错环节」；★ 与 ZeRO 那批**不同**：那批有原始字节计数）。
#    ★★★ **（P19）本模块【只引取值列表】**：**不解释**框架 API 的通用语义、
#      **不写** `torch.distributed` / `DDP` / `ProcessGroup` 的用法 ——
#      ★★ **全库当前在这上面没有正文**（实测 `70-frameworks/pytorch/` 4 篇 1053 行里
#      `torch.distributed` / `DDP` / `FSDP` / `ProcessGroup` 命中 **0** 次）
#      ⇒ **把禁区指向 70 域是【假让渡】**。
# ===========================================================================

FSDP_DOC_VERSION = "2.14"
# 来源：[B] PyTorch 官方文档。★ `/docs/stable/fsdp.html` **302 到 `/docs/2.14/fsdp.html`**。
# 口径：★★ **版本号必须写出来**（`CONTRIBUTING:99-101`：真系统那一档要写版本）——
#       否则「某实现的文档」会随版本漂移（本库 `80-model-architectures` 的行号锚点教训同源）。
# 消费者：`03_fsdp_correctness.py` 的 `D.5`、`D.5b`。
# 同值位置：无。

FSDP_DOC_URL = "https://docs.pytorch.org/docs/stable/fsdp.html"
# 来源：[B] PyTorch 官方文档的**入口 URL**（★ 它 302 到 `/docs/2.14/fsdp.html`）。
# 口径：★★ **把 URL 也登记成常量**：本域的头号错误模式（B 级当 A 级）的前一步就是
#       **载体指错地方**（同域 `expert-parallel` 实测过一次「指到 663 字节的 `{include}` 壳」）。
# 消费者：`03_fsdp_correctness.py` 的 `D.5b`。
# 同值位置：无。

FSDP_INSPIRED_QUOTE = "This is inspired by Xu et al. as well as the ZeRO Stage 3 from DeepSpeed."
# 来源：[B] PyTorch `torch.distributed.fsdp` 文档（版本 2.14，抓取日 2026-09-23，
#       按计划 §C.6 的转写逐字登记；★ 本机取回的是客户端渲染空壳 ⇒ **不声称原生字节复核**）。
# 口径：★★★ **FSDP 与 ZeRO-3 的关系的逐字依据** —— 本主题 03 用它把
#       「FSDP 的 `FULL_SHARD` ≈ ZeRO-3」这件事**锚成一句有出处的话**。
#       ★★ **它说的是 `ZeRO Stage 3`（`DeepSpeed` 的第三阶段）** —— 本主题**不解释**其语义，
#         只写它自己的**静态账**（每设备字节）。
# 消费者：`03_fsdp_correctness.py` 的 `D.5`、`B.1` 的说明文案。
# 同值位置：无。

FSDP_FULL_SHARD_ALLGATHER_QUOTE = "unshards (via all-gather) before the forward, reshards after the forward, unshards before the backward computation, and reshards after the backward computation"
# 来源：[B] PyTorch `torch.distributed.fsdp` 文档（版本 2.14，抓取日 2026-09-23，转写登记）。
# 口径：★★★ **「什么时刻谁手里有哪一块」的一手依据**（本域主线的落点②）——
#       `FULL_SHARD` 的参数在**前向后 reshard、反向后 reshard**（即**用完即丢**）。
#       ★★ **这是【可以写的】**（它有载体）；★★★ 但**不得**把它扩写成「FSDP 怎么用」的教程
#         （`P19`：那是框架 API 语义，**全库没有正文**）。
# 消费者：`03_fsdp_correctness.py` 的 `D.5`、`D.5b`（★ 与版本、URL 同一条链）。
# 同值位置：无。

FSDP_STRATEGY_VALUES = ("FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD", "HYBRID_SHARD", "_HYBRID_SHARD_ZERO2")
# 来源：[B] PyTorch `torch.distributed.fsdp` 文档（版本 2.14，抓取日 2026-09-23，转写登记）。
# 口径：★★★ **本模块【只引这个取值列表】** —— **不解释每个取值的语义**（`P19`）。
#       ★★ **五个取值**：`FULL_SHARD` / `SHARD_GRAD_OP` / `NO_SHARD` / `HYBRID_SHARD` /
#         `_HYBRID_SHARD_ZERO2`（★ 最后一个**带下划线前缀**，**照抄**）。
#       ★★ **必须逐元素钉住**（`03` 的 `D.7`）—— 只钉「整体」会让元素级变异漏网。
# 消费者：`03_fsdp_correctness.py` 的 `D.5`、`D.5b`、`D.7`；★ 另作 `B.1` 的**输入**。
# 同值位置：无。

FSDP_DEFAULT_STRATEGY = "FULL_SHARD"
# 来源：[B] PyTorch `torch.distributed.fsdp` 文档（版本 2.14，抓取日 2026-09-23，转写登记）。
# 口径：★★ **文档明写默认值是 `FULL_SHARD`** —— ★★ 登记它是因为**默认值**是「读者最容易猜错」
#       的那一项（本库的「同名不同指 / 口径」家族：默认 ≠ 唯一）。
# 消费者：`03_fsdp_correctness.py` 的 `D.5`、`D.5b`。
# 同值位置：无。

FSDP_LAST_UPDATED = "May 12, 2026"
# 来源：[B] PyTorch `torch.distributed.fsdp` 文档（版本 2.14）**页面自称**的更新日。
# 口径：★★ **它是【活文档】的自我标注**（页脚 `Last Updated On`）——
#       ★ 引用时写明抓取日，且**不做行号锚点**（本库 `80-model-architectures` 的教训：
#         行号锚点会随维护失效）。
# 消费者：`03_fsdp_correctness.py` 的 `D.5b`。
# 同值位置：无。


# ===========================================================================
# 4. UNSOURCED —— 待办清单，不是免责声明
# ===========================================================================
#
# ★ `CONTRIBUTING` §2：「不得**既保留数字又记 UNSOURCED**」—— 下面每一项都**没有**出现在
#   本模块的正文常量里。★ 本元组的字符串**没有消费者** ⇒ 它们会是 `make mutate` 的
#   **漏网候选**（本库的既有形态：`90-training/*/code/specs.py` 同节，由编号程序的 `part_D`
#   各钉一次，**不给它开后门**）。

# ★★ 本元组**留空**（与同域 `pipeline-parallel` / `sequence-context-parallel` / `expert-parallel`
#   的形态一致）—— 理由有两条，**都是实测教训**：
#   · **每一项的字符串都会成为 `make mutate` 的叶子**，而它们**没有消费者**
#     ⇒ 会全部记成**漏网**（`80` 实测「只用于打印的模块级常量表」制造过 2 条）。
#     ★★ **而【空元组本身】有消费者**：`01` 的 `D.10` 用 `specs.UNSOURCED == ()` 逐位钉住它
#       （★ 0 个叶子 ⇒ 不产生变异体）。
#   · 本主题的两处「未取到 / 未复核」已经**写在各自常量自己的 `# 来源：` 里**，不欠一条记录：
#     ① **FSDP 五个取值 / 两句引文**：本机 `curl` 到的是**客户端渲染空壳**（1275 字节）
#        ⇒ **未在原始字节上复核**（已写在 §3 抬头的「可核性」那段）；
#     ② **`P_os` 单独的通信量等式** —— ★★★ **原文【本来就没有】**，**不是「未取到」** ——
#        它是一条**禁写**：**不得自己造一个 `P_os = 2Ψ` 并标成「论文式(N)」**
#        （★ 依据只有 §7 开头的**总结句**，见 `ZERO_NO_EXTRA_COMM_QUOTE`）。
UNSOURCED = ()

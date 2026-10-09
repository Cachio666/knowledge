"""100-parallelism/expert-parallel 的外部事实常量源。

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
    本主题的**三条来源分属两级**，写清它们是本文档最容易出错的地方之一：
      · `2101.03961`（Switch Transformer）= **[C] 论文**（★ 同行评审：**是** ——
        依据在 **arXiv abs 页**上：该页 `Comments` 字段的渲染文本逐字是 `JMLR`
        （原始字节 `JMLR` **1** 次、`Comments` **1** 次）；见 `SWITCH_ABS_JMLR` 那一条）；
      · `2006.16668`（GShard）= **[C] 论文**（★★ 同行评审：**本模块不声称已确认** ——
        arXiv abs 页上 `Comments` / `JMLR` / `ICLR` / `NeurIPS` / `Accepted` / `Proceedings` /
        `Conference` / `MLSys` / `Journal` **各 0** 次（抓取日 2026-10-08）
        ⇒ ★★ **凡引用本论文的结论，一律带「同行评审未确认」这一句**）；
      · `megatron/core/transformer/moe/README.md` = **[B] 某实现的官方文档**
        （★ **不是规范**）。★★ 它是一条**活文档**（无版本号、随 `main` 分支变动）
        ⇒ 引用时**写明抓取日**、**不做行号锚点**
        （本库 `80-model-architectures` 的实测教训：行号锚点会随维护失效）。
        ★★★ 注意它讲的是 **`EP` 与 `DP`** 的传统约束，★ **不是** `EP` 与 `E` 的 ——
        见下面 `EP_MOE_README_QUOTE` 与 `EP_NOT_FOUND_NOTE` 两条。

★★ **本模块的【载体】选择**（★ 逐条都是本任务书要求的「自己 curl 核实」，不是转述）：
    · Switch Transformer 与 GShard 的引文载体都是 **`ar5iv.labs.arxiv.org` 的正文页**
      （URL 见两节抬头）；同行评审的依据载体是 **`arxiv.org/abs/<id>` 的 abs 页**；
    · Megatron MoE 文档的载体是 **GitHub `raw.githubusercontent.com` 上的
      `megatron/core/transformer/moe/README.md`**；
    · ★ 抓取日一律 **2026-10-08**，`curl -sSL --max-time 30` 取回**原始字节**后**逐条 `count()`**
      —— 下面每条的 `# 来源：` 里都写了实测次数。
    · ★★★ **引文片段一律【不跨标记】**：ar5iv 把变量与公式包在 `<math …>…</math>` 里
      （例如 Switch §5.4 的 `[E, C, d_model]`），公式编号包在 `<a …>` 里，
      **附录 F 的伪码更狠 —— 它把每一个词都切成独立的 `<span>`** ⇒
      `All-to-All communication.` 这个串在原始字节上**计到 `0`**
      （★★ 本模块因此**只登记 `All-to-All` 这个词本身**、**不登记那个句子**）。
      凡引言 **一律以 `… of size` / `… cores.` 这类词收尾**，**绝不把宾语公式或编号写进常量**。

★★★ **【数不到】与【不存在】是两件事**（本主题 Chunk 5 的**实测教训**：
    本模块的 02 曾据此写下一条**错的**否定性断言，见 `SWITCH_DROPPED_RATE_QUOTE`）：
    `raw.count(x) == 0` **只能**得出「**在这个载体上、用这个片段**数不到 `x`」——
    ★★★ **要断言「不存在」，必须【换锚点、换窗口、换载体】再试**。
    ★ 实例：本主题曾用 **`capacity`** 当锚点在两篇正文里数 `%`、各得 **0** 次，据此写下
      「本库在任何载体上都没有取到任何来源的丢弃率数字」—— ★★★ **那句话是错的**：
      Switch §2.2 **逐字**就有 `… the number of tokens dropped (typically <1%)`
      （见 `SWITCH_DROPPED_RATE_QUOTE` / `SWITCH_DROPPED_RATE_ALT`）。
      ★ 错因是**锚点选错**：那句话的锚点是 **`dropped tokens`**，而 `capacity` 与它附近那个 `%`
      **相距 694 字节**（2026-10-08 在 ar5iv 正文上实测）。
    ⇒ ★★ **凡写「本库在任何载体上都没有取到 X」，先自问：我换过锚点吗？换过窗口吗？换过载体吗？**
      ★ 三层错叠加的先例：本条曾同时出现「**假否定**」（数不到当成不存在）
      + 「**据假否定立的禁用规则**」（「引任何来源的丢弃率数字 ⇒ 不行」）—— 两条一起撤。

★★ **三类真值来源必须分开标注**：
    ① **外部规格值**：论文原文（**C 级，须注明是否确认同行评审**）、厂商/产品文档（A/B 级）
       ⇒ **本模块存放的就是这一类**。
    ② **本机实测值**：本域程序在本机跑出的数（**须写 python3 与 numpy 版本**）⇒ **不放在本模块**。
    ③ **推导值**：程序算出的量（可手算复核）⇒ **不放在本模块**
       （`CONTRIBUTING` §2：「只放从来源抄来的数，**不放推导值**」）。
    ★★ 本主题的 `EP_*` 那批（`EP_DEV_GRID` / `EP_SHARD_CASES` / `EP_DISPATCH_SHAPE` /
      `EP_A2A_DIM` / `EP_IMB_SEEDS` / `EP_IMB_MODEL` / `EP_CAP_LIMIT` / `EP_TP_GRID` /
      `EP_AXIS_SHAPE`）是 **[E] 本库设计常量**（「本程序的输入取哪一组值」），
      **不是任何来源里的数** —— 它们的 `# 来源：` 一律写 `[E] 本库设计`，
      并写明**为什么取这个值**（可手算复核）。

★★ **本域的两条自缚纪律**：
    · **所有数值常量取到 `< 2^53`** —— `mutate.py` 的浮点变异是 `v + 1.0`，
      大浮点数（`2^53` 以上）会**吞掉**这个增量（变异了等于没变异）。
      ★ 本模块的整数常量最大是 `2026`、浮点常量只有 `TOL_MACHINE`。
    · **不做幂次字面量**（`2^-25` 之类）—— `mutate.py` 的 `_walk()` **不下钻 `BinOp`**
      ⇒ `2.0 ** -6` 整体**不进变异分母**。（★ 本模块**一处幂次都没有**；
      `TOL_MACHINE` 是十进制字面量。）
    · **全部字符串写成【单行】** —— 隐式拼接的跨行字面量会被 `_patch()` 整条跳过
      （★ 本主题的第一版若写成跨行会栽在这上面；`sequence-context-parallel` 实测过 14 条）。

★★ **【种子】的处置**（与同域 `sequence-context-parallel` 的落法**不同**，理由写明）：
  那边**不把种子放进 `specs.py`**（改它没有可观察后果 ⇒ 进分母只会制造一条不过期漏网）。
  ★ 本主题的 `02` 不一样：它的 B 类判据是**统计量**（8 个种子的均值与范围），
    **种子集本身就是判据的口径的一部分** ⇒ 它**必须**进 `specs.py` 并被 `==` 逐元素钉住
    （否则「把 8 个种子换成别的」将没有任何断言能察觉）。
  ★ 种子一律取**非 2 的幂**，且**在文件里写明**（见 `EP_IMB_SEEDS`）。

★ **本主题的禁区**（细节见 `requirements.txt` 与域 README 的「本域不做什么」表）：
    **容量因子的定义与语义**（定义式、取整口径、dropping vs No-Token-Left-Behind、丢弃率口径）
      ⇒ 正文在 `80-model-architectures/moe/02-capacity-factor.md`（**全篇 563 行**）。
      ★★★ **让渡的方向**：是**本域把这几样让渡给该篇**（★ 该篇自己接住的是
        `00-meta/glossary.md` 的 `容量因子` 条 与 `40-kernels/moe-kernels/*` 的让渡，**不是本域**）
        ⇒ ★ **不得写成「该篇已逐字让渡给本域」** —— 实测该篇全篇里
        `EP` / `expert parallel` / `100-parallelism` / `all-to-all` / `并行` / `设备` / `通信`
        **各 0 次**（2026-10-08 实测）⇒ 它**根本没有**把任何东西让渡给本域。
      ★★ 本主题**一个字都不写**它们，只写**它对 EP 的影响**。
    **aux loss 的式子 / 路由 gating 的机制** ⇒ `80-model-architectures/moe/01-gating-and-routing.md`、
      `03-balance-and-granularity.md`。
    **all-to-all 的通信量分析** ⇒ `110-communication/collectives`+
      `20-gpu-architecture/interconnect/03-collectives-on-topology.md`；
      ★ `40-kernels/moe-kernels/01-moe-compute-pattern.md:243` **逐字让渡了它给 `110`**。
    **kernel 侧的 permute / unpermute / grouped GEMM** ⇒ `40-kernels/moe-kernels/02,03`；
      `00-meta/glossary.md` 的 `grouped GEMM` 条（`grouped GEMM` 词条）。
    **TP 每层 `2+2` 次 all-reduce** ⇒ `tensor-parallel/03` 与
      `10-foundations/performance-models/02-amdahl-and-scaling.md:417`（★ **不是 `:223`**）。
"""


# ===========================================================================
# 1. Switch Transformer —— [C] 论文（arXiv:2101.03961，同行评审：是）
#    ★★ 同行评审：**是** —— 依据在 **arXiv abs 页**上：
#      `https://arxiv.org/abs/2101.03961` 的 `Comments` 字段渲染文本**逐字是 `JMLR`**
#      （抓取日 2026-10-08）：该页上 `Comments` **1** 次、`JMLR` **1** 次，
#      而 `ICLR` / `NeurIPS` / `Accepted` / `Proceedings` / `Conference` / `MLSys` / `Journal`
#      **各 0** 次。★ 本模块据此把「同行评审：是」的**依据载体**停在 **abs 页**上。
#      ★★ 与同域 `sequence-context-parallel` 的处置**不同**：那边（`2205.05198`）的 abs 页
#        **给不出**评审依据（`Comments` **0** 次），依据停在 MLSys 2023 索引页；
#        本主题这一条**在 abs 页上就取到了** ⇒ 两处**必须分别写明载体**，不得互相照抄。
#    ★ 载体（引文）= `https://ar5iv.labs.arxiv.org/html/2101.03961` 的正文；
#      抓取日 2026-10-08；`curl -sSL --max-time 30` 取回后逐条 `count()`。
# ===========================================================================

SWITCH_ABS_JMLR = "JMLR"
# 来源：[C] Switch Transformer，**同行评审：是**的依据。载体 = **`https://arxiv.org/abs/2101.03961`
#       （abs 页本身）**，**原始字节 1 次**（2026-10-08 实测）；同页 `Comments` **1** 次。
# 口径：★★★ **本主题唯一的「同行评审：是」依据** —— abs 页把该条目的 `Comments` 字段渲染成 `JMLR`。
#       ★★ **它不是引文**（是书目字段）⇒ 只用来支撑「已确认同行评审」这件事本身。
# 消费者：`01_expert_sharding.py` 的 `D.10`（★ 与 `UNSOURCED == ()` 同一条）。
# 同值位置：无（`80-model-architectures/moe/02-capacity-factor.md` 的 §1.1 抬头也引了同一个字段，
#           ★ 但那是**那一篇**的转写；本模块这一份**独立存放**，靠本条互指）。

SWITCH_SECTION54_QUOTE = "Next we describe the partitioning strategy for expert and data parallelism."
# 来源：[C] Switch Transformer，§5.4 的**起句**。载体 = ar5iv 正文，**原始字节 1 次**。
#       ★★ **不取小节标题** `Expert and Data Parallelism`：那个串在同一页出现 **4 次**
#         （标题 / 目录 / 两处交叉引用）⇒ ★ **它不是唯一的**，用它当引文会掩盖
#         「我到底引的是哪一处」（本库的「同名不同指」教训）。
# 口径：★ §5.4 讲的是**专家并行**（EP）—— 原文把 EP 与 DP **合在一节里**讲。
#       ★★ 这一点**必须写明**：本域把 EP 单独成篇，是**本库的组织选择**，
#       **不是**原文的结构 ⇒ ★ 不得据此说「原文把 EP 当独立的一维」。
# 消费者：`01_expert_sharding.py` 的 `D.5`。
# 同值位置：无。

SWITCH_DISPATCH_ALLTOALL_QUOTE = "Because each core has its own expert, we do an all-to-all communication of size"
# 来源：[C] Switch Transformer，§5.4。载体 = ar5iv 正文，**原始字节 1 次**（2026-10-08 实测）。
# 口径：★★★ **本主题 01 的题眼之一**：**第一次 all-to-all** 的逐字依据 ——
#       它的位置在 **dispatch**（把 token 送去专家所在的核），
#       它的宾语是一个**尺寸** `[E, C, d_model]`（★ 那个方括号在 `<math>` 里 ⇒
#       **本模块只取到 `of size` 为止，绝不把那个尺寸写进常量**）。
#       ★★ 原文紧接着的一句给了**第二次**的方向（`SWITCH_COMBINE_QUOTE`）⇒ 两句必须成对引。
# 消费者：`01_expert_sharding.py` 的 `D.5`、`B.1` 的说明文案。
# 同值位置：无。

SWITCH_COMBINE_QUOTE = "in the forward pass to analogusly receive the tokens from each expert located on different cores."
# 来源：[C] Switch Transformer，§5.4。载体 = ar5iv 正文，**原始字节 1 次**。
#       ★★ 原句是 `There are additional communication costs of bfloat16 tensors of size
#         E × C × d_{model} in the forward pass to …` —— 尺寸那一截跨了 `<math>`
#         ⇒ **本模块从 `in the forward pass` 起取**（★ 这一段**不跨任何标记**）。
#       ★★★ 注意原文里那个词是 **`analogusly`**（拼写如此）—— **必须逐字照抄**，
#         改成 `analogously` 会在原始字节上计到 `0`。
# 口径：★★★ **第二次通信（combine）** 的逐字依据：把各专家算完的结果**收回来**。
#       ⇒ 与 `SWITCH_DISPATCH_ALLTOALL_QUOTE` 合起来 = **两次 all-to-all 的位置与语义**。
#       ★★ 原文这一句说的是 `additional communication costs`，**并没有**把这次也叫
#         `all-to-all` ⇒ ★ **本主题不声称「原文把两次都叫 all-to-all」**；
#         两次的**名字**依据在附录 F 的伪码注释里（见 `SWITCH_ALLTOALL_COUNT`）。
# 消费者：`01_expert_sharding.py` 的 `D.5`、`B.1` 的说明文案。
# 同值位置：无。

SWITCH_ALLTOALL_TERM = "All-to-All"
# 来源：[C] Switch Transformer，**附录 F 的伪码注释**。载体 = ar5iv 正文，**原始字节 2 次**。
#       ★★★ 该串**只**出现在那两个伪码注释的开头（逐字 `# All-to-All communication.`）——
#         ★★ 同一个句子里的 `communication.` 因为在原始字节里被 `<span>` 隔开而**计到 `0`**
#         ⇒ 本模块**只登记这个词本身**。
# 口径：★★★ **「两次 all-to-all」这个名字的逐字依据** —— 两处注释各一次。
#       ★★ **这是【计数】，不是【定义】**：本模块说的是「这个串在原始字节上出现 2 次」，
#       **不是**「原文说 EP 有两次 all-to-all」这个更强的断言（后者由 §5.4 的两句散文支撑）。
# 消费者：`01_expert_sharding.py` 的 `D.6`、`02_load_imbalance.py` 的 `D.5`。
# 同值位置：无。

SWITCH_ALLTOALL_COUNT = 2
# 来源：[C] Switch Transformer 附录 F —— 上一行那个词在**原始字节**上的实测次数。
# 口径：★★ 取它当**常量**而不是把 `2` 写死在断言里，是为了让 `mutate.py` 能看见它
#       （★ 同时 `01` 的 `D.6` 用 `== 2` 逐字钉住）。
#       ★★★ 它是**本主题最容易出错的地方**：同域 `sequence-context-parallel` 的前车之鉴是
#         「把散文里的一个词当成可计数的串」⇒ 在原始字节上计到 `0`。本模块的 `2` 是
#         **真的 `count()` 出来的**，且**只对着 `SWITCH_ALLTOALL_TERM` 那个串**。
# 消费者：`01_expert_sharding.py` 的 `D.6`、`02_load_imbalance.py` 的 `D.5`。
# 同值位置：无。

SWITCH_HYPHENATED_VERB_COUNT = 3
# 来源：[C] Switch Transformer。载体 = ar5iv 正文，**原始字节 `all-to-all`（全小写）3 次**。
# 口径：★★★ **与 `SWITCH_ALLTOALL_COUNT` 是【两个不同的串】** ——
#       小写 `all-to-all` 3 次（散文里用的**动词性**说法，含 §5.4 那句），
#       大写 `All-to-All` 2 次（**伪码注释**里的名字）。**同名不同指**的近亲：**同词不同形**。
#       ★★ 把两者合并成一个常量就抹掉了「散文用词 / 伪码注释名」这个区别本身。
# 消费者：`01_expert_sharding.py` 的 `D.6`（★ 与 `SWITCH_ALLTOALL_COUNT` **分别**钉住）。
# 同值位置：`SWITCH_ALLTOALL_COUNT`（同为「两次 all-to-all」这一件事的**另一个串**）。


SWITCH_DROPPED_RATE_QUOTE = "Throughout our experiments we didn’t notice any dependency on the number of experts for the number of tokens dropped"
# 来源：[C] Switch Transformer，**§2.2**（★ 不是 §5.4：它在讲负载均衡损失那一节）。
#       载体 = ar5iv 正文（`2101.03961`），**原始字节 1 次**（抓取日 2026-10-08 实测）。
#       ★★★ **本主题 02 原来那条「两篇正文里没有任何丢弃率数字」是【假否定】，这一条就是反例**：
#         原句逐字是 `Empirically we find ensuring lower rates of dropped tokens are important
#         for the scaling of sparse expert-models. Throughout our experiments we didn’t notice
#         any dependency on the number of experts for the number of tokens dropped (typically …)`
#         ⇒ ★ 本模块**只取到 `tokens dropped` 为止**（紧跟的 `(typically <1%)` 里那个数
#         在 `<math>` 里，**另立一条**见 `SWITCH_DROPPED_RATE_ALT`）。
# 口径：★★★ **它是「数不到 ≠ 不存在」这条教训的实物**（见模块抬头）：
#       当初用 `capacity` 当锚点数 `%` 得到 0 次 ⇒ 误判成「没有」；
#       ★ 这一句的锚点是 **`dropped tokens`** —— `capacity` 距该处最近的 `%` **694 字节**。
#       ★★ **不跨标记**：本段全是散文，`didn’t` 里的 `’` 是 **U+2019**（★ 不是 ASCII `'`，
#         换成 ASCII 会在原始字节上计到 `0`）。
# 消费者：`02_load_imbalance.py` 的 `D.5`。
# 同值位置：无。

SWITCH_DROPPED_RATE_ALT = "&lt;1\\%"
# 来源：[C] Switch Transformer，**§2.2** —— 上一条末尾那个丢弃率**经验值** `<1%` 的载体形态。
#       载体 = ar5iv 正文，**原始字节 2 次**（抓取日 2026-10-08 实测）。
#       ★★★ **它不是散文，是一个 `<math>` 元素的序列化形态**：原文那一处是
#         `(typically <math id="S2.SS2.p2.m1" class="ltx_Math" alttext="&lt;1\%" …>…</math>)`
#         ⇒ ★ 本模块按**【原始字节】登记**（HTML 转义 `&lt;` + LaTeX 的 `\%`）。
#       ★★★ **那「2 次」【不是两处出处】** —— 两次来自**同一个** `<math>` 元素的
#         `alttext` 与 `annotation`（`application/x-tex`）⇒ ★ **把口径写明**，
#         否则「2 次」会被读成「两个独立的出处」（本库的「同名不同指」教训的近亲）。
# 口径：★★★ **本模块【不】把这一条当【引文】**（本模块的引文一律不跨标记）——
#       它是**那个数的载体形态**，登记它是为了让「Switch 给过一个丢弃率经验值」这件事
#       **可被一条断言钉住**，而不是只听转述。★ 渲染后的形态是 `<1%`。
#       ★★ **它是一条【外部事实】**，**不是**「容量因子的定义与语义」
#         （后者仍全在 `80-model-architectures/moe/02-capacity-factor.md`）。
# 消费者：`02_load_imbalance.py` 的 `D.5`。
# 同值位置：无。


# ===========================================================================
# 2. GShard —— [C] 论文（arXiv:2006.16668）
#    ★★★ 同行评审：**本模块不声称已确认** —— arXiv abs 页
#      （`https://arxiv.org/abs/2006.16668`，抓取日 2026-10-08）上
#      `Comments` / `JMLR` / `ICLR` / `NeurIPS` / `Accepted` / `Proceedings` / `Conference` /
#      `MLSys` / `Journal` **各 0** 次 ⇒ **abs 页给不出任何评审依据**。
#      ⇒ ★★ **凡引用本论文，一律带「同行评审未确认」这一句**。
#    ★ 载体 = `https://ar5iv.labs.arxiv.org/html/2006.16668` 的正文；抓取日 2026-10-08。
# ===========================================================================

GSHARD_ALLTOALL_TERM = "AllToAll"
# 来源：[C] GShard。载体 = ar5iv 正文，**原始字节 15 次**（2026-10-08 实测）。
# 口径：★★★ **本主题 01 的一条【否定性】结论的依据**：GShard 用的是
#       **一个词 `AllToAll`**（没有连字符），与 Switch 的 `all-to-all` **写法不同**。
#       ⇒ ★★ **写两篇论文的对比时必须写出这个字形差异**，否则会把它当成同一个措辞。
# 消费者：`01_expert_sharding.py` 的 `D.9`、`02_load_imbalance.py` 的 `D.5`。
# 同值位置：无。

GSHARD_ALLTOALL_COUNT = 15
# 来源：[C] GShard —— 上一行那个词在**原始字节**上的实测次数（★ 含正文、图注、参考文献标题）。
# 口径：★★★ **它【不是】一次通信次数的计数** —— 15 次是**字符串出现次数**，
#       分布在散文、图注（`MoE dispatch and combine` 那条图）与参考文献里。
#       ★★★ 因此 **本主题绝不从它推出「GShard 做了几次 all-to-all」**：
#         原文**没有**给出次数（它只区分 `dispatch` 与 `combine` 两件事，用词是 `AllToAll`）。
#         ⇒ **不得给 GShard 编「2 次」**（那会是「编数字」）。
# 消费者：`01_expert_sharding.py` 的 `D.9`。
# 同值位置：无。

GSHARD_HYPHEN_COUNT = 0
# 来源：[C] GShard —— 串 `all-to-all`（带连字符）在**原始字节**上的实测次数：**0**。
# 口径：★★★ 与 `GSHARD_ALLTOALL_TERM`/`GSHARD_ALLTOALL_COUNT` **成对**才有意义：
#       「一个词 `AllToAll` 15 次、带连字符的 `all-to-all` **0 次**」
#       ⇒ 这是「两篇论文的措辞不同」的**可复核形式**（★ 光说「写法不同」不可复核）。
#       ★★ 同一次 `count()` 里 `all to all`（空格）与 `alltoall`（全小写无分隔）也各 **0** 次。
# 消费者：`01_expert_sharding.py` 的 `D.9`。
# 同值位置：无。

GSHARD_COLLECTIVE_QUOTE = "communication operators in the MoE model are"
# 来源：[C] GShard，§5.3（**同行评审未确认**）。载体 = ar5iv 正文，**原始字节 1 次**。
#       ★★ 原句是 `Two critical collective communication operators in the MoE model are
#         AllReduce and AllToAll.` —— 那两个词各自被 `<span>` 包住 ⇒
#         **本模块只取到 `are` 为止**（不把 `AllReduce`/`AllToAll` 写进这个常量）。
# 口径：★★★ **本主题 01 的另一条依据**：MoE 的两个关键集合通信是
#       **`AllReduce` 与 `AllToAll`** —— 即 **all-to-all 不是 MoE 唯一的通信**。
#       ⇒ ★★ 写「EP 的通信 = 两次 all-to-all」时**必须限定范围**（那是 Switch 的 EP 方案），
#         **不得**说成「MoE 只有 all-to-all」。
# 消费者：`01_expert_sharding.py` 的 `D.9`、`02_load_imbalance.py` 的 `D.5`。
# 同值位置：无。


# ===========================================================================
# 3. Megatron-LM MoE README —— [B] 某实现的官方文档（**不是规范**）
#    ★★★ 它是**活文档**（仓库存档在 `main` 分支，**无版本号**）⇒
#      ★ **引用时写明抓取日（2026-10-08）**，★★ **不做行号锚点**
#      （本库 `80-model-architectures` 的实测教训：行号锚点会随维护失效）。
#    ★ 载体 = `https://raw.githubusercontent.com/NVIDIA/Megatron-LM/main/
#      megatron/core/transformer/moe/README.md`（**36955 字节**），
#      `curl -sSL --max-time 30` 取回，逐条 `count()`。
#    ★★ 该 README 在本仓库里是**被 `docs/user-guide/features/moe.md` `{include}` 进去的**：
#      那个 `moe.md` 只有 **663 字节**、只是目录壳 ⇒ ★ **引用时必须指向被 include 的那一份**，
#      否则会指到一个没有正文的页面（★ 本模块实测：指向 `docs/.../moe.md` 会一无所获）。
# ===========================================================================

EP_MOE_README_PATH = "megatron/core/transformer/moe/README.md"
# 来源：[B] Megatron-LM 仓库，上列路径。★ 它**不是** `docs/user-guide/features/moe.md`
#       （那一份 663 字节、只 `{include}` 这一份）—— 抓取日 2026-10-08 实测路径。
# 口径：★★ 本模块**把路径也登记成常量**，因为「载体指错地方」正是本域的头号错误模式
#       （B 级当 A 级）的**前一步**：**先要指对文件**。
# 消费者：`03_ep_and_other_axes.py` 的 `D.5`/`D.8`/`D.9`。
# 同值位置：无。

EP_MOE_README_QUOTE = "Prior MoE frameworks constrain **EP ≤ DP** (Expert Parallelism must be a sub-group of Data Parallelism), which severely limits scalability."
# 来源：[B] Megatron-LM MoE README（活文档，抓取日 2026-10-08），**原始字节 1 次**。
#       ★★★ 它**不是** A 级：这是一个**实现**在描述**先前的框架**的做法
#         （`Prior MoE frameworks constrain …`）⇒ **本模块不把它当规范**。
# 口径：★★★ **本主题能给出的「`EP` 的约束」就只有这一条** ——
#       它是 **`EP` 与 `DP`** 的关系：**EP 组传统上是 DP 组的子组**。
#       ★★★ **原始字节里 `EP ≤ DP` 两侧有 Markdown 粗体标记 `**`**
#         ⇒ ★ **本模块按【原始字节】逐字存放**（含那两个 `**`）；
#         ★★ 任务书的转述把它写成了不带 `**` 的形式 ⇒ **本模块实测其原始形态与之不同**，
#           这里按实测形态登记（★ 去掉 `**` 的写法在原始字节上计到 **0**）。
#       ★★ 那个 `≤` 是 **U+2264**（不是 ASCII 的 `<=`）—— ★ 换成 ASCII 也计到 **0**。
# 消费者：`03_ep_and_other_axes.py` 的 `D.8`、`C.1` 的说明文案（`part_C()` 的打印）。
#         ★★ 原文写的是 `C.2` —— **`03` 只有 `C.1`，没有 `C.2`**（Chunk 5 修）。
# 同值位置：无。

EP_MOE_README_SUBQUOTE = "Expert Parallelism must be a sub-group of Data Parallelism"
# 来源：[B] Megatron-LM MoE README（活文档，抓取日 2026-10-08），**原始字节 1 次**。
# 口径：★★ 上一条里**不含任何标记**的那一段（★ 它**不跨 `**`**）⇒
#       即使有人把粗体标记的处置搞错了，这一条仍然在原始字节上计得到。
#       ★★ 它把「EP 组是 DP 组的子组」这件事**用散文**说了一遍。
# 消费者：`03_ep_and_other_axes.py` 的 `D.5`、`C.1` 的说明文案（`part_C()` 的打印）。
# 同值位置：`EP_MOE_README_QUOTE`（同一句话的另一种取法：一段 / 完整一句）。

EP_MOE_README_A2A_QUOTE = "NCCL-based All-to-All communication for token exchange"
# 来源：[B] Megatron-LM MoE README（活文档，抓取日 2026-10-08），**原始字节 1 次**。
# 口径：★★★ **实现文档侧对 EP 通信的原话**：**token exchange**（即 dispatch/combine）
#       用的是 **`All-to-All`**。★ 它给出的是**机制名**，**不给次数、也不给尺寸**
#       ⇒ ★★ **不得从它推出通信量**（那是 `110-communication/collectives`的正文）。
#       ★★ 注意这里用的是**带连字符的大写形**`All-to-All` —— 与 GShard 的
#         `AllToAll`（一个词）**不同形**（见 `GSHARD_ALLTOALL_TERM`）。
# 消费者：`03_ep_and_other_axes.py` 的 `D.5`。
# 同值位置：`SWITCH_ALLTOALL_TERM`（同一个名字的另一处出处）。

EP_NOT_FOUND_NOTE = "no source found for a constraint between the expert count and the expert-parallel degree"
# 来源：**[E] 本库的研究记录**（2026-10-08，抓取日同上）。
# 口径：★★★ **这是一条【否定性】记录，不是一条引文** —— 本库在**任何载体**上
#       （Switch §5.4 / 附录 F、GShard §5、Megatron MoE README 全文）
#       **都没有取到**「**专家数与 EP 度之间的大小关系**」这条规则。
#       ⇒ ★★ 本主题的正文**只写取到过的那一条**（`EP` 与 `DP` 的关系，见
#         `EP_MOE_README_QUOTE`），★★★ **不写任何「专家数必须不少于 EP 度」这类断言**。
#       ★ **把它登记成常量而不是只写在注释里**：注释会被忽略，
#       而这条记录**有一个真消费者**（`03` 的 `D.9`）⇒ 它**不是**一条无消费者的漏网候选。
# 消费者：`01_expert_sharding.py` 的 `D.10`、`03_ep_and_other_axes.py` 的 `D.9`
#         （★ 与 `UNSOURCED == ()` 同一条）、`03` 的 `B` 类说明文案（`part_B()` 的打印）。
# 同值位置：无。


# ===========================================================================
# 4. [E] 本库设计常量 —— 「本程序取哪一组输入」
# ===========================================================================

EP_DEV_GRID = (2, 4, 8)
# 来源：[E] 本库设计。**三档** EP 度（= 参与专家并行的设备数）。
# 口径：★★ 三档都是 **2 的幂** ⇒ 它们**整除** `EP_IMB_MODEL` 的 `E = 8`
#       （`8/2`、`8/4`、`8/8`）⇒ **每设备摊到的专家数恰好是 `E/EP`**，没有余数。
#       ★★ 第 1 档 `EP = 2` 是**退化对照**（每设备 4 个专家 ⇒ 负载波动被平均掉最多）；
#         `EP = 8` 时每设备 **1** 个专家 ⇒ 波动**完全不**被平均 —— 这就是 02 的 B 类要量的东西。
#       ★★ **本元组必须逐元素钉住**（`01`/`02` 的 `D.7`）—— 只钉「整体」会让元素级变异漏网。
# 消费者：`01_expert_sharding.py` 的 `B.2`/`D.7`、`02_load_imbalance.py` 的 `B.2`/`D.7`/`D.8`；
#         ★★ **另作 `01`/`02`/`03` 三程序 A/B/C 类循环的【输入】**（`03` 的 `A.1`/`A.2` 也依赖它
#         —— 它不在 `03` 任何一条 `check()` 的**实参**里，但改它会改变 `03` 报出的档位）。
# 同值位置：无。

EP_SHARD_CASES = ((8, 3), (8, 5))
# 来源：[E] 本库设计。**两组不整除**的 `(E, EP)`（★ 故意让 `E % EP != 0`）。
# 口径：★★ 它们钉的是 `01` 的 A 类：`E // EP` 之后的**余数** `E % EP` 个专家
#       **没有设备可放**（`8 % 3 = 2`、`8 % 5 = 3`）—— 与 TP 的「头切分」**同族**的问题。
#       ★ ★ 取 `3` 与 `5` 而不是 `4`/`2`，是因为**后者整除、量不出余数**。
#       ★★ **嵌套元组必须逐元素钉住**（`01` 的 `D.7`）：`_walk()` 会给它们
#          `[0][0]` / `[0][1]` 这样的后缀，★ 只钉外层会让**四个**内层叶子全部漏网。
# 消费者：`01_expert_sharding.py` 的 `D.7`；★ 另作 `A.1`/`A.1g`/`A.2g` 的【输入】（`for e_num, ep in …`）。
# 同值位置：无。

EP_DISPATCH_SHAPE = (32, 8, 4, 2)
# 来源：[E] 本库设计。`(T, D, E, top_k)` = `(token 数, 词元维, 专家数, 每 token 选几个专家)`
#       —— `01` 的 C 类（两条错法）与「纯搬运恒真性」用的形状。
# 口径：★★ `E = 4` 且 `EP ∈ {2, 4}` ⇒ 每设备 `2` 或 `1` 个专家；`top_k = 2` ⇒ 总投递 `64` 条。
#       ★★ 这个形状是**与域级探测 `probes/probe_100h.py` 的 `H3` 同族**的
#         （那一支用 `(64, 16, 4, 2)`，为了量「纯搬运」的恒真性）——
#         ★ 本模块**刻意不照抄它的数**：尺寸不同 ⇒ 差值的**具体数**不同
#         （★★ 这正是域 README 说的「**可引用的是『形状依赖』这个性质，某组具体形状给 `0` 的清单本身不可引用**」）。
#       ★★ **本元组必须逐元素钉住**（`01` 的 `D.7`）。
# 消费者：`01_expert_sharding.py` 的 `D.3b`/`D.7`；★ 另作 `make_inputs()` 的输入
#         （`A`/`C`/`D.4` 三类的数据都由它造）。
# 同值位置：无。

EP_A2A_DIM = (4, 8, 16)
# 来源：[E] 本库设计。`(E, C, d_model)` = 两次 all-to-all 的**张量尺寸**的三个维
#       （★ Switch §5.4 逐字给出的就是这三个符号：`E` / `C` / `d_model`）。
# 口径：★★★ **本域只给【尺寸】**：`01` 的 `B.1` 用 `E × C × d_model` 这个**乘积**做
#       **本库的记账模型**（每次 all-to-all 每设备发出的元素数 = 它手里那个张量的大小）。
#       ★★★ **all-to-all 的通信量分析照旧不在本域** —— 正文在
#         `110-communication/collectives`与 `20-gpu-architecture/interconnect/03`；
#         ★ `40-kernels/moe-kernels/01-moe-compute-pattern.md:243` **逐字让渡**了它。
#         ⇒ 本主题**只声明张量的尺寸**，**不声明任何「通信量 ∝ 什么」的结论**。
#       ★★ **本元组必须逐元素钉住**（`01` 的 `D.7`）。
# 消费者：`01_expert_sharding.py` 的 `D.3b`/`D.7`；★ 另作 `B.1`/`B.2` 的输入（`part_B()`）。
# 同值位置：无。

EP_IMB_SEEDS = (3, 7, 11, 13, 42, 100, 991, 2026)
# 来源：[E] 本库设计。`02` 的 B 类（统计量判据）用的**八个种子**。
# 口径：★★★ **八个种子都是【非 2 的幂】**（本域纪律）；
#       ★★ 它们**必须进 `specs.py` 并被逐元素钉住** —— `02` 的 B 类判据报的是
#          **这八个种子的均值与范围** ⇒ 换掉任何一个种子都会改变报出的数
#          （★ 与同域 `sequence-context-parallel` 的处置**不同**：那边的种子对任何断言都没有
#          可观察后果，所以**不**进 `specs.py`）。
#       ★★★ **口径必须与数一起报**：脱离这八个种子，`02` 的 B 类数**不可复核**
#          （`50-performance` 的「判据的小 n 稳定性」纪律）。
# 消费者：`02_load_imbalance.py` 的 `D.3b`/`D.7`/`D.8`；★ 另作 `A.1`/`A.1g`/`B.1`/`C.1` 的输入。
# 同值位置：无。

EP_IMB_MODEL = (256, 8, 2)
# 来源：[E] 本库设计。`(T, E, top_k)` = `(token 数, 专家数, 每 token 选几个专家)`
#       —— `02` 的负载账模型。
# 口径：★★★ **它是【第一代探测脚本 `probes/probe_100f.py` 的同一组输入】**
#       （那一支用 `T = 256` / `E = 8` / `top_k = 2`）—— ★ 取它的**理由是可复核**：
#       `E = 8` 与 `EP_DEV_GRID` 的三档**全部整除**，且 `EP = 8` 时每设备恰好 **1** 个专家。
#       ★★★ **【初始分布】必须与数一起报**：本主题用的是
#         「`scores ~ N(0,1)` 的 `(T, E)` 矩阵、取每行最大的 `top_k` 个」——
#         ★★ **换一种初始分布（例如均匀随机选 `top_k` 个）会给出不同的比值**
#         ⇒ 报「接收不均」的数时**必须写明初始分布**，否则两个数**不可比**。
#       ★★ **本元组必须逐元素钉住**（`02` 的 `D.7`）。
# 消费者：`02_load_imbalance.py` 的 `D.6`/`D.7`/`D.8`；★ 另作 `01` 的 `A.1g`/`A.2` 与
#         `02` 的 A/B/C 三类的输入。
# 同值位置：无。

EP_CAP_LIMIT = 64
# 来源：[E] 本库设计。`02` 的 C 类用的**每个专家的固定上限**（一个**直接给定的整数**）。
# 口径：★★★ **它是本库【直接给定】的，不是从任何公式算出来的** ——
#       容量怎么由「token 数 / 专家数 / 一个因子」算出、怎么取整、超了怎么办，
#       都是 `80-model-architectures/moe/02-capacity-factor.md` 的正文
#       （**全篇 563 行**），**本主题一个字都不写**；本主题只写**它对 EP 的影响**。
#       ★★ 取 `64` 的理由可手算复核：`EP_IMB_MODEL` 的 `(T, E, top_k) = (256, 8, 2)` ⇒
#         一共 `512` 条投递、分给 `8` 个专家 ⇒ **平均每个专家 `64` 条**；
#         上限**恰好压在平均负载上** ⇒ 「有的超、有的不超」**都量得到**
#         （★ **本库只声明这一点**：它是一个**让两侧都可观测**的取值；
#         ★★ **「上限该由 token 数与专家数怎么算出」不是本库的话** —— 那是回链那一篇的正文）。
# 消费者：`02_load_imbalance.py` 的 `D.3b`/`D.6`；★ 另作 `C.1` 的输入（`part_C()`）。
# 同值位置：无。

EP_TP_GRID = (1, 2, 4)
# 来源：[E] 本库设计。`03` 用的**三档 TP 度**（与 EP 组合）。
# 口径：★★ 第 1 档 `tp = 1` 是**退化对照**（只有 EP ⇒ 与单卡逐位相同的**搬运型**那一支）；
#       ★★ 另外两档是**改变了求值顺序**的那一支（专家的 `W1` 按**行**切 ⇒ 求和顺序变了）
#         ⇒ ★★★ 这正是域 README 说的「**A 类判据只在『切分改变了求值顺序』的地方才有信息量**」。
#       ★★ **本元组必须逐元素钉住**（`03` 的 `D.7`）。
# 消费者：`03_ep_and_other_axes.py` 的 `D.7`；★ 另作 `A.1`/`A.1g`/`A.2` 的输入。
# 同值位置：无。

EP_AXIS_SHAPE = (32, 8, 8, 2)
# 来源：[E] 本库设计。`03` 用的 `(T, D, E, top_k)` = `(token 数, 词元维, 专家数, 每 token 选几个专家)`。
# 口径：★★ 取 `D = 8` 与 `E = 8` 的理由：**`D` 要被 `EP_TP_GRID` 的最大档（`4`）整除**
#       ⇒ 行/列切都在**整除**处发生（★ 补齐的代价**不是** 03 的题目，那是
#       `sequence-context-parallel/03` 的正文）；`E = 8` 要被 EP 的三档整除。
#       ★★ **本元组必须逐元素钉住**（`03` 的 `D.7`）。
# 消费者：`03_ep_and_other_axes.py` 的 `D.6`/`D.7`；★ 另作 `make_layer()`/`part_B()` 的输入
#         （`A`/`B`/`C` 三类的数据都由它造）。★ 原文还写着 `C.2` —— **`03` 没有 `C.2`**。
# 同值位置：无。

EP_OVER_GRID = (16, 32)
# 来源：[E] 本库设计。`03` 的 B 类用的**两档 `EP > E` 的 EP 度**（`E = 8` ⇒ `16` / `32`）。
# 口径：★★★ **这是一条【本库设计的情景】，不是对任何规范的转述** ——
#       本库在**任何载体**上**都没有取到**「**专家数与 EP 度之间的大小关系**」这类约束
#       （见 `EP_NOT_FOUND_NOTE`）。⇒ ★★ 本节问的是「**如果** `EP` 超过专家数，
#       **本库的记账模型**会多出什么」，**不是**「规则要求 `EP` 不能超过专家数」。
#       ★★ 取 `16` 与 `32` 的理由：`ceil(EP/E)` = `2` 与 `4` ⇒ **因子翻倍看得见**
#         （★ `8`/`4`/`2` 那三档已经落在 `EP_DEV_GRID` 里，因子都是 `1`）。
#       ★★ **本元组必须逐元素钉住**（`03` 的 `D.7`）。
# 消费者：`03_ep_and_other_axes.py` 的 `B.2`/`D.6`/`D.7`；★ 另作 `B.1` 的输入。
# 同值位置：无。

EP_AXIS_PAIR = (2, 2)
# 来源：[E] 本库设计。`03` 的 C 类用的 `(EP, DP)` 组合 —— `world = EP * DP = 4` 台设备。
# 口径：★★ 取 `(2, 2)` 的理由：**两个轴都不为 `1`**（否则「组划分」无从谈起），
#       且 `EP * DP = 4` 小到**每一组都能逐个打印出来复核**。
#       ★★ **本元组必须逐元素钉住**（`03` 的 `D.7`）。
# 消费者：`03_ep_and_other_axes.py` 的 `D.3b`/`D.7`；★ 另作 `C.1` 的输入（`part_C()`）。
# 同值位置：无。

TOL_MACHINE = 1.0e-12
# 来源：[E] 本库设计。**机器精度档**的判据阈（相对/绝对通用）。
# 口径：★★ 本主题的判据有两类：**逐位对拍**（EP×TP ↔ 单卡）差落在 **1e-16 档**；
#       **内容级错法**（放置错位 / 不做 combine）差落在 **1e+0 档** —— **相隔十六个数量级**。
#       取 `1e-12` 是为了容纳「恰好为 `0`」与「几 ulp」两种正确形态。
#       ★★ **不得把它当作「物理量级的阈」**。
#       ★★★ 本主题**另有两条**与它并列的**内容级**阈，写在编号程序内部（**不进本模块**）：
#         · `1e-6`：`01` 的 C 类是**内容级**错法 ⇒ 用它区分「错法真的错了」与「舍入」；
#         · `1e-9`：`03` 的 A.1 是「几 ulp」档 ⇒ 用它区分「搬运/求和顺序」与「真的算错」。
# 消费者：`01`/`02`/`03` **各自**的 `D.3`（`preflight()`，先跑）与 `D.6`。
#         ★★★ 原文写的是「三个程序的 `A.1`/`D.6`」—— **三个 `A.1` 一个都不消费它**
#           （A 类是**精确等式**与**比值**，不用机器精度阈）⇒ Chunk 5 修正。
#         （★ `EP_OVER_GRID` 与 `EP_AXIS_PAIR` 的消费者见各自条目 —— 它们**在上面的 [E] 节**。）
# 同值位置：`data-parallel`/`tensor-parallel`/`pipeline-parallel`/`sequence-context-parallel`
#           的**编号程序**里各有一个同名常量（★ 它们**不在**那四个域的 `specs.py` 里）；
#           本域这一份**独立存放**，靠本条互指。


# ===========================================================================
# 5. UNSOURCED —— 待办清单，不是免责声明
# ===========================================================================
#
# ★ `CONTRIBUTING` §2：「不得**既保留数字又记 UNSOURCED**」—— 下面每一项都**没有**出现在
#   本模块的正文常量里。★ 本元组的字符串**没有消费者** ⇒ 它们会是 `make mutate` 的
#   **漏网候选**（本库的既有形态：`90-training/*/code/specs.py` 同节，由编号程序的 `part_D`
#   各钉一次，**不给它开后门**）。

# ★★ 本元组**留空**（与同域 `pipeline-parallel` / `sequence-context-parallel` 的形态一致）
#   —— 理由有两条，**都是实测教训**：
#   · **每一项的字符串都会成为 `make mutate` 的叶子**，而它们**没有消费者**
#     ⇒ 会全部记成**漏网**（`80` 实测「只用于打印的模块级常量表」制造过 2 条）。
#     ★★ **而【空元组本身】有消费者**：`01` 的 `D.10` 与 `03` 的 `D.9` 各用
#       `specs.UNSOURCED == ()` 逐位钉住它（★ 0 个叶子 ⇒ 不产生变异体）。
#   · 本主题的两处「未核到」已经**写在正文常量自己的 `# 来源：` 里**，不欠一条记录：
#     ① 「**专家数与 EP 度之间的约束**」在任何载体上**未取到** ⇒ 已登记为
#        `EP_NOT_FOUND_NOTE` 常量（★ 它**有**消费者：`03` 的 `D.9`）；
#     ② **丢弃率的数字** —— ★★★ **本条在 Chunk 5 修正；原写法是一条【假否定】**。
#        原写法：「任何来源给出的丢弃率数字在 Switch / GShard 的正文与 abs 页上**未取到**
#        （★ 两篇论文里 `capacity` 附近 `%` 各 **0** 次）」。★★★ **半对半错**：
#        · ★ **GShard 那一半【对】** —— 复核（抓取日 2026-10-08，ar5iv 正文）：
#          剥离标签后 `%` **12 处**，逐条看**全在** FLOPS / roofline / 重算占比 / 执行时间占比；
#          `drop` **5 处**全在 `dropout`、「**收益下降**」（`this drop in gains`）等处
#          ⇒ ★★ **GShard 确实不给任何丢弃率数字**。
#        · ★★★ **Switch 那一半【错】** —— §2.2 逐字给了 `typically <1%`
#          （见 `SWITCH_DROPPED_RATE_QUOTE` / `SWITCH_DROPPED_RATE_ALT`）。
#        ★ 错因 = **锚点选错**（用 `capacity` 而不是 `dropped tokens`）⇒ 见模块抬头那条教训。
#        ⇒ ★★★ **本元组仍留空**（丢弃率**不再**是一条「未取到」），并**撤掉**由这条假否定
#          立起来的禁用规则（原文是「引任何来源的丢弃率数字 ⇒ **不行**」—— **该规则已作废**）。
UNSOURCED = ()

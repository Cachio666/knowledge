"""100-parallelism/sequence-context-parallel 的外部事实常量源。

本模块是 `CONTRIBUTING` §2 约定的**共享事实源**（`specs.py`）：本主题的编号程序从它读常量，
推导过程可复核，**常量本身不可自证**。

★ **只放本主题真的会断言的常量**：没有消费者的常量就是 `make mutate` 的漏网候选
  （`80` 实测「只用于打印的模块级常量表」制造过 2 条漏网）⇒ **要么给它消费者、要么不写**。
  ★ 本模块里**每一条**都被至少一条 `D` 类断言用 `==` 逐字钉住（见三个程序的 `D.1`–`D.8`）。

★ **来源分级 A–E**（判据见 `CONTRIBUTING` §2：A 厂商规范 / B 产品官方文档 / C 论文 /
  D 社区 / E 本库推导）。**每条常量带**：
    `# 来源：` + `[A-E]` 级 + 抓取日 + `# 同值位置：`
  ★★ **来源必须分级并写明是哪一级** —— 本域的头号错误模式是
    **把「某实现的文档」当成「规范的文档」**（B 级当 A 级）。
    本主题里**三条来源分属三级**，写清它们是本文档最容易出错的地方之一：
      · `2205.05198` = **[C] 论文**（★ 同行评审：**是**，依据在 **MLSys 2023 会议论文索引**上，
        **不在 arXiv abs 页** —— 见 `SP_MLSYS_INDEX_TITLE` 那一条）；
      · `2310.01889` = **[C] 论文**（★ 同行评审：**本模块不声称已确认** ——
        arXiv abs 页上 `ICLR` / `MLSys` / `Accepted` / `Proceedings` / `Conference` / `NeurIPS`
        各 **0** 次；Comments 字段的渲染文本逐字是 `Code:` 加一个链接 ⇒ **无任何评审字样**）；
      · `context_parallel.md` = **[B] 某实现的官方文档**（★ **不是规范**）——
        它是一条**活文档**（无版本号、随 `main` 分支变动）⇒ 引用时**写明抓取日**、
        **不做行号锚点**（本库 `80-model-architectures` 的实测教训：行号锚点会随维护失效）。

★★ **本模块的【载体】选择**（★ 逐条都是本任务书要求的「自己 curl 核实」，不是转述）：
    · SP 与 Ring Attention 的引文载体都是 **`ar5iv.labs.arxiv.org` 的正文页**
      （URL 见两节抬头）；★ 抓取日 **2026-10-08**，`curl -sSL --max-time 30` 取回 **原始字节**后
      **逐条 `count()`** —— 下面每条的 `# 来源：` 里都写了实测次数。
    · Megatron CP 文档的载体是 **GitHub `raw.githubusercontent.com` 上的
      `docs/user-guide/features/context_parallel.md`**（抓取日同上）。
    · ★★★ **引文片段一律【不跨标记】**：ar5iv 把变量与公式包在 `<math …>…</math>` 里
      （例如 `sbh` / `10sbh` / `t`），公式编号包在 `<a …>` 里
      ⇒ **跨它的片段在原始字节上计不到**。本模块的处置是**把片段缩到标记之前**：
      凡引言 **一律以 `… to:` / `… from Equation` / `… with the` 这类词收尾**，
      **绝不把宾语公式或编号写进常量**（三个 chunk 的前车之鉴：写「原文是 `M ≥ 4 × K`」而实测两种口径都不是）。

★★ **三类真值来源必须分开标注**：
    ① **外部规格值**：论文原文（**C 级，须注明是否确认同行评审**）、厂商/产品文档（A/B 级）
       ⇒ **本模块存放的就是这一类**。
    ② **本机实测值**：本域程序在本机跑出的数（**须写 python3 与 numpy 版本**）⇒ **不放在本模块**。
    ③ **推导值**：程序算出的量（可手算复核）⇒ **不放在本模块**
       （`CONTRIBUTING` §2：「只放从来源抄来的数，**不放推导值**」）。
    ★★ 本主题的 `SP_GAIN_CASE_*` / `CP_PAD_*` / `ATTN_*` 是 **`[E]` 本库设计常量**
      （「本程序的输入取哪一组值」），**不是任何来源里的数** —— 它们的 `# 来源：` 一律写
      `[E] 本库设计`，并写明**为什么取这个值**（可手算复核）。

★★ **本域的两条自缚纪律**：
    · **所有数值常量取到 `< 2^53`** —— `mutate.py` 的浮点变异是 `v + 1.0`，
      大浮点数（`2^53` 以上）会**吞掉**这个增量（变异了等于没变异）。
    · **幂次一律写成十进制字面量**（`2^-25` 写作 `2.9802322387695312e-08`）——
      `mutate.py` 的 `_walk()` **不下钻 `BinOp`** ⇒ `2.0 ** -6` 整体**不进变异分母**。
      （★ 本模块**一处幂次都没有**，两个 `float` 常量 `TOL_MACHINE` / `RING_AMPLIFY`
      都是十进制字面量。）

★★ **【种子】不放这里**：数据生成的种子写在**程序内的函数体里**（不是模块级常量）——
  改它对任何断言都没有可观察后果 ⇒ 放进分母只会制造一条**不过期的漏网**。
  （★ 本主题的种子取 **`991`**，**不是 2 的幂**；它写在 `02`/`03` 的 `_make_inputs()` 里。）

★ **本主题的禁区**（细节见 `requirements.txt` 与域 README 的「本域不做什么」表）：
    **「在线 softmax」的 `(m, l)` 更新公式本身**（`40-kernels/softmax-normalization/02-online-softmax.md`）与
      **注意力变体的结构**（MHA / MQA / GQA 的定义与配比，`80-model-architectures/attention-variants/01-mha-mqa-gqa.md`）
      ⇒ 本主题写的是**跨设备**的块统计量重缩放与**头怎么分给设备**。
    ★★★ **术语纪律**：本主题**不使用「online softmax」这个词** —— 本域写的是
      **跨设备**的**块统计量重缩放**，术语一律用 Ring Attention 原文的措辞
      （**置换不变性** + **统计量重缩放**，见 `RING_PERMUTATION_QUOTE` / `RING_RESCALE_QUOTE`）。
"""


# ===========================================================================
# 1. Sequence Parallelism —— [C] 论文（arXiv:2205.05198，MLSys 2023）
#    ★★ 同行评审：**是** —— 依据是 **MLSys 2023 会议论文索引**
#      （`proceedings.mlsys.org/paper_files/paper/2023`，抓取日 2026-10-08，见下 `SP_MLSYS_INDEX_TITLE`）。
#      ★★★ **不是** arXiv abs 页 —— 该页上 `Comments` 字段**不存在**（实测 `Comments` 0 次）、
#        `MLSys` / `ICLR` / `Accepted` / `Proceedings` / `Conference` 各 **0** 次
#        ⇒ **abs 页本身给不出任何同行评审依据**（★ 本模块据此把同行评审的依据**停放在索引页**上）。
#    ★ 载体 = `https://ar5iv.labs.arxiv.org/html/2205.05198` 的正文；
#      抓取日 2026-10-08；`curl -sSL --max-time 30` 取回后逐条 `count()`。
# ===========================================================================

SP_MLSYS_INDEX_TITLE = "Reducing Activation Recomputation in Large Transformer Models"
# 来源：[C] 论文的**标题**，载体 = **MLSys 2023 会议论文索引页**
#       （`https://proceedings.mlsys.org/paper_files/paper/2023`），**原始字节 1 次**
#       （★ 同页上该条目的作者串与 `…-Abstract-mlsys2023.html` 也各 1 次，本模块只登记标题这一条）。
# 口径：★★★ **本主题唯一的「同行评审：是」依据** —— 索引页把该条目标在 `mlsys2023` 轨道下。
#       ★★ **它不是引文**（是书目条目）⇒ 只用来支撑「已确认同行评审」这件事本身。
# 消费者：`01_activation_sharding.py` 的 `D.10`（★ 与 `UNSOURCED == ()` 同一条）。
# 同值位置：`100-parallelism/README.md` 无（README 不登记文献）。

SP_EQ1_PROSE_QUOTE = "The above equation is for the case that no form of model parallelism is applied."
# 来源：[C] SP（arXiv:2205.05198），§4.1，**原始字节 1 次**（2026-10-08 实测）。
# 口径：★ 它紧跟的是**式(1)** `sbh(34 + 5as/h)`（式子在 `<math>` 里 ⇒ 本模块只取这句散文）。
#       ★ 式(1) 是**无任何模型并行**时的每层激活显存 ⇒ 它是 `A.1` 的退化对照那一支。
# 消费者：`01_activation_sharding.py` 的 `D.1`。
# 同值位置：无。

SP_EQ2_PROSE_QUOTE = "the per-layer memory required to store the activations reduces from Equation"
# 来源：[C] SP（同上），§4.2.1，**原始字节 1 次**。
#       ★★ 原句是 `Assuming t-way tensor parallelism, the per-layer memory … from Equation 2 to:` ——
#         `t` 与式号 `2` 都在标记里 ⇒ **只取标记之间的这段散文**（以 `Equation` 收尾，
#         **不带那个 `2`**）。
# 口径：★★ 它引出的是**式(2)**：`sbh(10 + 24/t + 5as/(ht))` —— **张量并行**下的每层激活显存。
#       ★★ **式(2) 的 `10sbh` 那一项【不除以 t】**（那是被复制的 LN / dropout 区，
#        见 `SP_REPLICATED_QUOTE`）。
# 消费者：`01_activation_sharding.py` 的 `D.1`。
# 同值位置：无。

SP_EQ4_PROSE_QUOTE = "Using sequence parallelism along with tensor parallelism, the memory required to store the activations per transformer layer reduces from Equation"
# 来源：[C] SP（同上），§4.2.2，**原始字节 1 次**。
#       ★★ 原句以 `… reduces from Equation 2 to:` 收尾，那个 `2` 是 `<a>` 链接 ⇒ **只取到 `Equation`**。
# 口径：★★ 它引出的是**式(4)**：`sbh(10/t + 24/t + 5as/(ht)) = (sbh/t)(34 + 5as/h)`。
#       ★★★ **式(4) = 式(1)/t** —— 这是 SP 的定义性结论（SP 把**整层**激活按 s 维切 t 份）。
# 消费者：`01_activation_sharding.py` 的 `D.1`。
# 同值位置：无。

SP_EQ4_TAIL_QUOTE = "The above equation is now Equation"
# 来源：[C] SP（同上），§4.2.2，**原始字节 1 次**（2026-10-08 实测，**不跨标记**）。
# 口径：★ 原句是 `The above equation is now Equation 1`（那个 `1` 在 `<a>` 里）——
#       原文**自己明说**式(4) 退化成式(1) 当 `t = 1`。
#       ⇒ ★★ **`A.1` 的引文依据就是这一句**（不是本库的推断）。
# 消费者：`01_activation_sharding.py` 的 `D.1`。
# 同值位置：无。

SP_REPLICATED_QUOTE = "is due to these replicated operations and as a result they are not divided by the tensor parallel size"
# 来源：[C] SP（同上），§4.2.2，**原始字节 1 次**。
#       ★★ 原句是 `the 10sbh part of Equation 2 is due to … size t)` —— `10sbh` 与末端的 `t`
#         都在标记里 ⇒ **只取标记之后的这段散文**（★ 这一条是本主题最容易抄错的地方：
#         把 `10sbh` 一起写进常量，就会像前三个 chunk 那样**在原始字节上计到 `0`**）。
# 口径：★★★ **它把「`10sbh` 是被复制的」这件事钉死**（复制的 = LN 与 attention / MLP 后的 dropout）
#       ⇒ `A.2` 的差式 `10sbh(1 − 1/t)` 的**物理含义**就是「SP 把那 `10sbh` 也切了」。
# 消费者：`01_activation_sharding.py` 的 `D.1`、`A.2` 的说明文案。
# 同值位置：无。

SP_NO_OVERHEAD_QUOTE = "our technique, which mixes both tensor and sequence parallelism, reduces the activation memory significantly without any additional compute, communication, or memory overhead."
# 来源：[C] SP（同上），§2（引言里的贡献陈述），**原始字节 1 次**（2026-10-08 实测，**不跨标记**）。
# 口径：★★★ **本主题 01 的题眼**：**SP（与 TP 合用）不引入额外的计算 / 通信 / 显存开销**。
#       ★★ 它的**前提是「混用」**（`mixes both tensor and sequence parallelism`）——
#         **裸着上 SP 是有开销的**（见 `SP_NAIVE_EXTRA_COMM_QUOTE`）⇒ 两句必须成对引。
#       ★★★ **本模块把它登记为【原文的结论】，一律不当 C 类反例用** ——
#         把「SP 不引入通信开销」写成一条会失败的反例，就是**与原文相反**。
# 消费者：`01_activation_sharding.py` 的 `D.3`（★ 明确标注「这是原文结论，不是 C 类」）。
# 同值位置：无。

SP_NAIVE_EXTRA_COMM_QUOTE = "These extra communications introduce overhead and will slow down the training."
# 来源：[C] SP（同上），§4.2.2，**原始字节 1 次**。
# 口径：★★ **「裸 SP 有开销」的那一半** —— 若不做下文那个合并，
#       序列并行与张量并行之间的转换集合通信**会拖慢训练**。
# 消费者：`01_activation_sharding.py` 的 `D.3`。
# 同值位置：无。

SP_FUSE_CONVERTER_QUOTE = "To avoid these extra communications, we combine these operations with the"
# 来源：[C] SP（同上），§4.2.2，**原始字节 1 次**（★ 切在变量 `f` 前，那个 `f` 是数学斜体 ⇒ 跨标记）。
# 口径：★★ **「开销怎么消掉的」另一半** —— 把转换操作**合并进** `f` / `f̄`（TP 的那两个集合通信）
#       得到 `g` / `ḡ`（前向 all-gather + reduce-scatter，后向反向）
#       ⇒ ★★ **它们替代了 TP 原有的通信，而不是在其之上再加一层** ⇒ 与 `SP_NO_OVERHEAD_QUOTE` 自洽。
# 消费者：`01_activation_sharding.py` 的 `D.3`。
# 同值位置：无。

SP_SEQ_DIM_QUOTE = "This characteristic allows us to partition these regions along the sequence dimension"
# 来源：[C] SP（同上），§4.2.2，**原始字节 1 次**。
# 口径：★★★ **「SP 切的是【序列】维」的逐字依据** —— 而 `C.2` 的错法正是**切错轴**（按 `h` 切）。
# 消费者：`01_activation_sharding.py` 的 `D.3`、`C.2` 的说明文案。
# 同值位置：`CP_DEFINITION_QUOTE`（CP 也切序列维，但范围不同 —— 见 `CP_VS_SP_QUOTE`）。

SP_LN_DROP_QUOTE = "These elements do not require a lot of compute but demand a considerable amount of activation memory."
# 来源：[C] SP（同上），§4.2.2，**原始字节 1 次**。
# 口径：★★ 它说的是 **LN 与 dropout**（TP 没切到的那些算子）—— **计算少、激活多**
#       ⇒ ★★ **这就是 SP 存在的理由**：切它们的**显存**，几乎不动它们的**算力账**。
# 消费者：`01_activation_sharding.py` 的 `D.3`。
# 同值位置：无。

SP_SPLIT_REPLICATED = 10
# 来源：[C] SP（同上），式(2) 的 `10sbh` 项（原文在 §4.2.2 直接用 `10sbh` 称呼它；
#       ★ 本模块的载体是 **`<math>` 的 `alttext`**，逐字 `10sbh`，不是散文 ——
#       散文里那句只到 `… not divided by the tensor parallel size` 为止，见 `SP_REPLICATED_QUOTE`）。
# 口径：★★ **式(2) 里【不除以 t】的那部分系数**（单位是 `sbh`）⇒ 它就是 `A.2` 差式里的那个 `10`。
#       ★★ 取它当**常量**而不是把 `10` 写死在 `A.2` 里，是为了让 `mutate.py` 能看见它
#       （★ 同时 `D.2` 用 `== 10` 逐字钉住）。
# 消费者：`01_activation_sharding.py` 的 `A.2`（差式的右端）、`D.2`。
# 同值位置：`SP_EQ2_TP_NUM`（同一个 `10` 在式(2) 的 `10` 与式(4) 的 `10/t` 里各出现一次 ——
#           ★ **它们不是同一个量**：本主题只登记「不除以 t 的那一个」为 `SP_SPLIT_REPLICATED`）。

SP_EQ2_TP_NUM = 10
# 来源：[C] SP（同上），式(4) 的 `10/t` 项（★ 载体同 `SP_SPLIT_REPLICATED`：`<math>` 的 `alttext`）。
# 口径：★★★ **与 `SP_SPLIT_REPLICATED` 数值相同、来源位置不同** —— 它是**被切开之后**的那一项。
#       ★★ 这正是「**同名不同指**」的近亲：**同值不同指**。把两者合并成一个常量
#       ⇒ 就抹掉了「一个是复制区、一个是切开后的复制区」这个区别本身。
# 消费者：`01_activation_sharding.py` 的 `D.2`（★ 与 `SP_SPLIT_REPLICATED` **分别**钉住）。
# 同值位置：`SP_SPLIT_REPLICATED`（同为 `10`，**不同项**）。

SP_EQ2_TP_SHARD = 24
# 来源：[C] SP（同上），式(2) / 式(4) 共同的那个 `24/t` 项（★ 载体：`<math>` 的 `alttext`）。
# 口径：★★ 它是**始终被 TP 切开**的那部分系数（单位 `sbh`）⇒ 与 `10` 相对：
#       `34 = 10 + 24`（见 `SP_EQ1_SBH`）。
# 消费者：`01_activation_sharding.py` 的 `D.2`。
# 同值位置：无。

SP_EQ1_SBH = 34
# 来源：[C] SP（同上），式(1) 的 `34`（★ 载体：式(1) 所在 `<math>` 的 `alttext`，逐字
#       `\text{Activations memory per layer = }sbh\left(34+5\frac{as}{h}\right).`）。
# 口径：★★ **`34 = 11 + 19 + 4`** —— 原文 §4.1 的三块之和（attention block `11sbh+5as²b`
#       + MLP `19sbh` + layer norm `4sbh`）⇒ ★★ **`D.2` 顺带钉住这个加法**。
#       ★ 式(1) 是「无任何模型并行」那一支，`A.1` 的退化对照要用它。
# 消费者：`01_activation_sharding.py` 的 `A.1`（`eq1` 的系数）、`D.2`。
# 同值位置：无。

SP_EQ1_ATTN_COEF = 5
# 来源：[C] SP（同上），三式共同的 `5as/h` 里的 `5`（★ 载体：`<math>` 的 `alttext`）。
# 口径：★★★ **本主题 01 的 B 类判据全靠它**：`5as²b/h` 是 `O(s²)`，
#       `10sbh` 是 `O(s)` ⇒ **序列越长，SP 的相对收益越小**（`B.1`/`B.2` 实测）。
# 消费者：`01_activation_sharding.py` 的 `A.1`/`A.2`/`A.3`（三式的注意力项）、`D.2`。
# 同值位置：无。

SP_TP_GRID = (1, 2, 4, 8, 16, 32)
# 来源：[E] 本库设计。**六档**张量并行度（★ 全为 2 的幂 ⇒ `1/t` 在二进制里**精确**，
#       ★★ 这样 `A.2` 的差式才能期望**恰好 `0.0`** 而不是「机器精度档」）。
# 口径：★★ 第 1 档 `t = 1` 是**退化对照**（`A.1` 要用）；其余五档是 `A.2` 的扫描点。
#       ★★ **本元组必须逐元素钉住**（`D.7`）—— 只钉「整体」会让元素级变异漏网。
# 消费者：`01_activation_sharding.py` 的 `preflight()`、`A.1`、`A.2`、`A.3`、`D.7`。
# 同值位置：无。

SP_GAIN_CASE_LONG = (1024, 1, 64, 8)
# 来源：[E] 本库设计。`(s, b, h, a) = (序列长度, batch, 隐维, 注意力头数)` 的**第一组**
#       （★ 由任务书给定）⇒ `a/h = 0.1250`。
# 口径：★★ 取它的目的是与下面**更长的** `SP_GAIN_CASE_SHORT` 对照：
#       `10sbh` 是 `O(s)`、`5as²b/h` 是 `O(s²)` ⇒ **长序列的相对收益更小**（`B.1`/`B.2`）。
# 消费者：`01_activation_sharding.py` 的 `B.1`、`B.2`、`D.7`。
# 同值位置：无。

SP_GAIN_CASE_SHORT = (4096, 1, 256, 16)
# 来源：[E] 本库设计。同上，**第二组** `(s, b, h, a)`（★ 由任务书给定）⇒ `a/h = 0.0625`。
# 口径：★★ **它比第一组【长】但 `a/h` 只有一半** —— 两者一起看才说明
#       「相对收益随 `s` 下降」**不是**因为 `a/h` 变小（`B.2` 会把两个方向都说清）。
# 消费者：`01_activation_sharding.py` 的 `B.1`、`B.2`、`D.7`。
# 同值位置：无。


# ===========================================================================
# 2. Ring Attention —— [C] 论文（arXiv:2310.01889）
#    ★★★ 同行评审：**本模块不声称已确认** —— arXiv abs 页上
#      `ICLR` / `MLSys` / `Accepted` / `Proceedings` / `Conference` / `NeurIPS` 各 **0** 次
#      （抓取日 2026-10-08）；该页 Comments 字段的**渲染文本**逐字是
#      `Code:` 加一个指向 `github.com/lhao499/llm_large_context` 的链接
#      （★ 原始字节上 `Code: ` 之后紧邻 `<a …>` ⇒ 该短语**跨标记**，
#        所以本模块**不把它登记成引文常量**，只以「上述几个词各 0 次」作为依据）。
#      ⇒ ★★ **凡引用本论文的结论，一律带「同行评审未确认」这一句**。
#    ★ 载体 = `https://ar5iv.labs.arxiv.org/html/2310.01889` 的正文；
#      抓取日 2026-10-08；逐条 `count()`。
# ===========================================================================

RING_PERMUTATION_QUOTE = "we leverage the permutation invariance property of the inner loop’s key-value block operations."
# 来源：[C] Ring Attention（arXiv:2310.01889），§3（**同行评审未确认**），**原始字节 1 次**。
#       ★ **句中的撇号是 U+2019（`’`），不是 ASCII `'`** —— 抄错一个字符就计到 `0`。
# 口径：★★★ **本主题 02 的术语来源**：Ring Attention 说的是「**置换不变性**」，
#       ★★ **不是「online softmax」**（那个词的正文在 `40-kernels/softmax-normalization/02`，
#       本主题**一个字都不写它**）。
# 消费者：`02_ring_attention.py` 的 `D.2`（★ 明确标注「本主题不用 online softmax 这个词」）。
# 同值位置：无。

RING_RESCALE_QUOTE = "the self-attention between a query block and a group of key-value blocks can be computed in any order, as long as the statistics of each block are combined correctly for rescaling."
# 来源：[C] Ring Attention（同上），§3（**同行评审未确认**），**原始字节 1 次**。
# 口径：★★★ **「跨设备的块统计量重缩放」这句话的逐字出处** ——
#       块**顺序可以任意**，前提是**每块的统计量被正确地【结合起来做重缩放】**。
#       ⇒ ★★ `02` 的 `A.1`（任意 CP 度下与全量注意力逐位相同）就是这句话的实现性检验；
#         `C.1`（不做重缩放 ⇒ 放大尺度下 `nan`）是它的**反面**。
# 消费者：`02_ring_attention.py` 的 `D.2`、`A.1` 与 `C.1` 的说明文案。
# 同值位置：无。

RING_ORDER_INVARIANT_QUOTE = "the results are invariant to the ordering of these blockwise computations."
# 来源：[C] Ring Attention（同上），§1（**同行评审未确认**），**原始字节 1 次**。
# 口径：★ 同一件事的**另一种说法**（§1 的观察 vs §3 的机制）⇒ 两句互证，但**都不当 A 类**。
# 消费者：`02_ring_attention.py` 的 `D.2`。
# 同值位置：`RING_PERMUTATION_QUOTE`（同一性质的两处措辞）。

RING_SEND_KV_QUOTE = "each device sends a copy of its key-value blocks being used for blockwise computation to the next device in the ring, while simultaneously receiving key-value blocks from the previous one."
# 来源：[C] Ring Attention（同上），§1（**同行评审未确认**），**原始字节 1 次**。
# 口径：★★★ **`02` 的 B 类通信账的形态来源**：每台设备**只发自己那一块 KV**（一次），
#       收上一台的 ⇒ 在环上转一圈，**每台设备的通信量 ∝ S**（`B.1` 用字节账钉住）。
#       ★★ **本模块不复核它的测量口径**（那是原文在真集群上的事）。
# 消费者：`02_ring_attention.py` 的 `D.3`、`B.1` 的说明文案。
# 同值位置：无。

RING_OVERLAP_QUOTE = "As long as block computations take longer than block transfers, overlapping these processes results in no added overhead compared to standard transformers."
# 来源：[C] Ring Attention（同上），§1（**同行评审未确认**），**原始字节 1 次**。
# 口径：★★ **「零开销」是【有条件的】** —— 条件是**块计算比块传输慢**（可重叠）。
#       ⇒ ★ **不得转述成「Ring Attention 一定零开销」**。
#       ★★ 本机**没有多卡、没有真实网络** ⇒ 本主题**不声称**任何关于重叠幅度的结论。
# 消费者：`02_ring_attention.py` 的 `D.3`。
# 同值位置：无。

RING_QUADRATIC_QUOTE = "the self-attention has memory cost quadratic in the input sequence length"
# 来源：[C] Ring Attention（同上），§1（**同行评审未确认**），**原始字节 1 次**。
# 口径：★★ 全量注意力的**显存**是 `O(S²)` ⇒ `02` 的 `B.2` 把它与分块版的 `O(S·d)` 对比
#       （★ **引用，不重推** —— 本模块只登记这一句，`B.2` 只做**数组规模**的对照）。
# 消费者：`02_ring_attention.py` 的 `D.3`、`B.2` 的说明文案。
# 同值位置：`CP_MEM_PER_GPU_QUOTE`（CP 文档说的是**每卡**的激活显存下降 —— 不同量）。

RING_ACCUMULATION_QUOTE = "the accumulation of key-value blocks leads to increased memory usage, which defeats the purpose of reducing memory cost."
# 来源：[C] Ring Attention（同上），§3（**同行评审未确认**），**原始字节 1 次**。
# 口径：★★ **为什么必须「边收边丢」** —— 把收到的 KV 块**攒起来**就白省显存了
#       ⇒ ★★ 这是 `02` 分块实现里「每步只用当前块、用完不存」的逐字依据。
# 消费者：`02_ring_attention.py` 的 `D.3`。
# 同值位置：无。

ATTN_S = 512
# 来源：[E] 本库设计。本主题 02 / 03 用的**序列长度**。
# 口径：★★ 取 `512` 的原因：它要被 `RING_CP_GRID` 的最大档（`16`）**整除**
#       （`512 / 16 = 32`）⇒ `A.1` 里**不需要补齐**，误差只来自块统计量重缩放本身。
#       ★★ **`S % cp != 0` 的补齐代价单独在 `03` 的 `B.1` 里量**（`CP_PAD_S` / `CP_PAD_DEGREE`）。
# 消费者：`02_ring_attention.py` 的 `A.1`/`A.2`/`B.1`/`B.2`/`C.1`/`D.7`、
#         `03_constraints.py` 的 `A.1`/`A.2`/`C.1`/`C.2`/`C.3`/`D.7`。
# 同值位置：无。

ATTN_D = 64
# 来源：[E] 本库设计。本主题 02 / 03 用的**注意力头维**（每头的 `d`）。
# 口径：★ 取 `64` 是为了让缩放因子 `sqrt(d) = 8` 是**精确**的二进制数
#       （⇒ 分数不引入额外舍入）。★ 单头（不做多头拆分 —— 头怎么分给设备是
#       `tensor-parallel/02` 的正文，本主题不写）。
# 消费者：`02_ring_attention.py` 的 `A.1`/`B.1`/`B.2`/`C.1`、
#         `03_constraints.py` 的 `A.1`/`B.1`/`B.2`/`C.1`/`C.2`/`C.3`、`D.7`。
# 同值位置：无。

RING_CP_GRID = (1, 2, 4, 8, 16)
# 来源：[E] 本库设计。**五档** CP（上下文并行）度（★ 全为 2 的幂 ⇒ 每档都整除 `ATTN_S`）。
# 口径：★★ 第 1 档 `cp = 1` 是**退化对照**（环上只有一台设备 ⇒ 差应当**恰为 `0.0`**）。
#       ★★ **本元组必须逐元素钉住**（`D.7`）。
# 消费者：`02_ring_attention.py` 的 `A.1`、`A.2`、`B.1`、`D.7`。
# 同值位置：`CP_GRID`（03 用的四档 —— **少一档 `16`**，因为 03 的 `A.1` 带 causal mask，
#           `cp = 16` 时每块只有 32 行、mask 的效应被块大小淹没 ⇒ 只到 `8`）。

RING_AMPLIFY = 100.0
# 来源：[E] 本库设计。触发朴素实现溢出的**放大倍数**（把 `Q` / `K` 同乘它）。
# 口径：★★★ **`02` 的 C 类判据必需它** —— 原尺度下朴素版**看起来完全正确**
#       （与参照只差 `1 ulp` 档），**必须把尺度放大到溢出区**才能让它现形。
#       ★★ **这是一条「反例必须证明它真的会错」的落法**：判据断言的**不是**「它这次没错」，
#         而是「**在这个尺度下它给 `nan`**」（`C.1` 用 `== 100.0` 的说明 + 实测数）。
# 消费者：`02_ring_attention.py` 的 `C.1`、`D.7`。
# 同值位置：无。

TOL_MACHINE = 1.0e-12
# 来源：[E] 本库设计。**机器精度档**的判据阈（相对/绝对通用）。
# 口径：★★ 本主题的判据大量是**逐位对拍**（blockwise ↔ 全量）⇒ 差应当落在 **1e-16 档**；
#       取 `1e-12` 是为了同时容纳「恰好为 `0`」与「几 ulp」两种正确形态。
#       ★★ **不得把它当作「物理量级的阈」**。
#       ★★★ **本主题【不】另有「精确为 0」的阈** —— `01` 的 `A.1`/`A.2`/`A.3` 实测**恰为 `0.0`**，
#         但判据仍写 `<= TOL_MACHINE`（★ 因为「恰好 0」是**实测观察**，不是设计保证）。
# 消费者：`01_activation_sharding.py` 的 `preflight()`/`A.1`/`A.2`/`A.3`/`B.1`/`B.2`、
#         `02_ring_attention.py` 的 `preflight()`/`A.1`/`B.1`/`C.1`、
#         `03_constraints.py` 的 `preflight()`/`A.1`/`B.1`/`C.3`。
# 同值位置：`data-parallel`/`tensor-parallel`/`pipeline-parallel` 的**编号程序**里各有一个同名常量
#           （★ 它们**不在**那三个域的 `specs.py` 里）；本域这一份**独立存放**，靠本条互指。


# ===========================================================================
# 3. Context Parallelism —— [B] 某实现的官方文档（NVIDIA Megatron-LM 仓库的 `context_parallel.md`）
#    ★★ **来源级别是 [B]，不是 [A]** —— 它是**一个实现**的文档，**不是规范**。
#       ★ 本域的头号错误模式就是**把「某实现的文档」当成「规范的文档」**。
#    ★★★ 它是**活文档**（仓库存档在 `main` 分支，**无版本号**、无标签）⇒
#       ★ **引用时写明抓取日（2026-10-08）**，★★ **不做行号锚点**
#       （本库 `80-model-architectures` 的实测教训：行号锚点会随维护失效）。
#    ★ 载体 = `https://raw.githubusercontent.com/NVIDIA/Megatron-LM/main/docs/user-guide/features/context_parallel.md`，
#      `curl -sSL --max-time 25` 取回（4673 字节），逐条 `count()`。
# ===========================================================================

CP_DEFINITION_QUOTE = "Context Parallelism (CP) is a parallelization scheme on the sequence-length dimension."
# 来源：[B] Megatron-LM `context_parallel.md`（活文档，抓取日 2026-10-08），**原始字节 1 次**。
# 口径：★ CP 的**一句话定义**：**切序列维**。★ 注意与 `SP_SEQ_DIM_QUOTE` 对照 ——
#       两者都切序列维，**但范围不同**（见下 `CP_VS_SP_QUOTE`）。
# 消费者：`03_constraints.py` 的 `D.1`。
# 同值位置：`SP_SEQ_DIM_QUOTE`（SP 也按序列维切，范围不同）。

CP_VS_SP_QUOTE = "Unlike prior SP (sequence parallelism), which only splits the sequence of Dropout and LayerNorm activations, CP partitions the network inputs and all activations along the sequence dimension."
# 来源：[B] Megatron-LM `context_parallel.md`（活文档，抓取日 2026-10-08），**原始字节 1 次**。
# 口径：★★★ **本主题 03 的题眼、也是「SP 与 CP 的分工」的逐字依据**：
#       · **SP 只切 Dropout 与 LayerNorm 的激活**（= 01 里那 `10sbh` 那一块）；
#       · **CP 连网络输入与【全部】激活一起按序列维切**。
#       ⇒ ★★ 这解释了 01 的结论（SP 的收益 `10sbh` 是 `O(s)`）：**SP 不切 attention 的 `O(s²)` 那一项**
#         ⇒ **序列越长，SP 越不够用、CP 越必需**（01 的 `B.1`/`B.2` 给出这条引子，03 的 `D.1` 收口）。
#       ★★ 它同时也是「**共享内存 + 单机**」与「**跨设备**」的分界：SP 是 **TP 组内**的副产物。
# 消费者：`03_constraints.py` 的 `D.1`（★ `01` 的 `B.2` 也回链这一句）。
# 同值位置：`100-parallelism/README.md` 的「五处同名不同指 + 一处极易被合并」一节**逐字引过同一句**
#           （★ 本模块的这份**独立存放**，靠本条互指）。

CP_ALLGATHER_QUOTE = "CP therefore requires an additional all-gather across GPUs to collect the full sequence of KV."
# 来源：[B] Megatron-LM `context_parallel.md`（活文档，抓取日 2026-10-08），**原始字节 1 次**。
# 口径：★★★ **CP 的第一条约束**：注意力要看到**全序列的 KV** ⇒ 必须有一次额外的 all-gather。
#       ★★ **本模块不复核它的通信口径**（那是实现侧的事）。
# 消费者：`03_constraints.py` 的 `D.2`、`B.1` 的说明文案。
# 同值位置：`CP_QKV_QUOTE`（同一条约束的另一半：**为什么**要 gather）。

CP_QKV_QUOTE = "For attention, the Q (query) of each token must combine with the KV (key and value) of all tokens in the same sequence."
# 来源：[B] Megatron-LM `context_parallel.md`（活文档，抓取日 2026-10-08），**原始字节 1 次**。
# 口径：★★ **all-gather 的动机**：注意力是**唯一有跨 token 算子**的模块（见 `CP_NO_CHANGE_QUOTE`）。
# 消费者：`03_constraints.py` 的 `D.2`。
# 同值位置：无。

CP_CHUNK_KV_QUOTE = "To reduce activation memory footprint, each GPU stores only the KV of a sequence chunk in forward and gathers KV again in backward."
# 来源：[B] Megatron-LM `context_parallel.md`（活文档，抓取日 2026-10-08），**原始字节 1 次**。
# 口径：★★★ **CP 的第二条约束（也是它的显存收益来源）**：前向**只存自己那一段的 KV**、
#       反向**再 gather 一次** ⇒ 用**通信**换**显存**（这正是 `RING_ACCUMULATION_QUOTE` 说的「别攒」）。
# 消费者：`03_constraints.py` 的 `D.2`。
# 同值位置：无。

CP_RING_P2P_QUOTE = "The all-gather and reduce-scatter are implemented as point-to-point communications in a ring topology."
# 来源：[B] Megatron-LM `context_parallel.md`（活文档，抓取日 2026-10-08），**原始字节 1 次**。
# 口径：★★ **「环」在 CP 里是【实现选择】**（把 all-gather / reduce-scatter 落成点对点环传），
#       与 `RING_SEND_KV_QUOTE` 描述的是同一件事。
#       ★★ **禁止重写集合通信原语的算法与复杂度**（域 README 的硬边界：正文在
#         `110-communication/collectives`与 `10-foundations/performance-models/02`）。
# 消费者：`03_constraints.py` 的 `D.2`、`B.1` 的说明文案。
# 同值位置：`RING_SEND_KV_QUOTE`（同一机制的两种措辞：论文侧 / 实现文档侧）。

CP_RING_HIGH_PERF_QUOTE = "but targets higher performance by"
# 来源：[B] Megatron-LM `context_parallel.md`（活文档，抓取日 2026-10-08），**原始字节 1 次**。
#       ★★ 原句是 `CP is similar to [Ring Attention](https://arxiv.org/abs/2310.01889) but targets
#         higher performance by (1) …（2) …` —— 那个 markdown 链接**跨标记** ⇒ **只取链接之后**。
# 口径：★★★ **实现文档【自己承认】CP 与 Ring Attention 是同一类机制** ——
#       所以本主题的 `02`（用 Ring Attention 的措辞讲机制）与 `03`（用 CP 文档讲约束）**不冲突**。
#       ★★ **本模块把它登记为 [B] 级**（实现的宣称），**不当规范结论**。
# 消费者：`03_constraints.py` 的 `D.3`。
# 同值位置：无。

CP_CAUSAL_QUOTE = "avoiding extra work from lower-triangle causal masking while keeping load balanced across GPUs"
# 来源：[B] Megatron-LM `context_parallel.md`（活文档，抓取日 2026-10-08），**原始字节 1 次**。
# 口径：★★★ **本主题 03 的 C 类锚**：**causal mask 与 CP 的负载均衡是有张力的** ——
#       朴素做法里「下三角之外的块算白算」；CP 的实现**特意避开**这份额外工作。
#       ⇒ ★★ `03` 的 `C.1`（把块的 token 用**局部位置**而不是**绝对位置**）
#         正是这份张力的**写错面**：mask 一错，**不报错但结果错**。
# 消费者：`03_constraints.py` 的 `D.3`、`C.1` 的说明文案。
# 同值位置：无。

CP_CP_DEGREE_QUOTE = "With CP, each GPU computes on part of the sequence, which scales down both compute and communication by the CP degree."
# 来源：[B] Megatron-LM `context_parallel.md`（活文档，抓取日 2026-10-08），**原始字节 1 次**。
# 口径：★★ **CP 把【计算与通信一起】**按 CP 度下降（因为序列维切了，注意力矩阵也切了）。
#       ★★ **本模块不复核这句的测量口径**（实现文档的宣称）。
# 消费者：`03_constraints.py` 的 `D.3`。
# 同值位置：无。

CP_MEM_PER_GPU_QUOTE = "The activation memory footprint per GPU is also smaller by the CP degree, which reduces OOM risk."
# 来源：[B] Megatron-LM `context_parallel.md`（活文档，抓取日 2026-10-08），**原始字节 1 次**。
# 口径：★★ **CP 的显存收益是【每卡】的说法**（与 `RING_QUADRATIC_QUOTE` 的「全量 `O(S²)`」是**不同量**）。
# 消费者：`03_constraints.py` 的 `D.3`、`B.2` 的说明文案。
# 同值位置：`RING_QUADRATIC_QUOTE`（不同量：一个说全量的复杂度，一个说每卡的降幅）。

CP_NO_CHANGE_QUOTE = "With CP, all modules except attention (for example, Linear and LayerNorm) can work as usual without any changes, because they do not have inter-token operations."
# 来源：[B] Megatron-LM `context_parallel.md`（活文档，抓取日 2026-10-08），**原始字节 1 次**。
# 口径：★★ **约束的边界**：CP 只动 attention（**有跨 token 算子的那一个**）；
#       其余模块**照常**（它们的算子逐 token 独立）。
#       ★★ 与 `CP_VS_SP_QUOTE` 合看：**SP 切的是那些「逐 token」的算子的激活，CP 切的是全部**。
# 消费者：`03_constraints.py` 的 `D.2`。
# 同值位置：无。

CP_GRID = (1, 2, 4, 8)
# 来源：[E] 本库设计。**四档** CP 度（★ 03 的 `A.1` 带 causal mask ⇒ 只到 `8`，见 `RING_CP_GRID`）。
# 口径：★★ 第 1 档 `cp = 1` 是**退化对照**（单卡 causal 注意力 ⇒ 差应当**恰为 `0.0`**）。
#       ★★ **本元组必须逐元素钉住**（`D.7`）。
# 消费者：`03_constraints.py` 的 `A.1`、`A.2`、`C.1`、`C.2`、`D.7`。
# 同值位置：`RING_CP_GRID`（五档，多一个 `16`；两处都是**独立存放**）。

CP_PAD_S = 100
# 来源：[E] 本库设计。**不整除**那组用的序列长度（★ 它故意**不是** 2 的幂）。
# 口径：★★ **`S % cp != 0` 的补齐代价**（计划的 P25）：每台设备分到 `ceil(S/cp)` 个 token
#       ⇒ 总计算量从 `S` 涨到 `cp·ceil(S/cp)`。★ 取 `100` 与 `8` 是为了让代价**小但可见**
#       （`100 → 104`，即 `4%`）—— ★★ **太小会被舍入淹没、太大就不像真实场景**。
# 消费者：`03_constraints.py` 的 `B.1`、`D.7`。
# 同值位置：无。

CP_PAD_DEGREE = 8
# 来源：[E] 本库设计。与 `CP_PAD_S` 配对的那一档 CP 度。
# 口径：★ `ceil(100/8) = 13`，`8 × 13 = 104` ⇒ 补齐 4 个 token 的当量。
# 消费者：`03_constraints.py` 的 `B.1`、`D.7`。
# 同值位置：无。


# ===========================================================================
# 4. UNSOURCED —— 待办清单，不是免责声明
# ===========================================================================
#
# ★ `CONTRIBUTING` §2：「不得**既保留数字又记 UNSOURCED**」—— 下面每一项都**没有**出现在
#   本模块的正文常量里。★ 本元组的字符串**没有消费者** ⇒ 它们会是 `make mutate` 的
#   **漏网候选**（本库的既有形态：`90-training/*/code/specs.py` 同节，由编号程序的 `part_D`
#   各钉一次，**不给它开后门**）。

# ★★ 本元组**留空**（与同域 `pipeline-parallel` 的形态一致）—— 理由有两条，**都是实测教训**：
#   · **每一项的字符串都会成为 `make mutate` 的叶子**，而它们**没有消费者**
#     ⇒ 会全部记成**漏网**（`80` 实测「只用于打印的模块级常量表」制造过 2 条）。
#   · 本主题的两处「未核到」已经**写在正文常量自己的 `# 来源：` 里**，不欠一条记录：
#     ① `context_parallel.md` 是**活文档**（无版本号）⇒ 抓取日之外的版本上是否逐字相同**未逐版核对**
#        ⇒ 引用时写抓取日、**不做行号锚点**（见 `CP_DEFINITION_QUOTE` 那一节的抬头）；
#     ② `2310.01889` 的**同行评审状态**只写成「**未确认**」（依据 = abs 页上
#        `ICLR` / `MLSys` / `Accepted` / `Proceedings` / `Conference` / `NeurIPS` 各 **0** 次）
#        ⇒ ★★ **本模块不登记任何会议**（见 Ring Attention 那一节的抬头）。
UNSOURCED = ()

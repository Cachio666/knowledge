"""100-parallelism/tensor-parallel 的外部事实常量源。

本模块是 `CONTRIBUTING` §2 约定的**共享事实源**（`specs.py`）：本主题的编号程序从它读常量，
推导过程可复核，**常量本身不可自证**。

★ **只放本主题真的会断言的常量**：没有消费者的常量就是 `make mutate` 的漏网候选
  （`80` 实测「只用于打印的模块级常量表」制造过 2 条漏网）⇒ **要么给它消费者、要么不写**。
  ★ 本模块的每一条常量都在**至少一个编号程序**的 `D.5` 里被 `==` 逐位钉住
  （消费位置逐条写在下面；`mutate.py` 对 `specs.py` 的变异体是**遍历全部编号程序**检验的）。

★ **来源分级 A–E**（判据见 `CONTRIBUTING` §2：A 厂商规范 / B 产品官方文档 / C 论文 /
  D 社区 / E 本库推导）。**每条常量带**：
    `# 来源：` + `[A-E]` 级 + 抓取日 + `# 同值位置：`
  ★★ **来源必须分级并写明是哪一级** —— 本域的头号错误模式是
    **把「某实现的文档」当成「规范的文档」**（B 级当 A 级）。
  ★ 引文类常量另写清**载体**（abs 页 / PDF 文本层 / docs 页 / 源码 tag），
    且**引文片段一律不含换行**。

★ **三类真值来源必须分开标注**：
    ① **外部规格值**：论文原文（**C 级，须注明是否确认同行评审**）、厂商/产品文档（A/B 级）
       ⇒ **本模块存放的就是这一类**。
    ② **本机实测值**：本域程序在本机跑出的数（**须写 python3 与 numpy 版本**）⇒ **不放在本模块**。
    ③ **推导值**：程序算出的量（可手算复核）⇒ **不放在本模块**
       （`CONTRIBUTING` §2：「只放从来源抄来的数，**不放推导值**」）。

★★ **本域的两条自缚纪律**：
    · **所有数值常量取到 `< 2^53`** —— `mutate.py` 的浮点变异是 `v + 1.0`，
      大浮点数（`2^53` 以上）会**吞掉**这个增量（变异了等于没变异）。
      本模块最大的数是 `51200`，远在限内。
    · **幂次一律写成十进制字面量** —— `mutate.py` 的 `_walk()` **不下钻 `BinOp`**
      ⇒ `2.0 ** -6` 这类写法整体**不进变异分母**。本模块**一处幂次都没有**。

★★ **不设跨主题共享模块**（`CONTRIBUTING:36`「`code/` 必须自包含」是**全库规则**）：
  跨主题同值的常量**各写一份**，靠 `# 同值位置：` 互指。

★ **本主题的禁区**（细节见 `requirements.txt` 与域 README 的「本域不做什么」表）：
    **集合通信原语的算法与复杂度**（ring / tree allreduce、`V(N) = 2(N−1)/N × S`、`S` 的饱和）
      ⇒ 回链 `110-communication` 与 `10-foundations/performance-models/02-amdahl-and-scaling.md:209`；
    **链路带宽的三种口径** ⇒ `20-gpu-architecture/interconnect/01-bandwidth-conventions.md`。
  ★★ 因此本模块**不含**任何带宽 / 延迟 / 链路规格常量 —— 那类常量一条都不许出现在这里。
    ★★★ **而它【不记 `UNSOURCED`】**：`V(N)` 的**真正性质是「不在本主题」，不是「找不到可定位
      一手来源」**（`CONTRIBUTING:65` 的口径）—— 它**有**来源（正文在 `:209`），
      而域 README 的「本域不做什么」表里**已经有它那一行**。

★★★ **本模块最要紧的一条读法（出处纪律，2026-09-23 当场复核）**：
  **「TP 限于单节点」的一手出处是 `2104.04473 §3.2` 的 `Takeaway #1`，不是 `1909.08053`。**
  当场实测（`curl -sSL` 取原始字节后 `count()`）：
    · `2104.04473` 的 ar5iv 页面上 `tensor model parallelism should generally be used up to degree`
      **计 1 次**（同页 `Takeaway #1:` 计 1 次）；
    · `1909.08053` 的 ar5iv 页面上 `single node` **计 0 次**、`g-GPU` **计 0 次** ——
      它只在**未来工作**一节提到 `inter-node model parallelism`（**计 1 次**），
      那是「未来可以怎么做」，**不是** `Takeaway`。
  ⇒ ★ 正文凡引这一条，只能引 `2104.04473 §3.2`（`MEGATRON_TP_NODE_TAKEAWAY_QUOTE`）。

★★★ **另一条出处口径（它直接挡住一类「编机制从句」）**：
  `8bsh((t−1)/t)` 里的 **`8 = 4 × 2` 这个分解，两篇原文【都没有给】**
  —— 原文只给了**结果式**与「前向 2 次、后向 2 次」这句话。
  分解是本库用 `performance-models/02:209` 的 `V(N) = 2(N−1)/N × S` 拼出来的
  ⇒ ★ 它是 **[E] 级推导**，引用时必须标明，**不得写成「原文说 `8 = 4 × 2`」**。
"""

# ===========================================================================
# 1. Megatron-LM（Shoeybi et al. 2019）—— [C] 论文（**不确认同行评审**）
#    本节的常量全部服务于 `01_column_row.py`（列/行并行的切法与配对）与
#    `02_heads_vocab.py`（头切分 · vocab 并行 · 不整除的三种处置）。
# ===========================================================================
#
# 来源：[C] M. Shoeybi, M. Patwary, R. Puri, P. LeGresley, J. Casper, B. Catanzaro,
#       **`Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism`**，
#       arXiv **1909.08053**。
#       评审状态：**不确认同行评审** —— abs 页的 `Comments:` 字段**实测不存在**
#         （2026-09-23 当场 `curl -sSL https://arxiv.org/abs/1909.08053` 后在该页 `count()`：
#          `Comments` **计 0 次**；同页 `journal-ref` 也计 0 次；
#          页面上唯一的 DOI 是 **arXiv 自己的** `10.48550/arXiv.1909.08053`）。
#       ★★ 因此本模块**不写** NeurIPS / ICLR / ACL 之类的会议名（那是编数字）。
#       抓取（ar5iv 正文）：`https://ar5iv.labs.arxiv.org/html/1909.08053`，2026-09-23，
#       **原始字节 203970**，`sha256[:16] = e29667b94577fae1`。
#       ★ **本页连抓 3 次，字节数与摘要完全一致** ⇒ 它是**可复核的锚点**。
#       ★★ 引文**一律不跨标记**：本文的公式（`GeLU(X_1A_1+X_2A_2)≠…`、`E=[E_1,E_2]`）
#         在原始字节上是 `<math …>…</math>` ⇒ 凡跨它的片段**计不到**
#         （本模块取的片段全部是**不含公式的散文**，每一条都在原始字节上核过次数）。

MEGATRON_TP_ARXIV_ID = "1909.08053"
# 来源：[C] Megatron-LM TP（同上），abs 页 `<title>`（去掉 `[1909.08053] ` 前缀后的编号本体）。
# 口径：★ 编号与标题**成对存放**（`P21` 的落法：缺一不可）。
# 消费者：`01_column_row.py` 的 `D.5`、`02_heads_vocab.py` 的 `D.5`。
# 同值位置：`03_saturation.py` 引的是**另一篇**（`2104.04473`）⇒ 不是同值位置。

MEGATRON_TP_TITLE = "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism"
# 来源：[C] Megatron-LM TP（同上），abs 页 `<title>`，**原始字节 37 次**（ar5iv 页面上它出现在
#       页眉、参考文献等处）—— 本模块登记的是**标题本体**，逐字。
# 口径：★★ 本主题的**立足点**：列/行并行、头切分、vocab 并行都出自这一篇。
# 消费者：`01_column_row.py` 的 `D.5`、`02_heads_vocab.py` 的 `D.5`。
# 同值位置：无。

MEGATRON_TP_AR5IV_URL = "https://ar5iv.labs.arxiv.org/html/1909.08053"
# 来源：[C] Megatron-LM TP（同上），`curl -sSL` 取回的**实际 URL**（HTTP 200，无重定向）。
# 口径：★ 引文类常量必须写清**载体**（ar5iv 正文页 + URL + 抓取日 + 字节数 + 摘要）。
# 消费者：`01_column_row.py` 的 `D.5`。
# 同值位置：无。

MEGATRON_TP_PAGE_BYTES = 203970
# 来源：[C] Megatron-LM TP（同上），`curl -sSL ... | wc -c` 口径的**原始字节数**（2026-09-23）。
# 口径：★ 字节数是「引的是哪一次抓取」的第二重锚点（第一重是 sha256）。
#       ★★ **载体的一条必须写明的性质：ar5iv 会重渲染** —— 本模块登记的是**上游那一次**的
#          字节数，它**会漂**；本轮的复核方式是「同一次会话内连抓 3 次是否一致」，**已一致**。
# 消费者：`01_column_row.py` 的 `D.5`。
# 同值位置：无。

MEGATRON_TP_PAGE_SHA256_16 = "e29667b94577fae1"
# 来源：[C] Megatron-LM TP（同上），对本机取回的**原始字节**求 `sha256` 后取前 16 位十六进制。
# 口径：★★ 摘要与字节数**成对存放**：只有字节数时，一次微小的页面改动不会被发现。
# 消费者：`01_column_row.py` 的 `D.5`。
# 同值位置：无。

MEGATRON_GELU_NONLINEAR_QUOTE = "Since GeLU is a nonlinear function,"
# 来源：[C] Megatron-LM TP（同上），§3 的 MLP 段，**原始字节 1 次**。
#       ★ 它紧接的后半句是一个**公式**：`GeLU(X_1A_1+X_2A_2)≠GeLU(X_1A_1)+GeLU(X_2A_2)`
#         （在原始字节上是 `<math …>` 包住的 ⇒ **跨标记，不取**）
#         ⇒ 本模块只取这句**散文**，公式部分见下一句常量的注记。
# 口径：★★★ **本主题最要紧的一句机制理由**：列并行必须放在非线性之前，
#       **根因是非线性不可交换**。★ 它是**原文的**，不是本库编的机制从句。
# 消费者：`01_column_row.py` 的 `A.2`/`D.5`。
# 同值位置：无。

MEGATRON_GELU_SYNC_QUOTE = "and this approach will require a synchronization point before the GeLU function."
# 来源：[C] Megatron-LM TP（同上），§3 的 MLP 段，**原始字节 1 次**。
# 口径：★★ 它说的是**行并行做在第一层**的代价：**非线性之前多一个同步点**
#       ⇒ 「列并行必须在前面」的**代价论证**就落在这句上。
#       ★★ 注意它的**作用域**：这句话只说「（行并行）需要一个同步点」，
#         **不是**说「列并行不需要任何通信」（列并行的**输出**仍然要被后面的行并行规约）。
# 消费者：`01_column_row.py` 的 `A.2`/`D.5`。
# 同值位置：无。

MEGATRON_COL_SPLIT_LEAD_QUOTE = "Another option is to split"
# 来源：[C] Megatron-LM TP（同上），§3 的 MLP 段，**原始字节 1 次**。
#       ★ 它后面的宾语是一个公式（`A=[A_1,A_2]`，跨标记）⇒ **只取这个引导语**。
# 口径：★ 它是「另一种做法」的**引导句** —— 与上一句（行并行）构成**对照**：
#       行并行（先）→ 非线性之前有同步点；列并行（先）→ **removes** 那个同步点。
# 消费者：`01_column_row.py` 的 `D.5`。
# 同值位置：无。

MEGATRON_COL_REMOVES_SYNC_QUOTE = "This is advantageous as it removes a synchronization point."
# 来源：[C] Megatron-LM TP（同上），§3 的 MLP 段，**原始字节 1 次**。
# 口径：★★★ **「列并行为什么非得在前面」的结论句**（逐字，`removes`）。
#       ★ 它**不是**一句可以外推的话：原文紧接着说明了列并行之后的第二层 GEMM
#         要**按行切**（`we … split the second GEMM along its rows so it takes the output of
#         the GeLU layer directly without requiring any communication`）——
#         即「**列 + 行**」这一个**配对**才是免通信的；单独的列并行不是。
#         ⇒ 本模块**只引这一句**，配对约束的完整表述由 `01` 的 `A.3` 用**四格**钉住。
# 消费者：`01_column_row.py` 的 `A.2`/`A.3`/`D.5`。
# 同值位置：无。

MEGATRON_TWO_ALLREDUCE_QUOTE = "two all-reduces in the forward path and two in the backward path"
# 来源：[C] Megatron-LM TP（同上），§3 末段，**原始字节 1 次**。
# 口径：★★★ **本主题的判定句，只能引它**：一个 transformer 层的前向有 **2** 次 all-reduce、
#       后向有 **2** 次。★★ **它的作用域必须写明**：这是**一个 transformer 层**
#       （1 个 MLP 块 + 1 个 self-attention 块**各 1 次**），**不含**词表并行输出的那次额外通信。
#       ★★ **`:223` 只说「attention 后、FFN 后各一次」（= 前向 2 次），漏了 backward 那 2 次**
#         ⇒ 本域补上的正是这后半句（见 `03_saturation.py` 的 `A.1` 与 `D.5`）。
# 消费者：`01_column_row.py` 的 `B.2`/`D.5`、`03_saturation.py` 的 `A.1`/`D.5`。
# 同值位置：无（本库只有本主题引这一句）。

MEGATRON_FIG4_QUOTE = "There are 4 total communication operations in the forward and backward pass of a single model parallel transformer layer."
# 来源：[C] Megatron-LM TP（同上），**Figure 4 的 caption**，**原始字节 1 次**。
# 口径：★★★ 它把 `2 + 2` 写成 **`4 total`**，并且把「总量」钉在**单个 transformer 层**上
#       ⇒ 这是 `8bsh((t−1)/t)` 里那个 **`4`** 的**一手依据**（`8 = 4 × 2`）。
#       ★★ 而 `8 = 4 × 2` 这个**分解本身**两篇原文都没有给（见模块头）⇒ 分解标 **[E]**。
# 消费者：`01_column_row.py` 的 `B.2`/`D.5`、`03_saturation.py` 的 `A.1`/`D.5`。
# 同值位置：无。

MEGATRON_HEAD_LOCAL_QUOTE = "in a column parallel fashion such that the matrix multiply corresponding to each attention head is done locally on one GPU"
# 来源：[C] Megatron-LM TP（同上），§3 的 self-attention 段，**原始字节 1 次**。
# 口径：★★★ **头切分的原文依据**：QKV 按列并行 ⇒ **每个 head 的矩阵乘在本卡本地算**。
#       ★★ **作用域**：原文接着说 `and doesn’t require any immediate communication to complete
#         the self-attention` —— 「**立即**」这个词很重要：**紧接着的 output linear 层是按行并行的**
#         ⇒ 那次 all-reduce 仍然发生（只是可以晚一点）⇒ **不得写成「头切分不需要通信」**。
#       ★★ 与本域 `80-model-architectures/attention-variants/01` 的**边界**：那边讲 MHA/MQA/GQA
#         的**结构**与 KV head 与 Q head 的**配比**，**本文只讲「头怎么分给设备」**。
# 消费者：`02_heads_vocab.py` 的 `D.5`。
# 同值位置：无。

MEGATRON_EMBED_ALLREDUCE_QUOTE = "Since each partition now only contains a portion of the embedding table, an all-reduce ("
# 来源：[C] Megatron-LM TP（同上），§3 的 vocab 并行段，**原始字节 1 次**。
#       ★ 它后面紧跟一个公式（那个算子名 `g` 被 `<math>` 包住）⇒ 本片段**到 `(` 为止**，
#         是原始字节上可数的**最长前缀**。
# 口径：★★★ **输入 embedding 列切 ⇒ 需要一次 all-reduce**（把各卡的**部分嵌入表**加起来）。
#       ★★ 本库实测（`02_heads_vocab.py` 的 `A.3`）：这次 all-reduce 是**精确的**——
#         每个 token 在每张卡上的分片里**恰有一项非零**（其余是**结构性的零**），
#         所以「加 t 个分片」与「直接查表」**逐位相同** ⇒ 那条判据**恒真**（不能当 A 类）。
# 消费者：`02_heads_vocab.py` 的 `A.3`/`B.1`/`D.5`。
# 同值位置：无。

MEGATRON_VOCAB_HUGE_QUOTE = "which is huge due to vocabulary size being large."
# 来源：[C] Megatron-LM TP（同上），§3 的 vocab 并行段，**原始字节 1 次**。
# 口径：★★ 原文用来形容**all-gather 的通信量**（`b×s×v` 个元素）的那个词是 **`huge`**
#       ⇒ 「vocab 并行的代价在输出侧」这句结论的**逐字依据**。
# 消费者：`02_heads_vocab.py` 的 `B.1`/`D.5`。
# 同值位置：无。

MEGATRON_VOCAB_FUSE_QUOTE = "with the cross entropy loss which reduces the dimension to"
# 来源：[C] Megatron-LM TP（同上），§3 的 vocab 并行段，**原始字节 1 次**。
#       ★ 它后面紧跟公式 `b×s`（跨标记）⇒ **只取引导语**；降到的那个维度由
#         `MEGATRON_VOCAB_FUSED_DIM_QUOTE` 之外的推导钉住（`02` 的 `B.1` 实测张量形状）。
# 口径：★★★ **把并行 GEMM 与交叉熵**融合** ⇒ 通信量从 `b×s×v` 降到 `b×s`**（差 `v` 倍）
#       ⇒ 「vocab 并行的通信量」这个知识点的**唯一出处**。
# 消费者：`02_heads_vocab.py` 的 `B.1`/`D.5`。
# 同值位置：无。

MEGATRON_VOCAB_PER_GPU_MULTIPLE = 128
# 来源：[C] Megatron-LM TP（同上），§5.1 逐字
#       `it is beneficial for the per-GPU vocabulary size to be a multiple of 128`
#       （**原始字节 1 次**）。
# 口径：★★★ **不整除的第三种处置（补齐）里那个 `128`** —— 它是**每卡**词表大小的倍数要求。
#       ★★ **它的理由原文只给了「beneficial for efficient GEMMs」**（同句所在段逐字
#         `to have efficient GEMMs for the logit layer` ⇒ 也是 1 次）⇒
#         **`128` 这个具体取值的理由（tile/对齐）原文没给** ⇒ 已记 `UNSOURCED`：
#         `megatron_vocab_pad_multiple_128_rationale`。
# 消费者：`02_heads_vocab.py` 的 `B.2`/`C.3`/`D.5`。
# 同值位置：无。

MEGATRON_VOCAB_PAD_TP_DEGREE = 8
# 来源：[C] Megatron-LM TP（同上），§5.1 逐字：
#       `Since we study up to 8-way model parallelism, we pad the vocabulary such that it is
#        divisible by` **（原始字节 1 次）**，其后的 `128×8=1024` 是**公式**（跨标记）。
# 口径：★★★ 这个 `8` 是**例子里取的 TP 度**（8 亿参数的模型用 8 路）——
#       ★ **它不是一个规格常量**，是**那组实验的配置**。
#       ★★ 与 `MEGATRON_VOCAB_PER_GPU_MULTIPLE` 相乘得 `1024`（见 `02` 的 `B.2`）。
# 消费者：`02_heads_vocab.py` 的 `B.2`/`C.3`/`D.5`。
# 同值位置：无。

MEGATRON_VOCAB_PAD_QUOTE = "pad the vocabulary such that it is divisible by"
# 来源：[C] Megatron-LM TP（同上），§5.1，**原始字节 1 次**（★ 不含公式 `128×8=1024`）。
# 口径：★★★ **不整除的第三种处置的逐字出处**：**补齐到能被整除**。
#       ★★ 本库实测（`02` 的 `A.1`/`A.2`）：**补齐之后「与单卡逐位相同」【不成立】** ——
#         补齐改变了张量的形状与内存布局 ⇒ BLAS 的路径可能变 ⇒ 判据**只能写「机器精度档」**。
# 消费者：`02_heads_vocab.py` 的 `A.1`/`A.2`/`D.5`。
# 同值位置：无。

MEGATRON_VOCAB_PAD_RESULT_QUOTE = ", resulting in a padded vocabulary size of 51,200."
# 来源：[C] Megatron-LM TP（同上），§5.1，**原始字节 1 次**。
# 口径：★ 它给出**补齐后的实际词表大小** `51,200`（由 `50,257` 补到 `1024` 的倍数）。
# 消费者：`02_heads_vocab.py` 的 `B.2`/`D.5`。
# 同值位置：无。

MEGATRON_VOCAB_ORIGINAL = 50257
# 来源：[C] Megatron-LM TP（同上），§5.1 逐字 `The original vocabulary size was 50,257`
#       （**原始字节 1 次**；同段还有 `GPT-2 used a vocabulary size of 50,257`，那是另一处，**不取**）。
# 口径：★ 它是补齐**之前**的词表大小（GPT-2 的 BPE 词表）⇒ 与 `51200` 成对使用。
# 消费者：`02_heads_vocab.py` 的 `B.2`/`D.5`。
# 同值位置：无。

MEGATRON_VOCAB_PAD_SIZE = 51200
# 来源：[C] Megatron-LM TP（同上），§5.1 的 `, resulting in a padded vocabulary size of 51,200.`
#       （同一个数在这里以**整数**形态寄存，供程序做整除判据）。
# 口径：★★ `51200 = 50 × 1024` ⇒ 它是 **`128 × 8 = 1024` 的整数倍**，而 `50257` **不是**
#       （`50257 % 1024 = 81`，`1024 × 49 = 50176`）⇒ 「补齐」这个词的**可机械判定的含义**
#       就在这里；★ 补齐多出来的是 `51200 − 50257 = 943` 个（`81 + 943 = 1024`）。
#       ★★ 它与 `MEGATRON_VOCAB_PAD_RESULT_QUOTE` 是**同一个数的两种形态**（引文 / 整数），
#         **两处都留**：引文供正文逐字引用，整数供程序做整除判据。
# 消费者：`02_heads_vocab.py` 的 `B.2`/`D.5`。
# 同值位置：无。

MEGATRON_8WAY_NOTE_QUOTE = "The configuration with 1.2 billion parameters fits on a single GPU whereas the 8 billion parameter model requires 8-way model parallelism (8 GPUs)."
# 来源：[C] Megatron-LM TP（同上），§5.1，**原始字节 1 次**。
# 口径：★★ **把 `MEGATRON_VOCAB_PAD_TP_DEGREE = 8` 变成可复核的**：那个 `8` 就是这一句里的
#       `8-way`（8 亿参数模型用 8 路模型并行）。
#       ★★★ **它【不是】「TP 限于单节点」的出处** —— 这一句只说**那一组实验**用了 8 卡；
#         「TP 一般只用到一个节点内的 `g` 卡」的出处是 **`2104.04473 §3.2` 的 `Takeaway #1`**
#         （见 `MEGATRON_TP_NODE_TAKEAWAY_QUOTE`）。★★ **这是一个极易被张冠李戴的地方。**
#       ★ 本域**不重写**这三条（它们已被 `performance-models/02:223` 逐字写完）。
# 消费者：`02_heads_vocab.py` 的 `D.5`。
# 同值位置：无。

# ===========================================================================
# 2. Megatron-LM 3D 并行（Narayanan et al. 2021）—— [C] 论文（**同行评审：是**）
#    本节的常量全部服务于 `03_saturation.py`（TP 的通信量也饱和）。
# ===========================================================================
#
# 来源：[C] D. Narayanan, M. Shoeybi, J. Casper, P. LeGresley, M. Patwary, V. Korthikanti,
#       D. Vainbrand, P. Kashinkunti, J. Bernauer, B. Catanzaro, A. Phanishayee, M. Zaharia,
#       **`Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM`**，
#       arXiv **2104.04473**。
#       评审状态：**确认同行评审** —— 一手依据是 abs 页的 `Comments:` 字段，逐字
#         **`Accepted to SC 2021`**（2026-09-23 当场 `curl -sSL https://arxiv.org/abs/2104.04473`
#         后在该页 `count('Comments') == 1`，其值逐字为上面那串）。
#       ★★ **一处必须写明的实测**：该 abs 页上**没有** ACM 的 DOI —— 页面上出现的 DOI 是
#         **arXiv 自己的** `10.48550/arXiv.2104.04473`（`count()` 实测）。
#         ⇒ 本条**只声称「Comments 字段写了 Accepted to SC 2021」**，
#         **不声称**该页含 ACM 的论文编号。
#       抓取（ar5iv 正文）：`https://ar5iv.labs.arxiv.org/html/2104.04473`，2026-09-23，
#       **原始字节 208446**，`sha256[:16] = 9dea9fa431b59daf`。
#       ★ **本页连抓 3 次，字节数与摘要完全一致** ⇒ 可复核的锚点。
#       ★★ 引文**一律不跨标记**：公式（`8bsh((t−1)/t)`、`M=t·p`、`g`）在原始字节上是
#         `<math …>…</math>` ⇒ 本模块取的片段**全部是不含公式的散文**，逐条核过次数。

MEGATRON_PP_ARXIV_ID = "2104.04473"
# 来源：[C] Megatron-LM 3D 并行（同上），abs 页 `<title>`（去掉前缀后的编号本体）。
# 口径：★ 编号与标题**成对存放**。
# 消费者：`03_saturation.py` 的 `D.5`。
# 同值位置：无。

MEGATRON_PP_TITLE = "Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM"
# 来源：[C] Megatron-LM 3D 并行（同上），abs 页 `<title>`，逐字。
# 口径：★★ 本主题 `03` 的立足点：**TP 的通信量公式**与**「TP 限于单节点」的出处**都在这一篇。
# 消费者：`03_saturation.py` 的 `D.5`。
# 同值位置：无。

MEGATRON_PP_AR5IV_URL = "https://ar5iv.labs.arxiv.org/html/2104.04473"
# 来源：[C] Megatron-LM 3D 并行（同上），`curl -sSL` 取回的**实际 URL**（HTTP 200）。
# 口径：★ 载体 = ar5iv 正文页。
# 消费者：`03_saturation.py` 的 `D.5`。
# 同值位置：无。

MEGATRON_PP_PAGE_BYTES = 208446
# 来源：[C] Megatron-LM 3D 并行（同上），`curl -sSL ... | wc -c` 口径的**原始字节数**（2026-09-23）。
# 口径：★★ 与第 1 节同形：**这是「上游那一次」的现场**，ar5iv 重渲染会让它漂。
# 消费者：`03_saturation.py` 的 `D.5`。
# 同值位置：无。

MEGATRON_PP_PAGE_SHA256_16 = "9dea9fa431b59daf"
# 来源：[C] Megatron-LM 3D 并行（同上），本机取回原始字节的 `sha256[:16]`（连抓 3 次一致）。
# 口径：★ 摘要与字节数成对；**与第 1 节那份页面一样，都是可复核的**。
# 消费者：`03_saturation.py` 的 `D.5`。
# 同值位置：无。

MEGATRON_PP_COMMENTS_QUOTE = "Accepted to SC 2021"
# 来源：[C] Megatron-LM 3D 并行（同上），**abs 页**的 `Comments:` 字段值，**原始字节 1 次**
#       （2026-09-23 当场取回 `https://arxiv.org/abs/2104.04473` 后 `count()` 核过）。
#       ★ 载体是 **abs 页**（不是 ar5iv 正文页）—— 两页的 URL 与字节数都不同，必须分清。
# 口径：★★★ **「这篇论文同行评审：是」的【一手依据】。**
#       ★★ 本库纪律：C 级来源必须注明是否确认同行评审 ⇒ 这一条就是那个注明的**证据**。
# 消费者：`03_saturation.py` 的 `D.5`。
# 同值位置：无。

MEGATRON_TP_FWD_BWD_QUOTE = "uses all-reduce communication (two all-reduce operations each in the forward and backward pass, see"
# 来源：[C] Megatron-LM 3D 并行（同上），§3.2，**原始字节 1 次**。
# 口径：★★★ **「前向 2 次、后向 2 次」的第二条独立来源**（第一条是 `1909.08053` 的
#       `MEGATRON_TWO_ALLREDUCE_QUOTE`）⇒ 两条来源不同的路径给出同一个 `4`。
#       ★★ 作用域：`two all-reduce operations each in the forward and backward pass`
#         指的是**一个 transformer 层**（与 `1909` 的 Figure 4 一致）。
# 消费者：`03_saturation.py` 的 `A.1`/`D.5`。
# 同值位置：无。

MEGATRON_TP_COMM_QUOTE = "twice each in the forward and backward pass for each layer, leading to a total communication of"
# 来源：[C] Megatron-LM 3D 并行（同上），§3.2，**原始字节 1 次**。
#       ★ 它后面紧跟**公式** `8bsh((t−1)/t)`（跨标记）⇒ 公式本体不入 `specs.py`
#         （本库规矩：跨标记不取），其**系数 `8`** 单独寄存为 `MEGATRON_TP_COMM_MULTIPLIER`。
# 口径：★★★ **语序是「先 `4`、后公式」**：`twice each in the forward and backward pass`
#       （= 4 次）**然后是** `leading to a total communication of 8bsh((t−1)/t)`
#       ⇒ ★★ 这就是「`8` 里**已经含了**前向与后向」的**语法依据**：
#         `8` 是那 `4` 次的**结果**，**不是**它的前置因子。
#         ⇒ **算「全模型」时不要再乘 2**（`03` 的 `C.1` 把这条变成一条实测的 2 倍差）。
# 消费者：`03_saturation.py` 的 `A.1`/`C.1`/`D.5`。
# 同值位置：无。

MEGATRON_TP_COMM_CALIBER_QUOTE = "per device for each microbatch."
# 来源：[C] Megatron-LM 3D 并行（同上），§3.2，**原始字节 1 次**
#       （★ 同句中间有斜体标记 `<em>per layer</em>`，故本片段**从 `per device` 起**）。
# 口径：★★★ **公式的口径三件套**：**每层**（`per layer`，斜体强调）· **每卡**（`per device`）·
#       **每 micro-batch**（`for each microbatch`）⇒ **凡引用 `8bsh((t−1)/t)` 必须带上这三件**。
#       ★ 同段还有 `Each device typically has multiple layers; the total amount of
#         tensor-parallel-communication per device for each microbatch is then …`
#         ⇒ 乘层数得到**每卡每 micro-batch 的全模型量**（`03` 的 `A.1` 用这一支）。
# 消费者：`03_saturation.py` 的 `A.1`/`D.5`。
# 同值位置：无。

MEGATRON_TP_COMM_MULTIPLIER = 8
# 来源：[C] Megatron-LM 3D 并行（同上），§3.2 的公式 `8bsh((t−1)/t)` 的**系数 `8`**
#       （★ 公式在原始字节上是 `<math>`，此数由 `alttext` 属性逐字读出 ⇒ 见模块头对
#         「跨标记不取」的口径说明：**引文片段不取，单个系数寄存为整数**）。
# 口径：★★★ **`8 = 4 × 2`，而拆开的两半各有出处**（★ 分解本身是本库拼的，标 **[E]**）：
#         · **`4`** = 前向 2 次 + 后向 2 次 all-reduce
#           ⇒ [C] `1909.08053` Fig 4 的 `4 total` + [C] `2104.04473` 的 `twice each …`；
#         · **`2`** = all-reduce 的 `2(n−1)/n` 因子
#           ⇒ [E] 本库正文 `performance-models/02-amdahl-and-scaling.md:209` 的
#             `V(N) = 2(N−1)/N × S`（**引用，不重推** —— 它的正文在彼）。
#       ⇒ 每次 all-reduce 搬 `2 × (t−1)/t × sbh`，`4 ×` 它 = `8sbh(t−1)/t` ✓
#       ★★★ **`8` 已经含了前向与后向** ⇒ 算全模型时**不要再乘 2**（`03` 的 `C.1`）。
# 消费者：`03_saturation.py` 的 `A.1`/`C.1`/`D.5`。
# 同值位置：无。

MEGATRON_TP_COMM_OPS_PER_LAYER = 4
# 来源：[C] **两条独立来源交叉印证**：`1909.08053` 的 Figure 4 caption 逐字
#       `There are 4 total communication operations in the forward and backward pass of a
#        single model parallel transformer layer.`（原始字节 1 次）；
#       `2104.04473` §3.2 逐字 `two all-reduce operations each in the forward and backward pass`
#       （原始字节 1 次）。
# 口径：★★ **一个 transformer 层**（1 个 MLP 块 + 1 个 self-attention 块）的**前向 + 后向**
#       合计 4 次 all-reduce ⇒ 每块每方向 **1** 次。
#       ★★★ **不含**词表并行输出的那次额外通信（那一支在 `02` 的 `B.1`）。
# 消费者：`01_column_row.py` 的 `B.2`/`D.5`、`03_saturation.py` 的 `A.1`/`D.5`。
# 同值位置：无。

MEGATRON_TP_NODE_TAKEAWAY_QUOTE = "tensor model parallelism should generally be used up to degree"
# 来源：[C] Megatron-LM 3D 并行（同上），§3.2 的 **`Takeaway #1`**，**原始字节 1 次**
#       （★ 它后面紧跟公式 `g`，且再后面是 ` when using ` —— 故本片段**在 `degree` 处截断**）。
# 口径：★★★ **「TP 限于单节点」的【唯一】一手出处**。
#       ★★ 当场实测的反面证据（见模块头）：`1909.08053` 的页面上 `single node` 计 **0** 次、
#         `g-GPU` 计 **0** 次 ⇒ **那句话不在 `1909` 里**。
#       ★ 本域**只给这一条出处与它的理由句**，**不重写** `performance-models/02:223`
#         已经写过的三条（那三条的正文在彼）。
# 消费者：`03_saturation.py` 的 `D.5`。
# 同值位置：无。

MEGATRON_TP_NODE_TAKEAWAY_TAIL_QUOTE = "-GPU servers, and then pipeline model parallelism can be used to scale up to larger models across servers."
# 来源：[C] Megatron-LM 3D 并行（同上），§3.2 的 `Takeaway #1`，**原始字节 1 次**
#       （★ 与上一条拼起来才是完整句：`…up to degree g when using g-GPU servers, …`，
#         中间那个 `g` 是公式 ⇒ **跨标记，切成两段存放**）。
# 口径：★★ 它给出**出路**：TP 到节点边界为止，再往上要靠**流水线并行**跨节点
#       ⇒ 「TP 的规模上限由节点内互联决定」（`performance-models/02:223` 的定性版）的
#         **原文支撑**（★ 本域只回链那句，不重写）。
# 消费者：`03_saturation.py` 的 `D.5`。
# 同值位置：无。

MEGATRON_TP_NODE_REASON_QUOTE = "is larger than the number of GPUs in a single node, the overhead of performing tensor model parallelism across slower inter-node links can be impractical."
# 来源：[C] Megatron-LM 3D 并行（同上），§3.2 的 `Takeaway #1` 之前一段，**原始字节 1 次**
#       （★ 句首的 `when t` 里那个 `t` 是公式 ⇒ 本片段**从 `is larger` 起**）。
# 口径：★★★ **Takeaway #1 的推导依据（逐字）**：`t` 超过单节点 GPU 数之后，
#       **跨节点慢链路上的 TP 开销会变得不实用**。
#       ★ 本域**只引这一句**，**不重推**「带宽对 TP 的影响」那一层
#         （`performance-models/02:223` 已经写完）。
# 消费者：`03_saturation.py` 的 `D.5`。
# 同值位置：无。

MEGATRON_TAKEAWAY1_LABEL = "Takeaway #1:"
# 来源：[C] Megatron-LM 3D 并行（同上），§3.2，**原始字节 1 次**（同一节另有 `#2` / `#3`）。
# 口径：★ 它是**定位用的标签**：`MEGATRON_TP_NODE_TAKEAWAY_QUOTE` 属于 `#1`
#       ⇒ 「`Takeaway #1`」这个坐标本身是可核的。
#       ★★ `1909.08053` 的页面上 `Takeaway` **计 0 次** ⇒ 「`Takeaway` 这个体裁是 2104 的」。
# 消费者：`03_saturation.py` 的 `D.5`。
# 同值位置：无。

# ===========================================================================
# 3. 本库正文（[E] 级）—— **引用，不重推**
#    ★★ 这一节的存在本身就是一条边界纪律：`V(N)` 的正文在
#       `performance-models/02-amdahl-and-scaling.md:209`，本主题**只引用它的系数**。
# ===========================================================================

RING_ALLREDUCE_VOLUME_FACTOR = 2.0
# 来源：[E] 本库正文 `10-foundations/performance-models/02-amdahl-and-scaling.md:209`
#       的 `V(N) = 2(N−1)/N × S`（**每卡发出的字节数是 `(N−1)/N × S` 的两倍**）。
#       ★ 该式的算法依据（NCCL 的 ring = reduce-scatter + all-gather）也在同一行，
#         **本模块不重推**，只登记那个系数 `2`。
# 口径：★★★ **它是 `8bsh((t−1)/t)` 里那个 `2` 的出处**（`8 = 4 × 2`）：
#       每次 all-reduce 搬 `2 × (t−1)/t × S` 字节（`S` = 被规约的张量字节数）。
#       ★★ **本主题不重推 `V(N)`**（域 README 的「本域不做什么」表第一行）——
#         本常量是「**引用一处已有正文**」，不是「在本主题重写它」。
# 消费者：`01_column_row.py` 的 `D.5`、`03_saturation.py` 的 `A.1`/`D.5`。
# 同值位置：无（本库只有本主题这个常量形态）。

DP_COMM_SATURATION_MULTIPLE = 2.0
# 来源：[E] 本库正文 `10-foundations/performance-models/02-amdahl-and-scaling.md:219`
#       逐字 `每卡通信量饱和到 2S`（同篇 `:215` 给出它的量：`梯度 140 GB`（70B、BF16）
#       ⇒ `2S` = `280 GB`，与 `5.600 s`（IB 50 GB/s 单向）自洽）。
# 口径：★★★ **它是 DP 那一侧的饱和值**：`2 × S`，`S` = **模型参数的字节数**（`2Ψ`，bf16）
#       ⇒ **`2S ∝ 参数量`**。★★ 与 TP 的 `8bsh`（**∝ 激活量**）**成对**才有意义 ——
#       「两者都饱和，但饱和值不同」正是本主题 `03_saturation.py` 的 `B.2`。
#       ★ 那个 `2` 与 `RING_ALLREDUCE_VOLUME_FACTOR` 是**同一个因子**（环 all-reduce 的
#         `2(n−1)/n` 在 `n → ∞` 时的极限），**但本模块不重推**（正文在 `:209`）。
# 消费者：`03_saturation.py` 的 `B.2`/`D.5`。
# 同值位置：`RING_ALLREDUCE_VOLUME_FACTOR`（**同值但不同项**：一个是 DP 的饱和倍数、
#           一个是单次 all-reduce 的系数）⇒ ★ **两条分开存放、不得合并**。

TP_BOUNDARY_REF = "10-foundations/performance-models/02-amdahl-and-scaling.md:223"
# 来源：[E] 本库**边界锚点**（不是外部事实）：`performance-models/02` 的 `:223` 那一行
#       已经逐字写完了 TP 的三条（通信发生在**每一层内部** · 通信量与**激活**成正比 ·
#       **只在高带宽互联上可用 / 通常限于单节点 8 卡**）。
# 口径：★★★ **本主题一篇都不重写那三条**（否则是「同一个载体重复」，`P2`）。
#       ★ 本常量的用途是**可执行的边界检查**：三篇配套文档**每一篇**都必须逐字出现这个
#         路径 —— 每个编号程序的 `D.5c` 会去读**自己那一篇**并检查（`P1`「订正不扩散」的落法）。
#       ★★ 文档不在时（`mutate.py` 的副本目录里只有 `*.py`）**跳过并打印**，
#         但**字面量本身**在 `D.5` 里另有逐位钉住 ⇒ 变异体不会被漏掉。
# 消费者：`01_column_row.py` 的 `D.5`/`D.5c`、`02_heads_vocab.py` 的 `D.5`/`D.5c`、
#         `03_saturation.py` 的 `D.5`/`D.5c`。
# 同值位置：无。

# ===========================================================================
# 4. UNSOURCED —— 待办清单，不是免责声明
# ===========================================================================
#
# ★ `CONTRIBUTING` §2：「不得**既保留数字又记 UNSOURCED**」—— 下面每一项都**没有**出现在
#   本模块的正文常量里（也**没有**出现在三篇文档的正文里）：
#   · 第 1 项是**一个取值的理由**（`128` 为什么是 128），本模块只保留「`128` 这个倍数」
#     这条**有出处**的事实，**不解释**为什么是 128；
#   · 第 2 项是**一个实现的机制**（Megatron 怎么屏蔽补齐位置），本模块**不给**这个机制，
#     `02` 的 `C.3` 用的是**本库自己构造的反例**（「不屏蔽会错」），不是「Megatron 的做法」。
# ★★ 本元组的字符串**有消费者**：三个编号程序的 `D.8` 各逐位钉住整条元组
#   （上游 `90-training/*/code/specs.py` 的同节形态）—— ⇒ 它们**不是**漏网候选。
# ★★★ **一项【不能】进本条**：`V(N) = 2(N−1)/N × S`。它的真正性质是「**不在本主题**」
#   而不是「找不到可定位一手来源」（`CONTRIBUTING:65`）—— 它**有**来源
#   （`performance-models/02:209`），而域 README 的「本域不做什么」表里**已经有它那一行**
#   ⇒ 把它写进 `UNSOURCED` 就是「**既保留了数字（公式）又记 `UNSOURCED`**」。

UNSOURCED = (
    # 每一项是一个**稳定标识符**（不是散文），格式：`<主题>_<条目>`
    # ① `1909.08053` 只说「每卡词表大小是 128 的倍数对 GEMM 有利」，**没有给 128 的来由**
    #    （tile 宽度？对齐？还是那组实验调出来的？）⇒ 本域**不解释**，只引「有出处的那半句」。
    "tensor_parallel_vocab_pad_multiple_128_rationale",
    # ② Megatron **具体怎样**屏蔽补齐位置的 logits（置 −inf？只取前 `V` 列？换 loss 的 index？）
    #    —— `1909.08053` 的页面上 `mask` 的 4 次出现**都不在这件事上**（实测：均属
    #    BERT 的 `whole word n-gram masking` 与数据掩码）⇒ **未取到** ⇒ 本域不写它的做法。
    "tensor_parallel_pad_logits_masking_mechanism",
)

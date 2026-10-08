"""90-training/checkpointing 的外部事实常量源。

本模块是 `CONTRIBUTING` §2 约定的**共享事实源**（`specs.py`）：本主题的三个编号程序从它读常量，
推导过程可复核，**常量本身不可自证**。

★★ **来源必须分级，并写明它是哪一级**（计划 §1.5）。本域的头号错误模式是
  **P19 —— 把「某实现的文档」当成「规范的文档」**（B 级当 A 级）。
  本主题的两条链**分属两级，必须分开写**：
    · **ZeRO 的字节账（`K = 12`、`4Ψ + 12Ψ = 16Ψ`）与 PaLM 的逐位确定性** → **[C] 论文**
      （逐条注明是否确认同行评审）；
    · **`torch.distributed.checkpoint`（DCP）的分片保存/恢复语义** → **[B] PyTorch 2.14 的文档**
      （**某实现的规格，不是规范**）⇒ 一律写明版本、URL、抓取日与字节数。
    · **分片格式的头部字节数、摘要长度** → **本程序的设计常量（E 级：本库设计）** ——
      ★★ **它不是任何来源里的数**：DCP 的文档**不给**每片头部的字节数
      （它只说「一片一个文件 + 一个全局 `.metadata`」）。这个区分写在本文件 §4。

★ **只放本主题真的会断言的常量**（计划 §3 的 R3 裁定）。每一条下面写明**预期消费者**。
  ★ **没有消费者的常量就是 `make mutate` 的漏网候选**（计划 §4 **T11 的盲区①**；
    且 **T52** 记着「有来源但无消费者的引文常量会漏网」——本主题的引文常量**逐条有消费者**）。

★★ **本模块的自缚纪律（盲区④的落法）**：`mutate.py` 的 `_walk()`
  **不下钻 `BinOp`/`UnaryOp`/`Call`** ⇒ `4 ** 2`、`2 ** -7` 这类**幂次字面量整体不进变异分母**
  （**T31/T37/T38**）。⇒ 本模块**所有数值一律写成十进制字面量**，幂次/开方写进注释。
  ★ 本主题**没有**幂次字面量（本域的第一个：`specs.py` 里 `2^-7` / `2^-14` 那类写法一处都没有）。
  ★★ 同时按**盲区⑥**（`1e300 + 1.0 == 1e300`）检查：本模块**没有 ≥ `2^53` 的浮点字面量**
    （最大的数是 `967566`，是整数；浮点里最大的是 `16.0`）⇒ 每一个叶子都**可变异**。

★ 每条常量带 `# 来源：` + `[A-E]` 级 + 抓取日；跨主题/跨文件同值的另带 `# 同值位置：`。
  分级判据见 `CONTRIBUTING` §2（A 厂商规范 / B 产品官方文档 / C 论文 / D 社区 / E 本库推导）。

★★ **量法口径（本模块逐条照办，取自本域 §0.3 与计划 §4 的 T18/T34/T46）**：
  「计数 N 次」一律写明是**在哪个载体**上数的：本模块对 DCP 那条链**两个数都记**
    （**原始字节** vs **剥标签 + 折叠空白**），因为二者的差距很大（见 §2 的第一组常量）——
  ★★ 这正是 **Y15 / T34 / T46** 的纪律：**裸计数不是证据，必须配对照并写明载体**。
  其余形态逐条注意：① 跨标记的引文在原始字节上必然计不到；② 标点后的空格数会变；
  ③ **引文片段不能含跨行空白**（本模块所有引文都**不含换行**，逐条实测过）；
  ④ 同一条引文在不同页面的计数可能不同（**必须写明是哪个页面**）；
  ⑤ 数字可能带千位分隔符（**本主题的两条**：`31.4GB` 无分隔符、`120 GB` **中间有空格**）；
  ⑥ 标点可能是**弯引号且有两侧空格**（**本模块因此不取任何含引号的片段**）。
"""

# ===========================================================================
# 1. ZeRO（Rajbhandari et al. 2019）—— [C] 论文（**不确认同行评审**）
#    ★★ 判据 ①（优化器状态的字节账按精度算）与判据 ③（`K = 12` 只用于 ZeRO 的举例）
#       的全部依据都在这一节
# ===========================================================================
#
# 来源：[C] S. Rajbhandari, J. Rasley, O. Ruwase, Y. He,
#       **`ZeRO: Memory Optimizations Toward Training Trillion Parameter Models`**，
#       arXiv **1910.02054**。
#       评审状态：**不确认同行评审** —— abs 页的 `Comments:` 字段**实测不存在**
#         （`Comments:` 计 0 次；同页 `ICLR` / `NeurIPS` / `MLSys` 也都计 0 次）。
#       ★ **P21 已办**：abs 页 `<title>` 逐字 =
#         `[1910.02054] ZeRO: Memory Optimizations Toward Training Trillion Parameter Models`。
#       抓取（ar5iv 正文）：`https://ar5iv.labs.arxiv.org/html/1910.02054`，2026-09-23，
#       **279500 B**（`/tmp` 临时取；本模块只登记数，不登记路径）。
#       ★★ **载体的一个坑**：ar5iv 把 `Ψ` 这类符号**渲染成两份**（`4 \Psi+K\Psi` 后面紧跟
#          `4\Psi+K\Psi`）⇒ **凡跨公式的引文都不要取**。本模块取的每一段都**不含公式**、
#          不含换行，且逐条实测过计数。

ZERO_ARXIV_ID = "1910.02054"
# 来源：[C] ZeRO（同上），abs 页 `<title>`（去掉 `[1910.02054] ` 前缀后的标题本体）。
# 口径：★ **P21 的落法**：编号与标题**成对存放**，程序里断言「标题里含 `Memory Optimizations`」
#       这种**结构性**区分（而不是只钉编号）。
# 消费者：`01_state_bytes.py` 的 `D.6`。
# 同值位置：无。

ZERO_TITLE = "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"
# 来源：[C] ZeRO（同上），abs 页 `<title>`，原始字节 1 次。
# 口径：★ 计划把它放在「`optimizers` 与 `100-parallelism` 的边界」上（本域只留**接口**）——
#       本主题**只用它的字节账**那一句，**不写**分片策略与通信量（那是 `100-parallelism/zero-fsdp/`）。
# 消费者：`01_state_bytes.py` 的 `D.6`、`02_sharded.py` 的 `D.7b`。
# 同值位置：`optimizers/code/specs.py`（那一篇引的是 Muon / Adam，不含本条）。

# ── 模型状态的四项（`Ψ` = 参数个数；单位是**字节**，都写成 `×Ψ` 的倍数）────────────
# ★★ 判据 ① 的**唯一来源**：ZeRO §5.1 那一段把它逐项列了出来。
#    逐字（剥标签 + 折叠空白后 1 次；**跨公式的片段不取**）：
#      `Mixed-precision Adam has K = 12`
#      `In total, this results in 2` … `bytes of memory requirement`
#    以及定义处：`i.e., the additional memory required to store them is K`
#      （完整句：`Let's use K to denote the memory multiplier of the optimizer states, i.e.,
#        the additional memory required to store them is KΨ bytes.` —— ★ `Let’s` 是**弯引号**，
#        按 §0.3 的形态 (5) **不取**，只取后半句；★ `KΨ` 里夹了公式标记，故切在 `is K`）。
ZERO_K_OPT_STATE = 12
# 来源：[C] ZeRO（同上），§5.1 逐字 `Mixed-precision Adam has K = 12`。
# 口径：★★ **`K = 12` 是本论文自己点名的举例口径**（`based on mixed-precision training with
#       Adam optimizer.`，见 `ZERO_FIGURE_CALIBER_QUOTE`），**不是四篇论文的共识**（计划 **Z4**）
#       ⇒ 凡引用它，正文必须写「这是 ZeRO 的举例口径」这句（`ZERO_K_CALIBER_NOTE`）。
# 消费者：`01_state_bytes.py` 的 `A.3`/`D.5`；`02_sharded.py` 的 `A.3`（判据 ③）。
# 同值位置：无（`12` 这个数**只在 ZeRO 这一处**有来源）。

ZERO_PARAM_FP16_MULTIPLE = 2.0
# 来源：[C] ZeRO（同上），§5.1：模型状态里「参数」按 **16 位**存的字节数 = `2Ψ`
#       （原文里这一项与 `2Ψ` 的梯度并列，`2 Ψ + 2 Ψ + K Ψ = 16 Ψ`；★ 公式段含标记，不取原文片段，
#        只取它的**项**）。★ 另见 `ZERO_TOTAL_QUOTE` 的两个片段。
# 口径：★ 这个 `2` 是**fp16 的字节数**（`2` 字节/参数），与 `10-foundations/floating-point/01`
#       的位分配一致；本主题**不重推**格式（禁区在 `130-quantization/numerical-formats/01`）。
# 消费者：`01_state_bytes.py` 的 `A.2`/`A.3`。
# 同值位置：`mixed-precision/code/specs.py`（那一篇的 `2` 是 fp16 的**字节数**，同源同值）。

ZERO_GRAD_FP16_MULTIPLE = 2.0
# 来源：[C] ZeRO（同上），§5.1（同上，梯度那一项 `2Ψ`，fp16）。
# 口径：★★★ **本主题最要紧的一处口径**：这一项**只出现在「训练时显存」的账里**，
#       **不出现**在「检查点文件」的账里 —— 梯度下一步就被覆盖，通常不入存档。
#       DCP 的文档里也**没有**任何一句说「检查点含梯度」（那是**推理**，已登记 `UNSOURCED`：
#       `dcp_checkpoint_excludes_gradients`）。
#       ⇒ `01_state_bytes.py` 的 `C.3` 把这**两笔账的差**量出来（差**恰好**是这一项）。
# 消费者：`01_state_bytes.py` 的 `A.3`/`C.3`。
# 同值位置：无。

ZERO_MASTER_WEIGHT_MULTIPLE = 4.0
# 来源：[C] ZeRO（同上），§5.1：优化器状态里的第一项是「参数的 fp32 副本」= `4Ψ`
#       （`4` 字节/参数）。★ 原文那句的**完整片段含公式**（`4Ψ, 4Ψ, and 4Ψ bytes`），
#         按 §0.3 的形态 (1)/(3) **不取**，只取 `optimizer states` 那一句的**非公式**部分：
#         `to denote the memory multiplier of the optimizer states`（1 次）。
# 口径：★★ 这一项是**主权重**（mixed-precision 的 fp32 副本）⇒ 它是 `K = 12` 的**三分之一**，
#       也是 `mixed-precision/01` 讲的「主权重」在**字节账**上的位置。
# 消费者：`01_state_bytes.py` 的 `A.2`/`A.3`。
# 同值位置：无。

ZERO_MOMENTUM_MULTIPLE = 4.0
# 来源：[C] ZeRO（同上），§5.1：优化器状态里的第二项 = 动量 = `4Ψ`（fp32，4 字节/参数）。
# 口径：★★ **判据 ① 的主角**：Adam 的 `m` 与 `v` **各 4Ψ** ⇒ **两者合起来 8Ψ**。
#       ★ 它们是 **fp32**（原文的 `4Ψ` 就是 4 字节/参数 = fp32）——
#         改成 fp16 就是 `2Ψ + 2Ψ = 4Ψ`，**少一半**，而程序**不会报错**（`C.2` 量这件事）。
# 消费者：`01_state_bytes.py` 的 `A.1`（判据 ①）/`C.2`。
# 同值位置：`optimizers/code/specs.py`（那一篇有 Adam 的 `m`/`v` 更新式，但没有字节倍数）。

ZERO_VARIANCE_MULTIPLE = 4.0
# 来源：[C] ZeRO（同上），§5.1：优化器状态里的第三项 = 方差（Adam 的 `v`）= `4Ψ`。
# 口径：★ 与 `ZERO_MOMENTUM_MULTIPLE` **同值但不同项**（`m` 与 `v`）⇒
#       **两条常量分开存放、不得合并**：合并会抹掉「`m` + `v` = 8Ψ」这个加法本身
#       （而判据 ① 量的正是这个加法）。
# 消费者：`01_state_bytes.py` 的 `A.1`/`C.2`。
# 同值位置：见上。

ZERO_TOTAL_MULTIPLE = 16.0
# 来源：[C] ZeRO（同上），§5.1 逐字 `In total, this results in 2` … `bytes of memory requirement`
#       （两个片段各 1 次；★ 中间夹着公式 `2Ψ+2Ψ+KΨ=16Ψ`，按 §0.3 **不取**整句）。
# 口径：★★★ **这是「训练时**显存**」的口径，不是「检查点文件字节」的口径** ——
#       原文用的词是 `memory requirement`，Figure 1 的标题是
#       `Comparing the per-device memory consumption of model states`。
#       ⇒ 把 `16Ψ` 当检查点大小是**跨口径套用**；两者差的**恰好是梯度那一项**
#         （`2Ψ`，`01_state_bytes.py` 的 `C.3` 断言这个恒等式）。
# 消费者：`01_state_bytes.py` 的 `A.3`/`C.3`。
# 同值位置：无。

ZERO_K12_QUOTE = "Mixed-precision Adam has K = 12"
# 来源：[C] ZeRO（同上），§5.1，**原始字节 1 次 / 剥标签后 1 次**。
# 口径：★ 这是 `K = 12` 的**逐字证据**（`ZERO_K_OPT_STATE` 的载体）。
# 消费者：`02_sharded.py` 的 `A.3`（判据 ③）。
# 同值位置：无。

ZERO_K_DEFINITION_QUOTE = "to denote the memory multiplier of the optimizer states"
# 来源：[C] ZeRO（同上），§5.1，**原始字节 1 次 / 剥标签后 1 次**。
# 口径：★ `K` 的**定义**（「优化器状态的字节倍数」）⇒ 它把 `K` 限定在**优化器状态**上，
#       **不含**参数与梯度 ⇒ 这是 `C.3` 那条「两笔账差 2Ψ」能算清的前提。
# 消费者：`02_sharded.py` 的 `A.3`。
# 同值位置：无。

ZERO_ADDITIONAL_QUOTE = "i.e., the additional memory required to store them is K"
# 来源：[C] ZeRO（同上），§5.1，**原始字节 1 次 / 剥标签后 1 次**
#       （★ 切在 `is K` 处：后面紧跟公式 `K\Psi`，含标记 ⇒ 按形态 (1) 不取）。
# 口径：★ 「**额外**需要的字节」 ⇒ 优化器状态是**加在**参数之上的，不是替代它们。
# 消费者：`02_sharded.py` 的 `A.3`。
# 同值位置：无。

ZERO_TOTAL_QUOTE = "In total, this results in 2"
# 来源：[C] ZeRO（同上），§5.1，**原始字节 1 次 / 剥标签后 1 次**。
# 口径：★ 与 `ZERO_TOTAL_TAIL_QUOTE` **成对读**：中间那一段是公式（含标记）⇒
#       本模块取头取尾、**不取中间**（这是 §0.3 形态 (1)/(3) 的标准做法）。
# 消费者：`02_sharded.py` 的 `A.3`。
# 同值位置：无。

ZERO_TOTAL_TAIL_QUOTE = "bytes of memory requirement"
# 来源：[C] ZeRO（同上），§5.1，**原始字节 1 次 / 剥标签后 1 次**。
# 口径：★★ 结尾是 `memory requirement`（**显存/内存**），不是 `checkpoint size` ⇒
#       这一句本身就把口径钉住了（`C.3` 的依据）。
# 消费者：`01_state_bytes.py` 的 `C.3`、`02_sharded.py` 的 `A.3`。
# 同值位置：无。

ZERO_FIGURE_CALIBER_QUOTE = "based on mixed-precision training with Adam optimizer."
# 来源：[C] ZeRO（同上），**Figure 1 的说明文字**，**原始字节 1 次 / 剥标签后 1 次**。
# 口径：★★ `K = 12` 的**口径来源**（`K` 只在 mixed-precision + Adam 这一组假设下是 12）
#       ⇒ 计划 **Z4** 要求的「必须注明这是 ZeRO 的举例口径」的**逐字依据**。
# 消费者：`02_sharded.py` 的 `A.3`、`D.7b`。
# 同值位置：无。

ZERO_FIGURE_MODEL_QUOTE = "denotes model size (number of parameters), K"
# 来源：[C] ZeRO（同上），**Figure 1 的说明文字**，**原始字节 1 次 / 剥标签后 1 次**。
# 口径：★ 它把 `Ψ` 定义成「**参数个数**」 ⇒ 本主题所有 `×Ψ` 的账都按**参数个数**计，
#       不是按「参数张量数」或「元素字节数」。★ `K` 出现在片段末尾（后接公式标记，已切掉）。
# 消费者：`02_sharded.py` 的 `A.3`。
# 同值位置：无。

ZERO_K_CALIBER_NOTE = "ZeRO 的举例口径"
# 来源：★ **本库的措辞常量（E 级）**，不是来源里的句子 —— 它是对 `ZERO_FIGURE_CALIBER_QUOTE`
#       的中文概括，作用是**让正文里那句限定语变成一个可断言的字面量**（计划 Z4 / 判据 ③）。
# 口径：★★ 配套文档 `01-state-and-bytes.md` 与 `02-sharded-save-and-load.md` 的正文里
#       **必须逐字出现这个字符串**（判据 ③ 的可执行落法：`02_sharded.py` 会去读它的配套文档）。
# 消费者：`02_sharded.py` 的 `A.3`/`D.7c`。
# 同值位置：`autograd-and-backward/code/specs.py` 无；本主题的两篇文档各出现 ≥ 1 次。

ZERO_OPT_4X_QUOTE = "4x memory reduction, same communication volume as DP"
# 来源：[C] ZeRO（同上），§5.1 列表项 `1) Optimizer State Partitioning (P_os)` 之后的逐字，
#       **原始字节 1 次 / 剥标签后 1 次**。
# 口径：★★★ **`4x` 属于 `P_os`（只分片优化器状态），`8x` 属于 `P_os+g`**（计划 **Y14**）——
#       把 `8x` 记成 `P_os` 的收益是**常见转述**里的错。本模块**两条分开存放**。
#       ★ 且 `4x` 的成立条件是 `N_d` 足够大（见 `ZERO_EXAMPLE_31_4_QUOTE` 那一组）。
# 消费者：`02_sharded.py` 的 `D.7b`。
# 同值位置：无。

ZERO_GRAD_8X_QUOTE = "8x memory reduction, same communication volume as DP"
# 来源：[C] ZeRO（同上），§5.1 列表项 `2) Add Gradient Partitioning (P_os+g)` 之后的逐字。
# 口径：★ 见 `ZERO_OPT_4X_QUOTE`：**两者不是同一个阶段的收益**。
# 消费者：`02_sharded.py` 的 `D.7b`。
# 同值位置：无。

ZERO_EXAMPLE_31_4_QUOTE = "parameter model requires 31.4GB of memory using P"
# 来源：[C] ZeRO（同上），§5.1，**原始字节 1 次 / 剥标签后 1 次**（切在公式标记处）。
# 口径：★ 这是 `4x` 那条的**具体例子**（7.5B 模型、`P_os`、64-way DP）。
#       ⚠ 数字写法：原文是 `31.4GB`（**无空格、无千位分隔符**）；
#         下一条 `120 GB` **中间有一个空格**（形态 (5) 的近亲，已逐字保留空格）。
# 消费者：`02_sharded.py` 的 `D.7b`。
# 同值位置：无。

ZERO_EXAMPLE_120_QUOTE = "while requiring 120 GB with standard DP."
# 来源：[C] ZeRO（同上），§5.1，**原始字节 3 次 / 剥标签后 3 次**
#       （三处小节都用了同一句：`P_os` / `P_os+g` / `P_os+g+p`）。
# 口径：⚠ **3 次**不是不稳健，是**同一句在三处复用** —— 本模块登记它时**把次数写明**，
#       免得后来人看到「3」以为取错了片段（形态 (4) 的近亲：数的**载体位置**要写明）。
# 消费者：`02_sharded.py` 的 `D.7b`。
# 同值位置：无。


# ===========================================================================
# 2. `torch.distributed.checkpoint`（DCP）—— [B] PyTorch 2.14 的文档
#    ★★ 判据 ②（分片的元数据开销）与 `02`/`03` 的工程事实全在这一节
#    ★★★ 这一节是**某实现的文档**，不是规范（**P19**）⇒ 每条都写明版本 + 页面
# ===========================================================================
#
# 来源：[B] **PyTorch 2.14** 的 `Distributed Checkpoint` 页，
#       `https://docs.pytorch.org/docs/2.14/distributed.checkpoint.html`，
#       抓取日 **2026-09-23**，**967566 B**，`sha256[:16] = 1e78400bd3be2148`
#       （★ 与计划 §C.2b 的 **T3** 逐位一致 ⇒ 同一个载体）。
#       ★ 页头自述：`Created On: Nov 16, 2022 | Last Updated On: Jul 08, 2026`。
#
# ★★★ **本主题第一组「两个数」的常量（量法口径的现场）**：
#   计划 §C.2b 记的是**原始字节**上的计数；本模块把**两个载体**都记下来：
#     `torch.distributed.checkpoint`：原始 **547** / 剥标签 + 折叠空白 **71**
#     `StateDict`                     ：原始 **48**  / 剥标签 + 折叠空白 **17**
#     `sharded`                       ：原始 **30**  / 剥标签 + 折叠空白 **14**
#   ⇒ ★★ **两个数都复现了**（原始那一列与计划逐位一致），但**它们差 ~7 倍** ——
#      原因是这个页面的**侧边导航栏**把 `torch.distributed.checkpoint` 之类的词重复了几百次。
#      ⇒ **引用计数时必须写明载体**（Y15 / T34 的又一次现场；而且这一次的差距是**数量级**的）。
DCP_DOC_URL = "https://docs.pytorch.org/docs/2.14/distributed.checkpoint.html"
# 来源：[B]（同上）。消费者：`02_sharded.py` 的 `D.7`。
# 同值位置：无。

DCP_DOC_BYTES = 967566
# 来源：[B]（同上，本地字节数）。★ 整数，**不涉及盲区⑥**。
# 消费者：`02_sharded.py` 的 `D.7`。
# 同值位置：无。

DCP_DOC_SHA256_16 = "1e78400bd3be2148"
# 来源：[B]（同上，`sha256` 的前 16 个十六进制字符）。★ 它是**载体同一性**的证据
#       （计划 §C.2b 的 T3 记的就是这个值）⇒ 三处（计划 / 本模块 / 文档正文）互钉。
# 消费者：`02_sharded.py` 的 `D.7`。
# 同值位置：无。

DCP_COUNT_RAW = 547
# 来源：[B]（同上），**原始字节**上 `torch.distributed.checkpoint` 出现 **547** 次。
# 口径：★★ **这是计划 §C.2b 记的那个数**（逐位一致）；★ 它**不是**正文里这个词的出现次数
#       （正文是 71 次）—— **载体不同，数就差 ~7 倍**。两个数都留着，谁都不删。
# 消费者：`02_sharded.py` 的 `D.7`（判据：`raw != stripped` 且 `raw > 5 × stripped`）。
# 同值位置：计划 §C.2b 的 T3 行。

DCP_COUNT_STRIPPED = 71
# 来源：[B]（同上），**剥标签 + 折叠空白后**同一串出现 **71** 次。
# 口径：★★ 本模块把「**剥标签后**」这个口径显式做成了一个常量 —— 因为**它就是结论本身**
#       （载体换一下，数差 7.7 倍）。
# 消费者：`02_sharded.py` 的 `D.7`。
# 同值位置：无。

DCP_STATE_DICT_COUNT_RAW = 48
# 来源：[B]（同上），**原始字节**上 `StateDict` 出现 **48** 次（计划 §C.2b 记的数，逐位一致）。
# 消费者：`02_sharded.py` 的 `D.7`。
# 同值位置：无。

DCP_STATE_DICT_COUNT_STRIPPED = 17
# 来源：[B]（同上），**剥标签 + 折叠空白后**出现 **17** 次。
# 消费者：`02_sharded.py` 的 `D.7`。
# 同值位置：无。

DCP_SHARDED_COUNT_RAW = 30
# 来源：[B]（同上），**原始字节**上 `sharded` 出现 **30** 次（计划 §C.2b 记的数，逐位一致）。
# 消费者：`02_sharded.py` 的 `D.7`。
# 同值位置：无。

DCP_SHARDED_COUNT_STRIPPED = 14
# 来源：[B]（同上），**剥标签 + 折叠空白后**出现 **14** 次。
# 消费者：`02_sharded.py` 的 `D.7`。
# 同值位置：无。

DCP_MULTIPLE_FILES_QUOTE = "It produces multiple files per checkpoint, with at least one per rank."
# 来源：[B]（同上），正文首段，**原始字节 2 次 / 剥标签后 1 次**（★ 载体差 2 倍的原因：
#       该段在页面里被渲染了两遍 —— 本模块**两个数都写**，判定用**剥标签**那个）。
# 口径：★★ 这是判据 ② 的**定性依据**：「一片一个文件（至少）」⇒ 分片**增加文件数**，
#       而每个文件都要**自己的头**（见 `SHARD_HEADER_BYTES`）。
# 消费者：`02_sharded.py` 的 `B.1`（判据 ②）。
# 同值位置：无。

DCP_IN_PLACE_QUOTE = "It operates in place, meaning that the model should allocate its data first and DCP uses that storage instead."
# 来源：[B]（同上），正文首段，**原始字节 2 次 / 剥标签后 1 次**。
# 口径：★★ 「**就地恢复**」⇒ 加载前**必须先分配**，且**中途失败会留下半新半旧的状态**。
#       ★ 后半句是本库的**推理**（文档没有写「失败会留下什么」）⇒ 已登记 `UNSOURCED`：
#         `dcp_partial_load_state`。程序里只钉**逐字**那一句。
# 消费者：`03_consistency.py` 的 `C.4`（判据 ④ 的边界面）、`02_sharded.py` 的 `D.7`。
# 同值位置：无。

DCP_RESHARDING_QUOTE = "It handles load-time resharding which enables saving in one cluster topology and loading into another."
# 来源：[B]（同上），正文首段，**原始字节 2 次 / 剥标签后 1 次**。
# 口径：★★ 判据 ② 的**功能依据**：**存的时候是 N 片、读的时候可以是 M 片**（`M ≠ N`）
#       ⇒ 分片的**边界不是格式的一部分**，**元数据才是**（读侧要能重算映射）。
# 消费者：`02_sharded.py` 的 `A.4`（重分片：写 3 片读 7 片）。
# 同值位置：无。

DCP_LOCAL_SHARDS_QUOTE = "having each rank only save their local shards"
# 来源：[B]（同上），`save()` 的说明，**原始字节 2 次 / 剥标签后 1 次**。
# 口径：★★ 这是「**省的是单卡峰值**，不是总字节」的逐字依据（`02_sharded.py` 的 `B.2`）。
# 消费者：`02_sharded.py` 的 `B.2`。
# 同值位置：无。

DCP_LEAST_DATA_QUOTE = "read the least amount of data necessary"
# 来源：[B]（同上），`load()` 的说明，**原始字节 2 次 / 剥标签后 1 次**。
# 口径：★★ 与上一条**成对**：写侧只写本卡的分片、读侧只读**够用**的那部分 ⇒
#       每卡的读写量 `≈ W/N` 而不是 `W`（`02_sharded.py` 的 `B.3` 把这件事量出来）。
# 消费者：`02_sharded.py` 的 `B.3`。
# 同值位置：无。

DCP_NO_BACKWARD_COMPAT_QUOTE = "There are no guarantees of Backwards Compatibility across PyTorch versions"
# 来源：[B]（同上），`save()` 的 `Warning`，**原始字节 2 次 / 剥标签后 1 次**。
# 口径：★★ 检查点的**跨版本**可用性**没有保证** ⇒ 「存下来的东西能读回来」只在**同一版本**内成立。
#       ★ 这一条支撑 `02` 的设计选择：**头部里必须有格式版本字段**（本程序的设计常量）。
# 消费者：`02_sharded.py` 的 `B.4`。
# 同值位置：无。

DCP_SAME_KEYS_QUOTE = "Mismatched keys may result in hangs or errors."
# 来源：[B]（同上），`load()` 的说明，**原始字节 2 次 / 剥标签后 1 次**。
# 口径：★★ 失败形态是「**hang 或 error**」—— **hang 是更坏的那一种**（没有异常、没有栈）。
#       前置句逐字：`Each rank must have the same keys in their state_dict provided to this API.`
# 消费者：`03_consistency.py` 的 `C.3`。
# 同值位置：无。

DCP_NON_TENSOR_QUOTE = "All non-tensor data is loaded using torch.load()"
# 来源：[B]（同上），`load()` 的 `Warning`，**原始字节 1 次 / 剥标签后 1 次**。
# 口径：★ **非张量**（`step` 计数、调度器状态、RNG 状态那类）走的是**另一条路**（pickle 侧）
#       ⇒ 「张量按分片读、非张量整体反序列化」是**两种代价**（`01_state_bytes.py` 的 `B.3`）。
# 消费者：`01_state_bytes.py` 的 `B.3`。
# 同值位置：无。

DCP_METADATA_MARKS_SUCCESS_QUOTE = "Write the metadata and marks the current checkpoint as successful."
# 来源：[B]（同上），`StorageWriter.finish()` 的说明，**原始字节 2 次 / 剥标签后 1 次**。
# 口径：★★★ 判据 ④ 的**元数据语义**：**元数据写下去 == 这份检查点算成功** ⇒
#       「目录里有没有元数据」就是「这份检查点完整不完整」的**唯一标记**。
#       ⇒ 恢复前必须**校验元数据**；只有分片文件、没有元数据的目录是一个**未完成的检查点**。
#       ★ 文档同时给了 `finish()` 的**次序**（写侧：`set_up_storage_writer` → `prepare_local_plan`
#         → `prepare_global_plan` → `write_data` → `finish`）⇒ 元数据**最后**写。
# 消费者：`03_consistency.py` 的 `C.4`、`02_sharded.py` 的 `D.7c`。
# 同值位置：无。

DCP_ATOMIC_CREATION_QUOTE = "File creation is atomic"
# 来源：[B]（同上），`FileSystemWriter` 的说明列表，**原始字节 2 次 / 剥标签后 1 次**。
# 口径：★★ 原子的粒度是**文件**，**不是目录** ⇒ 「所有分片都写完了」这件事**没有**由
#       文件系统保证，只能靠 `finish()` 写元数据来标记（上一条）。
#       ★ 同一列表里还有 `The checkpoint path is an empty or non-existing directory.`
#         （`DCP_EMPTY_DIR_QUOTE`）—— **写入侧不覆盖已有检查点**（`overwrite` 是构造参数）。
# 消费者：`02_sharded.py` 的 `C.3`。
# 同值位置：无。

DCP_EMPTY_DIR_QUOTE = "The checkpoint path is an empty or non-existing directory."
# 来源：[B]（同上），`FileSystemWriter` 的说明列表，**原始字节 2 次 / 剥标签后 1 次**。
# 口径：★ 它解释了为什么「**先写临时目录、成功了再改名**」是这套接口的常规用法
#       （本程序按这个用法实现 `atomic_install`）。
# 消费者：`02_sharded.py` 的 `C.3`。
# 同值位置：无。

DCP_GLOBAL_METADATA_QUOTE = "a global .metadata file with the serialized metadata if rank coordination is enabled."
# 来源：[B]（同上），`FileSystemWriter` 的说明列表，**原始字节 1 次 / 剥标签后 1 次**。
# 口径：★★★ **判据 ② 的逐字依据**：`一个全局 .metadata 文件` ——
#       即「每片一个文件」之外**还有一个**全局元数据文件 ⇒ 总文件数 = `N + 1`。
# 消费者：`02_sharded.py` 的 `A.5`/`B.1`（判据 ②）。
# 同值位置：无。

DCP_RANK_LOCAL_METADATA_QUOTE = "a rank local __{rank}.metadata file with the serialized metadata if rank coordination is NOT enabled."
# 来源：[B]（同上），`FileSystemWriter` 的说明列表，**原始字节 1 次 / 剥标签后 1 次**。
# 口径：★★★ **没有全局协调时，元数据是【每片一份】** ⇒ 这一支下总文件数 = `2N`、
#       且**每片都有自己的元数据头**（判据 ② 的 `N × 每片头开销` 在**这一支**上是字面的）。
#       ★ 本程序的玩具格式走的就是这一支（没有协调者）⇒ 头开销按 `N` 计。
# 消费者：`02_sharded.py` 的 `B.1`（判据 ②）。
# 同值位置：无。

DCP_ALLOW_PARTIAL_LOAD_QUOTE = "allow_partial_load: If False, will raise a runtime error if a key is present in state_dict, but not in the checkpoint."
# 来源：[B]（同上），`DefaultLoadPlanner` 的说明，**原始字节 2 次 / 剥标签后 1 次**。
# 口径：★★ 缺键的**默认行为是报错**（`allow_partial_load` 默认 `False`）⇒
#       「存档少了一项」**不会**被静默地用默认值顶上 —— ★ 但**只有**在 `state_dict`
#       里**声明了**那个键时才检查得到（`state_dict` 里没有的键，检查器看不见）。
#       ⇒ `03_consistency.py` 的 `C.2` 把**两个方向**都量出来。
# 消费者：`03_consistency.py` 的 `C.2`。
# 同值位置：无。

DCP_OPTIMIZER_ORDER_QUOTE = "WARN: set_optimizer_state_dict can only be called before backward()"
# 来源：[B]（同上），`set_optimizer_state_dict` 的说明，**原始字节 2 次 / 剥标签后 1 次**
#       （切在 `backward()` 处：后半句 `or after step() is called on the optimizers.`
#        在原始字节上计 **2** 次、剥标签后 **2** 次 —— 它在页面里出现在**两段**说明里）。
# 口径：★★ **恢复优化器状态有时序约束**：只能在 `backward()` 之前、或 `step()` 之后调。
#       原文紧跟着给的理由是 `Otherwise, the optimizer states won’t be initialized correctly.`
#       （★ `won’t` 是**弯引号** ⇒ 按形态 (6) **不取**那半句，只取前半句）。
#       ★★ 本主题**不重推**这条时序（它是**框架内部**的调用契约，本机没有 torch、复现不了）
#         ⇒ 它只作**引用判据**（`03_consistency.py` 的 `C.3` 逐字钉住），
#         **不作数值判据**（**P3**：不要求实现者复现跑不了的东西）。
# 消费者：`03_consistency.py` 的 `C.3`。
# 同值位置：无。


# ===========================================================================
# 3. PaLM（Chowdhery et al. 2022）—— [C] 论文（**不确认同行评审**）
#    ★★ 判据 ④（恢复后与原轨迹逐位一致）的**外部依据**在这一节
# ===========================================================================
#
# 来源：[C] A. Chowdhery et al., **`PaLM: Scaling Language Modeling with Pathways`**，
#       arXiv **2204.02311**。评审状态：**不确认同行评审**（abs 页 `Comments:` 计 0 次）。
#       ★ **P21 已办**：abs 页 `<title>` 逐字 = `[2204.02311] PaLM: Scaling Language Modeling with Pathways`。
#       抓取（ar5iv 正文）：`https://ar5iv.labs.arxiv.org/html/2204.02311`，2026-09-23，
#       **1372842 B**（同值位置：`initialization-and-stability/code/specs.py` 的 `PALM_*`）。

PALM_ARXIV_ID = "2204.02311"
# 来源：[C] PaLM（同上），abs 页 `<title>`。
# 口径：★ 与 `initialization-and-stability/code/specs.py` 的 `PALM_ARXIV_ID` **同值同源**，
#       两个主题**各存一份、靠 `# 同值位置：` 互指**（`CONTRIBUTING:36`：`code/` 必须自包含）。
# 消费者：`03_consistency.py` 的 `D.8`。
# 同值位置：`initialization-and-stability/code/specs.py`。

PALM_TITLE = "PaLM: Scaling Language Modeling with Pathways"
# 来源：[C] PaLM（同上），abs 页 `<title>`，原始字节 1 次。
# 消费者：`03_consistency.py` 的 `D.8`。
# 同值位置：`initialization-and-stability/code/specs.py`。

PALM_BITWISE_SECTION_QUOTE = "Bitwise determinism"
# 来源：[C] PaLM（同上），§5「Training Setup / 训练配置」的列表项名，
#       **原始字节 1 次 / 剥标签后 1 次**。
# 口径：★★★ **判据 ④ 的逐字来源**：PaLM **明说自己做到了这件事**（不是本库的推理）。
# 消费者：`03_consistency.py` 的 `A.6`。
# 同值位置：无（另一篇的同类说法没有逐字来源）。

PALM_BITWISE_QUOTE = "fully bitwise reproducible from any checkpoint"
# 来源：[C] PaLM（同上），§5 该列表项的第一句，**原始字节 1 次 / 剥标签后 1 次**。
# 口径：★★★ 完整句逐字 = `The model is fully bitwise reproducible from any checkpoint.`
#       ⇒ **这是一个可对照的目标**：本主题的判据 ④ 要复现的正是这句话的**玩具版**
#       （本机没有 TPU/JAX，量的是一维玩具模型的**参数轨迹逐位一致**）。
# 消费者：`03_consistency.py` 的 `A.6`。
# 同值位置：无。

PALM_RUN_PAIR_QUOTE = "if the model has been trained up to step 17,000 in a single run"
# 来源：[C] PaLM（同上），§5 该列表项，**原始字节 1 次 / 剥标签后 1 次**。
# 口径：★★ 完整句逐字 = `In other words, if the model has been trained up to step 17,000 in a
#       single run, and we restart from checkpoint 15,000, then the training framework is
#       guaranteed to produce identical results in both runs from checkpoint 15,000 to 17,000.`
#       ⇒ ★★ **这正是判据 ④ 的实验设计**（一个**一口气跑完**的参照 + 一个**从存档点重启**的
#         对照，比较**重启点之后**的每一步）—— 本主题的实现是**照这句话的形状**做的，
#         包括 `SAVE_AT`（对应 15,000）与 `STEPS`（对应 17,000）的**同构**。
# 消费者：`03_consistency.py` 的 `A.6`。
# 同值位置：无。

PALM_DATASET_RA_QUOTE = "a deterministic dataset pipeline where the shuffled data is written out in a random-access format"
# 来源：[C] PaLM（同上），§5 该列表项（`This is achieved in two ways: (1) … and (2) …` 的**第二支**），
#       **原始字节 1 次 / 剥标签后 1 次**。
# 口径：★★★ **P17 的正面答案**：PaLM 自己把「数据顺序的一致性」做成了**数据集侧的材料化**
#       （随机访问格式），而**不是**「检查点里存一个 RNG 状态」。
#       ⇒ `03_consistency.py` 的**第三条路径（重新 `seed()` + 快进）** 在工程上**不是错的** ——
#         它就是 PaLM 的第二条做法的玩具版；★ 但它**不检查存档是否完整**
#         （这正是本主题必须把「只允许从文件读」写成硬约束的原因）。
# 消费者：`03_consistency.py` 的 `A.5`。
# 同值位置：无。

PALM_DATASET_STEP_QUOTE = "the contents of a given training batch is only a function of the step number"
# 来源：[C] PaLM（同上），§5 该列表项（第二支的后半），**原始字节 1 次 / 剥标签后 1 次**。
# 口径：★★★ **「随机性被重写成步号的函数」的逐字表述** ⇒ 第三条路径成立的**前提**
#       就是这一句（步号唯一决定内容）；一旦消耗模式依赖**数据/时序**，快进就不再等价。
# 消费者：`03_consistency.py` 的 `A.5`。
# 同值位置：无。


# ===========================================================================
# 4. Adam 的默认超参 —— [C] 1412.6980（**已确认 ICLR 2015**）
#    ★ 本主题**只消费、不重定义**（`03_consistency.py` 的玩具优化器要用它）
# ===========================================================================
#
# 来源：[C] D. P. Kingma, J. Ba, **`Adam: A Method for Stochastic Optimization`**，
#       arXiv **1412.6980**（**已确认 ICLR 2015**）。
#       默认超参 `α = 0.001, β1 = 0.9, β2 = 0.999, ε = 10^−8`。
#       ★ **一处方法学让步**（计划 **T5**）：**该文的 ar5iv 页只有 226 字符**
#         （自述无 LaTeX 源可转）⇒ 逐字只能取自 **PDF 文本层**；
#         ★ 且 PDF 文本层里 `ε` 是**控制字符** ⇒ 只能取公式**之前**的片段（计划 **T45**）。
#       ⇒ 本模块登记的是**数值**（四个默认值），不是那句引文本身 ——
#         引文与它的处理在同值位置的 `optimizers/code/specs.py` 里。
ADAM_DEFAULT_BETA1 = 0.9
# 来源：[C] 1412.6980（同上），Algorithm 1 的 `Good default settings`（**PDF 文本层**）。
# 消费者：`03_consistency.py` 的 `C.1`（玩具优化器）。
# 同值位置：`optimizers/code/specs.py`、`initialization-and-stability/code/specs.py`。

ADAM_DEFAULT_BETA2 = 0.999
# 来源：[C] 1412.6980（同上）。消费者：同上。同值位置：同上。

ADAM_DEFAULT_EPS = 1e-08
# 来源：[C] 1412.6980（同上）。★ `ε` 在**根号外**（计划 §C.2 的逐字式）。
# 消费者：`03_consistency.py` 的 `C.1`。同值位置：同上。


# ===========================================================================
# 5. 本程序的设计常量（**E 级：本库设计**）—— 判据 ② 的 `HEADER` 就在这里
#    ★★★ 这一节**没有任何外部来源**：它是本玩具格式的**设计决定**。
#        写在一节里、每条都标 E 级，正是为了**不把它混进上面三节的 [B]/[C]**（**P19** 的反面）。
# ===========================================================================

SHARD_HEADER_BYTES = 64
# 来源：★ **本程序的设计常量（E 级）** —— **不是**任何来源里的数。
#       它存在的**理由**由 [B] 的两条逐字给出（都在 `02_sharded.py` 的 `A.5`/`B.1` 里钉住）：
#         · `DCP_MULTIPLE_FILES_QUOTE`：一片一个文件（至少）；
#         · `DCP_GLOBAL_METADATA_QUOTE` / `DCP_RANK_LOCAL_METADATA_QUOTE`：每片（或全局）都要元数据。
#       ⇒ 「每个分片文件有一个**固定长度的头**」这件事是**接口的必然**，
#         而**头有多少字节是设计**（DCP 的文档**不给**这个数，已登记 `UNSOURCED`）。
# 口径：★★★ 它是判据 ② 的被测常量：`02_sharded.py` 的 `B.1` 断言
#       「实测总字节 − 单卡字节 == `N × SHARD_HEADER_BYTES`」——
#       ★ 左边是**量出来的**、右边是**这个常量**，两边**来自不同的地方** ⇒ 不是恒真。
# 消费者：`02_sharded.py` 的 `A.5`/`B.1`（判据 ②）、`C.3`。
# 同值位置：无。

SHARD_HEADER_TEXT_BYTES = 57
# 来源：★ **本程序的设计常量（E 级）**。头的**文本部分**逐字是这个形状（`|` 是分隔符）：
#         `RRRRRRRR|VVVVVVVV|SSSSSSSS|LLLLLLLL|TTTTTTTT|DDDDDDDDDDDD`
#       即 **5 个 8 位十进制字段（rank / version / start / length / total）+ 5 个分隔符**
#       `= 5 × 9 = 45`，**加上 12 位十六进制摘要** `= 57`。
#       ★ 它是**定长**的（后面补 `SHARD_HEADER_PAD_BYTES` 个 `.`）：定长才让
#         「按固定偏移定位头」这件事可做；变长头会让读侧先解析再定位、多一次读。
#       ★ 这个算式由 `02_sharded.py` 的 `C.4b` 断言
#         （`SHARD_HEADER_TEXT_BYTES + SHARD_HEADER_PAD_BYTES == SHARD_HEADER_BYTES`，
#          且**实测**的头文本长度 == 这个常量）。
# 消费者：`02_sharded.py` 的 `C.4b`。
# 同值位置：无。

SHARD_HEADER_PAD_BYTES = 7
# 来源：★ **本程序的设计常量（E 级）**：填充字节数 = `SHARD_HEADER_BYTES − SHARD_HEADER_TEXT_BYTES`
#       = `64 − 57`。★ 填充用 `.`（ASCII 46）。
# 消费者：`02_sharded.py` 的 `C.4b`。
# 同值位置：无。

FRAME_VERSION = 3
# 来源：★ **本程序的设计常量（E 级）**：玩具分片格式的版本号，写在头里。
#       ★ 理由由 [B] 的逐字给出：`DCP_NO_BACKWARD_COMPAT_QUOTE`
#         （「跨 PyTorch 版本没有向后兼容保证」）⇒ 既然没有保证，**格式必须自报版本**，
#         读侧才能**拒绝**而不是**猜**。`02_sharded.py` 的 `B.4` 用两个版本量这件事。
# 消费者：`02_sharded.py` 的 `B.4`/`C.4`。
# 同值位置：无。

UNSOURCED = (
    "dcp_per_shard_header_bytes",
    "dcp_metadata_file_format",
    "dcp_partial_load_state",
    "dcp_checkpoint_excludes_gradients",
    "palm_pipeline_materialized_bytes",
    "numpy_pcg64_state_json_schema",
    "zero_16psi_checkpoint_caliber",
)
# 来源：★ **本库的登记（E 级）**：这七件事**没有**可引的一手来源 ⇒ 正文里不得写成事实。
# 逐条：
#   1. `dcp_per_shard_header_bytes` —— DCP 的文档**不给**每片头部的字节数
#      （`SHARD_HEADER_BYTES` 因此是 E 级设计常量）；
#   2. `dcp_metadata_file_format` —— 文档逐字说它是 `an implementation detail`
#      （`The actual format/schema used for serializing metadata is an implementation detail.`）；
#   3. `dcp_partial_load_state` —— 「就地恢复中途失败会留下半新半旧的状态」是**推理**，
#      文档只说「就地」与「原子的粒度是文件」；
#   4. `dcp_checkpoint_excludes_gradients` —— 没有任何一句说「梯度不入检查点」；
#      那是由 `save will call state_dict before serialization` **推**出来的；
#   5. `palm_pipeline_materialized_bytes` —— PaLM 只说数据写成 `random-access format`，
#      **没给字节数** ⇒ `03_consistency.py` 的「数据集侧 8N 字节」是**本程序的推导值**；
#   6. `numpy_pcg64_state_json_schema` —— `bit_generator.state` 是**四个字段**（实测），
#      NumPy 没有为这个字典结构给出 schema ⇒ 01 的 `C.1` 只钉**实测行为**；
#   7. `zero_16psi_checkpoint_caliber` —— `16Ψ` 是**训练显存**口径（原文 `memory requirement`），
#      **没有**一句把它和「检查点文件字节」对上 ⇒ 两笔账的差 `2Ψ` 是本程序的**推导**。
# 消费者：`01_state_bytes.py` 的 `D.9`、`02_sharded.py` 的 `D.9`、`03_consistency.py` 的 `D.9`
#         （三个程序各逐位钉一次 ⇒ 与本模块保持同步）。
# 同值位置：无。

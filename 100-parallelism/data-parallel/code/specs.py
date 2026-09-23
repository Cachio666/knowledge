"""100-parallelism/data-parallel 的外部事实常量源。

本模块是 `CONTRIBUTING` §2 约定的**共享事实源**（`specs.py`）：本主题的编号程序从它读常量，
推导过程可复核，**常量本身不可自证**。

★ **只放本主题真的会断言的常量**：没有消费者的常量就是 `make mutate` 的漏网候选
  （`80` 实测「只用于打印的模块级常量表」制造过 2 条漏网）⇒ **要么给它消费者、要么不写**。
  ★ 本模块的每一条常量都在**至少一个编号程序**的 `part_D` 里被 `==` 逐位钉住（消费位置逐条写在下面）。

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
    · **幂次一律写成十进制字面量**（`2^-25` 写作 `2.9802322387695312e-08`）——
      `mutate.py` 的 `_walk()` **不下钻 `BinOp`** ⇒ `2.0 ** -6` 整体**不进变异分母**。
      本模块**一处幂次都没有**（`16Ψ`、`4Ψ` 全是十进制字面量）。

★★ **不设跨主题共享模块**（`CONTRIBUTING:36`「`code/` 必须自包含」是**全库规则**）：
  跨主题同值的常量**各写一份**，靠 `# 同值位置：` 互指。

★ **本主题的禁区**（细节见 `requirements.txt` 与域 README 的「本域不做什么」表）：
    **梯度规约的算法**（ring allreduce / `V(N)`）与**扩展效率模型**（Amdahl / `N₅₀`）
      ⇒ 回链 `110-communication` 与 `10-foundations/performance-models/02-amdahl-and-scaling.md`。
  ★★ 因此本模块**不含**任何 `V(N)` / 带宽 / 延迟常量 —— 那类常量一条都不许出现在这里
    （登记在 `UNSOURCED`：`data-parallel_ring_allreduce_volume`）。
"""

# ===========================================================================
# 1. ZeRO（Rajbhandari et al. 2019）—— [C] 论文（**不确认同行评审**）
#    ★★ 本节只取**模型状态的字节账**与 `Ψ` 的定义；`16Ψ` 的**组成与分解**
#       已经由 `90-training/checkpointing/01-state-and-bytes.md` 写过（§1.3 + 判据表），
#       ⇒ 本主题**只写「分片后」的账**（`zero-fsdp/01`），本节是那一支的**上游引文**
#       ⇒ 与 `90-training/checkpointing/code/specs.py` 的对应常量**同值**（见各条 `# 同值位置：`）。
# ===========================================================================
#
# 来源：[C] S. Rajbhandari, J. Rasley, O. Ruwase, Y. He,
#       **`ZeRO: Memory Optimizations Toward Training Trillion Parameter Models`**，
#       arXiv **1910.02054**。
#       评审状态：**不确认同行评审** —— abs 页的 `Comments:` 字段**实测不存在**
#         （`90-training/checkpointing/code/specs.py` 已逐字核过：`Comments:` 计 0 次，
#          同页 `ICLR` / `NeurIPS` / `MLSys` 也都计 0 次）。
#       抓取（ar5iv 正文）：`https://ar5iv.labs.arxiv.org/html/1910.02054`，2026-09-23，
#       **279500 B**。
#       ★★ **载体的一个坑**（同上游）：ar5iv 把 `Ψ` 这类符号**渲染成两份**
#         （`4 \Psi+K\Psi` 后面紧跟 `4\Psi+K\Psi`）⇒ **凡跨公式的引文都不要取**。
#         本模块在本主题这一轮**没有重新抓取**（引文逐条来自 `90-training` 已取的现场，
#         ★ 这是一处**转述**：本文件不声称自己复核了抓取，只声称**与上游常量同值**）。

ZERO_ARXIV_ID = "1910.02054"
# 来源：[C] ZeRO（同上），abs 页 `<title>`（去掉 `[1910.02054] ` 前缀后的编号本体）。
# 口径：★ ZeRO 的**编号**（`P21` 的落法：编号与标题**成对存放**，缺一不可）。
# 消费者：`01_grad_averaging.py` 的 `D.5`、`03_redundancy.py` 的 `D.5`。
# 同值位置：`90-training/checkpointing/code/specs.py` 的 `ZERO_ARXIV_ID`（同值）。

ZERO_TITLE = "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"
# 来源：[C] ZeRO（同上），abs 页 `<title>`，原始字节 1 次。
# 口径：★★ **`Memory Optimizations` 那两个词是本主题的立足点** —— 本域 `zero-fsdp` 的
#       「冗余账」正是从这篇的**显存口径**出发的；`03_redundancy.py` 只引它的**字节系数**，
#       **不写**分片策略与通信量（那在 `zero-fsdp/01`/`02`）。
# 消费者：`01_grad_averaging.py` 的 `D.5`、`03_redundancy.py` 的 `D.5`。
# 同值位置：`90-training/checkpointing/code/specs.py` 的 `ZERO_TITLE`（同值）。

ZERO_K_OPT_STATE = 12
# 来源：[C] ZeRO（同上），§5.1 逐字 `Mixed-precision Adam has K = 12`（**原始字节 1 次**）。
# 口径：★★ **`K = 12` 是 ZeRO 自己的举例口径**（条件逐字
#       `based on mixed-precision training with Adam optimizer.`，见 `ZERO_FIGURE_CALIBER_QUOTE`）
#       —— **不是**四篇论文的共识 ⇒ 凡引用它，正文必须带上 `ZERO_K_CALIBER_NOTE` 那句。
#       ★ 它只限**优化器状态**（`ZERO_K_DEFINITION_QUOTE`），**不含**参数与梯度。
# 消费者：`01_grad_averaging.py` 的 `D.5`、`03_redundancy.py` 的 `A.2`/`D.5`。
# 同值位置：`90-training/checkpointing/code/specs.py` 的 `ZERO_K_OPT_STATE`（同值）。

ZERO_PARAM_FP16_MULTIPLE = 2.0
# 来源：[C] ZeRO（同上），§5.1：模型状态里「参数」按 **16 位**存的字节数 = `2Ψ`
#       （原文的求和式 `2 Ψ + 2 Ψ + K Ψ = 16 Ψ` 含公式标记，按上游的口径**不取整句**，只取它的**项**）。
# 口径：★ 这个 `2` 是 **fp16 的字节数**（2 字节/参数）；本主题**不重推**数值格式
#       （正文在 `130-quantization/numerical-formats/01`）。
# 消费者：`03_redundancy.py` 的 `A.2`（每卡状态 `16Ψ` 的**参数项**）。
# 同值位置：`90-training/checkpointing/code/specs.py` 的 `ZERO_PARAM_FP16_MULTIPLE`（同值）。

ZERO_GRAD_FP16_MULTIPLE = 2.0
# 来源：[C] ZeRO（同上），§5.1（同一式里的梯度项 `2Ψ`，fp16）。
# 口径：★★★ **本主题最要紧的一项**：DP 里**每个 rank 都持一份完整的梯度**（`2Ψ`）
#       ⇒ `01_grad_averaging.py` 的 `B.1` 把「实测的梯度字节数」与这条 [C] 常量对拍
#       （两条**来源不同**的路径）。
#       ★ 它与 `ZERO_PARAM_FP16_MULTIPLE` **同值但不同项** ⇒ **两条分开存放、不得合并**。
# 消费者：`01_grad_averaging.py` 的 `B.1`、`03_redundancy.py` 的 `A.2`。
# 同值位置：`90-training/checkpointing/code/specs.py` 的 `ZERO_GRAD_FP16_MULTIPLE`（同值）。

ZERO_TOTAL_MULTIPLE = 16.0
# 来源：[C] ZeRO（同上），§5.1 逐字 `In total, this results in 2` …
#       `bytes of memory requirement`（两个片段各 1 次；中间夹着公式，按上游口径切头取尾）。
# 口径：★★★ **这是「每卡的训练状态」的量**（原文的词是 `memory requirement`）⇒
#       DP 的冗余账 = `N 个副本各 16Ψ` 存一份 `16Ψ` 的状态。
#       ★★ **本主题不重推 `16Ψ` 的分解** —— 分解的正文在
#       `90-training/checkpointing/01-state-and-bytes.md`（§1.3 + 判据表 +
#       `code/01_state_bytes.py:310` 的 `A.3`）⇒ 本主题只**消费**这个数。
# 消费者：`03_redundancy.py` 的 `A.2`/`B.1`/`D.5`。
# 同值位置：`90-training/checkpointing/code/specs.py` 的 `ZERO_TOTAL_MULTIPLE`（同值）。

ZERO_FIGURE_MODEL_QUOTE = "denotes model size (number of parameters), K"
# 来源：[C] ZeRO（同上），**Figure 1 的说明文字**，**原始字节 1 次 / 剥标签后 1 次**。
# 口径：★ 它把 `Ψ` 定义成「**参数个数**」 ⇒ 本主题所有 `×Ψ` 的账都按**参数个数**计，
#       不是按「参数张量数」、也不是按「元素字节数」。
#       ★★ `01_grad_averaging.py` 的 `B.1` 与 `03_redundancy.py` 的 `A.2` 都靠它定口径：
#       程序自己数的是 `int(np.size(w))` 的和，**与这条逐字定义对拍**。
# 消费者：`01_grad_averaging.py` 的 `B.1`/`D.5`、`03_redundancy.py` 的 `A.2`/`D.5`。
# 同值位置：`90-training/checkpointing/code/specs.py` 的 `ZERO_FIGURE_MODEL_QUOTE`（同值）。

ZERO_K12_QUOTE = "Mixed-precision Adam has K = 12"
# 来源：[C] ZeRO（同上），§5.1，**原始字节 1 次 / 剥标签后 1 次**。
# 口径：★ 这是 `ZERO_K_OPT_STATE` 的**逐字证据**（值本身不可自证 ⇒ 引文与值成对存放）。
# 消费者：`01_grad_averaging.py` 的 `D.5`、`03_redundancy.py` 的 `D.5`。
# 同值位置：`90-training/checkpointing/code/specs.py` 的 `ZERO_K12_QUOTE`（同值）。

ZERO_TOTAL_QUOTE = "In total, this results in 2"
# 来源：[C] ZeRO（同上），§5.1，**原始字节 1 次 / 剥标签后 1 次**。
# 口径：★ 与 `ZERO_TOTAL_TAIL_QUOTE` **成对读**（中间那一段是公式，不取）。
# 消费者：`03_redundancy.py` 的 `D.5`。
# 同值位置：`90-training/checkpointing/code/specs.py` 的 `ZERO_TOTAL_QUOTE`（同值）。

ZERO_TOTAL_TAIL_QUOTE = "bytes of memory requirement"
# 来源：[C] ZeRO（同上），§5.1，**原始字节 1 次 / 剥标签后 1 次**。
# 口径：★★ 结尾是 `memory requirement`（**显存/内存需求**），不是 `checkpoint size`
#       ⇒ **这一句本身就是「这是每卡的量」的字面依据**。
# 消费者：`01_grad_averaging.py` 的 `D.5`、`03_redundancy.py` 的 `D.5`。
# 同值位置：`90-training/checkpointing/code/specs.py` 的 `ZERO_TOTAL_TAIL_QUOTE`（同值）。

ZERO_FIGURE_CALIBER_QUOTE = "based on mixed-precision training with Adam optimizer."
# 来源：[C] ZeRO（同上），**Figure 1 的说明文字**，**原始字节 1 次 / 剥标签后 1 次**。
# 口径：★★ `K = 12` 的**口径来源**（`K` 只在 mixed-precision + Adam 这一组假设下是 12）
#       ⇒ 「必须注明这是 ZeRO 的举例口径」这条要求的**逐字依据**。
# 消费者：`03_redundancy.py` 的 `A.2`/`D.5`。
# 同值位置：`90-training/checkpointing/code/specs.py` 的 `ZERO_FIGURE_CALIBER_QUOTE`（同值）。

ZERO_K_CALIBER_NOTE = "ZeRO 的举例口径"
# 来源：★ **本库的措辞常量（E 级）**，不是来源里的句子 —— 它是对 `ZERO_FIGURE_CALIBER_QUOTE`
#       的中文概括，作用是**让正文里那句限定语变成一个可断言的字面量**。
# 口径：★★ 配套文档 `03-redundancy-and-limits.md` 里**必须逐字出现这个字符串**
#       （它是 `16Ψ` / `12Ψ` 那个量的限定语）—— 这是「订正不扩散」（`P1`）在本主题的
#       可执行落法：`03_redundancy.py` 的 `D.5c` **会去读那份文档**并检查它逐字在
#       （★ 文档不在时**跳过并打印**，因为 `mutate.py` 的副本里只有 `*.py`）。
#       ★ 另两篇（`01`/`02`）不消费这条常量，故不在检查范围内。
# 消费者：`03_redundancy.py` 的 `A.2`/`D.5b`（**读配套文档**的那一条）。
# 同值位置：`90-training/checkpointing/code/specs.py` 的 `ZERO_K_CALIBER_NOTE`（同值）。

# ===========================================================================
# 2. [B] NVIDIA NCCL 的 AllReduce —— **产品官方文档**（不是规范）
#    ★★ 本主题的模拟器把通信实现成「显式的张量搬运」：`all_reduce = Σ局部`。
#       这一节的常量就是**那条语义的外部锚点**（域 README 的「代码约定」表：本域的知识对象
#       是**切分策略**，不是通信原语的实现）。
#    ★★★ **一处必须写明的读法**：NCCL 的文档**没有**说 `AllReduce` 的 `op` 参数有默认值
#       （`default` 这个词只出现在页面模板的 CSS 里）⇒ **「默认 SUM」这一条不是 NCCL 的**
#       （它登记在 `UNSOURCED`：`nccl_allreduce_default_op`）。NCCL 给的是**sum 形式**的定义。
# ===========================================================================
#
# 来源：[B] **NVIDIA NCCL 2.31.2 用户指南**（产品官方文档，**不是规范**）
#       `https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html`
#       抓取日 **2026-09-23**（`curl -sSL`，HTTP 200），**原始字节 86584**，
#       `sha256[:16] = d7d2b6cff49d9f79`。
#       ★ 版本号的载体：页面 `<title>` 逐字 `NCCL 2.31.2 documentation`（**原始字节 1 次**）
#         ⇒ 取**精确到补丁号的 `2.31.2`**；★ **同页侧边栏另有 `NCCL 2.31`**（原始字节 1 次，
#         少了补丁号）—— 两个都存在，本模块取**更精确的那一个**（见 `NCCL_DOC_VERSION`）。
#       ★★ 引文**一律不跨标记**（`sum` / `k` 这类斜体词在原始字节上是 `<em>` 包起来的
#         ⇒ 跨它的片段计不到）—— 本节的每一段都在**原始字节**上核过次数。

NCCL_DOC_VERSION = "2.31.2"
# 来源：[B] NCCL 用户指南（同上），页面 `<title>` 逐字 `NCCL 2.31.2 documentation`，
#       **原始字节 1 次**（★ **同页侧边栏另有 `NCCL 2.31`**，原始字节 1 次 —— 少了补丁号；
#        本模块取 `<title>` 里的**精确版本**，并把这个「同页两个版本串」记在这里：
#        ★ 这正是**量法口径**的一个小现场 —— 「哪一版」取决于**读页面的哪个部位**）。
# 口径：★★ **产品文档必须带版本**（`P19`：把「某实现的文档」当规范是本域头号错误模式；
#       而**不带版本的产品文档连「哪一版」都说不清**）⇒ 这一条是本节所有引文的**版本锚点**。
# 消费者：`01_grad_averaging.py` 的 `D.6`。
# 同值位置：无（本库只有本主题引 NCCL）。

NCCL_USAGE_COLLECTIVES_URL = (
    "https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html"
)
# 来源：[B] NCCL 用户指南（同上），`curl -sSL` 取回的**实际 URL**（HTTP 200，无重定向）。
# 口径：★ 引文类常量必须写清**载体**（docs 页 + URL + 抓取日 + 字节数 + 摘要）。
# 消费者：`01_grad_averaging.py` 的 `D.6`。
# 同值位置：无。

NCCL_PAGE_BYTES = 86584
# 来源：[B] NCCL 用户指南（同上），`curl -sSL ... | wc -c` 口径的**原始字节数**（2026-09-23）。
# 口径：★ 字节数是「引的是哪一版/哪一次抓取」的第二重锚点（第一重是 sha256）。
# 消费者：`01_grad_averaging.py` 的 `D.6`。
# 同值位置：无。

NCCL_PAGE_SHA256_16 = "d7d2b6cff49d9f79"
# 来源：[B] NCCL 用户指南（同上），对本机取回的**原始字节**求 `sha256` 后取前 16 位十六进制。
# 口径：★★ 摘要与字节数**成对存放**：只有字节数时，一次微小的页面改动不会被发现。
# 消费者：`01_grad_averaging.py` 的 `D.6`。
# 同值位置：无。

# ★★ **必须写成【一行】**（不留隐式拼接）：`mutate.py` 的 `_patch()` 对**跨行字面量**
#    直接**跳过**（计入 `跳过`，不进分母）⇒ 写成两行相邻字面量会让这一条**免检**。
NCCL_ALLREDUCE_REDUCES_QUOTE = "The AllReduce operation performs reductions on data (for example, sum, min, max) across devices and stores the result in the receive buffer of every rank."
# 来源：[B] NCCL 用户指南（同上），`AllReduce` 一节，**原始字节 1 次**。
# 口径：★★ 它给出 AllReduce 的**结果语义**：每个 rank 都拿到**同一份**归约结果
#       ⇒ 这正是 `01_grad_averaging.py` 的 `A.1`（「N 卡平均后各卡持有同一个梯度」）
#         与 `03_redundancy.py` 的 `A.1`（副本一致性）要的**结果形态**。
# 消费者：`01_grad_averaging.py` 的 `D.6`。
# 同值位置：无。

# ★★ 同上：**写在一行**（否则 `mutate.py` 会把它整条跳过 ⇒ 免检）。
NCCL_ALLREDUCE_SUM_ELEM_QUOTE = "each rank will provide an array in of N values, and receive identical results in array out of N values,"
# 来源：[B] NCCL 用户指南（同上），`AllReduce` 一节，**原始字节 1 次**。
#       ★ 它紧接的前一句是 `In a <em>sum</em> allreduce operation between <em>k</em> ranks,`
#         —— `sum` 与 `k` 被 `<em>` 包住 ⇒ **跨标记，不取**（故本片段从中段起）。
# 口径：★★★ **本主题的模拟器是被这条钉住的**：`all_reduce(parts) = Σ parts`
#       （逐元素的加法）**不是**本库的发明，而是这条逐字定义在本机单进程下的**可执行形态**。
#       ⇒ 它是 `01_grad_averaging.py` 的 `A.1` 里那条「N 卡平均」路径的**外部依据**。
# 消费者：`01_grad_averaging.py` 的 `D.6`。
# 同值位置：无。

NCCL_RS_AG_EQ_QUOTE = "Executing ReduceScatter, followed by AllGather, is equivalent to the AllReduce operation."
# 来源：[B] NCCL 用户指南（同上），`AllGather` 一节，**原始字节 1 次**。
# 口径：★★★ **两条边界同时由这一句钉住**：
#       ① 本主题把 all-reduce 拆成「reduce-scatter + all-gather」两个阶段**是文档里的分解**，
#          不是本库的发明 ⇒ `02_grad_sync.py` 的**桶（分段搬运）**正是这个分解的最细颗粒；
#       ② ★★ **而这两个阶段的算法与复杂度（`V(N) = 2(N−1)/N × S`）不在本主题**
#          ⇒ 回链 `110-communication`（域 README 的「本域不做什么」表第一行）。
# 消费者：`02_grad_sync.py` 的 `D.6`。
# 同值位置：无。

# ===========================================================================
# 3. [B] PyTorch 2.14 的 `torch.distributed.all_reduce` —— **产品官方文档**
#    ★★ 「AllReduce 默认 SUM」这一条的**真出处在这里**（NCCL 那边没有，见第 2 节的读法）。
#    ★★ 另一条更值钱的：签名下的说明逐字说结果**在所有进程里逐位相同** ——
#       这是 `03_redundancy.py` 的 `A.1`（副本一致性）的**外部锚点**。
# ===========================================================================
#
# 来源：[B] **PyTorch 2.14** 的 `torch.distributed` 页，
#       `https://docs.pytorch.org/docs/2.14/distributed.html`，
#       抓取日 **2026-09-23**（`curl -sSL`，HTTP 200），**原始字节 1509715**，
#       `sha256[:16] = 1a4a7e398b5a7744`。
#       ★★ **一处载体口径**：这一页的**签名是重标记的**（每个 token 各自一个 `<span>`）
#         ⇒ **签名整体在原始字节上不可数**；本模块取的片段都是**单个 token 或整段散文**
#         （前者在 `<span class="pre">…</span>` 里**是连续的**，后者在 `<p>` 里是连续的）。

TORCH_DIST_DOC_URL = "https://docs.pytorch.org/docs/2.14/distributed.html"
# 来源：[B] PyTorch 2.14 的 `torch.distributed` 页（同上）。
# 口径：★ **版本在 URL 里**（`/2.14/`）—— `docs.pytorch.org/docs/stable/...` 会**重定向**，
#       写 `stable` 就等于没写版本（`P19` 的落法：产品文档必须可定位到版本）。
# 消费者：`01_grad_averaging.py` 的 `D.6`。
# 同值位置：`90-training/checkpointing/code/specs.py` 的 `DCP_DOC_URL` 是**同一站点的另一页**
#           （`distributed.checkpoint.html`），**同站不同页** ⇒ 不是同值位置。

TORCH_DIST_PAGE_BYTES = 1509715
# 来源：[B] PyTorch 2.14 的 `torch.distributed` 页（同上），2026-09-23 的**原始字节数**。
# 口径：★ 与 `NCCL_PAGE_BYTES` 同一形态（字节数 + 摘要成对）。
# 消费者：`01_grad_averaging.py` 的 `D.6`。
# 同值位置：无。

TORCH_DIST_PAGE_SHA256_16 = "1a4a7e398b5a7744"
# 来源：[B] PyTorch 2.14 的 `torch.distributed` 页（同上），本机取回原始字节的 `sha256[:16]`。
# 口径：★★ 本库的**页面摘要惯例**（`90-training/checkpointing/code/specs.py` 的
#       `DCP_DOC_SHA256_16` 同形态）—— 但那边是与**计划 §C.2b 的 T3** 互钉，
#       ★ 本主题**没有**第二份现场可比 ⇒ 本摘要只与**本文件记录的那一次抓取**绑定。
# 消费者：`01_grad_averaging.py` 的 `D.6`。
# 同值位置：无（不同页 ⇒ 不同摘要）。

TORCH_ALLREDUCE_DEFAULT_OP_QUOTE = "ReduceOp.SUM"
# 来源：[B] PyTorch 2.14 的 `torch.distributed` 页（同上），
#       `torch.distributed.all_reduce` 的**签名**里的默认值参数，**原始字节 16 次**
#       （`ReduceOp.SUM` 这一串在整页上计 16 次：`all_reduce` / `all_reduce_coalesced`
#        等处的签名与示例都在用）。
# 口径：★★★ **`op` 的默认值是 `ReduceOp.SUM`（求和）** —— 这一条是
#       `01_grad_averaging.py` 的 `C.1`（「用求和代替平均 ⇒ 差 `N` 倍」）的**前提**：
#       真实系统的 all-reduce **默认就是求和**，所以**除以 `N` 那一步必须由 DP 自己补**。
#       ★★★ **它不是 NCCL 的**：NCCL 的文档没有默认值（`UNSOURCED`：
#         `nccl_allreduce_default_op`）⇒ **正文不得写成「NCCL 的 AllReduce 默认 SUM」**。
# 消费者：`01_grad_averaging.py` 的 `C.1`/`D.6`。
# 同值位置：无。

TORCH_ALLREDUCE_REDUCES_QUOTE = (
    "Reduces the tensor data across all machines in a way that all get the final result."
)
# 来源：[B] PyTorch 2.14 的 `torch.distributed` 页（同上），
#       `torch.distributed.all_reduce` 的正文首句，**原始字节 2 次**
#       （`all_reduce_coalesced` 的正文里同样一句）。
# 口径：★ 「**all get the final result**」= 每个进程都拿到结果 ⇒ 与 NCCL 那句同义，
#       两条**独立来源**互证「结果在所有 rank 上相同」这个形态。
# 消费者：`01_grad_averaging.py` 的 `D.6`。
# 同值位置：无。

TORCH_ALLREDUCE_BITWISE_QUOTE = "bitwise identical in all processes"
# 来源：[B] PyTorch 2.14 的 `torch.distributed` 页（同上），`all_reduce` 的正文第二句
#       （完整句：`After the call tensor is going to be bitwise identical in all processes.`），
#       **原始字节 3 次**。
# 口径：★★★ **这一句是 `03_redundancy.py` 的 `A.1` 的外部锚点**：
#       DP 的隐含假设是「`N` 份副本**逐位相同**」—— 文档把它写成了**逐位**（不是「近似相同」）。
#       ⇒ 本主题因此用 **sha256 摘要相等**做那一条判据（`A.1`），而不是用容差比较。
# 消费者：`03_redundancy.py` 的 `A.1`/`D.6`。
# 同值位置：无。

# ===========================================================================
# 4. UNSOURCED —— 待办清单，不是免责声明
# ===========================================================================
#
# ★ `CONTRIBUTING` §2：「不得**既保留数字又记 UNSOURCED**」—— 下面每一项都**没有**出现在
#   本模块的正文常量里（也**没有**出现在三篇文档的正文里）。
# ★★ 本元组的字符串**有消费者**：`01_grad_averaging.py` 的 `D.9` 逐位钉住整条元组
#   （上游 `90-training/*/code/specs.py` 的同节形态）—— ⇒ 它们**不是**漏网候选，
#   删掉一项或改掉一项都会让那条断言失败。

UNSOURCED = (
    # 每一项是一个**稳定标识符**（不是散文），格式：`<主题>_<条目>`
    # ★★ 第 1 条是本轮**真的撞到**的一处来源错配（任务书说「NCCL 的 AllReduce 默认 SUM」）：
    #    NCCL 2.31 的三份页面上 `default` 只出现在模板 CSS / 与 op 无关的位置
    #    （实测：usage/collectives 上 `default` 计 1 次，是侧边栏样式；
    #     api/colls 与 api/types 上的 `default` 均**不描述 `op` 的默认值**）
    #    ⇒ 「默认 SUM」这一条**只能**引 PyTorch 的签名（`TORCH_ALLREDUCE_DEFAULT_OP_QUOTE`）。
    "nccl_allreduce_default_op",
    # 「冗余度」这个量**全库没有既有定义** ⇒ 本主题自立的定义（`(总量−单份)/总量`）没有外部来源；
    # 它是 `03_redundancy.py` 的 `B.1` 的**本库定义（E 级）**，不是任何文献里的量。
    "data_parallel_redundancy_ratio_definition",
    # ★★★ **禁区**：ring all-reduce 的每卡搬运量 `V(N) = 2(N−1)/N × S` 与它的修正系数
    #   **不在本主题**（域 README 的「本域不做什么」表）⇒ 本模块**一个字节常量都没有**。
    #   正文在两处已有：`10-foundations/performance-models/02-amdahl-and-scaling.md:209`
    #   与 `20-gpu-architecture/interconnect/04-measuring-bandwidth.md:61-76`。
    "data_parallel_ring_allreduce_volume",
)

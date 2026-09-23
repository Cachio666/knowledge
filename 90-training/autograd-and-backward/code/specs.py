"""90-training/autograd-and-backward 的外部事实常量源。

本模块是 `CONTRIBUTING` §2 约定的**共享事实源**（`specs.py`）：本主题的编号程序从它读常量，
推导过程可复核，**常量本身不可自证**。

★★ **本模块最重要的约定：来源必须分级，并写明它是哪一级**（计划 §1.5）。
  本域的头号错误模式是 **P19 —— 把「某实现的文档」当成「规范的文档」**（B 级当 A 级）。
  凡引用 `O0/O1/O2/O3` / `GradScaler` / `clip_grad_norm_` 这类**「某实现的规格」**，
  **必须标 B 级并写明是哪个实现**（以及版本）。

★ 三类真值来源**必须分开标注**（计划 §1.5）：
    ① **外部规格值**：论文原文（**C 级，须注明是否确认同行评审**）、厂商/产品文档（A/B 级）
       ⇒ 本模块存放的就是这一类，**每条带 `# 来源：` + `[A-E]` 级 + 抓取日**。
    ② **本机实测值**：本域程序在本机跑出的数（**须写 python3 与 numpy 版本**）⇒ 不放在本模块。
    ③ **推导值**：程序算出的量（可手算复核）⇒ **不放在本模块**（`CONTRIBUTING` §2：
       「只放从来源抄来的数，**不放推导值**」）。

★ **只放本主题真的会断言的常量**（计划 §3 的 R3 裁定）。每一条下面写明**预期消费者**
  ——哪一篇的哪一条判据会用到它。★ **没有消费者的常量就是 `make mutate` 的漏网候选**
  （计划 §4 **T11 的盲区①**：`80` 实测「只用于打印的模块级常量表」贡献过 2 条漏网）
  ⇒ Chunk 1 落盘后**凡仍无消费者、又仍留在这里的**，必须在报告里逐条给出理由或删掉它。

★ 每条常量带 `# 来源：` + `[A-E]` 级 + 抓取日；跨主题/跨文件同值的另带 `# 同值位置：`。
  分级判据见 `CONTRIBUTING` §2（A 厂商规范 / B 产品官方文档 / C 论文 / D 社区 / E 本库推导）。

★★ **本文件不做的一件事**：**不在模块级放「只为了被引用而存在」的字符串**。
  理由同上（T11 盲区①）。有来源、但**没有可断言的消费者**的逐字引文写在 §3 的注释里，
  由**文档**引用、由**人**核对 —— 放成模块级常量只会变成漏网。
"""

# ===========================================================================
# 1. Chen 2016 —— 「重算换显存」的账
#    本主题**唯一**的 [C] 级来源（也是 `02-gradient-checkpointing.md` 的立足点）
# ===========================================================================
#
# 来源：Chen, Xu, Zhang, Guestrin,
#       **`Training Deep Nets with Sublinear Memory Cost`**，arXiv **1604.06174**  [C]
# ★★ **P21 已办**：本模块 2026-09-23 打开过 `https://arxiv.org/abs/1604.06174`，
#    `<title>` 逐字 = `[1604.06174] Training Deep Nets with Sublinear Memory Cost`，
#    `citation_arxiv_id` 逐字 = `1604.06174` ⇒ **编号与标题对得上**。
# 评审状态：★★ **不确认同行评审**。判据是**实测**：abs 页的原始字节里
#    `Comments:` = **0** 次、`journal-ref` = **0** 次、`Journal ref` = **0** 次
#    （⇒ 没有 venue 线索；**按实测填，不凭印象补**）。
# 抓取：`https://ar5iv.labs.arxiv.org/html/1604.06174`（HTML 全文），2026-09-23，
#    134317 B，`sha256[:16] = 59f865f132b7a506`（**两次抓取同一哈希**）。
# ★ 方法学让步：**逐字取自 ar5iv 的 HTML 全文，未与 PDF 文本层核对** —— 引用时不要给页码。
# ★★ **计数纪律（本库的 Y15）**：凡引文都**在原始字节上计数**（`raw.count`），并做对照。
#    本模块实测到两个坑，与 `130-…`/`Apex` 的坑同类：
#      · **跨行空白**：`from 48G` 计 **0** 次、`48G to 7G` 计 **1** 次
#        —— 原文在 `from` 之后有一个**换行**；⇒ 引文片段必须**不含跨行空白**才数得到。
#      · **跨标记**：`memory to train a` 计 **0** 次（`n` 被 `<math>` 包着）；
#        ⇒ 含公式的句子**不要整句引用**，只引公式之外的连续片段。

# ── 内存账 ──────────────────────────────────────────────────────────────────
CHEN2016_MEMORY_GB_1000_LAYER = (48, 7)
# 来源：[C] Chen 2016（同上）。逐字（原始字节 `raw.count` = 1）：
#         `Our experiments show that we can reduce the memory cost of a 1,000-layer
#          deep residual network from 48G to 7G on ImageNet problems.`
#       ★ 可数的连续片段：`48G to 7G`（1 次）、`1,000-layer deep residual network`（1 次）。
# 口径：**该实验的绝对内存**（1000 层 ResNet、ImageNet、单卡）——
#       **不是**「任意 n 的比例」；不要拿它当公式值。
# 消费者：`02_checkpointing.py`（本主题 表 A ③④ 的**量级对照**：本机的 toy 实现给出的
#        是「存 ⌈√n⌉、前向 ≈2n」的**形状**，与这组绝对值**不是同一个量** ⇒
#        程序要断言的正是「形状对得上、绝对值不可比」。）
# 同值位置：无（2026-09-23 全库扫描 `.md`/`.py`/`.cu`，未发现第二处）。

CHEN2016_LAYERS = 1000
# 来源：[C] Chen 2016。逐字同上句里的 `1,000-layer`（原始字节 1 次）。
# 口径：**该实验**的层数。★ 它**不是** `√n` 里的那个 `n` 的取值上限 —— 那个 `n` 由程序自己取。
# 消费者：`02_checkpointing.py`（用来把本机的 `n` 与「论文那个 n」区分开，写进输出对照）。
# 同值位置：无。

CHEN2016_MEMORY_COST_K_EQ_SQRT_N_FACTOR = 2.0
# 来源：[C] Chen 2016，§4.3。原始字节：`O(2\sqrt{n})` 计 **2** 次
#       （一次在 `<math alttext>`、一次在 `<annotation encoding="application/x-tex">`；
#        同一处公式被渲染了两个副本 —— 计数时**必须知道这件事**）。
#       ★ 同段的 `k=\sqrt{n}` 计 **4** 次（两个副本 + MathJax 的两处 alttext）。
# 口径：★★ **这是「内存」的系数，不是前向次数**。原文那一段逐字是
#       `cost-total = max_i cost-of-segment(i) + O(k) = O(n/k) + O(k)`（式 (1)），
#       紧接着 `Setting k = √n, we get the cost of O(2√n)`。
#       整段的主语是 **memory cost**（该节标题即 `An O(√n) Memory Cost Algorithm`）。
#       ⇒ **不要**把它当成「前向次数翻倍」的来源（那是 `CHEN2016_EXTRA_FORWARD_PASSES`）。
#       ★ 计划 §0.7-C.2 把这句登记在「梯度检查点」节下时**没标口径** —— 本模块补上。
# 消费者：`02_checkpointing.py`（断言本机实现的**存活激活数** ≈ `2√n` 的形状）。
# 同值位置：无。

CHEN2016_EXTRA_FORWARD_PASSES = 1
# 来源：[C] Chen 2016，摘要。可数的连续片段（原始字节 1 次）：
#         `with only the computational cost of an extra forward pass`
#       ⚠ 紧跟其后的 `per mini-batch` 数不到（原文在 `per` 与 `mini-batch` 之间**换行**）。
# 口径：**「多出一个完整前向」的工作量**（不是「多一次层前向」）。
#       ⇒ 与 `n + 重算 ≈ 2n` 是**同一件事的两种说法**：原 n 次 + 额外 n 次 = 2n。
# 消费者：`02_checkpointing.py`（表 A ④：实测前向次数 ∈ [1.8n, 2.1n]；
#        ★ 本常量就是那半边的**外部锚点** —— 本机实测 1.97n / 1.98n）。
# 同值位置：无。

CHEN2016_EXTRA_RUNTIME_FRACTION = 0.30
# 来源：[C] Chen 2016，§5.4 与 Fig. 7。可数的连续片段（原始字节各 1 次）：
#         `the sublinear allocation strategy costs` …（`30` 被 `<math>` 包着）… `additional runtime`
#       ⇒ **整句含公式，数不到**；这里取的是它的**数值 30**（`alttext="30"` 可数，2 次）。
# 口径：★★ **wall-clock 的额外开销**（`Titan X`、每次跑 20 个 batch、ResNet 与 LSTM 两组），
#       **不是**前向次数的比例 —— 同一篇的模型说前向多出一整个（= +100% 的前向工作量），
#       而 wall-clock 只涨 30%，因为原文逐字说 `Since the backward operation is nearly
#       twice as time consuming as the forward one`（原始字节 1 次）。
#       ⇒ **两个口径不可互换**；写正文时两个都要标出来。
# 消费者：`02_checkpointing.py`（断言「本机的 2n 是**计数**口径，与 0.30 的 wall-clock 口径不同」）。
# 同值位置：无。

CHEN2016_LOG_N_SHAPES = ("O(log n)", "O(n log n)")
# 来源：[C] Chen 2016，摘要。可数的连续片段（原始字节各 1 次）：
#         `with as little as` 与 `extra cost for forward computation`
#       （两者之间的 `O(n log n)` 被 `<math>` 包着 —— 同样数不到）。
# 口径：★ 这是**极端情形**的一对形状，**不是** `k = √n` 那一条：
#         **内存** `O(log n)`、**额外前向开销** `O(n log n)`（原文逐字限定 `extra cost
#         for forward computation`，即那个 `n log n` 属**计算**不属内存）。
#       ★ 本元组里的两个字符串**是「形状的名字」**，不是公式 —— 程序按名字比对（`==`）。
# 消费者：`02_checkpointing.py`（表 A ⑤：实现一个 log-n 的分段方案，断言它选中的**形状名**
#       精确等于 `CHEN2016_LOG_N_SHAPES[0]`；★ **用 `==`，不要用 `startswith`/`in` 的前缀式检查**，
#       否则变异末端追加一个字符会漏网 —— 计划 §3 开头的盲区③）。
# ★★ **`[1]`（`"O(n log n)"`）此刻没有消费者** —— 它是那对形状里的**第二个**
#    （额外前向开销）。本模块 2026-09-23 在**临时副本**上做了一次空跑演练
#    （往 `code/` 里塞一个只钉住 `[0]` 的编号程序后跑 `python3 mutate.py`）：
#    12 个叶子测下来 **捕获 9 / 漏网 3** ——漏网正是 `[1]` 与 `UNSOURCED` 的两条。
#    ⇒ **Chunk 1 的 `02` 必须给 `[1]` 一个消费者**（log-n 方案的**前向开销**形状），
#      **否则就把它从元组里删掉**（计划 §4 T11 盲区①：没有消费者的常量只能是漏网）。
# 同值位置：无。

# ── 本主题**刻意不搬**的 Chen 2016 条目（搬进来只会制造变异漏网）──────────────
#   · `O(n/k) + O(k)`（式 (1)）本身：它是**公式**不是数，程序里直接写出来即可；
#     放成常量只能被「常量 == 常量」这种**恒真断言**消费（本库的 P17）。
#   · 论文的 arXiv ID 与标题：**引用元数据**，由文档的「参考」节承担，程序不消费（见 §3）。


# ===========================================================================
# 2. PyTorch 的梯度裁剪口径 —— ★★ **B 级：「某实现的规格」，必须写明是哪个实现**
# ===========================================================================
#
# ★★ 这一节是 **P19 的现场**：`clip_grad_norm_` 不是「规范」，它是**某个实现**的 API。
#    引用它必须写 B 级 + 实现名 + 版本；**不得**写成「规范规定」。

TORCH_CLIP_COEF_DENOM_EPS = 1e-6
# 来源：**PyTorch** 的 `torch/nn/utils/clip_grad.py`  [B]（**某实现的源码**，不是规范）
#       版本：**v2.9.0**；抓取：`https://raw.githubusercontent.com/pytorch/pytorch/v2.9.0/
#       torch/nn/utils/clip_grad.py`，2026-09-23，11427 B，`sha256[:16] = 63efb273c62fd917`
#       逐字（原始字节 2 次）：`clip_coef = max_norm / (total_norm + 1e-6)`
#                             公式串 1 次：`grad = grad * \min(\frac{max\_norm}{total\_norm + 1e-6}, 1)`
# 口径：**分母上的那个 `1e-6`**（`total_norm = 0` 时的保护项），
#       ⇒ 裁剪后的全局范数 = `max_norm · T/(T + 1e-6)`，**相对误差 = `1e-6/T`**
#       （`T` = **裁剪前**的全局范数）。★ **`T < 1` 时这个「≈精确」的说法就不成立** ——
#       计划 §3 表 A ⑦ 因此把 `T ≥ 1` 写死、判据写成 `相对误差 ≤ 1.1e-6 / max(T,1)`。
# 消费者：`03_grad_lifecycle.py`（表 A ⑦：**裁剪后范数 == max_norm** 的那条判据；
#        ★ 本主题只在此处写一次阈值，Chunk 3 与 Chunk 4 回链本行，不重定义）。
# 同值位置：`90-training/probes/probe_90c.py` 的 `clip_global()`（`max_norm / (total + 1e-6)`）
#       —— ★ 那是**探测脚本**（`probes/`，非编号程序、不参与核对），同一个口径；
#       ★ **本模块是这个 `1e-6` 在本域的第一处「常量」**；`03_grad_lifecycle.py` 落盘后
#         它就是该值的**第 2 处**（那时本行应改成指它）。
#       ⚠ **未核**：本库别处的 torch 是 2.6.0；`+1e-6` 在不同版本间是否恒定**未逐版核对**
#         ⇒ 引用时**写出 v2.9.0**，不要写成「PyTorch 规定」。
# ★★ 与计划的一个**偏差（本模块按文件实测处理并记在此）**：
#    计划 §0.7-C.2 只写了「PyTorch `clip_grad_norm_`：★ 返回值逐字 =
#    `Total norm of the parameter gradients (viewed as a single vector)` ⇒ **是裁剪前的范数**」。
#    那句话**成立**（本模块 2026-09-23 在
#    `https://docs.pytorch.org/docs/2.14/generated/torch.nn.utils.clip_grad_norm_.html`
#    的原始字节上数到 2 次，对照：同页 `max_norm` 4 次、`viewed as a single vector` 2 次），
#    但**它没有可断言的消费者** ⇒ 见 §3。


# ===========================================================================
# 3. 有来源、但**不作模块级常量**的条目（★ 理由：没有消费者 ⇒ 只会成为漏网）
# ===========================================================================
#
# ★★ 计划 §4 **T11 的盲区①**：`80` 实测「只用于打印的模块级常量表」制造过 **2** 条漏网。
#    对策是「要么给它消费者、要么删掉」。下面这几条**有来源但无消费者**，因此**写在注释里**：
#    它们由**文档**引用、由**人**核对（表 B 的「引用判据」本来就是人工核对的口径）。
#
# (a) `clip_grad_norm_` 的返回值口径 —— [B] PyTorch 文档（**不是规范**）
#     逐字：`Total norm of the parameter gradients (viewed as a single vector).`
#     出处：`torch.nn.utils.clip_grad_norm_` 的 `Returns` 段；
#           URL `https://docs.pytorch.org/docs/2.14/generated/torch.nn.utils.clip_grad_norm_.html`
#           （★ **取法有坑**：`/docs/stable/...` 返回的是 **1470 B 的跳转壳**，
#            里面是 `<meta http-equiv="refresh" content="0; url=../../2.14/…">`；
#            **`curl -L` 跟的是 HTTP 重定向，跟不了 meta refresh** —— 正确做法是
#            **读出 meta 里的目标 URL 再取一次**。计划 §4 的 **T20** 记的正是这件事。）
#           抓取 2026-09-23，179139 B，`sha256[:16] = 04c1f86ba650d249`。
#     含义：**返回值 = 裁剪前**的范数（`get_total_norm()` 的结果，`clip_grads_with_norm_()` 只是消费它）。
#     去向：`03-gradient-lifecycle.md` 的**表 B（引用判据）**逐字引用 + 标明 [B] + 实现名。
#
# (b) Chen 2016 的引用元数据（arXiv ID / 标题 / 评审状态）→ 文档的「参考」节，见 §1 抬头。
#
# (c) 已知的**同名不同指**（写正文时必须带着的边界声明，不是常量）：
#       · **`checkpoint`**：**建域前**全库（除 `docs/` 与本域）`checkpoint` 忽略大小写共 **11 处**，
#         其中 **10 处是「权重/模型检查点」的用例**，第 11 处是 `00-meta/roadmap.md:71` 的**主题名**
#         （2026-09-23 逐行扫描 `.md`/`.py`/`.cu` 确认：`10-foundations/floating-point/01:232`、
#          `20-gpu-architecture/interconnect/01:43`、`80-…/positional-encoding/01:50,177`、
#          `80-…/positional-encoding/code/01_rope_definition.py:409`、
#          `80-…/attention-variants/01:192,206,212,215`、`40-kernels/attention/05:354`）。
#         ★ **量法口径**：`40-kernels/attention/05:354` 是大写 `Checkpoints` ⇒
#           **大小写敏感地数会少 1 处**；引用这个数时必须写明口径。
#         ★ 讲「重算」时全库用的是 **`重算` / `recompute`**（`40-kernels/attention/03` 里
#           `重算` 36 次、`recompute` 5 次、`checkpoint` **0** 次）。
#         ⇒ **中文「检查点」不得用于重算**；本主题的一律写「**重算换显存**」。
#           ★ 但**文件名** `02-gradient-checkpointing.py` 保留 —— 那是 Chen 2016 那一行的
#             **术语**（`gradient checkpointing` 在原文里计 **2** 次），改了就找不到论文。
#       · **`warmup`**：`00-meta/glossary.md:146` 的条目指的是 **benchmark 预热**
#         （`45-kernel-lifecycle/convergence-and-tuning/01-benchmark-hygiene.md` 里 23 次），
#         与本域的 **LR warmup**（正文在 `optimizers/03-schedules-and-clipping`）**同名不同指**。


# ===========================================================================
# 4. UNSOURCED —— 待办清单，不是免责声明
# ===========================================================================
#
# ★ `CONTRIBUTING` §2：「不得**既保留数字又记 UNSOURCED**」—— 下面每一项都**没有**出现在
#   本模块的正文常量里。
# ★★ 提醒（T11 盲区①）：本元组的字符串**没有消费者** ⇒ 它们会是 `make mutate` 的**漏网**候选。
#    这是本库的既有形态（先例：`80-model-architectures/transformer/code/specs.py` 同节），
#    Chunk 1 落盘后若仍然漏网，**在 Chunk 报告里逐条解释**，不要静默留着。

UNSOURCED = (
    # 每一项是一个**稳定标识符**（不是散文）。
    "chen2016_exact_segmented_forward_count",   # 「n 层、每 ⌈√n⌉ 层一段」的**精确**前向次数：
                                                # 原文只给 `O(·)`（式 (1)）与「多一个完整前向」，
                                                # **没有**给分段策略的精确计数公式 ⇒
                                                # `probe_90b.py` 的 E2 那张表是**本库的推导**，
                                                # 不是引文；进正文时必须标「推导值」。
    "torch_clip_coef_eps_version_span",         # 上面的 `1e-6` 只在 **v2.9.0** 上核过；
                                                # 它在别的版本（本库别处是 2.6.0）上是否相同
                                                # **未逐版核对** ⇒ 引用时写出 v2.9.0。
)

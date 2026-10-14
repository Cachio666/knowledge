"""90-training/mixed-precision 的外部事实常量源。

本模块是 `CONTRIBUTING` §2 约定的**共享事实源**（`specs.py`）：本主题的三个编号程序从它读常量，
推导过程可复核，**常量本身不可自证**。

★★ **本模块最重要的约定：来源必须分级，并写明它是哪一级**（计划 §1.5）。
  本域的头号错误模式是 **P19 —— 把「某实现的文档」当成「规范的文档」**（B 级当 A 级）。
  ⇒ 本主题有一半的常量来自 **B 级（某实现的文档）**：
  `O0/O1/O2/O3`（**Apex** 的文档）、`GradScaler` 的四个默认值（**PyTorch**）、
  loss scale 的初值（**NVIDIA FAQ**）。**每一条都写明是哪个实现**，不得写成「规范规定」。

★ 三类真值来源**必须分开标注**（计划 §1.5）：
    ① **外部规格值**：论文原文（**C 级，须注明是否确认同行评审**）、厂商/产品文档（A/B 级）
       ⇒ 本模块存放的就是这一类，**每条带 `# 来源：` + `[A-E]` 级 + 抓取日**。
    ② **本机实测值**：本域程序在本机跑出的数（**须写 python3 与 numpy 版本**）⇒ 不放在本模块。
    ③ **推导值**：程序算出的量（可手算复核）⇒ **不放在本模块**
       （`CONTRIBUTING` §2：「只放从来源抄来的数，**不放推导值**」）。
       ★ 本主题据此**刻意不搬**的一条：**动态范围的三个数**（fp16 正规 `9.03` /
       含次正规 `12.04` / fp32 正规 `76.46`）是**从格式参数算出来的推导值**，
       它们属 `01_types.py` 的判据④，**不进本模块**——放进来只会变成「常量 == 常量」的恒真断言。

★ **只放本主题真的会断言的常量**（计划 §3 的 R3 裁定）。每一条下面写明**预期消费者**。
  ★ **没有消费者的常量就是 `make mutate` 的漏网候选**（计划 §4 **T11 的盲区①**）。

★ 每条常量带 `# 来源：` + `[A-E]` 级 + 抓取日；跨主题/跨文件同值的另带 `# 同值位置：`。
  分级判据见 `CONTRIBUTING` §2（A 厂商规范 / B 产品官方文档 / C 论文 / D 社区 / E 本库推导）。

★★ **量法口径的四个现场**（本模块的每条引文都按它办；取自本域 §0.3 与计划 §4 的 T18/T20）：
  ① **跨标记**：Apex 的 HTML 里 `O0 : FP32 training` 计 **0**，而 `O0` 计 9、`FP32 training` 计 2
     —— `O0` 被包在 `<span class="pre">` 里，**跨标记的引文在原始字节上必然计不到**。
  ② **标点后的空格数**：Apex 的 raw HTML 里是 `O0</code>:  FP32 training`（**两个**空格）。
  ③ **跨行空白**：引文片段**不能含换行**（fp8 论文里 `E4M3 for weight and activation tensors`
     能数到，是因为它在同一行）。
  ④ **整个载体缺失**（本主题新踩到的一处，见 §1 抬头）：`Published as a conference paper at
     ICLR 2018` 在 **arXiv abs 页**计 1、在 **ar5iv 正文**计 **0** —— 两个**载体**，
     「在 A 处数不到」**不等于**「它不存在」。此类否定性结论**必须配对照**。
"""

# ===========================================================================
# 1. 格式参数与数值边界 —— [A] IEEE 754-2019 的位参数（不是某实现的规格）
# ===========================================================================
#
# 来源：[A] **IEEE Std 754-2019** 的 `binary16` / `binary32` 交换格式定义。
#       ★ 引用的是**标准**（A 级），不是某个实现的 `finfo` —— 所以「按定义算」与
#         「读 `numpy.finfo`」是**两条独立的路**（`01_types.py` 的 `A.2` 拿它们对拍）。
#       ★★ fp8 的两个变体**不在本节**：它们的位分配与四个变体是
#         `130-model-optimization/quantization/01-fp8-family.md` §0 的**正文**，
#         本主题只在 §5 里按「格式参数」取用并回链（见该节抬头）。
# 抓取：本模块 2026-09-23 打开过 `https://arxiv.org/abs/1710.03740` 与
#       `https://arxiv.org/abs/2209.05433` 的 abs 页，`<title>` 逐字分别 =
#       `[1710.03740] Mixed Precision Training`、`[2209.05433] FP8 Formats for Deep Learning`
#       ⇒ **P21 已办**（编号与标题对得上）。

F16_EXP_BITS = 5
F16_MAN_BITS = 10
F16_BIAS = 15
# 来源：[A] IEEE 754-2019 `binary16`。★ 这三个是**位参数**，由它们算出下面三个边界值。
# 消费者：`01_types.py` 的 `A.2`（用它按定义算出 `max / 最小次正规 / 最小正规`，
#        再与 `numpy.finfo` 对拍 —— 两条独立路径）。
# 同值位置：无（2026-09-23 全库扫描 `.md`/`.py`，别处未以常量形态出现）。

F32_EXP_BITS = 8
F32_MAN_BITS = 23
F32_BIAS = 127
# 来源：[A] IEEE 754-2019 `binary32`。
# 消费者：`01_types.py` 的 `A.3`（fp32 正规区间 = 推导值，它的位参数在这里）。
# 同值位置：无。

BF16_EXP_BITS = 8
BF16_MAN_BITS = 7
BF16_BIAS = 127
# 来源：[A] 惯例 —— bf16 不是 IEEE 交换格式，它的**指数位与 fp32 相同**是
#       Google 的 bfloat16 定义（本库先例：`130-model-optimization/quantization/03-bf16-fp16-tf32.md`）。
#       ★ 标 [A] 的理由：它是一个**格式定义**，不是一个产品的默认值；但它**不是** IEEE 标准里的格式。
# 消费者：`01_types.py` 的 `A.4`（判据⑤：bf16 的动态范围与 fp32 几乎相同）。
# 同值位置：`130-model-optimization/quantization/03-bf16-fp16-tf32.md` 的正文（散文形态，不是常量）。

F16_MAX_FINITE = 65504.0
# 来源：[A] IEEE 754-2019 `binary16` 的最大有限数 = `(2 − 2^-10)·2^15`。
#       ★ 计划 §3 的 `specs.py` 清单把它列为本主题该断言的常量之一（判据③）。
# 口径：**最大有限数**（不是「归零下限」，也不是「动态范围」）。
# 消费者：`01_types.py` 的 `A.2`（判据③，与 `numpy.finfo(np.float16).max` 逐位比）、`A.3`；
#         `02_amp.py` 的 `B.1`（scaling 的上溢边界：`×S` 后不许越过它）。
# 同值位置：`90-training/probes/probe_90.py` 的 `F16_MAX`（探测脚本，非编号程序）。

F16_MIN_SUBNORMAL = 5.960464477539063e-08
# 来源：[A] IEEE 754-2019 `binary16` 的最小**次正规**数 = `2^-24`。
#       ★ `2^-24` 这个幂次是 Micikevicius 2017 逐字用的那一个（见 `MICIKEVICIUS_ZERO_BELOW_QUOTE`）。
# 口径：**最小次正规**（= 量化器的**一格**），**不是**「归零阈值」——
#       舍入到最近时的归零阈值是它的**一半**（`2^-25`），见 `01_types.py` 的 `A.5`。
# 消费者：`01_types.py` 的 `A.2`（判据③）、`A.5`（判据⑥的 `thr`）。
# 同值位置：`90-training/probes/probe_90.py` 的 `F16_TINY`（注意那个**名字是错的**：
#           它取的是 `np.nextafter(fp16(0), 1)`，即最小次正规，不是 `finfo.tiny`）。

F16_MIN_NORMAL = 6.103515625e-05
# 来源：[A] IEEE 754-2019 `binary16` 的最小**正规**数 = `2^-14`。
# 口径：★ 与 `F16_MIN_SUBNORMAL`（`2^-24`）**不是同一个量** —— 两者相差 10 个二进幂。
#       库内先例已经逐字警告过这件事：`130-model-optimization/quantization/03-bf16-fp16-tf32.md:154`
#       （`2^-126` vs `2^-14`「**两个数不是同一个量，不要混用**」）。
# 消费者：`01_types.py` 的 `A.3`（判据④：fp16 **正规区间**的上/下限）。
# 同值位置：无。

MICIKEVICIUS_WEIGHT_TO_UPDATE_RATIO_MIN = 2048
# 来源：[C] Micikevicius et al., **`Mixed Precision Training`**，arXiv **1710.03740**。
# 评审状态：★★ **已确认同行评审** —— 判据是**实测**：该文 arXiv **abs 页**的 `Comments:`
#   字段逐字 = `Published as a conference paper at ICLR 2018`（原始字节 1 次）。
#   ⚠ **量法口径（本模块新踩到的第四个现场）**：同一句话在 **ar5iv 的正文 HTML** 里计 **0** 次
#     （`raw.count` 实测）—— **两个载体**。「在正文里数不到」**不等于**「它不存在」；
#     本模块据 abs 页判它已过评审，并把「在哪个载体上数的」写出来。
# 逐字（原始字节 1 次）：`This can happen when the magnitude of a normalized weight value is at
#   least 2048 times larger that of the weight update.`
#   ★ 紧接着的机制句（原始字节 1 次，同一行内、不含换行）：
#     `Since FP16 has 10 bits of mantissa, the implicit bit must be right-shifted by 11 or more
#      positions to potentially create a zero`
# 口径：★★ **「至少 2048 倍」是一个下界**，它对应「右移 11 位」即 `2^11`。
#       实测的临界比值 `|w| / Δw_crit` 落在 `[2048, 4096)` —— 取到 2048 **只在 `w` 恰为 2 的幂时**。
# 消费者：`01_types.py` 的 `B.1`（用本机二分实测的临界比值与这个外部锚点对齐）。
# 同值位置：无。

MICIKEVICIUS_ZERO_BELOW_QUOTE = "any value whose magnitude is smaller than"
# 来源：[C] Micikevicius 2017（同上）。逐字（原始字节 1 次）：
#         `updates (weight gradients multiplied by the learning rate) become too small to be
#          represented in FP16 - any value whose magnitude is smaller than 2^{-24} becomes zero in FP16`
#       ⚠ **整句含公式，数不到**（`2^{-24}` 被 `<math>` 包着、且在 ar5iv 里渲染成两份
#         `2−242^{-24}`）⇒ 这里只取**公式之外的连续片段**（本常量），
#         公式值 `2^-24` 另存为 `F16_MIN_SUBNORMAL`。
# 口径：★ 这是论文对「主权重为什么需要」的**第一条理由**（更新量被吞），
#       与第二条理由（权值/更新量比值 ≥ 2048）**是两个理由**，不要合成一条。
# 消费者：`01_types.py` 的 `part_D`（与 `F16_MIN_SUBNORMAL` 一起逐位钉住）。
# 同值位置：无。

MICIKEVICIUS_UNSCALE_QUOTE = \
    "It is simplest to perform this unscaling right after the backward pass but before gradient clipping"
# 来源：[C] Micikevicius 2017（同上），§3.2。逐字（原始字节 1 次）：
#         `It is simplest to perform this unscaling right after the backward pass but before
#          gradient clipping or any other gradient-related computations`
#       ★ 上句同时给了**理由**（原始字节 1 次）：`ensuring that no hyper-parameters (such as
#         gradient clipping threshold, weight decay, etc.) have to be adjusted`
# 口径：★★ **这就是「unscale 的时机」的唯一一手依据**（判据表 B）。
#       `02_amp.py` 的 `A.2` 用**两条顺序**的对照把它变成可断言的：顺序错了会**多裁 `S` 倍**。
# 消费者：`02_amp.py` 的 `A.2` 与 `part_D`（逐位钉住）。
# 同值位置：无。

MICIKEVICIUS_SCALE_FACTORS_TRIED = (8, 32768)
# 来源：[C] Micikevicius 2017（同上），§3.2。逐字（原始字节 2 次 —— 另一次是 Introduction）：
#         `We trained a variety of networks with scaling factors ranging from 8 to 32K`
#       ★ `32K` 按该文的用法是 `32000`；本常量取 **2 的幂口径 `32768`** 并**在正文里写明这个换算**，
#         因为程序里要对它做 2 的幂运算。⚠ 这是本模块**唯一**一处「按上下文换算」的常量。
# 口径：**实验里试过的倍率范围**（不是「推荐值」，也不是「上界」）。
# 消费者：`02_amp.py` 的 `B.2`（本机实测的可用窗口与这个范围对照）。
# 同值位置：无。

MICIKEVICIUS_OVERFLOW_SKIP_QUOTE = "One option is to skip the weight update when an overflow is detected"
# 来源：[C] Micikevicius 2017（同上），§3.2。逐字（原始字节 1 次）。
#       ★ 上句给了溢出**为什么必须处置**（原始字节 1 次）：
#         `overflows will result in infinities and NaNs in the weight gradients which will
#          irreversibly damage the weights after an update`
# 口径：★ 原文的措辞是 **`One option is`** —— 它是**一种**处置，不是唯一处置。
#       转述成「必须跳过」就是把选项升格成规定（**P20 的同类**）。
# 消费者：`02_amp.py` 的 `C.1`（阴性对照：不处置 ⇒ 权重被 inf 污染且不可恢复）。
# 同值位置：无。

# ── 本节**刻意不搬**的一条 ────────────────────────────────────────────────────
#   · `single-precision master weights and updates, loss-scaling, and accumulating FP16 products
#     into FP32`（§3 开头的三项技术清单，逐字 1 次）：它是**散文**，由文档引用、由人核对；
#     搬成模块级常量只会得到一个没有消费者的字符串（T11 盲区①）。


# ===========================================================================
# 2. ★★ Apex 的 `O0/O1/O2/O3` —— **B 级：某实现的文档，不是厂商规范**（P19 的现场）
# ===========================================================================
#
# ★★ 这一节是**本主题头号错误模式的现场**：
#   · 控制者的任务书让研究者去查「**NVIDIA 的 AMP 文档**」拿 `O0/O1/O2/O3` 的定义；
#   · 实测：**NVIDIA 的 AMP 页里 `whitelist` / `blacklist` / `master_weights` 各出现 0 次**
#     （该页只有一段 FAQ 提到 O0–O3，`O0` 计 2 次）；
#   · 完整描述在 **Apex 文档**：`whitelist` 1 次、`blacklist` 1 次、`master_weights` **9** 次。
#   ⇒ 引用 `O0–O3` **必须标 B 级 + 实现名（Apex）**，**不得**写成「NVIDIA 规定」。
#   ⚠ **两个 URL 都要留痕**（否定性结论必须配对照 —— 见模块抬头 ①②）。

APEX_OPT_LEVELS = ("O0", "O1", "O2", "O3")
# 来源：**Apex** 的 AMP 文档  [B]（**某实现的文档**，不是规范）
#       URL `https://nvidia.github.io/apex/amp.html`，抓取 2026-09-23，
#       **64968 B**，`sha256[:16] = 7484b043bf91e1b5`。
#       ★ 对照（同一个事实的另一个载体）：NVIDIA 的 AMP 页
#         `https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html`，
#         481333 B，`sha256[:16] = 690a696dbc995434`；该页 `whitelist` = **0**、`master_weights` = **0**。
# 口径：四个 **opt_level 的名字**。★ 这四个标题在 Apex 的 **raw HTML** 里同样**数不到**
#       （各 0 次，剥标签后才出现）—— **Y15 的坑不止 `O0` 一条**（计划 §4 **T18**）。
# 消费者：`02_amp.py` 的 `part_D`（四个标题逐位钉住；★ **不做 `raw.count`**，
#        `raw.count` 的活由文档的「参考」节与人工核对承担 —— 程序不做网络 I/O）。
# 同值位置：无。

APEX_O1_MASTER_WEIGHTS = None
# 来源：**[B] Apex** 文档（同上）。逐字（**剥标签后**计数 1 次；原始字节里
#       `master_weights=None (not applicable, model weights remain FP32)` 因跨 `<code>` 标记而计 0）：
#         `master_weights=None (not applicable, model weights remain FP32)`
# 口径：★★ **`None` 的含义是「不适用」，不是「没开」**——同段逐字还有
#       `keep_batchnorm_fp32=None (again, not applicable, all model weights remain FP32)`。
#       ⇒ **「O1 会额外维护一份 fp32 主权重」是错的**（O1 下权重**本来就是** fp32）。
#       ★ 用 `None` 而不是 `False`：`False` 与「不支持」不可区分，而原文说的是**不适用**。
# 消费者：`02_amp.py` 的 `part_D`（与 `APEX_O2_MASTER_WEIGHTS` 组成对照：`(None, True)`）。
# 同值位置：无。

APEX_O2_MASTER_WEIGHTS = True
# 来源：**[B] Apex** 文档（同上）。逐字（剥标签后 1 次）：
#         `O2: cast_model_type=torch.float16 patch_torch_functions=False keep_batchnorm_fp32=True
#          master_weights=True loss_scale="dynamic"`
# 口径：★ **只有 O2 才是 `master_weights=True`** —— 这是 W10/`APEX_O1_MASTER_WEIGHTS` 的对照物。
# 消费者：`02_amp.py` 的 `part_D`（`(APEX_O1_MASTER_WEIGHTS, APEX_O2_MASTER_WEIGHTS) == (None, True)`）。
# 同值位置：无。

APEX_O1_MASTER_WEIGHTS_QUOTE = "master_weights=None (not applicable, model weights remain FP32)"
# 来源：**[B] Apex** 文档（同上）。**剥标签后**逐字 1 次（原始字节里因跨 `<code>` 标记而计 0）。
# 口径：★★ 这是 **Y10 的逐字证据本身**（不是它的转述）—— 与 `APEX_O1_MASTER_WEIGHTS = None`
#       **成对**出现：那个 `None` 是这条引文**读出来的值**。
#       ★ 为什么要**两条**：`None` 在 `make mutate` 的 `_mutate_value()` 下**给不出另一个值**
#         （计入「跳过」）⇒ 只靠 `None` 这一个叶子，Y10 在变异测试里是**免检**的。
#         带上这条逐字字符串，Y10 就有了一个**会被变异到**的落点。
# 消费者：`02_amp.py` 的 `part_D`（`D.7c`：与 `"master_weights=None (not applicable"` 逐位比对）。
# 同值位置：无。

APEX_LOSS_SCALE_MAX = 16777216.0
# 来源：**[B] Apex** 文档（同上），`apex.amp.initialize(...)` 的签名。逐字（剥标签后 1 次）：
#         `max_loss_scale=16777216.0`
# 口径：`16777216 = 2^24`。★ 与 `NVIDIA_FAQ_INIT_SCALE` **数值相同但角色不同**：
#       这里是 `initialize()` 的**上界**，那里是动态缩放的**初值**。
#       ⚠ 两者**不是同一个量**（一个上界、一个初值）⇒ **不要因为相等就合并成一个常量**。
# 消费者：`02_amp.py` 的 `part_D`（与 `NVIDIA_FAQ_INIT_SCALE` 一起钉住，并断言
#        「两个数相等、但一个叫 max 一个叫 init」这件事在正文里写明了）。
# 同值位置：与 `NVIDIA_FAQ_INIT_SCALE` 数值同、**角色不同**（本行因此不写 `# 同值位置：`）。


# ===========================================================================
# 3. loss scale 的**两个官方初值** —— 都是 [B]，且**不得合成一个**
# ===========================================================================
#
# ★★ 计划 §0.7-C 的 **Y11**：这一条的正确写法是「**有两个官方数**」，不是「初值取 X」。
#    合成一个数会让读者以为存在唯一规定。

PYTORCH_GRADSCALER_INIT_SCALE = 65536.0
# 来源：**PyTorch（`torch.cuda.amp.GradScaler`）的文档**  [B]（**某实现的文档**）
#       URL `https://docs.pytorch.org/docs/2.14/amp.html`，抓取 2026-09-23，
#       **754077 B**，`sha256[:16] = df5bb3e3367f1ee1`。
#       ★ **取法有坑**（计划 §4 **T20**）：`/docs/stable/amp.html` 返回的是 **1270 B 的
#         `<meta http-equiv="refresh">` 跳转壳**；**`curl -L` 跟的是 HTTP 重定向、跟不了 meta refresh**
#         ⇒ 正确做法是**读出 meta 里的目标 URL 再取一次**（本行取的就是 `2.14`）。
# 逐字（原始字节 1 次）：`class torch.cuda.amp.GradScaler(init_scale=65536.0, growth_factor=2.0,
#       backoff_factor=0.5, growth_interval=2000, enabled=True)`
# 口径：`65536.0 = 2^16` —— **一个实现的默认值**。
# 消费者：`02_amp.py` 的 `C.2`（判据：两个官方初值都钉住、且 `== 256` 倍，**不得合成一个**）。
# 同值位置：无。

NVIDIA_FAQ_INIT_SCALE = 16777216.0
# 来源：**NVIDIA 的 AMP FAQ**  [B]（厂商的**产品文档**，不是规范）
#       URL `https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html`，
#       抓取 2026-09-23，481333 B，`sha256[:16] = 690a696dbc995434`。
# 逐字（原始字节 2 次 —— 该段落在页面上出现两遍）：`beginning with a high loss scale value (say, 2^24)`
# 口径：`2^24 = 16777216`。★ 原文的量词是 **`say`** ⇒ 它是**举例**，不是规定（**P20 的同类**：
#       不要把举例升格成规定）。与 PyTorch 的 `2^16` 相差 **256 倍**。
# 消费者：`02_amp.py` 的 `C.2`。
# 同值位置：与 `APEX_LOSS_SCALE_MAX` 数值同、**角色不同**。

PYTORCH_GRADSCALER_BACKOFF = 0.5
PYTORCH_GRADSCALER_GROWTH = 2.0
PYTORCH_GRADSCALER_GROWTH_INTERVAL = 2000
# 来源：**PyTorch** 的 `GradScaler` 文档 [B]（同上 URL 与哈希）。逐字（原始字节各 4 次）：
#         `init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000`
# 口径：★ **动态 loss scaling 的三个规则参数**（`growth_interval` 的单位是 **step**）。
#       它们全是**某实现的默认值**，不是规范。
# 消费者：`02_amp.py` 的 `B.2`（一次溢出把 `S` 减半后，要**回到原值至少需要 2000 步** ——
#       这是「动态缩放的代价」那笔账的闭式）。
# 同值位置：无。

TORCH_CLIP_COEF_DENOM_EPS = 1e-6
# 来源：**PyTorch** 的 `torch/nn/utils/clip_grad.py`  [B]（**某实现的源码**，不是规范）
#       版本：**v2.9.0**；抓取 `https://raw.githubusercontent.com/pytorch/pytorch/v2.9.0/
#       torch/nn/utils/clip_grad.py`，2026-09-23，11427 B，`sha256[:16] = 63efb273c62fd917`
#       逐字（原始字节 2 次）：`clip_coef = max_norm / (total_norm + 1e-6)`
# 口径：**分母上的那个 `1e-6`**（`total_norm = 0` 时的保护项）
#       ⇒ 裁剪后的全局范数 = `max_norm · T/(T + 1e-6)`，**相对误差 = `1e-6/T`**。
#       ★★ **本主题只消费这个口径，不重新定义它**：裁剪的**判据与阈值**在
#       `90-training/autograd-and-backward/03-gradient-lifecycle.md`（**在那里只写一次**，
#        连 `T ≥ 1` 的取值范围也钉在彼处）。
# 消费者：`02_amp.py` 的 `A.2`（`unscale → clip` 与 `clip → unscale` 的对照要用到它）
#       与 `preflight()` 的钉住。
# 同值位置：`90-training/autograd-and-backward/code/specs.py` 的 `TORCH_CLIP_COEF_DENOM_EPS`
#       （★ **另一份拷贝** —— `CONTRIBUTING:36`「`code/` 必须自包含」，**不设跨主题共享模块**；
#        跨主题同值的常量各写一份，靠本行互指）；
#       `90-training/probes/probe_90c.py` 的 `clip_global()`（**探测脚本**，非编号程序）。
#       ⚠ **未核**：本库别处的 torch 是 2.6.0；`+1e-6` 在不同版本间是否恒定**未逐版核对**
#         ⇒ 引用时**写出 v2.9.0**，不要写成「PyTorch 规定」。


# ===========================================================================
# 4. fp8：**训练循环里的分工与 scaling**（不是位分配）
# ===========================================================================
#
# ★★ 本节是**收窄原则的落点**：`03-fp8-training.md` **只写训练循环里怎么用 fp8**，
#    **位域与 scale 粒度一律回链** `130-model-optimization/quantization/01-fp8-family.md` §0
#    与 `02-e8m0-and-mx.md`（计划 §2 的边界表，结构层审查 **M1**）。
#    ⇒ 本节**没有**「四个变体」「±448 vs ±57344」「按 torch 版本的溢出差异」这类常量；
#      只有**训练循环要用到的三样**：分工、scaling 是不是格式的一部分、格式参数。

FP8_FORMATS = {
    # 格式名 -> (最大有限数, 最小正规数, eps)
    # ★★ **为什么写成十进制字面量而不是 `2.0 ** -6`**（本目录实测的一个工具约束）：
    #    `mutate.py` 的 `_walk()` **只下钻到字面量与容器**，**不下钻 `BinOp`** ⇒
    #    写成 `2.0 ** -6` 时，`2.0` 与 `-6` **根本不是分母里的叶子** ——
    #    「把 `-6` 改成 `-5`」**不会被变异到、也没有任何断言会发现**（本程序实测：
    #    改 `eps` 会让 `eps/2` 与量化器**同步**变化 ⇒ `A.3` 照样过）。
    #    ⇒ **幂次写在注释里**，值写成**字面量**，这样四个值都进分母。
    "e4m3fn": (448.0, 0.015625, 0.125),           # 最小正规 2^-6 · eps 2^-3
    "e5m2": (57344.0, 6.103515625e-05, 0.25),     # 最小正规 2^-14 · eps 2^-2
}
# 来源：[C] Micikevicius et al. 2022，arXiv **2209.05433**（ar5iv **85378 B**，
#       `sha256[:16] = a24bcf90ca36746f`）。**不确认同行评审**（判据是实测：abs 页
#       `Comments:` = **0** 次、`journal-ref` = **0** 次）。
#       Table 1 逐字（剥标签后）：`Max normal S.1111.110_2=1.75*2^8=448`、
#       `Max normal S.11110.11_2=1.75*2^15=57,344`、`Min normal ... =2^-6`、
#       `Min normal ... =2^-14`；eps 由尾数位算（`2^-3` / `2^-2`）。
#       ⚠ **量法口径**：原文里 `57,344` **带千位逗号** ⇒ `raw.count("57344")` = **0**
#         而 `raw.count("57,344")` = 1。本常量写的是**数值**，函数段里不写引文。
# 口径：★ 这只是**格式的三个可表示性参数**（最大有限数 / 最小正规 / eps），
#       **不是位分配** —— 「这 8 位怎么分、四个变体差在哪」的正文在
#       `130-model-optimization/quantization/01-fp8-family.md` §0，**此处回链**。
#       ★ 本库已核对同一组值（内部对照，不作外部来源）：`01-fp8-family.md` §0/§1。
# 消费者：`03_fp8.py` 的 `A.4`（量化器的最大相对误差 ≤ `eps/2`）、`B.1`、`C.2`。
# 同值位置：`130-model-optimization/quantization/01-fp8-family.md` §0 的正文（散文形态）。

FP8_DIVISION_QUOTE = "E4M3 for weight and activation tensors, and E5M2 for gradient tensors"
# 来源：[C] 2209.05433，§3。逐字（原始字节 1 次，同一行内）：
#         `The recommended use of FP8 encodings is E4M3 for weight and activation tensors, and
#          E5M2 for gradient tensors.`
# 口径：★★ 这是**分工**本身。★ 但它的**依据**是引用来的（见下一条）——
#       本文自己**没有**给证明，只说「consistent with findings in [20, 16]」。
# 消费者：`03_fp8.py` 的 `A.1`（编码表与它逐位比对）+ `part_D`。
# 同值位置：无。

FP8_DIVISION_IS_CITED_QUOTE = "This is consistent with findings in"
# 来源：[C] 2209.05433（同上）。逐字（原始字节 1 次）：
#         `This is consistent with findings in [20, 16], where inference and forward pass of
#          training use a variant of E4M3, gradients in the backward pass of training use a
#          variant of E5M2.`
# 口径：★★ 计划 §0.7-C 的 **Y8**：**「fp8 论文证明前向用 E4M3、反向用 E5M2」是错的** ——
#       那是它**引用别人的结论**；本文自己的措辞只有 `FP8_DIVISION_QUOTE` 那一句
#       （`The recommended use ...`）。⇒ 正文不得写「论文证明」。
# 消费者：`03_fp8.py` 的 `A.1`（把「分工是引用来的」写成断言，而不是写成结论）。
# 同值位置：无。

FP8_SOFTWARE_SCALING_QUOTE = "Leaving the per-tensor scaling to software implementation"
# 来源：[C] 2209.05433，§3.2。逐字（原始字节 1 次）：
#         `Leaving the per-tensor scaling to software implementation enables more flexibility than
#          is possible with a programmable exponent bias approach - the scaling factor can take on
#          any real value (typically represented in higher precision), while programmable bias is
#          equivalent to allowing only powers of 2 as scaling factors.`
# 口径：★★ 计划 §0.7-C 的 **Y7**：**per-tensor scaling 是软件层，不是格式的一部分**。
#       原文还给了理由：**可编程 bias 只能表达 2 的幂次 scale，软件层可以取任意实数**。
# 消费者：`03_fp8.py` 的 `A.2`（逐位钉住 + 机械判据）、`A.3`。
# 同值位置：无。

FP8_BEYOND_SCOPE_QUOTE = "beyond the scope of this paper"
# 来源：[C] 2209.05433，§3.2。逐字（原始字节 1 次）：
#         `Details of the heuristics to select the scaling factors are beyond the scope of this paper`
# 口径：★★ 计划 §4 的 **T17**：**`just-in-time` 与 `delayed scaling` 在本文里各出现 0 次**
#       （本模块 2026-09-23 在原始字节上复核：**都是 0**）。⇒ 那两个术语**不属于本文**，
#       它们的可引来源是 **FP8-LM 的附录 A.2**（见 §5）或 NVIDIA Transformer Engine 的文档。
# 消费者：`03_fp8.py` 的 `A.2`（「本程序不把 scaling 的启发式写成论文的规定」）。
# 同值位置：无。

FP8_OVERFLOW_NOT_SKIP_QUOTE = "is not a good choice for FP8"
# 来源：[C] 2209.05433，§3.2。逐字（原始字节 1 次）：
#         `Weight update skipping (and reduction of the scaling factor) on overflows, as used by
#          FP16 automatic mixed precision training [13], is not a good choice for FP8 as overflows
#          are much more likely due to the narrower dynamic range, resulting in too many skipped
#          updates.`
# 口径：★★ **与 fp16 相反的溢出处置** —— fp16 的「溢出就跳步 + 降 scale」**不适用于 fp8**。
#       这是 `03` 与 `02` 的正面接口（`02` 写 fp16 的处置、`03` 写为什么它搬不过去）。
# 消费者：`03_fp8.py` 的 `C.1`（定量复现「跳步太多」：跳步率与闭式 `1/(k+1)` 对拍）。
# 同值位置：无。

FP8_SIMULATED_QUOTE = "simulated FP8"
# 来源：[C] 2209.05433，§4。逐字（原始字节 1 次）：
#         `Training experiments were carried out with simulated FP8 - tensor values were clipped to
#          only those that could be represented in FP8`
#       ★ 同段还给了算术在哪里做（原始字节 1 次）：`Arithmetic was carried out using the wider
#         representation`。
# 口径：★★ **这是本主题最大的方法学让步**：论文的训练实验是**模拟 fp8** ——
#       **算术仍在宽类型上做**，只有**存储/输入**被量化。
#       ⇒ `03_fp8.py` 的实现是**同一个形态**，因此它**不得**支撑任何关于「真 fp8 算术」
#         （吞吐、累加顺序、Tensor Core）的结论。
# 消费者：`03_fp8.py` 的 `part_D`（本程序在 `main` 里打印这条限定语；常量用于逐位钉住）。
# 同值位置：无。

FP8_SCALE_SCOPE = "per-tensor"
FP8_SCALE_BITS = 32
# 来源：作用域取自 [C] 2209.05433 的 `per-tensor scaling`（原始字节 6 次，本模块用其中一致的含义）；
#       **bit 开销是本程序的记账**（不是引文）：per-tensor scale 在软件层是**每张量 1 个 fp32**，
#       即 **32 bit / 张量**，**与张量元素数无关**。
# 口径：★★ **库内硬要求**（`130-model-optimization/quantization/02-e8m0-and-mx.md:276` 逐字）：
#       「**凡是写「scale」这个词的地方，同时写出它的作用域（每张量 / 每通道 / 每 token /
#       每 32 个元素的块）与它的 bit 开销**」。
#       ⇒ 本主题**凡写到 scale 处一律带「per-tensor + 32 bit/张量」**；
#       **其它粒度（每通道 / 每 token / 每 32 元素的块）的正文在**
#       `130-model-optimization/quantization/06-granularity.md` 与 `02-e8m0-and-mx.md`，**本主题回链、不展开**。
# 消费者：`03_fp8.py` 的 `A.3`（断言开销与元素数无关）。
# 同值位置：无。


# ===========================================================================
# 5. `just-in-time` / `delayed scaling` —— ★★ **来源只在这里**（计划 §4 的 T17）
# ===========================================================================
#
# ★★ 为什么单列一节：这两个术语**在 fp8 论文（2209.05433）里各出现 0 次**，
#    论文自己逐字说 `Details of the heuristics ... are beyond the scope of this paper`
#    （原始字节实测：`just-in-time` = **0**、`delayed scaling` = **0**）。
#    ⇒ 把它们写成 2209.05433 的内容 = **又一次「把实现文档当规范」**（**P19**）。
#    它们的可引定义在 **FP8-LM 的附录 A.2**（本节），而 FP8-LM 自己把这两个术语的
#    出处标成 `(Nvidia 2022c)` —— **即 NVIDIA Transformer Engine 那一层**。

FP8_LM_JUST_IN_TIME_QUOTE = \
    "determining the scaling factor based on the maximum absolute value (amax) of the tensor being generated"
# 来源：[C] **FP8-LM**（arXiv **2310.18313**，ar5iv **266806 B**，
#       `sha256[:16] = a184329665e9ad2b`；**不确认同行评审**：abs 页 `Comments:` = 0），
#       **附录 A.2**。逐字（原始字节 1 次）：
#         `Just-in-time scaling. This strategy involves determining the scaling factor based on the
#          maximum absolute value (amax) of the tensor being generated.`
#       ★ 同段的**代价**逐字（原始字节 1 次）：
#         `this approach is often infeasible because it necessitates multiple passes through the data`
# 口径：★ 本常量是**定义句的后半**（前半 `This strategy involves ` 是行首，
#       合起来才是完整定义句 —— 切在中间是为了让片段**不含换行**，库内纪律见模块抬头③）。
# 消费者：`03_fp8.py` 的 `B.1`（本程序用它实现「just-in-time」策略）、`part_D`。
# 同值位置：无。

FP8_LM_DELAYED_QUOTE = \
    "selecting the scaling factor based on the maximum absolute values observed in a certain number of preceding iterations"
# 来源：[C] FP8-LM 附录 A.2（同上 URL 与哈希）。逐字（原始字节 1 次）：
#         `Delayed scaling. This strategy involves selecting the scaling factor based on the maximum
#          absolute values observed in a certain number of preceding iterations.`
#       ★ 同段的**代价**逐字（原始字节 1 次）：
#         `necessitates the storage of a history of maximum values as additional parameters of the
#          FP8 operators`
# 口径：★★ **历史长度 `k` 由实现定**（定义句只说 `a certain number of`）⇒ 本程序里的 `k`
#       是**本程序的选择**，不是引文；因此 `1/(k+1)` 那条闭式也是**本程序的推导**（`03` 的 `B.1`）。
# 消费者：`03_fp8.py` 的 `B.1`、`C.1`。
# 同值位置：无。


# ===========================================================================
# 6. 有来源、但**不作模块级常量**的条目（★ 理由：没有消费者 ⇒ 只会成为漏网）
# ===========================================================================
#
# ★★ 计划 §4 **T11 的盲区①**：`80` 实测「只用于打印的模块级常量表」制造过 **2** 条漏网。
#    对策是「要么给它消费者、要么删掉」。下面这几条**有来源但无消费者**，因此**写在注释里**：
#    它们由**文档**引用、由**人**核对（判据表的「引用判据」本来就是人工核对的口径）。
#
# (a) **`O0/O1/O2/O3` 的四个属性串**（各自一条，都来自 [B] Apex 文档，**剥标签后**各 1 次）：
#       `O0: cast_model_type=torch.float32 patch_torch_functions=False
#             keep_batchnorm_fp32=None (effectively, "not applicable," everything is FP32)
#             master_weights=False loss_scale=1.0`
#       `O1: cast_model_type=None (not applicable) patch_torch_functions=True
#             keep_batchnorm_fp32=None (again, not applicable, all model weights remain FP32)
#             master_weights=None (not applicable, model weights remain FP32) loss_scale="dynamic"`
#       `O2: cast_model_type=torch.float16 patch_torch_functions=False keep_batchnorm_fp32=True
#             master_weights=True loss_scale="dynamic"`
#       `O3: cast_model_type=torch.float16 patch_torch_functions=False keep_batchnorm_fp32=False
#             master_weights=False loss_scale=1.0`
#     去向：`02-amp-execution.md` 的**表 B（引用判据）**逐字引用 + 标 [B] + 实现名（Apex）。
#     ⚠ 程序只钉住 `APEX_OPT_LEVELS`（四个名字）与两个 `master_weights` 值 ——
#       四个**属性串**不搬成常量，因为它们的消费者是**人的核对**，不是断言。
#
# (b) **`loss scale` 的两个初值的完整句**：
#       `beginning with a high loss scale value (say, 2^24)`（[B] NVIDIA FAQ，原始字节 2 次）；
#       `class torch.cuda.amp.GradScaler(init_scale=65536.0, ...)`（[B] PyTorch 2.14，1 次）。
#     去向：`02-amp-execution.md` 的**表 B**；程序只钉住**两个数值**与它们的**比值 256**。
#
# (c) **Transformer Engine 的 fp8 文档**（[B]，计划 §C.2b 的 **T17** 兜底来源）：
#       URL `https://nvidia.github.io/TransformerEngine/`（★ **路径大小写敏感**：
#       小写的 `transformer-engine/` 会 404 —— 本模块 2026-09-23 实测：大写路径 **200**，
#       41908 B，`sha256[:16] = 6064f579f64290e9`；该页 `delayed scaling` **1** 次、
#       `current scaling` **1** 次、`just-in-time` **0** 次）。
#       ⇒ 它**证实**了那两个术语属于 TE 那一层，但**没有**给出逐字定义
#         ⇒ 本主题的定义句一律引 **FP8-LM 附录 A.2**（§5），TE 文档只作为**层属**的旁证。


# ===========================================================================
# 7. UNSOURCED —— 待办清单，不是免责声明
# ===========================================================================
#
# ★ `CONTRIBUTING` §2：「不得**既保留数字又记 UNSOURCED**」—— 下面每一项都**没有**出现在
#   本模块的正文常量里。
# ★★ 提醒（T11 盲区①）：本元组的字符串**没有消费者** ⇒ 它们会是 `make mutate` 的**漏网**候选。
#    这是本库的既有形态（先例：`80-model-architectures/transformer/code/specs.py` 同节、
#    `90-training/autograd-and-backward/code/specs.py` 同节），三个编号程序**各钉一次**
#    （`part_D` 里逐位比较），**不给它开后门**。

UNSOURCED = (
    # 每一项是一个**稳定标识符**（不是散文）。
    "bf16_dynamic_range_exact_equality_claim",   # 「bf16 的动态范围与 fp32 **精确相同**」这句话
                                                # **只在 4 位小数上成立**：实测比值 = 0.9999777…，
                                                # 它的来源是 `probe_90b.py` 用 `{:.4f}` 打印
                                                # ⇒ 见 `01_types.py` 的 `A.4` 与计划 §4 的新残留。
                                                # **本主题不把这个比值写成常量**（它是推导值）。
    "torch_e4m3fn_overflow_rule_version_span",  # `float8_e4m3fn` 的溢出语义（NaN / 饱和）**随 torch 版本变**
                                                # （2.6.0/2.7.1/2.8.0/2.11.0 → NaN；2.13.0 → 448）。
                                                # 正文在 `130-model-optimization/quantization/01-fp8-family.md`
                                                # §0/§3 ⇒ **本主题不写、也不断言**（它属位分配那一层）。
    "fp8_scaling_growth_rule_concrete_values",  # `delayed scaling` 的**具体**历史长度、增长/回退规则
                                                # **没有**可引的一手来源（fp8 论文逐字说 beyond the scope；
                                                # FP8-LM 只说 `a certain number of preceding iterations`）
                                                # ⇒ 本程序里的 `k` 与规则是**本程序的选择**，标为推导值。
)

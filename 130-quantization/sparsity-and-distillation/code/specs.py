"""130-quantization/sparsity-and-distillation 的外部规格常量源。

本文件只放两类东西：
  1. **外部常量**（硬件规格、论文印值、判据公式），每条带一行 `# 来源：`；
  2. **本机实测基线表**（与三篇文档 §5 逐行一致的数字），每条注明测它的版本与判据。

★ 核查范围（错误模式 P6 的对策）——**本主题的来源档位逐条写明**：
  · 标 `（规格）` 的：来自**硬件/库的正式文档**（PTX ISA、cuSPARSELt）。
    本实现者用 `WebSearch` 检索到与计划基线一致的表述，**但未逐字读到原文**
    （本轮 `WebFetch` 被策略拦截、`curl` 被 hook 拦截，可用通道只有 `WebSearch`），
    所以**连章节号一起按「计划基线 + 检索佐证」标注**，不写成「已核实原文」。
  · 标 `（引用）` 的：论文印值。附**来源 / 表号 / 基线**三元组。
  · 标 `（本机实测）` 的：实现者在本机真 torch 上跑出来的数，附**判据与测量条件**。

★ 「规范」与「实现」分开标注（错误模式 P8）。本主题确证**一处**真冲突：
  **2:4 的 metadata 宽度**——PTX 把 2:4 的 metadata 描述成 `.b32` 寄存器里的
  **16 个 2-bit 向量**（每两个 2-bit 向量给一个 4 元素组，= 4 bit/组）；
  而 torch 的 `sparse_semi_structured_from_dense_cutlass` 对 **float32**
  压根不用 2:4——它用 `ksparse = 2`（**1:2**），metadata 是 8 bit/2 元素对。
  两者不是矛盾（1:2 是 tf32 那一档的合法粒度），但**只写一边会让读者以为
  「4 bit/组」对所有 dtype 都成立**。实测与文档的对照见 §1 与 §5。

★ 本文件**不放任何算法**，也**不得 import torch**（`verify.py` 在默认 python3 下 import 它）。
  凡被断言检验的算法（压缩率算术、判据归约、T² 标度）**一律在 `verify.py` 里另写**——
  理由见 CONTRIBUTING §2 的「参考实现必须独立」。

★ **A6 约定**：本文件里**每一条常量都被 `verify.py` 或某个编号程序的断言实际读过**。
  判据是机械的：逐个改掉一个值，`make run` 或 `make verify` **必须报错**。
"""

# ===========================================================================
# 1. 硬件粒度（规格值）
# ===========================================================================

HW_NM_GRANULARITIES = (
    ("1:2", ("tf32",),
     "每 2 个相邻元素 1 个非零；只有非零元素进存储"),
    ("2:4", ("f16", "bf16", "int8", "e4m3", "e5m2", "f8f6f4", "mxf8f6f4"),
     "每 4 个相邻元素 2 个非零"),
    ("4:8 paired", ("u4", "s4", "e2m1", "mxf4"),
     "每 8 个元素 4 个非零；零与非零**以 2 元素子块成簇**"),
)
# 来源：（规格）PTX ISA 的稀疏矩阵存储一节：
#   `.kind::tf32` → 1:2；f16/bf16/int8/e4m3/e5m2 → 2:4；mxf4/e2m1 → 4:8 (in pairs)。
#   ⚠ **档位**：计划基线引的是 **PTX ISA 9.4 §9.7.16.6.1**；
#   本实现者检索到的是 **PTX 8.7** 的同类表述（稀疏 `tcgen05.mma.sp` 与
#   `mma.sp` 的 fragment 说明），**该章节号未逐字核实**。按「计划基线 + 检索佐证」处理。
# 口径：**粒度 = 子块内非零数 / 子块大小**，子块大小由形状决定。**只有 N:M 形式**。

UNSTRUCTURED_IN_HW_GRANULARITY_TABLE = False
# 来源：（规格）上面那张表就是全部——PTX 的稀疏粒度**只定义 N:M 三种**，
#   非结构化（任意位置）**不在表里**。正文与 verify.py 都对这张表做穷举检查。

CUSPARSELT_50PCT_FORMAT_BY_DTYPE = {
    "float": "1:2",
    "half": "2:4",
    "bfloat16": "2:4",
    "int8": "2:4",
    "e4m3": "2:4",
    "e5m2": "2:4",
    "e2m1": "4:8 paired",
}
# 来源：（规格）cuSPARSELt Data Types 文档里 `CUSPARSELT_SPARSITY_50_PERCENT` 的
#   类型映射（检索佐证，未逐字读到原文）。
# 口径：**同一个「50% 稀疏」按 dtype 落到不同粒度**——float 走 1:2、e2m1 走 4:8、
#   其余走 2:4。这一条解释了 torch 的 float32 路径为什么是 1:2（见 §5 实测）。

# ===========================================================================
# 2. 压缩率的算术常量
# ===========================================================================

GROUP_SIZE_2_4 = 4
GROUP_SIZE_4_8 = 8
NONZERO_PER_GROUP_2_4 = 2
NONZERO_PER_GROUP_4_8 = 4
META_INDEX_BITS = 2              # 一个「非零位置」用一个 2-bit 索引
META_INDEXES_PER_GROUP_2_4 = 2   # 一组 2:4 有两个非零 ⇒ 两个索引
META_BITS_PER_GROUP_4_8 = 4      # 4:8 的 metadata：每 8 元素组 4 bit
# 来源：（规格）PTX 对 2:4 的 metadata 描述是「`.b32` 寄存器里 16 个 2-bit 向量，
#   每两个向量存一个 4 元素组里两个非零的索引」⇒ **4 bit/组**（检索佐证）。
#   4:8 的 4 bit/组同源（cuSPARSELt 的 paired 4:8 描述，检索佐证）。
# 口径：**这是「组」的定义**——2:4 的一组是 4 个元素，4:8 的一组是 8 个元素。
#   ⚠ 两个分母不要混（见 verify.py 的 A.2）：metadata 占**原始数据**是
#   `4/(4×w)`，占**压缩后载荷**是 `4/(2×w)`，w = 元素位宽。

ELEM_BITS = {"fp16": 16, "bf16": 16, "int8": 8, "int4": 4}
BASE_ELEM_BITS = 16
# 来源：（规格）元素位宽本身。**这不是「压缩率」**，只是算式的输入。

# 叠加压缩率：**bit/元素（含 metadata）**。判据 = 载荷 + 分摊到每个元素的 metadata。
COMBINED_BITS_PER_ELEM = {
    "fp16 dense": 16.0,
    "int8 dense": 8.0,
    "int4 dense": 4.0,
    "fp16+2:4": 9.0,      # (2×16 + 4)/4
    "int8+2:4": 5.0,      # (2×8  + 4)/4
    "int4+2:4": 3.0,      # (2×4  + 4)/4
    "int4+4:8": 2.5,      # (4×4  + 4)/8
}
NAIVE_MULTIPLE = {
    "fp16 dense": 1.0, "int8 dense": 2.0, "int4 dense": 4.0,
    "fp16+2:4": 2.0, "int8+2:4": 4.0, "int4+2:4": 8.0, "int4+4:8": 8.0,
}
# 来源：（规格 + 推导）载荷与 metadata 的位宽来自上面那几条规格常量，
#   组合方式是算术；`verify.py` 的 A.3 逐行重算并与本表核对。
# 口径：**这里的「组」与元素位宽一起决定 metadata 的贵贱**——
#   元素越窄，同一个 4-bit metadata 越贵（fp16 → 6.25%，int8 → 12.5%）。
#   `NAIVE_MULTIPLE` 是**朴素相乘**（稠密倍数 × 稀疏的 2×），用来量「朴素高估」。

# ===========================================================================
# 3. 非结构化的加速口径（GPU vs CPU）
# ===========================================================================

UNSTRUCTURED_GPU_MIN_SPARSITY_FOR_SPEEDUP = 0.95
# 来源：（引用）SparseGPT 论文对 cuSparse 的 EW 路径的说明——
#   要到 **>95% 稀疏**才开始快于稠密（计划基线 §0.7.G；**本实现者未逐字读到原文**）。
# 口径：**这是 GPU 上的路径**，不是「非结构化不能加速」的普适结论——见下一条。

DEEPSPARSE_OPT27B_CPU_SPEEDUP = ((0.40, 1.57), (0.50, 1.82), (0.60, 2.16))
# 来源：（引用）SparseGPT 附录 E 表 7，CPU（DeepSparse，OPT-2.7B）。
#   基线 = 稠密 DeepSparse；硬件 = Intel Core i9-7980XE / 18 核 / 单批 400 token。
#   **检索佐证确认了这三个数与硬件、基线**（未逐字读到原文）。
# 口径：**非结构化在 CPU 上拿到加速**。与上一条**不是同一个硬件**——
#   两条数字**不得并列成「GPU 不行 CPU 行」的单一区间**，它们各自绑定自己的硬件与引擎。

# ===========================================================================
# 4. 剪枝判据（公式与成本）
# ===========================================================================

CRITERIA_FORMULA = {
    "magnitude": "|W_ij|",
    "SparseGPT": "|W_ij|^2 / diag((X^T X + lambda I)^-1)_j",
    "Wanda": "|W_ij| * ||X_j||_2",
}
# 来源：（引用）SparseGPT 论文与 Wanda 论文各自的判据式（计划基线 §0.7.G 的统计表；
#   Wanda 的判据式另经 `WebSearch` 找到第三方转述，**未逐字读到原文**）。
# 口径：★ **判据必须先写死定义，结论才成立**（见下一条）。
#   · `H := X^T X`（**不除以 n**）；`X` 是 `(n_samples, d)` 的校准输入；
#   · `crit_Wanda[i,j] := |W_ij| * ||X_j||_2`；`crit_SparseGPT[i,j] := |W_ij|^2 / (H^-1)_jj`
#     （**不带 sqrt**，`lambda = 0`）。
#   换了这两个定义中的任何一个，下面那条归约关系立刻不成立。

CRITERIA_COMPLEXITY = {
    "magnitude": "O(1)",
    "SparseGPT": "O(d^3)",
    "Wanda": "O(d^2)",
}
# 来源：（引用）Wanda 论文表 1 的复杂度列（计划基线）。
# 口径：**`O(1)` 是「每个元素的判据值」的代价**，另两条是**整层**的代价
#   （`d` = 输入维；SparseGPT 的 `O(d^3)` 来自 Hessian 求逆的 Cholesky 预计算）。

WANDA_REDUCTION_CONDITIONS = ("lambda = 0", "Hessian 逆只取对角")
# 来源：（引用）Wanda 论文式 4 的两条前提（计划基线 + 检索佐证）。
# 口径：**满足这两条时 `crit_SparseGPT = crit_Wanda^2`（是平方，不是恒等）**；
#   对两者都取 sqrt 之后才恒等——本机实测见 §5 与 verify.py 的 A.5。
#   ⚠ 这个关系还依赖 **H 是否除以 n**：`H = X^T X` 给 `sg = wa^2`，
#     `H = X^T X / n` 给 `sg = wa^2 / n`。**H 的定义写在上一条的口径里。**

WANDA_AS_IS_QUOTE = "the pruned LLM can be used as is"
# 来源：（引用）Wanda 论文——Wanda **没有任何权重更新**（计划基线；未逐字读到原文）。

WANDA_ACCURACY_TABLE2 = {
    "dense": 59.99,
    "magnitude": 46.94,
    "SparseGPT": 54.94,
    "Wanda": 54.21,
}
# 来源：（引用）Wanda 论文表 2：LLaMA-2 7B / 50% 非结构化 / 零样本均值。
#   基线 = `dense 59.99`；**同一张表**、同一个 harness。
# 口径：**这四行必须同一张表一起引**（跨论文的数字不得并列，见 CONTRIBUTING §2）。
WANDA_2_4_LLAMA7B = {"magnitude": 44.73, "SparseGPT": 50.60, "Wanda": 48.53}
# 来源：（引用）同上，2:4 结构化那一档（LLaMA-2 7B）。
# 口径：**Wanda 在 2:4 上稳定落后 SparseGPT 约 1–2 个点**——两条一起读才是结论。

PRUNE_METRIC_TIME_S = {
    # (LLaMA-7B, 13B, 30B, 65B)，A6000；**只算判据的时间，不含下游**
    "SparseGPT": (203.1, 339.0, 810.3, 1353.4),
    "Wanda": (0.54, 0.91, 2.9, 5.6),
}
# 来源：（引用）Wanda 论文表 5 附近（计划基线 §0.7.G；未逐字读到原文）。
# 口径：**是「剪枝指标计算」的耗时**，两边同模型同硬件；比值 ≈ 242×–376×。

# ===========================================================================
# 5. 蒸馏（温度、损失、压缩账）
# ===========================================================================

HINTON_T2_QUOTE = (
    "Since the magnitudes of the gradients produced by the soft targets scale as "
    "1/T^2 it is important to multiply them by T^2 when using both hard and soft "
    "targets."
)
# 来源：（引用）Hinton et al. 2015，arXiv:1503.02531 §2.1。
#   **本实现者没有逐字读到原文**——本轮 `WebFetch`/`curl` 均被拦截；
#   `WebSearch` 在多个独立站点（含 Keras 官方 issue）上取到**逐字相同**的这句，
#   故按「引用值 + 检索佐证」标注。正文引用时保留英文原句。

ALPHA_SYMBOL_IN_ORIGINAL_PAPER = False
# 来源：（引用）原文只说 "a weighted average of two different objective functions"，
#   并说 "best results were generally obtained by using a considerably lower weight
#   on the second objective function"——**通篇没有 α 这个符号**（计划基线）。
# 口径：流传很广的 `L = α·CE_hard + (1−α)·T^2·KL` 是**后来的社区约定**；
#   另一种常见写法把权重放在 KL 项上（`L = CE_hard + β·T^2·KL`）。
#   ⚠ **两者的换算不是「互为倒数」，是 `β = (1−α)/α`**：
#     把第一式除以 α（常数缩放不改变最优点）即得 `CE + ((1−α)/α)·T^2·KL`。
#     `α = 0.5` 时 `β = 1`；`α → 1` 时 `β → 0`。**倒数是 `1/α`，这里多减了一个 1。**
#   引用时必须写清是哪个约定——本主题正文只写「软目标项的权重」，不写符号 α。

BERT_BASE_CONFIG = {"vocab": 30522, "hidden": 768, "ffn": 3072,
                    "layers": 12, "max_pos": 512}
# 来源：（规格）BERT-base 的 config：词表 30522、hidden 768、FFN 3072、12 层、位置 512。
# 口径：**这是模型配置，不是「参数量」**。参数量由 `verify.py` 的 B 类模型算出，
#   再与 DistilBERT 论文自报的 110M / 66M 对照——那个对照才是断言。

T_SQUARED_ASYMPTOTE = 4096.0
# 来源：（本机实测的极限）`T = 64 ⇒ T² = 4096`。
# 口径：★ **只有这个渐近值是「定律」**——它是 `T→∞` 时的极限；
#   有限的 T=1→64 之比取决于**教师的 logit 尺度**（见下表），从 4098 到 15827 都有。
#   报「梯度掉了几倍」而不报教师尺度，那个倍数就没有意义（P9）。

TEACHER_LOGIT_SPREADS = (0.02, 0.10, 0.30, 1.00, 2.00)
# 来源：（本机实测）T² 标度实验里扫的**教师 logit 尺度**（高斯标准差）。
# 口径：**「梯度掉了几倍」这个数完全由这一列决定**——见下一条实测表。
#   判据缺了这一列，那个倍数就没有意义（P9）。

MEASURED_GRAD_RATIO_T1_T64 = {
    # 判据（**缺一个这个数就没有意义**）：
    #   · 维度 N = 1000（类数）；δ ~ N(0, 0.01²) 且**逐样本零均值**；
    #   · 度量 = `‖g‖₂`，`g_i = (1/T)(q_i − p_i)`，p/q 都按**同一个 T** 做 softmax；
    #   · 每个 σ_v 跑 20 次试验取**均值**；RNG 种子 = `1000003·round(σ_v·100) + trial`
    #     （**每个 (σ_v, trial) 一个独立种子**——不吃全局流，换代码不会让表漂移）。
    #   · 本表由 `verify.py` 的 A.10 复算（纯 python），**不是 torch 的产物**。
    0.02: 4097.53,
    0.10: 4120.14,
    0.30: 4296.88,
    1.00: 7025.01,
    2.00: 14473.15,
}
# 来源：（本机实测）torch 未参与——这条在 verify.py 里用标准库 `math` 复算
#   （见 verify.py 的 A.10），编号程序 03 用真 torch autograd 复算**另一条路径**
#   （它的 setup 不同，所以两边的数字**本来就不该相等**；能对上的是定性结构）。
# 口径：**只有 T→∞ 的渐近值 4096 = 64² 是「定律」**；有限的倍数由教师 logit 尺度决定，
#   40 倍之差全在这一列里。**不得只报一个倍数而不报判据。**

DISTILBERT = {
    "params_teacher_M": 110.0,   # BERT-base
    "params_student_M": 66.0,
    "reported_param_reduction": 0.40,
    "reported_speedup": 1.60,
    "glue_teacher": 79.5,
    "glue_student": 77.0,
    "loss_terms": ("L_ce", "L_mlm", "L_cos"),
}
# 来源：（引用）DistilBERT 论文（Sanh et al. 2019）。参数 110M / 66M 与 GLUE 79.5 / 77.0
#   是**论文印值**；`reported_param_reduction` 与 `reported_speedup` 是论文自报的
#   「少 40% 参数、快 60%」。损失是**三项之和**，**仍然没有 α**。
TINYBERT = {"smaller": 7.5, "faster": 9.4, "glue_ratio": 0.968}
# 来源：（引用）TinyBERT 论文（Jiao et al. 2020）：小 7.5×、快 9.4×、GLUE 达到教师 96.8%。
LLMQAT_LLAMA7B_4_8_4 = {
    # 设置 = 4-bit 权重 / 8-bit 激活 / 4-bit KV；基线不同 ⇒ 两列分开写
    "WikiText2": {"dense": 10.4, "RTN": 151.4, "SmoothQuant": 163.6, "LLM-QAT": 11.6},
    "C4": {"dense": 7.2, "RTN": 55.1, "SmoothQuant": 81.1, "LLM-QAT": 8.6},
}
# 来源：（引用）LLM-QAT，arXiv:2305.17888（计划基线 §0.7.G）。
# 口径：**两列是同一次实验的两个数据集，不是两个 harness**；同列内可比，跨列不可比。
#   ★ 该方法的要点是**用未量化的预训练模型当教师**、对自己生成的样本做数据无关蒸馏。

BITDISTILLER_ARXIV = "arXiv:2402.10631"
# 来源：（引用）`WebSearch` **独立佐证**：题名 "BitDistiller: Unleashing the Potential of
#   Sub-4-Bit LLMs via Self-Distillation"，ACL 2024 long paper，提出 CAKLD
#   （置信度感知 KL）做**自蒸馏**。
#   ⚠ 计划记的备选号 `2402.19115` **是粒子物理论文**——本主题一律用 2402.10631。

DUALGRAD_SOURCED = False
# 来源：（检索）`WebSearch` 对 "DualGraD" 的检索**零命中**。
# 口径：**来源定位不到的，不得标成引用值**（CONTRIBUTING §2：`UNSOURCED` 与数字
#   只能留一个）。本主题正文**不引用**它，verify.py 断言这个标志必须保持 False。

# ===========================================================================
# 6. 本机实测基线（torch 侧；判据与测量条件逐条写明）
# ===========================================================================

MEASURED_META_BITS_PER_GROUP = {
    "float16": 4.0, "bfloat16": 4.0, "int8": 4.0, "float32": 8.0,
}
MEASURED_TOTAL_OVER_DENSE = {
    "float16": 0.5625, "bfloat16": 0.5625, "int8": 0.625, "float32": 0.5625,
}
MEASURED_META_OVER_DENSE = {
    "float16": 0.0625, "bfloat16": 0.0625, "int8": 0.125, "float32": 0.0625,
}
# 来源：（本机实测）`torch.sparse._semi_structured_conversions` 的两个内部函数
#   （`sparse_semi_structured_from_dense_cutlass` / `..._to_dense_cutlass`）。
#   测量条件：64×256 的稠密矩阵；**五个版本逐格一致**
#   （torch 2.6.0 / 2.7.1 / 2.8.0 / 2.11.0 / 2.13.0）。
#   ★ **章节号式的判据**：分母是**原始稠密张量的位宽**（`numel × element_size × 8`），
#     不是「逻辑元素数 × 位宽」——压缩后的矩阵列数已经减半，两者不是一回事。
# 口径：**`float32` 那一列的「组」是 2 元素对，不是 4 元素组**——见下一条。
#   ⚠ **这一列不得与前三列并列成「metadata 都是 4 bit/组」**（P9）。

MEASURED_FP32_ENCODING_UNIT = 2
# 来源：（本机实测 + 本机源码）`_semi_structured_conversions.py` 第 87–95 行：
#   `if dense.dtype != torch.float: ksparse = 4  else: ksparse = 2`。
#   即 **float32 走的是 1:2（每 2 元素一个非零），不是 2:4**——与
#   `CUSPARSELT_50PCT_FORMAT_BY_DTYPE["float"] == "1:2"` 吻合。
# 口径：**「8 bit/组」里的「组」是 2 元素对**（转换成「每 4 元素」是 8 bit，
#   但那是两个编码单元）。写成「float32 的 metadata 是 4 bit/组」是错的。

MEASURED_FP32_PATTERN_ROUNDTRIP = {
    # 判据：在 32×64 的 float32 稠密矩阵上造**固定 2:4 模式**
    # （每组 4 个元素里第 p0、p1 位非零，值分别 1.0 / 2.0），编码后解码，
    # 逐元素 `torch.equal` 判定。**这是能区分「2:4」与「1:2」两个假设的样本**：
    #   · 假设 A（2:4）：6 种模式应当全部还原；
    #   · 假设 B（1:2，每对 (2i, 2i+1) 至多一个非零）：只有 (0,1) 与 (2,3)
    #     **同对两个非零**的那两种会失败。
    (0, 1): False,
    (0, 2): True,
    (0, 3): True,
    (1, 2): True,
    (1, 3): True,
    (2, 3): False,
}
# 来源：（本机实测）五个版本逐格一致（见文档 §5）。
# 口径：**失败 = 静默丢数据**（maxdiff = 2.0，等于被丢掉那个非零的值），
#   不是报错。所以这个假设**只能靠设计判别样本发现**，跑「随便一个稀疏矩阵」
#   的往返是发现不了的（随机模式几乎总有一点在同一对内）。
#   ⚠ 对 dtype 与形状的限定：**float32 且 m 可被 32 整除、k 可被 16 整除**
#     （约束在源码第 72–85 行，实测沿用）。

MEASURED_OVERLAP_50 = 0.9931640625
MEASURED_KENDALL_TAU = 0.9907614087301587
MEASURED_KENDALL_TAU_FLAT = 0.9905139380801172
MEASURED_EXACT_ARGSORT_EQUAL_ROWS = 0
# 来源：（本机实测）在**一般相关**（非正交列）的 X 上比较 Wanda 判据与 SparseGPT 判据：
#   · 形状：`X` 512×64、`W` 32×64、float64；**两侧都用显式 `torch.Generator` 播种**
#     （`X` seed=0、`W` seed=11）——依赖全局 RNG 会让基线对不上；
#   · 「重合率」的判据 = 两侧各取 `top-50%` 的元素集合（**展平后** topk），集合交集 / 基数；
#   · 「Kendall τ」的判据 = **逐行** τ 的均值（τ 用 O(n²) 数对统计，**并列不计**）；
#   · 「argsort 相等」的判据 = 逐行 `argsort` 结果**逐元素相等**的行数。
# 口径：★ 三个数**各绑定自己的判据**，不得互换。同一个 τ 换成「展平后整体算」是
#   **0.9905139380801172**，与逐行均值 **0.9907614087301587** 不是同一个量（**同量不同指**，P9）。
#   ⚠ 计划基线与审查者报的「98.6% / 97.6% / τ 95.5%」都**没有附判据**，
#     本主题一律用本机在写明判据下的值，**不与那些数并列**。

MEASURED_ORTHOGONAL_MAX_REL_DIFF = 2.220e-16
# 来源：（本机实测）列正交的 X（`torch.linalg.qr` 取正交列、再乘跨两个数量级的
#   列尺度；seed=0，W seed=11）上，`sqrt(crit_SparseGPT)/crit_Wanda` 与 1 的
#   最大**相对**差。
# 口径：★ **这不是 0.0**。写成「最大相对差 0.0，是恒等不是近似」需要一个**更强的判据**：
#   `H` 的对角元必须**精确可表示**且开方精确（如 2 的偶数次幂）。
#   本机实测（三个都取 sqrt 后比）：`H = diag(2^k)`、k 全偶数 ⇒ **恰好 0.0**；
#   k 全奇数或混合 ⇒ 1.7e-16 量级；QR 造的正交列 ⇒ 2.220e-16。
#   **「恰好为 0」不是算子的性质，是判据的产物。**
#   ⚠ 还有一个能让结论差 8 个数量级的坑：`torch.tensor([python float, ...])`
#     **默认给 float32**。`H` 若被这样造出来，量到的是 float32 舍入（1.7e-8），
#     看起来像「不恒等」。必须显式 `dtype=torch.float64`。

MEASURED_MAGNITUDE_VS_WANDA_ERR = (0.1974, 0.2044, 0.1984)
# 来源：（本机实测）C 类反例：列正交 X（列尺度跨两个数量级，且第 0 列再放大
#   1 / 20 / 100 倍）+ 权重 `W` 16×64（seed=7，第 0 列权重乘 0.3），
#   50% 稀疏下量 `‖X(W − W_pruned)‖_F`。判据 = magnitude 判据与 Wanda 判据各自
#   `topk` 保留一半后算出的这个范数之**比**（Wanda / magnitude），三个 outlier 倍数。
# 口径：**列正交时这个范数的平方恰好是 `Σ_j ‖X_j‖²·Σ_i Δ_ij²`**（无交叉项），
#   所以「最小化它」就是 Wanda 的定义——Wanda 赢是**可证的**，不是巧合。

MEASURE_HOST = "Apple M1 Pro（10 核）/ macOS / **CPU-only，无 GPU 数字**"
MEASURED_VERSIONS = ("2.6.0", "2.7.1", "2.8.0", "2.11.0", "2.13.0")
UNCOVERED_MINORS = "2.9 / 2.10 / 2.12（本机无解释器）"
TORCH_MIN_FOR_THIS_TOPIC = (2, 8)
TORCH_MIN_REASON = "对齐选择，不是依赖（本主题三个程序在 2.6.0 上也能跑通）"

# ===========================================================================
# 7. 明确未找到可靠来源的项——**留空，不填**
# ===========================================================================

UNSOURCED = (
    "「2:4 稀疏在端到端上到底几倍」的合成区间：**本主题不给合成区间**。"
    "各来源的模型/批次/硬件/内核全不同，能确认的只是「它们都远低于 2×」。"
    "层级/内核级约 1.3×–1.7×、端到端约 1.2×（小 batch 可低于 1×）——"
    "这两档是同域 20-gpu-architecture/tensor-core/05 的正文，此处只回链",

    "cuSPARSELt 压缩缓冲的**具体字节格式**：本主题只测得 torch/CUTLASS 布局的"
    "`36/64`。若某个实现对 metadata 做字节对齐，16-bit 的比值会变成 `(32+8)/64 = 62.5%`。"
    "**未复核，不下断言**",

    "「非结构化在 GPU 上为什么拿不到加速」的硬件级解释：只知道 cuSparse 的 EW 路径"
    "要 >95% 稀疏才开始快于稠密（引用值），**机制未核实**，不编",

    "BitDistiller 的具体精度数字：本主题只引用它的编号与「CAKLD + 自蒸馏」这个定性描述，"
    "**未逐字读到原文**，不给任何 perplexity",

    "DualGraD：`WebSearch` 零命中，**来源未定位**。正文不引用，数字不出现",
)
# 处理方式：正文写「未找到可靠来源」或直接回避，而不是填一个看起来合理的数。

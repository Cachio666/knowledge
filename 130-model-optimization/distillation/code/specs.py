"""130-model-optimization/distillation 的外部规格常量源。

本文件只放两类东西：
  1. **外部常量**（Hinton / DistilBERT / TinyBERT / LLM-QAT 的印值与原文引语），
     每条带一行 `# 来源：`；
  2. **本机实测基线表**（与 01-distillation.md §5 逐行一致的数字），
     每条注明测它的版本与判据。

★ 核查范围（错误模式 P6 的对策）——**本主题的来源档位逐条写明**：
  · 标 `（规格）` 的：来自**模型配置**（BERT-base 的 config）。
  · 标 `（引用）` 的：论文印值。附**来源 / 表号 / 基线**三元组。
  · 标 `（本机实测）` 的：实现者在本机真 torch 上跑出来的数，附**判据与测量条件**。

★ 本文件**不放任何算法**，也**不得 import torch**（`verify.py` 在默认 python3 下 import 它）。
  凡被断言检验的算法（`(1/T)(q−p)` 与 T² 标度）**一律在 `verify.py` 里另写**——
  理由见 CONTRIBUTING §2 的「参考实现必须独立」。

★ **A6 约定**：本文件里**每一条常量都被 `verify.py` 或 `01_distillation.py` 的断言实际读过**。
  判据是机械的：逐个改掉一个值，`make run` 或 `make verify` **必须报错**。
"""

# ===========================================================================
# 1. 蒸馏（温度、损失、压缩账）
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
# 口径：**这是模型配置，不是「参数量」**。参数量由**编号程序** `01_distillation.py` 的 B 类模型算出，
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
    #   · 本表由 `verify.py` 的 A.1 复算（纯 python），**不是 torch 的产物**。
    0.02: 4097.53,
    0.10: 4120.14,
    0.30: 4296.88,
    1.00: 7025.01,
    2.00: 14473.15,
}
# 来源：（本机实测）torch 未参与——这条在 verify.py 里用标准库 `math` 复算
#   （见 verify.py 的 A.1），编号程序 01_distillation.py 用真 torch autograd 复算**另一条路径**
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
#   只能留一个）。本主题正文**不引用**它，**编号程序** `01_distillation.py` 断言这个标志必须保持 False。

# ===========================================================================
# 2. 本机实测条件（torch 侧）
# ===========================================================================

MEASURE_HOST = "Apple M1 Pro（10 核）/ macOS / **CPU-only，无 GPU 数字**"
MEASURED_VERSIONS = ("2.6.0", "2.7.1", "2.8.0", "2.11.0", "2.13.0")
UNCOVERED_MINORS = "2.9 / 2.10 / 2.12（本机无解释器）"

# ===========================================================================
# 3. 明确未找到可靠来源的项——**留空，不填**
# ===========================================================================

UNSOURCED = (
    "BitDistiller 的具体精度数字：本主题只引用它的编号与「CAKLD + 自蒸馏」这个定性描述，"
    "**未逐字读到原文**，不给任何 perplexity",

    "DualGraD：`WebSearch` 零命中，**来源未定位**。正文不引用，数字不出现",
)
# 处理方式：正文写「未找到可靠来源」或直接回避，而不是填一个看起来合理的数。

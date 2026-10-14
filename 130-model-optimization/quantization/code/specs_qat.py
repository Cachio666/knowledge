"""130-model-optimization/quantization 的外部规格常量源。

本文件只放两类东西：
  1. **外部常量**（论文印值、源码事实、判据阈值），每条带一行 `# 来源：`；
  2. **本机实测基线表**（与三篇文档 §5 逐行一致的数字），每条注明测它的版本与测量条件。

★ 核查范围（错误模式 P6 的对策）——**本主题的来源档位逐条写明**：
  · 标 `（本机实现源码）` 的：直接读本机装着的 torch 源文件，附**文件路径**；
    五个解释器（2.6.0 / 2.7.1 / 2.8.0 / 2.11.0 / 2.13.0）读到的一致，逐条注明。
  · 标 `（本机实测）` 的：实现者在本机真 torch 上跑出来的数，附**判据与测量条件**。
  · 标 `（计划基线）` 的：只来自计划文件 §0.7.D 的事实基线或论文印值，
    本实现者**本次没有逐字读到原文**（本轮 `WebFetch` 被策略拦截、`curl` 被 hook 拦截，
    可用通道只有 `WebSearch`）。**这些写成「引用值」，不得写成「本机实测」或「已核实」。**
  · 标 `（检索佐证）` 的：本实现者用 `WebSearch` 拿到二手摘要，**不是原文**。

★ 「规范」与「实现」分开标注（错误模式 P8）。本主题的 P8 与量化格式那一域不同：
  **量化感知训练这一块没有 OCP / ONNX 那样的位级规范**，所有「规格」都是**某一实现的规格**。
  本主题已确证四处「同一件事、两份实现规格」：

  | 事项 | 一边 | 另一边 |
  |---|---|---|
  | 伪量化反传的掩码 | **计划的写法**：区间式 `(qmin−zp)·s ≤ x ≤ (qmax−zp)·s` | **本机实测**：舍入码式 `qmin ≤ round(x/s+zp) ≤ qmax` |
  | `qconfig` 声明的激活量程 | qconfig 里写 `quant_max=255, reduce_range=True` | 模块上**实际生效 0/127** |
  | 冻结 observer 的入口 | `Q.disable_observer(model)`（**静默空操作**） | `model.apply(Q.disable_observer)`（**有效**） |
  | LSQ 的 s 梯度 | 论文式 3（含 `−v/s` 项） | 朴素 `torch.round` + autograd（**不含**该项） |

  **只写一边就会丢掉最有信息量的那部分。**

★ 本文件**不放任何算法**，也**不得 import torch**（`verify.py` 在默认 python3 下 import 它）。
  凡被断言检验的算法（伪量化的前向与掩码、LSQ 的三段梯度、PACT 的 α 梯度、
  STE 的三种变体）**一律在 `verify.py` 里另写**——理由见 CONTRIBUTING §2
  的「参考实现必须独立」：`specs_qat.py` 只放常量，参照实现是 `verify.py` 里另写的 `ref_*`。

★ **A6 约定（本主题自加）**：本文件里**每一条常量都被 `verify.py` 或某个编号程序的断言实际读过**。
  判据是机械的：逐个改掉一个值，`make run` 或 `make verify` **必须报错**；
  且必须带**对照组**（把常量改成它自己，两条都必须仍然 rc=0）。
  **因此本文件不收「没人查的抄件」**——纯引用、无法参与任何断言的长句
  （例如论文里那些没法定位到常量上的叙述），写在三篇文档的正文与「参考」里。

UNSOURCED（查不到可靠来源，**不落成常量**，正文也不得写具体数值）：
  · **STE 偏差的定量大小**——`STE_BIAS_MAGNITUDE` 只记「未找到可靠来源」这件事本身。
  · **PACT 论文的 α 初始值之外的一切超参**（学习率、正则系数 λ 的具体数值）。
  · **计划里给 `prepare_qat` 的「断言 `model.training == True`」**——那是**旧版本那一档**的写法
    （2.6.0 / 2.7.1 / 2.8.0 上是 `assert model.training, "..."`），
    **2.11.0 / 2.13.0 上改成了 `if not model.training: raise AssertionError("...")`**。
    **两种都记**（`PREPARE_QAT_TRAINING_CHECK_FORMS`），不写成「torch 的写法是 X」。

UNRESOLVED（来源冲突或本仓库无法判定，正文按「未解决」写，不落成常量）：
  · **QAT vs PTQ 在 W4A4 上的差距**——Nagel 白皮书的 PTQ 表（Table 6）**没有 W4A4 列**，
    所以这一档「无法量化」。本文件只记「该列不存在」这件事（`NAGEL_PTQ_HAS_W4A4_COLUMN`）。
"""

# ===========================================================================
# 0. 测量条件（三篇文档 §5 直接引用这一节）
# ===========================================================================

# 来源：（本机实测）`python -c 'import torch; print(torch.__version__)'`，2026-09-18。
MEASURED_VERSIONS = ("2.6.0", "2.7.1", "2.8.0", "2.11.0", "2.13.0")
UNCOVERED_MINORS = "2.9.x / 2.10.x / 2.12.x"
MEASURE_HOST = "Apple M1 Pro（10 核）/ macOS / **CPU-only，无 GPU 数字**"

# ★ 本主题的版本下限**不是依赖，是对齐的选择**（避免 P4「编机制从句」）。
#   来源：（本机实测）三个编号程序只用到 `torch.ao.quantization` 的
#   `FakeQuantize` / observer / `prepare` / `prepare_qat` / `convert`、
#   `torch.autograd.Function`、`nn.Linear`、`torch.optim.SGD`——
#   **这些在 2.6.0 上全部存在**，五个解释器实测三个程序全部 rc=0。
#   ⚠ 本域 `numerical-formats` 的下限 ≥2.8 有真实依赖（`torch.float8_e8m0fnu`）；
#     **本主题没有**。照抄那句理由会说出一句假话。
TORCH_MIN_FOR_THIS_TOPIC = (2, 8)
TORCH_MIN_REASON = ("对齐选择（不是依赖）：使 make run 在装多个 torch 的机器上选中与本域"
                    "其余主题同一个解释器，§5 的测量条件全域可比")

# ===========================================================================
# 1. STE（01-ste）
# ===========================================================================

# 来源：（计划基线 + 检索佐证）Bengio, Léonard, Courville,
#   "Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation"。
#   `WebSearch` 佐证的题名、作者与 v1 提交日（2013-08-15）与计划一致。
#   ⚠ **本实现者未逐字读到式 (13) 的原文**，按引用值处理。
STE_PAPER = "arXiv:1308.3432v1"
STE_EQ_NUMBER = 13
# 来源：（计划基线）该文把 STE 的出处归给 Hinton 2012 的 lecture 15b。
#   ⚠ 本实现者**未能独立核实**这条归属（拿不到 lecture 的原文/录像）。
STE_ATTRIBUTED_TO = "Hinton 2012 lecture 15b"

# --- 两条与流行转述相反的原文（引用值，逐字来自计划 §0.7.D）---------------
# 来源：（计划基线）原文 "Better results were actually obtained without multiplying
#   by the derivative of the sigmoid"。**检索佐证**（WebSearch 的二手摘要）给出同一结论：
#   "The straight-through approach worked best without multiplying the estimated gradient
#   by the derivative of the sigmoid"。两条互相印证，但仍**不是**原文逐字。
STE_SIGMOID_VARIANT_QUOTE = ("Better results were actually obtained without multiplying "
                             "by the derivative of the sigmoid")
# 来源：（计划基线）原文 "when considering a single layer of neurons, it has the right sign
#   (this is not guaranteed anymore when back-propagating through more hidden layers)"。
#   ⚠ 这一条**没有**拿到独立的二手佐证，只来自计划基线。
STE_SIGN_SCOPE_QUOTE = ("when considering a single layer of neurons, it has the right sign "
                        "(this is not guaranteed anymore when back-propagating through "
                        "more hidden layers)")

# --- STE 的变体不是一种，而是三种 -------------------------------------------
# 来源：（计划基线 + 检索佐证）arXiv:1606.06160，**检索佐证确认该编号 = DoReFa-Net**
#   （Zhou, Wu, Ni, Zhou, Wen, Zou），它正是讨论 STE 变体与 XNOR-Net 逐通道缩放问题的那一篇。
#   ⚠ **本实现者未逐字读到 §2.3 的原文**，三种变体的归类按计划基线处理。
STE_VARIANTS_PAPER = "arXiv:1606.06160v2"
STE_VARIANTS = (
    "纯恒等（Bengio 2013）",
    "裁剪 STE（BNN）：反传乘区间指示函数",
    "XNOR-Net：恒等 + 逐通道缩放",
)
# 来源：（计划基线）BNN 的裁剪 STE 逐字：`∂c/∂r_i = ∂c/∂r_o · I_{|r_i| ≤ 1}`。
#   ⚠ **这是论文里的形式**：阈值写死成 `|r_i| ≤ 1`（BNN 的激活被约束在 ±1）。
#   本机实测的 torch 伪量化把它推广成「量化码在 [qmin, qmax] 内」，见下面 §1.2。
STE_BNN_CLIP_FORM = "dc/dr_i = dc/dr_o * I[|r_i| <= 1]"

# --- coarse gradient --------------------------------------------------------
# 来源：（计划基线 + 检索佐证）arXiv:1903.05662v1 =
#   *Understanding Straight-Through Estimator in Training Activation Quantized Neural Nets*
#   （Yin, Lyu, Zhang, Osher, Qi, Xin，ICLR 2019）。**检索佐证逐条确认**：
#   ① 激活量化的训练实际在最小化一个**分片常数损失**（piecewise constant loss，
#      梯度几乎处处为零）；② 用 STE 替换导数得到的「梯度」叫 **coarse gradient**，
#      「**不是真梯度，一般也不是任何函数的梯度**」；
#   ③ 论文证明**选对了 STE** 时期望 coarse gradient 与总体梯度正相关、其反方向是下降方向；
#      「**不好的 STE 选择**会让训练在局部极小附近不稳定」。
#   ⚠★ **计划把它转述成「恒等 STE 对 ReLU 是 poor choice」——本实现者未逐字核实这一具体归属。**
#     检索摘要里的原话是泛指的「a poor choice of STE」，**没有点名「恒等 STE + ReLU」这一组合**。
#     `STE_RELU_VERDICT_QUOTE` 只保留可复核的那个短语，正文写明计划的框定更窄。
STE_COARSE_GRADIENT_PAPER = "arXiv:1903.05662v1"
STE_COARSE_GRADIENT_TERM = "coarse gradient"
STE_RELU_VERDICT_QUOTE = "poor choice"

# 来源：（计划基线）「STE 偏差的定量大小」**未找到可靠来源**。
#   ⚠ 这是一条**否定性**常量：它的值就是「不许写具体数值」这件事本身。
#   `verify.py` 与 `12_ste.py` 都断言它仍等于 "UNSOURCED"，防止有人后来往里塞一个数。
STE_BIAS_MAGNITUDE = "UNSOURCED"

# --- 1.2 torch 伪量化的 forward / backward（**本主题最容易写错的一条**）------
# 来源：（本机实测）torch 2.11.0 与另外四个版本（2.6.0 / 2.7.1 / 2.8.0 / 2.13.0）
#   逐格一致。构造方式：`FakeQuantize(observer=MovingAverageMinMaxObserver,
#   quant_min=0, quant_max=255, dtype=torch.quint8)`，手工写死 `scale` / `zero_point`
#   并**关闭 observer**（`observer_enabled = 0`）——否则 observer 会把 scale 覆盖掉。
FQ_STE_SCALE = 0.1
FQ_STE_ZERO_POINT = 128
FQ_STE_QMIN = 0
FQ_STE_QMAX = 255
FQ_STE_X = (-100.0, -12.9, -12.8, 0.0, 12.7, 12.8, 100.0)
FQ_STE_FORWARD = (-12.8, -12.8, -12.8, 0.0, 12.7, 12.7, 12.7)
FQ_STE_BACKWARD = (0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0)

# ★★ 掩码的**判据**（P9：每个数字旁写判据；这里是判据本身）：
#   来源：（本机实测）在 x ∈ [−14, 14] 上取 4801 个点（`i*0.01` 的网格 + 随机点）扫描，
#   逐点比对 autograd 得到的 grad 与两种公式的预测：
#     · **舍入码式** `qmin ≤ round(x/s + zp) ≤ qmax`——**不符 0 个**（五个版本都是 0）；
#     · **区间式** `(qmin−zp)·s ≤ x ≤ (qmax−zp)·s`——**不符 16 个**。
#   ⇒ 计划 §0.7.D 把掩码写成区间式，**与实测不符**。两个式子在那张 7 点表上恰好同解，
#     所以那张表本身是对的、错的只是它旁边的公式。
FQ_STE_MASK_FORM = "quant_min <= round(x / scale + zero_point) <= quant_max"
# 两式给出**相反答案**的实测点（grad 实测为 1，而区间式预言 0）。
# 判据：−12.81 → round(−12.81/0.1 + 128) = 0（在 [0,255] 内）；而区间下界
#       (0−128)·0.1 的 float32 值是 −12.800000190734863，−12.81 < 它。
FQ_STE_INTERVAL_FORM_MISMATCH = (-12.81, 12.71)

# 来源：（本机实测）torch 2.11.0，N = 2^16 个样本、**scale 固定为 0.1**（不跑 observer）。
#   判据 = `grad == 0` 的样本占比（即被掩码挡掉的比例）。
#   ⚠ 这三个数是**跑 `12_ste.py` 时打的**（`torch.Generator().manual_seed(0)`，
#     三档共用同一个生成器、按 N(0,1) → StudentT(3) → 1% 放大 100 倍 的顺序抽）。
#     **换一个种子数字就不同**——`verify.py` 用的是另一套抽样，所以它只判量级（±40% 相对）。
FQ_STE_MASK_ZERO_FRACTION = {
    "N(0,1)": 0.0,
    "StudentT(3)": 0.0010529,          # 0.10529%
    "1% 元素放大 100 倍": 0.0088806,    # 0.88806%
}
# 来源：（本机实测）同一设置但**让 observer 自己定 scale**（重尾数据 1% 放大 100 倍）：
#   observer 给出 scale = 2.3789、zp = 126，掩码挡掉 **0.0000%**——
#   因为 MinMax 的 scale 已经被离群值撑大，量程把全部数据都装下了。
FQ_STE_MASK_ZERO_FRACTION_ADAPTIVE_SCALE = 0.0

# ===========================================================================
# 2. 伪量化的插入位置与 observer 机制（02-fake-quant-and-lsq）
# ===========================================================================

# 来源：（本机实现源码 + 本机实测）torch/ao/nn/qat/modules/linear.py 第 50–51 行：
#       def forward(self, input):
#           return F.linear(input, self.weight_fake_quant(self.weight), self.bias)
#   **权重侧的伪量化写死在 forward 里**；`weight_fake_quant` 在 `__init__` 时构造一次。
#   五个解释器读到的源码一致。
QAT_LINEAR_FORWARD_SRC = "return F.linear(input, self.weight_fake_quant(self.weight), self.bias)"

# 来源：（本机实现源码）torch/ao/quantization/quantize.py::prepare_qat。
#   ⚠ **「prepare_qat 只是挂 observer」是错的——它内部走了一次 `convert`。**
#   五版本读到的顺序一致（2.6.0 / 2.7.1 / 2.8.0 / 2.11.0 / 2.13.0）。
PREPARE_QAT_SRC_ANCHOR = "torch/ao/quantization/quantize.py::prepare_qat"
PREPARE_QAT_STEPS = (
    "检查 model.training 为真（否则抛 AssertionError，消息逐字见表下）",
    "propagate_qconfig_(model, qconfig_dict=None)",
    "convert(model, mapping=mapping, inplace=True, remove_qconfig=False)",
    "prepare(model, observer_non_leaf_module_list=set(mapping.values()), inplace=True)",
)
# ★★ 来源：（本机实测，**五版本对拍新发现**）第 ① 步的**写法跨版本变过**：
#   · torch 2.6.0 / 2.7.1 / 2.8.0 ：`assert model.training, "prepare_qat only works..."`
#   · torch 2.11.0 / 2.13.0      ：`if not model.training: raise AssertionError("...")`
#   两段的**消息文字相同、语义相同**，但**语句形式不同**。
#   ⚠ 计划 §0.7.D 写的是「① 断言 `model.training == True`」——那是**旧版本那一档的写法**，
#     被写成了普适陈述。这正是 §0.3 第 3 条（版本无关地陈述版本相关行为）的又一实例。
#   **引用时不得写成「torch 的写法是……」，必须带版本。**
PREPARE_QAT_TRAINING_CHECK_FORMS = (
    "assert model.training, \"prepare_qat only works on models in training mode\"",
    "if not model.training:\n        raise AssertionError(\"prepare_qat only works on models "
    "in training mode\")",
)
PREPARE_QAT_TRAINING_CHECK_MSG = "prepare_qat only works on models in training mode"
# `convert` 是 4 步里的**第 3 步**（0 基索引 2）——这个下标本身就是一条事实。
PREPARE_QAT_CONVERT_STEP_INDEX = 2
# 来源：（本机实测）prepare_qat 后第 0 层是 `torch.ao.nn.qat.modules.linear.Linear`，
#   `weight.dtype == torch.float32`，前向有 `grad_fn`，SGD 一步后权重确实变化。
#   五个版本一致。**参数本身始终是 fp32**（伪量化只发生在 forward 里）。
QAT_PREPARED_MODULE_CLASS = "torch.ao.nn.qat.modules.linear.Linear"
QAT_PREPARED_WEIGHT_DTYPE = "torch.float32"

# --- observer 不理会 model.eval() ------------------------------------------
# 来源：（本机实现源码）torch/ao/quantization/observer.py 全文
#   **既无 `self.training` 也无 `observer_enabled`**。五个版本都是 0 命中。
OBSERVER_PY_TRAINING_REFS = 0
OBSERVER_PY_OBSERVER_ENABLED_REFS = 0
# 来源：（本机实现源码）`observer_enabled` 是
#   torch/ao/quantization/fake_quantize.py 里 `FakeQuantizeBase.__init__` 注册的 buffer：
#       self.register_buffer("observer_enabled", torch.tensor([1], dtype=torch.uint8))
#   ——**它是 QAT 侧伪量化模块的机制，不是 observer 的机制。**
OBSERVER_ENABLED_DEFINED_IN = "torch/ao/quantization/fake_quantize.py::FakeQuantizeBase"
# 来源：（本机实测）torch 2.11.0，PTQ（`prepare` 插入的普通 observer）跑第一批得
#   (−1.37, 1.00)，切 `eval()` 后跑一批 ×100 的得 (−94.50, 96.37)——**继续更新**。
#   五个版本都复现（数值随随机种子不同，定性结论一致）。
OBSERVER_UPDATES_UNDER_EVAL = True

# --- disable_observer 的两种调用形式 ----------------------------------------
# 来源：（本机实现源码 + 本机实测）`torch.ao.quantization.disable_observer` 是一个
#   **apply 回调**：`if isinstance(mod, FakeQuantizeBase) or _is_fake_quant_script_module(mod): mod.disable_observer()`。
#   ⇒ `Q.disable_observer(model)` 把整个模型传进去是**静默空操作**（不报错、不生效），
#   必须写 `model.apply(Q.disable_observer)`。
#   ⚠ **而且对 `prepare()` 插入的普通 observer，连 `apply()` 也不起作用**
#     ——它不是 `FakeQuantizeBase`，没有 `observer_enabled`。
DISABLE_OBSERVER_IS_APPLY_CALLBACK = True
DISABLE_OBSERVER_WORKS_ON_PLAIN_OBSERVER = False
# 来源：（本机实现源码）`freeze_bn_stats` **不是**通用观察器冻结工具，
#   它只存在于 fused Conv+BN / Linear+BN 模块里。
#   本机 `/torch/ao/` 下 grep `def freeze_bn_stats` 的**全部落点**就是这两个文件。
FREEZE_BN_STATS_MODULES = (
    "torch/ao/nn/intrinsic/qat/modules/conv_fused.py",
    "torch/ao/nn/intrinsic/qat/modules/linear_fused.py",
)

# --- ★ DeprecationWarning 的**触发点**（计划与任务书都写错了的一处）----------
# 来源：（本机实测，**五版本对拍**）用 `warnings.catch_warnings(record=True)` 分别捕获两段：
#   · `import torch.ao.quantization` 时 DeprecationWarning 条数：**0**（五个版本都是 0）；
#   · `Q.prepare_qat(model)` 时：**2.6.0 = 0、2.7.1 = 0、2.8.0 = 3、2.11.0 = 3、2.13.0 = 3**。
#   ⇒ **它打在「调用」上，不是打在「import」上**——`prepare_qat` / `prepare` 在 2.8 起被
#     `typing_extensions.deprecated(...)` 包了一层，于是**每次调用**都发声。
#   ⇒ **版本分界在 2.7.1 与 2.8.0 之间**，不是「2.11.0+」。
#   ⚠ 计划 §0.5 与任务书都写成「每次 import 都会打一条」且「在 2.11.0+」——**两半都与实测不符**。
#   **它仍然是噪声、不是失败**（功能照常，五版本 rc=0）；错的是它**在哪一步、从哪一版起**出现。
DEPRECATION_WARNING_TRIGGER = "调用 prepare / prepare_qat（**不是 import**）"
DEPRECATION_WARNING_IMPORT_COUNT = 0
DEPRECATION_WARNING_FIRST_VERSION = "2.8.0"
DEPRECATION_WARNING_ABSENT_IN = ("2.6.0", "2.7.1")
# 来源：（本机实现源码 + 本机实测）同一次调用还会打若干条 `UserWarning`：
#   "Please use quant_min and quant_max to specify the range for observers.
#    reduce_range will be deprecated ..."——**另一条噪声**，同样不是失败。
OBSERVER_REDUCE_RANGE_DEPRECATION_QUOTE = ("Please use quant_min and quant_max to specify "
                                           "the range for observers.")

# --- qconfig 的「声明」与「实际生效」（P8 的一处实例）------------------------
# 来源：（本机实测）torch 2.11.0 与 2.8.0 一致。
#   ⚠ `qconfig.activation.p.keywords` 里**声明**的是 `quant_min=0, quant_max=255,
#     reduce_range=True`；而模块上 `activation_post_process.quant_min/max`
#     **实际生效 0/127**（reduce_range 把量程砍掉 1 bit）。
#   **两个都对，但它们不是同一个量**（一个是声明、一个是生效）。
QAT_X86_ACT_DECLARED_QMAX = 255
QAT_X86_ACT_DECLARED_REDUCE_RANGE = True
QAT_X86_ACT_EFFECTIVE_QMIN = 0
QAT_X86_ACT_EFFECTIVE_QMAX = 127
QAT_X86_WEIGHT_QMIN = -128
QAT_X86_WEIGHT_QMAX = 127
QAT_X86_WEIGHT_PER_CHANNEL_SYMMETRIC = True
# 来源：（本机实测）`get_default_qat_qconfig("qnnpack")`：激活 0/255、reduce_range=False，
#   权重 **per_channel=False**（与 x86 的 True 不同）。
QAT_QNNPACK_ACT_QMIN = 0
QAT_QNNPACK_ACT_QMAX = 255
QAT_QNNPACK_ACT_REDUCE_RANGE = False
QAT_QNNPACK_WEIGHT_PER_CHANNEL_SYMMETRIC = False
# 来源：（本机实测）`get_default_qconfig("x86")`（**PTQ**）的激活观察器是
#   `HistogramObserver`——不是 MinMax、不是 MovingAverage。
#   ⚠ 这说的是 **PTQ** 的默认；QAT 的默认激活侧是 `FusedMovingAvgObsFakeQuantize`。
PTQ_X86_ACT_OBSERVER = "HistogramObserver"
QAT_X86_ACT_FAKE_QUANT = "FusedMovingAvgObsFakeQuantize"
# 来源：（本机实测）`reduce_range=True` 的存在理由有源码级原文：
#   "For x86 quantized kernels, we need to ensure that the vpmaddubsw instruction does not overflow"。
REDUCE_RANGE_SRC_REASON_QUOTE = ("For x86 quantized kernels, we need to ensure that the "
                                 "vpmaddubsw instruction does not overflow")

# ===========================================================================
# 3. LSQ（02-fake-quant-and-lsq）
# ===========================================================================

# 来源：（检索佐证 + 计划基线）Esser, McKinstry, Bablani, Appuswamy, Modha,
#   "Learned Step Size Quantization"，ICLR 2020。`WebSearch` 独立佐证了 arXiv 编号、
#   作者与「v3 2020-05-07」。⚠ **本实现者未逐字读到式 (3) 的原文。**
LSQ_PAPER = "arXiv:1902.08153"

# 来源：（计划基线）量化器 `v̄ = ⌊clip(v/s, −Q_N, Q_P)⌉`、`v̂ = v̄·s`；
#   无符号（激活）`Q_N = 0, Q_P = 2^b − 1`；有符号（权重）`Q_N = 2^(b−1), Q_P = 2^(b−1) − 1`。
LSQ_QUANTIZER_FORM = "v_bar = round(clip(v / s, -Q_N, Q_P)); v_hat = v_bar * s"
LSQ_UNSIGNED_QN = 0
LSQ_UNSIGNED_QP = 255
LSQ_SIGNED_QN = 8
LSQ_SIGNED_QP = 7

# 来源：（计划基线）初始化 `s₀ = 2⟨|v|⟩/√Q_P`。**不学 clipping 阈值、不学 zero-point**。
LSQ_INIT_FORM = "s0 = 2 * mean(|v|) / sqrt(Q_P)"
LSQ_LEARNS_CLIP_THRESHOLD = False
LSQ_LEARNS_ZERO_POINT = False

# ★★ LSQ 的 s 梯度（论文式 3）：三段分段线性。
#   来源：（计划基线）论文式 3。**本实现者按「round 走 STE」独立复核了这三段**
#   （见 `verify.py` 与 `13_fake_quant_lsq.py` 的断言）——
#   判据：把 `round` 替换成反向恒等的自定义 Function 后，torch 的解析梯度**逐点等于**下式。
LSQ_GRAD_INTERIOR_FORM = "dv_hat/ds = round(v / s) - v / s     (当 -Q_N < v / s < Q_P)"
LSQ_GRAD_LOWER_FORM = "dv_hat/ds = -Q_N                        (当 v / s <= -Q_N)"
LSQ_GRAD_UPPER_FORM = "dv_hat/ds = Q_P                         (当 v / s >= Q_P)"
# ⚠ **本主题的第二处 P8**：朴素实现（`torch.round` + autograd，round 处处导数为 0）
#   得到的是 `v_bar`，**没有 `−v/s` 那一项**。两者在内部段相差整整一个 `v/s`。
#   来源：（本机实测）torch 2.11.0，见 `13_fake_quant_lsq.py` 的 A.4。
LSQ_GRAD_NAIVE_AUTOGRAD_FORM = "dv_hat/ds = round(v / s)        (朴素 autograd，round 导数为 0)"

# 来源：（计划基线）`1/√(N·Q_P)` 缩放因子的来历：让「平均更新幅度/平均参数幅度」
#   在网络各层大致相同；权重 `g = 1/√(N_W·Q_P)`、激活 `g = 1/√(N_F·Q_P)`。
LSQ_SCALE_FACTOR_FORM = "g = 1 / sqrt(N * Q_P)"
# 来源：（计划基线）消融：不加缩放时 step size 的相对更新幅度比权重的大 **2–3 个数量级**，
#   且精度越高失衡越大；**完全不加缩放会不收敛**（2-bit ResNet-18，即使初始 LR 降到 1e-4
#   仍比 baseline 低 **3.4** 个点）。⚠ 未逐字读到原文，按引用值处理。
LSQ_ABLATION_ORDERS_OF_MAGNITUDE = (2, 3)
LSQ_NO_SCALE_DIVERGENCE_GAP_PP = 3.4
LSQ_NO_SCALE_DIVERGENCE_LR = 1e-4

# 来源：（本机实测）torch 2.11.0。**小规模复现 \(\|Δs\|/\|s\|\) 与 \(\|Δw\|/\|w\|\) 的比值**。
#   判据：单层 LSQ 量化（4-bit 有符号权重，Q_P=7），SGD lr=0.01，60 步，
#   取每步相对变化的算术平均；`N` = 该层权重元素数。
#   ⚠ **规模不同 ⇒ 数字不同**：论文的 2–3 个数量级在 ImageNet 级网络上，
#     本机 N ≤ 16384。**只有定性结构必须一致**（不加缩放失衡、加了之后同量级）。
LSQ_LOCAL_REL_UPDATE = {
    "N=2048 no_g": 7.212, "N=2048 with_g": 0.2150,
    "N=4096 no_g": 10.04, "N=4096 with_g": 0.4417,
    "N=8192 no_g": 32.22, "N=8192 with_g": 0.1048,
    "N=16384 no_g": 21.59, "N=16384 with_g": 0.1179,
}

# ===========================================================================
# 4. PACT 与 QAT/PTQ 对照（03-pact-and-comparison）
# ===========================================================================

# 来源：（检索佐证 + 计划基线）Choi, Wang, Venkataramani, Chuang, Srinivasan, Gopalakrishnan,
#   "PACT: Parameterized Clipping Activation for Quantized Neural Networks"。
#   `WebSearch` 独立佐证了 arXiv 编号与作者。
#   ⚠ **同一检索结果指出一处常见误引**：把 PACT 的作者写成「Choi, El-Khamy, Lee」是**错的**
#     ——那一组属于 arXiv:1612.01543（"Towards the Limit of Network Quantization"）。
PACT_PAPER = "arXiv:1805.06085"
PACT_AUTHOR_MISATTRIBUTION = "Choi, El-Khamy, Lee（实为 arXiv:1612.01543 的作者）"

# 来源：（计划基线）激活函数替换为 `y = 0.5(|x| − |x−α| + α)`。
PACT_ACT_FORM = "y = 0.5 * (|x| - |x - alpha| + alpha)"
# 来源：（计划基线 + 本机实测）论文式 3：`∂y_q/∂y` 用 STE 估计为 1，于是
#   `∂y_q/∂α = 0`（x < α）、`= 1`（x ≥ α）——**只被「被截断的那部分」激活驱动**。
#   ⚠ 本机实测在 `x == α` 这一点上解析导数是 **0.5**（`|x−α|` 在 0 处不可导，
#     autograd 取 subgradient 0）。论文写的是「0（x<α）/ 1（x≥α）」两段，
#     **边界点的值在论文里没有定义**——本文件把它单列，正文写明这是本机实测。
PACT_GRAD_ALPHA_FORM = "dy/dalpha = 0 (x < alpha) / 1 (x > alpha)"
PACT_GRAD_ALPHA_AT_BOUNDARY_MEASURED = 0.5
# 来源：（计划基线）实验初始值 **α = 10**（CIFAR10-ResNet20），配 L2 正则防 α 变大。
PACT_ALPHA_INIT = 10.0
PACT_ALPHA_INIT_SETTING = "CIFAR10-ResNet20"
PACT_USES_L2_REGULARIZATION = True

# 来源：（计划基线）PACT 与 LSQ 的三层区别。
PACT_VS_LSQ_THREE_DIFFS = (
    "学什么：PACT 学 clip 上界 α；LSQ 学步长 s",
    "梯度形状：PACT 的 ∂y_q/∂α 在 clip 点以下处处为零；LSQ 的 ∂v̂/∂s 对每个跃迁点都敏感",
    "对 round 的处理：两者都用 STE 估 ∂y_q/∂y，但 LSQ 额外保留了 round 供下游求导",
)

# --- ★ QAT vs PTQ 的实测差距（同论文同协议，本域少见的对照）-------------------
# 来源：（计划基线）Nagel et al., "A White Paper on Neural Network Quantization"，
#   **PTQ 表 6 vs QAT 表 10**（PTQ 5 次平均、QAT 3 次平均）。
#   ⚠ **本实现者未逐字读到那两张表**，按引用值处理。
NAGEL_WHITEPAPER = "arXiv:2106.08295v1"
NAGEL_PTQ_TABLE = "Table 6"
NAGEL_QAT_TABLE = "Table 10"
# 来源：（计划基线）「QAT 的收益在低位宽最大」：
#   W8A8 只多 0.07–0.78 个点（**甚至有个别反例**）；W4A8 扩大到 0.2–1.1 个点。
#   判据：单位是 **ImageNet Top-1 百分点的差**（QAT − PTQ）。
NAGEL_W8A8_GAP_POINTS = (0.07, 0.78)
NAGEL_W4A8_GAP_POINTS = (0.2, 1.1)
# ★ 来源：（计划基线）**白皮书的 PTQ 表没有 W4A4 列**——所以「W4A4 下 PTQ vs QAT 差多少」
#   用这份来源**无法量化**。这条是否定性的，它记的是「那一档不存在」这件事本身。
NAGEL_PTQ_HAS_W4A4_COLUMN = False
# 来源：（计划基线）ResNet18 的具体行（per-tensor）。判据：ImageNet Top-1。
NAGEL_RESNET18_TOP1 = {
    "fp32": 69.68,
    "w8a8_ptq": 69.60,
    "w8a8_qat": 70.38,
    "w4a8_ptq": 68.62,
    "w4a8_qat": 69.76,
}

# ★ 来源：（本机实测）torch 2.11.0（2.8.0 / 2.13.0 逐格相同；2.6.0 / 2.7.1 见下）。
#   **本机小规模复现「QAT 的收益在低位宽最大」**。判据与条件（P9 要求写全）：
#     · 任务：32 维输入的小 MLP（32→64→2）、4096 条合成二分类样本、600 步 Adam（lr=3e-3）；
#     · FP32 基线 0.9766（同一网络初始化，`torch.manual_seed(42)`）；
#     · **PTQ** = 同一份 FP32 权重，`prepare()` 后跑一遍训练集校准 observer，冻结 qparams；
#     · **QAT** = 从同一份 FP32 权重出发 `prepare_qat()`，再训同样的步数；
#     · 两侧都在 **fake-quant（浮点）** 空间里评测，**不做 `convert`**——
#       §0.5 已定死：本机 `convert` 会撞 `NoQEngine`（该构建没编 x86 后端）。
#     · 权重 per_channel_symmetric、激活 per_tensor_affine（`reduce_range=False`）；
#       W/A 位宽见键名。
#   ⚠ **规模不同 ⇒ 数字不同**：真数字只有定性结构（低位宽收益大、W8A8 几乎无收益）
#     与本机一致。**不得把这张表与白皮书的 Table 6/10 并列**（判据、数据、网络全不同）。
QAT_PTQ_LOCAL_GAP = {
    "W8A8": -0.0005,
    "W4A8": -0.0010,
    "W4A4": 0.0088,
    "W3A3": 0.0112,
    "W2A2": 0.0791,
}
# 来源：（本机实测）同一实验在 torch **2.6.0 / 2.7.1** 上的那一组（两者逐格相同）。
#   ⚠ **这是本主题唯一一处实测到的版本差异**：2.6/2.7 与 2.8+ 的 PTQ 侧数字不同
#   （W8A8 的 PTQ 从 0.9756 变 0.9766），说明 observer 的行为在这两个版本之间有变化。
#   **定性结构两档一致**（W8A8 的 gap 为负、W2A2 的 gap 最大）。
QAT_PTQ_LOCAL_GAP_2_6 = {
    "W8A8": -0.0010,
    "W4A8": 0.0000,
    "W4A4": 0.0103,
    "W3A3": -0.0015,
    "W2A2": 0.0615,
}

# 来源：（本机实测）torch 5 个版本全部一致：torch 伪量化的 forward 在
#   **理论可达区间之外**是**饱和**（不是 NaN、不是回绕）。
#   判据：x = −100 与 +100 的 forward 分别等于区间端点 −12.8 / 12.7。
FQ_SATURATES_OUTSIDE_RANGE = True

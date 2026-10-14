"""130-model-optimization/quantization 的外部规格常量源。

本文件只放两类东西：
  1. **外部常量**（论文印值、判据阈值、实现默认值），每条带一行 `# 来源：`；
  2. **本机实测基线表**（与四篇文档 §5 逐行一致的数字），每条注明测它的版本与测量条件。

★ 核查范围（错误模式 P6 的对策）——**本主题的来源档位逐条写明**：
  · 标 `（源码直取）` 的：实现者 2026-09-18 用 python `urllib` 把该文件**逐字读下来**
    （TensorRT 的 `calib/histogram.py`、llm-awq 的 `awq/quantize/auto_scale.py`），
    结论可在同一 URL 上复核；
  · 标 `（本机实现源码）` 的：直接读本机装着的 torch / transformers 的源文件，附**文件路径与行号**；
  · 标 `（计划基线，未二次核实）` 的：只来自计划文件 §0.7.C 的事实基线，
    本实现者**本次没有拿到任何独立佐证**（本轮 `WebFetch` 被策略拦截、`curl` 被 hook 拦截、
    `raw.githubusercontent.com` 时通时断）。
  · **三者在上面的正文里都写成「引用值」**，因为「引用值」这个标签隐含的是「来源存在」，
    不是「本仓库复核过其测量口径」（CONTRIBUTING §2）。
    **不得把任何一个说成「本机实测」或「已核实」。**

★ 「规范」与「实现」分开标注（错误模式 P8）。本主题的真冲突**不在标准文本上**——
  量化后训练 / 训练后量化这一块**没有 OCP / ONNX 那样的位级规范**。
  它的 P8 落在**同一件事的两份实现规格**上，本主题已确证三处：
  · **`dampening` 的默认值**：GPTQ 论文与官方仓库 **1%**，transformers 的 `GPTQConfig`
    默认 **0.1（10%）**（且 docstring 写着 "Recommended value is 0.1"）。**引用必须说清是哪个。**
  · **`blocksize` 与 `group_size`**：官方实现里是两个独立参数
    （前者是计算批处理、**不改数学结果**；后者是量化粒度）。
    transformers 只暴露 `group_size`，`blocksize` 不出现在 `GPTQConfig` 里。
  · **PyTorch 的 `HistogramObserver` vs TensorRT 的 entropy 校准器**：
    两者都是「从直方图搜一个 amax」，但**目标函数不同**——
    前者是 **L2**（源码 docstring 逐字可查），后者是 **KL 散度**（源码 `scipy.stats.entropy`）。
    **把它们当成同一件事，是本主题最常见的错误。**

★ 本文件**不放任何算法**，也**不得 import torch**（`verify.py` 在默认 python3 下 import 它）。
  凡被断言检验的算法（EMA 递推、直方图 L2 搜索、TRT 的 KL 搜索、GPTQ 逐列补偿、
  对称/非对称 RTN、随机舍入）**一律在 `verify.py` 里另写**——
  理由见 CONTRIBUTING §2 的「参考实现必须独立」。

★ **A6 约定（本主题自加）**：本文件里**每一条常量都被 `verify.py` 或某个编号程序的断言实际读过**。
  判据是机械的：逐个改掉一个值，`make run` 或 `make verify` **必须报错**。
  **因此本文件不收「没人查的抄件」**——纯引用、无法参与任何断言的长句
  （例如「AWQ 论文图 2 的 43.2 → 13.0」），写在四篇文档的正文与「参考」里，**不落成常量**。

UNSOURCED（查不到一手来源，**正文不使用**）：
  · **「BERT 用 99.999 / YOLO 用 99.95」这类 percentile 分档**——计划文件已判为未找到可靠一手来源。
    本文件中 `TRT_PERCENTILE_ONE_TRUSTWORTHY` 只保留 **99.99** 这一个可定位的值
    （来源：TensorRT 源码里 `compute_amax` 的 `percentile` 形参默认值）。
  · **AdaRound 的 β 退火区间与 λ 的具体数值**——论文正文只说 "we anneal the parameter β"，
    未印区间与系数。**不落成常量**；`ADAROUND_UNPRINTED` 里记的是「哪些量论文没印」这件事本身。
  · **SmoothQuant 的「1% 显著权重」单独一条**——论文里 0.1%–1% 是**一个区间**，
    不宜拆成两个常量；本文件只留区间本身。

UNRESOLVED（来源冲突或本仓库无法判定，正文按「未解决」写，不落成常量）：
  · **「group size = 128」的一手出处**——常被引作此用的 Q-BERT（arXiv:1909.05840）
    Table 2 列名是 **"# groups"（组数）**，取值 1/12/128/768。**组数不是组大小**
    （对 768 宽的矩阵，128 组 ≈ group size 6）。**本文件不落「128 是 group size」这种常量。**
  · **GPTQ/AWQ/SmoothQuant 三者能否叠加**——未找到任何论文给出严格实验。
    能确证的只有「AWQ 论文把 GPTQ-R 当独立 baseline 比」。**不落成常量。**
"""

# ===========================================================================
# 0. 测量条件（四篇文档 §5 直接引用这一节）
# ===========================================================================

# 本主题的编号程序在真 torch 上跑；下面五个解释器都跑过全部四个程序。
# 来源：本机实测（`python3 -c 'import torch; print(torch.__version__)'`），2026-09-18。
MEASURED_VERSIONS = ("2.6.0", "2.7.1", "2.8.0", "2.11.0", "2.13.0")
UNCOVERED_MINORS = "2.9.x / 2.10.x / 2.12.x"
MEASURE_HOST = "Apple M1 Pro（10 核）/ macOS / **CPU-only，无 GPU 数字**"

# ★ 本主题的版本下限**不是依赖，是对齐的选择**（避免 P4「编机制从句」）。
#   来源：本主题代码自己的依赖清单（四个编号程序只用到 float32/float64、
#   `torch.ao.quantization` 的 observer / FakeQuantize / prepare_qat / convert、
#   以及 `torch.linalg.cholesky*`——**2.6.0 起全部存在**，五个解释器实测全通过）。
#   ⚠ 本域 `numerical-formats` 的下限 ≥2.8 有真实依赖（`torch.float8_e8m0fnu`）；
#     **本主题没有**。照抄那句理由会说出一句假话。
TORCH_MIN_FOR_THIS_TOPIC = (2, 8)
TORCH_MIN_REASON = ("对齐选择（不是依赖）：使 make run 在装多个 torch 的机器上选中与本域"
                    "其余主题同一个解释器，§5 的测量条件全域可比")

# ===========================================================================
# 1. 校准（01-calibration）
# ===========================================================================

# --- 1.1 移动平均（EMA）----------------------------------------------------
# 来源：（本机实现源码）torch/ao/quantization/observer.py，
#   `MovingAverageMinMaxObserver.__init__` 的形参默认值 `averaging_constant: float = 0.01`。
#   五个解释器（2.6.0 / 2.7.1 / 2.8.0 / 2.11.0 / 2.13.0）读到的默认值一致。
EMA_AVERAGING_CONSTANT = 0.01

# 来源：（本机实现源码 + 本机实测）同文件 `MovingAverageMinMaxObserver.forward`：
#   首次观测时 `min_val` 是 `inf`、`max_val` 是 `-inf`，于是走
#   `self.min_val.copy_(torch.min(min_val_cur, self.min_val))` ——
#   **`min(实际值, inf)` 就是实际值本身**，所以「第一次观测直接取 aminmax」。
#   ⚠ 这不是一条显式分支，而是**初值 inf 的后果**；本文件把它写成常量是为了可断言。
EMA_FIRST_OBSERVATION_MODE = "aminmax"   # 取值只允许 "aminmax"（初值 inf 的后果）

# 来源：（本机实测）torch 2.11.0，`MovingAverageMinMaxObserver()` 依次喂入
#   `[-10,10] / [-20,20] / [-30,30]`，读 `max_val` 得 **10.0 / 10.100000381469727 /
#   10.29900074005127**。判据：逐次单样本 `aminmax`，更新式 `x ← x + c·(新值 − x)`。
#   ⚠ 记录值 10.1 与 10.299 是**四舍五入后的打印值**，二进制里分别是上面那两串。
EMA_SEEN_MAXIMA = (10.0, 20.0, 30.0)
EMA_RECORDED_MAXIMA = (10.0, 10.100000381469727, 10.29900074005127)

# --- 1.2 直方图校准器 -----------------------------------------------------
# 来源：（本机实现源码）observer.py，`HistogramObserver.__init__` 的 `bins: int = 2048`。
HISTOGRAM_BINS = 2048
# 于是「搜索的最小裁剪量」= 1 个 bin = 1/2048 的量程。**这是网格分辨率，不是最优性证据。**
HISTOGRAM_ONE_BIN_FRACTION = 1.0 / HISTOGRAM_BINS      # = 0.00048828125

# 来源：（本机实现源码）observer.py `_non_linear_param_search` 内的 `stepsize = 1e-5`。
#   ⚠ 它是**分位数的步长**（每次把 alpha 加 1e-5 或把 beta 减 1e-5），不是量程的步长。
HISTOGRAM_SEARCH_STEPSIZE = 1e-5

# 来源：（本机实现源码，逐字）同文件 `_get_norm` 的 docstring：
#   "Currently only L2 norm is supported."，且函数体只有一种形式
#   `density * (end³ − begin³) / 3`。
HISTOGRAM_NORM_DOC = "Currently only L2 norm is supported."
# 来源：（本机实现源码，逐字）`_non_linear_param_search` 的 docstring 首两行：
#   "An approximation for L2 error minimization for selecting min/max.
#    By selecting new min/max, we filter out outliers in input distribution."
#   ⚠ **这是该方法（`_non_linear_param_search`）的 docstring，不是类 docstring**；
#     五个解释器读到的位置一致。
#   ⚠ **计划文件把这句记成了「它的 docstring」而没写清是哪个 docstring**——
#     类 docstring 里只有 "The search for the min/max values ensures the minimization of
#     the quantization error with respect to the floating model" 这句泛化表述。
HISTOGRAM_DOC_IS_L2 = True
HISTOGRAM_DOC_SOURCE = "_non_linear_param_search.__doc__（不是 HistogramObserver 类 docstring）"
# 来源：（本机实现源码，逐字）同 docstring 第三行："This follows the implementation of
#   NormMinimization::NonlinearQuantizationParamsSearch in
#   caffe2/quantization/server/norm_minimization.cc"
HISTOGRAM_FOLLOWS_CAFFE2 = "NormMinimization::NonlinearQuantizationParamsSearch"

# --- 1.3 TensorRT 的 entropy（KL）校准器（**另一份实现**，不要与上面混）--------
# 来源：（源码直取）TensorRT main 分支
#   tools/pytorch-quantization/pytorch_quantization/calib/histogram.py
#   （2026-09-18 用 python urllib 直取整文件，16541 字节，逐行读过）。
#   · `HistogramCalibrator.__init__(..., num_bins=2048, ...)`          —— 第 51 行
#   · `compute_amax(self, method, *, stride=1, start_bin=128, percentile=99.99)` —— 第 124 行
#   · `_compute_amax_entropy(..., stride=1, start_bin=128)`            —— 第 182 行
#   · 第 20 行 `from scipy.stats import entropy`；第 244 行
#     `ent = entropy(reference_density, new_density)`
#   · 第 250 行 `last_argmin = len(divergences) - 1 - np.argmin(divergences[::-1])`
#     —— **反向取 argmin ⇒ 并列时取最后一个（= 最大的 amax）**
TRT_ENTROPY_FILE = "TensorRT/tools/pytorch-quantization/pytorch_quantization/calib/histogram.py"
TRT_NUM_BINS = 2048
TRT_START_BIN = 128
TRT_PERCENTILE = 99.99
TRT_TIE_BREAK = "last_argmin"          # 并列取更大的 amax；取值只允许 "last_argmin"
TRT_KL_SOURCE = "scipy.stats.entropy(reference_density, new_density)"
# ⚠ **只有 99.99 这一个值可定位**。「BERT 用 99.999 / YOLO 用 99.95」**未找到一手来源**。
TRT_PERCENTILE_ONE_TRUSTWORTHY = 99.99

# --- 1.4 reduce_range ------------------------------------------------------
# 来源：（本机实现源码，逐字）observer.py `MinMaxObserver.__init__` 的注释：
#   "For x86 quantized kernels, we need to ensure that the vpmaddubsw instruction does not
#    overflow. We allow for a reduce_range argument to observers that reduces the quantized
#    range to (0,127) or (-64, 63).
#    For more details see aten/src/ATen/native/quantized/cpu/qconv.cpp
#    This is not an optimal choice for non x86 backends as it loses a bit of precision
#    for activations."
REDUCE_RANGE_REASON_OPCODE = "vpmaddubsw"
REDUCE_RANGE_QUINT8 = (0, 127)
REDUCE_RANGE_QINT8 = (-64, 63)
REDUCE_RANGE_NOT_OPTIMAL_NOTE = "This is not an optimal choice for non x86 backends"
# ⚠ `per_tensor_symmetric + reduce_range + quint8` **直接抛 NotImplementedError**。
#   来源：（本机实测）torch 2.11.0，异常原文
#   "Cannot reduce range for symmetric quantization for quint8"。
REDUCE_RANGE_SYMMETRIC_QUINT8_RAISES = True

# --- 1.5 对称量化的除数 -----------------------------------------------------
# 来源：（本机实现源码 + 本机实测）observer.py 的 `UniformQuantizationObserverBase`
#   走 `scale = max(|min|, |max|) / ((quant_max − quant_min) / 2)`；int8 的
#   quant_min/quant_max = −128/127，于是除数 = 255/2 = **127.5**，不是 127。
#   实测：`torch.randn(1000)*3` 的最大绝对值 10.7026…，observer 给出 scale 0.08394139，
#   而 `max/127.5 = 0.08394139`（127 给 0.08427187，128 给 0.08361349）。**只与 127.5 吻合。**
SYMMETRIC_DIVISOR_QINT8 = 127.5
SYMMETRIC_DIVISOR_DERIVATION = (255, 2)      # (quant_max − quant_min, 2)

# --- 1.6 本机实测基线（与 01-calibration.md §5 逐行一致）---------------------
# 来源：（本机实测）torch 2.11.0 / Python 3.12.9 / Apple M1 Pro，2026-09-18。
#   判据：张量 = 100000 点 N(0,1)，前 k 个点乘 100；比较的是
#   `scale × (quant_max − quant_min)`（= 覆盖量程），不是 `observer.min_val/max_val`
#   （后者是**直方图的范围**，搜索后并不回写，所以直接读它永远是 0% 裁剪）。
CALIB_CLIP_MEASURED = {
    # kind: (k, minmax_span, hist_span, clip_pct)
    "gauss_outlier":     (5, 316.1048, 315.9504, 0.05),
    "laplace_outlier":   (3, 512.0759, 511.8259, 0.05),
    "spike":             (1, 50.4343, 50.4097, 0.05),
    "studentt":          (5, 177.4705, 166.1186, 6.40),
    "lognormal":         (0, 9185.5875, 6875.7355, 25.15),
    "uniform_plus_out":  (1, 301.0000, 300.8530, 0.05),
}
# ⚠ **计划文件写的「0.05%–4.10%」只在控制者自己那 6 组数据上成立**，
#   而「6 组重尾数据」的构造方式计划里没有定义。**本主题的 6 组定义见上面这张表的 kind**，
#   实测区间是 **0.05%–25.15%**。**换一组数据这个区间就变，所以它是一个「随数据走」的量。**
CALIB_CLIP_MEASURED_RANGE_PCT = (0.05, 25.15)
# 1 个 bin 的分辨率（0.0488%）与实测最小裁剪量（0.05%）对得上 ⇒
# **「0.05%」是搜索网格的地板，不是「L2 最优恰好不想裁」的证据。**
CALIB_CLIP_ONE_BIN_IS_FLOOR = True

# 来源：（本机实测）torch 2.11.0，同一批数据（200000 点 N(0,1)，20 个点乘 40），
#   三个 amax 各自的判据：
#     · MinMax          = observer 的 scale×255/2，无裁剪（定义上裁 0%）
#     · HistogramObserver = observer 的 scale×255/2
#     · TensorRT KL     = 把上面那份源码的 `_compute_amax_entropy` 逐行移植后用同一份数据跑
#   （判据逐条写出，因为这几个数**不是同一个量**：见 01-calibration.md §1 第三步）
#   ⚠ `max_abs_x`（= max|x|）与 `minmax_amax`（= scale×255/2）**不是同一个量**：
#     minmax 的零点被约束为整数（见 04-integer-formats），量程会整体平移，
#     于是本机实测 `max|x| = 112.276 > minmax_amax = 104.440`。**两者不得混用。**
CALIB_AMAX_COMPARE_MEASURED = {
    "max_abs_x": 112.276,   # ★ 与 minmax_amax 不是同一个量（见下）
    "minmax_amax": 104.440,
    "histogram_amax": 71.599,
    "trt_kl_amax": 12.664,
    "histogram_over_trt_kl": 5.65,
}
# 真实 MSE 最优 amax（沿对称阈值扫描 400 点，量化用 256 级非对称 affine）——**第四个量**
CALIB_AMAX_MSE_OPTIMAL_MEASURED = 78.593

# --- 1.7 observer 的冻结 ----------------------------------------------------
# 来源：（本机实现源码 + 本机实测）
#   · **`observer.py` 全文既没有 `self.training` 也没有 `observer_enabled`**——
#     五个解释器逐个确认（grep 源码串）。所以「observer 不理会 eval()」成立。
#     实测：批 1 后 `(-0.42, 0.61)`，切 `eval()` 再喂 ×100 的批 → `(-2.24, 1.28)`，**继续更新**。
#   · `observer_enabled` 属于 **`FakeQuantizeBase`**（fake_quantize.py 第 84/93 行），
#     不属于任何 observer。
#   · `disable_observer(mod)` 的实现是
#     `if isinstance(mod, FakeQuantizeBase) ...: mod.disable_observer()`；
#     它是个 **apply 回调**，必须写成 `model.apply(Q.disable_observer)`。
#     **直接 `Q.disable_observer(model)` 传整个模型进去是个静默空操作。**
OBSERVER_HAS_TRAINING_FLAG = False
OBSERVER_ENABLED_LIVES_ON = "FakeQuantizeBase"
DISABLE_OBSERVER_IS_APPLY_CALLBACK = True

# ===========================================================================
# 2. GPTQ（02-gptq）
# ===========================================================================

# --- 2.1 论文超参（**规范档：论文印值**）-----------------------------------
# 来源：（计划基线，未二次核实）GPTQ 论文 arXiv:2210.17323 正文：
#   block size B = 128 列；校准集 = 128 条 2048-token 的 C4 随机片段；
#   "we always choose 1% of the average diagonal value"。
GPTQ_BLOCK_SIZE_PAPER = 128
GPTQ_CALIB_SEQUENCES = 128
GPTQ_CALIB_SEQ_LEN = 2048
GPTQ_CALIB_DATASET = "C4"
GPTQ_PERCDAMP_PAPER = 0.01          # = 1% of the average diagonal value

# --- 2.2 实现的默认（**实现档：库与仓库**）----------------------------------
# 来源：（计划基线，未二次核实）GPTQ 官方仓库（IST-DASLab/gptq）`quant.py` 的默认值
#   `blocksize=128, percdamp=.01, groupsize=-1, actorder=False`。
#   ⚠ 本实现者 2026-09-18 直取到的 `quant.py`（7330 字节）**不含 `fasterquant`**，
#     **未能逐字核到这组默认值**；按引用值处理。
GPTQ_IMPL_DEFAULT_BLOCKSIZE = 128
GPTQ_IMPL_DEFAULT_PERCDAMP = 0.01
GPTQ_IMPL_DEFAULT_GROUPSIZE = -1
GPTQ_IMPL_DEFAULT_ACTORDER = False

# 来源：（本机实现源码）本机装的 transformers 4.57.1，
#   `transformers/utils/quantization_config.py` 第 715–717 行：
#     `group_size: int = 128` / `damp_percent: float = 0.1` / `desc_act: bool = False`
#   同文件 docstring：「damp_percent (float, optional, defaults to 0.1):
#     The percent of the average Hessian diagonal to use for dampening.
#     **Recommended value is 0.1.**」
#   `GPTQConfig` 里**没有 blocksize 这个字段**。
#   ⚠ **这就是 P8 的实例：论文 1% vs transformers 0.1，差 10 倍。引用必须说清是哪个。**
GPTQ_CONFIG_DAMP_PERCENT_DEFAULT = 0.1
GPTQ_CONFIG_GROUP_SIZE_DEFAULT = 128
GPTQ_CONFIG_DESC_ACT_DEFAULT = False
GPTQ_CONFIG_HAS_BLOCKSIZE_FIELD = False
GPTQ_DAMP_DEFAULT_RATIO = 10        # 0.1 / 0.01 —— 两个默认值之比

# --- 2.3 act-order ----------------------------------------------------------
# 来源：（计划基线，未二次核实）计划文件 §0.7.C：「`act-order` 不在 GPTQ 论文里（全文检索确认）。
#   它是实现层特性：`perm = argsort(diag(H), descending=True)`」。
#   判据：`H = (2/n)Σ x xᵀ` ⇒ `diag(H) ∝ Σ_i x_i²`（该输入通道激活的**二阶矩**），
#   **不是「激活大小」（一阶量）**。
GPTQ_ACT_ORDER_IN_PAPER = False

# --- 2.4 本机实测基线（小矩阵上的本主题迷你实现）----------------------------
# 来源：（本机实测）torch 2.11.0 / Python 3.12.9 / Apple M1 Pro，2026-09-18。
#   判据：`torch.manual_seed(0)`；W:[d_out,d_in] = [256,512] × N(0,1)×0.05；
#   校准数据 X:[1024,512]，通道 1/2/3 = 通道 0 的 0.98 倍 + N(0,0.2)（造出病态 H），
#   **通道 7 幅值 ×50**；`H = (2/n)XᵀX`（含 1/n）；4-bit 对称（qmax=7）；
#   percdamp = 论文默认 0.01；blocksize = 官方默认 128。
#   误差口径：`mean((X Wᵀ − X W_qᵀ)²)`（**层输出**误差，不是权重误差）。
#   ★ 这组数是 `09_gptq.py` 的**实际输出**，与文档 §5 逐行一致。
GPTQ_MEASURED = {
    "rtn_output_mse_gs128": 1.0519e-1,
    "gptq_output_mse_gs128": 5.6128e-2,        # 比 RTN 低 46.6%
    "gptq_reduction_pct_gs128": 46.6,
    "rtn_output_mse_gs32": 6.5618e-2,
    "gptq_output_mse_gs32": 3.5642e-2,         # 比 RTN 低 45.7%
    "gptq_reduction_pct_gs32": 45.7,
    "rtn_output_mse_percol": 1.3766e-1,
    "gptq_output_mse_percol": 7.3200e-2,       # 比 RTN 低 46.8%
    # ★ 权重空间反而**变差**：GPTQ 用权重误差换输出误差
    "rtn_weight_mse_percol": 4.5301e-5,
    "gptq_weight_mse_percol": 2.2520e-4,
    # dampening 扫描（同一设定，gs=128）
    "percdamp_sweep": {0.0: 5.540717e-2, 0.001: 5.544126e-2, 0.01: 5.612819e-2,
                       0.05: 6.140117e-2, 0.1: 6.701501e-2, 0.5: 8.561905e-2},
}
# blocksize 是纯计算批处理：**bs=128 / 32 / 16 三个值给出逐元素完全相同的 Q**。
GPTQ_BLOCKSIZE_INVARIANCE_MAXDIFF = 0.0
# act-order 在本机这个合成设定下**没有稳定收益**（gs=128 时略差、gs=-1 时略好）
GPTQ_ACTORDER_MEASURED_DELTA_PCT = {"gs128": -4.71, "gs32": -4.62, "percol": +1.22}

# ===========================================================================
# 3. AWQ 与 SmoothQuant（03-awq-and-smoothquant）
# ===========================================================================

# --- 3.1 AWQ 的 α 搜索 ------------------------------------------------------
# 来源：（源码直取）llm-awq `awq/quantize/auto_scale.py`（2026-09-18 urllib 直取，15813 字节）：
#   第 124 行 `n_grid = 20`；第 128-129 行
#   `for ratio in range(n_grid): ratio = ratio * 1 / n_grid`
#   ⇒ α ∈ {0, 0.05, …, 0.95}，**α = 1.0 取不到**。
AWQ_N_GRID = 20
AWQ_ALPHA_STEP = 0.05
AWQ_ALPHA_MAX_REACHABLE = 0.95
# 来源：（源码直取，逐字）同文件第 130-131 行：
#   `scales = x_max.pow(ratio).clamp(min=1e-4).view(-1)`
#   `scales = scales / (scales.max() * scales.min()).sqrt()`
#   ⇒ **先按几何均值归一，再进网格**（论文没写这一步）。
AWQ_SCALE_GEOMEAN_NORM = True
# 来源：（源码直取，逐字）同文件第 29-30 行：
#   `def get_act_scale(x): return x.abs().view(-1, x.shape[-1]).mean(0)`
#   ⇒ 逐通道的**平均绝对值**，不是最大值。
AWQ_ACT_SCALE_STAT = "mean_abs"
# 来源：（计划基线，未二次核实）AWQ 论文：显著权重占比 **0.1%–1%**；
#   识别显著通道要看**激活**分布而不是权重分布。
AWQ_SALIENT_FRACTION = (0.001, 0.01)
# 来源：（计划基线，未二次核实）AWQ 误差分析：`Err(Q(w)x) = Δ·RoundErr(w/Δ)·x`，
#   放大 s 倍后新误差与原误差之比 = `(Δ'/Δ)·(1/s)`。
AWQ_ERROR_RATIO_FORM = "(Δ'/Δ)·(1/s)"

# --- 3.2 SmoothQuant --------------------------------------------------------
# 来源：（计划基线，未二次核实）SmoothQuant arXiv:2211.10438：
#   迁移公式 `Y = (X diag(s)⁻¹)·(diag(s) W)`，`s_j = max(|X_j|)^α / max(|W_j|)^(1−α)`；
#   OPT/BLOOM 全系 α = 0.5（"well-balanced point"）；消融甜蜜区 0.4–0.6；
#   GLM-130B 因激活离群更多用 α = 0.75；校准用 Pile 的 512 条随机句子。
SQ_MIGRATION_FORM = "s_j = max(|X_j|)^α / max(|W_j|)^(1-alpha)"
SQ_ALPHA_DEFAULT = 0.5
SQ_ALPHA_SWEET_ZONE = (0.4, 0.6)
SQ_ALPHA_GLM130B = 0.75
SQ_CALIB_CORPUS = "Pile"
SQ_CALIB_SENTENCES = 512
# ★ 两处收益数字**不同**，引用须标出处位置（**这就是 P9**：两个数都对，但不是同一句话）。
#   来源：（计划基线，未二次核实）摘要 "up to 1.56× speedup and 2× memory reduction"；
#   正文 §4.3 "up to 1.51× and 1.96×"。
SQ_SPEEDUP_ABSTRACT = 1.56
SQ_SPEEDUP_BODY = 1.51
SQ_MEMORY_ABSTRACT = 2.00
SQ_MEMORY_BODY = 1.96
# ★ **SmoothQuant 是 8-bit 方法**；W4A4 会崩。
#   来源：（计划基线，未二次核实）Atom arXiv:2310.19102v3 Table 2：
#   Llama-7B FP16 5.68 / SmoothQuant **W4A4 22.62** / W3A3 2.7e4。基线 = FP16 5.68。
SQ_TARGET_BITS = 8
SQ_W4A4_PPL = 22.62
SQ_FP16_PPL = 5.68

# --- 3.3 本机实测基线 -------------------------------------------------------
# 来源：（本机实测）torch 2.11.0，X:[128,512]，通道 {5,17,33,70,101} 幅值 ×30；
#   W:[256,512]×0.05。
#   · AWQ：3-bit per-output-channel 对称 RTN；α 网格 20 点；判据 = 层输出 MSE
#   · SmoothQuant：8-bit（激活 per-token 非对称、权重 per-output-channel 对称）；
#     「等价性」= `max|Xs@Ws.T − Y| / max|Y|`（float32 舍入量级即通过）
AWQ_MEASURED = {"argmin_alpha": 0.20, "rtn_out_mse": 1.345403e0,
                "best_out_mse": 4.991030e-1, "reduction_pct": 62.9,
                "alpha_0.50_out_mse": 1.014564e0, "worst_alpha": 0.95,
                "worst_out_mse": 1.529361e0}
SQ_MEASURED = {
    "equivalence_max_rel": 2.432e-7,        # 上界（21 个 α 里最大的那个，α=0.20）
    "argmin_alpha": 0.45,
    "out_mse_alpha_0.00": 9.548409e-3,      # 不迁移
    "out_mse_alpha_0.50": 1.179543e-3,
    "out_mse_alpha_1.00": 1.997518e-2,
    "act_max_before": 103.50,
    "act_max_after": 3.67,
    "wgt_max_before": 0.233,
    "wgt_max_after": 3.667,
    "rel_mse_8bit_alpha0.5": 8.7932e-5,
    "rel_mse_4bit_alpha0.5": 2.8498e-2,
}

# ===========================================================================
# 4. 舍入与 AdaRound（04-rounding）
# ===========================================================================

# --- 4.1 归属判定 -----------------------------------------------------------
# 来源：（计划基线，未二次核实）AdaRound 论文 arXiv:2004.10568 本身的**标题**：
#   "Up or Down? Adaptive Rounding for **Post-Training** Quantization"。
#   三条依据：① 标题含 "Post-Training"；② **不训练任何权重**，只学每个权重的舍入决策，
#   量化网格预先独立确定、之后固定；③ 只需 1024 张**无标签** ImageNet 图、不用任务损失。
ADAROUND_TITLE = "Up or Down? Adaptive Rounding for Post-Training Quantization"
ADAROUND_IS_PTQ = True
ADAROUND_TRAINS_WEIGHTS = False
ADAROUND_USES_LABELS = False
ADAROUND_IMAGES = 1024

# --- 4.2 软舍入函数 ---------------------------------------------------------
# 来源：（计划基线，未二次核实）AdaRound 式 (23)：
#   `h(V) = clip(σ(V)(ζ − γ) + γ, 0, 1)`，**ζ = 1.1、γ = −0.1 固定**；
#   正则 `f_reg(V) = Σ(1 − |2h(V) − 1|^β)`，β 退火（前期 β 高、后期 β 低）。
ADAROUND_ZETA = 1.1
ADAROUND_GAMMA = -0.1
ADAROUND_SOFT_ROUND_FORM = "clip(sigmoid(V)*(zeta-gamma)+gamma, 0, 1)"
ADAROUND_REG_FORM = "sum(1 - |2h(V)-1|^beta)"
ADAROUND_BETA_DIRECTION = "前期 beta 高、后期 beta 低"
# ⚠ **β 的退火区间与 λ 的数值，论文正文未印**（只说 "we anneal the parameter β"）。
#   **不得编具体数字。** 这一条常量记的就是「哪些量论文没印」这件事本身。
ADAROUND_UNPRINTED = ("beta 的退火区间", "λ（正则系数）的具体数值")

# 来源：（计划基线，未二次核实）AdaRound 实现细节：Adam 默认超参、10k iterations、
#   batch size 32、1024 张无标签 ImageNet、ResNet18 单次约 10 分钟（GTX 1080 Ti）。
ADAROUND_ITERS = 10000
ADAROUND_BATCH_SIZE = 32
# 来源：（计划基线，未二次核实）AdaRound 论文表 1：对 ResNet18 第一层做 4-bit，
#   **100 次随机舍入中有 48 次优于 RTN**；最好一次 63.06% vs Nearest 52.29%；
#   全层 AdaRound 4-bit 达 66.56%，RTN 只有 23.99%。
ADAROUND_RANDOM_BEATS_RTN = (48, 100)
ADAROUND_BEST_RANDOM_TOPK1 = 63.06
ADAROUND_NEAREST_TOPK1 = 52.29

# --- 4.3 本机实测基线 -------------------------------------------------------
# 来源：（本机实测）torch 2.11.0。
#   ★ 两组，**判据不同、结论方向不同**，必须分别标注：
#   (a) 判据 = **层输出 MSE**（线性层、合成权重）：
#       100 次随机舍入**优于 RTN 的次数 = 0 / 100**。
#       → 「随机舍入在逐元素意义上不可能赢 RTN」，这是定义决定的。
#   (b) 判据 = **任务准确率**（小 MLP + 合成二分类，只量化第一层到 4-bit，
#       RTN 准确率 0.9570，100 次随机舍入）：
#       **优于 RTN 的次数 = 39 / 100**，最好 0.9610、最差 0.9505、均值 0.9566。
#       → 复现的是论文 48/100 的**定性结构**（任务准确率上 RTN 不是最优），
#         **不是它的数值**（本机没有 ResNet18/ImageNet）。
ADAROUND_SR_VS_RTN_MEASURED = {
    # 判据 ① 层输出 MSE（线性层、合成权重）
    "output_mse_metric_wins": 0, "trials": 100,
    "mse_rtn": 5.492080e-2, "mse_best_random": 5.513e-2,
    # 判据 ② 任务准确率（小 MLP + 合成二分类，只量化第一层到 4-bit）
    "accuracy_metric_wins": 39,
    "rtn_accuracy": 0.9570, "best_random_accuracy": 0.9610,
    "worst_random_accuracy": 0.9505, "mean_random_accuracy": 0.9566,
    "fp32_accuracy": 0.9600,
}

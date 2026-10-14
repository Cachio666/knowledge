"""130-model-optimization/quantization 的外部规格常量源。

本文件只放两类东西：
  1. **外部常量**（教科书章节与式号、论文印值、规范章节号），每条带一行 `# 来源：`；
  2. **本机实测基线表**（与三篇文档 §5 逐行一致的数字），每条注明测它的版本与测量条件。

★ 核查范围（错误模式 P6 的对策）——**本主题的来源档位比 `numerical-formats` 低一档，必须写明**：
  `numerical-formats` 的常量大多来自本仓库能直取的页面（ONNX 文档页、torch 源码、
  公开规范）。**本主题的常量主要是论文印值，而本主题实现者没有逐字重读这些论文原文**
  （本轮网络受限：`WebFetch` 被策略拦截、`curl` 被 hook 拦截，可用通道只有 `WebSearch`
  与 python `urllib`）。所以：

  · 标 `（二次检索确认）` 的：本主题实现者 2026-09-18 用 `WebSearch` **看到过**
    支持该数字的检索摘要（仍未逐字读原文）；
  · 标 `（计划基线，未二次核实）` 的：只来自计划文件 §0.7.B 的事实基线，
    本主题实现者**本次没有拿到任何独立佐证**；
  · **两者在正文里都写成「引用值」**，因为「引用值」这个标签隐含的是「来源存在」，
    不是「本仓库复核过其测量口径」（CONTRIBUTING §2）。
  **不得把任何一个说成「本机实测」或「已核实」。**

★ 「规范」与「实现」分开标注（错误模式 P8）。本主题**没有**规范侧的条目——
  「6.02n + 1.76 是否适用于 LLM」是**数学命题**，不是规范命题。
  本主题的 P8 落在另一处：**「定点」与「浮点」是两套结论，混着用会自相矛盾**
  （见 FIXED_VS_FLOAT）。

★ 本文件**不放任何算法**，也**不得 import torch**（`verify.py` 在默认 python3 下 import 它）。
  凡被断言检验的算法（Δ/2 界、σ²=Δ²/12、SQNR 计算、四种粒度的量化）
  **一律在 `verify.py` 里另写**——理由见 CONTRIBUTING §2 的「参考实现必须独立」。

UNSOURCED（查不到一手来源，**正文不使用**）：
  · 「全模型 decode 的算术强度 = 1 FLOP/weight-byte」——只有二手博客的推导，
    本主题不落成常量（它属于邻域主题，本主题只是不写）。
  · Banner 附录高斯情形 Eq.(A.14) 的「排版笔误」——**本仓库读不到原文，无法确证**。
    正文只写「高斯情形的 α* 论文未印」，不写「论文有笔误」。

UNRESOLVED（来源冲突或本仓库无法判定，正文按「未解决」写，不落成常量）：
  · Banner 的 ResNet50 5.6%→60.8% / InceptionV3 34.1%→56.8%（8W3A ImageNet Top-1）：
    本主题实现者 2026-09-18 二次检索**未拿到佐证**（检索摘要只命中同名的其它表格）。
    正文引用时**必须**带这句限定语。
"""

# ===========================================================================
# 0. 测量条件（文档 §5 直接引用这一节）
# ===========================================================================

# 本主题的编号程序在真 torch 上跑；下面五个解释器都跑过全三个程序。
# 来源：本机实测（python3 -c 'import torch; print(torch.__version__)'），2026-09-18。
MEASURED_VERSIONS = ("2.6.0", "2.7.1", "2.8.0", "2.11.0", "2.13.0")
UNCOVERED_MINORS = "2.9.x / 2.10.x / 2.12.x"
MEASURE_HOST = "Apple M1 Pro（10 核）/ macOS / **CPU-only，无 GPU 数字**"

# ★ 本主题的版本下限**不是依赖，是对齐的选择**。
#   来源：本主题代码自己的依赖清单（三个编号程序只用到 float32/float64/bfloat16/
#   float16/float8_e4m3fn 与 torch.quantize_per_tensor，2.6.0 起全部存在）。
#   ⚠ 写这一条是为了**避免 P4（编机制从句）**：本域 `numerical-formats` 的下限
#     ≥2.8 有真实依赖（`torch.float8_e8m0fnu`），**本主题没有**。
#     若照抄那句理由，就会说出一句「程序会崩在找不到某个 dtype 上」的假话。
TORCH_MIN_FOR_THIS_TOPIC = (2, 8)
TORCH_MIN_REASON = "对齐选择（不是依赖）：使 make run 在装多个 torch 的机器上选中与本域其余主题同一个解释器，§5 的测量条件全域可比"


# ===========================================================================
# 1. 量化误差界（教科书）
# ===========================================================================

# 来源：A. V. Oppenheim, R. W. Schafer, *Discrete-Time Signal Processing*, 3rd ed.,
#       §4.8.3「量化误差的分析」，式 (4.128)–(4.133)。MIT OCW 官方 PDF（计划基线给出该出处）。
# 口径：**误差定义 e[n] ≜ x̂[n] − x[n]（量化后减量化前）**。⚠ 符号约定必须写明：
#       换成 e ≜ x − x̂，「恒为负」就变成「恒为正」（见 OAS_SIGN_CONVENTION）。
OAS_CITATION = "Oppenheim & Schafer, Discrete-Time Signal Processing, 3rd ed., §4.8.3, eq (4.128)-(4.133)"
OAS_SIGN_CONVENTION = "e[n] ≜ x̂[n] − x[n]（量化后 − 量化前）"

# 来源：同上 §4.8.3 eq (4.132)。
# 口径：B+1 位（含符号位）均匀量化、**最近舍入**、**输入未越界**。界是精确的，不是近似的。
OAS_ROUNDING_BOUND = (-0.5, 0.5)      # 以 Δ 为单位：−Δ/2 ≤ e < Δ/2

# 来源：同上 §4.8.3，截断（round-toward-−∞ / two's-complement 截尾）那一支。
# 口径：同一个 e ≜ x̂ − x。**恒 ≤ 0**，幅度上界是舍入的两倍。
OAS_TRUNCATION_BOUND = (-1.0, 0.0)    # 以 Δ 为单位：−Δ < e ≤ 0

# 来源：同上 §4.8.3 eq (4.133)：σ_e² = Δ²/12。
# 口径：四条经典假设「容易不成立」，教科书自己声明只在**步长小、信号复杂、未过载**时是好近似。
OAS_QUANT_NOISE_VARIANCE_NUM = 1
OAS_QUANT_NOISE_VARIANCE_DEN = 12     # σ_e² = Δ² / 12

# 来源：同上 §4.8.3 正文（计划基线转述）。
OAS_FOUR_ASSUMPTIONS = (
    "误差序列 e[n] 是平稳随机过程",
    "e[n] 与 x[n] 不相关",
    "误差样本之间互不相关",
    "误差在 [−Δ/2, Δ/2] 内均匀分布",
)
OAS_ASSUMPTION_CAVEAT = "教科书自述这四条「容易不成立」，仅在步长小、信号复杂、未过载时是好近似"


# ===========================================================================
# 2. SQNR = 6.02B + 1.76 与它的两种自洽口径
# ===========================================================================

# 来源：Analog Devices, MT-001（W. Kester）"Taking the Mystery out of the Infamous
#       Formula: SNR = 6.02N + 1.76dB"，式 (1)–(9)。2026-09-18 二次检索确认该文存在
#       且「full-scale sine / Nyquist band」这一前提的表述（二次检索确认）。
# 口径：**满量程正弦**、理想 B 位 ADC、量化噪声在 [−Δ/2, Δ/2] 均匀、测到 Nyquist 带。
MT001_CITATION = "Analog Devices MT-001 (W. Kester), eq (1)-(9)"
MT001_FULL_SCALE_SINE_SNR_DB = 1.76
MT001_SLOPE_DB_PER_BIT = 6.02
MT001_PRECONDITION = "前提是**满量程正弦**；不是「任意信号下都成立」"

# ★ 两种**各自自洽**的一般式。**不得混用**（混用会给 55.95 dB，比正确值高 6 dB）。
#   来源：两式的**常数**在 2026-09-18 的二次检索里被独立印证：
#     · `10.8 = 10·log10(12)`（跨度口径的常数，来源里写作 `10.8 + 6.0206b + 10log10(σx²/V_FS²)`，
#        其中 V_FS 是**峰峰满量程**）；
#     · `4.77 = 10·log10(12) − 20·log10(2)`（峰值口径的常数，来源里写作 `6.02n + α`，
#        `α = 4.77 − 20log10(V/x_rms)`，其中 V 是**峰值幅度**）。
#     **两个常数只差 20log10(2) = 6.02 dB，差别就是「Xm 指跨度还是峰值」。**（二次检索确认）
SQNR_CONVENTIONS = {
    # 名字 → (Xm 的含义, 常数项, 一般式, 正弦下的退化条件, B=8 时的退化结果)
    "A_span": {
        "Xm_meaning": "满量程跨度（peak-to-peak）",
        "const_db": 10.8,
        "general": "SNR = 6.02B + 10.8 − 20log10(Xm/σx)",
        "degenerate_when": "σx = Xm/(2√2)",
        "at_B8_db": 49.93,
    },
    "B_peak": {
        "Xm_meaning": "峰值幅度（peak）",
        "const_db": 4.77,
        "general": "SNR = 6.02B + 4.77 − 20log10(Xm/σx)",
        "degenerate_when": "σx = Xm/√2",
        "at_B8_db": 49.92,
    },
}
# 来源：上两式的常数（二次检索确认）。**判据**：两种口径都必须能退化到 `6.02B + 1.76`。
# 口径：10.8 = 10log10(12)；4.77 = 10log10(12) − 20log10(2)。
SQNR_CONST_DERIVATION = {
    "10log10(12)": 10.7918,
    "20log10(2)": 6.0206,
    "span_const": 10.7918,                 # = 10log10(12)
    "peak_const": 10.7918 - 6.0206,        # = 4.7712 ≈ 4.77
    "degenerate_sigma_over_Xm_span": 1.0 / (2 * 2 ** 0.5),   # σx/Xm = 1/(2√2)
    "degenerate_sigma_over_Xm_peak": 1.0 / (2 ** 0.5),       # σx/Xm = 1/√2
}
# ★ **混用**的后果（本仓库推导，见 verify.py §A3）：把 A 口径的 10.8 配上 B 口径的
#   σx = Xm/√2，B=8 时给出 **55.95 dB**，比正确值高 6.02 dB。
SQNR_MIXED_CONVENTION_DB_AT_B8 = 55.95   # 错误值，只用来当反例

# 本机实测（满量程正弦、mid-tread、step = 满量程跨度 / 2^B）：
# 来源：本机实测，`code/05_error_bounds.py` §A4（torch 2.11.0 / Python 3.12.9 与其余四个版本）。
# 口径：x_k = sin(2π·frac(k·φ))，φ = (√5−1)/2（**无理相位**，避免与采样格点共振），
#       N = 10^5；mid-tread 定义 `x̂ = round(x/step)·step`，`step = 2/2^B`。
SQNR_SINE_MEASURED = {8: 50.017, 16: 98.087}
SQNR_SINE_FORMULA = {8: 49.92, 16: 98.08}
# ⚠ 它**略高于** Δ²/12 的解析预测（50.02 vs 49.92），成因是**正弦的幅度分布不是均匀的**：
#   实测误差方差 / (Δ²/12) = 0.979（同 §A4）。
# ★ 这个实测值**依赖量化定义**。本机实测的四个定义（同 §C3，同一批样本）：
SQNR_SINE_BY_DEFINITION = {
    "mid-tread round(x/step)·step, step = 2/2^B": 50.017,   # 本文口径
    "round(x·2^(n−1))/2^(n−1)": 50.017,                     # **代数上与上一行等价**，不是另一个定义
    "对称 /127（restricted range）": 49.949,
    "mid-rise (floor(x/step)+0.5)·step": 49.788,
    "对称 /127.5": 49.753,
    "截断 floor(x/step)·step": 43.867,
}
# ⚠ **本仓库未能复现计划 §0.7.B 记的「`round(x·2^{n−1})` 给 49.12 dB」**：
#   在 span = 2、B = 8 下，`round(x·2^{n−1})/2^{n−1}` 与 mid-tread 的
#   `round(x/step)·step`（step = 2/2^B = 1/128）**是同一个式子**，实测同为 50.017。
#   → 正文不写 49.12，也不把那一项当成「另一种定义」（见程序 §C3 的断言）。

# 浮点 vs 定点：**同一个「6.02n+1.76 是否适用」的问题，两套答案**（P8 的落点）。
# 来源：本机实测，`code/05_error_bounds.py` §A5（torch 2.11.0，2×10^6 样本，
#       四个分布：N(0,1) / U(−1,1) / Laplace(0,1) / 单边指数）。
FIXED_VS_FLOAT = {
    # 判据：per-tensor min-max 对称 8-bit（QMAX = 127），N = 2×10^6，
    #       每个分布一个**专属种子**（9000+i），生成式见 `05_error_bounds.py` §A7。
    "fixed_8bit_sym_sqnr_db": {"N(0,1)": 38.90, "U(-1,1)": 48.09,
                               "Lap(0,1)": 32.57, "单边指数": 32.82},
    # 判据：同一批样本直接 `.to(dtype).to(float32)`，种子 9100+i。
    "float_sqnr_db": {
        "bfloat16": {"N(0,1)": 55.59, "U(-1,1)": 56.61, "Lap(0,1)": 55.61, "单边指数": 55.60},
        "float16": {"N(0,1)": 73.65, "U(-1,1)": 74.68, "Lap(0,1)": 73.67, "单边指数": 73.66},
        "float8_e4m3fn": {"N(0,1)": 31.52, "U(-1,1)": 32.54, "Lap(0,1)": 31.55, "单边指数": 31.53},
    },
    "fixed_spread_db": 15.52,   # 48.09 − 32.57（对称 8 bit 定点，同上判据）
    "float_spread_db": {"bfloat16": 1.02, "float16": 1.02, "float8_e4m3fn": 1.02},
}
# ⚠ **不要写「浮点的跨分布波动 < 1 dB」**：本机实测三个格式都是 **1.02 dB**，
#   两端点自己就超了 1 dB（计划 §0.7.B 明确禁掉这个写法）。判据：跨四个分布的最大−最小。
# ⚠ **也不要把「定点 15.52 dB」当成一个稳定的数**：它用的两格是 per-tensor min-max，
#   而那个量本身是随机的（见 MINMAX_SEED_SENSITIVITY）。定性结论（定点强分布、浮点弱分布）
#   稳，具体差值不稳。

# ★ per-tensor min-max 的 SQNR **是一个随机变量，不是一个数**。
# 来源：本机实测，`code/05_error_bounds.py` §B2（torch 2.11.0）。
# 判据：对称 8-bit、per-tensor min-max、N = 2×10^5 样本、**100 个不同的随机种子**，
#       统计 SQNR 的最小值 / 最大值 / 极差。
MINMAX_SEED_SENSITIVITY = {
    "N(0,1)": {"min": 38.18, "max": 40.49, "spread": 2.31},
    "Lap(0,1)": {"min": 29.91, "max": 35.20, "spread": 5.28},
}
MINMAX_SEED_SENSITIVITY_NOTE = (
    "成因：scale 完全由**样本极值**决定，而极值本身是小样本统计量。"
    "**这就是「per-tensor 那几格不要逐格引」的定量依据。**"
)


# ===========================================================================
# 3. 粒度（per-tensor / per-token / per-channel / per-group）
# ===========================================================================

# ★ 轴约定（**必须先写死**，否则两个标签互换而没有任何报错——P9 实例 2）。
# 来源：transformers 的通行布局 + 计划 §0.7.B 的表（该表已按此约定重排）。
AXIS_CONVENTION = {
    "layout": "[tokens, features]（行 = token，列 = feature/channel）",
    "row_scale": "每行一个 scale ⇒ **per-token**（沿 axis 1 归约）",
    "col_scale": "每列一个 scale ⇒ **per-channel**（沿 axis 0 归约）",
}
AXIS_CONVENTION_NOTE = (
    "plan §0.7.B 的初版把这两个标签写反；写反不会报错，"
    "只会推出「K 应该用 per-token」——与 KIVI 正好相反"
)

# 本机实测：5 个场景 × 4 种粒度（对称 8-bit RTN，QMAX = 127，(4096, 4096) 张量）。
# 来源：本机实测，`code/06_granularity.py`（torch 2.11.0；五个版本逐格一致）。
# 口径（**判据**）：scale = 该归约域内 max|x| / 127；量化 = clip(round(x/s), −127, 127)·s；
#   SQNR = 10log10(Σx²/Σ(x−x̂)²)；per-group 的 G = 128，**切分沿 feature（最后一维）**；
#   两个异质场景的尺度因子取 10^U(−3,0)；outlier 场景是 **7 列（随机位置）× 100**。
GRANULARITY_TABLE = {
    # 场景 → (per-tensor, per-token, per-channel, per-group(128))
    "iid 高斯":                     (37.44, 41.23, 41.24, 43.79),
    "iid 拉普拉斯":                 (31.09, 36.81, 36.82, 40.97),
    "行异质（每 token 尺度跨 1e-3..1e0）": (29.16, 41.23, 33.93, 43.79),
    "列异质（每 channel 尺度跨 1e-3..1e0）": (29.46, 34.05, 41.22, 38.33),
    "7/4096 列（随机位置）×100":     (13.99, 20.33, 41.41, 32.08),
}
GRANULARITY_TABLE_SEED = 20260918
# ⚠ **per-tensor 那一列的三格复现偏差较大**（0.4–2.1 dB，随随机种子变）：
#   它的 scale 完全由**构造出的极值**决定，对场景的生成方式敏感。
#   **正文以「per-tensor 最差」这个定性结论为准，不逐格引这三个数**（计划 §0.7.B 的要求）。
#   其余 16 格跨三个种子复现偏差 ≤ 0.3 dB。
GRANULARITY_REPRO_TOL_DB = 0.35
GRANULARITY_TENSOR_COL_CELLS_SENSITIVE = ("iid 高斯", "iid 拉普拉斯", "行异质")

# ★ **per-group 的切分维度必须另写**——各论文不统一，不写这一维 group size 就不可比。
# 来源（三条都来自计划基线）：Q-BERT 沿**输出维**、GPTQ/nuQmm 沿**输入维**、
#       Dettmers 等把张量**拉平**。
PER_GROUP_SPLIT_DIM = {
    "Q-BERT": "沿输出维",
    "GPTQ / nuQmm": "沿输入维",
    "Dettmers et al.": "把张量拉平",
}
PER_GROUP_SPLIT_DIM_SOURCE = "计划 §0.7.B（未二次核实）"
# 本机实测（同一个 outlier 场景，只换切分维度）：沿 feature 32.08 / 沿 token 43.89 / 拉平 32.08
# 来源：本机实测，`code/06_granularity.py` §B4。

# ★ **「group size = 128」的出处**（P9 实例 3：那是**组数**不是**组大小**）。
# 来源：Q-BERT, arXiv:1909.05840 —— ⚠ **不是** arXiv:1905.03696（那是 HAWQ，编号极易搞错）。
# 口径：该表的列名是 **"# groups"（组数）**，取值 1 / 12 / 128 / 768。
QBERT_ARXIV = "arXiv:1909.05840"
QBERT_NOT_THIS_ARXIV = "arXiv:1905.03696"        # 这是 HAWQ，不是 Q-BERT
QBERT_GROUP_SWEEP = {"# groups": (1, 12, 128, 768),
                     "SST-2": (85.67, 92.31, 92.66, 92.78),
                     "baseline_SST2": 93.00}
# ⚠ **表号在来源间不一致**：本仓库检索到的镜像页把该表标为 **Table 2**，
#   而同一批检索摘要称它是 **Table 3**。**正文不写死表号**，只写「group-wise 消融表」。
QBERT_TABLE_NUMBER_CONFLICT = "镜像页标 Table 2，检索摘要称 Table 3 —— 正文不写死表号"
# 来源：同上（二次检索确认）：1→12 组把退化从 7–11.5% 压到 2% 以内；
#   12→128 组在 SST-2 上再涨 0.12（92.66−92.54… 实测列到 92.66）；128→768 只涨 0.12。
QBERT_128_TO_768_DELTA_SST2 = 92.78 - 92.66      # = 0.12
QBERT_CONCLUSION = "组数超过 128 后收益饱和；组数越多 LUT 越多、硬件效率越差"
# ⚠ 对 768 宽的矩阵，**128 组 ≈ group size 6**。**用一张「组数」表去当
#   「group size = 128」的出处，是作用域不匹配（P7）。**

# 论文侧对照（引用）：SmoothQuant Table 1，OPT 系列 activation 量化平均精度。
# 来源：SmoothQuant, arXiv:2211.10438, Table 1（计划基线，未二次核实）。
# 口径：**该表用 pdftotext -layout 提取时行标有错位**，计划记录研究员按语义判定的对齐。
#       **本主题实现者无法读原文，引用时保留这句限定语。**
SMOOTHQUANT_TABLE1_OPT = {
    "6.7B": {"FP16": 64.9, "per-tensor": 39.9, "per-token": 42.5, "per-channel": 64.8},
    "175B": {"FP16": 71.6, "per-tensor": 32.3, "per-token": 31.7, "per-channel": 71.4},
}
SMOOTHQUANT_TABLE1_CAVEAT = "该表的行标在 pdftotext -layout 提取下错位，对齐由研究员按语义判定；本主题实现者未读原文"


# ===========================================================================
# 4. 最优 clipping（ACIQ）
# ===========================================================================

# 来源：Banner, Nahshan, Hoffer, Soudry, *Post-training 4-bit quantization of
#       convolution networks for rapid-deployment* / *ACIQ*, arXiv:1810.05723v3。
# 口径：**论文印值**。Laplace(0, b) 下的解析最优裁剪阈值 α*，单位是 b。
ACIQ_ALPHA_PAPER = {2: 2.83, 3: 3.89, 4: 5.03}
ACIQ_CAVEAT = (
    "**这是论文印值**；2026-09-18 的二次检索**未能佐证这组常数**（只确认了 ACIQ 的"
    "框架：解析裁剪、最小化量化噪声与裁剪失真的权衡）。"
)
# 本机数值解（**另标一档，不得与论文印值并列成同一个量**）。
# 来源：本机数值解，`code/05_error_bounds.py` §A6。判据（**目标函数必须写出**）：
#   MSE(α) = (1 − P(|x|>α))·Δ²/12 + 2∫_α^∞ (x−α)²·Laplace(0,b)(x)dx，Δ = 2α/2^M，b = 1。
ACIQ_ALPHA_LOCAL_NUMERIC = {2: 2.813, 3: 3.882, 4: 5.021}
ACIQ_ALPHA_LOCAL_CAVEAT = "与论文印值差 0.2–0.6%；差异来自目标函数的细节（Δ 的定义、尾部积分的近似）"

# 来源：同上（研究员换算，**论文未印**）。P(|X| > α*) = e^(−α*/b)。
ACIQ_SATURATION_PROB = {2: 0.0590, 3: 0.0204, 4: 0.0065}
ACIQ_SATURATION_PROB_CAVEAT = "换算是研究员自算、论文未印；本仓库复核了 e^(−α/b) 这一步的算术"

# 来源：同上 §2（计划基线，未二次核实）：点名 GEMMLOWP 的「裁剪到 max」是次优做法。
ACIQ_CLIP_TO_MAX_IS_SUBOPTIMAL = "arXiv:1810.05723v3 §2，原文点名 GEMMLOWP 的做法"
# 机理（**定性**）：Δ = 2α/2^M；舍入噪声 ∝ α²/(3·2^{2M})。
# SmoothQuant 的等价表述（引用）：非 outlier 通道的有效量化级数会低到 2–3 级。
# ⚠ **不要写「max 裁剪的 MSE 是 α* 裁剪的 N 倍」这个具体倍数**：
#   计划基线的控制者测得 Laplace 5.56× / 高斯 3.17×（M=4），审查者测得 4.63× / 2.38×，
#   **不具备跨实现的可复现性**。正文只写「数倍」与定性方向。
CLIP_MAX_VS_OPTIMAL_QUALITATIVE = "max 裁剪明显次优，MSE 差**数倍**；具体倍数不可复现，不给数值"
CLIP_MAX_VS_OPTIMAL_FORBIDDEN = "5.56× / 3.17× / 4.63× / 2.38× 这四个数不得写进正文"

# 来源：Banner Table 2（8W3A ImageNet Top-1）（计划基线，未二次核实）。
# ⚠ **本主题实现者 2026-09-18 二次检索未拿到佐证**（检索摘要只命中同名的其它表格）。
ACIQ_TABLE2_IMAGENET = {"ResNet50": (5.6, 60.8), "InceptionV3": (34.1, 56.8)}
ACIQ_TABLE2_CAVEAT = "引用时必须带：本主题实现者的二次检索未佐证这两个增量"
# 来源：Banner 附录，高斯情形的 α*（**论文未印**；计划记录该式末项疑有排版问题）。
ACIQ_GAUSSIAN_ALPHA = "UNSOURCED —— 论文未印，本仓库也读不到原文；正文写「论文未印」，不写「论文有笔误」"


# ===========================================================================
# 5. outlier 与 massive activations
# ===========================================================================

# 来源：Dettmers et al., *LLM.int8()*, arXiv:2208.07339v2 §1（**二次检索确认**三条阈值）。
# 口径：**形式化定义**——三个条件同时成立才算 outlier feature。
LLMINT8_OUTLIER_DEF = {
    "magnitude": 6.0,
    "fraction_of_layers": 0.25,
    "fraction_of_sequence_dims": 0.06,
}
LLMINT8_OUTLIER_DEF_SOURCE = "arXiv:2208.07339v2 §1（二次检索确认；仍未逐字读原文）"
LLMINT8_MAGNITUDE_6_REASON = "把幅值 ≥6 的当 outlier 处理，perplexity 就不再退化"
LLMINT8_MAX_MAGNITUDE_RATIO = "可达其它维度的 20×"

# 来源：同上（计划基线，未二次核实）。
LLMINT8_SPARSITY = {
    "model": "13B",
    "max_outliers_per_2048_tokens": 150000,
    "max_hidden_dims": 7,
    "fraction_of_feature_dims": 0.001,     # ≈ 0.1%
}
# 来源：同上（**二次检索确认**位置那一句）。
LLMINT8_LOCATION = "只出现在 attention 投影层（k/q/v/o）与 FFN 扩张层（第一子层）；attention 函数本身与 FFN 收缩层没有"
# 来源：同上（计划基线，未二次核实）：**6B → 6.7B 之间突然涌现**。
LLMINT8_EMERGENCE = {"affected_layers": (0.65, 1.00), "affected_seq_dims": (0.35, 0.75)}
# 来源：同上（二次检索确认了「移除 <7 个 outlier 特征使 top-1 softmax 概率减半、
#       验证困惑度升 6–10×」这一句；计划基线给的是 40%→20% 与 600–1000%）。
LLMINT8_IMPORTANCE = {
    "remove_outlier_dims": {"top1_softmax": (0.40, 0.20), "ppl_increase": (6.0, 10.0)},
    "remove_7_random_dims": {"top1_softmax_drop": (0.0002, 0.003), "ppl_increase": 0.001},
}
# 来源：同上（计划基线，未二次核实）：**符号结构**。
LLMINT8_SIGN_STRUCTURE = "几乎所有 outlier 严格非对称——要么全正要么全负；这正是 zero-point（非对称）量化对它们特别有效的原因"

# ★ **massive activations ≠ outlier features**（本域最常见的错误之一）。
# 来源：Sun et al., *Massive Activations in Large Language Models*, arXiv:2402.17762v2
#       （**二次检索确认**：原文明确写 "we show that massive activations are not the same
#        as outlier features"）。该文为 ICLR 投稿/会议论文（OpenReview 有公开 PDF）。
MASSIVE_VS_OUTLIER = {
    "massive": {
        "what": "**标量**（由序列维与特征维共同确定）",
        "count": "数千万个 activation 里常常不到 10 个",
        "magnitude": "比中位数大 4 个数量级以上；LLaMA2-70B 里绝对值可 > 15000",
        "location": "两个固定 feature 维度 × 两类 token（序列起始、第一个 `.` 或 `\\n`）",
    },
    "outlier_features": {
        "what": "**向量**（在某个 feature 维上的全部 token）",
        "count": "13B 下每 2048 token 最多 150k 个，但只落在至多 7 个 hidden 维度",
        "magnitude": "≥ 6.0 即计入；可达其它维度 20×",
        "location": "遍布所有 token；占全部 feature 维度约 0.1%",
    },
    "overlap": "该文实测：LLaMA2-7B / 13B 各识别出 10 / 25 个 outlier feature，**没有一个**与 massive activation 的 feature 维重合",
}
MASSIVE_VS_OUTLIER_SOURCE = "arXiv:2402.17762v2（二次检索确认；仍未逐字读原文）"
MASSIVE_LOOSE_DEF = "该文的宽松定义：幅值 > 100 且约为该 hidden state 中位数的 1000 倍"

# 来源：SmoothQuant, arXiv:2211.10438v7 §3（计划基线，未二次核实）：**持久性**。
OUTLIER_PERSISTENCE = "某通道有 outlier 则它在**所有 token 上**都出现；token 内通道间方差很大，但同一通道跨 token 的幅值方差很小"
# 本机实测（`code/07_outliers.py` §B2）：持久列 outlier 下 per-token 与 per-channel 的差距。

# activation 比 weight 难的三条机理（来源：LLM.int8() / SmoothQuant，计划基线转述）。
ACTIVATION_HARDER_WHY = (
    "outlier 在 feature 维（矩阵的列）",
    "权重分布平坦均匀、activation 长尾",
    "weight 可以 per-channel（硬件允许），activation 的 per-channel 会插进 GEMM 内维、硬件不可行，只能退到 per-token",
)


# ===========================================================================
# 6. 误差传播
# ===========================================================================

# 来源：QEP, arXiv:2504.09629v3, Fig.2（计划基线，未二次核实）。
QEP_OBSERVATION = "层间量化误差累积，在量化层内近似指数增长，且在**未量化的后续层继续增长**"
# 一阶理论（来源：同上）：δ_l = −Σ_{k<l}(∏_{s=k+1}^{l−1} J_s W_s)·J_k E_k X_k + O(max‖E_k‖²)
QEP_FIRST_ORDER = "δ_l = −Σ_{k<l}(∏_{s=k+1}^{l−1} J_s W_s)·J_k E_k X_k + O(max‖E_k‖²)"
QEP_SPECTRAL_CRITERION = "连乘项谱半径 > 1 则指数放大"
# 来源：arXiv:2608.08188v1（**2026 新预印本，未见同行评审**）：传播项 G_l·R_l 在早/中/晚
#       三阶段都主导总误差；小模型中层持续放大（G_l>1），大模型基本保持或轻微衰减（G_l<1）。
NEW_PREPRINT_2608 = "arXiv:2608.08188v1（2026 新预印本、未见同行评审）"
NEW_PREPRINT_2608_FINDING = "传播项 G_l·R_l 在早/中/晚三阶段都主导总误差；小模型中层 G_l>1 持续放大，大模型 G_l<1 基本保持"

# ★ 一条**不能外推**的结论。
# 来源：arXiv:2510.21770v2 Thm.2（计划基线，未二次核实）。
RESIDUAL_ADDITIVE_RELAXATION_SOURCE = "arXiv:2510.21770v2 Thm.2"
RESIDUAL_ADDITIVE_RELAXATION = (
    "残差连接把深度方向误差从乘性累积松弛为近似加性 —— "
    "**但那是针对浮点舍入误差（ε_mach 一阶）的，不是 bit-width 量化误差**"
)
RESIDUAL_ADDITIVE_FORBIDDEN = "不得把它当作量化误差界的证明"

# ★ 「层间对消」一侧**没有任何正面证据**。
CANCELLATION_EVIDENCE = "研究员检索未覆盖，但也未找到；**不得写「误差会互相抵消」**"


# ===========================================================================
# 7. 量化误差 vs 模型质量（引用值；harness / 表号 / 基线要写全）
# ===========================================================================

# ★ 四条可定位的「断崖」结论。
# 来源：计划基线（未二次核实）。**每条都带 harness**。
QUANT_QUALITY = {
    "RTN_3bit_OPT-175B_ppl": {"value": 7.3e3, "harness": "GPTQ 论文的评测口径"},
    "RTN_3bit_BLOOM-176B_ppl": {"value": 571, "harness": "同上"},
    "GPTQ_4bit_OPT-175B_ppl": {"value": 8.37, "harness": "GPTQ 论文 Table 5，WikiText2"},
    "FP16_OPT-175B_ppl": {"value": 8.34, "harness": "同上（**同一张表**）"},
    "GPTQ_3bit_OPT-175B_ppl": {"value": 8.68, "harness": "同上"},
    "RTN_4bit_OPT-175B_ppl": {"value": 10.54, "harness": "同上"},
}
# 来源：arXiv:2604.19884v1（**2026 新预印本，未见同行评审**）。
COLLAPSE_VS_DEGRADATION = "4-bit 是 Signal Degradation（计算模式完整）；2-bit 触发 Computation Collapse（关键组件失效，信号在早期层就被摧毁）"
COLLAPSE_SOURCE = "arXiv:2604.19884v1（2026 新预印本、未见同行评审）"
# 来源：arXiv:2608.08188v1（2026 新预印本）。
BITWIDTH_NOT_ENOUGH = "degradation is not determined by bit-width alone"
# ⚠ **同一模型两处数字不一致的实例（不得并列引用）**：
#   AWQ 论文 Figure 2 说 OPT-6.7B INT3-g128 RTN 是 43.2，Table 2 说 23.54，论文未说明口径差异。
AWQ_SELF_INCONSISTENT = "AWQ Figure 2 = 43.2 vs Table 2 = 23.54（OPT-6.7B INT3-g128 RTN）；**不得并列引用**"


# ===========================================================================
# 8. 本机实测基线（供 verify.py 与文档 §5 逐行咬合）
# ===========================================================================

# 全部来自 `code/0{1,2,3}_*.py`（torch 2.11.0 / Python 3.12.9；五个版本逐格一致）。
# 口径逐条写在上面各节。

# §A：Δ/2 界的数值验证（N = 2×10^6）
BOUND_CHECK = {
    "delta_over_2_holds": True,           # max|e| == Δ/2（未越界时，精确到 float64 舍入）
    "var_ratio_range": (0.995, 1.005),     # Var(e) / (Δ²/12)，判据见程序 §A2
    "truncation_rms_ratio": 1.998,         # 截断 RMS / 舍入 RMS → 2
    "overload_violation": {"B": 3, "cover": 1.0, "input_range": 2.0,
                           "max_err_over_delta_half": 8.0, "fraction_exceeding": 0.438},
}

# §B：代价模型的解析式（**这个式子是 §B1 断言的参照**）
#   SQNR_analytic(b) = 10·log10(12) + 20·log10(QMAX(b) / amax)，QMAX(b) = 2^(b−1) − 1
# 判据：per-tensor min-max 对称 RTN、对称受限量程（用 [−QMAX, QMAX]）、N(0,1)、N = 2×10^6。
# 来源：本机实测（`05_error_bounds.py` §B1）。
COST_MODEL_ANALYTIC = "SQNR(b) = 10log10(12) + 20log10((2^(b−1) − 1) / amax)"
COST_MODEL_FIT = {  # b → (实测, 解析, 差)
    3: (6.286, 6.280, 0.005), 4: (13.645, 13.640, 0.005),
    6: (26.569, 26.565, 0.004), 8: (38.819, 38.814, 0.005),
    10: (50.906, 50.906, 0.000), 12: (62.959, 62.960, -0.001),
}
COST_MODEL_B2_FAILS = (0.170, -3.262, 3.432)   # b=2：(实测, 解析, 差) —— **模型在 b=2 失效**
COST_MODEL_SLOPE = {  # (b 区间) → 实测 dB/bit（**有限 QMAX 修正**使它在小 b 处偏陡）
    (3, 4): 7.359, (4, 6): 6.462, (6, 8): 6.125, (8, 10): 6.043, (10, 12): 6.027,
}
COST_MODEL_SLOPE_NOTE = "斜率随 b 增大收敛到 6.02；小 b 处偏陡是因为 QMAX = 2^(b−1)−1 而不是 2^(b−1)"
COST_MODEL_SCALE_OVERHEAD_BITS = {"per-tensor": "≈0（32 bit / 整张量）", 128: 0.25, 32: 1.0, 16: 2.0}


# ===========================================================================
# 9. 术语（正文用到的、本文件里出现过的专名）
# ===========================================================================

# ===========================================================================
# 8b. outlier 构型与传播实验（03 的实测基线）
# ===========================================================================

# 来源：本机实测，`code/07_outliers.py`。判据（**必须连判据一起引**）：
#   [tokens, features] = (2048, 4096)、对称 8-bit RTN（QMAX = 127）、种子 2026；
#   SQNR = 10log10(Σx²/Σ(x−x̂)²)；"非极端" = |x| ≤ 10 的子集。
OUTLIER_SEED = 2026
OUTLIER_TABLE = {   # 场景 → (per-tensor, per-token, per-channel, per-group(128))
    "R0 干净": (37.75, 41.23, 41.65, 43.79),
    "R1 持久列 outlier（4 列，全 token）": (29.81, 34.75, 41.76, 43.06),
    "R2 列 outlier（4 列，仅 6% token）": (31.46, 40.37, 41.62, 43.74),
    "R3 massive activations（4 个标量）": (20.35, 48.52, 52.89, 60.32),
}
OUTLIER_TABLE_CAVEAT = (
    "**per-tensor 那一列对构型细节敏感**；正文的结论落在**非极端元素上的附带损伤**"
    "（`07_outliers.py` §A2），不落在全局 SQNR 上 —— 后者会被极端值自己主导（§C3）。"
)
# 来源：SmoothQuant / LLM.int8()（计划基线）：outlier 的持久性。
OUTLIER_PERSISTENCE_MEASURED = (
    "R1 **按构造**在全部 2048 个 token 上都有 outlier 列"
    "（判据：阈值 |x|>5 时 99% 的 token 至少含一个极端值；未到 100% 只是赋值抖动）"
)

TERMS = {
    "mid-tread": "量化台阶上有一级**恰好落在零点**（`round(x/Δ)·Δ` 即是）；"
                 "与之相对的是 mid-rise。**「n=8 → 50.02」这个实测值只在 mid-tread 下成立。**",
    "overload / 越界": "输入落在量化器覆盖范围之外。**Δ/2 界此时失效**——误差上界改由「到最近饱和点的距离」决定。",
    "per-token": "沿 **token 维（行）** 归约出 scale：每个 token 一个。**只治「行异质」。**",
    "per-channel": "沿 **channel 维（列）** 归约出 scale：每个 channel 一个。**只治「列异质」。**",
    "outlier feature": "LLM.int8() 定义的 outlier：某个 **feature 维** 上幅值 ≥6.0，且该维在 ≥25% 的层、"
                       "≥6% 的序列维上都出现。**是一个向量（在全部 token 上）。**",
    "massive activation": "**一个标量**：某个 (token, feature) 位置上的极端激活，比中位数大 4 个数量级以上，"
                          "只出现在两个固定 feature 维与两类 token。**与 outlier feature 不是同一现象。**",
    "group-wise": "每连续 G 个元素共享一个 scale。**「沿哪一维切」必须写明**——各论文不统一。",
    "SQNR": "信号量化噪声比，`10log10(Σx²/Σ(x−x̂)²)`。**判据（沿哪一维、什么归一化）必须与数字一起写。**",
}

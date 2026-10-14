"""130-model-optimization/quantization 的外部规格常量源。

本文件只放两类东西：
  1. **从一手来源抄来的**外部规格（ONNX 规范、OCP MX 规范与论文、PTX/白皮书、
     arXiv 论文的表号与印值、torch 源码行），每条带一行 `# 来源：`；
  2. **本机实测基线表**（与文档 §5 逐行一致的数字），每条注明测它的版本与测量条件。

★ 核查范围（错误模式 P6 的对策）：逐条写明核的是**哪个载体**——规范网页正文 /
  Changelog 的哪一节 / 论文哪张表 / 本机哪个解释器跑出来的。
  **不写「全库」「已核实全部」这类大于实际核查范围的措辞。**

★ 「规范」与「实现」分开标注（错误模式 P8）。本主题有三处**已知的真冲突**：
  · e8m0 的取整：ONNX 默认 `round_mode="up"`（ties away from zero），
    而 torch 的 C++ 注释写 ties-to-even 却实现成「恒向上」——**结果一致、理由不同**；
  · fp8 的溢出：ONNX 把它做成显式属性 `saturate`，而 torch 的行为**跨版本变了**
    （≤ 2.11.0 与 2.13.0 不同）；
  · e8m0 的负值：ONNX 明写 undefined，torch 有确定行为（取绝对值）。
  **每条都标了 `（规范）` / `（实现）` / `（规范 × 实现 冲突）`。**

★ 本文件**不放任何算法**，也**不得 import torch**（`verify.py` 在默认 python3 下 import 它）。
  凡被断言检验的算法（位域反解、e8m0 取整、MX 块量化、对称/非对称 RTN）
  **一律在 `verify.py` 里另写**——理由见 CONTRIBUTING §2 的「参考实现必须独立」。

UNSOURCED（查不到一手来源，**正文不使用**）：
  · 「OCP MX 规范 §5.4.1 提到 block scale」——**规范 PDF 本机取不到**（opencompute.org
    对 urllib 返回 HTTP 403）。§5.4.1 这个章节号来自计划文件的二手指认，
    **本文件不落成常量**；正文改为引「ONNX Cast 文档里 e8m0 段的 MX 链接」与
    「MX 论文 arXiv:2310.10537v2 Table 1」这两条可复核的来源。
  · 「tf32 的输入转换舍入模式」——两处来源冲突且本机无 tf32 存储 dtype，
    **记为 UNRESOLVED 而不是常量**（见 TERMS 末尾）。
  · 「`_weight_int8pack_mm` 的累加器类型」——属 `quantized-kernels` 主题，本文件不涉及。
"""

# ---------------------------------------------------------------------------
# 版本锚点（本机）
# ---------------------------------------------------------------------------

# 本机装了 torch 的解释器，五个（探测顺序见 code/Makefile）
# 来源：本机实测 `import torch; torch.__version__`（2026-09-18，逐条打印）
#   /usr/bin/python3                              → torch 2.8.0  / Python 3.9.6
#   ~/miniconda3/bin/python                       → torch 2.11.0 / Python 3.12.9
#   ~/miniconda3/envs/deep-prac/bin/python        → torch 2.6.0  / Python 3.12.11
#   ~/miniconda3/envs/dp-test/bin/python          → torch 2.7.1  / Python 3.13.7
#   ~/PyCharmMiscProject/.venv/bin/python         → torch 2.13.0 / Python 3.13.5
MEASURED_VERSIONS = ("2.6.0", "2.7.1", "2.8.0", "2.11.0", "2.13.0")

# ★ 本机确实没有解释器的三个 minor（不得声称覆盖）
UNCOVERED_MINORS = "2.9.x / 2.10.x / 2.12.x"

# 测量条件（文档 §5 的表头，全域统一口径）
MEASURE_HOST = "Apple M1 Pro（10 核）/ macOS / **CPU-only，无 GPU 数字**"
TORCH_MIN_FOR_E8M0 = (2, 8)

# `torch.float8_e8m0fnu` 与 `torch.float4_e2m1fn_x2` 在 2.6.0 / 2.7.1 上不存在。
# 来源：本机实测 `hasattr(torch, "float8_e8m0fnu")`，五个解释器逐个查（2026-09-18）
DTYPES_ABSENT_BEFORE_2_8 = ("float8_e8m0fnu", "float4_e2m1fn_x2")

# ---------------------------------------------------------------------------
# 【规范】ONNX Cast
# ---------------------------------------------------------------------------

# 来源：（规范）ONNX Cast 当前版本页 https://onnx.ai/onnx/operators/onnx__Cast.html
#   本机 urllib 直取整页 HTML 后逐字读（2026-09-18）。该页正文标题为 **Cast-28**、
#   页内 `since_version: 28`；页面列出的 Cast 历史版本为 1 / 6 / 9 / 13 / 19 / 21 / 23 / 24 / 25 / 28。
ONNX_CAST_CURRENT_VERSION = 28

# 来源：（规范）同上页 "Attributes" 段逐字。
# ⚠ `round_mode` 的默认值是 **`up`**，其定义是 **"round to nearest value away from zero"**，
#   **不是 RNE**；`nearest` 这个取值的定义才是 "round to nearest value and ties round up"。
ONNX_E8M0_ROUND_MODE_DEFAULT = "up"
ONNX_E8M0_ROUND_MODE_MEANINGS = {
    "up": "round to nearest value away from zero",
    "down": "round to nearest value towards zero",
    "nearest": "round to nearest value and ties round up",
}

# 来源：（规范）同上页 attributes：「saturate - INT (default is 1) ... It only applies for
#   float 8 conversion (float8e4m3fn, float8e4m3fnuz, float8e5m2, float8e5m2fnuz, float8e8m0)」
ONNX_SATURATE_DEFAULT = 1

# 来源：（规范）ONNX 官方 Changelog（main 分支 docs/Changelog.md，本机 urllib 直取 1 497 926 B
#   后按 `### <a name="Cast-N">` 分节逐节 grep，2026-09-18）：
#     · `saturate` 首次出现在 **Cast-19** 一节（Cast-1 / 6 / 9 / 13 四节里 0 命中）；
#     · `round_mode` 与 `FLOAT8E8M0` 首次出现在 **Cast-24** 一节
#       （Cast-19 / 21 / 23 三节里 `round_mode` 0 命中）。
#   ⚠ **这订正了计划文件的一处说法**：计划写「round_mode 自 21 起」，实测是自 **24** 起。
ONNX_SATURATE_SINCE_OPSET = 19
ONNX_ROUND_MODE_SINCE_OPSET = 24     # 计划文件写 21，此处按逐节核实结果订正

# 来源：（规范）同上页 "Float 8 types (E4M3FN, E4M3FNUZ, E5M2, E5M2FNUZ) were introduced ..."
#   之后的两张表**逐字**抄录。`[x]` 表示「舍入到目标尾数宽度之后的值」。
#   默认（`saturate=1`）表：越界与特殊值一律饱和到 ±FLT_MAX，其余走 RNE。
#   `saturate=0` 表：E4M3FN/E4M3FNUZ/E5M2FNUZ 的越界与 ±Inf 全部 → NaN，
#                     只有 E5M2 是 ±Inf（它有 Inf 码字），其余仍 RNE。
#   ⚠ 注意 `FLT_MAX` 是**目标格式**的最大有限数（源码里的记法是 FLT_MAX 复用），
#     不是 fp32 的 3.4e38；对 e4m3fn 即 448、对 e5m2 即 57344。
ONNX_F8_SATURATE_TRUE_OOR = "FLT_MAX"          # 越界 → 饱和到目标格式最大值
ONNX_F8_SATURATE_FALSE_OOR = {                 # saturate=0 时越界/Inf 的去向
    "E4M3FN": "NaN", "E4M3FNUZ": "NaN", "E5M2": "Inf", "E5M2FNUZ": "NaN",
}
ONNX_F8_DEFAULT_ROUND_MODE_ALL_FOUR = "RNE"    # 原文表末行 "else | RNE | RNE | RNE | RNE"

# 来源：（规范 × 实现 冲突）同上页 FLOAT8E8M0 段**逐字**：
#   "The current CUDA behavior is to round up and saturate. **Casting negative values to
#    FLOAT8E8M0 gives undefined behavior.**"
#   同段的特殊值表：`-0` → Unspecified/Unspecified；`x < 0` → Unspecified/Unspecified；
#   `0` → `0`（saturate+up 列）/ `NaN`（non-saturate+nearest 列）；
#   `Inf` → `E8M0_MAX`/`NaN`；`x > E8M0_MAX` → `E8M0_MAX`/`NaN`；`x < E8M0_MIN` → `E8M0_MIN`/`NaN`。
#   **而 torch 对负值有确定行为（取绝对值）——见 IMPL_E8M0_*。这是本域第一处真冲突。**
ONNX_E8M0_NEGATIVE_INPUT = "undefined behavior（原文逐字：Casting negative values to FLOAT8E8M0 gives undefined behavior）"
ONNX_E8M0_SPECIAL_TABLE = {
    "0": ("0", "NaN"),
    "-0": ("Unspecified", "Unspecified"),
    "NaN": ("NaN", "NaN"),
    "Inf": ("E8M0_MAX", "NaN"),
    "x > E8M0_MAX": ("E8M0_MAX", "NaN"),
    "x < E8M0_MIN": ("E8M0_MIN", "NaN"),
    "x < 0": ("Unspecified", "Unspecified"),
}
ONNX_E8M0_SPECIAL_TABLE_COLUMNS = ("saturate + up", "non-saturate + nearest")

# ---------------------------------------------------------------------------
# 【规范】OCP MX
# ---------------------------------------------------------------------------

# 来源：（规范）正式名称与版本号由 MX 论文 arXiv:2310.10537v2 正文指名；
#   规范的 PDF 本机取不到（HTTP 403），故**只落名称与版本，不落章节号**。
OCP_MX_SPEC_NAME = "OCP Microscaling Formats (MX) Specification"
OCP_MX_SPEC_VERSION = "Version 1.0"
OCP_MX_SPEC_DATE = "2023-09-07"

# 来源：（规范）三方独立印证，本机逐条读到：
#   ① MX 论文 arXiv:2310.10537v2 Table 1（四种格式的块大小与 scale 格式）；
#   ② vLLM 源码常量 `OCP_MX_BLOCK_SIZE = 32`（vllm/model_executor/layers/quantization/
#      compressed_tensors/... 与其 MX 相关模块）；
#   ③ ONNX Cast 文档 FLOAT8E8M0 段的 MX 链接指向同一份规范。
OCP_MX_BLOCK_SIZE = 32

# 来源：（规范）同上。四种 MX 格式的元素格式与 scale 格式。
OCP_MX_ELEMENT_FORMATS = {
    "MXFP8": "FP8 (E4M3 或 E5M2)",
    "MXFP6": "FP6",
    "MXFP4": "FP4 (E2M1)",
    "MXINT8": "INT8",
}
OCP_MX_SCALE_FORMAT = "E8M0"

# 来源：（**论文，不是规范**）arXiv:2310.10537v2 Algorithm 1 第 1 行
#   `shared_exp <- floor(log2(max|V_i|)) - emax`，论文自述遵循 OCP 规范 §6.3。
# ⚠ **这条是 P8 的镜像，必须标清**：MX **规范的 out-of-scope 清单里逐字包含
#   「块 scale 的具体计算算法」**——也就是说 **Floor 不是规范规定的**。
#   **把论文当规范，与把规范当实现，是同一个错误的两个方向。**
#   两次取整也必须分开：**块 scale 用 Floor（论文 Alg.1），元素取整才用 RTNE。**
MX_SCALE_ROUNDING_SOURCE = "论文 Algorithm 1（不是规范）"
MX_SCALE_ROUNDING = "floor"
MX_ELEMENT_ROUNDING = "RTNE（round to nearest, ties to even）"

# 来源：（规范）MX 论文 arXiv:2310.10537v2 正文逐字："The layout of an MX block is not prescribed"
#   ——**规范不规定 scale 在块内的存放位置**；
#   而 Blackwell 的 cuDNN 要求 "128×4 tiled layout"（NVIDIA cuDNN 博客）——
#   **规范不规定 / 某实现强制，分开写。**
MX_BLOCK_LAYOUT_PRESCRIBED_BY_SPEC = False
MX_BLOCK_LAYOUT_IMPL_FORCED = "cuDNN on Blackwell 要求 128×4 tiled layout（实现强制，非规范）"

# ---------------------------------------------------------------------------
# 【规范】NVFP4
# ---------------------------------------------------------------------------

# 来源：（规范）NVFP4 的两层 scale 定义，引自 NVIDIA 的 NVFP4 材料（本机未逐字读到原文，
#   按**引用值**落表，正文不得写成「规范逐字」）。
#   ⚠ 与 MXFP4 的三点区别逐条落进常量名，使误用可见。
NVFP4_ELEMENT_FORMAT = "E2M1"
NVFP4_ELEMENT_ABS_MAX = 6.0
NVFP4_BLOCK_SIZE = 16
NVFP4_SCALE_GLOBAL = "FP32，整张量一个：s_global = global_amax / (448 * 6)"
NVFP4_SCALE_BLOCK = "FP8 E4M3，每 16 个元素共享：s_block = (block_amax / 6) / s_global"
NVFP4_VS_MXFP4_DIFFS = (
    "块大小 16（MXFP4 是 32）",
    "block scale 是 E4M3（带尾数，不是 2 的幂）（MXFP4 是 E8M0）",
    "多一层 per-tensor FP32 global scale（MXFP4 没有）",
)

# ---------------------------------------------------------------------------
# 【位域】四个 fp8 变体（规格值：由 IEEE-754 式的位域定义决定，与实现无关）
# ---------------------------------------------------------------------------

# 来源：（规范）由位域定义**解析推出**的规格值：S 符号位 / E 指数位 / M 尾数位，
#   bias = 2^(E-1) - 1（fnuz 变体是 2^(E-1)），
#   max = (2 - 2^-M) * 2^(2^E - 1 - bias)、最小次正规 = 2^(1 - bias - M)。
#   **本文件的这些数已由 `verify.py` 的 `ref_*` 从位域独立反解复核**（A 类）。
#   ⚠ 与 torch 的 `torch.finfo` 输出是**两条不同的通道**：
#     finfo 报的 `tiny`/`smallest_normal` 是**最小正规数**（= 2^(1-bias)），
#     **不是最小次正规数**。两个数差 2^M 倍，混用会差 3 个 binade。
#   四位域（S/E/M）、bias、max、最小次正规、最小正规、eps：
FP8_FORMATS = {
    # dtype 名： (S, E, M, bias, max, 最小次正规, 最小正规, eps)
    "float8_e4m3fn":   (1, 4, 3, 7,  448.0,   2.0 ** -9,  2.0 ** -6,  2.0 ** -3),
    "float8_e5m2":     (1, 5, 2, 15, 57344.0, 2.0 ** -16, 2.0 ** -14, 2.0 ** -2),
    "float8_e4m3fnuz": (1, 4, 3, 8,  240.0,   2.0 ** -10, 2.0 ** -7,  2.0 ** -3),
    # ⚠ 下面这一行的 eps 是 **2^-2 = 0.25**，不是 torch.finfo 报的 0.125。
    #   见 IMPL_E5M2FNUZ_FINFO_EPS_IS_WRONG。
    "float8_e5m2fnuz": (1, 5, 2, 16, 57344.0, 2.0 ** -17, 2.0 ** -15, 2.0 ** -2),
}

# ★ 本主题实测发现的 **torch 实现缺陷**：`torch.finfo(torch.float8_e5m2fnuz).eps`
#   报 **0.125**，而该格式的真实 eps（1.0 处相邻可表示数之间的间距）是 **0.25**。
#   来源（**两条独立通道，都不是猜**）：
#     ① **本机实测**（torch 2.8.0 与 2.13.0，2026-09-18）：把 256 个码字全扫一遍，
#        找出 `1.0` 的**下一个可表示数**是 `1.25` → 实测 eps = 0.25；
#        同一次扫描里 `float8_e5m2` 的下一可表示数是 `1.25` → eps = 0.25（与 finfo 一致）；
#        `float8_e4m3fn` / `float8_e4m3fnuz` 的下一可表示数是 `1.125` → eps = 0.125（一致）。
#        **四个格式里只有 e5m2fnuz 对不上。**
#     ② **源码**（本机 torch 2.11.0 / 2.13.0 两处逐行打印，行号一致）：
#        `torch/include/torch/headeronly/util/Float8_e5m2.h:441-443` 的 `epsilon()`
#        与 `.../Float8_e5m2fnuz.h:429-431` 的 `epsilon()` **返回同一个字面码字 `0x34`**；
#        而两个格式的 bias 分别是 **15 与 16**，同一个码字因此读出
#        `2^-2 = 0.25`（e5m2）与 `2^-3 = 0.125`（e5m2fnuz）。
#        —— **常数是照抄的，没有按大 1 的 bias 调整。**
#   判据（必须与这个数字一起写出）：**「eps」按 2^-M 定义、M 是该格式的尾数位数**；
#     对 e5m2fnuz 的 M=2 → **0.25**。这一条在本文件里也只作**规格常量**使用，
#     它的正确性由 `verify.py` 从位域独立反解复核（A1 的 eps 一行）。
# ⚠ **这条是本主题独有的坑**：torch 的 `finfo` 在这里与位域自相矛盾，
#   而「照抄 finfo」正是最容易的写法——**照抄就会把 0.125 当成实测值写进文档**。
#   （计划文件 §0.7.A 的那张表在这一格上就是抄了 finfo，值是 0.125。）
IMPL_E5M2FNUZ_FINFO_EPS_IS_WRONG = {
    "finfo_reports": 0.125,
    "true_spacing_at_1.0": 0.25,
    "next_representable_after_1.0": 1.25,
    "cause": "Float8_e5m2fnuz.h 的 epsilon() 照抄 Float8_e5m2.h 的码字 0x34；两格式 bias 差 1",
    "src_lines": {"Float8_e5m2.h epsilon()": 441, "Float8_e5m2fnuz.h epsilon()": 429},
    "observed_versions": ("2.8.0", "2.13.0"),
}

# 来源：（规范）NaN / Inf 的码字集合，由位域定义决定。
#   · e4m3fn 与两个 fnuz 变体**没有 Inf**（全 1 指数域被征用/被 NaN 占掉）；
#   · e4m3fn 的 NaN 是 **2 个码字** `0x7F` / `0xFF`（全 1 指数 + 全 1 尾数）；
#   · e5m2 有 Inf（`S.11111.00`），NaN 是 `S.11111.{01,10,11}` 共 6 个码字；
#   · **fnuz 的 NaN 只有 `0x80` 一个码字** —— 这是「`-0` 与 NaN 抢码字」的直接后果，
#     所以 **fnuz 没有负零**。
FP8_NAN_CODES = {
    "float8_e4m3fn":   (0x7F, 0xFF),
    "float8_e5m2":     (0x7D, 0x7E, 0x7F, 0xFD, 0xFE, 0xFF),
    "float8_e4m3fnuz": (0x80,),
    "float8_e5m2fnuz": (0x80,),
}
FP8_FORMATS_WITH_INF = ("float8_e5m2",)
FP8_FORMATS_WITHOUT_NEG_ZERO = ("float8_e4m3fnuz", "float8_e5m2fnuz")

# 来源：（规范）fnuz 两个变体的 bias **恰比对应的非 fnuz 大 1**，
#   换来一个额外 binade。由上面的 bias 列直接读出（7→8、15→16）。
# 来源：（引用，来源可靠）fnuz 的命名由来：
#   LLVM commit `6109e70`（2023-01-10）标题逐字
#   "Add NaN-in-negative-zero formats by **AMD and GraphCore**"，正文提到 Qualcomm；
#   格式原始出处 **arXiv:2206.02915（Noune, Jones, Justus, Masters, Luschi，Graphcore）**。
# ⚠ **不要把这篇的作者写成 Kuzmin et al.**——Kuzmin 等是**另一篇** arXiv:2208.09225
#   （"FP8 Quantization: The Power of the Exponent"）。两篇都由 Graphcore 支持，
#   **编号与作者极易串**。
# ⚠ 也**不要**写「AMD 用 fnuz 是因为 MI300 选了它」——同样错。
FNUZ_ORIGIN_LLVM_COMMIT = "6109e70 (2023-01-10)"
FNUZ_ORIGIN_PAPER = "arXiv:2206.02915"
FNUZ_ORIGIN_AUTHORS = "Noune, Jones, Justus, Masters, Luschi (Graphcore)"
FNUZ_NOT_AMD_INVENTED = "LLVM commit 标题即 by AMD and GraphCore（含 Qualcomm）"

# 来源：（引用）谁用哪个：vLLM 按平台选，`is_fp8_fnuz()` 为真时用 `float8_e4m3fnuz`
#   （`vllm/platforms/rocm.py`）。
VLLM_FNUZ_SELECTOR = "vllm/platforms/rocm.py: is_fp8_fnuz()"

# ---------------------------------------------------------------------------
# 【实现】torch 的 fp8 溢出行为（跨版本断点）—— 本主题最重要的版本断点
# ---------------------------------------------------------------------------

# 来源：（实现，**本机实测**）五个解释器逐个跑 `torch.tensor([v]).to(torch.float8_e4m3fn)`，
#   2026-09-18。测量条件是「该解释器的 torch 版本 + Python 版本」，无 GPU。
#   **四个版本完全一致，只有 2.13.0 变**：
#     2.6.0 / 2.7.1 / 2.8.0 / 2.11.0 ： 460 → 448（近处饱和）  480 → NaN   1e30 → NaN
#     2.13.0                          ： 460 → 448（近处饱和）  480 → 448   1e30 → 448
#   ⚠ **分界在 2.11.0 与 2.13.0 之间**；2.9 / 2.10 / 2.12 本机无解释器，不声称覆盖。
#   ⚠ 「四个版本一致」这句话本身**需要四个版本都跑过才敢写**。
IMPL_E4M3FN_OVERFLOW_BY_VERSION = {
    "2.6.0":  {"460": 448.0, "480": "NaN", "1e30": "NaN"},
    "2.7.1":  {"460": 448.0, "480": "NaN", "1e30": "NaN"},
    "2.8.0":  {"460": 448.0, "480": "NaN", "1e30": "NaN"},
    "2.11.0": {"460": 448.0, "480": "NaN", "1e30": "NaN"},
    "2.13.0": {"460": 448.0, "480": 448.0, "1e30": 448.0},
}
IMPL_E4M3FN_OVERFLOW_BREAK_BETWEEN = ("2.11.0", "2.13.0")
# 同一批实测里，**e4m3fnuz 没有被这次变更覆盖**：五个版本一律「非饱和 + 最近舍入」。
#   241 → 240（RNE：中点 244，241 在下方）；248 → NaN；300 → NaN；460/480/1e30 → NaN。
#   2.13.0 与 2.8.0 在这一列上**逐格相同**。
IMPL_E4M3FNUZ_OVERFLOW_NOT_COVERED = "五个版本一律 非饱和 + 最近舍入（248/300/1e30 → NaN）"
IMPL_E5M2_OVERFLOW_ALL_VERSIONS = "1e30 → inf（e5m2 有 Inf 码字，五个版本一致）"
IMPL_E4M3FN_BITPATTERN = {
    "0x7F": "NaN", "0xFF": "NaN", "0x80": "-0.0", "0x00": "0.0",
}
IMPL_E4M3FNUZ_BITPATTERN = {
    "0x7F": 240.0, "0xFF": -240.0, "0x80": "NaN", "0x00": "0.0",
}

# 来源：（引用，**本机未逐字核实其因果**）这次变更的动因线索：
#   · pytorch issue #154028（2025-05-21 创建，标题 "Add clamped FP8 (E4M3) cast for
#     overflow-safe inference"），正文逐字：「Most of the hardware saturates (To the
#     maximum finite values) overflow values instead of making them NaN.」
#   · pytorch PR #178817（2026-03-30 创建，标题逐字 "[CUDA][FP8][CPU][TEST] Properly
#     saturate E4M3 on finite-overflow on CPU/C++ conversion code"），
#     **截至 2026-09-18 通过 GitHub API 读到 state=closed、merged=False**。
#   ⚠ **因此「是哪一个 PR 把 2.13.0 的行为改掉的」本仓库未确证，正文不得写因果。**
#   ⚠ 计划文件里「维护者说法是 eager 与 inductor/triton 静默不一致」**未逐字核实**，
#     本文件不落成常量。
IMPL_FP8_OVERFLOW_MOTIVATION_ISSUE = "pytorch issue #154028 (2025-05-21)"
IMPL_FP8_OVERFLOW_MOTIVATION_PR = "pytorch PR #178817 (2026-03-30, closed 未合并)"
IMPL_FP8_OVERFLOW_CAUSE_UNVERIFIED = True

# ---------------------------------------------------------------------------
# 【实现】e8m0（torch）
# ---------------------------------------------------------------------------

# 来源：（实现，**本机实测**）`torch.float8_e8m0fnu` 的 256 个码字**全扫**
#   （把 uint8 0..255 用 `.view()` 读成 e8m0，再转 float64），2026-09-18。
#   在 torch 2.8.0 / 2.11.0 / 2.13.0 上逐格相同（2.6.0 / 2.7.1 无此 dtype）。
IMPL_E8M0_NAN_CODE_COUNT = 1                  # 只有 0xFF
IMPL_E8M0_NAN_CODES = (0xFF,)
IMPL_E8M0_ZERO_CODE_COUNT = 0                 # **没有零**
IMPL_E8M0_NEGATIVE_CODE_COUNT = 0             # **没有负号**
IMPL_E8M0_MIN_FINITE = 2.0 ** -127            # 0x00 → 2^-127
IMPL_E8M0_MAX_FINITE = 2.0 ** 127             # 0xFE → 2^127
IMPL_E8M0_FINITE_COUNT = 255                  # 0x00..0xFE 全部有限且严格递增
IMPL_E8M0_ALL_VALUES_ARE_POW2 = True
IMPL_E8M0_DTYPE_ABSENT_IN = ("2.6.0", "2.7.1")

# 来源：（实现，**本机实测**）三组取整行为，2.8.0 / 2.11.0 / 2.13.0 一致：
#   · `-1.0 → 1.0`、`-1.5 → 2.0`   ← **符号位被直接丢弃**（ONNX 说这里 undefined）
#   · `0.0 → 2^-127`、`-0.0 → 2^-127` ← **零无法表示**
#   · 中点 `(2^20 + 2^21)/2 = 1572864 → 2097152 = 2^21`（**向上**）；
#     同一下方一个 ulp 的 `1572863 → 1048576 = 2^20`（**向下**，证明确实是按最近取整）
IMPL_E8M0_ABS_OF_NEGATIVE = {"-1.0": 1.0, "-1.5": 2.0}
IMPL_E8M0_ZERO_MAPS_TO = 2.0 ** -127
IMPL_E8M0_MIDPOINT_UP = {"1572864.0": 2097152.0, "1572863.0": 1048576.0}

# ★★ 本主题实测发现的**第二处** e8m0 规范/实现分歧（计划文件里没有这一条）：
#   **e8m0 的两端不对称——低频端饱和、高频端不饱和。**
#   来源：（实现，**本机实测**）torch 2.8.0 / 2.11.0 / 2.13.0 三个解释器上
#   同一组输入逐点打印，2026-09-18：
#     · 低频端：0.0 / −0.0 / 2^-130 / 2^-200 **一律 → 2^-127**（看起来像饱和）；
#     · 高频端：阈值 **1.5·2^127 = 2.5521177519070385e+38**，
#         < 阈值 → **2^127**；  ≥ 阈值 → **NaN**（**不是**饱和到 2^127）。
#   判据：e8m0 的尾数位为 0，所以 2^127 处的 ulp **就是 2^127**，
#     阈值 = 上限 + ulp/2 = 1.5·2^127 —— 与 01 里那四个 fp8 变体**同一条规则**。
#   机理（读源码得到的解释，与实测一致）：那份实现**没有任何显式饱和逻辑**，
#     它只是「按最近取整、平局向上」然后让码字落到该落的地方——
#     低频端落到的码字是 `0x00`（合法值 2^-127），高频端落到的码字是 **`0xFF`（NaN）**。
#     **两端不同的表现来自码字空间的边界，不是来自两条不同的规则。**
#   ⚠ **与规范的分歧点（必须写清是哪一侧）**：ONNX Cast 页把这件事列成两列，
#     正文写「**The current CUDA behavior is to round up and saturate**」——
#     **它描述的是 CUDA**；而**本机是 CPU-only，测到的是 CPU 路径**。
#     对 `x > E8M0_MAX`，ONNX 的 `saturate + up` 列写 `E8M0_MAX`、
#     `non-saturate + nearest` 列写 `NaN`——**torch 的 CPU 路径落在后者那一列**。
#   → **不得写「ONNX 错了」**，也不得写「torch 一律饱和」。
#     **正确的写法是：规范说的是 CUDA 的行为，本机测的是 CPU 的行为，两者不同。**
IMPL_E8M0_HIGH_END_NOT_SATURATING = {
    "threshold": 1.5 * 2.0 ** 127,
    "below": 2.0 ** 127,
    "at_or_above": "NaN",
    "low_end": "一律饱和到 2^-127",
    "mechanism": "无显式饱和逻辑；低频端落 0x00（合法），高频端落 0xFF（NaN）",
    "observed_versions": ("2.8.0", "2.11.0", "2.13.0"),
    "onnx_prose_scope": "ONNX 正文那句描述的是 CUDA；本机 CPU-only，测的是 CPU 路径",
}

# 来源：（**实现 × 实现 的 P8 实例**）torch 的 e8m0 取整源码。
#   文件（torch 2.13.0 与 2.11.0 两处都逐行打印过，行号一致）：
#     torch/include/torch/headeronly/util/Float8_e8m0fnu.h（226 行）
#     （`torch/include/c10/util/Float8_e8m0fnu.h` 只有 6 行，是 `#include` 转发壳）
#   第 74–75 行注释逐字：「next, we use guard, round, sticky bits and the LSB to
#     implement round to nearest, **with ties to even**」
#   第 85 行逐字：`uint8_t lsb = exponent > 0;`
#   第 83–84 行注释逐字：「in casting to e8m0, LSB is the implied mantissa bit.
#     It equals to 0 if the original float32 is denormal, and to 1 if the original
#     float32 is normal.」
#   → **`lsb` 取的是「源 float32 的隐含尾数位」，不是「e8m0 码字的 LSB」**；
#     对一切**正规** float32 输入 lsb 恒为 1，于是平局分支恒走「round up」——
#     **注释说的 ties-to-even 实现成了恒向上**。
#   → 实测旁证（本机，10 个中点逐点打印）：中点在 `2^a` 与 `2^(a+1)` 之间时，
#     a = -4..+5 **全部向上**；其中有 5 个点 ties-to-even 本应选**下**。
#   **→ 结论：不是 bug 意义上的「错」，而是「两条路殊途同归」——**
#     其结果与 ONNX 默认的 `round_mode="up"`（ties away from zero）**一致**，
#     理由却与注释自称的 ties-to-even **不同**。**结果一致、理由不同，就是 P8。**
IMPL_E8M0_ROUNDING_SRC_FILE = "torch/include/torch/headeronly/util/Float8_e8m0fnu.h"
IMPL_E8M0_ROUNDING_SRC_LINES = {"注释 ties to even": 74, "注释 ties to even 续": 75,
                               "隐含尾数位注释": 83, "lsb 赋值": 85}
IMPL_E8M0_ROUNDING_COMMENT = "round to nearest, with ties to even"
IMPL_E8M0_ROUNDING_LSB_LINE = "uint8_t lsb = exponent > 0;"
IMPL_E8M0_ROUNDING_TIES_ALWAYS_UP_FOR_NORMAL = True
IMPL_E8M0_TIE_PROBE_A_RANGE = (-4, 5)          # 实测覆盖的 a 区间（闭区间）
IMPL_E8M0_TIE_PROBE_ALL_UP = True

# ---------------------------------------------------------------------------
# 【实现 × 实现】MX 的「不 clamp 会怎样」是版本相关的（本主题实测发现）
# ---------------------------------------------------------------------------

# 来源：（实现，**本机实测**，2026-09-18）同一段 MX 块量化代码，
#   在 torch 2.8.0 与 2.13.0 上对**同一批数据**给出不同的失败形态：
#     元素格式 e4m3（emax=8）：2.8.0 不 clamp → 781 个 NaN；2.13.0 不 clamp → **0 个 NaN**（饱和）
#     元素格式 e5m2（emax=15）：两个版本都不 clamp → 490 个 Inf（e5m2 有 Inf 码字）
#   同一批数据里越过元素上限的元素数 = 1077 / 131072。
#   ⚠ **这是 01 的溢出断点在 02 上的直接后果**，计划文件里没有这一条。
#   → 「MX 必须 clamp，否则会出 NaN」这句话**在 torch ≥ 2.13 的 e4m3 路径上不再成立**
#     （原因不在 MX，在 01 那条 fp8 变更）。
#   → 但「规范要求 clamp」的**数值必要性**仍由 e5m2 那一列独立支撑：
#     换成有 Inf 的元素格式，不 clamp 就把 Inf 混进结果（不是 NaN，但同样是坏值）。
MX_UNCLAMPED_FAILURE_BY_VERSION = {
    "2.8.0":  {"e4m3": "781 个 NaN", "e5m2": "490 个 Inf"},
    "2.13.0": {"e4m3": "0 个 NaN（饱和到 448）", "e5m2": "490 个 Inf"},
}
MX_OVERFLOW_ELEMENTS_IN_PROBE = (1077, 131072)   # (越界元素数, 元素总数)

# ---------------------------------------------------------------------------
# 【推导】MX 的 Floor 取整致溢率（解析估算 vs 蒙特卡洛，**两个数不同判据**）
# ---------------------------------------------------------------------------

# 判据（**必须与数字一起写出，否则不得与另一个数字并列** —— 错误模式 P9）：
#   设块内最大幅值 amax，规约后尾数 m ≜ amax / 2^floor(log2(amax)) ∈ [1, 2)。
#   块 scale = 2^(floor(log2 amax) - emax)，所以规约后最大值 = m · 2^emax，
#   超过元素上限当且仅当 m > elem_max / 2^emax。
#   · e4m3：elem_max = 448 = 1.75 · 2^8，emax = 8 → 阈值 **m > 1.75**（= 448/256）；
#     均匀假设下 P = 2 − 1.75 = **25%**（解析）；N(0,1) 32 元素块 MC = **16.9%**。
#   · e2m1：elem_max = 6 = 1.5 · 2^2，emax = 2 → 阈值 **m > 1.5**（= 6/4）；
#     均匀假设下 P = 2 − 1.5 = **50%**（解析）；N(0,1) 32 元素块 MC = **29.6%**。
#   ⚠ **「25%」与「~29%」不是同一件事的估算与实测**：前者是 e4m3 的解析估算、
#     后者是 e2m1 的蒙特卡洛。**两个阈值不同、两个元素格式不同**（P9 实例 1）。
MX_OVERFLOW_THRESHOLD = {"e4m3": 448.0 / 256.0, "e2m1": 6.0 / 4.0}
MX_OVERFLOW_UNIFORM_ESTIMATE = {"e4m3": 0.25, "e2m1": 0.50}          # 解析（均匀 m）
MX_OVERFLOW_MC_GAUSSIAN_BLOCK32 = {"e4m3": 0.169, "e2m1": 0.296}     # MC，N(0,1)，本机

# ---------------------------------------------------------------------------
# 【实现，本机实测】bf16 / fp16 / tf32 的量化视角
# ---------------------------------------------------------------------------

# 来源：（实现，**本机实测**）`torch.finfo` 在 2.6.0 / 2.7.1 / 2.8.0 / 2.11.0 / 2.13.0
#   五个解释器上逐格相同（2026-09-18）：
BF16_MANTISSA_BITS = 7          # ★ 7 位，不是 8
BF16_EPS = 2.0 ** -7            # = 0.0078125
FP16_MANTISSA_BITS = 10
TF32_MANTISSA_BITS = 10
FP16_MAX = 65504.0
BF16_MAX = 3.3895313892515355e38

# 来源：（实现，**本机实测**）fp32 → bf16/fp16 的转换舍入是 **RNE**。
#   ★ 判据（2026-09-20 订正）：`1 + 3/256` = 1.01171875 **恰在** bf16（7 位尾数）的
#     两个相邻可表示数 `1+1/128`（= 1.0078125，尾数 0b0000001 **奇**）与
#     `1+1/64`（= 1.015625，尾数 0b0000010 **偶**）的**中点**上。
#     取偶 ⇒ **1.015625**（若为「向上」也会得 1.015625，故这一格**分不出**
#     RNE 与 round-half-up；要区分必须用「下界为偶」的中点，见 `RNE_TIE_CASES`）。
#
#   ⚠ **本组常量在 2026-09-20 之前是「死常量」**——全库无任何程序引用它，
#     因此下面这处错值在建成验收里不可见：
#       · 原写 `expect: 1.0` ⇒ **错**（正确是 1.015625）
#       · 原注释「1 与 1+1/128 的中点」⇒ **错**（那是 `1 + 1/256`，不是探针值）
#     **订正记录见 `docs/superpowers/plans/2026-09-20-kernel-lifecycle-domain.md` §E。**
#     断言已在 `verify.py` 的 `A1d` 小节补上（不再是无检索的抄件）。
RNE_MIDPOINT_PROBE = {"1 + 3/256": {"to": "bfloat16", "expect": 1.015625}}

# ★ 补：真正能**区分** RNE 与 round-half-up 的中点——两候选的**下界为偶**。
#   `1 + 1/256` = 1.00390625 是 `1.0`（尾数 0b0000000 **偶**）与
#   `1 + 1/128` = 1.0078125（尾数 **奇**）的中点 ⇒ RNE 取偶得 **1.0**，
#   而 round-half-up 会得 1.0078125。**这一格才分得出两种规则。**
RNE_TIE_CASES = {
    "1 + 1/256": {"to": "bfloat16", "expect": 1.0,          "half_up_would_give": 1.0078125},
    "1 + 3/256": {"to": "bfloat16", "expect": 1.015625,     "half_up_would_give": 1.015625},
}

# 来源：（实现，**本机实测**）SQNR = 10·log10(Σx² / Σ(x−x̂)²)，2×10⁶ 样本，
#   N(0,1) / U(−1,1) / Laplace(0,1) / StudentT(3)（后者截断到 ±1e3）四种分布。
#   在 torch 2.6.0 / 2.11.0 / 2.13.0 上**逐格相同**（2026-09-18）。
#   tf32 一列是**位掩码仿真**（RNE 到 10 位尾数），**不是硬件实测**。
SQNR_DB_BY_DIST = {
    # 分布            bf16     fp16     e4m3fn   tf32(仿真)
    "N(0,1)":       (55.59,  73.65,  31.52,   73.65),
    "U(-1,1)":      (56.62,  74.68,  32.53,   74.68),
    "Laplace(0,1)": (55.61,  73.67,  31.52,   73.67),
    "StudentT(3)":  (55.61,  73.69,  31.51,   73.69),
}
# 判据（**必须与数字一起写出**）：上面每一格都是
#   10·log10(Σx²/Σ(x−x̂)²)，2×10⁶ 个样本，`torch.manual_seed(20260918)`，
#   StudentT(3) 截断到 ±1e3 以避开 float64 溢出。
#   tf32 那一列是**位掩码仿真**（先按 fp32 取整、再舍到 10 位尾数），**不是硬件实测**。
SQNR_SEED = 20260918
SQNR_SAMPLE_COUNT = 2_000_000
SQNR_COLUMNS = ("bf16", "fp16", "e4m3fn", "tf32(位掩码仿真，不是硬件)")

# 来源：（引用）Kuzmin et al., arXiv:2208.09225 Table 1 印值：
#   "BFloat-16 55.6 dB / IEEE float-16 73.7 dB"。
#   **本机实测（torch 2.11.0，N(0,1)，2×10⁶ 样本）为 55.59 / 73.65 dB，
#     与印值吻合到 0.1 dB 以内** —— 一次跨来源的交叉验证。
#   ⚠ 该论文的 SQNR 定义（信号/误差比、样本集、分布）本仓库未逐字核实，
#     **这是「两个来源给出同一量级」的旁证，不是「定义相同」的证明**。
KUZMIN_TABLE1_BF16_DB = 55.6
KUZMIN_TABLE1_FP16_DB = 73.7
KUZMIN_PAPER = "arXiv:2208.09225 (Kuzmin et al., 'FP8 Quantization: The Power of the Exponent')"

# 来源：（推导 + 本机实测）「浮点 SQNR 与分布几乎无关」的**两个端点**：
#   跨四个分布，bf16 的极差 = 56.62 − 55.59 = **1.04 dB**；
#   e4m3fn 的极差 = 32.53 − 31.51 = **1.02 dB**。
#   ⚠ **不要写「< 1 dB」**——被它自己给的两个端点否证（55.59 → 56.62 就是 1.03+）。
#   ⚠ **两个端点是稳的**（控制者与审查者独立复现一致），**中间值随分布集变化**。
#   → 引用时**给端点 + 说明分布集**，**不要给一个笼统区间**。
SQNR_SPREAD_DB = {"bf16": 1.04, "fp16": 1.03, "e4m3fn": 1.02}
SQNR_SPREAD_DIST_SET = ("N(0,1)", "U(-1,1)", "Laplace(0,1)", "StudentT(3)")

# 来源：（推导）浮点 RNE 的相对误差 RMS 解析式，尾数均匀假设下：
#   SQNR ≈ 20·log10(√3 · √2 · 2^(M+1)) = 7.78 + 6.0206·(M+1) dB。
#   代入 M=7 → 55.94、M=10 → 74.02、M=3 → 31.86。
#   **本机实测 55.59 / 73.65 / 31.52 与它同量级**（差 0.3 dB 以内），
#   **给的是量级校验，不是同一个量的两种测法**（解析式假设尾数均匀，实测数据不是）。
FLOAT_SQNR_ANALYTIC_DB = {3: 31.86, 7: 55.95, 10: 74.01}

# 来源：（**未解决**）tf32 的输入转换舍入模式：两处来源冲突，且
#   「输入转换的舍入」与「累加过程的舍入」可能是两件事；本机无 tf32 存储 dtype
#   （torch 没有该 dtype）又是 CPU-only，**跑不了真 TF32**。
#   → **这一格写「未解决」，不给一个看起来合理的答案。**
TF32_INPUT_ROUNDING_MODE = "UNRESOLVED"
TF32_10BIT_MANTISSA_CONFIDENCE = "引用，多来源一致但均非 A100 白皮书原文，置信度中"

# ---------------------------------------------------------------------------
# 【规范】整数格式的量化公式（NVIDIA 两篇 arXiv 的式号）
# ---------------------------------------------------------------------------

# 来源：（规范）arXiv:2004.09602 Eq.1–5（非对称 / affine）：
#   s = (2^b − 1) / (α − β)
#   z = −round(β·s) − 2^(b−1)
#   xq = clip(round(s·x + z), −2^(b−1), 2^(b−1) − 1)
#   x̂  = (xq − z) / s
#   ⚠ 这里的 `z` 已经带上 −2^(b−1) 的偏移（把 [0, 2^b−1] 的码字映到有符号区间）。
#     本仓库的程序按**无符号码字** 0..2^b−1 实现，两者差一个常数偏移，
#     比较时**必须先写死用的是哪一套**（同一句话在两套约定下差一个偏移量）。
INT_ASYM_FORMULA_SOURCE = "arXiv:2004.09602 Eq.1–5"
INT_ASYM_FORMULA = "s=(2^b−1)/(α−β); z=−round(β·s)−2^(b−1); xq=clip(round(s·x+z),−2^(b−1),2^(b−1)−1); x̂=(xq−z)/s"
INT_ASYM_ZP_CONVENTION = "论文的 z 已含 −2^(b−1) 偏移；本仓库程序按无符号码字 0..2^b−1 实现"

# 来源：（规范）arXiv:2103.13630 Eq.2–5（对称）：
#   Z = 0；"restricted range" S = max|r| / (2^(n−1) − 1)（用 [−127, 127]）；
#   "full range"      S = 2·max|r| / (2^n − 1)。
#   ⚠ **torch 用的是 "full range" 那一支**（除数是 127.5 = 255/2），不是 127。
INT_SYM_FORMULA_SOURCE = "arXiv:2103.13630 Eq.2–5"
INT_SYM_RESTRICTED_RANGE = "S = max|r| / (2^(n−1) − 1)  即 /127（对 int8）"
INT_SYM_FULL_RANGE = "S = 2·max|r| / (2^n − 1)  即 /127.5（对 int8）"

# 来源：（实现，**本机实测**）torch 的对称量化除数是 **127.5**，不是 127。
#   判据：`per_tensor_symmetric` 的 `MinMaxObserver` 在 2.6.0 / 2.7.1 / 2.8.0 /
#   2.11.0 / 2.13.0 五个解释器上给出同一个数（2026-09-18）：
#     输入 maxabs = 17.961498  →  尺度 = 0.14087449
#     · 0.14087449 × 127.5 = 17.9614980  ← 与 maxabs 逐位一致
#     · 0.14087449 × 127   = 17.8962…    ← 对不上
#   torch 侧的口径是 `(quant_max − quant_min)/2`，int8 时 = 255/2 = 127.5。
#   **多数二手材料写「除以 127」，是错的**（那是 arXiv:2103.13630 的 restricted range）。
IMPL_SYM_DIVISOR_INT8 = 127.5
IMPL_SYM_DIVISOR_SOURCE = "(quant_max − quant_min)/2 = (127 − (−128))/2"
IMPL_OBSERVER_MAXABS_PROBE = {"maxabs": 17.961498, "scale": 0.14087449}

# 来源：（推导 + **本机实测**）零点被约束为整数带来什么误差。
#   设 δ ≜ −round(z)·s − α（把 z 释放为实数与取整之间的差）。
#   推导：a_eff = a + δ、b_eff = b + δ，**两端点同时平移 δ**，
#         范围既不缩小也不扩大，且 **|δ| ≤ s/2**。
#   本机实测（10 万样本，8 bit 仿射）：
#     |δ| 确实 ≤ s/2，且 U(−1,1) 那一组几乎顶到界（3.9205e-3 vs s/2 = 3.9215e-3）。
#   MSE 影响很小：10 万样本上把 z 从整数释放为实数，MSE 只在小数第 5 位变动。
INT_DELTA_BOUND = "|δ| ≤ s/2，两端点同时平移"
INT_DELTA_MSE_EFFECT = "MSE 只在小数第 5 位变动（本机实测，10 万样本，8 bit）"

# 来源：（**本机实测**）对称 vs 非对称的收益取决于**分布偏斜**（8 bit，20 万样本）：
#   判据写死：symmetric 用 torch 的 full-range 约定（S = max|r|/127.5）；
#              asymmetric 用无符号码字 0..255 的仿射（s=(b−a)/255, z=round(−a/s)）。
#   N(0,1)：非对称 39.77 dB / 对称 39.27 dB → 差 **0.50 dB**（几乎无差）
#   |N(0,1)|（单边，模拟 ReLU 后激活）：非对称 **45.96** / 对称 **39.95** → 差 **6.01 dB**
#   U(0,1)（单边）：54.15 / 48.13 → 差 6.03 dB
#   另外测了 restricted-range 那一支（`/127`）作对照：39.23 / 39.90 / 48.10，
#   与 full-range（`/127.5`）差 0.03–0.05 dB —— **两支约定在这三组分布上几乎不可分**，
#   但「torch 用哪一支」是**实现事实**，只能靠 observer 的 qparams 判别（见 04 的 §A1）。
#   ⚠ 计划文件写「45.69 / 39.64，差 6.05 dB」——**同量级但不是同一个数**
#     （种子、样本数与生成顺序都不同）。**正文按本仓库实测值写，并把判据一并写出。**
IMPL_SYM_VS_ASYM_DB = {
    # 分布：          (非对称, 对称, 差)
    "N(0,1)":        (39.77, 39.27, 0.50),
    "|N(0,1)|":      (45.96, 39.95, 6.01),
    "U(0,1)":        (54.15, 48.13, 6.03),
}
IMPL_SYM_VS_ASYM_RESTRICTED_DB = {"N(0,1)": 39.23, "|N(0,1)|": 39.90, "U(0,1)": 48.10}
IMPL_SYM_VS_ASYM_SAMPLES = 200_000
IMPL_SYM_VS_ASYM_SEED = 20260918

# ---------------------------------------------------------------------------
# 【实现】torch.int4 / torch.uint4 是空壳（五版本一致）
# ---------------------------------------------------------------------------

# ★ 行号引用的**版本坐标**（错误模式 P7 的对策：引文立刻在某版本上打印一次）：
#   文件与符号                                          2.7.1  2.8.0  2.11.0  2.13.0
#   torch/headeronly/util/Float8_e8m0fnu.h  ties to even   —      —      75      75
#   torch/headeronly/util/Float8_e8m0fnu.h  uint8_t lsb    —      —      85      85
#   torch/headeronly/util/Float8_e5m2fnuz.h epsilon()      —      —     429     429
#   torch/headeronly/util/Float8_e5m2.h     epsilon()      —      —     441     441
#   torch/ao/quantization/observer.py       vpmaddubsw    518    523     528     528
#   （e8m0fnu 那两个文件在 2.7.1 / 2.8.0 上**不存在**；`—` 表示该版本没有这个文件。）
#   **不是所有行号都跨版本一致**：`observer.py` 这一处四个版本三个值。
#   本文件与文档里凡引行号，都写明是哪个版本的。
#
# 来源：（实现，**本机实测**）五个解释器逐个跑，2026-09-18。
#   `torch.zeros(4, dtype=torch.int4)` **能构造**，`element_size()` 错报为 **1**
#   （语义上应为 0.5）；此后**任何承载数据的操作都失败**。
#   ⚠ **异常类型本身跨版本变了**（不是「行为」而是「异常类」变了）：
#     2.6.0 一律 `RuntimeError`；2.7.1 起 `fill_cpu` / `copy_kernel` /
#     `_local_scalar_dense_cpu` 变 `NotImplementedError`
#     （2.7.1 的 `fill_cpu` 还是 RuntimeError，2.8.0 起也变成 NotImplementedError）。
#   → 断言必须断「**抛了异常**」，**不能断具体异常类**，否则会在某个版本上假失败。
#   → **结论：目前只是可分配的空壳 dtype，不能承载数据。**
#     文档不得写成「torch 支持 int4 张量」。
INT4_ELEMENT_SIZE_REPORTED = 1
INT4_TRUE_ELEMENT_SIZE = 0.5
INT4_SHELL_PROBES = {
    "torch.zeros(4, dtype=dt)": "成功（只分配，不写数据）",
    "torch.tensor([1], dtype=dt)": "抛异常（2.8+ store_scalar: invalid type）",
    "tensor.fill_(1)": "抛异常（fill_cpu not implemented）",
    "tensor.copy_(...)": "抛异常（copy_kernel / copy_ not implemented）",
    "str(tensor)": "抛异常（_local_scalar_dense_cpu not implemented）",
    "tensor[0].item()": "抛异常（同上）",
    "tensor.float()": "抛异常（copy_ not implemented）",
    "tensor + 1": "抛异常（copy_ not implemented）",
}
INT4_EXC_TYPE_BY_VERSION = {
    "2.6.0":  "一律 RuntimeError",
    "2.7.1":  "RuntimeError 与 NotImplementedError 混用",
    "2.8.0":  "2.7.1 之后统一为 NotImplementedError（构造那一条仍是 RuntimeError）",
    "2.11.0": "同 2.8.0",
    "2.13.0": "同 2.8.0",
}

# ---------------------------------------------------------------------------
# 【实现 × 实现】NVFP4 vs MXFP4 的 SQNR 差（本机仿真，**不是硬件实测**）
# ---------------------------------------------------------------------------

# 来源：（**本机仿真**）元素格式 e2m1（值集 {0,±0.5,±1,±1.5,±2,±3,±4,±6}）在 float64 里
#   自己实现最近取整（torch 2.8.0 的 `float4_e2m1fn_x2` **没有 CPU copy kernel**，
#   实测 `x.to(torch.float4_e2m1fn_x2)` 抛 `NotImplementedError`，**跑不了真 e2m1 转换**）；
#   MXFP4 的 scale 走**真** `torch.float8_e8m0fnu` 取整；NVFP4 的 block scale 走
#   **真** `torch.float8_e4m3fn` 取整。10⁶ 样本，2026-09-18，torch 2.8.0。
#   ⚠ **这一节是仿真值，不是硬件实测值**，正文必须带限定语。
#   · N(0,1)：MXFP4 18.78 dB / NVFP4 20.43 dB → **+1.65 dB**
#   · 重尾分布族：优势随尾巴变重而扩大（见下表的四个实例）
#     ——**「重尾」不是一个数，必须写出分布名与参数**（P9：判据没写出来不得并列）
NVFP4_VS_MXFP4_SQNR_DB = {
    "N(0,1)":                    (18.78, 20.43, 1.65),
    "Laplace(0,1)":              (17.95, 20.66, 2.71),
    "StudentT(5)（截 ±1e3）":     (17.98, 20.70, 2.71),
    "StudentT(3)（截 ±1e3）":     (17.01, 20.81, 3.79),
    "StudentT(1)（截 ±1e3）":     (14.30, 28.13, 13.83),
}
NVFP4_VS_MXFP4_SAMPLES = 1_000_000
NVFP4_VS_MXFP4_IS_SIMULATION = True
NVFP4_VS_MXFP4_CONTROL = "同一套 e2m1 网格下只把 MXFP4 的块改成 16：N(0,1) 18.58 dB"
#   ⚠ 计划文件写「重尾 +2.97 dB」——本机在 StudentT(3) 上得 +3.79、在 Laplace 上得 +2.71，
#     **落在这条区间里但没有一个恰好是 2.97**。正文按本仓库实测值写并逐条标出分布。

# ---------------------------------------------------------------------------
# 术语（本主题用到的、容易被写成另一个意思的词）
# ---------------------------------------------------------------------------

TERMS = {
    "最小次正规数": "2^(1−bias−M)，**不是** torch.finfo 报的 tiny/smallest_normal",
    "smallest_normal": "2^(1−bias)；finfo 的 tiny 与它相等，不是最小次正规数",
    "eps": "2^(−M)，指相邻两个规格化数之间的相对间距",
    "bias": "2^(E−1) − 1；**fnuz 两个变体是 2^(E−1)**（比非 fnuz 恰大 1）",
    "块 scale": "MX 里每 32 个元素共享的 E8M0 尺度",
    "block scale": "NVFP4 里每 16 个元素共享的 E4M3 尺度（**与 MX 的块 scale 不是一个东西**）",
    "per-token": "沿 [tokens, features] 的 **axis 1** 归约，即**每行一个 scale**",
    "per-channel": "沿 [tokens, features] 的 **axis 0** 归约，即**每列一个 scale**",
    "UNRESOLVED": "tf32 的输入转换舍入模式：两处来源冲突，本机跑不了真 TF32，**不下断言**",
}

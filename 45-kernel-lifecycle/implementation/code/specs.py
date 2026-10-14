"""45-kernel-lifecycle/implementation 的外部事实常量源。

本模块是 CONTRIBUTING §2 约定的**共享事实源**（`specs.py`）：三个编号程序从它读常量，
推导过程可复核，**常量本身不可自证**。

★ 本主题的常量按来源等级分四类（`CONTRIBUTING.md` §2 的 A–E 五级，本模块用到 A/B/C/D/E）：

  · **A 级：厂商规范**——NVIDIA Compute Sanitizer 的工具清单与各工具的作用域、
    IEEE 754 的单位舍入误差、warp size。
  · **B 级：某一实现的官方规格**——vLLM 的 `vllm.ir.op` / `vllm/ir/tolerances.py`
    （IR 三合一、逐位一致的 dispatch 判据、按 dtype 的容差表）、PyTorch 的
    reproducibility 文档。**只能支撑「这一家这么做」**（P10）。
  · **C 级：预印本**——arXiv:2606.20128（Correctness Illusion）与 arXiv:2607.16228
    （Tolerance Calibration），**均未确认同行评审**，正文必须写明。
    ★ 本模块**刻意收录它们**：域 README 的硬规则 2 说「D 级来源不得单独支撑一条论断，
    须有 A/B/C 级佐证」——**这两篇就是那张 D 级容差表的 C 级佐证**。
  · **D 级：社区材料**——本域规格 §Chunk3 给出的那张容差表，**单一来源、无出处页面**。
    本模块保留它，**唯一目的是让三个程序算出它错在哪**（见 §2 的 `SPEC_D_*`）。
  · **E 级：本库推导/建模**——独立性定理的两个捕获数、调试顺序的分层模型与其边界条件。

★ **本模块刻意不含散文。** 上一轮（`requirement-discovery`）的 C6 教训：把来源里的
  **描述性段落**整段搬进 `specs.py`，程序**没法计算散文 ⇒ 只能 `print` ⇒ 必然漏网**。
  ⇒ **规则：每个模块级常量必须是可被计算的值**（数、名、布尔、阈值、元组、字典）。
    描述性文字归**文档**；来源里的每个数字以**结构化字段**的形式落在这里。

★ 本模块的常量**全部**被三个程序断言钉住。**判据是机械的**：`01_reference_independence.py`
  用 `ast` 读本目录另外两个程序里的 `COVERED_*` 元组，断言**三者的并集 == 本模块的
  全部模块级常量**（见该程序 §D.8）。⇒ 在这里加一个常量而没人断言它，`make run` 会失败。
"""

# ===========================================================================
# 1. 参考实现的独立性（**E 级：本库的定理与实测** + **B 级：vLLM 的做法**）
# ===========================================================================
#
# ★ 本篇的核心论断：若 `F = S ∘ f`、`R = S ∘ r`（`S` 共享），则在比对 **`S` 直接产出的量**时，
#   `F` 与 `R` 的差值**恒为 0**——共享部分的错误在两边同向出现，捕获率为零。
#
# ★ 本节前六条是**本库的推导与构造性实测**（[E]），后八条是 vLLM 的官方做法（[B]）。
#   两组的**分工**是本节的要点：定理说的是「什么条件下测不出来」，
#   vLLM 的做法说的是「一个工程为此付了什么代价（把三份东西合成一份）」。

INDEPENDENCE_SHARED_SUBEXPRESSION_DIFF_IS_ZERO = True
# 来源：**本库推导 [E]**（`F = S ∘ f`、`R = S ∘ r` 时的代数恒等），并由
#       `01_reference_independence.py` 的 §C 构造性复现。
# 口径：★★ **它是有前提的**：前提①比对的是 **`S` 之后未经独立处理的输出**；
#       前提②缺陷是**静默的**。⇒ 正文**不得**写成「共享代码的参考实现一定测不出任何 bug」。
#       丢掉前提①，`R_indep` 就测得出（这正是本程序 §C 的对照）。

SHARED_REFERENCE_CAPTURE_COUNT_ON_SHARED_DEFECT = 0
# 来源：本库构造性实测 [E]（`01_reference_independence.py` §C.2，固定种子）。
# 口径：★ 这是**捕获数**，不是「误差大小」——同一次运行里它报出的**最大误差也是 0.0**。
#       两个断言都要跑出来：「没抓到」与「显示一切正常」是两件事。

INDEPENDENT_REFERENCE_CAPTURE_COUNT_ON_SHARED_DEFECT = 3
# 来源：本库构造性实测 [E]（同上，n=13 ⇒ 补 3 个元素 ⇒ 差值恰为 3.0）。
# 口径：★ 这个 **3** 不是调出来的，是 `(-13) % 4`。换 n 就换数——所以它是**可反驳的**：
#       程序断言它恰好等于 `(-N_ELEMENTS) % 4`。

SHARED_DEFECT_IS_SILENT_IS_PREREQUISITE = True
# 来源：**本库的实跑教训 [E]**（本域规格 §Chunk3 逐字记录：第一版缺陷「少 pad 一个元素」
#       导致 `reshape` 抛 `ValueError`，脚本**崩溃**）。
# 口径：★★ 「崩溃」在变异测试里是**假捕获**的一种——它证明的是「有地方不对」，
#       不是「这个参考实现能测出这个缺陷」。⇒ 缺陷必须只改**结果的值**，不改**控制流**。

CRASH_COUNTS_AS_FALSE_CAPTURE = True
# 来源：本库判定 [E]（同一条教训的机器化）。
# 口径：★ 与上一条配对。本程序把一次运行分成三态：`captured` / `missed` / `false_capture`
#       （异常）。**报告里的「harness 异常数」就是第三态的计数**——它必须单列，
#       不能并进「捕获」里，否则变异测试会自己骗自己。

REFERENCE_SELECTION_ORDER = ("cpu_fp64", "cpu_fp32", "existing_framework_op", "naive_numpy")
# 来源：**本库的选取规则 [E]**（三条优先级，由「谁更可能独立」排序）。
# 口径：★ 排序的**唯一依据是独立性**，不是精度也不是速度：
#       ① CPU 上的 fp64/fp32 实现（与被测的 GPU 路径**机制不同**）；
#       ② 框架已内置的等价 op（**实现者不同**）；
#       ③ 都没有时，手写**明显正确的朴素 NumPy 版**（**算法不同**）。
#       ⚠ 第三档是最弱的一档：朴素版一旦与被测实现**共用同一个 helper**，就退化成 §1 的定理。

INDEPENDENCE_IS_MECHANICALLY_CHECKABLE = True
# 来源：本库判据 [E]。
# 口径：★★ **「独立」不是一个态度，是一个可以被机械判定的性质**：给共享代码装一个调用计数器，
#       参考实现跑完后计数必须**恰好为 0**。本程序 §B.3 就是这么做的。
#       ⇒ 「我们的参考实现是独立的」这句话在本篇里**有判据**，不是自我声明。

VLLM_IR_NATIVE_IMPL_ROLES = ("semantic_definition", "default_implementation", "test_baseline")
# 来源：vLLM 官方文档 `design/vllm_ir`（"The native implementation serves three roles"：
#       semantic definition / default implementation / reference for testing）  [B]
#       （**两次独立检索一致**：官方 design 文档 + `vllm.ir.op` 的 API 页）
# 口径：★★ **本篇引用它来支持「三合一」这条工程做法，不是来支持独立性**——
#       恰恰相反：**native 既是语义定义又是测试基线**，意味着
#       **拿 native 当参考实现去测 `register_impl` 的另一份实现时，参考的独立性来自
#       「两份实现是不同的人写的」，不是来自「native 更正确」**。
#       ⚠ 这与 §1 的定理**不矛盾**，但要把作用域写对（P7）：三合一解决的是
#         **漂移**（三份东西会互相不一致），**不解决**「两者共有部分的错误」。

VLLM_IR_NATIVE_PROVIDER_NAME = "native"
VLLM_IR_NATIVE_ALWAYS_SUPPORTED = True
# 来源：vLLM `vllm/ir/op.py`（`RESERVED_PROVIDERS = ['native', 'unfused']`；
#       native 恒 `supported=True`，无优先级列表时回退到 native）  [B]（**经检索转述**）
# 口径：★ 「永远有兜底」⇒ **不会因为所有 provider 都不支持而选不出实现**。

VLLM_IR_IMPL_MUST_MATCH_NATIVE_SCHEMA = True
VLLM_IR_SCHEMA_MISMATCH_RAISES = "ValueError"
# 来源：vLLM `vllm/ir/op.py` 的 `register_impl`（impl 的推断 schema 必须与 native
#       **逐个参数**在名字/类型/默认值上相同，否则 `ValueError`）  [B]（**经检索转述**）
# 口径：★ 这是「**语义定义只有一份**」的执行机制：schema 不一致**当场报错**，
#       而**不是**静默地注册一个签名不同的实现。

VLLM_IR_SUPPORTS_ARGS_KEYWORD_ONLY_DISALLOWED = True
# 来源：同上（"keyword-only params are disallowed (for faster hot-path dispatch)"）  [B]
# 口径：★ 一条**性能考虑反过来约束了接口**的例子——与本篇主线的关系是：
#       支持性检查跑在**热路径**上，所以它的签名被限制（**不能有关键字专用参数**）。

VLLM_IR_DISPATCH_CONSISTENCY_TOLERANCE = (0.0, 0.0)
# 来源：vLLM IR 的 op 测试基础设施（"verify dispatched call matches direct call exactly
#       (`rtol=0, atol=0`)"，commit #40167 / issue #38782）  [B]（**经检索转述**）
# 口径：★★★ **本篇最锋利的一条对照，两个 0 是同一个 0，含义相反**：
#       · 「dispatch 路径 vs 直接调用」的差 **必须是 0**——因为那**本来就是同一份实现**，
#         0 意味着**正确**；
#       · 本程序 §C 里「共享代码版参考实现」报出的 0——也是 0，但它意味着**测不出来**。
#       ⇒ 看到 `maxdiff == 0.0` 时，**第一件事是问「两边是不是同一个东西」**，
#         而不是「通过」。这一条把 §1 的定理从代数命题变成了**工程上每天都会遇到的判读错误**。

VLLM_IR_OP_SEMANTIC_SELFCHECK_API = "opcheck"
# 来源：同上（"`opcheck` per provider (args not mutated, outputs don't alias inputs
#       except in `maybe_inplace`)"）  [B]（**经检索转述**）
# 口径：★ 容差比对**查不出**「参数被就地改了」和「输出与输入别名」——它们是**两条独立的检查**。
#       ⇒ 验证矩阵里**必须有一维**不属于「数值比对」。


# ===========================================================================
# 2. 验证矩阵与容差（**B 级** vLLM · **C 级** 预印本 · **D 级** 规格表 · **A 级** IEEE 754）
# ===========================================================================
#
# ★★ 本节是本篇最需要**来源纪律**的地方：同一张「按 dtype 定容差」的表，本模块里有
#   **三个不同等级**的版本，正文必须**分别标注**，且必须写出它们**互相不一致在哪**。

VLLM_DEFAULT_TOLERANCES = {
    # dtype        : (atol, rtol)
    "float64": (1e-8, 1e-8),
    "float32": (1e-5, 1.3e-6),
    "float16": (1e-3, 1e-3),
    "bfloat16": (1e-3, 1.6e-2),
    "float8_e4m3fn": (1e-1, 1e-1),
    "float8_e5m2": (2e-1, 2e-1),
    "int8": (1.0, 0.0),
}
# 来源：vLLM `vllm/ir/tolerances.py` 的 `DEFAULT_TOLERANCES`（`ToleranceSpec = dict[dtype,
#       dict[str, float]]`，键为 `atol` / `rtol`）  [B]（**经检索转述**，两次检索）
# 口径：★ 条件是 **`torch.testing.assert_close` 的 `atol + rtol·|ref|` 口径**（不是绝对口径）。
#       ⚠ **一处两次检索不一致，必须记下来**：本实现者的两次检索里，`bfloat16` 的 **atol**
#         一次读到 `1.6e-2`、一次读到 `1e-3`（两次的 **rtol 都是 1.6e-2**）。
#         ⇒ 本模块**采用 `(1e-3, 1.6e-2)`**（第二次检索的表格更细，且与「bf16 的 rtol 比
#         fp16 宽」这条文字说明自洽），并**在正文里显式写出这处未决**——
#         按 §0.6 硬规则 5，**不把不确定的那个数当成已知**。
#       ⚠ 第二次检索还给出 `int8` 的 atol 是 `1`、第一次给出 `—`（未列）。
#         本模块取 `1.0`，因为 `rtol = 0` 时必须有 atol 才有意义。

VLLM_TOLERANCES_ARE_CONSERVATIVE_BY_DESIGN = True
# 来源：同上（官方注释：defaults are "intentionally conservative (permissive) to avoid
#       false failures across different hardware and kernel implementations"）  [B]
# 口径：★★ **这句话是整张表的定性说明，也是本篇的核心论断的来源**：
#       容差**故意松**——松到「避免假失败」与「吸收真缺陷」是同一件事。
#       ⚠ 它**不等于**「这张表错了」：松是有理由的（跨硬件、跨实现）。
#         本篇的结论是「**松的代价必须被量化**」，不是「改用紧的」。

VLLM_TOLERANCE_LOOKUP_RAISES_ON_UNLISTED_DTYPE = True
# 来源：同上（`get_tolerance` 在表里查不到 dtype 时 `raise ValueError("No tolerance
#       defined for dtype ...")`，**没有兜底默认值**）  [B]（**经检索转述**）
# 口径：★★ **反直觉且可证伪**：一个「故意松」的表，**却对没列的 dtype 直接报错**。
#       两者是同一套设计的两面——**松是「已知格子的取值」，不是「未知格子的默认」**。
#       ⇒ 判据表里有一条：「算子支持一个不在容差表里的 dtype」⇒ **不允许**跑测试，
#         必须先 `override_tolerance` 显式声明（**静默放行是不被允许的**）。

VLLM_TOLERANCE_OVERRIDE_API = "override_tolerance"
# 来源：同上（`IrOp.override_tolerance(dtype, *, atol, rtol)`；查找顺序
#       **per-op override → DEFAULT_TOLERANCES → else raise**）  [B]（**经检索转述**）
# 口径：★ 「更紧或更松的界」**必须由一个显式的、按 (op, dtype) 的声明给出**，
#       而不是在测试里临时写一个魔数。

SPEC_D_TOLERANCE_TABLE = {
    # 对照                    : (atol, rtol)
    "fp64_ref_vs_fp32_kernel": (1e-8, 1e-5),
    "fp64_ref_vs_fp16_kernel": (1e-3, 1e-2),
    "fp64_ref_vs_bf16_kernel": (1e-3, 5e-2),
}
# 来源：**本域规格 `2026-09-20-kernel-lifecycle-domain.md` §Chunk3 的容差表**  [D]
# 口径：★★★ **本模块保留这张表的唯一目的，是让程序算出它错在哪。**
#       · 它的来源是**社区材料**（D 级），且规格**没有给出出处页面**
#         ——正文出现它的每个数字时**必须写明「D 级、无出处页面、未被独立复现」**。
#       · §2 的另外两组常量（[B] vLLM 与 [C] 预印本）**就是它的佐证**：
#         域 README 硬规则 2 要求 D 级不得单独支撑论断，**本节满足这条要求的方式是
#         让它被更高等级的来源推翻**——见 `03` 号程序的 §A.4 与 02 篇正文。

SPEC_D_TOLERANCE_TABLE_IS_D_LEVEL = True
SPEC_D_TOLERANCE_TABLE_HAS_NO_SOURCE_PAGE = True
# 来源：本库对规格的核查记录 [E]。
# 口径：★ 两个布尔**各自会被断言**。它们的作用是**防止正文把 D 级表写成通用结论**：
#       02 篇正文必须出现「D 级」字样，程序断言这两个布尔为 `True`。

FP16_UNIT_ROUNDOFF_U = 2.0 ** -11
BF16_UNIT_ROUNDOFF_U = 2.0 ** -8        # [A]（与上面两条同源：IEEE 754 的 p = 8）
FP32_UNIT_ROUNDOFF_U = 2.0 ** -24
# 来源：IEEE 754 的**单位舍入误差** `u = 2^-p`，`p` = 有效二进制位数（含隐藏位）  [A]
#       （binary16 的 1+5+10 ⇒ p = 11；bfloat16 的 1+8+7 ⇒ p = 8；binary32 ⇒ p = 24）
# 同值位置：45-kernel-lifecycle/operator-design/code/specs.py（**同域内的上一主题**，
#           此处按「跨主题常量复制」纪律重抄，正文注明与该主题一致）
# 口径：★★ **`u` 与 `130-model-optimization` 的 `EPS` 不是同一个量，别混**：
#       `130-model-optimization/quantization/code/specs.py:441` 的 `BF16_EPS = 2^-7` 是
#       **1.0 处的间距**（`finfo.eps`），而 `u = 2^-8` 是**它的一半**。
#       ⇒ 引「本库 130-model-optimization 的精度推导」时必须说清引的是哪一个：
#         本主题的误差界用 **`u`**（与 `04-low-precision-formats` 的 `γ_K = K·u/(1-K·u)` 同口径），
#         不是 `EPS`。

UBF16_OVER_U16 = 8.0
UBF16_OVER_U16_ORDERS_OF_MAGNITUDE = 0.9031
# 来源：**本库推导值 [E]**（`BF16_UNIT_ROUNDOFF_U / FP16_UNIT_ROUNDOFF_U` = `2^-8 / 2^-11`
#       = `2^3` = 8；十进制 `log10(8) = 0.90309`）。
# 口径：★★ **命名必须写清谁除谁**——本常量是 `u(bf16) / u(fp16)`，**不是**反过来。
#       本主题的第一版把它写成 `UBF16_OVER_U16`（u16 除 ubf16），值为 **0.125**，
#       被 `02_verification_matrix.py` 的 A.2 断言当场抓出来。
#       ⇒ 这正是本域 P9（同名不同指）在**常量命名**上的形态：一个比值**方向反了**
#         仍然「看起来是个数」，只有在被断言时才会暴露。
#       ★ 与 `operator-design` 的 `U16_OVER_U32 = 8192`（13 位）**成对读**：
#         fp16 与 bf16 差 **3 位**，fp16 与 fp32 差 **13 位**。
#         ⇒ 「fp16 与 bf16 的容差该差多少」的**正确答案是放宽 8 倍**，不是「都取一个数」。

ARXIV_2606_20128_ILLUSION_SOURCES = ("shape", "dtype", "tolerance")
# 来源：arXiv:2606.20128 *The Correctness Illusion in LLM-Generated GPU Kernels*
#       （D. Sarkar, 2026-06-18）的「three sources of illusion」  [C]
#       （**预印本，未确认同行评审**；两次独立检索一致）
# 口径：★★ 本条与**本主题 §1 的测试矩阵五维**是**同一件事的两种说法**：
#       论文说的三个来源 **shape / dtype / tolerance** 正是本主题形状维、dtype 维、
#       §2 容差节的内容。⇒ 五维不是本库编的清单，外部有一篇专门论证它缺一不可的工作。
#       ⚠ 论文的语料是 **LLM 生成的 kernel 与作者播种的转写错误**，
#         **不是**任何真实部署系统的缺陷率——正文不得外推（这是它自己的 caveat）。

ARXIV_2606_20128_BOUNDARY_SHAPES = (1, 3, 7, 256, 1025)
# 来源：同上（"deliberately including boundary values (e.g. `H ∈ {1, 3, 7, 256, 1025}`)"）  [C]
# 口径：★★ **这一组值本身就是本篇 C 类的设计依据**：`1/3/7` 是**小的非 2 的幂**，
#       `256` 是 2 的幂，`1025 = 2^10 + 1` 是**「整块 + 1」**。
#       ⇒ 只测 2 的幂的矩阵**恰好漏掉** `1/3/7` 与 `1025` 这两类。
#       ⚠ 论文的 H 是 attention 的头数；本主题把它作为**边界形状集的示例**引用，
#         不声称它对所有算子都是完备集。

ARXIV_2606_20128_BOUNDARY_REMOVAL_ZEROES_SHAPE_BUG_RECALL = True
# 来源：同上（"removing boundary values drops recall on shape-dependent bugs to zero"）  [C]
# 口径：★★★ **这是本篇 C 类「捕获数为 0」的外部对应物**，而且比本库的构造更强：
#       它不是「某个缺陷漏了」，是**一整类缺陷（shape-dependent）的召回率归零**。
#       ⇒ 本程序 §C 构造的「只写完整块、尾部不写」正是这一类。

ARXIV_2606_20128_REFERENCE_PRECISION = "fp64_cpu"
ARXIV_2606_20128_TOLERANCE_IS_ABSOLUTE_ONLY = True
# 来源：同上（"compares outputs against a high-precision fp64 CPU reference using
#       per-(op, dtype) absolute-only tolerances (unlike PyTorch's combined absolute +
#       relative allclose)"）  [C]
# 口径：★★ **参考实现的选取规则在这里被外部独立执行了一遍**：fp64 / CPU。
#       这与本模块 §1 的 `REFERENCE_SELECTION_ORDER` 第一档**完全一致**
#       ——两处独立（本库规则 + 外部的做法）同时指向 CPU fp64。
#       ★ 另一处**不一致**要写出来：论文用**绝对容差**，vLLM 用 **`atol + rtol·|ref|`**。
#         两者**不可比**——同一个 `1e-3` 在两种口径下不是同一个界（P9 的同名不同指）。

ARXIV_2606_20128_GPUS = ("RTX 3060", "A10", "L40S", "A100 SXM4", "H100 NVL")
ARXIV_2606_20128_VERDICTS_IDENTICAL_ACROSS_GPUS = True
# 来源：同上（26 个算子在**五类 GPU** 上重跑，"Verdicts were identical across all GPUs:
#       10 of 10 illusions caught, 16 of 16 controls clean"）  [C]
# 口径：★★ **反直觉**：同一份**判定**在五类 GPU 上完全一致 ⇒
#       「换块卡再测一遍」**不能**作为「这个实现是对的」的补充证据。
#       ⚠ **不得**读成「换卡结果逐位相同」——论文比的是**判定**（flagged / clean），
#         不是逐位数值。这两件事的差别正是本主题 §1 的定理所关心的。

ARXIV_2606_20128_BUGGY_VARIANTS_ARE_AUTHOR_SEEDED = True
# 来源：同上（论文自陈 caveat："The buggy variants are author-seeded with documented LLM
#       transcription patterns, not pulled directly from real LLM outputs"）  [C]
# 口径：★★ **本主题 §C 的「本库构造」声明，与这篇论文的自陈是同一件事**，
#       但论文**把话说在了前面**。本主题照抄这个做法：构造就是构造，明写。

ARXIV_2607_16228_CALIBRATION_RULE = "p95_x_1.5"
ARXIV_2607_16228_CORPUS_SIZE_OPS = 26               # [C]（同下一条的来源块）
ARXIV_2607_16228_MEDIAN_TIGHTENING_RANGE = (100, 200)   # [C]（同上）
ARXIV_2607_16228_ATTENTION_TRITON_FP16_TIGHTENING = 2184   # [C]（同上）
ARXIV_2607_16228_HANDPICKED_ATOL_ATTENTION_TRITON_FP16 = 5.0e-2   # [C]（同上）
ARXIV_2607_16228_CALIBRATED_ATOL_ATTENTION_TRITON_FP16 = 2.29e-5
# 来源：arXiv:2607.16228 *Operator-Aware Mixed-Precision Tolerance Calibration for Tensor
#       Kernels*（D. Sarkar, 2026）  [C]（**预印本，未确认同行评审**；两次独立检索一致）
# 口径：★★★ **这是本篇最强的一条外部数字，也是 D 级容差表的直接反证**：
#       · 标定规则 = **正确 kernel 自身误差分布的 p95 × 1.5**（不是手挑）；
#       · 手挑容差与标定值的中位差距 **100×–200×**；
#       · 极端例 **`attention_triton` fp16**：手挑 `atol = 5.0e-2` → 标定 `2.29e-5`，
#         收紧 **2184 倍**。
#       ⚠ **口径必须写全**：语料 **26 个算子**（`gpuemu`），运行环境 **RTX 3060**，
#         采集窗口 **2026-05-27 – 2026-06-11**，标定是**逐 (op, dtype) 的绝对 atol**。
#         写不出条件的数字不得引用（域 README 硬规则 3）。
#       ⚠ 论文自陈：只做了 **atol**，未做 `atol + rtol·|r|`；语料是**精选的**。
#       ★ 注意 `5.0e-2` 与 D 级表的 `bf16` 行 rtol **数值巧合**——**两者不是同一个量**
#         （一个是绝对 atol、一个是相对 rtol），**不得并提**（P9）。

FP16_TO_BF16_CONSTANT_TOLERANCE_IS_WRONG = True
# 来源：**本库推导 [E]**（由 `UBF16_OVER_U16` 与 §D 表的两行算出）。
# 口径：★★ 本主题 02 篇的**核心可证伪论断**，一句话：
#       **「同一个 `atol` 走两个 dtype」与「容差随 `u` 按 8 倍缩放」在 bf16 上差 8 倍**——
#       D 级表给 fp16 的 rtol 是 `1e-2`、给 bf16 是 `5e-2`，**只差 5 倍**，
#       而两者的 `u` 差 **8 倍**。⇒ **那张表的 fp16 行相对太松、bf16 行相对太紧。**
#       ⇒ 「容差不能是一个常数」这句话在本主题里**有一个具体的数**。

FP16_ACCUMULATOR_IS_FP32_IN_REPO = True
# 来源：**本库既有事实 [E]**，与 `130-model-optimization/quantization/code/specs.py:349`
#       的 `FP16_MM_CPU_ACCUM = "fp32"` **一致**（该处用一个 CPU fp16 矩阵乘的探针证明
#       累加**不是** fp16：`[2048,1,1] @ ones → 4096.0`，纯 fp16 累加会得 2048）。
# 口径：★★ **它决定了 fp16 那行容差该按哪个 `u` 推**——这是本篇 A 类能不能算对的分水岭：
#       · 若累加在 **fp32**（硬件默认：PTX `wmma` 在 `.ctype/.dtype` 为 `.f32` 时
#         累加「至少单精度」，见 operator-design 的 `PTX_WMMA_ACC_F32_AT_LEAST`），
#         则误差与 `K` **基本无关**，量级是**输入端的存储舍入** `≈ u16`；
#       · 若累加在 **fp16**，则误差按 `√K·u16` 增长，`K = 4096` 时到 **3.1e-2**，
#         **超过 D 级表的 `1e-2`** ⇒ 那时是**表太紧、会漫天假失败**。
#       ⚠ 同一个数（`1e-2`）在两种累加口径下**一个太松一个太紧**——
#         所以「容差表」必须与「累加口径」一起读，单看容差无法判定。
#       ⚠ **不得外推到 GPU**（`130-model-optimization` 的原话）：PTX 里 `.f16` 累加变体
#         **确实存在**，只是本机 CPU 路径不走它。本主题断言的是**本库的口径**，
#         不是「所有硬件都这样」。


# ===========================================================================
# 3. 调试顺序与工具落点（**A 级** NVIDIA · **B 级** PyTorch · **E 级** 本库建模）
# ===========================================================================

COMPUTE_SANITIZER_TOOLS = ("memcheck", "racecheck", "initcheck", "synccheck")
# 来源：NVIDIA *Compute Sanitizer* 官方文档（`compute-sanitizer --tool
#       memcheck|racecheck|initcheck|synccheck`）  [A]（**两次独立检索一致**）
# 口径：★★★ **四档，一档都不能漏**。本主题的边界是「不重复讲 50-performance 已覆盖的工具」，
#       但 `synccheck` 属于**这一篇的落点**，理由有两层：
#       ① 它正对「共享内存同步分歧」这类**边界 bug**——本篇的 §C 要定位的正是这一类；
#       ② **本库自己在用它**：`40-kernels/reduce-scan/code/verify.py:1255` 的
#          `print("真机上的观测方式：\`compute-sanitizer --tool synccheck\`。")`
#          与 `40-kernels/reduce-scan/01-reduction-algorithms.md:327` 的规避段
#          （"真机自检用 `compute-sanitizer --tool synccheck`"）。
#       ⚠ 上一版规格把 `synccheck` 漏在外面，是 P2（整个载体缺失）的形态。

COMPUTE_SANITIZER_DEFAULT_TOOL = "memcheck"
# 来源：同上（"The default tool is `memcheck`"）  [A]
# 口径：★ **默认值是默认值，不是推荐值**——不指定 `--tool` 时你**只**跑了越界检查，
#       共享内存竞争、未初始化读、同步分歧**都没查**。
#       ⇒ 判据表里有一条：「只跑了不带 `--tool` 的 compute-sanitizer」⇒ 视为**只查了一维**。

COMPUTE_SANITIZER_MEMCHECK_SCOPE = (
    "out_of_bounds", "misaligned", "hardware_exception",
    "malloc_free_misuse", "cuda_api_error", "memory_leak",
)
# 来源：同上（memcheck 的检测项清单）  [A]（经检索转述）
# 口径：★ 它覆盖 global / local / **shared** 三种存储的越界与错位。
#       ⚠ 与 `racecheck` 的分工：**越界**是 memcheck 的，**竞争**是 racecheck 的，
#         两者**互不覆盖**——一次越界读可能同时是竞争，但只跑一个查不出另一个。

COMPUTE_SANITIZER_RACECHECK_HAZARDS = ("WAW", "WAR", "RAW")
# 来源：同上（"canonical hazards: Write-After-Write (WAW), Write-After-Read (WAR),
#       Read-After-Write (RAW)"）  [A]（经检索转述）
# 口径：★ 三者都是**共享内存**上的（"currently only for on-chip shared memory"）——
#       它对**全局内存的类型双关竞争、以及 `atomic` 的浮点不确定性都不报**。
#       ⇒ 判据表里有一条：「racecheck 干净」**不蕴含**「确定性」——两件事的检查器不同。

COMPUTE_SANITIZER_INITCHECK_SCOPE_ONLY_DEVICE_GLOBAL = True
# 来源：同上（"It currently supports only device global memory"）  [A]（经检索转述）
# 口径：★★ **这是一条会被误读成「我查过了」的限定**：`initcheck` **不查共享内存**。
#       ⇒ 一个「共享内存未初始化就读」的缺陷，`initcheck` **报不出来**。
#       本节收录它就是为了让「四档各查什么」变成一张**有边界的表**，而不是一个勾选清单。

COMPUTE_SANITIZER_SYNCHECK_INTRINSICS = ("__syncthreads", "__syncwarp")
COMPUTE_SANITIZER_SYNCHECK_REPORTS = (
    "divergent_threads_in_block", "divergent_threads_in_warp", "invalid_arguments",
)
# 来源：同上（synccheck "checks correct use of `__syncthreads()` and `__syncwarp()`
#       intrinsics and their Cooperative Groups API counterparts. It reports divergent
#       threads in a block, divergent threads in a warp, and invalid arguments."）  [A]
#       （经检索转述）
# 口径：★ **它不是性能工具，是正确性工具**：`__syncthreads()` 被块内**所有**线程
#       在同一点执行是**语义要求**（`40-kernels/reduce-scan/01-reduction-algorithms.md:327`
#       的原话），违反它得到的是**静默的错值**，不是崩溃。

COMPUTE_SANITIZER_PREDECESSOR_REMOVED_IN = "CUDA 12.0"
# 来源：同上（Compute Sanitizer 是 CUDA-MEMCHECK 的继任者；后者在 **CUDA 11.6 弃用**、
#       **CUDA 12.0 起移除**）  [A]（经检索转述）
# 口径：★ 收录它是为了一条**可机械判定的判据**：「照着一篇提到 `cuda-memcheck` 的旧
#       教程做」⇒ 在 CUDA ≥ 12.0 的环境里**命令不存在**。这不是「风格问题」。

SYNCHECK_ALREADY_USED_IN_REPO = True
# 来源：本库核查记录 [E]（`40-kernels/reduce-scan/code/verify.py:1255` 与
#       `40-kernels/reduce-scan/01-reduction-algorithms.md:327`，两处均已打开文件核对）。
# 口径：★ 本主题**回链**该篇，**不重讲** `__syncthreads()` 的用法（§0.5 边界）。

DEBUG_LAYERS = ("layout", "copy", "compute", "edges")
DEBUG_LAYER_ORDER_IS_DEPENDENCY_ORDER = True
# 来源：**本库建模 [E]**（本篇 §1 的分层模型）。
# 口径：★ 分层的**依据是依赖**，不是「重要性」：后面的层依赖前面的层**正确**——
#       `compute` 的输入是 `copy` 搬过来的、`copy` 的地址是 `layout` 算出来的。
#       ⇒ 「层」的判据是**可判定的**：把上一层的输出换成**手算的正确值**，
#         若缺陷消失，则缺陷在上一层（本程序 §A.2 用这条做定位）。

DEBUG_FIXED_ORDER_ALWAYS_OPTIMAL = False
# 来源：**本库推导 [E]**（本节的分层模型 + 史密斯规则）。
# 口径：★★★ **本篇最诚实的一条，也是「可证伪」的落点**：
#       固定顺序 `layout → copy → compute → edges` 在**步数模型**下最优
#       （当先验沿该顺序递减时），但在**成本模型**下**不最优**——
#       因为 `compute` 的检查成本远高于 `edges`，而史密斯规则要求按 `c/p` 升序检查。
#       ⇒ 正文**不得**写成「按依赖顺序检查总是最好」。**边界写在本常量与
#         `DEBUG_LAYER_SUBHYPOTHESES` 上，不是一个形容词**。

DEBUG_LAYER_SUBHYPOTHESES = {"layout": 2, "copy": 3, "compute": 6, "edges": 2}
# 来源：**本库建模 [E]**（每层「要排除多少个子假设才算排除这一层」）。
# 口径：★★ 它是**成本模型里的成本**，也是 §C 里「预算耗尽」的那个数：
#       `compute` 有 6 个子假设（数学、索引、同步、精度、展开、指令），
#       而 `edges` 只要 2 个（尾部、退化形状）。**这不是测出来的，是明写的建模输入**——
#       任何读者都能改成别的值重算（与 operator-design 的
#       `ADDRESS_AND_INDEX_REGS_ASSUMED = 0` 是同一个做法：**缺口要标注，不要填平**）。

DEBUG_LAYER_PRIORS = {"layout": 0.40, "copy": 0.25, "compute": 0.22, "edges": 0.13}
# 来源：**本库建模 [E]**（本主题用的先验分布，四项和为 1）。
# 口径：★★ **它是本程序 §B 的边界条件的输入**，而且这一组值是**刻意挑成「对固定顺序有利」
#       的那一侧**：先验沿依赖顺序**严格递减**（0.40 > 0.25 > 0.22 > 0.13）。
#       ⇒ 在**步数模型**下固定顺序恰好最优（这是对它最有利的输入）。
#       ⇒ 如果在这样的输入下，**成本模型**里固定顺序**仍然**不是最优，
#         那结论就不是「先验挑得不好」，而是**模型本身的问题**——
#         这是本程序 §B 的论证结构（**先把对手的条件给足，再看结论**）。
#       ⚠ 「先验递减」这个假设本身**没有来源**（见 `UNSOURCED` 第 5 项），
#         所以正文必须写明它是**建模输入**、并给出结论随它变化的方向。

DEBUG_CHECK_BUDGET_STEPS = 8
# 来源：**本库建模 [E]**（§C 里「一次调试会话我愿意花多少个检查步」）。
# 口径：★ 它决定 §C 的结论是「**在预算内无法定位**」而不是「永远定位不了」——
#       措辞必须是**前者**。`layout(2) + copy(3) + compute(6) + edges(2) = 13 > 8`。

PYTORCH_ATOMICADD_IS_A_NONDETERMINISM_SOURCE = True
PYTORCH_NONDETERMINISTIC_OPS = ("index_add_", "scatter_add_", "bincount")   # [B]（同下一条的来源块）
PYTORCH_NO_SIMPLE_WAY_TO_AVOID_ATOMIC_NONDETERMINISM = True
# 来源：PyTorch 官方 *Reproducibility* 文档（randomness notes）：`atomicAdd` 会使
#       **加法的顺序不确定**，而浮点加法不满足结合律，"can introduce different
#       floating-point rounding errors on each evaluation"；原文逐字写
#       **"There is currently no simple way of avoiding nondeterminism in these functions."**
#       [B]（**两次独立检索一致**；出处到文件级：`docs/source/notes/randomness.rst`）
# 口径：★★ **这是「确定性检查」这一节的规范依据**，而且它给的是**否定结论**：
#       不是「加个 flag 就好」，而是**这类算子没有简单办法**。
#       ⇒ 判据表里有一条：「一个算子用了 `atomicAdd` 且没有确定性替代」⇒
#         **不得**把它纳入逐位比对的测试集；要么换实现，要么把比对降级为容差比对
#         **并明写**该比对不查确定性。
#       ⚠ B 级：它只支撑「**PyTorch 这么做/这么说**」。

PYTORCH_DETERMINISTIC_API = "torch.use_deterministic_algorithms"
PYTORCH_DETERMINISTIC_RAISES_RUNTIMEERROR = True    # [B]（同上）
PYTORCH_WARN_ONLY_DOES_NOT_ENFORCE = True
# 来源：同上（`torch.use_deterministic_algorithms(True)` 在算子无非确定性替代时
#       **抛 `RuntimeError`**；`warn_only=True` 只发警告，**不会**真的强制；
#       `warn_only=False` 才启用确定性实现）  [B]（经检索转述）
# 口径：★★ **这三条合起来是一个判据**：一个「开了确定性模式」的测试**未必在查确定性**——
#       如果用的是 `warn_only=True`，非确定性算子**照跑**，只是发了条警告。
#       ⇒ 「我们开了 `use_deterministic_algorithms`」这句话**不足以**支撑
#         「我们的结果是可复现的」；必须同时说 `warn_only` 的取值。

WARP_SIZE = 32
# 来源：CUDA C++ Programming Guide（各 CC 的 "warp size" 恒为 32）  [A]
# 同值位置：20-gpu-architecture/architecture-evolution/code/specs.py:281、
#           45-kernel-lifecycle/operator-design/code/specs.py（**同域内的上一主题**，
#           此处按「跨主题常量复制」纪律重抄，正文注明与该主题一致）。


# ===========================================================================
# 4. UNSOURCED —— 待办清单，不是免责声明
# ===========================================================================
#
# ★ §0.6 硬规则 5：**不得「既保留数字又记 UNSOURCED」**。下表每一项都**没有**在
#   本模块的正文常量里出现。

UNSOURCED = (
    # ── 本清单的口径 ───────────────────────────────────────────────────────
    # ★ 每一项是一个**稳定标识符**（不是散文）。三个程序对本元组做**逐条比对**
    #   （01 篇 §D.7），所以改任何一个标识符都会让 `make run` 失败。

    # 1. **D 级容差表的原始出处页面**。
    #    状态：**找不到**。本域规格 §Chunk3 给了那张表并标 [D]，但**没有给出处**；
    #    本实现者的两轮检索**也没有找到**一张与之逐格吻合的公开页面。
    #    ★ 处置：**保留这张表但只把它当靶子**（§2 的 `SPEC_D_*` 三个常量 + 三个布尔），
    #      正文**不得**把它写成「通行做法」，且**必须**写明它 D 级、无出处页面。
    #      ⇒ 缺口被**标注**而不是被**填平**。
    "spec_d_tolerance_table_source_page",

    # 2. **「D 级表」里 fp64 vs int8 那一行**（规格原文：「以量化 scale 的比例度量，
    #    通常 < 0.5 × scale」）。
    #    状态：**处置了两处，都不写**：① 规格自己写「通常」，无可核实的条件；
    #    ② 检索到的更高等级材料给的是**另一种口径**（vLLM：`int8` 用
    #    `rtol = 0`、绝对容差），**与本行不可比**（P9）。⇒ 本模块**不收录**这一行。
    "spec_d_int8_tolerance_row",

    # 3. **`bfloat16` 那一格的 `atol`**：本实现者两次独立检索读到**两个不同的值**
    #    （`1.6e-2` 与 `1e-3`，两次的 rtol 都是 `1.6e-2`）。
    #    状态：**未解决**。★ 处置：**取其中一个值并把这个不确定写进正文**
    #    （见 `VLLM_DEFAULT_TOLERANCES` 的口径注），**不**声称逐字。
    #    ⚠ 这一项是「两次检索一致」这条规则的**反面实例**：不一致 ⇒ 只能标转述。
    "vllm_bf16_atol_two_readings_disagree",

    # 4. **「共享代码的参考实现漏掉过一个真实缺陷」的公开案例**。
    #    状态：**未找到**。检索没有任何一份可引用的、带复现记录的报告。
    #    ★ 处置：`01` 号程序的 §C 是**本库构造的**（固定种子、可复现），
    #      正文**必须明写它是构造的**，不得暗示它是某个项目的真实故障。
    "real_world_shared_reference_escaped_bug",

    # 5. **各层的「先验概率」与「子假设个数」的实测来源**。
    #    状态：**无来源**，也**测不了**（本机无 GPU，且「一次调试里先怀疑哪层」
    #    不是一个 $ 硬件量）。★ 处置：写成**明写的建模输入**
    #    （`DEBUG_LAYER_PRIORS` / `DEBUG_LAYER_SUBHYPOTHESES`），
    #    并让程序断言**结论会随它们变化**（§B 的边界条件）。
    #    ⚠ **不得**把它们换成「看起来更真实」的常数——那会是 P4（编机制从句）。
    "debug_layer_priors_and_costs_measured",

    # 6. **本域规格 §0.7.G 里那条冷启动数字**（JIT 冷缓存首次编译 `30 s–2 min`、
    #    模型重载冷启动 `2–5 s`）。
    #    状态：**属于 `convergence-and-tuning` 主题**（规格自己写明「留给该主题去核」），
    #    本主题的常量**不含**它。登记在此以免被「顺手」搬进本主题。
    "cold_start_times_belong_to_tuning_topic",
)
# ★ `UNSOURCED` 是**待办清单**：上面每一项都在等一个可定位的一手来源，
#   在它被找到之前，**正文里不得出现对应的数字或案例**。

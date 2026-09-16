"""Inductor / TorchDynamo 的外部事实源：每条带 `# 来源：` 与**可定位的载体**。

本文件只放**从一手来源抄来的**东西（固定版本的源码文件 / 官方文档原文 / PR 号），
不放推导值，不放本仓库自己跑出来的数。找不到可定位载体的项**不写进来**，
记入文件末尾的 UNSOURCED 一节。

★ 版本坐标（本主题强制）
  源码类条目钉在 **torch 2.13.0**（本机 `~/PyCharmMiscProject/.venv`，
  `torch.version.git_version = cf30153c4c131c8164ee7798e5022d810682e2cb`），
  并在 **2.6.0**（`~/miniconda3/envs/deep-prac`）与 **2.7.1**
  （`~/miniconda3/envs/dp-test`）上复核了**逐条标注的那些**默认值。
  这个项目改默认值与改名的频率不低（`cache_size_limit` → `recompile_limit`
  就发生在 2.6.0 与 2.7.1 之间），因此**凡引用常量处一律写明版本**，
  不得写「PyTorch 里 X 就是这样的」。

★ 核查范围（错误模式 P6 的对策）
  下面的每一条都写明**核的是什么载体**——某个文件 + 行号 / 某个文档页 / 某条 PR，
  **不写「全库」「全树」这类大于实际核查范围的措辞**。
  本文件里的「0 命中」一律带出**搜的是哪几个版本、哪棵子树**。
"""

TORCH_MAIN = "2.13.0"
TORCH_GIT = "cf30153c4c131c8164ee7798e5022d810682e2cb"
TORCH_OLD = ("2.6.0", "2.7.1")
SRC_ROOT = "torch/"

# 引用时若不带版本即视为不合格的常量集合
VERSION_BOUND = ("recompile_limit", "accumulated_recompile_limit", "cache_size_limit",
                 "accumulated_cache_size_limit", "skip_code_recursive_on_recompile_limit_hit",
                 "fused_n_spills 的 8 / MIN_ACCEPTED_OCCUPANCY / REGRESSED_OCCUPANCY_RATIO",
                 "mix_order_reduction 的全部子开关")

# ---------------------------------------------------------------------------
# （一）图捕获：帧求值机制
# ---------------------------------------------------------------------------

# 来源：官方文档 https://docs.pytorch.org/docs/main/torch.compiler
#   「TorchDynamo (torch._dynamo) —— the tracer that uses a CPython feature called the
#     Frame Evaluation API to safely capture PyTorch graphs.」
PEP523_DOC_QUOTE = ("uses a CPython feature called the Frame Evaluation API to safely "
                    "capture PyTorch graphs")

# 来源：torch 2.13.0 源码 torch/csrc/dynamo/debug_macros.h:64
#   `_PyInterpreterState_SetEvalFrameFunc(tstate->interp, eval_frame);`
#   —— 这是「装上去的那一个函数指针」的确切名字（PEP 523）。
PEP523_SYMBOL = "_PyInterpreterState_SetEvalFrameFunc"
PEP523_FILE = "torch/csrc/dynamo/debug_macros.h:64"

# 来源：同上文档页；以及 pdb/fdb 与 torch.compile 的兼容性说明（官方文档
#   torch.compiler troubleshooting 一节）。
#   ★ 措辞约束：**不得**把 `sys.settrace` / `sys.monitoring` 说成 Dynamo 的机制。
SYS_MONITORING = ("`sys.monitoring` 不是 Dynamo 的捕获机制，且与 torch.compile 已知"
                  "不兼容：注册 monitoring 回调会引发 graph break，pdb/fdb 在 3.12+ "
                  "不能与 torch.compile 共用。")

# 来源：官方文档 torch.compiler_dynamo_deepdive（«Deep Dive for PyTorch Developers»）
#   graph break 的官方定位是**特性**而不是纯失败。
GRAPH_BREAK_IS_FEATURE = "graph breaks are a feature"

# 来源：官方文档 «Common Graph Breaks» 页
#   官方列举的触发条件（**非穷举表**）
GRAPH_BREAK_TRIGGERS = [
    "数据依赖 if 分支 / 数据依赖控制流（含 .item()、.data_ptr）",
    "大量 Python 内建函数",
    "C 函数",
    "不支持的操作",
    "printing / logging / warnings",
    "本身就会报错的代码",
]

# 来源：torch 2.6.0 / 2.7.1 / 2.13.0 源码 torch/_dynamo/config.py
#   `reorderable_logging_functions: Set[Callable[[Any], None]] = set()`
#   （三版同形；2.6.0 在 config.py:459，2.7.1 在 config.py:502，2.13.0 在 config.py:700）
#   ★ 它的存在说明「有副作用就 break」是错的：print/logging 可以被**重排**以避免 cut。
REORDERABLE_LOGGING_DEFAULT = "set()（默认空集）"

# ---------------------------------------------------------------------------
# （二）guard 与重编译
# ---------------------------------------------------------------------------

# 来源：torch 2.13.0 源码 torch/__init__.py（torch.compile 的 docstring）
#   「Multiple compiled results can be associated with a frame up to
#     ``torch._dynamo.config.recompile_limit``, which defaults to 8; at which point we
#     will fall back to eager. Note that compile caches are per *code object*, not frame」
RECOMPILE_LIMIT_DOC = (
    "Multiple compiled results can be associated with a frame up to "
    "torch._dynamo.config.recompile_limit, which defaults to 8; at which point we will "
    "fall back to eager."
)

# 来源：torch 2.13.0 源码 torch/_dynamo/config.py:121 / :124
#        torch 2.7.1 源码 torch/_dynamo/config.py:53  / :56（同名同值）
#        torch 2.6.0 源码 torch/_dynamo/config.py:44  / :47（**旧名**）
#   ★ 2.6.0 里叫 cache_size_limit / accumulated_cache_size_limit；
#     2.7.1 起改名为 recompile_limit / accumulated_recompile_limit。
#     alias 反向保留：2.13.0 config.py:137-140 把旧名做成 alias。
RECOMPILE_LIMITS = {
    "2.13.0": {"recompile_limit": 8, "accumulated_recompile_limit": 256},
    "2.7.1": {"recompile_limit": 8, "accumulated_recompile_limit": 256},
    "2.6.0": {"cache_size_limit": 8, "accumulated_cache_size_limit": 256},
}

# 来源：torch 2.6.0 / 2.7.1 / 2.13.0 源码 torch/_dynamo/config.py
#   `automatic_dynamic_shapes` 三版默认均为真。
#   2.6.0 config.py:87 / 2.7.1 config.py:110 是字面量 True；
#   2.13.0 config.py:185 改读环境变量 TORCH_DYNAMO_AUTOMATIC_DYNAMIC_SHAPES，
#   默认字符串 "1"（即**仍然默认为真**）。
AUTOMATIC_DYNAMIC_SHAPES_DEFAULT = True

# 来源：torch 2.13.0 源码 torch/__init__.py（torch.compile 的 docstring）
#   「dynamic (bool or None): … When this is True, we will up-front attempt to generate
#     a kernel that is as dynamic as possible to avoid recompilations when sizes change.
#     … By default (None), we automatically detect if dynamism has occurred and compile
#     a more dynamic kernel upon recompile.」
#   ★ 默认值是 **None**，不是 False。
DYNAMIC_ARG_DEFAULT = None

# 来源：官方文档 torch.compiler_dynamic_shapes（动态形状页）
#   `torch.compile(dynamic=True)` 被标注 **"(Not recommended)"**、**"error prone"**、
#   归在 "for testing only"；推荐的做法是**用户标注**（`torch._dynamo.mark_dynamic`）。
DYNAMIC_TRUE_IS_NOT_RECOMMENDED = "Not recommended … error prone … for testing only"

# 来源：torch 2.13.0 源码 torch/_dynamo/convert_frame.py:2007-2039
#   触顶后 `gb_type="Dynamo recompile limit exceeded"`，并把该帧的
#   `frame_exec_strategy` 设成 `FrameExecStrategy(FrameAction.RUN_ONLY, FrameAction.RUN_ONLY)`
#   ——即这一帧改为直接跑 eager。
RECOMPILE_LIMIT_HIT_BEHAVIOR = (
    "超过上限时抛 Unsupported（gb_type=\"Dynamo recompile limit exceeded\"），"
    "并把该帧的执行策略设为 FrameAction.RUN_ONLY（直接跑，不再尝试编译）。"
)

# 来源：torch 2.13.0 源码 torch/_dynamo/graph_break_registry.json（键 GB0039）
GB_RECOMPILE_LIMIT = "GB0039"
GB_RECOMPILE_LIMIT_TYPE = "Dynamo recompile limit exceeded"

# 来源：torch 2.13.0 源码 torch/_dynamo/config.py:126-128
#   `skip_code_recursive_on_recompile_limit_hit: bool = Config(default=True,
#     deprecated=True, deprecation_message="does not do anything")`
#   对照 2.7.1 源码 config.py:59（`= True`，**仍然生效**）。
#   ★ 本仓库**无法判定**「嵌套调用是否也会被跳过」的方向，故正文不写这一条。
SKIP_RECURSIVE_IN_213 = "Config(default=True, deprecated=True, does not do anything)"
SKIP_RECURSIVE_IN_271 = True

# ---------------------------------------------------------------------------
# （三）融合判据
# ---------------------------------------------------------------------------

# 来源：torch 2.6.0 / 2.7.1 / 2.13.0 源码 torch/_inductor/config.py
#   （2.6.0:437/457/468；2.7.1:500/520/531；2.13.0:916/953/964）——三版同名同值
INDUCTOR_DEFAULTS = {
    "aggressive_fusion": False,
    "score_fusion_memory_threshold": 10,
    "max_fusion_size": 64,
}

# 来源：torch 2.13.0 源码 torch/_inductor/config.py:975
#   `max_fusion_unique_io_buffers: int | None = None`（None = 该检查关闭）
#   ★ 2.6.0 与 2.7.1 的 config.py 里**没有这个名字**（两版均 0 命中）。
MAX_FUSION_UNIQUE_IO_BUFFERS = {"2.13.0": None}

# 来源：torch 2.13.0 源码 torch/_inductor/config.py:1723-1725
#   `max_horizontal_fusion_size = int(os.environ.get(
#      "TORCHINDUCTOR_CPP_MAX_HORIZONTAL_FUSION_SIZE", "16"))`
#   ★ 它在 **class cpp** 里（config.py:1666 开始），即**CPU 的 C++ 后端**配置，
#     不是 GPU/Triton 的开关。2.6.0（config.py:781 class cpp / :830）与
#     2.7.1（:781 / :919）同样在 cpp 段内。
MAX_HORIZONTAL_FUSION_SIZE = {"value": 16, "scope": "config.cpp（CPU C++ 后端）"}

# 来源：torch 2.13.0 源码 torch/_inductor/config.py:2048
#   `spill_threshold: int = 32 if torch.version.hip else 16`
SPILL_THRESHOLD = {"hip": 32, "other": 16}

# 来源：torch 2.13.0 源码 torch/_inductor/choices.py:568-636（V.choices.can_fuse）
#   ★ 这是「拒绝融合的机制链」的逐行物证，顺序即源码顺序。
REJECT_REASONS = [
    ("no shared data", "choices.py:610（`shared_data_score == 0` 且不满足绕过条件）"),
    ("no shared data due to indexing mismatch",
     "choices.py:608（score==0 但两个节点确有公共缓冲名）"),
    ("exceeds max fusion", "choices.py:618（节点数 > max_fusion_size）"),
    ("Fusion will increase peak memory", "choices.py:622（can_fusion_increase_peak_memory）"),
    ("fusion_prevent_too_many_reads_and_writes", "choices.py:633"),
    ("score_fusion_memory_threshold", "choices.py:661（水平融合专用）"),
    ("Nodes are too far away. Fusing them may increase peak memory.", "choices.py:664"),
]

# 来源：torch 2.13.0 源码 torch/_inductor/choices.py:584-586
#   `if shared_data_score == 0 and (not config.aggressive_fusion
#                                   or node1.is_reduction() or node2.is_reduction()):`
#   ★ 所以要写准确：**`aggressive_fusion=True` 也救不了带归约的那一侧**。
AGGRESSIVE_FUSION_CONDITION = (
    "score==0 时：aggressive_fusion=False 一律拒；True 时只要有一侧是归约**仍然拒**。")

# 来源：torch 2.13.0 源码 torch/_inductor/scheduler.py:7813-7866（fusable_read_and_write）
#   + scheduler.py:7734-7740（报 "memory deps did not match" 的那一处）
#   ★ 该分支的注释逐字给出反例：
#       MemoryDep("foo", x) != MemoryDep("foo", x + 1)  ·  MemoryDep("foo", x) != StarDep("foo")
VERTICAL_INDEX_RULE = (
    "垂直融合要求消费者的读与生产者的写**是同一个 MemoryDep**（同一缓冲 + 同一索引）；"
    "不同索引就报 \"memory deps did not match\"。"
)
VERTICAL_INDEX_EXAMPLES = ["MemoryDep(\"foo\", x) != MemoryDep(\"foo\", x + 1)",
                           "MemoryDep(\"foo\", x) != StarDep(\"foo\")"]

# 来源：torch 2.13.0 源码 torch/_inductor/scheduler.py:8058-8193（score_fusion_memory）
#   + graph.py:690-710（get_dep_size_hint）
#   `count_bytes=True` 时 `dep_size_hint` 取 `dep.numbytes_hint()`，故**分数以字节计**。
SCORE_UNIT = "字节（count_bytes 默认 True，取 dep.numbytes_hint()）"

# 来源：torch 2.13.0 源码 torch/_inductor/scheduler.py:204-450（class MixOrderReduction）
#   can_fuse 的全部条件，按源码顺序：
#     · `config.triton.mix_order_reduction` 必须开（OSS 默认 "1" → 开）
#     · 兄弟关系：`node1.ancestors & node2.get_operation_names()` 非空即拒（不能是生产者-消费者）
#     · 循环序镜像：`tuple(g1) == tuple(reversed(g2))`，g = (numel, rnumel)
#     · **必须有公共读**（`MixOrderReduction.get_common_read`）
#     · 必须有一个「连续节点」
#     · 非 strict 模式下跳过：`nrow * ncol >= 5*2**20`（scheduler.py:382,387）、
#       `nrow >= 4096`（:408）、`ncol <= 1024*16`（:429）
#     · reduction_hint ∈ {INNER, DEFAULT}（:415-424）
#     · 不是 split reduction（:432）
#     · 其他归约类型（max/min）不支持（:435+）
MIX_ORDER_CONDITIONS = {
    "sibling_only": True,
    "mirrored_loop_order": "(numel, rnumel) 互为镜像",
    "common_read_required": True,
    "size_thres": 5 * 2 ** 20,
    "nrow_min": 4096,
    "ncol_max": 1024 * 16,
}
MIX_ORDER_SIZE_THRES_LOC = "scheduler.py:382 (`size_thres = 5 * 2**20`)"
MIX_ORDER_NROW_LOC = "scheduler.py:408 (`sympy.Ge(nrow, 4096)`)"
MIX_ORDER_NCOL_LOC = "scheduler.py:429 (`statically_known_leq(ncol, 1024 * 16)`)"

# 来源：torch 2.13.0 源码 torch/_inductor/config.py:2120-2143
MIX_ORDER_CONFIG = {
    "mix_order_reduction": "OSS 默认 \"1\"（fbcode 默认 \"0\"）",
    "mix_order_reduction_non_strict_mode": False,
    "mix_order_reduction_max_reads": 10,     # config.py:2137
}

# 来源：torch 2.13.0 源码 torch/_inductor/scheduler.py:3713-3760
#   epilogue 融进模板时的 occupancy / spill 判据：
#     `if fused_n_spills >= 8: return 0, -1`
#     `MIN_ACCEPTED_OCCUPANCY = 4`   `REGRESSED_OCCUPANCY_RATIO = 0.5`
#   ★ 三个名字在 2.6.0 与 2.7.1 的 torch/_inductor/**均 0 命中**（只搜了 _inductor 子树）。
EPILOGUE_SPILL = {"fused_n_spills_ge": 8, "MIN_ACCEPTED_OCCUPANCY": 4,
                  "REGRESSED_OCCUPANCY_RATIO": 0.5}
EPILOGUE_SPILL_LOC = "torch/_inductor/scheduler.py:3713,3745,3746（2.13.0）"

# ---------------------------------------------------------------------------
# （四）后端与 codegen
# ---------------------------------------------------------------------------

# 来源：torch 2.13.0 源码 torch/_inductor/codegen/common.py:407-421
#   `def register_backend_for_device(device, device_scheduling, device_wrapper_codegen, …)`
REGISTER_BACKEND_API = "register_backend_for_device"

# 来源：torch 2.13.0 源码 torch/_inductor/codegen/common.py:519-586（注册表本体）
DEVICE_BACKENDS = [
    ("cpu", "cpu_backends[config.cpu_backend]，默认 cpp（另有 triton/halide/pallas）"),
    ("cuda", "cuda_backends[config.cuda_backend]，默认 triton（CUDACombinedScheduling）"),
    ("tpu", "PallasScheduling"),
    ("xpu", "XPUCombinedScheduling"),
    ("mps", "MetalScheduling"),
    ("mtia", "TritonScheduling"),
]

# 来源：torch 2.13.0 源码 torch/_inductor/config.py:2666 / :2670
BACKEND_DEFAULTS = {"cpu_backend": "cpp", "cuda_backend": "triton"}

# 来源：torch 2.13.0 源码 torch/_inductor/config.py:617-620（含上一行的注释）
#   「# CPP: CPP templates and kernels for CPU.
#     max_autotune_gemm_backends = os.environ.get(
#         "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS", "ATEN,TRITON,CPP").upper()」
#   ★ 同一段注释列出的其它名字是 **NVGEMM / CK / CKTILE**（GPU/AMD 侧）。
#     **`CPP` 不是 GPU 后端**——GPU 侧的 C++ 模板叫 `CUTLASS`。
MAX_AUTOTUNE_GEMM_BACKENDS_DEFAULT = "ATEN,TRITON,CPP"
CPP_COMMENT = "CPP: CPP templates and kernels for CPU."

# 来源：torch 2.13.0 源码 torch/_inductor/config.py:1915-1917
#   「# use triton.autotune for pointwise ops with complex layouts
#     # this should only be disabled for debugging/testing
#     autotune_pointwise = True」
#   ★ 即：**pointwise 默认就搜**；默认不搜的是 GEMM/conv。
AUTOTUNE_POINTWISE_DEFAULT = True
AUTOTUNE_POINTWISE_COMMENT = ("use triton.autotune for pointwise ops with complex "
                              "layouts / this should only be disabled for "
                              "debugging/testing")

# 来源：torch 2.13.0 源码 torch/__init__.py（torch.compile 的 mode 说明）
MODE_OPTIONS_DOC = ('"default" / "reduce-overhead" / "max-autotune" / '
                    '"max-autotune-no-cudagraphs"')

# 来源：torch 2.13.0 源码，逐条给出可定位的文件与行
DEBUG_ENTRY_POINTS = [
    ("TORCH_COMPILE_DEBUG", "torch/_inductor/config.py:2724；"
                            "torch/_logging/_internal.py:1169"),
    ("TORCH_TRACE（trace 输出目录）", "torch/_logging/_internal.py:54"),
    ("IR 转储 ir_pre_fusion / ir_post_fusion", "torch/_logging/_internal.py:247-248, 422-425"),
    ("INDUCTOR_PROVENANCE", "torch/_inductor/config.py:2820-2823"),
    ("torch._dynamo.utils.compile_times()", "torch/_dynamo/utils.py:874-884"),
]
# ★ 这一组名字在源码树里能定位到，但本仓库**没有**跑过它们，
#   因此正文只写「这些是官方的读/定位入口」，不写它们的输出长什么样。

# 来源：torch 2.13.0 源码 torch/_inductor/config.py:274（`epilogue_fusion = True`）
EPILOGUE_FUSION_DEFAULT = True

# ---------------------------------------------------------------------------
# （五）引用值（外部来源的实测，本仓库不复核其测量口径）
# ---------------------------------------------------------------------------

# 来源：pytorch/pytorch PR #179494
#   https://github.com/pytorch/pytorch/pull/179494
#   过度融合的 mix-order reduction 在 **H100 上 +6.3 ms/step**；
#   profiling 显示 11 个读因 register spill 回退；
#   修法是在调度期统计唯一读缓冲数并设上限（对应 mix_order_reduction_max_reads = 10）。
PR_179494 = {"pr": 179494, "regression": "+6.3 ms/step（H100）",
             "cause": "11 个读因 register spill 回退",
             "fix": "调度期统计唯一读缓冲数并设上限（mix_order_reduction_max_reads = 10）"}

# 来源：pytorch/pytorch issue #180811
#   AOTInductor + max_autotune 会对小 GEMM **非确定性地**选 cuBLAS 而非融合的
#   Triton 模板，从而打破融合造成回退。
ISSUE_180811 = {"issue": 180811,
                "behavior": "小 GEMM 上非确定性地选 cuBLAS 而非融合 Triton 模板"}

# ---------------------------------------------------------------------------
# UNSOURCED：查不到可定位一手来源，**正文不使用**
# ---------------------------------------------------------------------------

UNSOURCED = {
    "has_strict_reduction()":
        "该函数名在 2.6.0 / 2.7.1 / 2.13.0 的 torch/_inductor/ 子树**均 0 命中**"
        "（另见 UNSOURCED[strict reduction 字符串]）。正文不得出现这个函数名。",
    "「template fusion does not preserve strict reduction ordering」字符串":
        "在 2.6.0 / 2.7.1 / 2.13.0 的 torch/_inductor/ 子树里 "
        "`strict reduction` 与 `strict_reduction` **均 0 命中**。"
        "能查到的相近物只有 2.13.0 的 `mix_order_reduction_non_strict_mode`"
        "（config.py:2133）与 scheduler.py:378 的读取处，那是**另一个开关**。"
        "正文只说「归约↔归约不是通用能力，只有 mix-order 特例放行」，不引这个字符串。",
    "「生成的 kernel 在编译缓存里会被覆盖」":
        "无来源。官方支持的是「**读**生成代码 + 定位」（TORCH_COMPILE_DEBUG / "
        "TORCH_LOGS=output_code / provenance tracking），"
        "没有找到任何「手改生成的 kernel 会被缓存覆盖」的一手表述。**正文不写这条机制。**",
    "「嵌套调用也会被跳过」":
        "控制它的开关在 2.13.0 已标 deprecated（`does not do anything`），"
        "而 2.7.1 上它还是生效的 → **方向无法判定**。正文不写。",
    "「reduction 可以用 C++ 后端更快」":
        "无一手来源，且是概念混淆：`CPP` 后端是 **CPU 的 C++ 模板**"
        "（见 CPP_COMMENT），GPU 侧 C++ 模板是 CUTLASS。正文不写。",
}

# 「未取到一手来源」的检索范围（错误模式 P6 的对策：写清核查范围）
GREP_SCOPE = ("torch 2.6.0 / 2.7.1 / 2.13.0 三份安装的 torch/_inductor/ 与 "
              "torch/_dynamo/ 两棵子树，按文件逐行文本匹配；"
              "**没有**核 C++ 源与算子库侧。")

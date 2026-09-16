"""TVM 的外部事实源：每条常量带一行 `# 来源：` 与**可定位的载体**。

本文件只放**从一手来源抄来的**东西（固定 tag 的源码文件 / 官方文档原文 / 论文 PDF），
不放推导值，不放本仓库自己跑出来的数。找不到可定位载体的项**不写进来**，记入文件末尾的
UNSOURCED 一节。

★ 版本坐标（本主题强制，见计划 §0.7.C）
  全部 API 结论钉在 **Apache TVM tag `v0.26.0`**（release 2026-08-11）。
  这个项目在两年内删掉了 `te` 整层调度 API、`tvm.tir` 整个模块、AutoTVM 全套，
  并改名了 `T.block` -> `T.sblock`、`get_block` -> `get_sblock`。
  **任何不带版本的「TVM 里 X 是这样的」在本主题一律视为不合格**——凡涉及 API 的条目
  都带 `TVM_TAG`，并且能指出核的是哪个文件。

★ 核查范围（错误模式 P6 的对策）
  下面的每一条都写明**核的是什么载体**——某个文件 / 某个文档页 / 某篇论文，
  **不写「全库」「全树」这类大于实际核查范围的措辞**。
  少数结论只有「控制者基线（两轮独立研究）」而**没有落到具体文件路径**的，
  条目里逐条标了 `CONTROLLER_BASELINE`，正文引用时会降级表述。
"""

TVM_TAG = "v0.26.0"
TVM_TAG_DATE = "2026-08-11"
TVM_REPO = "https://github.com/apache/tvm"


def src(path):
    """把 tag v0.26.0 上的仓库内路径拼成可打开的 URL。"""
    return "%s/blob/%s/%s" % (TVM_REPO, TVM_TAG, path)


# ---------------------------------------------------------------------------
# （一）被删除 / 被改名的 API —— 网上绝大多数 TVM 教程教的都是这些
# ---------------------------------------------------------------------------

# 来源：https://github.com/apache/tvm/blob/v0.26.0/python/tvm/te/__init__.py
#   `te.create_schedule` / `te.Schedule` / `Stage` 均无定义（该文件**没有**模块级
#   `__getattr__` 兜底，所以是 AttributeError 而不是 ImportError）。
TE_SCHEDULE_GONE = "AttributeError: module 'tvm.te' has no attribute 'create_schedule'"

# 来源：https://github.com/apache/tvm/blob/v0.26.0/python/tvm/te/__init__.py
# ★ 这条是本主题最反直觉的一条：**删了一半 API，却没有任何 deprecation 提示**。
#   `import tvm.te` 静默成功，`te.placeholder` / `te.compute` 也还能用。
TE_IMPORT_SILENT = (
    "`import tvm.te` 静默成功，无 deprecation warning；"
    "只有用到被删的名字时才在**调用处**报 AttributeError。"
)

# 来源：https://github.com/apache/tvm/blob/v0.26.0/python/tvm/script/parser/ 的方言注册表
#   （注册表里的名字只有 tirx / relax / ir；`tir` 不在其中）
SCRIPT_TIR_ALIAS_ERROR = "ImportError（方言 `tir` 未注册；正确的是 `tirx`）"
SCRIPT_CORRECT_IMPORTS = ("from tvm.script import ir as I", "from tvm.script import tirx as T")

# 来源：https://github.com/apache/tvm/blob/v0.26.0/python/tvm/script/tirx/ （块构造子的名字）
SBLOCK_RENAME = {"T.block": "T.sblock", "T.block_attr": "T.sblock_attr",
                 "sch.get_block": "sch.get_sblock"}

# 来源：https://github.com/apache/tvm/blob/v0.26.0/python/tvm/s_tir/meta_schedule/__init__.py
#       https://github.com/apache/tvm/blob/v0.26.0/python/tvm/s_tir/dlight/__init__.py
#   v0.24.0 起 meta_schedule / dlight 从 `tvm.*` 移到 `tvm.s_tir.*`。
META_SCHEDULE_IMPORT = "from tvm.s_tir import meta_schedule as ms"
DLIGHT_IMPORT = "from tvm.s_tir import dlight"

# 来源：https://github.com/apache/tvm/blob/v0.26.0/python/tvm/autotvm/ （目录不存在）
AUTOTVM_REMOVED = "from tvm import autotvm -> ModuleNotFoundError（AutoTVM 整体删除）"

# 来源：https://github.com/apache/tvm/blob/v0.26.0/python/tvm/te/__init__.py
#   te 的幸存名单：placeholder / compute / reduce_axis / sum / create_prim_func。
# ★ 核查范围：只核了**这些名字仍在 `te/__init__.py` 里**，
#   **没有**逐个核 te 下每个名字的存亡，不得写「te 只剩这五个」。
TE_SURVIVORS = ("placeholder", "compute", "reduce_axis", "sum", "create_prim_func")

# 来源：https://github.com/apache/tvm/blob/v0.26.0/python/tvm/driver/__init__.py
#   该文件全文只有一行导出：build, compile。
#   `tvm.build(mod, target)` 仍可用，但内部转发到 tvm.tirx.build 并抛 DeprecationWarning。
DRIVER_EXPORTS = ("build", "compile")

# 来源：控制者基线（两轮独立研究核实）——**未取到具体文件路径**。
#   §0.7.C 已把「`tvm.lower` 不存在」这条**撤下**：查到的证据偏向相反方向
#   （docs 提交里「removed the `tvm.build` entry while retaining the
#   `autofunction:: tvm.lower` directive」）。**本文不断言它不存在，也不断言它存在。**
CONTROLLER_BASELINE = "控制者基线（两轮研究），未落到具体文件路径"

TVM_LOWER_STATUS = "存疑：证据偏向「仍然存在」，本文不表态"

# ---------------------------------------------------------------------------
# （二）s_tir.Schedule 的确切 API（两态，不是三态）
# ---------------------------------------------------------------------------

# 来源：https://github.com/apache/tvm/blob/v0.26.0/python/tvm/s_tir/schedule/schedule.py
SCHEDULE_API = {
    "compute_at": "(block, loop, preserve_unit_loops=False, index=-1)",
    "reverse_compute_at": "(block, loop, preserve_unit_loops=False, index=-1)",
    "compute_inline": "(block)",
    "reverse_compute_inline": "(block)",
}

# 来源：同上文件——**API 目录里没有 `compute_root`**。
# ★ 核查范围（错误模式 P6）：核的是 `s_tir.Schedule` 的 **API 目录**；
#   计划里的「v0.26.0 全树 0 命中」这一表述**审查无法复现**，
#   所以正文一律降级为「API 不含」。
SCHEDULE_HAS_NO_COMPUTE_ROOT = True

# ★ 本主题最重要的对照：TE 时代的三态 vs s_tir 的两态。
# | TE（已删除）        | s_tir（v0.26.0）      |
# | compute_root        | **没有这个名字**       |
# | compute_at          | compute_at（物化到指定循环内） |
# | compute_inline      | compute_inline（不物化）|
#
# ★ 「没被移走的 block 本来就处于 root 位置」——**这是推断，不是官方明文**。
#   推断依据：API 只有上面两条「移走」的原语，没有把 block 放回 root 的名字。
#   控制者基线里给了一条源码行 `AttachType attach_type{kGroupRoot}` 作为佐证，
#   但**未取到文件路径**，所以本文把它标成推断 + 佐证，而不写成引文。
DEFAULT_PLACEMENT_IS_ROOT = "inference"
DEFAULT_PLACEMENT_EVIDENCE = "AttachType attach_type{kGroupRoot}（" + CONTROLLER_BASELINE + "）"

# 来源：同上文件，compute_at / reverse_compute_at 的 docstring
COMPUTE_AT_PRECONDITION = "block 的**所有 consumer** 都必须在目标 loop 下"
REVERSE_COMPUTE_AT_PRECONDITION = "block 的**所有 producer** 都必须在目标 loop 下"

# 来源：同上文件，Schedule 构造参数的注释（原文引文，逐字）
#   「The default schedule checks are too strict and might prevent us performing some valid
#    schedules. ... It's user duty to guarantee schedule correctness if enable_check is set
#    to False.」
ENABLE_CHECK_DEFAULT = True
ENABLE_CHECK_QUOTE = (
    "The default schedule checks are too strict and might prevent us performing some valid "
    "schedules. ... It's user duty to guarantee schedule correctness if enable_check is set "
    "to False."
)

# 来源：同上文件 / 官方 TIR 教程——调度是**过程式**的，不幂等
PROCEDURAL_QUOTE = (
    "the transformation process is procedural; thus, inadvertent execution of the block "
    "twice will yield an error stating the non-existence of variable `j`."
)

# 来源：同上文件，各原语的 docstring（逐字）
PRIMITIVE_WARNINGS = {
    "split(predicate)": "may result in incorrect code generation if not used carefully",
    "rolling_buffer": ("The region_cover property of the consumer block of the target buffer "
                       "will become false"),
    "unsafe_hide_buffer_access": "This schedule primitive is unsafe, and may fail dependency analysis",
    "annotate_buffer_access": "incorrect annotations may lead to incorrect code generation",
}

# 来源：同上文件——存在的核心原语（研究代理从源码正则抓取后剔除辅助方法）。
#   ★ 核查范围：这是**目录级**的抓取结果，不是逐条签名核对。
SCHEDULE_PRIMITIVES = {
    "取引用": ["get_sblock", "get_loops", "get_child_blocks", "get_producers",
             "get_consumers", "get_output_blocks"],
    "循环结构": ["merge", "fuse", "split", "loop_partition", "reorder",
             "reorder_block_iter_var", "add_unit_loop"],
    "循环标注": ["parallel", "vectorize", "bind", "unroll"],
    "缓存": ["cache_read", "cache_write", "reindex_cache_read", "reindex_cache_write",
           "cache_inplace", "cache_index", "reindex"],
    "块位置": ["read_at", "write_at", "compute_at", "reverse_compute_at",
             "compute_inline", "reverse_compute_inline"],
    "归约": ["fuse_reduction_epilogue", "decompose_reduction", "rfactor"],
    "存储/作用域": ["storage_align", "set_scope", "unsafe_set_dtype"],
    "块化/张量化": ["blockize", "tensorize"],
    "布局": ["transform_layout", "transform_block_layout"],
}
SCHEDULE_ABSENT_PRIMITIVES = ["compute_root", "get_block", "tile"]

# ---------------------------------------------------------------------------
# （三）两条并列的 TIR 轨道与 pipeline
# ---------------------------------------------------------------------------

# 来源：https://github.com/apache/tvm/blob/v0.26.0/python/tvm/driver/__init__.py
#       https://github.com/apache/tvm/blob/v0.26.0/python/tvm/s_tir/pipeline.py
#   get_tir_pipeline 的函数体第一句是 `if name == "default": name = "s_tir"`。
#   ★ 这一句是**已核实**的（计划 §0.7.C 的审查列表里明确标了 ✓）。
PIPELINE_DEFAULT_RULE = 'get_tir_pipeline("default") 的第一句：if name == "default": name = "s_tir"'

# 来源：https://github.com/apache/tvm/blob/v0.26.0/python/tvm/driver/__init__.py
COMPILE_SIGNATURE = ("tvm.compile(mod, target=..., *, relax_pipeline=\"default\", "
                     "tir_pipeline=\"default\") -> Executable")

# 来源：控制者基线（两轮研究）——**未取到逐个数 pass 的核对**。
#   「s_tir 那条约 40 个 pass」与「tirx 那条 18–19 个」属于审查标记为
#   **证据强度不足**的一组（§0.7.C 的 S6/S8 注），且官方文档写 18、
#   `tirx/compilation_pipeline.py` 里数出 19，差异**无官方解释**。
#   本文只依赖已核实的那一句（PIPELINE_DEFAULT_RULE），pass 计数仅作背景，
#   并标成 CONTROLLER_BASELINE。
PIPELINE_PASS_COUNTS = {"s_tir": "约 40", "tirx": "18–19（官方文档 18 / 代码 19，差异无官方解释）"}
PIPELINE_PASS_COUNTS_STATUS = CONTROLLER_BASELINE

TRACKS = {
    "s_tir": {
        "form": "@T.prim_func(s_tir=True) + T.sblock",
        "schedule": "tvm.s_tir.Schedule（DLight / MetaSchedule 作用在这条轨道上）",
        "pipeline": 'tir_pipeline="default"',
    },
    "tirx": {
        "form": "@T.prim_func（s_tir=False，**默认值**）+ T.device_entry() + "
                "T.cta_id() / T.thread_id()",
        "schedule": "无——并行映射手写",
        "pipeline": 'tir_pipeline="tirx"（**必须显式指定**）',
    },
}

# ---------------------------------------------------------------------------
# （四）tile 原语（TIRx）
# ---------------------------------------------------------------------------

# 来源：https://github.com/apache/tvm/blob/v0.26.0/docs/tirx/tile_primitives.rst
#   官方逐字：「The C++ registry currently defines 31 operation names」——**是 31**。
TILE_OP_COUNT = 31
TILE_OP_COUNT_QUOTE = "The C++ registry currently defines 31 operation names"

# 来源：同上页面——命名空间**编码了协作 scope**，且**多一层 `tile`**。
# ★ 写成 `Tx.<name>` 会找不到；正确的最小形态是 `Tx.tile.<name>`。
TILE_OP_NAMESPACES = {
    "Tx.tile.<name>": "thread 级（裸命名空间即 thread）",
    "Tx.tile.thread.*": "thread 级（显式写法）",
    "Tx.tile.warp.*": "warp 级",
    "Tx.tile.wg.*": "warpgroup 级（`Tx.tile.warpgroup.*` 的别名）",
    "Tx.tile.cta.*": "CTA 级",
    "Tx.tile.cluster.*": "cluster 级",
}

# 来源：同上页面——官方给的**分组**与组内**示例名**。
# ★ 核查范围：本文只拿到「计数 31」「命名空间列表」「这些组名与组内示例名」，
#   **31 个名字的完整清单未取到**——所以本仓库的迷你 registry 是它的一个**子集**，
#   不得声称与官方清单一一对应（见 UNSOURCED）。
TILE_OP_GROUPS = {
    "data movement": ["copy", "copy_async", "permute_layout"],
    "matrix multiply": ["gemm", "gemm_async"],
    "fill/memset/zero": [],
    "cast + elementwise": [],
    "reductions": ["sum", "max", "min"],
    "fused/compose": [],
}

# ---------------------------------------------------------------------------
# （五）自动调度：规则式与搜索式两条并列的路
# ---------------------------------------------------------------------------

# 来源：https://github.com/apache/tvm/blob/v0.26.0/docs/deep_dive/tensor_ir/tutorials/
#                                                      dlight_gpu_scheduling.py
METASCHEDULE_VS_DLIGHT_QUOTE = (
    "MetaSchedule: explores a search space to find the best schedule. "
    "High quality, but compilation takes minutes to hours.\n"
    "DLight: applies pre-defined scheduling rules deterministically. "
    "No tuning required, compilation completes in seconds."
)

# 来源：控制者基线（两轮研究，来自 meta_schedule 的默认配置）——未落到具体文件路径
METASCHEDULE_COMPONENTS = {
    "空间生成": "PostOrderApply",
    "搜索": "EvolutionarySearch",
    "成本模型": "XGBModel",
    "数据库": "JSONDatabase",
    "调度器": "GradientBasedScheduler",
}
METASCHEDULE_COMPONENTS_STATUS = CONTROLLER_BASELINE

# 来源：同上——DLight 的规则库可见于 `gpu/` 下，按算子名分文件
DLIGHT_RULES = ["matmul", "gemv", "low_batch_gemv", "general_reduction",
                "reduction", "rmsnorm", "transpose", "fallback"]

# 来源：https://github.com/apache/tvm/blob/v0.26.0/python/tvm/ansor/ —— **该目录不存在**。
ANSOR_IS_PAPER = "Ansor 从来不是 TVM 里的一个 Python 模块；它是论文"

# ★ 「MetaSchedule 是 Ansor 的实现」——组件对应关系的**推断**，TVM 源码里没有明文。
#   能查到的只有：MetaSchedule 由独立 RFC（AutoTensorIR）提出，第三方文献用
#   「absorbs / unifies」这类**等价**表述，**不是「改名」也不是「工程化」**。
#   本文必须写成推断并注明。
METASCHEDULE_IS_ANSOR_STATUS = "inference"

# ---------------------------------------------------------------------------
# （六）Ansor 论文（数字的**正文位置在 60-compilers/auto-tuning/03**，本主题只回链）
# ---------------------------------------------------------------------------

# 来源：https://www.usenix.org/system/files/osdi20-zheng.pdf
#   《Ansor: Generating High-Performance Tensor Programs for Deep Learning》，
#   14th USENIX OSDI，2020-11-04/06。
ANSOR_PAPER = "osdi20-zheng"
ANSOR_PAPER_URL = "https://www.usenix.org/system/files/osdi20-zheng.pdf"

# 来源：同上 PDF §3（hierarchical search space 的定义句，逐字）
ANSOR_SEARCH_SPACE_QUOTE = (
    "we define a hierarchical search space with two levels: sketch and annotation... "
    "At the top level, we generate sketches by recursively applying a few derivation "
    "rules. At the bottom level, we randomly annotate these sketches to get complete programs."
)

# 来源：同上 PDF Table 1——**sketch 阶段已经包含 fusion 决策**（逐字规则）。
#   ★ 这是「物化/重算的决策被放进搜索空间骨架层」的一手依据。
ANSOR_FUSION_RULE = "HasDataReuse ∧ HasFusibleConsumer -> FuseConsumer(MultiLevelTiling(S,i), i)"

# 来源：同上 PDF §7.5——代价模型的规模与损失口径
ANSOR_COST_MODEL = {
    "类型": "gradient boosting decision tree（引文 [9] = XGBoost, KDD 2016）",
    "训练": "we train a new model every time instead of doing incremental updates",
    "损失": "weighted squared error，权重直接用 throughput y",
    "特征": "arithmetic features + memory access features，对每条 innermost non-loop statement 抽取",
}

# ---------------------------------------------------------------------------
# （七）TileLang 与 TVM 的关系
# ---------------------------------------------------------------------------

# 来源：https://github.com/tile-ai/tilelang 的 `.gitmodules`
#   `3rdparty/tvm` -> `github.com/TileLang/tvm`；仓库为 **MIT 协议**、**Tile-AI 维护**。
#   **依赖方向是 TileLang -> TVM**（不是反向）。
TILELANG_GITMODULES = "3rdparty/tvm -> github.com/TileLang/tvm"
TILELANG_LICENSE = "MIT"
TILELANG_MAINTAINER = "Tile-AI"
TILELANG_REPO = "https://github.com/tile-ai/tilelang"

# 来源：https://arxiv.org/abs/2504.17577 （2025-04-24）
TILELANG_PAPER = "arXiv:2504.17577"

# ---------------------------------------------------------------------------
# 口号：本主题那句「默认全物化」的完整表述
# ---------------------------------------------------------------------------

DEFAULT_MATERIALIZE_HEADLINE = (
    "TVM 的默认调度是**把每个中间结果都物化在根作用域**，不是把它们 inline 掉。"
    "（结论来源：见 DEFAULT_PLACEMENT_IS_ROOT —— 证据含一条推断，本文写明。）"
)

# ---------------------------------------------------------------------------
# UNSOURCED（查不到可定位载体，**正文不使用**）
# ---------------------------------------------------------------------------
UNSOURCED = """
· 「tvm.lower 不存在」——**已撤下**。查到的证据偏向相反方向（见 TVM_LOWER_STATUS）。
  正文既不写它存在，也不写它不存在。
· 「v0.26.0 全树 grep `compute_root` 0 命中」——**复现不了**。
  能核到的只有 `s_tir.Schedule` 的 API 目录不含它。正文降级为「API 不含」。
· 「te 删于 v0.20.0 / tvm.tir 拆于 v0.24.0 / AutoTVM 删于 v0.20.0」这几个**版本归属**
  ——删除动作本身有一手证据，**版本号没有**。正文写「v0.26.0 上已不存在」，
  不写「在 X.Y.Z 版本被删」。
· 「s_tir 那条约 40 个 pass / tirx 那条 18–19 个」——控制者基线给出，**本文未逐个核对**，
  且官方文档与代码的 18/19 差异**无官方解释**。正文只写已核实的那一句
  （默认 pipeline 解析到 s_tir），计数只作背景并标明出处。
· 「两条 pipeline 读不同命名空间的 PassContext 键（tirx.disable_vectorize vs
  tir.disable_vectorize）」——只拿到**一半证据**（tirx 侧证实、s_tir 侧文件未取到）。
  计划 §0.7.C 明令：**不得作为失败模式写进文档**；本文把它记在正文的
  「可选观察」框里，并写明证据不完整、本仓库未复现。
· `tile-ai/tilelang` PR **#2216**——两次定向检索均 0 命中。**不写**。
  可核的替代表述见 TILELANG_* 三条。
· 「TVM 源码里 tilelang 只出现 1 次，且只是一句测试注释」——**复现不了**。不写。
· tile 原语 **31 个名字的完整清单**——未取到。本仓库的 mini registry 是**子集**，
  不得声称与官方一一对应。
· 「MetaSchedule 是 Ansor 的实现」——TVM 源码无明文，是**推断**（见
  METASCHEDULE_IS_ANSOR_STATUS）。正文必须标成推断。
· 「compute_root 原位置的官方替代写法」——**无明文**。正文只写「未移走的 block
  处于 root 位置」并标明是推断。
· Ansor 的 Table 3 / §7.3 / §7.5 数字——**它们在本仓库的唯一正文位置是
  `60-compilers/auto-tuning/03-search-algorithms` §5**。本主题（tvm）只回链、不复制
  （CONTRIBUTING §7.1：一个知识点只有一处正文）。
· TileLang vs Triton 的性能倍数（1.36× / 1.41× 等）——未一手核到。不写。
· 「TIR vs lowered TIR」这组对立术语——无一手来源。不使用。
· Ansor 的代价模型「最多约 10 轮」之类——无来源。不写。
"""

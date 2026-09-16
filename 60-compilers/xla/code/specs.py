"""XLA 的外部事实源：每条常量带一行 `# 来源：` 与**可打开的 URL**。

本文件只放**从一手来源抄来的**东西（源码文件 / 官方文档原文），不放推导值，
不放本仓库自己跑出来的数。找不到一手来源的项**不写进来**，记入 UNSOURCED。

★ 核查范围（错误模式 P6 的对策）：下面的条目逐条写明**核的是什么载体**
  ——哪个源码文件 / 哪个文档页 / 哪个 commit，**不写「全库」「全树」这类
  大于实际核查范围的措辞**。

★ 版本坐标：源码类条目取用于 openxla/xla 的 `main` 分支，取用日期 **2026-09-16**。
  这是**会漂移的坐标**——引用时须写明取用日期，不得写成「XLA 里 X 就是这样的」。
  本仓库所在环境**没有 XLA 二进制、没有 jax、没有 tensorflow**，
  下面的引文全部是**读源码/文档得到的**，不是本机跑出来的。

★ 本文件的任何条目**都不得作为编号程序的断言对象**（§0.3 规矩 4）：
  程序断言的是本仓库的迷你实现，真实 XLA 的行为只出现在文档与这里。

UNSOURCED（查不到一手来源，**正文不使用**）：
  · 「融合因此产生本可避免的 transpose」——**无一手文档/论文支持**。
    可引的事实只到「融合体内部不携带 layout」（见 TF_COMMIT_358FD36D 的提交信息）。
    最接近的公开材料是 XLA 仓库的 GitHub discussion，**既不是文档也不是论文**。
    正文若写这条因果，**必须标为「推论」**。
  · 「GPU 上的 alternate memory 具体对应哪种内存」——`xla/service/gpu/` 下
    没有任何一手文件把 `MemorySpaceAssignment` 的 default/alternate 与 GPU 的
    物理内存层级对应起来。GPU 侧跑的是另一套（见 GPU_MSA_COLOR_* 与
    GPU_PIPELINE_HOST_OFFLOADING）。**不得断言。**
  · 「HLO 本身没有兼容性保证」——只有**对照证据**（StableHLO 明文给出 5 年/2 年窗口），
    没有找到 HLO 侧的官方否认声明。正文措辞留余地。
"""

# ---------------------------------------------------------------------------
# 一、HLO 的 fusion kind（文档 02）
# ---------------------------------------------------------------------------

# 来源：https://github.com/openxla/xla/blob/main/xla/hlo/ir/hlo_instruction.h
#       （取用 2026-09-16；`enum class FusionKind` 及其上方注释）
# 逐字，四个成员：
HLO_FUSION_KIND_ENUM = """enum class FusionKind {
    kLoop,
    kInput,
    kOutput,
    kCustom,
  };"""

# 来源：同上，`kInput` 条目逐字
FUSION_KIND_KINPUT = "kInput: The primary node is the root of the fused instruction."

# 来源：同上，`kOutput` 条目逐字
FUSION_KIND_KOUTPUT = (
    "kOutput: The primary node is not the root of the fused instruction. "
    "This fusion kind requires that one operand buffer of the fusion instruction "
    "be able to alias the output buffer. This constraint is usually enough to let "
    "backends find the primary node unambiguously."
)

# 来源：同上，`kLoop` 条目逐字
FUSION_KIND_KLOOP = (
    "kLoop: The primary node is the root of the fused computation, but, unlike in "
    "input fusion, we prescribe a specific implementation for codegen. Rather than "
    "generating code that looks like the code we'd emit for an unfused version of "
    "the primary/root node, we emit code that generates one element of the root at "
    "a time."
)

# 来源：同上，`kCustom` 条目逐字
FUSION_KIND_KCUSTOM = (
    "kCustom: Custom category for backend-specific fusions that don't fit into the "
    "above patterns."
)

# 来源：同上，文件里的 TODO(b/79869434) 逐字。**这是「multi-output fusion 不是
#       一个 fusion kind」的一手依据**，也是「primary node 不是 root」的一手依据。
FUSION_KIND_TODO = (
    "TODO(b/79869434): This categorization scheme is not great. For one thing, input "
    "and loop fusion are basically the same thing: There is no reason for the HLO to "
    "encode backend-specific decisions about how e.g. a reduce that's the root of a "
    "fusion should be lowered. In addition, this scheme as written doesn't work for "
    "multi-output fusion, where the primary node is never actually the root (which is "
    "a kTuple instruction that gathers the multiple outputs of the fusion)."
)

# 来源：同上，`kInput`/`kOutput` 语义的总起句逐字
FUSION_KIND_ROLE = (
    "An HloInstruction's FusionKind helps us find the kFusion instruction's primary "
    "node, and can also affect how we generate code in step (2)."
)

# ---------------------------------------------------------------------------
# 二、预算类判据（文档 02）
# ---------------------------------------------------------------------------

# 来源：https://github.com/openxla/xla/blob/main/xla/service/gpu/gpu_fusible.h
#       逐字：inline constexpr int64_t MaxOperandsAndOutputsPerFusion() { return 96; }
MAX_OPERANDS_AND_OUTPUTS_PER_FUSION = 96

# 来源：同上，`FusionFitsInParameterLimit` 的注释逐字（「最多能传给 kernel 的参数个数」）
FUSION_FITS_IN_PARAMETER_LIMIT = (
    "Determines whether the combination of `instr1` and `instr2` into a (possibly "
    "multi-output) fusion fits within the maximum number of parameters that can be "
    "passed to a kernel."
)

# 来源：同上，`FusionFitsInBudget` 的注释逐字（operands/outputs 数量**以及** shared memory）
FUSION_FITS_IN_BUDGET = (
    "Determines whether the combination of `instr1` and `instr2` into a (possibly "
    "multi-output) fusion fits within a \"budget\" -- i.e., does have more operands "
    "and outputs than is allowed or occupy too much shared memory."
)

# ---------------------------------------------------------------------------
# 三、multi-output fusion（文档 02）
# ---------------------------------------------------------------------------

# 来源：https://github.com/openxla/xla/blob/main/xla/backends/gpu/transforms/multi_output_fusion.h
#       （注意路径已从 xla/service/gpu/ 迁移到 xla/backends/gpu/transforms/）
# 两类与两条硬约束，逐字：
MULTI_OUTPUT_FUSION_FLAVORS = (
    "1) Fusion of sibling ops reduces memory bandwidth requirements, because common "
    "input parameters have to be read only once. "
    "2) Fusion of producer-consumer ops reduces memory bandwidth requirements by "
    "saving one read from memory."
)

# 来源：同上，两条硬约束逐字
MULTI_OUTPUT_FUSION_HARD_CONSTRAINTS = (
    "* Sibling fusion (1) requires at least one op to be a kFusion. "
    "* Sibling fusion (1) does not fuse kInput fusions with kLoop fusions, i.e. the "
    "fusion kinds must match."
)

# 来源：同上，root 的形态逐字
MULTI_OUTPUT_FUSION_ROOT = (
    "Multi-output fusion ops have a tuple op at their root containing multiple "
    "elements as outputs. GetTupleElement ops (depicted as gte_* above) are inserted "
    "to extract tuple elements for consumers."
)

# ---------------------------------------------------------------------------
# 四、GPU 侧的融合判据链（文档 02）——「预算 + 合法性 + 代价模型」三层的一手依据
# ---------------------------------------------------------------------------

# 来源：https://github.com/openxla/xla/blob/main/xla/backends/gpu/transforms/multi_output_fusion.cc
#       函数 `ProducerCandidateIsFusible`。**这段代码本身就展示了两层**：
#       先 `FusionFitsInBudget`（预算），再走两个代价模型否决（cost model）。
#       逐字的拒绝理由字符串：
GPU_FUSION_REJECT_TOO_LARGE_IR = "will generate too large IR"
GPU_FUSION_REJECT_SLOWER_IF_FUSED = "will execute slower if fused"
GPU_FUSION_REJECT_MULTIPLE_DUS = "can't fuse multiple DUSs"

# 来源：同上，`ProducerCandidateIsFusible` 的两处逐字调用点
GPU_FUSION_LAYERED_EVIDENCE = (
    "RETURN_IF_NOT_FUSIBLE(FusionFitsInBudget(producer, consumer, device_info, "
    "/*is_consumer_producer_fusion=*/false, fusion_info_cache));"
    "  ...  "
    "if (cost_analysis->ProducerConsumerMergedTooLarge(producer, consumer)) "
    "{ return FusionDecision::Forbid(\"will generate too large IR\"); }"
    "  ...  "
    "GpuPerformanceModel::RunTimes t = "
    "gpu_performance_model.EstimateRunTimesForMultiOutputFusion(...); "
    "if (t.time_fused > t.time_unfused) "
    "{ return FusionDecision::Forbid(\"will execute slower if fused\"); }"
)

# 来源：同上（同文件），另一处代价模型驱动的拒绝——生产者-消费者的候选筛选
GPU_FUSION_REJECT_SCAN = "multi-output fusion for scan is not supported"
GPU_FUSION_REJECT_NOT_MO_ROOT = "consumer not eligible as multi-output fusion root."

# 来源：https://github.com/openxla/xla/blob/main/xla/service/gpu/hlo_fusion_analysis.h
#       （取用 2026-09-16）`enum class EmitterFusionKind` 逐字。
#       **这是 GPU emitter 侧的另一套枚举，与 `HloInstruction::FusionKind` 不是一回事。**
#       9 个成员：
EMITTER_FUSION_KIND_ENUM = """enum class EmitterFusionKind {
    kLoop,
    kCustomFusion,
    kTriton,
    kReduction,
    kTranspose,
    kConcatenate,
    kScatter,
    kCuDnn,
    kSort,
  };"""

# ---------------------------------------------------------------------------
# 五、layout assignment 与 fusion 的交互（文档 03）
# ---------------------------------------------------------------------------

# 来源：https://github.com/openxla/xla/blob/main/xla/service/layout_assignment.h
#       `class LayoutConstraint` 的类注释逐字
LAYOUT_ASSIGNMENT_MECHANISM = (
    "During layout assignment, constraint objects are created, prioritized, and "
    "propagated across the HLO graph. When conflicting constraints arise, the "
    "constraint with higher priority takes precedence, and layout copy instructions "
    "(`kCopy`) are inserted where necessary to bridge incompatible layouts."
)

# 来源：同上，约束类型的三个具体子类逐字
LAYOUT_CONSTRAINT_KINDS = (
    "BufferLayoutConstraint: Constrains the layout of a LogicalBuffer produced by an "
    "instruction. | OperandLayoutConstraint: Constrains the layout expected for an "
    "operand by its consumer instruction. | ComputationLayoutConstraint: Constrains "
    "the parameter and result layouts of an HloComputation interface."
)

# 来源：https://github.com/tensorflow/tensorflow/commit/358fd36d0f2c23b725bf952d7c919e7d704a45ec
#       标题与提交信息逐字。日期 2018-03-07（commit author date，2026-09-16 经
#       GitHub API 取回）。这是「融合体内部不携带 layout」的一手证据。
TF_COMMIT_358FD36D_TITLE = "[XLA:GPU] Move layout_assignment before fusion"
TF_COMMIT_358FD36D_MESSAGE = (
    "This will allow code simplification and opens up new optimization. Currently we "
    "don't emit layouts inside of fusion and tracing layouts through fusion is very "
    "hard. Changing the pipeline sidesteps this issue. This is mostly perf-neutral."
)
TF_COMMIT_358FD36D_URL = (
    "https://github.com/tensorflow/tensorflow/commit/358fd36d0f2c23b725bf952d7c919e7d704a45ec"
)

# 来源：https://github.com/openxla/xla/blob/main/xla/service/gpu/gpu_compiler.cc
#       `GpuCompiler::OptimizeHloModule` 中的调用顺序逐字（行号为取用 2026-09-16 的 main）：
#         RunLayoutAssignmentPasses(...)          // layout assignment
#         RunLayoutNormalizationPasses(...)
#         OptimizeHloPostLayoutAssignment(...)
#         RunDynamicSliceFusionPasses(...)
#         RunFusionPasses(...)                    // fusion
#         RunPostFusionPasses(...)
#       → **layout assignment 确实在 fusion 之前**。
GPU_PIPELINE_ORDER = [
    "RunLayoutAssignmentPasses",
    "RunLayoutNormalizationPasses",
    "OptimizeHloPostLayoutAssignment",
    "RunDynamicSliceFusionPasses",
    "RunFusionPasses",
    "RunPostFusionPasses",
]
GPU_PIPELINE_SOURCE = (
    "https://github.com/openxla/xla/blob/main/xla/service/gpu/gpu_compiler.cc"
)

# ---------------------------------------------------------------------------
# 六、MemorySpaceAssignment（文档 03）—— ★ 方向
# ---------------------------------------------------------------------------

# 来源：https://github.com/openxla/xla/blob/main/xla/service/memory_space_assignment/memory_space_assignment.h
#       `class MemorySpaceAssignment` 的类注释逐字。
#       ★ 源码自身有笔误：`as as many`（重复了一个 as）。**这里照抄并标明**。
#       ★ 方向是「尽量塞进 alternate」：default 只是「假定足够大」的兜底。
MSA_DIRECTION = (
    "MemorySpaceAssignment assigns memory spaces (default or alternate) to each "
    "instruction in the module. It will greedily try placing as as many values in the "
    "alternate memory space as possible. It uses the heap simulator to determine the "
    "actual allocation offsets of values in the alternate memory space to account for "
    "fragmentation. The default memory space is assumed to be large enough to hold the "
    "values that could not be placed in the alternate memory space."
)
MSA_SOURCE_TYPO_NOTE = "引文里的 `as as many` 是源码自身的笔误，照抄。"

# 来源：同上，Copy Allocation 条目逐字 —— **prefetch 是 default→alt**
MSA_COPY_DIRECTIONS = (
    "Copy Allocation: Instructions to copy an AllocationValue from one memory space to "
    "another. Used for prefetching (default mem -> alt mem), and eviction "
    "(alt mem -> default mem)."
)

# ---------------------------------------------------------------------------
# 七、GPU 上「同名但不同」的 buffer 着色 + 现行 pipeline 不跑 MSA（文档 03）
# ---------------------------------------------------------------------------

# 来源：https://github.com/openxla/xla/blob/main/xla/service/gpu/gpu_memory_space_assignment.h
#       `enum class MemorySpaceColor` 逐字。**只有三个成员，且与上面那个 pass 无关**：
GPU_MSA_COLOR_ENUM = """enum class MemorySpaceColor {
  kDefault = 0,
  kCollective = 1,
  kTempBuffer = 2,
};"""
GPU_MSA_COLOR_PURPOSE = (
    "Creates a buffer colorer that assigns memory space colors to HLO values during "
    "buffer assignment. It handles: Collective operations (all-reduce, all-gather, "
    "etc.) -> kCollective; Mosaic with multimem -> kCollective; Custom call "
    "`operands_memory_spaces` / `results_memory_spaces` frontend attributes -> "
    "requested memory space; Everything else -> kDefault"
)

# 来源：https://github.com/openxla/xla/blob/main/xla/service/gpu/gpu_compiler.cc
#       `OptimizeHloPostLayoutAssignment` 里的 `pipeline.AddPass<HostOffloader>(alias_info);`
#       以及 `CollectivePipeliner` 的 host-offloading 分支，
#       开关是 `debug_options.xla_gpu_enable_pipelined_host_offloading()`。
#       ★ GPU 现行 pipeline 中**看不到 MemorySpaceAssignment pass**；
#         host 卸载走 `CollectivePipeliner` + `HostOffloader`。
GPU_PIPELINE_HOST_OFFLOADING = (
    "if (debug_options.xla_gpu_enable_pipelined_host_offloading() || "
    "IsPassEnabledAtOptimizationEffort<CollectivePipeliner>(*hlo_module)) { ... }"
)
GPU_HOST_OFFLOADER_PASS = "pipeline.AddPass<HostOffloader>(alias_info);"

# 来源：https://github.com/openxla/xla/blob/main/xla/service/memory_annotations.h
#       逐字。**custom call target 是 PascalCase**，不是 kebab-case：
MOVE_TO_HOST_CUSTOM_CALL = "MoveToHost"
MOVE_TO_DEVICE_CUSTOM_CALL = "MoveToDevice"
PIN_TO_DEVICE_CUSTOM_CALL = "PinToDevice"

# 来源：https://github.com/openxla/xla/blob/main/xla/service/host_offload_utils.h
HOST_OFFLOAD_UTILS_API = (
    "bool IsMoveToHostWithDynamicUpdateSlice(const HloInstruction* instr); "
    "bool IsMoveToDeviceWithDynamicSlice(const HloInstruction* instr);"
)

# ---------------------------------------------------------------------------
# 八、dot_general 的维度命名与 jnp 语义（文档 01）
# ---------------------------------------------------------------------------

# 来源：https://github.com/openxla/xla/blob/main/xla/xla_data.proto
#       （取用 2026-09-16）`message DotDimensionNumbers` 的四个字段逐字
#       —— **注意是 `*_batch_dimensions`**：
HLO_PROTO_DIM_NAMES = ("lhs_contracting_dimensions", "rhs_contracting_dimensions",
                       "lhs_batch_dimensions", "rhs_batch_dimensions")

# 来源：https://github.com/openxla/xla/blob/main/xla/hlo/ir/hlo_instruction.cc
#       （取用 2026-09-16）`DotDimensionNumbersToString()` 逐字所用的**短路名**
#       —— HLO **文本格式**打印的是 `lhs_contracting_dims` / `lhs_batch_dims`：
#         result.push_back(StrCat("lhs_batch_dims={", ...));
#         result.push_back(StrCat("lhs_contracting_dims={", ...));
HLO_TEXT_DIM_NAMES = ("lhs_contracting_dims", "rhs_contracting_dims",
                      "lhs_batch_dims", "rhs_batch_dims")

# 来源：https://github.com/openxla/stablehlo/blob/main/docs/spec.md  §dot_general
#       **StableHLO op attribute 侧**的字段名逐字（注意是 `batching`，不是 `batch`）：
STABLEHLO_DIM_NAMES = ("lhs_contracting_dimensions", "rhs_contracting_dimensions",
                       "lhs_batching_dimensions", "rhs_batching_dimensions")

# 来源：同上，§dot_general 的 Constraints 逐字（C1–C10，非量化部分全列）
STABLEHLO_DOT_CONSTRAINTS = (
    "(C1) size(lhs_batching_dimensions) = size(rhs_batching_dimensions). "
    "(C2) size(lhs_contracting_dimensions) = size(rhs_contracting_dimensions). "
    "(C3) is_unique(lhs_batching_dimensions + lhs_contracting_dimensions). "
    "(C4) is_unique(rhs_batching_dimensions + rhs_contracting_dimensions). "
    "(C5) 0 <= lhs_batching_dimensions < rank(lhs). "
    "(C6) 0 <= lhs_contracting_dimensions < rank(lhs). "
    "(C7) 0 <= rhs_batching_dimensions < rank(rhs). "
    "(C8) 0 <= rhs_contracting_dimensions < rank(rhs). "
    "(C9) dim(lhs, lhs_batching_dimensions...) = dim(rhs, rhs_batching_dimensions...). "
    "(C10) dim(lhs, lhs_contracting_dimensions...) = "
    "dim(rhs, rhs_contracting_dimensions...). "
    "(C12) shape(result) = dim(lhs, lhs_batching_dimensions) + "
    "dim(lhs, lhs_result_dimensions) + dim(rhs, rhs_result_dimensions)."
)

# 来源：https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.dot.html
#       逐字（2026-09-16 取用）。**这是「读了文档仍会意外」的那一条**：
JNP_DOT_VS_MATMUL = (
    "This differs from jax.numpy.matmul() in two respects: if either a or b is a "
    "scalar, the result of dot is equivalent to jax.numpy.multiply(), while the result "
    "of matmul is an error. if a and b have more than 2 dimensions, the batch indices "
    "are stacked rather than broadcast."
)
JNP_DOT_SOURCE = "https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.dot.html"

# ---------------------------------------------------------------------------
# 九、StableHLO 与 HLO 的关系（文档 01/02 的边界说明）
# ---------------------------------------------------------------------------

# 来源：https://github.com/openxla/stablehlo/blob/main/docs/compatibility.md
#       逐字。**保证只对 portable artifact 生效**；pretty-print 与 bytecode 不在范围内。
STABLEHLO_COMPAT_WINDOW = (
    "**5 years of backward compatibility:** Portable artifacts serialized by an old "
    "version of libStablehlo have the same semantics when deserialized by a new version "
    "of libStablehlo if these versions are built from openxla/stablehlo commits which "
    "are less than 5 years apart. "
    "**2 years of forward compatibility:** Portable artifacts serialized by a new "
    "version of libStablehlo have the same semantics when deserialized by an old "
    "version of libStablehlo if these versions are built from openxla/stablehlo commits "
    "which are less than 2 years apart, unless the program is using new features "
    "introduced since the old version."
)

# 来源：https://github.com/openxla/stablehlo/blob/main/docs/spec.md 首段
STABLEHLO_POSITIONING = (
    "StableHLO is a backward compatible ML compute opset inspired by HLO/MHLO."
)

# ---------------------------------------------------------------------------
# 十、本仓迷你实现自己定的口径（**不是 XLA 的值**，用于让程序里的常数可追溯）
# ---------------------------------------------------------------------------

# 下面三条是**本仓库迷你实现自己的约定**，写在 specs.py 里是为了让编号程序
# 里的常数有单一出处；**它们不对应任何 XLA 常量，引用时不得写成 XLA 的值**。
MINI_BUDGET_LIMIT = 96          # 迷你实现照抄 MaxOperandsAndOutputsPerFusion 的量级
MINI_ALT_MEMORY_CAPACITY = 4096  # 迷你 alternate memory 容量（字节）
MINI_DEFAULT_MEMORY_UNBOUNDED = True  # 迷你 default memory「假定足够大」

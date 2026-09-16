"""MLIR 的外部事实源：每条常量带一行 `# 来源：` 与**可打开的 URL**。

本文件只放**从一手来源抄来的**东西（官方文档原文 / 源码文件），不放推导值，
不放本仓库自己跑出来的数。找不到一手来源的项**不写进来**，记入 UNSOURCED。

★ 核查范围（错误模式 P6 的对策）：下面的「已核实」逐条写明**核的是什么载体**
  ——文档页 / 源码文件 / PR，**不写「全库」「全树」这类大于实际核查范围的措辞**。

★ 版本坐标：源码类条目钉在 llvm-project `main`，取用于 2026-09-16。
  这是**会漂移的坐标**——引用时须写明取用日期，不得写成「MLIR 里 X 就是这样的」。

UNSOURCED（查不到一手来源，**正文不使用**）：
  · 「`applyPatternsGreedily` 这个名字在 2020 年被改成 `applyPatternsAndFoldGreedily`、
    2024 年又改回来」——2024 年那次有 PR #104649 的 diff 为证；
    **2020 年那个方向翻转本文未取到一手证据**，正文不写。
  · 「那四个旧式 dialect-conversion pass 在 LLVM 19/20 被移除、
    `-buffer-deallocation` 在 LLVM 21 被移除」——**版本归属未取到一手证据**。
    能核到的只有：2026-09-16 的 pass 索引页里这四个名字 0 命中。
  · 「Region 至少含一个 Block」——这条**不是缺来源，是错的**。
    见 LANGREF_REGION_GRAMMAR：`region ::= "{" entry-block? block* "}"` 允许零个 block。
"""

# ---------------------------------------------------------------------------
# IR 结构（LangRef）
# ---------------------------------------------------------------------------

# 来源：https://mlir.llvm.org/docs/LangRef/  §High-Level Structure
LANGREF_HIERARCHY = (
    "Operations are contained in Blocks and Blocks are contained in Regions. "
    "Operations may also contain regions, enabling hierarchical structures to be "
    "represented."
)

# 来源：https://mlir.llvm.org/docs/LangRef/  §High-Level Structure
LANGREF_VALUE_DEF = (
    "Each Value is the result of exactly one Operation or Block Argument, and has a "
    "Value Type defined by the type system."
)

# 来源：https://mlir.llvm.org/docs/LangRef/  §Regions
# 语法：region ::= "{" entry-block? block* "}"   —— `?` 与 `*` 都允许零次
#       entry-block ::= operation+               —— 一旦出现就非空
LANGREF_REGION_GRAMMAR = (
    'region      ::= "{" entry-block? block* "}"\n'
    'entry-block ::= operation+'
)

# 来源：https://mlir.llvm.org/docs/LangRef/  §Blocks
# block ::= block-label operation+   —— Block **非空**
LANGREF_BLOCK_GRAMMAR = (
    "block           ::= block-label operation+\n"
    "block-label     ::= block-id block-arg-list? `:`"
)

# 来源：https://mlir.llvm.org/docs/LangRef/  §Operations（operation 的语法）
# region-list ::= "(" region ("," region)* ")"   —— `*` 允许 0 个，故 Op 可以有零个 region
LANGREF_REGION_LIST = 'region-list           ::= "(" region ("," region)* ")"'

# 来源：https://mlir.llvm.org/docs/LangRef/  §Regions
LANGREF_ENTRY_BLOCK = (
    "An entry block is a block with no label and no arguments that may occur at the "
    "beginning of a region."
)

# ---------------------------------------------------------------------------
# 方言定位（各 dialect 文档页首段）
# ---------------------------------------------------------------------------

DIALECT_POSITIONING = {
    # 来源：https://mlir.llvm.org/docs/Dialects/ArithOps/
    "arith": "The arith dialect is intended to hold basic integer and floating point "
             "mathematical operations. This includes unary, binary, and ternary "
             "arithmetic ops, bitwise and shift ops, cast ops, and compare ops.",
    # 来源：https://mlir.llvm.org/docs/Dialects/MathOps/
    "math": "The math dialect is intended to hold mathematical operations on integer "
            "and floating types beyond simple arithmetics. Each operation works on "
            "scalar, vector or tensor type.",
    # 来源：https://mlir.llvm.org/docs/Dialects/SCFDialect/
    "scf": "Unstructured control flow operations are located in the cf (control flow) "
           "dialect. … Typically, scf is lowered to cf and then lowered to some final "
           "target like LLVM or SPIR-V.",
    # 来源：https://mlir.llvm.org/docs/Dialects/Vector/
    # ★ 这是 vector 方言文档里的一节标题：自动向量化**不在** vector 方言职责内
    "vector": "The Big Out-Of-Scope Piece: Automatic Vectorization",
    # 来源：https://mlir.llvm.org/docs/Dialects/GPU/
    "gpu": "Note: this dialect is more likely to change than others in the near "
           "future; use with caution. This dialect provides middle-level abstractions "
           "for launching GPU kernels following a programming model similar to that "
           "of CUDA or OpenCL.",
}

# ---------------------------------------------------------------------------
# tensor / memref / bufferization
# ---------------------------------------------------------------------------

# 来源：https://mlir.llvm.org/docs/Bufferization/  §Overview
# ★ 官方措辞是 「tensor semantics」「memref semantics」，
#   **不是**「value semantics / buffer semantics」——后者可作概念讲，不得写成引文。
BUFFERIZATION_DEF = (
    "Bufferization in MLIR is the process of converting ops with tensor semantics to "
    "ops with memref semantics."
)

# 来源：https://mlir.llvm.org/docs/Bufferization/  §What is One-Shot Bufferize?
BUFFERIZATION_ONESHOT = (
    "MLIR provides a pass -one-shot-bufferize that performs an analysis and bufferizes "
    "all ops with tensor semantics that implement BufferizableOpInterface."
)

# 来源：https://mlir.llvm.org/docs/Bufferization/  §Goals of Bufferization
BUFFERIZATION_2PHASE = (
    "2-Phase: Bufferization is internally broken down into 2 steps: First, analyze the "
    "entire IR and make bufferization decisions. Then, bufferize (rewrite) the IR."
)

# 来源：https://mlir.llvm.org/docs/Builtin/  §MemRefType
# ★ memref 的官方定义。注意它在 **Builtin** 方言页，不在 MemRef 方言页。
MEMREF_TYPE_DEF = "Shaped reference to a region of memory"

# 来源：https://mlir.llvm.org/docs/Bufferization/  §Overview
BUFFERIZATION_WHY_LATE = (
    "These passes typically run as one of the last steps in a pass pipeline, right "
    "before lowering to memref ops to LLVM. That is because many transformations are "
    "easier or only supported in tensor land; e.g., tile/fuse/… on tensors first, "
    "then bufferize the remaining IR."
)

# 来源：https://mlir.llvm.org/docs/Bufferization/  §What is One-Shot Bufferize?
BUFFERIZATION_ONESHOT_POSITIONING = (
    "One-Shot Bufferize is a tensor bufferization pass designed for IR in "
    "destination-passing style, and with aggressive in-place bufferization."
)

# 来源：https://mlir.llvm.org/docs/Bufferization/  §Deprecated Passes
# ★ 措辞要引全：官方说的是「in the resulting IR」，不是进程漏内存。
# ★ 这一节讲的是**释放**那一步（buffer deallocation），不是 tensor→memref 那一步。
BUFFERIZATION_DEALLOC_DEPRECATED = (
    "The buffer deallocation pass has been deprecated in favor of the ownership-based "
    "buffer deallocation pipeline. The deprecated pass has some limitations that may "
    "cause memory leaks in the resulting IR."
)

# 旧式 dialect-conversion bufferization 的四个 pass 名。
# 核查范围（错误模式 P6 的对策）：核对的是
# https://mlir.llvm.org/docs/Passes/ 的 **pass 索引页**，2026-09-16 取用，
# 四个名字**均 0 命中**。**没有**核 pass 历史、也**没有**核它们在哪个版本被移除。
BUFFERIZATION_OLD_CONVERSION_PASSES = (
    "linalg-bufferize / tensor-bufferize / scf-bufferize / finalizing-bufferize"
)

# ---------------------------------------------------------------------------
# lowering：pass 名与前置条件
# ---------------------------------------------------------------------------

# 来源：https://mlir.llvm.org/docs/Passes/  §-convert-linalg-to-parallel-loops
# ★ 该页对这条 pass 只写了「Lower the operations from the linalg dialect into
#   parallel loops」，**没有逐字写 scf.parallel**。
#   因此不硬引它，改用下一条（消费方）来支撑「parallel loops 才是进 GPU 的那一态」。
LINALG_TO_PARALLEL_LOOPS = (
    "Lower the operations from the linalg dialect into parallel loops"
)

# 来源：https://mlir.llvm.org/docs/Passes/  §-convert-linalg-to-loops
# ★ 这条是本主题文档 02 与文档 03 的扣合点：buffer 语义是 lowering 的**前置条件**
LINALG_TO_LOOPS = (
    "Lowers the linalg ops to loop nests using scf.for. Pre-condition: the operands "
    "used by the linalg ops have buffer semantics, i.e., tensor operands and results "
    "must be converted to memrefs via bufferization."
)

# 来源：llvm-project 源码 mlir/lib/Dialect/Linalg/Transforms/Loops.cpp（取用 2026-09-16）
# ★ 这是「三条 linalg→循环的 pass 各自产出哪种循环」的**逐行物证**：
#     LowerToLoops      :: runOnOperation -> lowerLinalgToLoopsImpl<scf::ForOp>
#     LowerToParallelLoops :: runOnOperation -> lowerLinalgToLoopsImpl<scf::ParallelOp>
#     以及 linalgOpToAffineLoops -> linalgOpToLoopsImpl<affine::AffineForOp>
#   同一文件里 **`forall` 命中数为 0**——forall 那条路不在这里，
#   要走 Transform 方言的 transform.structured.tile_using_forall。
#   （Passes 文档页对 -convert-linalg-to-parallel-loops 只说 "into parallel loops"，
#     没有逐字写 scf.parallel，所以这里改用源码。）
LINALG_LOOPS_SOURCE = "mlir/lib/Dialect/Linalg/Transforms/Loops.cpp"
LINALG_PARALLEL_LOOPS_PRODUCES = "scf.parallel"
LINALG_LOOPS_PRODUCES = "scf.for"

# 来源：https://mlir.llvm.org/docs/Passes/  §-convert-parallel-loops-to-gpu
# ★ 用这条来支撑「parallel loops 才是进 GPU 的那一态」——它明说消费的是 scf.parallel。
#   （`-convert-linalg-to-parallel-loops` 自己的页只写了 "Lower the operations from
#    the linalg dialect into parallel loops"，**没有逐字写 scf.parallel**，故不硬引。）
PARALLEL_LOOPS_TO_GPU = (
    "Convert mapped scf.parallel ops to gpu launch operations. Creates a pass that "
    "converts scf.parallel operations into a gpu.launch operation."
)

# 来源：https://mlir.llvm.org/docs/Dialects/Transform/（transform op 索引）
GPU_TRANSFORM_OPS = [
    "transform.structured.tile_using_forall",
    "transform.gpu.map_forall_to_blocks",
    "transform.gpu.map_nested_forall_to_threads",
    "transform.structured.vectorize",
]

# 来源：llvm-project 源码
#   mlir/include/mlir/Dialect/GPU/TransformOps/GPUTransformOps.td
#   （def MapForallToBlocks，description 段）
# ★ 这是「值语义 → 缓冲语义是并行化的前置条件」的**逐字**依据
MAP_FORALL_TO_BLOCKS_NOTE = (
    "Only bufferized scf.forall are currently supported. "
    "If any scf.forall with tensors is found, the transform definitely fails."
)

# 来源：https://mlir.llvm.org/docs/Dialects/GPU/  §GPU Compilation
GPU_NVVM_PIPELINE_NOTE = (
    "This pipeline specifically requires explicitly parallel IR and doesn't do GPU "
    "parallelization. To enable parallelism, necessary transformations must be applied "
    "before utilizing this pipeline."
)

# 物证（不是引文，是本仓库从源码读出的组合，用于佐证上一条）：
# 来源：llvm-project 源码 mlir/lib/Dialect/GPU/Pipelines/GPUToNVVMPipeline.cpp
#       buildCommonPassPipeline() 里的 pass 序列（取用于 2026-09-16）
# ★ 组合里**没有任何并行化 pass**（没有 map_forall_to_blocks / parallel-loop 映射），
#   只有既有并行结构的 lowering：convert-nvgpu-to-nvvm / gpu-kernel-outlining /
#   convert-vector-to-scf / scf-to-control-flow / convert-nvvm-to-llvm /
#   convert-func-to-llvm / nvvm-attach-target {chip, features, optLevel} /
#   lower-affine / arith-to-llvm / convert-index-to-llvm / canonicalize / cse
GPU_NVVM_PIPELINE_COMPOSITION_SOURCE = (
    "mlir/lib/Dialect/GPU/Pipelines/GPUToNVVMPipeline.cpp"
)

# ---------------------------------------------------------------------------
# pattern rewrite
# ---------------------------------------------------------------------------

# 来源：llvm-project 源码 mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h
#       （applyPatternsGreedily 的 doc 注释，取用于 2026-09-16）
GREEDY_SEMANTICS = (
    "repeatedly applying the highest benefit patterns in a greedy worklist driven "
    "manner until a fixpoint is reached."
)

# 来源：同上文件
GREEDY_PREMATURE = (
    "The greedy rewrite may prematurely stop after a maximum number of iterations, "
    "which can be configured in the configuration parameter."
)

# 来源：同上文件
GREEDY_RETURN = (
    'Returns "success" if the iterative process converged (i.e., fixpoint was reached) '
    "and no more patterns can be matched within the region."
)

# 来源：llvm-project 源码 mlir/lib/Transforms/Canonicalizer.cpp
# ★ 物证（不是引文）：canonicalize 调用的是**同一个** greedy driver。
CANONICALIZER_USES_GREEDY = "mlir/lib/Transforms/Canonicalizer.cpp"

# 来源：同上文件（GreedyPatternRewriteDriver.h，applyPatternsGreedily 的注释）
GREEDY_DCE = (
    "Also performs simple dead-code elimination before attempting to match any of the "
    "provided patterns."
)

# 来源：同上文件，GreedyRewriteConfig 的成员初值（取用于 2026-09-16）
#   int64_t maxIterations = 10;
#   bool    useTopDownTraversal = false;
#   GreedySimplifyRegionLevel regionSimplificationLevel = GreedySimplifyRegionLevel::Aggressive;
#   bool    fold = true;
GREEDY_CONFIG_DEFAULTS = {
    "maxIterations": 10,               # 来源：GreedyPatternRewriteDriver.h:169
    "fold": True,                      # 来源：GreedyPatternRewriteDriver.h:173
    "useTopDownTraversal": False,      # 来源：GreedyPatternRewriteDriver.h:166
    "regionSimplificationLevel": "Aggressive",   # 来源：GreedyPatternRewriteDriver.h:167
}

# 来源：同上文件——**当前名字**是 applyPatternsGreedily（同文件里有它的声明，
#       `applyPatternsAndFoldGreedily` 在该文件里 **0 命中**）。
GREEDY_CURRENT_API = "applyPatternsGreedily"
GREEDY_OLD_API = "applyPatternsAndFoldGreedily"

# 版本边界的复核（2026-09-16 取用）：在五个分支上 grep 同一个头文件里的
#   applyPatternsAndFoldGreedily
#     release/20.x : 2 命中
#     release/21.x : 2 命中
#     release/22.x : 2 命中
#     release/23.x : 0 命中
#     main         : 0 命中
# 即：**旧名在 20/21/22 上仍在，从 23 起消失**。不得写成「LLVM 21 已移除」。
GREEDY_OLD_NAME_PRESENT_IN = ("release/20.x", "release/21.x", "release/22.x")
GREEDY_OLD_NAME_ABSENT_IN = ("release/23.x", "main")

# 来源：llvm-project PR #104649
#   https://github.com/llvm/llvm-project/pull/104649
#   title: "[mlir] Enable decoupling two kinds of greedy behavior."
#   merged_at: 2024-12-20T16:15:49Z（经 GitHub API 查得）
# ★ 版本边界：旧名在 LLVM 21/22 上仍可用，main 已移除。
#   **不得写成「LLVM 21 已移除」。**
GREEDY_RENAME_PR = 104649
GREEDY_RENAME_PR_TITLE = "[mlir] Enable decoupling two kinds of greedy behavior."
GREEDY_RENAME_PR_MERGED = "2024-12-20"

# ---------------------------------------------------------------------------
# Interface 与 Trait
# ---------------------------------------------------------------------------

# 来源：https://mlir.llvm.org/docs/Interfaces/
# ★ 关键措辞：interface 是**注册到 op 上**（通过自带的 Trait），不是「注册到 MLIRContext」；
#   只有 **external model** 才与 context 绑定。
INTERFACE_REGISTRATION = (
    "it is registered to an operation by adding the provided trait "
    "ExampleOpInterface::Trait"
)

# 来源：https://mlir.llvm.org/docs/Interfaces/  §External Models
INTERFACE_EXTERNAL_MODEL = (
    "Other contexts will not see the interface unless registered."
)

# 来源：mlir/include/mlir/IR/OpDefinition.h（OpTrait::TraitBase 的模板形态）
TRAIT_BASE_TEMPLATE = "OpTrait::TraitBase<ConcreteType, TraitType>"

# 可引的确切接口名。来源：https://mlir.llvm.org/docs/Interfaces/
# （DestinationStyleOpInterface / BufferizableOpInterface 另见 Bufferization 文档；
#   TilingInterface 见 TilingInterface.td；其余见各自 .td）
INTERFACE_NAMES = [
    "DestinationStyleOpInterface",
    "BufferizableOpInterface",
    "TilingInterface",
    "MemoryEffectOpInterface",
    "InferTypeOpInterface",
]

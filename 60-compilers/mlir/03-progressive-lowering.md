---
title: 渐进式 lowering
slug: mlir-progressive-lowering
domain: 60-compilers
topic: mlir
level: 3
difficulty: advanced
prerequisites:
  - 60-compilers/mlir/01-ir-and-dialects
  - 60-compilers/mlir/02-tensor-vs-memref
tags: [mlir, lowering, linalg, scf, forall, gpu, nvvm, bufferization]
related:
  - 60-compilers/mlir/01-ir-and-dialects
  - 60-compilers/mlir/02-tensor-vs-memref
  - 30-gpu-programming/triton
  - 40-kernels/gemm/02-shared-memory-tiling
code:
  - code/03_lowering_pipeline.py
status: stable
updated: 2026-09-16
---

# 渐进式 lowering

> 从 `linalg` 张量表达式到 NVVM，中间要经过四次形态改变。这条链里最关键的一步不是最后那次"上 GPU"，而是最前面那次"tensor 变 memref"——因为**并行化那一步的合法性，以缓冲语义为前提**。官方在 `map_forall_to_blocks` 的文档里写得很直白：只要 forall 里还有 tensor，这条变换就"definitely fails"。

## 0. 背景与动机

同一个 `out = (a + 1.5) * b`，在 lowering 链的五个位置上长成五种样子（本文配套程序的实测输出，格式是本仓库迷你实现的）：

| 阶段 | 形态 | 这一层的 op 数 |
|---|---|---|
| S0 | `linalg` on **tensor** | 7 |
| S1 | `linalg` on **memref** + `memref.alloc` | 10 |
| S2 | 嵌套 `scf.for` + `memref.load/store` + `arith.addf` | 6 个 `scf.for` |
| S3 | `scf.parallel`（单层多维） | 3 个 `scf.parallel` |
| S4 | `scf.forall` | 3 个 `scf.forall` |
| S5 | `gpu.launch`（映射到 block） | 3 个 `gpu.launch` |

（op 数与循环个数都是**本仓库迷你实现**的实测值，见 §5。具体数字不重要，重要的是**每一层丢掉了什么、又必须保留什么**。）

"渐进式 lowering"的含义就在这张表里：**不存在一次从 S0 到 S5 的转换**，存在的是五次各自只做一件小事的转换。这样设计的好处是每一步都能被单独测试、单独替换；代价是**每一步都有前置条件**，而前置条件不满足时的失败方式往往很反直觉。

最反直觉的那一条，正是本文的主线：

> **S1（tensor → memref）不是"顺便做的优化"，它是 S4/S5（并行化）的硬前置条件。**

这与本域的统一主线（物化 vs 重算）的关系是：本文是主线的**判据来源**——它说明"这个决策最迟必须在 pipeline 的哪个位置做完"。要并行化，中间结果的存储就必须已经确定；而"存储确不确定"正是 `tensor` 与 `memref` 的唯一区别（见 [02-tensor-vs-memref](02-tensor-vs-memref.md) §1）。

§1 把这条链的每一步的确切名字与前置条件列出来，§2 讲官方给的那条手动 pipeline，§3 把主线扣起来。

## 1. 原理

**第一步：确切的 pass 名表。**

MLIR 的 pass 名是"拼写即接口"——名字写错就是在讨论另一个东西。下面这张表的每一行都取自官方 pass 索引或源码，逐条见 `code/specs.py`：

| 环节 | 确切名字 | 产出什么 |
|---|---|---|
| linalg → 循环 | `-convert-linalg-to-loops` | `scf.for` 嵌套 |
| linalg → 并行循环 | `-convert-linalg-to-parallel-loops` | 并行循环 |
| linalg → affine 循环 | `-convert-linalg-to-affine-loops` | `affine.for` |
| 并行循环 → GPU | `-convert-parallel-loops-to-gpu` | `gpu.launch`（**消费 `scf.parallel`**） |
| tile 成 forall | `transform.structured.tile_using_forall` | `scf.forall` |
| forall → block | `transform.gpu.map_forall_to_blocks` | block 级映射 |
| forall → thread | `transform.gpu.map_nested_forall_to_threads` | thread 级映射 |
| 向量化 | `transform.structured.vectorize` | vector 方言 |
| vector → gpu | `-convert-vector-to-gpu`（选项 `-use-nvgpu`） | `nvgpu` |
| gpu → nvvm | `-convert-gpu-to-nvvm` | NVVM 方言 |
| nvgpu → nvvm | `-convert-nvgpu-to-nvvm` | NVVM 方言 |
| 打包 | `-gpu-lower-to-nvvm-pipeline` | 整条 |

**两处最容易记错的**：

1. **`-convert-linalg-to-parallel-loops` 与 `scf.forall` 不是一回事。** 三条 linalg→循环的 pass 各自产出**不同种类**的循环，这在源码里逐行可查（`mlir/lib/Dialect/Linalg/Transforms/Loops.cpp`，取用 2026-09-16）：

   ```text
   LowerToLoops         :: runOnOperation -> lowerLinalgToLoopsImpl<scf::ForOp>
   LowerToParallelLoops :: runOnOperation -> lowerLinalgToLoopsImpl<scf::ParallelOp>
   （另有 linalgOpToAffineLoops   -> linalgOpToLoopsImpl<affine::AffineForOp>）
   ```

   即：`-convert-linalg-to-loops` 产出 **`scf.for`**、`-convert-linalg-to-parallel-loops` 产出 **`scf.parallel`**、`-convert-linalg-to-affine-loops` 产出 **`affine.for`**——**三者都不是 `scf.forall`**（同一文件里 `forall` 命中数为 **0**）。

   顺带一条旁证：把 `scf.parallel` 变成 gpu launch 的是另一条 pass，官方描述是 "Convert mapped **scf.parallel** ops to gpu launch operations"——消费方也确认了这一态是 `scf.parallel`。

   要走 `scf.forall` 那条路，得用 Transform 方言的 `transform.structured.tile_using_forall`，或者 C++ 侧 `scf::tileUsingSCF` 配 `SCFTilingOptions::LoopType`（**默认是 `ForOp`**）。**"`-convert-linalg-to-parallel-loops` 产出 `scf.forall`"是错的**，而且错了之后整条 GPU 路径的名字都会跟着错。
2. **`-gpu-lower-to-nvvm-pipeline` 不做并行化。** 官方 GPU 方言页原文：

   > This pipeline **specifically requires explicitly parallel IR and doesn't do GPU parallelization**. To enable parallelism, necessary transformations must be applied before utilizing this pipeline.

   把它当"一键上 GPU"是常见误读。它的名字里写的是 lower，不是 parallelize。§5 里的 C2 断言就是把这句话跑成了可执行检查：把只有顺序 `scf.for` 的 IR 交给它，程序必须报错。

**第二步：`-convert-linalg-to-loops` 的前置条件，逐字读。**

官方 pass 索引里这一条写得很清楚：

> Lowers the linalg ops to loop nests using **scf.for**. **Pre-condition: the operands used by the linalg ops have buffer semantics**, i.e., tensor operands and results must be converted to memrefs via bufferization.

这句话里有三件事：

- 产出是 **`scf.for`**（不是 `affine.for`，也不是 `scf.parallel`）；
- **前置条件是 buffer 语义**；
- 而且官方顺手解释了"buffer 语义"怎么达到：**"must be converted to memrefs via bufferization"**。

注意那个 "must"。这不是"建议先 bufferize 效果更好"，是**不 bufferize 就不满足前置条件**。

**第三步：链上最紧的那个扣。**

真正把整条链扣死的是 `transform.gpu.map_forall_to_blocks` 的文档（源码 `GPUTransformOps.td`，逐字）：

> **Only bufferized scf.forall are currently supported.** … If any **scf.forall with tensors** is found, the transform **definitely fails**.

拆开读：

- 前半句是**能力边界**：这个 transform 当前只支持 bufferize 过的 `scf.forall`；
- 后半句是**失败语义**：遇到带 tensor 的 forall，它**一定失败**（definitely fails，不是"可能"）。

把第二步与第三步连起来，就得到一条**跨两步的硬约束**：

```text
tensor 语义的 IR ──bufferize──▶ memref 语义的 IR ──并行化──▶ forall/parallel ──▶ gpu
                        ▲                                    ▲
                        └── 这一步不做，后面那一步一定失败 ────┘
```

**这就是"值语义到缓冲语义的转换不是预处理，而是后续并行化的前置条件"**——它也是 [02-tensor-vs-memref](02-tensor-vs-memref.md) 与本文的扣合点。

**第四步：每一层丢掉了什么。**

| 阶段 | 保留 | 丢掉 |
|---|---|---|
| S0 → S1 | 形状、元素类型、运算 | **"存储未指定"这个自由度**；以及 `tensor.empty` 这样的"内容未定"标记 |
| S1 → S2 | 循环的**嵌套结构**、迭代顺序 | **"这是同一个操作的循环"这个整体性**——变成了一堆标量 load/store/compute |
| S1 → S3 | 循环的**多维索引空间**变成了一个 op 的属性 | 嵌套结构（从 4 层降到 3 层，见 §5） |
| S3 → S4 | 同上，但换了方言 | —（这一步主要是给 Transform 方言一个可操作的把手） |
| S4 → S5 | 索引空间的语义 | "这是一次通用的并行循环"——变成"这是 block 网格的一次启动" |

S2 与 S3 的差别值得单独记：**`scf.for` 是嵌套的，`scf.parallel` 是单层多维的**。本机实测里 S2 的 op 嵌套深度是 4、S3 是 3（§5）——两者描述的是同一个 2×3 迭代空间，但一个用"循环套循环"表达，一个用"一个 op 带多维边界"表达。**这个差别在 lowering 到 GPU 时会变成硬约束**：block 网格天然是多维的，所以并行那一支走的是 S3/S4 那条路，不是 S2。

## 2. 执行映射

**组件一：pass manager 里的顺序。** §1 那条链不是建议，而是 pipeline 的顺序约束。一个典型的 GPU 编译 pipeline 大致是：

```text
linalg on tensors
  → one-shot-bufferize                       （S0 → S1）
  → tile / fuse（Transform 方言）              （→ S4）
  → transform.gpu.map_forall_to_blocks       （S4 → S5）
  → gpu-lower-to-nvvm-pipeline               （S5 → NVVM）
```

顺序颠倒的后果各不相同，但都不是"效果差一点"：bufferize 放太晚会撞上 `map_forall_to_blocks` 的 "definitely fails"；并行化放在 nvvm pipeline 之后则根本没有 pass 去做它。

**组件二：`-gpu-lower-to-nvvm-pipeline` 里面到底有什么。** 官方文档页给了这条 pipeline 的定位与限制，而它的**组成**要从源码读（`mlir/lib/Dialect/GPU/Pipelines/GPUToNVVMPipeline.cpp`，取用日期 2026-09-16）。它的 common 段依次挂的是：

```text
convert-nvgpu-to-nvvm · gpu-kernel-outlining · convert-vector-to-scf ·
scf-to-control-flow · convert-nvvm-to-llvm · convert-func-to-llvm ·
expand-strided-metadata · nvvm-attach-target{chip, triple, features, optLevel} ·
lower-affine · arith-to-llvm · convert-index-to-llvm · canonicalize · cse
```

**这一串里没有任何并行化 pass。** 它做的全是**既有并行结构的降低**：把 `gpu` / `nvgpu` 的世界翻译成 NVVM 与 LLVM。这就是 §1 那句 "doesn't do GPU parallelization" 的物证——不是靠一句声明，而是靠"里面真的没有那个 pass"。

顺带记一个细节：`nvvm-attach-target` 的 `chip` 是它的一个选项（源码里 `nvvmTargetOptions.chip = options.cubinChip`），所以"指定 `sm_90`"是挂在 attach-target 这一步上的，而不是 pipeline 的另外某个开关。

**组件三：诊断入口。** `mlir-opt --pass-pipeline=...` 可以直接跑这条链，每一步之间用 `-mlir-print-ir-after-all` 看形态变化。本文配套程序做的事是同一件事的迷你版：**每个阶段都解释执行一次并比对数值**（§5 的 A 类），这样"形态变了但语义没变"是被验证的，不是被假设的。

**量级**：一条三段 elementwise 链在 S1 到 S5 之间，循环结构从 6 个嵌套 `scf.for` 收敛到 3 个 `gpu.launch`，元素级工作量始终是 `3 × 6 = 18` 次 store（本机实测，§5）。**并行化改变的是"谁来做"，不是"要做多少"。**

## 3. 实现与演进

**第一代：`-convert-linalg-to-loops` 一条路。** linalg op 直接降成 `scf.for` 嵌套，标量化到底。简单、确定，但降完就没有"这是一次并行循环"这个信息了——后续要上 GPU 只能再从循环结构里反推。

**第二代：并行循环单独一支。** `-convert-linalg-to-parallel-loops` 把 linalg op 降成并行循环，`-convert-parallel-loops-to-gpu` 再把**被映射过的** `scf.parallel` 变成 gpu launch。这一代解决的是"并行性被标量化丢掉了"的问题，代价是多了一条要维护的路径。

**第三代：Transform 方言 + `scf.forall`。** `scf.forall` 与 Transform 方言配合，把"怎么 tile、tile 成几维、映射到 block 还是 thread"变成**可以写在 IR 里的变换脚本**（`transform.structured.tile_using_forall`、`transform.gpu.map_forall_to_blocks`、`transform.gpu.map_nested_forall_to_threads`）。好处是调度决策不再埋在 C++ 的 pass 选项里，而是 IR 的一部分。

**每一代为什么要下一步**：

- 第一代 → 第二代：标量化丢掉了并行语义，而 GPU 上"并行语义"就是可执行信息本身。
- 第二代 → 第三代：`scf.parallel` 只能表达"一个并行循环"，表达不了"先 tile 再映射、映射成几维、映射到哪一级"这一串决策。Transform 方言把这些决策显式化，于是它们可以被搜索、被调优（这条线接到 `60-compilers/auto-tuning`）。

**一处稳定性警告**，官方写在 gpu 方言页的开头：

> Note: this dialect is **more likely to change than others** in the near future; use with caution.

所以本文引用的 `gpu` 侧名字（`gpu.launch`、`gpu-kernel-outlining`、`nvvm-attach-target`）都注明了取用日期。**这类名字比 `arith` / `scf` 这些基础方言的名字更可能变。**

## 4. 踩坑与失败模式

四条。前三条都是"前置条件不满足"的不同面孔——它们的共同点是：**在正确的 IR 上做正确的变换，顺序错了就全错**。

**失败模式 1：先 lowering 再 bufferize。**

- **现象**：pipeline 里 `-convert-linalg-to-loops` 排在 bufferize 之前，报错说 operand 不是 buffer 语义；或者更糟——实现者为了让它过，把检查删了，得到一个"循环体里操作 tensor 值"的畸形 IR。
- **成因**：`-convert-linalg-to-loops` 的前置条件写在 pass 文档里，而它的产出（`scf.for` + 标量 load/store）**在 tensor 语义下没有意义**——"写第 (i,j) 个元素"这句话本身就要求存储已定。
- **规避**：把 bufferize 钉在 lowering 之前，并**让检查真的存在**。本机实测：一条 `linalg` on tensor 直接进循环 lowering，会抛 `TransformFailure: 前条件是 buffer 语义，但 linalg.fill 的 operand %v… 还是 tensor<2x3xf32>`；bufferize 之后再跑同一条，通过。

**失败模式 2：拿 `-convert-linalg-to-parallel-loops` 当 forall 用。**

- **现象**：想走 `transform.gpu.map_forall_to_blocks`，于是用 `-convert-linalg-to-parallel-loops` 造"并行循环"，然后把这个结果喂给 `map_forall_to_blocks` —— 找不到 `scf.forall`，无事可做。
- **成因**：并行循环 ≠ `scf.forall`。两者是**不同的方言级构造**，产出它们的是不同的 pass（§1 第一步的两处易错点之一）。
- **规避**：要 `scf.forall` 就走 Transform 方言的 `transform.structured.tile_using_forall`。本机实测：程序里专门有一条断言检查 `scf.parallel` 与 `scf.forall` 是**两条分别可数的路径**（S3 是 3 个 `scf.parallel`、S4 是 3 个 `scf.forall`，互不混淆）。

**失败模式 3：把 `-gpu-lower-to-nvvm-pipeline` 当"一键上 GPU"。**

- **现象**：IR 还全是顺序 `scf.for`，直接挂上这条 pipeline，期望它顺手把并行化做了。结果是它什么并行化也没做——顺序代码被忠实地一路降到 NVVM。
- **成因**：官方明说它"**requires explicitly parallel IR and doesn't do GPU parallelization**"（§1）。它在语义上是一条**下降**的 pipeline，不是变换的 pipeline。
- **规避**：并行化必须在它之前完成。本机实测：把只有 `scf.for` 的 IR 交给 nvvm 这一步，程序抛 `TransformFailure: gpu-lower-to-nvvm-pipeline requires explicitly parallel IR…`；S5（已有 3 个 `gpu.launch`）则通过，被识别出的并行构造是 `['gpu.launch']`。

**失败模式 4：在 tensor 版 forall 上做 block 映射。**

- **现象**：`transform.gpu.map_forall_to_blocks` 拿到一个 region 里还带着 tensor 值的 `scf.forall`，报 "definitely fails"。
- **成因**：这是 §1 第三步引的那半句。它的成因不是"实现不完整"这么简单——**block 映射要求每个 block 明确读写哪片内存**，而 tensor 值根本没有"哪片内存"可指。
- **规避**：保证 forall 是由 bufferize 之后的 IR tile 出来的。**这条也是本文与 [02-tensor-vs-memref](02-tensor-vs-memref.md) 的扣合点**：它把"物化决策"从"性能选择"提升成了"能不能继续往下走"。本机实测见 §5 的 C1。

**一条正面清单**：本文三次 C 类断言（C1/C2/C3）对应上面四条里的三条，做法都是"**造出前置条件不满足的 IR，断言变换真的拒绝**"，并且各自配一条**对照**——把前置条件补齐后同一个变换必须通过。只断言"拒绝了"而不做对照，无法排除"这个变换本来就在别处坏了"。

## 5. 实测数据

**测量条件**：`code/03_lowering_pipeline.py` 在本机 `python3 3.13.5` + `numpy 2.4.4` 上跑出的结果，`cd code && make run` 全绿（28 条断言：A 13 / B 11 / C 4）。

> **★ 限定语（本域强制）**：下面每一个数字都是**本仓库几百行的迷你 lowering 管线**跑出来的，**不是 MLIR 的 `mlir-opt`**，也**不是任何真实 GPU 上的结果**。pass 名借用了官方的名字以便对照，实现是本仓库自己简化的（例如迷你实现的 `gpu.launch` 在语法上是一个"带多维边界的 op"，**不是**真实 `gpu.launch` 的 grid/block + region 那一套）。这些数字支持的是"前置条件链会怎样约束 pipeline 顺序"这类**机制**结论，**不支持任何关于真实 MLIR 的结论**。

**被降低的样例**：`out = (a + 1.5) * b`，形状 `2×3×f32`；S1 用**朴素** bufferization（每个 tensor 结果各一块缓冲，4 次分配）——本文关心 lowering 链本身，物化次数的优化见 [02-tensor-vs-memref](02-tensor-vs-memref.md)。

**A 类：六个阶段逐一执行并与独立 numpy 参考实现比对**（参考实现在程序里是 `(a + k) * b` 这一行纯 numpy）：

| 阶段 | 最大误差 | `memref.store` 次数 | 执行 op 数 |
|---|---|---|---|
| S0 tensor 语义 | 0.000e+00 | 0 | 6 |
| S1 memref 语义 | 0.000e+00 | 0 | 9 |
| S2 `scf.for` 嵌套 | 0.000e+00 | 18 | 91 |
| S3 `scf.parallel` | 0.000e+00 | 18 | 85 |
| S4 `scf.forall` | 0.000e+00 | 18 | 85 |
| S5 `gpu.launch` | 0.000e+00 | 18 | 85 |

三条结论直接落在这张表上：

1. **六个阶段数值全等**（最大误差 0）。S2 的顺序路径与 S3/S4/S5 的并行路径结果相同。
2. **元素级工作量守恒**：S2 到 S5 都是 18 次 store = `3 × 6`（3 个 linalg op × 每趟 6 个元素）。**并行化不改变工作量**，只改变谁来做。
3. **执行 op 数先涨后平**：S2 的 91 是最多的——标量化把 3 个 linalg op 展开成了 `scf.for` + `memref.load` + `arith.addf` + `memref.store` 的密集序列。S3~S5 的 85 略少，因为它们用单层多维循环代替了嵌套。

**B 类：各阶段的 op 数与循环结构**（本机实测）：

| 项 | 本机实测值 |
|---|---|
| S0 tensor 语义的 op 数 | 7 |
| S1 memref 语义的 op 数 | 10 |
| S1 的 `memref.alloc` 次数 | 4 |
| S2 的 `scf.for` 个数 | 6 |
| S3 的 `scf.parallel` 个数 | 3 |
| S4 的 `scf.forall` 个数 | 3 |
| S5 的 `gpu.launch` 个数 | 3 |
| 最内层循环体的迭代次数（单次） | 6 |
| S5 的 op 嵌套深度 | 3 |

**深度的对比是 S2 与 S3 之间最清楚**：S2 的嵌套深度 **4**、S3 与 S4 都是 **3**。同一个 2×3 迭代空间，`scf.for` 用"循环套循环"（多一层），`scf.parallel`/`scf.forall` 用"一个 op 带多维边界"（少一层）。这两条都是 A 类断言（断言的是"两种循环结构的形态差异"这个**成功**事实，按 CONTRIBUTING §2 归 A 类）。

**C 类：三条前置条件都必须真的挡住**（每条都配一个"补齐前置条件后必须通过"的对照）：

| 检查 | 结果 |
|---|---|
| C1 tensor 版 `scf.forall` 交给 `map_forall_to_blocks` | `TransformFailure: map_forall_to_blocks 只支持 bufferized scf.forall；发现 … 里有 tensor 值 …` |
| C1 对照：先 bufferize 再映射 | 通过 |
| C2 纯 `scf.for`（无显式并行）交给 nvvm 那一步 | `TransformFailure: gpu-lower-to-nvvm-pipeline requires explicitly parallel IR and doesn't do GPU parallelization` |
| C2 对照：S5（有 `gpu.launch`） | 通过，识别到 `['gpu.launch']` |
| C3 tensor 语义的 linalg 交给 `-convert-linalg-to-loops` | `TransformFailure: 前条件是 buffer 语义，但 linalg.fill 的 operand … 还是 tensor<2x3xf32>` |
| C3 对照：bufferize 之后的同一 linalg | 通过 |
| C4 可证伪性反证：**去掉** buffer 语义检查的实现 | 同一份 tensor 版 forall **不再失败** |

C4 那条是本域对"自造地形"类断言的硬性要求的落实：地形的构造规则（用 `check_precondition=False` 造出"忘了 bufferize 就 lower"的畸形 IR）**写死在程序里、不因结果调整**；而反证证明的是"**C1 抓的正是那个检查，不是别的东西**"——去掉检查后同一个地形就不再失败。**做不到这一点的断言，检验的是地形而不是机制。**

**第一版被实跑推翻的三项**：

1. **"module 的 body block 需要 terminator"。** 这是 [01-ir-and-dialects](01-ir-and-dialects.md) 那处修正的连锁：本文的迷你管线上第一版 `verify()` 对每个阶段都报 1 个问题，原因是 module body 被套进了 terminator 要求。
2. **"S4 是独立的第四阶段"。** 第一版把 S4 与 S5 写成了"在 S3 的模块上继续改"，于是 `map_forall_to_blocks` 就地改写了 S4，**S4 与 S5 变成了同一个对象**——B 类表里 `S4 的 scf.forall 个数` 报 0、`S5 的 gpu.launch 个数` 报 3，两个阶段实际上重合了。修法是让每个阶段**从 S0 重新跑一遍**（`pipeline(kind)`），这也更接近真实管线的行为。
3. **"每个 linalg op 的循环体迭代 1 次"是错的。** 第一版 B 类期望"顺序路径执行 6 次 store"，**本仓库迷你实现**实测是 **18** 次。原因很直接：**3 个 linalg op 各自被降成了一趟 2×3 的循环**，所以是 `3 × 6`。这个数被 B 类断言当场抓出来——如果它没被抓住，文档里就会写下一个凭直的直觉推出的错数。

## 6. 小结与延伸

- lowering 是一条**五次小变换**组成的链，不是一次大变换：`linalg on tensor → linalg on memref → 循环 → 并行 → gpu → nvvm`。
- **每一步都有前置条件**，其中最硬的一条是：**并行化要求缓冲语义**。官方原文是 `map_forall_to_blocks` 的 "Only bufferized scf.forall are currently supported" 与 "If any scf.forall with tensors is found, the transform definitely fails"。
- 这条把 [02-tensor-vs-memref](02-tensor-vs-memref.md) 与本文扣在了一起：**物化决策的位置不由性能决定，由 lowering 链的顺序决定**。
- **`-convert-linalg-to-parallel-loops` 不产出 `scf.forall`**；**`-gpu-lower-to-nvvm-pipeline` 不做并行化**。
- **`gpu` 方言官方自标 "more likely to change than others"**，引用时须注明取用日期。

**延伸**：

- 这次 bufferize 的判据本身怎么推 → [02-tensor-vs-memref](02-tensor-vs-memref.md)
- 在 lowering 的每一步上挂 pattern rewrite，以及不动点为什么可能到不了 → [04-writing-a-pass](04-writing-a-pass.md)
- 生成的代码长成什么形状（Triton / PTX / SASS）→ `30-gpu-programming/triton`、`30-gpu-programming/ptx-sass`（正文在上游，此处回链）
- 被降下来的那些循环体本身怎么写才快 → `40-kernels/gemm`、`40-kernels/attention`（正文在上游，此处回链）

## 编译与运行

```bash
cd 60-compilers/mlir/code
make 03_lowering_pipeline      # 只跑本篇
make run                       # 跑全部四篇
```

程序：`code/03_lowering_pipeline.py`（共享迷你 IR 与解释器 `code/miniir.py`，外部事实源 `code/specs.py`）。

## 参考

- MLIR Passes 索引 — `-convert-linalg-to-loops` 的前置条件原文、`-convert-linalg-to-parallel-loops`、`-convert-parallel-loops-to-gpu`：`https://mlir.llvm.org/docs/Passes/`
- MLIR GPU 方言页 — `gpu-lower-to-nvvm-pipeline` 的 "requires explicitly parallel IR and doesn't do GPU parallelization"、以及该方言的稳定性警告：`https://mlir.llvm.org/docs/Dialects/GPU/`
- llvm-project 源码 `mlir/include/mlir/Dialect/GPU/TransformOps/GPUTransformOps.td` — `MapForallToBlocks` 的 "Only bufferized scf.forall…" 与 "If any scf.forall with tensors is found, the transform definitely fails."
- llvm-project 源码 `mlir/lib/Dialect/GPU/Pipelines/GPUToNVVMPipeline.cpp` — 该 pipeline 的组成（**物证**：里面没有并行化 pass）
- llvm-project 源码 `mlir/lib/Dialect/Linalg/Transforms/Loops.cpp` — 三条 linalg→循环 pass 各自产出哪种循环（**物证**：`scf::ForOp` / `scf::ParallelOp` / `affine::AffineForOp`，且 `forall` 0 命中）
- MLIR Transform 方言 — forall 相关的 transform op 索引：`https://mlir.llvm.org/docs/Dialects/Transform/`
- 以上每条来源都在 `code/specs.py` 里逐条对应，源码类条目的取用日期为 2026-09-16

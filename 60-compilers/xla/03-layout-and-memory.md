---
title: 布局指派与内存空间
slug: xla-layout-and-memory
domain: 60-compilers
topic: xla
level: 3
difficulty: advanced
prerequisites:
  - 60-compilers/xla/02-fusion
tags: [xla, layout-assignment, kcopy, memory-space-assignment, prefetch, host-offloading, gpu]
related:
  - 60-compilers/xla/01-hlo-and-shape-inference
  - 60-compilers/inductor/02-fusion-scheduling
code:
  - code/03_layout_assignment.py
status: stable
updated: 2026-09-16
---

# 布局指派与内存空间

> XLA 里有两个名字很像的东西，方向还正好相反：**layout assignment** 在融合**之前**跑，把布局约束沿图传播、冲突处插 `kCopy`；而 **memory space assignment** 的目标是**尽量把值塞进 alternate memory**——不是"把大 buffer 赶到慢存储"。更要紧的是第三件事：**GPU 上跑的并不是后者**。

## 0. 背景与动机

### 本文与主线的关系：**② 本文是主线的代价项**

本域的主线是**物化（materialize）还是重算（recompute）**（完整口径见 [01-hlo-and-shape-inference §0](01-hlo-and-shape-inference.md)）。本主题里这条线落在"producer-consumer fusion / multi-output fusion 的判据"上。

**本文提供的是那些判据的代价项**，而且是其中两项：

1. **布局的代价**：融合体内部只有一个工作布局，钉死布局不一致的内部值要补一次布局转换——**这是一条真实存在的融合代价**（[02-fusion §3](02-fusion.md) 的"代价 ②"）。
2. **内存空间的代价**：把值放进 alternate memory 能换来带宽，但要付 prefetch / eviction 这两类拷贝的账。

所以选 ②：本文是主线的代价项，不是判据本身，也不做物化/重算的选择。

### 一个具体的对照场景

`dot` 的结果接一个 `reduce`，这条链上有一个**物理事实**：`dot` 的输出布局通常被**钉死**在一个特定次序上（真实 XLA 里 GemmRewriter 把 matmul 的输出钉在 `{n-1,...,1,0}`），而 `reduce` 为了沿连续内存归约，要求归约维**最次**。两个要求不总是能同时满足——**不能同时满足时，编译器必须插一条布局转换**。

本机实测（迷你模型，见 §5）：`dot_general.0` 被钉在 `{1,0}`，而 `reduce(dim=0)` 要求它的操作数是 `{0,1}`——于是**恰好插入 1 条 `kCopy`**；把归约维换成 `dim=1`，两条要求一致，**一条 `kCopy` 都不需要**。

## 1. 原理

### 第一步：layout assignment 的机制

官方对机制写得非常直白（`xla/service/layout_assignment.h` 的 `class LayoutConstraint` 注释，逐字）：

> During layout assignment, constraint objects are **created, prioritized, and propagated** across the HLO graph. When conflicting constraints arise, **the constraint with higher priority takes precedence**, and **layout copy instructions (`kCopy`) are inserted** where necessary to bridge incompatible layouts.

拆成四个动作：

1. **创建**：某个 op 或者硬件后端提出一条要求（"这个 `dot` 的输出必须是这个次序"）。
2. **排优先级**：每条约束带一个 `priority`，以及一个 `mandatory` 标志（不可被更低或同优先级的约束覆盖）。
3. **传播**：约束沿图的边扩散——一个 op 对**输出**的要求，会推出它对**操作数**的要求。
4. **冲突处理**：两条约束打架时高优先级者胜，**并由 `kCopy` 把不兼容的布局桥接起来**。

具体有三类约束子类（同文件，逐字）：`BufferLayoutConstraint`（约束某个 `LogicalBuffer` 的布局）、`OperandLayoutConstraint`（约束某个消费者期望的操作数布局）、`ComputationLayoutConstraint`（约束一个计算的参数与结果布局）。

**第 3 步是本文所有例子的关键**：每个 op 都要回答"给定我的**输出**布局，我的**操作数**必须是什么布局"。这条函数写错，整张图的布局就是错的，而且**不会报错**——它只是把一个不同的布局算出来而已。

### 第二步：`kCopy` 什么时候出现

两个条件同时成立时才出现：

- 某个值 `v` 的布局被**钉死**（高优先级、通常是 `mandatory`）；
- 它的某个消费者**要求**另一种布局。

此时谁赢是确定的（高优先级胜），但"赢"的代价是必须插一条 `kCopy` 把 `v` 搬到消费者要的布局上。**`kCopy` 是这条冲突的证据**，不是可有可无的清理项。

### 第三步：融合锁死布局——一条 commit 级的证据

这是本主题最有价值的一条事实，因为它**不是从注释里推的，是从提交历史里读的**。

TF commit `358fd36d`（`https://github.com/tensorflow/tensorflow/commit/358fd36d0f2c23b725bf952d7c919e7d704a45ec`，日期 **2018-03-07**）：

- 标题逐字：**`[XLA:GPU] Move layout_assignment before fusion`**
- 提交信息逐字：**「we don't emit layouts inside of fusion and tracing layouts through fusion is very hard. Changing the pipeline sidesteps this issue.」**

**这两句话给出了一个明确的机制**：融合体**内部不携带布局**（"we don't emit layouts inside of fusion"），所以融合之后再去追布局**很难**——于是把布局指派挪到融合**之前**。

**现行 pipeline 的顺序印证了这一点**（`xla/service/gpu/gpu_compiler.cc` 的 `GpuCompiler::OptimizeHloModule`，直接读源码）：

```
RunLayoutAssignmentPasses          ← layout assignment
RunLayoutNormalizationPasses
OptimizeHloPostLayoutAssignment
RunDynamicSliceFusionPasses
RunFusionPasses                    ← fusion
RunPostFusionPasses
```

**layout assignment 确实在 fusion 之前。**

由此可以下的结论只有一条：**融合体内部的值共享一个工作布局**（就是 root 的布局），因为它内部不携带 per-value 的布局。

> ⚠ **必须降级的一条（重要）**：「**因此产生本可避免的 transpose**」这句**没有一手文档/论文支持**。最接近的公开材料是 XLA 仓库的 GitHub discussion（#10、#766；作者是 XLA 团队成员，但**既不是文档也不是论文**）。
> **本文因此只写到"融合体内部不携带 layout"为止。** 再往前一步的因果推论——如果写——**必须标为「推论」**，不得写成官方结论。§5 里那条"融合让算子数变多"的实测，断言的也只是**本仓库的迷你模型在什么条件下会这样**。

### 第四步：`MemorySpaceAssignment`——**方向是「尽量塞进 alternate」**

这是控制者在这个主题上记反过的一处，也是最容易写反的一段。官方逐字（`xla/service/memory_space_assignment/memory_space_assignment.h`）：

> MemorySpaceAssignment assigns memory spaces (default or alternate) to each instruction in the module. It will **greedily try placing as as many values in the alternate memory space as possible**. It uses the heap simulator to determine the actual allocation offsets of values in the alternate memory space to account for fragmentation. **The default memory space is assumed to be large enough** to hold the values that could not be placed in the alternate memory space.

三个要点：

1. **目标是"尽量多塞进 alternate"**——alternate 是**被抢着用的那块**，不是"慢存储"。
2. **default memory 只是兜底**："assumed to be large enough"——塞不进 alternate 的值一律落这里，**没有容量上限的概念**。
3. 两个方向各有一个名字（同文件，逐字）：

> Copy Allocation: Instructions to copy an AllocationValue from one memory space to another. Used for **prefetching (default mem -> alt mem)**, and **eviction (alt mem -> default mem)**.

**prefetch 是 default → alt**（往快的那块搬），**eviction 是 alt → default**（搬回去）。**方向反过来的写法（"把大 buffer 放到慢存储以腾出快存储"）是错的。**

> ★ **引文本身有个笔误**：源码里写的是 `as as many`（重复了一个 `as`）。本文照抄并标明——引用时**要么照抄并注明，要么标明是订正后的引用**，不要默默改掉。

### 第五步：★ 两个坑

**坑 ①：GPU 上有一个同名但不同的东西。**

`xla/service/gpu/gpu_memory_space_assignment.{h,cc}` 里也叫 "memory space assignment"，但它干的是**buffer assignment 阶段的着色**，枚举只有三个成员：

```cpp
enum class MemorySpaceColor {
  kDefault = 0,
  kCollective = 1,
  kTempBuffer = 2,
};
```

它的职责是（`CreateColorer` 的注释，逐字）：集合通信 → `kCollective`；带 multimem 的 Mosaic → `kCollective`；custom call 的 `operands_memory_spaces` / `results_memory_spaces` 前端属性 → 请求的空间；**其余一律 `kDefault`**。

**它和上面那个 pass 无关**：一个是"值放 default 还是 alternate"的**放置**决策，一个是"这块 buffer 属于哪个**颜色**类"的**分类**。一个值完全可以被着色成 `kCollective`，同时被 MSA 放进 **default**。

**坑 ②：GPU 现行 pipeline 里并没有跑 MSA pass。**

`gpu_compiler.cc` 里可以看到 `MemorySpaceColor` 那个头文件被 include，但**看不到 `MemorySpaceAssignment` 这个 pass**。GPU 上"把权重放 host 做流水线"走的是另一条路：

- **`CollectivePipeliner`**：由 `debug_options.xla_gpu_enable_pipelined_host_offloading()` 控制的分支；
- **`HostOffloader`**：`OptimizeHloPostLayoutAssignment` 里的 `pipeline.AddPass<HostOffloader>(alias_info);`；
- 两者靠 **custom call** 通信，target 是 **`MoveToHost` / `MoveToDevice`**（**PascalCase**，不是 kebab-case 的 `move-to-host`）。

> ⚠ **「alternate memory 在 GPU 上具体对应哪种内存」无法用一手文件确认**——`xla/service/gpu/` 下没有文件把 MSA 的 default/alternate 映射到 GPU 的物理内存层级上。**不得断言。**

## 2. 执行映射

这一节回答"这两件事在真实系统上由谁执行、代价是多少"。

**组件一：`LayoutAssignment` 与 `LayoutConstraint`**（`xla/service/layout_assignment.h`）。约束带 `priority` 与 `mandatory` 两个字段；传播还能按 DFS 或 BFS 走（`dfs()`）。**"谁赢"完全由优先级决定**，所以"哪条约束的优先级更高"是理解任何一次布局冲突的第一问题。

**组件二：`BufferAssigner::Colorer`**（`xla/service/gpu/gpu_memory_space_assignment.h` 的 `CreateColorer`）。这是坑 ① 那个着色机制的入口：它返回一个 colorer，在 buffer assignment 期间给值上色。**它与 MSA 是两条独立的路径**——一个在 GPU 的 buffer assignment 里，一个在（CPU 侧的）内存空间指派 pass 里。

**组件三：`HostOffloader` 与 `CollectivePipeliner`**（`xla/service/gpu/gpu_compiler.cc`）。这是 GPU 上做 host 卸载的实际路径，也是坑 ② 的答案。它们通过 `MoveToHost` / `MoveToDevice` 这两个 custom call 与外部通信，并由 `xla_gpu_enable_pipelined_host_offloading` 这个开关控制是否启用。

**一个数值量级**：本机迷你实现里的 alternate memory 容量设为 **4096 字节**（这是**我们自己的口径**，不是 XLA 的值）。这个数足够小，好让"塞不下"这件事真的发生——§5 里那个 **4096 字节**的值 `v3` 单独就超过了整个容量，而 **5120 字节**的 default 占用**超过了 alt 容量却没有报错**（因为 default "assumed to be large enough"）。真实的 alternate memory 是片上的、容量远小于 HBM 的那一级，具体是哪一级**本文不断言**（理由见 §1 第五步的警告）。

## 3. 实现与演进

**layout assignment 的演进是一条"把顺序挪对"的历史**，而不是"把算法改得更聪明"：

- 2018 年之前，layout assignment 在 fusion **之后**跑；
- commit `358fd36d`（2018-03-07）把它挪到 fusion **之前**，理由是"融合体内部不携带布局、追踪布局很难"；
- 此后 fusion 在**已经定好布局**的图上做决策——**融合体内部只有一个工作布局**这条性质就此固定下来。

这次改动的直接好处（提交信息自己说的）是"allow code simplification and opens up new optimization"，并且"mostly perf-neutral"。**代价是**：融合决策拿不到"如果我知道这里会融合，本来可以选更合适的布局"这个信息。§5 里那条"严格 layout 约束下融合反而变多一个算子"的迷你实测，就是这条性质的一个**可观察后果**——但再次强调，那测的是**本仓库的模型**。

**memory space 这一侧的演进则是"多出一条路"**：

- MSA（default / alternate 两态 + prefetch/eviction）是**为 CPU 一侧的异步拷贝**设计的机制；
- GPU 一侧需要的是"权重常驻 host、按需搬到 device"的**流水线**，这件事由 `CollectivePipeliner` + `HostOffloader` 做，用的是 custom call 而不是 MSA 的两态模型；
- 两者**同名不同物**（坑 ①），而 GPU 现行 pipeline **不跑 MSA**（坑 ②）。

## 4. 踩坑与失败模式

### 坑 1：把 CPU 侧的 MSA 当成 GPU 的机制

- **现象**：读到一个叫 "GPU memory space assignment" 的头文件，就以为 GPU 上跑的是那套 default/alternate + prefetch/eviction 的机制；于是去找 GPU 上"alternate memory 是哪一级"，找不到答案。
- **成因**：**两个东西同名**。`gpu_memory_space_assignment.{h,cc}` 干的是 buffer assignment 期间的**着色**，枚举只有 `kDefault` / `kCollective` / `kTempBuffer` 三个；MSA 那个 pass 管的是**值的放置**。而 GPU 的现行 pipeline 里**根本没有 MSA pass**。
- **规避**：看到 "memory space" 先问一句——**是着色还是放置？** 着色看 `MemorySpaceColor`；放置看 `MemorySpaceAssignment`。GPU 上做 host 卸载的入口是 **`CollectivePipeliner` + `HostOffloader`**，通信走 `MoveToHost` / `MoveToDevice` 两个 PascalCase 的 custom call。

### 坑 2：把 MSA 的方向写反

- **现象**：把 MSA 描述成"把大 buffer 放到慢存储里，以腾出宝贵的快存储"。
- **成因**：这个直觉对**容量**成立，对 **MSA 的优化目标**不成立。官方逐字是"greedily try placing **as many values in the alternate memory space as possible**"——**要尽量多塞进 alternate**；default 只是"assumed to be large enough"的兜底。
- **规避**：把两个方向与两个名字绑死记：**prefetch = default → alt**，**eviction = alt → default**。这两句都能在 `memory_space_assignment.h` 里逐字查到（本文 §1 第四步抄了）。

### 坑 3：以为 `kCopy` 是"多余的清理项"

- **现象**：看到 `kCopy` 就想去掉它（"这只是个没优化掉的拷贝"）。
- **成因**：`kCopy` 是**两条约束冲突的产物**，不是遗漏的优化。去掉它意味着两条约束中的一个必须让步——而"谁让步"由**优先级**决定。LayoutAssignment 之后确实有 pass 清理**重复或 NOP** 的 `kCopy`（`gpu_compiler.cc` 的注释原话："The LayoutAssignment pass may leave behind kCopy instructions which are duplicate or NOPs, so remove them with algebraic simplification and CSE"），但清理的是"重复或 NOP 的"，**不是所有**。
- **规避**：看到 `kCopy` 先问"哪两条约束冲突了、谁的优先级更高"，而不是"怎么把它删掉"。

### 坑 4：以为"融合会引入额外 transpose"是官方结论

- **现象**：在文档里写"XLA 把 layout assignment 放在 fusion 之前，因此产生了本可避免的 transpose"。
- **成因**：**前半句有 commit 级证据，后半句没有。**可引的事实只到"融合体内部不携带 layout"（`358fd36d` 的提交信息逐字）。后半句最接近的公开材料是一段 GitHub discussion，**既不是文档也不是论文**。
- **规避**：写前半句、把后半句标成**推论**或干脆不写。本文 §1 第三步末尾按这个口径处理了。

## 5. 实测数据

**测量条件**：`code/03_layout_assignment.py` 在本机 `python3 3.13.5` + `numpy 2.4.4` 上跑出的结果，`cd code && make run` 全绿（21 条断言：A 6 / B 7 / C 8）。

> **★ 限定语（本域强制）**：下面每一个数字都是**本仓库的迷你 layout / 内存实现**跑出来的，**不是 XLA**，**不是硬件**。迷你实现借用了 XLA 的术语（layout constraint / 优先级 / `kCopy` / default & alternate / prefetch / eviction）以便对照，**传播规则、优先级取值、alternate 容量、活跃区间上限全都是我们自己定的**。这些数字支持的是"这套约束传播与贪心放置可以怎样被实现与观察"这类**机制**结论，**不支持任何关于真实 XLA 的结论**。
>
> XLA 侧的那组事实（`358fd36d` 的提交信息、`gpu_compiler.cc` 的 pass 顺序、MSA 的逐字方向、`MemorySpaceColor` 的三个成员、`MoveToHost`/`MoveToDevice`）只作为**规格值**出现在 §1–§2 与 `code/specs.py` 里，**不作为程序断言对象**。

**`kCopy` 的插入**（地形：`dot` 后接 `reduce`）：

| 条件 | `dot_general.0` 的布局 | `reduce.0` 的布局 | 插入的 `kCopy` |
|---|---|---|---|
| `reduce(dim=0)`：要求操作数 `{0,1}`，`dot` 被钉在 `{1,0}` | `{1,0}` | `{0}` | **1 条**（`kCopy(dot_general.0: {0,1} -> {1,0})`） |
| `reduce(dim=1)`：要求操作数 `{1,0}`，与钉死布局一致 | `{1,0}` | `{0}` | **0 条** |

独立校验器（只看最终的布局表 + copy 列表、重新走一遍图）报告 **0 个问题**：每条边都满足"布局一致 **或** 有 `kCopy`"。

**MSA 的计数**（值表在源码里先于结果冻结；alternate 容量 `4096` 字节，活跃区间上限 `4`）：

| 值 | def | last use | 字节数 | 活跃区间 | 落在哪 | 原因 |
|---|---|---|---|---|---|---|
| `v0` | 0 | 3 | 1024 | 3 | alternate | 活跃区间 3 < 4 |
| `v1` | 1 | 6 | 1024 | **5** | **default** | 活跃区间 ≥ 上限 |
| `v2` | 2 | 3 | 2048 | 1 | alternate | 1024+2048 ≤ 4096 |
| `v3` | 3 | 4 | **4096** | 1 | **default** | 3072+4096 > 4096，塞不下 |
| `v4` | 4 | 5 | 512 | 1 | **alternate** | 3072+512 ≤ 4096 |

| 项 | 本机实测值 |
|---|---|
| alternate 里的值个数 / 字节数 | 3 / **3584** |
| default 里的值个数 / 字节数 | 2 / **5120** |
| prefetch 条数（default → alt） | 3 |
| eviction 条数（alt → default） | 1 |

**两条 C 类断言的行为**：

- **default 没有容量上限**：落在 default 里的 **5120 字节 > 4096 字节**（alt 的容量），程序不报错——对应官方那句 "The default memory space is assumed to be large enough"。
- **贪心是"尽量多塞"而不是"一次失败就停"**：`v3` 因为太大落进 default 之后，**更小的 `v4` 仍然进了 alternate**。反证也跑了：把贪心改成"一失败就停"，`v4` 就不再进 alt（程序里两条断言都在）。

**两个坑在迷你实现里的形态**：

| 坑 | 本机输出 |
|---|---|
| ① 着色是**另一套** | `{'all-reduce.0': 'kCollective', 'add.0': 'kDefault'}`；着色枚举 3 个成员 |
| ② GPU pipeline 不含 MSA | 从 `gpu_compiler.cc` 抄来的顺序映射后为 `['layout_assignment', 'layout_normalization', 'host_offloader', 'dynamic_slice_fusion', 'fusion', 'post_fusion']`——**没有 `memory_space_assignment`**；而迷你 CPU pipeline 里有 |

**第一版被实跑推翻的两项**：

1. **`reduce` 的操作数布局需求写错了秩**。第一版直接把"归约维 + 输出的次→主顺序"拼起来，但**输出各维的编号是压缩过的**（被归约的维已经不在），于是算出来的布局秩比操作数少一维。修法是先用 `remap` 把压缩编号还原成操作数的原编号。
2. **`kCopy` 的判据一开始只写了一半**。第一版只比较"操作数的钉死布局 vs 消费者要求"，忘了**同一个值可能被多个消费者要求不同布局**这条分支——程序里的独立校验器把它暴露了出来，于是补上了 `notes` 那条记录。

## 6. 小结与延伸

- **layout assignment 的机制**：约束被**创建、排优先级、沿图传播**；冲突时高优先级胜，**并插入 `kCopy`** 来桥接不兼容的布局。
- **融合锁死布局有一条 commit 级证据**：TF commit `358fd36d`（2018-03-07）`[XLA:GPU] Move layout_assignment before fusion`，理由逐字「we don't emit layouts inside of fusion and tracing layouts through fusion is very hard」。现行 `gpu_compiler.cc` 的 `OptimizeHloModule` 里 layout assignment 确实在 fusion **之前**。
- **⚠ 但「因此产生本可避免的 transpose」只能标为推论**——没有一手文档/论文支持。
- **MSA 的方向是「尽量塞进 alternate」**：default 只是"assumed to be large enough"的兜底；**prefetch 是 default → alt**，**eviction 是 alt → default**。
- **两个坑**：① GPU 上有**同名但不同**的 buffer 着色机制（`kDefault` / `kCollective` / `kTempBuffer`，共 3 个成员）；② **GPU 现行 pipeline 不跑 MSA**，host 卸载走 `CollectivePipeliner` + `HostOffloader`，custom call target 是 **`MoveToHost` / `MoveToDevice`**（PascalCase）。
- **「alternate memory 在 GPU 上对应哪种内存」不得断言**——没有一手文件能确认。

**延伸**：

- 融合判据的三层结构（本文是它的代价项） → [02-fusion](02-fusion.md)
- HLO 的语义与 `dot_general` 的形状规则 → [01-hlo-and-shape-inference](01-hlo-and-shape-inference.md)
- 融合的字节数收益怎么推（上游推导） → [40-kernels/elementwise-fusion/02](../../40-kernels/elementwise-fusion/02-fusion-and-traffic.md)
- 框架侧怎么决定哪些中间结果不落内存 → [inductor/02-fusion-scheduling](../inductor/02-fusion-scheduling.md)

## 编译与运行

```bash
cd 60-compilers/xla/code
make 03_layout_assignment      # 只跑本篇
make run                       # 跑全部三篇
```

程序：`code/03_layout_assignment.py`（配套 `code/minilayout.py`（布局指派 + MSA + 着色）、`code/minihlo.py`（迷你 HLO），外部事实源 `code/specs.py`）。

## 参考

- layout assignment 的机制（约束的创建 / 优先级 / 传播 / `kCopy` 插入）与三类约束子类：`https://github.com/openxla/xla/blob/main/xla/service/layout_assignment.h`
- `[XLA:GPU] Move layout_assignment before fusion`，2018-03-07，提交信息逐字：`https://github.com/tensorflow/tensorflow/commit/358fd36d0f2c23b725bf952d7c919e7d704a45ec`
- `OptimizeHloModule` 的 pass 顺序（layout assignment 在 fusion 之前）、`HostOffloader`、`CollectivePipeliner` 与 `xla_gpu_enable_pipelined_host_offloading`：`https://github.com/openxla/xla/blob/main/xla/service/gpu/gpu_compiler.cc`
- MSA 的方向（"greedily try placing as as many values in the alternate memory space as possible"；default "assumed to be large enough"）、prefetch 与 eviction 的方向：`https://github.com/openxla/xla/blob/main/xla/service/memory_space_assignment/memory_space_assignment.h`
- GPU 上同名但不同的 `MemorySpaceColor`（`kDefault` / `kCollective` / `kTempBuffer`）与 `CreateColorer`：`https://github.com/openxla/xla/blob/main/xla/service/gpu/gpu_memory_space_assignment.h`
- custom call target `MoveToHost` / `MoveToDevice`（PascalCase）：`https://github.com/openxla/xla/blob/main/xla/service/memory_annotations.h`
- `IsMoveToHostWithDynamicUpdateSlice` / `IsMoveToDeviceWithDynamicSlice`：`https://github.com/openxla/xla/blob/main/xla/service/host_offload_utils.h`
- 以上每条 URL 与抄录的原文都在 `code/specs.py` 里逐条对应，XLA 源码类条目取用日期 **2026-09-16**（会漂移的坐标）

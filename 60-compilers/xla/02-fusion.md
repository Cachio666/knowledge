---
title: 融合：kind、判据与代价模型
slug: xla-fusion-kinds
domain: 60-compilers
topic: xla
level: 3
difficulty: advanced
prerequisites:
  - 60-compilers/xla/01-hlo-and-shape-inference
  - 60-compilers/inductor/02-fusion-scheduling
tags: [xla, fusion, multi-output-fusion, fusion-kind, cost-model, gpu]
related:
  - 60-compilers/xla/03-layout-and-memory
  - 40-kernels/elementwise-fusion/02
code:
  - code/02_fusion_kinds.py
status: stable
updated: 2026-09-16
---

# 融合：kind、判据与代价模型

> 关于 XLA 融合，最常见的两个说法都是错的：**「multi-output fusion 是某一种 fusion kind」**（它不是——root 是一个 `kTuple`），以及**「XLA 的融合有一张判据清单」**（没有——判据是「预算 + 合法性 + **代价模型**」三层，最后一层是一个性能模型，它可以否决一个完全合法的融合）。

## 0. 背景与动机

### 本文与主线的关系：**① 本文是主线的判据来源**

本域的主线是**物化（materialize）还是重算（recompute）**（完整口径见 [01-hlo-and-shape-inference §0](01-hlo-and-shape-inference.md)）。本主题里这条线落在"**producer-consumer fusion / multi-output fusion 的判据**"上。

而**本文正是那些判据本身**：上游 `40-kernels/elementwise-fusion/02` 推出"融合的收益上界是链长 `k`，且只在中间结果往返 HBM 时成立"；本文回答的是它上面那一层——**编译器凭什么决定融不融**。所以选 ①：本文是主线的判据来源。

### 两个必须先纠正的通行误解

**误解一：multi-output fusion 是 `FusionKind` 的一个成员。**

不是。`HloInstruction::FusionKind` 一共只有**四个**成员：

```cpp
enum class FusionKind {
  kLoop,
  kInput,
  kOutput,
  kCustom,
};
```

XLA 自己的 TODO 把这件事写得非常直白（逐字，来源见 `code/specs.py`）：

> this scheme as written **doesn't work for multi-output fusion**, where the primary node is **never actually the root** (which is **a `kTuple` instruction** that gathers the multiple outputs of the fusion).

**multi-output fusion 描述的是一种「组的形态」**（root 是 `kTuple`），不是一个 kind。

**误解二：`kInput` / `kOutput` 是"输入融合 / 输出融合"，多输出用它俩区分。**

不是。两者的区别只有一条——**primary node 是不是 root**：

| kind | primary node 的位置 | 额外要求 |
|---|---|---|
| `kInput` | **是** root | — |
| `kOutput` | **不是** root | "one operand buffer of the fusion instruction be able to alias the output buffer" |
| `kLoop` | 是 root，但规定了实现 | 不生成"未融合版本看起来的样子"，而是**一次生成 root 的一个元素** |
| `kCustom` | — | 后端特有、不符合以上模式的 |

**与「多输出」无关。** 一个单输出的 fusion 同样可以是 `kOutput`。

这两条误解有代价：按误解一实现，你会去枚举一个不存在的 kind；按误解二实现，你会在"多输出"的路径上去查 primary node 的位置，而那两件事根本不搭界。

## 1. 原理

### 第一步：`kInput` / `kOutput` 到底在编码什么

XLA 的注释给出了这套编码的**用途**（逐字）：

> An HloInstruction's FusionKind helps us find the kFusion instruction's **primary node**, and can also affect how we generate code.

也就是说，kind 是给**代码生成器**看的提示：告诉它"这个 fusion 里哪条指令是主角"。找到主角之后，代码生成做两件事：把主角的操作数按融合体的内容**算出来**（而不是从内存读），把主角的输出**转发给消费者**（而不是写回内存）。

- 主角**是** root → 输出转发到融合体的出口就是最终结果 → `kInput`。
- 主角**不是** root → 主角的输出还要经过融合体内部的后续计算才到出口 → `kOutput`。此时"谁是主角"是有歧义的，所以额外要求"某个 operand buffer 能与 output buffer alias"来消除歧义。

### 第二步：multi-output fusion 的两类与两条硬约束

`xla/backends/gpu/transforms/multi_output_fusion.h` 的头部注释给出两类（逐字）：

> 1) Fusion of **sibling** ops reduces memory bandwidth requirements, because common input parameters have to be read only once.
> 2) Fusion of **producer-consumer** ops reduces memory bandwidth requirements by saving one read from memory.

两类可以**组合**。sibling 融合省的是"共同输入读两次"，producer-consumer 融合省的是"中间结果读一次"。

两条硬约束（同文件，逐字）：

> * Sibling fusion (1) requires **at least one op to be a `kFusion`**.
> * Sibling fusion (1) does not fuse `kInput` fusions with `kLoop` fusions, i.e. **the fusion kinds must match**.

第一条的意思是：sibling 融合的双方里**至少有一个已经是 fusion 了**——这解释了为什么 multi-output fusion 是一个**靠后**的 pass（前面的 producer-consumer 融合已经造出了一批 fusion）。第二条是 kind 的**匹配性**约束。

root 的形态（同文件，逐字）：multi-output fusion 的 root 是一个 tuple，多个输出装在里头，消费者用 `GetTupleElement` 把它们取出来。

### 第三步：判据的真实形态——**三层，不是一张清单**

**不存在一份官方文档给出 XLA 的融合判据清单。** 判据是 backend-specific 的 C++ 实现，拒绝条件以 `FusionDecision` 的**字符串**散在源码里。

以 GPU 的 `ProducerCandidateIsFusible`（`multi_output_fusion.cc`）为例，把它的调用次序读出来，就是三层：

```
层 1  预算      FusionFitsInBudget(...)                     ← operands/outputs 数量 + shared memory
层 2  合法性    ShapesCompatibleForMultiOutputFusion(...)
                OperandReachableFromProducer(...)
                ← 结构性的：形状兼容、不成环、kind 匹配、至少一个 kFusion …
层 3  代价模型  ProducerConsumerMergedTooLarge(...)  → "will generate too large IR"
                GpuPerformanceModel::EstimateRunTimesForMultiOutputFusion(...)
                if (t.time_fused > t.time_unfused) → "will execute slower if fused"
```

**第三层是判据链里最容易被漏掉的一层**：它不检查合法性，它检查**划不划算**——一个完全合法、形状兼容、预算也够的融合，**照样可以被否掉**，理由字符串就是 `"will execute slower if fused"`。

**可引的具体值**：

| 项 | 值 | 来源（`code/specs.py` 里逐条对应） |
|---|---|---|
| operands + outputs 上限 | **`MaxOperandsAndOutputsPerFusion() = 96`** | `xla/service/gpu/gpu_fusible.h` |
| `FusionFitsInBudget` 管什么 | 数量**以及** shared memory 占用 | 同上 |
| 代价模型拒绝理由 ① | `"will execute slower if fused"` | `xla/backends/gpu/transforms/multi_output_fusion.cc` |
| 代价模型拒绝理由 ② | `"will generate too large IR"` | 同上 |
| 另一条硬件型拒绝 | `"can't fuse multiple DUSs"` | 同上 |

### 第四步：成本模型的成本项是什么

把第二层的"省字节"与第三层的"否决"放在一起，才能看出真实判据的形状。**融合的收益与代价是两类不同的东西**：

- **收益**：省掉中间结果的一次写 + 一次读（这正是上游 `40-kernels/elementwise-fusion/02` 推的那部分，本主题只代入，不重推）。
- **代价 ①（重算）**：如果被融进融合体的那个值的**消费者不止一个**，融合体里只算得出一份，其它消费者还得从原来的位置拿——于是这份计算要**再做一遍**。**这就是主线的另一半：不物化，就得重算。**
- **代价 ②（布局转换）**：融合体内部只有**一个工作布局**，任何钉死布局与它不一致的内部值都要补一次转换。详见 [03-layout-and-memory](03-layout-and-memory.md)。
- **代价 ③（IR 规模）**：融合体太大直接拒。

### 第五步：另一套同名的枚举

**GPU emitter 侧还有一套 `EmitterFusionKind`**（`xla/service/gpu/hlo_fusion_analysis.h`），**9 个成员**：

```cpp
enum class EmitterFusionKind {
  kLoop, kCustomFusion, kTriton, kReduction, kTranspose,
  kConcatenate, kScatter, kCuDnn, kSort,
};
```

**它与 `HloInstruction::FusionKind` 不是一回事**：

| | `FusionKind`（4 个） | `EmitterFusionKind`（9 个） |
|---|---|---|
| 挂在哪 | `HloInstruction` 上 | fusion 的分析结果上 |
| 回答什么 | primary node **是不是 root** | 这个 fusion 由**哪个 emitter** 出码 |
| 多出来的成员 | — | `kTriton` / `kCuDnn` / `kSort` / `kReduction` / `kTranspose` / `kConcatenate` / `kScatter` / `kCustomFusion` |

两套枚举**唯一的交集是 `kLoop`**——一个名字，两套语义。把 `kTriton` 当成一种 `FusionKind` 去写 HLO，是走不通的：HLO 里根本没有这个成员。

## 2. 执行映射

这一节回答"三层判据在真实系统上由谁执行、代价是多少"。

**组件一：`GpuPerformanceModel`**（`xla/service/gpu/model/gpu_performance_model.{h,cc}`）。它是第三层的主体：给定一个候选融合，它估出**融合后**与**未融合**两种运行时间，然后比较。这个模型不是布尔判据——它自己也要读设备描述、算子 profile 表（同目录下的 `hlo_op_profiles_data.h`、`default_matmul_perf_table.txtpb` 等）来给出时长。

**组件二：`FusionFitsInBudget` 与 `FusionInfoCache`**（`xla/service/gpu/gpu_fusible.h`）。第一层的主体。同一个头文件里还提供了一批"这个 op 能不能当某种融合的 root"的谓词（`IsInputFusibleReduction` / `IsInputFusibleScatter` / `IsNestableVariadicReduction` …）——**这些才是"合法性"的实际形态**：不是一个集中的 `is_fusible()`，而是一族按 op 种类分派的谓词。

**组件三：`HloFusionAnalysis`**（`xla/service/gpu/hlo_fusion_analysis.h`）。它在**融合之后**运行：把 fusion 体分析成 `EmitterFusionKind` 之一，交给对应的 emitter。所以"用哪套枚举"取决于你站在哪一侧——**融合决策阶段用 `FusionKind`，出码阶段用 `EmitterFusionKind`**。

**一个数值量级**：`MaxOperandsAndOutputsPerFusion()` 是 **96**。把这个数放到上下文里：它管的是"能传进一个 kernel 的参数个数 + 输出个数"，本机迷你实现在这个边界上确实是在 96 处放行、97 处拒绝（§5）。本文 §5 里那张 `dot` 地形只用了 **4** 个 operands+outputs——距离上限极远，所以**否决它的不是预算层**。

## 3. 实现与演进

**朴素做法：一组静态布尔判据。** 最容易想到的形态是"能不能融"写成一张清单：形状兼容、不成环、没有 DUS、没超预算……照着实现出来的东西**能跑**，但会犯一个具体的错：**它会把"合法但更慢"的融合也融掉。**

**主流做法：把"划不划算"也做成一层。** 这就是 §1 第三步那三层。演进的方向很清楚——从"**能不能**"走向"**值不值**"。证据是拒绝理由字符串的措辞本身：`"will execute slower if fused"` 不是一条规则，而是**一次比较的结果**。

**为什么这一层非有不可**：因为"融合省字节"这件事**不是无条件的**。存在两类反例，都真实存在：

1. **重算**：生产者有多个消费者时，融进一个消费者意味着给其它消费者重算一遍（§1 第四步的代价 ①）。
2. **布局转换**：融合体内部只有一个工作布局，钉死布局不一致的内部值要补转换（代价 ②）。

**前沿：GPU 侧的融合判据正在被"模型化"。** 同目录下已经出现 `sol_gpu_cost_model.{h,cc}`、`gpu_dot_fusion_cost_model.{h,cc}`、`gpu_indexing_performance_model.{h,cc}` 这些**专门的**性能模型——融合决策从"读一组常量"转向"喂给一个越来越细的代价模型"。这也解释了为什么**不要指望找到一份判据清单**：判据正在从代码里的布尔表达式迁移进模型里。

## 4. 踩坑与失败模式

### 坑 1：照着"判据清单"实现融合

- **现象**：把形状兼容、不成环、不超预算这些检查都实现对了，程序仍然会把**明显不该融**的图融掉。
- **成因**：判据的第三层是**代价模型**，它不在"清单"上——一份静态布尔清单天然表达不了"这次融了会更慢"。本文 §5 的 F3 地形就是活例：预算层放行（`operands+outputs=4 ≤ 96`）、合法性层放行（结构检查通过），**前两层全绿**，第三层才否决。
- **规避**：把融合判据实现成**三层顺序调用**，并且**让最后一层能返回"不划算"**。判据不是"能不能"，是"能不能 **且** 值不值"。

### 坑 2：把 multi-output fusion 当成一个 kind

- **现象**：在 `FusionKind` 的枚举里找不到多输出对应的成员；或者反过来说"multi-output 用的是 `kOutput`"。
- **成因**：`FusionKind` 只有四个成员，它编码的是"primary node 是不是 root"，**没有**"输出个数"这一维。XLA 自己的 TODO 说得很直白：这套 scheme 对 multi-output fusion 不适用，因为那里的 root **是 `kTuple`**。
- **规避**：把 multi-output 当成**组的一种形态**（可有多个 root，由一个 tuple 汇聚），而不是一个 kind。本文 §5 的 F4 地形就是这么建模的——它给 `Group` 传 `multi_output=True`，而 `kind` 仍然是 `kLoop`。

### 坑 3：把 `EmitterFusionKind` 当成 `FusionKind` 的扩展

- **现象**：想给一个融合"标上 `kTriton`"，发现 HLO 里没有这个成员。
- **成因**：GPU emitter 侧另有一套 9 个成员的枚举，**只有 `kLoop` 一个名字与 `FusionKind` 重合**。两套枚举服务于两个阶段。
- **规避**：判断"primary node 是不是 root"用 `FusionKind`；判断"由哪个 emitter 出码"用 `EmitterFusionKind`。名字重合的那一个（`kLoop`）**在两套里语义不同**，尤其要小心。

### 坑 4：融合省了字节，却因为重算输掉

- **现象**：一个 producer 有两个消费者，把它融进其中一个之后，整体**更慢**。
- **成因**：中间结果省下的一次写 + 一次读（设中间结果为 `S` 字节，省 `2S`），换来的可能是**把整条生产者链重算一遍**。若生产者的操作数本身就很大（比如是个 `dot`，两个操作数各 `S` 字节、结果也是 `S` 字节），重算的代价是 `3S`——**比省下的 `2S` 还多**。
- **规避**：这一条**没有一个可以照抄的阈值**——它取决于具体的形状与后端。可操作的判据是：**"被融的值有没有第二个消费者"是必须先查的一件事**，它决定了这次融合要不要付重算的代价。本机实测（§5 的 F3）：`fused=7168 > unfused=6144`，其中重算项占 **3072 字节**。

## 5. 实测数据

**测量条件**：`code/02_fusion_kinds.py` 在本机 `python3 3.13.5` + `numpy 2.4.4` 上跑出的结果，`cd code && make run` 全绿（22 条断言：A 6 / B 8 / C 8）。

> **★ 限定语（本域强制）**：下面每一个数字都是**本仓库的迷你融合实现**跑出来的，**不是 XLA**，**不是 JAX/TF**，**也不是硬件**。迷你实现借用了 XLA 的术语（`FusionKind` / `multi-output` / `will execute slower if fused` …）以便对照，**阈值与判据形式都是我们自己定的**（例如布局转换的条数、重算项的算法）。这些数字支持的是"三层判据可以怎样被实现与观察"这类**机制**结论，**不支持任何关于真实 XLA 的结论**——尤其**不支持**"XLA 在这个模型上会/不会融合"。
>
> 真实 XLA 的那组值（`96`、两条拒绝理由字符串、两条硬约束）只作为**规格值**出现在 §1 与 `code/specs.py` 里，**不作为程序断言对象**。

**三层判据在四张冻结地形上的判定**（每张地形的构造在源码里**先于任何结果**写死）：

| 地形 | 候选组 | 各层判定 | 结果 |
|---|---|---|---|
| F1 | `transpose.0 + add.0` | 预算 `operands+outputs=3 ≤ 96` → 合法性通过 → 代价模型 `fused=72 ≤ unfused=120` | **Allow** |
| F2 | `multiply.0 + add.0`（`64×64`） | 预算 `4 ≤ 96` → 合法性通过 → 代价模型 `fused=65536 ≤ unfused=98304` | **Allow** |
| F3 | `dot_general.0 + add.0`（`16×16`） | 预算 `4 ≤ 96` → 合法性通过 → 代价模型 **`fused=7168 > unfused=6144`** | **Forbid**（`"will execute slower if fused"`） |
| F4 | `multiply.0 + add.0`（兄弟、multi-output，`64×64`） | 预算 `5 ≤ 96` → 合法性通过（`already_fusion=True`，2 个 root）→ 代价模型 `fused=81920 ≤ unfused=98304` | **Allow** |

**F3 的拒绝只发生在第三层**：前两层的返回值都是 `allowed=True`（程序里对这一点单独断言了一条）。重算项在本机测得 **3072 字节**。

**预算层的边界**：

| 构造 | operands + outputs | 判定 |
|---|---|---|
| 32 个互不相干的 `add`（64 操作数 + 32 root） | **96** | Allow（闭区间上界） |
| 再加 1 个常量 | **97** | Forbid（`"operands(64)+outputs(33)=97 超出上限 96"`） |

**布局约束对算子数的影响**（C1，F1 地形）：

| 条件 | 融合前算子数 | 融合后算子数 |
|---|---|---|
| 严格 layout 约束（本迷你版默认） | 2 | **3（反而变多）** |
| 放松 layout 约束（反事实） | 2 | **1** |

> ⚠ **这一格的口径要说清楚**：断言的是「**本仓库的迷你模型在什么条件下会这样**」，**不是「XLA 会这样」**。「融合因此产生本可避免的 transpose」这条因果**没有一手文档/论文支持**（最接近的公开材料是 XLA 仓库的 GitHub discussion，**既不是文档也不是论文**）——**若写，只能标为「推论」**。本文因此只把它作为**本仓库迷你模型的性质**呈现，并给出放松约束后它不再成立的反证（程序里两条断言都跑）。

**两套枚举的成员对照**（本机从迷你实现里读出的，成员名照抄 `code/specs.py` 的一手来源）：

| | 成员数 | 成员 |
|---|---|---|
| `FusionKind` | **4** | `kLoop` `kInput` `kOutput` `kCustom` |
| `EmitterFusionKind` | **9** | `kLoop` `kCustomFusion` `kTriton` `kReduction` `kTranspose` `kConcatenate` `kScatter` `kCuDnn` `kSort` |

只在 `EmitterFusionKind` 里出现的成员有 **8 个**（交集只有 `kLoop`）。

**C 类的失败行为**（都必须真的发生）：

| 构造 | 结果 |
|---|---|
| 兄弟融合、组内没有 `kFusion` | Forbid：`sibling fusion requires at least one op to be a kFusion` |
| 兄弟融合、一个 `kInput` 一个 `kLoop` | Forbid：`sibling fusion 不融合 kInput 与 kLoop，i.e. the fusion kinds must match` |
| 把 `parameter` 放进融合组 | Forbid：`parameter 不可融合（它不是计算）` |
| 严格 layout 下融合 `transpose` | 算子数 `2 → 3`（变多） |
| 放松 layout 下同一融合 | 算子数 `2 → 1`（反证） |

**第一版被实跑推翻的三项**：

1. **代价模型的成本项一开始就写错了**：第一版让布局转换也计入 HBM 流量，于是 F1 那条地形被代价模型否决——**而 C1 那条断言要求这次融合是"被允许的"**（否则算子数根本不会变）。修法是把转换从"流量"里移出去、只计入 IR 规模：转换发生在片内，流量模型不该收它的钱。这条是**跑出来的**。
2. **"重算"这一项第一版根本不存在**：没有它，任何融合在这个流量模型下都稳赚（省掉一次中间往返），第三层永远不会否决——**C3 那条断言无论如何都过不了**。加上重算项之后 F3 才真的被拒。
3. **地形的第一版用了"形状相同的 `add` 但不写 `broadcast_dimensions`"**，而迷你实现当时要求 `len(broadcast_dimensions) == rhs.rank`，于是**合法的同形状加法直接报错**。修法是给迷你实现补上一条约定：`broadcast_dimensions` 为空表示"两个操作数形状完全相同"。

## 6. 小结与延伸

- **`FusionKind` 只有 4 个成员**（`kLoop` / `kInput` / `kOutput` / `kCustom`），编码的是 **primary node 是不是 root**——`kInput` 是、`kOutput` 不是（并要求某个 operand buffer 能与 output buffer alias）。
- **multi-output fusion 不是一个 kind**：它是"root 是一个 `kTuple`、汇聚多个输出"的一种**组形态**。
- **判据不是清单，是三层**：预算（`MaxOperandsAndOutputsPerFusion() = 96`、shared memory）→ 合法性（结构性约束）→ **代价模型**（`GpuPerformanceModel`，可以否决一个合法的融合）。
- **两条逐字的硬约束**：sibling 融合要求"at least one op to be a `kFusion`"；sibling 融合不把 `kInput` 与 `kLoop` 混在一起（"the fusion kinds must match"）。
- **GPU emitter 侧另有一套 `EmitterFusionKind`**（9 个成员），与 `FusionKind` 不是一回事——**唯一重合的名字 `kLoop` 在两套里语义不同**。
- **融合可能因为两类代价而失败**：重算（生产者有第二个消费者）与布局转换（融合体只有一个工作布局）。

**延伸**：

- 布局指派与融合谁先谁后、以及 MSA 的方向 → [03-layout-and-memory](03-layout-and-memory.md)
- 形状与 `dot_general` 的语义（本文的前置） → [01-hlo-and-shape-inference](01-hlo-and-shape-inference.md)
- 融合省下的字节数怎么推（上游推导，本文只代入） → [40-kernels/elementwise-fusion/02](../../40-kernels/elementwise-fusion/02-fusion-and-traffic.md)
- 框架侧怎么组织 fusion group（相邻主题） → [inductor/02-fusion-scheduling](../inductor/02-fusion-scheduling.md)

## 编译与运行

```bash
cd 60-compilers/xla/code
make 02_fusion_kinds      # 只跑本篇
make run                  # 跑全部三篇
```

程序：`code/02_fusion_kinds.py`（配套 `code/minifuse.py`（三层判据 + 两套枚举）、`code/minilayout.py`（布局指派）、`code/minihlo.py`（迷你 HLO），外部事实源 `code/specs.py`）。

## 参考

- `HloInstruction::FusionKind` 的四个成员、`kInput`/`kOutput`/`kLoop`/`kCustom` 的语义、以及那条 multi-output fusion 的 TODO：`https://github.com/openxla/xla/blob/main/xla/hlo/ir/hlo_instruction.h`
- `MaxOperandsAndOutputsPerFusion() = 96`、`FusionFitsInParameterLimit`、`FusionFitsInBudget`：`https://github.com/openxla/xla/blob/main/xla/service/gpu/gpu_fusible.h`
- multi-output fusion 的两类、两条硬约束、tuple root：`https://github.com/openxla/xla/blob/main/xla/backends/gpu/transforms/multi_output_fusion.h`
- 三层的调用次序与拒绝理由字符串（`"will execute slower if fused"` / `"will generate too large IR"` / `"can't fuse multiple DUSs"`）、`GpuPerformanceModel` 的两处调用点：`https://github.com/openxla/xla/blob/main/xla/backends/gpu/transforms/multi_output_fusion.cc`
- `EmitterFusionKind` 的 9 个成员：`https://github.com/openxla/xla/blob/main/xla/service/gpu/hlo_fusion_analysis.h`
- `IsPhysicallyTransposing` 的注释（"Fusing such ops may lead to uncoalesced data access and may thus not be beneficial."）：`https://github.com/openxla/xla/blob/main/xla/service/gpu/gpu_fusible.h`
- 以上每条 URL 与抄录的原文都在 `code/specs.py` 里逐条对应，取用日期 **2026-09-16**（会漂移的坐标）

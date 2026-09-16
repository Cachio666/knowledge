---
title: HLO 与形状推断
slug: xla-hlo-and-shape-inference
domain: 60-compilers
topic: xla
level: 2
difficulty: intermediate
prerequisites:
  - 60-compilers/inductor/02-fusion-scheduling
tags: [xla, hlo, stablehlo, shape-inference, dot-general, jax, broadcasting]
related:
  - 60-compilers/xla/02-fusion
  - 60-compilers/xla/03-layout-and-memory
  - 40-kernels/elementwise-fusion/02
code:
  - code/01_hlo_shape_inference.py
status: stable
updated: 2026-09-16
---

# HLO 与形状推断

> HLO 是 XLA 与前端之间的中间表示，它的每一条指令都由**形状**定义边界：形状推错了，后面所有的融合、布局、内存决策都建在错的地基上。而 `dot_general` 这一条指令身上集中了三个极易混淆的点——**三套不同的字段名**、**batch 维的两种语义**、**标量退化时与 `matmul` 的分歧**。

## 0. 背景与动机

### 本主题的主线

本域（`60-compilers`）的五个主题共享同一条主线：

> **每一次 lowering、每一次变换、每一个 pass，都在做同一个权衡：物化（materialize）还是重算（recompute）。**

口径先写死（避免和上游的用法混起来）：

- **物化 = 为中间结果分配独立的存储。** 它**不指存储层级**——在寄存器、共享内存还是全局内存里，都算物化。
- **重算 = 根本不分配存储**，把计算直接替换进消费者的表达式。

这条主线在这五个主题里长这样：

| 主题 | 主线在那里长什么样 |
|---|---|
| `mlir` | `tensor`（值语义，未指定存储）vs `memref`（缓冲语义，已物化）；bufferization 就是那个决策点 |
| `tvm` | 三态：默认（物化在根作用域）/ `compute_at`（仍物化，但缩到一次迭代的切片）/ `compute_inline`（不物化 = 重算） |
| `inductor` | fusion group 的划分 = 决定哪些中间结果不落内存 |
| **`xla`** | **producer-consumer fusion / multi-output fusion 的判据** |
| `auto-tuning` | 搜索空间里 tile 大小这一维，本质就是「物化多大一块」 |

这条线接上 `40-kernels/elementwise-fusion/02` 的结论——那里推出「融合的收益上界是链长 `k`，且只在中间结果往返 HBM 时成立」。**本主题要回答的是上游的那个问题**：编译器靠什么判据决定融不融，以及为什么它**经常选择不融**。

一个具体的对照场景：`out = (a + b) * 1.5` 这条链，如果逐条指令执行，`a+b` 这个中间结果要先写进 HBM 再读回来（对 `f32[N]` 就是 `4N` 字节写出 + `4N` 字节读入）；如果融成一个 kernel，这份往返就没了。但编译器要判断的远不止"融了省字节"——判据见 [02-fusion](02-fusion.md)，代价项见 [03-layout-and-memory](03-layout-and-memory.md)。

### 本文与主线的关系：**③ 本文不涉及主线**

`xla/01` 讲的是 **HLO 的语义与形状推断**，它决定的是「这张图合不合法、每个值的形状是什么」，而**物化决策发生在它之后的融合阶段**。形状推错了，融合判据连可用的输入都没有——但形状推断本身不做物化/重算的任何选择。所以本文选 ③。选 ③ 不是失败：硬把形状推断说成"也是一种物化决策"才是反模式。

### 为什么形状推断值得单开一篇

因为它是**编译器最容易错的地方**，而错误只在很远的下游才暴露：

- HLO 里同一个概念有**三套字段名**（proto / 文本格式 / StableHLO op attribute），写混了不会报错，只会静默地按错误的维去做乘法；
- 同一个 `dot_general` 能表达两种**语义完全不同**的批量矩阵乘：batch 维要么成对匹配，要么根本不存在；而 `jnp.dot` 与 `jnp.matmul` 在这件事上的行为**不一样**；
- 广播的边界情形（大小为 1 的维、形状完全相同的"空广播"）写错了，程序照样跑出一张图，只是那张图算的是别的东西。

§1 把这三件事的确切规则推一遍。它们不是细节：§4 的失败模式全部是"越界但没人拦"。

## 1. 原理

### 第一步：HLO 是什么

HLO（High Level Operations）是 XLA 的输入 IR。两件性质决定了它的形状推断长什么样：

1. **HLO 是 SSA 的**，而且**每条指令恰好产生一个值**——所以"形状"可以挂成指令自身的属性，不需要另建一张值表。
2. **HLO 是纯的**（大部分情况下）：没有可变状态，值由一条指令产生、经依赖边流向消费者。

于是 shape inference 就是一个**逐指令的局部函数**：给定操作数的形状与指令的属性，算出结果的形状。本仓库的迷你实现把它写成 `infer_shape(op)`，一条指令一个分支，没有别的路径。

### 第二步：三套维度命名——**同一个概念，三种写法**

这是本主题最具体的易错点。`dot_general` 的维度编号字段在三个场合用的是**三套不同的名字**：

| 场合 | contracting | batch |
|---|---|---|
| HLO / StableHLO **proto** | `lhs_contracting_dimensions` / `rhs_contracting_dimensions` | **`lhs_batch_dimensions`** |
| HLO **文本格式** | `lhs_contracting_dims` / `rhs_contracting_dims` | **`lhs_batch_dims`** |
| **StableHLO op attribute** | `lhs_contracting_dimensions` / `rhs_contracting_dimensions` | **`lhs_batching_dimensions`** |

来源逐条在 `code/specs.py`：proto 侧是 `xla/xla_data.proto` 的 `message DotDimensionNumbers`；文本格式是 `hlo_instruction.cc` 的 `DotDimensionNumbersToString()`；StableHLO 侧是 `docs/spec.md` 的 `### dot_general` 条目。

三个坑按易中招程度排序：

1. **`batch` 与 `batching`**：proto 叫 `*_batch_dimensions`，StableHLO 叫 `*_batching_dimensions`。两个都合法、都常见，**写混了照样能编译**（如果你的工具链不检查名字）。
2. **`dimensions` 与 `dims`**：同一条指令，proto 里是长名、文本里是短名。
3. **三套名字指的是同一件事**：维度编号都是**该操作数自身 shape 的 0-based 下标**。这一点三套完全一致——`code/01_hlo_shape_inference.py` 里的 `rename_dims()` 就只是改名，值一个不动。

### 第三步：`dot_general` 的语义

`dot_general` 的结果形状由一条公式给出（StableHLO 规范的 C12）：

```
shape(result) = dim(lhs, lhs_batching_dimensions)
              + dim(lhs, lhs_result_dimensions)
              + dim(rhs, rhs_result_dimensions)
```

其中 `lhs_result_dimensions` 是 **lhs 里既不是 batching 也不是 contracting 的那些维，保持它们在 lhs 里的原有顺序**。

约束条件（C1–C10，非量化部分）在 `code/specs.py` 里逐条抄了，最要紧的四条是：

- **C1**：两侧 batching 维**个数**相等；
- **C2**：两侧 contracting 维**个数**相等；
- **C3/C4**：同一操作数内，batching 与 contracting 维**不得重复**；
- **C9/C10**：对应的 batching 维、contracting 维**大小必须相等**。

★ **C9 是理解本文所有分歧的钥匙**：`dot_general` 的 batch 维**要求两侧大小相等**，它**不做广播**。

### 第四步：`jnp.dot` 与 `jnp.matmul` 的差异

JAX 官方文档对 `jax.numpy.dot` 写得很清楚（原文见 `code/specs.py`）——它与 `jax.numpy.matmul` 有**两处**不同：

> ① 任一操作数是标量时，`dot` 退化为逐元素乘（等价于 `jnp.multiply`），而 `matmul` **直接报错**。
> ② 操作数超过 2 维时，**batch 维是被"堆叠"（stacked）而不是"广播"（broadcast）**。

第二条展开讲，因为它是"读了文档仍会意外"的那种。取 `a` 形状 `(3,2,4)`、`b` 形状 `(3,4,1)`：

| 写法 | 结果形状 | 为什么 |
|---|---|---|
| `jnp.dot(a, b)`（= `numpy.dot`） | **`(3,2,3,1)`** | 前导维**成对堆叠**：a 的第 0 维与 b 的第 0 维各自摊开，做**笛卡尔积**而不是配对 |
| `jnp.matmul(a, b)`（= `numpy.matmul`） | **`(3,2,1)`** | 前导维按 numpy 广播规则**配对**并对齐 |

两边**元素数差 3 倍**（`18` vs `6`，本机 numpy 2.4.4 复算，见 §5）：同一对输入、同一条"批量矩阵乘"的直觉，结果大小差 3 倍。

把这两条映射回 HLO：

- **`matmul` 的语义** ⇔ `dot_general` 带 `lhs_batch_dims={0}` / `rhs_batch_dims={0}`（两侧 batch 大小都是 3，满足 C9）；
- **`jnp.dot` 的语义** ⇔ `dot_general` **不带 batch 维**，全部前导维都算 `result_dimensions`——于是 a 的第 0 维与 b 的第 0 维各自出现在结果里，**互不相干**。

所以 HLO 这一层并没有"两种 batch 语义"，只有**一种**（成对匹配 + C9 强制相等）；`jnp.dot` 那种堆叠行为，在 HLO 里是靠**不声明 batch 维**实现的。

### 第五步：广播的两条边界

HLO 的逐元素二元运算用 `broadcast_dimensions` 描述"右操作数的第 i 维对应结果的第几维"。两条容易写错的边界：

1. **大小为 1 的维可以广播**，大小不等且非 1 则**不合法**。判据是"operand 的维 `d` 要么等于 1，要么等于结果对应维"，两者都不满足就是错误。
2. **`broadcast_dimensions` 必须严格递增**（它是一张有序映射，不是集合）。
3. `broadcast` 这条指令额外多一层：结果的前 `len(sizes)` 维由 `sizes` 决定，`dimensions` **不得指向这几维**。

## 2. 执行映射

这一节回答"形状推断在真实系统里发生在哪、代价是什么"。

**组件一：`HloInstruction` 与 `HloComputation`**（`xla/hlo/ir/hlo_instruction.h`）。形状直接挂在指令上（`HloInstruction::shape()`），而 `dot_general` 的维度编号挂在指令的 `dot_dimension_numbers` 属性里。这意味着**形状推断的输入就是这条指令自身 + 它操作数的形状**，不需要跨指令的全局分析——这是 HLO 是 SSA 且每条指令一个值带来的直接好处。

**组件二：`HloVerifier`**。HLO 在 pipeline 的多个位置被重新校验，形状不一致会在那里被抓住。**但校验只在你跑 verifier 的地方发生**——本仓库迷你版对应的函数是 `verify()`：它在**全模块**上重推一遍每个指令的形状，并和指令自己声明的形状对照。本机实测：`graph` 那张图上的问题是 **0 个**（见 §5）。

**组件三：StableHLO 的 portable artifact 与兼容性窗口**。StableHLO 是"HLO/MHLO 启发的、向后兼容的算子集"（官方首段逐字），它的定位是"不同 ML 框架与 ML 编译器之间的可移植层"。可引的硬数字来自 `docs/compatibility.md`（基于 v1.0 兼容性 RFC）：

- **5 年向后兼容**：旧版序列化的 portable artifact 用新版反序列化，语义不变——前提是两次构建的 commit 相距 **不到 5 年**；
- **2 年向前兼容**：新版序列化的用旧版读——前提是相距**不到 2 年**，且没有用到新特性。

**这个保证只对 portable artifact 生效**；pretty-print 形式与 bytecode 形式**不在范围内**；数值精度明确列为 out of scope。

> ⚠ **措辞留余地**：「HLO 本身没有兼容性保证」这句，我们能查到的只有**对照证据**（StableHLO 明文给出窗口、HLO 侧没有对应的窗口文档），**没有找到 HLO 侧的官方否认声明**。所以本文只说"StableHLO 给出了 5 年/2 年窗口"，不说"HLO 完全没有保证"。

**一个数值量级**：一篇形状推断的输入规模有多大？以本文 §5 那张小图为例——4 条指令（另有 1 个常量）、1 个参数，全部值合计 **300 字节**（`f32` 每个元素 4 字节）。真实的模型会大六到九个数量级，但**形状推断的代价与值的大小无关**：它只做维度的符号运算，不做任何搬数。

## 3. 实现与演进

**朴素做法：让每条 op 自己算形状。** 这是最自然的起点，也是本仓库迷你实现的做法。它的代价是：**没有统一入口**时，"右操作数怎么广播"这类规则会在十几个 op 里各写一遍，写法稍有出入就是静默的错。

**主流做法：形状推断成为 op 定义的一部分。** 在真实系统里这件事走得更远：

- **MLIR 路线**：`InferTypeOpInterface` 把"给定操作数类型推出结果类型"做成一个**可运行期查询的契约**——op 自己实现它，基础设施按接口调用。（接口与 Trait 的区别见 [mlir/01](../mlir/01-ir-and-dialects.md)。）
- **HLO 路线**：形状是 `HloInstruction` 的一个字段，由建图者（前端）给；推导规则散在 `shape_inference` 相关代码里，**verifier 负责事后抓不一致**。本文采取的是这一路线。

**前沿：StableHLO 把"语义"写成了规范。** 关键变化是：形状推断的判据从"某份 C++ 实现怎么做"变成了"**规范第 C1–C20 条怎么规定**"。`docs/spec.md` 的 `dot_general` 一节把每条约束编号列出，并给出参考伪代码（转置 → reshape → matmul → reshape）。本仓库的迷你实现就是照这条参考路径写的——**§5 里那条"迷你实现 vs numpy"的对拍能成立，靠的正是这一步**。

## 4. 踩坑与失败模式

### 坑 1：把 `jnp.dot` 当成 `jnp.matmul` 用

- **现象**：`a` 是 `(3,2,4)`、`b` 是 `(3,4,1)`，以为结果是 `(3,2,1)`，实际拿到 `(3,2,3,1)`；下游的 reshape 报错，或者更糟——下游正好也能广播，于是**结果静默地算错了**。
- **成因**：`jnp.dot` 的前导维是 **stacked**（笛卡尔积式堆叠），不是 broadcast。同一对操作数，两种语义的元素数可以差 3 倍（本机 numpy 2.4.4：`18` vs `6`）。
- **规避**：批量矩阵乘统一用 `jnp.matmul` / `jnp.einsum`；要写 `dot` 时先把形状打印出来，确认前导维的语义。在 HLO 这一层，**batch 维只在需要"成对匹配"时才声明**——声明了就会受 C9（大小必须相等）约束。

### 坑 2：batch 维大小不等时以为会广播

- **现象**：`dot_general(lhs f32[3,2,4], rhs f32[1,4,5])` 直接抛 `DimSizeError`（本机迷你实现的原话："第 0 个 batch 维大小不等（lhs[0]=3 vs rhs[0]=1）"）。
- **成因**：`numpy.matmul` 会把这个 `3` 与 `1` 广播成 `3`，而 StableHLO 的 **C9 要求两侧 batch 维大小相等**——`dot_general` **不做广播**。也就是"上游框架能跑的写法，落到 HLO 上不合法"。
- **规避**：要广播就**显式**在 HLO 里插一条 `broadcast`，把 `rhs` 的 batch 维先撑到 3；不要在 `dot_general` 上指望隐式广播。

### 坑 3：广播的退化情形——"大小相同的维"与"大小 1 的维"不是一回事

- **现象**：写 `add(lhs f32[2,3], rhs f32[4]), broadcast_dimensions={1}`，期望广播，实际被拒（本机错误："operand 的维 0（大小 4）与结果第 1 维（大小 3）不兼容"）。反过来，两个形状完全相同的操作数若按位对齐地写了 `broadcast_dimensions` 却**顺序写反**（`{1,0}`），也会被拒（"broadcast_dimensions 必须严格递增且不重复"）。
- **成因**：合法性判据是"operand 的维 `d` **要么等于 1、要么等于结果对应维**"，两个条件都不满足即错；且 `broadcast_dimensions` 是**有序映射**，必须严格递增。
- **规避**：同形状的二元运算在有歧义时**直接省略** `broadcast_dimensions`（本迷你实现把空列表定义为"形状完全相同"），不要硬写一个自己都没把握的映射。

### 坑 4：`dot_general` 的维度编号重复（C3）

- **现象**：一个值既被当作 batch 维又被当作 contracting 维，形状推断"算得出结果"但对不上任何真实语义。
- **成因**：C3/C4 要求同一操作数内 batching 与 contracting **互不重复**；这条约束在纯形状算术里**不需要**——你必须显式检查它，否则它永远不会被触发。
- **规避**：形状推断函数里对每个操作数做一次"两个集合求交"的检查。本机实测：迷你实现在这条上抛 `DimNumberError: lhs 的 batching 与 contracting 维有重叠：[1] ∩ [1]`。

### 坑 5：标量进 `dot_general`——`dot` 退化、`matmul` 报错、`dot_general` 两者都不是

- **现象**：`jnp.dot(a, 2.0)` 合法（退化为逐元素乘，形状不变）；`jnp.matmul(a, 2.0)` 报错；而 `dot_general` 遇到 rank-0 的操作数**必须拒绝**（C6：`0 <= lhs_contracting_dimensions < rank(lhs)`，rank=0 时没有任何合法编号）。
- **成因**：三条规则来自三套不同的规范，**它们不一致是正常的**。
- **规避**：不要用"能不能跑"来判断语义；把这三条当作**三份独立的规格**分别查。

## 5. 实测数据

**测量条件**：`code/01_hlo_shape_inference.py` 在本机 `python3 3.13.5` + `numpy 2.4.4` 上跑出的结果，`cd code && make run` 全绿（25 条断言：A 7 / B 8 / C 10）。

> **★ 限定语（本域强制）**：下面每一个数字都是**本仓库几百行的迷你 HLO 实现**跑出来的，**不是 XLA**，**也不是硬件**。迷你实现借用了 HLO 的术语（`dot_general` / `broadcast_dimensions` / `parameter` …）以便对照，**判据是我们自己简化的**。这些数字支持的是"这套形状规则可以怎样被实现与对拍"这类**机制**结论，**不支持任何关于真实 XLA 的结论**。
>
> 其中"与手写 numpy 参照的误差"一项，参照物是 **numpy**（不是 XLA、不是 JAX）——`jnp.dot` 的语义只在 `code/specs.py` 里以**官方文档引文**出现，本程序**不对它下断言**。

**形状推断与手写 numpy 参照的对拍**（`graph` 那张小图，4 条指令 + 1 个常量）：

| op | 迷你实现给出的 shape | numpy 参照给出的 shape | 最大绝对误差 |
|---|---|---|---|
| `add.0` | `(2,3)` | `(2,3)` | 0 |
| `broadcast.0` | `(4,2,3)` | `(4,2,3)` | 0 |
| `transpose.0` | `(3,4,2)` | `(3,4,2)` | 0 |
| `reduce.0` | `(3,4)` | `(3,4)` | 0 |
| 全模块 `verify()` 的问题数 | 0 | — | — |

**`(3,2,4) dot (3,4,1)` 的两种语义**（本机 numpy 复算 + 迷你实现）：

| 写法 | 结果形状 | 元素数 | 与 numpy 的最大绝对误差 |
|---|---|---|---|
| 无 batch 维的 `dot_general` | `(3,2,3,1)` | 18 | `3.58e-07` |
| 有 batch 维的 `dot_general` | `(3,2,1)` | 6 | 0 |
| `np.dot(a,b)`（参照） | `(3,2,3,1)` | 18 | — |
| `np.matmul(a,b)`（参照） | `(3,2,1)` | 6 | — |

`3.58e-07` 不是实现错误：fp32 下"先转置 reshape 再 matmul"与 numpy 的直接实现**累加次序不同**，末位会差一个 ulp 量级。程序里的容差因此取 `1e-5`，并把这个误差打印出来而不是藏起来。

**结构计数**（同一张小图）：

| 项 | 本机实测值 |
|---|---|
| op 总数（含常量） | 5 |
| 参数个数 | 1 |
| 全部值的字节数 | 300 |

**C 类的失败行为**（每条都真的抛了异常，不是"理论上会"）：

| 构造 | 抛出的异常 |
|---|---|
| 广播退化（rhs 维既不等于 1 也不等于结果维） | `BroadcastError` |
| `broadcast_dimensions` 非严格递增 | `BroadcastError` |
| `broadcast` 的 `dimensions` 指向 `sizes` 覆盖的前几维 | `BroadcastError` |
| `dot_general` 的 contracting 与 batch 维重叠（C3） | `DimNumberError` |
| 两侧 batch 维个数不等（C1） | `DimSizeError` |
| contracting 维大小不等（C10） | `DimSizeError` |
| 维度编号越界（C5） | `DimNumberError` |
| batch 维大小不等（C9） | `DimSizeError` |
| 标量操作数进 `dot_general`（rank 0 无合法 contracting 维） | `DimNumberError` |

**可证伪性反证**：把迷你实现里的 contracting 维**故意对调**（一个经典笔误），A 类的"与 numpy 一致"判据立刻失败（改错后的实现直接抛 `DimSizeError`）。这条反证在程序里跑了，见 `make run` 的第 7 节。

**第一版被实跑推翻的三项**（都是"跑出来才知道"，不是读文档读出来的）：

1. **`broadcast` 的 `dimensions` 语义写反了一格**。第一版按"操作数各维对应结果的前几维"写，于是 `verify()` 在一张**完全合法**的图上报了错。正确语义是"结果的前 `len(sizes)` 维由 `sizes` 决定，`dimensions` 不得指向它们"。
2. **`num_elements` 我自己算错了**。第一版 B 类表格里把 `(3,2,3,1)` 的元素数写成 `72`，实跑是 `18`（我漏了最后一维是 1）。B 类断言的对照值因此改了，程序的对照断言把这一项抓了出来。
3. **`dimension` 枚举的默认假设错了**：迷你实现里参数 op 的 `opcode` 取自名字的首段，参数名 `p0` 会让 `opcode` 变成 `p0` 而不是 `parameter`，解释器直接 `KeyError`。

## 6. 小结与延伸

- **HLO 的每一条指令恰好产生一个值**，所以形状就是指令的属性，shape inference 是一个逐指令的局部函数。
- **`dot_general` 的维度字段有三套写法**：proto 的 `*_batch_dimensions`、HLO 文本的 `*_batch_dims`、StableHLO op attribute 的 `*_batching_dimensions`。**三套指同一件事**，编号都是该操作数自身的 0-based 下标。
- **C9 是 `dot_general` 与 `matmul` 的分界线**：前者要求两侧 batch 维大小**相等**，不广播；`jnp.dot` 的"stacked"行为在 HLO 里是靠**不声明 batch 维**实现的。
- **同一对操作数** `(3,2,4)` 与 `(3,4,1)`：`jnp.dot` 给 `(3,2,3,1)`（18 个元素），`jnp.matmul` 给 `(3,2,1)`（6 个元素）。
- **标量退化三套规则不一致**：`dot` 退化为乘法、`matmul` 报错、`dot_general` 拒绝。
- **StableHLO 给出 5 年向后 / 2 年向前的兼容窗口**，但只对 portable artifact 生效，且数值精度 out of scope。

**延伸**：

- 形状确定之后，编译器靠什么判据决定融不融 → [02-fusion](02-fusion.md)
- 融合与布局指派谁先谁后、以及这个顺序的后果 → [03-layout-and-memory](03-layout-and-memory.md)
- 融合在框架侧是怎么被调度的（上游主题） → [inductor/02-fusion-scheduling](../inductor/02-fusion-scheduling.md)
- 融合省下的字节数是怎么推的（上游推导） → [40-kernels/elementwise-fusion/02](../../40-kernels/elementwise-fusion/02-fusion-and-traffic.md)

## 编译与运行

```bash
cd 60-compilers/xla/code
make 01_hlo_shape_inference      # 只跑本篇
make run                         # 跑全部三篇
```

程序：`code/01_hlo_shape_inference.py`（配套迷你 HLO 实现 `code/minihlo.py`，外部事实源 `code/specs.py`）。

## 参考

- HLO 的 `FusionKind` 与 `HloInstruction` 的形状字段：`https://github.com/openxla/xla/blob/main/xla/hlo/ir/hlo_instruction.h`
- `DotDimensionNumbers` 的 **proto** 字段名（`*_batch_dimensions`）：`https://github.com/openxla/xla/blob/main/xla/xla_data.proto`
- HLO **文本格式**的短名（`*_batch_dims` / `*_contracting_dims`）：`https://github.com/openxla/xla/blob/main/xla/hlo/ir/hlo_instruction.cc` 的 `DotDimensionNumbersToString()`
- StableHLO 规范 §`dot_general`（op attribute 用 `*_batching_dimensions`；C1–C20 约束；参考伪代码）：`https://github.com/openxla/stablehlo/blob/main/docs/spec.md`
- StableHLO 兼容性窗口（5 年 / 2 年，只对 portable artifact 生效）：`https://github.com/openxla/stablehlo/blob/main/docs/compatibility.md`
- `jnp.dot` 与 `jnp.matmul` 的两处差异（stacked 而非 broadcast；标量退化）：`https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.dot.html`
- 以上每条 URL 与抄录的原文都在 `code/specs.py` 里逐条对应，XLA/StableHLO 源码类条目取用日期 **2026-09-16**（会漂移的坐标）

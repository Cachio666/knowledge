---
title: IR 结构与方言体系
slug: mlir-ir-and-dialects
domain: 60-compilers
topic: mlir
level: 2
difficulty: intermediate
prerequisites: []
tags: [mlir, ir, dialect, region, block, ssa, interface, trait]
related:
  - 60-compilers/mlir/02-tensor-vs-memref
  - 60-compilers/mlir/04-writing-a-pass
  - 40-kernels/gemm/02-shared-memory-tiling
code:
  - code/01_ir_structure.py
status: stable
updated: 2026-09-16
---

# IR 结构与方言体系

> MLIR 的 IR 只有四个概念：Operation、Region、Block、Value，而它们之间是**递归**关系——Operation 里可以装 Region，Region 里装 Block，Block 里又装 Operation。四层递归加上"Value 恰有一个定义者"，就同时买到了两样东西：**任何抽象层次都用同一套结构表示**，以及**任何一层都能独立做 SSA 化简**。

## 0. 背景与动机

一个具体的对照场景：把 `out = (a + b) * 1.5` 这段计算从"张量表达式"一路送到 GPU 上执行，中间至少要经过四种完全不同的形态——

1. **张量层**：`out` 是一个值，存储在哪儿还没决定；
2. **缓冲层**：`out` 是某块 `memref` 上的写操作，存储定了；
3. **循环层**：`out[i][j]` 是两层 `scf.for` 里的一次 `memref.store`；
4. **设备层**：这两层循环被映射到了 GPU 的 block 与 thread 上。

在传统编译器里，这四层通常由**四套互不相干的数据结构**表示（表达式树 → 三地址码 → 循环 IR → 目标指令），于是每往下一层就要写一个新的 pass 框架、一个新的 visitor、一套新的模式匹配。更糟的是**跨层的优化做不了**：比如"这个中间张量只有一次使用，可以就地复用缓冲"这条判据，需要同时看到张量层的使用链和缓冲层的分配点，而它们分属两套数据结构。

MLIR 的答案是：**不要四套结构，只要一套递归结构**。同一套 `Operation / Region / Block / Value` 同时表示上面四层，区别只在**用哪个方言的 operation**。

这就是"可扩展 IR"的含义，而它的代价与收益都很具体：

- 收益：**任何一层都能用同一套 pass 基础设施**——包括同一个 pattern rewrite driver（见 [04-writing-a-pass](04-writing-a-pass.md)）。本机实测：本仓库的迷你实现里，一棵同时含张量层、循环层与设备层概念的小函数，用同一个递归遍历就能一次数清全部结构（op 15 个、region 6 个、block 5 个），树最多嵌套 4 层（见 §5）。
- 代价：**结构上的约束必须由 IR 自己把关**。递归结构允许你写出 `Operation` 里嵌 `Region`、`Region` 里嵌 `Block`、`Block` 里再嵌 `Operation` 的任意组合，其中一部分组合是非法的——而非法 IR 不会自己报错，它会**一路沉默地跑到底**，直到某个后端崩掉。

§1 把这套结构的**确切边界**推一遍。这些边界不是细节：§4 的三条失败模式全部是"越界但没人拦"。

## 1. 原理

先约定四个词，本文只用这四个（定义全部取自 MLIR Language Reference，来源见 `code/specs.py`）。

- **Operation**：一次操作。有零到多个 operand、零到多个 result、**零到多个 Region**。
- **Region**：一个有序的 Block 列表。**可以是空的**。
- **Block**：一个有序的 Operation 列表，可以带参数。**非空**。
- **Value**：一个 SSA 值。

**第一步：四层递归的定义。**

官方原文（LangRef §High-Level Structure）：

> Operations are contained in Blocks and Blocks are contained in Regions. Operations may also contain regions, enabling hierarchical structures to be represented.

把这句话按包含关系展开，就是一条闭环：

```text
Operation ──contains──▶ Region ──contains──▶ Block ──contains──▶ Operation ──▶ …
```

**没有第五个概念，也没有"根"的特殊结构**。一个编译单元就是一棵 `builtin.module` 开头的 Operation 树：module 是一个 Operation，它的 Region 里有一个 Block，Block 里装着函数；函数也是一个 Operation，它的 Region 里装基本块。

递归的收益立刻可见：**"在某处插一层循环"和"在某处插一个函数"是同一件事**——都是一个 Operation 里多了一个 Region。后面 [03-progressive-lowering](03-progressive-lowering.md) 讲的 lowering 链条，本质就是沿着这条递归逐层替换 Operation。

**第二步：Value 的唯一性——SSA 的地基。**

官方原文（同一页）：

> Each Value is the result of exactly one Operation or Block Argument, and has a Value Type defined by the type system.

"exactly one"这四个字是整篇文章里最要紧的约束。它排除了两件事：

1. **一个 Value 不能有两个定义者**。这条一旦破了，任何"沿着 def-use 链往回走"的分析（常量折叠、就地分析、死代码删除）都会走到歧义上。
2. **一个 Value 不能没有定义者**。引用一个不存在的东西，是 IR 构造错误而不是运行时错误——必须在构造/校验期就抓住。

**第三步：四条边界的确切语法。**

"Operation 里装 Region"这句话太粗，实际语法允许的组合比它宽，也比它严。官方语法（LangRef §Operations 与 §Regions）：

```text
region-list ::= "(" region ("," region)* ")"
region      ::= "{" entry-block? block* "}"
entry-block ::= operation+
block       ::= block-label operation+
```

逐条读这些产生式里的量词，得到四条边界：

| # | 边界 | 依据 | 反直觉之处 |
|---|---|---|---|
| 1 | **Operation 可以有零个** Region | `region-list` 里是 `*`，允许零次 | 大多数 op（`arith.addf` 之类）就是零个 region |
| 2 | **Operation 可以有两个以上** Region | 同上，`("," region)*` 不限个数 | `scf.if` 就有两个：then 与 else |
| 3 | **Region 可以是空的**（零个 Block） | `region ::= "{" entry-block? block* "}"` 两处都是可选/零次 | 「Region 至少含一个 Block」是**错的** |
| 4 | **Block 非空**，且必须以 terminator 结尾 | `block ::= block-label operation+` | `+` 要求至少一个 op |

第 3 条与第 4 条合起来看很有意思：**空是 Region 的属性，不是 Block 的属性**。一个"还没填内容的 Region"是合法的（`test.empty_region { }`），但一个"还没填内容的 Block"是非法语法。

第 4 条还有个例外，而且这个例外会咬人：

> The last operation in a block must be a terminator operation. A region with a single block may opt out of this requirement by attaching the NoTerminator on the enclosing op. **The top-level ModuleOp is an example of such an operation** which defines this trait and whose block body does not have a terminator.

也就是说，**module 的 body block 不需要 terminator**。这条的实践后果是：一个"Block 必须有 terminator"的校验器如果把 module 的 body 也算进去，它会**把每一个合法模块都判为非法**。§4 第 2 条失败模式就是这个坑的变体。

顺带记一个术语：官方把 Region 分成两类——**SSACFG region**（描述块间控制流）与 **Graph region**（不要求块间控制流，且"may only contain a single basic block"）。上引的 `NoTerminator` 豁免是走 trait 那条路，不要把它和"Graph region"混为一谈：两者是**不同的机制**，官方文档里也是两处分开写的。

**第四步：方言——把"同一套结构"变成"同一套结构上的分层"。**

结构统一了，抽象层次靠什么区分？靠 **Operation 的名字**。MLIR 用**方言**（dialect）给 operation 名字加命名空间：`arith.addf`、`linalg.matmul`、`scf.for`、`gpu.launch` 分属四个方言。官方对各方言的定位（逐条原文，见 `code/specs.py`）：

| 方言 | 官方定位 |
|---|---|
| `arith` | basic integer and floating point mathematical operations（含位运算/移位/转换/比较） |
| `math` | operations **beyond simple arithmetics**（超越函数），对 scalar / vector / tensor 逐元素 |
| `scf` | structured control flow；「Unstructured control flow operations are located in the **cf** dialect」；「Typically, scf is lowered to **cf** and then lowered to some final target like LLVM or SPIR-V」 |
| `vector` | n-D SIMD 向量层 |
| `gpu` | middle-level abstractions for launching GPU kernels |

`vector` 与 `gpu` 各有一条官方警告，值得单独抄下来：

- **`vector` 不负责自动向量化。** 该方言文档里有一节的标题就叫 **「The Big Out-Of-Scope Piece: Automatic Vectorization」**。所以"用了 vector 方言所以会自动向量化"是反向的：vector 方言提供的是**表达**向量操作的层，把标量代码变成向量代码那一步不归它管。
- **`gpu` 方言自身不稳定。** 官方原话：「Note: this dialect is **more likely to change than others** in the near future; use with caution.」

`arith` 与 `math` 的分界是"simple"：`arith` 管四则、位运算、转换、比较，`math` 管超越函数。这个分界有实际后果——`math` 的 op 定义在 scalar / vector / tensor 上都逐元素生效（官方：「On vector and tensor type operations apply elementwise unless explicitly specified otherwise」），所以 `math.sin` 可以一次性作用在整个张量上。

`scf` 那条给的是**下一条边**：结构化控制流最终要变成非结构化的 `cf`，再到 LLVM 或 SPIR-V。这一条正是 [03-progressive-lowering](03-progressive-lowering.md) 整条链的最后一环。

## 2. 执行映射

编译器的"执行"就是一条 pass pipeline，而 pipeline 能跑通的前提是 IR 结构合法。这一节说的是结构在真实系统里被谁消费。

**组件一：Verifier。** MLIR 每个 pass 跑完都能触发一次 IR 校验。校验的内容不是"语义对不对"，而是**结构约束成不成立**：Block 非空、terminator 在末尾、每个 operand 有定义者。这是一条便宜的防线——代价是 O(IR 大小) 的一次遍历，买到的是"非法 IR 不会流到下一个 pass"。本文配套的迷你实现里，同一件事由 `miniir.verify()` 完成，返回的是**问题列表**而不是布尔值（便于一次报全）。

**组件二：pass manager 与 Region 的作用域。** `builtin.module` 的 Region 是一层天然的作用域边界，pass manager 可以按 Region 嵌套地挂 pass。这也解释了为什么第 1 条边界（Op 有零个 Region）重要：**零个 Region 的 op 不构成作用域**，pass 遍历到它就直接过去。

**组件三：`RegionKindInterface`。** 一个 op 的 Region 是 SSACFG 还是 Graph，由这个 interface 声明。这是"结构不变、语义由包含者定义"的直接体现——同样一个 Region，挂在不同 op 下就换了语义。

**量级**：一棵中等大小的 MLIR 函数通常有 10²～10³ 个 Operation，"四层递归"只体现在**结构**上，不体现在元素的绝对数量上。本文的迷你实现在 §5 报的是 15 个 op / 4 层深度这个量级——它是**教学规模**，不是 MLIR 的典型规模。

## 3. 实现与演进

§1 讲的是 IR 的**数据**结构。真实系统还需要第二套东西：**怎么让不同的 op 共享行为**。MLIR 的答案是 Interface 与 Trait 两条并行的机制。

**Trait 是编译期 C++ mixin。** 一个 trait 通过 `OpTrait::TraitBase<ConcreteType, TraitType>` 直接挂在 op 类上，编译期就并进了这个类。上面提过的 `NoTerminator` 就是这样一个 trait。

**Interface 是一份带运行期查询能力的契约。** 官方关于注册方式的原文是：

> it is registered **to an operation** by adding the provided trait `ExampleOpInterface::Trait`

这句话要读准：interface 是通过**它自带的那个 Trait 类**挂到 op 上的，**不是"注册到 MLIRContext"**。只有 external model 才与 context 绑定——官方另一句：

> Other contexts will not see the interface unless registered.

所以准确的说法是：**Interface 自带一个 Trait；用 op 挂 Trait 这条路时，interface 就随之可用**；另有 external model 这条路，那条才与 context 绑定。

**两者不互斥。** 每个 interface 都会自带一个 Trait 类，实现该 interface 的 op 需要把这个 Trait 挂上——它们是"一件事的两种视角"，不是二选一。

**一处要标明是推论的地方**：Trait 因为是编译期 mixin，直觉上"没有运行期查表"；而 `dyn_cast<SomeInterface>(op)` 是运行期查询。这个对比在官方文档里**没有明文**，它是从"mixin vs 运行期注册"的结构推出来的。本文标明它是推论。

可引的确切接口名（用的时候按名字查文档，不要凭印象写）：

`DestinationStyleOpInterface` · `BufferizableOpInterface` · `TilingInterface` · `MemoryEffectOpInterface` · `InferTypeOpInterface`

其中 `DestinationStyleOpInterface` 是 [02-tensor-vs-memref](02-tensor-vs-memref.md) 与 [03-progressive-lowering](03-progressive-lowering.md) 的主角：`linalg` 系列 op 的 `outs` 操作数由它统一约定，这正是"同一个 op 在张量语义下有结果、在缓冲语义下没有结果"能成立的原因。

**演进线索**：本主题三次 lowering 的接口变化（`applyPatternsGreedily` 的改名、bufferize 系列 pass 的废弃）都说明一件事——**这套 IR 的**结构**很稳，**API** 很不稳。见 [04-writing-a-pass](04-writing-a-pass.md) §3 的版本注记。

## 4. 踩坑与失败模式

四条，全部是"结构越界但没人拦"。每条都给出本机实测的触发方式（见 §5）。

**失败模式 1：把空 Block 当"占位的空 Region"。**

- **现象**：构造 `builtin.module { }` 时想着"先搭个壳，内容待填"，于是 Region 里放了一个零 op 的 Block。
- **成因**：`region ::= "{" entry-block? block* "}"` 允许**零个 block**，但 `block ::= block-label operation+` 要求**至少一个 op**。这两条约束的方向相反，凭直觉容易记反成"Region 和 Block 都可以空"或"都不可以空"。
- **规避**：区分开——要占位就**一个 block 都不放**（合法的空 Region）；放了 block 就必须填 op。本机的校验器报的是 `^entry: Block 为空（官方语法要求 operation+）`。

**失败模式 2：给 module 的 body 补 terminator，或反过来要求它必须有 terminator。**

- **现象**：校验器把 `builtin.module` 的 body block 也纳入"必须终于 terminator"的检查，于是**每一个合法模块都被判非法**；或者反过来，为了让校验通过，往 module body 末尾塞一个假的 terminator。
- **成因**：module body 的单块 Region 走 `NoTerminator` 豁免。这条豁免不在"Block 必须有 terminator"这句主规则里，而在紧随其后的一句例外里——**只读主规则就会漏掉**。
- **规避**：把"是否需要 terminator"做成**按 op 决定**的查询（族 trait），而不是全局常量。本仓库的迷你实现在这一点上第一版写错了：错误地在 module body 上要求 terminator，实跑时 `verify()` 报出 1 个问题（见 §5 的第一版被推翻项）。

**失败模式 3：同一个 Value 被定义两次。**

- **现象**：重写时"顺手复用"旧的 `Value` 对象作为新 op 的结果，于是同一个 `%v` 有了两个定义者。IR 打印出来完全正常，形状、类型、运算都对。
- **成因**：`Value` 在很多实现里是一个"名字 + 类型"的轻量对象，没有身份约束。官方对 Value 的定义（"the result of **exactly one** Operation or Block Argument"）是一条**构造期约束**，不是打印期约束。
- **规避**：把绑定做成不可逆的一步——本仓库的迷你实现里 `_bind()` 在第二次绑定时直接抛 `SSAError`，因为"先定义后使用"是后面所有 def-use 分析的前提。这也是计划里 C2 那条断言的来源。

**失败模式 4：operand 引用了一个没有定义者的 Value。**

- **现象**：手写 IR 时先造 operand 再补定义，中间漏了一步；或者重写时删掉了生产者却留着消费者。
- **成因**：IR 构造与 IR 校验被分成了两步，而构造 API 通常不检查。
- **规避**：校验器必须同时查两件事——**每个 operand 有定义者**，以及**定义者先于使用者**。本文的迷你实现只强制了前者（单块函数体里文本顺序即支配关系，后者自动成立），这个简化写在了 `miniir.py` 的注释里。真实 MLIR 里这两件事都由 verifier 负责，且还有 cross-region 的支配关系。

**一条正面清单**：§1 的四条边界里，只有第 4 条（Block 非空 + terminator）是"必须拒绝"的，第 1～3 条（Op 零 region / Op 多 region / Region 空）是**必须接受**的。把它们混在一起做成"一律拒绝"是最常见的过度收紧。

## 5. 实测数据

**测量条件**：`code/01_ir_structure.py` 在本机 `python3 3.13.5` + `numpy 2.4.4` 上跑出的结果，`cd code && make run` 全绿（22 条断言）。

> **★ 限定语（本域强制）**：下面每一个数字都是**本仓库几百行的迷你 IR 实现**跑出来的，**不是 MLIR**。迷你实现借用了 MLIR 的术语（Operation / Region / Block / Value）以便对照，**结构是我们自己简化的**。这些数字支持的是"这套递归结构可以怎样被计数与校验"这类**机制**结论，**不支持任何关于真实 MLIR 的结论**。

**被计数的样例 IR**（打印格式是本仓库自己定的，不是 MLIR 的打印格式）：

```text
builtin.module {
  ^entry:
    func.func {sym = main}
    {
      ^entry(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>):
        %v3: f32 = arith.constant : () -> f32 {value = 0}
        %v4: tensor<2x3xf32> = tensor.empty : () -> tensor<2x3xf32>
        %v5: tensor<2x3xf32> = linalg.fill(%v3, %v4) : …
        %v6: tensor<2x3xf32> = linalg.add(%arg0, %arg1, %v5) : …
        …
        scf.for(%v7, %v8, %v9) : (index, index, index)
        {
          ^bb_iv(%iv: index):
            scf.if(%iv) : (index) { ^then: scf.yield } { ^else: scf.yield }
            scf.yield
        }
        test.empty_region { }
        func.return(%v6) : (tensor<2x3xf32>)
    }
}
```

（为节省篇幅省掉了类型全称与两个 region 的块头；完整输出见 `make run`。）

**结构计数表**（本仓库迷你实现，三列都是小函数里的实测值）：

| 项 | 本机实测值 | 文档 §1 的依据 |
|---|---|---|
| op 总数 | 15 | 递归结构里 op 是唯一的"内容"单位 |
| region 总数 | 6 | 其中 module 自己那个 1 个 |
| block 总数 | 5 | module body / 函数体 / for body / then / else |
| value 总数 | 10 | op 结果 7 + block 参数 3 |
| value 的总 use 数 | 10 | 10 个值共被使用 10 次 |
| 最大嵌套深度（op 层） | 4 | module → func.func → scf.for → scf.if |
| 空 region 数 | 1 | 第 3 条边界：**合法** |
| 零 region 的 op 数 | 11 | 第 1 条边界：**合法** |
| 两 region 的 op 数 | 1 | 第 2 条边界：**合法** |

**A 类的独立参照路径**：把同一棵树 `print_ir` 成文本，再用**独立的文本解析器**重新数一遍（正则抠 op 名、数独占一行的 `{`、收集所有 `%` 记号、按缩进推深度），与结构化遍历的结果逐项比对：

| 项 | 结构化遍历 | 文本重解析 |
|---|---|---|
| op 总数 | 15 | 15 |
| region 总数 | 6 | 5 + 1（module 那个不打印花括号） |
| value 总数 | 10 | 10 |
| 最大 op 深度 | 4 | 4 |

四条 A 类断言全部通过，最大差异 0。**这条路径的意义**：它把"结构遍历"和"文本重解析"做成了两条互不依赖的代码路径，任何一边写错都会立刻暴露。可证伪性反证也在程序里跑了——把 `count_ops` 故意改成漏数 `scf.*`，A 类对照立刻判为不一致（见 `make run` 的 5 节末）。

**C 类的失败行为**（每条都真的抛异常，不是"理论上会"）：

| 构造 | 结果 |
|---|---|
| Block 为空 | `IRVerificationError: ^entry: Block 为空（官方语法要求 operation+）` |
| Block 无 terminator（函数体内） | `IRVerificationError: 最后一个 op 是 arith.constant，不是 terminator` |
| 同一个 Value 定义两次 | `SSAError: 违反 SSA：值 %dup 已被 (arith.constant) 定义过` |
| operand 未定义 | `IRVerificationError: operand %ghost 未被定义` |

**第一版被实跑推翻的一项**：第一版把"Block 必须以 terminator 结尾"做成了对所有 block 的统一要求，于是 `verify()` 在**完全合法的** module 上报了 1 个问题（module body 的最后一个 op 是 `func.func`，被当成"缺 terminator"）。修法就是 §1 第三步末尾那条 `NoTerminator` 豁免——`miniir.Module` 现在建 module op 时显式带上 `no_terminator=True`。这条修正是**跑出来的**，不是读文档读出来的。

## 6. 小结与延伸

- **结构**：`Operation → Region → Block → Operation` 四层递归；Value 恰有一个定义者。
- **四条边界**：Op 有 0..n 个 Region（合法）、Region 可为空（合法）、Block 非空且终于 terminator（强制）——但带 `NoTerminator` 的 op 的单块 Region 例外，`builtin.module` 就是官方举的例子。
- **抽象层次由方言的名字区分**，不由结构区分；同一套 pass 基础设施因此可以跨层复用。
- **`vector` 方言不负责自动向量化**；**`gpu` 方言官方自标"more likely to change than others"**。
- **Interface 挂在 op 上**（通过自带的 Trait），不是挂在 MLIRContext 上；只有 external model 与 context 绑定。

**延伸**：

- 张量语义与缓冲语义的区别、以及连接它们的那次变换 → [02-tensor-vs-memref](02-tensor-vs-memref.md)
- 沿着这套递归逐层替换 Operation 的完整链条 → [03-progressive-lowering](03-progressive-lowering.md)
- 在这套结构上写 pattern rewrite，以及不动点为什么可能到不了 → [04-writing-a-pass](04-writing-a-pass.md)

## 编译与运行

```bash
cd 60-compilers/mlir/code
make 01_ir_structure      # 只跑本篇
make run                  # 跑全部四篇
```

程序：`code/01_ir_structure.py`（配套本仓库迷你 IR `code/miniir.py`，外部事实源 `code/specs.py`）。

## 参考

- MLIR Language Reference — 四层递归、Value 定义、region/block/operation 语法、NoTerminator 豁免、SSACFG 与 Graph region：`https://mlir.llvm.org/docs/LangRef/`
- MLIR Interfaces — interface 的注册方式与 external model：`https://mlir.llvm.org/docs/Interfaces/`
- 各方言文档页首段：`https://mlir.llvm.org/docs/Dialects/ArithOps/` · `…/MathOps/` · `…/SCFDialect/` · `…/Vector/` · `…/GPU/`
- 以上每条 URL 与抄录的原文都在 `code/specs.py` 里逐条对应，取用日期 2026-09-16

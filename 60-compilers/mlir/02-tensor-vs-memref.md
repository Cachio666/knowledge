---
title: tensor 语义与 memref 语义
slug: mlir-tensor-vs-memref
domain: 60-compilers
topic: mlir
level: 2
difficulty: intermediate
prerequisites:
  - 60-compilers/mlir/01-ir-and-dialects
tags: [mlir, tensor, memref, bufferization, one-shot-bufferize, destination-passing-style]
related:
  - 60-compilers/mlir/03-progressive-lowering
  - 60-compilers/mlir/01-ir-and-dialects
  - 40-kernels/elementwise-fusion/02-fusion-and-traffic
code:
  - code/02_bufferization.py
status: stable
updated: 2026-09-16
---

# tensor 语义与 memref 语义

> `tensor<2x3xf32>` 与 `memref<2x3xf32>` 的形状部分完全一样，差别只有一处：**存储有没有被指定**。前者只说"这是一块 2×3 的浮点数据"，后者说"这是某块内存上的一片引用"。把前者换成前者的那次变换叫 **bufferization**，而它不是一次改名——它同时决定了**哪些中间结果真的要被分配出来**。

## 0. 背景与动机

`out = (a + b) * 1.5`。在 tensor 语义下写出来是这样：

```text
%cst = arith.constant 1.5 : f32
%e0  = tensor.empty() : tensor<2x3xf32>
%t1  = linalg.fill(ins %cst, outs %e0) : tensor<2x3xf32>
%t2  = linalg.add(ins %a, %t1, outs %t1) : tensor<2x3xf32>
%t3  = linalg.mul(ins %t2, %b, outs %t2) : tensor<2x3xf32>
func.return %t3
```

（语法是本仓库迷你实现的打印格式，用于说明结构；真实 MLIR 的写法见 §3。）

这段 IR 里**没有任何一处提到"存储"**。`%t1`、`%t2`、`%t3` 都只是 SSA 值。所以你没法从这段 IR 回答一个硬件上最要紧的问题：**这块 `(a+b)` 到底要不要落到内存里？**

这不是"信息不足"，而是**故意推迟**。推迟的理由很实际：tile / fuse / 各种形状变换在 tensor 语义下做都更容易——官方文档原话是 "many transformations are easier or only supported in tensor land; e.g., tile/fuse/… on tensors first, then bufferize the remaining IR"。

但推迟有代价，而且这个代价正好是本域主线上的那个权衡：

- **每个 tensor 结果都物化**（各分配一块缓冲）：IR 好写，但 `%t1`、`%t2` 这些只被用一次、用一次就死的中间结果全都占着内存。
- **全部就地复用**：省内存，但只要判据错一次，**算出来的数就是错的**，而且 IR 照样打印得漂漂亮亮。

于是问题具体化成：**bufferization 凭什么决定某个 tensor 值可以就地写、哪个必须新分配？** §1 把这套判据推出来，§4 给出判据写错时的后果（本机实测：最大误差 4.257e+03）。

## 1. 原理

先立措辞（这一步在本主题里不能省）。

**官方用的是「tensor semantics」与「memref semantics」**。Bufferization 文档的第一句就是：

> Bufferization in MLIR is the process of converting ops with **tensor semantics** to ops with **memref semantics**.

**「值语义 / 缓冲语义」不是官方引文。** 这两个词作为**概念**讲得通（tensor 值像不可变值，memref 像可写缓冲），研究代理也 grep 不到逐字原句。本文允许用它们做解释，但**不得当成引文**——引用时只引上面那句。

`memref` 这边有一句确切的官方定义（在 Builtin 方言页的 `MemRefType` 条目下）：

> **Shaped reference to a region of memory**

"reference"这个词是关键：memref 不是内存，是**指向内存的引用**。这直接解释了三件事——memref 可以被 `memref.alloc` 创建、可以被 `memref.dealloc` 释放、可以有两个 memref 指向同一片内存（别名）。别名正是 §4 里所有危险的来源。

**第一步：两种语义的差别只有一维。**

| | `tensor<2x3xf32>` | `memref<2x3xf32>` |
|---|---|---|
| 形状 | 2×3 | 2×3 |
| 元素类型 | f32 | f32 |
| **存储** | **未指定** | **已指定（一片内存上的引用）** |
| 能否别名 | 不能（每个值是独立的值） | 能（两个 memref 可指向同一片内存） |
| 就地写 | 概念上不成立（写的是"另一个值"） | 成立，且正是优化的目标 |

差别只在最后一格，而最后一格决定了整个 bufferization 的难度。

**第二步：destination-passing style——让"写到哪儿"变成显式的。**

Bufferization 文档对 one-shot bufferize 的定位是：

> One-Shot Bufferize is a tensor bufferization pass designed for IR in **destination-passing style**, and with **aggressive in-place bufferization**.

destination-passing style 的意思是：**每个 op 都有一个显式的 `outs` 操作数，说明"结果写到哪里"**。上面那段 IR 里 `linalg.fill` 的 `%e0`、`linalg.add` 的 `%t1`、`linalg.mul` 的 `%t2` 就是 `outs`。

这一条很重要，因为它把"物化"这个模糊的问题变成了一个**有明确答案位置**的问题：缓冲要么来自 `outs` 已经指向的东西，要么必须新分配。

**第三步：`tensor.empty` 就是分配点。**

`tensor.empty` 的含义是"形状对、内容未定义"。在真实的 one-shot bufferize 里，它 bufferize 成 `bufferization.alloc_tensor`，最终变成 `memref.alloc`。所以：

```text
一次 tensor.empty   ⟺   一次 memref.alloc
```

这条把"物化次数"从一个抽象概念变成了可以数的东西：**数 `memref.alloc` 的个数**。

**第四步：就地判据。**

判据的核心是一个问题：**把结果写进 `outs` 指向的那块缓冲，会不会破坏后面还要读的数据？**

用 SSA 的语言说：设某个 op 的 `outs` 是张量值 `%o`。若 `%o` 的**最后一次使用就是这个 op**，那么此刻之后没人再读 `%o` 的旧内容，就地写是安全的；若 `%o` 在本 op 之后还有读者，就地写会把那个读者看到的旧值改掉。

于是三条规则（本文配套程序用的就是这三条）：

| # | 情形 | 决策 | 分配次数 |
|---|---|---|---|
| 1 | `outs` 是 `tensor.empty` 的结果（且该 empty 只被这一个 op 用） | 分配归 empty | **1** |
| 2 | `outs` 的最后一次使用**就是**本 op | 就地复用它的缓冲 | **0** |
| 3 | `outs` 在本 op 之后还有读者 | 必须新分配 | **1** |

另外一条是**拒绝**规则：同一个张量值被**两个 op** 当作 `outs`，就地分析无法给出唯一决策——因为两次写会互相覆盖，且哪个先写取决于执行顺序。

把这套规则套到 §0 的那段 IR 上：

```text
%e0 被 linalg.fill 用，且只被它用      → 规则 1，分配 1 次
%t1 的最后一次使用是 linalg.add        → 规则 2，就地，0 次
%t2 的最后一次使用是 linalg.mul        → 规则 2，就地，0 次
```

**三个中间结果，最终只分配了一块缓冲**（本机实测，见 §5）。而朴素做法——每个 tensor 结果各分配一块——要 4 块。

**第五步：这条判据是"保守"还是"激进"？**

官方说 one-shot 是 "aggressive in-place bufferization"，所以它的方向是**尽量就地**。但"激进"不等于"不检查"：上面规则 3 那一条就是那个检查，§4 会给出它被绕过的后果。

**第六步：bufferization 之外还有两件事，one-shot 不做。**

- **它不做 dealloc。** 官方把"插入 buffer 释放操作"归给**另外的 helper pass**（"additional helper passes … insert buffer deallocation ops so that the resulting memref IR has no memory leaks"）。所以 `-one-shot-bufferize` 跑完的 IR 里没有 `memref.dealloc`，这是**设计如此**，不是漏了。
- **它不是一步，是两步。** 官方原文：

  > **2-Phase**: Bufferization is internally broken down into 2 steps: First, analyze the entire IR and make bufferization decisions. Then, bufferize (rewrite) the IR.

  "先分析全 IR、再改写"这条顺序有实在的后果：就地决策在**改写之前**就全部定好了，所以决策之间不会互相影响——这也让决策表本身可以被单独拿出来核对（本文配套程序就是这么做的，见 §5 B 类最后一条）。

## 2. 执行映射

这一节说清楚"这一步在真实编译流程里的位置"。

**位置：pipeline 的倒数第二步。** 官方原话是这些 pass "typically run as one of the last steps in a pass pipeline, right before lowering to memref ops to LLVM"。所以顺序是固定的：**先在 tensor 语义下把该做的形状变换做完，再一次 bufferize**。反过来（先 bufferize 再做优化）不是不能做，而是所有基于值的变换都要重写成基于缓冲的形式。

**组件一：`BufferizableOpInterface`。** one-shot bufferize 处理的是"实现了这个 interface 的、有 tensor 语义的 op"。官方原文：

> MLIR provides a pass `-one-shot-bufferize` that performs an analysis and bufferizes all ops with tensor semantics that implement `BufferizableOpInterface`.

没实现这个 interface 的 op **不会被 bufferize**——它不是报错，是**跳过**。官方举的例子是 `tosa.matmul`：它没有 `BufferizableOpInterface` 实现，要先降到 `tensor.empty` + `linalg.matmul` 的组合才能被 bufferize。

**组件二：external model。** 官方提到这些 interface 实现"typically **external models** that live in a dialect's 'Transforms' build unit"。这也是 [01-ir-and-dialects](01-ir-and-dialects.md) §3 里"只有 external model 才与 context 绑定"那句话的落点：bufferization 的 interface 实现大量走这条路。

**组件三：废掉的旧路径。** 早期 MLIR 用一组 dialect-conversion pass 做 bufferization：`linalg-bufferize` / `tensor-bufferize` / `scf-bufferize` / `finalizing-bufferize`。这四个名字在当前的 pass 索引页 `mlir.llvm.org/docs/Passes/` 里**一个都查不到**——它们已被 one-shot 取代。**引用它们之前先确认版本**，网上大量教程讲的仍是这四个。

另有一个**仍在文档里、但状态不同**的项：`buffer-deallocation`。Bufferization 文档有一节专门叫 **Deprecated Passes**，写的是：

> The buffer deallocation pass has been deprecated in favor of the ownership-based buffer deallocation pipeline. The deprecated pass has some limitations that may cause memory leaks **in the resulting IR**.

这句话有两处值得记准：一是它讲的是**释放**那一步（不是转换那一步）；二是引全的时候"in the resulting IR"不能省——官方说的是**生成的 IR 里**会漏，不是进程会漏。

**量级**：一次 one-shot bufferize 在常见的 elementwise 链上能把分配次数压到 "链长 → 1" 这个量级。本机实测的 4 → 1 是**迷你实现**上的一个具体例子（§5），不要把它当成通用倍率。

## 3. 实现与演进

**朴素做法（第一代）：一堆 dialect-conversion pass。** 每个方言一个 pass 负责把自己的 tensor 语义 op 换成 memref 语义 op：`linalg-bufferize` 管 linalg、`tensor-bufferize` 管 tensor 方言、`scf-bufferize` 管 scf、最后 `finalizing-bufferize` 收尾。

这条路的问题不在"能不能做对"，而在**决策是局部的**：每个 pass 只看自己那几个 op，看不到跨方言的使用链。于是一个"结果只被用一次"的明显机会，可能因为使用者还没被转换而看不到——只能保守地多分配一块。今天的文档索引里这四个名字已经不存在了（见 §2）。

**主流做法（第二代）：one-shot bufferize。** 官方定位是"**one-shot**"——分析整个 IR 一次，把所有决策一起定下来，然后才改写。这正好解决了上一代的局部性问题，代价是分析阶段需要完整的使用链信息。

**第三步（前沿方向）：ownership-based buffer deallocation。** 它接的是 one-shot 剩下的那一半——**释放**。官方把它和旧的 `buffer-deallocation` 明确对立起来：一个"已废弃、会在 IR 里漏"，一个取代它。所以在今天的 pipeline 里，`-one-shot-bufferize` 与 `-ownership-based-buffer-deallocation` 是**成对出现**的。

**一条贯穿三代的线索**：bufferization 的难点从来不是"把 tensor 换成 memref"这个改写本身，而是**别名**。tensor 值没有别名，memref 有——所以第一代靠局部保守、第二代靠全局分析、第三代把释放交给显式的所有权模型。用 [01-ir-and-dialects](01-ir-and-dialects.md) 的话说：结构没变，变的是"谁来保证别名正确"。

## 4. 踩坑与失败模式

四条。前两条是本域主线（物化 vs 重算）的直接后果，都**能真的跑出错数来**。

**失败模式 1：就地判据漏掉"outs 之后还有读者"。**

- **现象**：程序照常跑完、IR 照常打印、断言汇总照常输出，只有数值是错的。
- **成因**：判据只查了"`outs` 是不是 `tensor.empty`"或"`outs` 是不是只被用一次"，没查"**这个唯一的使用是不是就是当前这个 op**"。两次写的先后顺序决定了覆盖关系，而 IR 里没有任何东西会因此报错。
- **规避**：判据必须按**位置**比，不能只按计数比。本机实测的后果很具体：一个 4 行的小 IR 上，强制就地让最大误差从 0 跳到 **4.257e+03**（参考值首元素 16.5，算出来 121.0）——见 §5。这不是舍入误差，是读了被覆盖的数据。

  > 顺带说明为什么"两个 op 写同一个 `outs`"必须直接拒绝：那种情况下"最后一次使用"这个说法本身就失效了，判据没有唯一答案。

**失败模式 2：把 `tensor.empty` 当成"零值"而不是"分配点"。**

- **现象**：以为 `tensor.empty` 只是"造个零张量"的语法糖，于是 bufferize 时把它丢掉，让下游 op 直接用上游的缓冲。IR 通过校验、数值也常常是对的——直到某个下游 op 的 `outs` 位置被填进了别的值。
- **成因**：`tensor.empty` **内容未定义**，它不是零。它存在的唯一理由是**携带形状并标记分配点**。把它当零值处理，等于同时丢掉分配点和一个"这块内存还没人写过"的事实——后者正是规则 1 安全的原因。
- **规避**：在 bufferize 的实现里，`tensor.empty` 必须**一对一**变成一个分配（真实 MLIR 路径是 `bufferization.alloc_tensor` → `memref.alloc`）。本仓库迷你实现的第一版就在这里走过弯路：它一开始把 `tensor.empty` 与 linalg op 的 `outs` 各算一次分配，导致分配次数多算——这个数被 B 类表逐行比对抓了出来（见 §5）。

**失败模式 3：以为 `-one-shot-bufferize` 会顺手把 `memref.dealloc` 插上。**

- **现象**：bufferize 之后直接跑内存检查，发现每块缓冲都没被释放，判定"pass 有 bug"。
- **成因**：释放不在这个 pass 的职责里（§1 第六步）。官方把它归给 helper pass，现状是 ownership-based 那条 pipeline。
- **规避**：pipeline 里成对挂 `-one-shot-bufferize` 与 `-ownership-based-buffer-deallocation`。**单独跑前者、然后抱怨漏内存，是把设计当成了缺陷。**

**失败模式 4：拿"值语义 / 缓冲语义"当官方术语引用。**

- **现象**：写"MLIR 官方称这两种语义为 value semantics 与 buffer semantics"，被读者按这句话去搜原文，搜不到。
- **成因**：这两个词是社区里流通的通俗说法，描述得也没错，但**不是文档用词**。官方用 `tensor semantics` / `memref semantics`。
- **规避**：概念上两个说法都能用；**引用时只引官方那句**。同一类问题在本主题反复出现（[01-ir-and-dialects](01-ir-and-dialects.md) 的 `vector` 方言、[03-progressive-lowering](03-progressive-lowering.md) 的 pass 名），根因都是"凭印象转述"。

## 5. 实测数据

**测量条件**：`code/02_bufferization.py` 在本机 `python3 3.13.5` + `numpy 2.4.4` 上跑出的结果，`cd code && make run` 全绿（22 条断言：A 7 / B 11 / C 4）。

> **★ 限定语（本域强制）**：下面每一个数字都是**本仓库的迷你 bufferization**跑出来的，**不是 MLIR 的 `-one-shot-bufferize`**。判据是本文 §1 第四步那三条规则的一个实现；真实系统还有别名分析、`BufferizableOpInterface` 的外部模型、以及一整套 pass 选项。这些数字支持的是"这类判据会怎样改变分配次数、判据写错会怎样"这类**机制**结论，**不支持任何关于真实 MLIR 的结论**。

**被 bufferize 的样例**：一条三段的 elementwise 链（`fill(1.5) → +a → ×b`），形状 `2×3×f32`。

**决策表**（本机实测，`make run` 的 1 节输出）：

| 位置 | op | outs | 决策 |
|---|---|---|---|
| 2 | `linalg.fill` | `%v4` | `fresh-empty`（分配归 empty） |
| 3 | `linalg.add` | `%v5` | `in-place` |
| 4 | `linalg.mul` | `%v6` | `in-place` |

**B 类：物化次数与字节数**（本机实测，与本文 §1 的规则逐条对应）：

| 项 | 本机实测值 |
|---|---|
| tensor 语义下的 tensor 结果数 | 4 |
| 朴素 bufferize 的 `memref.alloc` 次数 | 4 |
| one-shot 判据的 `memref.alloc` 次数 | **1** |
| 就地决策数（`in-place`） | 2 |
| tensor IR 的 op 数 | 7 |
| one-shot memref IR 的 op 数 | 7 |
| 朴素 memref IR 的 op 数 | 10 |
| 单块缓冲字节数 | 24 B（2×3×4 B） |
| 朴素总分配字节数 | 96 B |
| one-shot 总分配字节数 | **24 B** |

就地判据省下的分配：**4 → 1（省 3 次，75%）**。注意两个 memref IR 的 op 数差（7 vs 10）与分配次数差（1 vs 4）**不是同一件事**：差的 3 个 op 正是三次多出来的 `memref.alloc`。

还有一条把"决策"和"实现"解耦的检查：**用决策表独立算出分配次数（2 次就地 + 1 次 empty = 1 次分配），与 bufferize 实际产出的 `memref.alloc` 个数比对**，两者一致。这条断言的作用是：它不看 bufferize 的实现，只数决策——实现写错、决策对，它会报出来。

**A 类：独立参考实现。** 参照物是一条**纯 numpy 的表达式** `(a + k) * b`，位于 `code/02_bufferization.py` 的 `reference_chain`，与解释器/IR 没有任何共享代码：

| 比对 | 最大误差 |
|---|---|
| tensor 语义执行结果 vs numpy 参考 | 0.000e+00 |
| memref 语义执行结果 vs numpy 参考 | 0.000e+00 |
| bufferize 前后互相之差（语义等价） | 0.000e+00 |
| 朴素版本 vs numpy 参考 | 0.000e+00 |

**注意归类**："bufferize 前后语义等价"断言的是**成功**行为，因此按 CONTRIBUTING §2 归 **A 类**，不归 C 类。C 类断言的是**失败**行为成立——§4 的第一条失败模式正是用 C 类来跑的（见下）。

**C 类：判据被绕过时真的算错**（反例地形：`t1` 在被就地写之后还被 `t3` 读到）：

| 检查 | 结果 |
|---|---|
| C1 判据把该 op 判为 `out-of-place` | 通过（决策 = `out-of-place`） |
| C2 **强制就地**后的数值误差 | **4.257e+03**（参考首元素 16.500，算出来 121.000） |
| C2 对照：走判据这条路 | 最大误差 0.000e+00 |
| C3 一个值被两个 op 当 `outs` | `TransformFailure: 写冲突` |
| C4 可证伪性反证：把判据换成"永远就地"，A 类等价性判据 | **失败**（反证成立） |

**第一版被实跑推翻的两项**：

1. **"每个 tensor 结果各算一次分配"是错的。** 第一版的分配计数把 `tensor.empty` 与消费它的 linalg op 各算了一次，B 类表立刻报 MISMATCH。修正后 `tensor.empty` 是**唯一**的分配点（§1 第三步），linalg op 在 `fresh-empty` 情形下复用它。
2. **第一版的反例地形不成立。** 最初的冲突地形把 `%t1` 同时当作某个 op 的 `ins` 和 `outs`——这在 elementwise 语义下其实是安全的（逐元素先读后写）。改成"`%t1` 在**另一个** op 里被读到"之后，强制就地才真的算出错数。**这是跑出来的区分，不是推出来的**：第一次跑时 `max_err` 是 0，反例根本没成立。

## 6. 小结与延伸

- `tensor` 与 `memref` 的差别只有一维：**存储有没有被指定**。官方措辞是 tensor semantics / memref semantics。
- **bufferization** 就是消掉这一维的那次变换；one-shot 是当前的实现，**2-phase**，且**不做 dealloc**。
- **`tensor.empty` 是分配点**，不是零值。分配次数因此是可数的：数 `memref.alloc`。
- 就地判据的核心是"**`outs` 的最后一次使用是不是就是这个 op**"——按计数不按位置比，就会静默算错。
- **别用「值语义 / 缓冲语义」当引文**；别用那四个已从索引里消失的 dialect-conversion pass 名。

**延伸**：

- 为什么"先 bufferize"不是可选的优化而是后续 lowering 的**前置条件** → [03-progressive-lowering](03-progressive-lowering.md)
- `DestinationStyleOpInterface` 与 external model 的机制 → [01-ir-and-dialects](01-ir-and-dialects.md) §3
- 融合为什么能省下这些中间结果、以及它的收益上界从哪来 → `40-kernels/elementwise-fusion/02-fusion-and-traffic`（正文在上游，此处回链）

## 编译与运行

```bash
cd 60-compilers/mlir/code
make 02_bufferization      # 只跑本篇
make run                   # 跑全部四篇
```

程序：`code/02_bufferization.py`（共享迷你 IR 与解释器 `code/miniir.py`，外部事实源 `code/specs.py`）。

## 参考

- MLIR Bufferization — bufferization 定义、one-shot 的定位与 2-phase、Deprecated Passes 一节：`https://mlir.llvm.org/docs/Bufferization/`
- MLIR Builtin Dialect — `MemRefType`：**Shaped reference to a region of memory**：`https://mlir.llvm.org/docs/Dialects/Builtin/`
- MLIR Passes 索引 — 用于核对 `linalg-bufferize` / `tensor-bufferize` / `scf-bufferize` / `finalizing-bufferize` 四个名字的存否：`https://mlir.llvm.org/docs/Passes/`
- MLIR Interfaces — external model 与 context 的关系：`https://mlir.llvm.org/docs/Interfaces/`
- 以上每条 URL 与抄录的原文都在 `code/specs.py` 里逐条对应，取用日期 2026-09-16

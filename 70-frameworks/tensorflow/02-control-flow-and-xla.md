---
title: 控制流与 XLA：固化分支，还是把两条都放进图
slug: tensorflow-control-flow-and-xla
domain: 70-frameworks
topic: tensorflow
level: 2
difficulty: intermediate
prerequisites:
  - 70-frameworks/tensorflow/01-graph-and-eager
tags: [tensorflow, autograph, tf.cond, tf.while-loop, tracing, jit-compile, xla]
related:
  - 70-frameworks/tensorflow/01-graph-and-eager
  - 70-frameworks/pytorch/01-dispatch-and-aten
  - 60-compilers/xla/01-hlo-and-shape-inference
code:
  - code/02_control_flow.py
status: stable
updated: 2026-09-17
---

# 控制流与 XLA：固化分支，还是把两条都放进图

> 追踪期遇到 `if`，框架只有两个选择：**把当时走到的那条分支固化进图**（Python `if`：
> 图更小，但假设更硬），或者**把两条分支都放进图、运行时再选**（`tf.cond`：图更大，
> 但假设更软）。AutoGraph 的转换开关只有一条：**条件是 Tensor 才转**。
> 而 XLA（`jit_compile=True`）是**另一个编译单元**——它要求**整个函数都能被编译**，
> 否则直接抛 `InvalidArgumentError`。

## 0. 背景与动机

### 一个具体的对照场景

```
def loss(x, training):
    if training:            # 走哪条？什么时候定？
        return f(x)
    return g(x)
```

同一段代码，图上有两种完全不同的样子：

| | 图里有什么 | 什么时候决定走哪条 | 代价 |
|---|---|---|---|
| **Python `if`**（条件是 Python 值） | **只有一条分支** | **追踪的时候** | 分支变了 → **重追踪**（或更糟：不重追踪而算错） |
| **`tf.cond`**（条件是 Tensor） | **两条分支都在** | **运行时**（设备上） | 图变大；但**换分支不重追踪** |

### 本文在本域主线上的位置：**②代价项**

本域的主线是「**框架要把哪些假设固化进图**」（首篇
[01-graph-and-eager](01-graph-and-eager.md) 完整展开了这条主线）。本文挂的是其中的
**代价项**：**固化一支的代价是「换分支要重追踪」，不固化的代价是「图的节点数翻倍」**。
这两笔账都是可数的——§1.4 给出推导，§5 给出迷你实现量到的数。

XLA 那一半（§3）挂的也是代价项的另一面：**把图交给另一个编译单元，要付「整函数必须可编译」
这个硬门槛**。XLA 内部的 HLO / 融合 / layout **不在本文**，正文在
[60-compilers/xla](../../60-compilers/xla/01-hlo-and-shape-inference.md)，本文只回链。

## 1. 原理：AutoGraph 什么时候出手

### 1.1 转换的触发条件只有一条

官方逐字（`site/en/guide/function.ipynb` §Conditionals）：

> AutoGraph will convert some `if <condition>` statements into the equivalent `tf.cond` calls.
> **This substitution is made if `<condition>` is a Tensor.** Otherwise, the `if` statement is
> executed as a Python conditional.

也就是说：

- **条件是 Tensor** → 转成 `tf.cond`（**数据相关**的控制流，只能在运行时选）；
- **条件是 Python 值**（`bool`、`int`、配置项）→ **不转**，它就是一个普通的 Python `if`，
  **在追踪的时候执行一次**。

### 1.2 两条路各自的图长什么样

官方逐字（同页）：

> A Python conditional executes during tracing, so **exactly one branch** of the conditional
> will be added to the graph. Without AutoGraph, this traced graph would be unable to take the
> alternate branch if there is data-dependent control flow.

> `tf.cond` **traces and adds both branches** of the conditional to the graph, dynamically
> selecting a branch at execution time.

控制流参考文档里有一句更狠的表述（`autograph/g3doc/reference/control_flow.md`）：

> When constructing a graph, TensorFlow _traces_ the code. The tracing of control flow
> requires **visiting _every possible code path_** (usually once).

> In other words, **tracing executes both branches of an if statement**. Similarly, the body of
> loops is executed once (even if the loop would otherwise not iterate at all).

**「每条路径都跑一遍」是追踪的代价**——它是「为什么 Python 副作用只发生一次」
（[01](01-graph-and-eager.md) §2.2）的另一面：副作用在**追踪**时发生，而追踪**会走遍所有分支**。

### 1.3 循环的对应关系

官方给的转换规则（同页）：

- `for x in y`：**`y` 是 Tensor** → 转 `tf.while_loop`；
- `while <condition>`：**`<condition>` 是 Tensor** → 转 `tf.while_loop`；
- **不满足条件就不转**，那个 `for` / `while` 就是 **Python 循环**——**在追踪时被展开**。

### 1.4 推导：两种做法的图节点数

设一条分支的算子数为 `T`（then）、`E`（else），谓词本身的算子数为 `P`。则：

| 做法 | 追踪几次 | 图的节点数 |
|---|---|---|
| **`tf.cond`（条件是 Tensor）** | **1 次**（与走哪条无关） | `P + 1 + T + E`（`1` 是 `cond` 自己） |
| **Python `if` + Tensor 条件** | **不允许**：对符号张量取 `bool` 会抛 `OperatorNotAllowedInGraphError` | —— |
| **Python `if` + Python 布尔参数** | **2 次**（`True` / `False` 是两个缓存键，见 [01](01-graph-and-eager.md) §1.3） | 每张图 `P + max(T, E)` 之类，**只有一支** |
| **Python `if` + 自由变量** | **1 次**（自由变量不进 TraceType） | **只有一支**——而且**改了自由变量它也不追新的**（静默错） |

这张表就是主线上那笔账的全部：**`tf.cond` 用「图更大」换「不重追踪」**，
Python `if` 反过来。第三条（自由变量）是**两头都输**：图小、但假设硬到会算错。

同样地，循环：

| 做法 | 图的节点数 | 追踪次数 |
|---|---|---|
| `tf.while_loop`（条件是 Tensor） | **与迭代次数无关**（body 只记一遍 + 循环 op + 条件子图） | 1 次（跑 5 次与跑 500 次是同一张图） |
| Python `for` / `while` | **随迭代次数线性增长**（展开） | 迭代次数变了 → **新键、重追踪** |

**`while_loop` 的「不固化」与 `tf.cond` 是同一招**：把**变化的东西**（哪一支、迭代几次）
留在图上当**运行时的量**，而不是固化成**追踪期的常量**。

## 2. 执行映射：图上的两种控制流

### 2.1 `tf.cond`：一个节点 + 两张子图

```
graph 'with_cond' {
  %2 = tf.greater(%1)  [字面量 rhs_literal=0.0]
  %5 = tf.cond(%2)  [子图 then: 1 个节点]  [子图 else: 1 个节点]
}
```

（形态取自 `code/02_control_flow.py` 的 §1 输出，**本仓库迷你追踪器实测，不是 TensorFlow**。）

要点三条：

1. **两个分支的 Python 代码都跑了**（副作用各发生 1 次——本程序里数出来是
   `{'then': 1, 'else': 1}`）；
2. **两个分支的算子都在图里**（一张图 `all_op_names()` 里同时有 `tf.add` 与 `tf.subtract`）；
3. **谓词是一个普通的图内节点**（`tf.greater`），它的值在**运行时**才确定——
   所以**换输入不重追踪**：迷你实现里同一张图喂 `+2` 与 `-2`，结果分别是 `3.0` 与 `-3.0`，
   追踪计数始终是 **1**。

### 2.2 Python `if` + 布尔参数：两张图，各一支

`f(x, True)` 与 `f(x, False)` 的 `TraceType` 分别是 `LiteralTraceType<True>` 与
`LiteralTraceType<False>`——**两个缓存键、两次追踪**，每张图里**只有一支**。
结果**都是对的**：代价是**重追踪**，不是算错。这一条很容易被误读成「Python `if` 不能用」，
准确的说法是：**它对每个取值各付一次追踪成本**。

### 2.3 Python `if` + 自由变量：一张图、一支、**并且不会重追踪**

这是最坏的一种。`TraceType` 只看输入参数，所以：

- 图里**只固化了追踪时走的那一支**；
- 改了自由变量之后**没有新键**，于是**复用旧图**——**结果照着旧分支走**。

迷你实现里量到的是：改自由变量前后追踪计数都是 **1**，返回值**仍是旧值**
（见 `code/01_tracing_and_retracing.py` §C3）。官方对这条的表述见
[01](01-graph-and-eager.md) §2.3。

## 3. 实现与演进：XLA 是另一个编译单元

### 3.1 `jit_compile=True` 的官方语义

`tf.function` 的 docstring 逐字：

> jit_compile: If `True`, compiles the function using [XLA](https://tensorflow.org/xla).
> XLA performs compiler optimizations, such as fusion, and attempts to emit more efficient
> code. This may drastically improve the performance. **If set to `True`, the whole function
> needs to be compilable by XLA, or an `errors.InvalidArgumentError` is thrown.** If `None`
> (default), **compiles the function with XLA when running on TPU** and goes through the
> regular function execution path when running on other devices. If `False`, executes the
> function without XLA compilation. … **Not all functions are compilable**, see a list of
> [sharp corners](https://tensorflow.org/xla/known_issues).

三件事值得单独记住：

1. **门槛是「整函数」**——不是「大部分算子」，是一条**全有或全无**的约束；
2. **默认 `None` 是设备相关的**：**TPU 上走 XLA**，其他设备走常规路径。
   「我加了 `tf.function` 但没加 `jit_compile`，为什么没有 XLA」的答案就在这里；
3. **不是所有函数都能编译**，官方给了一个 sharp corners 清单（本环境打不开该页，
   见 §5 的说明）。

### 3.2 `tf.xla.experimental.compile`：**仍在，但已废弃**

源码里逐字（`tensorflow/python/compiler/xla/xla.py`）：

```python
@tf_export('xla.experimental.compile')
@deprecated(
    None, 'xla.experimental.compile is deprecated. Consider using '
    '`@tf.function(jit_compile=True)`.',
    warn_once=True)
def compile(computation, inputs=None):
```

三点：

- **它还在**（不是被删掉，而是被 `@deprecated` 标了）；
- **官方没给移除日期**——`@deprecated` 的第一个参数是 `None`；
  **也查不到「从哪个版本起废弃」**（本仓库没有取到版本归属的源码证据，故不写）；
- 成功路径上它**照常工作**：迷你实现里调用它得到正确结果 + 一条 `DeprecationWarning`，
  告警文案与源码逐字一致。

### 3.3 该谁写

| 知识点 | 正文位置 |
|---|---|
| `jit_compile` 怎么开、门槛是什么、和 `tf.xla.experimental.compile` 什么关系 | **本文**（用户接口视角） |
| HLO、融合、layout、shape 推断、内存规划 | [60-compilers/xla](../../60-compilers/xla/01-hlo-and-shape-inference.md)（**回链**） |

## 4. 踩坑与失败模式

### 失败模式 1：对符号张量取 `bool` 或迭代

**现象**：追踪期写 `if x:` 或 `for v in x:`，抛
`OperatorNotAllowedInGraphError`（官方文案的第一句逐字是
「Using a symbolic `tf.Tensor` as a Python `bool`」/「Iterating over a symbolic `tf.Tensor`」，
后缀给出的解法是「用 Eager 模式或加 `@tf.function`；若已在用 AutoGraph，
可能是它看不到你的源码」）。
**成因**：图里没有「一个张量的真值」这种概念——`bool` 只能作用在**已知值**上，
而追踪期它是**符号**。
**规避**：数据相关的分支用 `tf.cond`，数据相关的循环用 `tf.while_loop`。

### 失败模式 2：`tf.cond` 两个分支的**返回类型**对不上

**现象**：`tf.cond(p, lambda: x + 1, lambda: tf.cast(x, tf.int32))` 这类写法在追踪期直接失败。
**成因**：两支的结果要合成**同一个**输出类型，dtype（以及结构）必须一致——
`tf.cond` 是**一个节点**，它只能有一种输出类型。
**规避**：两支先 `.cast` 到同一 dtype；结构也要对齐（都在两支里返回相同的嵌套结构）。
顺带一条：**两支都会真的被 trace**，所以「另一支里的错误」也会在**这一支被执行之前**暴露出来——
这是好事（早失败），但会让第一次调用的报错位置看起来莫名其妙。

### 失败模式 3：把「数据相关」的分支写成 Python `if`

**现象**：`if loss > threshold:` 这类写法，要么直接抛错（条件是 Tensor），
要么**silently 固化一支**（条件是自由变量 / 配置项）。
**成因**：见 §1.1 的触发条件。
**规避**：问自己一句「这个条件在**运行时**会不会变」。会变 → `tf.cond`；
不会变（真的是超参）→ Python `if` 没问题，但**别改它**。

### 失败模式 4：以为 `jit_compile=True` 只是「更快一点」

**现象**：给一个含 `tf.print`（或任何 XLA 编不了的算子）的函数加 `jit_compile=True`，
直接抛 `errors.InvalidArgumentError`，整个函数**不能跑**。
**成因**：docstring 里那条硬门槛——**the whole function needs to be compilable**。
**规避**：先用 `jit_compile` 之外的路子验证逻辑，再把它打开；
遇到编译不了的算子就把它挪出被编译的函数。
> ⚠ **本项目对这条失败模式的限定**：「**整函数必须可编译，否则抛 `InvalidArgumentError`**」
> 是官方 docstring 的逐字规则（有来源）；但**「`tf.print` 在 XLA 下编不了」本仓库没有取到
> 一手来源**（官方 sharp corners 页在调查环境打不开）。编号程序里那条
> 「含 `tf.print` 的图编不过」用的是**本仓库迷你实现自己定的**不可编译算子集合，
> 代码与输出里都标注了这一点，**不得**当成 TF 的事实引用。

## 5. 实测数据

**测量条件**：`code/02_control_flow.py` 经 `make run` 在本机跑出
（python3 3.13.5 / numpy 2.4.4），全部 19 条断言通过（A 4 / B 5 / C 10）。

> **★ 限定语（本域强制）**：下面每一个数字都是**本仓库的迷你追踪器**跑出来的，
> **不是 TensorFlow**。本机**没有装 TF**。这些数字支持的是「控制流的两条路在图上的
> 节点数与追踪次数如何被计数」这类**机制**结论，**不支持任何关于真实 TensorFlow 的结论**。
> 关于真实 TF 的行为断言以 §参考 的一手 URL 为依据。

**（1）图的节点数**（迷你实现实测）：

| 构造 | 节点数 | 说明 |
|---|---|---|
| `tf.cond`（谓词 1 + cond 1 + then 3 + else 4） | **9** | 两支**同时**在图上 |
| `tf.while_loop`，`init=5` | **3** | 与迭代次数无关 |
| `tf.while_loop`，`init=500` | **3** | **与 `init=5` 是同一张图** |
| 对照：Python `for` 循环 N=3 | 3 | 展开 |
| 对照：Python `for` 循环 N=6 | 6 | **随 N 线性增长**（差恰好 3） |

**（2）追踪次数**（迷你实现实测）：

| 场景 | 追踪次数 |
|---|---|
| `tf.cond`，谓词取 `+2` / `-2` / `-0.5` / `+5` 等 6 个值 | **1** |
| `tf.while_loop`，`init=5` 与 `init=500` | **1** |
| Python `if` + 布尔参数 `True` / `False` | **2** |

**（3）XLA 门槛的模型**（迷你实现实测）：
整函数可编译时编译通过；含 `tf.print`（本模型约定的不可编译算子）时抛
`InvalidArgumentError`——**注意这条的具体规则是本仓库定义的，见上面失败模式 4 的说明**。

**（4）本页引用的**规格值**（来自 TF 源码，非本仓库测量）**：
`@deprecated(None, 'xla.experimental.compile is deprecated. …', warn_once=True)`——
第一个参数是 `None`，**官方没给移除日期**。

## 6. 小结与延伸

- AutoGraph 的转换**只有一个开关**：**条件是 Tensor**。是 → `tf.cond`；不是 → Python 条件。
- **Python `if` 只固化一支**（官方逐字：exactly one branch）；**`tf.cond` 两支都进图**
  （官方逐字：traces and adds both branches）。**两条分支的 Python 代码在追踪期都会跑。**
- 代价的账：`tf.cond` = 图更大、追踪 1 次；Python `if` + 布尔参数 = 图更小、**每个取值追一次**；
  Python `if` + 自由变量 = 图最小、**并且在改了变量之后静默算错**。
- 循环同理：`tf.while_loop` 的图**与迭代次数无关**；Python 循环**展开**、随 N 线性增长。
- `jit_compile=True` 的门槛是「**整函数**可编译」，否则 `errors.InvalidArgumentError`；
  默认 `None` 时**只在 TPU 上走 XLA**。`tf.xla.experimental.compile` **仍在但已废弃**，
  且**官方没给移除日期**。
- 往下一站：[03-custom-ops-and-export](03-custom-ops-and-export.md)（算子粒度与图的落地格式）；
  XLA 内部回链 [60-compilers/xla](../../60-compilers/xla/01-hlo-and-shape-inference.md)。

## 参考

**本主题的配套程序**：`code/02_control_flow.py`（三类断言齐全，失败非零退出）。

- **条件转换 / Python 条件只有一支 / `tf.cond` 两支都进图 / 循环转换规则**：
  `site/en/guide/function.ipynb`
  <https://github.com/tensorflow/docs/blob/35e0922e059d7bc6d515a83e03a7494f0640c314/site/en/guide/function.ipynb>
- **「追踪会走遍所有代码路径」「tracing executes both branches」**：
  `tensorflow/python/autograph/g3doc/reference/control_flow.md`
  <https://github.com/tensorflow/tensorflow/blob/v2.21.0/tensorflow/python/autograph/g3doc/reference/control_flow.md>
- **符号张量取 `bool` / 迭代的官方报错文案与解法**：`tensorflow/python/framework/tensor.py:302-319`
  <https://github.com/tensorflow/tensorflow/blob/v2.21.0/tensorflow/python/framework/tensor.py#L302-L319>
- **`jit_compile` 的语义与「整函数必须可编译」**：`polymorphic_function.py:1591-1602`
  <https://github.com/tensorflow/tensorflow/blob/v2.21.0/tensorflow/python/eager/polymorphic_function/polymorphic_function.py#L1591-L1602>
- **`xla.experimental.compile` 的 `@deprecated(None, …)`**：`python/compiler/xla/xla.py:62-67`
  <https://github.com/tensorflow/tensorflow/blob/v2.21.0/tensorflow/python/compiler/xla/xla.py#L62-L67>

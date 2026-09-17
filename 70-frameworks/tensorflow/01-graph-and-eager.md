---
title: graph 与 eager：TraceType 就是被固化进图的那份假设清单
slug: tensorflow-graph-and-eager
domain: 70-frameworks
topic: tensorflow
level: 2
difficulty: intermediate
prerequisites:
  - 70-frameworks/pytorch/01-dispatch-and-aten
tags: [tensorflow, tf.function, tracing, retracing, trace-type, function-type, concrete-function, literal-trace-type, reduce-retracing]
related:
  - 70-frameworks/pytorch/01-dispatch-and-aten
  - 70-frameworks/tensorflow/02-control-flow-and-xla
  - 70-frameworks/jax/01-jit-and-tracing
code:
  - code/01_tracing_and_retracing.py
status: stable
updated: 2026-09-17
---

# graph 与 eager：TraceType 就是被固化进图的那份假设清单

> `tf.function` 第二次调用同一个函数时不再跑你的 Python 代码——它拿每个实参的
> **`TraceType`**（Tensor 看 **dtype + shape**，Python 原生值看**值本身**）拼出一个
> **`FunctionType`** 当**缓存键**，命中就复用上次追踪出来的 `ConcreteFunction`。
> 所以「图固化了什么」这件事，答案是**可枚举的**：**凡是进了 `FunctionType` 的东西，
> 图就替它做了假设**。而**TF 默认对 shape 敏感**——官方自己的重追踪告警里就点名了这件事。

## 0. 背景与动机

### 一个具体的对照场景

```
对同一个函数体 f(x, k) 调用 1000 次，输入 shape 一直在变。
```

| | 每次都经过 | 代价 |
|---|---|---|
| **eager** | Python 函数体 → 逐个算子调用 → kernel | **Python 那一段每次都要付** |
| **`tf.function`** | 第一次 trace 出图，之后**重放图** | 假设被打破时要**重新 trace** |

两边都不是免费的：eager 付的是 Python 解释开销，图付的是**固化假设**——
而假设一旦和现实不符，代价是**重追踪**（retracing）。本文要回答的是：

> **`tf.function` 到底把哪些东西固化进了图？**

### 本文在本域主线上的位置：**完整展开本主题的主线**

本域的主线是「**框架要把哪些假设固化进图**」（见 `70-frameworks/README.md` 与 roadmap）。
在 TF 这个主题里，主线有一个**特别干净的落点**：

> **`TraceType` 就是那份假设清单本身。**
> 它不是一个比喻：`tf.function` 的缓存键就是由各参数的 `TraceType` 拼成的 `FunctionType`，
> 命不命中、要不要重追踪，全靠它。**清单里有什么，图就假设什么；清单里没有的，图就不管。**

本文完整展开这条主线（§1 讲清单的内容与匹配规则，§2 讲清单之外会发生什么、
§3 讲清单怎么被写小、§4 讲四条失败模式）。同主题的另两篇只挂主线的一段：
[02-control-flow-and-xla](02-control-flow-and-xla.md) 挂**代价项**（固化分支的代价是图的节点数），
[03-custom-ops-and-export](03-custom-ops-and-export.md) 挂**判据来源**（算子粒度与导出格式
决定假设被写进哪个载体）。与 JAX 的镜像对照见 [70-frameworks/jax/01-jit-and-tracing](../jax/01-jit-and-tracing.md)。

### 为什么这里容易记错

三个流行说法都**近乎**正确，但错的那一点恰好是本文的主角：

1. **「TF 用图，所以对 shape 不敏感」**——错。官方自己的告警文案里直接点名
   「**passing tensors with different shapes**」（§1.4 有逐字）。
2. **「Python 标量是静态的」**——在 TF 里**恰好成立**，但**机制不是「静态参数」**，
   而是「**按值特化**」：`3` 与 `4` 是两个不同的缓存键。这句话在 JAX 里**是错的**，
   两边的默认行为正好相反（§1.3）。
3. **「`tf.function` 是一次编译」**——它是**追踪 + 缓存**；编译（XLA）是**另一层**，
   见 [02-control-flow-and-xla](02-control-flow-and-xla.md) §3。

## 1. 原理：缓存键是怎么拼出来的

### 1.1 一次调用的三步

官方逐字（源码见文末参考）：

> When called, a `tf.function` first evaluates the type of each input argument using the
> `tf.types.experimental.TraceType` of each argument. This is used to construct a
> `tf.types.experimental.FunctionType` describing the signature of the desired
> `ConcreteFunction`. We compare this `FunctionType` to the `FunctionType`s of existing
> `ConcreteFunction`s. If a matching `ConcreteFunction` is found, the call is dispatched to it.
> If no match is found, a new `ConcreteFunction` is traced for the desired `FunctionType`.

拆成三步：

1. **每个实参 → 一个 `TraceType`**；
2. **所有 `TraceType` → 一个 `FunctionType`**（这就是缓存键）；
3. **拿键查表**：命中 → 重放那张图；不命中 → **跑一遍 Python 函数体**，把算子记进新图。

### 1.2 清单的内容：官方给的 `TraceType` 规则

| 实参类型 | `TraceType` 由什么参数化 | 后果 |
|---|---|---|
| `Tensor` | **dtype + shape**（ranked 是 unranked 的子类型；**定长维是未知维的子类型**） | 换 dtype 或换 shape → **新键** |
| `Variable` | 同 `Tensor`，**外加一个唯一 resource ID** | 换一个 Variable 变量 → 新键 |
| Python 原生值（`int` / `float` / `str` / `bool` / `None`） | **值本身**：`3` 的类型是 `LiteralTraceType<3>`，**不是 `int`** | **3 与 4 是两个键** |
| 有序容器（`list` / `tuple`） | **按元素逐个**，且**保序**（`[1,2]` 与 `[2,1]` 不同） | 元素个数或顺序变了 → 新键 |
| 映射（`dict`） | 键相同、值是**值的类型**；**与键序无关** | 键序变了**不**产生新键 |
| 其他 Python 对象 | 先比**对象身份**（`id`/`is`），再比 `==`（只留弱引用） | 同一个可变对象改了属性**仍是同一个键**（§4 第三个失败模式） |

### 1.3 一个必须记住的对照：TF 与 JAX 对 Python 标量的默认行为**相反**

官方逐字（同页，TraceType 规则第二条）：

> For Python primitive values, the type corresponds to the **value** itself. For example,
> the `TraceType` of the value `3` is `LiteralTraceType<3>`, not `int`.

所以 `f(x, 3)` 与 `f(x, 4)` 是**两个不同的缓存键**，会**各追一次**。
这就是「Python 标量是静态的」这句话在 TF 里成立的原因——**但它的机制是「按值进缓存键」，
不是「静态参数声明」**。JAX 那边默认**不**做这件事（详见
[70-frameworks/jax/01-jit-and-tracing](../jax/01-jit-and-tracing.md)）。
**同一个 `k`，在 TF 里是缓存键的一部分，在 JAX 里默认不是**——
这是本域最值钱的一条对照，也是「迷你实现」那一轨唯一能同时约束两个主题的量化对象。

### 1.4 为什么「TF 对 shape 不敏感」是错的

`tf.function` 有一个**自动重追踪告警**，它的文案里逐字点名了 shape。官方告警的触发条件
是「最近 **10** 次调用里 **≥5** 次触发了 tracing」（两个阈值常量见文末源码行）：

```
FREQUENT_TRACING_WARNING_MAX_CALL_HISTORY = 10
FREQUENT_TRACING_WARNING_THRESHOLD        = 5
```

> …Tracing is expensive and the excessive number of tracings could be due to
> (1) creating @tf.function repeatedly in a loop, **(2) passing tensors with different
> shapes**, (3) passing Python objects instead of tensors. … For (2), @tf.function has
> **reduce_retracing=True** option that can avoid unnecessary retracing.

**官方在告警里直接承认「换 shape 会导致重追踪」。** 另两条缓解路径见 §3。

### 1.5 推导：子类型方向为什么是「用 `None` 维才划算」

匹配不是相等匹配，而是**子类型匹配**。官方逐字：

> If multiple matches are found, the most specific signature is chosen. Matching is done by
> subtyping… For example, `TensorShape([1, 2])` is a subtype of `TensorShape([None, None])`
> and so a call to the tf.function with `TensorShape([1, 2])` can be dispatched to the
> `ConcreteFunction` produced with `TensorShape([None, None])`…

把这一步展开成「复用的条件」：

1. 记缓存里那张图的键为 `K`，本次调用的键为 `C`；
2. **命中要求 `C <: K`**（本次调用比缓存键**更具体**）；
3. 若 `K` 用 `None` 维 trace（`K = [None, None]`），则**任何**同秩的具体 shape 都满足 `C <: K` ⇒ **一张图服务所有同秩输入**；
4. 若 `K` 是具体形状（`K = [1, 2]`），那么只有 `C = [1, 2]` 满足 `C <: K` ⇒ **只服务那一个 shape**；
5. 反过来 `K = [1, 2]`、`C = [None, None]` **不成立**——未知维**不是**定长维的子类型。

**结论（方向性）**：**「用 `None` 维 trace 才能复用得更广；反过来不成立」**。
另外「已知秩 是 未知秩 的子类型」意味着**秩也参与**：把 `None`（未知秩）当键记下来，
它不能服务任何已知秩的调用；而反过来可以。这条方向性在
`code/01_tracing_and_retracing.py` 的 §A 里用一条**独立重写的谓词**逐项验过（169 对形状，差异 0）。

### 1.6 追踪次数是可数的

缓存键是有限集合，所以「这段调用序列要追几次」是一个**可计算的量**：

```
traces = |{ distinct FunctionType(call) : call ∈ 调用序列 }|   （无子类型复用，即每键一张图）
```

严格地说，因为有子类型复用，实际次数是「按到达顺序做一次在线去重、只在新键不是任何
已有键的子类型时才新增」——这是贪心的，`code/01_tracing_and_retracing.py` 的 §A 用一条
**另写的** `ref_trace_count` 从零重算，与迷你追踪器的实际计数逐项一致。

## 2. 执行映射：层级与「清单之外」

### 2.1 四层，不是一个东西

官方在源码里给出的基类链（`tensorflow/python/types/core.py`）：

| 层 | 官方定位（逐字） | 角色 |
|---|---|---|
| `PolymorphicFunction` | 「Base class for polymorphic graph functions. Graph functions are Python callable objects that **dispatch calls to a TensorFlow graph**. Polymorphic graph functions can be backed by **multiple TF graphs**, and automatically select the appropriate specialization based on the type of input they were called with. They may also create specializations on the fly if necessary, for example by tracing.」 | **缓存层**：`tf.function` 的对象形态 |
| `ConcreteFunction` | 「Base class for differentiable graph functions. A `ConcreteFunction` **encapsulates the original graph function definition** with support for differentiability under `tf.GradientTape` contexts.」 | **一次 trace 的产物** |
| `AtomicFunction` | 「A Python callable for functions in the TF Runtime.」 | **真正持有图缓存的那一层**（`_cached_graph`）——v2.21.0 里它夹在中间 |
| `tf.Graph` | —— | 图本体 |

> ⚠ **本项目订正**：计划里曾把 `ConcreteFunction` 写成「can be thought of as a wrapper
> around a `tf.Graph`」并称是官方逐字。**在 v2.21.0 里查不到这句**；实际存在的文本是上表那一段。
> 另外 v2.21.0 的 `ConcreteFunction` 与图之间**还夹着 `AtomicFunction`**，
> 所以「`tf.function` → `ConcreteFunction` → `tf.Graph`」这条三段式**少了中间一层**。

**`FunctionDef` 不在这条链上**：它是 proto 的序列化形态，进出运行时要过一次
`TF_FunctionToFunctionDef` / `TF_FunctionImportFunctionDef`（`tensorflow/c/c_api.h` 的声明
见文末参考）。**把 `FunctionDef` 当成运行时对象是常见的错**。

### 2.2 清单之外：Python 副作用只发生一次

官方逐字：

> Side effects, like printing, appending to lists, and mutating globals, can behave
> unexpectedly inside a `tf.function`, sometimes executing twice or not all. **They only
> happen the first time you call a `tf.function` with a set of inputs.** Afterwards, the
> traced `tf.Graph` is reexecuted, **without executing the Python code**.

`code/01_tracing_and_retracing.py` 的 §2 把这个对照跑了：同一个函数体调用 5 次，
套 `tf.function` 时 Python 侧副作用只发生 **1** 次，eager 下发生 **5** 次
（**本仓库迷你追踪器实测，不是 TensorFlow**）。

### 2.3 另一个「清单之外」：全局变量与自由变量不进 TraceType

官方逐字的注解：

> Note: `TraceType` is based on the `tf.function` input parameters so **changes to global and
> free variables alone will not create a new trace.**

这**不是**一条中性说明，而是一个**陷阱**：改了自由变量之后，函数**不会重追踪**，
于是**照旧用图里冻着的旧值**。见 §4 第三条。

### 2.4 诊断这三样东西

| 手段 | 说明 |
|---|---|
| `experimental_get_tracing_count()` | 源码 docstring 逐字：「**Returns the number of times the function has been traced.**」。**它不在 `tf.function` 手册正文里，只存在于源码 docstring**——查不到公开 API 文档页 |
| `pretty_printed_concrete_signatures()` | 打印所有具体签名（手册里有用法示例，定义在源码 `polymorphic_function.py`） |
| 自动重追踪告警 | §1.4 那段；只有当「10 次窗口里 ≥5 次触发」时才打，且每个 detector 最多打 2 次 |

## 3. 实现与演进：怎么把清单写小

三条缓解路径，代价各不相同：

| 路径 | 做法 | 效果 | 代价 |
|---|---|---|---|
| **固定 `input_signature`** | `@tf.function(input_signature=(tf.TensorSpec(shape=[None], dtype=...),))` | **强制只有一种** FunctionType | 官方逐字：「Calls that cannot be dispatched to this `FunctionType` will **throw an error**」——**拒绝了**原本能跑通的输入 |
| **`None` 维** | 用带未知维的 `TensorSpec` 先 trace | 同秩的具体 shape 全部复用（§1.5） | 图上多一个未知维；秩仍然要固定 |
| **`reduce_retracing=True`** | 出缓存未命中时**自动松弛**成更泛化的类型再 trace | 常见 shape 变化不再重追踪 | 定义上就不如 `input_signature` 精确 |

官方对 `reduce_retracing` 的逐字说明：

> When `reduce_retracing` is enabled, `tf.function` automatically identifies **supertypes** of
> the input types it is observing and chooses to trace **more generalized graphs**
> automatically. It is less efficient than setting the `input_signature` directly but useful
> when many types need to be supported.

**代价的量化（迷你实现档）**：追踪一次要跑一遍 Python 函数体并把算子记下来，
重放只是按图执行。在 `code/01_tracing_and_retracing.py` 的 §B 里量到：
一个 2 节点的图，**trace ≈ 28 μs**、**replay ≈ 4 μs**、eager ≈ 4 μs
（**本仓库迷你追踪器实测，不是 TensorFlow**）。
**注意最后两个数**——见 §4 第四条。

## 4. 踩坑与失败模式

### 失败模式 1：以为「TF 对 shape 不敏感」，于是不做任何 shape 治理

**现象**：训练脚本里 shape 每次都变，日志被「N out of the last M calls to … triggered
tf.function retracing」刷屏，吞吐掉下去。
**成因**：`Tensor` 的 `TraceType` **含 shape**，换 shape 就是换缓存键（§1.2、§1.4）。
**规避**：三条缓解路径择一（§3）；诊断用 §2.4 的两个计数入口。

### 失败模式 2：把 Python 标量当普通配置参数传

**现象**：`@tf.function def f(x, k)`，用 `k=1,2,3,…` 反复调用——每换一个 `k` 就重追踪一次。
**成因**：Python 原生值的 `TraceType` **是值本身**（`LiteralTraceType<3>`），**并且这个值被
烧进了图**（字面量进了节点属性，不是占位符）。
**规避**：把会变的量改成 **Tensor 参数**（进图当输入），或者用 `input_signature` 把它钉死；
诊断同样是 `experimental_get_tracing_count()`。

### 失败模式 3：改了自由变量却指望图跟着变（**静默算错**）

**现象**：函数体里读一个闭包/全局的量，改了它之后调用**不报错**，但结果还是旧的。
**成因**：`TraceType` 只看**输入参数**，自由变量不进键，所以**不会重追踪**（§2.3 官方逐字）。
**规避**：凡是要随调用变化的东西，**必须走参数**（Tensor 或显式入参）；
调试期可以拿 `experimental_get_tracing_count()` 的回零/不变作为「它没重追」的信号。
**这条是最坏的一类**：不报错、不算崩，只是**结果错了**。

### 失败模式 4：把 `reduce_retracing` 当万能药

**现象**：加了 `reduce_retracing=True` 以为从此不重追踪，结果某些输入仍然追。
**成因**：它松弛的是**已知维 → 未知维**，**不跨秩**。`(2,2)` 与 `(8,)` 仍会各追一次。
**规避**：形状的**秩**要稳定（batch 维之外的维数固定），或者用 padding 到固定 bucket。
另见 §3 的代价表：**它救的是追踪次数，不是每次调用的成本**。

## 5. 实测数据

**测量条件**：`code/01_tracing_and_retracing.py` 经 `make run` 在本机跑出
（python3 3.13.5 / numpy 2.4.4），全部 21 条断言通过（A 7 / B 6 / C 8）。

> **★ 限定语（本域强制）**：下面每一个数字都是**本仓库的迷你追踪器**跑出来的，
> **不是 TensorFlow**，也**不是任何真实框架**。本机**没有装 TF**（一个环境都没有），
> 所以这里没有任何「真实 TF 实测」。这些数字支持的是
> 「按 `TraceType` 查表这套机制可以怎样被计数与观测」这类**机制**结论，
> **不支持任何关于真实 TensorFlow 的结论**。
> 本文里**关于真实 TF 的每一条行为断言**都以 §参考 里的**一手 URL** 为依据
> （官方文档仓库 + `v2.21.0` 源码行），**本仓库不复核其测量口径**。

**（1）一次调用序列的追踪次数**（迷你实现实测，调用序列见 `code/` 的 §A）：

| 调用序列（shape） | 追踪次数 | 独立参考实现重算 |
|---|---|---|
| `(2,2), (2,2), (3,3), (2,2), (None,None), (7,7), (3,3)` | **3** | **3** |

**（2）三条路的耗时**（单位 μs，迷你实现实测；量级比较，不是精确值）：

| 路径 | 耗时 |
|---|---|
| eager 一次调用（2 个算子） | 3.8 |
| replay 同一张图 | 4.2 |
| 预解析闭包（把链折成一次 numpy 调用） | 2.8 |
| **trace 一次**（2 个节点的图） | **28.3** |

**★ 这一栏是本程序第一版被实跑推翻的断言**：第一版写的是
「replay 一次比 eager 一次便宜」，实跑得到 **4.2 μs vs 3.8 μs——更贵**。
在纯 Python 的迷你实现里，**图是被「解释执行」的数据结构**，每回合仍要付 dict 查表与
逐节点分派；「折叠」真正省下的是**可序列化、可交给编译器**，不是每次调用的时间。
**时间上的好处要等图被编译掉**——这正是 [02](02-control-flow-and-xla.md) §3 的位置。

**（3）告警阈值的模型核对**：按官方常量 `(10, 5)` 复算——
10 次窗口内 4 次触发**不**告警、5 次**告警**（迷你实现实测，与官方判定流程一致）。

## 6. 小结与延伸

- `tf.function` 的缓存键是 **`FunctionType`**，由各实参的 **`TraceType`** 拼成；
  `Tensor` 看 **dtype + shape**，**Python 原生值看值本身**（`LiteralTraceType<3>`）。
- **TF 默认对 shape 敏感**（官方告警逐字点名 `passing tensors with different shapes`）；
  缓解路径是 `input_signature` / `None` 维 / `reduce_retracing=True`。
- **子类型方向**：`[1,2] <: [None,None]` 成立、反向不成立；已知秩 `<:` 未知秩 成立、反向不成立。
  **所以「先用 `None` 维 trace」才划算，反过来不成立。**
- 层级是 `tf.function`(`PolymorphicFunction`，**缓存层**) → `ConcreteFunction`
  →（**v2.21.0 还有一层 `AtomicFunction`**）→ `tf.Graph`；**`FunctionDef` 是 proto，不是运行时对象**。
- 清单之外的两件事：**Python 副作用只在 tracing 时发生一次**；
  **自由变量与全局变量的改动不产生新 trace**（后者是静默算错的坑）。
- 往下一站：[02-control-flow-and-xla](02-control-flow-and-xla.md)（控制流固化了哪一支、XLA 是另一层）、
  [03-custom-ops-and-export](03-custom-ops-and-export.md)（算子粒度与图的落地格式）、
  [70-frameworks/jax/01-jit-and-tracing](../jax/01-jit-and-tracing.md)（**镜像对照**：同一个问题，
  cache key 换成了「JAX types + 静态参数 + 函数对象 identity」）。

## 参考

**本主题的配套程序**：`code/01_tracing_and_retracing.py`（三类断言齐全，失败非零退出）。
**本项目核过的一手来源**（全部在 `code/specs.py` 里逐条登记；文档类引文钉在
`tensorflow/docs` 的 commit `35e0922e059d7bc6d515a83e03a7494f0640c314`，取用于 2026-09-17）：

- **TraceType 规则 / FunctionType 派发 / 子类型与最具体者优先 / 副作用 / `reduce_retracing` /
  `input_signature`**：`site/en/guide/function.ipynb`
  <https://github.com/tensorflow/docs/blob/35e0922e059d7bc6d515a83e03a7494f0640c314/site/en/guide/function.ipynb>
- **重追踪告警的文案与两个阈值常量**（`polymorphic_function.py:110-111` 与 `:155-167`）
  <https://github.com/tensorflow/tensorflow/blob/v2.21.0/tensorflow/python/eager/polymorphic_function/polymorphic_function.py#L110-L111>
- **`experimental_get_tracing_count` 的 docstring**（同文件 `:770`）
  <https://github.com/tensorflow/tensorflow/blob/v2.21.0/tensorflow/python/eager/polymorphic_function/polymorphic_function.py#L770>
- **层级的三段 docstring**：`types/core.py`（`PolymorphicFunction` / `ConcreteFunction`）
  与 `polymorphic_function/atomic_function.py`
  <https://github.com/tensorflow/tensorflow/blob/v2.21.0/tensorflow/python/types/core.py#L170-L200>
- **`FunctionDef` 的 C API 入口**（`c_api.h:1147` 与 `:1159`）
  <https://github.com/tensorflow/tensorflow/blob/v2.21.0/tensorflow/c/c_api.h#L1147-L1165>

**回链**：eager 那条路的「层数可数」见 [70-frameworks/pytorch/01-dispatch-and-aten](../pytorch/01-dispatch-and-aten.md)；
XLA 的 HLO / 融合 / layout 正文在 [60-compilers/xla](../../60-compilers/xla/01-hlo-and-shape-inference.md)。

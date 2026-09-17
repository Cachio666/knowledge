---
title: jit 与追踪：cache key 就是假设清单，而 Python 标量默认不在里面
slug: jax-jit-and-tracing
domain: 70-frameworks
topic: jax
level: 2
difficulty: intermediate
prerequisites:
  - 70-frameworks/tensorflow/01-graph-and-eager
  - 70-frameworks/pytorch/01-dispatch-and-aten
tags: [jax, jit, tracing, cache-key, static-argnums, weak-type, shape-polymorphism, jaxpr, explain-cache-misses]
related:
  - 70-frameworks/tensorflow/01-graph-and-eager
  - 70-frameworks/jax/02-transformations
  - 60-compilers/xla/01-hlo-and-shape-inference
code:
  - code/01_jit_cache_key.py
status: stable
updated: 2026-09-17
---

# jit 与追踪：cache key 就是假设清单，而 Python 标量默认不在里面

> `jax.jit` 第一次调用某个函数时把它 trace 成一张 **jaxpr** 并把结果缓存起来，
> 缓存键是三项：**参数的 JAX types**（形状 + dtype）、**静态参数的值**、
> **函数对象的 identity**。
> 所以「图固化了什么」是可枚举的：**进了 cache key 的东西，图就替它做了假设**。
> 而这里有一条与 TensorFlow **正好相反**的默认行为：
> **Python 标量不是静态的**——官方 docstring 逐字
> 「**no arguments are treated as static**」。`f(x, 3)` 与 `f(x, 4)` **只追踪一次**。

## 0. 背景与动机

### 一个具体的对照场景

```
对同一个函数体 f(x, k) 调用 1000 次，k 是 Python 整数配置项，x 的 shape 一直在变。
```

| | 每次都经过 | 代价 |
|---|---|---|
| **eager（不 jit）** | Python 函数体 → 逐个 primitive → 逐次调用 numpy/后端 | Python 那一段**每次都付** |
| **`jax.jit`** | 第一次 trace 成 jaxpr，之后**重放那张图** | 假设被打破时**重新追踪 + 重新编译** |

两边都不是免费的：eager 付的是 Python 解释开销，图付的是**固化假设**——
而假设一旦和现实不符，代价是**重追踪**。本文要回答的是：

> **`jax.jit` 到底把哪些东西固化进了图？**

### 本文在本域主线上的位置：**完整展开本主题的主线**

本域的主线是「**框架要把哪些假设固化进图**」（见 [70-frameworks/README](../README.md) 与 roadmap）。
在 JAX 这个主题里，主线有一个**特别干净的落点**：

> **cache key 就是那份假设清单本身。**
> 它不是一个比喻：三项内容一条条可数，命不命中、要不要重追踪，全靠它。
> **清单里有什么，图就假设什么；清单里没有的，图就不管。**

本文完整展开这条主线（§1 讲清单的内容，§2 讲清单**写不小**的那一面——形状多态，
§3 讲怎么看见清单命没命中，§4 讲四条失败模式）。同主题的另两篇只挂主线的一段：
[02-transformations](02-transformations.md) 挂**代价项**（变换叠在同一层上，图不随 batch 变大），
[03-sharding-and-differences](03-sharding-and-differences.md) 挂**判据来源**（多设备那条路谁写假设）
与收束性的三者对照。

### ★ 本文与 TensorFlow 的 01 是**镜像**，两篇要对着读

| | TF `tf.function` | JAX `jax.jit` |
|---|---|---|
| 缓存键 | 各实参 `TraceType` 拼成的 **`FunctionType`** | 参数的 **JAX types** + 静态参数值 + **函数对象 identity** |
| **Python 标量** | **按值特化**：`3` 的 TraceType 是 `LiteralTraceType<3>`，**不是 `int`** | **默认不特化**：变成 **weak-typed 的 traced 实参** |
| 后果 | `f(x, 3)` 与 `f(x, 4)` ⇒ **两次追踪** | `f(x, 3)` 与 `f(x, 4)` ⇒ **一次追踪** |

**「Python 标量是静态的」这句话，在 TF 里对、在 JAX 里错。** 这是本域最值钱的一条对照，
两边的编号程序各自把它跑了出来（本文 §1.3 与 §4 第一条；
TF 侧的逐字见 [tensorflow/01-graph-and-eager](../tensorflow/01-graph-and-eager.md) §1.3）。

### 为什么这里容易记错

三个流行说法都**近乎**正确，但错的那一点恰好是本文的主角：

1. **「JAX 的 Python `int`/`float` 参数默认是静态值」**——**错**。默认**一个都不静态**，
   要静态必须显式 `static_argnums` / `static_argnames`（§1.3 有逐字）。
2. **「JAX 对 shape 不敏感」**——**也错**。三者（TF / JAX / PyTorch）**默认都对 shape 敏感**；
   JAX 的缓存小节里直接写着「new shapes or dtypes trigger a fresh trace and compile」。
3. **「`jax.jit` 是个编译器」**——它是**追踪 + 缓存**；编译（XLA）是**另一层**，
   HLO / 融合 / layout 的正文在 [60-compilers/xla](../../60-compilers/xla/01-hlo-and-shape-inference.md)。

## 1. 原理：cache key 是怎么拼出来的

### 1.1 三项清单

官方给的原文（**dev 文档**，`https://docs.jax.dev/en/latest/201/jit.html` §JIT and caching）：

> Suppose we define f = jax.jit(g). When f is first invoked, it gets compiled and the
> resulting executable is cached. The cache key includes: the JAX types of the arguments
> — new shapes or dtypes trigger a fresh trace and compile; the values of any static
> arguments; and the object identity of the function g itself.

拆成三项：

| # | 项 | 内容 | 什么时候变 |
|---|---|---|---|
| 1 | **参数的 JAX types** | 形状 + dtype（本仓库的迷你实现还带上 weak/strong，见下方说明） | 换 shape、换 dtype |
| 2 | **静态参数的值** | 只有显式 `static_argnums` / `static_argnames` 点名的实参才有 | 那个实参的值变了 |
| 3 | **函数对象的 identity** | 是**哪个**函数对象，不是「什么函数体」 | 换一个对象就换一张缓存 |

> ⚠ **版本提醒**：上面这段引文出自 **dev 文档**（它跟 `main`，不是 0.11.1 的冻结快照）。
> **`jax-v0.11.1` 的 tag 里那句话的第三项措辞不同**：写的是
> 「**the identity of the function `g` itself**」，而 dev 版写的是
> 「the **object** identity of the function g itself」。
> **两者不是同一句**，本仓库把两版都登记在 `code/specs.py` 的 `DOC_DRIFT` 里，
> 引用时**必须注明引的是哪一版**。（`docs.jax.dev` 没有 0.11.1 版本页，
> `/en/v0.11.1/…` 与 `/en/stable/…` 都是 404。）

**第 3 项的实际后果**（同页，dev 文档）：

> That last point has a practical consequence: avoid calling jax.jit on temporary
> functions defined inside loops or other inner scopes. The cache relies on the object
> identity of the function, so **freshly-created lambdas and partial objects — even ones
> wrapping the same underlying code — look like new functions every time, and recompile
> every time**.

这句话是本主题 **C2** 的官方依据：**两个内容相同的 lambda 是两次编译**。

### 1.2 清单怎么用：抽象化 → 拼键 → 查表

一次调用分三步：

1. **每个实参 → 一个 JAX type**（`abstractify`：形状 + dtype）；
2. **所有实参的 JAX type + 静态参数值 + 函数 identity → 一个 cache key**；
3. **拿键查表**：命中 → 重放上次那张 jaxpr；不命中 → **跑一遍 Python 函数体**，
   把 primitive 记进新 jaxpr，并**编译**。

`jaxpr` 是这张图的载体，官方定义逐字（`https://docs.jax.dev/en/latest/601/jaxpr.html`）：

> It is **explicitly typed, functional, first-order, and in algebraic normal form (ANF)**.

文法：`jaxpr ::= { lambda Var* ; Var+. let Eqn* in [ Expr+ ] }`。
**分号前是 `constvars`（烧进图的常量）、分号后是 `invars`（每次调用都要喂进来的输入）**——
这个区分正是本文 §1.3 的关键。（`jax.make_jaxpr()` 返回的是 **`ClosedJaxpr`**：jaxpr + 它闭包住的常量；
详细展开在 [02](02-transformations.md) §2。）

### 1.3 ★ 必须纠正的一条：Python 标量**默认不是静态的**

`jax.jit` 的 docstring 逐字（`jax/_src/api.py:253-254`，`jax-v0.11.1`）：

> If neither `static_argnums` nor `static_argnames` is provided, **no arguments are
> treated as static.** If `static_argnums` is not provided but `static_argnames` is, or
> vice versa, JAX uses `inspect.signature(fun)` to find any positional arguments that
> correspond to `static_argnames` (or vice versa). If both `static_argnums` and
> `static_argnames` are provided, `inspect.signature` is not used, and **only actual
> parameters listed in either** `static_argnums` or `static_argnames` **will be treated
> as static**.

紧接着的一句（`:249-251`）说明静态参数值进键的后果：

> Calling the jitted function **with different values for these constants will trigger
> recompilation**.

**所以把这两条合起来读**：

- **不标 `static_argnums` 时**：Python 标量是一个**被 trace 的实参**——
  它的 aval 是 `ShapedArray((), 'int32', weak_type=True)` 这样的东西，
  **`3` 与 `4` 抽出来一模一样** ⇒ **同一个 cache key** ⇒ **只追踪一次**；
- **标了 `static_argnums=(1,)` 之后**：那个实参的**值**进键 ⇒ `3` 与 `4` 是**两个键**
  ⇒ **两次追踪**。

**这不是一个无关紧要的默认值**，它直接决定了标量在图上以什么身份出现：

| | 不标 static（默认） | 标了 `static_argnums` |
|---|---|---|
| 进 cache key？ | **否** | 是（按值） |
| 在 jaxpr 里的身份 | **`invars` 里的一项**（每次调用喂进来） | **`constvars` / 常量**（烧进图） |
| `3` 与 `4` 的追踪次数 | **1 次** | **2 次** |

`code/01_jit_cache_key.py` 的 §C2 把这个对照跑了出来（**本仓库迷你追踪器实测，不是 JAX**）：
同一个 `f(v, k) = v * k`，默认调用时 jaxpr 有 **2 个 invar**；标了 `static_argnums` 之后
只剩 **1 个 invar**。**进不进 cache key，决定进不进图。**

### 1.4 与 TF 的镜像：同一条规则，两个相反的默认

| | TF `tf.function` | JAX `jax.jit` |
|---|---|---|
| Python 标量的类型表示 | **值本身**：`LiteralTraceType<3>`，**不是 `int`** | **weak-typed aval**：`3` 与 `4` 相同 |
| `f(x, 3)` / `f(x, 4)` | **两次追踪** | **一次追踪** |
| 标量在图上的身份 | **op 的字面量属性**（烧进节点） | **jaxpr 的一个 invar**（或，标了 static 才变常量） |

TF 侧的逐字见 [tensorflow/01-graph-and-eager](../tensorflow/01-graph-and-eager.md) §1.3
（`TraceType` 规则第二条）。**这是同一套机制在两个框架里选了不同的默认**——
不是谁对谁错，而是「标量算不算假设」这一个问题的两个答案。

**注意这里没有「哪个更快」的结论**：JAX 这条路省掉了重追踪，
但代价是那个标量**变成了一个运行时输入**（§4 第一条）。

### 1.5 追踪次数是可数的

cache key 是有限集合，所以「这段调用序列要追几次」是一个**可计算的量**：

```
traces = |{ cache_key(call) : call ∈ 调用序列 }|
```

它与 TF 那边（考虑子类型复用后的贪心去重）不同：JAX 这一侧**本模型里没有子类型复用**，
就是**不同键的个数**。`code/01_jit_cache_key.py` 的 §A 用一条**另写的** `ref_trace_count`
从零重算，与迷你追踪器的实际计数逐项一致。

## 2. 执行映射：清单、图、以及「清单写不小」的那一面

### 2.1 三层，不是一个东西

| 层 | 是什么 | 谁持有 |
|---|---|---|
| **`JitWrapped`** | `jax.jit(g)` 的返回物 | **缓存层**：每一个 jitted 函数一张缓存 |
| **`jaxpr`** / `ClosedJaxpr` | 一次 trace 的产物（+ 它闭包住的常量） | 追出来的那张图 |
| **编译产物** | jaxpr 交给 XLA 之后的东西 | XLA（正文在 [60-compilers/xla](../../60-compilers/xla/01-hlo-and-shape-inference.md)） |

**一个反直觉但重要的细节**：`jax.jit` 的**返回类型注解**在源码里是 `-> pjit.JitWrapped`
（`jax/_src/api.py` 的三处重载，`:184` / `:201` / `:217`）。原因是 **`pjit` 已经被并进 `jax.jit`**
了——这条演进的正文在 [03](03-sharding-and-differences.md) §2。

**`jit` 是透明的**：在别的变换（`vmap` / `grad` / 另一个 `jit`）之下调用一个 jitted 函数时，
它**不建自己的缓存**，只把被包的 Python 函数交给外层的 trace。
所以 `jax.grad(jax.jit(f))` 与 `jax.grad(f)` 得到同一张 jaxpr。
`code/01_jit_cache_key.py` 的 §C4 把这个跑了出来（**迷你实现实测，不是 JAX**）。

### 2.2 「清单写不小」：`jax.jit` 没有 dynamic shape 开关

`tf.function` 那边有三条缓解路径（`input_signature` / `None` 维 / `reduce_retracing=True`）。
**JAX 这一侧没有对应的东西**：官方那份 jit 文档的缓存小节只把 shapes 与 dtypes
列为**触发新追踪**的条件，**其中没有任何开关能把它变成形状多态**。

> ⚠ **核查范围声明（照 P6 的规矩写明）**：上面这条否证的范围是
> **「`https://docs.jax.dev/en/latest/201/jit.html` 这一页」**——
> 本仓库**没有**对 JAX 全库做过「有没有形状多态开关」的穷举。
> 因此准确的说法是：**在这份文档给出的缓存模型里，shapes/dtypes 是硬条件，没有例外开关。**

形状多态是**另一层**的东西，在 `jax.export`：

```python
a, b = jax.export.symbolic_shape("a, b")     # 签名逐字：symbolic_shape(shape_spec, *, constraints=(), scope=None, like=None)
```

**而且它也不免除重编译。** 官方逐字（`https://docs.jax.dev/en/latest/export/shape_poly.html`）：

> Note that such functions are **still re-compiled on demand for each concrete input
> shape they are invoked on**. **Only the tracing and the lowering are saved.**

这句值得逐词读两遍：**省下的是 tracing 与 lowering，不是 compile**。
所以「用 `jax.export` 就能一份产物吃所有 shape」是**错的**——
你能省掉的是「每个新 shape 重跑一遍 Python 追踪」，**编译仍然每个具体形状各来一次**。

`code/01_jit_cache_key.py` 的 §C6 把这个跑了出来：导出之后**同形状不重编译、换形状仍重编译**
（**迷你实现实测，不是 JAX**）。

### 2.3 诊断这三样东西

| 手段 | 说明 |
|---|---|
| `jax.config.update("jax_explain_cache_misses", True)` | cache miss 时打印原因。官方给的例子逐字：`TRACING CACHE MISS at ...: different input types: * at x, now f32[3] and before i32[3]` |
| `jax_log_compiles` | 每次编译打印一行 |
| `JAX_DUMP_IR_TO=<dir>` | 把 IR 落盘（本模型的对应物是把 jaxpr 写到目录） |
| `jax.clear_caches()` | **清掉所有**编译/追踪缓存——于是下一次调用必然重追踪 |

**诊断的起点永远是「同一次调用为什么没命中」**，而不是「为什么慢」。
`explain_cache_misses` 给的那一行直接点名了**是哪一项不一样**（`now … and before …`），
所以在调优前先把它打开，比盲猜 shape bucket 有效得多。

## 3. 实现与演进：从「每次重追」到「把清单写小」

JAX 这一侧的演进方向与 TF 不同：TF 是**给追踪加缓解选项**（`input_signature`、
`reduce_retracing`），JAX 是**把形状这件事挪到导出层**。三条路：

| 路径 | 做法 | 效果 | 代价 |
|---|---|---|---|
| **padding 到固定 bucket** | 把输入 pad 到少数几个固定形状 | cache key 数量可枚举 | 算了多余的数；padding 本身要写 |
| **把会变的量变成实参** | 别把配置当 Python 标量，直接当数组传 | 标量进 invar，`3`/`4` 共享一张图（§1.3） | 那个量在图上变成一个**输入**（§4 第一条） |
| **`jax.export` + 形状多态** | `symbolic_shape("a, b")` 导出 | 省下**每个新 shape 重跑 tracing** | **仍然逐形状重编译**（§2.2 逐字） |

三者的共同点是：**都承认 shapes/dtypes 是硬条件**，区别只在于「怎么让条件少变几次」。

## 4. 踩坑与失败模式

### 失败模式 1：把 Python 标量当配置参数传（**C1 的推论**）

**现象**：`@jax.jit def f(x, k)`，用 `k=1,2,3,…` 反复调用。
**与 TF 的预期相反**——它**不会**像 TF 那样反复重编译。但**它也不是免费的**：

1. 那个标量**变成了图的一个输入**（invars 里多一项），不再是编译期常量：
   - 本来能被**常量折叠**的运算（`x * 2`）现在不能折叠了——`2` 是个运行时值；
   - 想拿它当**形状 / 轴号 / 静态参数**用会直接失败（它是 traced 值，不是 Python `int`）。
2. **类型语义会变**：Python 标量是**弱类型**的。`x * 3`（`x` 是 float32）在 JAX 里仍是
   float32（弱类型让位给强类型）；但如果你后面拿它去和强类型混用，
   结果 dtype 取决于**谁更强**，而不是 numpy 那套「int 提升到 float64」的规则。
3. **想用 Python `if` 判断它会直接失败**（值相关的控制流，见 [03](03-sharding-and-differences.md) §1）。

**成因**：官方 docstring 那条默认——**no arguments are treated as static**（§1.3）。
**规避**：想让它当常量 → 显式 `static_argnums` / `static_argnames`（代价是每个值一份缓存与编译）；
想让它当运行时输入 → 保持默认，但**心里清楚它现在是个输入**。

### 失败模式 2：在循环里 `jax.jit(...)`（**静默地每次重编译**）

**现象**：在训练循环或某个内层作用域里写 `f = jax.jit(lambda x: ...)`，
每步都编译一次，吞吐远低于预期。
**成因**：cache key 的第 3 项是**函数对象的 identity**，而循环里每次都是**新对象**
（同页官方逐字：*freshly-created lambdas and partial objects — even ones wrapping the
same underlying code — look like new functions every time, and recompile every time*）。
**规避**：把 `jax.jit` 提到循环外；闭包要传的量走**实参**（并在必要时标 static），
而不是靠闭包捕获一个新 lambda。

### 失败模式 3：以为「导出了就免除重编译」

**现象**：用 `jax.export` 导出一份产物，以为从此任何 shape 都直接用；
结果换一个 shape 仍然付一次编译。
**成因**：官方逐字 **Only the tracing and the lowering are saved**（§2.2）——
**lowering 之后的编译不在被省下的范围内**。
**规避**：把 export 的价值定位成「省掉 tracing 与 Python 依赖、给部署一个稳定产物」，
**不要**把它当成「形状无关的一次性编译」。

### 失败模式 4：把对 shape 的敏感当成 JAX 独有的毛病

**现象**：在 TF 里被 `reduce_retracing` 惯坏了，到 JAX 里找不到等价开关，
于是以为「JAX 对 shape 更敏感」。
**成因**：**三者默认都对 shape 敏感**（§1.4 的表、[03](03-sharding-and-differences.md) §3 的表）。
差别在**缓解手段**：TF 有 `input_signature` / `None` 维 / `reduce_retracing`；
JAX 那侧要么 padding 到 bucket，要么走 export（而 export 也仍然逐形状重编译）。
**规避**：把「形状治理」当**训练脚本层**的事（bucket / 固定 batch / 定长序列），
别指望框架开关替你消掉。

## 5. 实测数据

**测量条件**：`code/01_jit_cache_key.py` 经 `make run` 在本机跑出
（python3 3.13.5 / numpy 2.4.4），全部 **28** 条断言通过（A 5 / B 11 / C 12）。

> **★ 限定语（本域强制）**：下面每一个数字都是**本仓库的迷你追踪器**跑出来的，
> **不是 JAX**，也**不是任何真实框架**。本机**没有装 JAX**（一个环境都没有），
> 所以这里没有任何「真实 JAX 实测」。这些数字支持的是
> 「按 cache key 三项查表这套机制可以怎样被计数与观测」这类**机制**结论，
> **不支持任何关于真实 JAX 的结论**。
> 本文里**关于真实 JAX 的每一条行为断言**都以 §参考 里的**一手 URL** 为依据
> （`jax-v0.11.1` 源码 + `docs.jax.dev` 的指定页），**本仓库不复核其测量口径**。

**（1）三项 cache key 各自的追踪次数**（迷你实现实测）：

| 变化 | 追踪次数 |
|---|---|
| 不标 static：`f(x, 3)` → `f(x, 4)` | **1 → 1**（同一个键） |
| 标 `static_argnums=(1,)`：`f(x, 3)` → `f(x, 4)` | **1 → 2** |
| 换 shape：`(2,2)` → `(2,3)` | 1 → 2 |
| 再换 dtype：f32 → f64 | 2 → 3 |
| 换**函数对象**（内容相同）：`g1` → `g2` | 各 1 次（合计 2） |
| `clear_caches()` 之后同一段调用 | +1 |

**（2）标量在图上的身份**（迷你实现实测）：
默认调用时 jaxpr 的 `invars` 有 **2** 项（张量 `v` + 标量 `k`）、`constvars` 为 **0**；
标 `static_argnums` 后 `invars` 只剩 **1** 项。**函数体里写的字面量（`+ 1.0`）则是常量。**

**（3）三条路的耗时**（单位 μs，迷你实现实测；**量级比较，不是精确值**）：

| 路径 | 耗时 | 与 eager 之比 |
|---|---|---|
| eager 一次调用（2 个算子） | 5.8 | 1.00× |
| replay 同一张 jaxpr | 6.2 | 1.06× |
| **trace 一次**（2 个节点的图） | **18.6** | 3.2× |

> ⚠ **这几个数逐次运行会波动**：交付前后若干次跑到的分别是
> 5.5 / 5.8 / 16.8、5.8 / 6.2 / 18.6、5.5 / 5.8 / 18.3，
> 所以**可用的是量级与比值，不是具体值**——
> 这也正是为什么程序里的断言判的是**区间与大小关系**而不是等号。

**★ 这一栏里有本程序第一版被实跑推翻的断言**：第一版写的是「trace 一次比 replay
贵**一个量级以上**」，实跑只有约 **3×**。更要紧的是第二、三行：
**在这个纯 Python 迷你实现里，replay 比 eager 还略贵**（6.2 vs 5.8）。
图是被**解释执行**的数据结构，每回合仍要付 dict 查表与逐节点分派——
**「折叠成图 ⇒ 每次调用更省」在这里不成立**。
折叠真正省下的是**可序列化、可交给编译器**，不是每次调用的时间。
（这与 [tensorflow/01-graph-and-eager](../tensorflow/01-graph-and-eager.md) §5 测到的
同一个现象一致；**两边的迷你实现各自独立跑出了它**。）

**（4）形状多态的导出**（迷你实现实测）：同一形状调用 → 重编译 **0** 次；
换形状调用 → 重编译 **1** 次。

**（5）本页引用的**规格值 / 引用值**（不来自本仓库测量）**：
「no arguments are treated as static」出自 `jax/_src/api.py:253-254`（tag `jax-v0.11.1`）；
cache key 三项与「Only the tracing and the lowering are saved」出自
`docs.jax.dev/en/latest`（**dev 文档**，与 0.11.1 tag 的措辞有漂移，见 §参考）。

## 6. 小结与延伸

- `jax.jit` 的 cache key 是**三项**：**参数的 JAX types**（形状 + dtype）、
  **静态参数的值**、**函数对象的 identity**。清单里有什么，图就假设什么。
- **★ Python 标量默认不是静态的**（官方 docstring 逐字 **no arguments are treated
  as static**）：`f(x, 3)` 与 `f(x, 4)` **只追踪一次**，而那个标量变成 jaxpr 的**一个 invar**。
  要它变常量必须显式 `static_argnums` / `static_argnames`。
  **这与 TF 正好相反**（TF 按值特化，`LiteralTraceType<3>`），两篇文档互为镜像。
- **`jax.jit` 没有 dynamic shape 开关**（在官方那份缓存模型的文档范围内）；
  形状多态在 `jax.export.symbolic_shape` 那一层，**而且它也不免除重编译**——
  官方逐字：**Only the tracing and the lowering are saved**。
- **诊断**：`jax_explain_cache_misses` 直接告诉你 miss 在哪一项上；
  `jax_log_compiles`、`JAX_DUMP_IR_TO`、`jax.clear_caches()` 配套。
- **第 3 项的实际后果**：别在循环里 `jax.jit(lambda …)`——每次都是新对象，每次都重编译。
- 往下一站：[02-transformations](02-transformations.md)（`vmap` 为什么**不展开循环**、jaxpr 的形态、
  `donate_argnums` 与就地修改）；[03-sharding-and-differences](03-sharding-and-differences.md)
  （多设备那条路的演进，以及三者对照的收束）。
  镜像对照：[tensorflow/01-graph-and-eager](../tensorflow/01-graph-and-eager.md)。

## 参考

**本主题的配套程序**：`code/01_jit_cache_key.py`（三类断言齐全，失败非零退出）。
**本项目核过的一手来源**（全部登记在 `code/specs.py`，取用于 2026-09-17）：

- **cache key 三项 / 「new shapes or dtypes trigger a fresh trace and compile」/
  `TRACING CACHE MISS` 的例子 / function identity 的实际后果 / donation 措辞**：
  `docs.jax.dev/en/latest/201/jit.html`（**dev 文档**，按 §0.7.C 的规矩标注）
  <https://docs.jax.dev/en/latest/201/jit.html>
  **tag 内同一篇**：`docs/new_docs/201/jit.md`（jax-v0.11.1，第三项与 donation 措辞不同）
  <https://raw.githubusercontent.com/jax-ml/jax/jax-v0.11.1/docs/new_docs/201/jit.md>
- **「no arguments are treated as static」与静态参数的重编译**：
  `jax/_src/api.py:249-254`（tag `jax-v0.11.1`）
  <https://github.com/jax-ml/jax/blob/jax-v0.11.1/jax/_src/api.py#L249-L254>
- **`jax.jit` 的返回类型 `pjit.JitWrapped`**：`jax/_src/api.py:184`/`:201`/`:217`
  <https://github.com/jax-ml/jax/blob/jax-v0.11.1/jax/_src/api.py#L184-L217>
- **jaxpr 的定义与文法**：`docs.jax.dev/en/latest/601/jaxpr.html`（**dev 文档**）
  <https://docs.jax.dev/en/latest/601/jaxpr.html>
- **`jax.export` 的形状多态仍会重编译**：`docs.jax.dev/en/latest/export/shape_poly.html`
  （**dev 文档**；tag 内同文见 `docs/export/shape_poly.md`）
  <https://docs.jax.dev/en/latest/export/shape_poly.html>
- **`jax.export.symbolic_shape` 的签名**：`jax/_src/export/shape_poly.py:1383`
  <https://github.com/jax-ml/jax/blob/jax-v0.11.1/jax/_src/export/shape_poly.py#L1383>

**记录在案、正文不用的项**（`code/specs.py` 文件头的 UNSOURCED 段）：
「`weak_type` 是否参与 jit cache key」**没有取到官方逐字**，
故本仓库只把它当作模型内部的 `aval` 相等性，**不作断言**。

**回链**：XLA 的 HLO / 融合 / layout 正文在
[60-compilers/xla](../../60-compilers/xla/01-hlo-and-shape-inference.md)；
「一次调用经过多少层」的 eager 侧见
[70-frameworks/pytorch/01-dispatch-and-aten](../pytorch/01-dispatch-and-aten.md)。

---
title: 分片与心智差异：多设备那条路的演进，以及三者对照的收束
slug: jax-sharding-and-differences
domain: 70-frameworks
topic: jax
level: 2
difficulty: advanced
prerequisites:
  - 70-frameworks/jax/01-jit-and-tracing
  - 70-frameworks/jax/02-transformations
tags: [jax, lax-cond, while-loop, scan, pmap, pjit, shard-map, spmd, guards]
related:
  - 70-frameworks/pytorch/01-dispatch-and-aten
  - 70-frameworks/tensorflow/02-control-flow-and-xla
  - 60-compilers/xla/01-hlo-and-shape-inference
code:
  - code/03_sharding_evolution.py
status: stable
updated: 2026-09-17
---

# 分片与心智差异：多设备那条路的演进，以及三者对照的收束

> 多设备那条路走过四个阶段：**`pmap` → `pjit` → 并入 `jax.jit` → `shard_map`**。
> 每一步换的都不是「能不能多卡」，而是**「谁来把假设写下来」**：
> `pmap` / `shard_map` 要你**显式**写（分片方式、集合通信），
> `jax.jit` 的自动分片则让**编译器替你推**。
> 而控制流那边只有一句话要记：**值相关的 Python 控制流在 JAX 里直接失败**，
> 官方给的替代是结构化四件套——其中 **`lax.cond` 的谓词可以是 traced 值**，
> **`while_loop` 却不是反向模式可微的**。

## 0. 背景与动机

### 一个具体的对照场景

```
同一个「循环里做归约」的意图，在三个框架里要写三套不同的东西：
  · JAX：lax.while_loop / lax.scan（显式的结构化算子）
  · TensorFlow：AutoGraph 把 `while x > 0:` 转成 tf.while_loop（一样是双轨）
  · PyTorch：eager 照写；进 torch.compile 后由 Dynamo 决定能不能捕获
```

### 本文在本域主线上的位置：**③本文不涉及（但它是收束）**

本域的主线是「**框架要把哪些假设固化进图**」（首篇
[01-jit-and-tracing](01-jit-and-tracing.md) 完整展开了这条主线）。

**§1（控制流）与 §2（多设备）不在「固化什么」这条主线上**——它们讲的是
**「假设以什么形式被写下来」**：控制流是「走哪条分支」这个假设，
多设备是「数据怎么切、通信谁来插」这个假设。§3 则是全主题的**收束**：
把三个框架放在同一张表里，看**同一组问题各自的答案**。

**不硬套主线**这件事本身要说清楚：控制流与分片都是**独立的知识点**，
把它们硬说成「固化假设」的某一面只会稀释主线。它们与主线的关系只有一条间接的：
**两者都会改变「重追踪 / 重编译」的次数**（换分支不重追、换分片方式要重编）。

### 本文要回答的三个问题

1. **值相关的控制流怎么办？**（§1）
2. **多设备那条路为什么换了三次写法？**（§2）
3. **三家到底差在哪？**（§3——这是全主题的收束）

## 1. 原理：结构化控制流四件套

### 1.1 为什么需要它们：Python 控制流在追踪期会断

在 [01](01-jit-and-tracing.md) 的框架里，追踪期拿到的是**符号值**：
它有形状、有 dtype，**没有数据**。所以「拿它当布尔用」这件事**做不了**。
`code/03_sharding_evolution.py` 的 §C1 把它跑了出来：对 traced 值写 `if p > 0:`
直接抛错（**迷你实现实测，不是 JAX**）。

> ⚠ **诚实标注**：真实 JAX 会抛 `TracerBoolConversionError`
> （本项目的计划 §0.7.D 转述为「Attempted boolean conversion of traced array…」）。
> 本仓库**没有取到这条报错的逐字一手来源**，所以编号程序里那条报错的**措辞是自写的**，
> 只作机制演示，**不作原文引用**（见 `code/specs.py` 的 UNSOURCED 段）。
> 能坐实的是**机制**：对 traced 值取 bool 会失败。

于是有了**结构化四件套**（都在 `jax.lax` 下）：

| 算子 | 干什么 | 关键性质 |
|---|---|---|
| `lax.cond(pred, true_fun, false_fun, *ops)` | 分支 | **谓词可以是 traced 值**；两支都进图 |
| `lax.while_loop(cond_fun, body_fun, init)` | 不定长循环 | body **只进图一次**；⚠ **不是反向模式可微的** |
| `lax.fori_loop(lo, hi, body_fun, init)` | 定长循环 | 比 `while_loop` 更好被反传（见下） |
| `lax.scan(body_fun, init, xs)` | 带携带状态的定长扫描 | body 只进图一次；**有反向规则** |

### 1.2 `lax.cond` 的关键性质

官方逐字（**dev 文档** `https://docs.jax.dev/en/latest/201/control-flow.html`）：

> Unlike a Python if, **the predicate here can be a traced value**. **The choice of
> branch happens on the device, at run time, inside the compiled program.**

> ⚠ **版本提醒**：`jax-v0.11.1` 的 tag 里同一句用的是**破折号**而不是句号
> （`… can be a traced value — the choice of branch happens …`），
> 且这句话**不在** `lax.cond` 的 docstring 里。**两版不是同一句**，引用须注明版本。
> 另外：**这条性质与 TF 的 `tf.cond` 是同一个设计**（
> [tensorflow/02](../tensorflow/02-control-flow-and-xla.md) §1.2）：
> 把「选哪条」从**追踪期**推迟到**运行时**，代价是**两条分支都进图**。

`code/03_sharding_evolution.py` 的 §1 把这件事跑了出来（**迷你实现实测，不是 JAX**）：
同一张图喂 `+3.0` 与 `-3.0` 走不同分支、**追踪计数始终是 1**；
图的形态是 `cond v2 v1 [子 jaxpr cond_true] [子 jaxpr cond_false]`——
**两支都在**。

### 1.3 ⚠ `while_loop` 只支持前向模式微分

官方 `jax.lax.while_loop` 的 docstring 逐字（`jax/_src/lax/control_flow/loops.py:1641-1643`）：

> Another difference from using Python-native loop constructs is that **`while_loop` is
> not reverse-mode differentiable** because XLA computations require static bounds on
> memory requirements.

**成因值得展开**（这是「为什么必须有 `scan`」的答案）：反向模式要**从后往前走一遍**，
而走回去需要**知道循环跑了几次、每一步的中间值在哪**——
这要求**迭代次数是静态有界的**。`while_loop` 的次数恰恰是**动态、无界**的，所以反传做不了。

> ⚠ **措辞提醒**：「**only forward-mode differentiable**」这个**正面**说法
> **不在 0.11.1 的 docstring 里**，它在 **dev 文档**
> （`201/control-flow.html`）里。tag 的 docstring 用的是**反向排除**
> （`not reverse-mode differentiable`）——**两者不是同一句**。
> 本仓库把两版都登记在 `code/specs.py` 里。

所以官方给的替代路径是：**用 `scan`（定长）或 `fori_loop`（静态边界）**——
它们把「跑多少次」变成静态信息，于是反传有了立足点。
`code/03_sharding_evolution.py` 的 §C2 把这一组对照跑了出来（**迷你实现实测，不是 JAX**）：
同一个「累乘」用 `while_loop` 写 ⇒ `grad` 抛 `NotImplementedError`；
用 `scan` 写 ⇒ 梯度算得出来，且与解析导数一致。

> ⚠ **本模型的一条限制**：编号程序里 `fori_loop` 落成 `scan` 是**本仓库迷你实现自己定的**，
> 真实 JAX 对 `fori_loop` 的降级规则**没有取到一手来源**，故**不作断言**
> （见 `code/specs.py` 的 UNSOURCED 段）。

### 1.4 与另外两家的对照

| | JAX | TensorFlow | PyTorch |
|---|---|---|---|
| 值相关的 `if` | **直接失败**，用 `lax.cond` | AutoGraph 转 `tf.cond`（**条件是 Tensor 才转**） | **graph break**（图截断后再续） |
| 值相关的 `while` | **直接失败**，用 `lax.while_loop` | AutoGraph 转 `tf.while_loop` | `torch.while_loop`（**prototype**，见 §3.2） |
| 循环体进图几次 | **1 次**（body 是符号执行一遍） | 1 次 | 看捕获情况 |

**本机核实过的一条 PyTorch 事实**（真框架实测，须写版本）：
`torch.while_loop` **存在**——`hasattr(torch, 'while_loop')` 在
**本机 torch 2.7.1 与 2.8.0 上均为 `True`**，docstring 首句为
「Run body_fn(*carried_inputs) while cond_fn(*carried_inputs) returns a True scalar
tensor.」，且 Warning 段自述「`torch.while_loop` is a prototype feature in PyTorch」。
它在 `torch.__all__` **里没有**；而 `torch.cond` 同样 prototype **但在** `__all__` 里。
**两版一致。** 所以「PyTorch 没有 while 版结构化控制流」这句话是**错的**，
准确说法是「有，但是 prototype，且不在 `__all__`」。

## 2. 执行映射：多设备那条路的演进

### 2.1 四个阶段

| 阶段 | 写法 | 假设由谁写 | 现在什么状态 |
|---|---|---|---|
| **`pmap`** | `pmap(f)`，函数体里调 `lax.psum` 等 | **你**（但复制整个函数到每台设备） | **已被官方劝退** |
| **`pjit`** | `pjit(f, in_shardings=…)` | **编译器**（GSPMD 式自动分片） | **已并入 `jax.jit`** |
| **`jax.jit` + shardings** | `jit(f, in_shardings=…)` | **编译器** | 现在的推荐写法 |
| **`shard_map`** | `shard_map(f, mesh, in_specs, out_specs)` | **你**（显式 collective） | 与 `jit` **互补且可组合** |

**注意这不是「新的淘汰旧的」那么简单**：`pmap` 与 `shard_map` 都让**用户**写假设，
而 `jit` 的自动分片让**编译器**推。演进的结果是**两条路并存**——
因为你有时候确实需要显式控制（§2.4 那张账）。

### 2.2 `pmap` 已被官方劝退，且它现在用 `jit` + `shard_map` 实现

`jax/_src/pmap.py:70` 的 docstring 逐字：

```
"""Old way of doing parallel map. Use :py:func:`jax.shard_map` instead.
```

渲染后（文档页）逐字：**Old way of doing parallel map. Use `jax.shard_map()` instead.**
同文件 `:79-81` 紧接着说：

> **`pmap()` is now implemented in terms of `jit()` and `shard_map()`.** Please see the
> migration guide for more information.

**这两句要一起读**：`pmap` 不是「被删掉」，而是**被劝退 + 改成用别的实现**。
「劝退」的准确含义因此是：**它还能跑，但它已经不是实现的主体**——
一层薄壳罢了。

> ⚠ **官方没给移除日期**：docstring 里没有任何时间信息，故**不写**「从某个版本起废弃」。
> 这一点与下面 `pjit` 的处置**不同**。

**`pmap` 与 `vmap` 的官方对照**（同文件 `:88-91`，**完整句**）：

> Semantically it is comparable to `vmap()` because both transformations map a function
> over array axes, **but where `vmap()` vectorizes functions by pushing the mapped axis
> down into primitive operations, `pmap()` instead replicates the function and executes
> each replica on its own XLA device in parallel.**

> ⚠ **本项目订正（P7）**：计划 §0.7.C 把这句话引成
> 「`vmap()` vectorizes … , `pmap()` instead replicates …」——
> **拆掉了原文的 `but where … , … instead` 关联结构**，那不是逐字的原句。
> 本文用完整句。

### 2.3 `pjit` 已并入 `jax.jit`——而且有精确日期

`jax-v0.11.1` 里 `jax/experimental/pjit.py` 已经只是一个 37 行的壳，逐字：

```python
_deprecations = {
    # Added Oct 13, 2025
    "pjit": (
        (
            "jax.experimental.pjit.pjit has been deprecated. Please use"
            " `jax.jit` instead."
        ),
        _deprecated_pjit,
    )
}
```

**源码注释里给了精确日期「Added Oct 13, 2025」**——这在废弃标记里不常见，
值得单独记住。消息逐字：
`jax.experimental.pjit.pjit has been deprecated. Please use \`jax.jit\` instead.`

**落点在哪**：`jax.jit` **自己的返回类型**就叫 `pjit.JitWrapped`——
`jax/_src/api.py` 的三处重载（`:184` / `:201` / `:217`）都写着 `-> pjit.JitWrapped`。
所以这不是「两个 API 合并成一个」的比喻：**返回类型真的就是那个名字**。

> 旁证：CHANGELOG 的 **JAX 0.8.0（October 15, 2025）** 段里有
> 「`jax.experimental.pjit.pjit` is deprecated; going forward use `jax.jit`」，
> 且 `_autosummary/jax.experimental.pjit.pjit` 那个文档页现在返回 **404**。

### 2.4 `shard_map`：显式的那条路

官方定位逐字：

> `shard_map` is a single-program multiple-data (SPMD) multi-device parallelism API …
> **Mapped function applications, or instances, communicate with each other via explicit
> collective communication operations.**

以及它与 `jit` 自动分片的关系：

> `shard_map` is **complementary to, and composable with**, the automatic
> compiler-based parallelization built into `jit`.

**`complementary` 与 `composable` 是两个不同的承诺**：
前者说「两条路各有适用面」，后者说「可以混着用」（自动分片的外层 + 显式分片的内层）。

### 2.5 两条路的差：一张可数的账

`code/03_sharding_evolution.py` 的 §2 把同一个「全局求和」用两条路各写一遍
（**迷你实现实测，不是 JAX**）：

| 写法 | 用户写的集合通信 | 包装器/分区器插的 | 结果 |
|---|---|---|---|
| `jit(f, in_shardings=P('x'))` | **0** | **1 个 `psum`**（自动） | 28.0 |
| `shard_map(f, …)`，函数体里写 `lax.psum` | **1** | **0** | 28.0 |

**两边的结果一样，差别只在「假设由谁写」**——这就是 §2.1 那张表的可数形态。
另外 `shard_map` 的函数体**只 trace 一次**（不是每设备一份）：
device 数从 2 换到 4，`psum` 的结果仍然是全局和。

## 3. 收束：三家到底差在哪

### 3.1 三者对照表

> **先纠正一个共同的误解**：**三者默认都对 shape 敏感。没有哪一个默认「对 shape 不敏感」。**

| | TF `tf.function` | JAX `jax.jit` | PyTorch（Dynamo） |
|---|---|---|---|
| **缓存/失效机制** | 各实参 `TraceType` 拼成 `FunctionType` 查表 | **cache key** = 参数的 JAX types + 静态参数值 + **函数对象 identity** | **guards**：trace 时装布尔断言，调用时逐个校验 |
| **shape 变化** | retrace | 重编译 | 重编译 |
| **dtype 变化** | retrace | 重编译 | guard 失败 → 重编译 |
| **★ Python 标量** | **按值特化**（`LiteralTraceType<3>`） | **默认不特化**（weak-typed traced array） | **看用法**（depends on its usage） |
| **缓解手段** | `input_signature`、`None` 维、`reduce_retracing=True` | 无 jit 级开关；padding 到固定 bucket；`jax.export` 的形状多态**仍会重编译** | `mark_dynamic` / `mark_static`；`torch.compile(dynamic=True)` 官方标 "(Not recommended)" |
| **诊断** | `experimental_get_tracing_count()`、自动重追踪告警 | `jax_explain_cache_misses`、`jax_log_compiles`、`JAX_DUMP_IR_TO` | `TORCH_LOGS=guards\|recompiles`、`tlparse` |
| **值相关的 `if`** | AutoGraph 转 `tf.cond`（**条件是 Tensor 才转**） | **直接失败**，用 `lax.cond` | **graph break** |
| **值相关的 `while`** | AutoGraph 转 `tf.while_loop` | **直接失败**，用 `lax.while_loop` / `scan` | `torch.while_loop`（**prototype**，不在 `__all__`） |

**读这张表的规矩（本域 §0.3 的强制要求）**：三列的证据强度**不一样**——

- **JAX 这一列**：本仓库有 `code/specs.py` 的一手 URL（**[01](01-jit-and-tracing.md) / [02](02-transformations.md) / 本文** 都在引）；
- **TF 那一列**：正文在 [tensorflow](../tensorflow/01-graph-and-eager.md)，那边有自己的 `specs.py`（**本仓库不复核其口径**）；
- **PyTorch 那一列**：本机有**真 torch**，属「真框架实测」档（须写版本）；
  §1.4 那条 `torch.while_loop` 的存在性就是本机两版实测的。

### 3.2 最容易记错的两条

**第一条：Python 标量。**
**「Python 标量是静态的」这句话，在 TF 里对、在 JAX 里错。**

| | TF | JAX |
|---|---|---|
| `f(x, 3)` 与 `f(x, 4)` | **两次追踪**（`LiteralTraceType<3>` / `<4>`） | **一次追踪**（`3` 与 `4` 的 aval 相同） |
| 标量在图上的身份 | **op 的字面量属性**（烧进节点） | **jaxpr 的一个 invar**（要 static 才变常量） |

逐字依据：JAX 侧是 `jax.jit` docstring 的 **no arguments are treated as static**
（[01](01-jit-and-tracing.md) §1.3）；TF 侧是 `TraceType` 规则第二条的
`LiteralTraceType<3>`。**两边的编号程序各自把这条跑出来了。**

**第二条：PyTorch 的结构化控制流。**
`torch.while_loop` **存在**（本机 2.7.1 / 2.8.0 实测 `hasattr` 均为 `True`），
只是**prototype 且不在 `torch.__all__`**；`torch.cond` 同样 prototype **但在 `__all__`**。
**不得写「PyTorch 没有 while 版结构化控制流」**——
用文档 404 当禁令依据是本域最贵的一类错误（会让人删掉正确内容）。

### 3.3 三个框架给的三个答案

回到主线上那个问题——**「把哪些假设固化进图」**——三家给了三种态度：

| | 对「假设」的态度 | 具体表现 |
|---|---|---|
| **TensorFlow** | **把假设写成一份可枚举的类型清单** | `TraceType` 明确列出「什么进键」；Python 标量**按值**进 |
| **JAX** | **把假设写成一个 cache key，并把「怎么少写」交给导出层** | 三项 cache key；形状多态在 `jax.export`，且**仍逐形状重编译** |
| **PyTorch** | **不预先把假设写下来，而是 trace 时随手记 guard、用错了再退回** | guards + graph break——**能追多少是多少** |

三种态度**都成立**，代价也各自不同：TF 的清单最硬（清单没写对就反复重追踪）；
JAX 的 key 最少（连标量都不进，所以少了一层重追踪、多了一个运行期输入）；
PyTorch 最宽松（容忍度高，但一个 graph break 就把图打断了）。

## 4. 踩坑与失败模式

### 失败模式 1：用 Python `if` / `while` 处理 traced 值

**现象**：`if x.sum() > 0:` 或 `while x > 0:` 在追踪期直接抛错
（真实 JAX 抛 `TracerBoolConversionError`，措辞见 §1.1 的标注）。
**成因**：追踪期的值是**符号**，没有真值；「走哪条」在追踪期根本无法确定。
**规避**：改成 `lax.cond` / `lax.while_loop` / `lax.scan`。
判断标准是一句话：**这个条件在运行时会变吗？** 会变就得用结构化算子；
不会变（真是超参）用 Python 控制流没问题——但**别事后去改它**
（在 TF 里那是静默算错，见 [tensorflow/01](../tensorflow/01-graph-and-eager.md) §4 第三条）。

### 失败模式 2：用 `while_loop` 写要反传的循环

**现象**：`grad` 穿过 `lax.while_loop` 时抛
`NotImplementedError`（官方措辞是 docstring 那句
「`while_loop` is **not reverse-mode differentiable**」）。
**成因**：反向模式要倒着走一遍，需要**迭代次数的静态上界**；
而 `while_loop` 的次数是动态无界的（§1.3）。
**规避**：换成 **`scan`**（定长）或 **`fori_loop`**（静态边界）——
官方指的就是这条路。`code/03_sharding_evolution.py` §C2 把「while 失败 / scan 成功」
这一组对照跑了出来。

### 失败模式 3：把 `pmap` 当成「多设备的标准写法」

**现象**：新代码里继续 `pmap`，然后发现官方文档处处让你换 `shard_map`。
**成因**：`pmap` 的 docstring 第一句就是 **Old way of doing parallel map**
（§2.2），而且它**现在就是用 `jit()` + `shard_map()` 实现的**。
**规避**：新代码用 `jit` 的自动分片（大多数情况够用）；
需要显式控制通信时用 `shard_map`（两者可以组合）。
**注意**：`pmap` **没有移除日期**，所以不必惊慌地大改——
但也不该在新代码里选它。

### 失败模式 4：以为「自动分片」和「显式分片」是二选一

**现象**：为了用 `shard_map` 而把整个模型的手动分片全写一遍，或者反之。
**成因**：官方说的是 **complementary to, and composable with**（§2.4）——
**两句话都要读**：`jit` 的自动分片是默认路径，而 `shard_map` 是让你在
**某些子模块里精确控制通信**的出口，两者可以嵌套。
**规避**：先让 `jit` 自动分；哪个子模块的通信不理想，再把它单独包进 `shard_map`
（这也是 §2.5 那张「谁插集合通信」的账的实际用法）。

## 5. 实测数据

**测量条件**：`code/03_sharding_evolution.py` 经 `make run` 在本机跑出
（python3 3.13.5 / numpy 2.4.4），全部 **22** 条断言通过（A 7 / B 6 / C 9）。

> **★ 限定语（本域强制）**：下面**（1）–（4）**每一个数字都是**本仓库的迷你追踪器**
> 跑出来的，**不是 JAX**。本机**没有装 JAX**，也**没有多设备**，
> 所以这里没有任何「真实 JAX 实测」，也没有任何关于多卡通信的实测。
> 这些数字支持的是「这套机制可以怎样被计数」这类**机制**结论。
> **（5）是唯一一处真框架实测**，它用的是**本机的真 torch**，且逐处标了版本。

**（1）结构化控制流的追踪次数**（迷你实现实测）：

| 场景 | 追踪次数 |
|---|---|
| `lax.cond`，谓词取 +3.0 / -3.0（走不同分支） | **1** |
| `lax.while_loop`，`init=5` 与 `init=500`（迭代次数差 100 倍） | **1**（且**同一张图**） |

`while_loop` 那张图的**语句数**（本层方程 + 子 jaxpr 递归）= **3**，
**与迭代次数无关**。

**（2）反向模式能不能过**（迷你实现实测）：

| 循环写法 | 前向 `jvp` | 反向 `grad` |
|---|---|---|
| `lax.while_loop` | ✅ 通（切向 1.0 原样穿过） | ❌ `NotImplementedError` |
| `lax.scan` | —— | ✅ 通（`d(carry)/d(init) = 2·3·4 = 24`） |

**（3）两条分片路的账**（迷你实现实测，2 个设备，8 个元素）：

| 写法 | 用户写的集合通信 | 自动插的 | 结果 |
|---|---|---|---|
| `jit(..., in_shardings=P('x'))` | 0 | **1** | 28.0 |
| `shard_map(...)`（函数体里写 `lax.psum`） | **1** | 0 | 28.0 |

设备数换成 4，`psum` 的结果仍是 **28.0**（全局和）；`shard_map` 的函数体
**只 trace 一次**。

**（4）分片的切法**（迷你实现实测）：沿轴切 2 份 ⇒ 每份形状 `(4,)`，
拼回来等于原数组。

**（5）★ 唯一一处真框架实测（PyTorch，须写版本）**：

| 项 | torch 2.7.1 | torch 2.8.0 |
|---|---|---|
| `hasattr(torch, "while_loop")` | **True** | **True** |
| `"while_loop" in torch.__all__` | **False** | **False** |
| `hasattr(torch, "cond")` / 在 `__all__` | True / **True** | True / **True** |

docstring 首句（两版一致）：「Run `body_fn(*carried_inputs)` while
`cond_fn(*carried_inputs)` returns a True scalar tensor.」，Warning 段自述 prototype。
**这一栏是本机真 torch 跑出来的，属「真框架实测」档**——
与上面（1）–（4）的「迷你实现实测」**不是同一档证据，不得互相背书**。

## 6. 小结与延伸

- **值相关的 Python 控制流在 JAX 里直接失败**；替代是结构化四件套
  （`lax.cond` / `while_loop` / `fori_loop` / `scan`）。
- **`lax.cond` 的谓词可以是 traced 值**，分支选择**在设备上、运行时**发生
  （官方逐字；**与 TF 的 `tf.cond` 同一设计**，代价是两支都进图）。
- **⚠ `while_loop` 不是反向模式可微的**（官方 docstring 逐字），
  因为反向要**静态有界的迭代次数**；要反传就用 `scan` / `fori_loop`。
- **多设备那条路的演进**：`pmap`（**Old way…，且现在用 `jit()` + `shard_map()` 实现**）
  → `pjit`（**已并入 `jax.jit`**，日期 **Oct 13, 2025**，落点是 `jax.jit` 的返回类型
  `pjit.JitWrapped`）→ `jax.jit` 的自动分片 + `shard_map` 的显式分片
  （官方说两者 **complementary to, and composable with**）。
- **可数的那笔账**：同一个全局求和，自动分片让**分区器插 1 个 `psum`**（用户 0 个），
  `shard_map` 让**用户写 1 个**（包装器 0 个）——**差别只在假设由谁写**。
- **收束**：三家对「固化假设」的态度不同——TF 写成**可枚举的类型清单**、
  JAX 写成**三项 cache key**（且把「怎么少写」交给导出层）、
  PyTorch 用 **guard + graph break** 边走边看。
  **★ 「Python 标量是静态的」在 TF 里对、在 JAX 里错**——这是本主题最值钱的一条对照。
- 回到本主题的另外两篇：[01-jit-and-tracing](01-jit-and-tracing.md)（cache key 三项）、
  [02-transformations](02-transformations.md)（vmap / jaxpr / 不可变性）。
  跨主题：[tensorflow/01](../tensorflow/01-graph-and-eager.md)（镜像对照）、
  [pytorch/01](../pytorch/01-dispatch-and-aten.md)（eager 那条路的层数）。

## 参考

**本主题的配套程序**：`code/03_sharding_evolution.py`（三类断言齐全，失败非零退出）。
**本项目核过的一手来源**（全部登记在 `code/specs.py`，取用于 2026-09-17）：

- **`lax.cond` 的「predicate can be a traced value」**：
  `docs.jax.dev/en/latest/201/control-flow.html`（**dev 文档**；tag 内同句用破折号）
  <https://docs.jax.dev/en/latest/201/control-flow.html>
- **「`while_loop` is not reverse-mode differentiable」**：
  `jax/_src/lax/control_flow/loops.py:1641-1643`（jax-v0.11.1）
  <https://github.com/jax-ml/jax/blob/jax-v0.11.1/jax/_src/lax/control_flow/loops.py#L1641-L1643>
- **`pmap` 的「Old way…」与「implemented in terms of `jit()` and `shard_map()`」**：
  `jax/_src/pmap.py:70` 与 `:79-81`
  <https://github.com/jax-ml/jax/blob/jax-v0.11.1/jax/_src/pmap.py#L70-L81>
- **`vmap`/`pmap` 的完整对照句**：`jax/_src/pmap.py:88-91`
  <https://github.com/jax-ml/jax/blob/jax-v0.11.1/jax/_src/pmap.py#L88-L91>
- **`pjit` 的废弃块（含 `Added Oct 13, 2025`）**：`jax/experimental/pjit.py:20-26`
  <https://github.com/jax-ml/jax/blob/jax-v0.11.1/jax/experimental/pjit.py#L20-L26>
- **`jax.jit` 的返回类型 `pjit.JitWrapped`**：`jax/_src/api.py:184`/`:201`/`:217`
  <https://github.com/jax-ml/jax/blob/jax-v0.11.1/jax/_src/api.py#L184-L217>
- **`shard_map` 的定位与「complementary / composable」**：
  `docs.jax.dev/en/latest/_autosummary/jax.shard_map.html`（**dev 文档**）
  <https://docs.jax.dev/en/latest/_autosummary/jax.shard_map.html>
- **缓存/形状/标量那三列的 JAX 依据**：见 [01](01-jit-and-tracing.md) §参考
  （`201/jit.html` 与 `jax/_src/api.py`）；TF 与 PyTorch 两列各回链其正文。

**本机真框架实测的来源（PyTorch）**：`/usr/bin/python3` = **torch 2.8.0**、
`~/miniconda3/envs/dp-test/bin/python` = **torch 2.7.1**，
`hasattr(torch, "while_loop")` / `"while_loop" in torch.__all__` 于 2026-09-17 逐一跑出。
**换一个版本，这组结论就要重测。**

**记录在案、正文不用的项**（`code/specs.py` 文件头的 UNSOURCED 段）：
「`lax.fori_loop` 何时降级成 `scan`」**没有取到一手来源**，故不作断言；
`TracerBoolConversionError` 的**报错原文**同样没有取到，故正文只作机制演示。

**回链**：XLA 的 HLO / 融合 / layout / 自动分片内部正文在
[60-compilers/xla](../../60-compilers/xla/01-hlo-and-shape-inference.md)；
控制流「固化一支还是两支都进图」的 TF 侧见
[tensorflow/02-control-flow-and-xla](../tensorflow/02-control-flow-and-xla.md)。

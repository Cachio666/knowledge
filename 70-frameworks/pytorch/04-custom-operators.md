---
title: 自定义算子：把算子接进这条链，以及编译入口
slug: pytorch-custom-operators
domain: 70-frameworks
topic: pytorch
level: 3
difficulty: intermediate
prerequisites:
  - 70-frameworks/pytorch/01-dispatch-and-aten
  - 70-frameworks/pytorch/03-tensor-memory-model
tags: [pytorch, custom-op, torch-library, register-fake, torch-compile, dynamo, guard, recompile]
related:
  - 60-compilers/inductor/01-graph-capture
  - 60-compilers/inductor/04-dynamic-shapes
  - 30-gpu-programming/cuda/01-execution-model
code:
  - code/04_custom_ops.py
status: stable
updated: 2026-09-17
---

# 自定义算子：把算子接进这条链，以及编译入口

> 自定义算子不是「绕过」PyTorch 的调度链，而是**接进这条链**：它要交出 schema（由类型标注推导）、要注册 kernel、要给 FakeTensor 一个形状推断。交出这三样之后，它在 eager 上的层数与内置算子**完全一样**（1 次 `__torch_dispatch__` 拦截），而在 `torch.compile` 下它是否被固化进图，取决于它有没有把假设交代清楚。

## 0. 背景与动机

### 一个具体的对照场景

你写了一个 SwiGLU 的融合算子 `swiglu(x)`，想让它在 `torch.compile` 生成的图里被当作**单个节点**，而不是被拆成一串 `chunk/sigmoid/mul`：

```python
@torch.library.custom_op("kb04::swiglu", mutates_args=())
def swiglu(x: torch.Tensor) -> torch.Tensor:
    a, b = x.chunk(2, dim=-1)
    return a * torch.sigmoid(a) * b
```

这段代码一写就有三个问题要回答：**`mutates_args` 为什么必须写**（它没有默认值）、**类型标注能不能省**（不能，schema 由它推导）、**返回的东西是函数吗**（不是，是 `CustomOpDef`）。回答错了**不会静默出错**——三个约束都会当场抛异常（本文 §4 把三条都真的跑出来）。

### 本文在本域主线上的位置：**①判据来源**

> **自定义算子必须遵守这条链的契约**——它就是主线里「固化什么」的**判据来源**：你要进图，就得把「输入什么类型、能不能原地改、形状怎么变」交代清楚；交代不清的部分，要么被拆开，要么在编译时让你自己补。

本文分两半：**§1–§2 讲注册**（把算子接进链），**§3 讲编译入口**（§0.5 的边界：`torch.compile` 的**编译器内部**在 [60-compilers/inductor](../../60-compilers/inductor/01-graph-capture.md)，本文只写**用户接口**视角——怎么用、怎么诊断、怎么数重编译）。

### 最容易记错的几条

1. 「**`custom_op` 返回的是函数**」——不是，是 `CustomOpDef` 对象（可直接调用，但它有 `register_kernel` / `register_fake` 这些方法）。
2. 「**改名的旧名发 DeprecationWarning**」——实测发的是 **`FutureWarning`**。
3. 「**`_dispatch_has_kernel` 必须写重载名**」——**只对 aten 成立**：对自定义算子裸名可用，加 `.default` 反而抛错。
4. 「**`get_kernel` / `EffectType` 是 2.13 新增的**」——实测 **2.11.0 上就有了**。

## 1. 原理：三个入口，各自交出什么

`torch.library` 提供三种注册方式，**能力与代价递增**：

| 入口 | 你交出去的 | 谁推导 schema | 典型用处 |
|---|---|---|---|
| `custom_op` | 一个带类型标注的 Python 函数 | **它自己推导** | 纯 Python 实现的自定义算子 |
| `define` + `impl` | 手写 schema 字符串 + kernel | 你（schema 你自己写） | 需要精确控制 schema、或 kernel 用别的方式注册 |
| `register_fake` | 形状/元数据推断函数 | — | 让 `torch.compile` 能在**不执行**的情况下推断输出形状 |

实测的签名（本机五版本一致，来源：`inspect.signature`）：

```
custom_op(name, fn=None, /, *, mutates_args, device_types=None, schema=None)
define(qualname, schema, *, lib=None, tags=())
impl(qualname, types, func=None, *, lib=None)
register_fake(fn, /)
```

### 1.1 三个易错点（都被实测触发过）

| 易错点 | 现象 |
|---|---|
| **`mutates_args` 是 keyword-only 且无默认值** | 不传 → `TypeError: custom_op() missing 1 required keyword-only argument: 'mutates_args'` |
| **被包装函数必须有类型标注** | 不标 → `ValueError: infer_schema(func): Parameter x must have a type annotation. Got func with signature (x))` |
| **返回的是 `CustomOpDef` 对象，不是函数** | `type(swiglu).__name__` → `CustomOpDef`；它有 `register_kernel` / `register_fake`，**没有** `register_impl` |

另外两条同样实测的边界：

- **`CustomOpDef` 有 `register_kernel`、没有 `register_impl`**（`hasattr(...,'register_impl')` → False）。名字容易记成 `register_impl`。
- **`torch.library.fallback` 在模块级不存在**：`hasattr(torch.library, "fallback")` → False；存在的是**方法** `torch.library.Library.fallback`。

### 1.2 `impl_abstract` → `register_fake` 的改名

旧名**仍然可用**，但会发一条警告。实测的类别与原文：

```
[FutureWarning] `torch.library.impl_abstract` was renamed to `torch.library.register_fake`.
                Please use that instead; we will remove `torch.library.impl_abstract` …
```

**是 `FutureWarning`，不是 `DeprecationWarning`**——写代码捕获警告时这个区别是实打实的。本机 2.6.0 / 2.7.1 / 2.8.0 / 2.11.0 / 2.13.0 **五个版本都还在发**（兼容期未结束）。

## 2. 执行映射：这些名字到底住在哪里

观测入口的位置本身也是个坑（用 `inspect.getsourcefile` 实测）：

| 名字 | 实际定义位置 |
|---|---|
| `torch.library.custom_op` | **`torch/_library/custom_ops.py`**（不是 `torch/library.py`），经 `@exposed_in("torch.library")` 暴露 |
| `torch.library.register_fake` | `torch/library.py` |
| `torch.library.define` / `impl_abstract` | 是 `functools.singledispatch` 包装过的对象，`__file__` 指向 `functools.py` / `typing_extensions.py`——**别拿它们的 `__file__` 当定义位置** |

`define` 的 singledispatch 包装还有一个实用后果：**它的签名随第一个参数的类型变**，而且**必须一次给两个位置参数**（`qualname, schema`）——只给一个字符串会得到 `TypeError: define() missing 1 required positional argument: 'schema'`（本项目第一版就踩了这个）。

### 2.1 自定义算子在 eager 链上的层数：和内置算子一样

实测（dispatch mode 拦截）：

```
torch.ops.kb04.swiglu.default(x)  →  ['kb04.swiglu.default']      1 次
x + 1                             →  ['aten.add.Tensor']           1 次
```

**自定义算子只是换了个 kernel，链本身没变**。它**内部**用到的 `chunk` / `sigmoid` / `mul` **不会再被拦一次**——因为 handler 里调用 `func` 时当前 mode 不会被重新进入（见 [01-dispatch-and-aten](01-dispatch-and-aten.md) §3）。这条对`torch.compile` 同样重要：把一串小算子包成一个 custom op，**eager 上省掉的正是这些中间节点的调度开销**。

## 3. 实现与演进：编译入口（用户接口视角）

> **边界**：`torch.compile` 的编译器内部（图捕获细节、融合、codegen、动态形状）**正文在 [60-compilers/inductor](../../60-compilers/inductor/01-graph-capture.md)**。本节只写「怎么用、怎么数」。
> **本文里的编译相关数字都是本机实测**，程序用 `torch.compile(fn, backend="eager")`——选 `eager` backend 是为了**不依赖本机的 C++ 编译器**，这样测的就是**图捕获与 guard**，而不是 codegen。

### 3.1 怎么数重编译

`torch._dynamo.utils.counters` 给出了两个可直接读的计数（本机五版本实测一致）：

| 字段 | 含义 |
|---|---|
| `counters["frames"]["total"]` | 被捕获的帧数 |
| `counters["stats"]["unique_graphs"]` | **编译出了几张不同的图** |
| `counters["stats"]["calls_captured"]` | 被捕获的调用次数 |

实测（`f(a, b) = relu(a + b).sum()`，`backend="eager"`，本机 torch 2.11.0）：

| 调用序列 | `unique_graphs` | `calls_captured` |
|---|---|---|
| 同 shape 调 3 次 `(4,)` | **1** | 3 |
| 再换 shape 到 `(8,)` 调 1 次 | **2** | 6 |
| 再把 dtype 换成 `float64` 调 1 次 | **3** | 9 |

**模型：guard 固化的假设是 `(shape, dtype)`；每出现一个新的组合就多一张图。** 这与本域主线（TF 的 `TraceType`、JAX 的 cache key）是同一个东西的三种写法——三者**默认都对 shape 敏感**。

### 3.2 Python 标量会被**按值**固化

`g(x, n) = x[:n].sum()`：`n=3` 连续两次只捕获 1 次；把 `n` 换成 `5` 会**再捕获一次**（本机实测：`frames.total` 增量 +1 → +2）。这与 TF 的 `LiteralTraceType<3>`（按值特化）同构，而 JAX 默认**不**特化 Python 标量。三者对照见 `70-frameworks/README.md` 的主题表与 TF/JAX 各篇。

### 3.3 诊断入口

| 手段 | 用来看什么 |
|---|---|
| `torch._dynamo.explain(f)(*args)` | 几张图、**几个 graph break**、每个 break 的原因（实测小函数：`Graph Count: 1 Graph Break Count: 0`） |
| `torch._dynamo.utils.counters` | §3.1 的三个计数 |
| `TORCH_LOGS=guards` / `recompiles` | 环境变量开关，打印 guard 与重编译原因 |
| `torch.compiler.is_compiling()` / `torch.compiler.disable` | 在编译上下文里做判断/局部退出 |

### 3.4 结构化控制流：`torch.cond` 与 `torch.while_loop`

实测（五版本一致）：

| | 存在 | 在 `torch.__all__` 里 |
|---|---|---|
| `torch.cond` | ✅ | **✅** |
| `torch.while_loop` | ✅ | **❌** |

两者的 docstring 都带同一条 Warning：**prototype feature**。`torch.while_loop` 的 docstring 逐字：

> 「`while_loop` is a structured control flow operator. It preserves the loop semantic across the torch.compile and torch.export.」

**「PyTorch 没有 while 版结构化控制流」是错的**——它有，只是不在 `__all__` 里、且是 prototype。**含 Python 控制流的代码在其它框架里的处理方式**（JAX 报错、TF 用 AutoGraph 转换、PyTorch 走 graph break）见各主题对应篇。

## 4. 踩坑与失败模式

### 失败模式 1：不传 `mutates_args` / 不给类型标注

**现象**：`TypeError: custom_op() missing 1 required keyword-only argument: 'mutates_args'`；或 `ValueError: infer_schema(func): Parameter x must have a type annotation.`

**成因**：`mutates_args` 是 keyword-only **且无默认值**——因为「这个算子会不会原地改输入」是**框架无法从代码推断**的信息，必须由作者声明；schema 则由**类型标注**推导，没有标注就推不出来。

**规避**：两个都写全。纯函数写 `mutates_args=()`；真的原地改就列出被改的参数名。

### 失败模式 2：拿 `_dispatch_has_kernel` 查自定义算子

**现象**：照 aten 的习惯写 `_dispatch_has_kernel("kb04::swiglu.default")`，得到
`RuntimeError: default is not a legal overload name for aten operators`。

**成因**：这个接口按 **aten 的重载名规则**解析。对自定义算子：**裸名**（`kb04::swiglu`）返回 `True`，**加 `.default`** 反而报错。而 aten 算子正相反（裸名 `False`、`aten::add.Tensor` 才 `True`）。

**规避**：查 aten 用重载名，查自定义算子用裸名。**「必须写重载名」这句规矩要带命名空间限定。**

### 失败模式 3：拿版号当门牌判「这个 API 有没有」

**现象**：照「2.13 新增了 `get_kernel` / `EffectType`」写兼容分支，结果在 2.11 上也走错了分支。

**成因**：本机实测推翻了这条版本归属——**2.11.0 上 `get_kernel` / `EffectType` / `CustomOpDef.register_effect` 就已经存在**；同样地，`torch.library` 泄漏的类型名**不是 2.13 才清的**（2.11.0 与 2.13.0 的泄漏名单**完全相同**）。

**规避**：**用特征探测（`hasattr`）而不是版号比较**。本机五个版本实测的公开面差异：

| 名字 | 2.6.0 | 2.7.1 | 2.8.0 | 2.11.0 | 2.13.0 |
|---|---|---|---|---|---|
| `overload` / `register_autocast` | ❌ | ✅ | ✅ | ✅ | ✅ |
| `get_kernel` / `EffectType` | ❌ | ❌ | **❌** | **✅** | ✅ |
| 泄漏的类型名 | …Dict/List/Optional/Sequence/Set/Tuple/Union | …Optional/Sequence/TypeVar/Union | 同 2.7.1 | …Sequence/TypeVar/Union（`Optional` 也清了） | 同 2.11.0 |

**且「2.13 把泄漏都清了」是错的**：`Any` / `Callable` / `Sequence` / `TypeVar` / `Union` 一直都在。
⚠ **2.9 / 2.10 / 2.12 本机没有解释器，这三个 minor 的公开面未核。**

### 失败模式 4：把「自定义算子」当成「绕过 dispatch」

**现象**：以为注册了 custom op 就"跳过"了调度链，于是在里面写任意 Python 副作用。

**成因**：自定义算子只是**换了 kernel**，链没变——它照样 1 次拦截、照样受 autograd 与 guard 的约束。

**规避**：把它当成「一个普通的 aten 算子」来对待：要有 schema、要有 fake 推断（否则 `torch.compile` 推断不出输出形状，只能保守处理）、要声明是否原地改。**要进图，就得把假设交代清楚**——这就是主线在本文的落点。

## 5. 实测数据

**测量条件**：`code/04_custom_ops.py` 经 `make run` 在本机跑出，选中的解释器是
`/Users/cachio/miniconda3/bin/python` = **torch 2.11.0 / python 3.12.9 / numpy 2.3.4 / CPU-only**。全部断言通过（A 6 / B 13 / C 12，共 31 条）。

> **★ 限定语（本域强制）**：下面每个数字都是**真实 PyTorch**跑出来的（不是迷你实现），**必须绑定版本**。**本机另有 2.6.0 / 2.7.1 / 2.8.0 / 2.13.0 四个解释器**（连被选中的那个共五个）；**没有 2.9 / 2.10 / 2.12 的解释器**，这三个 minor 不声称覆盖。

**（1）A 类：numpy 参照实现**（与被测 kernel 是**两条独立代码路径**）复核 `swiglu` 与 `affine` 两个自定义算子：最大绝对误差 **0.000e+00**（两条调用路径——直接调用与 `torch.ops.…default`——都一致）。反证也跑了：把参照的 sigmoid 门去掉，同一条「误差 < 1e-5」判据立刻判为不通过（误差 5.534e-01）。

**（2）B 类：Dynamo 的 guard 代价模型**（`backend="eager"`，计数来自 `torch._dynamo.utils.counters`）：

| 调用序列 | `unique_graphs` 模型 | 实测 | `calls_captured` 实测 |
|---|---|---|---|
| 同 shape 调 3 次 | 1 | **1** | 3 |
| 再换 shape 调 1 次 | 2 | **2** | 6 |
| 再换 dtype 调 1 次 | 3 | **3** | 9 |

Python 标量按值固化：`x[:n].sum()` 在 `n=3` 重复调用只捕获 1 次，换成 `n=5` **再捕获 1 次**。

**（3）C 类：约束与版本面**——`mutates_args` 缺失的 `TypeError`、无类型标注的 `ValueError`、`register_impl` 不存在、`torch.library.fallback` 模块级不存在、`impl_abstract` 的 `FutureWarning`、`_dispatch_has_kernel` 对自定义算子加 `.default` 的报错，**全部真的跑出来**；版本面见 §4 失败模式 3 的表（五个版本逐名核对，不是笼统断言）。

**（4）五个版本都是「全部通过」**：2.6.0 / 2.7.1 / 2.8.0 / 2.11.0 / 2.13.0 上本程序各以 **31 条断言全绿退出**（判据全部用特征探测写成，因此不随版本增减）。

## 6. 小结与延伸

- 三种注册入口：`custom_op`（schema 自动推导）、`define`+`impl`（schema 手写）、`register_fake`（形状推断）。三个易错点：`mutates_args` 是 keyword-only 且无默认值、函数必须有类型标注、返回的是 `CustomOpDef`。
- `custom_op` 定义在 `torch/_library/custom_ops.py`；`impl_abstract → register_fake` 发的是 **`FutureWarning`**，五个版本都还在兼容期。
- 自定义算子在 eager 上**与内置算子同层数**（1 次拦截）；注册的本质是**交出契约**，不是绕过调度——这就是主线里「判据来源」的落点。
- `torch.compile` 从用户接口看就是三件套：**数重编译**（`counters`）、**看 break**（`explain`）、**挂日志**（`TORCH_LOGS`）。guard 固化的假设是 `(shape, dtype)` + Python 标量值。
- **版本面的正确判据是特征探测，不是版号比较**（本机实测推翻了「2.13 新增 `get_kernel`」这类说法）。
- 编译器内部：[60-compilers/inductor](../../60-compilers/inductor/01-graph-capture.md)（图捕获）、[04-dynamic-shapes](../../60-compilers/inductor/04-dynamic-shapes.md)（动态形状与重编译）。

## 参考

- 实测程序：`code/04_custom_ops.py`（三类断言齐全，失败非零退出）
- 纯 python3 复核入口：`code/verify.py`（**不 import torch**，承担 A/C）
- 外部事实与版本锚点：`code/specs.py`
- 上游：`30-gpu-programming/cuda`（自定义 kernel 的写法；本文只讲「框架怎么注册与调用它」）
- 相邻：`60-compilers/inductor`（`torch.compile` 的编译器内部）

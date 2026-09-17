---
title: 自定义算子与导出：算子粒度与图的落地格式
slug: tensorflow-custom-ops-and-export
domain: 70-frameworks
topic: tensorflow
level: 2
difficulty: intermediate
prerequisites:
  - 70-frameworks/tensorflow/01-graph-and-eager
tags: [tensorflow, custom-op, custom-gradient, saved-model, export, keras, graph]
related:
  - 70-frameworks/tensorflow/01-graph-and-eager
  - 70-frameworks/pytorch/04-custom-operators
  - 70-frameworks/tensorflow/02-control-flow-and-xla
code:
  - code/03_custom_ops_and_export.py
status: stable
updated: 2026-09-17
---

# 自定义算子与导出：算子粒度与图的落地格式

> 同一个算子有两种接法：**用现有 Python 算子组合出来**（图上占好几个节点，每一步都暴露给
> 编译器）或**注册成单个 op**（图上只占一个节点，语义被藏进那条边界里）。
> 而图最终要**离开进程**——落地的格式带着**签名**，**签名就是被冻住的假设**：
> 载入之后换一个 shape 调用，会被直接拒掉。

## 0. 背景与动机

### 一个具体的对照场景

```
def swish(x):                      # 一个算子的三种落法
    return x * sigmoid(x)          # ① numpy 直算（参照）
                                   # ② 用现有算子组合（图上 5 个节点）
                                   # ③ 注册成一个 op（图上 1 个节点）
```

三种落法**数值上必须一致**，但在「图固化了什么」这件事上**完全不同**：

| 落法 | 图上占几个节点 | 编译器看得见什么 | 谁负责正确性 |
|---|---|---|---|
| Python 组合 | **多**（本例 5 个） | **每一步**（可以融合、可以重排） | 框架（每一步都是标准算子） |
| 注册成单个 op | **1 个** | **只有一条边界** | **你自己**（op 内部是黑盒） |

### 本文在本域主线上的位置：**①判据来源**

本域的主线是「**框架要把哪些假设固化进图**」（首篇
[01-graph-and-eager](01-graph-and-eager.md) 完整展开了这条主线）。本文挂的是
**①判据来源**这一面：

> **算子粒度决定「假设被写进哪个载体」**——写成分解的几个标准算子，
> 假设就在框架手里（框架知道 `exp` 会溢出、知道 `mul` 可以融合）；
> 注册成一个 op，假设就被搬进了**你的实现**里，框架（和编译器）**只能信你的声明**。
> 而**导出格式**是同一件事的另一半：**图离开进程时，带走的是一份带签名的、不可协商的假设**。

同主题另外两篇：01 展开主线，[02](02-control-flow-and-xla.md) 挂代价项。
与 PyTorch 的对照见 [70-frameworks/pytorch/04-custom-operators](../pytorch/04-custom-operators.md)
（PyTorch 那边的对应问题是「怎么把算子接进 dispatcher 那条链」）。

### 为什么这里容易记错

1. **「SavedModel 已经被废弃了」**——**过度概括**。`tf.saved_model.save` 在 v2.21.0
   源码里**没有任何 deprecated 装饰器**，而且 **Keras 3 的 `Model.export()` 默认格式就是
   SavedModel**（§3.1 有逐字）。
2. **「`tf.raw_ops` 是官方推荐的自定义算子入口」**——**官方文档里没这句话**：
   `create_op.md` 全文**0 次**提及 `raw_ops`（§1.1）。
3. **「注册成 op 只是换个写法」**——不。它把**正确性责任**从框架转移到了你：
   官方推荐的次序恰恰是**先组合**（§1.1）。

## 1. 原理：官方推荐的次序

### 1.1 先组合，不行才写 C++ op

官方逐字（`site/en/guide/create_op.md` 首段）：

> If you'd like to create an op that isn't covered by the existing TensorFlow library,
> **we recommend that you first try writing the op in Python as a composition of existing
> Python ops or functions**. If that isn't possible, you can create a custom C++ op.

官方紧接着给了「为什么值得写 C++ op」的三条理由：**表达不出来** / **组合出来效率不行** /
**想手工融合一段未来编译器也难融的组合**。三条都是**性能或表达力**的理由，
**没有一条是「这样更方便」**。

> ⚠ **不得写「`tf.raw_ops` 是官方推荐的 X」**：本仓库核对 `create_op.md` 全文，
> `raw_ops` 出现 **0** 次；而 `tf.raw_ops` 的官方定义原文这次也没取到
> （API 页在调查环境被网络拦死，`raw_ops.py` 是构建期生成的、不在源码仓库里）。
> **没有一手来源的东西不进正文。**

### 1.2 推导：算子粒度 → 图节点数 → 谁承担正确性

设一个算子被拆成 `k` 个标准算子（本例 `swish` 拆成 5 个：`mul(-1)` → `exp` → `add(1)`
→ `recip` → `mul`）。两种落法的差别：

| 量 | Python 组合 | 注册成单个 op |
|---|---|---|
| 图上节点数 | **`k`** | **1** |
| 图里能看到的中间值 | **`k-1` 个**（编译器可融合、可重排、可换精度） | **0 个** |
| 数值稳定性的责任 | 每一步都是标准算子，**框架的语义可预期** | **你的实现**（§2.2 的例子） |
| 需要写 C++ 吗 | 不需要 | 官方推荐「先组合」的理由就在于这一步的成本 |

一个具体的后果：**「组合出来的图」是可以被编译器优化的对象，「注册的 op」是一条不透明边界**。
所以 §1.1 的三条理由（表达不出来 / 效率不行 / 想手工融合）本质上是同一件事的三种说法——
**你在用「放弃可优化性」换「表达力或性能」**。

## 2. 执行映射：梯度与数值稳定性

### 2.1 `tf.custom_gradient` 的定位

官方 docstring 首句逐字（`tensorflow/python/ops/custom_gradient.py`）：

> **Decorator to define a function with a custom gradient.**

它的完整用法（同文件 docstring 的示例）：**正向照常写，反向额外给一个 `grad` 函数**：

```python
@tf.custom_gradient
def log1pexp(x):
  e = tf.exp(x)
  def grad(upstream):
    return upstream * (1 - 1 / (1 + e))
  return tf.math.log(1 + e), grad
```

**关键定位**：它作用在**已经能用现有算子写出来的函数**上——
它**不注册新 op**，只是**替换反向那一半**。这与 §1.1 的次序是同一件事的两面：

| 你的需求 | 用哪条路 |
|---|---|
| 现有算子拼不出来 | 写自定义 op（§1.1） |
| 拼得出来，但**反向**写得不好/不稳/太慢 | `tf.custom_gradient`（本节） |

> **本项目订正（计划 §3.1 的 S1）**：原计划里这条定位被标记为「没有来源」。
> 本次补上了一手来源：`custom_gradient.py` 的 docstring 首句 + 源码文件路径（见文末参考），
> 所以本文保留这条要求，并把它写在这里。

### 2.2 为什么需要自定义梯度：一个数值例子

官方 docstring 里那个例子（`log1pexp`）的动机是**数值稳定性**。
把两种写法摆在一起（**本仓库用 numpy 现场算的推导，不是 TF 的实测**）：

| `x` | 链式法则直接写的 `exp(x)/(1+exp(x))`（float32） | 化简式 `1 - 1/(1+exp(x))` |
|---|---|---|
| 1.0 | 0.731059 | 0.731059 |
| 20.0 | 1.000000 | 1.000000 |
| **100.0** | **NaN** | **1.000000** |

**机制**：float32 下 `exp(100)` 已经**溢出成 `inf`**，于是 `inf / (1 + inf)` = `inf/inf` = **NaN**。
化简式把同一件事重排成 `1 - 1/(1+exp(x))`，`1/(1+inf)` = `0`，结果是**正确的 1.0**。
**这就是「自定义梯度」要解决的问题**：同一段正向计算，**反向的写法决定它会不会 NaN**。

### 2.3 导出：图以什么格式离开进程

| 事实 | 依据 |
|---|---|
| `tf.saved_model.save` **没有被废弃** | v2.21.0 源码里它只有 `@tf_export(...)`，**没有任何 `@deprecated`** |
| **Keras 3 的 `Model.export()` 默认格式就是 SavedModel** | `Model.export(filepath, format="tf_saved_model", …)` |
| 被取代的是「把 SavedModel 当作 Keras 模型的**保存 / 再训练**格式」 | 官方原文（§3.1 引全） |

**导出的机制要点**（本仓库迷你模型，标注见 §5）：导出的是**已经 trace 好的图**，
**不是 Python 函数**——所以载入之后**调用它不产生任何追踪**，
但**它带着导出时固化的签名**：shape / dtype 不在签名允许的范围内，**直接被拒**。
**「图的落地」与「假设的落地」是同一次动作。**

## 3. 实现与演进

### 3.1 SavedModel 的真实状态（**引全官方那句**）

官方对 Keras 对象的原文逐字（含结尾那半句）：

> **Deprecated**: For Keras objects, it's recommended to use the new high-level `.keras`
> format and `tf.keras.Model.export`, as demonstrated in the guide here.
> **The low-level SavedModel format continues to be supported for existing code.**

准确说法分三层：

1. **SavedModel 没有被废弃**——`tf.saved_model.save` 在 v2.21.0 源码里**没有废弃装饰器**；
2. **它是 Keras 3 的默认导出格式**——`Model.export()` 的 `format` 默认值是 `"tf_saved_model"`；
3. **被取代的是用法，不是格式**：被推荐替掉的是「把 SavedModel 当作 Keras 模型的
   **保存 / 再训练**格式」这件事（那用 `.keras`）；SavedModel 自己的位置是**推理导出格式**。

### 3.2 演进的方向：从「可再训练」到「可部署」

把这三条放在一起看，方向是清楚的：

| 目的 | 格式 | 状态 |
|---|---|---|
| 断点续训 / 换框架继续训练 | `.keras` | **官方推荐**（对 Keras 对象） |
| **推理部署**（TF Serving / `saved_model_cli` / LiteRT） | **SavedModel** | **继续支持，且是 Keras 的默认导出格式** |
| 单一算子接进图 | 自定义 op / `tf.custom_gradient` | 见 §1、§2 |

## 4. 踩坑与失败模式

### 失败模式 1：把「导出」当成「保存源码」

**现象**：导出后换一个 shape（比如 batch 从 5 变成 7）调用载入的模型，
直接报错——而在导出前的 Python 里同样的调用是好的。
**成因**：导出的产物是**带签名的图**，签名里的 shape 是**导出那一刻**定下的。
**规避**：导出前**先想清楚服务期的形状范围**，用 `None` 维（未知维）去 trace，
让签名放得宽一些——这是 [01](01-graph-and-eager.md) §1.5 那条方向性在**部署**上的应用：
**「先用 `None` 维 trace 才能复用」不只对重追踪成立，对导出产物也成立。**

### 失败模式 2：为了「快」就注册自定义 op，结果把正确性责任揽到自己身上

**现象**：把一段组合换成一个手写 op，性能没怎么变，但数值行为出现了细微差异
（边界输入、非规格化数、极端值）。
**成因**：§1.2 那张表——注册成 op 之后，**中间值不再暴露给编译器**，
而**数值稳定性的责任转移到了你的实现里**。§2.2 那个 `NaN` 就是这类问题的典型形态。
**规避**：遵守官方的次序（**先组合，不行再写 op**）；写 op 之前先用组合版本
跑一遍**极端输入**（`x = ±100`、`x = 0`、极小/极大），把行为记录下来当规格。

### 失败模式 3：自定义梯度写成了**另一件事**

**现象**：`grad` 返回的梯度看起来「差不多」，训练也能跑，但收敛行为变了。
**成因**：`tf.custom_gradient` 是**替换**反向那一半，不是「额外加一条路径」——
它给什么，反传就用什么。**写错了没有任何自动检查**。
**规避**：把自定义梯度与**数值微分**（中心差分）对拍——这是唯一与解析推导**互不依赖**的参照。
`code/03_custom_ops_and_export.py` 的 §A 就是这么做的（含一条「故意写错必须被抓到」的反证）。

### 失败模式 4：**把 `tf.raw_ops` 当成官方推荐的自定义算子入口**

**现象**：文档/教程里写「要自定义算子就用 `tf.raw_ops`」，并称这是官方推荐。
**成因**：`create_op.md` **全文 0 次**提及它；官方推荐的是**先在 Python 里组合**
（§1.1 逐字）。而 `tf.raw_ops` 的官方定义原文**这次也没取到**（API 页被拦，
生成脚本不在源码仓库里）。
**规避**：**没有一手来源的话不写**。这一条是**写作侧**的失败模式，但它的代价和
技术错误一样大——读者会照着一个查不到出处的「推荐」去做技术选型。

## 5. 实测数据

**测量条件**：`code/03_custom_ops_and_export.py` 经 `make run` 在本机跑出
（python3 3.13.5 / numpy 2.4.4），全部 16 条断言通过（A 5 / B 5 / C 6）。

> **★ 限定语（本域强制）**：下面每一个数字都是**本仓库的迷你追踪器**跑出来的，
> **不是 TensorFlow**。本机**没有装 TF**。支持的是「算子粒度与导出产物怎么被计数与校验」
> 这类**机制**结论，**不支持任何关于真实 TensorFlow 的结论**。
> §2.2 那张数值表是**本仓库用 numpy 现场算的推导**，也不是 TF 的实测。
> 关于真实 TF 的行为断言以 §参考 的一手 URL 为依据。

**（1）节点的账**（迷你实现实测）：

| 落法 | 图上节点数 | 节点名 |
|---|---|---|
| Python 组合 `swish` | **5** | `tf.multiply, tf.exp, tf.add, __recip__…, tf.multiply` |
| 注册成单个 op | **1** | `swish__custom` |

**（2）导出产物**（迷你实现实测）：

| 项 | 值 |
|---|---|
| 签名 | `FunctionType(TensorSpec(TensorShape([5]), float32))` |
| 产物里的算子数 | **5**（== 导出前那张图的节点数） |
| 产物字节数 | **1827**（本仓库的 JSON 形态，**不是 SavedModel 的 protobuf**） |
| 载入后调用 | **不产生任何 tracing**；用同样的 shape 调用结果与参照一致（`max|Δ| = 4.4e-16`） |
| 载入后换 shape 调用 | **被签名拒掉**（`InvalidArgumentError`） |

**（3）梯度对拍**（迷你实现实测 + numpy 推导）：
化简式梯度与**中心差分**在 5 个点上的 `max|Δ| = 7.7e-11`；
把公式故意写错（丢掉 `1 -`）时差异 **1.0**——**说明这条 A 类断言不是恒真**。

## 6. 小结与延伸

- 官方次序：**先试着用现有 Python 算子组合，不行才写自定义 C++ op**（逐字）；
  写 op 的理由是**表达力 / 效率 / 手工融合**，不是「方便」。
- **算子粒度 = 图上的节点数 = 谁承担正确性**：组合（多节点、中间值可见、框架负责语义）
  对比注册 op（1 节点、黑盒、你负责）。
- `tf.custom_gradient` 的定位是「**给已有的组合换反向**」，**不是注册新 op**；
  它的动机常常是**数值稳定性**（§2.2 的 `NaN`）。
- **SavedModel 没有被废弃**：`tf.saved_model.save` 没有 `@deprecated`，
  **Keras 3 的 `Model.export()` 默认就走 `tf_saved_model`**；
  被取代的是「把 SavedModel 当 Keras 模型的保存 / 再训练格式」这件事，
  官方那句的结尾是「**The low-level SavedModel format continues to be supported for
  existing code.**」。
- **导出 = 图 + 签名一起落地**：载入后不重追踪，但**形状假设被冻住**——
  要用 `None` 维留出余量。
- 往下一站：[70-frameworks/jax](../jax/) 主题（同一件事在 JAX 里的形态：
  `jax.export` 的符号形状**仍会按具体形状重编译**）。

## 参考

**本主题的配套程序**：`code/03_custom_ops_and_export.py`（三类断言齐全，失败非零退出）。

- **自定义算子的官方次序（先组合）**：`site/en/guide/create_op.md`
  <https://github.com/tensorflow/docs/blob/35e0922e059d7bc6d515a83e03a7494f0640c314/site/en/guide/create_op.md>
- **`tf.custom_gradient` 的 docstring 首句**：`python/ops/custom_gradient.py:46-47`
  <https://github.com/tensorflow/tensorflow/blob/v2.21.0/tensorflow/python/ops/custom_gradient.py#L46-L47>
- **SavedModel 的 Deprecated 原文（含结尾那半句）**：`site/en/guide/saved_model.ipynb`
  <https://github.com/tensorflow/docs/blob/35e0922e059d7bc6d515a83e03a7494f0640c314/site/en/guide/saved_model.ipynb>
- **`tf.saved_model.save` 只有 `@tf_export`、没有 `@deprecated`**：
  `python/saved_model/save.py:1241-1244`
  <https://github.com/tensorflow/tensorflow/blob/v2.21.0/tensorflow/python/saved_model/save.py#L1241-L1244>
- **Keras 3 `Model.export()` 的默认格式**：<https://keras.io/api/models/model_saving_apis/export/>

**回链**：PyTorch 侧把算子接进 dispatcher 那条链的做法见
[70-frameworks/pytorch/04-custom-operators](../pytorch/04-custom-operators.md)；
编译入口（`torch.compile` 的用户接口视角）也在那一篇。

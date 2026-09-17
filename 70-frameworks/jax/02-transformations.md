---
title: 变换：vmap 给每个 primitive 定义 batching rule，它不展开循环
slug: jax-transformations
domain: 70-frameworks
topic: jax
level: 2
difficulty: intermediate
prerequisites:
  - 70-frameworks/jax/01-jit-and-tracing
tags: [jax, vmap, batching-rule, jaxpr, donate-argnums, immutability, out-of-place-update]
related:
  - 70-frameworks/jax/01-jit-and-tracing
  - 70-frameworks/jax/03-sharding-and-differences
  - 70-frameworks/tensorflow/02-control-flow-and-xla
code:
  - code/02_vmap_batching_rules.py
status: stable
updated: 2026-09-17
---

# 变换：vmap 给每个 primitive 定义 batching rule，它不展开循环

> `vmap` 不是「帮你写一个 for 循环」。它的机制是**给每个 primitive 定义一条
> batching rule（batcher）**，把被映射的轴**推进 primitive 的语义里**。
> 所以图里根本没有 batch 大小这个量：官方逐字
> 「**the program stays the same size no matter the batch**」，
> 而手写 Python 循环的版本「contains **a full copy of the body *per batch element***」。
> 同一个机制决定了另外两件事：**jaxpr 是显式类型的 ANF 中间表示**，
> 以及 **JAX 数组不可变**——`x.at[idx].set(y)` **不是**就地修改的语法糖。

## 0. 背景与动机

### 一个具体的对照场景

```
同一个逐行处理的函数 body，两种写法：
  · vmap(body)                       —— 一行代码，不管 batch 是 4 还是 4096
  · [body(x[i]) for i in range(B)]   —— 「我自己写循环」的朴素写法
```

在 eager 的 numpy 世界里这两件事看起来都只是「循环」。但**一旦要交给编译器**，
它们是两种完全不同的东西：

| | 图里有什么 | 图的大小 |
|---|---|---|
| **`vmap(body)`** | **和单条数据时一模一样的程序**，只是每个 primitive 多带一根 batch 轴 | **与 batch 无关** |
| **手写 Python 循环** | **每一步都复制了一份 body** | **随 batch 线性增长** |

### 本文在本域主线上的位置：**②代价项**

本域的主线是「**框架要把哪些假设固化进图**」（首篇
[01-jit-and-tracing](01-jit-and-tracing.md) 完整展开了这条主线）。本文挂的是其中的
**代价项**：**变换本身不建图，它叠在同一层上**。

`vmap` 固化的假设是「**有一根轴要被映射**」——**不是**「B 是 4」。
这个区别是可数的：`code/02_vmap_batching_rules.py` 的 §C1 量到，
vmap 版本的语句数在 `B = 4 / 8 / 64 / 256` 下**全是 3**，
手写循环是 `17 / 33 / 257 / 1025`（**迷你实现实测，不是 JAX**）。

jaxpr 那一半（§2）挂的也是代价项：**图的形态是被显式类型化的**，
所以「图上有什么」是一份可读、可数的清单。
`donate_argnums`（§3）与不可变性（§4）则是**同一套不可变语义**的两个落点。

## 1. 原理：batching rule 是什么

### 1.1 注册方式

官方讲 primitive 的那一页（`docs.jax.dev`，601 层）给的注册代码逐字：

```python
batching.primitive_batchers[multiply_add_p] = multiply_add_batch
```

也就是说：**一个 primitive 想在 `vmap` 之下工作，必须有人给它写一条 batcher**。
没写会怎样？源码里那条 `raise` 逐字（`jax/_src/interpreters/batching.py:275`）：

```python
raise NotImplementedError(f"Batching rule for '{p}' not implemented")
```

运行时渲染成 `Batching rule for '<primitive>' not implemented`。
**注意源码是 f-string、运行时才渲染**——引用时要分清引的是哪一层。

> ⚠ **本项目订正（P7）**：计划 §0.7.C 把
> `fancy_primitive_batchers` / `primitive_batchers` / `axis_primitive_batchers`
> 并列为「源码里的三张注册表」。**在 `jax-v0.11.1` 里这已经不准确**：
> 真正被查表的只有 **`fancy_primitive_batchers`**（`:585`）；
> 另外两个在源码里紧挨着 `# backwards compat shim. TODO: delete` 的注释，
> 是**写穿**到前者的代理对象（`:588` / `:596`）。
> 而**文档里教用户写的那句**用的恰好是代理名 `primitive_batchers`——
> 两者并不矛盾：代理就是为兼容旧写法留的。

### 1.2 ★ 必须纠正的一条：vmap **不是**把循环展开

官方那篇变换教程（`docs.jax.dev/en/latest/101/transformations.html`）的正文逐字：

> This is the same program, operation for operation. The only change is that every
> operation gained a batch axis: each `f32[5]` became `f32[2,5]`, and each `dot_general`
> picked up a batch dimension. That's what "`vmap` replaces each intercepted operation
> with a batched version" means concretely: the batching happens *inside* each operation,
> where array-level hardware parallelism lives, and **the program stays the same size no
> matter the batch**.

同页紧接着给了对照（讲一个手写循环的反例函数）：

> Compare the Python-loop version, `manually_batched_convolve`, whose recording contains
> **a full copy of the body *per batch element***: six `dot_general`s, not three, and
> **growing linearly with the batch size**.

**「不展开」这件事是本文的主结论**，它的成因在下一节。

> ⚠ **来源说明（这是本主题一条要交代清楚的账）**：上面两句**不在** `jax.vmap` 的
> docstring 里，也**不在** `jax/_src/api.py` 里——本仓库在 `jax-v0.11.1` 的源码树与
> 指定文档页里逐处排除过，它们**只在这篇教程的正文**里。
> 所以引用时必须写成「`101/transformations` 教程页逐字」，
> **不能写成「`jax.vmap` 的 docstring 说」**——那正是 P7（引文真实存在但贴错位置）。
> 出处：<https://docs.jax.dev/en/latest/101/transformations.html>；
> tag 内同文：`docs/new_docs/101/transformations.md`。

### 1.3 为什么图不随 batch 变大：batcher 拿不到 B

把 §1.1 与 §1.2 拼起来，机制就完整了：

1. 追到 `vmap` 时，框架压入一层 **batch trace**，把每个实参包成一个
   「**带一根被映射轴**」的符号值；
2. 函数体里每一个 primitive 调用都被这层截住，交给**该 primitive 注册的 batcher**；
3. **batcher 是在「低一层」的语言里写的**——它只知道「有一根轴要被映射」，
   **拿不到 batch 是几**；
4. batcher 在低一层调用**同一个 primitive**，把轴推下去（例如把 `reduce_sum` 的
   `axes` 整体**右移一位**，好让被映射轴不被吃掉）。

第 3 条就是全部：**B 从来没有进入过这张图**，所以图当然与 B 无关。
`code/02_vmap_batching_rules.py` 的 §B 把这一步打印出来了——
`jit(vmap(f))` 追出来的 jaxpr 是

```
{ lambda  ; v0. let
    v1 = mul v0 const(2.0)
    v2 = add v1 const(1.0)
    v3 = reduce_sum v2 axes=[1]
  in (v3) }
```

注意 `axes=[1]`：函数体里写的是「把所有元素加起来」（`axes=None`），
**batcher 把它改成了 `axes=[1]`**——轴 0 留给 batch。这就是「推下去」的具体样子。

### 1.4 变换是可组合的

`jit` / `vmap` / `grad` **不是三个各自建图的系统**，它们是**叠在同一层上的变换**：

- `jit(vmap(f))` 与 `vmap(jit(f))` 数值一致（§A 验过）；
- `jit` 在别的变换之下是**透明的**（[01](01-jit-and-tracing.md) §2.1）：
  `jax.grad(jax.jit(f))` 与 `jax.grad(f)` 得到同一张 jaxpr。

所以「变换」这个词是准确的：**每一个都在同一个程序上做一次改写**，
而不是各自造一个新程序再拼起来。

## 2. 执行映射：jaxpr 长什么样

### 2.1 定义与文法

官方的 jaxpr 专页（`https://docs.jax.dev/en/latest/601/jaxpr.html`）逐字：

> It is **explicitly typed, functional, first-order, and in algebraic normal form (ANF)**.

文法：

```
jaxpr ::= { lambda Var* ; Var+. let Eqn* in [ Expr+ ] }
```

四个性质各自意味着什么：

| 性质 | 含义 | 后果 |
|---|---|---|
| **explicitly typed** | 每个变量都带 aval（形状 + dtype） | 图**不依赖**具体数值就能读懂、能编译 |
| **functional** | 没有副作用、没有可变状态 | 重放 = 按方程顺序求值，天然可并行 |
| **first-order** | 只有 primitive 与变量，没有高阶函数 | `vmap` / `grad` 这类「高阶」的东西**必须在这一层之上**做 |
| **ANF** | 每个中间结果都绑定到一个**新变量** | 依赖关系显式、可数（本文的语句数就是数这个） |

**`first-order` 这一条尤其值得记住**：它正是「变换为什么必须叠在上面」的答案——
`vmap` / `grad` 是**对这张图的改写**，而不是图里的一种节点。

### 2.2 `constvars` 与 `invars`：分号是分界线

`jaxpr ::= { lambda Var* ; Var+. let … }` 里那个**分号**是硬边界：

| | 位置 | 含义 | 谁进来 |
|---|---|---|---|
| **`constvars`** | 分号**前** | 被**闭包进图**的常量 | `static_argnums` 的值、闭包捕获的常量 |
| **`invars`** | 分号**后** | **每次调用都要喂进来**的输入 | 数组实参、**默认情况下的 Python 标量** |

这个区分直接连回 [01](01-jit-and-tracing.md) §1.3：
**「进不进 cache key」决定「进 `constvars`/常量，还是进 `invars`」**。
`code/02_vmap_batching_rules.py` 的 §C3 把一张最小的 jaxpr 打出来逐项核对。

### 2.3 `jax.make_jaxpr()` 返回的是 `ClosedJaxpr`

官方把它的返回值叫 **`ClosedJaxpr`**——**jaxpr + 它闭包住的常量**（`consts`）。
名字里的「closed」就是「闭包」：图里引用的外部常量已经被收进来了。

## 3. 实现与演进：`donate_argnums`——把 buffer 的所有权交出去

### 3.1 它解决什么问题

一次 jit 调用的输入 buffer 与输出 buffer 通常是**两份**内存。
如果你确定「这个输入用完就不会再用」，可以让框架**复用**它的内存来放结果——
这在显存紧张时值一大笔。这就是 buffer donation。

官方在 jit 文档的 donation 一节逐字（`https://docs.jax.dev/en/latest/201/jit.html`）：

> Donated buffers are invalid afterwards. Using the donated input after the call is an
> error:

> ⚠ **版本提醒**：`jax-v0.11.1` 的 tag 里**同一页的另一版措辞**是
> 「**Donated means gone.** After the call, the donated input is invalid, and using it
> is an error:」。**两者不是同一句**，引 dev 版就不能说它是 0.11.1。
> （详见 `code/specs.py` 的 `DOC_DRIFT`。）

`donate_argnums` 的参数说明逐字（`jax/_src/api.py:267-288`）：

> `donate_argnums`: optional, collection of integers to specify which positional argument
> buffers can be overwritten by the computation and marked deleted in the caller. …
> **You should not reuse buffers that you donate to a computation; JAX will raise an
> error if you try to.** By default, no argument buffers are donated. If neither
> `donate_argnums` nor `donate_argnames` is provided, no arguments are donated.

### 3.2 一条真实存在的坑：`donate_argnums` **只覆盖位置实参**

同一段 docstring 说得很清楚：`donate_argnums` 与 `donate_argnames` 的对应关系由
`inspect.signature(fun)` 决定，**两者职责不同**。所以：

```python
@jax.jit(donate_argnums=(1,))
def f(x, y): ...
f(x, y)     # y 被捐 —— 位置实参
f(x, y=y)   # y **没有**被捐 —— 它这次是关键字传进来的
```

`code/02_vmap_batching_rules.py` 的 §C4 把这个跑了出来：
同一份 `donate_argnums=(1,)`，**位置传参时 buffer 失效；关键字传参时仍然可用**
（**迷你实现实测，不是 JAX**）。要用关键字捐，得写
`donate_argnames=('y',)`。

> ⚠ **计划里未取到来源的两条，本仓库不写**：计划 §0.7.C 还列了
> 「**pytree 参数会捐出全部 buffer**」与「**用不上的 donation 会被丢弃并告警**」。
> 本次核到的 docstring（`api.py:267-288`）里**没有这两条**，本仓库**没有取到**它们的一手来源，
> 故**正文不使用**（记在 `code/specs.py` 的 UNSOURCED 段）。

## 4. 踩坑与失败模式

### 失败模式 1：以为 `x.at[idx].set(y)` 是「就地修改的语法糖」

**现象**：写 `x = x.at[i].set(v)` 时以为它等价于 numpy 的 `x[i] = v`，
于是写出依赖「别名会被一起改」的代码，结果**静默算错**（原数组根本没变）。
**成因**：JAX 数组**不可变**，`.at[]` 是 **out-of-place** 的一套语法：
它返回**新数组**。`jax.numpy.ndarray.at` 的 docstring 逐字：

> **None of the x.at expressions modify the original x; instead they return a modified
> copy of x.** However, inside a jit() compiled function, expressions like
> `x = x.at[idx].set(y)` are **guaranteed** to be applied in-place.

注意最后半句说的是**「保证就地应用」**——这是一个**编译器的保证**，
它**不改变这套 API 的语义**：你在 Python 层看到的仍然是「返回新数组」。
**规避**：把 `.at[]` 一律当成「生成一份更新后的副本」来写；
如果确实需要就地语义，那是**编译器**该做的优化，不是你的代码该依赖的性质。

> ⚠ **位置订正（P7）**：另一句常被引用的话——
> 「However, inside `jit`-compiled code, if the input value `x` of `x.at[idx].set(y)`
> is not reused, **the compiler will optimize** the array update to occur in-place」——
> **不在** `.at` 的 docstring 里，它在 *Common Gotchas in JAX* 笔记里。
> **两句措辞不同**（「保证就地」vs「编译器会优化成就地」），**不得互相替代**。
> 来源：<https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html>

### 失败模式 2：写 `x += y` 指望它就地累加

**现象**：`x += 1.0` 之后，**别处仍指向旧数组的名字看到的还是旧值**——
如果代码里存在这样的别名，结果就错了。
**成因**：这一点官方文档自己就写了（*Common Gotchas* 笔记逐字）：

> That's because NumPy defines `__iadd__` to perform in-place mutation. In contrast,
> **`jax.Array` doesn't define an `__iadd__`**, so Python treats
> `jax_array_new += 10` as syntactic sugar for
> `jax_array_new = jax_array_new + 10`, **rebinding the variable without mutating any
> arrays**.

也就是说：`+=` 退化成了一次 `__add__` + 重新绑定，**没有任何就地修改发生**。
**规避**：把 JAX 数组一律当**不可变值**用；不要在脑子里建「同一个 buffer 被两个名字共享」的模型。
（`code/02_vmap_batching_rules.py` 的 §C5 用「别名不变」把这条跑了出来：
可观测面是**别名**，不依赖那句机制从句本身。）

### 失败模式 3：`x[idx] = y`——直接报错

**现象**：traceback 一句 `TypeError`。完整原文逐字（`jax/_src/numpy/array_methods.py:971-975`）：

```
JAX arrays are immutable and do not support in-place item assignment. Instead of
x[idx] = y, use x = x.at[idx].set(y) or another .at[] method:
https://docs.jax.dev/en/latest/_autosummary/jax.numpy.ndarray.at.html
```

**成因**：数组不可变，`__setitem__` 被实现成「抛错 + 教你怎么写」。
**规避**：照报错改写成 `.at[]` 形式。
> ⚠ **本项目订正**：计划 §0.7.C 给的引文**缺了尾部那半句**
> （`or another .at[] method: <url>`），那不是逐字；本文用完整版。

### 失败模式 4：对着没有 batcher 的 primitive 做 `vmap`

**现象**：`NotImplementedError: Batching rule for '<primitive>' not implemented`。
**成因**：§1.1 的机制——**batching rule 是要有人写的**。
一个 primitive 若只定义了「怎么算」而没定义「有被映射轴时怎么算」，`vmap` 就无从下手。
**规避**：改用有 batcher 的 primitive 组合出来（这也是 [tensorflow/03](../tensorflow/03-custom-ops-and-export.md)
里那条「先组合，再考虑写自定义算子」在 JAX 侧的对应物）；
实在不行就给那个 primitive 补一条 batcher。

## 5. 实测数据

**测量条件**：`code/02_vmap_batching_rules.py` 经 `make run` 在本机跑出
（python3 3.13.5 / numpy 2.4.4），全部 **25** 条断言通过（A 5 / B 6 / C 14）。

> **★ 限定语（本域强制）**：下面每一个数字都是**本仓库的迷你追踪器**跑出来的，
> **不是 JAX**。本机**没有装 JAX**，所以这里没有任何「真实 JAX 实测」。
> 这些数字支持的是「按 batching rule 把轴推下去这套机制可以怎样被计数」这类**机制**结论，
> **不支持任何关于真实 JAX 的结论**。关于真实 JAX 的行为断言以 §参考 的一手 URL 为依据。

**（1）图里的语句数**（迷你实现实测）——本文的核心那张表：

| batch 大小 | `jit(vmap(f))` 的语句数 | 手写 Python 循环的语句数 |
|---|---|---|
| 4 | **3** | 17 |
| 8 | **3** | 33 |
| 64 | **3** | 257 |
| 256 | **3** | 1025 |

vmap 版**恒为 3**；手写循环是 `4B + 1`（逐项差 16 / 224 / 768）。
**这就是「不展开循环」在这个模型里的可数形态。**

**（2）图里记着 batch 吗**：`jit(vmap(f))` 的 jaxpr 是
`mul` → `add` → `reduce_sum axes=[1]`——**三个 primitive，没有任何一项与 B 有关**；
`reduce_sum` 的 `axes` 被从「全部」改写成 `[1]`，正对应 §1.3 的「把轴推下去」。

**（3）追踪次数与缓存**：`B=4` 与 `B=64` 是**两个形状** ⇒ **两次追踪、两个缓存条目**
（cache key 的第一项含 shape，见 [01](01-jit-and-tracing.md) §1.1），
但**两张图的方程数相同**。

**（4）注册表可枚举**：本模型里已注册 batching rule 的 primitive 共 **20** 个；
`stack` **故意没有** batcher，用来演示 §4 那条报错。

**（5）本页引用的**引用值**（不来自本仓库测量）**：三句英文引文分别出自
`101/transformations` 教程页、`601/jax-primitives` 页、`jax/_src/interpreters/batching.py:275`；
`donate` 那句出自 `201/jit.html`（**dev 文档**）。**本仓库不复核其测量口径。**

## 6. 小结与延伸

- **`vmap` 的机制是「给每个 primitive 定义 batching rule（batcher）」**，
  它把被映射的轴**推下去**到 primitive 的语义里。
- **它不展开循环**：官方逐字 **the program stays the same size no matter the batch**，
  对比手写循环的 **a full copy of the body per batch element**。
  可数的形态：vmap 版语句数恒为 **3**（B = 4/8/64/256），手写循环 `4B+1`。
- **batcher 拿不到 batch 大小**——这就是「图里没有 B」的成因。
- **jaxpr 是 explicitly typed / functional / first-order / ANF** 的；
  `jaxpr ::= { lambda Var* ; Var+. let Eqn* in [ Expr+ ] }` 里**分号前是 `constvars`、
  分号后是 `invars`**——「进不进 cache key」决定进哪一边；
  `jax.make_jaxpr()` 返回的是 **`ClosedJaxpr`**。
- **`donate_argnums`**：官方逐字 **Donated buffers are invalid afterwards**；
  **只覆盖位置实参**，关键字实参要 `donate_argnames`。
- **不可变性**是这一节所有失败模式的共同根源：`x[idx] = y` 直接抛 `TypeError`；
  `x.at[idx].set(y)` **不是**就地的语法糖（**它是 out-of-place**，
  编译器**可能**把它优化成原地）；`+=` **不是就地**（`jax.Array` 没定义 `__iadd__`，
  是官方文档自己写的）。
- 往下一站：[03-sharding-and-differences](03-sharding-and-differences.md)
  （多设备那条路：`pmap` → `pjit` → `jax.jit` → `shard_map`，以及三者对照的收束）；
  回看 [01-jit-and-tracing](01-jit-and-tracing.md)（cache key 三项）。

## 参考

**本主题的配套程序**：`code/02_vmap_batching_rules.py`（三类断言齐全，失败非零退出）。
**本项目核过的一手来源**（全部登记在 `code/specs.py`，取用于 2026-09-17）：

- **「the program stays the same size no matter the batch」与其对照句**：
  `docs.jax.dev/en/latest/101/transformations.html`（**dev 文档**）
  <https://docs.jax.dev/en/latest/101/transformations.html>
  tag 内同文：`docs/new_docs/101/transformations.md`（jax-v0.11.1）
  <https://raw.githubusercontent.com/jax-ml/jax/jax-v0.11.1/docs/new_docs/101/transformations.md>
  > ⚠ **这两句不在 `jax.vmap` 的 docstring 里**（已逐处排除），只在教程正文。
- **batching rule 的注册写法**：`docs.jax.dev/en/latest/601/jax-primitives.html`
  <https://docs.jax.dev/en/latest/601/jax-primitives.html>
- **三张注册表的真实结构（只有 `fancy_primitive_batchers` 是真表）**：
  `jax/_src/interpreters/batching.py:585-608`（jax-v0.11.1）
  <https://github.com/jax-ml/jax/blob/jax-v0.11.1/jax/_src/interpreters/batching.py#L585-L608>
- **缺 batching rule 的报错**：`jax/_src/interpreters/batching.py:275`
  <https://github.com/jax-ml/jax/blob/jax-v0.11.1/jax/_src/interpreters/batching.py#L275>
- **jaxpr 的定义与文法**：`docs.jax.dev/en/latest/601/jaxpr.html`（**dev 文档**）
  <https://docs.jax.dev/en/latest/601/jaxpr.html>
- **`Donated buffers are invalid afterwards.`**：`docs.jax.dev/en/latest/201/jit.html`
  （**dev 文档**；tag 内同页写的是「Donated means gone.」）
  <https://docs.jax.dev/en/latest/201/jit.html>
- **`donate_argnums` 的参数说明**：`jax/_src/api.py:267-288`
  <https://github.com/jax-ml/jax/blob/jax-v0.11.1/jax/_src/api.py#L267-L288>
- **`x[idx] = y` 的完整报错原文**：`jax/_src/numpy/array_methods.py:971-975`
  <https://github.com/jax-ml/jax/blob/jax-v0.11.1/jax/_src/numpy/array_methods.py#L971-L975>
- **`jax.Array` 没定义 `__iadd__`**：*Common Gotchas in JAX* 笔记（**dev 文档**）
  <https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html>
- **`.at[]` 的 docstring（「guaranteed to be applied in-place」）**：
  <https://docs.jax.dev/en/latest/_autosummary/jax.numpy.ndarray.at.html>

**记录在案、正文不用的项**（`code/specs.py` 文件头的 UNSOURCED 段）：
「pytree 参数会捐出全部 buffer」与「用不上的 donation 会被丢弃并告警」**没有取到一手来源**，
故正文不写。

**回链**：控制流的两条路（固化一支还是两支都进图）见
[tensorflow/02-control-flow-and-xla](../tensorflow/02-control-flow-and-xla.md)；
XLA 的 HLO / 融合 / layout 正文在
[60-compilers/xla](../../60-compilers/xla/01-hlo-and-shape-inference.md)。

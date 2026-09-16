---
title: 动态形状与重编译
slug: inductor-dynamic-shapes
domain: 60-compilers
topic: inductor
level: 3
difficulty: advanced
prerequisites:
  - 60-compilers/inductor/01-graph-capture
tags: [inductor, dynamic-shapes, guards, recompilation, mark-dynamic, recompile-limit]
related:
  - 60-compilers/inductor/03-codegen
  - 60-compilers/inductor/02-fusion-scheduling
  - 70-frameworks/pytorch
code:
  - code/04_recompile_count.py
status: stable
updated: 2026-09-16
---

# 动态形状与重编译

> `torch.compile` 的 `dynamic` 参数默认是 **`None`**，而 `automatic_dynamic_shapes` 默认是 **`True`**：先按**静态**形状编译一次，guard 失败之后**自动把变化的那些维标成 dynamic** 再编一次，此后新形状命中同一张动态图。所以「N 个不同 shape 就重编译 N 次」是错的——本仓库的迷你实现测到的增量是 **1**（N=6）。而官方对 `dynamic=True` 的态度是 **"(Not recommended)"**，推荐的做法是**用户标注**。

## 0. 背景与动机

对照场景（两个都是真实世界的常用写法）：

```python
# A：形状会变，但变化是「同一维变大变小」
for batch in (4, 8, 16, 32):
    out = step(torch.randn(batch, 3), w)

# B：形状一直没变，但一个 float 一直在变
for lr in scheduler:
    out = step(x, w, lr)      # lr 是 Python float
```

直觉上 A 会不停重编译（每个 batch 一个新形状），B 完全不用担心（形状没变）。**实际是反过来的**：

- A 里，第二次 guard 失败之后，Dynamo 会把变化的那一维**自动标成 dynamic** 重编一次；此后 batch 再变，命中的是**同一张动态图**。
- B 里，`lr` 这种 Python `int`/`float` 会被 guard 成**常量的精确值**，改一次学习率就重编译一次——**而它与形状毫无关系**。

本文与主线的关系：**本文不涉及主线**。理由是：guard 与重编译决定的是「**这张图要不要重做一次**」，而物化决策（把哪些中间结果落内存）仍然发生在调度阶段。重编译会**重复**调度阶段的工作，但它自己不做任何物化决策。这条边界值得写清楚，因为「重编译很贵」与「融合很贵」是两种不同的贵，混起来会把优化方向搞反。

§1 推 guard 的内容与重编译语义，§2 推默认路径，§3 讲官方对 `dynamic=True` 的态度（**方向与直觉相反**），§4 讲三个有来源的失败模式。

## 1. 原理

### 第一步：guard 是什么

一份编译产物能不能复用，取决于一组**布尔假设**是否成立。官方对 guard 内容的措辞是：

> constant values, types, and object IDs

拆开看是三类：

- **types / object IDs**：操作数的类型与身份（是不是同一个张量对象、同一个 `nn.Module`）。形如 `TENSOR_MATCH` 的 guard 会检查 type / device / dtype / shape。
- **constant values**：被当作常量捕获的 Python 值——**包括 `int` 与 `float` 变量的精确值**。

第二类是 §4 失败模式 ② 的来源，也是「为什么改学习率会重编译」的全部答案。

### 第二步：什么时候必须重编译

官方措辞的关键在一个量词上：

> **If the guards fail for every instance of previously compiled code**, then `torch.compile` must recompile.

**「every instance」**——只有当**所有**已编译实例的 guard 都失败时，才必须重编译。这一点容易被读漏：一个帧可以同时挂着好几份编译产物（各自对应一组 guard），只要还有一份能用，就不重编。**量级**：一个 code object 上最多挂多少份是有上限的，见 §4 失败模式 ①。

### 第三步：默认路径——静态先行，失败后转动态

这是本文最要紧的一段，因为它决定了「N 个不同 shape 会发生什么」。

默认参数是：

| 参数 | 默认值 |
|---|---|
| `torch.compile(dynamic=...)` | **`None`** |
| `automatic_dynamic_shapes` | **`True`** |

`dynamic=None` 的官方解释逐字是：

> By default (None), we automatically detect if dynamism has occurred and compile a more dynamic kernel upon recompile.

于是默认路径是三步：

1. **第一次调用**：按完全静态的形状编译（`batch=4` 被写进生成物）。
2. **第二次调用**：形状变了 → guard 失败 → **把变化的那些维标成 dynamic**（`automatic_dynamic_shapes_mark_as` 默认 `"dynamic"`）重新编译一次。
3. **之后**：新的 batch 只要变化发生在**已标 dynamic 的那些维**上，就命中第 2 步那张图，**不再重编译**。

**所以重编译次数的期望是 O(变化的维度数)，不是 O(不同形状数)。**

一个容易被忽略的细节：**「每个新 shape 至少多付一次编译」是内禀的**。第 1 步永远是静态的，第 2 步永远要再来一次。要跳过第 1 步，唯一的办法是在**编译之前**就告诉它哪些维是动态的——这正是 §3 要讲的用户标注。

### 第四步：另一类恒定的重编译源

除了形状，还有一类值会被当成常量 guard：**`int` / `float` 变量的精确值**。经典的例子就是 LR scheduler 的 `lr`——每一步换一个值，每一步都重编译，而**形状从头到尾没变过**。

`nn.Module` 的整型属性也会被特化，属于同一类。

**这类重编译的特点**：它不随输入数据的形状变化，所以「我把 batch 固定住了」并不能避免它；而且它的出现与「模型是不是动态形状的」完全无关。

## 2. 执行映射

**组件一：guard 与 code object 级缓存。** 编译产物挂在 **code object** 上（官方原话 `compile caches are per code object, not frame`，见 [01-graph-capture](01-graph-capture.md) §1）。这意味着**同一个函数被一万次调用共享一份缓存**，而动态地造出一万个长得一样的函数也共享——后者常常是「为什么我的 guard 老失效」的真正原因。

**组件二：`automatic_dynamic_shapes` 的开关位置。** 默认值在三版上都是「真」：torch 2.6.0 与 2.7.1 的 `torch/_dynamo/config.py` 里是字面量 `True`；torch **2.13.0** 改成读环境变量 `TORCH_DYNAMO_AUTOMATIC_DYNAMIC_SHAPES`，默认字符串 `"1"`——**行为不变，仍然默认为真**。引用这条时**必须带版本**，因为这个仓库确实会改这种默认值的表达方式。

**组件三：上限与降级。** 一个帧上能挂的编译产物数与跨所有 code object 的总数各有一个上限。**量级（引用时必须带版本）**：

| 常量 | torch 2.6.0 | torch 2.7.1 | torch 2.13.0 |
|---|---|---|---|
| 单帧上限 | `cache_size_limit = 8` | `recompile_limit = 8` | `recompile_limit = 8` |
| 全局上限 | `accumulated_cache_size_limit = 256` | `accumulated_recompile_limit = 256` | `accumulated_recompile_limit = 256` |

**改名发生在 2.6.0 与 2.7.1 之间**：旧名保留成了 alias（2.13.0 的 `config.py:137-140`）。触顶后的行为见 §4。

**组件四：诊断入口。**

| 入口 | 用途 |
|---|---|
| `TORCH_LOGS=recompiles` | 打印每一次重编译的**原因** |
| `TORCH_LOGS=guards` | 打印 guard 失败 |
| `TORCH_LOGS=dynamic` | 排查**过特化**（本可以泛化的维度被写死成了常量） |
| `torch._dynamo.utils.compile_times()` | 各阶段编译耗时 |
| `fail_on_recompile_limit_hit` | 把「触顶降级」变成硬报错（默认 `False`） |
| graph break 编号 **GB0039** | 字符串是 `Dynamo recompile limit exceeded`，用来在日志里定位这一类事件 |

`fail_on_recompile_limit_hit` 的源码注释明确写着它与 `suppress_errors` **互斥**（`torch/_dynamo/eval_frame.py:1663-1665`）。

## 3. 实现与演进

**第一代：全静态。** 每个不同的形状各编一份。代价随形状种类线性增长，一个输入尺寸会抖动的训练循环很快就撞上限。

**第二代：`automatic_dynamic_shapes`（当前的默认）。** 让系统**自己发现**哪些维在变，并把它们标成 dynamic。这一代的收益就是本文的数字：N 个不同 shape 的重编译增量从 N 掉到「变化的维数」。

**第三代：把控制权交给用户——`mark_dynamic`。** 因为第二代的路子是「先静态编一次、失败后再动态」，所以**每个动态维至少要付一次多余编译**。用户如果能提前声明，就能跳过那一次。

**官方对 `dynamic=True` 的态度（方向必须写准）**：

| 写法 | 官方定位 |
|---|---|
| `dynamic=None`（**默认**） | 自动检测 dynamism，重编译时生成更动态的 kernel |
| `dynamic=False` | 永不生成动态 kernel，永远做特化 |
| `dynamic=True` | **"(Not recommended)"**、"error prone"，归在 **"for testing only"** |
| **用户标注**（`torch._dynamo.mark_dynamic`） | **推荐做法** |

**这一段最容易写反**：把 `dynamic=True` 当成「打开动态形状的开关」、把默认当成 `False`，都是错的。`dynamic=True` 的语义是「**一上来就把一切都做成动态的**」，官方认为它容易出错；推荐的路径是**自动检测（默认）**或**用户逐维标注**。

`mark_dynamic` 的用法要点（官方文档）：要在**把张量传进编译函数之前**标注（不是在 `forward` 里面标），可以带 `min` / `max` 范围。它还有一个第二代没有的能力：**标注那些不会被自动标出来的东西**，比如 `nn.Module` 的参数。

## 4. 踩坑与失败模式

**失败模式 1：触顶之后静默降级，不再尝试编译。**

- **现象**：某个函数跑着跑着突然变慢，日志里出现「hit config.recompile_limit (8)」之类的一行，然后……**程序照常跑**。之后无论输入怎么变，它都不再变快也不再报错。
- **成因**：上限是有的，而且触顶后的动作是**降级而不是报错**——源码的行为是把该帧的执行策略设成 `FrameAction.RUN_ONLY`（直接跑，不再尝试编译），并抛一个 `Unsupported`，其 `gb_type` 是 **`Dynamo recompile limit exceeded`**（graph break 编号 **GB0039**）。**默认不会让程序失败**：`fail_on_recompile_limit_hit` 默认 `False`。
- **规避**：把这条当成**必须监控**的事件，而不是当成正常运行。先打开 `TORCH_LOGS=recompiles` 看**重编译的原因**（是形状在抖，还是常量在变），再决定是固定形状、标注 dynamic，还是干脆提高上限。
  > **⚠ 引用这几个常量时必须带 torch 版本**：单帧上限在各版本分别是 `cache_size_limit = 8`（2.6.0）与 `recompile_limit = 8`（2.7.1 / 2.13.0），全局上限是 `256`。**光写「默认是 8」而不写版本，在本主题里视为不合格**（见 `code/specs.py` 的 `VERSION_BOUND` 与 `RECOMPILE_LIMITS`）。
  > **⚠ 一条不得写进来的机制**：「触顶后**嵌套调用**也会被跳过」这条**无法判定方向**——控制它的开关 `skip_code_recursive_on_recompile_limit_hit` 在 torch 2.7.1 上还是 `True`（生效），到 2.13.0 已被标成 `deprecated=True, deprecation_message="does not do anything"`。**既然方向判不出来，本仓库就不写它**（记录在 `code/specs.py` 的 `UNSOURCED`）。这一条恰恰是最容易被人记住的那种「反直觉细节」，所以更要克制。

**失败模式 2：改学习率就重编译，而它与形状无关。**

- **现象**：形状全程固定，编译次数却随训练步数增长。每一次 LR schedule 变化都触发一次编译。
- **成因**：Python 的 `int` / `float` 变量会被 guard 成**常量的精确值**（guard 内容里的 "constant values"）。LR scheduler 的 `lr` 是经典例子；`nn.Module` 的整型属性同理。**这些值被写进了编译产物**，所以值一变，产物就作废。
- **规避**：把「会被 guard 的常量」与「输入形状」当成**两类独立的重编译源**分别处理。本仓库的可跑对照是：形状一直是 `(4,3)`，只改 `lr` 就触发 **4 次**编译（见 §5）；诊断入口是 `TORCH_LOGS=recompiles`——它会直接告诉你「上一次是因为什么重编的」。

**失败模式 3：用 `dynamic=True` 去「打开动态形状」。**

- **现象**：为了处理形状抖动，写成 `torch.compile(model, dynamic=True)`，结果官方文档里它被标注 "error prone"，模型反而更慢或更容易出错。
- **成因**：`dynamic=True` 的语义是**一上来就把一切都做成动态的**，官方定位在 "for testing only"。而**默认的 `dynamic=None` 本来就会在重编译时自动转动态**——也就是说，那个「打开动态形状」的需求**默认就已经满足了**。
- **规避**：默认值就够用；需要提前声明时用 **`mark_dynamic`** 逐维标注（官方推荐）。**不要**把 `dynamic=True` 当成推荐做法，**也不要**把默认写成 `dynamic=False`。

## 5. 实测数据

**测量条件**：`code/04_recompile_count.py` 在本机 `python3 3.13.5` + `numpy 2.4.4` 上跑出的结果，`cd code && make run` 全绿（本篇 18 条断言：A 3 / B 8 / C 7）。

> **★ 限定语（本域强制）**：下面的数字全部来自**本仓库几百行的迷你编译缓存**，**不是 torch._dynamo 的 cache 与 guards**。默认值（`dynamic=None`、`automatic_dynamic_shapes=True`、`recompile_limit=8`、`accumulated_recompile_limit=256`）逐条抄自固定版本的源码（见 `code/specs.py`，**每条都带版本**），但**实现形态是重写的**：本仓库把「编译产物」建模成一张迷你图，把「形状特化」建模成按编译时的形状 reshape。这些数字支持的是「重编译次数随什么增长」这类**机制**结论，**不支持任何关于 torch.compile 实际重编译次数的结论**。

**地形**：`model(x, w, m, lr) = ((x*w).relu() @ m) * lr`，只变第 0 维的 6 个 shape（N = 6）。

| 项 | 本机实测值 |
|---|---|
| 同一个 shape 重复调用 20 次的**编译增量**（默认路径） | **0** |
| N = 6 个不同 shape 的**编译增量**（默认路径） | **1**（总编译 2 次） |
| N = 6 个不同 shape 的编译次数（**关掉**自动动态形状） | **6** |
| 自动动态形状省下的编译次数 | **5**（6 − 1） |
| 喂 14 个 shape 时的编译次数（`recompile_limit` = 8） | **8** |
| 喂 14 个 shape 时的 **eager 兜底**次数 | **6** |
| 只改 `lr`（形状一直是 `(4,3)`）触发的编译次数 | **4** |

**默认路径的两张缓存条目**（本机实测的中间过程）：

```text
#1 图 4 个算子，静态形状 ((4,3), (4,3), (3,2))，dynamic 维 无        ，命中 20 次
#2 图 4 个算子，静态形状 ((8,3), (8,3), (3,2))，dynamic 维 [(0,0),(1,0)]，命中 4 次
```

第 2 条的 `dynamic 维 [(0,0),(1,0)]` 就是「**自动把变化的维度标为 dynamic**」这件事的直接物证：`arg0` 与 `arg1` 的第 0 维被标掉了，此后 `(16,3)`、`(32,3)`、`(64,3)`、`(128,3)` 全部命中它。

**触顶之后的去向**（14 个不同 shape，上限 8）：**8 次编译 + 6 次 eager 兜底 = 14 次调用**。每一次调用要么编译、要么走 eager，**没有第三种去向**——这就是「静默降级」在计数上的样子。

**A 类的独立参照路径**：编译产物的执行结果与纯 numpy 参考实现逐元素比对，最大绝对误差 **0.000e+00**（两个不同形状各一次）。

**C 类的失败行为**（每条都真的发生）：

| 构造 | 结果 |
|---|---|
| 同一个 shape 重复调用 20 次 | 编译增量 **0**（不是「涨得慢」，是**完全不涨**） |
| N = 6 个不同 shape（默认路径） | 编译增量 **1**，**远小于 N** |
| N = 6 个不同 shape（关掉自动动态形状） | 编译次数 **= N**（这一组**与真实默认不符**，见下） |
| 喂超过上限的 shape 数 | 编译次数**停在 8**，其余走 eager |
| 只改 `lr` | 触发 4 次编译，而形状组数只有 **1** 组 |
| 可证伪性反证：把 guard 换成「永远通过」 | 形状特化的图被拿去算错形状 → 抛出 `ValueError: 编译产物是形状特化的…`，断言立刻失败 |

> **⚠ 一条必须自我声明的口径**：上表里「关掉自动动态形状」的那一行**不是真实默认行为**。官方默认是 `automatic_dynamic_shapes = True`。这一组是**反例专用**——它的作用是量化「自动动态形状省掉了什么」（省下 5 次编译）。计划 §0.7.A 明确纠正过「默认是 `dynamic=False`」这个记忆错误，**如果文档把这一组写成默认行为，就等于把那个错误又写回去了**。

**第一版被实跑推翻的三项**：

1. **「中间结果是点式、输出是向量」的地形一开始算不出矩阵乘**（秩对不上直接抛异常）。地形得重设计——这是「先跑再写」里最典型的一类：地形写在纸上看起来没问题。
2. **「触顶」那条一开始根本触不到顶**：默认路径的自动动态形状把 14 个形状并成了两张图，于是编译次数是 2 而不是 8。修法是**显式关掉自动动态形状**并**在正文与程序输出里都声明这是对照而不是默认**。
3. **可证伪性反证一开始是「通过」的**，因为第一版地形是纯广播的点式链——缓存图拿到错形状**碰巧还能算对**（numpy 会广播）。改成含 `matmul` 的地形、并把「编译产物形状特化」建模出来之后，坏 guard 才真的报错。**这条差点变成一条只演示、不检查的断言。**

## 6. 小结与延伸

- **`dynamic` 默认 `None`**，**`automatic_dynamic_shapes` 默认 `True`**：先静态编译，guard 失败后把变化的维度自动转 dynamic 再编一次，此后命中同一张动态图。**重编译次数的期望是 O(变化的维度数)，不是 O(不同形状数)。**
- **`dynamic=True` 是 "(Not recommended)"、"error prone"、"for testing only"**；推荐做法是**默认的自动检测**或 **`mark_dynamic` 用户标注**。
- **guard 的内容是 constant values, types, and object IDs**；语义是「guard 对**所有**已编译实例都失败」才重编译。
- **上限与降级**：`recompile_limit = 8` / `accumulated_recompile_limit = 256`（2.7.1 与 2.13.0；2.6.0 里叫 `cache_size_limit` / `accumulated_cache_size_limit`）。触顶后该帧改为直接跑（`FrameAction.RUN_ONLY`），默认**不报错**；编号 GB0039 = `Dynamo recompile limit exceeded`。
- **`int` / `float` 会被 guard 成常量的精确值**（LR scheduler 的 `lr` 是经典例子），这是一类与形状无关的重编译源。

**延伸**：

- guard 失效之后，被重编的图会重新走一遍融合与 codegen → [02-fusion-scheduling](02-fusion-scheduling.md) / [03-codegen](03-codegen.md)
- 图在什么位置被切断（切图与 guard 失败是两件事） → [01-graph-capture](01-graph-capture.md)

## 编译与运行

```bash
cd 60-compilers/inductor/code
make 04_recompile_count     # 只跑本篇
make run                    # 跑全部四篇
```

程序：`code/04_recompile_count.py`（配套 `code/minigraph.py` 的迷你捕获，外部事实源 `code/specs.py`）。

## 参考

- `dynamic` 参数的默认值与语义：torch 2.13.0 源码 `torch/__init__.py`（`torch.compile` 的 docstring）
- `recompile_limit` 的默认值 8 与「fall back to eager」：同上
- `automatic_dynamic_shapes`：torch 2.6.0 `torch/_dynamo/config.py:87` / 2.7.1 `:110` / 2.13.0 `:185`
- `recompile_limit` 与 `accumulated_recompile_limit` 的**逐版本**默认值与旧名 alias：`code/specs.py` 的 `RECOMPILE_LIMITS`
- 触顶后的行为与 `FrameAction.RUN_ONLY`：torch 2.13.0 `torch/_dynamo/convert_frame.py:2007-2039`
- GB0039：torch 2.13.0 `torch/_dynamo/graph_break_registry.json`
- `fail_on_recompile_limit_hit` 与 `suppress_errors` 互斥：torch 2.13.0 `torch/_dynamo/eval_frame.py:1663-1665`
- `dynamic=True` 的 "(Not recommended)" / "error prone" 与 `mark_dynamic` 的推荐：官方文档 torch.compiler 的动态形状一节 `https://docs.pytorch.org/docs/main/torch.compiler`
- 以上每条都在 `code/specs.py` 里逐条对应，取用日期 2026-09-16

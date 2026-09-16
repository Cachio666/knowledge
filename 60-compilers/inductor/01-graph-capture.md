---
title: 图捕获与 graph break
slug: inductor-graph-capture
domain: 60-compilers
topic: inductor
level: 2
difficulty: advanced
prerequisites:
  - 60-compilers/mlir/02-tensor-vs-memref
tags: [inductor, torchdynamo, torch-compile, graph-break, pep523, frame-evaluation]
related:
  - 60-compilers/inductor/02-fusion-scheduling
  - 40-kernels/elementwise-fusion/02-fusion-and-traffic
  - 70-frameworks/pytorch
code:
  - code/01_graph_capture.py
status: stable
updated: 2026-09-16
---

# 图捕获与 graph break

> TorchDynamo 用一个 CPython 的帧求值回调（PEP 523）在**每个函数帧进入时**做一次决策：这个帧要不要被编译。捕获中途遇到做不到的事就在那里**把图切断**——但切出来的只是**那一段**区域改为跑普通 Python，**前后两段仍然各自被编译成图**。把 graph break 理解成「整段退回 eager」是本主题最贵的一个误解。

## 0. 背景与动机

先给一个具体的对照场景。

```python
def step(x, w):
    a = (x * w).relu()          # 想做成一件事：一次遍历算完
    s = a.sum(axis=1)
    if s.item() > 0.0:          # 这一行是问题所在
        return a + w
    return a - w
```

第二行与第三行本来可以进同一张图——它们之间没有控制依赖，编译器可以把 `mul`、`relu` 融成一个核，中间那个 `a` 一次都不落内存。第四行把 `s` 变成一个 **Python 的 float**，于是控制流依赖了一个**张量的值**：图必须在这里停下，因为「接下来走哪个分支」在编译期不知道。

Dynamo 的处理是**把图切断**：前三行编译成一张图，`if` 与它管着的那一段用普通 Python 执行，`return` 之前再把后面的算子收进第二张图。**三段**，不是「整个函数退回 eager」。

### 主线（本主题的首篇在这里完整展开）

本域的统一主线是**物化还是重算**：

> **「物化」= 为中间结果分配独立的存储。** 它不指存储层级——在寄存器、共享内存还是全局内存里，都算物化；「重算」= **根本不分配存储**，把计算直接替换进消费者的表达式。

在 Inductor 这条线上，这个决策的名字叫 **fusion group 的划分**：若干个算子被划进同一个融合组，组内的中间结果就**不落内存**（重算/留在寄存器里）；跨组的那条边上的结果就必须**落内存**（物化）。所以「融不融」与「物化不物化」在这个系统里是**同一个问题的两种说法**。

本主题要回答上游那个问题（[40-kernels/elementwise-fusion/02](../../40-kernels/elementwise-fusion/02-fusion-and-traffic.md) 已经推出「融合的收益上界是链长 `k`，且只在中间结果往返 HBM 时成立」）：**编译器靠什么判据决定融不融，以及为什么它经常选择不融。**

| 本文 | 与主线的关系 |
|---|---|
| 01 图捕获（本篇） | **本文不涉及主线**——见下 |
| [02 融合与调度](02-fusion-scheduling.md) | **是主线的判据来源**：融合组怎么划 |
| [03 codegen](03-codegen.md) | **是主线的代价项**：融合组最终变成什么代码 |
| [04 动态形状](04-dynamic-shapes.md) | **本文不涉及主线**——guard 与重编译决定的是「这张图要不要重做一次」，物化决策仍在调度阶段做出（理由见该篇 §0） |

**本篇为什么不在主线上**（选③，必须写明理由）：图捕获决定的是「**这张图存不存在、有几张**」，而物化决策发生在它**之后**的调度阶段。一个帧被切成两张子图，切出来的是两张待调度的图，不是两条物化决策。切图会**改变**调度阶段的输入（两张小图而不是一张大图，各自能融的链长就短了），但它自己不做任何物化决策。

这条边界的实际后果，就是本文 §1 要推的那件事：**捕获的粒度决定了后面能融多少**——而这件事发生在下一篇。

## 1. 原理

### 第一步：捕获点装在什么位置

官方对 TorchDynamo 的定位原文是：

> TorchDynamo (`torch._dynamo`) —— the tracer that uses a CPython feature called the **Frame Evaluation API** to safely capture PyTorch graphs.

这个"Frame Evaluation API"就是 **PEP 523**。它给 CPython 加了一个钩子：解释器在**执行一个函数帧之前**，先调用一个由宿主注册的回调，由回调决定这个帧怎么执行。Dynamo 装上去的那个函数指针叫 **`_PyInterpreterState_SetEvalFrameFunc`**（确切名字见 `code/specs.py` 的 `PEP523_SYMBOL`，对应 torch 2.13.0 的 `torch/csrc/dynamo/debug_macros.h:64`）。

**这里要纠正两个很常见的错**：

| 常见说法 | 事实 |
|---|---|
| 「Dynamo 用 `sys.settrace`」 | **不是**。`sys.settrace` 是**行级**事件，粒度与语义都不同；PEP 523 是**帧级**的求值回调 |
| 「Dynamo 用 `sys.monitoring`」 | **不是**，而且 `sys.monitoring` 与 torch.compile **已知不兼容**：注册 monitoring 回调会引发 graph break，pdb/fdb 在 3.12+ 不能与 torch.compile 共用 |

为什么一定要在帧入口？因为**决策点必须在执行之前**。行级钩子是「执行到这一行时告诉我」，那时前几行已经跑掉了，来不及换成编译版本；帧入口是「这个帧还没开始，你决定怎么跑」。这个差别决定了能不能把整个帧换成一个编译产物。

**本仓库的迷你追踪器用的是 `sys.settrace` 的 `call` 事件**——那是标准库里唯一能拿到「帧入口」的钩子，只能模拟决策的**时机**，模拟不了 PEP 523 的实现层次。**这是一个刻意的近似，不是等价物**：Dynamo 是在 C 层替换帧的求值函数，我们是在 Python 层的 trace 钩子里记一笔（见 `code/minigraph.py` 文件头）。

### 第二步：帧入口处的三个去向

回调被调用时，帧完整的字节码、参数、globals 都还只是「一份要执行的代码」。可能的结果只有三种：

1. **跳过**（`SKIP`）：这个帧不编译，正常跑。
2. **命中缓存**：这个 code object 之前编过，且 **guard** 仍然成立——直接用那份编译产物。
3. **编译**：跑一遍字节码分析，建出一张图。

配套的两件事：

- **缓存挂在 `code object` 上，不是帧上。** 官方原话：`compile caches are per *code object*, not frame`。同一个函数被调用一万次，共享同一份缓存；动态地造出一万个长得一样的函数，它们也共享。这条决定了「缓存是按函数身份算的」。
- **guard** 是编译产物生效的前提。官方对 guard 内容的措辞是 **constant values, types, and object IDs**——既有形状/类型，也有**常量的精确值**。它与本文只擦边；`int`/`float` 被当作常量 guard 这件事的后果在 [04-dynamic-shapes](04-dynamic-shapes.md) §4 展开。

### 第三步：切图之后，图是怎么接上的

这是「不是整段退回 eager」这句话的机制部分。

一次捕获是这样的：拿当前的输入执行一遍函数，边执行边记录算子。走到做不到的位置（官方叫 graph break）时就**结束当前子图**，把控制交回普通 Python；等控制流重新回到可编译的算子序列上，就**开始下一张子图**。

这里有个必须解释的问题：第二张子图的输入是什么？答案是**断点处仍然存活的值**。Dynamo 在切图点生成一段 **resume function**（续跑函数），它的入参就是那些存活的值；从那里继续捕获。

**本仓库用重放模拟这件事**：切图后重跑整个函数，前 k 个算子以普通 Python 执行、从第 k+1 个起重新记录；那些「跑 eager 时算出来的中间值」在重新开始记录时被补成新图的输入占位符。这不是 Dynamo 的做法（它有字节码层，可以真的从断点续跑），但它把同一件事**做成真的**：`make run` 里子图数真的是 2，中间那段真的以普通 Python 执行。

### 第四步：切图触发条件

官方列举的触发条件（**这是列举，不是穷举表**）：

- 数据依赖 `if` 分支 / 数据依赖控制流（含 `.item()`、`.data_ptr`）；
- 大量 Python 内建函数；C 函数；不支持的操作；
- printing / logging / warnings；
- 本身就会报错的代码。

**还有一条要专门标出来：官方把 graph break 定位成特性而不是纯失败**——原话是 "graph breaks are a feature"。理由很直白：一个带图断点的程序能跑，只是跑得不够快；一个拒绝编译的程序直接不可用。

### 第五步：一份**必须否掉**的直觉

「有副作用就 break」——**这是错的**。

- Dynamo 能处理**大量原地修改与 mutation**：`x.add_(y)`、`x.copy_(y)` 这类会被正常记录，不切图。
- 官方甚至给了 `reorderable_logging_functions` 这个开关：把 print/logging 登记进去，Dynamo 会把它们**重排**到不影响捕获的位置，从而**避免**这个 cut。

也就是说，对 print/logging 这类调用，官方的默认态度是「换个位置放」而不是「切图」；而对真正的原地写，官方的默认态度是「能记就记」。

那什么样的「副作用」**真的**会切图？**它的结果影响控制流的时候**。`.item()` 之所以切，不是因为它有副作用，而是因为它把一个张量变成了一个 Python 值，而这个值接着被用来决定走哪条路。区分标准是**数据依赖控制流**，不是「有没有副作用」。

## 2. 执行映射

这一节说的是上面这些概念在真实系统里落在哪些**具体的部件**上。

**部件一：CPython 的 frame object 与 `_PyInterpreterState_SetEvalFrameFunc`。** 每次 Python 调用一个函数，CPython 造一个 frame 对象，里面装着字节码、局部变量、globals。PEP 523 允许宿主替换「怎么执行这个 frame」这个函数指针；Dynamo 的 C 扩展 `torch/csrc/dynamo/eval_frame.c` 做的就是这件事。**量级**：这个回调是**每次函数调用**一次，不是每行一次——所以在热循环里，Dynamo 的开销与「调用了多少个函数」成正比，而不是与「执行了多少条字节码」成正比。

**部件二：字节码分析器与变量包装。** 决策为「编译」之后，Dynamo 取出字节码逐条模拟执行，把每个 Python 对象包成 `VariableTracker` 的子类。这一步的产物就是一张 FX 图。**这是它能够「边捕获边求值」的原因**——被包装的对象同时知道自己「在真实计算里是什么」。

**部件三：挂在 code object 上的缓存与 guard。** guard 是一条布尔假设的集合，编译产物只有在它成立时才能复用。**量级**：一个帧最多能挂多少份编译产物是有上限的——`recompile_limit` **默认 8**（torch 2.13.0，`torch/_dynamo/config.py:121`；该常量在 torch 2.6.0 里叫 `cache_size_limit`，见 `code/specs.py` 的 `RECOMPILE_LIMITS`），跨所有 code object 的另一个上限是 `accumulated_recompile_limit` **默认 256**。触顶后的行为与它的后果在 [04-dynamic-shapes](04-dynamic-shapes.md) §4。

**量级（本机实测）**：本仓库迷你追踪器跑一个 3 算子的小函数，帧入口钩子被调用 **55 次**，其中只有 **1 次**是登记过、要进入捕获的那个函数，其余 54 次是代理与 numpy 内部调用的帧（见 §5）。真实系统不会有这个比例——它的帧入口决策写在 C 里，不会把 numpy 的每一层内部调用都过一遍。

## 3. 实现与演进

**朴素做法：手写 trace 钩子。** 早期做法是 `sys.settrace` 逐行记录。它的问题不是慢（虽然确实慢），而是**语义不对**：行级钩子在「这一行已经要执行」时才被调用，你无法把一个函数整体换成编译版本；而且它记录的是「执行过的行」，不是「这张图的数据依赖」。一个分支没走到，图里就少一块——这正是「跑起来了，但跑的不是你以为的东西」。

**主流实现：PEP 523 帧求值 + guard 缓存。** 这是现在 torch.compile 的路径。它把问题拆成两半：**帧入口决定「编不编」**，**guard 决定「编出来的那份还能不能用」**。两件事分开的好处是 guard 可以做得非常便宜——大部分调用只付 guard 的代价。

**切图点的接续：resume function。** 断点之后要能接着编译，就必须知道「断点处哪些值还活着」。Dynamo 的做法是生成一个续跑函数，把这些值当参数传进去。**这是本文最容易被忽略的一环**：切图不是「丢掉前半段」，而是「前半段产出的一组值成了后半段的输入」。

**诊断设施随之上位。** 这套机制一旦复杂到「为什么会重编译」「为什么这里切图」都答不上来（见 [04-dynamic-shapes](04-dynamic-shapes.md) §4）。

**本仓库的迷你实现怎么落。** `code/minigraph.py` 里：`MiniDynamo._hook` 对应帧入口决策；`P` 是代理张量（算子重载即记录）；`GraphBreak` 异常对应切图信号；`MiniDynamo.capture` 里的重放循环对应 resume function。**它没有字节码层**，因此不能声称复现了 Dynamo 的任何输出。

## 4. 踩坑与失败模式

四条。每条都给出本仓库能真的跑出来的触发方式。

**失败模式 1：把数据依赖控制流放在热路径上。**

- **现象**：每步都调用一次的函数里写着 `if loss.item() > thresh:`。程序能跑，但 `torch.compile` 报出两张以上的子图，而且中间那段以普通 Python 执行——性能与不编译差不多。
- **成因**：`.item()` 把张量变成了 Python 值，控制流因此依赖**运行期**的数据；编译期无法决定走哪条分支，只能在这里把图切断。切断本身不报错（graph break 是特性），所以它**静默**发生。
- **规避**：把这类判断当成**编译边界**来设计，而不是当成一行普通代码。先量出边界在哪（见 §5 的输出形式），再决定是把它挪出热路径，还是接受它并把两侧的子图各自优化好。

**失败模式 2：以为「有副作用就 break」，于是把原地修改改写成函数式。**

- **现象**：为了让编译器「能捕获」，把 `x.add_(y)` 改写成 `x = x + y`，多付一次分配与拷贝；而在某些地方（比如被别处引用的缓冲）改写之后语义还变了。
- **成因**：这条直觉来自「编译器怕副作用」的一般印象。**在这套系统上它是不成立的**：Dynamo 能处理大量原地与 mutation，并且专门提供 `reorderable_logging_functions` 让 print/logging **重排**以避免 cut。真正切图的是「结果被用来决定控制流」，不是「有写操作」。
- **规避**：区分两件事——**写操作**不影响捕获；**用张量的值做分支**才影响。本仓库的迷你实现把这条做成了可跑的对照：同一份代码里的 `add_` 不切图，`.item()` 切图（见 §5 的两个地形）。

**失败模式 3：拿 `sys.monitoring` 或调试器去跟 torch.compile 配合。**

- **现象**：装了 monitoring 回调之后，原本不切图的地方开始切图；或者 pdb/fdb 与 torch.compile 一起用时行为异常。
- **成因**：`sys.monitoring` **不是** Dynamo 用的机制，而且两者**已知不兼容**。这不是「还没适配」，是两套东西都想接管帧的执行方式。
- **规避**：调试用官方给的那一套（`TORCH_LOGS=graph_breaks` 一类），不要自己再装一个帧级钩子。**顺带纠正本文标题下最容易被记住的那个错**：Dynamo 用的是 PEP 523 的帧求值回调，不是 `sys.settrace`，也不是 `sys.monitoring`。

**失败模式 4：把切图当成「编译失败」，于是关掉编译。**

- **现象**：看到输出里有 graph break 就断定 `torch.compile` 没用，直接去掉它。
- **成因**：官方把 graph break 定位成**特性**（原话 "graph breaks are a feature"）。带断点的程序仍然把两侧的子图编译了；只有**被切出来的那一段**是普通 Python。
- **规避**：把指标从「有没有切图」换成「**切出来的 Python 段有多长**」。一张被切成 2 段、中间只有 3 行 Python 的图，与一张完整的图差别很小；被切成 20 段、每段之间夹着 Python 循环的图，才是真的没吃到编译的好处。

## 5. 实测数据

**测量条件**：`code/01_graph_capture.py` 在本机 `python3 3.13.5` + `numpy 2.4.4` 上跑出的结果，`cd code && make run` 全绿（本篇 25 条断言：A 7 / B 13 / C 5）。

> **★ 限定语（本域强制）**：下面每一个数字都是**本仓库几百行的迷你追踪器**跑出来的，**不是 TorchDynamo**。迷你实现借用了真实系统的术语（帧入口决策、graph break、子图）以便对照，**实现机制是重写的**——它用 `sys.settrace` 的 `call` 事件模拟帧入口，用**重放**模拟 resume function。这些数字支持的是「切图会把图分成几段、断点处的值怎么接上」这类**机制**结论，**不支持任何关于 TorchDynamo 的结论**（例如「Dynamo 在这种代码上会切几次」）。

**四个地形**（都是本仓库写的，与真实系统的行为无对应关系）：

| 地形 | 内容 | 结果 |
|---|---|---|
| A | 纯点式链 `x*2 → +w → relu` | 子图 1 张 · 切图 0 次 |
| B | 点式链 + `if s.item() > 0` 分支 | **子图 2 张** · 切图 1 次 · 重放 2 轮 |
| C | 点式链 + `mprint(a)` | 未登记：**子图 2 张**；登记进 `reorderable_logging_functions`：**子图 1 张 + effect 节点 1 个** |
| D | 点式链 + `a.add_(w)`（原地修改） | **子图 1 张**，算子序列 `[arg, arg, mul, add_]` |

**结构计数表**（全部是本机实测值）：

| 项 | 本机实测值 |
|---|---|
| 地形 A · 图输入数 | 2 |
| 地形 A · 被计数的算子数 | 3 |
| 地形 A · 子图数 / 切图次数 | 1 / 0 |
| 地形 B · 子图数 / 切图次数 / 重放轮数 | 2 / 1 / 2 |
| 地形 B · 子图 1 的算子数（不含占位符） | 2 |
| 地形 B · 子图 2 的 mid 占位符数（断点处存活的值） | 1 |
| 地形 C · 未登记 print 的子图数 | 2 |
| 地形 C · 登记 print 后的子图数 / effect 节点数 | 1 / 1 |
| 地形 D · 原地修改的子图数 | 1 |
| 地形 A · 帧入口钩子：放行 / 跳过 | 1 次 / 54 次 |

**A 类的独立参照路径**：地形 A 的图解释器输出与**纯 numpy 写的参考实现**逐元素比对，最大绝对误差 **0.000e+00**；地形 B 的**第二张子图**拿参考实现自己算出的中间值当输入，重新求值后再与参考实现的最终结果比对，最大绝对误差同样是 **0.000e+00**。两条路径互不依赖：一条走记录下来的图，一条走手写的 numpy。

**C 类的失败行为**（每条都真的发生，不是"理论上会"）：

| 构造 | 结果 |
|---|---|
| `.item()` 命中数据依赖控制流 | 图被切成 2 张（断言的是「**被切断**」，不是「没报错」） |
| 未登记的 `print` | 图被切成 2 张 |
| 可证伪性反证：把切图检测换成「不检测」的版本 | 子图数从 2 掉到 **1**，C1 判据立刻失败 |
| 可证伪性反证：把 A 类的参考实现故意写错 | 最大绝对误差从 0 变成 **7.000e+00**，A 类判据立刻失败 |

**第一版被实跑推翻的两项**（先跑再写的直接产物）：

1. **子图 2 的输入一开始是错的。** 第一版让重放时「跑 eager 的那些算子」返回无节点代理，结果第二张子图的算子**把输入丢光了**——图上只剩一个孤立的 `add`。修法是在 `_emit` 里为这些代理补占位符（对应真实系统的 resume function 入参）。这是跑出来的，不是想出来的。
2. **「登记 print 后不切图」这条对照一开始没写在程序里。** 第一版只断言了「未登记的 print 会切图」，那是一条**只演示、不检查**的断言——把 `reorderable_logging_functions` 整个删掉它照样通过。补上对照（登记后子图数必须回到 1）之后，这条才可证伪。

## 6. 小结与延伸

- **捕获发生在帧入口**：PEP 523 的帧求值回调（`_PyInterpreterState_SetEvalFrameFunc`），**不是** `sys.settrace`、**不是** `sys.monitoring`（后者还与 torch.compile 已知不兼容）。
- **切图不是退回 eager**：只有被切出的那段跑普通 Python，前后子图仍各自被编译。断点处存活的值成为下一张子图的输入（真实系统的 resume function）。
- **切图由「数据依赖控制流」触发**，不由「有副作用」触发：mutation 能被记录；print/logging 能通过 `reorderable_logging_functions` 重排以避免 cut。
- **graph break 是特性**（官方原话），指标应当是「切出来的 Python 段有多长」，不是「有没有切」。
- **缓存挂在 code object 上**，guard 的内容是 constant values / types / object IDs。

**延伸**：

- 切出来的那几张图，接下来怎么决定融不融、为什么常常选择不融 → [02-fusion-scheduling](02-fusion-scheduling.md)
- guard 失败之后会发生什么（自动动态形状、重编译上限） → [04-dynamic-shapes](04-dynamic-shapes.md)
- 融合的收益上界为什么是链长 → [40-kernels/elementwise-fusion/02](../../40-kernels/elementwise-fusion/02-fusion-and-traffic.md)

## 编译与运行

```bash
cd 60-compilers/inductor/code
make 01_graph_capture      # 只跑本篇
make run                   # 跑全部四篇
```

程序：`code/01_graph_capture.py`（配套本仓库迷你图与追踪器 `code/minigraph.py`，外部事实源 `code/specs.py`）。

## 参考

- TorchDynamo 与 Frame Evaluation API 的官方定位：`https://docs.pytorch.org/docs/main/torch.compiler`
- `torch.compile` 的 docstring（`dynamic` 参数、缓存按 code object、`recompile_limit` 默认 8）：torch 2.13.0 源码 `torch/__init__.py`
- `_PyInterpreterState_SetEvalFrameFunc` 的落点：torch 2.13.0 源码 `torch/csrc/dynamo/debug_macros.h:64`
- `reorderable_logging_functions` 的默认值：torch 2.6.0 / 2.7.1 / 2.13.0 的 `torch/_dynamo/config.py`
- `recompile_limit` / `accumulated_recompile_limit` 的默认值与**改名版本边界**：`code/specs.py` 的 `RECOMPILE_LIMITS`（逐版本给出）
- 以上每条都在 `code/specs.py` 里逐条对应，取用日期 2026-09-16

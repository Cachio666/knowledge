---
title: 调度原语与物化位置
slug: tvm-schedule-primitives
domain: 60-compilers
topic: tvm
level: 3
difficulty: advanced
prerequisites:
  - 60-compilers/tvm/01-compute-and-schedule
tags: [tvm, s-tir, schedule, compute-at, compute-inline, reverse-compute-at, materialization, v0.26.0]
related:
  - 60-compilers/tvm/01-compute-and-schedule
  - 60-compilers/tvm/03-native-track-and-tile-primitives
code:
  - code/02_schedule_primitives.py
status: stable
updated: 2026-09-16
---

# 调度原语与物化位置

> `s_tir.Schedule`（**v0.26.0**）里管「放到哪儿」的原语只有两个方向：`compute_at`（**物化**，缩到某个循环内）与 `compute_inline`（**不物化**）。`compute_root` 不在 API 里——没被搬走的 block 本来就在根位置。两个方向都是**过程式**的：先搬谁后搬谁，会影响另一条原语能不能调用。

## 0. 背景与动机

本文是主线的**判据来源**：上一篇文章（[01-compute-and-schedule](01-compute-and-schedule.md)）说「默认全物化」，本文回答**要把它搬到哪儿去、搬了之后代价怎么变**。

具体场景：同一条五级链（`a,b → s1 → … → s5 → out`，每级 4096 个元素）。三种放法：

```text
默认（根作用域）      s1..s5 各占一整级 → 峰值 81920 B，流量 40960 个元素
compute_at(out.i)     s1..s5 缩到一次 i 迭代的切片 → 峰值 10240 B，流量 40960
compute_inline        s1..s5 不占存储 → 峰值 0，流量 0
```

（上面三个数由本仓库的迷你 Schedule/Plan/Interp 跑出，见 §5 的限定语。）

**注意中间那一行**：字节数降了 8 倍，而**流量一个元素都没省**。把它当成「融合省了内存往返」是错的——省的是**同时存活的量**，不是往返次数。这两个量在这里第一次被分开，后面 §4 的失败模式都挂在这条区分上。

## 1. 原理

### 1.1 原语的语义：作用在循环嵌套上

调度原语的输入不是「一个算子」，而是**循环嵌套里的一个位置**。以 `compute_at` 为例：

```python
sch = tvm.s_tir.Schedule(mod)         # mod 是装着这个 prim_func 的 IRModule
block = sch.get_sblock("s5")          # ★ 不是 get_block
loop = sch.get_loops(sch.get_sblock("out"))[0]
sch.compute_at(block, loop)           # 把 s5 物化到 out 的这条循环内
```

两条原语的**前置条件**（来自 `schedule.py` 的 docstring）：

| 原语 | 前置条件 |
|---|---|
| `compute_at` | block 的**所有 consumer** 都必须在目标 loop 下 |
| `reverse_compute_at` | block 的**所有 producer** 都必须在目标 loop 下 |

这条「所有 consumer 都在 loop 下」不是形式主义——**多消费者地形上它会直接把原语挡死**：本机实测，一个中间结果被两个输出同时消费时，对其中一个消费者的循环调 `compute_at` 会被拒绝（`ScheduleFailure`），而 `compute_inline` **可以**（它没有这条前提）。见 §5 C1。

### 1.2 三态的整数对照

设 `k=5` 级、每级 `N=4096` 个元素（`8×512`）、元素 4 字节、被切的轴 `i` 的 extent 是 8：

| 放置 | 中间 buffer 数 | 峰值中间存储 | 中间流量 | 全 stage 求值 |
|---|---|---|---|---|
| 默认（根作用域） | `k` = 5 | `k·N·esize` = 81920 B | `2·k·N` = 40960 | `(k+1)·N` = 24576 |
| `compute_at(i)` | `k` = 5 | `k·(N/8)·esize` = 10240 B | `2·k·N` = 40960 | `(k+1)·N` = 24576 |
| `compute_inline` | **0** | **0** | **0** | `(k+1)·N` = 24576 |

三行要分开读：

1. **物化次数的分界**：默认与 `compute_at` 都是 5，`compute_inline` 是 0。**这是三态里唯一真正二分的量**——`compute_at` 站在「物化」这一侧。
2. **作用域的分界**：默认的峰值是 `k·N`，`compute_at` 是 `k·(N/8)`，**比值恰好等于被切轴的 extent**（本机实测 `20480 // 2560 = 8`）。这个量必须由程序算出来，写成 `O(N)` 就失去可证伪性。
3. **重算的分界**：单消费者链上，`compute_inline` 的求值次数与默认**完全一样**（都是 24576）——**「不物化」在单消费者链上确实免费**。它变贵要到消费者多于一个时（§1.3）。

### 1.3 「不物化」不是免费的：消费者个数是那个系数

把地形换成**多消费者**：`s = a*b` 同时被 `out1 = 2s` 与 `out2 = s+1` 消费。

| 放置 | `s` 被求值几次 | 总求值次数 |
|---|---|---|
| 物化 `s` | 1 | 12288 |
| `compute_inline` `s` | **2 = 消费者数** | **16384** |

推导：`times(stage) = 1`（物化）或 `Σ times(消费者)`（inline）。链上每个 stage 只有一个消费者，所以 inline 之后 `times` 一路保持 1；一旦某个 stage 被两个消费者读，它的 `times` 就变成 2，而**它上游的每一个 stage 也跟着翻倍**。

多算的量是可数的：`(消费者数 − 1) × 每级元素数`（本机实测 `(2−1)×4096 = 4096`，总求值从 12288 涨到 16384）。

### 1.4 混合：物化与重算在同一份调度里

三个状态不是单选。本机实测一个混合态：**只把 `s1` 留在默认（物化）、`s2`…`s5` 全部 inline** → 中间 buffer 个数 **1**。真实编译器里的调度几乎都是这种混合形态，所以「某篇文档说 TVM 默认物化」不等于「TVM 不支持重算」。

### 1.5 过程式：原语的**次序**会影响结果

`compute_at` 的前置条件看的是**当前**的放置状态，不是数据依赖图上的静态可达性。要在一条链上把所有中间结果搬进输出循环，**必须从后往前搬**：

```text
先 compute_at(s5, out.i)   → 现在 s5 在 out 的 i 循环内
再 compute_at(s4, out.i)   → s4 的消费者 s5 已经在 loop 下 → 放行
…
最后 compute_at(s1, out.i)
```

反过来做（先搬 `s1`）会被拒绝：`s1` 的消费者 `s2` 此时还在根位置。官方对这件事的原话是：

> the transformation process is procedural; thus, inadvertent execution of the block twice will yield an error stating the non-existence of variable `j`.

**这不是本仓库迷你实现的怪癖，是真实 `s_tir` 的性质**——本仓库的实现是有意复现它的（见 §5 被推翻的第一条）。

## 2. 执行映射

**位置**：调度层夹在「计算定义」与「编译」之间。`tvm.s_tir.Schedule` 作用在 `s_tir=True` 的 `prim_func` 上；DLight 与 MetaSchedule 就是**自动产生这些原语调用**的两个组件（[04-auto-scheduling](04-auto-scheduling.md)）。

**具体组件与量级**：

| 组件 | 在这里做什么 | 一个数值量级 |
|---|---|---|
| `s_tir.Schedule` | 承载原语；每个 stage 有一个放置状态 | 本机实测：峰值 81920 B → 10240 B（8 倍） |
| `Schedule` 的构造参数 `enable_check` | 默认 `True`，做调度合法性检查 | 见 §4 失败模式 2 |
| `PassContext` | 承载编译期开关（如 `tir.disable_*`） | 见 [03](03-native-track-and-tile-primitives.md) §4 的可选观察 |
| `get_sblock` / `get_loops` | 取 block 与循环的引用 | 一条 `k=5` 的链有 5 个可搬的 block |

**`enable_check=True` 是默认值，而官方自己说它「太严」**：

> The default schedule checks are too strict and might prevent us performing some valid schedules. ... It's user duty to guarantee schedule correctness if `enable_check` is set to `False`.

注意后半句的方向：关掉检查**不是**「解锁更多优化」，而是**把正确性责任从编译器转移到用户**。

## 3. 实现与演进

**TE 时代（已删除）：三态 + 不同参数。** `compute_at(parent_stage, scope_ivar)`——传的是**父 stage 与一个 IterVar**；想放回根位置有 `compute_root`。

**`s_tir`（v0.26.0）：两态 + 换参数。** 变化逐条：

| | TE（已删除） | `s_tir`（v0.26.0） |
|---|---|---|
| 三态 | `compute_root` / `compute_at` / `compute_inline` | **没有 `compute_root`**，只剩后两个 |
| `compute_at` 参数 | `(parent_stage, scope_ivar)` | `(block, loop, preserve_unit_loops=False, index=-1)` |
| 反向原语 | 无 | `reverse_compute_at` / `reverse_compute_inline` |

**为什么会少掉一态？** 因为「根位置」不再是需要主动选择的东西——它是**默认状态**（[01](01-compute-and-schedule.md) §1.3）。TE 时代 `compute_root` 存在的背景是：`te.compute` 出来的 stage 默认要显式声明「我算在根上」，而 `s_tir` 的 block 生成出来就在根上。

**`index` 参数是后加的**：`-1` 表示插到最后一个可插点，`-2` 表示第一个可插点。它的用途是当某个 block 有多个生产者时，控制它们在循环体内的先后——**这本身就是过程式性质的产物**：如果位置是纯函数式的，就不需要「插到第几个」。

## 4. 踩坑与失败模式

三条，**全部逐条带来源**，且**都不是本仓库程序的断言**——本仓库跑不了真实 TVM，把真实 API 的行为写成程序断言就是自证（计划 Chunk 2 第二条硬性要求）。

**失败模式 1：把调度当成纯函数式的变换，重复调用同一个原语。**

- **现象**：把同一段调度代码跑两遍（或对同一个 block 调两次 `compute_at`），第二次报「变量不存在」这类莫名其妙的错。
- **成因**：官方对这件事的原话是「the transformation process is **procedural**; thus, inadvertent execution of the block twice will yield an error stating the non-existence of variable `j`」。变换是**就地改写**的，第二次调用看到的已经不是第一次的 IR。
- **规避**：**调度脚本当成一次性脚本写**——`Schedule(func)` 与所有原语调用在同一个函数体内顺序执行完，不做「增量补几条原语」。同理，链上多个 block 要按**从消费者到生产者**的次序搬（本机实测的次序依赖见 §5 第一版被推翻的第一条）。

**失败模式 2：以为关掉 `enable_check` 会解锁更多优化。**

- **现象**：调度被检查挡住，于是设 `enable_check=False` 让它过去。
- **成因**：官方说默认检查「too strict and might prevent us performing some valid schedules」——听起来像是检查在阻碍你；但同一句话的后半段是「**It's user duty to guarantee schedule correctness** if `enable_check` is set to `False`」。检查拦下的东西里确实有合法的，也有**不合法的**，关掉之后两者一起放行。
- **规避**：关检查时把「用户负责正确性」当成真的：关掉之后必须有别的验证手段（数值对拍、或依赖 A 类断言）。**不要**把 `enable_check=False` 当成一个性能开关。

**失败模式 3：以为每个原语只做「位置」这一件事。**

- **现象**：用 `rolling_buffer` 做流水线，结果某个消费者读到了本该是旧值的数据。
- **成因**：该原语的 docstring 逐字写着「The **region_cover property** of the consumer block of the target buffer **will become false**」——它**主动破坏一个原本成立的不变量**（region cover）。同类还有 `split(..., disable_predication=True)`（「may result in incorrect code generation if not used carefully」）、`unsafe_hide_buffer_access`（「This schedule primitive is **unsafe**, and may fail dependency analysis」）、`annotate_buffer_access`（「incorrect annotations may lead to incorrect code generation」）。
- **规避**：把原语按「是否只改位置」分类——`compute_at` / `compute_inline` 只改位置；上面这几条会改**语义不变量**。用后者时，正确性不能靠「调度跑通了」判断，只能靠数值对拍。

## 5. 实测数据

**测量条件**：`code/02_schedule_primitives.py` 在本机 `python3 3.13.5` + `numpy 2.4.4` 上跑出的结果，`cd code && make run` 退出码 0，本篇 **43 条断言（A 8 / B 30 / C 5）**。

> **★ 限定语（本域强制）**：下面每一个数字都是**本仓库的迷你 Schedule/Plan/Interp** 跑出来的，**不是 `tvm.s_tir.Schedule`**。它复现的机制是「放置状态如何改变物化次数、峰值与重算次数」；真实系统还有 target 相关的检查、`index` 参数控制的插入位置、以及一整套 `cache_read` / `rfactor` / `blockize` 原语。这些数字不支持任何关于真实 TVM 的结论。

**三态对照**（`make run` 的输出，解析式与实测逐行一致）：

| 调度 | 中间 buffer | 峰值（字节） | 中间流量（元素） | 求值次数 |
|---|---|---|---|---|
| 默认（根作用域） | 5 | **81920** | 40960 | 24576 |
| `compute_at(out.i)` | **5** | **10240** | 40960 | 24576 |
| `compute_inline` | **0** | **0** | **0** | 24576 |

派生关系（都是断言，不是描述）：峰值比 `20480 // 2560 = 8` 恰等于 i 轴 extent；`compute_at` 与默认的流量**相等**；单消费者链上 inline 与默认的求值次数**相等**。混合态（只留 `s1` 物化）：中间 buffer 个数 **1**。

**A 类：三态语义等价。** 参照物是纯 numpy 的 `ref_chain`，三种调度各自与它比对，最大误差都是 `0.000e+00`。**「变换前后语义等价」断言的是成功行为，按 CONTRIBUTING §2 归 A 类，不归 C 类。**

**C 类：多消费者地形**

| 检查 | 结果 |
|---|---|
| `s` 的两个消费者是 `out1`、`out2` | 是（打印确认） |
| C1：`compute_at(s, out1.i)` | 被拒绝（`out2` 不在该 loop 下） |
| C1 对照：`compute_inline(s)` | 接受（它没有这条前提） |
| C2：inline 后 `times(s)` | **2 == 消费者数** |
| C2：inline 比物化多算 | **4096 个元素 = (2−1)×4096** |
| C2 对照：物化时 `times(s)` | 1 |
| C3：切片关系 `2560 × 8 == 20480` | 精确成立 |
| C4：单消费者链上 inline 与物化的求值差 | **0**（「inline 更贵」不成立） |

**第一版被实跑推翻的两项**：

1. **`compute_at` 的前置条件第一版按静态 DAG 判定，被 C2 的对照跑翻。** 第一版把「消费者在 loop 下」实现成「消费者在数据依赖图上是 loop 所属 stage 的后代」——结果「把所有中间结果搬进输出循环」这组调用，按任意次序都通过，`s1` 也能先搬。真跑时 C2 的对照用**当前放置状态**判定，才发现次序有依赖：**先搬 `s1` 会被拒绝**。改成按当前放置判定之后，调用必须从后往前。这个次序依赖恰好复现了官方说的「过程式、不幂等」（§4 失败模式 1）。
2. **`reverse_compute_at` 在本主题的地形上用不了。** 本仓库的迷你实现只支持把 block 物化到**汇（输出）stage** 的循环内（解释器只在汇的循环里展开切片迭代），而 `reverse_compute_at` 要求「所有 producer 都在目标 loop 下」——链条里 producer 在**上面**，永远不会在汇的循环下。所以本主题**不给 `reverse_compute_at` 的运行结果**，只讲它在真实 API 里的存在与前置条件。这条限制是本仓库的简化，真实 `s_tir` 没有。

## 6. 小结与延伸

- `s_tir.Schedule`（**v0.26.0**）里管位置的两条原语：`compute_at`（**物化**，缩作用域）与 `compute_inline`（**不物化**）。**`compute_root` 不在 API 里**。
- 三态里唯一真正二分的量是**中间 buffer 个数**（`k` / `k` / `0`）；峰值之比等于被切轴的 extent；**流量在 `compute_at` 下不变**。
- **「不物化」的代价系数是消费者个数**：单消费者时 inline 免费，多消费者时重算 `times = 消费者数`。
- 调度是**过程式**的：次序影响前置条件是否满足、也影响重复调用会不会崩。关掉 `enable_check` 是把正确性责任接过来，不是解锁优化。

**延伸**：

- 这些原语被**自动**产生出来的两条路 → [04-auto-scheduling](04-auto-scheduling.md)
- v0.26.0 里另一条不需要调度原语的轨道（手写协作 scope） → [03-native-track-and-tile-primitives](03-native-track-and-tile-primitives.md)
- 主线在 MLIR 里的形态（bufferization 的就地判据） → `60-compilers/mlir/02-tensor-vs-memref`
- 上游对「融合省下什么」的推导 → `40-kernels/elementwise-fusion/02-fusion-and-traffic`

## 编译与运行

```bash
cd 60-compilers/tvm/code
make 02_schedule_primitives   # 只跑本篇
make run                      # 跑全部四篇
```

程序：`code/02_schedule_primitives.py`（共享迷你实现 `code/minitir.py`，外部事实源 `code/specs.py`）。

## 参考

- TVM `v0.26.0` 的 `s_tir.Schedule`：`compute_at` / `reverse_compute_at` 的签名与前置条件、`compute_inline` / `reverse_compute_inline`、`enable_check` 的「too strict」原文、过程式调度的原文、`rolling_buffer` / `split` / `unsafe_hide_buffer_access` / `annotate_buffer_access` 的警告原文：`https://github.com/apache/tvm/blob/v0.26.0/python/tvm/s_tir/schedule/schedule.py`
- 以上每条与抄录的原文都在 `code/specs.py` 里逐条对应，版本坐标 `v0.26.0`（release 2026-08-11）
- **本文不写**（见 `code/specs.py` 的 UNSOURCED）：`compute_root` 的官方替代写法（无明文，只能写「未移走的 block 处于 root 位置」并标明是推断）；`compute_root` 的「全树 0 命中」（只核到 API 目录不含）

---
title: 计算与调度分离
slug: tvm-compute-and-schedule
domain: 60-compilers
topic: tvm
level: 2
difficulty: intermediate
prerequisites:
  - 60-compilers/mlir/02-tensor-vs-memref
tags: [tvm, s-tir, tirx, schedule, materialization, compute-at, compute-inline, v0.26.0]
related:
  - 60-compilers/tvm/02-schedule-primitives
  - 60-compilers/mlir/02-tensor-vs-memref
  - 40-kernels/elementwise-fusion/02-fusion-and-traffic
code:
  - code/01_compute_schedule.py
status: stable
updated: 2026-09-16
---

# 计算与调度分离

> 「算什么」和「中间结果放不放到独立存储里」是两件事。TVM v0.26.0 把它们分成 `@T.prim_func` 与 `tvm.s_tir.Schedule` 两层；而这一层的**默认行为反直觉**——**每个中间结果都被物化在根作用域**，不是被 inline 掉。

## 0. 背景与动机

一条五级的逐元素链：

```text
s1 = a + b
s2 = 1.5·s1 + 0.25
s3 = 1.5·s2 + 0.25
s4 = 1.5·s3 + 0.25
s5 = 1.5·s4 + 0.25
out = 2·s5
```

这段式子只回答了「算什么」。它没有回答一个决定性能的问题：**`s1`…`s5` 要不要各自占一块存储？**

两种极端都能算出正确的数：

| 做法 | `s1`…`s5` 占的存储 | 代价 |
|---|---|---|
| 各自物化 | 5 块（每块一整级的形状） | 存储占用 5 份，且每级都要写一次内存再读一次 |
| 全部 inline（重算） | 0 块 | 存储为零，但消费者每取一个元素都要把整条链重算一遍 |

**答案取决于消费者有几个、以及每级多大**——这就是本域的统一主线：**物化还是重算**。

本主题的第一个反直觉事实在这里：**v0.26.0 的默认不是上面第二种，而是第一种**。你什么都不写、直接把这份计算交给编译入口，得到的是「每级都物化在根作用域」。

本文配套程序（`code/01_compute_schedule.py`）用本仓库的迷你 TIR 把这条链跑出来：**默认调度下 5 个中间结果全部物化，峰值中间存储 80 KiB**；把它们搬进输出循环之后降到 **10 KiB**（本机实测，见 §5）。

## 1. 原理

### 1.1 口径先写死：「物化」指什么

本域全程用同一个口径，它不指存储层级：

> **物化 = 为中间结果分配独立的存储。** 在寄存器、共享内存还是全局内存里，都算物化。
> **重算 = 根本不分配存储**，把计算直接替换进消费者的表达式。

这条口径很重要，因为**同一个原语在不同的口径下会被归到不同的类**。按「有没有独立存储」分：

| 放置 | 有没有独立存储 | 每元素被求值几次 |
|---|---|---|
| 默认（根作用域） | **有**（一整级） | 1 |
| `compute_at`（缩到某个循环内） | **有**（一次迭代的切片） | 1 |
| `compute_inline` | **没有** | 消费者个数 |

`compute_at` 与默认**同属「物化」**，差别只在作用域大小。这一点最容易写错，见 §4。

### 1.2 计算层：`@T.prim_func` 只写「算什么」

在 v0.26.0 里，一个**可调度**的计算写成这样（`T` 是 `tirx` 方言的别名，见 §2）：

```python
from tvm.script import tirx as T          # ★ 不是 `from tvm.script import tir as T`

@T.prim_func(s_tir=True)
def chain(A: T.Buffer(...), B: T.Buffer(...), Out: T.Buffer(...)):
    with T.sblock("s1"):                   # ★ 不是 T.block
        ...
```

`T.prim_func(s_tir=True)` 这个参数是关键：它把计算标记成**可调度式**。本文配套程序里对应的对象是 `PrimFunc`——它只由 stage 组成（`s1`…`s5`、`out`），**一个字节的存储都没有提**。

### 1.3 调度层：默认位置是根作用域

调度（`tvm.s_tir.Schedule`）决定每个 stage 的**物化位置**。把它作用到上面的计算上，**在你不调用任何原语的情况下**：

| stage | 位置 | 存储 |
|---|---|---|
| `s1`…`s5` | 根作用域 | 各一整级 |
| `out` | 根作用域 | 函数结果 |

**为什么可以断定默认是「物化在根作用域」？**两条依据，强弱不同，必须分开说：

1. **API 只有两个方向**（已核实）：v0.26.0 的 `s_tir.Schedule` 里只有 `compute_at`（把 block 搬到某个循环内）与 `compute_inline`（不物化）这两类「移动」原语，**没有 `compute_root`**。既然如此，**没被搬走的 block 就留在原处**——原处是什么？是根作用域。**这一步是推断，不是官方明文**（v0.26.0 的文档里找不到「原 `compute_root` 现用 X」的迁移说明）。
2. **一条源码佐证**（弱证据）：控制者基线里给出 `AttachType attach_type{kGroupRoot}` 这一行，**但未落到具体文件路径**，本文不把它当引文。

口径写在这里：本文的「默认全物化」是**推断 + 一条未落路径的佐证**，不是官方声明。计划 §0.7.C 本身也把同一结论标成推断。

### 1.4 推导：默认调度的三个整数

设一条 `k` 级链（`k` 个中间结果），每级 `N` 个元素、元素 `esize` 字节。默认调度下：

| 量 | 推导 | 本仓库迷你实现实测（k=5，N=4096，esize=4） |
|---|---|---|
| 中间 buffer 个数 | 每个中间 stage 一个 → `k` | **5** |
| 峰值中间存储 | 全部经由根作用域、同时存活 → `k·N·esize` | **81920 B = 80 KiB** |
| 中间流量 | 每级写一次 + 读一次 → `2·k·N` 个元素 | **40960 个元素** |
| 全 stage 求值次数 | 每元素每级算一次 → `(k+1)·N` | **24576 个元素** |

三行都是**整数表达式**，不是 `O(·)`——这是刻意的：断言必须落在具体整数上，实现改错时才会失败。

### 1.5 为什么需要「换一个位置」

80 KiB 与 10 KiB 的差别不是审美问题。把可用容量 `CAP` 设成 32 KiB，默认调度**装不下**：

```text
默认：峰值 81920 B > CAP 32768 B   → 不可行
全部 compute_at 到输出循环内：峰值 10240 B ≤ CAP  → 可行
```

这就是本域主线上那个权衡在 TVM 里的形态：**同一份计算、同一份数值结果，只因为中间结果放在哪里，一份装不下、一份装得下**。而数值结果**完全一样**——本机实测两份调度的输出最大差 `0.000e+00`（§5 C3）。

## 2. 执行映射

### 2.1 这段东西在真实栈里的位置

| 层 | 载体 | 本文用到的具体组件 |
|---|---|---|
| 计算 | `@T.prim_func(s_tir=True)` | `T.sblock`、`T.prim_func` |
| 调度 | `tvm.s_tir.Schedule` | `compute_at` / `compute_inline`（§2.3 的表） |
| 编译 | `tvm.compile` | `PassContext`、`tir_pipeline="default"` |
| 自动调度 | `tvm.s_tir.dlight` / `tvm.s_tir.meta_schedule` | 见 [04-auto-scheduling](04-auto-scheduling.md) |

**别名只有一个正确写法。** 本主题通篇写 `T.`，它来自：

```python
from tvm.script import ir as I
from tvm.script import tirx as T          # ★ v0.26.0 只注册了 tirx / relax / ir
```

`from tvm.script import tir as T` 在 v0.26.0 会 `ImportError`（方言注册表里没有 `tir`）。

### 2.2 编译入口与默认 pipeline

`python/tvm/driver/__init__.py` 全文只导出一行：`build, compile`。

```python
tvm.compile(mod, target=..., *, relax_pipeline="default", tir_pipeline="default")
```

`tir_pipeline` 的默认值是 `"default"`，而 `get_tir_pipeline("default")` 的函数体第一句是：

```python
if name == "default":
    name = "s_tir"
```

即：**默认走的是 s_tir 这条轨道**。这一点在 [03-native-track-and-tile-primitives](03-native-track-and-tile-primitives.md) 里会长成一个完整的失败模式（native 式的 kernel 落进 s_tir pipeline）。

两条 pipeline 各有多少个 pass，控制者基线给的是「s_tir 约 40 个、tirx 18–19 个」，**但这一组数字本文未逐条核对**（官方文档写 18、代码里数出 19，差异无官方解释），因此只作背景、不作为断言。

### 2.3 `s_tir.Schedule` 的两态（钉在 v0.26.0）

| 方法 | 签名（`python/tvm/s_tir/schedule/schedule.py`） |
|---|---|
| `compute_at` | `(block, loop, preserve_unit_loops=False, index=-1)` |
| `reverse_compute_at` | `(block, loop, preserve_unit_loops=False, index=-1)` |
| `compute_inline` | `(block)` |
| `reverse_compute_inline` | `(block)` |
| **`compute_root`** | **不在 API 里** |

**量级感**：一个 stage 的物化位置从根作用域换到输出循环内，占用的变化倍数**恰好等于被切的那条轴的 extent**（本机实测 `20480 // 2560 = 8`，i 轴 extent 也是 8）——不是「大概小一点」。

## 3. 实现与演进

**第一代（TE，已删除）：三态。** 早期 TVM 的 `te.create_schedule` 给的是三态——`compute_root`（物化在根）、`compute_at`（物化到某个 stage 的循环内）、`compute_inline`（不物化）。参数是 `compute_at(parent_stage, scope_ivar)`：**父 stage + 一个 IterVar**。

**第二代（v0.26.0 的 `s_tir`）：两态。** `te.create_schedule` 在 v0.26.0 上已经不存在（`AttributeError`），调度搬到了 `tvm.s_tir.Schedule`，作用在 `tirx::PrimFunc` 的 block 上。变化有三处：

1. **三态变两态**：没有 `compute_root`——见 §1.3 的推断。
2. **参数含义变了**：从「父 stage + IterVar」变成「要移动的 block + 目标 loop」，并新增 `index`（`-1` = 插到最后可插点，`-2` = 第一个可插点）。
3. **多了两条反向原语**：`reverse_compute_at` / `reverse_compute_inline`，**TE 时代没有**。

**为什么要分成两层？** 因为「同一份计算的多份调度」是搜索的前提：自动调度（DLight / MetaSchedule）要做的正是**在同一份 `prim_func` 上反复生成不同的调度**。如果计算和调度写在一起，搜索空间就没有载体。这条线在 [04-auto-scheduling](04-auto-scheduling.md) 收口。

**一条容易踩空的历史**：`te` 这个模块**至今仍然可以静默 import**。删掉了 `create_schedule` 等一半 API，**却没有留下任何 deprecation warning**——`import tvm.te` 不报错，`te.placeholder` / `te.compute` 也还能用，只有在调用被删的名字时才在调用处看到 `AttributeError`。这条与 §4 的第一条失败模式是同一件事的两面。

## 4. 踩坑与失败模式

四条，全部关于**真实 API**，逐条带来源；它们**不是本仓库程序的断言**（本仓库跑不了真实 TVM）。

**失败模式 1：照 2023–2024 年的教程学 TVM，在 v0.26.0 上第一次调用就崩。**

- **现象**：教程里第一行代码 `sch = te.create_schedule(C.op)` 抛 `AttributeError: module 'tvm.te' has no attribute 'create_schedule'`；而**前面的 `import tvm` 与 `import tvm.te` 都成功了**，所以报错点看起来莫名其妙。
- **成因**：`te` 的调度层在 v0.26.0 已经不在。`te/__init__.py` 里没有这些名字，也没有模块级 `__getattr__` 兜底，所以是 `AttributeError` 而不是 `ImportError`。更麻烦的是**没有任何 deprecation warning**——`te` 模块**静默地**只保住了 `placeholder` / `compute` / `reduce_axis` / `sum` / `create_prim_func` 这一小撮名字（核查范围：只核了这些名字仍在 `te/__init__.py` 里，**没有**逐个核 te 下每个名字的存亡，所以不能说「te 只剩这五个」）。
- **规避**：把「教学年份」当成版本信号。凡看到 `te.create_schedule` / `tvm.tir` / `autotvm`，先假定是 v0.20.0 之前的写法。**删除动作的版本归属（te 删于哪个 release）本文未取到一手证据**——能确认的是「v0.26.0 上已不存在」。

**失败模式 2：以为默认是「把中间结果 inline 掉」。**

- **现象**：按「编译器会做融合」的直觉去读默认调度，得出「5 级链在默认下只占 1 块存储」。
- **成因**：默认位置是**根作用域**（§1.3），不是 inline。上游 `40-kernels/elementwise-fusion/02` 讲的「融合 = 生产者不落存储」对应的是 `compute_inline` 那一态，**不是**默认态，**也不是** `compute_at`（它仍然物化）。把上游那句口径直接套到 TVM 的默认行为上，方向就是反的。
- **规避**：在脑子里把三个状态和「有没有独立存储」绑死（§1.1 那张表）。本机实测：默认 5 个中间 buffer、全 inline 0 个，而**两者的数值逐元素相同**（`0.000e+00`）——所以这个错误**不会**被任何数值检查抓到。

**失败模式 3：到处找 `compute_root`。**

- **现象**：想把某个 block 显式「放回根位置」，搜到 TE 时代的 `sch.compute_root()`，在新 API 下找不到对应方法。
- **成因**：`compute_root` **不在 `s_tir.Schedule` 的 API 里**（核查范围：核的是该类的 API 目录，不是「全树 grep」）。TE 的三态在 `s_tir` 里只剩两态。
- **规避**：**默认就在根位置**——「放回根」这件事不需要一个原语。反过来要小心：如果某个 block 已经被 `compute_at` 搬走过，本主题的文档与程序都**不给「搬回去」的官方写法**，因为 v0.26.0 里没有明文说明。

**失败模式 4：以为 `tir` 还是那个别名。**

- **现象**：`from tvm.script import tir as T` → `ImportError`；`import tvm.tir` → `ModuleNotFoundError`。
- **成因**：v0.26.0 把 TIR 拆成了 `tirx`（native 式）与 `s_tir`（可调度式）两条轨道，方言注册表里只有 `tirx` / `relax` / `ir`。同理 `T.block` 已改名 `T.sblock`、`sch.get_block` 已改名 `sch.get_sblock`。
- **规避**：`from tvm.script import tirx as T`。**文档里每出现一次 `T.`，都要能说出它来自哪个别名**——本文 §2.1 就是为这条写的。

## 5. 实测数据

**测量条件**：`code/01_compute_schedule.py` 在本机 `python3 3.13.5` + `numpy 2.4.4` 上跑出的结果，`cd code && make run` 退出码 0，本篇 **19 条断言（A 3 / B 11 / C 5）**。

> **★ 限定语（本域强制）**：下面每一个数字都是**本仓库的迷你 TIR**（`code/minitir.py` 的 `Schedule` / `Plan` / `Interp`）跑出来的，**不是 TVM**。它复现的是「物化位置改变时，存储与流量的整数关系怎么变」这条**机制**，不支持任何关于真实 TVM 的结论。真实系统还有 target、`PassContext`、DLight 的规则库、后端 lowering 一整套东西。

**算例**：`a,b → s1 → … → s5 → out`，形状 `8×512×f32`，每级 `4096` 个元素（`16384 B = 16 KiB`）。

**默认调度（不调用任何原语）**：解析式与解释器实测**逐行一致**。

| 项 | 解析式 | 实测 |
|---|---|---|
| 中间 buffer 个数 | 5 | 5 |
| 峰值中间存储（元素） | 20480 | 20480 |
| 峰值中间存储（字节） | **81920** | **81920** |
| 中间写入（元素） | 20480 | 20480 |
| 中间读取（元素） | 20480 | 20480 |
| 中间流量（元素） | **40960** | **40960** |
| 全 stage 求值（元素） | 24576 | 24576 |

「解析式」是 `Plan` 按公式算的，「实测」是 `Interp` 执行时逐次累加的——**两条独立路径**，一致才有意义（B 类断言）。

**C 类：容量判定与它的对照**

| 检查 | 结果 |
|---|---|
| 默认调度在 `CAP = 32 KiB` 下 | `InfeasibleSchedule`（峰值 81920 B > 32768 B） |
| 实测峰值是否真的超 | 是，**81920 B > 32768 B** |
| 对照：全部 `compute_at` 到 `out` 的 `i` 循环 | 可行，峰值 **10240 B** ≤ 32768 B |
| C1：违反 `compute_at` 前置条件（`s1` 的消费者 `s2` 不在目标 loop 下） | 被拒绝（`ScheduleFailure`） |
| C3：默认（5 个 buffer）与全 inline（0 个 buffer）的数值差 | **0.000e+00** |
| C4：把解析式里的「流量」改成只算写（漏掉读） | 20480 ≠ 实测 40960，**失配** |

**A 类**：参照物是纯 numpy 的 `reference_chain`（与迷你 IR/解释器没有任何共享代码），最大误差 `0.000e+00`。

**第一版被实跑推翻的两项**（都是跑出来的，不是推出来的）：

1. **「峰值 = 级数 × 每级元素」第一版算进了输出 buffer。** 解析式第一版把 `out` 也计入「中间存储」，得到 24576 而不是 20480，B 类表立刻 MISMATCH。修正：**峰值只统计中间结果**——输出 buffer 是函数结果，本来就要存在，把它算进「中间存储」会让默认与 `compute_at` 的比较失去意义。
2. **汇 stage 被执行了两次。** 解释器第一版先在「根位置物化」那一步把 `out` 算了一遍，又在本步里算了一遍，导致「中间读取」多 4096、「求值」多 4096。修正：根位置那一步**跳过汇 stage**。

## 6. 小结与延伸

- 「算什么」与「中间结果放哪儿」是两层：`@T.prim_func(s_tir=True)` 与 `tvm.s_tir.Schedule`（**v0.26.0**）。
- **默认是物化在根作用域**，不是 inline。证据含**一条推断**（API 只有两个方向 + 没被搬走的 block 留在原处），本文已标明。
- 物化三量的整数关系：buffer 数 `k`、峰值 `k·N·esize`、流量 `2·k·N`、求值 `(k+1)·N`。
- **`compute_at` 仍然是物化**，它改的是作用域，不是「融不融」。
- **数值正确性完全无法区分这些调度**（本机实测差值 0）——所以调度写错了，只有代价模型能发现。

**延伸**：

- `compute_at` 的作用域语义、三态逐项对照、以及三条有官方原文的失败模式 → [02-schedule-primitives](02-schedule-primitives.md)
- v0.26.0 的另一条轨道（native 式）与 tile 原语 → [03-native-track-and-tile-primitives](03-native-track-and-tile-primitives.md)
- 这些调度由谁生成：规则式与搜索式 → [04-auto-scheduling](04-auto-scheduling.md)
- 上游的「融合为什么省字节、收益上界为什么是链长 `k`」→ `40-kernels/elementwise-fusion/02-fusion-and-traffic`（正文在上游，此处回链）
- MLIR 里同一件事的形态：`tensor`（未指定存储）vs `memref`（已物化）→ `60-compilers/mlir/02-tensor-vs-memref`（正文在上游，此处回链）

## 编译与运行

```bash
cd 60-compilers/tvm/code
make 01_compute_schedule      # 只跑本篇
make run                      # 跑全部四篇
```

程序：`code/01_compute_schedule.py`（共享迷你实现 `code/minitir.py`，外部事实源 `code/specs.py`）。

## 参考

- TVM `v0.26.0` 的 `s_tir.Schedule` API（`compute_at` / `compute_inline` 的签名、`compute_root` 不在 API 里）：`https://github.com/apache/tvm/blob/v0.26.0/python/tvm/s_tir/schedule/schedule.py`
- TVM `v0.26.0` 的 `te` 模块（`te.create_schedule` 无定义、`te` 无模块级 `__getattr__`）：`https://github.com/apache/tvm/blob/v0.26.0/python/tvm/te/__init__.py`
- TVM 编译入口（`build, compile` 两个导出、`tir_pipeline="default"`）：`https://github.com/apache/tvm/blob/v0.26.0/python/tvm/driver/__init__.py`
- TVM 的 `get_tir_pipeline`（`if name == "default": name = "s_tir"`）：`https://github.com/apache/tvm/blob/v0.26.0/python/tvm/s_tir/pipeline.py`
- 以上每条与抄录的原文都在 `code/specs.py` 里逐条对应，版本坐标 `v0.26.0`（release 2026-08-11）
- **本文不写**（查不到可定位载体，见 `code/specs.py` 的 UNSOURCED）：`tvm.lower` 的存在性；`compute_root` 在「全树」的命中数（只核到 API 目录不含）；te 删除动作的版本号归属；两条 pipeline 的 pass 计数

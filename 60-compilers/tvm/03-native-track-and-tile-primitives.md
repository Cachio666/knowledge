---
title: 两条并列的 TIR 轨道与 tile 原语
slug: tvm-native-track-and-tile-primitives
domain: 60-compilers
topic: tvm
level: 3
difficulty: advanced
prerequisites:
  - 60-compilers/tvm/01-compute-and-schedule
tags: [tvm, s-tir, tirx, tile-primitives, pipeline, tilelang, v0.26.0]
related:
  - 60-compilers/tvm/02-schedule-primitives
  - 60-compilers/tvm/04-auto-scheduling
  - 30-gpu-programming/triton
code:
  - code/03_two_tracks.py
status: stable
updated: 2026-09-16
---

# 两条并列的 TIR 轨道与 tile 原语

> v0.26.0 里同一个计算有**两种并列的写法**：`s_tir`（可调度式，交给调度器生成并行映射）与 `TIRx` native（手写式，`cta_id()`/`thread_id()` 与 tile 原语全写出来）。只看一份文档会以为另一套不存在。而 `tvm.compile` 的 `tir_pipeline` **默认值解析到 `s_tir`**——写了 native 式 kernel 的人会落进另一条轨道。

## 0. 背景与动机

本文与主线（物化 vs 重算）的关系是：**native 轨把「物化到哪一级存储」这件事从调度器手里移到了人手里**。在 `s_tir` 轨上，「中间结果放到哪儿」由 `compute_at` / `compute_inline` 决定（[02](02-schedule-primitives.md)）；在 native 轨上，这件事写死在 kernel 里——tile 原语把数据显式地搬进 CTA 级缓冲、搬进寄存器，**调度器不再有发言权**。所以这不是另一条主线，而是同一条主线在另一条轨道上的形态。

具体场景：`out[i] = 2·Σ_j A[i,j]·x[j]`（一个 `16×128` 的 GEMV）。

**s_tir 轨**：一个归约 block（`red`）+ 一个逐元素 block（`out`）。中间结果 `red` 物化与否**由调度决定**。

**native 轨**：`grid=(16,)`、`block=(128,)` 手写给出；数据先 `Tx.tile.cta.copy` 进 CTA 级缓冲 `xs`，每个线程 `Tx.tile.copy` 把自己那一份载进寄存器 `areg`，然后 **warp 级**原语做 32 线程内的部分和、**CTA 级**原语合并 warp 的结果。**五个原语调用里没有一个字提到调度**。

两条轨道在**本仓库的迷你实现**上跑出同一个数值（与独立 numpy 参考最大误差 `0.000e+00`，限定语见 §5）。

## 1. 原理

### 1.1 两条轨道的形状不同

| 轨道 | 写法 | 并行映射来自 | pipeline |
|---|---|---|---|
| **`s_tir`**（可调度） | `@T.prim_func(s_tir=True)` + `T.sblock` | **调度器**（DLight / MetaSchedule 产生原语调用） | `tir_pipeline="default"` |
| **`TIRx` native** | `@T.prim_func`（`s_tir=False`，**这是默认值**）+ `T.device_entry()` + `T.cta_id()` / `T.thread_id()` + `Tx.tile.*` | **人**（手写在 kernel 里） | `tir_pipeline="tirx"`（**必须显式指定**） |

（`T` 是别名：`from tvm.script import tirx as T`；`Tx` 是 tile 原语的命名空间根，见 §1.3。）

**「可调度」的定义性差别在这里**：一个 `s_tir=True` 的 `prim_func` 里有 block，调度原语才有下手的地方；一个 native 式 kernel 里只有手写的线程映射与 tile 原语，**没有可调度的 block**。这不是风格偏好，是结构差别。

### 1.2 默认落到哪条 pipeline

```python
tvm.compile(mod, target=..., *, relax_pipeline="default", tir_pipeline="default")
```

`tir_pipeline` 的默认值是 `"default"`；而 `get_tir_pipeline("default")` 的函数体第一句是：

```python
if name == "default":
    name = "s_tir"
```

于是：**`@T.prim_func` 的默认是 native 式（`s_tir=False`），而编译的默认 pipeline 是 s_tir 那条**。两头默认值指向相反方向——这是本文最值得记的一条。

两条 pipeline 各有多少个 pass，控制者基线给的是「s_tir 约 40 个、tirx 18–19 个」，**本文未逐条核对**（官方文档写 18、代码里数出 19，差异无官方解释），只作背景。本文配套程序里用的是**自建的类别模型**（「循环结构变换 / 块位置与缓存 / 并行映射 / 向量化 / 后端 lowering / 清理」六类），**不是官方 pass 表**——差异刻意的：本仓库不编 pass 名。

### 1.3 tile 原语：命名空间**编码协作 scope**

官方页面写的是：

> The C++ registry currently defines **31 operation names**

**是 31，不是 29。** 命名空间也不是 `Tx.*`，而是**多一层 `tile`**：

| 命名空间 | 协作 scope |
|---|---|
| `Tx.tile.<name>` | thread 级（裸命名空间即 thread） |
| `Tx.tile.thread.*` | thread 级（显式写法） |
| `Tx.tile.warp.*` | warp 级 |
| `Tx.tile.wg.*` | warpgroup 级（`Tx.tile.warpgroup.*` 的别名） |
| `Tx.tile.cta.*` | CTA 级 |
| `Tx.tile.cluster.*` | cluster 级 |

分组为 data movement（`copy` / `copy_async` / `permute_layout`）、matrix multiply（`gemm` / `gemm_async`）、fill/memset/zero、cast + elementwise、reductions（`sum` / `max` / `min`）、fused/compose **六组**。

**命名空间不是命名习惯，它是唯一的 scope 线索。** 而 scope 决定了这个原语**被执行多少次**——这是本主题最实用的一条量化关系：

| scope | 调用次数 | 本机实测（16 CTA × 128 线程） |
|---|---|---|
| cta | CTA 数 | `Tx.tile.cta.copy` **32**（2 次调用 × 16） |
| warp | CTA 数 × warp 数 | `Tx.tile.warp.sum` **64**（16 × 4） |
| thread | CTA 数 × 线程数 | `Tx.tile.copy` **2048**（16 × 128） |

同一个原语名换成不同 scope 的前缀，执行次数能差 **128 倍**（thread 级 vs cta 级，在这个形状上）。**scope 选错时，有的硬件会直接拒绝**（本机实测：16 个线程的 block 上调 warp 级原语被拒绝，见 §5 C1）。

> **本仓库的迷你 registry 是官方清单的子集**（7 个名字，覆盖 thread / warp / cta 三个 scope）。**31 个名字的完整清单本文未取到**，所以程序里只用有出处的组内示例名（`copy` / `copy_async` / `permute_layout` / `gemm` / `gemm_async` / `sum` / `max` / `min`），并且不声称与官方一一对应。

## 2. 执行映射

**这一段在真实栈里的位置**：两条轨道在 `gpu` / `cuda` target 上最终都要落到同一类后端（PTX → cubin），分歧在**中间**——`s_tir` 轨要经过「调度器产出的调度」这一层，native 轨没有。

**组件与量级**：

| 组件 | 作用 | 量级 |
|---|---|---|
| `tvm.compile(..., tir_pipeline=...)` | 选轨道 | 默认 `"default"` → `s_tir` |
| `get_tir_pipeline` | 解析 pipeline 名 | 一行 `if name == "default"` |
| `PassContext` | 承载编译期开关（`tir.disable_*` 一类） | 见 §4 的可选观察 |
| `Tx.tile.*` 原语 | 命名空间即协作 scope | **31** 个操作名、**6** 个命名空间 |
| `T.device_entry()` / `T.cta_id()` / `T.thread_id()` | native 轨的入口与索引 | 本机算例手写 2 个显式数（grid 1 个 + block 1 个），s_tir 轨 **0 个** |

## 3. 实现与演进

**第一代：只有可调度式。** TE 时代（已删除）与早期的 TIR 调度只有一种形态：写计算、写调度、编译。所有并行映射都由调度原语产生。

**第二代（v0.26.0）：两条并列。** `s_tir`（可调度式）与 `TIRx` native（手写式）并存，各有自己的 pipeline。native 轨的出现把「想精确控制协作 scope 的人」从调度器里解放出来——代价是他要自己写循环、自己保证正确性。

**与 TileLang 的关系（需要说清，容易记反）**：

- **TileLang 是独立项目，没有并入 TVM。** 证据：TileLang 自身仓库是 **MIT 协议、Tile-AI 维护**，通过 `.gitmodules` 把 TVM 作为子模块 vendor 进来（`3rdparty/tvm` → `github.com/TileLang/tvm`）。**依赖方向是 TileLang → TVM，不是反向。**
- **「它已迁到 TVM 的 TIRX 这一层」这条本文不写。** 计划里支撑它的是一条 `tile-ai/tilelang` 的 PR（含日期与 changelog 引文），而该 PR **两次定向检索均 0 命中**；按「引用值隐含来源存在」的规矩，**没有可打开的来源就不带这条结论出厂**。能核到的只有上面两条（独立仓库、依赖方向）。注意「独立项目」与「采用上游的某个中间表示」并不矛盾——但后者需要证据，本文没有。
- 论文：**arXiv:2504.17577**（2025-04-24）。

> **★ 一条查不到的东西，本文不写**：`tile-ai/tilelang` 的某个 PR 号（含具体日期与 changelog 引文）**两次定向检索均 0 命中**，按「引用值隐含来源存在」的规矩，这条**不带数字、不写出厂**。同理，「TVM 源码里 tilelang 只出现 N 次」这一表述也复现不了，不写。

## 4. 踩坑与失败模式

三条，**逐条带来源**；都不是本仓库程序的断言。

**失败模式 1：写了 native 式 kernel，却用默认 pipeline 编译。**

- **现象**：kernel 里全是 `T.cta_id()` / `T.thread_id()` / `Tx.tile.*`，`tvm.compile(mod)` 一调——落进的是 **s_tir** pipeline。
- **成因**：`tir_pipeline` 的默认值是 `"default"`，而它解析成 `"s_tir"`（§1.2）。真正跑 native 轨要**显式**写 `tir_pipeline="tirx"`。两个默认值方向相反：`@T.prim_func` 默认 `s_tir=False`（native），编译默认却是 s_tir。
- **规避**：写 native 式 kernel 时**永远显式指定 `tir_pipeline="tirx"`**，不要依赖默认值。顺带记一条：**轨道由参数选，不由模块内容自动判断**——本机实测，同一个 `s_tir` 模块只要把参数改成 `"tirx"`，也会落到另一条 pipeline 上。

**失败模式 2：以为 TileLang 是 TVM 的一个子模块/模块。**

- **现象**：`import tilelang` 之后按 TVM 的内部模块去理解它的版本关系，或者反过来以为 TVM 的 TIRX 是 TileLang 的产物。
- **成因**：**依赖方向是 TileLang → TVM**。TileLang 通过 `.gitmodules` 把 TVM vendored 进 `3rdparty/tvm`，自己维护一份（指向 `github.com/TileLang/tvm`）。它是 MIT 协议、Tile-AI 维护的**独立仓库**。
- **规避**：把两者当**上下游**而不是父子。TVM 自己的 tile 原语也已经有（`docs/tirx/tile_primitives.rst`），两者是同一层里并行的两套东西。

**失败模式 3：按 `Tx.<name>` 去找 tile 原语。**

- **现象**：照着博客写 `Tx.gemm(...)`，找不到；或者把操作数记成 29 个，对不上官方页面。
- **成因**：命名空间**多一层 `tile`**——最小形态是 `Tx.tile.<name>`（裸命名空间即 thread 级），带 scope 的是 `Tx.tile.warp.*` / `Tx.tile.cta.*` / `Tx.tile.cluster.*` 等。操作名计数官方逐字是 **31**。
- **规避**：**scope 从命名空间读**，不要从记忆读。写到哪一级协作，前缀就写到哪一级——写错 scope 不只是性能问题（thread 级与 cta 级的执行次数在这个形状上差 128 倍），有的组合在硬件约束下直接不成立（warp 级原语需要至少 32 个线程）。

> **★ 一条证据不完整的观察（本文不作为失败模式）**
>
> 研究过程中出现过一条很漂亮的观察：两条 pipeline 可能读**不同命名空间**的 PassContext 键
> （`tirx.disable_vectorize` vs `tir.disable_vectorize`），即同一个开关在一条路径上生效、
> 在另一条路径上静默无效。
>
> **但这条只拿到一半证据**：`tirx` 侧证实了读 `tir.disable_vectorize`，
> **`s_tir` 侧的源文件未取到**。按本域纪律（§0.7.C 末尾），它**不得作为失败模式**，
> 只能作为**可选观察**出现在这里。
>
> **本仓库不声称复现了它**——本仓库跑不了真实 TVM，本机也没有 `tvm` 包；
> 上面那两行代码里也没有任何一处去验它。**控制者补上一手来源之前，请把它当猜想。**

## 5. 实测数据

**测量条件**：`code/03_two_tracks.py` 在本机 `python3 3.13.5` + `numpy 2.4.4` 上跑出的结果，`cd code && make run` 退出码 0，本篇 **21 条断言（A 6 / B 11 / C 4）**。

> **★ 限定语（本域强制）**：下面每一个数字都是**本仓库的迷你实现**跑出来的，**不是 TVM**。两条轨道的执行、tile 原语的调用计数、pipeline 的类别构成，都是 `code/minitir.py` 里的迷你模型；**真实 TVM 的两条 pipeline 里到底有哪 40 个 / 18 个 pass，本仓库没有核对，也没有写进任何断言**。这些数字支持的是「scope 决定调用次数」这类**机制**结论。

**A 类**：同一个计算，两条轨道各自执行，与独立 numpy 参考比对的**最大误差都是 `0.000e+00`**；两条轨道互相之差也是 `0.000e+00`。

**B 类：tile 原语的调用次数由 scope 决定**（解析式与实测逐行一致）

| tile 原语 | scope | 调用次数 | 实测次数 | 解析式 |
|---|---|---|---|---|
| `Tx.tile.cta.copy` | cta | 2 | **32** | 16 × 2 |
| `Tx.tile.copy` | thread | 1 | **2048** | 16 × 128 |
| `Tx.tile.warp.sum` | warp | 1 | **64** | 16 × 4 |
| `Tx.tile.cta.sum` | cta | 1 | **16** | 16 × 1 |

其余 B 类：用到的**不同**原语 **4** 个；覆盖 **3** 个协作 scope（cta / thread / warp）；native 轨手写 **2** 个显式并行度数字，s_tir 轨 **0** 个；规格值 `31`（操作名）与 `6`（命名空间）逐条核过。

**pipeline 归属**（程序里的类别模型，非官方 pass 表）：

```text
compile(s_tir 模块, tir_pipeline="default") → s_tir ：循环结构变换 / 块位置与缓存 / 并行映射 / 向量化 / 后端 lowering / 清理
compile(native 模块, tir_pipeline="tirx")   → tirx  ：向量化 / 后端 lowering / 清理
```

**C 类**

| 检查 | 结果 |
|---|---|
| C1：16 线程的 block 上调 `Tx.tile.warp.sum` | 被拒绝（`ScopeMismatch`：warp 级需 ≥ 32 线程） |
| C1 对照：128 线程（4 个 warp） | 接受，warp 级原语执行 **64** 次 |
| C2：对 native kernel 调 `Schedule` | 被拒绝（它没有可调度的 block） |
| C2 对照：同一计算的 s_tir 版本 | 接受，拿到 **2** 个可调度 block |
| C3：native 模块用默认 `tir_pipeline` 编译 | 被拒绝（`PipelineError`，默认解析到 s_tir） |
| C3 对照：显式 `tir_pipeline="tirx"` | 接受 |
| C4：把 thread 级原语的次数按 cta 级记 | 16 ≠ 实测 2048，**失配** |

> **C2 与 C3 的断言语义要说清**：它们断言的是**本仓库迷你实现**的行为（`Schedule` 只接受 `s_tir=True` 的模块、`compile` 会拦住轨道不匹配的模块）。真实 TVM 在这两种情形下**具体报什么错，本文不写**——本仓库跑不了它（计划 Chunk 2 第二条硬性要求：真实 API 的行为不得写成程序断言）。

**第一版被实跑推翻的一项**：

1. **同一个 tile 原语出现两次时，按调用表逐行对账会 MISMATCH。** 第一版的 B 类表按「原语名 → 次数」列出，而 `Tx.tile.cta.copy` 在我的 kernel 里出现了**两次**（一次载入、一次写回），于是实测 32 与「单次调用 16」对不上。修正：解析式按**每个原语的调用次数 × 该 scope 的单次执行数**算，并把「次数/次」单列一栏。这条不是数值算错，是**记账口径**错——但它照样让断言失败，说明 B 类表的判别力是真的。

## 6. 小结与延伸

- v0.26.0 有**两条并列的 TIR 轨道**：`s_tir`（可调度，`T.sblock`）与 `TIRx` native（手写，`T.device_entry()` + `cta_id()`/`thread_id()` + `Tx.tile.*`）。
- **`tir_pipeline` 的默认值解析到 `s_tir`**，而 `@T.prim_func` 的默认是 `s_tir=False`——两个默认值方向相反，写 native kernel 必须显式指定 `"tirx"`。
- tile 原语**31 个操作名**，命名空间**多一层 `tile`** 且编码协作 scope；**scope 决定调用次数**（本机实测差可达 128 倍）。
- **TileLang 是独立项目，依赖方向 TileLang → TVM**（作者是 Tile-AI、MIT 协议、通过 `.gitmodules` vendored TVM）。「它已迁到 TIRX」这条**无来源，本文不写**。

**延伸**：

- 并行映射由谁产生（规则式 vs 搜索式） → [04-auto-scheduling](04-auto-scheduling.md)
- `s_tir` 轨上「物化到哪儿」怎么决定 → [02-schedule-primitives](02-schedule-primitives.md)
- Ansor 的两级搜索空间（sketch 阶段就含 fusion 决策） → `60-compilers/auto-tuning/01-search-space`（**正文在那里，此处回链**）
- Triton 语言本身 → `30-gpu-programming/triton`（正文在上游，此处回链）

## 编译与运行

```bash
cd 60-compilers/tvm/code
make 03_two_tracks            # 只跑本篇
make run                      # 跑全部四篇
```

程序：`code/03_two_tracks.py`（共享迷你实现 `code/minitir.py`，外部事实源 `code/specs.py`）。

## 参考

- TVM `v0.26.0` 的 tile 原语文档（「The C++ registry currently defines 31 operation names」、命名空间与分组）：`https://github.com/apache/tvm/blob/v0.26.0/docs/tirx/tile_primitives.rst`
- TVM `v0.26.0` 的编译入口与 pipeline 解析（`tir_pipeline="default"`、`if name == "default": name = "s_tir"`）：`https://github.com/apache/tvm/blob/v0.26.0/python/tvm/driver/__init__.py` 与 `https://github.com/apache/tvm/blob/v0.26.0/python/tvm/s_tir/pipeline.py`
- TileLang（独立仓库、MIT 协议、Tile-AI 维护、`.gitmodules` 里 `3rdparty/tvm`）：`https://github.com/tile-ai/tilelang`
- TileLang 论文：`https://arxiv.org/abs/2504.17577`
- 以上每条与抄录的原文都在 `code/specs.py` 里逐条对应，版本坐标 `v0.26.0`（release 2026-08-11）
- **本文不写**（见 `code/specs.py` 的 UNSOURCED）：两条 pipeline 的 pass 计数（未逐条核对）；两条 pipeline 读不同命名空间 PassContext 键的观察（只拿到一半证据，且不作为失败模式）；TileLang 的某个 PR 号（检索 0 命中）；tile 原语的 31 个名字的完整清单（未取到，本仓库 registry 是子集）

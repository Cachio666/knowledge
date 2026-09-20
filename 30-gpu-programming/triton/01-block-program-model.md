---
title: 块级程序模型：一个块里的元素归哪个线程管，你不说，编译器也不说
slug: triton-block-program-model
domain: 30-gpu-programming
topic: triton
level: 2
difficulty: intermediate
prerequisites:
  - 30-gpu-programming/cuda/01-thread-hierarchy-and-sync
  - 20-gpu-architecture/execution-model/02-simt-and-divergence
tags: [triton, block-program, tl-arange, reduction, online-softmax, axis, keep-dims, model-values]
related:
  - 40-kernels/softmax-normalization/02-online-softmax
  - 40-kernels/reduce-scan/01-reduction-algorithms
  - 60-compilers/inductor/03-codegen
  - 45-kernel-lifecycle/implementation/01-reference-first
code:
  - code/01_block_model.py
status: stable
updated: 2026-09-20
---

# 块级程序模型：一个块里的元素归哪个线程管，你不说，编译器也不说

> **块级程序写起来像「对整块数据操作」，但块内张量的每一个元素属于哪个线程，由编译器决定——而这个决定后来变成了代价。** 本篇用一个纯 Python + numpy 的**迷你 Triton**（`code/minitriton.py`，1113 行）把这件事逐条算出来，并给出四条带数字的结论：
>
> 1. ★★ **「平移量选哪个不影响结果」这句话是可证伪的。** 同一份 online softmax 输出，**同时**逐元素等于「减行最大」与「减行内首元素」两条独立参照（后者**根本不求最大值**）——32000 个输出位置的最大绝对误差都是 **3.986e-08**（容差 1e-05）。这条**看起来像装饰**的等价性，正是抓「最大值减错了 / 忘了减」的判据。
> 2. ★★ **忘掉最大值平移，`x ≈ 200` 的输入会产出 32000/32000 个 NaN** ——不是「有一些 NaN」：归一化常数本身变成了 `inf`。正确版本在**同一个输入**上全为有限值（该输入的元素级最大值 **207.6**）。
> 3. ★★ **`[BLOCK_N=256]` 的行归约不是一次操作，是三个相位**：线程内 128 ops → warp 内蝶形 **5 步 / 640 ops** → 跨 warp **4 ops / 32 B 共享内存 / 2 个 barrier**；再把结果广播回去**还要 2 个 barrier**。一行 1000 个元素的 softmax 统计量，两趟合计 **32 次 barrier**（4 个列块 × 2 趟 × 4）。**这些是模型值，本机无 GPU，一条实测都没有。**
> 4. ★ **同一个 `tl.sum` / `tl.max`，放在 32 个元素的块上（一个 warp）与 256 个元素的块上（4 个 warp），跨 warp 相位从 0 个 barrier 变成 2 个**。块级抽象**不告诉你**这件事。
>
> ★ **★ 本篇的「反直觉量化结论」**：**「平移量选哪个不影响结果」这句话是**可证伪的**——同一份 online softmax 输出**同时**逐元素等于「减行最大」与「减行内首元素」两条独立参照（后者**根本不求最大值**），32000 个输出位置的最大绝对误差都是 **3.986e-08**；而**忘掉**平移的那一版在 `x ≈ 200` 的输入上产出 **32000/32000 个 NaN**。
>
> ★ 本篇的**真值来源**是三条互不重叠的代码路径：`refs.py` 的纯 Python 标量参照、`minitriton.py` 的解释器、以及 numpy。**自造 DSL 的最大风险是「模拟器同时是被测对象」**——对策不是「小心一点」，而是 `code/meta_falsify.py` 的元检验（`make falsify`，见 §3.4）。

> ★ **本主题的三条限定语，在本篇的落点**（域 README 规定逐篇写明，不得省略）：
> 1. **「本模型的规则是自造的」**——本篇用到五条规则：**摊维规则**、**属主算法**（含「精确切分」约束）、**转换定价公式**、**三相归约的 ops 计数公式**、**交叉点 `K*=2` / `K*=64`**。**这五条不声称是 Triton 的规定**，逐条写在 `code/specs.py` 的 `MODEL_OWN_RULES` 里（本篇 §1.4 复述）。
> 2. **「本主题无实测，只有模型值」**——B 类的每一个数（ops / 共享内存字节 / barrier 数）都是**这个模型算出来的**，不是硬件读数、不是 Triton 编译器的输出。本机**无 GPU**，`triton` 包**全机器不存在**（macOS 无 wheel）。
> 3. **「不是硬件，也不是 Triton 编译器」**——本程序**不执行任何 GPU 代码**，也不调用任何 Triton 组件。凡程序说「代价是 2 个 barrier」的地方，请读作「**按本模型自定的定价公式算是 2 个**」。
> 4. ★ **一处已记录的简化**：本模型要求 `spt×tpw×wpc == shape`（**精确切分、无复制**），而真实 Triton 的 layout **可以覆盖不足**——官方第一个例子 `gl.BlockedLayout([1,1],[1,32],[1,4],[1,0])` 在 128×128 上只覆盖 **1×128**，要沿第 0 维复制。**这不改变归约的结论**，但它意味着「本模型复现了官方布局」这句话**比实际更强**，必须写明。

## 0. 背景与动机

### 0.1 两个具体的对照场景

**场景 A（看起来是一次操作，其实是三个相位）**。块级程序里最普通的一行是：

```python
blk_max = tl.max(x, axis=None)      # x 的形状是 [BLOCK_N]
```

它读起来像「对整个块取最大值」——可块里那 `BLOCK_N` 个元素**分散在 32 × num_warps 个线程的寄存器里**。这一行要变成一台**三相机器**：每个线程先合并自己寄存器里的几个元素（线程内），然后用 `__shfl_xor_sync` 的蝶形在 warp 内合并（warp 内），最后跨 warp 经共享内存汇总（跨 warp）。

在本篇的配置里（`BLOCK_N = 256`、`num_warps = 4`），这三相的代价分别是 **128 ops / 640 ops / 4 ops + 32 B + 2 barrier**；而把结果**广播回**每个线程又**再要 2 个 barrier**（`code/01_block_model.py` 的 §6 打印这一行）。

**同一个 `tl.max`，只把块大小换成 32（一个 warp）**，跨 warp 相位就**整体消失**（0 个 barrier）。**源码一个字没改。**

**场景 B（平移量选错，输出从错一点变成全错）**。online softmax 的第一趟要滚动维护行最大值 `m_i` 与指数和 `s_i`。数学上 `softmax(x)_i = exp(x_i − c) / Σ_j exp(x_j − c)` 对**任意常数 `c`** 都成立——所以「减行最大值」不是数学要求，是**数值要求**：

```python
xb2 = BIG                       # BIG = X + 200，元素级最大 207.6
s_i = Σ exp(x)                  # ← 忘了减最大值
y   = exp(x) / s_i
```

`exp(207.6)` 在 f32 上**上溢成 `inf`**，于是 `y = inf / inf = NaN`。产出不是「误差大了一点」，是 **32000/32000 个输出位置全是 NaN**（`code/01_block_model.py` 的 `C.1`）。同一份输入交给**减了最大值**的版本，则全为有限值（`A.18`）。

### 0.2 本篇覆盖什么、不覆盖什么

| 覆盖 | 判据在程序里的落点 |
|---|---|
| 1D / 2D block id 网格与 `program_id` 的最快变化维 | `A.1` / `A.6` |
| `tl.arange` 的等价物（**半开区间** `[lo, hi)`）与边界掩码读写 | `A.2`–`A.4`、`A.24`、`C.14` |
| 块内 `sum` / `max` 归约、`axis` 与 `keep_dims` | `A.7`–`A.11` |
| fused online softmax（两趟，统计量在寄存器里滚动） | `A.12`–`A.17` |
| 行统计量归约的**相位与 barrier 账** | `B.1`–`B.5` |
| 「忘掉平移」与「掩码写窄 / 写宽」的失败行为 | `C.1`–`C.4` |

**不覆盖**（本篇与程序**都不声称**）：

- **真实的 Triton 编译器会为某个 kernel 选出什么 layout**——本机没有 `triton`、没有 GPU，这句无从验证。本篇只讲**语言层承诺了什么**，以及**这个承诺在哪里断裂**（§4.5）。
- **任何性能数字**。「2 个 barrier」「640 ops」是**模型值**，不是时间、不是带宽、不是指令周期。域 README 明写：本主题没有任何实测数字。
- **共享内存 bank 冲突的机制**——回链 `20-gpu-architecture/memory-system/03-shared-memory-bank-conflict`，本篇不重复推导。
- **在线 softmax 这个算法本身怎么推导**——回链 `40-kernels/softmax-normalization/02-online-softmax`；本篇只把它当作「块级程序 + 滚动统计量」的载体。

### 0.3 本篇在本域主线上的位置

本域的统一主线是「**你写的程序、编译器生成的东西、硬件执行的东西，是三个不同的程序**」。`triton` 主题守的是**第二条断层**：

> 你写的块级程序 → 编译器**替你选**的 layout 与掩码。

本篇守的是这条断层的**入口**：**你写的块级程序，连「一个元素归谁」都没有说，而后面所有代价都从这个未说的决定里长出来。** 下一篇（`02-masking-and-boundaries`）接着讲「你没说的另一半：边界怎么处理」。

## 1. 原理

### 1.1 块内张量：数据是 numpy 数组，但**属主映射**才是关键

迷你 Triton 里，一个「块内张量」就是 `Tile(data, dtype, layout)`——`data` 是 numpy 数组，`layout` 可以为 `None`。**只要不涉及线程，`layout` 就无意义**：`sum(axis=None) = 32896` 这个结果与布局无关。

但一旦问「这次归约要几次线程间通信」，就必须知道**每个元素归哪个线程**——而这正是 layout 的全部内容。本模型用四个参数描述它（`code/minitriton.py` 的 `Layout`）：

| 参数 | 含义 |
|---|---|
| `spt` — `sizePerThread` | 每个线程在每一维上**连续**持有的元素数 |
| `tpw` — `threadsPerWarp` | 一个 warp 在每一维上的线程数（**乘积必须是 32**） |
| `wpc` — `warpsPerCTA` | 一个 CTA 在每一维上的 warp 数 |
| `order` | 维度主次序，`order[0]` 是最快变化的维（minor-to-major） |

**★ 这四个参数的名字与顺序来自官方文档的逐字字面量**，出处是 Gluon「Tensor Layouts」那一页（`gl.BlockedLayout([1, 128], [32, 1], [4, 1], [0, 1])`），完整引文与页面路径见 `code/specs.py` 的 `ANCHORS["G3"]`/`["G4"]` 与 `03_layout.py`。

**★ 但「谁摊在哪一维」的规则是本模型自造的**（`MODEL_OWN_RULES["R1"]`）。本模型的贪心规则在 `shape=(128,128) / num_warps=4 / order=[0,1]` 上**逐参数复现**了官方那条字面量——**复现同一个结果不等于同一个规则**，这一点在 `03_layout.py` 的 `B.1` 里明写。

### 1.2 ★ 多步推导：为什么「减任意常数」对 softmax 无影响，而「减最大值」是**必要的**

**第一步（代数上的平移不变性）**。设行向量 `x ∈ R^n`，常数 `c`。则

```text
exp(x_i − c) / Σ_j exp(x_j − c)
  = [exp(x_i)·exp(−c)] / [Σ_j exp(x_j)·exp(−c)]
  = exp(x_i) · exp(−c) / ( exp(−c) · Σ_j exp(x_j) )
  = exp(x_i) / Σ_j exp(x_j)
```

`exp(−c)` 在分子分母**同时**出现、**同时**约掉。⇒ **结果与 `c` 无关**，这是一个**恒等式**，不依赖任何浮点假设。这就是为什么本篇的 `A.13` 可以用「减行内首元素」当参照——它**根本不求最大值**，却必须给出同一个答案（**本仓库程序在本机跑出的**最大绝对误差 **3.986e-08**，测的是本模型，两条参照与「减行最大」参照**逐元素同值到该量级**）。

**第二步（浮点上的必要性）**。恒等式说的是**实数**。f32 的最大有限值约 `3.4e38`，而 `exp(t)` 在 `t > 88.7` 附近就上溢成 `inf`。取 `c = 0`（即不减）：

```text
x 的元素级最大值 = 207.6
exp(207.6) ≈ 10^90  ≫  3.4e38   ⇒  f32 上是 inf
s_i = Σ exp(x_j)  ⇒  inf
y_i = inf / inf   ⇒  NaN
```

而且这**不是**「只有最大那个元素变 NaN」：`s_i` 变成 `inf` 之后，**每一个** `y_i = exp(x_i)/inf` 里的 `exp(x_i)` 也都是 `inf` ⇒ `inf/inf = NaN`。**32000/32000 全中**（`C.1`）。

**第三步（为什么「减最大值」是最省事的选择）**。任何常数都能救这一行，但 `c = max(x)` 是唯一能**同时**给出 `exp(x_i − c) ∈ (0, 1]` 的选择：最大项恰好是 `exp(0) = 1`，其余 ≤ 1 ⇒ 分母 `s_i ∈ [1, n]`，**既不上溢也不下溢**（下溢只会让极小的项变成 0，那是**正确的近似**）。`c = 首元素` 也合法（`x_i − c` 仍可能大到 200），只是**不保证**量程。

**第四步（在线化）**。fused softmax 要在**不物化 `N×N` 中间矩阵**的前提下算完。既然 `c` 可以事后改，就可以滚动地维护一对 `(m_i, s_i)`：

```text
m_new = max(m_old, max(blk))
s_new = s_old · exp(m_old − m_new) + Σ_blk exp(x − m_new)
```

`exp(m_old − m_new)` 这一项就是「把之前的账按新平移量折算」——它又一次用到第一步的恒等式。**这一步是本篇 B 类代价的来源**：每个列块一次归约、一个 program 一行做 4 次（`N=1000` 切成 4 个 256 的块）。

### 1.3 `axis` 与 `keep_dims` 的官方语义

`reduce_op(x, axis, fn, keep_dims)` 里三条约定（`code/minitriton.py`）：

- `axis=None` ⇒ 对**整个张量**归约，产出**标量**（这是 Triton 的默认口径）。`A.7` 断言它等于纯 Python 循环参照（`naive_reduce` 不调 numpy 的 `sum`）。
- `axis=k` ⇒ 只掉第 `k` 维。`A.8` 断言逐行和等于 `refs.row_sum_ref`（**另一条代码路径**：`math.fsum` 的标量循环）。
- `keep_dims=True` ⇒ 保留被归约的维（`(16,) → (16,1)`），`A.9` 钉住形状。

★ **一个容易写错的细节**：`tl.arange(lo, hi)` 是**半开区间**。`A.24` 断言 `arange(3, 7)` 给出 `[3,4,5,6]`（**不含 7**）；`C.14` 断言空区间直接抛 `ShapeError`。掩码算术 `offs < N` 与半开区间是**配套**的——把 `arange` 当成闭区间，边界就会多算一个元素。

### 1.4 ★ 本模型自造的 5 条规则（**不声称是 Triton 的规定**）

域 README 要求逐篇写明。完整对照表在 `code/specs.py` 的 `MODEL_OWN_RULES`，程序里由 `03_layout.py` 的 `A.7` 与 `01_block_model.py` 的 `C.8` 共同钉住（条数、两两不同、每条的 `covered_by` 标签真实存在、且没有一条是官方引文的字面量）：

| # | 自造的规则 | 为什么需要写出它 |
|---|---|---|
| R1 | **摊维规则**：`coalesced_layout` 的贪心（先把 32 个 lane 摊到 `order[0]`，再 `order[1]` …） | 它在两个具体形状上复现了官方字面量，但「复现同一个结果」不等于「同一个规则」 |
| R2 | **属主算法** `owner(i)` 与**精确切分**约束 `spt×tpw×wpc == shape` | 这是一条**简化**：真实 Triton 允许覆盖不足（要沿某维复制） |
| R3 | **转换定价公式**：不走共享内存 0 字节；走则 `2 × numel × esize` 字节 + 2 个 barrier | 官方只锚定「走共享内存」这个**机制**，没有给公式 |
| R4 | **三相归约的 ops 计数公式**（线程内 / `log2(tpw)` 步蝶形 / 跨 warp 的 `wpc × positions`） | 官方只锚定**相位的存在**，没有给计数公式 |
| R5 | **交叉点 `K*=2` / `K*=64`** | 它们是 R3/R4 的算术结果，换一个定价公式就换一组交叉点 |

## 2. 执行映射

### 2.1 一次 `tl.max(x)` 在模型里被拆成什么

`BLOCK_N = 256`、`num_warps = 4`、`spt = 2, tpw = 32, wpc = 4`（= 128 个线程，每线程 2 个元素）：

```text
归约：线程内 128 ops ｜ warp 内蝶形 5 步 / 640 ops ｜ 跨 warp 4 ops / 32B / 2 barrier
广播回：smem-broadcast（128 ops / 516B / 2 barrier）
每个列块：4 次 barrier ｜ 全行两趟共 32 次 barrier（4 个列块 × 2 趟）
```

**参与的部件**（名字都来自官方那一页的逐字引文，见 `specs.ANCHORS["G2"]`）：**`__shfl_xor_sync` 蝶形**（warp 内的 `log2(tpw) = 5` 步）、**leader warp**（跨 warp 汇总时挑一个 warp 当汇总者）、**共享内存**（跨 warp 的通道，也是「precious resource」那句的所指）。

**★ 一个必须说清的算术**：`positions_per_thread = numel / (n_threads × spt) = 256 / (128 × 2) = 1`——**每线程只留一个部分和**。所以跨 warp 的字节数是 `2 × wpc × positions × esize = 2 × 4 × 1 × 4 = 32 B`。这个 `2 ×` 是「写一遍 + 读一遍」（`R3`）。

### 2.2 同一行源码，两种块大小的对照

| 配置 | `spt / tpw / wpc` | 线程内 | 蝶形 | 跨 warp | 共享内存 | barrier |
|---|---|---|---|---|---|---|
| `[BLOCK_N=32]`, 1 warp | `1 / 32 / 1` | 0 | 160 | 0 | 0 B | **0** |
| `[BLOCK_N=256]`, 4 warp | `2 / 32 / 4` | 128 | 640 | 4 | 32 B | **2** |

这两行是 `B.1` / `B.2`。**它们之间的差别不在块大小本身，在 `wpc`**——这一点是下一篇（`04-reduction-cost`）的主结论，本篇只把它的入口打开。

### 2.3 代价模型读的是什么，读不到什么

代价模型只做三件事：**按布局参数算属主摊布**、**按自定公式算三个相位的 ops**、**按自定公式算共享内存字节与 barrier 数**。

**它读不到**：真实编译器会不会**融合**两次相邻的归约（本模型按「每次归约单独付账」计，`B.3` 明写这个假设）、会不会**向量化**访存、会不会把 `exp` 换成更快但精度更低的指令、真实硬件的 `shfl` 吞吐与共享内存带宽。⇒ **本模型只能说「相位存在与否」和「相对量的方向」，不能给出任何绝对时间**。

## 3. 实现与演进

### 3.1 迷你引擎的骨架（`code/minitriton.py`，1113 行）

| 层 | 内容 |
|---|---|
| 缓冲 | `Buff`——全局内存 + 一段**毒区**（越界索引被显式接住，见 `02-masking-and-boundaries`） |
| 块内张量 | `Tile` / `arange` / `const` / `full` / `offs_add` / `less_than` |
| 访存 | `load` / `store`（带掩码，`strict=True` 时越界**抛错**） |
| 布局 | `Layout`（四个参数）、`SliceLayout`、`Conversion`、`BroadcastConversion`、`coalesced_layout` |
| 归约 | `ReducePlan`（三相分解）与 `simulate_three_phase_reduce`（**真的按三相跑一遍**） |
| 图与 pass | `Op` / `layout_pass`（前向布局推断、插入转换）/ `graph_execute`（忽略布局地跑 numpy） |
| 参照 | `naive_reduce`（纯 Python 循环，不调 numpy 的 `sum`） |

**★ 「三相分解」不是本模型的发明**：它的官方出处是 Gluon「Tensor Layouts」页的逐字引文（`specs.ANCHORS["G2"]`）：

> 「The compiler will generate **butterfly shuffles** to reduce within each warp, then pick a **leader warp** to reduce the remaining 4 values per row through **shared memory**.」

**★ 那句里的「4」不能读成常数**——它是**那个例子**的位数（`num_warps=4`）。本模型 18 组配置里 `wpc` 取过 `1 / 2 / 4 / 8` 四个值（`04_warp_reduce.py` 的 `A.6` 钉住这一点）。

### 3.2 `refs.py`：A 类断言的**全部意义**都在这个文件里

`code/refs.py` 里**没有一个函数调用 `minitriton`**，也**不 import 它**：全部是纯 Python 标量循环 + 标准库 `math`，**故意不向量化**。这样它与「解释器」、与 numpy 构成**三条互不重叠的代码路径**。

**判据（可证伪）**：把 `refs.py` 与源**故意改成不一致**，A 类断言必须失败。程序里做了**三处**：

| 落点 | 做法 | 结果 |
|---|---|---|
| `C.5` | 把**被测解释器**的 `exp` 换成 `exp(x+1)` | 最大绝对误差从 3.986e-08 涨到 **3.665e-01** ⇒ `A.12` 失败 |
| `C.6` | 把 `refs.softmax_rows_ref_shifted` 的**分母少累加一项** | 误差 **1.672e-03** ⇒ `A.13` 失败 |
| `C.13` | 把 `refs.row_sum_ref` 改成**少累加一项** | **16 行全部**超差 ⇒ `A.8` 失败 |

★ `C.5` 的构造方式值得单独说：**不能用「换一个平移量」来造假的反例**——softmax 对任意常数平移**不变**，换平移量在数学上仍然是**对的**（第一版就是这么写的，`C.5` 恒过）。要构造真的反例，必须改**算错的那一步**。

### 3.3 三次迭代：每一版都被一次「断言不可证伪」逼出来

| 版本 | 做法 | 被什么逼着改 |
|---|---|---|
| v1（探针） | 19 条断言，全部是「打印 + 单点比较」 | 断言太薄：**74 条里几乎没有一条问「改错会不会报错」** |
| v2 | 补 A/B/C 三类骨架 + `specs.py` 锚点 | `specs` 的 `covered_by` 只查了 2 个注册表，`MODEL_OWN_RULES` 那 24 个标签**一个都没被校验**（变异测试全报漏网） |
| v3（本篇） | **四个注册表全查** + `META_DEFECTS ↔ meta_falsify.DEFECTS` 双向绑定 + `FORBIDDEN_TOKENS ⊆ NOT_FIRST_HAND` | 两处断言被变异测试证明是**恒真**的（`MODEL_OWN_RULES[*].rule` 的「打印一致性」、`covered_by` 的存在性）——见 §4.4 |

### 3.4 ★ 元检验：对 P12 的**正面回答**（`code/meta_falsify.py`，`make falsify`）

**自造 DSL 的结构性危险**是：**被测对象和参照实现是同一段代码**。如果文档说「这样写会慢」，而程序用自己写的模型「证明」了它慢，那**什么也没证明**。

对策：**往引擎里注入缺陷，断言编号程序必须退出非 0**。**本机跑出的结果**（`make falsify` 的原始输出见程序）：

| 注入的缺陷 | 锚点（`minitriton.py` 中恰好出现 1 次的原文） | 应该失败 | **本机跑出的结果** |
|---|---|---|---|
| D1 掩码被忽略（读了越界位置） | `vals = np.where(m, vals, fill)` | `02` | 退出 1 ✓ |
| D2 所有布局转换都判成免费 | `return bool(np.array_equal(...owners...))` | `03` | 退出 1 ✓ |
| D3 跨 warp 相位被抹掉 | `self.cross_warp = wpc > 1` | `03` / `04` | 退出 1 ✓ |
| D4 三相归约的跨 warp 相位写错 | `combined = smem.sum(axis=axis, keepdims=True)` | `04` | 退出 1 ✓ |
| D5 `coalesced_layout` 把 lane 摊到错的维上 | `rem_lane, rem_warp = 32, num_warps` | `03` | 退出 1 ✓ |

**且不打补丁时四个程序全过**（基线）。⇒ 「解释器与被测对象是同一个东西」这个质疑，被这五条反证正面回答。

★ **元检验自己也会退化成空操作**：锚点一旦漂移（`src.count(anchor) != 1`），补丁打不上、程序当然全过。所以 `01_block_model.py` 的 `C.13` 用 `ast` 读 `meta_falsify.DEFECTS`、逐条核对「锚点在 `minitriton.py` 里恰好出现 1 次」，`C.9` 再把 `specs.META_DEFECTS` 与 `meta_falsify.DEFECTS` 绑在一起（两份账漂开就报错）。

## 4. 踩坑与失败模式

### 4.1 块级程序看起来像「对整块操作」，于是忘了块里还有线程

- **现象**：`tl.max(x)` / `tl.sum(x)` 读起来是一次标量操作；写块级程序时完全不会想到 barrier、shuffle、共享内存。
- **成因**：**块内张量的属主是由 layout 决定的，而 layout 不出现在源码里**。本模型里同一行 `reduce(axis=1)` 在两种布局下分别是「蝶形 + 跨 warp 经共享内存（2 barrier）」与「不需要任何线程间通信（0 barrier）」（`03_layout.py` 的 `B.3`）。
- **规避**：把「跨线程摊开多少」（`tpw[axis] × wpc[axis]`）当成一个**显式的量**来读；它的值是 `1` 时归约是纯线程内的事，大于 1 就会有 shuffle，`wpc[axis] > 1` 时**必然**有共享内存往返。
- **为什么它值得单列**：这是本主题的**断层本身**——源码到成本的中间那一步（谁摊在哪一维）**不在源码里**。

### 4.2 平移量减错了 / 忘了减

- **现象**：`x ≈ 200` 的输入上产出 **32000/32000 个 NaN**（不是「误差大」）。
- **成因**：`exp(207.6)` 在 f32 上上溢成 `inf` ⇒ `s_i = inf` ⇒ **每一项**都是 `inf/inf = NaN`（§1.2 第二、三步）。
- **规避**：第一趟就滚动维护 `m_i`；`exp(m_old − m_new)` 这一项是**折算**，不能省。
- **★ 为什么会话里有一条警告，而它是证据不是噪音**：直接跑程序（不加 `-W ignore::RuntimeWarning`，`Makefile` 里加了）会在 `stderr` 上看到两条：
  `minitriton.py:299: RuntimeWarning: overflow encountered in exp` 与
  `minitriton.py:275: RuntimeWarning: invalid value encountered in divide`。
  **它们正是「上溢 ⇒ `inf/inf` ⇒ NaN」这条链的两端**：上溢在 `exp`，`NaN` 产生在除法。
  ★ 顺手把警告压掉，就等于把这条链的**唯一外部信号**压掉——本仓库的做法是**只在 `make run` 里压**
  （让输出可读），**而在正文里把它当证据引**。
- **★ 为什么它藏得住**：这个错误在**一般的输入**上完全看不出来——随机标准正态输入的最大值只有 4.9（`program 0` 的 `m_i = 4.947`），`exp(4.9)` 远不到上溢。**测试数据的量程决定了这个 bug 会不会现身。**

### 4.3 把 `arange` 当闭区间

- **现象**：边界处的掩码少算 / 多算一个元素。
- **成因**：`tl.arange(lo, hi)` 是**半开区间**（`A.24` 断言 `arange(3,7) == [3,4,5,6]`）；掩码 `offs < N` 依赖这个约定。
- **规避**：`C.14` 把「空区间直接抛 `ShapeError`」钉成契约——**别让一个退化区间静默产出空张量**。

### 4.4 ★ 两处「形式正确但不可证伪」的断言（被变异测试抓出来的）

- **现象**：`make mutate` 报出 `MODEL_OWN_RULES[*].rule` 5 条、`covered_by` 24 条**全部漏网**。
- **成因**：① 第一版的「规则常量出现在打印列表里」是**恒真**的——打印列表由**同一个常量**拼出来；② `covered_by` 的标签存在性校验只覆盖了 `ANCHORS` / `LOAD_QUOTES` **两个**注册表，`MODEL_OWN_RULES` 与 `MODEL_SIMPLIFICATIONS` 那 24 个标签**无人校验**。
- **规避**：① 把 `A.7` 改成只断言可判定的部分（条数、两两不同、非官方引文字面量、`covered_by` 非空），并把「散文正文不是判据载体」**如实写在注释与漏网清单里**；② `C.8` 扩到**四个注册表**（现在校验 **33 个**标签），`C.9` 把 `META_DEFECTS` 与 `meta_falsify.DEFECTS` 双向绑定。
- **一般化**：**「我打印了它」不是断言**；断言必须有一个**不由被测常量自己提供**的判据。

### 4.5 ★ 那个「别处」在哪里

本域主线说：**抽象层的价值与它的泄漏是同一件事的两面**。那么这一篇里，代价被藏到了哪里？

> **被藏进了「块」这个词里。**
>
> `tl.max(x)` 里的 `x` 被读成「一块数据」，而「一块数据」暗示的是**一个整体**。可是块内张量的每一个元素**归 32×num_warps 个线程里的某一个管**——这件事在源码里**一个字符都没有**。于是归约的代价（shuffle 步数、共享内存往返、barrier）**不是从你写的那一行长出来的**，而是从一个你没写的决定长出来的。
>
> **这就是那个「别处」：代价不在你的程序里，它在编译器的布局决策里。** 而你唯一的输入是**几个关键字参数**——`load` 的 `order`、块大小、`num_warps`。下一篇（`03-layout-and-leaks`）会把这一点推到极端：**同一个 `a * b`，两次 `load` 只差一个 `order`，代价从 0 变成 131072 字节的共享内存往返**。

## 5. 判据表

> ★ 按本域约定（域 README），§5 的位置与编号保留，内容**不是实测数据**而是**判据表**。
> ★★ **本主题无实测，只有模型值**——本机**无 GPU**、`triton` 包**全机器不存在**。
> 下表的每一行都是「**可机械判定的条件 → 结论**」；凡带数字的结论都是**本模型算出来的**，
> **不是硬件读数、不是真实 Triton 编译器的输出**。

| # | 可机械判定的条件 | 结论 |
|---|---|---|
| 1 | `axis=None` 的块内归约 | 产出**标量**，等于纯 Python 循环参照（`A.7`） |
| 2 | `keep_dims=True` | 保留被归约的维（`(16,) → (16,1)`，`A.9`） |
| 3 | `tl.arange(lo, hi)` 且 `hi > lo` | 半开区间，`hi` **不在**结果里（`A.24`） |
| 4 | `tl.arange(lo, lo)` | **抛 `ShapeError`**，不静默给空张量（`C.14`） |
| 5 | grid 任一维 ≤ 0 | **抛 `ShapeError`**（`C.15`） |
| 6 | 掩码写**宽**（掩了越界位置为真） | **抛 `OutOfBoundsError`**——越界被检出（`C.3`） |
| 7 | 掩码写**窄**（把有效位置掩掉） | **不抛错、不越界，静默算成 `other`**；误差从 2.384e-07 涨到 **4.499e+00**（`C.2`）。★ 只有 A 类逐元素比对能抓它 |
| 8 | 去掉**写**掩码 | **抛 `OutOfBoundsError`**（`C.4`） |
| 9 | 行最大平移被省略，且输入元素级最大 > 88.7 | **全部输出为 NaN**（32000/32000，`C.1`）；补上平移后全为有限值（`A.18`） |
| 10 | online softmax 输出与三种参照比对 | 误差 ≤ 1e-05，且**同时**等于「减行最大」「减行首元素」「按块递推」三条（`A.12`–`A.14`） |
| 11 | 输出逐行和 | `|Σ − 1| ≤ 1e-05`（`A.15`） |
| 12 | 滚动统计量 `m_i` | **精确等于**行最大（`A.16`，浮点相等而非近似） |
| 13 | 滚动统计量 `s_i` | 在 `n×2⁻²³×64` 的相对误差上界内（`A.17`，容差按机器 epsilon 推） |
| 14 | 块布局的 `wpc[axis] == 1` | 跨 warp 相位**不存在**：0 barrier / 0 B（`B.2`） |
| 15 | 块布局的 `wpc[axis] > 1` | 跨 warp 相位**存在**：2 barrier（`B.1`） |
| 16 | 归约结果要广播回原形状，且 `wpc[axis] > 1` | **额外** 2 barrier / `positions × esize × (1 + n_threads)` 字节（`B.5`） |
| 17 | 把 `refs.py` 或被解释器**故意改成不一致** | A 类断言**必须失败**（`C.5` / `C.6` / `C.16`）——这是 A 类判据的**存在理由** |
| 18 | 元检验的锚点在 `minitriton.py` 里出现次数 ≠ 1 | `C.10` 报错——元检验会**静默退化成空操作** |
| 19 | 往 `minitriton.py` 注入 D1–D5 任一缺陷 | 对应的编号程序**必须退出非 0**（`make falsify`） |
| 20 | 在 `specs.py` 里加一条没人认领的锚点 | `C.7` 报错（四篇 `COVERED_ANCHORS` 的并集必须**恰好**等于全集） |
| 21 | `covered_by` 指向一个不存在的断言标签 | `C.8` 报错（四个注册表共 **33** 个标签逐个校验） |
| 22 | 某个 `.py` 里出现 `import triton` / `torch` / `numba` / `cuda`（含动态导入） | `C.12` 报错（扫**九个** .py，含 `specs.py`） |

**闭合性检查**：读完这张表能判完「一次块内归约在这个模型里的相位、barrier、共享内存字节，以及三类失败模式会不会被检出」。**唯一的表外输入**是「真实的 Triton 会为这段代码选什么 layout」——本机**无从验证**（无 `triton`、无 GPU），本表**不主张**任何关于它的结论。

## 6. 小结与延伸

- ★★ **「平移量选哪个不影响结果」是可证伪的**：同一份输出**同时**等于「减行最大」与「减行内首元素」两条独立参照（后者**根本不求最大值**），32000 个位置的最大绝对误差都是 **3.986e-08**。
- ★★ **忘掉平移 ⇒ 32000/32000 个 NaN**。而且**随机正态输入看不出这个 bug**——`program 0` 的行最大只有 `4.947`，`exp(4.9)` 远不到上溢。**测试数据的量程决定了这个 bug 会不会现身。**
- ★★ **`[BLOCK_N=256]` 的行归约是三个相位**：128 ops / 640 ops（5 步蝶形）/ 4 ops + 32 B + 2 barrier；广播回**再加** 2 barrier。一行 1000 元素的 softmax 统计量两趟合计 **32 次 barrier**。**全是模型值，无一条实测。**
- ★★ **元检验（`make falsify`）是本主题对「自造真值」的正面回答**：注入 D1–D5，**5 种缺陷全部被断言抓住**，且**不打补丁时四个程序全过**。
- ★ **`refs.py` 的独立性是 A 类的全部意义**：三处「故意改成不一致 ⇒ 断言必须失败」的构造（`C.5`/`C.6`/`C.16`）把这句话变成了可证伪的判据，而不是一句声明。
- ★ **两处「形式正确但不可证伪」的断言被变异测试抓出来并修掉**（§4.4）——这是 `make mutate` 存在的理由：**捕获率 56.7%、漏网 94 条**，漏网里最大的一类是散文正文（`note` / `rule` / `basis`），它们**不是行为判据的载体**，这一点如实列出而不粉饰。
- ★ **本模型有一条已记录的简化**：`spt×tpw×wpc == shape`（精确切分），而真实 Triton 允许覆盖不足。**这不改变归约的结论**（结论只依赖 `wpc[axis]`）。

**延伸**：

- 边界掩码的五种写法、`other` 的官方语义（「未定义」，**不是 0**）、以及「掩码写窄不会抛错」：`30-gpu-programming/triton/02-masking-and-boundaries`。
- layout 转换的代价与那个「离得很远的关键字参数」：`30-gpu-programming/triton/03-layout-and-leaks`。
- 三相归约的完整代价表与 `wpc[axis] > 1` 这条判据：`30-gpu-programming/triton/04-reduction-cost`。
- 在线 softmax 这个算法本身的推导：`40-kernels/softmax-normalization/02-online-softmax`。
- 归约算法与 warp 级原语：`40-kernels/reduce-scan/01-reduction-algorithms`、`.../02-warp-shuffle-and-hierarchy`。
- Triton 代码在编译流水线里的位置（本篇**不讲**编译器内部架构）：`60-compilers/inductor/03-codegen`。
- 「先有独立参照再谈验证」的方法论：`45-kernel-lifecycle/implementation/01-reference-first`。

## 编译与运行

```bash
cd 30-gpu-programming/triton/code
make run                           # 跑本主题全部四个编号程序；本主题这一篇是第一个
make 01_block_model                # 只跑本篇对应的那一个
make lint                          # AST 检查：不 import torch / triton / numba / cuda（含 specs.py）
make falsify                       # ★ 元检验：注入 D1–D5，断言编号程序必须失败
make mutate                        # 叶子级变异测试（非编号工具，**不进 `make run`**；跑几分钟）
make anchors                       # ★ 联网：从官方文档再抽取词频与 7 条锚点，与 specs.py 逐字比对
```

本篇对应的编号程序是 **`code/01_block_model.py`**。它**只用 `python3` + `numpy`**，**不 import** 任何 GPU / 框架依赖，**不编译、不执行任何 GPU 代码**，也**不需要网络**（`make run` 全程离线）。它**不依赖外部源码树**——本主题的真值是 `refs.py` 的独立参照、官方文档的逐字引文（`code/specs.py`）、以及元检验。

**四类数字的口径**（`CONTRIBUTING` 第 63 行的四分法，本域照用）：

| 类 | 本篇里的例子 | 说明 |
|---|---|---|
| **本机实测值** | 2.384e-07、3.986e-08、32000/32000、`m_i = 4.947357654571533` | **本仓库程序在本机跑出的解释器结果**；测的是**本模型**，不是 Triton、不是硬件 |
| **模型值（推导）** | 128 / 640 / 4 ops、32 B、2 barrier、32 次 barrier | **本模型的自定公式算出来的**，不是实测 |
| **引用值** | 官方文档的逐字引文（`specs.ANCHORS`） | 本仓库**不复核**其测量口径；`0.774 TB/s` 那条只有 `03` 用到 |
| **规格值** | 无 | 本主题**没有**规范级的常量（`triton` 包不在本机，无从引它的实现常量） |

**本机跑出的结果**（测量条件：`python3 3.13.5` + `numpy 2.4.4`，Apple M1 Pro，**无 GPU**、**无 `triton`**）：

```
汇总: A 25/25 · B 5/5 · C 16/16 ⇒ 断言 46 条，通过 46 条，失败 0 条
```

（本主题四个程序合计 **140 条断言、140 条通过、0 条失败、0 条跳过**——逐字汇总行见 `make run` 末行：
`断言 140 条 · 通过 140 条 · 失败 0 条 · 跳过 0 项`。四篇的分布是
**A 63 / B 31 / C 46**：`01` 46 条、`02` 28 条、`03` 38 条、`04` 28 条。）

**三类断言**（全部在程序内部，失败即以非零码退出）：

- **A 类（25 条）**——参照走**不同代码路径**：网格 pid 顺序 vs 手工枚举（`A.1`/`A.6`）、逐元素比对 `refs.masked_block_ref`（`A.2`/`A.3`）、块内归约 vs `naive_reduce` 与 `refs.row_sum_ref`（`A.7`/`A.8`）、softmax vs **三条**独立参照（`A.12`–`A.14`）、滚动统计量 vs 行最大与 epsilon 推导上界（`A.16`/`A.17`）。★ 另有**三处故意的失配**（`C.5`/`C.6`/`C.16`）证明这些参照真的在参与判断。
- **B 类（5 条）**——行统计量归约的相位分解（128/640/4 ops、32 B、2 barrier）、广播回代价、两趟合计 32 次 barrier、2D 网格槽位。**全部是模型值。**
- **C 类（16 条）**——`C.1` 无平移 ⇒ 32000/32000 NaN；`C.2` **掩码写窄不抛错**、`C.3` 写宽抛错；`C.4` 去写掩码抛错；`C.5`–`C.7` 三处「故意改成不一致」；`C.11` 禁用清单扫描（**自指陷阱**：清单放在 `specs.py`，扫描排除它）；`C.9`/`C.13` 元检验锚点与两份账的绑定、`C.10` 禁用清单与字面量投影的一致性、`C.12` import 扫描、`C.14`/`C.15` 形状契约、`C.16` 第二处 refs 独立性。

> ★ **程序不修改磁盘上的任何文件**：反例都在**内存里的字符串 / monkeypatch** 上构造，且每处都在 `try/finally` 里恢复。

## 参考

**本仓库程序（可复现）**

- `code/01_block_model.py` —— 本篇的编号程序（A/B/C 三类断言在程序内部，46 条）。
- `code/minitriton.py` —— 非编号辅助模块：迷你 Triton 引擎（1113 行）。**不是 Triton 的实现**，`triton` 包在本机不存在。
- `code/refs.py` —— 非编号：**独立参照**（纯 Python 标量 + `math`，不 import `minitriton`）。
- `code/specs.py` —— 非编号：7 条 A 级官方锚点（逐字引文 + 页面路径 + 小节名 + `covered_by`）、词频表、5 条自造规则、3 条已记录简化、禁用清单。
- `code/meta_falsify.py` —— 非编号：**元检验**（`make falsify`）——注入 D1–D5，断言编号程序必须失败。
- `code/mutate.py` —— 非编号诊断工具（`make mutate`）：叶子级变异测试 + 对照组。

**外部文档（A 级：官方文档站点逐字引文，可 `make anchors` 复算）**

- Gluon「Tensor Layouts」：`https://triton-lang.org/main/getting-started/tutorials/gluon/layouts.html`
  —— 锚点 `G1`（转换可能很贵、跨 warp 要用共享内存）、`G2`（蝶形 + leader warp）、`G3`（只重排线程内寄存器的转换是免费的）、`G4`（内维不连续 ⇒ 没有合并访存）、`G5`（先转换再归约通常更贵；★ 末词 `reeduction` 是**官方页面的拼写错误**，逐字保留）。
- `tl.load` 的 API 页：`https://triton-lang.org/main/python-api/generated/triton.language.load.html` —— 锚点 `L1`/`L2`（`other=None` ⇒「未定义」；掩码为假 ⇒ 不读那个地址）。

> ★ **这两条是本主题唯一引为「官方规定」的东西**，全部逐字保存在 `code/specs.py` 里，并由 `make anchors` 从**实际下载的页面**里按写明的口径**再抽取一遍**、逐字比对（本机跑出：**5 个量 + 7 条锚点的全部 `must_contain` 逐字命中**）。
> ★ **不引用**的（探针自报「未一手复核」，见 `specs.NOT_FIRST_HAND`）：`ReduceOpToLLVM.cpp` / `TritonOps.td` / `TritonGPUOps.td` / `Coalesce.cpp` 的**行号**；任何 release 版本号或 commit hash；**MAPL 2019 论文的词频统计**（本仓库**没有**复核那份 PDF，所以本主题**不使用**任何论文词频结论）；`60-compilers` 计划里钉的 Triton 版本号（那是**知识库自己的记录**，不是本次核实的）。`code/01_block_model.py` 的 `C.6` 扫**除 `specs.py` 以外**的全部源码确认这些字面量**一个都没出现**。

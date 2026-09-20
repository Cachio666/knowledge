---
title: 层次化 tiling：CTA = 指令形状 × 乘数，以及被抹掉的那一维
slug: cutlass-hierarchical-tiling
domain: 30-gpu-programming
topic: cutlass
level: 3
difficulty: advanced
prerequisites:
  - 30-gpu-programming/cutlass/01-layout-algebra
  - 20-gpu-architecture/tensor-core/01-mma-primitive
tags: [cutlass, tile-description, wgmma, sm90, kernel-generation, scheduling]
related:
  - 30-gpu-programming/cutlass/03-swizzle-and-collective
  - 40-kernels/gemm/01-naive-gemm-and-roofline
  - 40-kernels/gemm/02-shared-memory-tiling
  - 20-gpu-architecture/architecture-evolution/01-capability-breakpoints
code:
  - code/02_tiling_decompose.py
status: stable
updated: 2026-09-20
---

# 层次化 tiling：CTA = 指令形状 × 乘数，以及被抹掉的那一维

> **一个 GEMM 的 tile 要在三层同时成立（CTA / warp / 指令），而 CUTLASS 的生成器
> 只把前两层的形状写进 kernel——第三层被它自己抹掉了。**
>
> 1. ★★ **`CTA shape = 指令形状 × 乘数`，逐元素**（`sm90_utils.py:385–396`）。
>    本仓库自己重跑了一遍这个枚举循环，在 **13 个全局档位**上重算 **549** 个不同的
>    CTA 形状，与库发射出的 **1557** 个 `TileDescription` **逐条对上**。
> 2. ★★ **生成器故意把 `instruction_shape` 置零**（`sm90_utils.py:378–383`，作者逐字注释）。
>    后果是**实测出来的**：在全局档位 **39** 上发射 **327** 个 `TileDescription`，
>    却只有 **93** 个**不同的 `procedural_name()`**，**最多的一个名字出现 6 次**；
>    而其中的 **16/112** 个 CTA 形状**有不止一组 (指令形状, 乘数) 原像**。
> 3. ★★ **形状表里的 value 是「生成档位」，不是指令数、不是代价、不是权重。**
>    判据是可机械核的：**档位 0 的入选集恰好等于 `value == 0` 的键集**。
> 4. ★★ **`procedural_name()` 在 `min_compute` 从 80 变到 90 时换了整套命名方案**，
>    而**同一棵树里** `ArchitectureNames` 只到 **90**，`include/cute/arch/` 却已有
>    sm100 / sm107 / sm120。⇒ 问「CUTLASS 支持哪些架构」**必须指明问哪个子系统**。
>
> ★★ **本篇的反直觉量化结论**（读完这一篇之前你不会相信的那一条）：
> **`procedural_name()` 在 `min_compute` 从 80 变到 90 时换掉了整套命名方案** ——
> `128x128_32x3`（不含 cluster）直接变成 `128x128x32_2x1x1_3`（含 cluster），
> **而这只是同一个对象的两个字段各改了一个数**。更反直觉的是下游：
> 这个换了方案的**同一个名字**，在全局档位 39 上**会被 327 条记录里的 6 条共用**
> —— **名字标识的不是 kernel，是一个 tile 形状**（§4.1）。

> ★★ **真值标注（本主题特有，本篇逐处都要带）**：
>
> 本主题的数字全部来自**在本机跑 CUTLASS 4.8.0 自带的 Python 参考实现**
> （`python/pycute/` 与 `python/cutlass_library/`），**不是**从 GPU 上测的，
> **也不是**本仓库的推导。**测量条件是 CUTLASS 的版本，不是 GPU 型号。**
>
> 本篇的数字有两类：**本机实测值**（真跑 `cutlass_library` 的生成器得到的枚举结果）
> 与**推导值**（本仓库自己重跑那套枚举循环、以及两者的对拍统计）。
> **一条「规格值」都没有**——CUTLASS 的源码是**实现**，不是规范。
> ⇒ **本篇没有任何性能数字**：没有吞吐、没有占用率、没有延迟。
> 「生成器枚举了哪些 tile」与「这些 tile 在硬件上跑多快」是**两个不同的问题**，
> 本篇只回答前者。后者的机制在 `20-gpu-architecture/tensor-core/01-mma-primitive`。

> ★ **本域的统一主线**：「**你写的程序、编译器生成的东西、硬件执行的东西，
> 是三个不同的程序**」。本篇是它在 `cutlass` 主题上的**第二次穿越**：
> **你在模板参数里写的是 tile 形状，生成器把它拆成 (指令形状, 乘数, cluster)，
> 而 kernel 里最终只剩下一个 `threadblock_shape`。**

## 0. 背景与动机

### 0.1 具体对照场景：三层 tile，任何一层选错另两层被迫改

一个 SM90 的 fp16 GEMM，tile 要在**三层**同时成立：

| 层 | 形状（例） | 谁决定它 |
|---|---|---|
| **CTA**（线程块） | `128 × 128 × 64` | 你选的（它决定 SMEM 占用、wave 数量）|
| **warpgroup**（4 个 warp = 1 warpgroup）| `128 × 128 × 64`（CTA 里只有一个 warpgroup）| **写死的 `warp_count=[4,1,1]`** |
| **指令**（wgmma） | `64 × 16 × 16` | 硬件形状表 ∩ 你选的档位 |

三层之间的约束是**乘性**的：`CTA = 指令形状 × 乘数`（逐元素）。
所以：

- 把指令形状从 `64×16×16` 换成 `64×8×16`（N 减半），**CTA 的 N 要翻倍**才能保持同样的 tile；
- 而 CTA 的 N 有**上限 256**（`sm90_utils.py:352–353`），翻不动的时候只能**降 tile**；
- 降 tile 之后 SMEM 占用变了，**stage 数**跟着变——而 stage 数是**下游**算的，
  `TileDescription` 里那个 `stages` 字段在生成阶段**恒为 0**（§4.3）。

⇒ **三层不能分别选。** 想枚举出「所有合法的组合」，就必须**同时**枚举指令形状、
乘数、cluster 三样——这正是 CUTLASS 生成器做的事。

### 0.2 本篇覆盖什么、不覆盖什么

本篇落在 `code/02_tiling_decompose.py` 的 **29 条断言**里（A 9 / B 14 / C 6）。
它跑的是 `cutlass_library` 的**纯 Python 生成器**（12 个子模块全部导入成功）。

**覆盖**：`TileDescription` 与 `MathInstruction` 的**真实签名**、
「CTA = 指令形状 × 乘数」在**真实枚举结果**上的逐条验证、**档位（level）的四级解码**、
形状表的 value 到底是什么、以及**生成器抹掉指令形状**这一处一手泄漏。

**不覆盖**：

- **`threadblock_shape` 选多大才快** —— 那是 `40-kernels/gemm/02-shared-memory-tiling`
  与 `50-performance` 的问题。本篇只说「生成器枚举了哪些」，不说「哪个好」；
- **wgmma 指令本身的语义**（它怎么做 `m64n16k16` 的乘法）——
  `20-gpu-architecture/tensor-core/01-mma-primitive` 与 `.../03-wgmma-and-pipeline`；
- **生成器生成的 C++ 代码长什么样** —— 本机无 `nvcc`，**一个字都验证不了**，
  因此不写。本篇只碰生成器的**数据**（枚举出的 `TileDescription` 列表），不碰它吐出的文本。

### 0.3 统一主线：那个「别处」在哪里

> **契约在哪里断裂 → 断裂的表现是什么 → 怎么发现它 → 怎么绕开。**

**契约**：你填一个 `threadblock_shape`，期望的是「这一个 tile 配置会被生成出来」。

**断裂**：`threadblock_shape` **不携带它是怎么来的**。
同一个 `(128, 16, 64)` 可以由 `((64,8,16), (2,2,4))` 与 `((64,16,16), (2,1,4))`
**两组完全不同的来源**产生——而生成器把 `instruction_shape` **置零**，
于是发射出的两条记录**在每一个可见属性上都一样**（§4.1）。

**代价被藏到了哪里？**

> **它被藏进了「名字」里。** `procedural_name()` 只含 `threadblock_shape / cluster / stages`，
> **不含元素类型、也不含指令形状**。于是三条不同 dtype 的 `MathInstruction`
> （f16→f16 / f16→f32 / bf16→f32）**共用一个 CTA 形状时，名字完全相同**。
> 再往后一步，那个「别处」是**下游**：`stages` 在生成阶段是 0，
> 真正的流水级数要等**知道 SMEM 容量与数据类型**之后才算得出来——
> 也就是说，**这个名字所标识的 kernel，在生成阶段还没有定型**（§4.3）。

## 1. 原理

### 1.1 三层形状的乘法关系

`sm90_utils.py:385–396` 逐字：

```python
tile_desc = TileDescription(
    threadblock_shape=[
        math_inst.instruction_shape[0] * mma_mul[0],
        math_inst.instruction_shape[1] * mma_mul[1],
        math_inst.instruction_shape[2] * mma_mul[2]
    ],
    stages=0,
    warp_count=[4, 1, 1],
    math_instruction=math_inst_stub,
    min_compute=90,
    max_compute=90,
    cluster_shape=cluster_size)
```

**三点必须一起读**：

1. **逐元素乘法**（不是点积、不是外积）：CTA 的每一维各自 = 指令形状的对应维 × 乘数的对应维；
2. **`stages=0`** —— 流水级数**不在这里**决定（§4.3）；
3. **`warp_count=[4, 1, 1]` 是写死的**：SM90 的 CTA 一律是 4 个 warp 组成 1 个 warpgroup，
   **不随形状变**。本仓库在 **全部 603** 个发射结果上逐条核对（`B.5`，违例 0）。

多步推导（一个具体例子）：

```
指令形状 = (64, 16, 16)     乘数 = (2, 1, 4)
CTA[0] = 64 × 2 = 128
CTA[1] = 16 × 1 = 16
CTA[2] = 16 × 4 = 64
⇒ threadblock_shape = (128, 16, 64)
```

**上限**：`sm90_utils.py:351–353` 判的是 `cta_shape[i] > 256`，
即**含 256 合法**。★ 而它上面那行注释写的是 `<256, 256, 256>`——
**注释与代码不一致**。本库按**代码**断言（`B.6`）：
CUTLASS 的源码注释是**某个实现的作者写的**，不是规范（`CONTRIBUTING` 第 56–60 行）。

### 1.2 `level` 是一个**四位十进制数**，不是「难度」

这是本篇最容易被误读的一处（P9「同名不同指」）。`sm90_utils.py:90–103`：

```python
def get_wgmma_level_from_global_level(global_level):  return global_level % 10
def get_mma_level_from_global_level(global_level):    return (global_level // 10) % 10
def get_cluster_level_from_global_level(global_level):return (global_level // 100) % 10
def get_pruning_level_from_global_level(global_level):return (global_level // 1000) % 10
```

⇒ 一个 `global_level` 的**每一位**控制一个维度：

| 位 | 名字 | 控制什么 |
|---|---|---|
| 个位 | wgmma level | **指令形状**的入选量（形状表的 `min_level` 与它比）|
| 十位 | mma level | **乘数**的入选量（`SM90_MMA_MULTIPLIERS` 的 `min_level`）|
| 百位 | cluster level | **cluster 形状**的入选量（`SM90_CLUSTER_SIZES` 的 `min_level`）|
| 千位 | pruning level | 下游剪枝的开关 |

所以 `level = 30` 与 `level = 3` **不是同一件事**：
前者是「wgmma 档 0 + 乘数档 3」，后者是「wgmma 档 3 + 乘数档 0」。

本仓库**自己实现了一遍**这个解码（先拆位、再对应），与库的四个取模函数
在 `level ∈ [0, 4000)` 上**逐点对拍**（`A.1`，4000 个，不一致 0）。

生成数的实测（`B.1`，fp16/bf16 一族）：

| 全局档位 | 不同指令形状 | 乘数个数 | 发射数 |
|---:|---:|---:|---:|
| 0 | 1 | 1 | **3** |
| 10 | 1 | 2 | **6** |
| 20 | 1 | 3 | **9** |
| 30 | 1 | 4 | **12** |

⇒ **每抬一个乘数档多 3 个**（每个 wgmma 形状配 3 个 `MathInstruction`：
f16×f16→f16 / f16×f16→f32 / bf16×bf16→f32，见 `sm90_utils.py:182–200`）。

### 1.3 ★ 形状表的 value 是**生成档位**，不是指令数

`sm90_shapes.py:33–41` 的 docstring 逐字：

> 「These shape and level pairs are defined as dicts, where keys are shapes and values are their
> **associated levels**. If the user input level for that category (… ) is smaller than a shape's
> associated level, it will be **excluded**, and otherwise, included. …
> **Level 0 is always emitted.**」

对应的代码 `sm90_utils.py:131`：

```python
wgmma_shape for wgmma_shape, min_level in SM90_WGMMA_SHAPES_TF32_DENSE.items()
    if level >= min_level
```

**⇒ 它是一个阈值（阈值语义），不是计数、不是代价、不是权重。**

**可机械判定的判据**（`B.2`，本库用这个代替「解释」）：
取 `level = 0`，筛选结果是 `level >= min_level` 的全部键 ⇒ 恰好是 **`value == 0` 的那些键**。
本仓库对 `SM90_WGMMA_SHAPES_FP16_BF16_DENSE` 与 `SM90_WGMMA_SHAPES_TF32_DENSE`
两张表都核了一遍，**两个集合完全相等**。

举例：`SM90_WGMMA_SHAPES_FP16_BF16_DENSE` 共 **32** 个键，
其中 `value == 0` 的**恰好 1 个**：**`(64, 128, 16)`**；
`generate_fp16_bf16_math_instruction_shapes_sm90(0)` 也就恰好返回**这 1 个**形状。
`level = 9` 时返回 **32** 个（全部）。
（TF32 那张表同样是 32 个键、`value == 0` 的恰好是 `(64, 128, 8)`。）

★★ **写成「指令数」或「代价」就是错的**（`CONTRIBUTING` 的 P4「编机制从句」）：
看到 `(64, 8, 16) → 2` 就解释成「代价 2」，是**给一个已经写在 docstring 里的阈值
编了一个听起来专业的因果**。

### 1.4 `procedural_name()`：sm90 处的命名断点

`library.py:1485–1496`：

```python
def procedural_name(self):
  if self.minimum_compute_capability >= 90:
    return "{tbm}x{tbn}x{tbk}_{cm}x{cn}x{ck}_{s}".format(...)   # 含 cluster
  else:
    return "%dx%d_%dx%d" % (tbm, tbn, tbk, stages)              # 不含 cluster
```

**实测六行**（同一对象只改 `min_compute`，`stages=3`、`cluster_shape=[2,1,1]`，`B.4`）：

| `threadblock_shape` | `min_compute` | `procedural_name()` |
|---|---:|---|
| `[128,128,32]` | 75 | `128x128_32x3` |
| `[128,128,32]` | 80 | `128x128_32x3` |
| `[128,128,32]` | **90** | `128x128x32_2x1x1_3` |
| `[128,128,64]` | 75 | `128x128_64x3` |
| `[128,128,64]` | 80 | `128x128_64x3` |
| `[128,128,64]` | **90** | `128x128x64_2x1x1_3` |

★★ **这里有一个转写陷阱**（本库的规划阶段踩过，已订正）：
如果把这六行**读错行**，很容易写成「32x3 / 32x3 / **64**_2x1x1_3」——
那要求 `tbk` **同时是 32 与 64**，而这个 `else` 分支的第三个数**就是** `threadblock_shape[2]`，
**不存在这样的单一对象**。`C.4` 把这件事钉住：两行必须来自**同一个 `tbs`**。

## 2. 执行映射

### 2.1 生成器实际做的四步

| 步 | 函数（`python/cutlass_library/sm90_utils.py`）| 做什么 |
|---|---|---|
| 1 | `generate_fp16_bf16_math_instruction_shapes_sm90(level)` | 按形状表的阈值筛出**指令形状** |
| 1b | `generate_fp16_bf16_math_instructions_sm90(level)` | 每个形状配 **3 个 `MathInstruction`**（三种 dtype 组合）|
| 2 | `generate_tile_descriptions_sm90(...)` | 乘数 × cluster × 指令 ⇒ **CTA 形状**，再 `is_tile_desc_valid` 过滤 |
| 3 | `get_valid_schedules(...)` | 把 `TileDescription` 映射到**调度**（本篇不涉及）|

**实测的枚举规模**（`B.3`，本仓库自己重跑第 1–2 步）：

| 全局档位 | 不同指令形状 | 乘数 | 指令对象 | 乘积（去重）| 发射 | 发射的 CTA 形状（去重）|
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 1 | 3 | 1 | 3 | 1 |
| 9 | 32 | 1 | 96 | 32 | 93 | **31** |
| 10 | 1 | 2 | 3 | 2 | 6 | 2 |
| 19 | 32 | 2 | 96 | 64 | 186 | **62** |
| 20 | 1 | 3 | 3 | 3 | 9 | 3 |
| 29 | 32 | 3 | 96 | 96 | 279 | **93** |
| **30** | 1 | 4 | 3 | 4 | 12 | **4**（**无过滤**）|
| 39 | 32 | 4 | 96 | 112 | 327 | **93** |
| 49 | 32 | 8 | 96 | 224 | 603 | **169** |

两条可以机械判定的性质（都在 `B.3` 里断言）：

1. **乘积的带重数计数 == 「不同指令形状 × 乘数 × cluster」**
   （★ 分母是**不同的形状数**，不是指令对象数——每个形状配 3 个 dtype 的指令，
   它们给出**同一个** CTA 形状）；
2. **发射出的每一个 CTA 形状都落在乘积集合里**，且**不多于**它。
   档位 `0 / 10 / 20 / 30` 上两者**恰好相等**（没有过滤发生），其余严格小于
   ——即 `is_tile_desc_valid` **真的会滤掉东西**。

★ 注意 `A.2` 与 `B.3` 是**两个方向**的对拍：`A.2` 用 `sorted(emitted) ⊆ products` 逐条查，
`B.3` 用计数查。`make falsify` 的 `D7`（把「乘」改成「加」）**两条都会叫**。

### 2.2 枚举的取值面貌（引用值，带定义行号）

| 枚举 | 取值数 | 定义行（`python/cutlass_library/library.py`）| 备注 |
|---|---:|---|---|
| `OpcodeClass` | **6** | 1137 | 含 `BlockScaledTensorOp` / `BlockScaledSparseTensorOp` |
| `DataType` | **47** | 69 | 含 `e4m3` `e5m2` `e2m1` `f4` `f6` **`ue8m0`** `ue4m3` `tf32` `bf16`；`invalid` 是哨兵 |
| `LayoutType` | **20** | 401 | `ColumnMajor` / `RowMajor` / `*Interleaved2,32,64` / `TensorNHWC` … |
| `GemmKind` | **16** | 1234 | 含 `BlockScaledMoeGroupedUniversal3x`（MoE + block-scale 合成一个 kind）|
| `GemmUniversalMode` | **4** | 1624 | `Gemm` / `GemmSplitKParallel` / `Batched` / `Array` |
| `MathOperation` | **12** | 368 | 含 `multiply_add_fast_accum`（即 fp16 快累加）|

**依赖实况**（本条是**实测**，不是从 `pyproject.toml` 推的）：
`pyproject.toml` 的 `dependencies` 有 **6** 条，其中
`networkx` / `pydot` / `treelib` / `cuda-python` **本机都没装**，
而 `cutlass_library` 的 **12 个子模块导入全部成功**（`A.5`）
⇒ 它们是**懒加载**的，只有生成器的 CLI 路径才需要。
⇒ **本主题的 `requirements.txt` 不得照抄 `pyproject.toml`。**

## 3. 实现与演进

### 3.1 朴素做法：手挑一个 tile

「我知道 Hopper 上 `128×128×64` 不错」——手写 `#define` 三个数，
`warp_count` 写死 4，指令形状写死 64。

**问题**：这三个数**不独立**（§1.1）。而且你手写的组合**可能根本不合法**：
`cta_shape[1] < 16` 会被 `is_tile_desc_valid` 拒掉，
`cta_shape[0] % 64 != 0` 也会被拒。手挑的人**看不到这个合法性边界在哪**。

### 3.2 主流实现：形状表 + 档位筛选 + 合法性子句

CUTLASS 把「哪些指令形状存在」写成一张**表**（`sm90_shapes.py` 的
`SM90_WGMMA_SHAPES_*`），每个键配一个 `min_level`；把「乘数」也写成表；
`is_tile_desc_valid` 把**硬件的硬约束**写成子句：

```
cta_shape[0] < 64 或 % 64 != 0        → 拒（WGMMA 恒为 64 across M）
cta_shape[1] < 16 或 % 8  != 0        → 拒（min N 是 8，但 CTA 的 N 最小 16）
cta_shape[2] < 16 或 % 8  != 0        → 拒
任一维 > 256                          → 拒
cluster 里 CTA 数 < 1 或 > 16         → 拒
```

⇒ **「合法」变成了可枚举、可判定的东西**：给定档位，跑一遍循环就知道有哪些组合。

**为什么需要「档位」？** 因为**全枚举会爆炸**。实测：档位 49 一个就有 **603** 个
`TileDescription`；而形状表 32 个键 × 乘数 12 个 × cluster 11 个 = **4224** 种组合。
档位是一根**旋钮**：0 只出最小集，9 出全量。

### 3.3 前沿方案：生成器**故意**不记录指令形状

`sm90_utils.py:378–383`，**作者逐字注释**：

> 「generator can stamp out duplicate kernels, because it doesn't explicitly set instruction
> shape for SM90 kernels, and the 3.X collective API doesn't directly expose them when using
> the auto kernel schedule.」

```python
math_inst_stub = copy.deepcopy(math_inst)
math_inst_stub.instruction_shape = [0, 0, 0]
```

⇒ **发射出的 kernel 不携带「它是从哪个指令形状来的」这个信息。**

这**不是** bug：作者的理由是「3.X 的 collective API 在自动调度下不直接暴露指令形状」，
所以生成器**没法**把它写进 kernel 名。后果是它**必须**接受重复盖章
（同一个 CTA 形状被不同的 (指令形状, 乘数) 对生成出来）。

**本库对这一条的立场**（`CONTRIBUTING` 第 56–60 行）：
**这段注释是某个实现的作者写的，不是规范。** 它**只能**支撑
「CUTLASS 的这个生成器这么做」，**不能**支撑任何关于性能或硬件的断言。
本库把它当成**一手泄漏**引用，并**把后果量化**（§4.1）——而不是替它找理由。

## 4. 踩坑与失败模式

### 4.1 现象：同名 kernel **不是**同一个 kernel

**现象**（`B.8` / `C.5`，实测）：全局档位 **39** 下发射 **327** 个 `TileDescription`，
但只有 **93** 个不同的 `procedural_name()`。**最多的一个名字出现 6 次**。
具体到一个实例：`threadblock_shape = (128, 16, 64)` 在发射结果里出现 **6** 次，
它们只有 **1** 个不同的名字；把 `procedural_name()` 与 `instruction_shape` 一起看，
仍然只有 **1** 种组合 —— **6 条记录全都无法区分**。

**成因**：两件事叠加——

1. **名字的成分不够**：`procedural_name()` 只含 `tbs / cluster / stages`，
   **不含元素类型**。每个形状配 3 个 dtype 的 `MathInstruction`
   （`f16→f16` / `f16→f32` / `bf16→f32`），它们的 CTA 形状相同 ⇒ 名字相同 ⇒ **×3**。
2. **指令形状被置零**：乘积集合里 **16/112** 个 CTA 形状**有多于一个** (指令形状, 乘数) 原像
   ⇒ 再 **×2**。
   实例：`(128,16,64) ← ((64,8,16),(2,2,4))` 与 `((64,16,16),(2,1,4))`。

**规避**：把 `procedural_name()` 当成**人看的标签**，**不要**当成唯一键。
需要唯一标识时，必须把 **元素类型** 与 **指令形状** 一起带上——
而后者**在发射出的对象里已经没有了**（`instruction_shape == [0,0,0]`），
所以这件事**只能在生成之前**做（在自己的循环里记下来，别丢掉）。
★ 本库的 `02_tiling_decompose.py` 就是**因为**要报出这件事，
才必须**自己重跑一遍**那个枚举循环（`our_cta_products`）。

### 4.2 现象：`MathInstruction(16, 8, 16, f16, f16, f32)` **不报错**

**现象**（`C.1`，实测）：这个看起来完全合理的构造式**跑过去了**，
`instruction_shape` 变成了 **`16`**（一个 int，不是元组）。

**成因**：`library.py:1455–1461` 的首参就是 `instruction_shape`，
Python 的位置参数**不做类型检查** ⇒ `16` 被当成整个 `instruction_shape`，
`8` 被当成 `element_a`……一路错位，但**没有任何断言拦住**。

**规避**：首参**必须是元组**：
`MathInstruction((16,8,16), DataType.f16, DataType.f16, DataType.f32, OpcodeClass.TensorOp)`。

### 4.3 现象：`tile_shape` 与 `threadblock_shape` 是**同一个对象**

**现象**（`C.2`，实测）：`td.tile_shape is td.threadblock_shape` ⇒ **`True`**。
改其中一个，另一个**同步变**（把 `tile_shape[1]` 设成 999，`threadblock_shape[1]` 也是 999）。

**成因**：`library.py:1475–1476` 两行赋值的是**同一个列表对象**：

```python
self.threadblock_shape = threadblock_shape
self.tile_shape        = threadblock_shape
```

**规避**：**统一用 `threadblock_shape`**。这是 P9「同名不同指」的**镜像**：
不是「一个名字两个东西」，而是「**两个名字一个东西**」——
它更隐蔽，因为两个名字都「看起来对」。

### 4.4 现象：参数叫 `warp_count`，而属性里**没有 `warp_shape`**

**现象**（`C.3`，实测）：`hasattr(td, "warp_shape")` ⇒ **`False`**；取它会 **`AttributeError`**。

**成因**：`TileDescription.__init__` 的第三个参数是 **`warp_count`**（每维的 warp 数），
不是 warp 的形状。warp 形状是**下游算出来的**（`CTA / warp_count`），
`TileDescription` 不存它。

★ 本库的规划阶段在这一处**猜错过两次**（先猜参数叫 `warp_shape`，
又猜属性里有 `warp_shape`），**两次都是 `AttributeError` 纠正的**。
⇒ **「猜一个合理的参数名」是本域的高频错误**。

**规避**：读签名（`library.py:1474`），不要读直觉。

### 4.5 现象：生成器给的 `stages` 恒为 0

**现象**（`C.6`，实测）：全部发射结果里 `stages` 只有一个取值：**`0`**。

**成因**：`sm90_utils.py:391` 写死 `stages=0`。
真正的 stage 数要在**知道数据类型与 SMEM 容量**之后才算得出来，
而生成 `TileDescription` 这一步**还没有那些信息**。

**规避**：别拿这个 0 当「流水级数」——那是 **P9**：同一个词（stages）在
**两个层级**上指不同的东西。生成阶段的 0 是「**未定**」，调度阶段的
`stages` 才是「几级流水」。

### 4.6 补充：`cluster_shape` 的类型**随来源变**

**现象**：生成器发射的 `cluster_shape` 是 **`tuple`**（因为它直接取了
`SM90_CLUSTER_SIZES` 的键），而 `TileDescription` 的**默认值**是 `[1,1,1]`（**list**）。
本仓库实测全部 603 个结果的类型都是 `tuple`（`B.7`）。

**成因**：构造函数把参数**原样保存**，不转换类型。

**规避**：`cluster_shape` 只做**索引**用，不比较类型。要比较就先 `tuple(...)`。

### 4.7 ★ 那个「别处」在哪里

> **本篇的代价被藏进了「名字」里，以及「生成阶段还没有定型」这件事里。**

1. **最近一层**：`procedural_name()` 不含元素类型与指令形状（§4.1）——
   「这个名字标识什么」被藏进了一个**格式化字符串的成分表**里；
2. **远一层**：`instruction_shape` 被**刻意**置零（§3.3）——
   泄漏被藏进了**一行注释**（作者自己写明了理由）；
3. **再远一层**：`stages=0`（§4.5）—— 一个配置的**代价**（SMEM 占用 / 流水深度）
   在生成阶段**根本还不存在**，要等下游补齐；
4. **最远的一层**：形状表的 value 是**阈值**（§1.3）——
   你看到的数字不像「配置」，像「指标」，而它其实是**一根旋钮的位置**。

## 5. 判据表

> ★ 按本域约定（域 README），§5 的位置与编号保留，内容**不是实测数据**而是**判据表**。
> 前提：本机**无 GPU、无 CUDA 工具链**；本篇的数字**全部是生成器的枚举结果**
> （本机实测值）与**本仓库的重算结果**（推导值），**没有一个性能数字**。

| # | 可机械判定的条件 | 结论 |
|---|---|---|
| 1 | 全局档位 29 的四级解码 | `wgmma 9 / mma 2 / cluster 0 / pruning 0`（个/十/百/千位；程序 `A.1`）|
| 2 | `level = 3` 与 `level = 30` 是同一件事吗 | **不是** —— 前者是 wgmma 档 3，后者是乘数档 3（程序 `A.1`）|
| 3 | 发射出的 CTA 形状集合与「指令形状 × 乘数」的关系 | **前者 ⊆ 后者**，且不多于（13 个档位、549 个 CTA 形状；程序 `A.2`/`B.3`）|
| 4 | 乘积的带重数计数 | `== 不同指令形状数 × 乘数数 × cluster 数`（程序 `B.3`）|
| 5 | 哪些档位**没有过滤** | `0 / 10 / 20 / 30`（发射的 CTA 形状数 == 乘积数；程序 `B.3`）|
| 6 | `SM90_MMA_MULTIPLIERS` 里哪些乘数在档位 0 入选 | **恰好是 `value == 0` 的那些**（程序 `B.2`）|
| 7 | 形状表的 value 是什么 | **生成档位 `min_level`**（阈值；`sm90_shapes.py:33–41` + `sm90_utils.py:131`；程序 `B.2`）|
| 8 | 形状表的 value 是「指令数」吗 | **不是** —— 判据同第 6 行（程序 `B.2`）|
| 9 | `procedural_name()` 的 sm90 断点 | `min_compute >= 90` 时**换整套方案**（六行逐字；程序 `B.4`/`C.4`）|
| 10 | 同一对象只改 `min_compute`（80→90），`tbk` 会变吗 | **不变** —— 两行必须来自同一个 `tbs`（程序 `C.4`）|
| 11 | SM90 的 `warp_count` | **恒为 `[4, 1, 1]`**（603 个结果逐条；程序 `B.5`）|
| 12 | CTA 各维的上限 | **> 256 才拒**（即含 256 合法）—— 按**代码**，不按注释（程序 `B.6`）|
| 13 | `TileDescription` 里的 `stages`（生成阶段）| **恒为 0** ⇒ 是「未定」，不是「一级流水」（程序 `C.6`）|
| 14 | `tile_shape` 与 `threadblock_shape` | **同一个对象**（`is` ⇒ True；改一个两个都变；程序 `C.2`）|
| 15 | `warp_shape` 属性 | **不存在**（`AttributeError`）；参数名是 `warp_count`（程序 `C.3`）|
| 16 | `MathInstruction(16, 8, 16, …)`（错误构造式）| **不报错**，`instruction_shape` 变成 `16`（程序 `C.1`）|
| 17 | 档位 39 下 `procedural_name()` 的唯一性 | **327 条记录只有 93 个名字**，最多重 6 次（程序 `B.8`/`B.8b`）|
| 18 | `tbs = (128,16,64)` 的 (指令形状, 乘数) 原像数 | **2**（`((64,8,16),(2,2,4))` / `((64,16,16),(2,1,4))`；程序 `B.11`/`C.5`）|
| 19 | 发射出的 `instruction_shape` | **恒为 `[0, 0, 0]`**（作者在 `sm90_utils.py:378–383` 写明了理由；程序 `B.8`）|
| 20 | MXF4NVF4 族的 1SM/2SM 关系 | **M 恰好翻倍**（128→256）、N/K 不变（4 对逐条；程序 `B.9`）|
| 21 | 「SM100 的 2SM 让 M 翻倍」 | **不得这样写** —— `SM100_MMA_SHAPES_TF32_DENSE_1SM` 的 M 有 **64 与 128 两种**（程序 `B.9`）|
| 22 | 「CUTLASS 支持哪些架构」 | **必须指明子系统**：`ArchitectureNames` 只到 **90**，而 `include/cute/arch/` 有 **35** 个文件、含 sm100/sm107/sm120（程序 `B.10`）|
| 23 | `mma_sm107.hpp` 存在吗 | **不存在** —— 只有 `mma_sm107_desc.hpp` 与 `mma_sm107_umma.hpp`（程序 `B.10`）|

**闭合性检查**：读完这张表能判完「**给定一个全局档位**，
生成器会发射哪些 CTA 形状、每个形状有几个来源、名字唯一不唯一」，
以及哪几类字段（`stages` / `instruction_shape` / `cluster_shape` 的类型）
**不能按字面理解**。
**唯一的表外输入**是「这些配置在硬件上跑多快」——**本库不回答**（无 GPU、无性能数字）。

## 6. 小结与延伸

### 6.1 三句话

1. **`CTA = 指令形状 × 乘数`（逐元素）**，而 `warp_count` 在 SM90 上是**常数 `[4,1,1]`**。
   三层的形状不能分别选（§1.1）。
2. **`level` 是一个四位十进制数**：个位选指令形状、十位选乘数、百位选 cluster、千位管剪枝。
   **形状表里的 value 是这一套旋钮的档位（阈值），不是指令数也不是代价**（§1.2–1.3）。
3. **生成器刻意抹掉了指令形状**，于是**同名 kernel 不是同一个 kernel**：
   档位 39 上 327 条记录只有 93 个名字，最多重 6 次（§3.3、§4.1）。

### 6.2 延伸

- **本篇用到的 layout 代数**（`Layout` / `coalesce` / `logical_divide`）→
  `30-gpu-programming/cutlass/01-layout-algebra`
- **swizzle 与 collective**（SMEM 的布局怎么与这些 tile 形状配合）→
  `30-gpu-programming/cutlass/03-swizzle-and-collective`
- **wgmma 指令本身的语义**（`m64nNk16` 是怎么算的）→
  `20-gpu-architecture/tensor-core/01-mma-primitive`、`.../03-wgmma-and-pipeline`
- **tile 选多大才快**（SMEM 占用、wave 量化、occupancy）→
  `40-kernels/gemm/02-shared-memory-tiling`、`50-performance`
- **架构演进与 CUTLASS 的对应关系** →
  `20-gpu-architecture/architecture-evolution`

## 编译与运行

```bash
cd 30-gpu-programming/cutlass/code
make run                      # 跑本主题全部三个编号程序；本主题这一篇是第二个
make 02_tiling_decompose      # 只跑本篇对应的那一个
make cutlass-path             # 只做外部工件路径探测，打印选中的 CUTLASS 源码树
make lint                     # AST 检查：不 import torch / triton / numba / cuda
make falsify                  # 元检验：把本仓库那份重算实现改坏，断言必须失败
make mutate                   # 叶子级变异测试（非编号工具，**不进** `make run`）
```

本篇对应的编号程序是 **`code/02_tiling_decompose.py`**（约 460 行）。
它只依赖 `code/` 内的 `specs.py`，**不 import 任何 GPU / 框架依赖**；
但**依赖一棵 CUTLASS 4.8.0 源码树**（域 README 写明的那个例外）。
**找不到 CUTLASS 时它以可操作的错误退出（rc=2），绝不静默跳过对拍。**

**本机跑出的结果**（本仓库程序在本机跑出；测量条件是
`python3 3.13.5` + `numpy 2.4.4`，Apple M1 Pro，**无 GPU**；
**外部工件的版本是 CUTLASS 4.8.0**）：

```
汇总: A 9/9 · B 14/14 · C 6/6 ⇒ 断言 29 条，通过 29 条，失败 0 条
```

程序打印的**泄漏实例**（逐字取自 `make run` 的输出）：

```
  [ok]   B.8    ★★ **生成器抹掉指令形状的后果（实测）**：全局档位 39 下发射 **327** 个
               `TileDescription`，但只有 **93** 个**不同的 `procedural_name()`** ——
               **重名 93 个**，最多的一个名字出现 **6 次**；乘积集合里 **16/112** 个 CTA
               形状有**多于一个** (指令形状, 乘数) 原像；而发射出的对象携带的
               `instruction_shape` 恒为 `(0, 0, 0)`。
```

★★ **一句话的测量条件**：上面这些数字测的是 **CUTLASS 4.8.0 的生成器**，
**不是任何 GPU**。生成器枚举了哪些 tile，与这些 tile 在硬件上跑多快，是两个问题。

## 参考

**本仓库程序（可复现）**

- `code/02_tiling_decompose.py` —— 本篇的编号程序（A/B/C 三类断言在程序内部，共 29 条）。
  ★ 它的 §2 是**本仓库自己重跑的那套枚举循环**（`our_cta_products` / `our_decode_global_level`），
  与 `cutlass_library` 发射出的对象逐条对拍。
- `code/specs.py` —— 12 条 `cutlass_library` 锚点（带 `file:line`）+ 六行
  `procedural_name` 实测表 + 全局档位解码用例 + 形状表 / 枚举计数 / 泄漏量 + 两条陷阱。
- `code/meta_falsify.py` —— `make falsify`：`D6`（档位解码基数改错）与 `D7`
  （CTA 用加法）落在本篇的重算实现上。
- `code/mutate.py` —— `make mutate`：叶子级变异测试。

**外部工件（B 级：NVIDIA 自己的实现，**不是规范**）**

- `cutlass/python/cutlass_library/library.py` —— `MathInstruction`（1455–1469）、
  `TileDescription`（1472–1496）、`procedural_name`（1485–1496）、
  `ArchitectureNames`（1191–1200）、六个枚举的定义行。
- `cutlass/python/cutlass_library/sm90_utils.py` —— 档位解码（90–103）、生成器
  （128–200、373–404）、合法性子句（281–355）、**抹掉指令形状的那三行注释**（378–383）。
- `cutlass/python/cutlass_library/sm90_shapes.py` —— 形状表与 `min_level` 的 docstring（33–41）。
- `cutlass/python/cutlass_library/sm100_shapes.py` —— MXF4NVF4 与 TF32 的 1SM/2SM 表（162–373）。
- `include/cute/arch/` —— **35 个文件**（sm100 / sm107 / sm120 的内容），
  与 `ArchitectureNames` 只到 90 形成对照（`B.10`）。

**本仓库文档（回链，机制不重推）**

- `30-gpu-programming/cutlass/01-layout-algebra` —— `Layout` / `coalesce` / `logical_divide`。
- `20-gpu-architecture/tensor-core/01-mma-primitive`、`.../03-wgmma-and-pipeline` ——
  MMA 原语与 wgmma 的硬件语义（**本篇回链不复述**）。
- `20-gpu-architecture/architecture-evolution/01-capability-breakpoints` ——
  sm80 → sm90 → sm100 的算力断点。
- `40-kernels/gemm/02-shared-memory-tiling` —— tile 选多大、SMEM 怎么切。

**规格与实现的区分（`CONTRIBUTING` 第 56–60 行）**

- 本篇**没有** A 级（厂商规范）来源。上面这些 `.py` 文件是**实现**，
  里面的注释是**某个实现的作者写的**（§3.3 引的那三行尤其如此）。
  ⇒ 它们**只能**支撑「CUTLASS 的这个生成器这么做」，
  **不能**支撑「CUTLASS 规定必须这么做」，更不能支撑「硬件如此」。

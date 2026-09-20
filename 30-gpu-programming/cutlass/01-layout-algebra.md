---
title: CuTe 的 layout 代数：坐标 → 索引的函数
slug: cutlass-layout-algebra
domain: 30-gpu-programming
topic: cutlass
level: 3
difficulty: advanced
prerequisites:
  - 30-gpu-programming/triton/03-layout-and-leaks
  - 20-gpu-architecture/memory-system/02-coalescing-and-cache-line
tags: [cutlass, cute, layout, coalesce, complement, logical-divide, column-major]
related:
  - 30-gpu-programming/cutlass/02-hierarchical-tiling
  - 30-gpu-programming/cutlass/03-swizzle-and-collective
  - 40-kernels/gemm/02-shared-memory-tiling
  - 20-gpu-architecture/memory-system/03-shared-memory-bank-conflict
code:
  - code/01_layout_algebra.py
status: stable
updated: 2026-09-20
---

# CuTe 的 layout 代数：坐标 → 索引的函数

> **同一个 `(4,8)`，CuTe 给你的 stride 是 `(1,4)`，不是 numpy 给你的 `(8,1)`。**
> 本篇把 CuTe 的 `Layout` 拆成**一个函数**（坐标 → 索引）与**一组代数运算**
> （`coalesce` / `complement` / `composition` / `logical_divide` / `zipped_divide` / `right_inverse`），
> 并且**本仓库自己重写了一遍**这套代数，在 **6708 对**布局上与 `pycute` 逐点对拍。
>
> 1. ★★ **`Layout((4,8)) = (4,8):(1,4)`** —— CuTe 的默认是**列主序**（最左维最快），
>    与 numpy / 线性代数教材的默认**相反**。行主序要**显式写** `Layout((4,8),(8,1))`。
> 2. ★★ **`composition(A,B)` 不是函数复合。** 在 **3488** 个可比的 (A,B) 对里，
>    有 **273** 个不满足 `composition(A,B)(i) == A(B(i))` ——
>    **它不抛异常**，只是给了**另一个函数**（`C.9`）。
> 3. ★★ **`logical_divide` 里去掉 `complement` 那一项，定义域会静默缩水 4 倍**
>    （32 → 8），而**打印出来仍像正确答案的前两段**（`C.5b`）。
> 4. ★ 三条已实测的踩坑：`size(swizzle)` 返回 Swizzle **自身**、
>    `idx2crd` **不接受** `Layout` 对象、`right_inverse(6:2)` 退化成 `1:0`。

> ★★ **真值标注（本主题特有，本篇逐处都要带）**：
>
> 本主题的数字全部来自**在本机跑 CUTLASS 4.8.0 自带的 Python 参考实现**
> （`python/pycute/` 与 `python/cutlass_library/`），**不是**从 GPU 上测的，
> **也不是**本仓库的推导。**测量条件是 CUTLASS 的版本，不是 GPU 型号。**
>
> 按 `CONTRIBUTING` 第 63 行的四类口径，本篇有两类数字：
> **本机实测值**（真跑 `pycute` 得到的一切 —— 它是 NVIDIA 自己的参考实现，在本机跑）
> 与**推导值**（本仓库自己写的那份实现算出的、以及与工件对拍的统计）。
> **一条「规格值」都没有**：CUTLASS 没有为这套 layout 代数出过规范文本，
> 它的源码是**实现**，不是规范（`CONTRIBUTING` 第 56–60 行）。
> ⇒ 本篇的数字**只能支撑关于 layout 代数的结论，不能支撑关于 GPU 硬件的结论。**

> ★ **本域的统一主线**（域 README）是「**你写的程序、编译器生成的东西、硬件执行的东西，
> 是三个不同的程序**」。本篇是这条主线在 `cutlass` 主题上的**第一次穿越**：
> **你在模板参数里写的是类型，编译器算出的是布局，而硬件看到的是一串索引。**
> 三个「程序」在这里的形态是：**你写的嵌套元组 → 编译期的 layout 对象 → 运行期的整数**。

## 0. 背景与动机

### 0.1 具体对照场景：三层组合的索引算术

要给一块 64 个 fp32 的共享内存写一个「先分块、再转置、再广播」的取数函数，
手写索引大概长这样：

```c
// 手写：64 个 fp32 的 tile，切成 2×2 个 2×2 的块；坐标 i,j ∈ [0,4)、lane ∈ [0,4)
int block_row = (i / 2) % 2, block_col = j / 2;   // 块坐标 ∈ [0,2)
int in_row    = i % 2,       in_col    = j % 2;   // 块内坐标 ∈ [0,2)
int idx = (block_col * 2 + block_row) * 16      // 块：转置（把 block 的两个下标换了序）
        + (in_col    * 2 + in_row)    * 4       // 块内：转置
        + lane;                                  // 广播：4 个 lane 读同一个元素
```

★ 这段是**可复核的**：三项分别取值 `{0,16,32,48}` × `{0,4,8,12}` × `{0,1,2,3}`，
和恰好铺满 `[0, 64)` **且不重叠** ⇒ 它是 `(i, j, lane) → idx` 的一个**双射**。
把「块内转置」去掉（第 2 项改成 `(in_row * 2 + in_col) * 4`）它仍然是双射——
**这就是问题所在：转置在源码里不是一次「操作」，只是乘法顺序的差异，改错了也不会有人叫。**

三个问题同时出现，而且**互相纠缠**：

1. **层与层之间只能靠注释区分**。`(block_col * 2 + block_row) * 16` 里的那个 `16`
   是下一层的**步长**，但它在源码里只是一个裸数字；
2. **改一层要动全部**。把「块内转置」去掉，第 2 项的 `in_col * 2 + in_row` 要改成
   `in_row * 2 + in_col`，而**第 1 项的 `16` 也得跟着检查**——它们是同一条算术链的两段；
3. **转置这件事没有名字**。转置就是「把两个模式的 stride 换一下」，
   但手写代码里它表现为**重新排列乘法的顺序**，识别不出来。

`CuTe` 的答案是：把这套算术**提升成一个对象**，让「转置」变成一个**可运算的操作**
（交换 `stride` 的两个分量），让「分块」变成一个**有名字的运算**（`logical_divide`），
让「两块能不能合并」变成一个**可判定的条件**（`coalesce`）。

**代价**：你必须先接受它的默认约定——而它的默认约定**和 numpy 相反**（§1.2）。

### 0.2 本篇覆盖什么、不覆盖什么

本篇落在 `code/01_layout_algebra.py` 的 **46 条断言**里（A 23 / B 7 / C 16）：

| 组 | 内容 | 数量 |
|---|---|---|
| **A** | 本仓库自实现的一份 layout 代数 vs `pycute` 的**穷举逐点对拍**（6 个运算，含 6 档 `max_idx`）、三条**独立于两边结构输出**的语义恒等式、锚点引文核对、**常量表与原件的一致性** | 23 |
| **B** | `Layout` 的两笔账：**地址空洞**（`cosize − size`）与**模式数**（rank）、`coalesce` 的免费性、numpy 的第二条计数路径 | 7 |
| **C** | 三条已实测的踩坑 + **五处「静默」**（去掉一项不报错、退化成平凡、非函数复合……）+ 自证伪 | 16 |

**边界**：本篇讲的是 `pycute` 里**已经实现**的那套代数（`python/pycute/layout.py` 与
`int_tuple.py`，**31 条锚点里属于本篇的 13 条**）。它**不讲**：

- **怎么用它写 GEMM** —— 那在 `40-kernels/gemm/02-shared-memory-tiling`；
- **swizzle 怎么消 bank 冲突** —— 那在 `03-swizzle-and-collective` 与
  `20-gpu-architecture/memory-system/03-shared-memory-bank-conflict`，本篇**回链不复述**；
- **CuTe 的 C++ 模板元编程**（`cute::Layout` 的 `constexpr` 求值）—— 本机无 `nvcc`，
  一个字都验证不了，因此**不写**。

### 0.3 统一主线：那个「别处」在哪里

> **契约在哪里断裂 → 断裂的表现是什么 → 怎么发现它 → 怎么绕开。**

**契约**：`Layout = (shape, stride)`，你写的是**形状**，期望的是**「按这个形状铺一块内存」**。

**断裂**：`shape` 不决定 `stride`。**默认**由 `prefix_product` 给出**列主序**，
而你脑子里的默认很可能是行主序。⇒ 同一个 `Layout((4,8))`，
你以为是 `(4,8):(8,1)`，实际拿到 `(4,8):(1,4)`。

**代价被藏到了哪里？**

> **它被藏进了「默认 stride 由谁决定」这个问题里。**
> 本篇 §1.2 会指出：决定它的是一个**你没写下来的函数**（`prefix_product`），
> 而它在这份实现里的作用域**只覆盖「不写 stride」这一种写法**——
> 写了 `Layout((4,8),(8,1))` 之后，一切都回到你的控制下。
> 再往后一步，那个「别处」是**编译期**：`coalesce` 把 4 个模式并成 2 个，
> **省下的不是地址空间**（`size`/`cosize` 一个都不变，`B.4` 实测 **0 处改动**），
> 而是**编译期要展开的层数**——运行期的整数算术完全一样（§3.3）。

## 1. 原理

### 1.1 `Layout` 是一个函数：定义域是 `size`，陪域是 `cosize`

`pycute` 里 `Layout` 的求值只有两处（`python/pycute/layout.py:71–88` 与
`int_tuple.py:160–182`）：

```
Layout.__call__(crd)  →  crd2idx(crd, self.shape, self.stride)
```

而 `crd2idx` 对**标量索引**的定义（`int_tuple.py:174–182`）是一个逐模式的展开：

```
result = 0
for i in range(len(shape) - 1):
    result += crd2idx(crd % product(shape[i]), shape[i], stride[i])
    crd    = crd // product(shape[i])
return result + crd2idx(crd, shape[-1], stride[-1])
```

⇒ **`Layout` 的作用就是把「一个整数」按形状逐位拆开，再按 stride 加权求和。**

于是有两个**不同**的大小（`layout.py:99–104`）：

| 名字 | 定义 | 含义 |
|---|---|---|
| `size()` | `product(shape)` | **定义域**：有多少个坐标 |
| `cosize()` | `self(self.size() - 1) + 1` | **陪域**：最后一个坐标落到哪、再加一 |

**两者对非满射的 layout 不同**，而且**可以往两个方向不同**：

```
Layout((4, 8), (4, 1))     →  size 32，cosize 20   ⇒ cosize < size（**非单射**：坐标多、索引少）
Layout((16, 4), (32, 2))   →  size 64，cosize 487  ⇒ cosize > size（**有空洞**：索引多、坐标少）
```

多步推导（第二个，本篇 §B.5 的那个最坏情形）：

```
cosize((16,4):(32,2)) = 1 + 最后一个坐标的像
最后一个坐标 = (16-1, 4-1) = (15, 3)
它的像 = 15 * 32 + 3 * 2 = 480 + 6 = 486
⇒ cosize = 487
⇒ 空洞 = 487 - 64 = **423**
```

⇒ 「按这个 layout 铺一块 Buffer，要 **487** 个槽，但只有 **64** 个被用到。」
本仓库在全池 **86** 个 layout 上穷举：**43** 个非满射、最大空洞 **423**、最小空洞 **−49**
（`B.1`/`B.5`/`B.7`，numpy 第二条路径复核）。

### 1.2 ★★ 反直觉量化结论：`Layout(shape)` 的默认 stride 是**列主序**

`Layout.__init__` 只有两行（`python/pycute/layout.py:52–57`）：

```python
def __init__(self, _shape, _stride=None):
    self.shape  = _shape
    if _stride is None:
        self.stride = prefix_product(self.shape)      # ← 默认走这里
```

而 `prefix_product` 是**排他式前缀积**（`int_tuple.py:123–139`）：
第 *i* 个模式的 stride 是**它左边所有模式的乘积**。

多步推导（`Layout((2,3,4))`）：

```
stride[0] = 1                （init = 1）
stride[1] = 1 * shape[0]     = 1 * 2 = 2
stride[2] = 2 * shape[1]     = 2 * 3 = 6
⇒ Layout((2,3,4)) = (2, 3, 4):(1, 2, 6)
```

⇒ **最左维最快**。实测逐字（`pycute`，`01_layout_algebra.py: C.1`）：

| 写法 | 结果 | 说明 |
|---|---|---|
| `Layout((4,8))` | `(4, 8):(1, 4)` | **列主序**（最左维最快）|
| `Layout((8,4))` | `(8, 4):(1, 8)` | 同上 |
| `Layout((2,3,4))` | `(2, 3, 4):(1, 2, 6)` | 推导值，与上面手算一致 |
| `Layout((4,8),(8,1))` | `(4, 8):(8, 1)` | **行主序要显式写** |

**这与 `numpy` / 线性代数教材的默认相反。** `numpy` 的 `reshape(4,8)` 是 C 序（最后维最快），
CuTe 的 `Layout((4,8))` 是 **Fortran 序**（第一维最快）。

★ **这条结论的作用域**（P7——引文与断言的作用域必须匹配）：
它说的是 **`pycute` 这份 Python 实现**里 `Layout(shape)`（**不传 stride**）的行为。
`Layout(shape, stride)` 一旦写了 stride，就与「默认是哪个序」无关。
**不得推广成「CuTe 的 layout 都是列主序」**——那会把一个**默认值**说成一个**性质**。

### 1.3 `coalesce`：合并的条件是**相接**，不是**形状相同**

`coalesce` 的合并条件只有一行（`layout.py:154`）：

```python
elif result_shape[-1] * result_stride[-1] == stride:
    result_shape[-1] = result_shape[-1] * shape
```

⇒ **前一个模式的 `shape × stride` 恰好等于后一个模式的 `stride`**，才合并。

三个实测例子（`B.2`，`pycute` 的逐字输出，本仓库实现与之逐点一致，`A.3`）：

逐模式走一遍（`result_shape`/`result_stride` 初值是 `[1]`/`[0]`）：

| 输入 | 逐模式的判定 | `coalesce` 的结果 |
|---|---|---|
| `(2,2):(1,2)` | 模式0 `(2,1)` 占住空槽；模式1 `(2,2)`：`2×1 == 2` ✓ **合并** | **`4:1`** |
| `(2,2):(2,1)` | 模式0 `(2,2)` 占住空槽；模式1 `(2,1)`：`2×2 = 4 ≠ 1` ✗ 追加新模式 | **`(2, 2):(2, 1)`** |
| `(2,2):(4,1)` | 模式0 `(2,4)` 占住空槽；模式1 `(2,1)`：`2×4 = 8 ≠ 1` ✗ 追加新模式 | **`(2, 2):(4, 1)`** |

★ 看的是「**前一个模式的右端**（`shape × stride`）」与「**后一个模式的左端**（`stride`）」
是不是**相接**，而不是两个模式长得像不像。
⇒ **两个形状相同的模式，完全可以不合并**（第 2、3 行就是）。

**代价上的意义**（`B.3`/`B.4`）：`coalesce` 在全池 86 个 layout 上把 rank 减了
**0 维的 40 个 / 少 1 维的 26 个 / 少 2 维的 20 个**，而 `(size, cosize)` **一处都没变**。

⇒ **合并是免费的：它不省地址空间，只省模式数。** 模式数是**编译期**要展开的层数。

### 1.4 `complement`：把 layout 「缺的索引」补上——**默认参数下它什么都不补**

`complement(L, max_idx=1)` 的末段是（`layout.py:253–254`）：

```python
result_shape.append((max_idx + current_idx - 1) // current_idx)   # ceil_div
result_stride.append(current_idx)
```

⇒ **默认 `max_idx=1`** 时，`ceil_div(1, current_idx) == 1`，
于是补出来的那一段 shape 是 1、stride 是累积值，最后再**过一道 `coalesce`**，
shape-1 的模式被丢掉。实测（`C.3`）：

```
complement((4,8):(8,1))          = 1:0      ← 默认 max_idx=1，**平凡结果**
complement((4,8):(8,1), 32)      = 1:0      ← 已经 packed，也没什么可补
complement(Layout(4), 32)        = 8:4      ← 这个才真的补了
```

★ **`1:0` 是一个合法的 layout**（定义域大小 1、所有坐标映射到 0）。
它**不报错、也不抛异常**——你只有在拿它去 `composition` 的时候才会发现
「补了个寂寞」（§1.5 就是这个后果）。

## 2. 执行映射

### 2.1 五个组件名与它们的机械作用

| 组件（路径） | 作用 | 本仓库在本机的实测 |
|---|---|---|
| `python/pycute/layout.py:51` `class Layout` | 坐标 → 索引的函数 | 全池 **86** 个 layout |
| `layout.py:137` `coalesce` | 合并相接的模式 | 全池 rank 只减不增（`B.3`）|
| `layout.py:232` `complement` | 补上缺的索引段 | 默认参数下退化为 `1:0`（`C.3`）|
| `layout.py:190` `composition` | 按 A 的模式切分 B | **3488** 对可比、**273** 对不满足函数复合（`C.9`）|
| `layout.py:307` `logical_divide` | = `composition(A, make_layout(B, complement(B, size(A))))` | 去掉 `complement` ⇒ 定义域 32 → 8（`C.5b`）|
| `layout.py:260` `right_inverse` | CuTe 的 `T` 运算 | `right_inverse(6:2) = 1:0`（`C.4`）|

### 2.2 对拍的规模（本机跑出来的分母）

本仓库自己实现了一份同语义的 layout 代数（`01_layout_algebra.py` §2 的 `RefLayout` 系列，
**写法与 `pycute` 刻意不同**：迭代式逐模式剥离 vs 四分支递归），与 `pycute` 对拍：

| 运算 | 对数 | 不一致 |
|---|---:|---:|
| `coalesce` | **6708** | 0 |
| `complement`（默认 + 5 档 `max_idx`）| **40248** | 0 |
| `composition` | **6708** | 0 |
| `logical_divide` | **6708** | 0 |
| `zipped_divide` | **6708** | 0 |
| `right_inverse` | **6708** | 0 |
| 语义恒等式 `composition(A,B)(i) == A(B(i))`（限定子域）| **579** | 0 |

**「对拍」的意思**：两个实现**各自算一遍**，比较 `(shape, stride)` 的**结构**；
**改坏任一侧，断言立刻失败**（`C.12` 与 `make falsify` 的 `D1`–`D5` 就是这件事的机械化）。

★ **本库**没有**把「调用 pycute 然后打印它的输出」叫对拍**（那是转述，不是对拍）。

### 2.2b 常量表自己也要被反驳（`A.15`–`A.20`）

`specs.py` 里那些「引用的数」如果没有程序消费，就只是**散文**——
改成什么都不会有人叫（`CONTRIBUTING` 第 48 行：「只放从来源抄来的数」，
但**抄来的数也要有消费者**）。所以本篇另有六条断言**专门消费常量表**：

| 断言 | 消费什么 | 结论 |
|---|---|---|
| `A.15` | 三个程序的 `COVERED_ANCHORS` × `specs.ANCHORS` 的键集合 | **两两不交、并集 == 全部 31 个键**；且每条的 `level` 取值集合 = **`['B']`** ⇒ 本主题**刻意没有 A 级来源** |
| `A.16` | `specs.VERSION_SOURCES` | 4 处来源、**恰好 1 处与基准不一致**（`pyproject.toml`）；另 3 处的 `value` 能被**机械重建**（version.h 的三行拼成 `4.8.0`）；第 4 处（`git describe`）**本程序不核**，因为它要 spawn 一个进程 |
| `A.17` | `specs.PYCUTE_EXPORTED` | 本机 `dir(pycute)` **恰好 48 个**，与常量表**逐个相等** |
| `A.18` | `specs.PYCUTE_MEASURED` 的 `key` 与 `by` | `key` **无重复**；每个 `by` 都指向**真实存在**的标签（★ 这条断言上线后**当场抓出两处幽灵标签 `P.1`**）|
| `A.19` | `specs.PYTHON_README_COUNTS` | 在**原始字节**上重数 `python/README.md`：`pycute` **0 次**、`CuTeDSL` **0 次**、而 `cutlass_library` **7 次** ⇒ 「引 `pycute` 只能引源码路径」是**可机械核的** |
| `A.20` | `specs.TRAPS` | 3 条陷阱的 `file` / `also_file` **真的存在**、每个 `line` 都**落在该文件行数以内** |

★ **`A.18` 值得单独说一句**：它上线后立刻报出
`['TA1→P.1', 'TA2→P.1']` —— 我在常量表里把 `by` 写成了 `("C.6", "P.1")`，
而 **`P.1` 这个标签根本不存在**。**一条只写在注释里的「每行都被重跑过」
是不可证伪的；写成断言之后，它自己在第二次运行时就抓到了我的抄写错误。**
这正是 `CONTRIBUTING` 的 P5（断言形式正确但不可证伪）的反面。

### 2.3 三条**独立于两边结构输出**的语义恒等式

结构对拍有一个隐患：如果两份实现**都**从同一个错误的理解出发，它们会**一致地错**。
所以本库另加三条**不依赖结构**的判据：

| 恒等式 | 含义 | 实测 |
|---|---|---|
| `coalesce(L)(i) == L(i)`（逐点）且 `size`/`cosize` 不变 | 合并**保函数** | 全池 86 个，反例 **0** |
| `A(right_inverse(A)(i)) == i` | 右逆的**定义性质** | 全池 86 个，反例 **0** |
| `composition(A,B)(i) == A(B(i))`（**限定子域**）| 复合**保函数** | **579** 对，反例 **0** |

★ 第三条的**限定条件不是装饰**：它是「A 与 B 都 packed，且 `size(A) % size(B) == 0`」。
**去掉这个限定，它就不成立了**——见 §4.1。

## 3. 实现与演进

### 3.1 朴素做法：手写索引算术

回到 §0.1 的那段代码。它的**表达能力**没有问题（硬件只认整数），
问题在**可组合性**：分块、转置、广播三种变换**没有各自的表示**，
所以「组合它们」只能是「手写一遍新的乘法顺序」。

### 3.2 主流实现：`(shape, stride)` 二元组 + 一组代数运算

CuTe 的做法是把「铺一块内存」写成 `(shape, stride)`，然后**定义运算**：

```
coalesce(L)                合并相接的模式          （§1.3）
complement(L, M)           补上 [0, M) 上缺的索引段 （§1.4）
composition(A, B)          按 A 的模式切分 B        （§4.1 有坑）
logical_divide(A, B)       = composition(A, make_layout(B, complement(B, size(A))))
zipped_divide(A, B)        = hier_unzip(logical_divide, A, B)
right_inverse(A)           A 的右逆（CuTe 的 T 运算）
```

**为什么需要 `complement`？** 这是 `logical_divide` 那一行公式里最不直观的一项。
`composition(A, B)` 的**定义域**就是 `B` 的定义域：`size(B)` 个坐标。
而 `logical_divide(A, B)` 想要的是「用 `B` 去切 `A`」——它的定义域必须是 **`size(A)`**。
若 `size(B) < size(A)`，直接 `composition(A, B)` 会得到一个**定义域只有 `size(B)`** 的对象。
`complement(B, size(A))` 的作用就是把 `B` **续满到 `size(A)`**：

```
make_layout(B, complement(B, size(A)))   ⇒  定义域大小 == size(A)
```

**实测**（`C.5`/`C.5b`）：去掉 `complement` 之后，
`logical_divide((4,8):(8,1), 8:1)` 从 `((4,2),4):((8,1),2)`（size 32）
变成 `((4,2),):((8,1),)`（size **8**）——
**它不抛异常，而且打印出来仍像正确答案的前两段**。

### 3.3 前沿方案：把「哪一层」变成编译期的事

`logical_divide` 的输出是**嵌套的**：`((4, 2), 4):((8, 1), 2)`。
一次 `logical_divide((8,8), (2,2))` 就把 rank 从 **2** 抬到 **3**（`B.6`）。

嵌套的意义**不在运行期**——展平之后就是一组整数 stride，运行期的算术与手写完全一样。
嵌套的意义在于**编译期**：它让模板能按「模式」分别选择实现
（例如「块内这一维用向量化 copy、块间那一维用 TMA」）。

⇒ **`coalesce` 与 `logical_divide` 是方向相反的两个操作**：
一个把层压平成一层（省编译期的展开），一个把层显式抬出来（让编译期能分别处理）。
两个都不改变**运行期**的地址算术（`B.4` 实测 `coalesce` 对 `(size, cosize)`
**一处改动都没有**）。

## 4. 踩坑与失败模式

> 每一条写成「**现象 → 成因 → 规避**」，且每一条都是**真的跑出来的**。

### 4.1 现象：`composition` 静默地给出另一个函数

**现象**：把 `composition(A, B)` 当成函数复合 `A ∘ B` 用。
在 **3488** 个可比的 (A,B) 对里有 **273** 个不满足 `C(i) == A(B(i))`。
首例：`A = (2,2):(2,1)`、`B = (3,):(1,)`，
`i = 2` 时 `C(2) = 4`，而 `A(B(2)) = A(2) = 1`。
**它不抛异常。**

**成因**：`composition` 的算法（`layout.py:190–228`）第一步就做 `flat_A = coalesce(layoutA)`，
然后**用 B 的 stride 去切 A 的模式**（`new_shape = min(max(1, curr_shape // rest_stride), rest_shape)`）。
这套切分**假设 B 是 A 的一个「规则细分」**：`size(A)` 能被 `size(B)` 整除、
且 B 的 stride 落在 A 的模式边界上。
B 不满足这个条件时，切分算出一个「合法但语义不同」的 layout，**没有任何断言拦住它**。

**规避**：用**语义恒等式**去用它——在本库实测成立的那个**限定子域**上
（双方都 packed 且 `size(A) % size(B) == 0`），`C(i) == A(B(i))` 才成立。
超出这个子域时，先问「我要的到底是 `composition` 还是我自己写的函数复合」。
★ 这条已在 `01_layout_algebra.py` 里被**双向钉住**：`A.7` 断限定子域内成立，
`C.9` 断限定外**真的会失败**、且**首例能报出来**。

### 4.2 现象：`size(swizzle)` 返回 Swizzle **自身**

**现象**：对 `Swizzle(3,3,3)`：

```
Swizzle(3,3,3).size()   = 512        ← 方法，**正确**
size(Swizzle(3,3,3))    = SW_3_3_3   ← 模块级函数，**返回了 Swizzle 自己**
```

**不抛异常、不告警**——一个 layout 类的对象变成了它的字符串表示。

**成因**（两行代码，缺一不可）：

```python
# layout.py:125–128
def size(layout):
  if is_layout(layout):        # is_layout(x) == isinstance(x, LayoutBase)
    return layout.size()
  return product(layout)       # ← 落到这里
```

`class Swizzle:` 在 `swizzle.py:60` 处**没有基类** ⇒ `is_layout` 为 False
⇒ 走 `product(...)` 分支 ⇒ 而 `product`（`int_tuple.py:65–69`）对非 tuple 输入
**原样返回** ⇒ **返回 Swizzle 自己**。

**规避**：在 `Swizzle` 上**只用方法**（`S.size()`）。
或者更省事：**不要对 swizzle 调模块级 `size()`**。

★★ **这条结论的作用域必须写清**（本库实测后订正了一处**计划里的措辞**）：

| 对象 | `x.size()` | `size(x)` | 一致？ |
|---|---:|---:|---|
| `Layout((4,8),(8,1))` | 32 | 32 | **一致** |
| `Swizzle(3,3,3)` | 512 | `SW_3_3_3` | **不一致** |
| `Swizzle(3,3,3)` 的 **`cosize`** | 512 | **512** | **一致**（！）|

`cosize` **不失效**：模块级 `cosize()`（`layout.py:132–133`）里**没有** `is_layout` 分支，
它**无条件**调 `.cosize()`，而 `Swizzle` 恰好有这个方法。

⇒ 「模块级 `size()` / `cosize()` 对 `Swizzle` 失效」这句话，**只有 `size()` 那一半成立**。
说成两半都失效，就是把一个**局部 bug**升格成**通用规则**
（`CONTRIBUTING` 的 P10）。★ 这条写进判据表（§5 第 5、6 行）。

### 4.3 现象：`idx2crd` / `crd2idx` **不接受 `Layout` 对象**

**现象**：`idx2crd(19, L.shape, L.stride)` 给出 `(2, 3)`（正确）；
而 `idx2crd(19, L)` 抛 **`TypeError: unsupported operand type(s) for %: 'int' and 'Layout'`**；
`crd2idx((2,3), L)` 抛 **`AssertionError`**。

**成因**：两个函数的签名都是 `(idx, shape, stride=None)`（`int_tuple.py:142` / `:160`）——
第二个参数**是 shape，不是 layout**。传 `Layout` 进去时它被当成 shape，
于是 `idx % shape` 撞上一个对象。

**规避**：参数要**摊开**传：`idx2crd(19, L.shape, L.stride)`。
（★ 本库的规划阶段在这里差点把一个「标签」当成「实参」写进规格——
**输出里的标签不是证据，尤其当那标签是自己写的**。）

### 4.4 现象：`right_inverse(6:2)` 退化成 `1:0`

**现象**：`right_inverse(Layout(6, 2))` = **`1:0`** —— 一个定义域大小为 1 的平凡 layout。

**成因**：`right_inverse`（`layout.py:260–283`）按 stride 排序后逐段累积，
一旦某个 stride 与累积值对不上就 **`break`**：

```python
for (stride, shape, rstride) in sorted_DSA:
    if shape == 1: continue
    if current_idx != stride: break        # ← 全部机制在这里
```

`6:2` 的 stride 是 2，而它**不是前缀满射**（`6` 个坐标、步长 2 ⇒ 覆盖 `{0,2,4,6,8,10}`，
不是 `[0,6)`），累积到第一步就对不上，于是**一段都没累积**。

**规避**：用之前先问「这个 layout 是不是前缀满射」。
本库把它的**定义性质**做成了断言：`A(right_inverse(A)(i)) == i`
——上面那个退化的情形**也满足**这条（`i` 只有 0，而 `A(0)=0`），
所以**光靠这条恒等式抓不住退化**，必须**同时**断言形状（`C.4` 就是断形状的）。

### 4.5 现象：去掉 `complement`，定义域静默缩水

**现象**：`logical_divide` 的定义式里丢掉 `complement(B, size(A))` 那一项后，
`logical_divide((4,8):(8,1), 8:1)` 从 `((4,2),4):((8,1),2)`（size 32）
变成 `((4,2),):((8,1),)`（size **8**），**不抛异常**，而且**打印出来仍像正确答案的前两段**。

**成因**：见 §3.2——`complement` 是「把 B 续满到 size(A)」的那一步，去掉它，
`composition` 的定义域就退回 `size(B)`。

**规避**：**不要自己重写 `logical_divide` 的公式**。用现成的。
如果你**必须**重写（比如要移植到另一个框架），那就把
`A.10` 那条**定义式复核**一起带过去：本仓库用四个函数重算 **6708 对**、
反例 **0** —— 而把 `complement` 去掉，`A.4a` 立刻报出 `(4,):(8,)`（`make falsify` 的 `D3`）。

### 4.6 ★ 那个「别处」在哪里

> **本篇的代价被藏进了「默认 stride 由谁决定」这个问题里。**

四层说法，一层比一层远：

1. **最近的一层**：`Layout(shape)` 的 stride 由 `prefix_product` 决定（§1.2）——
   规则本身很短，但它**不在你写的那一行里**；
2. **远一层**：`complement` 的默认 `max_idx=1` 决定了它**默认什么都不补**（§1.4）——
   「补」这件事被藏进了一个**参数默认值**；
3. **再远一层**：`composition` 的「按模式切分」假设 B 是 A 的规则细分（§4.1）——
   这个假设**不在签名里**，只在算法内部；
4. **最远的一层**：嵌套的层数（rank）**只在编译期存在**（§3.3）——
   运行期只剩一串整数，`coalesce` 与 `logical_divide` 在运行期是**同一个东西**。

## 5. 判据表

> ★ 按本域约定（域 README），§5 的位置与编号保留，内容**不是实测数据**而是**判据表**。
> 每一行是「**可机械判定的条件 → 结论**」。
> 前提：本机**无 GPU**；本篇的数字是**本机实测值**（真跑 `pycute` = CUTLASS 4.8.0
> 自带的 Python 参考实现）或**推导值**，**不是 GPU 测量**。

| # | 可机械判定的条件 | 结论 |
|---|---|---|
| 1 | `Layout((4,8))` 的 stride | **`(1, 4)`** ⇒ 默认**列主序**（程序 `C.1`）|
| 2 | 要行主序 | **必须显式写** `Layout((4,8),(8,1))`（程序 `C.1`）|
| 3 | `Layout((2,3,4))` 的 stride | **`(1, 2, 6)`**（由 `prefix_product` 推出，与 `pycute` 逐字一致；程序 `C.1`/`A.13`）|
| 4 | `Layout((16,4),(32,2))` 的空洞 | **423**（`cosize 487 − size 64`；程序 `B.5`/`B.7`）|
| 5 | `size(Swizzle(3,3,3))`（**模块级函数**）| **返回 Swizzle 自身**（打印为 `SW_3_3_3`；程序 `C.6`）|
| 6 | `cosize(Swizzle(3,3,3))`（**模块级函数**）| **512，正确** —— `cosize` **不失效**；且在 `Layout` 上 `size`/`cosize` 方法 == 模块函数（程序 `C.6b`）|
| 7 | 「pycute 的 `size()` 不可靠」 | **这句是错的**（P10）—— 失效**只在 `Swizzle` 的 `size()` 上**（程序 `C.6`/`C.6b`）|
| 8 | `coalesce((2,2):(1,2))` | **`4:1`**（合并；`2×1 == 2`）（程序 `B.2`/`A.3`）|
| 9 | `coalesce((2,2):(2,1))`、`coalesce((2,2):(4,1))` | **都不合并**（程序 `B.2`/`C.2`）|
| 10 | `coalesce` 对 `(size, cosize)` 的影响 | **零改动**（全池 86 个逐点；程序 `B.4`/`A.8`）|
| 11 | `coalesce` 对 rank 的影响 | **只减不增**（40 个不变 / 26 个少 1 / 20 个少 2；程序 `B.3`）|
| 12 | `complement(L)`（**默认参数**）| 对已 packed 的 `L` 给出 **`1:0`**（平凡；程序 `C.3`）|
| 13 | `complement(Layout(4), 32)` | **`8:4`**（程序 `C.3`）|
| 14 | `composition(A,B)(i)` 是否等于 `A(B(i))` | **只在**「双方 packed 且 `size(A) % size(B) == 0`」时成立（579 对，0 反例）；否则 **273/3488 不成立**（程序 `A.7`/`C.9`）|
| 15 | `logical_divide(A,B)` 的定义式 | `composition(A, make_layout(B, complement(B, size(A))))`（`layout.py:307`；程序 `A.10`）|
| 16 | 去掉定义式里的 `complement` | 定义域 **32 → 8**，**不抛异常**，打印仍像正确答案的前两段（程序 `C.5`/`C.5b`）|
| 17 | `zipped_divide` 与 `logical_divide` 的关系 | **展平形式逐个相同**（5214 对可比、0 不一致；程序 `A.5b`）|
| 18 | `right_inverse((4,8):(8,1))` | **`(8, 4):(4, 1)`**（程序 `A.13`）|
| 19 | `right_inverse(6:2)` | **`1:0`**（非前缀满射 ⇒ 退化；程序 `C.4`）|
| 20 | `A(right_inverse(A)(i)) == i` | **成立**（全池 86 个，0 反例）——但**抓不住退化**，必须同时断言形状（程序 `A.9`/`C.4`）|
| 21 | `idx2crd(19, L)`（传 `Layout` 对象）| **`TypeError`**；正确形式是三参数（程序 `C.10`/`C.10b`）|
| 22 | `make_layout(((2,3),(1,4)))`（传嵌套元组）| **`AttributeError`** —— `make_layout` 只接受 `Layout` 对象（程序 `C.11`）|
| 23 | `Layout(shape)` 的默认 | **只在 `pycute` 的这份 Python 实现上**成立（`layout.py:55`）；**不得推广成「CuTe 的 layout 都是列主序」**（§1.2 的作用域）|

**闭合性检查**：读完这张表能判完「**给定一个 `(shape, stride)`**，
它的定义域/陪域各是多少、能不能与另一个合并、拿去做 `logical_divide` 会得到什么，
以及哪几类行为是**静默**的」。
**唯一的表外输入**是「你想让哪一维最快」——那必须**读你自己的意图**，不能从形状推出来。

## 6. 小结与延伸

### 6.1 三句话

1. **`Layout` 是一个函数**：定义域是 `size`、陪域是 `cosize`，两者对非满射的 layout 不同，
   而且**可以往两个方向不同**（`(4,8):(4,1)` 是 `cosize < size`，`(16,4):(32,2)` 是相反）。
2. **它的默认 stride 是列主序**（`prefix_product`），与 numpy / 教材相反；
   行主序要显式写。这条**只作用在「不写 stride」这一种写法上**。
3. **代数运算里有三处静默**：`complement` 的默认参数什么都不补、
   `composition` 在非规则细分下给另一个函数、`right_inverse` 在前缀满射不成立时退化为 `1:0`。
   **三处都不抛异常**——它们只能靠**语义断言**抓（本库的 A 类）。

### 6.2 延伸

- **层次化 tiling**（`CTA = 指令形状 × 乘数`、生成器抹掉的那一维）→
  `30-gpu-programming/cutlass/02-hierarchical-tiling`
- **swizzle 作为一种 layout 构造**（以及它的恒等区间）→
  `30-gpu-programming/cutlass/03-swizzle-and-collective`
- **bank 冲突的机制**（为什么是 32 个 bank、广播为什么免费）→
  `20-gpu-architecture/memory-system/03-shared-memory-bank-conflict`（**本篇回链不复述**）
- **合并访存**（global memory 一侧的 128 字节事务）→
  `20-gpu-architecture/memory-system/02-coalescing-and-cache-line`
- **拿这套 layout 写 GEMM** → `40-kernels/gemm/02-shared-memory-tiling`

## 编译与运行

```bash
cd 30-gpu-programming/cutlass/code
make run                      # 跑本主题全部三个编号程序；本主题这一篇是第一个
make 01_layout_algebra        # 只跑本篇对应的那一个
make cutlass-path             # 只做外部工件路径探测，打印选中的 CUTLASS 源码树
make lint                     # AST 检查：不 import torch / triton / numba / cuda
make falsify                  # 元检验：把本仓库那份参照实现改坏，断言必须失败
make mutate                   # 叶子级变异测试（非编号工具，**不进** `make run`）
```

本篇对应的编号程序是 **`code/01_layout_algebra.py`**（1100 行）。它只依赖 `code/` 内的
`specs.py`，**不 import 任何 GPU / 框架依赖**；但**依赖一棵 CUTLASS 4.8.0 源码树**
（域 README 写明的那个例外）。

**找不到 CUTLASS 时**它**以可操作的错误退出（rc=2）**，**绝不静默跳过对拍**：

```
!! 找不到 CUTLASS 源码树。**本程序不做任何对拍就退出**（不静默跳过）。
   请显式指定（两种方式任选）：
     make run CUTLASS_PATH=/path/to/cutlass
     export CUTLASS_PATH=/path/to/cutlass
```

**本机跑出的结果**（本仓库程序在本机跑出；测量条件是
`python3 3.13.5` + `numpy 2.4.4`，Apple M1 Pro，**无 GPU**；
**外部工件的版本是 CUTLASS 4.8.0**）：

```
汇总: A 23/23 · B 7/7 · C 16/16 ⇒ 断言 46 条，通过 46 条，失败 0 条
```

程序开头打印的**版本四处对照**（逐字取自 `make run` 的输出）：

```
    include/cutlass/version.h        = 4.8.0                  ✓ 与基准一致
    CHANGELOG.md 首条标题            = 4.8.0                  ✓ 与基准一致
    git describe --tags --always     = v4.8.0dev-2-gdc45f979  ✓ 与基准一致
    pyproject.toml 的 version        = 4.2.0.0                ✗ **与基准不一致**
```

★★ **四处不一致这件事本身就是一条结论**：`pyproject.toml` 的版本串**陈旧，
落后三个次版本**。照它写「本库基于 CUTLASS 4.2」，后面所有 `sm_100` 的 atom
都会变成「这个版本不该有的东西」。**引用版本不能只挑一处读。**

## 参考

**本仓库程序（可复现）**

- `code/01_layout_algebra.py` —— 本篇的编号程序（A/B/C 三类断言在程序内部，共 46 条）。
  ★ 它的 §2 是**本仓库自己写的一份 CuTe layout 代数**（`RefLayout`），与 `pycute` 穷举对拍。
- `code/specs.py` —— 31 条带 `file:line` 的源码锚点 + `pycute` 的 22 行实测输出表 +
  版本四处对照 + 外部工件路径探测。**每条带 `# 来源：` 与来源等级。**
- `code/meta_falsify.py` —— `make falsify`：把本仓库那份实现改坏（9 种缺陷，含 2 条方向相反的），
  断言程序必须失败。**这是「对拍」这一侧自己被反驳的机制。**
- `code/mutate.py` —— `make mutate`：叶子级变异测试（`D1`–`D5` 落在本篇的常量上）。

**外部工件（B 级：NVIDIA 自己的实现，**不是规范**）**

- `cutlass/python/pycute/layout.py` —— `Layout` / `coalesce` / `complement` / `composition` /
  `right_inverse` / `logical_divide` / `zipped_divide`（本篇 13 条锚点全部落在这个文件与
  `int_tuple.py` 上，逐条带行号，由 `A.12` 在**原始字节**上核对）。
- `cutlass/python/pycute/int_tuple.py` —— `product` / `flatten` / `prefix_product` /
  `idx2crd` / `crd2idx` / `shape_div`。
- `cutlass/python/pycute/swizzle.py` —— 本篇只用到它的 `Swizzle` 与 `ComposedLayout`
  两个类（§4.2 的陷阱），**它的代数用法在 `03` 篇**。

**本仓库文档（回链，机制不重推）**

- `20-gpu-architecture/memory-system/03-shared-memory-bank-conflict` ——
  bank 冲突的机制、`gcd(行距, 32)` 规则、64 位访问的判定池。
- `20-gpu-architecture/memory-system/02-coalescing-and-cache-line` —— 合并访存。
- `30-gpu-programming/triton/03-layout-and-leaks` —— 另一套 layout 系统的泄漏形态
  （**编译器替你选** layout，对照本篇的**你写 layout**）。
- `40-kernels/gemm/02-shared-memory-tiling` —— 拿这套 layout 写 GEMM。

**规格与实现的区分（`CONTRIBUTING` 第 56–60 行）**

- 本篇**没有**任何 A 级（厂商规范）来源。CuTe 的 layout 代数**没有规范文本**，
  上面两个 `pycute` 源文件是**实现**。⇒ 本篇的所有「结论」都带这个作用域，
  它们**只能支撑关于 `pycute` 这份实现的结论**，不能支撑「CuTe 规定必须如此」。

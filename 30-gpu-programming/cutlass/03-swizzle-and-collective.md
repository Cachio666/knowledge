---
title: swizzle 与 collective：一个用在 64 个元素上什么都看不出来的算子
slug: cutlass-swizzle-and-collective
domain: 30-gpu-programming
topic: cutlass
level: 3
difficulty: advanced
prerequisites:
  - 30-gpu-programming/cutlass/01-layout-algebra
  - 20-gpu-architecture/memory-system/03-shared-memory-bank-conflict
tags: [cutlass, cute, swizzle, composed-layout, smem, collective]
related:
  - 30-gpu-programming/cutlass/02-hierarchical-tiling
  - 40-kernels/gemm/02-shared-memory-tiling
  - 20-gpu-architecture/tensor-core/03-wgmma-and-pipeline
code:
  - code/03_swizzle_probe.py
status: stable
updated: 2026-09-20
---

# swizzle 与 collective：一个用在 64 个元素上什么都看不出来的算子

> **`Swizzle(3,3,3)` 在偏移小于 64 时是恒等映射。**
> 一个「用来打散布局」的算子，在**小规模测试用例上完全看不出它在做什么**——
> 8×8 的 tile 放在它下面，**64 个元素一个都不动**（`C.9`）。
>
> 1. ★★ **它的恒等区间是 `[0, 2^(base + max(0,shift)))`**，
>    `Swizzle(3,3,3)` 的掩码躺在 **bit 6–8**，所以前 **64** 个偏移**完全恒等**。
>    这个边界是**从掩码位置推出来的**，不是试出来的（`A.2`）。
> 2. ★★ **它改变的比例只跟 `bits` 走：`1 − 2^−bits`**。
>    `bits=3` ⇒ **87.5%**（448/512），**与 `base` / `shift` 完全无关**（`A.3`/`B.4`）。
> 3. ★★ **`Swizzle` 不是 `LayoutBase` 的子类**，于是模块级 `size(Swizzle)` **静默返回
>    Swizzle 自身**（打印成 `SW_3_3_3`）；而 **`cosize` 不失效**——这条陷阱
>    **只能限定在 `size()` 上**（`C.6`/`C.6b`）。
> 4. ★★ **swizzle 要用 `ComposedLayout(swizzle, offset, layout)` 表达**，
>    三段复合的语义是 `L_B(offset + L_A(crd))`；
>    `composition(swizzle, layout)` 抛 `AttributeError`（`C.8`）。

> ★★ **真值标注（本主题特有，本篇逐处都要带）**：
>
> 本主题的数字全部来自**在本机跑 CUTLASS 4.8.0 自带的 Python 参考实现**
> （`python/pycute/` 与 `python/cutlass_library/`），**不是**从 GPU 上测的，
> **也不是**本仓库的推导。**测量条件是 CUTLASS 的版本，不是 GPU 型号。**
>
> 本篇的数字有三类：**本机实测值**（真跑 `pycute.Swizzle` 得到的 11460 个映射点）、
> **推导值**（从掩码位置推出的恒等边界与闭式比例）、以及两者的对拍统计。
> **一条「规格值」都没有**——`swizzle.py` 是**实现**，不是规范。
> ⇒ **本篇一个硬件数字都没有**：没有 bank 冲突路数、没有带宽、没有延迟。

> ★★ **本篇不推导 bank 冲突的机制。** 冲突的成因（为什么是 32 个 bank、
> 广播为什么免费、64 位访问为什么换一套判定池、`gcd(行距, 32)` 规则）
> 在 `20-gpu-architecture/memory-system/03-shared-memory-bank-conflict`，
> **本篇回链不复述**。本篇讲的是**另一种东西**：
> **swizzle 作为一种 layout 代数的构造**——它是怎么定义的、它的定义域有多大、
> 它在什么范围里是个恒等映射、以及它怎么参与复合。

> ★ **本域的统一主线**：「**你写的程序、编译器生成的东西、硬件执行的东西，
> 是三个不同的程序**」。本篇是它在 `cutlass` 主题上的**第三次穿越**：
> **你写的是「这块 SMEM 按行取」，编译器把 swizzle 折进每一个地址，
> 而硬件看到的是一串异或过的偏移**——而**在 tile 小于 64 个元素时，这一折什么都没做**。**

## 0. 背景与动机

### 0.1 具体对照场景：同一块 SMEM，两种读法

一块 32×32 的 fp32 共享内存 tile（4 KB），一个 warp 的 32 个 lane 要以两种方式取用它。
两种方式的**地址集合**不同，因此**硬件看到的请求形状**也不同——
这个差异的机制在 `20-gpu-architecture/memory-system/03`（**本篇回链不复述**）。

**本篇关心的是另一侧**：当你要消掉那个差异时，你**能改的是什么**。

三种做法，代价依次升高：

| 做法 | 改的是什么 | 代价 |
|---|---|---|
| 给行距 `+1`（padding）| **地址算术**（`stride` 从 32 变 33）| 每行浪费 4 字节；**而且不一定有效**（见 `.../memory-system/03`）|
| 换一种 lane→数据的映射 | **lane 模式**（谁读哪个元素）| 你要改的是**算法**，不是布局 |
| **改数据在 SMEM 里的摆放** | **布局本身** | 需要一次**写入时的重排**；读的时候要**算回来** |

第三种就是 swizzle：**写入时按一个异或规则把偏移打散，读的时候按同一条规则算回来。**
关键在于「同一条规则」——它是**双向的**（异或的自逆性），所以不需要存储额外的映射表。

⇒ 但你付出了一个新的代价：**「这块数据在 SMEM 里的第 *i* 个元素在哪」这个问题
变得不能靠一个 stride 回答了。** 它需要一个**函数**。这就是 §1 的出发点。

### 0.2 本篇覆盖什么、不覆盖什么

本篇落在 `code/03_swizzle_probe.py` 的 **22 条断言**里（A 9 / B 6 / C 7）。

**覆盖**：`Swizzle(bits, base, shift)` 的位运算定义、**恒等区间的推导**、
**非平凡比例的闭式**、`ComposedLayout` 的三段复合、以及**两条已实测的陷阱**
（`size()` 对 Swizzle 失效、`composition(swizzle, …)` 抛异常）。

**不覆盖**：

- **bank 冲突的机制与判定** —— `20-gpu-architecture/memory-system/03-shared-memory-bank-conflict`
  （**本篇回链不复述**）；
- **collective（wgmma 的 SMEM 描述符、TMA）本身** —— 那是
  `20-gpu-architecture/tensor-core/03-wgmma-and-pipeline` 与
  `30-gpu-programming/cutlass/02-hierarchical-tiling` 的话题。
  本篇**只讲 swizzle 作为「一个 layout」的构造**，不讲它如何被 collective 消费；
- **CUTLASS 的 C++ `Swizzle` 模板** —— 本机无 `nvcc`，一个字都验证不了，因此不写。

### 0.3 统一主线：那个「别处」在哪里

> **契约在哪里断裂 → 断裂的表现是什么 → 怎么发现它 → 怎么绕开。**

**契约**：`Swizzle(bits, base, shift)` 是一个「打散布局」的算子。

**断裂**：**它的作用范围有下界**——偏移小于 `2^(base + max(0,shift))` 时，
它是**恒等映射**。`Swizzle(3,3,3)` 的这个下界是 **64**。

**代价被藏到了哪里？**

> **它被藏进了「掩码躺在哪一位」这件事里。** `Swizzle(bits, base, shift)` 的三个参数
> 不是三个自由的旋钮：`shift` 的绝对值**必须 ≥ `bits`**（构造式强制的断言），
> 而真正决定「多大的数据才看得见它」的是 `base + max(0, shift)`——
> 一个**你没写下来、要从掩码位置算出来的**量（§1.2）。
> 再往后一步，那个「别处」是**测试**：一个只在 8×8 tile 上做的单元测试
> **永远测不出**这个算子在工作（§4.1）。

## 1. 原理

### 1.1 定义：一个 `& / >> / ^` 构成的整数函数

`swizzle.py:48–58` 的注释图**逐字**：

```
## A generic Swizzle functor
 # 0bxxxxxxxxxxxxxxxYYYxxxxxxxZZZxxxx
 #                               ^--^  Base is the number of least-sig bits to keep constant
 #                  ^-^       ^-^      Bits is the number of bits in the mask
 #                    ^---------^      Shift is the distance to shift the YYY mask
 #                                       (pos shifts YYY to the right, neg shifts YYY to the left)
 #
 # e.g. Given
 # 0bxxxxxxxxxxxxxxxxYYxxxxxxxxxZZxxx
 # the result is
 # 0bxxxxxxxxxxxxxxxxYYxxxxxxxxxAAxxx where AA = ZZ xor YY
```

对应的求值只有一行（`swizzle.py:73–74`）：

```python
def __call__(self, offset):
  return offset ^ shiftr(offset & self.yyy_msk, self.shift)
```

而掩码的位置由 `swizzle.py:68–70` 决定：

```python
bit_msk = (1 << bits) - 1
self.yyy_msk = bit_msk << (base + max(0,shift))
self.zzz_msk = bit_msk << (base - min(0,shift))
```

**本仓库的实现**（`03_swizzle_probe.py` §2）**换了一条路径**——
不做双重掩码、也不移动掩码，直接把 YYY 字段取出来搬到 ZZZ 位：

```
result = offset ^ (((offset >> yyy_pos) & mask) << zzz_pos)
其中 yyy_pos = base + max(0, shift)，zzz_pos = base + max(0, -shift)
```

两条路径在 **18 个参数组合 × 全部偏移 = 11460 个点**上逐点一致（`A.1`）；
再加一条 **numpy 向量化**的第三条路径（`A.3b`），三条都一致。

### 1.2 ★ 恒等区间：**从掩码位置推出来的，不是试出来的**

**观察**：`S(o) != o` 当且仅当被异或进去的那一项非零，也就是
`(o >> yyy_pos) & mask != 0` ⇔ **YYY 字段非零**。

**推论一（恒等区间）**：YYY 字段的最低有效位在 `yyy_pos`，
所以 **`[0, 2^yyy_pos)` 上的每一个偏移，YYY 字段都是 0**，因此**处处恒等**。
而 `2^yyy_pos` 本身 YYY 字段非零 ⇒ **它恰好是第一个非平凡的偏移**。

**推论二（首个非平凡映射的像）**：偏移 `2^yyy_pos` 的 YYY 字段恰好是 `1`
⇒ 异或进去的是 `1 << zzz_pos` ⇒ 像 = `2^yyy_pos ^ 2^zzz_pos`。

**逐条验证**（`A.2`，18 个参数组合，本仓库实测）：

| `(bits,base,shift)` | `yyy_pos` | 恒等上界 `2^yyy_pos` | 实测首个非平凡偏移 | 它的像（实测 = 推导）|
|---|---:|---:|---:|---:|
| `(3,3,3)` | 6 | **64** | **64** | **72**（`64 ^ 8`）|
| `(3,4,3)` | 7 | **128** | **128** | **144**（`128 ^ 16`）|
| `(2,5,2)` | 7 | 128 | 128 | 160（`128 ^ 32`）|
| `(4,2,4)` | 6 | 64 | 64 | 68（`64 ^ 4`）|
| `(1,6,1)` | 7 | 128 | 128 | 192（`128 ^ 64`）|

★ **注意 `(3,3,3)` 与 `(3,4,3)` 的对照**：同一 `bits`，**`base` 抬一位就把恒等区间翻倍**
（64 → 128）。而 `bits` **完全不影响**恒等区间（`B.3` 把 `bits` 从 1 扫到 4 逐条确认）。

**⇒ 「恒等区间 = `2^(base + max(0,shift))`」与 `bits` 无关。**

### 1.3 ★ 非平凡比例的闭式：`1 − 2^−bits`

**推论三**：YYY 字段在整个定义域上**均匀铺满**（每个值出现同样多次）
⇒ 被改变的偏移个数 = `size × (2^bits − 1) / 2^bits`。

**⇒ 非平凡比例 = `1 − 2^−bits`，与 `base` / `shift` 完全无关。**

实测（`A.3` / `B.4`）：

| `bits` | 定义域 | 实测被改变 | 比例 | 闭式 |
|---:|---:|---:|---:|---:|
| 1 | 4 | 2 | 0.5000 | `1 − 1/2` |
| 2 | 16 | 12 | 0.7500 | `1 − 1/4` |
| **3** | **512** | **448** | **0.8750** | **`1 − 1/8`** |
| 4 | 256 | 240 | 0.9375 | `1 − 1/16` |
| 5 | 32768 | 31744 | 0.9688 | `1 − 1/32` |

⇒ **`Swizzle(3,3,3)` 的 87.5%（448/512）不是测出来的巧合，是 `bits=3` 的直接后果。**
`bits` 越大，打散得越彻底（趋近 1）。

★★ **「87.5% 被改变」不蕴含「改动是均匀的」**（`C.10`）：
实测 `|S(o) − o|` 的取值在 **0 … 56** 之间。掩码有 3 位 ⇒ 位移量有 **8** 档。
⇒ 被改变的 448 个偏移里，**有的是 8，有的是 56**。
说「它把布局均匀打散了」多走了一步——那是另一件事，本篇不声称。

### 1.4 `ComposedLayout`：swizzle 参与复合的唯一形式

`swizzle.py:93–109`：

```python
class ComposedLayout(LayoutBase):
  def __init__(self, layoutB, offset, layoutA): ...
  def __call__(self, *args):
    return self.layoutB(self.offset + self.layoutA(*args))
```

⇒ **三段复合**：先 `layoutA`（坐标 → 偏移），加 `offset`，再 `layoutB`（偏移 → 偏移）。

**语义**：`layoutA` 把**坐标**映射成 SMEM 里的偏移；`layoutB` 是 **swizzle**
（把偏移搅一下）；`offset` 是一个基址偏移。

**为什么必须是这个类？** 因为 `Swizzle` **不是 `Layout`**——
它没有 `shape`、没有 `stride`，只有一个 `__call__`。
所以 `composition(swizzle, layout)` 会在读 `B.shape` 时**抛 `AttributeError`**（`C.8`）。

**本仓库的对拍**（`A.4`）：用**自己的**三段复合公式（`lambda crd: sw(offset + layout(crd))`）
与 `pycute` 的 `ComposedLayout` 在 **896 个点**（8 组 swizzle × 3 个 layout 的全部元素）
上逐点比对，不一致 **0**。

**`size` / `cosize` 各看一边**（`A.5`，实测）：

```
ComposedLayout(Swizzle(3,3,3), 0, Layout((8,8),(8,1))).size()   = 64   ← size(layoutA)
ComposedLayout(同上).cosize()                                    = 512  ← cosize(layoutB) = cosize(Swizzle)
```

⇒ 「两边各看一个方向」。这也解释了为什么 §4.2 那个陷阱
**在 `ComposedLayout` 上不出现**：它调的是**模块级** `size(layoutA)`，
而 `layoutA` 是一个正常的 `Layout`。

## 2. 执行映射

### 2.1 掩码的位置决定了「多大的用例才看得见它」

`Swizzle(3,3,3)` 的三个实测参数（`B.1`，逐字）：

```
str(Swizzle(3,3,3)) = SW_3_3_3
yyy_msk = 0b111000000   （bit 6–8）
zzz_msk = 0b111000      （bit 3–5）
shift   = 3
size    = 512
被改变的偏移 = 448 / 512 = 87.5%
首个非平凡映射 = (64, 72)
```

`Swizzle(3,4,3)`（`B.2`）：

```
yyy_msk = 0b1110000000  （bit 7–9）
size    = 1024
被改变的偏移 = 896 / 1024 = 87.5%（★ 与上面**同一个比例**）
首个非平凡映射 = (128, 144)
```

⇒ **同一 `bits`，两组的比例一模一样（87.5%），而恒等区间差一倍（64 vs 128）。**
**「打散得多不多」与「多大数据才看得见」是两个独立的量**，分别由 `bits` 与 `base+shift` 决定。

### 2.2 一个 tile 上的实际观感（本机实测）

把 `ComposedLayout(Swizzle(3,3,3), 0, Layout((W,8),(8,1)))` 摊在一整块 `W×8` 的 tile 上，
数「有几个元素真的被挪动了」（`B.5`/`B.6`/`C.9`）：

| tile | 元素数 | 被挪动 | 说明 |
|---|---:|---:|---|
| **8×8** | 64 | **0** | 整块 tile 的偏移都 < 64 ⇒ **一个都不动** |
| 16×8 | 128 | 64 | 从第 64 个偏移起才开始动 |
| 32×8 | 256 | 192 | |
| 64×8 | 512 | 448 | 与 `Swizzle` 自身的统计一致 |

★ **第一行是整篇的核心**：一个 8×8 的单元测试**看不出这个算子在做什么**。
它甚至在**数值上完全正确**——因为 swizzle 是恒等映射，写进去和读出来当然对得上。

## 3. 实现与演进

### 3.1 朴素做法：不 swizzle

SMEM 里的数据按最自然的顺序摆放（行主序或列主序）。
在一部分访问模式下这会撞上 bank 冲突——**机制与判定**在
`20-gpu-architecture/memory-system/03-shared-memory-bank-conflict`，**本篇不复述**。

### 3.2 中间做法：padding

给行距 `+1`。改的是**一个整数**（`stride`），代价是 4 字节/行。
★ **它不一定有效**：那篇文档实测过「同一个 `+1`，在一份代码里有效、在另一份里完全无效」——
有效性取决于 lane 模式。**本篇不重启那个话题**，只提一句作为对照：
**padding 改的是 stride，swizzle 改的是「偏移到偏移」的映射。**

### 3.3 主流做法：把 swizzle 写成一个 layout 代数里的对象

CuTe 的做法是让 swizzle **参与复合**（§1.4）：

```
ComposedLayout(swizzle, offset, layout)
```

**为什么这是一步演进？** 因为：

1. **它可组合**：swizzle 可以套在任何一个 `layout` 外面，不需要为每种组合写专门的消冲突代码；
2. **它可运算**：`ComposedLayout` 有 `size()` / `cosize()` / `__getitem__`，
   能参与更大结构的构造（例如一个 CTA 的 SMEM 描述符）；
3. **它是双向的**：异或自逆 ⇒ **读的时候用同一个对象**就能算回来，不需要存映射表。

**代价**：**你的「第 i 个元素在哪」不再是一个 stride 能回答的了**（§0.1）——
必须有那个函数。而 **函数的可测试性远不如一个 stride**：§4.1 就是这件事的直接后果。

### 3.4 前沿：swizzle 从「手写的消冲突技巧」变成「描述符的一部分」

在 SM90/SM100 上，swizzle 被写进 **SMEM 描述符**里（TMA / wgmma 的 descriptor），
由硬件在搬运时应用——**这时候它不再是你的 kernel 里的一段算术，而是硬件的一个模式位**。

★ **本库对这一层的立场**：本机**无 GPU、无 `nvcc`**，
描述符的编码（几位、怎么填）**一个字都验证不了**，因此**本篇不写**。
它属于 `20-gpu-architecture/tensor-core/03-wgmma-and-pipeline`。
本库能做的、也已经做的，是**把 swizzle 作为一个 layout 对象的性质**讲清楚（§1–§2）——
那是可以被 `pycute` 在这台机器上**逐点反驳**的那一部分。

## 4. 踩坑与失败模式

### 4.1 ★★ 反直觉量化结论：小规模测试用例上它什么都看不出来

**现象**（`C.9`，实测）：把 `Swizzle(3,3,3)` 套在一个 **8×8** 的 tile 上（64 个元素），
**64 个元素一个都没被挪动**。

**成因**：恒等区间是 `[0, 2^yyy_pos)`，而 `yyy_pos = base + max(0,shift) = 6` ⇒ `[0, 64)`。
8×8 的 tile **全部落在里面**（§1.2）。

**规避**：测 swizzle 的用例**必须让它跨越恒等边界**。
本库的做法是**用 `W×8` 的 tile 扫 `W ∈ {8,16,32,64}`**，
并**先算出边界再断言**（`our_identity_bound`）——
**不是**「随便拿个大一点的 tile 试一下」。
★ 更一般地：**一个算子的作用范围有下界时，「测不出效果」与「效果不存在」
在小用例上无法区分**。这是本域「反直觉量化结论」的正面教材。

### 4.2 现象：`size(swizzle)` 返回 Swizzle **自身**

**现象**（`C.6`，实测）：

```
isinstance(Swizzle(3,3,3), LayoutBase)  = False
Swizzle(3,3,3).size()                   = 512        ← 方法，正确
size(Swizzle(3,3,3))                    = SW_3_3_3   ← 模块级函数，返回了 Swizzle 自己
```

**成因**：三处代码缺一不可（全部逐字核对过，见 `specs.ANCHORS` 的 `S2`/`S3`/`P8`/`P9`/`P11`）：

1. `class Swizzle:`（`swizzle.py:60`）**没有基类** ⇒ `is_layout(S) == isinstance(S, LayoutBase) == False`；
2. 模块级 `size()`（`layout.py:125–128`）写的是
   `if is_layout(layout): return layout.size()` **否则** `return product(layout)`；
3. `product()`（`int_tuple.py:65–69`）对**非 tuple 输入原样返回** ⇒ 返回的**就是 `S` 自己**。

**规避**：对 `Swizzle` **只用方法** `S.size()`。

★★ **这条结论的作用域**（本库实测后**订正了一处计划里的措辞**）：

| 对象 | `x.size()` | `size(x)` | `x.cosize()` | `cosize(x)` |
|---|---:|---|---:|---:|
| `Layout((4,8),(8,1))` | 32 | **32**（一致）| 32 | **32**（一致）|
| `Swizzle(3,3,3)` | 512 | **`SW_3_3_3`**（不一致）| 512 | **512**（**一致**！）|

**`cosize` 不失效**：模块级 `cosize()`（`layout.py:132–133`）里**没有** `is_layout` 分支，
它**无条件**调 `.cosize()`，而 `Swizzle` **恰好有**这个方法（`swizzle.py:80–82`，
返回 `self.size()`）。

⇒ 「模块级 `size()` / `cosize()` 对 `Swizzle` 失效」这句话，
**只有 `size()` 那一半成立**。说成两半都失效，是把一个**局部的入口 bug**
升格成**通用规则**（`CONTRIBUTING` 的 P10）。
★ 这也说明「坏的不是 swizzle 的**语义**」：`S(o)` 那 512 个映射
两条实现（本仓库的位运算 / `pycute` 的 `shiftr` 递归）**逐点一致**（`C.11`）——
**坏的是 `size()` 这个入口。**

### 4.3 现象：`Swizzle(4, 3, 3)` 直接 `AssertionError`

**现象**（`C.7`，实测）：`Swizzle(4,3,3)` / `Swizzle(3,3,2)` / `Swizzle(5,0,4)`
全部抛 **`AssertionError`**。

**成因**：构造式里有一行硬断言（`swizzle.py:64`）：

```python
assert abs(shift) >= bits
```

「4 位掩码、3 位基、3 位移位」在**纸面上**完全合理——掩码在 bit 6–9，
ZZZ 在 bit 3–5，看着一点问题都没有。但**构造式不允许**。

★ **为什么要有这条约束？** —— `swizzle.py` **没有写理由**，下面是**本库的推导**
（`CONTRIBUTING` 第 56–60 行：源码的注释是作者写的，源码里**没有**的理由更不能替它编）：

掩码的位置是 `yyy_msk = m << (base + max(0,shift))`、`zzz_msk = m << (base - min(0,shift))`
（`m = 2^bits − 1`）。分两种情况：

```
shift > 0:  YYY 在 [base+shift, base+shift+bits)，ZZZ 在 [base, base+bits)
            ⇒ 不相交 ⟺ base+shift >= base+bits ⟺ shift >= bits
shift < 0:  YYY 在 [base, base+bits)，ZZZ 在 [base+|shift|, base+|shift|+bits)
            ⇒ 不相交 ⟺ |shift| >= bits
```

本库在 8 个参数组合上**穷举验证了这条推导**：

| `(bits,base,shift)` | `\|shift\| >= bits` | 两个掩码不相交 | 映射是**自逆**的（`S(S(o)) == o`）|
|---|---|---|---|
| `(3,3,3)` / `(3,4,3)` / `(2,0,2)` / `(3,0,-3)` / `(2,3,-2)` | **是** | **是** | **是** |
| `(4,3,3)` / `(3,3,2)` / `(5,0,4)` | **否** | **否** | **否** |

⇒ **这条断言守的是「异或自逆」这条性质** —— 而自逆正是「写进去时搅一下、
读出来时用同一个对象算回来」这件事成立的全部依据（§0.1）。
★ 「自逆」那一列是**本库穷举算出来的**（对每个组合的整条定义域），不是抄的。

**规避**：把 `shift` 的绝对值抬到 `bits` 以上。**这是构造式强制的，不是文档建议**——
它在 `__init__` 的第一分钟就抛出来，你不可能绕过。

### 4.4 现象：`composition(swizzle, layout)` 抛 `AttributeError`

**现象**（`C.8`，实测）：
`AttributeError: 'Swizzle' object has no attribute 'shape'`。

**成因**：`composition(A, B)`（`layout.py:190–228`）要读 `B.shape` 与 `B.stride`
（它要把 B 按 A 的模式切分），而 `Swizzle` **既没有 `shape` 也没有 `stride`**
——它不是一个 `Layout`，是一个**函数**。

**规避**：用 `ComposedLayout(swizzle, offset, layout)`。
★ **它不是「另一种写法」，是唯一能参与复合的写法**——
`A.4` 通过 896 个点验证它的语义（`L_B(offset + L_A(crd))`）就是你想的那件事。

### 4.5 补充：同一个属性，两个类型

`TileDescription.cluster_shape` 的**默认值**是 `list`（`[1,1,1]`），
而生成器给的**永远**是 `tuple`（它直接取了表的键）。**同一个属性两种类型。**
本篇不展开（那在 `02` 篇 §4.6），只作为 P9 的又一例记一笔。

### 4.6 ★ 那个「别处」在哪里

> **本篇的代价被藏进了「掩码躺在哪一位」与「谁负责测试它」这两件事里。**

1. **最近的一层**：`shift` 的绝对值必须 ≥ `bits`（§4.3）——
   一条**构造式强制的约束**，它保证两个字段不重叠；
2. **远一层**：恒等区间由 `base + max(0,shift)` 决定（§1.2）——
   「多大的数据才看得见它」是**从掩码位置算出来的**，不是一个写在签名里的参数；
3. **再远一层**：`size()` 对 `Swizzle` 静默失效（§4.2）——
   而 `cosize()` 不失效。**同一个入口，一半坏一半好**；
4. **最远的一层**：**测试的规模决定了你能不能看见它**（§4.1）——
   8×8 的用例上它是恒等映射。**「测不出效果」与「效果不存在」在小用例上无法区分**，
   而这件事**不在 `swizzle.py` 的任何一行里**。

## 5. 判据表

> ★ 按本域约定（域 README），§5 的位置与编号保留，内容**不是实测数据**而是**判据表**。
> 前提：本机**无 GPU**；本篇的数字是**本机实测值**（真跑 `pycute.Swizzle`）
> 或**推导值**（从掩码位置推出），**不是 GPU 测量**。
> ★ **bank 冲突路数不在本表里** —— 那需要硬件，回链
> `20-gpu-architecture/memory-system/03-shared-memory-bank-conflict`。

| # | 可机械判定的条件 | 结论 |
|---|---|---|
| 1 | `Swizzle(3,3,3)` 的 `yyy_msk` / `zzz_msk` | **`0b111000000`（bit 6–8）/ `0b111000`（bit 3–5）**（程序 `B.1`）|
| 2 | `Swizzle(3,3,3)` 的定义域 | **512**（程序 `B.1`）|
| 3 | 恒等区间的上界 | **`2^(base + max(0,shift))`**，与 `bits` **无关**（程序 `A.2`/`B.3`）|
| 4 | `Swizzle(3,3,3)` 的恒等区间 | **`[0, 64)`**（18 个参数组合里逐条验证；程序 `A.2`）|
| 5 | 首个非平凡偏移 | **恰好 `2^yyy_pos`**；其像 = `偏移 ^ 2^zzz_pos`（程序 `A.2`）|
| 6 | `Swizzle(3,3,3)` 的首个非平凡映射 | **(64, 72)**（程序 `B.1`）|
| 7 | `Swizzle(3,4,3)` 的恒等区间与首例 | **`[0, 128)`**、**(128, 144)** —— 同一 `bits`，`base` 抬一位翻倍（程序 `B.2`）|
| 8 | 非平凡比例 | **`1 − 2^−bits`**，与 `base`/`shift` 完全无关（程序 `A.3`/`B.4`）|
| 9 | `bits = 3` 的非平凡比例 | **87.5%**（448/512 与 896/1024 都是它；程序 `B.1`/`B.2`）|
| 10 | 被改变的偏移，位移量是否均匀 | **不均匀** —— `\|S(o) − o\| ∈ [0, 56]`（8 档；程序 `C.10`）|
| 11 | 8×8 tile 上 `Swizzle(3,3,3)` 挪动几个元素 | **0 个**（全落在 `[0,64)` 里；程序 `C.9`/`B.5`）|
| 12 | 16×8 / 32×8 / 64×8 tile 上挪动几个 | **64 / 192 / 448**（程序 `C.9`）|
| 13 | `Swizzle(4,3,3)` 会怎样 | **`AssertionError`**（`abs(shift) >= bits` 是构造式强制的；程序 `C.7`）|
| 14 | `ComposedLayout(S, offset, L)(i)` | **`== S(offset + L(i))`**（896 个点；程序 `A.4`）|
| 15 | `composition(swizzle, layout)` | **`AttributeError: 'Swizzle' object has no attribute 'shape'`**（程序 `C.8`）|
| 16 | `ComposedLayout` 的 `size()` / `cosize()` | **各看一边**：`size(layoutA)` = 64 / `cosize(layoutB)` = 512（程序 `A.5`）|
| 17 | `isinstance(Swizzle(...), LayoutBase)` | **`False`**（`class Swizzle:` 没有基类；程序 `C.6`）|
| 18 | `size(Swizzle(3,3,3))`（**模块级函数**）| **返回 Swizzle 自身**（打印为 `SW_3_3_3`；程序 `C.6`）|
| 19 | `cosize(Swizzle(3,3,3))`（**模块级函数**）| **512，正确** —— `cosize` **不失效**（程序 `C.6b`）|
| 20 | 「pycute 的 `size()` / `cosize()` 对 Swizzle 都失效」 | **这句是错的** —— 只有 `size()` 那一半成立（P10；程序 `C.6b`）|
| 21 | 坏的是 `S(o)` 那 512 个映射吗 | **不是** —— 两条实现的 512 个映射逐点一致，坏的只是 `size()` 这个入口（程序 `C.11`）|
| 22 | 想知道某个 swizzle 在你们的 tile 上有没有效果 | **先算 `2^(base+max(0,shift))`，再比 tile 的偏移范围** —— 比不出来就是没效果（§1.2/§4.1）|

**闭合性检查**：读完这张表能判完「**给定 `(bits, base, shift)`**：
它的定义域多大、恒等区间多长、改变多少比例、第一个被改的是谁、
能不能复合、以及它在你手上的 tile 上到底动了没有」。
**表外的输入有两个**：(a) **bank 冲突路数**（需要硬件，回链既有篇章）；
(b) **`offset` 该取多少**（取决于你在 SMEM 里把它放在哪）。

## 6. 小结与延伸

### 6.1 三句话

1. **`Swizzle(bits, base, shift)` 的恒等区间是 `[0, 2^(base+max(0,shift)))`**——
   `bits` 只管「打散多少」（比例 `1 − 2^−bits`），
   `base+shift` 才管「多大数据才看得见」（§1.2–1.3）。
2. **它必须用 `ComposedLayout(swizzle, offset, layout)` 参与复合**，
   三段语义是 `L_B(offset + L_A(crd))`（§1.4）。
3. **它在 `size()` 上有一个静默陷阱、在 `cosize()` 上没有**——
   而这个差别来自模块级两个函数**是否带 `is_layout` 分支**（§4.2）。

### 6.2 延伸

- **bank 冲突的机制与判定**（`gcd(行距, 32)`、广播、64 位访问的判定池）→
  `20-gpu-architecture/memory-system/03-shared-memory-bank-conflict`
  （**本篇回链不复述**）
- **`+1` padding 的效果为什么依赖 lane 模式** → 同上那篇（它有同一块 SMEM 上两种模式的对照表）
- **swizzle 怎么被 collective 消费**（SMEM 描述符、wgmma 的 `GMMA::Layout`）→
  `30-gpu-programming/cutlass/02-hierarchical-tiling`（tile 形状那一侧）与
  `20-gpu-architecture/tensor-core/03-wgmma-and-pipeline`（硬件那一侧）
- **本篇用到的 layout 代数**（`composition` / `size` / `cosize`）→
  `30-gpu-programming/cutlass/01-layout-algebra`

## 编译与运行

```bash
cd 30-gpu-programming/cutlass/code
make run                      # 跑本主题全部三个编号程序；本主题这一篇是第三个
make 03_swizzle_probe         # 只跑本篇对应的那一个
make cutlass-path             # 只做外部工件路径探测，打印选中的 CUTLASS 源码树
make lint                     # AST 检查：不 import torch / triton / numba / cuda
make falsify                  # 元检验：把本仓库的位运算实现改坏，断言必须失败
make mutate                   # 叶子级变异测试（非编号工具，**不进** `make run`）
```

本篇对应的编号程序是 **`code/03_swizzle_probe.py`**（约 590 行）。
它只依赖 `code/` 内的 `specs.py`，**不 import 任何 GPU / 框架依赖**，
**不 spawn 任何进程**，也不碰任何硬件；但**依赖一棵 CUTLASS 4.8.0 源码树**。
**找不到 CUTLASS 时它以可操作的错误退出（rc=2），绝不静默跳过对拍。**

**本机跑出的结果**（本仓库程序在本机跑出；测量条件是
`python3 3.13.5` + `numpy 2.4.4`，Apple M1 Pro，**无 GPU**；
**外部工件的版本是 CUTLASS 4.8.0**）：

```
汇总: A 9/9 · B 6/6 · C 7/7 ⇒ 断言 22 条，通过 22 条，失败 0 条
```

程序打印的**恒等区间与首个非平凡映射**（推导 vs 实测，逐字取自 `make run` 的输出）：

```
     (bits,base,shift) yyy_pos  恒等上界   实测首个偏移   首个偏移的像
     (3,3,3)         6          64         64            72 (推导 72)
     (3,4,3)         7          128        128           144 (推导 144)
     (2,5,2)         7          128        128           160 (推导 160)
     (1,6,1)         7          128        128           192 (推导 192)
```

★ **每一行的「实测」与「推导」都是分开算的**：
推导那一列来自 `2^(base+max(0,shift))` 与 `2^yyy_pos ^ 2^zzz_pos` 两个**闭式**，
实测那一列来自**真跑 `pycute.Swizzle`**。`A.2` 断言两列必须相等——
**改坏任一侧，断言立刻失败**（`make falsify` 的 `D9` 就是把推导那一列换成另一个候选答案）。

## 参考

**本仓库程序（可复现）**

- `code/03_swizzle_probe.py` —— 本篇的编号程序（A/B/C 三类断言在程序内部，共 22 条）。
  ★ 它的 §2 是**本仓库自己的位运算实现**（`our_swizzle` / `our_identity_bound` /
  `our_nontrivial_count`），与 `pycute.Swizzle` 逐点对拍；另有一条 **numpy 向量化**的第三路径。
- `code/specs.py` —— 6 条 `swizzle.py` 锚点（带 `file:line`）+ 两条陷阱的实测值 +
  陷阱 A 的五条实测行（含**对计划标题的订正**：`cosize` 不失效）。
- `code/meta_falsify.py` —— `make falsify`：`D8`（YYY/ZZZ 位置互换）与
  `D9`（恒等上界换成另一个候选答案）是**方向相反**的两条，落在本篇的实现上。
- `code/mutate.py` —— `make mutate`：叶子级变异测试。

**外部工件（B 级：NVIDIA 自己的实现，**不是规范**）**

- `cutlass/python/pycute/swizzle.py` ——
  **位图的注释**（48–58）、`Swizzle.__init__` 与那行断言（60–70）、
  `__call__`（73–74）、`size` / `cosize`（76–82）、
  `ComposedLayout` 与它的三段复合（93–109）。
  全部由 `A.6` 在**原始字节**上逐字核对。
- `cutlass/python/pycute/layout.py` —— 模块级 `size()`（125–128，**陷阱的来源**）
  与 `cosize()`（132–133，**不失效的那一个**）。
- `cutlass/python/pycute/int_tuple.py` —— `product()`（65–69，陷阱的最后一环）。

**本仓库文档（回链，机制不重推）**

- `20-gpu-architecture/memory-system/03-shared-memory-bank-conflict` ——
  **bank 冲突的机制**、`gcd(行距, 32)` 规则、广播为什么免费、
  64 位访问为什么换一套判定池、`+1` padding 的有效性为什么依赖 lane 模式。
  ★ 本篇**只**讲 swizzle 作为一种 layout 代数构造，**不重复推导上述任何一条**。
- `30-gpu-programming/cutlass/01-layout-algebra` —— `Layout` / `coalesce` /
  `composition` / `logical_divide`（本篇用到的 `size` / `cosize` / `composition` 在那里）。
- `30-gpu-programming/cutlass/02-hierarchical-tiling` —— swizzle 要服务的那些 tile 形状。
- `20-gpu-architecture/tensor-core/03-wgmma-and-pipeline` —— 硬件侧怎么消费 SMEM 布局。

**规格与实现的区分（`CONTRIBUTING` 第 56–60 行）**

- 本篇**没有** A 级（厂商规范）来源。`swizzle.py` 是**实现**，
  它里面的那行断言（`abs(shift) >= bits`）是**某个实现的作者**写的约束，
  **不是**一条标准规定。⇒ 它**只能**支撑「`pycute` 的这个类这么做」，
  **不能**支撑「CuTe 规定 swizzle 必须满足 `|shift| >= bits`」，更不能支撑任何硬件断言。
- ★ 还要再强调一次：**本篇没有任何 bank 冲突数字**。
  那些数需要 GPU，本机没有；它们的合法载体是上面那篇既有文档。

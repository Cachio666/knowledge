---
title: 张量内存模型：stride、storage 与 view 的真正判据
slug: pytorch-tensor-memory-model
domain: 70-frameworks
topic: pytorch
level: 2
difficulty: intermediate
prerequisites:
  - 70-frameworks/pytorch/01-dispatch-and-aten
tags: [pytorch, stride, storage-offset, view, reshape, contiguous, as-strided]
related:
  - 10-foundations/performance-models/01-roofline-model
  - 20-gpu-architecture/memory-system/02-coalescing-and-cache-line
code:
  - code/03_strides_and_views.py
status: stable
updated: 2026-09-17
---

# 张量内存模型：stride、storage 与 view 的真正判据

> `view` 的判据**不是 `is_contiguous()`**，而是官方说的 **contiguity-like condition**（`stride[i] = stride[i+1] × size[i+1]`）——所以 `torch.randn(4,4)[:, ::2]` 这个**非连续**张量的 `.view(8)` 会**成功且共享存储**；而 `is_contiguous()` 本身**只吃 sizes/strides/numel**，既不查边界也不查重叠。

## 0. 背景与动机

### 一个具体的对照场景

```python
x = torch.randn(4, 4)
x[:, ::2].view(8)     # ✅ 成功，且与原张量共享存储
x.t().view(16)        # ❌ RuntimeError
x.t().reshape(16)     # ✅ 成功（必要时拷贝）
```

三行的差别，是「框架对内存形状的假设」这一层的全部内容。**写错了不会静默算错**（这点比 autograd 好），但会带来两种代价：要么多一次不必要的拷贝，要么在不该拷的地方拷了。

### 本文在本域主线上的位置：**③本文不涉及**

本域的主线是「**框架要把哪些假设固化进图**」。**本文讲的是 eager 的内存模型与 stride 判据，与「固化进图」无关**——这里显式写出来，而不是硬套一个归属：

- 本文的 `view` / `reshape` / `stride` 判据是**纯几何**的：给定 sizes/strides/numel 就能判定，**不需要任何编译或追踪**。
- 它与主线唯一的接触点是：Dynamo 的 guard 里**包含静态形状假设**，而形状假设的语义正是本文 §1 的 stride 语义（回链 [60-compilers/inductor](../../60-compilers/inductor/02-fusion-scheduling.md) 与 [04-custom-operators](04-custom-operators.md) §3）。

### 三条最容易记错的

1. 「**`view` 要求张量连续**」——**不要求**。判据是 contiguity-like condition；`is_contiguous() = False` 的张量照样能 view。
2. 「**`reshape` 在非连续时一定拷贝**」——**不一定**。它对非连续输入也可能返回 view（共享存储）。
3. 「**`is_contiguous()` 会检查边界/重叠**」——**都不检查**。它只吃 `sizes / strides / numel` 三个参数（签名里**没有 storage**）。

## 1. 原理：storage / offset / stride 三个量

一个 PyTorch 张量由 `(storage, storage_offset, sizes, strides)` 描述，元素 `(i₀,…,i_{d−1})` 的存储位置是

```
offset = storage_offset + Σ_d i_d × stride_d
```

本机实测的一组对照（`make run` 的 §1 表，base 是 `(4,6)` 的连续张量）：

| 表达式 | shape | stride | `is_contiguous()` | offset | 共享存储 |
|---|---|---|---|---|---|
| `base` | (4,6) | (6,1) | True | 0 | — |
| `base.t()` | (6,4) | (1,6) | False | 0 | True |
| `base[:, ::2]` | (4,3) | (6,2) | **False** | 0 | True |
| `base[1:3]` | (2,6) | (6,1) | True | **6** | True |
| `base[1:3, ::3]` | (2,2) | (6,3) | False | 6 | True |
| `base.reshape(6, 4)` | (6,4) | (4,1) | True | 0 | True |
| `base.t().reshape(6, 4)` | (6,4) | (1,6) | **False** | 0 | **True** |
| `base.expand(3,4,6)` | (3,4,6) | (0,6,1) | False | 0 | True |
| `base.flatten()` | (24,) | (1,) | True | 0 | True |

两行的形状相同、stride 不同（`base.t()` 的 (6,4) 与 `base.reshape(6,4)` 的 (6,4)）：**形状不是布局，stride 才是。**

### 1.1 `view` 的真正判据

官方 `Tensor.view` 的 docstring 逐字（本机 `torch.Tensor.view.__doc__`，五个版本一致）：

> 「each new view dimension must either be a **subspace of an original dimension**, or only span across original dimensions `d, d+1, …, d+k` that satisfy the following **contiguity-like condition** that `∀ i = d, …, d+k−1, stride[i] = stride[i+1] × size[i+1]`」

用一句话重述这个判据（本文的表述，**不是** torch 的代码）：

> **目标形状在元素序上切出的每一条边界，都必须落在原张量「连续段」的边界上。**

推导它的两条理由：

- 目标张量的**每个维只有一个 stride**，所以任何目标维都**不能横跨**原张量里两个不连续的段（那样元素序里的偏移就不是等差数列了）；
- 反过来，**段内部**可以随便切——一段是等 stride 的等差数列，切成几维后 stride 依次相乘仍是同一条数列。这就是官方说的「新维是原维的子空间」。

按这个判据，`(4,2)` / stride `(4,2)`（即 `randn(4,4)[:, ::2]`）整块只有一个连续段（因为 `stride[0] = 4 = stride[1] × size[1] = 2 × 2`），**段边界是空集**，所以任何「元素序边界」都在它里面——**任何 numel 为 8 的形状都能 view**。实测：`.view(8)` ✅、`.view(2,4)` ✅。

而 `(4,4).t()` 的 stride 是 `(1,4)`：`stride[0] = 1 ≠ stride[1] × size[1] = 4 × 4`，所以它是**两段**，段边界是 `{4}`。目标 `(16,)` 的边界集是空集——**空的边界集装不下 `{4}`**，于是失败。实测的错误原文：

```
RuntimeError: view size is not compatible with input tensor's size and stride
(at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
```

### 1.2 `is_contiguous()` 的权威判据

权威定义就在本机（`c10/core/Contiguity.h`，2.13.0 文件里 `_compute_contiguous` 在 **:15**）：

```cpp
bool _compute_contiguous(ArrayRef<T> sizes, ArrayRef<T> strides, T numel) {
  if (numel == 0) return true;
  T expected_stride = 1;
  for (int64_t d = sizes.size() - 1; d >= 0; d--) {
    if (sizes[d] == 1) continue;                 // ★ 跳过 size == 1 的维
    if (strides[d] != expected_stride) return false;
    expected_stride *= sizes[d];
  }
  return true;
}
```

三个可直接使用的结论：

1. **`numel == 0` 恒为 True**（实测：`torch.empty(0,3)` 与 stride 荒谬的 `as_strided` 空视图都判 True）。
2. **`size == 1` 的维被跳过**——该维的 stride 是多少都不影响判定。实测：`(4,1,5)` / stride `(5,7,1)` 判 **True**（那个 `7` 是"免检"的），而同样几何的 `(4,2,5)` / stride `(5,7,1)` 判 **False**。
3. **签名里没有 storage**——所以它**不可能**检查边界或重叠。「重叠」由同文件另一个**独立**函数负责：`_compute_non_overlapping_and_dense(ArrayRef<T> sizes, ArrayRef<T> strides)`。

## 2. 执行映射：`contiguous()`、`view()`、`reshape()` 三个入口的代价

### 2.1 `contiguous()` 在已连续时不进 dispatcher

实测（用 dispatch mode 拦，见 [01-dispatch-and-aten](01-dispatch-and-aten.md)）：

| 张量 | `contiguous()` 返回 | 拦截次数 | 拦到什么 |
|---|---|---|---|
| 已连续 | **`self`** | **0** | — |
| 非连续 | 新的副本 | 1 | `aten.clone.default` |

**「已连续时 0 次拦截」是个强判据**：它说明这条路连 dispatcher 都没进（而不是"进了但做了个空操作"）。所以 `x.contiguous()` 在连续输入上是**免费**的——不需要在代码里手动判断「要不要 contiguity」。

### 2.2 `view()` 报错 / `reshape()` fallback

| | 非连续输入 | 失败时 |
|---|---|---|
| `view` | 能 view 就 view | **抛 RuntimeError** |
| `reshape` | 能 view 就 **view**（共享存储）；不能才拷贝 | 总是成功 |

实测两组：

- `(4,4).t().view(16)` → 抛错；`.reshape(16)` → 成功，且**指针变了**（真的拷贝了）。
- `(4,6).t().reshape(6,4)` → 成功，且**共享存储**（指针相同）——**非连续也能 reshape 成 view**。

**实用区别就一句话**：`view` 是「我只接受零成本的做法」，`reshape` 是「零成本优先，不行就付一次拷贝」。

### 2.3 共享存储怎么判：别只看 `storage().data_ptr()`

实测 `full = torch.randn(8)`、`sl = full[2:6]`：

```
full.data_ptr()        = 4694844032
sl.data_ptr()          = 4694844040   ← 含 offset=2 的偏移，所以不相等
sl.storage() 的指针     = 4694844032   ← 相等
sl.storage_offset()    = 2
```

**两个判据会给出相反的结论。** 判「是不是同一块存储」要用**未加 offset 的 storage 指针**（`t.untyped_storage().data_ptr()`）配 `storage_offset()`；只看 `data_ptr()` 会把「同一 storage 的两个不同区域」判成两块。

## 3. 实现与演进：把判据写成可复核的独立实现

本文的 A 类断言采用的是一种**结构上更强**的做法：判据不写在被测程序里，而是先用纯 python3 写一份**独立实现**（`code/verify.py` 的 `ref_*` 系列，**不 import torch**），再用它去**预测**真 torch 的行为：

| 参照实现 | 预测什么 | 对拍结果 |
|---|---|---|
| `ref_compute_contiguous` | `is_contiguous()` | **16/16** 张量一致 |
| `ref_view_compatible` | `view()` 的成败 | 10/10 一致 |
| `ref_stride_slice` / `_transpose` / `_expand` | stride 传播 | 6/6 一致 |

**为什么这比「在程序里写断言」强**：判据的**语义**与**实现**分离了。`verify.py` 可以在**什么都没装的机器**上跑（它只用标准库），而它对拍的对象是真 torch——两份实现互相独立，任一边写错都会暴露。

独立性是**机械验证**过的（`code/03_strides_and_views.py` §A5、§A6）：把参照改错（不跳过 `size == 1` 的维），A1 对拍立刻判为不一致；同时程序按行正则核对 `verify.py` 里**没有** `import torch` 语句。

`ref_view_compatible` 的实现只有十几行（连续段合并 + 前缀积的包含关系），它的正确性证据是**在 16 个张量的连续判据、10 个 view 目标形状、6 条 stride 传播上与真 torch 逐条对拍**——这也解释了为什么第一版实现（试图逐维对齐）会被实跑推翻：那段代码在「单段被拆成多维」与「多段」两种情形上判错。

## 4. 踩坑与失败模式

### 失败模式 1：用 `as_strided` 造「不查边界」的反例——它会先抛错

**现象**：想在 4 个元素的 storage 上造一个 `(2,3)`/`(3,1)` 的视图，以此说明「`is_contiguous()` 不查边界」。

**成因**：`as_strided` **自带边界检查**。实测五版本一致的错误：

```
RuntimeError: setStorage: sizes [2, 3], strides [3, 1], storage offset 0, and itemsize 4
requiring a storage size of 24 are out of bounds for storage of size 16
```

**规避**：用 **wrapper 子类**造同一几何——`torch.Tensor._make_wrapper_subclass(cls, (2,3), strides=(3,1), …)` **被接受**，并且 `is_contiguous()` 判 **True**。同一几何下「`as_strided` 拒绝、wrapper 接受并判 True」，这才是「不查边界」的实证。（wrapper 子类**必须**定义 `__torch_dispatch__`，否则 `TypeError: … must define __torch_dispatch__`。）

**结构性的第二条证据**在 §1.2：`_compute_contiguous` 的参数里根本没有 storage，所以它不可能查边界。

### 失败模式 2：以为「非连续就不能 view」

**现象**：看到 `is_contiguous() = False` 就改用 `reshape`（或先 `.contiguous()` 再 view），多付一次拷贝。

**成因**：把 `is_contiguous()` 当成了 view 的判据。

**规避**：按 contiguity-like condition 判——判据是**连续段**，不是整体连续性。`(4,4)[:, ::2]`（非连续）能直接 `.view(8)`。

### 失败模式 3：以为「reshape 一定会拷贝」

**现象**：为了「避免 reshape 拷贝」，手写一堆 `view` 与 `transpose` 的组合。

**成因**：`reshape` 的先尝试 view、失败才拷贝这一点没被记住。

**规避**：需要「保住共享存储」时用 `view`（失败即报错，不会偷偷拷）；只需要「形状对」时用 `reshape`，并在关键路径上用 `data_ptr()` 验证它有没有拷。

### 失败模式 4：用 `data_ptr()` 判共享存储

**现象**：`full[2:6].data_ptr() != full.data_ptr()`，于是判定「它是一个新张量」。

**成因**：`data_ptr()` 已经把 `storage_offset` 算进去了。

**规避**：用 `untyped_storage().data_ptr()` + `storage_offset()`。实测这两个判据在上述例子里给出**相反**结论。

## 5. 实测数据

**测量条件**：`code/03_strides_and_views.py` 经 `make run` 在本机跑出，选中的解释器是
`/Users/cachio/miniconda3/bin/python` = **torch 2.11.0 / python 3.12.9 / CPU-only**。全部断言通过（A 36 / B 3 / C 9，共 48 条）。

> **★ 限定语（本域强制）**：下面每个数字都是**真实 PyTorch**跑出来的（不是迷你实现），**必须绑定版本**。A 类的参照 `ref_*` 来自 `verify.py`——它**跑在默认 python3 下、不 import torch**，其产出是**推导**，与这里记录的实测值是**两档不同的证据**，不得互相背书。
> **本机另有 2.6.0 / 2.7.1 / 2.8.0 / 2.13.0 四个解释器**（连被选中的那个共五个）；**没有 2.9 / 2.10 / 2.12 的解释器**，这三个 minor 不声称覆盖。

**（1）A 类对拍**（参照预测 = torch 实测，共 31 条对拍断言，全部一致）：

| 对拍项 | 条数 | 结果 |
|---|---|---|
| `is_contiguous()` | 16 | 全部一致 |
| `view()` 可行性 | 10 | 全部一致 |
| stride 传播（slice / transpose / expand） | 6 | 全部一致 |

**（2）B 类：非连续切片的访存字节模型**（张量：8,000,000 个 float32 = 32.0 MB 的 storage）：

| 操作 | 耗时（最快一组，ms） |
|---|---|
| 造视图 `x[::2]`（不拷贝） | **0.0008** |
| `x[::2].sum()` | 0.3974 |
| `x[:N/2].sum()` | 0.1087 |
| `x.sum()`（全量连续） | 0.3648 |
| `x[::2].contiguous()`（真的拷贝，16 MB） | 0.4597 |

模型预测 `T(x[::2].sum()) / T(x[:N/2].sum()) ≈ 2.0`（两者求和元素数相同，但跨步那一侧要走过整块 storage：32 MB vs 16 MB）；**实测 3.66**——比模型更高，因为跨步访问的缓存利用率更差。**模型给出的是下界，不是等号**，这一点必须明写。

「造视图 vs 真的拷贝」差 **三个数量级**（0.0008 ms vs 0.4597 ms）——这是 §2.1「连续时 `contiguous()` 连 dispatcher 都不进」在计时侧的同一条结论。

**（3）C 类的失败行为**（每条都真的跑出来）：`as_strided` 越界报错原文、wrapper 子类同一几何被接受且判连续、`view(16)` 的报错原文、`reshape` 的 fallback（指针变化）、`(4,6).t().reshape(6,4)` 共享存储、`data_ptr()` 与 `untyped_storage().data_ptr()` 结论相反。

**（4）版本敏感条目**：本文的全部判据与失败行为在 2.6.0 / 2.7.1 / 2.8.0 / 2.11.0 / 2.13.0 上**实测一致**（五个版本的程序输出除时序数字外逐行相同）。判据里唯一与版本有关的行号引用是 `c10/core/Contiguity.h` 的 `:15`（**2.13.0 文件的行号**）。

## 6. 小结与延伸

- 张量是 `(storage, offset, sizes, strides)`；**形状不是布局，stride 才是**。
- `view` 的判据是 **contiguity-like condition**，不是 `is_contiguous()`；`reshape` **优先 view、不行才拷贝**。
- `is_contiguous()` **只吃 sizes/strides/numel**：`numel == 0` 恒 True、**跳过 `size == 1` 的维**、既不查边界也不查重叠。
- 三个可复用的判据：`contiguous()` 在已连续时返回 `self` 且 **0 次 dispatcher 拦截**；`as_strided` **自带边界检查**（造反例要用 wrapper 子类）；判共享存储用 **`untyped_storage().data_ptr()` + `storage_offset()`**。
- 本文的 A 类参照是 `code/verify.py` 里的**独立实现**（不 import torch），它可以在什么都没装的机器上复核本主题的判据。
- 下一站：[04-custom-operators](04-custom-operators.md)（自定义算子的 schema 与 kernel 注册）。

## 参考

- 实测程序：`code/03_strides_and_views.py`（三类断言齐全，失败非零退出）
- 纯 python3 复核入口：`code/verify.py`（`ref_compute_contiguous` / `ref_view_compatible` / `ref_runs` / `ref_stride_*`，**不 import torch**）
- 外部事实与版本锚点：`code/specs.py`
- 上游：`10-foundations/numerical-linear-algebra`（行主序与 stride 的一般理论）、`20-gpu-architecture/memory-system/02-coalescing-and-cache-line`（stride 怎么影响 GPU 访存）
- 下游：`80-model-architectures`（`transpose`/`permute` 密集的注意力实现）

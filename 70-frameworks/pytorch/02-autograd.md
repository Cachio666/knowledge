---
title: autograd：反向图固化了什么
slug: pytorch-autograd
domain: 70-frameworks
topic: pytorch
level: 2
difficulty: intermediate
prerequisites:
  - 70-frameworks/pytorch/01-dispatch-and-aten
tags: [pytorch, autograd, grad-fn, saved-tensors, inference-mode, no-grad, retain-graph]
related:
  - 70-frameworks/pytorch/03-tensor-memory-model
  - 90-training/optimizers/01-sgd-and-momentum
code:
  - code/02_autograd_graph.py
status: stable
updated: 2026-09-17
---

# autograd：反向图固化了什么

> 前向时 autograd 建了一张反向图，而这张图里**被固化下来的东西就是 saved tensors**——每个算子的反向公式需要什么，它就得在前向时把什么留下来。本文把「留什么、留多少字节、什么时候留不了」逐条实测出来，并纠正三个流传很广的说法：`inference_mode` 的张量出作用域后**读是正常的**、`retain_graph` 的默认值是 **`create_graph`**、`autograd.grad()` **不写** `.grad`。

## 0. 背景与动机

### 一个具体的对照场景

同一个 `(w * 2 + 1).relu().sum()`，训练时要反向、推理时不要。两种写法：

```python
y = (w * 2 + 1).relu().sum()
y.backward()              # 训练：需要保存中间结果，反向时还要读回来
```

```python
with torch.inference_mode():
    y = (w * 2 + 1).relu().sum()      # 推理：不建反向图、不记版本号、不追踪 view
```

差别不止是「快一点」。**前向固化了什么，决定了后面哪些操作被禁止**：`inference_mode` 省下来的那部分记录能力，正是它出作用域后**原地改**与**参与反向**被拒的原因。

### 本文在本域主线上的位置：**②代价项**

> **反向图保存的中间结果，就是 autograd 这一层「固化」下来的东西。**
> 它与「图」那条路的主线（guard 固化假设）是同一个问题的两个面：**省下来的每一步，都是一条将来可能被打破的约束**。

本文**不**讨论 `torch.compile` 如何用 guard 固化形状/类型（见 [04-custom-operators](04-custom-operators.md) §3），也**不**讨论优化器怎么用梯度（正文在 [90-training/optimizers/01-sgd-and-momentum](../../90-training/optimizers/01-sgd-and-momentum.md)，此处回链）。

### 最容易记错的三条

1. 「**`inference_mode` 的张量出作用域后就失效了**」——**错**。读操作（`.shape` / `.sum()` / `.item()` / `.numpy()` / `.clone()` / `.detach()`）全部正常；只有**原地修改**与**被 autograd 记录**（save-for-backward）被拒。
2. 「**`retain_graph` 默认 `False`**」——**不准确**。源码里是 `if retain_graph is None: retain_graph = create_graph`。
3. 「**`autograd.grad()` 会写 `.grad`**」——**错**。它只**返回**梯度；写 `.grad` 的是 `backward()`，而且是**累加**。

## 1. 原理：反向图怎么建

### 1.1 `grad_fn` 挂在**输出**上

以 `y = (w*2 + 1).relu().sum()` 为例（`w` 是 `requires_grad=True` 的叶子），实测的节点链是：

```
SumBackward0    → [(ReluBackward0, 0)]
ReluBackward0   → [(AddBackward0, 0)]
AddBackward0    → [(MulBackward0, 0), (None, 0)]
MulBackward0    → [(AccumulateGrad, 0), (None, 0)]
AccumulateGrad  → []
```

三条读法：

- **`grad_fn` 挂在输出张量上**（`y.grad_fn` 是 `SumBackward0`），不在输入上。
- **`next_functions` 是 `[(Node, int)]`**，那个 `int` 是「这个 node 的第几个输入」。
- **`None` 表示该输入不需要梯度**（这里是常量 1 与标量 2）；**叶子输入对应 `AccumulateGrad`**——它就是把梯度累加进 `.grad` 的那个节点。

### 1.2 `is_leaf` 的形式判据

口语说法是「**用户创建的**张量是叶子」。官方 docstring 用的正是这个措辞，但**形式判据**是：

```
is_leaf  ⟺  (grad_fn is None)  or  (requires_grad is False)
```

两者在 **view** 上分道扬镳：`w.view(-1)` 是用户创建的，但实测 `is_leaf = False`、`_is_view() = True`、`_base is w = True`。**凡是要对 view 判 `is_leaf`，必须用形式判据。**

`code/02_autograd_graph.py` 的 §A3 把形式判据写进程序、逐例与 torch 对拍（6/6 一致），§A4 单独把「口语表述会给出相反答案」这件事跑出来。

### 1.3 `inference_mode` 与 `no_grad` 的差别表（本机五版本实测）

| 观察点 | `no_grad` | `inference_mode` |
|---|---|---|
| `torch.is_inference(x)` | False | **True** |
| `_version` 版本计数器 | 有；`add_` 后 0→1 | **完全没有**：**在 mode 内**读 `_version` 就抛 `RuntimeError: Inference tensors do not track version counter.` |
| view 追踪 | 有（`_is_view()` True、`_base` 非 None） | **无**（都 False/None） |
| 出作用域后**读** | 正常 | **正常**（`.shape`/`.sum()`/`.item()`/`.numpy()`/`.clone()`/`.detach()` 全通） |
| 出作用域后**原地改** | 正常 | **拒绝**：`Inplace update to inference tensor outside InferenceMode is not allowed.` |
| save-for-backward | 正常 | **拒绝**：`Inference tensors cannot be saved for backward. …` |
| forward-mode AD | 仍可用 | **⚠ 版本相关，见 §3.3** |

官方 docstring 的措辞（**2.13.0 才有后半句**）：*「removes additional overhead by **disabling view tracking and version counter bumps**. It is also more restrictive, in that **tensors created in this mode cannot be used in computations recorded by autograd**.」*

## 2. 执行映射：一次前向到底保存了什么

### 2.1 唯一通用的观测手段：`saved_tensors_hooks`

```python
with torch.autograd.graph.saved_tensors_hooks(pack_fn, unpack_fn):
    loss.backward()
```

`pack_fn` 在前向保存张量时被调用，`unpack_fn` 在反向取回时被调用。**这是唯一通用的观测手段**——原因很具体：`grad_fn.saved_tensors` 在 **C++ 定义的节点上根本不存在**。实测访问 `MmBackward0`（`(A@B).sum()` 的中间节点）的 `saved_tensors` 会抛 `AttributeError`；只有自定义 `torch.autograd.Function` 的节点才有这个属性（本机五个版本实测一致）。

### 2.2 保存量由**反向公式**决定，没有通用数字

本机实测（float32，`make run` 的 §B 表，torch 2.11.0）：

| 算例 | pack 次数 | 保存字节 | 为什么 |
|---|---|---|---|
| `(x*x).sum()`（x 是 (3,)） | **2** | 24 | `mul` 的反向需要**另一个操作数**；两个操作数是同一个张量也**各存一份** |
| `(k*2).sum()`（标量乘） | **0** | 0 | 标量不是张量，没有东西要存 |
| `(A@B).sum()`（A(3,4), B(4,5)） | **2** | 128 | `mm` 的反向需要 A 与 B：48 + 80 字节 |
| `(q*2+1).relu().sum()`（q 是 (4,)） | **1** | 16 | 只有 `relu` 保存了它的**输出**；mul（标量）与 add、sum 都不保存 |
| `(r*s).sum()` | 2 | 32 | 两个操作数各一份 |
| `m.pow(2).sum()` | 1 | 16 | 反向需要**底数** |
| `n.exp().sum()` | 1 | 16 | 反向需要**输出** |
| `o.sum(1).sum()`（o 是 (4,4)） | 0 | 0 | `sum` 的反向只需形状 |
| `p.sigmoid().sum()` | 1 | 16 | 反向需要**输出** |

**所以「一次前向保存 N 个中间结果」这类通用值是假的。** 正确的说法是：**逐算子查它的反向公式**。上表九行里有 0、1、2 三种答案。

## 3. 实现与演进

### 3.1 `backward()` 与 `autograd.grad()` 的分工

| | 写 `.grad` 吗 | 累加吗 | 返回什么 |
|---|---|---|---|
| `y.backward()` | **写** | **累加**（实测连调两次 `.grad` 从 `[1,1,1]` 变 `[2,2,2]`） | `None` |
| `torch.autograd.grad(y, x)` | **不写**（实测跑完 `x.grad is None`） | — | 梯度元组 |

这条区别在写训练循环时很实际：**用 `autograd.grad` 做梯度裁剪/二阶导时，别指望 `.grad` 里有东西**（训练循环里这两件事的正文在 [90-training/autograd-and-backward/03-gradient-lifecycle](../../90-training/autograd-and-backward/03-gradient-lifecycle.md)，此处回链）。

### 3.2 `retain_graph` 的默认值

源码里（`torch/autograd/__init__.py`，**2.13.0 行号 390**，2.7.1 在 348、2.6.0 在 342）：

```python
if retain_graph is None:
    retain_graph = create_graph
```

签名里两个默认值不同：`retain_graph: bool | None = None`、`create_graph: bool = False`。所以：
**`retain_graph` 的默认值是 `create_graph`（而不是 `False`）；因为 `create_graph` 默认 `False`，实践中表现为 `False`。**

不写 `retain_graph` 连着 backward 两次，第二次会抛 `Trying to backward through the graph a second time`（本机实测，五个版本一致）。

### 3.3 ★ 一处**版本行为相反**的地方：`inference_mode` 里的 forward-mode AD

| 版本 | `inference_mode` 内 `torch.func.jvp` 的 tangent |
|---|---|
| 2.6.0 | `[0.0, 0.0, 0.0]`（**静默全零，不抛异常**） |
| 2.7.1 | `[0.0, 0.0, 0.0]` |
| **2.8.0** | `[0.0, 0.0, 0.0]` |
| **2.11.0** | `[0.0, 0.0, 0.0]` |
| 2.13.0 | `[2.0, 2.0, 2.0]`（正确） |

docstring 也印证了这次变更：2.7.1 的文案里写着「entering inference_mode also **disables** … forward-mode AD」，**2.13.0 把这句删掉了**。

> **变更点只能定位到「2.11 与 2.13 之间」**——本机**没有 2.12 的解释器**，所以「2.12 改的」只是本仓库的**假设**，不得写成已核实。（2.8.0 实测也是全零，所以变更点被夹在 2.11 与 2.13 之间。）
> **最危险的形态是 2.6/2.7/2.8/2.11 那种：不抛异常、静默给出全零 tangent。** 一个「看起来能跑」的正向模式梯度是错的，这类错误不会自己报出来。

### 3.4 用自定义 `Function` 时：`save_for_backward` 是对外契约

只有自定义 `torch.autograd.Function` 的节点才暴露 `saved_tensors`。这也让它成为**唯一能在反向里「读回自己保存的东西」的稳定接口**——本文 §4 的失败模式 3 就是在 `forward` 里存了一个 inference 张量。

## 4. 踩坑与失败模式

### 失败模式 1：以为 `inference_mode` 张量出作用域就失效

**现象**：把推理期产生的张量传出去，看到「出作用域后不能用」的说法，于是加了一堆 `.clone()`。

**成因**：把「**被 autograd 记录**被拒」误读成「**一切使用**被拒」。

**规避**：按实测的边界来——**读全通**（`.shape`/`.sum()`/`.item()`/`.numpy()`/`.clone()`/`.detach()`）；**只有两类被拒**：①原地修改（`Inplace update to inference tensor outside InferenceMode is not allowed.`）②参与要被记录的计算（save-for-backward）。另外，`_version` **在 mode 内就已不可读**，不是「出作用域才出问题」。

### 失败模式 2：给「一次前向保存几个中间结果」编一个通用值

**现象**：看到 `(x*x).sum()` 保存 2 个，就写「一次前向保存 2 个中间结果」。

**成因**：把某个算例的数字当成了普遍规律。

**规避**：按 §2.2 的表逐算子给数。想知道某个具体模型保存了多少，用 `saved_tensors_hooks` **测**它——这个手段是通用的，数字不是。

### 失败模式 3：把 inference 张量存进了要参与反向的计算

**现象**：`ctx.save_for_backward(inf_tensor)` 或前向用 inference 张量参与 `requires_grad` 的计算，反向时报
`Inference tensors cannot be saved for backward.`（2.13.0 的文案更长，见 `code/specs.py`）。

**成因**：`inference_mode` 省掉了版本计数与 view 追踪，所以 autograd **无法**保证反向时这个张量还有效——于是干脆禁止保存。

**规避**：跨出 `inference_mode` 的边界时**显式 `.clone()`**（错误文案自己也是这么建议的），或者把整段推理放在 `no_grad()` 里而不是 `inference_mode()` 里。

### 失败模式 4：以为 `retain_graph=False` 是写死的默认值，或以为 `autograd.grad` 会写 `.grad`

**现象**：①读源码时看到 `retain_graph = create_graph` 与自己的记忆冲突；②用 `autograd.grad` 拿梯度后去读 `.grad`，发现是 `None`。

**成因**：默认值是「另一个参数」，且写 `.grad` 只是 `backward()` 的行为。

**规避**：记两条判据——`retain_graph` 默认跟随 `create_graph`；写 `.grad` 的只有 `backward()`（且**累加**）。需要在一段图里取多次梯度，显式传 `retain_graph=True`。

## 5. 实测数据

**测量条件**：`code/02_autograd_graph.py` 经 `make run` 在本机跑出，选中的解释器是
`/Users/cachio/miniconda3/bin/python` = **torch 2.11.0 / python 3.12.9 / numpy 2.3.4 / CPU-only**。全部断言通过（A 13 / B 21 / C 14，共 48 条）。

> **★ 限定语（本域强制）**：下面每个数字都是**真实 PyTorch**跑出来的（不是迷你实现），但**必须绑定版本**。**本机另有 2.6.0 / 2.7.1 / 2.8.0 / 2.13.0 四个解释器**（连被选中的那个共五个）；**没有 2.9 / 2.10 / 2.12 的解释器**，这三个 minor 不声称覆盖。

**（1）A 类的独立参照是有限差分**（纯 numpy 中心差分 `(f(x+ε) − f(x−ε))/2ε`，ε=1e-6，float64）：

| 项 | 最大绝对误差 | 量级 |
|---|---|---|
| `d/dX`（X 是 (3,4)） | 2.356e-09 | 1.763 |
| `d/dW`（W 是 (4,5)） | 3.569e-09 | 3.080 |
| `d/db`（b 是 (5,)） | 1.497e-09 | 3.000 |

**这条路径与 autograd 引擎完全不同**（数值微分 vs 反向图），任一边写错都会立刻暴露。反证也跑了：把参照的一列置零后，同一条「相对误差 < 1e-5」判据立刻判为不通过（§A2）。

**（2）saved tensors 实测表**：见 §2.2（九行，0/1/2 三种答案）。

**（3）`is_leaf` 形式判据**：6 个例子全部与 torch 一致（用户创建的叶子 / 不需要梯度 / `w*2` / `w.view(-1)` / `w.detach()` / `no_grad` 下的 `w*2`）。

**（4）版本敏感条目**（本机五版本逐一实测）：

| 条目 | 2.6.0 | 2.7.1 | 2.8.0 | 2.11.0 | 2.13.0 |
|---|---|---|---|---|---|
| inference_mode 内 `torch.func.jvp` 的 tangent | 全零 | 全零 | **全零** | **全零** | **正确** |
| `torch/autograd/__init__.py` 的 `retain_graph = create_graph` 行号 | 342 | 348 | 未逐行核 | 未逐行核 | 390 |
| `torch/_tensor.py` 的 `_disabled_torch_dispatch_impl` 行号 | 1654 | 1674 | 未逐行核 | 未逐行核 | 1536 |
| save-for-backward 的错误文案 | 短版 | 短版 | 短版 | 短版 | 长版（多一句「Please do not use …」） |

> **第 1 行是本文唯一一处「版本行为相反」**：文档照实测写，**不做外推**。变更点只能定位到 2.11 与 2.13 之间。

## 6. 小结与延伸

- 反向图挂在**输出**的 `grad_fn` 上；`next_functions` 里的 `None` = 该输入不需要梯度，叶子输入 = `AccumulateGrad`。
- `is_leaf` 的**形式判据**是 `grad_fn is None or not requires_grad`；口语的「用户创建的」在 **view** 上会给出相反答案。
- **saved tensors 就是 autograd 这一层固化下来的东西**：留什么由**反向公式**决定，逐算子不同（本机实测 0/1/2 三种），**没有通用数字**。
- 三条被实测推翻的常见说法：`inference_mode` 张量出作用域后**读正常**；`retain_graph` 默认是 **`create_graph`**；`autograd.grad()` **不写** `.grad`。
- **`inference_mode` 里的 forward-mode AD 五个版本行为不一致**，且老版本是「静默全零」——最危险的失败形态。
- 下一站：[03-tensor-memory-model](03-tensor-memory-model.md)（view 与 stride 的假设）、[04-custom-operators](04-custom-operators.md)（自定义算子怎么声明自己的反向）。

## 参考

- 实测程序：`code/02_autograd_graph.py`（三类断言齐全，失败非零退出）
- 纯 python3 复核入口：`code/verify.py`（**不 import torch**，其中 `ref_is_leaf` 是 `is_leaf` 判据的独立实现）
- 外部事实与版本锚点：`code/specs.py`
- 上游：`10-foundations/numerical-linear-algebra`（链式法则与数值微分）
- 下游：`90-training`（优化器怎么用梯度 ⇒ [01-sgd-and-momentum](../../90-training/optimizers/01-sgd-and-momentum.md)）

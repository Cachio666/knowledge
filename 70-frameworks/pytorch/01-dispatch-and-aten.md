---
title: dispatch 与 aten：eager 那条路的层数是可数的
slug: pytorch-dispatch-and-aten
domain: 70-frameworks
topic: pytorch
level: 2
difficulty: intermediate
prerequisites: []
tags: [pytorch, dispatch, dispatcher, aten, dispatch-key, torch-dispatche-mode, composite-implicit-autograd]
related:
  - 60-compilers/inductor/01-graph-capture
  - 70-frameworks/pytorch/04-custom-operators
  - 30-gpu-programming/cuda/01-execution-model
code:
  - code/01_dispatch_layers.py
status: stable
updated: 2026-09-17
---

# dispatch 与 aten：eager 那条路的层数是可数的

> 在 PyTorch 里写 `a + b`，从来不是「调用了一个加法函数」——它经过 **Python 绑定 → dispatcher → 若干 dispatch key → kernel**，而这条链上的**层数是可数的**：一次 `a + b` 恰好触发 **1 次** `__torch_dispatch__` 拦截，而 `+a`、`a.shape`、已同 dtype 的 `a.float()` 是 **0 次**。层数可数，才谈得上「把它折叠进图里能省多少」。

## 0. 背景与动机

### 一个具体的对照场景

同一句 `c = a + b`，在两种运行方式下的代价完全不同：

| | 每次调用经过 | 代价 |
|---|---|---|
| **eager**（默认） | 完整走一遍 Python 绑定 → dispatcher → autograd → kernel | **每一层每次都付** |
| **图**（`torch.compile`） | 把这条链折叠一次，之后重放 | 必须**固化假设**；假设被打破就要重新追踪/重编译 |

本文只处理左边那一列：**eager 的路到底有几层、每层谁来付**。右边那一列（Dynamo 怎么捕获、Inductor 怎么融合与 codegen）的正文在 [60-compilers/inductor](../../60-compilers/inductor/01-graph-capture.md)，本文只回链。

### 本文在本域主线上的位置：**②代价项**

本域的主线是「**框架要把哪些假设固化进图**」（见 `70-frameworks/README.md` 与 roadmap）。本文的归属是主线里的**代价项**：

> **eager 那条路的层数，就是「不固化」的代价。**
> 说得出「一次调用 = 1 次拦截」，才说得出图把它省成了几次。

本文**不**讨论 guard 与重编译（那是 [04-custom-operators](04-custom-operators.md) §3 与 `60-compilers/inductor` 的事），也**不**讨论 kernel 实现（回链 `40-kernels/*`）。

### 为什么这里容易记错

「PyTorch 是动态图，所以它没有图」——这句话半对半错，而错的那一半正好掩盖了本文的主角。真实结构是：**eager 有一个「按 key 分发的调度层」，它是可数、可观察、可用 dispatch mode 观测的**。本文 §1 把它数出来，§2 说明 key 集合是一个 64 位整数，§3 讲观测它时最容易搞错的一个顺序问题，§4 给四个能真的跑出问题的失败模式。

## 1. 原理：一次调用到底经过几层

### 1.1 三个层次，别混

先约定三个名字（本文只用这三个）：

- **`__torch_function__`**：在 **torch API 层**（`torch.Tensor` 的方法、`torch.*` 函数）。它拦截的是「Python 侧的 API 调用」。
- **`__torch_dispatch__` / `TorchDispatchMode`**：在 **aten 层**（dispatcher）。它拦截的是「**算子**」。
- **kernel**：真正干活的实现（CPU/CUDA/…）。

判据是实测出来的，而且很干净：对一个只定义了 `__torch_function__` 的子类，`x.shape` 触发 `__torch_function__` **1 次**、`torch.Tensor.dim(x)` 触发 **1 次**，而两者合起来触发 `__torch_dispatch__` **0 次**（`code/01_dispatch_layers.py` §C4，本机五个版本实测一致）。也就是说：**`x.shape` 根本不进 dispatcher**。

### 1.2 层数是可数的：一次表达式 = 几次拦截

下面是本机实测的**完整对照表**（`make run` 的 §1 输出，torch 2.11.0；**2.6.0 / 2.7.1 / 2.8.0 / 2.13.0 上这组数一字不差**——五个版本实测）：

| 表达式 | mode 拦截次数 | 拦到的算子 | profiler 看到的 aten op |
|---|---|---|---|
| `a + b` | **1** | `aten.add.Tensor` | `aten::add` |
| `torch.add(a, b)` | **1** | `aten.add.Tensor` | `aten::add` |
| `a.add(b)` | **1** | `aten.add.Tensor` | `aten::add` |
| `a.T` | 1 | `aten.permute.default` | `aten::numpy_T`, `aten::permute`, `aten::as_strided` |
| `a[0]` | 1 | `aten.select.int` | `aten::select`, `aten::as_strided` |
| `a.sum()` | 1 | `aten.sum.default` | `aten::sum`, `aten::as_strided`, `aten::fill_` |
| `(ag + b).sum()`（带 requires_grad） | **2** | `aten.add.Tensor`, `aten.sum.default` | `aten::add`, `aten::sum`, … |
| 一维 `v @ w` | 1 | **`aten.dot.default`** | `aten::matmul`, `aten::dot`, `aten::empty` |
| 二维 `A @ B` | 1 | `aten.mm.default` | `aten::matmul`, `aten::mm`, `aten::resolve_conj`×3 |
| `+a` | **0** | — | `aten::positive` |
| `a.float()`（已同 dtype） | **0** | — | `aten::to` |
| `a.shape` | **0** | — | — |

三个直接的结论：

1. **一次 `a + b` 恰好 1 次拦截，带 `requires_grad` 也是 1 次**。autograd 不是「额外一层拦截」，它是在同一次 dispatch 里被选中的 key（§2）。
2. **一维 `a @ b` 走的是 `aten.dot.default`，不是 mm/matmul**。想数层数就得按实际算子数，不能按你脑子里的数学运算数。
3. **「0 次」有两种成因，别混**（这是本文 §4 的第一条失败模式）：`a.shape` 是**根本没进 dispatcher**，而 `+a` / `a.float()` 是**进了 dispatcher 但被内联展开**——第二列与第三列的差异就是证据。

### 1.3 同一入口的两个绑定

一个常见的误解是「`a + b` 会调用 Python 里的 `torch.Tensor.add`」。实测：

- `torch.Tensor.add is torch._C.TensorBase.add` → **True**；
- `torch/_tensor.py` 里**没有** `def add`（全文搜索 0 命中）。

也就是说 `__add__` 与 `Tensor.add` 是**同一个 dispatcher 入口的两个 C 绑定**，Python 层没有中间函数。`torch.ops.aten.add` 是**另一条可选入口**，`a + b` 不经过它。

这条对性能直觉有直接影响：**「用 `a.add(b)` 代替 `a + b` 会不会更省一层」——不会，它们同价**（§5 的计时也印证：差价 0.060 μs，远小于一层 Python key 的 3.132 μs）。

## 2. 执行映射：keyset、64 位整数、线程局部栈

### 2.1 普通 CPU 张量的 keyset 里没有 Python

`torch._C._dispatch_keys` 直接把它打印出来（本机五个版本实测一致）：

```
torch.randn(3)                                  → DispatchKeySet(CPU, ADInplaceOrView, AutogradCPU, AutocastCPU)
class S(torch.Tensor): pass（空子类）的实例      → DispatchKeySet(CPU, ADInplaceOrView, AutogradCPU, AutocastCPU)
覆盖了 __torch_dispatch__ 的子类实例             → DispatchKeySet(CPU, Python, ADInplaceOrView, AutogradCPU, AutocastCPU, PythonTLSSnapshot)
```

**只有覆盖了 `__torch_dispatch__` 的子类**才会把 `Python` 与 `PythonTLSSnapshot` 加进去；空子类不会。（这条是本项目订正过的：容易写成「张量子类就会有 `Python` key」，实测不是。）

### 2.2 keyset 是一个 64 位整数，而且不是 1 key = 1 bit

规格层（来源见 `code/specs.py`）：

- `c10/core/DispatchKeySet.h` 里是 `uint64_t repr_ = 0;`（**2.13.0 行号 513**；2.6.0/2.7.1 在 510）；
- `DispatchKey` 本身是 `enum class DispatchKey : uint16_t`（**2.13.0 行号 137**）；
- 同一文件的设计注释里明确写着：**bits 与 dispatch key 之间不必然一一对应**——低位是 backend bits、高位是 functionality bits；
- backend 数量上限是 **16**（`DispatchKey.h` 的 `static_assert`，2.13.0 行号 553–556）。

**为什么这个数字重要**：它解释了「为什么加一个后端要做位复用」，也解释了为什么 keyset 能塞进一个 64 位整数还留有余量——**16 个后端 × 若干功能位**。

> ⚠ 上面四个行号都是 **2.13.0 的行号**。本机实测的行号差异（`code/specs.py` 的 `VERSION_LINE_SHIFT`）：`DispatchKeySet.h` 的 `repr_` 在 2.6.0/2.7.1 是 510，`DispatchKey.h` 的枚举在 2.6.0/2.7.1 是 135。**引用行号必须带版本**，否则读者在别的版本上复现不了。

### 2.3 `TorchDispatchMode` 走的是另一套：线程局部栈

`TorchDispatchMode` **不是** keyset 里的一位。实测：`torch._C._len_torch_dispatch_stack()` 在无 mode 时是 **0**、进入 mode 后是 **1**、退出后回到 **0**。它是**线程局部栈**，两个 mode 叠加时是**后进先出**（内层先看到）。

这个区别的实用后果：**mode 的开启与关闭本身有成本**——本机实测「进/出 mode（含建栈）」是 6.421 μs，而 mode 内一次 `a + b` 是 4.036 μs，裸 `a + b` 只要 0.904 μs（§5）。所以 **mode 是诊断工具，不是生产路径**。

## 3. 实现与演进：观测这条链的三种方法，与一次顺序订正

### 3.1 三种观测方法，成本与盲区都不同

| 方法 | 看到什么 | 盲区 |
|---|---|---|
| `TorchDispatchMode`（Python key） | 被 dispatch 的**算子** | **看不到 CompositeImplicitAutograd 的算子本身**（§4） |
| `torch.autograd.profiler` / `torch.profiler`（C++ RecordFunction 层） | C++ 里真正被调用的 aten op | 会多出实现细节（`as_strided`/`fill_`/`resolve_conj`/`numpy_T`） |
| `torch._C._dispatch_dump_table` / `_dispatch_has_kernel`（注册表） | **注册层级**：这个算子是不是 composite、有没有真 kernel | 是静态信息，不含调用次数 |

**正确的用法是三者互证**，不是任选其一。`code/01_dispatch_layers.py` 的 §A1 就是这么做的：**先用注册表预测「这个算子自己会不会被拦到」，再看 mode 的名单**。本机 6/6 个算子被正确预测（`aten::add.Tensor` / `aten::sum` / `aten::relu` 是 `kernel` → 自己可见；`aten::reshape` / `aten::contiguous` / `aten::positive` 是 `math kernel` → 自己不可见）。

> **判据必须写成「那个算子自己是否出现在名单里」，不能写成「拦截数是否为 0」。**
> 这是本项目第一版被实跑推翻的一处：composite 算子会**内联展开**，展开出来的子算子照样被拦。`aten.reshape` 在连续输入下被拦 **1 次**——拦到的是 `aten.view.default` 而不是它自己。

### 3.2 mode 与子类 `__torch_dispatch__` 的顺序：**mode 先**

两者同时在栈上时，调用序列是 `['mode', 'subclass']`（本机实测，2.6.0 / 2.7.1 / 2.8.0 / 2.11.0 / 2.13.0 一致）。

> ⚠ **这条的唯一依据是实测。**
> `c10/core/DispatchKey.h` 里确实有一条讲到顺序的注释（「Must be BELOW Python so that TorchDispatchModes … fire first」），但它①**只存在于 2.13.0**（2.6/2.7/2.8/2.11 的同一文件里 0 命中），②讲的是 **C++ 的 `Fake` 这个 key 在枚举里的位置**（为了让 C++ FakeTensor 与 Python FakeTensor 的次序一致），不是这条结论的判据本身。把它当依据属于「引文与断言的作用域不匹配」。本文只在实测那一行给出结论。

### 3.3 用子类观测时的一个硬要求：先把 Python key 关掉

在子类的 `__torch_dispatch__` 里直接 `return func(*args)` 会**无限递归**（本机五个版本都实测到 `RecursionError`）。这不是理论问题，本项目真的踩到了。

torch 自己的实现给出了标准做法：`torch/testing/_internal/logging_tensor.py` 里的 `LoggingTensor` **先 unwrap 再调用**（`func(*tree_map(unwrap, args))`），而同文件的 `LoggingTensorMode` 就直接 `func(*args)`——**两条路不一样，是因为 mode 与子类的重入行为不同**。

可用的三条路（本机实测）：

| 做法 | 结果 |
|---|---|
| `func(*args)`（不处理） | `RecursionError` |
| 用 `a.as_subclass(torch.Tensor)` 拆包 | **同样 `RecursionError`**（`as_subclass` 自己也要进 dispatcher） |
| `with torch._C._DisableTorchDispatch(): func(*args)` | ✅ 正常返回且数值正确 |

`torch.overrides.enable_reentrant_dispatch()` 也是官方给的口子（注释里写明它「technically public API」），但它**开启**的是重入，与上面的问题不是同一件事——本文不把它当作解决方案。

## 4. 踩坑与失败模式

### 失败模式 1：CompositeImplicitAutograd 的算子对 dispatch mode 不可见

**现象**：拦截 `aten.reshape`，观察到的却是 `aten.view.default`；拦截 `aten.contiguous`，观察到 `aten.clone.default`。**它不报错，只是让你看不到。**

**成因**：`aten::reshape` / `aten::ravel` / `aten::contiguous` 这一类算子注册为 **CompositeImplicitAutograd**——dispatcher 会把它们的实现**内联展开**，而内联发生在 Python key 之前。实测的注册层级（`_dispatch_dump_table`）就写着 `math kernel`。

本机实测的完整名单（`code/01_dispatch_layers.py` §C1）：

| 调用 | mode 看到 | 注册层级 |
|---|---|---|
| `aten.reshape.default(连续, [16])` | `aten.view.default` | math kernel |
| `aten.reshape.default(非连续, [16])` | `aten.clone.default`, `aten._unsafe_view.default` | math kernel |
| `aten.ravel(非连续)` | `aten.clone.default`, `aten.view.default` | math kernel |
| `aten.ravel(连续)` | `aten.view.default` | math kernel |
| `aten.contiguous(非连续)` | `aten.clone.default` | math kernel |
| `aten.contiguous(连续)` | **（0 次）** | math kernel |
| `aten.relu(非连续)` | `aten.relu.default` | kernel |
| `aten.sum.default(非连续)` | `aten.sum.default` | kernel |

**规避**：要观测 composite 算子，换一条通道（profiler 看得到 C++ 层的 `aten::positive` / `aten::to`）；或者查注册层级，别假设「mode 没看到 = 没发生」。

> 顺带一条同样是实测的：**`aten.ravel` 只有非连续输入才有 `clone`**，连续输入下只有 `aten.view.default`。写成「`ravel` 一定会 clone」是错的。

### 失败模式 2：`_dispatch_has_kernel` 必须写重载名——但**只对 aten 命名空间成立**

**现象**：`torch._C._dispatch_has_kernel("aten::add")` 返回 **False**，写成 `"aten::add.Tensor"` 才是 **True**。

**成因**：算子名带重载；`aten::add` 有多个重载，裸名解析不到唯一实现。

**规避**：查 aten 算子时写全 `aten::<name>.<overload>`。但要注意**这条规矩有边界**：对**自定义算子**，`_dispatch_has_kernel("kb04::swiglu")`（裸名）是 True，而加上 `.default` 反而抛
`RuntimeError: default is not a legal overload name for aten operators`——因为它按 **aten 的重载名规则**解析（详见 [04-custom-operators](04-custom-operators.md) §4）。

### 失败模式 3：以为「张量子类」就会有 Python key

**现象**：写了一个 `class MyTensor(torch.Tensor): pass` 的子类，以为它的 keyset 里会出现 `Python`，从而能用 `__torch_dispatch__` 兜住一切。

**成因**：`Python` key 是**因为覆盖了 `__torch_dispatch__`** 才加进去的，不是因为「是子类」。

**规避**：要么覆盖 `__torch_dispatch__`（并在里面按 §3.3 关掉 Python key 再调用），要么就别指望它参与 dispatcher 拦截。

### 失败模式 4：把「0 次拦截」当成「没有发生」

**现象**：看到 `+a` 是 0 次拦截，就写「取正号没有任何计算」。

**成因**：`+a` 与已同 dtype 的 `a.float()` 都**进了 dispatcher**，只是算子本身是 composite（`aten::positive` / `aten::to`）而被内联，所以 mode 看不到。

**规避**：把「0 次」拆成两类——**没进 dispatcher**（`a.shape`、`torch.Tensor.dim(x)`）与**进了但被内联**（`+a`、`a.float()`）；后者用 profiler 能看见 C++ 层的调用。判据是两条通道的**差集**。

## 5. 实测数据

**测量条件**：`code/01_dispatch_layers.py` 经 `make run` 在本机跑出，选中的解释器是
`/Users/cachio/miniconda3/bin/python` = **torch 2.11.0 / python 3.12.9 / CPU-only**（`torch.cuda.is_available()` = **False**）。全部断言通过（A 16 / B 3 / C 9，共 28 条）。

> **★ 限定语（本域强制）**：下面每一个数字都是**真实 PyTorch**跑出来的（`make run` 会用 `find_spec` 探测到一个装了 torch 的解释器，并打印它的版本）——**不是迷你实现**。但**必须绑定版本**：本机另有 2.6.0 / 2.7.1 / 2.8.0 / 2.13.0 四个解释器（连被选中的那个共五个），换一个版本，**时序类数字会变**；**结构性数字**（拦截次数、keyset 组成、算子名）在五个版本上实测一致。**本机没有 2.9 / 2.10 / 2.12 的解释器，这三个 minor 不声称覆盖。**

**（1）eager 路径的每次调用耗时**（最快一组，单位 μs，torch 2.11.0）：

> ⚠ **时序类数字每次运行会有几成波动**（同一台机器上重跑，`a + b` 在 0.89–0.90 μs 之间跳）。上表是**一次记录**；判据是「比值/量级」，不是这个具体数——程序里的三条 B 类断言都写成比值与量级形式，正是为此。

| 表达式 | 耗时 |
|---|---|
| `a + b` | 0.904 |
| `torch.add(a, b)` | 0.844 |
| `a.sum()` | 1.230 |
| `+a`（0 次拦截） | 0.189 |
| 进/出 mode（含建栈） | 6.421 |
| mode 内的 `a + b` | 4.036 |

**（2）代价模型的核对**（模型写在程序里，逐项被上面这组数否证）：
`|T(a+b) − T(torch.add)| = 0.060 μs`，而「一层 Python key」的标尺是 `4.036 − 0.904 = 3.132 μs`——**差价只有一个 dispatcher 层的 2%**。这从计时侧印证了 §1.3 的「同一个 dispatcher 入口的两个 C 绑定」。

**（3）mode 内比 eager 慢 4.5×**（4.036 / 0.904）。**mode 是诊断工具**：作者本人也只在观测时用它。

**（4）三个版本敏感的条目**（本机逐版本实测，写下的是**版本差异**本身）：

| 条目 | 2.6.0 | 2.7.1 | 2.11.0 | 2.13.0 |
|---|---|---|---|---|
| `c10/core/DispatchKey.h` 的 `Fake` 次序注释 | 无 | 无 | 无 | 无 | **有（仅此版本）** |
| `torch._C` 里名字含 `dispatch` 的函数个数 | 68 | 68 | 68 | **69** | 70 |
| `torch.library` 里的 `get_kernel` / `EffectType` | 无 | 无 | **无** | 有 | 有 |

> 第 2、3 行的意义是**反面**的：它们说明「2.13 新增 X」这类说法多半经不起实测。**凡是版本相关的能力，判据应当用特征探测（`hasattr`），不要拿版号当门牌。**

## 6. 小结与延伸

- 一次 `a + b` = **1 次** `__torch_dispatch__` 拦截；`+a` / `a.shape` / 已同 dtype 的 `a.float()` = **0 次**。一维 `a @ b` 走 `aten.dot.default`。**层数是可数的**，这是主线上「代价项」的第一项。
- `a + b` 与 `torch.add(a, b)` 同价：**同一个 dispatcher 入口的两个 C 绑定**，Python 层没有 `def add`。
- `DispatchKeySet` 是**一个 64 位整数**、**不是 1 key = 1 bit**、backend 上限 16；普通 CPU 张量的 keyset 里**没有** `Python`，`TorchDispatchMode` 走**线程局部栈**。
- 用 mode 观测时：**mode 先于子类 `__torch_dispatch__`**（依据只能是实测），且**CompositeImplicitAutograd 的算子对 mode 不可见**——这是对一个流行做法的否定。
- 往下一站：[02-autograd](02-autograd.md)（同一次 dispatch 里，autograd 这一层固化了什么）、[03-tensor-memory-model](03-tensor-memory-model.md)（张量本身的 stride/offset 假设）、[04-custom-operators](04-custom-operators.md)（怎么把自己的算子接进这条链）。

## 参考

- 本主题的实测程序：`code/01_dispatch_layers.py`（三类断言齐全，失败非零退出）
- 纯 python3 复核入口：`code/verify.py`（**不 import torch**，承担 A/C）
- 外部事实与版本锚点：`code/specs.py`（每条带 `# 来源：`）
- 编译器内部（图捕获、融合、codegen）：[60-compilers/inductor](../../60-compilers/inductor/01-graph-capture.md)
- 自定义算子的注册与编译入口：[04-custom-operators](04-custom-operators.md)

---
title: AMP 的执行约定：loss scaling 的算术、unscale 的时机与溢出的处置
slug: amp-execution
domain: 90-training
topic: mixed-precision
level: 3
difficulty: intermediate
prerequisites:
  - 90-training/mixed-precision/01-type-selection
tags: [amp, autocast, gradscaler, loss-scaling, unscale, gradient-clipping, apex]
related:
  - 90-training/mixed-precision/03-fp8-training
  - 90-training/autograd-and-backward/03-gradient-lifecycle
code:
  - code/02_amp.py
status: stable
updated: 2026-09-23
---

# AMP 的执行约定：loss scaling 的算术、unscale 的时机与溢出的处置

> 一次 AMP 步里有**一条顺序是硬的**：**unscale 必须在 `backward` 之后、`clip` 之前**。
> 顺序反了不会报错，只会**多裁 `S` 倍** —— 而那个比值恰好等于 `S`，本机实测 `1023.999992`。
> 另一半麻烦是**引用**：`O0/O1/O2/O3`、`GradScaler` 的默认值、loss scale 的初值
> **全是「某实现的文档」**，不是规范，而且**同一条东西有两个官方数**。

## 0. 背景与动机

### 一个具体的对照场景

上一行（`01-type-selection`）定下了「梯度用 fp16 存」。fp16 的范围窄，梯度会**掉到 0**。
补丁是 loss scaling：**把 loss 乘一个 `S`，反向就自动把梯度也乘 `S`，更新前再除回来**。

现在循环长这样，**每一步的先后是有讲究的**：

```
forward（fp16）           →  loss
loss ← loss × S           ← 缩放（在反向之前）
backward                  →  g_scaled（fp16 存）
── ★ 这里就是本文要判的那一步 ──
unscale： g ← g_scaled / S
clip   ： 按全局范数缩放到 max_norm
step   ： w ← w − lr·g      （在 fp32 主权重上）
```

**不用它 / 顺序写错的代价**：本机实测（`code/02_amp.py` 的 `A.2`），
同一个 `g`、同一个 `S = 1024`、同一个 `max_norm = 1.0`：

| 顺序 | 裁剪**看到**的范数 | 裁剪后 / unscale 后的范数 |
|---|---|---|
| `unscale → clip`（**对**） | `127.173540` | **`0.9999999921`** |
| `clip → unscale`（**错**） | `130225.704904` | **`0.0009765625`** |

后者是前者的 `1/1024` —— 也就是 **`1/S`**。**两种写法都不报错、都跑得下去**，
只是错的那一种把有效学习率悄悄压了 `S` 倍，而 `S` 是动态的 ⇒ 这个偏差**每次都不同**。

★ 题外但重要：`130225.704904 / 127.172158 = 1024.0` —— 那个 `1024` 就是 `S`。
**「误差恰好是个可算的数」是本文这一节的运气**：换任何别的 `S` 也一样成立。

⚠ **本文的范围**（`CONTRIBUTING` §7 与域 README 的边界表）：
「**为什么 fp16 要 loss scaling、bf16 为什么不用**」**整块**的正文在
`10-foundations/floating-point/04-low-precision-formats.md`，此处**回链**；
**loss scaling 自身的代价**（上溢侧被填满、次正规位被舍掉后不回来）在同上 `:197,229-231`，**回链**；
**`autocast` 的算子白名单 / 黑名单**属**框架 API 语义**（各框架互不相同），正文在 `70-frameworks/`，**回链**——
本文**一行都不重写**这些。它写的是**训练循环里的顺序与算术**。

## 1. 原理

### 1.1 为什么 `S` 必须取 2 的幂

设 `S = 2^k`。`g → fp16(S·g)` 只把指数加 `k`、**尾数一位不动**；再 `÷S` 只把指数减回来
⇒ 在结果仍落在**正规区**的前提下，`fp16(S·g)/S` 与 `fp16(g)` **逐位相等**。

本机实测（`A.1`，4096 个元素、`S = 1024`）：**4096 / 4096 个元素逐位相等**（容差 0）。
把 `S` 换成 `1000`（**不是** 2 的幂）后，**4060 / 4096 个元素不再相等**（`A.1b`）。

★★ **但「逐位相等」有一个前提，必须自己查**（本库的 **P22**）：
「正规区内」意味着 `|g| ≥ 最小正规数`。本机实测参与判据的 `|g|` 最小是 `6.738516e-04`，
最小正规数是 `6.103516e-05` ⇒ **11.04 倍**，够安全（`A.1c`）。
★ 这个前提不是形式主义：本程序的第一版把 `÷S` 写在 **fp16 里**
（NumPy 2.x 的 **NEP 50** 弱标量提升：`float16 数组 / Python float` **仍留在 fp16**），
于是 `1.025e-05 / 1024 = 1.001e-08` **低于最小次正规数 `5.96e-08`**，**静默变成 0**。
⇒ 显式转 fp64 再除，并在源码里断言这一点（`D.16`）。

### 1.2 unscale 的时机：为什么是「backward 之后、clip 之前」

一手依据只有一条，`[C]` Micikevicius 2017，§3.2（原始字节 1 次）：

> It is simplest to perform this unscaling **right after the backward pass but before gradient
> clipping** or any other gradient-related computations, ensuring that **no hyper-parameters
> (such as gradient clipping threshold, weight decay, etc.) have to be adjusted.**

把这句话翻成算术。设 `T = ‖g‖`（**未缩放**的梯度范数）、裁剪阈值 `max_norm = m`。

**正确顺序**：先除 `S` ⇒ 裁剪看到 `T` ⇒ 裁剪系数 `m/(T+ε)` ⇒ 裁剪后范数 `≈ m`。

**错误顺序**：裁剪先看到 `S·T` ⇒ 系数 `m/(S·T+ε)` ⇒ 再除 `S` ⇒ 最终范数

```
‖g_wrong‖ = S·T · m/(S·T+ε) · (1/S) = m·T/(S·T+ε)      ⇒   ‖g_right‖ / ‖g_wrong‖ ≈ S
```

**这个 `S` 就是那条「多裁」的倍率。** 而原文那句 `no hyper-parameters ... have to be adjusted`
说的正是：**阈值 `m` 的量纲是「未缩放梯度」的**。顺带一个推论：**weight decay 也一样** ——
它作用在更新量上，所以 unscale 也必须排在它前面。

本机实测（`A.2`）：比值 `1023.999992`（与 `S = 1024` 的相对差 `7.856e-09`）。
★ 那个 `7.86e-9` 不是噪声，它来自 `[B]` PyTorch 裁剪口径里分母上的 `+1e-6`；本文**如实报**它，
并把它变成第三条判据（`A.2e`：裁剪后范数必须**严格小于** `max_norm`）。

★ 另有一条**独立参照**（`A.2b`）：拿一步**不做任何缩放**的同 `g` 来对照 ——
正确顺序的相对差 `2.107e-04`、错误顺序 `9.990e-01`（差三个多数量级）。
⇒ 「顺序对不对」这件事**不需要知道 `S` 也能判**。

### 1.3 `S` 的可用窗口

`S` 不是越大越好，也不是越小越安全。它两侧都有边界：

```
下界：S 要大到让最小的梯度不落进归零区       ⇒  S_min = thr / min|g|        （thr = 最小次正规/2，见 01）
上界：S 不能让最大梯度溢出                   ⇒  S_max = 65504 / max|g|
```

★ **上界那句有一手依据**（`[C]` 同上，原始字节 1 次）：

> choosing a factor so that **its product with the maximum absolute gradient value is below 65,504**
> (the maximum value representable in FP16)

本机实测（`B.2b`，`max|g| = 8.498831`）：**`S_min = 4.42268e-05`、`S_max = 7707.41`**，
而本文用的 `S = 1024` **落在窗口里**（`4.42268e-05 ≤ 1024 ≤ 7707.41`）。
⇒ **`S` 有一个上界是真的会撞到的**（`C.2`）：`S = 2^10` 时**一个元素都不溢出**，
`S = 2^13` 时**溢出真的发生**。这就是「把 `S` 往大调以求更保险」为什么是错的。

**动态 scaling 的三个规则参数**（`[B]` PyTorch 的 `GradScaler`，原始字节各 4 次）：

```
溢出       ⇒ S ×= backoff_factor = 0.5
连续 2000 步无溢出 ⇒ S ×= growth_factor = 2.0       （growth_interval = 2000）
```

★ 由此得到**第二笔账**（`B.2`）：一次溢出把 `S` 减半，`growth = 2` ⇒ **一次增长就回到原值**，
但要**等满 2000 步**。⇒ **一次溢出的代价不止「跳过那一步」，还有之后最多 2000 步跑在偏小的 `S` 上。**

### 1.4 溢出的处置：原文说的是「一个选项」

`[C]` 同上，逐字（原始字节 1 次）：

> One option is to **skip the weight update** when an overflow is detected and simply move on to
> the next iteration.

⚠ **措辞是 `One option is`** —— 它是一种**选项**，不是规定。
原文给的**理由**也在同一段（原始字节 1 次）：

> overflows will result in infinities and NaNs in the weight gradients which will
> **irreversibly damage the weights after an update**.

**`irreversibly` 这个词是本机可以复现的**（`C.1`）：第 0 步的梯度里放一个 `inf`，
`w ← w − lr·inf = −inf`；此后**从第 1 步起喂进去的梯度全部有限**，
而 40 步之后 `w` 仍然是 `−inf`（`±inf − 有限 = ±inf`）。
⇒ 「**不处置**溢出的后果」是被证实的；但**「处置」的具体形式（跳步 / 饱和 / 别的）不在本文的断言里**。

## 2. 执行映射

### 2.1 三件事各发生在哪

| 事 | 谁做 | 正文位置 |
|---|---|---|
| **前向里哪些算子降精度** | 框架的 `autocast`（白名单 / 黑名单） | ★ **`70-frameworks/`**（**各框架互不相同**，属框架 API 语义），此处**回链** |
| **缩放 / 反缩放 / 溢出检测 / `S` 的调整** | `GradScaler` 那一层 | ★ **本文**（下 §2.2） |
| **裁剪的阈值与判据** | 按全局范数缩放 | ★ **`90-training/autograd-and-backward/03-gradient-lifecycle`**（阈值在彼处**只写一次**），此处**消费、不重定义** |

### 2.2 `GradScaler` 那一层实际做了什么（`[B]` PyTorch）

★ **本节是 `P19` 的现场**：下面每个数都来自**某个实现**的文档，必须写明是哪个实现与版本。

| 量 | 值 | 来源级别与实现 |
|---|---|---|
| `init_scale` | `65536.0`（`= 2^16`） | **[B]** PyTorch `torch.cuda.amp.GradScaler` |
| `growth_factor` / `backoff_factor` / `growth_interval` | `2.0` / `0.5` / `2000` | **[B]** 同上 |
| `unscale` 前不得做梯度相关计算 | —— | **[C]** Micikevicius 2017（§1.2 的逐字） |
| 溢出时 `optimizer.step` 被跳过 | —— | **[B]** 同上（规则）+ **[C]** 同上（`One option is`） |

★★ **另一个官方来源给的是另一个初值**（同一件事，两个实现）：

| 来源 | 逐字 | 值 |
|---|---|---|
| **[B]** NVIDIA 的 AMP FAQ | `beginning with a high loss scale value (say, 2^24)` | `16777216`（`= 2^24`） |
| **[B]** PyTorch `GradScaler` | `init_scale=65536.0` | `65536`（`= 2^16`） |

两者相差 **256 倍**。⇒ 正文**不得**写「初值取 X」——
**「有分歧」这件事本身就是结论**（`C.2` / `D.8` 把两个数都钉住，并断言比值精确为 `256`）。
★ 而 NVIDIA 那句的量词是 **`say`**：它是**举例**，不是规定。

### 2.3 具体组件与数值量级

| 组件 | 它在本文这一格里的角色 | 数值量级（本机实测 / `[B]` 文档） |
|---|---|---|
| **`GradScaler`**（`[B]` PyTorch） | 缩放、反缩放、溢出检测、`S` 的自适应 | `init_scale = 65536`、一次溢出后要 **2000 步**回到原值 |
| **`clip_grad_norm_`**（`[B]` PyTorch v2.9.0 源码） | 裁剪（`max_norm / (total_norm + 1e-6)`） | 那个 `+1e-6` 使裁剪后范数 `= 0.9999999921`（**不是** `1.0`） |
| **`Apex` 的 `O0–O3`**（`[B]` Apex 文档） | 「哪些张量用什么精度」的**四种预设** | `O1` 的 `master_weights = None`、`O2` 的是 `True`（见下） |
| **本机梯度范数** | 那一格里的被测量 | 未缩放 `‖g‖ = 127.172158`、放大后 `130225.704904` |

### 2.4 `O0–O3` 的四行规格（`[B]` Apex）

| 级别 | `cast_model_type` | `patch_torch_functions` | `keep_batchnorm_fp32` | `master_weights` | `loss_scale` |
|---|---|---|---|---|---|
| `O0` | `torch.float32` | `False` | `None`（**不适用**，一切 fp32） | `False` | `1.0` |
| `O1` | `None`（不适用） | `True` | `None`（**不适用**，权重全是 fp32） | **`None`** | `"dynamic"` |
| `O2` | `torch.float16` | `False` | `True` | **`True`** | `"dynamic"` |
| `O3` | `torch.float16` | `False` | `False` | `False` | `1.0` |

★★ **`O1` 的 `master_weights` 是 `None`，逐字解释是「不适用」**：

> `master_weights=None (not applicable, model weights remain FP32)`

⇒ **「O1 会额外维护一份 fp32 主权重」是错的** —— O1 下权重**本来就是** fp32，不需要「额外维护」。
**只有 `O2` 是 `True`**。本文因此用**两个**断言把它锁住（`D.7` 比对 `(None, True)`；
`D.7c` 比对那句逐字 —— `None` 在变异测试里给不出「另一个值」，所以必须配一条会被变异到的字符串）。

## 3. 实现与演进

| 阶段 | 怎么做 | 它换来什么、付什么账 |
|---|---|---|
| **手工 fp16 + 常数 `S`** | 自己乘 `S`、自己除回来 | 最少的机制；代价是 `S` 只能猜，且**猜大了会上溢** |
| **手工 + 常数 `S` + 溢出处置** | 检测 `inf`/`NaN`，跳过该步 | 不再被污染；代价是 `S` 仍不自适应 |
| **`Apex` 的 `O0–O3`**（`[B]`） | 四种预设，`O1`/`O2` 带动态 scaling | 第一次把「哪些张量用什么精度」变成**一行配置**；代价是**预设不透明** |
| **框架的 `autocast` + `GradScaler`**（`[B]` PyTorch） | 算子级降精度 + 动态缩放分离成两个对象 | 白名单/黑名单由框架维护；**顺序仍由调用者负责**（本文的对象） |

★ **四行里从头到尾没变的是「unscale 在 clip 之前」** —— 它是 [C] 原文那一句，
与实现怎么演进无关（`[B]` PyTorch 的文档也重复了同一件事：`Each parameter's gradient
(.grad attribute) should be unscaled before the optimizer updates the parameters`，
但它**没有**说「在 clip 之前」——那句只有论文有）。

## 4. 踩坑与失败模式

**① 把 unscale 放在 `clip` 之后。**

- **现象**：训练没报错，但有效学习率被悄悄压了；而且因为 `S` 是动态的，**这个偏差每步都不同**。
- **成因**：裁剪阈值 `max_norm` 的**量纲是未缩放梯度**。顺序反了，裁剪看到 `S·T`，
  系数变成 `m/(S·T+ε)`，最终范数 `≈ m/S` —— 本机实测比值 `1023.999992`（`= S`）。
- **规避**：**「unscale 在 backward 之后、clip 之前」是一条顺序约束**（[C] 逐字）。
  把它写成断言：正确顺序下载剪**看到**的范数必须 `≈` 真范数（`A.2c`），
  错误顺序下必须 `≈ S` 倍（`A.2d`）。
  ★★ **只查「两种顺序之比 == S」不够**：本文实测到过一个**方向盲**的假实现
  （把 unscale 挪到后面**并且做两次**：`÷S` 两次、前面一次也不做），**比值仍然是 `S`**、
  断言全过。⇒ 判据必须查「裁剪看到的是哪个范数」，而不是只查一个比值。

**② 拿「某实现的规格」当规范（本域的头号错误模式）。**

- **现象**：写「NVIDIA 的 AMP 规定 `O0/O1/O2/O3` 分别……」。
- **成因**：**NVIDIA 的 AMP 页里 `whitelist` / `blacklist` / `master_weights` 各出现 0 次**；
  完整描述在 **Apex 文档**（`whitelist` 1 次、`master_weights` **9** 次）。
  ⇒ 那是 **B 级（某实现的文档）**，不是 A 级（厂商规范）。
- **规避**：凡引 `O0–O3` / `GradScaler` / `clip_grad_norm_` 这类**「某实现的规格」**，
  **标 B 级 + 写明实现名（+ 版本）**；程序里把「来源是哪个实现」写进断言的理由串
  （`D.5`–`D.12` 每一条都写了）。

**③ 把 loss scale 的两个官方初值合成一个。**

- **现象**：写「loss scale 的初值取 `65536`」，读者以为这是规定。
- **成因**：**有两个官方数**（NVIDIA 的 AMP FAQ `say, 2^24` vs PyTorch 的 `init_scale=65536.0`），
  相差 **256 倍**。合成一个就丢掉了「有分歧」这个事实。
- **规避**：**两个都写、都钉住**，并断言它们的比值（`D.8`）。
  ★ 顺带：Apex 的 `initialize(...)` 里 `16777216.0` 是 **`max_loss_scale`（上界）** ——
  数值与 NVIDIA FAQ 的**初值**相同、**角色不同** ⇒ 同一个数不要因为相等就合并（`D.8b`）。

**④ 在 fp16 里做 `÷S`。**

- **现象**：`g / S` 之后一批小值**变成 0**，而且没有警告。
- **成因**：NumPy 2.x 的 **NEP 50** 弱标量提升 —— `float16 数组 / Python float` 的结果
  **仍是 fp16**。`1.025e-05 / 1024 = 1.001e-08` 落在最小次正规数 `5.96e-08` 之下 ⇒ 归零。
  ★ 这条与 `01-type-selection` 的「归零阈值是最小次正规的一半」是同一件事的两个现场。
- **规避**：**每一步的 dtype 都要显式写出来**（`np.asarray(h, dtype=np.float64) / scale`），
  并在源码里留下一句断言（`D.16`）。真实实现里 unscale 发生在 fp32 累加器上，本程序照办。

## 5. 判据表

| 可机械判定的条件 | 结论 |
|---|---|
| `S` 是 **2 的幂**、且结果仍落在**正规区** | `fp16(S·g)/S` 与 `fp16(g)` **逐位相等**（本机 4096/4096）；`S` 不是 2 的幂时**逐位相等不成立**（本机 4060/4096 不同） |
| 报「×S ÷S 是恒等」 | **必须先确认那个量没衰减到次正规区**（本机最小 `|g| / 最小正规数 = 11.04`）；否则量到的是次正规噪声 |
| `unscale` 排在 `backward` 之后、`clip` **之前** | 裁剪**看到**的范数 = 真范数（本机 `127.173540` vs `127.172158`） |
| `unscale` 排在 `clip` **之后** | 裁剪看到 `S` 倍大的范数（本机 `130225.704904`）⇒ 最终范数 `≈ m/S`，比值**恰好 `S`**（本机 `1023.999992`） |
| 只断言「两种顺序之比 == S」 | **不够**：有一个方向盲的假实现（unscale 做两次）能满足它。必须同时查「裁剪看到的是哪个范数」 |
| 裁剪后范数与 `max_norm` 的关系 | **严格小于**（本机 `0.9999999921`），因为分母上有 `+1e-6`（`[B]` PyTorch v2.9.0）；写成「精确等于」会放过一个把 `1e-6` 删掉的实现 |
| 要选 loss scale 的初值 | **有两个官方数**（`[B]` NVIDIA FAQ 的 `2^24` 与 `[B]` PyTorch 的 `2^16`，相差 **256 倍**）⇒ 不得合成一个 |
| 要选 `S` 的大小 | 有一个**可用窗口**（本机 `4.42268e-05 ≤ S ≤ 7707.41`）；上界由 `65504/max|g|` 给出（[C] 逐字） |
| 溢出被检测到时 | 跳过该步 ⇒ 权重**精确不变**、`S` **精确减半**（`×0.5` 对 2 的幂精确）；**不**跳过 ⇒ 权重被 `inf` 污染且**后续步也回不来** |
| 想报「一次溢出的代价」 | 不止跳过那一步：`S` 回到原值要**等满 `growth_interval` = 2000 步**（`[B]` 默认值） |
| 引 `O0/O1/O2/O3` | **必须标 B 级 + 实现名（Apex）**；`O1` 的 `master_weights` 是 `None`（**不适用**），只有 `O2` 是 `True` |

> ★ 阈值（`code/02_amp.py` 的 `D.3`/`D.4` 逐位钉住）：`TOL_UNSCALE_RATIO = 1e-6`、
> `TOL_TINY_GOOD = 1e-5` / `TOL_TINY_BAD = 1e-3`（判据⑦的两侧）、`MIN_REGION_FACTOR = 1.0`、
> **`TOL_EXACT = 0`**（正规区内的恒等是**逐位**的）、`SCALE = 1024.0`、`MAX_NORM = 1.0`。
> ★ 把 `TOL_EXACT` 从 0 放宽到 `1e-9` ⇒ 判据 A.1 会把「`S` 不是 2 的幂」那种实现也放过。

## 6. 小结与延伸

**一句话**：AMP 里真正容易写错的是**一条顺序**（unscale 在 `backward` 之后、`clip` 之前）
与**两类引用**（`O0–O3` 属 Apex 的 B 级文档；loss scale 的初值**有两个官方数**）——
而 `×S ÷S` 之所以能是恒等，**只因为 `S` 取 2 的幂**。

- **fp8 上为什么不能照搬这套** ⇒ 见 `03-fp8-training`
  （`[C]` 2209.05433 逐字：`Weight update skipping … is not a good choice for FP8`）。
- **裁剪的阈值与判据** ⇒ `90-training/autograd-and-backward/03-gradient-lifecycle`
  （正文在彼处，**只写一次**；本文只消费那个 `1e-6` 口径）。
- **`autocast` 的算子白名单 / 黑名单** ⇒ `70-frameworks/`（**各框架互不相同**，属框架 API 语义），
  正文在彼，此处回链；本文不写。
- **「为什么 fp16 要 loss scaling、bf16 为什么不用」** ⇒
  `10-foundations/floating-point/04-low-precision-formats.md`，回链。
- **上游**：`01-type-selection`（本文是它的下一个编号，`prerequisites` 里就是它）。

## 编译与运行

```bash
cd code
make run                        # 三个编号程序依次跑；本程序是 02_amp.py
python3 02_amp.py               # 只跑本程序
```

- 配套程序：**`code/02_amp.py`**（编号与外键严格对应）。
- 依赖：`numpy`（`requirements.txt` 里**只有**它）。**不需要网络、不需要 GPU。**
- ★★ **本程序一行都不 `import` PyTorch / Apex**（`make lint` 用 AST 查这件事，
  `BAN` 集合里连 `apex` 也列了）。做法是把那些实现**文档里逐字写的规则**当作 `specs.py` 的
  **`[B]` 常量**，再在 numpy 上**重演一遍** ⇒ 因此本文的数**不是「PyTorch 在本机跑出来的」**，
  而是「按 PyTorch / Apex 的**文档**在本机推出来的」⇒ **一律标为推导值**。
- 失败标记：`:: FAIL: mixed-precision`（本主题自定义，**不跨主题照抄**）。
- 程序的输出分四段：`A` unscale 的算术与时机、`B` 代价模型（scaling 的收益/边界/恢复时间）、
  `C` 边界与反例（不处置溢出 / `S` 越界）、`D` 标记自检、口径钉住与 specs 钉住。

## 参考

**本机实测值 / 推导值**

- `code/02_amp.py` 的 `make run` 输出：`python3 3.13.5` + `numpy 2.4.4`，macOS arm64，**无 GPU**。
  口径：`seed = 20260926`、4096 个梯度元素、`g ~ N(0, 2)`、`S = 1024`、`max_norm = 1.0`、
  `lr = 0.01`、`σ` 扫描同 `01`。**全部是推导值**（按 `[B]` 文档的规则算出，不是跑真 PyTorch）。
- ★ 与计划 §0.7-D 的对应：**D6**（loss scaling ×1024 的效果：`7.614e-06 / 1.172e-03 / 1.047e-02`）。
  ⚠ **D2 的 `0.696466` 是单种子值**（换种子得 `1.127157`）——本文**没有引用它**（那是 `optimizers` 的判据）。

**外部来源**

- **[C]** P. Micikevicius et al., *Mixed Precision Training*，arXiv **1710.03740**。
  **已确认同行评审**（abs 页 `Comments:` 逐字 `Published as a conference paper at ICLR 2018`；
  ⚠ 那句在 ar5iv **正文**里计 0）。逐字引用（原始字节各 1 次）：
  `It is simplest to perform this unscaling right after the backward pass but before gradient clipping`、
  `below 65,504`、`One option is to skip the weight update when an overflow is detected`、
  `irreversibly damage the weights after an update`、`ranging from 8 to 32K`。
  抓取（ar5iv）：**80840 B**，`sha256[:16] = f7493335335e7f4f`。
- **[B]** **PyTorch** 的 `torch.cuda.amp.GradScaler` 文档（`docs/2.14/amp.html`）——
  `init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000`。
  抓取 2026-09-23，**754077 B**，`sha256[:16] = df5bb3e3367f1ee1`。
  ⚠ **取法有坑**：`/docs/stable/amp.html` 返回的是 **1270 B 的 `<meta http-equiv="refresh">` 跳转壳**；
  **`curl -L` 跟的是 HTTP 重定向、跟不了 meta refresh** ⇒ 正确做法是**读出 meta 里的目标 URL 再取一次**。
- **[B]** **PyTorch** 的 `torch/nn/utils/clip_grad.py`（**v2.9.0 源码**）：
  `clip_coef = max_norm / (total_norm + 1e-6)`。抓取 11427 B，`sha256[:16] = 63efb273c62fd917`。
  ⚠ 那个 `1e-6` 只在 **v2.9.0** 上核过，**未逐版核对** ⇒ 引用时写出 v2.9.0。
- **[B]** **Apex** 的 AMP 文档（`nvidia.github.io/apex/amp.html`）：`O0–O3` 的四个属性串、
  `master_weights=None (not applicable, model weights remain FP32)`、`max_loss_scale=16777216.0`。
  抓取 64968 B，`sha256[:16] = 7484b043bf91e1b5`（`whitelist` 1 次、`master_weights` **9** 次）。
  ★ **对照**：NVIDIA 的 AMP 页（`docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html`，
  481333 B，`sha256[:16] = 690a696dbc995434`）里 `whitelist` = **0**、`master_weights` = **0** ——
  **同一个事实在另一个载体上数不到**，所以「`O0–O3` 出自 Apex」这句话必须**两个 URL 都留痕**。
- **[B]** **NVIDIA** 的 AMP FAQ（同上 URL）：`beginning with a high loss scale value (say, 2^24)`（原始字节 2 次）。
- **本库**：`10-foundations/floating-point/04-low-precision-formats.md`（范围 / 精度 /
  loss scaling 的代价）、`90-training/autograd-and-backward/03-gradient-lifecycle`（裁剪的阈值与判据）。

**上游契约**

- `01-type-selection` 是本文的 `prerequisites`：本文的每一步都作用在它选定的那四行上。
- **边界声明**：本文涉及「为什么 fp16 要 loss scaling」的地方一律回链
  `10-foundations/floating-point/04-low-precision-formats.md`；涉及 `autocast` 算子语义的地方
  一律回链 `70-frameworks/`；涉及裁剪阈值的地方一律回链
  `90-training/autograd-and-backward/03-gradient-lifecycle`。

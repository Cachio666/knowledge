---
title: 学习率调度与梯度裁剪：训练管线里的两格，以及它们的次序
slug: schedules-and-clipping
domain: 90-training
topic: optimizers
level: 3
difficulty: intermediate
prerequisites:
  - 90-training/optimizers/02-adam-and-adamw
  - 90-training/autograd-and-backward/03-gradient-lifecycle
tags: [lr-schedule, warmup, cosine-annealing, gradient-clipping, max-norm, training-loop]
related:
  - 90-training/optimizers/01-sgd-and-momentum
  - 90-training/mixed-precision/02-amp-execution
  - 90-training/autograd-and-backward/03-gradient-lifecycle
code:
  - code/03_schedule_clip.py
status: stable
updated: 2026-09-23
---

# 学习率调度与梯度裁剪：训练管线里的两格，以及它们的次序

> 训练循环里有两个**独立**的旋钮：`lr` 与 `max_norm`。它们**各自只作用在自己那一格** ——
> 但这句话只有在次序正确时才成立：把裁剪放在 `lr` **之后**（裁「更新量」而不是梯度），
> 阈值就被放大 `1/lr` 倍，而 **LR warmup 期 `lr` 最小 ⇒ 那一段等于没有裁剪**。
> ★ 顺带：**LR warmup 不改变总步长预算，只改变分配**（`Σ lr(t) = lr_max·T/2`，与 `W` 无关）。

## 0. 背景与动机

### ★★ 先说标题里那个词：`warmup` 是同名不同指

`00-meta/glossary.md:146` **已经有** `warmup` 这个词条，它指的是
「**预热 / 丢弃前导测量**」—— 即 **benchmark 预热**
（`45-kernel-lifecycle/convergence-and-tuning/01-benchmark-hygiene.md` 里 `warmup` 出现 **23** 次，
`50-performance` 等**全是同一语义**）。

本文写的是 **LR warmup（学习率 warmup）** —— **完全不同的两件事**。

⇒ ★★ **本文一律写全 `LR warmup`**，并**每处都带上这句限定语**。
（`00-meta/glossary.md` 那条改成 `warmup（benchmark 预热）`、以及新增
`学习率 warmup（LR warm-up）` 并互相指认，是**收尾 chunk** 的动作；
**在它做完之前，凡是提到 `warmup` 的地方都要自己带上限定语**。）

### ★★ 边界声明跟着知识点走：机制解释在别处

**LR warmup 的机制解释**（post-LN 的平均场理由、以及原文用 `hypothesize` 的措辞纪律）
**正文在** `80-model-architectures/normalization/02-placement-and-qk-norm.md:83-100`
（[打开该行](../../80-model-architectures/normalization/02-placement-and-qk-norm.md)）。

本文**一行都不重写它**。本文写的是**工程判据与设定**：
**形状**（峰值在哪一步、末值是多少）、**总账**（这些步长加在一起是多少）、
以及它与**裁剪的那一格**怎么接。

★ **这条声明必须写在这里、而不是只写在某一篇的抬头** —— 知识点走到哪，声明就跟到哪。

### 一个具体的对照场景：四步，顺序会错

一次训练步的这四格：

```
①  loss ← loss × S             （loss scaling，见 mixed-precision/02）
②  backward → g_raw            （fp16 存）
③  unscale  ： g ← g_raw / S   （★ 时机：[C] Micikevicius 2017 逐字
                                  "right after the backward pass but before gradient clipping"）
④  clip     ： 按**梯度**的全局范数缩放到 max_norm
⑤  step     ： w ← w − lr(t)·g  （★ lr(t) 由调度给）
```

★★ **`④` 与 `⑤` 的次序是本篇的核心**：裁剪作用在**梯度**上 ⇒ `max_norm` 与 `lr`
**是两个正交的旋钮**；把 `④` 挪到 `⑤` 之后（裁「更新量」）会让阈值被放大 `1/lr` 倍。

⚠ **本文的范围**（`CONTRIBUTING` §7 与域 README 的边界表）：

- **`unscale` 的时机**那条**判据**（`clip` 看到的是 unscaled 的范数）**正文在
  `90-training/mixed-precision/02-amp-execution.md` 的 `A.2`/`A.2c`/`A.2d`**
  —— **本文回链、不重定义**；本文只**钉住管线里三格的次序**并把「`S` 与 `lr` 各管一格」
  写成**可分离性**的断言（§2.2）。
- **优化器状态的分片与通信量** → `100-parallelism/README.md:14`（`zero-fsdp`）。
- **`LR warmup` 的机制解释** → `80-model-architectures/normalization/02`（见上）。
- **框架的 autograd 语义** → `70-frameworks/pytorch/02-autograd.md`。

## 1. 原理

### 1.1 调度：线性 `LR warmup` + 末端锚定的 cosine

```
t < W        :  lr(t) = lr_max · (t+1)/W
t >= W       :  lr(t) = lr_max · 0.5·(1 + cos(π·(t−W+1)/(T−W)))
```

`T` 是总步数、`W` 是 `LR warmup` 的步数。这条曲线有**三条手算得到的边界**（`A.1`）：

| 点 | 值 | 为什么 |
|---|---|---|
| `lr(0)` | `lr_max/W` | 线性段的第 1 步 |
| `lr(W−1)` | **`lr_max`**（**精确**） | 线性段的最后一步 |
| `lr(T−1)` | **`0`**（**精确**） | cosine 的 `cos(π) = −1` |

★ **末值是「精确 0」而不是「≈ 0」** —— 那是把 cosine **锚在末端**的结果
（分母用 `T−W` 而不是 `T−W−1`，第 `T−1` 步恰好落在 `π` 上）。

★ 本文的程序用**两条不同的算式**取 cosine 段（`1 + cos(2θ)` 与 `cos²θ`，半角恒等式）
互相对拍：本机整条曲线上的最大**绝对**偏差 `2.220e-16`（判据的口径是与 `lr_max` 比）。

### 1.2 ★★ 总账：`LR warmup` **不改变**总步长预算

这是本篇最不显然、也最容易变成一句空话的一条。把整条曲线的和算出来：

```
Σ_{t=0}^{T−1} lr(t)
  = [warmup 段]  lr_max·(W+1)/2
  + [cosine 段]  lr_max·(m−1)/2        （m = T−W，用 Σ_{k=1}^{m} cos(πk/m) = −1）
  = lr_max·T/2
```

★★ **与 `W` 无关**，**与 `lr_max` 成正比**。本机实测（`B.1`，四组 `(W, T)`）：
相对差 **`0.000e+00`**（逐位）。另一档 `lr_max = 0.1`、`T = 50` 给 `Σ = 2.5`
（= `0.1·50/2`，逐位）。

⇒ ★ **`LR warmup` 把前段的步长「让出去」，cosine 把总量「补回来」** ——
这正是本域的主线（**每个「省」都在别处记账**）在这一格的形态：
warmup 的代价不是「少走了一些路」，而是「**把路重新分配了**」。

**反面对照（`B.1b`）**：把 `LR warmup` 写成**常数** `lr_max`（而不是线性上升）
⇒ `Σ = 54.5 ≠ 50`（相对偏差 `9.0%`）⇒ **这条恒等式依赖「线性 warmup + 末端锚定的 cosine」
这一族调度**，不是恒真的。

### 1.3 裁剪：`max_norm·T/(T + ε)`

**[B] PyTorch v2.9.0 的源码**（`torch/nn/utils/clip_grad.py`）逐字：

```
clip_coef = max_norm / (total_norm + 1e-6)
```

⇒ 裁剪后的全局范数 = `max_norm · T/(T + ε)`，与 `max_norm` 的**相对误差 = `ε/(T+ε)`**。

★★ **判据与阈值定义在 `90-training/autograd-and-backward/03-gradient-lifecycle.md` 的「表 A ⑦」**
（**在那里只写一次**）—— **本文只实现、不重定义**。回链时**必须一并带上 `A.2b`**：

| 侧 | 判据 | 本机实测 |
|---|---|---|
| **上侧** | 相对误差 ≤ `1.1e-6/max(T,1)`（**`T ≥ 1`**，`T ∈ {2,4,8}`） | 最大只有阈值的 **`0.9091`** 倍 |
| **下侧（`A.2b`）** | 实测相对误差 **==** 闭式 `ε/(T+ε)`（容差 `1e-6`） | 最大相对偏差 **`1.870e-09`** |

★★ **下侧为什么必须有**（`T26`，Chunk 1 实施者实测）：只写单侧时，
把实现里的 `+1e-6` **删掉**会让 `25` 条断言**全绿** —— 因为那样结果反而**更精确**
（相对误差变成 `0`，而 `0` 满足任何上界）。⇒ **单侧判据会放过一个错实现。**

★ **`T ≥ 1` 与 `T > max_norm` 都是判据的一部分**：`T < max_norm` 时**不发生裁剪**
（`A.2c`：`T = 0.5`、`max_norm = 1.0` ⇒ 裁剪后范数仍是 `0.5`，**逐位**）。

### 1.4 ★★ `clip` 与 `lr` 是两个正交的旋钮

因为裁剪作用在**梯度**上（`lr` 还没参与），三档 `lr ∈ {0.001, 0.01, 0.1}` 上：

| | 实测 |
|---|---|
| `clip` **看到的量**的范数 | **逐位相同**（极差 `0.0`，容差 `0`）（`A.3a`） |
| `Δw` 的范数 | 跨三档共 **100 倍** —— 与 `lr` **严格成正比**（`A.3b`） |

★★ **而这条不是恒真的**：换成「把 `clip` 挪到 `lr` 之后」的实现在同一测法下，
**被裁的那个量**随 `lr` 变（三档跨 `100` 倍）⇒ 这条断言**区分了两个实现**。

**错序的后果有闭式（`A.3c`）**：在 `lr·T < max_norm` 这一档（★ **`LR warmup` 期正是如此**），
错序**根本不裁剪**（`coef = max_norm/(lr·T+ε) > 1`）⇒ 它真正用的是 `lr·T` 而不是 `lr·max_norm`：

```
‖Δw‖_错序 / ‖Δw‖_正确 = (T + ε)/max_norm        ★ 与 lr 无关
```

本机实测（`T = 10`、`max_norm = 1`、`lr ∈ {0.01, 0.05}`）：比值都是 **`10.000001`**，
与闭式的相对差 **`0.000e+00`**（逐位），且**两档 `lr` 给出同一个数**。

### 1.5 ★★ 错序 × `LR warmup`：耦合在一起的两个「旋钮」

把 §1.4 的结论放到 `LR warmup` 上看，就得到一个**只在这两格交叉处**才出现的数：

| | `LR warmup` 首步（`lr = lr_max/W`） | 峰值步（`lr = lr_max`） | 比值 |
|---|---|---|---|
| **正确顺序**的等效阈值（梯度口径） | `max_norm` | `max_norm` | **`1`** |
| **错序**的「等效阈值」（`Δw` 口径） | `max_norm/lr_first` | `max_norm/lr_peak` | **`W`** |

本机实测（`A.3d`，`W = 10`）：错序下首步的等效阈值是峰值步的 **`10.000000` 倍**，
而正确顺序下恒为 `1.000000000000`（容差 `0`）。

⇒ ★★ **「错序」这件事在 `LR warmup` 期最有害**（那时 `lr` 最小、阈值被放大得最多）。
一句话记住：**裁剪的阈值必须定义在梯度上；否则它的实际大小会被 `lr` 调制，
而 `lr` 在 `LR warmup` 期是**最小的**。

### 1.6 管线的三格与它们的可分离性

```
unscale(÷S)  →  clip(max_norm)  →  ×(−lr)
     ↑                ↑                ↑
   只有 S 能动      只有 max_norm 能动   只有 lr 能动
```

★ **`S = 1` 时「`unscale` 与 `clip` 换序」在数值上是 no-op**（`A.3e` 实测：两者的 `Δw`
**逐位相同**，差 `0.0`）—— ⇒ **任何数值断言都抓不到它**（本文的结构翻转抽查里，
这个错序确实 `rc=0` 溜了过去）。而 `S = 1024` 时它把 `‖Δw‖` 压小 **`1024` 倍**。
⇒ ★★ 本文因此**把次序钉在源码上**（`A.3f`：三个动作在源码里的索引严格递增），
与 `mixed-precision/02` 把口径钉在源码上的做法同源。

★ **口径**：`unscale` 的**时机**那条判据的正文在 `mixed-precision/02` 的 `A.2c`/`A.2d`
（**回链、不重定义**）；本文在这里只给「数值抓不到」这个**见证**，
用来说明为什么必须有那条**结构钉住**。

## 2. 执行映射

### 2.1 管线里三格各管一件事

| 格 | 它管什么 | 它**不**管什么 |
|---|---|---|
| `unscale(÷S)` | 把 loss scaling 还原（`S` 只在这一格） | 梯度的大小、方向 |
| `clip(max_norm)` | 全局范数的**上限** | `lr`、`S`、方向（见 §2.3） |
| `×(−lr(t))` | 步长（由调度给） | 阈值、方向 |

### 2.2 调度与裁剪的接口：两个旋钮、一个耦合点

| | 正确 | 错序（`clip` 在 `lr` 之后） |
|---|---|---|
| `max_norm` 的实际含义 | 「**梯度**的全局范数 ≤ `max_norm`」 | 「**更新量** ≤ `max_norm`」= 「梯度 ≤ `max_norm/lr`」 |
| 与 `lr` 的关系 | **无关**（`A.3a`：三档逐位相同） | **被 `lr` 调制**（放大 `1/lr` 倍） |
| 在 `LR warmup` 期 | 阈值不变 | 阈值被放大 **`W` 倍**（`A.3d`） |

★ **可分离性（`A.3e`）**：正确管线下，`S ∈ {1, 1024}` 给出**逐位相同**的 `Δw`
⇒ 三格各作用在自己那一格。

### 2.3 裁剪动的是范数，不是方向

`B.2` 实测：裁剪后与裁剪前的**方向余弦 = `1.00000000000000022`**（逐位级），
而**范数比恰好等于那个系数** `max_norm/(T+ε)`。
⇒ 裁剪是**一个标量乘子**，它的全部代价就是那个**相对偏差 `ε/(T+ε)`**。

★ 口径：这里量的是 `‖g_c‖/‖g‖`（**缩了多少**），与 `A.2b` 的 `|‖g_c‖/max_norm − 1|`
（**离阈值多远**）**是两个不同的量** —— 两者都由 `ε` 决定，但不要混用。

### 2.4 本机实测的一组典型值

| 量 | 值（`T = 100`、`W = 10`、`lr_max = 1`、`max_norm = 1`） |
|---|---|
| `lr(0)` / `lr(9)` / `lr(99)` | `0.1` / `1.0`（峰值）/ `0` |
| `Σ lr(t)` | **`50.0`** = `lr_max·T/2`（与 `W` 无关） |
| 裁剪后范数（`T_grad = 2, max_norm = 1`） | `0.9999995`（相对误差 `5.000e-07`） |
| 同上的闭式相对偏差 | `9.508e-11` |
| 错序 / 正确的 `‖Δw‖` 比值（`lr·T < max_norm`） | `10.000001` = `(T+ε)/max_norm` |

## 3. 实现与演进

| 阶段 | 写法 | 它换来了什么、付了什么账 |
|---|---|---|
| **常数 `lr`** | `lr(t) = lr_max` | 无超参；代价是前段与末段**用同一个步长** |
| **步进衰减（step decay）** | 在固定步数上乘 `0.1` | 简单；代价是**断点**（需要预先知道里程碑） |
| **`LR warmup` + cosine** | 本文 §1.1 的形状 | 前段渐进、末段平滑；★ 代价被 `B.1` 量清：**总步长不变，只是重新分配** |
| **梯度裁剪** | 按**全局**范数缩放到 `max_norm` | 抗偶发的范数突增；代价是 `ε/(T+ε)` 的偏差，★ 以及**它不能替代溢出处置**（§4 ④） |
| **调度 × 解耦衰减** | ★ AdamW 论文的 Algorithm 2 **第 11 行**逐字 `η_t ← SetScheduleMultiplier(t)`（注释 `can be fixed, decay, or also be used for warm restarts`） | **论文自己把调度写进了优化器的算法框** ⇒ 这一格与 `02` 是同一件事的两面 |

★ **`LR warmup` 的形状与「因为什么需要它」（post-LN 的平均场理由）是两件事** ——
后者在 `80-model-architectures/normalization/02:83-100`，本文只写前者。

## 4. 踩坑与失败模式

**① ★★ 相对判据在曲线的末端「假失败」。**

- **现象**：用**相对**差判「两条算式一致」，得到 `|a/b − 1| = 1`（即 100% 偏差）。
- **成因**：曲线**精确落在 0** 的那一步（`t = T−1`），两条算式一个给 `0`、
  另一个给 `cos²(π/2) ≈ 3.749e-33`（**不是 0**）⇒ `0/b − 1 = −1`。
  ★ 这是 **P22** 的现场：**参照量已经衰减到舍入残渣**，那时**相对差没有意义**。
- **规避**：**判据用「与 `lr_max` 比」的绝对口径**，并把**参照量的最小值打印出来**
  （`A.1b` + `A.1c` 两条：一条判据、一条量的检查）。

**② ★★ 裁剪只写单侧判据。**

- **现象**：断言「裁剪后范数 ≈ `max_norm`（相对误差 ≤ `1.1e-6/T`）」—— 全绿。
- **成因**：`ε/(T+ε) ≤ 1.1e-6/T` 是个**上界**，而「误差 = 0」也满足上界；
  把 `+1e-6` **删掉**反而让结果**更精确** ⇒ 25 条断言全绿（`T26`）。
- **规避**：**两侧**（`A.2` 上侧 + `A.2b` 下侧 `== ε/(T+ε)`）。

**③ 把裁剪放在 `lr` 之后（裁「更新量」）。**

- **现象**：收敛变慢、或者 `LR warmup` 期像没有裁剪；**不报错**。
- **成因**：阈值被放大 `1/lr` 倍（`A.3c` 有闭式）、而在 `LR warmup` 期放大 **`W` 倍**（`A.3d`）。
- **规避**：断言**两件事**：`clip` 的输出与 `lr` **逐位无关**（`A.3a`），
  且 `Δw` 与 `lr` **严格成正比**（`A.3b`）。

**④ ★★ 以为「开了裁剪就不会被 `inf` 污染」。**

- **现象**：梯度里有一个 `inf`，`clip` 之后**得到 `nan`**。
- **成因**：`T = inf` ⇒ `coef = max_norm/(inf+ε) = 0` ⇒ 有限元素 `x·0 = 0`，
  而 `inf·0 = nan`（IEEE 754）⇒ **一个 `inf` 把整个梯度向量变成 `nan`**
  （`C.1` 实测：`1` 个 `nan` + `7` 个精确 `0`），随后 `w ← w − lr·nan = nan`，**不可恢复**。
- **规避**：**溢出必须在裁剪之前处置**（`mixed-precision/02` 的 `A.3`：检测到溢出就**跳步**）
  —— 裁剪**不能替代**它。★ 本文的 `C.1` 是那条接口的**负面证明**。

**⑤ `max_norm = 0` 是一个静默的「停训」。**

- **现象**：`clip` 把梯度全清零（`C.2` 实测：裁剪后范数精确 `0`），更新恒为 `0`。
- **成因**：`coef = 0/(T+ε) = 0` ⇒ 全乘 `0`。**不报错、loss 不变、看起来像收敛了**。
- **规避**：把 `max_norm > 0` 作为配置校验的一部分；★ 顺带：`T = 0`（梯度恰好全 0）
  时保护项的作用是**防止 `0/0`**（`coef` 变成一个大数而不是 `inf`），**不会**误伤（`C.3`）。

**⑥ 把 `warmup` 不加限定语地用。**

- **现象**：读者按 `glossary.md:146` 的「benchmark 预热」去理解本文。
- **成因**：**同一个词指两件事**，而 glossary 已经收了其中一件。见 §0。
- **规避**：**一律写全 `LR warmup`**，并在每处带上「机制解释在 `80-…/normalization/02:83-100`」的声明。

## 5. 判据表

| 可机械判定的条件 | 结论 |
|---|---|
| `LR warmup` 的峰值在哪一步 | **`t = W−1`**，值**精确** = `lr_max`；末值在 `t = T−1` **精确** = `0`（cosine 锚在末端） |
| 要判断一条 `LR warmup + cosine` 写得对不对 | 与**另一条算式**（半角恒等式 `0.5(1+cosθ) = cos²(θ/2)`）逐点对拍；★ **判据要用绝对口径**（与 `lr_max` 比）—— 曲线**精确落在 0** 的那一步，相对差会假失败 |
| 想知道 `LR warmup` 的代价 | **总步长预算不变**：`Σ lr(t) = lr_max·T/2`（**与 `W` 无关**，本机逐位）；warmup 用**常数** `lr_max` 时这条破掉（`Σ` 从 50 变 54.5） |
| 裁剪后范数与 `max_norm` 的关系 | `= max_norm·T/(T+ε)`，相对误差 **`≈ ε/T`**；★ 判据**必须两侧**：上侧 `≤ 1.1e-6/max(T,1)`（`T ≥ 1`）**且**下侧 `== ε/(T+ε)` |
| `T < max_norm` | **不发生裁剪**（裁剪后范数仍 `== T`）⇒ 「裁剪后 == `max_norm`」只在 `T > max_norm` 时成立 |
| 想问「`clip` 会不会受 `lr` 影响」 | **正确顺序下不会**（三档 `lr` 上 `clip` 的输出逐位相同）；★ 错序下「被裁的量」随 `lr` 跨 100 倍 |
| 错序（`clip` 在 `lr` 之后）的后果 | 阈值被放大 `1/lr` 倍；在 `lr·T < max_norm` 时**完全不裁剪** ⇒ `‖Δw‖` 之比 **`= (T+ε)/max_norm`**（**与 `lr` 无关**） |
| 错序在什么时候最有害 | **`LR warmup` 期**：首步的等效阈值是峰值步的 **`W` 倍**（正确顺序下恒为 `1`） |
| 想判「`unscale` 与 `clip` 的次序」 | 判据的正文在 `mixed-precision/02` 的 `A.2c`/`A.2d`（**回链**）；★ 本文只用它作为「数值抓不到」的**见证**：`S = 1` 时换序是 **no-op**（逐位相同）⇒ 必须**钉在源码上** |
| 裁剪会不会改变梯度的方向 | **不会**：方向余弦 `== 1`（逐位级）；它是**一个标量乘子**，范数比 `== max_norm/(T+ε)` |
| 梯度里有 `inf` 时裁剪会怎样 | 给出 **`nan`**（`coef = 0`，`inf·0 = nan`）⇒ **裁剪不能替代溢出处置**（跳步在 `mixed-precision/02`） |
| `max_norm = 0` | 梯度全清零、更新恒为 `0` —— **一个静默的停训** |
| 梯度恰好全为 0（`T = 0`） | 保护项 `+ε` 让 `coef = max_norm/ε > 1` ⇒ **不缩放**（防止 `0/0`），**不会**误伤 |

> ★ 阈值与口径（`code/03_schedule_clip.py` 的 `D.3`/`D.4` 逐位钉住）：
> `TOL_SCHED = 1e-15`（与半角恒等式的**绝对**偏差，口径是与 `lr_max` 比）、
> `TOL_EXACT = 0.0`（`A.3a` 的逐位相同）、`TOL_PROP = 1e-15`、`TOL_RATIO = 1e-12`、
> `TOL_W_CROSS = 1e-12`、`TOL_TOTAL = 1e-12`、`TOL_MEASURABLE = 1e-2`（**可测性下界**）、
> `CLIP_TOL_REL = 1.1e-6`（上侧）、`CLIP_TOL_CLOSED = 1e-6`（**下侧 `A.2b`**）。
> ★ **口径常数**：`T = 100`、`W = 10`、`lr_max = 1.0`、`max_norm = 1.0`、`ε = 1e-6`（[B] PyTorch v2.9.0）。
> ★ 放宽 `TOL_MEASURABLE` ⇒ `A.3c`/`A.3e`/`B.1b` 的反面一起退化成恒真。

## 6. 小结与延伸

**一句话**：`lr` 与 `max_norm` 是两个**正交**的旋钮 —— 但只有在次序正确
（`unscale → clip（梯度）→ ×lr`）时才是；而 `LR warmup` **不改变总步长预算，只改变分配**。

- **SGD / 动量 / Nesterov** ⇒ 见 `01-sgd-and-momentum`。
- **Adam / AdamW 与自归一化** ⇒ 见 `02-adam-and-adamw`。
  ★ 接口：AdamW 论文的 Algorithm 2 **第 11 行**就是那个调度乘子。
- **裁剪的判据与阈值（含 `A.2b` 的那一侧）** ⇒
  `90-training/autograd-and-backward/03-gradient-lifecycle.md` 的表 A ⑦（**正文在彼，此处只实现**）。
- **`unscale` 的时机、以及溢出 ⇒ 跳步** ⇒ `90-training/mixed-precision/02-amp-execution.md`
  （**正文在彼**；本文的 `C.1` 是它与裁剪的接口的负面证明）。
- **`LR warmup` 的机制解释**（post-LN 的平均场理由） ⇒
  `80-model-architectures/normalization/02-placement-and-qk-norm.md:83-100`（**正文在彼**）。
- **`warmup` 这个词的另一个语义（benchmark 预热）** ⇒ `00-meta/glossary.md:146`。
- **优化器状态的分片与通信量** ⇒ `100-parallelism/README.md:14`（`zero-fsdp`）。
- ★★ **一处必须写明的让步**：本文的判据**全部**是「本机玩具规模的真跑」
  （`python3 3.13.5` + `numpy 2.4.4`、**无 GPU**、不训练任何真模型）⇒
  **不得支撑关于大规模训练的结论**。

## 编译与运行

```bash
cd code
make run                        # 三个编号程序依次跑；本程序是 03_schedule_clip.py
python3 03_schedule_clip.py     # 只跑本程序
```

- 配套程序：**`code/03_schedule_clip.py`**（编号与外键严格对应）。
- 依赖：`numpy`（`requirements.txt` 里**只有**它）。**不需要网络、不需要 GPU。**
- 失败标记：`++ FAIL: optimizers`（本主题自定义，**不跨主题照抄**）；程序内有两条「标记自检」
  断言（`D.1` / `D.2`）守着它，`mutate.py` 用 **AST** 读出该值（不得写死）。
- 程序的输出分四段：`A` 调度的形状 / 裁剪的两侧判据 / `clip` 与 `lr` 的正交性（含 `LR warmup` 的交叉）、
  `B` 代价模型（总步长账 / 裁剪改的是范数还是方向）、
  `C` 边界与反例（`inf` ⇒ `nan` / `max_norm = 0` / `T = 0`）、
  `D` 标记自检、口径与阈值钉住、specs 常量钉住、独立性证据。
- ★★ **口径/阈值/标记自检一律在 `preflight()` 里、跑在任何计算之前**（本库的 **P23**）。
- ★★ **本程序的开头有一段 `LR warmup` 的同名不同指声明**（`warmup` vs benchmark 预热），
  以及一条「机制解释在 `80-…/normalization/02:83-100`」的**边界声明** —— 声明跟着知识点走。

## 参考

**本机实测值 / 推导值**

- `code/03_schedule_clip.py` 的 `make run` 输出：`python3 3.13.5` + `numpy 2.4.4`，macOS arm64，
  **无 GPU**。口径：`seed = 20260930`、`T = 100`、`W = 10`、`lr_max = 1.0`、`max_norm = 1.0`、
  `ε = 1e-6`（[B] PyTorch v2.9.0 的源码口径）；`A.2` 扫 `T ∈ {2,4,8}` × `max_norm ∈ {1.0, 0.5, 0.1}`；
  `B.1` 扫四组 `(W, T)`。**全部是推导值。**
- ★ 与计划 §3 表 A 的对应：**判据⑥**（`LR warmup + cosine` 的形状、逐点可手算 ✓）、
  **判据⑦**（裁剪 —— ★ **回链 Chunk 1 的表 A ⑦，并带上 `A.2b`**，此处**只实现不重定义阈值**）。
  ★ 计划 §3 表 A ⑦ 写的阈值是「相对误差 ≤ `1.1e-6/T`」，本文实现成
  `1.1e-6/max(T,1)` —— 那是 Chunk 1 在表 A ⑦ 里写死的 `T ≥ 1` 口径（**同一个阈值，不是另一条**）。
- ★★ **本文自己新增的一组判据**（不在计划表里）：`A.3a`–`A.3f`
  （`clip` 与 `lr` 的正交性、错序的闭式、`LR warmup` 的交叉、可分离性、源码级次序）
  —— 它们的**必要性**是**测出来的**：结构翻转抽查里，「把 `unscale` 挪到 `clip` 之后」
  一开始**溜过去了**（`rc=0`），因为 `S = 1` 时它是 no-op。

**外部来源**

- **[B]** **PyTorch v2.9.0** 的 `torch/nn/utils/clip_grad.py`
  （`https://raw.githubusercontent.com/pytorch/pytorch/v2.9.0/torch/nn/utils/clip_grad.py`，
  抓取 2026-09-23，**11427 B**，`sha256[:16] = 63efb273c62fd917`）。
  逐字（原始字节 2 次）：`clip_coef = max_norm / (total_norm + 1e-6)`。
  ★★ 它是**某实现的源码**（不是规范）⇒ **必须写明版本**；★ **`+1e-6` 在别的版本上是否相同
  未逐版核对**。
- **[C]** `Mixed Precision Training`（Micikevicius et al. 2017，arXiv **1710.03740**，
  **已确认 ICLR 2018**）：`unscale` 时机的那句逐字
  `It is simplest to perform this unscaling right after the backward pass but before gradient clipping`
  —— ★ **本文只引用它来陈述管线里的位置**，那条**判据**在 `mixed-precision/02`。
- **[C]** `Decoupled Weight Decay Regularization`（arXiv **1711.05101**，**已确认 ICLR 2019**）
  的 Algorithm 2 **第 11 行**：`η_t ← SetScheduleMultiplier(t)`（注释逐字
  `can be fixed, decay, or also be used for warm restarts`）
  —— ★ 这是「**调度属于优化器算法框**」的一手依据，也是本文与 `02` 的接口。
  ★ 且这两篇都是本文唯一的两个 [C] 级来源；`LR warmup` 与 cosine 的**形状**本身
  **没有**在这里引论文 —— 它是**本程序的设定**（§1.1），判定它「对不对」靠的是
  **闭式**（§1.2 的总账、§1.1 的三条边界值），**不是**某个来源的背书。
- **本库**：`90-training/autograd-and-backward/03-gradient-lifecycle.md`（表 A ⑦ 与 `A.2b`
  —— **裁剪的判据与阈值在那里写一次**）；`90-training/mixed-precision/02-amp-execution.md`
  （`A.2`/`A.2c`/`A.2d`：`unscale` 的时机；`A.3`：溢出 ⇒ 跳步）；
  `80-model-architectures/normalization/02-placement-and-qk-norm.md:83-100`
  （`LR warmup` 的机制解释）；`00-meta/glossary.md:146`（`warmup` 的另一语义）。

**上游契约与边界声明**

- `10-foundations/numerical-linear-algebra/04-iterative-methods.md:386` 把
  「**更新量 Δw 在什么精度下算**」让渡给本域 ⇒ 本文的 §1.4/§2.2 是它的**次序那一面**
  （`Δw = −lr·g_clipped`：`lr` 与裁剪**各自**作用在自己那一格）。
- ★★ **`LR warmup` 的同名不同指**（`glossary.md:146` 的 `warmup` 是 **benchmark 预热**）：
  本文**一律写全 `LR warmup`**，并**在每一处带上限定语**（判据表 B 的形态）。
- ★★ **边界声明跟着知识点走**：`LR warmup` 的**机制解释**在
  `80-model-architectures/normalization/02:83-100`，本文**回链、不重写**，
  且**这条声明写在本篇里**（不是只写在别处的抬头）。
- **禁区**：优化器状态的**分片与通信量**在 `100-parallelism/README.md:14`（`zero-fsdp`）
  —— 本文**一行都不写**。

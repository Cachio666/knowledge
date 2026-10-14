---
title: 初始化：He、Xavier 与 μP —— 以及三条必须订正的转述
slug: initialization
domain: 90-training
topic: initialization-and-stability
level: 3
difficulty: intermediate
prerequisites:
  - 90-training/autograd-and-backward/01-backward-from-scratch
tags: [initialization, he-init, xavier-init, fan-in, fan-out, mup, coordinate-check]
related:
  - 90-training/optimizers/01-sgd-and-momentum
  - 90-training/mixed-precision/01-type-selection
code:
  - code/01_init.py
status: stable
updated: 2026-09-23
---

# 初始化：He、Xavier 与 μP —— 以及三条必须订正的转述

> 初始化是训练里**只付一次**的那笔账 —— 但它的账是**乘性**的：每层差 1%、100 层之后差 2.7 倍
> （`Var[y_L] = Var[x]·Π_l (fan_in,l·Var[W_l])`）。
> 本文量三件事：**He**（`Var[W] = 2/fan_in`）、**Xavier**（`Var[W] = 2/(fan_in+fan_out)` 的**均匀**）、
> **μP**（读出层的初值尺度决定「同一组超参能不能跨宽度」）。
> ★ 顺带订正三条**在常见转述里都错了**的事（下节）。

## 0. 背景与动机

### ★★ 先说三条「与常见转述不一致」的（本文的直接内容）

| # | 常见转述 | 一手原文 |
|---|---|---|
| **Y1** | ★★★ 「**Xavier 初始化 = `arXiv:1001.0014`**」 | **那个编号是另一篇论文**：`[1001.0014] Jet Shapes and Jet Algorithms in SCET` —— **高能物理**（SCET = Soft Collinear Effective Theory）。Glorot & Bengio 的论文**不在 arXiv 上**；本文按 **`PMLR 9:249-256, 2010`** 引 |
| **Y2** | ★★ 「Glorot 论文提出了 `Xavier initialization` 这个名字」 | **原文里没有这个词**：`"Xavier initialization"` 在原文的文本层上计 **0** 次（四种数法都稳健地给 0）；单字 `Xavier` 出现 **4** 次，**全部是页眉的作者名**（`Xavier Glorot    Yoshua Bengio`，出现在标题页与第 250/252/254 页页眉）。★ 而 He 2015 **自己**写着 `This is called “Xavier” initialization in Jia2014`，同文另一处 `Jia2014` = **Caffe** ⇒ 可以确证的只是：**在本次取到的一手文献中，带引号的 “Xavier” 最早见于 He 2015，且它自称引自 Caffe**（★ Caffe 源码本身本次无法核实） |
| **Y3** | ★★ 「Xavier = `N(0, 1/n)`」 | **过度简化且方向错**。原文的约束是 `n_i Var[W^i] = 1`（前向，Eqn 10）与 `n_{i+1} Var[W^i] = 1`（反向，Eqn 11），而「**作为两者的折中**」（Eqn 12 前逐字 `As a compromise between these two constraints`）的结论是 `Var[W^i] = 2/(n_i + n_{i+1})`，实现是**均匀分布**（Eqn 16：`W ∼ U(−√(6/(n_j+n_{j+1})), √(6/(n_j+n_{j+1})))`，原文自己叫它 `the normalized initialization`）。⇒ **分母是 `fan_in + fan_out` 的【和】**、分布是**均匀** |

★★ **Y1 与 Y3 在本文里不只是一句订正，它们是可执行的断言**（`A.4`）：
把 Xavier 实现成 `N(0, 1/fan_in)` 的那一版，在 `(fan_in, fan_out) = (256, 4096)` 上的前向增益是
**`0.9984405631`**（≈ 1 ⇒ **它反而「保前向」**），而正确的那一版是 **`0.117464`** ——
两者相差 **`8.5` 倍**（`(fan_in+fan_out)/(2·fan_in)`，逐位等于闭式）。
⇒ 「Xavier = `N(0,1/n)`」**不是措辞差异，是结果差异**，而且**方向相反**。

### ★★ 一条口径声明：Glorot 的目标是 `1`，He 的目标是 `2`

两条公式里的 `2` **不是同一个 `2` 的来源**，这一点决定了怎么读本文的全部数字：

| | `Var[W]` | 公式里的 `2` 来自 | 「保持」的目标 |
|---|---|---|---|
| **Glorot & Bengio** | `2/(fan_in + fan_out)` | **两个约束（前向 Eqn 10 + 反向 Eqn 11）的折中** | **`1`**（倍率 `fan_in·Var[W]` = `1`）—— 原文的前提是「对称激活、`f'(0) = 1`」（**线性**假设） |
| **He** | `2/fan_in` | **ReLU 把二阶矩折半**（原文用 `E[x_l²] = ½ Var[y_{l−1}]`，见 Eqn 7 → Eqn 8） | **`1`**（**在 ReLU 之后**）；而在**不施加 ReLU** 的测量口径下，它就是 **`2`** |

★★ **本文（与配套程序）一律用「不施加 ReLU」的口径**（`x @ W` 的方差 ⇒ `fan_in·Var[W]`），
所以 He 的「目标」在本文的数字里是 **`2`**、Xavier 的是 `2·fan_in/(fan_in+fan_out)`。
★ **换口径必须换说法**：说「He 让前向方差保持 1」也对，但那是在**过了 ReLU 之后**的口径。

### 本文的范围（与禁区的界线）

★★ **`B.1` 量的是「一条没有残差、没有归一化的前向链」上的乘性累积** ——
它是 Glorot 原文 §4.2 的那条账（原文逐字 `the multiplicative effect through layers`）。
它**不是**下面这两条，本文**只回链、不重推**：

- **残差流的范数随深度增长**（`80-model-architectures/transformer/01-block-and-residual-stream.md:191,232-233,263`）；
- **误差沿深度的传播**（`130-model-optimization/quantization/07-outliers-and-propagation.md:124,205`）。

★ **`LR warmup` 的机制解释**也在别处（`80-…/normalization/02-placement-and-qk-norm.md:83-100`）
—— 它属 `optimizers/03` 的正文，本文只在这一句里回链。

## 1. 原理

### 1.1 He：`Var[W] = 2/fan_in`（`fan_in` 是**连接数**）

He 2015 的推导链（§2.1）逐字：

```
Eqn (7)   Var[y_l] = n_l · Var[w_l] · E[x_l²]
          ★ 原文紧接着警告：`It is worth noticing that E[x_l²] ≠ Var[x_l] unless x_l has zero mean.`
Eqn (8)   Var[y_l] = ½ · n_l · Var[w_l] · Var[y_{l−1}]      ← `E[x_l²] = ½ Var[y_{l−1}]`（ReLU）
Eqn (10)  ½ · n_l · Var[w_l] = 1,  ∀l                       ← 原文逐字：`A sufficient condition is`
          ⇒ `whose standard deviation (std) is √(2/n_l)`
```

★★ **两条必须一起读的**：

- **Eqn 10 是「充分条件」**（原文的词就是 `A sufficient condition is`）—— 转述时不得写成「充要」；
- **`n_l` 是「连接数」**（原文逐字 `the number of connections`）：对全连接层 = `fan_in`，
  对**卷积**是 `k²·c_in` ⇒ ★ **漏掉 `k²` 会让方差差 `k²` 倍**（`k = 3` 时 **9 倍**，见 `B.2`）。

★ **He 也给了反向的那一条**（Eqn 14），并且原文逐字说
`it is sufficient to use either Eqn.(14) or Eqn.(10) alone` ⇒
「只用其一」是**原文自己说的**。而本机量出来：**He 的反向增益是 `2·fan_out/fan_in`**
（在 `(256, 4096)` 上是 **`32.025920`**，`A.2b`）—— 也就是说 **He 保前向、不保反向**，
`fan_out ≫ fan_in` 时反向会放大 32 倍。

### 1.2 Xavier：两个约束的折中，是一个**和**、一个**均匀分布**

```
Eqn (10)  ∀i, n_i Var[W^i] = 1                ← 前向
Eqn (11)  ∀i, n_{i+1} Var[W^i] = 1            ← 反向
          ★ Eqn 12 之前逐字：`As a compromise between these two constraints`
Eqn (12)  ∀i, Var[W^i] = 2/(n_i + n_{i+1})
          ★ `We call it the normalized initialization`
Eqn (16)  W ∼ U(−√(6/(n_j+n_{j+1})), √(6/(n_j+n_{j+1})))
```

★★ **「折中」的定量形态有两面，本文两面都量了**：

| 量 | 闭式 | 在 `(256, 4096)` 上 | 本机实测（`A.2`） |
|---|---|---|---|
| **两侧之和** | `2·fi/(fi+fo) + 2·fo/(fi+fo)` | **`2`**（**逐位**） | — |
| 前向增益 | `2·fi/(fi+fo)` | `0.117647` | `0.117631` |
| 反向增益 | `2·fo/(fi+fo)` | `1.882353` | `1.882093` |
| ★ **两者的几何平均** | `2√(fi·fo)/(fi+fo) ≤ 1`（**AM-GM**） | `0.470588` | — |

⇒ **「既不保前向也不保反向」的准确说法是**：`fi ≠ fo` 时**两侧都偏离 1**，
而它们的**几何平均被压到 1 以下**（`B.3`）。
★ **而 `fi == fo` 时两侧都 `== 1`**（`A.2c`：实测 `0.995750 / 0.995750`）——
⇒ 那句「既不保前向也不保反向」**只在 `fi ≠ fo` 时成立**，判据**不是恒真的**。

★ 一个容易忽略的**重合点**：`fi == fo` 时 `2/(fi+fo) = 1/fi`，**Xavier 与「`N(0,1/fan_in)`」数值相同**
⇒ **那个错读法在方阵上不会暴露**，只在 `fi ≠ fo` 时露出来（这正是 `A.4` 用 `(256,4096)` 的原因）。

### 1.3 与 `torch.nn.init` 的互校（[B] 级：**某实现的文档，不是规范**）

PyTorch 2.14 的 `nn.init` 页给出的两式（页面上是 MathML + LaTeX 双写）：

```
kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu', generator=None)
    bound = gain × √(3 / fan_mode)          ★ 页面上还有一句：`Also known as He initialization.`
xavier_uniform_(tensor, gain=1.0, generator=None)
    a     = gain × √(6 / (fan_in + fan_out))  ★ `Also known as Glorot initialization.`
```

因为 `Var[U(−b, b)] = b²/3`：

| 路径 | 计算 | 结果 | 与论文的关系 |
|---|---|---|---|
| `kaiming_uniform_(mode='fan_in')` | `b = √2·√(3/fan_in)`，`b²/3` | **`2/fan_in`** | **与 He 同方差**（实测相对差 **`2.22e-16`**，跨 6 个 `fan_in`） |
| `xavier_uniform_(gain=1)` | `a = √(6/(fi+fo))`，`a²/3` | **`2/(fi+fo)`** | **与 Glorot Eqn 12 同式**（实测相对差 **`2.22e-16`**） |

★★ **两个默认值都要看**：`mode` 的默认是 `'fan_in'`（He 那一支），
而 `nonlinearity` 的默认是 **`'leaky_relu'`、`a = 0`** ——
`a = 0` 时 leaky_relu 的 gain **逐位等于** relu 的 gain（都是 `√2`，`C.3`）⇒
**默认值恰好落在 He 的 ReLU 那一支上**。★ 但这条等价**不是恒真的**：`a = 0.2` 时 gain 变成
`1.386750491…`，差 **`2.746e-2`**。

### 1.4 μP：**读出层的初值尺度**决定「同一组超参能不能跨宽度」

μP（Tensor Programs V，`arXiv:2203.03466`，**已确认 NeurIPS 2021**）的主张逐字是

```
many optimal HPs remain stable even as model size changes
```

★★ **它给了一个可直接落地的诊断工具**（附录 D.1，逐字）：

```
implementing it correctly can in practice be error-prone …
An incorrect implementation will see some activation vector blow up or shrink to zero with width
```

⇒ **coordinate check（坐标检查）**：把宽度扫一遍，看每个（前）激活向量的**平均坐标尺度**
会不会随宽度 **blow up 或 shrink to zero**。

★★ **本文把它落成两个可拟合的斜率**（`A.5` / `A.5b`）：

| 量 | SP（标准参数化） | μP（读出层初值 `∝ 1/fan_in²`） |
|---|---|---|
| 读出层初值方差 | `1/width` | `1/width²` |
| logits 的方差 | `Var[h]`（**`Θ(σ²)`**） | `Var[h]/width`（**`Θ(σ²/width)`**） |
| **本机拟合的斜率**（`A.5`） | **`-0.0094`**（≈ 0） | **`-1.0094`**（≈ −1） |
| 一步 GD 的**相对**变化量 `\|Δy/y\|` | `Θ(width)`（斜率 **`+0.9869`**） | `Θ(1)`（斜率 **`-0.0131`**） |

★ 第二行是 `A.5b` —— 它量的正是 coordinate check 关心的那个量：**固定 `lr` 时，
相对变化量随宽度线性增长**（blow up）；把**读出的步长按宽度缩放**之后斜率归零。

★★ **口径声明**：本文量的是「**同一组超参在宽度上的稳定性**」这一半。
它**没有**证明「超参可以从窄模型迁移到宽模型」—— 那是论文的主张（`μTransfer`），
本文**不复述它当结论**。

### 1.5 一次 spike… 不，是一次**初始化偏差**的乘性账（`B.1`）

```
Var[y_L] = Var[x] · Π_{l} (fan_in,l · Var[W_l])
```

本机（`B.1`）：`L = 100` 层、每层增益 `1.01` ⇒ 乘积 = **`2.7048138294`**
（= `(1.01)^100`，与另一条路径 `exp(Σ ln(1+ε))` 对拍，相对差 `2.22e-16`）。
⇒ **单层 1% 的偏差，100 层之后是 170%**；而每层恰好为 1 时乘积**逐位等于 1**（`B.1b`）。

★ **范围限制再说一次**：这是**没有残差、没有归一化**的链；残差流的账在
`80-…/transformer/01`，误差沿深度的传播在 `130-…/07-outliers-and-propagation`。

## 2. 执行映射

### 2.1 选哪一个（判据表的前半）

| 情形 | 用哪个 | 为什么 |
|---|---|---|
| 后面接 **ReLU / Leaky ReLU** | **He**：`Var[W] = 2/fan_in`（或 `kaiming_uniform_`） | 折半的因子在公式里（Eqn 8 的 `½`） |
| 后面接 **tanh / sigmoid / linear** | **Xavier**：`Var[W] = 2/(fi+fo)` 的**均匀** | 原文的前提就是「对称、`f'(0)=1`」 |
| 想**同时**保住前向与反向 | 只有 `fi == fo` 时才可能 | `A.2`/`B.3`：`fi ≠ fo` 时几何平均 < 1 |
| 想让**超参跨宽度** | **μP**（读出层初值 `∝ 1/fan_in²` + 相应的步长缩放） | `A.5`/`A.5b` 的两个斜率 |
| 卷积 | `fan_in = k²·c_in`、`fan_out = k²·c_out` | 漏 `k²` ⇒ 差 `k²` 倍（`k=3` ⇒ 9 倍） |

### 2.2 「同方差」这条判据能证明什么、不能证明什么

★★ He 原文的实现是**正态**（`zero-mean Gaussian distribution whose standard deviation (std) is √(2/n_l)`），
而 `kaiming_uniform_` 是**均匀** —— 两者**同方差，不同分布**（`C.1`）：

| | 方差 | 超额峰度 |
|---|---|---|
| Gaussian（He Eqn 10 的实现） | `0.00779749` | **`+0.0037`**（理论 `0`） |
| Uniform（`kaiming_uniform_` 的实现） | `0.00778343` | **`-1.1966`**（理论 `−1.2`） |

⇒ `A.3` 的「同方差」**只说二阶矩**。把它读成「两个实现等价」就是**升格**（同方差**不蕴含**同分布）。

### 2.3 本机实测的一组典型值

| 量 | 值（`256×4096`，`python3 3.13.5` + `numpy 2.4.4`） |
|---|---|
| He 前向增益 | `2.001264`（闭式 `2`） |
| He 反向增益 | `32.025920`（闭式 `32`） |
| Xavier 前向 / 反向 | `0.117631` / `1.882093`（闭式 `0.117647` / `1.882353`） |
| `kaiming_uniform_(mode='fan_out')` 与 `'fan_in'` 的方差比 | **`16.000000`**（= `fan_out/fan_in`） |
| 错读法 `N(0,1/fi)` 与正读法的前向增益之比 | **`8.500000`**（= `(fi+fo)/(2·fi)`） |

## 3. 实现与演进

| 阶段 | 写法 | 它换来了什么、付了什么账 |
|---|---|---|
| **常数 / 任意小随机** | `W ∼ N(0, 0.01)` 之类 | 无脑；代价是**乘性累积**（`B.1`：100 层 2.7 倍） |
| **Xavier（2010）** | `U(±√(6/(fi+fo)))` | 前向与反向**各让一步**（Eqn 12）；代价是 `fi ≠ fo` 时**两侧都不保**（几何平均 < 1） |
| **He（2015）** | `std = √(2/fan_in)` | **把 ReLU 的 `½` 写进公式**（Eqn 8→10）；代价是**反向**变成 `2·fo/fi`（`fi ≠ fo` 时差很多） |
| **标准库默认（[B]）** | `kaiming_uniform_(mode='fan_in')` | **与 He 同方差**（方差层面等价）；代价是**分布不同**（均匀 vs 正态），且默认非线性名是 `leaky_relu` |
| **μP（2021）** | 读出层初值 `∝ 1/fan_in²` + 步长缩放 | **超参跨宽度**；代价是多了一套**容易写错**的规则（论文自己给的诊断工具就是 coordinate check） |
| **「Xavier」这个名字** | ★ 由**别人**安上 | He 2015 逐字 `This is called “Xavier” initialization in Jia2014`，而同文 `Jia2014` = **Caffe** ⇒ **Glorot 原文里没有这个名字**（Y2） |

★ **名字史与公式史是两条线**：本文的 `A.3`/`A.3c` 把「`kaiming_uniform_` ≡ He」「`xavier_uniform_` ≡ Glorot」
这两条**同义关系**钉在 [B] 文档自己的两句话上（`Also known as …`），
而 `D.6` 把「Glorot 原文里那个词计 0 次」钉住 —— **两条并列，正好把 Y2 的张力摆出来**。

## 4. 踩坑与失败模式

**① ★★★ 引用一个不存在的编号。**

- **现象**：写「Xavier 初始化（`arXiv:1001.0014`）」。
- **成因**：**那个编号属于另一篇论文**（`Jet Shapes and Jet Algorithms in SCET`，hep-ph）；
  Glorot & Bengio 的论文**不在 arXiv 上**。
- **规避**：**凡给外部编号，先打开看标题**；本文按 `PMLR 9:249-256, 2010` 引（`A.4a`）。

**② ★★ 把 Xavier 记成 `N(0, 1/n)`。**

- **现象**：`W ∼ N(0, 1/fan_in)`。
- **成因**：原文的 Eqn 12 是 `2/(n_i + n_{i+1})` 的**和**、Eqn 16 是**均匀**分布。
- **代价**：在 `(256,4096)` 上它与正确实现差 **8.5 倍**、而且**方向相反**
  （错的那个**保前向**）⇒ 不是措辞差异（`A.4b`）。
- **规避**：断言**分母是「和」**（`A.4c` 把这条钉在**源码结构**上 ——
  因为 `fi == fo` 时两条公式数值恰好相同，**光靠数值会退化成 no-op**）。

**③ ★ 把 `mode` 写成 `'fan_out'`。**

- **现象**：`kaiming_uniform_(mode='fan_out')` 与 `'fan_in'` 在 `(256,4096)` 上差 **16 倍**。
- **成因**：`fan_out/fan_in = 16`。
- **规避**：`A.3b` 把这条闭式钉住 ⇒ `A.3` 的「同方差」**区分了两个实现**。

**④ ★ 卷积的 `fan_in` 漏掉 `k²`。**

- **现象**：代码照跑，只是尺度错了 `k²` 倍（`k=3` ⇒ **9 倍**）。
- **成因**：`n_l` 是**连接数**（`k²·c_in`），不是通道数（He 原文逐字 `the number of connections`）。
- **规避**：`B.2` 把它量出来；`B.2b` 给出反面（`k=1` 时两种算法逐位相同 ⇒ 判据只在 `k>1` 时有载荷）。

**⑤ ★★ 把「解析口径」当成「没有噪声」。**

- **现象**：`A.1d` 的第一版用 `3·SE(B)` 当阈值，在 `(128,128)` 上算出 **4.15 倍**的偏差、
  看起来像「两条路径不一致」。
- **成因**：**口径 A（逐列解析）也有噪声，噪声来自 `W` 本身**：`SE_A = √(2/(fi·fo))`。
  实测 `SE_A/sd_B` 在 `(128,128)` 上是 **1.40**（比抽样口径**还大**）、在 `(256,256)` 上是 **0.29**
  ⇒ **跨过 1**（`A.1c`）。
- **规避**：阈值用**两条路径标准误的合成**（`A.1d`）；★ 教训是那句通用的：
  **「我量出来不一样」先问「我量的是不是同一个东西」**。

**⑥ ★★ 拿朴素公式当抽样口径的标准误。**

- **现象**：把 `√(2/n_rep)` 当口径 B 的标准误（`n_rep = 300` 时 `8.2%`）。
- **成因**：每次重复的 `(x@W).var()` **已经在 `fan_out` 个坐标上平均过一次** ⇒
  这个估计量的有效样本数**远大于** `n_rep`；实测经验 sd 只有 **`0.4%~1.0%`**，
  朴素公式**高估 ≥ 8.4 倍**（`A.1b2`）。
- **规避**：用**经验 sd**（5 个种子）作标准误；★ 并且知道**「抽样口径」本身不止一个口径**
  （口径 B 每次重抽 `W`、口径 B′ 固定 `W`，两者的 sd 之比实测跨 **2.6 倍**，`A.1b3`）。

**⑦ ★★ 把计划里的「多种子」判据用在单种子上。**

- **现象**：计划 §3 表 A ① 的带子 `[1.95, 2.05]`（±2.5%）用在**单种子**上。
- **成因**：单种子的最大相对偏差实测 **2.66%** ⇒ 与带子**半宽 2.5%** 同量级
  （比值 0.53，`A.1b`）⇒ **~1σ 宽**，换种子就可能假失败。
- **规避**：判据写成「**5 子均值**落在带子里」；★ 顺带把「单子偏差与带子同量级」也**断言**出来
  （让这条纪律有一个可执行的落点）。

**⑧ ★ 把「同方差」读成「同分布」。**

- **现象**：拿 `A.3` 的结论说「`kaiming_uniform_` 与 He 的实现等价」。
- **成因**：一个正态一个均匀 —— 二阶矩相同、**四阶矩差 `1.2`**（`C.1`）。
- **规避**：把峰度差也断言出来。

## 5. 判据表

| 可机械判定的条件 | 结论 |
|---|---|
| 想判断「He 的 `√(2/fan_in)` 对不对」 | 在**不施加 ReLU** 的口径下，前向增益 == `fan_in·Var[W]` == **`2`**（`A.1`，7 对 `(fi,fo)` 上偏差 ≤ 1.72·`SE_A`） |
| 想判断「Xavier 的 `2/(fi+fo)` 对不对」 | 前向 + 反向 == **`2`**（**逐位**）；`fi == fo` 时两侧都 `== 1`；`fi ≠ fo` 时几何平均 `2√(fi·fo)/(fi+fo) < 1`（`A.2`/`B.3`） |
| He 保不保**反向** | **不保**：反向增益 == `2·fan_out/fan_in`（`(256,4096)` 上是 **32**）—— 原文自己说「两者只用其一」（`A.2b`） |
| 卷积的 `fan_in` 是多少 | **连接数** `k²·c_in`；漏 `k²` ⇒ 差 `k²` 倍（`k=3` ⇒ 9 倍）（`B.2`） |
| `kaiming_uniform_(mode='fan_in')` 与 He 的关系 | **同方差**：`b²/3 == 2/fan_in`（相对差 `2.22e-16`）；★ **但不同分布**（正态 vs 均匀，峰度 `0` vs `−1.2`） |
| `xavier_uniform_` 与 Glorot 的关系 | `a²/3 == 2/(fi+fo)`，且都是**均匀**；[B] 文档自己写着 `Also known as Glorot initialization.` |
| 想知道 `mode='fan_out'` 错多少 | `fan_out/fan_in` 倍（`(256,4096)` 上是 **16**） |
| 想知道「Xavier = `N(0,1/n)`」错多少 | `(fi+fo)/(2·fi)` 倍（`(256,4096)` 上是 **8.5**），且**方向相反**（错的那个保前向） |
| `leaky_relu, a=0` 与 `relu` 的 gain | **逐位相同**（`√2`）；★ 但 `a=0.2` 时差 **2.7e-2** ⇒ 这条等价不是恒真的 |
| 一条前向链上每层偏 `ε` 的后果 | **乘性累积** `(1+ε)^L`（`L=100, ε=0.01` ⇒ **2.7048 倍**）；每层恰为 1 时乘积逐位为 1 |
| 想知道「He 到底是正态还是均匀」 | 原文是**正态**（`zero-mean Gaussian distribution`，Eqn 10 之后逐字）；`kaiming_uniform_` 是**均匀** ⇒ 同方差、不同分布 |
| 想知道「该用 Xavier 还是 He」 | 看**后面接什么**：ReLU 系 ⇒ He（`2/fan_in`）；tanh/linear ⇒ Xavier（`2/(fi+fo)`、均匀） |
| 想知道 μP 实现对不对 | **coordinate check**：logits 方差对宽度的斜率（SP `≈ 0`、μP `≈ −1`）；相对变化量的斜率（固定 `lr` 时 `+1`、按宽度缩放后 `0`） |

> ★ 阈值与口径（`code/01_init.py` 的 `D.3`/`D.4` 逐位钉住）：
> `SEED = 20260950`、`N_REP = 300`（口径 B 的重复数；探针 D9 用 1000）、
> `GAIN_SE = 5.0`（`|rel| < 5·SE_A`）、`HE_BAND_LO/HI = 1.95/2.05`、
> `TOL_EXACT = 0.0`、`TOL_TIGHT = 1e-12`、`TOL_IDENT = 1e-15`、`TOL_MEASURABLE = 1e-2`、
> `MUP_TOL_SLOPE = 0.05`、`K_FAN_TEST = (3, 64)`、`DEPTH_L = 100`、`PER_LAYER_EPS = 0.01`。
> ★ **两套测量口径**：口径 A（解析列方差，`SE_A = √(2/(fi·fo))`）· 口径 B（探针口径，抽样）。
> ★ 放宽任何一条容差都要问一句「它会不会变成恒真」。

## 6. 小结与延伸

**一句话**：初始化的**公式**只有三行（`2/fan_in`、`2/(fi+fo)`、读出层 `1/fan_in²`），
而**围绕它们的转述**错了三条 —— 一个不存在的编号、一个原文里没有的名字、一个错的分布与分母；
★ 三条都**不是靠读文献解决的，是靠断言解决的**（`A.4`）。

- **梯度怎么从这些权重里反传回来** ⇒ `01-backward-from-scratch`（本主题的底座）。
- **优化器怎么用这些梯度** ⇒ `90-training/optimizers/01-sgd-and-momentum`。
- **低精度下的初始权重** ⇒ `90-training/mixed-precision/01-type-selection`。
- **`LR warmup` 的机制解释**（post-LN 的平均场理由） ⇒
  `80-model-architectures/normalization/02-placement-and-qk-norm.md:83-100`（**正文在彼**）。
- **残差流的范数随深度增长** ⇒ `80-model-architectures/transformer/01-block-and-residual-stream.md:191,232-233,263`（**正文在彼**）。
- **误差沿深度的传播** ⇒ `130-model-optimization/quantization/07-outliers-and-propagation.md:124,205`（**正文在彼**）。
- ★★ **一处必须写明的让步**：本文的判据**全部**是「本机玩具规模的真跑」
  （`python3 3.13.5` + `numpy 2.4.4`、**无 GPU**、不训练任何真模型）⇒
  **不得支撑关于大规模训练的结论**。

## 编译与运行

```bash
cd code
make run                    # 三个编号程序依次跑；本程序是 01_init.py
python3 01_init.py          # 只跑本程序
```

- 配套程序：**`code/01_init.py`**（编号与外键严格对应）。
- 依赖：`numpy`（`requirements.txt` 里**只有**它）。**不需要网络、不需要 GPU。**
- 失败标记：`// FAIL: init-stability`（本主题自定义，**不跨主题照抄**）；程序内有两条「标记自检」
  断言（`D.1` / `D.2`）守着它，`mutate.py` 用 **AST** 读出该值（不得写死）。
- 程序的输出分四段：`A` 增益与互校（含 **Y1/Y3 的可执行对照**、coordinate check）、
  `B` 代价模型（乘性累积 / 卷积的 `k²` / 几何平均上界）、
  `C` 边界与反例（同方差 ≠ 同分布 / `fan_in = 0` / gain 表的巧合 / 漏掉那个 `3`）、
  `D` 标记自检、口径与阈值钉住、specs 常量钉住、独立性证据。
- ★★ **口径/阈值/标记自检一律在 `preflight()` 里、跑在任何计算之前**（本库的 **P23**）。

## 参考

**本机实测值 / 推导值**

- `code/01_init.py` 的 `make run` 输出：`python3 3.13.5` + `numpy 2.4.4`，macOS arm64，
  **无 GPU**。口径：`seed = 20260950`；口径 A 扫 7 对 `(fan_in,fan_out)`
  （含 `(256,4096)`、`(4096,4096)`、`(2048,512)`）；口径 B/B′ 扫
  `(128,128)`、`(256,256)`（`n_rep = 300`、5 个种子）；μP 扫
  `width ∈ {32,64,128,256,512,1024,2048,4096}`。**全部是推导值。**
- ★ 与计划 §3 表 A 的对应：**判据①**（He 的前向方差保持 `2` ✓）、
  **判据②**（Xavier 在 `fan_in ≠ fan_out` 时既不保前向也不保反向 ✓）、
  **判据③**（与 `kaiming_uniform_(mode='fan_in')` 同方差 ✓，★ **反面对照 `mode='fan_out'` 差 16 倍 ⇒ 判据可证伪**）。
  ★★ **一处收窄**：计划 §3 表 A ② 的括注写「（理论 `0.1176`，**都远离 1 与 2**）」——
  本机实测**前向** `0.1176` 确实远离 1 与 2，但**反向**是 `1.8824`，**离 2 只有 6%** ⇒
  本文只写「**两侧都偏离 1**」+「几何平均 `0.4706 < 1`」这两条**都能被逐位验证**的说法。
- ★★ **本文自己新增的一组判据**（不在计划表里）：`A.1b2`/`A.1b3`/`A.1c`/`A.1d`
  （两种抽样口径的标准误、解析口径也有噪声、三条路径互校）—— 它们的**必要性是测出来的**：
  `A.1d` 的第一版在 `(128,128)` 上给出 4.15 倍的「不一致」，追下去才发现**两边都有噪声**。

**外部来源**

- **[C]** K. He, X. Zhang, S. Ren, J. Sun，`Delving Deep into Rectifiers: Surpassing Human-Level
  Performance on ImageNet Classification`，**arXiv 1502.01852**（**不确认同行评审** —— abs 页
  `Comments:` 字段不存在）。抓取 `https://ar5iv.labs.arxiv.org/html/1502.01852`，
  2026-09-23（**289072 B**）。逐字：`A sufficient condition is`（原始字节 1 次）、
  `whose standard deviation (std) is`（1 次）、`it is sufficient to use either`（1 次）、
  `the number of connections`（1 次）、`It is worth noticing that`（1 次）。
- **[C]** X. Glorot & Y. Bengio，`Understanding the difficulty of training deep feedforward
  neural networks`。★★ **它不在 arXiv 上** —— 载体是 **PMLR 官方页**
  `https://proceedings.mlr.press/v9/glorot10a.html`（2026-09-23，**18993 B**）
  与它的 PDF 文本层。逐字：`PMLR 9:249-256, 2010`（剥标签后 1 次）、
  `As a compromise between these two constraints`（原始字节 1 次）、
  `the normalized initialization`（原始字节 3 次）、Eqn 10/11/12/16；
  ★★ **`Xavier initialization` 计 0 次**（四种数法都稳健），而 `Xavier` 单字 4 次**全是页眉作者名**。
  ★ **`normalized initialization` 的计数不可复现**（本次四种数法给 **13 / 14 / 17 / 18**）
  ⇒ **本文不写那个数**（只写 `Xavier initialization` = 0 这半）。
- **[C]** G. Yang 等，`Tensor Programs V: Tuning Large Neural Networks via Zero-Shot
  Hyperparameter Transfer`，**arXiv 2203.03466**（**已确认 NeurIPS 2021** —— abs 页
  `Comments:` 逐字 = `NeurIPS 2021`）。抓取 ar5iv（**1045657 B**）。逐字：
  `many optimal HPs remain stable even as model size changes`、
  `implementing it correctly can in practice be error-prone`、
  `An incorrect implementation will see some activation vector blow up or shrink to zero with width`。
  ★★ **一处方法学让步**：Table 3 的 **LR 缩放列**在 ar5iv 上是**表格扁平化**后的文本、列序无法可靠复原
  ⇒ 本文**不引用那张表的 LR 数值**；`A.5`/`A.5b` 用的是**本文自己推导的闭式**，
  论文只提供「判据形态」那一句（并已登记 `UNSOURCED`）。
- **[B]** **PyTorch 2.14** 的 `torch.nn.init` 页
  （`https://docs.pytorch.org/docs/2.14/nn.init.html`，2026-09-23，**736073 B**）。
  逐字：`Also known as He initialization.` / `Also known as Glorot initialization.`（剥标签后各 2 次）、
  `kaiming_uniform_ ( tensor , a = 0 , mode = 'fan_in' , nonlinearity = 'leaky_relu' , generator = None )`、
  `xavier_uniform_ ( tensor , gain = 1.0 , generator = None )`、`calculate_gain` 的表格
  （`Linear/Identity 1` · `Conv{1,2,3}D 1` · `Sigmoid 1` · `Tanh 5/3` · `ReLU √2` ·
  `Leaky Relu √(2/(1+negative_slope²))` · `SELU 3/4`）。
  ★★ **它是「某实现的文档」，不是规范**（本域的头号错误模式）——
  两式的 `bound` / `a` 在页面上是 **LaTeX + MathML 双写** ⇒ **`bound = gain × √(3/fan_mode)`
  这样的字面串在原始字节上计 0** ⇒ 本文只把两个数（`3` / `6`）做成常量，公式本身**不做引文核对**。
- **[B]** **PyTorch 2.14** 的 `torch.nn.utils.clip_grad_norm_` 页（同一次抓取，**179174 B**）
  —— 它在 `02-clipping-and-monitoring.md` 里用，本文只用到它的 `+1e-6`（与 `A.1` 的裁剪判据同源）。

**上游契约与边界声明**

- ★★ **本文与上游的关系是「自己立」**：`initialization-and-stability` 在全库**没有指向本域的让渡**
  （`80-…/transformer/01:49` 把「训练侧的账」让给了 `80` 自己的 `normalization/02`）⇒
  **`LR warmup` 被占**，本文一行都不写它的机制（只回链）。
- ★★ **边界声明放在本文里，而不是只写在某处的抬头**：本文写「方差沿深度的乘性累积」时，
  **同时**写明它的范围（无残差、无归一化）并把两处真正的正文（残差流 / 误差传播）回链出去。
- **禁区**：`warmup` 的机制解释 → `80-…/normalization/02:83-100`；
  残差流/深度 → `80-…/transformer/01:191,232-233,263`；误差传播 → `130-…/07-outliers-and-propagation:124,205`；
  MoE 的 aux loss 尺度 → `80-…/moe/03:260`。

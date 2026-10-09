---
title: 数据并行的梯度平均（为什么是平均不是求和）
slug: gradient-averaging
domain: 100-parallelism
topic: data-parallel
level: 3
difficulty: intermediate
prerequisites:
  - 90-training/optimizers/01-sgd-and-momentum
  - 90-training/autograd-and-backward/03-gradient-lifecycle
tags: [data-parallel, gradient-averaging, all-reduce, loss-reduction, floating-point]
related:
  - 100-parallelism/data-parallel/02-gradient-synchronization
  - 10-foundations/performance-models/02-amdahl-and-scaling
code:
  - code/01_grad_averaging.py
status: stable
updated: 2026-09-23
---

# 数据并行的梯度平均（为什么是平均不是求和）

> **DP 只有一刀：切数据。这一刀的正确性只由两件事决定 —— 除数在哪一边、`loss` 用什么 reduction 口径。**
> ★★ 而 `all_reduce` 在真实系统里的**默认运算就是求和**（[B] PyTorch 2.14 的
> `all_reduce(tensor, op=_ReduceOp = ReduceOp.SUM, …)`）⇒ **「平均」那一步必须由 DP 自己补**，
> 谁都不替你做。少了它，梯度**恰好是 `N` 倍**（本机 `4.000000`，`N = 4`）。
> ★★★ 而这条错误**在 `N = 1` 上完全测不出来**（两种写法恒等）—— 这是它的适用边界。

## 0. 背景与动机

### 一个具体的对照场景

单卡放不下一个 batch 了（不是模型放不下，是**数据**放不下：激活值与会话长度一起涨）。
最省事的一刀是**把 batch 切到 `N` 张卡上**：每张卡各拿 `B` 行，各算一份梯度。

```
单卡：32 行 → 一次算出 g        （本机玩具规模：N=4、B=8、Ψ=64 个参数）
DP：  4 张卡各 8 行 → 4 份 g_i → ??? 合并 → 更新
```

「合并」那一步的朴素读法是**求和**（`Σ g_i`）—— 因为集合通信的默认运算就是求和，
而求和是最好实现的。**这一步是错的**，而且错得**很安静**：训练照样跑、loss 照样降，
只是**有效学习率被放大了 `N` 倍**（本机比值 `4.000000`）。

真正的语义是：**全局 batch 的梯度 = 各卡梯度的平均**。

### ★★ 本文的范围（与禁区的界线）

| 本文**写** | 本文**不写**（回链目标） |
|---|---|
| DP 的**语义**：平均怎么来、`loss` 的 reduction 口径与除数怎么配对、除法的位置 | **梯度规约的算法**（ring allreduce、`V(N) = 2(N−1)/N × S`、reduce-scatter + all-gather 的阶段划分）→ `110-communication/collectives`）、`10-foundations/performance-models/02-amdahl-and-scaling.md:209` |
| 梯度的**字节账**（`2Ψ`）与「每卡常量、全局 batch 变」 | **扩展效率模型**（Amdahl / Gustafson / `N₅₀`）→ `10-foundations/performance-models/02-amdahl-and-scaling.md` |
| 平均**不增加通信量**这一句的落法（本机可跑的那一部分） | **通信-计算重叠**（双缓冲、分块流水）→ `110-communication` 的 `overlap` |
| —— | ★★ **梯度累积与 batch 放大的等价性（含归一化层时的失效）** → `90-training/autograd-and-backward/03-gradient-lifecycle.md:71,223-224`（**正文在彼**，本文只写**「无归一化层」这一支**并回链） |
| —— | **`16Ψ` 的组成与字节分解**（`2Ψ+2Ψ+KΨ=16Ψ`）→ `90-training/checkpointing/01-state-and-bytes.md`（§1.3 + 判据表 + `code/01_state_bytes.py:310` 的 `A.3`） |

★ **一处必须写明的让步**：本文的一切数值都是**本机玩具规模的真跑**
（`python3 3.13.5` + `numpy 2.4.4`、macOS arm64、**无 GPU、无多卡、无 NCCL、无真实网络**）。
★★ **强制限定语：本文验证的是「切分的代数」，不是 NCCL，也不是真机**
—— 通信在配套程序里被实现为一个**显式的张量搬运**（`all_reduce = Σ局部`），
依据是 [B] 的两条逐字语义（§1.1 的引文）。

## 1. 原理

### 1.1 ★ `all_reduce` 的语义：文档说它是**逐元素的求和**

[B] NCCL 2.31.2 的用户指南在 `AllReduce` 一节逐字给出**结果语义**：

> `The AllReduce operation performs reductions on data (for example, sum, min, max) across devices and stores the result in the receive buffer of every rank.`

以及 sum 形式的**逐元素**定义（★ 前一句里的 `sum` 与 `k` 在原页面上是被 `<em>` 包住的，
跨标记 ⇒ 我们只引中段）：

> `each rank will provide an array in of N values, and receive identical results in array out of N values,`

[B] PyTorch 2.14 的 `torch.distributed.all_reduce` 签名里，`op` 的**默认值**逐字是：

> `torch.distributed.all_reduce(tensor: Tensor, op: _ReduceOp = ReduceOp.SUM, group=…, async_op=False)`

★★★ **两条合起来就是本文的全部前提**：
1. **`all_reduce` 求的是和**（而且默认就是和）；
2. **它把同一份结果写回每一个 rank** —— 所以「求平均」这件事**没有任何一张卡能替别人做**，
   每张卡各自除以 `N` 是**等价的**。

★★ **一处必须写明的读法**：`NCCL` 的三份文档里**没有**说 `AllReduce` 的 `op` 有默认值
（我们数过 `default` 的出现位置：`usage/collectives` 上它**计 1 次**，是**搜索表单的
hidden 输入** `<input type="hidden" name="area" value="default" />`，
`api/colls` 与 `api/types` 上也不描述 `op` 的默认值）
⇒ **「默认 SUM」这一条只能引 PyTorch 的签名**，写成「NCCL 的 AllReduce 默认 SUM」是
**把两条来源混成一条**（已登记 `UNSOURCED`：`nccl_allreduce_default_op`）。

### 1.2 ★★ 判据 ①：`N` 卡平均 == 单卡大 batch 的梯度

记全局 batch 为 `N·B` 行、参数向量为 `g`。每张卡在自己的 `B` 行上算**平均**梯度
`g_i = (1/B)·Σ_{行∈i} ∇ℓ`，则

```
(1/N)·Σ_i g_i = (1/N)·Σ_i (1/B)·Σ_{行∈i} ∇ℓ = (1/(N·B))·Σ_{全部行} ∇ℓ
```

⇒ **卡间平均 == 单卡在 `N·B` 行上的平均梯度**（代数上完全相同），但**浮点路径不同**：
被测路径是 `N` 次矩阵乘 + `N−1` 次加法 + 一次除法，参照是**一次**矩阵乘
⇒ 两条路径的差落在**机器精度档**。

**本机实测**（`code/01_grad_averaging.py` 的 `A.1`，`N=4、B=8、Ψ=64`）：
`max|N 卡平均 − 单卡大 batch| = 1.665e-16`（`g` 的量级 `0.494`）。

⚠ **前提必须写出来：没有按片统计的归一化层。**
★★ **那一条等价性及其失效条件已经由 `90-training/autograd-and-backward/03-gradient-lifecycle.md`
写过**（`:71` 与判据表 `:223-224`）⇒ **本文只有「无归一化层」这一支** ——
**正文在彼，本文只回链**。
★ **订正（2026-09-23）**：本节原先在这里**复述了机制从句**（「BatchNorm 用的是当前这一片的
均值/方差…不是同一个函数」以及「有 BN 时必须改用同步 BN 或放弃累积」）与上游的实测值
`4.173e-01` —— 那**是 `90` 的正文**，违反 `CONTRIBUTING` §7.2 的「**只放链接，不复制内容**」，
也与本文 `§0` 范围表自称的「只写「无归一化层」这一支并回链」不一致。**已删**。

### 1.3 ★★★ 判据 ③：`loss` 的 reduction 口径与卡间除数**必须配对**

DP 里有两个独立的口径，它们**乘起来**才是全局的除数：

| 口径 | 取值 | 除数 |
|---|---|---|
| 每卡 `loss` 的 reduction | `mean` | `B`（本卡行数） |
| 卡之间的合并 | 平均 | `N` |

⇒ 「全局 batch 的平均梯度」要求 **两者乘起来 = `1/(N·B)`**。四种组合的实测比值：

| `loss` reduction | 卡间再除以 `N` | 比值（相对正确解） |
|---|---|---|
| `mean` | 是 | **`1.000000`** ✓ |
| `mean` | 否 | **`4.000000`**（= `N`） |
| `sum` | 是 | **`8.000000`**（= `B`） |
| `sum` | 否 | **`32.000000`**（= `N·B`） |

（本机 `N = 4`、`B = 8`；四格的比值恰好是 `{1, N, B, N·B}` ——
**这是一个可判定的集合**，不是「看情况」。）

★ 值得注意：`sum` + 卡间除以 `N` 这一格**也不是**对的 —— 它差 `B` 倍。
⇒ **「除以 `N`」并不等于「平均」**：除谁、什么时候除，是一个**两维**的口径问题。

### 1.4 ★★★ **除法的位置**：`(Σg)/N` vs `Σ(g/N)`

平均有两种代数等价的写法：

```
(Σg)/N      —— 先加后除
Σ(g/N)      —— 先除后加
```

**它们在浮点上不等**，而**差的量级是 1 ulp** —— 关键在于：**在什么取值下它们相等**。

**本机实测**（`code/01_grad_averaging.py` 的 `A.2`；
★ **口径三条**：**加法的实现 = Python 内建 `sum()`**（顺序累加）、
**被归约的轴 = 卡（rank）轴**（列表顺序）、**除法的位置**如上）：

| 维度 `D` | `N` 是 2 的幂（`2/4/8/64`） | `N` 非 2 的幂（`3/5/7/10/100`） |
|---|---|---|
| `2` | **`0/200` 全同** | `164 / 169 / 178 / 183 / 200` |
| `6` | **`0/200` 全同** | `198 / 200 / 200 / 200 / 200` |
| `16` | **`0/200` 全同** | **`200/200` 全不同** |
| `64` | **`0/200` 全同** | **`200/200` 全不同** |

⇒ ★★★ **「两种写法无差别」是一个由取值制造的幻觉**：`N` 是 2 的幂时，
除以 `N` 是**精确的指数偏移**，先除后加的每一步都与先加后除**逐位对应**。
用 `N ∈ {2,4,8,64}` 做测试，会得到一条**看起来很干净的结论**
（本机 `16/16` 格、**`3200/3200` 次试验**计数为 `0`），而它在非 2 的幂上**立刻失效**。

★★ **这条纪律的完整形态**（本域的错误模式 `P26`）：**凡判据里出现除法/乘法，
除数必须是「非 2 的幂」或「混合量级」，否则判据可能恒真。**
★ 且**凡报「`N`/`M` 次非零」这类计数，必须同时报三件事**：**除法的位置**、
**累加的实现**、**被归约的轴** —— 换掉其中任何一个，数都会变
（例如换成 `np.sum()` 的 pairwise，或在另一个轴上归约，分布就不同）。

### 1.5 ★ 梯度的字节账：`2Ψ`，而它与 `N` 无关

`Ψ` 的口径是**参数个数**（[C] ZeRO 的 Figure 1 说明文字逐字：
`denotes model size (number of parameters), K`）。按 16 位存一份梯度 ⇒ **`2Ψ` 字节**。

**本机实测**（`B.1`，`Ψ = 64`）：实测 `nbytes` = **`128 B`**，
`specs.ZERO_GRAD_FP16_MULTIPLE × Ψ = 2 × 64 = 128 B` ——
★ 两条**来源不同**的路径（NumPy 的 `size × itemsize` vs [C] 常量的整数倍）给出同一个数。

★★ **每卡持有的梯度字节是常量**：本机扫 `(N, B) ∈ {(2,4), (4,8), (8,8), (16,4)}`，
每卡都恰好 `128 B`（`= 2Ψ`），而**全局 batch** 从 `8` 涨到 `64`（**8 倍**）
⇒ **加卡买到的是更大的 batch，不是更小的梯度**。

## 2. 执行映射

这一节按框架层的切入：**梯度在 `all_reduce` 前后分别是什么、除数在谁手里**。

| 阶段 | 发生了什么 | 依据 / 后果 |
|---|---|---|
| 1. 前向 / 反向 | 每张卡在自己的 `B` 行上算出**本卡**的梯度缓冲（`2Ψ` 字节/卡） | §1.5；★ 缓冲的**初值就是本地梯度**（这一点在 `02` 的「分桶」里会用上） |
| 2. 规约 | 集合通信 **`SUM`** 掉各卡的缓冲 ⇒ 每张卡都拿到**同一个 `Σ`** | [B] NCCL 的逐字语义（§1.1）；★ **这一步没有「除以 `N`」** |
| 3. 缩放 | **每张卡各自**除以 `N`（或等价地：在反向之前把 `loss` 按 `1/N` 缩放） | ★★ 两种实现**等价**（因为结果已经写回每张卡）—— 这就是 `C.1` 的错法**没有报错**的原因 |
| 4. 更新 | `w ← w − lr·g` | ★ 除数错了，等效于**改了学习率**：求和 ⇒ 有效 `lr` 放大 `N` 倍 |

★★ **`SUM` 是默认值这件事是「这个错法不是假想的」的直接证据**（§1.1 的签名逐字）。

## 3. 实现与演进

| 阶段 | 做法 | 它换来了什么、付了什么账 |
|---|---|---|
| **单卡** | 一个 batch 一次算完 | 简单；★ 放不下就没有下一步 |
| **手写 DP** | 每卡算梯度 → `all_reduce`（和）→ **自己除以 `N`** | ★★ **除数成了用户代码的一部分** ⇒ 改 `N` 就要重新想一遍 `lr`（§4 的 ① 就是它） |
| **`loss` 侧缩放** | 前向时把 `loss` 按 `1/N` 缩放，`all_reduce` 照旧求和 | ★ 与上一行**等价**（结果写回每张卡），但**口径藏进了两处代码**（前向的缩放 + 反向的规约） |
| **`DistributedDataParallel` 式封装** | 由框架在梯度缓冲上做 `SUM` 并统一缩放 | ★ 用户不再写除数；★★ 代价是**「`loss` 用 mean 还是 sum」变成了一个必须与框架对齐的口径**（§1.3 的四格） |
| **梯度分桶** | 把 `Ψ` 个参数切成若干桶，**某桶就绪就发** | ★ 这是下一节的正文（`02-gradient-synchronization.md`）：桶换了**通信的发起时机**与**峰值缓冲**，**不换**这里的语义 |

★ 这一步的演进方向不是「更快」，而是**把除数从用户代码里挪走** ——
而挪走之后，一旦 `loss` 的口径与框架的假设不一致，就变成一条**沉默的 `N` 倍偏差**。

## 4. 踩坑与失败模式

**① ★★★ 用「求和」代替「平均」。**
- **现象**：训练能跑、loss 会降，但**更新恰好是 `N` 倍**（本机比值 `4.000000`，
  更新差的 `max|·| = 0.074087`，`N = 4`）。
- **成因**：`all_reduce` 的默认运算就是 `SUM`（§1.1 的签名逐字）⇒ 除 `N` 那一步
  **必须**由 DP 自己补。
- **规避**：把除数写进**一处**（要么全在 `loss` 侧、要么全在规约后），并让 `N = 1`
  **之外**的规模上都跑一次口径判据（★ `N = 1` 上测不出来，见 ③）。

**② ★★ 有按片统计的归一化层时，拿「N 卡平均 == 单卡大 batch」当判据。**
- **现象**：判据在小模型上通过，一换到带 BatchNorm 的模型就**悄悄不成立**。
- **规避**：判据必须**写明前提**「无按片统计的归一化层」。
  ★★ **该等价性与它的失效条件的正文在 `90-training/autograd-and-backward/03-gradient-lifecycle.md`**
  （`:71` / 判据表 `:223-224`），本文只回链。
  ★ **订正（2026-09-23）**：原「成因」那一条是**照抄 `90` 的机制从句**（外加它的实测值
  `4.173e-01`）—— 按 `CONTRIBUTING` §7.2 本文**只放链接、不复制内容**，故**已删**；
  想知道「为什么」，去看那篇的 `:69-72`。

**③ ★★★ 在 `N = 1`（或 `N` 是 2 的幂）上做口径测试。**
- **现象**：测试「通过」，但什么也没检查。
- **成因**：`N = 1` 时「求和」与「平均」**逐位相同**（本机实测：两种写法
  `array_equal == True`）；`N` 是 2 的幂时 `(Σg)/N` 与 `Σ(g/N)` **逐位相同**
  （本机 `16/16` 格 = **`3200/3200` 次试验**）。
- **规避**：口径判据的参数必须**同时**含 `N ≥ 2` 与**非 2 的幂**的 `N`（本机用
  `N ∈ {3,5,7,10,100}`）—— 这是「判据必须写明它的参数取值，以及为什么这组取值能暴露差异」。

**④ ★ 把「`sum` + 卡间除以 `N`」当成正确的写法。**
- **现象**：以为「除以 `N`」就等于「平均」。
- **成因**：全局除数是**两个口径的乘积**（`loss` 的 reduction × 卡间合并），
  `sum + 除 N` 差 `B` 倍（本机比值 `8.000000`，`B = 8`）。
- **规避**：把两个口径**放在一张表里**一起想（§1.3 的四格），并记住
  「分母 = `N·B`」而不是「分母 = `N`」。

**⑤ ★ 把 `N` 倍偏差诊断成「学习率需要调」。**
- **现象**：换了卡数之后 loss 曲线变了，于是去调 `lr`。
- **成因**：除数口径没变、`N` 变了 ⇒ **有效学习率随卡数漂移**；调 `lr` 只是把
  漂移**掩盖**掉，换一次卡数又漂。
- **规避**：先问「除数是几、除在哪一边、`loss` 是什么 reduction 口径」，
  再动 `lr`。§5 的判据表把这三个问题变成可机械判定的行。

## 5. 判据表

> ★ 本表是**闭合的**：每一行都是「**可机械判定的条件 → 结论**」，读完能照着做判断。
> 全部数值为**本机实测值**（`code/01_grad_averaging.py` 的 `make run`，
> `python3 3.13.5` + `numpy 2.4.4`、macOS arm64、**无 GPU/无多卡/无 NCCL**）。
> ★★ **口径**：`N = 4`、每卡 `B = 8`、`Ψ = 64`；差用 `max|·|`；
> **加法的实现 = Python 内建 `sum()`**、**被归约的轴 = 卡（rank）轴**。

| 可机械判定的条件 | 结论 |
|---|---|
| `N` 卡各自平均、再在卡间平均，与单卡在 `N·B` 行上算一次 | **相同**（本机 `max|diff| = 1.665e-16`，`g` 量级 `0.494`）；★★ **前提：无按片统计的归一化层**（该前提的失效条件与正文在 `90-training/autograd-and-backward/03-gradient-lifecycle.md:71`，本文只回链） |
| 把参照实现的输入少喂一张卡的行 | 差 `2.103e-01` ⇒ ★★ **上面的等价性判据【能失败】**（`CONTRIBUTING:235` 的守卫） |
| `loss` 用 `mean`、卡间再除以 `N` | **正确**（比值 `1.000000`） |
| `loss` 用 `mean`、卡间不除 | 比值 **`4.000000`**（= `N`） |
| `loss` 用 `sum`、卡间再除以 `N` | 比值 **`8.000000`**（= `B`）⇒ **「除以 `N`」不等于「平均」** |
| `loss` 用 `sum`、卡间不除 | 比值 **`32.000000`**（= `N·B`） |
| `N` 是 **2 的幂**时的 `(Σg)/N` vs `Σ(g/N)` | **逐位相同**（本机 `16/16` 格 = **`3200/3200` 次试验**，计数为 `0`）⇒ ★★ **这个「无差别」是取值造成的** |
| `N` **非** 2 的幂时（`D ≥ 16`） | **`200/200` 次试验全不同**（`1 ulp` 级）⇒ 结论**不可外推** |
| `N = 1` 时「求和」与「平均」 | **逐位相同** ⇒ 在 `N=1` 上做口径测试**没有信息量** |
| 每卡持有的梯度字节 | `2Ψ`（`Ψ = 64` ⇒ 本机 `128 B`）；★ 与 `N`、与每卡行数**都无关** |
| 全局 batch 随什么变 | 只随 `N × B`（本机 `8 → 64`，**8 倍**）；**每卡字节一个都没变** |
| 梯度的字节账怎么算 | `ZERO_GRAD_FP16_MULTIPLE × Ψ`（[C] ZeRO §5.1，`2Ψ`）；★ 两条来源不同的路径（NumPy 的 `size × itemsize` vs 常量）给出同一个数 |
| 平均给通信量加了什么 | ★ **什么都没加**：除 `N` 可以在每张卡本地做（结果已写回每张卡）⇒ **平均不是通信的代价，是除数的位置问题** |

> ★ 阈值与口径（`code/01_grad_averaging.py` 的 `preflight()` 逐位钉住）：
> `SEED = 20260941`、`N_CARDS = 4`、`PER_CARD_ROWS = 8`、`DIM = 64`、`LR = 0.05`、
> `POW2_N = (2,4,8,64)`、`NONPOW2_N = (3,5,7,10,100)`、`DIVORDER_DIMS = (2,6,16,64)`、
> `DIVORDER_TRIALS = 200`、`TOL_MACHINE = 1e-12`、`TOL_GUARD = 1e-6`、`TOL_RATIO = 1e-9`。

## 6. 小结与延伸

**一句话**：DP 的正确性只在**一个除数**上 —— 它是
「`loss` 的 reduction 口径」×「卡间合并口径」的**乘积**，必须等于 `1/(N·B)`；
而 `all_reduce` 默认求**和**，所以「平均」那一步是**你的责任**。
★ 而这条判据有**两个会骗人的参数取值**：`N = 1`（两种写法恒等）与
`N` 是 2 的幂（除法是精确的指数偏移）。

- **梯度怎么与参数对齐（梯度分桶、峰值缓冲、通信发起时机）** ⇒ `100-parallelism/data-parallel/02-gradient-synchronization.md`。
- **DP 的冗余账（`N` 个副本用 `N × 16Ψ` 存 `16Ψ` 的状态）** ⇒ `100-parallelism/data-parallel/03-redundancy-and-limits.md`。
- **梯度累积与 batch 放大的等价性（含归一化层时的失效）** ⇒
  `90-training/autograd-and-backward/03-gradient-lifecycle.md:71,223-224`（**正文在彼**）。
- **`16Ψ` 的组成与字节分解** ⇒ `90-training/checkpointing/01-state-and-bytes.md`（**正文在彼**，本文只消费 `2Ψ` 这一项）。
- **梯度规约的算法（ring allreduce / `V(N) = 2(N−1)/N × S` 与它的饱和）** ⇒
  `110-communication/collectives`）、`10-foundations/performance-models/02-amdahl-and-scaling.md:209`（**正文在彼，本文一行都不写**）。
- **扩展效率模型（Amdahl / Gustafson / `N₅₀`）** ⇒ `10-foundations/performance-models/02-amdahl-and-scaling.md`。
- ★★ **一处必须写明的让步**：本文的判据**全部**是「本机玩具规模的真跑」
  （`python3 3.13.5` + `numpy 2.4.4`、**无 GPU、无多卡、无 NCCL、无真实网络**）
  ⇒ **不是 NCCL，也不是真机**；**不得支撑关于真实集群性能的结论**。

## 编译与运行

```bash
cd code
make run                    # 三个编号程序依次跑；本程序是 01_grad_averaging.py
python3 01_grad_averaging.py   # 只跑本程序
```

- 配套程序：**`code/01_grad_averaging.py`**（编号与外键严格对应）。
- 依赖：`numpy`（`requirements.txt` 里**只有**它）。**不需要网络、不需要 GPU。**
- 失败标记：`%% FAIL: data-parallel`（本主题自定义，**不跨主题照抄**）；`D.1`/`D.2` 守着它。
- 程序的输出分四段：`A` 语义（`A.1` 平均 vs 大 batch / `A.1g` 守卫 / `A.2` 除法的位置 /
  `A.3` 口径配对四格）、`B` 代价模型（`2Ψ` 字节账 / 每卡常量而全局 batch 变）、
  `C` 边界与反例（求和代替平均差 `N` 倍 / `N=1` 的适用边界）、
  `D` 标记自检 + 口径阈值钉住 + specs 常量钉住 + **独立性证据**。
- ★★ **口径/阈值/标记自检一律在 `preflight()` 里、跑在任何计算之前**（本库的 **P23**）。
- ★★★ **本程序每条 A 类都带守卫**：`A.1g` 把参照的输入故意改错、`A.3` 的四格给出
  `{1, N, B, N·B}` 这组**可判定的**比值、`A.2` 同时给出「全 0」与「全非 0」两列
  —— 这是本域对抗「模拟器同时是被测对象」（`P12`）的标准落法。

## 参考

**本机实测值 / 推导值**

- `code/01_grad_averaging.py` 的 `make run` 输出：`python3 3.13.5` + `numpy 2.4.4`，
  macOS arm64，**无 GPU**。口径：`N = 4`、`B = 8`、`Ψ = 64`、`DIVORDER_TRIALS = 200`；
  「非零计数」的口径三件套 = **除法的位置**（先加后除 / 先除后加）·
  **累加的实现**（Python 内建 `sum()`，顺序累加）· **被归约的轴**（卡 `rank` 轴）。
  **全部是推导值 / 本机实测值，不是任何实现的实测。**
- ★ 与计划 §3（Chunk 1 表）的对应：**判据 A**（`N` 卡平均 vs 单卡大 batch，前提必须写出、
  且回链 `90` 的 `03:223-224`）、**B**（梯度字节账 `2Ψ`）、**C**（求和代替平均 ⇒ 差 `N` 倍）、
  ★ **另外必做**（`(Σg)/N` vs `Σ(g/N)` 的规模依赖）。四条**全部落在程序里**。
- ★★ **本文自己新增的一组判据**（不在计划表里）：`A.1g`（把参照改坏 ⇒ 必须失败的守卫）、
  `A.3`（`loss` reduction × 卡间除数的四格）、`B.2`（每卡常量而全局 batch 变）、
  `C.2`（`N=1` 的适用边界）。★ 它们的必要性是**测出来的**：`A.3` 与 `C.2` 各把一句
  **容易被说错的话**变成了一个数。

**外部来源**

- **[B]** **NVIDIA NCCL 2.31.2 用户指南**（产品官方文档，**不是规范**），
  `https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html`，
  抓取日 **2026-09-23**（`curl -sSL`，HTTP 200），**原始字节 86584**，
  `sha256[:16] = d7d2b6cff49d9f79`（★ **本页连抓 3 次，字节数与摘要完全一致** ⇒ 可复核）。
  版本号的载体：页面 `<title>` 逐字 `NCCL 2.31.2 documentation`（**原始字节 1 次**，
  且页面上**只有一个版本串**：更短的 `NCCL 2.31` 也计 1 次，它只是前者在 title 里的
  **前缀子串** —— ★ 订正 2026-09-23：本文曾写「同页侧边栏另有 `NCCL 2.31`」，**实测为假**）。
  逐字：`The AllReduce operation performs reductions on data (for example, sum, min, max)
  across devices and stores the result in the receive buffer of every rank.`
  （原始字节 1 次）与 `each rank will provide an array in of N values, and receive
  identical results in array out of N values,`（原始字节 1 次）。
  ★★ **一处载体口径**：同一节的 `In a <em>sum</em> allreduce operation between
  <em>k</em> ranks,` **跨标记** ⇒ 我们只取**不含斜体标记**的中段。
- **[B]** **PyTorch 2.14** 的 `torch.distributed` 页，
  `https://docs.pytorch.org/docs/2.14/distributed.html`，抓取 **2026-09-23**，
  **1509715 B**，`sha256[:16] = 1a4a7e398b5a7744`。
  逐字：`torch.distributed.all_reduce(tensor: Tensor, op: _ReduceOp = ReduceOp.SUM, …)`
  （`ReduceOp.SUM` 这一串整页计 **16** 次）与
  `Reduces the tensor data across all machines in a way that all get the final result.`
  （整页计 **2** 次）。
  ★★ **一处量法口径**：这一页的**签名是重标记的**（每个 token 各自一个 `<span>`）
  ⇒ **签名整体在原始字节上不可数**；`ReduceOp.SUM` 之所以可数，是因为它在**同一个
  `<span class="pre">` 里是连续的**。
- **[C]** ZeRO（`arXiv:1910.02054`，**不确认同行评审** —— abs 页 `Comments:` 字段
  **实测不存在**），§5.1 与 Figure 1 的说明文字。逐字（本主题**复核过的**三条）：
  `denotes model size (number of parameters), K`、
  `In total, this results in 2` … `bytes of memory requirement`、
  `Mixed-precision Adam has K = 12`。
  ★★ **本主题这一轮没有重新抓取**：这三条**与 `90-training/checkpointing/code/specs.py`
  的对应常量同值**（`# 同值位置：` 互指）⇒ 本文**不声称**自己复核了抓取，只声称同值。

**上游契约与边界声明**

- ★★ **梯度累积与 batch 放大的等价性**（含归一化层时的失效）的正文在
  `90-training/autograd-and-backward/03-gradient-lifecycle.md:71,223-224`；
  本文只写**「无归一化层」这一支**，并把它当作 `A.1` 的**前提**写出来。
- ★★ **`16Ψ` 的组成与字节分解**的正文在 `90-training/checkpointing/01-state-and-bytes.md`
  （§1.3 + 判据表 + `code/01_state_bytes.py:310` 的 `A.3`）；本文只消费 `2Ψ` 这一项。
- **禁区**：梯度规约的算法（ring allreduce / `V(N)`）→ `110-communication/collectives`
  与 `10-foundations/performance-models/02-amdahl-and-scaling.md:209`；
  扩展效率模型 → 同上（**全篇 429 行**，`raw.count('\n')` 口径）；通信-计算重叠 →
  `110-communication` 的 `overlap`。
  本文**一行都不重推**。
- ★★ **术语**：本文的 `all_reduce` 一律指**求和型集合通信**（`ReduceOp.SUM`），
  与 `00-meta/glossary.md` 已有的 `all-reduce` 条**同指**（★ 是「同一件事」而不是「同名不同指」，
  故本文不另立词条）。

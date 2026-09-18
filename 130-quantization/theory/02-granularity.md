---
title: 粒度：per-token 只治「行」，per-channel 只治「列」
slug: quantization-granularity
domain: 130-quantization
topic: theory
level: 2
difficulty: intermediate
prerequisites:
  - 130-quantization/theory/01-error-bounds
tags: [per-token, per-channel, per-group, granularity, kivi, q-bert]
related:
  - 130-quantization/theory/03-outliers-and-propagation
  - 130-quantization/kv-cache-quantization/02-kivi-asymmetry
  - 130-quantization/numerical-formats/04-integer-formats
code:
  - code/02_granularity.py
status: stable
updated: 2026-09-18
---

# 粒度：per-token 只治「行」，per-channel 只治「列」

> scale 沿哪一维取，就只治那一维的异质性。张量按 `[tokens, features]` 排布时，**「每行一个 scale」= per-token、「每列一个 scale」= per-channel**——这两个标签写反不会报错，只会推出与 KIVI 相反的结论。

## 0. 背景与动机

先看一个具体的账。取一张 `(4096, 4096)` 的激活张量，对称 8 位 RTN，三种粒度：

| 场景 | per-tensor | per-token | per-channel |
|---|---|---|---|
| iid 高斯 | 38.00 | 41.25 | 41.24 |
| **列异质**（每 channel 的尺度跨 1e-3…1e0） | 30.04 | 34.05 | **41.23** |
| **7/4096 列 ×100**（模拟 outlier） | 13.61 | 20.31 | **41.20** |

（判据见 §5；这是本机实测。）

同一张张量、同一个 8 位、同一个 RTN，**per-channel 在最后一行比 per-token 多 20.89 dB**——而它们的 scale 个数**完全一样**（都是 4096 个，都是 0.25 bit/元素）。**差别只在「沿哪一维归约」。**

这就是本节的全部内容，但它有一个必须先钉死的零件：**「per-token」和「per-channel」这两个词，在没说清张量布局时是没有意义的。** 计划基线里的第一版就把这两个标签写反了——**写反不会有任何报错**，只会让结论变成「K 应该用 per-token」，与 KIVI 正好相反。

## 1. 原理

### 第一步：先把轴约定写死

本文与配套程序一律用 **`[tokens, features]`** 的排布（transformers 里激活张量的通行布局）：

```
张量 A，形状 [T, F]
    A[i, j]：第 i 个 token 的第 j 个 feature（= channel）

「每行一个 scale」  沿 axis 1 归约出 T 个 scale   →  per-token
「每列一个 scale」  沿 axis 0 归约出 F 个 scale   →  per-channel
```

注意：**`torch.quantize_per_channel` 的 `axis` 参数不按这个直觉走。** 它的语义是「scale 的长度等于 `input.size(axis)`，且**沿 axis 这一维索引取 scale**」——所以「每列一个 scale」（长度 = `size(1) = F`）对应的是 `axis=1`，**不是** `axis=0`。这件事在 §4.3 展开，它是本文里最容易静默出错的一处。

**这不是术语洁癖。** 换一个布局（比如把权重排成 `[out_features, in_features]`），「per-channel」指的仍然是「每个输出通道一个 scale」，但它对应的归约轴会变。**判据是「沿哪一维归约」，不是那个词本身。**

### 第二步：四种粒度的定义与它们各自治什么

给定 b 个比特，量化器只有一个自由度：**这一组元素共享一个 Δ，这一组有多大、怎么切。** 四种粒度就是四种切法：

| 粒度 | 一组的大小 | scale 个数（`T×F` 张量） | scale 元数据（fp32） |
|---|---|---|---|
| per-tensor | 整张量 | 1 | ≈ 0 |
| per-token | 一**行**（一个 token 的全部 feature） | `T` | `32/F` bit/元素 |
| per-channel | 一**列**（一个 feature 维的全部 token） | `F` | `32/T` bit/元素 |
| per-group(G) | 一行内连续 `G` 个 feature | `T·F/G` | `32/G` bit/元素 |

**per-token 与 per-channel 的 scale 个数相同**（在方阵上都是 `T = F = 4096`），所以在「花多少比特存 scale」这件事上**它们的花费一模一样**——差别纯粹在**治哪一维**。

### 第三步：为什么「每行」治行异质、「每列」治列异质

这是本文的核心，可以严格说清楚。

设第 `i` 行有一个整体尺度因子 `r_i`（`A[i,j] = r_i · c[i,j]`，`c` 是 iid）。用 **每行一个 scale**：

```
s_i = max_j |A[i,j]| / QMAX = r_i · max_j |c[i,j]| / QMAX  ∝  r_i
```

**scale 精确地跟着 `r_i` 走**，于是归一化之后所有行的有效分布完全一致，行间的异质性被完全吸收。

换 **每列一个 scale**：

```
s_j = max_i |A[i,j]| / QMAX = max_i |r_i · c[i,j]| / QMAX  ≈  (max_i r_i) · max_i|c[i,j]| / QMAX
```

**`s_j` 由全局最大的那个 `r_i` 决定**，所有列被同一个大尺度撑开，行异质性一点没吸收。

把「行」与「列」对调，结论对调。这就是 §0 表里那两行对照的全部机制。用一句话概括：

> **scale 沿哪一维归约，就吸收那一维的尺度差异。**

所以 **per-token 与 per-channel 不可互替**：一个吸收行，一个吸收列。

### 第四步：outlier 在哪一维，粒度就给哪一维

把第三步推到极限：某几**列**的值比其它列大 100 倍。

- **每行一个 scale**：这一行里只要有那个 outlier，`s_i` 就被抬高 100 倍 → **这一行里所有 feature 都损失 6.6 bit 的分辨率**。而 outlier 在**每一个 token 上都出现**（这是 outlier 的持久性，[03](03-outliers-and-propagation.md) §1 会展开），所以**每一行都被毁**。
- **每列一个 scale**：只有那几个 outlier 列的 `s_j` 被抬高，**其余 4089 列完全不受影响**。

本机实测（§5 表 2，7/4096 列随机位置 ×100）：

| per-tensor | per-token | per-channel | per-group(128) |
|---|---|---|---|
| 13.61 | 20.31 | **41.20** | 32.30 |

**per-channel 比 per-token 多 20.89 dB。** 而两列之间只差一个数字——`A.abs().max(dim=0)` 还是 `A.abs().max(dim=1)`。

> **这就是 KIVI 的核心洞察的理论根据。** KIVI 的实测是：K cache 在固定的少数 channel 上有大幅值 outlier，所以 **K 用 per-channel**；Table 2（Llama-2-13B）给的重建误差是 per-token **13.67** / per-channel **4.55**。本文的合成实验与它同向，**但两者不构成互相背书**——本文测的是构造张量上的机制，KIVI 测的是真实模型。KIVI 的完整内容见 [kv-cache-quantization](../kv-cache-quantization/)。
>
> ⚠ **KIVI 的另一半本文复现不了**：它说 **V 必须用 per-token**，理由是 attention 分数高度稀疏、输出是少数重要 token 的 value 的加权组合——**这是关于 attention 结构的论证，不是关于 outlier 在哪一维的论证。本文的粒度表里没有任何东西能推出它。**

### 第五步：per-group 的切分维度必须另写

per-group 是唯一一个「组大小 G」和「沿哪一维切」**两个参数都要说**的粒度。各论文并不统一：

| 来源 | 切分维度 |
|---|---|
| Q-BERT | 沿**输出维** |
| GPTQ / nuQmm | 沿**输入维** |
| Dettmers 等 | 把张量**拉平** |

（出处见 `specs.py` 的 `PER_GROUP_SPLIT_DIM`，来自计划基线，本文实现者未逐条重读原文。）

**这一维不写，group size 就没有可比性。** 本机实测（§C2，同一张 outlier 张量、同样是 G = 128）：

| 切分维度 | SQNR |
|---|---|
| 沿 feature（GPTQ 口径） | 32.30 dB |
| 沿 token（KIVI 口径） | **43.81 dB** |
| 拉平（Dettmers 口径） | 32.30 dB |

**同样的 G、同样的张量，只换切分维度差 11.51 dB。**

### 第六步：粒度这笔账花掉多少比特

per-group(128) 的 scale 开销是 `32/128 = 0.25` bit/元素——**相对 4-bit 载荷是 +6.25%**。G = 32 时是 1.0 bit/元素，**相对 4-bit 载荷 +25%**。

| G | 32-bit scale | 16-bit scale |
|---|---|---|
| 4096 | 0.0078 | 0.0039 |
| 128 | **0.2500** | 0.1250 |
| 32 | **1.0000** | 0.5000 |
| 16 | 2.0000 | 1.0000 |

**「花更多比特存 scale，换更细的粒度」——这就是 §0.8 三角上「粒度 vs scale 开销」那条边。** 本机实测这笔交换的回报（§B2，代价都是 0.25 bit/元素）：

| 场景 | per-tensor → per-group(128) |
|---|---|
| iid 高斯 | +5.79 dB |
| 拉普拉斯 | +9.85 dB |
| 行异质 | +14.30 dB |
| 列异质 | +8.20 dB |
| **7/4096 列 ×100** | **+18.69 dB** |

**同样的 0.25 bit/元素，在不同数据上买到的 dB 差 3 倍以上。** 回报取决于数据，不取决于粒度本身。

## 2. 执行映射

**（1）粒度决定 `amax` 沿哪一条轴归约，也就是决定谁给谁定 Δ。** 在 `torch.ao.quantization` 里这就是 `observer` 的 `ch_axis`：`MinMaxObserver` 的 `ch_axis` 非负时走逐通道分支。前一篇 `01-error-bounds` 的 Δ/2 界对四种粒度**都成立**，变的只是 Δ 的宽度。

**（2）真实的 kernel 里，粒度是可实现的自由度，但代价不同。** 权重侧的 per-channel scale 可以折进 GEMM 的 epilogue（作为行向量广播到累加器 tile 的列上）；而**激活侧的 per-channel 会插进 GEMM 的内维，硬件不可行**——这就是为什么权重能做 per-channel 而激活只能退到 per-token。这条的完整推导在 [quantized-kernels](../quantized-kernels/)，本文只记结论。

**（3）本机实测的可实现性检查：库路径与手写 RTN 逐位吻合。** 两条互不依赖的代码路径（ATen 的 quantizer vs 本程序的 `round/clip`）在 `(4096,4096)` 上给出**同一个 SQNR**（差 < 0.001 dB，§A4）：

| 粒度 | 库路径 | 手写 RTN |
|---|---|---|
| per-tensor（`quantize_per_tensor`） | 38.00 dB | 38.00 dB |
| per-token（`quantize_per_channel`, `axis=0`） | 41.25 dB | 41.25 dB |
| per-channel（`quantize_per_channel`, `axis=1`） | 41.24 dB | 41.24 dB |

这说明「粒度」不是纸面概念——它在真 torch 里逐位实现出来了，且轴语义与我们写死的约定一致。

## 3. 实现与演进

**第一步：per-tensor。** 最早的一批量化推理（GEMMLOWP 一类的定点管线）就是整张量一个 scale。它的失败模式在 §0 的表里一眼可见：**outlier 场景掉到 13.61 dB**——因为 `amax` 是整张量的极值，一个 100 倍的值把所有人的格点撑开 100 倍。

**第二步：per-channel。** 卷积网络时代先用在**权重**上：每个输出通道一个 scale。这一举解决了权重的通道间尺度差异（BatchNorm 折叠之后尤其明显）。

**第三步：per-token / per-group。** 到了 Transformer，激活成了瓶颈。几件事同时发生：

- **per-token** 用来吸收**token 之间**的尺度差异（不同 token 的激活整体幅值可以差一个数量级）；
- **per-group** 用细粒度换 scale 开销，`G = 128` 成了一个事实上的默认；
- **权重侧的 per-channel** 保留。

**第四步：把粒度给到 outlier 所在的那一维。** KIVI 的贡献不是发明 per-channel，而是**指出 K 的 outlier 在 channel 维、V 不在**，于是同一个系统里 K 与 V 用**不同的**粒度（K per-channel / V per-token）。这需要两个新机制：K 攒够 `R` 个 token 再量化一次（绕开 per-channel 无法流式）、V 逐 token 在线算 scale。**本机复现不了这一层**（无 GPU、无 KV cache），只能给出合成张量上的同向结果。

**一个关于「group size = 128」的订正。** 这个数字常被引给 Q-BERT，但 **Q-BERT 那张表的列名是 `"# groups"`（组数）**，取值 1 / 12 / 128 / 768。对 768 宽的矩阵，**128 组 ≈ group size 6**——把它当成「group size = 128」是 **21 倍**的作用域错配。

| # groups | 1 | 12 | 128 | 768 |
|---|---|---|---|---|
| SST-2 | 85.67 | 92.31 | 92.66 | 92.78 |

（基线 93.00；出处 **arXiv:1909.05840**——⚠ **不是** arXiv:1905.03696，那是 HAWQ，两串编号极易搞错。128 → 768 的增量是 **0.12**，所以「≤0.1%」这个说法对 SST-2 **不成立**。）

**正确的引用方式**是：「Q-BERT 报告**组数**超过 128 后收益饱和」。**「group size = 128」需要另找一手出处**——本文没有找到，因此不给。

## 4. 踩坑与失败模式

### 4.1 轴标签写反（不会有任何报错）

**现象**：文档里 `per-token` 与 `per-channel` 两个标签互换。表格照印、数字照对、公式照推，**没有任何一处会报错**。

**成因**：这两个词是**相对**的，指代对象取决于张量布局。计划基线的第一版就把它们写反了，而**写反之后推出的结论是「K 应该用 per-token」——与 KIVI 正好相反**。本机把这个「写反会得到什么」跑了出来（§C1）：outlier 场景下 per-token 比 per-channel 低 **20.89 dB**，所以写反版本会给出一个自信而完全错误的工程建议。

**规避**：三条一起做。① **在文档里把轴约定写成一行公式**（`[tokens, features]`；每行 = per-token；每列 = per-channel），并让它出现在第一次使用这两个词之前；② **把「标签 ↔ 归约轴」的对应本身写成一条可失败的断言**——本机程序 §A1 就是这么做的：先测出「行异质数据上每行一个 scale 赢 7.10 dB」，再断言「per-token 治行异质」为真。**把标签换过来，这条断言就失败。** ③ 引用别人的结论时，凡出现这两个词，回头核对它的布局约定。

### 4.2 per-group 不写切分维度

**现象**：论文 A 说 group size 128 好，论文 B 说 group size 128 差，两边都有实验。

**成因**：**「group size」不是完整规格。** 沿输入维切、沿输出维切、把张量拉平——三者在同一个张量上给出的 SQNR 本机实测差 **11.51 dB**（§C2）。

**规避**：引用任何 group 相关数字时，连同切分维度一起引。若原文没写，**在不确定性上标注**，不要补一个看起来合理的维度。

### 4.3 库 API 的 `axis` 语义与直觉相反，而且方阵上不报错

**现象**：调用 `torch.quantize_per_channel(x, scales, zps, axis=0, ...)`，以为「axis=0 ⇒ 每列一个 scale」。在方阵上它**静默地**把列 j 的 scale 用在了行 j 上。

**成因**：库的语义是「**scale 的长度等于 `input.size(axis)`，且沿 axis 这一维索引取 scale**」。所以 `axis=0` 意味着**每行一个 scale**。本机实测（§A4/§C5）：

- 非方阵 `[64, 7]` 上传「长度 7 的 scale + `axis=0`」会**响亮地失败**：`length of scales must equal to channel, expected 64 got, 7`；
- **方阵上两个方向长度相同，错的那个照样被接受**：同一份列 scale，`axis=0` 给 **2.35 dB**，`axis=1` 给 **41.23 dB**，差 **38.88 dB**。

**而那个 2.35 dB 既不是每列（41.23）也不是每行（34.05）**——它把列 j 的 scale 用在了行 j 上，是一个**没有对应的量**。

**规避**：① 在**非方阵**上先验证一次轴语义；② 断言库路径与手写路径给出同一个数（本机 §A4 就是这么做的，per-token 与 per-channel 两个方向都比对了，差 < 0.001 dB）；③ 记住这条语义与「沿哪一维归约」的关系：**`axis=k` 的 scale 长度是 `size(k)`，等价于「每 `size(k)` 个…」不对——准确说是把第 k 维当作被量化的通道**。

### 4.4 逐格引用 per-tensor 的数字

**现象**：文档写「per-tensor 在 iid 高斯上是 37.44 dB」，读者复现出 38.0 或 36.6，于是怀疑实现有问题。

**成因**：per-tensor 的 scale 由**整张量的样本极值**决定，而极值是小样本统计量。本机实测（`01-error-bounds` §B2，2×10⁵ 样本、100 个种子）：`N(0,1)` 的 SQNR 极差 **2.31 dB**、Laplace 极差 **5.28 dB**。

**规避**：引用 per-tensor 的格子时**给宽容差并说明它对种子敏感**，或者干脆只引**定性结论**（「per-tensor 在每个场景里都是最差的那一列」——本机实测五个场景全部满足，§A3）。**per-token / per-channel / per-group 那三列的复现偏差 ≤ 0.35 dB**，可以逐格引。

## 编译与运行

```bash
cd code
make run                 # 自动探测装了 torch ≥ 2.8 的解释器，依次跑 01–03
make 02_granularity      # 只跑本文对应的程序
make verify              # 默认 python3，跑 verify.py（**不 import torch**）
```

配套程序是 **`code/02_granularity.py`**（Python ≥ 3.9 + torch，不需要 GPU）。它跑 `(4096, 4096)`、种子 20260918、对称 8-bit RTN、`QMAX = 127`、`G = 128`。

**配套程序与 `verify.py` 的分工**（CONTRIBUTING §2 第三种形态）——本文是「**同一条结论两边都要给**」的示范篇：

| | 规模 | 谁在跑 | 产出档位 |
|---|---|---|---|
| `02_granularity.py` | `(4096, 4096)`，种子 20260918 | 真 torch | **本机实测值** |
| `verify.py` §A8/§A9 | `(256, 256)`，标准库 `random` | 默认 python3，**不 import torch** | **推导** |

**两者的数字不同**（规模不同），**只有定性结构必须一致**：行异质上「每行一个 scale」赢、列异质上「每列一个 scale」赢、列 outlier 下 per-channel 领先。⚠ **那个领先的幅度依赖张量宽度**（256 宽上是 10.30 dB，4096 宽上是 20.89 dB，因为 7 列 outlier 在两种宽度下占的比例不同）——`verify.py` 里的阈值因此写成 8 dB 而不是 20 dB。**两档不得互相背书。**

## 5. 实测数据

**测量条件**：CPU 型号 **Apple M1 Pro（10 核）**；macOS；`python3` **3.12.9**；`torch` **2.11.0**（表 1–3 的每一格在 **2.6.0 / 2.7.1 / 2.8.0 / 2.11.0 / 2.13.0** 上**逐格一致**）；**CPU-only，无 GPU 数字**。
**未覆盖**：2.9.x / 2.10.x / 2.12.x。

**表 1：四种粒度 × 五种场景的 SQNR**（`code/02_granularity.py` §A2；`(4096,4096)`、对称 8-bit RTN、`QMAX = 127`、种子 20260918）

| 场景 | per-tensor | per-token（每行） | per-channel（每列） | per-group(128) |
|---|---|---|---|---|
| iid 高斯 | 38.00 | 41.25 | 41.24 | 43.79 |
| iid 拉普拉斯 | 31.13 | 36.83 | 36.81 | 40.97 |
| **行异质**（每 token 尺度跨 1e-3..1e0） | 29.48 | **41.16** | 34.06 | 43.78 |
| **列异质**（每 channel 尺度跨 1e-3..1e0） | 30.04 | 34.05 | **41.23** | 38.24 |
| **7/4096 列（随机位置）×100** | 13.61 | 20.31 | **41.20** | 32.30 |

**判据（缺一不可）**：张量按 `[tokens, features]` 排布；scale = 该归约域内 `max|x| / 127`；量化 = `clip(round(x/s), −127, 127)·s`（**restricted range**）；SQNR = `10log10(Σx²/Σ(x−x̂)²)`；per-group 的 `G = 128` 且**切分沿 feature（最后一维）**；行/列异质的尺度因子取 `10^U(−3,0)`；outlier 场景是**随机位置的 7 列 ×100**。

> ⚠ **两个必须一起写的细节：**
> 1. **outlier 列的位置是随机的**。若取「前 7 列」，per-group(128) 那一格会从 32.30 变成 **35.4 dB**（差 3.1 dB，而其余格子跨种子只差 0.01 dB）——**这是一个定义问题，不是噪声**。
> 2. **per-tensor 那一列是随机变量**（见 §4.4）。计划基线给的是 37.44 / 31.09 / 29.16 / 29.46 / 13.99，与上表差 0.2–0.6 dB，原因就是种子不同。**本文只引「per-tensor 最差」这个定性结论，不逐格引这三个数。**

**表 2：列 outlier 下的粒度收益**（§A5/§C4）

| | per-tensor | per-token | per-channel |
|---|---|---|---|
| 列 outlier | 13.61 | 20.31 | **41.20** |
| 相对 per-tensor | — | +6.70 | **+27.59** |
| **per-channel − per-token** | | | **+20.89 dB** |

**表 3：per-group 的切分维度**（§C2；同一张 outlier 张量、`G = 128`）

| 切分维度 | SQNR |
|---|---|
| 沿 feature（GPTQ 口径） | 32.30 dB |
| 沿 token（KIVI 口径） | **43.81 dB** |
| 拉平（Dettmers 口径） | 32.30 dB |

**表 4：轴约定本身**（§A1；判据：行异质/列异质的构造见 §5 表 1 的判据栏）

| 数据 | 每行一个 scale | 每列一个 scale |
|---|---|---|
| 行异质 | **41.16 dB** | 34.06 dB |
| 列异质 | 34.05 dB | **41.23 dB** |

**行异质上「每行」赢 7.10 dB；列异质上「每列」赢 7.18 dB。** 这两条是「per-token 治行、per-channel 治列」的直接实测。

**表 5：库路径与手写 RTN 逐位吻合**（§A4；`(4096,4096)`）

| 粒度 | 库路径 | 手写 RTN | 差 |
|---|---|---|---|
| per-tensor（`quantize_per_tensor`） | 38.00 | 38.00 | −0.000 |
| per-token（`quantize_per_channel(axis=0)`） | 41.25 | 41.25 | −0.000 |
| per-channel（`quantize_per_channel(axis=1)`） | 41.24 | 41.24 | −0.000 |

**表 6：`axis` 写错时的静默失败**（§C5；同一份**列** scale，方阵 `(4096,4096)`）

| 调用 | SQNR | 它实际是什么 |
|---|---|---|
| `axis=1` | **41.23 dB** | 每列一个 scale（正确） |
| `axis=0` | **2.35 dB** | 把列 j 的 scale 用在行 j 上（**不对应任何粒度**） |

非方阵 `[64, 7]` 上同样的错会抛 `RuntimeError: length of scales must equal to channel, expected 64 got, 7`。

## 6. 小结与延伸

三条可复用的结论：

1. **粒度是「沿哪一维归约」，不是两个可以互换的标签。** `[tokens, features]` 布局下，每行 = per-token、每列 = per-channel。写反不会报错，只会推出与 KIVI 相反的结论——**所以「标签 ↔ 归约轴」的对应必须被写成一条可失败的断言**（本文 §4.1 的做法）。
2. **per-token 只治行异质，per-channel 只治列异质，二者不可互替。** 而且**它们的 scale 开销完全一样**（方阵上都是 `T` 个 scale），所以这是一个纯粹的「花在哪一维」的选择。outlier 在列上时 per-channel 多 **20.89 dB**。
3. **per-group 的「group size」不是完整规格，必须连切分维度一起说。** 同一个 `G = 128` 换维度差 **11.51 dB**。而流传很广的「group size = 128」出处（Q-BERT）其实是 **「# groups」= 128**——组数，不是组大小；对 768 宽的矩阵，它相当于 group size 6。

**这条主线在 §0.8 的三角上是哪条边**：**「粒度 vs scale 开销」**。给定 b 个比特，多花 `32/G − 32/(T·F)` 的元数据比特，换回若干 dB——而本机实测这笔交换的回报**差 3 倍以上**（iid 高斯 +5.79 dB vs outlier 场景 +18.69 dB，代价都是 0.25 bit/元素）。**回报取决于数据，不取决于粒度本身。**

延伸有三条。**往下**：outlier 为什么会「持久」——本文 §1 第四步用到了它，但没论证它（[03-outliers-and-propagation](03-outliers-and-propagation.md) §1）。**往旁边**：`per-token` 与 `per-channel` 的取舍在 KV cache 上有一个非常干净的真实案例（K 用 per-channel、V 用 per-token），见 [kv-cache-quantization/02-kivi-asymmetry](../kv-cache-quantization/02-kivi-asymmetry.md)。**往硬件**：per-channel 的激活 scale 会插进 GEMM 内维——为什么硬件做不了、以及权重侧怎么做进 epilogue，见 [quantized-kernels](../quantized-kernels/)（本文不复述，此处回链）。

## 参考

- 四种粒度的定义与「per-group 切分维度不统一」（Q-BERT 沿输出维 / GPTQ、nuQmm 沿输入维 / Dettmers 等拉平）：**来自计划 §0.7.B 的事实基线，本文实现者未逐条重读原文**。
- **Q-BERT**, arXiv:**1909.05840**（⚠ **不是** arXiv:1905.03696，那是 HAWQ）—— group-wise 消融表，列名 **`"# groups"`**，取值 1 / 12 / 128 / 768，SST-2 85.67 / 92.31 / 92.66 / 92.78（基线 93.00）。**2026-09-18 二次检索确认了这组数与「组数超过 128 后收益饱和」的表述。**
  > ⚠ **该表的表号在来源间不一致**：本仓库检索到的镜像页标为 **Table 2**，而同一批检索摘要称它是 **Table 3**。本文因此**不写死表号**。
- **SmoothQuant**, arXiv:2211.10438, Table 1（OPT 系列 activation 量化的平均精度：6.7B `64.9 → 39.9 → 42.5 → 64.8`；175B `71.6 → 32.3 → 31.7 → 71.4`）。**来自计划基线，本文实现者未读原文**；计划记录该表在 `pdftotext -layout` 提取下**行标有错位**，对齐由研究员按语义判定。**本主题因此不引这组数作为本文结论的依据，只在此处列出出处。**
- KIVI 的结论（K per-channel / V per-token）见本仓库 [kv-cache-quantization](../kv-cache-quantization/)——**本文只引用它的方向，不复述它的数字**（CONTRIBUTING §7：一个知识点只有一处正文）。
- 本仓库 [01-error-bounds](01-error-bounds.md) —— Δ/2 界与 Δ²/12（本文的每种粒度都用同一套界，变的只是 Δ 的宽度）。
- 本仓库 [03-outliers-and-propagation](03-outliers-and-propagation.md) —— outlier 的持久性与 massive activations 的区分。
- 本仓库 [numerical-formats/04-integer-formats](../numerical-formats/04-integer-formats.md) —— 对称/非对称、`127.5` 这个除数、零点取整的 `|δ| ≤ s/2`（本文全篇用对称量化，不对称的那条边在那篇）。

---
title: 校准：五种选 amax 的方法，与 PyTorch 的直方图观察器到底在最小化什么
slug: calibration
domain: 130-model-optimization
topic: quantization
level: 2
difficulty: intermediate
prerequisites:
  - 130-model-optimization/quantization/06-granularity
  - 130-model-optimization/quantization/04-integer-formats
tags: [calibration, histogram-observer, kl-divergence, ema, reduce-range, percentile]
related:
  - 130-model-optimization/quantization/09-gptq
  - 130-model-optimization/quantization/10-awq-and-smoothquant
  - 130-model-optimization/quantization/07-outliers-and-propagation
code:
  - code/08_calibration.py
status: stable
updated: 2026-10-13
---

# 校准：五种选 amax 的方法，与 PyTorch 的直方图观察器到底在最小化什么

> 训练后量化只需一次前向校准，但「把 `amax` 定在哪」有五种做法，而**它们的差别不是实现细节、是目标函数**：MinMax 取极值；EMA 把极值做移动平均；`HistogramObserver` 最小化 **L2（平方）误差**；TensorRT 的 entropy 校准器最小化 **KL 散度**。后两者**方向相反**——本机实测同一份数据上 `HistogramObserver` 选出的 `amax` 是 TensorRT KL 的 **5.65 倍**。**把 PyTorch 的 `HistogramObserver` 说成「KL 校准」，是本主题最常见的错误。**

## 0. 背景与动机

拿一份具体的对照。同一个 8 位量化器、同一批数据（200000 点 `N(0,1)`，随机 20 个点乘 40），只在「把 `amax` 定在哪」这一步上换方法，得到的覆盖量程与代价是这样的（**本机实测**，判据见 §5）：

| 选 amax 的方法 | 选出的 `amax` | 相对 **MinMax 的 `amax`（=104.440）** 裁掉 | 它自称在最小化的东西 |
|---|---|---|---|
| MinMax | 104.440 | 0% | 无（不裁剪） |
| `HistogramObserver`（PyTorch） | 71.599 | 31.4% | **L2 / 平方误差** |
| TensorRT entropy（KL） | 12.664 | 87.9% | **KL 散度** |
| 「真实 MSE 最优」（沿阈值扫描取 argmin） | 78.593 | 24.7% | —— 这是数值真值，不是一种实现 |

三个 `amax` **不是同一个量**：`71.599 / 12.664 = 5.65`，而 `78.593` 只是一个用来「判断谁更接近误差最小化」的参照点。看清楚这张表，本篇要回答的问题就出来了：

1. 校准到底在校准什么？为什么不能只用 MinMax？
2. `ema` 的「移动平均」是怎么动的，为什么第一次观测**不走**平均？
3. `HistogramObserver` 的那个「非线性搜索」在最小化什么？**它不是 KL。**
4. 那么真正的 KL 校准器长什么样，两者差多少？
5. 为什么 PyTorch 的默认 x86 配置把量化范围**砍掉 1 bit**（`reduce_range`）？

## 1. 原理

### 第一步：校准问题的形式

设张量 `x`，我们把它对称裁到 `[−α, α]` 再均匀量化到 `2^B` 个级。步长 `Δ = 2α/2^B`。总误差由两项组成：

```
E(α)  =  P(|x| ≤ α) · E[(舍入误差)² | 不越界]   +   P(|x| > α) · E[(过载误差)²]
        ╰──────── ∝ Δ²/12 ────────╯               ╰── ∝ (|x| − α)² ──╯
```

- 第一项随 **α 单调递增**（α 越小，Δ 越小，量化越精细）；
- 第二项随 **α 单调递减**（α 越小，被裁掉的尾巴越多）。

**所以「最优 α」的问题就是这两条曲线的交点问题**，而**「最小化什么」决定了交点在哪儿**：

- 如果目标函数把越界点的代价按 **(|x| − α)²** 计（L2），那么**裁掉一个 50σ 的点**（α 压到 4σ）要付 `(50σ − 4σ)² ≈ 2100σ²`；而把 Δ 从 `50σ/128 = 0.391σ` 压到 `4σ/128 = 0.031σ`、套上 `Δ²/12`，在 20000 个点上只省下 `≈ 20000 × (0.391² − 0.031²)σ²/12 ≈ 254σ²`——**差将近一个数量级，不划算，于是不裁**。（算式里的 20000 是点数、128 是 `2^8/2`。这是量级估算，不是精确最小化。）这就是 L2 目标**天然保守**的原因。
- 如果目标函数按**整个分布的形状**计（KL 散度），那么「被裁掉的那部分质量」只按它占的比例计（这里是 `20/200000 = 1e-4`），代价小得多，于是**裁得激进**。

**这一节是整篇的支点：两个校准器不一样，不是实现写得不同，是目标函数不同。**

### 第二步：五种方法各自在做什么

| 方法 | 更新式 / 目标 | 要解决的问题 |
|---|---|---|
| **min-max** | `α = max|x|` | 最省事，但要付「一个离群值撑大整个 Δ」的代价 |
| **EMA**（移动平均） | `m ← m + c·(max|x| − m)`，`c = 0.01` | 单批数据里的极值不稳；用历史把它抹平 |
| **直方图 + L2** | 在直方图上最小化**平方误差**（见第三步） | 极值不稳的**根本解**：不看极值，看整条分布 |
| **percentile** | `α = quantile(|x|, p)` | 最直白：直接砍掉尾部 `1−p` |
| **entropy / KL** | `α = argmin KL(P_ref ‖ P_quant)` | 让「量化后的分布」尽量像「量化前的分布」 |

**注意倒数第二行与第三行的关系**：percentile 与直方图法都在「砍尾巴」，但判据完全不同——percentile 砍的是**固定比例**，直方图法砍的是**使某个误差泛函最小**的那个点。**这两个数不得并列**（一个是比例、一个是泛函的 argmin）。

### 第三步：★ `HistogramObserver` 最小化的是 L2，不是 KL

这一条是整篇最重要的一条，因为**它是错的最高频处**。三条独立证据，全部来自本机装着的 torch 源码：

**证据 1：`_get_norm` 的 docstring 逐字写着只有 L2。**

```python
def _get_norm(self, delta_begin, delta_end, density):
    r"""
    Compute the norm of the values uniformaly distributed between
    delta_begin and delta_end.
    Currently only L2 norm is supported.

    norm = density * (integral_{begin, end} x^2)
         = density * (end^3 - begin^3) / 3
    """
```

`density · (end³ − begin³)/3` 正是 `∫x²dx` 在 `[begin, end]` 上的解析积分。**函数体里只有这一种形式，没有 KL 的分支。**

**证据 2：搜索函数的 docstring 逐字写着 "An approximation for L2 error minimization"。**

```
An approximation for L2 error minimization for selecting min/max.
By selecting new min/max, we filter out outliers in input distribution.
This follows the implementation of NormMinimization::NonlinearQuantizationParamsSearch in
caffe2/quantization/server/norm_minimization.cc
```

> ⚠ **挂载点要说准**：这段是 **`_non_linear_param_search` 这个方法的 docstring**，不是 `HistogramObserver` 的类 docstring。类 docstring 里只有一句泛化的 `"The search for the min/max values ensures the minimization of the quantization error with respect to the floating model"`。**引用时写清是哪一个**（P7）。

**证据 3（行为反证）：它在离群值上裁得极少。**

`verify_ptq.py`（纯 python3）把「L2 目标」与「KL 目标」各自实现了一遍，在同一份合成重尾数据上求 argmin：

| 目标函数 | argmin 的 `amax` | 相对 MinMax 裁掉 | 离真实 MSE 最优的距离 |
|---|---|---|---|
| L2 目标（`∫(x−c)²`，2^8 级） | 131.769 | 1.0% | **10.870** |
| KL 目标（`start_bin=128`） | 16.898 | 87.3% | **104.002** |
| （真实 MSE 最优，参照点） | 120.900 | 9.2% | 0 |

**L2 目标离「真实误差最小」只差 9.0%；KL 目标离它有 86% 那么远。** 也就是说：**能把「离群值裁得保守」和「接近误差最小」这两件事同时做到的，正是 L2。**

### 第四步：真正的 KL 校准器长什么样

TensorRT 的 `HistogramCalibrator` 是开源且可读的。本实现者把整份源码直取下来逐行读过（见 §5 与 `specs.TRT_ENTROPY_FILE`），核心是一个对**候选阈值**的线性扫描：

```
for i in range(start_bin, len(bins) + 1):
    # 把 [0, i) 的直方图合并成 2^8 个量化桶，摊平、归一
    new_density = merge_and_normalize(bins[:i], nbins=1 << (num_bits-1+unsigned))
    # 参考分布：把 i 之后的质量全倒进最后一桶
    reference_density = bins[:i]; reference_density[-1] += sum(bins[i:])
    divergences.append(entropy(reference_density, new_density))   # scipy.stats.entropy
last_argmin = len(divergences) - 1 - np.argmin(divergences[::-1])
calib_amax  = calib_bin_edges[last_argmin * stride + start_bin]
```

四处可复核的实现规格（**逐行读过源码**，不是二手转述）：

| 参数 | 值 | 出处 |
|---|---|---|
| `num_bins` | 2048 | `HistogramCalibrator.__init__` 的形参默认值 |
| `start_bin` | 128 | `compute_amax(..., start_bin: int = 128)` 的形参默认值 |
| `percentile` | 99.99 | `compute_amax(..., percentile: float = 99.99)` 的形参默认值 |
| 平局 | **取更大的 `amax`** | `last_argmin = len(d)-1 - argmin(d[::-1])`——**反向 argmin 即取最后一个最小值** |

最后一行值得单说：`np.argmin` 遇并列取**第一个**，代码却先反转再取，于是取的是**最后一个**，也就是**更大的 `amax`**。这不是笔误——它把「并列时宁可少裁一点」写进了实现。

> ⚠ **percentile 的可靠一手答案只有 99.99 这一个**（它就是上表里那个形参默认值）。
> 「BERT 用 99.999 / YOLO 用 99.95」这类分档**未找到可靠一手来源**，**正文不使用**。

### 第五步：`reduce_range` 为什么存在

PyTorch 的 x86 默认配置把量化范围砍掉 1 bit。原因**不是精度更好**，源码注释逐字写着：

```
For x86 quantized kernels, we need to ensure that the vpmaddubsw instruction does not
overflow. We allow for a reduce_range argument to observers that reduces the quantized
range to (0,127) or (-64, 63).
For more details see aten/src/ATen/native/quantized/cpu/qconv.cpp
This is not an optimal choice for non x86 backends as it loses a bit of precision
for activations.
```

`vpmaddubsw` 是 x86 的「无符号字节 × 有符号字节再水平相加」指令：它把**成对**乘积相加，而它的中间结果是**有符号 16 位**。两个 255 量级的无符号激活乘两个 127 量级的权重、再相加，就容易溢出 16 位。**把激活范围压到 `(0,127)` 就是把最坏情况压回去。**

**这是一处典型的「规范 vs 实现」岔口**：源码自己声明这对非 x86 后端**不是最优选择**。所以「PyTorch 的默认量化范围是 0–127」这句话只在 x86 上成立——qnnpack 的默认是 **0–255、`reduce_range=False`**（本机实测）。

## 2. 执行映射

### 2.1 观察器在 torch 里是怎么插进去的

`torch.ao.quantization.prepare(model)` 把每个被量化的模块包一层，并在里面挂一个 `activation_post_process` 子模块。`HistogramObserver.forward` 只做两件事：

```
x_min, x_max = torch.aminmax(x)
if is_uninitialized:  self.reset_histogram(x, x_min, x_max)     # 第一次：直接建直方图
else:                 self._combine_histograms(...)             # 之后：合并 + 更新 min/max
```

**`min_val` / `max_val` 在这里被更新，但那是「直方图的范围」，不是「选出来的 `amax`」。** 选出来的 `amax` 只在 `calculate_qparams()` 里现算，**不回写**。这一点是本篇 §4 的第一个坑。

### 2.2 三个组件名与一个数值量级

- **`torch.ao.quantization.observer`** —— 三种观察器的实现（`MinMaxObserver`、`MovingAverageMinMaxObserver`、`HistogramObserver`）；
- **`torch.ao.quantization.fake_quantize.FakeQuantizeBase`** —— `observer_enabled` 这个开关**挂在这里**，不在 observer 上（见 §4.3）；
- **`torch.ao.quantization.quantize_fakes` / `prepare_qat`** —— QAT 路径，会把 `MinMaxObserver` 换成 `FusedMovingAvgObsFakeQuantize`（内层仍是 `MovingAverageMinMaxObserver`）；
- **量级**：`HistogramObserver` 的直方图固定 **2048 桶**，搜索步长 `stepsize = 1e-5`，而 `1/2048 = 0.0488%`——**这就是它能裁的最小幅度**（见 §4.2）。

### 2.3 裁剪换 Δ 的代价曲线

§1 第一步那两条曲线的交点，本程序把它量出来了（本机实测，判据见 §5）：`amax` 从最大值往下压，逐点 MSE **先降后升**；minmax 的 MSE 是真实最优的 **1.51 倍**。

> ⚠ **这里只写定性方向与「明显次优」这个结论，不写具体倍数**：这个倍数在控制者与审查者两次独立测量里给出过 5.56×/3.17× 与 4.63×/2.38× 两组不同的值——它随目标函数里 Δ 的定义与尾部积分的近似方式变，**不具备跨实现的可复现性**。上表那个 1.51 是**本程序的判据**（256 级非对称 affine、逐点平方误差）下量出来的，写在这儿只是为了让读者能复现，**不是一条可以外推的结论**。

## 3. 实现与演进

**第一代：MinMax。** 实现最简单，也最脆弱——§0 那张表里它的 `amax` 是 104.440，而真实误差最优只要 78.593。

**第二代：EMA。** 把「一批的极值」换成「跨批的移动平均」：

```
m ← m + c·(新观测的极值 − m),     c = averaging_constant = 0.01
```

`c = 0.01` 意味着**约 100 批的时间常数**。实测（本机，torch 2.11.0）：

| 依次观测到的 max | 10.0 | 20.0 | 30.0 |
|---|---|---|---|
| 记录下来的 `max_val` | **10.0** | 10.100000381469727 | 10.29900074005127 |

**第一个值就是 10.0 本身**，因为 `min_val` / `max_val` 的初值是 `+inf` / `−inf`，而 `min(x, inf) = x`——**「第一次不走 EMA」不是一条显式分支，是初值 `inf` 的后果**。如果第一次也走平均（从 0 起），会得到 `0.01 × 10 = 0.1`，**差 100 倍**。

> ⚠ 这两个记录值（10.100000381469727 / 10.29900074005127）不是四舍五入的打印误差——它们是 **float32 存储**的直接后果。`verify_ptq.py` 里用 `struct` 模拟 float32 舍入可以**逐位复现**；不做这一步，纯 double 递推给的是 `10.1 / 10.298999999999999`，**对不上**。这条差别正是最后一篇 `11-rounding` 里「舍入在哪里发生」那件事的先声。

**第三代：直方图 + 非线性搜索。** 不再依赖极值，而是保存整条分布（2048 桶），在桶边界上搜 `amax`。**它最小化 L2 误差**（§1 第三步）。

**第四代：entropy / KL 校准器。** 目标改成分布层面的相似度。TensorRT 的实现（§1 第四步）是本主题唯一一份可逐行读到的 KL 校准器。

**这四代不是替代关系**：上面那张表里 `HistogramObserver`（71.599）与 KL（12.664）服务的是两个不同的下游——单看「让量化误差最小」，前者更对；单看「让激活分布形状保住」，后者更对。**谁被选中取决于你要保什么，不取决于哪个更新。**

## 4. 踩坑与失败模式

### 4.1 把 `HistogramObserver` 说成「KL 校准器」

**现象**：文档／注释／课件里写「PyTorch 用 KL 散度校准，TensorRT 也用它」，于是读者以为两者的 `amax` 应该接近。

**成因**：这是**把两篇来源搞混**——「KL 校准」这个说法的原始出处是 TensorRT 的文档与 TensorRT 的 `_compute_amax_entropy`。PyTorch 的 `HistogramObserver` 借用的是 caffe2 的 `NormMinimization`，目标是 L2。

**规避**：读源码，不读二手转述。判据在 §1 第三步与 §5 那张表里：**同一份数据上两者差 5.65 倍，方向相反**。任何把它们当成同一件事的说法，都会被这组数直接否证。

> ⚠ **不要反过来矫枉过正**：`torch.ao.quantization` 里确实**没有** percentile observer（本机 observer 模块的类清单里搜不到 `*ercentile*`），但 TensorRT 的 `HistogramCalibrator` **三种都支持**（`method` 可取 `'entropy'` / `'mse'` / `'percentile'`）。「PyTorch 没有 percentile」与「TensorRT 没有 percentile」是两句不同的话。

### 4.2 读 `observer.min_val` 看裁剪量，永远是 0%

**现象**：想量「直方图观察器裁掉多少」，于是写 `1 − (ho.max_val − ho.min_val)/(mm.max_val − mm.min_val)`，得到 **0.00%**，于是得出结论「它根本不裁」。

**成因**：`min_val` / `max_val` 是**直方图的范围**，搜索的结果**不回写**它们。真正的 `amax` 在 `calculate_qparams()` 里现算。

**规避**：用 `scale × (quant_max − quant_min)`。换成这个判据，同一份数据给出 **0.05%**——而 `0.05% ≈ 1/2048 = 0.0488%`，**恰好是一个直方图桶的宽度**。

**这里还有第二层坑**：那 0.05% **不是**「L2 最优恰好不想裁」的证据，**是搜索网格的物理下限**。本机六组重尾数据的裁剪量是 **0.05% / 0.05% / 0.05% / 6.40% / 25.15% / 0.05%**——五组落在「一个桶」的地板上，一组（对数正态，整条尾巴都重、没有孤立离群值）真的裁了 25%。**换一组数据这个数就变，它是一个随数据走的量。**

### 4.3 以为 `eval()` 会冻结观察器

**现象**：`prepare` 之后切 `model.eval()`，以为 `amax` 定型了；结果标注阶段又跑了几批数据，`amax` 被后来那批撑大。

**成因**：**`observer.py` 全文既没有 `self.training`，也没有 `observer_enabled`**（五个 torch 版本逐一确认）。`observer_enabled` 挂在 **`FakeQuantizeBase`** 上，那是 QAT 用的伪量化模块。

**规避**：QAT 路径上写 `model.apply(torch.ao.quantization.disable_observer)`。注意两件事：

1. `disable_observer` 是一个 **apply 回调**（实现就是 `if isinstance(mod, FakeQuantizeBase): mod.disable_observer()`）。**写 `Q.disable_observer(model)` 把整个模型传进去是一个静默的空操作**——不报错，也不生效。
2. 对 `prepare()` 插入的**普通 observer**，**连 `apply()` 也不起作用**（它不是 `FakeQuantizeBase`）。要冻死它只能换成 `FakeQuantize` 路径，或者手工把 `min_val` / `max_val` / `histogram` 置成常量。

### 4.4 把「`reduce_range` 砍了 1 bit」当成精度优化

**现象**：看到 x86 默认配置里 `reduce_range=True`、激活范围是 0–127，以为这是「更省」或「更准」。

**成因**：`0–127` 比 `0–255` **少一个 bit 的分辨率**。它的存在理由是 `vpmaddubsw` 的**16 位中间结果溢出**（§1 第五步有源码原文），而且源码自己说这对非 x86 后端不是最优选择。

**规避**：在非 x86 后端把 `reduce_range` 关掉（qnnpack 的默认就是 `False`、范围 `0–255`）。另外注意一个**直接抛异常**的组合：`per_tensor_symmetric + reduce_range=True + quint8` 会抛 `NotImplementedError: Cannot reduce range for symmetric quantization for quint8`——**它拒绝执行，而不是给一个错的答案**，这是好事。

### 4.5 用 percentile 校准却没有为它写下来源

**现象**：文档里写「BERT 用 99.999、YOLO 用 99.95」这类分档。

**成因**：这类分档在 TensorRT 文档与源码里**都找不到**。可定位的一手值只有一处：`compute_amax(..., percentile: float = 99.99)` 这个形参默认值。

**规避**：引用 percentile 时只写 99.99 并给出处；**不要为了让段落好看而补一个分档表。**

## 编译与运行

```bash
cd 130-model-optimization/quantization/code
make run PYTHON=/path/to/python    # 或直接 make run，会自动探测装了 torch 的解释器
```

本篇对应的编号程序是 **`code/08_calibration.py`**（真 torch；B 类断言在程序内部，失败即非零退出）。只跑本篇：

```bash
make 08_calibration
```

纯 python3 的独立复核（A/C 类，**不 import torch**）在 `code/verify_ptq.py`：

```bash
make verify
```

> ⚠ **`08_calibration.py` 里的 KL 有两条路径。** 首选 `scipy.stats.entropy`——TensorRT 源码
> 第 20 行用的就是它；**没装 scipy 时退到 numpy 向量化的 `Σ p·ln(p/q)`**。
> 本机五个解释器**有两个（torch 2.7.1 / 2.13.0 那两个环境）没装 scipy**，靠这条兜底跑通。
> **兜底路径与 `verify_ptq.py` 的标量循环是同一公式的两份代码，不是两个独立来源**——
> 所以「KL 由 scipy 算」这句话只在装了 scipy 的解释器上成立，程序会把实际走的路径打出来。

## 5. 实测数据

**测量条件**：Apple M1 Pro（10 核）/ macOS / **CPU-only，无 GPU 数字**；python 3.12.9；**torch 2.11.0**。本主题五个解释器上都跑过（torch 2.6.0 / 2.7.1 / 2.8.0 / 2.11.0 / 2.13.0，`2.9 / 2.10 / 2.12` 本机无解释器、不声称覆盖）。下表全部来自 `make run`（**真 torch**），数值也逐条记在 `code/specs_ptq.py` 里。

### 5.1 EMA 的递推（`MovingAverageMinMaxObserver`）

| 依次观测到的 `max` | 记录下来的 `max_val` |
|---|---|
| 10.0 | **10.0**（第一次：直接取，不走 EMA） |
| 20.0 | 10.100000381469727 |
| 30.0 | 10.29900074005127 |

`averaging_constant = 0.01`（源码形参默认值）。判据：逐次单样本 `aminmax`，更新式 `m ← m + c·(新值 − m)`，全程 float32。`verify_ptq.py` 用 `struct` 模拟 float32 后**逐位复现**这三行。

### 5.2 同一份数据上的四个 `amax`（**判据不同，不得并列成「估算 vs 实测」**）

数据：200000 点 `N(0,1)`，随机 20 个点乘 40（`torch.manual_seed` 固定）。

| # | 判据 | `amax` | 相对 MinMax 裁掉 |
|---|---|---|---|
| ① | `MinMaxObserver`：`scale × 255/2` | 104.440 | 0% |
| ② | `HistogramObserver`：`scale × 255/2` | 71.599 | 31.4% |
| ③ | TensorRT `_compute_amax_entropy`（移植后跑，`start_bin=128`） | 12.664 | 87.9% |
| ④ | 真实 MSE 最优：沿对称阈值扫描 40 点取 argmin | 78.593 | 24.7% |

- ②/③ = **5.65×**；④ 到 ② 的距离 6.994，到 ③ 的距离 65.929。**②比③更靠近真实误差最优 ⇒ ②的目标是 L2。**
- `verify_ptq.py`（纯 python3、小规模）给出的是同一结论的**推导版**：L2 最优 131.769 / KL 最优 16.898 / 真实 MSE 最优 120.900。**规模不同 ⇒ 数字不同，只有定性结构一致。**
- ⚠ **「裁掉百分之几」必须写出分母。** 上表的分母是 `MinMax 的 amax`（104.440）；而编号程序的 B.1 那一段把分母取成 **`max|x|`（112.276）**，于是同一个 `amax = 78.593` 在那里的读数是 **30.0%** 而不是 24.7%。**两个百分比都对，但它们是两个量**——`max|x|` 与 `MinMax 的 amax` 不相等，因为 affine 量化的零点被约束为整数、量程会整体平移（见 `04-integer-formats` 的 `|δ| ≤ s/2`）。

### 5.3 六组重尾数据的裁剪量（`HistogramObserver` 相对 MinMax）

判据：`scale × (quant_max − quant_min)`，不是 `min_val/max_val`（见 §4.2）。

| 数据（构造方式） | 裁掉 | = 多少个直方图桶 |
|---|---|---|
| `N(0,1)` + 5 个点 ×100 | 0.05% | 1.0 |
| `Laplace(0,1)` + 3 个点 ×200 | 0.05% | 1.0 |
| `0.1σ` 窄分布 + 单个 50σ 尖峰 | 0.05% | 1.0 |
| 尺度混合的 t 型重尾 | 6.40% | 131.0 |
| **对数正态**（无孤立离群值，整条尾巴都重） | **25.15%** | 515.0 |
| `U(−1,1)` + 单个 300 | 0.05% | 1.0 |

**1 个桶 = 1/2048 = 0.0488%。** 五组落在这个地板上——**那不是「不裁」，那是网格的最小刻度**。

### 5.4 源码事实（真 torch，直接读文件）

| 事实 | 值 / 出处 |
|---|---|
| `HistogramObserver.bins` 默认 | 2048 |
| 搜索步长 | `stepsize = 1e-5`（**分位数**步长） |
| `_get_norm` docstring | `"Currently only L2 norm is supported."` |
| 搜索函数 docstring | `"An approximation for L2 error minimization for selecting min/max."`，挂在 **`_non_linear_param_search`** 上 |
| 跟随的实现 | caffe2 `NormMinimization::NonlinearQuantizationParamsSearch` |
| `observer.py` 里有没有 `self.training` | **没有** |
| `observer.py` 里有没有 `observer_enabled` | **没有**（它在 `FakeQuantizeBase` 上） |
| x86 默认 PTQ 激活观察器 | `HistogramObserver`，范围 **`(0,127)`**（`reduce_range=True`） |
| qnnpack 默认 PTQ 激活观察器 | `HistogramObserver`，范围 **`(0,255)`**（`reduce_range=False`） |
| `reduce_range=True` + `quint8` + `per_tensor_symmetric` | 抛 `NotImplementedError` |
| `min_val`/`max_val` 初值 | `+inf` / `−inf`（这就是「第一次不走 EMA」的成因） |

### 5.5 对称量化的除数

判据：`max(|min|, |max|) / ((quant_max − quant_min)/2)`。int8 的 `quant_min/max = −128/127`，于是除数 = **255/2 = 127.5**。

本机实测：`max|w| = 12.30447960` → observer 的 `scale = 0.09650572`。
`/127.5 = 0.09650572`（差 **1.55e-9**，float32 舍入量级）；`/127 = 0.09688567`（差 **3.80e-4**，≫ 舍入）；`/128 = 0.09612875`（差 3.77e-4）。**多数二手材料写 `/127`，是错的。**

## 6. 小结与延伸

- 校准不是「估计极值」，是**在一个目标函数下选 `amax`**；换目标函数就是换方法。
- **`HistogramObserver` 最小化 L2 误差，TensorRT 的 entropy 校准器最小化 KL 散度**；本机同一份数据上两者差 **5.65 倍**，方向相反。把前者说成 KL 是本主题最常见的错误。
- 「L2 目标天然保守」不是经验观察，是**目标函数里越界代价写成平方的必然后果**（§1 第一步的算账）。
- EMA 的第一次观测不走平均，是 `inf` 初值的后果；`c = 0.01` 与 float32 存储一起决定了那三个记录值。
- 一个数字是不是「实测」取决于**判据**：同一份数据、同一个观察器，读 `min_val` 得 0.00%，读 `qparams` 得 0.05%，而 0.05% 又是网格的地板。**判据没写出来的数字，不得与另一个数字并列。**

**延伸**：`09-gptq` 把「选 `amax`」换成「逐个权重决定舍入」，用的是二阶信息；`10-awq-and-smoothquant` 则不动 `amax`，改的是**把哪一维的难度搬到哪一维**；`11-rounding` 从另一个角度问同一个问题——**「最近舍入」真的是最优的舍入吗**。

## 参考

**本仓库程序（可复现）**

- `code/08_calibration.py` —— 真 torch 侧的 B/A/C/M 断言（本机实测值来源；01 共 **64** 条）。
- `code/verify_ptq.py` —— 纯 python3 侧的 A/C 推导（L2 目标 vs KL 目标的独立实现、TRT KL 的纯 python 复刻）。
- `code/specs_ptq.py` —— 本主题的外部常量源，每条带 `# 来源：`。

**本机实现源码（逐字读过，五个 torch 版本一致）**

- `torch/ao/quantization/observer.py`：`HistogramObserver.forward` / `_get_norm` / `_non_linear_param_search` / `MinMaxObserver.__init__` 的 `vpmaddubsw` 注释 / `MovingAverageMinMaxObserver.forward`。
- `torch/ao/quantization/fake_quantize.py`：`FakeQuantizeBase.observer_enabled`（第 84 / 93 行）。

**外部来源**

- TensorRT `tools/pytorch-quantization/pytorch_quantization/calib/histogram.py`（2026-09-18 用 python `urllib` 直取整文件 16541 字节并逐行读过；`num_bins=2048` 在第 51 行、`compute_amax` 的默认形参在第 124 行、`_compute_amax_entropy` 在第 182 行、`scipy.stats.entropy` 在第 20 / 244 行、`last_argmin` 在第 250 行）。**引用值**——本仓库不复核其测量口径。
- caffe2 `quantization/server/norm_minimization.cc` 的 `NormMinimization::NonlinearQuantizationParamsSearch`：**仅由 torch 源码注释指认，本仓库未逐字读到该文件**。

**UNSOURCED（正文未使用）**

- 「BERT 用 99.999 / YOLO 用 99.95」这类 percentile 分档：未找到可靠一手来源。

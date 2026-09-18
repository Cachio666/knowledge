---
title: 伪量化插在哪、observer 的机制错在哪、LSQ 的梯度与缩放
slug: quantization-qat-fake-quant-and-lsq
domain: 130-quantization
topic: qat
level: 3
difficulty: advanced
prerequisites:
  - 130-quantization/qat/01-ste
  - 130-quantization/ptq/01-calibration
tags: [fake-quantization, observer, lsq, learned-step-size, prepare-qat]
related:
  - 130-quantization/qat/03-pact-and-comparison
  - 130-quantization/ptq/04-rounding
  - 130-quantization/theory/02-granularity
code:
  - code/02_fake_quant_lsq.py
status: stable
updated: 2026-09-18
---

# 伪量化插在哪、observer 的机制错在哪、LSQ 的梯度与缩放

> 上一篇说清了反传长什么样；这一篇说它**在什么地方被调用**，并订正三处「机制挂错位置」的流行说法：`prepare_qat` **内部走了一次 `convert`**（不是只挂 observer）；`observer_enabled` **不在 `observer.py` 里**，它是 QAT 侧伪量化模块的 buffer；`Q.disable_observer(model)` 是**静默空操作**，必须写 `model.apply(Q.disable_observer)`——而且对 `prepare()` 插入的普通 observer，**连 `apply()` 也不起作用**。最后是 LSQ：它学的是步长 `s`，而它的梯度式在两种实现下**相差整整一个 `v/s`**。

## 0. 背景与动机

`01-ste` 回答了「梯度怎么穿过 `round`」。但那个反传**不是凭空发生的**——它挂在一个具体的模块上，由一段具体的代码调用。而这一段代码恰恰是本域**机制类错误最密集**的地方：三个说法各自都「听起来对」，实际都挂错了位置。

先把要回答的问题列出来：

1. 权重侧的伪量化**插在哪一行**？是 hooks、是包装器、还是别的？
2. 伪量化之后，被 optimizer 更新的到底是**谁**——量化后的权重，还是别的？
3. `prepare_qat` 到底做了什么？**「挂上 observer」够不够描述它？**
4. 切 `model.eval()` 能不能冻结 observer？如果不能，**该调哪个函数**？为什么那个函数在两种模块上效果不同？
5. **LSQ 学什么、怎么学**？它的梯度式为什么长成三段？式子里那个 `1/√(N·Q_P)` 是干什么的？

第 4 条与第 5 条各自都有一个「隐藏的分岔」：**同名的东西有两种含义**。第 4 条是「冻结」有两种入口（一个有效、一个静默失效）；第 5 条是「`s` 的梯度」有两种实现（论文式与朴素 autograd，**相差一个 `v/s`**）。两处都是 P8——**同一条结论，规范和实现写的不是一回事**。

## 1. 原理

### 第一步：权重侧的伪量化**写死在 forward 里**

不是 hook，不是包装器。`torch.ao.nn.qat` 的 `Linear.forward` 全文只有一行：

```python
return F.linear(input, self.weight_fake_quant(self.weight), self.bias)
```

`weight_fake_quant` 是一个 `FakeQuantize` 子模块，在 `__init__` 里构造一次，之后 forward 每次调用它。

**这一行决定了三件事**：

1. 伪量化是**前向图的一部分**（不是旁路），所以它出现在 `grad_fn` 链上；
2. 量化后的权重**只活在这一次 forward 里**——它没有被写回 `self.weight`；
3. 因此 `self.weight` 始终是**未量化的 fp32 数值**。

### 第二步：被更新的是 **fp32 master weight**

本机实测（`prepare_qat` 之后，`torch 2.11.0`，五版本一致）：

| 检查项 | 结果 |
|---|---|
| 第 0 层的类 | `torch.ao.nn.qat.modules.linear.Linear` |
| `weight.dtype` | `torch.float32` |
| forward 是否有 `grad_fn` | 是 |
| SGD 一步后权重是否变化 | 是 |

⇒ **准确回答「伪量化在 update 前还是后」**：**forward 里用伪量化后的权重算输出（update 前）**；反向梯度沿 `01-ste` 那条掩码路径回传到**未量化的 fp32 master weight**；**optimizer 更新的是那个 fp32 权重**。量化发生在 forward 每一次调用时，是**瞬时**的。

> ⚠ **不要说「QAT 训练的是量化后的权重」。** 量化后的权重是个临时值，它没有 `.grad`，也不出现在 `state_dict` 里。

### 第三步：★★ `prepare_qat` 内部**走了一次 `convert`**

`prepare_qat` 的源码（`torch/ao/quantization/quantize.py`）一共四步：

| # | 源码 | 说明 |
|---|---|---|
| ① | 检查 `model.training` 为真，否则抛 `AssertionError` | 消息逐字：`prepare_qat only works on models in training mode` |
| ② | `propagate_qconfig_(model, qconfig_dict=None)` | 把 qconfig 传播到子模块 |
| ③ | **`convert(model, mapping=mapping, inplace=True, remove_qconfig=False)`** | ★ **这里调用了 `convert`** |
| ④ | `prepare(model, observer_non_leaf_module_list=set(mapping.values()), inplace=True)` | 挂 observer |

**③ 的存在是这一节的全部要点。** 常见的说法是「`prepare` 与 `prepare_qat` 的区别是后者会换成 QAT 模块」——这话的结果对，**机制错**：它不是一步「换成 QAT 模块」，而是**调用了一次 `convert`**（把 float 模块换成 QAT 模块），然后再 `prepare` 挂 observer。`remove_qconfig=False` 正是留给第 ④ 步用的。

> ★ **一处真实的版本差异**（本机五版本对拍发现）：第 ① 步的**写法变过**。
> · **torch 2.6.0 / 2.7.1 / 2.8.0**：`assert model.training, "prepare_qat only works on models in training mode"`
> · **torch 2.11.0 / 2.13.0**：`if not model.training: raise AssertionError("...")`
> 两段**消息文字相同、语义相同，语句形式不同**。⇒ **不得写成「torch 的写法是 X」**，必须带版本。
> 计划文件记作「① 断言 `model.training == True`」——那是**旧版本那一档**的转述被当成了普适陈述。

### 第四步：★★ observer 不理会 `model.eval()`——但机制要写准

**现象**（本机实测，五版本一致）：`prepare()` 之后跑一批得 `(−1.3738, 1.0014)`；切 `eval()` 后再跑一批**放大 100 倍**的数据，得 `(−94.5049, 96.3650)`——**继续更新**。`eval()` 不冻结 observer。

**机制**常被写成「observer 里有个开关，`eval()` 时关掉」。**源码不是这样。**

本机在五个 torch 版本上逐个 grep `torch/ao/quantization/observer.py`（2.11.0 上是 2159 行）：

| 字符串 | 命中数（五个版本全部相同） |
|---|---|
| `self.training` | **0** |
| `observer_enabled` | **0** |

⇒ **`observer.py` 全文既没有 `self.training` 也没有 `observer_enabled`。** observer **根本没有**「训练/评估」这个概念。

那 `observer_enabled` 在哪？在 **`torch/ao/quantization/fake_quantize.py`**，由 `FakeQuantizeBase.__init__` 注册：

```python
self.register_buffer("observer_enabled", torch.tensor([1], dtype=torch.uint8))
```

⇒ **它是 QAT 侧伪量化模块的 buffer，不是 observer 的机制。**

> ⚠ 这一条是「**现象对、机制错**」的典型（P4：编机制从句）。**「`eval()` 不冻结 observer」这句话是对的**，但解释它的那句话如果是「observer 内部有个开关」，那就是编的。

### 第五步：★★ 冻结 observer 的正确入口——以及它对谁有效

**`torch.ao.quantization.disable_observer` 是一个 `apply` 回调**：

```python
def disable_observer(mod):
    if isinstance(mod, FakeQuantizeBase) or _is_fake_quant_script_module(mod):
        mod.disable_observer()
```

它期望 `mod` 是**单个模块**。本机实测的两种调用形式：

| 写法 | 结果 |
|---|---|
| `Q.disable_observer(model)` | **静默空操作**：不抛异常、`observer_enabled` 不变、`scale` 照样更新 |
| `model.apply(Q.disable_observer)` | 正确：`observer_enabled` 变 0，`scale` **冻结** |

**静默**两个字是重点：`Q.disable_observer(model)` 把整个模型传进去，`isinstance(model, FakeQuantizeBase)` 为假，函数什么都不做就返回。**既不报错也不生效**——这类失败最难发现。本机实测那一行：`scale` 从 `0.016060` 到 `0.016060`（没变），再跑一批 ×100 的数据后成了 `0.033375`（**还在更新**）。

**第三层，也是最容易被漏掉的一层**：对 **`prepare()` 插入的普通 observer**（比如 x86 默认的 `HistogramObserver`），**连 `apply()` 也不起作用**——它不是 `FakeQuantizeBase`，**没有 `observer_enabled` 这个属性**，回调整整不起作用。

本机实测：

| 模型 | `apply(Q.disable_observer)` 后 | 再跑一批之后 |
|---|---|---|
| QAT（`prepare_qat`） | `observer_enabled = 0`，`scale` 冻结 | **不变** ✅ |
| PTQ（`prepare`） | 普通 observer **无** `observer_enabled` | **继续更新** ❌ |

⇒ **准确的说法是：「冻结 observer」只对 QAT 侧的伪量化模块成立；PTQ 侧的普通 observer 没有可冻结的开关。**

> ⚠ **附带的**：`freeze_bn_stats` **不是**通用观察器冻结工具。本机在 `torch/ao/` 下 grep `def freeze_bn_stats` 的全部落点只有两个，都是 `nn/intrinsic/qat/modules/` 下的 **fused Conv+BN / Linear+BN** 模块。**它管的是 BN 的 running stats，不是 observer 的 qparams。**

### 第六步：LSQ 学什么、梯度长什么样

**量化器**（与 `theory/01` 的对称/非对称量化是同一套写法）：

```
v̄ = ⌊clip(v/s, −Q_N, Q_P)⌉          v̂ = v̄ · s
```

- **无符号（激活）**：`Q_N = 0, Q_P = 2^b − 1`；
- **有符号（权重）**：`Q_N = 2^(b−1), Q_P = 2^(b−1) − 1`。

**LSQ 学的是每层一个独立的 step size `s`**（一个 fp32 标量）。**它不学 clipping 阈值、不学 zero-point。** 初始化：

```
s₀ = 2·⟨|v|⟩ / √Q_P
```

**梯度式（论文式 3）是三段分段线性**：

```
∂v̂/∂s = ⌊v/s⌉ − v/s        当 −Q_N < v/s < Q_P
       = −Q_N               当 v/s ≤ −Q_N
       =  Q_P               当 v/s ≥ Q_P
```

三段的**分界点就是两个 clip 边界**。三段各自的形状值得逐一读出：

| 段 | 值 | 含义 |
|---|---|---|
| 内部 | `⌊r⌉ − r` = **量化残差的相反数** | **有界：`|·| ≤ 0.5`** |
| 下饱和 | 常数 `−Q_N` | clip 把 `v̄` 钉死了，`dv̄/ds = 0` |
| 上饱和 | 常数 `Q_P` | 同上 |

**这里有两条容易看漏的**：

1. **内部段的梯度非常小**（≤ 0.5），**饱和段非常大**（`Q_P`，8-bit 下是 255）。所以 **LSQ 的 `s` 梯度主要由「被截断的那部分」驱动**（本机实测：`Q_P = 255` 时饱和段的梯度是内部段上界的 **510 倍**）。
2. **两段在 clip 边界上不连续**：`r → Q_P⁻` 时内部段趋于 0，`r = Q_P` 时跳到 `Q_P`。**这是设计的一部分**，不是 bug——它正是「被截掉多少就惩罚多少」这个意图的体现。

### 第七步：★★ `1/√(N·Q_P)` 缩放因子——以及「`s` 的梯度」有两种

**缩放因子的动机**：让「平均更新幅度 / 平均参数幅度」在网络各层大致相同。权重侧 `g = 1/√(N_W·Q_P)`，激活侧 `g = 1/√(N_F·Q_P)`。它由两条经验合成：**精度越高、更新幅度越小**（`Q_P` 越大、`g` 越小）；**元素越多、更新幅度越大**（`N` 越大，未缩放的更新越大，于是要乘更小的 `g` 去抵消）。

**在本机小规模上量这个效应**（判据见 §5.2）：

| `N`（该层权重元素数） | 不加缩放：`\|Δs\|/\|s\| ÷ \|Δw\|/\|w\|` | 加 `g` 后 |
|---|---|---|
| 2048 | 7.21 | 0.215 |
| 4096 | 10.04 | 0.442 |
| 8192 | 32.22 | 0.105 |
| 16384 | 21.59 | 0.118 |

**两个结论**：① 不加缩放时 `s` 的相对更新幅度**比权重的大一个数量级上下**（论文报告的是 2–3 个数量级，那是 ImageNet 级网络上 `N` 比本机大 1–2 个数量级的引用值）；② 加上 `g` 之后两者**落到同一个量级**（比值 < 0.5）。**这就是那个因子的全部作用。**

**★ 第二处 P8：式子里那个 `−v/s` 项来自「`round` 走 STE」。**

论文式 3 的内部段是 `⌊v/s⌉ − v/s`。**但如果你直接用 `torch.round` 跑 autograd**（`round` 处处导数为 0），得到的是：

```
∂v̂/∂s = ⌊v/s⌉              （朴素 autograd，round 导数为 0）
```

**两者相差整整一个 `v/s`。** 本机实测（`torch 2.11.0`，`Q_N = 8, Q_P = 7`）：

| `v/s` | 论文式（STE） | 朴素 autograd | 关系 |
|---|---|---|---|
| −8.0 | −8.0000 | −8.0000 | 一致（饱和段） |
| −1.5 | −0.5000 | −2.0000 | 同号不同值 |
| 0.0 | 0.0000 | 0.0000 | 一致 |
| **1.2** | **−0.2000** | **+1.0000** | **符号相反** |
| **2.4** | **−0.4000** | **+2.0000** | **符号相反** |
| 7.0 | +7.0000 | +7.0000 | 一致（饱和段） |
| 9.0 | +7.0000 | +7.0000 | 一致（饱和段） |

**分歧只在内部段**——两个饱和段里 `v̄` 已经被 clip 钉死，两条路的 `dv̄/ds` 都是 0，结果一致。

而内部段的分歧有一条可以背下来的判据：`⌊r⌉ − r` 与 `⌊r⌉` **同号当且仅当** `⌊r⌉` 与残差同号；在 `k < r < k+1` 且 `k ≥ 1` 时**必然反号**（残差 `= k − r < 0` 而 `k > 0`）。

⇒ **所以「LSQ 的 `s` 梯度」这句话必须带上是哪一种**。这不是学究式的精度差别：**方向都可能相反**。

## 2. 执行映射

### 2.1 五个组件名与两个数值量级

- **`torch.ao.nn.qat.modules.linear.Linear`** —— QAT 版的 Linear，伪量化写在它的 `forward` 里。
- **`weight_fake_quant` / `activation_post_process`** —— 两个 `FakeQuantize` 实例，前者管权重（per-channel 对称）、后者管激活（per-tensor affine）。
- **`FusedMovingAvgObsFakeQuantize`** —— x86 QAT 默认的**激活**伪量化类（把 observer 与 fake-quant 融合成一个 ATen 算子）。
- **`HistogramObserver`** —— **PTQ** x86 的默认**激活观察器**。★ 这与 QAT 的默认不是一个东西。
- **`observer_enabled` / `fake_quant_enabled`** —— `FakeQuantizeBase` 的两个 buffer 开关。

**数值量级 ①（qconfig 的「声明」vs「实际生效」）**：

| 引擎 | 激活**声明** | 激活**实际生效** | 权重 |
|---|---|---|---|
| **x86** | `quant_max=255, reduce_range=True` | **0 / 127** | −128 / 127，per-channel symmetric |
| qnnpack | `quant_max=255, reduce_range=False` | 0 / 255 | −128 / 127，**per_channel = False** |

`reduce_range` 把量程砍掉 1 bit，源码里的理由是 x86 kernel 的溢出：

> *"For x86 quantized kernels, we need to ensure that the **vpmaddubsw** instruction does not overflow"*

**数值量级 ②（缩放因子）**：`g = 1/√(N·Q_P)`。`N = 4096, Q_P = 7` 时 `g ≈ 5.9e-3`；`N = 10^6, Q_P = 3` 时 `g ≈ 5.8e-4`。

### 2.2 一张表：三处「机制挂错位置」

| # | 流行说法 | 本机核实的结果 |
|---|---|---|
| 1 | `prepare_qat` 是「只挂 observer」 | **内部走了一次 `convert`**（第 ③ 步） |
| 2 | `eval()` 不冻结 observer，因为 observer 里有开关 | 现象对；**`observer.py` 里既无 `self.training` 也无 `observer_enabled`**，两个都是 0 命中 |
| 3 | 冻结用 `Q.disable_observer(model)` | 那是**静默空操作**；要写 `model.apply(Q.disable_observer)`；**且对普通 observer 连这个也不起作用** |

三处的共同形状是：**结论听起来对，机制挂在了错误的对象上**。

## 3. 实现与演进

**第一代：固定 step size（`s = max|v|/(2^(b−1)−1)`）。** 校准一次定死，训练中不动。这就是 PTQ 的做法——但用在 QAT 里会有一个直接的缺陷：**网络在适应量化误差，而量化网格本身不跟着动。**

**第二代：LSQ——把 `s` 变成可学参数。** 每层一个 fp32 标量。它有两个设计细节值得单独记住：

1. **初始化用 `s₀ = 2⟨|v|⟩/√Q_P`**，不是一个拍脑袋的常数。它把初始 step 定在「数据平均幅度对应的量级」上。
2. **梯度缩放 `g = 1/√(N·Q_P)`**。没有它，`s` 的相对更新幅度与权重的差 1–3 个数量级；加上它之后两者同量级（§1 第七步有本机数据）。

**第三代（同期的另一条路）：PACT——把 clip 上界 α 变成可学参数。** 详见 `03-pact-and-comparison`。两者是同一笔交换的两种买法：**多学一个参数，换更贴合数据分布的量化网格。**

> ⚠ **不要写「LSQ 是 PACT 的改进版」或反过来。** 它们学的东西不同（步长 vs clip 上界），梯度形状不同，甚至在「对 `round` 的处理」上也有差别。**没有一条证据说其中一个普遍更优。**

## 4. 踩坑与失败模式

### 4.1 把 `prepare_qat` 说成「只挂 observer」

**现象**：写「`prepare` 挂 observer，`prepare_qat` 挂 observer 并换成 QAT 模块」。

**成因**：这个说法**结果对**——调用完之后模块确实换成了 QAT 版、observer 确实挂上了。所以从输出看不出问题。

**规避**：去读源码，数一遍调用。`convert(model, mapping=..., inplace=True, remove_qconfig=False)` 是**第三步**，不是隐含在别处的副作用。`code/02_fake_quant_lsq.py` 的 B.2 直接对 `inspect.getsource` 做断言。

### 4.2 ★ 编一个 observer 的「训练/评估开关」

**现象**：写「observer 内部有一个 `training` 判断，所以 `eval()` 时会冻结」——然后解释不了为什么实测还在更新。

**成因**：这是**编机制从句**（P4）的形状：现象（`eval()` 不冻结）是真的、可复现的，于是顺手给它配一个听起来合理的机制。

**规避**：**grep 一遍**。`observer.py` 里 `self.training` 与 `observer_enabled` 都是 **0 命中**（本机五个版本一致）。真正带 `observer_enabled` 的是 `fake_quantize.py` 的 `FakeQuantizeBase`。正确写法是：**「`eval()` 不冻结」是现象；冻结的开关在 QAT 侧的伪量化模块上，而不在 observer 上。**

### 4.3 用 `Q.disable_observer(model)` 冻结，然后以为冻上了

**现象**：代码写了 `Q.disable_observer(model)`，跑得通、不报错，但 qparams 还在变。

**成因**：`disable_observer` 是 **`apply` 回调**，函数体第一行 `isinstance(mod, FakeQuantizeBase)` 对整个模型为假，直接返回。**静默空操作。**

**规避**：写成 `model.apply(Q.disable_observer)`，并且**断言 `observer_enabled` 真的变成 0**（`code/02_fake_quant_lsq.py` 的 A.3 就是这么做的）。**如果模型是 PTQ 的 `prepare()` 出来的，那两者都不行**——那些 observer 没有 `observer_enabled`。

### 4.4 把 `freeze_bn_stats` 当通用观察器冻结工具

**现象**：写「QAT 里用一个 `freeze_bn_stats` 把观察器冻住」。

**成因**：名字听起来通用，而且它确实出现在 QAT 的教程代码里。

**规避**：本机 grep 到的落点只有两个，都是 fused 模块。**它管 BN 的 running stats。** 要冻结 qparams 用 4.3 的办法。

### 4.5 ★ 把「LSQ 的 `s` 梯度」写成论文式，却用朴素 autograd 实现

**现象**：文档里贴论文式 3，代码里写 `torch.round(v/s)` 然后 `.backward()`。**两者都对，但它们不是同一个量。**

**成因**：`torch.round` 的反向是 0，这是 PyTorch 的默认行为；要拿到论文式必须**显式给 `round` 写一个反向恒等的 `autograd.Function`**。而多数实现忘了这一步。

**规避**：**判据是符号**。在 `k < r < k+1`（`k ≥ 1`）这一段上，两式的**符号相反**（本机实测 `r = 1.2` 时 `−0.2` vs `+1.0`）。文档里必须写明「论文式隐含 `round` 走 STE」这个前提。

## 编译与运行

```bash
cd 130-quantization/qat/code
make run PYTHON=/path/to/python
```

本篇对应的编号程序是 **`code/02_fake_quant_lsq.py`**（真 torch；B 类断言在程序内部，失败即非零退出）。只跑本篇：

```bash
make 02_fake_quant_lsq
```

纯 python3 的独立复核（A/C 类，**不 import torch**）在 `code/verify.py` 的 A.4–A.6、A.12–A.13、C.3–C.4 节；`code/02_fake_quant_lsq.py` 的 C.6 节另有 DeprecationWarning 触发点的实测断言。

> ★ **`DeprecationWarning` 的触发点**（本仓库五版本实测）：
> 它打在**调用** `prepare` / `prepare_qat` 上，**不是**打在 `import` 上；而且 **2.8.0 起**就有。
>
> | torch | `import torch.ao.quantization` 时 | `Q.prepare_qat(model)` 时 |
> |---|---|---|
> | 2.6.0 / 2.7.1 | **0 条** | **0 条** |
> | 2.8.0 / 2.11.0 / 2.13.0 | **0 条** | **3 条** |
>
> 判据：`warnings.catch_warnings(record=True)` 分别包住两段，数 `DeprecationWarning` 的条数。
> **它仍然是噪声、不是失败**（功能照常，五版本全部 `rc=0`）——错的是它**在哪一步、从哪一版起**出现。
> 计划文件与任务书都写成「每次 import 都会打一条」且「在 2.11.0+」，**两半都与实测不符**。
>
> ⚠ 同一批调用还会打若干条 **`UserWarning`**（*"Please use quant_min and quant_max to specify
> the range for observers..."*）——**也是噪声**。

## 5. 实测数据

**测量条件**：Apple M1 Pro（10 核）/ macOS / **CPU-only，无 GPU 数字**；python 3.12.9；**torch 2.11.0**。五个解释器（2.6.0 / 2.7.1 / 2.8.0 / 2.11.0 / 2.13.0）都跑过本篇全部断言；`2.9 / 2.10 / 2.12` 本机无解释器。数值逐条记在 `code/specs.py`。

### 5.1 伪量化的插入位置与参数类型（判据：`prepare_qat` 之后逐项读）

| 检查项 | 结果 |
|---|---|
| 第 0 层的类 | `torch.ao.nn.qat.modules.linear.Linear` |
| `weight.dtype` | `torch.float32` |
| forward 有 `grad_fn` | 是 |
| SGD 一步后权重变化 | > 0（**更新的是 fp32 master weight**） |
| 权重伪量化类 | `FusedMovingAvgObsFakeQuantize`，`per_channel = True` |

### 5.2 ★ LSQ 的相对更新幅度（判据：单层 LSQ 量化，4-bit 有符号权重 `Q_P = 7`；SGD `lr = 0.01`、60 步；每步取 `|Δw|/|w|` 与 `|Δs|/|s|`，对 60 步取算术平均，再取比值）

| `N` | 不加缩放 | 加 `g = 1/√(N·Q_P)` | 改善倍数 |
|---|---|---|---|
| 2048 | 7.212 | 0.2150 | 34× |
| 4096 | 10.043 | 0.4417 | 23× |
| 8192 | 32.220 | 0.1048 | 307× |
| 16384 | 21.589 | 0.1179 | 183× |

> ⚠ **规模不同 ⇒ 数字不同**：论文报告「不加缩放时差 **2–3 个数量级**」，那是 ImageNet 级网络上的数字（`N` 比本机大 1–2 个数量级，而 `|Δs|` 正比于 `N`）。**本表只支持定性结构**：不加缩放失衡、加了之后同一个量级。

### 5.3 ★ 两种 `s` 梯度的对照（判据：`Q_N = 8`、`Q_P = 7`）

| `v/s` | 论文式（`round` 走 STE） | 朴素 autograd | 关系 |
|---|---|---|---|
| −10.0 | −8.0000 | −8.0000 | 一致（下饱和） |
| −7.5 | −8.0000 | −8.0000 | 一致（下饱和） |
| −1.5 | −0.5000 | −2.0000 | 同号不同值 |
| 0.0 | 0.0000 | 0.0000 | 一致 |
| **1.2** | **−0.2000** | **+1.0000** | **符号相反** |
| **2.4** | **−0.4000** | **+2.0000** | **符号相反** |
| 7.0 | +7.0000 | +7.0000 | 一致（上饱和） |
| 9.0 | +7.0000 | +7.0000 | 一致（上饱和） |

**分歧只在内部段。**

### 5.4 observer 与开关（判据：源码 grep 计数 + 切 `eval()` 后重跑）

| 检查项 | 结果（五版本一致） |
|---|---|
| `observer.py` 里 `self.training` 命中数 | **0** |
| `observer.py` 里 `observer_enabled` 命中数 | **0** |
| `fake_quantize.py` 里 `observer_enabled` 命中数 | 12 |
| PTQ observer 切 `eval()` 后 | **继续更新**（`(−1.3738, 1.0014) → (−94.5049, 96.3650)`） |
| `Q.disable_observer(model)` | **静默空操作**（`observer_enabled` 仍为 1；`scale` `0.016060 → 0.016060 → 0.033375`，**继续更新**） |
| `model.apply(Q.disable_observer)`（QAT） | 生效，`scale` **冻结** |
| `model.apply(Q.disable_observer)`（PTQ 普通 observer） | **无效**，继续更新 |
| `def freeze_bn_stats` 在 `torch/ao/` 下的落点 | **2 个**，都是 `nn/intrinsic/qat/modules/*_fused.py` |

### 5.5 qconfig 的「声明」与「实际生效」（判据：`qconfig.activation.p.keywords` vs 模块属性）

| 引擎 | 声明 `quant_max` / `reduce_range` | 生效 qmin/qmax | 权重 per-channel symmetric |
|---|---|---|---|
| x86（QAT） | 255 / True | **0 / 127** | True |
| qnnpack（QAT） | 255 / False | 0 / 255 | **False** |
| x86（PTQ 激活观察器） | —— | `HistogramObserver` | —— |

### 5.6 引用值（**本仓库不复核其测量口径**）

| 项 | 值 | 出处 |
|---|---|---|
| LSQ 论文 | **arXiv:1902.08153**（ICLR 2020） | `WebSearch` 独立佐证编号/作者 |
| 初始化 | `s₀ = 2⟨|v|⟩/√Q_P` | 计划基线 |
| 梯度式（式 3） | 三段分段线性 | 计划基线 |
| 缩放因子 | `g = 1/√(N·Q_P)` | 计划基线 |
| 消融：不加缩放的失衡 | **2–3 个数量级** | 计划基线 |
| 消融：完全不加缩放 | **不收敛**（2-bit ResNet-18，LR 降到 `1e-4` 仍低 **3.4** 个点） | 计划基线 |

## 6. 小结与延伸

- **权重侧的伪量化写死在 `forward` 里**：`F.linear(input, self.weight_fake_quant(self.weight), self.bias)`。参数**始终是 fp32**，optimizer 更新的是它。
- ★ **`prepare_qat` 内部走了一次 `convert`**（第 ③ 步，在 `prepare` 之前）。「只挂 observer」是错的。
- ★ **`observer.py` 里既无 `self.training` 也无 `observer_enabled`**（五版本 0 命中）。`observer_enabled` 是 `fake_quantize.py` 里 `FakeQuantizeBase` 的 buffer，**是 QAT 侧伪量化模块的机制**。
- ★ **`Q.disable_observer(model)` 是静默空操作**；要写 `model.apply(Q.disable_observer)`；**对 PTQ 的普通 observer，连这个也不起作用**。`freeze_bn_stats` 只管 fused BN，不是通用工具。
- **LSQ 学每层一个 `s`**，初始化 `s₀ = 2⟨|v|⟩/√Q_P`，**不学 clip 阈值、不学 zero-point**。
- **它的梯度三段分段线性**：内部 `⌊r⌉ − r`（**有界 ≤ 0.5**）、两端饱和为常数 `±Q_P / −Q_N`。⇒ **梯度主要由被截断的部分驱动**。
- ★ **内部段那个 `−v/s` 项来自「`round` 走 STE」**；朴素 `torch.round` + autograd 得到的是 `⌊r⌉`，**两者在 `k < r < k+1` 上符号相反**。引用时必须带上是哪一种。
- **`g = 1/√(N·Q_P)` 把 `|Δs|/|s|` 从高于权重一个数量级压到同一个量级**（本机四个 `N` 全部成立）。

**延伸**：`03-pact-and-comparison` 给出另一条路（学 clip 上界 `α`）与两者的三层区别，以及 QAT vs PTQ 的实测差距。`ptq/01-calibration` 是 observer 那一侧的另一半（它是 PTQ 的入口）；本篇补的是 **QAT 侧的 observer 语义与它不同的地方**。

## 参考

**本仓库程序（可复现）**

- `code/02_fake_quant_lsq.py` —— 真 torch 侧：模块类型与 dtype、`prepare_qat` 源码断言、qconfig 声明 vs 生效、`eval()` 不冻结、`disable_observer` 的三种效果、LSQ 两种梯度的对照、相对更新幅度。
- `code/verify.py` 的 `ref_lsq_grad_s` / `ref_lsq_grad_s_naive` / `ref_lsq_init_scale` / `ref_lsq_grad_multiplier` —— 纯 python 的独立参照实现。
- `code/specs.py` —— 外部常量源，每条带 `# 来源：`。

**外部来源**

- Esser, McKinstry, Bablani, Appuswamy, Modha，*Learned Step Size Quantization*，**arXiv:1902.08153**（ICLR 2020）。量化器、初始化、式 3、`1/√(N·Q_P)` 与消融。**引用值**——本实现者未逐字读到式 (3) 的原文；编号与作者由 `WebSearch` 独立佐证。
- **本机 torch 源码**（五版本一致）：`torch/ao/quantization/quantize.py::prepare_qat`、`observer.py`、`fake_quantize.py`、`torch/ao/nn/qat/modules/linear.py`、`torch/ao/nn/intrinsic/qat/modules/{conv_fused,linear_fused}.py`。

**UNSOURCED（正文未使用）**

- LSQ / PACT 在 ImageNet 上的具体精度数字：本机没有 ImageNet 与 GPU，**不复现也不引用**。

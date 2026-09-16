---
title: flash attention：在线 softmax 与不物化 S
slug: flash-attention
domain: 40-kernels
topic: attention
level: 3
difficulty: advanced
prerequisites:
  - 40-kernels/attention/01-attention-as-a-kernel
  - 40-kernels/softmax-normalization/02-online-softmax
  - 40-kernels/softmax-normalization/01-softmax-stability
tags: [flash-attention, online-softmax, output-accumulator, rescaling, tiling, register-pressure]
related:
  - 40-kernels/gemm/02-shared-memory-tiling
  - 20-gpu-architecture/execution-model/03-occupancy-and-latency-hiding
  - 40-kernels/attention/03-flash-attention-backward
code:
  - code/02_flash_attention.cu
status: stable
updated: 2026-09-16
---

# flash attention：在线 softmax 与不物化 S

> 一句话：flash attention 把 S 从 HBM 里拿掉了——它在片上算出来、当场用掉、当场丢掉。
> **实现这件事的关键不在 `(m, l)` 的重标定**（那两条递推的正文在
> [softmax-normalization/02](../softmax-normalization/02-online-softmax.md)），
> 而在于**输出累加器 `O` 也要跟着同一个 `α` 重标定**：`O_new = O_old·exp(m_old − m_new) + P·V`。
> 漏掉这一份不会报错、不会溢出，只会安静地错。

## 0. 背景与动机

[01 篇](01-attention-as-a-kernel.md) 把账算清了：朴素实现把 `S = softmax(QKᵀ/√d)` 物化到 HBM，
于是每算一个 `S²` 的中间量要付两次 HBM 往返（写一次、读一次）。
S=16384、单 head 单 batch 时，**光是这笔往返就是 320.5 µs，而全部矩阵乘只要 138.9 µs**
（推导值，`make verify` §B2）。

**具体对照场景**：S=4096、d=128，单 head 单 batch。

| | HBM 字节 | AI | 与 `I*`(FP16)=295.4 的关系 |
|---|---|---|---|
| 朴素（01 篇） | 71.3 MB | 120.5 | 40.8%，**memory-bound** |
| flash（本文） | 4.19 MB | 2048.0 | **6.93 倍**，**compute-bound** |

**17 倍的字节差、17 倍的 AI 差**，来源只有一个：那 `2S²·esize` 的 S 写 + 读被删掉了。

问题是：**不物化 S，softmax 就做不完了**。softmax 的分母要跨整行求和，
而分母里的 `exp(x − max)` 需要先知道 `max`——「先看完全部数据、再回头算一遍」这件事，
在「数据一块一块流过芯片、算完就丢」的执行模型里不成立。

[softmax-normalization/02](../softmax-normalization/02-online-softmax.md) 已经给出了解法：
把「先求 max 再求 exp」压成一遍，维护一对状态 `(m, l)`，每来一块就重标定一次：

```text
m_new = max(m_old, rowmax(S_block))                          （1）  ← 正文在 softmax/02
l_new = l_old·exp(m_old − m_new) + Σ exp(S_block − m_new)    （2）  ← 正文在 softmax/02
```

**那两条递推的推导、证明、结合律、以及「状态必须住 fp32」的全部论证都在那边，本文只回链。**
但 attention 比 softmax 多一样东西：softmax 的输出是 `y = exp(x−m)/l`，可以最后回看一遍 `x` 再算；
attention 的输出是 `O = Σ p_j v_j`——**一个已经被加权求和过的向量**。
`(m, l)` 的重标定救不了它：基准从 `m_old` 抬到 `m_new` 时，`O` 里已经累积的那些项
是按**旧基准**算出来的。

本文要交付的就是这第三条递推，以及它带来的全部工程后果（寄存器压力、状态格式、失败模式）。

## 1. 原理

### 1.1 记号

沿用 [softmax/02](../softmax-normalization/02-online-softmax.md) 的记号：KV 沿序列维切成
若干块 `B₁, …, B_T`。对某一行 query（`q` 固定），第 `t` 块的 logits 记作
`s^(t) = scale·q·K_{B_t}ᵀ`，对应的 value 块记作 `V_{B_t}`。

状态三个：`m`（该行见过的最大 logit）、`l`（按当前基准归一化的 exp 之和）、
`O`（按当前基准归一化的**加权和**，未除以 `l`）。

`O` 的定义式要和 `l` 摆在一起看，这是本文全部推导的起点：

```text
l_t = Σ_{j ∈ S_t} exp(s_j − m_t)
O_t = Σ_{j ∈ S_t} exp(s_j − m_t)·v_j          ← 与 l_t **同一个基准 m_t**
```

两个状态用的是**同一个** `m_t`，这一点决定了它们必须用同一个 `α` 重标定。

### 1.2 第三条递推的推导

要的是 `O_{t+1} = Σ_{j ∈ S_{t+1}} exp(s_j − m_{t+1})·v_j`。

**第一步：把求和拆成「旧 + 新」。**

```text
O_{t+1} = Σ_{j ∈ S_t}     exp(s_j − m_{t+1})·v_j      ← 旧元素，归一到**新**基准
        + Σ_{j ∈ B_{t+1}} exp(s_j − m_{t+1})·v_j      ← 新块，同样归一到新基准
```

第二项当场就能算（`P = exp(s^(t+1) − m_{t+1})`，`P·V_{B_{t+1}}`）。
第一项要对一堆已经流过去的元素重算 `exp`——这正是要避免的。

**第二步：把基准的移动量从求和号里提出来。** 这一步与 `l` 的推导是**同一个支点**：

```text
s_j − m_{t+1} = (s_j − m_t) + (m_t − m_{t+1})
exp(s_j − m_{t+1}) = exp(s_j − m_t) · α,      α ≡ exp(m_t − m_{t+1})
```

`α` 与 `j` 无关，而且 `v_j` **不参与**这个拆分——`α` 是标量，可以提到求和号外面：

```text
Σ_{j ∈ S_t} exp(s_j − m_{t+1})·v_j
  = α · Σ_{j ∈ S_t} exp(s_j − m_t)·v_j
  = α · O_t
```

右边的 `O_t` 恰好就是旧状态本身——「旧元素按旧基准加权求和」这件事在 `t` 时刻已经做完了。

**第三步：合起来。**

```text
O_{t+1} = O_t · α + P·V_{B_{t+1}},      α = exp(m_t − m_{t+1})        （3）
```

**这就是本文的正文。** 三条式子并列：

```text
m_new = max(m_old, rowmax(S_block))                        （1）回链 softmax/02
l_new = l_old · α + Σ exp(S_block − m_new)                 （2）回链 softmax/02
O_new = O_old · α + exp(S_block − m_new) · V_block         （3）★ 本文
```

**（3）与（2）用的是同一个 `α`**，这是最要紧的一点：它们共享同一个基准 `m`，
所以基准一动，两者必须以**同样的倍数**跟着动。写成算法：

```text
m_new = max(m_old, rowmax(S_block))
α     = exp(m_old − m_new)                       ← 方向由 m_old ≤ m_new 保证，α ∈ (0, 1]
l_new = l_old · α + Σ exp(S_block − m_new)
O_new = O_old · α + exp(S_block − m_new) · V_block
```

循环结束后再统一除：`O_final = O/l`。**归一化只做一次，在循环外**——
这也解释了为什么 `O` 必须一直以「未除 `l`」的形态活着：如果每块都除一次，
下一次重标定时就还得乘回来。

### 1.3 一个可以手算的例子

`make verify` §A6 逐项打印了这个例子（`scale` 取 1 便于手算）：`q = [1, 0]`，两个 KV 块
`K₁ = [[1,0],[0,1]]`、`K₂ = [[2,0],[0,0]]`，`V₁ = [[1,2],[3,4]]`、`V₂ = [[5,6],[7,8]]`。

```text
块 1：s = [1, 0]，m₁ = 1
      l₁ = exp(0) + exp(−1) = 1.3678794411714423
      O₁ = exp(0)·[1,2] + exp(−1)·[3,4] = [2.1036383235143270, 3.4715177646857693]

块 2：s = [2, 0]，块内 max = 2
      m₂ = max(1, 2) = 2
      α  = exp(1 − 2) = 0.3678794411714423
      l₂ = l₁·α + (exp(0) + exp(−2)) = 1.6385500076446680
      O₂ = O₁·α + (exp(0)·[5,6] + exp(−2)·[7,8])
         = [6.7212322735375700, 8.3597822811822375]
      O₂/l₂ = [4.1019390572027756, 5.1019390572027756]
```

**验算**：一次算完全部 4 个 key（朴素两遍），`m = 2`、`l = 1.6385500076446680`、
`O = [4.1019390572027747, 5.1019390572027747]`——**前 15 位逐位相同**（相对差 `2.165e-16`）。

注意 `O₁·α` 那一项：它把「按基准 1 算出来的加权和」整体缩到了基准 2 的尺度上。
**少了这一项，`O₂` 就是把两个不同基准下的量直接相加**——§4.2 会把后果算出来。

### 1.4 正确性与性质

- **正确性**：`O_new = O_old·α + P·V` 只是把 `O` 的定义式按「旧/新」拆开再提公因子，
  与 [softmax/02](../softmax-normalization/02-online-softmax.md) 的（1）（2）共享同一个支点，
  所以**对任意分块方式都成立**。`make verify` §A1 在四种分块（16/32/64/128）上
  与朴素两遍比对，最大按范数误差 `7.951e-16`。
- **`α ∈ (0, 1]`**：因为 `m_old ≤ m_new`（max 的定义直接给的）。
  乘 `α` 只会缩小已累积的量，不会放大——**方向不会造成上溢**。写反（用 `exp(m_new − m_old)`）
  会立刻上溢：fp16 的上溢阈值是 `11.0901`（[softmax/01](../softmax-normalization/01-softmax-stability.md)）。
- **权重之和为 1**：把 V 取成全 1，在线版的输出就是每行权重之和，
  `make verify` §A1 算出与 1 的最大差 `4.441e-16`。
- **`O` 的大小与 `Br·d` 成正比**：这是 flash 付出的东西，§2 会给出具体数字。

### 1.5 为什么「不物化 S」不是「不物化任何东西」

一句话对比：

| | 朴素（01 篇） | flash（本文） |
|---|---|---|
| 中间量 | `S`：`S×S`，落 HBM | `(m, l, O)`：`O` 是 `Br×d`，住寄存器/片上 |
| 大小 | `O(S²)` | `O(S·d)` |
| 与什么成正比 | 序列长度的**平方** | 序列长度的**一次方** |
| 能不能流式 | 不能（要先看完全部数据） | 能（每块只依赖自己的 `(m,l,O)`） |

**这就是 01 篇 §1.5 说的「绝对字节数从 `O(S²)` 降到 `O(S·d)`」在实现层面的样子。**

## 2. 执行映射

### 一个 block 里怎么落地

两层分块 `Br`（query 行）× `Bc`（KV 列），状态与 tile 的归属：

| 数据 | 住哪 | 大小 |
|---|---|---|
| Q tile | SMEM | `Br·d·esize` |
| K / V tile | SMEM（双缓冲则 ×2） | `2·Bc·d·esize` |
| `m`、`l` | 寄存器 / SMEM | `Br` 个标量各一份 |
| **`O` 累加器** | **寄存器** | `Br·d` 个 fp32 |
| S tile | 寄存器（fragment） | `Br·Bc` 个 fp32 |

`make verify` §B4 把常见配置算了出来（`d = 128`、fp16、`esize = 2`）：

| `Br` | `Bc` | 线程 | SMEM | O 寄存器/线程 | KV 双缓冲后 |
|---|---|---|---|---|---|
| 64 | 64 | 128 | 48 KB | 64 | 80 KB |
| 128 | 64 | 256 | 64 KB | 64 | 96 KB |
| 128 | 128 | 256 | 96 KB | 64 | 160 KB |
| 256 | 64 | 256 | 96 KB | 128 | 128 KB |

全部 ≤ 228 KB 的 SMEM 上限。但**寄存器那一列才是真正的约束**：

> `O` 的寄存器数与 `Br·d` 成正比，与 `Bc` **无关**。
> `Br=128`、`d=128`、只开 4 个 warp（128 线程）时，**光 `O` 就要 128 个寄存器/线程**。
> 每 SM 有 65536 个 32-bit 寄存器；256 线程 × 64 个 = 16384 个，
> 还剩 75% 给 S 的 fragment、`m`/`l` 与地址（推导值，`make verify` §B4）。

这就是 flash attention 最硬的资源约束：**`Bc` 可以随便调，`Br` 不行**——
调 `Br` 会同时动寄存器和 SMEM 两边，而 `Bc` 只动 SMEM。

### 具体组件与量级

- **两段矩阵乘**走 tensor core（规格值：`m16n8k16` 需 `sm_80+`，`m64nNk16`（wgmma）需 `sm_90a`；
  见 [tensor-core/01](../../20-gpu-architecture/tensor-core/01-mma-primitive.md)），
- **中间的重标定**走 FP32 向量通路：`rowmax`（`S` 条 `fmaxf`）、`α`（1 次 `exp`）、
  `l·α`（1 次乘）、`O·α`（`d` 次乘）——**每块一次，不是每元素一次**。
  一块 `Br×Bc` 摊到每个 S 元素上是 `(d + 3)/(Br·Bc)` 次额外操作。
- 参考量级（规格值）：shared memory 延迟 20~30 周期、每周期交付 128 字节；
  一次片外访存往返 400~800 周期；每 SM 每周期最多发射 4 条指令。
- **重标定的算术量可以忽略**，因为它是 `O(Br·d)`；真正花时间的是把 K/V tile 搬进来
  （`2·Bc·d·esize` 字节）与两段 wgmma。这也是为什么 `Bc` 取大一点通常更好——
  搬运有更好的摊薄，而 `O` 的开销不变。

### 访存账

| 写法 | S=4096 单 head 单 batch 的字节 | 其中 S 相关的那部分 |
|---|---|---|
| 朴素（01 篇） | 71.3 MB | S 的写 + 读 = 67.1 MB |
| flash（本文） | 4.19 MB | 0（S 从未落 HBM） |

（字节数与 AI 见 `make verify` §B1，逐组核对；时间量级见 §B2：
S=4096 时纯算力 8.7 µs、S 的写 + 读 20.0 µs。两张表用的是同一套口径——
**同一个 `d`、同一个 `esize`、同一份 Q/K/V/O**；口径混用会让结论差 2 倍以上，见 01 篇 §4.1。）

## 3. 实现与演进

按「状态里有什么」这条线：

1. **两遍 softmax + 物化 S（01 篇）**。状态是 `S` 本身，`O(S²)`，不能流式。
2. **在线 softmax（[softmax/02](../softmax-normalization/02-online-softmax.md)）**。
   状态是 `(m, l)`，`O(1)`。但它的输出 `y = exp(x−m)/l` 还得**回看一遍 `x`**——
   在 softmax 里这一遍是读同一份数据，代价可以接受；在 attention 里「回看」意味着
   把 S 留着或者重算，两条路都不便宜。
3. **在线 softmax + 输出累加器（本文）**。把 `O` 纳入重标定，于是连回看都省了。
   代价是 `O` 必须一直活在寄存器里，大小 `Br·d`。
   **这是 flash attention 的全部机关**：用一份 `O(Br·d)` 的寄存器，换掉 `O(S²)` 的 HBM 流量。
4. **反向（[03 篇](03-flash-attention-backward.md)）**。前向不存 S，反向就得重算 S；
   而重算需要的 `(m, l)` 恰好就是前向已经算出来的那两个状态——**一份数据两处用**。
5. **Hopper 上的工程形态（[04 篇](04-warp-specialization-and-variants.md)）**。
   当 `O` 占据 128 个寄存器/线程时，用同一个 warp 去做 TMA 搬运会挤掉计算资源；
   把搬运交给另一个 warpgroup 就成了自然选择。

`code/02_flash_attention.cu` 给了两版对照：`rescale_O=1`（正确）与 `rescale_O=0`（只重标定 `l`）。

## 4. 踩坑与失败模式

### 4.1 「`m` 只增不减」被误读成「`m` 可以延后更新」

**现象**：既然 `m` 单调不减，那就一次定下来——用第一块的 max，或者预设一个上界（`m ≡ 0`）。
这样连 `α` 都不用算，代码更简单。

**这条已经在 [softmax-normalization/02 §4.1](../softmax-normalization/02-online-softmax.md)
的正文里，本文不重推，只回链。** 要强调的是它在 attention 里的**后果更重**：
softmax 那里丢的是 `y` 的分母，attention 这里丢的是**已经加权求和过的输出向量**。
`make verify` §C1 复现了溢出侧的现象：fp16 下 `exp(30) = inf`，`inf/inf = nan`；
而同样一批 logits 在 fp32 里不减 max 也**不溢出**（结果是 1.0000）——
**溢出是格式问题，不是算法问题**（阈值 `88.7228` 在 fp32 上要 logits 更大才会碰到）。

**成因**：不更新 `m` 时块内算的是 `exp(s − m_old)`，而 `m_old` 可以比块内 max 小很多，
指数直接冲出格式范围。**规避**：`m` 每块必须更新。「`m` 单调递增」与「`m` 可以不更新」
是两件事：前者是递推**保住**的性质，后者是主动放弃它。

### 4.2 忘了重标定 `O`（只重标定 `l`）

**现象**：`α` 算出来了、`l` 也乘了，`O` 那一份忘了乘。程序**不报错、不产生 inf、不产生 nan**，
输出看上去完全正常。`make verify` §C4 在 `8×8` 的输入上把两种写法并排跑：

```text
正确（重标定 O）：与朴素两遍的最大相对误差 = 1.404e-14
错误（忘标定 O）：与朴素两遍的最大相对误差 = 3.299e+01      ← 错了 33 倍
而且它既不产生 inf 也不产生 nan
```

**成因**：`O_old` 是按**旧基准** `m_old` 累积的，而新块的权重 `exp(s − m_new)` 用的是新基准。
直接相加等于把两个不同基准下的量混在一起——每一项的系数本该是 `exp(m_old − m_new)` 而不是 1。
注意这与 §4.1 **不是**同一件事：§4.1 错在 `m` 本身没更新，这条错在 `m` 抬起来之后**谁跟着一起走**。

**规避**：把 `α` 的乘**同时**写在 `l` 与 `O` 上，并且在代码里让它们挨着（`code/02_flash_attention.cu`
里这两行是连着的）。这条错误的隐蔽性在于它没有任何数值异常信号——
**只有对答案才能发现**，所以验证必须靠「与朴素两遍/参考实现对答案」，不能靠「看有没有 nan」。

### 4.3 rescale 因子在 fp16 状态下归零

**现象**：`α = exp(m_old − m_new)` 在 `gap = m_new − m_old` 足够大时舍入到 `0`，
于是 `O_old · α = 0`——**整段已累积的贡献在一次乘法里消失**，而且是**分量级**的消失：
本该非零的输出分量直接变成 0，其余分量看不出异常。`make verify` §C2：

```text
数据：第一块 60000 个 logit 全为 0（V = [1,0]），第二块 1 个 logit = 20（V = [0,1]）
真值      ：l = 60000·exp(−20) + 1 = 1.0001236692，O = [1.2365392516e-04, 0.9998763461]
fp32 状态 ：l = 1.0001236200，O = [1.2365393263e-04, 0.9998763952]
fp16 状态 ：l = 1.0000000000，O = [0.0000000000e+00, 1.0000000000]     ← 整段丢弃
```

那个 `1.236539e-04` 的分量在 fp16 状态下**恰好变成 0**。丢掉的量有上界：
`|ΔO| ≤ max|v|·l_old·α ≤ max|v|·65504·2^-25 = 1.952171e-03·max|v|`——
按整个输出向量的范数看是 `1.237e-04` 的相对误差，**有界、可见，但不是灾难**。

**先把这个阈值本身的口径说清楚**（`make verify` §C2 把两个口径并排打印）：

| `gap = m_new − m_old` | 口径一（`gap` 精确）fp16 | 口径二（`gap` 也过 fp16）fp16 | fp32 状态 |
|---|---|---|---|
| 17.000000 | 5.960464e-08 | 5.960464e-08 | 4.139938e-08 |
| 17.328125 | 5.960464e-08 | 5.960464e-08 | 2.981885e-08 |
| 17.328700 | **0.000000e+00** | 5.960464e-08 | 2.980171e-08 |
| 17.343750 | 0.000000e+00 | **0.000000e+00** | 2.935655e-08 |
| 20.000000 | 0.000000e+00 | 0.000000e+00 | 2.061154e-09 |
| 104.000000 | 0.000000e+00 | 0.000000e+00 | **0.000000e+00** |

- **口径一**（`gap` 用精确算术，只有 `exp` 的结果落进 fp16）：阈值就是
  [softmax/01](../softmax-normalization/01-softmax-stability.md) 给出的 **`17.3287`**（`= 25·ln2`，
  正好是最小次正规数的一半，舍入到偶）。
- **口径二**（状态存 `half` 时 `gap` 本身也要过一遍 fp16）：`17.3287` 会被舍到
  `17.328125`（fp16 在该区间的刻度是 `2^-6`），`exp` 反而**没有**归零；
  第一个真正归零的 fp16 刻度的 `gap` 是 **`17.34375`**。

两个口径差一个 fp16 ulp。**报数时必须注明是哪一种**——这正是本仓库在
[softmax/02 §4.2](../softmax-normalization/02-online-softmax.md) 里用 `17.3287` 的那个口径。

**成因**：`α` 的下溢边界（[softmax/01](../softmax-normalization/01-softmax-stability.md) §1）
在 attention 里落到了 `O` 上。**规避**：状态 `(m, l, O)` 一律住 **fp32**——
fp32 的阈值是 `103.9721`，真实 logits 范围内不可达。

**比 `l` 那一侧更隐蔽的一点**：`l` 少一点还会在分母上体现出来（每行权重之和不再等于 1），
而 `O` 的某个分量归零之后，输出看上去**完全正常**——它只是一个「本来是 1.2e-4 的值变成了 0」。

### 4.4 把「少 rescale 几次」当成优化目标

**现象**：既然每次重标定都要乘一次 `α`，那就增大 `Bc` 减少重标定次数——听起来是纯赚的。

**这条的正文在 [softmax/02 §4.3](../softmax-normalization/02-online-softmax.md)，本文只回链**：
rescale 次数从 0 涨到 31，误差**没有**单调增长；真正的误差来源是 `l` 的逐项累加。
在 attention 里这条的形态略有不同——`Bc` 的选取由**访存摊薄**决定
（每块搬 `2·Bc·d·esize` 字节，而重标定只花 `O(Br·d)`），
增大 `Bc` 的收益来自搬运效率，**不是**来自 rescale 次数。

## 编译与运行

配套程序是 **`code/02_flash_attention.cu`**，一个 kernel 两版对照：

- `rescale_O = 1` —— 正确版：`α` 同时乘到 `l` 与 `O` 上；
- `rescale_O = 0` —— 故意只在 `l` 上乘 `α`，用来复现 §4.2。

它需要 `nvcc` 与 GPU：

```bash
cd code
make 02_flash_attention
./02_flash_attention                 # S=1024，d=128，Br=64，Bc=64
./02_flash_attention 4096 128 128 64 # S=4096，d=128，Br=128，Bc=64
```

程序会打印四部分：资源账（SMEM 与 `O` 的寄存器数，与本文 §2 的表对照）、
访存账（与 §1.5 / §B1 对照）、两版与 CPU 在线参考的逐元素最大差、
以及本文与回链的分工说明。

**两类数字必须分开看**：

- 本文表格里的数、`7.951e-16`、`1.952171e-03`、`17.3287` / `17.34375`——
  **推导值**，可用 `make verify` 的 §A6 / §B4 / §C2 / §C4 复核；
- 各内核的逐元素差与耗时 —— **那台机器上的运行结果**。本仓库没有跑过。

本机（macOS，无 CUDA 工具链）只能跑：

```bash
make verify
```

它对应本文的是 §A6（两分块手算的逐项复核，与 §1.3 的数字逐位对照）、
§A1（四种分块与朴素两遍一致）、§B4（SMEM 与 `O` 的寄存器数）、
§C2（`α` 归零的两个口径与 attention 里的后果）、§C4（忘标定 `O` 的静默错误）。
`exp` 的四个阈值由 §A0 做一致性核对——它们的**正文**在
[softmax-normalization/01](../softmax-normalization/01-softmax-stability.md)，本文不重推。

## 6. 小结与延伸

全文一条主线：**`(m, l)` 的重标定救不了 `O`——`O` 里装的是已经加权求和过的向量，
基准一动，它必须跟着一起动。**

三条式子，其中只有第三条是本文的：

```text
m_new = max(m_old, rowmax(S_block))                  （1）回链 softmax/02
l_new = l_old·α + Σ exp(S_block − m_new)             （2）回链 softmax/02
O_new = O_old·α + exp(S_block − m_new)·V_block       （3）★ 同一个 α
```

三条可独立复用的结论：

1. **`α` 必须同时出现在 `l` 与 `O` 上**。它们共享同一个基准 `m`，所以缩放倍数必须一致。
   漏掉 `O` 那一份不报错、不溢出，只是安静地错（相对误差 33 倍）。
2. **`O` 的开销是 `Br·d` 个 fp32 寄存器**，与 `Bc` 无关。`Br=128`、`d=128`、
   128 线程时光 `O` 就要 128 个寄存器/线程——这是 flash 前向最硬的约束。
3. **状态必须住 fp32**。fp16 状态下 `α` 在 `gap > 17.3287`（精确口径）时归零，
   整段贡献被丢弃；而且丢弃可以是**分量级**的，输出看上去完全正常。

下一站：

- [03-flash-attention-backward](03-flash-attention-backward.md) —— 前向把 S 丢了，
  反向就得重算：多花 `2S²d` 的 FLOPs 换掉 `2S²·esize` 的流量。
- [04-warp-specialization-and-variants](04-warp-specialization-and-variants.md) ——
  当 `O` 占掉 128 个寄存器/线程时，谁来搬数据。
- [softmax-normalization/02](../softmax-normalization/02-online-softmax.md) ——
  本文回链的那两条递推的**正文**（连同结合律与分块无关性的证明）。

## 参考

- Dao, Fu, Ermon, Rudra, Ré, *FlashAttention: Fast and Memory-Efficient Exact Attention
  with IO-Awareness*, NeurIPS 2022 —— 第三条递推（输出累加器的重标定）与分块 IO 的方案出处。
- Milakov & Gimelshein, *Online normalizer calculation for softmax*, arXiv:1805.02867, 2018 ——
  `(m, l)` 重标定递推的标准出处；本文的（1）（2）由它而来（正文在 softmax/02）。
- 本仓库 [40-kernels/softmax-normalization/02-online-softmax](../softmax-normalization/02-online-softmax.md)
  —— **`(m, l)` 递推的正文**、结合律、以及「状态必须住 fp32」的全部论证。
- 本仓库 [40-kernels/softmax-normalization/01-softmax-stability](../softmax-normalization/01-softmax-stability.md)
  —— 四个 exp 阈值（`11.0901` / `−17.3287` / `88.7228` / `−103.9721`）的正文，本文只用结论。
- 本仓库 [40-kernels/attention/01-attention-as-a-kernel](01-attention-as-a-kernel.md) ——
  朴素实现的字节账与 AI 表，本文的对照基准。
- 本仓库 `code/verify.py` 的 §A1、§A6、§B4、§C2、§C4 —— 本文全部数字的来源。

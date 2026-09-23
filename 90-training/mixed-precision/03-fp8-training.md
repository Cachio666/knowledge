---
title: fp8 训练循环：分工、per-tensor scaling 与溢出的处置
slug: fp8-training
domain: 90-training
topic: mixed-precision
level: 3
difficulty: advanced
prerequisites:
  - 90-training/mixed-precision/01-type-selection
tags: [fp8, e4m3, e5m2, per-tensor-scaling, delayed-scaling, just-in-time-scaling, saturation]
related:
  - 90-training/mixed-precision/02-amp-execution
  - 90-training/autograd-and-backward/03-gradient-lifecycle
code:
  - code/03_fp8.py
status: stable
updated: 2026-09-23
---

# fp8 训练循环：分工、per-tensor scaling 与溢出的处置

> fp8 在训练循环里只解决三件事：**哪个张量用哪个编码**、**scale 从哪来**、**溢出怎么办**。
> 三件事里有两件**是软件层的事，不是格式的事** —— 而**溢出的处置与 fp16 相反**：
> fp16 那套「溢出就跳步」在 fp8 上会跳掉太多步（`k = 1` 的历史下**一半的步**）。

## 0. 背景与动机

### 一个具体的对照场景

上一行把梯度压到 fp16 或 bf16。再往下压到 8 位，**range 不够**了，
所以每一次编码前都要**乘一个 scale**、解码后除回来。

本文的收窄原则**写死**（计划 §2 的边界表，结构层审查 **M1**）：

> **只写训练循环里怎么用 fp8。位域与 scale 粒度一律回链。**

| 问题 | 正文位置 |
|---|---|
| 这 8 位怎么分给指数与尾数、四个变体（`e4m3fn` / `e5m2` / `fnuz`）差在哪、±448 vs ±57344、**按 torch 版本的溢出差异** | ★ `130-quantization/numerical-formats/01-fp8-family.md` **§0 全节**，此处**回链** |
| scale 的**其它**粒度（每通道 / 每 token / 每 32 元素的块） | ★ `130-quantization/theory/02-granularity.md` + `02-e8m0-and-mx.md:276`，此处**回链** |
| **累加器宽度**的误差界（fp32 → fp16 放大 8192 倍） | ★ `10-foundations/numerical-linear-algebra/02-gemm-rounding-error.md`，此处**回链** |

**本文只管训练循环里的三件事**，而它们的代价是**具体**的：

| 事 | 不用它 / 用错的代价（本机实测，`code/03_fp8.py`） |
|---|---|
| **分工** | 前向张量与反向张量用错编码（`code/03_fp8.py` 的 `A.1` 把这张表登记下来再断言） |
| **scale 是软件层的** | 若把 scale 限制成「只能取 2 的幂」（格式层的 bias 只能这么干），`amax` 只被摆到最大有限数的 **50.02%**（`C.1`） |
| **溢出的处置** | 照搬 fp16 的「溢出就跳步」：`k = 1` 的 delayed scale 下**一半的步**会溢出（`B.2`） |

★ **一个必须写明的让步**：fp8 论文自己的训练实验是 **`simulated FP8`** ——
`[C]` 2209.05433 §4 逐字（原始字节 1 次）：

> Training experiments were carried out with **simulated FP8** - tensor values were clipped to
> only those that could be represented in FP8

并且逐字说明算术在哪里做：`Arithmetic was carried out using the **wider representation**`。
⇒ **本文的程序是同一个形态**：算术在 fp64 上做，只有**存储 / 输入**被量化。
**因此它不得支撑任何关于「真 fp8 算术」（吞吐、累加顺序、Tensor Core）的结论。**

## 1. 原理

### 1.1 分工：三个理由，其中只有一个是本文自己给的

`[C]` 2209.05433 §3 逐字（原始字节 1 次）：

> The recommended use of FP8 encodings is **E4M3 for weight and activation tensors, and
> E5M2 for gradient tensors**.

★★ **但紧接着的那一句把这件事的来源说清楚了**（原始字节 1 次）：

> **This is consistent with findings in [20, 16]**, where inference and forward pass of training
> use a variant of E4M3, gradients in the backward pass of training use a variant of E5M2.

⇒ **论文自己只给了 `The recommended use of ...` 这一句（一个建议），没有给证明** ——
依据是**引用别人的工作**。所以本文的程序把「这条分工是**引用来的**」也写成断言（`A.1b`），
正文**不得**写「论文证明了前向用 E4M3、反向用 E5M2」。这是本库 **P20**（把假设/引用升格成结论）的现场。

**训练循环里的那张表**（本机实测，`A.1` 把每个张量的编码登记下来再断言）：

| 方向 | 张量 | 编码 |
|---|---|---|
| 前向 | `W1` / `W2` / `x` / `h`（激活）/ `out` | **`e4m3fn`** |
| 反向 | `dout` / `dW2` / `dh` / `dz` / `dW1` | **`e5m2`** |

### 1.2 per-tensor scaling：它是**软件层**的，而且这一层有理由

`[C]` 2209.05433 §3.2 逐字（原始字节 1 次）：

> **Leaving the per-tensor scaling to software implementation** enables more flexibility than is
> possible with a programmable exponent bias approach - the scaling factor **can take on any real
> value** (typically represented in higher precision), while programmable bias is equivalent to
> **allowing only powers of 2** as scaling factors.

**这句话是可算的，不是修辞。** 设一个张量的 `amax` 已知，你要选一个 `s` 让 `amax·s` 尽量贴近
格式的最大有限数（越贴近，表示范围用得越满）：

```
软件层（s 是任意实数）： s = max_finite / amax        ⇒  amax·s = max_finite      ← 精确
格式层（s 只能是 2 的幂）：s = 2^⌊log2(max_finite/amax)⌋  ⇒  amax·s ∈ (max_finite/2, max_finite]
```

取 `s_real = 31.99`（一个故意贴着 32 的值），本机实测（`C.1`）：

| scale 的取值方式 | `amax·s / 最大有限数` |
|---|---|
| 任意实数 | **`1.000000`** |
| 只能取 2 的幂（`16`） | **`0.5002`** |

⇒ **「只能取 2 的幂」把表示范围浪费掉将近一半**（`1 − 16/31.99 = 49.98%`）。
★ 这就是原文那句「软件层更强」的**全部内容**，也是本文把 `C.1` 写成判据的理由。

★★ **库内硬要求**（`130-quantization/numerical-formats/02-e8m0-and-mx.md:276` 逐字）：
「**凡是写「scale」这个词的地方，同时写出它的作用域（每张量 / 每通道 / 每 token /
每 32 个元素的块）与它的 bit 开销**」。本文一律写：

```
作用域 = per-tensor（每张量）     开销 = 每张量 1 个 fp32 = 32 bit（与元素数无关）
```

**这个开销是可算的**（`B.1`）：一次 fp8 训练步里有 **10** 个不同的张量被编码 / 缩放
（就是 §1.1 那张表的大小），于是

| 张量元素数 | 数据（8 bit/元素） | scale 开销（10 × 32 bit） | 相对开销 |
|---|---|---|---|
| `4096` | `32768 bit` | `320 bit` | `0.976562%` |
| `4096000` | `32768000 bit` | **`320 bit`（一个 bit 都不涨）** | `0.000977%` |

⇒ **元素数 ×1000 时开销不变、相对开销降 1000 倍** ——
这就是「per-tensor scaling 是**软件层**」在字节上的样子：**它不占元素位宽**。

### 1.3 scale 从哪来：两个策略，各自的代价都是原文点名的

★ **先声明一件事（本库 **T17**）**：下面这两个术语**在 fp8 论文（2209.05433）里各出现 0 次**。
论文自己逐字说（原始字节 1 次）：

> Details of the heuristics to select the scaling factors are **beyond the scope of this paper**

⇒ 它们的可引定义在**另一篇论文的附录**里。`[C]` **FP8-LM**（arXiv 2310.18313）附录 A.2 逐字：

| 策略 | 逐字定义 | 逐字代价 |
|---|---|---|
| **just-in-time** | `determining the scaling factor based on the maximum absolute value (amax) of the tensor being generated` | `often infeasible because it **necessitates multiple passes through the data**` |
| **delayed** | `selecting the scaling factor based on the maximum absolute values observed in a **certain number of preceding iterations**` | `necessitates the **storage of a history of maximum values** as additional parameters of the FP8 operators` |

★ FP8-LM 自己把这两个术语的出处标成 `(Nvidia 2022c)` —— 也就是 **NVIDIA Transformer Engine 那一层**。
⇒ **把 `just-in-time` / `delayed scaling` 写成 2209.05433 的内容，就是「把实现文档当规范」**（**P19**）。

### 1.4 delayed scaling 的溢出率：一个闭式 `1/(k+1)`

`delayed` 的 scale 由**前 `k` 步**的 `amax` 定：

```
s_t = max_finite / max(amax_{t−k..t−1})
```

**当前这一批的 `amax_t` 一旦比前 `k` 步都大，`amax_t · s_t` 就会超过 `max_finite` ⇒ 溢出。**
`amax` 独立同分布时，「`k+1` 个数里**最大的是最后一个**」的概率是 `1/(k+1)`（对称性，
不依赖具体分布）。本机实测（`B.2`，`U(0,1)` 抽 10 万个 `amax`）：

| `k` | 实测溢出率 | 闭式 `1/(k+1)` | 相对差 | 期望命中数 |
|---|---|---|---|---|
| `1` | `0.501265` | `0.500000` | `0.0025` | 50000 |
| `2` | `0.333357` | `0.333333` | `0.0001` | 33333 |
| `4` | `0.199738` | `0.200000` | `0.0013` | 19999 |
| `8` | `0.111319` | `0.111111` | `0.0019` | 11110 |
| `16` | `0.058319` | `0.058824` | `0.0086` | 5881 |

⇒ **`k` 越小、溢出越频繁**：**`k = 1` 时一半的步会溢出。**

### 1.5 为什么 fp16 的溢出处置搬不过来

`[C]` 2209.05433 §3.2 逐字（原始字节 1 次）：

> Weight update skipping (and reduction of the scaling factor) on overflows, as used by
> FP16 automatic mixed precision training [13], **is not a good choice for FP8** as overflows are
> much more likely due to the narrower dynamic range, **resulting in too many skipped updates**.

**「太多」是多少？** §1.4 给了上半段（溢出率 `1/(k+1)`）；下半段是**跳步的代价**。
本机造一个「突增批」（`x` 乘 25 ⇒ 梯度 `amax` 从 `1.673929` 涨到 `836.090347`，**499.5 倍**），
scale 由**前 8 步的正常批**定（`delayed`，`s = 34257.120286`）：

| 量 | 本机实测（`C.2`） |
|---|---|
| 突增批被**饱和**的元素数（`delayed` scale） | **`197`** |
| 同一个批用 `just-in-time` scale（`s = 68.585889`）时的饱和元素数 | **`0`** |
| 那一步 `dW1` 的偏差：`delayed` / `just-in-time` | `4.0604e+02` / `2.9834e+01` |

⇒ 然后就是两种处置的对照：

| 策略 | 突增那一批的更新量 |
|---|---|
| **跳步**（照搬 fp16） | **`0.000e+00`** —— 整步被丢干净 |
| **饱和**（`[C]` 的做法） | `1.674e+00` —— 走了一步，但被饱和扭曲 |

★ 原文给的处置是**饱和**（逐字：`Values that overflow are then **saturated to the maximum
representable value**`）。两者都是**近似**，只是一边丢信息、一边失真。

★ **一个口径区分**：本文的量化器只做**饱和**，**不对「格式的溢出语义是什么」下结论** ——
格式层面的溢出（`NaN` / `inf` / **按 torch 版本不同**）是位分配那一层的事，正文在
`130-quantization/numerical-formats/01-fp8-family.md` §0/§3，**此处回链**。

## 2. 执行映射

### 2.1 一次 fp8 训练步在软件栈上的样子

| 步骤 | 发生什么 | 谁负责 |
|---|---|---|
| 前向 | 权重与激活**按 per-tensor scale 编码**成 fp8，GEMM 的**输入**是 fp8 | 软件层（框架 / 手写 kernel 的前处理） |
| GEMM | 乘加在**更宽的累加器**里做 | ★ 硬件（Tensor Core）；累加器宽度的误差界回链 `nla/02-gemm-rounding-error.md` |
| 反向 | 梯度**按 per-tensor scale 编码**成 fp8 | 软件层 |
| 更新 | 解码回宽类型、在 fp32 主权重上更新 | ★ `01-type-selection` 的那四行 |

★ **两个具体组件**：`torch.float8_e4m3fn` / `torch.float8_e5m2` 这两个 **dtype**（它们的位布局与
**四个变体**的正文在 `130-quantization/numerical-formats/01-fp8-family.md` §0，此处回链），
以及**每张量一个 fp32 的 scale 标量**（本文 §1.2 把它算成 `32 bit/张量`）。

### 2.2 分工在数值上的后果（本机实测）

本机跑一步 fp8 训练（`code/03_fp8.py` 的 `A.3` / `A.3b` / `A.3c`）：

| 格式 | `eps` | 实测最大相对误差（正规区） | `eps/2`（舍入到最近的**硬上界**） |
|---|---|---|---|
| `e4m3fn` | `0.125000` | `0.058746` | `0.062500` |
| `e5m2` | `0.250000` | `0.111024` | `0.125000` |

★ 溢出侧另有一组判据（`A.3b`/`A.3c`）：`|x| ≥ 最大有限数` 时输出必须**精确饱和**到 `±最大有限数`
（五档输入，含 `1e30`），且输出**永不越界**。
⇒ **两组合起来才把量化器夹住**（下面 §4 ① 就是缺了这一组的结果）。

## 3. 实现与演进

| 阶段 | scale 从哪来 | 它换来了什么、付了什么账 |
|---|---|---|
| **无 scaling** | —— | 张量只要跨几个量级就用不了 fp8（这是 fp8 的 range 决定的） |
| **全局 loss scaling**（照搬 fp16） | 一个标量给所有梯度 | 沿用 fp16 的机制；代价是**溢出太频繁**（§1.4），跳步代价大 |
| **per-tensor + just-in-time** | 当前张量自己的 `amax` | 零溢出（本机实测 `C.2`：`just-in-time` 饱和数 **0**）；代价是**要多读一遍数据**（原文逐字） |
| **per-tensor + delayed** | 前 `k` 步的 `amax` 历史 | 不用多读；代价是**溢出率 `1/(k+1)`** 与一份历史存储（原文逐字） |
| **溢出处置：饱和** | —— | 不丢步；代价是那一步的梯度被**扭曲**（本机实测偏差 `4.0604e+02`） |

★ **演进的方向是「把 scale 的决策从格式搬到软件、把代价从丢步换成失真」。**

## 4. 踩坑与失败模式

**① 只查「正规区的相对误差」，对溢出区完全盲。**

- **现象**：量化器的最大相对误差 `≤ eps/2` 判据全过，但训练里偶尔出现一批"什么都不动"的更新。
- **成因**：`≤ eps/2` 那条只扫 `最小正规数 ≤ |x| ≤ 最大有限数`。把**饱和的钳位**从**舍入之后**
  挪到**舍入之前**（`rint(min(ax, MAX)/step)*step`），在 `|x|` 远大于 `MAX` 时 **`448/step < 1`**
  ⇒ 舍到 **0**。本机实测：100 万个样本里 **499211 个不同**、最大差 `448`，
  **而 `≤ eps/2` 那条判据与饱和计数那条全都过**。
- **规避**：**溢出区要单独下判据** —— `|x| ≥ 最大有限数` 时必须**精确饱和**到 `±最大有限数`
  （本机 `A.3b` 用五档输入含 `1e30`），并另加一条「输出永不越界」（`A.3c`）。
  ⇒ **「饱和」这件事有两个面：值对了没有、越界了没有；只查一面会漏掉整条路。**

**② 把「前向 E4M3、反向 E5M2」写成论文证明的。**

- **现象**：正文写「fp8 论文证明了前向用 E4M3、反向用 E5M2」。
- **成因**：论文自己的措辞只有 `The recommended use of ...`（一个建议），
  依据是**引用**别人的工作（`This is consistent with findings in [20, 16]`）。
- **规避**：把「这条分工是**引用来的**」写成断言（`A.1b`），正文如实写它的来源级别。

**③ 把 `just-in-time` / `delayed scaling` 写成 fp8 论文的内容。**

- **现象**：写「fp8 论文提出了 just-in-time 与 delayed scaling」。**两个术语在那篇里各出现 0 次。**
- **成因**：论文逐字说 `Details of the heuristics ... are beyond the scope of this paper`；
  那两个术语属 **NVIDIA Transformer Engine** 那一层（FP8-LM 把它们标为 `(Nvidia 2022c)`）。
  这是**把实现文档当规范**（**P19**）。
- **规避**：**引 FP8-LM 的附录 A.2**（那里有逐字定义），并在正文写明「这两个术语不在 fp8 论文里」。
  ★ 本程序的 `K_DELAYED` / `K_SWEEP`（历史长度）标为**本程序的选择**，因为**没有**可引的一手规则。

**④ 把「跳步」与「饱和」的代价搞反。**

- **现象**：认为「饱和」比「跳步」更糟（因为它改了数值）。
- **成因**：本机实测两者的**代价类型不同**：跳步把整步更新**精确地丢干净**（`0.000e+00`），
  饱和则**至少走了一步**（`1.674e+00`），代价是那一步的梯度被扭曲（偏差 `4.0604e+02`）。
  ⇒ 在 `k` 小、溢出频繁时，**丢掉整步**的累积代价更大（这正是原文 `too many skipped updates` 的意思）。
- **规避**：把两者**分开量化**（`C.2` / `C.2b` / `C.2d`），并**不**断言「饱和一定更好」——
  只说「`[C]` 推荐饱和、而跳步的代价是可量化的」。

## 5. 判据表

| 可机械判定的条件 | 结论 |
|---|---|
| 训练循环里要给一个张量选 fp8 编码 | 前向（权重 / 激活）用 `e4m3fn`、反向（梯度）用 `e5m2`；★ **依据是 `[C]` 的 `recommended use` 一句 + 引用**，不是证明 |
| 想给 per-tensor scaling 记一笔开销 | 作用域 = **每张量**、开销 = **32 bit/张量**（与元素数**无关**）；元素数 ×1000 时相对开销降 **1000 倍** |
| 想选 scale 的取值方式 | 软件层（任意实数）能把 `amax` **精确**摆到最大有限数上；**只能取 2 的幂**时最多只到它的 **50%**（本机 `0.5002`） |
| 想用 delayed scaling | 溢出率 **`≈ 1/(k+1)`**（本机五档 `k` 都对上，最大相对差 `8.6e-03`）；`k` 越小越频繁，`k = 1` 时**一半的步** |
| 溢出发生了怎么办 | **饱和**（`[C]` 的做法）：那一步照样走，但梯度被扭曲；**跳步**：整步 `0.000e+00` 丢弃，且 `k` 小时频繁发生 |
| 想量化一个 fp8 量化器好不好 | `最小正规数 ≤ |x| ≤ 最大有限数` 上的最大相对误差 `≤ eps/2`；**并且** `|x| ≥ 最大有限数` 时**精确饱和**到 `±最大有限数`；**并且**输出永不越界。★ 缺第二条会放过一个把大值舍成 0 的实现 |
| 想报「fp8 训练的结果」 | 必须写明是 **`simulated FP8`**（算术在宽类型上做，只有存储/输入被量化）—— 本机就是这一档 |
| 引 `just-in-time` / `delayed scaling` | 来源是 **FP8-LM 附录 A.2**（`[C]`）或 **NVIDIA Transformer Engine 文档**（`[B]`）；**不是** fp8 论文（那里各 0 次） |

> ★ 阈值（`code/03_fp8.py` 的 `D.3`/`D.4` 逐位钉住）：`TOL_Q_REL = 0`（`≤ eps/2` 是**硬上界**）、
> `TOL_OVERFLOW_LAW = 0.02`、**`MIN_HITS_AMAX = 200`**（小 n 门槛）、`TOL_RANGE_LOSS = 0.55`、
> `TOL_SAT_MIN = 100`、`K_SWEEP = (1, 2, 4, 8, 16)`、`SPIKE_FACTOR = 25.0`。
> ★ 把 `TOL_Q_REL` 放宽 ⇒ 判据退化成「差不多」；把 `MIN_HITS_AMAX` 去掉 ⇒
> `1/(k+1)` 那条会随样本数随机通过。

## 6. 小结与延伸

**一句话**：fp8 在训练循环里就是**「前向 E4M3、反向 E5M2」+「每张量一个实数 scale」+「溢出就饱和」**
三件事 —— 而这三件事里**两件是软件层的事**（格式本身不含 scaling），
**溢出的处置又与 fp16 相反**（因为 fp8 的溢出率是 `1/(k+1)` 这个量级）。

- **fp16 那一套（loss scaling / unscale 的时机）** ⇒ 见 `02-amp-execution`。
- **位分配、四个变体与按 torch 版本的溢出差异** ⇒
  `130-quantization/numerical-formats/01-fp8-family.md` §0（正文在彼，此处回链）。
- **scale 的其它粒度（每通道 / 每 token / 每 32 元素的块）** ⇒
  `130-quantization/theory/02-granularity.md`、`130-quantization/numerical-formats/02-e8m0-and-mx.md:276`。
- **累加器宽度** ⇒ `10-foundations/numerical-linear-algebra/02-gemm-rounding-error.md`。
- **主权重 / 优化器状态为什么仍要 fp32** ⇒ `01-type-selection`（上游，正文在本域）。

## 编译与运行

```bash
cd code
make run                        # 三个编号程序依次跑；本程序是 03_fp8.py
python3 03_fp8.py               # 只跑本程序
```

- 配套程序：**`code/03_fp8.py`**（编号与外键严格对应）。
- 依赖：`numpy`（`requirements.txt` 里**只有**它）。**不需要网络、不需要 GPU。**
- ★★ **格式参数是「输入」，不是「本程序推导的」**：`(最大有限数, 最小正规数, eps)` 三个数取自
  `specs.py` 的 `[C]` 常量（2209.05433 Table 1）。程序**不解释**它们是怎么从 8 位里分出来的。
  ★ 而且那三个值在 `specs.py` 里写成**十进制字面量**、幂次写在注释里 ——
  写成 `2.0 ** -6` 时 `mutate.py` 的 `_walk` 下钻不到，那四个值会**整体逃出变异分母**。
- ★★ **量化器只做饱和，不对格式的溢出语义下结论**（`NaN` / `inf` / 按 torch 版本不同是位分配的事）。
- 失败标记：`:: FAIL: mixed-precision`（本主题自定义，**不跨主题照抄**）。
- 程序的输出分四段：`A` 分工、scaling 的层属、量化器本身、恒等量化的阴性对照；
  `B` 代价模型（scale 的开销与 `1/(k+1)`）；`C` 边界与反例（2 的幂 scale 的范围浪费、
  突增批的饱和与跳步）；`D` 标记自检、口径钉住与 specs 钉住。

## 参考

**本机实测值 / 推导值**

- `code/03_fp8.py` 的 `make run` 输出：`python3 3.13.5` + `numpy 2.4.4`，macOS arm64，**无 GPU**。
  口径：`seed = 20260927`、`d = 32`、`m = 16`、量化器扫描 4096 个元素、
  `amax` 蒙特卡洛 10 万个、`k` 扫描 `{1, 2, 4, 8, 16}`、`SPIKE_FACTOR = 25`（`x` 乘 25）。
  **全部是推导值**（`simulated FP8`：算术在 fp64 上做）。
- ★ 计划 §0.7-D 里**没有** fp8 的行 —— fp8 的判据全部是本次新立的，来源见下（`[C]` 两篇）。

**外部来源**

- **[C]** P. Micikevicius et al., *FP8 Formats for Deep Learning*，arXiv **2209.05433**。
  **不确认同行评审**（判据是实测：abs 页 `Comments:` = **0** 次、`journal-ref` = **0** 次）。
  抓取（ar5iv）：`https://ar5iv.labs.arxiv.org/html/2209.05433`，2026-09-23，
  **85378 B**，`sha256[:16] = a24bcf90ca36746f`。
  逐字引用（原始字节各 1 次）：`The recommended use of FP8 encodings is`、
  `E4M3 for weight and activation tensors, and E5M2 for gradient tensors`、
  `This is consistent with findings in`、`Leaving the per-tensor scaling to software implementation`、
  `beyond the scope of this paper`、`is not a good choice for FP8`、`simulated FP8`、
  `saturated to the maximum representable value`。
  ⚠ **两个量法口径**：① 论文正文里 `57,344` **带千位逗号** ⇒ `raw.count("57344")` = **0**；
  ② `just-in-time` 与 `delayed scaling` 在本文里各 **0** 次（这是 T17 的实测依据）。
- **[C]** **FP8-LM**（arXiv **2310.18313**），**不确认同行评审**（abs 页 `Comments:` = 0）。
  附录 **A.2** 给出 `just-in-time scaling` 与 `delayed scaling` 的逐字定义（两个术语各 2 次）。
  抓取（ar5iv）：**266806 B**，`sha256[:16] = a184329665e9ad2b`。
- **[A]** / 本库：两个格式的三个参数（最大有限数 `448` / `57344`、最小正规数 `2^-6` / `2^-14`、
  `eps` `2^-3` / `2^-2`）—— 位分配与四个变体的完整说明在
  `130-quantization/numerical-formats/01-fp8-family.md` §0，本文只**取用**这三个数。
- **本库**：`130-quantization/numerical-formats/01-fp8-family.md` §0（位分配与四个变体）、
  `02-e8m0-and-mx.md:276`（「凡写 scale 必写作用域与 bit 开销」）、
  `130-quantization/theory/02-granularity.md`（其它粒度）、
  `10-foundations/numerical-linear-algebra/02-gemm-rounding-error.md`（累加器宽度）。

**上游契约**

- `01-type-selection` 是本文的 `prerequisites`：本文的每一个张量都落在它选定的那几行上。
- **边界声明跟着知识点走**：凡本文涉及位分配 / 四个变体的地方，正文在
  `130-quantization/numerical-formats/01-fp8-family.md` §0，本文**回链、不重写**；
  凡涉及 scale 的其它粒度，正文在 `130-quantization/theory/02-granularity.md` 与
  `02-e8m0-and-mx.md:276`，同样回链。

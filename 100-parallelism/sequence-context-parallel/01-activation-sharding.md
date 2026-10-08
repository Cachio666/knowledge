---
title: 序列并行切的是什么：张量并行没切到的那批逐 token 算子
slug: sequence-context-parallel-activation-sharding
domain: 100-parallelism
topic: sequence-context-parallel
level: 3
difficulty: intermediate
prerequisites:
  - 100-parallelism/tensor-parallel/01-column-and-row
  - 80-model-architectures/transformer/01-block-and-residual-stream
tags: [sequence-parallelism, tensor-parallelism, activation-memory, layer-norm, communication]
related:
  - 100-parallelism/tensor-parallel/03-saturation-and-limits
  - 100-parallelism/sequence-context-parallel/02-ring-attention
  - 100-parallelism/sequence-context-parallel/03-constraints
code:
  - code/01_activation_sharding.py
status: stable
updated: 2026-10-08
---

# 序列并行切的是什么：张量并行没切到的那批逐 token 算子

> **TP 把 attention 与 MLP 的矩阵乘按列/行切开了，但有一批算子它切不到** ——
> **LayerNorm、以及 attention / MLP 之后的 dropout**：它们**逐 token 独立**，
> 没有可切的矩阵维，于是在 TP 组内**被复制**（每台设备各存一份完整的激活）。
> **序列并行（SP）做的就是切掉这一批**：既然它们逐 token 独立，
> 那就**按序列维**切成 `t` 份。
> ★★★ **本文要钉死两件事**：① 它省的**恰好是** `10sbh`（`O(s)` 的那一项）；
> ② **把 SP 与 TP 混用之后，它【不引入任何额外的计算 / 通信 / 显存开销】** ——
> ★★ **而这句话是【论文的结论】，不是本库的推断**（`specs.SP_NO_OVERHEAD_QUOTE`）。

## 0. 背景与动机

先看 TP 切完之后剩下了什么。`tensor-parallel/01` 讲过：TP 把 attention 与 MLP 的
**矩阵乘**按列/行切开，每台设备只算自己那一段。但一个 transformer 层里还有一批算子
**没有矩阵维可切**：

- **LayerNorm**：对每个 token 的隐向量做归一化 —— **逐 token**，`s` 个 token 各自独立；
- **attention / MLP 之后的 dropout**：逐元素乘掩码 —— **逐 token、逐通道**。

⇒ 这批量**在 TP 组内被完整复制**：`t` 台设备各存一份**一样**的激活。
★★ 它们**几乎不吃算力**，却**吃显存** —— 论文的原文是
（`specs.SP_LN_DROP_QUOTE`，载体 = ar5iv `2205.05198` 正文）：

> `These elements do not require a lot of compute but demand a considerable amount of activation memory.`

★ **这就是 SP 存在的理由**：切它们的**显存**，几乎不动它们的**算力账**。

## 1. 原理

### 1.1 三式：无并行 / 只有 TP / TP + SP

论文（Korthikanti et al., `2205.05198`）把**每层激活的显存**（单位：元素数）写成三式。
记号：`s` 序列长度、`b` batch、`h` 隐维、`a` 注意力头数、`t` 张量并行度。

| 式 | 形态 | 出处（逐字，见 `specs.py`） |
|---|---|---|
| **式(1)** | `sbh(34 + 5as/h)` | `specs.SP_EQ1_PROSE_QUOTE`：`The above equation is for the case that no form of model parallelism is applied.` |
| **式(2)** | `sbh(10 + 24/t + 5as/(ht))` | `specs.SP_EQ2_PROSE_QUOTE`：`the per-layer memory required to store the activations reduces from Equation` |
| **式(4)** | `sbh(10/t + 24/t + 5as/(ht)) = (sbh/t)(34 + 5as/h)` | `specs.SP_EQ4_PROSE_QUOTE` |

★★★ **式(2) 与式(4) 只差一项**：`10`（在式(2) 里**不除以 `t`**）与 `10/t`（式(4)）。
那一项就是 §0 说的**被复制的那批算子** —— 原文逐字（`specs.SP_REPLICATED_QUOTE`）：

> `is due to these replicated operations and as a result they are not divided by the tensor parallel size`

★ 换句话：**式(4) == 式(1)/t** —— **SP 就是把【整层】激活按序列维切成 `t` 份，一个除法，没有别的。**

### 1.2 把这三式钉成【可验证的恒等式】

`code/01_activation_sharding.py` 做的**不是**「模拟器 vs 闭式」的对拍 ——
★ 三式都是**从引文抄来的闭式**，所以对拍无从谈起。它做的是把论文的**散文陈述**
变成**机器可核的代数**：

| 判据 | 恒等式 | 本机实测 |
|---|---|---|
| **`A.1`** | `t = 1` 时 **式(2) == 式(4) == 式(1)** | 两组 `(s,b,h,a)`，**相对差 `0.000e+00`** |
| **`A.2`** | **式(2) − 式(4) == `10·s·b·h·(1 − 1/t)`** | 六档 `t` × 两组，**相对差 `0.000e+00`** |
| **`A.3`** | **式(4) == 式(1)/t** | 六档 `t` × 两组，**相对差 `0.000e+00`** |

★★ `A.1` 不是本库的推断，**原文自己说了**（`specs.SP_EQ4_TAIL_QUOTE`）：
`The above equation is now Equation` … 1。⇒ ★★ **SP 在 TP 度 1 时没有收益** ——
没有可切的 TP 区域，序列并行只剩「把数据切成 1 份」。

★★★ **`A.2` 的右端【只有 `10sbh` 与 `t`】**：**不含 `a`，也不含注意力的那一项**。
⇒ 这是本文最要紧的结构性事实：**SP 省下的东西与注意力无关**（§1.4 会用它解释收益为何随 `s` 下降）。

> ★★★ **一处必须写明的自我限定**：`A.1`/`A.2`/`A.3` 是**恒等式核对** ——
> 它们**对转写错误敏感、对物理错误不敏感**。把 `10sbh` 抄成 `11sbh` 会被抓住；
> 但「SP 到底省不省」这种**物理**问题**不在它们守的范围内**。
> ⇒ **内容判据在 `C.1`（只切一半）与 `C.2`（切错轴）**，见 §4。

### 1.3 ★★★ 「不引入通信开销」是**结论**，而且它的前提是「与 TP 混用」

★★ 原文在引言里把结论写成一句（`specs.SP_NO_OVERHEAD_QUOTE`）：

> `our technique, which mixes both tensor and sequence parallelism, reduces the activation memory significantly without any additional compute, communication, or memory overhead.`

★★★ **注意那个前提：`mixes both`（混用）。** 裸着上 SP 是**有开销的**（`specs.SP_NAIVE_EXTRA_COMM_QUOTE`）：

> `These extra communications introduce overhead and will slow down the training.`

★ 因为序列并行区与张量并行区之间需要**转换**（前向 all-gather、后向 reduce-scatter）。
原文的处置是**把它合并进 TP 的那两个集合通信**（`specs.SP_FUSE_CONVERTER_QUOTE`）：

> `To avoid these extra communications, we combine these operations with the`

⇒ ★★★ **转换操作替代了 TP 原有的通信，不是在它之上再加一层** —— 所以混用之后**总量不变**。

> ★★★ **本库的一条纪律**：**「SP 不引入通信开销」是原文的【结论】，不得写成 C 类反例**
> （那会与原文相反）。★ `code/01_activation_sharding.py` 的 `D.9` 把这三句**成对**引住，
> 并在输出的那一行**明确标注**「这是原文结论，不是 C 类」。
> ★★ 本文的 `C` 类讲的是**激活显存**的两条错法（§4），**不碰通信那一句**。

### 1.4 ★★★ 序列越长，SP 的相对收益越小

`code/01_activation_sharding.py` 的 `B.1` 量**收益比** `式(2)/式(4)`。
★★ **必须两组 `a/h` 一起引**（只引一组会把下面这件事误读）：

| 组 | `(s, b, h, a)` | `a/h` | `t = 8` | `t = 16` |
|---|---|---|---|---|
| `LONG` | `(1024, 1, 64, 8)` | `0.1250` | **`1.1039`** | **`1.2226`** |
| `SHORT` | `(4096, 1, 256, 16)` | `0.0625` | **`1.0533`** | **`1.1142`** |

**两条读法**：

1. **同一个组内，`t` 越大收益越大**（`1.1039 → 1.2226`）—— 因为被切的那一项是 `10sbh/t`；
2. ★★★ **`s` 更长的那一组，收益【更小】**（`SHORT` 的 `s` 是 `LONG` 的 4 倍，收益却更低）。

★★★ **第 2 条有一个诱人的错误解释**：「是不是因为 `SHORT` 的 `a/h` 更小？」
—— **不是**，而且方向正好相反：按直觉 `a/h` 更小 ⇒ 注意力项更小 ⇒ 收益**应当更大**。
**两组一起看就排除了这个替代解释**（`B.2` 断言的就是这件事）。

**真正的成因**（`B.3` 量它）：`s` 翻倍时，

- **SP 省下的** `式(2)−式(4)` 变成 **`2.0000` 倍**（`O(s)`）；
- **注意力那一项** `5as²b/h` 变成 **`4.0000` 倍**（`O(s²)`）。

⇒ ★★★ **SP 切不到 `O(s²)` 的那一项**，所以**分母长得比分子快**。

> ★★ **这就是 `03` 的引子**：Megatron CP 文档把 SP 与 CP 的**分工**逐字分开
> （`specs.CP_VS_SP_QUOTE`；★★ **那句的正文在 [03-constraints](03-constraints.md) §0** ——
> 依 `CONTRIBUTING` §7「一个知识点只有一处正文，其他位置只放链接」，**此处不复制整句**）。
> ⇒ **CP 连 `O(s²)` 的注意力一起切** ⇒ **序列越长，SP 越不够用、CP 越必需**
> ⇒ [03-constraints](03-constraints.md)。

## 2. 执行映射

### 2.1 本机怎么算这件事

`code/01_activation_sharding.py` **不用 numpy** —— 它全是标量算术。
★ 这一点与本域其余主题不同，**必须写明**（本主题的 `02`/`03` 需要 numpy，`01` 不需要）。

| 角色 | 是什么 | 依据 |
|---|---|---|
| **被测** | 三条闭式 `eq1()` / `eq2()` / `eq4()` —— ★★ 系数是**函数体内的字面量**（论文的转写） | `specs.py` 的三条引文 |
| **参照** | 论文的散文陈述（三式的关系）—— ★★ 它**不是一个独立的算法** | 同上 |

★★★ **为什么系数写死在函数体里、而不是从 `specs` 读**：若两边都从同一个常量读，
那么改常量时**两边一起动**，恒等式仍然成立 ⇒ 改常量**没有可观察后果**
（一条**不过期的漏网**）。⇒ 现在的分工是：**闭式的系数在函数里**、
**`specs` 的常量只出现在差式的右端与 `D.6` 的钉住里**。
★ 本机实测：`mutate.py` 的 `81/81` 全部捕获、**一条也不漏**。

### 2.2 一条必须写明的限定语

> ★★ **本机的三个程序都是玩具规模的真跑**（`s ≤ 8192` 的**算术**，不是任何模型的实测）。
> ★★★ **`t_b = 2·t_f` 这类建模假设在这里【不存在】** —— 本文的一切都是**论文公式的代数**，
> **不是任何真实训练运行的测量**。★ 强制限定语：**「不是 NCCL，也不是真机」**。

## 3. 实现与演进

### 3.1 SP 在真实系统里怎么接上去

- **它必须与 TP 一起上**（§1.3）：`t = 1` 时 SP 没有收益（`A.1`），
  而**转换通信的合并**也是冲着 TP 的那两个集合通信做的；
- **它切的范围是「TP 没切到的那一批」**：LayerNorm 与 dropout；
- ★★ **它不改注意力**：注意力的 `O(s²)` 那一项**不在 SP 的账里**（§1.4）。

### 3.2 与 `tensor-parallel` 的分工

★ **两处必须分开写**（本文**一行都不重写**它们）：

1. **TP 的切法本身**（列切 `D_out` / 行切 `D_in`、`2+2` 次 all-reduce 的配对约束）→
   `tensor-parallel/01`、`tensor-parallel/02`、`tensor-parallel/03`；
2. **TP 的通信账**（通信发生在**每一层内部**、通信量 ∝ **激活**）→
   `10-foundations/performance-models/02-amdahl-and-scaling.md:223`。

⇒ ★★ 本文只写**它没写的**：**TP 切完之后剩下的那一块有多大、序列并行把它切成什么样**。

## 4. 踩坑与失败模式

### 4.1 ★★★ 「只切一半」：SP 只切了 LayerNorm 的那 `4sbh`

- **现象**：形状、类型、算术**全对**，程序**不报错**；
- **成因**：实现者以为「非 TP 区」只有 LayerNorm，忘了 attention / MLP 之后的 dropout；
- **后果**：差式变成 `4sbh(1−1/t)`，而正确值是 `10sbh(1−1/t)` ⇒ **账算少了一半**；
- **本机实测**（`code/` 的 `C.1`）：`s=1024, b=1, h=64, a=8, t=8` 时，
  完整 SP 的差式 `5.734400e+05`、只切 LN 的 `2.293760e+05`，
  与 `10sbh(1−1/t)` 的相对差 **`6.0e-01`**（≫ 阈 `1e-12`）⇒ **`A.2` 当场不成立** ✓
- **规避**：★★★ **唯一的信号是恒等式** —— 这就是 §1.2 那三条 A 类的用处。

### 4.2 ★★★ 「切错轴」：按隐维 `h` 切，而不是按序列维 `s` 切

- **现象**：同样**不报错**，只是**省得远不够**；
- **成因**：注意到 `h` 出现在 `34sbh` 里，就沿 `h` 切；
  ★★★ **但 `5as²b` 里【没有 `h`】** ⇒ 按 `h` 切**根本切不到它**，它**整项留全**；
- **本机实测**（`C.2`）：`s=1024, t=8` 时按 `s` 切得 `5.521408e+06`、按 `h` 切得 `4.222157e+07`，
  缺口 **`3.670016e+07`** —— ★★ 它**恰好**等于 `5as²b(1 − 1/t)`；
- ★★★ **缺口随 `s²` 增长**：`s` 翻倍 ⇒ 缺口变 **`4.0000` 倍**（实测四档：`9.18e6 → 3.67e7 → 1.47e8 → 5.87e8`）；
- **规避**：原文逐字给了方向（`specs.SP_SEQ_DIM_QUOTE`）：
  `This characteristic allows us to partition these regions along the sequence dimension` ——
  ★ **是【序列】维，不是隐维**。

### 4.3 ★ 把 SP 的收益当成「与注意力有关」

- **现象**：看到收益比只有 `1.1~1.2`，就以为「SP 没什么用」；
- **成因**：忘了 `5as²b/h` 是 `O(s²)`、而 SP 省下的 `10sbh` 是 `O(s)`（§1.4）；
- **规避**：**引用收益比时必须带上 `(s, b, h, a, t)`** —— 它不是一个数，是一族数。

### 4.4 ★★ 一条**不该有**的「C 类」

★★ **不要把「SP 引入通信开销」写成反例** —— 原文的结论正好相反（§1.3）。
★ 真正的边界是：**裸 SP 有开销、与 TP 混用之后没有**；两句**必须成对引**。

## 5. 判据表

| 可机械判定的条件 | 结论 |
|---|---|
| 看到「SP 省了多少显存」 | **必须带上 `(s, b, h, a, t)`** —— 收益比是一族数，不是常数 |
| 报出一个收益比 | ★★ **两组 `a/h` 一起引**（`0.1250` / `0.0625`），否则会把「随 `s` 下降」误读成「随 `a/h` 下降」 |
| `t = 1` | 式(2) == 式(4) == 式(1) ⇒ **SP 无收益**（原文自己说的，`specs.SP_EQ4_TAIL_QUOTE`） |
| 要验证「SP 省的是哪一项」 | 算 **式(2) − 式(4)**：它 **== `10sbh(1 − 1/t)`**（实测相对差 `0.000e+00`），**不含 `a`** |
| 序列变长 | ★★★ **SP 的相对收益【下降】**（`O(s)` vs `O(s²)`）；★ 要压注意力那一项，得换 **CP**（`03`） |
| 看到「SP 不引入通信开销」 | ★★ **它是原文的结论**（前提是**与 TP 混用**）；★ **不得**写成 C 类反例；★ 裸 SP 是**有**开销的 |
| 想让 SP 省得更多 | **不成立**：它切的那一项（`10sbh`）**与 `s` 成线性**，注意力那一项它**切不到** |
| 实现里「非 TP 区只想到 LayerNorm」 | **少了 dropout** ⇒ 差式少一半 ⇒ 恒等式会失败（`C.1`） |
| 沿**隐维**切激活 | ★★★ **切不到 `5as²b`**（它没有 `h`）⇒ 缺口 `5as²b(1−1/t)` **随 `s²` 增长**（`C.2`） |
| 拿三式当「实测」 | **不合格**：它们是**论文公式的代数**；本库只把它们核成**恒等式**，不声称任何真实运行的数 |

## 6. 小结与延伸

**一句话**：**TP 切完还剩一批逐 token 的算子（LayerNorm / dropout）没被切** ——
**SP 就是按序列维把它们切掉**，省下的**恰好是 `10sbh`**（`O(s)`）；
★★ **而与 TP 混用之后，它不引入额外的计算 / 通信 / 显存开销**（**那是论文的结论**）。

**三条可独立复用的结论**：

1. **式(2) − 式(4) == `10sbh(1 − 1/t)`**（实测相对差 `0.000e+00`，六档 `t`）——
   ★★ 右端**不含 `a`** ⇒ **SP 的收益与注意力无关**；
2. ★★★ **序列越长，SP 的相对收益越小**（`s` 翻倍 ⇒ 省的 ×2、注意力项 ×4）；
3. ★★ **「不引入通信开销」有前提**：**与 TP 混用**；★ **裸 SP 有开销**，
   原文靠**把转换合并进 TP 的集合通信**消掉它。

**延伸**：

- ★★ **CP 的机制**（跨设备的块统计量重缩放）→ [02-ring-attention](02-ring-attention.md)；
- ★★★ **SP 与 CP 的分工、`cp` 的约束、causal mask 跨块** → [03-constraints](03-constraints.md)；
- **TP 的切法与配对约束** → `100-parallelism/tensor-parallel/01`、`02`、`03`；
- **本域不写、只回链的正文**：
  · **「在线 softmax」的 `(m, l)` 更新公式本身** →
    `40-kernels/softmax-normalization/02-online-softmax.md`（★ 那篇逐字自述「本文是 `(m, l)` 更新公式的正文」）
    与 `40-kernels/attention/`。★★★ **本文**（以及本主题的另外两篇）**一律不使用「online softmax」这个词** ——
    本域写的是**跨设备**的块统计量重缩放，术语照 Ring Attention 原文（**置换不变性** + **统计量重缩放**）；
  · **`S×S` 矩阵的访存账**（FlashAttention 的字节数收益）→
    `40-kernels/attention/03-flash-attention-backward.md` 与 `00-meta/glossary.md` 的 `flash attention` 条；
  · **位置编码的外推**（PI / NTK / ALiBi、混叠周期）→
    `80-model-architectures/positional-encoding/02,03`；
  · **集合通信原语的算法与复杂度** → `110-communication/collectives`（**未建成**）与
    `10-foundations/performance-models/02-amdahl-and-scaling.md:209`；
  · **TP 每层 `2+2` 次 all-reduce** → `tensor-parallel/03` 与
    `10-foundations/performance-models/02-amdahl-and-scaling.md:417`
    （★ **不是 `:223`** —— 那一行是**纯 TP 段**）；
  · **重算换显存**的机制 → `90-training/autograd-and-backward/02-gradient-checkpointing.md`。

## 参考

- Korthikanti et al., *Reducing Activation Recomputation in Large Transformer Models*,
  arXiv:2205.05198，**§4.1 / §4.2.1 / §4.2.2** —— 式(1)/(2)/(4)、`10sbh` 的归属、
  「混用后无额外开销」的结论。
  ★★ **同行评审：是** —— 依据是 **MLSys 2023 会议论文索引**
  （`proceedings.mlsys.org/paper_files/paper/2023`，抓取日 2026-10-08）。
  ★★★ **不是** arXiv abs 页：该页上 `Comments` 字段**不存在**，
  `MLSys` / `ICLR` / `Accepted` / `Proceedings` / `Conference` 各 **0** 次
  ⇒ **abs 页本身给不出评审依据**。
  ★ 引文载体 = `https://ar5iv.labs.arxiv.org/html/2205.05198` 正文，抓取日 **2026-10-08**，
  逐条在**原始字节**上 `count()` = **1**。
- NVIDIA Megatron-LM 仓库文档 `docs/user-guide/features/context_parallel.md`
  （**来源级别 [B]**：某实现的官方文档，**不是规范**；★ 它是**活文档**，无版本号
  ⇒ 引用写明抓取日 2026-10-08、**不做行号锚点**）—— §1.4 末尾那句 **SP / CP 分工**的逐字。
- 本仓库 [02-ring-attention](02-ring-attention.md) —— CP 的机制；
  [03-constraints](03-constraints.md) —— SP 与 CP 的分工与 `cp` 的约束；
  [tensor-parallel/01](../../100-parallelism/tensor-parallel/01-column-and-row.md) —— TP 的切法（本文不重写）。

## 编译与运行

```sh
cd code && make run                 # 三个编号程序，任一断言失败即非零退出
python3 01_activation_sharding.py   # 只跑本文的配套程序（★ 直接跑它**不需要 numpy**）
make 01_activation_sharding         # ★★ 这条**仍要求 numpy** —— 它先走 `check-env`（缺则 exit 1）
```

★ 本机的运行条件：`python3 3.13.5`，macOS arm64，**无 GPU、无多卡、无 NCCL、无真实网络**。
★★ **本文的配套程序不用 `numpy`**（纯标量算术）—— 但**同主题的 `02`/`03` 要**，
所以 `requirements.txt` 与 `Makefile` 的 `check-env` 按**并集**要求它；
★★★ **那条并集要求对 `make` 的每一个入口都生效**（`$(SCRIPTS): check-env`）——
⇒ **「不需要 numpy」只对 `python3 01_activation_sharding.py` 成立**，对
`make 01_activation_sharding` **不成立**（★ 本机实测：挡掉 numpy 后跑那条 `make`，
`!! 缺 numpy` / `make: *** [check-env] Error 1`、`rc=2`，`01` 一行都没跑）。
★ 强制限定语：**不是 NCCL，也不是真机** —— 本文的一切数字都是**论文公式的代数**，
**不得**支撑关于真实训练运行的结论。
★ 本文的判据**不依赖 BLAS**（是标量算术）⇒ 换机器不会因 BLAS 路径不同而误报。

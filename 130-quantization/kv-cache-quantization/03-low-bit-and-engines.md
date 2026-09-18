---
title: 低比特 KV 与各家引擎把 scale 放在哪一维
slug: kv-lowbit-and-engines
domain: 130-quantization
topic: kv-cache-quantization
level: 3
difficulty: advanced
prerequisites:
  - 130-quantization/kv-cache-quantization/02-kivi-asymmetry
tags: [kv-cache, 1-bit, qjl, kvquant, vllm, llama.cpp, tensorrt-llm]
related:
  - 130-quantization/kv-cache-quantization/01-kv-cache-basics
  - 130-quantization/ptq/02-gptq
code:
  - code/03_lowbit_engines.py
status: stable
updated: 2026-09-18
---

# 低比特 KV 与各家引擎把 scale 放在哪一维

> 往下压到 1-bit 会撞上一堵墙：**1→2 bit 一步值 12 dB**，而 1-bit 本身在合成张量上只剩 4.41 dB。绕开这堵墙的办法是引入额外结构（QJL 的 JL 投影、KVQuant 的稀疏 outlier、CQ 的通道耦合）。而在这之前，得先把「2-bit」这个词拆开——**载荷 2 bit 与整块 4.00 bit 不是同一个量**。

## 0. 背景与动机

[01](01-kv-cache-basics.md) 把账算清楚了，[02](02-kivi-asymmetry.md) 把粒度选对了。现在的问题只剩一个：**压到多低？**

KVQuant 的表把可选项摆得很直（LLaMA-7B / WikiText-2 / 128K，逐字读到 Table 1）：

| 配置 | PPL | KV cache (GiB) |
|---|---|---|
| baseline（fp16） | 5.68 | 64.0 |
| int4 | 5.98 | 16.0 |
| **KVQuant-4bit** | 5.72 | 16.0 |
| **KVQuant-3bit-1%** | 5.75 | 13.3 |
| **KVQuant-2bit-1%** | **6.01** | 9.3 |

**从 64.0 GiB 压到 9.3 GiB 是 6.9×，而 PPL 只从 5.68 涨到 6.01（+0.33）。** 那再往下呢？1-bit 就是 4 bit 的四分之一——如果它成立，同样的显存能装四倍的上下文。

答案是**不行，除非引入额外技巧**。本机的合成张量上（§5 表 2），K 的 per-channel 1-bit 只有 **4.41 dB**——**信号与量化噪声同量级**。而 2-bit 立刻跳到 16.41 dB。**1→2 bit 这一步值 12 dB**，是整个比特轴上最陡的一段。

先看具体的对照场景：**KVQuant-2bit-1% 用 9.3 GiB 换 PPL +0.33；如果 1-bit 能成立，4.6 GiB 就够。** 那 4.6 GiB 换来的是 PPL 崩到三位数还是四位数——这就是本篇要回答的。

## 1. 原理

### 第一步：「2-bit」到底是多少 bit/元素——三种记账口径

一个量化组要存的东西不止载荷：还有 **scale** 和（非对称时的）**zero-point**。以「4-bit」为例，业界同时存在三种记账口径：

| 口径 | 元数据 | G=128 时的整块 | 谁在用 |
|---|---|---|---|
| **纯载荷** | 0 | 4.00 bit | KVQuant 论文的 `Avg. Num. Bits` 列（`KVQuant-4bit` 写 **4.00-4.02**，逐字读到） |
| **只算 scale 一份** | 32/G | 4.25 bit | 对称量化（`zero = 0`，不用存） |
| **scale + zero 都算**（本仓库） | **64/G** | **4.50 bit** | 非对称量化（KIVI 的 `s_X` 与 `z_X` 各一个 fp32） |

**同一个「4-bit」，整块宽度是 4.00 / 4.25 / 4.50 三个数。** 本仓库一律按第三种（`64/G`）记账，并在 `specs.py` 的注释里把口径写死——**换一个口径，下面整张表都要重算**。

| 配置 | 载荷 | 元数据（64/G） | **整块** | 相对 fp16 |
|---|---|---|---|---|
| fp16 | 16.0 | 0.00 | 16.00 | 1.00× |
| fp8（per-tensor 标量） | 8.0 | 0.00 | 8.00 | 2.00× |
| int4，G=32 | 4.0 | 2.00 | **6.00** | 2.67× |
| int4，G=128 | 4.0 | 0.50 | 4.50 | 3.56× |
| int2，G=32 | 2.0 | 2.00 | **4.00** | **4.00×** |
| int2，G=128 | 2.0 | 0.50 | 2.50 | 6.40× |

**「2-bit KV cache 是 8× 压缩」这句话在这张表下是假的**——G=32 时实压缩是 **4.00×**。而 G 从 128 缩到 32（粒度更细、更准）的代价是**元数据从 0.50 涨到 2.00 bit/元素；int4 的整块从 4.50 涨到 6.00，其中元数据占比从 0.50/4.50 = 11% 涨到 2.00/6.00 = 33%**。这就是 §0.8 三角上那条边：**granularity vs scale 开销**。

> ★ **这张表顺手解掉了引擎那一节的一个疑团。** SGLang 报 `BF16→FP8 2.00×` 与 `BF16→FP4 3.56×`。用位宽算术验：
> - FP8：`16 / 8.00 = 2.000` ✓
> - FP4：`16 / 4.50 = 3.556` → 两位小数 **3.56** ✓ **前提是「4 bit 载荷 + G=128 的 64/G 元数据」**
>
> **两个数都对得上，而后者反过来印证了「SGLang 的 FP4 KV 用 G=128」。** 若按 G=32（整块 6.00）算是 2.67×、按纯载荷算是 4.00×，两个都对不上 3.56。

### 第二步：1-bit 的两个量化器给出完全不同的东西

「1-bit」这个词**不完整**——它必须连量化器一起说：

| 口径 | 公式 | 可用码字 | b=1 时 |
|---|---|---|---|
| **非对称 min-max**（KIVI 用的） | `s = (max−min)/(2^b−1)`、`z = min` | `2^b` 个 | **2 个**（0 / 1） |
| **对称 restricted-range**（多数 RTN） | `qmax = 2^(b−1) − 1` | `2·qmax + 1` 个 | **0 个非零码字** |

**对称的 1-bit 在 restricted-range 约定下 `qmax = 2⁰ − 1 = 0`——一个正值都表示不了，重建结果恒为 0。** 本机实测（§5 表 3）：它的 SQNR 恰为 **0.00 dB**——**不是负无穷，而是「误差能量 == 信号能量」**（分子分母都是 `Σx²`）。同一张张量上非对称 1-bit 是 4.41 dB。

> ⚠ **这不是一个文字游戏。** 「1-bit」在两边指的级数差一整档（2 级 vs 0 级），而**两个都叫「1-bit 量化」**。引用任何 1-bit 方案时必须写明它用的是哪一支。

### 第三步：断崖在哪一段

本机在合成 K/V 上把整条比特曲线量了出来（§5 表 2）：

```
K（per-channel，治列）  1 bit →  4.41 dB
                       2 bit → 16.41 dB     ← 一步 +12.00 dB
                       3 bit → 23.83 dB     ← +7.42
                       4 bit → 30.45 dB     ← +6.62
                       8 bit → 55.07 dB
```

**每 bit 的斜率在 2→4 段是 7.02 dB/bit，4→8 段是 6.15 dB/bit**——单调递减，但**1→2 这一步的斜率是 12 dB/bit，整条曲线上最陡的一段**。

> ⚠ **判据必须是「每 bit」，不能拿两段的总增量比。** 本程序的第一版就写成了「2→4 的增量（14.03 dB）大于 4→8 的增量（24.62 dB）⇒ 收益递减」——**两个分母不同（2 个 bit vs 4 个 bit），结论正好相反**。这是 P9 的一种：两个数都对，但没有可比性。改成斜率后是 7.02 vs 6.15。

### 第四步：为什么 1-bit 需要额外结构

机制上，1-bit 的失败是**两个原因叠加**：

1. **级数太少。** 非对称 1-bit 只有 2 个码字。任何分布在这个格点上都会丢掉大部分形状信息。
2. **量化噪声不再与被量化值成比例。** 相对误差 `|x−x̂|/|x|` 在小幅值处爆炸，而那些位置恰恰是注意力分数区分度所在。

绕开的路线因此都是**引入载荷之外的结构**：

| 路线 | 引入什么结构 | 代表 |
|---|---|---|
| **JL 投影** | 用一个随机正交投影把 K 映射到低维，在低维上做 1-bit | QJL |
| **稀疏 outlier** | 把 1% 的显著通道单独存成稀疏格式，其余压到极低 | KVQuant 的 dense-and-sparse |
| **通道耦合** | 让多个通道共享量化决策，用耦合补回信息 | CQ |
| **通道混合（旋转变换）** | 用正交变换把 outlier 摊平到所有通道再量化 | QuaRot / SpinQuant 一类 |

**它们的共同点：都不是「量化器变好了」，而是「给量化器喂了更难被 1-bit 毁掉的数据」。**

QJL 那一支的定量结论很好看：**定理 3.6** 说每位 key 只需 `m ≈ ε⁻² log n` bit——**与嵌入维度无关、只随序列长度对数增长**。另一个关键细节是它的**载荷口径** **"3 bits per FPN"**（它自己的算式 16/3 ≈ 5.33，与本文 §1 那张表的记账口径不同），并把「zero overhead」定义为**不需要为每个 block 存 zero-point 和 scale**（论文称这些常数在 block 量化下要额外占 **1–2 bit/数**）。

> 🔴 **「zero overhead」不是「零计算开销」。** 它省掉的是**元数据比特**，而它引入的成本写在论文自己的 limitation 里：需要 JL 随机投影矩阵；**query 侧必须在算 attention 时做同样的（不量化的）JL 投影**——这就是它自称 "asymmetric" 的地方；还需要在 prompt 阶段识别 outlier 通道并各跑一套量化器。**把「zero overhead」读成「零成本」，就漏掉了整整一块计算。**

## 2. 执行映射

**（1）引擎决定了 scale 能放在哪一维，而这一维决定了哪些方案能落地。** 下表是 §0.7.F 的引擎矩阵（**引用值**，除 vLLM 的取值集合外，本仓库未独立核实）：

| 系统 | 写侧 | 读侧 | scale 粒度 | 支持 per-token |
|---|---|---|---|---|
| **vLLM**（FP8） | 融合在 `reshape_and_cache_flash_kernel`（**独立于 attention 的一个 kernel**） | attention kernel 内 | per-tensor 或 per-head | **否** |
| **SGLang / TRT-LLM MHA** | 原本 4 个独立 kernel，SGLang 融合为 1 个 | attention kernel 内 | 标量 scale | **否** |
| **KIVI** | K：每满 R 个 token 做一次 group per-channel；V：出队时逐 token | **dequant 与 matmul 在 tile 级融合** | K per-channel / V per-token，**动态** | **是** |

⚠ **vLLM 是 quantize-on-write，但「融合」指的是量化与写 cache 融合在同一个 kernel 里，不是把量化融进 attention。** 写入是一个独立于 attention 的 kernel。

**（2）vLLM 的 `CacheDType`：18 项，逐字读到，且版本相关。**

来源：`vllm/config/cache.py`（`main` 分支，2026-09-18 逐字读到）：

```
auto, float16, bfloat16, fp8, fp8_e4m3, fp8_e5m2, fp8_inc, fp8_ds_mla,
nvfp4_ds_mla, turboquant_k8v4, turboquant_4bit_nc, turboquant_k3v4_nc,
turboquant_3bit_nc, int4_per_token_head, int8_per_token_head,
fp8_per_token_head, nvfp4, nvfp4_4over6
```

**这是 18 项，不是老版本文档里的 7 项。** 两者都真实，**差别在版本/分支**——所以引用时必须写明是哪一个。这份清单里有三个名字带 `per_token_head`，它们与裸 `fp8` 是**不同的粒度**，不是别名。

**（3）llama.cpp 的 K/V 可以独立设类型。** `--cache-type-k/-ctk` 与 `--cache-type-v/-ctv`，允许值（**9 项**、默认 `f16`）：`f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1`。**不含** `q8_1`、`q2_K…q6_K`、`tq1_0/tq2_0`、`mxfp4`——「llama.cpp 支持 mxfp4 KV」这句话在这份列表下是假的。（**计划基线，本仓库未独立核实。**）

两条实现约束值得单独记：**量化 V cache 需要 flash attention**（`flash_attn == false && type_v != F16` 会报错），**K 可以在 FA 关闭时量化**；MLA 类模型（DeepSeek V2/V3/R1、Kimi K2）**直接拒绝 `K != V`**。另有一个静默陷阱：**CUDA 默认构建只为 `f16/q4_0/q8_0/bf16` 编了 FA kernel**，其他类型会**静默回落到 CPU 后端**导致 prefill 极慢，除非加 `-DGGML_CUDA_FA_ALL_QUANTS=ON`。

**（4）容量的账要按整块等效宽度算，不能按载荷算。** 引擎文档里的「token 容量比」如果只按载荷算，会与实现差出 12% 以上。SGLang 报的 `BF16→FP8 2.00×` 与 `BF16→FP4 3.56×` **在本仓库的记账口径下都能精确复现**（16/8 与 16/4.50 = 3.556），而后者**同时印证了它的 FP4 KV 用 G=128**——见 §1 第一步。

## 3. 实现与演进

**第一步：把载荷压到 4-bit，用 group-wise。** FlexGen 那一代：K 与 V 都用 per-token group-wise（G=128 或 64）。4-bit 在这条路上基本无损。

**第二步：2-bit + 正确的粒度轴。** [02](02-kivi-asymmetry.md) 的全部内容：K per-channel / V per-token。2-bit 在这一步变得可用——KVQuant 的 Table 16 里 `KVQuant-2bit-1%` 在 Llama-2-7B 上是 **5.41**（基线 5.12）。

**第三步：往 3-bit / 2-bit 走时，靠**额外的结构**而不是更好的量化器。KVQuant 的三个技巧（per-channel K 3.82 + pre-RoPE 0.82 + 稀疏 outlier 0.19）、QJL 的 JL 投影、CQ 的通道耦合，都属于这一类。

**CQ 那条线给出了 1-bit 的一个可用点**（引用值，arXiv:2405.03917v1）：Llama-2-7B 上 **KVQuant-1b = NaN**，而 **CQ-8c8b（1.00 bit/元素）= 7.75**、**CQ-8c10b（1.25 bit）= 6.25**。**同样的 1 bit，加不加通道耦合是「NaN」与「7.75」的差别。**

**第四步：1-bit 全量 KV 量化在没有额外技巧时不可用。** 本机在合成张量上把这件事量了出来（§5 表 2/表 3）：仅 K → **4.41 dB**、仅 V → **−7.98 dB**。**两个方向都掉到 10 dB 以下，即信号与量化噪声同量级。**

> ⚠ **本机给不出 PPL 数字**（无权重、无 GPU）。上面这条只支撑「**量化器层面**不可用」，不支撑「模型层面 PPL 是多少」。模型层面的数字只能引用。

## 4. 踩坑与失败模式

### 4.1 ★ 「zero overhead」被读成「零计算开销」

**现象**：QJL 的宣传语是 "zero overhead"，很容易被复述成「没有额外计算成本」。

**成因**：它省掉的是**元数据比特**（每个 block 的 zero-point 与 scale，论文说这些在 block 量化下要额外占 **1–2 bit/数**），而引入的成本写在论文自己的 limitation 里：JL 随机投影矩阵、**query 侧必须做同样的（不量化的）JL 投影**、prompt 阶段识别 outlier 通道。

**规避**：**引用时把「zero」后面那个名词写出来**——「zero **metadata** overhead」。判据：凡是「zero X」这种说法，先问 X 是什么。

### 4.2 ★ 两套 Llama PPL 表的基线不同（5.12 vs 5.47 vs 5.68），不得并列

**现象**：写文档时想把 KVQuant 的 `Llama-2-7B: 5.16` 和某篇复现工作的 `Llama-2-7B: 5.47` 放进同一张表对比——**两边都是 Llama-2-7B / WikiText-2**。

**成因**：**同一个模型、同一个数据集，harness 不同，基线就不同。**

| 基线 | 出处 | 档位 |
|---|---|---|
| **5.12** | KVQuant arXiv:2401.18079v4 **Table 16**（Llama-2-7B 列） | **逐字读到** |
| **5.68** | KVQuant 同一篇的 **Table 1**（**LLaMA-7B**，不是 Llama-2） | **逐字读到** |
| **5.47** | QuaRot / AURA 等一批复现工作与多个二手汇总页 | **中置信度**：只读到检索摘要的一致结论，**未逐字读原文** |

**同一篇论文里两套表（Table 1 与 Table 16）就是两个基线**，因为模型不同；而 5.47 与 5.12 的差别在评测代码 / 上下文长度 / tokenizer 拼装。

**规避**：引用值必须带 **`来源 / 表号 / 基线`** 三元组；**跨论文并列的另加显式标注「不同基线，不得并列比较」**。本仓库的 `specs.py` 把三组数字分成三个常量（`KVQUANT_TABLE1_LLAMA7B` / `KVQUANT_TABLE16_LLAMA2_7B` / `REFERENCE_BASELINE_LLAMA2_7B_WIKITEXT2`），**改名就是为了让混用可见**。

### 4.3 ★ FP8 KV 的硬件支持矩阵在来源间冲突——**不下断言**

**现象**：查「FP8 KV cache 支持哪些架构」，两处来源打架：一处列 Blackwell / Hopper / Ada / **Ampere**，另一处称 **Ampere 没有 FP8**。

**成因**：两者**可能都真实**——「**存储 dtype 支持**」（把 8 bit 存进显存）与「**硬件加速**」（有 FP8 tensor core 做乘加）是两件事。Ampere 做不了后者，但完全可以把 KV 存成 fp8 再反量化回 fp16 计算。

**规避**：**本仓库没有读到官方 support matrix**，所以 `specs.py` 里**不落任何「FP8 KV 支持哪些架构」的常量**，正文按「未解决」写。**不落成常量就不会被误引**——这与 P6 的对策同源。同理，TensorRT-LLM 那一侧能确证的只有：`NVFP4 KV cache 必须先用 ModelOpt 离线量化、且必须同时是 FP8 权重/激活量化`、`NVFP4 KV cache 仅 Blackwell`、`INT8 KV cache 标记为 "Calibration Required: Yes"`。

### 4.4 ★ 引用一个「来源未定位」的 1-bit 数字

**现象**：有些材料给出「仅 K 量化到 1-bit → PPL 17.17；仅 V → 177.13；K&V → 620.08（FP16 = 5.68）」，看起来是一组很干净的 channel-coupling 消融。

**成因与处置**：这组数出自计划文件 §0.7.F 记为「某 channel-coupling 消融」，**但未给出 arXiv 编号**。本实现者 2026-09-18 对 **KVQuant arXiv:2401.18079v4 全文逐字检索这三个数，命中 0 次**——**来源未定位**。

**规避**：按 CONTRIBUTING §2，**「引用值」这个标签隐含「来源存在」**。这三个数**不能标成引用值**，因为定位不到来源。本仓库的处置是：

- **正文不用它们**（`specs.py` 里也没有对应常量，程序 §C4 把「没有」本身断言下来）；
- 替换成**本机可复现的对应物**：合成张量上 1-bit 的 SQNR（§5 表 2）；
- 在报告里把它列为「被推翻的条目」。

> ⚠ 注意 `5.68` 这个基线**本身是对的**——它就是 KVQuant Table 1 的 LLaMA-7B baseline（逐字读到）。**所以这组数字「看起来有出处」**：基线对、量级也合理。**这正是 P9 最危险的地方——连带正确的数字一起，看起来很可信。**

## 编译与运行

```bash
cd code
make run                       # 自动探测装了 torch ≥ 2.8 的解释器，依次跑 01–03
make 03_lowbit_engines         # 只跑本文对应的程序
make verify                    # 默认 python3，跑 verify.py（**不 import torch**）
```

本篇对应的编号程序是 **`code/03_lowbit_engines.py`**（真 torch；B 类断言在程序内部，失败即非零退出）。

⚠ **本篇的引擎那一节全部是引用值**——**本机一根 kernel 都测不了**（无 GPU、无 CUDA）。程序只测**量化器的数学**（比特账、1-bit 断崖、对称/非对称的级数），并把引擎表的**内部自洽性**（取值集合的项数、容量比的位宽可复现性、两张表的基线不同）写成断言。**§5 表 4/表 5 是引用值的整理，不是本机实测。**

## 5. 实测数据

**测量条件**：CPU **Apple M1 Pro（10 核）**；macOS；python **3.12.9**；**torch 2.11.0**。表 1–3 的每一格在 **2.6.0 / 2.7.1 / 2.8.0 / 2.11.0 / 2.13.0** 五个解释器上**逐格一致**（本仓库实测）。**CPU-only，无 GPU 数字。**表 4/表 5 是**引用值的整理**，不是本机实测。

**表 1：两笔比特账——载荷 vs 整块等效宽度**（`03_lowbit_engines.py` §B1；判据：元数据 = 32 bit scale + 32 bit zero，摊到 `G` 个元素上）

| 配置 | 载荷 | 元数据（64/G） | **整块** | 相对 fp16 |
|---|---|---|---|---|
| fp16 | 16.0 | 0.00 | 16.00 | 1.00× |
| fp8（per-tensor 标量） | 8.0 | 0.00 | 8.00 | 2.00× |
| int4，G=32 | 4.0 | 2.00 | **6.00** | 2.67× |
| int4，G=128 | 4.0 | 0.50 | 4.50 | 3.56× |
| int2，G=32 | 2.0 | 2.00 | **4.00** | **4.00×** |
| int2，G=128 | 2.0 | 0.50 | 2.50 | 6.40× |

**「2-bit KV」是载荷口径 2.00 bit/元素；G=32 下整块是 4.00 bit——恰是载荷的两倍，实压缩 4.00× 而不是 8×。** G 从 128 缩到 32 让元数据占比从 **50%（2.00/4.00）涨到…不，是从 11%（0.50/4.50）涨到 50%（2.00/4.00）**。

**表 2：1-bit 断崖**（§B4；判据：K 用固定 2 列 outlier ×100、V 用行异质 `10^(−3·U(0,1))`、**非对称 min-max**、`1024×1024`）

| bit | 可用码字 | K per-token | **K per-channel** | V per-token | V per-channel |
|---|---|---|---|---|---|
| **1** | 2 | −26.35 | **4.41** | **−7.98** | −16.16 |
| **2** | 4 | −15.79 | **16.41** | **4.01** | −4.56 |
| 3 | 8 | −6.54 | 23.83 | 11.46 | 4.33 |
| 4 | 16 | 1.50 | 30.45 | 18.07 | 10.96 |
| 8 | 256 | 26.05 | 55.07 | 42.69 | 35.55 |

**1→2 bit 这一步值 12.00 dB（K）/ 11.99 dB（V）**，是整条曲线上最陡的一段。每 bit 的斜率：2→4 段 **7.02 dB/bit**、4→8 段 **6.15 dB/bit**。**判据必须是「每 bit」，两段的总增量比会给出相反结论。**

**表 3：1-bit 的两个量化器差一整档**（§C1；同一张 K，`1024×1024`）

| 约定 | `qmax` | 非零码字 | 重建 | SQNR |
|---|---|---|---|---|
| 对称 restricted-range | **0** | **0** | **恒为 0** | **0.00 dB** |
| 非对称 min-max | 1 | 2 | 正常 | **4.41 dB** |

**对称 1-bit 的 SQNR 恰为 0.00 dB**——不是负无穷，而是「**误差能量 == 信号能量**」（分子分母都是 `Σx²`）。**「1-bit」这个词在两边指的是 2 级与 0 级两种东西。**

**表 4：KVQuant 的两张表，两个模型、两个基线**（引用值；**逐字读到** arXiv:2401.18079v4）

| 出处 | 模型 | baseline | int4 | 4-bit 技巧版 | 3-bit 技巧版 | int3 | int2 |
|---|---|---|---|---|---|---|---|
| **Table 1**（128K，KV cache GiB 见括号） | **LLaMA-7B** | 5.68 (64.0) | 5.98 (16.0) | 5.72 (16.0) | 5.75 (13.3) | — | 6.01¹ (9.3) |
| **Table 16**（WikiText-2） | **Llama-2-7B** | **5.12** | 5.66 | 5.16 | 5.17 | **22.71** | **4708** (int2) |

¹ Table 1 那一格是 `KVQuant-2bit-1%`。

> 🔴 **★ 这两行不得并列比较**：模型不同（LLaMA-7B vs Llama-2-7B）、基线不同（5.68 vs 5.12）。
> 另有一个**中置信度**的第三方基线 **5.47**（QuaRot / AURA 等复现工作常引），**未逐字读原文**；本仓库只把它与 5.12 一起列出以说明「同名同数据集的基线不止一个」。

**表 5：引擎矩阵**（引用值；**本仓库未独立核实**，除 vLLM 那一项）

| 系统 | 允许值 / 粒度 | 默认 | 备注 |
|---|---|---|---|
| **vLLM** | **18 项**（`auto` … `nvfp4_4over6`）；FP8 走 per-tensor 或 per-head | `auto` | `auto` = 先看模型的 `kv_cache_quant_algo`，否则回落 `--dtype`；**CUDA 11.8+ 上 `fp8` ≡ `fp8_e4m3`**；e4m3 动态范围仅 ±240；**fp8 下前缀缓存不工作** |
| **TensorRT-LLM** | `QuantConfig.kv_cache_quant_algo` 或 `KvCacheConfig(dtype=...)` | — | NVFP4 KV **必须 ModelOpt 离线量化 + FP8 权重/激活**、**仅 Blackwell**；INT8 标 `Calibration Required: Yes`；FP8 `could be enabled even without calibration` |
| **llama.cpp** | **9 项**：`f32/f16/bf16/q8_0/q4_0/q4_1/iq4_nl/q5_0/q5_1` | `f16` | K/V **可独立设置**；量化 V 需 flash attention；MLA 类模型拒绝 `K != V` |
| **SGLang** | `auto / fp8_e4m3 / fp8_e5m2 / bfloat16 / float16`，仅标量 | `auto` | 报 BF16→FP8 **2.00×**、BF16→FP4 **3.56×** |
| **HF TGI** | `fp8 / fp8_e5m2`，仅 per-tensor | — | **仓库已归档** |

⚠ **FP8 KV 的硬件支持矩阵在来源间冲突**（一处含 Ampere、一处称 Ampere 无 FP8），**本仓库未读到官方 support matrix ⇒ 不下断言**。⚠ SGLang 的 `3.56×` 在本仓库的 `64/G` 口径下**精确复现**（16/4.50 = 3.556），并反过来印证它的 FP4 KV 用 G=128——见 §1 第一步。

## 6. 小结与延伸

三条可复用的结论：

1. **「N-bit KV」是一个不完整的说法，至少缺两样。** ① **载荷还是整块、以及元数据按哪个口径算**——本仓库按 `64/G` 记账时，G=32 的「2-bit」整块是 **4.00 bit/元素**，实压缩 **4.00×** 而不是 8×；换 `32/G` 或「不算元数据」两个口径，同一个「4-bit」分别是 4.25 / 4.00 / 4.50 bit；② **哪一个量化器**——对称 restricted-range 的 1-bit `qmax = 0`，重建恒为 0（**0.00 dB**），而非对称 1-bit 有 2 个码字（4.41 dB）。
2. **1-bit 的墙是「级数 + 相对误差爆炸」两件事叠加，绕开它靠的是额外结构**（JL 投影 / 稀疏 outlier / 通道耦合 / 旋转），不是更好的量化器。QJL 的 `m ≈ ε⁻² log n` 是这条路线上最干净的定量结论，而它的 "zero overhead" **只指元数据**。
3. **同一个模型 + 同一个数据集的 PPL 基线不止一个。** KVQuant 一篇之内就有 5.68（LLaMA-7B）与 5.12（Llama-2-7B），第三方还有 5.47。**引用值必须带 `来源 / 表号 / 基线` 三元组**；跨论文并列要显式标注「不得并列比较」。

**这条主线在 §0.8 的三角上是哪条边**：**「范围 / 精度 vs 额外结构」**。1-bit 的失败不是因为「比特不够」，而是因为**在这个比特预算下，任何纯粹的点对点量化器都做不到**——所有可用的路线都往系统里引入了载荷之外的东西（一个投影矩阵、一张稀疏索引表、一个通道耦合模式）。**这是 §0.8 三角上一个特殊的点：在它附近，交换不再发生在三个轴之间，而是必须引入第四个东西。**

延伸有三条。**往回**：粒度轴选错会怎样、以及 V 的论证为什么要看输出误差（[02-kivi-asymmetry](02-kivi-asymmetry.md)）。**往旁边**：权重量化那边的对应物——GPTQ 用二阶信息换舍入决策、AWQ 用激活感知的缩放换显著通道（[ptq/02-gptq](../ptq/02-gptq.md)、[ptq/03-awq-and-smoothquant](../ptq/03-awq-and-smoothquant.md)）。**往硬件**：为什么 per-channel 的激活 scale 会插进 GEMM 内维、以及各家 kernel 怎么把反量化融进 tile（[quantized-kernels](../quantized-kernels/)）。

## 参考

- **KVQuant**，arXiv:**2401.18079v4** —— 本文 **2026-09-18 逐字读到**。Table 1（LLaMA-7B，128K，PPL + KV cache GiB）与 Table 16（全部模型 × WikiText-2，含 Llama-2-7B 列）都在原文里；三个技巧的增量 3.82 / 0.82 / 0.19 与 "4.8× longer context length" 亦逐字读到。
- **vLLM `CacheDType`** —— `vllm/config/cache.py`（`main` 分支，2026-09-18 逐字读到），**18 项**。⚠ **快照会随版本变**；老文档的 7 项与它都真实。
- **QJL**，arXiv:2406.03482v2（AAAI 2025）—— 口径 "**3 bits per FPN**"、定理 3.6 的 `m ≈ ε⁻² log n`、"zero overhead" 的确切含义。**计划基线，本仓库未独立核实**（只从计划 §0.7.F 转录）。
- **CQ**，arXiv:2405.03917v1 —— Llama-2-7B 上 `KVQuant-1b = NaN`、`CQ-8c8b (1.00b) = 7.75`、`CQ-8c10b (1.25b) = 6.25`。**计划基线，本仓库未独立核实。**
- **TensorRT-LLM / llama.cpp / SGLang / HF TGI** 的取值集合与约束 —— **计划基线，本仓库未独立核实**（本轮 `WebFetch` 被策略拦截）。
- **Llama-2-7B / WikiText-2 的第三方基线 5.47** —— 检索摘要一致给出（QuaRot、AURA 等），**中置信度：未逐字读原文**。
- **`UNSOURCED`（正文不使用）**：
  1. 「**仅 K → 1-bit PPL 17.17 / 仅 V → 177.13 / K&V → 620.08**」——计划 §0.7.F 未给出 arXiv 编号；本实现对 KVQuant 全文逐字检索这三个数**命中 0 次** ⇒ **来源未定位**，按 CONTRIBUTING §2 不得标成「引用值」。本仓库改用本机可复现的合成张量 SQNR（§5 表 2）。
  2. 「**全模型 decode = 1 FLOP/weight-byte**」——见 [01](01-kv-cache-basics.md) 的同一段说明。
- 本仓库 [01-kv-cache-basics](01-kv-cache-basics.md) —— KV cache 的字节账；[02-kivi-asymmetry](02-kivi-asymmetry.md) —— K 与 V 的量化轴。
- 本仓库 [ptq/02-gptq](../ptq/02-gptq.md) / [ptq/03-awq-and-smoothquant](../ptq/03-awq-and-smoothquant.md) —— 权重量化侧用「额外结构」换比特的两条对应路线。
- 本仓库 [quantized-kernels](../quantized-kernels/) —— 反量化融进 tile 的 kernel 侧代价。

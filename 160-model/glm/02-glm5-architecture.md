---
title: GLM-5 代骨架：MoE + MLA-256 + DSA + IndexShare + MTP
slug: glm5-architecture
domain: 160-model
topic: glm
level: 3
difficulty: advanced
prerequisites:
  - 80-model-architectures/attention-variants/02-mla-low-rank
  - 160-model/glm/01-glm-technical-route
related:
  - 160-model/glm/03-serving-efficiency-and-deployment
  - 40-kernels/attention/02-flash-attention
  - 80-model-architectures/moe/01-gating-and-routing
  - 120-inference/decoding-strategies/02-speculative-decoding
code:
  - code/02_attention_budget.py
status: stable
updated: 2026-10-13
---

# GLM-5 代骨架：MoE + MLA-256 + DSA + IndexShare + MTP

> GLM-5.3 的模型结构就是 GLM-5.2 的结构，而 GLM-5.2 的结构是 GLM-5 的骨架加一层跨层复用。整个骨架由五个部件拼成：**256 选 8 的 MoE**（撑总参）、**MLA-256**（压 KV）、**DSA 的 lightning indexer**（把注意力从 $O(L^2)$ 降到 $O(Lk)$）、**IndexShare**（把 indexer 自己那个 $O(L^2)$ 再除以 4）、**共享参数的 3 层 MTP**（当投机解码的 draft）。

## 0. 背景与动机

### 为什么「GLM-5.3 的模型结构」要去看 GLM-5.2

GLM-5.3 的官方文档只有一句关于结构的说明：

> It uses the **same base model as GLM-5.2** — every gain comes from post-training.

所以「GLM-5.3 的模型结构」= **GLM-5.2 的基座**，而 GLM-5.2 = GLM-5 骨架 + IndexShare。这是一个容易踩的坑：去找 GLM-5.3 的新模块会一无所获（§4.1 的反面）。

### 一个具体的对照场景：1M 上下文下，注意力到底花在哪

取 1M（= 1 048 576）token 的单条序列，比较三种注意力在同一层内的开销（推导细节见 §1，程序见 `code/02_attention_budget.py`）：

| 方案 | KV 维度／token／层 | 注意力算力（每层，相对稠密 MLA） |
|---|---|---|
| GQA-8（GLM-4.5） | 2048 | —（层结构不同，不直接比） |
| MLA-256（GLM-5） | **576** | 1.0000（基准） |
| + DSA（top-k=2048） | 576 | **0.1448** |
| + IndexShare（4 层共享） | 576 | **0.0377** |

注意第三行与第四行的差别**不在 KV**——IndexShare 一个字节的 KV 都没省。它省的是 indexer 的打分算力。**把「省了什么」分清楚，是读懂这一代骨架的关键**：

| 部件 | 省的是什么 | 一句话机理 |
|---|---|---|
| MoE | 每 token 的算力 | 参数全驻留，每 token 只算 8/256 的 expert |
| MLA | KV 的**字节** | K/V 共享一份 576 维压缩 latent |
| DSA | 注意力的**算力** | 只对 indexer 选出的 top-2048 个位置算注意力 |
| IndexShare | indexer 的**算力** | 4 层共用一个 indexer 的选择结果 |
| MTP | **串行步数** | 一步猜多个 token，用 draft 换 decode 轮次 |

## 1. 原理

### 第一步：MoE 层（256 选 8 + 1 共享）

GLM-5 的 Table 10 只给**计数**：每层 256 个 routed expert、每 token 选 8 个、外加 1 个 shared expert。**门控的形态（打分函数与均衡机制）在 GLM-5 的报告里没有重述**——`sigmoid` / `loss-free` / `auxiliary` 三个词在 GLM-5 报告中零命中。能查到的一手描述来自 GLM-4.5 报告 §2.1：

> We employ **loss-free balance routing** and **sigmoid gates** for MoE layers.

> ⚠ **跨代沿用是推断，不是逐字核实**：GLM-5 报告附录 A 里那句「we follow the setting of GLM-4.5」**作用域是训练超参**（Muon、cosine decay、batch size warmup），**不是**路由形态。下面三条里第 2、3 条由 GLM-5 的 Table 10 直接支持，第 1 条是**推断**：

| # | 内容 | 来源等级 |
|---|---|---|
| 1 | 用 **sigmoid** 打分（不是 softmax），配 **loss-free balance routing**（无辅助损失的均衡偏置） | 【C】GLM-4.5；GLM-5 **未重述** |
| 2 | 每层 **256 个 routed expert，每 token 选 8 个**，外加 **1 个始终激活的 shared expert** | 【C】GLM-5 Table 10 |
| 3 | expert 中间维 $m=2048$，hidden $d=6144$ | 【C】GLM-5 Table 10 |

一个 expert 三个矩阵（gate / up / down），参数量 $3dm$；每层总参与激活的比例是：

$$\frac{P^{\text{act}}}{P^{\text{total}}} = \frac{k+s}{E+s} = \frac{8+1}{256+1} = \frac{9}{257} \approx 3.5\%$$

这就是「744B 总参、40B 激活」的来源。**注意 shared expert 恒被激活**：它把稀疏比从 $8/256$ 抬到 $9/257$，同时承担公共知识，让 routed expert 更专注于差异化能力（`80-model-architectures/moe/03-balance-and-granularity`）。

### 第二步：MLA-256——KV 从 2048 维压到 576 维，代价是解码算力

标准 MHA 每层每 token 要缓存 $2 \cdot H \cdot d_h$ 个数（K 一份、V 一份）。GQA-8 把它降成 $2 \cdot 8 \cdot 128 = 2048$。MLA 换了一条路：**不共享头，而是把 K 和 V 都从一个低秩 latent 里算出来**。

设 hidden $h_t \in \mathbb{R}^{d}$，MLA 的每层缓存只有两项：

$$c^{KV}_t = W^{DKV} h_t \in \mathbb{R}^{512}, \qquad k^R_t = \mathrm{RoPE}(W^{KR} h_t) \in \mathbb{R}^{64}$$

K 与 V 从 $c^{KV}$ 上投影出来：$k^C = W^{UK} c^{KV}$，$v^C = W^{UV} c^{KV}$。因为 $W^{UK}$、$W^{UV}$ **与位置无关**，它们可以在解码时被**吸收**进 $W_Q$ 和 $W_O$：

$$s_{t,i,j} = \underbrace{\left(q^{C}_{t,i}\right)^{\!\top} c^{KV}_j}_{512\ \text{维}} + \underbrace{\left(q^{R}_{t,i}\right)^{\!\top} k^{R}_j}_{64\ \text{维}}$$

于是**解码时所有头共享同一份 576 维的 latent**，每 token 每层只读 576 个数——这正是报告里「MLA performs a **576-dimensional** dot product」的含义。GLM-5 的 Table 10 给出的字段是 `KV LoRA Dim = 512`、`Q LoRA Dim = 2048`。

把两件事分开数清楚（推导在 `code/02_attention_budget.py`）：

| 量 | GQA-8（GLM-4.5） | MLA-256（GLM-5） | 比值 |
|---|---|---|---|
| KV 缓存的**元素数**／token／层 | $2\times8\times128 = 2048$ | $512+64 = 576$ | MLA 省 **3.56×** |
| KV 字节（BF16）／token／层 | 4096 B | **1152 B** | 同上 |
| 解码注意力**算力**／token／层 | $4L\cdot(H d_h) = 4L\cdot12288$ | $\approx 4L\cdot34816$ | MLA 贵 **2.83×** |

第三步是这里最容易被略过的：**MLA 不是一个「全面更优」的替换，它拿算力换显存带宽**。报告把这件事写得很直白——DeepSeek-V3 的头数是照着 H800 的 roofline 选的，「it is inappropriate for other hardware」。在解码被 KV 读带宽卡住的硬件上，3.56× 的带宽节省压过 2.83× 的算力增加；在算力受限的芯片上就未必。

**MLA-256 与 Muon Split。** 报告记录了一个反直觉现象：在 Muon 优化器下，576 维 latent 的 MLA **追不上** GQA-8（2048 维 KV）。解法不是改 attention，而是改优化器——把 $W^{UQ}, W^{UK}, W^{UV}$ **按头拆开**再做矩阵正交化（`Muon Split`），让不同头的投影权重以不同尺度更新。副作用是「attention logits 的幅度在预训练中保持稳定，不需要任何 clipping 策略」。

> ⚠ **UNRESOLVED**：报告说「head dim 从 192 提到 256、头数减 1/3，**保持训练计算与参数量不变**」，但 Table 10 的字段复算不出这个不变性：$96\times192 = 18432$，$64\times256 = 16384$（差 $-11.1\%$）。本仓库**不替它圆场**，把这个「对不上」变成 `code/02_attention_budget.py` 的 C 类断言。

### 第三步：DSA——把注意力从 $O(L^2)$ 降到 $O(Lk)$，但 indexer 自己还是 $O(L^2)$

DSA 把注意力拆成「选」和「算」两段。**lightning indexer** 用一个远比主干轻量的打分函数给每个历史位置打分，取 top-$k$（GLM-5 的 $k = \mathbf{2048}$），主注意力只在这个子集上做。

设 $H_I=32$、$d_I=128$（Table 10 的 `# Indexer Attn Heads` / `# Indexer Head Dim`），主干的 $H=64$、$qk\_dim=192$、$v\_dim=256$。每层的乘加量（FLOPs 按 1 MAC = 2 FLOPs 计）：

$$\underbrace{2L^2 H_I d_I}_{\text{indexer 打分}} \;+\; \underbrace{2Lk\,H\,(qk\_{dim}+v\_{dim})}_{\text{稀疏核心}}$$

稠密 MLA 的同层开销是 $2L^2 H (qk\_{dim}+v\_{dim})$。相除：

$$R(L) = \frac{H_I d_I}{H(qk\_{dim}+v\_{dim})} + \frac{k}{L} = \frac{32\times128}{64\times448} + \frac{2048}{L} = 0.1429 + \frac{2048}{L}$$

这一个式子把 DSA 的三个性质全说完了：

1. **$L \to \infty$ 时 $R \to 0.1429$**——省下的比例有**下限**，因为 indexer 的打分项本身就是 $O(L^2)$，它成了新的瓶颈。这是「DSA 的下一步必须是 IndexShare」的数学依据。
2. **$L = k = 2048$ 时 $R = 1.14$**——**短上下文下 DSA 比稠密注意力更贵**。稀疏化不是免费的。
3. $L$ 越大收益越大，但**有天花板**。

把 IndexShare 加进来（每 4 层只跑一次 indexer，$\div 4$）：

$$R_{\text{share}}(L) = \frac{0.1429}{4} + \frac{2048}{L} = 0.0357 + \frac{2048}{L}$$

在 $L = 1\,048\,576$ 处：$R_{\text{share}} = 0.0377$，$R = 0.1448$，**IndexShare 带来 3.84× 的进一步下降**。

> ⚠ **口径警告**：官方给的数字是「1M 上每 token FLOPs 降 **2.9×**」。本仓库的解析推导给的是 3.84×。**两者不是同一个量**：官方的计数口径（是否计入 top-k 选择、gather/scatter、kernel 发射开销，以及归一化到哪些层）**没有公布**。所以 2.9× 在本主题里一律标为**引用值**，本仓库只推导**结构性结论**（indexer 侧降 4×、稀疏核心不变、$\lim R = 0.0357$）。

**IndexShare 共享的到底是什么？** 只共享**选中的位置列表**（top-k 的索引），**不共享** Q/K/V 计算：被共享的那 3 层照样算自己的 query、注意力权重、value 组合、输出投影和 FFN，只是不再各自跑一遍打分与 top-k。这个区分很重要——它意味着**隐藏状态逐层仍在变化**，共享的是一个「往哪看」的地址簿。

### 第四步：MTP——3 层共享参数，换投机解码的接受长度

MTP 训练时预测未来 $n$ 个 token，推理时同一组头当 draft model 做投机解码。朴素做法要 $n$ 个 MTP 层，**draft 的参数量与 KV 随投机步数线性增长**；DeepSeek-V3 只用 1 个 MTP 层预测 2 个 token，代价是训练-推理不一致、第二个 token 接受率下降。

GLM-5 取的是中间点：**训练时共享 3 个 MTP 层的参数**。于是 draft 的显存与「DeepSeek-V3 的单层」一致，而接受长度更高。报告 Table 2 印值（同样 4 个投机步、私有 prompt set）：

| 模型 | accept length |
|---|---|
| DeepSeek-V3.2 | 2.55 |
| GLM-5 | **2.76** |

把接受长度换成加速比需要一个显式的代价模型（`code/02_attention_budget.py` 的 B 类）：设每个验证步跑 1 次主干 + $n_d$ 次 draft，单次 draft 的相对代价为 $c$，则

$$S \approx \frac{A}{1 + n_d\,c}$$

取 $n_d = 3$、$c \approx 1/79$（draft 每步约等于主干的单层成本，79 是含 MTP 的总层数）：$S \approx 2.76 / 1.038 \approx 2.66$。**这个 $c$ 是假设，不是测量值**——它的价值在于说明：接受长度只决定分子，draft 越贵，加速比越快地回落。

GLM-5.2 把 MTP 也接上 IndexShare，并报告接受长度**最多 +20%**（ablation 里平均接受长度 4.56 → 5.47）。

### 第五步：把五个部件放回一层

一层 GLM-5 MoE block 的数据流：

```
h ──► RMSNorm ──► MLA-256（576 维 latent 上的稀疏注意力，top-k=2048）
                     ▲
                     └── lightning indexer（32 头 × 128 维；每 4 层只在这里跑一次）
   ──► +h（残差）
   ──► RMSNorm ──► sigmoid 路由（跨代推断，见上）──► 8/256 routed + 1 shared expert ──► +h
```

97.8% 的参数在第二行，3.5% 的算力也花在第二行。**「GLM-5 是 744B 模型」这句话描述的是第二行。**

## 2. 执行映射

§1 是算式，这一节是这些算式在硬件上撞到什么。

**MLA 解码变成「共享 latent 上的 MQA」。** 矩阵吸收之后，每个头的 query 都被提升到 576 维，注意力的 key/value 是所有头**共用**的一份 latent。这在 kernel 层的后果是：KV cache 的读取量按「一份」而不是「头数份」计（这是 3.56× 节省的物理来源），但每个头都要拿自己的 576 维 query 去点积同一份数据——**读一次、用 64 次**，算术强度反而比 GQA 高。

**DSA 的 kernel 难点在「选」而不在「算」。** 昇腾侧的适配记录把这件事说得很清楚：`Lightning Indexer` 把 score 计算、ReLU、TopK **融进一个 kernel**，让 NPU 把计算与访存重叠；`Sparse Flash Attention` 专门针对 GLM-5 的稀疏pattern，**并行处理「从 KV cache 里选 TopK token」与「稀疏注意力计算」**。如果按朴素实现把 TopK 单独做成一个 kernel，中间张量（$L \times L$ 的分数矩阵）的写回与再读会吃掉全部收益——$L=1\text{M}$ 时那个中间矩阵根本不可能物化。

**MLAPO：把 13 个碎算子融成一个。** MLA 的预处理在推理框架里天然会碎成十几个小算子（各种 reshape、拼接、RoPE 应用、norm）。官方在昇腾上把这些融成**一个「super operator」**，并让 Vector 单元与 Cube 单元**并行**工作。这是一个通用教训：**MLA 的理论 FLOPs 优势，会被小算子的发射开销吃掉**——融合不是优化项，是 MLA 可用的前提。

**三个必须记住的量级：**

| 量 | 值 | 出处 |
|---|---|---|
| 每 token 每层的 KV 字节（BF16） | **1152 B** | Table 10 的 512 + 64，乘 2 B |
| 1M 上下文（1 048 576）、78 层、单序列 KV | **≈ 94.2 GB** | 推导（`code/02_attention_budget.py`） |
| indexer 每 token 的选中数 | **2048** | GLM-5 报告 §3.2 |
| 每 token 每 MoE 层的 EP 通信量（最坏） | **192 KiB（196 608 B）** | 推导（`01-glm-technical-route` §2） |

第三行与第四行的对比值得单独看一眼：**indexer 选 2048 个位置**，而 MoE 选 8 个 expert。两者都叫 top-k，量级差了 256 倍，**在实现上完全是两种资源**（前者是 KV cache 的 gather，后者是跨设备的 all-to-all）。

## 3. 实现与演进

### 3.1 高效注意力变体的消融：为什么最后选 DSA

GLM-5 报告 §2.1.2 在一个 GLM-9B 上（40 层 GQA、128K 微调）做了完整的对照，全部从**已训练的稠密模型续训** 190B token、64K 上下文、高效层与全注意力层按 1:1 排布。RULER 分数（64K / 128K，括号内是相对全注意力的落差）：

| 方案 | RULER 64K / 128K | 主线代价 |
|---|---|---|
| GLM-9B（全注意力） | 85.35 / 75.28 | 基准 |
| SWA 固定交替 | 65.94 / 44.93（**−19.4 / −30.4**） | 无需额外参数，但崩塌 |
| SWA + 搜索出的层选择 | 83.72 / 69.59（−1.6 / −5.7） | 需要一次性搜索 |
| GDN（线性注意力） | 76.76 / 64.00（−8.6 / −11.3） | 引入新参数 |
| **SimpleGDN** | 81.76 / 67.03（−3.6 / −8.3） | 复用预训练权重 |
| **DSA** | 参见 DSA 专项实验 | **无损（lossless by construction）** |

三条可直接复用的结论：

1. **固定交替的 SWA 会崩塌**（RULER@128K 掉 30 分），而**按搜索选层**能把落差压到 5 分以内。稀疏模式的位置比稀疏比例更重要。
2. **所有基于「丢信息」的高效注意力都有固有落差**——报告的原话是，即使一半的层保留全注意力，续训适应仍会在细粒度检索上带来最多 5.69 分（RULER@128K）与 7.33 分（RepoQA@128K）的损失。
3. **DSA 被论证为「构造上无损」**：indexer 做的是 token 级稀疏，**不丢弃任何长程依赖**，因此可以应用到**所有层**。

**搜索出的 SWA 模式**（GLM-9B，40 层，S=SWA、F=full attention）：

```
SFSSFFSSSFFFFSSFSFFFFFFSFSFSSFSSFSFSSFSSS
```

它在 16K 上以 beam size 8、每步优化 2 层搜出来，收敛约 10 步，然后**泛化到所有长度**。这是「模式搜索」类方法的一个干净样板。

### 3.2 DSA 的续训配方：把改造做成低成本后装

DSA 不是从零训练的，而是在 mid-training 结束的基座上两阶段续训：

| 阶段 | 规模 | 学习率 |
|---|---|---|
| warm-up | 1 000 步 × 14 条 × 202 752 token | 5e-3 → 2e-4 |
| 稀疏适配 | **20B token**（走 mid-training 的数据与超参） | 常数 1e-5 |

对照 DeepSeek-V3.2 的同类续训用了 **943.7B** token——**约 1/47**。另有小规模验证（GLM-4.7-Flash）：warm-up 只训 indexer、基座冻结 1 000 步（batch 16），再联合训练 150B token；RULER@128K 从 79.21 → 78.86，**几乎追平**。

> 这两组数字合起来给出一条工程结论：**稀疏注意力可以在一个已训练好的稠密/MLA 基座上后装**，成本以「十亿 token」计，而不是「万亿 token」。

### 3.3 演进链：每一步在解上一步留下的那一项

```
GQA-8（GLM-4.5）      KV = 2048 维/token/层          ← 长上下文显存与带宽
   │  ↓ 换压缩表示
MLA（576 维 latent）  KV 576 维，解码点积 576 vs 128  ← 解码算力变贵；Muon 下质量掉
   │  ↓ Muon Split + 头维/头数重配
MLA-256              质量追平 GQA-8，解码算力下降     ← 核心注意力仍是 O(L²)
   │  ↓ 加 indexer 选 top-2048
MLA + DSA            核心降到 O(Lk)                  ← indexer 打分仍是 O(L²)
   │  ↓ 跨层复用选择结果
+ IndexShare         indexer 成本 ÷4，撑起 1M         ← 只剩「线性注意力」这条路
   │  ↓ 换骨架（另一型号）
linear + sparse（GLM-5.3-Flash）  320B/18B，45 层，IndexPool
```

**GLM-5.3-Flash 是这条链的分叉**：它换了一个新基座（不是 5.2 的），用 **linear attention 做局部状态建模 + sparse attention 做全局检索**，并把 indexer 的 4 个 key 向量**加权池化成 1 个**（`IndexPool`）。相对 GLM-5.3 的官方口径是**注意力算力少 3.0×、KV cache 小 4.4×**（按「每头每层的注意力算力」与「每层平均 KV cache、BF16」归一）。

> ⚠ 官方**没有点名**用的是哪一种线性注意力（GDN / SimpleGDN / KDA 都可能）。本仓库记为 UNRESOLVED——§3.1 那张消融表给了候选，但**选哪个是官方的决定，不由我们替它填**。

## 4. 踩坑与失败模式

### 4.1 把 DSA 的 top-k 与 MoE 的 top-k 当成同一件事

**现象**：看到「$k=2048$」和「$k=8$」并列，写出「GLM-5 每 token 激活 2048 个 expert」或「indexer 选 8 个 token」。
**成因**：两个 $k$ 分属两个子系统，量级差 256 倍。MoE 的 $k=8$ 决定**跨设备通信量**（每个 expert 可能在另一个 rank 上，要走 all-to-all）；DSA 的 $k=2048$ 决定**同一张卡上的 KV gather 量**（不跨设备）。报告专门吐槽过这一点：indexer 的 $k=2048$「much larger than the $k$ typically used in MoE」，所以**不能像 MoE 那样存 top-k 索引做重放**。
**规避**：见到 top-k 先问它在哪个子系统、单位是「expert」还是「token 位置」。

### 4.2 以为 IndexShare 共享了 KV

**现象**：按「4 层共享」估算，得出「GLM-5.2 的 KV cache 是 GLM-5 的 1/4」。
**成因**：IndexShare 共享的只有**选中的位置列表**。每层仍要存自己的 KV latent（576 维），每层仍要算自己的注意力。
**规避**：把「共享选择」与「共享状态」分开。官方自己也明说 IndexShare **不按同比例降低 KV cache**——长上下文的瓶颈会从算力转移到 KV 容量与搬运。

### 4.3 只按 576 维算 MLA 的 KV，漏掉 RoPE 那 64 维

**现象**：算出每 token 每层 1024 B（$512\times2$），与真值差 12.5%。
**成因**：MLA 的缓存有**两项**——压缩 latent $c^{KV}$（512 维）**和**解耦的 RoPE key $k^R$（64 维）。$k^R$ 不能从 latent 里恢复（它必须带位置信息），所以必须单独存。
**规避**：KV 维度 = `kv_lora_dim + rope_dim`。GLM-5 是 $512+64=576$；报告里那个「576 维点积」正是这个数，两处可以互相印证。

### 4.4 把 SWA 直接插进已训练的模型，指望它无损

**现象**：在现有基座上把一半的层换成滑窗注意力，短上下文几乎不掉分（RULER@4K 只掉 0.3），长上下文直接崩塌（RULER@128K **−30.4**）。
**成因**：固定交替的窗口模式是**与内容无关**的硬切；被切掉的长程依赖没有任何机制去补偿，而模型在预训练时依赖过它们。层的位置很关键——有些层负责检索，换成窗口就等于删掉了检索路径。
**规避**：要么**搜索**层模式（报告的做法：16K 上 beam search、每步 2 层、约 10 步收敛，然后泛化到所有长度），要么用**不丢信息**的稀疏化（DSA：全层应用、只选不丢）。**不要只用「稀疏比例」描述一个稀疏方案**——1:1 的窗口/全局比例，在固定交替与搜索模式下差 25 分。

## 编译与运行

```sh
cd code
make run                 # 默认 python3 + numpy，依次跑 01 / 02 / 03
make 02_attention_budget # 只跑本文件对应的程序
make clean
```

本文的 KV 字节、MLA 解码算力、$R(L)$ 曲线、IndexShare 收益、MTP 加速比，全部来自 **`code/02_attention_budget.py`**。它只用 `math` 与 `numpy`，**不 import 任何推理框架、不加载任何权重**，因此产出的是**推导值**；其中与官方印值的对照一律标为「规格值 vs 推导值」。

## 5. 判据表

本域无 GPU、知识对象以方法与规格为主，§5 采用**判据表**形态（理由见 `160-model/README.md`）。

| 可机械判定的条件 | 结论 |
|---|---|
| 上下文长度 $L \le k_{\text{indexer}}$（GLM-5 为 2048） | DSA 的注意力算力 $\ge$ 稠密 MLA，**稀疏化此时是负收益** |
| $L \gg k$ 且未加 IndexShare | $R(L) \to H_I d_I / \big(H(qk\_{dim}+v\_{dim})\big)$，GLM-5 为 0.143；收益**不会继续变大** |
| 规格里给了 `indexer_attn_heads` | 必须把 indexer 的 $O(L^2)$ 项计入注意力账，只算稀疏核心会高估收益 |
| 规格同时给 `kv_lora_dim` 与 `q_lora_dim`、且**没有** `kv_heads` | 是 MLA；KV 维度 = `kv_lora_dim` + RoPE 维，**不要乘 2**（K/V 共享 latent） |
| 规格给 `n_kv_heads` 与 `head_dim` | 是 GQA/MQA；KV 维度 = $2\times n_{kv}\times d_h$ |
| 某方案的稀疏模式是「固定交替」且是**已训练模型续训**得到 | 长上下文会崩塌，落差随 $L$ 单调扩大；必须搜索层位置或改用不丢信息的稀疏化 |
| 官方给出「每 token FLOPs 降 N×」但未说明归一化口径 | 只作引用值；本仓库只推导结构性比例，不宣称复现了 N |
| 某 MTP 实现的接受长度为 $A$、draft 相对代价为 $c$、步数 $n_d$ | 加速比 $S \approx A/(1+n_d c)$；**接受长度只决定分子**，$c$ 不公布时不得给出绝对加速比 |
| 两个来源对同一模型给出不同总参（差 <2%） | 先核对是否含 embedding / 输出层；口径未对齐前不得判定某一方错 |

## 6. 小结与延伸

五个部件，各自的职责与代价：

- **MoE**：256 选 8 + 1 共享，$9/257 \approx 3.5\%$ 的激活比撑起 744B 总参。
- **MLA-256**：KV 压到 576 维（省 3.56× 字节），代价是解码算力涨 2.83×；**这是拿算力换带宽的交易**，是否划算取决于硬件 roofline。Muon Split 是让 MLA 在 Muon 下追平 GQA-8 的那个补丁。
- **DSA**：核心 $O(Lk)$，$k=2048$；但 indexer 自己 $O(L^2)$，于是收益有下限 0.143，且**短上下文为负**。
- **IndexShare**：把 indexer 再除以 4，收益下限降到 0.036；**不省一个字节的 KV**。
- **MTP**：3 层共享参数，接受长度 2.76（对照 DeepSeek-V3.2 的 2.55）。

下一步：`160-model/glm/03-serving-efficiency-and-deployment` 把这些结构量换成部署账（W4A8/W8A8、KV 量化、RadixCache、国产芯片的融合算子）；推理侧的通用机理见 `40-kernels/attention/02-flash-attention` 与 `120-inference/decoding-strategies/02-speculative-decoding`。

## 参考

- 【C】GLM-5 技术报告 arXiv:[2602.15763v2](https://arxiv.org/abs/2602.15763)：§2.1 与 Table 10（架构字段）、§2.1.1（DSA 续训）、§2.1.2 与 Table 4/5/6（高效注意力消融）、§3.2（indexer 的 $k=2048$ 与确定性 top-k）、附录 A（超参）。**预印本，未确认同行评审。**
- 【C】GLM-4.5 技术报告 arXiv:[2508.06471v1](https://arxiv.org/abs/2508.06471) §2.1 与 Table 1（首个 MoE 代的路由与 GQA-8 形态）。
- 【B】[z.ai/blog/glm-5.2](https://z.ai/blog/glm-5.2)（IndexShare 的 4 层分组、MTP 改进、2.9× 的官方口径）。
- 【B】[z.ai/blog/glm-5.3](https://z.ai/blog/glm-5.3) 与 [docs.z.ai · GLM-5.3](https://docs.z.ai/guides/llm/glm-5.3)（同基座声明）。
- 【B】[z.ai/blog/glm-5.3-flash](https://z.ai/blog/glm-5.3-flash) 与 [docs.z.ai · GLM-5.3-Flash](https://docs.z.ai/guides/vlm/glm-5.3-flash)（linear+sparse、IndexPool、mHC、3.0×/4.4× 口径）。
- 【D】Sebastian Raschka, *GLM-5.2 and IndexShare for Long-Context Sparse Attention*, 2026-06-18（共享的是位置列表而非 Q/K/V；IndexCache 重叠率；**旁证**）。
- 相关域：[80-model-architectures/attention-variants](../../80-model-architectures/attention-variants/)、[80-model-architectures/moe](../../80-model-architectures/moe/)、[40-kernels/attention](../../40-kernels/attention/)。

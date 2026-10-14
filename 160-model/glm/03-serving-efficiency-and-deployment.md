---
title: GLM-5 的部署账：权重比特、KV 容量、EP 通信与国产芯片适配
slug: glm-serving-efficiency-and-deployment
domain: 160-model
topic: glm
level: 4
difficulty: advanced
prerequisites:
  - 160-model/glm/02-glm5-architecture
  - 120-inference/kv-cache/01-capacity-planning
related:
  - 130-model-optimization/quantization/12-ste
  - 130-model-optimization/quantization/18-kv-cache-basics
  - 100-parallelism/expert-parallel/01-expert-sharding
  - 120-inference/serving-infra/02-inference-parallelism-and-pd
code:
  - code/03_serving_budget.py
status: stable
updated: 2026-10-13
---

# GLM-5 的部署账：权重比特、KV 容量、EP 通信与国产芯片适配

> 744B 的模型要落到机器上，只有三个变量可以动：**权重的比特数**（BF16 → W8A8 → expert W4A8）、**KV 的容量与搬运**（1M 上下文单序列 ≈ 94 GB）、**并行与调度的重叠**（EP all-to-all、RadixCache、EPD 分离）。三者的账都能用一行乘法算出来，而算错任何一项都会导致「模型跑不起来」而不是「跑得慢一点」。

## 0. 背景与动机

先把「跑不起来」量化。GLM-5 的 Table 10 给的是 744B 总参、40B 激活——**这两个数分别决定两类不同的失败**：

| 精度 | 权重字节 | 40B 激活的每 token 权重读取 | 1M 上下文单序列 KV |
|---|---|---|---|
| BF16 | **1488 GB** | 80.0 GB | 94.2 GB |
| FP8 / INT8（全部） | **744 GB** | 40.0 GB | 47.1 GB |
| 混合：expert W4 + 其余 W8 | **384 GB** | 27.3 GB | 47.1 GB |

（三行都由 `code/03_serving_budget.py` 算出；最后一行用到「expert 占 96.7% 参数」这个事实，见 §1 第一步。1M 按 config 口径的 1 048 576 token 计。）

这张表的读法：

- **第一列是「能不能装下」**。1488 GB 需要 19 张 80 GB 卡只放权重；744 GB 需要 10 张；384 GB 只要 **5 张**。官方的原话就是「To fit the 750B parameter GLM-5 model onto a **single** Atlas 800T A3 machine」——**W4A8 是「单机装下」这个目标逼出来的，不是精度实验的结果**。
- **第二列是「decode 有多快」**。decode 每生成一个 token 就要把激活的那部分权重完整读一遍，所以吞吐的上限是 `带宽 ÷ 每 token 字节`。
- **第三列是「上下文多长」**。1M 上下文下，KV 的搬运量（47.1 GB/token 步）**超过权重（27.3 GB）1.72 倍**。这是长上下文服务的核心事实：**瓶颈在上下文，不在模型。**

一个具体的对照场景：同一个 1M 上下文的请求，在 BF16 与「expert W4 + 其余 W8」两种部署下，单步要读的字节是 $94.2+80.0 = 174.2$ GB 对 $47.1+27.3 = 74.4$ GB。按 3.35 TB/s 的 HBM 带宽（H100 级别，**引用值**）折算，上限从 **19.2 tok/s** 提到 **45.0 tok/s**——**一个请求的吞吐差 2.34 倍**。

## 1. 原理

### 第一步：权重比特怎么分配——先算「谁占参数」

量化不是「全模型统一降比特」，而是**按组件分配比特**。官方在昇腾上的分法是：

> standard Attention and MLP blocks use **W8A8 (INT8)**, while the **MoE experts are compressed to W4A8 (INT4)**

为什么 expert 拿最低的比特？用 §1 的账回答：GLM-5 每层 MoE 的参数量是 $(256+1)\times3\times6144\times2048 \approx 9.70\text{B}$，75 层就是 **727.6B**；加上 3 个 dense 层（$3\times3\times6144\times12288 \approx 0.68\text{B}$）、78 层注意力（≈12.1B）、indexer（≈2.0B）与 1 个 MTP 层（≈9.7B），总参约 **752B**。于是

$$\text{expert 参数占比} = \frac{727.6}{752} \approx 96.7\%$$

**96.7% 的参数只换来 3.5% 的激活算力**（§1 第一步的 $9/257$）。让这 96.7% 从 8 bit 降到 4 bit，省下的是**近一半的总显存**，代价是那 3.5% 的算力路径上多了一点量化误差——而每个 expert 每次只处理一小部分 token，误差不会在全模型累加。**「谁占参数就压谁，谁占算力就保谁」**是这套分配的判据。

> ⚠ 口径：`code/03_serving_budget.py` 把这个 **96.72% 的占比**用在**官方 744B** 上，而不是用本仓库推导出的 752.2B——这样三档精度的字节账共用同一个总参口径。混用「内部推导值 × 官方便宜」会让表里的数看起来更精确，实际上更不可复核。

### 第二步：KV 的容量账——为什么必须做 cache 量化

MLA 已经读过一遍（`160-model/glm/02-glm5-architecture` §1 第二步）。把字节写出来：

$$B_{\text{KV}} = L \times n_{\text{layers}} \times (d_{c} + d_{R}) \times b = 1\,048\,576 \times 78 \times 576 \times 2 \approx 94.2\ \text{GB}$$

三项可以各自缩：$b$ 由 BF16 → INT8/FP8 减半；$d_c+d_R$ 已经是最紧的（576 是 MLA 的定义）；$n_{\text{layers}}$ 与 $L$ 不可动。

于是**KV 量化是长上下文服务的容量前提**，不是优化项。GLM-5.3-Flash 的服务栈里这一项写得很明确——**hybrid INT8/FP8/BF16 cache quantization**：不同的层/用途用不同的 cache 精度，而不是一刀切。

### 第三步：decode 的吞吐上限——把「读多少字节」写成一个乘积

decode 每生成一个 token，必须读两类东西：

$$B_{\text{step}} = \underbrace{P_{\text{act}} \cdot b_w}_{\text{权重}} + \underbrace{L \cdot n_{\text{layers}} \cdot (d_c+d_R)\cdot b_{kv}}_{\text{KV}}$$

（严格地说权重只读一次、KV 也读一次，且 batch > 1 时权重被摊薄而 KV 随 batch 线性增长。程序里 `batch` 是显式参数，下面先看 **batch = 1 的单流上限**。）

代入 GLM-5 的混合精度部署与 $L = 1\,048\,576$：$27.3 + 47.1 = 74.4$ GB/token 步。在 3.35 TB/s 上给出 $3350/74.4 \approx 45.0$ tok/s。

这个式子最有价值的地方是它的**交叉点**：

$$L^{*} = \frac{P_{\text{act}}\cdot b_w}{n_{\text{layers}}\cdot(d_c+d_R)\cdot b_{kv}} = \frac{27.3 \times 10^9}{78\times576\times1} \approx 608\text{K}$$

**上下文超过约 608K token 后，KV 的搬运量就超过权重**。这解释了为什么长上下文优化（cache 量化、prefix 复用、PD 分离）在 1M 时代从「锦上添花」变成「主线」。

> ⚠ 上面这个 608K 是**推导值**，它依赖三个假设：batch = 1、权重按混合精度（expert W4 + 其余 W8）、KV 按 INT8。换任何一项，这个数都会移动。它是用来做**量级判断**的，不是用来做容量规划的。

**再加一步：batch 救不了长上下文。** 把 batch 加大只会摊薄权重那一项，KV 随 batch 线性增长。于是在 $L = 1$M 处，聚合吞吐有一个**与 batch 无关的上限**：

$$\text{吞吐}_{\text{上限}} = \frac{\text{带宽}}{\text{单序列 KV 字节}} = \frac{3.35\times10^{12}}{47.1\times10^{9}} \approx 71\ \text{tok/s}$$

对照短上下文的 122.6 tok/s：**1M 上下文把吞吐上限砍掉约 42%，而且这个损失无法用批处理补回来**。这是「长上下文不是免费的」最直接的一条量化表述。

### 第四步：EP 通信——层数是通信量的一阶项

MoE 把 96.7% 的参数摊到多个 rank 上，每层前向都要做两次 all-to-all。每 token 每层的通信量（最坏情况，8 个 expert 落在 8 个不同 rank）：

$$B_{\text{EP}} \approx 2 \times k \times d \times 2\ \text{B} = 2\times8\times6144\times2 = 196\,608\ \text{B} = 192\ \text{KiB}$$

乘 75 个 MoE 层：**≈ 14.7 MB/token**。这里的结构性结论比数字更重要：

- 通信量 $\propto$ **MoE 层数** $\times$ $k$；
- **expert 池大小 $E$ 不进通信账**，只进显存账。

这正是 GLM-5 把 expert 从 160 扩到 256、同时把 MoE 层从 89 压到 75 的原因——**用「更胖但更浅」换通信量**。代价是每层的 expert 并行度要求更高（256 个 expert 要切到足够多的 rank 上才能均摊显存），并且**负载不均的方差也随 $E$ 增大**（`100-parallelism/expert-parallel/02-load-imbalance`）。

### 第五步：投机解码——接受长度只解决分子

MTP 的收益不能只看接受长度。设每验证步跑 1 次主干 + $n_d$ 次 draft，draft 相对代价 $c$：

$$S \approx \frac{A}{1 + n_d c}, \qquad A = \text{accept length}$$

GLM-5 的 $A = 2.76$（4 个投机步，对照 DeepSeek-V3.2 的 2.55）；GLM-5.2 把 MTP 接上 IndexShare 后接受长度最多 **+20%**。但 $c$ 才是分母：如果 draft 复用共享参数、每步约等于主干单层，$c \approx 1/79$，$S \approx 2.66$；如果 draft 要独立跑一份长上下文注意力，$c$ 会显著上升，加速比迅速回落。**这就是 GLM-5 让 3 个 MTP 层共享参数、并让 5.2 给 MTP 也加 IndexShare 的动机：压 $c$。**

### 第六步：INT4 QAT 的「逐比特一致」

GLM-5 的 INT4 量化感知训练放在 **SFT 阶段**，并且有一个不显眼但很关键的约束：

> a quantization kernel applicable to both training and offline weight quantization, which ensures **bitwise-identical behavior** between training and inference

这句话的工程含义：**训练时用的伪量化与推理时的真量化必须是同一个 kernel，结果逐比特相同**。否则训练时模型适应的是 A 的误差分布，推理时遇到的是 B 的误差分布——长链路的 agent 任务会把这点偏差放大成行为不一致。这是「QAT 的收益有一半来自一致性，而不是来自精度」的一个实例。

## 2. 执行映射

§1 的账落到硬件上，会变成具体的 kernel 与调度改动。以下都来自官方在**昇腾 Atlas 系列**上的适配记录（§5）。

**融合算子三件套——它们不是优化，是可用的前提：**

| kernel | 融了什么 | 解决什么问题 |
|---|---|---|
| **Lightning Indexer** | score 计算 + ReLU + TopK | 若 TopK 单独成 kernel，$L\times L$ 的分数矩阵要写回再读；$L=1\text{M}$ 时**无法物化** |
| **Sparse Flash Attention** | TopK token 选择 ∥ 稀疏注意力计算 | 选择与计算并行，避免串行等待 |
| **MLAPO** | **13** 个小预处理算子 → 1 个 | MLA 的预处理天然碎成十几个小算子；融合后让 Vector 与 Cube 单元并行 |

**引擎侧四个动作：**

1. **异步调度**（vLLM-Ascend）：把采样结果的 D2H 拷贝与**下一个 decode 步的准备**重叠，消掉调度气泡。
2. **RadixCache**（前缀共享）：agent 场景里系统提示 + 仓库快照在多轮之间完全重复，前缀复用直接等价于「免费上下文」。
3. **Prefix Cache 扩到系统内存**：KV 分层存放，把不活跃的前缀挤到主机内存——因为 1M 上下文的 KV（≈94.2 GB BF16）**装不进任何单卡**。
4. **并行策略**：**Attention DP + MoE EP** 的混合。注意力是「每 token 独立、KV 随样本走」，适合 DP；MoE 是「专家分片、token 要跨设备」，适合 EP。两轴组合而不是单一 TP，是因为它们的通信模式不同（前者几乎不通信，后者是两次 all-to-all）。

**集群侧：EPD 分离。** GLM-5.3-Flash 的部署把 **Encode（多模态编码）/ Prefill / Decode** 拆成三个独立调度的 worker 池。对多模态模型，Encode 的算力特征（视觉编码器）与 Prefill（长序列、算力密集）和 Decode（逐 token、带宽密集）都不同，混在一个池里会让扩缩容失去意义（`120-inference/serving-infra/02-inference-parallelism-and-pd`）。

**一个可交叉验证的量级。** GLM-5.3 的发布页在归一化评测预算时用了每模型吞吐 **115 TPS**（Kimi K3 按 40、Qwen3.8 Max 按 47）。而 §1 第三步的 `batch = 1` 推导上限在 1M 上下文、混合精度下是 **45.0 tok/s**，在短上下文（KV 项可忽略）下是 $3350/27.3 \approx$ **122.6 tok/s**。**115 落在短上下文的 122.6 之下、1M 的 45.0 之上**——量级吻合。这不是证明，但是它说明那套「带宽 ÷ 每 token 字节」的模型抓住了主项。

**性能修复的闭环。** GLM-5.3-Flash 的发布页记录了一件有意思的事：为国产芯片写专用推理引擎的工作，**由 GLM-5.3 自己当基础设施 agent 来加速**——帮工程师开发与优化 kernel、诊断瓶颈、改进服务栈。于是形成一个「模型优化服务自己的系统」的回路。相比同硬件的初始基线，端到端服务性能提升 **3×**。

## 3. 实现与演进

部署路线的四步，每一步都是被上一段的容量或带宽逼出来的：

| 阶段 | 做法 | 为什么必须走到下一步 |
|---|---|---|
| 朴素 | BF16 权重 + BF16 KV | 1488 GB 权重，19 张 80GB 卡才装得下；1M 上下文 KV 94.2 GB（BF16） |
| 全模型低精度 | FP8 / INT8 全部 | 744 GB，但 expert 那 96.7% 的参数仍占大头 |
| **分组件混合精度** | **expert W4A8 + 注意力/MLP W8A8** | 384 GB，单机可容纳；但训练与推理的量化行为不一致会带来偏差 |
| **INT4 QAT** | 在 SFT 阶段做量化感知训练，训练与推理**逐比特一致** | 精度与一致性同时解决 |

引擎侧则是另一条四步：

```
单实例前缀重复计算
   │ ↓ 前缀哈希复用
RadixCache（前缀共享）
   │ ↓ KV 容量超过单卡
Prefix Cache 扩到主机内存（KV 分层）
   │ ↓ 多模态 + 长短请求混合，资源特征冲突
EPD 分离（Encode / Prefill / Decode 三池）+ Attention DP + MoE EP
```

**国产芯片适配的定位。** GLM-5 声称「from the first day」完成 **7 家**国产平台的深度优化：华为昇腾、摩尔线程、海光、寒武纪、昆仑芯、沐曦、燧原。适配方法论浓缩成三条：**极端量化**（W4A8 + QuaRot 抑制 outlier + Flex_AWQ_SSZ 校准 scale）、**高性能融合算子**（上面那三件套）、**推理引擎调度**（异步调度、RadixCache、DP+EP 混合）。GLM-5.3-Flash 在国产芯片集群上的成绩是「端到端服务性能提升 3×，每 token 成本与主流 NVIDIA GPU 相当」。

> ⚠ **口径**：「3×」是**相对同硬件的自有初始基线**，不是「与 NVIDIA 的倍数」。官方那句「reaching hardware efficiency and per-token cost **comparable to** mainstream NVIDIA GPUs」是**量级相当**的表述，不是等号。引用时不得写成「国产芯片 3× 于 NVIDIA」。

## 4. 踩坑与失败模式

### 4.1 把「W4A8」当成「全模型 4 bit」

**现象**：看到「MoE experts 压到 W4A8」，按 744B × 0.5 B 算权重，得出 372 GB，然后按这个数做规划——实际会偏小。
**成因**：W4A8 只作用于 **expert**；注意力、MLP、embedding、indexer 仍是 W8A8。真实混合字节约 **384 GB**（比「全 W4」多约 3%），比「全 W8」少 48%。
**规避**：按组件算字节，不要按全模型一个比特数算。判据：expert 参数占比 × 4 bit + 其余占比 × 8 bit。

### 4.2 用「序列长度 × head 数」外推 MLA 的 1M KV

**现象**：按 GQA 的公式算 GLM-5 的 1M KV，得到约 320 GB，进而判定「1M 不可能服务」。
**成因**：MLA 的 KV 是 **576 维的共享 latent**，不是 $2\times H\times d_h$。按 GQA-8 的 2048 维算会高估 **3.56 倍**。
**规避**：识别的判据是规格里**没有 `kv_heads`、而有 `kv_lora_dim`**——这是 MLA 的指纹。KV 维度 = `kv_lora_dim + rope 维`，**且不乘 2**（K/V 共享）。

### 4.3 认为 DSA 一定让长上下文变便宜

**现象**：给 4K 上下文的短请求也开 DSA，结果比稠密注意力更慢。
**成因**：$R(L) = 0.1429 + 2048/L$。当 $L \le k = 2048$ 时，$R \ge 1.14$——**indexer 的打分与 top-k 选择是纯粹的新增开销，没有任何稀疏收益**。而且 TopK kernel 在序列短时有固定的启动与排序成本。
**规避**：按长度分档。稀疏注意力在长上下文是收益，在短上下文是税。判据：$L$ 与 $k$ 的比值，而不是「用没用 DSA」。

### 4.4 把 indexer 的中间分数矩阵当普通张量

**现象**：实现 DSA 时先算一个 $L\times L$ 的 indexer 分数矩阵，再对它做 top-k。小规模测试通过，1M 上下文时直接 OOM。
**成因**：$L=10^6$ 时单个 `fp16` 的 $L\times L$ 矩阵是 $2\times10^{12}$ B = **2 TB**。这不是「慢」的问题，是不存在。
**规避**：TopK 必须与打分**融合在一个 kernel 里**（昇腾侧的 `Lightning Indexer` 正是这个做法），按行（每个 query）流式处理分块，永不物化整矩阵。这条也是 §2 三件套算子存在的理由。

## 编译与运行

```sh
cd code
make run                  # 默认 python3 + numpy，依次跑 01 / 02 / 03
make 03_serving_budget    # 只跑本文件对应的程序
make clean
```

本文的权重字节、KV 容量、每步读取字节、$L^{*}$ 交叉点、EP 通信量，全部来自 **`code/03_serving_budget.py`**。它只用 `math` 与 `numpy`，**不 import 推理框架、不加载权重、不测硬件**，产出是**推导值**；引用的官方数字一律标为规格值或引用值。

## 5. 判据表

本域无 GPU、知识对象以方法与规格为主，§5 采用**判据表**形态（理由见 `160-model/README.md`）。

| 可机械判定的条件 | 结论 |
|---|---|
| 规格给 `total_params` 与 `kv_lora_dim`，且**无** `kv_heads` | 是 MLA；KV 单序列字节 = $L \times n_{\text{layers}} \times (kv\_lora + rope) \times b$，**不乘 2** |
| 规格给 `n_kv_heads` 与 `head_dim` | 是 GQA/MQA；KV 字节 = $2\times L\times n_{\text{layers}}\times n_{kv}\times d_h\times b$ |
| 量化方案写作「expert W4A8，其余 W8A8」 | 权重字节 = $P_{\text{expert}}\times0.5 + P_{\text{rest}}\times1$；**不得按 0.5 B/参数全算** |
| 部署要「单机装下」一个 MoE 模型 | 先算 expert 参数占比；若 > 90%，expert 必须降到 4 bit 才可能单机容纳 |
| batch = 1 的 decode 吞吐上限 | $= \text{带宽} / \big(P_{\text{act}}b_w + L\,n_{\text{layers}}(d_c+d_R)b_{kv}\big)$ |
| 上下文 $L$ 超过 $P_{\text{act}}b_w / \big(n_{\text{layers}}(d_c+d_R)b_{kv}\big)$ | KV 搬运量 > 权重搬运量；优化重心必须转向 cache 量化与前缀复用 |
| 请求上下文 $L \le k_{\text{indexer}}$（2048） | 不要开稀疏注意力；此时它是净开销 |
| DSA 实现里出现「先算完整 $L\times L$ 分数矩阵再 TopK」 | 该实现在 $L$ 大时必然 OOM；TopK 必须与打分融合 |
| 官方说「相对自有基线提升 N×」 | 这是**同硬件内的相对量**；不得换算成与另一硬件平台的倍数 |
| MoE 的 $k$ 与 indexer 的 $k$ 同时出现 | 前者进 EP all-to-all 通信账，后者进单卡 KV gather 账；**不得互换** |

## 6. 小结与延伸

部署这条线只有一个主问题：**每生成一个 token 要搬多少字节**。答案由三项构成，且它们随上下文长度此消彼长：

- **权重**：expert 占 96.7% 参数 → 压到 4 bit；混合精度下 744B → **384 GB**，40B 激活每步只读 **27.3 GB**。
- **KV**：MLA 压到 576 维；1M 上下文单序列 BF16 **94.2 GB**，INT8 **47.1 GB**。交叉点约 **608K** token，此后再优化权重对长上下文收益递减。
- **通信与调度**：EP 每 token **≈14.7 MB**（∝ MoE 层数 × $k$，与 expert 池无关）；引擎侧靠 RadixCache、KV 分层、EPD 分离与 Attention DP + MoE EP 混合把通信藏进计算。

三个可迁移的判据：**谁占参数就压谁、谁占算力就保谁**（量化分配）；**TopK 永不物化中间矩阵**（融合 kernel）；**相对基线倍数 ≠ 跨平台倍数**（口径）。

下一步：本主题的三篇文档构成一条链——`01-glm-technical-route`（为什么换路线）→ `02-glm5-architecture`（换了什么结构）→ 本文（结构怎么落到机器上）。相关域见 `130-model-optimization/quantization`、`130-model-optimization/quantization`、`120-inference/serving-infra`。

## 参考

- 【C】GLM-5 技术报告 arXiv:[2602.15763v2](https://arxiv.org/abs/2602.15763)：§2.4.1（显存优化五项）、§2.4.3（INT4 QAT 与逐比特一致）、§5（国产芯片适配、W4A8、三件套融合算子、RadixCache/Prefix Cache/DP+EP）。**预印本，未确认同行评审。**
- 【B】[z.ai/blog/glm-5.3-flash](https://z.ai/blog/glm-5.3-flash)：ReplaySSM、W8A8、hybrid INT8/FP8/BF16 cache quantization、Layer Split、EPD 分离、3× 提升与「与主流 NVIDIA GPU 相当」的原文口径。
- 【B】[z.ai/blog/glm-5.3](https://z.ai/blog/glm-5.3)：评测预算的每模型 TPS 归一（GLM-5.3 按 115 TPS）、slime 框架与 >2.3× 训练吞吐。
- 【B】[docs.z.ai · GLM-5.3-Flash](https://docs.z.ai/guides/vlm/glm-5.3-flash)：320B/18B、3.01×/4.44× 的归一化口径（每头每层注意力算力、每层平均 KV cache BF16）。
- 相关域：[130-model-optimization/quantization](../../130-model-optimization/quantization/)、[100-parallelism/expert-parallel](../../100-parallelism/expert-parallel/)、[120-inference/serving-infra](../../120-inference/serving-infra/)。

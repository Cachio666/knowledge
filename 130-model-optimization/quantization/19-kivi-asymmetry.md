---
title: K 与 V 为什么要用不同的量化轴
slug: kivi-asymmetry
domain: 130-model-optimization
topic: quantization
level: 2
difficulty: intermediate
prerequisites:
  - 130-model-optimization/quantization/18-kv-cache-basics
  - 130-model-optimization/quantization/06-granularity
tags: [kivi, per-channel, per-token, kv-cache, outlier, residual]
related:
  - 130-model-optimization/quantization/20-low-bit-and-engines
  - 130-model-optimization/quantization/07-outliers-and-propagation
  - 130-model-optimization/quantization/15-w8a8
code:
  - code/19_kivi.py
status: stable
updated: 2026-10-13
---

# K 与 V 为什么要用不同的量化轴

> K cache 在**固定的少数 channel** 上有大幅值 outlier，所以 K 用 **per-channel**（每列一个 scale）；V 没有这个结构，但注意力输出是**少数重要 token 的加权组合**，所以 V 用 **per-token**（每行一个 scale）。在受控合成张量上，这个选择值 **32.22 dB**（列 outlier 持久时）——**把轴标签写反不会报错，只会推出与 KIVI 相反的结论**。

## 0. 背景与动机

上一篇把 KV cache 的账算清楚了：Llama-2-7B 每 token 512 KiB，4096 ctx / batch 1 恰好 2 GiB。便宜的做法是换 dtype（fp16 → fp8），2× 一步到位。但要再往下压，就必须选**粒度**。

粒度这件事在别处已经讲过一遍（[06-granularity](06-granularity.md)）：给定 b 个比特，**scale 沿哪一维归约，就吸收那一维的尺度差异**。per-token 只治行异质，per-channel 只治列异质，二者不可互替。

KV cache 的特殊之处在于：**K 和 V 是两份不同的张量，而它们的异质性落在不同的维上。** 先看具体的账。取一张 `1024×1024` 的 K（列 outlier）和同样大小的 V（行异质），2-bit 非对称量化：

| 张量 | 构造 | per-token | per-channel | 差 |
|---|---|---|---|---|
| **K** | 2 列固定 outlier ×100（**跨 token 持久**） | −15.80 dB | **16.42 dB** | **per-channel 赢 32.22 dB** |
| **V** | 每 token 的尺度跨 `10^−3…10^0` | **3.96 dB** | −4.39 dB | **per-token 赢 8.35 dB** |

同一张表、同一个量化器、**同样的 scale 个数**（方阵上都是 T = 1024 个，元数据开销一模一样），**差别只在沿哪一维归约**。而这个差别大到 32 dB——足以决定 2-bit KV 是「几乎无损」还是「模型崩掉」。

这就是 KIVI 的全部内容。它的摘要把它叫做 "asymmetric 2bit quantization"——**不对称指的不是符号（它是非对称 min-max），而是 K 与 V 用了不同的量化轴**。

> ⚠ **这门课有一个必须先钉死的零件。** 「per-token」与「per-channel」是**相对**的词，指代对象取决于张量布局。计划基线里的第一版就把这两个标签写反了——**写反不会有任何报错**，只会让结论变成「K 应该用 per-token」，与 KIVI 正好相反。§4.1 把这个后果真跑了出来。

## 1. 原理

### 第一步：把轴约定写死

本文与配套程序一律用 **`[tokens, features]`** 的排布（transformers 里 KV cache 的通行布局）：

```
张量 X，形状 [T, D]
    X[t, j]：第 t 个 token 的第 j 个 feature（= channel）
    T = 序列长度（token 维）  D = head_dim（channel 维）

「每行一个 scale」  沿 axis 1 归约出 T 个 scale  →  per-token
「每列一个 scale」  沿 axis 0 归约出 D 个 scale  →  per-channel
```

**判据是「沿哪一维归约」，不是那个词本身。** KIVI 论文自己的话是 "certain fixed **columns** in activations exhibit larger outliers"——它把 outlier 落在**列**上，而列就是 channel。

### 第二步：为什么「每列」治列异质

设第 `j` 列有一个整体尺度 `c_j`，即 `X[t,j] = r[t] · c_j · ε`。用**每列一个 scale**：

```
s_j = range(X[:, j]) / (2^b − 1)
```

`s_j` 完全由第 `j` 列自己的取值范围决定，**列之间的尺度差异被逐列吸收**。用**每行一个 scale** 则相反：

```
s_t = range(X[t, :]) / (2^b − 1)
```

某一行里只要有一个大幅值列，`s_t` 就被它撑开，**这一行里其余所有 channel 一起损失分辨率**。而 K 的 outlier 是**持久**的（LLM.int8 指出：某通道有 outlier 则它在所有 token 上都出现，同通道跨 token 的幅值方差很小），所以**每一行都被同一个 outlier 撑开**——per-token 对 K 是全表性的破坏。

反过来对 V：V 没有 channel 维的 outlier 模式，取而代之的是**行（token）之间的尺度差异**，于是 per-token 赢。

### 第三步：受控消融——把「异质性在哪一维」单独拎出来

上面那两行对照还不是干净的证据：两个场景同时换了构造和结论。要让归因站得住，得**只改一个维度**，并且**先把一个混杂因子消掉**。

**混杂因子是组大小。** 非方阵上，per-token 的一组是「一行」= D 个元素，per-channel 的一组是「一列」= T 个元素；`T ≠ D` 时两种粒度连**组的规模**都不同，而组越大 `min-max` 范围越宽、误差越大。所以本文一律用**方阵 1024×1024**：两种粒度每组都是 1024 个元素，`amax` 的统计量同分布，**唯一变量只剩「异质性落在哪一维」**。

对照组成立与否有一个人眼可见的判据：**iid 数据上两者必须打平。** 实测差 **−0.00 dB** —— 这一格是整张表可信度的地基。

| 构造（1024² 方阵，B=2） | per-token | per-channel | ch − tok |
|---|---|---|---|
| **iid（对照组）** | 4.00 | 3.99 | **−0.00** |
| 列 outlier，**持久**（K 的结构） | −15.80 | **16.42** | **+32.22** |
| 列 outlier，**非持久** | −8.81 | 4.25 | +13.05 |
| 行异质（V 的结构） | **3.96** | −4.39 | −8.35 |
| 行 outlier（1/128 的 token ×100） | **3.90** | −10.31 | −14.21 |

三件事一眼可见：

1. **对照组打平**（−0.00 dB）⇒ 表里其余的差都是数据结构的贡献，不是组大小的混杂。
2. **异质性在哪一维，哪一维的粒度就赢**，而且符号严格对应。
3. **★ outlier 的「持久性」本身就是 +19.17 dB。** 同样是列 outlier，持久时 per-channel 领先 32.22 dB，非持久时只有 13.05 dB。机制：非持久的列 outlier 让**每列的最大值**由不同的 token 贡献，per-channel 的 scale 被那个偶然的极值撑大；持久的列 outlier 让 scale 与被量化值的比例**在所有 token 上一致**，于是 per-channel 真正做到「把误差限制在这一列内」。

> ⚠ **第 3 条是本文相对 `06-granularity` 的增量。** 那篇的列 outlier 场景（「7/4096 列 ×100」）没有区分持久与非持久，而 KV cache 的 K 恰恰是**持久**的那一种——这正是 LLM.int8 的「持久性」结论在这里真正做功的地方。

### 第四步：V 的论证必须看**输出**误差，不是重建误差

到这里 V 的结论还只是「per-token 重建更准」。KIVI 的论证比这更强，也更微妙——**它看的是注意力输出**。

V 不参与打分，它被加权求和：

```
O[i, :] = Σ_j  A[i, j] · V[j, :]
```

注意力分数高度稀疏（KIVI 报 Llama-2-13B 的平均稀疏度 **84.3%**），所以 `O` 本质上就是**少数几个重要 token 的 V 的加权组合**。per-token 量化把误差限制在单个 token 的向量内；per-channel 则让每个 token 的误差都由**整列的全局极值**决定，于是那些幅值小的 token 承受了与自己无关的误差——而它们恰恰可能带着大的注意力权重。

本机在合成张量上把这个放大效应量了出来（§5 表 2）：B=2 时**重建**误差的落差是 8.46 dB，而**输出**误差的落差是 **26.34 dB**——放大 3 倍。

**KIVI 的 Table 2 把这两件事分成了两行**，而它们的读法完全相反（Llama-2-13B，全层全头平均，逐字读到）：

| 行 | per-token | per-channel | 谁赢 |
|---|---|---|---|
| `K_recon`：`‖(X_K − X_K')/X_K‖_F` | 13.67 | **4.55** | per-channel |
| `K_score`：`‖(A − A')/A‖_F` | 47.00 | **9.60** | per-channel（约 5×） |
| `V_recon`：`‖(X_V − X_V')/X_V‖_F` | 4.57 | **3.73** | **per-channel 略胜** |
| `V_delta`：`‖(A·X_V − A·X_V')/(A·X_V)‖_F` | **3.55** | 49.89 | **per-token（14.05×）** |

> 🔴 **★ 这张表是本文最容易引错的一处，而且它已经在计划基线里被引错过一次。**
> 计划 §0.7.F 写的是：「V 侧：per-token 误差 **4.57**（输出误差 Δ = 3.55），per-channel **49.89**——per-token 约 **15×** 小于 per-channel」。
> **这句话把两行拼成了一行。** 正确的读法是：
> - **V 的重建误差 per-channel(3.73) 比 per-token(4.57) 还小**——单看重建，per-channel 甚至略胜；
> - **15×（精确 49.89/3.55 = 14.05×）只在 `V_delta` 这一行成立**；
> - 「4.57 与 49.89 之比」是 **0.092**，与「15×」毫无关系。
>
> **两个数各自都对、来源也真实，错在它们不是同一个量**（P9）。判据：**引用时必须写出它属于哪一行**。

### 第五步：KIVI 的超参与残差滑窗

KIVI §4.1 逐字（本次逐字读到）：

> "the **group size G** in Algorithm 1 for quantization is set as **32** across all experiments,
> the **residual length R** for key and value cache is set to **128**."

**这两个数是两个不同的量，极易互串**（P9 的另一种形态）：

- **G = 32** 是**量化分组**大小——一组 32 个元素共享 scale/zero。论文 Table 5 的消融：Llama2-13B / GSM8K 下 `G=32 → 20.77`、`G=64 → 21.00`、`G=128 → 17.29`——**G=32 与 64 相近，到 128 明显下降**。
- **R = 128** 是**全精度滑窗**长度——最近 R 个 token 的 K/V **不量化**，保持 fp16。论文的分析段还说 "In practice, we set `R ≤ 128`"。

**残差滑窗的收益完全由「窗口内落了多少注意力质量」决定。** 本机把它量了出来（§5 表 3）：窗口覆盖全部注意力质量时，输出误差**恰为 0**（不是「近似为 0」）——因为那几个 token 根本没被量化。

⚠ **本机的合成 V 复现不了 KIVI 说的「滑窗对 GSM8K crucial」那件事。** 那里的局部性是真实模型的性质（最近的 token 对下一个 token 最相关），我的注意力分布是**造**出来的。**这条只有引用值，没有本机实测。**

## 2. 执行映射

**（1）per-channel 无法流式，这是 KIVI 那个「攒够 R 个 token 再量化」的真正原因。** KVQuant §3.6 把代价写得最清楚（逐字读到）：

> per-channel：`all the prior cached Keys will need to be updated`
> per-token：`online calibration for this does not require updating prior cached Values`

**每次追加一个 token，per-channel 的那一列 scale 就可能变**，一变就得把此前所有缓存过的 K 重新量化一遍——这在 decode 的每一步都做是不可行的。所以 K 只能**离线校准**（KVQuant 走这条）或**攒成一批再量化**（KIVI 走这条，一批 = R 个 token）。

**（2）于是业界的分工是**：**K（per-channel）→ 离线或攒批**；**V（per-token）→ 在线逐 token 计算**，因为新 token 的 scale 只影响它自己。

**（3）反量化与 matmul 在 tile 级融合。** KIVI 的系统部分把反量化融进 `Q_MatMul`（"fuses the dequantization with matrix multiplication at the tiling level"）。这一点很重要：per-channel 的 scale 本来是「按 channel」的，而 GEMM 的归约沿 channel——把量化轴放在归约轴上会让 scale 插进内维，**这正是权重侧能做 per-channel、激活侧只能退到 per-token 的那条硬约束**（见 [17-kernel-design](17-kernel-design.md)）。KIVI 用 tiling 级融合把它绕了过去，代价是 kernel 复杂度。

**（4）三家引擎在这一维上的选择完全不同**（引用值，见 [03](20-low-bit-and-engines.md) §2）：

| 系统 | 写侧 | 读侧 | 支持 per-token？ |
|---|---|---|---|
| vLLM（FP8） | 融合在 `reshape_and_cache_flash_kernel`（**独立于 attention 的一个 kernel**） | attention kernel 内 | **否**（per-tensor / per-head） |
| SGLang | 原本 4 个独立 kernel，SGLang 融合为 1 个 | attention kernel 内 | **否**（标量 scale） |
| KIVI | K：每满 R 个 token 做一次；V：出队时逐 token | **dequant 与 matmul 在 tile 级融合** | **是**（K per-channel / V per-token，动态） |

⚠ **vLLM 是 quantize-on-write，但「融合」指的是量化与写 cache 融合在同一个 kernel 里，不是把量化融进 attention。** 写入是一个独立于 attention 的 kernel。

## 3. 实现与演进

**第一步：per-token 一个粒度打天下。** FlexGen 那一代的做法是 K 与 V 都用 per-token group-wise（G=128 或 64）。它对激活侧的硬件友好（scale 不插进 GEMM 内维），但对 K 是错的粒度——Table 1 的消融说明：`2bit (K-𝕋, V-𝕋)` 在 Llama-2-7B 的 GSM8K 上只有 **0.83**（16bit 是 13.50）。

**第二步：KIVI 把两个轴拆开。** `2bit (K-ℂ, V-𝕋)` = **59.08 CoQA / 5.76 GSM8K**——同一份数据、同一个 2-bit，只换 K 的量化轴，GSM8K 从 0.83 拉到 5.76。论文 §5 还写明 KVQuant 的同一个观察是 "independently discovered"（逐字读到）——**K 的 per-channel 结论有两个独立来源**。

**第三步：KVQuant 把 K 侧再往前推三步**（引用值，arXiv:2401.18079v4 逐字读到）：

1. **per-channel K + per-token V 的基线本身**就带来 **3.82** 的 PPL 改善（LLaMA-7B / WikiText-2 / 3-bit）；
2. **pre-RoPE 量化 K**（+0.82）：RoPE 会「jointly rotates pairs of channels by different angles depending on the position」，把原本一致的大幅值通道打散；先量化再套 RoPE 就绕开了（那句**结构陈述**的定义——**哪两个通道配对**、为什么叫「成对旋转」——正文在 `80-model-architectures/positional-encoding/01-rope-definition.md`，此处回链；本文只引用它对**量化轴**的后果）；
3. **per-vector dense-and-sparse**（+0.19）：只把 **1% 的 outlier** 单独存成稀疏格式。

合起来的主张是 **3-bit 下 perplexity 退化 <0.1**，对应 **4.8× 更长的上下文**。

**第四步：把残差滑窗与量化拼起来。** KIVI 的最终配置（KIVI-2）在 Llama-2-7B 的 GSM8K 上是 **12.74**，而「全量 2-bit、无滑窗」是 5.76——**滑窗把 2-bit 从「崩」救回「接近 16bit 的 13.50」**。这条也解释了为什么残差长度 R 是个必须调的超参（Table 5：R=128 给 20.77，R=64 给 19.86）。

## 4. 踩坑与失败模式

### 4.1 ★ 轴标签写反（不会有任何报错）

**现象**：文档里 `per-token` 与 `per-channel` 两个标签互换。表格照印、数字照对、公式照推，**没有任何一处会报错**。而结论变成「K 应该用 per-token」——与 KIVI 正好相反，且相差 **32.22 dB**。

**成因**：这两个词是**相对**的，指代对象取决于张量布局。本机把这个「写反会得到什么」跑了出来（§C1）：列 outlier 场景下真实的 per-token 是 **−15.80 dB**、per-channel 是 **16.42 dB**；**标签互换后两个数字一个没变**，只是读出来的建议反了。

**规避**：三条一起做。① **在文档里把轴约定写成一行公式**（`[tokens, features]`；每行 = per-token；每列 = per-channel），并让它出现在**第一次使用这两个词之前**；② **把「标签 ↔ 归约轴」的对应本身写成一条可失败的断言**；③ 引用别人的结论时，凡出现这两个词，回头核对它的布局约定。

### 4.2 ★ 库 API 的 `axis` 语义与直觉相反，而且**方阵上不报错**

**现象**：调用 `torch.quantize_per_channel(x, scales, zps, axis=0, ...)`，以为「axis=0 ⇒ 每列一个 scale」。在方阵上它**静默地**把列 j 的 scale 用在了行 j 上。

**成因**：库的语义是「**scale 的长度等于 `input.size(axis)`，且沿 axis 这一维索引取 scale**」。所以 `axis=0` 意味着**每行一个 scale**。本机实测（§A1/§A2）：

- 非方阵 `[64, 7]` 上传「长度 7 的列 scale + `axis=0`」会**响亮地失败**：
  `RuntimeError: length of scales must equal to channel, expected 64 got, 7`
- **方阵 `[64,64]` 上两个方向长度相同，错的那个照样被接受**：同一份列 scale，`axis=1` 给 **43.12 dB**，`axis=0` 给 **31.81 dB**，差 **11.31 dB**。

**规避**：① **在非方阵上先验证一次轴语义**（那里会报错）；② 断言库路径与手写路径给出同一个数（本机 §A3 两个方向都比对了，差 < 1e-3 dB）；③ 记住 `axis=k` 的含义是「把第 k 维当作**被量化的通道**」。

> ⚠ **幅度依赖张量宽度。** 本机在 `[64,64]` 上量到 11.31 dB 的差距，同域 `06-granularity` 在 `(4096,4096)` 的 outlier 张量上量到 **38.88 dB**。**引用这个差时必须带宽度与张量构造**——它不是常数。

### 4.3 ★ KIVI 的「逐元素相对误差」判据会被近零元素炸掉

**现象**：想复现 KIVI Table 2 的 `‖(X−X')/X‖_F`，在合成的高斯张量上得到 **10⁵–10⁷**，而论文印的是 13.67 / 4.55。

**成因**：那个判据**逐元素相除**，分母里的近零元素把结果整个主导。真实 KV cache 不会出现这种量级，是因为它的数值分布与 iid 高斯不同——**但这不是「论文错了」，是「同一个判据换一份数据就不可比」**。

**规避**：**用 SQNR（`10log10(Σx²/Σ(x−x̂)²)`，全表求和）作为主要判据**，逐元素相对误差只用来**比方向**（本机实测方向仍一致：per-channel 的逐元素相对误差更小），**不用来引绝对值**。

### 4.4 ★ 三处「看起来该有但其实没有」（口径订正）

| 说法 | 事实 | 判据 |
|---|---|---|
| 「KIVI 的 perplexity 是 …」 | **KIVI 全文 0 处「perplexity」**（逐字检索 arXiv:2402.02750v2，命中 0 次）。它只报任务准确率（CoQA / TruthfulQA / GSM8K / LongBench） | 引用 KIVI 的 PPL **必须标注第三方复现** |
| 「2.35×–3.47× 是显存压缩」 | **那是吞吐**。显存口径是 "2.6× less peak memory (**including model weight**)"——**含模型权重**，不是 KV cache 本身 | 逐字：`2.6x less peak memory (including model weight)` / `2.35x~3.47x throughput` |
| 「group size = 128」 | **G = 32**；**128 是 residual length R** | 逐字："the group size G … is set as **32** across all experiments, the residual length R … is set to **128**" |

三条都在 `specs_kv.py` 里落成了带来源的常量（`KIVI_PERPLEXITY_COUNT` / `KIVI_MEMORY_QUOTE` / `KIVI_THROUGHPUT_QUOTE` / `KIVI_GROUP_SIZE_G` / `KIVI_RESIDUAL_LENGTH_R`），并被断言核过——**不落成常量就会被误引**。

## 编译与运行

```bash
cd code
make run                     # 自动探测装了 torch ≥ 2.8 的解释器，依次跑 01–03
make 19_kivi                 # 只跑本文对应的程序
make verify                  # 默认 python3，跑 verify_kv.py（**不 import torch**）
```

本篇对应的编号程序是 **`code/19_kivi.py`**（真 torch；B 类断言在程序内部，失败即非零退出）。它跑的是 `(1024,1024)` 方阵、种子 `(20260918, 20260931, 20260944)`、KIVI 式**非对称 min-max**（`s = (max−min)/(2^b−1)`、`z = min`）。

**配套程序与 `verify_kv.py` 的分工**（CONTRIBUTING §2 第三种形态）：

| | 规模与工具 | 产出档位 | 本篇负责什么 |
|---|---|---|---|
| `19_kivi.py` | `(1024,1024)`，真 torch | **本机实测值** | 消融表、V 的输出误差、`quantize_per_channel` 的轴语义 |
| `verify_kv.py` §A.4–A.5 | `64×64`，标准库 `random` | **推导** | 同一批构造的另一套实现（列表的列表） |

⚠ **规模不同 ⇒ 数字不同，只有定性结构必须一致。** 本机在 64×64 上量到：列 outlier 的领先 **29.34 dB**、行异质的领先只有 **3.98 dB**（1024×1024 上是 32.22 / 8.35）。**领先幅度随张量宽度变**——同域 `06-granularity` 也实测过同一现象（256 宽 10.30 dB vs 4096 宽 20.89 dB）。**两档不得互相背书。**

## 5. 实测数据

**测量条件**：CPU **Apple M1 Pro（10 核）**；macOS；python **3.12.9**；**torch 2.11.0**。表 1–4 的每一格在 **2.6.0 / 2.7.1 / 2.8.0 / 2.11.0 / 2.13.0** 五个解释器上**逐格一致**（本仓库实测；注意 `torch.rand` 的构造已写成不经过 `uniform_` 的形式，那是为了让随机流跨版本稳定）。**CPU-only，无 GPU 数字。**

**表 1：受控消融——异质性在哪一维，粒度就给哪一维**（`19_kivi.py` §B2；`(1024,1024)` 方阵、3 个种子均值）

| 构造 | B=2 per-token | B=2 per-channel | B=4 per-token | B=4 per-channel |
|---|---|---|---|---|
| **iid（对照组）** | 4.00 | 3.99 | 18.05 | 18.05 |
| 列 outlier，**持久**（K） | −15.80 | **16.42** | 1.45 | **30.47** |
| 列 outlier，非持久 | −8.81 | 4.25 | 3.44 | 18.29 |
| 行异质（V） | **3.96** | −4.39 | **18.03** | 10.83 |
| 行 outlier | **3.90** | −10.31 | **18.00** | 3.74 |

**判据（缺一不可）**：张量按 `[tokens, features]`；`SQNR = 10log10(Σx²/Σ(x−x̂)²)`（**全表求和**）；量化器 = KIVI 式**非对称 min-max**（`s = (range)/(2^b−1)`、`z = min`，可用码字恰 `2^b` 个）；per-token = 沿 axis 1 归约（每行一组）；per-channel = 沿 axis 0 归约（每列一组）；**方阵 ⇒ 两种粒度的组大小都是 1024**；列 outlier 在固定位置 `{7, 61}`、幅值 ×100（持久的那一组带 10% 的跨 token 抖动）；行异质的尺度因子取 `10^(−3·U(0,1))`；种子 `(20260918, 20260931, 20260944)`。

- **对照组差 −0.00 dB** —— 组大小相等时两种粒度本该完全对称。
- **列 outlier 持久时 per-channel 领先 32.22 dB（B=2）/ 29.02 dB（B=4）。**
- **行异质时 per-token 领先 8.35 dB（B=2）/ 7.20 dB（B=4）。**
- **持久性的价值：+19.17 dB**（32.22 − 13.05）。

**表 2：V 的输出误差放大**（§B3；行异质 V、注意力权重非零项 15.6%、最大权重 0.0128）

| | 重建 per-token | 重建 per-channel | 落差 | 输出 per-token | 输出 per-channel | 落差 |
|---|---|---|---|---|---|---|
| B=2 | 3.94 | −4.52 | 8.46 dB | 4.28 | −22.06 | **26.34 dB** |
| B=4 | 18.02 | 10.72 | 7.30 dB | 18.51 | −3.76 | **22.27 dB** |

**输出误差的落差是重建误差落差的 3 倍以上**——与 KIVI Table 2 的 V 两行同向（那里 per-token 在重建上只赢 4.57 vs 3.73，在输出 Δ 上赢 49.89 vs 3.55）。

**表 3：残差滑窗（R）= 全精度保持最近 R 个 token**（§B4；注意力权重全部落在最后 128 个 token 上）

| R | 窗口内注意力质量 | 输出 SQNR |
|---|---|---|
| 0 | 0.0% | 4.28 dB |
| 16 | 13.5% | 6.02 dB |
| 32 | 28.3% | 7.65 dB |
| 64 | 50.6% | 7.64 dB |
| **128** | **100.0%** | **∞（误差恰为 0）** |
| 256 | 100.0% | ∞ |

**判据**：窗口内质量 100% 时输出误差**恰为 0**（不是「近似为 0」）——那几个 token 根本没被量化。⚠ **这条曲线是「注意力质量在窗口内的占比」的函数，不是 R 本身的函数**；本机那个注意力分布是造出来的，**复现不了** KIVI 说的「滑窗对 GSM8K crucial」。

**表 4：比特账——载荷与整块等效宽度**（§B5；判据：元数据 = (32 bit scale + 32 bit zero-point) / 组大小 G = **64/G**）

| 配置 | 载荷 | 元数据 | 整块 | 相对 fp16 |
|---|---|---|---|---|
| int4，G=32 | 4.0 | 2.00 | **6.00** | 2.67× |
| int4，G=128 | 4.0 | 0.50 | 4.50 | 3.56× |
| int2，G=32 | 2.0 | 2.00 | **4.00** | **4.00×** |
| int2，G=128 | 2.0 | 0.50 | 2.50 | 6.40× |

**「2-bit KV」是载荷口径（2.00 bit/元素）；G=32 下整块是 4.00 bit——恰是载荷的两倍。** 实压缩是 **4.00×**，不是「16/2 = 8×」。**两个不是同一个量。**

> ⚠ **元数据的口径有三种，必须写明是哪一种**：本表按 **64/G**（scale 与 zero 都算）；只算 scale 一份是 **32/G**（对称量化下才对）；**完全不算**是 KVQuant 论文 `Avg. Num. Bits` 列的口径（它的 KVQuant-4bit 写 4.00-4.02，即纯载荷）。**同一个「4-bit」，三个读数分别是 6.00 / 5.50 / 4.00 bit。** 见 [03](20-low-bit-and-engines.md) §1。

**表 5：`quantize_per_channel` 的轴语义**（§A1/§A2，真 torch 实测）

| 张量 | 调用 | 结果 |
|---|---|---|
| `[64,7]` | 长度 64 的行 scale，`axis=0` | 通过 |
| `[64,7]` | 长度 7 的**列** scale，`axis=0` | **`RuntimeError: length of scales must equal to channel, expected 64 got, 7`** |
| `[64,64]` | 列 scale，`axis=1`（对） | **43.12 dB** |
| `[64,64]` | **同一份列 scale**，`axis=0`（错） | **31.81 dB（不报错）** |
| `[64,64]` | 行 scale，`axis=0`（对） | 43.11 dB |
| `[64,64]` | **同一份行 scale**，`axis=1`（错） | 31.70 dB（不报错） |

**同一个 API，非方阵报错、方阵静默，差 11.31 dB。** ⚠ 这个差**依赖张量宽度**（同域 `06-granularity` 在 4096² 上量到 38.88 dB）。

**表 6：两种判据的读数完全不同**（§C2；同一张 K，列 outlier）

| B | 粒度 | 逐元素相对 `‖(X−X')/X‖_F` | SQNR |
|---|---|---|---|
| 2 | per-token | 2 557 271.25 | −9.94 dB |
| 2 | per-channel | **83 136.22** | **21.48 dB** |
| 4 | per-token | 406 969.13 | 7.32 dB |
| 4 | per-channel | **13 401.89** | **35.54 dB** |

**方向一致、量级完全不是一回事。** 论文印的 13.67 / 4.55 与这里的 10⁵ 量级差 4 个数量级——**同一个判据换一份数据就不可比**。

## 6. 小结与延伸

三条可复用的结论：

1. **K 与 V 的异质性落在不同的维上，所以量化轴必须分开。** K 的列 outlier **持久**，per-channel 领先 **32.22 dB**；V 的行异质，per-token 领先 **8.35 dB**。iid 对照组差 **−0.00 dB**，证明这个差来自结构、不来自组大小。
2. **★ outlier 的「持久性」本身值 +19.17 dB。** 同样是列 outlier，持久的那一种才让 per-channel 真正「把误差关在这一列里」。这是 KV cache 场景与一般激活场景的关键区别。
3. **V 的结论必须看输出误差，不是重建误差。** 输出落差（26.34 dB）是重建落差（8.46 dB）的 3 倍以上。KIVI Table 2 的 `V_recon` 与 `V_delta` 是**两行**，`V_recon` 上 per-channel 甚至略胜——**把 4.57 与 49.89 并列成「15 倍」是 P9。**

**这条主线在 §0.8 的三角上是哪条边**：**「粒度 vs scale 开销」**。per-token 与 per-channel 在方阵上花的 scale 比特**完全一样**（都是 T 个），所以这是一个**纯粹的「花在哪一维」的选择**——而本机实测这个选择值 32 dB。**当两个选项的成本相同时，选错的那一个没有任何代价信号，只会安静地输掉。**

延伸有三条。**往下**：把同样的思路推到 1-bit 会怎样——那时连「几个级数」都成了问题（[03-low-bit-and-engines](20-low-bit-and-engines.md)）。**往旁边**：outlier 为什么会持久、它与「massive activations」不是同一个现象（[07-outliers-and-propagation](07-outliers-and-propagation.md)）。**往硬件**：per-channel 的 scale 插进 GEMM 内维这件事在权重侧怎么解决（[17-kernel-design](17-kernel-design.md)）。

## 参考

- **KIVI**，arXiv:**2402.02750v2**（`arxiv.org/html/2402.02750v2`）—— 本文 **2026-09-18 逐字读到**。用到的原文：
  - §3.2："in key cache, some fixed channels exhibit very large magnitudes, whereas in value cache, there is no significant pattern for outliers"；"The **persistence** of outliers within each channel means that per-channel quantization can confine the quantization error to each individual channel without impacting the other normal channels."
  - §3.2："the per-token quantization can lead to **almost 5× larger attention score error** than per-channel quantization"；V 侧 "the per-token quantization error is **almost 15× smaller** than per-channel quantization, which explains why OB2 happens"
  - §4.1："the group size **G** … is set as **32** across all experiments, the residual length **R** … is set to **128**"
  - §5 Related Work："A similar observation and approach has been **independently discovered** and developed in the concurrent work KVQuant"
  - 摘要："**2.6×** less peak memory (**including model weight**)" + "up to **4×** larger batch size, bringing **2.35×~3.47× throughput**"
  > ⚠ **全文 0 处「perplexity」**（逐字检索确认）——它只报任务准确率。**引用 KIVI 的 PPL 必须标注第三方复现。**
  > ⚠ Table 2 的**四行是两个不同的量**（`recon` 与 `score`/`delta`），见 §1 第四步的红色框。
- **KVQuant**，arXiv:**2401.18079v4** —— **逐字读到**。用到的原文：§3.2 的 "**all the prior cached Keys will need to be updated**" 与 "online calibration for this **does not require updating prior cached Values**"；3.82 / 0.82 / 0.19 三个增量；"**4.8× longer context length**"。
- **LLM.int8()**，arXiv:2208.07339v2 —— outlier 的**持久性**（"某通道有 outlier 则它在所有 token 上都出现；同一通道跨 token 的幅值方差很小"）。**计划基线，本文实现者未逐条重读原文。**
- 本仓库 [01-kv-cache-basics](18-kv-cache-basics.md) —— KV cache 的字节账与 decode 的 memory-bound（本文不回述）。
- 本仓库 [06-granularity](06-granularity.md) —— 四种粒度的定义、`axis` 语义的完整版本、以及「领先幅度随张量宽度变化」的原始实测。
- 本仓库 [07-outliers-and-propagation](07-outliers-and-propagation.md) —— outlier 的持久性与 massive activations 的区分。
- 本仓库 [04-integer-formats](04-integer-formats.md) —— 对称 vs 非对称、零点取整的 `|δ| ≤ s/2`。

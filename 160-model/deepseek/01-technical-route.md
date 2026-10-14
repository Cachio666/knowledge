---
title: DeepSeek 的技术路线：从 MLA 到 CED + CSA2
slug: deepseek-technical-route
domain: 160-model
topic: deepseek
level: 2
difficulty: advanced
prerequisites:
  - 80-model-architectures/attention-variants/02-mla-low-rank
  - 80-model-architectures/moe/01-gating-and-routing
related:
  - 160-model/deepseek/02-v4-1-flash-architecture
  - 160-model/glm/02-glm5-architecture
  - 120-inference/kv-cache/01-capacity-planning
code:
  - code/01_param_budget.py
status: stable
updated: 2026-10-13
---

# DeepSeek 的技术路线：从 MLA 到 CED + CSA2

> DeepSeek 这条线的**目标函数换过两次**。V2 到 V3.2 在解「算不动」——MLA 压 KV、DSA 砍注意力；V4 之后在解「**存不下、搬不动**」——CSA/HCA 压序列维、CSA2 压层维、CED 砍掉一半 prefill，最后把全局 KV 从 V1 的 **389 120 字节/token** 压到 **890 字节/token（437×）**。理解这条线的关键不是记住每个缩写，而是看清它每代在压缩**哪一个维度**。

## 0. 背景与动机

先看一组数（全部来自官方技术报告，见「参考」）：

| 代 | 代表模型 | 总参 / 激活 | 上下文 | 每 token 全局 KV |
|---|---|---|---|---|
| V1 | DeepSeek-V1 | — | — | **389 120 B** |
| V3.2 | DeepSeek-V3.2 | 671B / 37B | 128K | — |
| V4 | V4-Pro | 1.6T / 49B | 1M | V3.2 的 **10%** |
| V4 | V4-Flash | 284B / 13B | 1M | 3560 B（由 890×4 反推） |
| **V4.1** | **V4.1-Flash** | **552B / 8B(prefill)·16B(decode)** | **1M** | **890 B** |

（KV 那两行由 `code/01_param_budget.py` 复算，见「编译与运行」。）

**437 倍**这个数字是这条线最浓缩的一句话。它意味着：如果 V1 时代一条 1M 上下文序列的 KV 是 389 GB，V4.1-Flash 只要 **0.89 GB**。

为什么必须做这件事？因为 agent 的工作负载形态变了。一个跑几小时的编码 agent，**每次工具调用都产生一次新的 prefill 请求，上下文只增不减，而真正需要生成的 token 是少数**——这是所谓 **input-heavy** 负载。V4.1 官方报告把随之而来的瓶颈拆成三块：

- **HBM 容量**限制运行时能同时装下多少并发请求的全局 KV；
- **SSD / 主机内存容量**限制前缀缓存能存多久、命中率多高；
- **I/O 与互联带宽**限制缓存迁移与加载的速度。

三块合起来决定了一个 agent 服务的吞吐上限与单位成本。**「算」已经不是瓶颈了**——DSA 一代（`160-model/glm/02-glm5-architecture` §1 第三步）已经把注意力的匹配算力降了一个量级。于是 DeepSeek 把全部火力对准「存」和「搬」。

## 1. 原理

### 第一步：把 KV 的账写成三个可乘的维度

这是读懂 V4 → V4.1 全部设计的**唯一框架**。每 token 的全局 KV 字节数可以写成四个因子的连乘：

$$B_{\text{KV/token}} = \underbrace{b_{\text{entry}}}_{\text{① 条目精度}} \times \underbrace{\frac{1}{m}}_{\text{② 序列维压缩}} \times \underbrace{\frac{n_{\text{store}}}{n_{\text{layers}}}}_{\text{③ 层维复用}} \times \underbrace{c_{\text{entry}}}_{\text{④ 条目宽度}}$$

四个因子对应四代不同的武器：

| 维度 | 武器 | 谁引入 |
|---|---|---|
| ① 条目精度 | BF16 → FP8 → **FP4（E2M1 + E4M3 每 16 通道 scale）** | V4 索引器 Q/K；V4.1 main KV |
| ② 序列维 | 每 $m$ 个 token 压成一条：**CSA**（$m=4$）与 **HCA**（$m'=128$）；V4.1 用 **CSA2**（encoder $m=2$、decoder $m=1$） | V4 / V4.1 |
| ③ 层维 | **CSA2 跨层复用** main KV / indexer K / Top-K 索引（V4.1 只有 4/40 层真正新建 KV） | V4.1 |
| ④ 条目宽度 | **MLA** 让各头共享一份低秩 latent（576 维而不是 2048 维） | V2 |

用 V4.1 的实际配置复算（`code/01_param_budget.py` 的 C1，**这是反推、不是官方明细**）：只有 **Full Mode** 的层需要产出并保存 main KV——

- encoder 3 组 × 每组 1 个 Full 层 = **3 层**，压缩率 $m=2$；
- decoder 5 组里只有第 1 组首层是 Full（其余组首层是 Reindex，复用别层的 KV）= **1 层**，$m=1$；
- 每条的宽度是 head dim **512**，FP4 下 **256 B**；indexer K 是 **128** 维。

$$\text{indexer K 取 FP4} \Rightarrow 3\times\frac{256+64}{2} + 1\times\frac{256+64}{1} = 800\ \text{B}$$
$$\text{indexer K 取 FP8} \Rightarrow 3\times\frac{256+128}{2} + 1\times\frac{256+128}{1} = 960\ \text{B}$$

官方给的数是 **890 B**——**正好落在两个读法之间**。这不是「谁算错了」，而是报告只公布了总数、没公布「哪些层存多少字节」的明细。本仓库把这个区间做成可复核的断言，并把它标为**反推**。

> ⚠ 判据：**不要**用「层数 × 每层 KV」这种线性直觉去算 CSA2 的 KV。按那个算法会得到 $40\times512\times0.5 = 10\,240$ B，比真值大 **11.5 倍**——它漏掉的正是③。

### 第二步：CED —— 用「把 decoder 的 KV 从 encoder 借来」砍掉一半 prefill

V4.1 的主干 40 层被切成两半：前 20 层是 **causal encoder**，后 20 层是 **decoder**。decoder 的全局 KV **不是**从自己的隐藏状态算的，而是**从第 20 层的隐藏状态投影出来**：

$$C^{l} = H^{L/2} W^{KV}_{l}, \qquad Z^{l} = H^{L/2} W^{Z}_{l}, \qquad l > \tfrac{L}{2}$$

（$C$ 是 KV 条目，$Z$ 是对应的压缩权重；每层有**自己的**投影矩阵，这是 CED 相对微软 YoCo 的加厚之处——YoCo 让上半部分直接共享下半部分那一份 KV，CED 给每层配独立投影，在同一份输入上换出不同的 KV。）

后果是 prefill 的层遍历从 $O(NL)$ 变成：

$$\underbrace{\frac{L}{2}\cdot N}_{\text{encoder 过完整 prompt}} + \underbrace{\frac{L}{2}\cdot n_{\text{win}}}_{\text{decoder 只为 SWA 回放}}\Big/ \underbrace{N L}_{\text{朴素}} = \frac{1}{2} + \frac{n_{\text{win}}}{2N}$$

**这一个式子给出了 CED 的全部边界行为**（`code/02_ced_csa2_budget.py` 逐条断言）：

| prompt 长度 $N$ | CED / 无 CED | 含义 |
|---|---|---|
| $N = n_{\text{win}} = 128$ | **1.0000** | 完全没有收益 |
| $N = 64$（比窗口还短） | **1.5000** | **比不用 CED 还慢** |
| $N = 1$M | **0.5001** | 渐近折半 |

**CED 只作用于 prefill。** decoder 仍然跑满 40 层，所以 decode 的激活量是 prefill 的整整 2 倍——这就是官方那对「**输入 8B、输出 16B**」数字的来源，也是 `code/01_param_budget.py` 的 B3 断言：

$$\text{prefill} = 20 \times 400\text{M} = 8.00\text{B}, \qquad \text{decode} = 40 \times 400\text{M} = 16.00\text{B}$$

其中每 token 每层的 400M 激活由「MoE 的 247.7M + 注意力 152.3M」构成。把 152.3M 代回 40 层，得到主干总参 **551.1B vs 官方 552B（差 +0.17%）**——这说明这套账是自洽的。

### 第三步：CSA2 —— 三种模式把「层」这一维也压掉

CSA2 给每个注意力层**静态分配**三种模式之一，区别只在「main KV / indexer K / Top-K 索引」这三样东西从哪来：

| 模式 | main KV | indexer K | Top-K 索引 | 代价 |
|---|---|---|---|---|
| **Full** | 自己算 | 从自己的 main KV 投影 | 自己跑 indexer | 完整路径 |
| **Reindex** | 复用前面某层 | 复用同一份 | **自己重打分** | 只跑 indexer Q |
| **Reuse** | 复用 | 复用 | 复用 | 连 indexer Q 都不算 |

关键设计是 **「缓存共享」与「选择复用」解耦**：Reindex 层共享了 KV，但仍然换出自己的 top-k，所以**稀疏选择可以逐层变化**，而缓存不必逐层复制。三种模式下每层**都保留自己的全局 Q 与 SWA KV**，层间的表达能力没有被抹平。

配比（V4.1 的实际配置）：encoder 18 个 CSA2 层分 3 组各 6 层，每组 `Full + Reuse×5`；decoder 20 层分 5 组各 4 层，第 1 组 `Full + Reuse×3`，其余 4 组 `Reindex + Reuse×3`。于是**只有 4/40 = 10% 的层真正新建 main KV**。

### 第四步：层级稀疏索引器 —— 让后续 indexer 的打分域与上下文长度脱钩

跨层复用省掉了 indexer 的**执行次数**，但剩下的 indexer 仍然要扫**整个因果可见的上下文**——长上下文下这依然是瓶颈。V4.1 的做法：decoder 里第一个 Full 层在打分的同时做 **block 级候选选择**（每个 block 取块内最大分，选分最高的 block），把命中 block 覆盖的位置收进一个**共享候选池**；后续 Reindex 层**只在这个池里打分**。

$$2048\ \text{block} \times 8\ \text{位置/block} = 16\,384\ \text{候选位置}$$

在 1M 上下文下，后续 indexer 的打分域缩小 **恰好 64×**（$1\,048\,576 / 16\,384 = 64.0$）。而最终注意力只读 **top-k = 512** 条，是候选池的 1/32。

这个机制有一个必须写清的边界：**候选池一旦大于等于上下文，收益归零**（上下文 ≤ 16 384 时缩减比就是 1×）。它是个长上下文专属的优化。

### 第五步：FP4 KV 与 SWA Bounded Replay —— 精度与部署各出一刀

**FP4 main KV**：格式是 E2M1 配**每 16 通道一个 E4M3 缩放因子**（官方口径：接近 NVFP4，但**省掉 NVFP4 的第二级 per-tensor 全局 scale**）。报告称训练时就用 FP4 全局 KV，性能只有边际退化。

**SWA Bounded Replay**：这一刀在部署层。SWA 的 KV 依赖逐层累积，**精确**重建 $L$ 层的 SWA KV 需要回放 $L \times n_{\text{win}} = 20\times128 = 2560$ 个 token；V4 报告提过 Zero SWA Caching，但生产中被证明代价过高。V4.1 干脆**接受近似**：只回放最近 $n_{\text{win}} = 128$ 个 token 并把 SWA 截断在回放段内——回放量直接降 **20×**。

换来的收益是缓存策略的解放：

| | 存哪 | TTL | 为什么 |
|---|---|---|---|
| **全局 KV** | 持久化缓存（SSD） | **≥ 72 小时** | 有长尾复用价值 |
| **SWA KV** | 每机 **10% 主机 DRAM** 组成的分布式内存池 | **几分钟** | 只在活跃会话的分钟级窗口内有用，会话一结束就是死数据 |

两者的访问模式根本不同，硬放在一起就是浪费。这就是持久化 KV 能压到 **1/8** 的原因——它是两个因子相乘：**不再持久化 SWA KV（几乎减半，×0.5）× 全局 KV 压缩到 1/4（×0.25）= 0.125**。这个乘法关系被写成 `code/02_ced_csa2_budget.py` 的 B6 断言，因为它是「1/8」这个数字唯一站得住的解释。

## 2. 执行映射

§1 的每一刀落到硬件上都会变成具体的 kernel 与调度改动。

**推理的 kernel 数被压到了两位数。** V4.1 报告的原话是：融合之后，**绝大多数 Transformer 层（CSA2 处于 Reuse Mode 的那些）prefill 只执行 15 个 kernel、decode 只执行 11 个**。参与的融合算子有名字：

| kernel | 在哪 | 融了什么 |
|---|---|---|
| `fused-RoPE-attention-RoPE-cast` | FlashMLA | RoPE、注意力、再 RoPE、类型转换串成一条 |
| `Mega-Gate` / `Mega-mHC` / `Mega-MoE` | DeepGEMM | 门控、残差混合、MoE 三段各自成为一个超级算子 |
| TopK kernel | **DeepSelect** | DSA / V4 / V4.1 共用的稀疏选择 |

`Mega-mHC` 是 Single-Pass mHC 的产物，它的账值得单独看：mHC 维护 $n$ 条残差流，按理残差变换的下界是 $(2n+2)d$ 的激活访存。DeepSeek-V4 的实现要三个 kernel 串行（残差更新 → 系数预测 → 输入混合），实际访存 $(4n+4)d$，**正好是下界的两倍**。原因是输入混合系数 $A_l$ 必须等隐藏维上的规约全部完成才能拿到。

V4.1 的解法是把混合系数**错开一个 block**——每个 block 消费上一个 block 产出的系数，依赖关系消失，三步融进一个 `Mega-mHC` kernel，访存从 $(4n+4)d$ 降到 $(2n+2)d$，**正好减半**。报告说这个错位带来的性能损失可以忽略。

**部署侧的 EPD 分离。** V4.1 用 **Encoder–Prefill–Decode 三段分离**，让视觉编码、prefill、解码各自独立扩缩容并重叠执行。这与 GLM-5.3-Flash 的选择一致（`160-model/glm/03-serving-efficiency-and-deployment` §2），说明多模态 + 长上下文时代「一段式部署」已经不够用。

**三个值得记住的量级：**

| 量 | 值 |
|---|---|
| 1M 上下文每 token 全局 KV | **890 B**（V4-Flash 的 1/4，V1 的 1/437） |
| Reuse Mode 层的 kernel 数 | prefill **15** / decode **11** |
| decode 激活量 / prefill 激活量 | **2.0×**（16B / 8B） |

## 3. 实现与演进

| 代 | 改了什么 | 为什么必须走到下一步 |
|---|---|---|
| **V2**（2024-05） | **MLA** + DeepSeekMoE：236B 总 / 21B 激活，128K | KV 的**条目宽度**降下来了，但它仍然随序列长度与层数线性增长 |
| **V3**（2024-12） | 671B / 37B；**FP8 训练**；**aux-loss-free** 均衡；**MTP** | 稠密注意力在 128K 时仍是 $O(L^2)$ |
| **R1**（2025-01） | 纯 RL（GRPO）激发推理 | 推理能力靠 RL 起来后，**长上下文与 agent 负载**成了新战场 |
| **V3.1**（2025-08） | 混合思考模式（同一权重支持 thinking / non-thinking） | 思考模式会拉长输出，进一步放大 KV 压力 |
| **V3.2**（2025-12） | **DSA**：lightning indexer（ReLU）选 top-2048，只对子集做注意力；从 V3.1-Terminus 的 128K checkpoint **续训 943.7B token** | 匹配算力降了，但 indexer 自己仍是 $O(L^2)$，且**层数维还没被碰过** |
| **V4**（2026-04 预览 / 08 GA） | **CSA + HCA 交错**（序列维压缩）+ **mHC** + **Muon**；1M 上下文；32T+ token | KV 的**序列维**压了，但每层仍各自存 KV |
| **V4.1-Flash**（2026-09-10） | **CED**（prefill 折半）+ **CSA2**（层维复用 + 层级索引器）+ **FP4 KV** + **SWA Bounded Replay** | 用 **1/3 总参、1/4 激活参**在基座评测上追平 V4-Pro-Base |

三处关键跳变的机理：

**V3.2 的 DSA：续训而不是重训。** 从 V3.1-Terminus 的 128K checkpoint 出发，分 dense warm-up 与 sparse training 两阶段，稀疏阶段用 lr 7.3e-6、15000 步 × 480 条 × 128K 序列 = **943.7B token**，每 query 选 **2048** 个 KV。这与 GLM 的做法同源（GLM 只用 20B token 做稀疏适配，见 `160-model/glm/01-glm-technical-route` §3）——**稀疏注意力是可以低成本后装的改造**。

**V4 的 CSA/HCA：把压缩率分成两档再用。** CSA 先按 $m$ 个 token 压成一条、再做 DSA；HCA 压得更狠（$m' = 128$）但保持**稠密**注意力。两者**交错**：V4-Pro 前 2 层 HCA，其余交错。1M 上下文下 V4-Pro 只需 V3.2 的 **27% 单 token FLOPs 与 10% KV**（原文单位：equivalent FP8 FLOPs）。

**V4.1 的三层同时下手。** 报告自己的总结值得逐字引用：架构层用 CED 砍一半 prefill、用 CSA2 把跨层复用做到**三个维度全覆盖**；精度层把 main KV 压到 FP4；部署层用 bounded replay 把 SWA KV 整个赶出持久化缓存。**任何单独一项都只有百分之几十的改善，叠起来才是 437×。**

## 4. 踩坑与失败模式

### 4.1 把 CED 说成「整体快一倍」

**现象**：写「CED 让 V4.1-Flash 推理速度翻倍」，或按 8B 估算整个推理成本。
**成因**：CED 只减少 **prefill** 的层遍历。decode 仍要跑满 40 层，激活量 16B；MoE 的 FFN 一点没少。官方把 8B 与 16B 并列写出来，正是要人分清这两个阶段。
**规避**：凡看到「输入激活 / 输出激活」这类不对称数字，先问「哪一阶段跑多少层」。判据：CED 的 prefill 比值是 $1/2 + n_{\text{win}}/(2N)$，**只对 prefill 成立**。

### 4.2 把 V4-Pro 的 27% / 10% 套到别的型号上

**现象**：引用「V4 只需 V3.2 的 27% FLOPs、10% KV」来描述 V4-Flash 或 V4.1-Flash。
**成因**：V4 报告正文只给了 **V4-Pro** 这一档。V4-Flash 只是「further」这种定性表述；V4.1-Flash 给的是**另一组**数（890 B/token、1/4 HBM、1/8 SSD）。三者的激活参数量分别是 49B / 13B / 16B，效率曲线不同。
**规避**：效率百分比必须与型号绑定。找不到对应型号的一手数字时，**写「未公布」而不是外推**。

### 4.3 用「KV = 层数 × 每层字节」算 CSA2

**现象**：算出 V4.1-Flash 的 KV 约 10 240 B/token，比官方 890 B 大一个数量级，于是怀疑官方数字。
**成因**：漏了 CSA2 的**层维复用**。只有 Full Mode 层新建 main KV（4/40），Reindex 复用 KV、Reuse 连索引都复用。
**规避**：把 KV 的账写成 §1 第一步那四个因子的连乘，逐项问「这一项这一层是多少」。判据表里有对应的可机械判定行。

### 4.4 把官方的「1/4 激活参数」当成能算出来的数

**现象**：看到报告写「用 **1/3 总参、1/4 激活参**追平 V4-Pro-Base」，就去验算，结果对不上。
**成因**：$552/1600 = 0.345$（≈1/3 ✓），但激活那侧 $8/49 = 0.163$、$16/49 = 0.327$——**两个都不是 1/4**。报告没说明这 1/4 的分母是哪个口径（V4-Pro 的 prefill 激活？平均激活？还是别的基线）。
**规避**：这条被 `code/01_param_budget.py` 的 C2 断言固定成「**1/3 成立、1/4 对不上**」这个可复核的事实。引用时写原话并注明未复现，**不要替它选一个分母**。

### 4.5 把 bounded replay 当成无损

**现象**：认为 SWA Bounded Replay 只是「重算一下」，等价于精确重建。
**成因**：报告明说重放出来的前缀状态是**近似**的——被回放的 token 只重新生成 SWA KV、复用已缓存的全局 KV，因此未缓存后缀的 global KV 与 SWA KV **依赖于 cache-hit 的位置，在不同的命中位置上并不数学等价**。官方为此在**后训练时模拟同样的 replay**做 train-aware 适配。
**规避**：把「接受近似」当成设计的一部分而不是 bug。凡引用 bounded replay，必须同时说「近似」与「后训练已对齐」。

## 编译与运行

```sh
cd code
make run                  # 默认 python3 + numpy，依次跑 01 / 02
make 01_param_budget      # 只跑本文件对应的程序
make 02_ced_csa2_budget   # 只跑实例分析对应的程序
make clean
```

本文的 8B/16B 推导、437× 的 KV 压缩、以及 §1 第一步那个 800–960 B 的**反推区间**，来自 **`code/01_param_budget.py`**；CED 的 $1/2 + n_{\text{win}}/(2N)$、64× 索引器缩减、20× 回放缩减、$0.5\times0.25=1/8$ 来自 **`code/02_ced_csa2_budget.py`**。两个程序只用 `math` 与 `numpy`，**不加载权重、不 import 推理框架**，产出是**推导值**。

## 5. 判据表

本域无 GPU、知识对象以方法与规格为主，§5 采用**判据表**形态（理由见 `160-model/README.md`）。

| 可机械判定的条件 | 结论 |
|---|---|
| 规格给出「输入激活 X / 输出激活 Y」且 $Y = 2X$ | 该模型是 CED 类结构：$Y$ 对应 decode（全层），$X$ 对应 prefill（前半层） |
| 要估 prefill 的收益，而 prompt 长度 $N \le n_{\text{win}}$ | CED 无收益（比值 1.0）；$N < n_{\text{win}}$ 时**比不用更慢** |
| 规格同时给 `encoder_compression` 与 `decoder_compression` | 两者是**不同的** $m$；KV 只由 **Full Mode** 层新建，Reindex/Reuse 不新建 |
| 用「层数 × 每层 KV 字节」算 CSA2 的 KV | 该算法必然高估一个数量级；必须按「新建 KV 的层数 / 压缩率」算 |
| 规格给出候选池大小（block × 每块位置） | 只有**上下文 > 候选池**时层级索引器才有收益；否则缩减比为 1× |
| 规格给出「持久化 KV 压到 1/N」 | 必须能写成若干**因子相乘**（如不再存 SWA × 全局 KV 压缩）；乘不出来的 1/N 不予采信 |
| 官方倍数（27% / 10% / 1/4）未注明适用型号与阶段 | 记为该型号该阶段的引用值；**不得跨型号套用** |
| 某结构的收益被描述成「整体加速」 | 先问它作用于 prefill、decode 还是两者；CED 只作用于前者 |
| 稀疏注意力被描述成「无损」 | 区分两类：**只选不丢**（DSA 系）与 **近似重建**（bounded replay）；后者必须另有 train-aware 适配 |

## 6. 小结与延伸

- **目标函数换过两次**：V2–V3.2 解「算不动」（MLA 压条目宽度、DSA 砍注意力），V4 之后解「存不下、搬不动」（CSA/HCA 压序列维、CSA2 压层维、CED 砍 prefill、FP4 压精度）。
- **唯一该记的框架**：KV 字节 = 条目精度 × 序列压缩 × 层维复用 × 条目宽度。V4.1 的 890 B/token 是这四个因子同时被拧的结果。
- **三个可复算的数**：prefill/decode 激活 = 8B/16B（CED 的 20:40 层）；持久化 KV 的 1/8 = 0.5 × 0.25；后续 indexer 打分域缩 64×。
- **两处口径缺口**：890 B 的逐层明细（反推区间 800–960 B）、「1/4 激活参」的分母。

下一步：`160-model/deepseek/02-v4-1-flash-architecture` 把 V4.1-Flash 逐部件拆开（CED、CSA2 三模式、层级索引器、mHC、Engram、DSpark）；横向对照见 `160-model/glm/02-glm5-architecture`（GLM 走的是 IndexShare 那条路）。

## 参考

- 【C】DeepSeek-AI, *DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression*（官方技术报告 PDF）。**取回渠道**：huggingface.co 在本机不可达，用的是 ModelScope 上的同名官方文件（1 809 802 B），`pdftotext -layout` 转文本后逐节读；**未逐字核对 HF 原文件与副本的一致性**。摘要、§2.1–§2.4、§3.2、§4.2.1。
- 【C】DeepSeek-AI, *DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence*, arXiv:[2606.19348v1](https://arxiv.org/abs/2606.19348)。摘要、§2.2–§2.4、§4.2.1。
- 【C】DeepSeek-AI, *DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models*, arXiv:[2512.02556v1](https://arxiv.org/abs/2512.02556)。§2.1 DSA。
- 【C】DeepSeek-AI, *DeepSeek-V2*, arXiv:[2405.04434](https://arxiv.org/abs/2405.04434)（abs 页摘要）。
- 【C】DeepSeek-AI, *DeepSeek-V3*, arXiv:[2412.19437](https://arxiv.org/abs/2412.19437)（编号经 V4 报告参考文献确认；**数值未逐字核对原文**）。
- 【B】[DeepSeek API Docs · DeepSeek-V4.1-Flash Release 2026/09/10](https://api-docs.deepseek.com/news/news260910/)（发布页与历史发布条目导航）。
- 【B】[DeepSeek 官网 · Introducing DeepSeek-V4.1-Flash](https://www.deepseek.com/en/news/deepseek-v4-1-flash/)。
- 【D】机器之心/36氪《DeepSeek V4.1 Flash 核心技术解析》（2026-09-14）——二次解读，仅用于交叉理解，**正文数字不以它为准**。
- 相关域：[80-model-architectures/attention-variants](../../80-model-architectures/attention-variants/)、[120-inference/kv-cache](../../120-inference/kv-cache/)。

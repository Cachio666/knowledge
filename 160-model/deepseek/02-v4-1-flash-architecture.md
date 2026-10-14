---
title: DeepSeek-V4.1-Flash 实例分析：CED + CSA2 + FP4 KV
slug: deepseek-v41-flash-architecture
domain: 160-model
topic: deepseek
level: 3
difficulty: advanced
prerequisites:
  - 160-model/deepseek/01-technical-route
  - 80-model-architectures/moe/01-gating-and-routing
related:
  - 160-model/kimi/02-k3-architecture
  - 160-model/glm/03-serving-efficiency-and-deployment
  - 40-kernels/attention/02-flash-attention
code:
  - code/02_ced_csa2_budget.py
status: stable
updated: 2026-10-13
---

# DeepSeek-V4.1-Flash 实例分析：CED + CSA2 + FP4 KV

> 用 **552B 主干**（8B prefill / 16B decode 激活）在基座评测上追平 **1.6T** 的 V4-Pro，靠的不是把模型做小，而是把**长上下文服务里最贵的三样东西同时压掉**：prefill 的层遍历（CED 折半）、KV 的存储（CSA2 跨层复用 + FP4，压到 890 B/token）、以及 SWA KV 的持久化（bounded replay，把它彻底赶出 SSD）。全文的账都可以在一张纸上复算。

## 0. 背景与动机

先看官方的定位（发布页与报告一致）：

| | V4-Pro | V4-Flash | **V4.1-Flash** |
|---|---|---|---|
| 总参 | 1.6T | 284B | **552B 主干 + 196B Engram** |
| 激活 | 49B | 13B | **8B（prefill）/ 16B（decode）** |
| 上下文 | 1M | 1M | 1M |
| 每 token 全局 KV | — | 3560 B（反推） | **890 B** |
| 结局 | 计划下线（后又保留） | **已下线** | 接管 `deepseek-flash` |

一升一降同时发生：**参数量降到 V4-Pro 的 1/3，KV 降到上一代的 1/4，而公开评测上反超 V4-Pro**。要理解它为什么成立，得先接受一个视角转换：

> 在 input-heavy 的 agent 负载里，**「一次调用要占多少显存、从 SSD 搬多少数据、重算多少次 prefill」比「模型多大」更决定成本**。

V4.1 的每一处设计都指向这三个量。这也是它被 Sebastian Raschka 认为「该叫 V5」的原因——不是参数更多，而是**目标函数换了**。

## 1. 原理

### 第一步：CED —— decoder 的 KV 从哪来

常规 Transformer 里每层的 K/V 都从**该层自己的隐藏状态**算出来，所以处理一段 prompt 必须把 40 层全部跑完。V4.1 把这 40 层切成 20 + 20，并让后半部分的全局 KV **改从第 20 层的隐藏状态投影**：

$$C^{l} = H^{L/2}W^{KV}_{l}, \qquad Z^{l} = H^{L/2}W^{Z}_{l}, \qquad l > \tfrac{L}{2}$$

三个要点：

1. **每层有自己的投影矩阵**（$W^{KV}_l$、$W^Z_l$），不是所有 decoder 层共用同一份 KV。这是 CED 相对其前身 YoCo 的加厚：YoCo 让上半部分直接共享下半部分那一份 KV，CED 在**同一份输入**上换出不同的 KV——在保住「只算一半」的前提下扩大了 KV 的有效容量与生成深度。
2. **SWA 不参与这个共享**：每一层的局部 K/V 仍从**本层**隐藏状态算，以维持局部信息的计算深度。代价是 decoder 需要回放，见第五步。
3. 于是 prefill 只需跑 encoder 的 20 层，**层遍历折半**：

$$\frac{\text{CED}}{\text{朴素}} = \frac{1}{2} + \frac{n_{\text{win}}}{2N} \quad\xrightarrow{N = 1\text{M}}\quad 0.5001$$

而 decode 仍跑满 40 层。**这就是「输入 8B、输出 16B」的全部来源**，也是 `code/01_param_budget.py` 的 B3/B4 断言：

$$\text{prefill} = 20 \times (247.7\text{M} + 152.3\text{M}) = 8.00\text{B}, \qquad \text{decode} = 40 \times 400.0\text{M} = 16.00\text{B}$$

其中每层 MoE 激活 **247.7M**（$7 \times 3\times5120\times2304$）、反解出的注意力激活 **152.3M**。把 152.3M 代回 40 层，主干总参回到 **551.1B vs 官方 552B（+0.17%）**——这套账自洽。

### 第二步：CSA2 的三种模式

CSA2 给每个注意力层**静态**指定一种模式，区别只在「main KV / indexer K / Top-K 索引」来自哪里：

| 模式 | main KV | indexer K | Top-K 索引 | 本层还要算什么 |
|---|---|---|---|---|
| **Full** | 自己算 | 从自己的 main KV 投影 | 自己跑 indexer | 主 Q、SWA KV、indexer Q |
| **Reindex** | 复用前面某层 | 复用同一份 | **自己重打分** | 主 Q、SWA KV、indexer Q |
| **Reuse** | 复用 | 复用 | 复用 | 主 Q、SWA KV |

三种模式**都保留自己的全局 Q 与 SWA KV**——「缓存共享」与「选择复用」被解耦了：Reindex 层共享 KV 但换出自己的 top-k，所以**稀疏选择可以逐层变化**而缓存不必逐层复制。

实际配比（40 层）：

```
encoder（m=2）  3 组 × [Full, Reuse, Reuse, Reuse, Reuse, Reuse]        = 18 层
decoder（m=1）  [Full, Reuse, Reuse, Reuse]
                + 4 × [Reindex, Reuse, Reuse, Reuse]                    = 20 层
前 2 层        纯 SWA                                                   =  2 层
```

**只有 4/40 = 10% 的层真正新建 main KV**（encoder 3 个 Full + decoder 1 个 Full）。这是「层」这一维被压掉的直接证据。

CSA2 还相对 V4 的 CSA 做了两处简化：**去掉相邻压缩条目的重叠源**、**去掉压缩时的绝对位置编码**，并把 indexer K 改为**从 main KV 投影**（而不是像 CSA 那样另走一条从隐藏状态出发的压缩路径）。官方给的理由是简化实现、提升训练效率。

### 第三步：KV 的账 —— 把 890 B 拆开

用「条目精度 × 序列压缩 × 层维复用 × 条目宽度」四项连乘（推导见 `160-model/deepseek/01-technical-route` §1 第一步），只算**新建 KV 的层**：

| 读法 | encoder（3 层，$m=2$） | decoder（1 层，$m=1$） | 合计 |
|---|---|---|---|
| indexer K 取 **FP4** | $3\times\frac{256+64}{2} = 480$ B | $256+64 = 320$ B | **800 B** |
| indexer K 取 **FP8** | $3\times\frac{256+128}{2} = 576$ B | $256+128 = 384$ B | **960 B** |

其中 main KV 每条宽度是 head dim **512**（FP4 → 256 B），indexer K 是 **128** 维。

**官方给的是 890 B——正好落在 800 与 960 之间。** 报告只公布了总数、没公布逐层明细，所以本仓库把这个区间标为**反推**并做成 C 类断言，而**不宣称复现了官方的拆法**。

同一套账的两个旁证：

- 相对 **DeepSeek-V1** 的 **437×**：$389\,120 / 890 = 437.2$ ✓（report 印值 437）
- 反推 **V4-Flash** 的 890 / 0.25 = **3560 B/token**
- **朴素算法**（每层各存一份 512 维 FP4 KV）给 $40\times512\times0.5 = 10\,240$ B，**高估 11.5×**——它漏掉了「层维复用」这一项。

### 第四步：层级稀疏索引器

跨层复用省掉了 indexer 的**次数**，但剩下的 indexer 仍要扫整个因果可见上下文。V4.1 让 decoder 里**第一个 Full 层**在打分时顺带做 block 级筛选：每个 block 取块内最大分，选分最高的 block，把命中位置收进**共享候选池**。

$$2048\ \text{block} \times 8\ \text{位置} = 16\,384\ \text{候选位置}$$

后续 Reindex 层**只在这个池里打分**，因此每个 query 的打分成本**与上下文长度脱钩**（从线性变常数）。

| 量 | 值 |
|---|---|
| 1M 上下文 → 候选池 | **64×** 缩小（$1\,048\,576/16\,384 = 64.0$） |
| 最终注意力读多少条 | top-k = **512**，是候选池的 **1/32** |
| 上下文 ≤ 16 384 时 | 缩减比 **1×**（收益归零） |

这是一个**训练感知**的机制，而且**在后训练阶段引入**：候选限制在训练与推理时施加得完全一致，所以更深的 indexer 是在它推理时所用的同一个搜索域里被优化的。

### 第五步：SWA Bounded Replay —— 接受近似，换掉一整个缓存类别

SWA 的 KV 依赖**逐层累积**：精确重建 $L$ 层的 SWA KV 需要回放 $L\times n_{\text{win}} = 20\times128 = 2560$ 个 token。V4 报告提过 Zero SWA Caching（完全不存、需要时重算），但**精确恢复的代价在生产中被证明过高**。

V4.1 的做法是**只回放最近 $n_{\text{win}} = 128$ 个 token**，并把 SWA 截断在回放段内：从位置 $s$ 开始的回放中，位置 $i$ 的 query 只看 $\max(s, i-W+1)$ 到 $i$ 的 SWA key。回放量降 **20×**（= $L/2$）。

代价被官方明确写出来了：重建的状态**不数学等价**于完整前向的结果，未缓存后缀的 global KV 与 SWA KV **依赖于 cache-hit 的位置**。为此他们在**后训练时模拟同样的 replay**做 train-aware 适配。

换来的是缓存策略的解放——两类 KV 的访问模式根本不同，硬放在一起就是浪费：

| | 存哪 | TTL | 理由 |
|---|---|---|---|
| **全局 KV** | 持久化缓存（SSD） | **≥ 72 小时** | 有长尾复用价值 |
| **SWA KV** | 每机 **10% 主机 DRAM** 的分布式内存池 | **几分钟** | 只在活跃会话的分钟级窗口内有用 |

于是持久化 KV 压到 **1/8**——它是**两个因子相乘**：不再持久化 SWA KV（几乎减半，×0.5）× 全局 KV 压缩到 1/4（×0.25）。`code/02_ced_csa2_budget.py` 的 B6 断言把这个乘法钉死，因为它是「1/8」唯一站得住的解释。

**把它写成一个指数。** 官方给的另一句话是「上下文从 4K 拉到 1M（**256 倍**），单 token decode FLOPs **只增加 1/4**」。假设 $F \propto L^{\alpha}$：

$$256^{\alpha} = 1 + \tfrac{1}{4} = 1.25 \quad\Longrightarrow\quad \alpha = \frac{\ln 1.25}{\ln 256} \approx \mathbf{0.0402}$$

$\alpha = 0$ 就是严格与长度无关。**0.0402 意味着这套架构把 decode 的上下文复杂度压到了近似常数**——这是 CSA2（序列维 + 层维压缩）与 FP4（精度）合起来的结果，也是「1M 上下文可以常规支持」这句话的量化形态。

### 第六步：三个结构性扩展

**Single-Pass mHC。** mHC 在相邻 block 之间维护 $n$ 条残差流，$X_{l+1} = B_lX_l + C_lF_l(A_lX_l)$，系数 $(A_l,B_l,C_l)$ 由 $X_l$ 预测。理想情况下残差变换的激活访存下界是 $(2n+2)d$；V4 的三 kernel 串行实现要 $(4n+4)d$，**正好两倍**——因为输入混合系数 $A_l$ 必须等隐藏维上的规约全部完成。V4.1 把混合系数**错开一个 block**（每个 block 消费上一个 block 产出的系数），依赖消失，三步融进 `Mega-mHC` 一个 kernel，访存降到 $(2n+2)d$，**正好减半**。扩展因子 $n=4$，Sinkhorn-Knopp 迭代 20。

**Engram。** **196B 参数平均分给两个模块**，放在**第 1 层与第 14 层**：{2,3,4} 阶 N-gram 哈希、8 个哈希头、每阶总嵌入维 2048、每个头约 1600 万条目、表大小取**互不相同的质数**。嵌入表与投影都用 FP8，推理时靠确定性寻址从主机内存**后台 RDMA 预取**（第一个模块的预取与第一个 Transformer block 的计算重叠）。相比原始设计**去掉了短因果卷积**。思路是让 MoE 专心做组合推理，把「记忆」卸载到便宜的 DRAM。

**DSpark。** 3 个 Transformer block 组成 drafter，滑动窗口 128；一次前向**并行给出 5 个草稿位置**的 base logits，另有一个轻量 **Markov 头**建模草稿 token 间的依赖、一个**置信度头**预测逐位置接受概率；调度器结合引擎吞吐曲线**动态决定每个请求的验证长度**。关键区别：V3 的 MTP 是**全程与主干联合训练**的，而 DSpark **在预训练之后单独训练**，后训练阶段跟随主干但**不回传梯度**——好处是它能持续对齐不断演化的策略，同时加速线上服务与 RL 的 rollout 生成。

> ⚠ 后训练本身「没有算法创新」。报告直说：流程就是标准 SFT + RL + on-policy 蒸馏，「without any modification beyond well-established practice used in DeepSeek-V4 development」，所有实质改动都在**数据管线**上。唯一对使用者可见的设计是 **reasoning effort**：把 1–100 的标量写进系统提示，同档多个采样构成子组、组内做奖励中心化，**长度惩罚系数随 effort 指数衰减**（每增加一个特征尺度乘 $1/e$）。API 的 max / high / low 对应 **b = 100 / 75 / 50**。

## 2. 执行映射

**kernel 数被压到两位数。** 报告的原话：融合之后，**绝大多数层（CSA2 处于 Reuse Mode 的那些）prefill 只执行 15 个 kernel、decode 只执行 11 个**。参与的命名算子：

| kernel | 来源 | 融了什么 |
|---|---|---|
| `fused-RoPE-attention-RoPE-cast` | FlashMLA | RoPE → 注意力 → 再 RoPE → 类型转换 |
| `Mega-Gate` / `Mega-mHC` / `Mega-MoE` | DeepGEMM | 门控 / 残差混合 / MoE 各成一个超级算子 |
| TopK kernel | **DeepSelect** | DSA / V4 / V4.1 共用的稀疏选择 |

为什么 Reuse Mode 的层能这么少？因为它的「省」是**结构性的**：不算 indexer Q、不打分、不做 top-k，只拿现成的 KV 与索引做核心注意力。这也解释了为什么官方把模式做成**静态分配**而不是动态路由——**静态才能让编译器为三类层各生成一条固定的 kernel 流水线**。

**部署侧：EPD 分离。** Encoder / Prefill / Decode 三段独立扩缩容并重叠执行。加上 SWA KV 的分布式内存池与最长 72 小时的全局 KV 持久化，整体是一套**按访问模式分层**的存储体系：HBM（运行时全局 KV）→ 主机 DRAM 池（活跃会话的 SWA KV，分钟级）→ SSD（全局 KV 前缀缓存，≥72 小时）。

**三个量级：**

| 量 | 值 |
|---|---|
| Reuse Mode 层的 kernel 数 | prefill **15** / decode **11** |
| decode 激活 / prefill 激活 | **2.0×** |
| MoE FFN 占主干参数 | **98.7%**（545.0B / 552B） |

## 3. 实现与演进

### 3.1 KV 压缩的阶梯

| 代 | 武器 | 压的是哪个维度 | 每 token 全局 KV |
|---|---|---|---|
| V1 | MHA | — | 389 120 B |
| V2 | **MLA** | ④ 条目宽度（2048 → 576 维） | 大幅下降 |
| V3.2 | **DSA** | 算力（不是 KV） | 基本不变 |
| V4 | **CSA + HCA** | ② 序列维（$m=4$ / $m'=128$） | V3.2 的 10% |
| V4-Flash | 小激活版 | — | 3560 B |
| **V4.1-Flash** | **CSA2 + FP4 + bounded replay** | **③ 层维 + ① 精度 + 部署** | **890 B** |

这张表最值得看的是**顺序**：DeepSeek 先压条目宽度（V2），再压算力（V3.2），再压序列维（V4），最后才压**层维**（V4.1）。前三维都压过之后，「每层各自存一份 KV」这件事才变成主要矛盾——**这是典型的「瓶颈转移」而不是「一步到位」**。

### 3.2 为什么 CED 是这条线的转折点

V2 → V4 的所有武器都在**同一个假设**下工作：推理要跑完全部层。CED 打破了这个假设——它承认 **prefill 与 decode 的最优结构可以不同**：

```
prefill：输入重、可并行 → 只需要 encoder 的 20 层 + 少量 SWA 回放
decode ：逐 token、需要深 → 跑满 40 层，但 KV 已经由 CED 备好
```

于是一个模型有了**两组不同的激活参数量**。这也是为什么「V4.1-Flash 是 8B 模型还是 16B 模型」这个问题没有唯一答案——**取决于问的是哪个阶段**。

## 4. 踩坑与失败模式

### 4.1 以为 decoder 各层的 KV 来自各层自己

**现象**：读完「40 层」就按常规 Transformer 推 KV 与 prefill 成本，算出每层都要过完整 prompt。
**成因**：CED 的核心恰恰是**反常规**——decoder 的全局 KV 由第 20 层投影而来（每层有自己的投影矩阵），所以 prefill 不需要跑 decoder。
**规避**：见到「encoder–decoder」「两半」这类描述，先问「上半部分的 KV 从哪来」。判据：若规格给出「输入激活约为输出激活的一半」，几乎必然是 CED 类结构。

### 4.2 把候选池当成最终选择的 top-k

**现象**：看到 16 384 就觉得「这个模型一次注意 16 384 条 KV」，据此估算力。
**成因**：16 384 是**打分域**（indexer 在哪里找），不是**读取域**。真正的注意力只读 **top-k = 512** 条。两者相差 32 倍，且用途完全不同：候选池决定 indexer 的搜索成本，top-k 决定核心注意力的成本。
**规避**：把「候选池 / 打分域 / top-k / 读取 KV 数」四个概念分开。层级索引器的收益体现在**前者**（64×），不在后者。

### 4.3 以为 FP4 只是「存储时压一下」

**现象**：把 FP4 KV 理解成「部署时把 BF16 的 KV 重量化成 FP4」，因此认为它不影响训练。
**成因**：报告写的是「we use FP4 global KV caches **during training** with only marginal performance degradation」——训练时就是 FP4。而且格式是 E2M1 + **每 16 通道一个 E4M3 scale**（省掉了 NVFP4 的第二级全局 scale），这是一个**需要在训练中适应的数值格式**，不是后处理。
**规避**：区分「训练时量化」（QAT / 直接低精度训练）与「部署时量化」。前者的误差被模型吸收过，后者没有。

### 4.4 把 DSpark 当成 V3 的 MTP

**现象**：看到「投机解码模块」就按 MTP 理解，认为它和主干联合训练、跟着主干一起更新。
**成因**：V4.1 **在主干预训练时去掉了 MTP 模块**，DSpark 是**预训练之后单独训练**的；后训练阶段它跟随主干但**不回传梯度**。结构也不同：V3 的 MTP 是一个预测头，DSpark 是 3 个 block + Markov 头 + 置信度头，验证长度还是**动态**的。
**规避**：读投机解码的规格时问三件事——「几个 block」「验证长度固定还是动态」「梯度是否回传」。三者都决定它能不能跟上策略演化。

### 4.5 把「552B」当成模型总大小

**现象**：拿 552B 去和 V4-Pro 的 1.6T 比参数效率，得出「1/3 总参」的漂亮结论，却解释不了显存需求。
**成因**：**Engram 还有 196B 参数，且它要驻留在主机内存**（推理时确定性寻址 + RDMA 预取）。模型的实际参数量是 **552 + 196 = 748B**。发布页写的是「552B-parameter MoE」，那是**主干**口径。
**规避**：区分「主干参数」与「含记忆模块的总参数」。判据：若规格里出现独立的 `engram` / `memory` / `embedding table` 参数量与主机内存驻留描述，两者必须分别计入。

## 编译与运行

```sh
cd code
make run                  # 默认 python3 + numpy，依次跑 01 / 02
make 01_param_budget      # 参数与激活量核算（本文件 §1 第一步的 8B/16B）
make 02_ced_csa2_budget   # CED / CSA2 / 层级索引器成本账（本文件 §1 第二–五步）
make clean
```

本文的 8B/16B 推导、890 B 的**反推区间**、437× 与 11.5× 的对照、CED 的 $1/2+n_{\text{win}}/(2N)$ 边界、64× 索引器缩减、20× 回放缩减、$0.5\times0.25=1/8$，全部来自 `code/` 下的两个编号程序。它们只用 `math` 与 `numpy`，**不加载权重、不测硬件**，产出是**推导值**。

## 5. 判据表

本域无 GPU、知识对象以方法与规格为主，§5 采用**判据表**形态（理由见 `160-model/README.md`）。

| 可机械判定的条件 | 结论 |
|---|---|
| 规格给出 `encoder_layers` 与 `decoder_layers` 且相等 | 是 CED 类结构；prefill 只跑 encoder，decode 跑全部 |
| 规格给出「输入激活 / 输出激活」两个数且比值 = 2 | 与「decoder 层数 = 总层数」一致；两者可互相校验 |
| 给出 Full / Reindex / Reuse 的模式表 | 新建 main KV 的层数 = **Full 层数**；Reindex 复用 KV、Reuse 连索引都复用 |
| 用「层数 × 每层 KV 字节」算 KV | 必然高估约 10×；必须按「新建 KV 的层数 ÷ 压缩率」算 |
| 规格同时给出候选池大小与 attention top-k | 前者是 indexer 的**打分域**，后者是注意力的**读取域**；两者比例决定层级索引器的收益 |
| 候选池大小 ≥ 上下文长度 | 层级索引器收益为 1×（无收益）；它是长上下文专属优化 |
| 规格写明「训练时」就用某低精度 KV 格式 | 属训练时量化；不得按部署时量化的误差预期去估 |
| 规格出现独立的 Engram / 记忆参数量 | 总参 = 主干 + 记忆；记忆通常驻留主机内存，不进 HBM 权重账 |
| 规格给出「持久化 KV 压到 1/N」 | 必须能分解为若干因子相乘；分解不出来的 1/N 不予采信 |
| 某优化被描述成「无损」而机制是「只回放最近 $n_{\text{win}}$ 个 token」 | 属**近似**；引用时必须同时说明近似性与是否有 train-aware 适配 |

## 6. 小结与延伸

- **一句话**：V4.1-Flash 用 CED 砍掉一半 prefill、用 CSA2 把 KV 压到 890 B/token、用 bounded replay 把 SWA KV 赶出持久化缓存，三件事叠起来让 552B 的模型在 agentic 负载上比 1.6T 更划算。
- **四个可复算的数**：prefill/decode = **8B/16B**（20:40 层）；KV 从 V1 降 **437×**；持久化 KV 的 **1/8 = 0.5 × 0.25**；后续 indexer 打分域缩 **64×**。
- **两处必须标注的不确定**：890 B 的逐层明细（反推区间 **800–960 B**）；「1/4 激活参」的分母（$8/49 = 0.163$、$16/49 = 0.327$，都不是 1/4）。
- **一个方法论**：当匹配算力被压到不再是瓶颈时，**瓶颈会转移到存储与搬运**；此时架构创新的方向不是「算得更少」，而是「存得更少、搬得更少」。

下一步：`160-model/kimi/02-k3-architecture` 是同一问题的另一条解法——Kimi 用**线性注意力（KDA）**把 $O(L)$ 的 KV 变成 $O(1)$ 的状态，而不是压缩一份 $O(L)$ 的缓存；两条路线可以对照读。

## 参考

- 【C】DeepSeek-AI, *DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression*（官方技术报告 PDF）。**取回渠道**：huggingface.co 在本机不可达，用的是 ModelScope 上的同名官方文件（1 809 802 B），`pdftotext -layout` 转文本后逐节读；**未逐字核对 HF 原文件与副本的一致性**。摘要、§2.1–§2.4、§3.1.2–§3.2.2、§4.2.1、§5.1.1、附录 C。
- 【C】DeepSeek-AI, *DeepSeek-V4*, arXiv:[2606.19348v1](https://arxiv.org/abs/2606.19348)（§2.2 mHC、§2.3 CSA/HCA、§4.2.1 配置）。
- 【C】DeepSeek-AI, *DeepSeek-V3.2*, arXiv:[2512.02556v1](https://arxiv.org/abs/2512.02556)（DSA 原型）。
- 【B】[DeepSeek API Docs · V4.1-Flash Release](https://api-docs.deepseek.com/news/news260910/)（8B/16B、1/4 HBM、1/8 SSD、下线与路由策略）。
- 【B】[DeepSeek 官网 · Introducing DeepSeek-V4.1-Flash](https://www.deepseek.com/en/news/deepseek-v4-1-flash/)。
- 【D】机器之心/36氪《DeepSeek V4.1 Flash 核心技术解析》（2026-09-14）——含三个新开源仓库（deepseek-recipe / DeepJIT / DeepSelect）的转述；**DeepSelect 的 2–20× 加速属该级转述，本仓库未复现**。
- 相关域：[40-kernels/attention](../../40-kernels/attention/)、[120-inference/kv-cache](../../120-inference/kv-cache/)、[120-inference/serving-infra](../../120-inference/serving-infra/)。

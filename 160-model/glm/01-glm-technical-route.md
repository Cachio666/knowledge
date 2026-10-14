---
title: GLM 的技术路线：从 GLM-130B 到 GLM-5.3
slug: glm-technical-route
domain: 160-model
topic: GLM
level: 2
difficulty: intermediate
prerequisites:
  - 80-model-architectures/moe/01-gating-and-routing
  - 80-model-architectures/attention-variants/02-mla-low-rank
related:
  - 160-model/GLM/02-glm5-architecture
  - 160-model/GLM/03-serving-efficiency-and-deployment
  - 80-model-architectures/moe/03-balance-and-granularity
  - 120-inference/kv-cache/01-capacity-planning
code:
  - code/01_param_budget.py
status: stable
updated: 2026-10-13
---

# GLM 的技术路线：从 GLM-130B 到 GLM-5.3

> GLM 这条线只有三根主线：**FFN 稀疏化**（稠密 → MoE）、**注意力稀疏化**（GQA → MLA → MLA+DSA → IndexShare → linear+sparse 混合）、**能力来源从预训练转向后训练**。2026-08 的 GLM-5.3 是第三根主线的极端样本——**它和 GLM-5.2 共用同一个基座，架构一个字没改，提升全部来自后训练**。

## 0. 背景与动机

先看一组对照，把「为什么必须换路线」钉死。同一个 agent 任务，上下文 128K，比较两代模型在**同一台机器**上的两份开销：

| 项 | GLM-4.5（355B-A32B，GQA-8，89 MoE 层） | GLM-5（744B-A40B，MLA+DSA，75 MoE 层） |
|---|---|---|
| KV cache / token / 层 | 4096 B（2 × 8 KV 头 × 128 维 × 2 B） | 1152 B（576 维 latent × 2 B） |
| KV cache @128K 单序列 | 49.4 GB | 11.8 GB |
| 能力上限 | 128K 上下文 | 200K（5.2 起 1M） |

（这两行由 `code/01_param_budget.py` 与 `code/02_attention_budget.py` 逐项算出，见「编译与运行」。）

第二列是路线问题的入口：**如果把 GLM-5 的注意力换回 GQA-8，1M 上下文的 KV 就是 335 GB／序列**（MLA 下是 94.2 GB，差 **3.56 倍**）——单个序列的中间结果比 355B 模型的 FP8 权重还大。而 agentic coding 的实际输入正是「整个仓库 + 几十轮工具调用」。

所以 GLM 的演进不是「把模型做大」，而是三件事同时发生：

1. **FFN 稀疏化**：把参数量堆到 744B，但每 token 只算 40B（MoE）。
2. **注意力稀疏化**：先把 KV 压进低秩 latent（MLA），再只对 top-2048 个 token 做注意力（DSA），最后让 4 层共用一个 indexer（IndexShare）去撑 1M。
3. **能力来源迁移**：GLM-5.1 之后的迭代越来越多落在后训练上，直到 GLM-5.3 **完全不换基座**。

第 3 条最反直觉，也最容易被误读，§4.3 专门处理。

## 1. 原理

### 第一步：把「总参」与「激活参」拆开，并核到官方数字上

MoE 的账要分两个量，混用会得出完全错误的规划（`80-model-architectures/moe/01-gating-and-routing` 讲路由，这里只算账）。

设 hidden 维 $d$，每层 routed expert 数 $E$、每 token 选中 $k$ 个、shared expert 数 $s$、expert 中间维 $m$。一个 expert 的参数是 $3dm$（gate / up / down 三个矩阵）。则**每层 MoE 的参数量**：

$$P_{\text{layer}} = (E + s)\cdot 3dm$$

而**每 token 真正参与的参数量**（只算被选中的）：

$$P^{\text{act}}_{\text{layer}} = (k + s)\cdot 3dm$$

代入 GLM-5 的 Table 10 值（$d=6144$，$m=2048$，$E=256$，$k=8$，$s=1$）：

- 每 expert：$3\times6144\times2048 = 37\,748\,736 \approx 37.7\text{M}$
- 每 MoE 层总参：$257 \times 37.7\text{M} \approx 9.70\text{B}$
- 每 MoE 层激活：$9 \times 37.7\text{M} \approx 339.7\text{M}$

再把注意力、dense 层、MTP 层加进来，就是 `code/01_param_budget.py` 的 A 类断言。它算出的**激活量是 40.24B，官方印值是 40B（差 +0.59%）**；**总参是 752.2B，官方印值 744B（差 +1.11%）**。

两个数的误差不一样大，这本身就是信息。而且更有信息量的是**对照组**：同一套算式用在 GLM-4.5 上，总参与激活量的偏差只有 **+0.03% / +0.02%**。也就是说这套公式是可信的，GLM-5 那 1.11% 的差**指向 Table 10 没写清的一件事**——MTP 层里装了什么：

| Table 10 的读法 | 推导总参 |
|---|---|
| MTP 层当**稠密层**（一个 FFN + 一层注意力 + indexer） | 742.8B |
| MTP 层当**整层 MoE**（256 个 expert 全带上） | 752.2B |
| **官方印值** | **744B** |

官方数字**正好落在两者之间**，而 Table 10 的 MTP 行只有「1」这一个信息。所以这不是「谁算错了」，而是**表没覆盖到**：`code/01_param_budget.py` 的 C1 断言把这个区间钉成可复核的事实。

> ⚠ 官方对参数量口径的原文：GLM-4.5 / GLM-5 的 Table 1 与 Table 10 都写明「include the parameters of MTP layers but **not word embeddings and the output layer**」。154 880 × 6144 ≈ 0.95B，embedding 与输出层若都算进去约 +1.9B。所以**任何第三方给出的 744B / 753B 之争，先问它是否含 embedding**——但对 GLM-5 来说，**MTP 层的口径比 embedding 更大**（9.5B vs 1.9B）。

### 第二步：为什么是「更深更窄」而不是「更矮更胖」

GLM-4.5 报告 §2.1 的原话是这条线的设计哲学：

> we reduce the width (hidden dimension and number of routed experts) of the model and increase its height (number of layers), as we found that **deeper models exhibited better reasoning capacity**.

把它变成一个可以对着看的账：同样约 32B 激活，

| | 层数 | hidden | experts（每 token 激活） | 注意力 |
|---|---|---|---|---|
| GLM-4.5 | 3 + 89 + 1 = 93 | 5120 | 160（8） | GQA 96 Q / 8 KV |
| DeepSeek-V3 | 3 + 58 + 1 = 62 | 7168 | 256（8） | MLA 128 头 |

GLM 用**多 30 层**换**窄 2048 的 hidden**。代价与收益都在同一处：注意力参数 $\propto d \cdot H \cdot d_h$，MoE 参数 $\propto d \cdot m \cdot E$——hidden 变窄让每层的绝对开销下降，于是同样的参数预算能买到更多层。深度的代价是**串行延迟**与**流水并行切分点变多**（§2）。

到 GLM-5，这条线被反向拉了一次：expert 从 160 扩到 **256**，MoE 层从 89 **压回 75**。报告给的理由是通信而不是表达力：

> GLM-5 scales to 256 experts and **reduces its layer count to 80 to minimize expert parallelism communication overhead**.

## 2. 执行映射

§1 只算参数量，真实系统上真正卡住的是**每次前向要在设备间搬多少字节**。MoE 加 MLA 加 DSA 之后，账变成三笔。

**第一笔：EP 的 all-to-all。** 一个 token 被路由到 $k=8$ 个 expert；若这 8 个 expert 分布在 $r$ 个不同的 EP rank 上，dispatch 要把该 token 的激活（$d \times 2$ B = 12 288 B）发出 $r$ 份，combine 再把结果收回来：

$$B_{\text{EP/token/layer}} \approx 2 \times r \times d \times 2\ \text{B} \qquad (r \le k)$$

把 $r$ 取最坏值 $k=8$：**每 token 每 MoE 层 196 608 B（192 KiB）**；75 个 MoE 层 → **约 14.7 MB/token**。这是 EP 通信量**随 MoE 层数线性增长**的直接后果——这就是 GLM-5 把 MoE 层从 89 降到 75、同时把 expert 池扩到 256 的原因：**expert 池大小只进显存账，$k$ 与层数进通信账**。真实系统里 `hierarchical all-to-all` 把节点内与节点间两级拆开，让 QKV 张量的通信与计算重叠（报告 §2.4.2）。

**第二笔：KV cache。** MLA 让每 token 每层只存 576 个值（$c_{KV}$ 的 512 维 + RoPE 的 64 维），而不是 GQA-8 的 2048 个值。1M 上下文（= 1 048 576 token，config 口径）、78 层、BF16：

$$78 \times 576 \times 2\ \text{B} \times 1\,048\,576 \approx 94.2\ \text{GB／序列}$$

即使压缩了 3.56×，**单条 1M 序列的 KV 仍然有 94.2 GB**。这就是为什么服务侧必须叠加 prefix cache（`RadixCache`）、KV 扩到主机内存、以及 INT8/FP8 cache 量化——这些不是优化，是**容量前提**。

**第三笔：后训练的 rollout。** GLM-5 的 agentic RL 用 `Multi-Task Rollout Orchestrator` 把推理引擎与训练引擎解耦，rollout 侧是 SGLang、训练侧是 Megatron，中间用 `Token-in-Token-out (TITO)` 网关保证动作级对应、用 `DP-aware routing` 提高长上下文 MoE 的 KV 复用。GLM-5.3 在这套框架（`slime`）上把训练-推理的 logprob 差压到 **1e-7 量级**，长程编码 RL 的端到端吞吐提升 **>2.3×**。速度在这里不是效率问题——**它是「能不能把 RL 规模继续放大」的问题**。

## 3. 实现与演进

| 阶段 | 模型 | 改的是什么 | 为什么必须走到下一步 |
|---|---|---|---|
| 稠密 | GLM-130B（2022-08） | 双语稠密 130B，开源权重 | 稠密模型的算力随参数线性增长，推理成本压不下来 |
| 过渡 | GLM-4 系列 → 4.5 | **首个 MoE**：355B 总 / 32B 激活；GQA + QK-Norm + partial RoPE | 激活量降下来了，但上下文还是 128K，GQA 的 KV 仍是 $O(L)$ 且带 8 个头 |
| 稀疏注意力 | GLM-5（2026-02） | 扩到 **256 expert / 744B 总 / 40B 激活**，层数压回；**MLA-256 + DSA**（indexer top-k=2048）；28.5T token | DSA 的 indexer 本身还是 $O(L^2)$，1M 上下文时 indexer 成为新瓶颈 |
| 跨层复用 | GLM-5.2（2026-06） | **IndexShare**：每 4 层共享一个 indexer 的 top-k；MTP 也复用；1M 上下文；MIT | 基座已接近当时工程上限，继续换骨架的边际收益下降 |
| 后训练放大 | GLM-5.3（2026-08） | **不换基座**。环境规模化 + 异步 RL + 难度扩展 | —— |
| 换骨架 | GLM-5.3-Flash（2026-08） | **linear + sparse 混合**、320B/18B、45 层、IndexPool、mHC、原生多模态 | 用另一条路线解同一个成本问题 |

> ⚠ 两个口径注记：**GLM-130B 那一行是【D】级**（社区综述与百科口径，本仓库未定位到一手发布材料），只作谱系起点；GLM-4.5 起的每一行都有【B】/【C】级来源（见「参考」）。另外 **GLM-5.3 的日期两个官方站不一致**：docs.z.ai 记 **2026-08-18**（英文站），bigmodel.cn 的中文发布记录记 **2026-8-19**。本仓库采用 2026-08-18。

三处关键跳变的机理：

**4.5 → 5：续训而不是重训。** DSA 不是从头训的，而是在 mid-training 结束的稠密基座上做**两阶段续训**：warm-up 1 000 步（每步 14 条 × 202 752 token，最大 lr 5e-3），再做 20B token 的稀疏适配。对照 DeepSeek-V3.2 的同类续训用了 **943.7B** token——GLM 只用了它的约 1/47。这意味着**稀疏注意力是一种可以低成本后装的改造**，不是必须从零做的架构选择。

**5 → 5.2：把 indexer 当成可复用的资源。** DSA 的稀疏核心是 $O(Lk)$（$k=2048$ 是常数），但 indexer 打分要拿每个 query 比过全部 $L$ 个 key，仍是 $O(L^2)$。5.2 观察到的经验事实是相邻层的选择高度重叠，于是每 4 层只跑一次 indexer、共享 top-k 的**位置**（**不共享** Q/K/V 计算）。

**5.2 → 5.3：把变量从架构换成后训练。** 官方原文：

> It uses the **same base model as GLM-5.2** — every gain comes from post-training.

具体做法是把难度搬到**环境**上：合成可执行、可验证、贴近真实专家工作的长程环境，用 judge agent 验证任务确实可解，再合成 verifier（不给参考解），并用 solver 轨迹去发现和堵住 reward shortcut。数字上：Terminal-Bench 3.0 从 4.6 → 28.3，DeepSWE v1.1 从 46.2 → 66.9。

## 4. 踩坑与失败模式

### 4.1 把 active params 当模型大小（或反过来）

**现象**：按 40B 估算 GLM-5 的权重显存，算出「一张 80GB 卡能放下」，实际要 8 张以上。
**成因**：40B 是**每 token 的算力口径**，744B 才是**权重显存口径**。MoE 的 744B 参数全都要驻留（或按 EP 切分后分驻在多个 rank 上），只是每次只用其中 40B。
**规避**：凡见到 MoE 的参数，先问是 total 还是 activated；显存规划用 total，延迟下限用 activated。

### 4.2 层数有三个都「正确」的口径

**现象**：GLM-5 的层数在不同来源里是 78、79、80 三个数，引用时互相矛盾。
**成因**：Table 10 给的是可加的三项——**3 dense + 75 MoE = 78 个 transformer 层**（config 的 `num_hidden_layers` 口径），**再 +1 MTP 层 = 79**；而报告 §2.1 正文写的是「**80**」，与表对不上（很可能是取整或把 MTP 另算）。
**规避**：写层数必须带口径后缀。本仓库以 Table 10 为准（78 / 79），把正文的 80 作为**原话记录**而不是事实采用。

### 4.3 把「同基座」读成「同模型」，或者读成「没进步」

**现象 A**：看到「GLM-5.3 用 GLM-5.2 的基座」，得出「5.3 只是换个版本号」。
**现象 B**：反过来，以为 5.3 换了架构，去翻 config 找新模块，找不到。
**成因**：这条线上「基座」指**权重与骨架**，「版本」指**对齐后的模型**。同一组权重经过不同后训练，可以是两个能力差距很大的模型——GLM-5.3 在自研 Code Bench 上较 5.2 提升 **50%**，CyberGym 从 77.2 → 84.5，ExploitBench 从 24.4 → 54.4。
**规避**：讨论「架构」时看基座，讨论「能力」时看后训练版本号；两者不要互相解释。

### 4.4 用榜单分数支撑架构论断

**现象**：写「DSA 让 GLM-5 在 Terminal-Bench 上超过 X」，把榜单当作架构效果的证据。
**成因**：榜单数字是在**特定 harness、特定 effort、特定预算**下测的。GLM-5.3 的 `Terminal-Bench-3` 分数附带的条件是 Claude Code 2.1.207、reasoning effort=max、400K 上下文、avg@3、10 小时超时；`ExploitGym` 的成绩还要按**每模型 TPS**（GLM-5.3 按 115 TPS）归一。换一个 harness，序关系会变。
**规避**：榜单数字一律标为**引用值**并附测量条件；架构论断只用可复算的量（参数量、FLOPs、KV 字节）。

## 编译与运行

```sh
cd code
make run          # 默认 python3 + numpy，跑 01 / 02 / 03 三个编号程序
make 01_param_budget   # 只跑本文件对应的程序
make clean
```

本文引用的参数量核算来自 **`code/01_param_budget.py`**：它按 §1 的公式逐层累加，把**推导值**与 specs.py 里的**规格值**并列打印，并以非零码退出报告超差。它只做算术与 numpy 推导，**产出是推导值，不是实测值**。

## 5. 判据表

本域无 GPU、且知识对象以方法与规格为主，§5 采用 CONTRIBUTING §4 允许的**判据表**形态（理由写在 `160-model/README.md`）。每行都是可机械判定的条件 → 结论。

| 可机械判定的条件 | 结论 |
|---|---|
| 一个 MoE 模型给出「总参 / 激活参」两个数，且激活参 < 总参的 10% | 该模型是稀疏 MoE；显存按总参算，算力下限按激活参算 |
| 某层注意力同时给出 `qk_lora_dim` 与 `kv_lora_dim`，且无 `kv_heads` | 该层是 MLA（K/V 共享压缩 latent），KV 字节按 `kv_lora_dim + rope_dim` 算，**不要乘 2** |
| 注意力层给出 `n_kv_heads` 与 `head_dim` | 是 GQA/MQA；KV 字节按 `2 × n_kv_heads × head_dim` 算 |
| 同一份规格里出现 `indexer_attn_heads` | 该模型用稀疏注意力（DSA 系）；必须另算 indexer 的 $O(L^2)$ 成本，不能只算稀疏核心 |
| 发布的模型版本号只加小数点后一位、且官方明确写「same base model as <前代>」 | 架构未变；所有差异都在后训练，检索新模块会一无所获 |
| 两个来源给出的总参差 1% 以内、但口径描述不同 | 差异来自 embedding / 输出层 / MTP 是否计入；**先对齐口径再比较**，不要断言某一方错 |
| 官方给出「每 token FLOPs 降 N×」但未公布哪些 FLOPs 计入 | 记为引用值，不得当作推导结论；本仓库只推导结构性的比例（如 indexer 侧降 4×） |

## 6. 小结与延伸

三根主线，一句话各一条：

- **FFN 稀疏化**：GLM-4.5 用「更深更窄 + 160 expert」吃到首个 MoE，GLM-5 用「256 expert + 75 层」把 EP 通信账压下来。
- **注意力稀疏化**：GQA → MLA（KV 从 2048 维降到 576 维）→ DSA（只算 top-2048）→ IndexShare（4 层共享 indexer）→ linear+sparse（Flash）。每一步都在解上一步留下的那个 $O(L^2)$ 或 $O(L)$ 项。
- **能力来源迁移**：GLM-5.3 把「换基座」这个变量彻底去掉，只留后训练与环境规模化。

下一步：`160-model/GLM/02-glm5-architecture` 把 GLM-5 代的骨架逐层拆开（MoE 路由、MLA-256、DSA、IndexShare、MTP）；`160-model/GLM/03-serving-efficiency-and-deployment` 把 KV / EP / INT4 / 国产芯片的账算到部署粒度。

## 参考

- 【C】GLM-5 Team, *GLM-5: from Vibe Coding to Agentic Engineering*, arXiv:[2602.15763v2](https://arxiv.org/abs/2602.15763)（**预印本，未确认同行评审**）。§2.1、§2.1.1、§2.4、§5、附录 A / Table 10。
- 【C】GLM-4.5 Team, *GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models*, arXiv:[2508.06471v1](https://arxiv.org/abs/2508.06471)。§2.1 与 Table 1。
- 【B】z.ai 发布页：[GLM-5.2](https://z.ai/blog/glm-5.2)（IndexShare / MTP / 1M）、[GLM-5.3](https://z.ai/blog/glm-5.3)（同基座 / 后训练 / 评测脚注）、[GLM-5.3-Flash](https://z.ai/blog/glm-5.3-flash)（混合架构 / 服务栈）。
- 【B】官方模型页：[docs.z.ai · New Released](https://docs.z.ai/release-notes/new-released)、[docs.bigmodel.cn · 模型与产品发布记录](https://docs.bigmodel.cn/cn/update/new-releases)、[GLM-5.3](https://docs.bigmodel.cn/cn/guide/models/text/glm-5.3)。
- 【D】Sebastian Raschka, *GLM-5.2 and IndexShare for Long-Context Sparse Attention*, 2026-06-18（转述 `config.json` 字段与 IndexCache 重叠率；**仅作旁证**）。
- 相关域：[80-model-architectures/moe](../../80-model-architectures/moe/)、[80-model-architectures/attention-variants](../../80-model-architectures/attention-variants/)、[120-inference/kv-cache](../../120-inference/kv-cache/)。

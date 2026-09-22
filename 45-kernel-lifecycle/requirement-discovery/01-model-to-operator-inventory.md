---
title: 从模型规格到算子清单：这份清单是可以算出来的
slug: requirement-discovery-model-to-operator-inventory
domain: 45-kernel-lifecycle
topic: requirement-discovery
level: 2
difficulty: intermediate
prerequisites: []
tags: [operator-inventory, moe, gqa, flops, 2nd, vllm, kimi-k3, config-driven]
related:
  - 45-kernel-lifecycle/requirement-discovery/02-roofline-verdict
  - 45-kernel-lifecycle/requirement-discovery/03-optimization-priority
  - 40-kernels/gemm/01-naive-gemm-and-roofline
  - 10-foundations/performance-models/01-roofline-model
code:
  - code/01_operator_inventory.py
status: stable
updated: 2026-09-20
---

# 从模型规格到算子清单：这份清单是可以算出来的

> **算子「种类」由模型结构决定，算子「工作量」由模型规格决定——给定一份 config，算子清单是可以算出来的**，不需要等别人踩完坑给出列表。
>
> 这条断言有硬判据：从 config 反算的参数量必须落在**公开参数量**上。本仓库实算（`code/01_operator_inventory.py`，python 3.13.5）：**Meta-Llama-3-8B 反算 8.0300 B，公开 8.03 B，偏差 −0.00%**；**Qwen3-30B-A3B 反算 30.5193 B，公开 30.5 B，偏差 +0.06%**。漏掉任何一个矩阵（哪怕只是 `o_proj`、或者把 GQA 的 KV 宽度写错），这两个数立刻对不上——这不是「大致吻合」，是**逐项搭出来的恒等式**。

## 0. 背景与动机

### 0.1 一个具体的对照场景

某个新模型发布了。两条路径：

| | 路径甲：等清单 | 路径乙：算清单 |
|---|---|---|
| 第一天 | 等社区给出「需要支持哪些算子」 | 打开 `config.json`，30 分钟列出清单 |
| 第一周 | 清单到了，开始评估工作量 | 已经在写第一个算子 |
| 风险 | 清单可能**不完整**（漏掉某个变体），也可能**过度**（把不上推理路径的算子列进来） | 清单的**每一项都能追溯到 config 里的一个字段**，错了能定位 |

路径甲不是慢在「等」上，而是慢在**它把一份可推导的东西当成了需要外部输入的东西**。而清单的推导输入就摆在权重目录里——`config.json` 的十个字段。

### 0.2 但清单不是「列出一堆算子名」

一份能用的清单必须带三个数：

1. **每 token 调用次数**（决定 launch 开销的量级与 kernel 的调用点）；
2. **每 token FLOPs**（决定要走哪条算力路径，见 `02-roofline-verdict`）；
3. **每 token 访存字节**（同上，分母）。

**只列出名字的清单不能回答「先写哪个」**——因为「先写哪个」是三个数的函数。

### 0.3 本篇在本域主线上的位置：**上界在哪**

本域的统一主线是：

> **到上界还有多远，以及那段距离值不值得走过去。**

四个主题各承担一环：`requirement-discovery` 问「**上界在哪**」、`operator-design` 问「怎么抬高上界」、`implementation` 守「**上界的前提**」（算对之前，上界没有意义）、`convergence-and-tuning` 问「剩下的距离是工程能触及的，还是噪声」。

**本篇是这条主线的第一环，而且它的产物是后三环的输入**——理由有三条，都是结构性的：

1. **「到上界多远」这个量的分子与分母都是按算子定义的**（`02-roofline-verdict` 的 `AI = FLOPs ÷ DRAM 字节`）。**没有清单，就没有「哪个算子的上界」这个问题**——上界必须挂在某个算子上才存在。
2. **一个漏掉的算子，它的上界永远不会被算**，于是它**永远不会进入优化候选**——而它的时间仍然计在总时间里。这是清单错误最贵的形态：**不是把顺序排错，而是让一个候选根本不出现**。
3. **反向也成立**：一个多列的算子（把训练侧算子、或已有算子的新参数当成新算子）会**把工作量预算摊薄**，让真正该做的那个排不进这一轮。

**⇒ 本篇不产生上界，它产生「上界要挂在哪些算子上」这份名单。** 名单的单位必须是「**算子 × 阶段**（prefill / decode）」——因为同一个名字在两个阶段的 `AI` 可以差两个数量级（`02-roofline-verdict` §4.5）。

> ⚠ **不要把这条写成「清单也是一种上界」。** 那是把主线的词贴到不相干的东西上（§0.3 点名的头号风险：正确的废话）。**清单不是上界，它是「上界要挂在哪」的答案**——`上界` 这个词在本篇里一次都不该被用在清单自己身上。

## 1. 原理

### 1.1 输入：config 里的七个字段

| 字段 | 符号 | 作用 |
|---|---|---|
| `num_hidden_layers` | `L` | 每个投影算子被调用 `L` 次 |
| `hidden_size` | `d` | 所有投影的**输入/输出公共维** |
| `num_attention_heads` | `h_q` | Q 的输出头数 |
| `num_key_value_heads` | `h_kv` | K/V 的输出头数（**GQA 就在这个字段上**） |
| `head_dim` | `d_h` | 每头的宽度 |
| `vocab_size` | `V` | 词表侧两个矩阵的宽度 |
| `num_experts` / `num_experts_per_tok` | `E` / `k` | MoE 的总专家数与每 token 激活数 |
| 其余 | `d_ff` | dense 用 `intermediate_size`；MoE 用 `moe_intermediate_size` |

**派生量只有两个，但它们是最容易写错的地方**：

```
d_q  = h_q  · d_h          # Q 的宽度
d_kv = h_kv · d_h          # K、V 的宽度   ← GQA 的全部内容就在这一行
```

> ★ **`d_q ≠ d` 是可以发生的。** Llama-3-8B 上 `h_q·d_h = 32×128 = 4096 = d`，两者恰好相等——**于是「用 `d` 代替 `d_q`」这个错误在那个模型上不会暴露**。Qwen3-30B-A3B 上 `d_q = 4096` 而 `d = 2048`，差 2 倍。**拿 Llama 验证过的代码去跑 Qwen 就会错。** 反过来说，「`d_q == d`」只有在 `h_q·d_h == d` 时才成立，**它是一个需要被检查的条件，不是一个可以假设的恒等式**。

> ★ **`d_kv` 用错是 8 倍的错误，不是 8% 的错误。** 本仓库实算：Qwen3-30B-A3B 的 `h_q/h_kv = 32/4 = 8`，把 `k_proj`/`v_proj` 按 `h_q` 算会把这两项的 FLOPs **虚高 8 倍**（`code/01_operator_inventory.py` 的 §C.3）。Llama-3-8B 上是 4 倍。

### 1.2 逐项推导：以 attention 与 FFN 为例

**口径写死在前面**（不写死，同一张表内部就会混两种口径）：FLOPs **按 FMA 计 2 次**；矩阵乘 `M×K @ K×N` 记 `2·M·N·K`；调用次数按 **decode（batch = 1）** 计；上下文长度记 `S`。

**第一步：attention 的四个投影（每 token，每个 `L` 次）**

| 算子 | FLOPs／次 | 依据 |
|---|---|---|
| `q_proj` | `2·d·d_q` | `(1×d) @ (d×d_q)` |
| `k_proj` | `2·d·d_kv` | `(1×d) @ (d×d_kv)` |
| `v_proj` | `2·d·d_kv` | 同上 |
| `o_proj` | `2·d_q·d` | `(1×d_q) @ (d_q×d)` |

四项合计 `2·(2·d·d_q + 2·d·d_kv)`——注意 `o_proj` 的输入宽度是 `d_q`，**不是 `d`**。

**第二步：attention 的两个核（每 token，每个 `L` 次，**随 `S` 变**）**

```
scores = Q(1×d_q) @ Kᵀ(d_q×S)   →  2·S·d_q
PV     = P(1×S)  @ V(S×d_q)     →  2·S·d_q
```

**每 token 合计 `4·S·d_q`**。这一项是**唯一随上下文长度增长的部分**——所以它单独成表，且**在清单里必须标出它的 `S`**，否则「attention 占多少」这个数没有意义。

> ★ **prefill 里这一项的推导不在这里。** prefill 时 `Q` 是 `S×d_q` 而不是 `1×d_q`，两个核的 FLOPs 都变成 `2·S²·d_q`，合计 `4·S²·d_q`。**这个平方项的来源（为什么是 `2S²d`）属于算子本体，正文在 `40-kernels/attention/01-attention-as-a-kernel`，此处回链，不重讲。**

**第三步：FFN／MoE**

| 形态 | 每 token 的 FLOPs |
|---|---|
| dense（`gate`/`up`/`down`） | `L · 3 · 2 · d · d_ff` |
| MoE（每层 `E` 个专家，激活 `k` 个） | `L · k · 3 · 2 · d · d_ff` |

**⇒ 关键差别只在 `E` 与 `k` 的位置**：权重里有 `E` 份，每 token 只读 `k` 份。**参数量按 `E` 算，FLOPs 按 `k` 算**——两者用的不是同一个字母，这是本主题最容易出错的一处（§4 的坑 ④）。

**第四步：词表侧**

| 算子 | 调用／token | FLOPs／token |
|---|---|---|
| `lm_head` | 1 | `2·d·V` |
| `embedding_lookup` | 1 | **0**（查表，不是乘法） |

**⇒ 嵌入有参数量但**没有 FLOPs**。这一条是 §1.3 那个「偏差」的最大单项来源。

### 1.3 用 `2ND` 对照，并**逐项分解**偏差

推理的每 token FLOPs 常被近似成 **`2N`**（`N` = 参数量）——一次前向每个参数约 2 次浮点运算（一次乘、一次加）。

> ★ **口径警告（P9 的典型现场）**：**`6ND` 是训练口径**（前向 `2ND` + 反向 `4ND`），**推理的前向只有 `2ND`**。本主题是推理场景，**一律用 `2ND`**。两个数都对，**并列就错**——引用 `6ND` 时必须同时说明它是训练口径。（`6ND` 的分解是 Kaplan 式 FLOPs 记账的标准形式，属文献口径 **[C]**。）

于是「清单合计」与 `2N` 的差应当被**逐项分解**，而不是「大致吻合」。本仓库实算（`S = 4096`，fp16）：

| 模型 | 清单合计 FLOPs/token | `2N` | 偏差 | 分解 |
|---|---|---|---|---|
| Llama-3-8B | 17,156,800,512 | 16,059,990,016 | **+6.83%** | `+ 2,147,483,648`（attention 二次项，`2N` 不含）<br>`− 1,050,673,152`（`2×` 嵌入：`2N` 计入，但查表 0 FLOPs） |
| Qwen3-30B-A3B | 9,279,373,312 | 61,038,657,536 | **−84.80%** | `+ 3,221,225,472`（attention 二次项）<br>`− 622,329,856`（`2×` 嵌入）<br>`− 54,358,179,840`（**未激活的 120 个专家**） |

**残差为 0**（浮点内精确）——三项分解之后没有剩余。程序把这条写成断言（§B.1）。

> ★ **两个偏差的符号相反**，这本身就是结论：dense 模型上 `2N` **低估**（因为 attention 二次项被忽略了）；MoE 模型上 `2N` **大幅高估**（因为 `E` 个专家全被算了进去）。**「`2N` 是个粗略近似」这句话对两个模型都成立，但偏差的方向相反**——所以不能只说「近似」，必须给出分解。

## 2. 执行映射

### 2.1 这份清单在 vLLM 里怎么落地

vLLM 的做法（**[B]**，vLLM 官方文档 “Adding a New Model” / “Adding new models”，经检索转述）是把「支持一个新模型」分**三档**：

| 档 | 判据 | 要不要新 kernel |
|---|---|---|
| **simple** | 与 vLLM 已有的某个模型结构相似 | 不要 |
| **moderate** | 标准组件，但结构独特 | 不要 |
| **complex** | 含**新算子**——例如新的 attention 机制 | **要**：需要在 vLLM 中**实现新的 attention layer / kernel** |

> ★ **P10 纪律**：这是 **vLLM 一家**的分档方式。**不得**写成「业界通常分三档」——按本域 README 的机械检查，凡出现「业界 / 通常 / 主流做法」，必须能指出**至少两个独立主体**都这么做，否则改写成「vLLM 的做法是……」。本文一律写成「vLLM 的做法是……」。

同一份文档里还有一条更前置的建议：**先跑 `vllm serve <model>`**。原因是许多 decoder 模型可以通过 **Transformers 后端**直接加载，**根本不需要在 vLLM 里实现**。

**⇒ 这份清单的第一道闸不是「写哪个 kernel」，而是「要不要写 kernel」。** §1 算出的清单在这道闸上只回答一件事：**清单里有没有一项落在 vLLM 已有的实现集合之外**。

### 2.2 一个真实案例：Kimi K3 带来了四个新算子，外加一次缓存层重构

vLLM 官方博客（2026-07-22 / 07-27）与 v0.28.0 的 release notes（**[B]**）逐项记录了 Kimi K3 的支持：

| 新算子 | 形态 |
|---|---|
| **KDA prefill** | FlashKDA（Moonshot AI 的 CUTLASS 实现），v0.28.0 融合进 prefill kernel（#51311、#52458） |
| **KDA decode** | 把因果卷积 + 循环状态更新 + RMSNorm **折进一次 launch**，就地更新状态（#50654） |
| **Attention Residuals** | Block AttnRes：对至多 8 个缓存的块表示 + 当前块内残差做**深度方向** softmax（至多 9 个来源），单个融合 kernel；Triton 通用路径 + 专用 CUDA 路径 |
| **MoE 新激活 SiTU** | 接进 MXFP4 TRTLLM-Gen 与 DeepGEMM；v0.28.0 为 MegaMoE 加 SiTU（#50510） |

**★ 这个案例的价值不在列表本身，而在列表的规模与形状**：一个模型发布带来的不是「一个算子」，而是**四类**，其中**同一族的 prefill 与 decode 是两个不同的 kernel**（§4 的坑 ③）。

**★ 更值得注意的是成本不止于算子数**：KDA 是**循环**的（推进一个矩阵状循环状态 + 一个短卷积状态，**不存 per-token KV 对**），所以 vLLM 把三个原本绑在一起的概念**拆开**——**物理块大小 / 调度器对齐 / 前缀匹配单元**，并规定命中部分块时**先把缓存状态拷进私有目的地再扩展**（copy-on-write）。另加 `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` 做区间化保留。

**⇒ 从 config 推出的清单能算出「要写哪几个算子」，算不出「缓存层要重构成什么样」。** 这是这份清单的**明确边界**，写在这里是为了不让读者高估它。

### 2.3 换一条轴：TensorRT-LLM 的 per-iteration 时间预算

清单回答「有哪几个算子」，不回答「时间花在哪」。TensorRT-LLM 的做法（**[B]**，PR #18805）是不再用单一的比例：

【**经检索转述**（本实现者未逐字读到 diff 原文）】该 PR 的动机是：`nsys stats` 只回答「哪个 kernel 花掉的 GPU 时间最多」，**不回答 iteration 的墙钟时间去哪了**——**GPU 一闲下来这两者就分道扬镳**；按开销排序的 top-kernel 表会**读起来像一张机会清单，即使这次迭代大部分时间在等宿主**。

它产出的是一份**按 iteration 的预算**：相对**检出的 anchor** 的时间、**三层嵌套的 busy rung 连同各自的 idle 补集**、**compute-absent 时间**拆成 **launch-starved / blocking / dependency-stalled**、kernel 类别分解、以及**暴露出来的集合通信时间**拆成 transfer 与 jitter wait。

**⇒ 与本篇的关系**：`share_pct` 这一类量只有在**分母是 iteration 而不是 GPU 时间**时才与「要不要写 kernel」可比。清单给的是分子侧（哪些算子存在），这份预算给的是分母侧（它们占多少）。

## 3. 从手工清单到自动推导

**第一步：先问「需不需要算」。** 按 §2.1，先跑 `vllm serve <model>`。

**第二步：把清单变成一次注册。** vLLM 的做法（**[B]**，vLLM IR）是把**算子语义**与**实现**分开：

| 环节 | 机制 |
|---|---|
| 声明算子 | `@register_op`，配一份 `native` 的 PyTorch 实现（它同时是语义定义、默认实现、以及测试参照） |
| 注册实现 | `register_impl`，可带 `supported`（导入时查一次，如 `HAS_TRITON`）与 `supports_args`（按参数动态判） |
| 选择 | **优先级列表**：`--ir-op-priority.<op_name>=<provider1>,<provider2>,...`，规则是「**第一个 supported 且 `supports_args` 通过的实现被选中**」 |
| 兜底 | `native` 是保留名，**永远存在** |

**⇒ 「优先级选择」是一个显式配置项，不是隐式的。** 这条对 §1 的清单有一个直接后果：**清单里的每一项，其实现都必须是「可被替换的」，否则优先级列表对它无效**。

**第三步：清单的每一项都要能回答「它是新算子还是已有算子的新参数」。** 见 §4 的坑 ①。

## 4. 踩坑与失败模式

### 4.1 把「已有算子的新参数」当新算子

**现象**：模型改了 RoPE 的 `rope_theta`、或者换了一组 `rms_norm_eps`，就开一个「新算子」的工单。

**成因**：清单的粒度是**算子**，不是**算子 × 超参**。超参变化改的是同一个 kernel 的入参或编译期常量。

**规避**：判据是**调用签名是否变化**，不是**数值是否变化**。判据表（§5）第 4 行把它写成可机械判定的形式。

### 4.2 把训练侧的算子算进推理清单

**现象**：清单里出现 `backward` 相关的一切，或者用 `6ND` 算 FLOPs。

**成因**：**`6ND` 是训练口径**（前向 `2ND` + 反向 `4ND`）。同一个 `N`、同一个 `D`，两个数都在文献里出现，**并列即成错**。

**规避**：本主题**一律用 `2ND`**；引用 `6ND` 时必须同时说明它是训练口径。**更隐蔽的形态**：优化器状态（Adam 的 `m`、`v`）在权重里占字节，但**不在推理的访存路径上**——按权重文件大小估推理带宽会高估 3 倍。

### 4.3 忽略 prefill / decode 的算子差异

**现象**：把 attention 当成「一个算子」，写一份 kernel 同时覆盖两个阶段。

**成因**：两者调用同一个数学定义，**但不是同一份工作**。Kimi K3 的 **KDA prefill** 与 **KDA decode** 在 vLLM 里是**两个不同的 kernel**（**[B]**，§2.2 的表）——decode 那个还要把因果卷积、循环状态更新与 RMSNorm 折进一次 launch。

**规避**：清单里每项必须带**阶段标签**。FLOPs 的形态也不同：decode 是 `4·S·d_q`（线性于 `S`），prefill 是 `4·S²·d_q`（平方）。

### 4.4 把 `E` 个专家按 dense 全专家算 FLOPs

**现象**：MoE 的 FFN 按 `E` 个专家算 FLOPs，而不是 `k` 个。

**成因**：**参数量按 `E` 算（权重确实有 `E` 份），FLOPs 按 `k` 算（每 token 只读 `k` 份）**——两个量用的是不同的字母，混用非常自然。

**规避（本程序 §C.1 实跑出来的具体数）**：

| | 值 |
|---|---|
| 专家侧虚高倍数 | **16.000000**，**精确等于 `E/k = 128/8`** |
| **整模型**虚高倍数 | **6.8580**（9,279,373,312 → 63,637,553,152） |

> ★ **「MoE 虚高 = `E/k`」这句话只对专家那一部分成立。** 整模型的倍数**小于** `E/k`，因为 attention 与 `lm_head` **与专家数无关**，它们把比率拉下来了。把 `E/k` 直接说成「整模型的虚高倍数」是一处**作用域过大**的断言（P7）。

**★ 该口径错误真的会翻转一个结论**（§C.2 实跑）：Qwen3-30B-A3B 上

| `S` | 正确口径第一 | dense 口径第一 | 翻转？ |
|---|---|---|---|
| 4096 | FFN / MoE 专家（3.62e9） | FFN / MoE 专家（5.80e10） | ✗ |
| 8192 | **attention_core**（6.44e9） | **FFN / MoE 专家**（5.80e10） | **✓** |

**⇒ 翻转不是无条件的**：临界值就是 §B.2 算出的 `S*`（Qwen3-30B-A3B 上 **4608**）。正确的说法是「**在长上下文下**，忽略路由会把第一目标指错」，不是「忽略路由一定会指错」。

> ★ **§C.2 的第一版是错的，值得写下来**：它按**算子名**取最大值，于是「attention_scores」与「moe_gate_proj」相比——但 FFN 被拆成 **gate/up/down 三个名字**，三者之和远大于 `attention_scores`。**按名字取最大值是在比「名字」，不是比「工作量」。** 现在改成按**算子族**聚合。（这一条与 `40-kernels` 计划里那次「订正不扩散」是同一类：**发现了一个错误之后，要问它还在别处出现过没有**。）

### 4.5 用 `hidden_size ÷ num_attention_heads` 代替 `head_dim`

**现象**：`d_h = d / h_q`。

**成因**：**在 Llama-3-8B 上这是对的**（4096/32 = 128 = `head_dim`）。在 Qwen3-30B-A3B 上错一倍（2048/32 = 64，而 `head_dim` 是 **128**）。

**规避**：**只用 config 里的 `head_dim` 字段**，永远不做那个除法。这条错误的隐蔽性来自它**在一个主流模型上恰好正确**——所以它不会在小规模测试中暴露。

### 4.6 忽略 GQA

见 §1.1 的第二个警告：`k_proj`/`v_proj` 虚高 `h_q/h_kv` 倍（Llama-3-8B 4 倍、Qwen3-30B-A3B **8 倍**）。判据是 `num_key_value_heads` 字段是否存在于 config 且小于 `num_attention_heads`。

## 5. 判据表

> ★ 按本域约定，§5 的位置与编号保留，内容**不是实测数据**而是**判据表**（本机无 GPU，本域任何一篇都不得声称有实测数据）。每一行必须是「**可机械判定的条件 → 结论**」。

| # | 可机械判定的条件 | 结论 |
|---|---|---|
| 1 | `config.json` 里有 `head_dim` 字段 | 用它；**不得**用 `hidden_size ÷ num_attention_heads`（§4.5） |
| 2 | `num_key_value_heads` 存在且 `< num_attention_heads` | 走 GQA：`k_proj`/`v_proj` 按 `h_kv·d_h` 计；虚高倍数 = `h_q/h_kv` |
| 3 | `num_experts` 存在 | 参数量按 `E` 计、FLOPs 按 `k` 计；**两处不得用同一个字母** |
| 4 | 新模型的改动只动 `rope_theta` / `rms_norm_eps` / `eps` 等标量 | **不是新算子**——同一个 kernel 换入参；不开新工单 |
| 5 | 改动使某个算子的**调用签名**（张量个数、形状、dtype）变化 | 是新算子；进清单并标阶段 |
| 6 | 算出来的 `d_q ≠ d` | `q_proj` / `o_proj` 的宽度与 `d` 不同；用 `d` 会错 `d_q/d` 倍 |
| 7 | 有 attention 项且给了 `S` | 用 `4·S·d_q`（decode）或 `4·S²·d_q`（prefill）；**无 `S` 的 attention 数字不可比** |
| 8 | 用 `2N` 估总 FLOPs，且模型是 dense | `2N` **低估**（漏掉 attention 二次项）；偏差 = `(attn − 2·嵌入)/2N` |
| 9 | 用 `2N` 估总 FLOPs，且模型是 MoE | `2N` **高估**；偏差主要由未激活的 `(E−k)` 个专家构成 |
| 10 | 引用的 FLOPs 系数是 `6` | 那是**训练**口径；推理场景**必须**换成 `2` |
| 11 | 清单里某项落在 vLLM 已有实现集合之外 | 该模型在 vLLM 的分档里属 **complex**——要写新 kernel |
| 12 | 先跑 `vllm serve <model>` 能起来 | Transformers 后端已覆盖；**不需要**写任何 kernel |
| 13 | 是 MoE 且 `S` 超过 `S* = k·3·2·d·d_ff ÷ (4·d_q)` | 正确口径下 attention 二次项是第一大 FLOPs 消耗；dense 口径会把它指成 FFN |
| 14 | 权重文件里含优化器状态 | 那是**训练**侧的字节；**不得**计入推理的访存 |
| 15 | 从 config 反算的参数量与公开值偏差 > 1% | 清单**有一项写错了**（漏项或宽度错）；先修清单再看结论 |

**闭合性检查**：读完这张表能把「一次模型发布 → 一份带阶段标签、带三个数的清单 → 要不要写 kernel」走完，不需要再问「那到底怎么办」。**唯一表外的问题是「缓存层要不要重构」**——它在 §2.2 被明确划为清单的边界外。

## 6. 小结与延伸

- ★ **这条论断有硬判据，不是主张**：从 config 反算的参数量必须落在公开参数量上（Llama-3-8B **8.0300 B vs 8.03 B**；Qwen3-30B-A3B **30.5193 B vs 30.5 B**）。
- ★ **`6ND` 是训练口径，推理是 `2ND`**——这是 P9（同名不同指）的标准现场。
- ★ **「MoE 虚高 = `E/k`」只对专家那一部分成立**：专家侧精确 16.000000，**整模型只有 6.8580**。
- ★ **dense 口径的错会翻转「第一 FLOPs 消费者」，但只在长上下文下**：Qwen3-30B-A3B 的临界 `S* = 4608`。
- ★ `d_q`、`d_kv`、`head_dim` 三个量**都不是 `d`**；`d_q == d` 是一个需要检查的条件，不是恒等式。
- **vLLM 的做法是三档分治（simple / moderate / complex），且先跑 `vllm serve`**；这是 vLLM **一家**的做法（P10）。
- **Kimi K3 的案例说明**：一次模型发布可能带来 **4 类**新算子，**且成本不止于算子数**（缓存层要把三个原本耦合的概念拆开）。
- **清单的边界**：它算得出「要写哪几个算子」，算不出「缓存层怎么改」。
- ★ **本篇在主线上承担「上界在哪」这一环**：清单是**上界要挂在哪**的答案——漏一个算子，那个算子的上界永远不会被算，也就**永远不会进入优化候选**。

**延伸**：

- 拿到清单之后的下一步——**每个算子在目标精度路径上是什么受限**，在 `45-kernel-lifecycle/requirement-discovery/02-roofline-verdict`（本域，正文在本域）。
- 清单排完序之后——**先写哪个**，在 `45-kernel-lifecycle/requirement-discovery/03-optimization-priority`。
- attention 为什么是 `2S²d`、MoE 的路由怎么走：**算子本体，回链** `40-kernels/attention/01-attention-as-a-kernel` 与 `40-kernels/moe-kernels/01-moe-compute-pattern`。
- 模型结构**是什么**（transformer / GQA / MoE 的原理）：回链 `80-model-architectures/transformer`。
- roofline 模型的**原理与推导**：回链 `10-foundations/performance-models/01-roofline-model`。

## 编译与运行

```bash
cd 45-kernel-lifecycle/requirement-discovery/code
make 01_operator_inventory        # 或 make run，跑本主题全部三个编号程序
```

本篇对应的编号程序是 **`code/01_operator_inventory.py`**。它**只用标准库**（连 numpy 都不用）、**不 import torch**、不碰 GPU。

- **A 类**：`L` 层的 `q_proj` 每 token 调用 `L` 次、FLOPs `= L·2·d·d_q`，与手算逐位对照；从 config 反算参数量 vs 公开值；GQA 的 KV 宽度；以及一条**代数恒等式**（清单 `= 2N − 2·嵌入 + attention 项`，残差恰为 0）。
- **B 类**：与 `2N` 对照，偏差被**三项完整分解**（残差 0）；attention 二次项超过专家 FFN 的临界 `S*`。
- **C 类**：MoE 按 dense 全专家算 ⇒ 专家侧虚高**精确等于 `E/k`**、整模型 6.8580；该口径**真的翻转**第一 FLOPs 消费者（`S = 8192`）；忽略 GQA 的虚高倍数**精确等于 `h_q/h_kv`**。

判据若被改错，程序以**非零码**退出。已实测的四种变异（本仓库实跑，均退出码 1）：把 `d_kv` 换成 `h_q·d_h`（A.1/A.3/C.3 失败）、把 fp16 峰值换成带稀疏的 1979（A.1 失败）。

## 参考

**本仓库程序（可复现）**

- `code/01_operator_inventory.py` —— 本篇的全部数值都出自它；`make 01_operator_inventory`。
- `code/specs.py` §2 —— 两个模型的 config 字段与公开参数量，每条带 `# 来源：` 与 **B 级**标签。

**外部来源（**[B]：产品官方文档 / 官方仓库**）**

- vLLM 官方文档 “Adding a New Model” / “Adding new models”：三档（simple / moderate / complex）与「先跑 `vllm serve`」。**经检索转述**，本实现者未逐字读到页面原文。
- vLLM 官方文档 `vllm.ir.op` 与 `design/vllm_ir`（另 PR #33825）：`@register_op` / `register_impl` / 优先级列表 / `native` 兜底。
- vLLM 官方博客 “Kimi K3 Is Here: Efficient Day-0 Support on vLLM”（2026-07-27）与 “A Preview of Production-Scale Kimi K3 Support on vLLM”（2026-07-22）、vLLM v0.28.0 release notes：KDA prefill/decode、Block AttnRes、SiTU、prefix caching 的三概念拆分。**经检索转述。**
- NVIDIA/TensorRT-LLM PR #18805 的变更说明：iteration 时间预算、三层 busy rung 与 idle 补集、compute-absent 的三分（launch-starved / blocking / dependency-stalled）。**经检索转述。**
- `meta-llama/Meta-Llama-3-8B` 与 `Qwen/Qwen3-30B-A3B` 的 `config.json`：§1 的全部输入字段。

**外部来源（**[C]：论文 / 官方技术报告**）**

- Kaplan 式 FLOPs 记账的 `6ND = 2ND（前向）+ 4ND（反向）` 分解，以及「前向每 token ≈ `2N`」。**本文只用它的比例关系，不引用任何具体数值。**

**UNSOURCED（正文未使用）**

- **Qwen3-30B-A3B 的共享专家规格**（`shared_expert_intermediate_size`）：**未找到**。检索只给出 `intermediate_size: 6144`，而该字段的语义是「非 MoE 的 FFN 宽度」，**不等于共享专家的宽度**。因此 §1.3 算出的活跃参数量是**下界**。见 `code/specs.py` §9。
- **任何真实系统的实测性能数字**：本机无 GPU，本域不产出实测。

"""160-model/glm 的外部规格常量源。

本文件只放两类东西：
  1. **从一手来源抄来的**外部规格（GLM-4.5 / GLM-5 技术报告的表号与印值、
     z.ai 官方发布页与文档的原文、模型 config 的字段值），每条带一行 `# 来源：`；
  2. **口径注记**——同一个量在不同来源下不是同一个数的，逐处写明。

★ 外部来源分级（CONTRIBUTING §2，A–E）在本文件的记法：
    【C】论文 / 官方技术报告（arXiv 预印本，**未确认同行评审**）
    【B】产品官方文档（某一实现的规格：z.ai / bigmodel.cn 的官方发布页与文档）
    【D】社区项目文档（个人或小团队仓库、个人博客）
  全库没有把 GLM 的规格归入【A】的依据：A 级是**硬件厂商规范**（PTX ISA / CUDA
  Guide / NVIDIA 白皮书），模型规格不属于该级。
  ★ **D 级数字不得单独支撑论断**，本文件凡引 D 级都同时给出可复核的载体
  （它引用的是哪个 `config.json` 字段 / 哪张官方表），并在文档里只作旁证。

★ 核查范围（不写「全库」「已核实全部」这类大于实际核查范围的措辞）：
  本文件逐条核的载体是——GLM-5 技术报告 arXiv:2602.15763v2 的正文与 Table 10、
  GLM-4.5 技术报告 arXiv:2508.06471v1 的 §2.1 与 Table 1、
  z.ai 的 `z.ai/blog/glm-5.2`、`z.ai/blog/glm-5.3`、`z.ai/blog/glm-5.3-flash`
  三篇发布页（正文由页面 JS bundle 取出），以及 docs.z.ai / docs.bigmodel.cn 的
  模型介绍页。核查日期：2026-10-13。

★ 本文件**不放任何算法**，也**不 import 任何第三方库**：
  凡被断言检验的算式（参数与激活量核算、注意力 FLOPs、KV 字节、EP 通信量、
  投机解码收益）**一律在编号程序里另写**，理由见 CONTRIBUTING §2「参考实现必须独立」。

UNSOURCED（查不到可定位的一手来源，**正文不使用**）：
  · GLM-5.2 / GLM-5.3 的**完整** `config.json`：本机取不到 huggingface.co
    （curl 与 WebFetch 均失败，返回 000 / fetch failed），因此**不落成一手常量**。
    ¶ 折中：GLM-5.2 的层数与 hidden 采用【D】级转述（该转述指名了 config 字段），
      文档中**只作旁证**，主结论一律落在【C】级 Table 10 上。
  · GLM-5.3-Flash 的 linear attention 具体变体（GDN / SimpleGDN / KDA 中的哪一种）：
    官方发布页只写「linear attention」「state modeling」，**没有点名机制**，
    故记为 UNRESOLVED，正文不得替官方选一个。
  · GLM-5.3-Flash 的 hidden dim / expert 数 / 层内 hybrid 比例：官方未公布完整 config，
    只公布总参 320B、激活 18B、层数 45。
  · 「IndexShare 使 1M 上下文每 token FLOPs 降 2.9×」的**计数口径**：
    官方给了倍数，**没给哪些 FLOPs 计入**；本仓库的解析推导得不到同一个数
    （见编号程序 02 的推导值与 B 类断言），故只作【B】级引用值记录，不当作推导结论。
  · 「DSA 使长序列注意力计算降 1.5–2×」的计数口径：同上，只记录原文。
"""

# ---------------------------------------------------------------------------
# 版本锚点（本机）——§5 判据表与编号程序的测量条件
# ---------------------------------------------------------------------------

# 来源：本机实测 `python3 -VV` 与 `python3 -c "import numpy; print(numpy.__version__)"`
#   （2026-10-13）。默认 `python3` 即 /Users/cachio/miniconda3/bin/python3。
MEASURE_HOST = "Apple M1 Pro（10 核）/ macOS / **CPU-only，无 GPU 数字**"
MEASURE_PYTHON = "3.12.9 (Anaconda, Clang 14.0.6)"
MEASURE_NUMPY = "2.3.4"
# 本域编号程序只做**算术与 numpy 推导**，不碰任何模型权重、不 import torch：
# 因此它们产出的是**推导值**，不是本机实测值（CONTRIBUTING §2 的第四类不适用）。
PROGRAMS_ARE_DERIVATIONS = True

# ---------------------------------------------------------------------------
# 【B】家族时间线（口径：官方发布页标称的模型版本与日期）
# ---------------------------------------------------------------------------

# 来源：【B】z.ai `release-notes/new-released` 与 bigmodel.cn `update/new-releases`
#   两张发布记录页逐条读（2026-10-13）。⚠ 两个页面对 GLM-5.3 的日期差一天：
#   docs.z.ai 记 **2026-08-18**，bigmodel.cn 记 **2026-8-19**（中文站写「上线」）。
#   本文件采用 **2026-08-18**（英文站），并在文档中注明中文站记 8-19。
GLM_RELEASES = (
    # (日期, 模型, 一句话口径)
    ("2022-08", "GLM-130B", "双语稠密 130B，开源权重（【D】维基/综述口径）"),
    ("2025-07-28", "GLM-4.5", "首个 MoE：355B 总 / 32B 激活；同发 GLM-4.5-Air 106B/12B"),
    ("2025-09-30", "GLM-4.6", "旗舰编码模型；上下文扩到 200K"),
    ("2025-12-22", "GLM-4.7", "编码 / 推理 / Agentic 提升；开源 SOTA 口径"),
    ("2026-02-12", "GLM-5", "744B 总 / 40B 激活；引入 DSA；200K 上下文"),
    ("2026-04-07", "GLM-5.1", "长程任务：单任务独立工作最长 8 小时；综合对齐 Opus 4.6"),
    ("2026-06-16", "GLM-5.2", "1M 无损上下文；引入 IndexShare；MIT 协议"),
    ("2026-08-18", "GLM-5.3", "**与 GLM-5.2 同一基座，全部提升来自后训练**"),
    ("2026-08-26", "GLM-5.3-Flash", "320B 总 / 18B 激活；GLM 系首个 linear+sparse 混合，原生多模态"),
)

# ---------------------------------------------------------------------------
# 【C】GLM-4.5 —— 首个 MoE 代（GQA + 稠密注意力）
# ---------------------------------------------------------------------------

# 来源：【C】GLM-4.5 技术报告 arXiv:2508.06471v1 Table 1 逐格抄录。
#   ⚠ 该表口径**逐字**：「When counting parameters, for GLM-4.5 and GLM-4.5-Air,
#     we include the parameters of MTP layers but not word embeddings and the output layer.」
#   → 即 **355B 不含 embedding 与输出层**。凡与别处口径不一（如含 embedding 的
#     753B/744B 之争），必须先回到这句话。
GLM45_CONFIG = {
    "total_params": 355e9,
    "active_params": 32e9,
    "dense_layers": 3,
    "moe_layers": 89,
    "mtp_layers": 1,
    "hidden_dim": 5120,
    "dense_intermediate_dim": 12288,
    "moe_intermediate_dim": 1536,
    "attn_head_dim": 128,        # 表中「Attention Head Dim」
    "attn_heads": 96,            # 表中「# Attention Heads」
    "kv_heads": 8,               # GQA-8；报告正文称 2048 维 KV cache
    "experts_total": 160,        # 表中「# Experts (total)」
    "routed_experts": 8,         # 每 token 激活的 routed expert 数
    "shared_experts": 1,
    "qk_norm": True,
    "vocab_size": 151552,
}
# 来源：【C】同上 §2.1 正文逐字：「we reduce the width (hidden dimension and number
#   of routed experts) of the model and increase its height (number of layers),
#   as we found that deeper models exhibited better reasoning capacity.」
#   这条是 GLM 与 DeepSeek-V3 / Kimi K2 的设计分歧点，文档 §0 直接引用。
GLM45_DESIGN_QUOTE = (
    "deeper models exhibited better reasoning capacity"
)
# 来源：【C】同上 §2.1 正文：注意力用 GQA + partial RoPE，96 头 for 5120 hidden；
#   QK-Norm 用于稳定 attention logits 的幅度。
GLM45_ATTENTION = "GQA（96 Q / 8 KV）+ partial RoPE + QK-Norm"

# ---------------------------------------------------------------------------
# 【C】GLM-5 / 5.1 —— DSA 代（MLA-256 + 稀疏注意力）
# ---------------------------------------------------------------------------

# 来源：【C】GLM-5 技术报告 arXiv:2602.15763v2 **Table 10** 逐格抄录
#   （「Model architecture of GLM-4.5 and GLM-5」，2026-10-13 读）。
#   ★ 口径同 GLM-4.5 那条：含 MTP 层参数，**不含 word embeddings 与 output layer**。
GLM5_CONFIG = {
    "total_params": 744e9,
    "active_params": 40e9,
    "dense_layers": 3,
    "moe_layers": 75,
    "mtp_layers": 1,
    "hidden_dim": 6144,
    "dense_intermediate_dim": 12288,
    "moe_intermediate_dim": 2048,
    "qk_head_dim": 192,          # 表中「QK Head Dim」
    "v_head_dim": 256,           # 表中「V Head Dim」
    "q_lora_dim": 2048,          # 表中「Q LoRA Dim」
    "kv_lora_dim": 512,          # 表中「KV LoRA Dim」
    "attn_heads": 64,            # 表中「# Attention Heads」（MLA 无独立 KV 头）
    "indexer_attn_heads": 32,    # 表中「# Indexer Attn Heads」
    "indexer_head_dim": 128,     # 表中「# Indexer Head Dim」
    "experts_total": 256,
    "routed_experts": 8,
    "shared_experts": 1,
    "vocab_size": 154880,
}
# ★ 层数的两种**都正确**的读法（同一个数字的不同含义并置）：
#   3 dense + 75 MoE = **78 个 transformer 层**（config 的 `num_hidden_layers` 口径），
#   再 +1 个 MTP 层 = **79 个「层」**（Table 10 的口径）。
#   来源：【C】Table 10 的 `# Dense Layers` / `# MoE Layers` / `# MTP Layers` 三行；
#         【D】Raschka 的 GLM-5.2 笔记转述 config 为「78 transformer layers」。
#   ⚠ 报告 §2.1 正文写「reduces its layer count to **80**」，与 Table 10 的 78/79
#     **对不上**（很可能是取整或把 MTP 另计）。**本文件以表为准**，正文引用时
#     必须写明是哪个口径，**不得把 78、79、80 混用**。
GLM5_TRANSFORMER_LAYERS = 78
GLM5_TOTAL_LAYERS_INCL_MTP = 79
GLM5_REPORT_PROSE_LAYER_COUNT = 80   # 正文原话，与表不符，仅作记录

# 来源：【C】同上 §2.1「Multi-latent Attention.」段逐字：
#   · MLA 用**572 维**量级的压缩 KV 比 GQA 省显存、长序列更快；
#   · 但在 Muon 优化器下，**576 维 latent KV-cache 的 MLA 追不上 GQA-8 的
#     2048 维 KV-cache**；
#   · 解法是把上投影矩阵 W^{UQ}, W^{UK}, W^{UV} 按头**拆开**再做正交化
#     （denoted as **Muon Split**），使不同头的投影权重以不同尺度更新；
#   · MLA 解码要做 **576 维点积**，高于 GQA 的 **128 维**；
#   · 故把 head dim 从 **192 提到 256**、attention 头数**减 1/3**（96 → 64），
#     记为 **MLA-256**，在 Muon Split 下与 MLA 打平。
GLM5_MLA_QUOTE = {
    "dense_kv_cache_dim_gqa8": 2048,
    "mla_latent_kv_cache_dim": 576,
    "mla_decode_dot_dim": 576,
    "gqa_decode_dot_dim": 128,
    "head_dim_before": 192,
    "head_dim_after": 256,
    "head_reduction": "1/3",
    "heads_after": 64,
    "variant_name": "MLA-256",
    "claimed_property": "This keeps the training computation and the number of parameters constant",
}
# ★ UNRESOLVED：报告那句「头维 192→256、头数减 1/3，训练计算与参数量不变」
#   **不能由 Table 10 的字段直接复算出来**：
#     96 头 × 192 = 18432  ≠  64 头 × 256 = 16384（差 −11.1%）。
#   若「head dim」指的是 V 维（表里 GLM-4.5 是 128、GLM-5 是 256，×2），
#   与「头数 ×2/3」相乘也不是常数（×4/3）。
#   → 本仓库**不据此推导**，只把原话与表值并列记录，并把这条列进编号程序 02 的
#     C 类断言（断言「两者不相等」，把「此处对不上」变成可复核的事实）。
GLM5_MLA_CONSTANT_CLAIM_UNRESOLVED = {
    "heads_x_qk_dim_before": 96 * 192,
    "heads_x_qk_dim_after": 64 * 256,
    "ratio": (64 * 256) / (96 * 192),
    "note": "报告称『不变』；按表值复算差 −11.1%。口径未知，记为 UNRESOLVED。",
}

# 来源：【C】同上 §2.1.1 与 §3.2 正文逐字：
#   · DSA 的 indexer 取 **k = 2048**（§3.2「the k = 2048 used by the indexer
#     is much larger than the k typically used in MoE」）；
#   · DSA **对所有层**都可应用（§2.1.2 结论段）；
#   · 「DSA reduces the attention computation by roughly **1.5-2×** for long sequences」；
#   · 「90% of attention entries in long contexts are indeed redundant」
#     （转述 DeepSeek-V3.2-Exp 的结论）。
DSA_TOPK = 2048
DSA_COMPUTE_REDUCTION_QUOTE = "roughly 1.5-2× for long sequences"
DSA_REDUNDANCY_QUOTE = "90% of attention entries in long contexts are indeed redundant"

# 来源：【C】同上 §2.1.1 的 DSA 续训配方：
#   起自 mid-training 结束时的基座；warm-up **1000 步**，每步 **14 条**
#   **202,752** token 的序列，最大学习率 **5e-3**；
#   稀疏适配阶段走 mid-training 的数据与超参，共 **20B** token。
#   ⚠ 对照：DeepSeek-V3.2 的 DSA 续训用了 **943.7B** token。
#   ★ **表 6** 另有小规模实验（GLM-4.7-Flash）：warm-up 只训 indexer、1000 步、
#     batch 16、基座冻结；联合训练 **150B** token。
DSA_CONTINUED_PRETRAIN = {
    "warmup_steps": 1000,
    "warmup_seqs_per_step": 14,
    "warmup_seq_len": 202752,
    "warmup_max_lr": 5e-3,
    "sparse_adapt_tokens": 20e9,
    "deepseek_v32_comparison_tokens": 943.7e9,
}
DSA_SMALL_SCALE_ABLATION = {
    "backbone": "GLM-4.7-Flash（MLA）",
    "warmup_steps": 1000,
    "warmup_batch": 16,
    "joint_tokens": 150e9,
}

# 来源：【C】同上 §2.1「Multi-token Prediction with Parameter Sharing.」逐字：
#   GLM-5 **共享 3 个 MTP 层的参数**用于训练，使 draft 模型的显存与 DeepSeek-V3
#   （单 MTP 层）一致，同时提高接受率。
#   Table 2 印值（私有 prompt set，同样 4 个投机步）：
#     DeepSeek-V3.2 accept length **2.55**；GLM-5 **2.76**。
MTP_SHARED_LAYERS = 3
MTP_ACCEPT_LENGTH = {"DeepSeek-V3.2": 2.55, "GLM-5": 2.76}
MTP_ACCEPT_LENGTH_SPEC = 4   # 比较时的投机步数

# 来源：【C】同上 §2.2 / §2.3 的训练规模：
#   base model 共 **28.5T** token；mid-training 三档上下文
#   **32K (1T) → 128K (500B) → 200K (50B)**；
#   SFT 阶段最大上下文扩到 **202,752**。
#   ⚠ 对照：GLM-4.5 是 **23T** token、最长 128K。
GLM5_TRAIN_TOKENS = 28.5e12
GLM45_TRAIN_TOKENS = 23e12
GLM5_MIDTRAIN_CONTEXT_STAGES = (("32K", 1e12), ("128K", 500e9), ("200K", 50e9))
GLM5_SFT_MAX_CONTEXT = 202752

# 来源：【C】同上附录 A（Hyper-Parameters）逐字：
#   「we follow the setting of GLM-4.5, including the **Muon optimizer**, cosine decay,
#    and batch size warmup. The learning rate goes through a warmup stage from **0 to 2e-4**,
#    and a decaying stage to **4e-5**…In the mid-training stage, the learning rate
#    decreases linearly from **4e-5 to 1e-5**.」
GLM5_OPTIMIZER = "Muon"
GLM5_LR_SCHEDULE = {"warmup_to": 2e-4, "decay_to": 4e-5, "midtrain_to": 1e-5}
DSA_LR = {"warmup_from": 5e-3, "warmup_to": 2e-4, "sparse_adapt": 1e-5}

# ---------------------------------------------------------------------------
# 【B】GLM-5.2 —— IndexShare 代（1M 上下文）
# ---------------------------------------------------------------------------

# 来源：【B】z.ai 发布页 `z.ai/blog/glm-5.2`（正文由页面 JS bundle 取出，2026-10-13）：
#   · IndexShare：**每 4 个 transformer 层共享一个轻量 indexer**；indexer 放在这 4 层
#     的**第一层**，top-k 索引被这 4 层共用 → 省掉 3/4 层的 indexer 点积与 top-k；
#   · 官方给的数字：「reducing per-token FLOPs by **2.9×** at a 1M context length」；
#   · MTP 层同样应用 IndexShare；改进后**接受长度最多 +20%**；
#   · 从 mid-training 起就以 **128K** 序列训练 IndexShare；
#   · 上下文 **1,048,576**（1M）；MIT 协议。
#   ⚠ 2.9× 的**计数口径官方未公布**：本仓库的解析推导给不出同一个数
#     （编号程序 02 会算出「indexer 侧降 4×、整体降幅远小于 4×」），
#     故此处只作引用值，**不得写成推导结论**。
INDEXSHARE_GROUP_SIZE = 4
INDEXSHARE_CLAIMED_FLOP_REDUCTION = 2.9
INDEXSHARE_CLAIMED_SCOPE = "1M 上下文；口径未公布（哪些 FLOPs 计入未知）"
INDEXSHARE_MTP_ACCEPT_GAIN_MAX = 0.20
INDEXSHARE_TRAINED_FROM = "mid-training，128K 序列"
GLM52_CONTEXT = 1_048_576

# 来源：【B】同上 + 【D】Raschka 笔记（2026-06-18）转述 GLM-5.2 的
#   `config.json`：「78 transformer layers with a hidden size of 6,144. The first
#   three feed-forward blocks are dense. The remaining blocks use 256 routed experts,
#   select 8 of them per token, and add one shared expert.」
#   ⚠【D】级，**只作旁证**：它与【C】Table 10（3 dense + 75 MoE, hidden 6144,
#     256 experts, k=8, shared 1）**逐项一致**，故采信为「5.2 沿用 5 代骨架」。
#   ⚠ 但**总参口径有分歧**：本文件**不落** 753B 这个数（来源为【D】级汇编，
#     且与【C】的 744B 差 1.2%，无法判定差异来自 embedding 还是 expert 计数）。
GLM52_CONFIG_SOURCE_D = {
    "transformer_layers": 78,
    "hidden_dim": 6144,
    "dense_ffn_layers": 3,
    "routed_experts": 256,
    "top_k": 8,
    "shared_experts": 1,
    "context": 1_048_576,
}
# 来源：【D】同上笔记对 IndexCache 论文的转述：相邻 DSA 层的 token 选择
#   **有 70%–100% 重叠**——这是 IndexShare 的动机。
#   ⚠【D】级转述，正文只用作**动机叙事**，不作数量结论。
INDEXCACHE_OVERLAP_QUOTE_D = "70% to 100% overlap between token selections in adjacent DSA layers"

# ---------------------------------------------------------------------------
# 【B】GLM-5.3 —— 与 5.2 同基座，提升全部来自后训练
# ---------------------------------------------------------------------------

# 来源：【B】z.ai 发布页 `z.ai/blog/glm-5.3` 与 docs.z.ai / docs.bigmodel.cn 的
#   GLM-5.3 介绍页逐字：
#   · 「It uses the **same base model as GLM-5.2** — every gain comes from post-training.」
#   · 1M 上下文；最大输出 128K；**仅文本**；
#   · 思考**不可关闭**，仅三个强度 `low` / `high` / `max`，**默认 max**；
#   · Z.ai Code Bench 上较 GLM-5.2 **+50%**；
#   · Terminal Bench 3.0、Agents' Last Exam (CLI) 上开源 SOTA；
#   · 权重在发布后两周放出（安全评估完成后）。
GLM53_SAME_BASE_AS = "GLM-5.2"
GLM53_IMPROVEMENT_SOURCE = "post-training only"
GLM53_CONTEXT = 1_048_576
GLM53_MAX_OUTPUT = 128_000
GLM53_THINKING = {"type": "enabled only", "reasoning_effort": ("low", "high", "max"),
                  "default": "max"}

# 来源：【B】同上发布页的**评测条件脚注**逐字（这些是「在什么条件下测的」，
#   CONTRIBUTING 要求引用外部数字必须写出）：
#   · Terminal-Bench-3：Claude Code **2.1.207** harness，reasoning effort=max，
#     400K 上下文、128K 最大输出，avg@3，每任务独立容器，**600 agent turns / 10h** 超时；
#   · ALE：官方协议 + Claude Code harness（max effort，1M 上下文，64K 输出），
#     105 个任务，默认 4h、上限 8h；
#   · SWE-Marathon / ExploitGym 的预算按**每模型 TPS** 归一：
#     GLM-5.3 按 **115 TPS**、Kimi K3 按 **40 TPS**、Qwen3.8 Max 按 **47 TPS** 折算。
GLM53_EVAL_CONDITIONS = {
    "harness": "Claude Code 2.1.207",
    "reasoning_effort": "max",
    "terminal_bench_3": {"context": 400_000, "max_output": 128_000,
                         "agent_turns": 600, "timeout_h": 10, "report": "avg@3"},
    "ale": {"context": 1_048_576, "max_output": 64_000, "tasks": 105,
            "timeout_h_default": 4, "timeout_h_max": 8},
    "tps_normalization": {"GLM-5.3": 115, "Kimi K3": 40, "Qwen3.8 Max": 47},
    "judge": "GPT-5.6-luna (medium)",
}
# 来源：【B】同上发布页的分数表（**引用值**，本仓库不复核其测量口径）。
#   ⚠ 表里的模型名是 2026 年 8 月的快照（含 Kimi K3 / DeepSeek-V4 Pro-0813 /
#     Qwen3.8-Max / Opus 4.8 / Fable 5 / GPT-5.6 Sol）——**榜单会过时**。
GLM53_BENCH = {
    "Terminal Bench 2.1": {"GLM-5.3": 88.2, "GLM-5.2": 81.0},
    "Terminal Bench 3.0": {"GLM-5.3": 28.3, "GLM-5.2": 4.6},
    "DeepSWE (v1.1)": {"GLM-5.3": 66.9, "GLM-5.2": 46.2},
    "NL2Repo": {"GLM-5.3": 58.0, "GLM-5.2": 48.9},
    "SWE-Marathon (v1.1)": {"GLM-5.3": 42.5, "GLM-5.2": 19.4},
    "Agents' Last Exam": {"GLM-5.3": 28.5, "GLM-5.2": 23.8},
    "CyberGym": {"GLM-5.3": 84.5, "GLM-5.2": 77.2},
    "ExploitBench": {"GLM-5.3": 54.4, "GLM-5.2": 24.4},
}
# 来源：【B】同上发布页的 token 效率图注：「At Max effort, GLM-5.3 reaches **34.5%**
#   at roughly **75K** output tokens per task, compared with **23.4%** at **96K**
#   for GLM-5.2.」
GLM53_TOKEN_EFFICIENCY = {
    "GLM-5.3@max": {"score": 34.5, "output_tokens": 75_000},
    "GLM-5.2@max": {"score": 23.4, "output_tokens": 96_000},
}
# 来源：【B】同上发布页的真实世界漏洞统计：与国内多家安全团队合作，在 **269** 个项目上
#   识别 **2,436** 个漏洞，其中 **1,097** 个中高危；最老的漏洞约 40 年前引入
#   （另一处写「Findings span 45 years of impact - the oldest flaw was introduced
#   in 1981」）。
GLM53_CVE_STATS = {"projects": 269, "vulns": 2436, "medium_high": 1097,
                   "oldest_introduced_year": 1981}

# 来源：【B】同上发布页的系统侧数字：训练-推理一致性把平均 logprob 差
#   控制在 **1e-7** 量级（较此前设置下降 **>99.99%**）；
#   长程编码 RL 任务的端到端 RL 训练吞吐提升 **>2.3×**；
#   框架为 **slime**（训练侧 Megatron、rollout 侧 SGLang）。
GLM53_RL_INFRA = {
    "framework": "slime",
    "train_side": "Megatron",
    "rollout_side": "SGLang",
    "logprob_gap": "1e-7 量级",
    "logprob_gap_reduction": ">99.99%",
    "throughput_gain": 2.3,
}

# ---------------------------------------------------------------------------
# 【B】GLM-5.3-Flash —— linear + sparse 混合（新基座）
# ---------------------------------------------------------------------------

# 来源：【B】docs.z.ai `guides/vlm/glm-5.3-flash` 与 `z.ai/blog/glm-5.3-flash`
#   （2026-08-26）逐字：
#   · **320B 总 / 18B 激活**；层数 **45**（对比 GLM-4.5 系列的 92）；
#   · GLM 系首个 **sparse + linear attention 混合**架构；
#     「Linear attention captures local dependencies through **state modeling**,
#      while sparse attention retrieves relevant global context through a
#      lightweight **indexer**.」
#   · **IndexPool**：把 **4 个 indexer key 向量加权池化成 1 个**，压低 1M 下的
#     indexer 延迟与显存；
#   · **mHC**（Manifold-Constrained Hyper-Connections）用于提升 scaling 效率；
#   · 30T token 多模态预训练语料；
#   · 效率对比口径逐字：「we calculate the **attention compute per head per layer**
#     and **average KV cache size per layer (BF16)**」→ 相对 GLM-5.3
#     **注意力计算少 3.0×、KV cache 小 4.4×**（API 页写 3.01× / 4.44×）。
GLM53_FLASH_CONFIG = {
    "total_params": 320e9,
    "active_params": 18e9,
    "layers": 45,
    "context": 1_048_576,
    "max_output": 128_000,
    "modalities_in": ("video", "image", "text", "file"),
    "attention": "linear + sparse hybrid（官方未点名 linear 变体）",
    "indexpool_group": 4,
    "mhc": True,
    "pretrain_tokens": 30e12,
}
GLM53_FLASH_EFFICIENCY = {
    "vs_model": "GLM-5.3",
    "attention_compute_reduction": 3.0,   # API 页写 3.01
    "kv_cache_reduction": 4.4,            # API 页写 4.44
    "normalization": "attention compute per head per layer；KV cache 为每层平均、BF16",
}
# 来源：【B】同上发布页的服务栈逐字：SGLang 之上的专用推理引擎；
#   「intra-node tensor parallelism for linear attention and the LM head,
#    **ReplaySSM**, **W8A8** quantization, **hybrid INT8/FP8/BF16 cache quantization**,
#    and **Layer Split**」；集群侧为 **EPD**（Encode–Prefill–Decode）分离架构；
#   相比同硬件初始基线端到端服务性能 **3×**，每 token 成本「与主流 NVIDIA GPU 相当」。
GLM53_FLASH_SERVING = {
    "engine_base": "SGLang",
    "techniques": ("intra-node TP for linear attention + LM head", "ReplaySSM",
                   "W8A8", "hybrid INT8/FP8/BF16 KV cache quantization", "Layer Split"),
    "cluster": "EPD disaggregation（Encode / Prefill / Decode 三池）",
    "speedup_vs_own_baseline": 3.0,
}

# ---------------------------------------------------------------------------
# 【C】工程落地：INT4 QAT 与国产芯片适配
# ---------------------------------------------------------------------------

# 来源：【C】GLM-5 技术报告 §2.4.3 逐字：「we apply **INT4 QAT in the SFT stage**…
#   we have developed a quantization kernel applicable to both training and offline
#   weight quantization, which ensures **bitwise-identical behavior** between training
#   and inference.」
GLM5_INT4_QAT = {
    "stage": "SFT",
    "claim": "训练与推理逐比特一致（同一量化 kernel 既用于训练也用于离线权重量化）",
}
# 来源：【C】同上 §5（Adapting GLM-5 to Chinese Chip Infrastructure）逐字：
#   · 7 家国产平台：**华为昇腾、摩尔线程、海光、寒武纪、昆仑芯、沐曦、燧原**；
#   · 案例机型：**Atlas 800T A3 单机**；
#   · 分组件精度：**Attention 与 MLP 用 W8A8（INT8），MoE experts 压到 W4A8（INT4）**；
#     用的是 **msModelSlim** 工具 + **QuaRot**（outlier 抑制）+ **Flex_AWQ_SSZ**（scale 校准）；
#   · 融合算子三件套：**Lightning Indexer**（score + ReLU + TopK 融进一个 kernel）、
#     **Sparse Flash Attention**（选 TopK token 与稀疏注意力并行）、
#     **MLAPO**（把 **13** 个小预处理算子融成 1 个，Vector 与 Cube 单元并行）；
#   · 引擎侧：vLLM-Ascend 的异步调度（D2H 采样拷贝与下一步 decode 准备重叠）、
#     **RadixCache**（前缀共享）与 **Prefix Cache**（KV 扩到系统内存）、
#     Attention DP + MoE EP 的混合并行。
GLM5_CHINESE_CHIPS = {
    "platforms": ("Huawei Ascend", "Moore Threads", "Hygon", "Cambricon",
                  "Kunlunxin", "MetaX", "Enflame"),
    "case_machine": "Atlas 800T A3（单机容纳 750B 模型）",
    "precision_split": {"attention_and_mlp": "W8A8 (INT8)", "moe_experts": "W4A8 (INT4)"},
    "quant_tools": ("msModelSlim", "QuaRot", "Flex_AWQ_SSZ"),
    "fused_kernels": ("Lightning Indexer", "Sparse Flash Attention", "MLAPO"),
    "mlapo_fused_ops": 13,
    "engine_optimizations": ("vLLM-Ascend 异步调度", "RadixCache", "Prefix Cache",
                             "Attention DP + MoE EP 混合并行"),
}

# ---------------------------------------------------------------------------
# 【C】训练基础设施（与本主题的并行 / 显存账直接相关）
# ---------------------------------------------------------------------------

# 来源：【C】GLM-5 技术报告 §2.4.1 逐字，五项显存优化：
#   · **Flexible MTP placement**（MTP 的输出层与主输出层同驻最后一个 stage 以共享参数，
#     embedding 与 transformer 部分放前一 stage）；
#   · **Pipeline ZeRO2 gradient sharding**（每 stage 只存 1/dp 的梯度，
#     只保留两个 stage 的完整累积缓冲并双缓冲复用）；
#   · **Zero-redundant communication for the Muon distributed optimizer**
#     （all-gather 限制在本 rank 拥有的参数分片，与本地计算重叠）；
#   · **Pipeline activation offloading**（按层粒度 offload 到主机内存，回载与计算重叠，
#     且避开 P2P 通信与 MoE 的 dispatch/combine 争用）；
#   · **Sequence-chunked output projection**（切块算投影与交叉熵，逐块释放激活）。
GLM5_MEMORY_OPTIMIZATIONS = (
    "Flexible MTP placement",
    "Pipeline ZeRO2 gradient sharding",
    "Zero-redundant Muon distributed optimizer",
    "Pipeline activation offloading（层粒度）",
    "Sequence-chunked output projection",
)
# 来源：【C】同上 §2.4.2 逐字：**deferred weight gradient computation**（把部分权重的
#   梯度计算移出关键路径以减气泡）；长序列训练用 workload-aware 重排、注意力计算的
#   动态重分配、DP rank 按 context-parallel 组大小灵活划分；**hierarchical all-to-all**
#   把 QKV 张量的节点内与节点间通信重叠。
GLM5_PARALLEL_OPTIMIZATIONS = (
    "deferred weight gradient computation",
    "workload-aware sequence reordering",
    "dynamic redistribution of attention computation",
    "flexible context-parallel grouping of DP ranks",
    "hierarchical all-to-all for QKV",
)

# ---------------------------------------------------------------------------
# 【C】后训练流水线（GLM-5；GLM-5.3 的提升即在此放大）
# ---------------------------------------------------------------------------

# 来源：【C】GLM-5 技术报告 §3 与 Figure 5 逐字：SFT → Reasoning RL → Agentic RL
#   → General RL → **On-Policy Cross-Stage Distillation**（最后阶段，用前序阶段的
#   最终 checkpoint 当教师，防能力回退）。
GLM5_POSTTRAIN_PIPELINE = ("SFT", "Reasoning RL", "Agentic RL", "General RL",
                           "On-Policy Cross-Stage Distillation")
# 来源：【C】同上 §3.1：三种思考特性——**Interleaved Thinking**（每次响应与每次
#   工具调用前都思考）、**Preserved Thinking**（编码 agent 场景跨多轮保留全部
#   thinking block，复用已有推理）、**Turn-level Thinking**（按轮控制开关）。
GLM5_THINKING_FEATURES = ("Interleaved Thinking", "Preserved Thinking",
                          "Turn-level Thinking")
# 来源：【C】同上 §3.3：异步解耦 RL，用 **Multi-Task Rollout Orchestrator** 解耦
#   推理与训练引擎；**TITO（Token-in-Token-out）** 网关消除重分词不匹配；
#   **Direct Double-sided Importance Sampling** 做 token 级裁剪；**DP-aware routing**
#   提高长上下文 MoE 推理的 KV 复用；环境侧规模化到 **>10K** 真实 SWE / terminal /
#   多跳搜索任务。
GLM5_AGENTIC_RL = {
    "orchestrator": "Multi-Task Rollout Orchestrator",
    "gateway": "Token-in-Token-out (TITO)",
    "off_policy_control": "Direct Double-sided Importance Sampling",
    "routing": "DP-aware routing",
    "environments": ">10K（SWE / terminal / 多跳搜索）",
}
# 来源：【C】同上 §3.2：RL 主干是 **GRPO** + **IcePop**（缓解 train-inference
#   mismatch），去掉了 IcePop 原有的 KL 正则项；indexer 侧重放不可行（k=2048），
#   改用**确定性 top-k**（`torch.topk`）解决训练-推理不一致，RL 期间**冻结 indexer 参数**。
GLM5_RL_BACKBONE = "GRPO + IcePop（去 KL 项）"
GLM5_DSA_RL_FIX = ("确定性 top-k（torch.topk）替代非确定性 CUDA/TileLang 实现",
                   "RL 期间冻结 indexer 参数")

# ---------------------------------------------------------------------------
# 术语
# ---------------------------------------------------------------------------

TERMS = {
    "total params": "权重占显存与 EP 切分规模的口径；**不等于**单 token 算力",
    "active params": "单 token 的 matmul 量口径（GLM-5 为 40B）；**不等于**模型大小",
    "MLA": "把 K/V 压进低秩 latent，只用一份压缩 KV；解码时按 MQA 模式共享该 latent",
    "MLA-256": "GLM-5 的 MLA 变体：head dim 表值 QK 192 / V 256，头数 64（口径见 UNRESOLVED）",
    "Muon Split": "把 W^{UQ}/W^{UK}/W^{UV} 按头拆开再做矩阵正交化，使各头以不同尺度更新",
    "DSA": "DeepSeek Sparse Attention：lightning indexer 先取 top-k token，再只对子集做注意力",
    "k（indexer 的）": "DSA 每 token 选中的 KV 条目数，GLM-5 为 **2048**；MoE 的 k 是专家数，**两者不同**",
    "IndexShare": "每 4 层共享一个 indexer 的 top-k 索引（只共享**选中的位置**，不共享 Q/K/V 计算）",
    "IndexPool": "GLM-5.3-Flash 里把 4 个 indexer key 向量加权池化成 1 个（省 indexer 的显存与延迟）",
    "MTP": "Multi-Token Prediction：训练时多步预测，推理时当 draft head 做投机解码",
    "mHC": "Manifold-Constrained Hyper-Connections：把残差扩成多路并约束到流形上（GLM-5.3-Flash 采用）",
    "W4A8 / W8A8": "权重 4bit/8bit、激活 8bit；GLM-5 的 MoE expert 用 W4A8，其余 W8A8",
    "口径": "同一个量在不同来源下的测量/计数约定；本主题里 78 / 79 / 80 层、744B / 753B 都是口径问题",
    "UNRESOLVED": "报告表述与表值对不上的项（MLA-256 的『计算不变』），**本仓库不替它圆场**",
}

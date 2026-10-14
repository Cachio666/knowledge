"""160-model/deepseek 的外部规格常量源。

本文件只放两类东西：
  1. **从一手来源抄来的**外部规格（DeepSeek 官方技术报告的章节与印值、
     DeepSeek API docs 的发布记录、arXiv 页面的摘要），每条带一行 `# 来源：`；
  2. **口径注记**——同一个量在不同来源下不是同一个数的，逐处写明。

★ 外部来源分级（CONTRIBUTING §2，A–E）在本文件的记法：
    【C】论文 / 官方技术报告（**含 arXiv 预印本与官方 PDF 技术报告，均未确认同行评审**）
    【B】产品官方文档（api-docs.deepseek.com 的发布页 / 模型页 / 价格页）
    【D】社区项目文档（个人博客、二次解读）
  本域没有把模型规格归入【A】的依据：A 级是**硬件厂商规范**，模型规格不属于该级。

★ 核查范围（不写「全库」「已核实全部」这类措辞）：本文件逐条核的载体是——
  · DeepSeek-V2 arXiv:2405.04434 的 abs 页摘要；
  · DeepSeek-V3 arXiv:2412.19437（经 V4 报告的参考文献确认编号）；
  · DeepSeek-V3.2 arXiv:2512.02556v1 的正文（本机 curl 取回 HTML 后转文本，2026-10-13）；
  · DeepSeek-V4 arXiv:2606.19348v1 的正文；
  · **DeepSeek-V4.1-Flash 官方技术报告 PDF**（标题 *Pushing the Limits of KV Cache
    Compression*）——⚠ **取回渠道要写明**：`huggingface.co` 在本机不可达
    （curl 返回 000、WebFetch 报 fetch failed），本文件用的是 **ModelScope 上的同名
    官方文件**（`modelscope.cn/models/deepseek-ai/DeepSeek-V4.1-Flash/resolve/master/
    DeepSeek_V41_Tech_Report.pdf`，1 809 802 B，`pdftotext -layout` 转文本后逐节读）。
    正文引用时标为【C】，并注明**未逐字核对 HF 原文件与 ModelScope 副本的一致性**。
  · DeepSeek API docs 的发布记录页 `api-docs.deepseek.com/news/news260910` 与其导航列出的
    全部历史发布条目（【B】）。

★ 本文件**不放任何算法**，也**不 import 任何第三方库**：
  凡被断言检验的算式（参数量与激活量、CED 的 prefill 折半、CSA2 的有效 KV 字节、
  层级稀疏索引器的打分成本）**一律在编号程序里另写**。

UNSOURCED / UNRESOLVED（正文不使用或必须标注）：
  · **V4-Flash 与 V4.1-Flash 的完整 `config.json`**：huggingface.co 不可达，取不到。
    本文件只用**技术报告正文里逐字给出的配置**，不落 config 字段。
  · **DeepSeek-V4-Flash 相对 V3.2 的效率百分比**：V4 报告正文只给了 **V4-Pro** 的
    27% FLOPs / 10% KV。V4-Flash 那一档本文件**不落数**（社区文档有 10% / 7% 的转述，
    属【D】级且无法定位一手，故不采用）。
  · **「890 字节/token」的逐项拆解**：报告给了**总数**，没有给「哪些层存多少字节」的
    明细。编号程序 02 会**反推**一个拆解（并断言它与 890 B 吻合），但**必须标为反推**，
    不得写成官方口径。
  · **DeepSeek-R1 的 arXiv 编号**：本文件**不落**（未在本次核查中取到一手页面）；
    R1 只以【B】级发布记录的形式进入时间线。
  · **DeepSeek-V3.1 的详细规格**：只有【B】级发布记录，报告级材料未取到。
"""

# ---------------------------------------------------------------------------
# 版本锚点（本机）——编号程序的测量条件
# ---------------------------------------------------------------------------

# 来源：本机实测 `python3 -VV` 与 `numpy.__version__`（2026-10-13）。
MEASURE_HOST = "Apple M1 Pro（10 核）/ macOS / **CPU-only，无 GPU 数字**"
MEASURE_PYTHON = "3.12.9 (Anaconda, Clang 14.0.6)"
MEASURE_NUMPY = "2.3.4"
PROGRAMS_ARE_DERIVATIONS = True

# ---------------------------------------------------------------------------
# 【B】发布时间线（DeepSeek API docs 的发布记录，逐条读）
# ---------------------------------------------------------------------------

# 来源：【B】`api-docs.deepseek.com` 的 News 导航逐条（2026-10-13 读）。日期为页面标题所示。
DEEPSEEK_TIMELINE = (
    ("2024-05", "DeepSeek-V2", "首发 MLA + DeepSeekMoE（236B 总 / 21B 激活，128K）"),
    ("2024-09-05", "DeepSeek-V2.5", "V2 系列的对话版合并"),
    ("2024-12-26", "DeepSeek-V3", "671B 总 / 37B 激活；FP8 训练；aux-loss-free 均衡；MTP"),
    ("2025-01-20", "DeepSeek-R1", "纯 RL 激发推理能力（GRPO）"),
    ("2025-03-25", "DeepSeek-V3-0324", "V3 的小版本更新"),
    ("2025-05-28", "DeepSeek-R1-0528", "R1 的小版本更新"),
    ("2025-08-21", "DeepSeek-V3.1", "混合思考模式（thinking / non-thinking 同一权重）"),
    ("2025-09-29", "DeepSeek-V3.2-Exp", "引入 DSA（稀疏注意力）的实验版"),
    ("2025-12-01", "DeepSeek-V3.2", "DSA 正式版"),
    ("2026-04-24", "DeepSeek-V4 Preview", "V4 预览：CSA + HCA 混合注意力、mHC、Muon"),
    ("2026-08-13", "DeepSeek-V4-Pro GA", "V4-Pro 正式可用"),
    ("2026-08-21", "DeepSeek-V4-Flash-Vision-Exp", "多模态 API"),
    ("2026-09-10", "DeepSeek-V4.1-Flash", "CED + CSA2 + FP4 KV + SWA Bounded Replay"),
)
# 来源：【B】同上 V4.1-Flash 发布页逐字：V4-Flash 与 V4-Flash-Vision-Exp **下线**，
#   `deepseek-v4-flash` 临时路由到 V4.1-Flash；V4-Pro 计划下线（后又保留）。
V41_RETIRES = ("DeepSeek-V4-Flash", "DeepSeek-V4-Flash-Vision-Exp")

# ---------------------------------------------------------------------------
# 【C】DeepSeek-V2 —— MLA 与 DeepSeekMoE 的起点
# ---------------------------------------------------------------------------

# 来源：【C】arXiv:2405.04434 abs 页摘要逐字（本机 curl 取回，2026-10-13）：
#   「236B total parameters, of which 21B are activated for each token, and
#    supports a context length of 128K tokens.」
V2_CONFIG = {"total_params": 236e9, "active_params": 21e9, "context": 131072,
             "innovations": ("Multi-head Latent Attention (MLA)", "DeepSeekMoE")}

# ---------------------------------------------------------------------------
# 【C】DeepSeek-V3 —— FP8 训练 + aux-loss-free + MTP 的经典骨架
# ---------------------------------------------------------------------------

# 来源：【C】arXiv:2412.19437（编号经 V4 报告参考文献确认；本文件未逐字读 V3 原文）。
#   ⚠ 这一组是**本仓库未逐字核实原文**的项，只在时间线里作世代标记，
#     编号程序**不拿它做断言**。
V3_CONFIG = {"total_params": 671e9, "active_params": 37e9, "layers": 61,
             "hidden_dim": 7168, "experts_total": 256, "routed_experts": 8,
             "shared_experts": 1, "expert_intermediate": 2048, "context": 131072}
V3_CONFIG_SOURCE_NOTE = "编号经 V4 参考文献确认；数值未逐字核对 V3 原文"

# ---------------------------------------------------------------------------
# 【C】DeepSeek-V3.2 —— DSA 的正式配方
# ---------------------------------------------------------------------------

# 来源：【C】arXiv:2512.02556v1 正文（本机取回 HTML 转文本，2026-10-13）：
#   · lightning indexer 的分数定义 I_{t,s} = Σ_j w^I_{t,j} · ReLU(q^I_{t,j} · k^I_s)；
#   · 激活函数选 **ReLU**（理由：throughput）；
#   · 稀疏训练阶段：学习率 7.3e-6，**每个 query token 选 2048 个 KV token**；
#     主模型与 indexer 一起训 15000 步 × 480 条 × 128K 序列 = **943.7B token**；
#   · 起点是 **V3.1-Terminus** 的 128K 基座 checkpoint，分 dense warm-up + sparse training 两阶段。
V32_CONFIG = {
    "continual_pretrain_from": "DeepSeek-V3.1-Terminus（128K）",
    "indexer_topk": 2048,
    "indexer_activation": "ReLU",
    "sparse_stage_lr": 7.3e-6,
    "sparse_stage_steps": 15000,
    "sparse_stage_seqs_per_step": 480,
    "sparse_stage_seq_len": 131072,
}
V32_SPARSE_TOKENS = 943.7e9
V32_INDEXER_FORMULA = "I_{t,s} = Σ_j w^I_{t,j} · ReLU(q^I_{t,j} · k^I_s)"

# ---------------------------------------------------------------------------
# 【C】DeepSeek-V4 —— CSA + HCA 混合、mHC、Muon
# ---------------------------------------------------------------------------

# 来源：【C】arXiv:2606.19348v1 摘要与 §4.2（本机取回，2026-10-13）：
#   · V4-Pro **1.6T / 49B**、V4-Flash **284B / 13B**，两者都支持 **1M** 上下文；
#   · 预训练 **>32T** token；
#   · 1M 上下文下 V4-Pro 只需 **V3.2 的 27% 单 token 推理 FLOPs 与 10% KV cache**
#     （原文明写「measured in equivalent FP8 FLOPs」）。
V4_FAMILY = {
    "V4-Pro": {"total_params": 1.6e12, "active_params": 49e9, "context": 1048576},
    "V4-Flash": {"total_params": 284e9, "active_params": 13e9, "context": 1048576},
}
V4_PRETRAIN_TOKENS = 32e12
V4_VS_V32_AT_1M = {"flops_fraction": 0.27, "kv_fraction": 0.10,
                   "unit": "equivalent FP8 FLOPs"}

# 来源：【C】同上 §4.2.1 逐字（**两个型号分别给出**，不得互相套用）：
#   V4-Pro：61 层，hidden 7168；前 2 层 HCA，其后 CSA/HCA **交错**；
#     CSA 压缩率 **m = 4**，indexer query 头 **64**，indexer head dim **c^I = 128**，
#     attention top-k = **1024**；HCA 压缩率 **m' = 128**；
#     两种注意力都是 query 头 **128**、head dim **c = 512**、query 压缩维 **d_c = 1536**；
#     输出投影分组 **g = 16**，每组中间维 **d_g = 1024**；SWA 窗口 **n_win = 128**；
#     MoE：前 3 层用 **Hash routing**，每层 1 shared + **384** routed，expert 中间维 **3072**，
#     每 token 激活 **6** 个；MTP 深度 1；mHC 扩展因子 **4**，Sinkhorn-Knopp 迭代 **20**。
#   V4-Flash：1 shared + **256** routed，expert 中间维 **2048**，每 token 激活 **6** 个；
#     其余与 Pro 同。
V4_PRO_CONFIG = {
    "layers": 61, "hidden_dim": 7168,
    "csa_compression": 4, "hca_compression": 128,
    "indexer_query_heads": 64, "indexer_head_dim": 128, "attention_topk": 1024,
    "query_heads": 128, "head_dim": 512, "query_compression_dim": 1536,
    "output_groups": 16, "output_group_dim": 1024,
    "swa_window": 128,
    "shared_experts": 1, "routed_experts": 384, "activated_experts": 6,
    "expert_intermediate": 3072,
    "hash_routing_layers": 3, "mtp_depth": 1,
    "mhc_expansion": 4, "sinkhorn_iters": 20,
}
V4_FLASH_CONFIG = dict(V4_PRO_CONFIG, routed_experts=256, expert_intermediate=2048)
V4_HYBRID_LAYOUT = "前 2 层 HCA，其余 CSA 与 HCA 交错"

# ---------------------------------------------------------------------------
# 【C】DeepSeek-V4.1-Flash —— CED + CSA2 + FP4 KV（本主题的实例）
# ---------------------------------------------------------------------------

# 来源：【C】DeepSeek-V4.1-Flash 官方技术报告 PDF 摘要与 §2.1 逐字（取回渠道见文件头）：
#   · **552B 主干参数 + 196B Engram 参数**，原生多模态，**1M** 上下文；
#   · 主干 **40 层 = 20 层 causal encoder + 20 层 decoder**；
#   · **prefill 每 token 激活 8B，decode 每 token 激活 16B**；
#   · 全局 KV cache（常驻 HBM）压到 **890 字节/token**，约为 V4-Flash 的 **1/4**、
#     DeepSeek-V1 的 **1/437**；持久化 KV（SSD / 主机内存）约为 V4-Flash 的 **1/8**；
#   · 预训练语料 **45T token**；稀疏注意力**从零开始**在 64K 序列上训练，
#     **没有 dense attention warm-up**；
#   · 上下文从 4K 拉到 1M（**256 倍**），单 token decode FLOPs **只增加 1/4**。
V41 = {
    "backbone_params": 552e9,
    "engram_params": 196e9,
    "context": 1048576,
    "layers": 40,
    "encoder_layers": 20,
    "decoder_layers": 20,
    "hidden_dim": 5120,
    "active_prefill": 8e9,
    "active_decode": 16e9,
    "pretrain_tokens": 45e12,
    "global_kv_bytes_per_token": 890,
    "global_kv_vs_v4flash": 0.25,
    "persistent_kv_vs_v4flash": 0.125,
    "v1_kv_bytes_per_token": 389120,
    "decode_flops_growth_for_256x_context": 0.25,
    "sparse_attention_trained_from_scratch_at": 65536,
    "dense_warmup": False,
}
# 来源：【C】同上摘要逐字：「DeepSeek-V4.1-Flash-Base achieves world knowledge, reasoning
#   and coding abilities comparable to DeepSeek-V4-Pro-Base, and delivers 5%–10% improvements
#   on held-out evaluations, using only **1/3 total parameters and 1/4 activated parameters**.」
#   ⚠ **这条口径对不上**：总参 552B/1600B = 0.345（≈1/3 ✓），但激活那侧
#     8B/49B = 0.163、16B/49B = 0.327 —— **两个都不是 1/4**。
#     报告没有说明 1/4 的分母是哪个口径。编号程序 01 的 C2 断言把这个
#     「1/3 成立、1/4 对不上」钉成可复核的事实；正文不得替官方选一个分母。
V41_VS_V4PRO_CLAIM = {"total": "1/3", "activated": "1/4",
                      "denominator_unspecified": True}

# 来源：【C】同上 §4.2.1「Model Setups」逐字（**这是本实例最硬的一组配置**）：
#   · 40 层，hidden **d = 5120**，encoder 20 + decoder 20；
#   · 前 2 层**纯 SWA**；其余 **18 个 encoder 层**用 CSA2，压缩率 **m = 2**，
#     分成 **3 组各 6 层**，每组第 1 层 **Full Mode**、其余 **5 层 Reuse Mode**；
#   · **20 个 decoder 层**用 CSA2，压缩率 **m = 1**，分成 **5 组各 4 层**；
#     第 1 组：第 1 层 Full、其余 3 层 Reuse；其余 4 组：第 1 层 **Reindex**、其余 3 层 Reuse；
#   · 所有 CSA2 层：indexer query 头 **32**、indexer head dim **128**、attention top-k **512**；
#     主注意力 query 头 **64**、head dim **512**、query 压缩维 **1280**；
#   · 层级稀疏索引器：最多选 **2048 个 block × 每块 8 个位置 = 至多 16384 个候选位置**；
#   · 输出投影分组 **8**，每组中间维 **1024**；SWA 窗口 **n_win = 128**；
#   · MoE：每层 1 shared + **384** routed，expert 中间维 **2304**，每 token 激活 **6** 个；
#     SwiGLU 带 **clamp 阈值 10**；
#   · mHC 扩展因子 **4**，Sinkhorn-Knopp 迭代 **20**；
#   · 视觉编码器：**32 层**、hidden **1024**、**16** 个注意力头、patch **14**；
#     视觉 MLP projector **2 层**、hidden **5120**。
V41_CONFIG = {
    "layers": 40, "encoder_layers": 20, "decoder_layers": 20, "hidden_dim": 5120,
    "pure_swa_layers": 2,
    "encoder_csa2_layers": 18, "encoder_compression": 2,
    "encoder_groups": 3, "encoder_group_size": 6,
    "encoder_group_pattern": ("Full", "Reuse", "Reuse", "Reuse", "Reuse", "Reuse"),
    "decoder_csa2_layers": 20, "decoder_compression": 1,
    "decoder_groups": 5, "decoder_group_size": 4,
    "decoder_group1_pattern": ("Full", "Reuse", "Reuse", "Reuse"),
    "decoder_group_rest_pattern": ("Reindex", "Reuse", "Reuse", "Reuse"),
    "indexer_query_heads": 32, "indexer_head_dim": 128, "attention_topk": 512,
    "query_heads": 64, "head_dim": 512, "query_compression_dim": 1280,
    "hier_blocks": 2048, "hier_block_positions": 8,
    "hier_candidates": 2048 * 8,
    "output_groups": 8, "output_group_dim": 1024,
    "swa_window": 128,
    "shared_experts": 1, "routed_experts": 384, "activated_experts": 6,
    "expert_intermediate": 2304, "swiglu_clamp": 10.0,
    "mhc_expansion": 4, "sinkhorn_iters": 20,
    "vit_layers": 32, "vit_hidden": 1024, "vit_heads": 16, "vit_patch": 14,
    "projector_layers": 2, "projector_hidden": 5120,
}

# 来源：【C】同上 §2.4.4「FP4 Main KV Cache」：格式是 **E2M1 配每 16 通道一个
#   E4M3 缩放因子**（报告口径：**接近 NVFP4 但省掉它的二级全局缩放**）。
V41_KV_FORMAT = {
    "element": "E2M1 (FP4)",
    "scale": "E4M3，每 16 个通道一个",
    "note": "官方口径：接近 NVFP4，但去掉 NVFP4 的第二级 per-tensor 全局 scale",
    "bytes_per_element": 0.5,
}

# 来源：【C】同上 §2.4.2「Engram」逐字：
#   · **196B 参数平均分给两个模块**，放在**第 1 层与第 14 层**；
#   · **{2, 3, 4} 阶 N-gram**、**8 个哈希头**、每阶总嵌入维 **2048**、
#     每个头索引约 **1600 万**条目、表大小取**互不相同的质数**；
#   · 嵌入表与投影都用 **FP8**；推理时用**确定性寻址**从主机内存**后台 RDMA 预取**；
#   · 相比原始 Engram 设计**去掉了短因果卷积**。
V41_ENGRAM = {
    "params": 196e9, "modules": 2, "layers": (1, 14),
    "ngram_orders": (2, 3, 4), "hash_heads": 8, "embed_dim_per_order": 2048,
    "entries_per_head": 16e6, "precision": "FP8",
    "prefetch": "确定性寻址 + 主机内存后台 RDMA 预取",
    "removed": "原始设计的短因果卷积",
}

# 来源：【C】同上 §2.4.3「DSpark」逐字：
#   · drafter 由 **3 个 Transformer block**组成，滑动窗口 **128** token；
#   · 一次前向**并行给出 5 个草稿位置**的 base logits，另加一个轻量 **Markov 头**
#     建模草稿 token 之间的依赖，以及一个**置信度头**预测逐位置接受概率；
#   · 调度器结合引擎吞吐曲线**动态决定每个请求的验证长度**；
#   · 与 V3 的 MTP（全程与主干联合训练）不同，DSpark **在预训练之后单独训练**，
#     后训练阶段跟随主干但**不回传梯度**。
V41_DSPARK = {
    "blocks": 3, "window": 128, "draft_positions": 5,
    "heads": ("base logits", "轻量 Markov 头", "置信度头"),
    "trained": "预训练之后单独训练；后训练跟随主干但不回传梯度",
    "contrast": "V3 的 MTP 全程与主干联合训练",
}

# 来源：【C】同上 §3.2「Inference System」与 §3.2.1 / §3.2.2 逐字：
#   · 融合后，绝大多数 Transformer 层（CSA2 处于 Reuse Mode 的那些）
#     **prefill 只执行 15 个 kernel、decode 只执行 11 个**；
#   · 部署用 **EPD（Encoder–Prefill–Decode）分离**；
#   · 持久化 KV 的 1/8 = 「不再存 SWA KV（几乎减半）」× 「全局 KV 再压到 1/4」；
#   · SWA KV 改放**每机 10% 主机 DRAM** 组成的分布式内存池，**TTL 只有几分钟**；
#     全局 KV 留在持久化缓存，**保证至少 72 小时**生命期；
#   · 精确重建 L 层 SWA 需要回放 **L × n_win** 个 token；V4 报告的 Zero SWA Caching
#     在生产中被证明代价过高；V4.1 改为**只回放最近 n_win 个 token**并截断 SWA。
V41_SERVING = {
    "reuse_layer_kernels": {"prefill": 15, "decode": 11},
    "disaggregation": "EPD（Encoder / Prefill / Decode）",
    "persistent_kv_factors": ("不再持久化 SWA KV（几乎减半）", "全局 KV 压缩到 1/4"),
    "swa_kv_pool": "每机 10% 主机 DRAM 的分布式内存池，TTL 几分钟",
    "global_kv_ttl": ">= 72 小时",
    "exact_swa_replay_tokens": "L × n_win",
    "bounded_replay_tokens": "n_win",
}

# 来源：【C】同上 §5.1.1「Reasoning Efforts」与附录 C：推理档位是**写进系统提示的
#   1–100 标量**，同档多个采样构成子组、组内做奖励中心化，**长度惩罚系数随 effort
#   指数衰减（每增加一个特征尺度乘 1/e）**；API 暴露的 max / high / low 对应
#   **b = 100 / 75 / 50**。
V41_EFFORT = {
    "scale": (1, 100),
    "api_levels": {"max": 100, "high": 75, "low": 50},
    "length_penalty": "随 effort 指数衰减，每增加一个特征尺度乘 1/e",
    "gain_25_to_100": {"avg8_reasoning_pass1": (67.1, 76.3),
                       "DeepSWE v1.1": (66.0, 74.2),
                       "Terminal-Bench 2.1": (82.4, 90.6),
                       "output_tokens": "约 2.5×"},
}

# 来源：【C】同上 §5.3.2 Table 3 的对标列（**引用值**，本仓库不复核其测量口径）。
#   ⚠ 表里的对手是 2026-09 的快照（Opus-5 / GPT-5.6 Sol / Kimi K3 / GLM-5.3）。
V41_BENCH_COLUMNS = ("Opus-5 Max", "GPT-5.6 Sol Max", "K3 Max", "GLM-5.3 Max",
                     "DS-V4-Pro Max", "DS-V4-Flash Max", "DS-V4.1-Flash Max")

# 来源：【B】V4.1-Flash 发布页逐字：
#   · 输入激活 8B、输出激活 16B（与报告一致）；
#   · 相对上一代：KV cache 只要 **1/4 的 HBM**、**1/8 的 SSD**；
#   · 多家测试认为 V4.1-Flash 在性能、成本、速度与总运行时间上**超过 V4-Pro**；
#   · 新价格 2026-09-10 04:00 UTC 生效，**off-peak 是 peak 的 50%**。
V41_RELEASE = {
    "kv_hbm_fraction": 0.25, "kv_ssd_fraction": 0.125,
    "outperforms_v4_pro": True, "off_peak_ratio": 0.5,
}

# 来源：【B】同上发布页的「Supporting open source」段与 36kr/机器之心的解读（【D】）：
#   DeepSeek 同时开源了三个仓库——
#     · **deepseek-recipe**：把不同格式的 API 请求统一成 Conversation 格式的 Rust 库 + Python 绑定；
#     · **DeepJIT**：header-only 的 C++20 JIT 运行时，面向 CUDA GPU 与华为昇腾 NPU；
#     · **DeepSelect**：DSA / V4 / V4.1 所用 **TopK 内核**与采样器的高性能实现，
#       相对原生 `torch.topk` 有 **2–20×** 加速（**此倍数来自【D】级解读，本仓库未复现**）。
DEEPSEEK_OSS_REPOS = {
    "deepseek-recipe": "请求格式 → Conversation 的统一层（Rust + Python bindings）",
    "DeepJIT": "header-only C++20 JIT 运行时（CUDA / 昇腾）",
    "DeepSelect": "TopK 内核与采样器（2–20× vs torch.topk，【D】级转述）",
}

# ---------------------------------------------------------------------------
# 术语
# ---------------------------------------------------------------------------

TERMS = {
    "MLA": "把 K/V 压进低秩 latent 的注意力；MLA 在 DeepSeek-V2 首次提出",
    "DSA": "DeepSeek Sparse Attention：lightning indexer 选 top-k，再只对子集做注意力",
    "CSA / HCA": "V4 的两种压缩注意力：CSA 先按 m 个 token 压一条再做 DSA；HCA 压得更狠但保持稠密",
    "CSA2": "V4.1 的 CSA 升级版：**跨层**复用 main KV / indexer K / Top-K 索引，去掉重叠与绝对位置编码",
    "CED": "Causal Encoder–Decoder：decoder 的全局 KV 由第 L/2 层的隐藏状态投影得到，prefill 只跑前半",
    "Full / Reindex / Reuse": "CSA2 的三种静态模式：自己算 KV+索引 / 复用 KV 但重打分 / KV 与索引全复用",
    "层级稀疏索引器": "decoder 里第一个 Full 层建候选池（block 级），后续 Reindex 层只在池内打分",
    "SWA Bounded Replay": "只回放最近 n_win 个 token 来近似重建 SWA KV，接受近似换取缓存不再持久化 SWA",
    "mHC": "Manifold-Constrained Hyper-Connections：把残差扩成 n 路并约束到流形上",
    "Engram": "条件记忆模块：N-gram 哈希做 O(1) 静态知识检索，把记忆卸载到 DRAM",
    "DSpark": "V4.1 的投机解码模块：半自回归草稿 + 置信度调度验证长度",
    "effort": "写进系统提示的 1–100 标量，同一权重在成本-质量曲线上滑动",
    "口径": "同一个量在不同来源下的计数约定；本主题里 Pro / Flash 的 expert 数、27% 与 10% 的适用范围都是口径问题",
}

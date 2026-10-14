"""160-model/kimi 的外部规格常量源。

本文件只放两类东西：
  1. **从一手来源抄来的**外部规格（Kimi K2 / K3 技术报告的表格与公式、
     Kimi Linear 论文页、官方博客），每条带一行 `# 来源：`；
  2. **口径注记**——同一个量在不同来源下不是同一个数的，逐处写明。

★ 外部来源分级（CONTRIBUTING §2，A–E）在本文件的记法：
    【C】论文 / 官方技术报告（**arXiv 预印本，未确认同行评审**）
    【B】产品官方文档（kimi.ai 官方博客、平台文档、官方模型卡）
    【D】社区项目文档（个人站点、财经/科技媒体转述）
  本域没有把模型规格归入【A】的依据：A 级是**硬件厂商规范**，模型规格不属于该级。

★ 核查范围：本文件逐条核的载体是——
  · Kimi K2 技术报告 arXiv:2507.20534v2 的正文与 Table 1 / Table 2（本机 curl 取回 HTML
    后转文本，2026-10-13）；
  · Kimi K3 技术报告 arXiv:2607.24653v2 的正文与 Table 1（同上）；
  · kimi.ai 官方博客 `kimi.ai/blog/kimi-k3`（本机 WebFetch 取回，2026-10-13）。

UNSOURCED / UNRESOLVED（正文不使用或必须标注）：
  · **Kimi K3 的预训练总 token 数**：报告 §3.3 只给了学习率调度、warmup 比例与
    上下文长度课程（8k → 64k → 256k → 1M），**没有给 token 总数**。
    社区流传的数字本仓库未定位到一手来源，故**不落常量**。
  · **Kimi K2.6 的规格与日期**：本次核查未取到可靠一手来源，**不落常量**。
    正文只把 K2.6 / K2.7 作为「存在但未核实」的条目提及。
  · **Kimi k1.5 的 arXiv 编号**：本次核查未取到 abs 页，故**不落编号**，
    只以【B】级的模型名与时间点进入时间线。
  · **K3 的 69 KDA + 24 Gated MLA 层**：Table 1 写的是「69 KDA + 24 MLA」，
    与「93 层、每块 3 KDA + 1 MLA」在算术上一致（69+24 = 93）；
    但「每块 4 层 × 多少块」与末尾额外那层的排布，报告正文与表格的细节需要对照读。
  · **Effective context / decay 的具体 α 取值**：报告给了 KDA 的公式与
    「lower-bounded decay」的设计，**没有给 α 的数值**；编号程序 02 只做
    解析推导（几何衰减的有效长度 = 1/(1−α)），**不声称官方用了某个 α**。
"""

# ---------------------------------------------------------------------------
# 版本锚点（本机）
# ---------------------------------------------------------------------------

MEASURE_HOST = "Apple M1 Pro（10 核）/ macOS / **CPU-only，无 GPU 数字**"
MEASURE_PYTHON = "3.12.9 (Anaconda, Clang 14.0.6)"
MEASURE_NUMPY = "2.3.4"
PROGRAMS_ARE_DERIVATIONS = True

# ---------------------------------------------------------------------------
# 时间线（按来源等级分列）
# ---------------------------------------------------------------------------

# 来源：【B】官方博客 `kimi.ai/blog/kimi-k3` 与【D】媒体转述。⚠ 带 `【D】` 的条目
#   本仓库未取到官方发布页，只作谱系参考。
KIMI_TIMELINE = (
    ("2025-01", "Kimi k1.5", "以 RL 激发长链式推理；多模态；128K（【B】仅模型名与时间点）"),
    ("2025-07", "Kimi K2", "1.04T 总 / 32.6B 激活；MuonClip；Agentic（【C】arXiv:2507.20534）"),
    ("2025-10", "Kimi Linear", "KDA（Kimi Delta Attention）首次提出；48B-A3B 开源（【B】官方模型卡）"),
    ("2025-11", "Kimi K2 Thinking", "K2 的思考版（【C】arXiv:2511.22891，本次未逐字读）"),
    ("2026-01", "Kimi K2.5", "开源；接 MoonViT 视觉（【D】媒体转述，未取到官方页）"),
    ("2026-07", "Kimi K3", "2.78T 总 / 104.2B 激活；KDA + AttnRes；1M；原生多模态（【C】arXiv:2607.24653v2）"),
)

# ---------------------------------------------------------------------------
# 【C】Kimi K2 —— 超稀疏 MoE + MuonClip
# ---------------------------------------------------------------------------

# 来源：【C】arXiv:2507.20534v2 Table 2 逐字（该表对比 DeepSeek-V3 与 Kimi K2）：
#   #Layers 61、Total 1.04T、Activated 32.6B、Experts 384、Experts Active 8、
#   Shared Experts 1、Attention Heads 64、Number of Dense Layers 1、
#   Expert hidden 2048、MLA、SwiGLU、词表 160K。
#   另有正文：「to reduce computational overhead during inference, we cut the number
#   of attention heads to 64, as opposed to 128 in DeepSeek-V3」。
K2_CONFIG = {
    "layers": 61,
    "total_params": 1.04e12,
    "active_params": 32.6e9,
    "hidden_dim": 7168,
    "experts_total": 384,
    "routed_experts": 8,
    "shared_experts": 1,
    "expert_intermediate": 2048,
    "attn_heads": 64,
    "dense_layers": 1,
    "attention": "MLA",
    "activation": "SwiGLU",
    "vocab_size": 160000,
    "mtp_layers": 1,
    "training_context": 131072,
}
# 来源：【C】同上 §2.5 Training recipe 逐字：
#   · 4 096 token 上下文预训练，**MuonClip** 优化器，**WSD** 学习率调度，
#     共 **15.5T** token；
#   · 前 **10T** token 常数 lr **2e-4**（500 步 warmup），后 **5.5T** token
#     cosine 从 2e-4 衰减到 2e-5；
#   · weight decay 0.1，全局 batch **67M** token；
#   · 预训练末段有 annealing + long-context activation 阶段。
K2_TRAINING = {
    "optimizer": "MuonClip",
    "schedule": "WSD",
    "tokens": 15.5e12,
    "const_lr_tokens": 10e12,
    "const_lr": 2e-4,
    "warmup_steps": 500,
    "decay_tokens": 5.5e12,
    "decay_to": 2e-5,
    "weight_decay": 0.1,
    "global_batch_tokens": 67e6,
    "pretrain_context": 4096,
}
# 来源：【C】同上 §2.3 逐字：稀疏度（sparsity）= 总专家数 / 每 token 激活专家数。
K2_SPARSITY = 384 / 8      # = 48

# ---------------------------------------------------------------------------
# 【B】Kimi Linear —— KDA 的提出处
# ---------------------------------------------------------------------------

# 来源：【B】官方模型卡 `moonshotai/Kimi-Linear-48B-A3B-Instruct`（本机经检索条目确认存在）
#   与【C】arXiv:2510.26692 *Kimi Linear: An Expressive, Efficient Attention Architecture*。
#   ⚠ 本仓库**未逐字读 Kimi Linear 论文正文**，只使用两条信息：
#     (a) KDA 由该工作提出；(b) 开源型号是 48B-A3B。
KIMI_LINEAR = {
    "paper": "arXiv:2510.26692",
    "model": "Kimi-Linear-48B-A3B-Instruct",
    "contribution": "Kimi Delta Attention (KDA)",
    "read_verbatim": False,
}

# ---------------------------------------------------------------------------
# 【C】Kimi K3 —— 本主题的实例
# ---------------------------------------------------------------------------

# 来源：【C】arXiv:2607.24653v2 **Table 1** 逐字（「Architectural comparison between
#   Kimi K2 and Kimi K3」，本机读）：
#     #Layers 61 → 93（+52%）；Total 1.04T → 2.78T（+167%）；
#     Activated 32.6B → 104.2B（+220%）；Hidden 7 168 → 7 168（不变）；
#     Latent MoE Dim — → 3 584（0.5×）；MoE Hidden per Expert 2 048 → 3 072（+50%）；
#     Routed Experts 384 → 896（+133%）；Experts Active 8 → 16（+100%）；
#     Shared Experts 1 → 2（+100%）；Attention Heads 64 → 96（+50%）；
#     Dense Layers 1 → 1；Vocab 160K → 160K；Training Context 128K → 1M（8×）；
#     Attention MLA → **Hybrid KDA–MLA**；Activation SwiGLU → **SiTU-GLU**；
#     Attention-Layer Composition 61 MLA → **69 KDA + 24 MLA**；MTP 1 → 1；
#     ViT 401M / 27 层 / patch 14 / 12 头。
K3_CONFIG = {
    "layers": 93,
    "total_params": 2.78e12,
    "active_params": 104.2e9,
    "hidden_dim": 7168,
    "latent_moe_dim": 3584,
    "latent_ratio": 0.5,
    "expert_intermediate": 3072,
    "experts_total": 896,
    "routed_experts": 16,
    "shared_experts": 2,
    "attn_heads": 96,
    "dense_layers": 1,
    "vocab_size": 160000,
    "mtp_layers": 1,
    "training_context": 1048576,
    "kda_layers": 69,
    "mla_layers": 24,
    "vit_params": 401e6,
    "vit_layers": 27,
    "vit_patch": 14,
    "vit_heads": 12,
}
# 来源：【C】同上 §2 与 §2.3 逐字：
#   · 「each block contains three KDA layers followed by one Gated MLA layer」
#     → **3:1** 混合比例；**末尾额外放一个 Gated MLA 层**，保证最后一层总是全局注意力；
#   · 「896 routed experts with 16 active experts per token, corresponding to a
#     sparsity of **56**」。
K3_HYBRID = {"pattern": "3 × KDA + 1 × Gated MLA", "extra_final": "额外的 Gated MLA"}
K3_SPARSITY = 896 / 16     # = 56

# 来源：【C】同上 §2.1.1 式 (1) 逐字——KDA 的递推：
#   S_t = (I − β_t k_t k_tᵀ) Diag(α_t) S_{t−1} + β_t k_t v_tᵀ,  õ_t = S_tᵀ q_t
#   其中 α_t ∈ (0,1)^{d_k} 是**逐通道**的一步保留因子，β_t ∈ (0,1) 控制
#   delta-rule 的写入强度；q/k 走 ShortConv → Swish → L2Norm，v 走 ShortConv → Swish，
#   β 走 Sigmoid，α 由一个低秩投影 + **每头偏置**给出（log 空间、有下界）。
K3_KDA_FORMULA = "S_t = (I − β_t k_t k_tᵀ) Diag(α_t) S_{t−1} + β_t k_t v_tᵀ,  õ_t = S_tᵀ q_t"
K3_KDA_DETAILS = {
    "alpha": "逐通道一步保留因子 α_t = exp(g_t) ∈ (e^{g_min}, 1)",
    "beta": "delta-rule 写入强度 β_t ∈ (0,1)，由 Sigmoid 给出",
    "qk_path": "ShortConv → Swish → L2Norm",
    "v_path": "ShortConv → Swish",
    "alpha_path": "低秩投影 + 每头偏置给出 decay logit z，再经**缩放的 Sigmoid**映射成 log-decay",
    "lower_bounded": "g = g_min · Sigmoid(e^{A_h} z) ∈ (g_min, 0)；**K3 相对 Kimi Linear 的改动**"
                     "（Kimi Linear 用 negative-Softplus，下界是 −∞），把 log-decay 从下方夹住",
    "numerical": "Kimi Linear 用**对数空间**算相对衰减并把 chunk 再切成 **16-token 的次级 tile**："
                 "非对角 tile 可用 Tensor Core 的稠密矩阵乘；对角 tile 仍需显式位置对计算，"
                 "是 chunk 内的主要瓶颈",
    "output_gate": "K3 把输出门从 Kimi Linear 的**低秩**参数化换成**输入相关的满秩**投影："
                   "y = W_o[ Sigmoid(W_g x) ⊙ RMSNorm(õ) ]",
    "parallel_form": "chunk 内并行、chunk 间递推（UT transform）",
}
# 来源：【C】同上 §2.1.2 逐字：K3 的 MLA 层用 **NoPE**（不做显式位置编码），
#   因为位置信息由 KDA 的递推门控与衰减隐式承载——因此**不需要任何 RoPE
#   缩放或插值**就能外推到 1M。
K3_POSITION = {
    "mla": "NoPE（无显式位置编码）",
    "implicit": "KDA 的递推门控与衰减",
    "extrapolation": "无需 RoPE rescaling / interpolation",
}
# 来源：【C】同上 §2.2 逐字（Attention Residuals）：
#   · 动机：「standard residual connections compress all prior information into a
#     single state over depth — a bottleneck reminiscent of RNNs over time」；
#   · Full 形式：每层一个可学伪 query w_l，对 embedding 与**所有前层输出**做
#     softmax 注意力（打分用 exp(qᵀ RMSNorm(k))），O(L²d) 算力、**O(Ld) 显存**；
#   · Block 形式：把 L 层分成 N 块各 S = L/N 层，块内输出**求和**成一个表示，
#     块间才做完整注意力；显存与通信从 O(Ld) 降到 O(Nd)。
K3_ATTNRES = {
    "motivation": "残差把深度方向压缩成一个状态，类似 RNN 在时间方向上的瓶颈",
    "full": {"compute": "O(L²d)", "memory": "O(Ld)", "score": "exp(qᵀ RMSNorm(k))"},
    "block": {"blocks": "N", "size": "S = L/N", "memory": "O(Nd)"},
}
# 来源：【C】同上 §2.3 逐字（Stable LatentMoE）：
#   · LatentMoE：shared expert 走**满宽** d，routed expert 在**压缩 latent** ℓ 上工作
#     → 让「896 专家 × 每 token 激活 16 个」在通信与权重流量上变得可承受；
#   · 三个稳定化组件：up-projection 前插 **RMSNorm**（Normalized LatentMoE）、
#     **SiTU-GLU**（Sigmoid Tanh Unit GLU）抑制激活爆炸、
#     **Quantile Balancing (QB)** 做负载均衡（近 10³ 个专家的规模超出
#     auxiliary-loss-free 偏置更新的适用区）；
#   · shared expert 数固定为 **N_s = 2**。
K3_LATENTMOE = {
    "shared": "满宽 d 路径（N_s = 2）",
    "routed": "在 latent 宽 ℓ = 3584 上工作",
    "stabilizers": ("up-projection 前 RMSNorm", "SiTU-GLU", "Quantile Balancing"),
    "failure_modes": ("routed 路径上近四连乘导致激活爆炸", "近千专家的负载均衡超出免辅助损失偏置的适用区"),
}
# 来源：【C】同上 §2.5 逐字（Per-Head Muon）：把 Q/K/V 投影矩阵的动量矩阵**沿头维切块**，
#   每块单独做 Newton–Schulz 正交化；理由是整矩阵正交化会把所有头当一个耦合块，
#   尺度大的头主导共享更新方向。
K3_OPTIMIZER = {
    "optimizer": "Per-Head Muon",
    "clip": "沿用 Kimi K2 的 weight clipping",
    "balance": "Quantile Balancing",
    "schedule": "cosine，1% linear warmup，weight decay 0.1",
}
# 来源：【C】同上 §3.3 / §3.4 逐字：
#   · **原生多模态**：语言与视觉从训练一开始联合优化，视觉/文本 token 交错在
#     同一个 next-token prediction 目标里；
#   · 预训练上下文 **8k 起步，后扩到 64k**；cooldown 阶段 **256K → 1M**（四阶段课程）；
#   · 长上下文数据：去重 + 感知哈希（视频帧）+ 质量过滤，并对长文档/视频**上采样**；
#     另外**合成**长上下文数据（重排拼接多模态文档与子任务，使任务只能靠
#     跨 1M 上下文的分散信息解出）。
K3_TRAINING = {
    "multimodal": "native（语言与视觉从一开始联合优化）",
    "context_curriculum": ("pretrain 8k → 64k", "cooldown 256K → 1M"),
    "scaling_efficiency_vs_k2": 2.5,
    "scaling_efficiency_note": "官方口径：held-out OOD 验证上的整体 scaling 效率提升约 2.5×；本仓库不复核",
}
# 来源：【C】同上 §4.1.4 逐字（Deployment-Aware Post-Training）：
#   · MoE **expert 权重**量化到 **MXFP4**，**激活算在 MXFP8**；
#     所有非 expert 组件（注意力投影、latent MoE 投影、shared expert、MoE 路由）
#     保持更高精度；
#   · **QAT 覆盖整个后训练阶段**（SFT 与 RL 都做）；
#   · RL 时 **rollout 与 training 共用同一套量化方案**，消除训练-推理不一致；
#   · 另有 Draft Model Fine-Tuning（投机解码的草稿模型微调）。
K3_MXFP4_QAT = {
    "expert_weights": "MXFP4",
    "activations": "MXFP8",
    "non_expert": "更高精度（注意力投影 / latent MoE 投影 / shared expert / 路由）",
    "scope": "整个后训练阶段（SFT + RL）",
    "rollout_consistency": "rollout 与 training 共用同一量化方案",
}

# 来源：【B】官方博客 `kimi.ai/blog/kimi-k3` 逐字：
#   · 「a 2.8T-parameter model built on our Kimi Delta Attention and Attention
#     Residuals, with native vision capabilities and a 1-million-token context
#     window. It is the world's first open 3T-class model」；
#   · 发布即用 **max thinking effort**，low / high 后续补；
#   · **完整权重在 2026-07-27 前放出**；
#   · 「its overall performance still trails ... Claude Fable 5 and GPT 5.6 Sol」。
K3_RELEASE = {
    "launch_effort": "max",
    "effort_modes_added_later": ("low", "high"),
    "weights_due": "2026-07-27",
    "trails": ("Claude Fable 5", "GPT 5.6 Sol"),
}

# ---------------------------------------------------------------------------
# 术语
# ---------------------------------------------------------------------------

TERMS = {
    "KDA": "Kimi Delta Attention：在 delta-rule 递推上加逐通道 forget gate 的线性注意力",
    "linear attention": "用固定大小的递推状态替代随长度增长的 KV cache；显存 O(1)、算力 O(L)",
    "Gated MLA": "带门控的 MLA 全局注意力层；K3 每 4 层放一层",
    "delta rule": "把「写入」做成对当前状态的修正（I − βkkᵀ）而不是简单相加",
    "AttnRes": "Attention Residuals：用注意力（可学伪 query）替代逐层累加的残差",
    "LatentMoE": "routed expert 在压缩 latent 上工作，shared expert 走满宽",
    "SiTU-GLU": "Sigmoid Tanh Unit GLU，用来抑制超深 routed 路径上的激活爆炸",
    "Quantile Balancing": "用分位数而非固定步长偏置来做 MoE 负载均衡",
    "MXFP4": "OCP 微缩放格式的一种：E2M1 元素 + 每 32 个元素一个 E8M0 scale",
    "NoPE": "不放显式位置编码；K3 靠 KDA 的衰减隐式编码位置",
    "sparsity": "官方口径：总专家数 / 每 token 激活专家数（K3 为 896/16 = 56）",
}

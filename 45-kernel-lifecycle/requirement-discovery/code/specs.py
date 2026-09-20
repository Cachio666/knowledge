"""45-kernel-lifecycle/requirement-discovery 的外部事实常量源。

本模块是 CONTRIBUTING §2 约定的**共享事实源**（`specs.py`）：三条编号程序从它读常量，
推导过程可复核，**常量本身不可自证**。

★ 本主题与 `20-gpu-architecture/tensor-core`、`40-kernels/*` 的形态相同，但多了一层
  本库从未有过的要求：**每条外部常量必须标注来源等级**（域 README 的 A–E 五级）。
  等级的判据见 `45-kernel-lifecycle/README.md`；本模块用 `[级]` 后缀写在 `# 来源：` 行末。

★ 本主题的常量里**有一大批是 D 级**（个人或小团队仓库的工程经验）。域 README 的硬规则：
  D 级数字在正文里**必须写明它是 D 级**，且**不得单独支撑一条论断**。
  因此本模块把它们集中放在 §5–§7，并在注释里写死「未被独立复现」这类限定语。

★ **口径注释与 `# 来源：` 同为必填**。写不出「什么条件下测的」的，按 §0.6 硬规则 4
  **不写进来**——本模块的 §9 `UNSOURCED` 记录这些被挡在门外的项。

★ **不得「既保留数字又记 UNSOURCED」**（§0.6 硬规则 5）：二者只能取其一。
"""

# ===========================================================================
# 1. 本库已有的可复用常量（**照抄口径，不重推**）
# ===========================================================================
#
# 本节全部常量从本库其他域**逐字复制**，本主题不重新溯源，也**不改任何一位数字**。
# 复制时保留原来的 `# 来源：` 行与口径注释——因为**口径是常量的一部分**。
#
# ⚠ 复制造成的**漂移风险**：这些值在仓库里有多份副本（见每条的「同值位置」）。
#   按 §0.7.A 的教训，**「口径一致」只在交集上成立**——下面 `softmax-normalization`
#   那一份就缺 `V100` 键（本模块已逐处核对，见 §2 的注）。

FLOAT16_BYTES = 2
FLOAT32_BYTES = 4
# 来源：IEEE 754 binary16 / binary32 的宽度（规范，[A]）
# 口径：**存储宽度**，不是「有效位」。fp16 的 1+5+10 与 fp32 的 1+8+23 是位域，
#       2024/2025 年的数据中心 GPU 上**没有 fp16 存储的例外**（fp8 才有）。

SM_COUNT_BY_GEN = {
    "V100": 80,
    "A100": 108,
    "H100": 132,
    "B200": 148,
}
# 来源：NVIDIA Volta V100 / A100 / Hopper 官方材料（[A]）；
#       B200 的 148 **非官方直述**，由官方 FP32 峰值 75 TFLOPS 反推 + Hot Chips 2024 交叉验证
# 口径：**数据中心 SXM 档**，不含 PCIe 档
# 同值位置：40-kernels/{gemm,attention,elementwise-fusion,moe-kernels,reduce-scan}/code/specs.py

HBM_BW_GBps_BY_GEN = {
    "V100": 900.0,
    "A100": 2039.0,
    "H100": 3350.0,
    "B200": 7700.0,
}
# 来源：NVIDIA Volta V100 Whitepaper（HBM2，900 GB/s）/ A100 Whitepaper 表4（80GB HBM2e）/
#       Hopper Architecture In-Depth 表3（HBM3，3.35 TB/s）/ Blackwell Technical Brief V2.1（HBM3e）
#       —— 全部 [A]
# 口径：**十进制 10^9 B/s**、**SXM 档**。**不得混入 PCIe 档**（同代 PCIe 的带宽不同）。
#       ⚠ 这是**标称**带宽（datasheet 值）；**可达**带宽低于它，缺口见 §3。
# 同值位置：★ 7 处 —— 20-gpu-architecture/tensor-core/code/specs.py:68、
#       40-kernels/{attention:111, elementwise-fusion:131, gemm:91, moe-kernels:81,
#       reduce-scan:178, softmax-normalization:81}/code/specs.py。
#       ⚠ **`softmax-normalization` 那一份没有 `V100` 键**（只有 A100/H100/B200）——
#         「口径一致」只在**交集**上成立；去那一份里取 V100 会 `KeyError`。
#         本模块取 7 份的**并集**（V100..B200 四代），并在此注明这是**并集**而非某一处。

FP32_VECTOR_TFLOPS_DENSE = {
    "V100": 15.7,
    "A100": 19.5,
    "H100": 67.0,
    "B200": 75.0,
}
# 来源：NVIDIA Volta V100 Whitepaper / A100 Whitepaper 表1 / Hopper Architecture In-Depth 表3
#       与 H100 datasheet / Blackwell Technical Brief V2.1 —— 全部 [A]
# 口径：**非 tensor core** 的 FP32 向量峰值（FFMA 路径）。逐元素算子走的就是这条通路。
# 同值位置：**只在 6 个 `40-kernels/*/code/specs.py` 里**（attention:125、elementwise-fusion:142、
#       gemm:99、moe-kernels:89、reduce-scan:186、softmax-normalization:88）。
#       ⚠ **`20-gpu-architecture/tensor-core/code/specs.py` 里没有这个名字**——那里叫
#         `TFLOPS_FP32_VECTOR_DENSE`（第 314 行），值相同。**按名字搜会漏掉那一份。**

TC_TFLOPS_DENSE_SPARSE = {
    "TF32": {"A100": (156.0, 312.0), "H100": (494.5, 989.0), "B200": (1100.0, 2200.0)},
    "FP16": {"V100": (125.0, None), "A100": (312.0, 624.0),
             "H100": (989.5, 1979.0), "B200": (2250.0, 4500.0)},
    "BF16": {"A100": (312.0, 624.0), "H100": (989.5, 1979.0), "B200": (2250.0, 4500.0)},
    "FP8":  {"H100": (1979.0, 3958.0), "B200": (4500.0, 9000.0)},
    "FP4":  {"B200": (9000.0, 18000.0)},
}
# 来源：NVIDIA 各代官方材料（[A]）
# 口径：**dense / sparse 二元组**，单位 TFLOPS。注释逐字警告：
#       「NVIDIA 产品页写的 H100 FP16 1979 TFLOPS **是带 2:4 稀疏的**，dense 只有 989.5」。
#       V100 的 sparse 位是 `None`——Volta 没有 2:4 结构化稀疏，**不是「未采集」**。
# 同值位置：★ **4 处**——20-gpu-architecture/tensor-core/code/specs.py:323、
#       40-kernels/{gemm:107, attention:134, moe-kernels:98}/code/specs.py。
#       ⚠ 本主题只取其中的 `FP16` 行；其余行**照抄但不使用**，为的是保持与上游逐字一致。

ACHIEVABLE_BW_SHORTFALL_PCT_IN_REPO = (5, 15)
# 来源：本仓库 10-foundations/performance-models/03-measuring-roofline §0——
#       「真实的**可达**峰值低于规格值——boost clock 漂移、占用率不足、ECC 开销都会扣掉
#        几个到十几个百分点」  [E]（本库已入库的口径）
# 口径：**必须连同它的自陈一起搬**——该注释写明「**本仓库对它的原始出处没有给出一手页面**，
#       引用值，本仓库不复核其测量口径」。
#       它对应「可达带宽系数 η」：η = 1 − 缺口，即 η ∈ [0.85, 0.95]。
#       η = 1（标称）只作为**乐观下界**出现，不是可达值。
# 同值位置：40-kernels/elementwise-fusion/code/specs.py:165

L2_MB_H100_IN_REPO = 50
# 来源：本仓库 10-foundations/performance-models/09_roofline_model 与
#       performance-models/03-measuring-roofline §2 的规格表  [E]（本库已入库的口径）
# 口径：⚠ **同样要搬自陈**——注释写明「**本仓库对 L2 容量这一项内部不一致**」：
#       `20-gpu-architecture/architecture-evolution/code/specs.py` 只把 **B200** 的 L2
#       列为 UNSOURCED，并未给 H100 的值。
#       **沿用它时必须保留这句限定**，且只用于量级判断，不做精确结论。
# 同值位置：40-kernels/elementwise-fusion/code/specs.py:174、moe-kernels/code/specs.py:140

KERNEL_LAUNCH_OVERHEAD_US_EXTERNAL = (3.0, 5.0)
# 来源：外部对 CUDA 空 kernel 启动开销的对照测量（**非 NVIDIA 官方**）  [D]
# 口径：**引用值，本仓库不复核其测量口径**。量级是「一次 kernel 启动数微秒」，
#       随驱动版本、是否用 CUDA Graph、是否走 MPS 而变。只用作量级对照。
#       ★ 本主题**只用它做一件事**：把「3.2% 的占比」换算成绝对时间量级，
#         从而判断它是不是落在噪声地板下——**不用它做任何端到端结论**。
# 同值位置：40-kernels/{gemm:198, attention:256, elementwise-fusion:182, moe-kernels:149,
#       reduce-scan:198}/code/specs.py


# ===========================================================================
# 2. 模型规格（**B 级：HuggingFace `config.json`，不是论文也不是实测**）
# ===========================================================================
#
# ★ 这是本主题的**输入**：01 篇的整个算子清单是从这些字段推出来的。
#   所以每条字段的来源必须能指到 `config.json`，**不能凭记忆写**。
#
# ⚠ 两处**已知缺口**，本模块显式记录、**不猜测填充**：
#   ① `Qwen3-30B-A3B` 的**共享专家**（shared expert）规格字段在检索到的片段里**没有**
#      （检索只给出 `intermediate_size: 6144`，而它的语义是「非 MoE 的 FFN 宽度」，
#      **不等于共享专家的宽度**）。因此 01 篇的清单**不含共享专家**，
#      算出的活跃参数量是**下界**。见 §9 `UNSOURCED`。
#   ② `tie_word_embeddings` 未核实。本模块的参数量估计**按不共享**（embed + lm_head
#      各算一次）——若实际共享，总数会少 3.1e8，**偏差 1%**，不影响任何结论。

LLAMA3_8B_CONFIG = {
    # 来源：meta-llama/Meta-Llama-3-8B 的 config.json（经 HuggingFace 与 torchtune 的
    #       llama3_8b() 构造器双向核对）  [B]
    "name": "Meta-Llama-3-8B",
    "num_hidden_layers": 32,
    "hidden_size": 4096,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "intermediate_size": 14336,
    "vocab_size": 128256,
    "num_experts": None,
    "num_experts_per_tok": None,
    "moe_intermediate_size": None,
}
# 口径：`head_dim` 单列（= 128），**不要用 `hidden_size // num_attention_heads` 代替**——
#       本例里两者恰好都等于 128，但 Qwen3-30B-A3B 上前者是 128、后者是 2048/32 = 64。
#       **这是本主题最容易算错的一个字段**（01 篇 §4 的坑 ①）。

QWEN3_30B_A3B_CONFIG = {
    # 来源：Qwen/Qwen3-30B-A3B 的 config.json（经 4 处独立转抄核对：b70-optimization-lab、
    #       radicalnumerics/RND1-Base-0910 的默认值、thu-pacman/chitu 的 YAML、
    #       sysHAX-adapter 的 CPU 默认配置）  [B]
    "name": "Qwen3-30B-A3B",
    "num_hidden_layers": 48,
    "hidden_size": 2048,
    "num_attention_heads": 32,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "intermediate_size": 6144,
    "vocab_size": 151936,
    "num_experts": 128,
    "num_experts_per_tok": 8,
    "moe_intermediate_size": 768,
}
# 口径：`num_key_value_heads` = 4 ⇒ **GQA 8:1**（32 个 Q 头对 4 个 KV 头）。
#       这直接决定 k_proj / v_proj 的宽度是 `4 × 128 = 512` 而不是 `32 × 128 = 4096`，
#       漏掉它会把这两个算子的 FLOPs **高估 8 倍**。

PUBLISHED_PARAM_COUNT_B = {
    # 来源：Meta-Llama-3-8B 的公开参数量 ≈ 8.03 B（[B]）；
    #       Qwen3-30B-A3B 的总参数量 ≈ 30.5 B、每 token 活跃 ≈ 3 B（[B]）
    "Meta-Llama-3-8B": 8.03,
    "Qwen3-30B-A3B": 30.5,
}
# 口径：**总参数量**（含嵌入、含全部专家）。Qwen3 的「≈3 B 活跃」是**不含共享专家**的
#       口径。★ 本主题用 Llama 的 8.03 B 做 **01 篇 A 类的独立对照**（从 config 反算的
#       参数量应当落在 8.03 B 上）；Qwen 的 30.5 B 只用于**量级**核对。

TOKENS_PER_PARAM_RULE_OF_THUMB = None
# 占位：本模块**不写**「每参数 20 token」这类训练口径的经验值。
# 理由：它是**训练**侧的 Chinchilla 结论，本主题是**推理**侧，写进来就是 P9（同名不同指）。
# 本主题在需要时说清「这是训练口径」并回链，不落成常量。


# ===========================================================================
# 3. 从本库常量推出的 ridge point（**E 级：本库推导值**）
# ===========================================================================
#
# ★ 这一节**不是常量**，是「按 §1 的常量算出来的值」。放在这里是为了让**文档与程序引用
#   同一个数**——文档不手抄，程序不重算。
#
# ★ 推导式：ridge = 峰值算力 ÷ 峰值带宽（单位 FLOP/byte）。
#   本库已有的同名函数在 `40-kernels/gemm/code/verify.py:274` 的 `roofline_ridge()`，
#   形式为 `peak_tflops * 1e12 / (bw_gbps * 1e9)`——本主题**沿用其形式**。
#
# ⚠ **本模块不把推导值写成常量**（CONTRIBUTING §2：「只放从来源抄来的数，不放推导值」）。
#   下面这张表**只作为注释里的对照**，程序每次运行时**自己算**，并断言与这张表一致。

RIDGE_EXPECTED = {
    # 值 = TC_TFLOPS_DENSE_SPARSE[prec][gen][0] ÷ HBM_BW_GBps_BY_GEN[gen]，单位 FLOP/byte
    # 本主题的推导结果（python3 3.13.5 + numpy 2.4.4，见 `make run` 的输出）：
    "H100_fp16_tc": 295.3731,   # 989.5e12 ÷ 3350e9
    "A100_fp16_tc": 153.0162,   # 312.0e12 ÷ 2039e9
    "B200_fp16_tc": 292.2078,   # 2250.0e12 ÷ 7700e9
    "V100_fp16_tc": 138.8889,   # 125.0e12 ÷ 900e9
    "H100_fp32_vec": 20.0000,   # 67.0e12 ÷ 3350e9
    "A100_fp32_vec": 9.5635,    # 19.5e12 ÷ 2039e9
    "B200_fp32_vec": 9.7403,    # 75.0e12 ÷ 7700e9
    "V100_fp32_vec": 17.4444,   # 15.7e12 ÷ 900e9
}
# 口径：**标称**峰值与**标称**带宽相除，即**乐观**的拐点。可达拐点更低
#       （带宽的可达值比标称低 5–15%，见 §1 的 ACHIEVABLE_BW_SHORTFALL_PCT_IN_REPO），
#       本文一律用乐观值并在正文注明——用乐观值做判定是**保守**方向：
#       它会把算子判成「计算受限」，而真实情况只会更偏访存。

B200_RIDGE_ALT_BW_TBPS = 8.0
# 来源：外部材料里 B200 的带宽有 **8.0 TB/s** 的说法（[D]）
# 口径：★ **这是口径分歧，不是算力分歧**。用 8.0 而不是本库的 7.7：
#       2250.0e12 ÷ 8000e9 = **281.25**，而本库口径给 **292.21**。
#       两者都不错，差的只是**分母上的带宽**。本主题**一律采用本库的 7.7 TB/s**
#       （`specs.py` 口径一致性强），并在正文注明另一口径的存在。
#       ⚠ 不要把 281 与 292.21 并列成「B200 的 ridge 有两个值」就完事——
#         **必须写出差异的成因**，否则读者会以为算力有争议。


# ===========================================================================
# 4. 引擎的官方做法（**B 级：某一实现的规格，不是「业界」**）
# ===========================================================================
#
# ★ P10 的正面用法：B 级来源只能支撑「**这一家**这么做」。
#   正文必须写成「vLLM 的做法是……」「TensorRT-LLM 的做法是……」，
#   **不得**写成「业界通常」「主流做法」。

VLLM_INTEGRATION_COMPLEXITY_TIERS = (
    # 来源：vLLM 官方文档 “Adding a New Model” / “Adding new models”  [B]
    #       （经检索转述；本实现者未逐字读到页面原文，见 §9）
    ("simple",   "与 vLLM 已有的某个模型结构相似"),
    ("moderate", "标准组件，但结构独特"),
    ("complex",  "含**新算子**——例如新的 attention 机制，此时需要在 vLLM 中"
                 "**实现新的 attention layer / kernel**"),
)
# 口径：**vLLM 一家的分档方式**。P10 警告：写成「业界通常分三档」即为违规。
#       同一份文档还有一条更强的建议：**先跑 `vllm serve <model>`**——许多 decoder 模型
#       可以通过 Transformers 后端直接加载，**根本不需要在 vLLM 里实现**。
#       ⇒ 「要不要写 kernel」这个问题，**先被「能不能直接加载」挡一道**。

VLLM_TRANSFORMERS_BACKEND_HINT = "vllm serve <model>"
# 来源：同上（vLLM 官方文档）  [B]
# 口径：它是**一个先跑再说的动作**，不是一个结论。本主题把它当作判据表的第一行输入。

VLLM_IR_PRIORITY_MECHANISM = {
    # 来源：vLLM 官方文档 `vllm.ir.op` / design/vllm_ir 与 PR #33825  [B]
    # 口径：**算子与实现分离**——`@register_op` 声明语义（带一份 `native` 参考实现），
    #       `register_impl` 注册实现，选择靠**优先级列表**：
    #       「**the first supported implementation is selected**」，先查静态 `supported`，
    #       再查动态 `supports_args`。
    "declaration_decorator": "@register_op",
    "impl_decorator": "register_impl",
    "selection_rule": "优先级列表中第一个 supported 且 supports_args 通过的实现",
    "cli": "--ir-op-priority.<op_name>=<provider1>,<provider2>,...",
    "providers": ("native", "vllm_c", "aiter", "xpu_kernels", "triton_*"),
}
# 口径：★ 这条回答的是本主题 §3 的问题——**「优先级选择」是显式配置项**，不是隐式的。
#       ⚠ `native` 是保留名（原生 torch 实现），**它永远存在**，所以列表永远有兜底。

KIMI_K3_NEW_OPERATORS = (
    # 来源：vLLM 官方博客 “Kimi K3 Is Here: Efficient Day-0 Support on vLLM”
    #       （2026-07-27）与 “A Preview of Production-Scale Kimi K3 Support on vLLM”
    #       （2026-07-22），以及 vLLM v0.28.0 的 release notes  [B]
    # 口径：★ 这是「**一个模型发布确实会带来若干新算子**」的**具体形态**——
    #       本主题要的不是这个列表本身，而是**列表有 4 项而不是 1 项**这个事实。
    ("KDA prefill",
     "FlashKDA：Moonshot AI 的 CUTLASS 实现；v0.28.0 融合进 FlashKDA prefill kernel"
     "（#51311、#52458）"),
    ("KDA decode",
     "把因果卷积 + 循环状态更新 + RMSNorm **折进一次 launch**，就地更新卷积与循环状态"
     "（#50654）——★ 与 prefill 是**两个不同的 kernel**"),
    ("Attention Residuals",
     "Block AttnRes：对至多 8 个缓存的块表示 + 当前块内残差做深度方向 softmax"
     "（至多 9 个来源），单个融合 kernel 完成；Triton 通用路径 + 专用 CUDA 路径"),
    ("MoE 新激活 SiTU",
     "接进 MXFP4 TRTLLM-Gen 与 DeepGEMM 路径；v0.28.0 为 MegaMoE 加 SiTU（#50510）"),
)
KIMI_K3_PREFIX_CACHE_REDESIGN = {
    # 来源：同上（vLLM 官方博客）  [B]
    # 口径：KDA 是**循环**的（推进一个矩阵状循环状态 + 一个短卷积状态，**不存 per-token
    #       KV 对**），所以 vLLM 把三个原本绑在一起的概念**拆开**：
    #       物理块大小 / 调度器对齐 / **前缀匹配单元**；命中部分块时**先拷进私有目的地
    #       再扩展**（copy-on-write）。`VLLM_PREFIX_CACHE_RETENTION_INTERVAL` 提供
    #       区间化保留，把 prompt 边界当检查点。
    "separated_concepts": ("physical block size", "scheduler alignment", "prefix-match unit"),
    "rule": "copy-on-write：命中部分块时把缓存状态拷进私有目的地再扩展",
}
# ★ 这一条是本主题 §0 论断的**最强实例**：「缓存层要重做」不是「加一个 kernel」，
#   而是**三个原本耦合的概念被拆开**。它说明新模型的成本**不止于算子数**。

KIMI_K3_METADATA_BUILDER_LATENCY_US = (870, 34)
# 来源：vLLM 官方博客（同上）  [B]
# 口径：**batch size 1**、专用 KDA metadata builder 把「小 eager PyTorch 算子」换成
#       融合 Triton kernel、每条序列压到一次 launch 之后，metadata 准备延迟
#       **870 µs → 34 µs（−96%）**，端到端 DSpark 延迟改善 **6%**。
#       ★ 这条是本主题**唯一一条来自 B 级来源的「小占比换来可测收益」的数字**。
#       ★ **反解出的迭代长度（[E]，本库推导）**：省下 870 − 34 = **836 µs** 换来端到端
#         **6%** ⇒ 被改善的那段总时长 ≈ 836 ÷ 0.06 ≈ **13.9 ms**。这是一次**自洽性
#         核对**（836/13900 = 6.0%），也是本主题唯一能给出「一个 decode 迭代有多长」
#         量级的**带来源**的推导。它把 metadata 占比钉在 870/13900 ≈ **6.3%**。
#       ⚠ 该反解**假设 870 µs 全在关键路径上**；若不是，13.9 ms 是**上界**。

TENSORRT_LLM_TIMELINE_DECOMPOSITION = {
    # 来源：NVIDIA/TensorRT-LLM PR #18805
    #       “[None][feat] Decompose the nsys timeline in the perf-analyze/optimize workflows”
    #       （作者 kaiyux）的变更说明  [B]
    # 口径：★ 这是本主题 §2 的**一手出处**，逐条列出，**不得扩写**：
    "motivation": (
        "`nsys stats` 只回答「哪个 kernel 花掉的 GPU 时间最多」，"
        "**不回答 iteration 的墙钟时间去哪了**——「those diverge the moment the GPU is not busy」。"
        "按开销排序的 top-kernel 表「reads as a list of opportunities even when the iteration "
        "is mostly waiting on the host」。"
    ),
    "products": (
        "按 iteration 的预算分解：相对**检出的 anchor** 的时间；"
        "**三层嵌套的 busy rung**（连同各自的 **idle 补集**）；"
        "**compute-absent 时间**拆成 **launch-starved / blocking / dependency-stalled**；"
        "kernel 类别分解；**暴露出来的集合通信时间**拆成 transfer 与 jitter wait。"
    ),
    "share_definition": (
        "`kernel_ledger.py` 的 `share_pct` 与 `enumerated_share_pct` 由 "
        "`analysis/nsys_analysis` 的 **interval-union time** 定义，`kern_sum` **只作显式兜底**。"
    ),
    "overlap_warning": (
        "流之间**重叠**会同时改变**排序**与**百分比**——所以分解优先于 `kern_sum`。"
    ),
    "gate": (
        "`headroom_verdict` 成为一道闸：一个 kernel 加速项**声称的加速比不得超过**"
        "该算子**到其约束资源的距离**。"
    ),
    "not_recoverable": (
        "「a launch-starved share is not recovered by a faster kernel」"
        "——launch-starved 那一份**不会因为 kernel 更快而恢复**。"
    ),
    "taxonomy_file": "agent_flow/workflows/perf_analyze/assets/taxonomy_trtllm.json",
}
# ★ **本主题最重要的一条 B 级事实**：`headroom_verdict` 那道闸是**上游实现里真实存在的**，
#   不是本主题编的框架。03 篇的 headroom gate 直接引自它。
# ★ 第二重要：`launch-starved` 那一份**不可恢复**——这正是 02 节 §3 的论断。
# ⚠ 本字典**不含**任何百分比数字。PR #18805 **没有给出各类别的百分比**，
#   任何写「典型 decode 有 30% 是 launch-starved」的句子都是编的。

TENSORRT_LLM_COVERAGE_DISPOSITIONS = (
    # 来源：NVIDIA/TensorRT-LLM PR #18825
    #       “[None][feat] perf-optimize: add elimination and overlap questions to the
    #        per-kernel coverage contract”（作者 kaiyux）  [B]
    ("elimination", "能不能不做"),
    ("faster execution", "做但更快"),
    ("fusion", "与别的算子合并"),
    ("overlap", "与别的流重叠"),
)
# 口径：每个 disposition 都要带 evidence 与 **数值化的 `gpu_busy_pct`** 验证。
#       ★ 「cover 了所有 kernel」≠「cover 了所有时间」——这四个 disposition 才是**闭合**的
#         处置集合（**elimination 排第一**：先问能不能不做，再问怎么更快）。
#       ⚠ 与 PR #18805 是**两个 PR**，不要合并成一条引用。


# ===========================================================================
# 5. vllm.cpp 的 CPU 剖面（**D 级 · 个人仓库 · 且是 CPU 不是 GPU**）
# ===========================================================================
#
# ★★ 本节的每一个数字，正文引用时**必须把「CPU」写在数字旁边**。
#    它是 **aarch64 DGX 上的 CPU prefill 剖面**，与任何 GPU 结论无关。
#    这是本主题最容易犯的 P7（引文与断言作用域不匹配）——把 CPU 剖面当成 GPU 案例。
#
# ★ 来源是**一个个人仓库**（`mudler/vllm.cpp`）的 `.agents/specs/` 文档，
#   因此全部标 **[D]**，且**不得单独支撑任何论断**。

VLLMCPP_PREFILL_PROFILE_TOTAL_MS = 1070.0
# 来源：mudler/vllm.cpp，`.agents/specs/cpu-gdn-proj-orientation-2026-07-23.md`  [D]
# 口径：★ **CPU 剖面**。测量条件（缺一不可）：
#       **aarch64 DGX** · **20 线程** · `--input-len 128 --output-len 1` · **q8_0** ·
#       **warm page cache** · 用临时的 `vt::GetOp` wall-time 钩子采集 · commit `bead27d`。
#       **1070 ms 是 TTFT**（被归属到各 op 的时间总和），不是吞吐。

VLLMCPP_PREFILL_SHARES_PCT = {
    # 来源：同上（同一张表）  [D]
    "kMatmulBTQuant": 50.4,   # q8_0 量化 GEMM，走 i8mm 档；**95 次调用**
    "kMatmul":        17.9,   # [K,N] 的慢 kernel，无 M-blocking，M=128 时带宽受限；**72 次调用**
    "kMatmulBT":      14.9,   # [N,K] 的快 kernel，f16 的 ffn/attn 展开权重；**20 次调用**
    "kGdnPrefill":     5.2,   # 已线程化；18 次调用
    "kPagedAttention": 3.2,   # 已线程化；6 次调用
}
VLLMCPP_PREFILL_GEMM_TOTAL_PCT = 83.0
# 来源：同上  [D]
# 口径：前三个 GEMM 合计 83% —— **与 50.4+17.9+14.9 = 83.2 自洽**（原文写 83）。
#       ★ 这条支撑「先看前 3 个桶」这条 Pareto 做法在**这一份数据上**成立；
#         **不**支撑「3 个桶总是够」这个一般性结论。

VLLMCPP_DEFERRED_OP = "kMatmulBTQuant"
VLLMCPP_DEFERRED_EFFORT_TIER = "MEDIUM–HIGH"
VLLMCPP_CHOSEN_OP = "kMatmul"
VLLMCPP_CHOSEN_EFFORT_TIER = "LOW"
# 来源：同上  [D]
# 口径：★★ 本主题的核心实例，**逐字对应**：
#       · 50.4% 的 `kMatmulBTQuant` 被明确写成「**the next lever**」（G7 quant-GEMM repack，
#         MEDIUM–HIGH effort）——**占比最大的那个被推后了**；
#       · 先做的是 17.9% 的 `kMatmul`（LOW effort，**一个 loader flag**：`gdn_expand_nk`
#         骑在 `expand_nk` 上，`VT_GGUF_GDN_NK=0` 是同二进制的 A/B 退出开关）。
#       ★ 被选中的那个是「**低工作量**」而不是「**高占比**」——这就是 03 篇的论点。
#       ⚠ 原文**没有给出工作量的数值**，只给了 **LOW / MEDIUM–HIGH 这样的档名**。
#         任何把档名换成「2.5 倍工作量」的写法都是**本库的秩映射**，须标 [E]。

VLLMCPP_CHOSEN_MEASURED_SPEEDUP_LOW = 1.090
# 来源：同上（同一系列文档的实测记录）  [D]
# 口径：**同二进制的 prefill 加速 1.090×**，decode **1.09×**；RSS 不变、token 逐字节相同；
#       `kMatmul` 在 prefill 里从 **72 次调用降到 0**。
#       ⚠ 这是**CPU 上的**同二进制 A/B；**不是** GPU 数字，也**不是**端到端。
#       ★ 它是本主题唯一一条「**工作量小 ⇒ 实测收益接近理论上界**」的证据。

VLLMCPP_CEILING_DERIVED = {
    # ★ 本库的**纯算术推导**（[E]）：把某占份额的算子完全消除，Amdahl 上界 = 1/(1 − 份额)
    "kMatmulBTQuant": 2.0161,   # 1/(1−0.504) = 1/0.496：**理论上界**，不是可达值
    "kMatmul":        1.2180,   # 1/(1−0.179) = 1/0.821
}
# 口径：★ 这两个数**只用来做一件反直觉的事**：说明「**占比大的那个上界更大**」
#       （2.02× > 1.22×），而**被选中的偏偏是上界小的那个**。
#       ⇒ 「按上界排」和「按实测决策排」会给出**相反的顺序**，差的正是**工作量**。
#       ⚠ 上界是**完全消除**才拿到的，而「完全消除」在 17.9% 那一项上确实做到了
#         （72 → 0 次调用）——所以这个上界对它是**紧的**；对 50.4% 那一项不紧。


# ===========================================================================
# 6. 跨架构的计时方差（**C 级：arXiv 预印本，未确认同行评审**）
# ===========================================================================

ARXIV_2607_02541 = {
    # 来源：arXiv:2607.02541
    #       “Static PTX Metrics Track Structural Kernel Regressions but Miss Semantic Ones”
    #       （**预印本，本实现者未核实其是否通过同行评审**）  [C]
    "gpus": ("RTX 3060", "A10", "L40S", "A100 SXM4", "H100 NVL"),
    "finding": (
        "同一份结构性改动（correct → buggy 对）的**静态**指标差（Δregs、Δinstrs）"
        "在五类 GPU 上**完全相同**，而**实测** Δperf% **不同**。"
    ),
    "table4": (
        "同一次结构性改动在 **L40S 上 +9.4%**、在 **A100 SXM4 上 −50%**。"
    ),
    "attribution": (
        "原文归因于**亚毫秒尺度**下**每次 launch 的方差**支配了那几条件的指令差异；"
        "五类里有**四类**落在零的 ±20% 以内，**A100 SXM4 被共享宿主机的启动方差支配**。"
    ),
    "conclusion": (
        "静态 PTX 门是可移植的**预筛**；此尺度下的**实测运行时间差**"
        "**不是**可移植的替代品。"
    ),
}
# ★ 本主题引用它的**唯一目的**：支撑「**单机测出的加速比不可跨架构迁移**」。
# ⚠ 标注时必须写明**预印本**；且必须写明 **±20% / 四类**这两个限定，
#   否则会退化成「跨架构完全不可比」这种更强的、原文不支持的断言。
# ⚠ 原文说 A100 SXM4 是「shared-host」——**共享宿主机的启动方差**是它的条件，
#   不是「A100 这个架构不行」。


# ===========================================================================
# 7. 显著性阈值与秩映射（**D 级经验值 + 本库的秩映射**）
# ===========================================================================

HIPFIRE_WITHIN_SESSION_NOISE_PCT_GFX1100 = (10, 15)
# 来源：warpfront/hipfire，`docs/methodology/perf-benchmarking.md`  [D]
# 口径：★★ **条件极具体，必须连条件一起写**：
#       · 平台 **gfx1100**（Radeon 7900 XTX）；
#       · 这是**同一会话内**做代码改动 A/B 的噪声带，**不是**通用噪声带；
#       · 成因：**DPM 状态**（时钟在前 ~5 s 持续负载里爬）、**热余量**、**固件版本**；
#       · 因此该文档要求**换新进程跑**。
#       ⚠ 该文档举的例子是「同一个 shell 会话里改代码重测得到的 +8% 赢，
#         落在噪声之内」——**这是 ±10–15% 这条的具体用法**，不是另一条阈值。

HIPFIRE_GRAPH_CAPTURE_SPREAD_PCT = (1.5, 3.0)
HIPFIRE_GRAPH_CAPTURE_OFF_SPREAD_PCT = 0.1
# 来源：同上（同一份方法论文档）  [D]
# 口径：★★ 原文：**`HIPFIRE_VERIFY_GRAPH=0` gives ~0.1% spread versus 1.5–3% with graph
#       capture on** ——这是**两档不同条件**的数。
#       ⚠ **不得混成一句「噪声带 ±1–3%」**：合并后既不是 graph on 的档，
#         也不是 graph off 的档，而且丢掉了「条件决定档位」这个唯一的结论。
#       ※ 注意方向：**关掉 graph capture 反而更稳**（0.1% < 1.5–3%）。
#         直觉上「用 CUDA Graph 更确定」是**错的**——本档的数是反直觉的。

HIPFIRE_SPEED_GATE_TOLERANCE_PCT = 5
# 来源：同上（`scripts/speed-gate.sh`，以及 DFlash 的 gate）  [D]
# 口径：**相对已提交基线**的容差 **±5%**；pre-commit hook 在 diff 触及 kernel /
#       dispatch / forward-pass 文件时自动跑 `--fast`。
#       ⚠ **不得写成「收益 <5% 不值得做」**——原文**没有**这句话。
#         原文说的事实是「容差 ±5%」与「改变规范 bench 超过 5% 的 flag 才默认开启」。
#         「不值得投入」是本库的**推论**，须标 [E] 并写明它是推论。

EFFORT_RANK_MAPPING = {
    # ★ 本库的**秩映射**（[E]，不是来源值）：把来源里的**档名**映射成正数，
    #   使 03 篇的排序量可计算。来源只给档名（LOW / MEDIUM–HIGH），**不给数值**。
    #   ⚠ 任何使用这张表的结论都必须同时给出**临界比值**，使读者能自己判断映射是否影响结论。
    "LOW": 1.0,
    "MEDIUM": 2.0,
    "MEDIUM–HIGH": 2.5,
    "HIGH": 3.0,
    "VERY HIGH": 4.0,
}
GAIN_BELOW_NOISE_IS_NOT_WORTH_IT = True
# 来源：**本库的推论**（[E]），由 HIPFIRE_SPEED_GATE_TOLERANCE_PCT 与
#       HIPFIRE_WITHIN_SESSION_NOISE_PCT_GFX1100 推出。
# 口径：⚠ **它是推论，不是引用值。** 正文写法必须是
#       「由 ±5% 的 gate 容差**可以推论**……」而不是「研究表明 <5% 不值得做」。
#       且推论**只在 gfx1100 那一档的测量条件上成立**——换平台，噪声带不同，阈值就不同。


# ===========================================================================
# 8. (占位) 本主题不使用的常量
# ===========================================================================
#
# 本主题是**流程域**，因此**刻意不引入**下面几类常量：
#   · 任何 kernel 的实测吞吐 / 加速比（本机无 GPU，一条都拿不到）；
#   · 任何具体 kernel 的实现参数（块大小、寄存器数、tile 形状）——那是 `40-kernels` 的边界；
#   · 任何 profiling 工具的参数（ncu 的 section、nsys 的 flag）——那是 `50-performance` 的边界。


# ===========================================================================
# 9. UNSOURCED —— 待办清单，不是免责声明
# ===========================================================================
#
# ★ §0.6 硬规则 5：**不得「既保留数字又记 UNSOURCED」**。下表每一项都**没有**在
#   本模块的正文常量里出现；它们要么被删掉，要么被换成了有来源的说法。

UNSOURCED = (
    # 1. hipfire 的「冷设置 warmup=24 比 warmup=128 的读数偏低约 12%」
    #    状态：**本次检索未找到**。检索到的 `docs/methodology/perf-benchmarking.md` 里，
    #    与 warmup 有关的只有 `--warmups`（bench 默认 **10** 次丢弃）与
    #    `HIPFIRE_DPM_WARMUP_SECS=10`；`128` 在检索片段里只作为 `--max-tokens 128`
    #    出现，**不是 warmup 计数**。检索到的最近的量是 **~10%** 的
    #    「probe `gen tok/s` 与 bench `gen_tok_s` 的残差」，而它的成因是
    #    **daemon 的 per-token 开销（JSONL 流式、detokenize、停止条件检查）**，
    #    **与 warmup 无关**。
    #    处置：**不写进正文**。
    "hipfire_warmup_24_vs_128_12pct",

    # 2. 「JIT 冷缓存首次编译 30 s–2 min；模型重载冷启动 2–5 s」
    #    状态：**本次未复核**（本主题的三个编号程序不涉及冷启动，该论断属
    #    `convergence-and-tuning` 主题）。**本主题不引用**，留给该主题去核。
    "cold_start_time_scales",

    # 3. 「TensorRT-LLM 的时间线分解类别名（busy / idle / launch-starved / blocking /
    #    dependency-stalled）」——**本次已核到**（PR #18805 的说明里逐字有
    #    「three nested busy rungs … their idle complements」与
    #    「launch-starved / blocking / dependency-stalled」）。**因此它不是 UNSOURCED**，
    #    已写进 §4 的 `TENSORRT_LLM_TIMELINE_DECOMPOSITION`；此处仅留一行说明它已被移出。
    #    ⚠ 但**各档的具体百分比**仍未检索到 —— 见下。
    "trtllm_taxonomy_category_names__RESOLVED_see_section_4",

    # 4. TensorRT-LLM 时间线分解里**各档的百分比**（例如「launch-starved 占 30%」）
    #    状态：**未找到**。PR #18805 定义了这些档，但检索片段里**没有任何一个百分比**。
    #    处置：§4 的字典**刻意不含数字**；正文不得出现这类百分比。
    "trtllm_per_category_percentages",

    # 5. Qwen3-30B-A3B 的**共享专家**规格
    #    状态：**未找到**。检索只给出 `intermediate_size: 6144`，而该字段的语义是
    #    「非 MoE 的 FFN 宽度」，**不等于共享专家的宽度**；`shared_expert_intermediate_size`
    #    这一项在检索片段里没有出现。
    #    后果：01 篇的 MoE 活跃参数量是**下界**（不含共享专家）。正文**必须**写明这一点。
    "qwen3_30b_a3b_shared_expert_size",

    # 6. 「绝大多数 kernel 在 80% 带宽处饱和」这类**编机制从句**（P4）
    #    状态：**无来源，也测不了**。处置：不写。
    "ungrounded_bandwidth_saturation_claims",
)
# ★ `UNSOURCED` 是**待办清单**：上面每一项都在等一个可定位的一手来源，
#   在它被找到之前，**正文里不得出现对应的数字**。

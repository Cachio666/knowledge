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
# 来源：IEEE 754 binary16 的宽度（规范，[A]）
# 口径：**存储宽度**，不是「有效位」。fp16 的 1+5+10 是位域；
#       2024/2025 年的数据中心 GPU 上**没有 fp16 存储的例外**（fp8 才有）。
# ★ **只保留本主题真正用到的那一个宽度。** `FLOAT32_BYTES = 4` 曾在这里，
#   但三个程序一次都没引用它（02 篇的口径三件套写死了 fp16）——按本域的
#   C6 判据（逐个改常量，`make run` 必须报错），**未被断言的常量必须删掉或补断言**，
#   故删。需要 fp32 宽度时去 `20-gpu-architecture/tensor-core/code/specs.py` 取。

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
    # 来源：NVIDIA 各代官方材料（[A]）
    # 口径：**dense / sparse 二元组**，单位 TFLOPS。上游注释逐字警告：
    #       「NVIDIA 产品页写的 H100 FP16 1979 TFLOPS **是带 2:4 稀疏的**，dense 只有 989.5」。
    #       V100 的 sparse 位是 `None`——Volta 没有 2:4 结构化稀疏，**不是「未采集」**。
    #       `None` 不是可变异的叶子，故只作为口径的一部分保留。
    #
    # ★★ **本模块只保留 `FP16` 一行**（本主题唯一的精度路径）。
    #    上游表还有 `TF32` / `BF16` / `FP8` / `FP4` 四行，**与本主题无关，已裁掉**——
    #    按本域的 C6 判据，「照抄但不使用」的部分**无法被任何断言约束**，
    #    留着就是在变异测试里制造漏网。完整五行的表在
    #    `20-gpu-architecture/tensor-core/code/specs.py:323`。
    "FP16": {"V100": (125.0, None), "A100": (312.0, 624.0),
             "H100": (989.5, 1979.0), "B200": (2250.0, 4500.0)},
}

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

# ★ **`KERNEL_LAUNCH_OVERHEAD_US_EXTERNAL = (3.0, 5.0)` 与 `SM_COUNT_BY_GEN` 曾在这里，
#   现已删除。**
#   理由（C6 判据）：本主题的**任何一篇文档都没有引用它**，三个程序也没有——
#   它的用途是「把占比换算成绝对时间量级，判断是否落在噪声地板下」，
#   而那是 `convergence-and-tuning`（Chunk 4）的知识对象。
#   **留着它就是给下一个主题一个「看起来已经采过源」的假象。**
#   需要时去 `40-kernels/gemm/code/specs.py:198` 取（那里有完整口径注释）。


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

# ★ 本模块**不写**「每参数 20 token」这类训练口径的经验值，也**不放占位常量**。
#   理由：它是**训练**侧的 Chinchilla 结论，本主题是**推理**侧，写进来就是 P9（同名不同指）。
#   （曾有一个 `TOKENS_PER_PARAM_RULE_OF_THUMB = None` 的占位，现已删除：
#     **一个值为 `None` 的常量不可能被任何断言约束**，留着只会在变异测试里算成漏网。）


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

# ── 为什么这一节**没有散文** ───────────────────────────────────────────────
#
# ★★ 本模块初版把来源里的**描述性段落**整段搬了进来（三档各一段、四个新算子各一段、
#    taxonomy 各条一段……）。那是一个**结构错误**，不是文风问题：
#      · **程序无法计算散文**——于是它只能被 `print`，也就**不可能被断言**，
#        在任何变异测试里都是**必然漏网**；
#      · 它还把**已裁掉的常量**（`HIPFIRE_*` 等）掩盖在一个「看起来很充实」的模块里，
#        让下一个主题以为这里已经采过源。
#    **⇒ 规则：`specs.py` 只放「程序会拿去算的东西」（数、名、布尔、阈值、URL）。
#      描述性文字归**文档**。来源里的每个**数字**仍必须落在这里（§0.6 硬规则 4），
#      只是以**结构化字段**的形式落。**

VLLM_MODEL_INTEGRATION_TIERS = ("simple", "moderate", "complex")
# 来源：vLLM 官方文档 “Adding a New Model” / “Adding new models”  [B]（**经检索转述**）
# 口径：**vLLM 一家**的分档方式（P10：不得写成「业界通常分三档」）。
VLLM_TIER_NEEDS_NEW_KERNEL = "complex"
# 口径：★ 只有这一档需要写新 kernel（含新 attention 机制那类）。
#       01 篇 §5 判据表第 11 行就是它的机器化。
VLLM_TRANSFORMERS_BACKEND_HINT = "vllm serve <model>"
# 口径：一个**先跑再说的动作**，是判据表的第一道闸。

VLLM_IR_DECLARATION_DECORATOR = "@register_op"
VLLM_IR_IMPL_DECORATOR = "register_impl"
VLLM_IR_PRIORITY_CLI_PREFIX = "--ir-op-priority"
VLLM_IR_RESERVED_FALLBACK_PROVIDER = "native"
# 来源：vLLM 官方文档 `vllm.ir.op` / `design/vllm_ir` 与 PR #33825  [B]（**经检索转述**）
# 口径：**算子语义与实现分离**；选择规则是「**优先级列表中第一个 supported 且
#       `supports_args` 通过的实现**」（`native` 是保留名，永远存在 ⇒ 列表永远有兜底）。
#       ★ 01 篇 §3 的论点就是「优先级选择是一个**显式**配置项」。

KIMI_K3_NEW_OPERATOR_NAMES = ("KDA prefill", "KDA decode", "Attention Residuals", "SiTU")
# 来源：vLLM 官方博客（2026-07-22 / 07-27）与 vLLM v0.28.0 release notes  [B]（经检索转述）
# 口径：★ 本主题要的不是这个列表本身，而是**它有 4 项、且 prefill 与 decode 是两项**。
KIMI_K3_KDA_PREFILL_PRS = (51311, 52458)
KIMI_K3_KDA_DECODE_PR = 50654
KIMI_K3_SITU_MEGAMOE_PR = 50510
KIMI_K3_ATTNRES_MAX_CACHED_BLOCKS = 8
KIMI_K3_ATTNRES_MAX_SOURCES = 9
# 口径：Block AttnRes 对**至多 8 个**缓存的块表示 + 当前块内残差做**深度方向** softmax
#       ⇒ 至多 **9** 个来源。（01 篇 §2.2 的表里用这两个数。）

KIMI_K3_PREFIX_CACHE_SEPARATED_CONCEPTS = (
    "physical block size", "scheduler alignment", "prefix-match unit"
)
# 来源：同上  [B]
# 口径：KDA 是**循环**的（不存 per-token KV 对），所以 vLLM 把三个原本绑在一起的概念拆开。
#       ★ 这是「新模型的成本**不止于算子数**」的证据。
KIMI_K3_PREFIX_CACHE_RULE_IS_COPY_ON_WRITE = True
KIMI_K3_PREFIX_CACHE_ENV_VAR = "VLLM_PREFIX_CACHE_RETENTION_INTERVAL"

KIMI_K3_METADATA_BUILDER_LATENCY_US = (870, 34)
KIMI_K3_METADATA_BUILDER_E2E_GAIN_PCT = 6
# 来源：同上  [B]
# 口径：**batch size 1**；metadata 准备延迟 870 µs → 34 µs；端到端 DSpark 改善 **6%**。
#       ★ 本主题唯一一条「小占比换来可测收益」的 B 级数字。

TRTLLM_18805_SHARE_PCT_IS_INTERVAL_UNION = True
TRTLLM_18805_KERN_SUM_IS_EXPLICIT_FALLBACK_ONLY = True
TRTLLM_18805_HEADROOM_VERDICT_IS_A_GATE = True
TRTLLM_18805_OVERLAP_CHANGES_RANK_AND_PCT = True
TRTLLM_18805_TAXONOMY_FILE = ("agent_flow/workflows/perf_analyze/"
                              "assets/taxonomy_trtllm.json")
# 来源：NVIDIA/TensorRT-LLM PR #18805  [B]（★ **经检索转述**——本实现者只读到检索摘要，
#       **未逐字读到 diff 原文**，故正文不得写「逐字」「原文如此」。）
# 口径：★ `share_pct` / `enumerated_share_pct` 由 `analysis/nsys_analysis` 的
#       **interval-union time** 定义，`kern_sum` **只作显式兜底**（且兜底要记进
#       ledger 的 `source` 字段）。**流之间重叠会同时改变排序与百分比。**
#       ★ `headroom_verdict` **是一道校验**：一个 kernel 加速项声称的加速比
#       **不得超过**该算子到其约束资源的距离 —— 03 篇 §1.2 的闸直接引自它。
# ⚠ **本模块刻意不含**（初版在这里归错过，见 §4 末的引用纪律）：
#     · 「three nested busy rungs … their idle complements」——出处**不是**这个 PR；
#     · 「a launch-starved share is not recovered by a faster kernel」——**两次独立检索
#       都查不到**，已改写为推论（`NOT_RECOVERABLE_IS_DERIVED`）；
#     · 任何**百分比**——该 PR 没有给出各类别的百分比。

NSYS_COMPUTE_ABSENT_DEFINITION = "Iteration minus compute busy"
NSYS_BUCKET_LAUNCH_STARVED = "launch-starved"
NSYS_BUCKET_BLOCKING = "blocking"
NSYS_BUCKET_DEPENDENCY_STALLED = "dependency-stalled"
NSYS_BUCKETS = (NSYS_BUCKET_LAUNCH_STARVED, NSYS_BUCKET_BLOCKING, NSYS_BUCKET_DEPENDENCY_STALLED)
NSYS_LAUNCH_STARVED_DEFINITION = "the next compute kernel had not yet been issued by the host"
NSYS_BLOCKING_DEFINITION = ("the kernel was issued and a resolved producer finished after the "
                            "gap began; this requires a device dependency and negative slack")
NSYS_DEPENDENCY_STALLED_DEFINITION = ("the kernel was issued, but no resolved producer explains "
                                      "the remaining delay")
NSYS_RECONCILIATION = "launch-starved + blocking + dependency-stalled ~ compute-absent"
NSYS_RESIDUAL_MS_THRESHOLD = 0.5
NSYS_WALL_TIME_RULE = ("kernel-duration sums must be labeled only as work volume, "
                       "never presented as wall time")
NSYS_URL = ("https://docs.nvidia.com/nemo/megatron-bridge/nightly/skills/"
            "nemo-mbridge-perf-nsys-analysis/SKILL.html")
# 来源：NVIDIA NeMo Megatron Bridge 的 Nsight Systems 分析文档（`NSYS_URL`）  [B]
# 档位：★★ **逐字**——上列各条由**两次独立检索得到同一段文字**（本实现者一次、验收者一次），
#       故可加引号；出处已给到**文件级 URL**。
# 口径：★ 本模块**初版**把「compute-absent 三分」记在 PR #18805 名下并标「逐字」——
#       **那是错的归属**。三分法的可复核出处是**这篇**。PR #18805 是**另一个主体**的
#       **另一件事**；「两个来源都涉及时间线分解」**不等于**「说的是同一句话」。
#       ★ 配平恒等式与 0.5 ms 残差门限是本主题**最强的一条可证伪断言**（03 篇程序 §B.5）。
#       ★ `NSYS_WALL_TIME_RULE` 与 PR 那条 `share_pct` 口径警告**同向**：
#         kernel 时长之和**只能当工作量，不能当墙钟**。

NOT_RECOVERABLE_IS_DERIVED = True
# 来源：**本库推论（[E]）**，不是引文。
# 口径：★★ 从 `NSYS_LAUNCH_STARVED_DEFINITION` 的**逐字定义**推出：
#       瓶颈在「宿主**还没把它发出去**」这一侧，而「更快的 kernel」改变的是
#       **已发出的 kernel 的执行时长**——两者不在同一侧。
#       ⇒ launch-starved 那一份**不会被更快的 kernel 收回**。
#       **这是推论，正文须标 [E]，不得写成「原文如此」。**

TRTLLM_COVERAGE_DISPOSITIONS = ("elimination", "faster execution", "fusion", "overlap")
# 来源：NVIDIA/TensorRT-LLM PR #18825  [B]（经检索转述）
# 口径：★ **elimination 排第一**（先问能不能不做，再问怎么更快）。
#       ⚠ 与 PR #18805 是**两个 PR**，不要合并引用。

NOISE_THRESHOLD_RULE = "任何显著性阈值都必须连带测量条件；条件给不出，阈值不可用"
NOISE_THRESHOLD_RULE_TAG = "E"
# 来源：**本库对一批 D 级材料的提炼（[E]）**，不是任何一条引用值。
# 口径：★ 它是一条**可机械判定的判据**，不是数。依据是结构性的：阈值取决于**测量方式**
#       （是否换新进程、是否用 CUDA Graph、warmup 多长、平台是否受 DPM/热限），
#       **换条件就换档** ⇒「阈值是多少」没有无条件的答案。
#       ★★ 因此本主题**不保留任何噪声带数值**：`HIPFIRE_*`（会话内 A/B 噪声带、
#         graph capture 两档离散度、speed gate 容差）与 `KERNEL_LAUNCH_OVERHEAD_US_EXTERNAL`
#         已**整体移除**——它们是 `convergence-and-tuning`（Chunk 4）的知识对象，
#         在本主题三个程序里 0 引用，无法被断言约束。
#         正文只保留**一条定性的 [D] 级观察**（「关掉 graph capture 反而更稳」），
#         标 D 级、单一项目、未被独立复现，且**不给数字**。


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
# 6. （已移除）跨架构的计时方差
# ===========================================================================
#
# ★ **`ARXIV_2607_02541`（L40S +9.4% / A100 SXM4 −50%）曾在这里，现已删除。**
#   理由（C6 判据 + 范围）：本主题**三篇文档一篇都没有引用它**，三个程序也没有。
#   它支撑的论断是「**单机测出的加速比不可跨架构迁移**」——那是**测量方法学**的结论，
#   属 `convergence-and-tuning`（Chunk 4）的知识对象，不是「需求发掘」的。
#   **留在这里只会让 Chunk 4 以为它已经采过源了。**
#   需要时按这个编号去 `docs/superpowers/plans/2026-09-20-kernel-lifecycle-domain.md`
#   §0.7.G 取（那里有完整的口径与限定：**预印本**、四类落在 ±20% 内、A100 SXM4 是
#   shared-host 的启动方差）。


# ===========================================================================
# 7. 秩映射（**本库的 [E] 级映射，不是来源值**）
# ===========================================================================

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
    # ── 本清单的口径 ───────────────────────────────────────────────────────
    # ★ 每一项是一个**稳定标识符**（不是散文）。三个程序对本元组做**逐条比对**
    #   （01 篇 §D.8），所以改任何一个标识符都会让 `make run` 失败。
    # ★ §0.6 硬规则 5：**不得「既保留数字又记 UNSOURCED」**——本清单里的每一项
    #   都**没有**出现在本模块的正文常量里。

    # 1. hipfire 的「冷设置 warmup=24 比 warmup=128 的读数偏低约 12%」
    #    状态：**检索未找到**。那篇方法论文档里与 warmup 有关的只有 `--warmups`
    #    （bench 默认 **10** 次丢弃）与 `HIPFIRE_DPM_WARMUP_SECS=10`；`128` 在检索片段里
    #    只作为 `--max-tokens 128` 出现，**不是 warmup 计数**。最近的一个量是 **~10%** 的
    #    「probe `gen tok/s` 与 bench `gen_tok_s` 的残差」，成因是 **daemon 的 per-token
    #    开销**，**与 warmup 无关**。处置：**不写进正文**。
    "hipfire_warmup_24_vs_128_12pct",

    # 2. 「JIT 冷缓存首次编译 30 s–2 min；模型重载冷启动 2–5 s」
    #    状态：**未复核**（本主题三个程序不涉及冷启动，该论断属
    #    `convergence-and-tuning`）。处置：本主题不引用，留给该主题去核。
    "cold_start_time_scales",

    # 3. 「compute-absent 三档的**百分比**」（例如「launch-starved 占 30%」）
    #    状态：**未找到**。两处出处（NeMo Megatron Bridge 的 NSYS 分析文档与
    #    TRT-LLM PR #18805）**都定义了档位、都没有给出任何百分比**。
    #    处置：§4 **刻意不含任何百分比**；正文不得出现这类句子。
    "compute_absent_per_category_pct",

    # 4. Qwen3-30B-A3B 的**共享专家**规格
    #    状态：**未找到**。检索只给出 `intermediate_size: 6144`，而该字段的语义是
    #    「非 MoE 的 FFN 宽度」，**不等于共享专家的宽度**；`shared_expert_intermediate_size`
    #    在检索片段里没有出现。
    #    后果：01 篇的 MoE 活跃参数量是**下界**（不含共享专家）。正文**必须**写明。
    "qwen3_30b_a3b_shared_expert_size",

    # 5. 「绝大多数 kernel 在 80% 带宽处饱和」这类**编机制从句**（P4）
    #    状态：**无来源，也测不了**。处置：不写。
    "ungrounded_bandwidth_saturation_claims",

    # 6. ★★ **一处已订正的错误归属**（保留在这里是**给后来者的警告**，不是待办）：
    #    「three nested busy rungs … their idle complements」与
    #    「a launch-starved share is not recovered by a faster kernel」——本模块**初版**
    #    把它们记在 TRT-LLM PR #18805 名下并标「逐字」。**两次独立检索都查不到这两处**：
    #    前者实际出自 NeMo Megatron Bridge 的 NSYS 分析文档（现在 `NSYS_*` 一组常量里），
    #    后者**任何出处都查不到**（已改写为 `NOT_RECOVERABLE_IS_DERIVED` 的本库推论）。
    #    处置：**正文与 specs.py 均不得再出现这两句原文**（03 篇程序 §D.5 有断言守着）。
    "misattributed_verbatim_quotes_in_v1",
)
# ★ `UNSOURCED` 是**待办清单**：上面每一项都在等一个可定位的一手来源，
#   在它被找到之前，**正文里不得出现对应的数字**。

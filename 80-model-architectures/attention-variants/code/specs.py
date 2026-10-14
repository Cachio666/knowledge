"""80-model-architectures/attention-variants 的外部事实常量源。

本模块是 `CONTRIBUTING` §2 约定的**共享事实源**（`specs.py`）：本主题的编号程序从它读常量，
推导过程可复核，**常量本身不可自证**。

★★ **本模块的约定：来源分三类，且必须分开标注**（计划 §C.4 的分组纪律；域 README §「代码约定」）：

  · **第一类 [B]**：字段**逐字来自** `45-kernel-lifecycle/requirement-discovery/code/specs.py`
    的 `LLAMA3_8B_CONFIG` / `QWEN3_30B_A3B_CONFIG` ⇒ 标 **[B]**，并给出 `# 同值位置：`。
  · **第二类（镜像）**：**不在那份文件里**、经 `hf-mirror.com` **镜像**取得的字段 ⇒ 带强制限定语
    「**经 `hf-mirror.com` 镜像取得，未与官方 repo 比对 sha256**」。
  · **第三类 [C]**：**论文正文里的 hyperparameter**（本主题新增的一类）。它不是 `config.json`，
    而是作者在论文里写下的取值 ⇒ 标 **[C]**，给 arXiv 号 + 抓取日 + **原始字节计数**，
    并按 `CONTRIBUTING` §2 注明**是否确认同行评审**。

★ **本模块只放本主题真的会拿去算的东西**（数、名、布尔、元组、阈值）。
  `45-…/code/specs.py:107-113` 记着一条教训：**「照抄但不使用」的部分无法被任何断言约束**，
  只会制造变异测试的漏网。⇒ 其它字段的**去向**写在文件尾注里，**不搬进来**。
  描述性文字归**文档**，不归本模块（先例：`45-…/code/specs.py` §4 的「为什么这一节没有散文」）。

★ 每条常量带 `# 来源：` + `[A-E]` 级 + 抓取日；跨主题同值的另带 `# 同值位置：`。
  分级判据见 `CONTRIBUTING` §2（A 厂商规范 / B 产品官方文档 / C 论文 / D 社区 / E 本库推导）。
"""

# ===========================================================================
# 1. 两个已核实 config 的「head 关系」 —— ★ 本主题的全部真实模型锚点
# ===========================================================================
#
# ★ **只搬「head 关系」这两个字段**（不是整份 config）：
#   本主题的账是**组数** `G = heads_q / heads_kv` 与由它派生的**倍率**，
#   而 `hidden_size` / `intermediate_size` / `vocab_size` 等字段与本主题无关
#   （它们的去向见文件尾注）。搬进来而不用，只会制造变异漏网。

LLAMA3_8B_ATTENTION = {
    "num_attention_heads": 32,      # heads_q
    "num_key_value_heads": 8,       # heads_kv  ⇒ G = 32/8 = 4
}
# 来源：meta-llama/Meta-Llama-3-8B 的 config.json，**转记自**
#       45-kernel-lifecycle/requirement-discovery/code/specs.py 的 LLAMA3_8B_CONFIG  [B]
#       （该模块自注「经 HuggingFace 与 torchtune 的 llama3_8b() 构造器双向核对」）
# 口径：★ 这是**转记**，不是本域重新溯源。**不改任何一位数字。**
# 同值位置：45-kernel-lifecycle/requirement-discovery/code/specs.py:131-145；
#           80-model-architectures/transformer/code/specs.py:40-47（LLAMA3_8B_FIELDS_FROM_UPSTREAM）
#       ★ 本域是**第 3 处副本**。那一份的 :27 逐字写着「⚠ 复制造成的**漂移风险**：
#         这些值在仓库里有多份副本（见每条的「同值位置」）」—— 本模块遵守同一条约定。

QWEN3_30B_A3B_ATTENTION = {
    "num_attention_heads": 32,      # heads_q
    "num_key_value_heads": 4,       # heads_kv  ⇒ G = 32/4 = 8（★ 与 Llama 的 4 不同）
}
# 来源：Qwen/Qwen3-30B-A3B 的 config.json，**转记自**
#       45-kernel-lifecycle/requirement-discovery/code/specs.py 的 QWEN3_30B_A3B_CONFIG  [B]
#       （该模块自注「经 4 处独立转抄核对」）
# 口径：★ 这两个字段**都是原生字段**（不是推导值）—— `num_attention_heads` 与
#       `num_key_value_heads` 在真实 config 里逐字存在。
#       ⚠ **不要**用 `hidden_size // num_attention_heads` 去推 `head_dim`（那是另一个字段，
#         在本模型上等于 64，而 config 里的 `head_dim` 是 128）—— 那个陷阱的正文在
#         `45-kernel-lifecycle/requirement-discovery/01-model-to-operator-inventory.md:270-276`
#         （正文在彼，此处回链）。**本主题不用 `head_dim`**，所以这条陷阱在这里不进入算式。
# 同值位置：45-kernel-lifecycle/requirement-discovery/code/specs.py:150-165；
#           80-model-architectures/transformer/code/specs.py:95-103

# ===========================================================================
# 2. 滑窗字段：★ **两个模型一个有、一个没有**（这是「config 原生 vs 不存在」的对照）
# ===========================================================================

QWEN3_30B_A3B_SLIDING_WINDOW_CONFIG = {
    "use_sliding_window": False,    # 字段存在，取值 false
    "sliding_window": None,         # 字段存在，取值 null
}
# 来源：Qwen/Qwen3-30B-A3B 的 config.json  **经 `hf-mirror.com` 镜像取得**（第二类）
#       （2026-09-22 直取 `https://hf-mirror.com/Qwen/Qwen3-30B-A3B/resolve/main/config.json`）
# ★★ **未与官方 repo 比对 sha256。** 镜像不是厂商域名，其副本与官方 repo 的一致性
#    **在本环境无法复核**（`huggingface.co` 直连不可达）。镜像副本的字节哈希
#    （**镜像的**，不是官方 repo 的）：sha256[:16] = 2850ddb3bf7aecad（963 B）
#    —— 与 `80-model-architectures/transformer/code/specs.py:123` 记的是**同一份副本**
#    （两处独立抓取，字节哈希一致）。
# 口径：★ `None` 是**来源里的真值**（JSON 的 `null`），不是「本域没查到」的占位符。
#       本模块**保留它就是原值**这一事实：它可被断言（见 `03_window_hybrid.py` 的 `C.3`），
#       所以不属 `45-…/specs.py` 禁止的那种「值为 None 的占位常量」。

LLAMA3_8B_ABSENT_SLIDING_WINDOW_FIELDS = ("sliding_window", "use_sliding_window")
# 来源：meta-llama/Meta-Llama-3-8B 的 config.json  **经 `hf-mirror.com` 镜像取得**（第二类）
#       （2026-09-22 直取 `https://hf-mirror.com/NousResearch/Meta-Llama-3-8B/resolve/main/config.json`，
#         sha256[:16] = 2430cee764b6530f（654 B）；与 `…/transformer/code/specs.py:66` 同）
# ★★ **未与官方 repo 比对 sha256**（同上的全部限定）。
# 口径：★★ 这**不是**「字段存在但为 null」，而是**字段根本不在 config 里** ——
#       该 config 的全部键是 23 个（`architectures` … `vocab_size`），**这两个名字一个都没有**。
#       ⇒ 「有没有这个字段」本身是两个模型之间的一处**真实差异**，而它正是
#         「同一个变体在不同模型上开或关」这条架构事实的**配置侧证据**。
#       ⚠ 反过来说：`use_sliding_window = False` 与「字段不存在」在**语义上并不等价**——
#         前者是**显式关掉**，后者是**该实现压根没有这个开关**。本域把两者分开写。

# ===========================================================================
# 3. 滑窗的窗口与层数：★ 来自 **Mistral 7B 论文**（[C]）——不是 config.json
# ===========================================================================
#
# ⚠ **口径**：本主题的滑窗锚点是**论文**（`§C.6` 的 W1/W2），**不是** HuggingFace 的 config。
#   理由（计划 §C.0 的硬约束）：`huggingface.co` 直连不可达，而本域只允许引用
#   `45/…/code/specs.py` 里**已核实**的两个 config。Mistral 7B 不在其中 ⇒ 它的数**只能**
#   从论文来，且必须标 [C] 并注明**未确认同行评审**。

MISTRAL7B_WINDOW_SIZE = 4096
# 来源：Mistral 7B 论文（arXiv 2310.06825）正文 §2.1 逐字
#       「At the last layer, using a window size of `W = 4096`, we have a theoretical
#         attention span of approximately 131K tokens.」  [C]
#       ★ 同文的 Table 1（Model architecture）逐字有 `window_size 4096` 一行。
#       抓取：`https://ar5iv.labs.arxiv.org/html/2310.06825`，2026-09-22，
#             **99865 B**，sha256[:16] = b9aa0304ed429dcb（与计划 §C.2 逐位一致）
# 评审状态：★ **无 venue 线索 ⇒ 未确认同行评审**（按 abs 页元数据实测填，**不凭印象补**）
# 口径：★ 论文里 `W` 是**距离**（可见范围是 `[i−W, i]`，即 `W+1` 个位置）——
#       逐字：「The hidden state in position `i` of the layer `k`, `h_i`, attends to all hidden
#       states from the previous layer with positions between `i − W` and `i`.
#       Recursively, `h_i` can access tokens from the input layer at a distance of up to
#       `W × k` tokens」。⇒ **距离**是乘积 `W×k`，而**可见位置数**是 `W+1`。
#       ⚠ 这条区分是可测的（见 `03_window_hybrid.py` 的 `A.2`），**不要**把它当成等价的说法。

MISTRAL7B_NUM_LAYERS = 32
# 来源：Mistral 7B 论文（arXiv 2310.06825）的 Table 1「Model architecture」逐字一行
#       `n_layers 32`（同行另有 `dim 4096`、`n_heads 32`、`n_kv_heads 8`、`window_size 4096`）  [C]
#       抓取与哈希同上（99865 B / b9aa0304ed429dcb）
# 口径：★★ 它是**这篇论文的 Table 1**，不是任何 `config.json`（那种一手来源在本环境取不到）。
#       「32 层」与 W2 的 `approximately 131K` 合起来才复现出 `4096 × 32 = 131072`
#       —— **这条乘法是本主题的判据⑥**，见 `03_window_hybrid.py` 的 `A.1`。

# ===========================================================================
# 4. MLA 的解耦 RoPE：论文自述的**两个比例**（[C]）—— 由此闭式复现「2.25 组」
# ===========================================================================
#
# ★★ 为什么放**比例**而不是绝对维度：论文只写了这两个比例（`d_c = 4 d_h`、`d_h^R = d_h/2`），
#    **绝对值 `d_h` 在本环境没有一手来源** ⇒ 记比例，`d_h` 在程序里当**符号单位**消掉。
#    ⇒ 判据（`equal to GQA with only 2.25 groups`）**可以纯由论文自己的两个数复现**，
#      不需要知道 DeepSeek-V2 的绝对 config。这就是「配一条闭式参照」的落法。

DEEPSEEK_V2_D_C_OVER_D_H = 4
# 来源：DeepSeek-V2 / MLA 论文（arXiv 2405.04434）§2.1.3 逐字
#       「For DeepSeek-V2, `d_c` is set to `4 d_h` and `d_h^R` is set to `d_h/2`.」
#       [C] 抓取：`https://ar5iv.labs.arxiv.org/html/2405.04434`，2026-09-22，
#            **694593 B**，sha256[:16] = 65b7c4dc9d1b38e0（与计划 §C.2 逐位一致）
# 评审状态：★ **无 venue 线索 ⇒ 未确认同行评审**（技术报告）
# 口径：它是**论文正文里作者写下的取值**，不是 `config.json` 字段（`huggingface.co` 不可达）。

DEEPSEEK_V2_D_H_R_OVER_D_H = 0.5
# 来源：同上一行（`d_h^R` is set to `d_h/2`）⇒ 比例 = 0.5   [C]
# 口径：★ 0.5 是**精确的二分之一** ⇒ 闭式里用 `Fraction` 算，**不引入浮点误差**
#       （`2.25` 是精确有理数 `9/4`，不是近似值）。
# ★★ 语义：`d_h^R` 是「解耦的那部分 queries 与 key 的**每头维度**」，而它对应的 key
#    **在所有头上只有一条**（逐字：`a shared key k_t^R ∈ R^{d_h^R}`）
#    ⇒ 这就是「解耦的代价只有 `d_h^R` 而不是 `n_h · d_h^R`」的原因。

# ===========================================================================
# 5. UNSOURCED —— 待办清单，不是免责声明
# ===========================================================================
#
# ★ CONTRIBUTING §2：「不得**既保留数字又记 UNSOURCED**」—— 下面每一项都**没有**出现在
#   本模块的正文常量里。

UNSOURCED = (
    # 每一项是一个**稳定标识符**（不是散文）。
    "mistral7b_config_json",          # Mistral 7B 的 `config.json`：HF 直连不可达
                                      # （计划 §C.0）⇒ 本域只引论文 Table 1 的 [C] 值。
                                      # ⚠ 因此**不能**用「Mistral 7B 的 config 里 `sliding_window=4096`」
                                      # 这种说法 —— 那句话在本环境**没有一手来源**。
                                      # 论文正文的说法才是可引的。
    "deepseek_v2_absolute_dims",      # DeepSeek-V2 的 `d_c` / `d_h^R` / `d_h` 的**绝对值**：
                                      # 论文只给了比例（已收进上面两条），绝对维度没有一手来源。
                                      # ⇒ 本域用「以 `d_h` 为单位」的闭式，不需要绝对值。
    "mla_93_3_percent_kv_reduction",  # `93.3%` 这个数**已被本域核实**，但它是**引用值**
                                      # （相对「DeepSeek 67B」这个具体 dense 模型），
                                      # 不能被程序判定 ⇒ 它归**表 B 引用判据**，不进本模块。
                                      # ★ 同名的另一个 `93.3` 是 ARC-Challenge 准确率（同名不同指）。
)

# ===========================================================================
# 尾注：本主题**刻意不搬**的字段（它们的正文在别处）
# ===========================================================================
#
#   · `head_dim`（两个模型的）                    → 本主题**不用**它：组数与倍率都与它无关。
#       它自己的陷阱（`hidden_size // heads` ≠ `head_dim`）正文在
#       `45-kernel-lifecycle/requirement-discovery/01-model-to-operator-inventory.md:270-276`。
#   · `num_hidden_layers` / `hidden_size` / `intermediate_size` / `vocab_size`
#                                                 → `transformer/code/specs.py`（参数量分配的账）
#   · `rope_theta` / `max_position_embeddings`    → `positional-encoding/code/specs.py`
#   · `rms_norm_eps` / `hidden_act`               → `normalization` / `transformer`
#   · `num_experts` / `num_experts_per_tok` /
#     `moe_intermediate_size` / `router_aux_loss_coef` → `moe/code/specs.py`
#   · KV cache 的**字节**公式（`2 × layers × kv_heads × head_dim × bytes`）
#                                                 → `130-model-optimization/quantization/18-kv-cache-basics:48-76`
#       ★★ 本主题只算**元素数**与**倍率**，**不算字节** —— 那是**禁区**（计划 §2 的边界表）。
#
#   （★ R3 的裁定：**每主题一份，每份只放本主题用得到的常量**，跨主题同值的常量各写一份、
#     靠 `# 同值位置：` 互指。本库 40 个 `code/` 目录**没有任何跨主题共享辅助模块的先例**，
#     `CONTRIBUTING:36`「`code/` 必须自包含」是**全库规则**。）

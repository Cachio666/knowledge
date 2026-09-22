"""80-model-architectures/moe 的外部事实常量源。

本模块是 `CONTRIBUTING` §2 约定的**共享事实源**（`specs.py`）：本主题的编号程序从它读常量，
推导过程可复核，**常量本身不可自证**。

★★ **本模块的约定：来源分三类，且必须分开标注**（计划 §C.4 的分组纪律；域 README §「代码约定」）：

  · **第一类 [B]**：字段**逐字来自** `45-kernel-lifecycle/requirement-discovery/code/specs.py`
    的 `QWEN3_30B_A3B_CONFIG` / `PUBLISHED_PARAM_COUNT_B` ⇒ 标 **[B]**，并给出 `# 同值位置：`。
  · **第二类（镜像）**：**不在那份文件里**、经 `hf-mirror.com` **镜像**取得的字段 ⇒ 带强制限定语
    「**经 `hf-mirror.com` 镜像取得，未与官方 repo 比对 sha256**」。
  · **第三类 [C]**：**论文正文里的 hyperparameter**。它不是 `config.json`，而是作者在论文里
    写下的取值 ⇒ 标 **[C]**，给 arXiv 号 + 抓取日 + **原始字节计数**，并按 `CONTRIBUTING` §2
    注明**是否确认同行评审**。

★ **本模块只放本主题真的会拿去算的东西**（数、名、布尔、元组、字典、阈值）。
  `45-…/code/specs.py:107-113` 记着一条教训：**「照抄但不使用」的部分无法被任何断言约束**，
  只会制造变异测试的漏网。⇒ 其它字段的**去向**写在文件尾注里，**不搬进来**。
  描述性文字归**文档**，不归本模块（先例：`45-…/code/specs.py` §4 的「为什么这一节没有散文」）。

★ 每条常量带 `# 来源：` + `[A-E]` 级 + 抓取日；跨主题同值的另带 `# 同值位置：`。
  分级判据见 `CONTRIBUTING` §2（A 厂商规范 / B 产品官方文档 / C 论文 / D 社区 / E 本库推导）。
"""

# ===========================================================================
# 1. ★★ Qwen3-30B-A3B 的 MoE 规格 —— 本主题的**全部真实模型锚点**
# ===========================================================================
#
# ★ **只搬 MoE 相关的字段**（不是整份 config）：
#   本主题的账是 `E` / `top_k` / 每个专家的宽度 / 路由的 α —— `head_dim`、`rope_theta`
#   之类与本主题无关（去向见文件尾注）。搬进来而不用，只会制造变异漏网。

QWEN3_30B_A3B_MOE = {
    "num_experts": 128,             # E
    "num_experts_per_tok": 8,       # top_k（★ 不是 1 —— Switch 论文里 top-1，这个模型是 top-8）
    "moe_intermediate_size": 768,   # **每个专家**的 FFN 宽度（≠ dense FFN 的 intermediate_size）
}
# 来源：Qwen/Qwen3-30B-A3B 的 config.json，**转记自**
#       45-kernel-lifecycle/requirement-discovery/code/specs.py 的 QWEN3_30B_A3B_CONFIG  [B]
#       （该模块自注「经 4 处独立转抄核对」）
# 口径：★ 这是**转记**，不是本域重新溯源。**不改任何一位数字。**
# 同值位置：45-kernel-lifecycle/requirement-discovery/code/specs.py:150-165；
#           80-model-architectures/transformer/code/specs.py:95-103（那一份**只取了其它字段**，
#           并把 `num_experts*`/`moe_intermediate_size`/`router_aux_loss_coef` 的去向写成 `moe`）
#       ★ 本域是**第 3 处副本**。`45-…/code/specs.py:27` 逐字写着
#         「⚠ 复制造成的**漂移风险**：这些值在仓库里有多份副本（见每条的「同值位置」）」。

QWEN3_30B_A3B_DENSE_SIDE = {
    "hidden_size": 2048,
    "intermediate_size": 6144,      # ★ **dense 侧的 FFN 宽度**，不是专家宽度
    "num_hidden_layers": 48,
    "num_attention_heads": 32,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "vocab_size": 151936,
}
# 来源：同上（转记自 `45-…/code/specs.py` 的 `QWEN3_30B_A3B_CONFIG`）  [B]
# 口径：★ `head_dim` 是**原生字段**（Qwen3 的 config 里真的有它，且 `≠ hidden/heads = 64`）。
#       ⚠ 那个陷阱的正文在 `45-…/requirement-discovery/01-model-to-operator-inventory.md:270-276`
#         （正文在彼，此处回链）。本主题**只把它当活跃参数量的一个加项**，不重推那个陷阱。
# ★★ `intermediate_size` 的**语义**在本环境**没有一手来源**（见 `UNSOURCED`）——
#    本模块只把它当**一个数**用（与 `moe_intermediate_size` 比大小），不解释它「是什么」。
# 同值位置：45-kernel-lifecycle/requirement-discovery/code/specs.py:150-165

QWEN3_30B_A3B_PUBLISHED_TOTAL_B = 30.5
# 来源：Qwen/Qwen3-30B-A3B 的**公开参数量** ≈ 30.5 B（总参数量，含全部专家），**转记自**
#       45-kernel-lifecycle/requirement-discovery/code/specs.py 的 PUBLISHED_PARAM_COUNT_B  [B]
#       （该模块 :170-176 逐字：「Qwen3-30B-A3B": 30.5」，并注明「每 token 活跃 ≈ 3 B」）
# ★★ **为什么它重要**：计划 §4 的 **R19** 逐字说「Qwen3-30B-A3B 在本域**不可能有外部锚点**：
#    `specs.PUBLISHED_PARAM_COUNT_B` 里**刻意只有 Llama**」——**那句话对域内那一份成立、
#    对上游这一份不成立**：上游 `45-…/code/specs.py:170-176` **明确有这一行**（[B] 级）。
#    ⇒ 本主题**因此拿到了 Qwen3 的外部锚点**，直接落 **P15**（「每条算出来的账必须配外部锚点」）。
#    ⚠ 这是一条**必须回报控制者的偏差**（见文档 §5 与本 chunk 的报告）。
# 口径：**总参数量**（含嵌入、含全部专家、含注意力），**不是**每 token 活跃量。
#       它是 3 位有效数字（`30.5`），所以判据只能落在**±1%** 档，不能逐位。
# 同值位置：45-kernel-lifecycle/requirement-discovery/code/specs.py:170-176；
#           80-model-architectures/transformer/code/specs.py:170-177（★ 那一份**只有 Llama**，
#           并注明「★ 那一份还有 `"Qwen3-30B-A3B": 30.5` …… **本主题不用它**」）

# ===========================================================================
# 2. 路由器的两个字段：**经镜像取得**（第二类）—— 其中一个是 α 的真实取值
# ===========================================================================
#
# ★★ 这一组是本主题「论文 vs 真实模型」的**唯一对照面**：
#    论文（Switch）说 α 取 `1e-2`，而这个真实模型的 `router_aux_loss_coef` 是 **`1e-3`**。

QWEN3_30B_A3B_ROUTER_FROM_MIRROR = {
    "router_aux_loss_coef": 0.001,  # ★ 真实模型用的 α（= 1e-3）
    "norm_topk_prob": True,         # ★ 字段**存在**且为 true
}
# 来源：Qwen/Qwen3-30B-A3B 的 config.json  **经 `hf-mirror.com` 镜像取得**（第二类）
#       （2026-09-22 直取 `https://hf-mirror.com/Qwen/Qwen3-30B-A3B/resolve/main/config.json`）
# ★★ **未与官方 repo 比对 sha256。** 镜像不是厂商域名，其副本与官方 repo 的一致性
#    **在本环境无法复核**（`huggingface.co` 直连不可达）。镜像副本的字节哈希
#    （**镜像的**，不是官方 repo 的）：sha256[:16] = 2850ddb3bf7aecad（963 B）
#    —— ★ 与 `80-model-architectures/attention-variants/code/specs.py:75` 记的
#    **是同一份副本**（两处独立抓取，字节哈希一致）。
#       ⚠ **这两个字段都不在** `45-…/code/specs.py` 里（计划 §C.4 把 `router_aux_loss_coef`
#         明确列进「不在那份文件里」的那一组）⇒ 它们**只能**走镜像这条限定路径。
# 口径：★★ `norm_topk_prob` **只记字段名与取值**。它的**语义**在本环境**没有一手来源**
#       （`transformers` 的源码不在本环境）⇒ 本模块**不推断**它「做了什么」，
#       它只用来钉住「这个模型的 config 里有这么一个字段、且为 true」这一事实。
#       ⚠ 这条限定是刻意的：把字段名**读成**机制从句，正是本库禁止的那一类错误。

# ===========================================================================
# 3. Switch Transformer 的 hyperparameter（**[C] 级：论文正文**）
# ===========================================================================
#
# ★ 抓取（2026-09-22，`curl -sSL`，`WebFetch` 被策略拦截）：`https://ar5iv.labs.arxiv.org/html/2101.03961`
#   **413955 B**，`sha256[:16] = e9cec95e4c274436`（★ 与计划 §C.1 的记法逐位一致）。
# 评审状态：★ abs 页的 `comments` 逐字为 `JMLR` ⇒ **已确认同行评审**（按页面实测填，不凭印象）。

SWITCH_AUX_ALPHA = 1e-2
# 来源：Switch Transformer（arXiv 2101.03961）§2.2 逐字「throughout this work we use an
#       `\alpha=10^{-2}` which was sufficiently large to ensure load balancing while small
#       enough to not to overwhelm the primary cross-entropy objective」  [C]
# ★ 原始字节计数：`\alpha=10^{-2}` = **2**（`alttext` 与 annotation 各一次，下同）。
# 口径：★★ 这是**论文选定的值**，**不是**本域或任何真实模型的取值 —— 与
#       `QWEN3_30B_A3B_ROUTER_FROM_MIRROR["router_aux_loss_coef"]`（`1e-3`）**相差 10 倍**。

SWITCH_AUX_ALPHA_SWEEP = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5)
# 来源：同上一段逐字「We swept hyper-parameter ranges of `\alpha` from `10^{-1}` to `10^{-5}`
#       in powers of 10 and found `10^{-2}` balanced load quickly without interfering with
#       training loss.」  [C]
# ★ 原始字节计数：`10^{-1}` = 2、`10^{-2}` = 4、`10^{-5}` = 2。
# ★★ 口径：**真实模型的 `1e-3` 落在这条扫描带内**（见 `03_balance_granularity.py` 的 `B.2`）
#       —— 这是「论文的结论 vs 真实取值」之间唯一一条**可机械判定**的关系。

SWITCH_CAPACITY_FACTOR_FINDING = (1.0, 1.25)
# 来源：Switch Transformer Table 1 的 finding (3) 逐字「Switch Transformers perform better at
#       lower capacity factors (`1.0, 1.25`).」  [C]
# ★★ 口径（**必须与 X6 一起写**）：这条 finding 属于 **Switch vs MoE vs Dense 的比较语境**
#       （同一段逐字：「A head-to-head comparison of the Switch Transformer and the MoE
#       Transformer is presented in Table 1」「All models were trained for the same number of
#       steps on identical hardware」）⇒ **不可读成普遍结论。**

SWITCH_MOE_SPEED_BY_CF = {2.0: 840.0, 1.25: 790.0}
# 来源：Switch Transformer §2.3 逐字「Note that the MoE model going from capacity factor 2.0 to
#       1.25 actually slows down (`840 to 790`) in the above experiment setup, which is
#       unexpected.」  [C]
# ★ 原始字节计数：`actually slows down (840 to 790)` = 1；`which is unexpected` = 1。
# ★★ 口径：**主语是 MoE 模型，不是 Switch**（逐字 `the MoE model going from capacity factor`）；
#       而且原文**自承 unexpected**。⚠ 同一段紧接着的脚注逐字：「Note that speed measurements
#       are both a function of the algorithm and the implementation details.」⇒
#       这两个数**只能当「原文观察到的一次现象」**引用，不能当架构结论。
#       ⇒ 本模块收录它们是为了**钉住**——不是为了让程序拿它们做推断。

SWITCH_EXPLORATION_NLL = {
    "argmax": -1.471, "sample_softmax": -1.570,
    "input_dropout": -1.480, "input_jitter": -1.468,
}
# 来源：Switch Transformer Appendix C 的 Table 11「Router Exploration Strategies」逐字四行
#       （`Argmax -1.471` / `Sample softmax -1.570` / `Input dropout -1.480` /
#        `Input jitter -1.468`），表题逐字：「Quality of the Switch Transformer, measured by the
#        **negative log perplexity**, under different randomness-strategies for selecting the
#        expert (**lower is better**). There is no material speed performance difference between
#        the variants.」  [C]
# ★★ 口径三条（★ 第 ③ 条是**本域新发现的论文内部不一致**，见 `01_gating_routing.py` 的 `C.4`）：
#       ① 四者极差 **0.102** ⇒ 表题逐字「**no material speed performance difference**」
#          ——「四个策略没有实质差别」是**唯一在两个方向下都成立**的结论。
#       ② 这张表**不是** GShard 的 `Random routing`（那一条是「按 g₂ 概率决定要不要派第 2 个
#          专家，为了省容量」，见 `01_gating_routing.py` 的 `C` 段）——**两个机制同名不同指**。
#       ③ ★★ **表题与表格/正文的方向冲突**：表题逐字 `(lower is better)`，但**同一张表**的
#          列头逐字带向上箭头 `Model Quality (Neg. Log Perp.) ( ↑ )`，而正文逐字
#          「Throughout this work, we use **input jitter** to inject noise as we have found it to
#          **empirically perform the best**」——`input jitter` 是四值里**最大**的那个。
#          ⇒ 按表题读，第一名是 `argmax`；按列头 ↑ / 正文读，第一名是 `input jitter`。
#          ⇒ **本域不替论文选方向**，只把「两种读法给出不同第一名」钉住
#            （机械形态见 `SWITCH_EXPLORATION_DIRECTION_FROM_CAPTION`）。
#       ⚠ **不要**把这条转述成「jitter 最好」——那个说法**依赖你选了哪个方向**。

# ===========================================================================
# 4. GShard 与 Expert Choice 的 hyperparameter（**[C] 级**）
# ===========================================================================
#
# ★ 抓取（2026-09-22）：GShard `https://ar5iv.labs.arxiv.org/html/2006.16668`，**451967 B**，
#   `sha256[:16] = 1a5e04e9d1604e8c`（★ 与计划 §C.1 逐位一致）。
#   评审状态：**无 venue 线索 ⇒ 未确认同行评审**。
#
# ★ Expert Choice：`https://ar5iv.labs.arxiv.org/html/2202.09368`，**175642 B**，
#   `sha256[:16] = 059b2abad496bcb1`。评审状态：abs 页**无 `Comments:` 也无
#   `Journal reference:`** ⇒ **未确认同行评审**（按页面实测填）。
#   ⚠★ **这一篇不在计划 §C 的论文清单里**（计划 §C.7 逐字称「仍待补：**无**」）——
#     但计划 §3 Chunk 3 把「**token choice vs expert choice**」写进了 `01` 的核心内容。
#     ⇒ 本域补取这一篇；**这是一条必须回报控制者的偏差**（来源缺口，不是缺口「无」）。

SWITCH_EXPLORATION_DIRECTION_FROM_CAPTION = "lower_is_better"
# 来源：Switch Transformer Appendix C 的 Table 11 表题逐字「…under different
#       randomness-strategies for selecting the expert (**lower is better**).」  [C]
# ★★ 语义：它**只记录表题声称的方向**，**不是**本域认可的方向 ——
#    它与**同一张表**的列头逐字 `Model Quality (Neg. Log Perp.) ( ↑ )`、以及正文那句
#    「we use **input jitter** to inject noise as we have found it to empirically perform the
#    best」**互相冲突**（`input jitter` 的四值里**最大**的那个）。
#    ⇒ 本域把它当成**一条可判定的冲突事实**来用（`01_gating_routing.py` 的 `C.4` 用
#      「两种读法给出不同的第一名」证明冲突存在），**不拿它做任何推断**。
#    ⚠ 这与 `UNSOURCED` 的处置不同：它不是「查不到」，而是「**查到了两个互斥的说法**」。

GSHARD_NUM_SELECTED_EXPERTS = 2
# 来源：GShard（arXiv 2006.16668）§2.2 逐字「each token is dispatched to **at most two experts**」  [C]
# ★ 原始字节计数：`at most two experts` = 2；`fractional capacity of each expert` = 1；
#   `2N/(G\cdot E)`（alttext）= 2；`O(N/E)`（alttext）= 2；`degenerates into a zero vector` = 1；
#   `passed on to the next layer via residual connections` = 1；`Random routing` = 1。
# 口径：★★ GShard 给的是**渐进记号**（`O(N/E)`）与「**fractional capacity** `2N/(G·E)`」，
#       **不是**「capacity factor」这个词 —— 逐字计数：`capacity factor` 在 GShard 里 = **0**，
#       在 Switch 里 = **11**（原始字节）。⇒ **这个词是 Switch 的**（计划 §C.1 的 M1）。

EXPERT_CHOICE_CAPACITY_FACTOR = 2
# 来源：Expert Choice（arXiv 2202.09368）Table 2 表题逐字「Expert choice with capacity factor
#       of **2** (EC-CF2) outperforms Top-1 gating in Switch Transformer (ST) and top-2 gating in
#       GShard (GS) on GLUE and SuperGLUE tasks.」  [C]
# ★ 原始字节计数：`the capacity factor` = 8；`expert choice` = 21；
#   `guarantees perfect load balancing` = 1；
#   `a fixed expert capacity (i.e. the number of tokens each expert can take)` = 1。
# ★★ 口径：**同一个词 `capacity factor`，在这篇里指的是「一个 token 平均被几个专家服务」**
#       （逐字：「The capacity factor `c` denotes on average how many experts are utilized by a
#       token.」），而在 Switch 里指的是「在均分之上**额外开出多少缓冲**」。
#       ⇒ 两者的闭式**恰好相同**（都是 `(tokens/experts) × factor`），**语义不同**：
#          **同名不同指**（本库的经典错误模式）。判据见 `02_capacity_factor.py` 的 `C` 段。

# ===========================================================================
# 5. UNSOURCED —— 待办清单，不是免责声明
# ===========================================================================
#
# ★ CONTRIBUTING §2：「不得**既保留数字又记 UNSOURCED**」—— 下面每一项都**没有**出现在
#   本模块的正文常量里。

UNSOURCED = (
    # 每一项是一个**稳定标识符**（不是散文）。
    "qwen3_mlp_only_layers_semantics",   # config 里有 `mlp_only_layers: []`、`decoder_sparse_step: 1`、
                                         # `max_window_layers: 48`、`output_router_logits: false` 四个字段，
                                         # 但它们在**本环境没有一手语义来源**（`transformers` 源码不在本环境）
                                         # ⇒ 本域**只字不提它们的含义**，更不据此推断任何机制从句。
    "qwen3_intermediate_size_semantics", # `intermediate_size = 6144` 在 MoE 模型里的**语义**
                                         # （是共享专家宽度？是 dense 层宽度？）**没有一手来源**。
                                         # ⇒ 本域只用它做**整数等式**（`6144 = 8 × 768`）与比大小，
                                         #    **不解释它「是什么」**。
    "qwen3_shared_expert",               # ★★ 计划 §4 的 **R11**：共享专家的规格字段
                                         # 在 `45-…/code/specs.py:123-127` 里逐字记载为
                                         # 「**在检索到的片段里没有**」⇒ **共享专家不在本域**，
                                         # 本主题 `03` 因此收窄为「`E` / `top_k` /
                                         # `moe_intermediate_size` 三者之间**可手算**的关系」。
    "qwen3_active_param_published",      # `45-…/code/specs.py:176` 有一句「每 token 活跃 ≈ 3 B」，
                                         # 但它是**一位有效数字**、且**没有给口径**（含不含嵌入？
                                         # 含不含 lm_head？）⇒ 本域**不拿它当外部锚点**，
                                         # 只把总参数量 `30.5 B` 当锚点（那一条有明确口径）。
    "switch_capacity_factor_definition", # ★ `capacity factor` 的**定义式**（Switch 式 (3)）已在本模块
                                         # 之外——它的正文在 `80-model-architectures/moe/02-capacity-factor.md`。
                                         # 本项记录的是：论文里那条式子**在原始字节上不是一个可检索的
                                         # 字面量**（`expert capacity=` 计数 **0**，被 MathML 切开），
                                         # ⇒ 引用它只能引**散文那一句**（计数 1）。见 `02` 的表 B。
)

# ===========================================================================
# 尾注：本主题**刻意不搬**的字段（它们的正文在别处）
# ===========================================================================
#
#   · `head_dim` / `rope_theta` / `max_position_embeddings`
#                                   → `transformer` / `positional-encoding/code/specs.py`
#   · `rms_norm_eps` / `hidden_act` → `normalization` / `transformer`
#   · `num_attention_heads` / `num_key_value_heads` / `vocab_size` / `hidden_size`
#                                   → `attention-variants/code/specs.py` / `transformer/code/specs.py`
#       ⚠ **例外**：`hidden_size` / `intermediate_size` / `num_hidden_layers` / `head_dim` /
#         `vocab_size` 在**本模块里出现了**（`QWEN3_30B_A3B_DENSE_SIDE`）—— 理由是
#         **`03` 的参数量锚点需要它们**（不算出一个总数，就没法跟 `30.5 B` 对）。
#         ★ 这与「照抄但不使用」相反：它们**每一位都被 `D.4c` 钉住并被 `B.1` 用上**。
#   · MoE 的 FLOPs / 权重流量 / 算术强度 / 拐点批大小 / permute / grouped GEMM / slot mask
#                                   → `40-kernels/moe-kernels/01,02,03` 与它们的 `code/verify.py`
#       ★★ **这是本主题最大的禁区**（计划 §2 的边界表）。本模块**一个 FLOPs 常量都没有**。
#
#   （★ R3 的裁定：**每主题一份，每份只放本主题用得到的常量**，跨主题同值的常量各写一份、
#     靠 `# 同值位置：` 互指。本库 40 个 `code/` 目录**没有任何跨主题共享辅助模块的先例**，
#     `CONTRIBUTING:36`「`code/` 必须自包含」是**全库规则**。）

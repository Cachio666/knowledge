"""80-model-architectures/transformer 的外部事实常量源。

本模块是 `CONTRIBUTING` §2 约定的**共享事实源**（`specs.py`）：本主题的编号程序从它读常量，
推导过程可复核，**常量本身不可自证**。

★★ **本模块最重要的约定：来源分两类，且必须分开标注**（计划 §C.4；这是事实层审查订正过的）：

  · **第一类**：字段**逐字来自** `45-kernel-lifecycle/requirement-discovery/code/specs.py`
    的 `LLAMA3_8B_CONFIG` / `QWEN3_30B_A3B_CONFIG` ⇒ 标 **[B]**，并给出 `# 同值位置：`。
  · **第二类**：**不在那份文件里**的字段 ⇒ 带强制限定语
    「**经 `hf-mirror.com` 镜像取得，未与官方 repo 比对 sha256**」。

  ⚠ 原稿把两者合并，导致 9 个字段被**静默升格**。本模块按字段分开，**逐条可核对**。

★ 为什么本主题**只放这两个 config**：`45-kernel-lifecycle` 已把这两个模型核实到 [B] 级
  （「经 HuggingFace 与 torchtune 双向核对」/「经 4 处独立转抄核对」）。**其余任何模型的
  config 在取到一手来源之前都不得出现在本域**（计划 §C.0 的硬约束）——凭记忆写就是自造真值。

★ **本模块只放本主题真的会拿去算的东西**（数、名、布尔、阈值、URL）。
  `45-…/code/specs.py:107-113` 记着一条教训：**「照抄但不使用」的部分无法被任何断言约束**，
  只会制造变异测试的漏网。⇒ 其它字段的**去向**写在每条下面的注释里，**不搬进来**。
  描述性文字归**文档**，不归本模块（先例：`45-…/code/specs.py` §4 的「为什么这一节没有散文」）。

★ 每条常量带 `# 来源：` + `[A-E]` 级 + 抓取日；跨主题同值的另带 `# 同值位置：`。
  分级判据见 `CONTRIBUTING` §2（A 厂商规范 / B 产品官方文档 / C 论文 / D 社区 / E 本库推导）。
"""

# ===========================================================================
# 1. 两个真实模型的规格 —— ★ 分三组，**不得混为一谈**
# ===========================================================================
#
# 三组的判据是「**这个字段的取值是从哪来的**」：
#   · `_FROM_UPSTREAM`：[B]，逐字来自 `45/…/code/specs.py`（那两份 config 已核实）
#   · `_FROM_MIRROR`  ：经 `hf-mirror.com` 镜像取得，**未与官方 repo 比对 sha256**
#   · `_DERIVED`      ：**推导值**（[E]），且必须给出推导式
#
# ★ 三组**必须构成一个划分**（两两不交、并集 = 配置的全部字段）。这是一个可机械判定的
#   不变量，Chunk 1 的程序会对它下断言 —— 它挡住「某个字段被静默换组」这类漂移。

LLAMA3_8B_FIELDS_FROM_UPSTREAM = {
    "num_hidden_layers": 32,
    "hidden_size": 4096,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "intermediate_size": 14336,
    "vocab_size": 128256,
}
# 来源：meta-llama/Meta-Llama-3-8B 的 config.json，**转记自**
#       45-kernel-lifecycle/requirement-discovery/code/specs.py 的 LLAMA3_8B_CONFIG  [B]
#       （该模块自注「经 HuggingFace 与 torchtune 的 llama3_8b() 构造器双向核对」）
# 口径：★ 这是**转记**，不是本域重新溯源。**不改任何一位数字。**
# 同值位置：45-kernel-lifecycle/requirement-discovery/code/specs.py:131-145
#       ★ 本域是**第 2 处副本**。那一份的 :27 逐字写着「⚠ 复制造成的**漂移风险**：
#         这些值在仓库里有多份副本（见每条的「同值位置」）」—— 本模块遵守同一条约定。

LLAMA3_8B_FIELDS_FROM_MIRROR = {
    "tie_word_embeddings": False,
    "hidden_act": "silu",
}
# 来源：meta-llama/Meta-Llama-3-8B 的 config.json  **经 `hf-mirror.com` 镜像取得**
#       （本模块 2026-09-22 直取 `https://hf-mirror.com/NousResearch/Meta-Llama-3-8B/resolve/main/config.json`）
# ★★ **未与官方 repo 比对 sha256。** 镜像不是厂商域名，其副本与官方 repo 的一致性
#    **在本环境无法复核**（`huggingface.co` 直连不可达；`meta-llama/Meta-Llama-3-8B`
#    经镜像也返回「your request … has been rejected by the repo's authors」，即**门控**）。
#    镜像副本的字节哈希（**镜像的**，不是官方 repo 的）：
#      NousResearch/Meta-Llama-3-8B  sha256[:16] = 2430cee764b6530f（654 B）
#      unsloth/llama-3-8b            sha256[:16] = f39827328c0284ab（768 B）
# ★★ **必须保留「未核实」的标注**：上游 `45-…/code/specs.py:128` 逐字写着
#    「② `tie_word_embeddings` **未核实**。本模块的参数量估计**按不共享**
#    （embed + lm_head 各算一次）」。
#    ⇒ 本域虽在**镜像**里看到 `false`（两个独立镜像一致），但
#      **不得把它静默升格为「已核实事实」**；引用它时必须带上本限定语。
#      （原稿据此「解释」了 `45-…/code/01_operator_inventory.py:549` 的 `8029995008`，
#       那一步**不成立**——它把未核实的前提当成了已核实。）

LLAMA3_8B_FIELDS_DERIVED = {
    "head_dim": 128,          # = hidden_size // num_attention_heads = 4096 // 32
}
# 来源：**本域推导**  [E]，推导式同上
# 口径：★★ **Llama-3-8B 的 `config.json` 里没有 `head_dim` 字段** —— 经**两个独立镜像**
#       （`NousResearch/Meta-Llama-3-8B`、`unsloth/llama-3-8b`）逐个字段核对确认。
#       所以这里的 128 是**推导值**，不是原生字段。
# ★ 上游的一处**内部矛盾**（计划 §4 R13，**本域记录、不改上游**）：
#     ① `45-…/code/specs.py:139` 的 `LLAMA3_8B_CONFIG` 把 `head_dim` 写成了**普通字段**；
#     ② 而 `45-…/01-model-to-operator-inventory.md:276`（§4.5）与 `specs.py:146-148` 都建议
#        「**只用 config 里的 `head_dim` 字段**，永远不做那个除法」。
#     ⇒ 照那条建议办事，实现者在 **Llama-3-8B** 上**会找不到该字段**。
#       （在 Qwen3-30B-A3B 上它确实存在，见下 —— 所以那条建议**不是普遍成立**的。）
# ★ 与计划 §C.4 的一处**不符**（本模块按**文件实测**归类，并把差异记在此处）：
#     §C.4 的表格把 `head_dim` 列进「**不在** `45/…/specs.py` 里」的一组，
#     但该文件 `:139`（Llama）与 `:159`（Qwen3）两行**逐字**有 `"head_dim": 128,`。
#     ⇒ 本模块把 Qwen3 的 `head_dim` 归入 `_FROM_UPSTREAM`；Llama 的那一个归入本组，
#       理由是**真实 config 里没有它**（这才是 §C.4 真正想说的那件事）。

QWEN3_30B_A3B_FIELDS_FROM_UPSTREAM = {
    "num_hidden_layers": 48,
    "hidden_size": 2048,
    "num_attention_heads": 32,
    "num_key_value_heads": 4,
    "head_dim": 128,          # ★ 原生字段（与 Llama 相反）
    "intermediate_size": 6144,
    "vocab_size": 151936,
}
# 来源：Qwen/Qwen3-30B-A3B 的 config.json，**转记自**
#       45-kernel-lifecycle/requirement-discovery/code/specs.py 的 QWEN3_30B_A3B_CONFIG  [B]
#       （该模块自注「经 4 处独立转抄核对」）
# 口径：★ `num_key_value_heads` = 4 ⇒ **GQA 8:1**（32 个 Q 头对 4 个 KV 头）。
#       ★★ `head_dim` = **128**，而 `hidden_size // num_attention_heads = 2048 // 32 = 64`
#         —— **两者不等**。按后者算会把 attention 的宽度**少算一半**。
#         这是本主题最容易算错的一个字段（先例：`45-…/01:270-276`）。
#       ★★ 本模型是 **MoE**：`intermediate_size` 的语义是「**非 MoE 的 FFN 宽度**」，
#         **不等于专家宽度**（专家宽度在 `moe_intermediate_size = 768`，属 `moe` 主题）。
#         ⚠ 把 dense FFN 公式套在它身上会得到 `1.444 B`，而真值 ≈ `30.5 B`（计划 §0.1 的 P15）。
# 同值位置：45-kernel-lifecycle/requirement-discovery/code/specs.py:150-165

QWEN3_30B_A3B_FIELDS_FROM_MIRROR = {
    "tie_word_embeddings": False,
    "hidden_act": "silu",
}
# 来源：Qwen/Qwen3-30B-A3B 的 config.json  **经 `hf-mirror.com` 镜像取得**
#       （本模块 2026-09-22 直取 `https://hf-mirror.com/Qwen/Qwen3-30B-A3B/resolve/main/config.json`）
# ★★ **未与官方 repo 比对 sha256**（同 LLAMA3_8B 那一条的全部限定）。
#       镜像副本的字节哈希：sha256[:16] = 2850ddb3bf7aecad（963 B）
# ★★ **保留「未核实」的标注**：上游 `45-…/code/specs.py:128` 逐字的那条限定语
#       **同时覆盖 Qwen3 这个字段**（它写在那份模块的模型规格节的开头，不是只对 Llama 说的）。

QWEN3_30B_A3B_FIELDS_DERIVED = {}        # 本模型**没有**推导字段（本主题用到的字段全是原生的）

# 合并视图：**编号程序直接用这两个字典**，不要自己拼。
LLAMA3_8B_CONFIG = {**LLAMA3_8B_FIELDS_FROM_UPSTREAM,
                    **LLAMA3_8B_FIELDS_FROM_MIRROR,
                    **LLAMA3_8B_FIELDS_DERIVED}
QWEN3_30B_A3B_CONFIG = {**QWEN3_30B_A3B_FIELDS_FROM_UPSTREAM,
                        **QWEN3_30B_A3B_FIELDS_FROM_MIRROR,
                        **QWEN3_30B_A3B_FIELDS_DERIVED}

# 逐字段的**来源组**标签（Chunk 1 的规格表按它标注「config 原生 / 推导」）。
# 判据是机械的：`"upstream"` 与 `"mirror"` 两种都属于**原生**（真实 config 里真有这个字段），
# 只有 `"derived"` 是**推导**。★ 让程序用这张表而不是硬编码字段名。
MODEL_FIELD_GROUP = {
    "Meta-Llama-3-8B": {**{k: "upstream" for k in LLAMA3_8B_FIELDS_FROM_UPSTREAM},
                        **{k: "mirror" for k in LLAMA3_8B_FIELDS_FROM_MIRROR},
                        **{k: "derived" for k in LLAMA3_8B_FIELDS_DERIVED}},
    "Qwen3-30B-A3B": {**{k: "upstream" for k in QWEN3_30B_A3B_FIELDS_FROM_UPSTREAM},
                      **{k: "mirror" for k in QWEN3_30B_A3B_FIELDS_FROM_MIRROR},
                      **{k: "derived" for k in QWEN3_30B_A3B_FIELDS_DERIVED}},
}
# 口径：★ 本表是**本域对来源的再组织**（[E]），不是任何来源里的东西。
#       「原生」= 真实 `config.json` 里有这个字段；「推导」= 没有，需要算出来。

# ★ 本主题**刻意不搬**的字段（它们属别的主题的 `specs.py`，照抄进来只会制造变异漏网）：
#   · `rope_theta` / `max_position_embeddings`       → `positional-encoding/code/specs.py`
#   · `rms_norm_eps`                                 → `normalization/code/specs.py`
#     （★ `hidden_act` **留在本主题**：它是「FFN 取什么形态」的一部分，属 `02-ffn-shapes`。）
#   · `num_experts` / `num_experts_per_tok` /
#     `moe_intermediate_size` / `router_aux_loss_coef` → `moe/code/specs.py`
#   · `sliding_window` / `use_sliding_window`         → `attention-variants/code/specs.py`
#   （★ 计划 §3 的 R3 裁定：**每主题一份，每份只放本主题用得到的常量**，
#     跨主题同值的常量各写一份、靠 `# 同值位置：` 互指。本库 40 个 `code/` 目录
#     **没有任何跨主题共享辅助模块的先例**，`CONTRIBUTING:36`「`code/` 必须自包含」是**全库规则**。）
#
# ★ 本主题也**不写**「每参数 20 token」这类训练侧经验值（P9 同名不同指：那是训练的结论，
#   本域是结构侧），也**不放占位常量**（值为 `None` 的常量不可能被任何断言约束）。


# ===========================================================================
# 2. 参数量对齐用的**外部锚点**（P15：从结构算出的账必须有外部锚点）
# ===========================================================================

PUBLISHED_PARAM_COUNT_B = {
    "Meta-Llama-3-8B": 8.03,
}
# 来源：meta-llama/Meta-Llama-3-8B 的公开参数量 ≈ 8.03 B  [B]
# 口径：**总参数量**（含嵌入；因 `tie_word_embeddings` 见上，`lm_head` **另算一次**）。
# 同值位置：45-kernel-lifecycle/requirement-discovery/code/specs.py:170-178
#       ★ 那一份还有 `"Qwen3-30B-A3B": 30.5`（总参数量，含全部专家）。**本主题不用它**：
#         从 config 反算参数量在这里**只对 Llama-3-8B 做一次对照**（Chunk 1 表 A ⑥），
#         MoE 的参数量按 `E` 算，属 `moe` 主题（P15 的现场就在那一侧）。
# ⚠ 与 `45-…/code/01_operator_inventory.py:549` 的**逐位断言** `8029995008` 同源：
#   本主题**只把它当 `03` 篇的一次对照，不重做整套算子清单**（那是 `45` 的正文，见域 README 边界表）。


# ===========================================================================
# 3. FFN 形态的前提（X9 / X10）—— ★ 一个来自论文，一个是**论文之外的约定**
# ===========================================================================

SWIGLU_DFF_REDUCTION_FACTOR = 2.0 / 3.0
# 来源：SwiGLU 论文（arXiv 2002.05202）逐字
#       「we reduce the number of hidden units d_{ff} … by a factor of \frac{2}{3}」  [C]
#       抓取：`https://ar5iv.labs.arxiv.org/html/2002.05202`，2026-09-22，
#             sha256[:16] = 54d6e4f552c74d53（计划 §C.2）
# 评审状态：★ **无 venue 线索 ⇒ 未确认同行评审**（按 abs 页元数据实测填，**不凭印象补**）
# 口径：★★ 论文里**没有** `8d/3` —— **原始字节计数**：`8d/3` = **0** 次、`\frac{8d}{3}` = **0** 次
#       （做过对照检验：已知存在的 `\frac{2}{3}` 计到 **2** ⇒ 「0」可信）。
#       ⇒ `8d/3` 是**推导**，不是引文。正文引用它时必须写明这一点。

FFN_BASELINE_D_FF_RATIO = 4
# 等级：**无** —— 它不是「来源值」，而是**一条前提（约定）**，所以不给 A–E 里的任何一级。
# 口径：★★ `8d/3 = (2/3) × 4d` 里的那个 `4` 来自「**原始两层 FFN 取 `d_ff = 4d`**」这条
#       **论文之外的约定**。SwiGLU 论文**从未**写过 `4d`，也**从未**写过 `8d/3`。
#       ⇒ 正文写 `d_ff = 8d/3` 时，**必须同时写出这条前提**，否则读者无法复核这个数
#         （计划 §3 Chunk 1 表 A ④ 已把「前提必须写明」写成硬要求）。
# ★ 三条**可从 config 直接读出的比值**（**推导值，不放在本模块里** —— CONTRIBUTING §2：
#   「只放从来源抄来的数，不放推导值」）。程序每次自己算，并与这三个数对照：
#     Llama-3-8B   intermediate_size / hidden_size = 14336 / 4096 = 3.5
#     Qwen3-30B-A3B                                 = 6144 / 2048  = 3.0
#   ⇒ ★★ **没有一个是 `4`，也没有一个是 `8/3 ≈ 2.667`。**
#      拿「Llama 用 `8d/3`」去解释会**直接算错**（计划 §C.3 的 X10）。


# ===========================================================================
# 4. UNSOURCED —— 待办清单，不是免责声明
# ===========================================================================
#
# ★ CONTRIBUTING §2：「不得**既保留数字又记 UNSOURCED**」—— 下面每一项都**没有**出现在
#   本模块的正文常量里。

UNSOURCED = (
    # 每一项是一个**稳定标识符**（不是散文）。
    "llama2_70b_config",       # Llama-2-70B 的 config：D21 用它做过第二条锚点（68.975 B / 公开 68.98），
                               # 但计划 §C.0 的硬约束是「本域只能引用那两个已核实的 config」
                               # ⇒ **本域不写它**。要用必须在能连通 HF 的环境重取并记来源。
    "qwen3_30b_a3b_shared_expert_size",   # Qwen3-30B-A3B 的**共享专家**规格：一手来源在本环境不存在
                               # （`45-…/code/specs.py:123-127` 逐字记载该字段「在检索到的片段里**没有**」）。
                               # 它属 `moe` 主题的边界（计划 §4 R11）—— 本模块只记录，不猜。
    "llama3_8b_head_dim_native_field",    # 「Llama-3-8B 的 config 里**有** `head_dim`」这个说法：
                               # 两个独立镜像都**否证**了它（该字段不存在）⇒ 本域按**推导**处理。
                               # 但「上游为何把它写成普通字段」这件事本身**无人解释**（R13）⇒ 挂在这里。
)

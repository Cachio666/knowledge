"""80-model-architectures/normalization 的外部事实常量源。

本模块是 `CONTRIBUTING` §2 约定的**共享事实源**（`specs.py`）：本主题的编号程序从它读常量，
推导过程可复核，**常量本身不可自证**。

★★ **本模块的约定：来源分三类，且必须分开标注**（计划 §C.4 的分组纪律；域 README §「代码约定」）：

  · **第一类（镜像）**：**不在** `45-kernel-lifecycle/requirement-discovery/code/specs.py` 里、
    经 `hf-mirror.com` **镜像**取得的 config 字段 ⇒ 必须带强制限定语
    「**经 `hf-mirror.com` 镜像取得，未与官方 repo 比对 sha256**」。
    ★ 本模块逐个词频核实过：`rms_norm_eps` 与 `layer_norm_eps` 在
      `45-…/code/specs.py` 里各出现 **0 次**（那份文件的 521 行里逐字没有这两个词）
      ⇒ 它们**只能**走镜像这条限定路径，与计划 §C.4 的分组一致。
  · **第二类 [C]**：**论文正文里的逐字**。给 arXiv 号 + 抓取日 + **原始字节计数** +
    `sha256[:16]`，并按 `CONTRIBUTING` §2 注明**是否确认同行评审**。
    ★ 这些是**字符串常量**，程序对它们做**逐位相等**的断言 —— 引用判据（表 B）本身
      「不能被程序判定」，但**「我抄的这一句是不是原文」可以被程序钉住**（两者不是一回事）。
  · **第三类 [E]**：本域自己写出的**定义**（不是任何来源里的东西）⇒ 标 **[E]**，并写明它为什么抄不到。

★ **本模块只放本主题真的会拿去算的东西**（数、字符串、元组）。
  `45-…/code/specs.py:107-113` 记着一条教训：**「照抄但不使用」的部分无法被任何断言约束**，
  只会制造变异测试的漏网。⇒ 别的字段的**去向**写在文件尾注里，**不搬进来**。
  描述性文字归**文档**，不归本模块（先例：`45-…/code/specs.py` §4 的「为什么这一节没有散文」）。

★ 每条常量带 `# 来源：` + `[A-E]` 级 + 抓取日；跨主题同值的另带 `# 同值位置：`。
  分级判据见 `CONTRIBUTING` §2（A 厂商规范 / B 产品官方文档 / C 论文 / D 社区 / E 本库推导）。
"""

# ===========================================================================
# 1. 两个真实模型的 `rms_norm_eps`（**经镜像取得**）—— 本主题的唯一真实模型锚点
# ===========================================================================
#
# ★ 为什么本主题需要它：判据 ② 说「`eps` 才是不变性的破坏者」（计划 §0.7-D 的 **D7**）。
#   那么**真实模型把 eps 取在哪一档**就直接决定了「这个不变性在工程上是否精确」。
#   这两个数是唯一的真实锚点 —— 没有它们，判据 ② 就只是一个玩具量级的演示。

LLAMA3_8B_RMS_NORM_EPS = 1e-05
# 来源：meta-llama/Meta-Llama-3-8B 的 config.json，**经 `hf-mirror.com` 镜像取得**  [镜像]
#       2026-09-22 直取 `https://hf-mirror.com/NousResearch/Meta-Llama-3-8B/resolve/main/config.json`
#       （★ 最简形式 `curl -sS` 返回的是 307 的**跳转正文**，必须加 `-L`；
#        计划 §0.3 说「`-L` 的必要性因 URL 而异，先试最简形式」—— 这里试完的结论是**要加**。）
# ★★ **未与官方 repo 比对 sha256。** 镜像不是厂商域名，其副本与官方 repo 的一致性
#    **在本环境无法复核**（`huggingface.co` 直连不可达）。
#    镜像副本的字节哈希（**镜像的**，不是官方 repo 的）：
#      sha256[:16] = 2430cee764b6530f（654 B）
#    ★ 与 `80-model-architectures/transformer/code/specs.py:66` 记的
#      **是同一份副本**（两处独立抓取，字节哈希与字节数都一致）。
# 口径：★ 字段名逐字是 `rms_norm_eps`（**不是** `layer_norm_eps`）。
#       原始字节计数：`rms_norm_eps` = 1、`layer_norm_eps` = **0**。
#       ⇒ 本模块**只记字段名与取值**，**不推断**「这个模型内部用了哪个算子」
#         （那是「把字段名读成机制从句」，本库禁止；真正的依据在别处，本域也不据此下结论）。
# 同值位置：80-model-architectures/transformer/code/specs.py:151-153（那一份**刻意不搬**它，
#           并把去向写成 `normalization/code/specs.py` —— 本模块就是那个去向）。
#           ⚠ `45-kernel-lifecycle/requirement-discovery/code/specs.py` 里**没有**这个字段
#             （逐字计数 0），所以**不存在**上游 [B] 副本可指。

QWEN3_30B_A3B_RMS_NORM_EPS = 1e-06
# 来源：Qwen/Qwen3-30B-A3B 的 config.json，**经 `hf-mirror.com` 镜像取得**  [镜像]
#       2026-09-22 直取 `https://hf-mirror.com/Qwen/Qwen3-30B-A3B/resolve/main/config.json`
# ★★ **未与官方 repo 比对 sha256**（同上一段的全部限定）。
#       镜像副本的字节哈希：sha256[:16] = 2850ddb3bf7aecad（963 B）
#    ★ 与 `moe/code/specs.py:95`、`attention-variants/code/specs.py:75` 记的是**同一份副本**。
# 口径：字段名逐字 `rms_norm_eps`（1 次）、`layer_norm_eps`（**0** 次）。
#       ★★ **两个模型相差 10 倍**（`1e-5` vs `1e-6`）—— 这与 §C.4 的
#         「`rope_theta` 相差 2 倍」是同一类事实：**同一个字段名、两个不同的取值**，
#         引用时必须写出是哪一个模型。
# 同值位置：80-model-architectures/{moe,attention-variants}/code/specs.py（同一份镜像副本，
#           但**只在本模块**把它当「归一化的 eps」用）。

# ===========================================================================
# 2. QK-norm 的**轴** —— ★ **[E]：本域自己写出的定义**，不是任何来源里的东西
# ===========================================================================

QK_NORM_AXIS = "head"          # 归一化沿 **head 维**（`d_h`）做，不是沿序列维（`S`）
# 等级：**[E]** —— 这是一条**本域明确写出的定义**，不是一个抄来的事实。
# ★★ **为什么它是 [E] 而不是 [B]/[C]**（**这是本模块里最需要说清的一条**）：
#     ① **论文**：本域的论文清单（计划 §C.2 的 12 篇）里**没有 QK-norm 的一手来源**；
#     ② **config**：两个真实模型的 config.json 里**都没有** `qk_norm` 这个字段
#        （原始字节计数：`qk_norm` 在 Qwen3-30B-A3B 的 config 里 **0** 次、
#          在 Llama-3-8B 的 config 里 **0** 次；对照词 `rms_norm_eps` 各 1 次 ⇒ 「0」可信）。
#     ⇒ **本域不给它编一个出处**（编一个 arXiv 号正是本库禁止的那类错误）。
#       本域的处理是：把「QK-norm 是什么」**明确写成一条本域的定义**，
#       然后**只对本机测到的后果下结论** —— 后果是可判定的，出处不是。
# ★ 口径：本定义**含两个选择**：(a) 对 `q`、`k` 各自归一化；(b) **沿 head 维**。
#    (b) 不是可有可无的修饰：`02` 的 `B.3` 实测——把轴换成序列维，
#    「抹掉输入幅度」这条性质**当场失效**（逐元素相对变化 `8.126e-01` vs `2.888e-16`）。
#    ⇒ 一个**可判定的**理由说明为什么定义里必须写死轴。

# ===========================================================================
# 3. RMSNorm 论文的逐字（**[C] 级：论文正文**）—— 表 B 判据 X11 的落点
# ===========================================================================
#
# ★ 抓取（2026-09-22，`curl -sS`，`WebFetch` 被策略拦截）：
#   `https://ar5iv.labs.arxiv.org/html/1910.07467`
#   **204694 B**，`sha256[:16] = 2c1f7f5b4b835d60`（★ 与计划 §C.2 的记法**逐位一致**）。
# 评审状态：★ abs 页的 `comments` 逐字为 `NeurIPS 2019` ⇒ **已确认同行评审**（按页面实测填）。
#
# ★★ 三句都在**原始字节**上计数过（`raw.count(b'...')`，不是先转文本再数）：
#    · `When the mean of summed inputs is zero` = **1**
#    · `RMSNorm is exactly equal to LayerNorm`   = **1**
#    · `we hypothesize that re-centering invariance` = **1**
#    ★ 对照检验：`hypothesize` 全文 **5** 次（分布见下），`RMS` 类的词数得到 ⇒ 「1」可信。
#    ⚠ **跨标记的引文在原始字节上必然计不到**（本域 §0.3 的纪律）：上面这三句都**逐字连续**，
#      所以直接计数有效；`re-centering invariance is dispensable` 这句**跨了标记**
#      （原始字节 **0** 次、剥标签后 **0** 次 —— 因为中间夹着 `in LayerNorm` 的标记），
#      本模块因此**按整句存**，不按片段存。

RMSNORM_MEAN_ZERO_EQUALITY = \
    "When the mean of summed inputs is zero, RMSNorm is exactly equal to LayerNorm."
# 来源：RMSNorm（arXiv 1910.07467）§4 逐字（紧接在「RMSNorm simplifies LayerNorm by totally
#       removing the mean statistic in Eq. (3) at the cost of sacrificing the invariance that
#       mean normalization affords」之后）  [C]
# ★★ 口径（**这就是 X11 的全部内容**）：这句是**条件句** —— `When the mean … is zero`。
#    它**不是**「RMSNorm 就是去均值的 LayerNorm」那种无条件说法。
#    ⇒ 本域把它的**充分必要形态**做成了可判定的断言（`01` 的 `B.3`）：
#      均值**恰好**为 0 时两者**逐位相等**（实测差 `0.0`），均值非 0 时**不相等**。
#    ★ 而「相等」的**闭式尺度**是 `cos(LN(x), RMS(x)) = σ/√ms`（见 `01` 的 `B.2`）：
#      `μ = 0 ⟺ cos = 1 ⟺ 两个算子相等`，一字不差地对应上面这句原文。

RMSNORM_HYPOTHESIS_RECENTERING = \
    "we hypothesize that re-centering invariance in LayerNorm is dispensable"
# 来源：RMSNorm（arXiv 1910.07467）**Abstract** 逐字（该句完整形态是
#       「In this paper, we hypothesize that re-centering invariance in LayerNorm is
#         dispensable and propose root mean square layer normalization, or RMSNorm.」）  [C]
# ★★ **措辞纪律（表 B 的第 1 条）**：原文用的是 **`hypothesize`**（加粗的 `we hypothesize`
#    在全文出现 5 次，这是其中 1 次）⇒ **不得转述成「已证明」**。
#    ⚠ 计划 §C.3 的 X11 把它写成 `we hypothesize that re-centering invariance … is dispensable`
#      （用 `…` 省掉了 `in LayerNorm`）—— 本模块按**原文的完整形态**存，
#      引用时的省略必须标出来（截断记号是**转写**的一部分，不是原文的一部分）。

RMSNORM_HYPOTHESIS_RESCALING_REASON = \
    ("we hypothesize that the re-scaling invariance is the reason for success of LayerNorm, "
     "rather than re-centering invariance.")
# 来源：RMSNorm（arXiv 1910.07467）§4 开头的逐字（前一句逐字是
#       「A well-known explanation of the success of LayerNorm is its re-centering and
#         re-scaling invariance property.」）  [C]
# ★★ **为什么要把这一句也收进来**：计划 §C.3 的 X11 写的是「**「re-scaling 才是成功原因」
#    原文用 `hypothesize`**」—— 那句话指向的**正是这一句**（不是 Abstract 那一句）。
#    两处 `hypothesize` 是**两个不同的主张**，都在原文里：
#      ① Abstract：re-centering invariance **is dispensable**（可省）；
#      ② §4      ：re-scaling invariance **is the reason for success**（才是原因）。
#    ⇒ 引用时必须指明是**哪一处**（这一点本域是独立核实后才知道的，见 `01` 的偏差回报）。
#    ★ 而「re-scaling invariance」**给 RMSNorm 的那一半**原文**没有**用 `hypothesize`：逐字
#      「RMSNorm regularizes the summed inputs … **giving the model re-scaling invariance
#        property** and implicit learning rate adaptation ability.」。

# ===========================================================================
# 4. pre-LN 论文的逐字（**[C] 级**）—— 表 B 判据 X8 的落点
# ===========================================================================
#
# ★ 抓取（2026-09-22，`curl -sS`）：`https://ar5iv.labs.arxiv.org/html/2002.04745`
#   **521794 B**，`sha256[:16] = d90ec3972ed0117c`（★ 与计划 §C.2 逐位一致）。
# 评审状态：abs 页 `Journal reference` 逐字含 `Published on ICML 2020` ⇒ **已确认同行评审**。
#
# ★ 原始字节计数：`comparable` = **5**、`hypothesize` = **1**（`hypothesise` = **0**）。

PRE_LN_VERDICT_WORD = "comparable"
# 来源：pre-LN vs post-LN（arXiv 2002.04745）**Abstract** 与 **§4.2** 逐字  [C]
# ★★ Abstract 逐字：「We show in our experiments that Pre-LN Transformers without the warm-up
#    stage can reach **comparable** results with baselines while requiring significantly less
#    training time and hyper-parameter tuning on a wide range of applications.」
#    §4.2 逐字：「the BLEU score and validation loss of the Pre-LN Transformer can achieve
#    around 34 and 4, which are **comparable** with the performance of the Post-LN Transformer.」
# ★★ **措辞纪律（表 B 的第 2 条）**：论文主张的是 **`comparable`**（可比），
#    **不是「更好」**。原始字节计数：`comparable` = 5；而 `outperform` / `better than`
#    这类比较级**没有出现在结论里**（该论文自己的用词是 `comparable` 与 `competitive`）。
#    ⚠ 它**确实**主张的两件事是：「**可以去掉 warm-up**」与「**收敛更快、超参更少**」——
#      那是**训练流程**的账，不是**最终质量**的账。两者不可互相顶替。

PRE_LN_ABSTRACT_COMPARABLE = \
    ("can reach comparable results with baselines while requiring significantly less training "
     "time and hyper-parameter tuning")
# 来源：pre-LN vs post-LN（arXiv 2002.04745）**Abstract** 逐字（该句完整形态是
#       「We show in our experiments that Pre-LN Transformers without the warm-up stage can reach
#         comparable results with baselines while requiring significantly less training time and
#         hyper-parameter tuning on a wide range of applications.」）  [C]
# ★ 原始字节计数：`comparable results with baselines` = **1**；
#   对照：`comparable` 全文 = **5**，逐处（**穷举，不是抽样**）：
#     · Abstract：「can reach **comparable** results with baselines …」（= 本常量）
#     · 引言：「It can achieve **comparable** final performances but use much less training
#        time.」（`It can achieve comparable final performances` 计数 **1**）
#     · §4.2：「… which are **comparable** with the performance of the Post-LN Transformer.」
#     · **数据集规模的 2 次**（正文 + 附录各一次）：「3.4B words in total, which is
#       **comparable** with the data corpus used in Devlin et al. 2018」⇒ **同名不同指**：
#       那两处比的是**语料规模**，不是模型质量。
#   ⇒ 「5 次」这个数字**不能**当作「论文说了 5 遍 comparable（关于质量）」的证据。
# ★★ 口径：**这句里没有「更好」两个字**。论文主张的是「**可比**（comparable）+ 训练更快 + 超参更少」。
#    ⚠ 本域**不**声称「原文没有 `outperform`」—— 全文有 **1** 次，但**主语是 `Wang et al. 2019`**
#      （见 `PRE_LN_OUTPERFORM_ATTRIBUTION`）。两个主张**都在原文里**，混起来就成了 X8 那条错。

PRE_LN_GRADIENT_HYPOTHESIS = \
    "we hypothesize that the gradient scale is one of the reasons that the Post-LN Transformer needs a careful learning rate scheduling"
# 来源：pre-LN vs post-LN（arXiv 2002.04745）§4 逐字（该句完整形态是
#       「Given the analysis above, we hypothesize that the gradient scale is one of the reasons
#         that the Post-LN Transformer needs a careful learning rate scheduling.」）  [C]
# ★★ **措辞纪律（表 B 的第 2 条的后半）**：**post-LN 梯度尺度**的那条论断原文用的是
#    **`hypothesize`**（全文 `hypothesize` = **1** 次，就是这一处；`hypothesise` = 0 次）。
#    ⇒ 不得写成「已证明 post-LN 的梯度尺度大」。
#    ★ 而同段里**紧邻的**另一句用的是直陈句：
#      「the scale of the expected gradients grows along with the layer index for the Post-LN
#        Transformer. On the contrary, the scale almost keeps the same for different layers in
#        the Pre-LN Transformer.」—— 那是**观察**的陈述（配 Figure 3），
#      后面那句 `hypothesize` 才是**因果解释**。**引用时必须分清哪一句是观察、哪一句是假设。**

# ===========================================================================
# 4b. 位置（placement）的逐字（**[C] 级**）—— `02` 的「位置之争」节
# ===========================================================================
#
# ★★ 这四条的用处：`02` 讲「归一化放在哪」，而**两种放法的定义**在原文里是逐字写着的 ——
#    不需要本域替它下定义。★ 原始字节计数全部做在**原始字节**上（§0.3）。

POST_LN_PLACEMENT_VERBATIM = "puts the layer normalization between the residual blocks"
# 来源：pre-LN vs post-LN（arXiv 2002.04745）§3 逐字  [C]
# ★ 原始字节计数：**1**（★ 对照：`layer normalization` 在全文的计数远大于 1 ⇒ 这句是独一份的措辞）。

PRE_LN_PLACEMENT_VERBATIM = "puts the layer normalization inside the residual connection"
# 来源：同上，同段逐字  [C]
# ★ 原始字节计数：**2**。
# ★ 该句完整形态逐字：「Different from the Post-LN Transformer that puts the layer normalization
#    between the residual blocks, the Pre-LN Transformer **puts the layer normalization inside the
#    residual connection** and **places it before all other non-linear transformations**.」
#    （后半句的逐字计数：`places it before all other non-linear transformations` = **1**。）

PRE_LN_FINAL_NORM_VERBATIM = "final layer normalization"
# 来源：同上 §3 逐字：「Additionally, the Pre-LN Transformer uses a **final layer normalization**
#       right before the prediction.」  [C]
# ★ 原始字节计数：`final layer normalization` = **2**、`right before the prediction` = **1**。
# ★★ 但 **`uses a final layer normalization right`** 的计数是 **0** —— 因为原文把那四个词
#    包在 `<em …>final layer normalization</em>` 里（本域实测的字节上下文），
#    所以**跨标记的片段在原始字节上必然计不到**（计划 §0.3 的纪律，本域又一次现场）。
#    ⇒ 本模块**只存计得到的那两段**，不存跨标记的整句（那会得到一个「计数 0 但我声称有」的常量）。
# 口径：★ 这条是**本域 `02` 篇的回链目标**：`final_norm` 的必要性**只回链** `transformer/01`
#   的判据 ③（logits std 之比），本文**不重做** —— 而「pre-LN 栈在最后补一个归一化」这个
#   **结构事实**的出处就是这一句（架构层，不是 kernel 层）。

PRE_LN_OUTPERFORM_ATTRIBUTION = \
    "outperforms the Post-LN Transformer when the number of layers increases"
# 来源：同上 §3 逐字：「Wang et al. 2019 suggested that the Pre-LN Transformer **outperforms the
#       Post-LN Transformer when the number of layers increases**.」  [C]
# ★ 原始字节计数：`outperforms the Post-LN Transformer when the number of layers increases` = **1**；
#   而 `Wang et al. 2019 suggested that the Pre-LN …` 的计数是 **0**
#   （`Wang et al. 2019` 是一个 `<a>` 链接 ⇒ **又一次跨标记**）。
# ★★ **这一条是本主题最容易写错的地方（表 B 判据 X8）**：`outperform` 这个词在全文出现
#    **1 次**，而**它的主语是 `Wang et al. 2019`（被引的第三方）**，**不是本文的主张**。
#    ⇒ 本域写「pre-LN 更好」时**必须**写出这个归属：**论文自己的结论词是 `comparable`**，
#      它引用的 Wang et al. 2019 说的是 `outperforms`。两者在同一条街上，但**不是同一个主张**。

# ===========================================================================
# 5. UNSOURCED —— 待办清单，不是免责声明
# ===========================================================================
#
# ★ CONTRIBUTING §2：「不得**既保留数字又记 UNSOURCED**」—— 下面每一项都**没有**出现在
#   本模块的正文常量里。

UNSOURCED = (
    # 每一项是一个**稳定标识符**（不是散文）。
    "qk_norm_primary_source",     # ★★ QK-norm 的**一手来源不在本环境**：计划 §C.2 列的 12 篇里
                                  # 没有它；本域**不替它编一个 arXiv 号**（那是本库禁止的错误）。
                                  # ⇒ 本域只把「它是什么」写成 [E] 定义（见 `QK_NORM_AXIS` 的注释），
                                  #    并只对本机测到的**后果**下结论。
    "qk_norm_in_real_configs",    # 「哪些真实模型用了 QK-norm」：两个真实 config 里
                                  # `qk_norm` **各 0 次**（原始字节计数）⇒ 本环境**没有**这个锚点。
                                  # ⇒ 本域不写「某某模型用了 QK-norm」这类句子。
    "eps_position_numerical_diff",# ★ `eps` **放在哪**（分母内/外、加在方差上还是标准差上）
                                  # 带来的数值差异 —— 这是**计划 §2 边界表的禁区**，
                                  # 正文在 `40-kernels/softmax-normalization/03:55-71`。
                                  # 本模块只把它当**一个数**（量级），不解释它的位置。
    "attention_entropy_collapse_threshold",
                                  # ★ 「归一化熵低于多少，注意力就算废了」——**本域不给这个阈值**：
                                  # 没有任何一手来源给出它。`02` 的 `B.4` 只断言
                                  # 「熵随 logits 尺度**严格递减**」这一条可判定的单调性，
                                  # **不**断言「多少算废」。
    "layernorm_vs_rmsnorm_quality",  # 「两者训练出来的模型质量谁高」——训练侧的账，
                                  # 本域无 GPU、不训练模型 ⇒ 不写（RMSNorm 论文自己的
                                  # 逐字是 `similarly or more effective`，那**是它的实验**，
                                  #  本域**转述时必须标为论文的实验**，不得写成本域实测）。
)

# ===========================================================================
# 尾注：本主题**刻意不搬**的字段（它们的正文在别处）
# ===========================================================================
#
#   · `hidden_size` / `num_hidden_layers` / `num_attention_heads` / `num_key_value_heads` /
#     `head_dim` / `intermediate_size` / `vocab_size` / `hidden_act` / `tie_word_embeddings`
#                                   → `transformer/code/specs.py`
#   · `rope_theta` / `max_position_embeddings` → `positional-encoding/code/specs.py`
#   · `num_experts*` / `moe_intermediate_size` / `router_aux_loss_coef` → `moe/code/specs.py`
#   · `sliding_window` / `use_sliding_window` → `attention-variants/code/specs.py`
#       ★ **本主题一个都不断言它们** —— 按 `45-…/code/specs.py:107-113` 的教训，
#         「照抄但不使用」的常量无法被任何断言约束，只会制造变异测试的漏网。
#       ★ 本模块只搬了 **2 个字段**（两个模型的 `rms_norm_eps`）：理由是判据 ② 需要它
#         （不然「eps 破坏不变性」只是一个玩具量级的演示），且它们**每一位都被
#         `01_rmsnorm_layernorm.py` 的 `D.4` / `B.5` 用上并逐位钉住**。
#
#   · ★★ **LayerNorm / RMSNorm 的公式、Welford 递推、`Σx²−nμ²` 的灾难、`eps` 位置的数值差异
#     → `40-kernels/softmax-normalization/03-normalization-fusion.md:55-71`
#     （**本主题最大的禁区**，计划 §2）—— 本模块与两个编号程序里**都没有**这些内容：
#     程序里出现的两个算子实现是**测量的对象**（没有它就没法测不变性），
#     而**公式的正文在彼，此处回链**。程序里也不写任何推导。
#
#   （★ R3 的裁定：**每主题一份，每份只放本主题用得到的常量**，跨主题同值的常量各写一份、
#     靠 `# 同值位置：` 互指。本库 40 个 `code/` 目录**没有任何跨主题共享辅助模块的先例**，
#     `CONTRIBUTING:36`「`code/` 必须自包含」是**全库规则**。）

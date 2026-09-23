"""90-training/initialization-and-stability 的外部事实常量源。

本模块是 `CONTRIBUTING` §2 约定的**共享事实源**（`specs.py`）：本主题的三个编号程序从它读常量，
推导过程可复核，**常量本身不可自证**。

★★ **来源必须分级，并写明它是哪一级**（计划 §1.5）。本域的头号错误模式是
  **P19 —— 把「某实现的文档」当成「规范的文档」**（B 级当 A 级）。
  本主题的三条链**分属三级，必须分开写**：
    · **He / Glorot / μP 的公式与论断** → **[C] 论文**（逐条注明是否确认同行评审）；
    · **`torch.nn.init.*` 的签名、上下界与 `calculate_gain` 表**、**`clip_grad_norm_` 的返回值口径**
      → **[B] PyTorch 文档**（**某实现的规格，不是规范**）⇒ 一律写明版本与页面；
    · **本机量出来的方差比值 / 斜率 / 更新量比值** → **推导值**（第二、三类），**不放进本模块**
      （`CONTRIBUTING` §2：「只放从来源抄来的数，**不放推导值**」）。

★★ **本主题要正面处理的三条「与常见转述不一致」**（计划 §0.7-C 的 **Y1 / Y2 / Y3**）——
  它们是本主题 `01` 的**核心内容**，也是本模块**必须把字面量钉死**的地方：
    · **Y1**：★★★ **「Xavier 初始化 = `arXiv:1001.0014`」是错的** —— 实测那个 arXiv 页的
      `<title>` 逐字 = `[1001.0014] Jet Shapes and Jet Algorithms in SCET`（**高能物理**）。
      ⇒ 引用格式用 **PMLR 9:249-256, 2010**（**P21 已办**：编号先打开看标题）。
    · **Y2**：**Glorot 原文里没有 `Xavier initialization` 这个词**（四种数法**都是 0**，稳健）。
      ★★ 但另外两条说法要收窄：**(a)** 「`normalized initialization` 出现 17 次」**不可复现**
      （本次四种数法给 **13 / 14 / 17 / 18**）⇒ **那个数不写**；**(b)** 「该词首次出现在 He 2015」
      **被 He 2015 自己否掉**（它逐字写着 `This is called “Xavier” initialization in Jia2014`，
      而同文另一处 `Jia2014` = **Caffe**）⇒ 只能说
      「**在本次取到的一手文献中最早见于 He 2015，且它自称引自 Caffe**」。
    · **Y3**：**Xavier 的公式是 `Var[W] = 2/(n_i + n_{i+1})` 的【均匀】分布**
      （`U(±√(6/(n_j+n_{j+1})))`，原文 Eqn 16），**不是 `N(0, 1/n)`**
      ⇒ **分母是 `fan_in + fan_out`**；推荐实现是**均匀**，不是正态。

★ **只放本主题真的会断言的常量**（计划 §3 的 R3 裁定）。每一条下面写明**预期消费者**。
  ★ **没有消费者的常量就是 `make mutate` 的漏网候选**（计划 §4 **T11 的盲区①**）。

★★ **本模块的自缚纪律（盲区④的落法）**：`mutate.py` 的 `_walk()`
  **不下钻 `BinOp`/`UnaryOp`/`Call`** ⇒ `2.0 ** -6` 这类**幂次字面量整体不进变异分母**，
  `√2` 若写成 `2 ** 0.5` 也**不会进分母**（**T31/T37/T38**）。
  ⇒ 本模块**所有数值一律写成十进制字面量**，幂次/开方写进注释
  （如 `√2` 写作 `1.4142135623730951`、`2^-7` 写作 `0.0078125`）。

★ 每条常量带 `# 来源：` + `[A-E]` 级 + 抓取日；跨主题/跨文件同值的另带 `# 同值位置：`。
  分级判据见 `CONTRIBUTING` §2（A 厂商规范 / B 产品官方文档 / C 论文 / D 社区 / E 本库推导）。

★★ **量法口径（本模块逐条照办，取自本域 §0.3 与计划 §4 的 T18/T34/T46）**：
  「计数 N 次」一律是**剥标签 + 折叠空白后**在**指定载体**上数的，且**引文片段不含换行**：
    ① **跨标记**的引文在原始字节上必然计不到；
    ② **标点后的空格数**会变；③ **引文片段不能含跨行空白**；
    ④ **同一条引文在不同页面的计数可能不同**（abs 页 vs ar5iv）；⑤ **数字可能带千位分隔符**；
    ★★ **本次新撞到的第六个形态：`raw` 计数为 0 而剥标签后为 1**
      （PMLR 页的 `PMLR 9:249-256, 2010`、PyTorch 页的 `mode = 'fan_in'` —— 中间夹了标记/特殊空格）
      ⇒ 本模块凡出现这种情形，**两个数都写出来**，并写明**判定用的是哪一个载体**。
  ⇒ **引文一律在剥标签 + 折叠空白的文本上计数**（除非另有说明），并配**对照**。
"""

# ===========================================================================
# 1. He 初始化（He et al. 2015）—— [C] 论文（**不确认同行评审**）
# ===========================================================================
#
# 来源：[C] K. He, X. Zhang, S. Ren, J. Sun,
#       **`Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet
#       Classification`**，arXiv **1502.01852**。
#       评审状态：**不确认同行评审** —— abs 页 `Comments:` 字段**不存在**（实测 0 次）。
#       ★ **P21 已办**：abs 页 `<title>` 逐字 =
#         `[1502.01852] Delving Deep into Rectifiers: Surpassing Human-Level Performance on
#          ImageNet Classification`（原始字节 1 次）。
#       抓取（ar5iv 正文）：`https://ar5iv.labs.arxiv.org/html/1502.01852`，2026-09-23，
#       **289072 B**；abs 页：**42118 B**。

HE_ARXIV_ID = "1502.01852"
# 来源：[C] He 2015（同上），abs 页 `<title>`。逐字（原始字节 1 次）。
# 口径：★ **P21 的落法**：编号与标题**成对存放**，程序里断言「标题里含 `Rectifiers`」
#       这种**结构性**区分（而不是只钉编号）—— 编号本身可以被读错，标题不会。
# 消费者：`01_init.py` 的 `part_D`（`D.5`）、`03_spikes.py` 的 `part_D`（不涉及）。
# 同值位置：无。

HE_TITLE = "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification"
# 来源：[C] He 2015（同上）**abs 页**的 `<title>`（★ 去掉 `[1502.01852] ` 前缀后的标题本体）。
#       逐字（原始字节 1 次，在 `<title>` 里；剥标签后同一字符串在页面元数据里另有 1 次）。
# 口径：★★ 它与 `GLOROT_WRONG_ARXIV_TITLE` **成对读**：那一个是 `arXiv:1001.0014` 的标题
#       （高能物理论文）⇒ 两条常量一起把 **Y1** 钉成一个**可执行**的对照
#       （`01_init.py` 的 `A.4`：两个标题必须**逐位不同**）。
# 消费者：`01_init.py` 的 `A.4` 与 `part_D`。
# 同值位置：无。

HE_VARIANCE_NUMERATOR = 2.0
# 来源：[C] He 2015（同上），§2.1 **Eqn 10** 与紧随其后的那句。剥标签后逐字（各 1 次）：
#         `A sufficient condition is`
#         `whose standard deviation (std) is`
#       完整式（同一行内，不含换行）：`1/2 n_l Var[w_l] = 1, ∀l.`（Eqn 10）
#       ⇒ `Var[w_l] = 2/n_l`，`std = √(2/n_l)`。
#       ★ 同文 §2.1 逐字（原始字节 1 次）：`the number of connections` ——
#         它定义了 `n_l`：`With n = k²c denoting the number of connections of a response`
#         ⇒ ★★ **`n_l` 是「连接数」**：对全连接层 = `fan_in`，对卷积 = `k²·c_in`
#         （**漏掉 `k²` 会让方差差 `k²` 倍**：`k = 3` 时差 **9** 倍）。
#       ★★ 同文 §2.2 逐字（剥标签后 1 次）：`it is sufficient to use either`
#         —— 完整句 `it is sufficient to use either Eqn.(14) or Eqn.(10) alone.`
#         ⇒ 原文**自己**说「前向那条（Eqn 10）与反向那条（Eqn 14）只用其一就够了」。
#         ⚠ **量法口径**：`it is sufficient to use either Eqn.( 14 ) or Eqn.( 10 ) alone`
#           这个**完整片段在原始字节上计 0**（`Eqn.(` 后面夹了标记）⇒ 本模块只取
#           不含公式引用、不含换行的短片段。
# 口径：★ `2` 这个分子**只属于前向**（Eqn 10）。★★ 它与 `GLOROT_VARIANCE_NUMERATOR` 的 `2`
#       **不是同一个 2 的来源**：Glorot 的 `2` 来自「前向与反向两个约束的折中」，
#       He 的 `2` 来自「ReLU 把方差折半」。⇒ 两条常量**分开存放**，不得合并成一个。
# 消费者：`01_init.py` 的 `A.1`（`Var[W] = HE_VARIANCE_NUMERATOR / fan_in`）与 `part_D`。
# 同值位置：无（`1.0` 那个 `1/2` 在程序里作为推导量出现，不进本模块 —— 它是推导值）。

HE_SUFFICIENT_CONDITION_QUOTE = "A sufficient condition is"
# 来源：[C] He 2015（同上），§2.1，**原始字节 1 次**（剥标签后同样 1 次）。
# 口径：★★ 这是**「Eqn 10 是充分条件」这句话的逐字证据**，也是**措辞纪律的现场**：
#       原文写的是 `A sufficient condition is`（**充分**），而**不是**「充要」。
#       ⇒ 正文不得把它升格成「充要条件」（**P20 的同类**：不要把原文的弱措辞加强）。
# 消费者：`01_init.py` 的 `part_D`。
# 同值位置：无。

HE_STD_QUOTE = "whose standard deviation (std) is"
# 来源：[C] He 2015（同上），§2.1，**原始字节 1 次**。
# 口径：★ 完整句逐字是 `This leads to a zero-mean Gaussian distribution whose standard
#       deviation (std) is √(2/n_l)` ⇒ ★★ **论文的实现是【正态】分布**，
#       而 [B] PyTorch 的 `kaiming_uniform_` 是**均匀**分布 —— 两者**同方差、不同分布**
#       （`C.1` 用四阶矩把这件事量出来）。
# 消费者：`01_init.py` 的 `C.1` 与 `part_D`。
# 同值位置：无。

HE_BACKWARD_ALTERNATIVE_QUOTE = "it is sufficient to use either"
# 来源：[C] He 2015（同上），§2.2，**原始字节 1 次**（剔掉公式引用后的短片段）。
# 口径：★★ 它把「前向」与「反向」**分成两条约束**（Eqn 10 / Eqn 14）—— 而本主题的
#       `A.2b` 量的正是**反向增益** `2·fan_out/fan_in`（在 `256/4096` 上是 **32**）。
# 消费者：`01_init.py` 的 `A.2b` 与 `part_D`。
# 同值位置：无。

HE_CONNECTIONS_QUOTE = "the number of connections"
# 来源：[C] He 2015（同上），§2.1，**原始字节 1 次**。
# 口径：★ `n_l` 的定义处 ⇒ 卷积的 `fan_in = k²·c_in`（`B.2` 量「漏掉 `k²` 的代价」）。
# 消费者：`01_init.py` 的 `B.2` 与 `part_D`。
# 同值位置：无。


# ===========================================================================
# 2. Glorot & Bengio 2010 —— [C] 论文（**已确认同行评审 AISTATS 2010**）
#    ★★ 这一节的每一条都是 **Y1 / Y2 / Y3** 的落点
# ===========================================================================
#
# 来源：[C] X. Glorot & Y. Bengio, **`Understanding the difficulty of training deep
#       feedforward neural networks`**。
#       ★★★ **它不在 arXiv 上** —— 本次取到的**载体是 PMLR 的官方页**：
#         `https://proceedings.mlr.press/v9/glorot10a.html`，2026-09-23，**18993 B**，
#         `sha256[:16] = 4d0a3b0f7f6e6a6b`（**本模块不把哈希当证据**，它只是抓取记录）。
#         PDF：`https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf`（**1647622 B**），
#         文本层用 `pdftotext`（`-layout` 版与非 layout 版**都取了**）。
#       评审状态：**已确认同行评审** —— PMLR 页逐字（**剥标签后 1 次、原始字节 0 次**：
#         `PMLR` 与 `9:249-256` 之间夹了标记）：
#         `Proceedings of the Thirteenth International Conference on Artificial Intelligence
#          and Statistics , PMLR 9:249-256, 2010.`
#       ★ **P21 已办**：PMLR 页的标题逐字 =
#         `Understanding the difficulty of training deep feedforward neural networks`
#         （原始字节 5 次 —— 标题在页面里出现多次，其中一次在 `<title>`）。

GLOROT_TITLE = "Understanding the difficulty of training deep feedforward neural networks"
# 来源：[C] Glorot & Bengio 2010（同上）**PMLR 官方页**，逐字（原始字节 5 次）。
# 口径：★★ 与 `HE_TITLE` 同形：**标题与编号成对存放**，程序里做**结构性**断言
#       （`A.4`：它与 `GLOROT_WRONG_ARXIV_TITLE` 必须逐位不同）。
# 消费者：`01_init.py` 的 `A.4` 与 `part_D`。
# 同值位置：无。

GLOROT_PMLR_CITATION = "PMLR 9:249-256, 2010"
# 来源：[C] Glorot & Bengio 2010（同上）**PMLR 官方页**，**剥标签后 1 次、原始字节 0 次**。
# 口径：★★★ **Y1 的正面落法**：正文与程序的引用格式**用 PMLR**，**不得**写成 arXiv
#       （`GLOROT_WRONG_ARXIV_ID` 那一篇是别的论文）。
#       ★ 本库先例：`80-model-architectures/normalization/02` 引的是 arXiv 2002.04745
#         （那篇**确实有** arXiv 版）—— 与本条**并列**时不要混。
# 消费者：`01_init.py` 的 `A.4` 与 `part_D`。
# 同值位置：无。

GLOROT_VENUE_QUOTE = \
    "Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics"
# 来源：[C] Glorot & Bengio 2010（同上）**PMLR 官方页**，剥标签后 5 次（原始字节 8 次）。
# 口径：★★ **「已确认同行评审」这条判定的逐字依据**（与 `specs.py` 里 Adam / AdamW
#       用 `Comments:` 字段判定同行评审同源，只是这里的载体是 PMLR 页而不是 arXiv abs 页）。
#       ★ 同一句还逐字出现在**论文 PDF 的脚注**里：
#         `Appearing in Proceedings of the 13th International Conference on Artificial
#          Intelligence and Statistics (AISTATS) 2010, Chia Laguna Resort, Sardinia, Italy.
#          Volume 9 of JMLR: W&CP 9.`
#         ⚠ **但它跨行**（`Conference` 之后换行）⇒ **不能**拿它做原始字节的计数。
# 消费者：`01_init.py` 的 `part_D`（`D.5`）。
# 同值位置：无。

GLOROT_VARIANCE_NUMERATOR = 2.0
# 来源：[C] Glorot & Bengio 2010（同上），§4.2。原文**三条约束**逐字（PDF 文本层）：
#         Eqn (10) `∀i, n_i Var[W^i] = 1`          ← 前向
#         Eqn (11) `∀i, n_{i+1} Var[W^i] = 1`      ← 反向
#         ★ Eqn (12) 前面那句逐字（**原始字节 1 次**，不含换行的最大片段）：
#           `As a compromise between these two constraints`
#         ⇒ Eqn (12) `Var[W^i] = 2/(n_i + n_{i+1})`
#         ★ Eqn (16) 的实现逐字：`W ∼ U(−√(6/(n_j+n_{j+1})), √(6/(n_j+n_{j+1})))`
#         ★ 紧接的名称逐字（**剥标签/折叠空白后 1 次**，原始字节里跨行 ⇒ 不计数）：
#           `We call it the normalized initialization`
# 口径：★★★ **Y3 的落法**：分母是 **`n_i + n_{i+1}` 之【和】**（= `fan_in + fan_out`），
#       **不是单独的 `fan_in`**；分布是**均匀**（Eqn 16），**不是正态**。
#       本常量的 `2` 与 `HE_VARIANCE_NUMERATOR` 的 `2` **同值但不同源**（见上）。
# 消费者：`01_init.py` 的 `A.2`（`Var[W] = 2/(fan_in+fan_out)`）与 `part_D`。
# 同值位置：无。

GLOROT_COMPROMISE_QUOTE = "As a compromise between these two constraints"
# 来源：[C] Glorot & Bengio 2010（同上），§4.2，**原始字节 1 次**（到 `constraints` 为止；
#       再往后一个字符就是换行 ⇒ 这已是**不含跨行空白的最大片段**）。
# 口径：★ 它证明 Eqn 12 **是「两个约束的折中」**，而不是「He 的 `2/fan_in` 的另一种写法」。
# 消费者：`01_init.py` 的 `part_D`。
# 同值位置：无。

GLOROT_NORMALIZED_INIT_QUOTE = "the normalized initialization"
# 来源：[C] Glorot & Bengio 2010（同上）。**原始字节 3 次**；`glorot.txt`（`-layout` 版）
#       上 **2 次**。★★ **这一个常量只作「原文里确实有这个名字」的取样，不作计数证据** ——
#       原因见下一条。
# 口径：★★★ **Y2 的处置**：计划 §0.7-C 记的「`normalized initialization` 出现 **17** 次」
#       **不可复现** —— 本次四种数法给 **13 / 14 / 17 / 18**
#       （区别来自：是否 `-layout` / 是否忽略大小写 / 是否把 `We call it the` 换行折叠）。
#       ⇒ **按本域自己的口径纪律（§0.3），这个数不写进正文，也不写成本模块的常量。**
#       ★ 而「`Xavier initialization` = 0」这一半是**稳健的**（四种数法**都是 0**）
#         ⇒ 只有它进常量（`GLOROT_XAVIER_TERM_COUNT`）。
# 消费者：`01_init.py` 的 `part_D`（逐位钉住 + 说明为什么不给计数）。
# 同值位置：无。

GLOROT_XAVIER_TERM_COUNT = 0
# 来源：[C] Glorot & Bengio 2010（同上）**PDF 文本层**（`pdftotext`，两份都数了）。
#       逐字计数（**四种数法全部给 0**）：
#         · `glorot_raw.txt`（非 layout）`: "Xavier initialization"` → **0**
#         · 同上，忽略大小写 → **0**
#         · `glorot.txt`（`-layout`）→ **0**
#         · 折叠空白 + 去连字符 → **0**
#       ★ **对照**：同文的 `Xavier`（单独）计 **4** 次，且**四处全部是页眉的作者名**
#         （`Xavier Glorot        Yoshua Bengio`，出现在标题页与第 250 / 252 / 254 页页眉）。
#       ★ **第二个对照**：`normalized initialization` 在同文计 ≥ 13 次
#         ⇒ **不是「数不到」，而是「那个名字就是不在这里」**（§0.3：否定性结论必须配对照）。
# 口径：★★★ **Y2 的核心落法**：Glorot 原文**没有**提出 `Xavier initialization` 这个名字。
#       ★★ 但**收窄**（计划已裁定）：**不得**由此写成「这个名字首次出现在 He 2015」——
#       He 2015 **自己**逐字写着 `This is called “Xavier” initialization in Jia2014`，
#       而同文另一处 `Jia2014` = **Caffe**（`our modification of the Caffe library Jia2014`）。
#       ⇒ 可确证的只是：**在本次取到的一手文献中，带引号的 “Xavier” 最早见于 He 2015，
#         且它自称引自 Caffe**。（★ **Caffe 源码本身本次无法核实**：`raw.githubusercontent.com`
#         取不到 ⇒ 已登记进 `UNSOURCED`。）
#       ⚠ **量法口径**：He 2015 那句话在 ar5iv 页上是**弯引号 + 两侧空格**：
#         `This is called “ Xavier ” initialization in Jia2014 .`
#         ⇒ 计划 §0.7-C 写的直引号形式 `This is called "Xavier" initialization in Jia2014`
#         **在原始字节上计 0**（这是 `raw.count` 的**第六个形态**：排版字符不同）。
# 消费者：`01_init.py` 的 `part_D`（`D.6`：逐位钉住 `== 0`）与 `A.4`。
# 同值位置：无。

GLOROT_WRONG_ARXIV_ID = "1001.0014"
GLOROT_WRONG_ARXIV_TITLE = "Jet Shapes and Jet Algorithms in SCET"
# 来源：★★★ **Y1 的否证**。抓取 `https://arxiv.org/abs/1001.0014`，2026-09-23，**44851 B**；
#       `<title>` 逐字 = `[1001.0014] Jet Shapes and Jet Algorithms in SCET`
#       （原始字节 11 次 —— 标题在页面元数据里出现多次）。
# 口径：★★★ 它**不是**神经网络论文（`SCET` = Soft Collinear Effective Theory，**hep-ph**）。
#       ⇒ 「Xavier 初始化 = `arXiv:1001.0014`」这句话是**引用了一个不存在的编号**（**P21**）。
#       ★ 本主题把它做成**可执行**的对照：`01_init.py` 的 `A.4` 断言
#         `GLOROT_WRONG_ARXIV_TITLE != GLOROT_TITLE` 且两条标题各自逐位钉住。
# 消费者：`01_init.py` 的 `A.4` 与 `part_D`。
# 同值位置：无。


# ===========================================================================
# 3. ★ [B] PyTorch 2.14 的规格 —— **某实现的文档，不是规范**（P19 的纪律）
# ===========================================================================
#
# 抓取：2026-09-23（★ **取法**：`/docs/stable/...` 返回的是 `<meta http-equiv="refresh">`
#   跳转壳，`curl -L` 跟不了 meta refresh ⇒ 直接取 `2.14` 的路径；本域 §0.3 与 **T20**）：
#     · `https://docs.pytorch.org/docs/2.14/nn.init.html`                 **736073 B**
#     · `https://docs.pytorch.org/docs/2.14/generated/torch.nn.utils.clip_grad_norm_.html`
#                                                                            **179174 B**
# ★★ 本主题**特别要写清这一级**：`kaiming_uniform_` / `xavier_uniform_` 的 `bound` 公式
#    **不是规范**，它是「某个实现怎么做的」；而 He 的 `√(2/n_l)` 是 **[C] 论文的公式**。
#    两者**同方差**这件事（`A.3`）是**两条不同等级的来源互校**的结果 —— 不是任一方的背书。

PYTORCH_KAIMING_DEFAULT_MODE = "fan_in"
PYTORCH_KAIMING_DEFAULT_NONLINEARITY = "leaky_relu"
PYTORCH_KAIMING_DEFAULT_A = 0.0
# 来源：**[B] PyTorch 2.14 的 `torch.nn.init` 页**（同上 URL），签名逐字
#       （**剥标签后 2 次**：页面 HTML 正文 1 + JSON-LD 1；**原始字节 0 次** ——
#        签名里词与词之间夹了特殊空格）：
#         `kaiming_uniform_ ( tensor , a = 0 , mode = 'fan_in' , nonlinearity = 'leaky_relu' ,
#          generator = None )`
# 口径：★★ **两个默认值都不是「为 ReLU 调的」**：`mode='fan_in'` 才是 He 的那一支，
#       而 `nonlinearity` 默认是 **`'leaky_relu'`**（**不是** `'relu'`）。
#       ★★ 但 `a = 0` 时 leaky_relu 的 gain 与 relu 的 gain **逐位相同**（都是 √2）
#         ⇒ **默认值恰好落在 He 的 ReLU 那一支上**。这是一条**巧合式的等价**，
#         它**不是恒真**的（`a = 0.2` 时 gain 变成 1.386750491…，差 2.7e-2）——
#         `C.3` 把两面都断言出来。
# 消费者：`01_init.py` 的 `A.3`（默认值 ⇒ `gain=√2` ⇒ `bound=√6/√fan_in`）、`C.3`、`part_D`。
# 同值位置：无（`0.0` 与 `C.3` 里的 `0.2` 是**两个不同的角色**，后者是反面对照的参数）。

PYTORCH_XAVIER_DEFAULT_GAIN = 1.0
# 来源：**[B] PyTorch 2.14 的 `torch.nn.init` 页**（同上 URL），签名逐字（剥标签后 3 次）：
#         `xavier_uniform_ ( tensor , gain = 1.0 , generator = None )`
# 口径：★ 默认 `gain = 1.0` ⇒ `a = √(6/(fan_in+fan_out))`，与 Glorot 的 **Eqn 16 逐位同式**。
#       ⚠★ **而 `kaiming_uniform_` 的默认 gain 是 √2（见上）** ⇒ **两个函数的默认值
#         对 `fan` 的账不一样**：一个是 `2/(fan_in+fan_out)` 的方差，一个是 `2/fan_in` 的方差。
# 消费者：`01_init.py` 的 `A.3b` 与 `part_D`。
# 同值位置：无。

PYTORCH_KAIMING_BOUND_NUMERATOR = 3.0
PYTORCH_XAVIER_BOUND_NUMERATOR = 6.0
# 来源：**[B] PyTorch 2.14 的 `torch.nn.init` 页**（同上 URL）。两式逐字**在页面上是
#       渲染出来的**（MathML + LaTeX 双写）：
#         `bound = gain × √(3 / fan_mode)`（`kaiming_uniform_`）
#         `a     = gain × √(6 / (fan_in + fan_out))`（`xavier_uniform_`）
#       ⚠ **量法口径（必须写明）**：这两个式子在页面源码里是
#         `\sqrt{\frac{3}{\text{fan\_mode}}}` 这类 **LaTeX 串 + MathML 元素**
#         ⇒ 字面量 `bound = gain × sqrt(3 / fan_mode)` 在原始字节上**计 0**
#         （MathML 的 `<msqrt>` 剥标签后整个消失）。
#         ⇒ 本模块**只把两个数（3 与 6）做成常量**，公式的**逐字核对不做 `raw.count`**
#         （与 Adam 的「ε 在根号外」同源，见 `optimizers/code/specs.py` 的 `UNSOURCED`）。
# 口径：★★ `Var[U(−b, b)] = b²/3` ⇒
#         · `kaiming_uniform_(mode='fan_in', nonlinearity='relu')`：`b² = 2·3/fan_in`
#           ⇒ `Var = 2/fan_in` ⇒ **与 He 同方差**（`A.3`）；
#         · `xavier_uniform_(gain=1)`：`a² = 6/(fan_in+fan_out)` ⇒ `Var = 2/(fan_in+fan_out)`
#           ⇒ **与 Glorot Eqn 12 同方差**（`A.3b`）。
# 消费者：`01_init.py` 的 `A.3` / `A.3b` / `A.3c` 与 `part_D`。
# 同值位置：无。

PYTORCH_CALCULATE_GAIN_LINEAR = 1.0
PYTORCH_CALCULATE_GAIN_TANH = 1.6666666666666667
PYTORCH_CALCULATE_GAIN_RELU = 1.4142135623730951
PYTORCH_CALCULATE_GAIN_SELU = 0.75
# 来源：**[B] PyTorch 2.14 的 `torch.nn.init` 页**（同上 URL），`calculate_gain` 的
#       **表格**（★ **计划 §3 的 `specs.py` 清单**点名要放 `calculate_gain` 表的地方）。
#       表格逐字（剥标签后）：`Linear / Identity 1` · `Conv{1,2,3}D 1` · `Sigmoid 1` ·
#       `Tanh 5/3` · `ReLU √2` · `Leaky Relu √(2/(1 + negative_slope²))` · `SELU 3/4`。
#       ⚠ **量法口径**：表里的 `5/3` 与 `√2` 在页面上是**分式与根号的渲染**
#         （`\frac{5}{3}` / `\sqrt{2}`）⇒ 本模块只存**十进制字面量**
#         （`1.6666666666666667` / `1.4142135623730951`），**幂次/开方写进注释**
#         —— 这是**盲区④**的落法（`2 ** 0.5` 整体是 `BinOp`，**不进变异分母**）。
# 口径：★ `Conv{1,2,3}D = 1` 与 `Linear = 1` 同值 ⇒ **默认 gain 下 `xavier_uniform_`
#       对卷积与全连接是同一个账**；★ 而 `kaiming_uniform_` 的默认非线性是
#       `leaky_relu`（`a=0` ⇒ gain = √2）见上。
# 消费者：`01_init.py` 的 `A.3` / `A.3c` / `C.3` 与 `part_D`。
# 同值位置：无。

PYTORCH_KAIMING_KNOWN_AS = "Also known as He initialization."
PYTORCH_XAVIER_KNOWN_AS = "Also known as Glorot initialization."
# 来源：**[B] PyTorch 2.14 的 `torch.nn.init` 页**（同上 URL），各逐字
#       （**剥标签后 2 次**：HTML 正文 1 + JSON-LD 1；原始字节 4 次）。
# 口径：★★ **PyTorch 的文档自己把两个函数与两个名字挂上了** ——
#       这是「`kaiming_uniform_` ≡ He」与「`xavier_uniform_` ≡ Glorot」这两条**同义关系**
#       的 **[B] 级依据**；★ 而 **Glorot 原文里没有 `Xavier` 这个名字**
#       （`GLOROT_XAVIER_TERM_COUNT = 0`）⇒ 两条常量**并列**时正好把 Y2 的张力摆出来：
#       **这个名字是后来由别人（He 2015 说是 Caffe）安上的**。
# 消费者：`01_init.py` 的 `A.3` / `A.3b` 与 `part_D`。
# 同值位置：无。

TORCH_CLIP_COEF_DENOM_EPS = 1e-6
# 来源：**PyTorch** 的 `torch/nn/utils/clip_grad.py`  [B]（**某实现的源码**，不是规范）
#       版本：**v2.9.0**；抓取 `https://raw.githubusercontent.com/pytorch/pytorch/v2.9.0/
#       torch/nn/utils/clip_grad.py`，2026-09-23，11427 B，`sha256[:16] = 63efb273c62fd917`
#       逐字（原始字节 2 次）：`clip_coef = max_norm / (total_norm + 1e-6)`
# 口径：**分母上的那个 `1e-6`**（`total_norm = 0` 时的保护项）
#       ⇒ 裁剪后的全局范数 = `max_norm · T/(T + 1e-6)`，**相对误差 = `1e-6/T`**。
#       ★★ **判据与阈值定义在 `90-training/autograd-and-backward/03-gradient-lifecycle.md`
#       的「表 A ⑦」（在那里写一次：`T ≥ 1`、上侧 `≤ 1.1e-6/max(T,1)`、
#       下侧 `== ε/(T+ε)` 的 `A.2b`）** —— `02` **只实现、不重定义**。
#       ⚠ **回链时必须一并带上 `A.2b`**（**T26**：只写单侧会放过一个「删掉保护项」的实现，
#         而那个实现**结果反而更精确** ⇒ Chunk 1 实测 25 条断言全绿）。
# 消费者：`02_clip_monitor.py` 的 `A.1`（两侧判据）与 `preflight()` 的钉住。
# 同值位置：`90-training/autograd-and-backward/code/specs.py`、
#       `90-training/mixed-precision/code/specs.py`、`90-training/optimizers/code/specs.py`
#       的同名常量（★ **各写一份** —— `CONTRIBUTING:36`「`code/` 必须自包含」，
#       **不设跨主题共享模块**；靠本行互指）。

PYTORCH_CLIP_DEFAULT_NORM_TYPE = 2.0
PYTORCH_CLIP_ERROR_IF_NONFINITE_DEFAULT = False
# 来源：**[B] PyTorch 2.14 的 `torch.nn.utils.clip_grad_norm_` 页**（同上 URL），签名逐字
#       （**剥标签后 1 次**）：`norm_type = 2.0` / `error_if_nonfinite = False`。
#       ★★ 完整签名（同一行内）：`clip_grad_norm_ ( parameters , max_norm , norm_type = 2.0 ,
#          error_if_nonfinite = False , foreach = None )`
#       ★ 参数说明逐字（剥标签后各 1 次）：`type of the used p-norm. Can be 'inf' for
#         infinity norm. Default: 2.0`；`if True, an error is thrown if the total norm of the
#         gradients from parameters is nan , inf , or -inf . Default: False`
# 口径：★★★ **`error_if_nonfinite` 的默认是 `False`** ⇒ **「范数是 `nan`/`inf`」这件事
#       默认不会报错**。这是本主题 `02` 的**监控判据**的核心依据：**默认的裁剪管线对
#       非有限值静默放行**，要靠监控自己用 `isfinite` 判（`A.3` / `C.2`）。
#       ★ **盲区⑤ 的落法**：本常量是 **bool**（不是 `None`）⇒ `_mutate_value` 会把它翻转成
#         `True`，断言 `== False` **当场失败**（`None` 则会被跳过、免检）。
# 消费者：`02_clip_monitor.py` 的 `A.3` / `C.2` 与 `part_D`。
# 同值位置：无。

TORCH_CLIP_RETURN_QUOTE = "Total norm of the parameter gradients (viewed as a single vector)"
# 来源：**[B] PyTorch 2.14 的 `clip_grad_norm_` 页**（同上 URL），`Returns:` 段逐字
#       （原始字节 2 次 / 剥标签后 1 次）。
# 口径：★★★ **返回值是【裁剪前】的范数** —— 这是 `02` 的 `A.2` 的依据：
#       **监控量必须取「裁剪前」的那一个**；若取裁剪后的范数，监控量**恒 ≤ `max_norm`**、
#       阈值**永远不触发**（假阴性 100%）。
# 消费者：`02_clip_monitor.py` 的 `A.2` 与 `part_D`。
# 同值位置：无。

TORCH_CLIP_SCOPE_QUOTE = \
    "as if the norms of the individual gradients were concatenated into a single vector."
# 来源：**[B] PyTorch 2.14 的 `clip_grad_norm_` 页**（同上 URL），正文逐字
#       （原始字节 2 次 / 剥标签后 1 次）。完整句（同一段内）：
#         `The norm is computed over the norms of the individual gradients of all parameters,
#          as if the norms of the individual gradients were concatenated into a single vector.`
# 口径：★★ **范数的口径是「全局」**：把全部参数的梯度当成**一个向量**。
#       ★ 它给了一个**监控侧的推论**（`B.1`）：全局范数可以由**逐张量范数**重算出来
#         （`√Σ_l ‖g_l‖²`），且**与顺序无关、与拼接方式无关** ⇒ **监控只需每层存一个标量**。
#       ★★ 反面（`A.4`）：**逐张量裁剪**（每层各自裁到 `max_norm`）在 `L` 个张量上会给出
#         一个**大 `√L` 倍**的全局范数 —— 这条判据区分了两个实现。
# 消费者：`02_clip_monitor.py` 的 `A.4` / `B.1` 与 `part_D`。
# 同值位置：无。


# ===========================================================================
# 4. ★★ μP（Tensor Programs V）—— [C] 论文（**已确认同行评审 NeurIPS 2021**）
# ===========================================================================
#
# 来源：[C] G. Yang, E. Hu, I. Babuschkin, S. Sidor, X. Liu, D. Farhi, N. Ryder,
#       J. Pachocki, W. Chen, J. Gao, **`Tensor Programs V: Tuning Large Neural Networks
#       via Zero-Shot Hyperparameter Transfer`**，arXiv **2203.03466**。
#       评审状态：**已确认同行评审** —— abs 页 `Comments:` 逐字 = `NeurIPS 2021`
#       （原始字节 1 次 —— ★ 本主题**先自己打开了 abs 页**才写这句话，**P21 已办**）。
#       ★ `<title>` 逐字 = `[2203.03466] Tensor Programs V: Tuning Large Neural Networks
#         via Zero-Shot Hyperparameter Transfer`。
#       抓取（ar5iv 正文）：`https://ar5iv.labs.arxiv.org/html/2203.03466`，2026-09-23，
#       **1045657 B**；abs 页：**43748 B**。
#       ⚠ **量法口径**：正文里 `μ \mu P` 与 `μ \mu Transfer` 都被 LaTeX + MathML 双写，
#         **`μTransfer` 连写在剥标签后计 0**（中间夹了 `\mu` 的渲染）
#         ⇒ 本模块**不存**含 `μ` 的片段。

MUP_ARXIV_ID = "2203.03466"
MUP_TITLE = "Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer"
MUP_VENUE = "NeurIPS 2021"
# 来源：[C] μP 论文（同上），abs 页 `<title>` 与 `Comments:` 字段。
#       逐字：`[2203.03466] Tensor Programs V: Tuning Large Neural Networks via Zero-Shot
#       Hyperparameter Transfer`（`<title>`，原始字节 1 次）；`NeurIPS 2021`（`Comments:`，
#       原始字节 1 次）。
# 口径：★ **P21 的落法**（编号 + 标题 + 评审状态三条一起钉住）。★ 本域另一处用到 NeurIPS
#       评审状态的常量是 `optimizers/code/specs.py` 里 Adam / AdamW 的 `Comments:` —— 同形。
# 消费者：`01_init.py` 的 `part_D`。
# 同值位置：无。

MUP_HP_STABILITY_QUOTE = "many optimal HPs remain stable even as model size changes"
# 来源：[C] μP 论文（同上），**摘要**，**原始字节 1 次**（剥标签后同样 1 次）。
# 口径：★ 这是 μP 的**主张**（不是「μP 有效」的证明）：它接着给出的是
#       `This leads to a new HP tuning paradigm we call μTransfer`。⇒ 转录时把它写成
#       **论文的主张**，不要写成「μP 已被证明能迁移超参」（**P20 的同类**）。
# 消费者：`01_init.py` 的 `A.5` 与 `part_D`。
# 同值位置：无。

MUP_COORDINATE_CHECK_QUOTE = \
    "An incorrect implementation will see some activation vector blow up or shrink to zero with width"
# 来源：[C] μP 论文（同上），**附录 D.1**，**原始字节 1 次**。
#       同段逐字（原始字节 1 次）：`implementing it correctly can in practice be error-prone`。
# 口径：★★★ **这是本主题 μP 那一节的落点**：μP 的**实用附带产物**是
#       **coordinate check**（坐标检查）—— 一个**可测的判据**：
#       「把宽度扫一遍，看每个（前）激活向量的**平均坐标尺度**会不会随宽度
#         **blow up 或 shrink to zero**」。
#       ⇒ `01_init.py` 的 `A.5` 把这条判据**落成两个可拟合的斜率**
#         （SP：`0`；把读出层的步长按宽度缩放后：`0` 而不是 `+1`），
#         **不是**复述「μP 能迁移超参」。
#       ★ 论文给的说法是「**正确的实现**下这些量稳定；**错误的实现**会 blow up 或 shrink」
#         —— 本程序只测「同一组超参在宽度上的稳定性」这一半，并**明确写出**它没有证明
#         「超参可迁移」。
# 消费者：`01_init.py` 的 `A.5` / `A.5b` 与 `part_D`。
# 同值位置：无。

MUP_ERROR_PRONE_QUOTE = "implementing it correctly can in practice be error-prone"
# 来源：[C] μP 论文（同上），附录 D.1，**原始字节 1 次**。完整句（同一行内）：
#         `Even though μP is neatly encapsulated by Table 3, implementing it correctly can in
#          practice be error-prone, just like how implementing autograd by hand can be
#          error-prone even though the math behind is just chain-rule.`
# 口径：★★ **这是「μP 实现容易写错」的一手证据** —— 论文**自己**这么说，并因此**才**给出
#       coordinate check。⇒ 本主题引用 coordinate check 的**理由**就在这一句，
#       而不是「因为 μP 很流行」。
# 消费者：`01_init.py` 的 `A.5` 与 `part_D`。
# 同值位置：无。


# ===========================================================================
# 5. ★★ loss spike —— [C] 论文（PaLM **不确认同行评审**；Spike No More **已确认 COLM 2025**）
# ===========================================================================
#
# ★★ 本节每一条都服务于**同一条纪律（P20）**：**原文显式标注的「假设」不得升格为结论**。

PALM_ARXIV_ID = "2204.02311"
PALM_TITLE = "PaLM: Scaling Language Modeling with Pathways"
PALM_SPIKE_COUNT = 20
# 来源：[C] PaLM 论文（arXiv **2204.02311**；★ 评审状态：**不确认同行评审** ——
#       abs 页 `Comments:` 字段**不存在**，实测 0 次）。抓取（ar5iv）2026-09-23，
#       **1372842 B**；abs 页 **53605 B**。
#       ★ `<title>` 逐字 = `[2204.02311] PaLM: Scaling Language Modeling with Pathways`。
#       §5.1 逐字（**原始字节 1 次**）：`spikes in the loss roughly 20 times during training`
# 口径：★ `20` 是**这一个模型（最大的那个）在这**一次训练里**的次数
#       （原文紧接着写 `These spikes occurred at highly irregular intervals, sometimes
#         happening late into training, and were not observed when training the smaller models.`）
#       ⇒ **不得**写成「训练一定会 spike 约 20 次」（**P20 的同类**：特定实验条件 ≠ 普遍规律）。
# 消费者：`03_spikes.py` 的 `part_D`（`D.5`）。
# 同值位置：无。

PALM_SPIKE_COUNT_QUOTE = "spikes in the loss roughly 20 times during training"
# 来源：[C] PaLM 论文（同上），§5.1，**原始字节 1 次**。
# 口径：★ 与 `PALM_SPIKE_COUNT` 成对：一个存数、一个存**逐字出处**（数字本身在正文里
#       可能是拼写出来的形式，量化入口只能靠这一句）。
# 消费者：`03_spikes.py` 的 `part_D`。
# 同值位置：无。

PALM_BAD_DATA_DENIAL_QUOTE = "We do not believe that the spikes were caused by"
# 来源：[C] PaLM 论文（同上），§5.1，**原始字节 1 次**。
#       完整句（★ **含弯引号** ⇒ 本常量**只取到 `by` 为止**）：
#         `We do not believe that the spikes were caused by “bad data” per se, because we ran
#          several ablation experiments where we took the batches of data that were surrounding
#          the spike, and then trained on those same data batches starting from a different,
#          earlier checkpoint. In these cases, we did not see a spike.`
# 口径：★★★ **Y12 的落法**：「PaLM 的 loss spike 是坏数据造成的」这条归因
#       **被原文逐字否认**。★ 原文还给了**它自己的**替代说法（同段逐字，**剥标签后 1 次、
#       原始字节 0 次** —— 句子中间夹了标记）：
#         `This implies that spikes only occur due to the combination of specific data batches
#          with a particular model parameter state.`
#       ⇒ 正文必须写成「**已被证伪的归因**」，并且**不得**把这个替代说法写成结论
#         （原文用的是 `This implies` —— 它是**推断**，不是机制证明）。
# 消费者：`03_spikes.py` 的 `C.2` 与 `part_D`。
# 同值位置：无。

PALM_CLIP_ENABLED_QUOTE = "despite the fact that gradient clipping was enabled"
# 来源：[C] PaLM 论文（同上），§5.1，**原始字节 1 次**。完整句（同一行内）：
#         `For the largest model, we observed spikes in the loss roughly 20 times during
#          training, despite the fact that gradient clipping was enabled.`
# 口径：★★★ **「开了 clip 就不会 spike」是错的** —— 原文**明确**说裁剪开着、
#       而 spike 仍然发生了约 20 次。★ 本主题把它落成一个**可测**的区分（`03` 的 `C.1`）：
#       **裁剪约束的是单步的梯度范数**（逐位可测），它**不约束** `Σ_t ‖Δw_t‖`
#       （闭式 `N·lr·max_norm`，随 `N` 线性增长）⇒ **逐点的上界推不出轨迹的稳定性**。
#       ★ **不得**由此写成「裁剪没用」：PaLM 自己用的缓解手段是
#         `We re-started training from a checkpoint roughly 100 steps before the spike started,
#          and skipped roughly 200–500 data batches`（**回滚 + 跳数据**），
#         而那个「为什么有效」的机制**原文没有给**。
# 消费者：`03_spikes.py` 的 `C.1` 与 `part_D`。
# 同值位置：无。

SNM_ARXIV_ID = "2312.16903"
SNM_TITLE = "Spike No More: Stabilizing the Pre-training of Large Language Models"
# 来源：[C] Spike No More（arXiv **2312.16903**；★ 评审状态：**已确认同行评审 COLM 2025**
#       —— 本次抓取的 abs 页 `Comments:` 字段实测 0 次 ⇒ **这一条本次未能复核**，
#       按本域纪律**只写「计划 §C.2 记为已确认 COLM 2025」**，不自行升格）。
#       抓取（ar5iv）2026-09-23，**395170 B**；abs 页 **40685 B**。
#       ★ `<title>` 逐字 = `[2312.16903] Spike No More: Stabilizing the Pre-training of
#         Large Language Models`。
# 口径：★ 与本模块其他 [C] 条目同形：**编号 + 标题成对存放**。
# 消费者：`03_spikes.py` 的 `part_D`。
# 同值位置：无。

SNM_ASSUMPTION_QUOTE = \
    "Based on the assumption that the loss spike is caused by the sudden growth of the gradient norm"
# 来源：[C] Spike No More（同上），**摘要**，**原始字节 1 次**（剥标签后同样 1 次）。
# 口径：★★★ **Y13 / P20 的落法**：「loss spike 由梯度范数突增引起」在**本域取到的最好的
#       一手来源里，仍被显式标注为假设**（`Based on the assumption that …`）。
#       ⇒ 正文**必须**把它写成**假设**，不得写成「成因是…」。
#       ★ **量法口径**：同页 `Based on the assumption that` 共 **2** 次
#         （另一处在附录的公式推导里，讲的是「参数与向量服从正态分布」这个**另一个**假设）
#         ⇒ 引用时**必须带上后半句**（本常量就是完整的 1 次那一条）。
# 消费者：`03_spikes.py` 的 `A.1` 与 `part_D`。
# 同值位置：无。

SNM_CONDITION_QUOTE = \
    "Our findings suggest that stabilizing the pre-training process requires two conditions: small sub-layers and large shortcut."
# 来源：[C] Spike No More（同上），**摘要**，**原始字节 1 次**。
#       ★ 同页 `small sub-layers and large shortcut` 共 **2** 次（摘要 + 正文§1）。
# 口径：★★ **两条限定必须一起写**：
#       **(a)** 论文自己的措辞是 `Our findings suggest`（**建议/发现**，不是定理）；
#       **(b)** 它给的是 `requires two conditions`（**条件**），而**计划 §C.2 的 Y13 把它
#       转述成「充分条件」—— 原文里没有 `sufficient condition` 这个词**（实测 0 次）
#       ⇒ 本主题**不沿用那个转述**，只写原文的两句逐字。
#       ⇒ 结论：`small sub-layers and large shortcut` 是**论文建议的两个条件**，
#         **不是**「loss spike 的机制已被证明」。
# 消费者：`03_spikes.py` 的 `A.1` / `C.3` 与 `part_D`。
# 同值位置：无。


# ===========================================================================
# 6. Adam 的默认超参（[C] 1412.6980，**已确认 ICLR 2015**）—— ★ **本主题只消费、不重定义**
# ===========================================================================
#
# 来源与逐字见 `90-training/optimizers/code/specs.py` 的 `ADAM_DEFAULT_*`（**正文在彼**）。
# ★ 本主题引它的**唯一理由**：`03` 的 spike 形态由 `β1 / β2 / ε` 决定 ——
#   尤其是 `β2`（记忆长度 `≈ 1/(1−β2)`）。⇒ 本模块**各写一份**，靠 `# 同值位置：` 互指。

ADAM_DEFAULT_BETA1 = 0.9
ADAM_DEFAULT_BETA2 = 0.999
ADAM_DEFAULT_EPS = 1e-8
# 来源：[C] Adam 论文（arXiv **1412.6980**，**已确认 ICLR 2015**；**PDF 文本层**逐字
#       `Good default settings for the tested machine learning problems are …`）。
# 口径：★ 论文的措辞是「**在作者试过的任务上**好用」⇒ 不是规范（与 `optimizers` 同）。
#       ★★ 本主题只用 `β2` 这一个：**`v` 的衰减因子是 `β2^k`** ⇒ 「记忆长度 ≈ `1/(1−β2)`」
#         这句话在 `03` 里是**可测的**（`C.2`：`β2 = 0.999` 上 `β2^1000 = 0.3677`）。
# 消费者：`03_spikes.py` 的 `A.1` / `A.2` / `C.2` 与 `part_D`。
# 同值位置：`90-training/optimizers/code/specs.py` 的 `ADAM_DEFAULT_BETA1 / _BETA2 / _EPS`
#       （**同值**，各写一份，靠本行互指）。


# ===========================================================================
# 7. UNSOURCED —— 待办清单，不是免责声明
# ===========================================================================
#
# ★ `CONTRIBUTING` §2：「不得**既保留数字又记 UNSOURCED**」—— 下面每一项都**没有**出现在
#   本模块的正文常量里。
# ★★ 提醒（T11 盲区①）：本元组的字符串**没有消费者** ⇒ 它们会是 `make mutate` 的**漏网**候选。
#    这是本库的既有形态（先例：`80-model-architectures/transformer/code/specs.py` 同节、
#    `90-training/optimizers/code/specs.py` 同节），三个编号程序**各钉一次**
#    （`part_D` 里逐位比较），**不给它开后门**。

UNSOURCED = (
    # 每一项是一个**稳定标识符**（不是散文）。
    "caffe_xavier_term_origin",             # ★★ Y2 的**缺口**：He 2015 逐字说
                                            # `This is called “Xavier” initialization in Jia2014`，
                                            # 而 `Jia2014` = **Caffe**（`our modification of the
                                            # Caffe library Jia2014`）⇒ **「这个名字出自 Caffe」这件事
                                            # 本次无法在一手工件上核实**（`raw.githubusercontent.com`
                                            # 取不到）⇒ 只能写「He 2015 自称引自 Caffe」。
    "glorot_normalized_init_count",          # ★★ Y2 的**不可复现计数**：
                                            # `normalized initialization` 的计数本次四种数法给
                                            # **13 / 14 / 17 / 18** ⇒ 按本域口径纪律**不写这个数**
                                            # （而 `Xavier initialization` = 0 是稳健的，已入库）。
    "pytorch_init_bound_formula_raw_count",  # `kaiming_uniform_` 的 `bound = gain × √(3/fan_mode)`
                                            # 与 `xavier_uniform_` 的 `a = gain × √(6/(fan_in+fan_out))`
                                            # 在页面上是 **LaTeX + MathML 双写** ⇒
                                            # **`raw.count` 计 0**（与 Adam 的「ε 在根号外」同源）
                                            # ⇒ 本模块只把 `3` / `6` 做成常量，公式不做引文核对。
    "pytorch_conv_fan_definition",          # `torch.nn.init._calculate_fan_in_and_fan_out` 对卷积的
                                            # `fan_in = c_in · k²` **本次没有取到该页的逐字**
                                            # ⇒ 本模块只用 **He 2015 的逐字**（`the number of
                                            # connections`）来支撑卷积 `fan_in` 含 `k²`，
                                            # **不引 PyTorch 的源码口径**。
    "mup_learning_rate_table_scaling",      # μP 论文 Table 3 的 **LR 缩放列**在 ar5iv 上是
                                            # 表格**扁平化**后的文本（列序无法可靠复原）
                                            # ⇒ 本主题**不引用**那张表的 LR 数值；
                                            # `01` 的 `A.5` 用**自己推导的闭式**（见该处注释），
                                            # 论文只提供「判据形态」那一句（coordinate check）。
    "spike_mechanism_one_source_conclusion", # ★★ **Z2**：loss spike 的**机制层**归因
                                            # **没有可定位的一手结论** ⇒ 本域只写
                                            # 「现象 + 已被证伪的归因 + 仍是假设的归因」。
    "snm_peer_review_status_recheck",       # `03` 引的 Spike No More 的同行评审状态
                                            # （计划 §C.2 记 COLM 2025）**本次未在 abs 页复核到**
                                            # ⇒ 只写「计划记为已确认 COLM 2025」，不自行升格。
)

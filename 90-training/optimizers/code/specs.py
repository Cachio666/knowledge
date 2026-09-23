"""90-training/optimizers 的外部事实常量源。

本模块是 `CONTRIBUTING` §2 约定的**共享事实源**（`specs.py`）：本主题的三个编号程序从它读常量，
推导过程可复核，**常量本身不可自证**。

★★ **来源必须分级，并写明它是哪一级**（计划 §1.5）。本域的头号错误模式是
  **P19 —— 把「某实现的文档」当成「规范的文档」**（B 级当 A 级）。
  本主题的两条链**分属两级，必须分开写**：
    · **Adam / AdamW 的更新式与默认超参** → **[C] 论文**（两篇都**已确认同行评审**）；
    · **`torch.optim.Adam` / `AdamW` / `SGD` 的签名与伪代码** → **[B] PyTorch 2.14 的文档**
      （**某实现的规格**，**不是规范**）⇒ 引用时一律写明版本。
  ★★ 本主题**另有一条「同名不同指」**（计划 §0.7-C 的 **Y5**）：
    「AdamW 与 Adam+L2 差在哪」这个问题，**论文与 PyTorch 文档给的归因不是同一句话** ——
    两个来源都要引、都要标出处，**不得把 PyTorch 的措辞当论文的理由**（规范与实现不分）。

★ 三类真值来源**必须分开标注**（计划 §1.5）：
    ① **外部规格值**：论文原文（**C 级，须注明是否确认同行评审**）、厂商/产品文档（A/B 级）
       ⇒ 本模块存放的就是这一类，**每条带 `# 来源：` + `[A-E]` 级 + 抓取日**。
    ② **本机实测值**：本域程序在本机跑出的数（**须写 python3 与 numpy 版本**）⇒ 不放在本模块。
    ③ **推导值**：程序算出的量（可手算复核）⇒ **不放在本模块**
       （`CONTRIBUTING` §2：「只放从来源抄来的数，**不放推导值**」）。
       ★ 本主题据此**刻意不搬**的一条：**`(1 − lr·wd)^n` 这个闭式**（计划 §0.7-D 的 **D1**）——
        它是**推导值**，消费者是 `02_adam_adamw.py` 的 `A.1`；放进来只会变成
        「常量 == 常量」的恒真断言（`D2` 还专门证明它**只在「梯度恒为 0」时成立**）。

★ **只放本主题真的会断言的常量**（计划 §3 的 R3 裁定）。每一条下面写明**预期消费者**。
  ★ **没有消费者的常量就是 `make mutate` 的漏网候选**（计划 §4 **T11 的盲区①**）。

★★ **本模块的一条自缚纪律（盲区④的落法）**：`mutate.py` 的 `_walk()`
  **不下钻 `BinOp`** ⇒ `2.0 ** -6` 这类**幂次字面量整体不进变异分母**。
  ⇒ 本模块**所有数值一律写成十进制字面量，幂次/换算写进注释**（`2^-25` 写作
    `2.9802322387695312e-08` 这样）。`0.9`/`0.999`/`1e-8` 与 `0.01` 都已是字面量。

★ 每条常量带 `# 来源：` + `[A-E]` 级 + 抓取日；跨主题/跨文件同值的另带 `# 同值位置：`。
  分级判据见 `CONTRIBUTING` §2（A 厂商规范 / B 产品官方文档 / C 论文 / D 社区 / E 本库推导）。

★★ **量法口径（本模块逐条照办，取自本域 §0.3 与计划 §4 的 T18/T34）**：
  本模块所有「计数 N 次」都是**剥标签 + 折叠空白后**在**指定载体**上数的：
    ① **跨标记**的引文在原始字节上必然计不到（Apex 那类 `<span>` 问题）；
    ② **标点后的空格数**会变（Apex 是 `O0</code>:  FP32`，**两个**空格）；
    ③ **引文片段不能含跨行空白**（`from 48G` 计 0、`48G to 7G` 计 1）；
    ④ **同一条引文在不同页面的计数可能不同**（abs 页 vs ar5iv 正文）⇒ **写明是哪个页面**；
    ⑤ **数字可能带千位分隔符**。
  ⇒ 本模块每条引文都写出**是哪个载体**（abs 页 / ar5iv / PDF 文本层 / docs 页），
    且**引文片段一律不含换行**。
"""

# ===========================================================================
# 1. Adam 的四个默认超参 —— [C] 原论文（**已确认同行评审**）
# ===========================================================================
#
# 来源：[C] D. P. Kingma & J. Ba, **`Adam: A Method for Stochastic Optimization`**，
#       arXiv **1412.6980**。
#       评审状态：★★ **已确认同行评审** —— 判据是**实测**：该文 arXiv **abs 页**的
#       `Comments:` 字段逐字 = `Published as a conference paper at the 3rd International
#       Conference for Learning Representations, San Diego, 2015`（原始字节 1 次）。
#       ⚠ **整句含换行**？不 —— 该字段在 abs 页的 HTML 里是**一行**；本模块 2026-09-23 复核过。
#       ★ **P21 已办**：`<title>` 逐字 = `[1412.6980] Adam: A Method for Stochastic Optimization`。
# 抓取：`https://arxiv.org/abs/1412.6980`，2026-09-23，**43245 B**，`sha256[:16] = 4129fc8bd4e092ed`。
#
# ★★ **方法学让步（计划 §4 的 T5，必须在正文里写明）**：
#   该文的 **ar5iv 页只有 226 字符**（自述无 LaTeX 源可转）⇒ 本主题的逐字引文
#   **取自 PDF 的文本层**（`https://arxiv.org/pdf/1412.6980`，**584641 B**，
#   `sha256[:16] = eab9c73ae2ceda88`；用 `pdftotext`）。
#   ⚠ 文本层里**分式与根号是跨行排版**的（`√` 单独占一行）⇒
#     **凡含公式的句子一律不做 `raw.count` 引文核对**，改由**数值断言**承担可证伪性
#     （见 `02_adam_adamw.py` 的 `A.5`）。

ADAM_DEFAULT_LR = 0.001
ADAM_DEFAULT_BETA1 = 0.9
ADAM_DEFAULT_BETA2 = 0.999
ADAM_DEFAULT_EPS = 1e-8
# 来源：[C] Adam 论文（同上），**Algorithm 1** 的说明句。逐字（**PDF 文本层** 2 次，
#       同一段在正文与算法框各出现一次）：
#         `Good default settings for the tested machine learning problems are α = 0.001,
#          β1 = 0.9, β2 = 0.999 and ε = 10−8 .`
#       ⚠ **量法口径**：原文里的 ε 在 `pdftotext` 输出里是**一个控制字符**（`\x0f`），
#         所以「`and ε = 10−8`」这一段**数不到** —— 上面那句是**按渲染读出来的**。
#         ⇒ 本模块**只把四个数值**做成常量，**引文只取不含公式的前半**（见 `ADAM_DEFAULTS_QUOTE`）。
# 口径：★ 论文自己的措辞是 **`Good default settings for the tested machine learning problems`**
#       ⇒ 它是**「在作者试过的任务上好用」**，**不是「规范规定」**（**P20 的同类**：
#       不要把「作者试过的」升格成「规定的」）。★ 这一点与 PyTorch 的实现默认值一致
#       （`lr=0.001, betas=(0.9, 0.999), eps=1e-08`），但**两者不是同一个论据**。
# 消费者：`02_adam_adamw.py` 的 `A.1`（Adam 系实现的四个超参直接取它）、`part_D`（逐位钉住）。
# 同值位置：`code/02_adam_adamw.py` 的 `BETA1` / `BETA2` / `EPS`（**同值**，
#       `CONTRIBUTING:36`「`code/` 必须自包含」⇒ 各写一份，靠本行互指）。

ADAM_DEFAULTS_QUOTE = "Good default settings for the tested machine learning problems are"
# 来源：[C] Adam 论文（同上），**PDF 文本层**逐字 2 次。
#       ⚠ **整句含公式**（`α = 0.001, β1 = 0.9, …`，其中 ε 那个字形在文本层里丢了）
#         ⇒ 本常量**只取公式之前的连续片段**（不含换行、不含公式），
#           四个数值另存为上面四个常量（本域 §0.3 的纪律 ③）。
# 口径：★★ 这是**「论文给了默认值」这件事的逐字证据本身**，也是**引文口径纪律的现场**：
#       「取到原文」≠「读对原文」——**公式那一段取不到**，所以它的一部分只能**按渲染读**。
# 消费者：`02_adam_adamw.py` 的 `part_D`（`D.5`：逐位钉住 + 说明公式部分的口径）。
# 同值位置：无。


# ===========================================================================
# 2. AdamW —— [C] 原论文（**已确认同行评审**）；★ 它的归因与 PyTorch 的**不是同一句话**
# ===========================================================================
#
# 来源：[C] I. Loshchilov & F. Hutter, **`Decoupled Weight Decay Regularization`**，
#       arXiv **1711.05101**。评审状态：★★ **已确认同行评审** —— abs 页 `Comments:` 逐字
#       = `Published as a conference paper at ICLR 2019`（原始字节 1 次）。
#       ★ **P21 已办**：`<title>` 逐字 = `[1711.05101] Decoupled Weight Decay Regularization`。
# 抓取：`https://arxiv.org/abs/1711.05101`，2026-09-23，**43412 B**，`sha256[:16] = e2402d8869fa2333`；
#       PDF：`https://arxiv.org/pdf/1711.05101`，**6063896 B**，`sha256[:16] = 9d5af48931e773d6`
#       （文本层用 `pdftotext`，47371 字符）。

ADAMW_DEFAULT_WEIGHT_DECAY = 0.01
# 来源：**两处都是同一个数，但只有一处是「默认值」**：
#       · [C] AdamW 论文的 **Algorithm 2** 第 1 行的 `λ ∈ IR` —— ★ 那是**符号域**
#         （`λ` 是任意实数），**不是默认值**；
#       · [B] **PyTorch 2.14 的 `AdamW` 签名**逐字（剥标签后 2 次，同一页面 HTML 与 JSON-LD 各 1 处）：
#         `class torch.optim.AdamW(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08,
#          weight_decay=0.01, amsgrad=False, ...)`。
# 口径：★★ **`0.01` 是 PyTorch 的默认值（[B]），不是论文的规定（[C]）** —— 论文只给 `λ` 的符号域。
#       ★ 与下面 `PYTORCH_ADAM_DEFAULT_WEIGHT_DECAY = 0` 成对照：**同一个框架里两个类的默认值不同**。
# 消费者：`02_adam_adamw.py` 的 `A.6`（pair 对照）与 `part_D` 的逐位钉住。
# 同值位置：无（`0.01` 在本主题只出现在这里；程序里的 `WD_LIST[1]` 是**另一个角色**，见该处注释）。

ADAMW_L2_ATTRIBUTION_QUOTE = \
    "weights x with large typical gradient magnitude s are regularized by a smaller relative"
# 来源：[C] AdamW 论文（同上），§2（**PDF 文本层**逐字 1 次）。
#       ⚠ **整句跨行**（原文在 `relative` 之后换行到 `amount than other weights`）⇒
#         本常量只取**到换行前**为止的片段（本域 §0.3 的纪律 ③）。
#       ★ 同段的**正面句**逐字（PDF 文本层 1 次，同一行内）：
#         `whereas with decoupled weight decay, only the gradients of the loss function`
# 口径：★★ 这就是 **Y5 的论文侧归因**：**L2 的梯度也被自适应归一化了** ——
#       所以「典型梯度大」的权重**被正则得更少**（相对量）。
#       ⚠ 它与 PyTorch 的措辞（`weight decay does not accumulate in the momentum nor
#         variance`）**不矛盾，但不是同一句话** ⇒ 转述时把 PyTorch 的措辞当论文的理由
#         = **规范与实现不分**（**Y5**）。
# 消费者：`02_adam_adamw.py` 的 `part_D`（`D.7`：与 PyTorch 那条**并列**钉住）。
# 同值位置：无。

ADAMW_MECHANISM_QUOTE = \
    "whereas with decoupled weight decay, only the gradients of the loss function"
# 来源：[C] AdamW 论文（同上），§2，**PDF 文本层**逐字 1 次（同一行内，不含换行）。
# 口径：★ 论文对「解耦后差别在哪」的**正面表述**：**只有损失函数的梯度被自适应**，
#       而 weight decay 那一步**在自适应机制之外**。
# 消费者：`02_adam_adamw.py` 的 `part_D`（`D.7b`）。
# 同值位置：无。

ADAMW_TEST_ERROR_QUOTE = "relative improvement in test error"
# 来源：[C] AdamW 论文（同上），§4，**PDF 文本层**逐字 1 次。完整句（同一行内，不含换行）：
#         `For the default learning rate of 0.001, AdamW achieved 15% relative improvement
#          in test error compared to Adam` —— ⚠ 后半句跨行，故本常量取到 `error` 为止。
#       ★ `15%` 这个数在同页出现 **2** 次（另一处是 `AdamWR achieved the same`），
#         但**只有本句把它归给 AdamW vs Adam**。
# 口径：★★ **这个 `15%` 只属于 CIFAR-10 与 ImageNet32x32 两个数据集、`lr = 0.001`** ——
#       原文逐字写的是 `on CIFAR-10 … and ImageNet32x32`。⇒ **不得写成「AdamW 普遍好 15%」**
#       （**P20 的同类**：把特定实验条件升格成普遍结论）。
# 消费者：`02_adam_adamw.py` 的 `part_D`（`D.8`）。
# 同值位置：无。


# ===========================================================================
# 3. ★★ [B] PyTorch 2.14 的规格 —— **某实现的文档，不是规范**（P19 的纪律）
# ===========================================================================
#
# 抓取：2026-09-23，`https://docs.pytorch.org/docs/2.14/generated/torch.optim.<类>.html`
#       （★ **取法**：`/docs/stable/...` 返回的是 `<meta http-equiv="refresh">` 跳转壳，
#        `curl -L` 跟的是 HTTP 重定向、**跟不了 meta refresh** ⇒ 直接取 `2.14` 的路径；
#        本域 §0.3 与计划 §4 的 **T20**）：
#         · `torch.optim.Adam.html`  **279768 B**，`sha256[:16] = 06331523d46939c3`
#         · `torch.optim.AdamW.html` **278114 B**，`sha256[:16] = 86b194838a86abe0`
#         · `torch.optim.SGD.html`  **274338 B**，`sha256[:16] = 33db1ab300a3476c`

PYTORCH_ADAM_DECOUPLED_FLAG = "decoupled_weight_decay=False"
# 来源：**[B] PyTorch 2.14 的 `torch.optim.Adam` 文档**（同上 URL 与哈希）。
#       逐字（**剥标签 + 折叠空白后 2 次**：同一页面的 HTML 正文 1 处 + JSON-LD 的
#       `articleBody` 重复 1 处）。完整签名（同一行内，不含换行）：
#         `class torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08,
#          weight_decay=0, amsgrad=False, *, …, decoupled_weight_decay=False)`
# 口径：★★ 这就是 **Y6 的逐字证据**：「**`torch.optim.Adam` + `weight_decay` 就是 AdamW**」
#       **是错的** —— 默认是 `False`，要解耦必须**显式**传 `decoupled_weight_decay=True`
#       或用 `AdamW` 类。
#       ★ 同页伪代码逐字（**同一段、不含换行**，剥标签后 1 次）：`if λ≠0gt←gt+λθt−1`
#         ⇒ **λ 进梯度、进而进 `m` 和 `v`**；而 AdamW 页的伪代码是
#         `θt←θt−1−γλθt−1` **先于** `mt←…` ⇒ 两条链的**顺序**就不一样。
# 消费者：`02_adam_adamw.py` 的 `A.7`（用两条链的**顺序差别**做判据）与 `part_D` 的 `D.6`。
# 同值位置：无。

PYTORCH_ADAM_DEFAULT_WEIGHT_DECAY = 0
# 来源：**[B] PyTorch 2.14 的 `torch.optim.Adam` 签名**（同上 URL 与哈希），逐字（剥标签后 2 次）：
#         `weight_decay=0`
# 口径：★★ 与 `ADAMW_DEFAULT_WEIGHT_DECAY = 0.01` **成对**读：**同一个框架、同一个语义的旋钮、
#       两个类给了不同的默认值** ⇒ 「用哪个类」这件事**不只是名字**。
#       ★ 这两个常量**必须成对消费**，单独一个只说了一半。
# 消费者：`02_adam_adamw.py` 的 `A.6`（`(0, 0.01)` 成对断言）与 `part_D`。
# 同值位置：无。

PYTORCH_ADAMW_NO_ACCUMULATE_QUOTE = \
    "weight decay does not accumulate in the momentum nor variance"
# 来源：**[B] PyTorch 2.14 的 `torch.optim.AdamW` 文档**（同上 URL 与哈希）。
#       逐字（剥标签后 **2 次**；同一页面 HTML 与 JSON-LD 各 1 处）。完整句（同一行内，不含换行）：
#         `Implements AdamW algorithm, where weight decay does not accumulate in the
#          momentum nor variance.`
# 口径：★★ 这就是 **Y5 的 PyTorch 侧归因**。它与 `ADAMW_L2_ATTRIBUTION_QUOTE`（论文侧）
#       **不矛盾但不是同一句话**：论文说的是「**L2 的梯度被自适应归一化**」，
#       PyTorch 说的是「**不进动量与方差**」。⇒ **两条都引、各标出处**，
#       **不得把 PyTorch 的措辞写成论文的理由**。
# 消费者：`02_adam_adamw.py` 的 `part_D`（`D.7`：与论文侧那条**并排**钉住）。
# 同值位置：无。

PYTORCH_SGD_SIGNATURE = \
    "class torch.optim.SGD(params, lr=0.001, momentum=0, dampening=0, weight_decay=0, nesterov=False"
# 来源：**[B] PyTorch 2.14 的 `torch.optim.SGD` 文档**（同上 URL 与哈希）。
#       逐字（剥标签后 2 次：HTML 正文 1 + JSON-LD 1）。
# 口径：★ 两个默认值对本文（`01`）是设定：`momentum=0`、**`nesterov=False`**。
#       ★★ 同页伪代码逐字（**同一段、不含换行**，剥标签后 1 次）：
#         `if nesterovgt←gt+μbtelsegt←bt` —— 即
#         `if nesterov: g_t ← g_t + μ b_t  else: g_t ← b_t`，然后 `θ_t ← θ_{t−1} − γ g_t`。
#         ⇒ ★ 这就是 `01` 的 `A.5` 的**两条链的逐字定义**（[B] 级，写明实现与版本）。
#       ⚠ 同一段里缓冲区的递推带 `dampening`（`b_t ← μ b_{t−1} + (1−τ) g_t`，且 `t=1` 时
#         `b_t ← g_t`）—— 本主题的程序取 `τ = 0`（默认），**这一点在正文里写明**。
# 消费者：`01_sgd_momentum.py` 的 `A.5`（两种写法之差）与 `part_D`。
# 同值位置：无。

PYTORCH_SGD_NESTEROV_ATTRIBUTION = \
    "Nesterov momentum is based on the formula from On the importance of initialization and momentum in deep learning"
# 来源：**[B] PyTorch 2.14 的 `torch.optim.SGD` 文档**（同上 URL 与哈希），逐字（剥标签后 2 次）。
# 口径：★★ 这是**「Nesterov」这个名字在 PyTorch 里的出处**：它**不是**引 Nesterov 1983，
#       而是引 **Sutskever et al. 2013 的 `On the importance of initialization and momentum
#       in deep learning`**（ICML）。⇒ 转述「Nesterov 动量」时**不要凭名字往 1983 年那篇上挂**：
#       框架里的那条式子是**这一篇**的形态（**P21 的同类：先看它到底指向谁**）。
# 消费者：`01_sgd_momentum.py` 的 `part_D`（`D.9`）。
# 同值位置：无。


# ===========================================================================
# 4. ★ Muon —— [C]（2026-09-23 补取；**不确认同行评审**）
# ===========================================================================
#
# 来源：[C] J. Liu et al., **`Muon is Scalable for LLM Training`**，arXiv **2502.16982**。
#       评审状态：**不确认同行评审** —— abs 页 `Comments:` 字段 **不存在**（实测 0 次）。
#       ★ **P21 已办**：`<title>` 逐字 = `[2502.16982] Muon is Scalable for LLM Training`。
#       抓取（ar5iv）：`https://ar5iv.labs.arxiv.org/html/2502.16982`，2026-09-23，
#       **281289 B**，`sha256[:16] = 8471ea44026fa7ed` —— ★ 与计划 §C.2b 记的**逐位相同**。
#       abs 页：**45291 B**，`sha256[:16] = 96f8d85d12832ef0`。
#       ⚠ **量法口径**：剥标签 + 折叠空白后，`orthogonal` 计 **8** 次、
#         `Newton-Schulz` 计 **13** 次（计划 §C.2b 记的是 9 / 16 —— **数法不同**，
#         与 `Y2` 的「`normalized initialization` 17 次不可复现」是同一类）。
#         ⇒ ★ 本模块**不把这两个计数当证据**，只作定性记录。

MUON_UPDATE_RMS = 0.2
MUON_ADAMW_UPDATE_RMS_LO = 0.2
MUON_ADAMW_UPDATE_RMS_HI = 0.4
# 来源：[C] Muon 论文（同上），§3 与 Table 8 附近。逐字（剥标签后各 1 次）：
#         `These findings are consistent with our empirical observation that AdamW's update
#          RMS is in the range of 0.2∼0.4. We opted to control the update RMS of Muon to 0.2.`
#       （本模块只取两个数；`0.2∼0.4` 拆成 LO/HI 两个常量，便于逐位钉住。）
#       ★ 同文**摘要**逐字（剥标签后 1 次）：`We identify two crucial techniques for scaling up
#         Muon: (1) adding weight decay and (2) carefully adjusting the per-parameter update scale.`
# 口径：★★ **这是本主题「更新量的量级」这一格的唯一外部锚点**：
#       Muon 是**按「更新量的 RMS」来调自己**的 —— 把 RMS 对齐到 AdamW 的量级（0.2~0.4）。
#       ⚠★ **口径警告**：论文**没有**说它量的是「含不含 `lr`」的那个量；
#       本机在玩具规模上量到的 `RMS(m̂/(√v̂+ε))`（**归一化后的更新量**）与它**不是同一个量**
#       ⇒ `02_adam_adamw.py` 的 `B.3` **只并列打印、不做比值断言**（**「量法口径」纪律**）。
# 消费者：`02_adam_adamw.py` 的 `B.3`（并列打印 + 口径说明）与 `part_D` 的逐位钉住。
# 同值位置：无。★ 这两个数**只有「钉住」这一个消费者**，那是**有意的**：
#   它们是 [C] 级的外部锚点，本主题**没有任何本机量能与它们做等量对齐**。


# ===========================================================================
# 5. 裁剪分母的保护项 ε —— ★ **本主题只消费、不重定义**（回链 `autograd-and-backward/03`）
# ===========================================================================
TORCH_CLIP_COEF_DENOM_EPS = 1e-6
# 来源：**PyTorch** 的 `torch/nn/utils/clip_grad.py`  [B]（**某实现的源码**，不是规范）
#       版本：**v2.9.0**；抓取 `https://raw.githubusercontent.com/pytorch/pytorch/v2.9.0/
#       torch/nn/utils/clip_grad.py`，2026-09-23，11427 B，`sha256[:16] = 63efb273c62fd917`
#       逐字（原始字节 2 次）：`clip_coef = max_norm / (total_norm + 1e-6)`
# 口径：**分母上的那个 `1e-6`**（`total_norm = 0` 时的保护项）
#       ⇒ 裁剪后的全局范数 = `max_norm · T/(T + 1e-6)`，**相对误差 = `1e-6/T`**。
#       ★★ **判据与阈值定义在 `90-training/autograd-and-backward/03-gradient-lifecycle.md`
#       的「表 A ⑦」（在那里写一次：`T ≥ 1`、上侧 `≤ 1.1e-6/max(T,1)`、
#       下侧 `== ε/(T+ε)` 的 `A.2b`）** —— 本主题**只实现、不重定义**。
#       ⚠ **回链时必须一并带上 `A.2b`**（**T26**：只写单侧会放过一个「删掉保护项」的实现，
#         而那个实现**结果反而更精确** ⇒ Chunk 1 实测 25 条断言全绿）。
# 消费者：`03_schedule_clip.py` 的 `A.3`（两侧判据）与 `preflight()` 的钉住。
# 同值位置：`90-training/autograd-and-backward/code/specs.py` 的 `TORCH_CLIP_COEF_DENOM_EPS`、
#       `90-training/mixed-precision/code/specs.py` 的同名常量（★ **各写一份** ——
#       `CONTRIBUTING:36`「`code/` 必须自包含」，**不设跨主题共享模块**；靠本行互指）。


# ===========================================================================
# 6. 有来源、但**不作模块级常量**的条目（★ 理由：没有消费者 ⇒ 只会成为漏网）
# ===========================================================================
#
# ★★ 计划 §4 **T11 的盲区①**：`80` 实测「只用于打印的模块级常量表」制造过 **2** 条漏网。
#    对策是「要么给它消费者、要么删掉」。下面几条**有来源但无消费者**，因此**写在注释里**：
#    它们由**文档**引用、由**人**核对（判据表的「引用判据」本来就是人工核对的口径）。
#
# (a) **Adam 论文 Algorithm 1 的更新式**（[C]，**PDF 文本层**）：
#       `θt ← θt−1 − α · m̂t /(√v̂t + ε)` —— ★ **ε 在根号外**。
#     ⚠ **为什么不搬成常量**：文本层里 `√` **单独占一行**（分式排版），
#       含公式的片段**跨行** ⇒ 按本域 §0.3 纪律 ③，**不能**用它做 `raw.count` 钉住；
#       而只取「不含公式的碎片」又成了没有信息的字符串。
#     ⇒ 它的**可证伪落点**是 `02_adam_adamw.py` 的 `A.5`（数值断言：两种写法给出不同的数）。
#     ★ 同一形态在 **[B] PyTorch 的 Adam / AdamW 伪代码**里：`θt←θt−1−γmt^/(vt^+ϵ)` ——
#       那里的 `√` 是 MathML 的 `<msqrt>` 元素，**剥标签后整个消失** ⇒ 同上。
#
# (b) **PyTorch AdamW 页的伪代码顺序**（[B]，逐字，剥标签后同一段内各 1 次）：
#       AdamW: `θt←θt−1−γλθt−1mt←β1mt−1+(1−β1)gt…`（**decay 那一步在 `m` 之前**）；
#       Adam : `if λ≠0gt←gt+λθt−1mt←β1mt−1+(1−β1)gt…`（**λ 先进梯度**）。
#     去向：`02` 的 `A.7` 用「顺序差别」做判据（**不是**字符串钉住）。
#
# (c) **AdamW 论文 Algorithm 2 第 11 行**（[C]，PDF 文本层）：
#       `ηt ← SetScheduleMultiplier(t)`，注释逐字 `can be fixed, decay, or also be used for
#        warm restarts` —— ★ **论文自己把「学习率调度」写进优化器的算法框**里，
#       这正是 `03_schedule_clip.py` 与 `02` 的**接口**。
#     去向：`03-schedules-and-clipping.md` 的 §0 与 `02-adam-and-adamw.md` 的 §6。
#     ⚠ 同样**不搬成常量**（它没有断言消费者，且含跨行的注释片段）。


# ===========================================================================
# 7. UNSOURCED —— 待办清单，不是免责声明
# ===========================================================================
#
# ★ `CONTRIBUTING` §2：「不得**既保留数字又记 UNSOURCED**」—— 下面每一项都**没有**出现在
#   本模块的正文常量里。
# ★★ 提醒（T11 盲区①）：本元组的字符串**没有消费者** ⇒ 它们会是 `make mutate` 的**漏网**候选。
#    这是本库的既有形态（先例：`80-model-architectures/transformer/code/specs.py` 同节、
#    `90-training/mixed-precision/code/specs.py` 同节），三个编号程序**各钉一次**
#    （`part_D` 里逐位比较），**不给它开后门**。

UNSOURCED = (
    # 每一项是一个**稳定标识符**（不是散文）。
    "adam_eps_position_raw_count",          # 「ε 在根号外」这句话**不能在原始字节上做引文核对**：
                                            # Adam 论文的 PDF 文本层里 `√` **单独占一行**（分式排版），
                                            # PyTorch 的 HTML 里 `√` 是 `<msqrt>` 元素
                                            # （剥标签后消失）。⇒ **本主题不写 `raw.count`**，
                                            # 可证伪性由 `02` 的 `A.5`（数值断言）承担。
    "pytorch_adam_float32_eps_version_span",  # `torch.optim.Adam` 的 `eps=1e-08` 与
                                            # 论文的 `ε = 10−8` **数值相同**，但「某个实现的默认值
                                            # 在别的版本上是否恒定」**未逐版核对** ⇒ 引用时写版本。
    "muon_orthogonalization_iteration_count",  # Muon 的 Newton-Schulz **系数与迭代次数**
                                            # 本主题**不搬**（它是 `2502.16982` 的正文，
                                            # 而本主题只把 Muon 用作 §3「实现与演进」的一节），
                                            # 且剥标签计数不可复现（8/13 vs 9/16）。
    "muon_update_rms_measurement_convention",  # 论文的 `update RMS` **没写**它含不含 `lr`、
                                            # 逐张量还是逐元素 ⇒ 本主题**不拿本机量去对齐它**
                                            # （只并列打印，见 `02` 的 `B.3`）。
)

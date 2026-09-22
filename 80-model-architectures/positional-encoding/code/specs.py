"""80-model-architectures/positional-encoding 的外部事实常量源。

本模块是 `CONTRIBUTING` §2 约定的**共享事实源**（`specs.py`）：本主题的编号程序从它读常量，
推导过程可复核，**常量本身不可自证**。

★★ **本模块的约定：来源分三类，且必须分开标注**（计划 §C.4 的分组纪律；域 README §「代码约定」）：

  · **第一类（镜像）**：**不在** `45-kernel-lifecycle/requirement-discovery/code/specs.py` 里、
    经 `hf-mirror.com` **镜像**取得的 config 字段 ⇒ 必须带强制限定语
    「**经 `hf-mirror.com` 镜像取得，未与官方 repo 比对 sha256**」。
    ★ 本模块逐个词频核实过：`rope_theta` 与 `max_position_embeddings` 在
      `45-…/code/specs.py` 里**各出现 0 次** ⇒ 它们**只能**走镜像这条限定路径。
  · **第二类 [C]**：**论文正文里的逐字**。给 arXiv 号 + 抓取日 + **原始字节计数** +
    `sha256[:16]`，并按 `CONTRIBUTING` §2 注明**是否确认同行评审**。
    ★ 这些是**字符串常量**，程序对它们做**逐位相等**的断言 —— 引用判据（表 B）本身
      「不能被程序判定」，但**「我抄的这一句是不是原文」可以被程序钉住**（两者不是一回事）。
  · **第三类 [E]**：本域自己写出的**定义**（不是任何来源里的东西）⇒ 标 **[E]**，
    并写明它为什么抄不到。

★ **本模块只放本主题真的会拿去算的东西**（数、字符串、元组）。
  `45-…/code/specs.py:107-113` 记着一条教训：**「照抄但不使用」的部分无法被任何断言约束**，
  只会制造变异测试的漏网。⇒ 别的字段的**去向**写在文件尾注里，**不搬进来**。
  描述性文字归**文档**，不归本模块。

★★ **本模块的一条本主题特有纪律**（计划 §3 开头的第二个盲区）：
  **模块级常量表一律正数存放。** 理由是 `mutate.py` 的叶子枚举只取 `Constant` 节点，
  而 `-1` 在 AST 里是 `UnaryOp(USub, Constant(1))` ⇒ **写成负数的表整体不进变异分母**。
  本主题的方向/符号密集（`m−n` vs `n−m`、置换的两半），所以「方向」一律用**正数索引**表达
  （见 `REL_DIRECTION_INDEX`），负号在**函数体内**派生。

★ 每条常量带 `# 来源：` + `[A-E]` 级 + 抓取日；跨主题同值的另带 `# 同值位置：`。
  分级判据见 `CONTRIBUTING` §2（A 厂商规范 / B 产品官方文档 / C 论文 / D 社区 / E 本库推导）。
"""

# ===========================================================================
# 1. 两个真实模型的位置编码规格（**经镜像取得**）—— 本主题的真实锚点
# ===========================================================================
#
# ★ 为什么本主题需要它们：判据 ③ 说「`θ₁ = base⁰ = 1` **与 base 无关**」，判据 ④⑤ 说
#   「波长 / 分辨窗口」—— 而这些话若不落到**真实模型的 `base` 与上下文长度**上，
#   就只是玩具量级的演示。这两个模型的 `rope_theta` **恰好差 2 倍**（`5e5` vs `1e6`），
#   是本主题最省力的一个对照。
# ★★ 口径：`rope_theta` 是 **RoPE 的 `base`**（原论文写作 `10000`）。
#   本域**只记字段名与取值**，不推断「这个模型内部用的是哪个 attention 实现」。

LLAMA3_8B_ROPE_THETA = 500000.0
# 来源：meta-llama/Meta-Llama-3-8B 的 config.json，**经 `hf-mirror.com` 镜像取得**  [镜像]
#       2026-09-22 直取 `https://hf-mirror.com/NousResearch/Meta-Llama-3-8B/resolve/main/config.json`
#       （★ 最简形式 `curl -sS` 返回 307 的**跳转正文**，必须加 `-L`。）
# ★★ **未与官方 repo 比对 sha256。** `huggingface.co` 在本环境直连不可达。
#       镜像副本的字节哈希（**镜像的**，不是官方 repo 的）：
#         sha256[:16] = 2430cee764b6530f（654 B）
#    ★ 与 `normalization/code/specs.py:38-47`、`transformer/code/specs.py` 记的是**同一份副本**。
# 口径：字段名逐字 `rope_theta`（**不是** `rotary_emb_base` 之类）。

QWEN3_30B_A3B_ROPE_THETA = 1000000.0
# 来源：Qwen/Qwen3-30B-A3B 的 config.json，**经 `hf-mirror.com` 镜像取得**  [镜像]
#       2026-09-22 直取 `https://hf-mirror.com/Qwen/Qwen3-30B-A3B/resolve/main/config.json`
# ★★ **未与官方 repo 比对 sha256**（同上的全部限定）。
#       镜像副本的字节哈希：sha256[:16] = 2850ddb3bf7aecad（963 B）
#    ★ 与 `normalization/`、`moe/`、`attention-variants/` 的 `specs.py` 记的是**同一份副本**。
# ★★ **两个模型相差 2 倍**（`5e5` vs `1e6`）—— 本主题的判据 ③ 说明：**这个差别
#    `θ₁` 一点都感觉不到**（`θ₁ = base⁰ = 1`），它只把**低频端**整体拉低。

LLAMA3_8B_MAX_POSITION_EMBEDDINGS = 8192
# 来源：同上（Llama-3-8B 的 config.json），字段名逐字 `max_position_embeddings`  [镜像]
# ★★ 未与官方 repo 比对 sha256（同上）。
# ★ 本主题的用途：它是**训练窗口 `L`** —— `03` 的 E5（「波长 λ ≥ 上下文长度 L 时
#   只做插值」）里的那个 `L`，也是 `02` 里判「整窗是否混叠」的分母。

QWEN3_30B_A3B_MAX_POSITION_EMBEDDINGS = 40960
# 来源：同上（Qwen3-30B-A3B 的 config.json），字段名逐字 `max_position_embeddings`  [镜像]
# ★★ 未与官方 repo 比对 sha256（同上）。
# ★ **两个模型的比值是 5**（`40960/8192`）—— 它们在**两个维度上都不同**（base 2 倍、窗口 5 倍），
#   所以引用时必须写出是哪一个模型；本域**不把它们混起来**下结论。

QWEN3_30B_A3B_HEAD_DIM = 128
# 来源：Qwen/Qwen3-30B-A3B 的 config.json，字段名逐字 `head_dim`，**经 `hf-mirror.com` 镜像取得**
#       （★ 未与官方 repo 比对 sha256；镜像副本见 `QWEN3_30B_A3B_ROPE_THETA` 的哈希）  [镜像]
# ★ 这是**原生字段**（config 里真的有它），而且 **`128 ≠ 2048/32 = 64`**
#   ⇒ ★★ 这正是 `45-…/01-model-to-operator-inventory.md:270-276` 那个例子要说的坑
#     （计划 §C.4 的 **X16**）。**本域不重推那个例子**，只把它当一个**给定值**用。

LLAMA3_8B_HEAD_DIM_DERIVED = 128
# 等级：**[E] 推导** —— Llama-3-8B 的 `config.json` 里**没有** `head_dim` 这个字段
#   （经两个独立镜像确认：`NousResearch/Meta-Llama-3-8B`、`unsloth/llama-3-8b`）。
# ★ 本域的来源是**同域已有的事实**，不是重新做那个除法：
#   `80-model-architectures/transformer/code/specs.py` 记着 `head_dim = 128`（**标为推导**）。
#   ⇒ ★★ 本域**只引用它**，并**照抄它的等级标注**（计划 §2 的边界表：本域 `transformer`
#     不得用 `hidden_size ÷ num_attention_heads` **代替** `head_dim`；
#     这里的用法是「**记录**该字段缺失、值由同域已有条目给出」，不是拿除法去顶替字段）。
# ★ 两个模型的 rotary 维数在数值上**恰好相同**（都是 `128`），但**来源等级不同**：
#   Qwen3 是原生字段、Llama 是推导值 ⇒ ★ **引用时必须分开标注**（本模块把它们分成两条常量
#   就是为了让这件事不能含混过去）。

ROTARY_DIM_MODEL = 128
# 等级：**[E] 本域写出的用法** —— 「两个真实模型的 rotary 维数都是 `128`」这个**汇总判断**。
# ★ 它的两个输入是上面两条常量（一条原生、一条推导），本域把「它们数值相同」写成一个**显式
#   常量**，这样 `02` 的计数就**只能**用这个值、不能各自另取
#   （★ 否则「两个模型用了不同的 `d`」会**静默**地改变计数结论）。
# ★★ 为什么 `02` 需要一个「真实模型的 `d`」：`02` 的谱计数（`C.2`/`C.3`）是关于
#   **真实模型**的陈述，而 `d = 64`（探测脚本 probe_80f 的口径，D3–D6 的出处）**不是**
#   任何真实模型的 rotary 维数 ⇒ ★ **两个 `d` 必须分开**，各自标注口径。

# ===========================================================================
# 2. RoPE 论文（arXiv 2104.09864）的逐字 —— 表 B 的 X2 / X3 / X4 落点
# ===========================================================================
#
# ★ 抓取（2026-09-22，`curl -sS`，`WebFetch` 被策略拦截）：
#   `https://ar5iv.labs.arxiv.org/html/2104.09864`
#   **283120 B**，`sha256[:16] = cfec27d698ce4881`（★ 与计划 §C.2 的记法**逐位一致**）。
# 评审状态：abs 页 `comments` 逐字为 `fixed some typos`、**无 venue 线索**
#   ⇒ **未确认同行评审**（按页面实测填；不得凭印象补成 Neurocomputing）。
#
# ★★ **X1 的复核留痕（计划 §0.3 的纪律：「否定性结论必须配对照检验」）**：
#   原始字节计数 `raw.count(b'extrapol')` = **0**（大小写两种写法都是 0）；
#   而**对照词都在**：`rotary` = 6、`position` = 126、`generaliz` = 3、
#   `long-term decay` = 2、`Limitations` = 1。
#   ⇒ 「RoPE 论文没有说过 `extrapol*`」这个否定性结论**成立**。
#   ⚠ 「计数 0」本身**不构成证据**（跨标记的引文在原始字节上必然计不到）
#     ⇒ **必须**配这样一组对照，本模块把它记在这里。

ROPE_FREQ_SET_EQ15 = "10000^{-2(i-1)/d}"
# 来源：RoPE（arXiv 2104.09864）§3.2.2，紧跟在**式 (15) 的旋转矩阵**之后的那句里的参数集合  [C]
# ★ 原始字节计数（**在 `alttext="…"` 与 `<annotation>` 两处各一份**）：
#   `10000^{-2(i-1)/d}` = **2**
# ★★ **口径订正（X2，本域实测的措辞）**：计划 §C.3 写「§3.2.2 **式 (15)** 是
#   `10000^{-2(i-1)/d}`」——**式 (15) 本身是那个块对角旋转矩阵**，而这个频率集合是
#   **紧跟其后的句子里的**（原文逐字：`(15) is the rotary matrix with pre-defined
#   parameters Θ = {…}`）。**两者是相邻但不是同一个式子**，本域按原文分开写。
# ★ 该集合的下标从 `i=1` 起：`θ₁ = base⁰ = 1`（**这就是判据 ③ 的出处**）。

ROPE_FREQ_SET_LONGTERM = "10000^{-2i/d}"
# 来源：RoPE（arXiv 2104.09864）§3.3「Properties of RoPE / Long-term decay」逐字：
#       `Following Vaswani et al. 2017, we set θi=10000^{-2i/d}.`  [C]
#       ★ §3.4.3 又出现一次（`by setting θi=10000^{-2i/d}`）。
# ★ 原始字节计数：`10000^{-2i/d}` = **4**（两处正文 × `alttext`/`annotation` 各一份）。
# ★★ **X2 的实质**：同一篇论文里**有两个版本**的频率公式（差一个下标平移）。
#   **两式给出的频率集合完全相同** ⇒ 判据 ③ 的结论不受影响，
#   但**引用时必须说明用的是哪一个**（否则读者照着抄会得到不一致的下标范围）。
# ★ 本域自己的实现**固定用 `10000^{-2(i-1)/d}`、`i∈[1,d/2]`**（`freqs()` 的默认），
#   并在正文里写明这一点（计划 §4 的 R7）。

ROPE_EQ15_PROSE = "is the rotary matrix with pre-defined parameters"
# 来源：同上，式 (15) 之后那句的逐字片段  [C]
# ★ 原始字节计数：**1**。★ 这条的作用是给 `ROPE_FREQ_SET_EQ15` **定位**：
#   它把「频率集合」与「式 (15)」的关系**逐字**钉在一个句子里，免得两条常量被混成一条。

ROPE_EQ11_REL = "g({\\boldsymbol{x}}_{m},{\\boldsymbol{x}}_{n},m-n)."
# 来源：RoPE（arXiv 2104.09864）§3.1 **式 (11)** 的 LaTeX 原文（取自 `alttext`）  [C]
# ★ 原始字节计数：**2**（`alttext` + `annotation`）。
# ★★ **X3 的口径**：论文内部的**相对位置符号不统一** —— 式 (11) 是 `m−n`，
#   式 (16) 推出的是 `R^d_{Θ,n−m}`，§3.4.1 的式 (21) 又写 `n−m`。
#   ⇒ **引用哪一处必须写明**（否则与另一处符号相反）。

ROPE_EQ21_REL = "g({\\boldsymbol{x}}_{m},{\\boldsymbol{x}}_{n},n-m),"
# 来源：RoPE（arXiv 2104.09864）§3.4.1 **式 (21)** 的 LaTeX 原文（取自 `alttext`）  [C]
# ★ 原始字节计数：**2**。★ 与 `ROPE_EQ11_REL` 的**唯一差别就是最后两个字母**
#   （`m-n` vs `n-m`）⇒ 这两条常量放在一起，X3 就是**可逐位判定**的。

ROPE_EQ16_ALT = "={\\boldsymbol{x}}^{\\intercal}{\\boldsymbol{W}}_{q}R^{d}_{\\Theta,n-m}"
# 来源：RoPE（arXiv 2104.09864）§3.2.2 **式 (16)** 的 LaTeX 原文（取自 `alttext`）  [C]
# ★ 原始字节计数：**2**。
# ★★ **这一条同时钉住两件事**：
#   ① **X4（笔误）**：式 (16) 末项的首因子原文写作 `x^⊤`（**缺下标 `m`**）。
#      ★ 本域**另存**一条「正确形态」的常量（`ROPE_EQ16_TYPO_ABSENT`，计数 **0**）
#        —— 用「正确形态计数 0」+「原文形态计数 2」这对**一正一反**钉住笔误。
#      照抄原文会得到**维度不符**的式子（左边是 `q_m^⊤k_n` 的标量，右边却出现未下标化的 `x`）。
#   ② **X3（方向）**：式 (16) 里是 `R^d_{Θ,n−m}`（**`n−m`**），与式 (11) 的 `m−n` 相反。

ROPE_EQ16_TYPO_ABSENT = "{\\boldsymbol{x}}_{m}^{\\intercal}{\\boldsymbol{W}}_{q}R^{d}_{\\Theta,n-m}"
# 来源：**不是原文** —— 它是「原文若**没有**那个笔误会写成什么」的形态。  [E]
# ★ 原始字节计数：**0**（★ 这就是它的全部用处：一个**否定性**的对照常量）。
# ★★ 一条**计数 0 的常量**为什么可以存在：它**不是**用来支撑「原文有什么」的，
#   而是**与 `ROPE_EQ16_ALT` 配对**、由程序断言「前者计数 0、后者计数 2」
#   这一对**一正一反**的读数 ⇒ 它自己不需要对照检验。
#   （计划 §0.3：**单独的「计数 0」不构成证据**，必须配对照。这里是**配对**用法。）

ROPE_LONGTERM_DECAY_CLAIM = "One can prove that this setting provides a long-term decay property"
# 来源：RoPE（arXiv 2104.09864）§3.3 逐字  [C]
# ★ 原始字节计数：**1**；对照：`long-term decay` 全文 = **2**（另一处在 §3.4.3 的
#   「Long-term decay of RoPE」小节标题里）⇒ 「1」可信。
# ★ 用途：`02` 讲「θ 的谱」时，这条是**论文自己给出的唯一一条关于谱性质的主张**
#   —— 注意它的措辞是 `One can prove`（**可证明**，不是实验结果）。

# ===========================================================================
# 3. 外推方法的一手来源：YaRN（arXiv 2309.00071）—— 表 B 的 E1–E9 落点
# ===========================================================================
#
# ★ 抓取（2026-09-22，`curl -sS`）：`https://ar5iv.labs.arxiv.org/html/2309.00071`
#   **232997 B**，`sha256[:16] = 70b5328c47374b18`（★ 与计划 §C.2 逐位一致）。
# 评审状态：abs 页**无 venue 线索** ⇒ **未确认同行评审**；★★ 不得凭印象补成 ICLR。
#
# ★★ **等级纪律（本域新增的硬约束）**：YaRN 论文逐字把 NTK-aware / Dynamic NTK /
#   NTK-by-parts 称为 **`previous unpublished works`**（其原始来源是 Reddit 帖与 GitHub 仓库）。
#   按 `CONTRIBUTING:77`：**D 级（社区文档）的数字不得单独支撑论断。**
#   ⇒ **引用 NTK 系列时只能引本模块这一批 YaRN 的转述（C 级）**，
#     **不得把 Reddit / GitHub 帖子当作数值来源**。

YARN_NTK_AWARE_E1 = "proposed the \"NTK-aware\" interpolation by taking the loss of high frequency into account"
# 来源：YaRN（arXiv 2309.00071）§1 逐字  [C]
# ★ 原始字节计数：**1**。
# ★★ 这条建立了本域 `03` 的**主线**：PI 丢的是**高频**，NTK-aware 是来补这一点的。
#   ⚠ 计划 §C.5 的 E1 曾把这条写成 `int[…` 的截断形态（记号插在**词内部**），
#     原文是完整的 `into account`；本模块按**完整句**存。

YARN_SPREAD_E3A = "we spread out the interpolation pressure across multiple dimensions by scaling high frequencies"
# 来源：YaRN（arXiv 2309.00071）§3.1 逐字  [C]
# ★ 原始字节计数：**1**。
# ★★ **这条句子没有结束**：原文继续 `less and low frequencies more`（见下一条）。
#   计划 §C.5 的 E3 在 `high frequencies` 处就闭合了引文并当作句末 —— 补齐后
#   **论点反而更完整**：**高频少缩、低频多缩**（这正是 `03` 的闭式要复现的东西）。

YARN_SPREAD_E3B = "less and low frequencies more"
# 来源：同上一句的**后半**（同一句，同一次抓取）  [C]
# ★ 原始字节计数：**1**。★ 单独存它的理由：**截断不是原文**（计划 §0.3 的纪律），
#   把两半分开存，引用时就不会把「截断后的句子」当成「完整的句子」。

YARN_OUT_OF_BOUND_E4 = "some dimensions are slightly extrapolated to \"out-of-bound\" values"
# 来源：YaRN（arXiv 2309.00071）§3.1 逐字  [C]
# ★ 原始字节计数：**1**。
# ★★ 它是 **NTK-aware 的代价**，而且**恰好是本域 `03` 的闭式能算出来的那一件事**：
#   由于缩放指数 `α_i = (i−1)/(d/2−1)` 在 `i = 1` 处**精确为 0**，
#   **最高频的那一维根本没被插值**（它的新 `θ` 仍是 `1`）⇒ 它就是那个「被外推越界」的维。

YARN_WAVELENGTH_E5A = "is equal to or bigger than the context size"
# 来源：YaRN（arXiv 2309.00071）§3.2「NTK-by-parts」三条 bullet 的第 2 条  [C]
# ★ 原始字节计数：**1**。

YARN_WAVELENGTH_E5B = "we want to only interpolate and avoid any extrapolation"
# 来源：同一条 bullet 的**后半句**（同一次抓取）  [C]
# ★ 原始字节计数：**1**。
# ★★ **这是 `02` 与 `03` 之间唯一的桥**（计划 §3 Chunk 5 表 B 的最后一条）：
#   原文的判据是 **`λ ≥ L` ⟺ 只插值**；而 `02` 的「分辨窗口」`θ_i·S` 与 `λ` 的关系是
#   **精确的** `θ_i·L = 2π·L/λ_i`（`03` 的 `B.3` 把这条做成断言）。
#   ⇒ 「波长 ≥ 上下文长度」与「在整段上下文里连一圈都没转完」**是同一件事**。

YARN_FITTING_E7 = "The equation above is found by fitting"
# 来源：YaRN（arXiv 2309.00071）§3.4 逐字  [C]
# ★ 原始字节计数：**1**；对照：`found by fitting` = **1**（全篇只此一处）。
# ★★ **措辞纪律**：`√(1/t) = 0.1·ln(s) + 1` 是**拟合值**（原文自承 `found by fitting …
#   at the lowest perplexity against the scale extension by various factors s`）
#   ⇒ **不得写成推导**。本域只把它当**一个数**用，并照抄这个限定。

YARN_PERPLEXITY_E8 = "perplexity may not be a great indicator"
# 来源：YaRN（arXiv 2309.00071）附录 B.2 逐字（完整形态：
#       `we hypothesize that perplexity may not be a great indicator of whether an LLM is
#        able to attend to all tokens`）  [C]
# ★ 原始字节计数：**1**。★ 注意前缀是 **`we hypothesize`** ⇒ 连作者自己都用假设的语气。
# ★ 用途：本域**引用外推效果的任何数字时**的纪律 —— 那个指标不是完整指标。

YARN_UNPUBLISHED_E9 = "previous unpublished works"
# 来源：YaRN（arXiv 2309.00071）§1 逐字（完整形态：
#       `in addition to making a complete account of the previous unpublished works on the
#        "NTK-aware", the "Dynamic NTK" and the "NTK-by-part" interpolations`）  [C]
# ★ 原始字节计数：**1**。★★ 这条是 §C.5 那条**等级纪律的唯一出处**。

YARN_BASE_CHANGE = "{b^{\\prime}}=b\\cdot s^{\\frac{|D|}{|D|-2}}."
# 来源：YaRN（arXiv 2309.00071）§3.1 **式 (16)** 的 LaTeX 原文（取自 `alttext`）  [C]
# ★ 原始字节计数：**2**。★ 它是 **NTK-aware 的全部内容**：把 base 换成 `b'`。
#   ★★ `03` 的 `B.1` 把它的**后果**做成闭式（`θ_i' = θ_i·s^{−α_i}`）并与
#     「直接换 base」的**独立计算**逐位对比 —— 两条路。

YARN_NTK_BY_PARTS_RAMP = "=\\Big(1-\\gamma\\big(r(d)\\big)\\Big)\\frac{\\theta_{d}}{s}+\\gamma\\big(r(d)\\big)\\theta_{d}."
# 来源：YaRN（arXiv 2309.00071）§3.2 的 `h(θ_d)` 定义式（取自 `alttext`）  [C]
# ★ 原始字节计数：**2**。
# ★★ **这一条是本域 `03` 的统一框架**：`h(θ) = (1−γ)·θ/s + γ·θ` 是
#   「**全插值**（`θ/s`）」与「**不插值**（`θ`）」的**凸组合**。
#   ⇒ PI 是 `γ ≡ 0`、完全不做外推是 `γ ≡ 1`、NTK-by-parts 是 `γ = ramp(r)`；
#     而 NTK-aware 也能写成同一族（它的权重是 `α_i`，见 `03` 的 `B.1`）。
#   **四种方法 = 同一个一参数族的四种「怎么挑权重」** —— 这就是本域要给的解释。

YARN_TEMPERATURE_FIT = "\\sqrt{\\frac{1}{t}}=0.1\\ln({s})+1"
# 来源：YaRN（arXiv 2309.00071）§3.4 **式 (22)** 的 LaTeX 原文（取自 `alttext`）  [C]
# ★ 原始字节计数：**4**（该式在 `alttext`/`annotation` 与正文两次引用里各出现）。
# ★★ **它是拟合值**（配 `YARN_FITTING_E7` 一起引用）。

YARN_NTK_BY_PARTS_QUOTED_PLURAL = 10
# 来源：YaRN（arXiv 2309.00071）原始字节计数：`NTK-by-parts"`（**带闭引号**）= **10**  [C]
YARN_NTK_BY_PARTS_QUOTED_SINGULAR = 1
# 来源：同上：`NTK-by-part"`（**带闭引号**）= **1**  [C]
# ★★ **这两条一起构成一条方法论注记**（本域实测）：
#   若**不带闭引号**地数，得到的是 `NTK-by-part` = **13**、`NTK-by-parts` = **12** ——
#   **13 > 12**，于是结论会**反过来**（「论文写单数」）！
#   原因：`NTK-by-part` 是 `NTK-by-parts` 的**子串**（前者被后者包含）
#   ⇒ ★ **计数必须带上能闭合的边界**，否则子串会污染读数。
#   （这正是「审查者把主流的判成错的」那条的机制；计划 §C.5 的术语口径注记已按
#    **带闭引号**的计数裁定为**复数**，本域沿用 `NTK-by-parts`。）

YARN_NTK_BY_PARTS_ALPHA = 1.0
YARN_NTK_BY_PARTS_BETA = 32.0
# 来源：YaRN（arXiv 2309.00071）§3.2 逐字：`and β are α = 1 and β = 32`  [C]
# ★ 原始字节计数：`\alpha=1` = **2**、`\beta=32` = **2**（`alttext` + `annotation` 各一份）。
# ★★ **口径**：论文里这两个参数的名字是 **`α`（下阈值）与 `β`（上阈值）**
#   （不是 `β`/`γ`）—— `γ` 是那个 **ramp 函数**的名字。
#   本域按论文的命名写，免得与 `γ(r)` 混成同一个符号。

# ===========================================================================
# 4. 位置插值 PI（arXiv 2306.15595）—— `03` 的第一种方法
# ===========================================================================
#
# ★ 抓取（2026-09-22，`curl -sS`）：`https://ar5iv.labs.arxiv.org/html/2306.15595`
#   **217296 B**，`sha256[:16] = fd15ea37b56bee34`。
#   ⚠ 本域在写这一行时**先写错了一次**：第一版写的是 `216083`，那是**解码后的字符数**
#     （`len(str)`）而不是**文件字节数** —— 本库已有一条教训「量字节别用 `len(str)`」，
#     这次是它的现场。
#   ⚠ 计划 §C.2 记的是 **abs 页**的哈希 `deb27db06526bf96`，本域取的是 **ar5iv HTML 页** ——
#     **两者不是同一个页面**，哈希不同是预期的（本域记的是自己取到的那一份）。
# 评审状态：abs 页 `comments` 逐字为 `Fix template issues` ⇒ **无 venue 线索**，未确认。

PI_RESCALE_ALT = "\\mathbf{f}\u2019(\\mathbf{x},m)=\\mathbf{f}\\left(\\mathbf{x},\\frac{mL}{L^{\\prime}}\\right)."
# 来源：PI（arXiv 2306.15595）§2 **式 (4)** 的 LaTeX 原文（取自 `alttext`）  [C]
# ★ 原始字节计数：**2**。
# ★ 口径：★ 注意原文里的撇号是 **U+2019（右单引号 `’`）**，不是 ASCII 的 `'`
#   —— 照 ASCII 抄会**计数 0**（本域实测）。本模块按原文的码位存。
# ★ 它就是 PI 的全部内容：**把位置下标 `m` 换成 `m·L/L'`**（`L'` 是扩展后的窗口）。

PI_WEAK_EXTRAPOLATION = "weak extrapolation properties"
# 来源：PI（arXiv 2306.15595）§1 逐字（`many existing pre-trained LLMs … use positional
#       encodings that have weak extrapolation properties (e.g., RoPE)`）  [C]
# ★ 原始字节计数：**1**。
# ★★ 用途：**X1 的落点**。「RoPE 外推不行」这句话的出处**不能**归给 RoPE 论文
#   （那里 `extrapol` 计数 **0**）；它属于**后续工作** —— 本域选了 PI 与 ALiBi 两处
#   （两条都是**独立的一手来源**，不是转述）。

# ===========================================================================
# 5. ALiBi（arXiv 2108.12409）—— 表 B 的 X17 落点
# ===========================================================================
#
# ★ 抓取（2026-09-22，`curl -sS`）：`https://ar5iv.labs.arxiv.org/html/2108.12409`
#   **333702 B**，`sha256[:16] = 971a61398ba0f189`（★ 与计划 §C.2 逐位一致）。
# 评审状态：abs 页**无 venue 线索** ⇒ **未确认同行评审**（不得凭印象补成 ICLR 2022）。

ALIBI_BIAS_ALT = "\\text{softmax}(\\mathbf{q}_{i}\\mathbf{K}^{\\top}+m\\cdot[-(i-1),...,-2,-1,0]),"
# 来源：ALiBi（arXiv 2108.12409）§3 **式 (2)** 的 LaTeX 原文（取自 `alttext`）  [C]
# ★ 原始字节计数：**2**。
# ★★ **X17 的前半**：偏置是加在 `q_i K^⊤` **之后**、**softmax 之前**；
#   `m` 是**训练前固定的、每个 head 一个的 slope**（`head-specific slope fixed before training`）。
#   ★ 注意这个偏置的取值集合是 `[−(i−1), …, −2, −1, 0]` —— **最小的那个是 `−(i−1)`**，
#     本模块**不把负数写进常量表**（见模块头的纪律），程序里由**正数距离**派生。

ALIBI_FOOTNOTE = "The ALiBi bias is not multiplied by the"
# 来源：ALiBi（arXiv 2108.12409）§3 **脚注 10** 逐字（完整形态：
#       `The ALiBi bias is not multiplied by the √d_k scaling factor from Equation 1 of
#        Vaswani et al. 2017.`）  [C]
# ★ 原始字节计数：**1**；★ 完整句里 `√d_k` 是一个数学标记 ⇒ **跨标记**（计划 §0.3），
#   所以本模块只存**计得到的那一段**，不存整句。
# ★★ **X17 的后半，也是本域 `03` 的一条断言（`C.2`）**：那条偏置**不**吃 `1/√d_k`。
#   可判定的后果：**偏置对注意力的影响不随 `d_k` 变**（若它也被 `1/√d_k` 除，
#   大 `d_k` 的 head 的「近因偏好」会被压掉 —— `03` 把这个对照跑出来）。

ALIBI_SLOPE_RULE = "our set of slopes is the geometric sequence that starts at"
# 来源：ALiBi（arXiv 2108.12409）§3 逐字（完整形态：
#       `In general, for n heads, our set of slopes is the geometric sequence that starts
#        at 2^{−8/n} and uses that same value as its ratio.`）  [C]
# ★ 原始字节计数：**1**。
# ★★ 这是一条**可手算的闭式**：`slope_h = (2^{−8/n})^h`，`h = 1..n`。
#   n=8  ⇒ `1/2, 1/4, …, 1/256`（原文另有一句逐字给出这串：`1/2^1, 1/2^2, …, 1/2^8`）；
#   n=16 ⇒ 起于 `1/√2`、比值 `1/√2`。
#   `03` 的 `C.3` 用这两点各断言一次。

ALIBI_ROPE_VERDICT = \
    "Though the rotary position method improves over the sinusoidal one, it still does not achieve satisfying results."
# 来源：ALiBi（arXiv 2108.12409）**§2** 逐字（该节标题逐字为
#       `Current Approaches Do Not Extrapolate Efficiently`）  [C]
# ★ 原始字节计数：**1**。★ 对照：`extrapolat` 在 ALiBi 全文 = **68**（与 RoPE 的 **0** 成对照）
#   ⇒ 「外推问题」的正文在**这一篇**，不在 RoPE 论文。
# ★★ **X1 的落点**：这是**一手来源**里对 RoPE 外推能力的**评价**，且**主语是 ALiBi**。

# ===========================================================================
# 6. [E] 本域自己写出的两条定义（**不是任何来源里的东西**）
# ===========================================================================

ROPE_FREQ_INDEX_ORIGIN = 1
# 等级：**[E]** —— 本域自己写出的**约定**：频率下标 `i` 从 **1** 开始（`i ∈ [1, d/2]`）。
# ★ 为什么它必须是 [E]：论文里**两个版本的下标起点不同**
#   （`10000^{-2(i-1)/d}` 的 `i∈[1,d/2]` vs `10000^{-2i/d}` 的 `i∈[0,d/2-1]`），
#   论文**没有**规定哪一个才是「正确」的 —— 它们是**同一个集合**的两种编号。
#   ⇒ 本域**固定一个**（`i` 从 1 起）并**写明**，这正是计划 §4 的 R7 要求的做法。
# ★ 它是**正数**存放的（见模块头的纪律）。`θ₁ = base⁰ = 1` 依赖这个约定。

REL_DIRECTION_INDEX = 1
# 等级：**[E]** —— 本域自己写出的**方向约定**：`1` = 相对位置取 `m − n`，`2` = 取 `n − m`。
# ★★ 为什么用**正数索引**而不是直接存 `±1`：`mutate.py` 的叶子枚举只取 `Constant`，
#   而 `-1` 是 `UnaryOp(USub, Constant(1))` ⇒ **写成负数的表整体不进变异分母**
#   （计划 §3 开头的第二个盲区，Chunk 4 实测）。程序里由 `1 - 2*idx` **派生**符号。
# ★ 为什么需要这条：论文内部符号**不统一**（X3：式 (11) 是 `m−n`、式 (16) 是 `n−m`、
#   式 (21) 是 `n−m`）⇒ 本域**必须固定一个方向并说明**（计划 §4 的 R7）。
#   本域的默认取 `m − n`（与式 (11) 一致），并在 `01` 的 `B.4` 里把**两个方向都实现一遍**。

# ===========================================================================
# 7. UNSOURCED —— 待办清单，不是免责声明
# ===========================================================================
#
# ★ CONTRIBUTING §2：「不得**既保留数字又记 UNSOURCED**」—— 下面每一项都**没有**出现在
#   本模块的正文常量里。

UNSOURCED = (
    # 每一项是一个**稳定标识符**（不是散文）。
    "ntk_aware_primary_source",      # ★★ NTK-aware / Dynamic NTK / NTK-by-parts 的**一手来源
                                     # 不在本环境**：YaRN 论文自己把它们称为
                                     # `previous unpublished works`（原始来源是 Reddit 帖与
                                     # GitHub 仓库，按 `CONTRIBUTING:77` 是 **D 级**）
                                     # ⇒ 本域**只引 YaRN 对它们的转述（C 级）**，
                                     # 不把 Reddit/GitHub 当数值来源。
    "rope_extrapolation_weakness_attribution",
                                     # ★ 「RoPE 外推不行」这句话**不能归给 RoPE 论文**
                                     # （`extrapol` 计数 0）。本域把它归给 **ALiBi §2** 与
                                     # **PI §1** 两处后续工作 ⇒ **不需要** UNSOURCED 兜底，
                                     # 但**保留这个标识符**记录「归属必须写明」这件事，
                                     # 免得下一个实现者又把它挂回 RoPE 论文。
    "long_context_quality_metric",   # ★ 「长上下文能力用什么指标衡量」——**本域不给这个指标**：
                                     # YaRN 自己逐字说 `perplexity may not be a great indicator`，
                                     # 而本域**无 GPU、不训练模型**、也**不跑评测**
                                     # ⇒ 本域只写**位置编码结构上的账**，不写「哪个方法效果更好」。
    "extrapolation_quality_ranking", # ★ 「PI / NTK-aware / NTK-by-parts / YaRN 谁更好」——
                                     # `03` 的 `UNSOURCED` 与这里同义：那是**训练与评测**的账
                                     # （YaRN 的表 1/2 是它自己的实验）。本域只做**结构对比**：
                                     # 「每种方法的每维缩放因子是多少」是**可手算**的，
                                     # 「谁的效果好」不是。
    "rope_in_real_attention_kernels",# ★ 「哪些真实 kernel 用了哪一种 RoPE 排布」——
                                     # 那要读 kernel 源码（`40-kernels/attention` 的正文在彼）。
                                     # 本域只给**两种排布的定义与它们的关系**。
)

# ===========================================================================
# 尾注：本主题**刻意不搬**的字段（它们的正文在别处）
# ===========================================================================
#
#   · `hidden_size` / `head_dim` / `num_attention_heads` / `num_hidden_layers` /
#     `intermediate_size` / `vocab_size` / `hidden_act` / `tie_word_embeddings`
#                                   → `transformer/code/specs.py`
#   · `rms_norm_eps`              → `normalization/code/specs.py`
#   · `num_experts*` / `moe_intermediate_size` / `router_aux_loss_coef` → `moe/code/specs.py`
#   · `sliding_window` / `use_sliding_window` → `attention-variants/code/specs.py`
#       ★ **本主题一个都不断言它们** —— 按 `45-…/code/specs.py:107-113` 的教训，
#         「照抄但不使用」的常量无法被任何断言约束，只会制造变异测试的漏网。
#       ★ 本模块只搬了 **4 个字段**（两个模型的 `rope_theta` 与 `max_position_embeddings`）：
#         前两个是判据 ③ 的真实锚点，后两个是 `02`/`03` 的窗口 `L`，
#         且四个都**被程序逐位钉住**（`01` 的 `D.3`、`02` 的 `D.3`、`03` 的 `D.3`）。
#
#   · ★★ **RoPE 的「成对通道旋转」这句结构陈述** → `130-quantization/kv-cache-quantization/
#     02-kivi-asymmetry.md:184`（**逐字**含 `jointly rotates pairs of channels by different
#     angles depending on the position`，那里讲的是 pre-RoPE 量化）。
#     ⇒ 本主题的 `01` 给出的是**定义**（哪两个通道配对、转多少），
#       `130` 那边的**结论**（先量化再套 RoPE）**不在本域**（计划 §2 的边界表）。
#
#   · ★★ **变体的 kernel 后果**（滑窗改循环范围、ALiBi/softcap 改 `exp` 的输入）
#     → `40-kernels/attention/04-warp-specialization-and-variants.md:126-156`。
#     本主题只补**定义与出处**（那正是那处「原为悬空让渡」缺的东西）。
#
#   · **位置编码自身的 FLOPs / 访存账**：本主题**不算** —— RoPE 是逐元素的旋转，
#     它的计算量与 attention 主体的比值不属于本域（正文在 `40-kernels/attention`）。
#
#   ★ R3 的裁定：**每主题一份 `specs.py`，每份只放本主题用得到的常量**，
#     跨主题同值的常量各写一份、靠 `# 同值位置：` 互指。
#     `CONTRIBUTING:36`「`code/` 必须自包含」是**全库规则**。

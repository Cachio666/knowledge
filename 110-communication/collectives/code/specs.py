"""110-communication/collectives 的外部事实常量源。

★ `CONTRIBUTING` §2 约定的**共享事实源**：本主题的编号程序从它读常量，
  推导过程可复核，**常量本身不可自证**。
★ **只放本主题真的会断言的常量**（无消费者的常量是 `make mutate` 的漏网候选）。
★ **来源分级 A–E**（A 厂商规范 / B 产品官方文档 / C 论文 / D 社区 / E 本库推导）；
  每条带 `# 来源：` + `[A-E]` 级 + **抓取日** + `# 同值位置：` + `# 口径：` + `# 消费者：`。
★★ **本域的头号错误模式是「把 B 级当 A 级」**（`nccl` 主题整个靠产品文档）。
★★ **所有数值常量取到 `< 2^53`**；**幂次写成十进制字面量**（`mutate.py` 的 `_walk()` 不下钻 `BinOp`）。

★★★ **【本主题唯一的载体】与它的读法纪律**（★ 这是本文件最容易出错的地方）：
  载体 = **Thakur, Rabenseifner, Gropp, "Optimization of Collective Communication Operations
  in MPICH", IJHPCA 19(1):49–66, 2005** —— **[C] 级，同行评审：是**
  （Crossref `10.1177/1094342005051521`；一手 PDF `mcs.anl.gov/~thakur/papers/ijhpca-coll.pdf`，
  `1100025` 字节）。抓取日 **2026-10-08**。
  ★★★ **`pdftotext -layout` 把数学符号提取成两种形态** —— 这一条**必须**写在每个公式常量的
    `# 来源：` 里，否则**复核者会看错**：
      · `⌈lg p⌉` ⇒ 提取成 **`dlg pe`**
      · `⌊lg p⌋` ⇒ 提取成 **`blg pc`**
      · **裸 `lg p`** ⇒ 表示**没有取整符号**
    ★★ 实测教训（本库）：把 `⌈lg p⌉` 误当成 `lg p`（或反之）**不会有任何报错**，
      只会让公式**悄悄错掉** —— 所以 `grep "lg p"` **会同时命中三种形态**，
      **必须看那个符号**。
  ★★ 本文件存放的是**排版后的数学形态**（`⌈⌉` / 分数线），`# 来源：` 里另记**提取形态**。

★★★ **一条【同名不同指】（★ 极易错）**：该论文里 **`Trec dbl` 这个名字被用在两处**：
    · **allgather** 的递归倍增：`Trec dbl = lg p α + (p−1)/p · nβ` —— 每设备搬 `(p−1)/p · n`；
    · **allreduce** 的递归倍增：`Trec−dbl = lg p α + n lg p β + n lg p γ` —— 每设备搬 `n lg p`。
  ★★ 两个式子的**带宽项相差一个 `lg p` 因子**（`(p−1)/p·n` 对 `n lg p`）——
    它们**不是同一个算法的两种写法**，而是**两个不同的算法**（一个 allgather、一个 allreduce）。
    ⇒ 本文件用**两个不同的常量名**存放它们（`MPICH_AG_RECDUB_TIME_QUOTE` /
      `MPICH_ALLREDUCE_RECDUB_TIME_QUOTE`），并在注释里互指。

★★★ **另一条【同名不同指】**：论文的 **Bruck 算法是 allgather 与 all-to-all（index）的算法**，
    **不是 allreduce 的算法** —— allreduce 里它只作为**组合件**出现
    （见 `MPICH_BRUCK_SCOPE_NOTE`）。

★★★ **一条【禁写清单】纪律（`F5`）**：论文里的**分档阈值**是
    **2005 年 MPICH 在 Myrinet / IBM SP 上的【实现选择】** ⇒ **不得**写成
    「集合通信的普适分档」—— 引用必须带 `MPICH_CUTOFF_QUALIFIER` 那句限定语。

★ **本主题的禁区**（本主题**一行都不重写**，只回链）：
  · ★★★ **`V(N) = 2(N−1)/N × S` 的【推导】与 `S` 的饱和**
    → `10-foundations/performance-models/02-amdahl-and-scaling.md:147-155,209`
    （★ 本主题只用它当 **ring allreduce 的闭式**去与消息序列模拟对拍，**不重推**）；
  · **bisection 下界 `t ≥ 2S/B`** · **all-to-all 的跨切口量 `N·V/4`** · **分层 all-reduce 三段式**
    → `20-gpu-architecture/interconnect/03-collectives-on-topology.md`；
  · **`algbw` vs `busbw` 与修正系数表** → `interconnect/04-measuring-bandwidth.md`；
  · **带宽口径**（线速率 / 载荷 / 营销取整）→ `interconnect/01-bandwidth-conventions.md`。

★ **本主题的判据纪律**：
  本主题的 A 类钉在【算法族的步数与每卡搬运量】上，参照是**真消息序列模拟**
  （逐步穷举「哪一步、谁发给谁、哪个块、多少元素」）——
  ★★ 与闭式是**两条独立路径**（不许把闭式用循环写一遍，那是 `P12`）。

★ **本机的限定语**：本机**无 GPU、无 NCCL、无 IB / RoCE 网卡、无 NVLink**
  ⇒ ★★★ **本主题最强的一条限定语：**「**不是 NCCL，也不是真机**」。
"""


# ===========================================================================
# 1. Thakur / Rabenseifner / Gropp 2005（IJHPCA 19(1):49–66）—— [C] 论文
#    同行评审：是（Crossref 10.1177/1094342005051521）
#    ★★ 载体 = 一手 PDF（见模块抬头）；抓取日 2026-10-08。
#    ★★★ 本节的**每条公式常量**都必须写明 `pdftotext -layout` 的提取形态
#      （`dlg pe` = ⌈lg p⌉ · `blg pc` = ⌊lg p⌋ · 裸 `lg p` = 无取整）。
# ===========================================================================

MPICH_CITATION = "Thakur, Rabenseifner, Gropp, Optimization of Collective Communication Operations in MPICH, IJHPCA 19(1):49-66, 2005"
# 来源：[C] 论文，**同行评审：是**（Crossref `10.1177/1094342005051521`）；
#       一手 PDF `mcs.anl.gov/~thakur/papers/ijhpca-coll.pdf`（`1100025` 字节），抓取日 2026-10-08。
#       ★★★ 本常量是**书目字段**（标题 + 载体 + 卷期），**不是一句引文**
#         ⇒ 它只用来支撑「本主题的算法族事实来自这篇论文」这件事本身。
#       ★★ 标题里的破折号/连字符按 Crossref 的写法存（ASCII）；卷期页 `19(1):49-66`。
# 口径：★★ 本主题**唯一**的一手来源 —— 本文件其余 `MPICH_*` 常量**全部**出自它。
# 消费者：`01_algorithm_family.py` 的 `D.3`（与 `D.5` 的引文成对打印）。
# 同值位置：无。

MPICH_BCAST_TREE_STEPS_QUOTE = "This communication takes a total of ⌈lg p⌉ steps."
# 来源：[C] 论文，**广播 · 二项树（binomial tree）**。载体 = 一手 PDF，抓取日 2026-10-08。
#       ★★ `pdftotext -layout` 提取形态 = `... a total of dlg pe steps.`（**有 ⌈⌉**）。
# 口径：★★ **步数**的逐字依据 —— 二项树广播的步数是 `⌈lg p⌉`（**对数级**）。
#       ★ 它与 ring allgather 的 `p − 1`（线性级）**是同一个原语的两种算法**
#       （见 `MPICH_AG_RING_STEPS_QUOTE`）⇒ 这就是「算法族」的题眼之一。
# 消费者：`01_algorithm_family.py` 的 `D.5`。
# 同值位置：`MPICH_BCAST_TREE_TIME_QUOTE`（同一条公式的「时间」写法）。

MPICH_BCAST_TREE_TIME_QUOTE = "Ttree = ⌈lg p⌉(α + nβ)."
# 来源：[C] 论文，**广播 · 二项树**。载体 = 一手 PDF，抓取日 2026-10-08。
#       ★★ `pdftotext -layout` 提取形态 = `Ttree = dlg pe(α + nβ).`（**有 ⌈⌉**）。
# 口径：★★ 逐字依据。★ 注意它的**带宽项是 `nβ`（不是 `n lg p β`）** ——
#       二项树的**每设备**只发**一次**（整条消息），总消息数 `p − 1`
#       ⇒ 与 ring allgather 的 `(p−1)/p · nβ` **量级不同**（一个是 `n`，一个是 `≈n`）。
#       ★★★ **别与 allreduce 的递归倍增混**：那条是 `n lg p β`（见
#       `MPICH_ALLREDUCE_RECDUB_TIME_QUOTE`）—— **`lg p` 的有无是两条不同的算法**。
# 消费者：`01_algorithm_family.py` 的 `D.5`。
# 同值位置：`MPICH_BCAST_TREE_STEPS_QUOTE`。

MPICH_BCAST_VDG_QUOTE = "the message to be broadcast is first divided up and scattered among the processes, similar to an MPI Scatter. The scattered data is then collected back to all processes, similar to an MPI Allgather."
# 来源：[C] 论文，**广播 · Van de Geijn**。载体 = 一手 PDF，抓取日 2026-10-08。
#       ★★ 该段全是**散文**（不含公式、不跨标记）⇒ 逐个词提取形态与排版形态**一致**。
# 口径：★★★ **Van de Geijn 广播是【两段式】的**：**scatter**（散开）+ **allgather**（收回来）。
#       ★★ 这两段与 allreduce 的 reduce-scatter / all-gather **是同一个结构**
#         ⇒ ★ 02-stages-and-costs 的「分阶段看」在本主题**不是 allreduce 独有的**。
#       ★ 论文用散文说这件事（`similar to an MPI Scatter` / `similar to an MPI Allgather`）。
# 消费者：`01_algorithm_family.py` 的 `D.6`。
# 同值位置：无。

MPICH_AG_RING_STEPS_QUOTE = "the entire algorithm takes p − 1 steps."
# 来源：[C] 论文，**allgather · ring**。载体 = 一手 PDF，抓取日 2026-10-08。
#       ★★ `pdftotext -layout` 提取形态 = `the entire algorithm takes p − 1 steps.`
#         （破折号是数学减号 U+2212，**不是** ASCII 的 `-`）。
# 口径：★★ **步数**的逐字依据 —— ring allgather 的步数是 `p − 1`（**线性级**），
#       与递归倍增/Bruck 的 `lg p` / `⌈lg p⌉`（**对数级**）**是同一个原语的两种算法**。
#       ★★ 每设备搬运量**两者相同**（`(p−1)/p · n`）⇒ ★★★ **「步数」与「搬运量」是两条
#         独立的轴** —— 这是本主题最要紧的一条分类依据。
# 消费者：`01_algorithm_family.py` 的 `D.5`。
# 同值位置：`MPICH_AG_RING_TIME_QUOTE`。

MPICH_AG_RING_TIME_QUOTE = "Tring = (p − 1)α + (p−1)/p · nβ."
# 来源：[C] 论文，**allgather · ring**。载体 = 一手 PDF，抓取日 2026-10-08。
#       ★★ `pdftotext -layout` 提取形态 = `Tring = (p − 1)α + p−1 p nβ.`
#         （分数线被压平成 `p−1 p`）。
#       ★★★ 注意两个 `p−1` 的**排版不同**：`(p − 1)α` 的减号**两侧有空格**
#         （延迟项），`(p−1)/p` 的减号**两侧无空格**（分数的分子）—— 照抄时**别统一**。
# 口径：★★ **时间**的逐字依据。★ 带宽项 `(p−1)/p · nβ` = **每设备发出的元素数 × 每元素成本**
#       ⇒ 它同时给出了「每卡搬运量」的闭式。
# 消费者：`01_algorithm_family.py` 的 `D.5`。
# 同值位置：`MPICH_AG_RECDUB_TIME_QUOTE` / `MPICH_AG_BRUCK_TIME_QUOTE`（**同一个带宽项**）。

MPICH_AG_RECDUB_QUOTE = "all processes get all the data in lg p steps."
# 来源：[C] 论文，**allgather · 递归倍增（recursive doubling）**。载体 = 一手 PDF，
#       抓取日 2026-10-08。
#       ★★ `pdftotext -layout` 提取形态 = 与排版形态**一致**（**裸 `lg p`，无取整符号**）。
# 口径：★★ **步数**的逐字依据（`lg p`，对数级）。
#       ★★ ★ **这里【没有】`⌈⌉`** —— 与同节的 `Tbruck`（**有** `⌈lg p⌉`）**不同**，
#         也与论文里 reduce-scatter 的 `blg pc`（**下取整**）**不同**
#         ⇒ ★★★ **三种形态并存**，正是本文件抬头那条读法纪律的现场。
# 消费者：`01_algorithm_family.py` 的 `D.5`。
# 同值位置：`MPICH_AG_RECDUB_TIME_QUOTE`。

MPICH_AG_RECDUB_TIME_QUOTE = "Trec dbl = lg p α + (p−1)/p · nβ."
# 来源：[C] 论文，**allgather · 递归倍增**。载体 = 一手 PDF，抓取日 2026-10-08。
#       ★★ `pdftotext -layout` 提取形态 = `Trec dbl = lg p α + p−1 p nβ.`
#         （裸 `lg p`；分数线压平成 `p−1 p`）。
#       ★★★ **这与 `MPICH_ALLREDUCE_RECDUB_TIME_QUOTE` 是【两条不同的式子】** ——
#         同名（都叫 `Trec dbl`）**不同指**：本条是 **allgather**（带宽项 `(p−1)/p·n`），
#         那条是 **allreduce**（带宽项 `n lg p`）。★ 差一个 `lg p` 因子。
# 口径：★★ **时间**的逐字依据。★ 与 ring allgather 的带宽项**逐字相同** ⇒
#       **递归倍增在【搬运量】上不比 ring 省，省的是【步数】**。
# 消费者：`01_algorithm_family.py` 的 `D.5`。
# 同值位置：`MPICH_AG_RING_TIME_QUOTE` / `MPICH_AG_BRUCK_TIME_QUOTE`（**同一个带宽项**）。

MPICH_AG_BRUCK_STEPS_QUOTE = "Both algorithms take ⌈lg p⌉ steps in all cases"
# 来源：[C] 论文，**allgather · Bruck**。载体 = 一手 PDF，抓取日 2026-10-08。
#       ★★ `pdftotext -layout` 提取形态 = `Both algorithms take dlg pe steps in all cases`
#         （**有 ⌈⌉**）。
# 口径：★★ **步数**的逐字依据，★ 且**带「in all cases」的强限定** —— 对**任意** `p`
#       （不必是 2 的幂）都是 `⌈lg p⌉` 步。★★ 这正是 Bruck 相对递归倍增的**存在理由**：
#       递归倍增用 `d XOR 2^k` 配对，**要求 `p` 是 2 的幂**；Bruck 用**旋转**配对，任意 `p` 都行。
# 消费者：`01_algorithm_family.py` 的 `D.5`。
# 同值位置：`MPICH_BCAST_TREE_STEPS_QUOTE`（同为 `dlg pe`）。

MPICH_AG_BRUCK_TIME_QUOTE = "Tbruck = ⌈lg p⌉α + (p−1)/p · nβ."
# 来源：[C] 论文，**allgather · Bruck**。载体 = 一手 PDF，抓取日 2026-10-08。
#       ★★ `pdftotext -layout` 提取形态 = `Tbruck = dlg pe α + p−1 p nβ.`（**有 ⌈⌉**）。
# 口径：★★ **时间**的逐字依据。★★ 它的带宽项与 ring / 递归倍增**逐字相同**
#       ⇒ ★★★ **三个 allgather 算法的【搬运量】全等，只有【步数】不同
#         （`p−1` / `lg p` / `⌈lg p⌉`）** —— 这是本主题 A 类要比对的核心事实。
#       ★★ 该带宽项**对任意 `p` 都成立**（Bruck 只发「目标缺的那些块」，
#         丢掉了最后一步的重复）—— 这一点由 `01` 的模拟在 `p = 6` 上验证。
# 消费者：`01_algorithm_family.py` 的 `A.4`/`D.5`。
# 同值位置：`MPICH_AG_RING_TIME_QUOTE` / `MPICH_AG_RECDUB_TIME_QUOTE`。

MPICH_ALLREDUCE_RECDUB_TIME_QUOTE = "Trec−dbl = lg p α + n lg p β + n lg p γ."
# 来源：[C] 论文，**allreduce · 递归倍增**。载体 = 一手 PDF，抓取日 2026-10-08。
#       ★★ `pdftotext -layout` 提取形态 = `Trec−dbl = lg p α + n lg p β + n lg p γ.`
#         （裸 `lg p`，**无取整**；名字里的短横是 U+2212）。
#       ★★★ **与 `MPICH_AG_RECDUB_TIME_QUOTE` 同名不同指**（见模块抬头）：
#         那条是 allgather（带宽项 `(p−1)/p·n`）、本条是 allreduce（带宽项 `n lg p`）。
# 口径：★★ **时间**的逐字依据。★★★ **它的带宽项 `n lg p β` 正是 Rabenseifner 要压掉的那一项**
#       （见 `MPICH_RABENSEIFNER_TIME_QUOTE` 的 `2(p−1)/p · nβ`，两者**同量纲、可对拍**）。
#       ★ `n lg p γ` 是**归约**成本（`γ` = 每元素的归约操作成本），也与 Rabenseifner 的
#         `(p−1)/p · nγ` 对拍。
# 消费者：`01_algorithm_family.py` 的 `B.2`/`D.6`。
# 同值位置：`MPICH_AG_RECDUB_TIME_QUOTE`（**同名不同指**的那一条）。

MPICH_RABENSEIFNER_TIME_QUOTE = "Trabenseifner = 2 lg p α + 2(p−1)/p · nβ + (p−1)/p · nγ."
# 来源：[C] 论文，**allreduce · halving-doubling（Rabenseifner）**。载体 = 一手 PDF，
#       抓取日 2026-10-08。
#       ★★ `pdftotext -layout` 提取形态 =
#         `Therefore, the total cost is Trabenseif ner = 2 lg p α + 2 p−1 p nβ + p−1 p nγ.`
#         ★★★ 名字被**切成两行**（`Trabenseif` / `ner`）、裸 `lg p`（**无取整**）、
#           分数线压平成 `p−1 p`。★ 本常量存**排版形态**，提取形态记在这里。
# 口径：★★★ **时间**的逐字依据，也是本主题「算法族」的**焦点**：
#       · 步数项 `2 lg p α`（对数级，**优于 ring 的 `2(p−1)α`**）；
#       · 带宽项 `2(p−1)/p · nβ`（★ **与 ring allreduce 的 `2(p−1)/p · S` 逐字同形**）；
#       · 归约项 `(p−1)/p · nγ`。
#       ⇒ ★★★ **它同时拿到 ring 的搬运量与对数级的步数** —— 这就是「最优」的算术含义。
#       ★★ 论文自述它把带宽项从 `n lg p β` 降到 `2nβ`：用本条与
#         `MPICH_ALLREDUCE_RECDUB_TIME_QUOTE` **对拍可复核**（`2(p−1)/p ≈ 2`，而 `lg p` 随 `p` 增长）。
# 消费者：`01_algorithm_family.py` 的 `B.2`/`D.6`；`02_stages_and_costs.py` 的 `D.5`。
# 同值位置：无（★ **本主题的焦点常量**，不要与别的合并）。

MPICH_TREC_HALF_QUOTE = "Trec half = (⌊lg p⌋ + 2)α + 2nβ + n(1 + (p−1)/p)γ."
# 来源：[C] 论文，**reduce-scatter（recursive halving）**，★ **`p` 不是 2 的幂**那一支。
#       载体 = 一手 PDF，抓取日 2026-10-08。
#       ★★ `pdftotext -layout` 提取形态 =
#         `Trec half = (blg pc + 2)α + 2nβ + n(1 + p−1 p )γ.`
#         ★★★ 这里是 **`blg pc` = ⌊lg p⌋（下取整）** —— **与 `dlg pe`（上取整）和裸 `lg p` 三足鼎立**。
# 口径：★★★ 【本主题唯一一条**分情况**的代价式】—— 它讲的是**不整除**时的 reduce-scatter 代价。
#       ★★ 本库的**独立价值**：它给「**不整除不是边界情况**」（本库 `P25`）提供了一条
#         **外部 [C] 级**的实例（本域 `EP` / `TP` 的余数账是内部实例）。
#       ★★★ **它自带「这是近似」的限定语**（见 `MPICH_TREC_HALF_QUALIFIER`）——
#         引用时**必须带上**，否则是把一条近似式当成精确式。
# 消费者：`02_stages_and_costs.py` 的 `D.6`。
# 同值位置：无。

MPICH_TREC_HALF_QUALIFIER = "This cost is approximate because some imbalance exists in the amount of work each process does"
# 来源：[C] 论文 —— 紧跟 `MPICH_TREC_HALF_QUOTE` 的**限定句**。载体 = 一手 PDF，抓取日 2026-10-08。
#       ★★ `pdftotext -layout` 提取形态与本行的排版形态**一致**（全是散文）。
# 口径：★★★ **它是【近似】的官方自述** —— 那句话说的是「每个进程的工作量存在不均衡」。
#       ⇒ ★★ 本主题引用 `Trec half` 时**必须并置本条**（本库纪律：**引一条式子的限定语
#         与式子本身是同一件事的两个部分**）。
# 消费者：`02_stages_and_costs.py` 的 `D.6`（★ 与 `MPICH_TREC_HALF_QUOTE` **同一条**断言）。
# 同值位置：无。

MPICH_A2A_INDEX_QUOTE = "we use the index algorithm by Bruck et al."
# 来源：[C] 论文，**all-to-all · index（Bruck）**。载体 = 一手 PDF，抓取日 2026-10-08。
#       ★★ `pdftotext -layout` 提取形态与该行排版形态**一致**。
#       ★★★ **这是一条【同名不同指】的现场**：`Bruck` 在本文**不是 allreduce 的算法**，
#         而是 **allgather 与 all-to-all（index）** 的算法（见 `MPICH_BRUCK_SCOPE_NOTE`）。
# 口径：★★ **all-to-all 的短消息分支**用 Bruck 的 index 算法（它的步数见下一条）。★ 该句
#       本身**不给步数**，步数在 `MPICH_A2A_INDEX_STEPS_QUOTE` 里（同一段的下半句）。
# 消费者：`03_all_to_all_and_p2p.py` 的 `D.5`。
# 同值位置：`MPICH_A2A_INDEX_STEPS_QUOTE`（同一段）。

MPICH_A2A_INDEX_STEPS_QUOTE = "It is a store-and-forward algorithm that takes ⌈lg p⌉ steps"
# 来源：[C] 论文，**all-to-all · index（Bruck）**。载体 = 一手 PDF，抓取日 2026-10-08。
#       ★★ `pdftotext -layout` 提取形态 = `It is a store-and-forward algorithm that takes dlg pe steps`
#         （**有 ⌈⌉**）。
# 口径：★★ **步数**的逐字依据：all-to-all 的 index 算法是 `⌈lg p⌉` 步（对数级）。
#       ★★ 它**只适用于短消息**（见 `MPICH_A2A_IRECVISEND_QUOTE` 的分档与其限定语）。
# 消费者：`03_all_to_all_and_p2p.py` 的 `D.5`。
# 同值位置：`MPICH_A2A_INDEX_QUOTE`（同一段）。

MPICH_A2A_IRECVISEND_QUOTE = "for medium-sized messages (256 bytes to 32 KB per message)"
# 来源：[C] 论文，**all-to-all · irecv-isend**。载体 = 一手 PDF，抓取日 2026-10-08。
#       ★★ `pdftotext -layout` 提取形态与该行排版形态一致（★ 那对数字 `256` / `32 KB` 是散文里的）。
# 口径：★★★ **这是 MPICH 在 2005 年选的分档**，**不是普适规律**
#       ⇒ ★★ 引用**必须**并置 `MPICH_CUTOFF_QUALIFIER`。
#       ★ 它给出了 all-to-all 的**第二个算法**（irecv-isend）与它的适用档。
# 消费者：`03_all_to_all_and_p2p.py` 的 `D.6`（★ 与 `MPICH_CUTOFF_QUALIFIER` 同一条）。
# 同值位置：无。

MPICH_A2A_PAIRWISE_QUOTE = "For long messages and power-of-two number of processes, we use a pairwise-exchange algorithm, which takes p − 1 steps."
# 来源：[C] 论文，**all-to-all · 成对交换（pairwise-exchange）**。载体 = 一手 PDF，
#       抓取日 2026-10-08。
#       ★★ `pdftotext -layout` 提取形态与该行排版形态一致（数学减号 U+2212）。
# 口径：★★ **步数**的逐字依据：成对交换是 `p − 1` 步（线性级）—— **与 index 的 `⌈lg p⌉` 相对**。
#       ★★ 它带**两个前提**：**长消息** + **`p` 是 2 的幂**
#         ⇒ 引用时**必须带上**（本库纪律：前提与结论同引）。
# 消费者：`03_all_to_all_and_p2p.py` 的 `D.5`。
# 同值位置：无。

MPICH_REDUCE_ALGOS = ("binomial tree", "recursive doubling",
                      "recursive halving and doubling", "binary blocks", "ring")
# 来源：[C] 论文，**reduce / allreduce 的算法枚举**。载体 = 一手 PDF，抓取日 2026-10-08。
#       ★★ 原文逐字（`pdftotext -layout` 提取形态与排版形态一致）：
#         `binomial tree and recursive doubling` … `recursive halving and doubling,
#          binary blocks, and ring.`
#       ★★ 本常量把那一串**拆成 5 个名字**（★ 拆开是为了能**逐元素**钉住；
#         把整句存成一个串会让**元素级变异**漏网）。
# 口径：★★★ **这是论文自己给的 allreduce 算法清单**（5 个）。
#       ★★ 注意 `recursive halving and doubling` 就是 halving-doubling（Rabenseifner）。
#       ★★ 而 **Bruck 不在这个清单里** —— 见 `MPICH_BRUCK_SCOPE_NOTE`。
# 消费者：`01_algorithm_family.py` 的 `D.6`（逐元素）。
# 同值位置：无。

MPICH_REDUCE_ALGO_COUNT = 5
# 来源：[C] 论文 —— 上一行那个清单的**元素个数**（★ 逐条数出来的，不是估的）。
# 口径：★★ 取它当**常量**而不是把 `5` 写死在断言里，是为了让 `mutate.py` 能看见它
#       （★ 同时 `01` 的 `D.6` 用 `== 5` 逐字钉住）。
# 消费者：`01_algorithm_family.py` 的 `D.6`（★ 与 `MPICH_REDUCE_ALGOS` **同一条**断言）。
# 同值位置：`MPICH_REDUCE_ALGOS`。

MPICH_CUTOFF_QUOTE = "experimentally determined cutoff points"
# 来源：[C] 论文 —— 论文**自述**它的分档阈值是「由实验确定的切分点」。载体 = 一手 PDF，
#       抓取日 2026-10-08。★★ `pdftotext -layout` 提取形态与该行排版形态一致（散文）。
# 口径：★★★ **本条是 `F5` 的依据**：论文里的 `256 B` / `32 KB` / `2 KB` / `12 KB` 是
#       **2005 年 MPICH 在 Myrinet / IBM SP 上的【实现选择】** ⇒ **不得**升格成
#       「集合通信的普适分档」。
# 消费者：`03_all_to_all_and_p2p.py` 的 `D.6`（★ 与 `MPICH_CUTOFF_QUALIFIER` 同一条）。
# 同值位置：无。

MPICH_CUTOFF_QUALIFIER = "★ 这些阈值是 2005 年 MPICH 在 Myrinet / IBM SP 上的实现选择（论文自述 experimentally determined cutoff points）⇒ 不得写成「集合通信的普适分档」"
# 来源：**[E] 本库的引用纪律**（★ 不是论文里的话）—— 它把 `MPICH_CUTOFF_QUOTE` 的**含义**
#       写成一句必须与引用**并置**的限定语（`F5`）。
# 口径：★★★ 它**没有外部出处**，所以**不得**把它写成引文（本库纪律：**自造真值**是一种错误模式）。
#       ★ 它的存在理由是**可复核**：`03` 的 `D.6` 用 `==` 钉住它，于是「引用分档时
#         有没有带限定语」变成了一条**可被变异测试抓住**的断言。
# 消费者：`03_all_to_all_and_p2p.py` 的 `D.6`。
# 同值位置：无。

MPICH_BRUCK_SCOPE_NOTE = "Bruck 算法在本文里是 allgather 与 all-to-all(index) 的算法，不是 allreduce 的算法"
# 来源：**[E] 本库的消歧记录**（★ 不是论文里的话）—— 依据是论文的**算法归属**
#       （allgather 节与 all-to-all 节各给了 Bruck；allreduce 节的清单见 `MPICH_REDUCE_ALGOS`）。
# 口径：★★★ 它是一条**同名不同指**的记录：把 `Bruck` 当成 allreduce 算法会**写错正文**
#       （allreduce 里 Bruck 只作为**组合件**出现）。
#       ★ 它的存在理由是**可复核**：`01` 与 `03` 各用 `==` 钉住它一次。
# 消费者：`01_algorithm_family.py` 的 `D.6`、`03_all_to_all_and_p2p.py` 的 `D.6`。
# 同值位置：无。


# ===========================================================================
# 2. [E] 本库设计常量 —— 「本程序取哪一组输入」
#    ★ 它们**不是任何来源里的数** ⇒ `# 来源：` 一律写 `[E] 本库设计`，
#      并写明**为什么取这个值**（可手算复核）。
# ===========================================================================

DEV_GRID = (2, 4, 8, 16, 64)
# 来源：[E] 本库设计。**五档设备数** —— ring allreduce 的消息序列模拟与闭式对拍的档位。
# 口径：★★ 取这五档的理由可手算复核：它们**都是 2 的幂** ⇒
#       ① 递归倍增的 `d XOR 2^k` 配对**正好可行**（不必是 2 的幂也能跑 ring，但要跑 RD 就必须是）；
#       ② `RING_ELEMS = 64` 被**五档全部整除**（`64/64 = 1`）⇒ 每块的元素数都是整数，
#          「块大小不等」这个干扰项在 A 类里**不存在**（余数的账在 `02` 的 D 类）。
#       ★★ 第 5 档 `64` 是**饱和那一条**（`V(64)/S = 1.968750`）的档位。
#       ★★ **本元组必须逐元素钉住**（`01` 的 `D.7`）—— 只钉「整体」会让元素级变异漏网。
# 消费者：`01_algorithm_family.py` 的 `A.1`/`A.2`/`D.7`。
# 同值位置：无。

SAT_GRID = (2, 4, 8, 64, 512, 4096)
# 来源：[E] 本库设计。**六档** —— `V(N)/S` 的**饱和表**档位（★ 与规划 `§D` 的 `D2b` 同档）。
# 口径：★★ 取 `512` 与 `4096` 的理由：`V(N)/S = 2(N−1)/N` 单调上升趋近 `2`，
#       前四档看得出「涨」，后两档看得出「**已经饱和**」（`1.996094` / `1.999512`）。
#       ★★★ **本主题【不推导】`V(N)`**（推导在 `performance-models/02:147-155`）——
#         这里只**回链**并报出它的饱和值，用来做 B 类的边界检查。
#       ★★ **本元组必须逐元素钉住**（`01` 的 `D.7`）。
# 消费者：`01_algorithm_family.py` 的 `B.1`/`D.7`。
# 同值位置：无。

RING_ELEMS = 64
# 来源：[E] 本库设计。ring allreduce 消息序列模拟里的**数据元素总数** `S`。
# 口径：★★ 取 `64` 的理由：`DEV_GRID` 的**五档全部整除它**（含 `64/64 = 1`）⇒
#       每个块的元素数是整数 ⇒ 「块大小不等」不进 A 类。
#       ★★ 它是**标量**（不是元组）⇒ `mutate.py` 的整型变异 `v + 1` 会把它变成 `65`，
#         那时 `64 % 65 != 0` ⇒ 会被 `A.2` 的**整除守卫**抓住。
# 消费者：`01_algorithm_family.py` 的 `A.1`/`A.2`/`D.7`。
# 同值位置：无。

GATHER_GRID = (2, 4, 8, 16)
# 来源：[E] 本库设计。**四档** —— allgather 三个算法（ring / 递归倍增 / Bruck）的步数对拍档位。
# 口径：★★ 取 2 的幂的理由：**递归倍增要求 `p` 是 2 的幂**（`d XOR 2^k`）⇒
#       只有在这几档上三个算法才**同时可跑**、能摆在一张表里比。
#       ★★ **本元组必须逐元素钉住**（`01` 的 `D.7`）。
# 消费者：`01_algorithm_family.py` 的 `A.3`/`A.4`/`D.7`。
# 同值位置：无。

GATHER_ELEMS = 32
# 来源：[E] 本库设计。allgather 模拟里的**元素总数** `n`。
# 口径：★★ 取 `32` 的理由：被 `GATHER_GRID` 的**四档全部整除**（`32/16 = 2`）⇒
#       每块元素数是整数。★ 它是标量 ⇒ 变异成 `33` 会被整除守卫抓住。
# 消费者：`01_algorithm_family.py` 的 `A.3`/`A.4`/`D.7`。
# 同值位置：无。

GATHER_ODD = 6
# 来源：[E] 本库设计。**一个【不是 2 的幂】的档** —— 用来验 Bruck 的「in all cases」。
# 口径：★★★ 取 `6` 的理由：`⌈lg 6⌉ = 3` 而 `⌊lg 6⌋ = 2` ⇒ **两个取整符号在这一档上分得开**
#       （在 2 的幂上它们相等、量不出区别）—— ★ 这正是 `MPICH_TREC_HALF_QUOTE` 用 `blg pc`
#       而 `MPICH_AG_BRUCK_STEPS_QUOTE` 用 `dlg pe` 的**可观测差别**。
#       ★★ 它**只进 Bruck**（递归倍增在 6 上根本不跑，因为 `d XOR 2^k` 走不完）。
#       ★★ 它是标量 ⇒ 变异会被 A 类的步数断言抓住（`6 → 7` 时 `⌈lg 7⌉ = 3` 不变，
#         但 `GATHER_ODD` 的**整除性**变了 ⇒ `D.7` 的逐元素钉住抓它）。
# 消费者：`01_algorithm_family.py` 的 `A.4`/`D.7`。
# 同值位置：无。

A2A_GRID = (2, 4, 8)
# 来源：[E] 本库设计。**三档** —— all-to-all 成对交换的步数/搬运量对拍档位。
# 口径：★★ 取 2 的幂的理由：成对交换的前提之一是 **`p` 是 2 的幂**（见
#       `MPICH_A2A_PAIRWISE_QUOTE`），而 `index` 算法任意 `p` 都行 ⇒ 取公用的档。
#       ★★ **本元组必须逐元素钉住**（`03` 的 `D.7`）。
# 消费者：`03_all_to_all_and_p2p.py` 的 `A.1`/`A.2`/`D.7`。
# 同值位置：无。

A2A_BLOCK = 4
# 来源：[E] 本库设计。all-to-all 里**每对设备之间**的消息元素数 `b`。
# 口径：★★ 取 `4` 的理由：每设备要发 `p − 1` 份，`p = 8` 时是 `7` 份 ⇒
#       每设备发出 `28` 个元素 —— 规模小到能手算复核，又足以让 `p − 1` 的**线性增长**看得出来。
#       ★★ 它是标量 ⇒ 变异会被 `A.1` 的搬运量断言抓住。
# 消费者：`03_all_to_all_and_p2p.py` 的 `A.1`/`A.2`/`B.1`/`D.7`。
# 同值位置：无。

STAGE_ELEMS = 8
# 来源：[E] 本库设计。**两段式对拍**（reduce-scatter ⊕ all-gather）用的元素总数。
# 口径：★★ 取 `8` 的理由：在 `02` 用的两档设备数（`2` 与 `4`）上**都整除**
#       （`8/2 = 4`、`8/4 = 2`）⇒ 两段各自的块大小都是整数。
#       ★★ 它是标量 ⇒ 变异会被 `A.1` 的分段对拍抓住。
# 消费者：`02_stages_and_costs.py` 的 `A.1`/`A.2`/`D.7`。
# 同值位置：无。

STAGE_GRID = (2, 4)
# 来源：[E] 本库设计。**两档** —— 两段式对拍的设备数。
# 口径：★★ 取 `2` 与 `4` 的理由：`2` 是**退化对照**（两段各只有一步，
#       分段与不分段的差别最小 ⇒ 若分段写错，在 `2` 上最难看出）；
#       `4` 让每段有 `3` 步 ⇒ 两段的**顺序与累加目标**都变得可观测。
#       ★★ **本元组必须逐元素钉住**（`02` 的 `D.7`）。
# 消费者：`02_stages_and_costs.py` 的 `A.1`/`A.3`/`D.7`。
# 同值位置：无。

TOL_MACHINE = 1.0e-12
# 来源：[E] 本库设计。**机器精度档**的判据阈（相对/绝对通用）。
# 口径：★★ 本主题的判据有两类：**逐位对拍**（消息序列模拟 vs 单进程参照）差落在 **1e-16 档**
#       或**恰好为 `0`**；**内容级错法**（块索引写错）差落在 **1e+0 档** ——
#       **相隔十几个数量级**。取 `1e-12` 是为了容纳「恰好为 `0`」与「几 ulp」两种正确形态。
#       ★★ **不得把它当作「物理量级的阈」**。
#       ★★★ 本主题**另有两条**与它并列的**内容级**阈，写在编号程序内部（**不进本模块**）：
#         · `1e-6`：C 类是**内容级**错法 ⇒ 用它区分「错法真的错了」与「舍入」；
#         · 本主题的 A 类是**计数**（步数、元素数、块数）⇒ **整数相等**，不用浮点阈。
# 消费者：`01`/`02`/`03` **各自**的 `D.3`（`preflight()`，先跑）。
# 同值位置：`100-parallelism/expert-parallel` 等**同域别的主题**的编号程序里各有一个
#           同名常量（★ 它们**不在**那几篇的 `specs.py` 里）；本文件这一份**独立存放**。


# ===========================================================================
# 3. UNSOURCED —— 待办清单，不是免责声明
# ===========================================================================
#
# ★ `CONTRIBUTING` §2：「不得**既保留数字又记 UNSOURCED**」—— 下面每一项都**没有**出现在
#   本模块的正文常量里。
# ★★ **本元组【留空】**，理由有两条，都是实测教训：
#   · **每一项的字符串都会成为 `make mutate` 的叶子**，而它们**没有消费者**
#     ⇒ 会全部记成**漏网**（本库实测：只用于打印的模块级常量表制造过漏网）。
#     ★★ **而【空元组本身】有消费者**：`01` 的 `D.9` 用 `specs.UNSOURCED == ()` 钉住它
#       （★ 0 个叶子 ⇒ 不产生变异体）。
#   · 本主题的两处「未核到」已经**写在正文常量自己的 `# 来源：` 里**，不欠一条记录：
#     ① **`Trec half` 的精确（非近似）形式** —— 论文**只给了近似式**并自述
#        `This cost is approximate …`（见 `MPICH_TREC_HALF_QUALIFIER`）；
#        ★ 本库**没有**取到更精确的形式 ⇒ **不得**给它补一个「精确式」。
#     ② **MPI 标准的 P2P 原语规范定义**（`MPI_Send` / `MPI_Recv` 的语义条文）——
#        本库**没有**下载 MPI 标准正文（只 HEAD 成功）⇒ ★★ **不得引「MPI 标准 §X 说」**；
#        `03` 的 P2P 部分只写**原语的语义要点**并在正文里标 `[E] 本库的组织`。
UNSOURCED = ()

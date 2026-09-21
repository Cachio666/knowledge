# -*- coding: utf-8 -*-
r"""30-gpu-programming/ptx-sass 的外部事实常量源。

本模块是 `CONTRIBUTING` §2 约定的**共享事实源**（`specs.py`）：三个编号程序从它读常量，
推导过程可复核，**常量本身不可自证**（域 README「代码约定」一节）。

★★ 本模块**只放两类东西**（计划 §0.6 第 3 条 + 本主题事实基线 §0.7 E）：

  ① **A 级语义锚点**：PTX ISA 原文的**逐字引文** + **章节号**（`ISA`）；
     ⇒ 引文**由脚本从下载的原始 HTML 抽取**，不手打（抽取器在 `01_virtual_isa.py`
        的 `--anchors-net` 模式里，抽取口径见下）。
  ② **可从这些引文机械推出的常量**（`PTX_SUBSET` 的清单本身、`RAW_COUNTS` 的计数、
     `ISA_ANCHOR_SECTIONS` 的归属、`UP_MODE_UNDECIDED` 的两式）。

  ⇒ **不放散文**。判据是 `CONTRIBUTING` 第 220–222 行那句：
     「**把这条常量改错，程序会报错吗？**」本模块里**每一个值都被三个编号程序之一断言**：

       · `ISA[i]['quote']`        —— 被 `must_contain` 的**逐字子串**钉住（改一个字即失败），
                                     并被 `--anchors-net` 的**在线再抽取**钉住；
       · `ISA[i]['section']`      —— 被 **`01` 的 `A.2`（归属断言）** 钉住：章节号前缀必须
                                     精确等于本模块记的值，且 `B4` 必须**不在** `9.7.15.1`；
       · `ISA[i]['covered_by']`   —— 被 `01` 的 `A.3`（非空 + 标签真实存在 + 并集覆盖）钉住；
       · `RAW_COUNTS['counts']`   —— 被 `01` 的 `A.4` 钉住（并**与引文自身**交叉核对）；
       · `PTX_SUBSET['supported']`—— 被 `01` 的 `A.1` 与 `ptxsub.SUPPORTED_OPS`
                                     **双向集合相等**（改任何一边都失败）；
       · `PTX_SUBSET['unsupported_in_isa']` —— 被 `01` 的 `C.3`/`C.4` 钉住（引擎必须拒绝它们），
                                    并被 `--anchors-net` 钉住（**每个字面量都必须在 ISA 原文里出现**）。

★★ 引文是怎么来的（本主题**唯一**允许的引文来源，计划 §0.7 E9 的机械执行）：

  ```
  raw = curl -sSL https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
  # ★★ 第一件事：在**原始字节**上数计数，**不先转成文本再数**
  #     raw.count('j &gt;= maxLane') == 2      ← 断言在 01 的 A.4 / --anchors-net 里
  #     raw.count('j &gt;= minLane') == 0
  # 再抽取引文：
  text = strip(<script>|<style>) → strip(<...>) → html.unescape(...) → 空白折叠
  ```

  ★★ **为什么先数原始字节**：本主题的规划阶段就死在这一步上——
     抽取脚本打印的是 `maxLane`（正确），**往规格里转写时写成了 `minLane`**，
     于是长出一条「ISA 用 `minLane`、二手转述是错的」的叙事，扩散到三处。
     **工具输出不是结论；转写是独立的出错环节**（计划 §2「本域特有的两条」第 3 条）。
     ⇒ 本模块的 `RAW_COUNTS` 是这条教训的**机械载体**。

  ★ **去标签会留下空格**（抽取口径的**已知产物**，不是页面内容）：行内 `<code>x</code>`
    被换成空格后会出现 `operand b ,` / `shfl.sync .` / `membermask .` 这种「标点前多一个空格」。
    引文**逐字保留**这个产物（因为它就是抽取器的输出），而 `must_contain` 的字面量
    **一律不跨标签边界**——跨了边界就匹配不上（`triton` 主题在这里翻过车）。
  ★ **这个产物也会出现在 `section` 里**：`7.15.1` 那条标题的原文是
    「… Instructions: bar, barrier」，抽取后成了「… Instructions: bar , barrier」。
    `section` 记的是**抽取器的输出**（所以 `B1`/`B2`/`B3` 三条的 `section` 里那个空格是**故意的**）。
  ★ **标题尾部的锚点字形被丢掉**：原文每个 `<h4>` 末尾有一个**私用区**码位
    （`U+F0C1`，是文档的锚点图标，不是文字）。抽取器把它剥掉——
    只剥**私用区**码位，别的字符一个不动（`--anchors-net` 会把剥之前的原样打出来）。

★ 来源等级（定义在 `CONTRIBUTING.md` §2「外部来源分级 A–E」）：

  · **A 级：厂商官方规范**——PTX ISA 的原文。`ISA` 的 9 条锚点全部是 A 级。
  · **B 级：某一实现的官方材料**——本模块**不含** B 级条目。本主题**没有**任何外部实现工件
    （三个源仓库里 `.ptx` / `.sass` / `.cubin` / `.fatbin` **共 0 个**，计划 §0.7 A）。
  · **E 级：本库推导**——`BANK_CONFLICT` / `UP_MODE_UNDECIDED` 的**数字**是推导值，
    在编号程序内部**当场算出并断言**；本模块只记**表的形状**（哪几个 stride、哪两种模式），
    以及**既有文档的规则在哪种模式下成立的结论**（E 级，带 `covered_by`）。
"""

PTX_ISA_URL = ("https://docs.nvidia.com/cuda/parallel-thread-execution/index.html")
"""PTX ISA 全文的 URL。★ **`WebFetch` 对它被拦截，但 `curl -sSL` 直接取得到**
（本机 `curl -sSL` 取得：HTTP **200**、`size = 3928316` 字节）。计划 §0.7 E9。"""

ISA_HTTP_SIZE_OBSERVED = 3928316
"""本机 `curl -sSL` 取到的字节数（观测值）。
★ 在线再抽取时**只断言下界**（`ISA_HTTP_SIZE_MIN`）而**不断言相等**——
ISA 是一份**会修订**的文档，把「等于 3928316」写成判据，会把一次正常的文档修订
报成一次回归。下界挡的是**截断的下载**，那才是真正会静默污染抽取结果的故障。"""

ISA_HTTP_SIZE_MIN = 3000000
"""在线再抽取时的字节数下界（见上）。"""


# ===========================================================================
# 0. ★★ 原始字节上的计数（本主题最贵的一条教训的机械载体）
# ===========================================================================
#
# 口径：在**下载回来的原始 HTML 字节**上做 `raw.count(<字面量>)`，
#       **不经过任何文本转写**（`raw.count` ≠ 「先 strip 标签再数」——
#       后者是另一个层，而那一层正是出错的地方）。
#
# ★ 下面四个字面量之所以成一组，是为了让「把 `maxLane` 抄成 `minLane`」这件事
#   **在结构上不可能通过检查**：
#     · `.up` 分支只有一处写 `j >= maxLane`，且**每一节各一处**（两节 ⇒ 共 2 处）；
#     · `j >= minLane` **一处都没有** —— `minLane` 只出现在定义行与 `.idx` 分支里。
#   把 `ISA['P1']['quote']` 里的 `maxLane` 改成 `minLane`，`01` 的 `A.4` 立刻失败。

RAW_COUNTS = {
    "url": PTX_ISA_URL,
    "counts": (
        ("j &gt;= maxLane", 2),                      # 9.7.10.5 与 9.7.10.6 各 1
        ("j &gt;= minLane", 0),                      # ★ 一处都没有
        ("j &lt;= maxLane", 6),                      # 每节 3（down / bfly / idx）
        ("maxLane = (lane[4:0] &amp; segmask", 1),  # 9.7.10.6 的定义行（变量名 segmask）
        ("maxLane = (lane[4:0] &amp; mask", 1),     # 9.7.10.5 的定义行（变量名 mask）
        ("minLane = (lane[4:0] &amp; ", 2),          # 两节各一条定义行
        ("j = minLane", 2),                          # 两节各一条 `.idx` 分支
    ),
    "covered_by": ("A.4",),
    "note": "★ `j &gt;= maxLane` = 2 与 `j &gt;= minLane` = 0 这两条是**一起读**的："
            "前者证明原文写的是 `maxLane`，后者证明「原文写 `minLane`」这个说法"
            "**在本页里找不到任何字面量支撑**。单独看任何一条都不足以定案。"
            "★ 本条的**载体**是 `01_virtual_isa.py` 的 `A.4`（离线：与引文自身交叉核对）"
            "与 `make anchors`（在线：对**刚下载的**原始字节重新计数）。",
}


# ===========================================================================
# 1. ★ A 级语义锚点（PTX ISA 原文，逐字）
# ===========================================================================
#
# 字段说明：
#   section      该引文在 ISA 里的**章节号 + 标题**（人工定位到最近的上位 `<h4>`，
#                `make anchors` 会**重新定位一遍并比对**——这是 P7 的机械载体）
#   quote        抽取器输出的**逐字**引文（空白折叠成单个空格）
#   must_contain 必须逐字出现的短字面量元组；**不跨标签边界**
#   slice        抽取器用的切片标记 `(起点, 终点, 再截断到)`，作用在**原始 HTML** 上
#   locate       用哪个字面量/正则去定位「这段引文在原文的哪个偏移」——只为查出
#                **它上面最近的那个 `<h4>` 标题**（= 归属检查）。`'re:'` 前缀表示正则。
#   covered_by   本锚点**被哪些断言支撑**（标签字符串元组，如 ("A.2", "C.3")）
#   note         会被打印的解释。它**不单独充当依据**，只说明该条的作用域限制。
#
# ★★ `note` 是**索引**，不是**唯一载体**（沿用 `cuda/code/specs.py` 的裁定）：
#   一个只在 `note` 里出现的论断是**不可证伪**的（P5）。三条规矩：
#     ① 每个锚点的 `covered_by` **非空**；
#     ② 里面每个标签都必须**真实存在**于某个编号程序的断言清单里（用 `ast` 提取，不是编的）；
#     ③ `note` 里凡属**承重论断**的句子，必须能被 `covered_by` 指向的断言机械判定。
#   ★ 校验落点：`01_virtual_isa.py` 的 **A.3**（非空 + 标签真实 + 并集覆盖全部锚点）。
#   ★ 标签允许**跨程序**：`A.2` 在 `01`、`C.3` 在 `02` —— 每个 `note` 里写明是哪个程序。

ISA = {

    # -----------------------------------------------------------------------
    # 1.1 shfl.sync 的语义伪代码（`9.7.10.6`）
    # -----------------------------------------------------------------------
    "P1": {
        "section": "9.7.10.6. Data Movement and Conversion Instructions: shfl.sync",
        "quote": "maxLane = (lane[4:0] & segmask[4:0]) | (cval[4:0] & ~segmask[4:0]); "
                 "minLane = (lane[4:0] & segmask[4:0]); "
                 "switch (.mode) { case .up: j = lane - bval; pval = (j >= maxLane); break; "
                 "case .down: j = lane + bval; pval = (j <= maxLane); break; "
                 "case .bfly: j = lane ^ bval; pval = (j <= maxLane); break; "
                 "case .idx: j = minLane | (bval[4:0] & ~segmask[4:0]); "
                 "pval = (j <= maxLane); break; } "
                 "if (!pval) j = lane; // copy from own lane "
                 "d = SourceA[j]; // copy input a from lane j "
                 "if (dest predicate selected) p = pval;",
        "must_contain": (
            "maxLane = (lane[4:0] & segmask[4:0]) | (cval[4:0] & ~segmask[4:0])",
            "minLane = (lane[4:0] & segmask[4:0])",
            "case .up: j = lane - bval; pval = (j >= maxLane); break;",
            "case .bfly: j = lane ^ bval; pval = (j <= maxLane); break;",
            "case .idx: j = minLane | (bval[4:0] & ~segmask[4:0]);",
            "if (!pval) j = lane; // copy from own lane",
        ),
        "slice": ("maxLane = (lane[4:0] &amp; segmask", "PTX ISA Notes", "p = pval;"),
        "locate": "maxLane = (lane[4:0] &amp; segmask",
        "covered_by": ("A.4", "A.2", "C.3"),
        "note": "★★ **本节就是本主题最贵的一条教训的现场**：`.up` 分支写的是 **`maxLane`**"
                "（`j >= maxLane`），**不是 `minLane`**。"
                "`minLane` 在本节里只出现两次：定义行与 `.idx` 分支。"
                "★ 变量名是 **`segmask`**（本节），**`9.7.10.5` 那一节才叫 `mask`**——"
                "两节是同一段伪代码的两个版本，混着引会串味（P9 同名不同指）。"
                "★ 覆盖它的断言分别在两个程序里：`A.4`（`01`：原始字节计数 + 引文内部计数）、"
                "`A.2`/`C.3`（`02`：独立参照实现、`.up` 的两式对照）。"
                "★ 本条**不是为了支持**「所以 `.up` 该用 `minLane`」——恰恰相反，"
                "它支持的结论是「**原文写 `maxLane`**」；"
                "「那本库为什么不照抄」是 `UP_MODE_UNDECIDED` 那条，且结论是**未定案**。",
    },
    "P1b": {
        "section": "9.7.10.5. Data Movement and Conversion Instructions: shfl (deprecated)",
        "quote": "maxLane = (lane[4:0] & mask[4:0]) | (cval[4:0] & ~mask[4:0]); "
                 "minLane = (lane[4:0] & mask[4:0]); "
                 "switch (.mode) { case .up: j = lane - bval; pval = (j >= maxLane); break; "
                 "case .down: j = lane + bval; pval = (j <= maxLane); break; "
                 "case .bfly: j = lane ^ bval; pval = (j <= maxLane); break; "
                 "case .idx: j = minLane | (bval[4:0] & ~mask[4:0]); "
                 "pval = (j <= maxLane); break; } "
                 "if (!pval) j = lane; // copy from own lane "
                 "d = SourceA[j]; // copy input a from lane j "
                 "if (dest predicate selected) p = pval;",
        "must_contain": (
            "maxLane = (lane[4:0] & mask[4:0]) | (cval[4:0] & ~mask[4:0])",
            "case .up: j = lane - bval; pval = (j >= maxLane); break;",
            "case .idx: j = minLane | (bval[4:0] & ~mask[4:0]);",
        ),
        "slice": ("maxLane = (lane[4:0] &amp; mask", "PTX ISA Notes", "p = pval;"),
        "locate": "maxLane = (lane[4:0] &amp; mask",
        "covered_by": ("A.4",),
        "note": "★ 弃用版 `shfl`（`9.7.10.5`）的**同一段伪代码**：`.up` 也写 `maxLane`。"
                "⇒ 两节对照才有意义——**只引一节**无法排除「另一节写的是 `minLane`」。"
                "★ 与 `P1` 的唯一差别是变量名（`mask` vs `segmask`）。"
                "★ 消费它的断言是 `01` 的 `A.4`：它要求**两节**各自的 "
                "`maxLane`/`minLane` 计数都成立。",
    },
    "P2": {
        "section": "9.7.10.6. Data Movement and Conversion Instructions: shfl.sync",
        "quote": "shfl.sync.up.b32 Ry|p, Rx, 0x1, 0x0, 0xffffffff;",
        "must_contain": ("shfl.sync.up.b32 Ry|p, Rx, 0x1, 0x0, 0xffffffff;",
                         "shfl.sync.up.b32"),
        "slice": ("shfl.sync.up.b32  Ry|p, Rx, 0x1,", "0xffffffff;", None),
        "locate": "shfl.sync.up.b32  Ry|p, Rx, 0x1,",
        "covered_by": ("A.4", "C.4"),
        "note": "★ ISA 自己给 `.up` 的例子，**`cval = 0x0`**（`9.7.10.6` 的 Examples 一节）。"
                "这条是 `UP_MODE_UNDECIDED` 里「两种可能」的分岔口："
                "`cval = 0` 时 `maxLane = lane & segmask = minLane`，**两式恰好重合**，"
                "于是「ISA 用 `maxLane`」与「`.up` 应取 `minLane`」在**这个参数下无法区分**。"
                "★ 消费它的断言：`01` 的 `A.4`（引文逐字）与 `02` 的 `C.4`"
                "（**从这条引文里解析出 `cval == 0`**，再断言该参数下两式逐 lane 相等）。",
    },
    "P3": {
        "section": "9.7.10.5. Data Movement and Conversion Instructions: shfl (deprecated)",
        "quote": "Introduced in PTX ISA version 3.0. Deprecated in PTX ISA version 6.0 "
                 "in favor of shfl.sync .",
        "must_contain": ("Introduced in PTX ISA version 3.0.", "in favor of shfl.sync"),
        "slice": ("re:Introduced in PTX ISA version 3\\.0\\.</p>\\s*<p>Deprecated "
                  "in PTX ISA version 6\\.0 in favor of",
                  "shfl.sync</span></code>.", None),
        "locate": "re:Introduced in PTX ISA version 3\\.0\\.</p>\\s*<p>Deprecated "
                  "in PTX ISA version 6\\.0 in favor of",
        "covered_by": ("B.3",),
        "note": "★ 这是 `02` 篇 §3「为什么有 `.sync` 后缀」的**唯一**依据："
                "`shfl` 在 PTX ISA **3.0** 引入、**6.0** 弃用并让位给 `shfl.sync`。"
                "⇒ 「用 `shfl` 而不用 `shfl.sync`」不是风格问题，是**语义断点**"
                "（`.sync` 带 `membermask`，无后缀的那个没有）。"
                "★ 消费它的断言是 `01` 的 `B.3`：**扫三个程序里出现的每一个 `shfl`**，"
                "断言**全部**是 `shfl.sync.` 形式——一个弃用形式都不许出现。"
                "★ 引文尾部的 `shfl.sync .` 里那个空格是**去标签的产物**（不是原文的空格）。",
    },
    "P4": {
        "section": "9.7.10.6. Data Movement and Conversion Instructions: shfl.sync",
        "quote": "shfl.sync is undefined if the executing thread is not in the membermask .",
        "must_contain": ("shfl.sync is undefined if the executing thread is not in the",),
        "slice": ("<span class=\"pre\">shfl.sync</span></code> is undefined if the "
                  "executing thread is not in the",
                  "membermask</span></code>.", None),
        "locate": "re:<span class=\"pre\">shfl\\.sync</span></code> is undefined if the "
                  "executing thread is not in the",
        "covered_by": ("C.6",),
        "note": "★ 本条给出的是**「未定义」而不是「某个值」**——ISA 对「执行线程不在"
                "`membermask` 内」只说 undefined。"
                "⇒ 本解释器**不实现 UB**，它**报错**（`PTXError`）；"
                "报错**不等于**复现了 UB（计划 §0.7 E7-5：UB 情形**没有复现**，"
                "正文不得声称覆盖 UB）。"
                "★ 消费它的断言是 `01` 的 `C.6`（`membermask = 0` ⇒ 必须报错）。",
    },

    # -----------------------------------------------------------------------
    # 1.2 barrier 的 warp 粒度（`9.7.15.1`）与那条**不属于它**的 `.aligned` 句
    # -----------------------------------------------------------------------
    "B1": {
        "section": "9.7.15.1. Parallel Synchronization and Communication Instructions: bar , barrier",
        "quote": "Depending on operand b , either specified number of threads "
                 "(in multiple of warp size) or all threads in the CTA participate "
                 "in barrier{.cta} instruction.",
        "must_contain": ("either specified number of threads (in multiple of warp size)",
                         "or all threads in the CTA participate in"),
        "slice": ("Depending on operand <code", "instruction.", None),
        "locate": "Depending on operand <code",
        "covered_by": ("B.3", "B.4"),
        "note": "★ 这条把「参与屏障的线程数」钉成 **warp size 的整数倍**"
                "（明文 `in multiple of warp size`）。"
                "⇒ 它是 `ptxsub` 里「到达计数按 warp 粒度 `+= 32`」的**规范依据**——"
                "**规范说**参与单位是 warp，**本解释器据此把到达量记成 32**（实现）。"
                "★ 消费它的断言在 `02`：`B.3`（到达计数按 warp 粒度）与 `B.4`"
                "（**只剩 1 个活跃 lane 时仍然整 warp 到达**）。"
                "★ 引文里的 `operand b ,` 那个空格是**去标签的产物**。",
    },
    "B2": {
        "section": "9.7.15.1. Parallel Synchronization and Communication Instructions: bar , barrier",
        "quote": "instruction causes executing thread to wait for all non-exited threads "
                 "from its warp and marks warps’ arrival at barrier.",
        "must_contain": ("wait for all non-exited threads from its warp",
                         "marks warps’ arrival at barrier."),
        "slice": ("instruction causes executing thread to wait for all non-exited",
                  "arrival at barrier.", None),
        "locate": "instruction causes executing thread to wait for all non-exited",
        "covered_by": ("B.3", "B.4"),
        "note": "★ 「**marks warps' arrival**」——到达是**按 warp 记的**，不是按线程记的。"
                "这是 `ptxsub._h_bar_sync` 里 `bar_count[bid] += 32` 这一行的原文出处。"
                "★ 原文用的是排版撇号 `’`（`warps’`），`must_contain` 里**逐字保留**。"
                "★ 消费它的断言同样在 `02`：`B.3`（到达计数按 warp 粒度）与 `B.4`"
                "（只剩 1 个活跃 lane 时仍然整 warp 到达）。",
    },
    "B3": {
        "section": "9.7.15.1. Parallel Synchronization and Communication Instructions: bar , barrier",
        "quote": "All threads in warp (except for those have exited) must execute "
                 "barrier{.cta} instruction in convergence.",
        "must_contain": ("All threads in warp (except for those have exited) must execute",
                         "instruction in convergence."),
        "slice": ("All threads in warp (except for those have exited) must execute",
                  "in convergence.", None),
        "locate": "All threads in warp (except for those have exited) must execute",
        "covered_by": ("B.3", "B.4"),
        "note": "★ 消费它的断言在 `02`：`B.3`（到达计数按 warp 粒度）与 `B.4`"
                "（只剩 1 个活跃 lane 时仍然整 warp 到达）。"
                "★ 「**in convergence**」：warp 内所有未退出的线程必须**收敛地**执行 barrier。"
                "⇒ 反过来读就是「发散地执行 barrier 是错用法」——"
                "这正是 `02` 篇 §3 那个「缺 `bar.sync` 」实验的反面注脚："
                "本解释器**允许**它执行（不检查收敛性），只按 warp 粒度记到达。"
                "★ 注意原文是 `except for those have exited`（原文语法如此），"
                "**照抄不改**——把它「修顺」成 `those that have exited` 会让逐字比对失败。",
    },
    "B4": {
        "section": "9.7.15.3. Parallel Synchronization and Communication Instructions: barrier.cluster",
        "quote": "The optional .aligned qualifier indicates that all threads in the warp "
                 "must execute the same barrier.cluster instruction. In conditionally "
                 "executed code, an aligned barrier.cluster instruction should only be "
                 "used if it is known that all threads in the warp evaluate the condition "
                 "identically, otherwise behavior is undefined.",
        "must_contain": ("The optional .aligned qualifier indicates that all threads "
                         "in the warp must execute the same barrier.cluster instruction.",
                         "an aligned barrier.cluster instruction should only be used if it "
                         "is known that all threads in the warp evaluate the condition "
                         "identically, otherwise behavior is undefined."),
        "slice": ("The optional <code class=\"docutils literal notranslate\">"
                  "<span class=\"pre\">.aligned</span></code> qualifier indicates that all "
                  "threads in the warp must execute the same",
                  "otherwise behavior is undefined.", None),
        "locate": "re:qualifier indicates that all threads in the warp must execute the "
                  "same\\s*<code[^>]*>(?:<span[^>]*>)?barrier\\.cluster",
        "covered_by": ("A.2",),
        "note": "★★ **这条是本库唯一一条「引文作用域」的机械载体，而且它的正确用法是"
                "「不要用它」**：这句 `The optional .aligned qualifier …` 出自 "
                "**`9.7.15.3` `barrier.cluster`**，**不属于** `9.7.15.1`（`bar.sync` / `barrier.cta`）。"
                "计划 v2 曾截掉 `.cluster` 并补上句号，**把作用域从 cluster 栅栏扩到了全部 barrier**"
                "（P7）——所以本模块把它**单独收成一条锚点**，并让 `01` 的 `A.2` 断言："
                "① 它的 `section` 前缀**精确等于** `9.7.15.3.`；"
                "② 它**不在** `9.7.15.1` 的三条引文（`B1`/`B2`/`B3`）里；"
                "③ `make anchors` 从原文重新定位它上面最近的标题，仍是 `9.7.15.3.`。"
                "⇒ **改一下它的 `section`、或把它并进 `B1`/`B2`/`B3`，`make run` 立刻失败。**",
    },
}


# ===========================================================================
# 2. ★ `shfl.sync.up` 的判据【未定案】
# ===========================================================================
#
# ★★ 这一节**存在的原因是**：本库的实现与 ISA 文本**不一致**，而且我们**判不出谁对**。
#    把它写成「已定案」是**伪造引文**（计划 §0.7 E9「正文的硬要求」第 4 条）。
#
# ★ 四种可能的写法与后果（逐条对照）：
#     ① 写「规范规定 `minLane`」            ⇒ **伪造引文**（原文写 `maxLane`）；
#     ② 写「二手转述是错的」                ⇒ 同上，而且**规划阶段真的这么写过**；
#     ③ 照抄原文 `maxLane` 并说明本库有意偏离 ⇒ ✅ 本模块采用；
#     ④ 不提这件事                          ⇒ 隐瞒实现与规范的分歧（P8 规范与实现不分）。
#
# ★ 哪些部分是**可机械判定**的（⇒ 进 `must_contain` / 被断言消费）：
#     · 原文写的是 `maxLane`（`P1` 的第 3 行 `must_contain`）；
#     · `minLane` 只出现在定义行与 `.idx` 分支（`P1` 的 `must_contain` + `RAW_COUNTS`）；
#     · ISA 例子的 `cval == 0`（`P2` 的引文里能解析出来）；
#     · `cval = 0` 时两式**逐 lane 恒等**（`02` 的 `C.4` 穷举 32 个 lane，也穷举 segmask）；
#     · `(lane, bval, cval) = (16, 1, 31)` 时两式**给出不同的源 lane**（`02` 的 `C.3`）。
# ★ 哪些**不可判定**（⇒ 只能写成「未定案」，不得写成结论）：
#     · 三种可能里**哪一种是事实**。**本库没有工具链也没有 GPU，无法判定。**

UP_MODE_UNDECIDED = {
    # --- 可机械判定的部分（被断言消费）---
    "isa_literal": "maxLane",          # 原文写的就是这个（P1 / P1b / RAW_COUNTS）
    "implemented": "minLane",          # ptxsub 有意偏离，用这个
    "isa_example_cval": 0x0,           # ISA 的 `.up` 例子用的 clamp 值（见 P2）
    "divergent_case": (16, 1, 31),     # (lane, bval, cval)
    "divergent_j": 15,                 # j = lane - bval
    "divergent_maxLane": 31,           # segmask = 0 → maxLane = cval
    "divergent_minLane": 0,            # segmask = 0 → minLane = lane & 0 = 0
    # --- 不可机械判定的部分（纯列举，不声称任何一条为真）---
    "verdict": "UNDECIDED",
    "possibilities": (
        "(a) ISA 文本在此处有笔误（`.up` 本该写 `minLane`）；",
        "(b) `cval` 对 `.up` 本就该传 0（ISA 自己的例子就是 0），"
        "于是两式等价、**根本不构成矛盾**；",
        "(c) 对 `cval` / `segmask` 的语义理解不对（例如 `maxLane` 在这里的含义"
        "与 `binomial` 里的「上界」不是一回事）。",
    ),
    "covered_by": ("A.4", "C.3", "C.4"),
    "note": "★★ **本库不判定是哪一种。** 正文（`02-warp-level-execution.md` §4.1）"
            "必须逐字引 `P1` 的原文、写明本实现是有意偏离、并列出三种可能。"
            "★ 本条的 `covered_by` 只覆盖上面「可机械判定的部分」——"
            "`possibilities` 是**纯列举的散文**，它**不是**任何断言的载体，"
            "`make mutate` 会把它的每个叶子如实地报成**漏网**（本库不粉饰这一点）。",
}


# ===========================================================================
# 3. ★ 本解释器覆盖的 PTX 子集（一张**闭合**的表）
# ===========================================================================
#
# ★★ 表**必须闭合**：表内 = 实现，表外 = **一律 `PTXError`，绝不静默跳过**
#    （计划 Chunk 3「代码形态」；判据落点 `ptxsub.Sim.validate()`）。
#
# ★★ `supported` **不是手写的第二份清单**：`01_virtual_isa.py` 的 `A.1` 断言
#    `set(PTX_SUBSET['supported']) == ptxsub.SUPPORTED_OPS`（**双向集合相等**），
#    而 `SUPPORTED_OPS` 是 `ptxsub` 里两张分派表（`_SCALAR` / `_WARP`）的键的并集。
#    ⇒ 「加了实现没进表」与「进了表没实现」都**不可能**通过 `make run`。

PTX_SUBSET = {
    # --- 表内（= 实现）-----------------------------------------------------
    "supported": (
        # 标量：数据搬运与算术
        "mov.u32", "mov.b32", "mov.s32", "mov.f32",
        "add.u32", "add.b32", "add.s32", "add.f32",
        "sub.u32", "mul.lo.u32", "mad.lo.s32",
        # 标量：位运算
        "and.b32", "and.u32", "or.b32", "or.u32", "xor.b32", "xor.u32",
        "shl.b32", "shr.u32", "shr.b32",
        # 谓词
        "setp.lt.s32", "setp.le.s32", "setp.gt.s32", "setp.ge.s32",
        "setp.eq.s32", "setp.ne.s32",
        "setp.lt.u32", "setp.le.u32", "setp.gt.u32", "setp.ge.u32",
        "setp.eq.u32", "setp.ne.u32",
        # 共享内存（**只有** shared 一个地址空间）
        "ld.shared.u32", "ld.shared.b32", "ld.shared.s32", "ld.shared.f32",
        "st.shared.u32", "st.shared.b32", "st.shared.f32",
        # warp 级
        "bra", "ret", "bar.sync",
        "shfl.sync.bfly.b32", "shfl.sync.idx.b32",
        "shfl.sync.up.b32", "shfl.sync.down.b32",
    ),
    # --- 表外（PTX 有、本解释器**没有**）----------------------------------
    # ★★ 这 14 条**不是**「我猜 PTX 有」：`make anchors` 会逐条断言
    #    **它们每个字面量都真的出现在刚下载的 ISA 原文里**，并把实际计数打印出来。
    "unsupported_in_isa": (
        "ld.global.u32",             # 全局内存访存：本子集**只有** shared
        "ld.param.u32",              # kernel 参数：本子集靠宿主机预置寄存器
        "ldmatrix.sync.aligned",     # Tensor Core 的矩阵装填
        "cp.async.ca.shared.global", # 异步拷贝
        "atom.global.add",           # 原子操作
        "red.shared.add",            # 归约访存
        "mbarrier.init.shared",      # 新版屏障对象
        "bar.arrive",                # 只到达不等待
        "bar.warp.sync",             # warp 级屏障
        "bar.red.popc",              # 带归约的屏障
        "vote.sync.ballot.b32",      # warp 投票
        "fma.rn.f32",                # 融合乘加
        "prmt.b32",                  # 字节置换
        "popc.b32",                  # 位计数
    ),
    "covered_by": ("A.1", "B.2", "C.3", "C.4"),
    "note": "★★ **表外的含义是「本解释器不支持」，不是「PTX 不合法」**——"
            "`unsupported_in_isa` 里的 14 条**全是真实 PTX**，"
            "本解释器遇到它们一律**报错**（`01` 的 `C.3`/`C.4` 对**每一条**都试一遍）。"
            "★ 反过来说：**表内 ≠ PTX 的全部**。本子集只解释**本文档手写的片段**"
            "（不得把没实现的写成已实现）。"
            "★ 这 14 条里**只有三条**（`fma.rn.f32` / `prmt.b32` / `popc.b32`）"
            "在算术上与表内指令重叠，其余是**另一整类能力**（全局内存 / 参数 / "
            "Tensor Core / 异步拷贝 / 原子 / 屏障对象 / 投票）——"
            "本解释器对这几类**一条都没有**，这不是遗漏，是**边界**。",
}


# ===========================================================================
# 4. ★ 共享内存 bank 冲突：两套 lane 模式（本域最值钱的一条反直觉量化结论）
# ===========================================================================
#
# ★★ 为什么**模式定义必须连同数字一起写**：「**列访问**」不是一个良定义的词——
#    同一个 32×32 tile，warp 的 32 个 lane 可以沿列铺开（模式 B），
#    也可以一条 warp 横跨 8 行 × 4 列（模式 A）。**同一份数据、同一个 stride，
#    两种模式的冲突路数不同**（本域 P9 的又一实例：同一个「列」，两个 lane 模式）。
#
# ★ 数字（`BANK_CONFLICT_TABLE`）是**推导值**：在 `03_smem_and_sass.py` 里当场算出
#   并断言（三种独立算法：解释器的实际 `ld.shared` 字节地址、numpy 向量化、
#   以及模式 B 的闭式 `gcd(行距, 32)`）——**不在这里硬编码**。
#    本模块记的只是**表的形状**（哪两个模式、哪五个 stride）。这里放数字是为了让
#    「正文引用的表」有一个**可对照的常量**；它由 `03` 的 `B.1`/`B.2` 逐格断言。

BANK_CONFLICT = {
    "modes": {
        "A": {
            "rows": 8, "cols": 4,
            "lane_row": "t % 8", "lane_col": "t // 8",
            "desc": "一个 warp 横跨 tile 的 **8 行 × 4 列**（R = 8）",
        },
        "B": {
            "rows": 32, "cols": 1,
            "lane_row": "t % 32", "lane_col": "t // 32",
            "desc": "一个 warp 横跨 tile 的 **32 行 × 1 列**（R = 32，教科书 32×32 场景）",
        },
    },
    "strides": (8, 9, 12, 32, 33),
    "table": {
        # 行距 → {'A': 模式 A 的最大冲突路数, 'B': 模式 B 的最大冲突路数}
        8:  {"A": 2, "B": 8},
        9:  {"A": 2, "B": 1},     # ★ +1 padding：模式 A **无效**、模式 B 有效
        12: {"A": 1, "B": 4},     # ★ 行距 12：模式 A 有效
        32: {"A": 8, "B": 32},
        33: {"A": 4, "B": 1},
    },
    "coverage_mode_A": {9: 29, 12: 32},
    "reloaded_banks_mode_A": {9: (0, 1, 2), 12: ()},
    "existing_doc_rule_scope": "B",
    "existing_doc_rule": "gcd(行距, 32)",
    "covered_by": ("A.2", "A.3", "A.4", "B.1", "B.2", "B.3", "B.4", "B.5",
                   "C.1", "C.2", "C.4", "C.5"),
    "note": "★★ **`existing_doc_rule_scope = 'B'` 是本模块最容易被误读的一条，**"
            "**它是一条「不是订正」的声明**："
            "`20-gpu-architecture/memory-system/03-shared-memory-bank-conflict.md`"
            "（第 358 行）**自己已经写明**它的结论带一个前提"
            "「`blockDim.x = 32`，也就是一个 warp 恰好铺满 tile 的一整行」——"
            "**那正是模式 B**。该篇的 `gcd(行距, 32)` 规则在模式 B 下**完全正确**，"
            "本模块用 `03` 的 `A.4`（穷举 stride 对照闭式 `gcd`）把它**复核了一遍**。"
            "⇒ 本主题的贡献是**把规则推广到「warp 不只铺满一行」的情形**（模式 A），"
            "**不是对它做订正**（`CONTRIBUTING` 第 56 行："
            "**不要在不构成冲突的地方硬造冲突**）。"
            "★ 正文写法固定为「**该篇讲 32 行 tile，本篇讲 warp 跨多行的情形**」并回链。"
            "★ 表里**没有**任何一条数字是「模式 A 下 `+1` padding 也有效」——"
            "这正是本篇的反直觉量化结论。",
}

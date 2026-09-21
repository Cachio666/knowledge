# -*- coding: utf-8 -*-
r"""30-gpu-programming/triton 的外部事实常量源。

本模块是 `CONTRIBUTING` §2 约定的**共享事实源**（`specs.py`）：四个编号程序从它读常量，
推导过程可复核，**常量本身不可自证**（域 README「代码约定」一节）。

★★ 本模块**刻意不含散文**（`CONTRIBUTING` 第 220–222 行 / 计划 §0.6 第 3 条）。
   判据是「**把这条常量改错，程序会报错吗？**」。
   ⇒ 本模块里**每一个值都被四个编号程序之一断言**：

     · `ANCHORS[i]['quote']`     —— 被 `must_contain` 的**逐字子串**钉住（改一个字即失败），
                                    并被 `03_layout.py --anchors-net` 的**在线再抽取**钉住；
     · `ANCHORS[i]['covered_by']` —— 被 `01_block_model.py` 的**并集 == 全集**与
                                    **标签真实存在**两条检查钉住（用 `ast` 读另外三个程序）；
     · `LOAD_QUOTES` / `WORD_FREQ` —— 被 02 / 03 的断言钉住（`WORD_FREQ` 的两个页面
                                    在离线断言里判「是 0 还是非 0」，在线再抽取里判**逐数相等**）；
     · `MODEL_OWN_RULES` / `MODEL_SIMPLIFICATIONS` —— **可判定的那一半**被钉住：条数、两两不同、
                                         每条 `covered_by` 的标签**真实存在**（`01:C.8`）、
                                         以及「没有一条是官方引文的字面量」（`03:A.7`）。
                                         ★ `rule` / `basis` / `item` / `consequence` 这些**散文正文**
                                         **不是**行为判据的载体 —— 它们如实出现在 `make mutate`
                                         的**漏网清单**里（`MODEL_OWN_RULES[*].rule` 5 条等），本模块
                                         不声称它们「改一个字程序就会失败」。

★ 来源等级（定义在 `CONTRIBUTING.md` §2「外部来源分级 A–E」）：

  · **A 级：厂商官方规范/官方文档原文** —— `triton-lang.org` 的**官方文档站点**，逐字引文。
    本模块的 7 条语义锚点（`ANCHORS` 5 条 + `LOAD_QUOTES` 2 条）全部是 A 级。
  · **B 级：某一实现的官方材料** —— 本模块**不含** B 级条目：`triton` 包在本机**不存在**
    （macOS 无 wheel，逐个解释器核实过），所以**没有「真实的 Triton 跑出来的东西」可引**。
  · **E 级：本库推导** —— 本模块**不含**推导值。所有模型值（相位 ops、smem 字节、barrier 数、
    交叉点 K*=2 / K*=64）都在编号程序内部**当场算出并断言**，**不搬进 `specs.py`**
    （`CONTRIBUTING` 第 48 行：「只放从来源抄来的数，**不放推导值**」）。

★★ `# 来源：` 的机器可核形式 = 每条锚点的 `page` + `section` + `quote` 三个字段 + 在线再抽取。
   `page` 是**逐字的 URL 路径**，`section` 是**该引文所在页面里最近的上位标题**。
   本模块**刻意不写「Triton 文档说」这种不可核的形式**（P7 的判据）。

★★ 抽取口径**必须写下来**（计划 §0.7 F4 的缺陷：口径没写，别人复现不出来）：

   ```
   raw = curl -sSL <url>                     # 原始字节（HTTP 200；只有 /main/ 一个版本路径）
   text = strip(<script>|<style>)            # 先去掉脚本与样式
        → strip(<...>)                       # 再去掉全部标签
        → html.unescape(...)                 # 反转义实体
        → re.sub(r'\s+', ' ', ...)           # 空白折叠成单个空格
   ```
   ★ 本模块的 `stripped_len` 与三个词频列**都用这个口径复算过**，
     并由 `03_layout.py --anchors-net` 在线重抽取时**逐数比对**。
   ★ 词频有**大小写敏感**与**不敏感**两个口径，本模块**两个都记**、不合并成一个数 ——
     因为计划 §0.7 F4 记的 `warp=73` 只有**不敏感**口径下成立（敏感口径是 71/72）。
     这是一个**口径分歧**，不是笔误：写成一个数会让复现的人得出「对不上」的结论。

   ★★ **去标签会留下空格**（抽取口径的**已知产物**，不是页面内容）：
     行内 `<code>x</code>` 被 `strip(tags)` 换成空格后，`x ,` 这种「标点前多一个空格」会出现。
     ⇒ `must_contain` 的字面量**不得跨标签边界**（见 `LOAD_QUOTES["L1"]` 的注释，
     那里第一版就因为跨了边界被 `make anchors` 判成「逐字不命中」）。
     ★ 这一条**只影响引文匹配**，不影响词频列与 `stripped_len`（后者就是按这个口径数的）。

★ `NOT_FIRST_HAND`：**探针自报「未一手复核」的东西**（计划 §0.7 F11）**只出现在这里**，
  **正文与常量名里都不出现**，`01_block_model.py` 的 `C.11` 还会**扫全部源码**确认它们没被引用
  （`CONTRIBUTING` 第 65 行：不得「既保留数字又记 UNSOURCED」）。
"""

# ===========================================================================
# 0. 测量条件
# ===========================================================================
#
# ★ 本模块**不含任何性能数字**——本机**无 GPU**，`triton` 包**全机器不存在**。
#   域 README 明写：本主题**没有任何实测数字，只有模型值**（计划 §0.7 F10-2）。
#
# ★ 解释器与 numpy 版本在**运行时**由编号程序打印，不在这里硬编码——它们随机器变。
#   见各程序的 `print_env()`。

BASE_URL = "https://triton-lang.org/main/"
"""官方文档站点的根。★ 只有 `/main/` 一个版本路径（`/stable/` 等 404，已核实）。"""

GLUON_PAGE = "getting-started/tutorials/gluon/layouts.html"
"""★ **本主题唯一能引「layout 转换昂贵」与「归约跨 warp」的页面**（计划 §0.7 F4）。

  计划 §0.7 F4 的表格（控制者已独立验证，本模块**重算过**）：

  | 页面 | `coalesc` | `Layout conversions` | `warp` |
  |---|---:|---:|---:|
  | `tutorials/01-vector-add.html` | 0 | 0 | 0 |
  | `tutorials/gluon/layouts.html` | 5 | 1 | 73（不敏感口径） |

  ⇒ **主 tutorial 里 grep 不到 `coalesc`，也完全没有 layout 转换的说法**——
    要引这两条机制，**位置没有选择余地**。
"""

VECTOR_ADD_PAGE = "getting-started/tutorials/01-vector-add.html"
"""主 tutorial 的第一篇。★ 它在本模块里的作用**只是「对照组」**：证明主 tutorial 里
   `coalesc` / `Layout conversions` / `warp` **都是 0 次**。"""

LOAD_PAGE = "python-api/generated/triton.language.load.html"
"""`tl.load` 的 API 页。★ 它是 `other=None` 语义（「未定义」）**唯一**的官方出处。"""


# ===========================================================================
# 1. ★ 语义锚点（A 级：Triton 官方文档原文）
# ===========================================================================
#
# 字段说明：
#   page         该引文所在的 URL 路径（相对 `BASE_URL`）
#   section      该引文所在页面里**最近的上位标题**（人工定位，`--anchors-net` 会打印它抓自哪里）
#   quote        抽取器输出的**逐字**引文（单行化后用单个空格连接）
#   must_contain 必须逐字出现的短字面量元组；改一个字，程序就报错
#   covered_by   本锚点**被哪些断言支撑**（标签字符串元组，如 ("C.3", "A.8")）
#   note         会被打印的解释。它**不单独充当依据**，只说明该条的作用域限制。
#
# ★★ `note` 是**索引**，不是**唯一载体**（沿用 `cuda/code/specs.py` 的裁定）：
#   一个只在 `note` 里出现的论断是**不可证伪**的（P5）。所以定了三条规矩：
#     ① 每个锚点的 `covered_by` **非空**；
#     ② 里面每个标签都必须**真实存在**于某个编号程序的断言清单里（不是编的）；
#     ③ `note` 里凡属**承重论断**的句子，必须能被 `covered_by` 指向的断言机械判定。
#   ★ 校验落点：`01_block_model.py` 的 **C.7**（并集 == 全集、两两不交）与 **C.8**
#     （`covered_by` 里每个标签都真实存在，用 `ast` 从源码字面量提取）。
#     标签由程序从**断言消息的字面量**里提取（`LABEL_RE`），不是手写的清单。

ANCHORS = {

    # -----------------------------------------------------------------------
    # 1.1 Gluon「Tensor Layouts」——layout 转换的代价
    #     ⇒ 只被 03_layout.py 消费
    # -----------------------------------------------------------------------
    "G1": {
        "page": GLUON_PAGE,
        "section": "Tensor Layouts",
        "quote": "Layout conversions are potentially expensive operations, because they often "
                 "result in data movement across threads and warps. Data movement across warps "
                 "also requires using shared memory, which is a precious resource on the GPU.",
        "must_contain": (
            "Layout conversions are potentially expensive operations",
            "data movement across threads and warps",
            "Data movement across warps also requires using shared memory",
            "which is a precious resource on the GPU",
        ),
        "note": "★★ 这是「**layout 转换可能很贵**」的**唯一**官方出处，而且它**在 Gluon 页上**——"
                "主 tutorial（01-vector-add）里 `Layout conversions` 出现 **0** 次（见 `WORD_FREQ`）。"
                "它把「为什么贵」拆成两级：① 跨线程/跨 warp 的数据移动；② 跨 warp 必须用"
                "**共享内存**（原文：`a precious resource`）。第二级是本模型里 `smem_bytes` "
                "那一列的锚点。"
                "★ 作用域限制（P7）：本句讲的是 **Gluon 层**的 `convert_layout` 的代价，"
                "**不是**「Triton 编译器一定会为某个 kernel 插转换」——后者本机无从验证（无 GPU、无 triton）。"
                "★ 落点：`03:B.6`（一次转换 = 2×numel×esize 字节 + 2 barrier，"
                "**口径是本模型自定的**，本句只锚定「走共享内存」这个**机制**）。",
        "covered_by": ("03:B.6", "03:B.9", "03:C.2"),
    },
    "G2": {
        "page": GLUON_PAGE,
        "section": "Tensor Layouts",
        "quote": "The compiler will generate butterfly shuffles to reduce within each warp, "
                 "then pick a leader warp to reduce the remaining 4 values per row through "
                 "shared memory.",
        "must_contain": (
            "butterfly shuffles to reduce within each warp",
            "pick a leader warp to reduce the remaining 4 values per row",
            "through shared memory",
        ),
        "note": "★★ 归约**三相分解**（线程内 → warp 内蝶形 → 跨 warp 经共享内存）的**唯一**官方出处。"
                "本主题的 `ReducePlan` 三个相位**逐条对应这句话**："
                "`butterfly shuffles within each warp` → 相位 2；`leader warp ... through shared memory` → 相位 3。"
                "★ 原句里的 **4** 是**那个例子**的位数（`num_warps=4`），不是常数——"
                "把它读成「跨 warp 相位总是 4 个值」是**作用域误读**（P7）；"
                "本模型的 `wpc` 是布局参数，可以是 2/4/8（见 `04` 的 18 组表）。"
                "★ 落点：`04:B.1`（跨 warp 相位**恰好**出现在 `wpc[axis] > 1` 的组合上）。"
                "★ 本句**没有**说「块大小 > warp 就要跨 warp」——那是本模型的口径，"
                "不是这句话的推论；反例见 `04:B.3b`（`[B=64, W=2]` 有、`[B=1024, W=1]` 没有）。",
        "covered_by": ("04:A.1", "04:B.1", "04:B.2"),
    },
    "G3": {
        "page": GLUON_PAGE,
        "section": "Tensor Layouts",
        "quote": "When converting between layouts you know are equivalent, or at most only "
                 "require reordering registers within a thread (which is free), you can use "
                 "gl.convert_layout(x, layout, assert_trivial=True) to ensure this.",
        "must_contain": (
            "at most only require reordering registers within a thread (which is free)",
            "gl.convert_layout(x, layout, assert_trivial=True)",
        ),
        "note": "★★ 这条是**代价模型的非退化性锚点**：官方**明说**存在**免费**的转换。"
                "⇒ 一个把「转换」一律定价为昂贵的模型是**错的**，而不是保守的。"
                "本模型的免费判据是「逐元素属主相同」——**这是本库自定的判据**，"
                "官方那句只说「只重排线程内寄存器」这一个**充分条件**（括号里就是 `which is free`）。"
                "★ 落点：`03:B.8`（`(64,2)` 那一对被判为 0 字节 / 0 barrier）与 `03:B.9`"
                "（`L -> Lrow` 那一对**不**免费）。`03:C.2`/`03:C.2b` 把免费判据往两个方向改错，"
                "两个方向的断言**都会失配**——所以它有判别力。",
        "covered_by": ("03:B.8", "03:C.2", "03:C.2b"),
    },
    "G4": {
        "page": GLUON_PAGE,
        "section": "Tensor Layouts",
        "quote": "Performance craters to 0.774 TB/s. Because the inner dimension is no longer "
                 "contiguous, we get no coalescing.",
        "must_contain": (
            "Performance craters to 0.774 TB/s.",
            "Because the inner dimension is no longer contiguous, we get no coalescing.",
        ),
        "note": "★ 这是「**非合并访存**」的官方**实测量**（`0.774 TB/s`）——但它是**官方在他们自己的"
                "机器上**测的，**不是本机实测**，本仓库也**没有**复核它的测量口径"
                "（本机无 GPU，`triton` 不存在）。⇒ 本模块只把它当**引用值**（`CONTRIBUTING` 第 63 行）。"
                "★ 本主题的 `contiguous_along(d)` 判据（`minor_dim == d`）锚的就是后半句"
                "「inner dimension is no longer contiguous ⇒ no coalescing」。"
                "★ 落点：`03:B.7b`（把转过置的结果写回一个要求合并访存的位置要付一次转换）"
                "与 `03:C.1c`（不许插转换时**拒绝**，即非合并访存不能被静默放过）。"
                "★ **不得**用它支撑「本模型的转换代价」——那是本库自定的定价公式，与这个数无关。",
        "covered_by": ("03:B.7b", "03:C.1c"),
    },
    "G5": {
        "page": GLUON_PAGE,
        "section": "Tensor Layouts",
        "quote": "Unlike global memory accesses, the compiler does a good job of generating "
                 "efficient reductions, scans, etc. regardless of the input layout, thus it is "
                 "typically more expensive to convert_layout to an efficient layout and then "
                 "perform the reeduction.",
        "must_contain": (
            "the compiler does a good job of generating efficient reductions",
            "regardless of the input layout",
            "typically more expensive to convert_layout to an efficient layout and then perform "
            "the reeduction",
        ),
        "note": "★★★ **末词 `reeduction` 是官方页面上的拼写错误**（原文如此）——"
                "本模块**逐字保留**，`must_contain` 也钉住这个拼法。**引用时不得「修好」它**："
                "把它改成 `reduction` 会让 `03:A.5` 与在线再抽取**同时**失败。"
                "★ 这条引文本身是一条**负面结论的官方版**："
                "「先把 layout 转成高效的再做归约」通常**更贵**，因为编译器对归约本来就不太挑布局。"
                "它与本主题 `F8` 的负面结论（`trans` 免费 ≠ 归约变便宜）**方向一致但不同源**："
                "本库那条是从**代价模型的两个数**推出来的，这条是**官方的一句话**。"
                "★ 落点：`03:B.5`（多写一个 `trans` 一点没省：`ops/smem/barriers` 逐项相等）。",
        "covered_by": ("03:A.5", "03:B.5"),
    },
}


# ===========================================================================
# 2. `tl.load` 的 `mask` / `other` 语义（A 级：API 页原文）
# ===========================================================================
#
# ★ 这两句是 `other=None` ⇒「未定义」的**唯一**官方出处。它们**不是**「other 默认为 0」——
#   把这句读成 0 是**本主题最重要的一次误读**，02 的 `C.4` 就是它的可证伪形式。

LOAD_QUOTES = {
    "L1": {
        "page": LOAD_PAGE,
        "section": "triton.language.load（Parameters 列表）",
        "quote": "other (Block, optional) – if mask[idx] is false, return other[idx]. "
                 "If other is None, the masked-out value is undefined.",
        # ★★ `must_contain` 刻意**拆成三段**、且每段都**不跨标签边界**：官方页面里
        #    行内 `<code>other</code>` 与 `<code>mask[idx]</code>` 被去掉标签后会**各留下一个空格**，
        #    所以抽取出来的文本里是 `If other is None , the …`（**逗号前多一个空格**）。
        #    第一版把整句当一条 `must_contain`，`make anchors` 当场判它「逐字不命中」——
        #    这不是页面变了，是**我的字面量跨过了标签边界**。拆开即两边都真。
        "must_contain": ("return other[idx]", "If other is None",
                         "the masked-out value is undefined."),
        "note": "★★ 官方语义是「**未定义**」，**不是 0**。"
                "本模型把「undefined」具体化成毒值 `POISON = 1.0e30` —— "
                "**「未定义」必须有个具体的坏值，否则它没法被断言**"
                "（这一步是**本模型自定的**，官方只说「未定义」）。"
                "★ 落点：`02:C.3`：不给 `other` 时，24 个越界位置的毒值"
                "**真的一路流进归约结果**（`max` 从 `149.75` 变成 `1.000000e+30`）。",
        "covered_by": ("02:C.3",),
    },
    "L2": {
        "page": LOAD_PAGE,
        "section": "triton.language.load（Parameters 列表）",
        "quote": "mask (Block of triton.int1, optional) – if mask[idx] is false, "
                 "do not load the data at address pointer[idx]",
        "must_contain": ("if mask[idx] is false", "do not load the data at address pointer[idx]"),
        "note": "★★ 掩码为假时**不读那个地址**——这不是「读回来再丢掉」。"
                "它是 `02:B.2` 的锚点：掩码把 **24 次访存**从「读」降级成「取常量」。"
                "★ 但**注意它的作用域**（P7）：本句说的是**不做这次访存**，"
                "**没有**说「掩码全真时越界会被检出」——后者是**本模型的加强**"
                "（`minitriton.load(strict=True)`），真实 GPU 上那是一次未定义行为。"
                "★ 落点：`02:B.2` 与 `02:C.1`/`02:C.2`（不写掩码 / 写宽 ⇒ **抛出**），"
                "`02:C.5`（**写窄 ⇒ 不抛出**，静默算成 other）与 `02:C.6`（偏差同阶）。",
        "covered_by": ("02:B.2", "02:C.1", "02:C.5"),
    },
}


# ===========================================================================
# 3. ★ F4 的词频表（页面级事实）
# ===========================================================================
#
# ★ 三个词频列的**口径**见模块头（去标记文本 + 空白折叠）。
#   `_cs` = 大小写敏感，`_ci` = 大小写不敏感。**两个都记**，不合并。
#
# ★ 本表**在线可复算**：`03_layout.py --anchors-net` 会重新下载两个页面、
#   用**同一口径**重算，并**逐数比对**本表；对不上就以非零码退出。
#   离线（`make run`）只能判「是 0 还是非 0」——那一档**不是**本表的可证伪形式，
#   它判的是**结论**（主 tutorial 里没有这三样东西）。

WORD_FREQ = {
    VECTOR_ADD_PAGE: {
        "coalesc_cs": 0,
        "layout_conversions_cs": 0,
        "warp_cs": 0,
        "warp_ci": 0,
        "stripped_len": 7315,
        "role": "对照组：主 tutorial",
    },
    GLUON_PAGE: {
        "coalesc_cs": 5,
        "layout_conversions_cs": 1,
        "warp_cs": 71,
        "warp_ci": 73,
        "stripped_len": 34789,
        "role": "唯一出处：Gluon「Tensor Layouts」",
    },
}

# ★ 词频**抽取口径**的机器可核形式：这三条字面量就是上面两页的口径。
#   `--anchors-net` 用它们重建文本；`03:A.6` 的离线断言用它们检查「本模型没有偷偷换口径」。
STRIP_PIPELINE = ("strip(script|style)", "strip(tags)", "html.unescape", r"collapse(\s+)")


# ===========================================================================
# 4. ★ 本模型**自造**的规则（不是 Triton 的规定）——5 条
# ===========================================================================
#
# ★ 计划 §0.7 F10-1：**必须明写「本模型的规则是自造的」**，并列出**不**声称是 Triton 规定的 5 条。
#   ⇒ 它们是**常量**（不是散文），并且被 `03_layout.py` 的打印块**双向绑定**：
#     把这里的任何一个字符串改掉（或从打印块里删掉），`03:A.7` 立刻失败。
#
# ★ `basis` 字段写的是「本模型为什么需要它」——它**不是**来源，**不要**读成引用。

MODEL_OWN_RULES = {
    "R1": {
        "rule": "摊维规则：coalesced_layout 的贪心（先把 32 个 lane 摊到 order[0]，再 order[1] …）",
        "basis": "本模型自定。★ 它在 shape=(128,128)/num_warps=4/order=[0,1] 上"
                 "**逐参数复现**官方给出的 BlockedLayout([1,128],[32,1],[4,1],[0,1])，"
                 "但「复现了同一个结果」不等于「同一个规则」——本模型不声称这个贪心是 Triton 的算法。",
        "covered_by": ("03:A.7", "03:B.1"),
    },
    "R2": {
        "rule": "属主算法：owner(i) 的算法与「精确切分」约束（spt×tpw×wpc == shape）",
        "basis": "本模型自定，而且是一条**简化**：真实 Triton 的 layout **可以覆盖不足**"
                 "（官方第一个例子 BlockedLayout([1,1],[1,32],[1,4],[1,0]) 在 128×128 上只覆盖 1×128，"
                 "要沿第 0 维复制）——本模型**拒绝**这种布局。",
        "covered_by": ("03:A.7", "03:A.8", "03:C.5"),
    },
    "R3": {
        "rule": "转换定价公式：不走共享内存 0 字节；走则 2 × numel × esize 字节 + 2 个 barrier",
        "basis": "本模型自定。官方只锚定「走共享内存」这个**机制**（G1），没有给公式。",
        "covered_by": ("03:A.7", "03:B.9"),
    },
    "R4": {
        "rule": "三相归约的 ops 计数公式（线程内 / 蝶形的 log2(tpw) 步 / 跨 warp 的 wpc×positions）",
        "basis": "本模型自定。官方只锚定**相位的存在**（G2），没有给计数公式。",
        "covered_by": ("03:A.7", "04:A.1b"),
    },
    "R5": {
        "rule": "交叉点 K*=2（barrier 口径）与 K*=64（字节口径）这两个数",
        "basis": "本模型自定——它们是**上面两条自造规则**在 K 次归约上的算术结果，"
                 "换一个定价公式就换一组交叉点。",
        "covered_by": ("03:A.7", "03:B.10"),
    },
}


# ===========================================================================
# 5. ★ 已知的简化与缺口（**写进代码，不写进散文**）
# ===========================================================================
#
# ★ 计划 §0.7 F10-3：`minitriton.py` 有一个**已记录的简化**——
#   模型要求 `spt×tpw×wpc == shape`（精确切分、无复制），而真实 Triton 的 layout 可以覆盖不足。
#   **这不改变归约的结论，但必须写明**，否则「本模型复现了官方布局」会被读成比实际更强的声明。
#
# ★ `stricter` 字段写的是本模型**加强**了的地方（加强会让「忘了加掩码」可证伪，
#   但也意味着**模型的行为比真实系统更严格**——不得反过来把加强读成「真实系统也这样」）。

MODEL_SIMPLIFICATIONS = {
    "S1": {
        "item": "精确切分：模型要求 spt×tpw×wpc == shape（无复制、无 padding）",
        "consequence": "真实 Triton 的 layout 允许**覆盖不足**（要沿某维复制）⇒ "
                       "本模型覆盖的是真实 layout 空间的一个**子集**。"
                       "本主题关于**归约相位存在与否**的结论不依赖这一点（它只依赖 wpc[axis]）。",
        "covered_by": ("03:A.8", "03:C.5"),
    },
    "S2": {
        "item": "越界读在 masked 为真时**抛错**（strict=True），而不是未定义行为",
        "consequence": "**这是加强**，不是更真实。真实 GPU 上掩码写错是一次未定义行为；"
                       "本模型把它变成可证伪的异常。⇒ 它能抓「**写宽**」，"
                       "但对「**写窄**」完全无效（写窄不越界）。",
        "covered_by": ("02:C.1", "02:C.5"),
    },
    "S3": {
        "item": "`other=None` 被具体化为毒值 1.0e30",
        "consequence": "官方只说「未定义」。本模型必须给「未定义」一个**具体的坏值**，"
                       "否则它没法被断言。毒值是**本模型的选择**，不是 Triton 的输出。",
        "covered_by": ("02:C.3",),
    },
}


# ===========================================================================
# 6. ★ 不得引用（计划 §0.7 F11）——探针自报「未一手复核」
# ===========================================================================
#
# ★ `CONTRIBUTING` 第 65 行：**不得「既保留数字又记 UNSOURCED」**。
#   ⇒ 这些项**只出现在这里**，作为一张**禁用清单**；
#      `01_block_model.py` 的 `C.11` 扫**本目录全部 .py**，确认它们**没有被引用**。
#   ★ 注意：本常量**自身**包含这些字符串（它就是那张清单）——所以扫的时候要**排除本文件**。

NOT_FIRST_HAND = {
    "N1": "lib/Conversion/TritonGPUToLLVM/ReduceOpToLLVM.cpp 的行号",
    "N2": "include/triton/Dialect/Triton/IR/TritonOps.td 的行号",
    "N3": "include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td 的行号",
    "N4": "lib/Dialect/TritonGPU/Transforms/Coalesce.cpp 的行号",
    "N5": "最新 release 的版本号（探针报 v3.8.0）与 main HEAD 的 commit hash",
    "N6": "MAPL 2019 论文（Triton 论文）的词频统计：layout / warp / shuffle 均为 0（**未复核**）",
    "N7": "60-compilers 计划里钉的 triton-lang/triton v3.6.0（**知识库自己的记录，不是本次核实的**）",
}
"""★ **禁用清单**。每一条都是探针**自报未一手复核**的东西（计划 §0.7 F11）。

   ⇒ 「若 N6 属实，则『Triton 论文里讲了 layout』这类说法要撤」——
     本仓库**没有**复核 MAPL 2019 的 PDF，所以本主题**不使用**任何论文词频结论。
   ⇒ 「N7 是知识库自己的记录」——所以本主题**不写任何 Triton 版本号**。
"""


# ===========================================================================
# 7. 非编号模块的角色（供文档与 `mutate.py` 引用）
# ===========================================================================
#
# ★ 这一节**不是**外部事实，是**本仓库自己的结构声明**——但它是**可证伪**的：
#   `01_block_model.py` 的 `C.5` 逐条检查「元检验的锚点在 `minitriton.py` 里恰好出现 1 次」，
#   锚点漂了就报错（否则元检验会**静默变成空操作**）。

META_DEFECTS = {
    "D1": ("掩码被忽略（读了越界位置）", "02_masking.py"),
    "D2": ("所有布局转换都判成免费", "03_layout.py"),
    "D3": ("跨 warp 相位被抹掉（wpc>1 也不认）", "03_layout.py / 04_warp_reduce.py"),
    "D4": ("三相归约的跨 warp 相位写错（只取 warp 0 的局部和）", "04_warp_reduce.py"),
    "D5": ("coalesced_layout 把 lane 摊到错的维上", "03_layout.py"),
}
"""★ 元检验注入的 5 种缺陷（`make falsify`）与它们**应该**打中的程序。
   实测：**5 种全部被抓住**，且**不打补丁时四个程序全过**。
   ⇒ 「解释器与被测对象是同一个东西」这个质疑，被这 5 条反证正面回答（P12 的正面回答）。
"""


FORBIDDEN_TOKENS = (
    "ReduceOpToLLVM", "TritonOps.td", "TritonGPUOps.td", "Coalesce.cpp",
    "v3.8.0", "v3.6.0", "MAPL",
)
"""★ `NOT_FIRST_HAND` 那张**禁用清单**的**机器可核形式**：一组短字面量，
   `01_block_model.py` 的 `C.11` 用它扫**除本文件以外**的全部 .py，命中即失败。

   ★ 为什么放在 `specs.py` 而不是检查器里：**检查器自己也不能含这些字面量**
     （否则它扫自己就会命中）—— 这是一个**自指**陷阱，第一版就是这么失败的。
     放在这里、并把 `specs.py` 从扫描范围里排除，两个条件同时成立。
   ★ 改动这里的任何一项，`C.6` 的行为立刻变化 —— 所以它**不是散文**。
"""

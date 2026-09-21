# -*- coding: utf-8 -*-
"""30-gpu-programming/cuda 的外部事实常量源。

本模块是 CONTRIBUTING §2 约定的**共享事实源**（`specs.py`）：三个编号程序从它读常量，
推导过程可复核，**常量本身不可自证**（域 README「代码约定」一节）。

★★ 本模块**刻意不含散文**（`CONTRIBUTING` 第 220–222 行 / 计划 §0.6 第 3 条）。
   判据是「**把这条常量改错，程序会报错吗？**」。
   ⇒ 本模块里**每一个值都被三个编号程序之一断言**，包括那些看起来只是说明文字的字段：

     · `ANCHORS[i]['quote']`  —— 被 `must_contain` 的**逐字子串**钉住（改一个字就失败）；
     · `ANCHORS[i]['doc']`    —— 被归属断言钉住（legacy 栅栏那几条必须归 Driver API）；
     · `ANCHORS[i]['regex']`  —— 被 `02_stream_dag.py --anchors-net` 的**在线再抽取**钉住
                                 （从下载的官方 HTML 里正则抓取，抓不到即以非零码退出）；
     · `ANCHORS[i]` 的存在本身 —— 被 `COVERED_ANCHORS` 的**并集 == 全集**检查钉住
                                 （`01_sync_contract.py` 用 `ast` 读另外两个程序）；
     · 外部工件的条目          —— 被**逐字行内容**与 `sha256` 前 16 位钉住（读真实文件比对）。

★ 来源等级（定义在 `CONTRIBUTING.md` §2「外部来源分级 A–E」）：

  · **A 级：厂商官方规范**——CUDA C++ Programming Guide 12.6.1 / CUDA Driver API Reference /
    CUDA Runtime API Reference 的原文。本模块的 21 条语义锚点全部是 A 级。
  · **B 级：某一实现的官方材料**——`cuda-training-series` 是 NVIDIA 官方的**课程材料**，
    `cuda-learning` 同理。⇒ **它们只能支撑「NVIDIA 自己这么写」，
    不能支撑「CUDA 规定必须这么写」**（P8 规范与实现不分）。
  · **E 级：本库推导**——本模块**不含**推导值。所有推导（977、512×、8:1 …）都在
    编号程序内部当场算出并断言，**不搬进 `specs.py`**（`CONTRIBUTING` 第 48 行：
    「只放从来源抄来的数，**不放推导值**」）。

★ **本模块的引文是怎么来的（方法论，计划 §0.7 G2 要求全域推广）**：

  > **手打一句像模像样的英文引文，比写错一个数字更危险。**

  ⇒ `quote` 字段**不是手打的**：它由 `02_stream_dag.py --anchors-net` 的
    `extract_anchors()` 从**实际下载的官方 HTML** 里按 `regex` 正则抓取后写入；
    抓不到就打印 `NOT FOUND` 并**不把那条当作依据**（该函数以非零码退出）。
    本文件里保存的是**那一次成功抽取的输出**，逐字复制。

★ **转写层是最容易出错又最难自查的一层**（计划 §2「本域特有的两条」第 3 条）：
  所以 `must_contain` 的每一条都是**短而无歧义的字面量**，且由程序按
  `str.count()` 计数——**不在「我记得是这样」上做判断**。

★ `UNSOURCED`：找不到可定位一手来源的项**只出现在 `UNSOURCED` 里**，
  **正文与常量名里都不出现**（`CONTRIBUTING` 第 65 行：不得「既保留数字又记 UNSOURCED」）。
"""

# ===========================================================================
# 0. 本机的测量条件（域 README：本域无 GPU，全部产出都是推导）
# ===========================================================================
#
# ★ 本模块**不含任何性能数字**——本机无 GPU、无 CUDA 工具链。
#   域 README 明写：本域没有任何 GPU 实测数字（计划 §3 R4）。
#
# ★ 解释器与 numpy 版本在**运行时**由编号程序打印，不在这里硬编码——它们随机器变。
#   见各程序的 `print_env()`。

# ===========================================================================
# 1. ★ 语义锚点（A 级：厂商官方文档原文）
# ===========================================================================
#
# `doc` 取值与含义：
#   guide     CUDA C++ Programming Guide 12.6.1
#   driver    CUDA Driver API Reference —— 3. Stream synchronization behavior
#   rtstream  CUDA Runtime API Reference —— Stream Management
#   rttypes   CUDA Runtime API Reference —— Data Types
#   （四个页面的 URL 见下面的 ANCHOR_DOC_URLS。）
#
# ★★ `section` 逐条用**该引文在抽取出的文本里最近的上位标题**定位，不是凭印象写的。
#    已核实的三处容易写错的地方：
#    · C1 的落点是 guide **§7.6 Synchronization Functions**（`__syncthreads()` 的条目），
#      **不是** §3.2.8.4 之类的线程同步节；
#    · C2 的落点是 guide **§16.6.4 Shared Memory**（compute capability 7.x 那一节），
#      所以它**天然带「7.x 设备」这个作用域**——引用时必须一起写（P7）；
#    · B4c 在 §3.2.8.7.3.1，而 B5a/B5b/B5c 在 §3.2.8.7.3.2 —— 两者**不是同一节**。
#
# ★★ 归属断言（P7 的机械载体）：legacy 栅栏规则（A1/A2/A3/A4）**只在 Driver API**，
#    **不在 Programming Guide**。Guide §3.2.8.5.2 Default Stream 全文只讲
#    `--default-stream legacy/per-thread` 与 NULL stream 的存在，**没有那句栅栏语**。
#    ⇒ 网上二手资料常把它当成 Guide 的内容。`02_stream_dag.py` 的
#    `assert_attribution()` 钉住这一条：**把 A1/A2/A3/A4 的 doc 改成 guide，程序立刻失败。**
#
# ★ `# 来源：` 的机器可核形式就是下面每个条目的 `doc` + `section` 两个字段 ——
#   它们**不是散文**：`doc` 被归属断言钉住（02 的 A.8）；`section` 由人工定位到
#   **该引文在抽取文本里最近的上位标题**，`make anchors` 会逐条打印它抓自哪个页面。
#   本模块**刻意不写「CUDA 文档说」这种不可核的形式**（P7 的判据）。
#
# 字段说明：
#   doc / section 该条的出处（= `# 来源：` 的结构化形式）
#   quote        抽取器输出的**逐字**引文（单行化后用单个空格连接）
#   must_contain 必须逐字出现的短字面量元组；改一个字，程序就报错
#   regex        在线再抽取用的正则（字面空格在抽取时会被放宽成 \s+）
#   note         会被打印的解释。它**不单独充当依据**，只说明该条的作用域限制。
#   covered_by   本锚点**被哪些断言支撑**（标签字符串的元组，如 ("C.3", "A.8")）。
#
# ★★ `note` 是**索引**，不是**唯一载体**（订正轮裁定）：
#   一个只在 `note` 里出现的论断，是**不可证伪**的（P5）—— 它既不在 `quote` 里，
#   也不被任何断言消费。所以本模块定了三条规矩：
#     ① 每个锚点的 `covered_by` **非空**；
#     ② 里面每个标签都必须**真实存在**于某个编号程序的断言清单里（不是编的）；
#     ③ `note` 里凡属**承重论断**的句子，必须能被 `covered_by` 指向的断言机械判定；
#        做不到的（如「本机无 GPU，无法验证」），`note` 里必须**当场写明这一点**，
#        不得写成结论。
#   ★ `covered_by` 允许**跨程序**：锚点归属哪个程序由各程序的 `COVERED_ANCHORS` 决定
#     （两两不交、并集 == 全集），但支撑它的断言可以落在另一个程序里
#     （例如 `B4c` 在本程序 `03` 里被 `C.4` 断言、其行号依据在 `02` 的 `A.6a` 里）。
#   ★ 校验落点：`01_sync_contract.py` 的 **A.11f**（全局：非空 + 标签真实 + 22 条并集覆盖），
#     以及 `02_stream_dag.py` 的 **A.10**、`03_graph_capture.py` 的 **A.4b**（各自的本程序部分）。
#     这三个标签由程序在**运行时**从断言名里提取（`LABEL_RE`），不是手写的清单。

ANCHORS = {

    # -----------------------------------------------------------------------
    # 1.1 legacy / per-thread default stream 的栅栏语义（Driver API）
    #     ⇒ 只被 02_stream_dag.py 的 RULE_STREAM / RULE_LEGACY 消费
    # -----------------------------------------------------------------------
    "A1": {
        "doc": "driver",
        "section": "3. Stream synchronization behavior",
        "quote": "The legacy default stream is an implicit stream which synchronizes "
                 "with all other streams in the same CUcontext except for non-blocking "
                 "streams, described below.",
        "must_contain": (
            "implicit stream which synchronizes with all other streams",
            "same CUcontext except for non-blocking streams",
        ),
        "regex": r"The legacy default stream is an implicit stream which synchronizes with "
                 r"all other streams in the same CUcontext except for non-blocking streams"
                 r"[^.]*\.",
        "note": "★ 同步的范围是「流 ↔ legacy 流」，**不是**「任意两条 blocking 流之间」。"
                "02 的 C 类断言就是这一条的推论：两条 blocking 流之间的直接边为 0"
                "（`C.3`）。★ 本条的**归属**（Driver API 页的 `3. Stream synchronization "
                "behavior`，且该页**只有** A1–A4 这四条）由 `A.8` 正反双向钉住。",
        "covered_by": ("C.3", "A.8"),
    },
    "A2": {
        "doc": "driver",
        "section": "3. Stream synchronization behavior",
        "quote": "When an action is taken in the legacy stream such as a kernel launch or "
                 "cudaStreamWaitEvent(), the legacy stream first waits on all blocking "
                 "streams, the action is queued in the legacy stream, and then all "
                 "blocking streams wait on the legacy stream.",
        "must_contain": (
            "the legacy stream first waits on all blocking streams",
            "then all blocking streams wait on the legacy stream",
        ),
        "regex": r"When an action is taken in the legacy stream[^.]*?the legacy stream first "
                 r"waits on all blocking streams, the action is queued in the legacy stream, "
                 r"and then all blocking streams wait on the legacy stream\.",
        "note": "★ 这一句是**三步栅栏**，不是一步。`must_contain` 拆成两半正是为了让"
                "「只抄前半句」这种漏写被机器抓到（P2 整个载体缺失）。"
                "「三步」这个说法本身有可运行判据：`C.6` 只读前两步会丢掉一条边。"
                "★ 落点：`A.1`（文档自带例子：legacy 上的动作给出一进一出的两条边）、"
                "`A.8`（本条与 A1/A3/A4 同属 Driver API 那一页，正反双向钉住）。",
        "covered_by": ("C.6", "A.1", "A.8"),
    },
    "A3": {
        "doc": "driver",
        "section": "3. Stream synchronization behavior",
        "quote": "Non-blocking streams which do not synchronize with the legacy stream can be "
                 "created using the cudaStreamNonBlocking flag with the stream creation APIs.",
        "must_contain": ("cudaStreamNonBlocking flag with the stream creation APIs",),
        "regex": r"Non-blocking streams which do not synchronize with the legacy stream can be "
                 r"created using the cudaStreamNonBlocking flag with the stream creation APIs\.",
        "note": "★ 这是 hw13 第 67 行 `cudaStreamCreateWithFlags(..., cudaStreamNonBlocking)`"
                " 的依据（03 篇的反直觉量化结论）。"
                "★ 02 里它的落点有两条：`A.2`（同一段程序换成 NonBlocking ⇒ 栅栏边消失）、"
                "`C.8`（故意删掉 `is_blocking` 判定 ⇒ 逐对交叉校验**必须发现**）。"
                "★ 那一行的**行号与逐字内容**由**两处**核对，都不是凭记忆写的："
                "`01` 的 C.13（14 处「行号 + 逐字内容」表）与 `03` 的 `A.5b`"
                "（变体表最后两行差的**就是这一行**）。"
                "★ 但 note 本身不是载体：本锚点的行号与内容是"
                "`specs.HW13_STREAM_FLAGS_LINE` / `HW13_STREAM_FLAGS_TEXT` 两个常量。",
        "covered_by": ("A.2", "C.8", "A.5b", "C.13"),
    },
    "A4": {
        "doc": "driver",
        "section": "3. Stream synchronization behavior",
        "quote": "The per-thread default stream is an implicit stream local to both the thread "
                 "and the CUcontext, and which does not synchronize with other streams "
                 "(just like explicitly created streams).",
        "must_contain": (
            "local to both the thread and the CUcontext",
            "does not synchronize with other streams (just like explicitly created streams)",
        ),
        "regex": r"The per-thread default stream is an implicit stream local to both the thread "
                 r"and the CUcontext, and which does not synchronize with other streams[^.]*\.",
        "note": "★ 括号里那半句是本条的要害：per-thread default stream **不去栅栏**。"
                "02 里它的落点是 `C.4`（`mode: per_thread` ⇒ 隐式栅栏边 **0** 条）。",
        "covered_by": ("C.4",),
    },

    # -----------------------------------------------------------------------
    # 1.2 stream / event 的显式同步语义
    #     ⇒ 只被 02_stream_dag.py 的 RULE_EVENT 消费
    # -----------------------------------------------------------------------
    "A5": {
        "doc": "guide",
        "section": "3.2.8.5.3. Explicit Synchronization",
        "quote": "cudaStreamWaitEvent()takes a stream and an event as parameters (see Events "
                 "for a description of events)and makes all the commands added to the given "
                 "stream after the call to cudaStreamWaitEvent()delay their execution until "
                 "the given event has completed.",
        "must_contain": (
            "makes all the commands added to the given stream after the call to",
            "delay their execution until the given event has completed",
        ),
        "regex": r"cudaStreamWaitEvent\(\)\s*takes a stream and an event as parameters[^.]*?"
                 r"delay their execution until the given event has completed\.",
        "note": "★★ `cudaStreamWaitEvent()takes`、`events)and`、`cudaStreamWaitEvent()delay` "
                "三处的**粘连是源页面本身如此**（`</code>` 后直接跟正文，HTML 里就没有空格），"
                "不是转写丢的 —— 已用 `raw.find()` 在**原始字节**上复核过。"
                "保留原样正是这条引文可信的证据：手打的人不会打出这种粘连。"
                "★ 本条是 `RULE_EVENT` 的依据之一，该引用由 `A.9`（规则 → 锚点的引用完整性）"
                "钉住。\n"
                "★ **覆盖边界必须写明**：`make run` 只能验到 `A.7`/`A.11e`（引文与正则自洽），"
                "「粘连确实来自页面字节」这一步**只在 `make anchors`（联网重抽取）里**成立，"
                "离线不可证伪。",
        "covered_by": ("A.7", "A.11e", "A.9"),
    },
    "A6": {
        "doc": "guide",
        "section": "3.2.8.8. Events",
        "quote": "An event has completed when all tasks - or optionally, all commands in a "
                 "given stream - preceding the event have completed.",
        "must_contain": ("An event has completed when all tasks",
                         "preceding the event have completed"),
        "regex": r"An event has completed when all tasks[^.]*?preceding the event have "
                 r"completed\.",
        "note": "★ 它回答「event 什么时候算完成」。"
                "「02 里 event 是**跨流唯一**的显式手段」这句是可查的：`A.9` 把 5 条建边规则"
                "引用的锚点摆出来，跨流那条（`RULE_EVENT`）是唯一引用本锚点的规则；"
                "`C.1` 则断言「wait 之前没有同事件的 record」必须抛错。",
        "covered_by": ("A.9", "C.1"),
    },
    "A7": {
        "doc": "rtstream",
        "section": "cudaStreamWaitEvent",
        "quote": "Makes all future work submitted to stream wait for all work captured in event.",
        "must_contain": ("Makes all future work submitted to stream wait for all work captured "
                         "in event",),
        "regex": r"Makes all future work submitted to stream wait for all work captured in event\.",
        "note": "★ 「all **future** work」是 02 里 event 语义实现成"
                "「record 之后所有 wait 都生效」的依据，**不是只对紧接着那一条生效**。"
                "★ 实现落点：`Recorder` 只记「事件最后 record 在哪条流」，"
                "所以**任何**后续的 wait 都建边 —— 这条实现由 `A.5` 的逐对交叉校验"
                "与 `A.9` 的规则→锚点引用表覆盖；**本机不验证真实运行时的语义**"
                "（无 GPU、无 CUDA 运行时）。",
        "covered_by": ("A.9", "A.5"),
    },

    # -----------------------------------------------------------------------
    # 1.3 stream capture 的合法边界
    #     ⇒ 只被 03_graph_capture.py 的 CaptureState 消费
    # -----------------------------------------------------------------------
    "B3": {
        "doc": "guide",
        "section": "3.2.8.7.3. Creating a Graph Using Stream Capture",
        "quote": "Stream capture can be used on any CUDA stream except cudaStreamLegacy "
                 "(the “NULL stream”).",
        "must_contain": ("on any CUDA stream except cudaStreamLegacy",),
        "regex": r"Stream capture can be used on any CUDA stream except cudaStreamLegacy[^.]*\.",
        "note": "★ 与 B5b 是**两条不同的规则**：B3 说 capture 不能**起始于** legacy 流，"
                "B5b 说 capture 期间**能不能用** legacy 流取决于那条流的建流标志。"
                "03 的 C 类断言把两者分开测，不合并（`C.3` 管「起始」，"
                "`C.1` 管「期间的使用」，后者才是 B5b 的地盘）。",
        "covered_by": ("C.3", "C.1"),
    },
    "B4a": {
        "doc": "guide",
        "section": "3.2.8.7.3. Creating a Graph Using Stream Capture",
        "quote": "Stream capture can handle cross-stream dependencies expressed with "
                 "cudaEventRecord() and cudaStreamWaitEvent(), provided the event being "
                 "waited upon was recorded into the same capture graph.",
        "must_contain": ("provided the event being waited upon was recorded into the same "
                         "capture graph",),
        "regex": r"Stream capture can handle cross-stream dependencies expressed with "
                 r"cudaEventRecord\(\) and cudaStreamWaitEvent\(\), provided the event being "
                 r"waited upon was recorded into the same capture graph\.",
        "note": "★ 「same capture graph」是**前提**。"
                "★ **覆盖边界必须写明**：原文只声明了这个前提，对「前提不成立」的情形"
                "**未作规定**（既没说不合法，也没定义行为）—— 所以本程序**不构造**"
                "「跨 capture 图等 event」的判据，也不推断它的结果（那会变成编机制从句，P4）。"
                "本锚点因此**只被引文断言覆盖**：`A.4`（`must_contain` 逐字命中）。"
                "★ 它是 `B4b`/`B4c` 的前提：那两条才是可运行判据。",
        "covered_by": ("A.4",),
    },
    "B4b": {
        "doc": "guide",
        "section": "3.2.8.7.3. Creating a Graph Using Stream Capture",
        "quote": "When a captured event is waited on by a stream, it places the stream in "
                 "capture mode if it is not already, and the next item in the stream will have "
                 "additional dependencies on the nodes in the captured event.",
        "must_contain": (
            "it places the stream in capture mode if it is not already",
            "will have additional dependencies on the nodes in the captured event",
        ),
        "regex": r"When a captured event is waited on by a stream, it places the stream in "
                 r"capture mode if it is not already, and the next item in the stream will have "
                 r"additional dependencies on the nodes in the captured event\.",
        "note": "★★ 这一句同时给出两件事：① 被 wait 的流**自动进入 capture**"
                "（可运行判据 `C.7`）；② 它只**加依赖**、不产生节点 —— 03 的 node 计数"
                "口径 B（折叠 rec/wait）依据是**后半句**，不是猜的（落点 `B.5`，"
                "它把两种口径的数都算出来并声明本机无法判定运行时给哪个）。",
        "covered_by": ("C.7", "B.5"),
    },
    "B4c": {
        "doc": "guide",
        "section": "3.2.8.7.3.1. Cross-stream Dependencies and Events",
        "quote": "Failure to rejoin to the origin stream will result in failure of the overall "
                 "capture operation.",
        "must_contain": ("Failure to rejoin to the origin stream will result in failure of the "
                         "overall capture operation",),
        "regex": r"Failure to rejoin to the origin stream will result in failure of the overall "
                 r"capture operation\.",
        "note": "★ hw13 第 132 行 `cudaStreamWaitEvent(streams[0], event2)` 就是**那个 join** "
                "—— 它看起来像一条普通的 event 边，实际是 capture 合法的必需动作。"
                "★ 那一行的行号与逐字内容由 `02` 的 `A.6a` 对着源文件核对（`HW13_TASK1_OPS`）；"
                "「不 join 就整体失败」由本程序的 `C.4` 跑出来。",
        "covered_by": ("C.4", "A.6a"),
    },
    "B5a": {
        "doc": "guide",
        "section": "3.2.8.7.3.2. Prohibited and Unhandled Operations",
        "quote": "It is invalid to synchronize or query the execution status of a stream which "
                 "is being captured or a captured event, because they do not represent items "
                 "scheduled for execution.",
        "must_contain": ("It is invalid to synchronize or query the execution status of a stream "
                         "which is being captured or a captured event",),
        "regex": r"It is invalid to synchronize or query the execution status of a stream which "
                 r"is being captured or a captured event, because they do not represent items "
                 r"scheduled for execution\.",
        "note": "★ **原因是句子里就有的**：被捕获的流「do not represent items scheduled for "
                "execution」—— 它那时根本不是一条能同步的执行队列。这句话**不能**改写成"
                "「因为会破坏图」之类（那是 P4 编机制从句）。"
                "★ 原因从句本身由 `A.4` 的 `must_contain` 逐字钉住；它的**后果**"
                "（同步/查询类动作一律失效）由 `C.1` 与 `A.5` 的变体表覆盖。",
        "covered_by": ("A.4", "C.1", "A.5"),
    },
    "B5b": {
        "doc": "guide",
        "section": "3.2.8.7.3.2. Prohibited and Unhandled Operations",
        "quote": "When any stream in the same context is being captured, and it was not created "
                 "with cudaStreamNonBlocking, any attempted use of the legacy stream is invalid.",
        "must_contain": ("and it was not created with cudaStreamNonBlocking, any attempted use "
                         "of the legacy stream is invalid",),
        "regex": r"When any stream in the same context is being captured, and it was not created "
                 r"with cudaStreamNonBlocking, any attempted use of the legacy stream is "
                 r"invalid\.",
        "note": "★★ 这一条是 03 篇反直觉量化结论的**全部依据**："
                "「NonBlocking ⇒ 合法 / blocking ⇒ cudaErrorStreamCaptureInvalidated」"
                "**只差一个建流标志**。注意 `not` 的位置——把 `not` 读掉会让结论整个反过来。"
                "★ 落点：`A.5b`（变体表最后两行只差一个标志）与它的反向测试 `C.2`；"
                "`A.5` 给出全部 7 个变体逐行一致。",
        "covered_by": ("A.5b", "A.5", "C.2"),
    },
    "B5c": {
        "doc": "guide",
        "section": "3.2.8.7.3.2. Prohibited and Unhandled Operations",
        "quote": "It is therefore also invalid to call synchronous APIs in this case. "
                 "Synchronous APIs, such as cudaMemcpy(), enqueue work to the legacy stream "
                 "and synchronize it before returning.",
        "must_contain": (
            "also invalid to call synchronous APIs in this case",
            "enqueue work to the legacy stream and synchronize it before returning",
        ),
        "regex": r"It is therefore also invalid to call synchronous APIs in this case\. "
                 r"Synchronous APIs, such as cudaMemcpy\(\), enqueue work to the legacy stream "
                 r"and synchronize it before returning\.",
        "note": "★ 它把「同步 API 非法」**归约到 B5b**（`therefore`）：同步 API 落到 legacy 流，"
                "于是按 B5b 非法。⇒ 03 里 memcpy_sync / memset_sync 的违规理由"
                "**与 legacy 流动作是同一个**，不是两条独立规则。"
                "★ 落点：`A.5` 的「插同步 cudaMemcpy（D2H）」那一行（期望失效），"
                "`C.1` 汇总「失效的 5 种」并当场复述 `B5a` 的理由句。",
        "covered_by": ("A.5", "C.1"),
    },
    "B6": {
        "doc": "guide",
        "section": "3.2.8.7.3.3. Invalidation",
        "quote": "When an invalid operation is attempted during stream capture, any associated "
                 "capture graphs are invalidated.",
        "must_contain": ("any associated capture graphs are invalidated",),
        "regex": r"When an invalid operation is attempted during stream capture, any associated "
                 r"capture graphs are invalidated\.",
        "note": "★ 「any associated capture graphs」是复数——一次违规废掉的是**所有**关联图，"
                "所以 03 的 CaptureState.end() 在 invalidated 时返回 NULL graph、"
                "**不返回部分图**（判据 `C.6`：同一个 context 里另有 `graph1` 时仍然返回 NULL）。",
        "covered_by": ("C.6",),
    },
    "B8": {
        "doc": "rtstream",
        "section": "cudaEventRecord",
        "quote": "Event is captured in the graph as an external event node when performing "
                 "stream capture.",
        "must_contain": ("as an external event node when performing stream capture",),
        "regex": r"Event is captured in the graph as an external event node when performing "
                 r"stream capture\.",
        "note": "★ external event node 只在**显式要求**时才产生（该 API 的 flag 参数）；"
                "⇒ 03 的 node 计数口径 B 不把普通 record/wait 当节点，与这一条不冲突。"
                "★ 落点：`B.5`（两种 node 口径的数都算出来，并列出三份间接证据）；"
                "引文本身由 `A.4` 覆盖。★ 本机无 GPU ⇒ 不判定运行时实际给哪一种口径。",
        "covered_by": ("B.5", "A.4"),
    },
    "B9a": {
        "doc": "rttypes",
        "section": "cudaStreamCaptureMode",
        "quote": "Possible modes for stream capture thread interactions.",
        "must_contain": ("Possible modes for stream capture thread interactions",),
        "regex": r"Possible modes for stream capture thread interactions\.",
        "note": "★ 这一条是 **B9b/B9c 的容器**：capture mode 约束的是**线程交互**，"
                "不是「图长什么样」。"
                "★★ **覆盖边界（如实声明，不写成结论）**：本机是**单线程**模拟器，"
                "没有第二个线程可以去违反 mode 的跨线程约束 ⇒ 这一条**只被引文断言覆盖**，"
                "**没有行为判据**（`covered_by` 里只有 `A.4`）。"
                "★ 相关的已知缺口同见 `UNSOURCED`（C6 那一条）。",
        "covered_by": ("A.4",),
    },
    "B9b": {
        "doc": "rtstream",
        "section": "cudaStreamCaptureModeGlobal",
        "quote": "This is the default mode. If the local thread has an ongoing capture sequence "
                 "that was not initiated with cudaStreamCaptureModeRelaxed at "
                 "cuStreamBeginCapture, or if any other thread has a concurrent capture "
                 "sequence initiated with cudaStreamCaptureModeGlobal, this thread is "
                 "prohibited from potentially unsafe API calls.",
        "must_contain": ("This is the default mode.",
                         "this thread is prohibited from potentially unsafe API calls"),
        "regex": r"This is the default mode\. If the local thread has an ongoing capture "
                 r"sequence that was not initiated with cudaStreamCaptureModeRelaxed[^.]*?"
                 r"prohibited from potentially unsafe API calls\.",
        "note": "★ hw13 第 111 行 `cudaStreamCaptureModeGlobal` 是**默认值**，"
                "写出来是自文档，不是必需。"
                "★ 「是默认值」这句话出自本锚点引文的**第一句**（`This is the default mode.`），"
                "它现在被 `A.4` 的 `must_contain` 逐字钉住（不再只活在 note 里）；"
                "第 111 行的**逐字内容**由 `02` 的 `A.6a` 核对。",
        "covered_by": ("A.4", "A.6a"),
    },
    "B9c": {
        "doc": "rtstream",
        "section": "cudaStreamCaptureModeRelaxed",
        "quote": "The local thread is not prohibited from potentially unsafe API calls. Note "
                 "that the thread is still prohibited from API calls which necessarily "
                 "conflict with stream capture, for example, attempting cudaEventQuery on an "
                 "event that was last recorded inside a capture sequence.",
        "must_contain": (
            "Note that the thread is still prohibited from API calls which necessarily "
            "conflict with stream capture",
            "attempting cudaEventQuery on an event that was last recorded inside a capture "
            "sequence",
        ),
        "regex": r"The local thread is not prohibited from potentially unsafe API calls\. Note "
                 r"that the thread is still prohibited from API calls which necessarily "
                 r"conflict with stream capture[^.]*\.",
        "note": "★★ Relaxed **不是豁免**。它举的例子恰好就是 cudaEventQuery —— "
                "03 的 event_query 因此**不看 mode**，一律违规（判据 `C.5`：relaxed 下"
                "query 一个被捕获的 event 仍然 `cudaErrorStreamCaptureInvalidated`）；"
                "「still prohibited」这个**否定词**本身由 `A.4` 的 `must_contain` 逐字钉住。"
                "把 mode 当成总开关是这一条最常见的误读。",
        "covered_by": ("C.5", "A.4"),
    },

    # -----------------------------------------------------------------------
    # 1.4 线程层次与共享内存的契约
    #     ⇒ 只被 01_sync_contract.py 的判据 C1 / C2 消费
    # -----------------------------------------------------------------------
    "C1": {
        "doc": "guide",
        "section": "7.6. Synchronization Functions",
        "quote": "__syncthreads() is allowed in conditional code but only if the conditional "
                 "evaluates identically across the entire thread block, otherwise the code "
                 "execution is likely to hang or produce unintended side effects.",
        "must_contain": (
            "allowed in conditional code but only if the conditional evaluates identically "
            "across the entire thread block",
            "likely to hang or produce unintended side effects",
        ),
        "regex": r"__syncthreads\(\) is allowed in conditional code but only if the conditional "
                 r"evaluates identically across the entire thread block, otherwise the code "
                 r"execution is likely to hang or produce unintended side effects\.",
        "note": "★ 判据 C1 **只**报「条件里含 threadIdx」的场合 —— "
                "「identically across the entire thread block」的可判定子集。"
                "★ 「只报那一类」是**可运行的**：`C.1`（统一控制流夹具不报）与 `C.2`"
                "（发散夹具报 ERROR 且行号现算）是一对正反判据；`C.16` 断言 C1 那条判据的"
                "**证据串里同时含本条与 C1b 两条原文**（契约 + 机制，缺一不可）。"
                "★ **已知局限（如实声明，不是结论）**：一个不含 threadIdx 却仍然发散的"
                "表达式，本检查器**报不出来** —— 这一点 `make run` 里**没有**判据，"
                "01 篇 §4 与 §5 的判据表第 2 行把它写成已知局限。",
        "covered_by": ("C.1", "C.2", "C.16"),
    },
    # ★★ `C1b` 与 `C1` 是**两条不同的锚点**：C1 是**契约**（条件必须整块一致），
    #    C1b 是**机制**（Volta 起 barrier 改成**逐线程**强制，所以违反契约会**死锁**
    #    而不是像 Pascal 那样「只要每个 warp 有人到就算过」）。
    #    ★ **编号命名空间提示（P9）**：锚点 id 有 `A1..A7 / B3..B9c / C1 / C1b / C2`，
    #    而 `01_sync_contract.py` 的**判据 id** 有 `C1..C6` —— 这是**两套编号**，
    #    同名不同指。凡正文提到 `C1` 必须当场说明是锚点还是判据。
    "C1b": {
        "doc": "guide",
        "section": "16.6.2. Independent Thread Scheduling",
        "quote": "Although __syncthreads() has been consistently documented as synchronizing "
                 "all threads in the thread block, Pascal and prior architectures could only "
                 "enforce synchronization at the warp level. In certain cases, this allowed a "
                 "barrier to succeed without being executed by every thread as long as at "
                 "least some thread in every warp reached the barrier. Starting with Volta, "
                 "the CUDA built-in __syncthreads() and PTX instruction bar.sync (and their "
                 "derivatives) are enforced per thread and thus will not succeed until "
                 "reached by all non-exited threads in the block. Code exploiting the "
                 "previous behavior will likely deadlock and must be modified to ensure that "
                 "all non-exited threads reach the barrier.",
        "must_contain": (
            "Pascal and prior architectures could only enforce synchronization at the warp "
            "level",
            "are enforced per thread and thus will not succeed until reached by all "
            "non-exited threads in the block",
            "Code exploiting the previous behavior will likely deadlock",
        ),
        "regex": r"Although __syncthreads\(\) has been consistently documented as synchronizing "
                 r"all threads in the thread block.*?Code exploiting the previous behavior will "
                 r"likely deadlock and must be modified to ensure that all non-exited threads "
                 r"reach the barrier\.",
        "note": "★★ 这一条给出 C1 的**机制**，而且是一个**架构分界**："
                "Pascal 及更早在 **warp 级**强制（只要每个 warp 有人到就算过），"
                "**Volta 起改成逐线程**（必须所有未退出的线程都到）。"
                "⇒ 同一个发散分支的 `__syncthreads()`，在 Pascal 上**可能侥幸不死锁**，"
                "在 Volta 及以后**不会成功**（引文第二句：`are enforced per thread and thus "
                "will not succeed until reached by all non-exited threads in the block`）。"
                "★ **两种措辞必须同时出现，缺一不可**：机制上「不会成功」（引文第二句，"
                "`A.7` 钉住）→ 推出「挂起」；而指南自己在这段末尾的用词是 `will likely "
                "deadlock`（也是 `A.7` 逐字钉住的子串）。01 篇判据表第 1 行的写法"
                "「**必然死锁**（指南：will likely deadlock）」就是把这个区别写在明处 ——"
                "只写前一半会强于原文，只写后一半会漏掉机制。"
                "★ 落点：`C.16`（C1 判据的证据串必须同时含本条与 C1 两条原文）、"
                "`C.2`（发散夹具上确实报 ERROR，且行号现算）。"
                "★ **本机无 GPU ⇒ 实际挂起行为无法验证**：这里只给机制与引文，"
                "「旧卡上没事」是**推断**，`make run` 里没有对应判据。",
        "covered_by": ("C.16", "C.2", "A.7"),
    },
    "C2": {
        "doc": "guide",
        "section": "16.6.4. Shared Memory",
        "quote": "Compute capability 7.x devices allow a single thread block to address the "
                 "full capacity of shared memory: 96 KB on Volta, 64 KB on Turing. Kernels "
                 "relying on shared memory allocations over 48 KB per block are "
                 "architecture-specific, as such they must use dynamic shared memory "
                 "(rather than statically sized arrays) and require an explicit opt-in using "
                 "cudaFuncSetAttribute()",
        "must_contain": (
            "over 48 KB per block are architecture-specific",
            "must use dynamic shared memory (rather than statically sized arrays)",
            "require an explicit opt-in using cudaFuncSetAttribute()",
        ),
        "regex": r"Compute capability 7\.x devices allow a single thread block to address the "
                 r"full capacity of shared memory: 96 KB on Volta, 64 KB on Turing\. Kernels "
                 r"relying on shared memory allocations over 48 KB per block are "
                 r"architecture-specific, as such they must use dynamic shared memory \(rather "
                 r"than statically sized arrays\) and require an explicit opt-in using "
                 r"cudaFuncSetAttribute\(\)",
        "note": "★★ `quote` 里那一句 `Compute capability 7.x devices …` **不是手加的**："
                "C2 的 regex 从那里开始，抽取器抓的就是这两句连在一起。"
                "⇒ 48 KB 这条规则的**作用域是 7.x 设备**，引用时**必须把这一句一起引**，"
                "否则读者会以为它是所有架构的通行上限（P7）。"
                "★ 落点：`A.11e`（离线复刻抽取逻辑，regex 必须匹配存下来的 quote），"
                "以及边界对 `C.10`/`C.11`（正好 48 KB 不报、48 KB + 1 个 float 报 ERROR）。",
        "covered_by": ("A.11e", "C.10", "C.11"),
    },
}

# 抽取器的输入页面。键与 ANCHORS[*]['doc'] 对应。
ANCHOR_DOC_URLS = {
    "guide": "https://docs.nvidia.com/cuda/archive/12.6.1/cuda-c-programming-guide/index.html",
    "driver": "https://docs.nvidia.com/cuda/cuda-driver-api/stream-sync-behavior.html",
    "rtstream": "https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html",
    "rttypes": "https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html",
}

# ★ 归属（P7 的机械载体）：legacy 栅栏规则只在 Driver API 的 Stream synchronization behavior 页。
#   程序据此断言：下列锚点**必须**归 driver，且 driver 页的锚点**只允许是**这几个。
LEGACY_FENCE_ANCHORS = ("A1", "A2", "A3", "A4")
LEGACY_FENCE_DOC = "driver"
LEGACY_FENCE_PAGE = "3. Stream synchronization behavior"

# ===========================================================================
# 2. 外部工件 A：NVIDIA cuda-training-series（B 级：厂商训练材料，不是规范）
# ===========================================================================
#
# 工件版本（`CONTRIBUTING` 第 136 行：**必须写明工件版本与文件路径**）：
#   仓库无 release tag（`TRAIN_NO_TAG`），`git describe --tags --always` 给 9cd77d5
#   （唯一一条提交 `9cd77d539d85ee3457e09a083ba299b787b7a16e`，
#    2021-10-13 "Updated Cori reservation name"）。
#   ⇒ 用**两组**标识固定它：① 提交号（能拿到 git 时的强标识）；
#     ② 每个文件一个 sha256 前 16 位（不依赖 git，任何拷贝都能核）。
#
# ★★ 为什么两个都要：只写提交号，读者拿一个未做 git 的拷贝就无法复核；
#    只写哈希，读者不知道这是哪个版本的材料。**行号引用尤其需要两者** ——
#    换一个版本，同一行号可能就不再是那个表达式了。

TRAIN_ROOT_ENV = "CUDA_TRAINING_SERIES"
TRAIN_COMMIT = "9cd77d539d85ee3457e09a083ba299b787b7a16e"
TRAIN_COMMIT_SHORT = "9cd77d5"
TRAIN_COMMIT_DATE = "2021-10-13"
TRAIN_NO_TAG = True

# ---------------------------------------------------------------------------
# 2.1 ★★ G3：`blocks = (N + (threads - 1) / threads)` —— **7 个文件**逐字相同
# ---------------------------------------------------------------------------
# 来源：**NVIDIA cuda-training-series（B 级：厂商训练材料，不是规范）**，
#       commit 9cd77d5（2021-10-13，无 release tag），文件与行号见下表每一项；
#       每一条都由 `01_sync_contract.py` 以 `raw.count()` + 逐字行内容 + sha256 三重核对。
# ★ 计划 v2 写「6/6」，那是**搜索范围就是手写的 6 条路径**造成的（P6）。v2c 订正为 7。
BLOCKS_BUG_EXPR = "blocks = (N + (threads - 1) / threads)"
BLOCKS_BUG_DECL = "int " + BLOCKS_BUG_EXPR + ";"
BLOCKS_BUG_FILES = (
    # (仓库相对路径, 行号)
    ("exercises/hw13/Solutions/axpy_cublas_solution.cu", 91),
    ("exercises/hw13/Solutions/axpy_stream_capture_solution.cu", 104),
    ("exercises/hw13/axpy_stream_capture_with_fixme.cu", 105),
    ("exercises/hw13/axpy_stream_capture_timer.cu", 97),
    ("exercises/hw13/axpy_stream_capture_from_scratch.cu", 93),
    ("exercises/hw13/axpy_cublas_with_fixme.cu", 92),
    ("exercises/hw13/axpy_cublas_from_scratch.cu", 76),
)
# ★ 这两份是**参考解**（hw13/README.md 第 5 行指路 Solutions 目录）。
#   ⇒ 七个同错**包含两份参考解** ⇒ 是写法习惯，不是偶发笔误。
BLOCKS_BUG_SOLUTION_FILES = (
    "exercises/hw13/Solutions/axpy_cublas_solution.cu",
    "exercises/hw13/Solutions/axpy_stream_capture_solution.cu",
)
# ★ 这条一次性替换是 C 类反向测试（C.8）的输入：正确写法。
BLOCKS_FIXED_DECL = "int blocks = (N + threads - 1) / threads;"

# 2.2 hw13 的规模常量（逐字来自源文件，行号已核实）
# 来源：**cuda-training-series（B 级）** `exercises/hw13/axpy_stream_capture_with_fixme.cu`
#       第 21 行（N）与第 104 行（threads）；行内容逐字记在下面的 `_TEXT` 常量里。
HW13_N_PATH = "exercises/hw13/axpy_stream_capture_with_fixme.cu"
HW13_N_LINE = 21
HW13_N_TEXT = "#define N 500000"
HW13_N = 500000
HW13_THREADS_LINE = 104
HW13_THREADS_TEXT = "int threads = 512;"
HW13_THREADS = 512

# 2.3 hw13 第 67 行的建流标志（03 篇反直觉量化结论的载体）
HW13_STREAM_FLAGS_LINE = 67
HW13_STREAM_FLAGS_TEXT = "cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);"

# 2.4 hw13 task1 的发射顺序（逐字行号，已核实）
#     ★ 用途：02 的内嵌 fixture 是**转写**；本表把转写**钉回源文件**（不靠记忆）。
#     BEGIN/END 是 capture 区的边界，不对应 fixture 里的 op 行。
HW13_TASK1_PATH = HW13_N_PATH
HW13_TASK1_OPS = (
    (111, "cudaStreamBeginCapture(streams[0], cudaStreamCaptureModeGlobal);", "BEGIN"),
    (114, "kernel_a<<<blocks, threads, 0, streams[0]>>>(d_x, d_y);",
     "streams[0] : kernel kernel_a"),
    (117, "cudaEventRecord(event1, streams[0]);", "streams[0] : record event1"),
    (120, "kernel_b<<<blocks, threads, 0, streams[0]>>>(d_x, d_y);",
     "streams[0] : kernel kernel_b"),
    (123, "cudaStreamWaitEvent(streams[1], event1);", "streams[1] : wait event1"),
    (126, "kernel_c<<<blocks, threads, 0, streams[1]>>>(d_x, d_y);",
     "streams[1] : kernel kernel_c"),
    (129, "cudaEventRecord(event2, streams[1]);", "streams[1] : record event2"),
    (132, "cudaStreamWaitEvent(streams[0], event2);", "streams[0] : wait event2"),
    (135, "kernel_d<<<blocks, threads, 0, streams[0]>>>(d_x, d_y);",
     "streams[0] : kernel kernel_d"),
    (138, "cudaStreamEndCapture(streams[0], &graph);", "END"),
)
# ★ 02 的 fixture 里那 4 个 kernel 的名字与顺序（供逐行比对用）
HW13_KERNEL_ORDER = ("kernel_a", "kernel_b", "kernel_c", "kernel_d")
# ★ hw13 里 `<<<blocks, threads...>>>` 的出现次数 —— C5 报 4 处的依据
HW13_LAUNCH_COUNT = 4

# 2.5 ★ G7：capture 的七种变体（前两列是输入，第三列是期望返回）
#     ★ 最后两行的**唯一差别**是那条被捕获的流有没有 cudaStreamNonBlocking。
CAPTURE_VARIANTS = (
    # (变体描述, 插入动作键, 被捕获流是否 nonblocking, 期望 cudaStreamEndCapture 返回)
    ("原样（不插任何东西）", None, True, "cudaSuccess"),
    ("插 cudaStreamSynchronize(streams[0])", "stream_synchronize", True,
     "cudaErrorStreamCaptureInvalidated"),
    ("插 cudaDeviceSynchronize()", "device_synchronize", True,
     "cudaErrorStreamCaptureInvalidated"),
    ("插 cudaEventSynchronize(event1)", "event_synchronize", True,
     "cudaErrorStreamCaptureInvalidated"),
    ("插同步 cudaMemcpy（D2H）", "memcpy_sync", True,
     "cudaErrorStreamCaptureInvalidated"),
    ("插 legacy 流动作（该流是 NonBlocking 的）", "legacy_command", True, "cudaSuccess"),
    ("插 legacy 流动作（该流是 blocking 的）", "legacy_command", False,
     "cudaErrorStreamCaptureInvalidated"),
)

# 2.6 hw13 task1 的宿主 API 调用计数（03 篇 B 类的对象）
#     ★ 口径：**只数 CUDA runtime API 调用**，不数 cudaCheckErrors(...) 展开出的
#       cudaGetLastError() —— 后者是诊断宏，每个调用后面都跟一个，
#       把它算进去会让两边**同时**翻倍（比值不变），但绝对数会失去意义。
#       ★ 这个口径必须写明，否则「8 次/迭代」对不上读者的数法。
HW13_TIMER_PATH = "exercises/hw13/axpy_stream_capture_timer.cu"
HW13_TIMER_LOOP = (134, 158, 1000)     # 第 134–158 行，for (i < 1000)
HW13_GRAPH_PATH = "exercises/hw13/Solutions/axpy_stream_capture_solution.cu"
HW13_GRAPH_LOOP = (166, 170, 1000)     # 第 166–170 行，for (i < 1000)
HW13_TIMER_CALLS_PER_ITER = 8
HW13_GRAPH_CALLS_PER_ITER = 1
HW13_KERNELS_PER_ITER = 4

# 2.6b hw13 task2（把库调用 capture 成 child graph）—— 逐字行号已核实
#      ★★ 注意：task2 的发射循环是 **100** 次（不是 task1 的 1000 次），
#         而且它的循环体里**没有** cudaCheckErrors —— 只有 launch + synchronize 两条。
#         计划 §0.7 G7 与探针把 task2 的循环也叫「定时循环」，
#         但源文件里 task2 **根本没有计时**（没有 std::chrono）。
#         ⇒ 正文只能说 task1 是定时循环；task2 是**发射循环**。这是实施时发现的规格偏差。
HW13_TASK2_PATH = "exercises/hw13/axpy_cublas_with_fixme.cu"
HW13_TASK2_LOOP = (153, 157, 100)   # ★ 闭合的 `}` 在第 157 行；第 158 行是 cudaCheckErrors
HW13_TASK2_LOOP_TEXT = "for (int i = 0; i < 100; ++i){"
HW13_TASK2_CALLS_PER_ITER = 2      # cudaGraphLaunch + cudaStreamSynchronize（无 cudaCheckErrors）
HW13_TASK2_OPS = (
    (38, "cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);",
     "库调用能被 capture 的前提之一：这条流是 NonBlocking 的"),
    (85, "cudaGraphCreate(FIXME, 0); // create the graph", "FIXME 之一"),
    (109, "cudaStreamBeginCapture(stream1, cudaStreamCaptureModeGlobal);", "capture 区开始"),
    (113, "cublasSaxpy(cublas_handle, N, &d_a, d_x, 1, d_y, 1);", "★ 被 capture 的库调用"),
    (116, "cudaStreamEndCapture(stream1, &libraryGraph);", "capture 区结束"),
    (120, "cudaGraphAddChildGraphNode(FIXME, graph, FIXME,", "FIXME 之一：child graph 节点"),
    (149, "cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);", "instantiate"),
    (155, "cudaGraphLaunch(FIXME, stream1);", "发射循环体内的第一条"),
    (156, "cudaStreamSynchronize(stream1);", "★ 与 task1 不同：task2 每次发射后都同步"),
)
# task1 的两个真实文件里的 capture 区边界与定时循环（逐字行号）
HW13_SOLUTION_PATH = "exercises/hw13/Solutions/axpy_stream_capture_solution.cu"
HW13_SOLUTION_PINS = (
    (110, "cudaStreamBeginCapture(streams[0], cudaStreamCaptureModeGlobal);"),
    (137, "cudaStreamEndCapture(streams[0], &graph);"),
    (141, "cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);"),
    (147, "cudaGraphLaunch(instance, streams[0]);"),
    (149, "cudaStreamSynchronize(streams[0]);"),
    (167, "cudaGraphLaunch(instance, streams[0]);"),
)
HW13_SOLUTION_ONLY_LAUNCH_IN_LOOP = True

# 2.7 ★ G4：hw5 的 max 归约幺元（真值来源是**两个文件的逐字差异**）
# 来源：**cuda-training-series（B 级）** `exercises/hw5/max_reduction.cu` 与
#       `max_reduction_solution.cu`；两条真值都在下面的「行号 + 逐字行内容」里。
HW5_SUM_PATH = "exercises/hw5/max_reduction.cu"
HW5_MAX_PATH = "exercises/hw5/max_reduction_solution.cu"
HW5_INIT_LINE = 23
HW5_INIT_TEXT = "sdata[tid] = 0.0f;"
HW5_SUM_LINE = 27
HW5_SUM_TEXT = "sdata[tid] += gdata[idx];"
HW5_SUM_LINE2 = 34
HW5_SUM_TEXT2 = "sdata[tid] += sdata[tid + s];"
HW5_MAX_LINE = 27
HW5_MAX_TEXT = "sdata[tid] = max(gdata[idx], sdata[tid]);"
HW5_MAX_LINE2 = 34
HW5_MAX_TEXT2 = "sdata[tid] = max(sdata[tid + s], sdata[tid]);"
# ★ 该文件自带的测试数据**全为正**（`h_A[i]=1.0f`、`h_A[100]=5.0f`）——
#   这正是「max 用 0.0f 当幺元」这个语义错误**在它自己的输入上看不出来**的原因。
#   `01_sync_contract.py` 的 C.18 把它与源文件里的字面量绑在一起断言。
HW5_TEST_DATA_POSITIVE_ONLY = True
# ★ 逐字行（行号已核实）：值分别是 1.0f 与 5.0f，**都是正数** —— 所以 max 归约
#   在这份输入上给出的是正确结果，`0.0f` 这个坏幺元**不会被它自己的测试暴露**。
#   （注：5.0f 不在第 48 行直接出现，而是在第 45 行 `float max_val = 5.0f;` 里。）
HW5_TEST_DATA_LINES = (
    (45, "float max_val = 5.0f;"),
    (47, "h_A[i] = 1.0f;"),
    (48, "h_A[100] = max_val;"),
)

# 2.8 ★ G5：hw2 的分块装填
# 来源：**cuda-training-series（B 级）** `exercises/hw2/matrix_mul_shared.cu` 与
#       `matrix_mul_shared_solution.cu` 第 40/41 行（两文件的循环头都在第 37 行）。
HW2_BAD_PATH = "exercises/hw2/matrix_mul_shared.cu"
HW2_GOOD_PATH = "exercises/hw2/matrix_mul_shared_solution.cu"
HW2_BAD_LINES = {
    40: "As[threadIdx.y][threadIdx.x] = A[idy*ds+idx];",
    41: "Bs[threadIdx.y][threadIdx.x] = B[idy*ds+idx];",
}
HW2_GOOD_LINES = {
    40: "As[threadIdx.y][threadIdx.x] = A[idy * ds + (i * block_size + threadIdx.x)];",
    41: "Bs[threadIdx.y][threadIdx.x] = B[(i * block_size + threadIdx.y) * ds + idx];",
}
# 循环头（两文件相同，都在第 37 行）—— C3 的「外层 for 的归纳变量」就是这里的 i
HW2_LOOP_LINE = 37
HW2_LOOP_TEXT = "for (int i = 0; i < ds/block_size; i++) {"

# 2.9 hw10：纯 legacy 栅栏，一个 event 都没有
# 来源：**cuda-training-series（B 级）** `exercises/hw10/streams_solution.cu`，
#       行号逐条列在下面（第 9/12/73/95/99–105 行）。
HW10_PATH = "exercises/hw10/streams_solution.cu"
HW10_CHUNKS_LINE = 9
HW10_CHUNKS_TEXT = "const int chunks = 64;"
HW10_NUM_STREAMS_LINE = 12
HW10_NUM_STREAMS_TEXT = "const int num_streams = 8;"
HW10_STREAM_CREATE_LINE = 73
HW10_STREAM_CREATE_TEXT = "cudaStreamCreate(&streams[i]);"       # ★ 没有 NonBlocking
HW10_MEMSET_LINE = 95
HW10_MEMSET_TEXT = "cudaMemset(d_y, 0, ds * sizeof(ft));"         # ★ 落到 NULL 流
HW10_PRAGMA_LINE = 99
HW10_PRAGMA_TEXT = "#pragma omp parallel for"
HW10_LOOP_FIRST_LINE = 100
HW10_LOOP_LAST_LINE = 104
HW10_LOOP_TEXT = "for (int i = 0; i < chunks; i++) { //depth-first launch"
HW10_SYNC_LINE = 105
HW10_SYNC_TEXT = "cudaDeviceSynchronize();"
# ★ 02 的 fixture 是**同构切片**（chunks=4 / num_streams=2），口径必须写明：
#   每个流上的语句种类与顺序、以及 NULL 流动作的插入位置完全一样。
HW10_FIXTURE_CHUNKS = 4
HW10_FIXTURE_STREAMS = 2
HW10_LIVE_STREAMS = 8

# 2.10 逐字行内容的哈希（核对「我们读到的确实是这份工件」）
#     ★ 用前 16 位十六进制：`hashlib.sha256(read_bytes()).hexdigest()[:16]`。
TRAIN_SHA16 = {
    "exercises/hw13/Solutions/axpy_cublas_solution.cu": "4208176584f943e8",
    "exercises/hw13/Solutions/axpy_stream_capture_solution.cu": "238ead9e728aa01f",
    "exercises/hw13/axpy_stream_capture_with_fixme.cu": "1491626ba42bff23",
    "exercises/hw13/axpy_stream_capture_timer.cu": "428427590c3662ca",
    "exercises/hw13/axpy_stream_capture_from_scratch.cu": "ffb82f51be4c6041",
    "exercises/hw13/axpy_cublas_with_fixme.cu": "6eda7971883cfec4",
    "exercises/hw13/axpy_cublas_from_scratch.cu": "e43d7ddc39d666b0",
    "exercises/hw5/max_reduction.cu": "4008b74d175b7b0f",
    "exercises/hw5/max_reduction_solution.cu": "fdd20aff90195915",
    "exercises/hw2/matrix_mul_shared.cu": "880200e730aa0ffe",
    "exercises/hw2/matrix_mul_shared_solution.cu": "839c755078c1aa4b",
    "exercises/hw10/streams_solution.cu": "b80fde00c851d52d",
    "exercises/hw13/README.md": "3b973c551552db28",
    "exercises/hw13/graph_stream_capture.png": "90ff527762142e60",
    "exercises/hw10/readme.md": "4e6249e5593caa4b",
}

# 2.11 hw13 README 里被引用的三行（逐字；行号已核实）
# 来源：**cuda-training-series（B 级）** `exercises/hw13/README.md`；
#       ★ 第 74 行的拼写 `preformance` 是**原文如此**，不要「订正」。
#   ★ README 里的拼写错误 preformance 是**原文如此**，不要「订正」。
HW13_README_PATH = "exercises/hw13/README.md"
HW13_README_CITES = {
    15: "This is the same example from the slides, feel free to refer to them for help "
        "and hints.",
    74: "you should still be able to see a small preformance increase using the graph from "
        "the launch overhead savings.",
    81: "We are creating 2 kernel nodes and a child graph derived from a cuBLAS axpy "
        "function call.",
}
# ★ hw13 README 里那条**唯一**的定性时间说法（第 74 行）。★ 它**没有给出倍数** ——
#   这是 03 篇「收益是宿主动作数、不是时间」这条结论的唯一外部依据。
HW13_README_TIME_FRAG = ("you should still be able to see a small preformance increase")

# ★ 被引用但**不能机器读**的那份证据（人眼读过）：NVIDIA 自己画的 capture 图。
#   必须与「README 第 15 行的文字」分开写 —— 后者并不支持前者那条断言（P7）。
HW13_DIAGRAM_PNG = "exercises/hw13/graph_stream_capture.png"
HW13_DIAGRAM_CLAIM = "右侧画出 A/B/C/D 四个 kernel 节点、四条边"

# 2.12 hw10 readme 里唯一一条时间说法（逐字；**本机无法验证**）
HW10_README_PATH = "exercises/hw10/readme.md"
HW10_README_TIME_LINE = 70
HW10_README_TIME_FRAG = "The overlapped version of the code should be about 2X faster"

# ===========================================================================
# 3. 外部工件 B：cuda-learning（B 级；**目标环境是 Tesla T4 / sm_75**）
# ===========================================================================
#
# 来源：**cuda-learning（B 级：某一实现的官方材料）** —— 它**不是 git 仓库**，
#       版本只能用文件 sha256（`git log` 报 not a repository）。
# ★★ 这个仓库**不是 git 仓库**（`git log` 报 not a repository），
#    ⇒ 版本只能用文件哈希，**不能用提交号**。这一点必须写明（P6：别声称有版本却拿不出）。
# ★★ 它的 README.md 第 16–17 行写明目标环境；第 22 行写明 T4 上**没有**什么。
#    ⇒ **凡引用它的代码，必须同时说明这一点**，否则读者会以为那是 Hopper/Blackwell 的写法。
LEARN_ROOT_ENV = "CUDA_LEARNING"
LEARN_IS_GIT = False
LEARN_ENV_LINE = 16
LEARN_ENV_TEXT = "- **GPU：Tesla T4**（计算能力 **7.5 / sm_75**，Turing 架构）"
LEARN_ARCH_LINE = 17
LEARN_ARCH_TEXT = "CUDA Toolkit ≥ 11.x（编译时 `-arch=sm_75`）"
LEARN_ABSENT_LINE = 22
LEARN_ABSENT_FRAG = ("但**没有** bf16 / TF32（Ampere+）、**没有** cp.async / TMA / "
                     "warp-specialization（Ampere/Hopper+）")
LEARN_GEMM_PATH = "02_gemm/gemm.cu"
LEARN_FLASH_PATH = "04_flash_attention/flash_attention.cu"
LEARN_SHA16 = {
    "README.md": "9db33e4420af202e",
    "02_gemm/gemm.cu": "29dc838bcd609d9b",
    "04_flash_attention/flash_attention.cu": "88f7ed464dca890d",
}
# ★ 误报回归的真值来源（两个已知正确的文件）：
LEARN_GEMM_TAINT_NOTE = "第 44 行 `int kA = t * TILE + threadIdx.x;`，第 49 行用 `A[row * K + kA]`"
LEARN_FLASH_NOTE = "第 28-29 行 `__shared__ __half Ks[Bc][D];`，装填走 `Ks[r][j] = K[row*D+j]`"

# ===========================================================================
# 4. ★ 判据的真值强度分级（计划 §0.7 G / Chunk 1 的表，必须逐条标注）
# ===========================================================================
#
# ★ 这张表**本身就是被断言的常量**：01 的程序断言它的键覆盖了全部判据 id，
#   且每一档都是允许值之一。⇒ 新加一条判据而忘了在这里分级，程序立刻失败。
RULE_STRENGTH = {
    "C1": ("A", "官方原文（specs.ANCHORS['C1']）"),
    "C2": ("A", "官方原文（specs.ANCHORS['C2']）"),
    "C3": ("B", "仓库里成对的「已知有问题 vs 已知正确」文件（hw2）"),
    "C4": ("B", "仓库里成对的「sum 版 vs max 参考版」文件（hw5）"),
    "C5": ("B", "仓库里的真实文件 + 逐字行内容（hw13，7 个文件）"),
    "C6": (None, "★ 设计了但没实现 —— 不得声称覆盖（计划 Chunk 1 的强度表）"),
}
STRENGTH_LEVELS = ("A", "B", None)

# C 类断言的期望最严重等级。
# ★ 这是「**C 类成立时程序应当失败**」的机制载体：把实际等级与这里比对，不等即非零退出。
RULE_EXPECTED_SEVERITY = {
    "C1": "ERROR",
    "C2": "ERROR",
    "C3": "ERROR",
    "C4": "WARN",
    "C5": "WARN",
}
SEVERITIES = ("ERROR", "WARN")

# ===========================================================================
# 5. hw13 capture 的 node 计数：两种口径（03 篇 §4 的踩坑之一）
# ===========================================================================
#
# 口径 A：record/wait 各算一个节点 —— §3.2.8.7.1.1 Node Types 里确实列了
#         waiting on an event / recording an event。
# 口径 B：record/wait 折叠成依赖边，不产生节点 —— 依据是 B4b 的后半句
#         （will have **additional dependencies** on the nodes in the captured event，
#          只加依赖、没加节点），以及 B8（只有 cudaEventWaitExternal 会变成显式节点）。
#
# ★★ 本机**无法判定**运行时给哪个：只有三份互相吻合的**间接证据**（计划 §0.7 G8-4）：
#      ① B4b 的措辞；② B8 的 external event node 专项；
#      ③ 仓库里 NVIDIA 自己画的 graph_stream_capture.png 右侧是 A/B/C/D **四个**节点。
#    ⇒ 正文必须写成「证据吻合，不是执行结果」，不得写成 node 数已确定。
HW13_TASK1_RAWNODES = 8           # 8 个设备侧动作（不是 8 个 kernel）
HW13_TASK1_COLLAPSED_NODES = 4    # 口径 B：A/B/C/D 四个 kernel 节点

# ===========================================================================
# 6. UNSOURCED：**只在这里出现、正文与常量名里都不出现**的项
# ===========================================================================
#
# ★ CONTRIBUTING 第 65 行：不得「既保留数字又记 UNSOURCED」——两者只能留一个。
#   本模块与三个编号程序**都不断言**下列内容；正文只在明确说「未验证 / 无法判定」时提到。
UNSOURCED = (
    ("kernel 在 GPU 上的实际执行行为",
     "本机无 GPU，一个 thread 都没跑过。计划 §0.7 G8-1。"),
    ("任何时间数字（hw10 的 about 2X、hw13 的 small increase）",
     "两个定性说法都**没有**给出倍数；本机也没有 GPU 去测。计划 §0.7 G8-3。"),
    ("hw13 capture 图的 node 数（4 还是 8）",
     "只有三份间接证据吻合口径 B，**不是执行结果**。计划 §0.7 G8-4。"),
    ("libcublas 是否真的支持被 capture",
     "hw13 README 描述的 cuBLAS 捕获是**成功案例**，但本机无 libcublas、无 GPU。"),
    ("cudaStreamCaptureMode 在多线程下的效果（B9a/B9b/B9c）",
     "模拟器是单线程的；这三条锚点**只被引述、不被行为验证**。"),
    ("C6 判据（设计了但没实现）",
     "计划 §0.7 G8-6：**不得声称覆盖**。"),
)

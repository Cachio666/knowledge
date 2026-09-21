#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""specs.py —— 非编号辅助模块：**外部事实常量**（`50-performance/bottleneck-diagnosis`）。

★ 它是「非编号支撑代码」，不计入「文档数 = 编号程序数」的核对（`CONTRIBUTING` §2）。

===============================================================================
★★ 本主题的测量条件 —— 逐处都要带，不许省
===============================================================================

按 `CONTRIBUTING` §2 的四类口径，本主题的数字分成**两类，标签必须分开**：

| 类 | 本主题里是什么 | 必须写明的限定语 |
|---|---|---|
| **本机实测值** | **CPU 上的真实测量**：`clang -O2` 的访存微基准（依赖链曲线 / 8 路展开）、`sample` 的调用图、`cProfile` 的调用计数、`perf_counter` / `getrusage` 的读数 | `python3` 版本 / `numpy` 版本 / `clang` 版本 / macOS |
| **规格值** | **NVIDIA 官方文档事实**（`FACTS` 的 F1–F15） | **URL + 章节号 + 抓取日 + 文档页脚 last-updated** |

**两者不得互证**（`CONTRIBUTING:90` 与域 README）：
规格事实只能支撑「工具是怎么设计的」，**不能支撑「你这台机器的性能」**——后者本机没有。

★ **本机没有 GPU、没有 `ncu`/`nsys`/`nvprof`/`nvidia-smi`/`CUPTI`/`perf`。**
  本主题**一切实测都是 CPU 上的**，**不支撑任何关于 GPU 的结论**。

===============================================================================
★★ 本主题**实际引用**哪几条规格事实（写清楚，免得读者以为 15 条都用到）
===============================================================================

  · **F10**（超范围不 clamp）与 **F3**（跨 pass 的比值）——`02-reading-counters` §1/§4；
  · **F11**（stall 归因给消费者指令）、**F12**（官方反模式：目标不是把 stall 压到 0）
    ——`01-latency-and-bandwidth` §4 与 `02-reading-counters` §1；
  · **F14**（同名不同指：2.4.7 与 2.4.8 是两套采样条件）——`02-reading-counters` §2。

★★ **但整张 F1–F15 表仍然原样放在这里**，理由有两条：
  ① 计划 §1.3 明令 `specs.py` 放 **F1–F15**；② `fact_invariants()` 那条自洽检查
     （**每条 `must_contain` 必须是它自己 `quote` 的子串**）需要**整张表**才有意义——
     只放被引用的那几条，等于把「引文与它的锚钉在一起」这条纪律缩小到几个特例。
  ★ **逐字引文块与 `50-performance/profiling-tools/code/specs.py` 同源**（同一份 HTML、
    同一个 sha256、同一次抓取）：它是**机械搬入**的，不是重新转写的
    （转写是独立的出错环节，见 `CONTRIBUTING:280` 与计划 §0.7 C 的抽取纪律）。

===============================================================================
★★ 来源等级（沿用 `CONTRIBUTING` §2 的 A–E）
===============================================================================

  · **A 级：厂商官方文档** —— `FACTS` 的 F1–F15 全部来自 `docs.nvidia.com`。
    **A 级支撑「NVIDIA 规定/记述该工具是这样」**，仍**不支撑**
    「你在本机看到的现象与它一致」（那是本机实测值那一类的事）。
  · **E 级：本库推导** —— 本模块只放**一个**转写下来的常数
    （`MDE_COEFFICIENT_AT_POWER_80`，从 `45/02` 代入，**本域不重推**）。
  · **D 级：0 条。** 本主题不引用任何社区项目文档。

===============================================================================
★★ 抽取纪律（本域特有，写规格时差点栽在这上面）
===============================================================================

HTML 标签会**切断字节连续串**。实测：`ProfilingGuide` 里那句话的原始 HTML 是
`to their <em>boost</em> value`，因此 `raw.count(b"boost value")`（字节口径）= **0**。
**若据此下结论「文档里没有这句话」，就是一次完全的假发现。**

★ 实施者踩到的**第二个**陷阱是**去标签的方式本身**：把标签**一律替换成空格**，
  会让 `single <em>pass</em>.` 变成 `single pass .`（**中间多一个空格**），
  于是 15 条逐字引文里有 **7 条**被假报「不在场」。
  ⇒ 正确做法：**先按 `<p>` 切段落，再在段内把标签删成【空串】**（不是替换成空格）。

⇒ **抽取纪律的完整形式是三条**：① 先确认锚点在**字节上连续**；② 用**正确的方式**去标签；
  ③ 在**原始字节**上计数。**三条缺一，都会产出「看起来很正常的假结论」。**

★ 本模块 `FACTS[*]["must_contain"]` 的每一条都是**短而无歧义、且在字节上连续**的字面量；
  `must_contain` 还必须**是 `quote` 的子串**——这一条由 `fact_invariants()` 当场核。

===============================================================================
★ 本模块的路径探测（文档在场时做逐字核对，不在场时**明确 SKIP**，不静默通过）
===============================================================================

候选列表见 `DOC_CANDIDATES`，**相对项以本文件所在目录为基准**（不是 cwd —— 见
`30-gpu-programming/cutlass` 的教训）。`doc_status()` 返回三态：

  · `"ok"`    —— 文件在、字节数与 sha256 都对；
  · `"bad"`   —— 文件在，但 sha256/字节数对不上（**必须让程序失败**，不是 SKIP）；
  · `"absent"`—— 文件不在（**SKIP**，并在程序输出里说明怎么拿到它）。

★ **`absent` 不等于通过**：编号程序把它打印成 `[SKIP]` 并**单独计数**。
"""

import hashlib
import os

# ===========================================================================
# 0. 测量条件（每处引用都要带的那句话）
# ===========================================================================
MEASURE_IMPL = "本机 CPU（Apple M1 Pro / macOS）"
MEASURE_NOTE = ("本主题的实测数字全部是**本机 CPU 上的**，不是 GPU、不是 Nsight Compute；"
                "它们**只支撑关于机制的结论**，不支撑任何关于 NVIDIA 工具具体实现的结论")
FETCH_DATE = "2026-09-21"

# ===========================================================================
# 1. 两份 NVIDIA 文档（A 级来源）
# ===========================================================================
# ★ 来源：`docs/superpowers/plans/2026-09-21-performance-domain.md` §0.7 C。
#   抓取方式：`curl -sS -o <file> -w '%{http_code}'`（**最简形式即 200，无需 `-L`**）。
#   ⚠ 这不是「通用规则」，而是「`CONTRIBUTING:273` 那条**因 URL 而异**的规则在本次的取值」。
DOCS = {
    "ProfilingGuide": {
        "url": "https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html",
        "bytes": 381812,                       # 实测
        "sha256": "6a84b367be4dd99a9df44a2ad5a0e2c58a1ba499562e939da8363b57889dc15f",
        "sha256_16": "6a84b367be4dd99a",
        "http": 200,
        "last_updated": "Aug 27, 2026",        # 文档页脚
        "fetched": FETCH_DATE,
        "file": "profiling_guide.html",
    },
    "NsightComputeCli": {
        "url": "https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html",
        "bytes": 162822,                       # 实测
        "sha256": "b6db571443a5a641b74ef5168105eae56e073ad77f190a3d5e96ea0db2ddac9a",
        "sha256_16": "b6db571443a5a641",
        "http": 200,
        "last_updated": "Aug 27, 2026",
        "fetched": FETCH_DATE,
        "file": "ncu_cli.html",
    },
}

# ★ 文档本体的探测：**不写死构造者本机的目录布局**。环境变量优先，其余为相对候选。
DOC_ENV = "PERF_DOC_DIR"
DOC_CANDIDATES = (
    "$PERF_DOC_DIR",        # ← 第一顺位由 DOC_ENV 提供，见 resolve_candidates()
    "~/ncu_src",            # 兜底（家目录，可移植）
    ".docs",                # 本目录下的快照
    "../ncu_src",           # 兄弟目录
    "/tmp/ncu_src",         # 兜底：控制者抓取时的临时目录（**不保证存在、会被系统清理**）
)

# ===========================================================================
# 2. F1–F15（**A 级**，逐字引文）
# ===========================================================================
# ★ 每条的结构：
#   id / title        —— 计划 §0.7 C 的编号与标题
#   doc / section     —— 来源：URL 见 DOCS[doc]["url"]，章节号**控制者已在原始 HTML 上核过**
#   quote             —— **逐字引文**（从原始 HTML 的 `<p>` 段落去标签后得到，未经人工转写）
#   must_contain      —— 用来核 `quote` 的短锚（**必须是 `quote` 的子串**，由 fact_invariants() 核）
#   level             —— 来源等级（本表恒为 "A"）
#
# ⚠ 每个 `quote` 的字符是**从 HTML 去标签后**得到的；`’` 是原文的 U+2019，不是 ASCII 撇号。
# ★★ 本块与 `50-performance/profiling-tools/code/specs.py` 的 FACTS **同源、机械搬入**
#    （同一份文档、同一个 sha256、同一次抓取）——**不是重新转写**。
FACTS = (
    {
        "id": "F1", "title": "一次 pass 收不齐所有 metric：硬件计数器同时可读的数量有限",
        "doc": "ProfilingGuide", "section": "2.2.3", "section_title": "Replay",
        "level": "A",
        "quote": ("Depending on which metrics are to be collected, kernels might need to be "
                  "replayed one or more times, since not all metrics can be collected in a "
                  "single pass. For example, the number of metrics originating from hardware "
                  "(HW) performance counters that the GPU can collect at the same time is "
                  "limited. In addition, patch-based software (SW) performance counters can "
                  "have a high impact on kernel runtime and would skew results for HW counters."),
        "must_contain": ("not all metrics can be collected in a single pass",
                         "the GPU can collect at the same time is limited",
                         "would skew results for HW counters"),
    },
    {
        "id": "F2", "title": "software-patched metric 被放在独立 replay pass 里收集",
        "doc": "ProfilingGuide", "section": "2.2.6", "section_title": "Overhead",
        "level": "A",
        "quote": ("Software-patched metrics incur the highest overhead, as they require the "
                  "kernel to be modified and all patched instructions to execute additional "
                  "code. The exact overhead depends on the number and type of software-patched "
                  "metrics being collected. It is important though that software-patched "
                  "metrics are collected in separate replay passes and therefore don’t "
                  "interfere with the correctness or precision of other metrics. Their values "
                  "are also independent of any overhead they may cause."),
        "must_contain": ("Software-patched metrics incur the highest overhead",
                         "software-patched metrics are collected in separate replay passes",
                         "don’t interfere with the correctness or precision"),
    },
    {
        "id": "F3", "title": "比值型指标（hit rate = hits/queries）的分子分母可能来自不同 pass",
        "doc": "ProfilingGuide", "section": "2.3.8", "section_title": "Range and Precision",
        "level": "A",
        "quote": ("Out-of-range metrics often occur when the profiler replays the kernel launch "
                  "to collect metrics, and work distribution is significantly different across "
                  "replay passes. A metric such as hit rate (hits / queries) can have "
                  "significant error if hits and queries are collected on different passes and "
                  "the kernel does not saturate the GPU to reach a steady state (generally > "
                  "20 µs). Similarly, it can show unexpected values when the workload is "
                  "inherently variable, as e.g. in the case of spin loops."),
        "must_contain": ("hit rate (hits / queries) can have significant error",
                         "collected on different passes",
                         "generally > 20 µs",
                         "in the case of spin loops"),
    },
    {
        "id": "F4", "title": "Kernel Replay 第一遍保存 kernel 可访问的全部显存，之后恢复被写过的那部分",
        "doc": "ProfilingGuide", "section": "2.2.3", "section_title": "Replay",
        "level": "A",
        "quote": ("In Kernel Replay, all metrics requested for a specific kernel instance in "
                  "NVIDIA Nsight Compute are grouped into one or more passes. For the first "
                  "pass, all GPU memory that can be accessed by the kernel is saved. After the "
                  "first pass, the subset of memory that is written by the kernel is "
                  "determined. Before each pass (except the first one), this subset is restored "
                  "in its original location to have the kernel access the same memory contents "
                  "in each replay pass."),
        "must_contain": ("all GPU memory that can be accessed by the kernel is saved",
                         "grouped into one or more passes",
                         "this subset is restored in its original location"),
    },
    {
        "id": "F5", "title": "只跑一遍时完全不做 save/restore",
        "doc": "ProfilingGuide", "section": "2.2.3", "section_title": "Replay",
        "level": "A",
        "quote": ("As explained in Overhead, the time needed for this increases the more memory "
                  "is accessed, especially written, by a kernel. If NVIDIA Nsight Compute "
                  "determines that only a single replay pass is necessary to collect the "
                  "requested metrics, no save-and-restore is performed at all to reduce "
                  "overhead."),
        "must_contain": ("only a single replay pass is necessary",
                         "no save-and-restore is performed at all to reduce overhead"),
    },
    {
        "id": "F6", "title": "ncu 默认清空所有 GPU cache（replay pass 之间）；nsys 单遍、不清",
        "doc": "ProfilingGuide", "section": "2.3.8", "section_title": "Range and Precision",
        "level": "A",
        "quote": ("Cache control: Nsight Compute flushes all GPU caches by default between "
                  "replay passes. Nsight Systems collects data only in a single pass, and "
                  "therefore also does not flush any caches. If the workload is highly sensitive "
                  "to cache state, it’s recommended to use --replay-mode application "
                  "--cache-control none for ncu to let the application handle priming the caches "
                  "implicitly."),
        "must_contain": ("Nsight Compute flushes all GPU caches by default between replay passes",
                         "Nsight Systems collects data only in a single pass",
                         "--replay-mode application --cache-control none"),
    },
    {
        "id": "F7", "title": "ncu 默认锁 SM 时钟到 boost；nsys 不锁",
        "doc": "ProfilingGuide", "section": "2.3.8", "section_title": "Range and Precision",
        "level": "A",
        "quote": ("Clock control: Nsight Compute locks SM clocks by default. Nsight Systems does "
                  "not lock clocks. When comparing results, we recommend to lock clocks with "
                  "nvidia-smi externally before profiling and use --clock-control none for ncu."),
        "must_contain": ("Nsight Compute locks SM clocks by default",
                         "Nsight Systems does not lock clocks",
                         "--clock-control none for ncu"),
    },
    {
        "id": "F8", "title": "ncu 串行化 kernel launch；nsys 不串行化",
        "doc": "ProfilingGuide", "section": "2.3.8", "section_title": "Range and Precision",
        "level": "A",
        "quote": ("Serialization: Nsight Compute serializes kernel launches, unless a dedicated "
                  "replay mode is used. Nsight Systems does not serialize launches. Concurrent "
                  "kernels on the same device can lead to differences in the reported duration "
                  "between the two tools."),
        "must_contain": ("Nsight Compute serializes kernel launches",
                         "Nsight Systems does not serialize launches",
                         "differences in the reported duration between the two tools"),
    },
    {
        "id": "F9", "title": "workload duration 从不在 software-patching 的 pass 里收集",
        "doc": "ProfilingGuide", "section": "2.3.8", "section_title": "Range and Precision",
        "level": "A",
        "quote": ("Metrics collected using software-patching introduce significant overhead "
                  "during the workload execution. For this reason, the workload duration is "
                  "never collected in such a replay pass."),
        "must_contain": ("introduce significant overhead during the workload execution",
                         "the workload duration is never collected in such a replay pass"),
    },
    {
        "id": "F10", "title": "metric 超出逻辑范围时故意不 clamp，以保证报告其余部分自洽",
        "doc": "ProfilingGuide", "section": "2.3.8", "section_title": "Range and Precision",
        "level": "A",
        "quote": ("In general, measurement values that lie outside the expected logical range of "
                  "a metric can be attributed to one or more of the below root-causes. If values "
                  "are exceeding such range, they are not clamped by the tool to their expected "
                  "value on purpose to ensure that the rest of the profiler report remains "
                  "self-consistent."),
        "must_contain": ("outside the expected logical range of a metric",
                         "they are not clamped by the tool to their expected value on purpose",
                         "to ensure that the rest of the profiler report remains self-consistent"),
    },
    {
        "id": "F11", "title": "stall 样本归因给消费者指令（等在依赖上的那条），不是生产者",
        "doc": "ProfilingGuide", "section": "2.4.7", "section_title": "Warp Stall Reasons",
        "level": "A",
        "quote": ("Interpretation guidance: selected and not_selected are scheduler states, not "
                  "true stalls. The fundamental optimization target is increasing issue-slot "
                  "utilization, not driving every stall counter to zero. Stall reductions "
                  "usually translate to speedups only when low issue activity makes the kernel "
                  "latency-limited. Warp stall samples are attributed to the consumer instruction "
                  "that is waiting on a dependency, not the producer; use source and SASS context "
                  "to find the root cause."),
        "must_contain": ("selected and not_selected are scheduler states, not true stalls",
                         "Warp stall samples are attributed to the consumer instruction that is "
                         "waiting on a dependency, not the producer"),
    },
    {
        "id": "F12", "title": "官方反模式：目标不是把 stall 计数器压到 0，而是提高 issue-slot 利用率",
        "doc": "ProfilingGuide", "section": "2.4.7", "section_title": "Warp Stall Reasons",
        "level": "A",
        "quote": ("Interpretation guidance: selected and not_selected are scheduler states, not "
                  "true stalls. The fundamental optimization target is increasing issue-slot "
                  "utilization, not driving every stall counter to zero. Stall reductions "
                  "usually translate to speedups only when low issue activity makes the kernel "
                  "latency-limited. Warp stall samples are attributed to the consumer instruction "
                  "that is waiting on a dependency, not the producer; use source and SASS context "
                  "to find the root cause."),
        "must_contain": ("increasing issue-slot utilization, not driving every stall counter to "
                         "zero",
                         "Stall reductions usually translate to speedups only when low issue "
                         "activity makes the kernel latency-limited"),
    },
    {
        "id": "F13", "title": "PM sampling 的 Workload Execution 行：ncu 与 nsys 用同一方法测时间戳",
        "doc": "ProfilingGuide", "section": "2.3.8", "section_title": "Range and Precision",
        "level": "A",
        "quote": ("Note that for Workload Execution rows in the PM Sampling timeline, Nsight "
                  "Compute and Nsight Systems use the same method to measure start and end "
                  "timestamps."),
        "must_contain": ("Workload Execution rows in the PM Sampling timeline",
                         "use the same method to measure start and end timestamps"),
    },
    {
        "id": "F14", "title": "同名不同指：2.4.7 与 2.4.8 是两个不同的 metric",
        "doc": "ProfilingGuide", "section": "2.4.8", "section_title": "Warp Stall Reasons (Not Issued)",
        "level": "A",
        # 2.4.7 与 2.4.8 各出一句：**同一个指标名，两套采样条件**（P9 的教科书实例）
        "quote": ("Collected using warp scheduler state sampling. They are incremented regardless "
                  "if the scheduler issued an instruction in the same cycle or not. These metrics "
                  "have instance values mapping from the function address (uint64) to the number "
                  "of samples (uint64)."),
        "must_contain": ("Collected using warp scheduler state sampling",
                         "They are incremented regardless if the scheduler issued an instruction "
                         "in the same cycle or not"),
    },
    {
        "id": "F15", "title": "ncu CLI 默认值：--cache-control = all；--clock-control = boost",
        "doc": "NsightComputeCli", "section": "5.4.4", "section_title": "Profile",
        "level": "A",
        "quote": ("cache-control Control the behavior of the GPU caches during profiling. "
                  "Allowed values: all: All GPU caches are flushed before each kernel replay "
                  "iteration during profiling."),
        "must_contain": ("All GPU caches are flushed before each kernel replay iteration during "
                         "profiling",
                         "Control the behavior of the GPU caches during profiling"),
    },
)

# ★ F14 的**第二条**逐字引文（2.4.7 与 2.4.8 的**对比**要用到，但只能有 15 条 FACTS）：
#   来源：ProfilingGuide §2.4.8（抓取 2026-09-21）。
#   ★★ **同名不同指**（P9）：两节的第一个句子**完全相同**（都是
#   `Collected using warp scheduler state sampling.`），**第二个句子方向相反**。
F14_DEFINITIONS = (
    {
        "metric_section": "2.4.7", "metric_name": "Warp Stall Reasons",
        "increment_rule": ("They are incremented regardless if the scheduler issued an "
                           "instruction in the same cycle or not."),
        "doc": "ProfilingGuide", "section": "2.4.7", "level": "A",
    },
    {
        "metric_section": "2.4.8", "metric_name": "Warp Stall Reasons (Not Issued)",
        "increment_rule": ("They are incremented only on cycles in which the warp scheduler "
                           "issued no instruction."),
        "doc": "ProfilingGuide", "section": "2.4.8", "level": "A",
    },
)
# ★ 两节共享的**同一句**（同名不同指的「同名」这一半的证据）
F14_SHARED_SENTENCE = "Collected using warp scheduler state sampling."

# ===========================================================================
# 3. F15 的默认值与允许值
# ===========================================================================
# ★ 来源：`NsightComputeCli` §5.4.4 Profile（抓取 2026-09-21）。
#   该页的表格每个选项**三个 `<td>`**：选项名 / 描述 / **默认值**。
#   控制者本轮实测：`cache-control` 行的 td[2] = `all`，`clock-control` 行的 td[2] = `boost`。
#   ⚠ **选项名在表里不带 `--` 前缀** —— 见模块头的「抽取纪律」。
F15_DEFAULTS = {
    "cache-control": "all",       # td[2] 逐字
    "clock-control": "boost",     # td[2] 逐字
}
F15_ALLOWED = {
    "cache-control": ("all", "none"),
    "clock-control": ("base", "boost", "force-boost", "none", "reset"),
}
# ★ `boost` 条目的逐字（同页）
F15_BOOST_VERBATIM = ("GPC and memory clocks are locked to their respective turbo boost "
                      "frequency during profiling.")
F15_BASE_VERBATIM = ("GPC and memory clocks are locked to their respective base (rated tdp) "
                     "frequency during profiling.")
# ★ `all` 条目的逐字（同页，与 FACTS["F15"]["quote"] 的 must_contain 互为印证）
F15_CACHE_ALL_VERBATIM = ("All GPU caches are flushed before each kernel replay iteration "
                          "during profiling.")

# ===========================================================================
# 4. 上游契约（**已建成域写好的前向声明**，本主题的存在理由）
# ===========================================================================
# ★ 来源：计划 §0.7 B，控制者逐字核实到行号。这些**不是引用值**，是仓库内部的事实
#   （由编号程序的 D 类断言直接核对同目录外的文件在场时的原文；不在场时 SKIP）。
#
# ★★ 本主题（`bottleneck-diagnosis`）的立足点**与同域 `profiling-tools` 不同**：
#   那一个的指派句在 `10/03:187`（「微基准与 profiler 的分工」），
#   **本主题的指派句在 `10/03:303`**（「把整卡级的峰值与**具体 kernel 的实测**做对比」）
#   —— 计划 §0.7 B 的表格里**这两行是分开的两行**（v1 曾把它们写成同一条，已订正）。
# ★★ 锚点形态：**前缀锚定**（2026-09-21 订正，两条）。
#   锚点要证明的是「**这一行确实是那句话**」，**不是**「这一行逐字永不改变」。
#   上游的**跨域链接**会**指向更具体的文档**（`50-performance` Chunk 4 就这么改过：
#   `[50-performance/bottleneck-diagnosis]` → `[…/bottleneck-diagnosis/01-latency-and-bandwidth]`）
#   —— 旧写法把**链接的文件名部分**也算进引文，于是**改动恰好落在被引用的那一行上**，
#   断言被打断（`D.5` 红），而**被引用的那句话一个字没变**。
#   ⇒ 引文**截到域名为止**（`正文在 [50-performance/bottleneck-diagnosis`）：
#     整段**可变部分**（尾部的 `]`、`/01-…`、`](url)`）都排除在锚点之外。
#   ★ 为什么截在域名末尾而**不是**再往前截到纯散文：域名就是这条断言的**落点**
#     （「指名指派**给本域**」）—— 去掉它，锚点就不再证明任何**跨域**的事。
#   ★ 为什么够长、不退化成 P5：它 40 字符、含一条具体路径；把域名换成
#     `profiling-tools`（或把「正文在」删掉）定向变异立刻 D.5 红。
UPSTREAM_CLAIMS = (
    {
        "path": "10-foundations/performance-models/03-measuring-roofline.md", "line": 303,
        "quote": "正文在 [50-performance/bottleneck-diagnosis",
        "why": "★ 本主题的**指名指派**：拿到数之后判「卡在哪」的正文在本域。"
               "★ 锚点**止于域名**（不含链接的文件名与 URL）—— 见本表上方的前缀锚定注",
    },
    {
        "path": "45-kernel-lifecycle/convergence-and-tuning/03-convergence.md", "line": 177,
        "quote": "需要用 `50-performance` 的工具定位",
        "why": "`45/03` 把「用工具定位」让渡给本域，自己只讲判据",
    },
    {
        "path": "45-kernel-lifecycle/convergence-and-tuning/03-convergence.md", "line": 256,
        "quote": "（本域只给判据，工具属 `50-performance`）",
        "why": "同上：收益转到 host / 调度 / 搬运时，定位手段属本域",
    },
    {
        "path": "10-foundations/performance-models/03-measuring-roofline.md", "line": 187,
        "quote": "微基准与 profiler 的分工是：前者给出**边界**，后者给出**当前位置**。",
        "why": "本主题 `01` 的 `§3` 用的正是「微基准给边界」这一半",
    },
)

# ===========================================================================
# 5. 本机实测值的**测量条件**（版本即是测量条件的一部分）
# ===========================================================================
# ★ 来源：本机实测（`sysctl` / `python3 -c`，2026-09-21）。
MACHINE = {
    "cpu": "Apple M1 Pro", "cores": 10,
    "l1d_bytes": 65536, "l2_bytes": 4194304, "l3_bytes": None,
    "cacheline_bytes": 128,          # ★ 不是 x86 的 64 B
    "mem_bytes": 17179869184,
    "python": "3.13.5",
    "numpy": "2.4.4",
    "clang": "/usr/bin/clang",
    "sample": "/usr/bin/sample",
    "has_gpu_tools": False,          # ncu / nsys / nvprof / nvidia-smi / CUPTI / perf 全无
}

# ===========================================================================
# 6. 控制者记录的**依赖链曲线**（计划 §0.7 **D-2**）与 8 路展开（**D-3**）
# ===========================================================================
# ★ 来源：计划 §0.7 D-2 / D-3，控制者在本机真跑（`clang -O2`，规模固定 64 MB）。
#
# ★★ **这两组数不是同一段代码，不得并列比较**（计划 §0.7 D-3 的警告）：
#   · `DEP_RECORDED`  ← `microbench.DEP_SRC`（**runtime-K 内层循环**）
#   · `UNROLL8_RECORDED` ← `microbench.UNROLL8_SRC`（**8 路写死的展开**，无内层循环）
#   两张表**各自都是真测量**，但「DEP 的 K=8」与「UNROLL8 的 8 路」不是同一个配置。
#   ⇒ 正文引用时必须写明是哪一段代码。这正是 P13（量法口径错）的正面教材。
DEP_RECORDED = {
    # K: (耗时 ms, 有效带宽 GB/s, 相对 K=1)
    1:  (29.734, 2.26,  1.00),
    2:  (21.678, 3.10,  1.37),
    4:  (14.758, 4.55,  2.01),
    8:  (3.692,  18.18, 8.05),
    16: (2.680,  25.04, 11.09),
    32: (1.779,  37.72, 16.71),
}
UNROLL8_RECORDED = {
    # 规模: 有效带宽 GB/s
    64 * 1024:   64.0,
    4 * 1024 * 1024: 64.0,
    8 * 1024 * 1024: 58.39,
    64 * 1024 * 1024: 51.28,
    256 * 1024 * 1024: 51.04,
}
# ★ 计划 §0.7 D-3 的结论（**实施者必须独立复现同一个量级，不是抄这个数**）：
#   「L1/L2 命中与 DRAM 之间只差 **1.25×**」⇒ 本机 cache 层次效应太弱，**不能当教学实验**。
UNROLL8_CACHE_RATIO_RECORDED = 1.25

# ===========================================================================
# 7. `03_diagnosis_walkthrough.py` 的实验参数与**记录值**
# ===========================================================================
# ★ 来源：实施者在本机真跑（`numpy 2.4.4` / `python3 3.13.5`）。**绝对值不可照抄**：
#   它们是**本机某一次运行的产物**，随运行时序变化；程序只对**结构与量级**下断言。
DIAG_N_ELEMS = 1 << 20          # 8 MiB doubles —— **超过本机 L2（4 MiB）**
DIAG_BYTES = DIAG_N_ELEMS * 8
DIAG_REPS = 15                  # 每个版本测几次（**交替编排**，见程序内注释）
DIAG_WARMUP_ROUNDS = 2          # 前两轮丢掉（`45/01` 讲 warmup，正文在那边）
DIAG_FAST_INNER = 64            # ★ 短操作：一次测量内重复 R 次再除以 R（见程序内注释）
DIAG_QUICK = {"reps": 7, "warmup_rounds": 1, "fast_inner": 32}
DIAG_RECORDED = {
    "slow_median_s": 0.1697,        # Python 循环版（中位）
    "fast_median_s": 0.000179,      # 向量化版（8 MiB / 0.186 ms = 45 GB/s）
    # ★ 下面两个区间是**用稳健 σ（1.4826×MAD）**得到的 `|Δ|/δ*` 的实测散布：
    #   实施者在本机连续跑 7 次，真实改动落在 20.3–69.6，安慰剂改动落在 0.03–0.39。
    #   ⚠ 若把 σ 换成**样本标准差**，同一批数据的比值会掉到 3.5（σ 被尾部定义，
    #     见 `§0.7` D-8/D-10）—— 那正是 `03` 里 `robust_sigma()` 那段注释的理由。
    "sham_gap_ratio_range": (0.03, 0.39),
    "real_gap_ratio_range": (20.3, 69.6),
    "dep_k1_s_at_8mb": 0.003688,    # 微基准：延迟限（K=1）
    "dep_k48_s_at_8mb": 0.000170,   # 微基准：带宽限（K=32）
}

# ===========================================================================
# 8. δ* 的系数 —— **从 `45` 代入，本域不重推**（`CONTRIBUTING` §7 的「一处正文」）
# ===========================================================================
# ★ 来源：`45-kernel-lifecycle/convergence-and-tuning/02-significance.md:71` 逐字
#   `= 2.801585218112968 · σ · √(2/n)`；同主题 `code/specs.py:419` 定义为
#   `MDE_COEFFICIENT_AT_POWER_80`。**来源等级 [E]（本库推导值）**：
#   它由 `z_{0.975} + z_{0.80} = 1.9599639845400534 + 0.8416212335729144` 算出。
#
# ★★ 本模块**不放推导**，只放**转写下来的那个常数**。而「转写对不对」由**两道**检查钉住：
#   ① `mde_coefficient_selfcheck()`：常数必须等于 `MDE_ANCHORS` 引文里那个数
#      —— **不需要外部文件在场**，所以 `make falsify` 把常数改错时它一定会叫；
#   ② `mde_anchor_check(repo_root)`：`MDE_ANCHORS` 的每句在**上游原文里逐字在场**
#      （上游不在场时返回 `None` ⇒ SKIP）。
#   这正是本域对 P7（引文与断言作用域不匹配）的处置：**引文与它的出处钉在一起**。
MDE_COEFFICIENT_AT_POWER_80 = 2.801585218112968

MDE_ANCHORS = (
    {"path": "45-kernel-lifecycle/convergence-and-tuning/02-significance.md", "line": 71,
     "quote": "= 2.801585218112968 · σ · √(2/n)"},
    {"path": "45-kernel-lifecycle/convergence-and-tuning/02-significance.md", "line": 108,
     "quote": "n = 2 · (2.801585218112968 · σ / δ)²"},
    {"path": "45-kernel-lifecycle/convergence-and-tuning/code/specs.py", "line": 419,
     "quote": "MDE_COEFFICIENT_AT_POWER_80 = 2.801585218112968"},
)


def mde_coefficient_literals():
    """从 `MDE_ANCHORS` 的引文里**抽出**那个常数（**不去外部文件也能核**）。

    ★ 为什么要有这一半：`mde_anchor_check()` 在**上游不在场时整体 SKIP**，
      那时把 `MDE_COEFFICIENT_AT_POWER_80` 改成 `4.0` 就**没人管**了。
      这一半把「常数」与「写在本文件里的引文」钉在一起 —— 改任一侧都会立刻失败。
    """
    import re
    out = []
    for a in MDE_ANCHORS:
        out += [float(m) for m in re.findall(r"\d+\.\d+", a["quote"])]
    return out


def mde_coefficient_selfcheck():
    """返回 `(是否一致, 从引文里抽出的那些数)`。**不依赖外部文件。**"""
    lits = mde_coefficient_literals()
    ok = bool(lits) and all(abs(x - MDE_COEFFICIENT_AT_POWER_80) < 1e-12 for x in lits)
    return ok, lits


def mde_anchor_check(repo_root):
    """核 `MDE_ANCHORS` 的每一句在**上游原文里逐字在场**。返回 `[(标签, n, ≥1)]`。

    ★ 上游文件不在场则返回 `None`（调用方 SKIP）。
    """
    rows = []
    for a in MDE_ANCHORS:
        p = os.path.join(repo_root, a["path"])
        if not os.path.exists(p):
            return None
        lines = open(p, encoding="utf-8").read().splitlines()
        hit = 1 if (1 <= a["line"] <= len(lines)
                    and a["quote"] in lines[a["line"] - 1]) else 0
        rows.append(("%s:%d" % (a["path"].split("/")[-1], a["line"]), hit, 1))
    return rows


def mde_delta_star(sigma, n):
    """`δ* = 2.801585218112968 · σ · √(2/n)` —— **从 `45/02` 代入，本域不重推**。

    ★ 检定力 80%、双侧 α = 0.05 的两组比较。正文在
      `45-kernel-lifecycle/convergence-and-tuning/02-significance`（此处只回链）。
    """
    import math
    return MDE_COEFFICIENT_AT_POWER_80 * sigma * math.sqrt(2.0 / n)


def mde_required_n(sigma, delta):
    """`n = 2 · (2.801585218112968 · σ / δ)²` —— 要让 δ 可判定所需的**每组**样本数。"""
    return 2.0 * (MDE_COEFFICIENT_AT_POWER_80 * sigma / delta) ** 2


# ===========================================================================
# 9. 文档探测（不写死构造者本机的目录布局；相对项以**本文件所在目录**为基准）
# ===========================================================================
SPECS_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_candidates(extra_env=None):
    """把 `DOC_CANDIDATES` 解析成**绝对路径**列表。

    ★ 相对项以 `SPECS_DIR`（**本文件所在目录**）为基准，**不是 cwd**
      —— `30-gpu-programming/cutlass` 的教训：用 `os.path.abspath()` 会让
      「从哪儿调用」影响「找到哪棵树」。`$PERF_DOC_DIR` / `~` 例外。
    """
    env = os.environ.get(DOC_ENV, "") if extra_env is None else extra_env
    out = []
    for c in DOC_CANDIDATES:
        if c == "$PERF_DOC_DIR":
            if env:
                out.append(os.path.abspath(env))
            continue
        if c.startswith("~"):
            out.append(os.path.expanduser(c))
        elif os.path.isabs(c):
            out.append(c)
        else:
            out.append(os.path.join(SPECS_DIR, c))
    return tuple(out)


def find_doc_dir(verbose=False):
    """返回第一个**含齐两份文档**的候选目录；找不到返回 `None`。"""
    for cand in resolve_candidates():
        if all(os.path.exists(os.path.join(cand, d["file"])) for d in DOCS.values()):
            if verbose:
                print("  文档目录 = %s" % cand)
            return cand
    if verbose:
        print("  文档目录 = **未找到**（将 SKIP 逐字核对）")
        for cand in resolve_candidates():
            print("    候选 %s ⇒ %s" % (cand, "在" if os.path.isdir(cand) else "不在"))
    return None


def doc_status(doc_dir=None):
    """逐份核 sha256/字节数，返回 `{docname: ("ok"|"bad"|"absent", 详情)}`。

    ★ `bad`（文件在但对不上）**必须让程序失败**；`absent` 才是 SKIP。
    """
    out = {}
    for name, d in DOCS.items():
        path = os.path.join(doc_dir, d["file"]) if doc_dir else None
        if not path or not os.path.exists(path):
            out[name] = ("absent", None)
            continue
        raw = open(path, "rb").read()
        got = hashlib.sha256(raw).hexdigest()
        if got != d["sha256"] or len(raw) != d["bytes"]:
            out[name] = ("bad", "bytes=%d sha256=%s" % (len(raw), got[:16]))
        else:
            out[name] = ("ok", "bytes=%d sha256=%s" % (len(raw), got[:16]))
    return out


def strip_tags(raw_bytes):
    """把 HTML 抽成纯文本（与 F1–F15 的 `quote` 同一口径）。

    ★★ **不能把标签一律替换成空格。** 那会让 `single <em>pass</em>.` 变成
      `single pass .`（多一个空格），于是 15 条引文里有 7 条被**假报不在场**。
      正确做法：**先按 `<p>` 切段落，再在段内把标签删成【空串】**，最后折叠空白。
    """
    import html as _html
    import re as _re
    t = raw_bytes.decode("utf-8", "replace")
    t = _re.sub(r"<script.*?</script>", " ", t, flags=_re.S | _re.I)
    t = _re.sub(r"<style.*?</style>", " ", t, flags=_re.S | _re.I)
    paras = []
    for m in _re.finditer(r"<p[^>]*>(.*?)</p>", t, flags=_re.S):
        s = _re.sub(r"<[^>]+>", "", m.group(1))          # ★ 段内删标签 = 删成空串
        s = _re.sub(r"\s+", " ", _html.unescape(s)).strip()
        if s:
            paras.append(s)
    return " ".join(paras)


def verify_docs(doc_dir):
    """对**每一份在场且 sha256 正确的**文档，逐字核 F1–F15 与 2.4.8 的引文。

    返回 `[(标签, 出现次数, 期望 ≥1), …]`。文档不在场时返回 `None`（调用方 SKIP）。
    """
    rows = []
    texts = {}
    for name, d in DOCS.items():
        p = os.path.join(doc_dir, d["file"])
        if os.path.exists(p):
            texts[name] = strip_tags(open(p, "rb").read())
    for f in FACTS:
        t = texts.get(f["doc"], "")
        rows.append(("《%s》%s 引文逐字在场" % (f["doc"], f["id"]), t.count(f["quote"]), 1))
    for d in F14_DEFINITIONS:
        t = texts.get(d["doc"], "")
        rows.append(("《%s》%s 的增量规则逐字在场" % (d["doc"], d["metric_section"]),
                     t.count(d["increment_rule"]), 1))
    t = texts.get("ProfilingGuide", "")
    rows.append(("《ProfilingGuide》两节共享句逐字在场（同名的一半）",
                 t.count(F14_SHARED_SENTENCE), 2))
    t = texts.get("NsightComputeCli", "")
    rows.append(("《NsightComputeCli》boost 条目逐字在场", t.count(F15_BOOST_VERBATIM), 1))
    rows.append(("《NsightComputeCli》base 条目逐字在场", t.count(F15_BASE_VERBATIM), 1))
    return rows


# ===========================================================================
# 10. 自洽不变量（**文档不在场时**仍然钉得住的那一半）
# ===========================================================================
def fact_invariants():
    """F1–F15 的**结构不变量**：每条返回 `(标签, 观测值, 期望值)`。

    ★ 为什么要有这一半：`verify_docs()` 在文档不在场时整体 SKIP，
      于是 `FACTS` 的 15 条引文会**没有任何断言钉住**。这一半补的就是那个缺口：
      **`must_contain` 必须是 `quote` 的子串**，改任一侧都会立刻失败。
    ★ 这一半**不能**替代 sha256 + 逐字核对。
    """
    ids = tuple(f["id"] for f in FACTS)
    rows = [
        ("F1–F15 恰好 15 条", len(FACTS), 16 - 1),
        ("编号恰为 F1..F15 且顺序不乱",
         ids, tuple("F%d" % i for i in range(1, 16))),
        ("来源等级集合恰为 {'A'}（厂商官方文档）",
         tuple(sorted({f["level"] for f in FACTS})), ("A",)),
        ("每条的 doc 都在 DOCS 里",
         tuple(sorted({f["doc"] for f in FACTS})), tuple(sorted(DOCS))),
        ("每条的 section 形如 N.N.N",
         all(f["section"].count(".") == 2 for f in FACTS), True),
        ("每条的 quote 非空且长度 ≥ 60",
         all(len(f["quote"]) >= 60 for f in FACTS), True),
        ("每条的 must_contain 非空", all(f["must_contain"] for f in FACTS), True),
    ]
    bad = [(f["id"], m) for f in FACTS for m in f["must_contain"] if m not in f["quote"]]
    rows.append(("★ 每条 must_contain 都是它自己 quote 的子串（改任一侧立刻失败）",
                 len(bad), 0))
    rows.append(("F14 的两套增量规则**方向相反且不同**",
                 F14_DEFINITIONS[0]["increment_rule"] != F14_DEFINITIONS[1]["increment_rule"],
                 True))
    rows.append(("F14 的两节共享同一句「Collected using warp scheduler state sampling.」",
                 F14_SHARED_SENTENCE in FACTS[13]["quote"], True))
    rows.append(("★ 本主题 §2 并置的两个 metric 名**互不相同**（`(Not Issued)` 后缀）",
                 (F14_DEFINITIONS[0]["metric_name"] != F14_DEFINITIONS[1]["metric_name"],
                  F14_DEFINITIONS[1]["metric_name"].endswith("(Not Issued)")),
                 (True, True)))
    rows.append(("★ 两条增量规则里「是否发射指令」的取向相反",
                 ("regardless if the scheduler issued" in
                  F14_DEFINITIONS[0]["increment_rule"],
                  "issued no instruction" in F14_DEFINITIONS[1]["increment_rule"]),
                 (True, True)))
    rows.append(("F10 的引文逐字含「not clamped … on purpose」",
                 "they are not clamped by the tool to their expected value on purpose"
                 in FACTS[9]["quote"], True))
    rows.append(("F11/F12 的引文逐字含「consumer instruction」与「issue-slot utilization」",
                 ("consumer instruction" in FACTS[10]["quote"],
                  "issue-slot" in FACTS[11]["quote"]),
                 (True, True)))
    return rows


def upstream_check(repo_root):
    """核 `UPSTREAM_CLAIMS` 的每一句在**原文里逐字在场**。返回 `[(标签, n, ≥1)]`。

    ★ 找不到时返回 `None`（SKIP）。
    """
    rows = []
    for c in UPSTREAM_CLAIMS:
        p = os.path.join(repo_root, c["path"])
        if not os.path.exists(p):
            return None
        lines = open(p, encoding="utf-8").read().splitlines()
        hit = 0
        if 1 <= c["line"] <= len(lines):
            if c["quote"] in lines[c["line"] - 1]:
                hit = 1
        rows.append(("%s:%d 逐字在场" % (c["path"].split("/")[-1], c["line"]), hit, 1))
    return rows


REPO_ROOT = os.path.abspath(os.path.join(SPECS_DIR, "..", "..", ".."))

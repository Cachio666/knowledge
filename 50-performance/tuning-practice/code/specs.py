#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""specs.py —— 非编号辅助模块：**外部事实常量**（`50-performance/tuning-practice`）。

★ 它是「非编号支撑代码」，不计入「文档数 = 编号程序数」的核对（`CONTRIBUTING` §2）。

===============================================================================
★★ 本主题的测量条件 —— 逐处都要带，不许省
===============================================================================

按 `CONTRIBUTING` §2 的四类口径，本主题的数字分成**两类，标签必须分开**：

| 类 | 本主题里是什么 | 必须写明的限定语 |
|---|---|---|
| **本机实测值** | **CPU 上的真实测量**：`time.perf_counter` 的量化步长与抖动、重复次数曲线、A/B 编排的对照 | `python3` 版本 / `numpy` 版本 / macOS |
| **规格值** | **NVIDIA 官方文档事实**（`FACTS` 的 F1–F15） | **URL + 章节号 + 抓取日 + 文档页脚 last-updated** |

**两者不得互证**（`CONTRIBUTING:90` 与域 README）。

★ **本机没有 GPU、没有 `ncu`/`nsys`/`nvprof`/`nvidia-smi`/`CUPTI`/`perf`。**
  本主题**一切实测都是 CPU 上的**，**不支撑任何关于 GPU 的结论**。

===============================================================================
★★ 本主题的立足点：`45` 声明为「表外输入」的 σ 与 τ（**逐字**）
===============================================================================

`45-kernel-lifecycle/convergence-and-tuning` 三篇的「闭合性检查」一段，各有一句
把 σ/tau 说成**必须测、不能查**的输入。**那三句话就是本主题的存在理由。**

| 来源 | 行 | 逐字 |
|---|---|---|
| `01-benchmark-hygiene.md` | **262** | 「**唯一的表外输入**是「本次会话的稳态 σ」——它必须**测**（同一批测量自身就能给），不能查。」 |
| `02-significance.md` | **246** | 「**表外输入**是「本批测量的 `σ` 与 `τ`」——它们必须测（材料从同一批数据来）。」 |
| `03-convergence.md` | **260** | 「**唯一表外输入**是「本机当前的 `σ` 与 `n`」——必须测。」 |

★ **`45/02:246` 是唯一同时点到 `τ` 的那一句**，而它只说「必须测（材料从同一批数据来）」——
  **没有说 `τ` 是什么、也没有说怎么测**。本模块的 `TAU_Q_*` 就是补这一格。

★★ 另一处**主动让渡**：`60-compilers/auto-tuning/code/mini_autotune.py:610-612` 逐字冻结了
  测量噪声（`MEASURE_NOISE = 0.0`），把「测量方法论」交接给本主题 —— 见 `UPSTREAM_CLAIMS`。

===============================================================================
★★ ★★ 本主题最该引以为戒的一次 P13：τ_q 的真值是 41.6667 ns，不是 250 ns
===============================================================================

`time.get_clock_info("perf_counter").resolution` 在**本机**返回
`4.166666666666666e-08` 秒 ⇒ **41.6667 ns**（mach 时基 24 MHz，125/3 ns）。

**计划初稿把「量化步长」写成 250 ns，那是错的。** 250 ns 是**同一个循环里相邻两次
迭代的间隔**（含循环体自身的开销），**不是计时器的刻度**。

  | 标签 | 错值 | 实测真值 |
  |---|---|---|
  | 「量化步长」 | 250 ns | **41.6667 ns**（`get_clock_info().resolution`） |
  | 「相邻迭代间隔」的中位 | （同一个数被当成两件事） | 本机实测 **125 ns**（3 tick）——**它随循环体变化，不是刻度** |
  | 「最小非零间隔」 | 84 ns | **≈ 1 tick**（实测 41.9 ns） |

**判据**：若「步长」真是 250 ns，**「最小非零」不可能只有 84 ns** ——
**两个数互相矛盾**。这正是 P13 判据第②句（「这个数荒谬不荒谬」在此处就是「它内部一致吗」）。
★ `TAU_Q_WITHDRAWN` 与 `TAU_Q_RECORDED` 都留在本文件里，**就是为了让这个对照可复核**。

===============================================================================
★★ 抽取纪律（沿用 `50-performance` 另两个主题；**不是本主题发明的**）
===============================================================================

HTML 标签会**切断字节连续串**。`to their <em>boost</em> value` 使
`raw.count(b"boost value")`（字节口径）= **0** —— 据此下结论「文档里没有这句话」是假发现。

★ 去标签时把标签**一律替换成空格**，会让 `single <em>pass</em>.` 变成 `single pass .`
  （**中间多一个空格**），于是逐字引文被**假报「不在场」**。
  ⇒ 正确做法：**先按 `<p>` 切段落，再在段内把标签删成【空串】**。

⇒ **抽取纪律的完整形式是三条**：① 先确认锚点在**字节上连续**；② 用**正确的方式**去标签；
  ③ 在**原始字节**上计数。**三条缺一，都会产出「看起来很正常的假结论」。**

===============================================================================
★ 路径探测（文档在场时做逐字核对，不在场时**明确 SKIP**，不静默通过）
===============================================================================

候选列表见 `DOC_CANDIDATES`，**相对项以本文件所在目录为基准**（不是 cwd）。
`doc_status()` 返回三态：`"ok"` / `"bad"`（**必须失败**）/ `"absent"`（**SKIP**）。
**`absent` 不等于通过**：编号程序把它打印成 `[SKIP]` 并**单独计数**。
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
# 1. 两份 NVIDIA 文档（A 级来源）—— 与同域另两个主题同源、同一个 sha256
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
# ★★ 本块与 `50-performance/profiling-tools/code/specs.py`、
#    `50-performance/bottleneck-diagnosis/code/specs.py` 的 FACTS **同源、机械搬入**
#    （同一份文档、同一个 sha256、同一次抓取）——**不是重新转写**。
#    转写是独立的出错环节（`CONTRIBUTING:280`），所以这里**逐字复制**而不是重打一遍。
# ★ 每条的结构：
#   id / title        —— 计划 §0.7 C 的编号与标题
#   doc / section     —— 来源：URL 见 DOCS[doc]["url"]
#   quote             —— **逐字引文**（从原始 HTML 的 `<p>` 段落去标签后得到，未经人工转写）
#   must_contain      —— 用来核 `quote` 的短锚（**必须是 `quote` 的子串**，由 fact_invariants() 核）
#   level             —— 来源等级（本表恒为 "A"）
#
# ⚠ 每个 `quote` 的字符是**从 HTML 去标签后**得到的；`’` 是原文的 U+2019，不是 ASCII 撇号。
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
F14_SHARED_SENTENCE = "Collected using warp scheduler state sampling."

# ===========================================================================
# 3. F15 的默认值与允许值
# ===========================================================================
# ★ 来源：`NsightComputeCli` §5.4.4 Profile（抓取 2026-09-21）。表里每个选项**三个 `<td>`**：
#   选项名 / 描述 / **默认值**。⚠ **选项名在表里不带 `--` 前缀**。
F15_DEFAULTS = {
    "cache-control": "all",       # td[2] 逐字
    "clock-control": "boost",     # td[2] 逐字
}
F15_ALLOWED = {
    "cache-control": ("all", "none"),
    "clock-control": ("base", "boost", "force-boost", "none", "reset"),
}
F15_BOOST_VERBATIM = ("GPC and memory clocks are locked to their respective turbo boost "
                      "frequency during profiling.")
F15_BASE_VERBATIM = ("GPC and memory clocks are locked to their respective base (rated tdp) "
                     "frequency during profiling.")
F15_CACHE_ALL_VERBATIM = ("All GPU caches are flushed before each kernel replay iteration "
                          "during profiling.")

# ===========================================================================
# 4. 上游契约（**已建成域写好的前向声明**，本主题的存在理由）
# ===========================================================================
# ★ 来源：计划 §0.7 B 与 §1.2，控制者逐字核实到行号。
#   这些**不是引用值**，是仓库内部的事实（编号程序的 C 类断言核它们在场时的原文；
#   不在场时 SKIP）。
#
# ★★ 本主题（`tuning-practice`）的立足点**与同域另两个主题都不同**：
#   · `profiling-tools` 的指派句在 `10/03:187`（微基准与 profiler 的分工）；
#   · `bottleneck-diagnosis` 的指派句在 `10/03:303`（整卡峰值 vs 具体 kernel 实测）；
#   · **本主题的立足点是「45 把 σ 与 τ 当作表外输入、明说必须测」这三句话**
#     + `60/auto-tuning` 显式冻结测量噪声（`MEASURE_NOISE = 0.0`）让出的那一块。
UPSTREAM_CLAIMS = (
    {
        "path": "45-kernel-lifecycle/convergence-and-tuning/01-benchmark-hygiene.md",
        "line": 262,
        "quote": "**唯一的表外输入**是「本次会话的稳态 σ」——它必须**测**（同一批测量自身就能给），不能查。",
        "why": "★ 本主题 `01` 的**存在理由**之一：σ 必须测 —— 本篇就是那个「怎么测」",
    },
    {
        "path": "45-kernel-lifecycle/convergence-and-tuning/02-significance.md",
        "line": 246,
        "quote": "**表外输入**是「本批测量的 `σ` 与 `τ`」——它们必须测（材料从同一批数据来）。",
        "why": "★ **唯一同时点到 `τ` 的一句**；它没说 `τ` 是什么、也没说怎么测（本主题补这一格）",
    },
    {
        "path": "45-kernel-lifecycle/convergence-and-tuning/03-convergence.md",
        "line": 260,
        "quote": "**唯一表外输入**是「本机当前的 `σ` 与 `n`」——必须测。",
        "why": "★ `45/03` 的停止判据也把 σ 与 n 当输入；`n` 的选取正是本主题 `01` 的核心",
    },
    {
        "path": "60-compilers/auto-tuning/code/mini_autotune.py",
        "line": 612,
        "quote": "MEASURE_NOISE = 0.0",
        "why": "★ **主动让渡**：auto-tuning 域显式冻结测量噪声，把「测量方法论」交接给本主题",
    },
    {
        "path": "60-compilers/auto-tuning/03-search-algorithms.md",
        "line": 255,
        "quote": "怎么测量、怎么定位瓶颈（测量方法论在上游）",
        "why": "★ `60` 逐字把「怎么测量」指到本主题",
    },
    {
        "path": "60-compilers/tvm/04-auto-scheduling.md",
        "line": 225,
        "quote": "属于测量方法论 → `50-performance/tuning-practice`",
        "why": "同上，`60/tvm` 也把「怎么测」指到本主题",
    },
)

# ★ `mini_autotune.py` 里那句注释的**上面两行**（逐字），是「为什么冻结」的理由，
#   与 `MEASURE_NOISE = 0.0` 那一行**一起**才构成让渡（单看那一行不知道它为什么被冻结）。
MEASURE_NOISE_RATIONALE = (
    "# ★ 冻结：本迷你地形**关闭测量噪声**。真实调优里 do_bench 的中位数本身有抖动，",
    "#   把它加进来会让「模型误导搜索」与「测量噪声」两件事混在一起，无法归因。",
)

# ===========================================================================
# 5. 本机实测值的**测量条件**（版本即是测量条件的一部分）
# ===========================================================================
# ★ 来源：本机实测（`sysctl` / `python3 -c` / `time.get_clock_info`，2026-09-21）。
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
# 6. ★★ τ：真正该量的那个量（计划 §0.7 **D-4** 的撤回与订正）
# ===========================================================================
# ★ 来源：本机实测 `time.get_clock_info("perf_counter")`，2026-09-21。
#   `implementation = 'mach_absolute_time()'`，`resolution = 4.166666666666666e-08` s。
TAU_Q_RECORDED_NS = 41.666666666666664      # = 125/3 ns，mach 时基 24 MHz
TAU_Q_RECORDED_S = 4.166666666666666e-08
# ★ **已撤回的错值**（计划 v1 初稿写的「量化步长 = 250 ns」）。留在源码里，
#   是为了让「它和「最小非零 84 ns」互相矛盾」这件事**可被程序当场复核**。
TAU_Q_WITHDRAWN_NS = 250.0
# ★ 那个 250 ns **真正是什么**：本机实测的**相邻迭代间隔**（同一个循环里连续两次
#   `perf_counter()` 之差）。它在**这个循环体**下是 208 ns（= 4.992 tick）附近；
#   换一个更紧的循环体则是 125 ns（= 3.000 tick）。
#   **⇒ 它随循环体变化，所以它是「迭代的节奏」，不是「计时器的刻度」。**
TAU_ITER_SPACING_RECORDED_NS = {"median": 208.0, "most_common": (208.0, 167.0)}
# ★ 本机实测（本主题实施者复跑，更紧的循环体）：
#   相邻迭代间隔的**中位 125.03 ns**（= 3 tick）、**最小 41.91 ns**（≈ 1 tick）。
#   ★ 「最小非零 ≈ 1 tick」这一条**反证**了「步长 = 250 ns」：若步长真是 250 ns，
#     最小非零间隔不可能只有 41.9 ns。
TAU_ITER_MEDIAN_OBSERVED_NS = 125.0
TAU_ITER_MIN_OBSERVED_NS = 41.9

# ★ ★ `timeit.timeit("pass", number=100000)` 平均到每次 = **5.9 ns**。
#   ⚠ **它不是「计时器的有效精度」**，而是**被测操作（循环体）自身的成本** ——
#     **两者是不同的量**（P9 同名不同指）。计划 v1 把这两者混为一谈，已订正。
TIMEIT_PASS_RECORDED_NS = 5.9

# ===========================================================================
# 7. ★★ N（重复次数）本身会改变你测出的「单次耗时」（计划 §0.7 **D-4b**）
# ===========================================================================
# ★ 来源：计划 §0.7 D-4b（控制者真跑）—— 一个固定开销的短操作，测「N 次的总耗时 ÷ N」，
#   每个 N 取 60 次中位数。**t/N 对 N 呈 U 形。**
U_CURVE_RECORDED_NS = {
    # N: t/N (ns)
    1: 375.0, 2: 333.5, 5: 316.6, 10: 304.1, 20: 297.9, 50: 293.3,
    100: 292.1, 200: 291.9, 500: 294.2, 1000: 297.6, 2000: 296.0, 5000: 316.2,
}
U_FIT_RECORDED = {
    # 拟合区间: (a 固定开销 ns, b 单次耗时 ns)
    "N=0..100": (106.5, 291.1),
    "N=0..20": (89.8, 293.9),
    "N=100..5000": (-13476.0, 316.6),      # ★ 负截距 ⇒ 该区间不满足线性模型
}
U_EMPTY_LOOP_RECORDED_NS = 84.0            # N=0 的空循环（= 两次计时读数的固定开销）
U_N1_VS_VALLEY_RECORDED_PCT = 28.5         # 计划记的「N=1 比谷底高 28.5%」
U_N5000_VS_VALLEY_RECORDED_PCT = 8.3       # 计划记的「N=5000 比谷底高 8.3%」
# ★★ **本主题实施者的复跑结论（与计划不同，见 01 文档 §3.2 与本文档末尾的「偏差」）**：
#   左端偏差**复现且更强**（N=1 比平台高约 37%），
#   右端在 N∈[500, 5000] 上**只有约 +2% 的回升**（不是 8.3%），且在低重复下**会被噪声淹没**；
#   用**交替采样 + 每点 151 次重复**才稳定测到正值（4/4 次为正，+1.7% ~ +4.6%）。
U_RISE_REPLICATED_PCT_RANGE = (1.7, 4.6)
U_N1_VS_PLATEAU_OBSERVED_PCT = 36.8        # 本机复跑：375.1 vs 274.2 ns
U_FIT_OBSERVED_A_RANGE_NS = (84.9, 106.5)  # 本机复跑的固定开销 a

# ===========================================================================
# 8. ★★ σ：噪声是重尾的，尖峰在时间上均匀分布（计划 §0.7 **D-8**）
# ===========================================================================
# ★ 来源：计划 §0.7 D-8（控制者真跑）：`clang -O2`，10 线程打满 CPU，主线程重复
#   同一份固定工作量 60000 次（总时长 6.87 s）。
NOISE_RECORDED = {
    "median_ms": 0.0730, "min_ms": 0.0720, "max_ms": 43.9210, "max_over_median": 601.66,
    "over_1.2x_median_pct": 7.852, "over_1.5x_median_pct": 3.475, "over_2.0x_median_pct": 2.323,
    "over_3.0x_median_pct": 1.573, "over_5.0x_median_pct": 1.127,
    "thirds": (686, 697, 702),             # ★ 尖峰按轮次三等分 ⇒ **时间上均匀，不是漂移/热降频**
}
# ★ 计划 §0.7 D-9（R1 的答案）：**几秒**尺度上本机**测不到热降频**
#   （120 轮 ±1.4%、无单调趋势、只有一个孤立尖峰）。R1 已据此处置。

# ===========================================================================
# 9. ★★ σ 可能不是一个好的刻画量（计划 §0.7 **D-10**）
# ===========================================================================
# ★ 来源：计划 §0.7 D-10（控制者真跑）：10000 次测量，每次 = 「100 次 op() 重复求平均」。
DIST_RECORDED = {
    "min_ns": 1916.670, "p50_ns": 1971.250, "p90_ns": 2062.910, "p99_ns": 2786.670,
    "p999_ns": 7157.500, "max_ns": 71277.920,
    "max_over_median": 36.159, "mean_over_median": 1.0256,
    "sigma_over_median_pct": 45.031, "skewness": 62.799, "excess_kurtosis": 4469.541,
    "p90_dev_pct": 4.6,
}
# ⚠ **口径边界**：那个实验测的是**一个约 2 µs 的操作**。
#   **不得外推成「所有性能测量的 σ 都不可用」** —— 长 kernel 的相对噪声远小于此。
#   这一条只作为「先用本方法看一眼自己的分布」的理由。

# ===========================================================================
# 10. ★★ A/B 编排：顺序 vs 交错（计划 §0.7 **D-12**）
# ===========================================================================
# ★ 来源：计划 §0.7 D-12（控制者真跑）：A 与 B **逐字等价**（真实差异严格为 0），
#   每组 N=3000，取中位，各重复 8 轮。
AB_RECORDED = {
    "sequential_pct": (1.340, 0.536, 1.001, 0.369, 0.133, 0.725, 2.455, 1.290),
    "interleaved_pct": (-0.020, -0.050, 0.043, 0.042, -0.077, -0.015, -0.016, 0.039),
    "seq_median_pct": 0.863, "seq_mean_pct": 0.981,
    "int_median_pct": -0.015, "int_mean_pct": -0.007,
    "seq_sign_consistency": (8, 8),         # 计划记：8/8 为正
    "ratio": 57.0,                          # 计划记：约 57×
    # ★ 两条【否定】结论：
    "interference_effect": "构造的周期干扰（后台线程每 100 ms 忙 20 ms）**造不出可观测效应**",
    "warmup_rounds_pct": (1.042, -2.239, 0.108, -0.992),   # 预热后顺序编排仍在波动
}
# ★★ **本主题实施者的复跑结论（与计划的「8/8 同号」不同，见 03 文档 §3.3）**：
#   交错把误差**压小一个数量级**这件事**稳健复现**（12 轮各跑一次：顺序 |x| 均值 0.456%、
#   交错 0.034%，**13.4×**），
#   **但顺序编排并不稳定地偏向同一个符号**（12 轮里只有 5/12 为正、中位 −0.081%）。
#   ⇒ 正确的机制表述是「**两个时间段各自带走一段系统状态，于是差值里混进一个时段差**」，
#     不是「顺序编排系统性地偏高」。计划那句「8/8 轮同号」**在本机不复现**。
AB_OBSERVED = {
    "seq_abs_mean_pct": 0.456, "int_abs_mean_pct": 0.034, "ratio": 13.4,
    "seq_sign_positive": (5, 12), "int_sign_positive": (5, 12),
    "seq_median_pct": -0.081, "int_median_pct": -0.004,
    "seq_range_pct": 2.066, "int_range_pct": 0.163, "range_ratio": 12.7,
}

# ===========================================================================
# 11. δ* 的系数 —— **从 `45` 代入，本域不重推**（`CONTRIBUTING` §7 的「一处正文」）
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
# 12. 文档探测（不写死构造者本机的目录布局；相对项以**本文件所在目录**为基准）
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
      `single pass .`（多一个空格），于是逐字引文被**假报不在场**。
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
# 13. 自洽不变量（**文档不在场时**仍然钉得住的那一半）
# ===========================================================================
def fact_invariants():
    """F1–F15 的**结构不变量** + **本主题自己的三个订正不变量**。

    返回 `[(标签, 观测值, 期望值), …]`。

    ★ 为什么要有 F 那一半：`verify_docs()` 在文档不在场时整体 SKIP，
      于是 `FACTS` 的 15 条引文会**没有任何断言钉住**。这一半补的就是那个缺口：
      **`must_contain` 必须是 `quote` 的子串**，改任一侧都会立刻失败。

    ★★ 为什么还要有 τ / σ / N 的那一半：本主题的核心结论**不是**规格引文，
      而是**本机实测的机制**。它必须能被「改坏就失败」钉住 ——
      尤其是 **τ_q = 41.6667 ns 而不是 250 ns** 这一条（本域最该引以为戒的一次 P13）。
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

    # ---- ★★ 本主题自己的三条订正不变量（**不依赖任何外部文件**）----
    # ① τ_q 的真值（= get_clock_info().resolution 的记录）**必须**与已撤回的 250 ns 不同
    rows.append(("★ τ_q 的记录值 ≠ 已撤回的错值 250 ns（P13 的撤回必须是**可见的**）",
                 (TAU_Q_RECORDED_NS != TAU_Q_WITHDRAWN_NS,
                  abs(TAU_Q_RECORDED_NS - 125.0 / 3.0) < 1e-9),
                 (True, True)))
    # ② 「最小非零间隔」与「步长 = 250 ns」互相矛盾 —— 这一条是本域那次 P13 的**判据**
    rows.append(("★ 「最小非零间隔」< 已撤回的 250 ns（两个数**不可能同时成立**）",
                 TAU_ITER_MIN_OBSERVED_NS < TAU_Q_WITHDRAWN_NS, True))
    # ③ 「相邻迭代间隔」与 τ_q 是**两个不同的量**（P9 同名不同指）
    rows.append(("★ 「相邻迭代间隔」≥ 2 × τ_q（它不是计时器的刻度，随循环体变化）",
                 TAU_ITER_MEDIAN_OBSERVED_NS >= 2.0 * TAU_Q_RECORDED_NS, True))
    # ④ t/N 曲线的**两端**都存在（谷底既不在 N=1 也不在 N=5000）
    uc = U_CURVE_RECORDED_NS
    rows.append(("★ D-4b 记录的 t/N 曲线：N=1 > 谷底 且 N=5000 > 谷底（两端都偏高）",
                 (uc[1] > min(uc.values()), uc[5000] > min(uc.values())), (True, True)))
    rows.append(("★ D-4b 的谷底严格在 (1, 5000) 的内部",
                 1 < min(uc, key=uc.get) < 5000, True))
    # ⑤ D-10 的分布是**右偏且尖峰**的（正态的偏度与超额峰度都是 0）
    rows.append(("★ D-10 记录：偏度 > 0 且超额峰度 > 0（正态两者都是 0）",
                 (DIST_RECORDED["skewness"] > 0, DIST_RECORDED["excess_kurtosis"] > 0),
                 (True, True)))
    rows.append(("★ D-10 记录：σ/中位 ≫ (p90−p50)/p50（σ 被尾部定义）",
                 DIST_RECORDED["sigma_over_median_pct"] >
                 3.0 * DIST_RECORDED["p90_dev_pct"], True))
    # ⑥ A/B 编排：**真实差异为 0**，而顺序编排给出的 |差| 是交错的一个数量级以上
    rows.append(("★ D-12 记录：顺序编排的 |中位| 是交错的 20 倍以上",
                 abs(AB_RECORDED["seq_median_pct"]) >
                  20.0 * abs(AB_RECORDED["int_median_pct"]), True))
    rows.append(("★ D-12 的两条【否定】结论都在 `AB_RECORDED` 里（干扰造不出效应 / 不是冷启动）",
                 ("造不出可观测效应" in AB_RECORDED["interference_effect"],
                  len(AB_RECORDED["warmup_rounds_pct"]) == 4),
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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""specs.py —— 非编号辅助模块：**外部事实常量**（`50-performance/profiling-tools`）。

★ 它是「非编号支撑代码」，不计入「文档数 = 编号程序数」的核对（`CONTRIBUTING` §2）。

===============================================================================
★★ 本主题的测量条件 —— 逐处都要带，不许省
===============================================================================

按 `CONTRIBUTING` §2 的四类口径，本主题的数字分成**两类，标签必须分开**：

| 类 | 本主题里是什么 | 必须写明的限定语 |
|---|---|---|
| **本机实测值** | **CPU 上的真实测量**：`cProfile` 的调用计数与时间、`sample` 的调用图、`perf_counter` 的分辨率、C 微基准的编译/运行结果 | `python3` 版本 / `clang` 版本 / macOS |
| **规格值** | **NVIDIA 官方文档事实**（`FACTS` 的 F1–F15） | **URL + 章节号 + 抓取日 + 文档页脚 last-updated** |

**两者不得互证**（`CONTRIBUTING:90` 与域 README）：
规格事实只能支撑「工具是怎么设计的」，**不能支撑「你这台机器的性能」**——后者本机没有。

★ **本机没有 GPU、没有 `ncu`/`nsys`/`nvprof`/`nvidia-smi`/`CUPTI`。**
  本主题**一切实测都是 CPU 上的**，**不支撑任何关于 GPU 的结论**。

===============================================================================
★★ 来源等级（沿用 `CONTRIBUTING` §2 的 A–E）
===============================================================================

  · **A 级：厂商官方文档** —— `FACTS` 的 F1–F15 全部来自 `docs.nvidia.com` 的
    Nsight Compute 官方文档。**A 级支撑「NVIDIA 规定/记述该工具是这样」**，
    仍**不支撑**「你在本机看到的现象与它一致」（那是本机实测值那一类的事）。
  · **E 级：本库推导** —— 本模块**不含推导值**（`CONTRIBUTING` §2：
    推导在编号程序内部当场算出，不搬进 `specs.py`）。
  · **D 级：0 条。** 本主题不引用任何社区项目文档。

★★ **`# 来源：` 的机器可核形式就是 `FACTS[*]` 的 `doc` + `section` 两个字段**：
  · **`doc` + `section`** 出处 —— 文档本体不在仓库里，所以由
    `verify_docs()`（文档在场时）在**原始字节**上核 sha256 + 逐字核 `quote`；
    **文档不在场时**由 `fact_invariants()` 做**自洽核对**（见下）。
  · **`level`** 来源等级 —— 由编号程序断言整张表的取值集合恰好是 `{'A'}`。
  · **仓库里没有任何一份 NVIDIA 文档的副本**，本库也**不复刻**它。

===============================================================================
★★ 抽取纪律（本域特有，写计划时差点栽在这上面）
===============================================================================

HTML 标签会**切断字节连续串**。本模块自己踩到的一手实例（控制者本轮实测）：

  · 在 `NsightComputeCli` 页的**原始字节**上
    `raw.count(b"--cache-control")` = **0**，而 `raw.count(b"cache-control")` = **1**。
    原因：选项名在表格的第一列里写作 `cache-control`（**不带 `--` 前缀**），
    `--` 只出现在别处的正文里。
  · `ProfilingGuide` 页里那句话的原始 HTML 是 `to their <em>boost</em> value`，
    因此 `raw.count(b"boost value")`（字节口径）= **0**，
    而 `raw.count(b"limit GPU clock frequencies")` = 1。

**若据此下结论「文档里没有这句话」，就是一次完全的假发现。**
正确做法是**先找连续锚点，再打印它周围的原始字节**看真实结构。
**「在原始字节上计数」（`CONTRIBUTING:280`）必须与「先确认锚点在字节上是连续的」配对使用。**

⇒ 本模块 `FACTS[*]["must_contain"]` 的每一条都是**短而无歧义、且在字节上连续**的字面量；
   `must_contain` 还必须**是 `quote` 的子串**——这一条由 `fact_invariants()` 当场核，
   它把「引文」与「用来核引文的锚」钉在一起（**两者不一致时程序立刻失败**）。

===============================================================================
★ 本模块的路径探测（文档在场时做逐字核对，不在场时**明确 SKIP**，不静默通过）
===============================================================================

候选列表见 `DOC_CANDIDATES`，**相对项以本文件所在目录为基准**（不是 cwd —— 见
`30-gpu-programming/cutlass` 的教训）。`doc_status()` 返回三态：

  · `"ok"`    —— 文件在、字节数与 sha256 都对；
  · `"bad"`   —— 文件在，但 sha256/字节数对不上（**必须让程序失败**，不是 SKIP）；
  · `"absent"`—— 文件不在（**SKIP**，并在程序输出里说明怎么拿到它）。

★ **`absent` 不等于通过**：编号程序把它打印成 `[SKIP]` 并**单独计数**，
  不计入「通过」的条数（`CONTRIBUTING` 的 SKIP 口径）。
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
#   doc / section     —— 来源：URL 见 DOCS[doc]["url"]，章节号**控制者本轮在原始 HTML 上核过**
#   quote             —— **逐字引文**（转写自原始 HTML 的 `<p>` 段落，未经人工转写）
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
#   （由 `A` 类断言直接核对同目录外的文件在场时的原文；不在场时 SKIP）。
UPSTREAM_CLAIMS = (
    {
        "path": "10-foundations/performance-models/03-measuring-roofline.md", "line": 187,
        "quote": "**第四层：用 profiler 的计数器校准。**",
        "why": "`10-foundations/03` 造好了尺子却从未使用（§5 全篇留空）——本域是用尺子的人",
    },
    {
        "path": "45-kernel-lifecycle/convergence-and-tuning/01-benchmark-hygiene.md", "line": 60,
        "quote": ("本主题**不出现任何 kernel 实现代码**，也不重复讲 `50-performance` "
                  "已覆盖的 profiling 工具。"),
        "why": "`45/01` 明确把 profiling 工具让渡给本域",
    },
    {
        "path": "45-kernel-lifecycle/convergence-and-tuning/01-benchmark-hygiene.md", "line": 262,
        "quote": ("**唯一的表外输入**是「本次会话的稳态 σ」——它必须**测**"
                  "（同一批测量自身就能给），不能查。"),
        "why": "★ 本域最硬的立足点：`45` 把 σ 当输入、明说「必须测不能查」，却不教怎么测",
    },
    {
        "path": "45-kernel-lifecycle/convergence-and-tuning/02-significance.md", "line": 246,
        "quote": ("**表外输入**是「本批测量的 `σ` 与 `τ`」——它们必须测"
                  "（材料从同一批数据来）。"),
        "why": "同上，σ 与 τ 都是表外输入",
    },
    {
        "path": "45-kernel-lifecycle/convergence-and-tuning/03-convergence.md", "line": 260,
        "quote": "**唯一表外输入**是「本机当前的 `σ` 与 `n`」——必须测。",
        "why": "同上",
    },
)

# ===========================================================================
# 5. 本机实测值的**测量条件**（版本即是测量条件的一部分）
# ===========================================================================
# ★ 来源：本机实测（`python3 -c 'import sys,platform; ...'`，2026-09-21）。
MACHINE = {
    "cpu": "Apple M1 Pro", "cores": 10,
    "l1d_bytes": 65536, "l2_bytes": 4194304, "l3_bytes": None,
    "cacheline_bytes": 128,          # ★ 不是 x86 的 64 B
    "mem_bytes": 17179869184,
    "python": "3.13.5",
    "clang": "/usr/bin/clang",
    "sample": "/usr/bin/sample",
    "has_gpu_tools": False,          # ncu / nsys / nvprof / nvidia-smi / CUPTI / perf 全无
}

# ===========================================================================
# 6. 控制者记录的两组对照（计划 §0.7 D-11，**实施者已独立复现，绝对值随运行时序变化**）
# ===========================================================================
# ★ 来源：计划 §0.7 D-11(a)、(b)，控制者在本机真跑。
#   ⚠ **绝对值不可照抄**：实施者本轮复跑得到 3498968 次 / 0.684 s（控制者记录 3556568 / 0.690）。
#     两者的**结构完全一致**（cumtime = tottime + 子调用；wall-bounded 下插桩开销 ≈ 0）。
#     ⇒ 编号程序**只对结构下断言**，具体的调用次数只打印、不设等式
#     （与 §0.7 D-13 对 `sample` 计数采用的同一条纪律：**正文只写结论**）。
CONTROLLER_RECORDED = {
    # (a) 插桩 vs 采样（`01_collection_mechanisms.py` 复现）
    "instr_total_s": 0.690,          # cProfile 报告的总时间（= 两部分之和，**不是插桩开销**）
    "instr_spin_tottime_s": 0.314,
    "instr_spin_cumtime_s": 0.600,
    "instr_perf_counter_calls": 3556568,
    "instr_perf_counter_tottime_s": 0.286,
    "instr_compute_hot_tottime_s": 0.090,
    # (b) 跨 pass 组合（`02_replay_multipass.py` 复现）
    "hot_hits": 4, "hot_queries": 14,
    "cold_hits": 1, "cold_queries": 4100,
}

# ★ 定次数 vs wall-bounded 的插桩开销（计划 §1.2 §4；事实层审查者的实测区间）
#   来源：计划 §1.2（控制者已核实）。本程序实测落在同一区间内。
INSTR_OVERHEAD_FIXED_ITERS_PCT = (174.0, 276.0)   # 定次数循环下的 cProfile 开销区间
INSTR_OVERHEAD_WALL_BOUNDED_PCT = 0.1             # wall-bounded 循环下的实测值

# ===========================================================================
# 7. `02_replay_multipass.py` 的实验参数（**状态依赖工作负载**）
# ===========================================================================
# ★ 来源：本机实测的设计参数（复现计划 §0.7 D-11(b) 的四个值）。
#
#   **自旋查找**（F3 逐字点名的 `spin loop`）：消费者在**环形数组**上按路径位置推进，
#   每探一个位置记 **1 次 query**；探到的值等于目标键记 **1 次 hit**。
#   停止条件：路径位置越过「生产者写下的停止位置」，或 query 数达到 cap。
#
#   ★ **「目标在前 / 在后」说的是「探测路径上靠前 / 靠后」**：
#     冷 pass 的停止位置 4099 **越过数组末尾（4096）** ⇒ 消费者**自旋到第二圈**——
#     这正是「spin loop」的字面含义，也正是 F3 说「工作量本身可变」的那一类负载。
N_SLOTS = 4096                     # 环形数组长度
TARGET_KEY = 0xA5                  # 目标键

# (名称, 目标键出现的槽位, 生产者写下的停止路径位置, cap)
PASSES = (
    ("hot",  (3, 7, 10, 13), 13,   N_SLOTS),   # 目标在前；cap 大（= 整圈），不构成约束
    ("cold", (4095,),        4099, 4100),      # 目标在后；停止位置越过数组末尾
    ("tiny", (0,),           0,    1),         # 只探一次 —— 用来构造 > 100% 的组合
)

# ★ 超过 100% 的构造（计划 §0.7 D-11(b)）：**热 hits(4) / 最小 pass 的 queries(1) = 400%**。
#   ⇒ 物理上不可能，而这正是 F10「工具故意不 clamp」要你看到的形态。
#   程序**不做 clamp**（对 F10 的正面回答），只把它打印出来。

# ===========================================================================
# 8. `03_defaults_attribution.py` 的实验参数
# ===========================================================================
# ★ 来源：计划 §0.7 D-6（控制者真跑，实施者已独立复现）。
#   ⚠ **绝对值不可照抄**：控制者记录 26,504,981 / 24,390,308、组内极差 8.27% / 24.12%、
#     组间差 −7.98%；事实层审查者复跑得 30,486,666 / 27,535,466、27.24% / 35.76%、−9.68%；
#     实施者本轮复跑得 25,314,935 / 23,337,482、13.07% / 12.86%、−7.81%。
#     三次的**方向与结论完全一致**：**组内极差 ≥ 组间差 ⇒ 测不出可靠的观测者效应**。
#     ⇒ 程序只断言**结构**（组间差 ≤ 组内极差），不写死任何绝对值。
D6 = {
    "duration_s": 3.0,     # 子进程跑多久（满载计数循环）
    "reps": 7,             # 每组重复次数
    "sample_secs": 2,      # attach 时 `sample <pid> <N>` 的 N
    "attach_delay_s": 0.25,
}
# ★ 诊断工具（`make mutate` / `make falsify`）用这一档跑得更快；
#   **断言全是结构性的**，所以快档仍然走同一条代码路径。默认档才是正文引用的口径。
#   ⚠ **`reps` 不能太小**：`组内极差` 是**极差**（range），它随 `reps` 迅速变大，
#     而 `组间差` 是**中位数之差**，与 `reps` 无关。`reps=3` 时极差是**被低估**的，
#     会让 `C.2`（组间差 ≤ 组内极差）在「效果真实存在」时**假失败**。
#     ⇒ 快档取 5（实测 3 次里有 1 次假失败，5 次里 0 次）。
D6_QUICK = {"duration_s": 1.8, "reps": 5, "sample_secs": 1, "attach_delay_s": 0.2}

# ★ `sample` 报告的**结构性结论**（计划 §0.7 D-5 / D-13）：
#   正文引用时**只写结论、不写具体计数**（D-13：绝对计数随程序与采样窗口变化）。
SAMPLE_ATTRIBUTION_CONCLUSION = (
    "采样报告里最热的符号是**机器栈上的 C 函数**（`_PyEval_EvalFrameDefault` / "
    "`mach_absolute_time` / `PyObject_Vectorcall` 一类），**你写的 Python 函数一次都不出现**"
)

UNSOURCED = ()   # 本模块不含无来源的项


# ===========================================================================
# 8b. δ* 的系数 —— **从 `45` 代入，本域不重推**（`CONTRIBUTING` §7 的「一处正文」）
# ===========================================================================
# ★ 来源：`45-kernel-lifecycle/convergence-and-tuning/02-significance.md:71` 逐字
#   `= 2.801585218112968 · σ · √(2/n)`；同主题 `code/specs.py:419` 定义为
#   `MDE_COEFFICIENT_AT_POWER_80`。**来源等级 [E]（本库推导值）**：
#   它由 `z_{0.975} + z_{0.80} = 1.9599639845400534 + 0.8416212335729144` 算出。
#
# ★★ 本模块**不放推导**，只放**转写下来的那个常数**；而「转写对不对」由
#    `mde_anchor_check()` 在**上游文件在场时逐字核对**（A 类断言）。
#    这正是本域对 P7（引文与断言作用域不匹配）的处置：**引文与它的出处钉在一起**。
MDE_COEFFICIENT_AT_POWER_80 = 2.801585218112968

MDE_ANCHORS = (
    {"path": "45-kernel-lifecycle/convergence-and-tuning/02-significance.md", "line": 71,
     "quote": "= 2.801585218112968 · σ · √(2/n)"},
    {"path": "45-kernel-lifecycle/convergence-and-tuning/02-significance.md", "line": 108,
     "quote": "n = 2 · (2.801585218112968 · σ / δ)²"},
    {"path": "45-kernel-lifecycle/convergence-and-tuning/code/specs.py", "line": 419,
     "quote": "MDE_COEFFICIENT_AT_POWER_80 = 2.801585218112968"},
)


def mde_anchor_check(repo_root):
    """核 δ* 的系数在**上游原文里逐字在场**。返回 `[(标签, n, ≥1)]`；文件不在则 `None`。

    ★ 这一条把「本域代入的那个常数」与「`45` 原文里的那个数」**钉在一起** ——
      把 `MDE_COEFFICIENT_AT_POWER_80` 改错（例如改成 `4.0`），
      `03_defaults_attribution.py` 的 `A.9` 立刻失败。
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
    """`n = 2 · (2.801585218112968 · σ / δ)²` —— 要让 δ 可判定所需的**每组**样本数。

    ★ 同样是从 `45/02` 代入（该文 L108 逐字给出这一式），**本域不重推**。
    """
    return 2.0 * (MDE_COEFFICIENT_AT_POWER_80 * sigma / delta) ** 2


# ===========================================================================
# 9. 文档探测（不写死构造者本机的目录布局；相对项以**本文件所在目录**为基准）
# ===========================================================================
SPECS_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_candidates(extra_env=None):
    """把 `DOC_CANDIDATES` 解析成**绝对路径**列表。

    ★ 相对项以 `SPECS_DIR`（**本文件所在目录**）为基准，**不是 cwd**
      —— `30-gpu-programming/cutlass` 的教训：用 `os.path.abspath()` 会让
      「从哪儿调用」影响「找到哪棵树」。`$PERF_DOC_DIR`/`~` 例外，
      它们本身就是绝对或可展开的。
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

    ★ `bad`（文件在但对不上）**必须让程序失败** —— 那是「我们核的那份不是这份」，
      不是「没得核」。`absent` 才是 SKIP。
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

    ★ 这个函数是 `verify_docs()` 的一半；**另一半是 sha256** ——
      先确认「这是同一份字节」，再谈「这句话在不在里面」。

    ★★ **不能把标签一律替换成空格。** 本文件第一版就是那么写的，结果
      `verify_docs()` 对 15 条引文里的 **7 条**报「不在场」——
      **而它们全都在**。原因是 **HTML 标签会切断字节连续串**（本模块头已写明），
      而且切断的方式**不止一种**：

        · `be collected in a single <em>pass</em>.` —— 把标签换成空格会得到
          `... in a single  pass . ...`，于是 `pass.` 变成 `pass .`，
          **引文里的 `single pass.` 永远匹配不上**；
        · `As explained in <a ...>Overhead</a>,` —— 同理得到 `Overhead ,`；
        · `Cache control:` 在原始字节里**根本不连续**（选项名被标签切开）。

      ⇒ 正确做法是**先把 `<p>` 段落整块取出来，再在段内把标签删掉（替换成空串）**，
        最后统一折叠空白。段内删标签不会把相邻的词黏在一起——真正的词间空白
        在 HTML 里本来就是空白字符，不靠标签分隔。

      ★ 这与本模块头的「先找连续锚点，再打印它周围的原始字节」是同一条纪律的
        **第二半**：光「在原始字节上计数」还不够，**抽取函数本身也得先确认它没改变字面量**。
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
        n = t.count(f["quote"])
        rows.append(("《%s》%s 引文逐字在场" % (f["doc"], f["id"]), n, 1))
    for d in F14_DEFINITIONS:
        t = texts.get(d["doc"], "")
        rows.append(("《%s》%s 的增量规则逐字在场" % (d["doc"], d["metric_section"]),
                     t.count(d["increment_rule"]), 1))
    t = texts.get("ProfilingGuide", "")
    rows.append(("《ProfilingGuide》两节共享句逐字在场（同名的一半）",
                 t.count(F14_SHARED_SENTENCE), 2))
    t = texts.get("NsightComputeCli", "")
    rows.append(("《NsightComputeCli》boost 条目逐字在场",
                 t.count(F15_BOOST_VERBATIM), 1))
    rows.append(("《NsightComputeCli》base 条目逐字在场",
                 t.count(F15_BASE_VERBATIM), 1))
    return rows


# ===========================================================================
# 10. 自洽不变量（**文档不在场时**仍然钉得住的那一半）
# ===========================================================================
def fact_invariants():
    """F1–F15 的**结构不变量**：每条返回 `(标签, 观测值, 期望值)`。

    ★ 为什么要有这一半：`verify_docs()` 在文档不在场时整体 SKIP，
      于是 `FACTS` 的 15 条引文会**没有任何断言钉住** —— 把 `F3` 的 `20 µs`
      改成别的数，程序照样全过。这一半补的就是那个缺口：
      **`must_contain` 必须是 `quote` 的子串**，改任一侧都会立刻失败。
    ★ 这一半**不能**替代 sha256 + 逐字核对：它只证明「引文与它的锚一致」，
      **不证明「引文确实在 NVIDIA 的文档里」**——后者只有 `verify_docs()` 能给。
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
    rows.append(("★ F15 默认值 = all / boost",
                 (F15_DEFAULTS["cache-control"], F15_DEFAULTS["clock-control"]),
                 ("all", "boost")))
    rows.append(("★ F15 默认值必须落在各自的允许值集合里",
                 all(F15_DEFAULTS[k] in F15_ALLOWED[k] for k in F15_DEFAULTS), True))
    rows.append(("★ 两个默认值**方向相反**：all 让 kernel 冷启动、boost 让 kernel 跑最高频",
                 (F15_DEFAULTS["cache-control"] != "none",
                  F15_DEFAULTS["clock-control"] in ("boost", "force-boost")),
                 (True, True)))
    rows.append(("F3 的稳态阈值字面含「20 µs」",
                 "20 µs" in FACTS[2]["quote"], True))
    return rows


def upstream_check(repo_root):
    """核 `UPSTREAM_CLAIMS` 的每一句在**原文里逐字在场**。返回 `[(标签, n, ≥1)]`。

    ★ 仓库根之外的文件（`10-foundations/...`）相对**本文件**上溯三级；
      找不到时返回 `None`（SKIP）。
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

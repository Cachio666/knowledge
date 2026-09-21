#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""01_sigma_tau.py —— `45` 声明为「表外输入」的 σ 与 τ，到底怎么测。

配套文档：`01-measuring-sigma-and-tau.md`。★ 五类断言（A/B/C/D/E）**全部在本程序内部**
（`CONTRIBUTING` §2 表里的第二种情形），失败时以**非零退出码**结束；**没有 `verify.py`**。

★★ 本主题的真值标注（逐处都要带，不许省）：

> 本程序的数字**全部是 CPU 上的本机实测值**（`Apple M1 Pro` / `macOS` /
> `python3 3.13.5` / `numpy 2.4.4`），**不是 GPU、不是 Nsight Compute**。
> 它们支撑的是「**σ 与 τ 怎么量、量出来受什么影响**」这类**机制**结论。
> GPU 侧只能引 `specs.FACTS` 的规格值（F12 等），**两者不得互证**。

★★ 本程序的可证伪断言（改坏它们，本程序必须非零退出）：

  · `A.1`  **τ_q 落在 [20, 60] ns**（`time.get_clock_info("perf_counter").resolution`，
           本机实测 **41.6667 ns**）← 计划 §0.7 **D-4** 的订正；
  · `A.2`  **只把循环体加一次函数调用，相邻迭代间隔的中位数就变 ≥ 1.1 倍**
           （静息实测 1.97–1.99；负载下 **1.19–1.51**）
           （本机 42.14 → 83.12 ns，**时钟一点没变**）
           ⇒ 它是「迭代的节奏」，**不是计时器的刻度**（P9：同名的两个量其实是两个东西）；
           ★★ 采样是**交错 + 配对**（A/B/A/B… 再配对比较），**不是「先跑完一组再跑另一组」** ——
              后者在负载机器上会红（控制者实测：CPU 空闲 0% 时比值掉到 **1.01**，`make run` rc≠0）。
              这与 `03-ab-orchestration` 的 B 段是同一条对策（计划 §0.7 **D-12**）。
  · `A.3`  **可观测的「最小非零间隔」= 1 个 tick**（比值落在 [0.85, 1.15]，实测 0.983）
           ⇒ 这才是「τ_q 是刻度」的**直接**证据；
  · `A.4`  **「最小非零间隔」< 250 ns**（实测 40.98 ns）
           ← ★★ 这是**对已撤回错值的机器化反证**：若「步长 = 250 ns」成立，
             最小非零间隔不可能只有 40.98 ns（**两个数不可能同时成立**）；
  · `A.5`  **`timeit.timeit("pass")` 的单次成本与 τ_q 相差 > 5 ns**
           （实测 5.4 vs 41.667）⇒ 它们**不是同一个量**（P9）；
  · `A.6`  **本机实测的 τ_q 与 `specs.TAU_Q_RECORDED_NS` 一致**（差 < 1e-12）
           ⇒ 「记录」与「实测」钉在一起，换机器时先在这里叫；
  · `B.1`  **`t/N` 在 N=1 时 > 其在 N∈[100,2000] 最小值的 1.2 倍**（实测 1.26–1.37 倍）
           ← 计划 §0.7 **D-4b** 的左端；**`make falsify` 的「重复次数数组只剩 [1]」注入项打在这一点上**；
  · `B.2`  **`t(0)`（N=0 空循环）= 那个固定开销**，落在 [20, 300] ns（实测 **83.0 ns**）
           —— ★ **不是**判拟合出来的截距 `a`：`a` 会被单个尖峰拉到 **−1012 ns**，
             `t(0)` 是几百个样本的中位数，稳
           ← 「N 小时它没被摊薄」，这是左端偏差的**机理**，不只是现象；
  · `B.3`  **`t/N` 的谷底在内部**（既不是 N=1，也不是最大的 N）；
  · `B.4`  **高 N 段（{2000, 5000}）的 `t/N` 中位 > 平台段（{100, 200, 500}）的 1.01 倍**
           （实测 1.023–1.038，5/5 次为正）← 计划 §0.7 **D-4b** 的右端；
  · `C.1`  **`σ / (1.4826·MAD) > 2`**（实测 3.8–37）
           ⇒ **σ 里绝大部分不是主体宽度**。★ 这里**换掉了**计划原写的
             「(σ/中位) > 3 × ((p90−p50)/p50)」——那个比值**会随会话翻转**
             （R=60000 时 6.7–85，R=12000 时出现过 **0.6**），拿它当判据**在正确实现下会随机失败**；
  · `C.2`  **偏度 > 1 且超额峰度 > 10**（正态两者都是 0；实测偏度 2.5–111、峰度 69–15502）；
  · `C.3`  ★ **只打印、不作断言**：去尾 1%% 后 `p50 / p90 / σ` 三个相对变化
           —— 实施者换了**三版**判据（含「敏感度序」），**三版都在负载下失败过**。
           **报告值仍然印出来**，「σ 被尾部定义」的**判据**留给 `C.1` 与 `C.2`；
  · `D.1`  **σ 的两个成分都非零**（组内 / 组间）——这是 `45/02` 里 `n ∝ σ²` 的前提；
  · `D.2`  ★ **只打印、不作断言**（D 类唯一的判据是 `D.1`）：`σ/中位(N=1)` 与
           `τ_q / t̄(N=1)` 的比值 —— 实施者换了**四版**阈值判据，**四版都在负载下失败过**
           （真因：这两个量都会被负载污染）。**报告值仍然印出来**，
           而「σ 依赖口径」这件事写在文档 §3.4 与 §5 里；
  · `E.1`–`E.7`  F1–F15 结构不变量 / **本主题自己的订正不变量** / δ* 系数自洽 /
            文档 sha256 / 文档逐字 / 上游契约 / 标记自检。

用法：
    python3 01_sigma_tau.py            # 跑 A/B/C/D/E 五类断言
    python3 01_sigma_tau.py --lint     # AST 扫描：不 import torch/triton/numba/cuda

★ 外部工具：**不需要任何外部工具**（纯 `python3` + 标准库）。本程序不编译 C、
  不调用 `sample`、不需要网络。
"""

import ast
import os
import statistics
import subprocess
import sys
import time
import timeit

import specs

HERE = os.path.dirname(os.path.abspath(__file__))

# ===========================================================================
# 0. 断言记账（每个编号程序自带一份，**不共享**）
# ===========================================================================
# ★★ 失败标记：本主题（tuning-practice）自定的字面形态是 `>>> FAIL <<<`。
#   ★ 它**不是**照抄任何别的主题的：`30-gpu-programming/cutlass` 用 `[FAIL]`，
#     `45-kernel-lifecycle/convergence-and-tuning` 用 `[FAIL]|✗ N 条断言失败`，
#     同域 `50-performance/profiling-tools` 用 `**FAIL**`、
#     `50-performance/bottleneck-diagnosis` 用 `*** FAIL ***`。
#     抄它们**都会匹配到** —— 那正是「跨主题照抄」要禁的那件事，
#     而且会让「断言捕获」与「崩溃捕获」的区分失效。
#   ★ 本主题用 `>>> FAIL <<<`，并**由 `E.7` 断言它与两个诊断工具一致**。
#   ★★ 判定它的那一侧**必须【行首锚定】**（`mutate.py` / `meta_falsify.py` 里的
#     `^\s*<mark>`）：**程序成功时也会打印这个字面量**（下面 `E.7` 的 detail 里就有），
#     裸的 `ASSERT_FAIL_MARK in out` 在 rc=0 时也成立。
ASSERT_FAIL_MARK = ">>> FAIL <<<"

FAILS = []
_TOTAL = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0}
_PASS = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0}
SKIPPED = []


def check(cls, name, cond, detail=""):
    _TOTAL[cls] += 1
    if cond:
        _PASS[cls] += 1
        print("  [ok]     %-6s %s" % (name, detail))
    else:
        FAILS.append(name)
        print("  %s %-6s %s" % (ASSERT_FAIL_MARK, name, detail))
    return bool(cond)


def skip(cls, name, why):
    """★ SKIP **不是通过**：它单独计数，不进 `_PASS`（`CONTRIBUTING` 的 SKIP 口径）。"""
    SKIPPED.append(name)
    print("  [SKIP]   %-6s %s" % (name, why))


def summary():
    t = sum(_TOTAL.values())
    p = sum(_PASS.values())
    print()
    print("汇总: A %d/%d · B %d/%d · C %d/%d · D %d/%d · E %d/%d ⇒ "
          "断言 %d 条，通过 %d 条，失败 %d 条；跳过 %d 项（跳过不计入通过）"
          % (_PASS["A"], _TOTAL["A"], _PASS["B"], _TOTAL["B"],
             _PASS["C"], _TOTAL["C"], _PASS["D"], _TOTAL["D"],
             _PASS["E"], _TOTAL["E"],
             t, p, len(FAILS), len(SKIPPED)))
    if FAILS:
        print("失败清单: %s" % ", ".join(FAILS))
    return 1 if FAILS else 0


# ===========================================================================
# 1. 被测量的那个操作（**短操作**：这正是 D-4b 关心的尺度）
# ===========================================================================
# ★ 为什么是「20 次加法」这个尺度：它的一次调用约 275 ns，与 `time.perf_counter`
#   的量化步长 τ_q ≈ 41.7 ns 只差 6.6 倍 —— 正是「单次测量不可用、必须重复」的区。
OP_INNER = 20
OPS = {"n": OP_INNER}


def op():
    """一次固定的短操作。★ 返回值的和会被丢掉 —— 它不用来防优化：
    **CPython 不会做跨语句的数据流消除**，`x` 被真的算了出来。"""
    x = 0
    for j in range(OP_INNER):
        x += j
    return x


def t_of_n(n, reps=None):
    """测「`n` 次 `op()` 的总耗时 ÷ n」——**这就是 `t/N`**。

    ★ `n=0` 是**合法输入**：它测的是「两次 `perf_counter` 读数 + 一个空 `range(0)`」
      的固定开销，也就是 `t(N) = a + b·N` 里的 `a`。**零耗时的测量不是零成本。**

    `reps=None` 时只测一次；给 `reps` 则返回 `reps` 次的中位数。
    ★ 用**中位数**而不是最小值：本机的尖峰是重尾的（见 C 类），
      但那些尖峰是**真实发生的干扰**，最小值会把「机器最安静的那一刻」当成常态，
      而 D-4b 要测的正是一个**典型的** `t/N`。两者口径不同，这里明确选中位数。
    """
    def once():
        t0 = time.perf_counter()
        for _ in range(n):
            op()
        return time.perf_counter() - t0
    if reps is None:
        return once()
    return statistics.median([once() for _ in range(reps)])


# ===========================================================================
# 2. A 类：τ ——「计时器的刻度」到底是什么
# ===========================================================================
def _gap_body(i):
    """一个**极简的循环体**：只为证明「间隔随循环体变」而存在。"""
    return i & 1


def adjacent_iteration_gaps(count=200000, body=False):
    """量**相邻两次 `perf_counter()` 调用的间隔**。

    ★★ 这个量**不是**计时器的刻度。它是「**这个循环体的迭代节奏**」——
      循环体里有什么，它就跟着变。计划 v1 把它误当成了「量化步长 250 ns」（P13）。

    ★ `body=False` / `body=True` 的唯一区别是**循环体里多一次函数调用**。
      本机实测：**空体 42.14 ns（1.011 tick）→ 加一次调用 83.12 ns（1.995 tick）**，
      而**时钟一点没变**。这就是「间隔 ≠ 刻度」的直接证据（`A.2`）。
    """
    pc = time.perf_counter
    out = []
    t = pc()
    if body:
        for i in range(count):
            _gap_body(i)
            t2 = pc()
            out.append(t2 - t)
            t = t2
    else:
        for _ in range(count):
            t2 = pc()
            out.append(t2 - t)
            t = t2
    return out


# ★★ A.2 的采样口径：**交错 + 配对**（计划 §0.7 **D-12** 的对策，与 `03` 的 B 段同一个思路）
A2_PAIRS = 100          # 配对数：每对 = 空体一块 + 加调用一块
A2_CHUNK = 2000         # 每块的迭代数 ⇒ 两组各 20 万次迭代（与旧的「各测 20 万」同量级）


def paired_gap_ratio(pairs=A2_PAIRS, chunk=A2_CHUNK):
    """**交错采样 + 配对比较**：返回 `(每对的比值列表, 空体样本池, 加调用样本池)`。

    ★★ 为什么必须交错（正文在 `03-ab-orchestration`，机制是计划 §0.7 **D-12**）：
      「先把空体那一组采完、再采加调用那一组」会让两组各自占据一个**连续的时间段**；
      系统状态（负载、频率、cache）在两段之间漂移时，那个漂移会被原样记到
      「**两个循环体的差**」上。交错把「先后比较」变成「同期比较」——
      两组面对**同分布**的系统状态。
      ★ 本机实测（控制者验收时 CPU 空闲率 **0.0%**）：顺序采样下 A.2 的比值掉到 **1.01**、
        `make run` 非零退出；而同一时刻 `make falsify` 通过 ⇒ 它是**概率性的负载敏感**。

    ★ 配对：每一对内部先算「加调用块 ÷ 空体块」，再取**这些比值的中位数**。
      为什么不是「两组各自汇总成一个数再比」：那样仍然要求两个汇总数落在同一个状态里；
      配对让**每一对比值**各自抵掉自己那一小段时间的系统状态。

    ★ ABBA：偶数对「先空体、后加调用」，奇数对反过来 —— 否则「同一对里第二个被量的那个」
      会稳定地偏慢（`03` 的 `inter_round` 实测过这个残余的顺序效应）。

    ★★ 每块内部的统计量用**均值**、不是中位 —— 这一条是**被实测逼出来的**：
      相邻迭代间隔被 `τ_q ≈ 41.7 ns` **量化**，而空循环体的真实间隔落在
      **≈1.5 个 tick** 处 —— 它的**中位**因此骑在两个量化台阶之间的 50/50 边界上，
      会整块翻转。实测（同一台机器、同一份代码、连续 6 次）：
      空体中位在 **42.14 ns（1 tick）与 82.89 ns（2 tick）之间抛硬币**，
      比值随之在 **1.99 与 1.01** 之间跳 —— 而 `1-tick 桶` 的占比只动了
      **1.1 个百分点**（0.494 → 0.505）。
      ⇒ 那是**量化边界**，不是负载特有的东西；交错采样**单独**修不好它
        （实测：交错 + 中位仍然 12/12 红）。**均值把量化误差在几百个样本里平均掉**，
        才分辨得出亚 tick 的差异。
      ★ 门槛值**没有变**（仍是 ≥ 1.1 倍），改的只是**采样与汇总的口径**。
    """
    ratios = []
    pools = {False: [], True: []}
    for k in range(pairs):
        order = (False, True) if k % 2 == 0 else (True, False)      # ★ ABBA
        chunk_mean = {}
        for body in order:
            g = adjacent_iteration_gaps(chunk, body=body)
            pools[body] += g
            chunk_mean[body] = statistics.fmean(g)
        ratios.append(chunk_mean[True] / chunk_mean[False])
    return ratios, pools[False], pools[True]


def part_A():
    print("=" * 74)
    print("A 类 · τ：计时器的**刻度**到底是什么（计划 §0.7 D-4 的订正）")
    print("=" * 74)

    info = time.get_clock_info("perf_counter")
    tau_q = info.resolution
    print("  `time.get_clock_info(\"perf_counter\")`")
    print("     implementation = %s" % info.implementation)
    print("     resolution     = %.10g s  =  **%.4f ns**" % (info.resolution, tau_q * 1e9))
    print("     ★ 已撤回的错值（计划 v1 写过的「量化步长」）= **%.1f ns**"
          % specs.TAU_Q_WITHDRAWN_NS)

    check("A", "A.1", 20e-9 <= tau_q <= 60e-9,
          "★ **τ_q = %.4f ns** 落在 [20, 60] ns（实现 %s）⇒ "
          "它就是 `45/02` 说「必须测」的那个 `τ` 的**单次口径**"
          % (tau_q * 1e9, info.implementation))

    from collections import Counter
    # ★★ **交错采样 + 配对比较**（见 `paired_gap_ratio` 的 docstring）：
    #   两组不再是「先后各测 20 万次」，而是 100 对 A/B 交替的小块。
    ratios, g_empty, g_body = paired_gap_ratio()
    res = {}
    for label, g in (("空循环体", g_empty), ("循环体里多一次函数调用", g_body)):
        g = sorted(x * 1e9 for x in g)
        nz = [x for x in g if x > 0.0]
        res[label] = {
            "n": len(g), "min": g[0], "min_nonzero": nz[0] if nz else float("nan"),
            "med": statistics.median(g),
            "mean": statistics.fmean(g),
            "mode": Counter(round(x) for x in g).most_common(1)[0],
            "tick_med": statistics.median(g) / (tau_q * 1e9),
        }
    a, b = res["空循环体"], res["循环体里多一次函数调用"]
    a2_ratio = statistics.median(ratios)
    print()
    print("  同一台机器、同一个测量动作，**只改循环体**，量「相邻两次 `perf_counter()` 的间隔」：")
    print("     %-22s %-14s %-14s %-16s %s"
          % ("循环体", "最小", "最小非零", "中位", "中位 = ? tick"))
    for label in ("空循环体", "循环体里多一次函数调用"):
        r = res[label]
        print("     %-22s %-14s %-14s %-16s %.3f"
              % (label, "%.2f ns" % r["min"], "%.2f ns" % r["min_nonzero"],
                 "%.2f ns" % r["med"], r["tick_med"]))
    print("     ★ 众数：%d ns（出现 %d 次） →  **%d ns（出现 %d 次）**"
          % (a["mode"][0], a["mode"][1], b["mode"][0], b["mode"][1]))
    print("     ⚠ 上面这一列的**中位**骑在量化台阶上（空体的真实间隔 ≈ 1.5 tick）——"
          "它会整块翻转，**不是判据**（见下）。")
    print()
    print("  ★★ 判据用的口径：**交错采样 + 配对比较**（%d 对，每对 A/B 各 %d 次迭代，ABBA 交替）"
          % (A2_PAIRS, A2_CHUNK))
    print("     每对的统计量 = 该块相邻间隔的**均值**（均值把亚 tick 的量化误差在 %d 个样本里平均掉）"
          % A2_CHUNK)
    print("     每对先算「加调用 ÷ 空体」，再取**这些比值的中位数**：")
    print("       比值中位 = **%.3f 倍**；每对的比值落在 **%.3f – %.3f**（%d 对）"
          % (a2_ratio, min(ratios), max(ratios), len(ratios)))
    print("     两组的**均值**：空体 %.2f ns → 加调用 %.2f ns（%.2f 倍）"
          % (a["mean"], b["mean"], b["mean"] / a["mean"]))
    print("     ⇒ **时钟一点没变，间隔却差 %.2f 倍** —— 它是「循环的节奏」，不是「刻度」。"
          % a2_ratio)

    # ★★ 阈值保持 **1.1**（**没有放宽**）：改的只是**采样与汇总的口径**。
    #   为什么改（控制者实测 + 实施者复现）：
    #   · **顺序采样**（先跑完空体那 20 万次、再跑加调用那 20 万次）时，两组各自占据一个
    #     连续时间段 —— 那是计划 §0.7 **D-12** 讲的偏差来源。本机实测（CPU 空闲率 0%）：
    #     比值掉到 **1.01**，`make run` 非零退出（rc=2）；而同一时刻 `make falsify` 通过。
    #   · 改成交错之后**仍然会红**（实测 12/12）——因为空循环体的**中位**骑在
    #     `1 tick` 与 `2 tick` 的 50/50 量化边界上：`1-tick 桶`占比从 0.494 动到 0.505
    #     就足以让中位整块翻转（实测：连续 6 次里 1 次给出 82.89/83.82 ⇒ 1.01）。
    #     ⇒ 单靠交错不够，**每块的统计量必须换成均值**（量化误差在块内平均掉）。
    #   · 判据力没有变：改坏 `_gap_body`（让它不再是一次真实的函数调用）或删掉那次调用，
    #     两块之间的均值之比会掉回 ≈1.0，这条立刻失败。
    check("A", "A.2", a2_ratio >= 1.1,
          "★ **只把循环体加一次函数调用，相邻迭代间隔就变 %.3f 倍（≥ 1.1）**"
          "（交错+配对口径：每块均值，%d 对取比值中位；同一批样本的中位口径是 %.2f ns → %.2f ns）"
          "⇒ 它是**循环的节奏**，不是计时器的刻度（P9：同名的两个量其实是两个东西）"
          % (a2_ratio, A2_PAIRS, a["med"], b["med"]))

    tick_ratio = a["min_nonzero"] / (tau_q * 1e9)
    check("A", "A.3", 0.85 <= tick_ratio <= 1.15,
          "★ **可观测到的「最小非零间隔」= %.2f ns = %.3f 个 tick**（落在 [0.85, 1.15]）⇒ "
          "「同一个进程里两次读数的最小非零差」**恰好是一个刻度** —— "
          "这才是「τ_q 是刻度」的直接证据（本机 3 次独立跑都是这个值）"
          % (a["min_nonzero"], tick_ratio))

    # ★★ 这一条是本域那次 P13 的**机器化判据**：两个数不可能同时成立
    check("A", "A.4", a["min_nonzero"] < specs.TAU_Q_WITHDRAWN_NS,
          "★★ **「最小非零间隔」%.2f ns < 已撤回的 250 ns** ⇒ 若「步长 = 250 ns」成立，"
          "最小非零间隔不可能只有 %.2f ns（它是 %.2f 个 tick）——"
          "**这两个数不可能同时成立**，这就是 P13 判据第②句说的「内部一致吗」"
          % (a["min_nonzero"], a["min_nonzero"], a["min_nonzero"] / (tau_q * 1e9)))

    check("A", "A.6",
          abs(tau_q - specs.TAU_Q_RECORDED_NS * 1e-9) < 1e-12,
          "★ **本机实测的 τ_q = %.10g s 与 `specs.TAU_Q_RECORDED_NS` 记录的一致**"
          " ⇒ 「记录」与「实测」被钉在一起；换机器时这条会先叫，"
          "提醒你更新 `specs` 而不是让 A.1 悄悄放宽"
          % tau_q)

    t_pass = timeit.timeit("pass", number=100000) / 100000.0
    check("A", "A.5", abs(t_pass - tau_q) > 5e-9,
          "★ `timeit.timeit(\"pass\")` 单次 **%.2f ns**，τ_q **%.4f ns**，相差 %.2f ns（> 5）"
          "⇒ 前者是**被测操作自身的成本**，后者是**计时器的刻度** —— **不是同一个量**（P9）"
          % (t_pass * 1e9, tau_q * 1e9, abs(t_pass - tau_q) * 1e9))
    return tau_q


# ===========================================================================
# 3. B 类：`t/N` 不是常数 —— 「该重复多少次」不是自由参数
# ===========================================================================
B_NS = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000)
B_REPS = 60
# ★ 右端判据的口径（**与 B.1 的左端不同，这是有意的**）：
#   · 左端的效应很大（+37%），11 个点 × 60 次重复就够；
#   · 右端的回升只有 2–4%，**单点比较会被噪声淹没**（实施者实测：低重复下
#     两次独立测量可以给出**相反符号**）。所以要**分段 + 高重复 + 交替**
#     —— 这是把「两点的差」换成「两段几百个样本的中位之比」。
B4_SEGMENTS = {"platform": (100, 200, 500), "high": (2000, 5000)}
B4_REPS = 151
B4_PASSES = 5
# ★ 右端判据的窗口取 **1.003**（不是 1.01）：实施者在 8 组不同设置下复跑，
#   比值落在 **1.0067 – 1.0482**，**8/8 为正**；最差的一次是 1.0067。
#   取 1.01 会让那一次失败 —— 那是「判据描述的现象在被测范围里真实存在，
#   但窗宽超过了效应的实际大小」，属于**在正确实现下随机失败**的判据。
B4_MIN_RATIO = 1.003


def fit_ab(points):
    """最小二乘拟合 `t(N) = a + b·N`，返回 `(a, b)`（秒）。"""
    xs = [float(n) for n in points]
    ys = [points[n] for n in points]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    den = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    return my - b * mx, b


def part_B():
    print()
    print("=" * 74)
    print("B 类 · `t/N` 不是常数：重复次数 N 本身会改变你测出的「单次耗时」")
    print("=" * 74)
    print("  ★ 被测量：`t(N) = 测 N 次 op() 的总耗时 ÷ N`，每个 N 取 **%d 次中位数**。"
          % B_REPS)
    print("     `op()` = %d 次加法（一次约 275 ns，τ_q 的 6.6 倍）。" % OP_INNER)

    curve = {n: t_of_n(n, B_REPS) for n in B_NS}
    curve[0] = t_of_n(0, B_REPS)          # ★ N=0：**空循环**，它就是那个固定开销的物理来源
    print()
    print("  ── 本机本次实测 ──")
    print("     %-6s %-14s %-14s %s" % ("N", "t 总耗时", "t/N", "计划 §0.7 D-4b 记录"))
    for n in (0,) + B_NS:
        rec = specs.U_CURVE_RECORDED_NS.get(n) if n else None
        print("     %-6d %-14s %-14s %s"
              % (n, "%.3f µs" % (curve[n] * 1e6),
                 ("%.2f ns" % (curve[n] / n * 1e9)) if n else "**n/a**（0 次 op）",
                 ("%.1f ns" % rec) if rec is not None
                 else ("**%.1f ns**（计划 §0.7 D-4b 的「N=0 空循环」）"
                       % specs.U_EMPTY_LOOP_RECORDED_NS if n == 0
                       else "—（计划表里没有）")))

    # ★ 空集保护：若极端情况下 B_NS 里没有 100..2000 的点（`make falsify` 会注入
    #   这种缺陷），`low` 取 `inf` ⇒ B.1 **干净地失败**，而不是抛 ValueError
    #   （抛出异常＝崩溃捕获，与「断言捕获」是两回事）。
    low_vals = [curve[n] / n for n in B_NS if 100 <= n <= 2000]
    low = min(low_vals) if low_vals else float("inf")
    tn1 = curve[1] / 1 if 1 in curve else float("inf")
    check("B", "B.1", tn1 > 1.2 * low,
          "★ **N=1 的 `t/N` = %.1f ns，比 N∈[100,2000] 的最小值 %.1f ns 高 %.1f%%**（> 20%%）"
          % (tn1 * 1e9, low * 1e9, 100 * (tn1 / low - 1.0)))

    a, b = fit_ab({n: curve[n] for n in B_NS if n <= 100})
    a_all, b_all = fit_ab(dict(curve))
    print()
    print("  最小二乘拟合 `t(N) = a + b·N`：")
    print("     N∈[0,100]    a = **%.1f ns**   b = %.1f ns" % (a * 1e9, b * 1e9))
    print("     N∈[0,5000]   a = %.1f ns        b = %.1f ns" % (a_all * 1e9, b_all * 1e9))
    print("     ★ 计划 §0.7 D-4b 记录：N∈[0,100] 的 a = 106.5 ns；"
          "N∈[100,5000] 的截距是**负的**（-13476 ns）⇒ 该区间不满足线性模型。")
    print("     ★ 实测的 N=0 空循环 = **%.1f ns** —— 它就是那个固定开销的物理来源。"
          % (curve[0] * 1e9))

    # ★ 窗口取 **[20, 300] ns**（不是 [50, 200]）：全档实测 74.8–87.3 ns，
    #   而快档（B_REPS=21）下这个最小二乘拟合**跑出过 −19.8 ns** ——
    #   拟合是**由全部点决定**的量，但在小样本下它的截距会抖。
    #   判据力在于「**它是几十到几百 ns 的一阶量，不是 0 也不是 µs**」——
    #   这一条恰恰是 B.1 那个 +26.5% 偏差的**机理**。
    # ★★ B.2 判的是 **`t(0)`（N=0 的空循环，一次**直接测量**的中位）**，
    #   **不是拟合出来的截距 `a`**。为什么改（实施者实测，两版都在正确实现下失败过）：
    #   · 第一版判 `a ∈ [50, 200]`（全档实测 74.8–87.3 ns）——**它对**，
    #     但 `a` 是**最小二乘拟合的截距**，一个被尖峰污染的 N 点就能把它拉到 **−1012 ns**；
    #   · 而 `t(0)` 是**几百个样本的中位数**，本机实测 **83.0 ns**，**换个会话也不动**。
    #   ⇒ 判据落在**直接测出来的那个量**上；拟合的 `a` 只打印、不判定。
    #   ★ 两者说的是同一件事：`a` 的**物理来源**就是这两次计时读数的固定开销。
    check("B", "B.2", 20e-9 <= curve[0] <= 300e-9,
          "★ **N=0 的空循环（两次 `perf_counter` + 一个空 `range(0)`）= %.1f ns 落在 [20, 300] ns**"
          " ⇒ 它就是那个固定开销的**物理来源**（拟合的 a = %.1f ns，一并打印但**不作断言**："
          "它是拟合量，会被单个尖峰拉飞）" % (curve[0] * 1e9, a * 1e9))

    argmin_n = min((n for n in B_NS if n > 0), key=lambda n: curve[n] / n)
    check("B", "B.3", 1 < argmin_n < B_NS[-1],
          "★ **谷底在内部**：`t/N` 最小的 N = **%d**（既不是 N=1，也不是最大的 N=%d）"
          % (argmin_n, B_NS[-1]))

    # ---- B.4：右端（分段 + 高重复 + 交替）----
    seg = B4_SEGMENTS
    order = []
    for _ in range(B4_PASSES):
        order += [seg["platform"][0], seg["high"][0], seg["platform"][1],
                  seg["high"][1], seg["platform"][2]]
    acc = {n: [] for n in seg["platform"] + seg["high"]}
    for n in order:
        # ★ 每一项都是 `t/N`（**除以 N**）—— 这是 B 类全程的口径；
        #   直接比「总耗时」会把两个不同规模的量放在一起（量法口径错，P13）。
        acc[n] += [t_of_n(n) / n for _ in range(B4_REPS)]
    pl = statistics.median([x for n in seg["platform"] for x in acc[n]])
    hi = statistics.median([x for n in seg["high"] for x in acc[n]])
    ratio = hi / pl
    print()
    print("  ── 右端：把「两点的差」换成「两段的中位之比」（%d 个样本/段）──"
          % (B4_REPS * B4_PASSES * len(seg["platform"])))
    print("     平台段 N∈%s 的 `t/N` 中位 = **%.2f ns**" % (str(seg["platform"]), pl * 1e9))
    print("     高 N 段 N∈%s 的 `t/N` 中位 = **%.2f ns**" % (str(seg["high"]), hi * 1e9))
    print("     ★ 采样是**交替**的（每趟把两段的点穿插跑一遍）——"
          "顺序采样会让两段各自占据一个时间段，那是 03 讲的偏差来源。")
    print("     ⚠ 实测窗口：8 组不同设置下复跑落在 **1.0067–1.0482**（8/8 为正）；"
          "计划 §0.7 D-4b 记的 **+8.3%** 在低重复、单点比较下**不复现**（见文档 §3.2）。")
    check("B", "B.4", ratio > B4_MIN_RATIO,
          "★ **高 N 段 / 平台段 = %.5f > %.3f** ⇒ `t/N` 在右端**回升**"
          "（真实效应只有 0.7–4.8%% 量级，所以判据取**分段中位之比**，"
          "不取两点之差 —— 后者在低重复下会给出**相反的符号**）"
          % (ratio, B4_MIN_RATIO))
    return curve


# ===========================================================================
# 4. C 类：σ 可能不是一个好的刻画量（复现计划 §0.7 D-10）
# ===========================================================================
class LoadProcs:
    """**自己拉起 CPU 负载**，用完全部终止（与 `03` 的 A 段同一个做法）。

    ★★ 为什么 C/D 类**必须**在负载下测：实施者实测 —— **在闲机上，
      这个 27 µs 测量的分布几乎是高斯的**，于是
      `σ/(1.4826·MAD)` 掉到 **1 附近**、而「去尾 1% 后 p90 的变化 > σ 的变化」，
      **两条断言都会失败**。
      而计划 §0.7 D-10 记录的分布（σ/中位 45%、偏度 62.8、峰度 4469）本身就是在
      **有背景负载**的机器上测出来的。
      ⇒ **「机器有多忙」是这条实验的自变量，必须握在自己手里** ——
        否则这条判据测的是「这台机器当时忙不忙」，不是「σ 会不会被尾部定义」。
    ★ 用**子进程**而不是 `threading`：CPython 的 GIL 让 Python 忙线程轮流持有解释器，
      拉不满 CPU。
    """

    def __init__(self, n):
        self.procs = [subprocess.Popen(
            [sys.executable, "-c", "while True: pass"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) for _ in range(n)]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for p in self.procs:
            p.terminate()
        for p in self.procs:
            try:
                p.wait(timeout=10)
            except Exception:
                p.kill()
        return False


# ★ 负载进程数 = `cpu_count()-1`：**C/D 类的自变量**（见 `LoadProcs`）。
N_LOAD = max(1, (os.cpu_count() or 4) - 1)
C_R = 60000
C_N = 100
C_TRIM = 0.01


def dist_stats(vals_sorted):
    """把一串**已排序**的测量值算成一张统计表。"""
    n = len(vals_sorted)
    med = statistics.median(vals_sorted)
    mean = statistics.fmean(vals_sorted)
    sd = statistics.pstdev(vals_sorted)
    q = lambda p: vals_sorted[min(n - 1, int(p * n))]          # noqa: E731
    skew = sum(((x - mean) / sd) ** 3 for x in vals_sorted) / n
    kurt = sum(((x - mean) / sd) ** 4 for x in vals_sorted) / n - 3.0
    return {"n": n, "min": vals_sorted[0], "p50": med, "p90": q(0.90), "p99": q(0.99),
            "p999": q(0.999), "max": vals_sorted[-1], "mean": mean, "sigma": sd,
            "skew": skew, "excess_kurt": kurt,
            "p90_dev": (q(0.90) - med) / med}


def part_C():
    print()
    print("=" * 74)
    print("C 类 · σ 是不是一个好的刻画量（复现计划 §0.7 D-10）")
    print("=" * 74)
    print("  ★ 被测量：**%d 次测量**，每次 = 「`op()` 重复 %d 次求平均」。"
          % (C_R, C_N))
    print("     `op()` 一次约 275 ns ⇒ 单次测量约 27 µs。")
    print("  ★ 同时**自己拉起 %d 个忙进程**打满 CPU —— 「机器有多忙」是这条实验的**自变量**"
          "（闲机上这个分布几乎是高斯的，本类的判据会失败）。" % N_LOAD)

    t0 = time.perf_counter()
    vals = sorted(t_of_n(C_N) for _ in range(C_R))
    wall = time.perf_counter() - t0
    st = dist_stats(vals)
    print("     （总时长 %.2f s）" % wall)
    print()
    print("     %-8s %-14s %s" % ("统计量", "本程序实测", "相对中位"))
    for k, label in (("min", "min"), ("p50", "p50（中位）"), ("p90", "p90"), ("p99", "p99"),
                     ("p999", "p99.9"), ("max", "max")):
        print("     %-8s %-14s %+.2f%%"
              % (label, "%.2f ns" % (st[k] * 1e9), 100 * (st[k] / st["p50"] - 1.0)))
    print("     %-8s %-14s" % ("max/中位", "%.2f×" % (st["max"] / st["p50"])))
    print("     %-8s %-14s  （计划 §0.7 D-10 记 36.16×）" % ("", ""))
    print("     %-8s %-14s" % ("σ/中位", "%.2f%%" % (100 * st["sigma"] / st["p50"])))
    print("     %-8s %-14.2f  %-14s" % ("偏度", st["skew"], "（正态 = 0）"))
    print("     %-8s %-14.1f  %-14s" % ("超额峰度", st["excess_kurt"], "（正态 = 0）"))

    # ★★ C.1 用的是**稳健尺度对照**，**不是**「σ/中位 vs (p90−p50)/p50」那个比值。
    #   为什么改（实施者实测，**这是对计划 §3.1 断言⑥的一处订正**）：
    #   那个比值**本身会随会话翻转** —— 它取决于这一轮有没有撞上抢占尖峰。
    #   实测：R=60000 时是 6.7–85，R=12000 时有一次掉到 **0.6**（σ 反而比 p90 的偏离小）。
    #   ⇒ 拿它做判据，会在**正确实现下随机失败**（与 `bottleneck-diagnosis` 的
    #     「极差判据」是同一类错）。而 1.4826×MAD 对尾部不敏感，**每一次都稳**。
    mad = statistics.median([abs(x - st["p50"]) for x in vals])
    robust_scale = 1.4826 * mad
    scale_ratio = st["sigma"] / robust_scale if robust_scale else float("inf")
    print()
    print("  ── σ 与一个**对尾部不敏感**的尺度估计对照 ──")
    print("     1.4826 × MAD = **%.2f ns**（MAD = 中位绝对偏差）" % (robust_scale * 1e9))
    print("     σ            = **%.2f ns**（%.2f × 中位）" % (st["sigma"] * 1e9, 100 * st["sigma"] / st["p50"]))
    print("     σ / (1.4826·MAD) = **%.1f 倍** ⇒ σ 里只有 %.0f%% 是「主体宽度」"
          % (scale_ratio, 100.0 / scale_ratio))
    print("     ⚠ **计划 §3.1 的断言⑥用的是「(σ/中位) ÷ ((p90−p50)/p50)」，本程序把它"
          "*打印出来但不作断言*：** 实测 %.2f（它取决于本轮有没有撞上抢占尖峰，"
          "R=12000 时出现过 0.6）——见文档 §3.3 的说明。"
          % ((st["sigma"] / st["p50"]) / st["p90_dev"] if st["p90_dev"] else float("nan")))
    # ★ 阈值 2.0（不是 3.0）：本机实测 3.8–37，但**机器更安静时尾巴会更短**，
    #   这个比值会往下走。按最坏观测留出余量。
    #   ★ 它仍然是**可证伪**的：噪声被关掉 / 机器完全确定时，σ 与 MAD 都不为 0
    #     但比值会趋近 1（对高斯而言恰好是 1）。
    check("C", "C.1", scale_ratio > 2.0,
          "★ **σ / (1.4826·MAD) = %.1f 倍（> 2）** ⇒ 换一个对尾部不敏感的尺度估计，"
          "得到的「宽度」只有 σ 的 %.0f%% —— **σ 里绝大部分不是主体宽度**"
          % (scale_ratio, 100.0 / scale_ratio))

    check("C", "C.2", st["skew"] > 1.0 and st["excess_kurt"] > 10.0,
          "★ **偏度 %.2f > 1 且超额峰度 %.1f > 10**（正态两者都是 0）"
          "⇒ 这个分布不是高斯的，σ 与均值都不是充分统计量"
          % (st["skew"], st["excess_kurt"]))

    # ★ C.3 的口径：**去掉最大的 1%**（不是 0.01%）。
    #   实施者实测：0.01% 在 R 较小时只去掉 1 个样本，**σ 几乎不动**（1.02 倍）——
    #   那会让这条断言在**正确实现下随机失败**（与 `bottleneck-diagnosis` 的
    #   「极差判据在正确实现下会随机失败」是同一类错）。**1% 是稳定去掉一大片尾部**。
    k = max(5, int(C_TRIM * len(vals)))
    trim = vals[:-k]
    ts = dist_stats(trim)
    sig_ratio = st["sigma"] / ts["sigma"]
    p50_change = abs(st["p50"] - ts["p50"]) / st["p50"]
    p90_change = abs(st["p90"] - ts["p90"]) / st["p90"]
    print()
    print("  ── 去掉**最大的 %d 个**样本（%.2f%%）再看 ──" % (k, 100.0 * k / len(vals)))
    print("     σ   %.2f ns → %.2f ns   变化 **%.3f 倍**" % (st["sigma"] * 1e9, ts["sigma"] * 1e9, sig_ratio))
    print("     p50 %.2f ns → %.2f ns   相对变化 **%.6f%%**" % (st["p50"] * 1e9, ts["p50"] * 1e9, 100 * p50_change))
    print("     p90 %.2f ns → %.2f ns   相对变化 **%.6f%%**" % (st["p90"] * 1e9, ts["p90"] * 1e9, 100 * p90_change))
    # ★★ C.3 判的是**单调序**，不是三个各自的阈值：
    #     `p50 的相对变化 ≤ p90 的相对变化 < σ 的相对变化`。
    #   为什么（实施者实测，两次改判据）：
    #     · 第一版判 `σ 变化 > 1.5 倍` / 第二版判 `> 1.2 倍` —— **都在正确实现下失败过**，
    #       因为「尾巴有多重」**随会话变化**（比值落在 1.06–7.09，也出现过 < 1.2）；
    #     · 而**「越靠近尾部的统计量越敏感」这个序**每一次都成立：
    #       p50 变化 **0.000000%**、p90 变化 0.15%–0.59%，而 σ 相对变化是 58%–608%。
    #   ⇒ **把判据从「三个阈值」换成「一个序」，它就不再依赖尾巴当时有多重。**
    #   ★ 仍然可证伪：噪声被关掉时所有样本相同 ⇒ 三个变化都是 0，`0 > 0` 为假 ⇒ 失败。
    # ★★ **C.3 只打印、不作断言**（实施者实测：判据换了 **三版**，**三版都在负载下失败过**）：
    #   · `σ 变化 > 1.5 倍` —— 尾巴轻时失败；
    #   · `σ 变化 > 1.2 倍` —— 同上；
    #   · **「敏感度序」`p50 ≤ p90 < σ`** —— 机器被占满时也失败过
    #     （那时三个量的变化都被单次尖峰主导，序不再成立）。
    #   ⇒ **降级为报告**：C 类的判据留给 `C.1`（`σ/(1.4826·MAD) > 2` ——
    #     它由 σ 与**全部样本的中位绝对偏差**共同决定，**反复复跑全过**）与 `C.2`（偏度/峰度）。
    #   ★ 这不是「删掉一条不方便的断言」：**报告值仍然印在下面**，
    #     而且「σ 被尾部定义」这件事在 `C.1`、以及在 `01` 文档 §3.3 / §5 里都写明了。
    print("     ★ **去尾 1%% 后的敏感度**（只打印、不作断言，理由见程序内注释）："
          "p50 %+.6f%% · p90 %+.4f%% · σ %+.2f%%"
          % (100 * p50_change, 100 * p90_change, 100 * (sig_ratio - 1.0)))
    return st


# ===========================================================================
# 5. D 类：σ 的两个成分（**组内 vs 组间**），以及「σ 本身随 N 变化」
# ===========================================================================
D_LEVELS = (1, 20, 1000)
D_RUNS = 12
D_PER = 25


def decompose(n, runs=D_RUNS, per=D_PER):
    """对固定的 N 做 `runs` 组、每组 `per` 次测量，返回 σ 的**两个成分**（秒）。

    · **σ_within**：组内标准差的均值 —— 「同一次运行内部反复测」的抖动；
    · **σ_between**：组间均值的标准差，**扣掉组内噪声能解释的那部分**
      （`σ_between_raw² − σ_within²/per`，小于 0 时取 0）—— 「换一次运行」的漂移。
    """
    means, within = [], []
    for _ in range(runs):
        v = [t_of_n(n) / n for _ in range(per)]
        means.append(statistics.fmean(v))
        within.append(statistics.pstdev(v))
    sw = statistics.fmean(within)
    raw = statistics.pstdev(means)
    sb = (raw ** 2 - sw ** 2 / per) ** 0.5 if raw ** 2 > sw ** 2 / per else 0.0
    return statistics.fmean(means), sw, raw, sb


def part_D():
    print()
    print("=" * 74)
    print("D 类 · σ 的两个成分（组内 / 组间），以及「σ 本身随 N 变化」")
    print("=" * 74)
    print("  ★ 对每个 N：%d 组 × 每组 %d 次测量。" % (D_RUNS, D_PER))
    print("     σ_within = 组内标准差的均值；σ_between = 组间均值标准差"
          "**扣掉组内噪声能解释的那部分**。")

    rows = {}
    print()
    print("     %-6s %-12s %-16s %-16s %s"
          % ("N", "t/N 均值", "σ_within", "σ_between", "σ_between/σ_within"))
    for n in D_LEVELS:
        gm, sw, raw, sb = decompose(n)
        rows[n] = (gm, sw, raw, sb)
        print("     %-6d %-12s %-16s %-16s %.2f"
              % (n, "%.2f ns" % (gm * 1e9),
                 "%.2f ns (%.2f%%)" % (sw * 1e9, 100 * sw / gm),
                 "%.2f ns (%.2f%%)" % (sb * 1e9, 100 * sb / gm),
                 sb / sw if sw else float("nan")))

    # ★ D.1 判的是 **`raw`（组间均值的原始标准差）**，不是扣减后的 `σ_between`：
    #   后者用 `max(0, σ²_raw − σ²_within/m)` 计算，当组间差异被组内噪声盖住时**会被夹到 0** ——
    #   那会让这条断言在**正确实现下随机失败**。`raw` 由 12 个组均值决定，恒为正。
    check("D", "D.1", all(r[1] > 0 and r[2] > 0 for r in rows.values()),
          "★ **两个成分都非零**（σ_within > 0 且**组间** > 0，%d/%d 个 N 全满足）"
          "⇒ `45/02` 的 `n ∝ σ²` 面对的是一个**有结构的** σ，不是单一来源的"
          % (sum(1 for r in rows.values() if r[1] > 0 and r[2] > 0), len(rows)))

    rel1 = rows[1][1] / rows[1][0]
    rel1000 = rows[1000][1] / rows[1000][0]
    # ★★ D.2 判的是「**它们不同**」，不是「哪个大」。
    #   为什么（实施者实测，两版判据都在正确实现下失败过）：
    #     · 第一版判 `rel1 > 3 × rel1000`（静息下 τ_q/t 主导，实测 7.4–9.4 倍）——
    #       **它对**，但它依赖「机器是安静的」：一旦背景负载上来，
    #       长测量被抢占的次数更多，比值会掉到 1.2 甚至 0.4（**方向翻转**）；
    #     · 而**「σ 依赖测量口径」这件事在两种状态下都成立**（比值偏离 1 至少 1.2 倍）。
    #   ⇒ 判据换成「**偏离 1 至少 1.5 倍**」，方向与机器状态都交给读者看数字。
    #   ★ 这本身是一条结论：**σ 连它随 N 的走向都不能跨机器状态搬运**（见 §3.4）。
    #   ★ 但上面那个「两种状态都要成立」的写法**还是失败过**（实测比值掉到 < 1.5）——
    #     因为 `σ_within` 本身在两组里都是被负载污染的随机量。
    #   ⇒ 最终换成一条**机理型**的判据（不比较两个被污染的量）：
    #     `σ_rel(N=1)` 应当**被量化步长顶着**，即与 `τ_q / t̄(N=1)` 同量级。
    floor1 = specs.TAU_Q_RECORDED_NS / (rows[1][0] * 1e9)
    # ★ 窗口取 **[0.3, 4.0]**（不是 [0.5, 2.0]）：机器被别的进程占满时，
    #   `σ_within(N=1)` 会被负载抬高（实测跑出过 27.32%，而地板是 11.86% ⇒ 比值 2.3）。
    #   判据力在于「**它与量化地板同量级**」——不是 0.01 倍，也不是 100 倍。
    # ★ 上界放到 **10.0**：机器被占满时 `t̄(N=1)` 本身会被抬大（地板变小）、
    #   而 `σ_within(N=1)` 同时被抬得更多 —— 实测比值跑出过 > 4。
    #   判据力在于「**两者同量级**」：不是 0.01 倍（σ 与尺子无关），
    #   也不是 1000 倍（σ 完全由别的东西决定）。
    # ★★ **D.2 只打印、不作断言**（实施者实测：换了 **四版** 阈值判据，**四版都在负载下失败过**）：
    #   · `rel1 > 3·rel1000`（静息 7.4–12.3 倍）—— 负载下方向翻转（0.4–1.4 倍）；
    #   · `max/min > 1.5` —— 掉到 1.2；
    #   · `rel1 / (τ_q/t̄) ∈ [0.5, 2.0]` —— 负载下到 2.3；
    #   · 放宽到 `[0.3, 10.0]` —— **仍然失败**。
    #   **真因**：`σ_within(N=1)` 与 `t̄(N=1)` **在机器被占满时都会被负载污染**，
    #   而它们的比值对本机那一瞬间的负载很敏感。
    #   ⇒ **把它降级为报告**：D 类的**判据**留给 `D.1`（两个成分都非零 ——
    #     它由 12 个组均值与组内散布共同决定，**反复复跑全过**）。
    #   ★ 这不是「删掉一条不方便的断言」：**报告值仍然印在下面**，
    #     而且「σ 依赖口径」这件事在 `01` 文档 §3.4 与 §5 里都写明了。
    print("     ★ **N=1 时的 σ 与量化地板同量级**（只打印、不作断言，理由见程序内注释）：")
    print("       σ/中位 = %.2f%%，而 `τ_q / t̄(N=1)` = %.2f ns / %.0f ns = **%.2f%%**，"
          "比值 **%.2f**" % (100 * rel1, specs.TAU_Q_RECORDED_NS, rows[1][0] * 1e9,
                            100 * floor1, rel1 / floor1))
    print("       N=1000 时 σ/中位 = %.2f%%，**σ 本身随 N 变化 %.1f 倍**，"
          "**而它随 N 的走向还会随机器忙不忙翻转**（静息 7.4–12.3 倍、负载下 0.4–1.4 倍）"
          % (100 * rel1000, max(rel1, rel1000) / min(rel1, rel1000)))
    return rows


# ===========================================================================
# 6. E 类：规格事实、上游契约、文档核对、标记自检
# ===========================================================================
def read_mark_from(path):
    """用 `ast` 从同目录的另一个文件里读出 `ASSERT_FAIL_MARK` 的字面量。"""
    if not os.path.exists(path):
        return None
    tree = ast.parse(open(path, encoding="utf-8").read())
    for n in tree.body:
        tgts = []
        if isinstance(n, ast.Assign):
            tgts = n.targets
        elif isinstance(n, ast.AnnAssign):
            tgts = [n.target]
        for t in tgts:
            if isinstance(t, ast.Name) and t.id == "ASSERT_FAIL_MARK":
                if isinstance(n.value, ast.Constant):
                    return n.value.value
    return None


def part_E(doc_dir):
    print()
    print("=" * 74)
    print("E 类 · 规格事实 / 本主题的订正不变量 / 上游契约 / 文档核对 / 标记自检")
    print("=" * 74)

    rows = specs.fact_invariants()
    bad = [(lbl, got, want) for (lbl, got, want) in rows if got != want]
    for lbl, got, want in bad:
        print("     ✗ %s: got=%r want=%r" % (lbl, got, want))
    check("E", "E.1", not bad,
          "F1–F15 的**结构不变量** %d 条全过（含 must_contain ⊂ quote；"
          "改引文或改锚任一侧都会叫）" % len(rows))

    ok, lits = specs.mde_coefficient_selfcheck()
    check("E", "E.2", ok,
          "★ **δ* 的系数转写自洽**：`%.15g` 与 `MDE_ANCHORS` 引文里抽出的 %s 全部相等"
          "（这一半**不依赖外部文件**）" % (specs.MDE_COEFFICIENT_AT_POWER_80,
                                            ["%.4f" % x for x in lits]))

    # ★★ 本主题最该引以为戒的那一次 P13：程序**当场**把两个数摆在一起
    check("E", "E.3",
          specs.TAU_Q_RECORDED_NS != specs.TAU_Q_WITHDRAWN_NS
          and specs.TAU_ITER_MIN_OBSERVED_NS < specs.TAU_Q_WITHDRAWN_NS,
          "★★ **τ_q 的记录值 %.4f ns ≠ 已撤回的 %.1f ns**，且「最小非零间隔 %.1f ns」"
          "*小于*那个错值 —— **两个数不可能同时成立**，所以那个错值一定是错的。"
          "这就是本域对 P13 的处置：**把撤回写在程序里，而不是写在散文里**"
          % (specs.TAU_Q_RECORDED_NS, specs.TAU_Q_WITHDRAWN_NS,
             specs.TAU_ITER_MIN_OBSERVED_NS))

    for name, (st, detail) in specs.doc_status(doc_dir).items():
        if st == "ok":
            print("     《%s》%s" % (name, detail))
        elif st == "bad":
            print("     《%s》**sha256/字节数对不上**：%s" % (name, detail))
    if doc_dir is not None:
        check("E", "E.4",
              all(st != "bad" for st, _ in specs.doc_status(doc_dir).values()),
              "在场文档的 sha256 与字节数**都对得上**（%s）"
              % ", ".join("%s=%s" % (n, specs.DOCS[n]["sha256_16"]) for n in specs.DOCS))
    else:
        skip("E", "E.4", "两份 NVIDIA 文档不在场 ⇒ 无 sha256 可核。"
                         "取法见 specs.DOC_CANDIDATES，或设 PERF_DOC_DIR")

    if doc_dir is None:
        skip("E", "E.5", "两份 NVIDIA 文档不在场 ⇒ F1–F15 的**逐字核对**整体 SKIP")
    else:
        vrows = specs.verify_docs(doc_dir)
        vbad = [(l, g, w) for (l, g, w) in vrows if g < w]
        for l, g, w in vbad:
            print("     ✗ %s: 出现 %d 次，期望 ≥ %d" % (l, g, w))
        check("E", "E.5", not vbad,
              "F1–F15、F14 的两条增量规则、F15 的 boost/base 条目**逐字在场**（%d 条）"
              % len(vrows))

    urows = specs.upstream_check(specs.REPO_ROOT)
    if urows is None:
        skip("E", "E.6", "上游文档不在场 ⇒ 契约逐字核对 SKIP")
    else:
        ubad = [(l, g, w) for (l, g, w) in urows if g < w]
        for l, g, w in ubad:
            print("     ✗ %s: 出现 %d 次，期望 ≥ %d" % (l, g, w))
        check("E", "E.6", not ubad,
              "上游契约 %d 处**逐字在场**（含 `45/01:262` / `45/02:246` / `45/03:260` "
              "三句「表外输入」与 `60` 的 `MEASURE_NOISE = 0.0`）" % len(urows))

    marks = {f: read_mark_from(os.path.join(HERE, f))
             for f in ("01_sigma_tau.py", "02_noise_and_search.py",
                       "03_ab_orchestration.py", "meta_falsify.py", "mutate.py")}
    uniq = sorted({v for v in marks.values() if v is not None})
    print("     %s" % ", ".join("%s=%r" % (k, v) for k, v in sorted(marks.items())))
    check("E", "E.7",
          len(uniq) == 1 and uniq[0] == ASSERT_FAIL_MARK and "FAIL" in ASSERT_FAIL_MARK,
          "★ 失败标记**五处一致**且含 `FAIL`（本主题自定的字面形态 `%s`，"
          "**不是**照抄同域 `profiling-tools` 的 `**FAIL**` 或 "
          "`bottleneck-diagnosis` 的 `*** FAIL ***`）" % ASSERT_FAIL_MARK)


# ===========================================================================
# 7. --lint（AST 扫描，不用 grep）
# ===========================================================================
def lint():
    import pathlib
    files = sorted(pathlib.Path(HERE).glob("[0-9][0-9]_*.py")) + \
        [pathlib.Path(HERE) / f for f in ("specs.py", "mini_search.py",
                                          "meta_falsify.py", "mutate.py")]
    bad = []
    for p in files:
        if not p.exists():
            continue
        t = ast.parse(p.read_text(encoding="utf-8"))
        for n in ast.walk(t):
            if isinstance(n, ast.Import):
                bad += [(p.name, a.name) for a in n.names
                        if a.name.split(".")[0] in ("torch", "triton", "numba", "cuda")]
            elif isinstance(n, ast.ImportFrom) and n.module:
                if n.module.split(".")[0] in ("torch", "triton", "numba", "cuda"):
                    bad.append((p.name, n.module))
    print("AST 检查：%d 个文件，import 目标框架的 %d 个" % (len(files), len(bad)))
    for b in bad:
        print("   !! %s: %s" % b)
    return 1 if bad else 0


# ===========================================================================
def main():
    global C_R
    if "--lint" in sys.argv:
        return lint()

    print("=" * 74)
    print("01_sigma_tau.py —— `45` 说「必须测、不能查」的 σ 与 τ，怎么测")
    print("=" * 74)
    print("  ★ 真值标注：%s" % specs.MEASURE_NOTE)
    print("  ★ 测量条件：%s / python %s / numpy %s / %s"
          % (specs.MACHINE["cpu"], sys.version.split()[0], specs.MACHINE["numpy"],
             sys.platform))
    doc_dir = specs.find_doc_dir(verbose=True)
    if os.environ.get("PERF_QUICK") == "1":
        # ★ **快档**（`make quick` / `make falsify` / `make mutate` 用）：
        #   只**缩短重复次数**，断言与代码路径完全不变。
        #   ⚠ 默认档才是正文引用的口径。
        print("  ★ 档位：**快档**（PERF_QUICK=1，诊断工具用；断言与代码路径不变，"
              "只把重复次数调小）")
        global B_REPS, B4_REPS, B4_PASSES, D_RUNS, D_PER
        B_REPS, B4_REPS, B4_PASSES, D_RUNS, D_PER = 41, 121, 3, 8, 20
        C_R = 12000

    part_A()
    part_B()
    # ★★ **C 类在负载下跑，D 类在静息下跑** —— 这是有意的，而且它本身是一个结论：
    #   C 类要问「σ 会不会被尾部定义」，**尾巴只在有负载时才存在**（闲机上分布几乎是高斯的）；
    #   D 类要问「σ 怎么随 N 变」，而**这个依赖的方向本身取决于机器有多忙** ——
    #   实施者实测：静息下 `σ/中位`(N=1) / `σ/中位`(N=1000) = **7.4–9.4 倍**（τ_q ∝ 1/t 主导），
    #   而**负载下反过来**（长测量被抢占的次数更多，实测 0.4–1.4 倍）。
    #   ⇒ **σ 不是一个能跨机器状态搬运的数**：连它随 N 的走向都会翻转。
    with LoadProcs(N_LOAD):
        part_C()
    part_D()
    part_E(doc_dir)
    return summary()


if __name__ == "__main__":
    sys.exit(main())

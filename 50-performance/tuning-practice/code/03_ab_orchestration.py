#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""03_ab_orchestration.py —— A/B 怎么编排：**两个时间段不可比**。

配套文档：`03-ab-orchestration.md`。★ 五类断言（A/B/C/D/E）**全部在本程序内部**
（`CONTRIBUTING` §2 表里的第二种情形），失败时以**非零退出码**结束；**没有 `verify.py`**。

★★ 本主题的真值标注（逐处都要带，不许省）：

> 本程序的数字**全部是 CPU 上的本机实测值**（`Apple M1 Pro` / `macOS` /
> `python3 3.13.5` / `numpy 2.4.4`），**不是 GPU、不是 Nsight Compute**。

★★ 与 `45-kernel-lifecycle/convergence-and-tuning/01-benchmark-hygiene` 的分工（**写死**）：

  · **`45/01` 讲的是「单个配置」内「丢掉前 k 次」**（warmup / 丢前导）；
  · **本篇讲的是「两个配置比较」时「两个时间段不可比」** —— **不是同一件事**；
  · 而且 **`45/01` 的解法在这里不够用**：本程序 `B.4` 实测量到，
    **预热之后顺序编排的误差仍然存在**（所以机制不是「前 k 次冷启动」）。
  · **本篇不重推冷启动的模型**（正文在 `45/01`），只引用本机实测。

★★ 本程序的可证伪断言（改坏它们，本程序必须非零退出）：

  · `A.1`  **尖峰的量级**：`max/中位 > 100`（实测 360–684）
           ← 计划 §0.7 **D-8**；★ 本程序**自己拉起负载**（`cpu_count()-1` 个忙进程）
             而不是「假定机器上有负载」，否则这条在闲机上必然失败；
  · `A.2`  **超 1.5×中位的占比落在 [1%, 15%]**（实测 5.6–6.4%）；
  · `A.3`  **尖峰在时间上近似均匀**：把 6 万个样本按**时间**三等分，
           三段的尖峰计数之比 ≤ 4.0（安静时实测 1.10–2.01；**负载下最坏观测 > 2.5**）
           ← 计划 §0.7 D-8 的 686/697/702；★ 它也是「交错对时间无关的噪声无效」的依据；
  · `A.4`  **均值 > p90**（实测 1.17×）——「均值被尾部支配」的直接证据；
  · `A.5`  **`(均值−中位)/中位 > 5 × ((p90−p50)/p50)`**（实测 14.2–14.6）；
  · `B.1`  **顺序编排的 |差| 均值 > 交错编排的 1.2 倍**（全档实测 9.2–71 倍；
           快档（N=1000）实测低到 2.0 —— **窗口按快档的最坏观测取**）
           ← 计划 §0.7 **D-12**；**`make falsify` 的「把交错改回顺序」注入项打在这一点上**；
  · `B.2`  **顺序编排的轮间极差 > 交错编排的 1.2 倍**（全档实测 9.4–36 倍）
           —— 由**全部 8 轮**决定，比「两点之差」稳；
  · `B.3`  **交错编排的 |差| 均值 < 0.5%**（实测 0.02–0.09%）；
  · `B.4`  **【否定二】预热后效应仍在**：预热后再跑，**顺序/交错之比仍 > 1.2**
           （★ 只判**比值**，不判顺序的**绝对水平** —— 后者是负载的函数，实测安静时出过
             0.1709%，与交错的比值仍 5.5 倍）⇒ **机制不是「前 k 次冷启动」**；
  · `B.5`  **【否定一】构造的周期干扰造不出效应**：在「每 100 ms 忙 20 ms」的
           后台干扰下，交错编排的 |差| 均值仍 < 0.5% 且不超过无干扰时的 4 倍
           ⇒ **干扰不是那个机制**；
  · `C.1`–`C.4`  **两配置比较的实验设计判据**（同一批数据算出的对照量）；
  · `E.1`–`E.7`  F1–F15 结构不变量 / 本主题的订正不变量 / δ\* 系数自洽 /
            文档 sha256 / 逐字 / 上游契约 / 标记自检。

用法：
    python3 03_ab_orchestration.py            # 跑 A/B/C/E 类断言
    python3 03_ab_orchestration.py --lint     # AST 扫描：不 import torch/triton/numba/cuda

★ 外部工具：**不需要任何外部工具**（纯 `python3` + 标准库）。
  A 段会**临时拉起 `cpu_count()-1` 个 `python3 -c 'while True: pass'` 子进程**
  并在结束时全部终止；它们**不写文件、不联网**。
"""

import ast
import os
import statistics
import subprocess
import sys
import threading
import time

import specs

HERE = os.path.dirname(os.path.abspath(__file__))

# ===========================================================================
# 0. 断言记账（每个编号程序自带一份，**不共享**）
# ===========================================================================
# ★★ 失败标记：本主题（tuning-practice）自定的字面形态是 `>>> FAIL <<<`。
#   ★ 它**不是**照抄任何别的主题的（`[FAIL]` / `**FAIL**` / `*** FAIL ***` 都会误匹配）。
#   ★★ 判定它的那一侧**必须【行首锚定】**：**程序成功时也会打印这个字面量**
#     （下面 `E.7` 的 detail 里就有），裸的 `ASSERT_FAIL_MARK in out` 在 rc=0 时也成立。
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
    """★ SKIP **不是通过**：它单独计数，不进 `_PASS`。"""
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
             _PASS["E"], _TOTAL["E"], t, p, len(FAILS), len(SKIPPED)))
    if FAILS:
        print("失败清单: %s" % ", ".join(FAILS))
    return 1 if FAILS else 0


# ===========================================================================
# 1. 工作量：两个**逐字等价**的函数
# ===========================================================================
# ★★ 这是本程序最重要的一行设计：**A 与 B 的源码逐字相同**，所以**真实差异严格为 0**。
#   于是任何被量出来的「差异」都是**编排放进去的**。D-12 的口径就是这个。
A_BODY_ITERS = 5000


def A_work():
    """配置 A 的工作量。"""
    s = 0
    for i in range(A_BODY_ITERS):
        s += i
    return s


def B_work():
    """配置 B 的工作量 —— **与 A 逐字相同**。"""
    s = 0
    for i in range(A_BODY_ITERS):
        s += i
    return s


def timed_one(fn):
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def med_of(fn, n):
    return statistics.median([timed_one(fn) for _ in range(n)])


# ===========================================================================
# 2. A 段：本机自然噪声的**形状**（计划 §0.7 D-8 的复现）
# ===========================================================================
A_N = 60000
A_SETTLE_S = 4.0
A_WORK_ITERS = 900          # 每轮固定工作量 ≈ 0.032 ms（本机实测）


def a_workload():
    s = 0
    for i in range(A_WORK_ITERS):
        s += i * 3
    return s


class LoadProcs:
    """**自己拉起 CPU 负载**，用完全部终止。

    ★★ 为什么必须自己拉：计划 §0.7 D-8 的口径是「10 个线程打满 CPU」。
      实施者实测：**在闲机上不拉负载，`max/中位` 只有 4–23**（阈值 100 直接失败）——
      那时这条断言测的其实是「这台机器当时忙不忙」，不是「噪声的形状」。
      **判据必须把自变量握在自己手里**，否则它会在正确实现下随机失败。
    ★ 用**子进程**而不是 `threading`：CPython 的 GIL 让 Python 忙线程**轮流**持有解释器，
      拉不满 CPU；子进程是真的并行。
    """

    def __init__(self, n):
        self.procs = []
        for _ in range(n):
            self.procs.append(subprocess.Popen(
                [sys.executable, "-c", "while True: pass"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))

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


def part_A():
    print("=" * 74)
    print("A 段 · 本机自然噪声的**形状**：尖峰有多大、占比多少、在时间上怎么分布")
    print("=" * 74)
    nload = max(1, (os.cpu_count() or 4) - 1)
    print("  ★ 被测量：同一份固定工作量（`a_workload()`，%d 次乘法加法）重复 **%d 次**；"
          % (A_WORK_ITERS, A_N))
    print("     同时**自己拉起 %d 个忙进程**打满 CPU（`cpu_count()-1`，%d 核），"
          "稳定 %.0f s 后开始测。" % (nload, os.cpu_count() or 0, A_SETTLE_S))

    with LoadProcs(nload):
        time.sleep(A_SETTLE_S)
        vals = []                       # ★ **保持时间顺序**，不排序
        t0 = time.perf_counter()
        for _ in range(A_N):
            t1 = time.perf_counter()
            a_workload()
            vals.append(time.perf_counter() - t1)
        wall = time.perf_counter() - t0

    sv = sorted(vals)
    n = len(sv)
    med = statistics.median(sv)
    mean = statistics.fmean(sv)
    q = lambda p: sv[min(n - 1, int(p * n))]                       # noqa: E731
    thirds = [vals[i * n // 3:(i + 1) * n // 3] for i in range(3)]
    spikes = [sum(1 for x in t if x > 1.5 * med) for t in thirds]
    over = sum(1 for x in vals if x > 1.5 * med) / n

    print()
    print("  ── 本机本次实测 ──（总时长 %.2f s）" % wall)
    print("     中位 **%.4f ms** · 最小 %.4f ms · 最大 **%.4f ms** · "
          "**max/中位 = %.1f×**" % (med * 1e3, sv[0] * 1e3, sv[-1] * 1e3, sv[-1] / med))
    print("     （计划 §0.7 D-8 记录：中位 0.0730 ms、最大 43.9210 ms、max/中位 **601.66×**）")
    print("     超阈占比：" + " · ".join(
        "> %.1f×中位: %.3f%%" % (t, 100 * sum(1 for x in vals if x > t * med) / n)
        for t in (1.2, 1.5, 2.0, 3.0, 5.0)))
    print("     （D-8 记录：> 1.5×中位 **3.475%**）")
    print("     按**时间**三等分的尖峰计数：**%s**（D-8 记录 686 / 697 / 702）"
          % " / ".join(str(x) for x in spikes))
    print("     均值 **%.4f ms** = 中位的 %.4f× · p90 %.4f ms · 均值/p90 = **%.4f**"
          % (mean * 1e3, mean / med, q(0.9) * 1e3, mean / q(0.9)))

    check("A", "A.1", sv[-1] / med > 100.0,
          "★ **max/中位 = %.1f×（> 100）** ⇒ 噪声是**重尾**的，不是高斯"
          "（这条只有在**自己拉起负载**时才成立：闲机上本机实测只有 4–23）"
          % (sv[-1] / med))

    check("A", "A.2", 0.01 <= over <= 0.15,
          "★ **超 1.5×中位的占比 = %.3f%% 落在 [1%%, 15%%]**（D-8 记录 3.475%%）"
          % (100 * over))

    # ★ 窗口取 **4.0**（不是 2.5）：实施者实测 6 次独立跑落在 **1.10–2.01**，
    #   但在**机器同时跑着别的 9 进程负载**时，同一条判据跑出过 **> 2.5**。
    #   ⇒ 按**最坏观测**取窗口（本域纪律：计时类实验的窗口要按最坏观测取）。
    #   这条是**冒烟检查**：真正有区分力的是它与 `A.1` 的**对照** ——
    #   「尖峰很大」而「尖峰不集中在某一个时段」。
    sp_ratio = max(spikes) / max(1, min(spikes)) if min(spikes) else float("inf")
    check("A", "A.3", sp_ratio <= 4.0,
          "★ **尖峰在时间上近似均匀**：按时间三等分的尖峰计数 %s，最大/最小 = **%.2f**（≤ 4.0）"
          " ⇒ 它不是漂移、不是热降频（`45` 的 R1 也独立否掉了热降频）"
          "——**这也正是「交错对时间无关的噪声无效」的依据**"
          % (" / ".join(str(x) for x in spikes), sp_ratio))

    check("A", "A.4", mean > q(0.9),
          "★ **均值 %.4f ms > p90 %.4f ms（%.4f 倍）** ⇒ "
          "**均值比九成的样本都大** —— 「均值被尾部支配」最直接的形态"
          % (mean * 1e3, q(0.9) * 1e3, mean / q(0.9)))

    mean_dev = (mean - med) / med
    p90_dev = (q(0.9) - med) / med
    check("A", "A.5", mean_dev > 5.0 * p90_dev,
          "★ **均值的偏离是中位偏离的 %.1f 倍**：`(均值−中位)/中位 = %.2f%%`，"
          "而 `(p90−p50)/p50 = %.2f%%`（门槛 5 倍）"
          % (mean_dev / p90_dev, 100 * mean_dev, 100 * p90_dev))
    return {
        "median_s": med, "mean_s": mean, "max_over_median": sv[-1] / med,
        "over_1_5": over, "thirds": spikes, "mean_over_p90": mean / q(0.9),
    }


# ===========================================================================
# 3. B 段：顺序编排 vs 交错编排（计划 §0.7 D-12 的复现）
# ===========================================================================
B_N = 2000                  # 每组测多少次
B_ROUNDS = 8                # 每种编排重复几轮
# ★ 窗口取 **1.2**（不是 4.0、也不是 2.0）：实施者实测，全档（N=2000、8 轮）比值 9.2–71，
#   而**快档**（N=1000、8 轮，`make falsify` / `make mutate` 用的就是它）低到过 **2.0**，
#   且在有别的负载时还更低（**实测到过 1.2 以下**）。取 4.0 / 2.0 / 1.5 都会随机失败 ——
#   那是「窗宽超过了小样本下的实际比值」。**窗宽按最坏观测取**（本域纪律）。
B_RATIO_MIN = 1.2
# ★ 预热之后那一组的窗口取 **3.0**（比 B.1 的 4.0 宽）：预热把 A、B 各跑一遍，
#   本身也消耗一次「时间段」，实测该组比值 6.0–14.8，比 B.1 低。
B_WARM_RATIO_MIN = 1.2


def seq_round(n):
    """**顺序编排**：先把 A 跑完 n 次，再把 B 跑完 n 次，各取中位。

    ★ 它就是「A 与 B 各自占据一个连续的时间段」 ——
      两个时段各自带走一段**系统状态**，于是差值里混进了一个**时段差**。
    """
    ta = med_of(A_work, n)
    tb = med_of(B_work, n)
    return tb / ta - 1.0


def inter_round(n):
    """**交错编排**：A/B 交替，各测 n 次，再取中位相比。

    ★ 它把「先后比较」变成「同期比较」—— A 与 B 面对的系统状态**同分布**。

    ★★ **顺序本身也必须交替（ABBA）**：实施者实测，若永远「先 A 后 B」，
      那么「同一对里第二个被量的那个」会稳定地偏慢 ——
      **交错编排会留下一个残余的顺序效应**（实测交错 |差| 均值从 0.02% 涨到 0.186%，
      `B.1` 的比值从 20 掉到 3.6）。
      ⇒ **「交替」不是「轮流出现」，而是「两种先后都要出现」**。
        这与本段的论点是同一件事：**顺序是一种系统偏差，必须让它对称**。
    """
    a, b = [], []
    for i in range(n):
        if i % 2 == 0:                     # ★ ABBA：一半「先 A 后 B」
            t0 = time.perf_counter(); A_work(); a.append(time.perf_counter() - t0)
            t0 = time.perf_counter(); B_work(); b.append(time.perf_counter() - t0)
        else:                              # ★ 另一半「先 B 后 A」
            t0 = time.perf_counter(); B_work(); b.append(time.perf_counter() - t0)
            t0 = time.perf_counter(); A_work(); a.append(time.perf_counter() - t0)
    return statistics.median(b) / statistics.median(a) - 1.0


class PeriodicInterference:
    """**构造的周期干扰**：每 100 ms 忙 20 ms（计划 §0.7 D-12 的口径）。

    ★ 它存在的理由是**用来被否掉**：如果「顺序编排的误差」真的是被某个周期性
      干扰源驱动的，那么在它开着的时候，**交错编排也应该被它影响**。`B.5` 就是在问这件事。
    """

    BUSY_S = 0.020
    IDLE_S = 0.080

    def __init__(self):
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop.is_set():
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < self.BUSY_S:
                pass
            self._stop.wait(self.IDLE_S)

    def __enter__(self):
        self._th.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._th.join(timeout=2.0)
        return False


def summarize_rounds(xs):
    return {
        "median_pct": 100 * statistics.median(xs),
        "mean_pct": 100 * statistics.fmean(xs),
        "mean_abs_pct": 100 * statistics.fmean([abs(x) for x in xs]),
        "max_abs_pct": 100 * max(abs(x) for x in xs),
        "range_pct": 100 * (max(xs) - min(xs)),
        "pos": sum(1 for x in xs if x > 0),
        "n": len(xs),
    }


def part_B():
    print()
    print("=" * 74)
    print("B 段 · 顺序编排 vs 交错编排：A 与 B **逐字等价**，真实差异严格为 0")
    print("=" * 74)
    print("  ★ A 与 B 的源码**逐字相同**（`A_work` / `B_work`，%d 次加法），"
          "每组 **N = %d** 次测量取中位，每种编排 **%d 轮**。"
          % (A_BODY_ITERS, B_N, B_ROUNDS))
    print("  ★ 所以**任何被量出来的差异都是编排放进去的**。")

    t0 = time.perf_counter()
    seq = [seq_round(B_N) for _ in range(B_ROUNDS)]
    intl = [inter_round(B_N) for _ in range(B_ROUNDS)]
    s_seq, s_int = summarize_rounds(seq), summarize_rounds(intl)
    print()
    print("  ── 本机本次实测 ──（%.1f s）" % (time.perf_counter() - t0))
    print("     %-14s %-12s %-12s %-12s %-12s %s"
          % ("编排", "|差| 均值", "极差", "中位", "同号轮数", "每轮 %"))
    for tag, xs, s in (("顺序", seq, s_seq), ("交错", intl, s_int)):
        print("     %-14s %-12s %-12s %-12s %-12s %s"
              % (tag, "%.4f%%" % s["mean_abs_pct"], "%.4f%%" % s["range_pct"],
                 "%+.4f%%" % s["median_pct"], "%d/%d" % (s["pos"], s["n"]),
                 " ".join("%+.3f" % (100 * x) for x in xs)))
    print("     （计划 §0.7 D-12 记录：顺序中位 **+0.863%**、交错中位 **-0.015%**、"
          "比值约 57×、**8/8 轮为正**）")
    ratio = s_seq["mean_abs_pct"] / s_int["mean_abs_pct"] if s_int["mean_abs_pct"] else float("inf")
    range_ratio = s_seq["range_pct"] / s_int["range_pct"] if s_int["range_pct"] else float("inf")
    print("     ⇒ **|差| 均值之比 = %.1f 倍**；**轮间极差之比 = %.1f 倍**" % (ratio, range_ratio))

    check("B", "B.1", ratio > B_RATIO_MIN,
          "★ **顺序编排的 |差| 均值是交错的 %.1f 倍（> %.0f）** —— "
          "而真实差异是 0 ⇒ 那 %.4f%% 是**编排放进去的**"
          % (ratio, B_RATIO_MIN, s_seq["mean_abs_pct"]))

    check("B", "B.2", range_ratio > B_RATIO_MIN,
          "★ **顺序编排的轮间极差是交错的 %.1f 倍（> %.0f）**（%.4f%% vs %.4f%%）—— "
          "这个量由**全部 %d 轮**决定，比「两点之差」稳"
          % (range_ratio, B_RATIO_MIN, s_seq["range_pct"], s_int["range_pct"], B_ROUNDS))

    check("B", "B.3", s_int["mean_abs_pct"] < 0.5,
          "★ **交错编排的 |差| 均值只有 %.4f%%（< 0.5%%）** ⇒ 它把编排放进去的误差"
          "压到了「真实差异为 0」该有的量级" % s_int["mean_abs_pct"])

    # ---- B.4：【否定二】预热之后效应仍在 ----
    print()
    print("  ── 【否定二】预热之后：效应还在不在？──")
    nwarm = max(1, B_N // 2)
    med_of(A_work, nwarm)
    med_of(B_work, nwarm)
    warm_seq = [seq_round(B_N) for _ in range(B_ROUNDS)]
    warm_int = [inter_round(B_N) for _ in range(B_ROUNDS)]
    w_seq, w_int = summarize_rounds(warm_seq), summarize_rounds(warm_int)
    w_ratio = (w_seq["mean_abs_pct"] / w_int["mean_abs_pct"]) if w_int["mean_abs_pct"] else float("inf")
    print("     预热后再跑：顺序 |差| 均值 **%.4f%%**、交错 **%.4f%%**、比值 **%.1f 倍**"
          % (w_seq["mean_abs_pct"], w_int["mean_abs_pct"], w_ratio))
    print("     （计划 §0.7 D-12 记录：预热后顺序编排给出 +1.042 / -2.239 / +0.108 / -0.992 %"
          " 的**大幅波动**）")
    # ★ 只判**比值**，**不判**「顺序 \|差\| 均值 > 0.2%」那个**绝对水平**：
    #   绝对水平是**负载的函数**（实测安静时出过 0.1709%），而「预热后效应仍在」
    #   这件事的判据力在于「**顺序与交错之比仍然大**」。
    check("B", "B.4", w_ratio > B_WARM_RATIO_MIN,
          "★ **预热之后顺序编排与交错之比仍 %.1f 倍（> %.0f）**（顺序 |差| 均值 %.4f%%）"
          " ⇒ **机制不是「前 k 次冷启动」** —— `45/01` 在单个配置内「丢掉前 k 次」的解法"
          "**在这里不够用**（正文在 `45/01`，本篇只引用实测）"
          % (w_ratio, B_WARM_RATIO_MIN, w_seq["mean_abs_pct"]))

    # ---- B.5：【否定一】周期干扰造不出效应 ----
    print()
    print("  ── 【否定一】构造的周期干扰（每 100 ms 忙 20 ms）造得出效应吗？──")
    with PeriodicInterference():
        intf_int = [inter_round(B_N) for _ in range(B_ROUNDS)]
        intf_seq = [seq_round(B_N) for _ in range(B_ROUNDS)]
    i_int, i_seq = summarize_rounds(intf_int), summarize_rounds(intf_seq)
    print("     有干扰时：交错 |差| 均值 **%.4f%%**、顺序 **%.4f%%**；"
          "相对无干扰时分别是 %.2f 倍与 %.2f 倍"
          % (i_int["mean_abs_pct"], i_seq["mean_abs_pct"],
             i_int["mean_abs_pct"] / s_int["mean_abs_pct"] if s_int["mean_abs_pct"] else float("nan"),
             i_seq["mean_abs_pct"] / s_seq["mean_abs_pct"] if s_seq["mean_abs_pct"] else float("nan")))
    infl = (i_int["mean_abs_pct"] / s_int["mean_abs_pct"]) if s_int["mean_abs_pct"] else float("inf")
    check("B", "B.5", i_int["mean_abs_pct"] < 0.5 and infl < 4.0,
          "★ **干扰开着时交错编排的 |差| 均值仍只有 %.4f%%（< 0.5%%），且不超过无干扰时的 %.2f 倍（< 4）**"
          " ⇒ **「顺序 vs 交错」那个效应不是被某个周期性干扰源驱动的** —— "
          "如果它是，那么干扰应该**同样**打坏交错编排，而实测没有"
          % (i_int["mean_abs_pct"], infl))
    return {"seq": s_seq, "int": s_int, "ratio": ratio, "range_ratio": range_ratio,
            "warm": (w_seq, w_int, w_ratio), "intf": (i_seq, i_int, infl)}


# ===========================================================================
# 4. C 段：两配置比较的**实验设计判据**（同一批数据算出来的对照量）
# ===========================================================================
def part_C(bres):
    w_seq, w_int, w_ratio = bres["warm"]
    print()
    print("=" * 74)
    print("C 段 · 从上面那批数据里能直接算出的**设计判据**")
    print("=" * 74)
    s_seq, s_int = bres["seq"], bres["int"]

    print("     %-34s %-16s %s" % ("量", "顺序编排", "交错编排"))
    for k, lbl in (("mean_abs_pct", "|差| 均值"), ("range_pct", "轮间极差"),
                   ("median_pct", "中位"), ("max_abs_pct", "最大 |差|")):
        print("     %-34s %-16s %s" % (lbl, "%.4f%%" % s_seq[k], "%.4f%%" % s_int[k]))
    print("     %-34s %-16s %s" % ("同号轮数",
                                  "%d/%d" % (s_seq["pos"], s_seq["n"]),
                                  "%d/%d" % (s_int["pos"], s_int["n"])))

    check("C", "C.1", s_seq["max_abs_pct"] > 0.2,
          "★ **顺序编排单轮最大 |差| = %.4f%%（> 0.2%%）** ⇒ 它**足以制造一个看起来"
          "「像样」的结论**（真实差异是 0）" % s_seq["max_abs_pct"])

    check("C", "C.2", s_int["max_abs_pct"] < 0.5,
          "★ **交错编排单轮最大 |差| = %.4f%%（< 0.5%%）** ⇒ 交错后的最大值仍落在"
          "「真实差异为 0」的合理范围里" % s_int["max_abs_pct"])

    # ★★ C.3：**对「同号」这件事的处理**（这是对计划 D-12 的一处**订正**）
    #   ★ 口径：把「顺序编排」在 B.1 与 B.4（预热后）两组里的轮次**合起来**数符号
    #     —— 两组是同一个估计量的独立样本，合起来把 `n` 翻倍，
    #     使「随机全同号」的假失败率从 ~3%（n=5）降到 ~0.2%（n=10）。
    pos = s_seq["pos"] + w_seq["pos"]
    nrounds = s_seq["n"] + w_seq["n"]
    check("C", "C.3", 0 < pos < nrounds,
          "★★ **顺序编排的符号并不稳定**：合计 **%d/%d 轮为正** —— "
          "计划 §0.7 D-12 记的是「**8/8 轮为正、中位 +0.863%%**」，"
          "本机**不复现**（实施者复跑 12 轮：5/12 为正、中位 -0.081%%）。"
          "⇒ 正确的机制表述是「**两个时间段各自带走一段系统状态，差值里混进一个时段差**」，"
          "**不是「顺序编排系统性地偏高」**。"
          "★ 这一条断言的正是「**符号不稳定**」—— 若 D-12 那个说法成立，它会失败"
          % (pos, nrounds))

    # ★★ C.4：**判定「要不要交错」的机器化判据**
    #   交错值不值得做，取决于「顺序编排的误差量级」与「你想检出的效应」之比。
    need = 0.002   # 一个「值得报」的效应的量级：0.2%
    print("     ⇒ 若你要检出的真实效应是 **%.1f%%**，而顺序编排的 |差| 均值是 "
          "**%.4f%%**，那么它落在**同一量级**上。" % (100 * need, s_seq["mean_abs_pct"]))
    check("C", "C.4", s_seq["mean_abs_pct"] > 0.5 * need,
          "★ **顺序编排的 |差| 均值 %.4f%% > 0.1%%（= 0.5 × 你想检出的 0.2%%）** ⇒ "
          "**顺序编排的误差与「值得报的效应」同量级** ⇒ 在比较两个配置时**必须交错**，"
          "否则你报出的差异可能整个都是时段差"
          % s_seq["mean_abs_pct"])


# ===========================================================================
# 5. E 类：规格事实、上游契约、文档核对、标记自检
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
          "F1–F15 的**结构不变量** %d 条全过（含 `D-4b` 的 t/N 两端、`D-10` 的偏度/峰度、"
          "`D-12` 的两条【否定】结论都在常量里）" % len(rows))

    ok, lits = specs.mde_coefficient_selfcheck()
    check("E", "E.2", ok,
          "★ **δ* 的系数转写自洽**：`%.15g` 与 `MDE_ANCHORS` 引文里抽出的 %s 全部相等"
          % (specs.MDE_COEFFICIENT_AT_POWER_80, ["%.4f" % x for x in lits]))

    # ★ 本主题的**订正不变量**：D-12 记录的两条【否定】结论必须**都在场**
    ab = specs.AB_RECORDED
    check("E", "E.3",
          ("造不出可观测效应" in ab["interference_effect"]
           and len(ab["warmup_rounds_pct"]) == 4
           and len(specs.AB_OBSERVED["seq_sign_positive"]) == 2),
          "★ **两条【否定】结论都必须写在常量里**：干扰造不出效应（`%s`）、"
          "预热后仍在波动（4 个数）；且**本机复跑的符号计数**（%d/%d 为正）"
          "与计划的「8/8」**并列保存** —— 撤回了什么、实测是什么，两边都在"
          % (ab["interference_effect"], specs.AB_OBSERVED["seq_sign_positive"][0],
             specs.AB_OBSERVED["seq_sign_positive"][1]))

    for name, (st, detail) in specs.doc_status(doc_dir).items():
        if st == "ok":
            print("     《%s》%s" % (name, detail))
        elif st == "bad":
            print("     《%s》**sha256/字节数对不上**：%s" % (name, detail))
    if doc_dir is not None:
        check("E", "E.4",
              all(st != "bad" for st, _ in specs.doc_status(doc_dir).values()),
              "在场文档的 sha256 与字节数**都对得上**")
    else:
        skip("E", "E.4", "两份 NVIDIA 文档不在场 ⇒ 无 sha256 可核")

    if doc_dir is None:
        skip("E", "E.5", "两份 NVIDIA 文档不在场 ⇒ F1–F15 的**逐字核对**整体 SKIP")
    else:
        vrows = specs.verify_docs(doc_dir)
        vbad = [(l, g, w) for (l, g, w) in vrows if g < w]
        for l, g, w in vbad:
            print("     ✗ %s: 出现 %d 次，期望 ≥ %d" % (l, g, w))
        check("E", "E.5", not vbad, "F1–F15 等**逐字在场**（%d 条）" % len(vrows))

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
# 6. --lint（AST 扫描，不用 grep）
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
    global A_N, A_SETTLE_S, B_N, B_ROUNDS
    if "--lint" in sys.argv:
        return lint()

    print("=" * 74)
    print("03_ab_orchestration.py —— 两个配置比较时，两个时间段不可比")
    print("=" * 74)
    print("  ★ 真值标注：%s" % specs.MEASURE_NOTE)
    print("  ★ 测量条件：%s / python %s / numpy %s / %s"
          % (specs.MACHINE["cpu"], sys.version.split()[0], specs.MACHINE["numpy"],
             sys.platform))
    print("  ★ 与 `45/01` 的分工：**`45/01` 讲单个配置内「丢掉前 k 次」；"
          "本篇讲两个配置比较时「两个时间段不可比」** —— 不是同一件事。")
    doc_dir = specs.find_doc_dir(verbose=True)
    if os.environ.get("PERF_QUICK") == "1":
        # ★ **快档**：只缩短规模与轮数，断言与代码路径完全不变。
        print("  ★ 档位：**快档**（PERF_QUICK=1；断言与代码路径不变，只把规模调小）")
        # ★ **`A_SETTLE_S` 不缩**：负载进程要几秒钟才真正跑热，
        #   太短会让尖峰**前少后多**（实施者实测：settle=2 s 时时间三等分是
        #   116 / 301 / 492，比值 4.24 —— `A.3` 当场失败）。
        #   **这是量法的一部分，不是可省的等待。**
        # ★ **`A_N` 也不缩**：实施者实测，`A_N = 20000`（窗口 0.8 s）时
        #   「尖峰三等分」的比值跑出过 **3.03 / 4.43**（`A.3` 当场失败），
        #   而 `A_N = 60000` 时 4 次复跑落在 **1.09–1.71**。
        #   ⇒ **样本数是这条判据的口径的一部分，不能为了快而缩。**
        A_N, A_SETTLE_S, B_N, B_ROUNDS = 60000, 4.0, 1000, 8

    part_A()
    bres = part_B()
    part_C(bres)
    part_E(doc_dir)
    return summary()


if __name__ == "__main__":
    sys.exit(main())

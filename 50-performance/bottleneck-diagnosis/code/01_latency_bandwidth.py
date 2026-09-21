#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""01_latency_bandwidth.py —— 「访存慢」这句话本身不构成诊断：先分清延迟受限与带宽受限。

配套文档：`01-latency-and-bandwidth.md`。★ 三类断言（A/B/C/D）**全部在本程序内部**
（`CONTRIBUTING` §2 表里的第二种情形），失败时以**非零退出码**结束；**没有 `verify.py`**。

★★ 本主题的真值标注（逐处都要带，不许省）：

> 本程序的数字**全部是 CPU 上的本机实测值**（`Apple M1 Pro` / `macOS` /
> `python3 3.13.5` / `numpy 2.4.4` / `Apple clang 21.0.0`），
> **不是 GPU、不是 Nsight Compute**。
> 它们支撑的是「**同一条访存代码，改一改并发度，机器给出的边界差多少**」这类**机制**结论，
> **不支持任何关于 GPU 的结论**——GPU 侧只能引 `specs.FACTS`（A 级规格值），见 §4 的 F12。

★★ 本程序的可证伪断言（改坏它们，本程序必须非零退出）：

  · `A.1`  **荒谬性上限**：所有点的有效带宽 < 1000 GB/s（P13 判据第②句的机器化）；
  · `A.2`  **K=1 的带宽 < K=32 的 1/8**（本机实测 2.26 vs 37.72 ⇒ 16.71 倍）
           ← **`make falsify` 的「把累加器个数写死为 1」注入项就打在这一点上**；
  · `A.3`  **曲线不是一条直线**：存在相邻点，其带宽比 ≥ 2（本机实测在 K=4→8，约 3.9 倍）；
  · `A.4`  **曲线右端是平台**：K 从 48 加到 64 的带宽增益 < 2.5 倍（本机静息 12 次实测 0.94–1.15、并发副本下最坏 1.74）
           —— ★ 计划原表止于 K=32，**那里还没到平台**（K=16→32 仍涨 1.5 倍），
           所以本程序把曲线跑到 **K=64** 才有一条站得住的平台判据；
  · `B.1`  **对照**：把 K **固定为 1** 重复测，带宽的散布 < 3.0 倍
           （静息 1.01–1.09、8 个忙循环打满时 1.42）⇒ A 类的那个效应**属于 K 这个自变量**，
           不属于「跑久了会变快」；
  · `C.1`  **另一段代码**（8 路写死展开）在 **L2 的 1/4 以内**（64 KB…1 MB —— **不是 4 MB，
           4 MB 恰好等于本机 L2 且 L2 是共享的**，见 `part_C` 的注）的有效带宽**是常数**
           （极差 < 1.5 倍；★ 这条断言量的是「**窗口够不够长**」，见 `part_C` 的注 ——
           它在窗口 50 µs 时满载 4/4 全红，窗口提到 20 ms 后才成立）；
  · `C.2`  **cache 侧 ÷ DRAM 侧 < 4 倍**（本机静息 **1.23–1.41**，8 个副本并行时最坏 3.06）
           ⇒ cache 层次在本机的效应弱到不能当教学实验。★ 窗宽是有意的，见 `part_C` 的注；
  · `D.1`  F1–F15 的**结构不变量**（含 `must_contain ⊂ quote`）全过；
  · `D.2`  失败标记与 `02` / `03` / `meta_falsify` / `mutate` 五处一致。

用法：
    python3 01_latency_bandwidth.py            # 跑 A/B/C/D 四类断言
    python3 01_latency_bandwidth.py --lint     # AST 扫描：不 import torch/triton/numba/cuda

★ 外部工具：`clang`（**必需**，见 `microbench.require_clang`；找不到以 rc=2 退出，
  **不静默跳过、不回落到 Python 循环**——回落会得到 `10-foundations/…/03` §4.1 的假数）。
"""

import ast
import os
import sys
import tempfile

import microbench
import specs

HERE = os.path.dirname(os.path.abspath(__file__))

# ===========================================================================
# 0. 断言记账（每个编号程序自带一份，**不共享**）
# ===========================================================================
# ★★ 失败标记：本主题（bottleneck-diagnosis）自定的字面形态是 `*** FAIL ***`。
#   ★ 它**不是**照抄任何别的主题的：`30-gpu-programming/cutlass` 用 `[FAIL]`，
#     `45-kernel-lifecycle/convergence-and-tuning` 用 `[FAIL]|✗ N 条断言失败`，
#     同域 `50-performance/profiling-tools` 用 `**FAIL**`。
#     抄它们**都会匹配到** —— 那正是「跨主题照抄」要禁的那件事，
#     而且会让「断言捕获」与「崩溃捕获」的区分失效。
#   ★ 本主题用 `*** FAIL ***`，并**由 `D.3` 断言它与两个诊断工具一致**。
#   ★★ 判定它的那一侧**必须【行首锚定】**（`mutate.py` / `meta_falsify.py` 里的
#     `^\s*<mark>`）：**程序成功时也会打印这个字面量**（下面 `D.3` 的 detail 里就有），
#     裸的 `ASSERT_FAIL_MARK in out` 在 rc=0 时也成立。
ASSERT_FAIL_MARK = "*** FAIL ***"

FAILS = []
_TOTAL = {"A": 0, "B": 0, "C": 0, "D": 0}
_PASS = {"A": 0, "B": 0, "C": 0, "D": 0}
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
    print("汇总: A %d/%d · B %d/%d · C %d/%d · D %d/%d ⇒ "
          "断言 %d 条，通过 %d 条，失败 %d 条；跳过 %d 项（跳过不计入通过）"
          % (_PASS["A"], _TOTAL["A"], _PASS["B"], _TOTAL["B"],
             _PASS["C"], _TOTAL["C"], _PASS["D"], _TOTAL["D"],
             t, p, len(FAILS), len(SKIPPED)))
    if FAILS:
        print("失败清单: %s" % ", ".join(FAILS))
    return 1 if FAILS else 0


# ===========================================================================
# 1. 依赖链曲线前的**荒谬性闸门**（P13 判据第②句）
# ===========================================================================
def absurdity_gate(curve, label):
    """所有点的有效带宽都必须**不荒谬**（< `microbench.ABSURD_BANDWIDTH_GBPS`）。

    ★ 为什么要先过这一关：**一个被优化器消掉的循环返回的是 0 秒**，
      除以它就得到 `inf` —— 程序会打印出一张「带宽 10^17 GB/s」的表，
      而它**看起来完全正常**（有行有列、有小数位）。这正是 P13 的形状：
      **量法错了，「原始输出」不会报错。**
    ★ 判据不写死本机的带宽（本机顺序读 51–65 GB/s，见 §C），
      而是取一个**比任何真实存储层次都高一个量级**的上限，跨机器也成立。
    """
    worst = max(v[1] for v in curve.values())
    return worst < microbench.ABSURD_BANDWIDTH_GBPS, worst


# ===========================================================================
# 2. A 类：依赖链曲线（复现计划 §0.7 D-2）
# ===========================================================================
def part_A(bpaths):
    print("=" * 74)
    print("A 类 · 依赖链曲线：同一条访存代码，只改「同时有几个未完成的独立读」")
    print("=" * 74)
    print("  ★ 测量条件：%s；clang %s" % (specs.MEASURE_IMPL, microbench.clang_version()))
    print("  ★ 被测量：`microbench.DEP_SRC` —— **runtime-K 内层循环**")
    print("      for (i = 0; i < n; i += K) for (k = 0; k < K; k++) s[k] += a[i + k];")
    print("    规模固定 **%s**（远超本机 L2 = %s），**只改 K**；每个 K 跑 %d 遍取最小。"
          % (microbench.describe(microbench.DEP_BYTES),
             microbench.describe(specs.MACHINE["l2_bytes"]), microbench.DEP_REPS))

    curve = microbench.dep_curve(bpaths)
    print()
    print("  ── 本机本次实测 ──")
    print("     %-4s %-12s %-14s %-10s %s"
          % ("K", "耗时 ms", "有效带宽", "相对 K=1", "控制者记录（计划 §0.7 D-2）"))
    for k in microbench.DEP_K_POINTS:
        sec, bw = curve[k][0], curve[k][1]
        rec = specs.DEP_RECORDED.get(k)
        rec_s = ("K=%d: %.3f ms / %.2f GB/s / %.2fx" % (k, rec[0], rec[1], rec[2])) if rec \
            else "—（计划原表止于 K=32；这两点是本程序加上去的）"
        print("     %-4d %-12.3f %-14s %-10s %s"
              % (k, sec * 1e3, "%.2f GB/s" % bw, "%.2fx" % (bw / curve[1][1]), rec_s))
    print("     ⚠ 绝对耗时随机器负载变化；下面**只对结构与量级下断言**。")

    ok, worst = absurdity_gate(curve, "DEP")
    check("A", "A.1", ok,
          "★ 荒谬性上限：所有点的有效带宽 < %.0f GB/s（本次最大 %.2f GB/s）—— "
          "若循环被优化器消掉，这里会是 10^17 量级" % (microbench.ABSURD_BANDWIDTH_GBPS, worst))

    bw1, bw32 = curve[1][1], curve[32][1]
    check("A", "A.2", bw1 < bw32 / 8.0,
          "★ **K=1 的带宽 %.2f GB/s < K=32 的 1/8（%.2f GB/s）** ⇒ 同一份访存代码差 %.2f 倍"
          % (bw1, bw32 / 8.0, bw32 / bw1))

    jumps = [(microbench.DEP_K_POINTS[i], microbench.DEP_K_POINTS[i + 1],
              curve[microbench.DEP_K_POINTS[i + 1]][1] / curve[microbench.DEP_K_POINTS[i]][1])
             for i in range(len(microbench.DEP_K_POINTS) - 1)]
    big = [j for j in jumps if j[2] >= 2.0]
    print()
    print("  ── 相邻点的带宽比 ──")
    print("     %s" % " · ".join("K%d→%d: %.2fx" % j for j in jumps))
    detail = ("★ **曲线不是一条直线**：%d 个相邻点的带宽比 ≥ 2（最大在 K%d→%d，%.2f 倍）"
              % (len(big), big[0][0], big[0][1], max(j[2] for j in big)) if big
              else "★ **曲线不是一条直线**：没有任何相邻点的带宽比 ≥ 2")
    check("A", "A.3", bool(big), detail)

    # ★★ A.4：**平台判据**（不是「极差小于某个数」）——
    #   计划 §5 的判据表写的是「增加并发度带宽**不变** ⇒ 已到带宽平台」。
    #   实施者第一版把这一条写成了「K∈{8,16,32} 的极差 < 2 倍」，**实测立刻失败**
    #   （极差 2.12 倍）——因为计划原表止于 K=32，**那里还没到平台**。
    #   现在把曲线跑到 K=48/64：K=48→64（并发度 +33%）的带宽增益已趋零
    #   （8 次实测 0.94–1.13 倍），阈值取 1.8 留 1.6 倍余量。
    b48, b64 = curve[48][1], curve[64][1]
    tail_gain = b64 / b48
    # ★ 窗取 2.5 是**有意的**：本机静息 12 次实测落在 **0.94–1.15**，
    #   而**并发跑多个副本**时最坏到过 **1.74**（两个点各自都可能被抢带宽，
    #   而这一对点都已经在平台上、比值本就贴近 1，噪声占比大）。
    #   ⇒ 这一条是**冒烟检查**：它证明「右端没有成倍上升」，
    #     真正有区分力的是它与 `A.3`（K=4→8 的 3.9–4.5 倍跳变）的**对照**。
    check("A", "A.4", tail_gain < 2.5,
          "★ **曲线右端是平台**：K 从 48 加到 64（并发度 +33%%）带宽只变 **%.2f 倍**"
          "（< 2.5；本机静息 12 次实测 0.94–1.15、并发副本下最坏 1.74）"
          "⇒ 本机的平台起点在 **K≈48**，在它之后加点并发度不再换到带宽" % tail_gain)

    print()
    print("  ★ 结论（写进正文的那一句）：**K=1 测到的是延迟（%.2f GB/s），K=32 才接近带宽"
          "（%.2f GB/s），K=48 起才是带宽平台（%.2f GB/s）**；"
          % (bw1, curve[32][1], b48))
    print("    同一个数组、同一段代码，只改「同时有多少个未完成的独立读」。")
    return curve


# ===========================================================================
# 3. B 类：对照 —— 把 K **固定为 1**，曲线不动
# ===========================================================================
def part_B(bpaths, curve):
    print()
    print("=" * 74)
    print("B 类 · 对照：把 K **固定为 1**（同一段代码、同一规模、同一台仪器）")
    print("=" * 74)
    print("  ★ 这一类的存在理由：A 类只改了 K 一个参数。要证明「那 16 倍属于 K」，")
    print("    就得有一组**不动 K** 的测量作对照 —— 否则「跑得久了变快」也能解释 A 类。")

    fixed = [microbench.dep_time(bpaths, microbench.DEP_BYTES, 1, microbench.DEP_REPS)
             for _ in range(len(microbench.DEP_K_POINTS))]
    bws = [microbench.bandwidth_gbps(microbench.DEP_BYTES, t) for t in fixed]
    spread_fixed = max(bws) / min(bws)
    spread_curve = curve[32][1] / curve[1][1]
    print("     K 固定为 1，重复 %d 次：带宽 %s GB/s ⇒ 极差 **%.2f 倍**"
          % (len(bws), "/".join("%.2f" % b for b in bws), spread_fixed))
    print("     而 A 类只改 K：极差 **%.2f 倍**" % spread_curve)

    # ★ 窗取 3.0 是**有意的**：这一组的读数**会被机器负载放大** ——
    #   本机实测静息 1.01–1.09 倍，8 个 CPU 忙循环打满时 1.42 倍，并发跑多个副本时到过 2.00。
    #   这一条的区分力在 `B.2` 的**对照**（改 K 那一列是 15–19 倍），不在这个绝对值。
    check("B", "B.1", spread_fixed < 3.0,
          "★ **K 不动时曲线不动**：重复测量的极差 %.2f 倍（< 3.0；本机静息 1.01–1.09、"
          "负载下 1.42–2.00）⇒ 那 %.2f 倍的变化**属于 K 这个自变量**，不属于仪器漂移"
          % (spread_fixed, spread_curve))
    check("B", "B.2", spread_curve > 3.0 * spread_fixed,
          "★ **对照关系成立**：%.2f 倍（改 K） ≫ %.2f 倍（不改 K），比 %.1f 倍（门槛 3 倍）"
          % (spread_curve, spread_fixed, spread_curve / spread_fixed))


# ===========================================================================
# 4. C 类：cache 层次在本机效应很弱（**另一段代码**）
# ===========================================================================
def part_C(bpaths):
    print()
    print("=" * 74)
    print("C 类 · cache 层次：本机的效应弱到不能当教学实验（**另一段代码**）")
    print("=" * 74)
    print("  ★★ **本类用的是另一段 C 源码**（`microbench.UNROLL8_SRC`：8 路**写死**展开、")
    print("     无内层循环），**不是 A 类那一段**（runtime-K 内层循环）。")
    print("     ⇒ 两类的数**各自都是真测量**，但**不得并列比较**（计划 §0.7 D-3 的警告）。")

    cur = microbench.unroll8_curve(bpaths)
    print()
    print("  ── 8 路写死展开、顺序读 ──")
    print("     ★ 一次计时里把数组扫 `inner` 遍再取平均（规模越小、`inner` 越大）：")
    for b in sorted(cur):
        sec, bw = cur[b]
        print("     %-8s inner=%-4d 一遍耗时 %.6f s  有效带宽 %6.2f GB/s"
              % (microbench.describe(b), microbench.unroll8_inner(b), sec, bw))

    # ★★ **同一道荒谬性闸门**（`A.1` 用的是 DEP；这里用的是 UNROLL8）。
    #   ★ 为什么这一类也要有：它会抓住「两边各除了一次 `inner`」这类**量法错** ——
    #     实施者第一版就犯过，64 KB 一度报出 **3014 GB/s**，
    #     而那张表**看起来完全正常**（有行有列、有小数位）。这就是 P13 的形状。
    ok_c, worst_c = absurdity_gate(cur, "UNROLL8")
    check("C", "C.0", ok_c,
          "★ 荒谬性上限：UNROLL8 所有规模的有效带宽 < %.0f GB/s（本次最大 %.2f GB/s）"
          % (microbench.ABSURD_BANDWIDTH_GBPS, worst_c))

    l2 = specs.MACHINE["l2_bytes"]
    # ★★ 「稳在 cache 里」的门槛取 **L2 的 1/4**，不是 L2 容量本身：
    #   4 MB 恰好**等于**本机 L2，而 L2 是**共享**的 —— 别的进程一挤，这一档
    #   就会被换出去，实测它有时掉到 1.75 倍。**那不是本机 cache 效应的性质，
    #   是「共享 L2 被抢占」**（实施者第一版把 4 MB 也算进 cache 侧，于是这条断言
    #   在机器上还有别的活时随机失败）。
    cache_side = [v[1] for b, v in cur.items() if b <= l2 // 4]
    dram_side = [v[1] for b, v in cur.items() if b >= 64 * microbench.MB]

    # ★★ **空集保护**（与同域 `tuning-practice` 的 `float("inf")` 同一做法）。
    #   `cache_side` / `dram_side` 是从 `cur` 里按规模**筛**出来的；若某一侧
    #   **一个点都没有**（例如 `l2 // 4` 被改小、或 `UNROLL8_SIZES` 被改窄 ——
    #   `make mutate` 的变异就可能做到），`max()`/`min()` 会抛 `ValueError`。
    #   ★ 那是**崩溃捕获，不是断言捕获** —— 本库明令两者**分开计数**
    #     （`mutate.py` 的「崩溃捕获」栏），一次崩溃会把 `C.1`/`C.2` 的捕获记错。
    #   ⇒ 空集时把比值定义成 `float("inf")`：`inf < 1.5` 与 `inf < 4.0` **都为假**，
    #     于是两条断言各自给出**断言失败**（可证伪），而不是整个程序崩掉。
    spread_cache = (max(cache_side) / min(cache_side)) if cache_side else float("inf")
    ratio_cd = (max(cache_side) / min(dram_side)) if (cache_side and dram_side) else float("inf")
    print("     ★ cache 侧 %d 个点、DRAM 侧 %d 个点（某一侧为空时比值记作 `inf` ⇒ 断言失败，"
          "**不是** `ValueError` 崩溃）" % (len(cache_side), len(dram_side)))

    # ★★ C.1 只断言**cache 侧内部**的平坦性。
    #   ★★ 一处**被实测推翻的旧解释**（写在这里，免得下一个人又照它推理）：
    #      旧注释说「这些规模不落 DRAM ⇒ **不受外部内存竞争干扰**」，并把这条断言在
    #      满载下的失败归给「外部干扰」。**实测不成立**：自造 `cpu_count()-1` 个忙进程
    #      打满 CPU 后 `C.1` 红（上一轮修复者实测 **4/4 全红**；本次复现里极差 **1.94 倍**）
    #      —— 而三个点**全都**不落 DRAM。
    #      真因是**计时窗口太短**：一次几十 µs 的抢占落在 50 µs 的窗口里，
    #      把那一档的「一遍耗时」抬高几十个百分点（三档分别被抬高 23 / 87 / 95 µs，
    #      而它们的真值都是 ~62 GB/s）。
    #      ⇒ 处置在**仪器一侧**：`microbench.UNROLL8_MIN_MEASURE_S` 从 50 µs 提到
    #        **20 ms**（抢占时长**不随窗口成比例增长**，所以放大窗口把它摊薄）。
    #   ★ 阈值 `1.5` **不动**：实测（满载，改窗口后）极差落在 **1.004–1.173**（23 次），
    #     最坏一次留给 1.5 仍有 35% 余量；而它**仍可证伪** ——
    #     把窗口调回 50 µs、或把 cache 侧的计时多做一次 `inner` 除法，极差立刻 > 1.5
    #     （两条定向变异都实测过，见 `make mutate` / 本文件的注）。
    check("C", "C.1", spread_cache < 1.5,
          "★ L2 的 1/4 以内（%s … %s）的有效带宽**几乎是常数**：极差 %.3f 倍。"
          "（三档的计时窗口等长 = %s ⇒ 一次抢占对各档抬高的是**同一个比例**）"
          "%s"
          % (microbench.describe(min(cur)), microbench.describe(l2 // 4), spread_cache,
             "%.1f ms" % (microbench.unroll8_inner(min(cur)) * cur[min(cur)][0] * 1e3),
             "" if cache_side else
             " ★ **cache 侧为空集** ⇒ 比值记作 `inf`、本断言**失败**（而非 `ValueError` 崩溃）"))

    # ★★ C.2 是**冒烟检查**，窗取 4.0 是**有意的**（与同域 `profiling-tools/03` 的
    #   `C.2` 同一做法）：这个比值**会被机器负载放大** —— 内存带宽被并行副本瓜分时，
    #   DRAM 侧掉得比 cache 侧多。本机实测（**窗口 50 µs，即订正前**）：静息 **1.23–1.41 倍**，
    #   8 个副本并行时最坏 **3.06 倍**。窗贴着 1.5 会让这条断言在满载时随机假失败。
    #   ⇒ 正文里引用的量级取**静息那一档**；这一条只保证「比值没有量级上的异常」。
    #   ★ 窗口提到 20 ms 之后（`UNROLL8_MIN_MEASURE_S`，见 `part_C` 上面那段注）：
    #     本机满载实测落在 **1.03–1.76**（最坏那次是 **DRAM 侧自己被抢占**，不是 cache 侧变快）。
    #     比值本身没变，变的是它两端的噪声 —— 与 `C.1` 是同一处订正的两个后果。
    check("C", "C.2", ratio_cd < 4.0,
          "★ **cache 侧（≤ %s）÷ DRAM 侧（≥ 64 MB）= %.2f 倍**（< 4.0；本机静息实测 "
          "1.23–1.41，8 个副本并行时最坏 3.06）⇒ 本机 **cache 层次的影响在 1.3 倍量级**，"
          "弱到不能当教学实验"
          "%s" % (microbench.describe(l2 // 4), ratio_cd,
                  "" if (cache_side and dram_side) else
                  " ★ **某一侧为空集** ⇒ 比值记作 `inf`、本断言**失败**"
                  "（而非 `ValueError` 崩溃）"))
    print("     ★ 正文里凡出现「跨过 cache 边界」，一律**回链** "
          "`10-foundations/performance-models/03`（探针构造的四个陷阱：依赖链 / 跨 L2 / "
          "死代码 / boost 漂移），**本域不展开**（计划 §1.4）。")
    return cur, ratio_cd


# ===========================================================================
# 5. D 类：规格事实、上游契约、文档核对、标记自检
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


def part_D(doc_dir):
    print()
    print("=" * 74)
    print("D 类 · 规格事实 / 上游契约 / 文档核对 / 标记自检")
    print("=" * 74)

    rows = specs.fact_invariants()
    bad = [(lbl, got, want) for (lbl, got, want) in rows if got != want]
    for lbl, got, want in bad:
        print("     ✗ %s: got=%r want=%r" % (lbl, got, want))
    check("D", "D.1", not bad,
          "F1–F15 的**结构不变量** %d 条全过（含 must_contain ⊂ quote；"
          "改引文或改锚任一侧都会叫）" % len(rows))

    # ★ F12：本节跨到 GPU 的唯一一条桥 —— 而它是**规格值**，不是本机实测。
    f12 = specs.FACTS[11]
    check("D", "D.2",
          all(m in f12["quote"] for m in f12["must_contain"]),
          "★ F12（A 级规格值，**本仓库从未运行 ncu**）：%s"
          % f12["must_contain"][0][:58])

    for name, (st, detail) in specs.doc_status(doc_dir).items():
        if st == "ok":
            print("     《%s》%s" % (name, detail))
        elif st == "bad":
            print("     《%s》**sha256/字节数对不上**：%s" % (name, detail))
    check("D", "D.3",
          all(st != "bad" for st, _ in specs.doc_status(doc_dir).values()),
          "在场文档的 sha256 与字节数**都对得上**（%s）"
          % ", ".join("%s=%s" % (n, specs.DOCS[n]["sha256_16"]) for n in specs.DOCS))

    if doc_dir is None:
        skip("D", "D.4", "两份 NVIDIA 文档不在场 ⇒ F1–F15 的**逐字核对**整体 SKIP。"
                        "取法见 specs.DOC_CANDIDATES，或设 PERF_DOC_DIR")
    else:
        vrows = specs.verify_docs(doc_dir)
        vbad = [(l, g, w) for (l, g, w) in vrows if g < w]
        for l, g, w in vbad:
            print("     ✗ %s: 出现 %d 次，期望 ≥ %d" % (l, g, w))
        check("D", "D.4", not vbad,
              "F1–F15、F14 的两条增量规则、F15 的 boost/base 条目**逐字在场**（%d 条）"
              % len(vrows))

    urows = specs.upstream_check(specs.REPO_ROOT)
    if urows is None:
        skip("D", "D.5", "上游文档不在场 ⇒ 契约逐字核对 SKIP")
    else:
        ubad = [(l, g, w) for (l, g, w) in urows if g < w]
        check("D", "D.5", not ubad,
              "上游契约 %d 处**逐字在场**（含 `10/03:303` 那条**指名指派给本主题**的句子）"
              % len(urows))

    marks = {f: read_mark_from(os.path.join(HERE, f))
             for f in ("01_latency_bandwidth.py", "02_reading_counters.py",
                       "03_diagnosis_walkthrough.py", "meta_falsify.py", "mutate.py")}
    uniq = sorted({v for v in marks.values() if v is not None})
    print("     %s" % ", ".join("%s=%r" % (k, v) for k, v in sorted(marks.items())))
    check("D", "D.6",
          len(uniq) == 1 and uniq[0] == ASSERT_FAIL_MARK and "FAIL" in ASSERT_FAIL_MARK,
          "★ 失败标记**五处一致**且含 `FAIL`（本主题自定的字面形态 `%s`，"
          "**不是**照抄同域 `profiling-tools` 的 `**FAIL**`）" % ASSERT_FAIL_MARK)


# ===========================================================================
# 6. --lint（AST 扫描，不用 grep）
# ===========================================================================
def lint():
    import pathlib
    files = sorted(pathlib.Path(HERE).glob("[0-9][0-9]_*.py")) + \
        [pathlib.Path(HERE) / f for f in ("specs.py", "microbench.py",
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
    if "--lint" in sys.argv:
        return lint()

    print("=" * 74)
    print("01_latency_bandwidth.py —— 「访存慢」不构成诊断：延迟受限还是带宽受限")
    print("=" * 74)
    print("  ★ 真值标注：%s" % specs.MEASURE_NOTE)
    print("  ★ 测量条件：%s / python %s / numpy %s / %s / clang %s"
          % (specs.MACHINE["cpu"], sys.version.split()[0], specs.MACHINE["numpy"],
             sys.platform, microbench.clang_version()))
    clang = microbench.require_clang()
    doc_dir = specs.find_doc_dir(verbose=True)
    if os.environ.get("PERF_QUICK") == "1":
        # ★ **快档**（`make quick` / `make falsify` / `make mutate` 用）：
        #   只**缩短重复次数与规模档数**，断言与代码路径完全不变。
        #   ⚠ 默认档才是正文引用的口径。
        print("  ★ 档位：**快档**（PERF_QUICK=1，诊断工具用；断言与代码路径不变，"
              "只把重复次数与规模档数调小）")
        microbench.DEP_REPS = 4
        microbench.UNROLL8_REPS = 3
        microbench.UNROLL8_SIZES = (64 * 1024, 1024 * 1024, 8 * microbench.MB,
                                    64 * microbench.MB)

    tmpdir = microbench.make_tmpdir("perf01b_")
    try:
        bpaths = microbench.build_all(tmpdir, clang=clang)
        curve = part_A(bpaths)
        part_B(bpaths, curve)
        part_C(bpaths)
        part_D(doc_dir)
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
    return summary()


if __name__ == "__main__":
    sys.exit(main())

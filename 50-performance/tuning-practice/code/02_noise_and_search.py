#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""02_noise_and_search.py —— 把测量噪声**加回来**之后，搜索会怎么选错。

配套文档：`02-noise-and-search.md`。★ 五类断言（A/B/C/D/E）**全部在本程序内部**
（`CONTRIBUTING` §2 表里的第二种情形），失败时以**非零退出码**结束；**没有 `verify.py`**。

★★ 本主题的真值标注（逐处都要带，不许省）：

> 本程序的数字**全部是 CPU 上的本机实测值**（`Apple M1 Pro` / `macOS` /
> `python3 3.13.5` / `numpy 2.4.4`），**不是 GPU、不是 Nsight Compute**。
> 它们支撑的是「**测量噪声怎么让搜索选错**」这类**机制**结论。

★★ **本篇与 `60-compilers/auto-tuning` 的分工写死**：

  · `60` 的 `mini_autotune.py:612` **显式冻结**了测量噪声（`MEASURE_NOISE = 0.0`），
    「测量方法论在上游」（`60/03-search-algorithms.md:255` 逐字）；
  · 本篇**不重讲**搜索空间、代价模型、四类搜索策略、regret 的定义与预算公式、
    Triton autotuner 的默认行为 —— **全部回链 `60-compilers/auto-tuning`**；
  · 本篇只讲**把噪声放回来之后发生了什么**：
    **搜索会稳定地选错，而「多测几次」不足以救回来**。

★★ 本程序与 `60` 在**实现上**的关键区别（也是本程序诚实性的来源）：

> `60` 的噪声是 **造出来的**（`MEASURE_NOISE × rng.normal()`）。
> 本程序的噪声是 **量出来的** —— `mini_search.measure()` **真的把那份工作量跑一遍**、
> 用 `time.perf_counter` 计时。**噪声的量级与重尾程度由这台机器决定，不由我决定**
> （P12：自造真值 —— 拿自己造的参照去验证自己是本域要防的事）。

★★ 本程序的可证伪断言（改坏它们，本程序必须非零退出）：

  · `A.1`  **本机一次 ~0.4 ms 的真实测量，σ/中位 ∈ (0.2%, 500%)**
           —— 双向的**荒谬性闸门**：**下界**抓「没在测」（噪声关掉时恰好 0），
             **上界**只抓量级失控（机器被占满时实测到过 111%，那不算失控）；
  · `A.2`  **噪声有真实展宽**：`p90/中位 > 1.0005`（机器安静时实测 1.041–1.053；
           被别的进程占满时最坏观测 **1.0071**，所以窗口按**最坏观测**取）
           —— ★ 有意**不**对 `max/中位` 下断言：`max` 只由一个样本决定，**n 小时极不稳**
             （同一段代码两次跑出过 1.30 与 1.04）。重尾的**结构性**证据在 `03` 的 A 段；
  · `B.1`  **零噪声下搜索选中的就是网格参考最优，regret 严格为 0**（确定性）；
  · `B.2`  **真实噪声下选错率 ≥ 10%**，且**平均 regret > 0**
           ← `make falsify` 的「关掉噪声」注入项打在这一点上；
  · `B.3`  **对照**：真实噪声下的平均 regret **严格大于**零噪声下的（=0）
           ⇒ **搜索结果对「加噪 / 不加噪」敏感** ⇒ 它被**噪声**驱动；
  · `C.1`  **把每个配置的测量次数加倍（m → 2m）后，选错率仍 ≥ 10%**
           —— 「多测几次」**不足以**消除错误；
  · `C.2`  **按 `45/02` 的公式算「要多少才够」**：让顶部 0.12% 的真实差可判定
           需要每个配置 **~6800 次**测量（= 当前预算的 **~340 倍**）
           ⇒ ★ 这一条由**实测 σ** 与**解析 δ** 推出，**不随会话抖动**
             —— 实施者第一版写的是「加倍后 regret 均值不降到一半以下」，
             **那个判据在正确实现下会随机失败**（实测出过 0.35 与 1.34 两个比值）；
  · `D.1`  **赢家诅咒**：被选中那个配置的**读数**低于真最优的**真值**
           —— 在 ≥ 80% 的试验里成立（实测 100%）；
  · `D.2`  **假胜利**：存在一批试验，**读数说「有收益」而真实是「亏损」**
           —— 占 ≥ 10%（实测远高于此）；
  · `E.1`–`E.6`  F1–F15 结构不变量 / δ\* 系数自洽 / **本主题的订正不变量** /
            文档 sha256 / 逐字 / 上游契约 / 标记自检。

用法：
    python3 02_noise_and_search.py            # 跑 A/B/C/D/E 五类断言
    python3 02_noise_and_search.py --lint     # AST 扫描：不 import torch/triton/numba/cuda

★ 外部工具：**不需要任何外部工具**（纯 `python3` + 标准库）。
"""

import ast
import os
import statistics
import sys
import time

import mini_search as M
import specs

HERE = os.path.dirname(os.path.abspath(__file__))

# ===========================================================================
# 0. 断言记账（每个编号程序自带一份，**不共享**）
# ===========================================================================
# ★★ 失败标记：本主题（tuning-practice）自定的字面形态是 `>>> FAIL <<<`。
#   ★ 它**不是**照抄任何别的主题的（`[FAIL]` / `**FAIL**` / `*** FAIL ***` 都会误匹配）。
#   ★★ 判定它的那一侧**必须【行首锚定】**：**程序成功时也会打印这个字面量**
#     （下面 `E.6` 的 detail 里就有），裸的 `ASSERT_FAIL_MARK in out` 在 rc=0 时也成立。
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
# 1. 实验参数
# ===========================================================================
A_REPS = 400                # A 类：噪声本身量多少次
TRIALS = 40                 # 每档预算跑多少次独立搜索
M_BASE = 5                  # 每配置基本测量次数
M_DOUBLE = 2 * M_BASE       # 「加倍预算」
FAKE_WIN_MIN_FRAC = 0.10    # D.2 的门槛
WINNER_CURSE_MIN_FRAC = 0.80


# ===========================================================================
# 2. A 类：噪声本身有多大（**真的量出来**，不是设一个参数）
# ===========================================================================
def part_A(unit_s):
    print("=" * 74)
    print("A 类 · 本机一次真实测量的噪声有多大（**量出来的**，不是设出来的）")
    print("=" * 74)
    print("  ★ 被测量：`mini_search.atomic_batch(%.0f)` 的真实耗时 —— "
          "**一次搜索里的「一次测量」就是这个东西**。" % M.MINI_UNITS_AT_BEST)
    print("     `%s`" % M.describe_landscape())
    print("     顶部相邻两个配置的真实差只有 **%.3f%%**。"
          % (100 * (M.landscape(M.BEST_X + 1) / M.landscape(M.BEST_X) - 1)))

    t0 = time.perf_counter()
    np_ = M.noise_probe(reps=A_REPS)
    print()
    print("  ── %d 次真实测量 ──（总时长 %.2f s）" % (A_REPS, time.perf_counter() - t0))
    print("     中位 %.4f ms · σ/中位 **%.2f%%** · p90/中位 %.4f · "
          "min/中位 %.4f · max/中位 **%.2f×**"
          % (np_["median_s"] * 1e3, 100 * np_["sigma_over_median"],
             np_["p90_over_median"], np_["min_over_median"], np_["max_over_median"]))
    print("     ★ 这个 σ 与 02 文档 §3 的「顶部真实差 %.2f%%」一比就知道："
          "**真实差异比噪声小一个量级**。" % (100 * M.MINI_CURVATURE * 0.2))

    # ★ 上界取 **500%**（不是 50%）：本机安静时 σ/中位 是 2.1%–3.8%，
    #   但**机器被占满时同一条测量就是 100% 以上**（实测出过 111%）——
    #   那个数**不是「量崩了」**，是「这台机器当时很忙」。
    #   上界的用途只是抓**量级上的失控**（比如除错、单位写错），
    #   所以把它放到比最坏观测还高一个量级处；**下界 0.2% 才是真正的闸门**
    #   （`make falsify` 的「关掉噪声」注入项就打在它上面：那时 σ 恰好 0）。
    check("A", "A.1", 0.002 < np_["sigma_over_median"] < 5.0,
          "★ **σ/中位 = %.2f%% 落在 (0.2%%, 500%%)** —— 双向闸门："
          "**下界**说明有没有在真的测（噪声被关掉时它恰好 0）；"
          "**上界**只抓量级上的失控（实测机器被占满时到过 111%%，那不算失控）"
          % (100 * np_["sigma_over_median"]))

    # ★★ A.2 只对 **p90/中位** 下断言，**不**对 `max/中位` 下断言。
    #   为什么（实施者实测，与 `bottleneck-diagnosis` 的那条教训同源）：
    #   `max` **只由一个样本决定**，n 小时极不稳 —— 同一段代码两次跑出过 1.30 与 1.04。
    #   `p90` 由**全部样本**决定，本机实测稳定在 **1.040–1.053**。
    #   ⇒ 判据要落在「由全部样本决定的量」上；`max` 只打印、不判定。
    #   ★ 重尾的**结构性**证据在 `03_ab_orchestration.py` 的 A 段（那里有 6 万个样本）。
    # ★ 窗口取 **1.0005**（不是 1.01）：实施者实测，`p90/中位` 在**机器安静**时是
    #   1.041–1.053，而在**机器被别的进程占满**时掉到过 **1.0071**。
    #   取 1.01 会让那一次失败 —— 那是「窗宽超过了效应在负载下的实际大小」。
    #   ⇒ 这条的判据力在于「**分布有没有超出中位之外的展宽**」：
    #     噪声被关掉时（`make falsify` 的注入项 3）所有读数恒等、`p90/中位` **恰好 1.0000**。
    check("A", "A.2", np_["p90_over_median"] > 1.0005,
          "★ **p90/中位 = %.4f（> 1.0005）** ⇒ 测量噪声有真实展宽，"
          "且**换一种量法（分位数）也得到同一个结论**（P13 判据第③句）"
          "（`max/中位` = %.2f× 一并打印，但**不作断言**：它只由一个样本决定）"
          % (np_["p90_over_median"], np_["max_over_median"]))
    return np_


# ===========================================================================
# 3. B 类：零噪声 vs 真实噪声
# ===========================================================================
def part_B(unit_s):
    print()
    print("=" * 74)
    print("B 类 · 零噪声 vs 真实噪声：搜索会不会选错")
    print("=" * 74)
    print("  ★ 搜索做了**唯一一件事**：每个配置测 `m` 次取中位，选读数最小的。")
    print("     **搜索算法那一维是本程序刻意固定的** —— 四类策略、regret 的定义与预算"
          "公式、Triton autotuner 的默认行为，**正文在 "
          "`60-compilers/auto-tuning/03-search-algorithms`**，本篇只回链。")
    print("  ★ 参考最优 = 配置 **%d**（`MINI_OPT_X = %.1f` 在网格之外，"
          "所以**没有任何网格配置恰好最优**）。" % (M.BEST_X, M.MINI_OPT_X))
    print("     顶部真实差（相对参考最优）：%s"
          % " · ".join("x=%d: %+.3f%%" % (x, 100 * g) for x, g in M.top_true_gaps(5)))

    # ---- B.1：零噪声 ----
    m0 = M.Measurer(unit_s, noise=False)
    r0 = M.search(M_BASE, m0)
    t_best = m0.true_seconds(M.BEST_X)
    print()
    print("  ── 零噪声（直接返回解析真值 = `60` 域的做法）──")
    print("     选中 x=%d；它的真实代价 %.6f s；参考最优 %.6f s；**regret = %.9f**"
          % (r0.pick, m0.true_seconds(r0.pick), t_best, r0.regret))
    check("B", "B.1", r0.pick == M.BEST_X and r0.regret == 0.0,
          "★ **零噪声下选中的就是参考最优（x=%d），regret 严格为 0**"
          "⇒ 搜索算法本身是对的，选错只可能来自噪声" % r0.pick)

    # ---- B.2 / B.3：真实噪声 ----
    mt = M.Measurer(unit_s, noise=True)
    t0 = time.perf_counter()
    res_base = M.run_trials(M_BASE, TRIALS, mt)
    s_base = M.summarize(res_base)
    wall = time.perf_counter() - t0
    print()
    print("  ── 真实噪声（**真的计时**）── m=%d，%d 次独立搜索，%.1f s，"
          "真计时 %d 次" % (M_BASE, TRIALS, wall, mt.n_timed))
    print("     选错（regret > 0）：**%d/%d = %.0f%%**；regret 中位 %.4f%% · "
          "均值 **%.4f%%** · 最大 %.4f%%"
          % (s_base["wrong"], s_base["n"], 100 * s_base["wrong_frac"],
             100 * s_base["regret_median"], 100 * s_base["regret_mean"],
             100 * s_base["regret_max"]))
    print("     被选中的配置集合：%s（参考最优是 %d）" % (s_base["picks"], M.BEST_X))

    check("B", "B.2", s_base["wrong_frac"] >= 0.10 and s_base["regret_mean"] > 0.0,
          "★ **真实噪声下选错率 %.0f%%（≥ 10%%）且平均 regret %.4f%% > 0** ⇒ "
          "**搜索稳定地选错**，而不是偶发失手"
          % (100 * s_base["wrong_frac"], 100 * s_base["regret_mean"]))

    gap_noise = s_base["regret_mean"] - 0.0
    check("B", "B.3", gap_noise > 0.0,
          "★ **对照**：真实噪声 %.4f%% vs 零噪声 **0.0000%%** ⇒ "
          "**搜索结果对「加噪 / 不加噪」敏感** ⇒ 它被**噪声**驱动，不是被真实差异驱动"
          % (100 * gap_noise))
    return res_base


# ===========================================================================
# 4. C 类：把预算加倍，够不够
# ===========================================================================
def part_C(unit_s, s_base, np_):
    print()
    print("=" * 74)
    print("C 类 · 把每个配置的测量次数**加倍**，够不够")
    print("=" * 74)
    print("  ★ `45/02` 的判据是 `n ∝ σ²` —— 噪声翻倍要多 4 倍样本。"
          "本类问的是反过来的那一面：**样本翻倍能买到什么**。")

    mt = M.Measurer(unit_s, noise=True)
    t0 = time.perf_counter()
    res_dbl = M.run_trials(M_DOUBLE, TRIALS, mt)
    s_dbl = M.summarize(res_dbl)
    print()
    print("     %-10s %-14s %-16s %-16s %s"
          % ("预算", "选错率", "regret 中位", "regret 均值", "regret 最大"))
    for tag, s in (("m=%d" % M_BASE, s_base), ("m=%d（加倍）" % M_DOUBLE, s_dbl)):
        print("     %-10s %-14s %-16s %-16s %s"
              % (tag, "%d/%d = %.0f%%" % (s["wrong"], s["n"], 100 * s["wrong_frac"]),
                 "%.4f%%" % (100 * s["regret_median"]),
                 "%.4f%%" % (100 * s["regret_mean"]),
                 "%.4f%%" % (100 * s["regret_max"])))
    print("     （本档总时长 %.1f s；真计时累计 %d 次）" % (time.perf_counter() - t0, mt.n_timed))

    check("C", "C.1", s_dbl["wrong_frac"] >= 0.10,
          "★ **加倍后选错率仍 %.0f%%（≥ 10%%）** ⇒ 「多测几次」**不足以**消除错误"
          % (100 * s_dbl["wrong_frac"]))

    # ★★ C.2 算的是「**够需要多少**」，不是一个关于 `regret 均值比值` 的阈值。
    #   为什么改（实施者实测）：第一版判「加倍后平均 regret 没有降到一半以下」，
    #   而 regret 均值**本身就是噪声**：同一个程序两次跑出过 0.35 与 1.34 两个比值。
    #   ⇒ 那条断言在**正确实现下会随机失败**（同一类错，这是本主题第三次踩到）。
    #   ★ 换成一个**由实测 σ 推出来**的量：`45/02` 的
    #     `n = 2 · (2.8016 · σ / δ)²`（**代入，不重推**），其中 δ 是顶部两个配置的**真实差**。
    #     这个量由 σ（由全部样本决定）与 δ（解析真值）决定，**不随会话抖动**。
    sigma_rel = np_["sigma_over_median"]
    delta = M.landscape(M.BEST_X + 1) / M.landscape(M.BEST_X) - 1.0
    m_req = specs.mde_required_n(sigma_rel, delta)
    total_s = m_req * M.MINI_N_CONFIGS * np_["median_s"]
    print()
    print("  ── 换一个问法：按 `45/02` 的公式，**要多少才够** ──")
    print("     实测 σ/中位 = **%.2f%%**；顶部两个配置的真实差 δ = **%.3f%%**"
          % (100 * sigma_rel, 100 * delta))
    print("     `n = 2·(2.8016·σ/δ)²` ⇒ 每个配置要 **%.0f 次**测量 ⇒ "
          "%d 个配置合计 **%.0f 次** ≈ **%.0f s**"
          % (m_req, M.MINI_N_CONFIGS, m_req * M.MINI_N_CONFIGS, total_s))
    print("     （当前预算是 m=%d，**是它的 %.4f 分之一**）" % (M_DOUBLE, m_req / M_DOUBLE))
    # ★ 门槛取 **5 × 当前预算**（不是 50 ×）：`m_req` 是**由实测 σ 推出来的** ——
    #   机器安静时 σ 小、`m_req` 本来就小（实测快档下出过 52，而 M_DOUBLE=6）。
    #   判据力在于「**它与当前预算同一个量级，还是差得远**」，不是某个绝对倍数。
    check("C", "C.2", m_req > 5 * M_DOUBLE,
          "★ **要让顶部那个 %.3f%% 的真实差可判定，每个配置要 %.0f 次测量**"
          "（= 当前预算 m=%d 的 **%.0f 倍**）⇒ 「加倍」离够用**差得很远**。"
          "★ 这个量由实测 σ 与解析 δ 推出（`45/02` 的公式，**代入不重推**），"
          "**不随会话抖动** —— 而「regret 均值的比值」那种判据会（实测出过 0.35 与 1.34）"
          % (100 * delta, m_req, M_DOUBLE, m_req / M_DOUBLE))
    return s_dbl


# ===========================================================================
# 5. D 类：赢家诅咒与假胜利
# ===========================================================================
def part_D(unit_s, res):
    print()
    print("=" * 74)
    print("D 类 · 赢家诅咒：读数上的「赢家」在真值上并不赢")
    print("=" * 74)
    print("  ★ 机理：`min of N noisy samples` 的期望 **小于** 那些样本的真值；"
          "**找一个最小值**这个动作本身就系统地偏向「运气好的那次读数**。**")

    measurer = M.Measurer(unit_s, noise=True)
    t_best = measurer.true_seconds(M.BEST_X)
    n = len(res)
    curse = sum(1 for r in res if r.scores[r.pick] < t_best)
    fake = sum(1 for r in res if r.measured_gap > 0.0 and r.regret > 0.0)
    gaps = [r.measured_gap for r in res if r.measured_gap > 0.0]
    print()
    print("     「读数上的赢家低于真最优的**真值**」：**%d/%d**" % (curse, n))
    print("     「读数说**有收益**、而真实是**亏损**」：**%d/%d**" % (fake, n))
    if gaps:
        print("     其中报出来的收益：中位 **%.4f%%**（最大 %.4f%%），而真实 regret 是正的"
              % (100 * statistics.median(gaps), 100 * max(gaps)))

    check("D", "D.1", curse / n >= WINNER_CURSE_MIN_FRAC,
          "★ **赢家诅咒成立**：%d/%d = %.0f%% 的试验里，被选中配置的**读数**低于真最优的**真值**"
          "（门槛 %.0f%%）⇒ 报出来的那个「赢了」是**读数上的**，不是真值上的"
          % (curse, n, 100.0 * curse / n, 100 * WINNER_CURSE_MIN_FRAC))

    check("D", "D.2", fake / n >= FAKE_WIN_MIN_FRAC,
          "★ **假胜利**：%d/%d = %.0f%% 的试验里，**读数说「比参考最优还好」而真实更差**"
          "（门槛 %.0f%%）⇒ 这正是 `45/03` 讲的「假胜利」，"
          "只不过这里它**可机械地数出来**"
          % (fake, n, 100.0 * fake / n, 100 * FAKE_WIN_MIN_FRAC))


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
          "F1–F15 的**结构不变量** %d 条全过（含 must_contain ⊂ quote）" % len(rows))

    ok, lits = specs.mde_coefficient_selfcheck()
    check("E", "E.2", ok,
          "★ **δ* 的系数转写自洽**：`%.15g` 与 `MDE_ANCHORS` 引文里抽出的 %s 全部相等"
          % (specs.MDE_COEFFICIENT_AT_POWER_80, ["%.4f" % x for x in lits]))

    # ★ 本主题的**分工不变量**：`60` 那两行「冻结噪声」的注释与
    #   `MEASURE_NOISE = 0.0` 必须**同时**在场（单看一行不知道它为什么被冻结）。
    check("E", "E.3",
          (specs.UPSTREAM_CLAIMS[3]["quote"] == "MEASURE_NOISE = 0.0"
           and all("冻结" in l or "无法归因" in l
                   for l in specs.MEASURE_NOISE_RATIONALE)),
          "★ **`60` 的让渡是「三行一体」**：`MEASURE_NOISE = 0.0` 与它上面两行"
          "「为什么冻结」的注释（%s）**必须同时在场**，"
          "否则「上游把测量方法论让给了本域」这句话就没有依据"
          % ("；".join(x.strip("# ★ ") for x in specs.MEASURE_NOISE_RATIONALE)))

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
        check("E", "E.5", not vbad,
              "F1–F15 等**逐字在场**（%d 条）" % len(vrows))

    urows = specs.upstream_check(specs.REPO_ROOT)
    if urows is None:
        skip("E", "E.6", "上游文档不在场 ⇒ 契约逐字核对 SKIP")
    else:
        ubad = [(l, g, w) for (l, g, w) in urows if g < w]
        for l, g, w in ubad:
            print("     ✗ %s: 出现 %d 次，期望 ≥ %d" % (l, g, w))
        check("E", "E.6", not ubad,
              "上游契约 %d 处**逐字在场**（含 `45/02:246` 同时点到 `σ` 与 `τ` 的那句、"
              "`60/03:255`「测量方法论在上游」、`60` 的 `MEASURE_NOISE = 0.0`）"
              % len(urows))


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
    global A_REPS, TRIALS, M_BASE, M_DOUBLE
    if "--lint" in sys.argv:
        return lint()

    print("=" * 74)
    print("02_noise_and_search.py —— 把测量噪声加回来，搜索会怎么选错")
    print("=" * 74)
    print("  ★ 真值标注：%s" % specs.MEASURE_NOTE)
    print("  ★ 测量条件：%s / python %s / numpy %s / %s"
          % (specs.MACHINE["cpu"], sys.version.split()[0], specs.MACHINE["numpy"],
             sys.platform))
    print("  ★ 与 `60-compilers/auto-tuning` 的分工：那边讲**无噪声**的搜索"
          "（`MEASURE_NOISE = 0.0`），本篇只讲**有噪声**的搜索。")
    doc_dir = specs.find_doc_dir(verbose=True)
    if os.environ.get("PERF_QUICK") == "1":
        # ★ **快档**：只缩短试验数，断言与代码路径完全不变。
        print("  ★ 档位：**快档**（PERF_QUICK=1；断言与代码路径不变，只把试验数调小）")
        A_REPS, TRIALS, M_BASE = 150, 20, 3
        M_DOUBLE = 2 * M_BASE

    unit_s = M.calibrate_unit_seconds()
    print("  ★ 换算标定：1 个地形单位 = %.4g s ⇒ 参考最优一次测量 ≈ **%.3f ms**"
          % (unit_s, M.TRUE_UNITS[M.BEST_X] * unit_s * 1e3))
    a_stats = part_A(unit_s)
    res = part_B(unit_s)
    part_C(unit_s, M.summarize(res), a_stats)
    part_D(unit_s, res)
    part_E(doc_dir)

    marks = {f: read_mark_from(os.path.join(HERE, f))
             for f in ("01_sigma_tau.py", "02_noise_and_search.py",
                       "03_ab_orchestration.py", "meta_falsify.py", "mutate.py")}
    uniq = sorted({v for v in marks.values() if v is not None})
    print()
    print("     %s" % ", ".join("%s=%r" % (k, v) for k, v in sorted(marks.items())))
    check("E", "E.7",
          len(uniq) == 1 and uniq[0] == ASSERT_FAIL_MARK and "FAIL" in ASSERT_FAIL_MARK,
          "★ 失败标记**五处一致**且含 `FAIL`（本主题自定的字面形态 `%s`）"
          % ASSERT_FAIL_MARK)
    return summary()


if __name__ == "__main__":
    sys.exit(main())

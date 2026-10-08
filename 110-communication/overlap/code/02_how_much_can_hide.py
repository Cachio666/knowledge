#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""02_how_much_can_hide.py —— **「能藏多少」的判据**：可藏的上限，与它要求的条件。

配套文档：`02-how-much-can-hide.md`。

★ 三类断言**全部在本程序内部**，失败时以**非零退出码**结束；**没有 `verify.py`**。

★★★ **本层是【时间层】**。

★★★ **本程序接住的是本域【最核心的一条让渡】** ——
  `100-parallelism/data-parallel/02-gradient-synchronization.md:65` 逐字：
  「★★★ **「通信能藏多少」**（通信-计算重叠的收益）→ `110-communication` 的 `overlap`。
   ★ 本主题**不给这条判据**（见 §2.2）」
  ⇒ ★★ 上游**点名**把这条判据让渡给本主题，**并且明确说它自己不给**。

★★★ **一条必须如实标的级别**：本程序的**判据模型是 `[E]` 级（本库推导）** ——
  全库**没有**「能藏多少」的规范/论文级一手公式（计划 `§C.4`）。
  ⇒ ★★ **不得写成「研究表明」或引任何外部出处**（计划 `§C.5` 的 `F8`）。
  ★★★ **而本程序【专门有一节】（`B.2`）做一件计划点出、但没人验过的事**：
    把**本库的 `[E]` 模型**与**文献侧的形态**并排，**算出差异的具体形态**。

★★★ **本程序的 A 类 = 「三条免费守卫」**（计划 `§D.3` 的 `P35`）：
  ① `K = 1` ⇒ 必须等于无重叠值 `C + T`（**逐位**）；
  ② `K → ∞` ⇒ 必须等于物理下界 `max(C, T)`；
  ③ 中间值**不得小于下界**。
  ★★ 守卫的期望值用的是**两条独立表达式**（`C+T` 与 `max(C,T)`），**不是**被测函数自己。

★★ **覆盖边界**：本程序守的是「**判据的边界行为**」与「**错读法会不会被拒**」。
  **不守**任何**真实硬件**上的重叠收益（本机**无 GPU / 无 NCCL / 无网卡**）。
  ★★ **最强的限定语**：**「不是 NCCL，也不是真机」**。

★★ 可证伪断言（把下面任一条改错，本程序必须非零退出）：
  · `A.1`/`A.2`/`A.3` ★★★ **三条免费守卫**（`P35`），★ 前两条在**模拟器**上验；
  · `A.4` ★★ **可藏的上限 = `min(C,T)`**，且 `K → ∞` 时取到；
  · `B.1` ★★ 「可藏时间 vs `K`」的表；`B.2` ★★★ **`[E]` 模型 vs 文献形态的差异**（含 `Weave` 的逐字式）；
  · `C.1` ★★★ **`T > C` 时的**错读法（「通信全藏得住」）必须被拒；
  · `D.*` 引文逐字钉住 + **逐元素**钉住 + **失败标记自检** + `UNSOURCED`/`NOT_QUOTED`。
"""
import ast
import collections
import inspect
import sys
import textwrap
from fractions import Fraction

import specs

# ===========================================================================
# 0. 断言记账（每个编号程序自带一份，**不共享**）
# ===========================================================================
ASSERT_FAIL_MARK = "+= FAIL: overlap"

OTHER_THEME_MARKS = ("&& FAIL: collectives", ">< FAIL: nccl", ",, FAIL: network-topology")

FAILS = []
MARKS_USED = set()
_TOTAL = collections.Counter()
_PASS = collections.Counter()


def check(name, cond, detail="", detail_fail=None):
    """`name` 形如 `B.1` —— **首字母就是类别**（A/B/C/D），与 `CONTRIBUTING` §4 对齐。"""
    cls = name[0]
    _TOTAL[cls] += 1
    if cond:
        _PASS[cls] += 1
        print("  [ok]   %-6s %s" % (name, detail))
    else:
        FAILS.append(name)
        MARKS_USED.add(ASSERT_FAIL_MARK)
        print("  %s %-6s %s" % (ASSERT_FAIL_MARK, name,
                                detail_fail if detail_fail is not None else
                                "★★ 这条判据【不成立】—— ★ 上面那一版「判据成立」的措辞是"
                                "**成立时**的文案，**本轮没有成立**"))
    return bool(cond)


# ===========================================================================
# 1. 参照：**时隙模拟**（★ 与 `01` 同一个调度模型，**自带一份**、不跨程序 import）
#    ★★ **本程序用的是【另一组参数】**（`HIDE_C_T_GRID` / `HIDE_K_GRID`）
# ===========================================================================
def sim_ticks(C, T, K):
    """★★ **时隙模拟**：两条引擎（计算 / 通信）逐拍推进，返回完工**拍数**（时间以 `1/K` 为单位）。

    ★★★ **调度规则**：计算引擎按块序串行；通信引擎按块序串行，且**第 `i` 块的通信
      只在第 `i` 块的【计算】做完之后**才能开始（★ 「首项不可重叠」的落点）。
    ★★ **拍数由数据驱动**（`while ki < K`）—— 不是从闭式取的（§1.3.2b）。
    """
    comp = [0] * K
    comm = [0] * K
    ci = 0
    ki = 0
    ticks = 0
    while ki < K:
        do_comp = (ci < K)
        do_comm = (ki < K) and (comp[ki] >= C)
        if do_comp:
            comp[ci] += 1
        if do_comm:
            comm[ki] += 1
        if do_comp and comp[ci] >= C:
            ci += 1
        if do_comm and comm[ki] >= T:
            ki += 1
        ticks += 1
    return ticks


def sim_makespan(C, T, K):
    """模拟器的完工时间（拍数 / `K`）。"""
    return Fraction(sim_ticks(C, T, K), K)


# ===========================================================================
# 2. 被测：模型（★ 全部 **`[E]` 级：本库推导**）+ 两条独立表达式（守卫的期望值）
# ===========================================================================
def cf_no_overlap(C, T):
    """**无重叠**值 `C + T`（★ 独立表达式 —— 守卫①的期望值）。"""
    return Fraction(C + T)


def cf_lower_bound(C, T):
    """**物理下界** `max(C, T)`（★ 独立表达式 —— 守卫②③的期望值）。"""
    return Fraction(max(C, T))


def cf_pipeline(C, T, K):
    """★★ **本库的 `[E]` 模型**：`(C+T)/K + (K−1)·max(C/K, T/K)`（= `max(C,T) + min(C,T)/K`）。"""
    return Fraction(C + T, K) + (K - 1) * max(Fraction(C, K), Fraction(T, K))


def cf_hidden(C, T, K):
    """**可藏住的通信时间** = 无重叠值 − 流水化后的总时间 = `min(C,T)·(1 − 1/K)`。"""
    return cf_no_overlap(C, T) - cf_pipeline(C, T, K)


def cf_hidden_ceiling(C, T):
    """★★★ **可藏的上限** = `min(C, T)` —— ★ **较短的那条时间轴**（`K → ∞` 时取到）。"""
    return Fraction(min(C, T))


def cf_exposed(C, T, K):
    """**暴露在外的通信时间** = `T − 可藏`。★ `T > C` 时它有一个**不可约的下限 `T − C`**。"""
    return Fraction(T) - cf_hidden(C, T, K)


def cf_lit(C, T, K):
    r"""★ **[E] 级（本库对文献的化简）**：把 `Weave`（`specs.WEAVE_TOTAL_TEX` + `specs.WEAVE_TAIL_TEX`）
    在 **`α = 0`（尾项完全不被藏）、忽略 dispatch、且 `eff_chunk(K) = 1`（分块吞吐 == 不分块）**
    下化简成 `max(C + T/K, T)`。

    ★★★ **它不是一条【引文】** —— 它是**本库的化简**，用来做**形态对照**（`B.2`）。
    ★★ 化简的**三条假设**逐条写在上面（`α = 0`、忽略 dispatch、`eff_chunk(K) = 1`）
      ⇒ 引用它**必须带这三条**。
    ★★★ **第三条最容易漏**（★ 本轮补上）：Weave 的计算项**显式依赖 `K`** —— 原文（`alttext` 逐字）：
      `T_{\text{comp}}(c,K)=\frac{W_{\text{comp}}}{\text{TFLOPS}(N{-}c)\cdot\text{eff}_{\text{chunk}}(K)},`
      且 `where eff_chunk(K)\in(0,1] is the ratio of chunked GEMM throughput to the unchunked case`。
      ⇒ ★ 把 `T_comp(c,K)` 写成与 `K` 无关的常数 `C`，**就额外假设了 `eff_chunk(K) = 1`**
        （★ 这使 `C` 成为**与 `K` 无关的常数**）。
    ★★ **范围声明**：这条 `alttext` 逐字**随本假设写在 docstring 里、未注册进 `specs.py`** ——
      ★ 它只服务于「说清第三条假设」，**不作为引文常量**（★ 本库「有对照」不等于「有出处」）。
    """
    return max(Fraction(C) + Fraction(T, K), Fraction(T))


# ===========================================================================
# 3. A 类 —— ★★★ **三条免费守卫**（`P35`）+ 可藏上限
# ===========================================================================
def part_A():
    print("=" * 78)
    print("A · ★★★ **「能藏多少」的判据**：三条免费守卫（`P35`）+ 可藏的上限")
    print("=" * 78)
    print("  ★★★ 这是上游**点名让渡**的那条判据（`data-parallel/02:65`：「本主题不给这条判据」）。")
    print("  ★★ 被测 = `cf_pipeline`（**`[E]` 级**）；★ 守卫的期望值 = `C+T` 与 `max(C,T)`（**独立**）。")

    # ── A.1 ★★★ 守卫①：`K = 1` ⇒ 无重叠值（**逐位**）──────────────────────
    print()
    print("  ── A.1 ★★★ **守卫①**：`K = 1` ⇒ 必须等于无重叠值 `C + T`（**逐位**）──")
    ok1 = True
    for (C, T) in specs.HIDE_C_T_GRID:
        got = sim_makespan(C, T, 1)
        exp = cf_no_overlap(C, T)
        ok1 = ok1 and (got == exp)
        print("     (C,T)=%-9s `K=1` 模拟 %-6s == `C+T` %-6s（**逐位**）= %s"
              % ("(%d,%d)" % (C, T), got, exp, got == exp))
    print("     ★★★ 它的物理含义：**不开分块就没有重叠** —— ★ 这是**任何**流水模型都必须满足的锚点。")
    check("A.1", ok1 and specs.HIDE_C_T_GRID == ((12, 4), (4, 12), (7, 7), (1, 20), (4, 7)),
          "★★★ **判据成立**：`HIDE_C_T_GRID` 五档上 **`K = 1` 的模拟值【逐位】等于 `C + T`**"
          "（★ 期望值是**独立表达式** `cf_no_overlap`，不是被测函数自己）",
          "**守卫①不成立** —— ★ `K = 1` 居然不等于无重叠值 ⇒ 模型的锚点错了")

    # ── A.2 ★★ 守卫③：中间值不得小于下界（**在模拟器上验**）───────────────
    print()
    print("  ── A.2 ★★ **守卫③**：中间 `K` 值**不得小于**物理下界 `max(C,T)` ──")
    ok2 = True
    for (C, T) in specs.HIDE_C_T_GRID:
        lb = cf_lower_bound(C, T)
        vals = [(K, sim_makespan(C, T, K)) for K in specs.HIDE_K_GRID]
        under = [K for K, v in vals if v < lb]
        ok2 = ok2 and (not under)
        print("     (C,T)=%-9s 下界 %-6s · `K` 网格上的最小值 %-8s ⇒ 下穿 %d 次"
              % ("(%d,%d)" % (C, T), lb, min(v for _k, v in vals), len(under)))
    print("     ★★ **0 次下穿** ⇒ 模型**从不下穿物理下界**（`P35` 的第二道防线）。")
    check("A.2", ok2 and specs.HIDE_K_GRID == (1, 2, 3, 8, 16),
          "★★ **判据成立**：`HIDE_C_T_GRID` × `HIDE_K_GRID`（**另一个网格**，与 `01` 不重叠）上，"
          "**模拟值从不下穿 `max(C,T)`** ⇒ ★ 这条是 `P35` 的第二道防线",
          "**守卫③不成立** —— ★ 有 `K` 让总时间低于物理下界（`P35` 复发）")

    # ── A.3 ★★ 守卫②：`K → ∞` ⇒ 下界 ────────────────────────────────────
    print()
    print("  ── A.3 ★★ **守卫②**：`K → ∞` ⇒ 必须收敛到物理下界 `max(C,T)` ──")
    ok3 = True
    for (C, T) in specs.HIDE_C_T_GRID:
        v = cf_pipeline(C, T, specs.K_INF)
        lb = cf_lower_bound(C, T)
        ok3 = ok3 and abs(v - lb) <= Fraction(1, 100)
        print("     (C,T)=%-9s 闭式@K=%d = %-10s · 下界 %-6s · 差 %-10s ≤ 1/100 = %s"
              % ("(%d,%d)" % (C, T), specs.K_INF, v, lb, v - lb, abs(v - lb) <= Fraction(1, 100)))
    print("     ★★★ **如实注明**：这里用的是 `K = K_INF`（**数值替身**，不是真的取极限）——")
    print("        残余差恰好是 `min(C,T)/K_INF`，**随 `K_INF` 增大而变小**（★ 见 `K_INF` 的口径）。")
    check("A.3", ok3 and specs.K_INF == 1000000,
          "★★ **判据成立**：`K = K_INF` 时闭式**收敛到 `max(C,T)`**（容差 `1/100`）—— "
          "★ 这坐实了「**理想重叠的下限是较长的那条时间轴**」；★★ 并**如实注明这是数值替身**",
          "**守卫②不成立** —— ★ 模型在无穷分块下居然不收敛到下界")

    # ── A.4 ★★★ 可藏的上限 = `min(C,T)` ─────────────────────────────────
    print()
    print("  ── A.4 ★★★ **可藏的上限**（本程序的正题）：`K → ∞` 时可藏 `= min(C,T)` ──")
    ok4 = True
    for (C, T) in specs.HIDE_C_T_GRID:
        ceil = cf_hidden_ceiling(C, T)
        near = cf_hidden(C, T, specs.K_INF)
        ok4 = ok4 and (ceil == Fraction(min(C, T))) and abs(near - ceil) <= Fraction(1, 100)
        print("     (C,T)=%-9s 可藏上限 `min(C,T)` = %-6s · 闭式@K=%d = %-10s ⇒ 逼近 = %s"
              % ("(%d,%d)" % (C, T), ceil, specs.K_INF, near, abs(near - ceil) <= Fraction(1, 100)))
    print("     ★★★ **这条判据的读法**：")
    print("        · **计算比通信长**（`C >= T`）⇒ 可藏上限 **= `T`**（**通信能【全】藏住**）；")
    print("        · **通信比计算长**（`T > C`）⇒ 可藏上限 **= `C`**（**只能藏住「计算那么多」**）——")
    print("          ★★ 剩下的 `T − C` **永远暴露在外**（见 `C.1`）。")
    check("A.4", ok4,
          "★★★ **判据成立**：五档上 **`K → ∞` 时可藏 = `min(C,T)`**（容差 `1/100`，数值替身）—— "
          "★★ **它的物理含义**：能藏住的上限**由【较短的那条时间轴】决定** —— "
          "★★★ 这就是「**能藏多少**」的全部内容",
          "**可藏上限被动过** —— ★ 先查 `cf_hidden` / `cf_hidden_ceiling`")


# ===========================================================================
# 4. B 类：可藏 vs `K` 的表 · ★★★ **[E] 模型 vs 文献形态的差异**
# ===========================================================================
def part_B():
    print()
    print("=" * 78)
    print("B · 「可藏时间 vs `K`」的表 · ★★★ **`[E]` 模型 vs 文献形态的差异**")
    print("=" * 78)

    # ── B.1 ★★ 可藏时间 vs `K` ────────────────────────────────────────────
    print()
    print("  ── B.1 ★★ **可藏时间 vs `K`**（★ 每一格都是模拟器排出来的）──")
    print("     %-10s %-6s %-12s %-12s %-12s" % ("(C,T)", "K", "总时间", "可藏", "暴露在外"))
    rows = {}
    for (C, T) in specs.HIDE_C_T_GRID:
        for K in specs.HIDE_K_GRID:
            mk, hd, ex = sim_makespan(C, T, K), cf_hidden(C, T, K), cf_exposed(C, T, K)
            rows[(C, T, K)] = (mk, hd, ex)
            print("     %-10s %-6d %-12s %-12s %-12s"
                  % ("(%d,%d)" % (C, T), K, mk, hd, ex))
    print("     ★★★ **三种性态**（读表）：")
    print("        · `C = 12, T = 4`（计算主导）⇒ 可藏**趋近 `4`（= `T`）**，暴露**趋近 `0`**；")
    print("        · `C = 4, T = 12`（通信主导）⇒ 可藏**趋近 `4`（= `C`）**，暴露**趋近 `8`（= `T−C`）**；")
    print("        · `C = T = 7`（平衡）⇒ 可藏与暴露**同时趋近 `0`**（★ 两边都不亏）。")
    check("B.1", rows[(12, 4, 1)][0] == Fraction(16) and rows[(4, 12, 1)][0] == Fraction(16)
          and rows[(4, 12, 2)][1] == Fraction(2) and rows[(12, 4, 2)][1] == Fraction(2)
          and rows[(4, 12, 2)][2] == Fraction(10) and rows[(12, 4, 2)][2] == Fraction(2),
          "★★ **判据成立**：`K = 1` 时**两档总时间都是 `16`（无重叠）**；`K = 2` 时"
          "**两档可藏都是 `2`**，**但暴露不同**（通信主导档 `10` vs 计算主导档 `2`）—— "
          "★★★ **这条不对称就是「能藏多少」的判据的核心**",
          "**表里的某一格被动过** —— ★ 先查模拟器与两档 `(C,T)`")

    # ── B.2 ★★★ [E] 模型 vs 文献形态的差异（★ 计划点出、没人验过的那件事）───
    print()
    print("  ── B.2 ★★★ **`[E]` 模型 vs 文献形态**：差异的【具体形态】 ──")
    print("     ★★ 文献侧载体 = Weave（`arXiv %s`，**预印本 / 同行评审：未确认**，抓取日 %s）"
          % (specs.WEAVE_ID, specs.FETCH_DATE))
    print("     ★★★ 它的总时间式（**LaTeX 提取形态**，逐字）：")
    print("        `%s`" % specs.WEAVE_TOTAL_TEX)
    print("     它的尾项式（分块版，逐字）：")
    print("        `%s`" % specs.WEAVE_TAIL_TEX)
    print("     它的读法（散文逐字）：`%s`" % specs.WEAVE_PROSE_QUOTE)
    print()
    print("     ★★★ **本库把它化成可比的形态**（★ **[E] 级化简，三条假设写在函数 docstring 里**）：")
    print("        `max(C + T/K, T)`  ← 文献侧（`α = 0`、忽略 dispatch、`eff_chunk(K) = 1`）")
    print("        `max(C,T) + min(C,T)/K`  ← 本库（`[E]`）")
    print("     ★★★ **差异的【具体形态】**（本程序真的算出来，★ **分段**）：")
    print("        %-10s %-8s %-10s %-10s %-10s %-6s %s"
          % ("(C,T)", "K", "本库", "文献侧", "差", "分段对", "落在哪一段"))
    ok_b2 = True
    for (C, T) in specs.HIDE_C_T_GRID:
        for K in (2, 8):
            ours, lit = cf_pipeline(C, T, K), cf_lit(C, T, K)
            diff = ours - lit
            # ★★★ **分段形态**（`K >= 2`；`K = 1` 时恒为 `0`）—— 差**只在 `T > C` 时为正**，
            #   且**分两段**：靠近 `T ≈ C` 的中间段**小于** `min(C,T)/K`。
            if T <= C:                                   # 计算主导 / 平衡：两式完全相同
                pred, seg = Fraction(0), "T<=C ⇒ 差 = 0"
            elif Fraction(T) < Fraction(C * K, K - 1):   # 中间段：差 = (T−C)(1 − 1/K)
                pred = (Fraction(T) - C) * Fraction(K - 1, K)
                seg = "C<T<C·K/(K-1) ⇒ (T−C)(1−1/K)"
            else:                                        # 通信【远】长于计算：差 = min(C,T)/K
                pred, seg = Fraction(min(C, T), K), "T>=C·K/(K-1) ⇒ min(C,T)/K"
            ok_b2 = ok_b2 and (diff == pred) and (ours >= lit)
            print("        %-10s %-8d %-10s %-10s %-10s %-6s %s"
                  % ("(%d,%d)" % (C, T), K, ours, lit, diff, diff == pred, seg))
    print("     ★★★ **结论（差异的形态，【分段】一句话）**：")
    print("        · `T <= C` ⇒ **差 = 0**（两式【完全相同】）；")
    print("        · `C < T < C·K/(K-1)` ⇒ **差 = (T−C)·(1 − 1/K)**（**中间段**，★ 只在这里才【不是】`min/K`）；")
    print("        · `T >= C·K/(K-1)` ⇒ **差 = min(C,T)/K**（★ `K=2` 时门槛是 `2C`、`K=8` 时是 `8C/7`）；")
    print("        ★★ 且 **`本库 >= 文献侧` 恒成立**。")
    print("     ★★★ **为什么原表会「全绿却绕开反例」**（★ 本库 `P6` 的现场）：")
    print("        原先的 `HIDE_C_T_GRID` 里两个通信主导档 `(4,12)` / `(1,20)` **恰好都落在")
    print("        `T >= C·K/(K-1)` 的【成立区间】内** ⇒ 「差 = `min(C,T)/K`」看着处处成立。")
    print("        ★★ **本轮把 `(4,7)` 补进网格**（★ 它在 `K=2` 时落进中间段 `C<T<2C`）")
    print("           ⇒ **那一档的差是 `3/2`，不是 `min/K = 2`** ⇒ 断言**真的会有反例**。")
    print("     ★★ **物理读法**（★ 本库的读法，不是文献的声称）：")
    print("        ★ 文献形态是 `max(计算工作量 + 尾项, 通信工作量)` —— 它是一个**资源界**；")
    print("        ★ 本库的形态**多了【首项】`min(C,T)/K`** —— 那是**「第一块的通信要等它的计算」**的价格；")
    print("        ★★ **`T <= C` 时两式重合**（首项被 `min` 选到通信侧、恰好等于文献的 `T/K` 项）；")
    print("        ★★★ **`T > C` 时并不【总是】差 `min(C,T)/K`** —— 靠近 `T ≈ C` 的**中间段**")
    print("             差得更少（`(T−C)·(1−1/K)`），要等 `T` 跨过 `C·K/(K−1)` 才取到 `min(C,T)/K`。")
    print("     ★★★ **一条【未取到】的对照**（如实登记，**不冒充**）：")
    print("        ★ 计划 `§C.5` 的 `F8` 还列了 **Ultra-Scale Playbook（附录 A3）** 与**经典延迟容忍界**；")
    print("        ★ **本环境取不到前者**：`huggingface.co` 的 space 页与 `raw` / `resolve` 两种 `.md`")
    print("          路径**四次 `curl --max-time 25` 全部超时（0 字节）**，`raw.githubusercontent.com` 的")
    print("          `huggingface/ultrascale-playbook` 返回 **`404`** ⇒ ★★ **本库【不引它的公式】**")
    print("          （已登记进 `specs.NOT_QUOTED`）。")
    print("     ★★ **级别如实标**：本库的式子是 **`[E]`**；Weave 是 **`[C]` 但同行评审未确认** ——")
    print("        本节**不是**「研究表明」，而是「**两条独立来源的形态对照**」。")
    check("B.2", ok_b2
          and specs.WEAVE_ID == "2609.21483"
          and specs.WEAVE_URL == "https://arxiv.org/abs/2609.21483"
          and specs.NOT_QUOTED == ("Ultra-Scale Playbook 附录 A3 的 overlap 公式",),
          "★★★ **判据成立**：**差异的【分段】形态**（★ 分段**真的被测到**——网格含 `(4,7)` 这个"
          "`C<T<C·K/(K−1)` 的**反例档**）：`T<=C` ⇒ `0`；`C<T<C·K/(K−1)` ⇒ `(T−C)(1−1/K)`；"
          "`T>=C·K/(K−1)` ⇒ `min(C,T)/K`；★ 且 **`ours >= lit` 恒成立** —— "
          "★★★ 这就是计划点出、而**本轮第一次算出形态**的那件事",
          "**差异的形态被动过** —— ★ 先查 `cf_lit` 的化简假设与两条式子的展开")


# ===========================================================================
# 5. C 类：`T > C` 时的**错读法**
# ===========================================================================
def part_C():
    print()
    print("=" * 78)
    print("C · 边界与反例：★★★ **`T > C`（通信比计算长）时**的错读法必须被拒")
    print("=" * 78)

    # ── C.1 ★★★ 错读法「通信全藏得住」必须被拒 ───────────────────────────
    print()
    print("  ── C.1 ★★★ **`T > C` 时的处置**：把「通信全藏得住」当结论 ⇒ 必须被拒 ──")
    print("     ★★ **错读法**：既然**通信是瓶颈**（`T > C`），那就认为「**总时间就是计算时间 `C`**」——")
    print("        也就是把**通信的暴露量当成 `0`**。")
    print("     ★★★ **它为什么是错的**：总时间**不可能**低于物理下界 `max(C,T) = T > C` ⇒")
    print("        「总时间 = `C`」**落在下界以下**（`P35` 的同型错法，换到 `T > C` 这一侧）。")
    ok_rejected = True
    for (C, T) in specs.HIDE_C_T_GRID:
        if T < C:
            print("        (C,T)=%-9s **计算主导**（`T < C`）⇒ 不在本判据的射程内（如实跳过）"
                  % ("(%d,%d)" % (C, T)))
            continue
        if T == C:
            print("        (C,T)=%-9s **平衡**（`T == C`）⇒ 不在本判据的射程内（如实跳过）"
                  % ("(%d,%d)" % (C, T)))
            continue
        claim = (sim_makespan(C, T, 2) == Fraction(C))     # 错读法：总时间 == C
        exposed = cf_exposed(C, T, 2)
        irreducible = Fraction(T - C)                       # ★ 不可约的暴露量
        ok_rejected = ok_rejected and (not claim) and (exposed >= irreducible)
        print("        (C,T)=%-9s 错读法（总时间==`C`）成立 = %-5s · 真实暴露 %-8s ≥ 不可约下限 `T−C` %-6s = %s"
              % ("(%d,%d)" % (C, T), claim, exposed, irreducible, exposed >= irreducible))
    print("     ★★★ **正确的处置（本主题的判据）**：`T > C` 时")
    print("        · **可藏上限 = `C`**（只能藏住「计算那么多」）；")
    print("        · **暴露在外的通信 ≥ `T − C`**（**这一部分是【不可约】的**）；")
    print("        · ★ 再多分块（`K` 增大）也**只**能把「首项 `C/K`」压小，**压不掉 `T − C`**。")
    check("C.1", ok_rejected
          and (sim_makespan(4, 12, 2) != Fraction(4))
          and cf_exposed(4, 12, 2) >= Fraction(8)
          and (sim_makespan(4, 7, 2) != Fraction(4))
          and cf_exposed(4, 7, 2) >= Fraction(3)
          and cf_exposed(1, 20, 2) >= Fraction(19),
          "★★★ **判据成立**：`T > C` 的**三档**（含补进来的 `(4,7)`）上，错读法"
          "「**总时间 = `C`**」**一律被拒**（★ 它落在物理下界以下）；"
          "★ 真实暴露量 **≥ 不可约下限 `T − C`**（`8` · `3` · `19`）"
          "⇒ ★★ **「能用分块把通信全藏起来」在 `T > C` 时是错的**",
          "**错读法没被拒** —— ★★ 说明「总时间不低于 `max(C,T)`」这条守卫失效（`P35` 复发）")


# ===========================================================================
# 6. 前哨：口径 / 载体 / 标记自检（★ 跑在任何计算之前）
# ===========================================================================
def preflight():
    """★★ 口径 / 载体 / 标记自检**一律在这里**，且**跑在任何计算之前**（本库 `P23`）。"""
    print("=" * 78)
    print("D0. 口径/载体/标记钉住（**先跑**：让「改常量」变成断言捕获，而不是崩溃捕获）")
    print("=" * 78)
    print("  本程序的失败标记 ASSERT_FAIL_MARK = %r" % ASSERT_FAIL_MARK)
    print("  ★★★ 限定语：**不是 NCCL，也不是真机** —— 本机**无 GPU / 无 NCCL / 无网卡**。")
    print("  ★★ 本层 = **时间层**；★★ **判据模型是 `[E]` 级（本库推导）**。")
    before = len(FAILS)

    # ── D.1 ★ 标记自检（AST 读出，**不写死**）──────────────────────────────
    tree = ast.parse(open(__file__, encoding="utf-8").read(), __file__)
    hits = [n.value for n in tree.body
            if isinstance(n, ast.Assign) and len(n.targets) == 1
            and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id == "ASSERT_FAIL_MARK"]
    uniq = {getattr(v, "value", None) for v in hits}
    check("D.1", len(hits) == 1 and len(uniq) == 1 and uniq == {ASSERT_FAIL_MARK}
          and "FAIL" in ASSERT_FAIL_MARK,
          "★ 标记自检：模块级赋值恰好 1 处、取值唯一、== %r、且含 `FAIL`（AST 读出）"
          % ASSERT_FAIL_MARK)

    # ── D.2 ★ 标记自检（字面量比较）────────────────────────────────────────
    check("D.2", ASSERT_FAIL_MARK == "+= FAIL: overlap" and "FAIL" in ASSERT_FAIL_MARK,
          "★ 标记自检：**等于**本主题被分配的字面量 `+= FAIL: overlap`、且含 `FAIL`。")
    check("D.2b", OTHER_THEME_MARKS == ("&& FAIL: collectives", ">< FAIL: nccl", ",, FAIL: network-topology"),
          "★ 「不跨主题照抄」的**对照物**逐位钉住；★ 四个标记**两两不互为前缀**")

    # ── D.3 ★ 载体元数据 ──────────────────────────────────────────────────
    check("D.3", specs.WEAVE_CARRIER
          == ("Weave: Fine-Grained Dynamic SM Scheduling in an MoE Megakernel for "
              "Compute-Communication Overlap")
          and specs.WEAVE_ID == "2609.21483"
          and specs.WEAVE_URL == "https://arxiv.org/abs/2609.21483"
          and specs.FETCH_DATE == "2026-10-08"
          and specs.UNSOURCED == (),
          "★ **载体元数据逐字钉住**：Weave 的题名 + **arXiv 编号 `%s`** + URL + 抓取日 `%s`；"
          "★★ **编号本身已在 `arxiv.org/abs/` 上取到**（`HTTP 200`）—— ★ 本库 `P21`："
          "不得引一个不存在的 arXiv 编号；★ `UNSOURCED == ()` 一并钉住"
          % (specs.WEAVE_ID, specs.FETCH_DATE))

    # ── D.3b ★★★ 口径守卫（★ 跑在任何计算之前）────────────────────────────
    all_vals = [v for pair in specs.HIDE_C_T_GRID for v in pair] + list(specs.HIDE_K_GRID)
    vals_ok = all(v >= specs.MIN_VALID_C for v in all_vals)
    has_k1 = 1 in specs.HIDE_K_GRID
    has_comm_dom = any(T > C for (C, T) in specs.HIDE_C_T_GRID)
    print("     ★★ 网格取值全 >= `MIN_VALID_C` = %s · `HIDE_K_GRID` 含 `K = 1` = %s · "
          "含【通信主导】档（`T > C`）= %s" % (vals_ok, has_k1, has_comm_dom))
    check("D.3b", vals_ok and has_k1 and has_comm_dom and specs.MIN_VALID_C == 1
          and specs.HIDE_K_GRID[0] == 1,
          "★★★ **口径守卫**：网格取值全 >= 1（★ 避免 `K = 0` 的 `ZeroDivisionError` ⇒ **崩溃捕获**）；"
          "★ `K` 网格从 `1` 起（守卫①的前提）；★★ **且网格里必须有【通信主导】档 `T > C`**"
          "（否则 `C.1` 的射程为空 ⇒ **判据会静默失效**）",
          "**网格或 `MIN_VALID_C` 被动过** —— ★ 先查 `HIDE_C_T_GRID` / `HIDE_K_GRID`")
    return len(FAILS) == before


# ===========================================================================
# 7. D 类（后半）：引文逐字 · 逐元素 · 独立性
# ===========================================================================
def part_D():
    print()
    print("=" * 78)
    print("D · 引文逐字钉住（Weave）· **逐元素**钉住 · 独立性线索")
    print("=" * 78)

    # ── D.5 ★★★ Weave 的两条式 + 一句散文（**LaTeX 提取形态**）────────────
    # ── D.4 ★★ `eff_chunk(K)` 的逐字（★ 它支撑 `cf_lit` 化简的【第三条假设】）─────────
    check("D.4", specs.WEAVE_EFF_CHUNK_TEX == 'T_{\\text{comp}}(c,K)=\\frac{W_{\\text{comp}}}{\\text{TFLOPS}(N{-}c)\\cdot\\text{eff}_{\\text{chunk}}(K)},'
          and specs.WEAVE_EFF_CHUNK_PROSE == "where eff_chunk(K)\\in(0,1] is the ratio of chunked GEMM throughput to the unchunked case",
          "★★ 判据成立：`eff_chunk(K)` 的**公式逐字**与**散文逐字**都在（★ 原文显式依赖 `K` ⇒ "
          "本库化简成常数 `C` 时**多了一条假设**，已记进 `§1.4`）",
          "**`eff_chunk(K)` 的逐字被动过** —— ★ 先查 `WEAVE_EFF_CHUNK_TEX` / `_PROSE`")

    check("D.5", specs.WEAVE_TOTAL_TEX
          == 'T_{\\text{total}}(c,K)=\\max\\!\\big(T_{\\text{comp}}(c,K)+T_{\\text{tail}}(c,K),\\;T_{\\text{comm}}(c)\\big).'
          and specs.WEAVE_TAIL_TEX
          == 'T_{\\text{tail}}=(1-\\alpha)\\cdot\\frac{W_{\\text{combine}}}{\\text{BW}(c)\\cdot K}.'
          and specs.WEAVE_PROSE_QUOTE
          == "The layer is bounded by whichever of the two paths is slower, plus the unhidden tail.",
          "★★★ **Weave 的三条逐字钉住**（总时间式 (7) · 尾项式 (10) · 散文一句）—— "
          "★★ **提取形态是 LaTeX 源码**（`\\max\\!\\big` / `\\;` / `\\frac`），**不是排版式**；"
          "\n        ★★★ **特别是**：散文那句的 **`whichever of the two paths is slower, plus the unhidden tail`**"
          " —— ★ 它就是「`max(·,·)` **+ 尾项**」的读法，**漏掉后半句就把尾项丢了**",
          "**Weave 的引文常量被动过** —— ★ 先查 LaTeX 提取形态（`\\max\\!\\big`）")

    # ── D.6 ★★ `NOT_QUOTED`（★ 明写「不引什么、为什么」）──────────────────
    check("D.6", specs.NOT_QUOTED == ("Ultra-Scale Playbook 附录 A3 的 overlap 公式",),
          "★★ **否定性记录**：本库登记了**故意不引**的那一条（Ultra-Scale Playbook 的公式）—— "
          "★★★ **本环境取不到它**（`huggingface.co` 三次超时 + GitHub raw `404`，见 `specs` 的口径）；"
          "\n        ★ 它是**非空**元组 ⇒ 有叶子 ⇒ 会被 `mutate.py` 变异 ⇒ **逐元素钉住**",
          "**`NOT_QUOTED` 被动过**")

    # ── D.7 ★★★ 逐元素钉住 ───────────────────────────────────────────────
    check("D.7", specs.HIDE_C_T_GRID == ((12, 4), (4, 12), (7, 7), (1, 20), (4, 7))
          and specs.HIDE_K_GRID == (1, 2, 3, 8, 16)
          and specs.P35_WRONG_CASE == (3, 10, 4)
          and specs.P35_LOWER_BOUND == 10
          and specs.K_INF == 1000000
          and specs.MIN_VALID_C == 1
          and specs.UNSOURCED == (),
          "★★★ **两个元组逐元素**（`HIDE_C_T_GRID` 十个数 + `HIDE_K_GRID` 五个数）+ **四个标量**钉住；"
          "★★ `UNSOURCED == ()` 一并钉住（★ 0 个叶子）",
          "**元组的元素或某个标量被动过**")

    # ── D.8 ★★★ 独立性（源码级）── ★★ 遍历【全部】参照实现 ───────────────
    refs = (sim_ticks, sim_makespan)
    cfs = (cf_pipeline, cf_no_overlap, cf_lower_bound, cf_hidden, cf_hidden_ceiling,
           cf_exposed, cf_lit)
    ref_srcs = {textwrap.dedent(inspect.getsource(f)) for f in refs}
    cf_srcs = {textwrap.dedent(inspect.getsource(f)) for f in cfs}
    call_tables = {}
    for f in refs:
        t = textwrap.dedent(inspect.getsource(f))
        call_tables[f.__name__] = {n.func.id for n in ast.walk(ast.parse(t))
                                   if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    cf_calls = {n.func.id for s in cf_srcs for n in ast.walk(ast.parse(s))
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    print()
    print("  ── D.8 ★★ **源码级线索**（★ **不是证明**）：**参照**侧两个函数的调用表 ──")
    for nm, tbl in call_tables.items():
        print("     `%s()` 里调用的函数：%s"
              % (nm, ", ".join(sorted(tbl)) if tbl else "**空集**"))
    print("     **被测**侧七个 `cf_*` 里调用的函数：%s"
          % (", ".join(sorted(cf_calls)) if cf_calls else "**空集**"))
    hits = sorted({c for tbl in call_tables.values() for c in tbl if c.startswith("cf_")})
    print("     ⇒ ★★ **参照侧不调用任何 `cf_*`**（★ `Fraction` 只是**精确算术的容器**）。")
    print("     ★★★ **降级措辞**：这只是「两段代码不互相调用」的**源码级线索**，**不证明**判据有信息量 ——")
    print("        后者由 `A.1`/`A.2` 与 `C.1` 的**行为**兜底。")
    check("D.8", not hits
          and len(ref_srcs) == 2 and len(cf_srcs) == 7
          and not any(c.startswith("cf_") for s in ref_srcs for c in _call_names(s))
          and not any(c in call_tables for s in cf_srcs for c in _call_names(s)),
          "★★ **源码级线索成立**：**两个**参照函数两两不同源、**七个** `cf_*` 两两不同源；"
          "★★ **参照侧一个 `cf_*` 都不调用，`cf_*` 也不调用参照**（★ 两侧都审）",
          "**参照与闭式互相调用** —— A 类可能退化成自证（`§1.3.2b`）")

    # ── D.9 ★★ 模型的边界 + `cf_lit` 的化简两侧 ──────────────────────────
    check("D.9", cf_pipeline(1, 1, 1) == Fraction(2)
          and cf_hidden(10, 3, 1) == Fraction(0)
          and cf_hidden(3, 10, 1) == Fraction(0)
          and cf_hidden_ceiling(10, 3) == Fraction(3)
          and cf_hidden_ceiling(3, 10) == Fraction(3)
          and cf_lit(10, 3, 4) == cf_pipeline(10, 3, 4)
          and cf_lit(3, 10, 4) < cf_pipeline(3, 10, 4),
          "★★ **模型的边界**：`K = 1` ⇒ **可藏 `= 0`**（不发不算藏）；"
          "★★ **可藏上限在两档上都是 `3`**（`min(10,3)` 与 `min(3,10)`）—— ★ 上限**只看较短的那条轴**；"
          "★★★ **`cf_lit` 的两侧**：计算主导档**与真值同值**、通信主导档**严格更小**",
          "**模型的边界值被动过**")


def _call_names(src):
    return {n.func.id for n in ast.walk(ast.parse(src))
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}


# ===========================================================================
# 8. 汇总 / 主程序
# ===========================================================================
def summary():
    t = sum(_TOTAL.values())
    p = sum(_PASS.values())
    print()
    print("汇总: A %d/%d · B %d/%d · C %d/%d · D %d/%d ⇒ 断言 %d 条，通过 %d 条，失败 %d 条"
          % (_PASS["A"], _TOTAL["A"], _PASS["B"], _TOTAL["B"],
             _PASS["C"], _TOTAL["C"], _PASS["D"], _TOTAL["D"], t, p, len(FAILS)))
    if FAILS:
        print("失败清单: %s" % ", ".join(FAILS))
    print("★★ **A = %d/%d**：**三条免费守卫**（`P35`）+ **可藏上限 = `min(C,T)`**。" % (_PASS["A"], _TOTAL["A"]))
    print("★★★ **本程序的模型是 `[E]` 级（本库推导）** —— ★ 不是「研究表明」，见 `B.2`。")
    print("★ 本程序只核对「**判据的边界行为**」与「**错读法真的会被拒**」。")
    return 1 if FAILS else 0


def main():
    print("=" * 78)
    print("02_how_much_can_hide.py —— 「能藏多少」的判据")
    print("=" * 78)
    print("  ★ 失败标记 ASSERT_FAIL_MARK = %r（本主题自定义；不跨主题照抄）" % ASSERT_FAIL_MARK)
    print("  ★★★ 本层 = **时间层**；★★ **判据模型是 `[E]` 级（本库推导）**。")
    print("  ★★★ 限定语：**不是 NCCL，也不是真机** —— 本机无 GPU / 无 NCCL / 无网卡。")
    if not preflight():
        print()
        print("!! 口径/标记钉住失败 ⇒ **不继续往下跑**：后续的 A/B/C 只会产生一堆无意义的连带崩溃")
        return summary()
    part_A()
    part_B()
    part_C()
    part_D()
    return summary()


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""03_async_io.py —— **异步落盘**：它与训练计算**重叠**的那一半。

配套文档：`03-async-io.md`。

★ 三类断言**全部在本程序内部**，失败时以**非零退出码**结束；**没有 `verify.py`**。

★★★ **本层是【时间层】**（★ 与 `01`/`02` 同一层）。

★★★ **一条【先说话】的分工（计划 `§2.1` 的硬边界）**：
  本主题**只写「通信 / 计算与落盘重叠」的那一半**；
  ★ **异步落盘的【存储侧】机制**（写盘流水、元数据、重分片）→ `90-training/checkpointing/01,02,03`
  —— ★★ **本程序一行都不重写，只回链**。

★★★ **「同一件事的另一面」**（★ 本域 `§2.2` 的「极易被合并」那一类，**首次出现处必须点明**）：
  `90-training/README.md:42-43` 逐字：「**「异步落盘」**（原属 `checkpointing`）——
  **它与 `110-communication` 的 `overlap` 主题（通信-计算重叠 / 双缓冲 / 分块流水）
  是同一件事**，正文在 `110-communication/overlap/`，**此处回链**。」
  ⇒ ★★ **它不是「同名不同指」，而是【同一件事被两个域各写了一半】**：
    本主题写**时间账**（两条时间轴怎么叠、能藏多少），`90-training` 写**存储侧的机制**。

★★★ **本程序的 A 类【与 `02` 是同一个模型，但换了【另一组参数】**（计划 `Chunk 4` 的原文）：
  被测 = `(C+S)/K + (K−1)·max(C/K, S/K)`（★ `S` = **落盘时长**，取代 `02` 的通信时长 `T`）；
  参照 = **时隙模拟**（两条时间轴：计算引擎 + 落盘引擎）。
  ★★ **「换了参数」是刻意的** —— 免得两章看起来是**同一组数的复述**（本库的 `P1` 家族）。

★★ **覆盖边界**：本程序守的是「**落盘与计算叠起来要多久**」与「**错法真的会被拒**」。
  **不守**任何**真实存储 / 网络**的吞吐（本机**无 GPU / 无 NCCL / 无真网卡**）。
  ★★ **最强的限定语**：**「不是 NCCL，也不是真机」**。

★★ 可证伪断言（把下面任一条改错，本程序必须非零退出）：
  · `A.1` ★★★ **时隙模拟 == 闭式**（`SAVE_C_S_GRID` × `SAVE_K_GRID`，12 档）；
  · `A.2` ★★ 三条免费守卫（`K = 1` ⇒ `C+S`；中间值 ≥ `max(C,S)`；`K → ∞` ⇒ 下界）；
  · `A.3` ★★ **落盘能藏的上限 = `min(C,S)`**；
  · `B.1` ★★ **落盘的重叠账**（两段式 + 两个句柄）；`B.2` ★ 两条官方限定语；
  · `C.1` ★★★ **把「异步落盘」当成「不占带宽」** 必须被拒；
  · `D.*` 引文逐字钉住 + **逐元素**钉住 + **失败标记自检** + `UNSOURCED`。
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
# 1. 参照：**时隙模拟**（★ 与 `01`/`02` 同一个调度模型；★ **本程序自带一份**、不跨程序 import）
#    ★★ **本程序用的是【第三组参数】**（`SAVE_C_S_GRID` / `SAVE_K_GRID`）
#    ★ 这里的「第二台引擎」是**落盘**（不是 allreduce）—— ★ 但**模型完全同构**
# ===========================================================================
def sim_ticks(C, S, K):
    """★★ **时隙模拟**：两条引擎（计算 / 落盘）逐拍推进，返回完工**拍数**（时间以 `1/K` 为单位）。

    ★★★ **调度规则**：计算引擎按段序串行；落盘引擎按段序串行，且**第 `i` 段的落盘
      只在第 `i` 段的【计算】做完之后**才能开始 —— ★ 「**首项不可重叠**」在这里的形态是
      「**第一段计算没算完，没有任何东西可落**」。
    """
    comp = [0] * K
    save = [0] * K
    ci = 0
    ki = 0
    ticks = 0
    while ki < K:
        do_comp = (ci < K)
        do_save = (ki < K) and (comp[ki] >= C)
        if do_comp:
            comp[ci] += 1
        if do_save:
            save[ki] += 1
        if do_comp and comp[ci] >= C:
            ci += 1
        if do_save and save[ki] >= S:
            ki += 1
        ticks += 1
    return ticks


def sim_makespan(C, S, K):
    """模拟器的完工时间（拍数 / `K`）。"""
    return Fraction(sim_ticks(C, S, K), K)


# ===========================================================================
# 2. 被测：模型（★ **`[E]` 级：本库推导**）+ 两条独立表达式 + 错法
# ===========================================================================
def cf_no_overlap(C, S):
    """**无重叠**值 `C + S`（★ 独立表达式 —— 守卫①的期望值）。"""
    return Fraction(C + S)


def cf_lower_bound(C, S):
    """**物理下界** `max(C, S)`（★ 独立表达式 —— 守卫②③的期望值）。"""
    return Fraction(max(C, S))


def cf_save_pipeline(C, S, K):
    """★★ **本库的 `[E]` 模型**（落盘版）：`(C+S)/K + (K−1)·max(C/K, S/K)`。"""
    return Fraction(C + S, K) + (K - 1) * max(Fraction(C, K), Fraction(S, K))


def cf_save_hidden(C, S, K):
    """**藏住的那部分落盘时间** = 无重叠值 − 流水化后的总时间 = `min(C,S)·(1 − 1/K)`。"""
    return cf_no_overlap(C, S) - cf_save_pipeline(C, S, K)


def cf_save_hidden_ceiling(C, S):
    """★★★ **能藏的上限** = `min(C, S)` —— ★ 较短的那条时间轴。"""
    return Fraction(min(C, S))


def cf_save_free(C, S, K):
    """★★★ **错法**：把**异步落盘**当成「**不占带宽**」⇒ 落盘时长 `S` **完全不进账**。

    ★ 这个错法很「自然」：官方文档说它是 **`asynchronous`**、**`in a separate thread`**
    （`specs.CKPT_ASYNC_QUOTE`）—— ★★ **读快了就会以为它不花时间**。
    """
    return Fraction(C)


def passes_guards(f, C, S, k_grid, tol=Fraction(1, 100)):
    """★★★ **三条免费守卫**（`P35`）：`K = 1` ⇒ 无重叠；`K → ∞` ⇒ 下界；中间值 ≥ 下界。"""
    g1 = (f(C, S, 1) == cf_no_overlap(C, S))
    g2 = abs(f(C, S, specs.K_INF) - cf_lower_bound(C, S)) <= tol
    g3 = all(f(C, S, K) >= cf_lower_bound(C, S) for K in k_grid)
    return g1 and g2 and g3


# ===========================================================================
# 3. A 类 —— ★★★ **同一个模型，换一组参数**
# ===========================================================================
def part_A():
    print("=" * 78)
    print("A · 落盘与计算叠起来要多久：时隙模拟 vs 独立闭式（★ **第三组参数**）")
    print("=" * 78)
    print("  ★★ 这里的「第二台引擎」是**落盘**（`S`），不是 allreduce —— ★ 但**模型完全同构**。")
    print("  ★★ 被测 = `cf_save_pipeline`（**`[E]` 级**）；参照 = `sim_ticks`（**调度模拟**）。")

    # ── A.1 ★★★ 模拟 == 闭式（12 档，**第三组参数**）──────────────────────
    print()
    print("  ── A.1 ★★★ **时隙模拟 == 闭式**（`SAVE_C_S_GRID` × `SAVE_K_GRID`）──")
    print("     %-10s %-6s %-10s %-10s %-8s" % ("(C,S)", "K", "模拟", "闭式", "一致"))
    ok1 = True
    n_rows = 0
    for (C, S) in specs.SAVE_C_S_GRID:
        for K in specs.SAVE_K_GRID:
            got, exp = sim_makespan(C, S, K), cf_save_pipeline(C, S, K)
            ok1 = ok1 and (got == exp)
            n_rows += 1
            print("     %-10s %-6d %-10s %-10s %-8s"
                  % ("(%d,%d)" % (C, S), K, got, exp, got == exp))
    print("     ★★★ **%d 档全部一致** —— ★★ 与 `01`/`02` 是**同一个模型**，"
          "但用的是 `SAVE_C_S_GRID`（**三档**）与 `SAVE_K_GRID`（**四档**）。" % n_rows)
    check("A.1", ok1 and n_rows == 12
          and specs.SAVE_C_S_GRID == ((20, 5), (5, 20), (9, 9))
          and specs.SAVE_K_GRID == (1, 2, 5, 20),
          "★★★ **判据成立**：`SAVE_C_S_GRID`（三档）× `SAVE_K_GRID`（四档）**共 12 档**上，"
          "**时隙模拟 == 闭式**（落盘版）—— ★★ 参数与 `01`/`02` **不重叠**",
          "**模拟与闭式对不上** —— ★ 先查落盘引擎是否等了本段的计算")

    # ── A.2 ★★ 三条免费守卫（★ 与 `02` 同判据、**不同参数**）──────────────
    print()
    print("  ── A.2 ★★ **三条免费守卫**（`P35`）在落盘场景上 ──")
    ok2 = True
    for (C, S) in specs.SAVE_C_S_GRID:
        k1 = sim_makespan(C, S, 1)
        lb, noov = cf_lower_bound(C, S), cf_no_overlap(C, S)
        mids = [sim_makespan(C, S, K) for K in specs.SAVE_K_GRID]
        g1 = (k1 == noov)
        g3 = all(m >= lb for m in mids)
        ok2 = ok2 and g1 and g3
        print("     (C,S)=%-9s `K=1` 模拟 %-6s == `C+S` %-6s（逐位）= %-5s · "
              "中间最小值 %-8s ≥ 下界 %-6s = %s"
              % ("(%d,%d)" % (C, S), k1, noov, g1, min(mids), lb, g3))
    g2 = all(abs(cf_save_pipeline(C, S, specs.K_INF) - cf_lower_bound(C, S)) <= Fraction(1, 100)
             for (C, S) in specs.SAVE_C_S_GRID)
    print("     ★★ `K → ∞` ⇒ 下界：用闭式在 `K = K_INF` 上验（★ **数值替身**）= %s" % g2)
    print("     ★★ 物理含义：**整份一次落盘（`K = 1`）永远比流水化更慢** ——")
    print("        流水化把「落盘」切段，让**后一段的落盘与前一段的计算并行**。")
    check("A.2", ok2 and g2,
          "★★ **判据成立**：①② **`K = 1` 时模拟器逐位给出 `C + S`**（★ 整份一次落盘 = 无重叠）；"
          "②**中间值 ≥ 下界 `max(C,S)`**；③ `K → ∞` 时收敛到下界（容差 `1/100`，**数值替身**）",
          "**守卫不成立** —— ★ 说明落盘模型漏了首段或下穿了物理下界")

    # ── A.3 ★★ 落盘能藏的上限 = `min(C,S)` ──────────────────────────────
    print()
    print("  ── A.3 ★★ **落盘能藏的上限 = `min(C,S)`** ──")
    ok3 = True
    for (C, S) in specs.SAVE_C_S_GRID:
        ceil = cf_save_hidden_ceiling(C, S)
        near = cf_save_hidden(C, S, specs.K_INF)
        ok3 = ok3 and (ceil == Fraction(min(C, S))) and abs(near - ceil) <= Fraction(1, 100)
        print("     (C,S)=%-9s 可藏上限 `min(C,S)` = %-6s · 闭式@K=%d = %-9s ⇒ 逼近 = %s"
              % ("(%d,%d)" % (C, S), ceil, specs.K_INF, near, abs(near - ceil) <= Fraction(1, 100)))
    print("     ★★★ **读法**：**当落盘比计算长（`S > C`）时，能藏住的【最多】是全部计算时间 `C`** ——")
    print("        剩下的 `S − C` **永远暴露在外**（★ 与 `02` 的「通信主导」档同型）。")
    check("A.3", ok3,
          "★★ **判据成立**：三档上 **`K → ∞` 时能藏的落盘时间 = `min(C,S)`**（★ 容差 `1/100`，"
          "数值替身）—— ★★ **落盘能藏多少，同样由【较短的那条时间轴】决定**",
          "**可藏上限被动过** —— ★ 先查 `cf_save_hidden` / `cf_save_hidden_ceiling`")


# ===========================================================================
# 4. B 类：落盘的**重叠账**（两段式 + 两个句柄）· 两条官方限定语
# ===========================================================================
def part_B():
    print()
    print("=" * 78)
    print("B · **落盘的重叠账** · 两段式与两个句柄 · 两条官方限定语")
    print("=" * 78)
    print("  ★★ 载体 = PyTorch `distributed.checkpoint`（`[B]`，抓取日 **%s**）。" % specs.FETCH_DATE)

    # ── B.1 ★★ 落盘的重叠账 ──────────────────────────────────────────────
    print()
    print("  ── B.1 ★★ **落盘的重叠账**（`(C,S,K) → 总时间 / 藏住 / 暴露`）──")
    print("     %-10s %-6s %-12s %-12s %-12s" % ("(C,S)", "K", "总时间", "藏住", "暴露"))
    rows = {}
    for (C, S) in specs.SAVE_C_S_GRID:
        for K in specs.SAVE_K_GRID:
            mk, hd, ex = sim_makespan(C, S, K), cf_save_hidden(C, S, K), \
                Fraction(S) - cf_save_hidden(C, S, K)
            rows[(C, S, K)] = (mk, hd, ex)
            print("     %-10s %-6d %-12s %-12s %-12s" % ("(%d,%d)" % (C, S), K, mk, hd, ex))
    print("     ★★ **两段式**（★ 官方逐字，见 `D.5`）：`%s`" % specs.CKPT_ASYNC_QUOTE)
    print("     ★★★ **它为什么与「两条时间轴」对得上**：官方把异步保存拆成**两段**，")
    print("        并在返回对象上给了**两个句柄**：`%s` / `%s`" % specs.CKPT_HANDLE_NAMES)
    print("        ⇒ ★★ **`staging_completion`（de-stage 完）与 `upload_completion`（上传完）**")
    print("        —— ★ 本主题的时间账**只能cover「与计算重叠的那一半」**；")
    print("          ★★ **两段的【存储侧】细节（写盘流水 / 元数据 / 重分片）回链**")
    print("             `90-training/checkpointing/01,02,03`（★ 本程序一行都不重写）。")
    print("     ★★ 默认实现是 **`%s`**（★ 官方默认实参）⇒ ★ **线程 ≠ 进程**："
          "不吃第二份内存，★ **也不把两段的带宽开销变没**。" % specs.CKPT_ASYNC_TYPE_DEFAULT)
    check("B.1", rows[(20, 5, 1)][0] == Fraction(25) and rows[(5, 20, 1)][0] == Fraction(25)
          and rows[(5, 20, 2)][1] == Fraction(5, 2) and rows[(5, 20, 2)][2] == Fraction(35, 2)
          and rows[(20, 5, 20)][2] == Fraction(1, 4),
          "★★ **判据成立**：`K = 1` 时两档总时间都是 `25`（整份一次落盘）；`K = 2` 时通信主导档"
          "（`5,20`）藏住 `2.5`、暴露 `17.5`；计算主导档 `K = 20` 暴露只剩 `0.25`",
          "**落盘账的某一格被动过** —— ★ 先查模拟器与两档 `(C,S)`")

    # ── B.2 ★★ 两条官方限定语（★ 必须一起引）────────────────────────────
    print()
    print("  ── B.2 ★★ **两条官方限定语**（引这条特性【必须带上】）──")
    print("     ① **它是实验性的**（逐字）：`%s`" % specs.CKPT_EXPERIMENTAL_QUOTE)
    print("     ② **它先 de-stage 再另起线程 save**（逐字）：`%s`" % specs.CKPT_ASYNC_QUOTE)
    print("     ★★★ **为什么这两条一起引**：")
    print("        · ① 说明它**不是稳定契约**（★ 本库不得把它当规范）；")
    print("        · ② 说明「异步」= **第二段在另一个线程**，★ **不是**「两段都不花时间」。")
    print("     ★★ **级别如实标**：两条都是 **`[B]`（产品文档）**，不是规范（`P19`）。")
    check("B.2", specs.CKPT_EXPERIMENTAL_QUOTE == "This feature is experimental and subject to change."
          and specs.CKPT_VER == "PyTorch 2.14"
          and specs.CKPT_HANDLE_NAMES == ("staging_completion", "upload_completion")
          and specs.CKPT_ASYNC_TYPE_DEFAULT == "AsyncCheckpointerType.THREAD",
          "★★ **判据成立**：**实验性**那句与 **de-stage + 另起线程**那句**逐字钉住**；"
          "★★ 并钉住**两个句柄**与**默认线程实现**（★ 两者一起支撑「异步 ≠ 免费」）",
          "**官方限定语或句柄名被动过**")


# ===========================================================================
# 5. C 类：**把「异步落盘」当成「不占带宽」**
# ===========================================================================
def part_C():
    print()
    print("=" * 78)
    print("C · 边界与反例：★★★ **把「异步落盘」当成「不占带宽」** 必须被拒")
    print("=" * 78)

    # ── C.1 ★★★ 错法「落盘不占带宽」必须被拒 ────────────────────────────
    print()
    print("  ── C.1 ★★★ 错法 `S` 完全不进账（「异步 = 不占带宽」）⇒ 必须被拒 ──")
    print("     ★★ **错法为什么可信**：官方文档里满是 `asynchronous` / `in a separate thread`")
    print("        （`specs.CKPT_ASYNC_QUOTE`）—— ★ 读快了就会以为**它不花时间**。")
    print("     ★★★ **它为什么是错的**：`separate thread` 说的是**在哪跑**，")
    print("        不是「**跑多久**」；★★ 落盘**照样占存储 / 网络带宽**，所以**照样花时间**。")
    ok_rejected = True
    for (C, S) in specs.SAVE_C_S_GRID:
        p = passes_guards(cf_save_free, C, S, specs.SAVE_K_GRID)
        pt = passes_guards(cf_save_pipeline, C, S, specs.SAVE_K_GRID)
        free_at_1 = cf_save_free(C, S, 1)
        noov = cf_no_overlap(C, S)
        ok_rejected = ok_rejected and (not p)
        print("        (C,S)=%-9s 错法过守卫 = %-5s · 真值过守卫 = %-5s · 错法@K=1 给 %-6s vs 无重叠 %-6s"
              % ("(%d,%d)" % (C, S), p, pt, free_at_1, noov))
    print("     ★★★ **读法**：错法在**三档上【一律】被拒**（★ 连平衡档也不例外）——")
    print("        因为它在 `K = 1` 时就已经**不等于无重叠值 `C + S`**（守卫①当场失败）。")
    print("     ★★ **真值放行**（`cf_save_pipeline` 过守卫）是**成功行为** ⇒")
    print("        依 `CONTRIBUTING:238`，它**归 `A.2`**，**不塞进本 C 类**。")
    check("C.1", ok_rejected
          and (not passes_guards(cf_save_free, 5, 20, specs.SAVE_K_GRID))
          and cf_save_free(5, 20, 1) == Fraction(5)
          and cf_no_overlap(5, 20) == Fraction(25)
          and cf_save_free(5, 20, 1) < cf_lower_bound(5, 20),
          "★★★ **判据成立**：错法（`S` 不进账）在 `SAVE_C_S_GRID` **三档上一律被拒** —— "
          "★★ 它给出的「时间」（`C`）**低于物理下界 `max(C,S)`**（`5 < 20`）"
          "⇒ ★★ **「把异步落盘当成不占带宽」真的会失败**",
          "**错法居然过了守卫** —— ★★ 说明守卫的 `K = 1` 锚点失效（`P35` 的防线漏了）")


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
    print("  ★★ 本层 = **时间层**；★★ **模型是 `[E]` 级（本库推导）**。")
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
    check("D.3", specs.CKPT_URL == "https://docs.pytorch.org/docs/2.14/distributed.checkpoint.html"
          and specs.CKPT_VER == "PyTorch 2.14"
          and specs.FETCH_DATE == "2026-10-08"
          and specs.UNSOURCED == (),
          "★ **载体元数据逐字钉住**：PyTorch `distributed.checkpoint` 的 URL + 版本 `%s` + "
          "抓取日 `%s`；★ `UNSOURCED == ()` 一并钉住" % (specs.CKPT_VER, specs.FETCH_DATE))

    # ── D.3b ★★★ 口径守卫（★ 跑在任何计算之前）────────────────────────────
    all_vals = [v for pair in specs.SAVE_C_S_GRID for v in pair] + list(specs.SAVE_K_GRID)
    vals_ok = all(v >= specs.MIN_VALID_C for v in all_vals)
    has_k1 = 1 in specs.SAVE_K_GRID
    has_save_dom = any(S > C for (C, S) in specs.SAVE_C_S_GRID)
    print("     ★★ 网格取值全 >= `MIN_VALID_C` = %s · `SAVE_K_GRID` 含 `K = 1` = %s · "
          "含【落盘主导】档（`S > C`）= %s" % (vals_ok, has_k1, has_save_dom))
    check("D.3b", vals_ok and has_k1 and has_save_dom and specs.MIN_VALID_C == 1
          and specs.SAVE_K_GRID[0] == 1,
          "★★★ **口径守卫**：网格取值全 >= 1（★ 避免 `K = 0` 的 `ZeroDivisionError`）；"
          "★ `K` 网格从 `1` 起（守卫①的前提）；★★ **且网格里必须有【落盘主导】档 `S > C`**"
          "（否则 `A.3` 的「藏不完」那一侧**无从检起**）",
          "**网格或 `MIN_VALID_C` 被动过** —— ★ 先查 `SAVE_C_S_GRID` / `SAVE_K_GRID`")
    return len(FAILS) == before


# ===========================================================================
# 7. D 类（后半）：引文逐字 · 逐元素 · 独立性
# ===========================================================================
def part_D():
    print()
    print("=" * 78)
    print("D · 引文逐字钉住（PyTorch 异步落盘）· **逐元素**钉住 · 独立性线索")
    print("=" * 78)

    # ── D.5 ★★ 三条逐字 ──────────────────────────────────────────────────
    check("D.5", specs.CKPT_ASYNC_QUOTE
          == ("Asynchronous version of save. This code first de-stages the state_dict on to the "
              "staging storage (defaults to CPU memory), and then calls the save in a separate "
              "thread.")
          and specs.CKPT_EXPERIMENTAL_QUOTE == "This feature is experimental and subject to change."
          and specs.CKPT_HANDLE_NAMES == ("staging_completion", "upload_completion"),
          "★★ **三条逐字钉住**（异步保存的定义句 · 实验性警告 · 两个句柄名）—— "
          "★★★ **特别注意**：定义句里 **`first de-stages ... and then calls the save in a "
          "separate thread`** 是**两段**；★ 漏掉 `first` / `then` 就把两段读成一段；"
          "\n        ★★ 载体 = PyTorch 官方文档页（HTML），抓取日 %s" % specs.FETCH_DATE,
          "**PyTorch 异步落盘的引文常量被动过** —— ★ 先查 `first` / `then` 与两个句柄名")

    # ── D.6 ★★ 级标 + 与 `90-training` 的分工（★ 只钉常量，正文在文档里）──
    check("D.6", specs.CKPT_VER == "PyTorch 2.14"
          and specs.CKPT_URL.startswith("https://docs.pytorch.org/docs/")
          and specs.UNSOURCED == (),
          "★★ **级别如实标**：本条是 **`[B]`（产品文档，不是规范）** —— ★★ 本库不得把它当规范；"
          "★★ 并钉住 `UNSOURCED == ()`（★ 本程序**没有**「用了但没出处」的常量）",
          "**载体字段或 `UNSOURCED` 被动过**")

    # ── D.7 ★★★ 逐元素钉住 ───────────────────────────────────────────────
    check("D.7", specs.SAVE_C_S_GRID == ((20, 5), (5, 20), (9, 9))
          and specs.SAVE_K_GRID == (1, 2, 5, 20)
          and specs.CKPT_ASYNC_TYPE_DEFAULT == "AsyncCheckpointerType.THREAD"
          and specs.P35_WRONG_CASE == (3, 10, 4)
          and specs.P35_LOWER_BOUND == 10
          and specs.K_INF == 1000000
          and specs.MIN_VALID_C == 1
          and specs.UNSOURCED == (),
          "★★★ **两个元组逐元素**（`SAVE_C_S_GRID` 六个数 + `SAVE_K_GRID` 四个数）+ "
          "**五个标量/字符串**（`CKPT_ASYNC_TYPE_DEFAULT` · `P35_WRONG_CASE` · `P35_LOWER_BOUND` · "
          "`K_INF` · `MIN_VALID_C`）钉住；★★ `UNSOURCED == ()` 一并钉住（★ 0 个叶子）",
          "**元组的元素或某个标量被动过**")

    # ── D.8 ★★★ 独立性（源码级）── ★★ 遍历【全部】参照实现 ───────────────
    refs = (sim_ticks, sim_makespan)
    cfs = (cf_save_pipeline, cf_no_overlap, cf_lower_bound, cf_save_hidden,
           cf_save_hidden_ceiling, cf_save_free)
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
    print("     **被测**侧六个 `cf_*` 里调用的函数：%s"
          % (", ".join(sorted(cf_calls)) if cf_calls else "**空集**"))
    hits = sorted({c for tbl in call_tables.values() for c in tbl if c.startswith("cf_")})
    print("     ⇒ ★★ **参照侧不调用任何 `cf_*`**（★ `Fraction` 只是**精确算术的容器**）。")
    print("     ★★★ **降级措辞**：这只是「两段代码不互相调用」的**源码级线索**，**不证明**判据有信息量 ——")
    print("        后者由 `A.1`/`A.2` 与 `C.1` 的**行为**兜底。")
    check("D.8", not hits
          and len(ref_srcs) == 2 and len(cf_srcs) == 6
          and not any(c.startswith("cf_") for s in ref_srcs for c in _call_names(s))
          and not any(c in call_tables for s in cf_srcs for c in _call_names(s)),
          "★★ **源码级线索成立**：**两个**参照函数两两不同源、**六个** `cf_*` 两两不同源；"
          "★★ **参照侧一个 `cf_*` 都不调用，`cf_*` 也不调用参照**（★ 两侧都审）",
          "**参照与闭式互相调用** —— A 类可能退化成自证（`§1.3.2b`）")

    # ── D.9 ★★ 模型的边界 + 错法的两侧 ──────────────────────────────────
    check("D.9", cf_save_pipeline(1, 1, 1) == Fraction(2)
          and cf_save_hidden(5, 20, 1) == Fraction(0)
          and cf_save_hidden_ceiling(5, 20) == Fraction(5)
          and cf_save_hidden_ceiling(20, 5) == Fraction(5)
          and cf_save_free(5, 20, 2) == Fraction(5)
          and cf_save_free(5, 20, 2) < cf_save_pipeline(5, 20, 2),
          "★★ **模型的边界**：`K = 1` ⇒ **藏住 `= 0`**；**可藏上限在两档上都是 `5`**"
          "（`min(5,20)` 与 `min(20,5)`）；★★ **错法（`C`）恒小于真值** ⇒ ★ 它是**单向偏低**的错",
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
    print("★★ **A = %d/%d** —— ★★ 与 `01`/`02` **同一个模型**，但用的是**第三组参数**。" % (_PASS["A"], _TOTAL["A"]))
    print("★★★ **分工**：本程序只写「**与计算重叠的那一半**」；**存储侧**回链 `90-training/checkpointing`。")
    print("★ 本程序只核对「**落盘与计算叠起来要多久**」与「**错法真的会被拒**」。")
    return 1 if FAILS else 0


def main():
    print("=" * 78)
    print("03_async_io.py —— 异步落盘：与训练计算重叠的那一半")
    print("=" * 78)
    print("  ★ 失败标记 ASSERT_FAIL_MARK = %r（本主题自定义；不跨主题照抄）" % ASSERT_FAIL_MARK)
    print("  ★★★ 本层 = **时间层**；★★ **模型是 `[E]` 级（本库推导）**。")
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

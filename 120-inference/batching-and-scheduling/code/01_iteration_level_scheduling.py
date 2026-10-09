#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""01_iteration_level_scheduling.py —— **迭代级调度**：continuous batching 的机制与收益来源。

配套文档：`01-iteration-level-scheduling.md`。

★ 三类断言**全部在本程序内部**，失败时以**非零退出码**结束；**没有 `verify.py`**。

★★★ **本主题不写【定义】**：`continuous batching` / `chunked prefill` 的**定义**在
  `10-foundations/performance-models/01-roofline-model.md:251`（括注定义），**本程序只回链**。
  本程序写的是**机制**与**收益来源**。

★★★ **本程序最强的一条 A 类：间隙的闭式**（计划 `§0.7 D9`）——
  · **被测** = `sim_cont`：**逐 token 步推进**的 continuous batching 模拟器（事件循环）；
  · **参照** = `makespan_closed`：**闭式** `N·L/B + 间隙`，`间隙 = (B − N mod B)·L/B`；
  · **两条路径无共享代码**（一个循环、一个式子；`D.8` 在源码层核）。
  ★★★ **判据**：**等长 + 同时到达** 的 `(N, B)` 网格上，**模拟与闭式逐位一致**。
  ★★ **守卫**（`D.4`/`D.5`）：把模拟器的**填批**逻辑改坏（每步只填 `B−1`）⇒ 对不上；
    把闭式的**间隙项**去掉 ⇒ 对不上。**两端都过 `CONTRIBUTING:235`**。

★★★ **`D3`/`D4` 的实测**（`§0.7 D3`/`D4`）：
  · **收益来自【长度方差】**：等长 `1.00×` / 均匀 `1.65×` / 指数 `2.81×`（本机模拟器实测）；
  · ★★ **不来自【到达过程】**：泊松 `λ=8`、`λ=1.2`、以及**突发到达**均在 `1.00–1.03×`；
    ★ 突发**粒度 ≥ 批上限**时**恰为 `1.0000×`**（`C.3` 给出反例：粒度更小则会 >1，但**那仍是间隙**）；
  · ★★★ **加速比必须连基线写**：逐条处理（`B=1`）基线 `27.81×` vs 静态批基线 `1.65×`
    ⇒ ★★ **同一件事的两个加速比差 `17` 倍**（`B.4`）。

★★ **覆盖边界**：本程序守的是「**调度的算术与不变量**」（makespan / 闭式 / 下界 / 间隙）。
  **不守**任何**真实引擎**的调度器行为（本机**无 GPU / 无引擎**，也**无 vLLM / ORCA**）。
  ★★ **最强的限定语**：**「不是任何真实引擎，也不是真机」**。

★★ 可证伪断言（把下面任一条改错，本程序必须非零退出）：
  · `A.1` ★★★ **间隙闭式与模拟逐位一致**（`(N, B)` 网格，等长 + 同时到达）；
  · `A.2` ★★★ **免费守卫两端**：`B | N ⇒ 间隙 0`；`N = 1 ⇒ 间隙 (B−1)L/B`；
  · `A.3` ★★ **闭式 == `⌈N/B⌉·L`**（另一条独立恒等式）；
  · `B.1` ★★★ **下界**：任何调度器的 makespan `≥ max(Σlen/B, 最后到达 + 其长度)`；
  · `B.2` ★★ **收益来自长度方差**（等长 `1.00×` 与两个有方差的档拉开）；
  · `B.3` ★★ **到达过程几乎不起作用**（泊松 / 突发均在 `1.00–1.03×`）；
  · `B.4` ★★★ **两个基线的加速比差 17 倍**（不写基线则「提升 N 倍」无信息量）；
  · `C.1` ★★★ **`P38` 现场**：静态批**忽略到达时间** ⇒ **到达过程从未被激活**（消融）；
  · `C.2/C.3/C.4` 边界（一批装完 / 突发粒度 / 空输入）；
  · `D.4`/`D.5` ★★★ **守卫实验**：改坏模拟器 / 改坏闭式 ⇒ 断言必须失败。
"""
import ast
import collections
import inspect
import math
import sys
import textwrap

import numpy as np

import specs

# ===========================================================================
# 0. 断言记账（每个编号程序自带一份，**不共享**）
# ===========================================================================
# ★★★ 失败标记：本主题（batching-and-scheduling）取 `-- FAIL: batching-and-scheduling`；
#   `mutate.py` 用 AST 读出（不写死）。
ASSERT_FAIL_MARK = "-- FAIL: batching-and-scheduling"

OTHER_THEME_MARKS = ("-- FAIL: kv-cache", "-- FAIL: decoding-strategies",
                     "-- FAIL: engines", "-- FAIL: serving-infra")

FAILS = []
MARKS_USED = set()
_TOTAL = collections.Counter()
_PASS = collections.Counter()

TIME_EPS = 1e-9                      # ★ 浮点比较容差（本程序的两条路径在幂次分母上**逐位精确**）

# ★★★ **本机实测值**（本程序在本机跑出的数）—— 正文里的每个比值都在这里**钉死**。
#   顺序 = (等长, 均匀, 指数, 逐条基线, 泊松 λ=8, 泊松 λ=1.2, 突发(S≥B), 突发(S<B))。
PINNED_RATIOS = (1.0, 1.646, 2.8135, 27.805, 1.0039, 1.031, 1.0, 1.0229)
PINNED_GAP_N1 = 96.875               # ★ `N=1, B=32` 的间隙 = (32−1)×100/32
PINNED_STATIC_MAKESPAN = 700.0       # ★ 等长 + 同时到达，静态批的 makespan
PINNED_LOWER_BOUND = 625.0           # ★ 同例的下界 max(Σlen/B, 最后到达 + 其长度)
PINNED_SEQ_MAKESPAN = 20000.0        # ★ 同例**逐条处理**（B=1）的 makespan = N·L


def check(name, cond, detail="", detail_fail=None):
    """`name` 形如 `B.1` —— **首字母就是类别**（A/B/C/D）。★ 失败路径**不复述结论**。"""
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
# 1. 被测：**逐 token 步推进**的 continuous batching 模拟器
# ===========================================================================
def sim_cont(arrivals, lens, cap, dt=1.0, corrupt=None):
    """★★★ **被测实现**：**事件循环**，每个时间步让**每个在飞请求**各产出 1 个 token。

    ★ 与闭式 `makespan_closed` **无共享代码**（`D.8` 在源码层核）：
      这里是**一个 while 循环 + 一个在飞表**，那里是**一个式子**。
    ★★ 排程规则（**本主题的模型假设**）：
      · 每步开头，**在飞数 `< cap`** 且**已到达**（`arrival ≤ t`）的请求按**到达序**填入；
      · 空转（在飞为空）时，时间**跳到下一条请求的到达**；
      · 每个在飞请求**每步恰好推进 1 个 token** ⇒ 长度为 `L` 的请求耗 `L` 步。
    ★★ `corrupt`（**只用于守卫实验**）：
      · `"cap_minus1"` —— 每步只填 `cap − 1`（**留一个空位**，模拟「填批」逻辑被改坏）；
      · `"admit_one"`  —— 每步只填 `1`（退化成逐条处理）。
    """
    n = len(lens)
    if n == 0:
        return 0.0
    order = sorted(range(n), key=lambda i: (arrivals[i], i))
    if corrupt == "cap_minus1":
        eff = cap - 1
    elif corrupt == "admit_one":
        eff = 1
    else:
        eff = cap
    eff = max(1, eff)                          # ★ 防死循环：容量至少 1
    ptr = 0
    inflight = []                              # 每项 = [请求号, 剩余 token]
    t = 0.0
    done = 0
    while done < n:
        while len(inflight) < eff and ptr < n and arrivals[order[ptr]] <= t:
            i = order[ptr]
            ptr += 1
            inflight.append([i, lens[i]])
        if not inflight:
            t = arrivals[order[ptr]]           # ★ 空转 ⇒ 跳到下一条到达
            continue
        t += dt
        keep = []
        for req in inflight:
            req[1] -= 1
            if req[1] <= 0:
                done += 1
            else:
                keep.append(req)
        inflight = keep
    return t


# ===========================================================================
# 2. 参照：**闭式**（两条路径无共享代码）
# ===========================================================================
def makespan_closed(n, L, B, corrupt=None):
    """★★★ **参照实现**：**闭式** `makespan = N·L/B + 间隙`，`间隙 = (B − N mod B)·L/B`。

    ★★ 它只对**等长（`L`）+ 同时到达**成立 —— ★★ 这正是它的**适用域**，
      离开它就要用模拟器（`B.2`/`B.3` 的长度不等档就是这么算的）。
    ★★ `corrupt="no_gap"`（**只用于守卫实验**）：把**间隙项去掉**。
    """
    r = n % B
    gap = 0.0 if r == 0 else (B - r) * L / B
    if corrupt == "no_gap":
        gap = 0.0
    return n * L / B + gap


def gap_closed(n, L, B):
    """间隙项单独取出来（`A.2` 的两端守卫直接断言它）。"""
    r = n % B
    return 0.0 if r == 0 else (B - r) * L / B


# ===========================================================================
# 3. 基线：**静态批**（两组：尊重到达时间 / 忽略到达时间）
# ===========================================================================
def sim_static(arrivals, lens, cap, ignore_arrival=False):
    """**静态批**基线：按**到达序**每 `cap` 条一组，**整组等最长**（`max` 服务时间）。

    ★★ `ignore_arrival=True` 是**那个被推翻的版本**（`P38` 的现场）：
      它只按长度切批，**不看到达时间** ⇒ ★★ 「到达过程」在这个实现里**从未被激活**。
    ★★ 尊重到达时间的版本：一组的开始 = `max(上一组结束, 组内最后一条到达)`。
    """
    n = len(lens)
    if n == 0:
        return 0.0
    order = sorted(range(n), key=lambda i: (arrivals[i], i))
    end = 0.0
    for j in range(0, n, cap):
        grp = order[j:j + cap]
        last_arr = 0.0 if ignore_arrival else max(arrivals[i] for i in grp)
        end = max(end, last_arr) + max(lens[i] for i in grp)
    return end


# ===========================================================================
# 4. 负载与到达过程的生成（**钉死种子** ⇒ 结论可复现）
# ===========================================================================
def gen_lens(kind, n, rng):
    """长度分布：`equal` / `uniform`（`SCHED_LEN_LO..HI`）/ `exp`（均值 = 均匀的均值）。"""
    if kind == "equal":
        return [specs.SCHED_EQUAL_LEN] * n
    if kind == "uniform":
        return [int(x) for x in rng.integers(specs.SCHED_LEN_LO, specs.SCHED_LEN_HI + 1, n)]
    if kind == "exp":
        return [int(round(x)) for x in rng.exponential(float(specs.SCHED_EXP_MEAN), n)]
    raise ValueError("未知长度分布 %r" % (kind,))


def gen_arrivals(kind, n, rng, lam=None):
    """到达过程：`simul`（同时）/ `poisson`（速率 `lam`）/ 突发（`S` 条每 `P` 时间单位）。

    ★ 两种突发：`burst_sharp`（`S ≥ 批上限`，一次突发就填满一批）与
      `burst_small`（`S < 批上限`，静态批要等下一批突发）—— ★ `C.3` 用它们区分两种 `>1`。
    """
    if kind == "simul":
        return [0.0] * n
    if kind == "poisson":
        return [float(x) for x in np.cumsum(rng.exponential(1.0 / lam, n))]
    if kind in ("burst", "burst_sharp"):
        S, P = specs.SCHED_BURST_SIZE, specs.SCHED_BURST_PERIOD
    elif kind == "burst_small":
        S, P = specs.SCHED_BURST_SIZE // 2, specs.SCHED_BURST_PERIOD // 2
    else:
        raise ValueError("未知到达过程 %r" % (kind,))
    return [float((i // S) * P) for i in range(n)]


def ratio_static_over_cont(lenkind, arrkind, lam=None):
    """返回 `(static/cont, seq/cont, static, cont, seq)` —— **五元组**，全部由模拟器算出。"""
    rng = np.random.default_rng(specs.SCHED_SEED)
    lens = gen_lens(lenkind, specs.SCHED_N_REQ, rng)
    arr = gen_arrivals(arrkind, specs.SCHED_N_REQ, rng, lam)
    c = sim_cont(arr, lens, specs.SCHED_BATCH_CAP)
    s = sim_static(arr, lens, specs.SCHED_BATCH_CAP)
    seq = float(sum(lens))
    return s / c, seq / c, s, c, seq


# ===========================================================================
# 5. A 类 —— ★★★ **间隙的闭式**（本程序最强的一条）
# ===========================================================================
def part_A():
    print("=" * 78)
    print("A · **间隙的闭式**：被测 = 逐 token 步推进的模拟器；参照 = 闭式（两条独立路径）")
    print("=" * 78)
    print("  ★★★ 场景（闭式的**适用域**）：**等长 `L` + 同时到达 + 批上限 `B`**。")
    print("  ★★ 闭式：`间隙 = (B − N mod B)·L/B`（`N mod B ≠ 0`；否则 0），`makespan = N·L/B + 间隙`。")
    print("  ★★ 本程序的两条路径：`sim_cont` 是**事件循环**，`makespan_closed` 是**一个式子**。")

    L = specs.SCHED_EQUAL_LEN
    n_grid = specs.SCHED_N_GRID
    b_grid = specs.SCHED_B_GRID

    # ── A.1 ★★★ 网格上逐位一致 ───────────────────────────────────────────
    print()
    print("  ── A.1 ★★★ **`(N, B)` 网格上，模拟与闭式逐位一致** ──")
    print("     %-6s %-6s %-14s %-14s %-12s %s"
          % ("N", "B", "模拟 makespan", "闭式 makespan", "间隙", "一致"))
    all_match = True
    rows = []
    for N in n_grid:
        for B in b_grid:
            sim = sim_cont([0.0] * N, [L] * N, B)
            closed = makespan_closed(N, L, B)
            g = gap_closed(N, L, B)
            ok = abs(sim - closed) < TIME_EPS
            all_match = all_match and ok
            rows.append((N, B, sim, closed, g, ok))
            print("     %-6d %-6d %-14.4f %-14.4f %-12.4f %s"
                  % (N, B, sim, closed, g, "✓" if ok else "✗"))
    n_pairs = len(n_grid) * len(b_grid)
    print("     ★★ 共 **%d 档**（`N ∈ %s` × `B ∈ %s`），全部**逐位吻合** = %s"
          % (n_pairs, n_grid, b_grid, all_match))
    check("A.1", all_match and n_pairs >= 9,
          "★★★ **判据成立**：**%d 档** `(N, B)` 上，**逐 token 步推进的模拟器**与**闭式**"
          "**逐位一致**（容差 `%g`）—— ★★ 一条是事件循环、一条是一个式子，**两条独立路径会合**"
          % (n_pairs, TIME_EPS),
          "**模拟与闭式对不上** —— ★ 先查「每步填批」的规则（`len(inflight) < cap`）与请求的到达判定")

    # ── A.2 ★★★ 免费守卫（两端都成立）────────────────────────────────────
    print()
    print("  ── A.2 ★★★ **免费守卫两端**：`B | N ⇒ 间隙 0`；`N = 1 ⇒ 间隙 (B−1)L/B` ──")
    B0 = specs.SCHED_GUARD_CAP                       # = 32
    gap_one = gap_closed(1, L, B0)
    # ★ `B | N` 的三种情形（网格自身也含它们，这里显式再钉一次）
    div_cases = [(B0, B0), (2 * B0, B0), (B0 // 8 * B0 // 8, B0 // 8)]
    div_gaps = [gap_closed(n, L, b) for (n, b) in div_cases]
    print("     `N = 1, B = %d` ⇒ 间隙 = `(B−1)L/B` = **%.4f**（= (%d−1)×%d/%d）"
          % (B0, gap_one, B0, L, B0))
    print("     `B | N` 的三种情形 %s ⇒ 间隙 = %s（**恒为 0**）"
          % ([(n, b) for (n, b) in div_cases], ["%.4f" % g for g in div_gaps]))
    check("A.2", abs(gap_one - (B0 - 1) * L / B0) < TIME_EPS
          and abs(gap_one - PINNED_GAP_N1) < 1e-3
          and all(g == 0.0 for g in div_gaps),
          "★★★ **两端守卫都成立**：`N=1, B=%d` ⇒ 间隙 **%.4f**（★ 一批只装一个，"
          "「最后一批没填满」的极端）；`B | N`（三种：`%s`）⇒ 间隙**恒为 0**（一个不剩）"
          % (B0, gap_one, [(n, b) for (n, b) in div_cases]),
          "**间隙的两端守卫被破坏** —— ★ 先查 `N mod B` 的判定与 `B|N` 时的分支")

    # ── A.3 ★★ 闭式 == ceil(N/B)·L（另一条独立恒等式）────────────────────
    print()
    print("  ── A.3 ★★ **闭式 == `⌈N/B⌉·L`**（另一条独立恒等式：批数 × 每批时长）──")
    id_ok = True
    for (N, B, sim, closed, g, ok) in rows:
        want = math.ceil(N / B) * L
        id_ok = id_ok and abs(closed - want) < TIME_EPS
    print("     ★ `⌈N/B⌉ · L` 与闭式在全部 **%d 档**上相等 = %s" % (n_pairs, id_ok))
    print("     ★★ **读法**：闭式**不是**新假设 —— 它把「批数 × 每批时长」重写成了"
          "「总量 + 最后一批没填满的那部分」⇒ ★★ **间隙就是「最后一批没填满」**。")
    check("A.3", id_ok,
          "★★ **判据成立**：闭式 `N·L/B + 间隙` **恒等于** `⌈N/B⌉·L` —— ★★ 这条恒等式"
          "给出间隙的**机理**：它**恰好**是「最后一批没填满」的那部分",
          "**闭式与「批数 × 每批时长」不等** —— ★ 先查间隙项的符号与 `N mod B == 0` 的分支")


# ===========================================================================
# 6. B 类：下界 · 收益来源 · 基线
# ===========================================================================
def part_B():
    print()
    print("=" * 78)
    print("B · **代价模型**：下界 · 收益来源（长度方差 / 到达过程）· **基线**")
    print("=" * 78)

    L = specs.SCHED_EQUAL_LEN

    # ── B.1 ★★★ 下界 ─────────────────────────────────────────────────────
    print()
    print("  ── B.1 ★★★ **下界**：任何调度器的 makespan `≥ max(Σlen/B, 最后到达 + 其长度)` ──")
    print("     `%-12s %-12s %-16s %-16s %s"
          % ("负载", "调度器", "makespan", "下界", "≥ 下界"))
    rng = np.random.default_rng(specs.SCHED_SEED)
    lb_rows = []
    for lk in ("equal", "uniform"):
        lens = gen_lens(lk, specs.SCHED_N_REQ, rng)
        arr = [0.0] * specs.SCHED_N_REQ
        lb = max(sum(lens) / specs.SCHED_BATCH_CAP,
                 max(arr[i] + lens[i] for i in range(len(lens))))
        for name, val in (("连续批", sim_cont(arr, lens, specs.SCHED_BATCH_CAP)),
                          ("静态批", sim_static(arr, lens, specs.SCHED_BATCH_CAP))):
            lb_rows.append((lk, name, val, lb, val >= lb - TIME_EPS))
            print("     %-12s %-12s %-16.2f %-16.2f %s"
                  % (lk, name, val, lb, "✓" if val >= lb - TIME_EPS else "✗"))
    eq_static = [r for r in lb_rows if r[0] == "equal" and r[1] == "静态批"][0]
    # ★ 逐条处理（`B=1`）这一基线的等长档：`= N·L`（决定 §0 那张表的第一/第二列）
    seq_eq = sim_cont([0.0] * specs.SCHED_N_REQ, [L] * specs.SCHED_N_REQ, 1)
    print("     ★★ 等长 + 同时到达这一档：`Σlen/B` = `%d×%d/%d` = **%.1f**（下界）；"
          "静态批 = **%.1f**（★ 与闭式同值）；**逐条处理** = **%.1f**（= `N·L`）"
          % (specs.SCHED_N_REQ, L, specs.SCHED_BATCH_CAP, eq_static[3], eq_static[2], seq_eq))
    check("B.1", all(r[4] for r in lb_rows)
          and abs(eq_static[3] - PINNED_LOWER_BOUND) < TIME_EPS
          and abs(eq_static[2] - PINNED_STATIC_MAKESPAN) < TIME_EPS
          and abs(seq_eq - PINNED_SEQ_MAKESPAN) < TIME_EPS,
          "★★★ **判据成立**：**两个调度器 × 两种负载**的 makespan **全部 ≥ 下界**；"
          "★ 等长档的下界 = **%.1f**、静态批 = **%.1f**（`B=1` 时才达到下界）"
          % (PINNED_LOWER_BOUND, PINNED_STATIC_MAKESPAN),
          "**有调度器跑到了下界以下** —— ★ 下界或模拟器被动过")

    # ── B.2 ★★ 收益来自长度方差 ─────────────────────────────────────────
    print()
    print("  ── B.2 ★★ **收益来自【长度方差】**（同时到达，只改长度分布）──")
    print("     ★ `static/cont` = 静态批的 makespan ÷ 连续批的 makespan（>1 = 连续批更快）")
    print("     %-10s %-14s %-14s %-12s %s"
          % ("长度分布", "static/cont", "逐条/连续", "静态批", "连续批"))
    var_rows = []
    for lk in ("equal", "uniform", "exp"):
        st, sq, s, c, seq = ratio_static_over_cont(lk, "simul")
        var_rows.append((lk, st, sq, s, c, seq))
        print("     %-10s %-14.4f %-14.4f %-12.1f %.1f" % (lk, st, sq, s, c))
    r_eq, r_uni, r_exp = var_rows[0][1], var_rows[1][1], var_rows[2][1]
    print("     ★★ **读法**：等长 ⇒ **%.4f**（★ 连续批与静态批**没有差别**）；"
          "有方差 ⇒ **%.4f**；重尾 ⇒ **%.4f** —— **方差越大收益越大**"
          % (r_eq, r_uni, r_exp))
    check("B.2", abs(r_eq - PINNED_RATIOS[0]) < 2e-3
          and abs(r_uni - PINNED_RATIOS[1]) < 2e-3
          and abs(r_exp - PINNED_RATIOS[2]) < 2e-3
          and r_eq < r_uni < r_exp,
          "★★ **判据成立**：**等长 `%.4f` → 均匀 `%.4f` → 指数 `%.4f`**（**单调递增**）——"
          " ★★★ **收益的【主因】是长度方差**：等长时两者**没有差别**，方差越大差距越大"
          % (r_eq, r_uni, r_exp),
          "**长度方差的单调性被破坏**（或某个钉住值对不上）—— ★ 先查长度分布与种子")

    # ── B.3 ★★ 到达过程几乎不起作用 ────────────────────────────────────
    print()
    print("  ── B.3 ★★ **到达过程几乎不起作用**（长度**等长**，只改到达过程）──")
    print("     %-16s %-14s %-12s %s" % ("到达过程", "static/cont", "静态批", "连续批"))
    arr_rows = []
    for label, kind, lam in (("同时到达", "simul", None),
                             ("泊松 λ=%g" % specs.SCHED_POISSON_LAMBDA_FAST, "poisson",
                              specs.SCHED_POISSON_LAMBDA_FAST),
                             ("泊松 λ=%g" % specs.SCHED_POISSON_LAMBDA_SLOW, "poisson",
                              specs.SCHED_POISSON_LAMBDA_SLOW),
                             ("突发 S≥B", "burst_sharp", None),
                             ("突发 S<B", "burst_small", None)):
        st, sq, s, c, seq = ratio_static_over_cont("equal", kind, lam)
        arr_rows.append((label, st, s, c))
        print("     %-16s %-14.4f %-12.1f %.1f" % (label, st, s, c))
    arr_vals = [r[1] for r in arr_rows]
    # ★ 处理速率（`B/L`）：`λ=8` 是它的 25 倍 ⇒ 「远快于处理」（正文这一句被这句断言钉住）
    proc_rate = specs.SCHED_BATCH_CAP / specs.SCHED_EQUAL_LEN
    print("     ★ 处理速率 ≈ `B/L` = `%d/%d` = **%.2f** / 时间单位 ⇒ `λ=8` 是它的 **%.0f 倍**"
          % (specs.SCHED_BATCH_CAP, specs.SCHED_EQUAL_LEN, proc_rate,
             specs.SCHED_POISSON_LAMBDA_FAST / proc_rate))
    print("     ★★ **读法**：泊松两种速率、突发（粒度 ≥ 批上限）都在 **%.4f–%.4f** ——"
          " ★★ 与长度方差那三个档（%.2f–%.2f）**不在一个量级**"
          % (min(arr_vals[:4]), max(arr_vals[:4]),
             PINNED_RATIOS[1], PINNED_RATIOS[2]))
    print("        ★ 突发**粒度 < 批上限**（最后一行）确实 **> 1** ——")
    print("           ★★ 但**那仍是「等批次填满」这个间隙**，不是「到达过程」本身的新收益（见 `C.3`）。")
    check("B.3", abs(arr_rows[0][1] - PINNED_RATIOS[0]) < 2e-3
          and abs(arr_rows[1][1] - PINNED_RATIOS[4]) < 2e-3
          and abs(arr_rows[2][1] - PINNED_RATIOS[5]) < 2e-3
          and abs(arr_rows[3][1] - PINNED_RATIOS[6]) < 2e-3
          and abs(arr_rows[4][1] - PINNED_RATIOS[7]) < 2e-3
          and max(arr_vals[:4]) < 1.05 and abs(arr_rows[3][1] - 1.0) < TIME_EPS
          and abs(proc_rate - 0.32) < 1e-9,
          "★★ **判据成立**：**同时到达 / 泊松 λ=%g / 泊松 λ=%g / 突发(S≥B)** 四种到达过程下，"
          "收益分别是 `%.4f / %.4f / %.4f / %.4f` —— ★★ **全部 ≈ `1.00×`**"
          "（★ 与长度方差的 `%.2f–%.2f` 对比）⇒ ★★★ **收益不来自到达过程**"
          % (specs.SCHED_POISSON_LAMBDA_FAST, specs.SCHED_POISSON_LAMBDA_SLOW,
             arr_vals[0], arr_vals[1], arr_vals[2], arr_vals[3],
             PINNED_RATIOS[1], PINNED_RATIOS[2]),
          "**到达过程的影响超出了预期量级** —— ★ 先查到达过程的生成（尤其突发粒度与批上限的关系）")

    # ── B.4 ★★★ 两个基线的加速比差 17 倍 ───────────────────────────────
    print()
    print("  ── B.4 ★★★ **同一个 continuous batching，两个加速比**（`P9` 的现场）──")
    seq_speed = var_rows[1][2]           # 逐条/连续（均匀档）
    stat_speed = var_rows[1][1]          # 静态批/连续（均匀档）
    print("     均匀 `%d–%d`、`N=%d`、`B=%d` 同一组请求："
          % (specs.SCHED_LEN_LO, specs.SCHED_LEN_HI, specs.SCHED_N_REQ, specs.SCHED_BATCH_CAP))
    print("       · 基线 = **逐条处理**（`B=1`）⇒ 连续批 **%.2f×**" % seq_speed)
    print("       · 基线 = **静态批**（`B=%d`）⇒ 连续批 **%.2f×**" % (specs.SCHED_BATCH_CAP, stat_speed))
    print("     ★★★ **读法：不写基线，「提升 N 倍」没有信息量** —— 两个数差 **%.1f 倍**"
          % (seq_speed / stat_speed))
    print("        ★★ 文献里的两个数同样必须**连出处与基线一起写**：")
    print("           ORCA（OSDI '22）报 **36.9×**（对比 FasterTransformer），"
          "vLLM 首页引的一篇**博客**报 **23×** ——")
    print("           ★★★ **它们是引用值，基线是各自论文/博客的条件，`[C]`/`[D]` 级，"
          "与本机模拟器无关**（`F4`）。")
    check("B.4", abs(seq_speed - PINNED_RATIOS[3]) < 2e-3
          and abs(stat_speed - PINNED_RATIOS[1]) < 2e-3
          and seq_speed / stat_speed > 10.0,
          "★★★ **判据成立**：**同一组请求**，对**逐条基线**是 **%.2f×**、对**静态批基线**是 "
          "**%.2f×** —— ★★ **差 %.1f 倍**（★ 所以「加速比」必须写清分母）"
          % (seq_speed, stat_speed, seq_speed / stat_speed),
          "**两个基线没有拉开差距** —— ★ 那说明某条基线算错了（先查 `B=1` 的逐条处理）")


# ===========================================================================
# 7. C 类：**`P38` 现场**（消融）· 边界与反例
# ===========================================================================
def part_C():
    print()
    print("=" * 78)
    print("C · 边界与反例：**被测量的那个因素有没有真的进入路径**（`P38`）")
    print("=" * 78)

    # ── C.1 ★★★ P38 现场：到达过程有没有被激活 ─────────────────────────
    print()
    print("  ── C.1 ★★★ **`P38` 现场**：静态批**忽略到达时间**时，到达过程**从未被激活** ──")
    rng = np.random.default_rng(specs.SCHED_SEED)
    lens = gen_lens("equal", specs.SCHED_N_REQ, rng)
    a_sim = gen_arrivals("simul", specs.SCHED_N_REQ, np.random.default_rng(specs.SCHED_SEED))
    a_bur = gen_arrivals("burst_small", specs.SCHED_N_REQ, np.random.default_rng(specs.SCHED_SEED))
    a_poi = gen_arrivals("poisson", specs.SCHED_N_REQ, np.random.default_rng(specs.SCHED_SEED),
                         specs.SCHED_POISSON_LAMBDA_SLOW)
    bad = [sim_static(a, lens, specs.SCHED_BATCH_CAP, ignore_arrival=True)
           for a in (a_sim, a_bur, a_poi)]
    good = [sim_static(a, lens, specs.SCHED_BATCH_CAP, ignore_arrival=False)
            for a in (a_sim, a_bur, a_poi)]
    print("     ★ **忽略到达时间**的静态批：三种到达过程 ⇒ makespan = %s" % ["%.2f" % x for x in bad])
    print("     ★ **尊重到达时间**的静态批：同一三种到达过程 ⇒ makespan = %s" % ["%.2f" % x for x in good])
    print("     ★★★ **读法**：「忽略到达」的实现里，**三种到达过程给出完全相同的输出** ⇒")
    print("         ★★ **在那个实验设计里「到达过程」根本没有进入路径** ——")
    print("         而**据此写出的关于到达过程的结论**是无效的（`P38` / `P12` 的实验设计层形态）。")
    print("     ★★ **消融判据**：把该因素改到极端，**结论必须变** —— 尊重到达的版本确实**变了**。")
    check("C.1", bad[0] == bad[1] == bad[2]
          and not (good[0] == good[1] == good[2]),
          "★★★ **判据成立**：**忽略到达**的版本对三种到达过程给出**同一个数**（`%s`，"
          "★ 因素**未被激活**）；**尊重到达**的版本给出**不同的数**（`%s`）——"
          " ★★★ 这就是「消融」：**把声称的原因关掉，看结论是否真的变**"
          % (["%.2f" % x for x in bad], ["%.2f" % x for x in good]),
          "**消融没有跑出差别** —— ★ 检查 `ignore_arrival` 分支是否真的绕过了到达时间")

    # ── C.2 ★★ 边界：一批装完 / 空输入 ─────────────────────────────────
    print()
    print("  ── C.2 ★★ **边界**：`B ≥ N ⇒ 一批装完`；**空输入 ⇒ 0** ──")
    L = specs.SCHED_EQUAL_LEN
    N = 5
    one = sim_cont([0.0] * N, [L] * N, N + 3)
    empty = sim_cont([], [], specs.SCHED_BATCH_CAP)
    print("     `N=%d, B=N+3` ⇒ makespan = **%.1f**（= `L` = %d：**一批装完**）" % (N, one, L))
    print("     `N=0` ⇒ makespan = **%.1f**（空输入）" % empty)
    check("C.2", abs(one - L) < TIME_EPS and empty == 0.0,
          "★★ **判据成立**：`B ≥ N` 时**一批装完** ⇒ `makespan == L`（★ 间隙项为 0 的两个来源之一）；"
          "空输入 ⇒ `0.0`（★ 两个边界都不触发异常）",
          "**边界值被动过** —— ★ 先查 `B ≥ N` 与空输入分支")

    # ── C.3 ★★ 突发粒度的反例：`>1` 仍是间隙，不是新收益 ────────────────
    print()
    print("  ── C.3 ★★ **突发粒度 < 批上限** 时的 `>1` 从哪来（★ `B.3` 最后一行的解释）──")
    print("     `%-20s %-14s %-14s %s" % ("突发 (S, P)", "static/cont", "静态批", "连续批"))
    b_rows = []
    for S, P in ((specs.SCHED_BATCH_CAP, specs.SCHED_BATCH_CAP),
                 (specs.SCHED_BATCH_CAP // 2, specs.SCHED_BATCH_CAP // 2)):
        r = np.random.default_rng(specs.SCHED_SEED)
        lens = gen_lens("equal", specs.SCHED_N_REQ, r)
        arr = [float((i // S) * P) for i in range(specs.SCHED_N_REQ)]
        s = sim_static(arr, lens, specs.SCHED_BATCH_CAP)
        c = sim_cont(arr, lens, specs.SCHED_BATCH_CAP)
        b_rows.append((S, P, s / c, s, c))
        print("     `S=%-3d P=%-3d`        %-14.4f %-14.1f %.1f" % (S, P, s / c, s, c))
    print("     ★★ **读法**：`S ≥ B`（一次突发就填满一批）⇒ **恰好 `1.0000×`**；")
    print("        `S < B` ⇒ 静态批**要等下一批突发**才凑满 ⇒ `>1` ——")
    print("        ★★ **但那还是「等批次填满」这一个间隙**，只是被到达节奏放大了；")
    print("        ★ 它**不随「到达过程的方向」单调**，与长度方差的机理**不是同一件事**。")
    check("C.3", abs(b_rows[0][2] - 1.0) < TIME_EPS and b_rows[1][2] > 1.0,
          "★★ **判据成立**：突发粒度 `S ≥ B` ⇒ **%.4f**（★ 批次总是满的，间隙不出现）；"
          "`S < B` ⇒ **%.4f**（★ 静态批等填批）—— 两者都归**间隙**解释"
          % (b_rows[0][2], b_rows[1][2]),
          "**突发粒度的对比没有跑出来** —— ★ 先查突发到达的生成与批上限的关系")

    # ── C.4 ★★ 空转：在飞为空时时间跳到下一条到达 ──────────────────────
    print()
    print("  ── C.4 ★★ **空转**：请求稀疏到处理器会空闲时，makespan 由**到达**决定 ──")
    sparse = [0.0] + [1000.0 + i * 1000.0 for i in range(3)]
    slen = [10, 10, 10, 10]
    m = sim_cont(sparse, slen, specs.SCHED_BATCH_CAP)
    lb = max(sum(slen) / specs.SCHED_BATCH_CAP, max(sparse[i] + slen[i] for i in range(4)))
    print("     到达 = %s，长度 = %s ⇒ makespan = **%.1f**；下界 = **%.1f**"
          % (sparse, slen, m, lb))
    check("C.4", abs(m - lb) < TIME_EPS,
          "★★ **判据成立**：到达极稀疏时，makespan **恰好等于下界** `最后到达 + 其长度` "
          "（**处理器空转** ⇒ 下界的第二项起决定作用）—— ★ 与 `B.1` 的第一项互补",
          "**稀疏导出的 makespan 不是「最后到达 + 其长度」** —— ★ 先查空转分支")


# ===========================================================================
# 8. 前哨：口径 / 载体 / 标记自检（★ 跑在任何计算之前）
# ===========================================================================
def preflight():
    """★★ 口径 / 载体 / 标记自检**一律在这里**，且**跑在任何计算之前**（本库 `P23`）。"""
    print("=" * 78)
    print("D0. 口径/载体/标记钉住（**先跑**：让「改常量」变成断言捕获，而不是崩溃捕获）")
    print("=" * 78)
    print("  本程序的失败标记 ASSERT_FAIL_MARK = %r" % ASSERT_FAIL_MARK)
    print("  ★★★ 限定语：**不是任何真实引擎，也不是真机** —— 本机**无 GPU / 无 vLLM / 无 ORCA**；")
    print("      ★★ 但**本程序的 A 类是真的**：**间隙闭式与模拟器是两条独立路径**。")
    before = len(FAILS)

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

    check("D.2", ASSERT_FAIL_MARK == "-- FAIL: batching-and-scheduling" and "FAIL" in ASSERT_FAIL_MARK,
          "★ 标记自检：**等于**本主题被分配的字面量 `-- FAIL: batching-and-scheduling`、且含 `FAIL`")
    check("D.2b", OTHER_THEME_MARKS == ("-- FAIL: kv-cache", "-- FAIL: decoding-strategies",
                                        "-- FAIL: engines", "-- FAIL: serving-infra"),
          "★ 「不跨主题照抄」的**对照物**（本域另外四个主题的标记）逐位钉住")

    check("D.3", specs.SCHED_FETCH_DAY == "2026-10-09" and specs.UNSOURCED == ()
          and abs(TIME_EPS - 1e-9) < 1e-18,
          "★ **载体元数据逐字钉住**：抓取日 `%s`；★ `UNSOURCED == ()` 一并钉住；"
          "★ 容差 `TIME_EPS` **在 preflight 里就钉住**（★ 否则它的变异会变成**崩溃捕获**、"
          "而不是断言捕获）" % specs.SCHED_FETCH_DAY)

    # ★★★ 口径守卫（★ 跑在任何计算之前）：闭式的**适用域**是「等长 + 同时到达」；
    #   网格必须**含** `N = 1`（间隙的极端）与 `B | N`（间隙为 0）两类，否则 `A.2` 的两端
    #   守卫会退化成恒真。★ 且批上限必须 > 1（否则「填批」无从谈起）。
    ok_grid = (specs.SCHED_N_GRID[0] == 1
               and any(n % b == 0 for n in specs.SCHED_N_GRID for b in specs.SCHED_B_GRID)
               and any(n % b != 0 for n in specs.SCHED_N_GRID for b in specs.SCHED_B_GRID)
               and specs.SCHED_BATCH_CAP > 1
               and specs.SCHED_BATCH_CAP in specs.SCHED_B_GRID
               and specs.SCHED_EQUAL_LEN > 0)
    print("     ★★ 口径守卫：网格含 `N=1` = %s；含 `B|N` = %s；含 `B∤N` = %s；`B>1` = %s"
          % (specs.SCHED_N_GRID[0] == 1,
             any(n % b == 0 for n in specs.SCHED_N_GRID for b in specs.SCHED_B_GRID),
             any(n % b != 0 for n in specs.SCHED_N_GRID for b in specs.SCHED_B_GRID),
             specs.SCHED_BATCH_CAP > 1))
    check("D.3b", ok_grid,
          "★★★ **口径守卫**：网格**必须同时含** `B|N`（间隙 0）与 `B∤N`（间隙 > 0）两类，"
          "且含 `N=1` —— ★ 否则 `A.2` 的两端守卫会**退化成恒真**（那正是 Chunk 1 的 `H1` 形态）",
          "**网格的覆盖被动过** —— ★ 先查 `SCHED_N_GRID` / `SCHED_B_GRID` / `SCHED_EQUAL_LEN`")
    return len(FAILS) == before


# ===========================================================================
# 9. D 类（后半）：守卫实验 · 引文逐字 · 逐元素 · 独立性
# ===========================================================================
def _names_in(obj):
    src = textwrap.dedent(inspect.getsource(obj))
    names = set()
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.Name):
            names.add(n.id)
        elif isinstance(n, ast.Attribute):
            names.add(n.attr)
        elif isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            names.add(n.name)
    return names


def part_D():
    print()
    print("=" * 78)
    print("D · **守卫实验** · 引文逐字钉住 · **逐元素**钉住 · 独立性线索")
    print("=" * 78)

    L = specs.SCHED_EQUAL_LEN
    B0 = specs.SCHED_GUARD_CAP

    # ── D.4 ★★★ 守卫实验 ①：把模拟器的「填批」改坏 ──────────────────────
    print()
    print("  ── D.4 ★★★ **守卫实验 ①**：模拟器每步只填 `cap − 1`（**留一个空位**）──")
    g_sim = sim_cont([0.0] * 5, [L] * 5, 5)                 # 正常：5 条一步填满
    g_bad = sim_cont([0.0] * 5, [L] * 5, 5, corrupt="cap_minus1")   # 坏法：每步只填 4
    g_ref = makespan_closed(5, L, 5)
    print("     `N=5, B=5, L=%d`：正常 = **%.1f** · 坏法 = **%.1f** · 闭式 = **%.1f**"
          % (L, g_sim, g_bad, g_ref))
    print("     ★★★ **读法**：正常版**一步填满 5 条** ⇒ `makespan = L`；"
          "坏法每步只填 4 条 ⇒ 要多走一趟 ⇒ **比闭式大**。")
    check("D.4", abs(g_sim - g_ref) < TIME_EPS and g_bad > g_ref + TIME_EPS,
          "★★★ **守卫成立**：把「填批」逻辑改坏后，模拟器从 **%.1f**（= 闭式）变成 **%.1f** ——"
          " ★★ 于是 `A.1` 的「逐位一致」对「每步填满」这条规则**是敏感的**" % (g_sim, g_bad),
          "**守卫失效** —— ★ 改坏填批后模拟器居然仍等于闭式（那 `A.1` 抓不到它 ⇒ 需要加判据）")

    # ── D.5 ★★★ 守卫实验 ②：把闭式的【间隙项】去掉 ─────────────────────
    print()
    print("  ── D.5 ★★★ **守卫实验 ②**：把**闭式的间隙项去掉**（`no_gap`）──")
    h_ref = makespan_closed(specs.SCHED_N_REQ, L, B0)
    h_bad = makespan_closed(specs.SCHED_N_REQ, L, B0, corrupt="no_gap")
    h_sim = sim_cont([0.0] * specs.SCHED_N_REQ, [L] * specs.SCHED_N_REQ, B0)
    print("     `N=%d, B=%d`：闭式（含间隙）= **%.1f** · 闭式（去间隙）= **%.1f** · 模拟 = **%.1f**"
          % (specs.SCHED_N_REQ, B0, h_ref, h_bad, h_sim))
    check("D.5", abs(h_ref - h_sim) < TIME_EPS and abs(h_bad - h_sim) > TIME_EPS,
          "★★★ **守卫成立**：去掉间隙项后闭式给出 **%.1f**，而模拟器给出 **%.1f** ——"
          " ★★ **对不上**（差 `%.1f`，恰好是间隙 `%.1f`）⇒ `A.1` 对**参照的表达式**也敏感"
          % (h_bad, h_sim, h_sim - h_bad, gap_closed(specs.SCHED_N_REQ, L, B0)),
          "**守卫失效** —— ★ 去掉间隙后闭式仍与模拟一致（那 `A.1` 抓不到它）")

    # ── D.6 ★★★ 引文逐字：vLLM 的批参数默认值（**无单一默认值**）─────────
    check("D.6", specs.VLLM_MAX_NUM_SEQS_CLASS_DEFAULT == 128
          and specs.VLLM_MAX_NUM_BATCHED_TOKENS_CLASS_DEFAULT == 2048
          and specs.VLLM_MAX_NUM_BATCHED_TOKENS_FOR_BATCHED_DP == 256
          and specs.VLLM_API_SERVER_H100_MAX_TOKENS == 8192
          and specs.VLLM_API_SERVER_H100_MAX_SEQS == 1024
          and specs.VLLM_API_SERVER_DEFAULT_MAX_TOKENS == 2048
          and specs.VLLM_API_SERVER_DEFAULT_MAX_SEQS == 256
          and specs.VLLM_API_SERVER_B200_MAX_TOKENS == 16384
          and specs.VLLM_API_SERVER_B200_MAX_SEQS == 1024
          and specs.VLLM_TEST_CONVENIENCE_QUOTE == "The default value here is mainly for convenience when testing. In real usage, this should be set in EngineArgs.create_engine_config."
          and specs.VLLM_MAX_NUM_SEQS_NO_SINGLE_DEFAULT == "vLLM 的 max_num_seqs / max_num_batched_tokens 没有单一默认值：类常量 2048/128 只是兜底，get_batch_defaults() 按显存+场景覆盖，throughput 模式再各自翻倍",
          "★★★ **引文/取值逐字钉住**：类常量兜底 `128` / `2048`；★ **覆盖值**按设备+场景分档"
          "（H100 API server `8192`/`1024`；A100/其余 `2048`/`256`；B200 `16384`/`1024`）——"
          " ★★★ **「vLLM 的 `max_num_seqs` 默认值」没有单一答案**（`[B]` 级，见 `F2`）",
          "**批参数常量被动过** —— ★ 先查 vLLM 的兜底值与各档覆盖值")

    # ── D.7 ★★★ 逐元素钉住 ──────────────────────────────────────────────
    check("D.7", specs.SCHED_FETCH_DAY == "2026-10-09"
          and specs.SCHED_N_REQ == 200 and specs.SCHED_BATCH_CAP == 32
          and specs.SCHED_EQUAL_LEN == 100 and specs.SCHED_LEN_LO == 20
          and specs.SCHED_LEN_HI == 200 and specs.SCHED_EXP_MEAN == 110
          and specs.SCHED_SEED == 20261009 and specs.SCHED_GUARD_CAP == 32
          and specs.SCHED_BURST_SIZE == 32 and specs.SCHED_BURST_PERIOD == 32
          and specs.SCHED_N_GRID == (1, 7, 32, 100, 200)
          and specs.SCHED_B_GRID == (8, 16, 32, 64)
          and specs.SCHED_POISSON_LAMBDA_FAST == 8.0
          and specs.SCHED_POISSON_LAMBDA_SLOW == 1.2
          and specs.UNSOURCED == ()
          and abs(TIME_EPS - 1e-9) < 1e-18
          and PINNED_RATIOS == (1.0, 1.646, 2.8135, 27.805, 1.0039, 1.031, 1.0, 1.0229)
          and abs(PINNED_GAP_N1 - 96.875) < 1e-9
          and abs(PINNED_STATIC_MAKESPAN - 700.0) < 1e-9
          and abs(PINNED_LOWER_BOUND - 625.0) < 1e-9
          and abs(PINNED_SEQ_MAKESPAN - 20000.0) < 1e-9,
          "★★★ **规格常量逐元素 + 程序内常量**钉住（`N` · 批上限 · 等长 · 长度区间 · 种子 ·"
          " 突发参数 · 两个网格 · 两个 λ · 全 `PINNED_*`）；★★ `UNSOURCED == ()` 一并钉住",
          "**某个常量被动过**")

    # ── D.8 ★★★ 独立性（源码级）：两实现互不引用 ────────────────────────
    sn = _names_in(sim_cont)
    on = _names_in(makespan_closed)
    print()
    print("  ── D.8 ★★ **源码级线索**（★ **不是证明**）：被测与参照互不引用 ──")
    print("     `sim_cont` 里出现 `makespan_closed` = %s ；`makespan_closed` 里出现 `sim_cont` = %s"
          % ("makespan_closed" in sn, "sim_cont" in on))
    print("     ★★★ **降级措辞**：这只是「**两段代码不互相引用**」的**源码级线索**，"
          "**不证明**判据有信息量 —— 后者由 `A.1` 与 `D.4`/`D.5` 的**行为**兜底。")
    check("D.8", "makespan_closed" not in sn and "sim_cont" not in on
          and "gap_closed" not in sn and "sim_cont" not in _names_in(gap_closed),
          "★★ **源码级线索成立**：**被测（事件循环）与参照（闭式）互不引用**（★ **两侧都审**）——"
          " ★★ 于是「两条路径一致」**不是恒真**，而是**两条独立路径的会合**",
          "**两实现互相引用** —— A 类可能退化成自证（`§1.3.2b`）")

    # ── D.9 ★★ 边界：间隙的连续性 ───────────────────────────────────────
    B1 = specs.SCHED_GUARD_CAP
    gaps = [gap_closed(n, L, B1) for n in range(1, 2 * B1 + 1)]
    print()
    print("  ── D.9 ★★ **间隙随 `N` 的锯齿**（`B=%d`）：`B|N` 处**恰好 0**、`N≡1` 处**最大** ──" % B1)
    print("     `N=1..%d` 的间隙：%s" % (2 * B1, ["%.0f" % g for g in gaps[:8]] + ["..."]))
    check("D.9", gaps[0] == (B1 - 1) * L / B1
          and gaps[B1 - 1] == 0.0 and gaps[2 * B1 - 1] == 0.0
          and max(gaps) == (B1 - 1) * L / B1,
          "★★ **边界成立**：`B=%d` 时间隙在 `N ≡ 1 (mod B)` 处**最大**（`%.1f`）、"
          "在 `B | N` 处**恰好 0** —— ★★ 锯齿的**周期就是批上限**，★ 故间隙是**离散**的"
          % (B1, gaps[0]),
          "**间隙的锯齿形态被动过** —— ★ 先查 `N mod B` 与 `B|N` 分支")


# ===========================================================================
# 10. 汇总 / 主程序
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
    print("★★ **A = %d/%d** —— ★★★ **本程序有 A 类**：间隙闭式（**事件循环 vs 一个式子**）。"
          % (_PASS["A"], _TOTAL["A"]))
    print("★ 判据来源：本主题的调度模型是 **`[E]` 级（本库实现 / 假设）**；"
          "vLLM 的批参数是 `[B]` 级引用值。")
    print("★ 本程序只核对「**调度的算术与不变量**」——**不是**任何真实引擎的调度器。")
    return 1 if FAILS else 0


def main():
    print("=" * 78)
    print("01_iteration_level_scheduling.py —— 迭代级调度：continuous batching 的机制与收益来源")
    print("=" * 78)
    print("  ★ 失败标记 ASSERT_FAIL_MARK = %r（本主题自定义；不跨主题照抄）" % ASSERT_FAIL_MARK)
    print("  ★★★ 定义不在本文：`continuous batching` / `chunked prefill` 的定义在")
    print("      `10-foundations/performance-models/01-roofline-model.md:251`（**只回链**）。")
    print("  ★★★ 限定语：**不是任何真实引擎，也不是真机** —— 本机无 GPU / 无引擎。")
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

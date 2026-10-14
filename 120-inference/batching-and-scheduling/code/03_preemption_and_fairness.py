#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""03_preemption_and_fairness.py —— **抢占的代价账** 与 **公平**。

配套文档：`03-preemption-and-fairness.md`。

★ 三类断言**全部在本程序内部**，失败时以**非零退出码**结束；**没有 `verify.py`**。

★★★ **本程序有两条独立的 A 类**（都过 `CONTRIBUTING:235`）：
  ① **换出 vs KV 重算的代价账** —— 被测 = **账本**（逐块搬运 / 逐 token KV 重算的循环）；
     参照 = **闭式**（`2·s·bpt/BW` 与 `s·τ_lin + s²·τ_quad`）。★★ 两条路径无共享代码。
     ★★★ **临界长度** `s* = (2·bpt/BW − τ_lin)/τ_quad` 有**第二条独立路径**：
     **在账本上二分**找符号变号点（`A.2`）。
  ② **SJF 的最优性** —— 被测 = **SJF 贪心**；参照 = **`n!` 穷举**（`A.4`）。
     ★★ 穷举是**最强的一种参照**（不共享任何启发式）；★★ **目标函数也不共享**：
     穷举用独立的 `avg_completion_alt`、SJF 用 `avg_completion`（`D.8` 在源码层核）。

★★★ **术语纪律（`§2.2`）**：本域的 **`KV 重算`** 是**第三义**（抢占时丢掉 KV、需要时把它重跑一遍）——
  ★ 与**编译器**的「物化 vs 重算」、**训练**的「重算换显存」（gradient checkpointing）**同名不同指**。
  本文**一律写「KV 重算」全称**。

★★★ **模型假设（必须写明）**：
  · **换出** = 把这段 KV **搬出去再搬回来** ⇒ 代价 `2·s·bpt/BW`（★ **`×2` 是「出+入」**）；
  · **KV 重算** = 把这段 prefill **重跑一遍** ⇒ 代价 `s·τ_lin + s²·τ_quad`
    （★ 线性项 = 矩阵乘 / MLP、二次项 = **自注意力**；★ **二次项是长度临界点的唯一来源**）；
  · `BW` 是**单向、载荷口径**的搬运带宽（★ 三种口径的正文在 `20-gpu-architecture/interconnect/01`，
    **本主题只回链**）。

★★ **覆盖边界**：本程序守的是「**代价的算术与不变量**」（临界长度 / 临界带宽 / 策略总代价 / 公平指标）。
  **不守**任何**真实引擎**的抢占行为（本机**无 GPU / 无引擎**）。★★ **最弱的限定语**同全库：
  **「不是任何真实引擎，也不是真机」**。

★★ 可证伪断言（把下面任一条改错，本程序必须非零退出）：
  · `A.1` ★★★ **账本与闭式逐位一致**（换出 / KV 重算，长度网格上）；
  · `A.2` ★★★ **临界长度 `s*` 的两条独立路径**（闭式 vs 账本二分）+ 两侧方向；
  · `A.3` ★★ **临界带宽 `BW*`** 与方向（带宽越大换出越划算）；
  · `A.4` ★★★ **SJF 最优性**（`n!` 穷举作为参照）；
  · `B.1` ★★ **代价表 + 三种策略的总代价**（always-swap / always-recompute / 按临界点选）；
  · `B.2` ★★ **公平指标**（FCFS 与 SJF 的平均 / 最大等待）；
  · `C.1`–`C.4` 边界与反例（长度 0 / 带宽两端 / **忘掉「出+入」** / **动态饥饿**）；
  · `D.4`/`D.5` ★★★ **守卫实验**：改坏账本 / 改坏闭式 ⇒ 断言必须失败。
"""
import ast
import collections
import inspect
import itertools
import math
import sys
import textwrap

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

TIME_EPS = 1e-9                      # ★ 浮点比较容差
REL_EPS = 1e-9                       # ★ 账本 vs 闭式的**相对**容差（浮点累加的舍入）

# ★★★ **本机实测值 / 推导值** —— 正文里的每个数都在这里**钉死**。
PINNED_SSTAR = 4000.0                # ★ 临界长度 s* = (2·512KiB/32GBps − 0.02)/3.192e-6
PINNED_FAIR_FCFS_AVG = 17.833333333333332   # ★ 107/6
PINNED_FAIR_SJF_AVG = 11.666666666666666    # ★ 70/6 = 穷举最小值
PINNED_FAIR_FCFS_MAXWAIT = 23.0
PINNED_FAIR_SJF_MAXWAIT = 19.0
PINNED_TOT_SWAP = 1062.207488          # ★ always-换出 的总代价（长度网格上求和）
PINNED_TOT_RECOMPUTE = 1788.2306969600002   # ★ always-KV 重算 的总代价
PINNED_TOT_THRESHOLD = 1030.95959552   # ★ 按临界点选 的总代价（逐条取较小者）
PINNED_STARVE_K10 = 30.0               # ★ 动态饥饿：`K=10` 时长作业的开始时刻
PINNED_STARVE_K100 = 300.0             # ★ 动态饥饿：`K=100` 时
# ★★★ **正文那张代价表**（`PREEMPT_LEN_GRID` 顺序，`%.4f` 口径）—— ★ **整表钉住**。
PINNED_SWAP_R4 = (8.3886, 16.7772, 33.5544, 67.1089, 131.072, 268.4355, 536.8709)
PINNED_RECOMPUTE_R4 = (5.3292, 11.0768, 23.8271, 54.3482, 131.072, 378.0515, 1184.526)


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
# 1. 被测：**账本**（逐块搬运 / 逐 token KV 重算的循环）
# ===========================================================================
def ledger_swap(s, bytes_per_token, bw_bytes_per_ms, block_tokens, corrupt=None):
    """★★★ **被测（换出）**：**逐块**搬运这段 KV，**搬出再搬回**，代价逐块累加。

    ★ 与闭式 `closed_swap` **无共享代码**（`D.8` 在源码层核）：这里是**一个 while 循环**。
    ★★ `corrupt="one_way"`（**只用于守卫实验**）：**忘掉「出+入」的第二个方向**。
    """
    t = 0.0
    b = 0
    factor = 1.0 if corrupt == "one_way" else 2.0
    while b < s:
        toks = min(block_tokens, s - b)
        t += factor * toks * bytes_per_token / bw_bytes_per_ms
        b += toks
    return t


def ledger_recompute(s, lin_ms, quad_ms, corrupt=None):
    """★★★ **被测（KV 重算）**：把这段 prefill **重跑一遍** —— **逐 token** 累加。

    ★ 与闭式 `closed_recompute` **无共享代码**：这里是**一个 `for` 循环**。
    ★★ 每个 token 的代价 = **线性项**（矩阵乘 / MLP）+ **注意力项**（与 `s` 个位置逐对，
      模型假设 —— ★ 因果掩码下其实是 `i` 个，本模型取 `s`）；
    ★★ `corrupt="no_quad"`（**只用于守卫实验**）：**去掉二次项**。
    """
    t = 0.0
    for _ in range(s):
        t += lin_ms
        if corrupt != "no_quad":
            t += quad_ms * s
    return t


# ===========================================================================
# 2. 参照：**闭式**（两条路径无共享代码）
# ===========================================================================
def closed_swap(s, bytes_per_token, bw_bytes_per_ms):
    """闭式：`2·s·bpt/BW`（**出 + 入**）。"""
    return 2.0 * s * bytes_per_token / bw_bytes_per_ms


def closed_recompute(s, lin_ms, quad_ms):
    """闭式：`s·τ_lin + s²·τ_quad`（线性项 + 自注意力的二次项）。"""
    return s * lin_ms + s * s * quad_ms


def closed_sstar(bytes_per_token, bw_bytes_per_ms, lin_ms, quad_ms):
    """★★★ **临界长度的闭式**：解 `2·bpt/BW = τ_lin + s·τ_quad` ⇒ `s* = (2·bpt/BW − τ_lin)/τ_quad`。"""
    return (2.0 * bytes_per_token / bw_bytes_per_ms - lin_ms) / quad_ms


def cross_scan_ledger(bytes_per_token, bw_bytes_per_ms, lin_ms, quad_ms, lo, hi,
                      block_tokens, corrupt=None):
    """★★★ **`s*` 的第二条独立路径**：在**账本**上二分，找「换出 − 重算」**变号**的地方。

    ★★ `g(s) = 换出(s) − KV 重算(s)`：在 `(0, s*)` 上为正、在 `s > s*` 上为负
      ⇒ 找**最后一个** `g(s) > 0` 的整数 `s`。★ 它**不调用** `closed_sstar`（两条路径独立）。
    """
    def g(x):
        return (ledger_swap(x, bytes_per_token, bw_bytes_per_ms, block_tokens, corrupt)
                - ledger_recompute(x, lin_ms, quad_ms, corrupt))
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if g(mid) > 0.0:
            lo = mid
        else:
            hi = mid - 1
    return lo


# ===========================================================================
# 3. 调度：静态集合上的顺序（供公平性对照）
# ===========================================================================
def avg_completion(services, order):
    """给定处理顺序，返回**平均完成时间**（`A.4` 里 `sjf_avg`/`fcfs_avg` 用它）。"""
    t = 0.0
    tot = 0.0
    for i in order:
        t += services[i]
        tot += t
    return tot / len(services)


def avg_completion_alt(services, order):
    """★★★ **独立的第二份目标函数**（★ 只供 `brute_force_min_avg` 用）。

    ★★ 与 `avg_completion` **无共享代码**：两者都算「完成时刻的算术平均」，
      但**写法不同**（这里先收集全部完成时刻、再用 `math.fsum` 求均值）。
    ★★★ 它是 `A.4` 的**防自证装置**：若目标函数本身写错（如 `sum` ↔ `avg`），
      `sjf_avg` 与穷举会**一起错**；现在**两份独立实现**必须给出同一个最小值
      （`D.8` 在源码层核 `brute_force_min_avg` 不调用 `avg_completion`）。
    """
    finish = []
    t = 0.0
    for i in order:
        t = t + services[i]
        finish.append(t)
    return math.fsum(finish) / len(services)


def max_wait(services, order):
    """给定处理顺序，返回**最大等待时间**（`max(完成 − 服务)`）。"""
    t = 0.0
    m = 0.0
    for i in order:
        t += services[i]
        m = max(m, t - services[i])
    return m


def brute_force_min_avg(services):
    """★★★ **参照**：对全部 `n!` 个处理顺序**穷举**，取平均完成时间的**最小值**。

    ★★ 它**不共享任何启发式**（不排序、不贪心）—— ★ 是最强的一种参照。
    ★★ 它调用的是**独立的第二份目标函数** `avg_completion_alt`（★ **不是** `avg_completion`）——
      于是「目标函数本身写错」不会让 `sjf` 与穷举**一起错**（`D.8` 在源码层核这一点）。
    """
    n = len(services)
    return min(avg_completion_alt(services, p) for p in itertools.permutations(range(n)))


def sim_srtf(arrivals, services):
    """**抢占式最短剩余时间（SRTF）**：每一步选**剩余时间最小**的可运行作业。

    ★ 返回 `(start, finish)`。★ 用于 `C.4` 的**动态饥饿**演示（静态集合上不会出现的现象）。
    """
    n = len(services)
    rem = [float(s) for s in services]
    start = [None] * n
    finish = [None] * n
    t = 0.0
    done = 0
    while done < n:
        avail = [i for i in range(n) if arrivals[i] <= t + TIME_EPS and rem[i] > TIME_EPS]
        if not avail:
            future = [arrivals[i] for i in range(n) if rem[i] > TIME_EPS]
            t = min(future)
            continue
        i = min(avail, key=lambda x: (rem[x], arrivals[x], x))
        if start[i] is None:
            start[i] = t
        future = [arrivals[j] for j in range(n) if rem[j] > TIME_EPS and arrivals[j] > t + TIME_EPS]
        nxt = min(future) if future else None
        run_until = t + rem[i]
        if nxt is not None and nxt < run_until:
            rem[i] -= nxt - t
            t = nxt
        else:
            t = run_until
            rem[i] = 0.0
            finish[i] = t
            done += 1
    return start, finish


# ===========================================================================
# 4. A 类 —— ★★★ **换出 vs KV 重算的代价账** 与 **SJF 最优性**
# ===========================================================================
def part_A():
    print("=" * 78)
    print("A · **抢占的代价账**（账本 vs 闭式） · **SJF 最优性**（贪心 vs 穷举）")
    print("=" * 78)
    bpt = specs.PREEMPT_KV_BYTES_PER_TOKEN
    bw = specs.PREEMPT_SWAP_BW_BYTES_PER_MS
    lin = specs.PREEMPT_RECOMPUTE_LIN_MS_PER_TOKEN
    quad = specs.PREEMPT_RECOMPUTE_QUAD_MS_PER_TOKEN_SQ
    blk = specs.PREEMPT_SWAP_BLOCK_TOKENS
    grid = specs.PREEMPT_LEN_GRID
    print("  ★★ 模型：`换出 = 2·s·bpt/BW`（出+入）· `KV 重算 = s·τ_lin + s²·τ_quad`")
    print("  ★★ 参数：`bpt={:,} B` · `BW={:,.0f} B/ms` · `τ_lin={}` · `τ_quad={}`"
          .format(bpt, bw, lin, quad))

    # ── A.1 ★★★ 账本与闭式逐位一致 ─────────────────────────────────────
    print()
    print("  ── A.1 ★★★ **账本（循环）与闭式（式子）逐位一致** ──")
    print("     %-8s %-16s %-16s %-16s %-16s %s"
          % ("s", "账本换出", "闭式换出", "账本KV 重算", "闭式KV 重算", "一致"))
    all_match = True
    for s in grid:
        ls = ledger_swap(s, bpt, bw, blk)
        cs = closed_swap(s, bpt, bw)
        lr = ledger_recompute(s, lin, quad)
        cr = closed_recompute(s, lin, quad)
        ok = (abs(ls - cs) <= REL_EPS * max(1.0, abs(cs))
              and abs(lr - cr) <= REL_EPS * max(1.0, abs(cr)))
        all_match = all_match and ok
        print("     %-8d %-16.6f %-16.6f %-16.4f %-16.4f %s" % (s, ls, cs, lr, cr, "✓" if ok else "✗"))
    print("     ★★ 共 **%d 档**，两条路径全部吻合 = %s" % (len(grid), all_match))
    check("A.1", all_match and len(grid) >= 5
          and tuple(round(closed_swap(s, bpt, bw), 4) for s in grid) == PINNED_SWAP_R4
          and tuple(round(closed_recompute(s, lin, quad), 4) for s in grid) == PINNED_RECOMPUTE_R4,
          "★★★ **判据成立**：**%d 档**长度上，**逐块搬运 / 逐 token KV 重算的账本**与**闭式**"
          "**逐位一致** —— ★★ 一条是循环、一条是式子，**两条独立路径会合**" % len(grid),
          "**账本与闭式对不上** —— ★ 先查换出的 `×2`（出+入）与KV 重算的二次项")

    # ── A.2 ★★★ 临界长度 s* 的两条独立路径 ─────────────────────────────
    print()
    print("  ── A.2 ★★★ **临界长度 `s*`**：闭式 vs **在账本上二分** ──")
    sstar = closed_sstar(bpt, bw, lin, quad)
    cross = cross_scan_ledger(bpt, bw, lin, quad, 1, specs.PREEMPT_SCAN_MAX, blk)
    print("     闭式 `s* = (2·bpt/BW − τ_lin)/τ_quad` = **%.4f**" % sstar)
    print("     账本二分：`g(s) = 换出 − KV 重算` 最后一次 > 0 的整数 `s` = **%d**" % cross)
    # ★ 两侧方向（用账本，不用闭式）
    lo_s = specs.PREEMPT_LEN_GRID[0]
    hi_s = specs.PREEMPT_LEN_GRID[-1]
    swap_lo, rc_lo = ledger_swap(lo_s, bpt, bw, blk), ledger_recompute(lo_s, lin, quad)
    swap_hi, rc_hi = ledger_swap(hi_s, bpt, bw, blk), ledger_recompute(hi_s, lin, quad)
    print("     `s=%d`（< `s*`）：换出 %.2f vs KV 重算 %.2f ⇒ **KV 重算更便宜 = %s**"
          % (lo_s, swap_lo, rc_lo, rc_lo < swap_lo))
    print("     `s=%d`（> `s*`）：换出 %.2f vs KV 重算 %.2f ⇒ **换出更便宜 = %s**"
          % (hi_s, swap_hi, rc_hi, swap_hi < rc_hi))
    print("     ★★★ **读法**：**短序列 KV 重算、长序列换出** —— 因为KV 重算的**自注意力项按 `s²` 涨**，")
    print("        而换出只按 `s` 涨；`s*` 就是这两条线的交点。")
    check("A.2", abs(sstar - PINNED_SSTAR) < 1e-6
          and abs(cross - sstar) <= 1.0
          and rc_lo < swap_lo and swap_hi < rc_hi,
          "★★★ **判据成立**：闭式 `s*` = **%.1f**，账本二分落点 **%d**（★ 两条路径相差 ≤ 1，"
          "★ 差 ≤ 1 是因为临界点上 `g` 恰好为 0、落点取整数侧）；"
          "★ 两侧方向都成立（`s<%d` KV 重算便宜、`s>%d` 换出便宜）"
          % (sstar, cross, int(sstar), int(sstar)),
          "**临界长度的两条路径不一致** —— ★ 先查闭式的分子（`2·bpt/BW − τ_lin`）与二分的比较符")

    # ── A.3 ★★ 临界带宽 ────────────────────────────────────────────────
    print()
    print("  ── A.3 ★★ **临界带宽 `BW*`**：让「换出 = 重算」的那个带宽 ──")
    s_fix = 1024
    bw_star = 2.0 * bpt / (lin + s_fix * quad)
    # ★ 两侧：带宽更小 ⇒ 换出更贵；带宽更大 ⇒ 换出更便宜
    lo_bw, hi_bw = bw_star * 0.5, bw_star * 2.0
    rc_fix = closed_recompute(s_fix, lin, quad)
    sw_lo = closed_swap(s_fix, bpt, lo_bw)
    sw_hi = closed_swap(s_fix, bpt, hi_bw)
    print("     `s=%d` 固定：`BW* = 2·bpt/(τ_lin + s·τ_quad)` = **%s B/ms**"
          % (s_fix, format(bw_star, ",.0f")))
    print("     `BW = 0.5·BW*` ⇒ 换出 %.2f > KV 重算 %.2f ⇒ **KV 重算更便宜 = %s**"
          % (sw_lo, rc_fix, rc_fix < sw_lo))
    print("     `BW = 2.0·BW*` ⇒ 换出 %.2f < KV 重算 %.2f ⇒ **换出更便宜 = %s**"
          % (sw_hi, rc_fix, sw_hi < rc_fix))
    print("     ★★ **读法**：带宽**越低**（越贵），越应当 **KV 重算**；带宽**越高**，越应当**换出** ——")
    print("        ★★ 与 `A.2` 的**长度**是**两个独立的旋钮**（一个改 `s`、一个改 `BW`）。")
    check("A.3", abs(2.0 * bpt / (lin + s_fix * quad) - bw_star) < 1e-6
          and rc_fix < sw_lo and sw_hi < rc_fix,
          "★★ **判据成立**：`s=%d` 时的临界带宽 **%s B/ms** —— 带宽减半则**KV 重算更便宜**、"
          "加倍则**换出更便宜**（★ 两个方向都成立）" % (s_fix, format(bw_star, ",.0f")),
          "**临界带宽的方向被破坏** —— ★ 先查 `2·bpt/(τ_lin + s·τ_quad)`")

    # ── A.4 ★★★ SJF 最优性（穷举参照）──────────────────────────────────
    print()
    print("  ── A.4 ★★★ **SJF 最优性**：贪心 vs **全部 `n!` 个顺序穷举** ──")
    svc = specs.FAIR_SERVICE_TOKENS
    n = len(svc)
    fcfs_order = tuple(range(n))
    sjf_order = tuple(sorted(range(n), key=lambda i: svc[i]))
    fcfs_avg = avg_completion(svc, fcfs_order)
    sjf_avg = avg_completion(svc, sjf_order)
    # ★★ 第二份**独立**目标函数（★ `brute_force_min_avg` 用的就是它，不是 `avg_completion`）
    sjf_avg_alt = avg_completion_alt(svc, sjf_order)
    best = brute_force_min_avg(svc)
    print("     服务时间（按到达序）= %s（`n=%d` ⇒ **%d 个顺序**）" % (svc, n, math.factorial(n)))
    print("     FCFS 顺序 %s ⇒ 平均完成 **%.4f**" % (fcfs_order, fcfs_avg))
    print("     SJF  顺序 %s ⇒ 平均完成 **%.4f**" % (sjf_order, sjf_avg))
    print("     **穷举最小** ⇒ **%.4f**（★ 穷举用的是**独立的第二份目标函数**）" % best)
    print("     ★★ **两份独立的目标函数**在 SJF 序上一致：`avg_completion` **%.4f** == "
          "`avg_completion_alt` **%.4f**" % (sjf_avg, sjf_avg_alt))
    print("     ★★★ **读法**：SJF 的平均完成时间**恰好等于穷举最小值** ⇒ 它在**这组数据上最优**；")
    print("        而 FCFS 比它差 **%.2f**（★ 贪心不是「近似」，在这条判据上**就是最优**）。" % (fcfs_avg - sjf_avg))
    check("A.4", abs(sjf_avg - best) < TIME_EPS and fcfs_avg > best + TIME_EPS
          and abs(sjf_avg - sjf_avg_alt) < TIME_EPS
          and math.factorial(n) == 720,
          "★★★ **判据成立**：SJF 贪心 **%.4f** == **穷举最小值** %.4f（`%d!` 个顺序）；"
          "★ FCFS **%.4f** 比它差；★★ 穷举用的是**独立的第二份目标函数**（`%.4f` == `%.4f`）——"
          " ★★ 参照是**穷举**，**不共享任何启发式**，**目标函数也不共享**"
          % (sjf_avg, best, n, fcfs_avg, sjf_avg, sjf_avg_alt),
          "**SJF 与穷举最小值不一致** —— ★ 先查 `SJF` 的排序方向（升序 = 最短优先）")


# ===========================================================================
# 5. B 类：代价表 · 策略总代价 · 公平指标
# ===========================================================================
def part_B():
    print()
    print("=" * 78)
    print("B · **代价模型**：三种策略的总代价 · 公平指标")
    print("=" * 78)
    bpt = specs.PREEMPT_KV_BYTES_PER_TOKEN
    bw = specs.PREEMPT_SWAP_BW_BYTES_PER_MS
    lin = specs.PREEMPT_RECOMPUTE_LIN_MS_PER_TOKEN
    quad = specs.PREEMPT_RECOMPUTE_QUAD_MS_PER_TOKEN_SQ
    grid = specs.PREEMPT_LEN_GRID
    sstar = closed_sstar(bpt, bw, lin, quad)

    # ── B.1 ★★ 代价表 + 三种策略 ────────────────────────────────────────
    print()
    print("  ── B.1 ★★ **代价表** 与 **三种抢占策略的总代价** ──")
    print("     %-8s %-14s %-14s %-14s %s" % ("s", "换出", "KV 重算", "取较小者", "谁更便宜"))
    tot_swap = tot_rc = tot_best = 0.0
    for s in grid:
        sw = closed_swap(s, bpt, bw)
        rc = closed_recompute(s, lin, quad)
        who = "换出" if sw < rc else ("KV 重算" if rc < sw else "相等")
        tot_swap += sw
        tot_rc += rc
        tot_best += min(sw, rc)
        print("     %-8d %-14.4f %-14.4f %-14.4f %s" % (s, sw, rc, min(sw, rc), who))
    print("     ★★ **总代价**：always-换出 = **%.2f** · always-KV 重算 = **%.2f** · 按临界点选 = **%.2f**"
          % (tot_swap, tot_rc, tot_best))
    print("     ★★ **读法**：**按临界点选**的总代价**严格小于**任一固定策略 ——")
    print("        它逐条取较小者，而固定策略在某些长度上**必然付更贵的那一边**。")
    check("B.1", tot_best <= tot_swap + TIME_EPS and tot_best <= tot_rc + TIME_EPS
          and tot_best > 0.0 and (tot_best < tot_swap - TIME_EPS or tot_best < tot_rc - TIME_EPS)
          and abs(tot_swap - PINNED_TOT_SWAP) < 1e-6
          and abs(tot_rc - PINNED_TOT_RECOMPUTE) < 1e-6
          and abs(tot_best - PINNED_TOT_THRESHOLD) < 1e-6,
          "★★ **判据成立**：**按临界点选** = **%.2f** ≤ 两个固定策略（%.2f / %.2f），"
          "且**严格更小** —— ★★ 策略的价值来自「**两条曲线在 `s*` 处交叉**」（`A.2`）"
          % (tot_best, tot_swap, tot_rc),
          "**按临界点选没有比固定策略更好** —— ★ 先查临界点是否落在网格内（`D.3b`）")

    # ── B.2 ★★ 公平指标 ────────────────────────────────────────────────
    print()
    print("  ── B.2 ★★ **公平指标**：FCFS 与 SJF（静态集合，同时到达）──")
    svc = specs.FAIR_SERVICE_TOKENS
    n = len(svc)
    fcfs = tuple(range(n))
    sjf = tuple(sorted(range(n), key=lambda i: svc[i]))
    f_avg, s_avg = avg_completion(svc, fcfs), avg_completion(svc, sjf)
    f_mw, s_mw = max_wait(svc, fcfs), max_wait(svc, sjf)
    print("     %-10s %-16s %s" % ("策略", "平均完成时间", "最大等待"))
    print("     %-10s %-16.4f %.1f" % ("FCFS", f_avg, f_mw))
    print("     %-10s %-16.4f %.1f" % ("SJF", s_avg, s_mw))
    print("     ★★ **读法**：SJF 把**平均**完成时间降了 **%.2f**（%.2f → %.2f）；"
          % (f_avg - s_avg, f_avg, s_avg))
    print("        ★★ 而**最大等待也不比 FCFS 差**（%.0f ≤ %.0f）—— " % (s_mw, f_mw))
    print("        ★★★ **注意**：这与「SJF 会饿死长作业」**不矛盾** —— 饥饿**只在持续到达时出现**（`C.4`）。")
    check("B.2", abs(f_avg - PINNED_FAIR_FCFS_AVG) < 1e-9
          and abs(s_avg - PINNED_FAIR_SJF_AVG) < 1e-9
          and abs(f_mw - PINNED_FAIR_FCFS_MAXWAIT) < TIME_EPS
          and abs(s_mw - PINNED_FAIR_SJF_MAXWAIT) < TIME_EPS
          and s_mw <= f_mw + TIME_EPS,
          "★★ **判据成立**：FCFS 平均 **%.4f** / 最大等待 **%.0f**；SJF 平均 **%.4f** / 最大等待 **%.0f** ——"
          " ★★ 在**静态集合**上 SJF 两项都不差（★ 最大等待反而更小）"
          % (f_avg, f_mw, s_avg, s_mw),
          "**公平指标被动过** —— ★ 先查 `FAIR_SERVICE_TOKENS` 与两种顺序")


# ===========================================================================
# 6. C 类：边界与反例
# ===========================================================================
def part_C():
    print()
    print("=" * 78)
    print("C · 边界与反例：长度 0 · 带宽两端 · **忘掉「出+入」** · **动态饥饿**")
    print("=" * 78)
    bpt = specs.PREEMPT_KV_BYTES_PER_TOKEN
    bw = specs.PREEMPT_SWAP_BW_BYTES_PER_MS
    lin = specs.PREEMPT_RECOMPUTE_LIN_MS_PER_TOKEN
    quad = specs.PREEMPT_RECOMPUTE_QUAD_MS_PER_TOKEN_SQ
    blk = specs.PREEMPT_SWAP_BLOCK_TOKENS
    sstar = closed_sstar(bpt, bw, lin, quad)

    # ── C.1 ★★ s = 0 退化 ──────────────────────────────────────────────
    print()
    print("  ── C.1 ★★ **边界**：`s = 0` ⇒ 两种代价都为 0 ──")
    zs = ledger_swap(0, bpt, bw, blk)
    zr = ledger_recompute(0, lin, quad)
    print("     `s=0` ⇒ 账本换出 = **%.1f** · 账本KV 重算 = **%.1f**（★ 没有 KV 可搬、也没有 token 要KV 重算）"
          % (zs, zr))
    check("C.1", zs == 0.0 and zr == 0.0,
          "★★ **判据成立**：`s=0` 时两种代价**都为 0** —— ★ 于是 `s*` 的**非零根**才是有意义的那个"
          "（★ 闭式的式子 `s·(...)` 在 `s=0` 处也有一个平凡根）",
          "**`s=0` 的边界被动过** —— ★ 先查两个账本的循环初值")

    # ── C.2 ★★ 带宽两端 ────────────────────────────────────────────────
    print()
    print("  ── C.2 ★★ **带宽两端**：极慢 ⇒ **永远 KV 重算**；极快 ⇒ **永远换出** ──")
    s_fix = 4096
    rc_fix = closed_recompute(s_fix, lin, quad)
    for label, factor in (("极慢 (BW/1000)", 1e-3), ("极快 (BW×1000)", 1e3)):
        sw = closed_swap(s_fix, bpt, bw * factor)
        print("     `%s`：`s=%d` 换出 = %.4f vs KV 重算 = %.4f ⇒ **%s**"
              % (label, s_fix, sw, rc_fix, "换出" if sw < rc_fix else "KV 重算"))
    sw_slow = closed_swap(s_fix, bpt, bw * 1e-3)
    sw_fast = closed_swap(s_fix, bpt, bw * 1e3)
    check("C.2", rc_fix < sw_slow and sw_fast < rc_fix,
          "★★ **判据成立**：带宽缩 `1000` 倍 ⇒ 换出 **%.2f** ≫ KV 重算 **%.2f**（**永远KV 重算**）；"
          "扩 `1000` 倍 ⇒ 换出 **%.4f** ≪ KV 重算（**永远换出**）—— ★★ 两端都把临界点挤出了可行域"
          % (sw_slow, rc_fix, sw_fast),
          "**带宽两端的行为被动过** —— ★ 先查换出的 `1/BW` 依赖")

    # ── C.3 ★★★ 忘掉「出+入」= 结论反转 ───────────────────────────────
    print()
    print("  ── C.3 ★★★ **反例：忘掉「出+入」** ⇒ 临界点消失、结论**反转** ──")
    # ★ 只算单向的坏法
    one_lo = ledger_swap(specs.PREEMPT_LEN_GRID[0], bpt, bw, blk, corrupt="one_way")
    one_hi = ledger_swap(specs.PREEMPT_LEN_GRID[-1], bpt, bw, blk, corrupt="one_way")
    good_lo = ledger_swap(specs.PREEMPT_LEN_GRID[0], bpt, bw, blk)
    good_hi = ledger_swap(specs.PREEMPT_LEN_GRID[-1], bpt, bw, blk)
    rc_lo = ledger_recompute(specs.PREEMPT_LEN_GRID[0], lin, quad)
    rc_hi = ledger_recompute(specs.PREEMPT_LEN_GRID[-1], lin, quad)
    print("     正确版（出+入）：`s=%d` 换出 %.2f · `s=%d` 换出 %.2f"
          % (specs.PREEMPT_LEN_GRID[0], good_lo, specs.PREEMPT_LEN_GRID[-1], good_hi))
    print("     坏法（只算出）：`s=%d` 换出 %.2f · `s=%d` 换出 %.2f"
          % (specs.PREEMPT_LEN_GRID[0], one_lo, specs.PREEMPT_LEN_GRID[-1], one_hi))
    print("     对 `s=%d`（**正确版里换出更便宜**）：坏法换出 %.2f vs KV 重算 %.2f ⇒ **%s**"
          % (specs.PREEMPT_LEN_GRID[-1], one_hi, rc_hi, "换出" if one_hi < rc_hi else "KV 重算"))
    print("     ★★★ **读法**：忘掉一半 ⇒ 换出的代价**系统性地低估 2 倍** ⇒ 临界长度 `s*` **左移或消失** ⇒")
    print("        结论从「**短序列重算、长序列换出**」变成「**几乎总是换出**」。")
    print("        ★★ 这正是 `20-gpu-architecture/interconnect/01` 说的**单向 / 双向口径**之分的现场。")
    # ★ 坏法下：在网格上是否还有「重算更便宜」的档？
    worst = all(ledger_swap(s, bpt, bw, blk, corrupt="one_way")
                <= ledger_recompute(s, lin, quad) + TIME_EPS for s in specs.PREEMPT_LEN_GRID)
    check("C.3", abs(one_hi - good_hi / 2.0) < 1e-6 and len(specs.PREEMPT_LEN_GRID) > 0
          and one_lo < rc_lo and worst,
          "★★★ **判据成立**：坏法把换出代价**恰好缩到一半**（`%.2f = %.2f / 2`）⇒ "
          "在整个网格上**再没有任何长度适合 KV 重算**（`worst=%s`）—— ★★ **结论反转**"
          % (one_hi, good_hi, worst),
          "**「忘掉出+入」没有导致结论反转** —— ★ 先查 `corrupt=\"one_way\"` 的 factor")

    # ── C.4 ★★★ 动态饥饿（静态集合上不出现的现象）─────────────────────
    print()
    print("  ── C.4 ★★★ **动态饥饿**：持续到达的短作业把长作业**无限期推迟** ──")
    short, long_ = specs.FAIR_STARVE_SHORT, specs.FAIR_STARVE_LONG
    starts = {}
    for K in (10, 100):
        arrivals = [0.0] + [float(i) for i in range(K)]
        services = [long_] + [short] * K
        st, _ = sim_srtf(arrivals, services)
        starts[K] = st[0]
    print("     **长作业**（服务 `%d`）在 `t=0` 到达；**短作业**（服务 `%d`）每单位到一条，共 `K` 条" % (long_, short))
    print("     `K=10`  ⇒ 长作业**开始**时刻 = **%.0f**" % starts[10])
    print("     `K=100` ⇒ 长作业**开始**时刻 = **%.0f**" % starts[100])
    print("     ★★★ **读法**：长作业的开始时刻**随短作业条数线性增长**（`.%d` 每来一条）——" % short)
    print("        ★★ 若短作业**永不停止到达**，长作业**永远轮不到**（**等待无界**）= **饥饿**。")
    print("        ★★ 而 `B.2` 的**静态集合**上**看不到**这个现象（那里 SJF 的最大等待反而更小）——")
    print("        ⇒ ★★★ **「饥饿」是【动态到达】的属性，不是【排序】的属性**。")
    check("C.4", abs((starts[100] - starts[10]) - short * 90.0) < TIME_EPS
          and abs(starts[10] - PINNED_STARVE_K10) < TIME_EPS
          and abs(starts[100] - PINNED_STARVE_K100) < TIME_EPS
          and starts[100] > starts[10],
          "★★★ **判据成立**：短作业条数从 `10` 涨到 `100`，长作业的开始时刻沿 **每条约 `%d`** 线性后移"
          "（`%.0f → %.0f`）⇒ ★★ **到达不停则等待无界**" % (short, starts[10], starts[100]),
          "**动态饥饿没有跑出来** —— ★ 先查 SRTF 的「最小剩余时间」选择与到达序列")


# ===========================================================================
# 7. 前哨：口径 / 载体 / 标记自检（★ 跑在任何计算之前）
# ===========================================================================
def preflight():
    """★★ 口径 / 载体 / 标记自检**一律在这里**，且**跑在任何计算之前**（本库 `P23`）。"""
    print("=" * 78)
    print("D0. 口径/载体/标记钉住（**先跑**：让「改常量」变成断言捕获，而不是崩溃捕获）")
    print("=" * 78)
    print("  本程序的失败标记 ASSERT_FAIL_MARK = %r" % ASSERT_FAIL_MARK)
    print("  ★★★ 限定语：**不是任何真实引擎，也不是真机** —— 本机**无 GPU / 无引擎**；")
    print("      ★★ 但**本程序的 A 类是真的**：**代价账的两条独立路径** + **SJF vs 穷举**。")
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
          and abs(TIME_EPS - 1e-9) < 1e-18 and abs(REL_EPS - 1e-9) < 1e-18,
          "★ **载体元数据逐字钉住**：抓取日 `%s`；★ `UNSOURCED == ()` 一并钉住；"
          "★ 两个容差 `TIME_EPS` / `REL_EPS` **在 preflight 里就钉住**（★ 否则它们的变异会变成"
          "**崩溃捕获**，而不是断言捕获）" % specs.SCHED_FETCH_DAY)

    # ★★★ 口径守卫（★ 跑在任何计算之前）：`s*` 必须**落在长度网格内**（两侧都有落点），
    #   否则 `A.2` 的方向判据会退化成恒真；★ 且二次项必须 > 0（否则没有长度临界点）。
    sstar = closed_sstar(specs.PREEMPT_KV_BYTES_PER_TOKEN, specs.PREEMPT_SWAP_BW_BYTES_PER_MS,
                         specs.PREEMPT_RECOMPUTE_LIN_MS_PER_TOKEN,
                         specs.PREEMPT_RECOMPUTE_QUAD_MS_PER_TOKEN_SQ)
    grid = specs.PREEMPT_LEN_GRID
    ok = (specs.PREEMPT_RECOMPUTE_QUAD_MS_PER_TOKEN_SQ > 0.0
          and any(s < sstar for s in grid) and any(s > sstar for s in grid)
          and specs.PREEMPT_SWAP_BLOCK_TOKENS >= 1
          and specs.PREEMPT_SCAN_MAX > sstar
          and len(specs.FAIR_SERVICE_TOKENS) == 6
          and specs.FAIR_STARVE_LONG > specs.FAIR_STARVE_SHORT)
    print("     ★★ 口径守卫：`s*={:.1f}` 落在网格内 = {}（含 `s<s*` = {}、含 `s>s*` = {}）；"
          "二次项 > 0 = {}；扫描上界 > `s*` = {}"
          .format(sstar, any(s < sstar for s in grid) and any(s > sstar for s in grid),
                  any(s < sstar for s in grid), any(s > sstar for s in grid),
                  specs.PREEMPT_RECOMPUTE_QUAD_MS_PER_TOKEN_SQ > 0.0,
                  specs.PREEMPT_SCAN_MAX > sstar))
    check("D.3b", ok,
          "★★★ **口径守卫**：**`s*` 必须落在长度网格内**（★ 否则 `A.2` 的「两侧方向」只测到一侧）；"
          "★ **二次项必须 > 0**（★ 它是长度临界点的**唯一来源**，取 0 则没有临界点）；"
          "★ 扫描上界必须 `> s*`（否则二分落在区间外）",
          "**临界点或网格的覆盖被动过** —— ★ 先查 `PREEMPT_*` 与 `FAIR_*`")
    return len(FAILS) == before


# ===========================================================================
# 8. D 类（后半）：守卫实验 · 逐元素 · 独立性
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
    print("D · **守卫实验** · **逐元素**钉住 · 独立性线索")
    print("=" * 78)
    bpt = specs.PREEMPT_KV_BYTES_PER_TOKEN
    bw = specs.PREEMPT_SWAP_BW_BYTES_PER_MS
    lin = specs.PREEMPT_RECOMPUTE_LIN_MS_PER_TOKEN
    quad = specs.PREEMPT_RECOMPUTE_QUAD_MS_PER_TOKEN_SQ
    blk = specs.PREEMPT_SWAP_BLOCK_TOKENS
    s_fix = 4096

    # ── D.4 ★★★ 守卫实验 ①：把账本的换出改成单向 ────────────────────────
    print()
    print("  ── D.4 ★★★ **守卫实验 ①**：把账本的换出改成**单向**（只算出，不算入）──")
    g_good = ledger_swap(s_fix, bpt, bw, blk)
    g_bad = ledger_swap(s_fix, bpt, bw, blk, corrupt="one_way")
    g_ref = closed_swap(s_fix, bpt, bw)
    print("     `s=%d`：正确（出+入） = **%.4f**（= 闭式 %.4f）· 坏法（单向） = **%.4f**"
          % (s_fix, g_good, g_ref, g_bad))
    check("D.4", abs(g_good - g_ref) <= REL_EPS * g_ref and abs(g_bad - g_ref) > 1e-6,
          "★★★ **守卫成立**：改单向后账本从 **%.4f**（= 闭式）变成 **%.4f** —— ★★ 于是 `A.1` 的"
          "「逐位一致」对「**出 + 入**」这条规则**是敏感的**" % (g_good, g_bad),
          "**守卫失效** —— ★ 改单向后账本居然仍等于闭式（那 `A.1` 抓不到它）")

    # ── D.5 ★★★ 守卫实验 ②：把闭式的【二次项】去掉 ─────────────────────
    print()
    print("  ── D.5 ★★★ **守卫实验 ②**：把 KV 重算的**二次项去掉**（`no_quad`）──")
    h_good = ledger_recompute(s_fix, lin, quad)
    h_bad = ledger_recompute(s_fix, lin, quad, corrupt="no_quad")
    h_ref = closed_recompute(s_fix, lin, quad)
    sstar_bad = (2.0 * bpt / bw - lin) / (quad)  # 仍用闭式对照：坏法下无临界点
    print("     `s=%d`：正确 = **%.4f**（= 闭式 %.4f）· 坏法（去二次项） = **%.4f**（= `s·τ_lin`）"
          % (s_fix, h_good, h_ref, h_bad))
    print("     ★★★ **读法**：去掉二次项后KV 重算变成**纯线性** ⇒ **没有长度临界点**（两条线不再交叉）——")
    print("        而 `s*` 的闭式**分母**正是那个二次项系数。")
    check("D.5", abs(h_good - h_ref) <= REL_EPS * h_ref
          and abs(h_bad - s_fix * lin) < TIME_EPS and abs(h_bad - h_ref) > 1e-6,
          "★★★ **守卫成立**：去二次项后KV 重算 = **%.4f**（纯线性），与闭式 **%.4f** 对不上 ——"
          " ★★ `A.1` 的「逐位一致」对**参照的表达式**（含 `s²` 项）也敏感；★ 且**临界点因此消失**"
          % (h_bad, h_ref),
          "**守卫失效** —— ★ 去二次项后账本仍与闭式一致（那 `A.1` 抓不到它）")

    # ── D.6 ★★★ 逐元素钉住 ──────────────────────────────────────────────
    check("D.6", specs.SCHED_FETCH_DAY == "2026-10-09"
          and specs.PREEMPT_KV_BYTES_PER_TOKEN == 524288
          and specs.PREEMPT_SWAP_BW_BYTES_PER_MS == 32000000.0
          and specs.PREEMPT_SWAP_BLOCK_TOKENS == 16
          and specs.PREEMPT_RECOMPUTE_LIN_MS_PER_TOKEN == 0.02
          and specs.PREEMPT_RECOMPUTE_QUAD_MS_PER_TOKEN_SQ == 0.000003192
          and specs.PREEMPT_LEN_GRID == (256, 512, 1024, 2048, 4000, 8192, 16384)
          and specs.PREEMPT_SCAN_MAX == 65536
          and specs.FAIR_SERVICE_TOKENS == (7, 2, 9, 4, 1, 5)
          and specs.FAIR_STARVE_SHORT == 3 and specs.FAIR_STARVE_LONG == 100000
          and specs.UNSOURCED == ()
          and abs(TIME_EPS - 1e-9) < 1e-18 and abs(REL_EPS - 1e-9) < 1e-18
          and abs(PINNED_SSTAR - 4000.0) < 1e-9
          and abs(PINNED_FAIR_FCFS_AVG - 107.0 / 6) < 1e-12
          and abs(PINNED_FAIR_SJF_AVG - 70.0 / 6) < 1e-12
          and abs(PINNED_FAIR_FCFS_MAXWAIT - 23.0) < 1e-9
          and abs(PINNED_FAIR_SJF_MAXWAIT - 19.0) < 1e-9
          and abs(PINNED_TOT_SWAP - 1062.207488) < 1e-6
          and abs(PINNED_TOT_RECOMPUTE - 1788.2306969600002) < 1e-6
          and abs(PINNED_TOT_THRESHOLD - 1030.95959552) < 1e-6
          and abs(PINNED_STARVE_K10 - 30.0) < 1e-9
          and abs(PINNED_STARVE_K100 - 300.0) < 1e-9
          and PINNED_SWAP_R4 == (8.3886, 16.7772, 33.5544, 67.1089, 131.072, 268.4355, 536.8709)
          and PINNED_RECOMPUTE_R4 == (5.3292, 11.0768, 23.8271, 54.3482, 131.072, 378.0515, 1184.526),
          "★★★ **规格常量逐元素 + 程序内常量**钉住（`bpt` · `BW` · 块粒度 · 两个KV 重算系数 ·"
          " 长度网格 · 扫描上界 · 公平服务时间 · 两个饥饿参数 · 五个 `PINNED_*`）；★★ `UNSOURCED == ()` 一并钉住",
          "**某个常量被动过**")

    # ── D.7 ★★★ 术语纪律：全称 + **裸写计数 == 0** ──────────────────────
    print()
    print("  ── D.7 ★★★ **术语纪律（可机械判定）**：本文一律写「KV 重算」，**裸写「重算」的计数 == 0** ──")
    src = open(__file__, encoding="utf-8").read()
    # ★ **被测串【拼接构造】**：`"重" + "算"` —— 这样本程序的源码里**不出现连续的「重算」**，
    #   于是检测器**不会把自身的比较字面量误判成裸写**（否则它会自己抓自己）。
    needle = "重" + "算"
    full = "KV " + needle
    n_all = sum(1 for ln in src.splitlines() if needle in ln)
    n_full = sum(1 for ln in src.splitlines() if full in ln)
    # ★★★ **裸写**的机械定义：出现「重算」，但**既不**紧跟 `KV `、**也不**被 `「…」` 包住。
    #   ★ 被「」包住的那些是**消歧时对【别义】的引用**（如「物化 vs 重算」「重算」换显存）——
    #     ★★ 它们**必须**保留裸字，否则消歧就失去对象。★ 本义（第三义）**一处也不许裸写**。
    bare = []
    for lineno, ln in enumerate(src.splitlines(), 1):
        depth = 0
        i = 0
        while i < len(ln):
            if ln[i] == "「":
                depth += 1
            elif ln[i] == "」":
                depth = max(0, depth - 1)
            elif ln.startswith(needle, i):
                if ln[max(0, i - 3):i] != "KV " and depth == 0:
                    bare.append((lineno, ln.strip()[:48]))
                i += 1
            i += 1
    print("     含「重算」的行 = **%d** 行，其中含「KV 重算」全称的 = **%d** 行" % (n_all, n_full))
    print("     ★★★ **裸写「重算」（既非 `KV 重算`、也不在「」内）的处数 = %d**%s"
          % (len(bare), ("  ⇒ %s" % bare[:3]) if bare else ""))
    print("     ★★ **读法**：库内「重算」已有**三义**（编译器的「物化 vs 重算」 · 训练的「重算」换显存 ·")
    print("        本域的 **KV 重算**）—— ★★★ 引用本义时**必须写全称**（`§2.2`）。")
    check("D.7", "**KV 重算**" in src and "同名不同指" in src and n_full >= 1 and len(bare) == 0,
          "★★ **判据成立**：本程序**使用「KV 重算」全称**、**显式写下「同名不同指」的消歧**，"
          "且**裸写「重算」的计数 == 0**（机械判定：非 `KV ` 前缀且不在「」内）——"
          " ★★ 只断言「含全称」抓不住裸写（那正是 `§5` v1.5 的 `H1` 形态）",
          "**术语全称 / 消歧缺失，或本义出现了裸写「重算」** —— ★ 先查模块抬头与摘要里的术语纪律")

    # ── D.8 ★★★ 独立性（源码级）────────────────────────────────────────
    ln_ = _names_in(ledger_swap) | _names_in(ledger_recompute)
    cn_ = (_names_in(closed_swap) | _names_in(closed_recompute)
           | _names_in(closed_sstar))
    sj_ = _names_in(avg_completion) | _names_in(brute_force_min_avg)
    print()
    print("  ── D.8 ★★ **源码级线索**（★ **不是证明**）：被测与参照互不引用 ──")
    print("     账本里出现任一闭式名 = %s ；闭式里出现任一账本名 = %s"
          % (sorted(n for n in ln_ if n.startswith(("closed_", "cross_"))),
             sorted(n for n in cn_ if n.startswith("ledger_"))))
    print("     ★★★ **降级措辞**：这只是「**两段代码不互相引用**」的**源码级线索**，"
          "**不证明**判据有信息量 —— 后者由 `A.1`–`A.4` 与 `D.4`/`D.5` 的**行为**兜底。")
    check("D.8", not any(n.startswith(("closed_", "cross_scan")) for n in ln_)
          and not any(n.startswith("ledger_") for n in cn_)
          and "sorted" not in _names_in(brute_force_min_avg)
          and "avg_completion" not in _names_in(brute_force_min_avg),
          "★★ **源码级线索成立**：**账本（循环）与闭式（式子）互不引用**；★ `brute_force_min_avg`"
          "（穷举参照）**不调用 `sorted`**（SJF 的排序原语）；★★ **也不调用 `avg_completion`**"
          "（SJF 的目标函数）—— ★★ 它只做**排列枚举**，**目标函数也独立**（`avg_completion_alt`）",
          "**两实现互相引用** —— A 类可能退化成自证（`§1.3.2b`）")

    # ── D.9 ★★ 边界：临界点是唯一的（两条线只交于一点） ────────────────
    sstar = closed_sstar(bpt, bw, lin, quad)
    diff_lo = closed_swap(int(sstar) - 100, bpt, bw) - closed_recompute(int(sstar) - 100, lin, quad)
    diff_hi = closed_swap(int(sstar) + 100, bpt, bw) - closed_recompute(int(sstar) + 100, lin, quad)
    print()
    print("  ── D.9 ★★ **临界点是唯一的**：`g(s) = 换出 − KV 重算` 在 `s*` 两侧**变号** ──")
    print("     `g(%.0f − 100) = %.4f`（**正**：换出更贵）· `g(%.0f + 100) = %.4f`（**负**：换出更便宜）"
          % (sstar, diff_lo, sstar, diff_hi))
    check("D.9", diff_lo > 0.0 > diff_hi and abs(diff_lo + diff_hi) < 1.0 + TIME_EPS,
          "★★ **边界成立**：`g` 在 `s*` 两侧**符号相反**，且**关于 `s*` 近似奇对称** ——"
          " ★★ 于是临界点是**唯一**的（两条线的交点只有一个，`s=0` 那个是平凡根）",
          "**临界点的唯一性被动过** —— ★ 先查 `g` 的两侧符号")


# ===========================================================================
# 9. 汇总 / 主程序
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
    print("★★ **A = %d/%d** —— ★★★ **本程序有两条 A 类**：代价账（账本 vs 闭式）+ SJF（贪心 vs 穷举）。"
          % (_PASS["A"], _TOTAL["A"]))
    print("★ 判据来源：本主题的代价模型是 **`[E]` 级（本库实现 / 假设）**；"
          "每 token 的 KV 字节取用 `130-model-optimization/quantization/18-kv-cache-basics`。")
    print("★ 本程序只核对「**代价的算术与不变量**」——**不是**任何真实引擎的抢占行为。")
    return 1 if FAILS else 0


def main():
    print("=" * 78)
    print("03_preemption_and_fairness.py —— 抢占的代价账（换出 vs KV 重算）与公平")
    print("=" * 78)
    print("  ★ 失败标记 ASSERT_FAIL_MARK = %r（本主题自定义；不跨主题照抄）" % ASSERT_FAIL_MARK)
    print("  ★★★ 术语：本域的 **KV 重算** 是**第三义**（与编译器 / 训练的「重算」同名不同指）。")
    print("  ★★★ 模型：换出 = `2·s·bpt/BW`（出+入）· KV 重算 = `s·τ_lin + s²·τ_quad`。")
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

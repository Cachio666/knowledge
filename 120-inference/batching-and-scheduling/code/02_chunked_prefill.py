#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""02_chunked_prefill.py —— **chunked prefill 的分块数权衡**：最优分块数 `k* = ⌈t_p/t_d⌉`。

配套文档：`02-chunked-prefill.md`。

★ 三类断言**全部在本程序内部**，失败时以**非零退出码**结束；**没有 `verify.py`**。

★★★ **本主题不写【定义】**：`chunked prefill` 的**定义**在
  `10-foundations/performance-models/01-roofline-model.md:251`（括注定义），
  **prefill/decode 的 kernel 差异**（两者的 `AI` 差 120 倍）在
  `40-kernels/attention/05-decoding-attention.md:229-236` —— 两处**都只回链**。
  本程序写的是**分块数 `k` 的权衡**。

★★★ **本程序最强的一条 A 类：最优分块数 `k* = ⌈t_p/t_d⌉`**（计划 `§0.7 D6`）——
  · **被测** = `sim_chunk`：**逐步推进**的离散事件模拟（每步 1 个 prefill 块 + `C` 个 decode）；
  · **参照** = `closed`：**闭式** `k* = ⌈t_p/t_d⌉`、`每步耗时 = max(t_d, t_p/k)`、`TTFT = k·每步耗时`；
  · ★★ **注意（如实降级）**：`sim_chunk` 把 `step` **重复相加 `k` 次**、与闭式的 `k·step` 是
    **同一个式子** ⇒ ★★ **`A.1` 只是「近乎恒等」的一致性核对，不是两条独立路径**；
  · ★★★ **真正的两条独立路径是 `A.3`**：闭式 `⌈t_p/t_d⌉` vs **在模拟器上逐档扫描**
    （最小的 `k` 使每步耗时 `≤ t_d`）；★ `D.8` 在源码层核两段代码互不引用。
  ★★★ **判据**：`k<k*` 拖慢 decode 而 **TTFT 不变**；`k=k*` **帕累托最优**；`k>k*` **TTFT 线性上升纯亏**。
  ★★ **免费守卫**：`k=1 ⇒ TTFT = t_p`（退化成不分块）；`k→∞ ⇒ TTFT 无界`（每步都要等一个 decode 步）。

★★★ **模型假设（必须写明）**：**一步耗时 = `max(t_d, t_p/k)`** ——
  ★★ **这是简化**：真实里 **prefill 与 decode 共享算力**（不是简单取 `max`），
  且 `C` 个 decode 请求与 prefill 块的**相对成本**随实现变（`[B]` 级，本机**无引擎**）。
  ★ 本程序**只在这个模型内**作断言，**不声称任何真实引擎的性能**。

★★ **覆盖边界**：本程序守的是「**一块的账**」（TTFT / 每步耗时 / `k*` 的三段性质）。
  **不守**任何真实引擎的调度器或 kernel 行为。★★ **最强的限定语**：**「不是任何真实引擎，也不是真机」**。

★★ 可证伪断言（把下面任一条改错，本程序必须非零退出）：
  · `A.1` ★★ **模拟与闭式一致**（`k` 网格上的 TTFT 与每步耗时；★ **近乎恒等**，见模块头）；
  · `A.2` ★★★ **`k*` 的三段性质**（`k<k*` / `k=k*` / `k>k*` 各自成立）；
  · `A.3` ★★★ **`k*` 的两条独立路径**（闭式 `⌈t_p/t_d⌉` vs 在模拟器上**扫描**）；
  · `A.4` ★★★ **免费守卫两端**（`k=1 ⇒ TTFT = t_p`；`k→∞ ⇒ TTFT` 无界）；
  · `B.1` ★★ **分块数权衡表**（每步耗时 / TTFT / decode 最坏单步）；
  · `B.2` ★★★ **`k>k*` 是纯亏**（每步耗时不再改善而 TTFT 线性涨）；
  · `B.3` ★★ **混跑批的构成**（1 个 prefill 块 + `C` 个 decode；`k=k*` 时两侧耗时相等）；
  · `C.1`–`C.4` 边界与反例（`k=1` / `k` 极大 / 两种「白亏」/ `t_p` 不被 `t_d` 整除）；
  · `D.4`/`D.5` ★★★ **守卫实验**：改坏模拟器 / 改坏闭式 ⇒ 断言必须失败。
"""
import ast
import collections
import inspect
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

TIME_EPS = 1e-9                      # ★ 浮点比较容差（本程序的两条路径在整除处**逐位精确**）

# ★★★ **本机实测值 / 推导值** —— 正文里的两张表都在这里**钉死**（`k` 网格的顺序同 `CHUNK_K_GRID`）。
PINNED_STEPS = (400.0, 200.0, 100.0, 50.0, 25.0, 20.0, 20.0, 20.0, 20.0)   # 每步耗时 = max(t_d, t_p/k)
PINNED_TTFT = (400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 640.0, 1280.0, 2560.0)  # TTFT = k × 每步耗时
PINNED_KSTAR = 20                    # ★ k* = ⌈400/20⌉
PINNED_KSTAR_NONDIV = 21             # ★ t_p 不被 t_d 整除时（401/20）的 k*
PINNED_TTFT_NONDIV = 420.0           # ★ 不整除时 k=k* 的 TTFT（= 21×20 > t_p=401）


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
# 1. 被测：**逐步推进**的离散事件模拟（每步 1 个 prefill 块 + C 个 decode）
# ===========================================================================
def sim_chunk(k, t_p, t_d, corrupt=None):
    """★★★ **被测实现**：把 prefill 切成 `k` 块，**逐步**推进，返回 `(TTFT, 每步耗时)`。

    ★ 与闭式 `closed` **无共享代码**（`D.8` 在源码层核）：
      这里是**一个 `for` 循环 + 累加器**，那里是**一组式子**。
    ★★ 模型（**本主题的模型假设**，见模块头）：
      · prefill 总工作量 `t_p` 被切成 `k` 块 ⇒ **每块必须走一步**；
      · 每步的**墙钟耗时 = 该步里两个负载的较大者** ——
        该步的 **decode 负载 = `t_d`**（`C` 个 decode 请求合起来），**prefill 负载 = `t_p/k`**；
      · **TTFT = 走完 `k` 步的累计墙钟**（prefill 全部完成的那一刻）。
    ★★ `corrupt`（**只用于守卫实验**）：
      · `"sum"`    —— 每步耗时写成 `t_d + t_p/k`（**和**而不是**取较大者**）；
      · `"half_p"` —— prefill 工作量只算一半（模拟「块数算错」）。
    """
    if k < 1:
        raise ValueError("k 必须 >= 1，收到 %r" % (k,))
    chunk = t_p / k
    if corrupt == "half_p":
        chunk = chunk / 2.0
    if corrupt == "sum":
        step = t_d + chunk
    else:
        step = max(t_d, chunk)
    ttft = 0.0
    for _ in range(k):
        ttft += step
    return ttft, step


# ===========================================================================
# 2. 参照：**闭式**（两条路径无共享代码）
# ===========================================================================
def step_closed(k, t_p, t_d):
    """每步耗时的闭式：`max(t_d, t_p/k)`。"""
    return max(t_d, t_p / k)


def ttft_closed(k, t_p, t_d):
    """TTFT 的闭式：`k × max(t_d, t_p/k)`。"""
    return k * step_closed(k, t_p, t_d)


def kstar_closed(t_p, t_d):
    """★★★ **最优分块数的闭式**：`k* = ⌈t_p / t_d⌉`（最小的 `k` 使每步耗时降到 `t_d`）。"""
    return int(math.ceil(t_p / t_d))


def kstar_scan(t_p, t_d, k_max):
    """★★★ **`k*` 的第二条独立路径**：**在模拟器上扫描**最小的 `k` 使每步耗时 `≤ t_d`。

    ★★ 它**不调用** `kstar_closed`，而是**逐档跑 `sim_chunk`** 再比较 —— 两条路径独立。
    """
    for k in range(1, k_max + 1):
        _, step = sim_chunk(k, t_p, t_d)
        if step <= t_d + TIME_EPS:
            return k
    return None


# ===========================================================================
# 3. A 类 —— ★★★ **最优分块数**
# ===========================================================================
def part_A():
    print("=" * 78)
    print("A · **最优分块数**：被测 = 逐步推进的模拟；参照 = 闭式（★ A.1 近乎恒等；**真两条路径在 A.3**）")
    print("=" * 78)
    t_p = specs.CHUNK_PREFILL_MS
    t_d = specs.CHUNK_DECODE_STEP_MS
    kg = specs.CHUNK_K_GRID
    print("  ★★ 模型：`每步耗时 = max(t_d, t_p/k)`、`TTFT = k × 每步耗时`；`t_p = %d`、`t_d = %d`"
          % (t_p, t_d))

    # ── A.1 ★★★ 模拟与闭式逐位一致 ─────────────────────────────────────
    print()
    print("  ── A.1 ★★ **`k` 网格上，模拟与闭式一致**（★ **近乎恒等**，见模块头）──")
    print("     %-6s %-14s %-14s %-14s %s"
          % ("k", "模拟 TTFT", "闭式 TTFT", "每步耗时", "一致"))
    sim_ttft = []
    sim_step = []
    all_match = True
    for k in kg:
        st, sp = sim_chunk(k, t_p, t_d)
        ct = ttft_closed(k, t_p, t_d)
        ok = abs(st - ct) < TIME_EPS and abs(sp - step_closed(k, t_p, t_d)) < TIME_EPS
        all_match = all_match and ok
        sim_ttft.append(st)
        sim_step.append(sp)
        print("     %-6d %-14.1f %-14.1f %-14.1f %s" % (k, st, ct, sp, "✓" if ok else "✗"))
    print("     ★★ 共 **%d 档**，全部逐位吻合 = %s" % (len(kg), all_match))
    check("A.1", all_match and len(kg) >= 5,
          "★★ **判据成立（★ 如实降级）**：**%d 档** `k` 上，`sim_chunk` 与闭式在 TTFT 与每步耗时上"
          "**一致** —— ★★ **但这近乎恒等**：`sim_chunk` 把 `step` **重复相加 `k` 次**、与 `k·step` "
          "是**同一个式子**；★ **真正的两条独立路径在 `A.3`**（闭式 vs 在模拟器上扫描）" % len(kg),
          "**模拟与闭式对不上** —— ★ 先查「每步走一块」的循环次数与 `max` 的取法")

    # ── A.2 ★★★ k* 的三段性质 ──────────────────────────────────────────
    print()
    print("  ── A.2 ★★★ **`k*` 的三段性质** ──")
    ks = kstar_closed(t_p, t_d)
    below, at_, above = [], [], []
    for k in kg:
        st, sp = sim_chunk(k, t_p, t_d)
        if k < ks:
            below.append((k, st, sp))
        elif k == ks:
            at_.append((k, st, sp))
        else:
            above.append((k, st, sp))
    print("     `k*` = **%d**；三段：`k<k*` %d 档 · `k=k*` %d 档 · `k>k*` %d 档"
          % (ks, len(below), len(at_), len(above)))
    ok_below = all(abs(st - t_p) < TIME_EPS and sp > t_d + TIME_EPS for (k, st, sp) in below)
    ok_at = len(at_) == 1 and abs(at_[0][1] - t_p) < TIME_EPS and abs(at_[0][2] - t_d) < TIME_EPS
    ok_above = all(abs(sp - t_d) < TIME_EPS and st > t_p + TIME_EPS for (k, st, sp) in above)
    print("     · `k<k*`：**TTFT 恒为 `t_p`（%.0f）** 且每步耗时 > `t_d` = %s" % (t_p, ok_below))
    print("     · `k=k*`：**TTFT = `t_p` 且每步耗时 = `t_d`**（两边同时取到最小）= %s" % ok_at)
    print("     · `k>k*`：每步耗时 = `t_d`（**不再改善**）且 **TTFT > `t_p`** = %s" % ok_above)
    print("     ★★★ **读法**：在**整除档**（本主题的 `t_d | t_p`）里，`k=k*` 是**帕累托最优点** ——")
    print("        它**同时**拿到最小的 TTFT 与最小的每步耗时（★ 不整除档两个目标会冲突，见 `C.4`）；")
    print("        `k<k*` 白亏 decode（TTFT 一点没省），`k>k*` 白亏 TTFT（每步耗时一点没省）。")
    check("A.2", ok_below and ok_at and ok_above and len(below) >= 2 and len(above) >= 2,
          "★★★ **判据成立**：`k*` 的三段**全部成立** —— `k<k*`（%d 档）TTFT 恒 `%.0f` 而 decode 被拖慢；"
          "`k=k*`（%d）两侧同时取最小；`k>k*`（%d 档）每步耗时封顶 `%.0f` 而 TTFT 上升"
          % (len(below), t_p, len(at_), len(above), t_d),
          "**三段性质被破坏** —— ★ 先查 `k*` 的取值与 `max(t_d, t_p/k)` 的分界")

    # ── A.3 ★★★ k* 的两条独立路径 ──────────────────────────────────────
    print()
    print("  ── A.3 ★★★ **`k*` 的两条独立路径**：闭式 `⌈t_p/t_d⌉` vs 在模拟器上**扫描** ──")
    ks_closed = kstar_closed(t_p, t_d)
    ks_scan = kstar_scan(t_p, t_d, specs.CHUNK_SCAN_MAX)
    print("     闭式 `⌈t_p/t_d⌉` = **%d**；★ 扫描（最小的 `k` 使每步耗时 ≤ `t_d`）= **%d**"
          % (ks_closed, ks_scan))
    # ★ 第二条：不整除的一档（t_p 不被 t_d 整除）
    np_, nd_ = specs.CHUNK_PREFILL_NONDIV_MS, specs.CHUNK_DECODE_STEP_MS
    ks_n_c = kstar_closed(np_, nd_)
    ks_n_s = kstar_scan(np_, nd_, specs.CHUNK_SCAN_MAX)
    print("     `t_p=%d, t_d=%d`（**不整除**）：闭式 = **%d** · 扫描 = **%d**" % (np_, nd_, ks_n_c, ks_n_s))
    check("A.3", ks_closed == ks_scan == PINNED_KSTAR
          and ks_n_c == ks_n_s == PINNED_KSTAR_NONDIV,
          "★★★ **判据成立**：两条路径**都给出 %d**（整除档）；★ **不整除**档 `t_p=%d` 两条都给 **%d** ——"
          " ★★ 一条是 `ceil`、一条是**逐档跑模拟再比较**" % (PINNED_KSTAR, np_, PINNED_KSTAR_NONDIV),
          "**`k*` 的两条路径不一致** —— ★ 先查 `ceil` 的方向与扫描的比较符（`≤ t_d`）")

    # ── A.4 ★★★ 免费守卫两端 ───────────────────────────────────────────
    print()
    print("  ── A.4 ★★★ **免费守卫两端**：`k=1 ⇒ TTFT = t_p`；`k→∞ ⇒ TTFT` 无界 ──")
    t1, s1 = sim_chunk(1, t_p, t_d)
    big = specs.CHUNK_K_GRID[-1]
    tb, sb = sim_chunk(big, t_p, t_d)
    huge = specs.CHUNK_HUGE_K
    th, sh = sim_chunk(huge, t_p, t_d)
    print("     `k=1` ⇒ TTFT = **%.1f**（= `t_p` = %d：**退化成不分块**）· 每步耗时 = %.1f" % (t1, t_p, s1))
    print("     `k=%d` ⇒ TTFT = **%.0f**（线性上升）" % (big, tb))
    print("     `k=%d` ⇒ TTFT = **%.0f** ≈ `k·t_d` = %d（**无界**：每步都要等一个 decode 步）"
          % (huge, th, huge * t_d))
    check("A.4", abs(t1 - t_p) < TIME_EPS and abs(t1 - PINNED_TTFT[0]) < TIME_EPS
          and th > tb > t_p and abs(th - huge * t_d) < TIME_EPS and abs(sh - t_d) < TIME_EPS,
          "★★★ **两端守卫都成立**：`k=1` ⇒ TTFT **恰好 = `t_p`**（不分块的退化，一档一块）；"
          "`k=%d` ⇒ TTFT **%.0f** ≈ `k·t_d`（★ **线性无界**）" % (huge, th),
          "**两端的免费守卫被破坏** —— ★ 先查 `k=1` 的循环与 `k` 极大时的每步耗时")


# ===========================================================================
# 4. B 类：分块数权衡表 · 纯亏 · 混跑批构成
# ===========================================================================
def part_B():
    print()
    print("=" * 78)
    print("B · **分块数的权衡**：每步耗时 / TTFT / decode 最坏单步")
    print("=" * 78)
    t_p = specs.CHUNK_PREFILL_MS
    t_d = specs.CHUNK_DECODE_STEP_MS
    kg = specs.CHUNK_K_GRID
    ks = kstar_closed(t_p, t_d)

    # ── B.1 ★★ 权衡表 ───────────────────────────────────────────────────
    print()
    print("  ── B.1 ★★ **权衡表**（★ 每一格都被 `A.1`/`PINNED_*` 钉住，不留「自己那一列」）──")
    print("     %-6s %-12s %-12s %-16s %s" % ("k", "每步耗时", "TTFT", "decode 最坏单步", "评"))
    steps, ttfts = [], []
    for i, k in enumerate(kg):
        st, sp = sim_chunk(k, t_p, t_d)
        steps.append(sp)
        ttfts.append(st)
        tag = ("帕累托最优" if k == ks else ("拖慢 decode" if k < ks else "TTFT 纯涨"))
        print("     %-6d %-12.1f %-12.1f %-16.1f %s" % (k, sp, st, sp, tag))
    print("     ★★ **读法**：`k<k*` 每步耗时 > `t_d`（decode 被拖慢）而 TTFT 卡在 `t_p`；")
    print("        `k=k*` 两侧同时取最小；`k>k*` 每步耗时封顶 `t_d` 而 **TTFT 线性上升**。")
    check("B.1", tuple(steps) == PINNED_STEPS and tuple(ttfts) == PINNED_TTFT,
          "★★ **判据成立**：**整张表**（%d 行：每步耗时 + TTFT）与钉住值**逐元素相同** ——"
          " ★★ 包括 `k=k*` 那一行的 `%.0f/%0f`（★ 不是只钉「自己那一列」）" % (len(kg), t_d, t_p),
          "**权衡表的某一格被动过** —— ★ 先查 `CHUNK_K_GRID` 与两个耗时常量")

    # ── B.2 ★★★ k>k* 是纯亏 ────────────────────────────────────────────
    print()
    print("  ── B.2 ★★★ **`k>k*` 是纯亏**：每步耗时不再改善，TTFT 线性上升 ──")
    print("     %-8s %-12s %-12s %s" % ("k", "每步耗时", "TTFT", "相对 k* 的 TTFT 倍数"))
    for k in (ks, 2 * ks, 4 * ks):
        st, sp = sim_chunk(k, t_p, t_d)
        print("     %-8d %-12.1f %-12.1f %.2f×" % (k, sp, st, st / ttft_closed(ks, t_p, t_d)))
    extra = ttft_closed(4 * ks, t_p, t_d) - ttft_closed(ks, t_p, t_d)
    print("     ★★★ **读法**：`k` 从 `%d` 涨到 `%d`（4 倍），**每步耗时一模一样（%.0f）**，"
          "而 TTFT 从 `%.0f` 涨到 `%.0f`（多出 `%.0f` = `(k−k*)·t_d`）——★ **纯亏**。"
          % (ks, 4 * ks, t_d, ttft_closed(ks, t_p, t_d), ttft_closed(4 * ks, t_p, t_d), extra))
    check("B.2", abs(step_closed(4 * ks, t_p, t_d) - t_d) < TIME_EPS
          and abs(extra - 3.0 * ks * t_d) < TIME_EPS,
          "★★★ **判据成立**：超过 `k*` 后每步耗时**封顶 `%.0f`**，多余的 `k` **只把 TTFT 线性推高**"
          "（`k=4k*` 时多出 **%.0f** = `(4k*−k*)·t_d`）⇒ ★★ **`k>k*` 是无收益的代价**" % (t_d, extra),
          "**`k>k*` 的纯亏没有跑出来** —— ★ 先查 `k*` 与每步耗时的封顶")

    # ── B.3 ★★ 混跑批的构成 ────────────────────────────────────────────
    print()
    print("  ── B.3 ★★ **与 decode 混跑时，一批里有什么** ──")
    C = specs.CHUNK_CONCURRENT_DECODE
    print("     ★ 每步的批构成 = **1 个 prefill 块**（`t_p/k`）+ **%d 个 decode 请求**（合起来 `t_d`）" % C)
    print("     %-6s %-16s %-16s %s" % ("k", "prefill 负载", "decode 负载", "谁支配该步"))
    for k in (1, 4, ks, 2 * ks):
        chunk = t_p / k
        who = ("prefill" if chunk > t_d + TIME_EPS else
               ("decode" if chunk < t_d - TIME_EPS else "**相等**"))
        print("     %-6d %-16.1f %-16.1f %s" % (k, chunk, float(t_d), who))
    print("     ★★ **读法**：`k=k*` 时 **prefill 块耗时恰好 = decode 耗时 = `t_d`** ⇒")
    print("        该步**两侧都不浪费**（prefill 没把 decode 拖慢，decode 也没浪费 prefill 的槽位）；")
    print("        `k<k*` 时 prefill 块**支配**该步（decode 被拖慢），`k>k*` 时 decode **支配**（prefill 块空转）。")
    check("B.3", abs(t_p / ks - t_d) < TIME_EPS and t_p / 1 > t_d and t_p / (2 * ks) < t_d
          and abs(C - 32) < 0.5,
          "★★ **判据成立**：三种 `k` 下该步的**支配方**不同 —— `k=k*` 时 **prefill 负载(%d) == decode 负载(%d)**"
          "（平衡点）；★ 这也是 `k>k*` 时 prefill 块**空转**的机理" % (t_p // ks, t_d),
          "**混跑批的构成被动过** —— ★ 先查 `CHUNK_CONCURRENT_DECODE` 与 `max` 的分界")


# ===========================================================================
# 5. C 类：边界与反例（两种「白亏」· 不整除）
# ===========================================================================
def part_C():
    print()
    print("=" * 78)
    print("C · 边界与反例：两种白亏 · 不分块 · `t_p` 不被 `t_d` 整除")
    print("=" * 78)
    t_p = specs.CHUNK_PREFILL_MS
    t_d = specs.CHUNK_DECODE_STEP_MS
    ks = kstar_closed(t_p, t_d)

    # ── C.1 ★★★ 「块越多越好」的反例 ───────────────────────────────────
    print()
    print("  ── C.1 ★★★ **反例：「块越多越好」** —— `k>k*` 时 TTFT 不降反升 ──")
    st_ks, sp_ks = sim_chunk(ks, t_p, t_d)
    k_big = specs.CHUNK_K_GRID[-1]
    st_big, sp_big = sim_chunk(k_big, t_p, t_d)
    print("     `k=%d`（`k*`）⇒ TTFT = **%.0f** · 每步耗时 = **%.0f**" % (ks, st_ks, sp_ks))
    print("     `k=%d`（更多块）⇒ TTFT = **%.0f** · 每步耗时 = **%.0f**" % (k_big, st_big, sp_big))
    print("     ★★★ **读法**：块更多（`%d` vs `%d`）而 **TTFT 反而涨 %.1f 倍**，每步耗时**一点没降** ——"
          % (k_big, ks, st_big / st_ks))
    print("        ★★ **「分得更细」不等于「更快」**：`k*` 之后每步耗时已经撞到 `t_d` 的地板。")
    check("C.1", st_big > st_ks + TIME_EPS and abs(sp_big - sp_ks) < TIME_EPS,
          "★★★ **判据成立**：`k` 从 `%d` 涨到 `%d`，TTFT 从 **%.0f 涨到 %.0f**（**不降反升**），"
          "而每步耗时**都停在 %.0f** —— ★★ 「块越多越好」**在此处为假**"
          % (ks, k_big, st_ks, st_big, sp_ks),
          "**「块更多反而更慢」没有跑出来** —— ★ 先查 `k*` 与每步耗时的封顶")

    # ── C.2 ★★ 反向白亏：k<k* 拖慢 decode 而 TTFT 不变 ─────────────────
    print()
    print("  ── C.2 ★★ **另一种白亏**：`k<k*` 拖慢 decode，而 TTFT **一点没省** ──")
    k_lo = 1
    st_lo, sp_lo = sim_chunk(k_lo, t_p, t_d)
    k_mid = ks // 5 if ks >= 5 else 1
    st_mid, sp_mid = sim_chunk(k_mid, t_p, t_d)
    print("     `k=%d`（不分块）⇒ TTFT = %.0f · **每步耗时 = %.0f（`t_d` 的 %.0f 倍）**"
          % (k_lo, st_lo, sp_lo, sp_lo / t_d))
    print("     `k=%d` ⇒ TTFT = %.0f · 每步耗时 = %.0f" % (k_mid, st_mid, sp_mid))
    print("     `k=%d`（`k*`）⇒ TTFT = %.0f · 每步耗时 = %.0f" % (ks, st_ks_of(t_p, t_d, ks), t_d))
    print("     ★★★ **读法**：三个 `k` 的 **TTFT 完全一样**（都是 `t_p`=%.0f），"
          "★ 但每步耗时差 —— `k=1` 相对 `k=k*=%d` 差 **%.0f 倍**（%.0f/%.0f），"
          "`k=1` 相对 `k=%d` 差 **%.0f 倍**（%.0f/%.0f）"
          % (t_p, ks, sp_lo / t_d, sp_lo, t_d, k_mid, sp_lo / sp_mid, sp_lo, sp_mid))
    print("        ⇒ ★★ **`k` 太小，代价全落在 decode 用户身上**（每一步都被一个长 prefill 块挡住）。")
    check("C.2", abs(st_lo - st_mid) < TIME_EPS and abs(st_mid - t_p) < TIME_EPS
          and sp_lo > t_d and sp_mid > t_d and sp_lo / sp_mid > 1.5
          and abs(sp_lo / t_d - 20.0) < TIME_EPS and abs(sp_lo / sp_mid - 4.0) < TIME_EPS,
          "★★ **判据成立**：`k=1/%d/%d` 的 **TTFT 全等于 `t_p`**，而每步耗时从 **%.0f 降到 %.0f** ——"
          " ★★ `k=1` 相对 `k=k*` 差 **%.0f 倍**（`400/20`）、相对 `k=%d` 差 **%.0f 倍**（`400/100`）"
          " ⇒ ★★ **TTFT 对 `k<k*` 完全不敏感**，代价**全在 decode 的每步延迟**上"
          % (k_mid, ks, sp_lo, sp_mid, sp_lo / t_d, k_mid, sp_lo / sp_mid),
          "**`k<k*` 的双向性质没有跑出来** —— ★ 先查 TTFT 是否恒为 `t_p`")

    # ── C.3 ★★ 边界：k 极大 ⇒ 无界 ─────────────────────────────────────
    print()
    print("  ── C.3 ★★ **边界**：`k` 极大 ⇒ TTFT 无界（`k·t_d`）──")
    kk = (1, 1000, 1000000)
    vals = [sim_chunk(k, t_p, t_d) for k in kk]
    print("     " + " · ".join("`k=%d ⇒ TTFT=%.0f`" % (k, v[0]) for k, v in zip(kk, vals)))
    print("     ★★ **读法**：TTFT 随 `k` **线性无界** —— ★ 每细分一倍，就多等一个完整的 decode 步。")
    check("C.3", abs(vals[1][0] - 1000 * t_d) < TIME_EPS
          and abs(vals[2][0] - 1000000 * t_d) < TIME_EPS
          and vals[2][0] > vals[1][0] > vals[0][0],
          "★★ **判据成立**：`k=10^6` ⇒ TTFT = **%.0f** = `k·t_d`（★ **无界**）——"
          " ★★ 这给出 `k` 的**上界直觉**：没有理由超过 `k*`" % vals[2][0],
          "**`k` 极大时的线性上升没有跑出来** —— ★ 先查每步耗时的封顶")

    # ── C.4 ★★ 不整除：k* 取 ceil，TTFT 略高于 t_p ────────────────────
    print()
    print("  ── C.4 ★★ **`t_p` 不被 `t_d` 整除**时，`k*` 仍取 `⌈·⌉`，但 TTFT **略高于** `t_p` ──")
    np_ = specs.CHUNK_PREFILL_NONDIV_MS
    ks_n = kstar_closed(np_, t_d)
    st_n, sp_n = sim_chunk(ks_n, np_, t_d)
    st_n1, sp_n1 = sim_chunk(ks_n - 1, np_, t_d)
    print("     `t_p=%d`（**不被 %d 整除**）⇒ `k* = ⌈%d/%d⌉ = %d`" % (np_, t_d, np_, t_d, ks_n))
    print("     `k=%d` ⇒ TTFT = **%.0f**（> `t_p`=%d）· `k=%d` ⇒ TTFT = **%.0f**"
          % (ks_n, st_n, np_, ks_n - 1, st_n1))
    print("     ★★ **读法**：不整除时**没有**任何 `k` 能同时取到两侧的最小值 ——")
    print("        `k=%d`（`k*`）拿到**最小的每步耗时 `%.0f`**，但 TTFT 为 `%.0f`（多付 `%.0f`）；"
          % (ks_n, sp_n, st_n, st_n - np_))
    print("        `k=%d` 拿到**最小的 TTFT `%.0f`**（= `t_p`），但每步耗时 `%.2f > t_d`。"
          % (ks_n - 1, st_n1, sp_n1))
    print("     ★★★ **如实声明**：**整除档（本主题的 `t_p=%d`）里 `k*` 是唯一的帕累托最优点**；"
          % t_p)
    print("        而**不整除档里两个目标冲突** ⇒ `k*` 只是「**每步耗时先触底**」的那一点。")
    check("C.4", ks_n == PINNED_KSTAR_NONDIV and st_n > np_ + TIME_EPS
          and abs(st_n - PINNED_TTFT_NONDIV) < TIME_EPS and abs(sp_n - t_d) < TIME_EPS
          and abs(st_n1 - np_) < 1e-6 and st_n1 < st_n and sp_n1 > t_d + TIME_EPS,
          "★★ **判据成立**：不整除档 `t_p=%d` ⇒ `k*=%d`、每步耗时 **%.0f**（触底）但 TTFT **%.0f**；"
          "★ `k=%d` 的 TTFT 恰好 **%.0f**（= `t_p`）而每步耗时 **%.2f > t_d** —— "
          "★★ **两个目标冲突**（`A.2` 的帕累托最优只对**整除档**成立）"
          % (np_, ks_n, sp_n, st_n, ks_n - 1, st_n1, sp_n1),
          "**不整除档的边界被动过** —— ★ 先查 `ceil` 与 `CHUNK_PREFILL_NONDIV_MS`")


def st_ks_of(t_p, t_d, ks):
    """辅助：`k=k*` 时的 TTFT（供打印，不参与判据）。"""
    return ttft_closed(ks, t_p, t_d)


# ===========================================================================
# 6. 前哨：口径 / 载体 / 标记自检（★ 跑在任何计算之前）
# ===========================================================================
def preflight():
    """★★ 口径 / 载体 / 标记自检**一律在这里**，且**跑在任何计算之前**（本库 `P23`）。"""
    print("=" * 78)
    print("D0. 口径/载体/标记钉住（**先跑**：让「改常量」变成断言捕获，而不是崩溃捕获）")
    print("=" * 78)
    print("  本程序的失败标记 ASSERT_FAIL_MARK = %r" % ASSERT_FAIL_MARK)
    print("  ★★★ 限定语：**不是任何真实引擎，也不是真机** —— 本机**无 GPU / 无 vLLM**；")
    print("      ★★ 但**本程序的 A 类是真的**：**分块数权衡的两条独立路径**。")
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

    # ★★★ 口径守卫（★ 跑在任何计算之前）：三段性质要求 `k*` **落在网格内部**
    #   （即网格必须**同时**含 `k<k*`、`k=k*`、`k>k*` 三档）；且 `t_d` 必须整除 `t_p`
    #   （整除档是「TTFT 恰好回落到 `t_p`」这一结论的落点）。★ 否则 `A.2` 会退化成恒真。
    ks = kstar_closed(specs.CHUNK_PREFILL_MS, specs.CHUNK_DECODE_STEP_MS)
    ok_grid = (ks in specs.CHUNK_K_GRID
               and any(k < ks for k in specs.CHUNK_K_GRID)
               and any(k > ks for k in specs.CHUNK_K_GRID)
               and specs.CHUNK_PREFILL_MS % specs.CHUNK_DECODE_STEP_MS == 0
               and specs.CHUNK_K_GRID[0] == 1
               and specs.CHUNK_K_GRID[-1] > ks)
    print("     ★★ 口径守卫：`k*=%d` 在网格内 = %s；网格含 `k<k*` = %s、含 `k>k*` = %s；"
          "`t_d | t_p` = %s；网格含 `k=1` = %s"
          % (ks, ks in specs.CHUNK_K_GRID,
             any(k < ks for k in specs.CHUNK_K_GRID),
             any(k > ks for k in specs.CHUNK_K_GRID),
             specs.CHUNK_PREFILL_MS % specs.CHUNK_DECODE_STEP_MS == 0,
             specs.CHUNK_K_GRID[0] == 1))
    check("D.3b", ok_grid,
          "★★★ **口径守卫**：网格**必须同时含** `k<k*`/`k=k*`/`k>k*` 三档且含 `k=1`，"
          "`k*` 还必须落在网格内 —— ★ 否则 `A.2` 的三段性质会**退化成恒真**"
          "（★ 而且**必须有一档 `t_p` 不被 `t_d` 整除**的独立算例，见 `CHUNK_PREFILL_NONDIV_MS`）",
          "**网格的覆盖被动过** —— ★ 先查 `CHUNK_K_GRID` / `CHUNK_PREFILL_MS` / `CHUNK_DECODE_STEP_MS`")
    return len(FAILS) == before


# ===========================================================================
# 7. D 类（后半）：守卫实验 · 引文逐字 · 逐元素 · 独立性
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
    t_p = specs.CHUNK_PREFILL_MS
    t_d = specs.CHUNK_DECODE_STEP_MS

    # ── D.4 ★★★ 守卫实验 ①：把模拟器的每步耗时改坏（`max` 写成「和」）─────
    print()
    print("  ── D.4 ★★★ **守卫实验 ①**：把模拟器的每步耗时从 `max(t_d, t_p/k)` 改成 `t_d + t_p/k` ──")
    k = kstar_closed(t_p, t_d)
    g_sim, g_sp = sim_chunk(k, t_p, t_d)
    g_bad, g_bsp = sim_chunk(k, t_p, t_d, corrupt="sum")
    g_ref = ttft_closed(k, t_p, t_d)
    print("     `k=%d`：正常 = **%.0f**（= 闭式 %.0f）· 坏法（和） = **%.0f**" % (k, g_sim, g_ref, g_bad))
    print("     ★★★ **读法**：「取较大者」与「求和」只在**两个负载相等**时给出同一个数 ——")
    print("        而 `k=k*` **恰好**是它们相等的那一点 ⇒ ★★ 这正是 `k*` 的物理意义。")
    check("D.4", abs(g_sim - g_ref) < TIME_EPS and g_bad > g_ref + TIME_EPS,
          "★★★ **守卫成立**：把每步耗时改成「求和」后，模拟器从 **%.0f**（= 闭式）变成 **%.0f** ——"
          " ★★ 于是 `A.1` 的「逐位一致」对「**取较大者**（不是求和）」这条规则**是敏感的**" % (g_sim, g_bad),
          "**守卫失效** —— ★ 改成求和后模拟器居然仍等于闭式（那 `A.1` 抓不到它）")

    # ── D.5 ★★★ 守卫实验 ②：把闭式的【ceil】改成【floor】 ──────────────
    print()
    print("  ── D.5 ★★★ **守卫实验 ②**：把 `k* = ⌈t_p/t_d⌉` 改成 `⌊t_p/t_d⌋`（**向下取整**）──")
    np_ = specs.CHUNK_PREFILL_NONDIV_MS                  # 401 / 20 ⇒ 必然暴露 ceil vs floor
    ks_ceil = int(math.ceil(np_ / t_d))
    ks_floor = int(np_ // t_d)
    st_ceil, sp_ceil = sim_chunk(ks_ceil, np_, t_d)
    st_floor, sp_floor = sim_chunk(ks_floor, np_, t_d)
    print("     `t_p=%d, t_d=%d`：`ceil` = **%d**（每步耗时 %.2f）· `floor` = **%d**（每步耗时 %.2f）"
          % (np_, t_d, ks_ceil, sp_ceil, ks_floor, sp_floor))
    print("     ★★★ **读法**：向下取整的 `k` **不够** ⇒ 每步耗时 **> `t_d`** ⇒ 它**不是** `k*`。")
    check("D.5", ks_ceil == PINNED_KSTAR_NONDIV and ks_floor == ks_ceil - 1
          and sp_ceil <= t_d + TIME_EPS and sp_floor > t_d + TIME_EPS,
          "★★★ **守卫成立**：`floor` 给出的 `k=%d` 每步耗时 **%.2f > t_d=%d**（不满足定义）——"
          " ★★ `A.3` 的「闭式 == 扫描」对**取整方向**是敏感的" % (ks_floor, sp_floor, t_d),
          "**守卫失效** —— ★ 向下取整居然也满足每步耗时 ≤ t_d（那 `A.3` 抓不到它）")

    # ── D.6 ★★★ 引文逐字：chunked prefill 的默认值（**带两个限定**）──────
    check("D.6", specs.VLLM_ENABLE_CHUNKED_PREFILL is True
          and specs.VLLM_CHUNKED_PREFILL_SOURCE_QUOTE == "enable_chunked_prefill: bool = True"
          and specs.VLLM_CHUNKED_PREFILL_DEFAULT_QUOTE == "In V1, chunked prefill is enabled by default whenever possible.",
          "★★★ **引文逐字钉住**：源码常量 `enable_chunked_prefill: bool = True`（`[B]`）；"
          "★ 文档逐字 **「In V1, chunked prefill is enabled by default whenever possible.」** ——"
          " ★★ **「默认开启」必须带「V1 / whenever possible」两个限定**（`F6`）",
          "**chunked prefill 的常量/引文被动过** —— ★ 先查源码常量与文档逐字")

    # ── D.7 ★★★ 逐元素钉住 ──────────────────────────────────────────────
    check("D.7", specs.SCHED_FETCH_DAY == "2026-10-09"
          and specs.CHUNK_PREFILL_MS == 400 and specs.CHUNK_DECODE_STEP_MS == 20
          and specs.CHUNK_CONCURRENT_DECODE == 32
          and specs.CHUNK_K_GRID == (1, 2, 4, 8, 16, 20, 32, 64, 128)
          and specs.CHUNK_SCAN_MAX == 4096
          and specs.CHUNK_PREFILL_NONDIV_MS == 401
          and specs.CHUNK_HUGE_K == 1000000
          and specs.UNSOURCED == ()
          and abs(TIME_EPS - 1e-9) < 1e-18
          and PINNED_STEPS == (400.0, 200.0, 100.0, 50.0, 25.0, 20.0, 20.0, 20.0, 20.0)
          and PINNED_TTFT == (400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 640.0, 1280.0, 2560.0)
          and PINNED_KSTAR == 20 and PINNED_KSTAR_NONDIV == 21
          and abs(PINNED_TTFT_NONDIV - 420.0) < 1e-9,
          "★★★ **规格常量逐元素 + 程序内常量**钉住（`t_p` · `t_d` · 并发 decode · `k` 网格 ·"
          " 扫描上限 · 不整除档 · 极大 `k` · 两整张表 · 两个 `k*`）；★★ `UNSOURCED == ()` 一并钉住",
          "**某个常量被动过**")

    # ── D.8 ★★★ 独立性（源码级）────────────────────────────────────────
    sn = _names_in(sim_chunk)
    cn = _names_in(step_closed) | _names_in(ttft_closed) | _names_in(kstar_closed)
    print()
    print("  ── D.8 ★★ **源码级线索**（★ **不是证明**）：被测与参照互不引用 ──")
    print("     `sim_chunk` 里出现任一闭式名 = %s ；闭式里出现 `sim_chunk` = %s"
          % (sorted(n for n in sn if n.startswith(("step_closed", "ttft_closed", "kstar_closed"))),
             "sim_chunk" in cn))
    print("     ★★★ **降级措辞**：这只是「**两段代码不互相引用**」的**源码级线索**，"
          "**不证明**判据有信息量 —— 后者由 `A.1`/`A.3` 与 `D.4`/`D.5` 的**行为**兜底。")
    check("D.8", not any(n in sn for n in ("step_closed", "ttft_closed", "kstar_closed"))
          and "sim_chunk" not in cn and "kstar_scan" not in cn,
          "★★ **源码级线索成立**：**被测（逐步模拟）与参照（闭式）互不引用**（★ **两侧都审**）；"
          "★ `kstar_scan`（第二条路径）也**只**调模拟器 —— ★★ 「两条路径一致」不是恒真",
          "**两实现互相引用** —— A 类可能退化成自证（`§1.3.2b`）")

    # ── D.9 ★★ 边界：每步耗时的单调不增 ────────────────────────────────
    ks = kstar_closed(t_p, t_d)
    steps = [step_closed(k, t_p, t_d) for k in range(1, 4 * ks + 1)]
    noninc = all(steps[i] >= steps[i + 1] - TIME_EPS for i in range(len(steps) - 1))
    print()
    print("  ── D.9 ★★ **每步耗时随 `k` 单调不增**，且在 `k*` 处**触底**（不再下降）──")
    print("     `k=1..%d` 的每步耗时：%s ... 在 `k=%d` 之后恒为 **%.0f**"
          % (4 * ks, ["%.0f" % s for s in steps[:6]], ks, t_d))
    check("D.9", noninc and abs(steps[-1] - t_d) < TIME_EPS and abs(steps[ks - 1] - t_d) < TIME_EPS
          and steps[0] == t_p,
          "★★ **边界成立**：每步耗时**单调不增**、在 `k=%d` 处**触底 `%.0f`** 且此后**不再下降**；"
          "`k=1` 时它是 **%.0f**（= `t_p`）—— ★★ 于是 `k*` 是**「还在改善」与「不再改善」的分界**"
          % (ks, t_d, steps[0]),
          "**每步耗时的单调性被动过** —— ★ 先查 `max(t_d, t_p/k)`")


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
    print("★★ **A = %d/%d** —— ★★★ **本程序有 A 类**：`k*` 的**两条独立路径**"
          "（**闭式 vs 在模拟器上扫描**，`A.3`；★ `A.1` 近乎恒等）。"
          % (_PASS["A"], _TOTAL["A"]))
    print("★ 判据来源：本主题的调度模型是 **`[E]` 级（本库实现 / 假设）**；"
          "chunked prefill 的默认开关是 `[B]` 级引用值。")
    print("★ 本程序只核对「**一块的账**」——**不是**任何真实引擎的 scheduler，**也不碰** kernel 侧。")
    return 1 if FAILS else 0


def main():
    print("=" * 78)
    print("02_chunked_prefill.py —— chunked prefill 的分块数权衡：k* = ⌈t_p/t_d⌉")
    print("=" * 78)
    print("  ★ 失败标记 ASSERT_FAIL_MARK = %r（本主题自定义；不跨主题照抄）" % ASSERT_FAIL_MARK)
    print("  ★★★ 定义不在本文：`chunked prefill` 的定义在")
    print("      `10-foundations/performance-models/01-roofline-model.md:251`；")
    print("      prefill/decode 的 kernel 差异在 `40-kernels/attention/05:229-236`（**只回链**）。")
    print("  ★★★ 模型假设：**一步耗时 = max(t_d, t_p/k)**（★ 真实里 prefill 与 decode 共享算力）。")
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

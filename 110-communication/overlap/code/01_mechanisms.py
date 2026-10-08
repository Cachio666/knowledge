#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""01_mechanisms.py —— **双缓冲 / 通信分块 / 异步发起**：三个机制，与它们各自要求的条件。

配套文档：`01-mechanisms.md`。

★ 三类断言**全部在本程序内部**，失败时以**非零退出码**结束；**没有 `verify.py`**。

★★★ **本层是【时间层】**（通信-计算重叠的收益），**不是**算法层（`collectives`）也**不是**实现层（`nccl`），
  **也不是**物理层（`network-topology`）。★ 时间层的判据对象是「**两条时间轴排起来要多久**」。

★★★ **本主题【从零建】**：全库**没有**「通信-计算重叠」的正文，只有上游的让渡句
  （`100-parallelism/data-parallel/02-gradient-synchronization.md:65` 逐字把「**「能藏多少」**」
  让渡给本主题，并明说「本主题**不给这条判据**」）。★ 判据的正文在 `02-how-much-can-hide`，
  本程序只做**机制层**的那一半。

★★★ **本主题的 A 类【可以真做】，而且参照是【独立路径】**：
  · **被测** = 闭式 `cf_pipeline(C,T,K) = (C+T)/K + (K−1)·max(C/K, T/K)`（**本库 `[E]` 级推导**）；
  · **参照** = **时隙模拟** `sim_ticks(C,T,K,corrupt)` —— ★★ 它把 `K` 个块在**两条时间轴**
    （计算引擎 + 通信引擎）上**逐拍排出来**，**数**出完工时刻（★ 一个**调度器**，不是一条公式）。
  ★★★ **两条独立路径**：模拟器的**控制流**里**没有**那个闭式（`D.8` 在源码层核），
    ★ 于是「模拟 == 闭式」**不是恒真**，而是**两条独立路径的会合**（★ 这正是 `§1.3.2b` 要守的那条）。

★★★ **一条 §1.3.2b 的自查**（本程序最要紧的设计说明）：
  本域的知识对象**就是闭式** ⇒ 「把闭式（或它的等价重写）当参照」是**系统性诱惑**。
  本程序的做法是把参照换成 **tick 级的调度模拟**：
    · 两条引擎**各自**逐拍推进（`comp_progress` / `comm_progress` 两个计数器）；
    · 通信引擎只在「**这一块的通信还没做完、且这一块的计算【在上一拍就】已经做完**」时推进；
    · 循环的**拍数由数据驱动**（`while ki < K`，`ki` 是「已发完的块数」）—— **不是**从闭式取的。
  ★★ **残余的诚实交代**：模拟器**用** `C` / `T` / `K` 三个**输入参数**（它们是题设，不是被测量），
    但**一个都不出现在闭式里**地参与运算 —— ★ `D.8` 在源码层核「模拟器不调用任何 `cf_*`」。

★★ **覆盖边界**：本程序守的是「**机制在时间轴上排出来的结果**」与「**错式会不会被拒**」。
  **不守**任何一条**真实网络的性能**（本机**无 GPU / 无 NCCL / 无真网卡**）。
  ★★ **最强的限定语**：**「不是 NCCL，也不是真机」**。

★★ 可证伪断言（把下面任一条改错，本程序必须非零退出）：
  · `A.1` ★★★ **时隙模拟 == 闭式**（`OVERLAP_C_T_GRID` × `OVERLAP_K_GRID`，20 档）；
  · `A.2` ★★ 模拟器过**三条免费守卫的前两条**（`K = 1` ⇒ **逐位**等于 `C+T`；中间值 ≥ 下界）；
  · `B.1` ★★ 代价账表（`(C,T,K) → 时间`）；`B.2` ★★ **三个机制各自要求的条件**（四条，逐条可数）；
  · `C.1` ★★★ **`P35` 的现场**：错式 `C + T/K` 在 `T > C` 时**被守卫拒绝**（★ 真值放行在 `A.2`）；
  · `D.4` ★★★ **守卫实验**：把模拟器的**依赖**改坏 ⇒ 模拟结果**变**（且**掉到物理下界**）；
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
# ★★★ 失败标记：本主题（overlap）取 `+= FAIL: overlap`
#   （★ 三个程序**同一个**标记；本域另外三个主题各用 `&&` / `><` / `,,`，**两两不互为前缀**）。
#   ★★ `+` 在 **shell 里通常需引号**、在**正则里才是元字符** ⇒ 它**只允许出现在 `print()` 的字符串里**。
#   ★ `mutate.py` 用 **AST** 从本文件读出这个值（不得写死）；`D.1`/`D.2` 两条断言守着它。
ASSERT_FAIL_MARK = "+= FAIL: overlap"

# 本域另外三个主题的失败标记 —— **只用来验「不照抄」**，本程序不打印它们。
OTHER_THEME_MARKS = ("&& FAIL: collectives", ">< FAIL: nccl", ",, FAIL: network-topology")

FAILS = []
MARKS_USED = set()
_TOTAL = collections.Counter()
_PASS = collections.Counter()


def check(name, cond, detail="", detail_fail=None):
    """`name` 形如 `B.1` —— **首字母就是类别**（A/B/C/D），与 `CONTRIBUTING` §4 对齐。

    ★★★ **失败路径【不复述结论】**：`detail` 是**判据成立时**的措辞 ⇒ 失败走**另一版文案**。
    ★ `mutate.py` 的分类是「**行首锚定**」的 ⇒ 失败标记必然在**行首**。
    """
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
# 1. 被测：闭式（**只有这些**；它们**不得**出现在模拟器里 —— `D.8` 在源码层核）
# ===========================================================================
def cf_no_overlap(C, T):
    """**无重叠**值：两条时间轴**串起来**（★ 独立表达式，不是任何流水模型）。"""
    return Fraction(C + T)


def cf_lower_bound(C, T):
    """**物理下界** `max(C, T)`：总时间**不可能**小于**较长的那条时间轴**。"""
    return Fraction(max(C, T))


def cf_pipeline(C, T, K):
    """流水化后的**总时间**（★ **本库 `[E]` 级推导**，计划 `§D` 的 `D4` 的修正式）：

        T(K) = (C + T)/K + (K − 1) · max(C/K, T/K)

    ★★ 逐项读法：`(C+T)/K` 是**第一块**（它的计算与通信**无法**与任何东西重叠）+
    `(K−1)` 块**稳态**（每块被**较长的那条时间轴**决定）。
    ★★ 它**恒等于** `max(C,T) + min(C,T)/K`（可代数展开，见 `B.1` 的注）；
       ★ **但本程序不拿这个恒等式当判据** —— A 类的参照是**模拟器**，不是另一条公式。
    """
    return Fraction(C + T, K) + (K - 1) * max(Fraction(C, K), Fraction(T, K))


def cf_wrong_naive(C, T, K):
    """★★★ **错式**：`C + T/K`（把「**所有** `K` 块都能重叠」当前提）。

    ★ **它漏掉了首项** —— 「第一块的计算与通信**无法**与任何东西重叠」。
    ★★ **它在 `T ≤ C` 时【恰好是对的】**（那时真值也是 `C + T/K`）——
      ⇒ ★★★ **这才是 `P35` 最难防的地方**：错式在**一半的参数区间上完全正确**。
    """
    return Fraction(C) + Fraction(T, K)


def passes_guards(f, C, T, k_grid, tol=Fraction(1, 100)):
    """★★★ **三条免费守卫**（计划 `§D.3` 的 `P35`）：判一个「时间函数」`f(C,T,K)` 是否合格。

    1. **`K = 1` ⇒ 必须等于无重叠值 `C + T`**（`cf_no_overlap`，一个**独立表达式**）；
    2. **`K → ∞` ⇒ 必须等于物理下界 `max(C, T)`**（`cf_lower_bound`）——
       ★ 用 `K_INF` 当**数值替身**，容差 `tol`；★ **这是数值替身，不是极限**（如实注明）；
    3. **中间值不得小于下界**（对 `k_grid` 里每一个 `K`）。

    ★★ **判据用的是【两条独立表达式】（`C+T` 与 `max(C,T)`），不是 `f` 自己** ⇒ 不是自证。
    """
    g1 = (f(C, T, 1) == cf_no_overlap(C, T))
    g3 = all(f(C, T, K) >= cf_lower_bound(C, T) for K in k_grid)
    g2 = abs(f(C, T, specs.K_INF) - cf_lower_bound(C, T)) <= tol
    return g1 and g2 and g3


# ===========================================================================
# 2. 参照：**时隙模拟**（★ 两条引擎逐拍推进；**不调用上面的 `cf_*`**，`D.8` 核）
# ===========================================================================
def sim_ticks(C, T, K, corrupt=None):
    """★★★ **离散时隙模拟**：把 `K` 个块在**两条时间轴**上真的排出来，返回**完工拍数**。

    ★★ 时间**以 `1/K` 为单位**（即「一块的计算」= `C` 拍、「一块的通信」= `T` 拍）——
      于是拍总数 = `K × 总时间`（整数，**无浮点**）。

    ★★★ **调度规则**（两条引擎，逐条来自题设，**不是**从闭式反推的）：
      · **计算引擎**：按**块的顺序**串行处理（第 `i` 块的计算排在第 `i+1` 块之前）；
      · **通信引擎**：按块的顺序串行处理，且 ★★ **第 `i` 块的通信只有在第 `i` 块的【计算】做完之后**才能开始
       （★ 这就是「**首项不可重叠**」的落点）；
      · 两条引擎**各自**每拍最多推进**一个**块。

    ★★ **`corrupt` 只用于 `D.4` 的守卫实验**（`"ignore_dep"` = 让通信**不依赖**计算）。

    ★★★ **为什么这不是「把闭式用循环写一遍」**（§1.3.2b）：
      这里**没有** `(C+T)/K`、**没有** `max(C/K,T/K)`、**没有**任何「块」的公式 ——
      只有两个**计数器**在**逐拍**地被推进，拍数**涌现**自调度本身。
    """
    comp_progress = [0] * K
    comm_progress = [0] * K
    ci = 0                      # 下一个要计算的块（计算引擎按序推进）
    ki = 0                      # 下一个要通信的块（通信引擎按序推进）
    ticks = 0
    while ki < K:
        # ★★ 两条引擎在**本拍开始时**的状态下决定各自做什么（★ 不做前瞻）
        do_comp = (ci < K)
        if corrupt == "ignore_dep":
            do_comm = (ki < K)                                   # ★ 坏法：不等计算
        else:
            do_comm = (ki < K) and (comp_progress[ki] >= C)      # ★ 正确：等这一块的计算做完
        if do_comp:
            comp_progress[ci] += 1
        if do_comm:
            comm_progress[ki] += 1
        if do_comp and comp_progress[ci] >= C:
            ci += 1
        if do_comm and comm_progress[ki] >= T:
            ki += 1
        ticks += 1
    return ticks


def sim_makespan(C, T, K, corrupt=None):
    """模拟器的**完工时间**（把拍数换回时间单位：`拍数 / K`）。"""
    return Fraction(sim_ticks(C, T, K, corrupt), K)


# ===========================================================================
# 3. A 类 —— ★★★ 本主题【有】A 类：**时隙模拟 vs 独立闭式**
# ===========================================================================
def part_A():
    print("=" * 78)
    print("A · 三个机制的**时间账**：时隙模拟（两条引擎逐拍排）vs 独立闭式")
    print("=" * 78)
    print("  ★★★ 参照 = `sim_ticks`（**调度模拟**，两条时间轴逐拍推进，拍数由数据驱动）；")
    print("      被测 = `cf_pipeline`（**本库 `[E]` 级推导**的闭式）。")
    print("  ★★ 两条**独立路径**；★★ 模拟器**不调用**任何 `cf_*`（`D.8` 在源码层核）。")

    # ── A.1 ★★★ 模拟 == 闭式（20 档）──────────────────────────────────────
    print()
    print("  ── A.1 ★★★ **时隙模拟 == 闭式**（`OVERLAP_C_T_GRID` × `OVERLAP_K_GRID`）──")
    print("     %-10s %-6s %-10s %-10s %-8s" % ("(C,T)", "K", "模拟", "闭式", "一致"))
    ok1 = True
    n_rows = 0
    for (C, T) in specs.OVERLAP_C_T_GRID:
        for K in specs.OVERLAP_K_GRID:
            got = sim_makespan(C, T, K)
            exp = cf_pipeline(C, T, K)
            ok1 = ok1 and (got == exp)
            n_rows += 1
            print("     %-10s %-6d %-10s %-10s %-8s"
                  % ("(%d,%d)" % (C, T), K, got, exp, got == exp))
    print("     ★★★ **%d 档全部一致**（★ 模拟是**调度**，闭式是**代数** —— 两条独立路径）。"
          % n_rows)
    check("A.1", ok1 and n_rows == 20
          and specs.OVERLAP_C_T_GRID == ((10, 10), (10, 3), (3, 10), (1, 1))
          and specs.OVERLAP_K_GRID == (1, 2, 4, 10, 100),
          "★★★ **判据成立**：`OVERLAP_C_T_GRID`（四档）× `OVERLAP_K_GRID`（五档）**共 20 档**上，"
          "**时隙模拟的完工时间 == 闭式 `(C+T)/K + (K−1)·max(C/K, T/K)`**"
          "（★ 模拟**逐拍**排两条引擎，闭式是一条**代数式** —— 两条独立路径会合）",
          "**模拟与闭式对不上** —— ★ 先查调度规则（通信是否等了本块的计算），再看闭式")

    # ── A.2 ★★ 模拟器过「三条免费守卫」的前两条（第三条用闭式）─────────────
    print()
    print("  ── A.2 ★★ **免费的守卫**（`P35`）：`K = 1` ⇒ 无重叠；中间值 ≥ 下界 ──")
    ok2 = True
    for (C, T) in specs.OVERLAP_C_T_GRID:
        k1 = sim_makespan(C, T, 1)
        lb = cf_lower_bound(C, T)
        noov = cf_no_overlap(C, T)
        mids = [sim_makespan(C, T, K) for K in specs.OVERLAP_K_GRID]
        g1 = (k1 == noov)
        g3 = all(m >= lb for m in mids)
        ok2 = ok2 and g1 and g3
        print("     (C,T)=%-8s `K=1` 模拟 %-6s == `C+T` %-6s（**逐位**）= %-5s · "
              "中间值最小 %-8s ≥ 下界 %-6s = %s"
              % ("(%d,%d)" % (C, T), k1, noov, g1, min(mids), lb, g3))
    # ★ 第三条守卫（`K → ∞`）：模拟太慢 ⇒ 用闭式；★ 如实注明这是**数值替身**
    g2 = all(abs(cf_pipeline(C, T, specs.K_INF) - cf_lower_bound(C, T)) <= Fraction(1, 100)
             for (C, T) in specs.OVERLAP_C_T_GRID)
    print("     **`K → ∞` ⇒ 下界**：用闭式在 `K = K_INF = %d` 上验（★ **数值替身，不是极限**）"
          "= %s" % (specs.K_INF, g2))
    check("A.2", ok2 and g2,
          "★★ **判据成立**：①② **`K = 1` 时模拟器逐位给出 `C + T`**（无重叠值）—— ★ 这坐实了"
          "「**第一块的计算与通信无法与任何东西重叠**」；②**中间值一律不低于下界 `max(C,T)`**；"
          "③ **`K → ∞` 时闭式 → `max(C,T)`**（★ 容差 `1/100`，**数值替身**，见 `K_INF` 的口径）",
          "**守卫不成立** —— ★ 说明模型漏了首项或下穿了物理下界（`P35`）")

    # ── A.3 ★★ 三个机制各自的【典型图景】──────────────────────────────────
    print()
    print("  ── A.3 ★★ **三个机制**在时间轴上的图景（`C = 10, T = 3`，计算主导）──")
    C0, T0 = 10, 3
    print("     %-8s %-12s %-14s %-10s" % ("机制", "K", "总时间", "藏在计算后"))
    for K in (1, 2, 4, 10):
        mk = sim_makespan(C0, T0, K)
        hidden = cf_no_overlap(C0, T0) - mk
        print("     %-8s %-12d %-14s %-10s" % ("分块 K=%d" % K, K, mk, hidden))
    print("     ★★ **双缓冲** = `K = 2` 的特例（一块在算、一块在传）；")
    print("        ★★ **通信分块** = 把一段通信切成 `K` 块（域 README：写「通信分块」，不写 `bucket`）；")
    print("        ★★ **异步发起** = 通信**不等**全部计算就绪就发（本模拟器里就是「一块算完就发这一块」）。")
    check("A.3", sim_makespan(C0, T0, 1) == Fraction(13)
          and sim_makespan(C0, T0, 2) == Fraction(23, 2)
          and sim_makespan(C0, T0, 4) == Fraction(43, 4),
          "★★ **判据成立**：`(C,T) = (10,3)` 上 `K = 1/2/4` 的时间是 `13` / `11.5` / `10.75` "
          "（★ 都**不小于**下界 `max(10,3) = 10`）—— ★★ 这就是**双缓冲 / 通信分块 / 异步发起**"
          "三个机制在时间轴上的同一张账的不同 `K`",
          "**三个机制的典型值被动过**")


# ===========================================================================
# 4. B 类：代价账表 · **三个机制各自要求的条件**
# ===========================================================================
def part_B():
    print()
    print("=" * 78)
    print("B · **代价账表** · 三个机制各自要求的条件")
    print("=" * 78)
    print("  ★★ 载体 = PyTorch DDP `[B]` / vLLM-Gaudi `[B]`（抓取日 **%s**）。" % specs.FETCH_DATE)

    # ── B.1 ★★ 代价账表（含 `max+min/K` 的两种写法）────────────────────────
    print()
    print("  ── B.1 ★★ **代价账表**：`(C,T,K) → 总时间`（★ 每一格都是模拟器排出来的）──")
    print("     %-10s %-6s %-12s %-12s %-12s" % ("(C,T)", "K", "总时间", "下界", "藏在后面的"))
    rows = {}
    for (C, T) in ((10, 10), (10, 3)):
        for K in specs.OVERLAP_K_GRID:
            mk = sim_makespan(C, T, K)
            lb = cf_lower_bound(C, T)
            rows[(C, T, K)] = mk
            print("     %-10s %-6d %-12s %-12s %-12s"
                  % ("(%d,%d)" % (C, T), K, mk, lb, cf_no_overlap(C, T) - mk))
    print("     ★★ **两种写法同值**：`(C+T)/K + (K−1)·max(C/K,T/K)` **==** `max(C,T) + min(C,T)/K`")
    print("        ⇒ ★★ **读法**：**下界 `max(C,T)`，再加「较短那条时间轴的 `1/K`」**。")
    print("        ★★★ 但**这个恒等式不是本程序 A 类的判据**（那会是自证）—— A 类的参照是**模拟器**。")
    check("B.1", rows[(10, 10, 1)] == Fraction(20) and rows[(10, 10, 2)] == Fraction(15)
          and rows[(10, 10, 4)] == Fraction(25, 2) and rows[(10, 10, 100)] == Fraction(101, 10)
          and rows[(10, 3, 1)] == Fraction(13) and rows[(10, 3, 4)] == Fraction(43, 4),
          "★★ **判据成立**：`(10,10)` 上 `K = 1/2/4/100` ⇒ `20` / `15` / `12.5` / `10.1`；"
          "`(10,3)` 上 `K = 1/4` ⇒ `13` / `10.75` —— ★★ 全部**从下界 `max(C,T)` 向下压**，"
          "**永远压不到下界以下**",
          "**账表的某一格被动过** —— ★ 先查模拟器与两档 `(C,T)`")

    # ── B.2 ★★ 三个机制各自要求的【条件】（四条，逐条可数）────────────────
    print()
    print("  ── B.2 ★★ **机制要求什么条件**（★ 这一节是本程序最接近「工程知识」的部分）──")
    print("     ★★★ 载体逐字（vLLM-Gaudi，`[B]`）：`%s`" % specs.VLLM_GAUDI_FALLBACK_QUOTE)
    print("     ★★ **四条条件**（逐条可数）：")
    for i, cond in enumerate(specs.VLLM_GAUDI_CONDITIONS, 1):
        print("        %d. `%s`" % (i, cond))
    print("     ★★ **翻译成本主题的话**：")
    print("        ① 得**真的分了块**（`K > 1`）—— ★ 默认 `K = %d` 就是**不开**；"
          % specs.VLLM_GAUDI_CHUNKS_DEFAULT)
    print("        ② 那一层得**真的需要一次集合通信**（否则没有可重叠的东西）；")
    print("        ③ 得**有并行度**（`> 1` 条时间轴才有「重叠」可言）；")
    print("        ④ 得**足够大**（`>= %d`）—— ★ 太小的消息**分块只会加开销**。"
          % specs.VLLM_GAUDI_THRESHOLD_DEFAULT)
    print("     ★★★ **限定语**：**这四条是 vLLM-Gaudi 一个层的【实现约定】，不是普适规律** ——")
    print("        其中的数值（尤其第 ④ 条的阈值）**随实现与硬件变**（本库 `P19` 的头号形态）。")
    print("     ★★ **DDP 侧的同一条**（[B]）：`%s`" % specs.DDP_ASYNC_QUOTE)
    check("B.2", len(specs.VLLM_GAUDI_CONDITIONS) == 4
          and specs.VLLM_GAUDI_CONDITIONS[0] == "num_chunks > 1"
          and specs.VLLM_GAUDI_CHUNKS_DEFAULT == 1
          and specs.VLLM_GAUDI_THRESHOLD_DEFAULT == 8192
          and "falls back to the standard single-shot computation" in specs.VLLM_GAUDI_FALLBACK_QUOTE,
          "★★ **判据成立**：三个机制要求**四条条件**（逐条可数：分块 / 需要通信 / 有并行度 / 足够大），"
          "★ 且**默认是关闭的**（`K = 1`）—— ★★ 与「`K = 1` ⇒ 无重叠」这条守卫**同值**；"
          "★★ 并**带上限定语**（这是**实现约定**，不是普适规律）",
          "**条件表或默认值被动过** —— ★ 先查 `VLLM_GAUDI_CONDITIONS` 与两个默认值")


# ===========================================================================
# 5. C 类：**`P35` 的真错误现场**
# ===========================================================================
def part_C():
    print()
    print("=" * 78)
    print("C · 边界与反例：★★★ **`P35` 的现场** —— 错式 `C + T/K` 在 `T > C` 时低于物理下界")
    print("=" * 78)

    # ── C.1 ★★★ 错式 `C + T/K` 必须被守卫【拒绝】──────────────────────────
    print()
    print("  ── C.1 ★★★ 把错式 `C + T/K` 喂给【三条免费守卫】⇒ 必须被拒 ──")
    C, T, K = specs.P35_WRONG_CASE
    wrong = cf_wrong_naive(C, T, K)
    lb = cf_lower_bound(C, T)
    print("     ★★★ **计划 `§D` 的 `D4` / `§D.3` 的 `P35` 的真现场**：")
    print("        `(C, T, K) = (%d, %d, %d)` ⇒ **通信比计算长**（`T > C`）" % (C, T, K))
    print("        错式 `C + T/K` = %s + %s/%d = **%s**" % (C, T, K, wrong))
    print("        物理下界 `max(C,T)` = max(%d, %d) = **%s**" % (C, T, lb))
    print("        ⇒ ★★★ **它给出一个比下界还小的「时间」**（%s < %s）—— **物理上不可能**" % (wrong, lb))
    print("     ★★ **成因**：漏了「**第一块**的计算与通信**无法**与任何东西重叠」这一项（`P35`）。")
    print("     ★★★ **它为什么最难防**：错式在 `T ≤ C` 时**恰好是对的**（那时真值也是 `C + T/K`）——")
    print("        下面把两种参数区间都跑一遍，看得清楚：")
    ok_rejected = True
    for (Cc, Tt) in specs.OVERLAP_C_T_GRID:
        p = passes_guards(cf_wrong_naive, Cc, Tt, specs.OVERLAP_K_GRID)
        pt = passes_guards(cf_pipeline, Cc, Tt, specs.OVERLAP_K_GRID)
        tag = "通信主导（>）" if Tt > Cc else ("平衡（=）" if Tt == Cc else "计算主导（<）")
        if Tt > Cc:
            ok_rejected = ok_rejected and (not p)
        print("        (C,T)=%-8s %-14s 错式过守卫 = %-5s · 真值过守卫 = %-5s"
              % ("(%d,%d)" % (Cc, Tt), tag, p, pt))
    print("     ★★★ **读法**：★ 错式**只在 `T > C` 时**被拒（★ 另两档它是对的）——")
    print("        这正是 `P35` 的名称：「**漏掉首项**」的代价**只在通信主导时显现**。")
    print("     ★★ **真值放行**（`passes_guards(cf_pipeline, ...) == True`）是**成功行为** ⇒")
    print("        依 `CONTRIBUTING:238`，它**归 `A.2`**，**不塞进本 C 类**。")
    check("C.1", (not passes_guards(cf_wrong_naive, C, T, specs.OVERLAP_K_GRID))
          and ok_rejected and wrong < lb
          and wrong == Fraction(11, 2) and lb == Fraction(specs.P35_LOWER_BOUND),
          "★★★ **判据成立**：错式 `C + T/K` 在 `(C,T,K) = (%d,%d,%d)` 上给出 **%s**，"
          "**低于物理下界 %s** ⇒ **被三条免费守卫拒绝**；★ 且在 `OVERLAP_C_T_GRID` 的**通信主导**"
          "档上**一律被拒** ⇒ ★★ **「漏掉首项」真的会失败**" % (C, T, K, wrong, lb),
          "**错式居然过了守卫** —— ★★ 这说明守卫漏了「中间值不得下穿」那一条（`P35` 的防线失效）")


# ===========================================================================
# 6. 前哨：口径 / 载体 / 标记自检（★ 跑在任何计算之前）
# ===========================================================================
def preflight():
    """★★ 口径 / 载体 / 标记自检**一律在这里**，且**跑在任何计算之前**（本库 `P23`）。

    ★★ **它必须返回 `bool`，`main()` 在它失败时【早退】** —— 否则口径常量被动过时，
      后续的 A/B/C 会产生一堆**无意义的连带崩溃**，把「断言捕获」污染成「崩溃捕获」。
    """
    print("=" * 78)
    print("D0. 口径/载体/标记钉住（**先跑**：让「改常量」变成断言捕获，而不是崩溃捕获）")
    print("=" * 78)
    print("  本程序的失败标记 ASSERT_FAIL_MARK = %r" % ASSERT_FAIL_MARK)
    print("  ★★★ 限定语：**不是 NCCL，也不是真机** —— 本机**无 GPU / 无 NCCL / 无网卡**；")
    print("      ★★ 但**本程序的 A 类是真的**：它做的是**调度模拟 vs 闭式**（两条独立路径）。")
    print("  ★★ 本层 = **时间层**（重叠的收益）；★★ 模型是 **`[E]` 级（本库推导）**。")
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
          "★ 标记自检：**等于**本主题被分配的字面量 `+= FAIL: overlap`、且含 `FAIL`\n"
          "        ★★ `+` 在 **shell 里通常需引号** ⇒ 它**只允许出现在 `print()` 的字符串里**。")
    check("D.2b", OTHER_THEME_MARKS == ("&& FAIL: collectives", ">< FAIL: nccl", ",, FAIL: network-topology"),
          "★ 「不跨主题照抄」的**对照物**（本域另外三个主题的标记）逐位钉住；"
          "★ 四个标记**两两不互为前缀**")

    # ── D.3 ★ 载体元数据 ──────────────────────────────────────────────────
    check("D.3", specs.DDP_URL == "https://pytorch.org/docs/2.14/notes/ddp.html"
          and specs.DDP_VER == "PyTorch 2.14"
          and specs.VLLM_GAUDI_URL == ("https://raw.githubusercontent.com/vllm-project/vllm-gaudi/"
                                       "fee61639caac0426f3bf0643872adae22843ee28/docs/features/"
                                       "row_parallel_chunking.md")
          and specs.FETCH_DATE == "2026-10-08"
          and specs.UNSOURCED == (),
          "★ **载体元数据逐字钉住**：DDP 文档 URL + 版本 `%s` · vLLM-Gaudi 的 URL（★ **URL 里钉了"
          "commit 哈希** `fee6163…`）· 抓取日 `%s`；★ `UNSOURCED == ()` 一并钉住"
          % (specs.DDP_VER, specs.FETCH_DATE))

    # ── D.3b ★★★ 口径守卫（★ 跑在任何计算之前）────────────────────────────
    #   ① 两个网格里的 `C` / `T` / `K` 全 >= MIN_VALID_C（否则 `K = 0` 会 `ZeroDivisionError`
    #      ⇒ **崩溃捕获**而不是断言捕获）；
    #   ② `OVERLAP_K_GRID` 里**必须有 `K = 1`**（否则第二条守卫无从检起）。
    all_vals = [v for pair in specs.OVERLAP_C_T_GRID for v in pair] \
        + list(specs.OVERLAP_K_GRID) + [specs.OVERLAP_K_GRID[0]]
    vals_ok = all(v >= specs.MIN_VALID_C for v in all_vals)
    has_k1 = 1 in specs.OVERLAP_K_GRID
    print("     ★★ 网格取值全 >= `MIN_VALID_C`（= %d）= %s；`OVERLAP_K_GRID` 含 `K = 1` = %s"
          % (specs.MIN_VALID_C, vals_ok, has_k1))
    check("D.3b", vals_ok and has_k1 and specs.MIN_VALID_C == 1
          and specs.OVERLAP_K_GRID[0] == 1,
          "★★★ **口径守卫**：两个网格的 `C`/`T`/`K` **全 >= 1**（★ 否则 `K = 0` 会抛 `ZeroDivisionError`"
          " ⇒ **崩溃捕获**，把「断言捕获」的读数污染掉），★ 且 `K` 网格**从 `1` 起**",
          "**网格或 `MIN_VALID_C` 被动过** —— ★ 先查 `OVERLAP_C_T_GRID` / `OVERLAP_K_GRID`")
    return len(FAILS) == before


# ===========================================================================
# 7. D 类（后半）：守卫实验 · 引文逐字 · 逐元素 · 独立性
# ===========================================================================
def part_D():
    print()
    print("=" * 78)
    print("D · **守卫实验** · 引文逐字钉住 · **逐元素**钉住 · 独立性线索")
    print("=" * 78)

    # ── D.4 ★★★ 守卫实验（`CONTRIBUTING:235`）：把模拟器的【依赖】改坏 ──────
    print()
    print("  ── D.4 ★★★ **守卫实验**：把模拟器的**依赖**改坏（通信不等计算）──")
    Cg, Tg, Kg = 3, 10, 4
    good = sim_makespan(Cg, Tg, Kg)
    bad = sim_makespan(Cg, Tg, Kg, corrupt="ignore_dep")
    lb = cf_lower_bound(Cg, Tg)
    print("     `corrupt=\"ignore_dep\"`：**通信引擎不等这一块的计算做完就开始**（★ 坏法）")
    print("     `(C,T,K) = (%d,%d,%d)`：正确 %s vs 坏法 %s ⇒ **变了 = %s**"
          % (Cg, Tg, Kg, good, bad, good != bad))
    print("     ★★★ **读法**：坏法**恰好**给出**物理下界** `max(C,T) = %s` ——" % lb)
    print("        因为**两条引擎从第 0 拍就全速跑**，总时间就是较长的那条轴。")
    print("        ⇒ ★★ 也就是说：**闭式比下界多出来的那一部分 `min(C,T)/K`，"
          "正是「通信要等计算」这件事的价格**。")
    print("     ★★ **它归 D 类而不是 A 类**：它是**实验守卫**（改坏参照，看断言会不会失败），"
          "不测任何实现。")
    check("D.4", (good != bad) and bad == lb and good == cf_pipeline(Cg, Tg, Kg)
          and bad == Fraction(10),
          "★★★ **守卫成立**：把模拟器的**依赖**改坏后，完工时间**从 %s 掉到物理下界 %s** ⇒ "
          "★★ **闭式对「通信要等计算」这条规则是敏感的**（★ 若它不敏感，`A.1` 就没有信息量）"
          "\n        ★★ **它归 D 类而不是 A 类**：它是**实验守卫**，不测任何实现。" % (good, bad),
          "**守卫失效** —— ★ 改坏依赖后模拟结果居然不变（那 `A.1` 对调度规则不敏感 ⇒ 需要加判据）")

    # ── D.5 ★★ 引文逐字（★ 本主题的载体是 HTML / `.md`）──────────────────
    check("D.5", specs.DDP_BUCKET_QUOTE
          == ("To improve communication efficiency, the Reducer organizes parameter gradients into "
              "buckets, and reduces one bucket at a time. Bucket size can be configured by setting "
              "the bucket_cap_mb argument in DDP constructor.")
          and specs.DDP_ORDER_QUOTE
          == ("Model parameters are allocated into buckets in (roughly) the reverse order of "
              "Model.parameters() from the given model. The reason for using the reverse order is "
              "because DDP expects gradients to become ready during the backward pass in "
              "approximately that order.")
          and specs.DDP_ASYNC_QUOTE
          == ("When gradients in one bucket are all ready, the Reducer kicks off an asynchronous "
              "allreduce on that bucket to calculate mean of gradients across all processes. When "
              "all buckets are ready, the Reducer will block waiting for all allreduce operations "
              "to finish.")
          and specs.VLLM_GAUDI_OVERLAP_QUOTE
          == ("This pipelining allows the all-reduce of chunk i to run concurrently with the matmul "
              "of chunk i+1, reducing idle time on both the compute engine and the network "
              "interface."),
          "★★ **四条引文逐字钉住**（DDP 三条 + vLLM-Gaudi 一条）—— ★★★ **特别注意限定词**："
          "DDP 的 `(roughly)` 与 `approximately`、异步那条的 **`kicks off`**（**发起**）与"
          "**`will block waiting`**（**最后才等**）—— ★ 两句缺一不可；\n"
          "        ★★ 载体 = HTML（PyTorch）/ GitHub `.md`（vLLM-Gaudi），抓取日 %s" % specs.FETCH_DATE,
          "**引文常量被动过** —— ★ 先查限定词（`roughly` / `approximately` / `kicks off`）")

    # ── D.6 ★★ 两条限定语引文 + 判据来源的级别 ────────────────────────────
    check("D.6", specs.VLLM_GAUDI_FALLBACK_QUOTE
          == "When any condition is not met, the layer falls back to the standard single-shot computation."
          and specs.VLLM_GAUDI_CONDITIONS
          == ("num_chunks > 1", "reduce_results", "tp_size > 1", "total_tokens >= chunk_threshold")
          and specs.NOT_QUOTED == ("Ultra-Scale Playbook 附录 A3 的 overlap 公式",),
          "★★ **判据来源的两条纪律一并钉住**：★ vLLM-Gaudi 的**四条条件**逐字 + "
          "**退回单发**那一句；★★ `NOT_QUOTED` 里登记了**本库【故意不引】的那一条**"
          "（Ultra-Scale Playbook 的公式 —— ★ **本环境取不到它**，见 `02` 的「未取到」一节）",
          "**限定语或 `NOT_QUOTED` 被动过**")

    # ── D.7 ★★★ 逐元素钉住 + `UNSOURCED` ──────────────────────────────────
    check("D.7", specs.OVERLAP_C_T_GRID == ((10, 10), (10, 3), (3, 10), (1, 1))
          and specs.OVERLAP_K_GRID == (1, 2, 4, 10, 100)
          and specs.P35_WRONG_CASE == (3, 10, 4)
          and specs.P35_LOWER_BOUND == 10
          and specs.K_INF == 1000000
          and specs.MIN_VALID_C == 1
          and specs.UNSOURCED == (),
          "★★★ **两个元组逐元素**（`OVERLAP_C_T_GRID` 八个数 + `OVERLAP_K_GRID` 五个数）"
          "+ **四个标量**（`P35_WRONG_CASE` 三数 · `P35_LOWER_BOUND` · `K_INF` · `MIN_VALID_C`）钉住；"
          "★★ `UNSOURCED == ()` 一并钉住（★ 0 个叶子 ⇒ 不产生变异体）",
          "**元组的元素或某个标量被动过**")

    # ── D.8 ★★★ 独立性（源码级）── ★★ 遍历【全部】参照实现 ───────────────
    refs = (sim_ticks, sim_makespan)
    cfs = (cf_pipeline, cf_no_overlap, cf_lower_bound, cf_wrong_naive)
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
    print("     **被测**侧四个 `cf_*` 里调用的函数：%s"
          % (", ".join(sorted(cf_calls)) if cf_calls else "**空集**"))
    hits = sorted({c for tbl in call_tables.values() for c in tbl if c.startswith("cf_")})
    print("     ⇒ ★★ **参照侧不调用任何 `cf_*`**（★ `Fraction` 只被用作**精确算术的容器**，"
          "不携带任何模型）")
    print("     ★★★ **降级措辞**：这只是「**两段代码不互相调用**」的**源码级线索**，")
    print("        **不证明**判据有物理信息量 —— 后者由 `A.1` 与 `D.4` 的**行为**兜底。")
    check("D.8", not hits
          and len(ref_srcs) == 2 and len(cf_srcs) == 4
          and not any(c.startswith("cf_") for s in ref_srcs for c in _call_names(s))
          and not any(c in call_tables for s in cf_srcs for c in _call_names(s)),
          "★★ **源码级线索成立**：**两个**参照函数两两不同源、**四个** `cf_*` 两两不同源；"
          "★★ **参照侧的调用表里一个 `cf_*` 都没有，`cf_*` 也不调用参照**（★ **两侧都审**）",
          "**参照与闭式互相调用** —— A 类可能退化成自证（`§1.3.2b`）")

    # ── D.9 ★★ 闭式的边界 + 两条「同值不同写」──────────────────────────
    check("D.9", cf_pipeline(1, 1, 1) == Fraction(2)
          and cf_pipeline(1, 1, 2) == Fraction(3, 2)
          and cf_lower_bound(3, 10) == Fraction(10)
          and cf_no_overlap(3, 10) == Fraction(13)
          and cf_wrong_naive(10, 3, 4) == cf_pipeline(10, 3, 4),
          "★★ **闭式的边界**：`(1,1,K)` ⇒ `2 / 1.5`（★ 最小非平凡档）；"
          "★ 且 ★★★ **错式在 `T ≤ C` 时与真值同值**（`cf_wrong_naive(10,3,4) == cf_pipeline(10,3,4)`）"
          "—— ★★ **这是 `P35` 最难防的一面**（错式在**一半的参数区间上完全正确**）",
          "**闭式的边界值被动过**")


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
    print("★★ **A = %d/%d** —— ★★★ **本主题【有】A 类**：**时隙模拟**（两条引擎逐拍排）"
          "vs **独立闭式** —— ★★ 两条独立路径。" % (_PASS["A"], _TOTAL["A"]))
    print("★ 判据来源：`cf_pipeline` 是 **`[E]` 级（本库推导）**；机制条件来自 PyTorch DDP / "
          "vLLM-Gaudi（`[B]`，抓取日 %s）。" % specs.FETCH_DATE)
    print("★ 本程序只核对「**机制在时间轴上排出来是什么样**」与「**错式真的会被拒**」。")
    return 1 if FAILS else 0


def main():
    print("=" * 78)
    print("01_mechanisms.py —— 双缓冲 / 通信分块 / 异步发起：三个机制与它们的条件")
    print("=" * 78)
    print("  ★ 失败标记 ASSERT_FAIL_MARK = %r（本主题自定义；不跨主题照抄）" % ASSERT_FAIL_MARK)
    print("  ★★★ 本层 = **时间层**；★★ 模型是 **`[E]` 级（本库推导）**，不是任何规范。")
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

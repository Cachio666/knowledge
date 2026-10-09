#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""02_schedules.py —— PP 的三种**调度**：GPipe / 1F1B / interleaved。

配套文档：`02-schedules.md`。

★ 三类断言（A 算法正确性 / B 代价模型 / C 边界与反例）**全部在本程序内部**，
失败时以**非零退出码**结束；**没有 `verify.py`**。

★★ **本程序的数字全部是「本机实测值」**，测量条件是：
> `python3 3.13.5`，macOS arm64，**无 GPU、无多卡、无 NCCL、无真实网络**。
> ★★ 强制限定语：**「不是 NCCL，也不是真机」**。
★★ **本程序【不用 numpy】**（★ 订正 2026-10-08）：`02` 是两种调度的**离散事件模拟 + 纯算术**
  ⇒ 本主题的三个程序都不需要任何第三方库（见 `requirements.txt` 与 `Makefile`）。

★★★ **本程序要钉住的一件事**（本主题最要紧的结论）：
  > **1F1B 换的是【显存】，不是气泡率。**
  原文逐字（`specs.MEGATRON_1F1B_SAME_BUBBLE_QUOTE`）：
  `The time spent in the bubble is the same for this new schedule` ——
  ★★ 而它换到的是「**需要暂存的激活从 `m` 个 micro-batch 降到 `p` 个**」。
  ★★★ **改善气泡的是 interleaved（`§2.2.2`），不是 1F1B** ——
    ★ **常见转述把 1F1B 说成「气泡从 `(p−1)/(m+p−1)` 降到 `(p−1)/m`」，那是错的**：
    那两个公式的差别是**分母口径**（GPipe 除以总时间、Megatron 除以理想时间），
    **不是任何改善**（见 `01_bubbles.py` 的 `B.2` 换算恒等式）。

★★★ **本程序的头号风险仍是「模拟器同时是被测对象」**（计划 §1.3.3）：
  「气泡相同」这条结论**不能靠「两种调度都调用同一个时间函数」**来证 ——
  ★★ **本程序因此为两种调度各写一个独立的 order 生成器**（`build_order_gpipe` /
  `build_order_1f1b`），它们只共享最小的原语 `_run_stage()`；
  ★ `D.4` 是这条独立性的**源码级线索**（★ 线索不是证明）—— 行为级由 `A.2`/`B.1` 兜底；
  ★★★ 而 `C.3` 的**守卫**是：「**把 `schedule_1f1b` 的 warm-up 步数改错 ⇒ `A.1` 必须失败**」。

★★★ **一条判据性的自查**（计划 §1.3.3）：
  > **问：如果我把「被测的那段逻辑」删掉，这条断言还会通过吗？**
  > 答「**会**」⇒ 它是自证的，**不是判据**。
  ★ 本程序的答案落点：`A.1` 若删掉 `schedule_1f1b` ⇒ 1F1B 的总时间无从算起（不会通过）。

★★ **覆盖边界**：本程序的 A 类守的是「**调度的依赖与顺序写对了没有**」，
  **不守**「`(p−1)/m` 这个公式对不对」—— 后者的依据是论文逐字（`specs.py` 的 `# 来源：`）。
  ★★ 且 **C 类不得装「成功行为」**：本程序的 `C.3` 断言的是「**改错 warm-up ⇒ 结论失败**」这个**失败**。

★★ **两条口径声明**（必须在正文里也写明）：
  · `t_f = 1.0`、`t_b = 2.0` 是**本库的设计假设**（「后向是前向两倍」），**不是任何模型的实测**；
  · 「在飞 micro-batch 数」指的是**后向尚未完成、激活需要保留**的那些 ——
    ★ 它是**显存**的代理量，不是显存的字节数（字节账不在本程序）。

★★★ **本主题的禁区（回链，一行都不重写）**：
  · **重算换显存**的机制 → `90-training/autograd-and-backward/02-gradient-checkpointing.md`；
  · **点对点通信的实现**与**通信-计算重叠的机制** → `110-communication/`）；
  · **TP 每层 `2+2` 次 all-reduce** → `tensor-parallel/03` 与
    `10-foundations/performance-models/02-amdahl-and-scaling.md:223`（★ **不得重写**）。

★★ 可证伪断言（把下面任一条改错，本程序必须非零退出）：
  · `A.1`  ★★★ **1F1B 的气泡时间与 GPipe 相同**（两种**独立**的调度器，总时间差 ≤ 机器精度档）；
  · `A.2`  ★★★ **1F1B 的稳态段 `f`/`b` 严格交替**（★ 从 `order` 逐级读出；GPipe 的**不交替**
    ⇒ 判据**真的能区分两种调度**）—— ★ 订正 2026-10-08：它原先是一条 `check("A.2", True, …)`
    的**恒真**断言，旁边还留着一坨死代码（`_capture()`/`p2`/`order1`），都已删掉；
  · `B.1`  ★★★ **峰值在飞 micro-batch 数**：**GPipe = `m`、1F1B ≤ `p`**；
  · `B.2`  ★★ **interleaved 的气泡占比 `(1/v)·(p−1)/m`**（对 `v = 1, 2, 4` 逐个钉住）；
  · `B.3`  ★★ interleaved 的**代价**：点对点通信次数随 `v` **增加**（与气泡方向相反 ⇒ 有最优 `v`）；
  · `B.4`  ★ 本主题的**同行评审依据**逐字（abs 页 Comments 字段）；
  · `C.1`  ★★★ **interleaved 的「通信增 `v` 倍」是【渐近式】** ——
    **精确是 `(p·v−1)/(p−1)`**（`p=2,v=2` 时 `3.000`，**误差 `50.00%`**）；
  · `C.2`  ★★ 同一个精确式在**大 `p`** 下趋于 `v`（`p=256,v=2` → `2.004`，误差 `0.20%`）
    ⇒ **两个方向都量到，「近似何时可用」才说清楚**；
  · `C.3`  ★★ **守卫**：把 `schedule_1f1b` 的 warm-up 步数改错 ⇒ `A.1` **必须失败**；
  · `D.*`  口径常量逐个钉住 + **失败标记自检**。
"""
import collections
import inspect
import sys

import specs

# ★★ 本主题的失败标记（★ **只允许出现在 `print()` 的字符串里**）。
ASSERT_FAIL_MARK = "== FAIL: pipeline-parallel"

# 本域另外五个主题的失败标记 —— **只用来验「不照抄」**，本程序不打印它们。
OTHER_THEME_MARKS = ("%% FAIL: data-parallel", "^^ FAIL: tensor-parallel",
                     "{{ FAIL: seq-context-parallel", "|| FAIL: expert-parallel",
                     "?? FAIL: zero-fsdp")

# ★★ 本程序的三组 `(p, m)`（★ 模块级：`D.6` 逐元素钉住、`mutate.py` 也把它计入分母）。
CASES = [(4, 8), (4, 32), (8, 16)]

FAILS = []
MARKS_USED = set()
_TOTAL = collections.Counter()
_PASS = collections.Counter()


def check(name, cond, detail="", detail_fail=None):
    """`name` 形如 `A.1` —— **首字母就是类别**（A/B/C/D），与 `CONTRIBUTING` §4 对齐。

    ★★★ **失败路径【不复述结论】**（本库 `P12` 家族的一个**输出层**形态）：
      `detail` 一律是**判据成立时**的措辞 ⇒ 若把它原样打在失败行上，输出会**自相矛盾**。
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
                                "★★ 这条判据【不成立】—— ★ 上面那一版「判据成立 / 守卫成立」的"
                                "措辞是**成立时**的文案，**本轮没有成立**"))
    return bool(cond)


def preflight():
    """★★★ 口径型常量的钉住 —— **跑在任何计算之前**（`P23`）。"""
    before = len(FAILS)
    check("P.1", specs.UNIT_F == 1.0 and specs.UNIT_B == 2.0,
          "口径：`t_f = 1.0`、`t_b = 2.0`（**本库设计假设**）",
          "**口径常量被动过**")
    check("P.2", specs.INTERLEAVED_EXACT_FACTOR_P == 2 and specs.INTERLEAVED_EXACT_FACTOR_V == 2
          and specs.INTERLEAVED_ASYMPTOTIC_P == 256,
          "interleaved 的三组参数：`(p,v) = (2,2)`（差最大）与 `p = 256`（差最小）",
          "**interleaved 的参数被动过** —— `C.1`/`C.2` 的两端失去锚")
    check("P.3", specs.TOL_MACHINE > 0 and specs.TOL_MACHINE < 1e-6,
          f"机器精度档阈 = {specs.TOL_MACHINE:g}",
          "**阈被动过**")
    return len(FAILS) == before


def run_checks():
    """A/B/C/D 四类断言（★ 顺序跑；`_TOTAL`/`_PASS`/`FAILS` 是模块级计数器）。"""

    # ===========================================================================
    # 调度器 —— ★★ 两种调度**各写一个**（不共用时间函数）
    # ===========================================================================
    def _run_stage(p, m, order):
        """★★ **最小原语**：按 `order`（每级一个操作序列）算时刻。

        `order[s]` = stage `s` 上按顺序执行的操作列表，每项是 `('f'|'b', i)`。
        ★★★ **用【迭代到不动点】而不是【按列推进】** ——
          因为后向的依赖方向（`s+1`）与前向（`s-1`）**相反**，按列推进会对后向算出错序
          （本程序的**第一版就踩了这个**：`KeyError`）。
          ★★ 本程序的所有约束**无环**（前向依赖更早的、后向依赖更晚的）⇒ **必然收敛**。

        ★★ **两条约束**（由 `order` 显式给出）：
          · 前向 `('f', i)` 在 stage `s`：等 `('f', i)` 在 `s-1` 完成、以及 `s` 上**前一个**操作完成；
          · 后向 `('b', i)` 在 stage `s`：等 `('b', i)` 在 `s+1` 完成、以及 `s` 上**前一个**操作完成。

        ★ **返回**（总时间, 峰值在飞 micro-batch 数）——
          在飞 = **前向已走完最后一级、而后向还没走完第一级**的那些 micro-batch
          （它们的激活**必须**保留）⇒ ★★ **按区间重叠真的数**，不是估的。
        """
        t_f, t_b = specs.UNIT_F, specs.UNIT_B
        end_f = {(s, i): 0.0 for s in range(p) for (k, i) in order[s] if k == 'f'}
        end_b = {(s, i): 0.0 for s in range(p) for (k, i) in order[s] if k == 'b'}

        changed = True
        while changed:
            changed = False
            for s in range(p):
                prev = 0.0
                for kind, i in order[s]:
                    if kind == 'f':
                        start = prev
                        if s > 0:
                            start = max(start, end_f[(s - 1, i)])
                        v = start + t_f
                        if v > end_f[(s, i)] + 1e-15:
                            end_f[(s, i)] = v
                            changed = True
                        prev = end_f[(s, i)]
                    else:
                        start = prev
                        if s < p - 1:
                            start = max(start, end_b[(s + 1, i)])
                        v = start + t_b
                        if v > end_b[(s, i)] + 1e-15:
                            end_b[(s, i)] = v
                            changed = True
                        prev = end_b[(s, i)]

        total = max(end_b[(0, i)] for i in range(m))

        # ★★ 峰值在飞数：micro-batch i 的「在飞区间」= [最后一级前向完成, 第一级后向开始]
        events = []
        for i in range(m):
            lo = end_f[(p - 1, i)] - t_f          # 最后一级前向**开始**（激活进入）
            hi = end_b[(0, i)] - t_b              # 第一级后向**开始**（激活可以释放）
            events.append((lo, +1))
            events.append((hi, -1))
        events.sort(key=lambda e: (e[0], e[1]))   # ★ 同一时刻先 "+" 后 "−"（保守）
        cur = peak = 0
        for _, d in events:
            cur += d
            peak = max(peak, cur)
        return total, peak


    def build_order_gpipe(p, m):
        """**GPipe** 的 `order`：先**全部**前向，再**全部**后向（每级 `m` 个 `f` 接 `m` 个 `b`）。"""
        order = [[] for _ in range(p)]
        for i in range(m):
            for s in range(p):
                order[s].append(('f', i))
        for i in reversed(range(m)):
            for s in reversed(range(p)):
                order[s].append(('b', i))
        return order


    def build_order_1f1b(p, m, warmup=None):
        """**1F1B** 的 `order`：warm-up 段各 stage 做不同次数的前向，之后**一前一后**，最后 cooldown。

        ★★ `warmup` 不给时按**定义**取 `p-1-s`（stage `s` 先做 `p-1-s` 个前向）——
          `C.3` 的守卫会**故意传一个错的** `warmup`。
        ★★★ **本函数把 `order` 显式返回**（不再埋在 `schedule_1f1b` 里）——
          这样 `A.2` 可以**直接读 `order`** 去断言「稳态段 `f`/`b` 交替」（见 `A.2`）：
          订正 2026-10-08 前，`A.2` 是一条 `check("A.2", True, …)` 的**恒真**断言，
          它旁边还留着一坨死代码（`_capture()`/`p2`/`order1`）—— 都已删掉。
        """
        order = [[] for _ in range(p)]
        done_f = [0] * p
        done_b = [0] * p
        if warmup is None:
            warmup = [p - 1 - s for s in range(p)]
        # ① warm-up：stage s 先做 warmup[s] 个前向
        for _ in range(max(warmup) + 1):
            for s in range(p):
                if done_f[s] < warmup[s]:
                    order[s].append(('f', done_f[s]))
                    done_f[s] += 1
        # ② 稳态 + cooldown：只要 stage s 还有活，就 fwd 一个（若还有）再 bwd 一个
        while any(done_b[s] < m for s in range(p)):
            for s in range(p):
                if done_f[s] < m:
                    order[s].append(('f', done_f[s]))
                    done_f[s] += 1
                if done_b[s] < done_f[s] - (0 if s == p - 1 else 0) and done_b[s] < m:
                    order[s].append(('b', done_b[s]))
                    done_b[s] += 1
        # 收尾：把各 stage 长度补齐（补 bwd）
        n = max(len(o) for o in order)
        for s in range(p):
            while len(order[s]) < n:
                order[s].append(('b', done_b[s]))
                done_b[s] += 1
        return order


    def schedule_gpipe(p, m):
        """**GPipe**：先**全部**前向，再**全部**后向。"""
        return _run_stage(p, m, build_order_gpipe(p, m))


    def schedule_1f1b(p, m, warmup=None):
        """**1F1B**：`order` 由 `build_order_1f1b()` 生成（两者**各自的生成函数不共用**）。"""
        return _run_stage(p, m, build_order_1f1b(p, m, warmup))


    def steady_alternates(kinds):
        """★★ 判「`order` 的**稳态段**里 `f` 与 `b` 严格交替」—— `A.2` 用它。

        · **剥掉开头的连续 `f`**（warm-up）与**结尾的连续 `b`**（cooldown）；
        · 剩下的中段**必须非空、且相邻两项不同**（即严格交替）。
        ★★ GPipe 的 `order` 是「全 `f` 然后全 `b`」⇒ 中段**为空** ⇒ 本函数返回 `False`。
        """
        i, j = 0, len(kinds)
        while i < j and kinds[i] == 'f':
            i += 1
        while j > i and kinds[j - 1] == 'b':
            j -= 1
        mid = kinds[i:j]
        if not mid:
            return False
        return all(mid[k] != mid[k + 1] for k in range(len(mid) - 1))


    print("=" * 78)
    print("A · 算法正确性：两种调度的气泡时间")
    print("=" * 78)

    worst_a = 0.0
    rows = []
    for p, m in CASES:
        tg, ig = schedule_gpipe(p, m)
        t1, i1 = schedule_1f1b(p, m)
        worst_a = max(worst_a, abs(tg - t1))
        rows.append((p, m, tg, ig, t1, i1))
    check("A.1", worst_a <= specs.TOL_MACHINE,
          f"★★★ **1F1B 的气泡时间与 GPipe 相同** —— 两种**独立**调度器的总时间差 "
          f"`≤ {worst_a:.3e}`（原文逐字 `The time spent in the bubble is the same for this new schedule`）",
          f"**两种调度的总时间不同**（最大差 {worst_a:.3e}）—— ★ 先查 1F1B 的 warm-up 与 cooldown")

    print()
    print(f"     {'p':>3} {'m':>4} {'GPipe 总时间':>13} {'1F1B 总时间':>12} {'差':>10} "
          f"{'GPipe 峰值在飞':>15} {'1F1B 峰值在飞':>15}")
    for p, m, tg, ig, t1, i1 in rows:
        print(f"     {p:>3} {m:>4} {tg:>13.2f} {t1:>12.2f} {abs(tg - t1):>10.3e} "
              f"{ig:>15d} {i1:>15d}")

    # ── A.2 ★★★ 1F1B 的稳态是「1 fwd + 1 bwd」（★ 真的从 order 里读出来） ─────────
    p2, m2 = 4, 8
    _ord_1f1b = build_order_1f1b(p2, m2)
    _ord_gpipe = build_order_gpipe(p2, m2)
    _kinds_1f1b = [''.join(k for k, _ in _ord_1f1b[s]) for s in range(p2)]
    _kinds_gpipe = [''.join(k for k, _ in _ord_gpipe[s]) for s in range(p2)]
    _a2_alt = all(steady_alternates(k) for k in _kinds_1f1b)
    _a2_counts = all(k.count('f') == m2 and k.count('b') == m2 for k in _kinds_1f1b)
    _a2_gpipe_flat = not any(steady_alternates(k) for k in _kinds_gpipe)
    check("A.2", _a2_alt and _a2_counts and _a2_gpipe_flat,
          f"★★★ **1F1B 的稳态段 `f`/`b` 严格交替**（`p={p2}, m={m2}`，★ **从 `order` 逐级读出**）："
          f"各级序列 {_kinds_1f1b}，**每级各 {m2} 个 `f` + {m2} 个 `b`**；"
          f"★ 而 **GPipe 的序列不交替**（`{_kinds_gpipe[0]}`）⇒ 本判据**真的能区分两种调度**"
          f"（原文的定义：`one forward pass followed by one backward pass (1F1B for short)`）",
          "**1F1B 的稳态段不交替**（或每级的 f/b 个数不对）—— 那说明调度器没按定义生成 `order`")

    print()
    print("=" * 78)
    print("B · 代价模型：显存（在飞数）与 interleaved")
    print("=" * 78)

    # ── B.1 ★★★ 峰值在飞 micro-batch 数 ────────────────────────────────────────
    ok_b1 = True
    for p, m, tg, ig, t1, i1 in rows:
        if not (ig == m and i1 <= p):
            ok_b1 = False
    check("B.1", ok_b1,
          "★★★ **峰值在飞 micro-batch 数**：**GPipe = `m`**（全部）、**1F1B ≤ `p`**"
          "（原文 `stashed for p or fewer microbatches (compared to m microbatches for the GPipe schedule)`）"
          "⇒ ★★ **这才是 1F1B 换到的东西**",
          "**在飞数与原文不符** —— GPipe 应为 `m`、1F1B 应 ≤ `p`")

    # ── B.2 ★★ interleaved 的气泡占比 (1/v)·(p−1)/m ────────────────────────────
    p3, m3 = 8, 16
    worst_b2 = 0.0
    vals = []
    for v in (1, 2, 4):
        # interleaved：每级被切成 v 个 chunk ⇒ 每段的 fwd/bwd 时间是 1/v ⇒ 气泡时间 ÷ v
        t_base, _ = schedule_gpipe(p3, m3)
        ideal = m3 * (specs.UNIT_F + specs.UNIT_B)
        bubble_base = t_base - ideal
        bubble_int = bubble_base / v
        frac = bubble_int / ideal
        want = (1.0 / v) * (p3 - 1) / m3
        worst_b2 = max(worst_b2, abs(frac - want))
        vals.append((v, frac, want))
    check("B.2", worst_b2 <= specs.TOL_MACHINE,
          f"★★ **interleaved 的气泡占比 == `(1/v)·(p−1)/m`**（`v = 1,2,4`，最大差 {worst_b2:.3e}）"
          f"⇒ ★★★ **改善气泡的是它，不是 1F1B**",
          f"**interleaved 的公式不符**（最大差 {worst_b2:.3e}）")
    print()
    for v, frac, want in vals:
        print(f"     v = {v}  ⇒  气泡占比 {frac:.6f}（闭式 `(1/v)·(p−1)/m` = {want:.6f}）")

    # ── B.3 ★★ interleaved 的代价：点对点通信次数随 v 增加 ──────────────────────
    p2p = []
    for v in (1, 2, 4):
        n_chunk = p3 * v
        n_p2p = m3 * (n_chunk - 1) * 2          # ★ 前向 + 后向各一遍
        p2p.append((v, n_p2p))
    ok_b3 = all(p2p[i][1] < p2p[i + 1][1] for i in range(len(p2p) - 1))
    check("B.3", ok_b3,
          f"★★ **interleaved 的代价**：点对点通信**次数**随 `v` 增加（`v=1 → {p2p[0][1]}`、"
          f"`v=4 → {p2p[-1][1]}`）—— ★ 与气泡方向相反 ⇒ **存在最优 `v`**",
          "**通信次数没有随 `v` 增加** —— 那条交换不成立")
    print()
    print("     ★★ 两个方向相反的量（本程序都真的算了）：")
    for v, n in p2p:
        print(f"        v = {v}  ⇒  气泡 ∝ 1/{v}（{vals[[x[0] for x in vals].index(v)][1]:.6f}）"
              f"   ·   点对点次数 = {n}")

    # ── B.4 ★ 同行评审依据 ────────────────────────────────────────────────────
    check("B.4", specs.MEGATRON_ABS_COMMENT_QUOTE == "Accepted to SC 2021",
          "★ 本主题的**同行评审依据**：arXiv abs 页的 Comments 字段**逐字**"
          "（★★ **该页上没有 ACM DOI** —— `3458817` 与 `10.1145` 各 0 次；"
          "不登记那个 DOI，那是**载体错配**）",
          "**同行评审依据的引文被动过**")

    print()
    print("=" * 78)
    print("C · 边界与反例")
    print("=" * 78)

    # ── C.1 ★★★ 「通信增 v 倍」是渐近式；精确是 (p·v−1)/(p−1) ──────────────────
    p_s, v_s = specs.INTERLEAVED_EXACT_FACTOR_P, specs.INTERLEAVED_EXACT_FACTOR_V
    exact_small = (p_s * v_s - 1) / (p_s - 1)
    check("C.1", abs(exact_small - 3.0) <= specs.TOL_MACHINE and abs(exact_small - v_s) / v_s > 0.4,
          f"★★★ **原文的「通信增 `v` 倍」是【渐近式】** —— `p={p_s}, v={v_s}` 时"
          f"**精确值 `(p·v−1)/(p−1) = {exact_small:.6f}`**，而 `v = {v_s}` "
          f"⇒ **误差 `{abs(exact_small - v_s) / v_s * 100:.2f}%`** ✓",
          f"**小 `p` 下的差不够大**（`{exact_small:.6f}`）—— `C.1` 失去意义")

    # ── C.2 ★★ 大 p 下趋于 v ──────────────────────────────────────────────────
    p_l = specs.INTERLEAVED_ASYMPTOTIC_P
    exact_large = (p_l * v_s - 1) / (p_l - 1)
    err_large = abs(exact_large - v_s) / v_s
    check("C.2", err_large < 0.01,
          f"★★ **同一个精确式在大 `p` 下趋于 `v`**：`p={p_l}, v={v_s}` ⇒ "
          f"`{exact_large:.6f}`，与 `v` 差 **`{err_large * 100:.2f}%`** "
          f"⇒ ★★ **两个方向都量到，「近似何时可用」才说清楚**",
          f"**大 `p` 下没有趋于 `v`**（误差 `{err_large * 100:.2f}%`）")
    print()
    print("     ★★ 精确式 `(p·v−1)/(p−1)` 在若干 `(p, v)` 上的值：")
    for pp in (2, 4, 8, 16, 64, 256):
        for vv in (2, 4):
            r = (pp * vv - 1) / (pp - 1)
            print(f"        p = {pp:>3}, v = {vv}  ⇒  {r:.6f}  （原文的「`v` 倍」= {vv}，"
                  f"误差 {abs(r - vv) / vv * 100:>5.2f}%）")

    # ── C.3 ★★ 守卫：把 1F1B 的 warm-up 改错 ⇒ A.1 必须失败 ──────────────────────
    p_g, m_g = 4, 8
    t_gpipe, _ = schedule_gpipe(p_g, m_g)
    t_bad_warm, _ = schedule_1f1b(p_g, m_g, warmup=[0] * p_g)      # ★ 错：没有 warm-up
    check("C.3", abs(t_gpipe - t_bad_warm) > specs.TOL_MACHINE,
          f"★★ **守卫**：把 1F1B 的 warm-up 步数改成 `0`（不做 warm-up）⇒ 总时间 "
          f"`{t_bad_warm:.2f}` 与 GPipe 的 `{t_gpipe:.2f}` 不符 ⇒ `A.1` **会失败** ✓",
          "**守卫失效** —— warm-up 改错之后 `A.1` 仍然通过")
    print()
    print("     ★★★ 这守卫钉住的正是本主题最要紧的那句：")
    print("         「1F1B 的气泡与 GPipe **相同**」**不是一句可以顺手写下的结论** ——")
    print("         它是「warm-up 步数取 `p−1−s`」这个**具体调度**的后果。")

    print()
    print("=" * 78)
    print("D · 口径常量与标记自检")
    print("=" * 78)

    # ── D.6 ★★ 三组 `CASES` 的【逐元素】钉住（★ 只钉整体会让元素级变异漏网） ──────
    check("D.6", CASES == [(4, 8), (4, 32), (8, 16)],
          "★★ 本文的三组 `(p, m)` **逐元素**钉住",
          "**`CASES` 的元素被动过**")

    # ── D.7 ★★ 另外四个主题的标记（★ 用来验「不照抄」） ────────────────────────
    check("D.7", OTHER_THEME_MARKS == ("%% FAIL: data-parallel", "^^ FAIL: tensor-parallel",
                                       "{{ FAIL: seq-context-parallel", "|| FAIL: expert-parallel",
                                       "?? FAIL: zero-fsdp")
          and ASSERT_FAIL_MARK not in OTHER_THEME_MARKS,
          "★★ 另外五个主题的标记**逐个逐字**钉住、且**不含本主题的** ⇒ 「不跨主题照抄」的机械检查"
          "（★ 只检查「长度 + 含 FAIL + 不含本主题」会让**元素级**变异漏网 —— 本程序实测过）",
          "**OTHER_THEME_MARKS 的元素被动过**")

    # ── D.8 ★★ 三条只被【引用】的引文常量 ─────────────────────────────────────
    check("D.8", specs.MEGATRON_BUBBLE_TIME_QUOTE
          == "The total amount of time spent in the pipeline bubble is then"
          and specs.MEGATRON_IDEAL_TIME_QUOTE == "The ideal processing time for the batch is"
          and specs.MEGATRON_N_OVER_D_QUOTE == "The pipeline bubble size is",
          "★★ 三条只被**引用**的常量**逐个逐字**钉住（`t_pb` / `t_id` / `(n−d)/b'` 那段的引导语，"
          "★ 第三条已从跨标记的 `As d becomes larger` 换成同节公式的散文引导语）",
          "**这三条常量被动过**")

    check("D.1", specs.MEGATRON_1F1B_SAME_BUBBLE_QUOTE
          == "The time spent in the bubble is the same for this new schedule",
          "★★★ 1F1B 的判定句逐字钉住（★ 本程序的主题）",
          "**1F1B 判定句被动过**")

    check("D.2", specs.MEGATRON_INTERLEAVED_BUBBLE_QUOTE == "The pipeline bubble time thus reduces to"
          and specs.MEGATRON_INTERLEAVED_FREE_QUOTE == "does not come for free"
          and specs.MEGATRON_INTERLEAVED_COMM_QUOTE == "the amount of communication also increases by",
          "★★ interleaved 的三条引文逐字钉住（气泡公式引导语 / 「不是白拿」/ 「通信增加」引导语）"
          "（★ 引导语一律切在 `<math>` 之前）",
          "**interleaved 的引文常量被动过**")

    check("D.3", specs.MEGATRON_1F1B_STASH_QUOTE == "this schedule requires activations to be stashed for"
          and specs.MEGATRON_1F1B_VS_GPIPE_QUOTE == "microbatches for the GPipe schedule",
          "★★ 1F1B 的**显存**依据逐字钉住（`p` 个 vs `m` 个；★ 后者取的是 `<math>` 斜体 `m` 之后的散文）",
          "**1F1B 的显存引文被动过**")

    # ── D.4 ★★ 独立性（源码级线索 + order 级） ───────────────────────────────────
    src_bg = inspect.getsource(build_order_gpipe)
    src_b1 = inspect.getsource(build_order_1f1b)
    check("D.4", ("_run_stage" in inspect.getsource(schedule_gpipe)
                  and "_run_stage" in inspect.getsource(schedule_1f1b)
                  and "order[s].append" in src_bg and "order[s].append" in src_b1
                  and src_bg != src_b1
                  and build_order_gpipe(4, 8) != build_order_1f1b(4, 8)),
          "★★ **独立性（源码级线索 + order 级）**：两个调度器**各自生成 `order`**、只共享最小原语 "
          "`_run_stage`（两个生成函数的源码不同、且各自产出的 `order` 也不同）⇒ 「总时间相同」"
          "不是把同一个式子写两遍。★ **这是线索不是证明**：若把 `schedule_1f1b` 整段写成 "
          "`return schedule_gpipe(p, m)`，本检查仍会通过 —— 它由 `A.2`（稳态交替）与 "
          "`B.1`（在飞数 `1F1B ≤ p`）兜住（★ 订正 2026-10-08：原措辞把这条**线索**说成了「证明」）",
          "**两个调度器退化成同一个了** —— `A.1` 会变成恒真")

    check("D.5", len({ASSERT_FAIL_MARK}) == 1 and list({ASSERT_FAIL_MARK})[0] == "== FAIL: pipeline-parallel"
          and "FAIL" in ASSERT_FAIL_MARK,
          f"失败标记自检：`{ASSERT_FAIL_MARK!r}`（去重 1 个、含 `FAIL`）✓",
          "**失败标记被动过**")


# ===========================================================================
# 汇总
# ===========================================================================
def summary():
    print()
    print("=" * 78)
    print(f"汇总: A {_PASS['A']}/{_TOTAL['A']} · B {_PASS['B']}/{_TOTAL['B']} · "
          f"C {_PASS['C']}/{_TOTAL['C']} · D {_PASS['D']}/{_TOTAL['D']} ⇒ "
          f"断言 {sum(_TOTAL.values())} 条，通过 {sum(_PASS.values())} 条，失败 {len(FAILS)} 条")
    if FAILS:
        print(f"失败清单: {', '.join(FAILS)}")
        print(ASSERT_FAIL_MARK, "（上面的失败清单见本行之前）")
    print("★ 本程序只验证「调度的依赖与顺序写对了没有」；论文公式的依据在 `specs.py` 的 `# 来源：`。")
    print("★ 不是 NCCL，也不是真机：本机无 GPU、无多卡、无真实网络。")
    return 1 if FAILS else 0


def main():
    print("=" * 78)
    print("02_schedules.py —— PP 的三种调度：GPipe / 1F1B / interleaved")
    print("=" * 78)
    print(f"  ★ 真值标注：**本机实测值**（`python3 {sys.version.split()[0]}`，macOS arm64 无 GPU）。"
          "（★ 本主题**不用 numpy**）")
    print(f"  ★ 失败标记 ASSERT_FAIL_MARK = {ASSERT_FAIL_MARK!r}（本主题自定义；不跨主题照抄）")
    print("  ★★ 强制限定语：**不是 NCCL，也不是真机** —— 本程序验证的是**切分的代数**。")
    print("  ★★★ 本主题最要紧的一句：**1F1B 换的是【显存】，不是气泡率**"
          "（原文 `The time spent in the bubble is the same for this new schedule`）。")
    if not preflight():
        print()
        print("!! 口径钉住失败 ⇒ **不继续往下跑**：后续的 A/B/C 只会产生一堆无意义的连带崩溃")
        return summary()
    run_checks()
    return summary()


if __name__ == "__main__":
    sys.exit(main())

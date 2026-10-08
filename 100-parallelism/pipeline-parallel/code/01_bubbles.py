#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""01_bubbles.py —— PP 的**微批与气泡率**，以及★★★ **气泡率的两个口径**。

配套文档：`01-microbatches-and-bubbles.md`。

★ 三类断言（A 算法正确性 / B 代价模型 / C 边界与反例）**全部在本程序内部**，
失败时以**非零退出码**结束；**没有 `verify.py`**（`CONTRIBUTING` §2 表里的第二种情形）。

★★ **本程序的数字全部是「本机实测值」**，测量条件是：
> `python3 3.13.5`，macOS arm64，**无 GPU、无多卡、无 NCCL、无真实网络**。
> ★★ 强制限定语：**「不是 NCCL，也不是真机」**。
★★ **本程序【不用 numpy】**（★ 订正 2026-10-08：原先有一行 `import numpy as np` 却**一次都没用**）：
  它全是**离散事件模拟 + 纯浮点算术**，本主题的三个程序都不需要任何第三方库
  ⇒ `requirements.txt` 与 `Makefile` 的 `check-env` 都**不**要求 numpy（见那两处的说明）。

★★★ **本主题的头号风险是「模拟器同时是被测对象」**（计划 §1.3.3）：
  本程序的核心是一个**离散事件模拟器**，而它的参照是**闭式公式** ——
  若两者都由同一个式子生成，判据就**恒真**。★ 本程序的处置：
    · **模拟器【只依赖两条串行约束】**（同一 stage 上 micro-batch 的 fwd 串行、**bwd 也串行**），
      **闭式公式完全不出现**在模拟器里 ⇒ 两者是**独立的**
      （★ `D.3` 是**源码级线索**（AST）；真正的独立性由 `C.1`/`C.2` 两条**行为级**守卫兜底）；
    · ★★ **`C.1`/`C.2` 是这条独立性的守卫**：**把模拟器的 bwd 串行约束删掉 ⇒ 气泡率变负**
      ⇒ `A.1`/`C.1` 必须失败（本库控制者实测漏约束时气泡率 `-0.33`）。

★★★ **一条判据性的自查**（计划 §1.3.3，放在每个 `code/` 的头部）：
  > **问：如果我把「被测的那段逻辑」删掉，这条断言还会通过吗？**
  > 答「**会**」⇒ 它是自证的，**不是判据**。
  ★ 本程序的答案落点：`A.1` 若删掉 `simulate()` ⇒ 总时间无从算起（不会通过）；
    `A.2`/`B.1`/`B.2`/`B.5` 同理。
  ★★★ **订正 2026-10-08（文案，不是代码）**：原程序头把「`p = 1` 时气泡为 0」说成
    「`D.2` 故意标出的**一条恒真**，**不当 A 类用**、是一条 P 类」—— ★ **三处全错**：
    · 做这件事的是 **`A.2`**（★ 程序里的 `D.2` 是**引文钉住**，与 `p=1` 无关）；
    · **`A.2` 计入 A 类**（汇总行是 `A 3/3`）；
    · ★★ **`A.2` 并不是恒真的** —— 把 `simulate()` 的依赖方向改反，它**就会失败**
      ⇒ **它就是一条正常的 A 类退化对照**，不是「P 类」也不是「不当 A 类用」。

★★ **覆盖边界**：本程序的 A 类守的是「**模拟器的调度约束写全了没有**」
  （串行约束、依赖方向、起止时刻），**不守**「闭式公式对不对」——
  ★★ **后者的依据是两篇论文的逐字**（`specs.py` 的 `# 来源：`），**不是本程序**。
  ★★ 且 **C 类不得装「成功行为」**（`CONTRIBUTING:238`）：本程序的 `C.1`/`C.2` 断言的是
  「**漏约束 ⇒ 气泡率变负**」这个**失败**，不是「一致」这个成功。

★★ **一条口径声明**（它必须在正文里也写明）：
  本程序把 `t_f` 与 `t_b` 取成 **`1.0` 与 `2.0`**（`specs.UNIT_F`/`UNIT_B`）——
  ★★ **那是本库的设计假设**（「后向是前向两倍」），**不是任何模型的实测** ⇒
  本程序的一切气泡数**只在这一组假设下成立**，引用时必须带这一句。

★★★ **本主题的禁区（回链，一行都不重写）**：
  · **重算换显存**的机制（`O(√n)` 内存 vs `≈2n` 前向次数）→
    `90-training/autograd-and-backward/02-gradient-checkpointing.md`
    （★ 本程序**只引** `GPIPE_RECOMPUTE_CAUSAL_QUOTE` 那一句的**因果**，
      **不写**重算的机制）；
  · **transformer block 的结构** → `80-model-architectures/transformer/`；
  · **点对点通信的实现**与**通信-计算重叠的机制** → `110-communication/`（**未建成**）。

★★ 可证伪断言（把下面任一条改错，本程序必须非零退出）：
  · `A.1`  ★★★ 离散事件模拟 ↔ 闭式 `(m+p−1)·(t_f+t_b)`（**8 组**，差 ≤ 机器精度档）；
  · `A.2`  ★★ `p = 1` 时气泡率**恰为 `0`**（退化对照：没有流水线就没有气泡）；
  · `B.1`  ★★ `1 − 利用率 == (p−1)/(m+p−1)`（**GPipe 口径**：分母是**总时间**）；
  · `B.2`  ★★★ **换算恒等式** `b_megatron == b_gpipe/(1 − b_gpipe)`（**两个口径不矛盾**）；
  · `B.4`  ★★ Megatron 口径的 `(p−1)/m`（分母是**理想时间**）与同一组数自洽；
  · `B.5`  ★★ `(n−d)/b'` 与 `(p−1)/m` 同值，且 **`d` 越大它越小**；
  · `C.1`  ★★★ **漏一条串行约束 ⇒ 气泡率变负** ⇒ 断言「`≥ 0`」**失败**（真的会失败）；
  · `C.2`  ★★ **守卫**：把闭式改成错的 ⇒ `A.1` **必须失败**；
  · `D.*`  口径常量逐个钉住 + **失败标记自检**。
"""
import collections
import os
import sys

import specs

# ===========================================================================
# 断言机制
# ===========================================================================
# ★★ 本主题的失败标记（★ **只允许出现在 `print()` 的字符串里**）。
ASSERT_FAIL_MARK = "== FAIL: pipeline-parallel"

# 本域另外五个主题的失败标记 —— **只用来验「不照抄」**，本程序不打印它们。
OTHER_THEME_MARKS = ("%% FAIL: data-parallel", "^^ FAIL: tensor-parallel",
                     "{{ FAIL: seq-context-parallel", "|| FAIL: expert-parallel",
                     "?? FAIL: zero-fsdp")

FAILS = []
MARKS_USED = set()
_TOTAL = collections.Counter()
_PASS = collections.Counter()


def check(name, cond, detail="", detail_fail=None):
    """`name` 形如 `A.1` —— **首字母就是类别**（A/B/C/D），与 `CONTRIBUTING` §4 对齐。

    ★★★ **失败路径【不复述结论】**（本库 `P12` 家族的一个**输出层**形态）：
      `detail` 一律是**判据成立时**的措辞 ⇒ 若把它原样打在失败行上，输出会**自相矛盾**。
      ⇒ 所以 `cond` 为假时走**另一版文案**：默认只报「这条判据【不成立】」。
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
    """★★★ 口径型常量的钉住 —— **跑在任何计算之前**（计划 §1.3 的 `P23`）。

    ★ 理由：若把口径常量的检查放在计算之后，变异会落进「崩溃捕获」而不算**真捕获**。
    ★★ 本函数**只依赖 `specs`**（不依赖任何计算结果）⇒ 放在最前面。
    """
    before = len(FAILS)
    check("P.1", specs.UNIT_F == 1.0 and specs.UNIT_B == 2.0,
          "口径：t_f = 1.0、t_b = 2.0（**本库设计假设**：后向是前向两倍）",
          "**口径常量被动过** —— 本程序的一切气泡数都建立在 t_f=1.0 / t_b=2.0 上")
    check("P.2", len(specs.BUBBLE_GRID_P) == len(specs.BUBBLE_GRID_M) == 8,
          "`BUBBLE_GRID_P` 与 `BUBBLE_GRID_M` 等长（各 8 组）",
          "**两组网格长度不等** —— 按位配对会静默错位")
    check("P.3", specs.BUBBLE_GRID_P[-1] == 1,
          "网格最后一组是 `p = 1`（**退化对照**，`A.2` 要用）",
          "**退化对照组不在了** —— `A.2` 会失去意义")
    check("P.4", specs.SANITY_NEG_BUBBLE_EXPECTED == 0.0,
          "气泡率的**物理下界**是 `0.0`（气泡率没有负的）",
          "**下界被动过** —— `C.1` 的断言失去锚")
    check("P.5", specs.TOL_MACHINE > 0 and specs.TOL_MACHINE < 1e-6,
          f"机器精度档阈 = {specs.TOL_MACHINE:g}（只用于**浮点**判据）",
          "**阈被动过** —— 它不再是「机器精度档」")
    return len(FAILS) == before


def run_checks():
    """A/B/C/D 四类断言（★ 顺序跑；`_TOTAL`/`_PASS`/`FAILS` 是模块级计数器）。"""

    # ===========================================================================
    # 模拟器 —— ★★ **只依赖两条串行约束**，闭式公式**完全不出现**在里面
    # ===========================================================================
    def simulate(p, m, t_f=None, t_b=None, fwd_serial=True, bwd_serial=True):
        """离散事件模拟：`p` 级流水线、`m` 个 micro-batch。

        ★★★ **它只编码两条约束**（这是它与闭式的**独立性**的全部来源）：
          · 一个 micro-batch 的 stage `s` 的**前向**必须等：① stage `s-1` 的前向完成；
            ② 同一 stage 上**前一个** micro-batch 的前向完成
            （★ `fwd_serial=False` 会**删掉这一条** —— `C.2` 用它）。
          · **后向**必须等：① stage `s+1` 的后向完成；② 同一 stage 上**后一个**
            micro-batch 的后向完成（★ `bwd_serial=False` 会**删掉这一条** —— `C.1` 用它）。

        ★ **返回**（`总时间`, `每 stage 的忙时间`）—— 忙时间 = 该 stage 真正干活的时间。
        """
        t_f = specs.UNIT_F if t_f is None else t_f
        t_b = specs.UNIT_B if t_b is None else t_b

        # 前向：fin[s][i] = micro-batch i 在 stage s 上的前向**结束**时刻
        fin = [[0.0] * m for _ in range(p)]
        for i in range(m):
            for s in range(p):
                start = 0.0 if s == 0 else fin[s - 1][i]
                if fwd_serial and i > 0:
                    start = max(start, fin[s][i - 1])          # ★ 约束①（同 stage 串行）
                fin[s][i] = start + t_f

        # 后向：bin_[s][i] = micro-batch i 在 stage s 上的后向**结束**时刻
        bin_ = [[0.0] * m for _ in range(p)]
        for i in reversed(range(m)):
            for s in reversed(range(p)):
                end = fin[s][i]
                if s < p - 1:
                    end = max(end, bin_[s + 1][i])             # 依赖下游
                if bwd_serial and i < m - 1:
                    end = max(end, bin_[s][i + 1])             # ★ 约束②（同 stage 的 bwd 串行）
                bin_[s][i] = end + t_b

        total = max(bin_[s][i] for s in range(p) for i in range(m))
        busy = m * (t_f + t_b)                                 # 每个 stage 的忙时间（与 p 无关）
        return total, busy


    def bubble_gpipe(p, m):
        """**GPipe 口径**的气泡占比：气泡 ÷ **总时间** = `(p−1)/(m+p−1)`。"""
        return (p - 1) / (m + p - 1)


    def bubble_megatron(p, m):
        """**Megatron 口径**的气泡占比：气泡 ÷ **理想时间** = `(p−1)/m`。"""
        return (p - 1) / m


    print("=" * 78)
    print("A · 算法正确性：模拟 ↔ 闭式")
    print("=" * 78)

    # ── A.1 ★★★ 离散事件模拟 ↔ 闭式 ─────────────────────────────────────────
    rows = []
    worst = 0.0
    for p, m in zip(specs.BUBBLE_GRID_P, specs.BUBBLE_GRID_M):
        total, busy = simulate(p, m)
        closed = (m + p - 1) * (specs.UNIT_F + specs.UNIT_B)
        worst = max(worst, abs(total - closed))
        rows.append((p, m, total, closed))
    check("A.1", worst <= specs.TOL_MACHINE,
          f"8 组 `(p,m)`：模拟总时间 == 闭式 `(m+p−1)(t_f+t_b)`（最大差 {worst:.3e}）",
          f"**模拟与闭式不符**（最大差 {worst:.3e}）—— ★ 先查串行约束写全了没有")

    print()
    print(f"     {'p':>3} {'m':>4} {'模拟总时间':>12} {'闭式':>12} {'利用率':>10} {'1−利用率':>10} {'(p−1)/(m+p−1)':>15}")
    for p, m, total, closed in rows:
        util = (m * (specs.UNIT_F + specs.UNIT_B)) / total
        print(f"     {p:>3} {m:>4} {total:>12.2f} {closed:>12.2f} {util:>10.6f} "
              f"{1 - util:>10.6f} {bubble_gpipe(p, m):>15.6f}")

    # ── A.2 ★★ 退化对照：p = 1 ────────────────────────────────────────────────
    p1 = specs.BUBBLE_GRID_P[-1]
    m1 = specs.BUBBLE_GRID_M[-1]
    tot1, busy1 = simulate(p1, m1)
    check("A.2", p1 == 1 and abs(1 - busy1 / tot1) <= specs.TOL_MACHINE,
          f"`p = {p1}`（无流水线）⇒ 利用率恰为 `1.000000`（气泡率 0）",
          "**`p=1` 时出现气泡** —— 那说明模拟器把依赖方向搞反了")

    # ── A.1g ★★ 守卫：把闭式改错 ⇒ A.1 必须失败 ────────────────────────────────
    _bad = (m1 + p1 - 1) * (specs.UNIT_F + specs.UNIT_B) + 1.0
    check("A.1g", abs(tot1 - _bad) > specs.TOL_MACHINE,
          "守卫：把闭式加 `1.0` ⇒ 与模拟的差**超出**阈（`A.1` 会失败）",
          "**守卫失效** —— 闭式改错之后 `A.1` 仍然通过 ⇒ `A.1` 是恒真的")

    print()
    print("=" * 78)
    print("B · 代价模型：两个口径 + 换算恒等式")
    print("=" * 78)

    # ── B.1 ★★ GPipe 口径 ────────────────────────────────────────────────────
    worst_g = 0.0
    for p, m, total, closed in rows:
        util = (m * (specs.UNIT_F + specs.UNIT_B)) / total
        worst_g = max(worst_g, abs((1 - util) - bubble_gpipe(p, m)))
    check("B.1", worst_g <= specs.TOL_MACHINE,
          f"**GPipe 口径**：`1 − 利用率` == `(p−1)/(m+p−1)`（最大差 {worst_g:.3e}）",
          f"**GPipe 口径不符**（最大差 {worst_g:.3e}）")

    # ── B.2 ★★★ 换算恒等式（两个口径不矛盾） ───────────────────────────────────
    worst_c = 0.0
    for p, m in zip(specs.BUBBLE_GRID_P, specs.BUBBLE_GRID_M):
        bg = bubble_gpipe(p, m)
        worst_c = max(worst_c, abs(bubble_megatron(p, m) - bg / (1 - bg)))
    check("B.2", worst_c <= specs.TOL_MACHINE,
          f"★★★ **换算恒等式** `b_megatron == b_gpipe/(1 − b_gpipe)`（最大差 {worst_c:.3e}）",
          f"**换算恒等式不成立**（最大差 {worst_c:.3e}）—— 两个口径的分母搞混了")

    print()
    print(f"     {'p':>3} {'m':>4} {'b_gpipe（÷总时间）':>20} {'b_megatron（÷理想时间）':>24} {'b_g/(1−b_g)':>14}")
    for p, m in zip(specs.BUBBLE_GRID_P, specs.BUBBLE_GRID_M):
        bg = bubble_gpipe(p, m)
        print(f"     {p:>3} {m:>4} {bg:>20.6f} {bubble_megatron(p, m):>24.6f} {bg / (1 - bg):>14.6f}")

    # ── B.3 ★ `M ≥ 4K` 时气泡占比已经很小 ─────────────────────────────────────
    k4 = specs.GPIPE_NEGLIGIBLE_M_OVER_K
    p4, m4 = 4, k4 * 4
    t4, b4 = simulate(p4, m4)
    u4 = b4 / t4
    check("B.3", (m4 / p4) == k4 and (1 - u4) < 0.25,
          f"`M = {k4}·K`（`p={p4}, m={m4}`）⇒ 气泡占比 `{1 - u4:.4f} < 0.25`"
          f"（原文：`M ≥ {k4}×K` 时可忽略，★ 那是**实验观察**不是定理）",
          "**`M ≥ 4K` 时气泡仍然很大** —— 或那组参数没算对")

    # ── B.4 ★★ Megatron 口径 ─────────────────────────────────────────────────
    worst_m = 0.0
    for p, m, total, closed in rows:
        ideal = m * (specs.UNIT_F + specs.UNIT_B)
        bubble_time = total - ideal
        worst_m = max(worst_m, abs(bubble_time / ideal - bubble_megatron(p, m)))
    check("B.4", worst_m <= specs.TOL_MACHINE,
          f"**Megatron 口径**：`气泡/理想时间` == `(p−1)/m`（最大差 {worst_m:.3e}）",
          f"**Megatron 口径不符**（最大差 {worst_m:.3e}）")

    # ── B.5 ★★ (n−d)/b' 的换写，以及「d 越大气泡越小」 ──────────────────────────
    N_GPU, B_PRIME = 64, 32
    worst_nd = 0.0
    seq = []
    for d in (2, 4, 8, 16, 32):
        p_nd = N_GPU / d
        m_nd = B_PRIME / d
        lhs = (p_nd - 1) / m_nd
        rhs = (N_GPU - d) / B_PRIME
        worst_nd = max(worst_nd, abs(lhs - rhs))
        seq.append((d, lhs))
    check("B.5", worst_nd <= specs.TOL_MACHINE and all(seq[i][1] > seq[i + 1][1] for i in range(len(seq) - 1)),
          f"★★ `(p−1)/m == (n−d)/b'`（最大差 {worst_nd:.3e}）**且 `d` 越大它越小**"
          f"（`d=2 → {seq[0][1]:.6f}` … `d=32 → {seq[-1][1]:.6f}`）",
          "**换写不成立、或 `d` 的单调性不成立**")
    print()
    print("     ★★ `(p−1)/m = (n−d)/b'`：在**固定总卡数** `n` 下，**把卡多分给 DP 会减小 PP 气泡**")
    for d, v in seq:
        print(f"        d = {d:>3}  ⇒  (n−d)/b' = {v:.6f}")

    print()
    print("=" * 78)
    print("C · 边界与反例")
    print("=" * 78)

    # ── C.1 ★★★ 漏一条串行约束 ⇒ 气泡率变负 ────────────────────────────────────
    pc, mc = 4, 8
    tot_ok, busy_c = simulate(pc, mc, bwd_serial=True)
    tot_bad, _ = simulate(pc, mc, bwd_serial=False)
    rate_ok = 1 - busy_c / tot_ok
    rate_bad = 1 - busy_c / tot_bad
    check("C.1", rate_ok >= specs.SANITY_NEG_BUBBLE_EXPECTED and rate_bad < specs.SANITY_NEG_BUBBLE_EXPECTED,
          f"★★★ **漏掉「同 stage 的 bwd 也必须串行」这条约束 ⇒ 气泡率变负**："
          f"完整 `{rate_ok:+.6f}` vs 漏约束 `{rate_bad:+.6f}` ⇒ 断言「`≥ 0`」**失败** ✓"
          f"（★ 控制者本人踩过这个坑：`-0.33`）",
          f"**漏约束之后气泡率没有变负**（`{rate_bad:+.6f}`）—— 那条守卫失去作用")

    # ── C.2 ★★ 守卫：漏掉「同 stage 的 fwd 串行」 ⇒ A.1 必须失败 ────────────────
    tot_nf, _ = simulate(pc, mc, fwd_serial=False)
    closed_c = (mc + pc - 1) * (specs.UNIT_F + specs.UNIT_B)
    check("C.2", abs(tot_nf - closed_c) > specs.TOL_MACHINE,
          f"守卫：漏掉「同一 stage 上 micro-batch 的**前向也必须串行**」⇒ "
          f"总时间 `{tot_nf:.2f}` **小于**闭式 `{closed_c:.2f}`（**把流水线算快了**）⇒ `A.1` 会失败 ✓",
          "**守卫失效** —— 漏掉 fwd 串行之后 `A.1` 仍通过")
    print()
    print("     ★★ **为什么守卫选「漏 fwd 串行」而不是「对调 `t_f`/`t_b`」**：")
    print("        闭式 `(m+p−1)·(t_f+t_b)` 对 `t_f` 与 `t_b` 是**对称**的 ⇒ "
          "**对调它们不改变任何数**，那样的「守卫」**恒真**（本程序的第一版就写错成它了）。")
    print("        ⇒ ★★★ **守卫必须选一个【真的会改变被判量】的错法** —— 这是 `CONTRIBUTING:235` 的落法。")

    # ── C.3 ★ 重算与气泡的【因果】是「调度自由度」（引文，不是实测） ──────────────
    check("C.3", "scheduled earlier" in specs.GPIPE_RECOMPUTE_CAUSAL_QUOTE,
          "★ 重算与气泡的因果：原文说重算**可以被更早地调度** —— "
          "★ **不是**「重算降低气泡率」（★ 常见转述把它说成后者，那是错的）",
          "**引文被动过** —— 那句因果的逐字依据不在了")
    print()
    print("     ★★ 为什么 `C.3` 是【引用检查】而不是【本机实测】：")
    print("        重算对气泡的真实影响需要**真的重算**才能量，而本机没有可重算的前向")
    print("        ⇒ 本程序**只引**原文那一句的因果、**不声称**任何关于它幅度的结论。")

    print()
    print("=" * 78)
    print("D · 口径常量与标记自检")
    print("=" * 78)

    # ── D.1 口径常量逐个钉住（`==`，不是 `in`） ─────────────────────────────────
    check("D.1", specs.GPIPE_BUBBLE_QUOTE == "This bubble time is"
          and specs.GPIPE_NEGLIGIBLE_QUOTE == "we found the bubble overhead to be negligible when"
          and specs.GPIPE_NEGLIGIBLE_M_OVER_K == 4,
          "GPipe 两条引文的**引导语**与 `4` 逐字钉住（★ **两条都切在 `<math>` 之前**，"
          "宾语公式跨标记 ⇒ 只取散文；★ **`O(...)` 是 Big-O，不是等式**；"
          "★ `M≥4×K` 那一段在 `<math>` 里、渲染文本**无空格**）",
          "**GPipe 的引文常量被动过**")

    check("D.2", specs.MEGATRON_1F1B_SAME_BUBBLE_QUOTE == "The time spent in the bubble is the same for this new schedule"
          and specs.MEGATRON_ABS_COMMENT_QUOTE == "Accepted to SC 2021",
          "Megatron 的两条关键引文逐字钉住（★ 1F1B 判定句 + **abs 页的同行评审依据**）",
          "**Megatron 的引文常量被动过**")

    # ── D.3 ★★ 独立性证据：闭式**不出现**在模拟器里（★ AST 级，2026-10-08 加强） ──
    import ast
    import inspect
    import textwrap
    src = inspect.getsource(simulate)
    # ★★★ 加强理由（订正 2026-10-08）：原先只查两种**拼法**的字面量
    #   （`"m + p - 1"` / `"(m+p-1)"`）⇒ 在 `simulate()` 里加一行 `_x = p + m - 1`
    #   （换序、或任何等价的算术）**抓不到**。⇒ 现在另加一条 **AST 级**判据：
    #   `simulate()` 的**表达式节点**里，**没有任何一个**同时用到名字 `m` 与 `p`
    #   （闭式的任何算术形态都必须把这两个名字放进同一个 `BinOp`/`Compare`/… 节点）。
    #   ★★ 这条判据**仍然不是证明**（它拦不住 `_x = m; _x += p` 这种跨语句写法）——
    #      它是**源码级线索**；真正的独立性由 `C.1`（漏约束 ⇒ 气泡率变负）与
    #      `C.2`（漏 fwd 串行 ⇒ 总时间偏小）两条**行为级**守卫兜底。
    _EXPR_NODES = (ast.BinOp, ast.AugAssign, ast.Compare, ast.BoolOp,
                   ast.IfExp, ast.UnaryOp)
    _both_mp = []
    for _n in ast.walk(ast.parse(textwrap.dedent(src))):
        if isinstance(_n, _EXPR_NODES):
            _names = {x.id for x in ast.walk(_n) if isinstance(x, ast.Name)}
            if {"m", "p"} <= _names:
                _both_mp.append(type(_n).__name__)
    has_closed = ("m + p - 1" in src) or ("(m+p-1)" in src) or ("p + m - 1" in src)
    check("D.3", not has_closed and not _both_mp,
          "★★ **独立性证据（源码级线索）**：闭式 `(m+p−1)` 既**不作为字面量出现**、"
          "也**没有任何单个表达式同时用到 `m` 与 `p`**（AST 遍历 `simulate()` 的表达式节点）"
          "⇒ 模拟器不可能把闭式算出来（★ 另加两条**行为级**守卫：`C.1`/`C.2`）",
          "**模拟器里出现了同时用 `m` 与 `p` 的表达式**（或闭式的字面量）"
          "—— 闭式可能被偷偷算了一遍 ⇒ `A.1` 会退化成恒真")

    # ── D.4 ★ `p` 与 `m` 是【整数】（防止有人改成浮点让闭式退化） ────────────────
    check("D.4", all(isinstance(x, int) for x in specs.BUBBLE_GRID_P + specs.BUBBLE_GRID_M),
          "网格里的 `p` 与 `m` **全是整数**（★ 浮点会让 `÷` 的下取整口径漂）",
          "**网格里出现了非整数**")

    # ── D.6 ★★ 网格的【逐元素】钉住（★ 只钉「整体」会让**元素级变异漏网**） ──────
    check("D.6", specs.BUBBLE_GRID_P == (4, 4, 4, 4, 8, 8, 16, 1)
          and specs.BUBBLE_GRID_M == (4, 8, 16, 32, 16, 64, 64, 4),
          "★★ 两个网格**逐元素**钉住（8 对 `(p, m)`）—— ★ 只钉「整体」会让元素级变异漏网",
          "**网格的元素被动过**")

    # ── D.7 ★★ 另外四个主题的标记（★ 用来验「不照抄」） ────────────────────────
    check("D.7", OTHER_THEME_MARKS == ("%% FAIL: data-parallel", "^^ FAIL: tensor-parallel",
                                       "{{ FAIL: seq-context-parallel", "|| FAIL: expert-parallel",
                                       "?? FAIL: zero-fsdp")
          and ASSERT_FAIL_MARK not in OTHER_THEME_MARKS,
          "★★ 另外五个主题的标记**逐个逐字**钉住、且**不含本主题的** ⇒ 「不跨主题照抄」的机械检查"
          "（★ 只检查「长度 + 含 FAIL + 不含本主题」会让**元素级**变异漏网 —— 本程序实测过）",
          "**OTHER_THEME_MARKS 的元素被动过**")

    # ── D.8 ★★ 那四条**只被引用、没被逐个钉住**的引文 ─────────────────────────
    check("D.8", specs.GPIPE_RECOMPUTE_CAUSAL_QUOTE
          == "re-computation during the backward pass can be scheduled earlier, "
             "without waiting for the gradients from earlier layers."
          and specs.GPIPE_RECOMPUTE_MEMORY_QUOTE == "peak activation memory requirement is reduced to"
          and specs.GPIPE_NEGLIGIBLE_QUOTE == "we found the bubble overhead to be negligible when"
          and specs.GPIPE_NEGLIGIBLE_M_OVER_K == 4
          and specs.SANITY_NEG_BUBBLE_EXPECTED == 0.0,
          "★★ 五条只被**引用**的常量**逐个逐字**钉住（重算的因果 / 重算的显存引导语 / `M≥4K` 引导语 / `4` / `0.0`）",
          "**这五条常量被动过**")

    # ── D.5 标记自检（★ AST 可读、去重后只有一个、等于本主题的值、含 FAIL） ──────
    _marks = {ASSERT_FAIL_MARK}
    check("D.5", len(_marks) == 1 and list(_marks)[0] == "== FAIL: pipeline-parallel"
          and "FAIL" in list(_marks)[0],
          f"失败标记自检：去重后 `{len(_marks)}` 个、等于 `{list(_marks)[0]!r}`、含 `FAIL` ✓",
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
    print("★ 本程序只验证「模拟器的调度约束写全了没有」；两篇论文的公式依据在 `specs.py` 的 `# 来源：`。")
    print("★ 不是 NCCL，也不是真机：本机无 GPU、无多卡、无真实网络。")
    return 1 if FAILS else 0


def main():
    print("=" * 78)
    print("01_bubbles.py —— PP 的微批与气泡率（★★★ **两个口径**）")
    print("=" * 78)
    print(f"  ★ 真值标注：**本机实测值**（`python3 {sys.version.split()[0]}`，macOS arm64 无 GPU）。"
          "（★ 本主题**不用 numpy**）")
    print(f"  ★ 失败标记 ASSERT_FAIL_MARK = {ASSERT_FAIL_MARK!r}（本主题自定义；不跨主题照抄）")
    print("  ★★ 强制限定语：**不是 NCCL，也不是真机** —— 本程序验证的是**切分的代数**。")
    print("  ★★★ 头号风险是「模拟器同时是被测对象」：本条唯一的活法是「每条 A 类都配一个"
          "把参照故意改错的目标」—— 见 `C.1`/`C.2`。")
    if not preflight():
        print()
        print("!! 口径钉住失败 ⇒ **不继续往下跑**：后续的 A/B/C 只会产生一堆无意义的连带崩溃")
        return summary()
    run_checks()
    return summary()


if __name__ == "__main__":
    sys.exit(main())

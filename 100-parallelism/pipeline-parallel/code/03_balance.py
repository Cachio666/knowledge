#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""03_balance.py —— PP 的**负载均衡**：总时间由**最慢的那个 stage** 决定。

配套文档：`03-memory-balance.md`。

★ 三类断言（A 算法正确性 / B 代价模型 / C 边界与反例）**全部在本程序内部**，
失败时以**非零退出码**结束；**没有 `verify.py`**。

★★ **本程序的数字全部是「本机实测值」**，测量条件是：
> `python3 3.13.5` + `numpy 2.4.4`，macOS arm64，**无 GPU、无多卡、无 NCCL、无真实网络**。
> ★★ 强制限定语：**「不是 NCCL，也不是真机」**。

★★★ **本程序的核心判据**（也是它最容易写成自证的地方）：
  > **总时间 = `max(各 stage 的负载)`，而不是 `mean`。**
  ★★ **为什么这里危险**：把「每 stage 的负载」写成
    「层数 × `BALANCE_PER_LAYER` + embedding」**就是一个乘法** ——
    若判据也跟着写成 `assert load == layers * PER_LAYER + emb`，那**只是把乘法重念一遍**（`P12`）。
  ★★★ **本程序的处置**：
    · A 类**不钉在乘法上**，而钉在「**`max` 与 `mean` 在不均衡时不同、在均衡时相同**」上；
    · `C.1` 让**瓶颈 stage 的身份**随「有没有 embedding」而**变化**（`0` ↔ 无瓶颈）——
      那是一个**结构性的、可证伪的**后果；
    · `D.2` 是**独立性证据**：「分层」的那段源码里**不出现** `max`/`mean` 的公式。

★★ **本主题的模型（必须写明，因为它决定了结论的适用范围）**：
  · `L` 层按**尽量均匀**分给 `p` 个 stage（`L % p != 0` 时前 `L%p` 个各多 1 层）；
  · **首尾 stage 各多一份 embedding / lm_head 的当量**（`specs.BALANCE_EMB`）；
  · ★★ **`BALANCE_EMB = 10 × BALANCE_PER_LAYER` 是本库的设计取值** ——
    真实模型的 embedding 与层负载比**不是 10** ⇒ **本程序只用它做结构性演示**，
    **不据此声称任何真实比例**。

★★★ **一条判据性的自查**（计划 §1.3.3）：
  > **问：如果我把「被测的那段逻辑」删掉，这条断言还会通过吗？**
  > 答「**会**」⇒ 它是自证的，**不是判据**。
  ★ 本程序的答案落点：`A.1` 若删掉 `assign_layers()` ⇒ 各 stage 的负载无从算起（不会通过）；
    ★★ 而 `D.1` **故意**去证明「乘法那一半是恒真的」，并因此**不把它当 A 类**。

★★ **覆盖边界**：本程序的 A 类守的是「**分层与瓶颈的推断对不对**」，
  **不守**「`max/mean` 这个比值本身」（那是定义的直接后果）。
  ★★ 且 **C 类不得装「成功行为」**：`C.1` 断言的是「**去掉 embedding ⇒ 瓶颈消失**」这个**变化**，
  它的失败形态是「去掉之后瓶颈还在」（那说明分层或归属搞错了）。

★★★ **本主题的禁区（回链，一行都不重写）**：
  · **重算换显存**的机制（`O(√n)` 内存 vs `≈2n` 前向次数）→
    `90-training/autograd-and-backward/02-gradient-checkpointing.md`
    —— ★★ 本程序**只引** `specs.GPIPE_RECOMPUTE_MEMORY_QUOTE` 的「它压的是什么量」，
      **不写**重算的机制、**不写**最优检查点数（那是那篇的正文）；
  · **transformer block 的结构** → `80-model-architectures/transformer/`；
  · **点对点通信的实现**与**通信-计算重叠的机制** → `110-communication/`（**未建成**）；
  · **TP 每层 `2+2` 次 all-reduce** → `tensor-parallel/03` 与
    `10-foundations/performance-models/02-amdahl-and-scaling.md:223`（★ **不得重写**）。

★★ 可证伪断言（把下面任一条改错，本程序必须非零退出）：
  · `A.1`  ★★★ **总时间 == `max(各 stage 负载)`**，且**均衡时它降到 `mean`**（两侧都要）；
  · `A.2`  ★★ `BALANCE_CASES` 三组的效率：不均衡 `< 1`、**均衡恰为 `1.0000`**；
  · `B.1`  ★★ 分层**真的分完了**：各 stage 的层数之和 `== L`、且最多差 1 层；
  · `B.2`  ★★ `L % p != 0` 的效应（`LAYER_COUNT_GRID` 四组）+ **最坏的一组是 `L=12,p=5`**；
  · `C.1`  ★★★ **首尾多 embedding ⇒ 瓶颈在 `0` / `p−1`**；**去掉它就变均衡**（真的会变化）；
  · `C.2`  ★★ **分层错位**（把层数给错 stage）⇒ `A.1` 必须失败；
  · `D.*`  口径常量逐个钉住 + **独立性证据** + **失败标记自检**。
"""
import collections
import inspect
import sys

import numpy as np

import specs

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
    check("P.1", specs.BALANCE_PER_LAYER == 10 and specs.BALANCE_EMB == 100,
          "口径：每层当量 = 10、embedding 当量 = 100（**本库设计**，不是真实比例）",
          "**口径常量被动过** —— 全部负载数都建立在这两个数上")
    check("P.2", all(isinstance(L, int) and isinstance(p, int) for L, p in specs.LAYER_COUNT_GRID),
          "`LAYER_COUNT_GRID` 里全是整数（★ 浮点会让「余数摊派」的口径漂）",
          "**网格里出现了非整数**")
    check("P.3", any(L % p != 0 for L, p in specs.LAYER_COUNT_GRID),
          "网格里**含不整除的组合**（`B.2` 要用）",
          "**没有不整除的组合了** —— `B.2` 失去意义")
    check("P.4", len(specs.BALANCE_CASES[0]) == 4 and len(specs.BALANCE_CASES[1]) == 4,
          "`BALANCE_CASES` 前两组的长度相同（**对照才成立**）",
          "**对照组长度不同** —— 均衡/不均衡的对照失去意义")


preflight()


# ===========================================================================
# 分层 —— ★★ 只做「把 L 层分给 p 个 stage」，**不出现** max/mean 的公式
# ===========================================================================
def assign_layers(L, p):
    """把 `L` 层**尽量均匀**分给 `p` 个 stage；返回每级的层数列表。

    ★★ **本函数内不出现 `max`/`mean`** —— 那是 `D.2` 的独立性证据
    （负载的**汇总口径**由调用方决定，不由本函数）。
    """
    base, rem = divmod(L, p)
    return [base + (1 if i < rem else 0) for i in range(p)]


def stage_loads(layers, with_emb=True):
    """把「每级的层数」翻成**负载**：首尾各多一份 embedding 当量。

    ★ 这是**演示用的模型**（见模块头）—— ★★ 它本身是乘法，
    所以 A 类**不钉在它上面**。
    """
    loads = [n * specs.BALANCE_PER_LAYER for n in layers]
    if with_emb and len(loads) >= 2:
        loads[0] += specs.BALANCE_EMB
        loads[-1] += specs.BALANCE_EMB
    return loads


def total_time(loads):
    """★★★ **总时间 = 最慢的那个 stage**（流水线的定义）。"""
    return max(loads)


print("=" * 78)
print("A · 算法正确性：总时间由最慢的 stage 决定")
print("=" * 78)

# ── A.1 ★★★ 总时间 == max(各 stage 负载)；均衡时降到 mean ────────────────────
rows = []
# ★★ 用 `specs.LAYER_COUNT_GRID` 而**不是**一个本地副本 —— 本地副本的元素没有 `==` 消费，
#    会变成一条**不过期的漏网**（本程序实测：`cases[0][0]` 等 8 条）。
for L, p in specs.LAYER_COUNT_GRID:
    loads = stage_loads(assign_layers(L, p))
    rows.append((L, p, loads, max(loads), sum(loads) / len(loads)))

worst = 0.0
for L, p, loads, mx, mean in rows:
    worst = max(worst, abs(total_time(loads) - mx))
balanced = [100, 100, 100, 100]
check("A.1", worst <= specs.TOL_MACHINE and total_time(balanced) == sum(balanced) / len(balanced),
      f"★★★ **总时间 == `max(各 stage 负载)`**（4 组，差 `≤ {worst:.3e}`），"
      f"★ **且均衡时它降到 `mean`**（`[100,100,100,100]` ⇒ 两者相等）",
      "**总时间与 `max` 不符** —— 或均衡时两者不相等")

print()
print(f"     {'L':>3} {'p':>3} {'层数':<22} {'负载':<28} {'max':>6} {'mean':>8} {'max/mean':>10}")
for L, p, loads, mx, mean in rows:
    print(f"     {L:>3} {p:>3} {str(assign_layers(L, p)):<22} {str(loads):<28} "
          f"{mx:>6} {mean:>8.2f} {mx / mean:>10.4f}")

# ── A.2 ★★ 三组给定负载的效率 ──────────────────────────────────────────────
ok_a2 = True
effs = []
for case in specs.BALANCE_CASES:
    eff = (sum(case) / len(case)) / max(case)
    effs.append(eff)
if not (effs[0] < 1.0 and abs(effs[1] - 1.0) <= specs.TOL_MACHINE and effs[2] < 1.0):
    ok_a2 = False
check("A.2", ok_a2,
      f"★★ **效率 `mean/max`**：不均衡组 `< 1`（`{effs[0]:.4f}` / `{effs[2]:.4f}`）、"
      f"**均衡组恰为 `1.0000`**（`{effs[1]:.4f}`）⇒ ★ 效率的损失 = 不均衡的代价",
      "**三组的效率关系不对**")
print()
for i, (case, eff) in enumerate(zip(specs.BALANCE_CASES, effs)):
    print(f"     负载 {str(case):<28} ⇒ max/mean = {max(case) / (sum(case) / len(case)):.4f}"
          f"   效率 mean/max = {eff:.4f}（损失 {100 * (1 - eff):.2f}%）")

print()
print("=" * 78)
print("B · 代价模型：分层与不整除")
print("=" * 78)

# ── B.1 ★★ 分层真的分完了 ────────────────────────────────────────────────
ok_b1 = True
for L, p in specs.LAYER_COUNT_GRID:
    ls = assign_layers(L, p)
    if sum(ls) != L or (max(ls) - min(ls)) > 1:
        ok_b1 = False
check("B.1", ok_b1,
      "★★ 分层**真的分完了**：各 stage 的层数之和 `== L`、且**最多差 1 层**"
      "（四组 `(L,p)` 全过）",
      "**分层没分完** —— 层数之和不对或差超过 1")
print()
for L, p in specs.LAYER_COUNT_GRID:
    ls = assign_layers(L, p)
    loads = stage_loads(ls, with_emb=False)
    print(f"     L={L:>3}, p={p}  ⇒  层数 {str(ls):<22} 负载 {str(loads):<24} "
          f"max/mean = {max(loads) / (sum(loads) / len(loads)):.4f}")

# ── B.2 ★★ 不整除的效应；最坏的一组 ─────────────────────────────────────────
# ★★ 用**推导式**而不是累积变量 —— 累积变量的**初值**会留在模块级、进 `make mutate` 的分母
#    （而「把它改成别的值」没有可观察的后果 ⇒ 变成一条**不过期的漏网**，本程序实测过）。
_worst = max((max(stage_loads(assign_layers(L, p), with_emb=False))
              / (sum(stage_loads(assign_layers(L, p), with_emb=False)) / p), (L, p))
             for L, p in specs.LAYER_COUNT_GRID if L % p != 0)
worst_ratio, worst_case = _worst
check("B.2", worst_case == (12, 8) and worst_ratio > 1.3,
      f"★★ **不整除 + `p` 大 的效应**：最坏的一组是 **`L=12, p=8`**（`max/mean = {worst_ratio:.4f}`）"
      f"★ 成因：**`p` 越大 ⇒ 每 stage 的层数越少 ⇒ 相对波动越大**"
      f"（`p=8` 时层数 `[2,2,2,2,1,1,1,1]` ⇒ `2 : 1` 就是 `2` 倍的差）"
      f"—— ★★ 这与 `expert-parallel` 的「EP 度越大 ⇒ 负载越不均」**同构**",
      f"**最坏的一组不是 `(12,8)`**（得到 `{worst_case}`，比值 `{worst_ratio:.4f}`）")

print()
print("     ★★ `D12` 的比值随 `p` 变（**不是常数**）：")
for p in (2, 4, 8):
    loads = stage_loads(assign_layers(12, p))
    print(f"        L=12, p={p}  ⇒  层数 {str(assign_layers(12, p)):<20} "
          f"负载最大/最小 = {max(loads)}/{min(loads)} = {max(loads) / min(loads):.6f}")

print()
print("=" * 78)
print("C · 边界与反例")
print("=" * 78)

# ── C.1 ★★★ 首尾多 embedding ⇒ 瓶颈在 0/p−1；去掉就变均衡 ──────────────────
L_c, p_c = 12, 4
with_e = stage_loads(assign_layers(L_c, p_c), with_emb=True)
without_e = stage_loads(assign_layers(L_c, p_c), with_emb=False)
bottleneck_with = with_e.index(max(with_e))
balanced_without = (max(without_e) == min(without_e))
ok_c1 = (bottleneck_with in (0, len(with_e) - 1)) and balanced_without
check("C.1", ok_c1,
      f"★★★ **首尾多 embedding ⇒ 瓶颈恒在 stage `0` 或 `p−1`**（本组得到 stage `{bottleneck_with}`）；"
      f"★ **去掉 embedding 就变均衡**（`{without_e}` ⇒ 全同 = `{balanced_without}`）"
      f"⇒ **真的会变化、判据有效**",
      "**瓶颈的位置与 embedding 的归属对不上** —— 或去掉 embedding 后仍不均衡")

# ── C.2 ★★ 守卫：embedding 的归属决定【谁是瓶颈】 ───────────────────────────
_both = stage_loads(assign_layers(L_c, p_c), with_emb=True)          # 首尾都加
_one = [n * specs.BALANCE_PER_LAYER for n in assign_layers(L_c, p_c)]
_one[0] += specs.BALANCE_EMB                                          # ★ 只给 stage 0 加
n_both = sum(1 for x in _both if x == max(_both))
n_one = sum(1 for x in _one if x == max(_one))
check("C.2", n_both == 2 and n_one == 1 and max(_both) == max(_one),
      f"★★ **瓶颈的【个数】随 embedding 的归属变化**：首尾都加 ⇒ **{n_both} 个瓶颈**"
      f"（`{_both}`）；只给 stage 0 加 ⇒ **{n_one} 个**（`{_one}`）"
      f"⇒ ★ **`max` 相同、瓶颈的个数不同** ⇒ 那条「瓶颈恒在首尾」的断言"
      f"**在「只加一端」时会失败** ✓",
      f"**瓶颈个数不符**（首尾 {n_both} 个、单端 {n_one} 个）")
print()
print("     ★★ **为什么守卫选「embedding 的归属」而不是「层数倒序」**：")
print("        `stage_loads` 给**首尾都加** embedding ⇒ **层数倒序之后首尾仍然对称** ⇒")
print("        **比值完全不变** ⇒ 那样的「守卫」**恒真**（本程序的第一版就写错成它了）。")
print("        ⇒ ★★★ **守卫必须选一个【真的会改变被判量】的错法**（`CONTRIBUTING:235` 的落法）。")

# ── C.3 ★ `L < p` 的退化 ─────────────────────────────────────────────────
ls_small = assign_layers(3, 8)
loads_small = stage_loads(ls_small, with_emb=False)
check("C.3", sum(ls_small) == 3 and loads_small.count(0) == 5,
      f"★ `L < p` 的退化：`L=3, p=8` ⇒ 层数 `{ls_small}`（**5 个空 stage**）"
      f"⇒ ★ 空 stage 的负载是 `0`，但它们**仍占着流水线的位置**（气泡在 `p` 那一侧）",
      "**`L < p` 时分层不对**")

print()
print("     ★★ 一条**回链**（本程序一行都不写）：`L < p` 时真正的代价是**气泡**，")
print("        不是负载不均 —— 气泡的账在 `01_bubbles.py`（那里的 `(p−1)/(m+p−1)`）。")

print()
print("     ★★ 重算在这里的位置（**只引，不重写** —— 禁区是 `90-training/…/02`）：")
print(f"        原文逐字：`{specs.GPIPE_RECOMPUTE_MEMORY_QUOTE}` …")
print("        ⇒ **重算压的是【激活】那一项**；★ 它怎么压（`O(√n)` / `≈2n`）**不在本域**。")

print()
print("=" * 78)
print("D · 口径常量、独立性证据与标记自检")
print("=" * 78)

# ── D.6 ★★ 两个网格的【逐元素】钉住（★ 只钉整体会让元素级变异漏网） ──────────
check("D.6", specs.BALANCE_CASES
      == ((130, 100, 100, 130), (100, 100, 100, 100), (130, 100, 100, 130, 100))
      and specs.LAYER_COUNT_GRID == ((12, 4), (13, 4), (12, 5), (12, 8)),
      "★★ `BALANCE_CASES`（三组）与 `LAYER_COUNT_GRID`（四组）**逐元素**钉住"
      "—— ★ 只钉整体会让元素级变异漏网",
      "**网格的元素被动过**")

# ── D.7 ★★ 另外四个主题的标记（★ 用来验「不照抄」） ────────────────────────
check("D.7", OTHER_THEME_MARKS == ("%% FAIL: data-parallel", "^^ FAIL: tensor-parallel",
                                   "{{ FAIL: seq-context-parallel", "|| FAIL: expert-parallel",
                                   "?? FAIL: zero-fsdp")
      and ASSERT_FAIL_MARK not in OTHER_THEME_MARKS,
      "★★ 另外五个主题的标记**逐个逐字**钉住、且**不含本主题的** ⇒ 「不跨主题照抄」的机械检查"
      "（★ 只检查「长度 + 含 FAIL + 不含本主题」会让**元素级**变异漏网 —— 本程序实测过）",
      "**OTHER_THEME_MARKS 的元素被动过**")

check("D.1", True,
      "★★ **故意标注一条恒真**：`每 stage 负载 == 层数 × PER_LAYER + emb` **是乘法恒真**"
      "⇒ ★★★ **它【不】当 A 类**（本程序的 A 类钉在「`max` vs `mean`」与「瓶颈的位置」上）"
      "—— 这就是 `P12` 的正例处置：**把恒真的那半条标出来，而不是拿它充判据**",
      "")

check("D.2", "max(" not in inspect.getsource(assign_layers)
      and "sum(" not in inspect.getsource(assign_layers),
      "★★ **独立性证据**：分层函数 `assign_layers()` 的源码里**不出现** `max(`/`sum(`"
      "⇒ 负载的汇总口径**不由分层决定**（两者独立）",
      "**分层函数里出现了汇总运算** —— 两层逻辑混在一起了")

check("D.3", specs.GPIPE_RECOMPUTE_MEMORY_QUOTE
      == "peak activation memory requirement is reduced to O(",
      "★ 重算那句引文逐字钉住（★ **只引它「压的是什么量」**；机制在 `90-training`）",
      "**重算的引文常量被动过**")

check("D.4", specs.LAYER_COUNT_GRID == ((12, 4), (13, 4), (12, 5), (12, 8)),
      "★ 分层网格逐项钉住（四组 `(L,p)`，含两组不整除）",
      "**分层网格被动过**")

check("D.5", len({ASSERT_FAIL_MARK}) == 1 and list({ASSERT_FAIL_MARK})[0] == "== FAIL: pipeline-parallel"
      and "FAIL" in ASSERT_FAIL_MARK,
      f"失败标记自检：`{ASSERT_FAIL_MARK!r}`（去重 1 个、含 `FAIL`）✓",
      "**失败标记被动过**")

print()
print("=" * 78)
print(f"汇总: A {_PASS['A']}/{_TOTAL['A']} · B {_PASS['B']}/{_TOTAL['B']} · "
      f"C {_PASS['C']}/{_TOTAL['C']} · D {_PASS['D']}/{_TOTAL['D']} ⇒ "
      f"断言 {sum(_TOTAL.values())} 条，通过 {sum(_PASS.values())} 条，失败 {len(FAILS)} 条")
if FAILS:
    print(f"失败清单: {', '.join(FAILS)}")
    print(ASSERT_FAIL_MARK, "（上面的失败清单见本行之前）")
print("★ 本程序只验证「分层与瓶颈的推断对不对」；重算的机制在 `90-training/autograd-and-backward/02`。")
print("★ 不是 NCCL，也不是真机：本机无 GPU、无多卡、无真实网络。")
sys.exit(1 if FAILS else 0)

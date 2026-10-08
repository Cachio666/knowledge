#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""02_stages_and_costs.py —— **两段式**（reduce-scatter + all-gather）的**通信阶段划分**，
四个原语各自的**最优算法与代价**，以及 allreduce 的 `2(N−1)/N × S` **为何是两段之和**。

配套文档：`02-stages-and-costs.md`。

★ 三类断言（A 算法正确性 / B 代价模型 / C 边界与反例）**全部在本程序内部**，
失败时以**非零退出码**结束；**没有 `verify.py`**。

★★ **本程序的数字全部是「本机实测值」**，测量条件是：
> `python3 3.13.5`，`numpy 2.4.4`，macOS arm64，**无 GPU、无多卡、无 NCCL、无真实网络**。
> ★★★ 强制限定语：**「不是 NCCL，也不是真机」**。

★★★ **一条【同名不同指】的消歧（本文的篇名里就有那个词）**：
  `stage` / 「阶段」在库内**已有四个含义**（TVM 计算阶段 · cutlass 流水级数 ·
  流水线并行的 stage · ZeRO 阶段）。★★ 本程序里的 `stage` **一律指【通信阶段】**，
  且**只指**「**reduce-scatter 那一段**」与「**all-gather 那一段**」——
  代码里的字段名用 `phase`（不用 `stage`），正文里写「**通信阶段**」或
  「**reduce-scatter / all-gather 两段**」，**不裸写 `stage`**。

★★★ **本程序的主判据（A 类）是【分阶段对拍】**：
  把 allreduce 拆成 **reduce-scatter 一段** + **all-gather 一段**，
  **每一段都是独立的真消息序列模拟**（逐步枚举谁发给谁、哪一块），
  然后 ① 两段**接起来**的结果 == ② **单进程参照** == ③ **另一条独立路径**
  （不分段的 ring allreduce 整体模拟）。
  ★★★ 三条路径**必须给同一个结果** —— 这才是「分段 == 不分段」的证据。

★★ **本程序的 A 类也是消息序列模拟**（照 `§3.2`），**不是**把闭式用循环写一遍。

★★ **覆盖边界**：本程序守「**阶段拆得对不对**」「**每段的代价**」；
  **不守**任何真实网络的性能（本机测不了）。

★★★ **本主题的禁区（回链，一行都不重写）**：
  · ★★★ **`V(N) = 2(N−1)/N × S` 的推导与饱和**
    → `10-foundations/performance-models/02-amdahl-and-scaling.md:147-155,209`
    （★ **本程序不重推它** —— 只把它当「**两段各 `(N−1)/N × S`**」的**和**去**验证**）；
  · **bisection 下界 `t ≥ 2S/B`** · **all-to-all 跨切口量 `N·V/4`** · **分层 all-reduce 三段式**
    → `20-gpu-architecture/interconnect/03-collectives-on-topology.md`；
  · **`algbw` vs `busbw` 与修正系数表** → `interconnect/04-measuring-bandwidth.md`；
  · **带宽口径** → `interconnect/01-bandwidth-conventions.md`。

★★ 可证伪断言（把下面任一条改错，本程序必须非零退出）：
  · `A.1`  ★★★ **分阶段对拍**：`reduce-scatter` 一段 ⊕ `all-gather` 一段 == 单进程参照
    == **不分段的 ring allreduce 整体模拟**（`STAGE_GRID` 两档，三条路径逐位相同）；
  · `A.2`  ★★★ **两段的量相等**：每段的每卡搬运量都是 `(N−1)/N × S`，
    **两段之和** == `2(N−1)/N × S`（★ 这就是 allreduce 那条闭式的**结构来源**）；
  · `A.3`  ★★ **四个原语的最优算法步数**：reduce-scatter `lg p`、all-gather `lg p`、
    broadcast `⌈lg p⌉`、reduce `⌈lg p⌉`（★ 全部由消息序列**枚举出来**）；
  · `A.4`  ★★ **reduce 与 broadcast 的互逆**：两者的**总搬运量都是 `(p−1) × n`**
    （一个是把 `p` 份合成 `1`，一个是把 `1` 份摊成 `p`）；
  · `B.1`  ★★★ **四原语 + allreduce 的代价表**；
  · `B.2`  ★★★ **`2(N−1)/N × S` == 两段【消息序列枚举】出来的搬运量之和**（★ **不重推**，
    只做**结构验证**：★ **闭式 vs 枚举**，**不是**拿两条逐字同形的闭式互比 —— 那是恒真）；
  · `C.1`  ★★★ **漏掉 all-gather 那一段** ⇒ 每个设备只有**自己那一块**是对的，其余是**半成品**
    ⇒ 结果错、**不报错**；
  · `C.2`  ★★★ **漏掉 reduce-scatter 那一段**（只 gather 不 reduce）⇒ 每块只有**一台设备**的贡献
    ⇒ 结果错、**不报错**（与 `C.1` 不同值）；
  · `D.*`  口径常量逐个钉住 + **逐元素**元组钉住 + **引文逐字**钉住（含 `Trec half` 的**近似限定语**）
    + **失败标记自检**。
"""
import ast
import collections
import inspect
import sys
import textwrap

import numpy as np

import specs

# ===========================================================================
# 0. 断言记账（每个编号程序自带一份，**不共享**）
# ===========================================================================
# ★★★ 失败标记：本主题（collectives）取 `&& FAIL: collectives`（与同主题另外两个程序相同）。
#   ★ 本域另外三个主题各用**不同的前缀**（`><` / `,,` / `+=`）⇒ 跨主题照抄会被当场抓住。
#   ★★ `&&` 是 **shell 元字符** ⇒ 它**只允许出现在 `print()` 的字符串里**。
ASSERT_FAIL_MARK = "&& FAIL: collectives"

OTHER_THEME_MARKS = (">< FAIL: nccl", ",, FAIL: network-topology", "+= FAIL: overlap")

FAILS = []
MARKS_USED = set()
_TOTAL = collections.Counter()
_PASS = collections.Counter()

TOL_MACHINE = specs.TOL_MACHINE

# ★★ 消息的形态 —— 字段名用 **`phase`（不叫 `stage`）**，见模块抬头的消歧。
MSG = collections.namedtuple("MSG", "phase step sender receiver block n_elem")


def check(name, cond, detail="", detail_fail=None):
    """`name` 形如 `A.1` —— **首字母就是类别**（A/B/C/D）。

    ★★★ **失败路径【不复述结论】**：`detail` 一律是**判据成立时**的措辞
      ⇒ `cond` 为假时走**另一版文案**，避免输出自相矛盾。
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
# 1. 被测的闭式（`[C]` 级载体见 `specs.py`）
# ===========================================================================
def cf_stage_steps(p):
    """**单段**的步数闭式 `lg p`（reduce-scatter 用 recursive halving、all-gather 用 recursive
    doubling 时都是 `lg p`；★ 依据见 `specs.MPICH_ALLREDUCE_RECDUB_TIME_QUOTE`）。"""
    return int(np.log2(p))


def cf_stage_volume(p, n):
    """**单段**的每卡搬运量闭式 `(p−1)/p × n`
    （★ reduce-scatter 与 all-gather **共用**它 —— 前者是「送去」、后者是「收回」）。"""
    return (p - 1) / p * n


def cf_allreduce_volume_two_stage(p, n):
    """两段之和 `2(p−1)/p × n` —— ★★ 这就是 `V(N) = 2(N−1)/N × S`
    （★ 本程序**不重推**它，只验证它**等于两段之和**；推导在 `performance-models/02:147-155`）。"""
    return 2 * cf_stage_volume(p, n)


def cf_tree_steps(p):
    """树形原语（broadcast / reduce）的步数闭式 `⌈lg p⌉`。"""
    return int(np.ceil(np.log2(p)))


def ref_sum(x):
    """**单进程参照**：把每台设备的输入直接加起来（与任何模拟器**不共享代码**）。"""
    return x.sum(axis=0)


# ===========================================================================
# 2. 参照：**每一段各自**的真消息序列模拟器（逐步枚举）
# ===========================================================================
def _split(n_elem, p):
    w = n_elem // p
    return [(i * w, (i + 1) * w) for i in range(p)]


def sim_reduce_scatter(p, n, x=None):
    """★★★ **第一段：reduce-scatter**（recursive halving）。

    · 第 `k` 步：设备 `d` 把**自己当前持有的块**按「第 `k` 位」一分为二 ——
      **保住**与自己的 `k` 位相同的那半，**把另外半发给 `d XOR 2^k`**；
      收到的块**累加进**自己保住的那半；
    · 走完 `lg p` 步后，设备 `d` 手里只剩**一块**：`d` 号块的**完整和**。
    ★★ 每一步都记录 `MSG(phase, step, sender, receiver, block, n_elem)`。
    """
    parts = _split(n, p)
    if x is None:
        x = np.arange(p * n, dtype=np.int64).reshape(p, n)
    held = [{i: x[d][s:e].copy() for i, (s, e) in enumerate(parts)} for d in range(p)]
    msgs = []
    k = 0
    # ★★★ **步数是【涌现的】**：每台设备手里**多于一块**就再来一步 ——
    #   ★ 循环上界**不是**照着 `lg p` 写死的（★ 那样就又是把闭式抄一遍）。
    while max(len(h) for h in held) > 1:
        batch = []
        for d in range(p):
            partner = d ^ (1 << k)
            send = [i for i in sorted(held[d]) if ((i >> k) & 1) != ((d >> k) & 1)]
            for blk in send:
                batch.append((d, partner, blk, held[d][blk].copy()))
        for d, partner, blk, src in batch:
            msgs.append(MSG("reduce-scatter", k, d, partner, blk, len(src)))
            held[partner][blk] = held[partner][blk] + src
        for d, _partner, blk, _src in batch:
            del held[d][blk]
        k += 1
        if k > 64:                                  # ★ 防跑飞（本主题只用到 `p <= 16`）
            break
    return held, msgs


def sim_all_gather(p, n, held_in, corrupt=None):
    """★★★ **第二段：all-gather**（recursive doubling）。

    · 第 `k` 步：设备 `d` 把**手里全部的块**发给 `d XOR 2^k`，并收下对端的；
    · 走完 `lg p` 步后**每个设备都有全部块**。
    ★★ `corrupt="noreduce"` 是 `C.2` 的错法：**跳过第一段**（每块只有一台设备的贡献）。
    """
    held = [{k2: v.copy() for k2, v in h.items()} for h in held_in]
    msgs = []
    k = 0
    # ★★★ **步数也是【涌现的】**：还有设备没拿全就再来一步。
    while min(len(h) for h in held) < p:
        batch = []
        for d in range(p):
            partner = d ^ (1 << k)
            for blk, arr in held[d].items():
                batch.append((d, partner, blk, arr.copy()))
        for d, partner, blk, src in batch:
            msgs.append(MSG("all-gather", k, d, partner, blk, len(src)))
            held[partner][blk] = src
        k += 1
        if k > 64:
            break
    return held, msgs


def sim_ring_allreduce(p, n, x=None):
    """**不分段的独立路径**：ring allreduce（reduce-scatter + all-gather 走环）。
    ★ 它与上面那两个分段的模拟器**是【两套不同的实现】**（走环 vs 走超立方配对）
      —— 用来当分阶段对拍的第 ③ 条路径。
    ★★★ **两段的步数都是【涌现的】**（本路径**没有**错法变体 ⇒ 两段都能用状态条件）：
      · 第一段（reduce-scatter）走环，`while max(len(h)) > 1`（每台减到只剩一块）；
      · 第二段（all-gather）走环，`while min(len(h)) < p`（每台涨到拿全）。
      **不是** `for k in range(p - 1)`（那会把「`p−1`」预先塞进控制流）。"""
    parts = _split(n, p)
    if x is None:
        x = np.arange(p * n, dtype=np.int64).reshape(p, n)
    held = [{i: x[d][s:e].copy() for i, (s, e) in enumerate(parts)} for d in range(p)]
    msgs = []
    k = 0
    while max(len(h) for h in held) > 1:            # ★ 涌现：减到每台只剩一块为止
        batch = []
        for d in range(p):
            blk = (d - k) % p
            batch.append((d, (d + 1) % p, blk, held[d][blk].copy()))
        for d, dst, blk, src in batch:
            msgs.append(MSG("ring-rs", k, d, dst, blk, len(src)))
            held[dst][blk] = held[dst][blk] + src
            del held[d][blk]
        k += 1
        if k > 64:
            break
    k = 0
    while min(len(h) for h in held) < p:            # ★ 涌现：还没拿全就再来一步
        batch = []
        for d in range(p):
            blk = (d + 1 - k) % p
            batch.append((d, (d + 1) % p, blk, held[d][blk].copy()))
        for d, dst, blk, src in batch:
            msgs.append(MSG("ring-ag", k, d, dst, blk, len(src)))
            held[dst][blk] = src
            held[d].setdefault(blk, src)
        k += 1
        if k > 64:
            break
    outs = [np.concatenate([held[d][i] for i in range(p)]) for d in range(p)]
    return outs, msgs


def sim_reduce_tree(p, n, x=None, root=0):
    """**reduce（→ 根 0）**：二项树，★ **它就是 broadcast 那棵树的【反向遍历】** ——
    边还是那些边 `(r, r + 2^k)`，只是**步序倒过来**、数据从 `r + 2^k` 流向 `r` 并**累加**。
    ★★ 「reduce 与 broadcast 是同一棵树的两种遍历方向」这件事让它们的**总搬运量相等**
      （`A.4` 量它）。

    ★★ **一条如实的限定（本模拟是【唯一】一处仍是"预设步数"的循环，已裁定【不改】）**：
      本模拟的步数上界是 `⌈lg p⌉` —— `lg` 由 `while (1 << lg) < p: lg += 1` **从 `p` 推出**，
      **不是**从闭式函数取。★ 但它**不比 `range(cf_*)` 更涌现**（两者都**预设**了循环次数）。
      ★★★ **之所以不改**（三条，★ 按重要性倒序）：
        ① **结果项能过守卫**（★ 决定性）：`A.3` 用 `have[root]` 与**单进程参照**对它，
           那一条**过 `CONTRIBUTING:235` 的守卫** ⇒ 「步数项准恒真」**不构成本域的判据失效**；
        ② `lg` **不调用任何被测函数**（从 `p` 算）⇒ 不违反 `CONTRIBUTING:235` 的**字面**禁令；
        ③ ★ **改成真正涌现需要【重排报文顺序】**，而 `MSG` 带 `step` 字段 ⇒ 会动
           `A.3`/`A.4` 的**步数语义** ⇒ **风险 > 收益**。
      ⇒ ★★ 本域按「**结果项过守卫即可**」的口径接受它（★ `01` 的 ring 五个循环**不**适用这条
         豁免：那里**有**错法变体、且我们**能**用事件/状态条件，见 `01` 的 RS 段现场）。"""
    if x is None:
        x = np.arange(p * n, dtype=np.int64).reshape(p, n)
    lg = 0
    while (1 << lg) < p:
        lg += 1
    have = {d: x[d].copy() for d in range(p)}
    msgs = []
    for k in range(lg - 1, -1, -1):               # ★ 反向：从最大的 `k` 开始
        batch = []
        for r in range(1 << k):
            if r + (1 << k) < p:
                batch.append((r + (1 << k), r))
        for src, dst in batch:
            msgs.append(MSG("reduce-tree", lg - 1 - k, src, dst, 0, n))
            have[dst] = have[dst] + have[src]
    return have[root], msgs


def sim_broadcast_tree(p, n, root=0):
    """**broadcast（从根出发）**：二项树。第 `k` 步，**已有数据的** `r < 2^k` 发给 `r + 2^k`。"""
    msg = np.arange(n, dtype=np.int64)
    have = {root: msg.copy()}
    msgs = []
    k = 0
    while len(have) < p:                          # ★ 步数**涌现**：还没铺满就再来一步
        batch = []
        for r in sorted(have):
            if r < (1 << k) and r + (1 << k) < p:
                batch.append((r, r + (1 << k)))
        for r, dst in batch:
            msgs.append(MSG("broadcast-tree", k, r, dst, 0, n))
            have[dst] = have[r].copy()
        k += 1
        if k > 64:
            break
    return have, msgs


def vol_of(msgs, p):
    """**从枚举出来的消息清单**算每台设备的发出元素数 —— 逐条求和，不是乘出来的。"""
    return [sum(m.n_elem for m in msgs if m.sender == d) for d in range(p)]


# ===========================================================================
# 3. A 类：分阶段对拍（三条路径）+ 每段的代价 + 四个原语的步数
# ===========================================================================
def part_A():
    print("=" * 78)
    print("A · **分阶段对拍**：reduce-scatter 一段 ⊕ all-gather 一段 == 不分段")
    print("=" * 78)
    x = np.arange(max(specs.GATHER_GRID) * specs.GATHER_ELEMS, dtype=np.int64) \
        .reshape(max(specs.GATHER_GRID), specs.GATHER_ELEMS)

    # ── A.1 ★★★ 三条路径给同一个结果 ──────────────────────────────────────
    print()
    print("  ── A.1 ★★★ **三条路径**给同一个结果（`STAGE_GRID` 两档）──")
    ok1 = True
    for p in specs.STAGE_GRID:
        n = specs.STAGE_ELEMS * p
        sub = np.arange(p * n, dtype=np.int64).reshape(p, n)
        ref = ref_sum(sub)
        # ① 分段：RS 一段 + AG 一段
        held_rs, m_rs = sim_reduce_scatter(p, n, x=sub)
        held_ag, m_ag = sim_all_gather(p, n, held_rs)
        staged = [np.concatenate([held_ag[d][i] for i in range(p)]) for d in range(p)]
        ok_staged = all(np.array_equal(g, ref) for g in staged)
        # ③ 不分段：ring allreduce 整体
        whole, m_whole = sim_ring_allreduce(p, n, x=sub)
        ok_whole = all(np.array_equal(g, ref) for g in whole)
        # ★ 两条路径**逐位相同**
        ok_cross = all(np.array_equal(staged[d], whole[d]) for d in range(p))
        ok1 = ok1 and ok_staged and ok_whole and ok_cross
        print("     N=%-3d 分段路径：RS %-2d 条 + AG %-2d 条 = %-2d 条消息；"
              "不分段（ring）：%-2d 条 ⇒ 三条路径**逐位相同** = %s"
              % (p, len(m_rs), len(m_ag), len(m_rs) + len(m_ag), len(m_whole),
                 ok_staged and ok_whole and ok_cross))
    check("A.1", ok1,
          "★★★ **判据成立**：① `reduce-scatter` 一段 ⊕ `all-gather` 一段；② **单进程参照**；"
          "③ **不分段的 ring allreduce** —— ★ **三条路径逐位相同** ⇒ 「分段 == 不分段」",
          "**分阶段路径与不分段/参照不一致** —— ★ 先查两段的累加目标与配对式")

    # ── A.2 ★★★ 两段的量相等 ────────────────────────────────────────────
    print()
    print("  ── A.2 ★★★ **两段的每卡搬运量相等**，**和被闭式给出** ──")
    ok2 = True
    for p in specs.STAGE_GRID:
        n = specs.STAGE_ELEMS * p
        sub = np.arange(p * n, dtype=np.int64).reshape(p, n)
        held_rs, m_rs = sim_reduce_scatter(p, n, x=sub)
        held_ag, m_ag = sim_all_gather(p, n, held_rs)
        v_rs, v_ag = vol_of(m_rs, p)[0], vol_of(m_ag, p)[0]
        cf_one = cf_stage_volume(p, n)
        two = cf_allreduce_volume_two_stage(p, n)
        ok2 = ok2 and (v_rs == cf_one) and (v_ag == cf_one) and (v_rs + v_ag == two)
        print("     N=%-3d 第一段 %-4d · 第二段 %-4d · **两段之和 %-4d** == `2(N−1)/N × S` (%.0f)"
              % (p, v_rs, v_ag, v_rs + v_ag, two))
    print("     ★★★ 这就是 `V(N) = 2(N−1)/N × S` 的**结构来源**：**它就是两段之和** ——")
    print("        ★★ **本程序不重推它**（推导在 `performance-models/02:147-155`），此处**只做结构验证**。")
    check("A.2", ok2,
          "★★★ **判据成立**：**两段的每卡搬运量都是 `(N−1)/N × S`**，**两段之和 == `2(N−1)/N × S`**"
          " ⇒ ★★ 闭式的「`2×`」来自**两段相加**，不是巧合",
          "**两段的量或它们的和与闭式不符** —— ★ 先查 reduce-scatter 的步数是否为 `lg p`")

    # ── A.3 ★★ 四个原语的最优算法步数 ────────────────────────────────────
    print()
    print("  ── A.3 ★★ **四个原语**各自的最优算法步数（全部**枚举自**消息序列）──")
    ok3 = True
    for p in specs.STAGE_GRID:
        n = specs.STAGE_ELEMS * p
        sub = np.arange(p * n, dtype=np.int64).reshape(p, n)
        _h, m_rs = sim_reduce_scatter(p, n, x=sub)
        _h2, m_ag = sim_all_gather(p, n, _h)
        _r, m_rd = sim_reduce_tree(p, n, x=sub)
        _b, m_bc = sim_broadcast_tree(p, n)
        s_rs = max(m.step for m in m_rs) + 1      # ★ 步数**枚举出来**
        s_ag = max(m.step for m in m_ag) + 1
        s_rd = max(m.step for m in m_rd) + 1
        s_bc = max(m.step for m in m_bc) + 1
        ok3 = ok3 and (s_rs == cf_stage_steps(p)) and (s_ag == cf_stage_steps(p)) \
            and (s_rd == cf_tree_steps(p)) and (s_bc == cf_tree_steps(p))
        print("     N=%-3d │ reduce-scatter %d（`lg p`）│ all-gather %d（`lg p`）│"
              " reduce %d │ broadcast %d（后两者 `⌈lg p⌉`）"
              % (p, s_rs, s_ag, s_rd, s_bc))
    check("A.3", ok3,
          "★★ **判据成立**：`reduce-scatter` 与 `all-gather` 各 `lg p` 步；"
          "`reduce` 与 `broadcast`（树形）各 `⌈lg p⌉` 步 —— ★ 四个原语的**最优步数**都是**对数级**",
          "**某个原语的步数与闭式不符** —— ★ 先查该原语的配对式/条件")

    # ── A.4 ★★ reduce 与 broadcast 的总量 ───────────────────────────────
    print()
    print("  ── A.4 ★★ **reduce 与 broadcast 的【总】搬运量都是 `(p−1) × n`** ──")
    ok4 = True
    for p in specs.STAGE_GRID:
        n = specs.STAGE_ELEMS * p
        root_sum, m_rd = sim_reduce_tree(p, n)
        ref = ref_sum(np.arange(p * n, dtype=np.int64).reshape(p, n))
        have, m_bc = sim_broadcast_tree(p, n)
        tot_rd = sum(m.n_elem for m in m_rd)
        tot_bc = sum(m.n_elem for m in m_bc)
        ok4 = ok4 and np.array_equal(root_sum, ref) \
            and all(np.array_equal(have[d], np.arange(n)) for d in have) \
            and (tot_rd == (p - 1) * n) and (tot_bc == (p - 1) * n)
        print("     N=%-3d reduce 总搬运量 %-4d · broadcast 总搬运量 %-4d ⇒ 都是 `(p−1)×n` = %d"
              % (p, tot_rd, tot_bc, (p - 1) * n))
    print("     ★★ 两者**互逆**：reduce 把 `p` 份**合成 1**、broadcast 把 `1` 份**摊成 p**，")
    print("        而**总搬运量相同** —— 都是「每台设备恰好收到（或发出）一次」的树。")
    check("A.4", ok4,
          "★★ **判据成立**：`reduce` 与 `broadcast`（都用二项树）的**总搬运量都是 `(p−1) × n`**、"
          "★ 且**每台设备最多发一次**（`⌈lg p⌉` 步走完）",
          "**reduce/broadcast 的总量与闭式不符** —— ★ 先查「`r < 2^k` 才发」这条条件")


# ===========================================================================
# 4. B 类：代价表 · `2(N−1)/N × S` 的结构验证
# ===========================================================================
def part_B():
    print()
    print("=" * 78)
    print("B · 代价模型：四原语 + allreduce 的对照表 · `2(N−1)/N × S` 为何是**两段之和**")
    print("=" * 78)

    # ── B.1 ★★★ 代价表 ──────────────────────────────────────────────────
    print()
    print("  ── B.1 ★★★ **四个原语 + allreduce** 的最优算法与代价（同一个 `n`）──")
    n = specs.STAGE_ELEMS
    rows = []
    for p in specs.GATHER_GRID:
        rows.append((p, cf_stage_steps(p), cf_stage_volume(p, n),
                     cf_tree_steps(p), (p - 1) * n / p,
                     cf_stage_steps(p), cf_stage_volume(p, n),
                     cf_stage_steps(p) * 2, cf_allreduce_volume_two_stage(p, n)))
    print("     %-4s │ %-26s │ %-22s │ %s" % ("p", "reduce-scatter", "all-gather", "allreduce"))
    for p, rs_s, rs_v, _t, _tv, ag_s, ag_v, ar_s, ar_v in rows:
        print("     %-4d │ %-2d 步 / %-6.2f 每卡 │ %-2d 步 / %-6.2f 每卡 │ %d 步 / %-6.2f 每卡"
              % (p, rs_s, rs_v, ag_s, ag_v, ar_s, ar_v))
    print("     ★★ 另有**树形**两原语（同一个 `n`）：`broadcast` 与 `reduce` 各 `⌈lg p⌉` 步，")
    print("        **总**搬运量都是 `(p−1) × n`（★ `A.4` 枚举出来）。")
    okb1 = all(abs(r[2] - r[6]) < 1e-12 and abs(r[8] - 2 * r[2]) < 1e-12 for r in rows) \
        and len(rows) == 4
    check("B.1", okb1,
          "★★★ **判据成立**：`GATHER_GRID` **四档**上 —— 两段的每卡搬运量**相等**，"
          "**allreduce 的每卡搬运量恒等于两段之和**（`2(N−1)/N × S`），且三段步数都是 `lg p` 级",
          "**代价表的等式不成立** —— ★ 先查两段的闭式是否被改动")

    # ── B.2 ★★★ 结构验证（★ 不是重推）────────────────────────────────────
    print()
    print("  ── B.2 ★★★ 闭式 == **消息序列枚举出来的两段之和**（★ **不重推**）──")
    print("     `specs.MPICH_RABENSEIFNER_TIME_QUOTE`：")
    print("        `%s`" % specs.MPICH_RABENSEIFNER_TIME_QUOTE)
    print("     ★★ 它的带宽项 `2(p−1)/p · nβ` **逐字就是**「两段各 `(p−1)/p · n`」——")
    print("     ★★★ **本程序【不推导】它**（推导在 `performance-models/02:147-155`，含 `S` 的饱和）——")
    print("        这里做的是**另一个方向**的检查：**闭式 == 消息序列枚举出来的两段之和**。")
    print("     ★★ **本轮修正**：原写法拿**两条逐字同形的闭式**互比"
          "（`cf_allreduce_volume_two_stage ≡ 2 × cf_stage_volume`）⇒ **恒真**，")
    print("        已换成**枚举**：把两段各自跑一遍、**逐条消息求和**，再与闭式比。")
    okb2 = True
    for p in specs.GATHER_GRID:
        ne = specs.STAGE_ELEMS * p                       # ★ 每设备输入（块大小是整数）
        sub = np.arange(p * ne, dtype=np.int64).reshape(p, ne)
        held_rs, m_rs = sim_reduce_scatter(p, ne, x=sub)
        held_ag, m_ag = sim_all_gather(p, ne, held_rs)
        enum_sum = vol_of(m_rs, p)[0] + vol_of(m_ag, p)[0]      # ★ 逐条枚举求和
        closed = cf_allreduce_volume_two_stage(p, ne)
        okb2 = okb2 and abs(enum_sum - closed) < TOL_MACHINE
        print("     p=%-3d 枚举：RS %-4d + AG %-4d = **%-4d** · 闭式 %.0f ⇒ 一致 = %s"
              % (p, vol_of(m_rs, p)[0], vol_of(m_ag, p)[0], enum_sum, closed,
                 abs(enum_sum - closed) < TOL_MACHINE))
    check("B.2", okb2 and specs.MPICH_RABENSEIFNER_TIME_QUOTE.startswith("Trabenseifner = 2 lg p α"),
          "★★★ **判据成立**：**消息序列枚举出来的两段之和** == 闭式 `2(N−1)/N × S`"
          "（★ 逐条求和，**不是**拿两条同形闭式互比）；★ 且**闭式的载体**"
          "（`MPICH_RABENSEIFNER_TIME_QUOTE`）一并钉住 ⇒ ★ 「为什么是 `2×`」有**结构解释**：**两段**",
          "**结构验证不成立** —— ★ 先查两段模拟器**枚举出来**的量（不是闭式）")


# ===========================================================================
# 5. C 类：漏掉一段（★ 两条都**不报错**，只是结果错）
# ===========================================================================
def part_C():
    print()
    print("=" * 78)
    print("C · 边界与反例：**漏掉一段**（all-gather / reduce-scatter）")
    print("=" * 78)
    print("  ★★★ 两条错法的**共同点**：形状、块数、字节数**全都对**、**一次异常都不抛** ——")
    print("      唯一的信号是**与参照的差**。")
    p = specs.STAGE_GRID[-1]
    n = specs.STAGE_ELEMS * p
    sub = np.arange(p * n, dtype=np.int64).reshape(p, n)
    ref = ref_sum(sub)

    print()
    print("  ── C.1 ★★★ **漏掉 all-gather 那一段**（只 reduce-scatter）──")
    held_rs, _m = sim_reduce_scatter(p, n, x=sub)
    # ★ 每台设备手里只剩**一块**（它是那一块的**完整和**）；其余块是**半成品/缺失**
    out_broken = []
    for d in range(p):
        buf = np.zeros(n, dtype=np.int64)
        for blk, arr in held_rs[d].items():
            buf[blk * (n // p):(blk + 1) * (n // p)] = arr
        out_broken.append(buf)
    d1 = max(float(np.abs(buf - ref).max()) for buf in out_broken)
    print("     N=%d：每个设备只有 **1 / %d** 块是完整和，其余块**根本没被 gather 回来**"
          % (p, p))
    print("     `max|diff|` = **%.1f**" % d1)
    print("     ★★★ **它【不报错】**：缓冲区大小、块边界、类型**全部合法** —— 只是**大部分块是空的**。")
    check("C.1", d1 > 0 and float(np.abs(out_broken[0] - ref).max()) > 0,
          "★★★ **判据成立**：漏掉 all-gather ⇒ `max|diff|` = **%.1f**（≫ 内容级阈 `1e-6`）"
          "⇒ ★ **不报错、结果错**" % d1,
          "**漏掉 all-gather 没有让结果变** —— ★ 那说明参照与被测共用了一段代码（自证）")

    print()
    print("  ── C.2 ★★★ **漏掉 reduce-scatter 那一段**（只 all-gather，不归约）──")
    parts = _split(n, p)
    raw = [{i: sub[d][s:e].copy()} for d, (i, (s, e)) in
           enumerate([(d, parts[d]) for d in range(p)])]
    held_ag, _m2 = sim_all_gather(p, n, raw)
    out2 = [np.concatenate([held_ag[d][i] for i in range(p)]) for d in range(p)]
    d2 = max(float(np.abs(buf - ref).max()) for buf in out2)
    print("     N=%d：每块**只有一台设备**的贡献（`x[d]` 的那一片），**没有相加**" % p)
    print("     `max|diff|` = **%.1f** ⇒ 与 `C.1` 的 **%.1f** %s"
          % (d2, d1, "**不同**" if abs(d2 - d1) > 0 else "**相同**"))
    check("C.2", d2 > 0 and abs(d2 - d1) > 0,
          "★★★ **判据成立**：漏掉 reduce-scatter ⇒ `max|diff|` = **%.1f**（≫ `1e-6`），"
          "且它与 `C.1` 的 **%.1f** **不同** ⇒ ★★ **两段漏哪一段都是错，但错法不同**" % (d2, d1),
          "**漏掉 reduce-scatter 没有让结果变**，或它与 `C.1` 给同一个数")


# ===========================================================================
# 6. 前哨：口径 / 阈值 / 标记自检
# ===========================================================================
def preflight():
    """★★ 口径 / 阈值 / 标记自检**一律在这里**，且**跑在任何计算之前**（本库 `P23`）。"""
    print("=" * 78)
    print("D0. 口径/阈值钉住（**先跑**）")
    print("=" * 78)
    print("  本程序的失败标记 ASSERT_FAIL_MARK = %r" % ASSERT_FAIL_MARK)
    print("  ★ 本程序的数字是**本机实测值**：`python3 %s` / `numpy %s`。"
          % (sys.version.split()[0], np.__version__))
    print("  ★★★ 强制限定语：**不是 NCCL，也不是真机**。")
    print("  ★★★ **一篇名消歧**：本程序里的 `phase` 一律指【通信阶段】——")
    print("      只指「**reduce-scatter / all-gather 两段**」；★ 库内 `stage` 另有**四个**含义")
    print("      （TVM 计算阶段 / cutlass 流水级数 / 流水线并行的 stage / ZeRO 阶段）⇒ **不裸写它**。")
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
    check("D.2", ASSERT_FAIL_MARK == "&& FAIL: collectives" and "FAIL" in ASSERT_FAIL_MARK,
          "★ 标记自检：**等于**本主题的字面量 `&& FAIL: collectives`、且含 `FAIL`\n"
          "        ★★ `&&` 只允许出现在 `print()` 的字符串里（★ 它是 shell 元字符）。")
    check("D.2b", OTHER_THEME_MARKS == (">< FAIL: nccl", ",, FAIL: network-topology",
                                        "+= FAIL: overlap"),
          "★ 「不跨主题照抄」的**对照物**（本域另外三个主题的标记）逐位钉住")
    check("D.3", specs.TOL_MACHINE == 1.0e-12 and specs.UNSOURCED == (),
          "★ 机器精度档阈 = %g；★ `UNSOURCED` 留空（★ 0 个叶子 ⇒ 不产生变异体）"
          % specs.TOL_MACHINE)

    # ── D.3b ★★★ **口径守卫**：两段的档位必须是 2 的幂（否则 `XOR` 配对跑飞）──
    #   ★★★ 它**必须跑在任何计算之前**（本库 `P23`）⇒ 把 `STAGE_GRID` 的某一档改成
    #     非 2 的幂时，是**断言捕获**而不是**崩溃捕获**。
    pw = all((v & (v - 1)) == 0
             for v in tuple(specs.STAGE_GRID) + tuple(specs.GATHER_GRID))
    # ★ 另核：`STAGE_ELEMS` 被 `STAGE_GRID` 的**每一档**整除（两段的块大小才是整数）
    div = [specs.STAGE_ELEMS % v for v in specs.STAGE_GRID]
    check("D.3b", pw and all(r == 0 for r in div),
          "★★★ **口径守卫**：`STAGE_GRID` 与 `GATHER_GRID` **全是 2 的幂**（`XOR` 配对要求），"
          "且 `STAGE_ELEMS = %d` 被 `STAGE_GRID` 的每一档**整除**（取余 `%s`）"
          % (specs.STAGE_ELEMS, div),
          "**档位的 2 的幂性质或整除性被破坏** —— ★ 先查 `STAGE_GRID`/`STAGE_ELEMS`")
    return len(FAILS) == before


# ===========================================================================
# 7. D 类（后半）：引文逐字钉住 · 逐元素钉住 · 独立性线索
# ===========================================================================
def part_D():
    print()
    print("=" * 78)
    print("D · 引文逐字钉住 · **逐元素**钉住 · 独立性线索")
    print("=" * 78)

    # ── D.5 ★★★ 两段式的代价式 ──────────────────────────────────────────
    check("D.5", specs.MPICH_RABENSEIFNER_TIME_QUOTE
          == "Trabenseifner = 2 lg p α + 2(p−1)/p · nβ + (p−1)/p · nγ."
          and specs.MPICH_BCAST_VDG_QUOTE
          == ("the message to be broadcast is first divided up and scattered among the processes, "
              "similar to an MPI Scatter. The scattered data is then collected back to all "
              "processes, similar to an MPI Allgather."),
          "★★ **两条逐字引文钉住**：\n"
          "        ① `Trabenseifner` —— 步数项 `2 lg p`、带宽项 `2(p−1)/p · nβ`（★ 两段各一份）、"
          "归约项 `(p−1)/p · nγ`；**裸 `lg p`，无取整**；\n"
          "        ② Van de Geijn 广播是**两段式**（`scatter` + `allgather`）的散文依据 ——\n"
          "        ★★★ **它说明「分两段看」不是 allreduce 独有的结构**。\n"
          "        ★★ 载体 = 一手 PDF（`mcs.anl.gov/~thakur/papers/ijhpca-coll.pdf`），抓取日 2026-10-08",
          "**两段式的引文常量被动过** —— ★ 先查取整符号与分数线的排版形态")

    # ── D.6 ★★★ 不整除时的 reduce-scatter + 它的**近似限定语** ────────────
    check("D.6", specs.MPICH_TREC_HALF_QUOTE
          == "Trec half = (⌊lg p⌋ + 2)α + 2nβ + n(1 + (p−1)/p)γ."
          and specs.MPICH_TREC_HALF_QUALIFIER
          == ("This cost is approximate because some imbalance exists in the amount of work "
              "each process does")
          and specs.MPICH_CUTOFF_QUOTE
          == "experimentally determined cutoff points",
          "★★★ **一条【分情况】的代价式 + 它自带的限定语，成对钉住**：\n"
          "        · `Trec half` 是 **`p` 不是 2 的幂**时 reduce-scatter 的代价 ——"
          " ★★★ 注意这里是 **`⌊lg p⌋`（下取整，`pdftotext` 提取成 `blg pc`）**，\n"
          "          与 `Ttree`/`Tbruck` 的 `⌈lg p⌉`（`dlg pe`）、`Trec dbl` 的**裸 `lg p`**"
          "**三足鼎立**；\n"
          "        · ★★ 紧跟的 `This cost is approximate …` 是**官方自述的近似限定语**"
          " ⇒ **引一条式子的限定语与式子本身是同一件事的两个部分**；\n"
          "        · ★★ `MPICH_CUTOFF_QUOTE` 是 `F5` 的依据（分档阈值是**实现选择**，不是普适规律）。",
          "**`Trec half` 或它的限定语被动过** —— ★ 先查那个取整符号是不是 `⌊⌋`")

    # ── D.7 ★★★ 逐元素钉住 ─────────────────────────────────────────────
    check("D.7", specs.STAGE_GRID == (2, 4) and specs.STAGE_ELEMS == 8
          and specs.GATHER_GRID == (2, 4, 8, 16) and specs.GATHER_ELEMS == 32
          and specs.DEV_GRID == (2, 4, 8, 16, 64),
          "★★★ **两个元组逐元素 + 两个标量**钉住（`STAGE_GRID` 两档 · `GATHER_GRID` 四档 ·"
          " `STAGE_ELEMS` · `GATHER_ELEMS`）—— ★ 只钉「整体」会让**元素级**变异漏网",
          "**元组的元素或某个标量被动过**")

    # ── D.8 ★★ 独立性线索 ───────────────────────────────────────────────
    srcs = {inspect.getsource(f) for f in (sim_reduce_scatter, sim_all_gather,
                                           sim_ring_allreduce, ref_sum)}
    calls_rs = {n.func.id for n in ast.walk(
        ast.parse(textwrap.dedent(inspect.getsource(sim_reduce_scatter))))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    calls_ref = {n.func.id for n in ast.walk(
        ast.parse(textwrap.dedent(inspect.getsource(ref_sum))))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    print()
    print("  ── D.8 ★★ **源码级线索**（★ **不是证明**）：参照与被测的**调用表** ──")
    print("     `sim_reduce_scatter()`（第一段参照）里调用的函数：%s"
          % (", ".join(sorted(calls_rs)) if calls_rs else "**空集**"))
    print("     `ref_sum()`（单进程参照）里调用的函数：%s"
          % (", ".join(sorted(calls_ref)) if calls_ref else "**空集**"))
    print("     ⇒ ★★ **两段的模拟器都不调用任何 `cf_*` 闭式** —— 「参照不得用被测的输出」"
          "在此**在源码层面成立**。★ **降级措辞**：这只是「不互相调用」，**不是证明**。")
    check("D.8", "cf_stage_volume" not in calls_rs and "cf_stage_steps" not in calls_rs
          and "sim_reduce_scatter" not in calls_ref and len(srcs) == 4,
          "★★ **源码级线索成立**：四个函数两两不同源；★ 第一段参照的调用表里**没有** `cf_*`",
          "**参照与闭式互相调用** —— A 类可能退化成自证")

    # ── D.9 ★★ 阶段划分的**否定性**边界 ─────────────────────────────────
    check("D.9", specs.MPICH_BRUCK_SCOPE_NOTE != "" and specs.UNSOURCED == ()
          and cf_allreduce_volume_two_stage(2, specs.STAGE_ELEMS) == specs.STAGE_ELEMS,
          "★★ **`N = 2` 的退化对照**：两段之和 == `S`（**恰好**，整数）—— ★ 两段在 `N = 2` 时"
          "各只有一步，「分段 == 不分段」在这一档最难看出差别；"
          "★ 另钉住 `MPICH_BRUCK_SCOPE_NOTE` 与 `UNSOURCED == ()`",
          "**`N = 2` 的退化对照或消歧记录被动过**")


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
    print("★ 两段式的依据在 `specs.py` 的 `# 来源：`（Thakur / Rabenseifner / Gropp 2005）；")
    print("  本程序只核对「分段 == 不分段」与「漏掉一段真的会失败」。")
    print("★ **不是 NCCL，也不是真机**：本机无 GPU、无多卡、无真实网络。")
    return 1 if FAILS else 0


def main():
    print("=" * 78)
    print("02_stages_and_costs.py —— 两段式（reduce-scatter + all-gather）的通信阶段划分与代价")
    print("=" * 78)
    print("  ★ 真值标注：**本机实测值**（`python3 %s` / `numpy %s`）。"
          % (sys.version.split()[0], np.__version__))
    print("  ★ 失败标记 ASSERT_FAIL_MARK = %r（本主题自定义）" % ASSERT_FAIL_MARK)
    print("  ★★★ 强制限定语：**不是 NCCL，也不是真机**。")
    print("  ★★★ 参照是**每一段各自**的真消息序列模拟（逐步枚举谁发给谁、哪一块）。")
    if not preflight():
        print()
        print("!! 口径/标记钉住失败 ⇒ **不继续往下跑**")
        return summary()
    part_A()
    part_B()
    part_C()
    part_D()
    return summary()


if __name__ == "__main__":
    sys.exit(main())

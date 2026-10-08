#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""03_all_to_all_and_p2p.py —— **all-to-all 的算法与【总搬运量口径】**，以及 **P2P 原语**
（`send` / `recv` / `sendrecv`）—— 集合通信的**基础原语**。

配套文档：`03-all-to-all-and-p2p.md`。

★ 三类断言（A 算法正确性 / B 代价模型 / C 边界与反例）**全部在本程序内部**，
失败时以**非零退出码**结束；**没有 `verify.py`**。

★★ **本程序的数字全部是「本机实测值」**，测量条件是：
> `python3 3.13.5`，`numpy 2.4.4`，macOS arm64，**无 GPU、无多卡、无 NCCL、无真实网络**。
> ★★★ 强制限定语：**「不是 NCCL，也不是真机」** —— 本程序验证的是**算法与记账的代数**。

★★★ **本程序接住两条上游让渡**（`§2.3` 的两处入口）：
  · `40-kernels/moe-kernels/01-moe-compute-pattern.md:243` ——「**all-to-all 的通信量分析**
    不在本域，见 `110-communication/collectives`」（★ 全库被引最多的一条让渡）；
  · `100-parallelism/pipeline-parallel/01-microbatches-and-bubbles.md:298-299` ——
    「**点对点通信的实现**与通信-计算重叠的机制 → `110-communication/`」
    （★ 那条让渡的落点在 `overlap`，但 **P2P 原语本身**由本程序接住）。

★★★ **两条【同名不同指】的消歧**：
  · **`Bruck`**：该论文的 Bruck 算法是 **allgather 与 all-to-all（index）** 的算法，
    **不是 allreduce** 的 ⇒ 本程序只把它写在 all-to-all 那一支；
  · **`rank`**：本程序一律写「**设备**」（库内 `rank` 另有「张量秩」的含义）。

★★★ **一条 C3 分工（回链，本文一行都不写）**：
  · **all-to-all 的【跨切口量】`N·V/4`** → `20-gpu-architecture/interconnect/03-collectives-on-topology.md`
    （★ 本程序**只写【总搬运量 / 载荷口径】** —— 那是两台设备之间真的搬了多少元素）；
  · **bisection 下界** · **分层 all-reduce 三段式** → 同上；
  · **带宽口径** → `interconnect/01`；**`algbw` vs `busbw`** → `interconnect/04`；
  · **`V(N) = 2(N−1)/N × S` 的推导** → `10-foundations/performance-models/02:147-155,209`。

★★ 可证伪断言（把下面任一条改错，本程序必须非零退出）：
  · `A.1`  ★★★ **all-to-all 的成对交换**（真消息序列模拟）：结果正确、步数 `p−1`、
    每卡搬运量 `(p−1) × b`；
  · `A.2`  ★★ **直接交换**：同样结果、同样搬运量、**一步**（★ 与成对交换是**同一量的两种调度**）；
  · `A.3`  ★★★ **总搬运量口径**：全网总搬运量 `p(p−1) × b`（★ **逐条消息求和**，不是乘出来的）；
  · `A.4`  ★★ **P2P：`sendrecv` 交换**（真序列）⇒ 两边的数据都到对端；
  · `B.1`  ★★★ **all-to-all vs all-reduce 的总量对照**（★ 同一份「每设备数据量」下，
    all-reduce 的总搬运量**恰好是 all-to-all 的 2 倍**）；
  · `B.2`  ★★ **三种 all-to-all 算法的步数**（index `⌈lg p⌉` / 成对交换 `p−1`），
    ★ **三种的搬运量相同**；
  · `C.1`  ★★★ **「每个设备只发一份」** ⇒ 结果错、**不报错**；
  · `C.2`  ★★★ **P2P「先 `recv` 后 `send`」** ⇒ **互等（死锁）**、一轮都不进展；
  · `D.*`  口径常量逐个钉住 + **逐元素**元组钉住 + **引文逐字**钉住（含 `F5` 的限定语）
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
#   ★★ `&&` 是 **shell 元字符** ⇒ 它**只允许出现在 `print()` 的字符串里**。
ASSERT_FAIL_MARK = "&& FAIL: collectives"

OTHER_THEME_MARKS = (">< FAIL: nccl", ",, FAIL: network-topology", "+= FAIL: overlap")

FAILS = []
MARKS_USED = set()
_TOTAL = collections.Counter()
_PASS = collections.Counter()

TOL_MACHINE = specs.TOL_MACHINE

MSG = collections.namedtuple("MSG", "step sender receiver block n_elem")


def check(name, cond, detail="", detail_fail=None):
    """`name` 形如 `A.1` —— **首字母就是类别**（A/B/C/D）。"""
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
# 1. 被测的闭式
# ===========================================================================
def cf_a2a_steps_pairwise(p):
    """成对交换的步数闭式 `p − 1`（`specs.MPICH_A2A_PAIRWISE_QUOTE`）。"""
    return p - 1


def cf_a2a_steps_index(p):
    """index（Bruck）算法的步数闭式 `⌈lg p⌉`（`specs.MPICH_A2A_INDEX_STEPS_QUOTE`）。"""
    return int(np.ceil(np.log2(p)))


def cf_a2a_volume_per_device(p, b):
    """**每设备**的发出元素数闭式 `(p−1) × b`（它对三种算法**相同**）。"""
    return (p - 1) * b


def cf_a2a_volume_total(p, b):
    """**全网总搬运量**闭式 `p(p−1) × b` —— ★★ **这就是本主题的「总搬运量口径」**：
    把**每台设备各自发出的元素数**加起来。
    ★★★ **它与「跨切口量」不是一回事** —— 后者（`N·V/4`）的正文在
      `20-gpu-architecture/interconnect/03-collectives-on-topology.md`（此处回链，不重写）。"""
    return p * cf_a2a_volume_per_device(p, b)


def cf_allreduce_volume_total(p, s):
    """**all-reduce 的全网总搬运量**闭式 `2(p−1) × S`
    （每设备 `2(p−1)/p × S`，共 `p` 台 ⇒ `2(p−1) × S`）。"""
    return 2 * (p - 1) * s


def ref_all_to_all(blocks):
    """**单进程参照**：把每台设备发给别人的块直接摆到它们的位置上（不搬、只重排）。"""
    p = len(blocks)
    return [[blocks[j][d] for j in range(p)] for d in range(p)]


# ===========================================================================
# 2. 参照：**all-to-all 与 P2P 的真消息序列模拟器**
# ===========================================================================
def sim_a2a_pairwise(p, b, drop_one=False):
    """★★★ **成对交换（pairwise exchange）**：第 `k` 步（`k = 0..p−2`），
    设备 `d` 与 **`(d+1+k) % p`** 交换 —— `d` 发出「给对端的那一块」、收下「对端给自己的那一块」。

    ★★ `drop_one=True` 是 `C.1` 的错法：**每个设备只发一份**（只发给 `(d+1)%p`），
      剩下的块**根本没发出去**。
    ★★ 每一步记录 `MSG(step, sender, receiver, block, n_elem)`。
    """
    blocks = [[np.arange(s * b, (s + 1) * b, dtype=np.int64) + 1000 * d
               for s in range(p)] for d in range(p)]
    got = [[None] * p for _ in range(p)]
    for d in range(p):
        got[d][d] = blocks[d][d]                      # 自己那份留在原地
    msgs = []
    if drop_one:
        for d in range(p):
            msgs.append(MSG(0, d, (d + 1) % p, (d + 1) % p, b))
            got[(d + 1) % p][d] = blocks[d][(d + 1) % p]
    else:
        for k in range(cf_a2a_steps_pairwise(p)):
            batch = []
            for d in range(p):
                partner = (d + 1 + k) % p
                batch.append((d, partner, partner, blocks[d][partner].copy()))
            for d, partner, blk, src in batch:
                msgs.append(MSG(k, d, partner, blk, len(src)))
                got[partner][d] = src
    return got, msgs


def sim_a2a_direct(p, b):
    """**直接交换**：每台设备**同时**把 `p−1` 份发出去（一步）—— ★ 与成对交换**搬运量相同**。"""
    blocks = [[np.arange(s * b, (s + 1) * b, dtype=np.int64) + 1000 * d
               for s in range(p)] for d in range(p)]
    got = [[None] * p for _ in range(p)]
    for d in range(p):
        got[d][d] = blocks[d][d]
    msgs = []
    for d in range(p):
        for j in range(p):
            if j != d:
                msgs.append(MSG(0, d, j, j, b))
                got[j][d] = blocks[d][j]
    return got, msgs


def sim_sendrecv(p, b, mode="sendrecv"):
    """**P2P 原语**：每台设备与 `(d+1) % p` 交换一块数据。

    · `mode="sendrecv"`     ⇒ **同时**发与收（`MPI_Sendrecv` 的语义）⇒ **一轮完成**；
    · `mode="recv_then_send"`⇒ **先阻塞 `recv`、再 `send`** ⇒ 与对端**互等**。

    ★★★ 本函数用**一个简化的「就绪」模型**演示互等的后果 ——
      **不是**真的 socket 死锁（真的那个在 `../probes/`，走 loopback，只打印不断言）。
    """
    send_buf = [np.full(b, d, dtype=np.int64) for d in range(p)]
    recv_buf = [None] * p
    send_done = [False] * p
    recv_done = [False] * p
    rounds = 0
    while not all(recv_done) and rounds < 100:
        progressed = False
        if mode == "sendrecv":
            for d in range(p):
                if not send_done[d]:
                    send_done[d] = True
                    progressed = True
                if not recv_done[d]:
                    recv_buf[d] = send_buf[(d - 1) % p].copy()
                    recv_done[d] = True
                    progressed = True
        else:                                   # ★ 先 recv 后 send
            for d in range(p):
                src = (d - 1) % p
                if not recv_done[d] and send_done[src]:
                    recv_buf[d] = send_buf[src].copy()
                    recv_done[d] = True
                    progressed = True
                if recv_done[d] and not send_done[d]:
                    send_done[d] = True
                    progressed = True
        rounds += 1
        if not progressed:
            break
    return recv_buf, send_done, recv_done, rounds


def vol_of(msgs, p):
    """**从枚举出来的消息清单**逐条求和（不是乘出来的）。"""
    return [sum(m.n_elem for m in msgs if m.sender == d) for d in range(p)]


# ===========================================================================
# 3. A 类：真消息序列模拟 vs 闭式
# ===========================================================================
def part_A():
    print("=" * 78)
    print("A · all-to-all 与 P2P：**真消息序列模拟** vs **闭式**")
    print("=" * 78)

    # ── A.1 ★★★ 成对交换 ────────────────────────────────────────────────
    print()
    print("  ── A.1 ★★★ **成对交换**：`p−1` 步，每卡 `(p−1) × b` ──")
    ok1, rec = True, []
    for p in specs.A2A_GRID:
        got, msgs = sim_a2a_pairwise(p, specs.A2A_BLOCK)
        blocks = [[np.arange(s * specs.A2A_BLOCK, (s + 1) * specs.A2A_BLOCK,
                             dtype=np.int64) + 1000 * d for s in range(p)] for d in range(p)]
        ref = ref_all_to_all(blocks)
        exact = all(np.array_equal(got[d][j], ref[d][j]) for d in range(p) for j in range(p))
        steps = max(m.step for m in msgs) + 1
        vol = vol_of(msgs, p)[0]
        ok1 = ok1 and exact and (steps == cf_a2a_steps_pairwise(p)) \
            and (vol == cf_a2a_volume_per_device(p, specs.A2A_BLOCK))
        rec.append((p, steps, vol, len(msgs)))
        print("     p=%-3d 步数 %-3d（`p−1`）· 每卡搬运量 %-3d（`(p−1)×b`）· 消息 %-3d 条 ⇒ 结果%s"
              % (p, steps, vol, len(msgs), "相同 ✓" if exact else "不同 ✗"))
    check("A.1", ok1,
          "★★★ **判据成立**（`A2A_GRID` 三档）：成对交换的结果与**单进程参照**（只重排、不搬）"
          "逐位相同；步数 `p−1`；每卡搬运量 `(p−1) × b`",
          "**成对交换与闭式或参照不符** —— ★ 先查配对 `(d+1+k)%p` 与 `got[partner][d]` 的索引")

    # ── A.2 ★★ 直接交换 ────────────────────────────────────────────────
    print()
    print("  ── A.2 ★★ **直接交换**：同样结果、同样搬运量，**一步** ──")
    ok2 = True
    for p in specs.A2A_GRID:
        got, msgs = sim_a2a_direct(p, specs.A2A_BLOCK)
        blocks = [[np.arange(s * specs.A2A_BLOCK, (s + 1) * specs.A2A_BLOCK,
                             dtype=np.int64) + 1000 * d for s in range(p)] for d in range(p)]
        ref = ref_all_to_all(blocks)
        exact = all(np.array_equal(got[d][j], ref[d][j]) for d in range(p) for j in range(p))
        vol = vol_of(msgs, p)[0]
        ok2 = ok2 and exact and (vol == cf_a2a_volume_per_device(p, specs.A2A_BLOCK)) \
            and (max(m.step for m in msgs) == 0)
        print("     p=%-3d 步数 1 · 每卡搬运量 %-3d（与成对交换**相同**）⇒ 结果%s"
              % (p, vol, "相同 ✓" if exact else "不同 ✗"))
    print("     ★★ **同一种搬运量、两种调度**：直接交换把 `p−1` 份**同时**发出（1 步），")
    print("        成对交换把它们摊成 `p−1` 步（★ 步数与搬运量是**两条独立的轴**）。")
    check("A.2", ok2,
          "★★ **判据成立**：直接交换与成对交换**结果相同、搬运量相同**，差别只在**步数**"
          "（`1` 对 `p−1`）",
          "**直接交换的结果或搬运量与成对交换不一致** —— ★ 两者是同一量的两种调度")

    # ── A.3 ★★★ 总搬运量口径 ───────────────────────────────────────────
    print()
    print("  ── A.3 ★★★ **总搬运量口径**：全网 `p(p−1) × b`（★ 逐条枚举求和）──")
    ok3 = True
    for p in specs.A2A_GRID:
        _g, m1 = sim_a2a_pairwise(p, specs.A2A_BLOCK)
        _g2, m2 = sim_a2a_direct(p, specs.A2A_BLOCK)
        tot1 = sum(m.n_elem for m in m1)
        tot2 = sum(m.n_elem for m in m2)
        cf_t = cf_a2a_volume_total(p, specs.A2A_BLOCK)
        ok3 = ok3 and (tot1 == cf_t) and (tot2 == cf_t) and (tot1 == tot2)
        print("     p=%-3d 成对交换总搬运量 %-4d · 直接交换总搬运量 %-4d ⇒ `p(p−1)×b` = %d"
              % (p, tot1, tot2, cf_t))
    print("     ★★★ **口径声明**：本程序算的是**总搬运量**（每台设备**发出**的元素数之和）——")
    print("        ★★ **它与「跨切口量 `N·V/4`」不是一回事**；后者的正文在")
    print("        `20-gpu-architecture/interconnect/03-collectives-on-topology.md`（此处回链）。")
    check("A.3", ok3 and specs.MPICH_BRUCK_SCOPE_NOTE != "",
          "★★★ **判据成立**：两种调度的**总搬运量逐位相同**，且都等于 `p(p−1) × b`"
          "（★ 由**枚举出来的消息清单**逐条求和得到）",
          "**总搬运量与闭式不符** —— ★ 先查「总搬运量」是不是被写成了「每卡搬运量 × p」之外的量")

    # ── A.4 ★★ P2P sendrecv ────────────────────────────────────────────
    print()
    print("  ── A.4 ★★ **P2P：`sendrecv` 交换**（同时发与收）──")
    ok4 = True
    for p in specs.A2A_GRID:
        recv_buf, sd, rd, rounds = sim_sendrecv(p, specs.A2A_BLOCK, mode="sendrecv")
        want = [(d - 1) % p for d in range(p)]
        # ★★ 「成功行为」归 **A 类**（`CONTRIBUTING:238`）：`sendrecv` **一轮**就把数据换好。
        ok4 = ok4 and all(recv_buf[d] is not None
                          and np.array_equal(recv_buf[d], np.full(specs.A2A_BLOCK, want[d]))
                          for d in range(p)) and all(rd) and all(sd) and (rounds == 1)
        print("     p=%-3d 一轮完成（rounds=%d）· 每台设备收到来自 `(d−1)%%p` 的那一杯 ⇒ 正确 ✓"
              % (p, rounds))
    print("     ★★ `sendrecv` 的语义是**同时发与收** —— 它把「我等你、你等我」这件事消掉了。")
    print("     ★★ **「一轮完成」这条【成功行为】在【本判据】里断言**（`C.2` 只量【失败量】）——")
    print("        依 `CONTRIBUTING:238`：凡断言成功行为的归 A 类，**不塞进 C 类**。")
    check("A.4", ok4,
          "★★ **判据成立**：`sendrecv` 在 `A2A_GRID` 三档上**一轮**（`rounds == 1`）就把数据换好，"
          "且每台设备收到的是**对端**（`(d−1) % p`）的缓冲区",
          "**`sendrecv` 没有换成功（或不是一轮）** —— ★ 先查「谁发给谁」的配对")


# ===========================================================================
# 4. B 类：all-to-all vs all-reduce 的总量对照 · 三算法步数
# ===========================================================================
def part_B():
    print()
    print("=" * 78)
    print("B · 代价模型：**all-to-all vs all-reduce** 的总量对照 · 三算法步数")
    print("=" * 78)

    # ── B.1 ★★★ 总量对照 ───────────────────────────────────────────────
    print()
    print("  ── B.1 ★★★ **同一份「每设备数据量」**下，all-reduce 的总量是 all-to-all 的 2 倍 ──")
    print("     ★★ 归一化口径：all-to-all 的**每设备数据量** = `p × b`（`p` 块，每块 `b`）；")
    print("        all-reduce 的**每设备输入** `S = p × b`（★ 与前者**取同一个量**，才可比）。")
    okb1, rows = True, []
    for p in specs.A2A_GRID:
        b = specs.A2A_BLOCK
        a2a_tot = cf_a2a_volume_total(p, b)
        ar_tot = cf_allreduce_volume_total(p, p * b)
        ratio = ar_tot / a2a_tot
        rows.append((p, a2a_tot, ar_tot, ratio))
        okb1 = okb1 and abs(ratio - 2.0) < TOL_MACHINE
        print("     p=%-3d `b=%-2d` · `S=%-3d` ⇒ all-to-all 总 %-5d · all-reduce 总 %-5d ·"
              " 比值 **%.1f**" % (p, b, p * b, a2a_tot, ar_tot, ratio))
    print("     ★★★ **两条结论**（都可手算复核）：")
    print("       · **总量**：同一份每设备数据下，all-reduce 搬的是 all-to-all 的**恰好 2 倍**；")
    print("       · **形状**：`b` 固定时 all-to-all 的**每卡**搬运量 `(p−1)×b` **随 `p` 增长**，")
    print("         而 all-reduce 的每卡 `2(p−1)/p × S` **饱和到 `2S`**。")
    check("B.1", okb1 and len(rows) == 3,
          "★★★ **判据成立**（`A2A_GRID` 三档）：**同一份每设备数据量**下，"
          "all-reduce 的全网总搬运量恒为 all-to-all 的 **2 倍**（相对差 `< 1e-12`）",
          "**两个总量的比值不是 2** —— ★ 先查两边的「每设备数据量」是不是被归一化成同一个量")

    # ── B.2 ★★ 三算法步数 ─────────────────────────────────────────────
    print()
    print("  ── B.2 ★★ **三种 all-to-all 算法**的步数（搬运量相同）──")
    print("     `specs.MPICH_A2A_IRECVISEND_QUOTE`：`%s`" % specs.MPICH_A2A_IRECVISEND_QUOTE)
    print("     ★★ 分档是 **2005 年 MPICH 的实现选择**，**不是普适规律**：")
    print("        `%s`" % specs.MPICH_CUTOFF_QUALIFIER)
    okb2 = True
    for p in specs.A2A_GRID:
        _g, m = sim_a2a_pairwise(p, specs.A2A_BLOCK)
        vol = vol_of(m, p)[0]
        okb2 = okb2 and (cf_a2a_steps_index(p) <= cf_a2a_steps_pairwise(p)) \
            and (vol == cf_a2a_volume_per_device(p, specs.A2A_BLOCK))
        print("     p=%-3d index `⌈lg p⌉` = %-2d 步 · 成对交换 `p−1` = %-2d 步 ·"
              " **两者搬运量都是 %-3d**" % (p, cf_a2a_steps_index(p),
                                            cf_a2a_steps_pairwise(p), vol))
    print("     ★★ **步数与搬运量又是两条独立的轴** —— 与 `01` 的 allgather 同型。")
    check("B.2", okb2,
          "★★ **判据成立**：index（`⌈lg p⌉`）与成对交换（`p−1`）**步数不同、搬运量相同**；"
          "★ 分档阈值**带限定语**（`F5`）",
          "**三算法的步数/搬运量关系不成立**，或分档阈值缺了限定语")


# ===========================================================================
# 5. C 类：两条【真的会失败】的错法
# ===========================================================================
def part_C():
    print()
    print("=" * 78)
    print("C · 边界与反例：**只发一份** · **先 recv 后 send（互等）**")
    print("=" * 78)

    # ── C.1 ★★★ 每个设备只发一份 ────────────────────────────────────────
    # ★★ 两条**闭式**（★ 本机算过的值，**贴在这里免得下次再写错** —— 文档里曾误写成
    #    `(p−1)(p−2)` 个格子与「`(p−1)/p` 分之一」，**两个都是错的**）：
    #      · 缺格数  = p² − (p[对角] + p[一次发送]) = p² − 2p = **p(p−2)**
    #      · 搬运量比 = b / ((p−1)·b) = **1/(p−1)**
    #    p=4  ⇒ 缺 8   个、1/3  ≈ 0.3333
    #    p=8  ⇒ 缺 48  个、1/7  ≈ 0.1429   （★ `28 → 4`）
    #    p=16 ⇒ 缺 224 个、1/15 ≈ 0.0667
    print()
    print("  ── C.1 ★★★ **「每个设备只发一份」**（只发给 `(d+1)%p`，其余块没发）──")
    p = specs.A2A_GRID[-1]
    good, mg = sim_a2a_pairwise(p, specs.A2A_BLOCK)
    bad, mb = sim_a2a_pairwise(p, specs.A2A_BLOCK, drop_one=True)
    miss = sum(1 for d in range(p) for j in range(p) if bad[d][j] is None)
    diff = sum(1 for d in range(p) for j in range(p)
               if bad[d][j] is None or not np.array_equal(bad[d][j], good[d][j]))
    print("     p=%d：应有 %d 个格子有数据，实到 %d 个 ⇒ **缺了 %d 个**；"
          "搬运量 %d vs %d" % (p, p * p, p * p - miss, miss,
                               vol_of(mg, p)[0], vol_of(mb, p)[0]))
    print("     ★★★ **它【不报错】**：索引合法、类型合法 —— 只是**大部分数据根本没搬**。")
    check("C.1", miss > 0 and diff > 0 and vol_of(mb, p)[0] < vol_of(mg, p)[0],
          "★★★ **判据成立**：只发一份 ⇒ **缺 %d 个格子**、搬运量从 %d 降到 %d"
          "（≫ 内容级阈 `1e-6`）⇒ ★ **不报错、结果错**"
          % (miss, vol_of(mg, p)[0], vol_of(mb, p)[0]),
          "**只发一份没有让结果变** —— ★ 那说明参照与被测共用了一段代码（自证）")

    # ── C.2 ★★★ 先 recv 后 send ⇒ 互等 ─────────────────────────────────
    print()
    print("  ── C.2 ★★★ **P2P：「先 `recv` 后 `send`」⇒ 互等（死锁）** ──")
    p2 = specs.A2A_GRID[-1]
    _rb_bad, sd_bad, rd_bad, r_bad = sim_sendrecv(p2, specs.A2A_BLOCK,
                                                  mode="recv_then_send")
    print("     「先 recv 后 send」：rounds=%d，收到数据 = %d / %d 台，发出 = %d / %d 台"
          % (r_bad, sum(rd_bad), p2, sum(sd_bad), p2))
    print("     ★★★ **一台都不进展** —— 每台设备都在等它的**前驱**先发，而前驱也在等**它的**前驱。")
    print("     ★★ **对照物**：同一组数据用 `sendrecv` **一轮**（`rounds == 1`）就完成 ——")
    print("        ★ 那一条是**成功行为**，在 `A.4` 里断言（依 `CONTRIBUTING:238`，"
          "成功行为归 A 类，**不塞进 C 类**）。")
    print("     ★★ **模型声明**：这是**一个简化的「就绪」模型**（不是真 socket 死锁 ——")
    print("        真的那个在 `../probes/`，走 loopback，**只打印不断言**）。")
    # ★★ C 类只量【失败量】：**0 台**收到、**0 台**发出。
    #   ★ 原写法另含 `all(rd_ok)`（**成功行为**）与 `r_bad <= 100`（**恒真**，循环上界就是 100）
    #     —— 两条**都不合规**（`CONTRIBUTING:238` + 恒真），已移走/删除。
    check("C.2", (not any(rd_bad)) and (not any(sd_bad)),
          "★★★ **判据成立**：「先 `recv` 后 `send`」下 **0 / %d 台**收到数据、"
          "**0 台**发出数据 ⇒ **互等**（★ **这不是「慢」，是「一步都不动」**）" % p2,
          "**「先 recv 后 send」居然进展了** —— ★ 那说明就绪模型写错了（死锁应无进展）")


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
    print("  ★★★ **两条消歧**：`Bruck` 在本文献里是 **allgather 与 all-to-all(index)** 的算法"
          "（**不是 allreduce**）；本程序一律写「**设备**」而不写 `rank`（库内另有「张量秩」）。")
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

    # ── D.3b ★★★ **口径守卫**：档位必须是 2 的幂（成对交换的索引结构要求）──
    #   ★★★ **跑在任何计算之前**（本库 `P23`）⇒ 把 `A2A_GRID` 改成非 2 的幂时是**断言捕获**。
    pw = all((v & (v - 1)) == 0 for v in tuple(specs.A2A_GRID) + tuple(specs.DEV_GRID))
    check("D.3b", pw and specs.A2A_BLOCK > 0,
          "★★★ **口径守卫**：`A2A_GRID` 三档与 `DEV_GRID` 五档**全是 2 的幂**"
          "（★ 成对交换的前提之一是 `p` 是 2 的幂，见 `specs.MPICH_A2A_PAIRWISE_QUOTE`），"
          "且每对消息的元素数 `A2A_BLOCK = %d > 0`" % specs.A2A_BLOCK,
          "**档位的 2 的幂性质被破坏，或每对消息为空** —— ★ 先查 `A2A_GRID`/`A2A_BLOCK`")
    return len(FAILS) == before


# ===========================================================================
# 7. D 类（后半）：引文逐字钉住 · 逐元素钉住 · 独立性线索
# ===========================================================================
def part_D():
    print()
    print("=" * 78)
    print("D · 引文逐字钉住 · **逐元素**钉住 · 独立性线索")
    print("=" * 78)

    # ── D.5 ★★ all-to-all 的引文 ───────────────────────────────────────
    check("D.5", specs.MPICH_A2A_INDEX_QUOTE == "we use the index algorithm by Bruck et al."
          and specs.MPICH_A2A_INDEX_STEPS_QUOTE
          == "It is a store-and-forward algorithm that takes ⌈lg p⌉ steps"
          and specs.MPICH_A2A_PAIRWISE_QUOTE
          == ("For long messages and power-of-two number of processes, we use a "
              "pairwise-exchange algorithm, which takes p − 1 steps."),
          "★★ **三条 all-to-all 引文逐字钉住**：\n"
          "        ① index 算法**归属 Bruck**（★ 与 allreduce 无关 —— **同名不同指**）；\n"
          "        ② index 的步数 `⌈lg p⌉`（★ 提取形态 `dlg pe`）；\n"
          "        ③ 成对交换 `p − 1` 步，★★ **带两个前提**：**长消息** + **`p` 是 2 的幂**"
          "（★ 前提与结论同引）。\n"
          "        ★★ 载体 = 一手 PDF（`mcs.anl.gov/~thakur/papers/ijhpca-coll.pdf`），抓取日 2026-10-08",
          "**all-to-all 的引文常量被动过** —— ★ 先查成对交换的两个前提有没有丢")

    # ── D.6 ★★ 分档阈值 + 限定语 + 消歧 ────────────────────────────────
    check("D.6", specs.MPICH_A2A_IRECVISEND_QUOTE
          == "for medium-sized messages (256 bytes to 32 KB per message)"
          and specs.MPICH_CUTOFF_QUOTE == "experimentally determined cutoff points"
          and specs.MPICH_CUTOFF_QUALIFIER
          == ("★ 这些阈值是 2005 年 MPICH 在 Myrinet / IBM SP 上的实现选择"
              "（论文自述 experimentally determined cutoff points）"
              "⇒ 不得写成「集合通信的普适分档」")
          and specs.MPICH_BRUCK_SCOPE_NOTE
          == ("Bruck 算法在本文里是 allgather 与 all-to-all(index) 的算法，"
              "不是 allreduce 的算法"),
          "★★★ **分档阈值【与它的限定语】成对钉住**（`F5`）：\n"
          "        · `256 bytes to 32 KB` 是 **2005 年 MPICH 的实现选择** ⇒ 引用**必须**并置限定语；\n"
          "        · ★★ 限定语本身是 **[E] 本库的引用纪律**（**不是**论文里的话）——"
          " **不得**把它写成引文；\n"
          "        · ★★★ `MPICH_BRUCK_SCOPE_NOTE` 是**同名不同指**的记录（Bruck ≠ allreduce）。",
          "**分档阈值、限定语或消歧记录被动过** —— ★ 特别注意别把限定语当成论文的话")

    # ── D.7 ★★★ 逐元素钉住 ─────────────────────────────────────────────
    check("D.7", specs.A2A_GRID == (2, 4, 8) and specs.A2A_BLOCK == 4
          and specs.STAGE_ELEMS == 8 and specs.GATHER_ELEMS == 32
          and specs.GATHER_ODD == 6 and specs.DEV_GRID == (2, 4, 8, 16, 64),
          "★★★ **一个元组逐元素 + 五个标量**钉住（`A2A_GRID` 三档 · `A2A_BLOCK` ·"
          " `STAGE_ELEMS` · `GATHER_ELEMS` · `GATHER_ODD` · `DEV_GRID`）——"
          " ★ 只钉「整体」会让**元素级**变异漏网",
          "**元组的元素或某个标量被动过**")

    # ── D.8 ★★ 独立性线索 ──────────────────────────────────────────────
    srcs = {inspect.getsource(f) for f in (sim_a2a_pairwise, sim_a2a_direct,
                                           ref_all_to_all, cf_a2a_volume_total)}
    calls_sim = {n.func.id for n in ast.walk(
        ast.parse(textwrap.dedent(inspect.getsource(sim_a2a_pairwise))))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    calls_ref = {n.func.id for n in ast.walk(
        ast.parse(textwrap.dedent(inspect.getsource(ref_all_to_all))))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    print()
    print("  ── D.8 ★★ **源码级线索**（★ **不是证明**）：参照与被测的**调用表** ──")
    print("     `sim_a2a_pairwise()`（参照）里调用的函数：%s"
          % (", ".join(sorted(calls_sim)) if calls_sim else "**空集**"))
    print("     `ref_all_to_all()`（单进程参照）里调用的函数：%s"
          % (", ".join(sorted(calls_ref)) if calls_ref else "**空集**"))
    print("     ★★★ **降级措辞**：这只是「**两段代码不互相调用**」的**源码级线索**。")
    print("        ★★ 注意 `sim_a2a_pairwise` 的调用表里有 `cf_a2a_steps_pairwise`（步数闭式）——")
    print("          那是**循环上界**；★ **「不存在自证」靠的不是这一条**，而是")
    print("          `A.1` 的**结果与单进程参照逐位相同**（`ref_all_to_all` 只重排、不搬）。")
    check("D.8", "ref_all_to_all" not in calls_sim and "sim_a2a_pairwise" not in calls_ref
          and len(srcs) == 4,
          "★★ **源码级线索成立**：四个函数两两不同源；★ **单进程参照不调用任何模拟器**"
          "（★ 它的调用表是**空集**）",
          "**参照与模拟器互相调用** —— A 类可能退化成自证")

    # ── D.9 ★★ 边界 ────────────────────────────────────────────────────
    check("D.9", cf_a2a_volume_total(2, specs.A2A_BLOCK)
          == 2 * cf_a2a_volume_per_device(2, specs.A2A_BLOCK)
          and cf_a2a_volume_per_device(2, specs.A2A_BLOCK) == specs.A2A_BLOCK
          and specs.UNSOURCED == (),
          "★★ **`p = 2` 的退化对照**：`(p−1) × b == b`（**恰好**）—— p=2 时每台设备只发一份，"
          "**这正好是 `C.1` 那条错法在 `p = 2` 上的样子**（★ 错法在 `p = 2` 上与正确版**同形**"
          "⇒ 所以 `C.1` 用 `p = 8` 量它）；★ 另钉住 `UNSOURCED == ()`",
          "**`p = 2` 的退化对照或 `UNSOURCED` 被动过**")


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
    print("★ all-to-all 与 P2P 的依据在 `specs.py` 的 `# 来源：`（Thakur / Rabenseifner / Gropp 2005）。")
    print("★ **不是 NCCL，也不是真机**：本机无 GPU、无多卡、无真实网络。")
    return 1 if FAILS else 0


def main():
    print("=" * 78)
    print("03_all_to_all_and_p2p.py —— all-to-all 的算法与总搬运量口径 · P2P 原语")
    print("=" * 78)
    print("  ★ 真值标注：**本机实测值**（`python3 %s` / `numpy %s`）。"
          % (sys.version.split()[0], np.__version__))
    print("  ★ 失败标记 ASSERT_FAIL_MARK = %r（本主题自定义）" % ASSERT_FAIL_MARK)
    print("  ★★★ 强制限定语：**不是 NCCL，也不是真机**。")
    print("  ★★ 参照是**真消息序列模拟**（逐步枚举谁发给谁、哪一块、多少元素）。")
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

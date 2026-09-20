#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""03_smem_and_sass.py —— 共享内存访存计数与 PTX→SASS 的落差。

配套文档：`03-memory-and-sass-gap.md`。三类断言全部在本程序内部。

★★ 本篇的反直觉量化结论（**本域最值钱的一条**）：

    同一个 `+1` padding（行距 8 → 9），
    **在 32 行 × 1 列的 lane 模式下有效（8 路 → 1 路），
    在 8 行 × 4 列的 lane 模式下完全无效（2 路 → 2 路）** ——
    而两者都是「按列取用」。

★★ 本域三条强制限定语，在本篇的落点：

  1. **不是硬件**——bank 冲突路数是**本解释器**从其 `ld.shared` 的**字节地址轨迹**上
     算出来的模型值（32 个 bank × 4 字节，同 word 是广播）。真实硬件上还有
     32-bit / 64-bit 与 128-bit 访问的不同判定池、以及跨请求的合并
     （`20-gpu-architecture/memory-system/03` §2 末三条讲的就是这个）。
     本程序**一个硬件数字都没有**（本机**无 GPU**）。
  2. **按 PTX ISA `9.7.10.6` 等章节实现**——本篇用到的 `ld.shared` 的字节地址口径
     （`bank = (addr / 4) mod 32`）来自**该文档的既有篇章**
     `20-gpu-architecture/memory-system/03-shared-memory-bank-conflict`，
     **回链不复述**；ISA 引文与章节号在 `specs.ISA`。
     ★ `shfl.sync.up` 的有意偏离与「未定案」状态见 `02` 篇 §4.1 与
     `specs.UP_MODE_UNDECIDED`（本篇不用到 `.up`）。
  3. **手写 PTX 的语法未经过任何工具验证**——本机**无 `ptxas`**。

★★ **本篇不是对既有文档的订正**：

  `20-gpu-architecture/memory-system/03-shared-memory-bank-conflict.md`（第 358 行）
  **自己已经写明**它的结论带一个前提——「`blockDim.x = 32`，也就是一个 warp
  恰好铺满 tile 的一整行」。**那正是本程序的模式 B。**
  该篇的 `gcd(行距, 32)` 规则在模式 B 下**完全正确**，本程序的 `A.4` 穷举 stride
  把它**复核了一遍**（闭式 vs 解释器地址，两条路径）。
  ⇒ 本篇的贡献是**把规则推广到「warp 不只铺满一行」的情形**（模式 A），**不是订正**。
"""

import ast
import math
import os
import re
import sys

import numpy as np

import ptxsub
import specs

HERE = os.path.dirname(os.path.abspath(__file__))

# ★ 03 的判据消费的是 `specs.BANK_CONFLICT` 那张**推导表**，不消费 ISA 锚点
#   （ISA 的 9 条由 `01` 与 `02` 覆盖，三者两两不交、并集 == `specs.ISA`）。
COVERED_ANCHORS = ()


# ===========================================================================
# 0. 断言记账
# ===========================================================================
FAILS = []
_TOTAL = {"A": 0, "B": 0, "C": 0}
_PASS = {"A": 0, "B": 0, "C": 0}
RUNTIME_LABELS = set()


def check(cls, name, cond, detail=""):
    _TOTAL[cls] += 1
    RUNTIME_LABELS.add(name)
    if cond:
        _PASS[cls] += 1
        print("  [ok]   %-5s %s" % (name, detail))
    else:
        FAILS.append(name)
        print("  [FAIL] %-5s %s" % (name, detail))
    return bool(cond)


def eq(cls, name, got, want, detail=""):
    return check(cls, name, got == want,
                 "%s: %r%s" % (detail, got,
                               "" if got == want else "  ≠ 期望 %r" % (want,)))


def _s(tpl, *args):
    """`tpl % args`，但先把模板里的 PTX 寄存器名（`%r` / `%p`）转义（理由同 `01`）。"""
    return tpl.replace("%r", "%%r").replace("%p", "%%p") % args


def summary():
    t = sum(_TOTAL.values())
    p = sum(_PASS.values())
    print()
    print("汇总: A %d/%d · B %d/%d · C %d/%d ⇒ 断言 %d 条，通过 %d 条，失败 %d 条"
          % (_PASS["A"], _TOTAL["A"], _PASS["B"], _TOTAL["B"],
             _PASS["C"], _TOTAL["C"], t, p, len(FAILS)))
    if FAILS:
        print("失败清单: %s" % ", ".join(FAILS))
    return 1 if FAILS else 0


# ===========================================================================
# 1. 两套 lane 模式的 PTX（★ 手写，语法未经任何工具验证）
# ===========================================================================
# 两个片段**只差两条指令的立即数**：`(and, shr)` 决定 lane → (row, col) 的映射。
#   模式 A：row = t & 7,  col = t >> 3    ⇒ 8 行 × 4 列（R = 8）
#   模式 B：row = t & 31, col = t >> 5    ⇒ 32 行 × 1 列（R = 32）
# 行距由宿主预置在 `%r8` —— 一个 warp 的 32 个 lane 读 **32 个不同的元素**。

COL_A_BODY = """
.version 6.0
.target sm_70
.reg .b32 %r<9>;
.shared .align 4 .b8 smem[8192];

    mov.u32  %r0, %tid.x;
    mov.u32  %r6, %r8;                 // 行距（宿主预置）
    and.b32  %r1, %r0, 7;              // row = t % 8
    mul.lo.u32 %r1, %r1, %r6;          // row * stride
    shr.u32  %r2, %r0, 3;              // col = t // 8
    add.u32  %r1, %r1, %r2;            // flat = row * stride + col
    mul.lo.u32 %r2, %r1, 4;            // 字节地址（fp32 = 4 字节）
    ld.shared.u32 %r3, [%r2];
    ret;
"""

COL_B_BODY = """
.version 6.0
.target sm_70
.reg .b32 %r<9>;
.shared .align 4 .b8 smem[8192];

    mov.u32  %r0, %tid.x;
    mov.u32  %r6, %r8;
    and.b32  %r1, %r0, 31;             // row = t % 32
    mul.lo.u32 %r1, %r1, %r6;
    shr.u32  %r2, %r0, 5;              // col = t // 32
    add.u32  %r1, %r1, %r2;
    mul.lo.u32 %r2, %r1, 4;
    ld.shared.u32 %r3, [%r2];
    ret;
"""

# 广播对照：全部 lane 读**同一个 word**
BCAST_BODY = """
.version 6.0
.target sm_70
.reg .b32 %r<4>;
.shared .align 4 .b8 smem[256];

    mov.u32  %r0, 0;
    ld.shared.u32 %r1, [%r0];
    ret;
"""

# 广播对照：每 4 个 lane 读同一个 word（共 8 个 word）
QUAD_BODY = """
.version 6.0
.target sm_70
.reg .b32 %r<4>;
.shared .align 4 .b8 smem[256];

    mov.u32  %r0, %tid.x;
    shr.u32  %r0, %r0, 2;
    mul.lo.u32 %r0, %r0, 4;
    ld.shared.u32 %r1, [%r0];
    ret;
"""

# 32 个 lane 读 32 个**不同**的 word（无冲突的另一个极端）
DISTINCT_BODY = """
.version 6.0
.target sm_70
.reg .b32 %r<4>;
.shared .align 4 .b8 smem[256];

    mov.u32  %r0, %tid.x;
    mul.lo.u32 %r0, %r0, 4;
    ld.shared.u32 %r1, [%r0];
    ret;
"""


# ===========================================================================
# 2. 取地址与三条互不重叠的冲突计数路径
# ===========================================================================
def lane_addrs(body, stride):
    """★ 路径 ①（被测口径）：**解释器实际产生的 `ld.shared` 字节地址**。

    ★ 一条硬纪律：地址**从解释器的访存轨迹里取**，不另算一遍公式 ——
      否则「被测对象」与「参照实现」就变成了同一段代码（P12 自造真值）。
    """
    prog = ptxsub.parse(body)
    sim = ptxsub.Sim(prog, nthreads=32, smem_words=2048, fill=0)
    # ★ 行距放在 `%r8`（两个 COL 片段都声明了 `%r<9>`）。
    #   对照用的三个广播片段只声明 `%r<4>`，**不改它们的寄存器文件** —— 所以先问一句
    #   「这个片段里到底有没有 %r8」，没有就跳过预置（不是静默忽略：下面立刻断言它是 0）。
    if prog.nreg > 8:
        for l in range(32):
            sim.regs[0][l][8] = ptxsub.u32(stride)
    else:
        assert stride == 0, "片段 %r 没有 %r8，却要求行距 %r" % (body[:20], stride)
    sim.run()
    by = {l: a for (w, l, a) in sim.ld_shared_trace if w == 0}
    assert len(by) == 32, "warp0 的 32 个 lane 必须各产生 1 次 ld.shared，实际 %d" % len(by)
    return [by[l] for l in range(32)]


def conflict_numpy(byte_addrs, nbank=32, wb=4):
    """★ 路径 ②：numpy 向量化（与 `ptxsub.max_bank_conflict` 无共同代码）。

    同一个 bank 上落在**不同 word** 的线程数 = 冲突路数；同 word 是广播，只算 1 路。
    """
    words = np.array([a // wb for a in byte_addrs], dtype=np.int64)
    bank = words % nbank
    return max(int(np.unique(words[bank == k]).size) for k in range(nbank))


def conflict_closed_form(rows, cols, stride, nbank=32):
    """★ 路径 ③：**纯 Python 闭式**（不用解释器、不用 numpy、不看任何地址）。

    直接按模式的**定义**算 `flat = row * stride + col`，再按 `(bank, word)` 分组。
    """
    groups = {}
    for t in range(rows * cols):
        r, c = t % rows, t // rows
        word = r * stride + c
        groups.setdefault(word % nbank, set()).add(word)
    return max(len(v) for v in groups.values())


def banks_covered(rows, cols, stride, nbank=32):
    """模式覆盖了多少个**不同的 bank**（`+1` padding 机制解释用的量）。"""
    used = {}
    for t in range(rows * cols):
        r, c = t % rows, t // rows
        word = r * stride + c
        used.setdefault(word % nbank, []).append(word)
    return len(used), {b: len(set(ws)) for b, ws in used.items() if len(set(ws)) > 1}


SPAWN_MODULES = ("subprocess", "ctypes", "pty", "shlex", "multiprocessing")
SPAWN_CALLS = ("system", "popen", "execv", "execvp", "execve", "spawnv", "spawnl", "fork")

MODES = {
    "A": (specs.BANK_CONFLICT["modes"]["A"]["rows"],
          specs.BANK_CONFLICT["modes"]["A"]["cols"], COL_A_BODY),
    "B": (specs.BANK_CONFLICT["modes"]["B"]["rows"],
          specs.BANK_CONFLICT["modes"]["B"]["cols"], COL_B_BODY),
}
STRIDES = specs.BANK_CONFLICT["strides"]


# ===========================================================================
# 3. A 类
# ===========================================================================
def section_A():
    print("=" * 78)
    print("A 类：算法正确性 —— 三条**互不重叠**的冲突计数路径")
    print("=" * 78)
    print("  路径① 解释器的 ld.shared 字节地址 → ptxsub.max_bank_conflict")
    print("  路径② numpy 向量化（(word//4)%32 分箱，逐 bank 数不同 word）")
    print("  路径③ 纯 Python 闭式（按模式定义算 flat = row*stride + col，不看地址）")
    print()

    allstrides = list(range(8, 41))
    ok12 = ok13 = True
    first_bad = None
    for m in ("A", "B"):
        rows, cols, body = MODES[m]
        for st in allstrides:
            a = ptxsub.max_bank_conflict(lane_addrs(body, st))
            b = conflict_numpy(lane_addrs(body, st))          # 同一批地址，另一套统计
            c = conflict_closed_form(rows, cols, st)
            if a != b:
                ok12, first_bad = False, (m, st, a, b)
            if a != c:
                ok13, first_bad = False, (m, st, a, c)
    eq("A", "A.1", ok12, True,
       "★ 路径①（解释器地址 + `ptxsub` 统计）与路径②（numpy 统计）在 **2 模式 × 33 个行距**"
       " 上逐格一致（不一致的第一处：%r）" % (first_bad,))
    eq("A", "A.2", ok13, True,
       "★ 路径① 与路径③（**闭式，完全不看地址**）在同样 66 格上逐格一致")
    eq("A", "A.3", len(allstrides) * 2, 66, "★ 交叉校验的格子数 = 2 × 33")

    # ---- A.4 ★★ 模式 B 就是既有文档那条规则：`gcd(行距, 32)` ----
    gcd_ok, gcd_bad = True, []
    for st in range(1, 65):
        rows, cols, body = MODES["B"]
        a = ptxsub.max_bank_conflict(lane_addrs(body, st))
        g = math.gcd(st, 32)
        if a != g:
            gcd_ok = False
            gcd_bad.append((st, a, g))
    check("A", "A.4", gcd_ok,
          "★★ **模式 B 的冲突路数 == `gcd(行距, 32)`**（行距 1..64 穷举）：%s"
          % ("全部吻合" if gcd_ok else "例外 %r" % gcd_bad[:4]))
    eq("A", "A.4b", specs.BANK_CONFLICT["existing_doc_rule"], "gcd(行距, 32)",
       "★ `specs.BANK_CONFLICT['existing_doc_rule']` = %r —— 这与既有文档"
       " `20-gpu-architecture/memory-system/03-shared-memory-bank-conflict.md`"
       " §1 第五步的规则**逐字相同**" % specs.BANK_CONFLICT["existing_doc_rule"])
    eq("A", "A.4c", specs.BANK_CONFLICT["existing_doc_rule_scope"], "B",
       "★★ 那条规则**只在模式 B 下成立**（所以本程序的 `A.4` 用模式 B 复核它）——"
       "这不是订正，是把适用范围写清楚")
    print("      ★★ **本篇不是对既有的 `memory-system/03` 的订正**：该篇第 358 行"
          "自己已声明前提「`blockDim.x = 32`，一个 warp 恰好铺满 tile 的一整行」，"
          "那正是**模式 B**；规则在该范围内完全正确。本篇讲的是 **warp 跨多行**（模式 A）。")
    return allstrides


# ===========================================================================
# 4. B 类
# ===========================================================================
def section_B(allstrides):
    print()
    print("=" * 78)
    print("B 类：代价模型 —— ★ 本域最值钱的量化结论（两套 lane 模式 × 五个行距）")
    print("=" * 78)
    rowsA, colsA, _ = MODES["A"]
    rowsB, colsB, _ = MODES["B"]
    print()
    print("  模式 A：lane t 读 (row = t %% %d, col = t // %d) ⇒ warp 覆盖 **%d 行 × %d 列**（R = %d）"
          % (rowsA, rowsA, rowsA, colsA, rowsA))
    print("  模式 B：lane t 读 (row = t %% %d, col = t // %d) ⇒ warp 覆盖 **%d 行 × %d 列**（R = %d）"
          % (rowsB, rowsB, rowsB, colsB, rowsB))
    print()
    print("  %-8s %-14s %-14s" % ("行距", "模式 A（R=8）", "模式 B（R=32）"))
    got = {}
    for st in STRIDES:
        a = ptxsub.max_bank_conflict(lane_addrs(MODES["A"][2], st))
        b = ptxsub.max_bank_conflict(lane_addrs(MODES["B"][2], st))
        got[st] = {"A": a, "B": b}
        tag = {8: "（无 padding）", 9: "（+1 padding）", 12: "（行距 12）",
               32: "", 33: "（+1 padding）"}[st]
        print("  %-8s %-14s %-14s %s" % (st, "%d 路" % a, "%d 路" % b, tag))
    print()

    want = {int(k): v for k, v in specs.BANK_CONFLICT["table"].items()}
    eq("B", "B.1", {st: got[st]["A"] for st in STRIDES},
       {st: want[st]["A"] for st in STRIDES},
       "★ 模式 A 一列")
    eq("B", "B.2", {st: got[st]["B"] for st in STRIDES},
       {st: want[st]["B"] for st in STRIDES},
       "★ 模式 B 一列")

    # ---- B.3/B.4 机制：覆盖了多少 bank、哪些 bank 被重复用到 ----
    cov9, over9 = banks_covered(rowsA, colsA, 9)
    cov12, over12 = banks_covered(rowsA, colsA, 12)
    eq("B", "B.3", cov9, specs.BANK_CONFLICT["coverage_mode_A"][9],
       "★ 模式 A、行距 9：四个列合起来只覆盖 **%d/32** 个 bank；被**重复用到**（不同 word）"
       "的 bank = %s ⇒ 仍 %d 路" % (cov9, sorted(over9), got[9]["A"]))
    eq("B", "B.3b", tuple(sorted(over9)), specs.BANK_CONFLICT["reloaded_banks_mode_A"][9],
       "★ 重复的正好是 bank {0,1,2}（第 8 行那一组的 {31,0,1,2} 与第 0 行的 {0,1,2,3} 重叠）")
    eq("B", "B.4", (cov12, sorted(over12)), (specs.BANK_CONFLICT["coverage_mode_A"][12], []),
       "★ 模式 A、行距 12：**恰好铺满 32/32**、零重叠 ⇒ %d 路" % got[12]["A"])

    # ---- B.5 「冲突数被 R 封顶」 ----
    worst = max(ptxsub.max_bank_conflict(lane_addrs(MODES["A"][2], st)) for st in allstrides)
    check("B", "B.5", worst <= rowsA,
          "★★ **R 封顶**：模式 A（R = %d）在行距 8..40 上的最大冲突 = **%d 路 ≤ R** ——"
          " 一趟最多只有 %d 个 lane 落在同一列，**物理上到不了 32 路**"
          % (rowsA, worst, rowsA))
    eq("B", "B.5b", max(ptxsub.max_bank_conflict(lane_addrs(MODES["B"][2], st))
                        for st in allstrides), rowsB,
       "★ 对照：模式 B（R = 32）能到 **%d 路**（同一个 R 封顶机制的另一端）" % rowsB)

    # ---- B.6 广播与「全是不同 word」都不算冲突 ----
    eq("B", "B.6", ptxsub.max_bank_conflict(lane_addrs(BCAST_BODY, 0)), 1,
       "★ 32 个 lane 读**同一个 word** ⇒ 1 路（广播，不算冲突）")
    eq("B", "B.6b", ptxsub.max_bank_conflict(lane_addrs(QUAD_BODY, 0)), 1,
       "★ 每 4 个 lane 读同一个 word（共 8 个 word）⇒ 1 路")
    eq("B", "B.6c", ptxsub.max_bank_conflict(lane_addrs(DISTINCT_BODY, 0)), 1,
       "★ 32 个 lane 读 32 个**不同** word（铺满 32 个 bank）⇒ 1 路")
    print("      ★ 三个极端都是 1 路 ⇒ 冲突的判据是「**同一 bank 上的不同 word**」，"
          "两端都不触发")
    return got


# ===========================================================================
# 5. C 类
# ===========================================================================
def section_C(got, allstrides):
    print()
    print("=" * 78)
    print("C 类：边界与反例 —— **必须真的跑出问题**")
    print("=" * 78)

    # ---- C.1 ★★ 教科书式的 +1 padding 在模式 A 下**无效** ----
    eq("C", "C.1", (got[8]["A"], got[9]["A"]), (2, 2),
       "★★ 模式 A：行距 8（无 padding）%d 路，行距 9（**+1 padding**）**仍然** %d 路 ——"
       " 教科书做法在这里**完全无效**" % (got[8]["A"], got[9]["A"]))
    print("      机制：9 与 32 互质，但**四个列合起来只有 29/32 个 bank**"
          "（{0,1,2} 被两个列同时用到、且是不同 word）⇒ 仍 2 路；"
          "同时 bank 8/17/26 空着。")

    # ---- C.2 ★★ 同一个 +1 padding 在模式 B 下**有效** ----
    eq("C", "C.2", (got[8]["B"], got[9]["B"]), (8, 1),
       "★★ 同一个 `+1` padding（8 → 9），换到模式 B：**%d 路 → %d 路** —— 有效"
       % (got[8]["B"], got[9]["B"]))
    check("C", "C.2b", got[9]["A"] != got[9]["B"],
          "★★ **同一个 stride、同一份数据、两张相反结论**（模式 A %d 路 vs 模式 B %d 路）"
          " ⇒ P9：写「列访问」时必须**连 lane 模式一起定义**" % (got[9]["A"], got[9]["B"]))

    # ---- C.3 模式定义是数字的一部分 ----
    #   出题：**同一句「按列取用」**（lane t 读 tile[t % R][t // R]），
    #   只改 R，冲突路数就变。所以「列访问的冲突是几路」在给出 lane 模式之前
    #   **不是一个良定义的问题**（P9）。
    mixed = [(st, ptxsub.max_bank_conflict(lane_addrs(MODES["A"][2], st)),
              ptxsub.max_bank_conflict(lane_addrs(MODES["B"][2], st))) for st in STRIDES]
    ndiff = sum(1 for _, a, b in mixed if a != b)
    check("C", "C.3", ndiff > 0,
          "★ 五个行距里有 **%d/5** 个两模式的冲突路数不同：%s ⇒ 同一个 stride、"
          "同一句话（「按列取用」），**换一个 lane 模式就是另一个数**"
          % (ndiff, ", ".join("行距%d: %d vs %d" % t for t in mixed)))
    addr_same = all(lane_addrs(MODES["A"][2], st) == lane_addrs(MODES["B"][2], st)
                    for st in STRIDES)
    check("C", "C.3b", not addr_same,
          "★ 两个模式产生的 `ld.shared` 字节地址集合**并不相同** ⇒ 差别不在「怎么数」，"
          "而在「读的是哪些地址」")

    # ---- C.4 ★★ R 封顶的直接反例：同一个行距 32，两个模式差 4 倍 ----
    eq("C", "C.4", (got[32]["A"], got[32]["B"]), (8, 32),
       "★★ 行距 32：模式 A **%d 路**（不是 32 路，被 R=8 封顶）、模式 B **%d 路**"
       "；同一个 stride、同一个 tile、差 **%d 倍**"
       % (got[32]["A"], got[32]["B"], got[32]["B"] // got[32]["A"]))

    # ---- C.5 行距 33（+1 padding）在两个模式下的效果又是反的 ----
    eq("C", "C.5", (got[33]["A"], got[33]["B"]), (4, 1),
       "★ 行距 33（再 +1）：模式 A 反而**升到** %d 路、模式 B 仍是 %d 路 ⇒ "
       "「加 padding 一定变好」在两个模式上都不是无条件成立的"
       % (got[33]["A"], got[33]["B"]))

    # ---- C.6 ★ `gcd(行距,32)` 这条规则**只在其声明的范围内**成立 ----
    fails = []
    for st in allstrides:
        rule = math.gcd(st, 32)
        actual = ptxsub.max_bank_conflict(lane_addrs(MODES["A"][2], st))
        if rule != actual:
            fails.append(st)
    check("C", "C.6", len(fails) > 0,
          "★★ `gcd(行距, 32)` 规则用在**模式 A** 上：行距 8..40 里 **%d/33 个不成立**"
          "（例如行距 9：规则给 %d、实际 %d）—— 「规则有适用范围」这件事是**可算的**"
          % (len(fails), math.gcd(9, 32),
             ptxsub.max_bank_conflict(lane_addrs(MODES["A"][2], 9))))
    eq("C", "C.6b", specs.BANK_CONFLICT["existing_doc_rule_scope"], "B",
       "★★ 所以 `specs.BANK_CONFLICT` 把那条规则的**作用域**记成 'B'（模式 B）——"
       " 这不是订正，是**把既有文档第 358 行自己写下的前提抄进常量**")

    # ---- C.7 反向断言：本篇没有任何能力去调用外部工具 ----
    #   ★ 用 `ast` 扫**导入与调用**，不做字符串匹配 —— 否则这条断言会被它自己的
    #     说明文字（里面必须点名那几个工具）触发成假阳性（第一版就是这么错的）。
    src = open(os.path.join(HERE, "03_smem_and_sass.py"), encoding="utf-8").read()
    spawned = []
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.Import):
            spawned += [a.name for a in n.names if a.name.split(".")[0] in SPAWN_MODULES]
        elif isinstance(n, ast.ImportFrom):
            if (n.module or "").split(".")[0] in SPAWN_MODULES:
                spawned.append(n.module)
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
            if n.func.attr in SPAWN_CALLS:
                spawned.append(n.func.attr)
    eq("C", "C.7", sorted(set(spawned)), [],
       "★ 本程序**不导入** %s、**不调用** %s ⇒ 它没有能力 spawn 任何进程，"
       "也就没有能力调用 `nvdisasm` / `ncu` / `ptxas`"
       % (", ".join(SPAWN_MODULES), ", ".join(SPAWN_CALLS)))
    bare = "实" + "测"
    eq("C", "C.7b", src.count(bare), 0,
       "★ 本程序源码里 0 次出现裸用的那两个汉字（本域 B10）")
    return True


# ===========================================================================
# 6. main
# ===========================================================================
def main():
    print("=" * 78)
    print("03_smem_and_sass.py —— 共享内存访存计数与 PTX→SASS 的落差")
    print("=" * 78)
    print("  解释器      : %s" % sys.executable)
    print("  python      : %s / numpy %s" % (sys.version.split()[0], np.__version__))
    print("  ★ 限定语 1  : **不是硬件** —— bank 路数是本解释器从其 ld.shared 字节地址上算的模型值")
    print("  ★ 限定语 2  : 按 PTX ISA 9.7.10.6 等章节实现；bank 冲突机制**回链**"
          " 20-gpu-architecture/memory-system/03，本篇不重推")
    print("  ★ 限定语 3  : **手写 PTX 的语法未经任何工具验证** —— 本机无 ptxas / nvcc / nvdisasm")
    print()
    allstrides = section_A()
    got = section_B(allstrides)
    section_C(got, allstrides)
    static = _static_labels()
    print()
    print("-" * 78)
    eq("A", "A.5", sorted(static - RUNTIME_LABELS - {"A.5"}), [],
       "★ 静态标签减去运行时标签 == 空（静态提取不漏）")
    print()
    print("  ── PTX → SASS 的两大落差（★ 只做**描述性**介绍，本机无 `nvdisasm`）──")
    print("     ① **寄存器分配**：PTX 的虚拟寄存器是**无上限**的（`.reg .b32 %%r<N>` 是声明），")
    print("        SASS 的物理寄存器是**有限**的（每线程最多 255 个）⇒ 编译器要做分配，")
    print("        分配不下就 spill 到 local memory。**本库无法观测这一步**（无 `nvdisasm`）。")
    print("     ② **指令调度**：源码顺序 ≠ 发射顺序。编译器为掩盖延迟会重排、")
    print("        甚至把 PTX 的一条指令拆成多条 SASS（或反之）。")
    print("        ⇒ **本程序在 PTX 层数出来的发射数，不等于 SASS 层的发射数**。")
    return summary()


def _static_labels():
    src = open(os.path.join(HERE, "03_smem_and_sass.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    return {n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and re.match(r"^[ABC]\.\d+[a-z]?$", n.value)}


if __name__ == "__main__":
    sys.exit(main())

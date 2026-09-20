#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""02_warp_execution.py —— warp 级执行：**同一个结果，两种写法，两笔代价**。

配套文档：`02-warp-level-execution.md`。三类断言全部在本程序内部
（`CONTRIBUTING` §2 第二种情形），失败以非零退出码结束。

★★ 本域三条强制限定语，在本篇的落点：

  1. **不是硬件**——`ptxsub` **逐 warp 推进到阻塞点**，不是真实的 warp 交错调度。
     本篇的发射数 / 活跃 lane-指令数 / 发散步数**全是模型值**。
     尤其：`BRANCH` 比 `PREDICATED` 多 2 次发射这件事，是**本模型的产物**；
     真实硬件上两者都会被 SASS 层的调度与谓词化改写，「多 2 条」**不能**读作硬件行为。
  2. **按 PTX ISA `9.7.10.6`（`shfl.sync`）/ `9.7.15.1`（`bar, barrier`）实现**——
     引文与章节号逐字在 `specs.ISA`。★ **`shfl.sync.up` 处必须写明本库的实现是对
     ISA 文本的有意偏离**：原文写 `maxLane`，本库用 `minLane`，**且这一点未定案**
     （`specs.UP_MODE_UNDECIDED`；本程序 §C.3/§C.4 把「可判定的那一半」钉住了）。
  3. **手写 PTX 的语法未经过任何工具验证**——本机**无 `ptxas`**。

★ 本程序的 A 类参照实现走**三条互不重叠的路径**：
    ① numpy 的 `reshape(-1,32).sum(axis=1)`；
    ② **纯 Python 位运算**的 butterfly 归约（不用解释器、不用 numpy）；
    ③ 闭式的 clamp/segmask 逐 lane 映射公式（与 `ptxsub` 的分支实现无共同代码）。

★ 本篇的反直觉量化结论：**同一份输入、同一个结果，`BRANCH` 与 `PREDICATED`
  在模型里的发射数差 2、活跃 lane-指令数差 16**；而**重聚是 PC 相等自然发生的**，
  不需要任何显式的 re-convergence 栈。
"""

import ast
import os
import re
import sys

import numpy as np

import ptxsub
import specs

HERE = os.path.dirname(os.path.abspath(__file__))

# ★ 三个集合两两不交、并集 == `specs.ISA` 的全部键（`01` 的 `A.3c`/`A.3d` 钉住）。
#   本程序消费 5 条：warp 级数据搬运（shfl.sync）与屏障（bar, barrier）。
COVERED_ANCHORS = ("P1", "P2", "B1", "B2", "B3")

# ===========================================================================
# 0. 断言记账（每个编号程序自带一份，**不共享**）
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
# 1. PTX 片段（★ 手写的，语法**未经任何工具验证**）
# ===========================================================================
# ★ 这四个常量**被 `01_virtual_isa.py` 用 `ast` 读取**（不 import 本模块）：
#   `SHFL_BODY` / `BAR_BODY` 用来算「`.reg` 声明量 vs 实际触及量」。
#   ⇒ 它们必须是**纯字符串字面量**（不能写成 `.replace(...)` 的结果）。

SHFL_BODY = """
.version 6.0
.target sm_70
.reg .b32 %r<8>;
.reg .pred %p<4>;
.shared .align 4 .b8 smem[4096];

    // %r1 由宿主预置：本 lane 的初值
    mov.u32  %r2, %r1;                            // acc = v
    shfl.sync.bfly.b32 %r3, %r2, 1,  31, 0xffffffff;
    add.u32  %r2, %r2, %r3;
    shfl.sync.bfly.b32 %r3, %r2, 2,  31, 0xffffffff;
    add.u32  %r2, %r2, %r3;
    shfl.sync.bfly.b32 %r3, %r2, 4,  31, 0xffffffff;
    add.u32  %r2, %r2, %r3;
    shfl.sync.bfly.b32 %r3, %r2, 8,  31, 0xffffffff;
    add.u32  %r2, %r2, %r3;
    shfl.sync.bfly.b32 %r3, %r2, 16, 31, 0xffffffff;
    add.u32  %r2, %r2, %r3;
    mov.u32  %r4, %tid.x;
    mul.lo.u32 %r4, %r4, 4;                       // 字节地址
    st.shared.u32 [%r4], %r2;
    ret;
"""

# ★★ C.2 的**可证伪性**实验：把某一轮的异或步长改错，A 类的对拍**必须**失败。
#    两个变异体，**结论相反**，两个都要断言：
#      · `SHFL_BODY_BADSTEP`（8 → 3）：**结果错** ⇒ 对拍必须抓到；
#      · `SHFL_BODY_EQSTEP`（8 → 9）：**结果竟然还是对的** —— 这是一个**等价变异**，
#        不是断言没抓住。代数：前 3 轮（步长 1/2/4）之后每个 lane 持有的是
#        「与它同处一个 8-lane 块」的局部和 S[l>>3]；步长 9 = 0b1001 翻转 bit3 与 bit0，
#        于是 `l^9` 的块号正好是 `(l>>3)^1` ⇒ 该轮把相邻两块配对 →
#        S[q]+S[q^1]；最后步长 16 再把「q 与 q^2」配对 ⇒ **恰好还是总和**。
#      ⇒ 「改了它，断言没叫」有两种原因：**断言弱**，或者**变异等价**。
#        分不清这两者，就会把等价变异当成漏网（P5 的反面）。
SHFL_BODY_BADSTEP = SHFL_BODY.replace(
    "shfl.sync.bfly.b32 %r3, %r2, 8,  31, 0xffffffff;",
    "shfl.sync.bfly.b32 %r3, %r2, 3,  31, 0xffffffff;")
SHFL_BODY_EQSTEP = SHFL_BODY.replace(
    "shfl.sync.bfly.b32 %r3, %r2, 8,  31, 0xffffffff;",
    "shfl.sync.bfly.b32 %r3, %r2, 9,  31, 0xffffffff;")

BAR_BODY = """
.version 6.0
.target sm_70
.reg .b32 %r<8>;
.reg .pred %p<4>;
.shared .align 4 .b8 smem[4096];

    mov.u32  %r0, %tid.x;
    mul.lo.u32 %r1, %r0, 4;            // 区域 A 的字节地址
    st.shared.u32 [%r1], %r0;          // 阶段 1：写入自己的 tid
    {{BAR}}
    add.u32  %r2, %r0, 32;
    and.b32  %r2, %r2, 63;             // (tid + 32) % 64
    mul.lo.u32 %r3, %r2, 4;
    ld.shared.u32 %r4, [%r3];          // 阶段 2：读邻居（**跨 warp**）
    add.u32  %r5, %r1, 256;            // 区域 B
    st.shared.u32 [%r5], %r4;          // 结果存回
    ret;
"""

BAR_NONE = BAR_BODY.replace("    {{BAR}}", "    // 这里故意不放 bar.sync")
BAR_SYNC = BAR_BODY.replace("    {{BAR}}", "    bar.sync 0;")            # 省略 count
BAR_SYNC_CNT = BAR_BODY.replace("    {{BAR}}", "    bar.sync 0, 64;")   # 显式 count=64

# 只有一个 lane 执行 bar.sync（用来验证「到达按 **warp** 粒度记」）
BAR_ONE_LANE = """
.version 6.0
.target sm_70
.reg .b32 %r<4>;
.reg .pred %p<4>;
.shared .align 4 .b8 smem[256];

    mov.u32  %r0, %tid.x;
    setp.lt.s32 %p0, %r0, 1;           // 只有 lane0 的 p0 为真
    @%p0 bar.sync 0, 32;               // ★ 1 个活跃 lane 执行，count 只有 32
    ret;
"""

BRANCH_BODY = """
.version 6.0
.target sm_70
.reg .b32 %r<8>;
.reg .pred %p<4>;
.shared .align 4 .b8 smem[4096];

    mov.u32  %r0, %tid.x;
    mov.u32  %r7, %r0;
    mul.lo.u32 %r7, %r7, 4;
    ld.shared.u32 %r1, [%r7];          // %r1 = 输入数据
    setp.lt.s32 %p0, %r1, 0;           // p0 = (data < 0)
    @%p0 bra L_NEG;
    add.u32  %r2, %r1, 1000;           // then: 非负数 + 1000
    bra      L_END;
L_NEG:
    sub.u32  %r2, 0, %r1;              // else: 取绝对值
L_END:
    st.shared.u32 [%r7], %r2;
    ret;
"""

PREDICATED_BODY = """
.version 6.0
.target sm_70
.reg .b32 %r<8>;
.reg .pred %p<4>;
.shared .align 4 .b8 smem[4096];

    mov.u32  %r0, %tid.x;
    mov.u32  %r7, %r0;
    mul.lo.u32 %r7, %r7, 4;
    ld.shared.u32 %r1, [%r7];
    setp.lt.s32 %p0, %r1, 0;
    @%p0  sub.u32 %r2, 0, %r1;         // 谓词执行，**无跳转**
    @!%p0 add.u32 %r2, %r1, 1000;
    st.shared.u32 [%r7], %r2;
    ret;
"""

# clamp / segmask 探针：`%r1 = laneid`，shfl 之后 `%r2` 就是**源 lane 的编号**。
CLAMP_BODY = """
.version 6.0
.target sm_70
.reg .b32 %r<7>;
.shared .align 4 .b8 smem[1024];

    mov.u32  %r1, %laneid;
    shfl.sync.bfly.b32 %r2, %r1, %r5, %r6, 0xffffffff;
    ret;
"""

# 位搬运检查：载荷在 `%r4`（**宿主预置，PTX 里不写它**），`%r2` 拿到的就是
# **源 lane 那个字的位模式**。★ 用独立寄存器是有原因的：第一版让载荷待在 `%r1`，
# 而片段里那句 `mov.u32 %r1, %laneid;` 会**覆盖**它 —— 于是测的成了 laneid。
BITS_BODY = """
.version 6.0
.target sm_70
.reg .b32 %r<7>;
.shared .align 4 .b8 smem[1024];

    shfl.sync.bfly.b32 %r2, %r4, %r5, %r6, 0xffffffff;
    ret;
"""


# ===========================================================================
# 2. 独立参照实现（A 类：三条互不重叠的路径）
# ===========================================================================
def bfly_reduce_pure_python(vals, nthreads):
    """**路径 ②**：纯 Python 位运算的 butterfly 归约（不用解释器、不用 numpy）。

    与 `ptxsub` 的唯一共同点是**算法**：每一轮 `acc[l] += acc[l ^ step]`。
    实现上没有任何共享代码（没有 `Sim`、没有寄存器文件、没有 shared memory）。
    """
    out = []
    for w in range(nthreads // 32):
        acc = [int(vals[32 * w + l]) for l in range(32)]
        for step in (1, 2, 4, 8, 16):
            acc = [acc[l] + acc[l ^ step] for l in range(32)]
        out += acc
    return np.array(out, dtype=np.int64)


def bfly_src_lane(lane, bval, cval):
    """**路径 ③**：闭式逐 lane 映射（`9.7.10.6` 的伪代码，**逐字照抄原文**）。

    ★★ 注意这里是**原文的式子**：`.up` 用的是 **`maxLane`**。
       本程序用它来**展示分歧**（`C.3`），而**不是**用它当被测实现 ——
       被测实现是 `ptxsub`（它用 `minLane`，见 `specs.UP_MODE_UNDECIDED`）。
    """
    clamp = cval & 0x1F
    segmask = (cval >> 8) & 0x1F
    maxLane = (lane & segmask) | (clamp & ~segmask & 0x1F)
    minLane = (lane & segmask)
    j = lane ^ (bval & 0x1F)
    return j if j <= maxLane else lane, maxLane, minLane


def up_src_lane_isa(lane, bval, cval):
    """`.up` 按 **ISA 原文**（`maxLane`）算出的源 lane。"""
    clamp = cval & 0x1F
    segmask = (cval >> 8) & 0x1F
    maxLane = (lane & segmask) | (clamp & ~segmask & 0x1F)
    j = lane - (bval & 0x1F)
    return j if j >= maxLane else lane


def up_src_lane_impl(lane, bval, cval):
    """`.up` 按 **本库实现**（`minLane`）算出的源 lane。"""
    segmask = (cval >> 8) & 0x1F
    minLane = (lane & segmask)
    j = lane - (bval & 0x1F)
    return j if j >= minLane else lane


# ===========================================================================
# 3. 跑一个片段的小工具
# ===========================================================================
def run_ptx(src, nthreads, fill=0, smem_words=4096, preset=None, trace=None):
    sim = ptxsub.Sim(ptxsub.parse(src), nthreads=nthreads,
                     smem_words=smem_words, fill=fill)
    if preset:
        preset(sim)
    if trace is not None:
        sim.trace_hook = lambda w, pc, op, n: trace.append((pc, op, n))
    sim.run()
    return sim


# ===========================================================================
# 4. A 类
# ===========================================================================
def section_A():
    print("=" * 78)
    print("A 类：算法正确性 —— 三条**互不重叠**的参照路径")
    print("=" * 78)

    rng = np.random.default_rng(20260920)
    nthreads = 128
    vals = rng.integers(-1000, 1000, size=nthreads).astype(np.int64)

    def preset(sim):
        for w in range(sim.nwarp):
            for l in range(32):
                sim.regs[w][l][1] = ptxsub.u32(int(vals[32 * w + l]))

    sim = run_ptx(SHFL_BODY, nthreads, preset=preset)
    got = np.array([sim.smem_i32(t) for t in range(nthreads)], dtype=np.int64)

    # ---- A.1 路径 ①：numpy 的分段求和 ----
    want = np.repeat(vals.reshape(-1, 32).sum(axis=1), 32)
    eq("A", "A.1", int(np.count_nonzero(got != want)), 0,
       "★ 解释器 vs numpy `reshape(-1,32).sum(axis=1)` 逐元素对拍：不匹配 %d 个"
       "（array_equal = %s）" % (int(np.count_nonzero(got != want)), bool(np.array_equal(got, want))))
    # numpy 的 `sum` 走的是它的 pairwise/累加实现，与逐 lane 的加法**不是同一条路径**
    check("A", "A.1b", len(set(got[:32].tolist())) == 1,
          "★ warp 内 32 个 lane 全都拿到同一个和（butterfly 的性质）：%s" % got[0])

    # ---- A.2 路径 ②：纯 Python 位运算 ----
    ref2 = bfly_reduce_pure_python(vals, nthreads)
    eq("A", "A.2", int(np.count_nonzero(got != ref2)), 0,
       "★ 解释器 vs **纯 Python 位运算**参照：不匹配 %d 个" % int(np.count_nonzero(got != ref2)))

    # ---- A.3 路径 ③：闭式 clamp 映射（三档）----
    ok = True
    detail = []
    for cval in (31, 15, 7):
        s = ptxsub.Sim(ptxsub.parse(CLAMP_BODY), nthreads=32, smem_words=256, fill=0)
        for l in range(32):
            s.regs[0][l][5] = ptxsub.u32(8)      # bval = 8
            s.regs[0][l][6] = ptxsub.u32(cval)
        s.run()
        eng = [s.regs[0][l][2] for l in range(32)]
        ref = [bfly_src_lane(l, 8, cval)[0] for l in range(32)]
        detail.append("cval=%2d 一致=%s（返回自己的 lane %d 个）"
                      % (cval, eng == ref, sum(1 for l in range(32) if eng[l] == l)))
        ok = ok and eng == ref
    check("A", "A.3", ok,
          "★ clamp 三档（31/15/7）的逐 lane 源映射：解释器 vs 闭式公式 —— " + " | ".join(detail))

    # ---- A.4 up / down / idx 的逐 lane 映射（独立逐 lane 模型）----
    def engine_map(mode, bval, cval):
        src = {
            "up": ".reg .b32 %r<7>;\n    mov.u32 %r1, %laneid;\n    "
                  "shfl.sync.up.b32 %r2, %r1, %r5, %r6, 0xffffffff;\n    ret;\n",
            "down": ".reg .b32 %r<7>;\n    mov.u32 %r1, %laneid;\n    "
                    "shfl.sync.down.b32 %r2, %r1, %r5, %r6, 0xffffffff;\n    ret;\n",
            "idx": ".reg .b32 %r<7>;\n    mov.u32 %r1, %laneid;\n    "
                   "shfl.sync.idx.b32 %r2, %r1, %r5, %r6, 0xffffffff;\n    ret;\n",
        }[mode]
        s = ptxsub.Sim(ptxsub.parse(src), nthreads=32, smem_words=64, fill=0)
        for l in range(32):
            s.regs[0][l][5] = ptxsub.u32(bval)
            s.regs[0][l][6] = ptxsub.u32(cval)
        s.run()
        return [s.regs[0][l][2] for l in range(32)]

    def ref_map(mode, bval, cval):
        clamp = cval & 0x1F
        seg = (cval >> 8) & 0x1F
        out = []
        for l in range(32):
            maxLane = (l & seg) | (clamp & ~seg & 0x1F)
            minLane = (l & seg)
            if mode == "up":
                j, ok_ = l - (bval & 0x1F), None
                ok_ = j >= minLane                      # ★ 本库实现（有意偏离原文）
            elif mode == "down":
                j = l + (bval & 0x1F)
                ok_ = j <= maxLane
            else:
                j = minLane | (bval & 0x1F & ~seg & 0x1F)
                ok_ = j <= maxLane
            out.append(j if ok_ else l)
        return out

    ok, detail = True, []
    for mode, bval, cval in (("up", 4, 31), ("down", 4, 31), ("idx", 3, 31),
                             ("up", 1, 0), ("down", 1, 0x1F), ("idx", 5, 0x041F)):
        a = engine_map(mode, bval, cval)
        b = ref_map(mode, bval, cval)
        detail.append("%s(b=%d,c=0x%04X):%s" % (mode, bval, cval, "OK" if a == b else "**不一致**"))
        ok = ok and a == b
    check("A", "A.4", ok, "★ 三种模式的逐 lane 源映射（6 组参数）：解释器 vs 独立模型 —— "
                          + " ".join(detail))

    # ---- A.5 shfl 是**位搬运**，与类型无关 ----
    bits = [0x3F800000, 0x40000000]        # 1.0f / 2.0f
    s = ptxsub.Sim(ptxsub.parse(BITS_BODY), nthreads=32, smem_words=256, fill=0)
    for l in range(32):
        s.regs[0][l][4] = bits[l & 1]      # 载荷在 %r4（PTX 里不写它）
        s.regs[0][l][5] = ptxsub.u32(1)    # bval = 1
        s.regs[0][l][6] = ptxsub.u32(31)   # cval = 31
    s.run()
    eq("A", "A.5", [ptxsub.u32(s.regs[0][l][2]) for l in range(2)], [0x40000000, 0x3F800000],
       "★ `shfl.sync.bfly.b32` 搬的是 **32 位位模式**：偶数 lane 输入 1.0f，输出拿到 2.0f 的位模式")
    return got, want, sim


# ===========================================================================
# 5. B 类
# ===========================================================================
def section_B():
    print()
    print("=" * 78)
    print("B 类：代价模型 —— ★ 反直觉量化结论：同结果，**两种写法的发射数不同**")
    print("=" * 78)

    nthreads = 32
    data = [-5] * 16 + [5] * 16
    expected = [5] * 16 + [1005] * 16

    def run_variant(src, trace=None):
        s = ptxsub.Sim(ptxsub.parse(src), nthreads=nthreads, smem_words=4096, fill=0)
        for t, v in enumerate(data):
            s.smem[t] = ptxsub.u32(v)
        if trace is not None:
            s.trace_hook = lambda w, pc, op, n: trace.append((pc, op, n))
        s.run()
        return s, [s.smem_i32(t) for t in range(nthreads)]

    tr_b, tr_p = [], []
    sb, ob = run_variant(BRANCH_BODY, tr_b)
    sp, op = run_variant(PREDICATED_BODY, tr_p)

    eq("B", "B.1", ob, expected, "★ BRANCH 变体的输出 == 期望（16 个 5 + 16 个 1005）")
    eq("B", "B.1b", op, expected, "★ PREDICATED 变体的输出 == 期望（同一个结果）")
    eq("B", "B.1c", ob, op, "★ **两种写法输出逐元素相同** —— 结果一样")

    tb, tp = sb.totals(), sp.totals()
    print()
    print("  %-26s %8s %12s %10s" % ("", "issue", "lane_issue", "divergent"))
    print("  %-26s %8d %12d %10d" % ("BRANCH（真分支）", tb["issue"],
                                     tb["lane_issue"], tb["divergent_steps"]))
    print("  %-26s %8d %12d %10d" % ("PREDICATED（谓词化）", tp["issue"],
                                     tp["lane_issue"], tp["divergent_steps"]))
    print()

    eq("B", "B.2", (tb["issue"], tb["lane_issue"], tb["divergent_steps"]), (11, 304, 3),
       "★ BRANCH 三元组")
    eq("B", "B.2b", (tp["issue"], tp["lane_issue"], tp["divergent_steps"]), (9, 288, 0),
       "★ PREDICATED 三元组")
    eq("B", "B.3", (tb["issue"] - tp["issue"], tb["lane_issue"] - tp["lane_issue"]), (2, 16),
       "★★ 同输入同结果：**发射数差 2、活跃 lane-指令数差 16**")
    eq("B", "B.3b", (tb["divergent_steps"], tp["divergent_steps"]), (3, 0),
       "★ BRANCH 有 3 步处于发散态（PC 不唯一），PREDICATED 恒为 0")

    # B.4：逐发射 trace —— 姿态与手算一致
    eq("B", "B.4", [n for _, _, n in tr_b], [32] * 5 + [16] * 4 + [32, 32],
       "★ BRANCH 的 11 次发射逐条命中（前 5 条满 32 lane，中间 4 条 16 lane，末 2 条满）")
    eq("B", "B.4b", len(tr_p), 9, "★ PREDICATED 共 9 次发射（没有 bra，所以少 2 次）")
    eq("B", "B.4c", [pc for pc, _, _ in tr_b], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
       "★ BRANCH 的 pc 序列**单调递增**：重聚是 **PC 相等自然发生的**"
       "（`bra L_END` 让两条路径走到同一个 pc=9）——不需要显式 re-convergence 栈")

    # B.5：bar.sync 的到达计数按 **warp 粒度**
    def run_bar(src):
        sim = ptxsub.Sim(ptxsub.parse(src), nthreads=64, smem_words=4096, fill=-1)
        sim.run()
        return sim, [sim.smem_i32(64 + t) for t in range(64)]

    s_yes, b_yes = run_bar(BAR_SYNC)
    s_cnt, b_cnt = run_bar(BAR_SYNC_CNT)
    exp64 = [(t + 32) % 64 for t in range(64)]
    eq("B", "B.5", s_yes.bar_arrivals, [(0, 0, 32), (1, 0, 32)],
       "★ 两个 warp 各到达一次，**每次 +32**（不是 +活跃 lane 数）")
    eq("B", "B.5b", b_cnt, b_yes, "★ `bar.sync 0`（省略 count）与 `bar.sync 0, 64` 结果相同")
    eq("B", "B.5c", b_yes, exp64, "★ 有 bar.sync 时 0 个错误元素")

    # ★ B.6：只有 1 个活跃 lane 执行 bar.sync，**整 warp 仍然算到达**
    #   —— 这是 ISA `9.7.15.1`（`B1`/`B2`）那句「marks warps' arrival」的直接后果：
    #      若改成「按活跃 lane 计数」，count=32 时只有 1 个 lane 到达 ⇒ **死锁**。
    one = ptxsub.Sim(ptxsub.parse(BAR_ONE_LANE), nthreads=32, smem_words=64, fill=0)
    one.run()
    eq("B", "B.6", one.bar_arrivals, [(0, 0, 1)],
       "★ 抵达时**活跃 lane = 1**，而 `bar.sync 0, 32` 仍然放行 ——"
       " 到达量按 **warp** 记（+32），不是按活跃 lane 记（那是 1 ⇒ 会死锁）")
    print("      ★ 这条是**可证伪**的：把 `bar_count[bid] += 32` 改成 `+= len(act)`，")
    print("        本断言立刻失败（`meta_falsify.py` 的 D2 走的是另一个方向：整条不阻塞）。")

    # B.7：ISA 的 `.up` 例子用的是 cval=0（从引文里**解析**出来，不手写）
    m = re.search(r"shfl\.sync\.up\.b32 Ry\|p, Rx, 0x([0-9a-fA-F]+), 0x([0-9a-fA-F]+),",
                  specs.ISA["P2"]["quote"])
    check("B", "B.7", bool(m), "★ 从 `specs.ISA['P2']['quote']` 里解析出 bval/cval：%s"
          % (str(m.groups()) if m else "（没解析出来）",))
    eq("B", "B.7b", int(m.group(1), 16) if m else None, 1, "★ bval = 0x1（偏移 1）")
    eq("B", "B.7c", int(m.group(2), 16) if m else None, 0,
       "★★ cval = **0x0** —— ISA 自己给 `.up` 的例子用的是 clamp=0、segmask=0")
    return sb, sp


# ===========================================================================
# 6. C 类
# ===========================================================================
def section_C(got, want):
    print()
    print("=" * 78)
    print("C 类：边界与反例 —— **必须真的跑出问题**")
    print("=" * 78)

    # ---- C.1 缺 bar.sync ⇒ 32/64 个元素错，且错的那一半读到的**全是哨兵 -1** ----
    s_no = ptxsub.Sim(ptxsub.parse(BAR_NONE), nthreads=64, smem_words=4096, fill=-1)
    s_no.run()
    b_no = [s_no.smem_i32(64 + t) for t in range(64)]
    exp64 = [(t + 32) % 64 for t in range(64)]
    bad = [t for t in range(64) if b_no[t] != exp64[t]]
    eq("C", "C.1", len(bad), 32, "★ 缺 `bar.sync` 时错误元素 **32/64** 个")
    eq("C", "C.1b", bad, list(range(32)),
       "★ 错的**恰好是 t < 32 的那一半**（warp0 在 warp1 写入前就跑完了阶段 2）")
    eq("C", "C.1c", sorted({b_no[t] for t in bad}), [-1],
       "★★ 错值**全是初始哨兵 -1** —— 这是「读到了尚未写入的共享内存」的**指纹**，"
       "不是随机的数值错误（本模型的产物：逐 warp 推进到阻塞点）")
    # ★ 必须带限定语：这是**本模型**的产物
    print("      ★ **不是硬件**：真实 SM 上 warp0 未必能一路跑完阶段 2；"
          "「错 32 个」来自本解释器「逐 warp 推进到阻塞点」的调度模型。")

    # ---- C.2 ★★ 可证伪性判据：两个变异体，结论相反 ----
    def mutant_nbad(body):
        s = run_ptx(body, 128, preset=_preset_from(got))
        arr = np.array([s.smem_i32(t) for t in range(128)], dtype=np.int64)
        return int(np.count_nonzero(arr != want))

    n_badstep = mutant_nbad(SHFL_BODY_BADSTEP)
    n_eqstep = mutant_nbad(SHFL_BODY_EQSTEP)
    eq("C", "C.2", n_badstep, 128,
       "★★ **可证伪性**：把 `bfly` 的**异或步长 8 改成 3** ⇒ 与 numpy 参照不匹配 **%d/128** 个"
       "（A.1 的对拍抓得住这个改动 ⇒ 它**不是**「只演示、不检查」）" % n_badstep)
    eq("C", "C.2b", n_eqstep, 0,
       "★★ **等价变异**：同一个位置把 8 改成 **9** ⇒ 不匹配 **%d/128** 个 —— **结果是错的吗？"
       "不是，是对的**（9 = 0b1001：前 3 轮已把 warp 切成 8-lane 块，"
       "翻转 bit3 恰好让「相邻块」配对，最后一步再配 q 与 q^2）" % n_eqstep)
    check("C", "C.2c", n_badstep != n_eqstep,
          "★★ 两个变异体**结论相反**（%d vs %d）⇒「改了它断言没叫」有两种原因："
          "**断言弱** 或 **变异等价**；分不清这两者，就会把等价变异记成漏网"
          % (n_badstep, n_eqstep))

    # ---- C.3 ★★ `.up` 的判据【未定案】：两式在常见参数下**结论相反** ----
    lane, bval, cval = specs.UP_MODE_UNDECIDED["divergent_case"]
    j_isa = up_src_lane_isa(lane, bval, cval)
    j_imp = up_src_lane_impl(lane, bval, cval)
    eq("C", "C.3", (j_isa, j_imp), (lane, 15),
       "★★ lane=%d, bval=%d, cval=%d ⇒ j=%d；ISA 原文（maxLane=%d）给 **%d**（返回自己），"
       "本库实现（minLane=%d）给 **%d** —— **两者结论相反**"
       % (lane, bval, cval, lane - bval,
          specs.UP_MODE_UNDECIDED["divergent_maxLane"], j_isa,
          specs.UP_MODE_UNDECIDED["divergent_minLane"], j_imp))
    eq("C", "C.3b", specs.UP_MODE_UNDECIDED["verdict"], "UNDECIDED",
       "★★ **本库不判定是哪一种** —— 三种可能见 `specs.UP_MODE_UNDECIDED['possibilities']`")
    print("      三种可能（**逐条照抄，一条也不声称**）：")
    for p in specs.UP_MODE_UNDECIDED["possibilities"]:
        print("        · %s" % p)
    print("      ★ 本库**没有** `ptxas`、**没有** GPU ⇒ **无法判定**。"
          "正文**不得**写「规范规定 minLane」「二手转述是错的」「已定案」。")

    # ---- C.4 ★★ 但 `cval = 0` 时**两式逐 lane 恒等**（穷举，不是举例）----
    diff = []
    for seg in range(32):
        cval0 = (seg << 8) | 0x00          # clamp = 0
        for l in range(32):
            if up_src_lane_isa(l, 1, cval0) != up_src_lane_impl(l, 1, cval0):
                diff.append((seg, l))
    eq("C", "C.4", diff, [],
       "★★ `clamp = 0` ⇒ 对**全部 32 个 segmask × 32 个 lane**（1024 组）两式**逐 lane 相等** ——"
       " ISA 自己的 `.up` 例子正是 `cval = 0`（`B.7c`），所以那个例子**无法**用来区分两式")
    # 反向：clamp=31 且 segmask=0 时两式必然分道（这正是 C.3 的参数）
    diff31 = [(l, up_src_lane_isa(l, 1, 31), up_src_lane_impl(l, 1, 31))
              for l in range(32) if up_src_lane_isa(l, 1, 31) != up_src_lane_impl(l, 1, 31)]
    eq("C", "C.4b", len(diff31), 31,
       "★ 换一个参数（clamp=31、segmask=0）⇒ 32 个 lane 里有 %d 个两式分道"
       "（只有 lane 0 例外：j = -1 < 0）" % len(diff31))
    return True


def _preset_from(got):
    """重建 A 类的输入（同一串随机数），供 C.2 的对照实验复用。"""
    rng = np.random.default_rng(20260920)
    vals = rng.integers(-1000, 1000, size=128).astype(np.int64)

    def preset(sim):
        for w in range(sim.nwarp):
            for l in range(32):
                sim.regs[w][l][1] = ptxsub.u32(int(vals[32 * w + l]))
    return preset


# ===========================================================================
# 7. main
# ===========================================================================
def main():
    print("=" * 78)
    print("02_warp_execution.py —— warp 级执行：同一个结果，两种写法，两笔代价")
    print("=" * 78)
    print("  解释器      : %s" % sys.executable)
    print("  python      : %s / numpy %s" % (sys.version.split()[0], np.__version__))
    print("  ★ 限定语 1  : **不是硬件** —— 逐 warp 推进到阻塞点，不是真实 warp 交错调度")
    print("  ★ 限定语 2  : 按 PTX ISA 9.7.10.6（shfl.sync）/ 9.7.15.1（bar, barrier）实现；"
          "**shfl.sync.up 处本库有意偏离 ISA 文本且未定案**")
    print("  ★ 限定语 3  : **手写 PTX 的语法未经任何工具验证** —— 本机无 ptxas / nvcc / nvdisasm")
    print()
    got, want, sim = section_A()
    section_B()
    section_C(got, want)

    static = _static_labels()
    print()
    print("-" * 78)
    # ★ 减掉 `A.6` 自己：这条断言正在被执行，而它的实参在 `check()` 落账**之前**求值。
    eq("A", "A.6", sorted(static - RUNTIME_LABELS - {"A.6"}), [],
       "★ 静态标签减去运行时标签 == 空（静态提取不漏）")
    return summary()


def _static_labels():
    src = open(os.path.join(HERE, "02_warp_execution.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    return {n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and re.match(r"^[ABC]\.\d+[a-z]?$", n.value)}


if __name__ == "__main__":
    sys.exit(main())

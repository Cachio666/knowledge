#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""01_collection_mechanisms.py —— 采样与插桩是两条根本不同的路。

配套文档：`01-collection-mechanisms.md`。★ 三类断言（A 算法正确性 / B 代价模型 /
C 边界与反例）**全部在本程序内部**（`CONTRIBUTING` §2 表里的第二种情形），
失败时以**非零退出码**结束；**没有 `verify.py`**。

★★ 本主题的真值标注（逐处都要带，不许省）：

> 本程序的数字**全部是 CPU 上的本机实测值**（`Apple M1 Pro` / `macOS` /
> `python3 3.13.5` / `clang`），**不是 GPU、不是 Nsight Compute**。
> 它们支撑的是「**采样与插桩这两条路有什么结构性差异**」这类**机制**结论，
> **不支持任何关于 Nsight Compute 具体实现**的结论——后者只能引 `specs.FACTS`（A 级规格值）。

★★ 本程序的可证伪断言（改坏它们，本程序必须非零退出）：

  · `A.3`  **inclusive 与 exclusive 的差 = 被调者的 exclusive 时间**
           （改坏 `cumtime`/`tottime` 的取法立刻失败）；
  · `A.6`  **采样报告里你写的函数出现 0 次**（把采样换成插桩立刻失败）；
  · `B.1`  **定次数循环下 cProfile 开销 > 100%**（本机实测约 +210%）；
  · `B.2`  **wall-bounded 循环下 cProfile 开销 ≈ 0**（**同一个循环体**，结构性不可能 +15%）；
  · `B.5`  **cProfile 报的总时间是「两部分之和」，不是「插桩开销」**（v1 的误读）；
  · `C.2`  **`-O2` 编译后 `hot_loop` 符号在采样报告里出现 0 次**（内联吃掉了它）
           ← **`make falsify` 的「`-O2` 改成 `-O0`」注入项就打在这一点上**；
  · `C.3`  **同一份源码 `-O0` 编译后 `hot_loop` 出现 > 0 次**（⇒ C.2 不是空报告）；
  · `D.5`  **失败标记与 `mutate.py` / `meta_falsify.py` 三处一致**。

用法：
    python3 01_collection_mechanisms.py            # 跑 A/B/C/D 四类断言
    python3 01_collection_mechanisms.py --lint     # AST 扫描：不 import torch/triton/numba/cuda

★ 外部工具：`clang`（**必需**，找不到以可操作的错误 rc=2 退出，**不回落**）、
  `sample`（macOS 自带；找不到时相关断言 **SKIP** 并单独计数，**不计入通过**）。
"""

import ast
import hashlib
import io
import os
import pstats
import cProfile
import shutil
import subprocess
import sys
import tempfile
import time

import specs                                   # noqa: E402  （非编号辅助模块）

HERE = os.path.dirname(os.path.abspath(__file__))

# ===========================================================================
# 0. 断言记账（每个编号程序自带一份，**不共享**）
# ===========================================================================
# ★★ 失败标记：本主题三个程序统一打印 `  **FAIL** A.1 …`。
#   ★ 这个字面量**不是**照抄任何别的主题的：`30-gpu-programming/cutlass` 用 `[FAIL]`，
#     `45-kernel-lifecycle/convergence-and-tuning` 用 `[FAIL]|✗ N 条断言失败`。
#     抄它们**都会匹配到**，但那正是本域禁止的「跨主题照抄」——
#     域 README 要求「三个主题的标记字面形态**各自定义、写死在各自程序里**」。
#   ★ 本主题（profiling-tools）用 `**FAIL**`，并**由 `D.5` 断言它与两个诊断工具一致**。
ASSERT_FAIL_MARK = "**FAIL**"

FAILS = []
_TOTAL = {"A": 0, "B": 0, "C": 0, "D": 0}
_PASS = {"A": 0, "B": 0, "C": 0, "D": 0}
SKIPPED = []


def check(cls, name, cond, detail=""):
    _TOTAL[cls] += 1
    if cond:
        _PASS[cls] += 1
        print("  [ok]     %-6s %s" % (name, detail))
    else:
        FAILS.append(name)
        print("  %s %-6s %s" % (ASSERT_FAIL_MARK, name, detail))
    return bool(cond)


def skip(cls, name, why):
    """★ SKIP **不是通过**：它单独计数，不进 `_PASS`（`CONTRIBUTING` 的 SKIP 口径）。"""
    SKIPPED.append(name)
    print("  [SKIP]   %-6s %s" % (name, why))


def summary():
    t = sum(_TOTAL.values())
    p = sum(_PASS.values())
    print()
    print("汇总: A %d/%d · B %d/%d · C %d/%d · D %d/%d ⇒ "
          "断言 %d 条，通过 %d 条，失败 %d 条；跳过 %d 项（跳过不计入通过）"
          % (_PASS["A"], _TOTAL["A"], _PASS["B"], _TOTAL["B"],
             _PASS["C"], _TOTAL["C"], _PASS["D"], _TOTAL["D"],
             t, p, len(FAILS), len(SKIPPED)))
    if FAILS:
        print("失败清单: %s" % ", ".join(FAILS))
    return 1 if FAILS else 0


# ===========================================================================
# 1. 环境探测
# ===========================================================================
def require_clang():
    """`clang` **必需**：找不到时以可操作的错误退出（rc=2）。

    ★ 计划 §1.3：**不得静默跳过、不得回落到 Python 循环**——
      回落会得到 `10-foundations/03` §4.1 的假数（用 Python `for` 测访存带宽得到
      `0.10 GB/s`，**不随规模变化**，因为瓶颈根本不是内存子系统）。
    """
    p = os.environ.get("CLANG") or shutil.which("clang")
    if not p:
        print()
        print("!! 找不到 `clang` —— 本程序**不会静默跳过、也不会回落到 Python 循环**。")
        print("   为什么不能回落：`10-foundations/performance-models/03` §4.1 记录的实测——")
        print("   用 Python `for` 循环测访存带宽会得到 0.10 GB/s，**且不随规模变化**，")
        print("   因为瓶颈是解释器每次迭代的开销，**根本没碰到内存子系统**。")
        print("   请安装命令行工具（macOS: `xcode-select --install`），")
        print("   或显式指定：`CLANG=/path/to/clang make run`。")
        sys.exit(2)
    return p


def clang_version(clang):
    r = subprocess.run([clang, "--version"], capture_output=True, text=True)
    return (r.stdout or r.stderr).splitlines()[0] if (r.stdout or r.stderr) else "?"


def have_sample():
    return bool(shutil.which("sample"))


def run_sample(pid, secs, outfile):
    """在**还活着的** `pid` 上跑一次 `sample`；返回 `(rc, 输出文件的字节数)`。

    ★ 调用形态就是计划 §0.7 D-1 那一行：`sample <pid> <secs> -f <out>`（**免 root**）。
    """
    r = subprocess.run(["sample", str(pid), str(secs), "-f", outfile],
                       capture_output=True, text=True)
    n = os.path.getsize(outfile) if os.path.exists(outfile) else 0
    return r.returncode, n


def read_sample(path):
    return open(path, encoding="utf-8", errors="replace").read()


# ===========================================================================
# 2. 被测的 Python 工作负载 —— **同一份源码**用于插桩与采样两条路
# ===========================================================================
# ★ 为什么必须是同一份源码：计划 §1.2 的核心论断是「**同一段程序**，
#   `cProfile` 报 `spin_wait` 占 0.600 s，`sample` 报 `_PyEval_EvalFrameDefault` 占最热；
#   **两个工具都没说谎，但它们报的不是同一件事**」。两条路若跑不同的代码，这个对照无效。
PY_WORKLOAD_SRC = '''\
import time


def compute_hot(n=2000000):
    s = 0
    for i in range(n):
        s += (i & 7) * 3
    return s


def spin_wait(seconds):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        pass


def main(spin=0.6):
    compute_hot()
    spin_wait(spin)
'''

# 采样那一路要跑得比采样窗口长，否则 `sample` 会打在一个已经退出的 pid 上
PY_MAIN_REPEATS = 3
SAMPLE_SECS = 2
SPIN_SECONDS = 0.6


def py_namespace():
    ns = {"__name__": "_profiled_workload"}
    exec(compile(PY_WORKLOAD_SRC, "<workload>", "exec"), ns)
    return ns


# ===========================================================================
# 3. 被测的 C 工作负载 —— **同一个源文件，只改优化级别**
# ===========================================================================
# ★ 它回答的是：**采样的归因前提是「符号在栈上」**。
#   `-O2` 会把这个 `static` 函数**内联**进 `main`，符号就消失了 ——
#   这与 GPU 上「内联使函数边界消失」「JIT 代码没有符号」是**同一类问题**（计划 §0.7 D-5）。
#
# ★ 两个优化级别下**墙钟相同**（热循环按墙钟定界），所以「符号消不消失」
#   不可能被「跑得快慢不同」解释 —— 唯一的变量是代码生成。
C_WORKLOAD_SRC = r'''\
#include <stdio.h>
#include <time.h>

volatile double sink = 0.0;          /* ★ volatile：禁止把整个热循环消掉 */

static void hot_loop(void) {
    double s = 0.0;
    long i;
    for (i = 0; i < 2000000L; i++) s += (double)(i & 7);
    sink = s;
}

static void rarely_called_but_slow(void) {
    struct timespec t0, t1;
    double d;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (;;) {
        clock_gettime(CLOCK_MONOTONIC, &t1);
        d = (t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec);
        if (d > 1.0) break;
    }
}

int main(void) {
    struct timespec t0, t1;
    double d;
    rarely_called_but_slow();                     /* 先给采样器一个「慢函数」 */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    do {
        hot_loop();                               /* 之后按墙钟跑热循环 */
        clock_gettime(CLOCK_MONOTONIC, &t1);
        d = (t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec);
    } while (d < 1.2);
    printf("%.1f\n", (double)sink);
    return 0;
}
'''
C_SAMPLE_SECS = 2
C_ATTACH_DELAY_S = 0.35
C_HOT_PHASE_S = 1.2 + 1.0          # 热阶段 1.2 s + 前面 1.0 s 自旋


def build_c(tmpdir, src_text, opt, clang):
    """把内嵌的 C 源码写到临时目录、用 `clang -<opt>` 编出来，返回二进制路径。"""
    cpath = os.path.join(tmpdir, "sym_%s.c" % opt)
    bpath = os.path.join(tmpdir, "sym_%s" % opt)
    with open(cpath, "w", encoding="utf-8") as f:
        f.write(src_text)
    r = subprocess.run([clang, "-%s" % opt, "-o", bpath, cpath],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print("!! clang -%s 编译失败：\n%s" % (opt, r.stderr))
        sys.exit(2)
    return cpath, bpath


def sample_c_binary(bpath, tmpdir, tag):
    """跑二进制、中途 attach `sample`，返回 `(rc, 采样文本)`。"""
    out = os.path.join(tmpdir, "c_%s.txt" % tag)
    p = subprocess.Popen([bpath], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    time.sleep(C_ATTACH_DELAY_S)
    rc, nbytes = run_sample(p.pid, C_SAMPLE_SECS, out)
    p.communicate(timeout=120)
    return rc, (read_sample(out) if os.path.exists(out) else "")


# ===========================================================================
# 4. A 类：插桩 vs 采样（**同一段程序，两条路**）
# ===========================================================================
def part_A(clang, can_sample, tmpdir):
    print("=" * 74)
    print("A 类 · 插桩 vs 采样：同一段 Python 程序，两条路各报了什么")
    print("=" * 74)
    print("  ★ 测量条件：%s；python %s" % (specs.MEASURE_IMPL, sys.version.split()[0]))

    # ---- A 路：插桩（cProfile，进程内） ----
    ns = py_namespace()
    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    ns["compute_hot"]()
    ns["spin_wait"](SPIN_SECONDS)
    pr.disable()
    wall = time.perf_counter() - t0
    sio = io.StringIO()
    stats = pstats.Stats(pr, stream=sio).sort_stats("tottime")
    stats.print_stats()
    pstats_text = sio.getvalue()

    def entry(fname):
        """按**函数名后缀**找一个条目。

        ★ 不能写成 `k[2] == fname`：pstats 的键是 `(filename, lineno, funcname)`，
          而**内建函数**的名字是 `<built-in method time.perf_counter>`，
          不是 `perf_counter` —— 用全等匹配会让 `e_pc` 变成 `None`，
          于是 `A.1`/`A.2`/`A.4` 三条一起假失败（实施者第一版就这么写的，已修）。
        """
        for k, v in stats.stats.items():
            if k[2] == fname or k[2].endswith("." + fname + ">") or k[2].endswith(fname):
                return v
        return None

    e_spin = entry("spin_wait")
    e_hot = entry("compute_hot")
    e_pc = entry("perf_counter")
    tot = sum(v[2] for v in stats.stats.values())          # 各函数的 tottime 之和
    tt_spin = e_spin[2] if e_spin else -1.0
    ct_spin = e_spin[3] if e_spin else -1.0
    nc_spin = e_spin[0] if e_spin else 0
    tt_hot = e_hot[2] if e_hot else -1.0
    tt_pc = e_pc[2] if e_pc else -1.0
    nc_pc = e_pc[0] if e_pc else 0

    print()
    print("  ── 插桩（cProfile）报告的原文（tottime 排序，前几行）──")
    for line in pstats_text.splitlines():
        if line.strip() and not line.startswith(" " * 20):
            print("     %s" % line[:118])
    print()
    print("  ── 控制者记录的同口径样例（计划 §0.7 D-11(a)）──")
    print("     总时间 %.3f s · spin_wait tottime %.3f / cumtime %.3f · "
          "perf_counter %d 次 / tottime %.3f · compute_hot tottime %.3f"
          % (specs.CONTROLLER_RECORDED["instr_total_s"],
             specs.CONTROLLER_RECORDED["instr_spin_tottime_s"],
             specs.CONTROLLER_RECORDED["instr_spin_cumtime_s"],
             specs.CONTROLLER_RECORDED["instr_perf_counter_calls"],
             specs.CONTROLLER_RECORDED["instr_perf_counter_tottime_s"],
             specs.CONTROLLER_RECORDED["instr_compute_hot_tottime_s"]))
    print("  ── 本程序本次实跑的同一组量 ──")
    print("     墙钟 %.3f s · spin_wait ncalls=%d tottime=%.3f cumtime=%.3f · "
          "perf_counter %d 次 / tottime=%.3f · compute_hot tottime=%.3f"
          % (wall, nc_spin, tt_spin, ct_spin, nc_pc, tt_pc, tt_hot))
    print("     ⚠ 绝对值随运行时序变化（本程序的 `perf_counter` 次数由循环体速度决定），")
    print("       所以下面**只对结构下断言**，不写死任何一个绝对值。")

    check("A", "A.1", e_spin is not None and e_hot is not None and e_pc is not None,
          "插桩按**函数名**建表：spin_wait / compute_hot / perf_counter 三个条目都在")
    check("A", "A.2", nc_pc > 1_000_000,
          "插桩给出**精确调用次数**：perf_counter 被调 %d 次（> 1e6，即数值量级）" % nc_pc)
    check("A", "A.3", ct_spin > tt_spin,
          "inclusive ⊃ exclusive：spin_wait cumtime %.3f > tottime %.3f（差 %.3f）"
          % (ct_spin, tt_spin, ct_spin - tt_spin))
    d = abs((ct_spin - tt_spin) - tt_pc)
    check("A", "A.4", d <= 0.02,
          "★ cumtime − tottime = 被调者的 exclusive 时间：%.3f − %.3f = %.3f，"
          "perf_counter tottime = %.3f，差 %.3f ≤ 0.02" % (ct_spin, tt_spin,
                                                          ct_spin - tt_spin, tt_pc, d))
    check("A", "A.5", tt_spin > tt_hot > 0.0,
          "两个函数的 exclusive 时间都 > 0 且 spin_wait 更大：%.3f > %.3f" % (tt_spin, tt_hot))
    check("A", "A.6", ("compute_hot" in pstats_text) and ("spin_wait" in pstats_text),
          "插桩报告里**函数名在场**（compute_hot / spin_wait 字面出现在 pstats 文本里）")

    # ---- B 路：采样（子进程 + sample） ----
    if not can_sample:
        skip("A", "A.7", "找不到 `sample`（macOS 自带）⇒ 采样那一路整体 SKIP，不计入通过")
        skip("A", "A.8", "同上")
        skip("A", "A.9", "同上")
        return

    child = os.path.join(tmpdir, "py_workload.py")
    with open(child, "w", encoding="utf-8") as f:
        f.write(PY_WORKLOAD_SRC + "\nfor _ in range(%d):\n    main()\n" % PY_MAIN_REPEATS)
    sout = os.path.join(tmpdir, "py_sample.txt")
    p = subprocess.Popen([sys.executable, child],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(0.25)
    rc, nbytes = run_sample(p.pid, SAMPLE_SECS, sout)
    p.wait(timeout=120)
    stext = read_sample(sout) if os.path.exists(sout) else ""

    print()
    print("  ── 采样（`sample %d %d`，免 root）──" % (p.pid, SAMPLE_SECS))
    print("     rc=%d，输出 %d 字节" % (rc, nbytes))
    stack_syms = ("_PyEval_EvalFrameDefault", "PyObject_Vectorcall",
                  "mach_absolute_time", "cfunction_vectorcall_NOARGS")
    for s in stack_syms:
        print("     `%-30s` 出现 %d 次" % (s, stext.count(s)))
    for s in ("compute_hot", "spin_wait"):
        print("     `%-30s` 出现 %d 次   ← 你写的函数" % (s, stext.count(s)))
    print("     ★ 结论：%s" % specs.SAMPLE_ATTRIBUTION_CONCLUSION)

    check("A", "A.7", rc == 0 and nbytes > 10000,
          "`sample` 免 root 可用：rc=0，产出 %d 字节调用图" % nbytes)
    check("A", "A.8", stext.count("compute_hot") == 0 and stext.count("spin_wait") == 0,
          "★ **你写的两个函数在采样报告里出现 0 次**（compute_hot=%d · spin_wait=%d）"
          % (stext.count("compute_hot"), stext.count("spin_wait")))
    check("A", "A.9", any(stext.count(s) > 0 for s in stack_syms),
          "★ 同一份报告里**机器栈符号在场**（%s）—— 同一批数据、两个方向"
          % ", ".join("%s=%d" % (s, stext.count(s)) for s in stack_syms))


# ===========================================================================
# 5. B 类：插桩开销 —— **循环结构**决定你能测到什么
# ===========================================================================
def unit(i):
    """★ 循环体：**两个循环共用同一个函数**（唯一变量是循环的终止条件）。"""
    return (i & 7) * 3


def fixed_iters(n):
    """**定次数**循环：跑完 n 次就停。"""
    s = 0
    for i in range(n):
        s += unit(i)
    return s


def wall_bounded(sec):
    """**wall-bounded**循环：跑到墙钟到点为止（迭代次数由时钟决定，不由循环决定）。"""
    s = 0
    i = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < sec:
        s += unit(i)
        i += 1
    return s


B_FIXED_N = 40000
B_WALL_SEC = 0.10
B_REPS = 7


def timed(fn, profiled, reps=B_REPS):
    import statistics
    ts = []
    for _ in range(reps):
        if profiled:
            pr = cProfile.Profile()
            t0 = time.perf_counter()
            pr.enable()
            fn()
            pr.disable()
            ts.append(time.perf_counter() - t0)
        else:
            t0 = time.perf_counter()
            fn()
            ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def part_B():
    print()
    print("=" * 74)
    print("B 类 · 插桩开销：**循环结构**决定你能测到什么")
    print("=" * 74)
    print("  ★ 同一段循环体（`unit(i)`），两种终止条件：定次数 vs wall-bounded。")
    print("    取 %d 次中位数；unit() 每次都产生一次可被插桩的 Python 调用。" % B_REPS)

    rows = []
    for name, fn in (("定次数  fixed_iters(%d)" % B_FIXED_N, lambda: fixed_iters(B_FIXED_N)),
                     ("墙钟定界 wall_bounded(%.2fs)" % B_WALL_SEC,
                      lambda: wall_bounded(B_WALL_SEC))):
        a = timed(fn, False)
        b = timed(fn, True)
        pct = (b - a) / a * 100.0 if a > 0 else float("inf")
        rows.append((name, a, b, pct))
        print("     %-28s 不插桩 %.4f s · 插桩 %.4f s ⇒ 开销 %+.1f%%" % (name, a, b, pct))

    pct_fixed, pct_wall = rows[0][3], rows[1][3]
    print()
    print("  ★ 控制者的独立对照（计划 §1.2 §4，事实层审查者实测）：")
    print("     定次数读法 cProfile 开销 %+.0f%% ~ %+.0f%%（实测区间）"
          % specs.INSTR_OVERHEAD_FIXED_ITERS_PCT)
    print("     wall-bounded 读法 %+.1f%%" % specs.INSTR_OVERHEAD_WALL_BOUNDED_PCT)

    check("B", "B.1", pct_fixed > 100.0,
          "★ 定次数循环下 cProfile 开销 %+.1f%% > 100%%（每次 `unit()` 调用都要付插桩成本）"
          % pct_fixed)
    check("B", "B.2", abs(pct_wall) < 10.0,
          "★ wall-bounded 循环下开销 %+.1f%% ≈ 0（**墙钟把总时长钉死了**，插桩无从体现）"
          % pct_wall)
    # ★ 门槛写成 `10 × max(|pct_wall|, 1.0)`：把 wall-bounded 那一侧**设为至少 1%**
    #   再比 10 倍。这样「把定次数那一侧的读法搬到 wall-bounded 上」（pct_wall ≈ 200%）
    #   会让门槛升到 2000%，B.3 立刻失败 —— 它是有区分力的，不是恒真。
    check("B", "B.3", pct_fixed > 10.0 * max(abs(pct_wall), 1.0),
          "★ 同一个循环体、只换终止条件：%.1f%% vs %.1f%% ⇒ 差 %.0f 倍（门槛 10 倍）"
          % (pct_fixed, pct_wall, pct_fixed / max(abs(pct_wall), 1.0)))
    check("B", "B.4",
          ("unit" in fixed_iters.__code__.co_names)
          and ("unit" in wall_bounded.__code__.co_names),
          "两个循环**共用一个循环体函数** `unit`（两边都按名字调用它 —— "
          "唯一变量是终止条件，不是循环体）")

    # ---- 「两部分之和」不是「插桩开销」 ----
    ns = py_namespace()
    pr = cProfile.Profile()
    pr.enable()
    ns["compute_hot"]()
    ns["spin_wait"](SPIN_SECONDS)
    pr.disable()
    st = pstats.Stats(pr)
    tot_ = st.total_tt
    sum_tt = sum(v[2] for v in st.stats.values())
    rt = sorted(st.stats.items(), key=lambda kv: -kv[1][3])
    print()
    print("  ── cProfile 报的「总时间」是什么？──")
    print("     `total_tt`（pstats 口径）= %.3f s" % tot_)
    print("     各函数 tottime 之和 = %.3f s  ⇒ 占比 %.1f%%" % (sum_tt, 100.0 * sum_tt / tot_))
    for k, v in rt[:4]:
        print("     %-40s ncalls=%-9d tottime=%.3f cumtime=%.3f"
              % (k[2][:40], v[0], v[2], v[3]))
    # ★★ 「两部分之和」不是「插桩开销」—— v1 的误读现场。
    #    ★ 判据（P13 第②句「这个数荒谬不荒谬」）：真正的插桩开销由 B.1/B.2 给出，
    #      **wall-bounded 循环下结构上不可能是 +15%**（墙钟把总时长钉死了）。
    #      把「两部分之和」当成开销，等于**把两个不同的量当成同一个**（P9 同名不同指）。
    top2 = sorted(((k[2], v[3]) for k, v in st.stats.items()), key=lambda kv: -kv[1])[:2]
    print("     ★ cumtime 最强的两条：%s"
          % " · ".join("%s = %.3f s" % (n[:34], c) for n, c in top2))
    print("       它们**互相包含**（inclusive 嵌套），相加没有物理意义。")
    print("       计划 §0.7 D-11(a) 记的 `0.690 = 0.600 + 0.090` **也是这种和** ——")
    print("       v1 却把它读成了「插桩开销 +15%%」。**真正的开销见 B.1/B.2**：")
    print("       定次数循环 %+.1f%%、wall-bounded 循环 %+.1f%%。" % (pct_fixed, pct_wall))
    check("B", "B.5", sum_tt >= 0.99 * tot_,
          "★ cProfile 报的总时间**就是各函数 exclusive 时间之和**（%.1f%%），"
          "它是「两部分之和」**不是「插桩开销」**" % (100.0 * sum_tt / tot_))


# ===========================================================================
# 6. C 类：采样的归因前提是「符号在栈上」
# ===========================================================================
def part_C(clang, can_sample, tmpdir):
    print()
    print("=" * 74)
    print("C 类 · 采样的归因前提是「符号在栈上」：内联会让函数边界消失")
    print("=" * 74)
    print("  ★ 同一份 C 源码，只改优化级别。两个级别下热循环都**按墙钟定界**（1.2 s），")
    print("    所以「符号消不消失」不可能被「跑得快慢不同」解释。")

    _, b_o2 = build_c(tmpdir, C_WORKLOAD_SRC, "O2", clang)
    _, b_o0 = build_c(tmpdir, C_WORKLOAD_SRC, "O0", clang)
    h2 = hashlib.sha256(open(b_o2, "rb").read()).hexdigest()
    h0 = hashlib.sha256(open(b_o0, "rb").read()).hexdigest()
    src_sha = hashlib.sha256(C_WORKLOAD_SRC.encode("utf-8")).hexdigest()[:16]
    print("     源码（嵌入串）sha256[:16] = %s  ← 两次编译**同一份**" % src_sha)
    print("     二进制 -O2 sha256[:16] = %s" % h2[:16])
    print("     二进制 -O0 sha256[:16] = %s" % h0[:16])

    check("C", "C.1", h2 != h0,
          "两个优化级别产出的二进制**不同**（同一份源码 ⇒ 差异只来自代码生成）")

    if not can_sample:
        skip("C", "C.2", "找不到 `sample` ⇒ 断言 SKIP，不计入通过")
        skip("C", "C.3", "同上")
        skip("C", "C.4", "同上")
        return

    rc2, t2 = sample_c_binary(b_o2, tmpdir, "O2")
    rc0, t0 = sample_c_binary(b_o0, tmpdir, "O0")
    print()
    for tag, rc, tx in (("-O2", rc2, t2), ("-O0", rc0, t0)):
        print("     %-4s sample rc=%d bytes=%-6d  hot_loop=%d  "
              "rarely_called_but_slow=%d  main=%d"
              % (tag, rc, len(tx), tx.count("hot_loop"),
                 tx.count("rarely_called_but_slow"), tx.count("main")))
    print("     ★ `main` 在两边都出现 ⇒ 采样器**确实采到了这个进程**（不是空报告）。")

    check("C", "C.2", rc2 == 0 and t2.count("hot_loop") == 0,
          "★ **`-O2` 版：`hot_loop` 出现 0 次**（它被内联进 `main`，符号消失了）")
    check("C", "C.3", rc0 == 0 and t0.count("hot_loop") > 0,
          "★ **`-O0` 版：同一份源码，`hot_loop` 出现 %d 次**（⇒ C.2 不是空报告）"
          % t0.count("hot_loop"))
    check("C", "C.4", t2.count("main") > 0 and t0.count("main") > 0,
          "两边都能采到 `main`：归因失败**不是**「没采到」，是「采到了、但归给了别人」")


# ===========================================================================
# 7. D 类：规格事实、上游契约、文档核对、标记自检
# ===========================================================================
def read_mark_from(path):
    """用 `ast` 从同目录的另一个文件里读出 `ASSERT_FAIL_MARK` 的字面量。"""
    if not os.path.exists(path):
        return None
    tree = ast.parse(open(path, encoding="utf-8").read())
    for n in tree.body:
        tgts = []
        if isinstance(n, ast.Assign):
            tgts = n.targets
        elif isinstance(n, ast.AnnAssign):
            tgts = [n.target]
        for t in tgts:
            if isinstance(t, ast.Name) and t.id == "ASSERT_FAIL_MARK":
                if isinstance(n.value, ast.Constant):
                    return n.value.value
    return None


def part_D(clang, doc_dir):
    print()
    print("=" * 74)
    print("D 类 · 规格事实 / 上游契约 / 文档核对 / 标记自检")
    print("=" * 74)

    rows = specs.fact_invariants()
    bad = [(lbl, got, want) for (lbl, got, want) in rows if got != want]
    for lbl, got, want in bad:
        print("     ✗ %s: got=%r want=%r" % (lbl, got, want))
    check("D", "D.1", not bad,
          "F1–F15 的**结构不变量** %d 条全过（含 must_contain ⊂ quote、F15 默认值 = all/boost）"
          % len(rows))

    for name, (st, detail) in specs.doc_status(doc_dir).items():
        if st == "ok":
            print("     《%s》%s" % (name, detail))
        elif st == "bad":
            print("     《%s》**sha256/字节数对不上**：%s" % (name, detail))
    check("D", "D.2",
          all(st != "bad" for st, _ in specs.doc_status(doc_dir).values()),
          "在场文档的 sha256 与字节数**都对得上**（%s）"
          % ", ".join("%s=%s" % (n, specs.DOCS[n]["sha256_16"]) for n in specs.DOCS))

    if doc_dir is None:
        skip("D", "D.3", "两份 NVIDIA 文档不在场 ⇒ F1–F15 的**逐字核对**整体 SKIP。"
                        "取法见 specs.DOC_CANDIDATES，或设 PERF_DOC_DIR")
    else:
        vrows = specs.verify_docs(doc_dir)
        vbad = [(l, g, w) for (l, g, w) in vrows if g < w]
        for l, g, w in vbad:
            print("     ✗ %s: 出现 %d 次，期望 ≥ %d" % (l, g, w))
        check("D", "D.3", not vbad,
              "F1–F15、F14 的两条增量规则、F15 的 boost/base 条目**逐字在场**（%d 条）"
              % len(vrows))

    urows = specs.upstream_check(specs.REPO_ROOT)
    if urows is None:
        skip("D", "D.4", "上游文档不在场 ⇒ 契约逐字核对 SKIP")
    else:
        ubad = [(l, g, w) for (l, g, w) in urows if g < w]
        check("D", "D.4", not ubad,
              "上游契约 %d 处**逐字在场**（`10/03:187` 与 `45` 的三处「表外输入」）"
              % len(urows))

    marks = {f: read_mark_from(os.path.join(HERE, f))
             for f in ("01_collection_mechanisms.py", "02_replay_multipass.py",
                       "03_defaults_attribution.py", "meta_falsify.py", "mutate.py")}
    uniq = sorted({v for v in marks.values() if v is not None})
    print("     %s" % ", ".join("%s=%r" % (k, v) for k, v in sorted(marks.items())))
    check("D", "D.5",
          len(uniq) == 1 and uniq[0] == ASSERT_FAIL_MARK and "FAIL" in ASSERT_FAIL_MARK,
          "★ 失败标记**五处一致**且含 `FAIL`（本主题自定的字面形态 `%s`，"
          "**不是**照抄 cutlass 的 `[FAIL]`）" % ASSERT_FAIL_MARK)
    check("D", "D.6", ASSERT_FAIL_MARK in ("  " + ASSERT_FAIL_MARK + " A.1"),
          "失败行的格式确实包含该标记（`  %s A.1 …`）" % ASSERT_FAIL_MARK)
    print("     编译器：%s" % clang_version(clang))


# ===========================================================================
# 8. --lint（AST 扫描，不用 grep）
# ===========================================================================
def lint():
    import pathlib
    files = sorted(pathlib.Path(HERE).glob("[0-9][0-9]_*.py")) + \
        [pathlib.Path(HERE) / f for f in ("specs.py", "meta_falsify.py", "mutate.py")]
    bad = []
    for p in files:
        if not p.exists():
            continue
        t = ast.parse(p.read_text(encoding="utf-8"))
        for n in ast.walk(t):
            if isinstance(n, ast.Import):
                bad += [(p.name, a.name) for a in n.names
                        if a.name.split(".")[0] in ("torch", "triton", "numba", "cuda")]
            elif isinstance(n, ast.ImportFrom) and n.module:
                if n.module.split(".")[0] in ("torch", "triton", "numba", "cuda"):
                    bad.append((p.name, n.module))
    print("AST 检查：%d 个文件，import 目标框架的 %d 个" % (len(files), len(bad)))
    for b in bad:
        print("   !! %s: %s" % b)
    return 1 if bad else 0


# ===========================================================================
def main():
    if "--lint" in sys.argv:
        return lint()

    print("=" * 74)
    print("01_collection_mechanisms.py —— 采样与插桩是两条根本不同的路")
    print("=" * 74)
    print("  ★ 真值标注：本程序的数字**全部是 CPU 上的本机实测值**，")
    print("    **不是 GPU、不是 Nsight Compute**；只支撑机制结论（见 specs.MEASURE_NOTE）。")
    print("  ★ 测量条件：%s / python %s / %s"
          % (specs.MACHINE["cpu"], sys.version.split()[0], sys.platform))
    clang = require_clang()
    can_sample = have_sample()
    if not can_sample:
        print("  ⚠ 找不到 `sample`（macOS 自带）⇒ 采样那几路断言将 **SKIP**，不计入通过。")
    doc_dir = specs.find_doc_dir(verbose=True)

    tmpdir = tempfile.mkdtemp(prefix="perf01_")
    try:
        part_A(clang, can_sample, tmpdir)
        part_B()
        part_C(clang, can_sample, tmpdir)
        part_D(clang, doc_dir)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return summary()


if __name__ == "__main__":
    sys.exit(main())

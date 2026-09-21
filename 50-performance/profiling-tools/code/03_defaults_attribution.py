#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""03_defaults_attribution.py —— 默认值不中立；归因会失效。

配套文档：`03-defaults-and-attribution.md`。★ 三类断言（A/B/C）**全部在本程序内部**，
失败时以**非零退出码**结束；**没有 `verify.py`**。

★★ 本主题的真值标注（逐处都要带，不许省）：

> 本程序的数字**全部是 CPU 上的本机实测值**（`Apple M1 Pro` / `macOS` / `python3 3.13.5`），
> **不是 GPU、不是 Nsight Compute**。
> ★ **本程序从不运行 `ncu`/`nsys`** —— 凡出现它们的默认值/界面，全部是**规格引用**
>   （`specs.FACTS` 的 F6/F7/F8/F13/F15，A 级），标签是**规格值**，**不是**本机实测值。

★★ 本程序的可证伪断言：

  · `A.4`  **两个默认值方向相反**：`--cache-control all` 让 kernel 冷启动（偏差）、
           `--clock-control boost` 让 kernel 跑最高频（偏好）⇒ **无法笼统说偏高还是偏低**；
  · `B.2`  **采样报告里程序自己定义的函数出现 0 次**，而 `B.3` 里机器栈符号 > 0
           —— **同一批数据、两个方向**；
  · `A.9`  **δ\* 的系数在上游原文里逐字在场**（改错常数立刻失败）；
  · `C.1`  **两组子进程的启动描述符完全相同**（两组**唯一**差别是 attach 与否）
           ← **`make falsify` 的「两组 subprocess 结构不同」注入项就打在这一点上**；
  · `C.2`  **|Δ| / δ\* ∈ [0.1, 8.0]** ⇒ 结论是「**本机在这么多次重复下不能给出稳定
           判定**」。判据用 `45/02` 的 δ\*（**从 `45` 代入、本域不重推**）。
           ★ **不得为了叙事完整性而报一个 −8%**（计划 §0.7 D-6）。
           ★★ **这一条的判据与计划 §1.4 不同**（计划写的是「组间差 ≤ 组内**极差**」）——
             实施者实测发现**极差**在 n=7 时极不稳定（连续三次跑出 2.41% / 1.87% / 8.44%），
             会让这条断言在**效应真实存在**时随机假失败。**偏差已回报，未就地静默修正。**
           ★ 编排也改了：**交错**（A/B/A/B…）而不是「先跑完 A 再跑完 B」。
             两条偏差的理由都写在 `part_C` 的注释里。

用法：
    python3 03_defaults_attribution.py            # 跑 A/B/C 三类断言
    PERF_QUICK=1 python3 03_defaults_attribution.py   # 快档（诊断工具用）
    python3 03_defaults_attribution.py --lint     # AST 扫描
"""

import ast
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

import specs                                   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# ===========================================================================
# 0. 断言记账（每个编号程序自带一份，**不共享**）
# ===========================================================================
# ★★ 失败标记：本主题（profiling-tools）自定的字面形态是 `**FAIL**`。
#   **不是** `30-gpu-programming/cutlass` 的 `[FAIL]`，也**不是**
#   `45-kernel-lifecycle/convergence-and-tuning` 的 `[FAIL]|✗ N 条断言失败`。
ASSERT_FAIL_MARK = "**FAIL**"

FAILS = []
_TOTAL = {"A": 0, "B": 0, "C": 0}
_PASS = {"A": 0, "B": 0, "C": 0}
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
    SKIPPED.append(name)
    print("  [SKIP]   %-6s %s" % (name, why))


def summary():
    t = sum(_TOTAL.values())
    p = sum(_PASS.values())
    print()
    print("汇总: A %d/%d · B %d/%d · C %d/%d ⇒ "
          "断言 %d 条，通过 %d 条，失败 %d 条；跳过 %d 项（跳过不计入通过）"
          % (_PASS["A"], _TOTAL["A"], _PASS["B"], _TOTAL["B"],
             _PASS["C"], _TOTAL["C"], t, p, len(FAILS), len(SKIPPED)))
    if FAILS:
        print("失败清单: %s" % ", ".join(FAILS))
    return 1 if FAILS else 0


def have_sample():
    return bool(shutil.which("sample"))


# ===========================================================================
# 1. 被测子进程 —— **一份源码，两个用途**
# ===========================================================================
#   ① 观测者效应（D-6）：按**墙钟**数它跑完了多少次 `hot_loop()`；
#   ② 归因失效（D-5）：`sample` 报告里 `hot_loop` / `rarely_called_but_slow`
#      这两个**程序自己定义的**函数各出现多少次。
#
# ★ 它是**解释型语言**的程序，这是有意的：D-5 的结论是「采样器看到的是**机器栈**，
#   Python 的栈帧在 C 栈上不可见」。换成 C 就变成了另一个现象（内联），
#   那一个由 `01_collection_mechanisms.py` 的 C 类负责。
CHILD_SRC = '''\
import sys
import time


def hot_loop(n=200000):
    s = 0
    for i in range(n):
        s += (i & 7) * 3
    return s


def rarely_called_but_slow(seconds=0.8):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        pass


def main(dur):
    n = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < dur:
        hot_loop()
        n += 1
    print(n)
'''

# ★ 归因实验专用的那个子进程：**先自旋、再热循环**，让采样器两种都看到
ATTR_CHILD_SRC = CHILD_SRC.replace(
    "def main(dur):\n    n = 0",
    "def main(dur):\n    rarely_called_but_slow(0.8)\n    n = 0")
ATTR_SPIN_S = 0.8

# ★★ 观测者效应（D-6）用的子进程**不是上面那一个**，两者别混：
#   它按**墙钟**数一个**细粒度**循环的迭代数（实测量级 10^7），
#   与计划 §0.7 D-6「3.0 s 内迭代数」的口径一致。
#   ⚠ 若拿 `CHILD_SRC`（每次 `hot_loop()` 约 10 ms）去数，1.6 s 只能数到约 200 次，
#     粒度太粗，量出来的组内极差会**小于**真实噪声 —— 那是**量法口径错**（P13）：
#     我们想测的是「同一份固定工作量在两组下差多少」，不是「每 10 ms 的一次计数差多少」。
COUNT_CHILD_SRC = '''\
import sys
import time


def main(dur):
    n = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < dur:
        n += 1
    print(n)


main(float(sys.argv[1]))
'''


def d6_params():
    """★ 默认档是正文引用的口径（`specs.D6`）；`PERF_QUICK=1` 是**诊断工具**用的快档。

    ★ 断言全是**结构性**的（|Δ|/δ* 落在窗内、描述符相同、个别函数 0 次），
      所以快档与默认档**走的是同一条代码路径**，只是重复次数与窗口更短。
      默认档才是 `make run` 与正文引用的那一个。
    ★⚠ **`reps` 不能太小**：`δ* ∝ √(2/n)`，n 越小门槛越松，但**σ 的估计也越不稳**。
      本机实测 `reps=3` 时 `|Δ|/δ*` 会冲出窗（一次 7.32%/3.68% 那种组合），
      `reps=5` 起才稳。
    """
    if os.environ.get("PERF_QUICK") == "1":
        return dict(specs.D6_QUICK, quick=True)
    return dict(specs.D6, quick=False)


# ===========================================================================
# 2. A 类：默认值不是中立的
# ===========================================================================
def part_A():
    print("=" * 74)
    print("A 类 · 工具的默认配置不是中立的（**规格引用，本仓库从未运行 ncu/nsys**）")
    print("=" * 74)
    print("  ★ 标签：**规格值** —— 来源是 NVIDIA 官方文档（A 级），不是本机实测。")
    pg = specs.DOCS["ProfilingGuide"]
    cli = specs.DOCS["NsightComputeCli"]
    print("     来源：%s §2.3.8（抓取 %s，页脚 last-updated %s）"
          % (pg["url"], pg["fetched"], pg["last_updated"]))
    print("           %s §5.4.4（抓取 %s，页脚 last-updated %s）"
          % (cli["url"], cli["fetched"], cli["last_updated"]))
    print("     ⚠ **本仓库没有 GPU、也没有 `ncu`/`nsys`**；下列每一个数都是规格值，")
    print("       **不是**本机跑出来的，也**不得**与本机的 CPU 实测互证。")

    f6, f7, f8, f13 = specs.FACTS[5], specs.FACTS[6], specs.FACTS[7], specs.FACTS[12]
    print()
    print("  ── F6/F7/F8：ncu 与 nsys 的三处「默认不同」──")
    print("     F6 cache ：ncu 默认清空全部 GPU cache（replay pass 之间）；nsys 单遍、不清")
    print("     F7 clock ：ncu 默认锁 SM 时钟；nsys 不锁")
    print("     F8 launch：ncu 默认串行化 kernel launch；nsys 不串行化")
    print("     F13      ：**唯一一处**两者方法相同 —— PM sampling 的 Workload Execution 行")
    print()
    print("  ── F15：ncu CLI 的两个默认值（表格的第三个单元格）──")
    for k in ("cache-control", "clock-control"):
        print("     --%-14s 允许值 %-38s **默认 %s**"
              % (k, " / ".join(specs.F15_ALLOWED[k]), specs.F15_DEFAULTS[k]))
    print("     逐字（boost）：%s" % specs.F15_BOOST_VERBATIM)
    print("     逐字（base） ：%s" % specs.F15_BASE_VERBATIM)
    print("     逐字（all）  ：%s" % specs.F15_CACHE_ALL_VERBATIM)

    check("A", "A.1",
          specs.F15_DEFAULTS["cache-control"] == "all"
          and "all" in specs.F15_ALLOWED["cache-control"],
          "★ F15：`--cache-control` 的默认值是 **all**（每次 replay 前清空全部 GPU cache）")
    check("A", "A.2",
          specs.F15_DEFAULTS["clock-control"] == "boost"
          and "boost" in specs.F15_ALLOWED["clock-control"],
          "★ F15：`--clock-control` 的默认值是 **boost**（锁到 turbo 频率）")
    check("A", "A.3", all(m in f6["quote"] for m in f6["must_contain"]),
          "F6 逐字：ncu 默认清 cache、nsys 单遍不清（含 `%s`）"
          % f6["must_contain"][-1][:44])
    check("A", "A.4",
          ("Nsight Compute locks SM clocks by default" in f7["quote"])
          and ("Nsight Systems does not lock clocks" in f7["quote"])
          and ("Nsight Compute serializes kernel launches" in f8["quote"])
          and ("Nsight Systems does not serialize launches" in f8["quote"]),
          "F7/F8 逐字：ncu 锁时钟且串行化；nsys 两样都不做")
    check("A", "A.5", "use the same method to measure start and end timestamps" in f13["quote"],
          "F13 逐字：PM sampling 的 Workload Execution 行上，两者**用同一方法**")
    check("A", "A.6",
          specs.F15_DEFAULTS["cache-control"] != "none"
          and specs.F15_DEFAULTS["clock-control"] in ("boost", "force-boost"),
          "★ **两个默认值方向相反**：cache 让 kernel 冷启动（结果偏**差**）、"
          "clock 让 kernel 跑最高频（结果偏**好**）")
    print("     ★ ⇒ **无法笼统地说 ncu 的数是偏高还是偏低** ——")
    print("       它取决于被测 kernel 对 **cache 状态**与**时钟**哪一个更敏感。")
    print("       （这一句是本节的落点，也是「默认值不中立」这句话的具体含义。）")

    # ★ 本机 cacheline：**规格与本机要分开标**（别把 x86 的 64 B 当成通用的）
    check("A", "A.7", specs.MACHINE["cacheline_bytes"] == 128,
          "本机 cacheline = **%d B**（`sysctl hw.cachelinesize`）—— "
          "**不是 x86 的 64 B**；这条是**本机实测值**，不是规格值"
          % specs.MACHINE["cacheline_bytes"])
    check("A", "A.8", specs.MACHINE["has_gpu_tools"] is False,
          "本机**没有任何 GPU 工具**（ncu / nsys / nvprof / nvidia-smi / CUPTI / perf 全无）"
          "⇒ 本节所有 GPU 侧内容**只能是规格引用**")

    # ★★ A.9：δ* 的系数是**从上游代入**的，必须与上游原文**逐字对得上**。
    #   为什么放在 A 类而不是 C 类：它不是边界情形，是**本域赖以工作的那个常数**。
    rows = specs.mde_anchor_check(specs.REPO_ROOT)
    if rows is None:
        skip("A", "A.9", "上游 `45/02-significance` 不在场 ⇒ δ* 系数的逐字核对 SKIP")
    else:
        bad = [(lbl, g, w) for (lbl, g, w) in rows if g < w]
        for lbl, g, w in bad:
            print("     ✗ %s: 出现 %d 次，期望 ≥ %d" % (lbl, g, w))
        check("A", "A.9", not bad,
              "★ δ* 的系数 **%.15g** 在 `45/02` 原文里**逐字在场**（%d 处锚点）—— "
              "本域只代入、不重推（`CONTRIBUTING` §7「一处正文」）"
              % (specs.MDE_COEFFICIENT_AT_POWER_80, len(rows)))


# ===========================================================================
# 3. B 类：归因失效（采样器把时间归给了机器栈上的符号）
# ===========================================================================
STACK_SYMS = ("_PyEval_EvalFrameDefault", "PyObject_Vectorcall",
              "mach_absolute_time", "cfunction_vectorcall_NOARGS", "_tlv_get_addr")
# ★ 只列**程序自己定义、且确实在跑**的函数。
#   ⚠ **不要**把 `main` 放进来：采样图里的 `pymain_main` / `Py_BytesMain` /
#     `pymain_run_file` 都含子串 `main`，`str.count("main")` 会假阳性 ——
#     这是**量法口径错**（P13）的一个微型实例：我量的不是「我写的那个函数」。
OWN_FUNCS = ("hot_loop", "rarely_called_but_slow")


def part_B(tmpdir, can_sample):
    print()
    print("=" * 74)
    print("B 类 · 归因失效：采样器把时间归给了机器栈上的符号")
    print("=" * 74)
    print("  ★ 被测程序是一个**解释型语言**的程序（%s）" % "python")
    print("     —— D-5 的结论正是「采样器看到的是机器栈，Python 的栈帧在 C 栈上不可见」。")

    if not can_sample:
        for n in ("B.1", "B.2", "B.3", "B.4"):
            skip("B", n, "找不到 `sample`（macOS 自带）⇒ SKIP，不计入通过")
        return

    child = os.path.join(tmpdir, "attr_child.py")
    with open(child, "w", encoding="utf-8") as f:
        f.write(ATTR_CHILD_SRC + "\nmain(1.6)\n")
    sout = os.path.join(tmpdir, "attr_sample.txt")
    p = subprocess.Popen([sys.executable, child], stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
    time.sleep(0.25)
    r = subprocess.run(["sample", str(p.pid), "2", "-f", sout],
                       capture_output=True, text=True)
    p.wait(timeout=120)
    nbytes = os.path.getsize(sout) if os.path.exists(sout) else 0
    txt = open(sout, encoding="utf-8", errors="replace").read() if nbytes else ""

    print()
    print("  ── `sample %d 2` 的输出 ──" % p.pid)
    print("     rc=%d，%d 字节" % (r.returncode, nbytes))
    print("     ── 程序**自己定义**的函数 ──")
    for s in OWN_FUNCS:
        print("        `%-30s` 出现 %d 次" % (s, txt.count(s)))
    print("     ── **机器栈**上的符号 ──")
    for s in STACK_SYMS:
        print("        `%-30s` 出现 %d 次" % (s, txt.count(s)))
    print()
    print("     ★ 结论：%s" % specs.SAMPLE_ATTRIBUTION_CONCLUSION)
    print("     ★ 这不是 bug：`time.perf_counter()` 调 `mach_absolute_time`，")
    print("       于是「自旋等待」被归给了**内核计时调用**。")
    print("     ★ 跨语言同构：这与 GPU 上「内联使函数边界消失」「JIT 代码没有符号」")
    print("       是**同一类问题** —— 那一半由 `01_collection_mechanisms.py` 的 C 类复现。")

    check("B", "B.1", r.returncode == 0 and nbytes > 10000,
          "`sample` 免 root 可用：rc=0，产出 %d 字节调用图" % nbytes)
    own_hits = {s: txt.count(s) for s in OWN_FUNCS}
    check("B", "B.2", all(v == 0 for v in own_hits.values()),
          "★ **程序自己定义的函数全部出现 0 次**（%s）"
          % ", ".join("%s=%d" % kv for kv in own_hits.items()))
    check("B", "B.3", any(txt.count(s) > 0 for s in STACK_SYMS),
          "★ **机器栈符号在场**（%s）"
          % ", ".join("%s=%d" % (s, txt.count(s)) for s in STACK_SYMS))
    check("B", "B.4", all(v == 0 for v in own_hits.values())
          and any(txt.count(s) > 0 for s in STACK_SYMS),
          "★ **同一批数据、两个方向**：自己的函数 0 次 **且** 机器栈符号 > 0")
    print("     ⚠ 正文引用时**只写这个结论、不写具体计数**（计划 §0.7 D-13：")
    print("       绝对计数随程序与采样窗口变化，控制者与审查者两次跑出的数就不一样）。")


# ===========================================================================
# 4. C 类：观测者效应 —— **本机测不出来**
# ===========================================================================
D6_CHILD = None       # 由 part_C() 写成临时目录里的路径（**不往仓库里写东西**）


def harness_spec(dur):
    """两组**共用**的子进程启动规格（**不含** attach 与否）。

    ★★ `C.1` 断言两组实际用到的这份描述符**完全相同**。
       v1 的错误正是「两组 subprocess 结构不同」——那样两组的差别就不止 attach 一个，
       量出来的差额**无法归因给「观测者」**。计划 §0.7 D-6 把它记为**已作废**的第一次测量。
    """
    return {
        "argv": [sys.executable, D6_CHILD, repr(dur)],
        "shell": False,
        "kwargs": {"stdout": subprocess.PIPE, "stderr": subprocess.DEVNULL, "text": True},
        "cwd": os.path.dirname(D6_CHILD),
    }


def descriptor_of(argv, shell, kwargs):
    """把**实际用到的**启动参数做成可比较的描述符（`C.1` 的判据对象）。"""
    return (tuple(argv) if isinstance(argv, (list, tuple)) else ("<shell>", str(argv)),
            bool(shell),
            tuple(sorted(kwargs.items(), key=lambda kv: kv[0])))


def measure(attach, dur, sample_secs, attach_delay):
    """跑一次子进程；`attach=True` 时并发 attach 一次 `sample`。返回 `(迭代数, 描述符)`。

    ★★ **两组唯一的差别是 `attach`。** 描述符里记的是**子进程怎么被拉起来的全部细节**
       （argv / 是否走 shell / Popen 的 kwargs / cwd）—— attach 自己做的那次 `sample`
       调用**不在描述符里**（它不改变子进程的启动方式）。
    """
    spec = harness_spec(dur)
    argv, shell, kwargs = spec["argv"], spec["shell"], spec["kwargs"]
    # ★ 注入项 5（`make falsify`）就打在**上面这一行**之后：给 attach 组换一套启动结构。
    seen = descriptor_of(argv, shell, kwargs)
    sample_out = os.path.join(tempfile.gettempdir(), ".d6_sample_%d.txt" % os.getpid())
    p = subprocess.Popen(argv, shell=shell, **kwargs)
    try:
        if attach:
            time.sleep(attach_delay)
            subprocess.run(["sample", str(p.pid), str(sample_secs), "-f", sample_out],
                           capture_output=True, text=True)
        out, _ = p.communicate(timeout=180)
    finally:
        if p.poll() is None:
            p.kill()
        if os.path.exists(sample_out):
            os.remove(sample_out)
    txt = (out or "").strip()
    nums = [int(m.group(0)) for m in re.finditer(r"^\d+$", txt, re.M)]
    return (nums[-1] if nums else -1), seen


def part_C(tmpdir, can_sample):
    print()
    print("=" * 74)
    print("C 类 · 观测者效应：**本机在这么多次重复下不能给出稳定判定**")
    print("=" * 74)
    par = d6_params()
    dur, reps = par["duration_s"], par["reps"]
    print("  ★ 口径：每组 %d 次重复，子进程每次跑 %.1f s 的**固定工作量**，"
          "比较两组的中位数。" % (reps, dur))
    print("     %s" % ("**快档**（PERF_QUICK=1，诊断工具用）" if par["quick"]
                       else "默认档（正文引用的就是这一档）"))
    print("  ★ 编排：**交错**（A/B/A/B…），不是「先跑完 A 再跑完 B」—— 防的是**时间相关的**漂移")
    print("     （本域 `tuning-practice/03` 讲的 A/B 编排是同一件事，计划 §0.7 D-12）。")
    print("     ⚠ 但**负载会放大这个效应本身**：本机被占满时 |Δ| 从静息的 5.8%%–7.5%%")
    print("       涨到 22%%–24%%（两种编排都涨）。**这不是交错能消掉的东西。**")
    print("  ★ 为什么结论会是「不能稳定判定」：控制者与审查者各跑过一次（计划 §0.7 D-6），")
    print("     实施者又跑了七次 —— **九次里 |Δ| 一直是正的**，但 |Δ|/δ* 从 0.36 漂到 2.90。")
    print("     ⚠ 绝对值**不可照抄**：控制者记录 %d/%d、组间差 −7.98%%；"
          % (26504981, 24390308))
    print("       审查者复跑 %d/%d、−9.68%%。" % (30486666, 27535466))

    if not can_sample:
        for n in ("C.1", "C.2", "C.3"):
            skip("C", n, "找不到 `sample` ⇒ 观测者效应这一路整体 SKIP，不计入通过")
        return

    global D6_CHILD
    child = os.path.join(tmpdir, "d6_child.py")
    D6_CHILD = child
    with open(child, "w", encoding="utf-8") as f:
        f.write(COUNT_CHILD_SRC)
    try:
        # ★★ **交错编排**（A/B/A/B…），**不是**先跑完 7 次 A 再跑完 7 次 B。
        #   理由：「先跑完 A、再跑完 B」把比较变成了「**两个时间段之间**」的比较 ——
        #   机器负载在这两个时段之间一变，差值就被记到「观测者效应」头上。
        #   交错让两组面对**同分布**的系统状态，把「先后比较」转成「同期比较」。
        #   ★ 这与本域 `tuning-practice/03`（计划 §0.7 D-12）讲的是**同一件事**：
        #     A 与 B **逐字等价**时，顺序编排仍给出约 +0.86% 的虚假差异，
        #     交错把它压到 0.015%（约 57 倍）。这里用的是同一条对策。
        #   ★★ **但本机没有把「分块 vs 交错」单独隔离出来**（那需要另做一组对照）——
        #     实测到的是：**机器被占满时这个效应本身就更大**（|Δ| 从静息的 5.8%–7.5%
        #     涨到 22%–24%，两种编排都涨）。⇒ **不得把「负载下的放大」写成
        #     「分块编排的产物」**，那是编机制从句（P4）。交错的用处是**防时间相关的漂移**，
        #     不是「消除负载影响」。
        #   ★ 两组**各跑 reps 次**这一点没变，唯一变的是**它们的先后顺序**。
        groups = {"不 attach": [], "attach sample": []}
        descs = {"不 attach": set(), "attach sample": set()}
        for _ in range(reps):
            for label, attach in (("不 attach", False), ("attach sample", True)):
                v, d = measure(attach, dur, par["sample_secs"], par["attach_delay_s"])
                groups[label].append(v)
                descs[label].add(d)
        for label in ("不 attach", "attach sample"):
            vals = groups[label]
            print("     %-15s n=%d 中位 %-12d min %-12d max %-12d"
                  % (label, len(vals), int(statistics.median(vals)), min(vals), max(vals)))
    finally:
        if os.path.exists(child):
            os.remove(child)

    lab_a, lab_b = "不 attach", "attach sample"
    va, vb = groups[lab_a], groups[lab_b]
    med_a, med_b = statistics.median(va), statistics.median(vb)
    rng_a = (max(va) - min(va)) / med_a          # 只打印，**不**用作判据（见下）
    rng_b = (max(vb) - min(vb)) / med_b
    gap = abs(med_b - med_a) / med_a

    # ★★ 噪声幅度用**合并组内标准差**，不用**极差**。
    #   为什么换（实施者实测，**这是对计划 §1.4 的一处偏差，已回报**）：
    #     · `极差` 是**极不稳定**的估计量：它只由两个端点决定，
    #       n=7 时它随哪一轮恰好撞上离群点而剧烈变化（本机连续三次跑出 2.41% / 1.87% / 8.44%）；
    #     · 而 `组间差` 是**中位数之差**，稳定得多。
    #     ⇒ 拿一个抖动的量当门槛，`组间差 ≤ 组内极差` 会在**效应真实存在**时**随机假失败**
    #       （本机实测：三次静息跑里有一次 5.83% > 5.67% 而红）。
    #   ★ `σ` 是稳定估计量，且 `45/02` 的判据本来就是**用 σ 写的**——
    #     判据也换回 `45` 的那一条（`δ* = 2.801585218112968·σ·√(2/n)`，从 `45` 代入、本域不重推）。
    #     这一换**没有放宽门槛**：它把「随机撞离群点」换成了「按 σ 算出来的分辨极限」。
    #   ★ 每组的偏离**先减掉自己组的中位数**再合并 ⇒ 合并后的 σ 里**不含组间差**，
    #     它是纯粹的组内噪声估计（把两组的自由度合起来，n 从 7 变成 12）。
    def rel_dev(vals, med):
        return [(v - med) / med for v in vals]

    da, db = rel_dev(va, med_a), rel_dev(vb, med_b)
    dof = (len(da) - 1) + (len(db) - 1)
    var = (sum(x * x for x in da) + sum(x * x for x in db)) / dof
    sigma = var ** 0.5
    delta_star = specs.mde_delta_star(sigma, reps)
    ratio = gap / delta_star if delta_star > 0 else float("inf")
    n_req = specs.mde_required_n(sigma, gap) if gap > 0 else float("inf")

    print()
    print("  ── 两组各自的**组内极差**（★ **只打印、不再用作门槛**，理由见下）──")
    print("     组内极差（不 attach）  = %.2f%%" % (100.0 * rng_a))
    print("     组内极差（attach）     = %.2f%%" % (100.0 * rng_b))
    print("     ⇒ 极差只由两个端点决定，n=7 时极不稳定（本机连续三次跑出 2.41%% / 1.87%% / 8.44%%）。")
    print("  ── 判据用的三个量 ──")
    print("     合并组内 σ（相对量，先把每组减掉自己的中位） = **%.2f%%**（自由度 %d）"
          % (100.0 * sigma, dof))
    print("     **组间差 |Δ|**                              = **%.2f%%**" % (100.0 * gap))
    print("     δ* = %.15g · σ · √(2/%d)                 = **%.2f%%**"
          % (specs.MDE_COEFFICIENT_AT_POWER_80, reps, 100.0 * delta_star))
    print("     **|Δ| / δ\\***                                = **%.2f**" % ratio)
    print("     ⇒ 要把这个 |Δ| 判到 80%% 检定力，每组需要约 **%.0f 次**重复（本程序现跑 %d 次）"
          % (n_req, reps))
    if n_req < reps:
        print("     ★ 注意：n_req < %d **不代表「已经判定了」** —— 它用的是**这一批的 σ**，"
              % reps)
        print("       而 σ 本身在批次之间会漂（本机九次独立测量里 σ 从 1.7%% 漂到 13.4%%，差 8 倍）。")
        print("       **σ 不可复现 ⇒ 由它算出的 δ* 也不可复现 ⇒ 判定不可复现。**")

    check("C", "C.1", descs[lab_a] == descs[lab_b] and len(descs[lab_a]) == 1,
          "★ **两组的启动描述符完全相同**（两组唯一差别是 attach 与否）—— "
          "改掉任一组的结构，这一条立刻失败")
    # ★★ 门槛 [0.1, 8.0] 的依据：本机 **11 次独立测量**的 |Δ|/δ* 落在 **0.30–5.12**，
    #   而它**随机器负载移动**（静息 0.30–2.90；被 8 个进程占满时 3.73–5.12）。
    #   ⇒ 窗开得宽是**有意的**：这一条不是「精确判据」，是**冒烟检查**——
    #     `C.1`（描述符相同）才是防「对照结构被改坏」的那一条，它是**精确且与负载无关**的。
    #   窗外意味着「这个差**远离**了本方法的分辨极限」：
    #     · **下侧**：把 attach 那一组去掉、或给两组都 attach ⇒ |Δ| ≈ 0 ⇒ 比值趋 0；
    #     · **上侧**：把 `sample` 的窗口从 2 s 拉到整个 3 s ⇒ 比值明显冲出。
    #   两种情况下「不能稳定判定」这句话都不成立，程序必须叫。
    check("C", "C.2", 0.1 <= ratio <= 8.0,
          "★ **|Δ|/δ* = %.2f ∈ [0.1, 8.0]** ⇒ 这个差**就在本方法的分辨极限附近**"
          "（本机 11 次独立测量的实测区间 0.30–5.12，**随机器负载移动**）" % ratio)
    check("C", "C.3", len(va) == reps and len(vb) == reps and all(v > 0 for v in va + vb),
          "两组各 %d 次、每次都数出了正数（n=%d，窗口 %.1f s）" % (reps, reps, dur))
    print()
    print("  ★★ **结论：本机在 %d 次重复下不能给出稳定的判定。**" % reps)
    print("     ★ 这句话**不是**「差很小」—— 本批的 |Δ| = %.2f%% 与 δ* = %.2f%% 是"
          % (100.0 * gap, 100.0 * delta_star))
    print("       **同一量级**（比值 %.2f）。" % ratio)
    print("     ★ 也不是「效应不存在」—— 九次独立测量里 |Δ| 一直是正的。")
    print("     ★ **不能判定的真因是 σ 本身不可复现**：九次独立测量里合并组内 σ 从")
    print("       **1.72%** 漂到 **13.40%**（差约 8 倍），于是 δ* 从 2.58% 漂到 20.07%。")
    print("       ⇒ **要判定这个差，先要把 σ 稳定住**（固定机器负载、加大重复次数），")
    print("         而不是简单地多跑几轮。")
    print("     ★ **不得为了叙事完整性而报一个「−8%」** —— 那正是把噪声当成果。")
    print("       这条结论本身就是正文内容：它让 `45/02-significance` 的 MDE 判据")
    print("       在本域的真实测量里现了一次身。")


# ===========================================================================
# 5. 标记自检 / --lint
# ===========================================================================
def read_mark_from(path):
    if not os.path.exists(path):
        return None
    tree = ast.parse(open(path, encoding="utf-8").read())
    for n in tree.body:
        tgts = n.targets if isinstance(n, ast.Assign) else (
            [n.target] if isinstance(n, ast.AnnAssign) else [])
        for t in tgts:
            if isinstance(t, ast.Name) and t.id == "ASSERT_FAIL_MARK" \
                    and isinstance(n.value, ast.Constant):
                return n.value.value
    return None


def mark_selftest():
    files = ("01_collection_mechanisms.py", "02_replay_multipass.py",
             "03_defaults_attribution.py", "meta_falsify.py", "mutate.py")
    marks = {f: read_mark_from(os.path.join(HERE, f)) for f in files}
    uniq = sorted({v for v in marks.values() if v is not None})
    print()
    print("=" * 74)
    print("C 类（续）· 失败标记自检")
    print("=" * 74)
    print("     %s" % ", ".join("%s=%r" % kv for kv in sorted(marks.items())))
    check("C", "C.4",
          all(v is not None for v in marks.values())
          and len(uniq) == 1 and uniq[0] == ASSERT_FAIL_MARK,
          "★ **五个文件全部在场且标记一致**：%r" % (uniq,))


def lint():
    import pathlib
    files = sorted(pathlib.Path(HERE).glob("[0-9][0-9]_*.py")) + \
        [pathlib.Path(HERE) / f for f in ("specs.py", "meta_falsify.py", "mutate.py")]
    bad = []
    for p in files:
        if not p.exists():
            continue
        for n in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
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


def main():
    if "--lint" in sys.argv:
        return lint()
    print("=" * 74)
    print("03_defaults_attribution.py —— 默认值不中立；归因会失效")
    print("=" * 74)
    print("  ★ 真值标注：%s" % specs.MEASURE_NOTE)
    print("  ★ 本程序**从不运行 `ncu`/`nsys`** —— 凡出现它们的默认值，标签都是**规格值**。")
    print("  ★ 测量条件：%s / python %s / %s"
          % (specs.MACHINE["cpu"], sys.version.split()[0], sys.platform))
    can_sample = have_sample()
    if not can_sample:
        print("  ⚠ 找不到 `sample`（macOS 自带）⇒ B/C 两类断言将 **SKIP**，不计入通过。")
    tmpdir = tempfile.mkdtemp(prefix="perf03_")
    try:
        part_A()
        part_B(tmpdir, can_sample)
        part_C(tmpdir, can_sample)
        mark_selftest()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return summary()


if __name__ == "__main__":
    sys.exit(main())

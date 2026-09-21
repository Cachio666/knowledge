#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""02_reading_counters.py —— 读计数器有三种系统性误读，**而且都不报错**。

配套文档：`02-reading-counters.md`。★ 三类断言（A/B/C）**全部在本程序内部**，
失败时以**非零退出码**结束；**没有 `verify.py`**。

★★ 本主题的真值标注（逐处都要带，不许省）：

> 本程序的数字**全部是 CPU 上的本机实测值**（`Apple M1 Pro` / `macOS` / `python3 3.13.5`），
> **不是 GPU、不是 Nsight Compute**。
> 本程序**从不运行 `ncu`/`nsys`** —— 凡出现它们的 metric 名与定义，
> 全部是**规格引用**（`specs.FACTS` 的 F10 / F11 / F12 / F14，A 级），
> **不是**本机实测值，**也不得**与本机实测互证。

★★ 本程序的可证伪断言：

  · `A.3`  **采样报告里，程序自己定义的函数各出现 0 次** ← 本主题的核心断言；
  · `A.4`  同一份报告里**机器栈符号 > 0**（⇒ A.3 不是「采样器什么都没采到」）；
  · `A.5`  **「报告提到的符号集合」∩「程序自己定义的函数集合」= 空集**
           （计划 §2.2 写明的断言；两边的**口径都写在 `part_A` 的注释里**）；
  · `A.2`  **耗时最长的那个函数，就是 `A.3` 里出现 0 次的那个** —— 用**显式计时**立起来，
           否则 `A.3` 只证明「采样器没报它」，不证明「它最耗时」；
  · `B.2`  **同一个名字（「耗时」）下有两个计数器**：挂钟时间与 CPU 时间可以差 **> 10 倍**，
           而**两者都正确** —— 这是 F14「同名不同指」在本机的对应物；
  · `C.3`  失败标记与 `01` / `03` / `meta_falsify` / `mutate` 五处一致。

用法：
    python3 02_reading_counters.py            # 跑 A/B/C 三类断言
    python3 02_reading_counters.py --lint     # AST 扫描：不 import torch/triton/numba/cuda

★ 外部工具：**`sample`**（macOS 自带；找不到时 A 类整体 **SKIP** 并单独计数，
  **不计入通过**）。本程序**不需要 `clang`、不需要网络**。
"""

import ast
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import time

import specs                                   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# ===========================================================================
# 0. 断言记账（每个编号程序自带一份，**不共享**）
# ===========================================================================
# ★★ 失败标记：本主题自定的字面形态是 `*** FAIL ***`（**不是**同域 `profiling-tools`
#   的 `**FAIL**`，也**不是** cutlass 的 `[FAIL]` / 45 的 `[FAIL]|✗ N 条断言失败`）。
#   ★★ 判定侧必须【行首锚定】：程序**成功时也会打印这个字面量**（`C.3` 的 detail 里就有）。
ASSERT_FAIL_MARK = "*** FAIL ***"

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


# ===========================================================================
# 1. 被测程序 —— **一份源码，两个用途**
# ===========================================================================
#   ① 子进程：交给 `sample` 去采（A 类）；
#   ② 进程内：用 `perf_counter` 各自计时，确立「谁最耗时」（A.2）。
#
# ★ 它是**解释型语言**的程序，这是有意的：D-5 的结论是「采样器看到的是**机器栈**，
#   Python 的栈帧在 C 栈上不可见」。换成 C 就变成另一个现象（内联），
#   那一个由同域 `profiling-tools` 的 `01-collection-mechanisms.md` §4.3 负责。
#
# ★★ **函数名不要选「会出现在机器栈上的词」** —— 同域 `profiling-tools/profiling-tools`
#   的 `03` 记着这个教训：采样图里的 `pymain_main` / `Py_BytesMain` / `pymain_run_file`
#   都含子串 `main`，`str.count("main")` 会**假阳性**。那是 P13 的一个微型实例：
#   **我量的不是「我写的那个函数」。** 下面两个名字都避开了 `main` / `run` / `start`。
COUNTER_CHILD_SRC = '''\
import time


def busy_accumulate(seconds=0.05):
    """**按墙钟定界**的纯计算循环：每 4096 次加法看一眼时钟。"""
    s = 0
    t0 = time.perf_counter()
    while True:
        for _ in range(4096):
            s += 1
        if time.perf_counter() - t0 >= seconds:
            break
    return s


def slow_spin_probe(seconds=0.30):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        pass


def main(dur=2.4):
    slow_spin_probe(0.30)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < dur:
        busy_accumulate()
'''
OWN_FUNCS = ("busy_accumulate", "slow_spin_probe")

# ★★ **两个函数都按墙钟定界**，这是有意的 —— `A.2` 要断言「谁更耗时」：
#   若其中一个按**迭代次数**定界，机器一忙它的耗时就被别的进程抬起来，比值不再稳定。
#   实施者实测：按次数定界的那一版在 8 个忙循环打满时从 0.08 s 涨到 0.30 s，
#   `A.2` 直接失败（0.30 vs 0.30，比值 1.6）。
#   **两个都按墙钟定界，比值就由参数决定（0.30/0.05 = 6），与负载无关。**
BUSY_SECONDS = 0.05
SPIN_SECONDS = 0.30
CHILD_DUR = 2.4

# ★ 机器栈上「一定有、而你的代码没写」的符号（计数**只写 0 / 非 0**，见 A.5 的口径）。
STACK_SYMS = ("_PyEval_EvalFrameDefault", "PyObject_Vectorcall",
              "mach_absolute_time", "cfunction_vectorcall_NOARGS", "_tlv_get_addr")

SAMPLE_SECS = 2
ATTACH_DELAY_S = 0.25


def child_namespace():
    ns = {"__name__": "_counter_workload"}
    exec(compile(COUNTER_CHILD_SRC, "<workload>", "exec"), ns)
    return ns


def have_sample():
    return bool(shutil.which("sample"))


def read_text(path):
    return open(path, encoding="utf-8", errors="replace").read() if os.path.exists(path) else ""


# ===========================================================================
# 2. A 类：归因失效 —— 采样器把时间归给了机器栈上的符号
# ===========================================================================
def part_A(tmpdir, can_sample):
    print("=" * 74)
    print("A 类 · 归因失效：耗时最长的函数，在采样报告里出现 **0 次**")
    print("=" * 74)
    print("  ★ 测量条件：%s；python %s" % (specs.MEASURE_IMPL, sys.version.split()[0]))
    print("  ★ 被测程序是**解释型语言**的：%s" % "python")

    # ---- A.2 的前半：进程内各自计时，先确立「谁最耗时」 ----
    ns = child_namespace()
    t0 = time.perf_counter()
    ns["slow_spin_probe"](SPIN_SECONDS)
    t_spin = time.perf_counter() - t0
    t0 = time.perf_counter()
    ns["busy_accumulate"](BUSY_SECONDS)
    t_busy = time.perf_counter() - t0
    print("  ── 进程内**显式计时**（确立谁最耗时，不靠采样器）──")
    print("     两个函数**都按墙钟定界**（%.2f s vs %.2f s）⇒ 比值与机器负载无关"
          % (SPIN_SECONDS, BUSY_SECONDS))
    print("     `slow_spin_probe(%.2f)`  %.4f s" % (SPIN_SECONDS, t_spin))
    print("     `busy_accumulate(%.2f)`  %.4f s" % (BUSY_SECONDS, t_busy))

    if not can_sample:
        for n in ("A.1", "A.3", "A.4", "A.5"):
            skip("A", n, "找不到 `sample`（macOS 自带）⇒ A 类整体 SKIP，不计入通过")
        return
    check("A", "A.2", t_spin > 3.0 * t_busy,
          "★ 耗时最长的函数是 `slow_spin_probe`：%.4f s，是 `busy_accumulate` 的 %.1f 倍"
          "（两个都按墙钟定界 ⇒ 这个比值不随机器负载漂）"
          % (t_spin, t_spin / max(t_busy, 1e-9)))

    child = os.path.join(tmpdir, "counter_child.py")
    with open(child, "w", encoding="utf-8") as f:
        f.write(COUNTER_CHILD_SRC + "\nmain(%r)\n" % CHILD_DUR)
    sout = os.path.join(tmpdir, "counter_sample.txt")
    p = subprocess.Popen([sys.executable, child],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(ATTACH_DELAY_S)
    r = subprocess.run(["sample", str(p.pid), str(SAMPLE_SECS), "-f", sout],
                       capture_output=True, text=True)
    p.wait(timeout=120)
    nbytes = os.path.getsize(sout) if os.path.exists(sout) else 0
    txt = read_text(sout)

    print()
    print("  ── `sample %d %d` 的输出 ──" % (p.pid, SAMPLE_SECS))
    print("     rc=%d，%d 字节" % (r.returncode, nbytes))
    print("     ── 程序**自己定义**的函数 ──")
    for s in OWN_FUNCS:
        print("        `%-22s` 出现 %d 次" % (s, txt.count(s)))
    print("     ── **机器栈**上的符号 ──")
    for s in STACK_SYMS:
        print("        `%-30s` 出现 %d 次" % (s, txt.count(s)))

    check("A", "A.1", r.returncode == 0 and nbytes > 10000,
          "`sample` 免 root 可用：rc=0，产出 %d 字节调用图" % nbytes)

    own_hits = {s: txt.count(s) for s in OWN_FUNCS}
    check("A", "A.3", all(v == 0 for v in own_hits.values()),
          "★ **程序自己定义的函数各出现 0 次**（%s）—— 其中 `slow_spin_probe` 就是 A.2 里"
          "耗时最长的那个" % ", ".join("%s=%d" % kv for kv in own_hits.items()))

    stack_hits = {s: txt.count(s) for s in STACK_SYMS}
    n_stack = sum(1 for v in stack_hits.values() if v > 0)
    check("A", "A.4", n_stack > 0,
          "★ 同一份报告里**机器栈符号在场**（%d/%d 个符号 > 0）—— 同一批数据、两个方向"
          % (n_stack, len(STACK_SYMS)))

    # ---- A.5：**符号集合的交集 = 空集**（计划 §2.2 写明的断言） ----
    #   ★★ **口径写在这里**（不写清口径的集合运算就是 P13）：
    #     · `MENTIONED`（「报告提到的符号集合」）：在**下面的候选符号集合**范围内，
    #       报告里出现次数 > 0 的那些。候选集合 = 程序自己定义的函数 ∪ 机器栈符号。
    #       ⚠ 它是**有限候选集合上的**交集判据，**不是**「把报告里所有符号都解析出来」——
    #         后者要解析 `sample` 的文本格式，格式一变判据就失效（且解析本身是新错源）。
    #         有限候选集合让判据**稳定且可复核**，代价是它只覆盖这两个集合。
    #     · `OWN`（「程序自己定义的函数集合」）= `OWN_FUNCS`。
    candidates = set(OWN_FUNCS) | set(STACK_SYMS)
    mentioned = {s for s in candidates if txt.count(s) > 0}
    inter = mentioned & set(OWN_FUNCS)
    print()
    print("  ── 集合口径（★ 不写清口径的集合运算就是 P13）──")
    print("     候选符号集合 = 程序自己定义的函数（%d 个） ∪ 机器栈符号（%d 个）"
          % (len(OWN_FUNCS), len(STACK_SYMS)))
    print("     MENTIONED（报告里出现 > 0 次的候选）= %s" % (sorted(mentioned) or "(空)"))
    print("     OWN（程序自己定义的函数）          = %s" % sorted(OWN_FUNCS))
    print("     MENTIONED ∩ OWN                    = %s" % (sorted(inter) or "∅"))
    check("A", "A.5", (not inter) and bool(mentioned),
          "★ **「报告提到的符号集合」∩「程序自己定义的函数集合」= 空集**"
          "（且 MENTIONED 非空 ⇒ 不是空报告）")

    print()
    print("     ★ 结论：采样器**不是没看见**，它看见了一个**没有你名字的栈**。")
    print("       `time.perf_counter()` 调 `mach_absolute_time`，于是「自旋等待」")
    print("       被归给了**内核计时调用** —— 报告没错，它在回答**另一个问题**：")
    print("       「这台机器此刻在哪个符号里」，而不是「你写的哪个函数最耗时」。")
    print("     ⚠ 正文引用时**只写「0 次 / > 0 次」这个结论，不写具体计数**")
    print("       （同域 `profiling-tools` 的 §0.7 D-13：绝对计数随程序与采样窗口变化）。")


# ===========================================================================
# 3. B 类：同一个名字（「耗时」）下的两个计数器
# ===========================================================================
def cpu_time():
    """读内核维护的 **per-process 计数器**：用户态 + 内核态 CPU 时间（秒）。

    ★ 它和 `time.perf_counter()`（挂钟）**不是同一个量**，而两者在中文里
      都可以叫「耗时」。这就是 F14「同名不同指」在本机的对应物。
    """
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return ru.ru_utime + ru.ru_stime


def measure_pair(fn):
    """跑 `fn`，返回 `(挂钟秒, CPU 秒)`。★ 顺序固定、且**只看差值**，不受累计值影响。"""
    w0, c0 = time.perf_counter(), cpu_time()
    fn()
    return time.perf_counter() - w0, cpu_time() - c0


def burn(seconds):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        pass


B_BURN_S = 0.25
B_SLEEP_S = 0.30


def part_B():
    print()
    print("=" * 74)
    print("B 类 · 同一个名字下的两个计数器：挂钟时间 vs CPU 时间")
    print("=" * 74)
    print("  ★ 两个计数器都是**内核/库维护的**读数，**都正确**，回答的是不同问题：")
    print("     · `time.perf_counter()`       —— **挂钟**：现在离开始过了多久；")
    print("     · `resource.getrusage()`      —— **CPU 时间**：这个进程真的占了 CPU 多久。")
    print("     ⇒ 中文里两者都可以叫「耗时」——这正是 F14（同名不同指）的本地对应物。")

    w_burn, c_burn = measure_pair(lambda: burn(B_BURN_S))
    w_sleep, c_sleep = measure_pair(lambda: time.sleep(B_SLEEP_S))
    r_burn = w_burn / c_burn if c_burn > 0 else float("inf")
    r_sleep = w_sleep / c_sleep if c_sleep > 0 else float("inf")

    print()
    print("     %-26s 挂钟 %-10s CPU %-10s 挂钟/CPU" % ("负载", "", ""))
    print("     %-26s %-10.4f %-10.4f %.3f" % ("纯计算 burn(%.2fs)" % B_BURN_S, w_burn, c_burn, r_burn))
    print("     %-26s %-10.4f %-10.4f %s"
          % ("睡眠 sleep(%.2fs)" % B_SLEEP_S, w_sleep, c_sleep,
             ("%.1f" % r_sleep) if r_sleep != float("inf") else "∞"))

    # ★ 窗取 20 是**有意的**：静息实测 1.005，但机器被打满时这个比值会被抬高
    #   （进程分到的 CPU 少了，挂钟还在走）—— 实施者实测 8 个忙循环打满时它能到 1.6 以上。
    #   这一条的区分力**不在这个绝对值**，在 `B.2` 的**对照**（6355 vs 1.0）。
    check("B", "B.1", 0.9 <= r_burn < 20.0,
          "纯计算负载：两个计数器**在同一量级**（挂钟 %.4f s ÷ CPU %.4f s = %.3f；"
          "静息实测 1.005）—— 单线程、不睡眠、不等待 I/O 时它们本来就该接近"
          % (w_burn, c_burn, r_burn))
    check("B", "B.2", r_sleep > 10.0 * max(r_burn, 1.0),
          "★ **同一个名字（「耗时」），两个计数器差 %.0f 倍**：睡眠 %.3f s 里 CPU 时间只有 "
          "%.4f s —— 而**两者都对**，它们量的不是同一件事（对照：纯计算那一行只差 %.3f 倍）"
          % (r_sleep, w_sleep, c_sleep, r_burn))
    check("B", "B.3", w_burn > 0 and w_sleep > 0 and c_burn > 0,
          "两个读数都取到了正值（不是 0 除以 0，也不是读数溢出）")


# ===========================================================================
# 4. C 类：规格事实（F10 / F11 / F12 / F14）与标记自检
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


def part_C(doc_dir):
    print()
    print("=" * 74)
    print("C 类 · 规格事实（**规格值，本仓库从未运行 ncu**）与标记自检")
    print("=" * 74)
    f10, f11, f12 = specs.FACTS[9], specs.FACTS[10], specs.FACTS[11]
    d47, d48 = specs.F14_DEFINITIONS

    print("  ── F14：**同名不同指**的两个 metric（并置，别只看其中一个）──")
    for d in specs.F14_DEFINITIONS:
        print("     §%s  `%s`" % (d["metric_section"], d["metric_name"]))
        print("          %s" % d["increment_rule"])
    print("     两节共享的第一句（「同名」的那一半）：%s" % specs.F14_SHARED_SENTENCE)

    check("C", "C.1",
          all(m in f10["quote"] for m in f10["must_contain"]),
          "F10 逐字：超范围的值**故意不 clamp**（`%s`）" % f10["must_contain"][1][:52])
    check("C", "C.2",
          ("consumer instruction" in f11["quote"]) and ("issue-slot" in f12["quote"]),
          "F11/F12 逐字：stall 归因给**消费者指令**；目标是**提高 issue-slot 利用率**，"
          "不是把 stall 计数器压到 0")
    check("C", "C.3",
          (d47["metric_section"] != d48["metric_section"])
          and (d47["increment_rule"] != d48["increment_rule"])
          and ("regardless if the scheduler issued" in d47["increment_rule"])
          and ("issued no instruction" in d48["increment_rule"]),
          "★ F14 的两套定义**同名、不同采样条件、取向相反**"
          "（§%s 与 §%s）—— 这正是 P9（同名不同指）的教科书实例"
          % (d47["metric_section"], d48["metric_section"]))

    rows = specs.fact_invariants()
    bad = [(lbl, got, want) for (lbl, got, want) in rows if got != want]
    for lbl, got, want in bad:
        print("     ✗ %s: got=%r want=%r" % (lbl, got, want))
    check("C", "C.4", not bad,
          "F1–F15 的**结构不变量** %d 条全过" % len(rows))

    if doc_dir is None:
        skip("C", "C.5", "两份 NVIDIA 文档不在场 ⇒ F1–F15 的**逐字核对**整体 SKIP")
    else:
        vrows = specs.verify_docs(doc_dir)
        vbad = [(l, g, w) for (l, g, w) in vrows if g < w]
        for l, g, w in vbad:
            print("     ✗ %s: 出现 %d 次，期望 ≥ %d" % (l, g, w))
        check("C", "C.5", not vbad,
              "F1–F15 与 F14 的两条增量规则**逐字在场**（%d 条）" % len(vrows))

    marks = {f: read_mark_from(os.path.join(HERE, f))
             for f in ("01_latency_bandwidth.py", "02_reading_counters.py",
                       "03_diagnosis_walkthrough.py", "meta_falsify.py", "mutate.py")}
    uniq = sorted({v for v in marks.values() if v is not None})
    print("     %s" % ", ".join("%s=%r" % (k, v) for k, v in sorted(marks.items())))
    check("C", "C.6",
          all(v is not None for v in marks.values())
          and len(uniq) == 1 and uniq[0] == ASSERT_FAIL_MARK,
          "★ 失败标记**五个文件全部在场且一致**：%r（本主题自定形态，非照抄别域）" % (uniq,))


# ===========================================================================
def lint():
    import pathlib
    files = sorted(pathlib.Path(HERE).glob("[0-9][0-9]_*.py")) + \
        [pathlib.Path(HERE) / f for f in ("specs.py", "microbench.py",
                                          "meta_falsify.py", "mutate.py")]
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
    print("02_reading_counters.py —— 读计数器有三种系统性误读，而且都不报错")
    print("=" * 74)
    print("  ★ 真值标注：%s" % specs.MEASURE_NOTE)
    print("  ★ 本程序**从不运行 `ncu`/`nsys`** —— 凡出现它们的 metric 名，标签都是**规格值**。")
    print("  ★ 测量条件：%s / python %s / %s"
          % (specs.MACHINE["cpu"], sys.version.split()[0], sys.platform))
    if os.environ.get("PERF_QUICK") == "1":
        # ★ **快档**：只把采样窗口与子进程时长调短，断言与代码路径完全不变。
        global SAMPLE_SECS, ATTACH_DELAY_S, CHILD_DUR
        SAMPLE_SECS, ATTACH_DELAY_S, CHILD_DUR = 1, 0.2, 1.2
        print("  ★ 档位：**快档**（PERF_QUICK=1，诊断工具用；采样窗口 %d s）" % SAMPLE_SECS)
    can_sample = have_sample()
    if not can_sample:
        print("  ⚠ 找不到 `sample`（macOS 自带）⇒ A 类断言将 **SKIP**，不计入通过。")
    doc_dir = specs.find_doc_dir(verbose=True)
    tmpdir = tempfile.mkdtemp(prefix="perf02b_")
    try:
        part_A(tmpdir, can_sample)
        part_B()
        part_C(doc_dir)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return summary()


if __name__ == "__main__":
    sys.exit(main())

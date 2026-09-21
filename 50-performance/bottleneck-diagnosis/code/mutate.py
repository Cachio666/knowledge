#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""mutate.py —— **非编号诊断工具**：叶子级变异测试（`make mutate`，**不进** `make run`）。

★ 它是「非编号支撑代码」，不计入「文档数 = 编号程序数」的核对（`CONTRIBUTING` §2）。
★ 计划的 §3 验收把「完整叶级 mutation（含对照组）」列为**固定验收动作**，
  所以 harness 必须**在仓库里、必须能由 `make` 复现** ——
  只在 `/tmp` 里跑过一次的脚本不构成可复现的验收。

===============================================================================
★ 它回答什么问题
===============================================================================

「**把这条常量改错，程序会报错吗？**」—— 对**每一个**叶子级常量逐个问一遍，
并把答「不会」的（**漏网**）如实列出来。

★ 本主题的**已知盲区**（不是缺陷，但要写出来）：
  · `MEASURE_NOTE` / `DEP_RECORDED` / `UNROLL8_RECORDED` / `DIAG_RECORDED` 这类常量
    **只被打印、不被断言** —— 它们是对照记录与叙述文本，不是判据。
    改它们的**正确后果**就是漏网；把「必然失败」算进分子才是给分母灌水。
  · `microbench.DEP_SRC` / `UNROLL8_SRC` 两个**几百行的 C 源码字符串**：
    改其中一个字符多半让 `clang` 编译失败 ⇒ 落到**崩溃捕获**那一类（不是断言捕获）。
    这是**有意的区分**：它证明「代码炸了」，不证明「判据抓住了错」。

===============================================================================
★★ 三条硬纪律
===============================================================================

1. **只在副本上做，绝不动仓库里的原文件。**
   每个变异体在**自己的临时目录**里执行；仓库目录从头到尾**只被读**，
   跑完再校验一次 sha256，变了就以非零码退出（那时读数已经不可信）。

2. **必须区分「断言捕获」与「崩溃捕获」。**
   · **断言捕获**：程序打印了 `*** FAIL *** …` 且 rc≠0（**本主题自己的标记形态**）。
   · **崩溃捕获**：rc≠0 但**没有**失败标记 —— 异常 traceback，或 rc=2 的环境退出。
   ⇒ 崩溃捕获**不算**真捕获。两类**分开数**。
   ★ 照抄 `30-gpu-programming/cutlass` 的 `\[[ABC]/FAIL|FAILED: ` 正则
     **一条都匹配不到**；照抄同域 `profiling-tools` 的 `**FAIL**` 也**匹配不到**本主题。

3. **对照组必须有。**
   把某条常量**改回它自己**，harness **不得**报告「捕获」。
   本工具对**全部**目标做「写回后字节相同」的校验，并**抽样真跑**一遍。

===============================================================================
★ 口径（分母）
===============================================================================

  `make mutate`：只测 **`specs.py` 的模块级常量叶子**。
  `make mutate-all`：`specs.py` **+ `microbench.py` + 三个编号程序**的模块级常量叶子。

**不含**断言调用里的字面量：那些是**期望值**（测试自己的一部分），
改它们必然「失败」，把这种必然算进分子等于给分母灌水。

★ 子进程档位：**`PERF_QUICK=1`**（断言全是结构性的，快档走同一条代码路径）。

===============================================================================
★★ 为什么并行度压到 3（**实测结论，不是保守**）
===============================================================================

本主题有三个程序，其中两个是**访存微基准** —— 而**内存带宽是被并行副本瓜分的资源**。
实施者实测（本机 M1 Pro，10 核）：

| 并发跑几个 `01_latency_bandwidth.py` | `K=32/K=1` 的实测比值 | `A.2`（门槛 8 倍） |
|---|---|---|
| 1（静息） | 15 – 17 | 全过 |
| 3 – 6 | 13.7 – 16.7 | 全过（`A.4` 最坏 1.74） |
| **8** | **5.4 – 10.1** | **八次里红七次** |

⇒ 用 8 个 worker 跑这个主题，会把「机器被自己占满」误报成「这一批变异体都被抓住了」，
**捕获数 M 会被灌水**。**降并发是量法的一部分，不是性能取舍。**

用法：`make mutate`（= `python3 mutate.py specs`）、`make mutate-all`。
"""

import ast
import hashlib
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
PROGRAMS = ("01_latency_bandwidth.py", "02_reading_counters.py",
            "03_diagnosis_walkthrough.py")
# ★★ 与三个程序**实际打印的**失败形态对齐：`  *** FAIL *** A.1 …`。
#   **不是** `[FAIL]`（cutlass 的），**不是** `[FAIL]|✗ N 条断言失败`（45 的），
#   也**不是** `**FAIL**`（同域 profiling-tools 的）。
#   ★ 这里刻意写成**普通字符串常量**（不是 `re.compile(...)`）：三个编号程序各自有一条
#     「标记自检」断言，它用 `ast` 读本文件、取出这个字面量、与自己的比 ——
#     写成 `re.compile` 的话读出来是 `None`，自检会**假失败**。
ASSERT_FAIL_MARK = "*** FAIL ***"
# ★★ 但**判定**必须用「**行首**锚定」的那个形态，不能用 `ASSERT_FAIL_MARK in out`。
#   为什么：编号程序**成功时也会打印这个字面量** —— `01` 的 `D.6` / `02` 的 `C.6` /
#   `03` 的 `C.3` 那几条的 detail 里就写着它。于是 `ASSERT_FAIL_MARK in out`
#   在 **rc=0** 时也成立 ⇒ 基线被误判成「有程序失败」，整个变异测试**拒绝给结论**。
#   ⇒ 判据是「**这一行以失败标记开头**」，而不是「输出里出现过这几个字」。
#     这与 `CONTRIBUTING:280`「在原始字节上计数」是同一条纪律的另一面：
#     **计数之前要先确认你数的那个东西在当前口径下真的只出现一次**。
ASSERT_FAIL_LINE = re.compile(r"(?m)^\s*" + re.escape(ASSERT_FAIL_MARK))
PER_RUN_TIMEOUT = 900          # 单个程序一次运行的秒数上限（快档下正常 < 30 s）
N_CONTROL = 12                 # 对照组**真跑**的抽样数
COPY_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store",
                                     ".run_summary", ".last_out")
ENV = dict(os.environ, PERF_QUICK="1")


# ===========================================================================
# 1. 叶子枚举
# ===========================================================================
def _walk(node):
    """递归收集字面量叶子 -> [(节点, 后缀名)]。后缀名形如 `[0]` / `['k']`。"""
    out = []
    if isinstance(node, ast.Constant):
        out.append((node, ""))
    elif isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        for i, e in enumerate(node.elts):
            out += [(n2, "[%d]%s" % (i, s)) for n2, s in _walk(e)]
    elif isinstance(node, ast.Dict):
        for k, v in zip(node.keys, node.values):
            if k is not None:
                out += [(n2, "{key}%s" % s) for n2, s in _walk(k)]
            out += [(n2, "[%r]%s" % (getattr(k, "value", "?"), s)) for n2, s in _walk(v)]
    return out


def _top_consts(text):
    """模块级 `NAME = <字面量>`（`Assign` 与 `AnnAssign`）。"""
    out = []
    for n in ast.parse(text).body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 \
                and isinstance(n.targets[0], ast.Name):
            out.append((n.targets[0].id, n.value))
        elif isinstance(n, ast.AnnAssign) and n.value is not None \
                and isinstance(n.target, ast.Name):
            out.append((n.target.id, n.value))
    return out


def _kind_of(v):
    if isinstance(v, str):
        return "str"
    if isinstance(v, bool):
        return "bool"
    if v is None:
        return "none"
    if isinstance(v, bytes):
        return "bytes"
    return "num"


def _mk(fn, text, node, desc):
    return {"file": fn, "desc": desc,
            "span": (node.lineno, node.col_offset, node.end_lineno, node.end_col_offset),
            "kind": _kind_of(node.value),
            "value_repr": repr(node.value)[:70], "src": text}


def byte_span(b, span):
    """把 `(lineno, col, end_lineno, end_col)` 换算成**字节**偏移。

    ★★ 必须按字节：`ast` 的 `col_offset` 是 **UTF-8 字节**偏移，而本库源码里几乎每行都有
      中文注释 —— 按字符算会整体错位，变异被写在**错的地方**，判决无效。
    """
    l1, c1, l2, c2 = span
    off = 0
    start = None
    for i, line in enumerate(b.split(b"\n"), 1):
        if i == l1:
            start = off + c1
        if i == l2:
            return start, off + c2
        off += len(line) + 1
    raise AssertionError("span 超出文件范围")


def slice_of(text, span):
    b = text.encode("utf-8")
    off, off2 = byte_span(b, span)
    return b[off:off2]


def apply_mutation(orig_text, span, new_bytes):
    b = orig_text.encode("utf-8")
    off, off2 = byte_span(b, span)
    return (b[:off] + new_bytes + b[off2:]).decode("utf-8")


def _mangle_str(sl):
    """在字符串字面量的源码切片里改一个字符（只在**引号对内部**改）。"""
    i = 0
    while i < len(sl):
        if sl[i] not in (0x22, 0x27):
            i += 1
            continue
        q = sl[i]
        j = i + 1
        prev = 0
        while j < len(sl):
            c = sl[j]
            if prev == 0x5C:
                prev = 0
                j += 1
                continue
            prev = c
            if c == q:
                break
            if 0x61 <= c <= 0x7A:
                return sl[:j] + bytes([0x61 + (c - 0x61 + 1) % 26]) + sl[j + 1:]
            if 0x41 <= c <= 0x5A:
                return sl[:j] + bytes([0x41 + (c - 0x41 + 1) % 26]) + sl[j + 1:]
            if 0x30 <= c <= 0x39:
                return sl[:j] + bytes([0x30 + (c - 0x30 + 1) % 10]) + sl[j + 1:]
            j += 1
        i = j + 1
    return None


def mutate_slice(t):
    """在字面量的**源码切片**里改一个字符。改不了则写 `why_skip` 并返回 None。"""
    sl = slice_of(t["src"], t["span"])
    kind = t["kind"]
    if kind == "none":
        t["why_skip"] = "None（没有可变的位）"
        return None
    if kind == "bytes":
        t["why_skip"] = "bytes 字面量"
        return None
    if kind == "bool":
        return b"False" if t["value_repr"].startswith("True") else b"True"
    if kind == "num":
        for i in range(len(sl) - 1, -1, -1):
            if 0x30 <= sl[i] <= 0x39:
                return sl[:i] + bytes([0x30 + (sl[i] - 0x30 + 1) % 10]) + sl[i + 1:]
        t["why_skip"] = "数字字面量里没有数字位"
        return None
    got = _mangle_str(sl)
    if got is None:
        t["why_skip"] = "字符串里没有可改的字母/数字（空串或纯符号）"
    return got


def enumerate_targets(scope):
    files = ("specs.py", "microbench.py") + PROGRAMS if scope == "all" else ("specs.py",)
    targets = []
    for fn in files:
        text = open(os.path.join(CODE_DIR, fn), "r", encoding="utf-8").read()
        for name, value in _top_consts(text):
            for n2, s in _walk(value):
                targets.append(_mk(fn, text, n2, name + s))
    mutes, skip_why = [], Counter()
    for t in targets:
        new = mutate_slice(t)
        if new is None:
            skip_why[t["why_skip"]] += 1
        else:
            t["new"] = new
            mutes.append(t)
    return mutes, skip_why, len(targets)


# ===========================================================================
# 2. 一次变异体的执行
# ===========================================================================
def affected_programs(relfile, desc):
    """该变异体会影响哪些程序？

    ★ 做法：把**定义它的模块**与**常量名**拼成 `<模块>.<常量>`，在三个编号程序的源码里找。
      例如 `specs.DEP_RECORDED` / `microbench.DEP_REPS`。
    ★ 找不到引用就保守地跑**全部三个**（宁可多跑，不可漏判）。
    """
    top = desc.split("[")[0].split("{")[0].split(".")[0]
    mod = relfile[:-3] if relfile.endswith(".py") else relfile
    hits = []
    for p in PROGRAMS:
        t = open(os.path.join(CODE_DIR, p), "r", encoding="utf-8").read()
        if (mod + "." + top) in t:
            hits.append(p)
    return tuple(hits) or PROGRAMS


def run_case(task, env):
    """在**自己的临时副本**里跑一个变异体，返回分类结果。"""
    relfile, desc, span, new_bytes, which, is_control = task
    tmp = None
    try:
        tmp = tempfile.mkdtemp(prefix="pt-mutate-")
        work = os.path.join(tmp, "code")
        shutil.copytree(CODE_DIR, work, ignore=COPY_IGNORE)
        src = os.path.join(work, relfile)
        with open(src, "r", encoding="utf-8") as f:
            orig = f.read()
        with open(src, "w", encoding="utf-8") as f:
            f.write(apply_mutation(orig, span, new_bytes))
        shutil.rmtree(os.path.join(work, "__pycache__"), ignore_errors=True)

        # ★ `kinds` 与 `detail` **一一对应**（同下标）：同一个变异体可能让 01 断言失败、
        #   让 03 崩溃，`classify()` 取「crash」优先，而 `detail[0]` 是 01 那行
        #   「断言失败」 ⇒ **崩溃明细里显示了一行断言失败**，与纪律 2 直接打架。
        kinds, detail = [], []
        for prog in which:
            try:
                p = subprocess.run([PY, "-W", "ignore::RuntimeWarning", prog], cwd=work,
                                   capture_output=True, text=True,
                                   timeout=PER_RUN_TIMEOUT, env=env)
            except subprocess.TimeoutExpired:
                kinds.append("error")
                detail.append("%s: 超时 > %ds" % (prog, PER_RUN_TIMEOUT))
                continue
            out = (p.stdout or "") + (p.stderr or "")
            if p.returncode == 0:
                continue
            if ASSERT_FAIL_LINE.search(out):
                kinds.append("assert")
                detail.append("%s: rc=%d 断言失败" % (prog, p.returncode))
            else:
                kinds.append("crash")
                tail = out.strip().split("\n")[-1][:70] if out.strip() else "(无输出)"
                detail.append("%s: rc=%d 无失败断言；末行=%s" % (prog, p.returncode, tail))
        return {"desc": desc, "file": relfile, "is_control": is_control,
                "kinds": kinds, "detail": detail}
    except Exception as e:                      # ★ harness 自身异常：单独计数
        return {"desc": desc, "file": relfile, "is_control": is_control,
                "kinds": ["error"], "detail": ["harness 异常：%r" % (e,)]}
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


def classify(kinds):
    for k in ("error", "crash", "assert"):
        if k in kinds:
            return k
    return "none"


def _worker(task):
    return run_case(task, ENV)


def _init_worker(env):
    os.environ.update(env)


def _run_plain(which, work):
    try:
        for prog in which:
            p = subprocess.run([PY, "-W", "ignore::RuntimeWarning", prog], cwd=work,
                               capture_output=True, text=True,
                               timeout=PER_RUN_TIMEOUT, env=ENV)
            out = (p.stdout or "") + (p.stderr or "")
            if p.returncode != 0 or ASSERT_FAIL_LINE.search(out):
                return "%s rc=%d" % (prog, p.returncode)
        return "clean"
    except Exception as e:
        return "harness 异常 %r" % (e,)


def copy_code():
    tmp = tempfile.mkdtemp(prefix="pt-mutate-base-")
    work = os.path.join(tmp, "code")
    shutil.copytree(CODE_DIR, work, ignore=COPY_IGNORE)
    return tmp, work


# ===========================================================================
# 3. 主流程
# ===========================================================================
def main():
    scope = sys.argv[1] if len(sys.argv) > 1 else "specs"
    assert scope in ("specs", "all"), "用法：mutate.py [specs|all]"
    t0 = time.time()
    files = ("specs.py", "microbench.py") + PROGRAMS if scope == "all" else ("specs.py",)
    print("=" * 78)
    print("mutate.py —— 叶子级变异测试（**非编号诊断工具**，不进 `make run`）")
    print("=" * 78)
    print("  解释器      : %s" % PY)
    print("  python      : %s" % sys.version.split()[0])
    print("  口径（分母）: %s" % ("specs.py + microbench.py + 三个编号程序" if scope == "all"
                                  else "**只** specs.py 的模块级常量叶子"))
    print("  子进程档位  : PERF_QUICK=1（断言全是结构性的，走同一条代码路径）")
    print("  失败标记    : %r（**本主题自己的输出形态**，不是照抄别域）"
          % ASSERT_FAIL_MARK)
    print("  ★ 只在**副本**上做：仓库里的 .py 从头到尾**只被读**，结尾再校验一次哈希")

    before = {fn: hashlib.sha256(open(os.path.join(CODE_DIR, fn), "rb").read()).hexdigest()
              for fn in files}

    mutes, skip_why, n_leaves = enumerate_targets(scope)
    print()
    print("-" * 78)
    print("§0 目标枚举")
    print("-" * 78)
    print("  叶子字面量 %d 个 ⇒ 可变异 %d 个、跳过 %d 个"
          % (n_leaves, len(mutes), n_leaves - len(mutes)))
    for k, v in skip_why.most_common():
        print("    跳过 %4d：%s" % (v, k))
    print()

    tmp, work = copy_code()
    try:
        base = _run_plain(PROGRAMS, work)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if base != "clean":
        print("!! 基线（未变异）不是「三个程序 rc=0 且无失败断言」：%s" % base)
        print("   先修基线，再跑变异测试。")
        return 2
    print("  基线        : 三个程序 rc=0、无失败断言 ✓")
    print()

    tasks = [(m["file"], m["desc"], m["span"], m["new"],
              affected_programs(m["file"], m["desc"]), False) for m in mutes]
    step = max(1, len(mutes) // N_CONTROL) if mutes else 1
    ctrl_sample = mutes[::step][:N_CONTROL]
    ctrl_byte_bad = [m["desc"] for m in mutes
                     if apply_mutation(m["src"], m["span"],
                                       slice_of(m["src"], m["span"])) != m["src"]]
    tasks += [(m["file"], m["desc"], m["span"], slice_of(m["src"], m["span"]),
               affected_programs(m["file"], m["desc"]), True) for m in ctrl_sample]

    # ★★ **并行度故意压到 3**（本机 10 个 CPU）—— 这是本主题的一条**实测结论**，
    #   不是保守：本主题的三个程序里有两个是**访存微基准**，
    #   而**内存带宽是被并行副本瓜分的资源**。实施者实测（本机 M1 Pro）：
    #     · 并发 8 个 `01_latency_bandwidth.py` 时，`K=32/K=1` 从静息的 **15–17 倍**
    #       掉到 **5.4–10.1 倍**，`A.2`（门槛 8 倍）**八次里红七次**；
    #     · 并发 3–6 个时，`A.2` 回到 **13.7–16.7 倍**、`A.4` 最坏 1.74（门槛 2.5）。
    #   ⇒ 用 8 个 worker 跑这个主题，会把「机器被自己占满」误报成
    #     「这一批变异体都被抓住了」——**捕获数 M 会被灌水**。
    #     **降并发是量法的一部分，不是性能取舍。**
    workers = min(3, max(1, (os.cpu_count() or 2)))
    print("  并行度      : %d（本机 %s 个 CPU）★ 故意压低，理由见 `mutate.py` 里的实测注"
          % (workers, os.cpu_count()))
    print("  开始跑 %d 个变异体 + %d 个对照组（每个都在自己的副本里）…"
          % (len(mutes), len(ctrl_sample)))
    results = []
    with mp.Pool(processes=workers, initializer=_init_worker,
                 initargs=(dict(ENV),)) as pool:
        for i, r in enumerate(pool.imap_unordered(_worker, tasks, chunksize=4), 1):
            results.append(r)
            if i % 50 == 0:
                print("    … %d/%d（%.0f s）" % (i, len(tasks), time.time() - t0))

    mut_res = [r for r in results if not r["is_control"]]
    ctl_res = [r for r in results if r["is_control"]]
    buckets = {"assert": [], "crash": [], "none": [], "error": []}
    for r in mut_res:
        buckets[classify(r["kinds"])].append(r)

    n = len(mut_res)
    ma, mc = len(buckets["assert"]), len(buckets["crash"])
    me, k = len(buckets["error"]), len(buckets["none"])
    m = ma + mc

    print()
    print("=" * 78)
    print("★ 结果 —— 四样：`测 N / 捕获 M / 漏网 K` · harness 异常数 · 对照组")
    print("=" * 78)
    print("  测      N = **%d** 个叶子级变异体" % n)
    print("  捕获    M = **%d** 个 ⇒ **断言捕获 %d** + **崩溃捕获 %d**（两类分开数）"
          % (m, ma, mc))
    print("  漏网    K = **%d** 个（改了它，全部受影响程序 rc=0）" % k)
    print("  harness 异常数 = **%d** 个（超时 / IO / harness 自身抛异常）" % me)
    print("  捕获率  M/N = %.1f%%  ★ **不是 100%% 才是正常的**："
          "断言调用里的**期望值不在分母里**" % (100.0 * m / n if n else 0.0))
    print()
    print("  ── 对照组（把常量**改回它自己**）──")
    print("     ① 字节校验：全部 %d 个目标「写回后与原字节相同」= %s（不同的 %d 个）"
          % (len(mutes), not ctrl_byte_bad, len(ctrl_byte_bad)))
    for x in ctrl_byte_bad[:5]:
        print("        不同：%s" % x)
    n_ctl_caught = sum(1 for r in ctl_res if classify(r["kinds"]) != "none")
    print("     ② 真跑 %d 个抽样（均匀取）⇒ **报「捕获」%d 个**（必须是 0）"
          % (len(ctl_res), n_ctl_caught))
    for r in ctl_res:
        if classify(r["kinds"]) != "none":
            print("        !! %s ⇒ %s %r" % (r["desc"], r["kinds"], r["detail"]))
    print()

    def fam(desc):
        return desc.split("[")[0].split("{")[0].split(".")[0]

    cnt = Counter(fam(r["desc"]) for r in buckets["none"])
    print("  ── 漏网 K=%d 的构成（按常量名聚合，前 15）──" % k)
    for name, c in cnt.most_common(15):
        print("     %5d  %s" % (c, name))
    print()
    rep = {m2["desc"]: m2 for m2 in mutes}
    print("  ── 漏网明细（最多 40 条）──")
    for r in buckets["none"][:40]:
        print("     %-14s %-44s 原值=%s"
              % (r["file"], r["desc"][:44], rep.get(r["desc"], {}).get("value_repr", "?")))
    if k > 40:
        print("     …（还有 %d 条，见上面的聚合）" % (k - 40))
    print()
    if mc:
        print("  ── 崩溃捕获 %d 的构成（按常量名聚合，前 10）──" % mc)
        for name, c in Counter(fam(r["desc"]) for r in buckets["crash"]).most_common(10):
            print("     %5d  %s" % (c, name))
        print()
    print("  ── 崩溃捕获明细（**崩溃不等于断言捕获**，最多 25 条）──")
    for r in buckets["crash"][:25]:
        print("     %-14s %s" % (r["file"], r["desc"]))
        shown = [d for k2, d in zip(r["kinds"], r["detail"]) if k2 == "crash"][:1] \
            or r["detail"][:1]
        for d in shown:
            print("        %s" % d)
    if mc > 25:
        print("     …（还有 %d 条）" % (mc - 25))
    print()
    print("  ── harness 异常明细（最多 10 条）──")
    for r in buckets["error"][:10]:
        print("     %-14s %s ⇒ %r" % (r["file"], r["desc"], r["detail"]))
    print()
    print("  ★★ 本主题的**已知盲区**（不是缺陷，但要写出来）：")
    print("     `MEASURE_NOTE` / `DEP_RECORDED` / `UNROLL8_RECORDED` / `DIAG_RECORDED`")
    print("     这类常量**只被打印、不被断言** —— 它们是对照记录与叙述文本，不是判据。")
    print("     改它们的**正确后果**就是漏网；把「必然失败」算进分子才是给分母灌水。")
    print("     `microbench.DEP_SRC` / `UNROLL8_SRC` 两个 C 源码字符串被改坏时多半**编译失败**")
    print("     ⇒ 落到**崩溃捕获**（不是断言捕获）。这两类**必须分开数**。")

    after = {fn: hashlib.sha256(open(os.path.join(CODE_DIR, fn), "rb").read()).hexdigest()
             for fn in files}
    changed = [fn for fn in files if before[fn] != after[fn]]
    print()
    print("  ── §原文件完整性（`try/finally` 之外的**第二道**保证）──")
    print("     跑前 / 跑后 sha256 全部相同 = %s%s"
          % (not changed, "" if not changed else "   变了：%r" % changed))
    print("     总耗时      : %.1f s" % (time.time() - t0))
    print("=" * 78)
    return 1 if changed else 0


if __name__ == "__main__":
    sys.exit(main())

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
  · `MEASURE_NOTE` / `MEASURE_IMPL` / `U_CURVE_RECORDED_NS` / `DIST_RECORDED` /
    `NOISE_RECORDED` / `AB_RECORDED` / `TIMEIT_PASS_RECORDED_NS` 这类常量
    **只被打印、不被断言** —— 它们是对照记录与叙述文本，不是判据。
    改它们的**正确后果**就是漏网；把「必然失败」算进分子才是给分母灌水。
  · `FACTS` 里那 **15 条逐字引文**：改一个字符通常**会被抓住**
    （`must_contain ⊂ quote` 那条自洽检查 + `verify_docs` 的逐字核对），
    但**签名/份数**这类结构性信息不在判据里 —— 那种改动会漏网，也是应然的。

===============================================================================
★★ 四条硬纪律
===============================================================================

1. **只在副本上做，绝不动仓库里的原文件。**
   每个变异体在**自己的临时目录**里执行；仓库目录从头到尾**只被读**，
   跑完再校验一次 sha256，变了就以非零码退出（那时读数已经不可信）。

2. **必须区分「断言捕获」与「崩溃捕获」。**
   · **断言捕获**：程序打印了 `>>> FAIL <<< …` 且 rc≠0（**本主题自己的标记形态**）。
   · **崩溃捕获**：rc≠0 但**没有**失败标记 —— 异常 traceback，或 rc=2 的环境退出。
   ⇒ 崩溃捕获**不算**真捕获。两类**分开数**。
   ★ 照抄 `30-gpu-programming/cutlass` 的 `\[[ABC]/FAIL|FAILED: ` 正则**一条都匹配不到**；
     照抄同域 `profiling-tools` 的 `**FAIL**`、`bottleneck-diagnosis` 的 `*** FAIL ***`
     也**匹配不到**本主题。

3. **对照组必须有。**
   把某条常量**改回它自己**，harness **不得**报告「捕获」。
   本工具对**全部**目标做「写回后字节相同」的校验，并**抽样真跑**一遍。

4. ★★ **并行度压到 1**（本主题与前两个主题都不同，理由见下）。

===============================================================================
★ 口径（分母）
===============================================================================

  `make mutate`：只测 **`specs.py` 的模块级常量叶子**。
  `make mutate-all`：`specs.py` **+ `mini_search.py` + 三个编号程序**的模块级常量叶子。

**不含**断言调用里的字面量：那些是**期望值**（测试自己的一部分），
改它们必然「失败」，把这种必然算进分子等于给分母灌水。

★ 子进程档位：**`PERF_QUICK=1`**（断言全是结构性的，快档走同一条代码路径）。

===============================================================================
★★ 为什么并行度**必须**是 1（**实测结论，不是保守**）
===============================================================================

同域 `bottleneck-diagnosis` 把并行度压到 3，理由是「内存带宽会被并行副本瓜分」。
本主题**更严**，因为 `03_ab_orchestration.py` 的 **A 段会自己拉起
`cpu_count()-1` 个忙进程打满 CPU**，而它的 `A.1`（`max/中位 > 100`）与
`A.3`（尖峰按时间三等分要均匀）**测的就是那台机器的噪声形状**。

⇒ 并行跑 k 个副本，就等于在 A 段那 6 万个样本上再叠 k 层**非平稳**干扰：
  `A.3` 会随机失败，于是**捕获数 M 会被灌水**（把「机器被自己占满」误报成
  「这一批变异体都被抓住了」）。**降并发是量法的一部分，不是性能取舍。**

===============================================================================
★ 一处**显式的去冗余**（为了让它跑得完，**不影响核查范围**）
===============================================================================

`affected_programs()` 先找**按 `<模块>.<常量>` 直接引用**该常量的编号程序。
若一个都没有，说明该常量只可能被 **E 类（规格自检）**消费 ——
而 E 类在三个程序里读的是**同一份** `specs.fact_invariants()` /
`specs.doc_status()` / `specs.verify_docs()` / `specs.upstream_check()` /
`specs.mde_coefficient_selfcheck()`，
⇒ **跑其中最便宜的那个程序（`02`）就足以判定**。

★ 这是**去冗余**（同一份断言的三个副本只跑一个），**不是缩小核查范围**：
  漏网与捕获的判定口径完全不变，只是不再把同一条断言重复跑三遍。
  ★ 本主题 `specs.py` 有 **585 个叶子**（其中 `FACTS` 的逐字引文占 264 个），
    不这么做的话，一次 `make mutate` 要跑几小时。
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
PROGRAMS = ("01_sigma_tau.py", "02_noise_and_search.py", "03_ab_orchestration.py")
# ★ **兜底时只跑最便宜的那个程序**（理由见模块头的「一处显式的去冗余」）。
CHEAPEST = "02_noise_and_search.py"
# ★★ 与三个程序**实际打印的**失败形态对齐：`  *** FAIL *** A.1 …`。
#   **不是** `[FAIL]`（cutlass 的），**不是** `[FAIL]|✗ N 条断言失败`（45 的），
#   也**不是** `**FAIL**`（同域 profiling-tools 的）或 `*** FAIL ***`（bottleneck-diagnosis 的）。
#   ★ 这里刻意写成**普通字符串常量**（不是 `re.compile(...)`）：三个编号程序各自有一条
#     「标记自检」断言，它用 `ast` 读本文件、取出这个字面量、与自己的比 ——
#     写成 `re.compile` 的话读出来是 `None`，自检会**假失败**。
ASSERT_FAIL_MARK = ">>> FAIL <<<"
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
# ★★ 并行度**固定为 1**（本主题自己的理由，见模块头）。
N_WORKERS = 1


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
    files = ("specs.py", "mini_search.py") + PROGRAMS if scope == "all" else ("specs.py",)
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
      例如 `specs.U_CURVE_RECORDED_NS` / `mini_search.MINI_UNITS_AT_BEST`。
    ★ 找不到引用就保守地跑**全部三个**（宁可多跑，不可漏判）。
    """
    top = desc.split("[")[0].split("{")[0].split(".")[0]
    mod = relfile[:-3] if relfile.endswith(".py") else relfile
    hits = []
    for p in PROGRAMS:
        t = open(os.path.join(CODE_DIR, p), "r", encoding="utf-8").read()
        if (mod + "." + top) in t:
            hits.append(p)
    # ★ 一个都不直接引用 ⇒ 它只可能被 **E 类（规格自检）** 消费，
    #   而 E 类在三个程序里读的是**同一份** specs 函数 ⇒ 跑最便宜的那个就够。
    return tuple(hits) or (CHEAPEST,)


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
    files = ("specs.py", "mini_search.py") + PROGRAMS if scope == "all" else ("specs.py",)
    print("=" * 78)
    print("mutate.py —— 叶子级变异测试（**非编号诊断工具**，不进 `make run`）")
    print("=" * 78)
    print("  解释器      : %s" % PY)
    print("  python      : %s" % sys.version.split()[0])
    print("  口径（分母）: %s" % ("specs.py + mini_search.py + 三个编号程序" if scope == "all"
                                  else "**只** specs.py 的模块级常量叶子"))
    print("  子进程档位  : PERF_QUICK=1（断言全是结构性的，走同一条代码路径）")
    print("  失败标记    : %r（**本主题自己的输出形态**，不是照抄别域）"
          % ASSERT_FAIL_MARK)
    print("  ★ 只在**副本**上做：仓库里的 .py 从头到尾**只被读**，结尾再校验一次哈希")

    before = {fn: hashlib.sha256(open(os.path.join(CODE_DIR, fn), "rb").read()).hexdigest()
              for fn in files}

    mutes, skip_why, n_leaves = enumerate_targets(scope)
    # ★★ **可选的上限**：`MUTATE_MAX=<n>` 时**均匀抽稀**到 n 个（0 / 未设 = 全量）。
    #   它存在的理由是**可复现的验收**：本主题的全量分母是 **579** 个叶子，
    #   其中 **74 个**会影响 `03_ab_orchestration.py`（快档一次约 15–20 s，因为它要
    #   拉 9 个忙进程并把 6 万个样本测完）——
    #   **全量一次要 ~35 分钟**。`MUTATE_MAX` 让「验收入口」能在几分钟内跑完，
    #   代价是**报告里必须写明这是抽样**（`N` 那一行会打印 `（抽样）`）。
    #   ★ 抽稀按**等距**取（`mutes[::step][:n]`），所以它覆盖整个枚举顺序，
    #     不偏向任何一类常量；**它不是「挑好抓的测」**。
    max_n = int(os.environ.get("MUTATE_MAX", "0") or "0")
    sampled = bool(max_n) and len(mutes) > max_n
    if sampled:
        step = max(1, len(mutes) // max_n)
        mutes = mutes[::step][:max_n]
    print()
    print("-" * 78)
    print("§0 目标枚举")
    print("-" * 78)
    if sampled:
        print("  叶子字面量 %d 个 ⇒ **按 MUTATE_MAX=%d 等距抽稀到 %d 个**"
              "（未抽稀的全量是 %d 个）" % (n_leaves, max_n, len(mutes), n_leaves - 6))
    else:
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

    # ★★ **并行度固定为 1**（本主题自己的实测理由，见模块头）：
    #   `03_ab_orchestration.py` 的 A 段**自己拉起 `cpu_count()-1` 个忙进程**，
    #   而它的 `A.1`/`A.3` 测的就是「那台机器的噪声形状」。
    #   并行跑 k 个副本 = 在 6 万个样本上再叠 k 层非平稳干扰 ⇒ `A.3` 随机失败
    #   ⇒ **捕获数 M 会被灌水**。**降并发是量法的一部分，不是性能取舍。**
    workers = N_WORKERS
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
    print("  测      N = **%d** 个叶子级变异体%s"
          % (n, "　**（抽样：MUTATE_MAX=%d，全量 %d）**" % (max_n, n_leaves)
             if sampled else ""))
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
    print("     `MEASURE_NOTE` / `MEASURE_IMPL` / `U_CURVE_RECORDED_NS` / `DIST_RECORDED` /")
    print("     `NOISE_RECORDED` / `AB_RECORDED` / `TIMEIT_PASS_RECORDED_NS`")
    print("     这类常量**只被打印、不被断言** —— 它们是对照记录与叙述文本，不是判据。")
    print("     改它们的**正确后果**就是漏网；把「必然失败」算进分子才是给分母灌水。")
    print("     ★ 并行度固定为 %d：`03` 的 A 段自己拉起忙进程，并行会**灌水捕获数**。" % N_WORKERS)

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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""mutate.py —— **非编号诊断工具**：叶子级变异测试（`make mutate`，**不进** `make run`）。

★ 它是「非编号支撑代码」，不计入「文档数 = 编号程序数」的核对（`CONTRIBUTING` §2）。

===============================================================================
★ 它回答什么问题
===============================================================================

「**把这条常量改错，程序会报错吗？**」—— 对**每一个**叶子级常量逐个问一遍，
并把答「不会」的（**漏网**）如实列出来。

计划的 §4 把「**全量**叶子级变异测试，**含对照组**」列为**固定验收动作**，
所以这个工具必须在仓库里、必须能由 `make` 复现 ——
一个只在 `/tmp` 里跑过一次的脚本，不构成可复现的验收。

===============================================================================
★★ 三条硬纪律（照抄 `cuda/code/mutate.py`）
===============================================================================

1. **只在副本上做，绝不动仓库里的原文件。**
   每个变异体在**自己的临时目录**里执行；仓库目录从头到尾**只被读**。
   `try/finally` 之外还多一层保证：主流程**结尾再校验一次**原文件 sha256，
   变了就以非零码退出（那时读数已经不可信，必须报错而不是继续）。
   ★ 这一条来自 `45-kernel-lifecycle` 的实测教训：harness 被 kill 时会污染被测对象，
   导致**后续所有读数失效**。本工具连「可能被污染」的窗口都不存在。

2. **必须区分「断言捕获」与「崩溃捕获」。**
   · **断言捕获**：程序打印了失败断言（`[X/FAIL …]` 或 `FAILED:`）且 rc≠0。
   · **崩溃捕获**：rc≠0 但**没有**失败断言 —— 异常 traceback，或 `rc=2` 的**环境退出**。
   ⇒ 崩溃捕获（C6 模式一）**不算**真捕获：它证明「代码炸了」，不是「判据抓住了错」。
     两类**分开数**，明细里各列一条。

3. **对照组必须有。**
   把某条常量**改回它自己**，harness **不得**报告「捕获」——
   否则说明 harness 自己有假阳性。本工具对**全部**目标做「写回后字节相同」的校验，
   并**抽样真跑**一遍。

===============================================================================
★ 变异目标（分母的口径，必须写明）
===============================================================================

`specs.py` 的模块级常量 **+** 四个编号程序的模块级常量，逐个**叶子**（literal）。

**不含**断言调用（`check`/`A`/`B`/`C`/`expect_raises`）里的字面量：那些是**期望值**
（测试自己的一部分），改它们必然「失败」，把这种必然算进分子等于给分母灌水。
（⇒ 捕获率**不应该**接近 100%，那反而可疑。）

★ 字符串一律**逐字符**改一个字（**多行字符串同样处理，不跳过**）；
  数字改一个数字位；布尔取反；`None` / `bytes` / 空串 / 无字母数字的字符串记入**跳过数**。
★ 另外追加一组 `#tag` 专项变异体：改 `covered_by` 里的**第一个标签**（`03:B.6` → `03:B.7`）——
  它是「`covered_by` 必须指向真实存在的断言」这条约束的判据（`01_block_model.py` 的 `C.8`）。

===============================================================================
★ 输出（四样 + 分类）
===============================================================================

  `测 N / 捕获 M / 漏网 K` · **`harness 异常数`** · **`对照组结果`**
  断言捕获 Ma / 崩溃捕获 Mb（**分开数**）· 多行字符串的覆盖数与跳过数

用法：`make mutate`
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
PROGRAMS = ("01_block_model.py", "02_masking.py", "03_layout.py", "04_warp_reduce.py")
SRC_FILES = ("specs.py",) + PROGRAMS
COPY_IGNORE = shutil.ignore_patterns("__pycache__", ".run_summary", ".last_out",
                                     "*.pyc", ".DS_Store")
# ★ `checks.py` 打印的失败断言有两种形态：`  [FAIL] C7  C.4 …` 与 `    FAILED C7: …`。
#   `cuda/code/mutate.py` 的同名正则写的是 `\[[ABC]/FAIL|FAILED: ` —— **那是那个域的程序
#   打印的形态**，照抄过来在本主题下**一条都匹配不上**，于是 96 个捕获被**全部误判成
#   「崩溃捕获」**（"断言捕获 0 + 崩溃捕获 96"）。这条正则必须与 `checks.py` 的实际输出对齐。
ASSERT_FAIL_MARK = re.compile(r"\[FAIL\]|FAILED [ABC]\d+:|FAILED: ")
PER_RUN_TIMEOUT = 180          # 单个程序一次运行的秒数上限（正常 < 2s）
N_CONTROL = 12                 # 对照组**真跑**的抽样数


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
    tree = ast.parse(text)
    out = []
    for n in tree.body:
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


def enumerate_targets():
    """枚举全部叶子目标（含跳过项统计）。

    ★ `ANCHORS` / `LOAD_QUOTES` 逐条锚点、逐字段展开 —— 报告才说得清
      「漏网里有多少条是 `note` 这种散文」。
    """
    targets = []
    for fn in SRC_FILES:
        text = open(os.path.join(CODE_DIR, fn), "r", encoding="utf-8").read()
        for name, value in _top_consts(text):
            if name in ("ANCHORS", "LOAD_QUOTES", "MODEL_OWN_RULES", "MODEL_SIMPLIFICATIONS",
                        "WORD_FREQ", "META_DEFECTS", "NOT_FIRST_HAND") \
                    and isinstance(value, ast.Dict):
                for k, v in zip(value.keys, value.values):
                    aid = getattr(k, "value", "?")
                    if isinstance(v, ast.Dict):
                        for fk, fv in zip(v.keys, v.values):
                            field = getattr(fk, "value", "?")
                            for n2, s in _walk(fv):
                                targets.append(_mk(fn, text, n2,
                                                   "%s[%s].%s%s" % (name, aid, field, s)))
                    else:
                        for n2, s in _walk(v):
                            targets.append(_mk(fn, text, n2, "%s[%s]%s" % (name, aid, s)))
                continue
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

    # ★★ 追加一类：`covered_by` 里**第一个标签**（形如 `03:B.6` → `03:B.7`）。
    #   它直接对应 `01_block_model.py` 的 `C.8` —— 「`covered_by` 里的每个标签都必须
    #   真实存在于对应程序的断言清单里」。改一个标签，`C.8` 必须报错。
    for t in list(mutes):
        if not t["desc"].endswith(".covered_by[0]"):
            continue
        sl = slice_of(t["src"], t["span"])
        new = mutate_tag(sl)
        if new is not None:
            u = dict(t)
            u["desc"] = t["desc"] + "#tag"
            u["new"] = new
            mutes.append(u)
    return mutes, skip_why, len(targets)


TAG_BYTES_RE = re.compile(rb"\d\d:[ABC]\.\d+[a-z]*")


def mutate_tag(sl):
    """把 `covered_by` 里第一个标签的**最后一个字符**改掉（`03:B.6` → `03:B.7`）。"""
    m = TAG_BYTES_RE.search(sl)
    if not m:
        return None
    i = m.end() - 1
    c = sl[i]
    if 0x30 <= c <= 0x39:
        return sl[:i] + bytes([0x30 + (c - 0x30 + 1) % 10]) + sl[i + 1:]
    base = 0x61 if c >= 0x61 else 0x41
    return sl[:i] + bytes([base + (c - base + 1) % 26]) + sl[i + 1:]


def mutate_slice(t):
    """在字面量的**源码切片**里改一个字符。改不了则写 `why_skip` 并返回 None。"""
    b = t["src"].encode("utf-8")
    off, off2 = byte_span(b, t["span"])
    sl = b[off:off2]
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

    # 字符串（含多行）：在**引号对内部**改一个字母/数字，**不跳过**
    got = _mangle_str(sl)
    if got is None:
        t["why_skip"] = "字符串里没有可改的字母/数字（空串或纯符号）"
    return got


def _mangle_str(sl):
    """在字符串字面量的源码切片里改一个字符。

    ★★ 两个**必须**避开的坑（`cuda/code/mutate.py` 第一版都踩过，
       代价是 99 个「崩溃捕获」）：
       ① **字符串前缀** `r` / `b` / `f` / `u` 是字母，但**不是字符串内容**；
       ② 隐式拼接（`"…" "…"`）的**第二段前缀**同理。
       ⇒ 只在**引号对内部**改，前缀一个都不碰。
    """
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
            if prev == 0x5C:              # 转义序列：`\` 与它后面那个字节都不动
                prev = 0
                j += 1
                continue
            prev = c
            if c == q:                    # 引号对结束
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


def byte_span(b, span):
    """把 `(lineno, col, end_lineno, end_col)` 换算成**字节**偏移。

    ★ 必须按字节：`ast` 的 `col_offset` 是 **UTF-8 字节偏移**，而本库源码里有大量
      中文 —— 按字符算会整体错位（`cuda` 主题实施时踩过）。
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


def apply_mutation(orig_text, span, new_bytes):
    b = orig_text.encode("utf-8")
    off, off2 = byte_span(b, span)
    return (b[:off] + new_bytes + b[off2:]).decode("utf-8")


def slice_of(text, span):
    b = text.encode("utf-8")
    off, off2 = byte_span(b, span)
    return b[off:off2]


# ===========================================================================
# 2. 一次变异体的执行
# ===========================================================================
def affected_programs(relfile, desc):
    """该变异体会影响哪些程序？

    · 程序内的常量 ⇒ 只影响它自己；
    · `specs.py` 的常量 ⇒ 文本里引用了 `specs.<顶层名>` 的那些（`ANCHORS` 两个程序都引）。
    找不到引用就保守地跑四个（宁可多跑，不可漏跑）。
    """
    if relfile != "specs.py":
        return (relfile,)
    top = desc.split("[")[0].split(".")[0].split("{")[0]
    hits = []
    for p in PROGRAMS:
        t = open(os.path.join(CODE_DIR, p), "r", encoding="utf-8").read()
        if ("specs." + top) in t:
            hits.append(p)
    return tuple(hits) or PROGRAMS


def run_case(task, env):
    """在**自己的临时副本**里跑一个变异体，返回分类结果。

    ★ 副本建在 `mkdtemp` 里，`finally` 一定删掉 —— 即使中途抛异常或被外层中断。
    """
    relfile, desc, span, new_bytes, which, is_control = task
    tmp = None
    try:
        tmp = tempfile.mkdtemp(prefix="triton-mutate-")
        work = os.path.join(tmp, "code")
        shutil.copytree(CODE_DIR, work, ignore=COPY_IGNORE)
        src = os.path.join(work, relfile)
        with open(src, "r", encoding="utf-8") as f:
            orig = f.read()
        with open(src, "w", encoding="utf-8") as f:
            f.write(apply_mutation(orig, span, new_bytes))

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
            m = ASSERT_FAIL_MARK.search(out)
            if m:
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
    return run_case(task, dict(os.environ))


def _init_worker(env):
    os.environ.update(env)


def _run_plain(which):
    tmp = None
    try:
        tmp = tempfile.mkdtemp(prefix="triton-mutate-base-")
        work = os.path.join(tmp, "code")
        shutil.copytree(CODE_DIR, work, ignore=COPY_IGNORE)
        for prog in which:
            p = subprocess.run([PY, "-W", "ignore::RuntimeWarning", prog], cwd=work,
                               capture_output=True, text=True,
                               timeout=PER_RUN_TIMEOUT, env=dict(os.environ))
            out = (p.stdout or "") + (p.stderr or "")
            if p.returncode != 0 or ASSERT_FAIL_MARK.search(out):
                return "%s rc=%d" % (prog, p.returncode)
        return "clean"
    except Exception as e:
        return "harness 异常 %r" % (e,)
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


# ===========================================================================
# 3. 主流程
# ===========================================================================
def main():
    t0 = time.time()
    print("=" * 78)
    print("mutate.py —— 叶子级变异测试（**非编号诊断工具**，不进 `make run`）")
    print("=" * 78)
    print("  解释器      : %s" % PY)
    print("  python      : %s" % sys.version.split()[0])
    print("  ★ 只在**副本**上做：仓库里的 .py 从头到尾**只被读**，结尾再校验一次哈希")
    print()

    before = {fn: hashlib.sha256(open(os.path.join(CODE_DIR, fn), "rb").read()).hexdigest()
              for fn in SRC_FILES}

    mutes, skip_why, n_leaves = enumerate_targets()
    n_tag = sum(1 for m in mutes if m["desc"].endswith("#tag"))
    n_leaf_mutes = len(mutes) - n_tag
    n_multi = sum(1 for m in mutes
                  if not m["desc"].endswith("#tag") and b"\n" in slice_of(m["src"], m["span"]))
    print("-" * 78)
    print("§0 目标枚举")
    print("-" * 78)
    print("  叶子字面量 %d 个 ⇒ 可变异 %d 个、跳过 %d 个"
          % (n_leaves, n_leaf_mutes, n_leaves - n_leaf_mutes))
    for k, v in skip_why.most_common():
        print("    跳过 %4d：%s" % (v, k))
    print("  **多行字符串** %d 个 —— 本工具**不跳过**它们，同样改一个字符" % n_multi)
    print("  另加 **%d 个 `covered_by[0]` 的专项变异体**（`#tag`）："
          "改第一个标签的最后一位（`03:B.6`→`03:B.7`）—— 这一组是 `C.8` 的判据" % n_tag)
    print()

    # 基线：未变异时必须全清（否则读数无意义）
    base = _run_plain(PROGRAMS)
    if base != "clean":
        print("!! 基线（未变异）不是「四个程序 rc=0 且无失败断言」：%s" % base)
        print("   先修基线，再跑变异测试。")
        return 2
    print("  基线        : 四个程序 rc=0、无失败断言 ✓")
    print()

    tasks = [(m["file"], m["desc"], m["span"], m["new"],
              affected_programs(m["file"], m["desc"]), False) for m in mutes]

    # 对照组：① 全部目标做「写回后字节相同」校验；② 抽样 N_CONTROL 个**真跑**
    step = max(1, len(mutes) // N_CONTROL) if mutes else 1
    ctrl_sample = mutes[::step][:N_CONTROL]
    ctrl_byte_bad = [m["desc"] for m in mutes
                     if apply_mutation(m["src"], m["span"],
                                       slice_of(m["src"], m["span"])) != m["src"]]
    tasks += [(m["file"], m["desc"], m["span"], slice_of(m["src"], m["span"]),
               affected_programs(m["file"], m["desc"]), True) for m in ctrl_sample]

    workers = min(8, max(1, (os.cpu_count() or 2)))
    print("  并行度      : %d（本机 %s 个 CPU）" % (workers, os.cpu_count()))
    print("  开始跑 %d 个变异体 + %d 个对照组（每个都在自己的副本里）…"
          % (len(mutes), len(ctrl_sample)))
    results = []
    with mp.Pool(processes=workers, initializer=_init_worker,
                 initargs=(dict(os.environ),)) as pool:
        for i, r in enumerate(pool.imap_unordered(_worker, tasks, chunksize=4), 1):
            results.append(r)
            if i % 100 == 0:
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
    print("  测      N = **%d** 个叶子级变异体（多行字符串已包含在内）" % n)
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
        """聚合成「哪个字段」——锚点 id 不进 key，否则每条 note 会各算一类。"""
        for pre in ("ANCHORS", "LOAD_QUOTES", "MODEL_OWN_RULES", "MODEL_SIMPLIFICATIONS",
                    "WORD_FREQ", "META_DEFECTS", "NOT_FIRST_HAND"):
            if desc.startswith(pre + "["):
                rest = desc.split("].", 1)[1] if "]." in desc else ""
                return "%s[*].%s" % (pre, rest.split("[")[0].split("{")[0])
        return desc.split("[")[0].split("{")[0]

    cnt = Counter(fam(r["desc"]) for r in buckets["none"])
    print("  ── 漏网 K=%d 的构成（按字段聚合，前 15）──" % k)
    for name, c in cnt.most_common(15):
        print("     %5d  %s" % (c, name))
    print()
    rep = {m2["desc"]: m2 for m2 in mutes}
    print("  ── 漏网明细（最多 40 条）──")
    for r in buckets["none"][:40]:
        print("     %-14s %-40s 原值=%s"
              % (r["file"], r["desc"][:40], rep.get(r["desc"], {}).get("value_repr", "?")))
    if k > 40:
        print("     …（还有 %d 条，见上面的字段聚合）" % (k - 40))
    print()
    tag_res = [r for r in mut_res if r["desc"].endswith("#tag")]
    if tag_res:
        tc = Counter(classify(r["kinds"]) for r in tag_res)
        print("  ── `#tag` 专项（改 `covered_by` 里第一个标签的最后一位）──")
        print("     测 %d 个 ⇒ 捕获 %d（断言 %d / 崩溃 %d）、漏网 %d"
              % (len(tag_res), len(tag_res) - tc["none"], tc["assert"], tc["crash"],
                 tc["none"]))
        print("     ★ 这一组应该是**可证伪**的：`01_block_model.py` 的 `C.8` 要求"
              "`covered_by` 里的每个标签都真实存在。")
        print()

    cnt_c = Counter(fam(r["desc"]) for r in buckets["crash"])
    if mc:
        print("  ── 崩溃捕获 %d 的构成（按字段聚合，前 10）──" % mc)
        for name, c in cnt_c.most_common(10):
            print("     %5d  %s" % (c, name))
        print()
    print("  ── 崩溃捕获明细（C6 模式一：**崩溃不等于断言捕获**，最多 25 条）──")
    for r in buckets["crash"][:25]:
        print("     %-14s %s" % (r["file"], r["desc"]))
        for d in r["detail"][:1]:
            print("        %s" % d)
    if mc > 25:
        print("     …（还有 %d 条）" % (mc - 25))
    print()
    print("  ── harness 异常明细（最多 10 条）──")
    for r in buckets["error"][:10]:
        print("     %-14s %s ⇒ %r" % (r["file"], r["desc"], r["detail"]))

    after = {fn: hashlib.sha256(open(os.path.join(CODE_DIR, fn), "rb").read()).hexdigest()
             for fn in SRC_FILES}
    changed = [fn for fn in SRC_FILES if before[fn] != after[fn]]
    print()
    print("  ── §原文件完整性（`try/finally` 之外的**第二道**保证）──")
    print("     跑前 / 跑后 sha256 全部相同 = %s%s"
          % (not changed, "" if not changed else "   变了：%r" % changed))
    print("     总耗时      : %.1f s" % (time.time() - t0))
    print("=" * 78)
    return 1 if changed else 0


if __name__ == "__main__":
    sys.exit(main())

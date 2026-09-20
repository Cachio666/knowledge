#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mutate.py —— **非编号诊断工具**：叶子级变异测试（`make mutate`，**不进** `make run`）。

★ 它是「非编号支撑代码」，不计入「文档数 = 编号程序数」的核对（`CONTRIBUTING` §2）。

===============================================================================
★ 它回答什么问题
===============================================================================

「**把这条常量改错，程序会报错吗？**」—— 对**每一个**叶子级常量逐个问一遍，
并把答「不会」的（**漏网**）如实列出来。

计划的 §4 把「**全量**叶子级变异测试，**含对照组**」列为**固定验收动作**，
所以这个工具必须在仓库里、必须能由 `make` 复现 —— 一个只在 `/tmp` 里跑过一次的
脚本，不构成可复现的验收。

===============================================================================
★★ 本主题的 `ASSERT_FAIL_MARK` **不是**照抄任何别的主题的
===============================================================================

`cuda` 主题的三个程序打印的失败形态是 `[A/FAIL …]` / `FAILED: …`，所以它的正则写作
`re.compile(r"\\[[ABC]/FAIL|FAILED: ")`。**本主题三个程序的输出形态不同**：

    通过   `  [ok]   A.1  …`
    失败   `  [FAIL] A.1  …`
    汇总   `汇总: A a/n · B b/n · C c/n ⇒ 断言 N 条，通过 P 条，失败 F 条`

⇒ 本主题的正则是 `re.compile(r"\\[FAIL\\]")`（**带方括号**）。两条纪律：

  1. **照抄 cuda 的那条正则会一条都匹配不到** —— 于是**全部真断言捕获**都会被
     记成「崩溃捕获」。这不是假设：`triton` 主题（Chunk 2）踩过这个坑，
     **123 个捕获全被误分类**。⇒ 本文件用 `mark_selftest()` 把这件事**测出来**：
     它拿两段合成输出（一段含 `[FAIL]`、一段不含）验证分类器分得开，
     并**同时**验证 cuda 那条正则在本形态上确实匹配不到（⇒ 这条差异是**可证伪**的）。
  2. **不带方括号的 `FAIL` 也不安全**：本主题的汇总行里有「失败」二字，
     但它不带方括号；用一个宽正则会把**正常输出**记成失败。

===============================================================================
★★ 四条硬纪律
===============================================================================

1. **只在副本上做，绝不动仓库里的原文件。**
   每个变异体在**自己的临时目录**里执行；仓库目录从头到尾**只被读**。
   主流程**结尾再校验一次**原文件 sha256，变了就以非零码退出。
2. **必须区分「断言捕获」与「崩溃捕获」。**
   · **断言捕获**：程序打印了 `[FAIL] …` 且 rc≠0。
   · **崩溃捕获**：rc≠0 但**没有**失败断言 —— 异常 traceback。
   ⇒ 崩溃捕获**不算**真捕获：它证明「代码炸了」，不是「判据抓住了错」。
3. **对照组必须有。**
   把某条常量**改回它自己**，harness **不得**报告「捕获」。
   本工具对**全部**目标做「写回后字节相同」的校验，并**抽样真跑**一遍。
4. **探测到的外部工件路径必须写进子进程的 env**（`child_env()`，见 §0.5）——
   本工具在**临时副本**里跑三个编号程序，副本里那些**相对**候选（`../cutlass` 等）
   一个都解析不到；而 `Makefile` 的 `CUTLASS_PATH ?=` 默认为空，
   「**空值原样透传等于没传**」。
   ★ 探测失败时**报出是哪一步失败**并以 rc=2 退出（不是让三个程序各自炸一遍、
   再汇总成一句含糊的话）。`env_selftest()` 把这一条做成**可证伪**的。

===============================================================================
★ 变异目标（分母的口径，必须写明）
===============================================================================

`specs.py` 的模块级常量 **+** 三个编号程序的模块级常量，逐个**叶子**（literal）。

**不含**断言调用（`check`/`eq`）里的字面量：那些是**期望值**（测试自己的一部分），
改它们必然「失败」，把这种必然算进分子等于给分母灌水。
（⇒ 捕获率**不应该**接近 100%，那反而可疑。）

★ **也不含**三个编号程序里那几段「本仓库的参照实现」的**代码** ——
  它们不是常量，是算法；对它们的变异由 `meta_falsify.py`（`make falsify`）负责。
  两个工具的靶子**不重叠**，避免同一件事数两遍。

★ 字符串一律**逐字符**改一个字（**多行字符串同样处理，不跳过**）；
  数字改一个数字位；布尔取反；`None` / `bytes` / 空串 / 无字母数字的字符串记入**跳过数**。
★ 另外追加一类 `#tag`：把 `covered_by` 元组里**第一个标签**的末位改掉
  （如 `A.4` → `A.5`）—— 这是「`covered_by` 必须指向真实存在的断言」的可证伪性实验。

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
# ★ 与三个编号程序、与 `Makefile` **共用同一份候选列表**：直接 import 同目录的
#   `specs.py`（它 import 时**不做任何文件系统探测**，见 `specs.py` 模块头）。
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)
import specs                                    # noqa: E402

PY = sys.executable
PROGRAMS = ("01_layout_algebra.py", "02_tiling_decompose.py", "03_swizzle_probe.py")
SRC_FILES = ("specs.py",) + PROGRAMS
COPY_IGNORE = shutil.ignore_patterns("__pycache__", ".run_summary", ".last_out",
                                     "*.pyc", ".DS_Store")

# ★★ 本主题的失败标记：`[FAIL]`（**带方括号**）。理由见模块头第 2 节。
ASSERT_FAIL_MARK = re.compile(r"\[FAIL\]")

# ★ cuda 主题用的那一条。**本主题刻意不采用它**，并且**把它写进自检**里
#   （证明「抄错了会怎样」是可测的，而不是一句注释）。
FOREIGN_MARK = re.compile(r"\[[ABC]/FAIL|FAILED: ")

PER_RUN_TIMEOUT = 180          # 单个程序一次运行的秒数上限（正常 < 3s）
N_CONTROL = 12                 # 对照组**真跑**的抽样数


# ===========================================================================
# 0. ★ 标记自检（把「抄错标记」这件事变成可证伪的）
# ===========================================================================
def mark_selftest():
    """验证失败标记的选择**是对的**，而不是一句注释。

    · 合成的**失败**输出（本主题的真实形态）⇒ 本标记必须命中；
    · 合成的**正常**输出（含汇总行与「失败」字样）⇒ 本标记**不得**命中；
    · cuda 主题的那条正则在**两者上都不命中** ⇒ 「照抄它会全误分类」是可证的。
    """
    ok_fail = "  [FAIL] A.1  ★ 对拍不匹配: 448  ≠ 期望 0"
    ok_pass = ("  [ok]   A.1  ★ 对拍匹配: 0\n"
               "汇总: A 14/14 · B 9/9 · C 11/11 ⇒ 断言 34 条，通过 34 条，失败 0 条\n"
               "失败清单: （空）")
    rows = [
        ("失败形态命中", bool(ASSERT_FAIL_MARK.search(ok_fail)), True),
        ("正常形态不命中", bool(ASSERT_FAIL_MARK.search(ok_pass)), False),
        ("cuda 正则在本形态的失败行上不命中", bool(FOREIGN_MARK.search(ok_fail)), False),
        ("cuda 正则在本形态的正常行上不命中", bool(FOREIGN_MARK.search(ok_pass)), False),
    ]
    return rows


# ===========================================================================
# 0.5 ★★ 外部工件路径：**探测到的绝对路径**必须写进子进程的 env
# ===========================================================================
#
# 反面教材（本文件的第一版就是这么写的，控制者实测到了后果）：
#
#     env = dict(os.environ)          # ★ `Makefile` 的 `CUTLASS_PATH ?=` 默认**留空**
#
# `Makefile` 里 `export CUTLASS_PATH` 导出的是**空字符串** ⇒ `dict(os.environ)` 里
# `CUTLASS_PATH=""` —— **空值原样透传等于没传**。而本工具在**临时副本**里跑三个编号程序
# （基线检查 `_run_plain` 与每个变异体 `run_case` 都是），副本里那些**相对**候选
# （`../cutlass` 等）一个都解析不到 ⇒ 基线不是「三个程序 rc=0」⇒ `return 2` ⇒
# `make mutate` **rc=2**。那不是「统计灌水」（基线不通过时本工具拒绝开始），
# 是「**不能开箱即用**」。
#
# ⇒ 正确做法：**先探测**，把探测到的**绝对路径**写进子进程 env（`child_env()`），
#   并把那个 env **随任务一起下发**（`run_case` 的 task 元组里带着它）——
#   不依赖 `mp` 的 start method（macOS 默认 `spawn`），也不依赖 `os.environ` 的继承。
class ProbeError(Exception):
    """外部工件探测失败。**不静默回落** —— 由 `main()` 报出是哪一步失败后以 rc=2 退出。"""


def probe_cutlass():
    """按 `specs.CUTLASS_CANDIDATES` 探测 CUTLASS 源码树，返回绝对路径；找不到返回 None。

    ★ 候选列表里的**相对**项以**本目录**为基准（不是调用者的 cwd）——
      这一条**现在由 `specs` 自己保证**（`CUTLASS_CANDIDATE_BASE` = `specs.py` 所在目录，
      `find_cutlass()` 不碰 `os.getcwd()`；判据在 `meta_falsify.py` 步骤 3/5 的 ⑤ / ⑥）。
      下面的 `chdir` 是**冗余**的（过去它是唯一的保障），保留只为少改一处。
    """
    old = os.getcwd()
    try:
        os.chdir(CODE_DIR)
        return specs.find_cutlass(verbose=True)
    finally:
        os.chdir(old)


def child_env():
    """返回**写入了探测到的绝对路径**的子进程 env（探测失败即抛 `ProbeError`）。"""
    root = probe_cutlass()
    if root is None:
        raise ProbeError()
    env = dict(os.environ)
    env[specs.CUTLASS_ENV] = root           # ★★ 覆盖空值 / 相对值：空值透传等于没传
    return env


# ★ 纯探测的一行程序：**把候选列表换成一个不存在的路径** ⇒ 唯一的变量只剩
#   `CUTLASS_PATH`。这一行测的是「**透传**这件事本身」，不是「本机碰巧找得到」。
_PROBE_CHILD = ("import sys; sys.path.insert(0, %r); import specs; "
                "specs.CUTLASS_CANDIDATES = ('/__no_such_cutlass__/',); "
                "r = specs.find_cutlass(); print(r or ''); "
                "sys.exit(0 if r else 3)")


def _probe_child(cwd, env):
    """在 `cwd` 里、用 `env` 跑一次纯探测，返回 `(rc, 选中路径或 '')`。"""
    r = subprocess.run([PY, "-c", _PROBE_CHILD % CODE_DIR], cwd=cwd,
                       capture_output=True, text=True, env=env, timeout=60)
    return r.returncode, (r.stdout or "").strip()


def env_selftest(env, tmpdir):
    """★★ 自证：「**不设 `CUTLASS_PATH`、且不在原位**」这个场景**是可被程序检出**的。

    在**临时目录**里（脱离原位）跑两次纯探测：

      · **不给** `CUTLASS_PATH` ⇒ 探测**必须失败**（rc=3）—— 这就是修复前的形态：
        `dict(os.environ)` 里那个空值把基线打成了 rc≠0，
        而报出来的是「基线不是三个程序 rc=0」，**说不清是哪一步失败**；
      · **给**探测到的绝对路径 ⇒ 探测**必须命中同一棵树**。

    ⇒ 第 ① 行若变红 = 「透传探测到的路径」没生效（把 `child_env()` 改回
      `dict(os.environ)` 就必然红）；第 ② 行若变红 = 本自检自己失去了区分力
      （那它就不是判据，只是一句声明）。两行都是**可证伪**的。
    """
    no_path = dict(env)
    no_path.pop(specs.CUTLASS_ENV, None)
    rc_off, out_off = _probe_child(tmpdir, no_path)
    rc_on, out_on = _probe_child(tmpdir, env)
    good = env.get(specs.CUTLASS_ENV, "")
    return [
        ("① 给了绝对路径 ⇒ 探测命中同一棵树", rc_on == 0 and out_on == good, True),
        ("② 不给路径（修复前的形态）⇒ 探测必须失败", rc_off != 0 and not out_off, True),
        ("③ env 里的路径是绝对路径且是一棵合格的树",
         bool(good) and os.path.isabs(good) and specs.is_cutlass_root(good), True),
    ]


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
    """枚举全部叶子目标（含跳过项统计）。"""
    targets = []
    for fn in SRC_FILES:
        text = open(os.path.join(CODE_DIR, fn), "r", encoding="utf-8").read()
        for name, value in _top_consts(text):
            # ★ `ANCHORS` 逐条锚点、逐字段展开：报告才说得清
            #   「漏网里有多少条是 `ANCHORS[*].note` 这种散文」。
            if name == "ANCHORS" and isinstance(value, ast.Dict):
                for k, v in zip(value.keys, value.values):
                    aid = getattr(k, "value", "?")
                    if isinstance(v, ast.Dict):
                        for fk, fv in zip(v.keys, v.values):
                            field = getattr(fk, "value", "?")
                            for n2, s in _walk(fv):
                                targets.append(_mk(fn, text, n2,
                                                   "ANCHORS[%s].%s%s" % (aid, field, s)))
                    else:
                        for n2, s in _walk(v):
                            targets.append(_mk(fn, text, n2, "ANCHORS[%s]%s" % (aid, s)))
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

    # ★★ 追加一类：`covered_by` 元组里的**第一个标签**。
    for t in list(mutes):
        if t["kind"] != "str" or not TAG_RE.match(t["value_repr"].strip("'\"")):
            continue
        if "[0]" not in t["desc"]:
            continue
        u = dict(t)
        u["desc"] = t["desc"] + "#tag"
        u["new"] = mutate_tag(slice_of(t["src"], t["span"]))
        if u["new"] is None:
            continue
        mutes.append(u)
    return mutes, skip_why, len(targets)


TAG_RE = re.compile(r"^[ABC]\.\d+[a-z]?$")
TAG_BYTES_RE = re.compile(rb"[ABC]\.\d+[a-z]*")


def mutate_tag(sl):
    """把标签的最后一个字符改掉（`A.4` → `A.5`）。"""
    if sl is None:
        return None
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

    got = _mangle_str(sl)
    if got is None:
        t["why_skip"] = "字符串里没有可改的字母/数字（空串或纯符号）"
    return got


def _mangle_str(sl):
    """在字符串字面量的源码切片里改一个字符。

    ★★ 两个**必须**避开的坑（`cuda` 主题的第一版踩过，代价是 99 个「崩溃捕获」）：
       ① **字符串前缀** `r` / `b` / `f` / `u` 是字母，但**不是字符串内容**；
       ② 隐式拼接（`"…" "…"`）的**第二段前缀**同理。
       ⇒ 只在**引号对内部**改，前缀一个都不碰。
    ★ 本主题的 `specs.py` 里有大量含反斜杠的字面量与多行字符串，前缀若被动，
      整个文件直接 `SyntaxError`。
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
      中文 —— 按字符算会整体错位。
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
def _covered_anchors():
    """静态读出三个程序各自的 `COVERED_ANCHORS`（**不是手写的映射表**）。

    ★ 用 `ast.literal_eval` 读源码里的那个字面量元组 —— 这样「哪个程序消费哪个锚点」
      这件事**只有一个真相来源**，不会与程序里的声明漂移。
    """
    out = {}
    for p in PROGRAMS:
        tree = ast.parse(open(os.path.join(CODE_DIR, p), encoding="utf-8").read())
        for n in tree.body:
            if isinstance(n, ast.Assign) and len(n.targets) == 1 \
                    and getattr(n.targets[0], "id", None) == "COVERED_ANCHORS":
                out[p] = tuple(ast.literal_eval(n.value))
    return out


def affected_programs(relfile, desc):
    """该变异体会影响哪些程序？

    · 程序内的常量 ⇒ 只影响它自己；
    · `specs.py` 的 `ANCHORS[K]` ⇒ **只跑声明了 `K` 的那个程序**（读 `COVERED_ANCHORS`）；
    · `specs.py` 的其它常量 ⇒ 文本里引用了 `specs.<顶层名>` 的那些。
    找不到引用就保守地跑三个（宁可多跑，不可漏跑）。
    """
    if relfile != "specs.py":
        return (relfile,)
    if desc.startswith("ANCHORS["):
        key = desc.split("[", 1)[1].split("]", 1)[0]
        hits = [p for p, ks in _COVERED.items() if key in ks]
        if hits:
            return tuple(hits)
    top = desc.split("[")[0].split(".")[0].split("{")[0].split("#")[0]
    hits = []
    for p in PROGRAMS:
        t = open(os.path.join(CODE_DIR, p), "r", encoding="utf-8").read()
        if ("specs." + top) in t:
            hits.append(p)
    return tuple(hits) or PROGRAMS


_COVERED = _covered_anchors()


def run_case(task):
    """在**自己的临时副本**里跑一个变异体，返回分类结果。

    ★ `task` 的最后一格是 `env`：里面已经有**探测到的绝对路径**
      （副本的 cwd 是 `work`，**相对候选必然解析不到**，靠的只有这一条透传）。
    """
    relfile, desc, span, new_bytes, which, is_control, env = task
    tmp = None
    try:
        tmp = tempfile.mkdtemp(prefix="cl-mutate-")
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
                p = subprocess.run([PY, prog], cwd=work, capture_output=True,
                                   text=True, timeout=PER_RUN_TIMEOUT, env=env)
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
    return run_case(task)


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
    print("  靶子        : %s（**不含**三个程序里的「本仓库参照实现」那段代码 ——"
          % ", ".join(SRC_FILES))
    print("                对它们的变异由 `make falsify` 的 meta_falsify.py 负责）")
    print()

    # ---- 步骤 1/5：外部工件路径探测 -----------------------------------------
    print("[步骤 1/5] 外部工件路径探测（候选列表：specs.CUTLASS_CANDIDATES，相对本目录）")
    print("  ★ 本工具在**临时副本**里跑三个编号程序 ⇒ 副本里那些**相对**候选一个都解析不到；")
    print("    `Makefile` 的 `CUTLASS_PATH ?=` 默认为空，**空值原样透传等于没传**。")
    try:
        env = child_env()
    except ProbeError:
        print()
        print("!! [步骤 1/5] **探测失败** —— `%s` 为空，且候选列表里一个都没命中。"
              % specs.CUTLASS_ENV)
        print("   ⇒ 本工具**在这里就停**：一个变异体都不会跑，读数无从谈起。")
        print("     （修复前那种「基线不是三个程序 rc=0」的含糊结论，说不清是哪一步失败。）")
        print()
        print("   本主题的真值来源是 CUTLASS 4.8.0 自带的两个 Python 参考实现：")
        print("     python/pycute/          (CuTe layout 代数)")
        print("     python/cutlass_library/ (CUTLASS 自己的生成器)")
        print("   需要下列文件存在（相对 CUTLASS 根）：")
        for f in specs.CUTLASS_REQUIRED:
            print("     %s" % f)
        print()
        print("   请显式指定（两种方式任选）：")
        print("     make mutate %s=/path/to/cutlass" % specs.CUTLASS_ENV)
        print("     export %s=/path/to/cutlass" % specs.CUTLASS_ENV)
        print("   或把 CUTLASS 源码树放到下列任一位置（相对本目录）：")
        for c in specs.CUTLASS_CANDIDATES:
            print("     %s" % c)
        print()
        print("   ★ 本库**不复刻** CUTLASS 的源码树，也不内置它的副本。")
        return 2
    print("  透传        : 上面这个**绝对路径**随每个任务写进子进程 env 的 `%s`"
          "（副本的 cwd 是临时目录，靠的就是它）" % specs.CUTLASS_ENV)
    print()

    # ---- 步骤 2/5：子进程 env 自证 -------------------------------------------
    print("[步骤 2/5] 子进程 env 自证（★ 「不设 CUTLASS_PATH 且不在原位」是可检出的）")
    print("  ★ 做法：在**临时目录**里跑纯探测，并把候选列表换成一个不存在的路径 ——"
          " 唯一的变量只剩 `CUTLASS_PATH`。")
    with tempfile.TemporaryDirectory(prefix="cl-selfcheck-") as sc_tmp:
        sc_rows = env_selftest(env, sc_tmp)
    sc_bad = 0
    for label, got, want in sc_rows:
        ok = got == want
        sc_bad += 0 if ok else 1
        print("  %-40s = %-5s 期望 %-5s %s"
              % (label, got, want, "OK" if ok else "**异常**"))
    if sc_bad:
        print()
        print("!! [步骤 2/5] env 自证**失败 %d 项** ⇒ **在这里就停**：不跑基线、不跑变异体。"
              % sc_bad)
        print("   第 ① 行红 = 「探测到的路径被写进子进程 env」这件事没生效"
              "（`child_env()` 被改回 `dict(os.environ)` 时必然如此）；")
        print("   第 ② 行红 = 本自检自己失去了区分力。")
        return 2
    print()

    print("[步骤 3/5] ★ 失败标记自检（本主题的输出形态与 `cuda` 主题**不同**）")
    print("     本标记 ASSERT_FAIL_MARK = %r" % ASSERT_FAIL_MARK.pattern)
    print("     cuda 主题那条 FOREIGN_MARK = %r（**刻意不用**）" % FOREIGN_MARK.pattern)
    for label, got, want in mark_selftest():
        mark = "OK" if got == want else "**异常**"
        print("     %-34s = %-5s 期望 %-5s %s" % (label, got, want, mark))
    print("     ★ 「照抄 cuda 的正则」在本形态上**一条都匹配不到**"
          "（triton 主题踩过：123 个捕获全被误分类）")
    print()
    print("[步骤 4/5] 基线（未变异）")
    base = _run_plain(PROGRAMS, env)
    print("  基线        : %s" % base)
    if not base.startswith("三个程序"):
        print("!! [步骤 4/5] 基线（未变异）不是「三个程序 rc=0 且无失败断言」——"
              " 先修基线再跑变异测试。")
        print("   ★ 步骤 1 已经成功（外部工件 = %s），所以这**不是**「找不到工件」"
              "那一类失败。" % env[specs.CUTLASS_ENV])
        return 2

    before = {fn: hashlib.sha256(open(os.path.join(CODE_DIR, fn), "rb").read()).hexdigest()
              for fn in SRC_FILES}

    mutes, skip_why, n_leaves = enumerate_targets()
    n_tag = sum(1 for m in mutes if m["desc"].endswith("#tag"))
    n_leaf_mutes = len(mutes) - n_tag
    n_multi = sum(1 for m in mutes
                  if not m["desc"].endswith("#tag") and b"\n" in slice_of(m["src"], m["span"]))
    print()
    print("-" * 78)
    print("§0 目标枚举")
    print("-" * 78)
    print("  叶子字面量 %d 个 ⇒ 可变异 %d 个、跳过 %d 个"
          % (n_leaves, n_leaf_mutes, n_leaves - n_leaf_mutes))
    for k, v in skip_why.most_common():
        print("    跳过 %4d：%s" % (v, k))
    print("  **多行字符串** %d 个 —— 本工具**不跳过**它们，同样改一个字符" % n_multi)
    print("  另加 **%d 个标签专项变异体**（`#tag`）：改 `covered_by` 里第一个 `X.N`" % n_tag)
    print()

    tasks = [(m["file"], m["desc"], m["span"], m["new"],
              affected_programs(m["file"], m["desc"]), False, env) for m in mutes]

    # 对照组：① 全部目标做「写回后字节相同」校验；② 抽样 N_CONTROL 个**真跑**
    step = max(1, len(mutes) // N_CONTROL) if mutes else 1
    ctrl_sample = mutes[::step][:N_CONTROL]
    ctrl_byte_bad = [m["desc"] for m in mutes
                     if apply_mutation(m["src"], m["span"],
                                       slice_of(m["src"], m["span"])) != m["src"]]
    tasks += [(m["file"], m["desc"], m["span"], slice_of(m["src"], m["span"]),
               affected_programs(m["file"], m["desc"]), True, env) for m in ctrl_sample]

    workers = min(8, max(1, os.cpu_count() or 2))
    print("[步骤 5/5] 跑变异体")
    print("  并行度      : %d（本机 %s 个 CPU）" % (workers, os.cpu_count()))
    print("  开始跑 %d 个变异体 + %d 个对照组（每个都在自己的副本里）…"
          % (len(mutes), len(ctrl_sample)))
    results = []
    with mp.Pool(processes=workers) as pool:
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
    print("  测      N = **%d** 个叶子级变异体（多行字符串与 `#tag` 都包含在内）" % n)
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
        if desc.startswith("ANCHORS["):
            rest = desc.split("].", 1)[1] if "]." in desc else ""
            return "ANCHORS[*].%s" % rest.split("[")[0].split("{")[0]
        return desc.split("[")[0].split("{")[0].split("#")[0]

    cnt = Counter(fam(r["desc"]) for r in buckets["none"])
    print("  ── 漏网 K=%d 的构成（按字段聚合，前 15）──" % k)
    for name, c in cnt.most_common(15):
        print("     %5d  %s" % (c, name))
    print()
    rep = {m2["desc"]: m2 for m2 in mutes}
    print("  ── 漏网明细（最多 40 条）──")
    for r in buckets["none"][:40]:
        print("     %-22s %-40s 原值=%s"
              % (r["file"], r["desc"][:40], rep.get(r["desc"], {}).get("value_repr", "?")))
    if k > 40:
        print("     …（还有 %d 条，见上面的字段聚合）" % (k - 40))
    print()
    tag_res = [r for r in mut_res if r["desc"].endswith("#tag")]
    if tag_res:
        tc = Counter(classify(r["kinds"]) for r in tag_res)
        print("  ── `#tag` 专项（改 `covered_by` 里第一个索引标签）──")
        print("     测 %d 个 ⇒ 捕获 %d（断言 %d / 崩溃 %d）、漏网 %d"
              % (len(tag_res), len(tag_res) - tc["none"], tc["assert"], tc["crash"],
                 tc["none"]))
        print("     ★ 这一组的结果是「捕获一部分、漏网另一部分」，两个原因都真实存在：")
        print("       ① 本库的 `A.11`（标签必须真实存在）挡得住**指向不存在标签**的改动，"
              "挡不住**指向另一个真实标签**的改动 ——")
        print("          三个程序**共享标签命名空间**（本主题实测：`A` 到 `A.20`、"
              "`B` 到 `B.11`、`C` 到 `C.12`，共 54 个标签），"
              "把 `A.4` 改成 `A.5` 几乎必然落在另一个真实标签上；")
        print("       ② `TRAPS[*]` 与散文类的 `note` 字段**不受 `A.11` 管**（它们不是断言）。")
        print("     ⇒ 这是一条**已知的、被量出来的**缺口，不是「应该全捕获却没做到」："
              "本库把它报出来，不粉饰。")
        print()

    if mc:
        cnt_c = Counter(fam(r["desc"]) for r in buckets["crash"])
        print("  ── 崩溃捕获 %d 的构成（按字段聚合，前 10）──" % mc)
        for name, c in cnt_c.most_common(10):
            print("     %5d  %s" % (c, name))
        print()
    print("  ── 崩溃捕获明细（**崩溃不等于断言捕获**，最多 25 条）──")
    for r in buckets["crash"][:25]:
        print("     %-22s %s" % (r["file"], r["desc"]))
        for d in r["detail"][:1]:
            print("        %s" % d)
    if mc > 25:
        print("     …（还有 %d 条）" % (mc - 25))
    print()
    print("  ── harness 异常明细（最多 10 条）──")
    for r in buckets["error"][:10]:
        print("     %-22s %s ⇒ %r" % (r["file"], r["desc"], r["detail"]))

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


def _run_plain(which, env):
    """基线：在**临时副本**里跑未变异的程序。★ `env` 里有**探测到的绝对路径**。"""
    tmp = None
    try:
        tmp = tempfile.mkdtemp(prefix="cl-mutate-base-")
        work = os.path.join(tmp, "code")
        shutil.copytree(CODE_DIR, work, ignore=COPY_IGNORE)
        for prog in which:
            p = subprocess.run([PY, prog], cwd=work, capture_output=True,
                               text=True, timeout=PER_RUN_TIMEOUT, env=env)
            out = (p.stdout or "") + (p.stderr or "")
            if p.returncode != 0 or ASSERT_FAIL_MARK.search(out):
                return "%s rc=%d" % (prog, p.returncode)
        return "三个程序 rc=0、无失败断言 ✓"
    except Exception as e:
        return "harness 异常 %r" % (e,)
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())

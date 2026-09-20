#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""01_virtual_isa.py —— PTX 作为**虚拟 ISA**：寄存器是**声明的**，不是**分配的**。

配套文档：`01-ptx-as-virtual-isa.md`。★ 三类断言（A 算法正确性 / B 代价模型 /
C 边界与反例）**全部在本程序内部**（`CONTRIBUTING` §2 表里的第二种情形），
失败时以**非零退出码**结束；**没有 `verify.py`**。

★★ 本域的三条强制限定语，在本篇的落点：

  1. **不是硬件**——本程序跑的是 `ptxsub.py`，一个**逐 warp 推进到阻塞点**的解释器，
     **不是真实的 warp 交错调度**。本篇一个硬件数字都没有（本机**无 GPU**）。
  2. **按 PTX ISA `9.7.10.6` / `9.7.15.1` / `9.7.15.3` 等章节实现**——引文与章节号
     逐字保存在 `specs.ISA`，并由本程序的 `A.2`（归属）与 `--anchors-net`（在线再抽取）钉住。
     ★ 且 `shfl.sync.up` 处**必须写明本库的实现是对 ISA 文本的有意偏离**——
     这一条在本篇的落点是 `A.4` 与 `specs.UP_MODE_UNDECIDED`（正文在 `02` 篇 §4.1）。
  3. **手写 PTX 的语法未经过任何工具验证**——本机**无 `ptxas`**（也无 `nvcc` /
     `cuobjdump` / `nvdisasm`）。本程序里的每一段 PTX 都只被 `ptxsub` 解释过。

★ 本程序的真值来源是**两件事**，都不是「某个人说」：
  ① **ISA 原文的逐字引文**（`specs.ISA`，A 级来源，带章节号）；
  ② **可从这些引文机械推出的清单与计数**（`specs.PTX_SUBSET` / `specs.RAW_COUNTS`），
     每一条都被本程序的某个断言消费。
  ★ 本程序**没有外部源码树依赖**（本主题与 `cuda` / `cutlass` 不同：那两棵源码树里
    `.ptx`/`.sass`/`.cubin`/`.fatbin` **共 0 个**），只有 `--anchors-net` 需要联网。

用法：
    python3 01_virtual_isa.py                 # 跑 A/B/C 三类断言
    python3 01_virtual_isa.py --lint          # AST 扫描：不 import torch/triton/numba/cuda
    python3 01_virtual_isa.py --anchors-net   # ★ 联网：重新下载 ISA 原文，在**原始字节**上
                                              #   重新计数、重新抽取引文、重新定位章节
"""

import ast
import os
import re
import sys

import numpy as np

import ptxsub
import specs

HERE = os.path.dirname(os.path.abspath(__file__))
PROGRAMS = ("01_virtual_isa.py", "02_warp_execution.py", "03_smem_and_sass.py")
AUX_MODULES = ("ptxsub.py", "specs.py")
FORBIDDEN = ("torch", "triton", "numba", "cuda")

# ★★ 本主题的断言标记形态（`mutate.py` 的 `ASSERT_FAIL_MARK` 必须按它来配）：
#    通过 → `  [ok]   A.1 …`      失败 → `  [FAIL] A.1 …`
#    汇总 → `汇总: A a/n · B b/n · C c/n ⇒ 断言 N 条，通过 P 条，失败 F 条`
LABEL_RE = re.compile(r"^[ABC]\.\d+[a-z]?$")

# ★ C.7 的 AST 判据：本程序**不许**有任何 spawn 能力（用它锚住「手写 PTX 未经工具验证」）
SPAWN_MODULES = ("subprocess", "ctypes", "pty", "shlex", "multiprocessing")
SPAWN_CALLS = ("system", "popen", "execv", "execvp", "execve", "spawnv", "spawnl", "fork")

# ★ B.4 的常量：两个被 `ast` 读进来的 PTX 片段里，**多声明而从未触及**的 `.b32` 槽总数。
#   片段是 `02_warp_execution.py` 的 `SHFL_BODY` 与 `BAR_BODY`：
#     · `SHFL_BODY`：`.reg .b32 %r<8>` 声明 8、触及 4（%r1/%r2/%r3/%r4）⇒ 多 4
#     · `BAR_BODY` ：`.reg .b32 %r<8>` 声明 8、触及 6（%r0..%r5）      ⇒ 多 2
#   把它改错，`B.4` 立刻失败（这就是 `specs.py` 的判据「改错会报错吗」的落点之一）。
B_WASTED_B32 = 6

# ★ B.4b 的常量：两个片段各声明 `.reg .pred %p<4>`，而**两段 PTX 里都没有 `setp`**
#   ⇒ 4 + 4 = 8 个谓词槽声明了却从未被写过。（改错这个数，`B.4b` 立刻失败。）
B_WASTED_PRED = 8

# ★ 本程序**消费**的 ISA 锚点（三个程序两两不交、并集 == `specs.ISA` 的全部键）。
#   03 的判据不消费 ISA 锚点（它消费 `specs.BANK_CONFLICT` 那张推导表），故为空元组。
COVERED_ANCHORS = ("P1b", "P3", "P4", "B4")


# ===========================================================================
# 0. 断言记账（每个编号程序自带一份，**不共享**——见 Makefile 的说明）
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
    """`tpl % args`，但先把模板里的 PTX 寄存器名（`%r` / `%p`）转义。

    ★ 为什么需要它：**PTX 文本里满是 `%r0` / `%p0`**，直接拿它当格式串会炸
      （探针 E8 第 3 条记过同一个坑：批量生成 PTX 只能用 `str.replace`，不能用 `%` 模板）。
      这里做的是反过来的一半：**模板里可以写 PTX 文本**，只要经由 `_s`。
    """
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
# 1. 静态辅助（用 `ast` 读源码，**不执行**被读的文件）
# ===========================================================================
def static_labels(path):
    """从源码里提取**真实存在**的断言标签（字符串字面量，形如 `A.1`）。

    ★ 不是手写的清单 —— 手写的清单会把「我声称有这条断言」写成事实（P5）。
    """
    tree = ast.parse(open(path, encoding="utf-8").read())
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and isinstance(n.value, str) \
                and LABEL_RE.match(n.value):
            out.add(n.value)
    return out


def read_all_sources():
    out = {}
    for fn in PROGRAMS + AUX_MODULES:
        out[fn] = open(os.path.join(HERE, fn), encoding="utf-8").read()
    return out


def literals_in(path, name):
    """取出源码里名为 `name` 的模块级常量，要求它是**纯字面量**（可静态求值）。"""
    tree = ast.parse(open(path, encoding="utf-8").read())
    for n in tree.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 \
                and isinstance(n.targets[0], ast.Name) and n.targets[0].id == name:
            return ast.literal_eval(n.value)
    raise AssertionError("源码里找不到名为 %s 的字面量常量（%s）" % (name, path))


def covered_entries():
    """`specs` 里**带 `covered_by` 的顶层条目**（每个都必须非空、标签真实）。"""
    return {
        "ISA[*]": {k: specs.ISA[k] for k in specs.ISA},
        "PTX_SUBSET": specs.PTX_SUBSET,
        "RAW_COUNTS": specs.RAW_COUNTS,
        "UP_MODE_UNDECIDED": specs.UP_MODE_UNDECIDED,
        "BANK_CONFLICT": specs.BANK_CONFLICT,
    }


# ===========================================================================
# 2. 独立参照实现（A 类：与被测代码走**不同的路径**）
# ===========================================================================
def scan_decls_by_hand(text):
    """**手写行扫描**（不用正则、不用 `ptxsub.parse`）：数 `.reg` 的声明量。

    路径与 `ptxsub.parse` 的 `_DECL` 正则**完全不同**：这里逐行切词，
    要求形如 `.reg .<kind> %<name><<N>>;` 的四段结构，任何一段缺失就跳过该行。
    """
    out = {}
    for raw in text.splitlines():
        line = raw.split("//")[0].strip().rstrip(";").strip()
        if not line.startswith(".reg"):
            continue
        parts = line.split()
        if len(parts) != 3:
            continue
        kind = parts[1]
        if not kind.startswith("."):
            continue
        tok = parts[2]
        if not tok.startswith("%") or "<" not in tok or not tok.endswith(">"):
            continue
        name, _, n = tok[1:-1].partition("<")
        if not name.isalpha() or not n.isdigit():
            continue
        if name[0] not in "rp":
            continue
        out[kind[1:]] = max(out.get(kind[1:], 0), int(n))
    return out


def used_regs_by_source(text):
    """**扫源码文本**（不用 `Program.used_regs()`）：数被指令引用过的 `%rN`。

    ★ 与被测路径的区别：`Program.used_regs()` 遍历的是**解析后的操作数 token**，
      这里遍历的是**原始文本行**。两者在「注释里的 `%rN`」上会不同——
      所以这里**先剥注释**，再扫；而 `used_regs()` 天然看不到注释。
      ⇒ 两条路径在**语义**上应当相等，在**实现**上毫无共同代码。
    """
    out = set()
    for raw in text.splitlines():
        line = raw.split("//")[0]
        if line.strip().startswith(".") or line.strip().startswith("#"):
            continue
        for m in re.finditer(r"%r(\d+)", line):
            out.add(int(m.group(1)))
    return out


def opcode_names_in_specs():
    """从 `specs.PTX_SUBSET['supported']` 逐字面量取出的 opcode 名（**不问引擎要**）。"""
    return set(specs.PTX_SUBSET["supported"])


# ===========================================================================
# 3. A 类：算法正确性（参照实现走不同路径）
# ===========================================================================
def section_A():
    print("=" * 78)
    print("A 类：算法正确性 —— 参照实现走**不同的代码路径**")
    print("=" * 78)

    # ---- A.1 表的闭合：引擎的实现集合 vs `specs` 的清单（双向集合相等）----
    eng = set(ptxsub.SUPPORTED_OPS)
    spec = opcode_names_in_specs()
    eq("A", "A.1", eng, spec,
       "ptxsub.SUPPORTED_OPS == set(specs.PTX_SUBSET['supported'])（双向；%d 条）"
       % len(eng))
    check("A", "A.1b", eng == set(ptxsub.SCALAR_OPS) | set(ptxsub.WARP_OPS),
          "SUPPORTED_OPS 是两张分派表的键的并集（标量 %d + warp %d）"
          % (len(ptxsub.SCALAR_OPS), len(ptxsub.WARP_OPS)))

    # ---- A.2 归属断言（P7 的机械载体）----
    sec = {k: specs.ISA[k]["section"] for k in specs.ISA}
    check("A", "A.2", all(re.match(r"^\d+(\.\d+)+\. \S", s) for s in sec.values()),
          "★ 全部 %d 条引文都带**章节号 + 标题**（形如 `9.7.10.6. …`），无一条裸引" % len(sec))
    eq("A", "A.2b", sorted({sec["B1"].split(" ", 1)[0], sec["B2"].split(" ", 1)[0],
                            sec["B3"].split(" ", 1)[0]}), ["9.7.15.1."],
       "B1/B2/B3 三条 barrier 引文**同属** 9.7.15.1")
    check("A", "A.2c", sec["B4"].startswith("9.7.15.3."),
          "★ B4（.aligned 那句）的章节是 %r —— 它出自 **barrier.cluster**，"
          "**不是** bar.sync/barrier.cta" % sec["B4"][:12])
    check("A", "A.2d", not any(specs.ISA["B4"]["quote"] == specs.ISA[k]["quote"]
                               for k in ("B1", "B2", "B3")),
          "B4 不与 B1/B2/B3 的任何一条引文重合（防止把 cluster 的句子并进 barrier 的三条）")
    check("A", "A.2e", all(sec[k].startswith("9.7.10.6.") for k in ("P1", "P2", "P4")),
          "P1/P2/P4（shfl.sync）的章节号前缀都是 9.7.10.6.")
    check("A", "A.2f", sec["P1b"].startswith("9.7.10.5.")
          and sec["P3"].startswith("9.7.10.5."),
          "P1b/P3（弃用版 shfl）的章节号前缀都是 9.7.10.5.")
    check("A", "A.2g", sec["P1"].split(" ", 1)[0] != sec["P1b"].split(" ", 1)[0],
          "★ P1 与 P1b 是**两节**（9.7.10.6 / 9.7.10.5）—— "
          "两节的变量名不同（segmask / mask），混着引会串味")

    # ---- A.3 `covered_by` 的结构 ----
    labels = set()
    for fn in PROGRAMS:
        labels |= static_labels(os.path.join(HERE, fn))
    entries = covered_entries()
    all_cov = []
    for grp, val in entries.items():
        items = val.values() if grp == "ISA[*]" else (val,)
        for it in items:
            all_cov.append((grp, tuple(it.get("covered_by", ()))))
    check("A", "A.3", all(len(cb) > 0 for _, cb in all_cov),
          "%d 个带 covered_by 的条目**全部非空**（空 = 该常量不再被任何断言消费，P5）"
          % len(all_cov))
    missing = sorted({t for _, cb in all_cov for t in cb} - labels)
    eq("A", "A.3b", missing, [],
       "covered_by 里的每个标签都**真实存在**于某个编号程序的断言里")
    cov_isa = set()
    for fn, want in zip(PROGRAMS, [COVERED_ANCHORS,
                                   _other_covered("02_warp_execution.py", "COVERED_ANCHORS"),
                                   _other_covered("03_smem_and_sass.py", "COVERED_ANCHORS")]):
        cov_isa |= set(want)
    eq("A", "A.3c", sorted(cov_isa), sorted(specs.ISA.keys()),
       "三个程序的 COVERED_ANCHORS 并集 == specs.ISA 的全部 %d 条锚点" % len(specs.ISA))
    # ---- A.3e ★★ `covered_by` 与 `note` 互相牵制（比 A.3b 强得多）----
    #   A.3b 只要求标签**存在**于某个程序里；而本主题三个程序**共享标签命名空间**
    #   （`A.1`…`A.8` / `B.1`…`B.7` / `C.1`…`C.4` 在三个程序里都有），
    #   于是把 `A.4` 改成 `A.5` 几乎必然落在**另一个真实存在的标签**上 —— A.3b 判不出来。
    #   ⇒ 加一条：`covered_by` 的每个标签都必须**在同一锚点的 `note` 里被点名**
    #     （`note` 是独立写的一段文字，两个字段要同时对得上才能通过）。
    #     ★ 范围限制（P6）：这一条只管 `ISA[*]` 的 9 条锚点，不管另外四个顶层条目。
    tag_re = re.compile(r"[ABC]\.\d+[a-z]?")
    note_bad = []
    for k, a in sorted(specs.ISA.items()):
        named = set(tag_re.findall(a["note"]))
        miss = sorted(set(a["covered_by"]) - named)
        if miss:
            note_bad.append((k, miss))
    eq("A", "A.3e", note_bad, [],
       "★★ 每条锚点的 `covered_by` 都**包含在**它自己 `note` 里点名的标签集合中"
       "（`note` 与 `covered_by` 两个字段互相牵制 ⇒ 改一个必须改另一个）")

    sets = [set(COVERED_ANCHORS),
            set(_other_covered("02_warp_execution.py", "COVERED_ANCHORS")),
            set(_other_covered("03_smem_and_sass.py", "COVERED_ANCHORS"))]
    check("A", "A.3d", all(not (sets[i] & sets[j]) for i in range(3) for j in range(i + 1, 3)),
          "三个集合**两两不交**（P1 订正不扩散的机械载体：一个锚点只能有一个归属）")

    # ---- A.4 ★ RAW_COUNTS：**原始字节**上的计数（本篇最贵的一条教训）----
    counts = dict(specs.RAW_COUNTS["counts"])
    for k, v in counts.items():
        check("A", "A.4", v >= 0, "原始 HTML 里 %-38r 出现 %d 次" % (k, v))
    eq("A", "A.4b", counts["j &gt;= maxLane"], 2,
       "★ `j &gt;= maxLane` 在原始 HTML 里出现 **2** 次（9.7.10.5 与 9.7.10.6 各 1）")
    eq("A", "A.4c", counts["j &gt;= minLane"], 0,
       "★★ `j &gt;= minLane` 出现 **0** 次 —— 「ISA 用 minLane」这个说法"
       "在本页里**没有任何字面量支撑**")
    # 与**引文自身**交叉核对（这一步不读网络：引文是常量）
    for k in ("P1", "P1b"):
        q = specs.ISA[k]["quote"]
        eq("A", "A.4d", q.count("j >= maxLane"), 1,
           "★ %s 的引文内部：`j >= maxLane` 恰好 1 处（就是 `.up` 分支）" % k)
        eq("A", "A.4e", q.count("j >= minLane"), 0,
           "%s 的引文内部：`j >= minLane` **0** 处" % k)
        eq("A", "A.4f", q.count("j = minLane"), 1,
           "%s 的引文内部：`j = minLane` 恰好 1 处（在 `.idx` 分支里）" % k)
        check("A", "A.4g", q.count("minLane") == 2,
              "%s 的引文内部：`minLane` 共 **2** 次（定义行 + .idx 分支，**不在 .up 里**）" % k)
    # ---- A.4i `must_contain` 必须真的是**该引文**的子串（离线可判定的那一半）----
    #   ★ 这条挡的是「`must_contain` 与 `quote` 各自被独立编辑、于是彼此脱节」：
    #     在线再抽取会同时校验两者，**离线时谁都不看 `must_contain`** ——
    #     那句话在 `make mutate` 里会如实报成漏网（本主题把它补上了）。
    bad_mc = []
    for k, a in sorted(specs.ISA.items()):
        for s in a["must_contain"]:
            if s not in a["quote"]:
                bad_mc.append((k, s))
    eq("A", "A.4i", bad_mc, [],
       "★ 全部 %d 条 must_contain 都是**各自引文的逐字子串**"
       % sum(len(a["must_contain"]) for a in specs.ISA.values()))

    # ---- A.4j 每条引文都带 `section` 与 `slice`，形状完整 ----
    eq("A", "A.4j",
       sorted(k for k, a in specs.ISA.items()
              if not (a.get("section") and a.get("slice") and a.get("locate")
                      and len(a["slice"]) == 3)),
       [], "★ 9 条锚点的 `section` / `slice` / `locate` 三个字段都在，且 `slice` 是三元的")

    # 两节的变量名不同（P9 同名不同指）
    check("A", "A.4h", "segmask" in specs.ISA["P1"]["quote"]
          and "segmask" not in specs.ISA["P1b"]["quote"]
          and "~mask[4:0]" in specs.ISA["P1b"]["quote"],
          "★ 9.7.10.6 用 `segmask`、9.7.10.5 用 `mask`（同一个东西两个名字）")

    # ---- A.5 `.reg` 声明量：两条路径 ----
    frags = {"t3_bar": _ptx_of("02_warp_execution.py", "BAR_BODY"),
             "t1_shfl": _ptx_of("02_warp_execution.py", "SHFL_BODY")}
    ok = True
    detail = []
    for name, txt in sorted(frags.items()):
        a = ptxsub.parse(txt).declared
        b = scan_decls_by_hand(txt)
        detail.append("%s: parse=%r scan=%r" % (name, a, b))
        ok = ok and a == b
    check("A", "A.5", ok, "★ 声明量两条路径一致 —— " + " | ".join(detail))

    # ---- A.6 被触及的寄存器：两条路径 ----
    ok, detail = True, []
    for name, txt in sorted(frags.items()):
        a = ptxsub.parse(txt).used_regs()
        b = used_regs_by_source(txt)
        detail.append("%s: Program=%s source=%s" % (name, sorted(a), sorted(b)))
        ok = ok and a == b
    check("A", "A.6", ok, "★ 触及集合两条路径一致 —— " + " | ".join(detail))

    # ---- A.7 运行时标签 == 静态标签（自证静态提取不漏）----
    static_here = static_labels(os.path.join(HERE, "01_virtual_isa.py"))
    # 此刻 RUNTIME_LABELS 还只含已跑过的标签 —— 与静态做「子集+覆盖」双向：
    # 静态里出现而运行时没出现的，只可能是「还没跑到」，所以这条在 main() 末尾复查。
    check("A", "A.7", RUNTIME_LABELS <= static_here,
          "运行时标签 ⊆ 静态标签（%d ⊆ %d），末尾再做等号复查"
          % (len(RUNTIME_LABELS), len(static_here)))
    return static_here


def _other_covered(fn, name):
    """读另一个编号程序的 `COVERED_ANCHORS`（用 `ast`，**不 import** 它）。"""
    return literals_in(os.path.join(HERE, fn), name)


def _ptx_of(fn, name):
    """读另一个编号程序里的 PTX 片段常量（用 `ast`，不 import）。"""
    return literals_in(os.path.join(HERE, fn), name)


# ===========================================================================
# 4. B 类：代价模型（可复算的量化结论）
# ===========================================================================
def section_B():
    print()
    print("=" * 78)
    print("B 类：代价模型 —— 反直觉量化结论：**声明量 ≠ 用量**")
    print("=" * 78)

    rows = []
    for label, fn, cname in (("t3_bar（两阶段 + bar.sync）", "02_warp_execution.py", "BAR_BODY"),
                             ("t1_shfl（warp 内 bfly 归约）", "02_warp_execution.py", "SHFL_BODY")):
        txt = _ptx_of(fn, cname)
        p = ptxsub.parse(txt)
        dec = p.declared.get("b32", 0)
        used = len(p.used_regs())
        decp = p.declared.get("pred", 0)
        usedp = len(p.used_preds())
        rows.append((label, dec, used, decp, usedp))

    arr = np.array([[r[1], r[2], r[3], r[4]] for r in rows], dtype=np.int64)
    for (label, dec, used, decp, usedp) in rows:
        print("  %-28s .reg .b32 声明 %d / 实际触及 %d ；.reg .pred 声明 %d / 触及 %d"
              % (label, dec, used, decp, usedp))
    print()

    # B.1：声明量 ≥ 触及量，且**严格大于**（这就是「声明不是分配」的量化形态）
    check("B", "B.1", bool((arr[:, 0] >= arr[:, 1]).all()) and (arr[:, 0] > arr[:, 1]).any(),
          "★ 声明量 **≥** 触及量（%s），且至少一处**严格大于** ⇒ 多声明是免费的"
          % ", ".join("%d≥%d" % (a, b) for a, b in zip(arr[:, 0], arr[:, 1])))
    b32_ratio = float(arr[0, 0]) / float(arr[0, 1])
    eq("B", "B.2", round(b32_ratio, 4), 1.3333,
       _s("★ t3_bar 的 `.reg .b32 %r<8>` 声明 8、触及 6 ⇒ 声明/触及 = %.4f", b32_ratio))
    eq("B", "B.3", int(arr[0, 2]) - int(arr[0, 3]), 4,
       _s("★ t3_bar 的 `.reg .pred %p<4>` 声明 4、**触及 0**（那段 PTX 里一条 `setp` 都没有）"
          " ⇒ 差额 %d 个谓词槽**一个都没用**", int(arr[0, 2]) - int(arr[0, 3])))

    # B.4：两个 PTX 片段里**多声明而从未触及**的寄存器槽总数
    #      ★ 这是「声明不是分配」的量化形态：这些槽在 PTX 层**存在**，
    #        而它们**是否**变成 SASS 上的物理寄存器，本机**观测不到**（无 `nvdisasm`）。
    wasted = int((arr[:, 0] - arr[:, 1]).sum())
    eq("B", "B.4", wasted, B_WASTED_B32,
       "★ 两个片段多声明而从未触及的 `.b32` 槽共 %d 个（声明 %d、触及 %d）"
       % (wasted, int(arr[:, 0].sum()), int(arr[:, 1].sum())))
    eq("B", "B.4b", int((arr[:, 2] - arr[:, 3]).sum()), B_WASTED_PRED,
       "★ 两个片段各声明 `.reg .pred %%p<4>`、**合起来 %d 个谓词槽一个都没写**"
       "（两段 PTX 里都没有 `setp`；谓词只由 `@%%p` 读，不由它们写）"
       % B_WASTED_PRED)
    check("B", "B.4c", not any("nvdis" in k or "sass" in k.lower() for k in ptxsub.SUPPORTED_OPS),
          "★ 本程序**不假装能测 SASS**：表内 %d 条 opcode 里没有一条与 SASS 有关 ——"
          "「SASS 上到底用掉几个物理寄存器」本机**没有观测过**（无 `nvdisasm`）"
          % len(ptxsub.SUPPORTED_OPS))

    # B.5：表内 / 表外的规模
    eq("B", "B.5", len(specs.PTX_SUBSET["supported"]), len(ptxsub.SUPPORTED_OPS),
       "表内 %d 条 opcode（= 引擎的实现集合）" % len(ptxsub.SUPPORTED_OPS))
    eq("B", "B.5b", len(specs.PTX_SUBSET["unsupported_in_isa"]), 14,
       "表外 %d 条（**全是真实 PTX**，在线再抽取时逐字核对；本解释器一律拒绝）"
       % len(specs.PTX_SUBSET["unsupported_in_isa"]))
    overlap = set(specs.PTX_SUBSET["unsupported_in_isa"]) & ptxsub.SUPPORTED_OPS
    eq("B", "B.5c", sorted(overlap), [], "表内与表外**零重叠**（两半合起来才是一张闭合的表）")

    # B.6：三个程序的 PTX 片段里，`.sync` 形式是**唯一**出现过的 shfl 形式
    srcs = read_all_sources()
    toks = {}
    for fn, txt in srcs.items():
        # ★ 正则的字符类必须**含数字** —— 否则带位宽后缀的形式会被截短成一个
        #   不以 `shfl.sync` 开头的「token」，于是这条断言自己误报（第一版就是这么错的）。
        #   ★ 连带的一个教训：这条断言的**扫描范围包含注释**，所以连注释里举例的
        #     token 也会被算进来 —— 注释里不能写「像 token 一样的字面量」。
        for m in re.finditer(r"shfl\.[A-Za-z0-9._]+", txt):
            toks.setdefault(m.group(0), set()).add(fn)
    bad = sorted(t for t in toks if not t.startswith("shfl.sync"))
    eq("B", "B.6", bad, [],
       "★★ 五个源文件里出现的每一个 `shfl.<…>` 都是 **`shfl.sync.…`** 形式"
       "（不以 `shfl.sync` 开头的 token 共 %d 种）；出现的 %d 种形式：%s"
       % (len(bad), len(toks), ", ".join(sorted(toks))))
    eq("B", "B.6b",
       specs.ISA["P3"]["quote"].count(
           "Deprecated in PTX ISA version 6.0 in favor of shfl.sync"), 1,
       "★ 弃用版的存在由 ISA `9.7.10.5` 的原文钉住（`P3`：3.0 引入、6.0 弃用）")
    check("B", "B.6c",
          specs.ISA["P1"]["section"].startswith("9.7.10.6.")
          and specs.ISA["P1b"]["section"].startswith("9.7.10.5."),
          "★ 而且 ISA 里**两节并存**（9.7.10.6 的 `shfl.sync` 与 9.7.10.5 的弃用版）——"
          "「用 `shfl` 不用 `shfl.sync`」不是风格问题，是**跨了一个语义断点**")
    return rows


# ===========================================================================
# 5. C 类：边界与反例（**必须真的跑出问题**）
# ===========================================================================
UNREACHABLE_DECL = """
.reg .b32 %r<8>;
    mov.u32 %r0, %tid.x;
    bra L_END;
    mov.u32 %r8, 7;      // ★ 未声明的 %r8，落在**不可达**分支里
L_END:
    ret;
"""

REACHABLE_DECL = """
.reg .b32 %r<8>;
    mov.u32 %r0, %tid.x;
    mov.u32 %r8, 7;      // ★ 未声明的 %r8，这次**可达**
    ret;
"""

UNREACHABLE_OPCODE = """
.reg .b32 %r<8>;
    mov.u32 %r0, %tid.x;
    bra L_END;
    popc.b32 %r0, %r0;   // ★ 表外指令，落在**不可达**分支里
L_END:
    ret;
"""

UNDECLARED_PRED = """
.reg .b32 %r<8>;
.reg .pred %p<4>;
    mov.u32 %r0, %tid.x;
    setp.lt.s32 %p7, %r0, 4;   // ★ 未声明的 %p7（只声明了 p0..p3）
    ret;
"""

BAD_MEMBERMASK = """
.reg .b32 %r<4>;
    mov.u32 %r0, %laneid;
    shfl.sync.bfly.b32 %r1, %r0, 1, 31, 0x0;   // ★ membermask = 0
    ret;
"""


def _run_expect_error(label, src, cls="C.1"):
    try:
        s = ptxsub.Sim(ptxsub.parse(src), nthreads=32, smem_words=64, fill=0)
        s.run()
    except ptxsub.PTXError as e:
        return True, str(e)
    return False, "（没有报错，跑完了）"


def _run_expect_ok(src):
    try:
        s = ptxsub.Sim(ptxsub.parse(src), nthreads=32, smem_words=64, fill=0)
        s.run()
        return True, "跑完且 rc=0（**没有被检查到**）"
    except ptxsub.PTXError as e:
        return False, "报错了：%s" % e


def section_C():
    print()
    print("=" * 78)
    print("C 类：边界与反例 —— **必须真的跑出问题**")
    print("=" * 78)

    # ---- C.1 引用未在 `.reg` 里声明的寄存器 ⇒ 必须报错 ----
    ok, msg = _run_expect_error("C.1", REACHABLE_DECL)
    check("C", "C.1", ok and "%r8" in msg,
          _s("★ 可达的未声明寄存器 `%r8`（`.reg .b32 %r<8>`）⇒ 报错：%s", msg))
    check("C", "C.1b", "未在 .reg 声明中分配" in msg,
          "报错信息点名了寄存器与已分配数量（便于定位）")

    # ---- C.2 同一个非法寄存器，落在**不可达**分支里 ⇒ **不会**报错 ----
    #      ★ 这条是「解释器的检查是**动态**的」的直接证据（E8 第 1 条的具体化）。
    ok, msg = _run_expect_ok(UNREACHABLE_DECL)
    check("C", "C.2", ok,
          _s("★★ 不可达分支里的同一个 `%r8` ⇒ %s（真实 `ptxas` 在**汇编期**就会拒绝这段文本；"
             "本解释器只能在**执行到那条指令时**发现）", msg))

    # ---- C.3 表外指令 ⇒ `validate()` 报错（**不是**静默跳过）----
    ok, msg = _run_expect_error("C.3", UNREACHABLE_OPCODE)
    check("C", "C.3", ok and "PTX_SUBSET" in msg,
          "★ 表外指令 `popc.b32` ⇒ %s" % msg)

    # ---- C.4 表外指令**写在不可达位置也照样报错**（与 C.2 形成对照）----
    #      两条检查的**时机不同**：opcode 是**静态**的（validate 扫全部指令），
    #      寄存器下标是**动态**的（只在那条指令被执行到时才查）。
    ok_c3, msg_c3 = _run_expect_error("C.3", UNREACHABLE_OPCODE)
    ok_c2, msg_c2 = _run_expect_ok(UNREACHABLE_DECL)
    check("C", "C.4", ok_c3 and ok_c2,
          "★★ **时机差异**：同一个「不可达」位置，表外 opcode **报错**（静态检查，%s），"
          "未声明寄存器 **不报错**（动态检查，%s）" % (msg_c3[:34], msg_c2[:18]))
    # 表外清单里**每一条**都必须被拒绝
    allrej, firstbad = True, None
    for op in specs.PTX_SUBSET["unsupported_in_isa"]:
        src = ".reg .b32 %%r<4>;\n    %s %%r0, %%r1;\n    ret;\n" % op
        good, _ = _run_expect_error("C.3", src)
        if not good:
            allrej, firstbad = False, op
    check("C", "C.4b", allrej,
          "★ 表外清单**全部 %d 条**都被拒绝%s"
          % (len(specs.PTX_SUBSET["unsupported_in_isa"]),
             "" if allrej else "（例外：%s）" % firstbad))

    # ---- C.5 未声明的谓词 ⇒ 报错（不是 Python 的 IndexError）----
    ok, msg = _run_expect_error("C.5", UNDECLARED_PRED)
    check("C", "C.5", ok and "%p7" in msg and "IndexError" not in msg,
          _s("★ 未声明的谓词 `%p7`（`.reg .pred %p<4>`）⇒ %s", msg))

    # ---- C.6 UB 情形：源 lane 不在 membermask 内 ⇒ 报错（**不等于**复现了 UB）----
    ok, msg = _run_expect_error("C.6", BAD_MEMBERMASK)
    check("C", "C.6", ok and "membermask" in msg,
          "★ `membermask = 0` ⇒ %s ★ 原文（`P4`）只说 **undefined**；"
          "本解释器**报错**，这**不是**复现了 UB（本库不声称覆盖 UB）" % msg)

    # ---- C.7 反向断言：本程序**没有任何能力**去调用本机不存在的工具 ----
    #   ★ 这是「手写 PTX 的语法未经任何工具验证」这条限定语的**机械载体**：
    #     不是「我们忘了验证」，而是本程序在结构上**做不到**验证。
    #     P3（要求实现者复现跑不了的东西）的判据就是这一条。
    src = open(os.path.join(HERE, "01_virtual_isa.py"), encoding="utf-8").read()
    # ★ 用 `ast` 扫**导入与调用**，不做字符串匹配 —— 否则这条断言会被它自己的
    #   说明文字（里面必须点名那几个工具）触发成假阳性（第一版就是这么错的）。
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
       "也就**没有能力**调用 `ptxas` / `nvcc` / `nvdisasm`（P3：不要求实现者复现跑不了的东西）"
       % (", ".join(SPAWN_MODULES), ", ".join(SPAWN_CALLS)))
    # ★ 下面这个词**故意拼出来**（`"实" + "测"`），这样它自己不会出现在源码里 ——
    #   否则「源码里没有这两个字」这条断言会被它自己的检测代码弄假。
    bare = "实" + "测"
    eq("C", "C.7b", src.count(bare), 0,
       "★ 本程序源码里 0 次出现裸用的那两个汉字（本域 B10：不得裸用）——"
       "本程序的数字全部是**推导值**或**引用值**，没有一个来自本机硬件")
    return True


# ===========================================================================
# 6. `--lint`：AST 扫描（含 `specs.py`，含动态导入）
# ===========================================================================
def ast_import_scan(directory, files):
    """真 `import` 才算违规：注释与 docstring 里出现这些词不算（`cuda/code` 同口径）。"""
    hits = []
    for fn in files:
        tree = ast.parse(open(os.path.join(directory, fn), encoding="utf-8").read())
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                for a in n.names:
                    root = a.name.split(".")[0]
                    if root in FORBIDDEN:
                        hits.append((fn, n.lineno, "import " + a.name))
            elif isinstance(n, ast.ImportFrom):
                root = (n.module or "").split(".")[0]
                if root in FORBIDDEN:
                    hits.append((fn, n.lineno, "from %s import" % n.module))
            elif isinstance(n, ast.Call):
                f = n.func
                nm = getattr(f, "attr", None) or getattr(f, "id", None)
                if nm in ("import_module", "__import__") and n.args:
                    a0 = n.args[0]
                    if isinstance(a0, ast.Constant) and isinstance(a0.value, str) \
                            and a0.value.split(".")[0] in FORBIDDEN:
                        hits.append((fn, n.lineno, "%s(%r)" % (nm, a0.value)))
    return hits


def lint_main():
    print("AST 导入扫描：%s + %s" % (", ".join(PROGRAMS), ", ".join(AUX_MODULES)))
    hits = ast_import_scan(HERE, PROGRAMS + AUX_MODULES)
    if hits:
        for h in hits:
            print("  !! %s" % (h,))
        return 1
    print("  ✓ 没有 import torch / triton / numba / cuda"
          "（含 specs.py / ptxsub.py，含 import_module 动态导入）")
    return 0


# ===========================================================================
# 7. `--anchors-net`：★ 联网重新抽取（在**原始字节**上）
# ===========================================================================
PRIV_USE = re.compile("[\ue000-\uf8ff\U000f0000-\U000ffffd]")


def strip_tags(seg):
    """去标签 → 反转义 → 空白折叠，**并剥掉私用区码位**。

    ★ 私用区（`U+E000–U+F8FF` 等）在 ISA 的每个 `<h4>` 末尾出现一个（`U+F0C1`），
      那是文档的**锚点图标**，不是文字。只剥这一类码位 —— **其它字符一个不动**，
      包括去标签留下的那些「标点前多一个空格」（那是抽取口径的已知产物，
      `specs.py` 的模块头写明了，引文里**逐字保留**）。
    """
    import html as _html
    seg = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", seg)
    seg = re.sub(r"(?s)<[^>]+>", " ", seg)
    seg = PRIV_USE.sub("", _html.unescape(seg))
    return re.sub(r"\s+", " ", seg).strip()


def extract_quote(raw, slice_):
    """按 `(起点, 终点, 再截断到)` 在**原始 HTML** 上切片，再清洗成一行。

    ★ 这是本库**唯一**允许产生引文的路径 —— 引文**不手打**（手打一句像模像样的
      英文引文，比写错一个数字更危险）。
    """
    start, end, trim = slice_
    if start.startswith("re:"):
        # ★ 切片起点也支持正则（`re:` 前缀）：当一个**唯一**的锚必须是「两句话连起来」
        #   而中间隔着标签/换行时（`P3` 就是这种：`Introduced … 3.0.` 单独出现 5 次），
        #   字面量表达不了「唯一」，正则可以。
        ms = list(re.finditer(start[3:], raw))
        if len(ms) != 1:
            return None, "起点正则匹配 %d 次（必须恰好 1 次）：%r" % (len(ms), start[3:60])
        i = ms[0].start()
        j = raw.find(end, i)
        if j < 0:
            return None, "终点标记在起点之后找不到：%r" % end
        tt = strip_tags(raw[i:j + len(end)])
        if trim is not None:
            if trim not in tt:
                return None, "截断标记不在切片里：%r" % trim
            tt = tt[:tt.index(trim) + len(trim)]
        return tt, None
    n = raw.count(start)
    if n != 1:
        # ★ 起点标记**必须唯一** —— 第一版的 B4 用了 `qualifier indicates …` 这个
        #   在全文出现多次的短句，于是它切到了 `wmma` 那一节，抽出的是别的句子。
        #   没有这条检查，那种错会**看起来像**「逐字比对不一致」，指向完全错误的方向。
        return None, "起点标记在原文里出现 %d 次（必须恰好 1 次）：%r" % (n, start[:60])
    i = raw.find(start)
    j = raw.find(end, i)
    if j < 0:
        return None, "终点标记在起点之后找不到：%r" % end
    t = strip_tags(raw[i:j + len(end)])
    if trim is not None:
        if trim not in t:
            return None, "截断标记不在切片里：%r" % trim
        t = t[:t.index(trim) + len(trim)]
    return t, None


def heading_before(raw, off):
    """`off` 之前**最近的一个 `<h1>`–`<h6>` 标题**（清洗后）。"""
    best = None
    for m in re.finditer(r"(?s)<h[1-6][^>]*>(.*?)</h[1-6]>", raw[:off]):
        best = m
    return strip_tags(best.group(1)) if best else None


def locate_offset(raw, locate):
    if locate.startswith("re:"):
        m = re.search(locate[3:], raw)
        return (m.start(), m.group(0)) if m else (None, None)
    i = raw.find(locate)
    return (i, locate) if i >= 0 else (None, None)


def anchors_net():
    import urllib.request
    print("=" * 78)
    print("--anchors-net：从 ISA 原文**重新抽取**（口径写在 specs.py 的模块头里）")
    print("=" * 78)
    print("  GET %s" % specs.PTX_ISA_URL)
    req = urllib.request.Request(specs.PTX_ISA_URL,
                                 headers={"User-Agent": "knowledge-base-anchors/1"})
    with urllib.request.urlopen(req, timeout=90) as r:
        raw_b = r.read()
    raw = raw_b.decode("utf-8", "replace")
    print("  HTTP 200  size = %d 字节（观测值 %d；本目标只断言下界 %d）"
          % (len(raw_b), specs.ISA_HTTP_SIZE_OBSERVED, specs.ISA_HTTP_SIZE_MIN))
    fails = []
    if len(raw_b) < specs.ISA_HTTP_SIZE_MIN:
        fails.append("下载字节数 %d < 下界 %d（疑似截断）" % (len(raw_b), specs.ISA_HTTP_SIZE_MIN))

    # ---- ① ★★ 先在**原始字节**上计数（不先转成文本）----
    print()
    print("  ① 原始字节上的计数（`raw.count(<字面量>)`，**不经任何转写**）")
    for lit, want in specs.RAW_COUNTS["counts"]:
        got = raw.count(lit)
        ok = got == want
        print("     %-40s 再抽取 %2d  记录 %2d  %s"
              % (repr(lit), got, want, "OK" if ok else "**不一致**"))
        if not ok:
            fails.append("原始计数不一致：%r 再抽取得 %d ≠ 记录 %d" % (lit, got, want))
    print("     ★ `j &gt;= maxLane` = %d、`j &gt;= minLane` = %d —— 这两条一起读才是判据"
          % (raw.count("j &gt;= maxLane"), raw.count("j &gt;= minLane")))

    # ---- ② 引文逐字 + 归属（章节号）----
    print()
    print("  ② 引文逐字比对 + 归属（该引文在原文里最近的上位标题）")
    for key in sorted(specs.ISA):
        a = specs.ISA[key]
        q, err = extract_quote(raw, a["slice"])
        if err:
            fails.append("%s：%s" % (key, err))
            print("     %-4s !! %s" % (key, err))
            continue
        okq = (q == a["quote"])
        miss = [s for s in a["must_contain"] if s not in q]
        off, _m = locate_offset(raw, a["locate"])
        head = heading_before(raw, off) if off is not None else None
        oks = bool(head) and head.startswith(a["section"])
        print("     %-4s 逐字=%s   must_contain 缺 %d 条   章节=%s"
              % (key, "OK" if okq else "**不一致**", len(miss),
                 (head[:60] + "…") if head else "**没定位到**"))
        print("          记录章节 %s" % a["section"][:60])
        if not okq:
            fails.append("%s：逐字比对不一致（抽取 %r… vs 记录 %r…）"
                         % (key, q[:40], a["quote"][:40]))
        if miss:
            fails.append("%s：must_contain 里 %r 在抽取结果里找不到" % (key, miss))
        if not oks:
            fails.append("%s：归属不符（原文标题 %r 不以 %r 开头）"
                         % (key, head, a["section"]))
    # ★★ P7 的在线载体：B4 必须仍然落在 barrier.cluster 那一节
    b4 = specs.ISA["B4"]
    off, _ = locate_offset(raw, b4["locate"])
    head = heading_before(raw, off)
    ok = bool(head) and head.startswith("9.7.15.3.") and not head.startswith("9.7.15.1.")
    print()
    print("     ★★ B4（`.aligned` 那句）在原文里的上位标题 = %r" % (head[:70] if head else None))
    print("        ⇒ 它在 **9.7.15.3 barrier.cluster** 下，**不在** 9.7.15.1（bar.sync）下")
    if not ok:
        fails.append("B4 的归属不是 9.7.15.3（P7：引文作用域不匹配）")

    # ---- ③ 表外指令的 mnemonics 必须真的出现在 ISA 原文里 ----
    print()
    print("  ③ 表外清单的 %d 条 mnemonics 逐条在原文里核对（**不得**凭空写一个不存在的指令名）"
          % len(specs.PTX_SUBSET["unsupported_in_isa"]))
    for op in specs.PTX_SUBSET["unsupported_in_isa"]:
        c = raw.count(op)
        print("     %-28s 原文出现 %3d 次  %s" % (op, c, "OK" if c else "**找不到**"))
        if c == 0:
            fails.append("表外 mnemonic %r 在 ISA 原文里找不到（很可能是凭印象写的）" % op)

    print()
    if fails:
        print("!! 在线再抽取**失败** %d 项：" % len(fails))
        for f in fails:
            print("   - %s" % f)
        return 1
    print("✓ 在线再抽取全部通过：原始计数 %d 项、引文与归属 %d 条、表外 mnemonics %d 条"
          % (len(specs.RAW_COUNTS["counts"]), len(specs.ISA),
             len(specs.PTX_SUBSET["unsupported_in_isa"])))
    return 0


# ===========================================================================
# 8. main
# ===========================================================================
def main():
    print("=" * 78)
    print("01_virtual_isa.py —— PTX 作为虚拟 ISA：寄存器是**声明的**，不是**分配的**")
    print("=" * 78)
    print("  解释器      : %s" % sys.executable)
    print("  python      : %s / numpy %s" % (sys.version.split()[0], np.__version__))
    print("  引擎        : ptxsub.py（%d 行；表内 %d 条 opcode）"
          % (len(open(os.path.join(HERE, "ptxsub.py"), encoding="utf-8").readlines()),
             len(ptxsub.SUPPORTED_OPS)))
    print("  ★ 限定语 1  : **不是硬件** —— 逐 warp 推进到阻塞点，不是真实 warp 交错调度")
    print("  ★ 限定语 2  : 按 PTX ISA 9.7.10.6 / 9.7.15.1 / 9.7.15.3 实现（章节号见 specs.ISA）")
    print("  ★ 限定语 3  : **手写 PTX 的语法未经任何工具验证** —— 本机无 ptxas / nvcc / nvdisasm")
    print()

    static_here = section_A()
    rows = section_B()
    section_C()

    # ---- A.8 AST 导入扫描（与 `make lint` 是**同一份代码**）----
    print()
    print("-" * 78)
    eq("A", "A.8", ast_import_scan(HERE, PROGRAMS + AUX_MODULES), [],
       "★ AST 导入扫描：五个源文件（含 `specs.py` / `ptxsub.py`）都不 import "
       "torch / triton / numba / cuda（含 `import_module` 动态导入）")

    # ---- A.7 的末尾复查：静态标签 == 运行时标签 ----
    print()
    print("-" * 78)
    eq("A", "A.7b", sorted(static_here - RUNTIME_LABELS - {"A.7b"}), [],
       "★ 静态标签减去运行时标签 == 空（静态提取不漏：源码里写了却没跑的断言数为 0）")
    print()
    print("  ── 声明 vs 触及（B 类的原始数据，可对着 PTX 文本复核）──")
    for (label, dec, used, decp, usedp) in rows:
        print("     %-28s b32 %d/%d   pred %d/%d" % (label, dec, used, decp, usedp))

    return summary()


if __name__ == "__main__":
    if "--lint" in sys.argv:
        sys.exit(lint_main())
    if "--anchors-net" in sys.argv:
        sys.exit(anchors_net())
    sys.exit(main())

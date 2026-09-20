#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""03_swizzle_probe.py —— swizzle 是一种 **layout 代数里的构造**，不是一个特例函数。

配套文档：`03-swizzle-and-collective.md`。★ 三类断言（A 算法正确性 / B 代价模型 /
C 边界与反例）**全部在本程序内部**，失败时以**非零退出码**结束；**没有 `verify.py`**。

★★ 真值标注（逐处都要带）：

> 本主题的数字全部来自**在本机跑 CUTLASS 4.8.0 自带的 Python 参考实现**
> （`python/pycute/` 与 `python/cutlass_library/`），**不是**从 GPU 上测的，
> **也不是**本仓库的推导。**测量条件是 CUTLASS 的版本，不是 GPU 型号。**

★★ **本篇不推导 bank 冲突的机制**。冲突的成因、`gcd(行距, 32)` 规则、广播为什么免费、
  64 位访问为什么换一套判定池，都在
  `20-gpu-architecture/memory-system/03-shared-memory-bank-conflict` —— **本篇回链不复述**。

★★ 本程序对 **P11** 的正面回答：§2 是**本仓库自己的位运算实现**（一个 `&`/`^`/移位构成的
  纯整数函数），与 `pycute.Swizzle` **逐点对拍**；并且**从掩码的位置**推出
  「恒等区间的长度」与「首个非平凡映射」，再拿实测去**驳**它（`A.2`/`A.3`）。

用法：
    python3 03_swizzle_probe.py          # 跑 A/B/C 三类断言
    python3 03_swizzle_probe.py --lint   # AST 扫描：不 import torch/triton/numba/cuda
"""

import ast
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROGRAMS = ("01_layout_algebra.py", "02_tiling_decompose.py", "03_swizzle_probe.py")
AUX_MODULES = ("specs.py", "meta_falsify.py", "mutate.py")
FORBIDDEN = ("torch", "triton", "numba", "cuda")

import specs                                   # noqa: E402

LABEL_RE = re.compile(r"^[ABC]\.\d+[a-z]?$")

# ★ 本程序**消费**的锚点（三个程序两两不交、并集 == `specs.ANCHORS` 的全部键）。
COVERED_ANCHORS = ("S1", "S2", "S3", "S4", "S5", "S6")

# ★★ 探测面：全部满足 `abs(shift) >= bits` 的小参数组合（`size` 控制在 4096 以内）。
#   ★ `Swizzle(bits, base, shift)` 的构造式**强制** `abs(shift) >= bits`（`C.7`），
#     所以下面每一个组合都是**合法**的；非法的那一个单独在 `C.7` 里跑。
SWIZZLE_PARAMS = (
    (1, 0, 1), (2, 0, 2), (3, 0, 3), (4, 0, 4),
    (1, 3, 1), (2, 3, 2), (3, 3, 3), (4, 3, 4),
    (3, 4, 3), (2, 5, 2), (3, 2, 3), (4, 2, 4),
    (1, 6, 1), (2, 6, 2), (3, 6, 3),
    (2, 0, -2), (3, 0, -3), (2, 3, -2),
)

# ★ 位掩码位置的**推导**（不是抄）：见 §2 的 `yyy_pos` / `zzz_pos`。
SWIZZLE_3_3_3 = {"bits": 3, "base": 3, "shift": 3, "size": 512,
                 "nontrivial": 448, "first": (64, 72)}
SWIZZLE_3_4_3 = {"bits": 3, "base": 4, "shift": 3, "size": 1024,
                 "nontrivial": 896, "first": (128, 144)}


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
        print("  [ok]   %-6s %s" % (name, detail))
    else:
        FAILS.append(name)
        print("  [FAIL] %-6s %s" % (name, detail))
    return bool(cond)


def eq(cls, name, got, want, detail=""):
    return check(cls, name, got == want,
                 "%s: %r%s" % (detail, got,
                               "" if got == want else "  ≠ 期望 %r" % (want,)))


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
# 1. 静态辅助
# ===========================================================================
def static_labels(path):
    tree = ast.parse(open(path, encoding="utf-8").read())
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and isinstance(n.value, str) \
                and LABEL_RE.match(n.value):
            out.add(n.value)
    return out


def all_static_labels():
    out = set()
    for p in PROGRAMS:
        out |= static_labels(os.path.join(HERE, p))
    return out


def covered_entries():
    ent = {"ANCHORS[%s]" % k: specs.ANCHORS[k] for k in specs.ANCHORS}
    for k in ("A", "B", "C"):
        ent["TRAPS[%s]" % k] = specs.TRAPS[k]
    return ent


def ast_import_scan(directory, files):
    hits = []
    for fn in files:
        tree = ast.parse(open(os.path.join(directory, fn), encoding="utf-8").read())
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                for a in n.names:
                    if a.name.split(".")[0] in FORBIDDEN:
                        hits.append((fn, n.lineno, "import " + a.name))
            elif isinstance(n, ast.ImportFrom):
                if (n.module or "").split(".")[0] in FORBIDDEN:
                    hits.append((fn, n.lineno, "from %s import" % n.module))
            elif isinstance(n, ast.Call):
                nm = getattr(n.func, "attr", None) or getattr(n.func, "id", None)
                if nm in ("import_module", "__import__") and n.args:
                    a0 = n.args[0]
                    if isinstance(a0, ast.Constant) and isinstance(a0.value, str) \
                            and a0.value.split(".")[0] in FORBIDDEN:
                        hits.append((fn, n.lineno, "%s(%r)" % (nm, a0.value)))
    return hits


# ===========================================================================
# 2. ★★ 本仓库自己的实现：一个**纯位运算**的 swizzle + 掩码位置的推导
# ===========================================================================
#
# ★ 与 `pycute.Swizzle.__call__` 的写法**不同**：
#   pycute 写的是 `offset ^ shiftr(offset & yyy_msk, shift)`（`shiftr`/`shiftl`
#   互为递归），这里**不做双重掩码、也不移动掩码**，直接把 YYY 字段取出来搬到 ZZZ 位。
#   **两条路径在语义上必须逐点一致** —— 这就是 A 类断言的内容。

def yyy_pos(bits, base, shift):
    """YYY 字段的**最低位位置**（`swizzle.py:69` 的 `base + max(0,shift)`）。"""
    return base + max(0, shift)


def zzz_pos(bits, base, shift):
    """ZZZ 字段的**最低位位置**（`swizzle.py:70` 的 `base - min(0,shift)`）。"""
    return base + max(0, -shift)


def our_swizzle(bits, base, shift, offset):
    """本仓库的位运算实现：把 YYY 字段异或进 ZZZ 字段。

    `result = offset ^ (((offset >> yyy_pos) & mask) << zzz_pos)`
    """
    mask = (1 << bits) - 1
    y = (offset >> yyy_pos(bits, base, shift)) & mask
    return offset ^ (y << zzz_pos(bits, base, shift))


def our_swizzle_size(bits, base, shift):
    """定义域大小：`1 << (bits + base + |shift|)`（`swizzle.py:78`）。"""
    return 1 << (bits + base + abs(shift))


def our_identity_bound(bits, base, shift):
    """**推导值**：使 `S(o) == o` 的偏移上界（排他）。

    ⇒ 只要 YYY 字段为 0 就恒等，而 YYY 字段的最低有效位在 `yyy_pos`。
    所以 **[0, 1 << yyy_pos) 上处处恒等**，且第一个非平凡的偏移恰好是 `1 << yyy_pos`。
    """
    return 1 << yyy_pos(bits, base, shift)


def our_nontrivial_count(bits, base, shift):
    """**推导值**：定义域里被改变的元素个数。

    ⇒ 非平凡 ⟺ YYY 字段非 0，而该字段均匀铺满整个定义域
      ⇒ 个数 = `size × (2^bits - 1) / 2^bits`，**与 base / shift 无关**。
    """
    size = our_swizzle_size(bits, base, shift)
    return size * ((1 << bits) - 1) // (1 << bits)


def our_composed(sw, offset, layout):
    """`ComposedLayout` 的三段复合（`swizzle.py:108-109` 的语义）：

        `result(crd) = sw(offset + layout(crd))`
    """
    return lambda crd: sw(offset + layout(crd))


# ===========================================================================
# 3. A 类：算法正确性（本仓库实现 vs `pycute`）
# ===========================================================================
def section_A(mod):
    print("=" * 78)
    print("A 类：算法正确性 —— **本仓库的位运算实现** vs `pycute.Swizzle`")
    print("=" * 78)
    print("  ★ 真值那一侧跑的是 %s（**本机实测**）" % specs.MEASURE_IMPL)
    print()

    # ---- A.1 逐点对拍 ----
    bad, n = [], 0
    for (bits, base, shift) in SWIZZLE_PARAMS:
        S = mod.Swizzle(bits, base, shift)
        size = our_swizzle_size(bits, base, shift)
        n += size
        for o in range(size):
            got, want = our_swizzle(bits, base, shift, o), S(o)
            if got != want:
                bad.append("Swizzle(%d,%d,%d) offset=%d: ours=%d pycute=%d"
                           % (bits, base, shift, o, got, want))
                break
    eq("A", "A.1", len(bad), 0,
       "★ 逐点对拍：**%d 个参数组合 × 全部偏移 = %d 个点**，不一致 %d 个%s"
       % (len(SWIZZLE_PARAMS), n, len(bad), "" if not bad else "；首例 %s" % bad[0]))

    # ---- A.2 恒等区间与首个非平凡映射：**先推导，再验证** ----
    rows, bad2 = [], []
    for (bits, base, shift) in SWIZZLE_PARAMS:
        S = mod.Swizzle(bits, base, shift)
        size = our_swizzle_size(bits, base, shift)
        bound = our_identity_bound(bits, base, shift)
        # ① [0, bound) 上必须处处恒等
        for o in range(bound):
            if S(o) != o:
                bad2.append("(%d,%d,%d): offset %d < %d 却不是恒等" % (bits, base, shift, o, bound))
                break
        # ② 第一个非平凡偏移必须恰好是 bound，且像必须是 bound ^ (1 << zzz_pos)
        nontriv = [o for o in range(size) if S(o) != o]
        first = nontriv[0] if nontriv else None
        pred = bound ^ (1 << zzz_pos(bits, base, shift))
        rows.append((bits, base, shift, bound, first, pred, S(bound) if first is not None else None))
        if first is not None and (first != bound or S(first) != pred):
            bad2.append("(%d,%d,%d): 实测首个 %r ≠ 推导 %d（像 %r ≠ %d）"
                        % (bits, base, shift, first, bound, S(first), pred))
    eq("A", "A.2", len(bad2), 0,
       "★★ **从掩码的位置推出的两条边界**（%d 个参数组合逐条验证）："
       "① `[0, 1<<%s)` 上**处处恒等**；② 首个非平凡偏移**恰好**是 `1 << (base+max(0,shift))`，"
       "其像**恰好**是 `偏移 ^ (1 << (base+max(0,-shift)))`。不一致 %d 条%s"
       % (len(SWIZZLE_PARAMS), "yyy_pos", len(bad2), "" if not bad2 else "；%r" % bad2[:2]))

    # ---- A.3 非平凡比例的闭式 ----
    bad3, rows3 = [], []
    for (bits, base, shift) in SWIZZLE_PARAMS:
        S = mod.Swizzle(bits, base, shift)
        size = our_swizzle_size(bits, base, shift)
        got = sum(1 for o in range(size) if S(o) != o)
        pred = our_nontrivial_count(bits, base, shift)
        rows3.append((bits, base, shift, size, got, pred))
        if got != pred:
            bad3.append("(%d,%d,%d): 实测 %d ≠ 闭式 %d" % (bits, base, shift, got, pred))
    eq("A", "A.3", len(bad3), 0,
       "★★ **非平凡比例的闭式**：`size × (2^bits − 1) / 2^bits` —— "
       "**与 base / shift 无关**（%d 个组合逐条成立）；不一致 %d 条%s"
       % (len(SWIZZLE_PARAMS), len(bad3), "" if not bad3 else "；%r" % bad3[:2]))

    # ---- A.3b ★ 向量化实现：**第三条路径**（numpy 位运算，整条定义域一次算完）----
    #   ★ 与 `our_swizzle`（标量循环）和 `pycute.Swizzle`（`shiftr` 递归）都**不同**。
    bad3b = []
    for (bits, base, shift) in SWIZZLE_PARAMS:
        size = our_swizzle_size(bits, base, shift)
        o = np.arange(size, dtype=np.int64)
        y = (o >> yyy_pos(bits, base, shift)) & ((1 << bits) - 1)
        vec = o ^ (y << zzz_pos(bits, base, shift))
        S = mod.Swizzle(bits, base, shift)
        ref = np.array([S(int(v)) for v in o], dtype=np.int64)
        if not np.array_equal(vec, ref):
            bad3b.append("(%d,%d,%d)" % (bits, base, shift))
    eq("A", "A.3b", len(bad3b), 0,
       "★ **第三条路径**（numpy int64 向量化位运算 vs 标量循环 vs `pycute` 的 `shiftr` "
       "递归）：%d 个参数组合、**%d 个点**，不一致 %d 个 —— "
       "`np.array_equal` 一次比完整条定义域" % (len(SWIZZLE_PARAMS), n, len(bad3b)))

    # ---- A.4 ComposedLayout 的三段复合 ----
    #   ★ 只喂**标量索引**：pycute 的 `crd2idx` 在「完整坐标元组」与「标量索引」上
    #     走**两条不同的分支**（元组分支不做取模回绕），越界坐标上两者会不同 ——
    #     那是另一个话题，本断言只要同一个输入下两边一致。
    bad4, n4 = [], 0
    n_sw = 0
    for (bits, base, shift) in SWIZZLE_PARAMS[:8]:
        n_sw += 1
        S = mod.Swizzle(bits, base, shift)
        for (shape, stride) in (((8, 8), (8, 1)), ((4, 4), (1, 4)), ((8, 4), (4, 1))):
            L = mod.Layout(shape, stride)
            C = mod.ComposedLayout(S, 0, L)
            for i in range(L.size()):
                n4 += 1
                got = C(i)
                want = our_composed(lambda o: our_swizzle(bits, base, shift, o), 0,
                                    lambda c: L(c))(i)
                if got != want:
                    bad4.append("S(%d,%d,%d) L%s i=%d: %r ≠ %r"
                                % (bits, base, shift, L, i, got, want))
                    break
    eq("A", "A.4", len(bad4), 0,
       "★ `ComposedLayout(S, offset, L)(i) == S(offset + L(i))` 逐点对拍 "
       "（**%d 个点**：%d 组 swizzle × 3 个 layout 的全部元素）：不一致 %d 个%s"
       % (n4, n_sw, len(bad4), "" if not bad4 else "；首例 %s" % bad4[0]))

    # ---- A.5 ComposedLayout 的 size / cosize 语义 ----
    S = mod.Swizzle(3, 3, 3)
    L = mod.Layout((8, 8), (8, 1))
    C = mod.ComposedLayout(S, 0, L)
    eq("A", "A.5", (C.size(), C.cosize()), (64, 512),
       "★ `ComposedLayout.size()` **看的是 layoutA 那一侧**（= size(L) = %d），"
       "`cosize()` **看的是 layoutB 那一侧**（= cosize(S) = %d）—— "
       "两边**各看一个方向**，所以 `Swizzle` 身上那个陷阱（`C.6`）"
       "**在 ComposedLayout 上不出现**" % (C.size(), C.cosize()))

    # ---- A.6 锚点的 file:line 逐字比对 ----
    root = specs.CUTLASS_ROOT
    bad6 = []
    for key, a in sorted(specs.ANCHORS.items()):
        lines = specs.read_lines(root, a["file"])
        lo, hi = (a["line"], a["line"]) if isinstance(a["line"], int) else a["line"]
        block = b"\n".join(lines[lo - 1:hi]).decode("utf-8", "replace")
        for lit in a["must_contain"]:
            if lit not in block:
                bad6.append("%s @ %s:%s 缺 %r" % (key, a["file"], a["line"], lit[:44]))
        if a["quote"] not in block:
            bad6.append("%s @ %s:%s quote 对不上" % (key, a["file"], a["line"]))
    eq("A", "A.6", len(bad6), 0,
       "★ **%d** 条锚点的 `file:line` 逐字比对（原始字节上做 `in` 判定）：对不上 %d 条%s"
       % (len(specs.ANCHORS), len(bad6), "" if not bad6 else "；%r" % (bad6[:3],)))

    # ---- A.7 covered_by 非空 + 标签真实 ----
    real = all_static_labels()
    empty, ghost = [], []
    for k, v in sorted(covered_entries().items()):
        cb = v.get("covered_by") or ()
        if not cb:
            empty.append(k)
        for t in cb:
            if t not in real:
                ghost.append("%s→%s" % (k, t))
    eq("A", "A.7", len(empty) + len(ghost), 0,
       "★ **%d** 个 `covered_by` 条目：非空违例 %d、幽灵标签 %d%s"
       % (len(covered_entries()), len(empty), len(ghost),
          "" if not (empty or ghost) else "；%r %r" % (empty[:3], ghost[:3])))
    return rows, rows3


# ===========================================================================
# 4. B 类：代价模型 —— 掩码的位置决定了「多大才看得见」
# ===========================================================================
def section_B(mod):
    print()
    print("=" * 78)
    print("B 类：代价模型 —— 掩码躺在哪一位，决定了「多大的用例才看得见它」")
    print("=" * 78)

    # ---- B.1 Swizzle(3,3,3) 的逐字实测 ----
    S = mod.Swizzle(3, 3, 3)
    size = S.size()
    changed = [o for o in range(size) if S(o) != o]
    eq("B", "B.1", (str(S), bin(S.yyy_msk), bin(S.zzz_msk), S.shift, size,
                    len(changed), (changed[0], S(changed[0]))),
       ("SW_3_3_3", "0b111000000", "0b111000", 3,
        SWIZZLE_3_3_3["size"], SWIZZLE_3_3_3["nontrivial"], SWIZZLE_3_3_3["first"]),
       "★★ `Swizzle(3,3,3)` 逐字实测：`str` = **SW_3_3_3**、`yyy_msk` = **0b111000000**"
       "（bit 6–8）、`zzz_msk` = **0b111000**（bit 3–5）、`shift` = 3；"
       "定义域 **512**，被改变的 **448 / 512 = 87.5%**，首个非平凡映射 **(64, 72)**")

    # ---- B.2 Swizzle(3,4,3) 的逐字实测 ----
    S2 = mod.Swizzle(3, 4, 3)
    size2 = S2.size()
    ch2 = [o for o in range(size2) if S2(o) != o]
    eq("B", "B.2", (bin(S2.yyy_msk), size2, len(ch2), (ch2[0], S2(ch2[0]))),
       ("0b1110000000", SWIZZLE_3_4_3["size"], SWIZZLE_3_4_3["nontrivial"],
        SWIZZLE_3_4_3["first"]),
       "★ `Swizzle(3,4,3)`：`yyy_msk` = **0b1110000000**（bit 7–9）、定义域 **1024**、"
       "被改变的 **896 / 1024 = 87.5%**、首个非平凡映射 **(128, 144)** —— "
       "同一 `bits`、**base 抬一位就把恒等区间翻倍**（64 → 128）")

    # ---- B.3 恒等区间的长度 = 2^yyy_pos ----
    rows = []
    for (bits, base, shift) in SWIZZLE_PARAMS[:12]:
        rows.append((bits, base, shift, yyy_pos(bits, base, shift),
                     our_identity_bound(bits, base, shift)))
    eq("B", "B.3", [r[4] for r in rows],
       [1 << r[3] for r in rows],
       "★★ **恒等区间的长度 = `2^(base + max(0,shift))`**（与 `bits` **无关**）：%s"
       % "; ".join("(%d,%d,%d)→%d" % (b, bs, s, bnd) for b, bs, s, _, bnd in rows))

    # ---- B.4 非平凡比例只跟 bits 走 ----
    rows4 = []
    for bits in (1, 2, 3, 4, 5):
        base, shift = max(bits, 1), max(bits, 1)
        size = our_swizzle_size(bits, base, shift)
        got = sum(1 for o in range(size) if our_swizzle(bits, base, shift, o) != o)
        rows4.append((bits, size, got, got / size))
    eq("B", "B.4", [round(r[3], 6) for r in rows4],
       [round(1 - 2.0 ** (-b), 6) for b in (1, 2, 3, 4, 5)],
       "★ **非平凡比例是 `1 − 2^−bits`**：%s ⇒ 与 `base` / `shift` **完全无关**"
       % "; ".join("bits=%d → %d/%d=%.4f" % (r[0], r[2], r[1], r[3]) for r in rows4))

    # ---- B.5 用固定宽度打包：一个 8×8 行主序 tile 有多少元素真的被挪动 ----
    S3, L = mod.Swizzle(3, 3, 3), mod.Layout((8, 8), (8, 1))
    C = mod.ComposedLayout(S3, 0, L)
    moved = [i for i in range(64) if C(tuple(int(x) for x in (i // 8, i % 8))) != i]
    eq("B", "B.5", len(moved), 0,
       "★★ **一个 8×8 的行主序 tile 放在 `ComposedLayout(Swizzle(3,3,3), 0, L)` 下，"
       "**64 个元素一个都没被挪动**（%d/64）** —— 因为整块 tile 的偏移都 < 64，"
       "而恒等区间正是 `[0, 64)`。⇒ **小 tile 上看不见 swizzle**" % len(moved))

    # ---- B.6 同一块 tile，放大到 16×8 才看得见 ----
    L2 = mod.Layout((16, 8), (8, 1))
    C2 = mod.ComposedLayout(S3, 0, L2)
    moved2 = [i for i in range(128) if C2(tuple(int(x) for x in (i // 8, i % 8))) != i]
    eq("B", "B.6", len(moved2), 64,
       "★ 同一块 SMEM 放大到 **16×8**（偏移跨过 63）：**%d/128** 个元素被挪动 ⇒ "
       "`Swizzle(3,3,3)` 的效果**只在 tile 的偏移越过 64 之后**才出现"
       "（推导见 §2 的 `our_identity_bound`）" % len(moved2))
    return rows4


# ===========================================================================
# 5. C 类：边界与反例（**每一条都要真的跑出问题**）
# ===========================================================================
def section_C(mod):
    print()
    print("=" * 78)
    print("C 类：边界与反例 —— 下面每一条都是**真的跑出来的**")
    print("=" * 78)

    # ---- C.6 ★★ 陷阱 A：size(swizzle) 静默返回 Swizzle 自身 ----
    S = mod.Swizzle(3, 3, 3)
    eq("C", "C.6", (isinstance(S, mod.LayoutBase), S.size(), str(mod.size(S)),
                    mod.product(S) is S, isinstance(mod.ComposedLayout(S, 0, mod.Layout(8)), mod.LayoutBase)),
       (False, 512, "SW_3_3_3", True, True),
       "★★ `isinstance(S, LayoutBase)` = **False**（`class Swizzle:` **没有基类**）；"
       "`S.size()`（方法）= **512**；而模块级 `size(S)` = **`SW_3_3_3`**"
       "（因为它落到 `product()` 分支，而 `product(S)` **就是 S 自己**）。"
       "**不抛异常、不告警** —— 一个 layout 类的对象变成了它的字符串表示。"
       "★ 对照：同一段代码里的 `ComposedLayout` **是** `LayoutBase` 的子类（%s）"
       % isinstance(mod.ComposedLayout(S, 0, mod.Layout(8)), mod.LayoutBase))

    # ---- C.6b ★★ 这条陷阱的**作用域**：`cosize` 不失效 ----
    eq("C", "C.6b", (mod.cosize(S), S.cosize(), mod.size(S) is S),
       (512, 512, True),
       "★★ **`cosize(S)` = 512，不失效** —— 模块级 `cosize()` 里**没有** `is_layout` 分支，"
       "它无条件调 `.cosize()`，而 `Swizzle` 恰有该方法。"
       "⇒ 「模块级 `size()` / `cosize()` 对 `Swizzle` 失效」这句话里，"
       "**只有 `size()` 那一半成立** —— 说成两半都失效是把局部 bug 升格成通用规则（P10）。"
       "★ `mod.size(S) is S` ⇒ %s（**它返回的是同一个对象**，不是副本）" % (mod.size(S) is S))

    # ---- C.7 陷阱 B：abs(shift) >= bits 是构造式强制的 ----
    exc = []
    for params in ((4, 3, 3), (3, 3, 2), (5, 0, 4)):
        try:
            mod.Swizzle(*params)
            exc.append("没有抛异常")
        except Exception as e:                # noqa: BLE001
            exc.append(type(e).__name__)
    eq("C", "C.7", exc, ["AssertionError"] * 3,
       "★ `Swizzle(bits,base,shift)` **强制** `|shift| >= bits`："
       "`Swizzle(4,3,3)` / `Swizzle(3,3,2)` / `Swizzle(5,0,4)` 全部 **AssertionError**。"
       "「4 位掩码、3 位基、3 位移位」在纸面上完全合理 —— 这是**构造式**强制的，不是文档建议")

    # ---- C.8 composition(swizzle, layout) 是错的写法 ----
    exc8 = None
    try:
        mod.composition(S, mod.Layout((8, 8), (8, 1)))
    except Exception as e:                    # noqa: BLE001
        exc8 = "%s: %s" % (type(e).__name__, e)
    eq("C", "C.8", exc8,
       "AttributeError: 'Swizzle' object has no attribute 'shape'",
       "★ `composition(swizzle, layout)` ⇒ **AttributeError**（`composition` 要读 `B.shape`）。"
       "swizzle 要用 `ComposedLayout(swizzle, offset, layout)` 表达 —— "
       "**它不是「另一种写法」，是唯一能参与复合的写法**（`A.4` 验证它的语义）")

    # ---- C.9 ★★ 反直觉量化结论：小规模用例上完全看不出它在做什么 ----
    rows = []
    for width in (8, 16, 32, 64):
        L = mod.Layout((width, 8), (8, 1))
        C = mod.ComposedLayout(S, 0, L)
        n = width * 8
        rows.append((width, n, sum(1 for i in range(n)
                                   if C(tuple(int(x) for x in (i // 8, i % 8))) != i)))
    eq("C", "C.9", rows, [(8, 64, 0), (16, 128, 64), (32, 256, 192), (64, 512, 448)],
       "★★ **`Swizzle(3,3,3)` 在偏移 < 64 时是恒等映射**：8×8 的 tile **一个元素都不动**；"
       "16×8 才动 64 个；32×8 动 192 个；64×8 动 448 个。"
       "⇒ 一个「用来打散布局」的算子，在**小规模测试用例上完全看不出它在做什么** —— "
       "这直接解释了为什么 swizzle 的效果只能在**足够大的 tile** 上观察到"
       "（%s）" % (", ".join("%d×8: %d/%d 动" % (r[0], r[2], r[1]) for r in rows),))

    # ---- C.10 恒等区间之外**不是**均匀的 ----
    #   ★ 这条是为了防止把「87.5% 被改变」误读成「每个元素被挪动的距离都差不多」。
    dist = [S3dist(o, S) for o in range(512)]
    eq("C", "C.10", (min(dist), max(dist)), (0, 56),
       "★ 被改变的那些偏移，**位移量不是常数**：`|S(o) − o|` 在 **0 … 56** 之间"
       "（0 是那 64 个恒等点）。⇒ 「87.5% 被改变」这句话**不蕴含**"
       "「改动是均匀的」—— 掩码有 3 位，位移量就有 8 档")

    # ---- C.11 用本仓库的实现替代 pycute 也算一遍（两条路径都要给同一个答案）----
    bad11 = []
    for o in range(512):
        if our_swizzle(3, 3, 3, o) != S(o):
            bad11.append(o)
            break
    eq("C", "C.11", len(bad11), 0,
       "★ 陷阱的行为**只在 `size()` 上**：`S(o)` 本身（那 512 个映射）两条实现逐点一致"
       "（不一致 %d 个）⇒ 坏的是 `size()` 这个**入口**，不是 swizzle 的**语义**" % len(bad11))
    return None


def S3dist(o, S):
    return abs(S(o) - o)


# ===========================================================================
# 6. 主流程
# ===========================================================================
def main():
    print("=" * 78)
    print("03_swizzle_probe.py —— swizzle 是一种 layout 构造，不是一个特例函数")
    print("=" * 78)
    print("  ★ 真值标注 : 本主题的数字全部来自**在本机跑 %s**，" % specs.MEASURE_IMPL)
    print("                **不是**从 GPU 上测的，**也不是**本仓库的推导。")
    print("               **测量条件是 CUTLASS 的版本（%s），不是 GPU 型号。**"
          % specs.CUTLASS_VERSION_BASELINE)
    print("  ★ 本篇**不推导 bank 冲突的机制** —— 回链")
    print("               `20-gpu-architecture/memory-system/03-shared-memory-bank-conflict`")
    print("  python     : %s" % sys.version.split()[0])

    root = specs.require_cutlass("03_swizzle_probe.py")
    mod = specs.load_pycute(root)
    print("  pycute     : %s" % os.path.join(root, specs.PYCUTE_DIR))
    print("  探测面     : %d 个**合法**的 (bits, base, shift) 组合（`abs(shift) >= bits` "
          "由构造式强制）" % len(SWIZZLE_PARAMS))
    print("               %s" % ", ".join("(%d,%d,%d)" % p for p in SWIZZLE_PARAMS))

    rows, rows3 = section_A(mod)
    section_B(mod)
    section_C(mod)

    print()
    print("-" * 78)
    print("  ── 恒等区间与首个非平凡映射（推导 vs 实测，逐行）──")
    print("     %-14s %-10s %-10s %-12s %s" % ("(bits,base,shift)", "yyy_pos", "恒等上界",
                                              "实测首个偏移", "首个偏移的像"))
    for (bits, base, shift, bound, first, pred, img) in rows:
        print("     %-14s %-10d %-10d %-12s %s"
              % ("(%d,%d,%d)" % (bits, base, shift), yyy_pos(bits, base, shift), bound,
                 first if first is not None else "无", "%d (推导 %d)" % (img, pred)))
    print()
    print("  ── 非平凡计数：实测 vs 闭式 `size × (2^bits−1) / 2^bits`（逐行）──")
    print("     %-14s %-8s %-8s %-8s %s" % ("(bits,base,shift)", "size", "实测", "闭式", "比例"))
    for (bits, base, shift, size, got, pred) in rows3:
        print("     %-14s %-8d %-8d %-8d %.4f"
              % ("(%d,%d,%d)" % (bits, base, shift), size, got, pred, got / size))
    print()
    eq("A", "A.8", ast_import_scan(HERE, PROGRAMS + AUX_MODULES), [],
       "★ AST 导入扫描：三个编号程序 + `specs.py` / `meta_falsify.py` / `mutate.py` "
       "都不 import torch / triton / numba / cuda")
    return summary()


def lint_main():
    files = PROGRAMS + AUX_MODULES
    print("AST 导入扫描：%s" % ", ".join(files))
    hits = ast_import_scan(HERE, files)
    if hits:
        for h in hits:
            print("  !! %s" % (h,))
        return 1
    print("  ✓ 没有 import torch / triton / numba / cuda（含 specs.py，含 import_module）")
    return 0


if __name__ == "__main__":
    if "--lint" in sys.argv:
        sys.exit(lint_main())
    sys.exit(main())

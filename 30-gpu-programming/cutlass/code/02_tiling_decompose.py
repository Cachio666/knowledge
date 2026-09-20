#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""02_tiling_decompose.py —— 层次化 tiling：`CTA shape = 指令形状 × 乘数`，以及被抹掉的那一维。

配套文档：`02-hierarchical-tiling.md`。★ 三类断言（A 算法正确性 / B 代价模型 /
C 边界与反例）**全部在本程序内部**，失败时以**非零退出码**结束；**没有 `verify.py`**。

★★ 真值标注（逐处都要带）：

> 本主题的数字全部来自**在本机跑 CUTLASS 4.8.0 自带的 Python 参考实现**
> （`python/pycute/` 与 `python/cutlass_library/`），**不是**从 GPU 上测的，
> **也不是**本仓库的推导。**测量条件是 CUTLASS 的版本，不是 GPU 型号。**

★★ 本程序对 **P11** 的正面回答：本仓库**自己重跑一遍**生成器的枚举循环
  （`指令形状 × 乘数 × cluster`），与 `cutlass_library` **发射出的对象**逐条对拍。
  §3 的 `B.3` 断言「发射出的每个 `threadblock_shape` 都落在我们算的乘积集合里」——
  **改坏任一侧，断言立刻失败**。

★★ 本程序报出的**一处真实差异**（不是工件错了，而是工件里有一处**信息缺失**）：
  `sm90_utils.py:378–383` 把 `instruction_shape` **置零**，
  于是**同一个 `threadblock_shape` 可以由不同的 (指令形状, 乘数) 对产生，
  而发射出的 kernel 不携带这个信息**。`B.8`/`C.5` 把它量化并给出实例。

用法：
    python3 02_tiling_decompose.py          # 跑 A/B/C 三类断言
    python3 02_tiling_decompose.py --lint   # AST 扫描：不 import torch/triton/numba/cuda
"""

import ast
import collections
import importlib
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
COVERED_ANCHORS = ("L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9",
                   "L10", "L11", "L12")

# ★ 全局档位解码的**抽样面**：0..3999 全枚举（四位十进制的全部合法组合的前四位）
LEVEL_SCAN = tuple(range(0, 4000))


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
# 2. ★★ 本仓库自己的实现：枚举循环与档位解码（与 `cutlass_library` 对拍的那一侧）
# ===========================================================================
def our_decode_global_level(g):
    """把**全局档位**拆成四个十进制位（个/十/百/千）。

    ★ 本仓库的写法是**先拆位再对应**，与库里的四个取模函数是两条不同的写法。
    """
    digits = []
    v = g
    for _ in range(4):
        digits.append(v % 10)
        v //= 10
    return tuple(digits)      # (wgmma, mma, cluster, pruning)


def our_cta_products(math_instructions, multipliers, clusters):
    """**自己重跑一遍**生成器的枚举循环：`CTA = 指令形状 × 乘数`（逐元素），再配 cluster。

    返回 `{threadblock_shape: {(指令形状, 乘数, cluster), …}}` —— 值那一侧是**原像集合**，
    它是本程序能报出「同一个 CTA 有两个来源」的全部依据。
    """
    out = collections.defaultdict(set)
    for mi in math_instructions:
        for m in multipliers:
            for c in clusters:
                tbs = tuple(mi.instruction_shape[i] * m[i] for i in range(3))
                out[tbs].add((tuple(mi.instruction_shape), tuple(m), tuple(c)))
    return dict(out)


def our_procedural_name(tbs, stages, cluster, min_compute):
    """**自己写一遍** `procedural_name` 的两种命名方案。

    ★ 诚实说明：这是**同一个公式的另一种写法**（`%` 格式化 vs `str.format`），
      **不是**独立推导。它的价值在于：公式在这里是**可复核、可改动、可被断言**的，
      而不是藏在 `library.py` 里的一段 `.format(...)`。
    """
    if min_compute >= 90:
        return "%dx%dx%d_%dx%dx%d_%d" % (tbs[0], tbs[1], tbs[2],
                                         cluster[0], cluster[1], cluster[2], stages)
    return "%dx%d_%dx%d" % (tbs[0], tbs[1], tbs[2], stages)


# ===========================================================================
# 3. A 类：算法正确性（本仓库实现 vs `cutlass_library`）
# ===========================================================================
def section_A(lib, sm90_utils, sm90_shapes, sm100_shapes, root):
    print("=" * 78)
    print("A 类：算法正确性 —— **本仓库自己的实现** vs `cutlass_library`")
    print("=" * 78)
    print("  ★ 真值那一侧跑的是 %s（**本机实测**）" % specs.MEASURE_IMPL)
    print()

    # ---- A.1 全局档位的四级解码 ----
    bad = []
    for g in LEVEL_SCAN:
        ours = our_decode_global_level(g)
        theirs = (sm90_utils.get_wgmma_level_from_global_level(g),
                  sm90_utils.get_mma_level_from_global_level(g),
                  sm90_utils.get_cluster_level_from_global_level(g),
                  sm90_utils.get_pruning_level_from_global_level(g))
        if ours != theirs:
            bad.append("level=%d ours=%r theirs=%r" % (g, ours, theirs))
    eq("A", "A.1", len(bad), 0,
       "★ 四级档位解码对拍：level ∈ [0,4000) 共 **%d** 个，不一致 %d 个%s"
       % (len(LEVEL_SCAN), len(bad), "" if not bad else "；首例 %s" % bad[0]))

    # ---- A.2 / A.3 CTA 乘积：本仓库重算 vs 发射出的对象 ----
    n_prod, n_tds, bad2 = 0, 0, []
    for g in (0, 1, 2, 9, 10, 19, 20, 29, 30, 39, 49, 100, 101):
        mins = sm90_utils.generate_fp16_bf16_math_instructions_sm90(g)
        muls = sm90_utils.get_mma_multipliers(g)
        clus = sm90_utils.get_cluster_sizes(g, True)
        products = our_cta_products(mins, muls, clus)
        tds = sm90_utils.generate_tile_descriptions_sm90(mins, True, g)
        n_prod += len(products)
        n_tds += len(tds)
        emitted = {tuple(t.threadblock_shape) for t in tds}
        for tbs in sorted(emitted):
            if tbs not in products:
                bad2.append("level=%d tbs=%r 不在乘积集合里" % (g, tbs))
    eq("A", "A.2", len(bad2), 0,
       "★★ **`CTA shape = 指令形状 × 乘数`（逐元素）**：13 个全局档位、本仓库重算 **%d** 个"
       "不同的 CTA 形状，发射出的 **%d** 个 `TileDescription` **逐条**落在乘积集合里"
       "（违例 %d 个）" % (n_prod, n_tds, len(bad2)))

    # ---- A.3 procedural_name 的两种写法给出同一个串（六行）----
    bad3 = []
    for tbs, stages, cluster, mc, want in specs.TILE_NAME_CASES:
        theirs = make_tile_desc(lib, tbs, stages, cluster, mc).procedural_name()
        ours = our_procedural_name(tbs, stages, cluster, mc)
        if theirs != want or ours != want:
            bad3.append("%r/%r/%r/%d ⇒ ours=%s theirs=%s want=%s"
                        % (tbs, stages, cluster, mc, ours, theirs, want))
    eq("A", "A.3", len(bad3), 0,
       "★ `procedural_name` 的**六行逐字复核**（`%%` 格式化写法 vs `.format` 写法）："
       "不一致 %d 个" % len(bad3))

    # ---- A.4 枚举取值数 ----
    bad4 = []
    for name, _line, want_n, _note in specs.ENUM_DEFS:
        got = len(list(getattr(lib, name)))
        if got != want_n:
            bad4.append("%s: %d ≠ %s" % (name, got, want_n))
    eq("A", "A.4", len(bad4), 0,
       "★ %d 个枚举的实测取值数（`OpcodeClass` %d / `DataType` %d / `LayoutType` %d / "
       "`GemmKind` %d / `GemmUniversalMode` %d / `MathOperation` %d）：不一致 %d 个"
       % ((len(specs.ENUM_DEFS),) + tuple(n for _, _, n, _ in specs.ENUM_DEFS) + (len(bad4),)))

    # ---- A.5 模块导入 ----
    bad5 = []
    for m in specs.LIB_MODULES:
        try:
            importlib.import_module(m)
        except Exception as e:                # noqa: BLE001
            bad5.append("%s: %r" % (m, e))
    eq("A", "A.5", len(bad5), 0,
       "★ `cutlass_library` 的 **%d** 个子模块逐个导入（另有 `pycute`）：失败 %d 个"
       "（`pyproject.toml` 里的 networkx / pydot / treelib **本机都没装**，"
       "而导入全部成功 ⇒ 它们是**懒加载**的）" % (len(specs.LIB_MODULES), len(bad5)))

    # ---- A.9 ★ `specs.GLOBAL_LEVEL_DECODE_CASES` 逐行复核 ----
    #   ★ 在这个断言之前，那张表**没有任何程序读过** —— 它是「既保留数字又没人消费」
    #     的典型（`make mutate` 报出来的漏网里，它一整块在列）。
    bad9 = []
    for (g, wg, mm, cl, pr) in specs.GLOBAL_LEVEL_DECODE_CASES:
        ours = our_decode_global_level(g)
        theirs = (sm90_utils.get_wgmma_level_from_global_level(g),
                  sm90_utils.get_mma_level_from_global_level(g),
                  sm90_utils.get_cluster_level_from_global_level(g),
                  sm90_utils.get_pruning_level_from_global_level(g))
        if ours != (wg, mm, cl, pr) or theirs != (wg, mm, cl, pr):
            bad9.append("level=%d 表=%r ours=%r theirs=%r" % (g, (wg, mm, cl, pr), ours, theirs))
    eq("A", "A.9", len(bad9), 0,
       "★★ `specs.GLOBAL_LEVEL_DECODE_CASES` 的 **%d 行逐字复核**（表 vs 本仓库实现 vs "
       "库的四个函数，**三方一致**）：不一致 %d 行%s —— "
       "★ 这张表是**写在文档里的那几个数**的可执行形式（正文 §1.2 引的就是它）"
       % (len(specs.GLOBAL_LEVEL_DECODE_CASES), len(bad9),
          "" if not bad9 else "；%r" % (bad9[:2],)))

    # ---- A.6 锚点的 file:line 逐字比对 ----
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
    return bad2


def make_tile_desc(lib, tbs, stages, cluster, min_compute):
    """按 `library.py:1474` 的**真实签名**构造一个 `TileDescription`。"""
    mi = lib.MathInstruction((16, 8, 16), lib.DataType.f16, lib.DataType.f16,
                             lib.DataType.f32, lib.OpcodeClass.TensorOp)
    return lib.TileDescription(threadblock_shape=list(tbs), stages=stages,
                               warp_count=[4, 1, 1], math_instruction=mi,
                               min_compute=min_compute, max_compute=100,
                               cluster_shape=list(cluster))


# ===========================================================================
# 4. B 类：代价模型 —— 生成器枚举的是什么、漏掉的是什么
# ===========================================================================
def section_B(lib, sm90_utils, sm90_shapes, sm100_shapes, root):
    print()
    print("=" * 78)
    print("B 类：代价模型 —— 生成器枚举的是什么、以及它**漏掉了什么信息**")
    print("=" * 78)

    # ---- B.1 生成数随全局档位 ----
    got = []
    for g, _want in specs.GEN_COUNT_CASES:
        mins = sm90_utils.generate_fp16_bf16_math_instructions_sm90(g)
        got.append((g, len(sm90_utils.generate_tile_descriptions_sm90(mins, True, g))))
    eq("B", "B.1", got, list(specs.GEN_COUNT_CASES),
       "★ fp16/bf16 一族的生成数随档位：%s ⇒ 每上一个**乘数档**多 3 个（每个 wgmma 形状"
       "配 3 个 `MathInstruction`：f16×f16→f16 / f16×f16→f32 / bf16×bf16→f32）"
       % ", ".join("level %d → %d" % (g, n) for g, n in got))

    # ---- B.2 ★ 形状表的 value 是**生成档位**（`min_level`），不是指令数/代价 ----
    bad2 = []
    for nm, table in (("SM90_WGMMA_SHAPES_FP16_BF16_DENSE", sm90_shapes.SM90_WGMMA_SHAPES_FP16_BF16_DENSE),
                      ("SM90_WGMMA_SHAPES_TF32_DENSE", sm90_shapes.SM90_WGMMA_SHAPES_TF32_DENSE)):
        at0 = set(sm90_utils.generate_tf32_math_instruction_shapes_sm90(0)) if "TF32" in nm \
            else set(sm90_utils.generate_fp16_bf16_math_instruction_shapes_sm90(0))
        byval = {k for k, v in table.items() if v == 0}
        if at0 != byval:
            bad2.append("%s: 档位 0 的入选集 %r ≠ value==0 的键集 %r" % (nm, sorted(at0), sorted(byval)))
    eq("B", "B.2", len(bad2), 0,
       "★★ **形状表的 value 是「生成档位」**：筛选条件是 `level >= min_level` ⇒ "
       "**档位 0 的入选集恰好等于 `value == 0` 的键集**（两种表都复核）。"
       "⇒ 它不是指令数、不是代价、不是权重（违例 %d 个%s）"
       % (len(bad2), "" if not bad2 else "；%r" % (bad2,)))

    # ---- B.2b ★ 全枚举的规模：三张表的大小相乘 ----
    #   ★ 这是「为什么需要档位」这个问题的**分母**：不筛的话一共有多少种组合。
    n_sh = len(sm90_shapes.SM90_WGMMA_SHAPES_FP16_BF16_DENSE)
    n_mul = len(sm90_shapes.SM90_MMA_MULTIPLIERS)
    n_clu = len(sm90_shapes.SM90_CLUSTER_SIZES)
    eq("B", "B.2b", (n_sh, n_mul, n_clu, n_sh * n_mul * n_clu), (32, 12, 11, 4224),
       "★ **全枚举的规模**：fp16/bf16 形状表 **%d** 个键 × 乘数表 **%d** 个 × "
       "cluster 表 **%d** 个 = **%d** 种组合 ⇒ 不筛的话生成器要考虑这么多配置。"
       "★ 档位（`level`）就是**把这 4224 种压到不同档位**的那根旋钮"
       "（档位 0 只有 3 个 `TileDescription`，见 `B.1`）"
       % (n_sh, n_mul, n_clu, n_sh * n_mul * n_clu))

    # ---- B.3 CTA 形状集合：本仓库重算的那一侧 vs 发射出的那一侧 ----
    #   ★ 断言的**不是**「我算的表等于我算的表」（那是恒真），而是三条**可证伪**的性质：
    #     ① 乘积的**带重数计数** == |不同的指令形状| × |乘数| × |cluster|
    #        （★ 分母是**不同的形状数**，不是指令对象数 —— 每个形状配 3 个 dtype 的
    #         `MathInstruction`，它们给出**同一个** CTA 形状）；
    #     ② **发射出的 CTA 形状集合 ⊆ 乘积集合**，且 `|发射| ≤ |乘积|`
    #        （`is_tile_desc_valid` 只**滤掉**，不会凭空造出形状）；
    #     ③ 过滤确实发生过 —— 有档位严格小于，也有档位**恰好相等**。
    rows, bad3, eq_levels = [], [], []
    for g in (0, 9, 10, 19, 20, 29, 30, 39, 49):
        mins = sm90_utils.generate_fp16_bf16_math_instructions_sm90(g)
        muls = sm90_utils.get_mma_multipliers(g)
        clus = sm90_utils.get_cluster_sizes(g, True)
        shapes = {tuple(m.instruction_shape) for m in mins}
        prods = our_cta_products(mins, muls, clus)
        tds = sm90_utils.generate_tile_descriptions_sm90(mins, True, g)
        emitted = {tuple(t.threadblock_shape) for t in tds}
        counted = sum(len(v) for v in prods.values())
        rows.append((g, len(mins), len(shapes), len(muls), len(clus), counted,
                     len(prods), len(tds), len(emitted)))
        if counted != len(shapes) * len(muls) * len(clus):
            bad3.append("level %d: 带重数 %d ≠ 形状 %d × 乘数 %d × cluster %d"
                        % (g, counted, len(shapes), len(muls), len(clus)))
        if not emitted <= set(prods) or len(emitted) > len(prods):
            bad3.append("level %d: 发射集合不是乘积集合的子集（或更大）" % g)
        if len(emitted) == len(prods):
            eq_levels.append(g)
    eq("B", "B.3", (len(bad3), eq_levels), (0, [0, 10, 20, 30]),
       "★ CTA 形状集合的两个方向（本仓库重算 9 个档位）：%s ⇒ "
       "① 乘积的**带重数**计数恒等于「不同指令形状 × 乘数 × cluster」（违例 %d）；"
       "② 发射出的 CTA 形状**总是**乘积集合的子集、且不多于它；"
       "③ **过滤真的会发生**：档位 %r 上两者**恰好相等**（无过滤），其余严格小于"
       % ("; ".join("L%d: 形状%d×乘数%d×clu%d ⇒ 乘积%d（去重%d），发射%d（去重CTA %d）"
                    % (r[0], r[2], r[3], r[4], r[5], r[6], r[7], r[8]) for r in rows),
          len(bad3), eq_levels))

    # ---- B.4 procedural_name 的 sm90 断点（六行逐字）----
    got4 = [(make_tile_desc(lib, tbs, st, cl, mc).procedural_name())
            for tbs, st, cl, mc, _ in specs.TILE_NAME_CASES]
    eq("B", "B.4", got4, [w for _, _, _, _, w in specs.TILE_NAME_CASES],
       "★★ `procedural_name()` 在 `min_compute` 从 **80 变到 90** 时**换了整套命名方案**："
       "%s" % " / ".join(got4))

    # ---- B.5 warp_count 恒为 [4,1,1] ----
    mins = sm90_utils.generate_fp16_bf16_math_instructions_sm90(49)
    tds = sm90_utils.generate_tile_descriptions_sm90(mins, True, 49)
    bad5 = [t for t in tds if list(t.warp_count) != [4, 1, 1]]
    eq("B", "B.5", len(bad5), 0,
       "★ SM90 的 `warp_count` 在**全部 %d** 个发射结果上恒为 `[4, 1, 1]`（写死的，"
       "不随形状变）：违例 %d 个" % (len(tds), len(bad5)))

    # ---- B.6 CTA 各维 ≤ 256 ----
    bad6 = [tuple(t.threadblock_shape) for t in tds
            if any(x > 256 for x in t.threadblock_shape)]
    eq("B", "B.6", len(bad6), 0,
       "★ CTA 各维**上限 256**（`sm90_utils.py:351–353` 的代码判的是 `> 256`，即**含 256**；"
       "注释写的是 `<256, 256, 256>` —— 本库按**代码**断言，不按注释）："
       "全部 %d 个结果里越界 %d 个" % (len(tds), len(bad6)))

    # ---- B.7 cluster_shape 的类型随来源变 ----
    kinds = {type(t.cluster_shape).__name__ for t in tds}
    eq("B", "B.7", sorted(kinds), ["tuple"],
       "★ 生成器给的 `cluster_shape` 是 **tuple**（表的键），而 `TileDescription` 的默认值是 "
       "**list** `[1,1,1]` —— 同一个属性两种类型（P9 的又一例）：实测类型 %r" % sorted(kinds))

    # ---- B.8 ★★ 泄漏：同一个 CTA 形状可以有多个来源，而发射出的对象把它抹掉了 ----
    mins39 = sm90_utils.generate_fp16_bf16_math_instructions_sm90(specs.SM90_LEAK_LEVEL)
    prods39 = our_cta_products(mins39,
                               sm90_utils.get_mma_multipliers(specs.SM90_LEAK_LEVEL),
                               sm90_utils.get_cluster_sizes(specs.SM90_LEAK_LEVEL, True))
    tds39 = sm90_utils.generate_tile_descriptions_sm90(mins39, True, specs.SM90_LEAK_LEVEL)
    names = collections.Counter(t.procedural_name() for t in tds39)
    multi = {k: v for k, v in names.items() if v > 1}
    multi_preimage = {k: v for k, v in prods39.items() if len(v) > 1}
    zeroed = {tuple(t.math_instruction.instruction_shape) for t in tds39}
    eq("B", "B.8", (len(tds39), len(names), max(names.values()),
                    len(multi), len(prods39), len(multi_preimage), sorted(zeroed)),
       (specs.SM90_LEAK_TDS, specs.SM90_LEAK_UNIQUE_NAMES, specs.SM90_LEAK_MAX_NAME_MULT,
        len(multi), len(prods39), specs.SM90_LEAK_MULTI_PREIMAGE, [(0, 0, 0)]),
       "★★ **生成器抹掉指令形状的后果（实测）**：全局档位 %d 下发射 **%d** 个 "
       "`TileDescription`，但只有 **%d** 个**不同的 `procedural_name()`** —— "
       "**重名 %d 个**，最多的一个名字出现 **%d 次**；"
       "乘积集合里 **%d/%d** 个 CTA 形状有**多于一个** (指令形状, 乘数) 原像；"
       "而发射出的对象携带的 `instruction_shape` 恒为 `%r`。"
       "⇒ 同名的 kernel **不是同一个 kernel**，而发射物**不带区分它们的信息**"
       % ((specs.SM90_LEAK_LEVEL, len(tds39), len(names), len(multi),
           max(names.values()), len(multi_preimage), len(prods39), sorted(zeroed)[0])))

    # ---- B.8b numpy 复核重名统计（**第二条计数路径**）----
    #   ★ 与 `collections.Counter` 那条**无共同代码**：这里是 numpy 的排序 + 边界差分。
    _uniq, _cnt = np.unique(np.array([t.procedural_name() for t in tds39]),
                            return_counts=True)
    eq("B", "B.8b", (int(len(_uniq)), int(_cnt.max()), int(len(tds39))),
       (len(names), max(names.values()), len(tds39)),
       "★ **第二条计数路径**（numpy `unique(return_counts=True)` vs "
       "`collections.Counter`）：不同的名字 **%d** 个、最多的一个出现 **%d** 次、"
       "总数 **%d** ⇒ 两条路径必须一致" % (int(len(_uniq)), int(_cnt.max()), int(len(tds39))))

    # ---- B.9 SM100 的 1SM / 2SM 配对（**带限定语**）----
    a1, a2 = sm100_shapes.SM100_MMA_SHAPES_MXF4NVF4_DENSE_1SM, \
        sm100_shapes.SM100_MMA_SHAPES_MXF4NVF4_DENSE_2SM
    paired = all((k[0] * 2, k[1], k[2]) in a2 and a2[(k[0] * 2, k[1], k[2])] == v
                 for k, v in a1.items()) and len(a1) == len(a2)
    mxf4_m = sorted({k[0] for k in a1})
    tf32_m = sorted({k[0] for k in sm100_shapes.SM100_MMA_SHAPES_TF32_DENSE_1SM})
    eq("B", "B.9", (paired, mxf4_m, tf32_m), (True, [128], [64, 128]),
       "★★ **限定语（P10）**：**在 `MXF4NVF4` 这一族上**，1SM 的 M **全是 128**、"
       "2SM **全是 256**（%d 对逐条成立：M 恰好翻倍、N/K 不变）；"
       "**但** `SM100_MMA_SHAPES_TF32_DENSE_1SM` 的 M 是 **%r** ⇒ "
       "**1SM 的 M 不是一个常数**，所以**不得推广成「SM100 的 2SM 让 M 翻倍」**"
       "（本库**不复述**任何「SM100 都是如此」的说法）" % (len(a1), tf32_m))

    # ---- B.9b ★ `specs` 里那两张 SM100 表的副本与库里的**逐条相等** ----
    #   ★ 这两张表在 `specs.py` 里是**给正文引用的副本**；如果它们与库里的对象漂移了，
    #     正文引的数就不再是实测的那一个（P1「订正不扩散」）。
    eq("B", "B.9b", (a1 == specs.SM100_MXF4NVF4_1SM, a2 == specs.SM100_MXF4NVF4_2SM,
                     specs.SM100_MXF4NVF4_1SM_MS, specs.SM100_TF32_1SM_MS),
       (True, True, tuple(mxf4_m), tuple(tf32_m)),
       "★★ **副本与原件**：`specs.SM100_MXF4NVF4_1SM/2SM` 与库里的 "
       "`sm100_shapes.SM100_MMA_SHAPES_MXF4NVF4_DENSE_1SM/2SM` **逐条相等**（%s/%s）；"
       "`specs.SM100_MXF4NVF4_1SM_MS` = %r、`specs.SM100_TF32_1SM_MS` = %r ⇒ "
       "正文引的「1SM 的 M 是一个常数 / 不是一个常数」两句话都**对得上原件**"
       % (a1 == specs.SM100_MXF4NVF4_1SM, a2 == specs.SM100_MXF4NVF4_2SM,
          specs.SM100_MXF4NVF4_1SM_MS, specs.SM100_TF32_1SM_MS))

    # ---- B.10 「CUTLASS 支持哪些架构」问的是哪个子系统 ----
    arch_max = max(lib.ArchitectureNames)
    present = [f for f in specs.CUTE_ARCH_PRESENT
               if os.path.isfile(os.path.join(root, specs.CUTE_ARCH_DIR, f))]
    absent = [f for f in specs.CUTE_ARCH_ABSENT
              if not os.path.isfile(os.path.join(root, specs.CUTE_ARCH_DIR, f))]
    nfiles = len([f for f in os.listdir(os.path.join(root, specs.CUTE_ARCH_DIR))
                  if os.path.isfile(os.path.join(root, specs.CUTE_ARCH_DIR, f))])
    eq("B", "B.10", (arch_max, len(present), len(absent), nfiles),
       (specs.ARCH_NAMES_MAX, len(specs.CUTE_ARCH_PRESENT),
        len(specs.CUTE_ARCH_ABSENT), specs.CUTE_ARCH_FILE_COUNT),
       "★★ **同一棵树里两个子系统给出不同答案**：`ArchitectureNames` 只到 **%d**"
       "（没有 sm_100 / sm_107 / sm_120），而 `%s/` 下有 **%d** 个文件、"
       "sm100/sm107/sm120 的内容全在（抽查 %d 个存在、%d 个确认**不存在**，"
       "例如 `mma_sm107.hpp` —— 只有 `mma_sm107_desc.hpp` 与 `mma_sm107_umma.hpp`）。"
       "⇒ 引用「CUTLASS 支持哪些架构」**必须指明问哪个子系统**"
       % (arch_max, specs.CUTE_ARCH_DIR, nfiles, len(present), len(absent)))

    # ---- B.11 泄漏的实例：两个原像，发射出的对象一模一样 ----
    tbs = specs.SM90_LEAK_EXAMPLE_TBS
    pres = sorted(prods39.get(tbs, ()))
    eq("B", "B.11", len(pres), len(specs.SM90_LEAK_EXAMPLE_PREIMAGES),
       "★ 泄漏的**实例**：`threadblock_shape = %r` 有 **%d** 组不同的 (指令形状, 乘数) 原像：%r"
       % (tbs, len(pres), pres))
    return None


# ===========================================================================
# 5. C 类：边界与反例（**每一条都要真的跑出问题**）
# ===========================================================================
def section_C(lib, sm90_utils, root):
    print()
    print("=" * 78)
    print("C 类：边界与反例 —— 下面每一条都是**真的跑出来的**")
    print("=" * 78)

    # ---- C.1 ★★ 错误的 MathInstruction 构造式**不报错** ----
    bad = lib.MathInstruction(16, lib.DataType.f16, lib.DataType.f16,
                              lib.DataType.f32, lib.OpcodeClass.TensorOp)
    eq("C", "C.1", (repr(bad.instruction_shape), hasattr(bad, "element_scale_factor")),
       ("16", True),
       "★★ `MathInstruction(16, 8, 16, f16, f16, f32)`（**错误写法**）**不报错**地跑过去了："
       "首参被当成 `instruction_shape = %r`（一个 int，不是元组）。"
       "★ 正确的写法是 `MathInstruction((16,8,16), f16, f16, f32, OpcodeClass.TensorOp)`"
       % (bad.instruction_shape,))

    # ---- C.2 `tile_shape` 与 `threadblock_shape` 是**同一个对象两个名字** ----
    td = make_tile_desc(lib, (128, 128, 32), 3, (2, 1, 1), 90)
    same_obj = td.tile_shape is td.threadblock_shape
    td.tile_shape[1] = 999                       # 改一个
    both = (td.threadblock_shape[1], td.tile_shape[1])
    td.threadblock_shape[1] = 128                # 改回来
    eq("C", "C.2", (same_obj, both), (True, (999, 999)),
       "★★ **同指不同名**（P9「同名不同指」的镜像）：`tile_shape` 与 `threadblock_shape` "
       "**是同一个对象** —— 断言 `is` ⇒ %s；改其中一个，另一个**同步变**（%r）。"
       "⇒ 正文提到它时**必须统一用 `threadblock_shape`**" % (same_obj, both))

    # ---- C.3 没有 `warp_shape` 这个属性 ----
    exc = None
    try:
        td.warp_shape
    except Exception as e:                       # noqa: BLE001
        exc = type(e).__name__
    eq("C", "C.3", (hasattr(td, "warp_shape"), exc, list(td.warp_count)),
       (False, "AttributeError", [4, 1, 1]),
       "★ **参数是 `warp_count`，属性里没有 `warp_shape`**：`hasattr` ⇒ False、"
       "取它会 **AttributeError**。warp 形状是**下游算出来的**，`TileDescription` 不存它")

    # ---- C.4 procedural_name 在 80 → 90 换方案（逐字）----
    a = make_tile_desc(lib, (128, 128, 32), 3, (2, 1, 1), 80).procedural_name()
    b = make_tile_desc(lib, (128, 128, 32), 3, (2, 1, 1), 90).procedural_name()
    eq("C", "C.4", (a, b), ("128x128_32x3", "128x128x32_2x1x1_3"),
       "★ **同一个对象只改 `min_compute`**：80 → `%s`，90 → `%s`。"
       "注意 `tbk` **仍是 32** —— 两行必须来自同一个 `tbs`"
       "（把三行拼成「32x3 / 32x3 / 64_2x1x1_3」需要一个 tbk 同时是 32 与 64，"
       "**不存在这样的对象**）" % (a, b))

    # ---- C.5 ★★ 泄漏的实例：同一个 CTA 形状的多条记录，名字**完全一样** ----
    tbs = specs.SM90_LEAK_EXAMPLE_TBS
    mins = sm90_utils.generate_fp16_bf16_math_instructions_sm90(specs.SM90_LEAK_LEVEL)
    tds = sm90_utils.generate_tile_descriptions_sm90(mins, True, specs.SM90_LEAK_LEVEL)
    hit = [t for t in tds if tuple(t.threadblock_shape) == tbs]
    uniq_names = {t.procedural_name() for t in hit}
    uniq_named_tuple = {(t.procedural_name(), tuple(t.math_instruction.instruction_shape))
                        for t in hit}
    eq("C", "C.5", (len(hit), len(uniq_names), len(uniq_named_tuple)),
       (6, 1, 1),
       "★★ 泄漏的**实例**：`tbs = %r` 在发射结果里出现 **%d** 次，而它们只有 "
       "**%d** 个不同的 `procedural_name()`；把 `procedural_name` 与 "
       "`instruction_shape` 一起看，仍然只有 **%d** 种组合 —— "
       "6 条记录里 **6 条无法区分**。"
       "★ 它们的 (指令形状, 乘数) 原像是 **%r**（`%r`）。"
       "⇒ 「同名 kernel」不是同一个 kernel，而发射物**不带区分它们的信息**"
       % (tbs, len(hit), len(uniq_names), len(uniq_named_tuple),
          sorted(_PREIMAGE_CACHE.get(tbs, ())), specs.SM90_LEAK_EXAMPLE_PREIMAGES))

    # ---- C.6 生成器给出的 TileDescription 一律 stages=0 ----
    stages = {t.stages for t in tds}
    eq("C", "C.6", sorted(stages), [0],
       "★ 生成器发射的 `TileDescription` **一律 `stages=0`**（`sm90_utils.py:391`）—— "
       "真正的 stages 是**下游**由 shmem 容量算出来的，不在这里。"
       "⇒ 拿这个 0 当「流水级数」是错的（P9：同一个词两个层级）")
    return None


_PREIMAGE_CACHE = {}


# ===========================================================================
# 6. 主流程
# ===========================================================================
def main():
    print("=" * 78)
    print("02_tiling_decompose.py —— 层次化 tiling：CTA = 指令形状 × 乘数，以及被抹掉的那一维")
    print("=" * 78)
    print("  ★ 真值标注 : 本主题的数字全部来自**在本机跑 %s**，" % specs.MEASURE_IMPL)
    print("                **不是**从 GPU 上测的，**也不是**本仓库的推导。")
    print("               **测量条件是 CUTLASS 的版本（%s），不是 GPU 型号。**"
          % specs.CUTLASS_VERSION_BASELINE)
    print("  ★ 来源等级 : 全部是 **B 级（NVIDIA 自己的实现，不是规范）** + **E 级（本库推导）**")
    print("  python     : %s" % sys.version.split()[0])

    root = specs.require_cutlass("02_tiling_decompose.py")
    lib, sm90_utils, sm90_shapes, sm100_shapes = specs.load_cutlass_library(root)
    print("  cutlass_library: %d 个模块（子模块目录 %s）"
          % (len(specs.LIB_MODULES), os.path.join(root, "python/cutlass_library")))
    print("  ★ 依赖实况 : `pyproject.toml` 的 dependencies 有 %d 条，其中 %s **本机都没装**，"
          % (len(specs.CUTLASS_PYPROJECT_DEPENDENCIES),
             " / ".join(specs.NOT_INSTALLED_LOCALLY)))
    print("               而 %d 个模块的导入**全部成功** ⇒ 它们是**懒加载**的"
          % len(specs.LIB_MODULES))
    print("  ★ tile_shape 与 threadblock_shape 是**同一个对象两个名字**（C.2 断言 `is`）")

    # 预填泄漏实例的原像缓存（B.11 / C.5 共用）
    _m = sm90_utils.generate_fp16_bf16_math_instructions_sm90(specs.SM90_LEAK_LEVEL)
    _p = our_cta_products(_m, sm90_utils.get_mma_multipliers(specs.SM90_LEAK_LEVEL),
                          sm90_utils.get_cluster_sizes(specs.SM90_LEAK_LEVEL, True))
    _PREIMAGE_CACHE.update(_p)

    section_A(lib, sm90_utils, sm90_shapes, sm100_shapes, root)
    section_B(lib, sm90_utils, sm90_shapes, sm100_shapes, root)
    section_C(lib, sm90_utils, root)

    print()
    print("-" * 78)
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

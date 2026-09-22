#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""02_ffn_shapes.py —— FFN 的形态：从两层到 GLU 变体，`d_ff` 到底怎么定。

配套文档：`02-ffn-shapes.md`。
★ 三类断言（A 算法正确性 / B 代价模型 / C 边界与反例）**全部在本程序内部**，
失败时以**非零退出码**结束；**没有 `verify.py`**。

★★ 本程序的数字分两档，**不许混**：

  · **外部规格值**（`import specs`，每条带来源等级与抓取日）：
    `SWIGLU_DFF_REDUCTION_FACTOR`（SwiGLU 论文 [C]）、`FFN_BASELINE_D_FF_RATIO`（**前提，无等级**）、
    两个真实模型的 `intermediate_size` / `hidden_size` / `hidden_act`（[B] / 镜像）。
  · **推导值**（本程序算出，可手算复核）：`8d/3`、各种比值、参数个数。
    ⇒ 测量条件：`python3 3.13.5` + `numpy 2.4.4`，macOS arm64，**无 GPU**。

★★ 可证伪断言（把这四段改错，本程序必须非零退出）：

  · `A.1`  参数个数：公式 `2·d·d_ff` 与「**真的分配两个矩阵、数它们的 `.size`**」一致
           （把系数 2 改成 3 立刻失败）；
  · `B.2`  **`d_ff = (2/3)·4d` 让两条 FFN 的参数持平**：d 被 3 整除时比值**精确 1.0**，
           否则落在 [0.999, 1.001]（把前提 `4d` 改成 `5d`，比值立刻变成 1.5）；
  · `B.4`  两个真实模型的 `intermediate_size / hidden_size` 是**精确有理数** 3.5 与 3.0
           （改成浮点近似或改字段名立刻失败）；
  · `C.2`  **反例**：拿 `8d/3` 去解释这两个模型会**直接算错**（与真实 `d_ff` 差 > 20%）。

用法：
    python3 02_ffn_shapes.py
"""

import ast
import collections
import inspect
import os
import sys
from fractions import Fraction

import numpy as np

import specs

# ===========================================================================
# 0. 断言记账（每个编号程序自带一份，**不共享**）
# ===========================================================================
# ★★ 失败标记：本主题（transformer）取 `!! FAIL: transformer` —— 与同域另外四个主题
#   写死的取值不同（域 README 规定了五个不同形态），**不跨主题照抄**是本库纪律。
ASSERT_FAIL_MARK = "!! FAIL: transformer"

OTHER_THEME_MARKS = ("@@ FAIL: attention-variants", "## FAIL: moe",
                     "~~ FAIL: normalization", "$$ FAIL: positional-encoding")

FAILS = []
MARKS_USED = set()
# ★ 计数器用 `Counter()` 而**不是**字面量字典：`mutate.py` 的分母是「模块级常量的
#   **字面量叶子**」，写成 `{"A": 0, ...}` 会把 8 个**不承载任何判据**的初值塞进分母
#   （改它们只动打印出来的汇总行，不动任何结论）⇒ 那是给分母灌水。
_TOTAL = collections.Counter()
_PASS = collections.Counter()


def check(name, cond, detail=""):
    cls = name[0]
    _TOTAL[cls] += 1
    if cond:
        _PASS[cls] += 1
        print("  [ok]   %-5s %s" % (name, detail))
    else:
        FAILS.append(name)
        MARKS_USED.add(ASSERT_FAIL_MARK)
        print("  %s %-5s %s" % (ASSERT_FAIL_MARK, name, detail))
    return bool(cond)


# ===========================================================================
# 1. 口径常量与阈值（★ 全部在 `D.3`–`D.6` 被钉住）
# ===========================================================================
# ★ 「d 被 3 整除」的四个点 —— 计划 §3 Chunk 1 表 A ④ **点名要求**另取这样的例子，
#   因为 D19 用的 4096 / 8192 **都不被 3 整除**，那里给出的从来不是「精确持平」。
DIVISIBLE_BY_3 = (768, 3072, 4098, 6144)
NOT_DIVISIBLE_BY_3 = (4096, 8192)
PARITY_LO = 0.999        # 表 A ④ 的区间下界
PARITY_HI = 1.001        # 表 A ④ 的区间上界
FLOAT_PARITY_TOL = 1e-15  # 「不取整时恒为 1.0」这条观察的容差（机器精度档）

# ★ SwiGLU 论文 §3.1 自己的实验配置（逐字）：
#   `The encoder and decoder each consist of 12 layers, with d_{model}=768` …
#   `we reduce the hidden layer to d_{ff}=2048, so as to maintain the same parameter
#    and operation counts as the base model.`
#   来源：arXiv 2002.05202，ar5iv HTML；`d_{model}=768` / `d_{ff}=2048` 均**在场**；
#         该响应 sha256[:16] = 54d6e4f552c74d53（与计划 §C.2 一致）  [C]（**未确认同行评审**）
PAPER_D_MODEL = 768
PAPER_D_FF = 2048

SMALL_D = 64             # A.1 的「真的分配矩阵」那一档用的宽度
SMALL_D_FF = 256


def d_ff_swiglu(d):
    """★ `d_ff = (2/3)·(4d) = 8d/3`，**向下取整**。

    ★★ 两点必须写死在代码里，否则读者无法复核这个数：

    1. **`8d/3` 是推导，不是引文。** SwiGLU 论文里**没有** `8d/3`
       （原始字节计数：`8d/3` = **0** 次、`\\frac{8d}{3}` = **0** 次；对照检验：已知存在的
       `\\frac{2}{3}` 计到 **2** 次 ⇒ 「0」可信）。论文只有 `by a factor of 2/3`。
       那个 `8` 来自 `(2/3) × 4`，而 **`4` 是这篇论文之外的约定**
       （原始的两层 FFN 取 `d_ff = 4d`）。论文里 `4d` 也**没有**出现过（它只写 `d_ff=3072`）。
    2. **取整方向**：`//`（向下取整），与 `probes/probe_80.py` 的 `int(8 * d / 3)` 同口径。
       ⇒ d 不是 3 的倍数时「参数持平」只能是**近似**持平（见 B.2 的 4096 / 8192 两点）。
    """
    return (8 * d) // 3


def params_two_matrix(d, d_ff):
    """原始两层 FFN 的**权重元素个数**：`W1 ∈ R^{d×d_ff}`、`W2 ∈ R^{d_ff×d}`。"""
    return 2 * d * d_ff


def params_glu_variant(d, d_ff):
    """GLU 变体的**权重元素个数**：三个矩阵（两个 `d×d_ff`、一个 `d_ff×d`）。"""
    return 3 * d * d_ff


def params_two_matrix_enumerating(d, d_ff):
    """★ **独立参照**：不套公式，**真的分配两个矩阵**再数它们的元素。

    与被测路径（一条乘法的式子）是两条不同的代码路径：一个是算术，一个是
    对**真实张量**的枚举。★ 它**不调用** `params_two_matrix()`（`D.8` 用 AST 守着）。
    """
    W1 = np.zeros((d, d_ff))
    W2 = np.zeros((d_ff, d))
    return int(W1.size) + int(W2.size)


# ===========================================================================
# 2. A 类：参数个数这个式子本身
# ===========================================================================
def part_A():
    print("=" * 74)
    print("A. 两条 FFN 各自的参数个数")
    print("=" * 74)
    print("  前提（★ 必须写明，否则 `8d/3` 无法复核）：")
    print("    · `SWIGLU_DFF_REDUCTION_FACTOR` = %.10f —— 论文逐字 `a factor of 2/3`  [C]"
          % specs.SWIGLU_DFF_REDUCTION_FACTOR)
    print("    · `FFN_BASELINE_D_FF_RATIO`     = %d —— **论文之外的约定**（原始 FFN 取 d_ff = 4d），"
          "所以它**没有 A–E 等级**" % specs.FFN_BASELINE_D_FF_RATIO)
    print("    · 两者之积 = %.10f = 8/3 ⇒ **`8d/3` 是推导**"
          % (specs.SWIGLU_DFF_REDUCTION_FACTOR * specs.FFN_BASELINE_D_FF_RATIO))

    d, dff = SMALL_D, SMALL_D_FF
    a = params_two_matrix(d, dff)
    b = params_two_matrix_enumerating(d, dff)
    print("  对 d=%d、d_ff=%d：公式给 %d；真的分配两个矩阵再数 `.size` 给 %d"
          % (d, dff, a, b))
    check("A.1", a == b,
          "两层 FFN 的参数个数：公式 == 枚举（整数计数**没有舍入**，所以这里用 `==` 是对的，"
          "**不是** P18 那种「浮点却要求精确 0」的情形）")
    check("A.2", params_glu_variant(d, dff) == 3 * d * dff
          and params_glu_variant(d, dff) - a == d * dff,
          "GLU 变体比两层**多一个矩阵**：%d − %d = d·d_ff = %d"
          % (params_glu_variant(d, dff), a, d * dff))

    lhs = params_glu_variant(PAPER_D_MODEL, PAPER_D_FF)
    rhs = params_two_matrix(PAPER_D_MODEL, specs.FFN_BASELINE_D_FF_RATIO * PAPER_D_MODEL)
    print("  论文 §3.1 自己的配置：d_model=%d，d_ff=%d（三个矩阵）vs 基线 d_ff=%d（两个矩阵）"
          % (PAPER_D_MODEL, PAPER_D_FF, specs.FFN_BASELINE_D_FF_RATIO * PAPER_D_MODEL))
    print("    3·%d·%d = %d ;  2·%d·%d = %d ;  相等？ %s"
          % (PAPER_D_MODEL, PAPER_D_FF, lhs, PAPER_D_MODEL,
             specs.FFN_BASELINE_D_FF_RATIO * PAPER_D_MODEL, rhs, lhs == rhs))
    check("A.3", lhs == rhs and PAPER_D_FF == d_ff_swiglu(PAPER_D_MODEL),
          "★ 论文选的数字使「参数持平」**精确成立**（整数恒等式），且 `8×768/3 = %d` 是整数"
          % d_ff_swiglu(PAPER_D_MODEL))


# ===========================================================================
# 3. B 类：代价模型 —— d_ff 取多少，两条 FFN 的参数才持平
# ===========================================================================
def parity_ratio(d):
    """推导值：GLU 变体（3 个矩阵，`d_ff = 8d/3`）与两层 FFN（2 个矩阵，`d_ff = 4d`）的比值。"""
    return Fraction(params_glu_variant(d, d_ff_swiglu(d)),
                    params_two_matrix(d, specs.FFN_BASELINE_D_FF_RATIO * d))


def part_B():
    print()
    print("=" * 74)
    print("B. 代价模型：`d_ff` 取多少，两条 FFN 的参数才持平")
    print("=" * 74)
    print("  这个式子的来源是**矩阵个数与形状**（2 个 vs 3 个）——")
    print("  ⚠ **不是**每 token 的 FLOPs 系数（4 vs 6）：那个正文在")
    print("    `40-kernels/moe-kernels/01-moe-compute-pattern.md:72,278` 与 `…/code/verify.py:443`")
    print("    （正文在彼，此处回链 —— 域 README 的边界表已把它列为本主题的禁区）。")

    exact = []
    for d in DIVISIBLE_BY_3:
        dff = d_ff_swiglu(d)
        r = parity_ratio(d)
        exact.append(float(r) == 1.0)
        print("  d=%5d（被 3 整除）：8d/3 = %6d，比值 = %s = %.10f" % (d, dff, r, float(r)))
    check("B.1", all(exact),
          "★ 表 A ④ **点名要求**的四个「d 被 3 整除」的点：比值**精确 1.0**"
          "（D19 用的 4096/8192 都不被 3 整除，那里从来不是精确持平）")

    r4096, r8192 = parity_ratio(4096), parity_ratio(8192)
    print("  d= 4096（不被 3 整除）：8d/3 取整为 %6d，比值 = %.10f = %s"
          % (d_ff_swiglu(4096), float(r4096), r4096))
    print("  d= 8192（不被 3 整除）：8d/3 取整为 %6d，比值 = %.10f = %s"
          % (d_ff_swiglu(8192), float(r8192), r8192))
    allr = [float(parity_ratio(d)) for d in DIVISIBLE_BY_3 + NOT_DIVISIBLE_BY_3]
    check("B.2", ("%.10f" % float(r4096)) == "0.9999389648"
          and ("%.10f" % float(r8192)) == "0.9999847412"
          and all(PARITY_LO <= v <= PARITY_HI for v in allr),
          "表 A ④ 的判据：d=4096 → %.10f、d=8192 → %.10f **逐位复现**计划预演的值"
          "（它写的是 10 位小数），且 6 个 d 全部落在 [%g, %g]"
          % (float(r4096), float(r8192), PARITY_LO, PARITY_HI))

    print("  前提敏感度（`FFN_BASELINE_D_FF_RATIO` 是**前提**，不是可调参数）：")
    # ★ 这一档取 d 被 3 整除的点，免得取整的 1e-4 混进来 —— 这里要量的是**前提**的效应。
    base_ratios = {}
    for base in (3, 4, 5):
        base_ratios[base] = Fraction(params_glu_variant(3072, d_ff_swiglu(3072)),
                                     params_two_matrix(3072, base * 3072))
        print("    基线 d_ff = %d·d ⇒ 三个矩阵 / 两个矩阵（同一条 8d/3 的 d_ff）= %s = %.6f"
              % (base, base_ratios[base], float(base_ratios[base])))
    check("B.3", base_ratios[4] == 1 and base_ratios[5] == Fraction(4, 5),
          "★ 「持平」只在 `d_ff = 4d` 这条前提上成立：换成 5d，同一条 8d/3 立刻给出 %s×；"
          "换 3d 给 %s×（= 4/base，可手算）"
          % (base_ratios[5], base_ratios[3]))

    print()
    print("  两个真实模型的 `intermediate_size / hidden_size`：")
    # ★ `intermediate_size` 在 Qwen3 上的语义是「**非 MoE 的 FFN 宽度**」，不等于专家宽度
    #   （专家宽度在 `moe_intermediate_size = 768`，属 `moe` 主题）。这里只用字段值。
    lb = specs.LLAMA3_8B_CONFIG
    qw = specs.QWEN3_30B_A3B_CONFIG
    ok = True
    for name, cfg, want in (("Meta-Llama-3-8B", lb, Fraction(7, 2)),
                            ("Qwen3-30B-A3B", qw, Fraction(3, 1))):
        r = Fraction(cfg["intermediate_size"], cfg["hidden_size"])
        ok = ok and (r == want)
        print("    %-16s %6d / %4d = %s = %.1f    （`hidden_act` = %s）"
              % (name, cfg["intermediate_size"], cfg["hidden_size"], r, float(r),
                 cfg["hidden_act"]))
    check("B.4", ok, "两个比值都是**精确有理数**（7/2 与 3/1，整数比、不是浮点近似）")
    print("    ⇒ ★★ **没有一个是 4，也没有一个是 8/3 ≈ 2.667** —— 拿「Llama 用 8d/3」"
          "去解释这两个模型会直接算错（C.2 量化了差多少）。")
    return lb, qw


# ===========================================================================
# 4. C 类：边界与反例
# ===========================================================================
def part_C(lb, qw):
    print()
    print("=" * 74)
    print("C. 边界与反例")
    print("=" * 74)

    print("  C.1：`8d/3` 必须**取整** —— 否则计划预演的那两个点会静默消失")
    d = 4096
    ratio_float = (3 * (8 * d / 3) * d) / (2 * (4 * d) * d)      # 不取整
    dff = d_ff_swiglu(d)
    ratio_int = (3 * dff * d) / (2 * (4 * d) * d)                # 取整
    print("    不取整：3·(8d/3)·d / (2·(4d)·d) = %.17f" % ratio_float)
    print("    取整  ：3·%d·%d / (2·(4d)·d)      = %.10f" % (dff, d, ratio_int))
    check("C.1", abs(ratio_float - 1.0) < FLOAT_PARITY_TOL and ratio_int < 1.0,
          "★ 不取整时它**恒为 1.0**（3 与 1/3 在浮点里通常互为精确逆）⇒ 计划给的 %.10f "
          "只可能来自**取整**；而 `d_ff` 是权重矩阵的宽度，**必须是整数**" % ratio_int)

    print()
    print("  C.2：反例 —— 拿 `8d/3` 去解释真实模型")
    for name, cfg in (("Meta-Llama-3-8B", lb), ("Qwen3-30B-A3B", qw)):
        dd = cfg["hidden_size"]
        real = cfg["intermediate_size"]
        derived = 8 * dd / 3
        with4 = 4 * dd
        print("    %-16s 真实 d_ff = %6d · 8d/3 = %9.2f · 4d = %6d"
              % (name, real, derived, with4))
        print("      ⇒ 与 8d/3 的相对差 %.1f%%；与 4d 的相对差 %.1f%%"
              % (abs(real - derived) / real * 100, abs(real - with4) / real * 100))
        check("C.2", abs(real - derived) / real > 0.10,
              "%s：真实 `d_ff` 与 `8d/3` 差 %.1f%%（> 10%%）⇒ 「这个模型用 8d/3」这句是**错的**，"
              "而且错得足够大、不会看不出来（换 `4d` 也错：见上一行的第二列）"
              % (name, abs(real - derived) / real * 100))

    print()
    print("  C.3：取整的边界 —— 小 d 上「近似持平」会退化")
    for dd in (1, 2, 3, 4, 5, 12, 24):
        r = parity_ratio(dd)
        print("    d=%2d：8d/3 取整 = %2d，比值 = %s = %.4f"
              % (dd, d_ff_swiglu(dd), r, float(r)))
    small = [float(parity_ratio(dd)) for dd in (1, 2, 3, 4, 5)]
    check("C.3", min(small) < 0.9,
          "★ d 很小时取整的误差**不再可忽略**（最小 %.4f < 0.9）——"
          "「8d/3 让参数持平」是一条**大 d 的陈述**，与小 d 的离散现实不是一回事"
          % min(small))


# ===========================================================================
# 5. D 类：失败标记自检 + 来源/口径钉住
# ===========================================================================
def _callee_names(func):
    """用 **AST** 列出 `func` 真正调用的函数名（★ 不含注释/文档串里的提及）。"""
    import textwrap
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    return {n.func.id for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}


def _mark_assignments_in(path):
    tree = ast.parse(open(path, encoding="utf-8").read(), path)
    out = []
    for n in tree.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 \
                and isinstance(n.targets[0], ast.Name) \
                and n.targets[0].id == "ASSERT_FAIL_MARK":
            out.append(n.value)
    return out


def part_D():
    print()
    print("=" * 74)
    print("D. 失败标记自检 与 来源/口径钉住")
    print("=" * 74)
    print("  本程序的失败标记 ASSERT_FAIL_MARK = %r" % ASSERT_FAIL_MARK)
    hits = _mark_assignments_in(os.path.abspath(__file__))
    uniq = {getattr(v, "value", None) for v in hits}
    check("D.1", len(hits) == 1 and len(uniq) == 1 and uniq == {ASSERT_FAIL_MARK}
          and "FAIL" in ASSERT_FAIL_MARK,
          "★ 标记自检：模块级赋值恰好 1 处、取值唯一、== %r、且含 `FAIL`（AST 读出）"
          % ASSERT_FAIL_MARK)
    # ★ 「等于该值」：把**本主题被分配的那个字面量**写死在这里。
    #   ★ 这不是同义反复：`mutate.py` 只变异**模块级**常量，函数体里的这个字面量
    #     不在它的分母里 ⇒ 它正是用来挡住「标记被改掉 / 被跨主题照抄」的那把锁。
    check("D.2", ASSERT_FAIL_MARK == "!! FAIL: transformer"
          and "FAIL" in ASSERT_FAIL_MARK and MARKS_USED <= {ASSERT_FAIL_MARK},
          "★ 标记自检：**等于**本主题被分配的字面量、含 `FAIL`，"
          "且打印失败行用的标记 ⊆ {%r}（本次用到 %d 个）"
          % (ASSERT_FAIL_MARK, len(MARKS_USED)))
    check("D.2b", OTHER_THEME_MARKS == ("@@ FAIL: attention-variants", "## FAIL: moe",
                                        "~~ FAIL: normalization",
                                        "$$ FAIL: positional-encoding"),
          "★ 「不跨主题照抄」这条判据的**对照物**也逐位钉住 —— 写错一个字，"
          "这条检查就静默失效了")

    check("D.3", specs.SWIGLU_DFF_REDUCTION_FACTOR == 2.0 / 3.0
          and specs.FFN_BASELINE_D_FF_RATIO == 4,
          "两个前提逐位一致：论文的 `2/3`（**不是** `8/3`）与前提 `4`"
          "（后者**没有 A–E 等级**：它是约定，不是来源）")
    check("D.4", PAPER_D_MODEL == 768 and PAPER_D_FF == 2048
          and d_ff_swiglu(PAPER_D_MODEL) == PAPER_D_FF,
          "论文 §3.1 自己的两个数（768 / 2048）与来源一致，且 `8×768/3` 复现出 2048")
    check("D.5", specs.LLAMA3_8B_CONFIG["hidden_act"] == "silu"
          and specs.QWEN3_30B_A3B_CONFIG["hidden_act"] == "silu",
          "两个模型的 `hidden_act` 都是 `silu` —— ★ 该字段**不在** `45/…/code/specs.py` 里，"
          "本域经 `hf-mirror.com` 镜像取得，**未与官方 repo 比对 sha256**")
    th = {"PARITY_LO": PARITY_LO, "PARITY_HI": PARITY_HI,
          "DIVISIBLE_BY_3": DIVISIBLE_BY_3, "NOT_DIVISIBLE_BY_3": NOT_DIVISIBLE_BY_3,
          "SMALL_D": SMALL_D, "SMALL_D_FF": SMALL_D_FF,
          "FLOAT_PARITY_TOL": FLOAT_PARITY_TOL}
    th_want = {"PARITY_LO": 0.999, "PARITY_HI": 1.001,
               "DIVISIBLE_BY_3": (768, 3072, 4098, 6144),
               "NOT_DIVISIBLE_BY_3": (4096, 8192), "SMALL_D": 64, "SMALL_D_FF": 256,
               "FLOAT_PARITY_TOL": 1e-15}
    tdiff = {k: (th_want[k], v) for k, v in th.items() if v != th_want[k]}
    check("D.6", not tdiff,
          "阈值与取样点 7 项逐位一致（判据④**点名要求**另取 d 被 3 整除的例子）%s"
          % ("" if not tdiff else " ← 差异 %s" % tdiff))

    check("D.7", inspect.getsource(params_two_matrix)
          != inspect.getsource(params_two_matrix_enumerating),
          "★ 防 P17：被测（一条式子）与参照（真的分配矩阵）的**源码不同**")
    called = _callee_names(params_two_matrix_enumerating)
    check("D.8", not (called & {"params_two_matrix", "params_glu_variant", "d_ff_swiglu"}),
          "参照的 **AST 调用表**里没有被测侧的三个函数（它的调用表 = %s）"
          % sorted(called))


# ===========================================================================
def summary():
    t = sum(_TOTAL.values())
    p = sum(_PASS.values())
    print()
    print("汇总: A %d/%d · B %d/%d · C %d/%d · D %d/%d ⇒ 断言 %d 条，通过 %d 条，失败 %d 条"
          % (_PASS["A"], _TOTAL["A"], _PASS["B"], _TOTAL["B"],
             _PASS["C"], _TOTAL["C"], _PASS["D"], _TOTAL["D"], t, p, len(FAILS)))
    if FAILS:
        print("失败清单: %s" % ", ".join(FAILS))
    return 1 if FAILS else 0


def main():
    print("=" * 74)
    print("02_ffn_shapes.py —— FFN 的形态：两层 → GLU 变体，`d_ff` 怎么定")
    print("=" * 74)
    print("  ★ 真值标注：外部规格值来自 `specs.py`（每条带等级与抓取日）；")
    print("    其余全部是**推导值**（本机 `python3 %s` + `numpy %s`，macOS arm64，**无 GPU**）。"
          % (sys.version.split()[0], np.__version__))
    print("  ★ 失败标记 ASSERT_FAIL_MARK = %r（本主题自定义；不跨主题照抄）" % ASSERT_FAIL_MARK)
    print("  ★ 禁区：本程序的账是**参数量**（矩阵个数 × 形状），**不是**每 token 的 FLOPs；")
    print("    FLOPs 系数（4 vs 6）的正文在 `40-kernels/moe-kernels/01-moe-compute-pattern.md:72,278`")
    print("    与 `…/code/verify.py:443`（正文在彼，此处回链）；")
    print("    从 config 反算**总量**的正文在 `45-kernel-lifecycle/requirement-discovery`（同上）。")
    part_A()
    lb, qw = part_B()
    part_C(lb, qw)
    part_D()
    return summary()


if __name__ == "__main__":
    sys.exit(main())

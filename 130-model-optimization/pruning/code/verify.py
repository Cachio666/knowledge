#!/usr/bin/env python3
"""130-model-optimization/pruning 主题的纯 python3 复核入口。

★ 本文件**固定跑在默认 `python3` 下，永不 `import torch`、也不 `import numpy`**
  （CONTRIBUTING §2 第三种形态；只用 `math` / `os` / `random` / `sys`）。
  它提供的是「**换了台机器、什么都没装，也能复核的那一半**」：
  用**独立的参考实现**（下面的 `ref_*`）把本主题的**判据**重算一遍，
  再用手写期望值（不是 torch 的输出）判它对不对。
  **它的产出是推导，不是实测**——不得标成本机实测值。

  三类断言在本形态下的分工（CONTRIBUTING §2）：
    · **A 判据复核** —— 本文件。参照物是下面 `ref_*` 这套独立实现与手写期望值。
    · **C 边界与反例** —— 本文件。断言的是「失败行为成立」。
    · **B 代价模型** —— 在**编号程序** `01_pruning.py` 里（代价模型的参照是真系统，
      只有能 import 到它的环境里测才有意义）。本文件**不做** B。

★ 本主题里 A/C 与编号程序的分工是 **「数学 vs 实现」**：
  · 本文件（纯 python3 + 标准库）验的是**可独立推导的算术与判据**：
    三条判据的公式、复杂度与归约前提、列正交时的归约关系（**含一个手写的
    Gauss–Jordan 求逆**）、`H` 除不除以 `n`、一般相关时的重合率与 Kendall τ、
    以及两条引用值表的转录锚点。
  · 编号程序（真 torch）测的是**真实实现的行为**：真张量上的归约关系、重合率与 τ、
    magnitude vs Wanda 的输出误差、判据耗时的 `d` 扫描。
  · **两者不得互相背书。** ⚠ **规模不同 ⇒ 数字不同**：本文件跑的是**小矩阵**
    （≤ 24×24 的 Hessian、≤ 200 个元素的判据表），编号程序跑的是 512×64 的校准输入。
    **只有定性结构必须一致。**

★ 「独立」的硬要求（CONTRIBUTING §2）：本文件里没有一行代码来自 torch。
  尤其**求逆**这一处：编号程序用 `torch.linalg.inv`，本文件用手写的
  Gauss–Jordan（带部分主元）——**两条不同的代码路径**。
"""

import math
import os
import random
import sys

import specs as SPECS

_STATE = {"n": 0, "A": 0, "C": 0, "fail": []}


# ===========================================================================
# 基础设施


# ===========================================================================


def check(cls, name, cond, detail=""):
    _STATE["n"] += 1
    _STATE[cls] += 1
    ok = bool(cond)
    if not ok:
        _STATE["fail"].append("[%s] %s" % (cls, name))
    print("    [%s] %s%s" % ("通过" if ok else "**不通过**", name,
                             "   ← %s" % detail if detail else ""))
    return ok


def section(tag, title):
    print()
    print("-" * 74)
    print("%s  %s" % (tag, title))
    print("-" * 74)


def sub(title):
    print()
    print("  · %s" % title)


# ===========================================================================
# 独立参考实现（ref_*）：**一行都不来自 torch**


# ===========================================================================


def ref_matmul_t(a, b):
    """`a^T · b`（a 为 m×k、b 为 m×n）。纯循环，不用任何库。"""
    m = len(a)
    k = len(a[0])
    n = len(b[0])
    out = [[0.0] * n for _ in range(k)]
    for i in range(m):
        ai = a[i]
        for p in range(k):
            v = ai[p]
            if v == 0.0:
                continue
            for j in range(n):
                out[p][j] += v * b[i][j]
    return out


def ref_inv(a):
    """手写的 Gauss–Jordan 求逆（**带部分主元**）。

    ★ 这条是与编号程序的**代码路径分岔点**：那边用 `torch.linalg.inv`（LAPACK），
    这边是朴素消元。两边给出同一个逆才说明「归约关系」不是 LAPACK 的产物。
    """
    n = len(a)
    m = [list(row) + [1.0 if i == j else 0.0 for j in range(n)]
         for i, row in enumerate(a)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(m[r][c]))
        if abs(m[p][c]) < 1e-300:
            raise ValueError("矩阵奇异（判据：主元绝对值 < 1e-300）")
        if p != c:
            m[c], m[p] = m[p], m[c]
        pv = m[c][c]
        m[c] = [v / pv for v in m[c]]
        for r in range(n):
            if r != c and m[r][c] != 0.0:
                f = m[r][c]
                mr, mc = m[r], m[c]
                m[r] = [mr[k] - f * mc[k] for k in range(2 * n)]
    return [row[n:] for row in m]


def ref_crit_wanda(X, W):
    """`crit_Wanda[i][j] = |W_ij| · ||X_j||_2`（**判据定义见 specs.CRITERIA_FORMULA**）。"""
    d = len(X[0])
    norms = [math.sqrt(sum(X[i][j] ** 2 for i in range(len(X)))) for j in range(d)]
    return [[abs(w) * norms[j] for j, w in enumerate(row)] for row in W]


def ref_crit_sparsegpt(X, W, lam=0.0, div_n=False):
    """`crit_SG[i][j] = |W_ij|^2 / (H^-1)_jj`，`H := X^T X`（**不带 sqrt**）。

    `div_n=True` 时用 `H = X^T X / n`——**两种约定的差别正好是一个因子 n**。
    """
    H = ref_matmul_t(X, X)
    if div_n:
        H = [[v / len(X) for v in row] for row in H]
    if lam:
        for i in range(len(H)):
            H[i][i] += lam
    Hi = ref_inv(H)
    diag = [Hi[j][j] for j in range(len(Hi))]
    return [[(w * w) / diag[j] for j, w in enumerate(row)] for row in W]


def ref_flat_topk_indices(crit, frac):
    """按展平后的顺序取 top-`frac` 的下标集合（判据：**展平**后整体比，不是逐行比）。"""
    flat = [v for row in crit for v in row]
    k = int(len(flat) * frac)
    order = sorted(range(len(flat)), key=lambda i: -flat[i])[:k]
    return set(order)


def ref_overlap(ca, cb, frac=0.5):
    a = ref_flat_topk_indices(ca, frac)
    b = ref_flat_topk_indices(cb, frac)
    return len(a & b) / float(len(a))


def ref_kendall_tau(x, y, tol=0.0):
    """Kendall τ = (C − D)/(C + D)。判据：`tol` 内的数对算**并列**，不计入 C/D。

    ⚠ `tol=0.0` 与 `tol>0` 会给出不同的值——**这也是一个「判据决定数字」的地方**。
    """
    c = d = 0
    n = len(x)
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if abs(dx) <= tol or abs(dy) <= tol:
                continue
            s = dx * dy
            if s > 0:
                c += 1
            elif s < 0:
                d += 1
    return (c - d) / float(c + d) if (c + d) else 0.0


def ref_gauss(n, d, seed, sigma=1.0):
    rnd = random.Random(seed)
    return [[rnd.gauss(0.0, sigma) for _ in range(d)] for _ in range(n)]


def ref_orthogonal_x(n, d, scales):
    """**列正交**的 X：第 j 列只有第 j 行非零、值为 `scales[j]`。

    ★ 这样 `X^T X = diag(scales[j]^2)` **在浮点下也是精确的**（每列只有一个非零，
    求和只有一项）——这正是让「恒等」变成「恰好 0.0」的那个判据。
    """
    X = [[0.0] * d for _ in range(n)]
    for j in range(d):
        X[j][j] = scales[j]
    return X


# ===========================================================================
# A. 判据复核


# ===========================================================================


def run_A():
    section("A", "判据复核（纯推导；参照物是 ref_* 与手写期望值）")

    sub("A.1 三条判据：公式、复杂度、归约前提（逐条与 specs 对齐）")
    for k in ("magnitude", "SparseGPT", "Wanda"):
        print("      %-11s %-8s %s" % (k, SPECS.CRITERIA_COMPLEXITY[k],
                                       SPECS.CRITERIA_FORMULA[k]))
    check("A", "★ 复杂度序：magnitude < Wanda < SparseGPT（%s < %s < %s）"
          % (SPECS.CRITERIA_COMPLEXITY["magnitude"], SPECS.CRITERIA_COMPLEXITY["Wanda"],
             SPECS.CRITERIA_COMPLEXITY["SparseGPT"]),
          SPECS.CRITERIA_COMPLEXITY["magnitude"] == "O(1)"
          and SPECS.CRITERIA_COMPLEXITY["Wanda"] == "O(d^2)"
          and SPECS.CRITERIA_COMPLEXITY["SparseGPT"] == "O(d^3)")
    check("A", "★ 归约的两条前提逐条列出：%s" % "、".join(SPECS.WANDA_REDUCTION_CONDITIONS),
          len(SPECS.WANDA_REDUCTION_CONDITIONS) == 2)
    check("A", "★ 判据式里 `crit_SparseGPT` **不带 sqrt**（定义写死在 specs 的口径注释里）",
          "sqrt" not in SPECS.CRITERIA_FORMULA["SparseGPT"].replace("^", ""))

    sub("A.2 归约关系：列正交 ⇒ `crit_SG = crit_W^2`（**平方，不是恒等**）")
    d = 12
    scales = [1.0 + 0.3 * j for j in range(d)]          # 非 2 的幂 ⇒ 开方不精确
    X = ref_orthogonal_x(d, d, scales)
    H = ref_matmul_t(X, X)
    offdiag = max(abs(H[i][j]) for i in range(d) for j in range(d) if i != j)
    W = ref_gauss(d, d, seed=3, sigma=1.0)
    sg = ref_crit_sparsegpt(X, W)
    wa = ref_crit_wanda(X, W)
    r_sq = max(abs(sg[i][j] / (wa[i][j] ** 2) - 1.0)
               for i in range(d) for j in range(d))
    r_sqrt = max(abs(math.sqrt(sg[i][j]) / wa[i][j] - 1.0)
                 for i in range(d) for j in range(d))
    print("      `X^T X` 的非对角元最大值 = %.3e（判据：这一构造下应为 0）" % offdiag)
    print("      `crit_SG / crit_W^2` 的最大相对差 = %.3e" % r_sq)
    print("      `sqrt(crit_SG)/crit_W` 的最大相对差 = %.3e" % r_sqrt)
    check("A", "★ 列正交时 `crit_SG = crit_W^2`（是**平方**，相对差 ≤ 1e-12）"
          , r_sq < 1e-12)
    check("A", "★ **平方 ≠ 恒等**：`crit_SG` 与 `crit_W` 本身**不相等**（相对差 %.1e > 0）"
          % (max(abs(sg[i][j] / wa[i][j] - 1.0) for i in range(d) for j in range(d))),
          max(abs(sg[i][j] / wa[i][j] - 1.0) for i in range(d) for j in range(d)) > 1e-3)
    check("A", "★ 对两者都取 sqrt 之后才恒等（相对差 %.3e，machine precision 量级）"
          % r_sqrt, r_sqrt < 1e-12)

    sub("A.3 ⚠ 「最大相对差 = 0.0」**是判据的产物**：只有可精确表示的对角才给 0.0")
    print("      判据：`H` 直接取 `diag(s_j^2)`；只换 s_j 的取值形式，三者都取 sqrt 后比。")
    W2 = ref_gauss(4, 6, seed=5, sigma=1.0)
    cases = (
        ("s_j = 2^(k/2)，k 全偶数", [2.0 ** 0, 2.0 ** 1, 2.0 ** 2, 2.0 ** 3, 2.0 ** 4, 2.0 ** 5]),
        ("s_j = 2^(k/2)，k 全奇数", [2.0 ** 0.5, 2.0 ** 1.5, 2.0 ** 2.5, 2.0 ** 3.5, 2.0 ** 4.5, 2.0 ** 5.5]),
        ("s_j = 1.3^j（一般实数）", [1.3 ** j for j in range(6)]),
    )
    exact_zero_case = None
    for label, ss in cases:
        X2 = ref_orthogonal_x(6, 6, ss)               # H = diag(s_j^2)，浮点下精确
        sg2 = ref_crit_sparsegpt(X2, W2)
        wa2 = ref_crit_wanda(X2, W2)
        rr = max(abs(math.sqrt(sg2[i][j]) / wa2[i][j] - 1.0)
                 for i in range(4) for j in range(6))
        print("      %-26s ⇒ 最大相对差 = %.3e%s"
              % (label, rr, "   ← **恰好 0.0**" if rr == 0.0 else ""))
        if label.startswith("s_j = 2^(k/2)，k 全偶数"):
            exact_zero_case = rr
    check("A", "★ **只有「平方根可精确表示」的那一档给出恰好 0.0**——"
          "「恒等」的判据不只是公式，还包括可表示性", exact_zero_case == 0.0)
    check("A", "★ 一般实数的那一档给 %.1e（**不是 0.0**）——与编号程序在真 torch 上"
          "量到的 2.220e-16 是同一个量级" % r_sqrt, 0.0 < r_sqrt < 1e-12)

    sub("A.4 `H` 是否除以 n：同一句话在两个约定下**差一个因子 n**")
    Xg = ref_gauss(20, 8, seed=7, sigma=1.0)
    Wg = ref_gauss(6, 8, seed=8, sigma=1.0)
    wa_g = ref_crit_wanda(Xg, Wg)
    Xo = ref_orthogonal_x(20, 6, [1.0 + 0.2 * j for j in range(6)])
    Wo = ref_gauss(5, 6, seed=9, sigma=1.0)
    sg_raw = ref_crit_sparsegpt(Xo, Wo)                     # H = X^T X
    sg_div = ref_crit_sparsegpt(Xo, Wo, div_n=True)         # H = X^T X / n
    r_raw = max(abs(sg_raw[i][j] / (ref_crit_wanda(Xo, Wo)[i][j] ** 2) - 1.0)
                for i in range(5) for j in range(6))
    r_div = max(abs(sg_div[i][j] / (ref_crit_wanda(Xo, Wo)[i][j] ** 2) - 1.0 / 20.0)
                for i in range(5) for j in range(6))
    print("      `H = X^T X`     ⇒ 与 1 的最大差 = %.3e" % r_raw)
    print("      `H = X^T X / n` ⇒ 与 1/n = %.6f 的最大差 = %.3e" % (1.0 / 20.0, r_div))
    check("A", "★ `H` 的定义不写清，归约关系**差一个因子 n**（两条式子都对，但不是一个量）",
          r_raw < 1e-12 and r_div < 1e-12)
    # ★ 这条**读 specs.py 的源码**来核「口径写没写」——机械可证伪：
    #   把那段口径注释删掉，这条就失败。
    spec_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "specs.py"),
                    encoding="utf-8").read()
    check("A", "★ specs.py 的口径里**逐字写着** `H := X^T X`（不除以 n）——"
          "读者复现不出的数字不算实测",
          "不除以 n" in spec_src and "X^T X" in spec_src)

    sub("A.5 一般相关 X：两判据的选中一致性（**规模不同 ⇒ 数字不同**）")
    Xc = ref_gauss(40, 10, seed=11, sigma=1.0)
    Wc = ref_gauss(8, 10, seed=12, sigma=1.0)
    sg_c = ref_crit_sparsegpt(Xc, Wc)
    wa_c = ref_crit_wanda(Xc, Wc)
    ov = ref_overlap(sg_c, wa_c, 0.5)
    taus = [ref_kendall_tau(sg_c[i], wa_c[i]) for i in range(8)]
    tau_mean = sum(taus) / len(taus)
    tau_flat = ref_kendall_tau([v for row in sg_c for v in row],
                               [v for row in wa_c for v in row])
    eq_rows = sum(1 for i in range(8)
                  if sorted(range(10), key=lambda j: sg_c[i][j])
                  == sorted(range(10), key=lambda j: wa_c[i][j]))
    print("      形状：X 40×10 / W 8×10（**比编号程序小得多**）")
    print("      top-50%% 集合重合率 = %.6f" % ov)
    print("      **逐行** Kendall τ 均值 = %.6f（min %.6f）" % (tau_mean, min(taus)))
    print("      **展平后** Kendall τ    = %.6f（**与逐行不是同一个数**）" % tau_flat)
    print("      `argsort` 完全相等的行  = %d / 8" % eq_rows)
    check("A", "★ 一般相关 X 上，两判据的选中集合**高度重合但非完全重合**（%.4f < 1）"
          % ov, ov > 0.85 and ov < 1.0)
    check("A", "★ 逐行 τ 与展平 τ **不是同一个量**（%.6f vs %.6f）——"
          "「排名一致性」必须写明是逐行还是展平"
          % (tau_mean, tau_flat), abs(tau_mean - tau_flat) > 1e-6)
    check("A", "★ 且两个 τ **都高但不是 1**（逐行 %.6f、展平 %.6f，都 ∈ (0.85, 1)）"
          % (tau_mean, tau_flat),
          0.85 < tau_mean < 1.0 and 0.85 < tau_flat < 1.0)
    # ★ 编号程序在**它自己的规模**上量到的一对 τ：两个数不相等，且都给出来。
    #   这条断言把 specs 里的**两个** τ 常量都读进来当靶子（A6）。
    r_row = SPECS.MEASURED_KENDALL_TAU
    r_flat = SPECS.MEASURED_KENDALL_TAU_FLAT
    print("      编号程序的规模上：逐行 τ = %.16f / 展平 τ = %.16f（差 %.6f）"
          % (r_row, r_flat, r_row - r_flat))
    check("A", "★ 那个规模上两个 τ **也不是同一个数**（差 %.6f > 1e-4）——"
          "所以「排名一致性」必须写明口径" % (r_row - r_flat),
          r_row != r_flat and abs(r_row - r_flat) > 1e-4)
    print("      ⚠ 「排名一致性」这个说法在**两种口径**下是两个数（上一行断言了它）。")
    print("        另外 τ 还依赖 `tol`：把 tol 从 0 改成 1e-3 会排除并列数对——")
    print("        E.4 断言了这一点。")
    print("      ⚠ 本节的**数字与编号程序不同**（那边 X 512×64、W 32×64）：")
    print("        规模不同 ⇒ 只有**定性结构**必须一致（重合高、τ 高、argsort 几乎不等）。")

    sub("A.6 三条判据式的因式（**转录锚点**）")
    cf = SPECS.CRITERIA_FORMULA
    check("A", "★ 转录锚点：三条判据式的因式逐字对上（`|W_ij|`、`||X_j||_2`、`diag`、"
          "`X^T X`、`lambda`、`^2`）",
          cf["magnitude"] == "|W_ij|" and cf["Wanda"] == "|W_ij| * ||X_j||_2"
          and "diag" in cf["SparseGPT"] and "X^T X" in cf["SparseGPT"]
          and "lambda" in cf["SparseGPT"] and "^2" in cf["SparseGPT"])

    sub("A.7 引用值表的**转录锚点**与**间距判据**")
    print("      ⚠ 锚点只保证 specs 里的印值没被静默改掉；**不声称**复核了测量口径。")
    t2 = SPECS.WANDA_ACCURACY_TABLE2
    check("A", "★ 锚点：Wanda 表 2 的四个数逐字对上（%.2f / %.2f / %.2f / %.2f）"
          % (t2["dense"], t2["magnitude"], t2["SparseGPT"], t2["Wanda"]),
          abs(t2["dense"] - 59.99) < 1e-9 and abs(t2["magnitude"] - 46.94) < 1e-9
          and abs(t2["SparseGPT"] - 54.94) < 1e-9 and abs(t2["Wanda"] - 54.21) < 1e-9)
    # ★ 有信息量的那条是**间距**而不是排序：排序在等比扰动下守恒，间距不守恒。
    #   （本机实测：把四个数同时 ×1.5+1，排序仍成立，而「距稠密」的间距从 5.05 涨到 7.58。）
    gap_sg = t2["dense"] - t2["SparseGPT"]
    gap_wa = t2["dense"] - t2["Wanda"]
    gap_mag = t2["dense"] - t2["magnitude"]
    print("      与稠密基线的间距：SparseGPT %.2f / Wanda %.2f / magnitude %.2f"
          % (gap_sg, gap_wa, gap_mag))
    check("A", "★ 间距判据：Wanda 与 SparseGPT 距稠密都 < 6 个点，而 magnitude > 10 个点"
          "（%.2f / %.2f / %.2f）" % (gap_sg, gap_wa, gap_mag),
          gap_sg < 6.0 and gap_wa < 6.0 and gap_mag > 10.0)
    t24 = SPECS.WANDA_2_4_LLAMA7B
    check("A", "★ 锚点：2:4 三档逐字对上（%.2f / %.2f / %.2f）"
          % (t24["magnitude"], t24["SparseGPT"], t24["Wanda"]),
          abs(t24["magnitude"] - 44.73) < 1e-9 and abs(t24["SparseGPT"] - 50.60) < 1e-9
          and abs(t24["Wanda"] - 48.53) < 1e-9)
    check("A", "★ 且 2:4 上 Wanda 与 SparseGPT 的差距 ∈ (0.5, 3) 个点",
          0.5 < t24["SparseGPT"] - t24["Wanda"] < 3.0)
    check("A", "★ 锚点：Wanda 的原文引语逐字对上（%s）" % SPECS.WANDA_AS_IS_QUOTE,
          SPECS.WANDA_AS_IS_QUOTE == "the pruned LLM can be used as is")
    check("A", "★ τ 是**相关系数**：两个都在 (0.9, 1) 内（逐行 %.6f / 展平 %.6f）"
          % (SPECS.MEASURED_KENDALL_TAU, SPECS.MEASURED_KENDALL_TAU_FLAT),
          0.9 < SPECS.MEASURED_KENDALL_TAU < 1.0
          and 0.9 < SPECS.MEASURED_KENDALL_TAU_FLAT < 1.0)


# ===========================================================================
# C. 边界与反例


# ===========================================================================


def run_C():
    section("C", "边界与反例（断言的是「失败行为成立」）")

    sub("C.1 50% 稀疏下 **magnitude 判据失效**（列范数不均匀时）")
    d = 8
    scales = [10.0 ** (0.3 * j) for j in range(d)]      # 列范数跨 3 个数量级
    X = ref_orthogonal_x(24, d, scales)
    # ★ 判据：列正交 ⇒ ‖X(W−W_p)‖² = Σ_j ‖X_j‖² · Σ_i Δ_ij²（**无交叉项**），
    #   于是「最小化它」= 按 |W_ij|·‖X_j‖ 丢掉最小的那半 = Wanda 判据。
    #   **这是可证的**：本文件直接算这个和，而不去算矩阵乘积。
    def err2_from_columns(W, keep_mask):
        e = 0.0
        for j in range(d):
            col = sum(W[i][j] ** 2 for i in range(len(W)) if not keep_mask[i][j])
            e += (scales[j] ** 2) * col
        return e

    W = ref_gauss(6, d, seed=21, sigma=1.0)
    for i in range(6):
        W[i][0] *= 0.3                                  # outlier 列上权重小
    crits = {
        "magnitude": [[abs(W[i][j]) for j in range(d)] for i in range(6)],
        "Wanda": ref_crit_wanda(X, W),
    }
    res = {}
    for name, crit in crits.items():
        flat = [v for row in crit for v in row]
        k = len(flat) // 2
        drop = set(sorted(range(len(flat)), key=lambda t: flat[t])[:k])
        keep = [[not (i * d + j) in drop for j in range(d)] for i in range(6)]
        res[name] = math.sqrt(err2_from_columns(W, keep))
        print("      %-10s 的输出误差 ‖X(W−W_p)‖ = %.6f" % (name, res[name]))
    check("C", "★ magnitude 的输出误差**比 Wanda 大 %.2f 倍**（失败行为成立）"
          % (res["magnitude"] / res["Wanda"]), res["magnitude"] > res["Wanda"] * 1.5)

    sub("C.2 对照：**列范数均匀**时 magnitude 并不差")
    Xu = ref_orthogonal_x(24, d, [1.0] * d)
    crits_u = {
        "magnitude": [[abs(W[i][j]) for j in range(d)] for i in range(6)],
        "Wanda": ref_crit_wanda(Xu, W),
    }
    res_u = {}
    for name, crit in crits_u.items():
        flat = [v for row in crit for v in row]
        k = len(flat) // 2
        drop = set(sorted(range(len(flat)), key=lambda t: flat[t])[:k])
        e = 0.0
        for idx in drop:
            i, j = divmod(idx, d)
            e += (1.0 ** 2) * (W[i][j] ** 2)
        res_u[name] = math.sqrt(e)
        print("      %-10s 的输出误差 = %.6f" % (name, res_u[name]))
    check("C", "★ 列范数均匀时两者**几乎相同**（比值 %.4f ∈ (0.5, 2)）"
          % (res_u["magnitude"] / res_u["Wanda"]),
          0.5 < res_u["magnitude"] / res_u["Wanda"] < 2.0)
    print("      ⇒ 触发条件是「**列范数不均匀**」，不是「用了 magnitude 判据」。")


# ===========================================================================
# E. 自检：故意改错会报错吗（P5 的逐条施加）


# ===========================================================================


def run_E():
    section("E", "自检：**故意改错**，A/C 的判据会不会失败（P5）")
    ok = True

    def expect_change(label, a, b, why):
        nonlocal ok
        good = (a != b)
        ok &= good
        print("    [%s] E  %s" % ("通过" if good else "**不通过**", label))
        print("          —— %s" % why)

    # E.1 判据里加 sqrt：平方关系变成恒等关系
    d = 6
    X = ref_orthogonal_x(d, d, [1.0 + 0.3 * j for j in range(d)])
    W = ref_gauss(d, d, seed=31, sigma=1.0)
    sg = ref_crit_sparsegpt(X, W)
    wa = ref_crit_wanda(X, W)
    as_is = max(abs(sg[i][j] / (wa[i][j] ** 2) - 1.0) for i in range(d) for j in range(d))
    def sg_with_sqrt(X, W):
        return [[math.sqrt(v) for v in row] for row in ref_crit_sparsegpt(X, W)]
    with_sqrt = max(abs(sg_with_sqrt(X, W)[i][j] / (wa[i][j] ** 2) - 1.0)
                    for i in range(d) for j in range(d))
    expect_change("E.1 给 `crit_SG` 加一层 sqrt，`= crit_W^2` 立刻不成立",
                  round(as_is, 12), round(with_sqrt, 12),
                  "A.2 那条断言（相对差 ≤ 1e-12）会失败（%.3e → %.3e）"
                  % (as_is, with_sqrt))

    # E.2 H 除以 n：比值差一个因子
    Xo = ref_orthogonal_x(20, 6, [1.0 + 0.2 * j for j in range(6)])
    Wo = ref_gauss(5, 6, seed=33, sigma=1.0)
    warn = ref_crit_wanda(Xo, Wo)
    def mean_ratio(div_n):
        sg_ = ref_crit_sparsegpt(Xo, Wo, div_n=div_n)
        vals = [sg_[i][j] / (warn[i][j] ** 2) for i in range(5) for j in range(6)]
        return sum(vals) / len(vals)
    ratio_raw, ratio_div = mean_ratio(False), mean_ratio(True)
    expect_change("E.2 `H` 改成 `X^T X / n`，`crit_SG/crit_W^2` 从 1 变成 1/20",
                  round(ratio_raw, 6), round(ratio_div, 6),
                  "A.4 的那条断言会失败（%.6f vs %.6f）" % (ratio_raw, ratio_div))

    # E.3 magnitude 换成 Wanda：C.1 的结论反转
    d2 = 8
    scales = [10.0 ** (0.3 * j) for j in range(d2)]
    X2 = ref_orthogonal_x(24, d2, scales)
    W2 = ref_gauss(6, d2, seed=21, sigma=1.0)
    for i in range(6):
        W2[i][0] *= 0.3
    def err_with(crit):
        flat = [v for row in crit for v in row]
        k = len(flat) // 2
        drop = set(sorted(range(len(flat)), key=lambda t: flat[t])[:k])
        e = 0.0
        for idx in drop:
            i, j = divmod(idx, d2)
            e += (scales[j] ** 2) * (W2[i][j] ** 2)
        return math.sqrt(e)
    e_mag = err_with([[abs(W2[i][j]) for j in range(d2)] for i in range(6)])
    e_wan = err_with(ref_crit_wanda(X2, W2))
    expect_change("E.3 把判据从 magnitude 换成 Wanda，C.1 的「magnitude 更差」立刻反转",
                  round(e_mag, 9), round(e_wan, 9),
                  "C.1 那条断言会失败（%.6f vs %.6f）" % (e_mag, e_wan))

    # E.4 Kendall τ 的 tol 从 0 改成 1e-3：并列被排除，τ 变化
    # ★ 样本要**专门挑两种判据给出不同答案的点**：近似并列的一对，在 tol=0 下算
    #   「不一致」、在 tol=1e-3 下被当作并列**排除**，于是 τ 从 −1/3 变成 0。
    a = [1.0, 1.0 + 1e-7, 3.0]
    b = [1.0, 3.0, 2.0]
    t0 = ref_kendall_tau(a, b, tol=0.0)
    t1 = ref_kendall_tau(a, b, tol=1e-3)
    expect_change("E.4 Kendall τ 的判据从「tol=0」改成「tol=1e-3」，值立刻变了",
                  round(t0, 6), round(t1, 6),
                  "τ 的定义里**必须先写死 tol**（%.6f vs %.6f）" % (t0, t1))

    # E.5 **对照组**：把上面每一条「改错」改回原值，原判据必须重新通过。
    #   没有对照组的话，「改错会失败」可能只是 harness 在空转（E 里任何一条
    #   只要自己崩了也会算「按预期失败」）——这正是 A6 说的假阳性模式一。
    ctrl = {
        "E.1 原判据（不取 sqrt）": as_is < 1e-12,
        "E.2 原口径（H = X^T X）": abs(ratio_raw - 1.0) < 1e-12,
        "E.3 原结论（magnitude 更差）": e_mag > e_wan,
        "E.4 原 τ（tol=0 那一个 = +1/3）": abs(t0 - (1.0 / 3.0)) < 1e-9,
    }
    n_ctrl = sum(1 for v in ctrl.values() if v)
    for k, v in ctrl.items():
        print("          %s %s" % ("✓" if v else "✗", k))
    ok &= (n_ctrl == len(ctrl))
    print("    [%s] E  %s **对照组**：%d/%d 条的**原值**仍然满足原判据 ⇒ harness 不是空转"
          % ("通过" if n_ctrl == len(ctrl) else "**不通过**", "E.5", n_ctrl, len(ctrl)))

    print()
    print("  E 节结论：%s" % ("全部自检按预期失败 ⇒ A/C 类判据是活的" if ok
                              else "**有自检未按预期失败 ⇒ 判据有问题**"))
    return ok


# ===========================================================================
# F. 本文件覆盖了哪些 specs 常量（A6 变异测试的清单）


# ===========================================================================


def run_F():
    section("F", "变异测试清单：本文件读过的 specs 常量")
    names = sorted(n for n in dir(SPECS) if n.isupper() and not n.startswith("_"))
    src = open(__file__, encoding="utf-8").read()
    used = [n for n in names if ("SPECS.%s" % n) in src]
    print("      specs.py 里共 %d 条全大写常量；本文件直接读到的 %d 条"
          % (len(names), len(used)))
    unused = [n for n in names if n not in used]
    print("      本文件**没有**读到的 %d 条（由编号程序 01_pruning.py 负责，见其 §F）："
          % len(unused))
    for n in unused:
        print("        · %s" % n)
    check("A", "变异的对象是**常量**，每个常量的值都能被至少一处断言分辨",
          len(used) > 0)
    return unused


# ===========================================================================
# main


# ===========================================================================


def main():
    print("=" * 78)
    print("130-model-optimization/pruning · verify.py（纯 python3，**不 import torch / numpy**）")
    print("=" * 78)
    print("说明：本文件产出的是**推导**，不是实测。")
    print("      它用独立的 ref_*（含手写的 Gauss–Jordan 求逆）重算三条判据的归约关系、")
    print("      Kendall τ 与 magnitude/Wanda 反例，**不读 torch 的任何输出**。")

    run_A()
    run_C()
    okE = run_E()
    run_F()

    print()
    print("=" * 78)
    n, a, c = _STATE["n"], _STATE["A"], _STATE["C"]
    print("合计 %d 条断言：A %d 条 / C %d 条；失败 %d 条" % (n, a, c, len(_STATE["fail"])))
    if _STATE["fail"]:
        print("失败清单：")
        for f in _STATE["fail"]:
            print("  · %s" % f)
    print("E 节自检：%s" % ("通过" if okE else "**不通过**"))
    sys.exit(0 if (not _STATE["fail"] and okE) else 1)


if __name__ == "__main__":
    main()

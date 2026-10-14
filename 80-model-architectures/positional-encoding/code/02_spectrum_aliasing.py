#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""02_spectrum_aliasing.py —— `θ_i` 的**谱**、**混叠周期**、**分辨窗口**。

★ 本程序给的是**定量框架**（计划 §3 Chunk 5 的分工写死：`02` 给定量框架，
  `03` 用它解释工程方法）。★ **本程序不讲任何外推方法**（PI / NTK / YaRN 一律不出现
  在正文里；只有 `C.3` 末尾一句**指路**）。
配套文档：`02-spectrum-and-aliasing.md`。

===============================================================================
★★ 本程序要回答的那三个问题
===============================================================================

  1. **`θ_i` 这条谱长什么样、`base` 动得了它的哪一端？**（判据 ③）
     `θ_i = base^{-2(i-1)/d}`、`i ∈ [1, d/2]` ⇒ **`θ₁ = base⁰ = 1`，与 `base` 无关**
     （在 `base ∈ {1e4, 5e5, 1e6}` 上**精确** `== 1.0`）。
     ⇒ ★★ **提高 `base` 只把低频端拉低**（`λ_{d/2}` 变大），**动不了最高频那一维**。
     这正是两个真实模型 `rope_theta` 差 2 倍（`5e5` vs `1e6`）却**都**受同一个上限约束的原因。

  2. **为什么「位置 `m` 与 `m + 2π` 不可区分」？**（判据 ④）
     因为 `θ₁` **精确等于 1** ⇒ 第 1 对的**周期精确等于 `2π`**。
     ⇒ 在**最高频分量**上，`m` 与 `m + 2π` 是**同一个位置**（实测相对差 `< 1e-12`）。
     ★ 这条与 `C.1` 的负对照合起来才完整：换个位移（`π` 而不是 `2π`）它**不闭合**。

  3. **为什么「两端闭合、中间不闭合」，而且两端的原因相反？**（判据 ⑤ + `B.4`）
     ★★ 本程序给出一个**闭式**（这是本程序最有用的一条）：
     把第 `i` 对的贡献写成 `c_i(δ) = R_i·cos(δθ_i − φ_i)`，则

         max_δ |c_i(δ + 2π) − c_i(δ)|  ==  **2·|sin(π·θ_i)|·R_i**     （恒等）

     ⇒ **一个式子解释了三个频段**：
       · `θ_i` **是整数**（本谱里**只有 `i = 1`**，因为 `θ₁ = 1`）⇒ `sin(π·1) = 0`
         ⇒ 闭合是**精确的**（机器精度）；
       · `θ_i·2π ≪ 1`（低频端）⇒ `2|sin(πθ_i)| ≈ 2πθ_i` **很小** ⇒ 近似闭合
         （★ 与高频端的**原因相反**：那一端是「正好一整圈」，这一端是「连一小段都没转完」）；
       · `θ_i` 在 (0,1) 的**中间段** ⇒ `2|sin(πθ_i)|` 是 `O(1)` ⇒ **不闭合**。
     ★★ 而计划 §3 表 A ⑤ 给的那个 `err_i = max_n|Δ| / mean_n|c|` 是**另一个口径**：
       它的分子取**样本上的 max**、分母取**样本上的 mean** ⇒ 它不是「相对误差」，
       会把闭式**放大一个 ~5~7.5 倍的样本离散因子**（`B.4b` 把这件事量出来）。
       ⇒ 两个口径**都写出来**（各自都有断言），并且**说清它们是两个量**。

===============================================================================
★★ 本程序的数字与它们的来源等级（三类标签必须分开）
===============================================================================

  · **本机实测值 / 推导值**：`python3 3.13.5` + `numpy 2.4.4`，macOS arm64，**无 GPU**。
    ⇒ **不支撑任何关于真实模型或 GPU 的结论。**
  · **外部规格值**：两个真实模型的 `rope_theta` 与 `max_position_embeddings`
    （`specs.py`，**经镜像取得**）。
  · **论文逐字**：`specs.py` 的 RoPE 那批 [C] 常量（arXiv 2104.09864，**未确认同行评审**）。

===============================================================================
★★ 可证伪断言（把下列任一处改错，本程序必须非零退出）
===============================================================================

  · `A.1`  ③ `θ₁ == 1.0` **精确成立**，在 `base ∈ {10000, 500000, 1000000}` **三档**上
           （★ 后两档就是两个真实模型的 `rope_theta`）；
  · `A.2`  ★★ `θ₁` **与 `base` 无关**，而 `θ_{d/2}`（最低频）**随 `base` 单调下降**
           ⇒ 「提高 `base` 只动低频那一端」是一对**两条**断言，不是一句概括；
  · `A.3`  ★ 谱的**单调性**：`θ_i` 严格递减、`λ_i = 2π/θ_i` 严格递增；
  · `A.4`  ★ **分辨窗口**（与计划 D6 同口径）：`i=1` 在 `S=128` 内转 `128 rad`（`20.4` 圈），
           `i=32` 在 `S=4096` 内只转 `0.546 rad`（`0.087` 圈）—— 一个是**早就转过很多圈**、
           一个是**连一小段都没转完**；
  · `B.1`  ④ 位置 `m` 与 `m+2π` 在**最高频分量**上的分数：**尺度无关**的相对差 `< 1e-12`
           （`m` 走四档）；
  · `B.2`  ⑤ 计划 §3 表 A ⑤ 的闭合误差谱：`err_1 < 0.01`、`err_32 < 0.01`，
           而 `err_i > 1.0` 对 `i ∈ {2,3,4,8}` —— ★ 四个中间点**逐个**断言；
  · `B.3`  ★ `i = 16` 落在**过渡带**（不在上面任何一组里）⇒ ★ **不**断言它
           —— 这正是计划原稿把四个数写成「`i = 2..8`」时**范围与个数不符**的原因；
  · `B.4`  ★★ **闭式**：`max_δ|c_i(δ+2π) − c_i(δ)| == 2·|sin(π·θ_i)|·R_i`（**恒等**），
           在 `i ∈ {2,3,4,8,16,32}` 上逐位成立（`i=1` 的两边都在机器精度档，另断言）；
  · `B.4b` ★★ **量法口径注记**：`err_i / (2|sin(πθ_i)|)` 在 `i ≥ 2` 上**恒 `> 1`** 且
           **跨 `1.2` 倍以上**（实测 `5.11 ~ 7.57`）⇒ 那个因子**不是 `θ` 的性质**，
           是「分子取 max、分母取 mean」的口径混用（★ 计划 §0.1 第一类错误的又一现场）；
  · `C.1`  ★★ **判据④ 不是恒真断言**：把位移换成 `π`（半个周期）⇒ 同一个量从 `< 1e-12`
           掉到 `> 0.1`；
  · `C.2`  ★★ **两端互斥（可证的不等式，不是测量）**：「相邻位置可分」（`θ_i ≥ 1` rad）与
           「整窗不混叠」（`turns_i(L) ≤ 1`）这两个集合，对 `L > 2π` 的**任何** `L`
           **交集必空**（因为 `2π/L < 1`）⇒ 位置编码**必须**是一条谱，而谱的两端**必然**
           分别落进两个失败模式；
  · `C.3`  ★ 真实模型的**计数**（`d = 128` = 两个模型的 rotary 维数，与探测脚本的
           `d = 64` **是两个口径**）：Qwen3（`base = 1e6`、`L = 40960`）「整窗不混叠」
           **23** 维（`i = 42..64`）、「可分辨相邻位置」**1** 维（`i = 1`）；
           Llama（`5e5`、`8192`）是 **29** 维 / **1** 维；两者的交集都是 **0**。

===============================================================================
★★ 本程序刻意不做的事（计划 §2 的边界表，越界即作废）
===============================================================================

  * ★★ **不讲任何外推方法**：PI / NTK-aware / NTK-by-parts / YaRN / ALiBi 的**定义与对比
    全在 `03`**。本程序只给 `03` 要用的**定量框架**（谱、混叠、分辨窗口、`turns`）。
  * **不写**「成对通道旋转」那句结构陈述的正文（在
    `130-model-optimization/quantization/19-kivi-asymmetry.md:184`）。
  * **不算**位置编码的 FLOPs / 访存账（正文在 `40-kernels/attention`）。
  * ★ **不做**「谱的哪一段最重要」这类**没有判据**的排序 —— 本程序只给**可判定的计数**。

用法：
    python3 02_spectrum_aliasing.py
"""

import ast
import collections
import inspect
import os
import sys

import numpy as np

import specs

# ===========================================================================
# 0. 断言记账
# ===========================================================================
ASSERT_FAIL_MARK = "$$ FAIL: positional-encoding"

OTHER_THEME_MARKS = ("!! FAIL: transformer", "@@ FAIL: attention-variants",
                     "## FAIL: moe", "~~ FAIL: normalization")

FAILS = []
MARKS_USED = set()
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
# 1. 口径常量（★ 全部在 `D.1`/`D.2` 被逐位钉住；★★ 全部正数存放）
# ===========================================================================
# ★★ **本主题的第二个 `mutate` 盲区**（计划 §3 开头）：叶子枚举只取 `ast.Constant`，
#    而 `-1` 是 `UnaryOp(USub, Constant(1))` ⇒ **负数表整体免检**。
#    所以下面这张表里**没有负数**：频段用**一基下标**（`i = 1, 2, 3, …`，全是正数）表达，
#    位移用**正的弧度常数**表达。
D_SPEC = 64                # 谱的维度（与探测脚本 probe_80f.py 同口径，D3–D6 出自它）
# ★★ **两个 `d` 必须分开、各自标口径**：
#   · `D_SPEC = 64` —— **探测脚本的口径**（计划 §0.7-D 的 D3–D6 全出自 `probe_80f.py`，
#     它的 `d = 64`）。判据 ③④⑤ 与 `B.4` 的闭式都用它，才能与 D3–D6 **逐位对照**。
#   · `D_MODEL = 128` —— **两个真实模型的 rotary 维数**（见 `specs.ROTARY_DIM_MODEL`）。
#     `C.2`/`C.3` 的计数是关于**真实模型**的陈述 ⇒ 只能用这个值。
#   ⇒ 用错哪一个都**不会报错**，只会静默换掉结论（实测：Qwen3 的「整窗不混叠」维数
#     在 `d = 64` 时是 **11**、在 `d = 128` 时是 **23** —— **差一倍**）。
D_MODEL = 128
BASE_LADDER = (10000.0, 500000.0, 1000000.0)  # ★ 后两档 = 两个真实模型的 `rope_theta`
N_PAIR = 200               # 闭合误差谱的样本数（与 probe_80f 同口径）
BASE_DELTA = 3.0           # 闭合误差谱的 `δ`（与 probe_80f 同口径）
TURN_RAD = 6.283185307179586   # 一整圈（`2π`）—— ★ 写成**正的字面量**，不进表达式
HALF_TURN_RAD = 3.141592653589793  # 半圈（`π`）—— `C.1` 的负对照用
M_LADDER = (10.0, 50.0, 200.0, 1000.0)  # 判据④ 的 `m` 四档（与 probe_80f 同口径）
S_LADDER = (128, 4096)     # 分辨窗口的两个上下文长度（与计划 D6 同口径）
SPEC_INDEX = (1, 2, 3, 4, 8, 16, 32)   # 打印用的 1-based 对号（★ 全是正数）
CLOSING_INDEX = (1, 32)    # ⑤ 里判「闭合」的两端（1-based）
NONCLOSING_INDEX = (2, 3, 4, 8)   # ⑤ 里判「不闭合」的中间段（1-based）—— ★ 四个，逐个断言
TRANSITION_INDEX = 16      # ★ 落在过渡带的那一个（**不断言**，只打印）
ADJ_PHASE_THRESHOLD = 1.0  # [E] 「相邻位置可分」的阈值（相位步进 ≥ 1 rad）
ADJ_PHASE_THRESHOLD_LO = 0.1    # `C.3` 的第二个阈值（★ 用来证明结论不挑阈值）
ADJ_PHASE_THRESHOLD_LOLO = 0.01  # `C.3` 的第三个阈值
TURNS_LIMIT = 1.0          # [E] 「整窗不混叠」的阈值（整段上下文里最多转 1 圈）
SEED = 20260922            # 主数据的种子（与 probe_80f 同源）

# ── 阈值（★ 逐个在 `D.2` 钉住）─────────────────────────────────────────────
THETA1_EXACT = 1.0             # ③ `θ₁` 的**精确**取值
ALIAS_TOL = 1e-12              # ④ 混叠（尺度无关的相对差）
CLOSURE_TOL = 0.01             # ⑤ 两端「闭合」的上界
NONCLOSING_FLOOR = 1.0         # ⑤ 中间段「不闭合」的下界
CLOSED_FORM_TOL = 1e-12        # `B.4` 闭式的**相对**容差
MACHINE_FLOOR = 1e-12          # `B.4` 里 `i=1` 那一档的判定（两边都在机器精度档）
SPREAD_FLOOR = 1.2             # `B.4b` 放大因子的相对跨度下界（实测 `1.71`）
SPREAD_RATIO_LO = 0.6          # `B.4b` 「放大因子 ÷ 样本振幅离散度」的下界（实测 `0.78`）
SPREAD_RATIO_HI = 1.5          # …上界（实测 `1.15`）
ALIAS_MISMATCH_FLOOR = 0.1     # `C.1` 负对照的下界


# ===========================================================================
# 2. 频率表与「圈数」（★ **测量的对象**，不是本程序的正文）
# ===========================================================================
def freqs(d, base):
    """`θ_i = base^{-2(i-1)/d}`，`i = 1..d/2`（返回长度 `d/2` 的数组）。"""
    i = np.arange(1, d // 2 + 1, dtype=np.float64)
    return base ** (-2.0 * (i - 1.0) / d)


def turns(th, span):
    """★ 在长度 `span` 的上下文里，这条频率转了**几圈**：`θ·span / 2π`。

    ★★ **这个量就是 `03` 里 YaRN 的 `r = L/λ`**（本程序与 `03` 之间**共用的公式**）：
      `r = L/λ = L·θ/(2π) = turns(θ, L)` —— `02` 与 `03` 的桥就是它。
    """
    return th * span / TURN_RAD


def pair_contrib(q, k, delta, th, idx):
    """第 `idx` 对（0-based）对 `⟨R_0 q, R_δ k⟩` 的贡献。`q`/`k` 可以带样本维。"""
    q0, q1 = q[..., 2 * idx], q[..., 2 * idx + 1]
    k0, k1 = k[..., 2 * idx], k[..., 2 * idx + 1]
    c, s = np.cos(delta * th[idx]), np.sin(delta * th[idx])
    return q0 * (k0 * c - k1 * s) + q1 * (k0 * s + k1 * c)


# ── ★ 参照实现（**不调用 `pair_contrib`**，`D.11` 的 AST + 文本检查守着这一点）───
def pair_AB(q, k, idx):
    """把第 `idx` 对的贡献写成 `A·cos(δθ) + B·sin(δθ)` 的两个系数（闭式的输入）。"""
    q0, q1 = q[..., 2 * idx], q[..., 2 * idx + 1]
    k0, k1 = k[..., 2 * idx], k[..., 2 * idx + 1]
    return q0 * k0 + q1 * k1, q1 * k0 - q0 * k1


def closure_closed_form(q, k, th, idx):
    """★★ **闭式**：`max_δ |c(δ + 2π) − c(δ)| == 2·|sin(π·θ_i)|·R`，其中 `R = √(A²+B²)`。

    ★ 推导（本域自己的，只用 `c(δ) = R·cos(δθ − φ)`、`φ = atan2(B, A)`）：
      `c(δ+2π) − c(δ) = −2·sin(πθ)·R·sin(δθ − φ + πθ)` ⇒ 逐点 `≤ 2|sin(πθ)|R`，
      而取到等号的 `δ*` 是 `δ* = (φ − πθ + π/2)/θ`（下面 `closing_delta_star` 给的就是它）。
    ★★ **一个式子解释三个频段**：`θ` 为整数 ⇒ 0（精确闭合）；`θ·2π ≪ 1` ⇒ 很小（近似闭合）；
      中间 ⇒ `O(1)`（不闭合）。
    """
    A, B = pair_AB(q, k, idx)
    R = np.sqrt(A * A + B * B)
    return 2.0 * np.abs(np.sin(TURN_RAD / 2.0 * th[idx])) * R


def closing_delta_star(q, k, th, idx):
    """取到闭式等号的那个 `δ*`（解析给出的，不是搜出来的）。"""
    A, B = pair_AB(q, k, idx)
    phi = np.arctan2(B, A)
    half = TURN_RAD / 2.0
    return (phi - half * th[idx] + half / 2.0) / th[idx]


# ===========================================================================
# 3. A 类：判据 ③ —— `θ₁ = 1` 与 `base` 的关系
# ===========================================================================
def part_A():
    print("=" * 74)
    print("A. 判据 ③：`θ_i` 的谱，与 `base` 动得了哪一端")
    print("=" * 74)
    print("  口径：`d = %d`（`d/2 = %d` 条频率）、`base ∈ %s`、种子 `%d`"
          % (D_SPEC, D_SPEC // 2, list(BASE_LADDER), SEED))
    print("  ★ 判据式：`θ_i = base^{-2(i-1)/d}`，`i = 1..d/2`"
          "（★ 下标起点是本域的 `specs.ROPE_FREQ_INDEX_ORIGIN`）")

    # ── ③ θ₁ == 1.0 精确，且与 base 无关 ──────────────────────────────────
    print()
    print("  ── ③ `θ₁ == 1.0`（**精确**）在三档 `base` 上 ──")
    th1 = {}
    for base in BASE_LADDER:
        th = freqs(D_SPEC, base)
        th1[base] = float(th[0])
        print("    base=%-10.0f  θ₁ = %r   （`θ₁` 的周期 = %.6f 个位置）"
              % (base, th1[base], TURN_RAD / th1[base]))
    check("A.1", all(v == THETA1_EXACT for v in th1.values()),
          "③ `θ₁ == %.1f` **精确**成立（三档 `base` 全过，实测逐位 `%r`）—— "
          "因为 `θ₁ = base⁰`，**指数就是 0**，与 `base` 无关"
          % (THETA1_EXACT, sorted(set(th1.values()))))

    # ── θ_{d/2} 随 base 单调下降 ──────────────────────────────────────────
    print()
    print("  ── ★★ `base` 动得了哪一端？──")
    thetas = [freqs(D_SPEC, b) for b in BASE_LADDER]
    low = [float(t[-1]) for t in thetas]
    lam_low = [TURN_RAD / v for v in low]
    for base, v, lm in zip(BASE_LADDER, low, lam_low):
        print("    base=%-10.0f  θ_%-3d = %.6e   最低频的波长 λ = %.3e 个位置"
              % (base, D_SPEC // 2, v, lm))
    strict = all(low[k] > low[k + 1] for k in range(len(low) - 1))
    check("A.2", strict and len(set(th1.values())) == 1,
          "★★ `θ₁` **与 `base` 无关**（三档同一个值 `%r`），而 `θ_{d/2}` **随 `base` 严格下降**"
          "（`%.3e → %.3e → %.3e`）⇒ ★ **提高 `base` 只把低频端拉低**"
          "（「可分辨的总跨度」变长），**动不了最高频那一维** —— "
          "★ 这是一对**两条**断言，不是一句概括"
          % (th1[BASE_LADDER[0]], low[0], low[1], low[2]))

    # ── 谱的单调性 ────────────────────────────────────────────────────────
    th0 = freqs(D_SPEC, BASE_LADDER[0])
    lam0 = TURN_RAD / th0
    mono_th = bool((np.diff(th0) < 0).all())
    mono_lam = bool((np.diff(lam0) > 0).all())
    print()
    print("  ── 谱本身（`base = %.0f`）──" % BASE_LADDER[0])
    print("    %4s %16s %18s %14s" % ("i", "θ_i", "λ_i = 2π/θ_i（位置）", "S=4096 圈数"))
    for i in SPEC_INDEX:
        print("    %4d %16.8f %18.3f %14.4f"
              % (i, th0[i - 1], lam0[i - 1], turns(th0[i - 1], S_LADDER[1])))
    check("A.3", mono_th and mono_lam,
          "★ 谱的单调性：`θ_i` **严格递减**、`λ_i = 2π/θ_i` **严格递增**"
          "（逐差全过，不是抽样）⇒ 这是一条**几何阶梯**：高频密、低频疏")

    # ── 分辨窗口（与计划 D6 同口径）────────────────────────────────────────
    print()
    print("  ── ★ 分辨窗口 `θ_i·S`（与计划 D6 同口径：`i=1` 取 `S=128`、`i=32` 取 `S=4096`）──")
    w1 = float(th0[0] * S_LADDER[0])
    w32 = float(th0[31] * S_LADDER[1])
    print("    i=1  在 S=%d 内：θ₁·S = %.4f rad = %.4f 圈（★ 周期只有 %.4f 个位置）"
          % (S_LADDER[0], w1, turns(th0[0], S_LADDER[0]), TURN_RAD / th0[0]))
    print("    i=32 在 S=%d 内：θ_32·S = %.4f rad = %.4f 圈"
          % (S_LADDER[1], w32, turns(th0[31], S_LADDER[1])))
    check("A.4", w1 > TURN_RAD * 20.0 and w32 < TURN_RAD * 0.1,
          "★ 分辨窗口的两端**是两种不同的失败**：`i=1` 在 `S=%d` 内转了 `%.1f rad`"
          "（`%.1f` 圈 ⇒ **早就混叠**），`i=32` 在 `S=%d` 内只转 `%.4f rad`"
          "（`%.4f` 圈 ⇒ **连一小段都没转完，分辨不出**）—— ★ 实测与 D6 的"
          "「`128 rad`」与「`0.546 rad`」**逐位一致**（同口径、同 `d`、同 `base`）"
          % (S_LADDER[0], w1, turns(th0[0], S_LADDER[0]), S_LADDER[1],
             w32, turns(th0[31], S_LADDER[1])))


# ===========================================================================
# 4. B 类：判据 ④ 与 ⑤（混叠周期、闭合误差谱、闭式）
# ===========================================================================
def part_B(th):
    print()
    print("=" * 74)
    print("B. 判据 ④ 混叠周期 / ⑤ 闭合误差谱 / 闭式")
    print("=" * 74)

    # ── ④ 位置 m 与 m+2π 在最高频分量上不可区分 ──────────────────────────
    print()
    print("  ── ④ 位置 `m` 与 `m+%g` 在**最高频分量**（第 1 对，`θ₁ = 1`）上的分数 ──"
          % TURN_RAD)
    rng = np.random.default_rng(SEED)
    q = rng.standard_normal(D_SPEC)
    k = rng.standard_normal(D_SPEC)
    p_scale = float(np.hypot(q[0], q[1]) * np.hypot(k[0], k[1]))
    print("    尺度（这一对的两个分量）：`‖q_pair‖·‖k_pair‖ = %.6f`" % p_scale)
    worst = 0.0
    print("    %8s %16s %16s %16s %16s"
          % ("m", "s(m)", "s(m+2π)", "差/尺度（**判据**）", "差/|s|（D4 的口径）"))
    for m in M_LADDER:
        s1 = float(pair_contrib(q, k, m, th, 0))
        s2 = float(pair_contrib(q, k, m + TURN_RAD, th, 0))
        rel = abs(s2 - s1) / p_scale
        rel_d4 = abs(s2 - s1) / abs(s1)
        worst = max(worst, rel)
        print("    %8.1f %16.10f %16.10f %16.4e %16.4e" % (m, s1, s2, rel, rel_d4))
    check("B.1", worst < ALIAS_TOL,
          "④ 位置 `m` 与 `m+2π` 在最高频分量上**不可区分**：四个 `m` 上最差 `%.4e` < `%.0e`"
          "（★ 判据用**尺度无关**的相对差；★ 右列是计划 D4 的口径 `差/|s|`，"
          "实测 `2.2e-16 ~ 2.3e-14` 与 D4 记录的 `2.2e-16 / 1.85e-14 / 2.28e-14` **逐位一致**）"
          % (worst, ALIAS_TOL))

    # ── ⑤ 闭合误差谱（计划 §3 表 A ⑤ 的原式）─────────────────────────────
    print()
    print("  ── ⑤ 闭合误差谱：`err_i = max_n |c_i(δ+2π) − c_i(δ)| / mean_n |c_i|` ──")
    print("     口径：`N = %d` 组随机 `(q,k)`、`δ = %g`、`d = %d`（与 probe_80f 同口径）"
          % (N_PAIR, BASE_DELTA, D_SPEC))
    Q = rng.standard_normal((N_PAIR, D_SPEC))
    K = rng.standard_normal((N_PAIR, D_SPEC))
    errs, cfs = {}, {}
    print("    ★ 「闭式」那一列是**逐样本振幅加权后的样本均值** `2·|sin(πθ_i)|·E[R_n]`"
          "（不是 `2·|sin(πθ_i)|` 本身，也不是它的逐样本 max）")
    print("    %4s %16s %16s %18s %9s" % ("i", "θ_i", "err_i", "闭式(样本均值)", "err/闭式"))
    for i in SPEC_INDEX:
        idx = i - 1
        v0 = pair_contrib(Q, K, BASE_DELTA, th, idx)
        v1 = pair_contrib(Q, K, BASE_DELTA + TURN_RAD, th, idx)
        errs[i] = float(np.abs(v1 - v0).max() / np.abs(v0).mean())
        cfs[i] = float(np.mean(closure_closed_form(Q, K, th, idx)))   # 闭式的量级（逐样本取均值）
        print("    %4d %16.8f %16.4f %18.4e %9s"
              % (i, th[idx], errs[i], cfs[i],
                 "-" if i == 1 else "%.2f" % (errs[i] / cfs[i])))
    check("B.2", all(errs[i] < CLOSURE_TOL for i in CLOSING_INDEX)
          and all(errs[i] > NONCLOSING_FLOOR for i in NONCLOSING_INDEX),
          "⑤ **两端闭合、中间不闭合**：`err_%d = %.4f`、`err_%d = %.4f`（都 < `%.2f`），"
          "而 `err_i` 在 `i ∈ %s` 上 = `%s`（逐个 `> %.1f`）⇒ ★ 四个中间点**逐个**断言"
          "（不是「`i = 2..8`」那种范围写法）"
          % (CLOSING_INDEX[0], errs[CLOSING_INDEX[0]], CLOSING_INDEX[1],
             errs[CLOSING_INDEX[1]], CLOSURE_TOL, list(NONCLOSING_INDEX),
             " / ".join("%.2f" % errs[i] for i in NONCLOSING_INDEX), NONCLOSING_FLOOR))
    check("B.3", errs[TRANSITION_INDEX] > CLOSURE_TOL
          and errs[TRANSITION_INDEX] < NONCLOSING_FLOOR,
          "★ **`i = %d` 落在过渡带**（`err = %.4f`，既不 `< %.2f` 也不 `> %.1f`）"
          "⇒ ★ **本程序不断言它** —— 这正是计划原稿把五个读数写成「`i = 2..8`」时"
          "**范围与个数不符**的原因：闭式 `2|sin(πθ_i)|` 是**连续**下降的，"
          "两端的阈值之间**必然**有一段过渡带，它不是「异常」，是**中间态**"
          % (TRANSITION_INDEX, errs[TRANSITION_INDEX], CLOSURE_TOL, NONCLOSING_FLOOR))

    # ── B.4 闭式（恒等）───────────────────────────────────────────────────
    print()
    print("  ── B.4 ★★ **闭式**：`max_δ |c(δ+2π) − c(δ)| == 2·|sin(πθ_i)|·R` ──")
    print("     （`R = √(A²+B²)` 是该对贡献的振幅；等号在 `δ* = (φ − πθ + π/2)/θ` 处取到）")
    worst_cf = 0.0
    machine_ok = True
    for i in SPEC_INDEX:
        idx = i - 1
        ds = float(closing_delta_star(q, k, th, idx))
        lhs = abs(float(pair_contrib(q, k, ds + TURN_RAD, th, idx))
                  - float(pair_contrib(q, k, ds, th, idx)))
        rhs = float(closure_closed_form(q, k, th, idx))
        if rhs > MACHINE_FLOOR:
            worst_cf = max(worst_cf, abs(lhs - rhs) / rhs)
            tag = "%.2e" % (abs(lhs - rhs) / rhs)
        else:
            machine_ok = machine_ok and lhs < MACHINE_FLOOR
            tag = "两边都在机器精度档"
        print("    i=%2d  实测 %.6e   闭式 %.6e   相对差 %s" % (i, lhs, rhs, tag))
    check("B.4", worst_cf < CLOSED_FORM_TOL and machine_ok,
          "★★ **闭式逐位成立**（`i ≥ 2` 最差相对差 `%.2e` < `%.0e`；`i=1` 的两边都在"
          " `%.0e` 档）；⇒ **一个式子解释三个频段**：`θ` 是整数 ⇒ `sin(πθ) = 0` ⇒ **精确闭合**"
          "（本谱里只有 `i = 1`，因为 `θ₁ = 1`）；`θ·2π ≪ 1` ⇒ `≈ 2πθ` **很小** ⇒ 近似闭合；"
          "中间 ⇒ `O(1)` ⇒ 不闭合。★★ **两端的原因确实相反**："
          "`i=1` 是「正好一整圈」（周期 = `2π`，位移恰好是一个周期），"
          "`i=32` 是「连一小段都没转完」（`2|sin(πθ_32)| = %.2e`）"
          % (worst_cf, CLOSED_FORM_TOL, MACHINE_FLOOR, cfs[32]))

    # ── B.4b 量法口径：err_i 与闭式差一个「样本离散因子」───────────────────
    print()
    print("  ── B.4b ★★ **量法口径注记**：那个放大因子到底是什么？──")
    print("    `err_i` 的分子是**样本上的 `max`**、分母是**样本上的 `mean`** ⇒ 它不是相对误差。")
    print("    把它与闭式的样本均值相除，并与**样本振幅离散度** `max_n R_n / mean_n R_n` 比：")
    print("    %4s %14s %18s %14s" % ("i", "err/闭式", "max R/mean R", "两者之比"))
    fac = {}
    amp = {}
    ratios = {}
    for i in SPEC_INDEX[1:]:
        idx = i - 1
        fac[i] = errs[i] / cfs[i]
        A, B = pair_AB(Q, K, idx)
        R = np.sqrt(A * A + B * B)
        amp[i] = float(R.max() / R.mean())
        ratios[i] = fac[i] / amp[i]
        print("    %4d %14.4f %18.4f %14.4f" % (i, fac[i], amp[i], ratios[i]))
    print("    i=1 时两边都在机器精度档 ⇒ 这个比值**没有意义**（分子分母都是舍入噪声）")
    check("B.4b", all(v > 1.0 for v in fac.values())
          and max(fac.values()) / min(fac.values()) > SPREAD_FLOOR
          and all(SPREAD_RATIO_LO < v < SPREAD_RATIO_HI for v in ratios.values()),
          "★★ 放大因子**恒 `> 1`**（`%.2f ~ %.2f`）且**跨 `%.2f` 倍**（阈值 `> %.1f`），"
          "而它与**样本振幅离散度** `max_n R_n / mean_n R_n` 的比值**落在 `[%.2f, %.2f]`**"
          "（阈值 `(%.1f, %.1f)`）⇒ ★★ **它就是那个离散度**（外带一个固定的 `δ0` 相位因子）："
          "`2·|sin(πθ_i)|` 在分子分母里**都出现、被约掉** ⇒ 这个因子**与 `θ` 无关**"
          "（实测：`θ` 从 `i=2` 到 `i=32` 跨 `%.0f` 倍，而这个因子只动 `%.2f` 倍）"
          "⇒ ★ 计划 §0.1 第一类错误（**判据的统计量不稳**）的又一现场 ⇒ "
          "本程序**两个口径都写出来**：`B.2` 按计划的原式断言、`B.4` 用闭式"
          % (min(fac.values()), max(fac.values()),
             max(fac.values()) / min(fac.values()), SPREAD_FLOOR,
             min(ratios.values()), max(ratios.values()),
             SPREAD_RATIO_LO, SPREAD_RATIO_HI,
             float(th[2] / th[31]), max(fac.values()) / min(fac.values())))


# ===========================================================================
# 5. C 类：边界与反例
# ===========================================================================
def part_C(th):
    print()
    print("=" * 74)
    print("C. 边界与反例")
    print("=" * 74)
    rng = np.random.default_rng(SEED)
    q = rng.standard_normal(D_SPEC)
    k = rng.standard_normal(D_SPEC)
    p_scale = float(np.hypot(q[0], q[1]) * np.hypot(k[0], k[1]))

    # ── C.1 判据④ 不是恒真断言：换半个周期就不闭合 ─────────────────────────
    print()
    print("  ── C.1 ★★ **负对照**：把位移从一整圈换成**半圈**（`π`）──")
    w_full, w_half = 0.0, 0.0
    for m in M_LADDER:
        s1 = float(pair_contrib(q, k, m, th, 0))
        w_full = max(w_full, abs(float(pair_contrib(q, k, m + TURN_RAD, th, 0)) - s1) / p_scale)
        w_half = max(w_half, abs(float(pair_contrib(q, k, m + HALF_TURN_RAD, th, 0)) - s1)
                     / p_scale)
    print("    位移 = 一整圈（`2π`）：最差 %.4e" % w_full)
    print("    位移 = 半圈（`π`）  ：最差 %.4e" % w_half)
    check("C.1", w_full < ALIAS_TOL and w_half > ALIAS_MISMATCH_FLOOR,
          "★★ 判据④ **不是恒真断言**：位移换成半圈后同一个量从 `%.2e` 掉到 `%.4f` > `%.1f` ⇒ "
          "「不可区分」**专指`2π`这一位移**（因为 `θ₁ = 1` 使周期**精确**等于 `2π`），"
          "不是「高频分量对位移不敏感」" % (w_full, w_half, ALIAS_MISMATCH_FLOOR))

    # ── C.2 两端互斥：可证的不等式 ────────────────────────────────────────
    print()
    print("  ── C.2 ★★ **两端互斥**：两个条件是**空交集**（这是不等式，不是测量）──")
    print("    「相邻位置可分」：`θ_i ≥ %.2f` rad（每走一个位置相位至少变这么多）" % ADJ_PHASE_THRESHOLD)
    print("    「整窗不混叠」  ：`turns_i(L) ≤ %.1f`（整段上下文里最多转这么多圈）" % TURNS_LIMIT)
    total_intersection = 0
    qwen_counts = None
    llama_ok = True
    for name, base, L in (("Llama-3-8B", specs.LLAMA3_8B_ROPE_THETA,
                           specs.LLAMA3_8B_MAX_POSITION_EMBEDDINGS),
                          ("Qwen3-30B-A3B", specs.QWEN3_30B_A3B_ROPE_THETA,
                           specs.QWEN3_30B_A3B_MAX_POSITION_EMBEDDINGS)):
        thm = freqs(D_MODEL, base)
        print("    %-14s base=%-9.0f L=%-6d （`d = %d`）阈值 → （可分 / 不混叠 / **交集**）:"
              % (name, base, L, D_MODEL))
        for thr in (ADJ_PHASE_THRESHOLD, ADJ_PHASE_THRESHOLD_LO, ADJ_PHASE_THRESHOLD_LOLO):
            resolvable = thm >= thr
            aliasfree = turns(thm, float(L)) <= TURNS_LIMIT
            ni = int((resolvable & aliasfree).sum())
            total_intersection += ni
            print("         θ ≥ %-6.2f  %3d / %3d / **%d**"
                  % (thr, int(resolvable.sum()), int(aliasfree.sum()), ni))
            if name.startswith("Qwen3") and thr == ADJ_PHASE_THRESHOLD:
                qwen_counts = (int(resolvable.sum()), int(aliasfree.sum()))
            if name.startswith("Llama") and thr == ADJ_PHASE_THRESHOLD:
                llama_ok = (int(resolvable.sum()), int(aliasfree.sum())) == (1, 29)
    check("C.2", total_intersection == 0 and qwen_counts == (1, 23) and llama_ok
          and (TURN_RAD / float(specs.QWEN3_30B_A3B_MAX_POSITION_EMBEDDINGS)) < 1.0
          and (TURN_RAD / float(specs.LLAMA3_8B_MAX_POSITION_EMBEDDINGS)) < 1.0,
          "★★ 两个集合**交集恒为 0**（六个格子全过），而且这**不是**凑出来的："
          "两个条件是 `θ ≥ %.2f`（相邻位置可分）与 `θ ≤ 2π/L`（整窗不混叠），"
          "而任何 `L > 2π` 都有 `2π/L < 1` ⇒ 两个区间 `[1, 2π/L]` 与 `[2π/L, 1]` "
          "**方向相反** ⇒ **空集**。★ 三个阈值上交集都是 `0` ⇒ 结论**不挑阈值**。"
          "★ 在 `d = %d`（真实模型的 rotary 维数）上：Qwen3（`base = 1e6`、`L = %d`）"
          "不混叠 **%d** 维、可分 **%d** 维（`i = 1`）；Llama 不混叠 **29** 维、可分 **1** 维"
          % (ADJ_PHASE_THRESHOLD, D_MODEL,
             specs.QWEN3_30B_A3B_MAX_POSITION_EMBEDDINGS,
             qwen_counts[1], qwen_counts[0]))
    print("    ★ `2π/L`：Llama-3-8B `%.3e`、Qwen3 `%.3e`（都 ≪ 1）"
          % (TURN_RAD / float(specs.LLAMA3_8B_MAX_POSITION_EMBEDDINGS),
             TURN_RAD / float(specs.QWEN3_30B_A3B_MAX_POSITION_EMBEDDINGS)))

    # ── C.3 真实模型上的计数 + 指路 ───────────────────────────────────────
    print()
    print("  ── C.3 两个真实模型上「`λ_i ≥ L`」的那些维（1-based）──")
    span = {}
    for nm, b, L2 in (("Llama-3-8B", specs.LLAMA3_8B_ROPE_THETA,
                       specs.LLAMA3_8B_MAX_POSITION_EMBEDDINGS),
                      ("Qwen3-30B-A3B", specs.QWEN3_30B_A3B_ROPE_THETA,
                       specs.QWEN3_30B_A3B_MAX_POSITION_EMBEDDINGS)):
        thm = freqs(D_MODEL, b)
        lam = TURN_RAD / thm
        idx = np.flatnonzero(lam >= float(L2))
        span[nm] = (int(idx[0]) + 1, int(idx.size))
        print("    %-14s `λ_i ≥ L`：i = %d..%d（共 **%d** 维，占 %d 维的 %.1f%%）"
              % (nm, int(idx[0]) + 1, int(idx[-1]) + 1, int(idx.size), D_MODEL // 2,
                 100.0 * idx.size / (D_MODEL // 2)))
    check("C.3", span["Llama-3-8B"] == (36, 29) and span["Qwen3-30B-A3B"] == (42, 23),
          "★ 起维与维数**逐位钉住**（`d = %d` 上：Llama 从 `i = %d` 起、`%d` 维；"
          "Qwen3 从 `i = %d` 起、`%d` 维）—— ★ 这两行是 `02` 交给 `03` 的**输入**："
          "它们正是 YaRN 的 `L/λ_i` 判据里「`λ_i ≥ L` ⇒ 只插值」那一类维（E5）。"
          "★ 本程序**到此为止**：外推方法的定义与对比在 `03`"
          "（计划 §3 的分工写死：`02` 给定量框架）"
          % (D_MODEL, span["Llama-3-8B"][0], span["Llama-3-8B"][1],
             span["Qwen3-30B-A3B"][0], span["Qwen3-30B-A3B"][1]))


# ===========================================================================
# 6. D 类：口径/阈值/specs 的钉住（**刻意在 A/B/C 之前**）与失败标记自检
# ===========================================================================
def _callee_names(func):
    """★ 用 **AST** 列出 `func` 真正**调用**的函数名（不含注释/文档串里的提及）。"""
    import textwrap
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    return {n.func.id for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}


def _body_text(func):
    """`func` 的源码**去掉 `def` 行**（否则函数名本身会把「调用」这个词顶进来）。"""
    return "\n".join(l for l in inspect.getsource(func).splitlines()
                     if not l.lstrip().startswith("def "))


def _mark_assignments_in(path):
    """用 **AST** 从本文件里数出模块级的 `ASSERT_FAIL_MARK = <字符串字面量>`。"""
    tree = ast.parse(open(path, encoding="utf-8").read(), path)
    out = []
    for n in tree.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 \
                and isinstance(n.targets[0], ast.Name) \
                and n.targets[0].id == "ASSERT_FAIL_MARK":
            out.append(n.value)
    return out


def pin_part():
    """★★ **口径 / 阈值 / specs 的钉住 —— 刻意放在 A/B/C 之前**（Chunk 2 确立的硬要求）。"""
    print("=" * 74)
    print("D. 口径/阈值/specs 的钉住（★ 放在 A/B/C **之前**：先钉住前提，再看数）")
    print("=" * 74)

    measured = {"D_SPEC": D_SPEC, "BASE_LADDER": BASE_LADDER, "N_PAIR": N_PAIR,
                "BASE_DELTA": BASE_DELTA, "TURN_RAD": TURN_RAD,
                "HALF_TURN_RAD": HALF_TURN_RAD, "M_LADDER": M_LADDER,
                "S_LADDER": S_LADDER, "SPEC_INDEX": SPEC_INDEX,
                "CLOSING_INDEX": CLOSING_INDEX, "NONCLOSING_INDEX": NONCLOSING_INDEX,
                "TRANSITION_INDEX": TRANSITION_INDEX,
                "ADJ_PHASE_THRESHOLD": ADJ_PHASE_THRESHOLD, "TURNS_LIMIT": TURNS_LIMIT,
                "SEED": SEED, "D_MODEL": D_MODEL}
    expect = {"D_SPEC": 64, "BASE_LADDER": (10000.0, 500000.0, 1000000.0), "N_PAIR": 200,
              "BASE_DELTA": 3.0, "TURN_RAD": 6.283185307179586,
              "HALF_TURN_RAD": 3.141592653589793, "M_LADDER": (10.0, 50.0, 200.0, 1000.0),
              "S_LADDER": (128, 4096), "SPEC_INDEX": (1, 2, 3, 4, 8, 16, 32),
              "CLOSING_INDEX": (1, 32), "NONCLOSING_INDEX": (2, 3, 4, 8),
              "TRANSITION_INDEX": 16,
              "ADJ_PHASE_THRESHOLD": 1.0, "TURNS_LIMIT": 1.0,
              "SEED": 20260922, "D_MODEL": 128}
    diff = {k: (expect[k], v) for k, v in measured.items() if v != expect[k]}
    check("D.1", not diff,
          "口径常量 16 个逐位一致（★ 改 `δ`/样本数/维度，本程序的**不变量**断言照样通过 ⇒ "
          "口径只能靠这一条守住；★ **`TURN_RAD`/`HALF_TURN_RAD` 写成字面量**而不是 `2*np.pi` "
          "—— 后者在 AST 里是 `BinOp`、**根本没有叶子进变异分母**）%s"
          % ("" if not diff else " ← 差异 %s" % diff))

    th = {"THETA1_EXACT": THETA1_EXACT, "ALIAS_TOL": ALIAS_TOL,
          "CLOSURE_TOL": CLOSURE_TOL, "NONCLOSING_FLOOR": NONCLOSING_FLOOR,
          "CLOSED_FORM_TOL": CLOSED_FORM_TOL, "MACHINE_FLOOR": MACHINE_FLOOR,
          "SPREAD_FLOOR": SPREAD_FLOOR, "ALIAS_MISMATCH_FLOOR": ALIAS_MISMATCH_FLOOR,
          "ADJ_PHASE_THRESHOLD_LO": ADJ_PHASE_THRESHOLD_LO,
          "ADJ_PHASE_THRESHOLD_LOLO": ADJ_PHASE_THRESHOLD_LOLO,
          "SPREAD_RATIO_LO": SPREAD_RATIO_LO, "SPREAD_RATIO_HI": SPREAD_RATIO_HI}
    th_expect = {"THETA1_EXACT": 1.0, "ALIAS_TOL": 1e-12, "CLOSURE_TOL": 0.01,
                 "NONCLOSING_FLOOR": 1.0, "CLOSED_FORM_TOL": 1e-12,
                 "MACHINE_FLOOR": 1e-12, "SPREAD_FLOOR": 1.2,
                 "ALIAS_MISMATCH_FLOOR": 0.1,
                 "ADJ_PHASE_THRESHOLD_LO": 0.1, "ADJ_PHASE_THRESHOLD_LOLO": 0.01,
                 "SPREAD_RATIO_LO": 0.6, "SPREAD_RATIO_HI": 1.5}
    tdiff = {k: (th_expect[k], v) for k, v in th.items() if v != th_expect[k]}
    check("D.2", not tdiff,
          "阈值 12 个逐位一致（★ 把 `CLOSURE_TOL` 放宽到 `> NONCLOSING_FLOOR` ⇒ ⑤ 的两半"
          "**同时**退化成恒真；把 `THETA1_EXACT` 改成 `1.000001` ⇒ ③ 的「精确」当场失效）%s"
          % ("" if not tdiff else " ← 差异 %s" % tdiff))

    check("D.3", specs.LLAMA3_8B_ROPE_THETA in BASE_LADDER
          and specs.QWEN3_30B_A3B_ROPE_THETA in BASE_LADDER
          and specs.LLAMA3_8B_MAX_POSITION_EMBEDDINGS == 8192
          and specs.QWEN3_30B_A3B_MAX_POSITION_EMBEDDINGS == 40960
          and D_MODEL == specs.ROTARY_DIM_MODEL == specs.QWEN3_30B_A3B_HEAD_DIM
          == specs.LLAMA3_8B_HEAD_DIM_DERIVED and D_MODEL != D_SPEC,
          "★ 判据 ③ 的 `base` 阶梯里**后两档就是两个真实模型的 `rope_theta`**"
          "（`%.0f` / `%.0f`），窗口是 `%d` / `%d`"
          "（**经 `hf-mirror.com` 镜像取得，未与官方 repo 比对 sha256** —— "
          "这条限定语必须与数字一起引用）；★ `C.2`/`C.3` 的 rotary 维数 `D_MODEL = %d` "
          "**同时**钉在两条 `specs` 常量上（Qwen3 是 **原生字段**、Llama 是 **[E] 推导** —— "
          "★ **来源等级不同、数值相同**，引用时必须分开标注）；★ `D_MODEL != D_SPEC` "
          "是**故意的**（一个是真实模型的维数、一个是探测脚本的口径）"
          % (specs.LLAMA3_8B_ROPE_THETA, specs.QWEN3_30B_A3B_ROPE_THETA,
             specs.LLAMA3_8B_MAX_POSITION_EMBEDDINGS,
             specs.QWEN3_30B_A3B_MAX_POSITION_EMBEDDINGS, D_MODEL))
    check("D.4", specs.ROPE_LONGTERM_DECAY_CLAIM ==
          "One can prove that this setting provides a long-term decay property"
          and specs.ROPE_FREQ_INDEX_ORIGIN == 1,
          "★ 论文逐字逐位钉住：§3.3 关于谱的**唯一**一条主张（措辞是 `One can prove`，"
          "**不是**实验结果）；以及本域固定的下标起点 `%d`"
          % specs.ROPE_FREQ_INDEX_ORIGIN)
    check("D.5", specs.UNSOURCED == (
        "ntk_aware_primary_source", "rope_extrapolation_weakness_attribution",
        "long_context_quality_metric", "extrapolation_quality_ranking",
        "rope_in_real_attention_kernels"),
          "★ `specs.UNSOURCED` 是**待办清单**（%d 项），**逐位（含顺序）钉住**"
          "（★ R17 的落法：不给它开后门，而是**给它一个消费者**）" % len(specs.UNSOURCED))


def mark_part():
    print()
    print("=" * 74)
    print("D（续）. 失败标记自检 与 参照实现的独立性")
    print("=" * 74)
    print("  本程序的失败标记 ASSERT_FAIL_MARK = %r" % ASSERT_FAIL_MARK)

    hits = _mark_assignments_in(os.path.abspath(__file__))
    uniq = {getattr(v, "value", None) for v in hits}
    check("D.6", len(hits) == 1 and len(uniq) == 1
          and uniq == {ASSERT_FAIL_MARK} and "FAIL" in ASSERT_FAIL_MARK,
          "★ 标记自检：文件里模块级赋值恰好 1 处、取值唯一、== %r、且含 `FAIL`（AST 读出）"
          % ASSERT_FAIL_MARK)
    check("D.7", ASSERT_FAIL_MARK == "$$ FAIL: positional-encoding"
          and "FAIL" in ASSERT_FAIL_MARK and MARKS_USED <= {ASSERT_FAIL_MARK},
          "★ 标记自检（**字面量比较**，R18）：**等于**本主题被分配的字面量、含 `FAIL`，"
          "且打印失败行时用的标记 ⊆ {%r}（本次实际用到 %d 个）"
          % (ASSERT_FAIL_MARK, len(MARKS_USED)))
    check("D.8", OTHER_THEME_MARKS == ("!! FAIL: transformer", "@@ FAIL: attention-variants",
                                       "## FAIL: moe", "~~ FAIL: normalization"),
          "★ 「不跨主题照抄」这条判据的**对照物**也逐位钉住")

    measured_fns = {pair_contrib}
    m_names = {f.__name__ for f in measured_fns}
    ref_fns = {closure_closed_form, closing_delta_star, pair_AB}
    shared = sorted(set().union(*[_callee_names(f) for f in ref_fns]) & m_names)
    text_ref = "\n".join(_body_text(f) for f in sorted(ref_fns, key=lambda g: g.__name__))
    textual = "pair_contrib(" not in text_ref and "turns(" not in text_ref
    check("D.9", not shared and textual,
          "★ **参照实现的独立性**：闭式那三个函数的调用表与测量函数 `pair_contrib`"
          "**交集 = %s**；文本级回退 = %s ⇒ `B.4` 那一列是**另一条路**算出来的"
          "（★ 闭式的输入只有 `A`/`B`/`θ`/`2π`，**没有**任何一次「先把贡献算出来」）"
          % (shared, textual))


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
    print("02_spectrum_aliasing.py —— θ_i 的谱、混叠周期、分辨窗口")
    print("=" * 74)
    print("  ★ 真值标注：本程序的数字**全部是推导值/本机实测值**"
          "（本机 `python3 %s` + `numpy %s`）。"
          % (sys.version.split()[0], np.__version__))
    print("    本机 macOS arm64 **无 GPU**：这些数不支撑任何关于真实模型或 GPU 的结论。")
    print("  ★ 失败标记 ASSERT_FAIL_MARK = %r（本主题自定义；不跨主题照抄）"
          % ASSERT_FAIL_MARK)
    print("  ★★ 本程序给的是**定量框架**（计划 §3 的分工写死）：谱 / 混叠 / 分辨窗口 / `turns`。")
    print("     ★ **不讲任何外推方法** —— 那是 `03`。")
    print("  ★ 本程序最有用的一条是 `B.4` 的**闭式** `2·|sin(π·θ_i)|`：")
    print("     它一个式子解释了「两端闭合、中间不闭合」，且**两端的原因相反**。")
    pin_part()
    part_A()
    part_B(freqs(D_SPEC, BASE_LADDER[0]))
    part_C(freqs(D_SPEC, BASE_LADDER[0]))
    mark_part()
    return summary()


if __name__ == "__main__":
    sys.exit(main())

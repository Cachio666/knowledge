#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""01_rmsnorm_layernorm.py —— ★★ 不变性的**精确形态**：**哪个**不变、哪个**只在 `eps → 0` 时**不变。

配套文档：`01-rmsnorm-vs-layernorm.md`。
★ 三类断言（A 算法正确性 / B 代价模型 / C 边界与反例）**全部在本程序内部**，
失败时以**非零退出码**结束；**没有 `verify.py`**。

===============================================================================
★★ 本程序要回答的那一个问题
===============================================================================

两个算子常被一句「RMSNorm 就是去掉均值的 LayerNorm」带过。
**本程序不讨论这句话对不对**（那是一句空话），而是把它换成三个**可判定**的问题：

  1. **哪一个不变性真的丢了？** —— 实测：丢的是**平移**不变性
     （LayerNorm `< 1e-12` vs RMSNorm `> 1.0`），而**缩放**不变性**两者都有**
     （`eps = 0` 时两者都 `< 1e-14`）。⇒ ★ **缩放不是它们的分界线**（见 `C.3`）。
  2. **`eps` 在这件事里扮什么角色？** —— 实测：它是**缩放**不变性的破坏者
     （偏差与 `eps` **同阶**：`1e-6 → ~2e-7`、`1e-3 → ~2e-4`；两个真实模型的 `1e-5`/`1e-6`
     给出的偏差**比值恰是 10**）。
     但它**不是平移那一半的原因**：LayerNorm 的平移偏差与 `eps` **精确无关**
     （行方差对平移不变），而 RMSNorm 的平移偏差**三档都 `> 1.0`**、`eps` 只带来 `< 1%` 的微扰。
     ⇒ ★ 「`eps` 是破坏者」这句话**只对缩放那一半成立**（计划 §0.7-D 的 **D7** 的精确形态）。
  3. **「相等」的边界在哪？** —— 论文逐字：`When the mean of summed inputs is zero, RMSNorm is
     exactly equal to LayerNorm.` 本程序把它做成**充分必要**形态：均值**恰好**为 0 时两者**逐位相等**
     （实测 `0.0`），并给出**闭式尺度** `cos(LN(x), RMS(x)) = σ/√ms`（`μ = 0 ⟺ cos = 1`）。

===============================================================================
★★ 本程序的数字与它们的来源等级（三类标签必须分开）
===============================================================================

  · **本机实测值 / 推导值**：`python3 3.13.5` + `numpy 2.4.4`，macOS arm64，**无 GPU**。
    ⇒ **不支撑任何关于真实模型或 GPU 的结论。**
  · **外部规格值**：两个真实模型的 `rms_norm_eps`（`specs.py`，**经镜像取得**）。
  · **论文逐字**：`specs.py` 的三条 [C] 常量（arXiv 1910.07467，`NeurIPS 2019` 已确认同行评审）。
  ⇒ ★ **不得把推导写成「实测」**（`CONTRIBUTING:63`）。

===============================================================================
★★ 可证伪断言（把下列任一处改错，本程序必须非零退出）
===============================================================================

  · `A.1`  ① `eps = 0` ⇒ **两个算子都**满足 `f(αx) == f(x)`（`max < 1e-14`，α 走三档）；
  · `A.2`  ② `eps` 三档（`0 / 1e-6 / 1e-3`）的偏差**严格递增**，且 `1e-3` 档 `> 1e-5`（**两个算子各一条**）；
  · `A.3`  ③ **平移**：LayerNorm `max|f(x+b) − f(x)| < 1e-12`；
  · `A.4`  ③ **平移**：RMSNorm 同一个量 **`> 1.0`** —— 与 `A.3` 构成分界（两条都是**不等式**，不是约等号）；
  · `A.5`  ★ 平移偏差对 LayerNorm **与 `eps` 精确无关**（三档极差 `< 1e-14`）；
  · `A.6`  ★ RMSNorm 的平移偏差**三档都 `> 1.0`**，且 `eps` 对它的影响 `< 1%`（⇒ 原因不在 `eps`）；
  · `B.1`  ★ **闭式对照**：两个算子的缩放偏差**逐行等于**一个只依赖该行统计量的闭式
           （以 `|f|` 为单位的绝对差 `< 1e-12`）—— 量级判据的证伪能力弱，
           所以这里配的是**可手算的闭式**，不是「差不多大」；
  · `B.1b` ★★ 同一组比较改用**相对**误差时最差只有 `~1e-8`（阈值 `1e-6`）——
           那不是闭式不准，是**「偏差的相对误差」这个量法口径本身坏了**（两个相近大数之差）；
  · `B.2`  ★ **闭式**：`cos(LN(x), RMS(x)) == σ/√ms`（`< 1e-14`，**三个 `eps` 档都成立**）；
  · `B.2b` ★ `eps = 0` 时两个算子的输出**范数都恰好是 `√n`**（相对误差 `< 1e-14`）
           —— 这是「夹角」这句话的前提，也是 `B.3` 那条恒等式的另一半依据；
  · `B.3`  ★★ 均值为 0（**构造**成对 `±v`）⇒ `max|RMS(x) − LN(x)| **恰好** == 0.0`（**恒等式**，见下）
           **且**对照组（均值非 0）同一个量 `> 1e-3` —— 没有对照组，这条可能只是「所有行都这样」；
  · `B.4`  ★ 结构计数：LayerNorm 要 **2** 个归约量、RMSNorm 要 **1** 个（用返回值元组长度钉住）；
  · `B.5`  ★ 真实模型锚点：两个模型的 `rms_norm_eps`（`1e-5` / `1e-6`）都落在「偏差 ∈ `(1e-9, 1e-5)`」这一档，
           且**严格有序**、比值 `≈ 10`（与 `eps` 同阶）—— ⇒ 真实模型工作在**一个被有意轻微破坏**的缩放不变性上；
  · `C.1`  ★★ **反例（P17 挡板）**：`eps = 0` 时 RMSNorm 的平移偏差**仍然 `> 1.0`**
           ⇒ 「平移偏差是 `eps` 造成的」这个说法**当场被否**；
  · `C.2`  ★ **反例**：偏差在 **α 方向不是单调的**（`α → 0⁺` 时 `eps` 接管、`f(αx) → 0`，偏差趋于 `|f(x)|`）
           ⇒ 「偏差随 α 单调」这个直觉是错的；
  · `C.3`  ★ **分界澄清**：`eps = 0` 时**两者都**缩放不变 ⇒ 引用时**不得**把缩放当作两者的差别。

★★ **`B.3` 那条「恰好 == 0.0」的口径（必须与断言一起读）**：
   `00-meta/glossary.md:240` 逐字说「`maxdiff == 0.0` 有**两种相反含义**」——
   一种是「测出来的精确」，一种是「**两边是同一份代码，构造出来的**」。
   本条属于**第三种**：两边是**两条不同的实现**（`B.4` 的 AST 检查会证明它们互不调用），
   但**执行的是同一批标量运算**（`μ ≡ 0.0` 使 `x − μ` 逐位等于 `x`，
   于是 `mean((x−μ)²)` 与 `mean(x²)` 是同一个浮点表达式）⇒ **它是恒等式，不是机器精度比较**。
   ⇒ 所以本程序**同时**给出：`B.2` 的**闭式**（`cos = σ/√ms`，`μ=0` 时才是 1）
     与 `A.3/A.4` 的**不等式**（那两条才是真正「测出来的」量）。

===============================================================================
★★ 本程序刻意不做的事（计划 §2 的边界表，越界即作废）
===============================================================================

  * ★★ **不写公式、不推公式**：LayerNorm / RMSNorm 的**公式定义**、Welford 递推、
    `Σx² − nμ²` 的灾难、**`eps` 位置的数值差异**，正文在
    `40-kernels/softmax-normalization/03-normalization-fusion.md:55-71`（**正文在彼，此处回链**）。
    程序里出现的两个算子实现是**测量的对象**（没有它就没法测不变性），
    它们保持**最朴素的形态**，且**不带任何推导注释**。
    ★ 本程序**只谈 `eps` 的「量级」**（它破坏缩放不变性的量级，D7），**不谈它的「位置」**。
  * **不算**归约的访存账、不写融合形态（正文在 `40-kernels/softmax-normalization/03:213-263`）。
  * **不下训练侧结论**（哪个算子训练出来的模型更好）：本域无 GPU、不训练模型
    ⇒ 论文的实验结论**只能转述并标明是论文的**（见 `specs.UNSOURCED`）。

用法：
    python3 01_rmsnorm_layernorm.py
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
ASSERT_FAIL_MARK = "~~ FAIL: normalization"

OTHER_THEME_MARKS = ("!! FAIL: transformer", "@@ FAIL: attention-variants",
                     "## FAIL: moe", "$$ FAIL: positional-encoding")

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
# 1. 口径常量（★ 全部在 `D.1`/`D.2` 被逐位钉住）
# ===========================================================================
# ★★ 为什么口径常量必须被**逐位钉住**（而不是「用上就行」）：本程序的断言几乎全是
#    「不变量」形态 —— 把行数从 `64` 改成 `65`、把种子换掉、把 `x` 的均值从 `5.0` 改成 `6.0`，
#    **所有断言照样通过**。⇒ 口径本身就是**测量条件**，只能靠 `D.1` 的**字面量比较**守住。
#    （这正是本域 `mutate.py` 存在的理由：本域的断言极易写成无条件成立的伪断言。）
#
# ★ 本段的口径**刻意照抄探测脚本** `probes/probe_80b.py` 的「复验 Q4」段
#   （计划 §0.7-D 的 **D7 / D8** 的唯一口径记录）：`X = rng.standard_normal((64, 512)) * 3.0 + 5.0`，
#   `eps ∈ {0, 1e-6, 1e-3}` × `α ∈ {2, 50, 1000}`，平移档 `b = 100`。
N_ROWS = 64              # 行数
N_COLS = 512             # 行宽（= 一根残差流的宽度）
X_MEAN = 5.0             # ★ 行均值**刻意不为 0**：`A.3/A.4` 的平移对照才有意义
X_STD = 3.0
SEED = 20260922          # 数据种子
EPS_LADDER = (0.0, 1e-6, 1e-3)        # ★ 判据 ② 的三档（`1e-6` 恰好是 Qwen3 的真实取值）
ALPHA_LADDER = (2.0, 50.0, 1000.0)    # ★ 判据 ① 的 α 三档（偏差在 α 上**不单调**，见 `C.2`）
SHIFT_B = 100.0          # 判据 ③ 的平移量（探测脚本的口径）
ALPHA_DECAY_LADDER = (1e-3, 1e-2, 0.1)  # `C.2`：α → 0 的一侧（eps 接管的端点）
ROW_SCALE = 7.0          # `C.3` 的对照缩放（一个平凡的非单位尺度）

# ── 阈值（★ 逐个在 `D.2` 钉住）─────────────────────────────────────────────
INVARIANCE_TOL = 1e-14   # ① 缩放不变性（**机器精度档**）
SHIFT_LN_TOL = 1e-12     # ③ LayerNorm 的平移不变性
SHIFT_RMS_FLOOR = 1.0    # ③ RMSNorm 的平移偏差下界（★ **不等式**，不是约等号）
EPS_DEV_FLOOR = 1e-5     # ② `1e-3` 档的偏差下界
CLOSED_FORM_SCALED_TOL = 1e-12  # `B.1` 闭式对照：以 `|f|` 为单位的绝对差（★ 比「同量级」严得多）
CLOSED_FORM_REL_TOL = 1e-6       # `B.1b` 同一个比较的**相对**误差上界（★ 这个口径**比机器精度差**
                                 #   7~8 个数量级 —— 原因见 `B.1b` 的打印，那是**量法口径**的问题）
COS_TOL = 1e-14          # `B.2` 闭式的容差
IDENTITY_EXACT = 0.0     # `B.3` 恒等式的容差（**恰好 0**，见模块头）
CONTROL_FLOOR = 1e-3     # `B.3` 对照组的**下界**（没有它，那条恒等式可能只是「所有行都这样」）
STAT_COUNT_LN = 2        # `B.4` LayerNorm 的统计量个数
STAT_COUNT_RMS = 1       # `B.4` RMSNorm 的统计量个数
REAL_EPS_BAND = (1e-9, 1e-5)  # `B.5` 真实模型的 eps 使偏差落进的那一档


# ===========================================================================
# 2. 两个算子（★ **测量的对象**，不是本程序的正文）
# ===========================================================================
# ★★ 公式的正文在 `40-kernels/softmax-normalization/03-normalization-fusion.md:55-71`
#    （**正文在彼，此处回链**）。这里的两段实现刻意保持**最朴素的形态**，
#    只做测量需要的那一件事；本程序**不注释它们的数学**，也不讨论 `eps` 放在哪。
def rmsnorm(x, eps):
    """按行的均方根归一化（**测量对象**）。"""
    return x / np.sqrt((x ** 2).mean(axis=-1, keepdims=True) + eps)


def layernorm(x, eps):
    """按行的均值/方差归一化（**测量对象**）。"""
    mu = x.mean(axis=-1, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)


def stats_layernorm(x):
    """★ `B.4` 的载体：LayerNorm 需要的**归约量**，返回值长度 = 统计量个数。"""
    mu = x.mean(axis=-1, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
    return (mu, var)


def stats_rmsnorm(x):
    """★ `B.4` 的载体：RMSNorm 需要的**归约量**（只有一个）。"""
    return (x ** 2).mean(axis=-1, keepdims=True),


# ── 闭式参照（★ **不调用上面任何一个函数**，`D.8` 的 AST 检查守着这一点）─────────
def closed_form_scale_dev_rmsnorm(ms, x_absmax, alpha, eps):
    """`f(αx)` 与 `f(x)` 的逐行最大绝对差，**只用该行的均方值**（闭式，可手算）。

    ★ 它**不调用** `rmsnorm` —— 这就是「参考实现的独立性」（域 README §「代码约定」）。
    """
    return np.abs(alpha / np.sqrt(alpha * alpha * ms + eps)
                  - 1.0 / np.sqrt(ms + eps)) * x_absmax


def closed_form_scale_dev_layernorm(sig2, xc_absmax, alpha, eps):
    """LayerNorm 的同一个量（闭式：`eps` 被 `α²` 除掉，这是它**与 RMSNorm 不同**的地方）。"""
    return np.abs(1.0 / np.sqrt(sig2 + eps / (alpha * alpha))
                  - 1.0 / np.sqrt(sig2 + eps)) * xc_absmax


def closed_form_cos_layernorm_rmsnorm(sig2, ms):
    """`cos(LN(x), RMS(x))` 的闭式 = `σ/√ms`（**与 `eps` 无关**）。"""
    return np.sqrt(sig2) / np.sqrt(ms)


# ===========================================================================
# 3. A 类：判据表 A 的数值判据（① ② ③ + 一条本体的）
# ===========================================================================
def part_A(x):
    print("=" * 74)
    print("A. 不变性的精确形态（① 缩放 / ② eps / ③ 平移）")
    print("=" * 74)
    print("  口径：`X = rng.standard_normal((%d, %d)) * %g + %g`，种子 %d；"
          % (N_ROWS, N_COLS, X_STD, X_MEAN, SEED))
    print("        `eps ∈ %s`、`α ∈ %s`、平移 `b = %g`（与 `probes/probe_80b.py` 的 Q4 同口径）"
          % (list(EPS_LADDER), list(ALPHA_LADDER), SHIFT_B))

    # ── ① eps = 0 ⇒ 两个算子都 f(αx) == f(x) ────────────────────────────────
    print()
    print("  ── ① `eps = 0` 时 `f(αx) == f(x)`（**两个算子都成立**）──")
    worst = {"RMS": 0.0, "LN": 0.0}
    for fn, key in ((rmsnorm, "RMS"), (layernorm, "LN")):
        for alpha in ALPHA_LADDER:
            d = float(np.abs(fn(alpha * x, 0.0) - fn(x, 0.0)).max())
            worst[key] = max(worst[key], d)
            print("    %-4s α=%-7.1f  max|f(αx) − f(x)| = %.4e" % (key, alpha, d))
    check("A.1", worst["RMS"] < INVARIANCE_TOL and worst["LN"] < INVARIANCE_TOL,
          "① `eps = 0` ⇒ **两个算子都**缩放不变：RMS `%.3e` / LN `%.3e`，"
          "两者都 < `%.1e`（★ 实测与计划 D7 的 `0.000e+00 / 8.882e-16` 同档）"
          % (worst["RMS"], worst["LN"], INVARIANCE_TOL))

    # ── ② 偏差随 eps 严格递增 ──────────────────────────────────────────────
    print()
    print("  ── ② `eps > 0` 时偏差随 `eps` 增长（三档**严格递增**）──")
    dev = {"RMS": [], "LN": []}
    for eps in EPS_LADDER:
        for fn, key in ((rmsnorm, "RMS"), (layernorm, "LN")):
            d = max(float(np.abs(fn(a * x, eps) - fn(x, eps)).max()) for a in ALPHA_LADDER)
            dev[key].append(d)
        print("    eps=%-8.1e  RMS=%.6e   LN=%.6e" % (eps, dev["RMS"][-1], dev["LN"][-1]))
    for key in ("RMS", "LN"):
        strict = all(dev[key][i] < dev[key][i + 1] for i in range(len(EPS_LADDER) - 1))
        check("A.2" + ("a" if key == "RMS" else "b"),
              strict and dev[key][-1] > EPS_DEV_FLOOR,
              "② %s：三档**严格递增**（%s），且 `1e-3` 档 `%.3e` > `%.0e`"
              % (key, " → ".join("%.2e" % d for d in dev[key]), dev[key][-1], EPS_DEV_FLOOR))
    print("    ★ 量级核对（计划 D7 的记录：`1e-6 → ~2e-7`、`1e-3 → ~2e-4`）："
          "本机 LN 档 = `%.3e` / `%.3e`" % (dev["LN"][1], dev["LN"][2]))

    # ── ③ 平移：LayerNorm 成立、RMSNorm 不成立 ─────────────────────────────
    print()
    print("  ── ③ **平移** `x + b`（`b = %g`，`eps = 1e-6`）──" % SHIFT_B)
    sh = {}
    for fn, key in ((rmsnorm, "RMS"), (layernorm, "LN")):
        sh[key] = float(np.abs(fn(x + SHIFT_B, 1e-6) - fn(x, 1e-6)).max())
        print("    %-4s max|f(x+b) − f(x)| = %.6e" % (key, sh[key]))
    check("A.3", sh["LN"] < SHIFT_LN_TOL,
          "③ LayerNorm 平移不变：`%.3e` < `%.0e`（★ 与 D8 的 `7.99e-15` 同档；"
          "判据只写**不等式**——两个数的口径相同、抽样次数不同，不可逐位对齐）"
          % (sh["LN"], SHIFT_LN_TOL))
    check("A.4", sh["RMS"] > SHIFT_RMS_FLOOR,
          "③ RMSNorm **不**平移不变：`%.6f` > `%.1f`（★ 判据是**不等式**，不是「≈2」；"
          "D8 的 `2.006` 同口径不同抽样）" % (sh["RMS"], SHIFT_RMS_FLOOR))

    # ── ③（续）平移偏差与 eps 的关系 ───────────────────────────────────────
    ln_shift = {eps: float(np.abs(layernorm(x + SHIFT_B, eps) - layernorm(x, eps)).max())
                for eps in EPS_LADDER}
    rms_shift = {eps: float(np.abs(rmsnorm(x + SHIFT_B, eps) - rmsnorm(x, eps)).max())
                 for eps in EPS_LADDER}
    print("    ★ 平移偏差 × eps 三档（★ 这一栏是本程序的一条**新**结论，D7 没有单列）：")
    for eps in EPS_LADDER:
        print("      eps=%-8.1e  RMS=%.9f   LN=%.6e" % (eps, rms_shift[eps], ln_shift[eps]))
    spread_ln = max(ln_shift.values()) - min(ln_shift.values())
    spread_rms = max(rms_shift.values()) - min(rms_shift.values())
    check("A.5", spread_ln < INVARIANCE_TOL,
          "★ LayerNorm 的平移偏差与 `eps` **精确无关**（极差 `%.2e` < `%.0e`）—— "
          "结构上的原因：行**方差**对平移不变，所以 `eps` 只改一个平移无关的因子"
          % (spread_ln, INVARIANCE_TOL))
    check("A.6", all(v > SHIFT_RMS_FLOOR for v in rms_shift.values())
          and spread_rms < 0.01 * min(rms_shift.values()),
          "★ RMSNorm 的平移偏差三档**都** `> %.1f`（`%.6f`/`%.6f`/`%.6f`），而 `eps` 对它的影响"
          "只占自身 `%.1e`（`< 1%%`）⇒ **「`eps` 是破坏者」这句话只对缩放那一半成立**，"
          "平移那一半的原因在别处"
          % (SHIFT_RMS_FLOOR, rms_shift[EPS_LADDER[0]], rms_shift[EPS_LADDER[1]],
             rms_shift[EPS_LADDER[2]], spread_rms / min(rms_shift.values())))


# ===========================================================================
# 4. B 类：闭式参照、恒等式、结构计数、真实模型锚点
# ===========================================================================
def part_B(x):
    print()
    print("=" * 74)
    print("B. 闭式参照 / 恒等式 / 结构计数 / 真实模型锚点")
    print("=" * 74)

    mu = x.mean(axis=-1, keepdims=True)
    ms = (x ** 2).mean(axis=-1)
    sig2 = ((x - mu) ** 2).mean(axis=-1)
    x_absmax = np.abs(x).max(axis=-1)
    xc_absmax = np.abs(x - mu).max(axis=-1)

    # ── B.1 闭式对照：缩放偏差的逐行闭式 ───────────────────────────────────
    print()
    print("  ── B.1 缩放偏差的**逐行闭式** vs 实测（量级判据的证伪能力弱 ⇒ 配闭式）──")
    f_rms = float(np.abs(rmsnorm(x, 1e-3)).max())      # 偏差的**单位**（|f| 的量级）
    f_ln = float(np.abs(layernorm(x, 1e-3)).max())
    print("    偏差的单位：`max|f(x)|` = RMS `%.4f` / LN `%.4f`" % (f_rms, f_ln))
    ok_eps0 = True
    ok_abs = True
    ok_rel = True
    worst_rel = 0.0
    for eps in EPS_LADDER:
        for alpha in ALPHA_LADDER:
            rms_me = float(np.abs(rmsnorm(alpha * x, eps) - rmsnorm(x, eps)).max(axis=-1).max())
            rms_cf = float(closed_form_scale_dev_rmsnorm(ms, x_absmax, alpha, eps).max())
            ln_me = float(np.abs(layernorm(alpha * x, eps) - layernorm(x, eps)).max(axis=-1).max())
            ln_cf = float(closed_form_scale_dev_layernorm(sig2, xc_absmax, alpha, eps).max())
            if eps == 0.0:
                ok_eps0 = ok_eps0 and rms_me < INVARIANCE_TOL and ln_me < INVARIANCE_TOL
                print("    eps=0      α=%-7.1f RMS 实测/闭式 = %.2e/%.2e  LN = %.2e/%.2e"
                      "  （两边都在机器精度档 ⇒ **不逐位比**，只比档位）"
                      % (alpha, rms_me, rms_cf, ln_me, ln_cf))
                continue
            a_rms = abs(rms_cf - rms_me) / f_rms
            a_ln = abs(ln_cf - ln_me) / f_ln
            r_rms = abs(rms_cf - rms_me) / rms_me
            r_ln = abs(ln_cf - ln_me) / ln_me
            worst_rel = max(worst_rel, r_rms, r_ln)
            ok_abs = ok_abs and a_rms < CLOSED_FORM_SCALED_TOL and a_ln < CLOSED_FORM_SCALED_TOL
            ok_rel = ok_rel and r_rms < CLOSED_FORM_REL_TOL and r_ln < CLOSED_FORM_REL_TOL
            print("    eps=%-8.1e α=%-7.1f RMS 实测/闭式 = %.6e/%.6e  LN = %.6e/%.6e"
                  "  （以 `|f|` 为单位的绝对差：%.1e / %.1e）"
                  % (eps, alpha, rms_me, rms_cf, ln_me, ln_cf, a_rms, a_ln))
    check("B.1", ok_abs and ok_eps0,
          "★ 两个算子的缩放偏差**逐行等于闭式**：`eps > 0` 的 6 组，以 `|f|` 为单位的绝对差"
          " `< %.0e`（`eps = 0` 的 3 组：两边都在 `< %.0e` 档）⇒ 偏差不是「差不多大」，"
          "是**可手算的**" % (CLOSED_FORM_SCALED_TOL, INVARIANCE_TOL))
    check("B.1b", ok_rel,
          "★★ **口径注记**（★ 这一条本身就是一条「量法口径」的教训）：同一组比较若改用"
          "**相对**误差，最差是 `%.1e`（阈值 `< %.0e`）—— 它**比机器精度差 7~8 个数量级**。"
          "原因不是闭式不准，而是**被测量的量本身是两个相近大数之差**："
          "`|f(αx) − f(x)|` 只有 `|f|` 的 `~1e-8`，而 `f` 的每个值只有 `~1e-16` 的相对精度 ⇒ "
          "这个差的相对精度被限制在 `1e-16 / 1e-8 = 1e-8`。"
          "⇒ **「偏差的相对误差」这个量口径本身就是坏的**，改用「以 `|f|` 为单位」才可比"
          % (worst_rel, CLOSED_FORM_REL_TOL))

    # ── B.2 闭式：cos(LN, RMS) = σ/√ms ─────────────────────────────────────
    print()
    print("  ── B.2 `cos(LN(x), RMS(x))` 的闭式 = `σ/√ms`（**与 `eps` 无关**）──")
    worst_cos = 0.0
    cf = closed_form_cos_layernorm_rmsnorm(sig2, ms)
    for eps in EPS_LADDER:
        a, b = layernorm(x, eps), rmsnorm(x, eps)
        cos = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
        err = float(np.abs(cos - cf).max())
        worst_cos = max(worst_cos, err)
        print("    eps=%-8.1e  max|cos − σ/√ms| = %.3e   （cos 的最小值 %.6f）"
              % (eps, err, float(cos.min())))
    check("B.2", worst_cos < COS_TOL,
          "★ `cos(LN(x), RMS(x)) == σ/√ms` 在**三个 `eps` 档上都成立**（最差 `%.3e` < `%.0e`）"
          "⇒ `μ = 0 ⟺ cos = 1 ⟺ 两个算子相等` —— 这正是论文那句条件句的闭式形态"
          % (worst_cos, COS_TOL))

    # ── B.2b 两个算子都把这一行投到半径 √n 的球面上（eps = 0）──────────────
    # ★ 这一条是 `§1.4` 的**推论 2**（「相等」还需要范数相等）那一步的依据，
    #   也是「夹角」这句话有意义的前提（两者在同一个球面上，夹角才定义得清楚）。
    print()
    print("  ── B.2b `eps = 0` 时两个算子都把行投到**半径 `√n` 的球面**上 ──")
    rad = {}
    for fn, key in ((layernorm, "LN"), (rmsnorm, "RMS")):
        y = fn(x, 0.0)
        rad[key] = float(np.abs(np.linalg.norm(y, axis=-1) / np.sqrt(N_COLS) - 1.0).max())
        print("    %-4s max|‖f(x)‖/√n − 1| = %.3e   （√n = %.6f）"
              % (key, rad[key], np.sqrt(N_COLS)))
    check("B.2b", rad["LN"] < COS_TOL and rad["RMS"] < COS_TOL,
          "★ 两个算子在 `eps = 0` 时都把这一行投到**半径 `√n = %.6f` 的球面**上"
          "（最大相对误差 LN `%.3e` / RMS `%.3e`）⇒ `§1.4` 的推论 2 那一步（范数相等）"
          "有依据；也说明「两个算子之差」**只**是一个夹角问题"
          % (np.sqrt(N_COLS), rad["LN"], rad["RMS"]))

    # ── B.3 恒等式：均值为 0 ⇒ 相等（+ 对照组）────────────────────────────
    print()
    print("  ── B.3 `mean == 0` ⇒ 两个算子**逐位相等**（论文逐字，见 specs）──")
    rng = np.random.default_rng(SEED + 1)
    v = rng.standard_normal((N_ROWS, N_COLS // 2)) * 2.0 + 0.3
    x0 = np.concatenate([v, -v], axis=-1)        # ★ 成对 ±v ⇒ 行和**恰好**是 0.0
    print("    构造：`x0 = [v, −v]` ⇒ 实测 `max|mean(x0)| = %.3e`" % float(np.abs(x0.mean(-1)).max()))
    ident = {}
    for eps in (0.0, 1e-6):
        ident[eps] = float(np.abs(rmsnorm(x0, eps) - layernorm(x0, eps)).max())
        print("      eps=%-8.1e  max|RMS(x0) − LN(x0)| = %r" % (eps, ident[eps]))
    ctrl = float(np.abs(rmsnorm(x, 0.0) - layernorm(x, 0.0)).max())
    print("      对照组（`mean ≈ %.2f` 的 `x`）max|RMS − LN| = %.6f" % (X_MEAN, ctrl))
    check("B.3", all(d == IDENTITY_EXACT for d in ident.values()) and ctrl > CONTROL_FLOOR,
          "★ 论文逐字（`specs.RMSNORM_MEAN_ZERO_EQUALITY`）的**充要形态**："
          "`mean == 0` ⇒ 差**恰好** `0.0`（**恒等式**，不是机器精度比较，见模块头）；"
          "对照组（`mean ≈ %g`）`%.6f` > `%.0e` ⇒ 「相等」**只在均值为 0 时**成立"
          % (X_MEAN, ctrl, CONTROL_FLOOR))

    # ── B.4 结构计数：2 个归约量 vs 1 个 ───────────────────────────────────
    print()
    print("  ── B.4 结构计数：两个算子各需要几个归约量 ──")
    n_ln = len(stats_layernorm(x))
    n_rms = len(stats_rmsnorm(x))
    print("    LayerNorm 的统计量 = %d 个（规模 %s）；RMSNorm 的统计量 = %d 个（规模 %s）"
          % (n_ln, stats_layernorm(x)[0].shape, n_rms, stats_rmsnorm(x)[0].shape))
    check("B.4", n_ln == STAT_COUNT_LN and n_rms == STAT_COUNT_RMS,
          "★ 结构计数（本域是**结构侧的账**，不是访存账）：LayerNorm **%d** 个归约量、"
          "RMSNorm **%d** 个 —— 这是「去掉均值」这句话唯一**可机械判定**的内容"
          % (n_ln, n_rms))

    # ── B.5 真实模型锚点：eps 取在哪一档 ───────────────────────────────────
    print()
    print("  ── B.5 真实模型的 `rms_norm_eps`（**经镜像取得**）落在哪一档 ──")
    real = (("Llama-3-8B", specs.LLAMA3_8B_RMS_NORM_EPS),
            ("Qwen3-30B-A3B", specs.QWEN3_30B_A3B_RMS_NORM_EPS))
    devs = []
    for name, eps in real:
        d = float(np.abs(rmsnorm(ALPHA_LADDER[0] * x, eps) - rmsnorm(x, eps)).max())
        devs.append(d)
        print("    %-16s eps=%-8.1e  ⇒ RMS 的缩放偏差（α=%g）= %.6e"
              % (name, eps, ALPHA_LADDER[0], d))
    ratio = devs[0] / devs[1]
    inband = all(REAL_EPS_BAND[0] < v < REAL_EPS_BAND[1] for v in devs)
    check("B.5", inband and devs[0] > devs[1] and 9.0 < ratio < 11.0,
          "★ 两个真实模型的 eps（`1e-5` / `1e-6`，**相差 10 倍**）使偏差落在 `(%.0e, %.0e)` 档，"
          "且**严格有序**、**比值 `%.3f` ≈ 10**（`1e-5` 档同时就是判据 ② 的中档）⇒ "
          "偏差与 `eps` **同阶**（D7 的那句话在这里被量化成「比值 ≈ 10」）"
          % (REAL_EPS_BAND[0], REAL_EPS_BAND[1], ratio))


# ===========================================================================
# 5. C 类：边界与反例
# ===========================================================================
def part_C(x):
    print()
    print("=" * 74)
    print("C. 边界与反例")
    print("=" * 74)

    # ── C.1 「平移偏差是 eps 造成的」当场被否 ──────────────────────────────
    r0 = float(np.abs(rmsnorm(x + SHIFT_B, 0.0) - rmsnorm(x, 0.0)).max())
    l0 = float(np.abs(layernorm(x + SHIFT_B, 0.0) - layernorm(x, 0.0)).max())
    print()
    print("  ── C.1 把 `eps` 设成 **0**，平移偏差会消失吗？──")
    print("    eps = 0：RMS `%.6f`   LN `%.3e`" % (r0, l0))
    check("C.1", r0 > SHIFT_RMS_FLOOR and l0 < SHIFT_LN_TOL,
          "★ **反例**：`eps = 0` 时 RMS 的平移偏差**仍然** `%.6f` > `%.1f` ⇒ "
          "「平移偏差是 `eps` 造成的」**当场被否**；而对 LN 它仍是 `%.2e`（机器精度）"
          % (r0, SHIFT_RMS_FLOOR, l0))

    # ── C.2 偏差在 α 方向**不是**单调的 ───────────────────────────────────
    print()
    print("  ── C.2 偏差在 **α 方向不是单调的**（`eps = 1e-3` 固定）──")
    lobe = []
    for alpha in ALPHA_DECAY_LADDER + (1.0,) + ALPHA_LADDER:
        d_rms = float(np.abs(rmsnorm(alpha * x, 1e-3) - rmsnorm(x, 1e-3)).max())
        d_ln = float(np.abs(layernorm(alpha * x, 1e-3) - layernorm(x, 1e-3)).max())
        lobe.append((alpha, d_rms, d_ln))
        print("    α=%-9.4g  RMS=%.6e   LN=%.6e" % (alpha, d_rms, d_ln))
    fmax = float(np.abs(rmsnorm(x, 1e-3)).max())
    small = dict((a, d) for a, d, _ in lobe)
    check("C.2", small[1.0] == IDENTITY_EXACT and small[ALPHA_DECAY_LADDER[0]] > fmax * 0.5,
          "★ **反例**：偏差在 α 上**不是单调的** —— `α = 1` 时**恰好** `0.0`（不变性的定义），"
          "`α → 0⁺` 的一端升到 `%.4f`（≈ `|f(x)|` 的 `%.2f` 倍：`eps` 成了主项、"
          "`f(αx) → 0`，那是**尺度塌缩**而不是不变性被破坏）"
          % (small[ALPHA_DECAY_LADDER[0]], small[ALPHA_DECAY_LADDER[0]] / fmax))
    check("C.2b", 1.0 not in ALPHA_LADDER,
          "★ 对照：判据 ① 的 α 阶梯 `%s` **不含** `α = 1`（含它会让偏差恒为 0，"
          "那是一条恒真断言）" % (list(ALPHA_LADDER),))

    # ── C.3 「缩放不是两者的分界」 ─────────────────────────────────────────
    print()
    print("  ── C.3 分界澄清：**缩放**不变性两者都有（`eps = 0`，尺度 `%g`）──" % ROW_SCALE)
    d_rms = float(np.abs(rmsnorm(ROW_SCALE * x, 0.0) - rmsnorm(x, 0.0)).max())
    d_ln = float(np.abs(layernorm(ROW_SCALE * x, 0.0) - layernorm(x, 0.0)).max())
    print("    RMS `%.3e` / LN `%.3e` —— 两者都在机器精度档" % (d_rms, d_ln))
    check("C.3", d_rms < INVARIANCE_TOL and d_ln < INVARIANCE_TOL,
          "★ 两者**都**缩放不变（`%.2e` / `%.2e`，尺度 `%.1f`）⇒ 引用时**不得**把缩放当作"
          "它们的差别：**分界线是平移**（`A.3` vs `A.4`）"
          % (d_rms, d_ln, ROW_SCALE))
    print("    （本域的裁决：探测阶段控制者**预期错过一次**——先以为「LN 不缩放等变、RMS 等变」，"
          "实测说两者都等变，见 `probes/probe_80b.py` 的模块头）")


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

    measured = {"N_ROWS": N_ROWS, "N_COLS": N_COLS, "X_MEAN": X_MEAN, "X_STD": X_STD,
                "SEED": SEED, "EPS_LADDER": EPS_LADDER, "ALPHA_LADDER": ALPHA_LADDER,
                "SHIFT_B": SHIFT_B, "ALPHA_DECAY_LADDER": ALPHA_DECAY_LADDER,
                "ROW_SCALE": ROW_SCALE}
    expect = {"N_ROWS": 64, "N_COLS": 512, "X_MEAN": 5.0, "X_STD": 3.0, "SEED": 20260922,
              "EPS_LADDER": (0.0, 1e-6, 1e-3), "ALPHA_LADDER": (2.0, 50.0, 1000.0),
              "SHIFT_B": 100.0, "ALPHA_DECAY_LADDER": (1e-3, 1e-2, 0.1), "ROW_SCALE": 7.0}
    diff = {k: (expect[k], v) for k, v in measured.items() if v != expect[k]}
    check("D.1", not diff,
          "口径常量 10 个逐位一致（★ 本程序的断言几乎全是**不变量**形态 ⇒ 改掉行数/种子/均值，"
          "所有断言**照样通过**；口径本身只能靠这一条守住）%s"
          % ("" if not diff else " ← 差异 %s" % diff))

    th = {"INVARIANCE_TOL": INVARIANCE_TOL, "SHIFT_LN_TOL": SHIFT_LN_TOL,
          "SHIFT_RMS_FLOOR": SHIFT_RMS_FLOOR, "EPS_DEV_FLOOR": EPS_DEV_FLOOR,
          "CLOSED_FORM_SCALED_TOL": CLOSED_FORM_SCALED_TOL,
          "CLOSED_FORM_REL_TOL": CLOSED_FORM_REL_TOL, "COS_TOL": COS_TOL,
          "IDENTITY_EXACT": IDENTITY_EXACT, "CONTROL_FLOOR": CONTROL_FLOOR,
          "STAT_COUNT_LN": STAT_COUNT_LN, "STAT_COUNT_RMS": STAT_COUNT_RMS,
          "REAL_EPS_BAND": REAL_EPS_BAND}
    th_expect = {"INVARIANCE_TOL": 1e-14, "SHIFT_LN_TOL": 1e-12,
                 "SHIFT_RMS_FLOOR": 1.0, "EPS_DEV_FLOOR": 1e-5,
                 "CLOSED_FORM_SCALED_TOL": 1e-12, "CLOSED_FORM_REL_TOL": 1e-6, "COS_TOL": 1e-14, "IDENTITY_EXACT": 0.0,
                 "CONTROL_FLOOR": 1e-3, "STAT_COUNT_LN": 2, "STAT_COUNT_RMS": 1,
                 "REAL_EPS_BAND": (1e-9, 1e-5)}
    tdiff = {k: (th_expect[k], v) for k, v in th.items() if v != th_expect[k]}
    check("D.2", not tdiff,
          "阈值 12 个逐位一致（★ 放宽 `INVARIANCE_TOL` / `SHIFT_LN_TOL` ⇒ 判据退化成恒真；"
          "把 `IDENTITY_EXACT` 放宽 ⇒ 「恰好 0」降级成「差不多」）%s"
          % ("" if not tdiff else " ← 差异 %s" % tdiff))

    # ── specs 的常量：**先钉字符串/取值，再谈用** ──────────────────────────
    pin = (specs.LLAMA3_8B_RMS_NORM_EPS == 1e-05
           and specs.QWEN3_30B_A3B_RMS_NORM_EPS == 1e-06)
    check("D.3", pin,
          "★ `specs.py` 的两个真实模型 eps **逐位钉住**：Llama `%r`、Qwen3 `%r`"
          "（★ **经 `hf-mirror.com` 镜像取得，未与官方 repo 比对 sha256** —— "
          "这条限定语必须与数字一起引用）"
          % (specs.LLAMA3_8B_RMS_NORM_EPS, specs.QWEN3_30B_A3B_RMS_NORM_EPS))

    # ★★ 引用判据（表 B）本身不能被程序判定，但**「我抄的这一句是不是原文」可以**：
    #    下面三条把 [C] 常量做**逐位**比较 —— 静默改掉一个字，这里当场红。
    quotes_ok = (
        specs.RMSNORM_MEAN_ZERO_EQUALITY ==
        "When the mean of summed inputs is zero, RMSNorm is exactly equal to LayerNorm."
        and specs.RMSNORM_HYPOTHESIS_RECENTERING ==
        "we hypothesize that re-centering invariance in LayerNorm is dispensable"
    )
    check("D.4", quotes_ok,
          "★★ **论文逐字的逐位钉住**（表 B 的 X11 落点）："
          "① `%s`；② `%s` —— ★ ② 用的是 **`hypothesize`**，"
          "**不得**在正文里转述成「已证明」"
          % (specs.RMSNORM_MEAN_ZERO_EQUALITY, specs.RMSNORM_HYPOTHESIS_RECENTERING))

    check("D.5", specs.UNSOURCED == (
        "qk_norm_primary_source", "qk_norm_in_real_configs", "eps_position_numerical_diff",
        "attention_entropy_collapse_threshold", "layernorm_vs_rmsnorm_quality"),
          "★ `specs.UNSOURCED` 是**待办清单**（%d 项），**五项逐位（含顺序）钉住** —— "
          "★ 这是 R17 的落法：**不给它开后门**（不从变异测试的分母里排除），"
          "而是**给它一个消费者**（先例：Chunk 1 让 `03_depth_width.py` 的 `D.4c` 把三项逐位钉住）。"
          "其中 `eps_position_numerical_diff` 记的是**禁区**"
          "（`eps` 位置的数值差异 → 正文在 `40-kernels/softmax-normalization/03`）、"
          "`attention_entropy_collapse_threshold` 记的是本域**拒绝给出**的阈值"
          % len(specs.UNSOURCED))


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
    check("D.7", ASSERT_FAIL_MARK == "~~ FAIL: normalization"
          and "FAIL" in ASSERT_FAIL_MARK
          and MARKS_USED <= {ASSERT_FAIL_MARK},
          "★ 标记自检（**字面量比较**，R18）：**等于**本主题被分配的字面量、含 `FAIL`，"
          "且打印失败行时用的标记 ⊆ {%r}（本次实际用到 %d 个）"
          % (ASSERT_FAIL_MARK, len(MARKS_USED)))
    check("D.8", OTHER_THEME_MARKS == ("!! FAIL: transformer", "@@ FAIL: attention-variants",
                                       "## FAIL: moe", "$$ FAIL: positional-encoding"),
          "★ 「不跨主题照抄」这条判据的**对照物**也逐位钉住")

    # ── 两个算子与三个闭式必须**互不调用**（AST 级 + 文本级双证）──────────
    measured = {rmsnorm, layernorm, stats_layernorm, stats_rmsnorm}
    m_names = {f.__name__ for f in measured}
    cf_names = {closed_form_scale_dev_rmsnorm, closed_form_scale_dev_layernorm,
                closed_form_cos_layernorm_rmsnorm}
    shared_cf = sorted(set().union(*[_callee_names(f) for f in cf_names]) & m_names)
    cross = sorted(_callee_names(rmsnorm) & {"layernorm", "stats_layernorm"}
                   | (_callee_names(layernorm) & {"rmsnorm", "stats_rmsnorm"}))
    # ★ 文本级回退：闭式的**函数体**里不许出现那两个算子（或 `stats_`）的名字。
    #   与 AST 检查互为补集 —— AST 只看得见「以名字调用的函数」，
    #   `getattr(globals(), "rmsnorm")(...)` 那类绕过只有文本级能挡。
    text_cf = "\n".join([_body_text(f) for f in sorted(cf_names, key=lambda g: g.__name__)]
                        + [_body_text(rmsnorm), _body_text(layernorm)])
    textual = not any(s in text_cf for s in ("rmsnorm(", "layernorm(", "stats_"))
    src_r = inspect.getsource(rmsnorm)
    src_l = inspect.getsource(layernorm)
    check("D.9", not shared_cf and not cross and textual and src_r != src_l,
          "★ **参考实现的独立性**（`CONTRIBUTING` §2 / 域 README）：三个闭式的调用表与两个算子"
          "**交集 = %s**（⇒ `B.1`/`B.2` 是**另一条路**算出来的，不是把测量结果抄一遍）；"
          "两个算子**互不调用**（交集 = %s）；文本级回退 = %s；两个算子的源码不同"
          "（%d vs %d 字节，`B.3` 的「逐位相等」因此**不是**同一份代码产出的）"
          % (shared_cf, cross, textual, len(src_r), len(src_l)))

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
    print("01_rmsnorm_layernorm.py —— ★★ 不变性的精确形态（哪个不变、哪个只在 eps→0 时不变）")
    print("=" * 74)
    print("  ★ 真值标注：本程序的数字**全部是推导值/本机实测值**"
          "（本机 `python3 %s` + `numpy %s`）。"
          % (sys.version.split()[0], np.__version__))
    print("    本机 macOS arm64 **无 GPU**：这些数不支撑任何关于真实模型或 GPU 的结论。")
    print("  ★ 失败标记 ASSERT_FAIL_MARK = %r（本主题自定义；不跨主题照抄）"
          % ASSERT_FAIL_MARK)
    print("  ★★ 禁区（计划 §2）：本程序**不写公式、不推公式**（LayerNorm / RMSNorm 的公式定义、")
    print("     Welford 递推、`Σx²−nμ²` 的灾难、`eps` **位置**的数值差异 —— 正文在")
    print("     `40-kernels/softmax-normalization/03-normalization-fusion.md:55-71`，正文在彼，此处回链）；")
    print("     **不算**归约的访存账（在 `03:213-263`）；**不下训练侧结论**。")
    print("  ★ 本程序谈的是 `eps` 的**量级**，不是它的**位置**。")
    pin_part()
    rng = np.random.default_rng(SEED)
    x = rng.standard_normal((N_ROWS, N_COLS)) * X_STD + X_MEAN
    part_A(x)
    part_B(x)
    part_C(x)
    mark_part()
    return summary()


if __name__ == "__main__":
    sys.exit(main())

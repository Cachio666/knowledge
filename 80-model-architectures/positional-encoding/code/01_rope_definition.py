#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""01_rope_definition.py —— RoPE 的**旋转定义**、**相对位置性质**、以及**两套实现约定**。

配套文档：`01-rope-definition.md`。
★ 三类断言（A 算法正确性 / B 代价模型 / C 边界与反例）**全部在本程序内部**，
失败时以**非零退出码**结束；**没有 `verify.py`**。

===============================================================================
★★ 本程序要回答的那三个问题
===============================================================================

  1. **`⟨R_m q, R_n k⟩` 真的只依赖 `n−m` 吗？**（判据 ①）
     ⇒ 实测：**是**，而且它有一个**闭式** —— 把 `m` 代进去会**解析地抵消**，
       剩下的量只用 `δ = n−m` 与**逐对的两个分量**就能算出来，
       **完全不需要构造 `R_m`**（`B.1` 的两条路）。
     ★★ 判据必须用**尺度无关的绝对判据**：`max_m |v_m − v_0| / (‖q‖·‖k‖) < 1e-12`
       —— **不得**用相对极差 `(max−min)/|mean|`（`C.1` 把那个统计量的坏处量出来了）。

  2. **两套实现约定是什么关系？**（判据 ②）—— ★★ **这是本程序的核心**。
     原论文的旋转矩阵是**块对角、相邻对**（`(x_1,x_2), (x_3,x_4), …`，式 (15)）；
     而 Llama / HF 一系用的是 `rotate_half`（**前后两半配对**：`(x_i, x_{i+d/2})`）。
     ★★ **两者不是两个不同的运算**：存在一个固定的坐标置换 `I`
       （`x[:d/2]` 散到**偶数位**、`x[d/2:]` 散到**奇数位**），使得

           I(half(x, pos)) == adj(I(x), pos)

       **差恒为 `0.0`**（不是机器精度，是**恒等**）。
     ★★ **这是 P14（「把实现的约定说成架构的定义」）的正面示范**：
       论文那套是**定义**（哪两个分量共享一个角度），`rotate_half` 那套是**实现怎么摆维度**
       —— 它们**只差一次坐标置换**。
     ★★ **这条恒等式是「索引恒等式」，不是「测出来的精度」**（计划 §4 的 R16）：
       两边执行的是**同一批标量运算**（同一个 `a*c − b*s`），只是结果摆放的位置不同
       ⇒ 所以它**必须**写成 `== 0.0`，且**不得**包装成「机器精度比较」。
       ★ 本程序用**另一条独立的证据**补上它的实质内容（`B.2`）：
         **把同一个置换施到 `W_q`/`W_k` 的输出维上，两套约定给出的注意力分数逐位相同**
         （实测相对差 `2.1e-16`）；而**不做**这个置换时差到 `0.41`（`B.2b` 的对照）。

  3. **一个实现「对不对」怎么判？**（`C.2` 的负对照）
     把 `q` 与 `k` 的**频率表错开一位** ⇒ 判据 ① **当场失效**（实测 `3.2e-01`）。
     ⇒ 这说明判据 ① 不是恒真断言：它真的在测「两个位置用的是同一张频率表」。

===============================================================================
★★ 本程序的数字与它们的来源等级（三类标签必须分开）
===============================================================================

  · **本机实测值 / 推导值**：`python3 3.13.5` + `numpy 2.4.4`，macOS arm64，**无 GPU**。
    ⇒ **不支撑任何关于真实模型或 GPU 的结论。**
  · **外部规格值**：两个真实模型的 `rope_theta` 与 `max_position_embeddings`
    （`specs.py`，**经镜像取得**）。
  · **论文逐字**：`specs.py` 的 RoPE 那批 [C] 常量（arXiv 2104.09864，
    ★ abs 页 `comments` 只有 `fixed some typos`、**无 venue 线索** ⇒ **未确认同行评审**）。

===============================================================================
★★ 可证伪断言（把下列任一处改错，本程序必须非零退出）
===============================================================================

  · `A.1`  ① 相对位置性质：**五个** `δ`（`1 / 7 / 64 / 255 / 511`）上
           `max_m |v_m − v_0| / (‖q‖·‖k‖) < 1e-12`（实测 `~1e-15` 档）；
  · `A.2`  ★ **判据① 不是恒真断言**：`v_δ` 本身随 `δ` 变化（`max_δ |v_δ − v_0| / (‖q‖‖k‖) > 0.01`）
           —— 没有这一条，「`v_m` 与 `m` 无关」可能只是因为「什么都无关」；
  · `B.1`  ★★ **闭式（独立的一路）**：上面那个量与**只用 `δ`、逐对两个分量**算出的闭式
           逐位相等（同一个 `< 1e-12`）⇒ `m` 是**解析地**抵消的，不是「数值上差不多」；
  · `B.2`  ★★ 判据②：`I(half(x, pos))` 与 `adj(I(x), pos)` 的差 **恰好 `== 0.0`**
           （**恒等式**，见上）；且 `pos` 走**四档**（含非整数）；
  · `B.2b` ★★ **置换的实质内容（独立路径）**：把 `I^{-1}` 的行置换施到 `W_q`/`W_k` 上，
           `rotate_half` 那套给出的**注意力分数**与论文那套逐位相同（相对差 `< 1e-14`）；
           **不施**置换时相对差 `> 0.1`（实测 `0.41`）⇒ 置换**不是**可省的；
  · `B.3`  ★ **两套约定各自**满足判据 ①，且**各自保范数**（相对误差 `< 1e-14`）
           ⇒ 「都满足相对位置性质、都保范数」这句话是两条独立断言，不是一句概括；
  · `B.4`  ★ **表 B 的引用钉住（X2 / X3 / X4）**：论文里**两个**频率版本、**两个**相对位置
           符号方向、式 (16) 的**笔误**，三段原文逐位钉在 `specs` 里，程序逐位比较；
  · `C.1`  ★★ **量法口径注记**：同一份数据、同一个实现，改用 `(max−min)/|mean|` 时
           读数比尺度无关判据**大 3 个数量级以上** ⇒ 那个统计量**测的是数据不是实现**；
  · `C.2`  ★★ **负对照**：`q` 与 `k` 的频率表错开一位 ⇒ 判据 ① 从 `~1e-15` 掉到 `> 0.01`；
  · `C.3`  ★ **方向（符号）是可判定的**：把相对位置反过来（`m−n` ↔ `n−m`）⇒
           同一个量在**同一批数据**上给出**不同的值**，而两条式子各自**内部自洽**
           ⇒ 「抄哪一处」这件事**有后果**（这正是 X3 的内容）。

===============================================================================
★★ 本程序刻意不做的事（计划 §2 的边界表，越界即作废）
===============================================================================

  * ★★ **不写「成对通道旋转」那句结构陈述的正文** —— 它的逐字在
    `130-model-optimization/quantization/19-kivi-asymmetry.md:184`
    （那里讲 pre-RoPE 量化）。本程序给的是**定义**（哪两个分量配对、转多少角度）；
    **KIVI 的量化结论不在本域**。
  * **不算**位置编码的 FLOPs / 访存账（正文在 `40-kernels/attention`）。
  * **不讲**长短期的衰减率证明（`02` 从**本机实测的频率谱**讲；本程序只钉住
    论文那句 `One can prove that this setting provides a long-term decay property` 的逐字）。
  * **不讲**任何外推方法（PI / NTK / YaRN）—— 那是 `03`（★ 本主题的分工写死：
    `02` 给定量框架、`03` 用它解释工程方法）。
  * ★ **不下**「哪种实现约定更快 / 更省」的结论：本域**无 GPU**，且那是 kernel 侧的事。

用法：
    python3 01_rope_definition.py
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
# ★★ **为什么口径常量必须被逐位钉住**：本程序的断言几乎全是「不变量」形态 ——
#    把 `N_POS` 从 `512` 改成 `513`、把种子换掉、把 `δ` 阶梯换一组，**断言照样通过**。
#    ⇒ 口径本身就是**测量条件**，只能靠 `D.1` 的**字面量比较**守住。
# ★★ **为什么这张表里一个负数都没有**（计划 §3 开头的第二个盲区）：
#    `mutate.py` 的叶子枚举只取 `ast.Constant`，而 `-1` 是 `UnaryOp(USub, Constant(1))`
#    ⇒ **写成负数的表整体不进变异分母**。本主题方向/符号密集（`m−n` vs `n−m`、置换的两半），
#    所以「方向」用**正数索引**表达（见 `REL_DIRECTION_INDEX`、`SHIFT_FREQ_SLOTS`），
#    负号一律在**函数体里派生**（`1 - 2*idx`）。
D_MAIN = 64              # 主实验的 head 维度（★ 真实模型是 128，这里用小维度换来可读性）
D_SMALL = 8              # 判据② 的维度（与计划 §0.7-D 的 D2 同口径：`d = 8`）
N_POS = 512              # 判据① 里 `m` 的取值范围（`m = 0 .. N_POS-1`）
DELTA_LADDER = (1, 7, 64, 255, 511)   # 判据① 的 δ 阶梯（**五个点**）
POS_LADDER = (0.0, 1.3, 5.0, 127.5)   # 判据② 的 `pos` 档（★ 含**非整数**：RoPE 支持它）
M_SCAN_MAX = 512         # `C.1` 的 δ 全扫描上界（`1 .. M_SCAN_MAX-1`）
DH_PERM = 32             # `B.2` 权重置换实验的 head 维度
SEQ_PERM = 6             # `B.2` 权重置换实验的序列长度
SEED = 20260922          # 主数据的种子（与探测脚本同源）
SEED_PERM = 11           # `B.2` 的种子（独立）
NORM_SAMPLES = 8         # `B.3b` 每个 `pos` 抽几组 `x` 来测保范数（★ 单样本可能恰好是 0）

# ── 阈值（★ 逐个在 `D.2` 钉住）─────────────────────────────────────────────
REL_POS_TOL = 1e-12            # ① 相对位置性质（**尺度无关的绝对判据**）
REL_POS_VARY_FLOOR = 0.01      # ② 判据① 的**非空性**下界（`v_δ` 必须随 δ 变）
CONJUGATION_EXACT = 0.0        # ② 共轭恒等式的容差（**恰好 0**，见模块头）
NORM_TOL = 1e-14               # ② 保范数（相对）
PERMUTED_SCORE_TOL = 1e-14     # `B.2` 置换权重后分数的**相对**差上界
UNPERMUTED_SCORE_FLOOR = 0.1   # `B.2b` 不置换时的**相对**差下界（实测 `0.41`）
REL_RANGE_RATIO_FLOOR = 1000.0 # `C.1` 两个统计量的比值下界（实测 `~3200`）
DIRECTION_MISMATCH_FLOOR = 0.01  # `C.2` 负对照的下界（实测 `0.32`）
DIRECTION_SEPARATION_FLOOR = 1e-3  # `C.3` 两个方向给出的值必须分开（实测 `2.2e-1`）


# ===========================================================================
# 2. 频率表与两套旋转实现（★ **测量的对象**，不是本程序的正文）
# ===========================================================================
# ★★ **本域固定用哪一版频率公式**（计划 §4 的 R7）：`θ_i = base^{-2(i-1)/d}`，`i ∈ [1, d/2]`
#    —— 与 `specs.ROPE_FREQ_SET_EQ15` 一致、下标从 `specs.ROPE_FREQ_INDEX_ORIGIN` 起。
#    论文里另有一版 `10000^{-2i/d}`（`specs.ROPE_FREQ_SET_LONGTERM`），**两式给出的频率
#    集合完全相同**（差一个下标平移）⇒ 结论不受影响，但**引用时必须说明用的是哪一版**。
def freqs(d, base):
    """`θ_i = base^{-2(i-1)/d}`，`i = 1..d/2`（返回长度 `d/2` 的数组）。"""
    i = np.arange(1, d // 2 + 1, dtype=np.float64)
    return base ** (-2.0 * (i - 1.0) / d)


def _angles(pos, th):
    """`pos`（标量或一维数组）× `th` 的外积，形状 `np.shape(pos) + (d/2,)`。

    ★ 用**外积**而不是 `pos * th`：后者在 `pos` 是一维数组时**广播不了**
      （`(n,)` 与 `(d/2,)` 不对齐）。★ 标量档下两者逐位相同 ⇒ 共轭恒等式不受影响。
    """
    return np.multiply.outer(np.asarray(pos, dtype=np.float64), th)


def rope_adjacent(x, pos, th):
    """**论文那套**（式 (15)）：**相邻对** `(x_1,x_2), (x_3,x_4), …`，每对各转 `pos·θ_i`。

    `pos` 可以是标量，也可以是一维数组（返回形状 `pos.shape + (d,)`）。
    """
    xr = x.reshape(-1, 2)
    ang = _angles(pos, th)
    c = np.cos(ang)
    s = np.sin(ang)
    out = np.stack([xr[:, 0] * c - xr[:, 1] * s,
                    xr[:, 0] * s + xr[:, 1] * c], axis=-1)
    return out.reshape(np.shape(np.asarray(pos)) + (int(x.shape[-1]),))


def rope_half(x, pos, th):
    """**Llama / HF 那套**：前后**两半**配对 `(x_i, x_{i+d/2})`，同样各转 `pos·θ_i`。"""
    h = x.shape[-1] // 2
    a, b = x[..., :h], x[..., h:]
    ang = _angles(pos, th)
    c = np.cos(ang)
    s = np.sin(ang)
    return np.concatenate([a * c - b * s, a * s + b * c], axis=-1)


def perm_I(x):
    """坐标置换 `I`：`x[:d/2]` 散到**偶数位**、`x[d/2:]` 散到**奇数位**。"""
    h = x.shape[-1] // 2
    y = np.empty_like(x)
    y[..., 0::2] = x[..., :h]
    y[..., 1::2] = x[..., h:]
    return y


def perm_Iinv(y):
    """`I` 的逆：偶数位收成前半、奇数位收成后半。"""
    return np.concatenate([y[..., 0::2], y[..., 1::2]], axis=-1)


def row_perm_index(dh):
    """`I^{-1}` 作为**行置换**作用在 `(dh, dh)` 矩阵上时用的下标序列。

    ★ 为什么是 `I^{-1}` 而不是 `I`：我们要把 `q = W_q x` 换成 `I^{-1}(W_q x)`，
      于是新矩阵的行是 `W_q` 的**第 `idx[j]` 行** ⇒ 置换作用在**输出维（行）**上。
      ★★ 方向搞反是**静默**的（分数照样算得出来，只是错的）—— 见 `B.2b` 的对照。
    """
    return np.concatenate([np.arange(0, dh, 2), np.arange(1, dh, 2)])


# ── ★ 参照实现（**不构造 `R_m`**，`D.9` 的 AST + 文本检查守着这一点）─────────────
def score_closed_form(q, k, delta, th):
    """`⟨R_m q, R_{m+δ} k⟩` 的**闭式**：`m` 解析地抵消，只剩 `δ`。

    逐对展开：`⟨R_m q, R_{m+δ}k⟩ = Σ_i Re[ q_i conj(k_i) · e^{-i·δθ_i} ]`
    ⇒ 只用 `δ` 与**每一对的两个分量**，**从头到尾没有 `m`、也没有旋转矩阵**。
    """
    qr, kr = q.reshape(-1, 2), k.reshape(-1, 2)
    c, s = np.cos(delta * th), np.sin(delta * th)
    per_pair = qr[:, 0] * (kr[:, 0] * c - kr[:, 1] * s) \
        + qr[:, 1] * (kr[:, 0] * s + kr[:, 1] * c)
    return float(per_pair.sum())


def score_with_direction(q, k, delta, th, direction_index):
    """★ `C.3` 的载体：把相对位置**反一个方向**再算同一个量。

    `direction_index == 1` ⇒ 用 `δ`（`m − n` 那一支）；`2` ⇒ 用 `−δ`（`n − m` 那一支）。
    ★ 符号由**正数索引派生**（`1 − 2*idx`），表里不出现负字面量（见 §1 的注记）。
    """
    sign = 1.0 - 2.0 * (direction_index - specs.REL_DIRECTION_INDEX)
    return score_closed_form(q, k, sign * delta, th)


# ===========================================================================
# 3. A 类：判据 ① —— 相对位置性质（含「它不是恒真断言」的对照）
# ===========================================================================
def part_A(th):
    print("=" * 74)
    print("A. 判据 ①：`⟨R_m q, R_n k⟩` 只依赖 `n−m`")
    print("=" * 74)
    rng = np.random.default_rng(SEED)
    q = rng.standard_normal(D_MAIN)
    k = rng.standard_normal(D_MAIN)
    scale = float(np.linalg.norm(q) * np.linalg.norm(k))
    print("  口径：`d = %d`、`m = 0..%d`、`δ ∈ %s`、种子 `%d`；`‖q‖·‖k‖ = %.6f`"
          % (D_MAIN, N_POS - 1, list(DELTA_LADDER), SEED, scale))
    print("  ★ 判据式（**尺度无关的绝对判据**）：`max_m |v_m − v_0| / (‖q‖·‖k‖)`")
    print("    —— ★ **不得**用相对极差 `(max−min)/|mean|`，理由见 `C.1`（量法口径）。")

    m_all = np.arange(N_POS, dtype=np.float64)
    qrot = rope_adjacent(q, m_all, th)          # (N_POS, d)
    worst_abs = 0.0
    worst_closed = 0.0
    vary = 0.0
    print()
    print("  %6s %18s %18s %14s" % ("δ", "v_m 的绝对判据", "闭式的差", "v_δ 本身变化"))
    for delta in DELTA_LADDER:
        krot = rope_adjacent(k, m_all + delta, th)
        v = (qrot * krot).sum(axis=1)
        crit = float(np.abs(v - v[0]).max()) / scale
        # ★ 闭式那一路与实测**逐位**比较：它只用 δ，不用 m
        closed_all = score_closed_form(q, k, float(delta), th)
        crit_closed = float(np.abs(v - closed_all).max()) / scale
        v0d = score_closed_form(q, k, 0.0, th)
        vary = max(vary, abs(closed_all - v0d) / scale)
        worst_abs = max(worst_abs, crit)
        worst_closed = max(worst_closed, crit_closed)
        print("  %6d %18.4e %18.4e %14.6f"
              % (delta, crit, crit_closed, abs(closed_all - v0d) / scale))

    check("A.1", worst_abs < REL_POS_TOL,
          "① 相对位置性质：五个 `δ` 上最差 `%.4e` < `%.0e`（★ 尺度无关；"
          "实测与计划 D1 的 `2.1e-15 ~ 9.7e-15` **同档**，口径相同、抽样历史不同）"
          % (worst_abs, REL_POS_TOL))
    check("A.2", vary > REL_POS_VARY_FLOOR,
          "★ **判据① 不是恒真断言**：`v_δ` 自己随 `δ` 变化（最差 `%.6f` > `%.2f`）—— "
          "没有这一条，「`v_m` 与 `m` 无关」可能只是「什么都不相关」"
          % (vary, REL_POS_VARY_FLOOR))
    check("B.1", worst_closed < REL_POS_TOL,
          "★★ **闭式（独立的一路）**：`v_m` 与「只用 `δ`、逐对两个分量」算出的闭式"
          "**逐位相等**（最差 `%.4e` < `%.0e`）⇒ `m` 是**解析地**抵消的，"
          "不是「数值上差不多」；★ 这条闭式**从头到尾没有构造过 `R_m`**（`D.9` 检查）"
          % (worst_closed, REL_POS_TOL))

    # ── 判据② 的两条前置：两套约定**各自**满足 ① ──────────────────────────
    print()
    print("  ── 判据② 的前置：**两套约定各自**满足判据①（`d = %d`）──" % D_MAIN)
    worst = {}
    for name, fn in (("论文的相邻对", rope_adjacent), ("rotate_half", rope_half)):
        qr = fn(q, m_all, th)
        w = 0.0
        for delta in DELTA_LADDER:
            kr = fn(k, m_all + delta, th)
            w = max(w, float(np.abs((qr * kr).sum(axis=1) - (qr * kr).sum(axis=1)[0]).max())
                    / float(np.linalg.norm(qr[0]) * np.linalg.norm(kr[0])))
        worst[name] = w
        print("    %-14s 最差 = %.4e" % (name, w))
    check("B.3a", worst["论文的相邻对"] < REL_POS_TOL and worst["rotate_half"] < REL_POS_TOL,
          "★ 两套约定**各自**满足相对位置性质（`%.4e` / `%.4e`）—— "
          "★ 这是两条**独立**断言，「都满足」不是一句概括"
          % (worst["论文的相邻对"], worst["rotate_half"]))


# ===========================================================================
# 4. B 类：判据 ② —— 两套实现是「同一操作被坐标置换共轭」
# ===========================================================================
def part_B(th_small):
    print()
    print("=" * 74)
    print("B. 判据 ②：两套实现是「同一操作被**坐标置换共轭**」")
    print("=" * 74)

    # ── B.2 共轭恒等式（★ 恰好 0.0）───────────────────────────────────────
    rng = np.random.default_rng(SEED)
    worst_id = 0.0
    worst_norm = {}
    print()
    print("  ── `I(half(x, pos))` vs `adj(I(x), pos)`（`d = %d`，`pos ∈ %s`）──"
          % (D_SMALL, list(POS_LADDER)))
    for pos in POS_LADDER:
        x = rng.standard_normal(D_SMALL)
        lhs = perm_I(rope_half(x, pos, th_small))
        rhs = rope_adjacent(perm_I(x), pos, th_small)
        d = float(np.abs(lhs - rhs).max())
        worst_id = max(worst_id, d)
        print("    pos=%-7.1f  max|I(half(x,pos)) − adj(I(x),pos)| = %r" % (pos, d))
        # ★ 保范数要**多抽样**才有意义：单个样本上这个误差可能恰好是 0
        #   （舍入项互相抵消），所以这里每个 `pos` 换 `NORM_SAMPLES` 组 `x`。
        for k in range(NORM_SAMPLES):
            xk = rng.standard_normal(D_SMALL)
            for nm, fn in (("adj", rope_adjacent), ("half", rope_half)):
                y = fn(xk, pos, th_small)
                e = abs(float(np.linalg.norm(y)) - float(np.linalg.norm(xk))) \
                    / float(np.linalg.norm(xk))
                worst_norm[nm] = max(worst_norm.get(nm, 0.0), e)
    check("B.2", worst_id == CONJUGATION_EXACT,
          "★★ **共轭恒等式**：差 **恰好 == %.1f**（四个 `pos` 档全过，含非整数）—— "
          "★ 这是**索引恒等式**，**不是**机器精度比较（两边执行**同一批标量运算**，"
          "只是结果摆放的位置不同）⇒ 计划 §4 的 **R16**；"
          "★ **不得**说成「两个不同的运算」或「数值完全不同」（那是 P14 的措辞陷阱）"
          % CONJUGATION_EXACT)
    x_probe = rng.standard_normal(D_SMALL)
    print("    ★ 对照：`perm_I` 与 `perm_Iinv` 互逆 ⇒ max|I⁻¹(I(x)) − x| = %r"
          % float(np.abs(perm_Iinv(perm_I(x_probe)) - x_probe).max()))
    check("B.3b", worst_norm["adj"] < NORM_TOL and worst_norm["half"] < NORM_TOL,
          "★ **各自保范数**（旋转是正交变换）：论文那套 `%.3e`、`rotate_half` 那套 `%.3e`"
          "（**每个 `pos` 换 %d 组 `x`，共 %d 个样本**），两者都 < `%.0e` ⇒ 两套都合法"
          % (worst_norm["adj"], worst_norm["half"], NORM_SAMPLES,
             NORM_SAMPLES * len(POS_LADDER), NORM_TOL))

    # ── B.2b 置换的实质内容：作用到 W_q / W_k 的输出维上 ────────────────────
    print()
    print("  ── B.2b **置换的实质内容（独立路径）**：把它施到 `W_q`/`W_k` 的**行**上 ──")
    rng = np.random.default_rng(SEED_PERM)
    th_perm = freqs(DH_PERM, 10000.0)      # ★ 频率表必须与 `DH_PERM` 配（不借 `D_MAIN` 那张）
    idx = row_perm_index(DH_PERM)
    Wq = rng.standard_normal((DH_PERM, DH_PERM))
    Wk = rng.standard_normal((DH_PERM, DH_PERM))
    X = rng.standard_normal((SEQ_PERM, DH_PERM))
    Qa = np.stack([rope_adjacent(Wq @ X[s], float(s), th_perm) for s in range(SEQ_PERM)])
    Ka = np.stack([rope_adjacent(Wk @ X[s], float(s), th_perm) for s in range(SEQ_PERM)])
    Qb = np.stack([rope_half((Wq[idx] @ X[s]), float(s), th_perm) for s in range(SEQ_PERM)])
    Kb = np.stack([rope_half((Wk[idx] @ X[s]), float(s), th_perm) for s in range(SEQ_PERM)])
    Qc = np.stack([rope_half(Wq @ X[s], float(s), th_perm) for s in range(SEQ_PERM)])
    Kc = np.stack([rope_half(Wk @ X[s], float(s), th_perm) for s in range(SEQ_PERM)])
    Sa = Qa @ Ka.T
    amax = float(np.abs(Sa).max())
    rel_perm = float(np.abs(Sa - Qb @ Kb.T).max()) / amax
    rel_raw = float(np.abs(Sa - Qc @ Kc.T).max()) / amax
    print("    `d_h = %d`、`S = %d`；`max|score| = %.6f`" % (DH_PERM, SEQ_PERM, amax))
    print("    行置换 `idx = %s`" % (idx.tolist(),))
    print("    A vs B（**施**置换）  相对差 = %.3e" % rel_perm)
    print("    A vs B（**不施**置换）相对差 = %.6f" % rel_raw)
    check("B.2b", rel_perm < PERMUTED_SCORE_TOL and rel_raw > UNPERMUTED_SCORE_FLOOR,
          "★★ 把 `I⁻¹` 的行置换施到 `W_q`/`W_k` 上后，两套约定给出的**注意力分数**"
          "逐位相同（相对差 `%.2e` < `%.0e`）；**不施**置换时相对差 `%.4f` > `%.1f` ⇒ "
          "★ 置换**不是**可省的：两套约定的差别**恰好是**这一次坐标置换，"
          "而一个按某套约定训练出来的 checkpoint 直接喂给另一套实现会得到**错的分数**"
          "（这就是 P14 说的「权重不兼容」的**精确形态**：绑定在约定上，可转换，不是两个运算）"
          % (rel_perm, PERMUTED_SCORE_TOL, rel_raw, UNPERMUTED_SCORE_FLOOR))

    # ── B.4 表 B 的引用钉住（X2 / X3 / X4）────────────────────────────────
    print()
    print("  ── B.4 表 B：论文逐字的逐位钉住（X2 / X3 / X4）──")
    quotes_ok = (
        specs.ROPE_FREQ_SET_EQ15 == "10000^{-2(i-1)/d}"
        and specs.ROPE_FREQ_SET_LONGTERM == "10000^{-2i/d}"
        and specs.ROPE_EQ11_REL == "g({\\boldsymbol{x}}_{m},{\\boldsymbol{x}}_{n},m-n)."
        and specs.ROPE_EQ21_REL == "g({\\boldsymbol{x}}_{m},{\\boldsymbol{x}}_{n},n-m),"
        and specs.ROPE_EQ16_ALT ==
        "={\\boldsymbol{x}}^{\\intercal}{\\boldsymbol{W}}_{q}R^{d}_{\\Theta,n-m}"
        and specs.ROPE_EQ16_TYPO_ABSENT ==
        "{\\boldsymbol{x}}_{m}^{\\intercal}{\\boldsymbol{W}}_{q}R^{d}_{\\Theta,n-m}"
    )
    check("B.4", quotes_ok,
          "★ **X2**：论文有**两个**频率版本（`%s` 与 `%s`，下标起点不同、集合相同）"
          "⇒ 引用时必须说明用哪一个；★ **X3**：`m−n`（式 11）与 `n−m`（式 21）"
          "**两处都在原文里**，符号相反 ⇒ 引用必须写明出处；"
          "★ **X4**：式 (16) 末项首因子原文写作 `x^⊤`（**缺下标 `m`**，照抄会维度不符）"
          "—— 本域用「原文形态」与「正确形态」**一对**常量钉住它"
          % (specs.ROPE_FREQ_SET_EQ15, specs.ROPE_FREQ_SET_LONGTERM))
    print("    ★ 本域**自己的实现**固定用 `%s`（`i ∈ [1, d/2]`，"
          "`specs.ROPE_FREQ_INDEX_ORIGIN = %d`），方向固定用 `specs.REL_DIRECTION_INDEX = %d`"
          "（**正数索引**，不是 `±1` 字面量 —— 理由见 §1 的注记）"
          % (specs.ROPE_FREQ_SET_EQ15, specs.ROPE_FREQ_INDEX_ORIGIN,
             specs.REL_DIRECTION_INDEX))


# ===========================================================================
# 5. C 类：边界与反例
# ===========================================================================
def part_C(th):
    print()
    print("=" * 74)
    print("C. 边界与反例")
    print("=" * 74)
    rng = np.random.default_rng(SEED)
    q = rng.standard_normal(D_MAIN)
    k = rng.standard_normal(D_MAIN)
    scale = float(np.linalg.norm(q) * np.linalg.norm(k))
    m_all = np.arange(N_POS, dtype=np.float64)
    qrot = rope_adjacent(q, m_all, th)

    # ── C.1 量法口径：相对极差 vs 尺度无关判据 ─────────────────────────────
    print()
    print("  ── C.1 ★★ **量法口径**：同一份数据、同一个实现，两个统计量差多少？──")
    abs_list, rel_list = [], []
    for delta in range(1, M_SCAN_MAX):
        krot = rope_adjacent(k, m_all + delta, th)
        v = (qrot * krot).sum(axis=1)
        abs_list.append(float(np.abs(v - v[0]).max()) / scale)
        rel_list.append(float((v.max() - v.min()) / abs(v.mean())))
    abs_arr = np.array(abs_list)
    rel_arr = np.array(rel_list)
    print("    `δ = 1..%d`（%d 个点）" % (M_SCAN_MAX - 1, M_SCAN_MAX - 1))
    print("    尺度无关绝对判据：min `%.3e`  max `%.3e`（跨 %.2f 个数量级）"
          % (abs_arr.min(), abs_arr.max(), np.log10(abs_arr.max() / abs_arr.min())))
    print("    相对极差 `(max−min)/|mean|`：min `%.3e`  max `%.3e`（跨 %.2f 个数量级）"
          % (rel_arr.min(), rel_arr.max(), np.log10(rel_arr.max() / rel_arr.min())))
    ratio = float((rel_arr / abs_arr).max())
    print("    两个统计量的比值（逐 δ）：min `%.2f`  max `%.1f`" % ((rel_arr / abs_arr).min(), ratio))
    check("C.1", abs_arr.max() < REL_POS_TOL and ratio > REL_RANGE_RATIO_FLOOR,
          "★★ 同一个实现、同一份数据：改用 `(max−min)/|mean|` 后读数比尺度无关判据"
          "大 `%.0f` 倍（阈值 `%.0f`），而且它**自己**跨 `%.2f` 个数量级 ⇒ "
          "★ 那个统计量**测的是数据与 δ 的选取，不是实现**（计划 §0.1 第一类错误的现场）"
          % (ratio, REL_RANGE_RATIO_FLOOR, np.log10(rel_arr.max() / rel_arr.min())))

    # ── C.2 负对照：q 与 k 的频率表错开一位 ────────────────────────────────
    print()
    print("  ── C.2 ★★ **负对照**：让 `q` 与 `k` 用**不同的频率表**（错开一位）──")
    th_shift = np.concatenate([th[1:], th[-1:]])   # θ 表整体左移一格（**不引入负号**）
    w_ok, w_bad = 0.0, 0.0
    for delta in DELTA_LADDER:
        kr_ok = rope_adjacent(k, m_all + delta, th)
        kr_bad = rope_adjacent(k, m_all + delta, th_shift)
        w_ok = max(w_ok, float(np.abs((qrot * kr_ok).sum(axis=1)
                                      - (qrot * kr_ok).sum(axis=1)[0]).max()) / scale)
        w_bad = max(w_bad, float(np.abs((qrot * kr_bad).sum(axis=1)
                                        - (qrot * kr_bad).sum(axis=1)[0]).max()) / scale)
    print("    同一张频率表（正确）  最差 = %.4e" % w_ok)
    print("    错开一位的频率表      最差 = %.4e" % w_bad)
    check("C.2", w_ok < REL_POS_TOL and w_bad > DIRECTION_MISMATCH_FLOOR,
          "★★ 判据① **不是恒真断言**：频率表错开一位后它从 `%.2e` 掉到 `%.4f` > `%.2f` ⇒ "
          "它真的在测「`q` 与 `k` 用的是同一张频率表」这件事（★ 这条同时是"
          "「输入错了程序必须报错」的证伪证据）" % (w_ok, w_bad, DIRECTION_MISMATCH_FLOOR))

    # ── C.3 方向（符号）是可判定的 ─────────────────────────────────────────
    print()
    print("  ── C.3 方向（`m−n` vs `n−m`）在同一批数据上给出不同的值 ──")
    worst_sep = 0.0
    for delta in DELTA_LADDER:
        v1 = score_with_direction(q, k, float(delta), th, 1)
        v2 = score_with_direction(q, k, float(delta), th, 2)
        worst_sep = max(worst_sep, abs(v1 - v2) / scale)
        print("    δ=%-4d  `m−n` 支 = %+.6f   `n−m` 支 = %+.6f   差/尺度 = %.4e"
              % (delta, v1, v2, abs(v1 - v2) / scale))
    check("C.3", worst_sep > DIRECTION_SEPARATION_FLOOR,
          "★ **方向有后果**：两支在同一批数据上差 `%.4e` > `%.4f` ⇒ "
          "X3（论文内部 `m−n` 与 `n−m` 两处都在）**不是排版问题**："
          "抄错一处就得到一个**不同的函数**（两支各自内部自洽，所以这个错是**静默的**）"
          % (worst_sep, DIRECTION_SEPARATION_FLOOR))
    print("    ★ 本域固定用 `specs.REL_DIRECTION_INDEX = %d`（`m−n`，与式 (11) 一致）"
          % specs.REL_DIRECTION_INDEX)


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

    measured = {"D_MAIN": D_MAIN, "D_SMALL": D_SMALL, "N_POS": N_POS,
                "DELTA_LADDER": DELTA_LADDER, "POS_LADDER": POS_LADDER,
                "M_SCAN_MAX": M_SCAN_MAX, "DH_PERM": DH_PERM, "SEQ_PERM": SEQ_PERM,
                "SEED": SEED, "SEED_PERM": SEED_PERM, "NORM_SAMPLES": NORM_SAMPLES}
    expect = {"D_MAIN": 64, "D_SMALL": 8, "N_POS": 512,
              "DELTA_LADDER": (1, 7, 64, 255, 511), "POS_LADDER": (0.0, 1.3, 5.0, 127.5),
              "M_SCAN_MAX": 512, "DH_PERM": 32, "SEQ_PERM": 6,
              "SEED": 20260922, "SEED_PERM": 11, "NORM_SAMPLES": 8}
    diff = {k: (expect[k], v) for k, v in measured.items() if v != expect[k]}
    check("D.1", not diff,
          "口径常量 11 个逐位一致（★ 改维度/种子/δ 阶梯，本程序的**不变量**断言照样通过 ⇒ "
          "口径只能靠这一条守住）%s" % ("" if not diff else " ← 差异 %s" % diff))

    th = {"REL_POS_TOL": REL_POS_TOL, "REL_POS_VARY_FLOOR": REL_POS_VARY_FLOOR,
          "CONJUGATION_EXACT": CONJUGATION_EXACT, "NORM_TOL": NORM_TOL,
          "PERMUTED_SCORE_TOL": PERMUTED_SCORE_TOL,
          "UNPERMUTED_SCORE_FLOOR": UNPERMUTED_SCORE_FLOOR,
          "REL_RANGE_RATIO_FLOOR": REL_RANGE_RATIO_FLOOR,
          "DIRECTION_MISMATCH_FLOOR": DIRECTION_MISMATCH_FLOOR,
          "DIRECTION_SEPARATION_FLOOR": DIRECTION_SEPARATION_FLOOR}
    th_expect = {"REL_POS_TOL": 1e-12, "REL_POS_VARY_FLOOR": 0.01,
                 "CONJUGATION_EXACT": 0.0, "NORM_TOL": 1e-14,
                 "PERMUTED_SCORE_TOL": 1e-14, "UNPERMUTED_SCORE_FLOOR": 0.1,
                 "REL_RANGE_RATIO_FLOOR": 1000.0, "DIRECTION_MISMATCH_FLOOR": 0.01,
                 "DIRECTION_SEPARATION_FLOOR": 1e-3}
    tdiff = {k: (th_expect[k], v) for k, v in th.items() if v != th_expect[k]}
    check("D.2", not tdiff,
          "阈值 9 个逐位一致（★ 放宽 `REL_POS_TOL` ⇒ 判据① 退化成恒真；"
          "放宽 `CONJUGATION_EXACT` ⇒ 「恰好 0」降级成「差不多」；"
          "`REL_RANGE_RATIO_FLOOR` 是 `C.1` 那条**口径结论**的下界）%s"
          % ("" if not tdiff else " ← 差异 %s" % tdiff))

    # ── 两套约定 / 置换 / 方向的**结构性常量**先钉住 ──────────────────────
    check("D.3", specs.ROPE_FREQ_INDEX_ORIGIN == 1 and specs.REL_DIRECTION_INDEX == 1,
          "★ `specs` 的两条 **[E] 定义**逐位钉住：频率下标起点 `%d`、方向索引 `%d`"
          "（★ 两者都是**正数**：`mutate.py` 的叶子枚举不取 `UnaryOp(USub, …)` ⇒ "
          "写成 `-1` 的表会**整体免检**，见计划 §3 开头的第二个盲区）"
          % (specs.ROPE_FREQ_INDEX_ORIGIN, specs.REL_DIRECTION_INDEX))
    idx8 = row_perm_index(8).tolist()
    print("    `row_perm_index(8) = %s`（`d_h = 8` 时的行置换）" % (idx8,))
    check("D.4", idx8 == [0, 2, 4, 6, 1, 3, 5, 7],
          "★ **行置换本身逐位钉住**：`%s` —— ★ 为什么单列这一条：置换搞反是**静默**的"
          "（`B.2b` 的对照说明反了会差 `%.2f`），所以它的形状必须被钉死"
          % (idx8, UNPERMUTED_SCORE_FLOOR))

    # ── 真实模型的锚点 + 论文逐字 ─────────────────────────────────────────
    check("D.5", specs.LLAMA3_8B_ROPE_THETA == 500000.0
          and specs.QWEN3_30B_A3B_ROPE_THETA == 1000000.0
          and specs.LLAMA3_8B_MAX_POSITION_EMBEDDINGS == 8192
          and specs.QWEN3_30B_A3B_MAX_POSITION_EMBEDDINGS == 40960,
          "★ 两个真实模型的 `rope_theta` 与 `max_position_embeddings` 逐位钉住"
          "（`%.0f`/`%.0f` 与 `%d`/`%d`；**经 `hf-mirror.com` 镜像取得，"
          "未与官方 repo 比对 sha256** —— 这条限定语必须与数字一起引用）"
          % (specs.LLAMA3_8B_ROPE_THETA, specs.QWEN3_30B_A3B_ROPE_THETA,
             specs.LLAMA3_8B_MAX_POSITION_EMBEDDINGS,
             specs.QWEN3_30B_A3B_MAX_POSITION_EMBEDDINGS))
    check("D.6", specs.ROPE_EQ15_PROSE == "is the rotary matrix with pre-defined parameters"
          and specs.ROPE_LONGTERM_DECAY_CLAIM ==
          "One can prove that this setting provides a long-term decay property",
          "★ 论文逐字的逐位钉住：式 (15) 之后那句（**它把「频率集合」与「式 (15)」的关系"
          "钉在一个句子里** —— 计划 §C.3 原稿把两者写成同一个式子，本域按原文分开）；"
          "以及 §3.3 关于谱的唯一一条主张（措辞是 `One can prove`，**不是**实验结果）")
    check("D.7", specs.UNSOURCED == (
        "ntk_aware_primary_source", "rope_extrapolation_weakness_attribution",
        "long_context_quality_metric", "extrapolation_quality_ranking",
        "rope_in_real_attention_kernels"),
          "★ `specs.UNSOURCED` 是**待办清单**（%d 项），**逐位（含顺序）钉住** —— "
          "★ 这是 R17 的落法：**不给它开后门**（不从变异测试的分母里排除），"
          "而是**给它一个消费者**。其中 `ntk_aware_primary_source` 记的是**等级纪律**"
          "（NTK 系列的一手来源是 Reddit/GitHub，**D 级** ⇒ 只能引 YaRN 的转述）"
          % len(specs.UNSOURCED))


def mark_part():
    print()
    print("=" * 74)
    print("D（续）. 失败标记自检 与 测量工具的独立性")
    print("=" * 74)
    print("  本程序的失败标记 ASSERT_FAIL_MARK = %r" % ASSERT_FAIL_MARK)

    hits = _mark_assignments_in(os.path.abspath(__file__))
    uniq = {getattr(v, "value", None) for v in hits}
    check("D.8", len(hits) == 1 and len(uniq) == 1
          and uniq == {ASSERT_FAIL_MARK} and "FAIL" in ASSERT_FAIL_MARK,
          "★ 标记自检：文件里模块级赋值恰好 1 处、取值唯一、== %r、且含 `FAIL`（AST 读出）"
          % ASSERT_FAIL_MARK)
    check("D.9", ASSERT_FAIL_MARK == "$$ FAIL: positional-encoding"
          and "FAIL" in ASSERT_FAIL_MARK and MARKS_USED <= {ASSERT_FAIL_MARK},
          "★ 标记自检（**字面量比较**，R18）：**等于**本主题被分配的字面量、含 `FAIL`，"
          "且打印失败行时用的标记 ⊆ {%r}（本次实际用到 %d 个）"
          % (ASSERT_FAIL_MARK, len(MARKS_USED)))
    check("D.10", OTHER_THEME_MARKS == ("!! FAIL: transformer", "@@ FAIL: attention-variants",
                                        "## FAIL: moe", "~~ FAIL: normalization"),
          "★ 「不跨主题照抄」这条判据的**对照物**也逐位钉住")

    # ── 闭式与两个旋转实现必须**互不调用**（AST 级 + 文本级双证）────────────
    measured_fns = {rope_adjacent, rope_half}
    m_names = {f.__name__ for f in measured_fns}
    ref_fns = {score_closed_form, score_with_direction}
    ref_names = {f.__name__ for f in ref_fns}
    shared_ref = sorted(set().union(*[_callee_names(f) for f in ref_fns]) & m_names)
    cross = sorted(_callee_names(rope_adjacent) & {"rope_half"}
                   | (_callee_names(rope_half) & {"rope_adjacent"}))
    text_ref = "\n".join([_body_text(f) for f in sorted(ref_fns, key=lambda g: g.__name__)])
    textual = not any(s in text_ref for s in ("rope_adjacent(", "rope_half(", "perm_I"))
    src_a = inspect.getsource(rope_adjacent)
    src_h = inspect.getsource(rope_half)
    check("D.11", not shared_ref and not cross and textual and src_a != src_h
          and ref_names == {"score_closed_form", "score_with_direction"},
          "★ **参照实现的独立性**（`CONTRIBUTING` §2 / 域 README）：闭式与两个旋转实现的"
          "调用表**交集 = %s**（⇒ `B.1` 那一路**不构造 `R_m`**，是另一条路）；"
          "两个旋转实现**互不调用**（交集 = %s）；文本级回退 = %s；两者源码不同"
          "（%d vs %d 字节）⇒ `B.2` 的「恰好 0」**不是**同一份代码产出的"
          % (shared_ref, cross, textual, len(src_a), len(src_h)))


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
    print("01_rope_definition.py —— RoPE 的旋转定义、相对位置性质、两套实现约定")
    print("=" * 74)
    print("  ★ 真值标注：本程序的数字**全部是推导值/本机实测值**"
          "（本机 `python3 %s` + `numpy %s`）。"
          % (sys.version.split()[0], np.__version__))
    print("    本机 macOS arm64 **无 GPU**：这些数不支撑任何关于真实模型或 GPU 的结论。")
    print("  ★ 失败标记 ASSERT_FAIL_MARK = %r（本主题自定义；不跨主题照抄）"
          % ASSERT_FAIL_MARK)
    print("  ★★ 本程序的核心（计划 §0.1 的 **P14** 的正面示范）：")
    print("     「RoPE 是什么」（哪两个分量共享一个角度）是**架构**；")
    print("     「哪两个维度配对」（相邻对 vs 前后两半）是**实现的坐标摆放**。")
    print("     后者与前者的关系是**一次固定置换** —— `B.2` 是恒等式、`B.2b` 是它的实质内容。")
    print("  ★ 禁区（计划 §2）：不写「成对通道旋转」那句结构陈述的正文（在")
    print("     `130-model-optimization/quantization/19-kivi-asymmetry.md:184`）；")
    print("     不算位置编码的 FLOPs/访存账；不讲任何外推方法（那是 `03`）。")
    pin_part()
    th_main = freqs(D_MAIN, 10000.0)
    th_small = freqs(D_SMALL, 10000.0)
    part_A(th_main)
    part_B(th_small)
    part_C(th_main)
    mark_part()
    return summary()


if __name__ == "__main__":
    sys.exit(main())

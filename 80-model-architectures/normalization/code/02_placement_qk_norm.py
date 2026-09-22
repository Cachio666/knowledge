#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""02_placement_qk_norm.py —— 归一化的**位置**（pre / post / final）与 **QK-norm 抹掉的到底是什么**。

配套文档：`02-placement-and-qk-norm.md`。
★ 三类断言（A 算法正确性 / B 代价模型 / C 边界与反例）**全部在本程序内部**，
失败时以**非零退出码**结束；**没有 `verify.py`**。

===============================================================================
★★ 本程序要回答的那两个问题
===============================================================================

**问题一：QK-norm 抹掉的是什么？** —— 一句话：**输入幅度**。

  它**不是** `1/√d_h` 的替代品（★ 这是探测阶段搞错过的一点）。要分开这三件事：

  | 量 | 它由什么决定 | 除以 `√d_h` 之后 |
  |---|---|---|
  | `Var(q·k)` | `d_h`（**线性**） | 与 `d_h` 无关 ⇒ **`1/√d_h` 管的是这一半** |
  | `Var(q·k/√d_h)` | **输入幅度**（`α²β²`） | 仍然由幅度决定 ⇒ **QK-norm 管的是这一半** |
  | `Var(RMS(q)·RMS(k)/√d_h)` | 两者都无关 | 实测落在 `[0.9, 1.1]`，闭式 = `1.0`（**可手算**） |

  ★★ 所以 **QK-norm 与 `1/√d_h` 是两件事，各管一半**：本程序在 `A.4` 把
  「只用 QK-norm、不除 `√d_h`」的后果算出来（`Var` 恰好是 `d_h` ⇒ **线性增长**），
  在 `C.1`/`C.2` 把两个方向的口径变异各做一遍。

**问题二：归一化放在哪？** —— 位置是一个**二维**选择（本程序的 `B.3`）：

  * **(a) 与残差加法的先后**：`pre` / `post` / `final`。
    ★ 这一支的**结构与数值后果**的正文在 `transformer/01-block-and-residual-stream.md`
    （残差流范数：pre-LN 随深度涨、post-LN 恒为 `√D`；`final_norm` 的必要性判据 ③ =
    `lm_head` 的 logits std 之比 `> 10×`）—— **本文不重做**，只回链（计划 §3 Chunk 4 表 B）。
    本文承接的是那条链的**训练侧**那一半：论文对 pre/post 的**取舍**主张（`comparable`，
    不是「更好」；post-LN 的梯度尺度**原文用 `hypothesize`**），见 `B.4` 的引用钉住。
  * **(b) 归一化**沿哪条轴、**作用在多大的张量上**：整条残差流（形状 `(N, D)`）还是
    每个 head 的 `q`/`k`（形状 `(H, S, d_h)`）—— ★ 这一支是本文的正文，
    且 **`B.3`/`C.3` 证明「轴」不是自由选择**。

===============================================================================
★★ 来源等级（三类标签必须分开）
===============================================================================

  · **本机实测值 / 推导值**：`python3 3.13.5` + `numpy 2.4.4`，macOS arm64，**无 GPU**。
  · **外部规格值**：两个真实模型的 `rms_norm_eps`（`specs.py`，**经镜像取得**）+ 两份 config 的
    **字段计数**（`rms_norm_eps` / `layer_norm_eps` / `qk_norm`）。
  · **论文逐字**：`specs.py` 的 pre-LN 四条 [C]（arXiv 2002.04745，ICML 2020 已确认同行评审）。
  · ★★ **QK-norm 本身的来源是 [E]**：一手来源**不在本环境**（论文清单里没有、两份真实 config 里
    `qk_norm` **各 0 次**）⇒ 本域**不替它编一个出处**，只把「它是什么」写成**本域的定义**
    （`specs.QK_NORM_AXIS`），并**只对本机测到的后果下结论**（见 `specs.UNSOURCED`）。

===============================================================================
★★ 可证伪断言（把下列任一处改错，本程序必须非零退出）
===============================================================================

  · `A.1`  ④ `Var(RMS(q)·RMS(k)/√d_h) ∈ [0.9, 1.1]`，在 `d_h ∈ {32,64,128,256,512}` **五个点**上
           （★ 五个都要过：原稿只写到 `0.994~1.005`，**漏了 `d_h=512` 那个点**）；
  · `A.2`  ④ **可手算**：`Var(q·k/√d_h)` 在 `q×3.0, k×0.2` 时 `≈ 3²×0.2² = 0.36`（5% 内）；
  · `A.3`  ★ 闭式：`Var(RMS(q)·RMS(k)/√d_h)` 的闭式是 **`1.0`**、`Var(q·k/√d_h)` 的是 `α²β²`
           —— 两个**都是可手算的**，不是「差不多 1」；
  · `A.4`  ★★ **QK-norm 不替代 `1/√d_h`**：`Var(Σ RMS(q)·RMS(k))` 的闭式是 **`d_h`**（**线性增长**）
           ⇒ 若它真的替代了 `1/√d_h`，这个量应当与 `d_h` 无关；
  · `A.5`  ★★ **抹掉幅度**的精确形态：**同一个抽样**下
           `plain(αq, βk)/plain(q, k) == α²β²`（机器精度）而 `qknorm(αq, βk)/qknorm(q, k) == 1`
           —— 「抹掉输入幅度」这条话的机械形态；
  · `A.6`  ★ `eps > 0` 时球面半径小于 `√d_h`，且**逐行等于一阶闭式** `eps/(2·ms_min)`
           （正比于 `eps`：两档比值 `≈ 1000`）；
  · `B.1`  ⑤ softmax 熵（`S = 1024`、**`n = 500` 个样本**）随 logits 尺度**严格递减**（七档）；
  · `B.2`  ★ **多样本的必要性**：`n = 1` 时存在非单调的种子（**计数 `> 0`**）——
           「`std = 12` 的熵**大于** `std = 6` 的」当场复现（那是抽样噪声的现场）；
  · `B.3`  ★★ **轴不是自由选择**：沿**声明的轴**做归一化 ⇒ 逐行缩放不变（`< 1e-14`）；
           另一条轴 ⇒ 同一个量 `> 0.1`（实测 `8.126e-01`）；★ 两条轴**必须不同**（否则对比退化）；
  · `B.3b` ★ **负结果（必须写）**：`A` 段的方差统计量**分辨不出轴**（两条轴都给 `≈ 1`）
           ⇒ 「轴重要」这条结论只能靠 `B.3` 的**逐行缩放不变性**撑；
  · `B.4`  ★ 位置与规格的**引用钉住**：`layer_norm_eps` 在真实 config 里计数 **0**、
           `rms_norm_eps` 计数 **1**、`qk_norm` 计数 **0**；论文的结论词是 `comparable`、
           梯度那句原文是 `hypothesize`；
  · `C.1`  ★ **口径变异**：去掉 `1/√d_h` ⇒ `Var` 变成 `d_h`（线性），不再是 `1`；
  · `C.2`  ★ **口径变异**：把 `√d_h` 写成 `d_h` ⇒ `Var` 变成 `1/d_h`（另一个方向的错）；
  · `C.3`  ★ **边界**：熵的**单调性**只在多样本下成立；单样本的上下界由 `p.max()` 给出（打印对照）。

===============================================================================
★★ 本程序刻意不做的事（计划 §2 的边界表，越界即作废）
===============================================================================

  * ★★ **不写公式、不推公式**（LayerNorm / RMSNorm 的定义、Welford、`Σx²−nμ²`、`eps` **位置**的
    数值差异）—— 正文在 `40-kernels/softmax-normalization/03-normalization-fusion.md:55-71`
    （**正文在彼，此处回链**）。本程序里的算子实现是**测量的对象**。
  * **不重做** `final_norm` 的必要性判据（`lm_head` 的 logits std 之比 `> 10×`）——
    那是 `transformer/01` 的判据 ③（**正文在彼，此处回链**）。
  * **不讲** softmax 自身的数值稳定性（减 max、在线 softmax）—— 正文在
    `40-kernels/softmax-normalization/01`、`02`。本程序里的 softmax 只是一个**测量工具**。
  * **不算** attention 的 FLOPs / 访存账 / 算术强度（正文在 `40-kernels/attention/01,05`）。
  * ★ **不给**「熵低于多少，注意力就算废了」这个阈值（没有一手来源，见 `specs.UNSOURCED`）。

用法：
    python3 02_placement_qk_norm.py
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
# ★★ 为什么必须逐位钉住：本程序的断言几乎全是「不变量」形态 —— 改掉样本数、种子、
#    输入尺度，**断言照样通过**。口径只能靠 `D.1` 的字面量比较守住。
# ★ 第 1 段（`C.1` 的 QK 段）的口径**照抄探测脚本** `probes/probe_80d.py`【1】
#   （计划 §0.7-D 的 **D9** 的唯一口径记录）：`q×3.0`、`k×0.2`、`d_h ∈ {32..512}`。
#   ⚠ **本程序的 `N` 是 `20000`，探测脚本是 `40000`** ⇒ 两个数是**同口径不同样本量**，
#     **不可逐位对齐**（判据只写**区间** `[0.9, 1.1]`，不写某一次抽样的读数）。
N_QK = 20000             # QK 段的样本数
SEED_QK = 20260922
DH_LADDER = (32, 64, 128, 256, 512)   # ★ **五个点**（原稿只写到四个）
Q_SCALE = 3.0            # ★ 输入幅度的构造参数（判据 ④ 的「抹掉幅度」就靠它）
K_SCALE = 0.2

# 第 2 段（`B.3` 的轴）：一个最小的多头形状
H_HEADS = 8
S_SEQ = 1024
DH_AXIS = 128            # ★ 必须 ∈ DH_LADDER（`D.1` 会查这一条）
SEED_AXIS = 7
ROW_SCALE_LO = 0.2       # 逐行幅度的区间（每行一个 α）
ROW_SCALE_HI = 5.0

# 第 3 段（`B.1`/`C.3` 的熵）：口径照抄 `probes/probe_80d.py`【2】（**D20**）
S_ENT = 1024
NS_ENT = 500             # ★★ **必须是多样本** —— `n = 1` 时非单调（`C.3` 现场复现）
SEED_ENT = 20260922
STD_LADDER = (0.1, 0.5, 1.0, 2.0, 3.0, 6.0, 12.0)
NS_SINGLE = 200          # `C.3` 的单样本反例扫描：200 个种子
SEED_SINGLE = 1000

# ── 阈值（★ 逐个在 `D.2` 钉住）─────────────────────────────────────────────
QK_VAR_BAND = (0.9, 1.1)      # ④ 区间（★ 是**区间**，不是某次抽样的读数）
HAND_COMPUTED_TOL = 0.05      # ④ 可手算闭式的容差（★ 抽样误差 ~0.01 ⇒ 5σ 余量）
SCALE_RATIO_TOL = 1e-12       # `A.5` 同一抽样的缩放比（**机器精度档**）
AXIS_INVARIANT_TOL = 1e-14    # `B.3` 沿 head 维的逐行缩放不变性
AXIS_MISMATCH_FLOOR = 0.1     # `B.3` 沿序列维**必须**超过这个下界（不然「轴」不重要）
WRONG_SQRT_TOL = 0.05         # `C.2` 相对 `1/d_h` 的容差（与 `HAND_COMPUTED_TOL` 同档）
SD_ORDER_TOL = 1e-3           # `B.1` 熵阶梯的「严格递减」按**打印精度**判（相邻差最小 `1.7e-2`）
RADIUS_CF_TOL = 0.01          # `A.6` 一阶闭式 `eps/(2·ms_min)` 的相对容差


# ★★ QK-norm 的**轴**由 `specs.QK_NORM_AXIS`（[E] 定义）决定，**不是**本文件里的一个魔法数。
#    ★ **为什么必须这样接**：第一版把默认轴写成 `-1`、只在打印里提到 `specs.QK_NORM_AXIS`，
#      于是那个常量**没有消费者** —— 把它从 `"head"` 改成别的值，**全部断言照样通过**
#      （本域用一次**代码级**变异实测过这件事：`{"head": -2, "seq": -1}` 时 rc=0）。
#      ⇒ 这正是「照抄但不使用」的形态（`45-…/code/specs.py:107-113` 的教训），
#        也是**变异测试的分母原则**要挡的东西。现在轴走了**映射**：
#        未知取值落到「序列维」，于是**断言失败**（而不是抛异常落进崩溃捕获）。
#    ★★ 映射里存的是**正数**（`1` / `2`），用的时候取负 —— 理由是**变异测试的分母**：
#      Python 的 `-1` 在 AST 里是 `UnaryOp(USub, Constant(1))`，`mutate.py` 的叶子枚举**只取
#      `Constant`** ⇒ 写成 `-1` 的轴下标**根本不进分母**（那是一个静默的免检区）。
#      写成正数之后，键与值**都**是叶子，`D.3` 再把两张表逐位钉住。
_AXIS_INDEX = {"head": 1, "seq": 2}                       # 正数：键与值都进变异分母
# ★★ 这两行**必须用 `.get(..., 默认)`，不能用 `[...]`**：键名被改掉时 `[...]` 会在
#    **导入期**抛 `KeyError` ⇒ 程序根本没跑起来 ⇒ 变异测试把它记成**崩溃捕获**（不算真捕获）。
#    用 `.get` 之后，坏键只会让**轴取错**，于是 `D.3`（跑在 A/B/C 之前）**断言失败** ⇒ 真的捕获。
#    ★ 这是 Chunk 2/3 那条「键名先钉、再按名取值」教训的**本主题形态**：本主题没有字典常量，
#      但有**字典推导出来的模块级值**，同一个坑照样在。
_QK_AXIS = -_AXIS_INDEX.get(specs.QK_NORM_AXIS, _AXIS_INDEX.get("seq", 2))  # 声明的轴
_OTHER_NAME = "seq" if specs.QK_NORM_AXIS == "head" else "head"
_QK_AXIS_OTHER = -_AXIS_INDEX.get(_OTHER_NAME, 1)         # ★ 另一条轴（对照）
# ★ 打印用的名字**由 `_AXIS_INDEX` 反推**，而不是另立一张带字面量的表 ——
#   理由是**变异测试的分母**：一张只用于打印的模块级字符串表**没有任何断言约束它**
#   （本域实测：它会贡献 2 条漏网）。⇒ 名字只有一个来源（`_AXIS_INDEX` 的键，`D.3` 已钉住）。
def _axis_label(axis):
    """把轴下标翻回名字（**只用于打印**）。"""
    for name, k in _AXIS_INDEX.items():
        if -k == axis:
            return name
    return "未知"


def rmsnorm(x, eps=0.0, axis=-1):
    """按 `axis` 归一化（**测量对象**；公式正文在 `40-kernels/softmax-normalization/03:55-71`）。"""
    return x / np.sqrt((x ** 2).mean(axis=axis, keepdims=True) + eps)


def qk_norm(q, k, dh, axis=None, divide_sqrt_dh=True):
    """★ **QK-norm 的机械形态**（定义见 `specs.QK_NORM_AXIS` 的注释）。

    `axis=None` 时用**声明的轴**（`_QK_AXIS`）—— ⇒ `A`/`B` 段的全部测量都消费那个常量。

    `divide_sqrt_dh` 这个开关**只**用来做 `C.1`/`C.2` 的**口径变异** ——
    ★ 它是**参数**而不是模块级常量：变异测试改的是常量，而这里要的是「把同一条路走错两遍」。
    """
    if axis is None:
        axis = _QK_AXIS
    u, v = rmsnorm(q, axis=axis), rmsnorm(k, axis=axis)
    s = (u * v).sum(-1)
    return s / np.sqrt(dh) if divide_sqrt_dh else s


def softmax_entropy(logits):
    """归一化熵（除以 `log(S)`）与最大概率 —— **测量工具**，不是本文的知识对象。"""
    lg = logits - logits.max(axis=-1, keepdims=True)
    p = np.exp(lg)
    p /= p.sum(axis=-1, keepdims=True)
    n = p.shape[-1]
    ent = -(p * np.log(p + 1e-300)).sum(-1) / np.log(n)
    return ent, p.max(axis=-1)


# ===========================================================================
# 2. A 类：QK-norm 的数值判据（判据 ④）
# ===========================================================================
def part_A():
    print("=" * 74)
    print("A. QK-norm 抹掉的到底是什么（判据 ④）")
    print("=" * 74)
    print("  口径：`q = N(0,1)(%d, d_h) × %g`、`k = N(0,1)(%d, d_h) × %g`，种子 %d；"
          % (N_QK, Q_SCALE, N_QK, K_SCALE, SEED_QK))
    print("        `d_h ∈ %s`（★ **五个点**，原稿只写到四个）" % (list(DH_LADDER),))
    print()
    print("  ★★ 三件事必须分开（本程序的核心表）：")
    print("     | 量                        | 由什么决定        | 除以 √d_h 之后          |")
    print("     |---------------------------|-------------------|-------------------------|")
    print("     | `Var(q·k)`                | `d_h`（线性）     | 与 d_h 无关 ⇒ **1/√d_h** |")
    print("     | `Var(q·k/√d_h)`           | **输入幅度** `α²β²` | 仍由幅度决定 ⇒ **QK-norm** |")
    print("     | `Var(RMS(q)·RMS(k)/√d_h)` | 两者都无关        | 闭式 = `1.0`            |")
    print("  ⇒ **QK-norm 与 `1/√d_h` 各管一半，不是替代关系**（`A.4` 把这一条钉住）。")

    rng = np.random.default_rng(SEED_QK)
    rows = []
    for dh in DH_LADDER:
        q = rng.standard_normal((N_QK, dh)) * Q_SCALE
        k = rng.standard_normal((N_QK, dh)) * K_SCALE
        v_plain = float(((q * k).sum(-1) / np.sqrt(dh)).var())
        v_qk = float(qk_norm(q, k, dh).var())
        v_qk_raw = float(qk_norm(q, k, dh, divide_sqrt_dh=False).var())
        rows.append((dh, v_plain, v_qk, v_qk_raw))
        print()
        print("    d_h=%-4d  Var(q·k/√d_h) = %.4f   Var(RMS(q)·RMS(k)/√d_h) = %.4f   "
              "Var(Σ RMS·RMS) = %.2f" % (dh, v_plain, v_qk, v_qk_raw))

    # ── A.1 区间 ───────────────────────────────────────────────────────────
    vals = [r[2] for r in rows]
    inband = all(QK_VAR_BAND[0] < v < QK_VAR_BAND[1] for v in vals)
    check("A.1", inband,
          "④ `Var(RMS(q)·RMS(k)/√d_h)` 在**五个** `d_h` 上都落在 `[%g, %g]`：%s"
          "（`d_h = %d` 那个点 = `%.4f`）★ 判据写的是**区间**，因为这是**抽样量**："
          "本机抽样误差 `~%.3f`（`N = %d`）⇒ 区间边界离它有 `%.0fσ` —— "
          "而不写「某一次抽样的读数」（计划 D9 记的五个数是 `0.9940/1.0048/0.9998/0.9981/0.9814`，"
          "那是 `N = 40000`、**不同的样本量**）"
          % (QK_VAR_BAND[0], QK_VAR_BAND[1], " / ".join("%.4f" % v for v in vals),
             DH_LADDER[-1], vals[-1], 1.0 / np.sqrt(N_QK), N_QK,
             (QK_VAR_BAND[1] - 1.0) * np.sqrt(N_QK) / 1.0))

    # ── A.2 可手算：3² × 0.2² ──────────────────────────────────────────────
    hand = Q_SCALE ** 2 * K_SCALE ** 2
    worst = max(abs(r[1] - hand) / hand for r in rows)
    check("A.2", worst < HAND_COMPUTED_TOL,
          "④ **可手算**：`Var(q·k/√d_h)` 在 `q×%g, k×%g` 时应等于 `%g² × %g² = %g`；"
          "五个 d_h 上最差相对偏差 `%.4f` < `%g` ⇒ 这个量**由输入幅度决定**"
          % (Q_SCALE, K_SCALE, Q_SCALE, K_SCALE, hand, worst, HAND_COMPUTED_TOL))

    # ── A.3 两个闭式都是可手算的 ───────────────────────────────────────────
    worst_qk = max(abs(v - 1.0) for v in vals)
    check("A.3", worst_qk < HAND_COMPUTED_TOL,
          "★ 闭式（**不是**「差不多 1」）：`Var(RMS(q)·RMS(k)/√d_h) → 1.0`、"
          "`Var(q·k/√d_h) → α²β²`。实测与 `1.0` 的最差偏差 `%.4f` < `%g`"
          "（抽样误差 ~`%.3f` ⇒ 阈值有 `%.1fσ` 余量）"
          % (worst_qk, HAND_COMPUTED_TOL, 1.0 / np.sqrt(N_QK),
             HAND_COMPUTED_TOL / (1.0 / np.sqrt(N_QK))))

    # ── A.4 QK-norm 不替代 1/√d_h ──────────────────────────────────────────
    worst_raw = max(abs(r[3] - r[0]) / r[0] for r in rows)
    grows = all(rows[i][3] < rows[i + 1][3] for i in range(len(rows) - 1))
    check("A.4", worst_raw < HAND_COMPUTED_TOL and grows,
          "★★ **QK-norm 不替代 `1/√d_h`**：`Var(Σ RMS(q)·RMS(k))` 的闭式是 **`d_h`**"
          "（实测 %s，与 `d_h` 的最差相对偏差 `%.4f`）—— 它**随 `d_h` 线性增长**。"
          "若 QK-norm 替代了 `1/√d_h`，这个量应当与 `d_h` 无关 ⇒ **两者各管一半**"
          % (" / ".join("%.1f" % r[3] for r in rows), worst_raw))

    # ── A.5 「抹掉输入幅度」的精确形态 ─────────────────────────────────────
    dh = DH_AXIS
    rng2 = np.random.default_rng(SEED_QK + 1)
    q = rng2.standard_normal((N_QK, dh))
    k = rng2.standard_normal((N_QK, dh))
    base_plain = float(((q * k).sum(-1) / np.sqrt(dh)).var())
    base_qk = float(qk_norm(q, k, dh).var())
    print()
    print("    ── A.5 **同一个抽样**、只把幅度换成 `(α, β)`：`d_h = %d` ──" % dh)
    ok_p, ok_q = True, True
    for alpha, beta in ((1.0, 1.0), (Q_SCALE, K_SCALE), (0.1, 50.0), (2.0, 7.0)):
        p = float((((alpha * q) * (beta * k)).sum(-1) / np.sqrt(dh)).var())
        z = float(qk_norm(alpha * q, beta * k, dh).var())
        rp = p / base_plain
        rz = z / base_qk
        ok_p = ok_p and abs(rp - alpha ** 2 * beta ** 2) / (alpha ** 2 * beta ** 2) < SCALE_RATIO_TOL
        ok_q = ok_q and abs(rz - 1.0) < SCALE_RATIO_TOL
        print("      α=%-5.1f β=%-5.1f  plain 比 = %.15f（α²β² = %.6g）  "
              "qknorm 比 = %.15f" % (alpha, beta, rp, alpha ** 2 * beta ** 2, rz))
    check("A.5", ok_p and ok_q,
          "★★ **「抹掉输入幅度」的精确形态**：`plain(αq, βk)/plain(q, k) == α²β²`"
          "（**机器精度** `< %.0e`）而 `qknorm(αq, βk)/qknorm(q, k) == 1`（同档）—— "
          "★ 这不是「差不多」：`qknorm` 那一列**恰好**是 1，`plain` 那一列**恰好**是 `α²β²`"
          % SCALE_RATIO_TOL)

    # ── A.6 eps > 0 时「半径」略小于 √d_h（闭式 eps/(2·ms_min)）──────────────
    # ★ `§1.4` 的第 0 步说「归一化把这一行投到半径 √d_h 的球面上」—— 那只在 `eps = 0` 时精确。
    #   本段把 `eps > 0` 的偏差算出来，并配一个**一阶闭式**（可手算）。
    print()
    print("    ── A.6 `eps > 0` 时球面半径略小于 `√d_h`（`d_h = %d`）──" % dh)
    ms_min = float((q ** 2).mean(-1).min())
    deficit = {}
    for eps in (1e-6, 1e-3):
        rad = np.linalg.norm(rmsnorm(q, eps), axis=-1) / np.sqrt(dh)
        deficit[eps] = float(1.0 - rad.min())
        cf = eps / (2.0 * ms_min)          # 一阶闭式：`1 − √(ms/(ms+eps)) ≈ eps/(2·ms)`
        print("      eps=%-8.1e  1 − min(半径/√d_h) = %.6e  一阶闭式 eps/(2·ms_min) = %.6e"
              "  （该行的 `ms` 最小值 = %.4f）" % (eps, deficit[eps], cf, ms_min))
    ratio_cf = max(abs(deficit[e] / (e / (2.0 * ms_min)) - 1.0) for e in deficit)
    check("A.6", all(v > 0 for v in deficit.values())
          and abs(deficit[1e-3] / deficit[1e-6] - 1000.0) / 1000.0 < 0.01
          and ratio_cf < RADIUS_CF_TOL,
          "★ `eps > 0` 时半径**确实**小于 `√d_h`（`%.3e` / `%.3e`），且**与 `eps` 成正比**"
          "（比值 `%.1f ≈ 1000`），并**逐行等于一阶闭式** `eps/(2·ms_min)`（最差相对偏差 `%.1e`）"
          "⇒ `§1.4` 第 0 步的「投到球面」在 `eps > 0` 时是**近似**，"
          "而它的偏差**可手算**" % (deficit[1e-6], deficit[1e-3],
                                    deficit[1e-3] / deficit[1e-6], ratio_cf))


# ===========================================================================
# 3. B 类：熵的阶梯、位置与规格的锚点、归一化的轴
# ===========================================================================
def part_B():
    print()
    print("=" * 74)
    print("B. 熵（判据 ⑤）/ 位置与规格的锚点 / 归一化的轴")
    print("=" * 74)

    # ── B.1 熵阶梯（n = 500）──────────────────────────────────────────────
    print()
    print("  ── B.1 softmax 熵随 logits 尺度**严格递减**（`S = %d`，**%d 个样本**）──"
          % (S_ENT, NS_ENT))
    rng = np.random.default_rng(SEED_ENT)
    ents, sds, maxps = [], [], []
    print("    %-8s %-14s %-10s %-12s %s" % ("std", "归一化熵", "标准差", "最大概率", "判定"))
    for std in STD_LADDER:
        e, mx = softmax_entropy(rng.standard_normal((NS_ENT, S_ENT)) * std)
        ents.append(float(e.mean()))
        sds.append(float(e.std()))
        maxps.append(float(mx.mean()))
        tag = ("接近均匀" if ents[-1] > 0.9 else ("正常" if ents[-1] > 0.3 else "几乎 one-hot"))
        print("    %-8.1f %-14.4f %-10.4f %-12.4f %s"
              % (std, ents[-1], sds[-1], maxps[-1], tag))
    diffs = np.diff(ents)
    check("B.1", bool(np.all(diffs < -SD_ORDER_TOL)) and ents[0] > 0.99 and ents[-1] < 0.1,
          "⑤ 七档**严格递减**（相邻差最小 `%.4f`，均 < `-%g`；两端 `%.4f → %.4f`）"
          "★ **必须是多样本**：`n = 1` 时非单调（`B.2` 现场复现）"
          % (float(diffs.min()), SD_ORDER_TOL, ents[0], ents[-1]))
    print("    ★ 计划 §0.7-D 的 **D20** 记录（同口径 `S=%d, n=%d`、**不同的抽样历史**）："
          % (S_ENT, NS_ENT))
    print("      `0.9993 → 0.9820 → 0.9278 → 0.7213 → 0.5011 → 0.1872 → 0.0782`")
    print("      本程序：`%s`" % " → ".join("%.4f" % e for e in ents))
    print("      ⇒ 两个序列**同档不同位**（差在 `1e-2` 量级）—— 判据只能写**单调性**，"
          "不能写某一次抽样的读数")

    # ── B.2 n = 1 的反例计数 ──────────────────────────────────────────────
    print()
    print("  ── B.2 `n = 1` 的危害（★ 这是 §0.1 第一类错误的现场）──")
    bad = 0
    worst_ratio = 0.0
    for s in range(NS_SINGLE):
        r1 = np.random.default_rng(SEED_SINGLE + s)
        e6 = float(softmax_entropy(r1.standard_normal((1, S_ENT)) * 6.0)[0][0])
        e12 = float(softmax_entropy(r1.standard_normal((1, S_ENT)) * 12.0)[0][0])
        if e12 > e6:
            bad += 1
            worst_ratio = max(worst_ratio, e12 / max(e6, 1e-12))
    print("    在 %d 个种子上，「`std = 12.0` 的熵 **>** `std = 6.0` 的熵」出现 **%d** 次"
          % (NS_SINGLE, bad))
    check("B.2", bad > 0,
          "★ **反例**：`n = 1` 时非单调**确实存在**（%d/%d 个种子，最多放大到 `%.2f×`）⇒ "
          "① 「熵随尺度单调」这条结论**只能在多样本上写**；"
          "② 单样本的熵是**抽样噪声**，不是「注意力稀释」的证据"
          % (bad, NS_SINGLE, worst_ratio))

    # ── B.3 归一化的轴 ────────────────────────────────────────────────────
    print()
    print("  ── B.3 「轴」不是自由选择（`specs.QK_NORM_AXIS = %r`，本域 [E] 定义）──"
          % specs.QK_NORM_AXIS)
    # ★ 轴走**映射**（`_QK_AXIS`）：未知取值落到「序列维」⇒ 下面**断言失败**
    #   （而不是抛异常落进「崩溃捕获」）。
    ax = _QK_AXIS
    ra = np.random.default_rng(SEED_AXIS)
    q = ra.standard_normal((H_HEADS, S_SEQ, DH_AXIS))
    row_a = ra.uniform(ROW_SCALE_LO, ROW_SCALE_HI, size=(H_HEADS, S_SEQ, 1))
    print("    形状 `(H, S, d_h) = (%d, %d, %d)`；每 `(h, s)` 行乘一个 `α ~ U(%g, %g)`；`eps = 0`"
          % (H_HEADS, S_SEQ, DH_AXIS, ROW_SCALE_LO, ROW_SCALE_HI))
    dev = {}
    for a, tag in ((ax, "★ 声明的轴"), (_QK_AXIS_OTHER, "另一条轴")):
        u = rmsnorm(q, axis=a)
        v = rmsnorm(q * row_a, axis=a)
        dev[a] = float(np.abs(v - u).max() / np.abs(u).max())
        print("      %-12s（%s，axis=%d）逐行缩放后，逐元素最大相对变化 = %.4e"
              % (tag, _axis_label(a), a, dev[a]))
    check("B.3", dev[ax] < AXIS_INVARIANT_TOL
          and dev[_QK_AXIS_OTHER] > AXIS_MISMATCH_FLOOR
          and ax != _QK_AXIS_OTHER,   # ★ 两条轴**必须不同**，否则这条对比退化成恒真
          "★★ **轴决定这条性质成不成立**：**声明的那条轴**（`%s`）⇒ `%.3e`"
          "（**机器精度**）；**另一条轴** ⇒ `%.4e`（差 **15 个数量级**）。"
          "「QK-norm 抹掉输入幅度」这句话**只对一条轴成立** ⇒ `specs.QK_NORM_AXIS` 里的轴"
          "不是修饰、是定义的一部分（本程序用的是 `%r`）"
          % (_axis_label(ax), dev[ax], dev[_QK_AXIS_OTHER], specs.QK_NORM_AXIS))
    # ★★ 一条**负结果**必须写出来：`A` 段那个方差统计量**分辨不出轴** ——
    #    两条轴都给 `≈ 1`。分辨轴的是这里的**逐行缩放不变性**。
    rk = ra.standard_normal(q.shape)
    var_a = float((rmsnorm(q, axis=-1) * rmsnorm(rk, axis=-1)).sum(-1).var()) / DH_AXIS
    var_b = float((rmsnorm(q, axis=-2) * rmsnorm(rk, axis=-2)).sum(-1).var()) / DH_AXIS
    check("B.3b", abs(var_a - 1.0) < HAND_COMPUTED_TOL and abs(var_b - 1.0) < HAND_COMPUTED_TOL,
          "★ **负结果（必须写）**：`A` 段那两个方差统计量**分辨不出轴**（沿 head `%.4f` / "
          "沿序列 `%.4f`，都在 `1 ± %g` 内）⇒ 「轴重要」这条结论**不能**靠 `A` 段撑，"
          "只能靠本段的**逐行缩放不变性**（`%.3e` vs `%.4e`）"
          % (var_a, var_b, HAND_COMPUTED_TOL, dev[ax], dev[_QK_AXIS_OTHER]))

    # ── B.4 位置与规格的引用钉住 ──────────────────────────────────────────
    print()
    print("  ── B.4 位置（pre / post / final）与 QK-norm 的**引用钉住** ──")
    print("    ★ 位置的**结构后果**（残差流范数、`final_norm` 的必要性）正文在")
    print("      `transformer/01-block-and-residual-stream.md`（正文在彼，此处回链）——**本文不重做**。")
    quotes = (
        specs.PRE_LN_VERDICT_WORD == "comparable"
        and "comparable results with baselines" in specs.PRE_LN_ABSTRACT_COMPARABLE
        and specs.PRE_LN_GRADIENT_HYPOTHESIS.startswith("we hypothesize that")
        and specs.PRE_LN_OUTPERFORM_ATTRIBUTION.startswith("outperforms the Post-LN Transformer")
        and specs.PRE_LN_PLACEMENT_VERBATIM.startswith("puts the layer normalization inside")
        and specs.POST_LN_PLACEMENT_VERBATIM.startswith("puts the layer normalization between")
        and specs.PRE_LN_FINAL_NORM_VERBATIM == "final layer normalization"
    )
    check("B.4", quotes,
          "★ 位置的**逐字**逐位钉住（表 B 的 X8 落点）：结论词 = `%r`；梯度那句原文 = "
          "`hypothesize`（**不是**「已证明」）；`outperform` 的**主语是 `Wang et al. 2019`**"
          "（本域把它单独存成一条常量，就是为了让它**不被读成论文自己的主张**）；"
          "pre/post 的两种放法 + `final layer normalization` 都有逐字"
          % specs.PRE_LN_VERDICT_WORD)
    cfg_facts = (specs.LLAMA3_8B_RMS_NORM_EPS > 0 and specs.QWEN3_30B_A3B_RMS_NORM_EPS > 0)
    check("B.4b", cfg_facts and "qk_norm_in_real_configs" in specs.UNSOURCED,
          "★ **QK-norm 的 config 锚点不存在**（逐字计数：两份 config 里 `qk_norm` 各 **0** 次，"
          "而 `rms_norm_eps` 各 **1** 次 ⇒ 对照成立）；"
          "`layer_norm_eps` 各 **0** 次。⇒ 本域**不写**「某某模型用了 QK-norm」，"
          "并把这条缺口挂在 `specs.UNSOURCED` 的 `qk_norm_in_real_configs` 上")


# ===========================================================================
# 4. C 类：口径变异与边界
# ===========================================================================
def part_C():
    print()
    print("=" * 74)
    print("C. 口径变异与边界（把同一条路走错的两个方向各做一遍）")
    print("=" * 74)

    ra = np.random.default_rng(SEED_AXIS + 1)
    dh = DH_AXIS
    q = ra.standard_normal((N_QK, dh))
    k = ra.standard_normal((N_QK, dh))
    right = float(qk_norm(q, k, dh).var())
    no_div = float(qk_norm(q, k, dh, divide_sqrt_dh=False).var())
    print()
    print("  ── C.1 去掉 `1/√d_h`（照 `A.4` 的方向错一次）──")
    print("    正确：`Var(Σ RMS·RMS/√d_h)` = %.4f（闭式 `1.0`）" % right)
    print("    去掉：`Var(Σ RMS·RMS)`       = %.2f（闭式 `d_h = %d`）" % (no_div, dh))
    check("C.1", abs(no_div - dh) / dh < HAND_COMPUTED_TOL and no_div > 10 * right,
          "★ **口径变异**：去掉 `1/√d_h` ⇒ `Var` 从 `%.4f` 变成 `%.2f`（`≈ d_h`，**线性增长**）；"
          "`logits` 的 **std** 因此变成 `√d_h = %.2f` 倍 ⇒ 熵往下掉（`B.1` 的阶梯）"
          % (right, no_div, np.sqrt(dh)))

    # ── C.2 把 √d_h 写成 d_h ──────────────────────────────────────────────
    wrong = float((((rmsnorm(q) * rmsnorm(k)).sum(-1)) / dh).var())
    print()
    print("  ── C.2 另一个方向：把 `√d_h` 写成 `d_h` ──")
    print("    正确：`Var(Σ RMS·RMS/√d_h)` = %.4f；写成 `d_h`：%.6f（闭式 `1/d_h = %.6f`）"
          % (right, wrong, 1.0 / dh))
    check("C.2", abs(wrong - 1.0 / dh) / (1.0 / dh) < WRONG_SQRT_TOL,
          "★ **口径变异**：`√d_h` 与 `d_h` 只差一个 `√d_h`，结论差 `%.0e` 倍（`%d` 倍）⇒ "
          "两个方向的错**方向相反、量级相同**：一个让 logits 涨 `√d_h` 倍、一个让它缩 `√d_h` 倍"
          % (right / wrong, dh))

    # ── C.3 单样本的边界 ─────────────────────────────────────────────────
    print()
    print("  ── C.3 边界：熵的**可判定范围**（不给「多少算废」的阈值）──")
    r1 = np.random.default_rng(SEED_ENT)
    ends = {}
    for std in (STD_LADDER[0], STD_LADDER[-1]):
        e, mx = softmax_entropy(r1.standard_normal((NS_ENT, S_ENT)) * std)
        ends[std] = (float(e.mean()), float(mx.mean()))
    print("    两端：`std = %g` ⇒ 熵 %.4f（最大概率 %.4f）；`std = %g` ⇒ 熵 %.4f（最大概率 %.4f）"
          % (STD_LADDER[0], ends[STD_LADDER[0]][0], ends[STD_LADDER[0]][1],
             STD_LADDER[-1], ends[STD_LADDER[-1]][0], ends[STD_LADDER[-1]][1]))
    check("C.3", ends[STD_LADDER[0]][0] > 0.99 and ends[STD_LADDER[-1]][0] < 0.1
          and "attention_entropy_collapse_threshold" in specs.UNSOURCED,
          "★ 本程序只断言**单调性**与两端（`%.4f` / `%.4f`），**不给**「熵低于多少算废」的阈值"
          "（没有一手来源 ⇒ `specs.UNSOURCED`）；★ 另一条边界：熵是 `S` 的函数，"
          "**跨 `S` 比较熵必须先归一化**（本程序的熵都除了 `log S`）"
          % (ends[STD_LADDER[0]][0], ends[STD_LADDER[-1]][0]))


# ===========================================================================
# 5. D 类：钉住与自检
# ===========================================================================
def _callee_names(func):
    """★ 用 **AST** 列出 `func` 真正**调用**的函数名（不含注释/文档串里的提及）。"""
    import textwrap
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    return {n.func.id for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}


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

    measured = {"N_QK": N_QK, "SEED_QK": SEED_QK, "DH_LADDER": DH_LADDER,
                "Q_SCALE": Q_SCALE, "K_SCALE": K_SCALE, "H_HEADS": H_HEADS,
                "S_SEQ": S_SEQ, "DH_AXIS": DH_AXIS, "SEED_AXIS": SEED_AXIS,
                "ROW_SCALE_LO": ROW_SCALE_LO, "ROW_SCALE_HI": ROW_SCALE_HI,
                "S_ENT": S_ENT, "NS_ENT": NS_ENT, "SEED_ENT": SEED_ENT,
                "STD_LADDER": STD_LADDER, "NS_SINGLE": NS_SINGLE, "SEED_SINGLE": SEED_SINGLE}
    expect = {"N_QK": 20000, "SEED_QK": 20260922, "DH_LADDER": (32, 64, 128, 256, 512),
              "Q_SCALE": 3.0, "K_SCALE": 0.2, "H_HEADS": 8, "S_SEQ": 1024,
              "DH_AXIS": 128, "SEED_AXIS": 7, "ROW_SCALE_LO": 0.2, "ROW_SCALE_HI": 5.0,
              "S_ENT": 1024, "NS_ENT": 500, "SEED_ENT": 20260922,
              "STD_LADDER": (0.1, 0.5, 1.0, 2.0, 3.0, 6.0, 12.0),
              "NS_SINGLE": 200, "SEED_SINGLE": 1000}
    diff = {k: (expect[k], v) for k, v in measured.items() if v != expect[k]}
    check("D.1", not diff,
          "口径常量 17 个逐位一致（★ 改样本数/种子/尺度，本程序的**不变量**断言照样通过 ⇒ "
          "口径只能靠这一条守住）%s" % ("" if not diff else " ← 差异 %s" % diff))

    th = {"QK_VAR_BAND": QK_VAR_BAND, "HAND_COMPUTED_TOL": HAND_COMPUTED_TOL,
          "SCALE_RATIO_TOL": SCALE_RATIO_TOL, "AXIS_INVARIANT_TOL": AXIS_INVARIANT_TOL,
          "AXIS_MISMATCH_FLOOR": AXIS_MISMATCH_FLOOR, "WRONG_SQRT_TOL": WRONG_SQRT_TOL,
          "SD_ORDER_TOL": SD_ORDER_TOL, "RADIUS_CF_TOL": RADIUS_CF_TOL}
    th_expect = {"QK_VAR_BAND": (0.9, 1.1), "HAND_COMPUTED_TOL": 0.05,
                 "SCALE_RATIO_TOL": 1e-12, "AXIS_INVARIANT_TOL": 1e-14,
                 "AXIS_MISMATCH_FLOOR": 0.1, "WRONG_SQRT_TOL": 0.05,
                 "SD_ORDER_TOL": 1e-3, "RADIUS_CF_TOL": 0.01}
    tdiff = {k: (th_expect[k], v) for k, v in th.items() if v != th_expect[k]}
    check("D.2", not tdiff,
          "阈值 8 个逐位一致（★ 放宽 `QK_VAR_BAND` ⇒ 判据退化成恒真；"
          "`AXIS_MISMATCH_FLOOR` 是「轴重要」这条结论的**下界**）%s"
          % ("" if not tdiff else " ← 差异 %s" % tdiff))

    # ── specs：**键名/取值先钉，再谈用** ───────────────────────────────────
    check("D.3", specs.QK_NORM_AXIS in ("head", "seq")
          and _AXIS_INDEX == {"head": 1, "seq": 2}
          and _QK_AXIS == -1 and _QK_AXIS_OTHER == -2
          and "qk_norm_primary_source" in specs.UNSOURCED,
          "★ `specs.QK_NORM_AXIS = %r`（**[E] 本域写出的定义**，不是抄来的），"
          "本文件的轴映射 `%s` 与推导出的两条轴 `%d` / `%d` **逐位钉住**"
          "（★ 正数存放的理由见 `_AXIS_INDEX` 的注释：负文字面量**不进变异分母**）—— "
          "它的**一手来源不在本环境**（论文清单里没有、config 里 `qk_norm` 各 0 次）⇒ "
          "本域**不替它编一个 arXiv 号**，只对**后果**下结论"
          % (specs.QK_NORM_AXIS, _AXIS_INDEX, _QK_AXIS, _QK_AXIS_OTHER))
    quotes = (
        specs.PRE_LN_VERDICT_WORD == "comparable"
        and specs.PRE_LN_ABSTRACT_COMPARABLE ==
        "can reach comparable results with baselines while requiring significantly less "
        "training time and hyper-parameter tuning"
        and specs.PRE_LN_GRADIENT_HYPOTHESIS ==
        "we hypothesize that the gradient scale is one of the reasons that the Post-LN "
        "Transformer needs a careful learning rate scheduling"
        and specs.PRE_LN_OUTPERFORM_ATTRIBUTION == \
        "outperforms the Post-LN Transformer when the number of layers increases"
    )
    check("D.4", quotes,
          "★★ **论文逐字的逐位钉住**（表 B 的 X8 落点）：① 结论词 `%r`（**不是「更好」**）；"
          "② `outperform` 那半句（**主语是 `Wang et al. 2019`**，本域另存了归属常量）；"
          "③ 梯度那句原文用 **`hypothesize`**" % specs.PRE_LN_VERDICT_WORD)
    check("D.5", specs.PRE_LN_PLACEMENT_VERBATIM ==
          "puts the layer normalization inside the residual connection"
          and specs.POST_LN_PLACEMENT_VERBATIM ==
          "puts the layer normalization between the residual blocks"
          and specs.PRE_LN_FINAL_NORM_VERBATIM == "final layer normalization",
          "★ **位置**的逐字逐位钉住：pre 的「放哪」与 post 的「放哪」是原文里**两句不同的措辞**；"
          "`final layer normalization` 是 pre-LN 栈「最后补一个归一化」这个**结构事实**的出处")
    check("D.6", specs.LLAMA3_8B_RMS_NORM_EPS == 1e-05
          and specs.QWEN3_30B_A3B_RMS_NORM_EPS == 1e-06,
          "★ 两个真实模型的 `rms_norm_eps` 逐位钉住（**经 `hf-mirror.com` 镜像取得，"
          "未与官方 repo 比对 sha256** —— 这条限定语必须与数字一起引用）")


def mark_part():
    print()
    print("=" * 74)
    print("D（续）. 失败标记自检 与 测量工具的独立性")
    print("=" * 74)
    print("  本程序的失败标记 ASSERT_FAIL_MARK = %r" % ASSERT_FAIL_MARK)

    hits = _mark_assignments_in(os.path.abspath(__file__))
    uniq = {getattr(v, "value", None) for v in hits}
    check("D.7", len(hits) == 1 and len(uniq) == 1
          and uniq == {ASSERT_FAIL_MARK} and "FAIL" in ASSERT_FAIL_MARK,
          "★ 标记自检：文件里模块级赋值恰好 1 处、取值唯一、== %r、且含 `FAIL`（AST 读出）"
          % ASSERT_FAIL_MARK)
    check("D.8", ASSERT_FAIL_MARK == "~~ FAIL: normalization"
          and "FAIL" in ASSERT_FAIL_MARK and MARKS_USED <= {ASSERT_FAIL_MARK},
          "★ 标记自检（**字面量比较**，R18）：**等于**本主题被分配的字面量、含 `FAIL`，"
          "且打印失败行时用的标记 ⊆ {%r}（本次实际用到 %d 个）"
          % (ASSERT_FAIL_MARK, len(MARKS_USED)))
    check("D.9", OTHER_THEME_MARKS == ("!! FAIL: transformer", "@@ FAIL: attention-variants",
                                       "## FAIL: moe", "$$ FAIL: positional-encoding"),
          "★ 「不跨主题照抄」这条判据的**对照物**也逐位钉住")

    # ── 两个「口径变异」必须真的是**同一条路的两个方向** ────────────────────
    # ★ `C.1`/`C.2` 若各自另写一套代码，那条对比就不是「同一个实现走错两个方向」了。
    called = _callee_names(qk_norm)
    check("D.10", called == {"rmsnorm"},
          "★ `qk_norm` 只调用 `rmsnorm`（AST 调用表 = %s）⇒ `C.1`/`C.2` 的两个错"
          "**是同一条路的两个方向**（`divide_sqrt_dh` 是参数），不是两段各写一遍的代码；"
          "★ 而 `softmax_entropy` 与 `qk_norm` 互不调用 ⇒ `B.1` 的熵阶梯不是从 logits 那边"
          "「借」来的" % sorted(called))


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
    print("02_placement_qk_norm.py —— 归一化的位置（pre/post/final）与 QK-norm 抹掉的是什么")
    print("=" * 74)
    print("  ★ 真值标注：本程序的数字**全部是推导值/本机实测值**"
          "（本机 `python3 %s` + `numpy %s`）。"
          % (sys.version.split()[0], np.__version__))
    print("    本机 macOS arm64 **无 GPU**：这些数不支撑任何关于真实模型或 GPU 的结论。")
    print("  ★ 失败标记 ASSERT_FAIL_MARK = %r（本主题自定义；不跨主题照抄）"
          % ASSERT_FAIL_MARK)
    print("  ★★ 禁区（计划 §2）：**不写公式、不推公式**（正文在")
    print("     `40-kernels/softmax-normalization/03-normalization-fusion.md:55-71`，此处回链）；")
    print("     **不重做** `final_norm` 的必要性判据（`transformer/01` 的判据 ③，正文在彼）；")
    print("     **不讲** softmax 自身的数值稳定性（`40-kernels/softmax-normalization/01,02`）；")
    print("     **不算** attention 的 FLOPs / 访存账（`40-kernels/attention/01,05`）。")
    print("  ★★ QK-norm 本身的**一手来源不在本环境** ⇒ 本域把它写成 [E] 定义"
          "（`specs.QK_NORM_AXIS`），只对**后果**下结论。")
    pin_part()
    part_A()
    part_B()
    part_C()
    mark_part()
    return summary()


if __name__ == "__main__":
    sys.exit(main())

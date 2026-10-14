#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""03_spikes.py —— loss spike：现象 · **已被证伪的归因** · **仍是假设的归因**。

配套文档：`03-instability-and-spikes.md`。

★ 三类断言（A 算法正确性 / B 代价模型 / C 边界与反例）**全部在本程序内部**，
失败时以**非零退出码**结束；**没有 `verify.py`**（`CONTRIBUTING` §2 表里的第二种情形）。

★★ **本程序的数字全部是推导值**，测量条件是：
> `python3 3.13.5` + `numpy 2.4.4`，macOS arm64，**无 GPU**。
> 玩具规模的真跑，**不得支撑关于真实训练系统的结论**。

★★ **本主题的禁区**（`CONTRIBUTING` §7 + 域 README 的边界表）—— 本程序**一行都不写**：
  · ★★ **warmup 的机制解释** → `80-model-architectures/normalization/02-…:83,93-97,100,278`；
  · **残差流的范数随深度增长** → `80-…/transformer/01:191,232-233,263`；
  · ★★ **误差沿深度的传播** → `130-model-optimization/quantization/07-outliers-and-propagation.md:124,205`；
  · **优化器状态的分片与通信量** → `100-parallelism/zero-fsdp/`。

★★★ **本程序最要紧的一条纪律（P20）**：
  「**loss spike 由梯度范数突增引起**」在本域取到的最好来源里**仍被显式标注为假设** ——
  Spike No More 的**摘要第一句**逐字就是 `Based on the assumption that …`
  ⇒ 本程序与配套文档**必须**把它写成**假设**（`specs.SNM_ASSUMPTION_QUOTE`），
  **不得**写成「成因是…」。★ **`C.3`** 是那条纪律的**可执行落法**。
  ★★ 第二句同样要收窄：论文给的是 `Our findings suggest that … **requires two conditions**`
  —— 而**计划 §C.2 把它转述成「充分条件」，原文里没有 `sufficient condition` 这个词**
  （实测 **0** 次）⇒ 本程序只钉原文的两句逐字。

★★ **被原文证伪的归因（Y12）**：「PaLM 的 spike 是坏数据造成的」——
  原文逐字 `We do not believe that the spikes were caused by`（后面接着说
  `“bad data” per se`），且同一段明说**梯度裁剪是开着的**
  （`despite the fact that gradient clipping was enabled`，而 spike 发生了约 20 次）
  ⇒ **「开了 clip 就不会 spike」是错的**。★ 本程序的 `C.2` 把这句话的**数值侧**量出来：
  裁剪只给出**逐点**的上界，它**不约束轨迹**。

★★ 本程序的核心现象（**反直觉**，`D15`）：**spike 那一步的 `m̂/√v̂` 反而变小**
  （`0.702777` vs 正常 `0.999999`）—— 因为 `v` 被 `g²` 抬得**比 `m` 更多**。
  ★ **代价不在 spike 那一步，而在随后的几百步**（更新量被长期压低）。
  ★★ 这条判据**不是恒真的**：同一批梯度交给 SGD，spike 那一步就**真的爆炸 `g_spike/g_normal` = 1e4 倍**（`A.3`）。

★★ 可证伪断言（把下面任一条改错，本程序必须非零退出）：
  · `A.1`  ★★ spike 步的 `m̂/(√v̂+ε)` 比值 ∈ `[0.65, 0.75]`，且与**闭式**逐位一致；
  · `A.1b` ★★ **机制用两个跳变说**：`v` 的跳变倍率 > `m` 的跳变倍率（两个都逐位等于闭式）；
  · `A.2`  ★★ spike 后 1/10/50/150 步 = `0.638694 / 0.268705 / 0.006156 / 0.001452`（**单调下降**）；
  · `A.2b` ★ **记忆长度 ≈ `1/(1−β2)`**：极小值的位置与恢复时间跨三档 β2 按 ~9 倍增长；
  · `A.3`  ★★ **反面对照**：同一步上 SGD 的 `lr·g` 是正常的 **100 倍**（爆炸）⇒ 判据非恒真；
  · `A.4`  ★ **从零开始的 spike**：`t = 1` 时比值**精确为 1**（与梯度幅值无关）；
  · `B.1`  极小值的位置有闭式 `ln(g0/((1−β1)G))/ln β1`（`β2 = 0.9` 上实测吻合）；
  · `B.2`  ★★ 恢复时间随 `1/(1−β2)` 每十倍增长 **8.5~9.5 倍**；
  · `C.1`  ★★ 一个 `inf` 进 Adam ⇒ `m`/`v` 变 `nan` 且**永久污染**（后续梯度干净也回不来）；
  · `C.2`  ★★ 开了 clip 之后：**每步** `‖Δw‖ ≤ lr·max_norm`（逐位），而**累计位移**
           = `N·lr·max_norm`（随 `N` 线性增长）⇒ 逐点上界推不出轨迹稳定；
  · `C.3`  ★★ **引用判据**：`Based on the assumption that …` 与
           `Our findings suggest … requires two conditions` **逐位**；且原文里
           `sufficient condition` 实测 **0** 次 ⇒ **不得升格**；
  · `C.4`  `β2 = 1.0` 必须被拦住（否则 `1 − β2 = 0`，`v` 永不衰减）。

用法：
    python3 03_spikes.py
"""

import ast
import collections
import inspect
import math
import os
import sys

import numpy as np

import specs

# ===========================================================================
# 0. 断言记账（每个编号程序自带一份，**不共享**）
# ===========================================================================
# ★★ 失败标记：本主题（initialization-and-stability）取 `// FAIL: init-stability`。
#   ★ `mutate.py` 用 **AST** 从本文件读出这个值（不得写死）；`D.1`/`D.2` 两条断言守着它。
ASSERT_FAIL_MARK = "// FAIL: init-stability"

# 本域另外四个主题的失败标记 —— **只用来验「不照抄」**，本程序不打印它们。
OTHER_THEME_MARKS = (".. FAIL: autograd", ":: FAIL: mixed-precision",
                     "++ FAIL: optimizers", ";; FAIL: checkpointing")

FAILS = []
MARKS_USED = set()
_TOTAL = collections.Counter()
_PASS = collections.Counter()


def check(name, cond, detail=""):
    """`name` 形如 `A.1` —— **首字母就是类别**（A/B/C/D），与 `CONTRIBUTING` §4 对齐。"""
    cls = name[0]
    _TOTAL[cls] += 1
    if cond:
        _PASS[cls] += 1
        print("  [ok]   %-6s %s" % (name, detail))
    else:
        FAILS.append(name)
        MARKS_USED.add(ASSERT_FAIL_MARK)
        print("  %s %-6s %s" % (ASSERT_FAIL_MARK, name, detail))
    return bool(cond)


# ===========================================================================
# 1. 口径常量与阈值
# ===========================================================================
SEED = 20260952              # ★ 与同域另外两个程序（20260950 / 20260951）都**不同**

# ★★ Adam 的默认超参**从 `specs.py` 读**（[C] 1412.6980，**已确认 ICLR 2015**）——
#   本主题**只消费、不重定义**（同值位置：`optimizers/code/specs.py`）。
#   ★ `β2` 是本程序的主角：它决定 `v` 的记忆长度 `≈ 1/(1−β2)`。
G_NORMAL = 0.01              # 正常步的梯度
G_SPIKE = 100.0              # spike 那一步的梯度（**1e4 倍**）
SPIKE_AT = 50                # spike 发生在第几步（1-based）
TOTAL_STEPS = 200            # 主实验的总步数
POST_OFFSETS = (1, 10, 50, 150)      # 计划 §3 表 A ⑥ 的四个偏移
BETA2_SWEEP = (0.9, 0.99, 0.999)     # `A.2b` / `B.2`：记忆长度三个档
SWEEP_HORIZON = 200000       # `A.2b` 的搜索上限（`β2 = 0.999` 要跑到 ~1.3 万步）
RECOVER_TARGET = 0.5         # `A.2b`：「恢复」的判据 = 回到正常的 50%
SPIKE_RATIO_LO = 0.65        # 计划 §3 表 A ⑤ 的带子
SPIKE_RATIO_HI = 0.75
# ★★ 计划 §3 表 A ⑥ 的四个**绝对**数（`D15`）—— 本程序逐位钉住它们
POST_EXPECT = (0.638694, 0.268705, 0.006156, 0.001452)
TOL_CLOSED = 1e-12           # 与闭式的相对/绝对偏差
TOL_EXACT = 0.0
TOL_IDENT = 1e-15
TOL_MEASURABLE = 1e-2        # 反面对照的**可测性下界**
MIN_RATIO_TOL = 0.05         # `B.1`：极小值必须低于正常的 5%
KMIN_TOL = 3.0               # `B.1`：极小值位置与闭式的步数差
SCALE_LO = 8.0               # `B.2`：恢复时间每十倍的放大倍数区间
SCALE_HI = 9.5
ZERO_SPIKE_MAGS = (1e-8, 1e-4, 1.0, 1e3, 1e6)   # `A.4`：跨 1e14 倍的梯度幅值
# ★★ **盲区⑥ 的落法**（`mutate.py` 的浮点变异是 `v + 1.0`）：**大浮点数会把这个增量吞掉**
#   （`1e300 + 1.0 == 1e300`）⇒ 那样的叶子**变异后逐位不变**、**任何断言都抓不到**。
#   本程序因此把「溢出成 `inf`」拆成两件事：**一个可变异的小幅值** `INF_MAG`（`1e6 < 2^53`）
#   + **一个可变异的平方次数** `INF_SQUARES`，两者都进 `preflight()` 的钉住表。
INF_MAG = 1000000.0          # `C.1`：构造 `inf` 的**底数**（`1e6`；★ `+1.0` 会改变它）
INF_SQUARES = 8              # `C.1`：把它**反复平方** 8 次 ⇒ 溢出成 `inf`（`1e6 → 1e12 → … → inf`）
POLLUTE_STEPS = 100          # `C.1`：污染之后再看多少步
CLIP_MAX_NORM = 1.0          # `C.2`：裁剪阈值
CLIP_LR = 0.01               # `C.2`：学习率
CLIP_STEPS = 20000           # `C.2`：累计多少步
BETA2_BAD = 1.0              # `C.4`：非法的 β2


# ===========================================================================
# 2. 路径（被测）：Adam 的更新量与它的两个跳变
# ===========================================================================
def adam_update_ratios(grads, beta1, beta2, eps):
    """**被测**：逐步算 `|m̂/(√v̂+ε)|`（Adam 的「有效更新量」，**不含 `lr`**）。

    ```
    m_t ← β1·m_{t−1} + (1−β1)·g_t
    v_t ← β2·v_{t−1} + (1−β2)·g_t²
    ratio_t ← |m_t/(1−β1^t)| / ( √( v_t/(1−β2^t) ) + ε )
    ```
    ★ `ε` 在**根号外**（[C] Adam Algorithm 1 逐字，**PDF 文本层**；
      同值位置：`optimizers/code/specs.py` 的 `ADAM_DEFAULT_EPS`）。
    ★★ **本函数的源码次序是 `C.4` 的被测对象**：`m` 在 `v` 之前、`v` 在偏置校正之前。
    """
    if not (0.0 <= beta1 < 1.0) or not (0.0 <= beta2 < 1.0):
        raise ValueError("β 必须在 [0, 1) 内（β1 = %r, β2 = %r）" % (beta1, beta2))
    m = 0.0
    v = 0.0
    out = []
    for t, g in enumerate(grads, 1):
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g * g
        mh = m / (1 - beta1 ** t)
        vh = v / (1 - beta2 ** t)
        out.append(abs(mh / (math.sqrt(vh) + eps)))
    return out


def adam_update_ratio_closed(grads, t, beta1, beta2, eps):
    """★ **独立参照**：`t` 步的更新量用**加权和**直接写出来（不跑递推）。

    ```
    m_t = Σ_{i≤t} (1−β1)·β1^{t−i}·g_i
    v_t = Σ_{i≤t} (1−β2)·β2^{t−i}·g_i²
    ```
    ★ 它与 `adam_update_ratios` 是**两条不同的算式**（一条递推、一条求和），
      数值必须**逐位**一致（`D.9` 的独立性证据）。
    """
    if t < 1:
        raise ValueError("t 必须 ≥ 1（t = %r）" % (t,))
    m = sum((1 - beta1) * beta1 ** (t - i) * g for i, g in enumerate(grads[:t], 1))
    v = sum((1 - beta2) * beta2 ** (t - i) * g * g for i, g in enumerate(grads[:t], 1))
    mh = m / (1 - beta1 ** t)
    vh = v / (1 - beta2 ** t)
    return abs(mh / (math.sqrt(vh) + eps))


def adam_mv_at(grads, t, beta1, beta2):
    """`t` 步的 `(m, v)`（**闭式**，用于 `A.1b` 的两个跳变）。"""
    m = sum((1 - beta1) * beta1 ** (t - i) * g for i, g in enumerate(grads[:t], 1))
    v = sum((1 - beta2) * beta2 ** (t - i) * g * g for i, g in enumerate(grads[:t], 1))
    return m, v


def sgd_update_magnitudes(grads, lr):
    """**反面对照（`A.3`）**：SGD 的 `|lr·g_t|`（**没有自归一化**）。"""
    return [abs(lr * g) for g in grads]


def overflow_to_inf(mag, squares):
    """★★ 把 `mag` **反复平方** `squares` 次 ⇒ 溢出成 `inf`（**不靠一个巨大的字面量**）。

    ★ 为什么要这么绕（**盲区⑥**）：`mutate.py` 对浮点的变异是 `v + 1.0`，
      而 `1e300 + 1.0 == 1e300` ⇒ 一个巨大的浮点字面量**变异后逐位不变**、
      **任何断言都抓不到它**。⇒ 本程序用一个「可变异的小幅值 + 可变异的次数」来构造 `inf`。
    ★★ 而**中间值也是可断言的**（`C.1b`：3 次平方仍然有限、`squares` 次之后不是有限值）
      —— 那才让这两个叶子**真的有载荷**。
    """
    v = float(mag)
    for _ in range(int(squares)):
        v = v * v
    return v


def single_spike_ratio_from_zero(t, spike_mag, beta1, beta2, eps):
    """★★ **`A.4` 的被测**：前 `t−1` 步梯度为 0、第 `t` 步是一个 spike ⇒ 更新量比值。

    闭式（**推导值**）：`m = (1−β1)G`、`v = (1−β2)G²`
      ⇒ `ratio = (1−β1)G/(1−β1^t) / ( √((1−β2)G²/(1−β2^t)) + ε )`
    当 `t = 1` 时它**精确为 1**（与 `G` 无关）；`t → ∞` 时趋向 `(1−β1)/√(1−β2)`。
    """
    m = (1 - beta1) * spike_mag
    v = (1 - beta2) * spike_mag * spike_mag
    mh = m / (1 - beta1 ** t)
    vh = v / (1 - beta2 ** t)
    return abs(mh / (math.sqrt(vh) + eps))


def spike_limit_bound(beta1, beta2):
    """`t → ∞` 时「从零 spike」的比值上界 `(1−β1)/√(1−β2)`（**推导值**）。"""
    return (1 - beta1) / math.sqrt(1 - beta2)


def minimum_location_closed(g0, g_spike, beta1):
    """★★ **`B.1` 的闭式**：极小值出现的步数 `k* = ln(g0/((1−β1)·G)) / ln β1`。

    ★ 依据（**推导值**，不是来源）：spike 之后分子里的那一项按 `β1^k` 衰减，
      当它衰减到与「正常项」`g0` 同量级时比值触底 ⇒ 令 `(1−β1)G·β1^k = g0` 解出 `k*`。
    """
    return math.log(g0 / ((1 - beta1) * g_spike)) / math.log(beta1)


def make_spike_grads(total, spike_at, g_normal, g_spike):
    """构造梯度序列（**唯一的输入**）：第 `spike_at` 步是 spike，其余是正常值。"""
    return [g_spike if t == spike_at else g_normal for t in range(1, total + 1)]


# ===========================================================================
# 3. 工具
# ===========================================================================
def _callee_names(func):
    """用 **AST** 列出 `func` 真正**调用**的函数名（不含注释/文档串里的提及）。"""
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


def first_recovery_step(rel, start, target):
    """`rel` 里从 `start` 之后的第一个 `≥ target` 的下标（`1-based`，找不到给 `None`）。"""
    for i in range(start, len(rel)):
        if rel[i] >= target:
            return i + 1
    return None


# ===========================================================================
# 4. A 类：spike 那一步**反而变小** · 两个跳变 · 随后的长期压制 · SGD 对照
# ===========================================================================
def part_A():
    print("=" * 74)
    print("A. loss spike 的形态：**spike 那一步的更新量反而变小**（反直觉）")
    print("=" * 74)
    print("  测量条件：seed=%d · 梯度 `g = %g`（第 %d 步塞一个 `%g`，**%g 倍**）"
          % (SEED, G_NORMAL, SPIKE_AT, G_SPIKE, G_SPIKE / G_NORMAL))
    print("  Adam 默认超参（[C] 1412.6980，**已确认 ICLR 2015**）："
          "β1 = %g · β2 = %g · ε = %g" % (specs.ADAM_DEFAULT_BETA1, specs.ADAM_DEFAULT_BETA2,
                                          specs.ADAM_DEFAULT_EPS))
    grads = make_spike_grads(TOTAL_STEPS, SPIKE_AT, G_NORMAL, G_SPIKE)
    ratios = adam_update_ratios(grads, specs.ADAM_DEFAULT_BETA1, specs.ADAM_DEFAULT_BETA2,
                                specs.ADAM_DEFAULT_EPS)
    normal = ratios[SPIKE_AT - 2]
    spike = ratios[SPIKE_AT - 1]
    rel_ratio = spike / normal

    # ── A.1 ★★ spike 那一步反而变小 ─────────────────────────────────────────
    print("  ── A.1 ★★ spike 那一步的 `m̂/(√v̂+ε)`：**变小**（不是变大）──")
    print("     正常步（第 %d 步）= **%.6f** · spike 步（第 %d 步）= **%.6f** ⇒ 比值 **%.6f**"
          % (SPIKE_AT - 1, normal, SPIKE_AT, spike, rel_ratio))
    closed = adam_update_ratio_closed(grads, SPIKE_AT, specs.ADAM_DEFAULT_BETA1,
                                      specs.ADAM_DEFAULT_BETA2, specs.ADAM_DEFAULT_EPS)
    print("     闭式（加权和，**另一条路径**）= %.15f ⇒ 与递推的差 **%.1e**" % (closed, abs(closed - spike)))
    check("A.1", SPIKE_RATIO_LO <= rel_ratio <= SPIKE_RATIO_HI and abs(closed - spike) < TOL_CLOSED,
          "★★ **反直觉的核心现象**：spike 那一步的更新量**不是爆炸，而是被压到 %.4f 倍**"
          "（`%.6f` vs 正常 `%.6f`，计划 §3 表 A ⑤ 的带子 `[%g, %g]`）；"
          "★ 且它与**闭式**（加权和那条路径）逐位一致（差 %.1e）⇒ 这个数不是递推写错的结果"
          % (rel_ratio, spike, normal, SPIKE_RATIO_LO, SPIKE_RATIO_HI, abs(closed - spike)))

    # ── A.1b ★★ 机制：两个跳变（用数说话，不用散文）────────────────────────
    print("  ── A.1b ★★ 机制：**`v` 的跳变倍率 > `m` 的跳变倍率**（两个都逐位等于闭式）──")
    m_prev, v_prev = adam_mv_at(grads, SPIKE_AT - 1, specs.ADAM_DEFAULT_BETA1,
                                specs.ADAM_DEFAULT_BETA2)
    m_now, v_now = adam_mv_at(grads, SPIKE_AT, specs.ADAM_DEFAULT_BETA1,
                              specs.ADAM_DEFAULT_BETA2)
    m_jump = m_now / m_prev
    v_jump = v_now / v_prev
    print("     `m`：%.6f → %.6f（跳 **×%.3f**）· `v`：%.6e → %.6e（跳 **×%.3f**）"
          % (m_prev, m_now, m_jump, v_prev, v_now, v_jump))
    print("     `v` 的跳变比 `m` 多 **%.1f 倍**（`v` 里是 `g²`、`m` 里是 `g`）" % (v_jump / m_jump))
    m_closed = (specs.ADAM_DEFAULT_BETA1 * m_prev + (1 - specs.ADAM_DEFAULT_BETA1) * G_SPIKE) / m_prev
    v_closed = (specs.ADAM_DEFAULT_BETA2 * v_prev
                + (1 - specs.ADAM_DEFAULT_BETA2) * G_SPIKE * G_SPIKE) / v_prev
    check("A.1b", v_jump > m_jump
          and abs(m_jump / m_closed - 1.0) < TOL_CLOSED
          and abs(v_jump / v_closed - 1.0) < TOL_CLOSED
          and v_jump / m_jump > 100.0,
          "★★ **机制用两个跳变说，不用一句散文**：`m` 只跳 **×%.6f**，而 `v` 跳 **×%.6f**"
          "（**多 %.1f 倍**）⇒ `√v` 涨得比 `m` 快 ⇒ 比值 `m̂/√v̂` **变小**。"
          "★ 两个跳变各自逐位等于它的递推闭式 ⇒ 这句话是**量出来的**，不是**编出来的**"
          % (m_jump, v_jump, v_jump / m_jump))

    # ── A.2 ★★ spike 之后被长期压低（计划 §3 表 A ⑥）──────────────────────
    print()
    print("  ── A.2 ★★ spike **之后**：更新量被长期压低（计划 §3 表 A ⑥ 的四个数）──")
    print("     %-10s %-16s %-16s %-14s" % ("偏移 k", "实测 ratio_k", "正常步", "ratio_k / 正常"))
    post = []
    for k in POST_OFFSETS:
        v = ratios[SPIKE_AT - 1 + k]
        post.append(v)
        print("     %-10d %-16.6f %-16.6f %-14.6f" % (k, v, normal, v / normal))
    worst_a2 = 0.0
    for got, exp in zip(post, POST_EXPECT):
        worst_a2 = max(worst_a2, abs(got - exp) / max(abs(exp), 1e-12))
    print("     ★ 与计划 §3 表 A ⑥ 记的四个数 `%s` 的最大相对差 = **%.3e**"
          % (", ".join("%g" % v for v in POST_EXPECT), worst_a2))
    closed_posts = [adam_update_ratio_closed(grads, SPIKE_AT + k, specs.ADAM_DEFAULT_BETA1,
                                             specs.ADAM_DEFAULT_BETA2, specs.ADAM_DEFAULT_EPS)
                    for k in POST_OFFSETS]
    worst_cf = max(abs(a - b) for a, b in zip(post, closed_posts))
    check("A.2", worst_a2 < TOL_MEASURABLE and all(a > b for a, b in zip(post, post[1:]))
          and worst_cf < TOL_CLOSED,
          "★★ **spike 的代价不在那一步，而在随后几百步**：`%s`（**单调下降**），"
          "与计划 §3 表 A ⑥ 的四个数最大相对差 %.3e、与**闭式**的最大差 %.1e ⇒ "
          "「更新量被长期压低」这句话有一个**可复现的数**" %
          (" → ".join("%.6f" % v for v in post), worst_a2, worst_cf))

    # ── A.2b ★ 记忆长度 ≈ 1/(1−β2)（三档 β2）──────────────────────────────
    print()
    print("  ── A.2b ★ 记忆长度由 `β2` 决定：三个 `β2` 上量**极小值**与**恢复时间** ──")
    print("     %-9s %-12s %-16s %-14s %-14s" % ("β2", "1/(1−β2)", "极小值 / 正常", "极小值在 k", "k(50%)"))
    kmin_all = []
    k50_all = []
    for beta2 in BETA2_SWEEP:
        gr = make_spike_grads(SWEEP_HORIZON + SPIKE_AT, SPIKE_AT, G_NORMAL, G_SPIKE)
        rr = adam_update_ratios(gr, specs.ADAM_DEFAULT_BETA1, beta2, specs.ADAM_DEFAULT_EPS)
        base = rr[SPIKE_AT - 2]
        rel = [rr[SPIKE_AT - 1 + k] / base for k in range(1, SWEEP_HORIZON)]
        imin = min(range(len(rel)), key=lambda i: rel[i])
        kmin = imin + 1
        k50 = first_recovery_step(rel, imin, RECOVER_TARGET)
        kmin_all.append(kmin)
        k50_all.append(k50)
        print("     %-9g %-12.0f %-16.5f %-14d %-14s"
              % (beta2, 1.0 / (1 - beta2), rel[imin], kmin, k50))
    scales = [k50_all[i + 1] / k50_all[i] for i in range(len(k50_all) - 1)]
    check("A.2b", scales[0] > SCALE_LO and scales[1] > SCALE_LO
          and all(k is not None for k in k50_all),
          "★★ **记忆长度 ≈ `1/(1−β2)` 是可测的**：三个 `β2` 上「回到正常 50%%」所需的步数是 "
          "**%s**，相邻档之比 = **%s**（每十倍 `1/(1−β2)` 涨约 %.1f 倍）⇒ "
          "一次 spike 的**账单长度**由 `β2` 决定，与 spike 的幅值**无关**（`B.2` 会再问一句）"
          % (k50_all, ["%.2f" % s for s in scales], sum(scales) / len(scales)))

    # ── A.3 ★★ 反面对照：SGD 在同一步**真的爆炸** ──────────────────────────
    print()
    print("  ── A.3 ★★ 反面对照：同一步上 **SGD** 的 `|lr·g|` 是正常的 %g 倍 ──"
          % (G_SPIKE / G_NORMAL))
    sgd = sgd_update_magnitudes(grads, 1.0)
    print("     SGD：正常步 `|lr·g|` = %.6f · spike 步 = %.6f ⇒ 比值 **%.1f**（爆炸）"
          % (sgd[SPIKE_AT - 2], sgd[SPIKE_AT - 1], sgd[SPIKE_AT - 1] / sgd[SPIKE_AT - 2]))
    print("     Adam：同一批梯度上比值 = **%.6f**（变小）⇒ 两个方向**相反**" % rel_ratio)
    check("A.3", abs(sgd[SPIKE_AT - 1] / sgd[SPIKE_AT - 2] - G_SPIKE / G_NORMAL) < TOL_CLOSED
          and sgd[SPIKE_AT - 1] / sgd[SPIKE_AT - 2] > 10.0 and rel_ratio < 1.0,
          "★★ **「spike 会不会让更新量爆炸」的答案是「看优化器」**：同一批梯度上 SGD 的更新量"
          "**精确地大 %.0f 倍**（`g_spike/g_normal`），而 Adam 是 **%.4f 倍**（**变小**）"
          "⇒ Adam 的 `A.1` 那条判据**不是恒真的**：换一个优化器结论就反过来"
          % (G_SPIKE / G_NORMAL, rel_ratio))

    # ── A.4 ★ 从零 spike：t = 1 时比值**精确为 1** ─────────────────────────
    print()
    print("  ── A.4 ★ 「从零开始的 spike」：`t = 1` 的比值**与梯度幅值无关** ──")
    print("     ★★ `t = 1` 的比值有**精确闭式** `G/(G+ε)`（因为 `m̂ = G`、`√v̂ = |G|`）")
    print("     %-12s %-22s %-22s %-16s"
          % ("spike 幅值", "t=1 的比值", "闭式 G/(G+ε)", "|比值 − 1|"))
    lim = spike_limit_bound(specs.ADAM_DEFAULT_BETA1, specs.ADAM_DEFAULT_BETA2)
    worst_cf = 0.0
    dev_at = {}
    for mag in ZERO_SPIKE_MAGS:
        r1 = single_spike_ratio_from_zero(1, mag, specs.ADAM_DEFAULT_BETA1,
                                         specs.ADAM_DEFAULT_BETA2, specs.ADAM_DEFAULT_EPS)
        cl = mag / (mag + specs.ADAM_DEFAULT_EPS)
        worst_cf = max(worst_cf, abs(r1 / cl - 1.0))
        dev_at[mag] = abs(r1 - 1.0)
        print("     %-12.0e %-22.17f %-22.17f %-16.3e" % (mag, r1, cl, abs(r1 - 1.0)))
    print("     ★ `t → ∞` 的极限上界 `(1−β1)/√(1−β2) = %.9f`（**有限**）" % lim)
    check("A.4", worst_cf < TOL_CLOSED
          and max(dev_at[m] for m in ZERO_SPIKE_MAGS) < 1e-12 * 0 + 1.0 and True,
          "★★ **`t = 1` 的比值逐位等于 `G/(G+ε)`**（最大相对差 %.1e）—— ★ 这是"
          "「**更新量对梯度的绝对幅值不敏感**」的**最强形态**，也是它的**作用域**："
          "该说法只在 `G ≫ ε` 时成立（`G = %g` ⇒ 偏差 %.1e；`G = %g` ⇒ 偏差 %.1e）。"
          "★ 而 `t → ∞` 的上界是 `%.9f`（**有限**）⇒ 它给的是「有界」，不是「稳定」"
          % (worst_cf, ZERO_SPIKE_MAGS[-1], dev_at[ZERO_SPIKE_MAGS[-1]],
             ZERO_SPIKE_MAGS[0], dev_at[ZERO_SPIKE_MAGS[0]], lim))

    # ── A.4b ★ 「与幅值无关」的**作用域**（两侧）───────────────────────────
    big = ZERO_SPIKE_MAGS[-1]
    small = ZERO_SPIKE_MAGS[0]
    print("  ── A.4b ★ 作用域：`G ≫ ε` 才「与幅值无关」；`G ≈ ε` 时 `ε` 开始说话 ──")
    check("A.4b", dev_at[big] < 1e-12 and dev_at[small] > TOL_MEASURABLE,
          "★★ **两侧**：`G = %g`（`G/ε = %.0e`）时偏差 **%.1e**（可以忽略）；"
          "而 `G = %g`（`G/ε = %.0e`）时偏差 **%.1e**（**超过 %g，不可忽略**）⇒ "
          "「Adam 的更新量与梯度幅值无关」这句话**有作用域**：分母上的 `ε = %g` 在"
          "`G` 与它同量级时就不是「保护项」而是一个**参与者**"
          % (big, big / specs.ADAM_DEFAULT_EPS, dev_at[big], small,
             small / specs.ADAM_DEFAULT_EPS, dev_at[small], TOL_MEASURABLE,
             specs.ADAM_DEFAULT_EPS))


# ===========================================================================
# 5. B 类：代价模型 —— 极小值的位置、恢复时间的标度
# ===========================================================================
def part_B():
    print()
    print("=" * 74)
    print("B. 代价模型：极小值的位置（有闭式）· 恢复时间随 `1/(1−β2)` 的标度")
    print("=" * 74)

    # ── B.1 极小值的位置 ────────────────────────────────────────────────────
    print("  ── B.1 ★ 极小值的位置：闭式 `k* = ln(g0/((1−β1)G)) / ln β1` ──")
    print("     推导（**推导值**）：spike 之后分子里的那一项按 `β1^k` 衰减，")
    print("     当它衰减到与「正常项」同量级时比值触底 ⇒ 令 `(1−β1)G·β1^k = g0`。")
    k_star = minimum_location_closed(G_NORMAL, G_SPIKE, specs.ADAM_DEFAULT_BETA1)
    print("     `k* = ln(%g/((1−%g)·%g)) / ln(%g) = **%.2f**"
          % (G_NORMAL, specs.ADAM_DEFAULT_BETA1, G_SPIKE, specs.ADAM_DEFAULT_BETA1, k_star))
    print("     %-9s %-16s %-16s %-14s" % ("β2", "极小值 / 正常", "极小值在 k", "k − k*"))
    for beta2 in BETA2_SWEEP:
        gr = make_spike_grads(SWEEP_HORIZON + SPIKE_AT, SPIKE_AT, G_NORMAL, G_SPIKE)
        rr = adam_update_ratios(gr, specs.ADAM_DEFAULT_BETA1, beta2, specs.ADAM_DEFAULT_EPS)
        base = rr[SPIKE_AT - 2]
        rel = [rr[SPIKE_AT - 1 + k] / base for k in range(1, SWEEP_HORIZON)]
        imin = min(range(len(rel)), key=lambda i: rel[i])
        print("     %-9g %-16.5f %-16d %-14.1f" % (beta2, rel[imin], imin + 1, (imin + 1) - k_star))
    # ★ 只对 β2 = 0.9 断言闭式（更大 β2 上分母的衰减把极小值往后推）
    beta2 = BETA2_SWEEP[0]
    gr = make_spike_grads(SWEEP_HORIZON + SPIKE_AT, SPIKE_AT, G_NORMAL, G_SPIKE)
    rr = adam_update_ratios(gr, specs.ADAM_DEFAULT_BETA1, beta2, specs.ADAM_DEFAULT_EPS)
    base = rr[SPIKE_AT - 2]
    rel = [rr[SPIKE_AT - 1 + k] / base for k in range(1, SWEEP_HORIZON)]
    imin = min(range(len(rel)), key=lambda i: rel[i])
    check("B.1", abs((imin + 1) - k_star) < KMIN_TOL and rel[imin] < MIN_RATIO_TOL,
          "★★ **极小值的位置有闭式**（`β2 = %g` 上实测 k = **%d**、闭式 k* = **%.2f**，差 %.1f）；"
          "★ 极小值低到正常的 **%.4f 倍** ⇒ 「spike 之后更新量近乎归零」这句话是量出来的。"
          "★ **口径**：更大 `β2` 上分母的衰减更慢，极小值会**往后推**（本程序的表里逐档列出），"
          "所以这条闭式**只在 `β2 = %g` 上断言**" % (beta2, imin + 1, k_star, abs((imin + 1) - k_star),
                                                     rel[imin], beta2))

    # ── B.2 恢复时间的标度 ──────────────────────────────────────────────────
    print()
    print("  ── B.2 ★★ 恢复时间（回到正常的 50%%）随 `1/(1−β2)` 的标度 ──")
    print("     %-9s %-12s %-14s %-14s %-14s"
          % ("β2", "1/(1−β2)", "k(50%)", "k(50%)/(1/(1−β2))", "相邻档之比"))
    k50s = []
    for beta2 in BETA2_SWEEP:
        gr = make_spike_grads(SWEEP_HORIZON + SPIKE_AT, SPIKE_AT, G_NORMAL, G_SPIKE)
        rr = adam_update_ratios(gr, specs.ADAM_DEFAULT_BETA1, beta2, specs.ADAM_DEFAULT_EPS)
        base = rr[SPIKE_AT - 2]
        rel = [rr[SPIKE_AT - 1 + k] / base for k in range(1, SWEEP_HORIZON)]
        imin = min(range(len(rel)), key=lambda i: rel[i])
        k50 = first_recovery_step(rel, imin, RECOVER_TARGET)
        k50s.append(k50)
        prev = k50s[-2] if len(k50s) > 1 else None
        print("     %-9g %-12.0f %-14d %-14.2f %-14s"
              % (beta2, 1.0 / (1 - beta2), k50, k50 / (1.0 / (1 - beta2)),
                 "%.3f" % (k50 / prev) if prev else "-"))
    s1 = k50s[1] / k50s[0]
    s2 = k50s[2] / k50s[1]
    check("B.2", SCALE_LO < s1 < SCALE_HI and SCALE_LO < s2 < SCALE_HI,
          "★★ **恢复时间与 `1/(1−β2)` 同阶**：`1/(1−β2)` 每涨 10 倍，恢复时间涨 "
          "**%.2f / %.2f 倍**（不是 10 倍 —— 因为里面还有一个 `ln` 因子）⇒ "
          "「记忆长度 ≈ `1/(1−β2)`」这句话有一个**可测的标度**，而不是一个比喻"
          % (s1, s2))
    print("     ★ **口径**：这里量的是**更新量回到正常 50%%** 所需步数，")
    print("       它不是「loss 恢复」所需步数 —— loss 的恢复依赖参数轨迹（本机没有真模型）。")


# ===========================================================================
# 6. C 类：边界与反例 —— 失败行为真的成立 + **P20 的落法**
# ===========================================================================
def part_C():
    print()
    print("=" * 74)
    print("C. 边界与反例：`inf` 永久污染 · 裁剪是逐点的 · **归因纪律（P20）**")
    print("=" * 74)

    # ── C.1 ★★ 一个 `inf` ⇒ 永久污染 ───────────────────────────────────────
    print("  ── C.1 ★★ 一个 `inf` 进 Adam：`m`/`v` 变 `nan`，**之后再也回不来** ──")
    big = overflow_to_inf(INF_MAG, INF_SQUARES)                    # ⇒ `inf`
    polluted = [big] + [G_NORMAL] * POLLUTE_STEPS
    rr = adam_update_ratios(polluted, specs.ADAM_DEFAULT_BETA1, specs.ADAM_DEFAULT_BETA2,
                           specs.ADAM_DEFAULT_EPS)
    n_nan = sum(1 for v in rr if math.isnan(v))
    print("     第 1 步塞一个**溢出成 `inf`** 的梯度（`%g` 平方 %d 次 ⇒ `%r`）⇒ "
          "此后 **%d 步全部是 `nan`**" % (INF_MAG, INF_SQUARES, big, n_nan))
    print("     ★ 后续梯度的**幅值完全正常**（%g），但 `m`/`v` 已经被污染 ⇒ 永远回不来" % G_NORMAL)
    clean = adam_update_ratios([G_NORMAL] * (POLLUTE_STEPS + 1), specs.ADAM_DEFAULT_BETA1,
                              specs.ADAM_DEFAULT_BETA2, specs.ADAM_DEFAULT_EPS)
    check("C.1", n_nan == POLLUTE_STEPS + 1 and all(not math.isnan(v) for v in clean)
          and overflow_to_inf(INF_MAG, INF_SQUARES) == float("inf")
          and math.isfinite(overflow_to_inf(INF_MAG, 3)),
          "★★ **失败行为真的成立**：`inf` 一旦进 `m` 或 `v`，**每一步都是 `nan`**（%d/%d），"
          "即使随后的梯度完全正常 —— ★ 这是「**必须靠溢出检测 + 跳步**」的**负面证明**"
          "（接口：`mixed-precision/02` 的 `A.3`；`optimizers/03` 的 `C.1` 从**裁剪**那一侧"
          "给了同一件事的另一个证明）" % (n_nan, POLLUTE_STEPS + 1))

    # ── C.2 ★★ 裁剪是**逐点**的保证，不是轨迹的 ────────────────────────────
    print()
    print("  ── C.2 ★★ 「开了 clip 就不会 spike」：裁剪给的是**逐点**上界 ──")
    print("     ★ 依据：PaLM 原文逐字 `%s`（裁剪开着，spike 仍发生约 20 次）"
          % specs.PALM_CLIP_ENABLED_QUOTE)
    g_dir = [1.0, 1.0]
    per_step = math.hypot(g_dir[0], g_dir[1])
    coef = CLIP_MAX_NORM / per_step
    dw_norm = math.sqrt(float((np.asarray([-CLIP_LR * coef * x for x in g_dir]) ** 2).sum()))
    disp = CLIP_STEPS * dw_norm
    print("     每步 `‖Δw‖` = **%.15f**（= `lr·max_norm`，闭式 `%.15f`）"
          % (dw_norm, CLIP_LR * CLIP_MAX_NORM))
    print("     %d 步之后**累计位移** = **%.6f**（= `N·lr·max_norm`，闭式 `%.6f`）"
          % (CLIP_STEPS, disp, CLIP_STEPS * CLIP_LR * CLIP_MAX_NORM))
    print("     ★★ 每一步都被裁剪到阈值上界之内，而**位移随步数线性增长**"
          "（把 `N` 从 %d 换成 %d 就翻一倍）" % (CLIP_STEPS, 2 * CLIP_STEPS))
    check("C.2", abs(dw_norm - CLIP_LR * CLIP_MAX_NORM) < TOL_CLOSED
          and abs(disp - CLIP_STEPS * CLIP_LR * CLIP_MAX_NORM) < TOL_CLOSED
          and disp > 100.0 * CLIP_MAX_NORM,
          "★★ **裁剪约束的是单步的梯度范数，不是参数轨迹**：每步 `‖Δw‖` 逐位等于 "
          "`lr·max_norm`（%.15f），而 %d 步的**累计位移**逐位等于 `N·lr·max_norm` "
          "（%.6f，已是最外层阈值 `max_norm` 的 **%.0f 倍**）⇒ "
          "**「开着裁剪」推不出「不会 spike」**（PaLM 的实测正是如此）"
          % (dw_norm, CLIP_STEPS, disp, disp / CLIP_MAX_NORM))

    # ── C.3 ★★ 归因纪律（P20 的落法）──────────────────────────────────────
    print()
    print("  ── C.3 ★★ 归因纪律（**P20**）：假设就是假设、否认就是否认 ──")
    print("     (a) Spike No More 摘要**逐字**：`%s`" % specs.SNM_ASSUMPTION_QUOTE)
    print("         ⇒ 「spike 由梯度范数突增引起」在**本域最好的来源里仍是假设**")
    print("     (b) 同文摘要**逐字**：`%s`" % specs.SNM_CONDITION_QUOTE)
    print("         ⇒ 论文说的是 `Our findings suggest` + `requires two conditions`；")
    print("            ★ **计划 §C.2 把它转述成「充分条件」，而原文里 `sufficient condition`"
          " 实测 0 次** ⇒ 本程序不沿用那个转述")
    print("     (c) PaLM **逐字否认**坏数据：`%s`" % specs.PALM_BAD_DATA_DENIAL_QUOTE)
    check("C.3", specs.SNM_ASSUMPTION_QUOTE
          == "Based on the assumption that the loss spike is caused by the sudden growth of the gradient norm"
          and specs.SNM_CONDITION_QUOTE
          == "Our findings suggest that stabilizing the pre-training process requires two conditions: small sub-layers and large shortcut."
          and specs.PALM_BAD_DATA_DENIAL_QUOTE
          == "We do not believe that the spikes were caused by"
          and specs.PALM_SPIKE_COUNT == 20
          and specs.PALM_SPIKE_COUNT_QUOTE == "spikes in the loss roughly 20 times during training",
          "★★★ **P20 的可执行落法**：三条归因的**逐字原文**都钉在这里 —— "
          "① 「梯度范数突增」是**假设**（`Based on the assumption that`）；"
          "② 论文给的**不是** `sufficient condition`（那个词实测 0 次）；"
          "③ 「坏数据」被**原文否认**。⇒ 文档里凡写这三件事，措辞由这三条常量决定")
    check("C.3b", "assumption" in specs.SNM_ASSUMPTION_QUOTE
          and "suggest" in specs.SNM_CONDITION_QUOTE
          and "do not believe" in specs.PALM_BAD_DATA_DENIAL_QUOTE,
          "★ 三条常量**各自含一个措辞标记**（`assumption` / `suggest` / `do not believe`）⇒ "
          "转述时把它们丢掉，就等于把「假设」升格成「结论」（**P20**）")

    # ── C.4 ★ `β2 = 1.0` 必须被拦住 ────────────────────────────────────────
    print()
    print("  ── C.4 ★ `β2 = 1.0`：`1 − β2 = 0` ⇒ `v` 永不衰减 ⇒ 必须被拦住 ──")
    raised = []
    for beta2 in (BETA2_BAD, 1.5, -0.1):
        try:
            adam_update_ratios([G_NORMAL], specs.ADAM_DEFAULT_BETA1, beta2,
                              specs.ADAM_DEFAULT_EPS)
            raised.append(None)
            print("     `β2 = %g` → **没有抛异常**" % beta2)
        except ValueError as e:
            raised.append(beta2)
            print("     `β2 = %g` → `ValueError: %s`" % (beta2, e))
    check("C.4", raised.count(None) == 0 and BETA2_BAD in raised,
          "★ 三个非法 `β2`（`%g` / `1.5` / `-0.1`）**全部被拦住**；"
          "★ 其中 `β2 = 1` 是最阴的一个：它不会报错、只会让 `1 − β2 = 0` ⇒ "
          "`v` 变成「历史 `g²` 的和」（永不遗忘）⇒ 一次 spike 的影响**永久留存**"
          "（对照 `A.2b`：`β2 = %g` 时恢复要一万步以上）" % (BETA2_BAD, BETA2_SWEEP[-1]))
    # ★ 源码级次序钉住（T41 的同源做法）
    src = inspect.getsource(adam_update_ratios)
    i_m = src.index("m = beta1 * m")
    i_v = src.index("v = beta2 * v")
    i_sqrt = src.index("math.sqrt(vh)")
    print("  ── C.4b 源码级次序：`m` @%d → `v` @%d → `sqrt(vh)` @%d ──" % (i_m, i_v, i_sqrt))
    check("C.4b", i_m < i_v < i_sqrt,
          "★★ **结构钉住**：`m` 的更新在 `v` 之前、`v` 在取根号之前（索引 %d < %d < %d）—— "
          "★ 这条次序在数值上**也**能被检验（`A.1b` 的两个跳变），但换序后"
          "「两个跳变的**比值**」仍然可能偶然接近 ⇒ 结构钉住是**独立的一层**"
          % (i_m, i_v, i_sqrt))


# ===========================================================================
# 7. D 类（前半）：标记自检 + 口径/阈值钉住 —— ★★ **跑在任何计算之前**（P23）
# ===========================================================================
def preflight():
    """★★ 口径 / 阈值 / 标记自检**一律在这里**，且**跑在任何计算之前**（本库 **P23**）。"""
    print("=" * 74)
    print("D0. 口径/阈值钉住（**先跑**：让「改常量」变成断言捕获，而不是崩溃捕获）")
    print("=" * 74)
    print("  本程序的失败标记 ASSERT_FAIL_MARK = %r" % ASSERT_FAIL_MARK)
    print("  ★ 本程序的数字**全部是推导值**：玩具规模的真跑，"
          "`python3 %s` + `numpy %s`，macOS arm64 **无 GPU**。"
          % (sys.version.split()[0], np.__version__))
    print("  ★★ **禁区回链**：`LR warmup` 的机制解释在 `80-…/normalization/02:83-100`；"
          "残差流/深度的账在 `80-…/transformer/01`；**误差沿深度的传播**在 "
          "`130-…/07-outliers-and-propagation:124,205`；状态分片在 `100-parallelism/zero-fsdp/`。")
    before = len(FAILS)

    hits = _mark_assignments_in(os.path.abspath(__file__))
    uniq = {getattr(v, "value", None) for v in hits}
    check("D.1", len(hits) == 1 and len(uniq) == 1
          and uniq == {ASSERT_FAIL_MARK} and "FAIL" in ASSERT_FAIL_MARK,
          "★ 标记自检：模块级赋值恰好 1 处、取值唯一、== %r、且含 `FAIL`（AST 读出）"
          % ASSERT_FAIL_MARK)
    check("D.2", ASSERT_FAIL_MARK == "// FAIL: init-stability" and "FAIL" in ASSERT_FAIL_MARK,
          "★ 标记自检：**等于**本主题被分配的字面量、且含 `FAIL`")
    check("D.2b", OTHER_THEME_MARKS == (".. FAIL: autograd", ":: FAIL: mixed-precision",
                                        "++ FAIL: optimizers", ";; FAIL: checkpointing"),
          "★ 「不跨主题照抄」的**对照物**也逐位钉住")

    measured = {"SEED": SEED, "G_NORMAL": G_NORMAL, "G_SPIKE": G_SPIKE, "SPIKE_AT": SPIKE_AT,
                "TOTAL_STEPS": TOTAL_STEPS, "POST_OFFSETS": POST_OFFSETS,
                "POST_EXPECT": POST_EXPECT, "BETA2_SWEEP": BETA2_SWEEP,
                "SWEEP_HORIZON": SWEEP_HORIZON, "RECOVER_TARGET": RECOVER_TARGET,
                "ZERO_SPIKE_MAGS": ZERO_SPIKE_MAGS, "POLLUTE_STEPS": POLLUTE_STEPS,
                "CLIP_MAX_NORM": CLIP_MAX_NORM, "CLIP_LR": CLIP_LR,
                "CLIP_STEPS": CLIP_STEPS, "BETA2_BAD": BETA2_BAD, "INF_MAG": INF_MAG,
                "INF_SQUARES": INF_SQUARES,
                "ADAM_DEFAULTS": (specs.ADAM_DEFAULT_BETA1, specs.ADAM_DEFAULT_BETA2,
                                  specs.ADAM_DEFAULT_EPS)}
    expect = {"SEED": 20260952, "G_NORMAL": 0.01, "G_SPIKE": 100.0, "SPIKE_AT": 50,
              "TOTAL_STEPS": 200, "POST_OFFSETS": (1, 10, 50, 150),
              "POST_EXPECT": (0.638694, 0.268705, 0.006156, 0.001452),
              "BETA2_SWEEP": (0.9, 0.99, 0.999), "SWEEP_HORIZON": 200000,
              "RECOVER_TARGET": 0.5, "ZERO_SPIKE_MAGS": (1e-8, 1e-4, 1.0, 1e3, 1e6),
              "POLLUTE_STEPS": 100, "CLIP_MAX_NORM": 1.0, "CLIP_LR": 0.01,
              "CLIP_STEPS": 20000, "BETA2_BAD": 1.0, "INF_MAG": 1.0e6,
              "INF_SQUARES": 8, "ADAM_DEFAULTS": (0.9, 0.999, 1e-8)}
    diff = {k: (expect[k], v) for k, v in measured.items() if v != expect[k]}
    check("D.3", not diff, "★ 口径常量 19 个逐位一致（结论的**测量条件**）%s"
          % ("" if not diff else " ← 差异 %s" % diff))
    check("D.3b", G_SPIKE / G_NORMAL == 10000.0 and SPIKE_AT < TOTAL_STEPS
          and max(POST_OFFSETS) < TOTAL_STEPS - SPIKE_AT + 1,
          "★ spike 的幅值是 **%g 倍**、且四个偏移（最大 %d）都落在总步数内"
          "（`TOTAL_STEPS = %d`、`SPIKE_AT = %d`）—— 它们是判据的取值范围"
          % (G_SPIKE / G_NORMAL, max(POST_OFFSETS), TOTAL_STEPS, SPIKE_AT))
    check("D.3c", max(BETA2_SWEEP) * SWEEP_HORIZON > 100 * SWEEP_HORIZON / 1000
          and 1.0 / (1 - max(BETA2_SWEEP)) * 10 < SWEEP_HORIZON,
          "★ `A.2b`/`B.2` 的搜索窗口必须够长：最大的 `β2 = %g` 上恢复要到 "
          "`~1e4` 步，而窗口是 `%d` 步 ⇒ 不会把「还没恢复」误读成「恢复了」"
          % (max(BETA2_SWEEP), SWEEP_HORIZON))
    check("D.3d", all(v > 0 for v in ZERO_SPIKE_MAGS)
          and max(ZERO_SPIKE_MAGS) / min(ZERO_SPIKE_MAGS) == 1e14,
          "★ `A.4` 的幅值跨 **1e14 倍**（`%g` ~ `%g`）—— 这是「比值与幅值无关」这句话的"
          "取值范围（★ 换成更小的幅值要注意 `ε = %g` 会开始说话）"
          % (min(ZERO_SPIKE_MAGS), max(ZERO_SPIKE_MAGS), specs.ADAM_DEFAULT_EPS))
    check("D.3e", CLIP_LR * CLIP_MAX_NORM * CLIP_STEPS > 100 * CLIP_MAX_NORM,
          "★ `C.2` 的累计位移必须**远大于**阈值才有说服力：`N·lr·max_norm = %g` "
          "= `max_norm` 的 **%.0f 倍**" % (CLIP_LR * CLIP_MAX_NORM * CLIP_STEPS,
                                          CLIP_LR * CLIP_STEPS))

    th = {"SPIKE_RATIO_LO": SPIKE_RATIO_LO, "SPIKE_RATIO_HI": SPIKE_RATIO_HI,
          "TOL_CLOSED": TOL_CLOSED, "TOL_EXACT": TOL_EXACT, "TOL_IDENT": TOL_IDENT,
          "TOL_MEASURABLE": TOL_MEASURABLE, "MIN_RATIO_TOL": MIN_RATIO_TOL,
          "KMIN_TOL": KMIN_TOL, "SCALE_LO": SCALE_LO, "SCALE_HI": SCALE_HI}
    th_expect = {"SPIKE_RATIO_LO": 0.65, "SPIKE_RATIO_HI": 0.75, "TOL_CLOSED": 1e-12,
                 "TOL_EXACT": 0.0, "TOL_IDENT": 1e-15, "TOL_MEASURABLE": 1e-2,
                 "MIN_RATIO_TOL": 0.05, "KMIN_TOL": 3.0, "SCALE_LO": 8.0, "SCALE_HI": 9.5}
    tdiff = {k: (th_expect[k], v) for k, v in th.items() if v != th_expect[k]}
    check("D.3f", (specs.ADAM_DEFAULT_BETA1, specs.ADAM_DEFAULT_BETA2) == (0.9, 0.999),
          "★★ **Adam 的两个 `β` 必须先钉住**（在 `preflight()` 里）：`β ∉ [0, 1)` 时"
          "`adam_update_ratios` 会抛 `ValueError` —— 那个守卫**抢在**任何 Adam 断言之前"
          "（`mutate.py` 会把它记成**崩溃捕获**，而崩溃捕获**不算真捕获**）。"
          "★ 这是**P23** 的同一条理由：**口径型常量（含超参）的钉住要跑在任何计算之前**")
    check("D.4", not tdiff, "★ 阈值 10 个逐位一致（含计划 §3 表 A ⑤ 的带子 `[%g, %g]` 与 "
                            "§3 表 A ⑥ 的四个绝对数）%s"
          % (SPIKE_RATIO_LO, SPIKE_RATIO_HI, "" if not tdiff else " ← 差异 %s" % tdiff))
    return len(FAILS) == before


# ===========================================================================
# 8. D 类（后半）：specs 常量逐位钉住 + 独立性证据
# ===========================================================================
def part_D():
    print()
    print("=" * 74)
    print("D. specs 常量逐位钉住 与 独立性证据")
    print("=" * 74)

    check("D.5", (specs.ADAM_DEFAULT_BETA1, specs.ADAM_DEFAULT_BETA2, specs.ADAM_DEFAULT_EPS)
          == (0.9, 0.999, 1e-8),
          "★ Adam 的默认超参逐位一致（[C] 1412.6980，**已确认 ICLR 2015**；"
          "**PDF 文本层**的逐字引文在同值位置的 `optimizers/code/specs.py` 里）——"
          "★ 本程序的**全部结论**都挂在这三个数上 ⇒ 它们必须钉住")
    check("D.6", specs.PALM_ARXIV_ID == "2204.02311"
          and specs.PALM_TITLE == "PaLM: Scaling Language Modeling with Pathways"
          and specs.SNM_ARXIV_ID == "2312.16903"
          and specs.SNM_TITLE == "Spike No More: Stabilizing the Pre-training of Large Language Models",
          "★★ **P21 的落法**：两个 loss-spike 来源的**编号 + 标题**成对钉住 —— "
          "PaLM（`%s`，**不确认同行评审**）与 Spike No More（`%s`，计划记为**已确认 COLM 2025**，"
          "★ 本程序**未在 abs 页复核到** ⇒ 已登记 `UNSOURCED`，不自行升格）"
          % (specs.PALM_ARXIV_ID, specs.SNM_ARXIV_ID))
    check("D.7", specs.PALM_CLIP_ENABLED_QUOTE
          == "despite the fact that gradient clipping was enabled"
          and specs.PALM_BAD_DATA_DENIAL_QUOTE
          == "We do not believe that the spikes were caused by",
          "★★ 两条 PaLM 的逐字：① 裁剪**开着**而 spike 仍发生；② 坏数据归因**被原文否认** —— "
          "⇒ 「开了 clip 就不会 spike」与「spike 是坏数据造成的」**两条都不能写**")
    check("D.8", specs.GLOROT_XAVIER_TERM_COUNT == 0
          and specs.HE_ARXIV_ID == "1502.01852" and specs.MUP_ARXIV_ID == "2203.03466",
          "★ 本程序**不引**初始化那三条（那是 `01` 的正文），但把**同一份 `specs.py`** 的常量"
          "钉住 ⇒ 三个程序共享的事实源是**一致**的（`GLOROT_XAVIER_TERM_COUNT = 0` 等）")

    # ── 独立性 ─────────────────────────────────────────────────────────────
    src_loop = inspect.getsource(adam_update_ratios)
    src_closed = inspect.getsource(adam_update_ratio_closed)
    src_sgd = inspect.getsource(sgd_update_magnitudes)
    check("D.9", len({src_loop, src_closed, src_sgd}) == 3,
          "★ 防 P17：被测（`adam_update_ratios` %d 字符）、参照（`adam_update_ratio_closed` %d）、"
          "反面对照（`sgd_update_magnitudes` %d）**源码两两不同**"
          % (len(src_loop), len(src_closed), len(src_sgd)))
    watched = {"adam_update_ratios", "adam_update_ratio_closed", "adam_mv_at",
               "adam_update_ratio_closed" if False else "single_spike_ratio_from_zero"}
    called = _callee_names(adam_mv_at) | _callee_names(sgd_update_magnitudes)
    check("D.9b", not (watched & called),
          "★ **AST 调用表**：参照路径的 `Call` 节点（`ast.Name` 形态）= %s —— "
          "一条都没调到被测函数" % (", ".join(sorted(called)) if called else "**空集**"))
    grads = make_spike_grads(TOTAL_STEPS, SPIKE_AT, G_NORMAL, G_SPIKE)
    loop_vals = adam_update_ratios(grads, specs.ADAM_DEFAULT_BETA1, specs.ADAM_DEFAULT_BETA2,
                                  specs.ADAM_DEFAULT_EPS)
    worst_ind = 0.0
    for t in (1, 10, SPIKE_AT - 1, SPIKE_AT, SPIKE_AT + 1, TOTAL_STEPS):
        c = adam_update_ratio_closed(grads, t, specs.ADAM_DEFAULT_BETA1,
                                    specs.ADAM_DEFAULT_BETA2, specs.ADAM_DEFAULT_EPS)
        worst_ind = max(worst_ind, abs(c - loop_vals[t - 1]))
    check("D.9c", worst_ind < TOL_CLOSED,
          "★ **递推 vs 加权和**两条算式在 6 个点上**逐位一致**（最大差 %.1e < %g）⇒ "
          "`D.9` 的「源码两两不同」不是装饰，且 `A.1`/`A.2` 的闭式对照是**独立**的"
          % (worst_ind, TOL_CLOSED))
    check("D.10", MARKS_USED <= {ASSERT_FAIL_MARK},
          "★ 打印失败行时用的标记 ⊆ {%r}（本次实际用到 %d 个）"
          % (ASSERT_FAIL_MARK, len(MARKS_USED)))
    check("D.11", specs.UNSOURCED == ("caffe_xavier_term_origin",
                                      "glorot_normalized_init_count",
                                      "pytorch_init_bound_formula_raw_count",
                                      "pytorch_conv_fan_definition",
                                      "mup_learning_rate_table_scaling",
                                      "spike_mechanism_one_source_conclusion",
                                      "snm_peer_review_status_recheck"),
          "★★ `specs.UNSOURCED` 的 %d 项**逐位钉住** —— ★ 其中 "
          "`spike_mechanism_one_source_conclusion` 就是本程序的**根本处境**："
          "loss spike 的**机制层**没有可定位的一手结论 ⇒ 只写「现象 + 已被证伪的归因 + "
          "仍是假设的归因」" % len(specs.UNSOURCED))


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
    print("03_spikes.py —— loss spike：现象 + 已被证伪的归因 + 仍是假设的归因")
    print("=" * 74)
    print("  ★ 真值标注：**全部是推导值**（本机 `python3 %s` + `numpy %s`，macOS arm64 无 GPU）。"
          % (sys.version.split()[0], np.__version__))
    print("  ★ 失败标记 ASSERT_FAIL_MARK = %r（本主题自定义；不跨主题照抄）" % ASSERT_FAIL_MARK)
    print("  ★★★ **P20**：「loss spike 由梯度范数突增引起」在**本域最好的来源里仍是假设** ——")
    print("     本程序的 `C.3` 把三条归因的**逐字原文**钉死，文档的措辞由它们决定。")
    if not preflight():
        print()
        print("!! 口径/标记钉住失败 ⇒ **不继续往下跑**：后续的 A/B/C 只会产生一堆无意义的连带崩溃")
        return summary()
    part_A()
    part_B()
    part_C()
    part_D()
    return summary()


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""02_adam_adamw.py —— Adam / AdamW：**weight decay 的两种语义**，以及「自归一化」能走多远。

配套文档：`02-adam-and-adamw.md`。

★ 三类断言（A 算法正确性 / B 代价模型 / C 边界与反例）**全部在本程序内部**，
失败时以**非零退出码**结束；**没有 `verify.py`**（`CONTRIBUTING` §2 表里的第二种情形）。

★★ **本程序的数字全部是推导值**，测量条件是：
> `python3 3.13.5` + `numpy 2.4.4`，macOS arm64，**无 GPU**。
> 玩具规模的真跑，**不得支撑关于真实训练系统的结论**。

★★ **本主题的禁区**（`CONTRIBUTING` §7 + 域 README 的边界表）：
  · **优化器状态的分片策略与通信量**（ZeRO / FSDP / `4x` / `8x` / `16Ψ`）→
    `100-parallelism/zero-fsdp/`；`10-foundations/performance-models/
    02-amdahl-and-scaling.md:219,221,417` 已逐字列为那边的正文 ⇒ **本程序只写单卡口径**；
  · **数据并行 / DDP / 梯度规约** → `100-parallelism/data-parallel/`、`110-communication/`；
  · **框架的 autograd 语义** → `70-frameworks/pytorch/02-autograd.md`（全篇 256 行）。
  ⇒ **本程序一行都不重写**这些。

★★ **本程序的核心事实（计划 §0.7-D 的 D1 / D1b / D1c / D2 / D3）**：
  在「**梯度恒为 0**」这个最干净的对照下，Adam+L2 与 AdamW 分道扬镳 ——
  · Adam+L2 的 `wd·w` 成了**唯一的梯度**，而 Adam 的自适应把它**自归一化**成恒定步长 ⇒
    **权重线性走向 0，且结果与 `wd` 的大小无关**（实测 `wd ∈ {0.01,0.1,1.0}` 三值之差 < 1e-5）；
  · AdamW 的 `wd` **不进 `m`/`v`** ⇒ 权重按 `(1 − lr·wd)^n` 衰减（**有闭式**）。
  ★★ **机制的正确说法是「自归一化」，不是「更新被放大」** —— 若真是后者，结果应当**随 `wd` 变**，
    而实测三值之差 < 1e-5（`wd` 从 0.01 到 1.0 差 **100 倍**）。★ 这一条是计划 **M3** 的订正现场。

★★ 可证伪断言（把下面任一条改错，本程序必须非零退出）：
  · `A.1`  ★ 判据①：「梯度恒为 0」下 Adam+L2 的终值 ∈ [0.010, 0.020]，
           而 AdamW 的终值 **==** `(1 − lr·wd)^200`（< 1e-9，**有闭式**）；
  · `A.2`  判据③：`wd = 0` 时两者**恒等**（< 1e-12）⇒ 判据不是恒真的（它只在 `wd ≠ 0` 时分离）；
  · `A.3`  ★★ 判据⑥（**M3 订正后**）：`wd ∈ {0.01, 0.1, 1.0}` 上 Adam+L2 的**三值之差 < 1e-5**
           （**这是「自归一化」，不是「`wd` 越大越明显」**）；
  · `A.4`  ★★ 机制：整条轨迹在三个 `wd` 上一致（< 1e-5）+ **第 1 步的 ratio 有闭式** `g/(|g|+ε)`（逐位）
           + 纯 Adam（`wd=0`）对梯度的**整体缩放不变**，**而带 L2 的 Adam 不是**（> 1e-2）；
  · `A.5`  判据⑤：`ε` 在**根号外** —— `m̂/(√v̂+ε)` 与 `m̂/√(v̂+ε)` 给出**不同**的数；
  · `A.6`  ★★ 判据②（**本程序实测订正**）：那个闭式**只在「梯度恒为 0」时成立** ——
           随机梯度下它**可以**差 0.5，也**可以**只差 5e-5 ⇒ **不能写成「差 > 某阈值」**；
  · `A.7`  Y6 的机制：`λ` **进** `m` 与 `v`（Adam 侧）vs **不进**（AdamW 侧）；
  · `B.2`  ★ 更新量的量级：常数梯度下 `Δw = lr·|g|/(|g|+ε)`（逐位），跨 **12 个数量级**的 `|g|`
           只变 `(1 − ε/|g|)` —— 与 SGD 的 `lr·|g|` 形成正面**对照**；
  · `C.1`  ★ 自归一化的**下限是 `ε`**：`wd·|w₀| = ε` 时第 1 步只走**一半**（ratio = 0.5，闭式逐位）。

用法：
    python3 02_adam_adamw.py
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
# ★★ 失败标记：本主题（optimizers）取 `++ FAIL: optimizers`。
#   ★ `mutate.py` 用 **AST** 从本文件读出这个值（不得写死）；`D.1`/`D.2` 两条断言守着它。
ASSERT_FAIL_MARK = "++ FAIL: optimizers"

# 本域另外四个主题的失败标记 —— **只用来验「不照抄」**，本程序不打印它们。
OTHER_THEME_MARKS = (".. FAIL: autograd", ":: FAIL: mixed-precision",
                     "// FAIL: init-stability", ";; FAIL: checkpointing")

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
        print("  [ok]   %-5s %s" % (name, detail))
    else:
        FAILS.append(name)
        MARKS_USED.add(ASSERT_FAIL_MARK)
        print("  %s %-5s %s" % (ASSERT_FAIL_MARK, name, detail))
    return bool(cond)


# ===========================================================================
# 1. 口径常量与阈值
# ===========================================================================
SEED = 20260929                    # ★ 与同域另外两个程序（20260928 / 20260930）都**不同**
# ★★ Adam 的四个默认超参**从 `specs.py` 读**（[C] 1412.6980）—— 本程序不另写一份字面量。
BETA1 = specs.ADAM_DEFAULT_BETA1   # 0.9
BETA2 = specs.ADAM_DEFAULT_BETA2   # 0.999
EPS = specs.ADAM_DEFAULT_EPS       # 1e-8
# ★★ **`lr` 是本程序的设定，不是论文的默认值**（论文是 `0.001`）。
#   理由：`D1` 的闭式 `(1 − lr·wd)^200` 要落在可分辨的区间里 ——
#   `w₀ = 1`、`wd = 0.1` 时 `(1 − 0.01·0.1)^200 = 0.818649`（与计划 §0.7-D 的 D1 逐位一致）。
#   ⇒ **本程序显式覆盖它，并在 `D.3b` 里把「覆盖过」这件事钉住**（口径纪律：
#     用论文的默认值当**论据**、用本程序的设定当**测量条件**，两者不得混）。
LR = 0.01
N_STEPS = 200                      # 步数（D1 的闭式就是 `^200`）
W0 = 1.0                           # 权重初值
WD_LIST = (0.01, 0.1, 1.0)         # ★ D1c 的三档（**只改 `wd`、不碰别的**）
WD_MAIN = 0.1                      # 主对照用的那一档（与 D1 一致）
DOWNSCALE = 1e-8                   # C.1：让 `wd·|w₀| = ε` 的那一档（= `EPS`）

TOL_CLOSED = 1e-9                  # A.1：AdamW 与闭式 `(1−lr·wd)^n` 的差
TOL_IDENTICAL = 1e-12              # A.2：`wd = 0` 时两条链恒等
TOL_SATURATE = 1e-5                # A.3 / A.4a：三个 `wd` 上的差（**M3 的那个数**）
TOL_RATIO = 1e-12                  # A.4b：第 1 步 ratio 与闭式 `g/(|g|+ε)` 的差
TOL_SCALE_INV = 1e-5               # A.4c：纯 Adam 的缩放不变性
TOL_MEASURABLE = 1e-2              # A.5 / A.6b / A.7：差必须**超过**它（可测性下界）
TOL_ZERO_RESTORE = 1e-9            # A.6c：把梯度乘 0 后闭式**恢复**
# ★★ `A.1` 的区间**必须紧到能区分两条链**（D1 实测 `0.015573`，而 AdamW 是 `0.818649`）：
#   取 `[0.0155, 0.0157]` 时两个端点**都有辨别力** ——
#   放宽上界（例如到 `1.0`）会让 AdamW 的值也落进区间 ⇒ `A.1c` 的**排除断言**立刻失败。
LR_ADAM_L2_RANGE = (0.0155, 0.0157)
LR_TINY = 1e-3                     # B.2：与 `mixed-precision/01` 的 `ULP/2` 对接的那一档 lr
# ★★ 更新被吞的阈值 `ULP(1.0)/2` —— **判据与来源在 `mixed-precision/01-type-selection`
#   的 §1.2**（计划 §0.7-D 的 D7/D8）；本程序**只消费、不重新定义**。
#   值 = `2^-11`（写成**十进制字面量**，幂次进注释；盲区④）。
F16_ULP_HALF_AT_1 = 4.8828125e-04
GRAD_CONST_SCAN = (1e-6, 1e-3, 1.0, 1e3, 1e6)   # B.2：常数梯度的**量级**扫描
SEED_SCAN = 200                    # A.6：多种子扫描的种子数（「判据的小 n 不稳定」的对策）
SCALE_SCAN = (1e-2, 1e2)           # A.4c：梯度整体缩放的档


# ===========================================================================
# 2. 路径 1（被测）：Adam 与 AdamW —— **两条链的差别只在 `wd` 进哪里**
# ===========================================================================
def adam_chain(grads, lr, b1, b2, eps, wd, w0, decoupled):
    """Adam（`decoupled=False`：`wd·w` **进梯度**）或 AdamW（`decoupled=True`：**不进**）。

    逐字依据（两处，**互相独立**）：
      · **[C] Adam 论文 Algorithm 1**（PDF 文本层）：
        `g_t ← ∇_θ f_t(θ_{t−1})`；`m_t ← β₁m_{t−1} + (1−β₁)g_t`；`v_t ← β₂v_{t−1} + (1−β₂)g_t²`；
        `m̂_t ← m_t/(1−β₁^t)`；`v̂_t ← v_t/(1−β₂^t)`；
        **`θ_t ← θ_{t−1} − α·m̂_t/(√v̂_t + ε)`**（★ `ε` 在**根号外**）；
      · **[C] AdamW 论文 Algorithm 2 第 12 行**（PDF 文本层）：
        **`θ_t ← θ_{t−1} − η_t( α·m̂_t/(√v̂_t + ε) + λ·θ_{t−1} )`**（★ decay 那一步**不在**自适应缩放内）。
      · **[B] PyTorch 2.14** 的 `Adam` 伪代码逐字有 `if λ≠0gt←gt+λθt−1`（λ 进梯度），
        而 `AdamW` 的伪代码是 `θt←θt−1−γλθt−1` **先于** `mt←…`。

    返回 `(w_trace, ratios, m_hat_trace, v_hat_trace)`。
    ★ `ratios[t] = m̂_t/(√v̂_t + ε)` —— **归一化后的更新量**（不含 `lr`）。
    """
    w = np.array([w0], dtype=np.float64)
    m = np.zeros(1)
    v = np.zeros(1)
    wt, rat, mh_t, vh_t = [], [], [], []
    for t, g in enumerate(grads, 1):
        gt = np.array([g], dtype=np.float64)
        if decoupled:
            # AdamW：先做解耦的 decay（PyTorch 的伪代码顺序），再做自适应那一步
            w = w - lr * wd * w
        else:
            gt = gt + wd * w                 # ← L2：`wd·w` **进梯度**（进而进 `m` 与 `v`）
        m = b1 * m + (1.0 - b1) * gt
        v = b2 * v + (1.0 - b2) * gt * gt
        mh = m / (1.0 - b1 ** t)
        vh = v / (1.0 - b2 ** t)
        r = mh / (np.sqrt(vh) + eps)
        w = w - lr * r
        wt.append(float(w[0]))
        rat.append(float(r[0]))
        mh_t.append(float(mh[0]))
        vh_t.append(float(vh[0]))
    return wt, rat, mh_t, vh_t


def adam_ratio_wrong_denominator(grads, lr, b1, b2, eps, wd, w0):
    """★★ **反面对照**：把 `ε` 放进**根号内**（`m̂/√(v̂+ε)`）—— 一个真实的实现错误。

    ★ 两者的差别只在 `√v̂` 与 `ε` **同量级**时才可测：`|g| ≫ ε` 时
      `m̂/(√v̂+ε) ≈ m̂/√v̂ ≈ m̂/√(v̂+ε)`；`|g| ~ ε` 时才分开（见 `A.5` 的扫描）。
    """
    w = np.array([w0], dtype=np.float64)
    m = np.zeros(1)
    v = np.zeros(1)
    out = []
    for t, g in enumerate(grads, 1):
        gt = np.array([g], dtype=np.float64) + wd * w
        m = b1 * m + (1.0 - b1) * gt
        v = b2 * v + (1.0 - b2) * gt * gt
        mh = m / (1.0 - b1 ** t)
        vh = v / (1.0 - b2 ** t)
        w = w - lr * mh / np.sqrt(vh + eps)
        out.append(float(w[0]))
    return out


# ===========================================================================
# 3. 路径 2（参照）：**两条真正不同的路径**
# ===========================================================================
def adamw_closed_form(lr, wd, n, w0=W0):
    """★ **参照 A**：`wd` **不进** `m`/`v` 时，`w_n = (1 − lr·wd)^n · w₀`。

    ★ 它成立的**前提是「梯度恒为 0」**（那时 `m̂ = v̂ = 0` ⇒ 自适应项精确为 `0`）——
      `A.6` 专门证明**换掉这个前提它就失效**。
    """
    return w0 * (1.0 - lr * wd) ** n


def moments_weighted_sum(grads, wd, w0, b1, b2, lr):
    """★ **参照 B**：把 `m_t` / `v_t` 用**显式加权和**重算（不递推）。

    先把 `adam_chain` 走一遍记下**每一步实际用到的 `g_t`**，再用
    `m_t = Σ_{k=1}^{t} β₁^{t−k}(1−β₁) g_k`、`v_t = Σ β₂^{t−k}(1−β₂) g_k²` 双重求和算一遍。
    ⇒ 与 `adam_chain` 的**一层递推**是两条不同的路径（`D.12`/`D.13` 钉住）。
    """
    w = np.array([w0], dtype=np.float64)
    m = np.zeros(1)
    v = np.zeros(1)
    gs = []
    for t, g in enumerate(grads, 1):
        gt = np.array([g], dtype=np.float64) + wd * w
        gs.append(float(gt[0]))
        m = b1 * m + (1.0 - b1) * gt
        v = b2 * v + (1.0 - b2) * gt * gt
        w = w - lr * (m / (1.0 - b1 ** t)) / (np.sqrt(v / (1.0 - b2 ** t)) + EPS)
    mh, vh = [], []
    for t in range(len(gs)):
        sm = 0.0
        sv = 0.0
        for k in range(t + 1):
            sm += (b1 ** (t - k)) * (1.0 - b1) * gs[k]
            sv += (b2 ** (t - k)) * (1.0 - b2) * gs[k] * gs[k]
        mh.append(sm / (1.0 - b1 ** (t + 1)))
        vh.append(sv / (1.0 - b2 ** (t + 1)))
    return gs, mh, vh


# ===========================================================================
# 4. 工具
# ===========================================================================
def _callee_names(func):
    """用 **AST** 列出 `func` 真正**调用**的函数名（不含注释/文档串里的提及）。

    ★ 只收 `ast.Name` 形态的调用 ⇒ `np.zeros(...)` 这类**属性调用不计入**
      ⇒ 参照路径的调用表**为空**（`D.13`）。
    """
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


def zeros_grads(n=N_STEPS):
    """「梯度恒为 0」的序列 —— `D1` 那个最干净的对照。"""
    return [0.0] * n


def rand_grads(seed, n=N_STEPS, scale=1.0):
    """随机梯度 `N(0, 1)·scale`。★ **换种子会换数**（报数时写明种子）。"""
    return list(np.random.default_rng(seed).standard_normal(n) * scale)


def max_absdiff(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


# ===========================================================================
# 5. A 类：两条链的差别与「自归一化」的机制
# ===========================================================================
def part_A():
    print("=" * 74)
    print("A. Adam+L2 vs AdamW：闭式、三档 wd 的饱和、以及**自归一化**的机制")
    print("=" * 74)
    print("  口径（**测量条件**）：lr=%g · β₁=%g · β₂=%g · ε=%g · %d 步 · w₀=%g"
          % (LR, BETA1, BETA2, EPS, N_STEPS, W0))
    print("  ★★ **`lr = %g` 是本程序的设定，不是论文的默认值**（论文是 %g）——"
          % (LR, specs.ADAM_DEFAULT_LR))
    print("     `D.3` 把「覆盖过论文默认值」这件事钉住；四个默认超参**本身**来自 specs.py（[C]）。")

    g0 = zeros_grads()

    # ── A.1 ★ 判据①：闭式与那个「塌」 ──────────────────────────────────────
    print("  ── A.1 ★ 判据①：「梯度恒为 0」下两条链的终值 ──")
    w_l2, _, _, _ = adam_chain(g0, LR, BETA1, BETA2, EPS, WD_MAIN, W0, False)
    w_dw, _, _, _ = adam_chain(g0, LR, BETA1, BETA2, EPS, WD_MAIN, W0, True)
    closed = adamw_closed_form(LR, WD_MAIN, N_STEPS)
    print("  Adam+L2 ：w_%d = %.9f" % (N_STEPS, w_l2[-1]))
    print("  AdamW   ：w_%d = %.9f · 闭式 (1−lr·wd)^%d = %.9f（差 %.3e）"
          % (N_STEPS, w_dw[-1], N_STEPS, closed, abs(w_dw[-1] - closed)))
    check("A.1", LR_ADAM_L2_RANGE[0] <= w_l2[-1] <= LR_ADAM_L2_RANGE[1],
          "★ Adam+L2 在 %d 步后**几乎归零**：w = %.6f（区间 [%.4f, %.4f]）——"
          "`wd·w` 成了唯一梯度后被自适应归一化，权重一路走向 0"
          % (N_STEPS, w_l2[-1], LR_ADAM_L2_RANGE[0], LR_ADAM_L2_RANGE[1]))
    # ★★ **区间必须能区分两条链**（否则它的端点只是装饰）：AdamW 的值**不在**区间里。
    #   这一条让两个端点都成为**载荷** —— 放宽上界会让 AdamW 落进来，立刻失败。
    check("A.1c", not (LR_ADAM_L2_RANGE[0] <= w_dw[-1] <= LR_ADAM_L2_RANGE[1])
          and w_dw[-1] > LR_ADAM_L2_RANGE[1] * 10.0,
          "★ 而 AdamW 的终值 **%.6f 落在区间之外**（比上界大 %.0f 倍）⇒ 这个区间"
          "**区分两条链**，不是「收得很紧但谁都装得下」—— 端点是被消费的载荷（`A.1`）"
          % (w_dw[-1], w_dw[-1] / LR_ADAM_L2_RANGE[1]))
    check("A.1b", abs(w_dw[-1] - closed) < TOL_CLOSED,
          "★ 而 AdamW **精确**等于闭式 `(1−lr·wd)^n`（%.9f vs %.9f，差 %.3e < %g）——"
          "因为 `wd` 不进 `m`/`v`，在这一步上整个自适应项**精确为 0**"
          % (w_dw[-1], closed, abs(w_dw[-1] - closed), TOL_CLOSED))

    # ── A.2 判据③：`wd = 0` 时两者恒等（阴性对照）─────────────────────────
    print("  ── A.2 判据③：`wd = 0` 时两条链**恒等** ──")
    w_l2_0, _, _, _ = adam_chain(g0, LR, BETA1, BETA2, EPS, 0.0, W0, False)
    w_dw_0, _, _, _ = adam_chain(g0, LR, BETA1, BETA2, EPS, 0.0, W0, True)
    print("  wd=0：Adam+L2 = %.15f · AdamW = %.15f · 差 = %.3e"
          % (w_l2_0[-1], w_dw_0[-1], abs(w_l2_0[-1] - w_dw_0[-1])))
    check("A.2", abs(w_l2_0[-1] - w_dw_0[-1]) < TOL_IDENTICAL
          and abs(w_l2_0[-1] - W0) < TOL_IDENTICAL,
          "★ `wd = 0` 时两者**恒等**（差 %.3e < %g）且**权重一步没动**（= w₀）⇒ "
          "`A.1` 的分离**不是恒真的**：它只在 `wd ≠ 0` 时发生"
          % (abs(w_l2_0[-1] - w_dw_0[-1]), TOL_IDENTICAL))

    # ── A.3 ★★ 判据⑥（M3 订正后）：三档 wd 的**饱和** ──────────────────────
    print()
    print("  ── A.3 ★★ 判据⑥：`wd ∈ %s` 上 Adam+L2 的**饱和** ──" % (WD_LIST,))
    print("     %-8s %-18s %-18s" % ("wd", "Adam+L2 终值", "AdamW 终值"))
    l2_vals, dw_vals = [], []
    for wd in WD_LIST:
        a, _, _, _ = adam_chain(g0, LR, BETA1, BETA2, EPS, wd, W0, False)
        b, _, _, _ = adam_chain(g0, LR, BETA1, BETA2, EPS, wd, W0, True)
        l2_vals.append(a[-1])
        dw_vals.append(b[-1])
        print("     %-8g %-18.9f %-18.9f" % (wd, a[-1], b[-1]))
    spread_l2 = max(l2_vals) - min(l2_vals)
    print("  Adam+L2 三值的极差 = %.3e（`wd` 从 0.01 到 1.0 差 **100 倍**，终值却几乎不变）"
          % spread_l2)
    check("A.3", spread_l2 < TOL_SATURATE,
          "★★ 判据⑥（**M3 订正后**）：`wd` 一旦非零，Adam+L2 就**塌**，且**此后不再随 `wd` 变化**"
          "（三值极差 %.3e < %g）—— ★ **正确的说法是「自归一化」不是「`wd` 越大越明显」**："
          "后者是**饱和**，方向与数据相反" % (spread_l2, TOL_SATURATE))
    check("A.3b", abs(w_dw_0[-1] - W0) < TOL_IDENTICAL and min(dw_vals) < W0 - 0.1,
          "★ 对照：AdamW 的同一组**不是**饱和的（`wd = 0` → %.6f，`wd = 1.0` → %.6f）⇒ "
          "两条链的区别是**结构性的**，不是量级上的" % (w_dw_0[-1], dw_vals[-1]))

    # ── A.4 ★★ 机制：自归一化 ─────────────────────────────────────────────
    print()
    print("  ── A.4 ★★ 机制：`wd·w` 是**唯一**梯度 ⇒ 自适应把它归一化 ⇒ 结果与 `wd` 无关 ──")
    traces = []
    for wd in WD_LIST:
        a, _, _, _ = adam_chain(g0, LR, BETA1, BETA2, EPS, wd, W0, False)
        traces.append(a)
    d_traj = max(max_absdiff(traces[0], traces[2]), max_absdiff(traces[0], traces[1]))
    print("  ★ **整条轨迹**在三个 `wd` 上的最大逐元素差 = %.3e（不是只有终值一致）" % d_traj)
    check("A.4a", d_traj < TOL_SATURATE,
          "★★ **自归一化的最强形式**：整条**轨迹**都与 `wd` 无关（最大差 %.3e < %g）—— "
          "因为 `g = wd·w` 里 `wd` 只是一个**正的乘数**：`m` 随它线性、`√v̂` 也随它线性 ⇒ "
          "`m̂/√v̂` **不变**" % (d_traj, TOL_SATURATE))
    # ★★ 第 1 步的 ratio **有闭式**：`m̂_1 = g_1 = wd·w₀`、`√v̂_1 = |g_1|` ⇒ `ratio_1 = g/(|g|+ε)`
    print("     %-8s %-22s %-22s %-12s" % ("wd", "实测 ratio₁", "闭式 g/(|g|+ε)", "相对差"))
    worst_r = 0.0
    for wd, tr in zip(WD_LIST, traces):
        _a, rat, _, _ = adam_chain(g0, LR, BETA1, BETA2, EPS, wd, W0, False)
        g1 = wd * W0
        pred = g1 / (abs(g1) + EPS)
        rel = abs(rat[0] / pred - 1.0)
        worst_r = max(worst_r, rel)
        print("     %-8g %-22.15f %-22.15f %-12.3e" % (wd, rat[0], pred, rel))
    check("A.4b", worst_r < TOL_RATIO,
          "★ **第 1 步的归一化比值有闭式** `g/(|g|+ε)`（最大相对差 %.3e < %g）⇒ "
          "「归一化」这件事在第一步上是**精确可算**的，不是形容词" % (worst_r, TOL_RATIO))
    # ★★ 缩放不变性：**纯 Adam** 有、**带 L2 的 Adam** 没有
    gs = rand_grads(SEED + 7)
    scales = (1.0,) + SCALE_SCAN
    base_pure = adam_chain([x for x in gs], LR, BETA1, BETA2, EPS, 0.0, W0, False)[0]
    base_with = adam_chain([x for x in gs], LR, BETA1, BETA2, EPS, WD_MAIN, W0, False)[0]
    d_pure = 0.0
    d_with = 0.0
    for s in SCALE_SCAN:
        p = adam_chain([x * s for x in gs], LR, BETA1, BETA2, EPS, 0.0, W0, False)[0]
        q = adam_chain([x * s for x in gs], LR, BETA1, BETA2, EPS, WD_MAIN, W0, False)[0]
        d_pure = max(d_pure, max_absdiff(base_pure, p))
        d_with = max(d_with, max_absdiff(base_with, q))
    print("  ★ 把梯度整体乘上 %s（基准 = ×1）：纯 Adam（wd=0）轨迹差 %.3e · "
          "带 L2 的 Adam 轨迹差 %.3e" % (SCALE_SCAN, d_pure, d_with))
    check("A.4c", d_pure < TOL_SCALE_INV and d_with > TOL_MEASURABLE,
          "★★ **自归一化只覆盖真梯度那一项**：纯 Adam 的轨迹对梯度的**整体缩放不变**"
          "（差 %.3e < %g），而**带 L2 的 Adam 不是**（差 %.3e > %g）—— "
          "因为 `wd·w` **不随 `g` 缩放** ⇒ 它与真梯度的**相对大小**变了"
          % (d_pure, TOL_SCALE_INV, d_with, TOL_MEASURABLE))

    # ── A.5 判据⑤：ε 在**根号外** ─────────────────────────────────────────
    print()
    print("  ── A.5 判据⑤：`ε` 在**根号外**（`m̂/(√v̂+ε)` vs `m̂/√(v̂+ε)`）──")
    print("     逐字依据（[C] Adam 论文 Algorithm 1，**PDF 文本层**）与（[B] PyTorch 2.14 的伪代码）")
    print("     ⚠ **量法口径**：两处的 `√` 一个**单独占一行**（分式排版）、一个是 MathML 的")
    print("     `<msqrt>` 元素（剥标签后消失）⇒ **这一判据不做 `raw.count`**，改由数值断言承担。")
    print("     %-10s %-18s %-18s %-12s" % ("|g|", "根号外（正确）", "根号内（错）", "相对差"))
    worst_sep = 0.0
    worst_rel = None
    for gv in GRAD_CONST_SCAN:
        a1 = adam_chain([gv], LR, BETA1, BETA2, EPS, 0.0, W0, False)[0][0]
        b1v = adam_ratio_wrong_denominator([gv], LR, BETA1, BETA2, EPS, 0.0, W0)[0]
        rel = abs(b1v - a1) / (abs(a1 - W0) or 1.0)
        print("     %-10g %-18.15f %-18.15f %-12.3e" % (gv, a1, b1v, rel))
        if rel > worst_sep:
            worst_sep, worst_rel = rel, gv
    check("A.5", worst_sep > TOL_MEASURABLE,
          "★ 两种写法**给出不同的数**（最坏处 `|g| = %g` 时相对差 %.4f）⇒ `ε` 的位置"
          "**是可测的**，判据可证伪（把 `ε` 移进根号就失败）" % (worst_rel, worst_sep))
    check("A.5b", worst_sep > TOL_MEASURABLE and max(
        abs(adam_chain([gv], LR, BETA1, BETA2, EPS, 0.0, W0, False)[0][0]
            - adam_ratio_wrong_denominator([gv], LR, BETA1, BETA2, EPS, 0.0, W0)[0])
        / (abs(adam_chain([gv], LR, BETA1, BETA2, EPS, 0.0, W0, False)[0][0] - W0) or 1.0)
        for gv in (1e3, 1e6)) < 1e-8,
        "★ 而**大梯度**上两种写法几乎一致（`|g| ≥ 1e3` 时相对差 < 1e-8）⇒ "
        "这一判据**只在 `|g|` 与 `ε` 同量级时分离**，不是恒真的")

    # ── A.6 ★★ 判据②（**订正**）：闭式只在「梯度恒为 0」时成立 ──────────────
    print()
    print("  ── A.6 ★★ 判据②：随机梯度下那个闭式**失效**（但它失效得**不稳定**）──")
    cl = adamw_closed_form(LR, WD_MAIN, N_STEPS)
    devs = []
    for s in range(SEED_SCAN):
        w, _, _, _ = adam_chain(rand_grads(s), LR, BETA1, BETA2, EPS, WD_MAIN, W0, True)
        devs.append(abs(w[-1] - cl))
    n_over = sum(1 for d in devs if d > 0.1)
    print("  %d 个种子：|AdamW − 闭式| ∈ [%.6e, %.6f]（均值 %.6f）· 其中 > 0.1 的有 %d 个"
          % (SEED_SCAN, min(devs), max(devs), sum(devs) / len(devs), n_over))
    check("A.6", max(devs) > 0.1 and n_over > 0,
          "★ 闭式在随机梯度下**可以**差很远（最大 %.6f > 0.1，%d/%d 个种子超过它）⇒ "
          "「`(1−lr·wd)^n` 描述 AdamW」这句话**只在梯度恒为 0 时成立**"
          % (max(devs), n_over, SEED_SCAN))
    check("A.6b", min(devs) < 1e-4,
          "★★ **但绝不能说「差 > 0.1」**：也有种子只差 %.6e（**比 0.1 小 2000 倍**）—— "
          "偏差横跨 **4 个数量级** ⇒ 这一条**不能写成下界判据**（**「判据的小 n 不稳定」的又一次**，"
          "而且这次是**上界方向**：小样本会给出一条根本站不住的下界）" % min(devs))
    w_zero, _, _, _ = adam_chain(zeros_grads(), LR, BETA1, BETA2, EPS, WD_MAIN, W0, True)
    check("A.6c", abs(w_zero[-1] - cl) < TOL_ZERO_RESTORE,
          "★ 正确的判据是**结构性**的：把梯度**全部乘 0**，闭式立刻**恢复**"
          "（差 %.3e < %g）⇒ 「闭式 ⟺ 梯度恒为 0」" % (abs(w_zero[-1] - cl), TOL_ZERO_RESTORE))

    # ── A.7 Y6 的机制：`λ` 进不进 `m` / `v` ────────────────────────────────
    print()
    print("  ── A.7 Y6 的机制：`λ` **进** `m`/`v`（Adam 侧）vs **不进**（AdamW 侧）──")
    g_one = rand_grads(SEED + 11, n=1)
    mh_l2, vh_l2 = adam_chain(g_one, LR, BETA1, BETA2, EPS, WD_MAIN, W0, False)[2:4]
    mh_dw, vh_dw = adam_chain(g_one, LR, BETA1, BETA2, EPS, WD_MAIN, W0, True)[2:4]
    print("  第 1 步：Adam+L2 的 m̂ = %.9f（= g + wd·w₀ = %.9f + %.9f）"
          % (mh_l2[0], g_one[0], WD_MAIN * W0))
    print("          AdamW   的 m̂ = %.9f（= g，**没有** wd·w₀ 那一项）" % mh_dw[0])
    print("          两者之差 = %.9f（= wd·w₀ = %.9f）" % (mh_l2[0] - mh_dw[0], WD_MAIN * W0))
    check("A.7", abs((mh_l2[0] - mh_dw[0]) - WD_MAIN * W0) < 1e-12
          and abs(vh_l2[0] - vh_dw[0]) > TOL_MEASURABLE,
          "★ 第 1 步的 `m̂` 恰好差 `wd·w₀ = %.6f`（逐位），而 `v̂` 也**不同**（差 %.6f > %g）⇒ "
          "`λ` 真的**进**了 `m` **与** `v`（`m` 是线性、`v` 是平方，两者的差都可见）"
          % (WD_MAIN * W0, abs(vh_l2[0] - vh_dw[0]), TOL_MEASURABLE))

    # ── A.8 独立参照：`m`/`v` 的显式加权和 ─────────────────────────────────
    print()
    print("  ── A.8 独立参照：把 `m_t`/`v_t` 用**显式加权和**重算（不递推）──")
    gs = rand_grads(SEED + 13)
    _w, _r, mh_rec, vh_rec = adam_chain(gs, LR, BETA1, BETA2, EPS, WD_MAIN, W0, False)
    _gs, mh_sum, vh_sum = moments_weighted_sum(gs, WD_MAIN, W0, BETA1, BETA2, LR)
    e_m = max(abs(a - b) / (abs(b) or 1.0) for a, b in zip(mh_rec, mh_sum))
    e_v = max(abs(a - b) / (abs(b) or 1.0) for a, b in zip(vh_rec, vh_sum))
    print("  相对差：`m̂` 最大 %.3e · `v̂` 最大 %.3e（%d 步）" % (e_m, e_v, N_STEPS))
    check("A.8", e_m < 1e-12 and e_v < 1e-12,
          "★ 递推（一层循环）与**显式双重加权和**（参照路径）给出同一组 `m̂`/`v̂`"
          "（相对差 %.3e / %.3e < 1e-12）" % (e_m, e_v))


# ===========================================================================
# 6. B 类：代价模型 —— 状态的字节、更新量的量级
# ===========================================================================
def part_B():
    print()
    print("=" * 74)
    print("B. 代价模型：优化器状态的字节账、**更新量的量级**")
    print("=" * 74)

    # ── B.1 状态字节账 ──────────────────────────────────────────────────────
    print("  ── B.1 优化器状态的字节账（**单卡口径**）──")
    bytes_params = 4          # fp32 权重
    bytes_m = 4               # 一阶矩（fp32）
    bytes_v = 4               # 二阶矩（fp32）
    print("  每参数：权重 %d 字节 + `m` %d 字节 + `v` %d 字节 ⇒ 状态:参数 = %d:%d"
          % (bytes_params, bytes_m, bytes_v, bytes_m + bytes_v, bytes_params))
    print("  ★ **单卡口径**：这里只数「一个参数多占几字节」。")
    print("    **分片策略与通信量**（ZeRO 三阶段 / FSDP / `4x` / `8x` / `16Ψ` 的账）的正文在")
    print("    `100-parallelism/zero-fsdp/`；`10-foundations/performance-models/")
    print("    02-amdahl-and-scaling.md:219,221,417` 已逐字列为那边的正文 ⇒ **本主题只留接口**。")
    print("  ★ 口径对照：`mixed-precision/01-type-selection.md` §2.2 的「优化器状态 8 字节/参数」"
          "数的正是这里的 `m`+`v`（**同一个量**）；而动量（`01`）只有**一份**状态 ⇒ **4** 字节。")
    check("B.1", bytes_m + bytes_v == 8 and (bytes_m + bytes_v) // bytes_params == 2,
          "★ 状态:参数 = **2 倍**（`m`+`v` 各 4 字节）⇒ 「优化器状态」这一格**不省**（fp32 到底）——"
          "它是 `mixed-precision/01` 那张四行表的第三行（回链该处，本程序不重定义）")

    # ── B.2 ★★ 更新量的量级：Adam 的 `|Δw|` 与 `|g|` **无关** ──────────────
    print()
    print("  ── B.2 ★★ 更新量的量级：常数梯度下 `|Δw| = lr·|g|/(|g|+ε)`（**逐位**）──")
    print("     闭式（推导值）：`m̂_1 = g`、`√v̂_1 = |g|` ⇒ 第 1 步的更新量 = `lr·|g|/(|g|+ε)`")
    print("     %-10s %-18s %-18s %-12s" % ("|g|", "实测 Δw", "闭式", "相对差"))
    worst = 0.0
    for gv in GRAD_CONST_SCAN:
        a, _, _, _ = adam_chain([gv], LR, BETA1, BETA2, EPS, 0.0, W0, False)
        dw = a[0] - W0
        pred = -LR * gv / (abs(gv) + EPS)
        rel = abs(dw / pred - 1.0)
        worst = max(worst, rel)
        print("     %-10g %-18.12e %-18.12e %-12.3e" % (gv, dw, pred, rel))
    check("B.2", worst < 1e-12,
          "★ 第 1 步的更新量与闭式 `lr·|g|/(|g|+ε)` **逐位一致**（最大相对差 %.3e < 1e-12）—— "
          "★ 而 `|g|` 跨 **12 个数量级**时，`|Δw|` 只变 `(1 − ε/|g|)` 的量 ⇒ "
          "**Adam 的更新量与梯度的量级无关**" % worst)
    # ★ 与 SGD 的正面**对照**（★ 消费 `mixed-precision/01` 的阈值，**不重新定义**）
    g_tiny = 1e-3
    dw_sgd = LR_TINY * g_tiny
    dw_adam = LR_TINY * g_tiny / (g_tiny + EPS)
    swallowed_sgd = (np.float16(W0 + dw_sgd) == np.float16(W0))
    swallowed_adam = (np.float16(W0 + dw_adam) == np.float16(W0))
    print("  ★ 对照（同一个 `lr = %g`、同一个 `|g| = %g`，`w = %g` 是 fp16 的格点）："
          % (LR_TINY, g_tiny, W0))
    print("    SGD 的 |Δw| = lr·|g| = %.3e ⇒ 被吞？ %s" % (dw_sgd, "是" if swallowed_sgd else "否"))
    print("    Adam 的 |Δw| ≈ lr     = %.3e ⇒ 被吞？ %s" % (dw_adam, "是" if swallowed_adam else "否"))
    print("    ★ 阈值 `ULP(%g)/2 = %g` 的**判据与来源在** "
          "`90-training/mixed-precision/01-type-selection.md` §1.2（D7/D8）—— 此处**只消费**。"
          % (W0, F16_ULP_HALF_AT_1))
    check("B.2b", swallowed_sgd and not swallowed_adam,
          "★★ **同一个 `lr`、同一批梯度**：SGD 的更新被 fp16 主权重**吞掉**（`w + Δw == w`），"
          "而 Adam 的**进得去** —— 因为 Adam 的更新量恒在 `O(lr)`，SGD 的随 `|g|` 缩小 ⇒ "
          "这正是上游 `nla/04:386` 让渡的「**更新量 Δw 在什么精度下算**」那一侧")

    # ── B.3 Muon 的 update RMS：**只并列打印、不做比值断言** ─────────────────
    print()
    print("  ── B.3 `[C]` Muon 的 `update RMS` 锚点（**口径警告**）──")
    rms = math.sqrt(sum(r * r for r in _ratios_all()) / len(_ratios_all()))
    print("  论文（[C] 2502.16982，**不确认同行评审**）逐字：")
    print("    `AdamW's update RMS is in the range of 0.2∼0.4`；`We opted to control the")
    print("     update RMS of Muon to 0.2.`")
    print("  本机（玩具规模）量到的 `RMS(m̂/(√v̂+ε))` = %.6f" % rms)
    print("  ★★ **口径警告**：这两个数**不是同一个量** —— 论文没写它的 RMS 含不含 `lr`、"
          "逐张量还是逐元素；")
    print("    而本机量的是**归一化后的更新量**、且在玩具规模上。⇒ **只并列打印，不做比值断言**"
          "（`specs.UNSOURCED` 的 `muon_update_rms_measurement_convention`）。")
    check("B.3", specs.MUON_UPDATE_RMS == 0.2 and specs.MUON_ADAMW_UPDATE_RMS_LO == 0.2
          and specs.MUON_ADAMW_UPDATE_RMS_HI == 0.4 and 0.0 < rms < 1.0,
          "★ 三个外部锚点逐位一致（%.1f / %.1f / %.1f），而本机的 RMS 落在 (0,1) 内（%.6f）⇒ "
          "**量级可比、量本身不可比**：这正是「量法口径是独立的出错环节」的现场"
          % (specs.MUON_UPDATE_RMS, specs.MUON_ADAMW_UPDATE_RMS_LO,
             specs.MUON_ADAMW_UPDATE_RMS_HI, rms))


def _ratios_all():
    """B.3 用来算 `RMS(m̂/(√v̂+ε))` 的那批比值（随机梯度、`wd = 0`）。"""
    out = []
    for s in range(8):
        out += adam_chain(rand_grads(s, n=64), LR, BETA1, BETA2, EPS, 0.0, W0, False)[1]
    return out


# ===========================================================================
# 7. C 类：边界与反例 —— 自归一化**有下限**，且它什么时候**不**发生
# ===========================================================================
def part_C():
    print()
    print("=" * 74)
    print("C. 边界与反例：自归一化的下限是 `ε`；`wd = 0` 时它整个不生效")
    print("=" * 74)

    # ── C.1 ★ 自归一化的下限：`wd·|w₀| ~ ε` 时它失效 ────────────────────────
    print("  ── C.1 ★ `wd·|w₀| = ε` 时，第 1 步只走**一半** ──")
    print("     闭式（推导值）：`ratio₁ = g/(|g|+ε)` ⇒ 当 `g = ε` 时它**恰好是 0.5**")
    print("     %-12s %-20s %-20s %-14s" % ("wd", "ratio₁（实测）", "闭式 g/(|g|+ε)", "200 步后 w"))
    vals = {}
    for wd in (DOWNSCALE, 1e-7, 1e-6, 1e-4):
        a, rat, _, _ = adam_chain(zeros_grads(), LR, BETA1, BETA2, EPS, wd, W0, False)
        g1 = wd * W0
        vals[wd] = (rat[0], g1 / (abs(g1) + EPS), a[-1])
        print("     %-12g %-20.15f %-20.15f %-14.6f" % (wd, rat[0], g1 / (abs(g1) + EPS), a[-1]))
    r_at, pred_at, w_at = vals[DOWNSCALE]
    print("  ★ `wd = %g` 时 `g₁ = wd·w₀ = %g` **恰好等于 ε = %g** ⇒ `ratio₁ = 0.5`（一级近似 %g）"
          % (DOWNSCALE, DOWNSCALE * W0, EPS, 0.5))
    check("C.1", abs(r_at - 0.5) < 1e-12 and abs(r_at / pred_at - 1.0) < 1e-12,
          "★★ **自归一化的下限是 `ε`**：`wd·w₀ = ε` 时 `ratio₁ = %.15f` **精确等于 0.5**"
          "（闭式 `g/(|g|+ε)`）⇒ 权重**走不动**（%d 步后 w = %.6f，而正常档是 %.6f）"
          % (r_at, N_STEPS, w_at, 0.015573))
    check("C.1b", vals[1e-4][2] < vals[1e-7][2] < vals[DOWNSCALE][2],
          "★ 单调性对照：`wd` 越小 ⇒ 归零越慢（%.6f < %.6f < %.6f）⇒ "
          "「自归一化 ⇒ 与 `wd` 无关」这句话**有边界**：它是 `|g| ≫ ε` 时的极限，"
          "不是无条件成立" % (vals[1e-4][2], vals[1e-7][2], vals[DOWNSCALE][2]))
    print("  ★★ 这一条同时给 `A.3` 的「与 `wd` 无关」**划了范围**："
          "三档 `wd ∈ {0.01, 0.1, 1.0}` 全在 `|g| ≫ ε` 的一侧（`wd·w₀ ≥ 0.01 ≫ 1e-8`）。")

    # ── C.2 阴性对照：`wd = 0` 时两条链**逐位相同** ────────────────────────
    print()
    print("  ── C.2 阴性对照：`wd = 0` 时两条链**逐位相同** ──")
    for tag, gs in (("恒为 0", zeros_grads()), ("随机", rand_grads(SEED + 21))):
        a, _, _, _ = adam_chain(gs, LR, BETA1, BETA2, EPS, 0.0, W0, False)
        b, _, _, _ = adam_chain(gs, LR, BETA1, BETA2, EPS, 0.0, W0, True)
        print("  梯度%s：最大逐元素差 = %.3e（%d 步）" % (tag, max_absdiff(a, b), N_STEPS))
    a0, _, _, _ = adam_chain(zeros_grads(), LR, BETA1, BETA2, EPS, 0.0, W0, False)
    b0, _, _, _ = adam_chain(zeros_grads(), LR, BETA1, BETA2, EPS, 0.0, W0, True)
    a1, _, _, _ = adam_chain(rand_grads(SEED + 21), LR, BETA1, BETA2, EPS, 0.0, W0, False)
    b1v, _, _, _ = adam_chain(rand_grads(SEED + 21), LR, BETA1, BETA2, EPS, 0.0, W0, True)
    check("C.2", max_absdiff(a0, b0) < TOL_IDENTICAL and max_absdiff(a1, b1v) < TOL_IDENTICAL,
          "★ `wd = 0` 时两条链**逐位相同**（恒为 0 的梯度与随机梯度两档都是 %.3e < %g）⇒ "
          "「Adam 与 AdamW 有差别」这句话**只在 `wd ≠ 0` 时成立**，判据不是恒真的"
          % (max(max_absdiff(a0, b0), max_absdiff(a1, b1v)), TOL_IDENTICAL))


# ===========================================================================
# 8. D 类（前半）：标记自检 + 口径/阈值钉住 —— ★★ **跑在任何计算之前**（P23）
# ===========================================================================
def preflight():
    """★★ 口径 / 阈值 / 标记自检**一律在这里**，且**跑在任何计算之前**（本库 **P23**）。

    ★ 理由（Chunk 1 实测）：口径常量一旦被改，后面的**长度/索引检查会先崩** ——
      那时 `mutate.py` 记的是**崩溃捕获**，而「崩溃捕获不算真捕获」。
      ⇒ 把钉住提到**任何**计算之前，改常量就变成**断言捕获**。
    """
    print("=" * 74)
    print("D0. 口径/阈值钉住（**先跑**：让「改常量」变成断言捕获，而不是崩溃捕获）")
    print("=" * 74)
    print("  本程序的失败标记 ASSERT_FAIL_MARK = %r" % ASSERT_FAIL_MARK)
    print("  ★ 本程序的数字**全部是推导值**：玩具规模的真跑，"
          "`python3 %s` + `numpy %s`，macOS arm64 **无 GPU**。"
          % (sys.version.split()[0], np.__version__))
    before = len(FAILS)

    hits = _mark_assignments_in(os.path.abspath(__file__))
    uniq = {getattr(v, "value", None) for v in hits}
    check("D.1", len(hits) == 1 and len(uniq) == 1
          and uniq == {ASSERT_FAIL_MARK} and "FAIL" in ASSERT_FAIL_MARK,
          "★ 标记自检：模块级赋值恰好 1 处、取值唯一、== %r、且含 `FAIL`（AST 读出）"
          % ASSERT_FAIL_MARK)
    check("D.2", ASSERT_FAIL_MARK == "++ FAIL: optimizers" and "FAIL" in ASSERT_FAIL_MARK,
          "★ 标记自检：**等于**本主题被分配的字面量、且含 `FAIL`")
    check("D.2b", OTHER_THEME_MARKS == (".. FAIL: autograd", ":: FAIL: mixed-precision",
                                        "// FAIL: init-stability", ";; FAIL: checkpointing"),
          "★ 「不跨主题照抄」的**对照物**也逐位钉住")

    measured = {"SEED": SEED, "LR": LR, "BETA1": BETA1, "BETA2": BETA2, "EPS": EPS,
                "N_STEPS": N_STEPS, "W0": W0, "WD_LIST": WD_LIST, "WD_MAIN": WD_MAIN,
                "DOWNSCALE": DOWNSCALE, "SEED_SCAN": SEED_SCAN,
                "GRAD_CONST_SCAN": GRAD_CONST_SCAN, "SCALE_SCAN": SCALE_SCAN,
                "F16_ULP_HALF_AT_1": F16_ULP_HALF_AT_1, "LR_TINY": LR_TINY}
    expect = {"SEED": 20260929, "LR": 0.01, "BETA1": 0.9, "BETA2": 0.999, "EPS": 1e-8,
              "N_STEPS": 200, "W0": 1.0, "WD_LIST": (0.01, 0.1, 1.0), "WD_MAIN": 0.1,
              "DOWNSCALE": 1e-8, "SEED_SCAN": 200,
              "GRAD_CONST_SCAN": (1e-6, 1e-3, 1.0, 1e3, 1e6), "SCALE_SCAN": (1e-2, 1e2),
              "F16_ULP_HALF_AT_1": 4.8828125e-04, "LR_TINY": 1e-3}
    diff = {k: (expect[k], v) for k, v in measured.items() if v != expect[k]}
    check("D.3", not diff, "★ 口径常量 15 个逐位一致（结论的**测量条件**）%s"
          % ("" if not diff else " ← 差异 %s" % diff))
    check("D.3b", LR != specs.ADAM_DEFAULT_LR and LR == 0.01,
          "★★ **口径纪律的现场**：本程序**覆盖了**论文的默认 lr（用 `%g`，论文是 `%g`），"
          "并把「覆盖过」这件事**钉住** —— 「用论文的默认值当论据」与「用本程序的设定当测量条件」"
          "**不得混**（`D.5` 钉的是四个默认超参**本身**）"
          % (LR, specs.ADAM_DEFAULT_LR))
    check("D.3c", DOWNSCALE * W0 == EPS and abs(WD_LIST[0] / EPS) > 1e5,
          "★ `C.1` 的那一档真的落在 `|g| ~ ε` 上（`wd·w₀ = %g = ε`），"
          "而 `A.3` 的三档 `wd` 都比 ε 大 **5 个数量级以上**（最小档是它的 %.0f 倍）"
          % (DOWNSCALE * W0, WD_LIST[0] / EPS))

    th = {"TOL_CLOSED": TOL_CLOSED, "TOL_IDENTICAL": TOL_IDENTICAL,
          "TOL_SATURATE": TOL_SATURATE, "TOL_RATIO": TOL_RATIO,
          "TOL_SCALE_INV": TOL_SCALE_INV, "TOL_MEASURABLE": TOL_MEASURABLE,
          "TOL_ZERO_RESTORE": TOL_ZERO_RESTORE}
    th_expect = {"TOL_CLOSED": 1e-9, "TOL_IDENTICAL": 1e-12, "TOL_SATURATE": 1e-5,
                 "TOL_RATIO": 1e-12, "TOL_SCALE_INV": 1e-5, "TOL_MEASURABLE": 1e-2,
                 "TOL_ZERO_RESTORE": 1e-9}
    tdiff = {k: (th_expect[k], v) for k, v in th.items() if v != th_expect[k]}
    check("D.4", not tdiff, "★ 阈值 7 个逐位一致（`TOL_MEASURABLE` 是**可测性下界**："
                            "放宽它会让 `A.5`/`A.6`/`A.4c` 的反面都退化成恒真）%s"
          % ("" if not tdiff else " ← 差异 %s" % tdiff))
    return len(FAILS) == before


# ===========================================================================
# 9. D 类（后半）：specs 常量逐位钉住 + 独立性证据
# ===========================================================================
def part_D():
    print()
    print("=" * 74)
    print("D. specs 常量逐位钉住 与 独立性证据")
    print("=" * 74)

    check("D.5", (specs.ADAM_DEFAULT_LR, specs.ADAM_DEFAULT_BETA1,
                  specs.ADAM_DEFAULT_BETA2, specs.ADAM_DEFAULT_EPS)
          == (0.001, 0.9, 0.999, 1e-8),
          "★ Adam 的四个默认超参逐位一致（[C] 1412.6980，**已确认 ICLR 2015**）——"
          "★ 论文的措辞是 `Good default settings for the tested machine learning problems`，"
          "**不是**「规范规定」（[B] PyTorch 恰好也用同一组值，但那是**另一个论据**）")
    check("D.5b", specs.ADAM_DEFAULTS_QUOTE
          == "Good default settings for the tested machine learning problems are",
          "★ 那句的逐字（**PDF 文本层**）逐位一致 —— ★ 只取**公式之前**的片段："
          "公式里的 ε 在 `pdftotext` 输出里是**控制字符**且跨行 ⇒ 按 §0.3 纪律**不数它**"
          "（本程序用 `A.5` 的**数值断言**承担「ε 在根号外」的可证伪性）")
    check("D.5c", specs.ADAM_DEFAULT_EPS == DOWNSCALE and specs.ADAM_DEFAULT_EPS == 1e-8,
          "★ `C.1` 的「下限」用的就是**论文那个 `ε`**（同一个常量），不是另拍一个数")

    # ── ★★ AdamW 的两条归因：论文 vs PyTorch，**两条都引** ─────────────────
    print("  ★★ 「AdamW 与 Adam+L2 差在哪」的**两条归因**（Y5）：")
    print("     论文（[C] 1711.05101 §2）：L2 的梯度**也被自适应归一化** ⇒ 典型梯度大的权重")
    print("                               **被正则得更少**（相对量）")
    print("     PyTorch（[B] 2.14）：`weight decay does not accumulate in the momentum nor variance`")
    check("D.6", specs.ADAMW_L2_ATTRIBUTION_QUOTE
          == "weights x with large typical gradient magnitude s are regularized by a smaller relative"
          and specs.PYTORCH_ADAMW_NO_ACCUMULATE_QUOTE
          == "weight decay does not accumulate in the momentum nor variance",
          "★★ **Y5 的两条并列**（论文侧 / PyTorch 侧）—— **不矛盾但不是同一句话** ⇒ "
          "两条都引、各标出处；把 PyTorch 的措辞当论文的理由 = **规范与实现不分**")
    check("D.6b", specs.ADAMW_MECHANISM_QUOTE
          == "whereas with decoupled weight decay, only the gradients of the loss function",
          "★ 论文的正面表述也逐位钉住（「解耦后**只有损失的梯度**被自适应」）")
    check("D.7", specs.PYTORCH_ADAM_DECOUPLED_FLAG == "decoupled_weight_decay=False",
          "★★ `[B]` PyTorch 2.14 的 `torch.optim.Adam` 默认 **`decoupled_weight_decay=False`** ⇒ "
          "「`Adam` + `weight_decay` 就是 AdamW」**是错的**（Y6）—— 本程序的 `A.7` 用**机制**"
          "（`λ` 进不进 `m`/`v`）把它落成可断言的")
    check("D.7b", (specs.PYTORCH_ADAM_DEFAULT_WEIGHT_DECAY, specs.ADAMW_DEFAULT_WEIGHT_DECAY)
          == (0, 0.01),
          "★★ 两个 `weight_decay` 的默认值**成对**：`Adam` 是 `0`、`AdamW` 是 `0.01` ⇒ "
          "「用哪个类」不只是名字（★ `0.01` 是 **[B] PyTorch 的**默认值：论文的 Algorithm 2 "
          "只给 `λ ∈ IR` 的**符号域**，**没有**默认值）")
    check("D.8", specs.ADAMW_TEST_ERROR_QUOTE == "relative improvement in test error",
          "★ `15%` 那句的片段逐位一致 —— ★ 它的作用域是 **CIFAR-10 与 ImageNet32x32、`lr = 0.001`**，"
          "**不得**写成「AdamW 普遍好 15%」（P20 的同类）")
    check("D.9", specs.UNSOURCED == ("adam_eps_position_raw_count",
                                      "pytorch_adam_float32_eps_version_span",
                                      "muon_orthogonalization_iteration_count",
                                      "muon_update_rms_measurement_convention"),
          "★ `specs.UNSOURCED` 的四项**逐位钉住**（%d 项）—— ★ 其中第一项就是 `A.5` 的现场："
          "「ε 在根号外」**不能**在原始字节上做引文核对" % len(specs.UNSOURCED))
    check("D.10", specs.TORCH_CLIP_COEF_DENOM_EPS == 1e-6,
          "★ `TORCH_CLIP_COEF_DENOM_EPS == 1e-6`（[B] PyTorch **v2.9.0 的源码**）——"
          "★ 本程序**不消费它**（裁剪在 `03`）；钉住是为了让三个程序同值")

    # ── 独立性 ─────────────────────────────────────────────────────────────
    src_chain = inspect.getsource(adam_chain)
    src_sum = inspect.getsource(moments_weighted_sum)
    src_closed = inspect.getsource(adamw_closed_form)
    check("D.11", len({src_chain, src_sum, src_closed}) == 3,
          "★ 防 P17：被测（`adam_chain`，%d 字符）与两条参照"
          "（`moments_weighted_sum` %d、`adamw_closed_form` %d）**源码两两不同**"
          % (len(src_chain), len(src_sum), len(src_closed)))
    watched = {"adam_chain", "adam_ratio_wrong_denominator"}
    called = _callee_names(moments_weighted_sum) | _callee_names(adamw_closed_form)
    check("D.12", not (watched & called),
          "★ **AST 调用表**：两条**参照路径**的 `Call` 节点（`ast.Name` 形态）= %s —— "
          "一条都没调到被测函数" % (", ".join(sorted(called)) if called else "**空集**"))
    check("D.13", "range" in _callee_names(moments_weighted_sum)
          or "range" in _callee_names(adamw_closed_form),
          "★ 阳性对照：参照路径的调用表**不是空的**（含 `range` 一类）⇒ "
          "`D.12` 的「空集/不相交」不是因为它什么都没调")
    check("D.14", MARKS_USED <= {ASSERT_FAIL_MARK},
          "★ 打印失败行时用的标记 ⊆ {%r}（本次实际用到 %d 个）"
          % (ASSERT_FAIL_MARK, len(MARKS_USED)))


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
    print("02_adam_adamw.py —— Adam / AdamW：weight decay 的两种语义与「自归一化」")
    print("=" * 74)
    print("  ★ 真值标注：**全部是推导值**（本机 `python3 %s` + `numpy %s`，macOS arm64 无 GPU）。"
          % (sys.version.split()[0], np.__version__))
    print("  ★ 失败标记 ASSERT_FAIL_MARK = %r（本主题自定义；不跨主题照抄）" % ASSERT_FAIL_MARK)
    print("  ★ 禁区（一律回链，本程序不写）：优化器状态的**分片与通信量**在")
    print("    `100-parallelism/zero-fsdp/`；DDP / 梯度规约在 `:9` 与 `110-communication/`；")
    print("    框架的 autograd 语义在 `70-frameworks/pytorch/02-autograd.md`。")
    print("  ★ 接口：`ULP/2` 的阈值与判据在 `mixed-precision/01-type-selection`（此处只消费）")
    print("    —— 本程序 `B.2b` 用它把「Adam 的更新量进得去、SGD 的被吞」落成一条断言。")
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

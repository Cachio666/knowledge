#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""01_sgd_momentum.py —— SGD / 动量 / Nesterov 的更新式与**状态**。

配套文档：`01-sgd-and-momentum.md`（★ **文件名由上游指名**：
`70-frameworks/pytorch/02-autograd.md` 的 frontmatter 逐字写着
`90-training/optimizers/01-sgd-and-momentum` —— **全库唯一一处「编号级」让渡**）。

★ 三类断言（A 算法正确性 / B 代价模型 / C 边界与反例）**全部在本程序内部**，
失败时以**非零退出码**结束；**没有 `verify.py`**（`CONTRIBUTING` §2 表里的第二种情形）。

★★ **本程序的数字全部是推导值**，测量条件是：
> `python3 3.13.5` + `numpy 2.4.4`，macOS arm64，**无 GPU**。
> 玩具规模的真跑，**不得支撑关于真实训练系统的结论**。

★★ **本主题的禁区**（`CONTRIBUTING` §7 + 域 README 的边界表）：
  · **优化器状态的分片策略与通信量**（ZeRO 三阶段、FSDP、`4x/8x`、`16Ψ` 的账）→
    `100-parallelism/README.md:14`（`zero-fsdp` 那一栏）；`10-foundations/performance-models/
    02-amdahl-and-scaling.md:219,221,417` 已逐字列为那边的正文。**本程序只写单卡口径**；
  · **数据并行 / DDP / 梯度规约** → `100-parallelism/README.md:9`（`data-parallel`）、`110-communication/`；
  · **`lr` 被编译 guard 固化** → `60-compilers/inductor/04-dynamic-shapes.md:42,163`（编译角度）；
  · **框架的 autograd 语义**（`grad_fn` / `is_leaf` / `saved_tensors`）→
    `70-frameworks/pytorch/02-autograd.md`（**全篇 256 行**）—— 本程序一行都不重写。
  ⇒ **本程序一行都不重写**这些。

★★ **与本域其它主题的接口**（回链、不重定义）：
  · 「`Δw < ULP/2` 时更新被静默吞掉」这个**阈值与判据**在
    `90-training/mixed-precision/01-type-selection.md` 的 §1.2 /
    `code/01_types.py` 的 `C.1`（计划 §0.7-D 的 **D8**）—— 本程序**只消费**它，
    用它算「SGD 的更新量什么时候掉进那个阈值以下」（`B.2`）；
  · 「**更新量 Δw 在什么精度下算**」是上游 `10-foundations/numerical-linear-algebra/
    04-iterative-methods.md:386` 指名让渡给本主题的那一侧（本主题的第一篇是它的落点）。

★★ 可证伪断言（把下面任一条改错，本程序必须非零退出）：
  · `A.1`  动量的**递推** == **闭式** `Σ μ^k g`（逐元素差 < 1e-12）；
  · `A.1b` 反面对照：把递推写成 `b ← μ b + (1−μ) g`（把 Adam 的 `(1−β)` 搬到动量上）
           ⇒ 与闭式**差 > 1e-3**（所以 `A.1` **不是恒真**）；
  · `A.2`  重球与纯 SGD 的**轨迹差有闭式** `−lr·Σ_s Σ_{k<s} μ^{s−k} g_k`（相对差 < 1e-12）；
  · `A.3`  ★★ **判据⑤**：PyTorch 的 `nesterov` 两支（`g + μ b_t` 与 `b_t`）在**同一个 `g` 序列**上
           给出**不同**的 θ（差 > 1e-3），且差**等于闭式** `lr·Σ_t (g_t − (1−μ) b_t)`；
  · `A.3b` 而「用**更新前**的 buffer」（`g + μ b_{t−1}`）与**重球逐位相同** ⇒ **它不是第三种写法**；
  · `A.3c` ★★ 「**先走一步**」与「**在偏移点算梯度**」这两种记账**逐位相同**（< 1e-12）——
           它们是**同一个算法**，**不能拿来当对照**；真正的对照是「在**当前点**求梯度」（差 > 1e-3）；
  · `B.1`  把动量 buffer 从 fp32 换成 fp16 ⇒ **状态字节减半**，代价是轨迹偏离 **> 1e-3**；
  · `B.2`  更新量的量级：`lr·|g| < ULP(1.0)/2` 时 `w + Δw == w` **逐位**（阈值回链 `mixed-precision/01`）；
  · `C.1`  SGD 的发散边界**精确**是 `lr·λ = 2`（`= 2` 时 `|w|` 不衰减也不发散，略大即发散）；
  · `C.2`  动量把这个边界**放大**到 `lr·λ = 2(1+μ)`（实测 μ=0.5 → 3.0、μ=0.9 → 3.8）。

用法：
    python3 01_sgd_momentum.py
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
SEED = 20260928              # ★ 与同域另外两个程序（20260929 / 20260930）都**不同**
N_PARAMS = 32                # 一个「参数张量」的元素个数
N_STEPS = 20                 # 递推步数
LR = 0.1                     # 学习率
MU = 0.9                     # 动量系数（β₁ 的那个 0.9，但**角色不同**：见 `A.1b`）
W0 = 0.0                     # 权重的初值（零起点，让闭式只有一项）
LAM = 1.0                    # 二次型 `f(w) = ½λ‖w‖²` 的 `λ`（只给 `A.3c` 用）
W0_LOOKAHEAD = 1.0           # `A.3c` 的起点（非零，好让两种记账的差看得出来）

TOL_CLOSED = 1e-12           # A.1 / A.2 / A.3：与闭式的逐元素最大差
TOL_SAME_ALGO = 1e-12        # A.3c：两种**记账**的差（它们是同一个算法）
TOL_DIVERGE = 1e-3           # A.1b / A.3 / A.3c 的**反面下界**：差必须**超过**它
TOL_BUFFER_FP16 = 1e-3       # B.1：fp16 buffer 造成的轨迹偏离下界
TOL_BITWISE = 0.0            # A.3b：逐位（容差 0）
MIN_REGION = 1e-3            # ★ P22：参与「机器精度档相等」判据的量必须 ≥ 它
                             #   （远大于次正规下界 2.2e-308 ⇒ 那个「相等」不是次正规噪声）
# ★★ 更新被吞的阈值：`ULP(1.0)/2` —— ★ **本程序只消费它，不重新定义**。
#   来源与判据在 `90-training/mixed-precision/01-type-selection.md` §1.2
#   （`code/01_types.py` 的 `A.1`/`C.1`，计划 §0.7-D 的 **D7/D8**）。
#   值 = `2^⌊log2 1⌋ · 2^-10 / 2` = `2^-11`。写成**十进制字面量**（幂次进注释）：
F16_ULP_HALF_AT_1 = 4.8828125e-04   # = 2^-11
TINY_GRAD = 1e-3             # B.2：一个「小但正常」的梯度
LR_TINY = 1e-4               # B.2：一个「小」的学习率（让 `lr·g < ULP/2`）
LR_BIG = 1e-3                # B.2：同一条轨迹上「够大」的那个 lr（让 `lr·g < ULP/2` 仍成立）


# ===========================================================================
# 2. 路径 1（被测）：递推式的两种直接实现
# ===========================================================================
def sgd_plain(w, grads, lr):
    """纯 SGD：`w ← w − lr·g`（无状态）。"""
    w = np.array(w, dtype=np.float64)
    for g in grads:
        w = w - lr * g
    return w


def momentum_recursion(w, grads, lr, mu):
    """**重球**（heavy ball）的递推：`b ← μ b + g`；`w ← w − lr b`。

    ★ 返回 `(w, b_trace)`；`b_trace[t]` 是第 `t` 步**更新后**的 buffer ——
      闭式（路径 2）对的就是它。
    """
    w = np.array(w, dtype=np.float64)
    b = np.zeros_like(w)
    trace = []
    for g in grads:
        b = mu * b + g
        w = w - lr * b
        trace.append(b.copy())
    return w, trace


def momentum_scaled_wrong(w, grads, lr, mu):
    """★★ **反面对照**：把 buffer 写成 `b ← μ b + (1−μ) g`。

    那是 Adam 的 `(1−β₁)` 形式被**误搬**到动量上的结果 —— 两者**不是同一个递推**：
    动量缓冲区的定义里**没有** `(1−μ)` 这个因子（`[B]` PyTorch 的 SGD 伪代码逐字是
    `b_t ← μ b_{t−1} + (1−τ) g_t`，那个 `(1−τ)` 是 **dampening**，默认 `τ = 0`）。
    """
    w = np.array(w, dtype=np.float64)
    b = np.zeros_like(w)
    for g in grads:
        b = mu * b + (1.0 - mu) * g
        w = w - lr * b
    return w


def nesterov_traj(w, grads, lr, mu, mode):
    """★ `[B]` PyTorch 的 `nesterov` 两支（**在同一个 `g` 序列上**）。

    逐字（PyTorch 2.14 的 `torch.optim.SGD` 文档，剥标签后同一段内，见 `specs.py`）：
    ```
    b_t ← μ b_{t−1} + (1−τ) g_t          （τ = dampening，默认 0）
    if nesterov:  g_t ← g_t + μ b_t
    else:         g_t ← b_t
    θ_t ← θ_{t−1} − γ g_t
    ```
    `mode ∈ {"nesterov", "heavy", "old_buffer"}` —— ★ **三支的 buffer 递推完全相同**，
    唯一区别是**更新里用哪一支方向**：这样「在同一个 `g` 序列上比较」这件事才严格成立。
      · `"nesterov"`：`g_t + μ·b_t`（PyTorch 的 `nesterov=True`）；
      · `"heavy"`：`b_t`（`nesterov=False`，即重球）；
      · `"old_buffer"`：`g_t + μ·b_{t−1}`（★ 用**更新前**的 buffer —— 一种常见写法）。
    """
    w = np.array(w, dtype=np.float64)
    b = np.zeros_like(w)
    for g in grads:
        if mode == "old_buffer":
            w = w - lr * (g + mu * b)      # ← 用**更新前**的 buffer
            b = mu * b + g
        else:
            b = mu * b + g
            if mode == "nesterov":
                w = w - lr * (g + mu * b)
            else:                           # "heavy"
                w = w - lr * b
    return w


def nesterov_bookkeeping(w, v, lr, mu, lam, mode):
    """★★ `A.3c`：Nesterov 动量的**两种记账**，用**同一个可微函数** `f(w) = ½λ‖w‖²`。

    ```
    记账①「先走一步」  ： y ← w − lr·μ·v;  g ← ∇f(y);  w ← y − lr·g;      v ← μ·v + g
    记账②「在偏移点算梯度」： y ← w − lr·μ·v;  g ← ∇f(y);  w ← w − lr·(μ·v + g); v ← μ·v + g
    记账③「在当前点」  ： g ← ∇f(w);       v ← μ·v + g;                    w ← w − lr·v
    ```
    ★★ **①②是同一个算法的两种记账**（代数上 `y − lr·g = w − lr μ v − lr g = w − lr(μ v + g)`）
      ⇒ 它们**逐位相同**，**不能拿来当对照**；**③才是另一种算法**（重球）。
      这正是计划 §3 表 A ⑤ 那个括号（「先走一步 vs 在偏移点算梯度」）的正确读法。
    """
    w = np.array(w, dtype=np.float64)
    v = np.array(v, dtype=np.float64)
    for _ in range(N_STEPS):
        if mode == "current":
            g = lam * w
            v = mu * v + g
            w = w - lr * v
        else:
            y = w - lr * mu * v
            g = lam * y
            if mode == "lookahead":
                w = y - lr * g
            else:                            # "offset"
                w = w - lr * (mu * v + g)
            v = mu * v + g
    return w


# ===========================================================================
# 3. 路径 2（参照）：**闭式**（显式双重求和，与被测路径的调用表不相交）
# ===========================================================================
def momentum_closed_form(grads, mu):
    """★ **独立参照**：`b_t = Σ_{k=0}^{t} μ^{t−k} g_k` —— 逐项**显式求和**得到。

    ★ 它与 `momentum_recursion`（原地递推）是**两条真正不同的路径**：
      一个只有一层 `for`，另一个是**双重**求和；`D.14`/`D.15` 用
      「源码不同 + AST 调用表为空」双保险钉住这件事。
    """
    n = len(grads[0])
    trace = []
    for t in range(len(grads)):
        acc = np.zeros(n)
        for k in range(t + 1):
            acc = acc + (mu ** (t - k)) * grads[k]
        trace.append(acc)
    return trace


def momentum_minus_sgd_closed_form(grads, lr, mu):
    """★ 参照：重球轨迹与纯 SGD 轨迹之差的闭式。

    `b_s = Σ_{k<s} μ^{s−k} g_k`（`k` 只到 `s−1`，因为 `b_0 = 0`）⇒
    `w_heavy,t − w_sgd,t = −lr·Σ_{s≤t} b_s = −lr·Σ_{s≤t} Σ_{k<s} μ^{s−k} g_k`。
    """
    n = len(grads[0])
    out = []
    for t in range(len(grads)):
        corr = np.zeros(n)
        for s in range(t + 1):
            for k in range(s):
                corr = corr + (mu ** (s - k)) * grads[k]
        out.append(-lr * corr)
    return out


def nesterov_minus_heavy_closed_form(grads, lr, mu):
    """★ 参照：`nesterov` 支与 `heavy` 支之差的闭式。

    两条链的 buffer 递推**完全相同**（`b_t` 一样）⇒ 每步之差 = `−lr·(g_t + μ b_t − b_t)`
    = `−lr·(g_t − (1−μ)·b_t)`。逐项显式求和。
    """
    n = len(grads[0])
    b = np.zeros(n)
    out = []
    acc = np.zeros(n)
    for t in range(len(grads)):
        b = mu * b + grads[t]
        acc = acc - lr * (grads[t] - (1.0 - mu) * b)
        out.append(acc.copy())
    return out


# ===========================================================================
# 4. 工具
# ===========================================================================
def _callee_names(func):
    """用 **AST** 列出 `func` 真正**调用**的函数名（不含注释/文档串里的提及）。

    ★ 只收 `ast.Name` 形态的调用 ⇒ `np.zeros(...)` 这类**属性调用不计入**
      （那正是本程序两条路径的形态）⇒ 参照路径的调用表**为空**。
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


def make_grads(seed=SEED, steps=N_STEPS, n=N_PARAMS):
    """一批梯度：`N(0, 1)`，形状 `(steps, n)`。

    ★ 只用一条种子，**换种子会换数**（报数时写明种子）。
    ★★ **P22 检查**：本程序的判据全是「机器精度档的相等」（`< 1e-12`）⇒
      **必须先确认参与判据的量没有衰减到次正规**。`A.1` 会把 `min|g|` / `min|b|` 打印出来。
    """
    return np.random.default_rng(seed).standard_normal((steps, n))


def global_norm(x):
    """全局范数 —— `A.2` 报到屏幕上用的量（`C` 类不用它：那里是标量递推）。"""
    x = np.asarray(x, dtype=np.float64)
    return float(math.sqrt(float((x ** 2).sum())))


# ===========================================================================
# 5. A 类：更新式与它的闭式
# ===========================================================================
def part_A():
    grads = make_grads()
    print("=" * 74)
    print("A. 更新式与闭式（动量递推 / PyTorch 的 nesterov 两支 / 两种记账）")
    print("=" * 74)
    print("  口径：seed=%d · %d 个参数 · %d 步 · lr=%g · μ=%g · w₀=%g"
          % (SEED, N_PARAMS, N_STEPS, LR, MU, W0))

    # ── A.1 ★ 判据④：动量递推 == 闭式 `Σ μ^k g` ─────────────────────────────
    print("  ── A.1 动量 buffer 的递推 vs 闭式 `b_t = Σ_{k≤t} μ^{t−k} g_k` ──")
    _w, b_trace = momentum_recursion(np.full(N_PARAMS, W0), grads, LR, MU)
    b_closed = momentum_closed_form(grads, MU)
    diffs = [float(np.abs(np.asarray(b_trace[t]) - np.asarray(b_closed[t])).max())
             for t in range(N_STEPS)]
    worst = max(diffs)
    mag_g = float(np.abs(grads).min())
    mag_b = min(float(np.abs(b).min()) for b in b_trace)
    print("  逐元素最大差 = %.3e（最后一步 %.3e，最坏出现在第 %d 步）"
          % (worst, diffs[-1], diffs.index(worst) + 1))
    print("  ★ 参照量级（**P22**）：min|g| = %.3e · min|b_t| = %.3e · 最大 |b| = %.3e"
          % (mag_g, mag_b, max(float(np.abs(b).max()) for b in b_trace)))
    check("A.1", worst < TOL_CLOSED,
          "★ 递推与闭式**逐元素**一致（最大差 %.3e < %g）—— 这是**定义恒等式**，"
          "用 `==` 级容差钉住；改写它就会失败" % (worst, TOL_CLOSED))
    check("A.1c", mag_g >= MIN_REGION and mag_b >= MIN_REGION,
          "★ **P22 检查**：参与「机器精度档相等」的两个量都 ≥ %g（min|g| = %.3e、min|b| = %.3e）"
          "⇒ 那个「相等」**不是次正规噪声**（次正规下界是 2.2e-308，差 300 个数量级）"
          % (MIN_REGION, mag_g, mag_b))

    # ── A.1b ★ 反面对照：`(1−μ)` 那个因子不是动量的 ─────────────────────────
    print("  ── A.1b 反面对照：把 buffer 写成 `b ← μ b + (1−μ) g` ──")
    w_wrong = momentum_scaled_wrong(np.full(N_PARAMS, W0), grads, LR, MU)
    d_wrong = float(np.abs(w_wrong - _w).max())
    print("  两条轨迹的差（最大逐元素）= %.6e" % d_wrong)
    print("  ★ 依据（[B] PyTorch 2.14 的 `torch.optim.SGD` 伪代码逐字）：缓冲区那一行是")
    print("    `b_t ← μ·b_{t−1} + (1−τ)·g_t`，那个 `(1−τ)` 是 **dampening**（默认 τ=0）——")
    print("    **不是** Adam 的 `(1−β₁)`。把 Adam 的形式搬过来就换了一个递推。")
    check("A.1b", d_wrong > TOL_DIVERGE,
          "★ 反面对照：把 `(1−μ)` 搬进 buffer 后与正确轨迹相差 %.6e（> %g）⇒ "
          "`A.1` 的「递推 == 闭式」**不是恒真的**" % (d_wrong, TOL_DIVERGE))

    # ── A.2 重球与纯 SGD 的差：闭式 ─────────────────────────────────────────
    print("  ── A.2 重球 − 纯 SGD 的差（闭式 `−lr·Σ_s Σ_{k<s} μ^{s−k} g_k`）──")
    w_sgd = sgd_plain(np.full(N_PARAMS, W0), grads, LR)
    w_heavy, _ = momentum_recursion(np.full(N_PARAMS, W0), grads, LR, MU)
    diff_closed = momentum_minus_sgd_closed_form(grads, LR, MU)
    err = float(np.abs((w_heavy - w_sgd) - diff_closed[-1]).max())
    scale = float(np.abs(diff_closed[-1]).max())
    print("  ‖w_heavy − w_sgd‖∞ = %.6f · 闭式的 ‖·‖∞ = %.6f · 差 %.3e（相对 %.3e）"
          % (float(np.abs(w_heavy - w_sgd).max()), scale, err, err / (scale or 1.0)))
    print("  （口径：`global_norm` 与本处的 `max|·|` **不是同一个量** —— 前者是 2-范数、"
          "后者是 ∞-范数；判据用的是后者，两个量在本程序里不混用。）")
    print("  ‖w_heavy‖₂ = %.6f · ‖w_sgd‖₂ = %.6f"
          % (global_norm(w_heavy), global_norm(w_sgd)))
    check("A.2", err / (scale or 1.0) < TOL_CLOSED,
          "★ 动机（重球）与无动量 SGD 的**轨迹差**有闭式，相对差 %.3e < %g ⇒ "
          "两条链都被同一个闭式钉住" % (err / (scale or 1.0), TOL_CLOSED))

    # ── A.3 ★★ 判据⑤：nesterov 的两支（同一个 g 序列）───────────────────────
    print()
    print("  ── A.3 ★★ 判据⑤：`nesterov` 两支在**同一个 `g` 序列**上给出不同的 θ ──")
    print("     逐字依据（[B] PyTorch 2.14，剥标签后同一段内）：")
    print("     `if nesterovgt←gt+μbtelsegt←bt`  ⇒  `g_t + μ·b_t`  vs  `b_t`")
    w_nes = nesterov_traj(np.full(N_PARAMS, W0), grads, LR, MU, "nesterov")
    w_hea = nesterov_traj(np.full(N_PARAMS, W0), grads, LR, MU, "heavy")
    d_nes = w_nes - w_hea
    cf_nes = nesterov_minus_heavy_closed_form(grads, LR, MU)[-1]
    err_nes = float(np.abs(d_nes - cf_nes).max())
    scale_nes = float(np.abs(cf_nes).max())
    print("  ‖差‖∞ = %.6f · 闭式 ‖·‖∞ = %.6f · 与闭式的差 %.3e（相对 %.3e）"
          % (float(np.abs(d_nes).max()), scale_nes, err_nes, err_nes / (scale_nes or 1.0)))
    check("A.3", float(np.abs(d_nes).max()) > TOL_DIVERGE,
          "★ **混用会被抓住**：两支的 θ 相差 %.6f（> %g）—— 把 `nesterov` 实现成另一支，"
          "轨迹就与参照分开" % (float(np.abs(d_nes).max()), TOL_DIVERGE))
    check("A.3d", err_nes / (scale_nes or 1.0) < TOL_CLOSED,
          "★ 而那个差**有闭式** `−lr·Σ_t (g_t − (1−μ)·b_t)`（相对差 %.3e < %g）⇒ "
          "判据不是「随便两个数不相等」，是「恰好差这么多」"
          % (err_nes / (scale_nes or 1.0), TOL_CLOSED))

    # ── A.3b 「用更新前的 buffer」= 重球（逐位）──────────────────────────────
    print("  ── A.3b 反面对照：`g_t + μ·b_{t−1}`（用**更新前**的 buffer）──")
    w_old = nesterov_traj(np.full(N_PARAMS, W0), grads, LR, MU, "old_buffer")
    d_old = float(np.abs(w_old - w_hea).max())
    print("  与重球轨迹的最大差 = %.3e" % d_old)
    check("A.3b", d_old == TOL_BITWISE,
          "★★ 「用**更新前**的 buffer」与重球**逐位相同**（最大差 %.1e，容差 %g）⇒ "
          "它**不是第三种写法**，它就是 `nesterov=False` —— 一个 buffer 时序错误会让 "
          "Nesterov **静默退化成重球**" % (d_old, TOL_BITWISE))

    # ── A.3c ★★ 两种记账是同一个算法（不能当对照）───────────────────────────
    print()
    print("  ── A.3c ★★ 「先走一步」vs「在偏移点算梯度」（用同一个 `f(w) = ½λ‖w‖²`）──")
    rng = np.random.default_rng(SEED + 1)
    w_init = np.full(N_PARAMS, W0_LOOKAHEAD) + 0.1 * rng.standard_normal(N_PARAMS)
    v_init = np.zeros(N_PARAMS)
    w_la = nesterov_bookkeeping(w_init, v_init, LR, MU, LAM, "lookahead")
    w_off = nesterov_bookkeeping(w_init, v_init, LR, MU, LAM, "offset")
    w_cur = nesterov_bookkeeping(w_init, v_init, LR, MU, LAM, "current")
    d_la_off = float(np.abs(w_la - w_off).max())
    d_la_cur = float(np.abs(w_la - w_cur).max())
    rel_la_cur = d_la_cur / (float(np.abs(w_cur).max()) or 1.0)
    print("  ①「先走一步」与 ②「在偏移点算梯度」的差 = %.3e" % d_la_off)
    print("  ① 与 ③「在**当前点**求梯度」的差 = %.6f（相对 %.4f）" % (d_la_cur, rel_la_cur))
    check("A.3c", d_la_off < TOL_SAME_ALGO,
          "★★ ①②**逐位相同**（%.3e < %g）：它们是**同一个算法的两种记账**"
          "（`y − lr·g` 与 `w − lr(μ·v + g)` 代数上相等）⇒ **不能拿它们做对照** ——"
          "这正是计划 §3 表 A ⑤ 那个括号的正确读法" % (d_la_off, TOL_SAME_ALGO))
    check("A.3e", d_la_cur > TOL_DIVERGE,
          "★ 而 ①（偏移点求梯度）与 ③（**当前点**求梯度，即重球）相差 %.6f（> %g）⇒ "
          "**真正的两种算法**是「前瞻点」与「当前点」，不是那两种记账"
          % (d_la_cur, TOL_DIVERGE))


# ===========================================================================
# 6. B 类：代价模型 —— 状态的字节、更新量的量级
# ===========================================================================
def part_B():
    print()
    print("=" * 74)
    print("B. 代价模型：动量的状态占几字节、更新量掉到哪个量级")
    print("=" * 74)

    # ── B.1 动量的状态账：省一半字节的代价是轨迹偏离 ────────────────────────
    print("  ── B.1 动量 buffer 的字节账：fp32 vs fp16 ──")
    n_par = 4096
    rng = np.random.default_rng(SEED + 2)
    grads = rng.standard_normal((N_STEPS, n_par))
    w_fp32 = _buffer_traj(grads, np.float64)
    w_fp16 = _buffer_traj(grads, np.float16)
    dev = float(np.abs(w_fp32 - w_fp16).max())
    rel_dev = dev / (float(np.abs(w_fp32).max()) or 1.0)
    print("  状态字节/参数：fp32 buffer = 4 · fp16 buffer = 2 ⇒ 省 **50%%**")
    print("  轨迹偏离：最大逐元素 %.6e（相对 %.4e）" % (dev, rel_dev))
    print("  ★ **单卡口径**：这里只数「一个参数多占几字节」。")
    print("    **分片策略与通信量**（`4x`/`8x`/`16Ψ` 的账）的正文在")
    print("    `100-parallelism/README.md:14`（`zero-fsdp`）—— **本主题只留接口**。")
    check("B.1", dev > TOL_BUFFER_FP16,
          "★ **省一半状态字节的代价是真的**：fp16 buffer 让 %d 步后的轨迹偏离 %.3e（> %g）——"
          "「省精度」这一格的账记在**更新量的精度**上" % (N_STEPS, dev, TOL_BUFFER_FP16))
    print("  ★ 口径：`4` 与 `2` 是**字节/参数**（fp32 / fp16 各一个 buffer）；")
    print("    而 `mixed-precision/01` 的「优化器状态 8 字节/参数」数的是 **Adam 的 `m`+`v`**")
    print("    （`90-training/mixed-precision/01-type-selection.md` §2.2）—— **两个数不是同一个量**"
          "（动量只有一份状态、Adam 有两份）。")

    # ── B.2 ★ 更新量的量级：它与 `ULP/2` 的账（回链 mixed-precision/01）──────
    print()
    print("  ── B.2 更新量的量级：`|Δw| = lr·|b|` 会掉到 `ULP(1.0)/2` 以下 ──")
    print("  ★ 阈值 `ULP(1.0)/2 = %g` 的**判据与来源在**" % F16_ULP_HALF_AT_1)
    print("    `90-training/mixed-precision/01-type-selection.md` §1.2（`D7`/`D8`）——"
          "本程序**只消费、不重新定义**。")
    w = np.array([1.0])                      # `w = 1.0` 是 fp16 的**格点**（`D7` 的口径）
    for lr, name in ((LR_TINY, "lr=1e-4"), (LR_BIG, "lr=1e-3")):
        dw = lr * TINY_GRAD
        swallowed = (np.float16(w[0] + dw) == np.float16(w[0]))
        print("    %-9s |Δw| = lr·|g| = %.3e · 被吞？ %s"
              % (name, dw, "是" if swallowed else "否"))
    crit_lr = F16_ULP_HALF_AT_1 / TINY_GRAD
    print("  ★ 闭式临界：`lr* = (ULP/2)/|g| = %g / %g = %.6f`" % (F16_ULP_HALF_AT_1, TINY_GRAD, crit_lr))
    check("B.2", (np.float16(1.0 + LR_TINY * TINY_GRAD) == np.float16(1.0))
          and (np.float16(1.0 + LR_BIG * TINY_GRAD) == np.float16(1.0)),
          "★ 在 `|g| = %g` 上，`lr = %g` 与 `lr = %g` **两个档的更新都被吞**"
          "（`w + Δw == w` 逐位成立）⇒ 更新量的量级账是**可精确测的**，不是「精度不够」这种话"
          % (TINY_GRAD, LR_TINY, LR_BIG))
    check("B.2b", 0.0 < LR_BIG < crit_lr,
          "★ 两侧对照：`lr = %g < lr* = %.4f` 时**被吞** —— 所以「被吞」有一个可算的"
          "临界 `lr`，不是「任何 lr 都被吞」" % (LR_BIG, crit_lr))
    # ★★ **B.2c 是本程序实测出来的一个「边界取值」细节**：`ULP/2` 那个点是一个**平局**
    #   （恰好落在两个格点的中点上），而 IEEE 754 的舍入是 **round-half-to-even** ⇒
    #   翻不翻**取决于那个格点的尾数奇偶**。实测（本机）：
    #     · `w = 1.0`（位模式 `15360`，尾数**偶**）在 `Δw = ULP/2` 上**不翻**；
    #     · `w = 1 + ULP`（位模式 `15361`，尾数**奇**）在同一个 `Δw` 上**翻**（舍向偶数）。
    #   ⇒ 「临界值 = `ULP/2`」这句话在**平局点**上是不够的：它**含不含等号**要看格点。
    #   ★ 这不是本程序的口径选择，是 IEEE 754 的规则；判据因此写成**严格的两侧 + 平局单列**。
    w_even = np.float16(1.0)
    w_odd = np.float16(1.0 + float(np.spacing(np.float16(1.0))))
    half = F16_ULP_HALF_AT_1
    even_bits = int(np.float16(w_even).view(np.uint16))
    odd_bits = int(np.float16(w_odd).view(np.uint16))
    print("  ★ `ULP/2` 那个点是**平局**（round-half-to-even）⇒ 翻不翻看格点的尾数奇偶：")
    print("    w=%.10f（位模式 %d，尾数**偶**）Δw=ULP/2 ⇒ 翻？ %s"
          % (float(w_even), even_bits, np.float16(float(w_even) + half) != w_even))
    print("    w=%.10f（位模式 %d，尾数**奇**）Δw=ULP/2 ⇒ 翻？ %s"
          % (float(w_odd), odd_bits, np.float16(float(w_odd) + half) != w_odd))
    check("B.2c", int(np.float16(w_even).view(np.uint16)) % 2 == 0
          and int(np.float16(w_odd).view(np.uint16)) % 2 == 1
          and np.float16(float(w_even) + half) == w_even
          and np.float16(float(w_odd) + half) != w_odd
          and np.float16(float(w_even) + half * (1.0 - 1e-6)) == w_even
          and np.float16(float(w_even) + half * (1.0 + 1e-6)) != w_even,
          "★★ `Δw = ULP/2` 处是**平局**：`w = 1.0`（尾数偶）**不翻**、`w = 1 + ULP`（尾数奇）**翻**"
          "（round-half-to-even）；而 `ULP/2·(1∓1e-6)` 严格的两侧分别是「不翻 / 翻」 ⇒ "
          "「临界 = `ULP/2`」这句话**在平局点上含不含等号要看格点**")


def _buffer_traj(grads, dtype):
    """用 `dtype` 存 buffer 的重球轨迹（`w` 始终 fp64）。"""
    n = grads.shape[1]
    w = np.zeros(n)
    b = np.zeros(n)
    for g in grads:
        b = np.asarray(np.asarray(0.9 * b + g, dtype=dtype), dtype=np.float64)
        w = w - LR * b
    return w


# ===========================================================================
# 7. C 类：边界与反例 —— 失败行为真的成立
# ===========================================================================
def part_C():
    print()
    print("=" * 74)
    print("C. 边界与反例：`lr` 太大时真的发散，且动量把边界**挪了位置**")
    print("=" * 74)
    n_use = 400

    # ── C.1 纯 SGD 的发散边界 `lr·λ = 2` ────────────────────────────────────
    print("  ── C.1 纯 SGD 在 `f(w) = ½λ‖w‖²` 上：边界 `lr·λ = 2` ──")
    print("     闭式（推导值）：`w_n = (1 − lr·λ)^n·w₀` ⇒ 收敛 ⟺ `|1 − lr·λ| < 1` ⟺ `0 < lr·λ < 2`")
    for lr, tag in ((2.0 - 1e-3, "略小于 2"), (2.0, "**恰好** 2"), (2.0 + 1e-9, "略大于 2")):
        wn = _quad_sgd(lr, 1.0, n_use)
        print("    lr·λ = %.10f（%s）⇒ w = %.10e" % (lr, tag, wn))
    w_at = _quad_sgd(2.0, 1.0, n_use)
    w_over = _quad_sgd(2.0 + 1e-9, 1.0, n_use)
    w_under = _quad_sgd(1.999, 1.0, n_use)
    check("C.1", abs(w_at - 1.0) < 1e-12,
          "★ 边界**精确**在 `lr·λ = 2`：恰好 2 时 `|w|` **不衰减也不发散**（w = %.12f，"
          "它是 `−1` 的不动点：`1 − 2 = −1`）" % w_at)
    check("C.1b", abs(w_over) > 1.0 and abs(w_under) < 1.0,
          "★ 两侧对照：`lr·λ = 2 + 1e-9` 时 w = %.6e（**发散**）；`= 1.999` 时 w = %.6e（**收敛**）"
          " ⇒ 「`lr` 太大就发散」是**可证伪**的，且它的位置有闭式"
          % (w_over, w_under))

    # ── C.2 动量把边界放大到 `2(1+μ)` ───────────────────────────────────────
    print()
    print("  ── C.2 动量把同一个边界放大到 `lr·λ = 2(1+μ)` ──")
    for mu in (0.5, 0.9):
        bd = 2.0 * (1.0 + mu)
        for factor, tag in ((0.999, "略小于"), (1.0, "**恰好**"), (1.001, "略大于")):
            wn = _quad_heavy(bd * factor, mu, 300)
            print("    μ=%.1f  lr·λ = %.6f（%s %.3f）⇒ w = %.6e" % (mu, bd * factor, tag, bd, wn))
    bd09 = 2.0 * (1.0 + MU)
    check("C.2", abs(bd09 - 3.8) < 1e-12,
          "★ μ=%g 时边界 = `2(1+μ)` = %.1f（推导值：特征方程 `z² − (1+μ−lr·λ)z + μ = 0` "
          "的根都在单位圆内）—— 比无动量的 `2` **宽 %.1f 倍**" % (MU, bd09, 1.0 + MU))
    check("C.2b", abs(_quad_heavy(bd09 * 0.999, MU, 300)) < 1.0
          and abs(_quad_heavy(bd09 * 1.001, MU, 300)) > 1.0,
          "★ 两侧对照：略小于边界时收敛（w = %.3e）、略大于时发散（w = %.3e）⇒ "
          "边界的位置是真的，不是拍的"
          % (_quad_heavy(bd09 * 0.999, MU, 300), _quad_heavy(bd09 * 1.001, MU, 300)))


def _quad_sgd(lr, lam, n):
    """`f = ½λw²` 上的纯 SGD：`w ← w − lr·λ·w`（闭式 `w_n = (1 − lr λ)^n w₀`）。"""
    w = 1.0
    for _ in range(n):
        w = w - lr * lam * w
    return w


def _quad_heavy(lr, mu, n):
    """`f = ½λw²`（`λ = 1`）上的重球：`b ← μ b + w`；`w ← w − lr b`。"""
    w = 1.0
    b = 0.0
    for _ in range(n):
        b = mu * b + w
        w = w - lr * b
    return w


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

    measured = {"SEED": SEED, "N_PARAMS": N_PARAMS, "N_STEPS": N_STEPS, "LR": LR,
                "MU": MU, "W0": W0, "LAM": LAM, "W0_LOOKAHEAD": W0_LOOKAHEAD,
                "TINY_GRAD": TINY_GRAD, "LR_TINY": LR_TINY, "LR_BIG": LR_BIG,
                "F16_ULP_HALF_AT_1": F16_ULP_HALF_AT_1}
    expect = {"SEED": 20260928, "N_PARAMS": 32, "N_STEPS": 20, "LR": 0.1,
              "MU": 0.9, "W0": 0.0, "LAM": 1.0, "W0_LOOKAHEAD": 1.0,
              "TINY_GRAD": 1e-3, "LR_TINY": 1e-4, "LR_BIG": 1e-3,
              "F16_ULP_HALF_AT_1": 4.8828125e-04}
    diff = {k: (expect[k], v) for k, v in measured.items() if v != expect[k]}
    check("D.3", not diff, "★ 口径常量 12 个逐位一致（结论的**测量条件**）%s"
          % ("" if not diff else " ← 差异 %s" % diff))
    check("D.3b", F16_ULP_HALF_AT_1 == 2.0 ** -11 and LR_TINY * TINY_GRAD < F16_ULP_HALF_AT_1,
          "★ `F16_ULP_HALF_AT_1` 恰是 `2^-11`（`ULP(1.0)/2`），且 `LR_TINY·|g|` 真的落在它下面"
          "（`B.2` 的「被吞」全靠这两个事实）")
    check("D.3c", LR == 0.1 and MU == 0.9 and specs.ADAM_DEFAULT_BETA1 == MU,
          "★ `LR = 0.1`、`MU = 0.9` 逐位（★ **它们是本程序的设定值，不是从论文抄的**；"
          "而 `specs.ADAM_DEFAULT_BETA1` **恰好也是 0.9** —— 两个 `0.9` **数值相同、角色不同**，"
          "见 `A.1b` 与 `D.5`）")

    th = {"TOL_CLOSED": TOL_CLOSED, "TOL_SAME_ALGO": TOL_SAME_ALGO,
          "TOL_DIVERGE": TOL_DIVERGE, "TOL_BUFFER_FP16": TOL_BUFFER_FP16,
          "TOL_BITWISE": TOL_BITWISE, "MIN_REGION": MIN_REGION}
    th_expect = {"TOL_CLOSED": 1e-12, "TOL_SAME_ALGO": 1e-12, "TOL_DIVERGE": 1e-3,
                 "TOL_BUFFER_FP16": 1e-3, "TOL_BITWISE": 0.0, "MIN_REGION": 1e-3}
    tdiff = {k: (th_expect[k], v) for k, v in th.items() if v != th_expect[k]}
    check("D.4", not tdiff, "★ 阈值 6 个逐位一致（含**一个 0 容差**：`A.3b` 的逐位相等；"
                            "★ 放宽 `MIN_REGION` 会让 P22 检查退化）%s"
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

    # ── Adam 的四个默认超参（★ 本程序不实现 Adam，但它定义了 `β₁ = 0.9` 那个 0.9 的出处）──
    check("D.5", (specs.ADAM_DEFAULT_LR, specs.ADAM_DEFAULT_BETA1,
                  specs.ADAM_DEFAULT_BETA2, specs.ADAM_DEFAULT_EPS)
          == (0.001, 0.9, 0.999, 1e-8),
          "★ Adam 的四个默认超参逐位一致（[C] 1412.6980，**已确认 ICLR 2015**）—— "
          "★ 其中 `β₁ = 0.9` 与本程序的 `MU = 0.9` **数值相同、角色不同**（一个是指数衰减率、"
          "一个是动量系数）")
    check("D.5b", specs.ADAM_DEFAULTS_QUOTE
          == "Good default settings for the tested machine learning problems are",
          "★ 该句的逐字（**PDF 文本层**）逐位一致 —— ★ 只取**公式之前**的片段："
          "公式里的 ε 在 `pdftotext` 输出里是**控制字符**，跨行 ⇒ 按 §0.3 纪律**不数它**")
    check("D.5c", specs.MUON_UPDATE_RMS == 0.2 and specs.MUON_ADAMW_UPDATE_RMS_LO == 0.2
          and specs.MUON_ADAMW_UPDATE_RMS_HI == 0.4,
          "★ Muon 的 `update RMS` 三个数逐位一致（[C] 2502.16982，**不确认同行评审**）——"
          "★ 它是本主题「更新量的量级」这一格的**唯一外部锚点**，见 `02` 的 `B.3`")

    # ── ★★ PyTorch 的规格：两条链的 nesterov 定义与出处 ────────────────────
    check("D.6", specs.PYTORCH_SGD_SIGNATURE
          == "class torch.optim.SGD(params, lr=0.001, momentum=0, dampening=0, weight_decay=0, nesterov=False",
          "★ `[B]` PyTorch 2.14 的 `SGD` 签名逐位一致（★ **两个默认值是判据的一部分**："
          "`momentum=0`、`nesterov=False`）—— **用 `==`**：它的变异形态是末尾追加字符")
    check("D.6b", specs.PYTORCH_SGD_NESTEROV_ATTRIBUTION
          == "Nesterov momentum is based on the formula from On the importance of initialization and momentum in deep learning",
          "★ 而 PyTorch 把「Nesterov 动量」的出处标成 **Sutskever et al. 2013 那一篇**，"
          "**不是** Nesterov 1983 ⇒ 转述时不得凭名字往 1983 那篇上挂（P21 的同类：先看它指向谁）")

    # ── ★★ Adam 与 AdamW：`decoupled_weight_decay` 的默认值 + 两个 weight_decay 默认值 ──
    check("D.7", specs.PYTORCH_ADAM_DECOUPLED_FLAG == "decoupled_weight_decay=False",
          "★★ `[B]` PyTorch 2.14 的 `torch.optim.Adam` 默认 **`decoupled_weight_decay=False`** ——"
          "「`Adam` + `weight_decay` 就是 AdamW」**是错的**（Y6）")
    check("D.7b", (specs.PYTORCH_ADAM_DEFAULT_WEIGHT_DECAY,
                   specs.ADAMW_DEFAULT_WEIGHT_DECAY) == (0, 0.01),
          "★★ 两个 `weight_decay` 的**默认值成对**：`Adam` 是 `0`、`AdamW` 是 `0.01` ——"
          "同一个框架、同一个语义的旋钮，两个类给了不同的默认值 ⇒ 「用哪个类」不只是名字")

    # ── 论文侧的归因与 PyTorch 侧的归因：**两条都引** ──────────────────────
    check("D.8", specs.ADAMW_L2_ATTRIBUTION_QUOTE
          == "weights x with large typical gradient magnitude s are regularized by a smaller relative"
          and specs.PYTORCH_ADAMW_NO_ACCUMULATE_QUOTE
          == "weight decay does not accumulate in the momentum nor variance",
          "★★ **Y5 的两条并列**：论文说「典型梯度大的权重被正则得更少」（L2 的梯度也被自适应"
          "归一化了），PyTorch 说「不进动量与方差」—— **不矛盾但不是同一句话** ⇒ "
          "把 PyTorch 的措辞当论文的理由 = **规范与实现不分**")
    check("D.8b", specs.ADAMW_MECHANISM_QUOTE
          == "whereas with decoupled weight decay, only the gradients of the loss function",
          "★ 论文的**正面表述**也逐位钉住（「解耦后只有损失的梯度被自适应」）")
    check("D.9", specs.ADAMW_TEST_ERROR_QUOTE == "relative improvement in test error",
          "★ `15%` 那句的片段逐位一致 —— ★ 它的作用域是 **CIFAR-10 与 ImageNet32x32、`lr = 0.001`**，"
          "**不得**写成「AdamW 普遍好 15%」（P20 的同类）")

    # ── 裁剪的 `1e-6`：本主题只消费、不重定义 ──────────────────────────────
    check("D.10", specs.TORCH_CLIP_COEF_DENOM_EPS == 1e-6,
          "★ `TORCH_CLIP_COEF_DENOM_EPS == 1e-6`（[B] PyTorch **v2.9.0 的源码**）——"
          "★ 本程序**不消费它**（裁剪在 `03`）；这里钉住是为了让 `03` 的 `A.3` 与本行同值")
    check("D.11", specs.UNSOURCED == ("adam_eps_position_raw_count",
                                      "pytorch_adam_float32_eps_version_span",
                                      "muon_orthogonalization_iteration_count",
                                      "muon_update_rms_measurement_convention"),
          "★ `specs.UNSOURCED` 的四项**逐位钉住**（%d 项）" % len(specs.UNSOURCED))

    # ── 独立性：两条路径的源码不同 + 调用表不相交 ──────────────────────────
    src_rec = inspect.getsource(momentum_recursion)
    src_cls = inspect.getsource(momentum_closed_form)
    check("D.12", src_rec != src_cls,
          "★ 防 P17：被测（`momentum_recursion`）与参照（`momentum_closed_form`）的**源码不同**"
          "（%d vs %d 字符）—— 前者是**一层** `for` 的原地递推、后者是**双重**求和的闭式"
          % (len(src_rec), len(src_cls)))
    watched = {"momentum_recursion", "momentum_closed_form", "nesterov_minus_heavy_closed_form"}
    called = _callee_names(momentum_closed_form) | _callee_names(momentum_minus_sgd_closed_form) \
        | _callee_names(nesterov_minus_heavy_closed_form)
    check("D.13", not (watched & called),
          "★ **AST 调用表**：三条**参照路径**的 `Call` 节点（`ast.Name` 形态）= %s —— "
          "一条都没调到被测函数" % (", ".join(sorted(called)) if called else "**空集**"))
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
    print("01_sgd_momentum.py —— SGD / 动量 / Nesterov 的更新式与状态")
    print("=" * 74)
    print("  ★ 真值标注：**全部是推导值**（本机 `python3 %s` + `numpy %s`，macOS arm64 无 GPU）。"
          % (sys.version.split()[0], np.__version__))
    print("  ★ 失败标记 ASSERT_FAIL_MARK = %r（本主题自定义；不跨主题照抄）" % ASSERT_FAIL_MARK)
    print("  ★ 禁区（一律回链，本程序不写）：优化器状态的**分片与通信量**在")
    print("    `100-parallelism/README.md:14`（`zero-fsdp`）；DDP / 梯度规约在 `:9` 与 `110-communication/`；")
    print("    `lr` 被编译 guard 固化在 `60-compilers/inductor/04-dynamic-shapes.md:42,163`；")
    print("    框架的 autograd 语义在 `70-frameworks/pytorch/02-autograd.md`。")
    print("  ★ 接口：`Δw < ULP/2` 的阈值与判据在 `mixed-precision/01-type-selection`（此处只消费）。")
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

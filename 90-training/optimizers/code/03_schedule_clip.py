#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""03_schedule_clip.py —— 学习率调度 与 **梯度裁剪**（含它在训练管线里的位置）。

配套文档：`03-schedules-and-clipping.md`。

★ 三类断言（A 算法正确性 / B 代价模型 / C 边界与反例）**全部在本程序内部**，
失败时以**非零退出码**结束；**没有 `verify.py`**（`CONTRIBUTING` §2 表里的第二种情形）。

★★ **本程序的数字全部是推导值**，测量条件是：
> `python3 3.13.5` + `numpy 2.4.4`，macOS arm64，**无 GPU**。
> 玩具规模的真跑，**不得支撑关于真实训练系统的结论**。

★★ **本主题的禁区**（`CONTRIBUTING` §7 + 域 README 的边界表）：
  · **优化器状态的分片策略与通信量** → `100-parallelism/README.md:14`（`zero-fsdp`）；
  · **数据并行 / DDP / 梯度规约** → `100-parallelism/README.md:9`、`110-communication/`；
  · ★★ **`LR warmup` 的机制解释**（post-LN 的平均场理由、`hypothesize` 的措辞纪律）→
    `80-model-architectures/normalization/02-placement-and-qk-norm.md:83-100`；
    **本程序一行都不重写它**，只做**工程判据与设定**（形状、峰值位置、末值、总账）。
  · **框架的 autograd 语义** → `70-frameworks/pytorch/02-autograd.md`。
  · ★★ **同名不同指（M9）**：`00-meta/glossary.md:146` 已有的 `warmup` 指
    **benchmark 预热**（`45-kernel-lifecycle/convergence-and-tuning/01-benchmark-hygiene.md`
    里 `warmup` 23 次）—— 与本程序的 **LR warmup** **不是同一件事** ⇒
    正文与本程序**一律写全 `LR warmup`**，并**各自带上这句限定语**。

★★ **两处回链（阈值与判据都定义在别处，本程序只实现）**：
  ① **裁剪的判据与阈值**在 `90-training/autograd-and-backward/03-gradient-lifecycle.md`
     的「表 A ⑦」——**在那里只写一次**（`T ≥ 1`、上侧 `≤ 1.1e-6/max(T,1)`、
     ★★ **下侧 `== ε/(T+ε)`，即 `A.2b`**）。★ **回链时必须一并带上 `A.2b`**（**T26**）：
     只写单侧会放过一个「把保护项 `+1e-6` 删掉」的实现，而那个实现**结果反而更精确**
     （Chunk 1 实测：删掉后 **25 条断言全绿**）。
  ② **`unscale` 的时机**（`right after the backward pass but before gradient clipping`，
     [C] Micikevicius 2017 逐字）在 `90-training/mixed-precision/02-amp-execution.md`
     的 `A.2`/`A.2c`/`A.2d` —— 本程序**不重定义**那条判据，它只钉**调度与裁剪的接口**：
     **`clip` 与 `lr` 是两个正交的旋钮**。

★★ 可证伪断言（把下面任一条改错，本程序必须非零退出）：
  · `A.1`  LR warmup + cosine 的**形状**：峰值**精确**在 warmup 步、末值**精确** `0`、
           且逐点与另一条路径（半角恒等式 `0.5(1+cosθ) = cos²(θ/2)`）一致（< 1e-15）；
  · `A.2`  裁剪的上侧（相对误差 ≤ `1.1e-6/max(T,1)`，`T ≥ 1`）与 **`A.2b` 的下侧**
           （实测相对误差 **== 闭式 `ε/(T+ε)`**，容差 1e-6）—— **两侧**（**T26**）；
  · `A.3`  ★★ **`clip` 与 `lr` 正交**：三档 `lr` 上 `clip` 的输出**逐位相同**，
           **而错序实现（`clip` 作用在 `Δw` 上）不是**（差 > 1e-2）；
  · `A.3c` ★ 错序把阈值放大 `1/lr` 倍 —— 在 `lr·T < max_norm` 那一档的后果是**完全不裁剪**，
           比值有**闭式** `(T+ε)/max_norm`（且**与 `lr` 无关**）；
  · `A.3d` ★★ 错序 × LR warmup 的交叉：错序下 warmup 首步与峰值步的**等效阈值之比 == W**，
           而正确顺序下 == `1`；
  · `B.1`  调度步长的**总账**：`Σ lr(t) == lr_max·T/2`（**与 `W` 无关**）；
  · `B.2`  裁剪**只改范数、不改方向**（方向余弦 == `1`，逐位）；
  · `C.1`  ★ 梯度里有 `inf` 时裁剪给出 **`nan`** ⇒ **裁剪不能替代溢出处置**
           （接口：`mixed-precision/02` 的跳步）；
  · `C.2`  `max_norm = 0` 时梯度全清零 ⇒ 更新恒为 `0`（失败行为真的成立）。

用法：
    python3 03_schedule_clip.py
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
SEED = 20260930              # ★ 与同域另外两个程序（20260928 / 20260929）都**不同**
TOTAL_STEPS = 100            # 调度的总步数 `T`
WARMUP_STEPS = 10            # `LR warmup` 的步数 `W`
LR_MAX = 1.0                 # 峰值学习率
LR_MAX_SMALL = 0.1           # 另一档峰值（用来验「总账与 `lr_max` 成正比」）
WARMUP_LIST = (1, 10, 25, 50)   # B.1：跨多组 `(W, T)` 验「总账与 `W` 无关」
TOTAL_LIST = (100, 50)       # 与 `WARMUP_LIST` 配对
SCHEDULE_T_LIST = (10, 50, 100, 7)   # A.1：逐点扫的几组步数

# ★★ 裁剪的阈值口径 —— **判据与阈值定义在 `autograd-and-backward/03` 的「表 A ⑦」**，
#   本程序**只实现、不重定义**。`ε` 从 `specs.py` 读（[B] PyTorch v2.9.0 的源码口径）。
CLIP_T_LIST = (2.0, 4.0, 8.0)      # ★ `T ≥ 1` 是判据的一部分（计划 §3 表 A ⑦ 写死）
CLIP_MAX_NORM = 1.0                # `max_norm`
CLIP_MAX_NORM_LIST = (1.0, 0.5, 0.1)
CLIP_TOL_REL = 1.1e-6              # 上侧：实际阈值是 `CLIP_TOL_REL / max(T, 1)`
CLIP_TOL_CLOSED = 1e-6             # 下侧（`A.2b`）：与闭式 `ε/(T+ε)` 的相对偏差
CLIP_T_SMALL = 0.5                 # A.2c：`T < max_norm`（**不发生裁剪**）那一档
CLIP_NORM_T = 10.0                 # A.3：构造一个范数**恰好**为它的梯度
CLIP_LR_LIST = (0.01, 0.05)        # A.3c：`lr·T < max_norm` 的两档（比值应与 `lr` 无关）
CLIP_LR_BIG = 0.1                  # A.3c 的边界档（`lr·T == max_norm`）

TOL_SCHED = 1e-15            # A.1：与半角恒等式的逐点相对差
TOL_EXACT = 0.0              # A.3a：三档 lr 上 `clip` 输出**逐位**相同
TOL_PROP = 1e-15             # A.3b：`Δw/lr` 逐位相同
TOL_RATIO = 1e-12            # A.3c：与闭式 `(T+ε)/max_norm` 的相对差
TOL_W_CROSS = 1e-12          # A.3d：与 `W` 的相对差
TOL_TOTAL = 1e-12            # B.1：`Σ lr == lr_max·T/2`
TOL_MEASURABLE = 1e-2        # A.3a 的反面下界：错序实现必须**差超过**它
N_GRAD = 64                  # 一个「梯度张量」的元素个数


# ===========================================================================
# 2. 路径（被测）：学习率调度
# ===========================================================================
def lr_schedule(t, warmup, total, lr_max):
    """LR warmup（线性）+ cosine 退火（**锚在末端**）。

    ```
    t < warmup            :  lr_max · (t+1)/warmup
    t >= warmup           :  lr_max · 0.5·(1 + cos(π·(t−warmup+1)/(total−warmup)))
    ```
    ★★ **两条边界是「手算得到」的**：`lr(warmup−1) == lr_max`（**峰值精确**）、
      `lr(total−1) == 0`（**末值精确**，因为 `cos(π) = −1`）。
    ★ 它还满足一条**不显然的恒等式**（`B.1`）：`Σ_{t<T} lr(t) == lr_max·T/2`，**与 `warmup` 无关**。
    """
    if warmup > 0 and t < warmup:
        return lr_max * (t + 1) / warmup
    m = total - warmup
    if m <= 0:
        return lr_max
    k = t - warmup + 1
    return lr_max * 0.5 * (1.0 + math.cos(math.pi * k / m))


def lr_schedule_half_angle(t, warmup, total, lr_max):
    """★ **独立参照**：cosine 那一段用**半角恒等式** `0.5(1+cosθ) = cos²(θ/2)` 重算。

    ⇒ 与 `lr_schedule` 是**两条不同的算式**（一个用 `1 + cos`、一个用 `cos²`），
      在同一个 `t` 上必须给出同一个数（`A.1b` 逐点对拍）。
      ★ 而 warmup 那一段两条路径**共用同一个线性式** —— 那是**定义**，不是「参照」。
    """
    if warmup > 0 and t < warmup:
        return lr_max * (t + 1) / warmup
    m = total - warmup
    if m <= 0:
        return lr_max
    k = t - warmup + 1
    c = math.cos(0.5 * math.pi * k / m)
    return lr_max * c * c


def lr_schedule_wrong_warmup(t, warmup, total, lr_max):
    """★★ **反面对照**：LR warmup 那一段用**常数** `lr_max`（而不是线性上升）。

    ⇒ `B.1` 的「总账与 `warmup` 无关」这条恒等式**立刻破掉**（实测 `Σ` 从 50 变成 54.5）。
    """
    if warmup > 0 and t < warmup:
        return lr_max
    m = total - warmup
    if m <= 0:
        return lr_max
    k = t - warmup + 1
    return lr_max * 0.5 * (1.0 + math.cos(math.pi * k / m))


def total_steps_closed_form(lr_max, total):
    """★ **独立参照**：`Σ_{t<T} lr(t) = lr_max·T/2`（**推导值**，见 `B.1` 的推导）。"""
    return lr_max * total / 2.0


# ===========================================================================
# 3. 路径（被测）：裁剪 与 训练管线
# ===========================================================================
def global_norm(g):
    """全局范数（把**全部参数**的梯度看成一个向量）。"""
    g = np.asarray(g, dtype=np.float64)
    return float(math.sqrt(float((g ** 2).sum())))


def clip_global_norm(g, max_norm, eps=specs.TORCH_CLIP_COEF_DENOM_EPS):
    """按全局范数缩放。★★ **`(T + 1e-6)` 这个分母口径是 [B] PyTorch 的**（v2.9.0 源码），

    **不是规范**：裁剪后的范数 = `max_norm · T/(T + ε)`，**是一个近似**。
    ★★ **判据与阈值在 `90-training/autograd-and-backward/03-gradient-lifecycle.md` 的表 A ⑦**
      （**在那里只写一次**，连 `T ≥ 1` 的取值范围也钉在彼处）—— **本程序只实现、不重定义**。
    """
    total = global_norm(g)
    coef = max_norm / (total + eps)
    if coef < 1.0:
        return np.asarray(g, dtype=np.float64) * coef
    return np.asarray(g, dtype=np.float64)


def train_step_correct(g_raw, scale, max_norm, lr):
    """★★ **正确顺序**（本主题钉的那一条）：

    ```
    ①  unscale ： g ← g_raw / S    （[C] Micikevicius 2017 逐字：
                                     "right after the backward pass but before gradient clipping"）
    ②  clip    ： 按**梯度**的全局范数缩放到 max_norm
    ③  step    ： Δw ← −lr·g_clipped   （在 fp32 主权重上做）
    ```
    ★★ **关键在于 ② 与 ③ 的次序**：裁剪作用在**梯度**上 ⇒ 阈值 `max_norm` 与 `lr`
      **是两个正交的旋钮**（`A.3`）。把 ② 挪到 ③ 之后（裁「更新量」）会把阈值放大 `1/lr` 倍。
    """
    g = np.asarray(g_raw, dtype=np.float64) / scale
    g = clip_global_norm(g, max_norm)
    return -lr * g, g


def train_step_clip_after_lr(g_raw, scale, max_norm, lr):
    """★★ **错序 A**（对照）：先做 step、再对 **`Δw`** 做裁剪。"""
    g = np.asarray(g_raw, dtype=np.float64) / scale
    dw = -lr * g
    return clip_global_norm(dw, max_norm), g


def train_step_unscale_after_clip(g_raw, scale, max_norm, lr):
    """★★ **错序 B**（对照）：把 `unscale` 挪到 `clip` **之后**。

    ⚠★ **本函数不用来判 `unscale` 的时机** —— 那条判据的正文与阈值在
    `90-training/mixed-precision/02-amp-execution.md` 的 `A.2`/`A.2c`/`A.2d`
    （**在那里写一次，本程序回链、不重定义**）。
    它在这里的唯一用途是 `A.3f` 的**见证**：「`S = 1` 时这个错序在数值上是 no-op」
    ⇒ **数值断言抓不到它**，所以本程序改用**源码级次序**钉住。
    """
    g = clip_global_norm(np.asarray(g_raw, dtype=np.float64), max_norm)
    return -lr * (g / scale), g / scale


# ===========================================================================
# 4. 工具
# ===========================================================================
def _callee_names(func):
    """用 **AST** 列出 `func` 真正**调用**的函数名（不含注释/文档串里的提及）。

    ★ 只收 `ast.Name` 形态的调用 ⇒ `np.asarray(...)` 这类**属性调用不计入**
      ⇒ 参照路径的调用表**为空**（`D.11`）。
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


def make_grads(seed, n=N_GRAD, target_norm=None):
    """一批梯度：`N(0,1)`（可缩放到**恰好**指定的范数）。★ 换种子会换数。"""
    g = np.random.default_rng(seed).standard_normal(n)
    if target_norm is not None:
        g = g * (target_norm / global_norm(g))
    return g


def max_absdiff(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


# ===========================================================================
# 5. A 类：调度的形状、裁剪的两侧判据、**管线里 `clip` 与 `lr` 的正交性**
# ===========================================================================
def part_A():
    print("=" * 74)
    print("A. LR warmup + cosine 的形状 · 裁剪的两侧判据 · `clip` 与 `lr` 的正交性")
    print("=" * 74)
    print("  口径（**测量条件**）：T=%d · W=%d · lr_max=%g · max_norm=%g · ε=%g"
          % (TOTAL_STEPS, WARMUP_STEPS, LR_MAX, CLIP_MAX_NORM,
             specs.TORCH_CLIP_COEF_DENOM_EPS))

    # ── A.1 ★ 调度的形状：三条边界值 + 逐点对拍 ────────────────────────────
    print("  ── A.1 ★ 形状：峰值**精确**在 warmup 步、末值**精确** 0 ──")
    lr_0 = lr_schedule(0, WARMUP_STEPS, TOTAL_STEPS, LR_MAX)
    lr_peak = lr_schedule(WARMUP_STEPS - 1, WARMUP_STEPS, TOTAL_STEPS, LR_MAX)
    lr_end = lr_schedule(TOTAL_STEPS - 1, WARMUP_STEPS, TOTAL_STEPS, LR_MAX)
    print("  lr(0) = %.6f（闭式 lr_max/W = %.6f）" % (lr_0, LR_MAX / WARMUP_STEPS))
    print("  lr(W−1) = %.6f（= lr_max = %g，**峰值**）" % (lr_peak, LR_MAX))
    print("  lr(T−1) = %.1f（闭式 0，因为 cos(π) = −1）" % lr_end)
    check("A.1", abs(lr_0 - LR_MAX / WARMUP_STEPS) < TOL_SCHED
          and abs(lr_peak - LR_MAX) < TOL_SCHED and lr_end == 0.0,
          "★ 三条边界值**逐点手算得到**：`lr(0) = lr_max/W`、`lr(W−1) = lr_max`（峰值）、"
          "`lr(T−1) = 0`（末值**精确** 0，不是「≈ 0」）")
    # ★ 逐点对拍：cosine 段用**半角恒等式**重算
    # ★★ **P22 现场**：这条判据如果写成**相对**差（`|a/b − 1|`）会在**最后一步**假失败 ——
    #   那里 `b = cos²(π/2)` 已经衰减到 `3.7e-33`（**不是 0**，而是 `cos(π/2)` 的舍入残渣），
    #   于是 `0/b − 1 = −1`，相对差恒等于 1。⇒ **判据必须用「量级参照」**：
    #   与 `lr_max` 比（绝对口径），并把那个最小参照量级**打印出来**。
    worst_abs = 0.0
    min_ref = float("inf")
    argmin_ref = None
    for T in SCHEDULE_T_LIST:
        for W in WARMUP_LIST:
            if W >= T:
                continue
            for t in range(T):
                a = lr_schedule(t, W, T, LR_MAX)
                b = lr_schedule_half_angle(t, W, T, LR_MAX)
                worst_abs = max(worst_abs, abs(a - b))
                if b < min_ref:
                    min_ref, argmin_ref = b, (t, W, T)
    print("  ★ 与**半角恒等式** `0.5(1+cosθ) = cos²(θ/2)` 的逐点最大**绝对**偏差 = %.3e"
          "（判据口径：与 `lr_max = %g` 比）" % (worst_abs, LR_MAX))
    print("  ★★ **P22 检查**：参照量的**最小值** = %.3e（在 t=%d, W=%d, T=%d 处，"
          "那一步 `lr` 的真值就是 0，而 `cos²(π/2)` 只给到舍入残渣）"
          % (min_ref, argmin_ref[0], argmin_ref[1], argmin_ref[2]))
    check("A.1b", worst_abs < TOL_SCHED * LR_MAX and min_ref < 1e-30,
          "★ 逐点与**另一条算式**（半角恒等式）一致（最大绝对偏差 %.3e < %g·lr_max）⇒ "
          "cosine 那一段不是「照着写了一遍」，而是**两条独立算式给出同一个数**"
          % (worst_abs, TOL_SCHED))
    check("A.1c", min_ref < 1e-30 and min_ref > 0.0,
          "★ **P22 的落法**：参照量在末端**衰减到 %.3e**（不是 0）⇒ **那里必须用绝对口径判**；"
          "写成相对差会得到 `0/b − 1 = −1` 的**假失败** —— 这是「先确认参照量级」"
          "在本主题的现场（★ 那个 `%.3e` 是 `cos(π/2)` 的舍入残渣，不是任何物理量）"
          % (min_ref, min_ref))

    # ── A.2 ★ 裁剪的两侧判据（回链 `autograd-and-backward/03` 的表 A ⑦ + `A.2b`）──
    print()
    print("  ── A.2 ★ 裁剪：上侧 `≤ 1.1e-6/max(T,1)` + **下侧 `== ε/(T+ε)`**（**两侧**）──")
    print("     ★★ 阈值与判据定义在 `autograd-and-backward/03-gradient-lifecycle.md` 的表 A ⑦")
    print("     （**在那里只写一次**）—— **本程序只实现、不重定义**。")
    print("     ★★ 下侧（`A.2b`）是**必须的**：只写单侧会放过一个「删掉保护项 `+1e-6`」的实现，")
    print("        而它**结果反而更精确**（Chunk 1 实测：删掉后 25 条断言全绿）——**T26**。")
    print("     %-6s %-10s %-16s %-14s %-14s %-14s"
          % ("T", "max_norm", "裁剪后范数", "相对误差", "阈值 1.1e-6/T", "闭式相对偏差"))
    worst_up = 0.0
    worst_cf = 0.0
    for T in CLIP_T_LIST:
        base = make_grads(SEED + int(T) * 7, target_norm=T)
        for max_norm in CLIP_MAX_NORM_LIST:
            g_c = clip_global_norm(base, max_norm)
            new = global_norm(g_c)
            rel = abs(new / max_norm - 1.0)
            thr = CLIP_TOL_REL / max(T, 1.0)
            pred = specs.TORCH_CLIP_COEF_DENOM_EPS / (T + specs.TORCH_CLIP_COEF_DENOM_EPS)
            cf = abs(rel / pred - 1.0)
            worst_up = max(worst_up, rel / thr)
            worst_cf = max(worst_cf, cf)
            print("     %-6g %-10g %-16.12f %-14.3e %-14.3e %-14.3e"
                  % (T, max_norm, new, rel, thr, cf))
    check("A.2", worst_up <= 1.0,
          "★ **上侧**：裁剪后范数与 `max_norm` 的相对误差最大只有阈值的 %.4f 倍（≤ 1）—— "
          "`T ≥ 1` 是判据的一部分（`T ∈ %s`）" % (worst_up, CLIP_T_LIST))
    check("A.2b", worst_cf < CLIP_TOL_CLOSED,
          "★★ **下侧（`A.2b`，必须）**：实测的相对误差**等于**闭式 `ε/(T+ε)`"
          "（最大相对偏差 %.3e < %g）—— 删掉保护项 `+1e-6` 会让它变成 `0`，"
          "这条**立刻失败**（只写单侧是抓不到的）" % (worst_cf, CLIP_TOL_CLOSED))
    g_small = make_grads(SEED + 99, target_norm=CLIP_T_SMALL)
    g_unc = clip_global_norm(g_small, CLIP_MAX_NORM)
    check("A.2c", global_norm(g_unc) == global_norm(g_small) and CLIP_T_SMALL < CLIP_MAX_NORM,
          "★ 阴面对照：`T = %g < max_norm = %g` 时**不发生裁剪**（裁剪后范数 == `T`，逐位）⇒ "
          "「裁剪后 == `max_norm`」这句话**只在 `T > max_norm` 时成立**"
          % (CLIP_T_SMALL, CLIP_MAX_NORM))

    # ── A.3 ★★ `clip` 与 `lr` 正交 ─────────────────────────────────────────
    print()
    print("  ── A.3 ★★ `clip` 与 `lr` 是**两个正交的旋钮**（裁剪作用在**梯度**上）──")
    g_raw = make_grads(SEED + 41, target_norm=CLIP_NORM_T)
    scale = 1024.0                 # ★ 2 的幂 ⇒ unscale 是**逐位**精确的（不是判据的需要，是口径的干净）
    print("     %-8s %-20s %-20s %-14s %-18s"
          % ("lr", "clip 输出范数", "Δw 范数", "Δw/lr 范数", "错序**被裁的量**的范数"))
    norms_c = []
    norms_dw = []
    norms_ratio = []
    norms_wrong_clipped = []
    for lr in (0.001, 0.01, 0.1):
        dw, g_c = train_step_correct(g_raw, scale, CLIP_MAX_NORM, lr)
        _dw_w, _g_w = train_step_clip_after_lr(g_raw, scale, CLIP_MAX_NORM, lr)
        norms_c.append(global_norm(g_c))
        norms_dw.append(global_norm(dw))
        norms_ratio.append(global_norm(dw) / lr)
        norms_wrong_clipped.append(lr * global_norm(np.asarray(g_raw, dtype=np.float64) / scale))
        print("     %-8g %-20.15f %-20.15f %-14.15f %-18.15f"
              % (lr, norms_c[-1], norms_dw[-1], norms_ratio[-1], norms_wrong_clipped[-1]))
    spread_c = max(norms_c) - min(norms_c)
    spread_r = max(norms_ratio) - min(norms_ratio)
    ratio_r = spread_r / max(norms_ratio)
    lr_dep = max(norms_wrong_clipped) / min(norms_wrong_clipped)
    check("A.3a", spread_c == TOL_EXACT and lr_dep > 1.5,
          "★★ **两侧**：正确实现下三档 `lr` 上 **`clip` 看到的量逐位相同**（极差 %.1e = %g），"
          "而错序实现下**被裁的那个量**随 `lr` 变（三档跨 %.0f 倍）⇒ "
          "「`clip` 与 `lr` 正交」**不是恒真**的：它区分了两个实现"
          % (spread_c, TOL_EXACT, lr_dep))
    check("A.3b", ratio_r < TOL_PROP and max(norms_dw) / min(norms_dw) > 50.0,
          "★ 而 `Δw/lr` 在三档上也一致（**相对**极差 %.3e < %g ⇒ `Δw` **与 `lr` 严格成正比**；"
          "★ 用**相对**口径而不用逐位：`0.001·x/0.001` 与 `0.01·x/0.01` 在浮点里可以差最后一个比特）"
          "（`Δw` 本身跨了三档共 %.0f 倍）" % (ratio_r, TOL_PROP, max(norms_dw) / min(norms_dw)))
    # ★★ A.3c：错序 —— 阈值被放大 `1/lr` 倍
    print("  ── A.3c ★ 错序（`clip` 作用在 `Δw` 上）：在 `lr·T < max_norm` 时**完全不裁剪** ──")
    g_raw = make_grads(SEED + 43, target_norm=CLIP_NORM_T)
    print("     %-8s %-18s %-18s %-14s %-14s"
          % ("lr", "正确 ‖Δw‖", "错序 ‖Δw‖", "比值", "闭式 (T+ε)/max_norm"))
    closed_ratio = (CLIP_NORM_T + specs.TORCH_CLIP_COEF_DENOM_EPS) / CLIP_MAX_NORM
    worst_ratio = 0.0
    for lr in CLIP_LR_LIST:
        dw_r, _ = train_step_correct(g_raw, 1.0, CLIP_MAX_NORM, lr)
        dw_w, _ = train_step_clip_after_lr(g_raw, 1.0, CLIP_MAX_NORM, lr)
        r = global_norm(dw_w) / global_norm(dw_r)
        worst_ratio = max(worst_ratio, abs(r / closed_ratio - 1.0))
        print("     %-8g %-18.12f %-18.12f %-14.9f %-14.9f"
              % (lr, global_norm(dw_r), global_norm(dw_w), r, closed_ratio))
    print("     ★ 比值**与 `lr` 无关**（两档给出同一个数）—— 因为两档都落在 `lr·T < max_norm` 上，")
    print("       错序**根本没裁剪**（`coef ≥ 1`）⇒ 错序真正用的是 `lr·T` 而不是 `lr·max_norm`。")
    check("A.3c", worst_ratio < TOL_RATIO,
          "★★ 错序把阈值放大 `1/lr` 倍 ⇒ 后果是**完全不裁剪**：比值与闭式 `(T+ε)/max_norm` 一致"
          "（最大相对差 %.3e < %g），且**与 `lr` 无关**" % (worst_ratio, TOL_RATIO))
    # ★★ A.3d：错序 × LR warmup 的交叉
    print("  ── A.3d ★★ 错序 × LR warmup：warmup 首步的**等效阈值**被放大 `W` 倍 ──")
    lr_first = lr_schedule(0, WARMUP_STEPS, TOTAL_STEPS, LR_MAX)      # warmup 首步
    lr_peak2 = lr_schedule(WARMUP_STEPS - 1, WARMUP_STEPS, TOTAL_STEPS, LR_MAX)
    eff_right = CLIP_MAX_NORM / CLIP_MAX_NORM                          # 正确：与 lr 无关
    eff_wrong_first = CLIP_MAX_NORM / lr_first
    eff_wrong_peak = CLIP_MAX_NORM / lr_peak2
    ratio_wrong = eff_wrong_first / eff_wrong_peak
    print("     正确顺序的等效阈值（梯度口径）：首步 = %.6f · 峰值步 = %.6f ⇒ 比值 %.12f"
          % (CLIP_MAX_NORM, CLIP_MAX_NORM, eff_right))
    print("     错序的「等效阈值」（Δw 口径）：首步 = %.3f · 峰值步 = %.3f ⇒ 比值 %.6f"
          % (eff_wrong_first, eff_wrong_peak, ratio_wrong))
    print("     ★ 而 `lr_peak/lr_first = %g/%g = %g = W`" % (lr_peak2, lr_first, lr_peak2 / lr_first))
    check("A.3d", abs(ratio_wrong / WARMUP_STEPS - 1.0) < TOL_W_CROSS
          and abs(eff_right - 1.0) < TOL_W_CROSS and WARMUP_STEPS > 1,
          "★★ **错序在 LR warmup 期最有害**：错序下 warmup 首步的等效阈值是峰值步的 **`W` = %d 倍**"
          "（`%.6f` vs `%.6f`），而正确顺序下这个比值恒为 `1` ⇒ "
          "「先 unscale、再 clip（**梯度**）、最后乘 `lr`」这条顺序与 warmup **耦合**在一起"
          % (WARMUP_STEPS, ratio_wrong, eff_right))
    print("     ★ 接口（**回链、不重定义**）：`unscale` 的时机那条判据在")
    print("       `90-training/mixed-precision/02-amp-execution.md` 的 `A.2`/`A.2c`/`A.2d`；")
    print("       本程序只钉**调度与裁剪的接口**（两者的次序与正交性）。")
    # ★★ A.3e：`clip` 的**左边那个邻居**（`S`）—— 它的**错序在数值上是 no-op**
    print("  ── A.3e ★★ 见证：`S = 1` 时「`clip` 与 `unscale` 换序」**数值上测不出来** ──")
    g_base = make_grads(SEED + 45, target_norm=CLIP_NORM_T)
    dw_ok, _ = train_step_correct(g_base, 1.0, CLIP_MAX_NORM, 0.01)
    dw_bad, _ = train_step_unscale_after_clip(g_base, 1.0, CLIP_MAX_NORM, 0.01)
    d_noop = max_absdiff(dw_ok, dw_bad)
    dw_bad2, _ = train_step_unscale_after_clip(np.asarray(g_base, dtype=np.float64) * 1024.0,
                                               1024.0, CLIP_MAX_NORM, 0.01)
    dw_ok2, _ = train_step_correct(np.asarray(g_base, dtype=np.float64) * 1024.0, 1024.0,
                                   CLIP_MAX_NORM, 0.01)
    n_ok2 = global_norm(dw_ok2)
    n_bad2 = global_norm(dw_bad2)
    ratio_bad = n_ok2 / n_bad2
    print("     `S = 1`   ：正确顺序与错序 B 的 Δw 最大逐元素差 = %.1e（**完全一样**）" % d_noop)
    print("     `S = 1024`：正确 ‖Δw‖ = %.9f · 错序 B ‖Δw‖ = %.9f ⇒ 相差 %.1f 倍"
          % (n_ok2, n_bad2, ratio_bad))
    check("A.3e", d_noop == TOL_EXACT and ratio_bad > 100.0,
          "★★ **为什么必须用源码级次序钉住**：`S = 1` 时「`clip` 与 `unscale` 换序」"
          "**逐位相同**（差 %.1e）⇒ **任何数值断言都抓不到它**（本程序的结构翻转抽查里，"
          "这个错序确实 `rc=0` 溜了过去）；而 `S = 1024` 时它把 `‖Δw‖` 压小 **%.0f 倍** ⇒ "
          "「错序的后果」是真的，只是**在 `S = 1` 那一档上藏在 no-op 里**"
          % (d_noop, ratio_bad))
    print("     ★ 口径：`unscale` 的**时机**那条判据（`clip` 看到的是 unscaled 的范数）的"
          "正文在")
    print("       `90-training/mixed-precision/02-amp-execution.md` 的 `A.2c`/`A.2d` ——"
          " **本程序回链、不重定义**；")
    print("       本处只给「数值抓不到」这个**见证**，用来说明下一条为什么必须是结构钉住。")
    # ★★ A.3f：源码级次序（唯一能抓住「S=1 下的换序」的判据）
    src_step = inspect.getsource(train_step_correct)
    idx_unscale = src_step.index("/ scale")
    idx_clip = src_step.index("clip_global_norm")
    idx_lr = src_step.index("-lr * ")
    print("  ── A.3f 源码级次序：`unscale(/ scale)` @%d → `clip` @%d → `-lr *` @%d ──"
          % (idx_unscale, idx_clip, idx_lr))
    check("A.3f", idx_unscale < idx_clip < idx_lr,
          "★★ **源码里三个动作的次序严格是 `unscale → clip → lr`**（索引 %d < %d < %d）——"
          "★ 这一条是**结构钉住**（与 `02_amp.py` 的 `D.16` 同源：把口径钉在**源码**上），"
          "它关掉 `A.3e` 量出来的那个洞" % (idx_unscale, idx_clip, idx_lr))


# ===========================================================================
# 6. B 类：代价模型 —— 调度步长的总账、裁剪改的是什么
# ===========================================================================
def part_B():
    print()
    print("=" * 74)
    print("B. 代价模型：LR warmup 的「总账」、裁剪动的是范数还是方向")
    print("=" * 74)

    # ── B.1 总步长账 ────────────────────────────────────────────────────────
    print("  ── B.1 调度步长的总账：`Σ lr(t) = lr_max·T/2`（**与 W 无关**）──")
    print("     推导：cosine 段的 `Σ_{k=1}^{m} cos(πk/m) = −1`（`m = T−W`）⇒")
    print("       warmup 段 `Σ = lr_max·(W+1)/2`、cosine 段 `Σ = lr_max·(m−1)/2` ⇒ 两者之和 = `lr_max·T/2`")
    print("     %-6s %-6s %-18s %-18s %-12s" % ("W", "T", "实测 Σ lr", "闭式 lr_max·T/2", "相对差"))
    worst_tot = 0.0
    for W, T in zip(WARMUP_LIST, TOTAL_LIST):
        s = sum(lr_schedule(t, W, T, LR_MAX) for t in range(T))
        cl = total_steps_closed_form(LR_MAX, T)
        rel = abs(s / cl - 1.0)
        worst_tot = max(worst_tot, rel)
        print("     %-6d %-6d %-18.12f %-18.12f %-12.3e" % (W, T, s, cl, rel))
    s_alt = sum(lr_schedule(t, 10, 50, LR_MAX_SMALL) for t in range(50))
    print("  ★ 另一档 `lr_max = %g`：Σ = %.12f（= lr_max·T/2 逐位；★ 那个**数值** 2.5 单独钉住，"
          "否则这条只钉了「成正比」而 `LR_MAX_SMALL` 本身没有载荷）" % (LR_MAX_SMALL, s_alt))
    check("B.1c", abs(s_alt - 2.5) < TOL_TOTAL and LR_MAX_SMALL != LR_MAX,
          "★ 另一档的**数值**也钉住（Σ = %.12f == `lr_max·T/2 = 0.1·50/2 = 2.5`）⇒ "
          "「总账与 `lr_max` 成正比」这句话**不是**靠「两边同步变」蒙过去的" % s_alt)
    check("B.1", worst_tot < TOL_TOTAL and abs(s_alt / total_steps_closed_form(LR_MAX_SMALL, 50) - 1.0) < TOL_TOTAL,
          "★★ **LR warmup 不改变总步长预算，只改变分配**：`Σ lr(t)` **精确**等于 `lr_max·T/2`"
          "（跨 %d 组 `(W,T)` 最大相对差 %.3e < %g），**与 `W` 无关** —— "
          "这是「每个省都在别处记账」在这一格的形态：warmup 把前段**让出去**，"
          "cosine 把总量**补回来**" % (len(WARMUP_LIST), worst_tot, TOL_TOTAL))
    s_wrong = sum(lr_schedule_wrong_warmup(t, 10, 100, LR_MAX) for t in range(100))
    print("  ★ 反面对照：warmup 段用**常数** `lr_max`（而不是线性上升）⇒ Σ = %.6f ≠ 50"
          "（相对偏差 %.4f）" % (s_wrong, abs(s_wrong / 50.0 - 1.0)))
    check("B.1b", abs(s_wrong / 50.0 - 1.0) > TOL_MEASURABLE,
          "★ 反面对照：warmup 用常数 `lr_max` 时总账**破掉**（Σ = %.4f，偏差 %.4f > %g）⇒ "
          "`B.1` 的恒等式**依赖「线性 warmup + 末端锚定的 cosine」这一族**，不是恒真的"
          % (s_wrong, abs(s_wrong / 50.0 - 1.0), TOL_MEASURABLE))

    # ── B.2 裁剪只改范数、不改方向 ──────────────────────────────────────────
    print()
    print("  ── B.2 裁剪动的是**范数**，不是**方向** ──")
    g = make_grads(SEED + 51, target_norm=CLIP_NORM_T)
    g_c = clip_global_norm(g, CLIP_MAX_NORM)
    cos_sim = float(np.dot(g, g_c) / (global_norm(g) * global_norm(g_c)))
    ratio = global_norm(g_c) / global_norm(g)
    print("  方向余弦 = %.17f · 范数比 = %.12f（= `max_norm/(T+ε)` = %.12f）"
          % (cos_sim, ratio, CLIP_MAX_NORM / (CLIP_NORM_T + specs.TORCH_CLIP_COEF_DENOM_EPS)))
    check("B.2", abs(cos_sim - 1.0) < 1e-15
          and abs(ratio - CLIP_MAX_NORM / (CLIP_NORM_T + specs.TORCH_CLIP_COEF_DENOM_EPS)) < 1e-15,
          "★ 裁剪**只缩放**：方向余弦 == 1（逐位级），而范数比**恰好**等于那个系数 "
          "`max_norm/(T+ε)` ⇒ 它是**一个标量乘子**，代价只有一个（相对偏差 `ε/(T+ε)`）")
    print("  ★ 口径：这里量的是 `‖g_c‖/‖g‖`（**比值**），与 `A.2b` 的 `|‖g_c‖/max_norm − 1|`")
    print("    **是两个不同的量** —— 后者是「裁剪后离阈值多远」，前者是「缩了多少」。")


# ===========================================================================
# 7. C 类：边界与反例 —— 失败行为真的成立
# ===========================================================================
def part_C():
    print()
    print("=" * 74)
    print("C. 边界与反例：`inf` 会被裁剪放大成 `nan`；`max_norm = 0` 会把更新清零")
    print("=" * 74)

    # ── C.1 ★ `inf` 在梯度里 ⇒ 裁剪给出 `nan` ──────────────────────────────
    print("  ── C.1 ★ 梯度里有一个 `inf` 时，裁剪给出 `nan` ──")
    g = make_grads(SEED + 61, n=8).copy()
    g[0] = np.inf
    g_c = clip_global_norm(g, CLIP_MAX_NORM)
    n_nan = int(np.count_nonzero(np.isnan(g_c)))
    n_inf = int(np.count_nonzero(np.isinf(g_c)))
    n_zero = int(np.count_nonzero(g_c == 0.0))
    print("  裁剪前：`inf` 1 个 · 有限 7 个（范数 = %r）" % global_norm(g))
    print("  裁剪后：`nan` %d 个 · `inf` %d 个 · **精确 0** %d 个（`coef = max_norm/(inf+ε) = 0`）"
          % (n_nan, n_inf, n_zero))
    print("  ★ 机制：`coef = 0` ⇒ 有限元素 `x·0 = 0`、而 `inf·0 = nan`（IEEE 754）⇒")
    print("    整个梯度向量被**一个** `inf` 污染成 `nan`。")
    check("C.1", n_nan == 1 and n_zero == 7 and n_inf == 0,
          "★★ **裁剪不能替代溢出处置**：`inf` 进来、`nan` 出去（1 个 `nan` + 7 个 `0`）⇒ "
          "`w ← w − lr·nan = nan`，权重被污染且不可恢复 —— 所以 `mixed-precision/02` 的"
          "「溢出 ⇒ **跳步**（`A.3`）」必须在裁剪**之前**发生")
    check("C.1b", np.isfinite(make_grads(SEED + 61, n=8)).all(),
          "★ 阴性对照：**干净**的梯度裁剪后一个 `nan` 都没有 ⇒ 这一条不是恒真的")

    # ── C.2 `max_norm = 0` ⇒ 更新恒为 0 ────────────────────────────────────
    print()
    print("  ── C.2 `max_norm = 0`：梯度全清零、更新恒为 0 ──")
    g = make_grads(SEED + 63)
    g_c = clip_global_norm(g, 0.0)
    print("  裁剪前范数 = %.6f ⇒ 裁剪后范数 = %.1f（全部元素 == 0）"
          % (global_norm(g), global_norm(g_c)))
    dw, _ = train_step_correct(g, 1.0, 0.0, 0.1)
    print("  `Δw = −lr·g_c` 的范数 = %.1f" % global_norm(dw))
    check("C.2", global_norm(g_c) == 0.0 and global_norm(dw) == 0.0,
          "★ `max_norm = 0` ⇒ **梯度全清零**（范数精确 0）、更新也精确 0 ⇒ "
          "「阈值取 0」是一个**静默停训**的实现（失败行为真的成立，且它是静默的）")

    # ── C.3 `T = 0`（梯度全 0）时保护项**不会**误伤 ────────────────────────
    print()
    print("  ── C.3 阴面对照：梯度**恒为 0** 时保护项 `+ε` 不会误伤 ──")
    g_zero = np.zeros(N_GRAD)
    g_c0 = clip_global_norm(g_zero, CLIP_MAX_NORM)
    coef0 = CLIP_MAX_NORM / (0.0 + specs.TORCH_CLIP_COEF_DENOM_EPS)
    print("  `T = 0` ⇒ `coef = max_norm/(0+ε) = %.3e > 1` ⇒ **不缩放**（梯度仍是 0）" % coef0)
    check("C.3", coef0 > 1.0 and global_norm(g_c0) == 0.0,
          "★ `T = 0` 时保护项的作用是**防止 `0/0`**（`coef` 变成一个大数而不是 `inf`）⇒ "
          "它**不会**误伤「梯度恰好全为 0」的情形（判据不是恒真的）")


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

    measured = {"SEED": SEED, "TOTAL_STEPS": TOTAL_STEPS, "WARMUP_STEPS": WARMUP_STEPS,
                "LR_MAX": LR_MAX, "N_GRAD": N_GRAD, "CLIP_T_LIST": CLIP_T_LIST,
                "CLIP_MAX_NORM": CLIP_MAX_NORM, "CLIP_MAX_NORM_LIST": CLIP_MAX_NORM_LIST,
                "CLIP_NORM_T": CLIP_NORM_T, "WARMUP_LIST": WARMUP_LIST,
                "TOTAL_LIST": TOTAL_LIST, "CLIP_LR_LIST": CLIP_LR_LIST,
                "SCHEDULE_T_LIST": SCHEDULE_T_LIST, "LR_MAX_SMALL": LR_MAX_SMALL,
                "CLIP_T_SMALL": CLIP_T_SMALL, "CLIP_LR_BIG": CLIP_LR_BIG}
    expect = {"SEED": 20260930, "TOTAL_STEPS": 100, "WARMUP_STEPS": 10, "LR_MAX": 1.0,
              "N_GRAD": 64, "CLIP_T_LIST": (2.0, 4.0, 8.0), "CLIP_MAX_NORM": 1.0,
              "CLIP_MAX_NORM_LIST": (1.0, 0.5, 0.1), "CLIP_NORM_T": 10.0,
              "WARMUP_LIST": (1, 10, 25, 50), "TOTAL_LIST": (100, 50),
              "CLIP_LR_LIST": (0.01, 0.05), "SCHEDULE_T_LIST": (10, 50, 100, 7),
              "LR_MAX_SMALL": 0.1, "CLIP_T_SMALL": 0.5, "CLIP_LR_BIG": 0.1}
    diff = {k: (expect[k], v) for k, v in measured.items() if v != expect[k]}
    check("D.3", not diff, "★ 口径常量 16 个逐位一致（结论的**测量条件**）%s"
          % ("" if not diff else " ← 差异 %s" % diff))
    check("D.3e", CLIP_LR_BIG * CLIP_NORM_T == CLIP_MAX_NORM
          and CLIP_LR_BIG not in CLIP_LR_LIST,
          "★★ `CLIP_LR_BIG × CLIP_NORM_T == max_norm` **精确成立**（%.2f）—— 它是「`lr·T` **恰好**"
          "等于 `max_norm`」的**边界档**，因此**必须**排在 `A.3c` 的两档之外（那两档要求 "
          "`lr·T < max_norm`）⇒ 这个常量有**载荷**，不是只用来打印的"
          % (CLIP_LR_BIG * CLIP_NORM_T))
    check("D.3b", min(CLIP_T_LIST) >= 1.0 and CLIP_T_SMALL < CLIP_MAX_NORM
          and all(T > CLIP_MAX_NORM for T in CLIP_T_LIST),
          "★ **`T ≥ 1` 与 `T > max_norm` 这两条取值范围**都成立（`T ∈ %s`、`max_norm ∈ %s`）——"
          "它们是判据的一部分：`T < 1` 时 `1.1e-6/T` 这个上界不成立（**回链 Chunk 1 的 C.3**）"
          % (CLIP_T_LIST, CLIP_MAX_NORM_LIST))
    check("D.3c", WARMUP_STEPS > 1 and LR_MAX / lr_schedule(0, WARMUP_STEPS, TOTAL_STEPS, LR_MAX) == WARMUP_STEPS,
          "★ `A.3d` 的「放大 `W` 倍」全靠 `lr_max/lr(0) == W` 这件事（%g）"
          % (LR_MAX / lr_schedule(0, WARMUP_STEPS, TOTAL_STEPS, LR_MAX)))
    check("D.3d", CLIP_NORM_T > CLIP_MAX_NORM and all(lr * CLIP_NORM_T < CLIP_MAX_NORM for lr in CLIP_LR_LIST),
          "★ `A.3c` 的两档 `lr` **全落在 `lr·T < max_norm`** 上（`%.2f < %.1f`）——"
          "那是「错序完全不裁剪」的前提；用 `lr = %g`（边界档）时 `lr·T == max_norm`，"
          "那正是 `CLIP_LR_BIG` 单列出来的原因"
          % (max(CLIP_LR_LIST) * CLIP_NORM_T, CLIP_MAX_NORM, CLIP_LR_BIG))

    th = {"TOL_SCHED": TOL_SCHED, "TOL_EXACT": TOL_EXACT, "TOL_PROP": TOL_PROP,
          "TOL_RATIO": TOL_RATIO, "TOL_W_CROSS": TOL_W_CROSS, "TOL_TOTAL": TOL_TOTAL,
          "TOL_MEASURABLE": TOL_MEASURABLE, "CLIP_TOL_REL": CLIP_TOL_REL,
          "CLIP_TOL_CLOSED": CLIP_TOL_CLOSED}
    th_expect = {"TOL_SCHED": 1e-15, "TOL_EXACT": 0.0, "TOL_PROP": 1e-15,
                 "TOL_RATIO": 1e-12, "TOL_W_CROSS": 1e-12, "TOL_TOTAL": 1e-12,
                 "TOL_MEASURABLE": 1e-2, "CLIP_TOL_REL": 1.1e-6,
                 "CLIP_TOL_CLOSED": 1e-6}
    tdiff = {k: (th_expect[k], v) for k, v in th.items() if v != th_expect[k]}
    check("D.4", not tdiff, "★ 阈值 9 个逐位一致（含**两个 0 容差**：`A.3a` 的逐位相同与 "
                            "`TOL_EXACT`；★ `CLIP_TOL_CLOSED` 就是 `A.2b` 那两侧判据的下侧，"
                            "**回链 Chunk 1 的 T26**）%s" % ("" if not tdiff else " ← 差异 %s" % tdiff))
    return len(FAILS) == before


# ===========================================================================
# 9. D 类（后半）：specs 常量逐位钉住 + 独立性证据
# ===========================================================================
def part_D():
    print()
    print("=" * 74)
    print("D. specs 常量逐位钉住 与 独立性证据")
    print("=" * 74)

    check("D.5", specs.TORCH_CLIP_COEF_DENOM_EPS == 1e-6,
          "★★ 裁剪分母的保护项 `ε == 1e-6`（[B] **PyTorch v2.9.0 的源码**，不是规范）——"
          "★ 本程序**只消费、不重定义**：判据与阈值（含 `T ≥ 1`、上侧 `1.1e-6/max(T,1)`、"
          "下侧 `== ε/(T+ε)` 的 `A.2b`）在 `autograd-and-backward/03` 的**表 A ⑦** 写一次")
    check("D.5b", abs(CLIP_TOL_REL / 1.1 - 1e-6) < 1e-18 or CLIP_TOL_REL == 1.1e-6,
          "★ 上侧的阈值口径 `CLIP_TOL_REL = 1.1e-6` 与本程序一致（真正的判据是 "
          "`CLIP_TOL_REL / max(T, 1)`）")
    check("D.6", (specs.ADAM_DEFAULT_LR, specs.ADAM_DEFAULT_BETA1, specs.ADAM_DEFAULT_BETA2,
                  specs.ADAM_DEFAULT_EPS) == (0.001, 0.9, 0.999, 1e-8),
          "★ Adam 的四个默认超参逐位一致（[C] 1412.6980，**已确认 ICLR 2015**）——"
          "★ 本程序**不用**它们（本程序的 `lr_max` 是调度峰值），钉住是为了三篇同值")
    check("D.6b", specs.ADAMW_DEFAULT_WEIGHT_DECAY == 0.01
          and specs.PYTORCH_ADAM_DEFAULT_WEIGHT_DECAY == 0,
          "★ AdamW 论文 Algorithm 2 的**第 11 行**逐字是 `η_t ← SetScheduleMultiplier(t)`，"
          "注释 `can be fixed, decay, or also be used for warm restarts` ⇒ **论文自己把"
          "「学习率调度」写进了优化器的算法框** —— 这正是本程序（`03`）与 `02` 的**接口**；"
          "而两个 `weight_decay` 默认值成对钉住（`0` vs `0.01`）")
    check("D.7", specs.UNSOURCED == ("adam_eps_position_raw_count",
                                     "pytorch_adam_float32_eps_version_span",
                                     "muon_orthogonalization_iteration_count",
                                     "muon_update_rms_measurement_convention"),
          "★ `specs.UNSOURCED` 的四项**逐位钉住**（%d 项）" % len(specs.UNSOURCED))

    # ── 独立性：三条路径 ───────────────────────────────────────────────────
    src_s = inspect.getsource(lr_schedule)
    src_h = inspect.getsource(lr_schedule_half_angle)
    src_w = inspect.getsource(lr_schedule_wrong_warmup)
    check("D.8", len({src_s, src_h, src_w}) == 3,
          "★ 防 P17：被测（`lr_schedule` %d 字符）、参照（`lr_schedule_half_angle` %d）、"
          "反面对照（`lr_schedule_wrong_warmup` %d）**源码两两不同**"
          % (len(src_s), len(src_h), len(src_w)))
    watched = {"lr_schedule", "clip_global_norm", "train_step_correct",
               "train_step_clip_after_lr"}
    called = _callee_names(lr_schedule_half_angle) | _callee_names(total_steps_closed_form)
    check("D.9", not (watched & called),
          "★ **AST 调用表**：两条**参照路径**的 `Call` 节点（`ast.Name` 形态）= %s —— "
          "一条都没调到被测函数" % (", ".join(sorted(called)) if called else "**空集**"))
    check("D.9b", "math.cos" in src_s and "math.cos" in src_h and "0.5 * math.pi" in src_h,
          "★ 阳性对照（**源码级**，因为 `math.cos` 是**属性调用**、不进 `ast.Name` 的调用表）："
          "两条路径都用了 `math.cos`，而 cosine 的**算式不同**（`1 + cos(π·k/m)` vs "
          "`cos(0.5·π·k/m)²`）⇒ `D.8` 的「源码两两不同」不是装饰")
    check("D.10", max_absdiff(
        [lr_schedule(t, WARMUP_STEPS, TOTAL_STEPS, LR_MAX) for t in range(TOTAL_STEPS)],
        [lr_schedule_half_angle(t, WARMUP_STEPS, TOTAL_STEPS, LR_MAX) for t in range(TOTAL_STEPS)]) < TOL_SCHED,
        "★ 两条算式在整条调度上处处一致（这是 `A.1b` 的口径：**算式不同、数相同**）")
    check("D.11", MARKS_USED <= {ASSERT_FAIL_MARK},
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
    print("03_schedule_clip.py —— 学习率调度 与 梯度裁剪（它在训练管线里的位置）")
    print("=" * 74)
    print("  ★ 真值标注：**全部是推导值**（本机 `python3 %s` + `numpy %s`，macOS arm64 无 GPU）。"
          % (sys.version.split()[0], np.__version__))
    print("  ★ 失败标记 ASSERT_FAIL_MARK = %r（本主题自定义；不跨主题照抄）" % ASSERT_FAIL_MARK)
    print("  ★★ **同名不同指（M9）**：本程序写的是 **LR warmup（学习率 warmup）** ——")
    print("     它与 `00-meta/glossary.md:146` 已收录的 `warmup`（**benchmark 预热**，")
    print("     `45-kernel-lifecycle/.../01-benchmark-hygiene.md` 里 23 次）**不是同一件事**。")
    print("     本程序**一律写全 `LR warmup`**，并带上这句限定语（glossary 的改名是收尾 chunk 的动作）。")
    print("  ★★ **边界声明跟着知识点走**：`LR warmup` 的**机制解释**（post-LN 的平均场理由、")
    print("     `hypothesize` 的措辞纪律）**正文在**")
    print("     `80-model-architectures/normalization/02-placement-and-qk-norm.md:83-100` ——")
    print("     本程序**不重写**它，只做**工程判据与设定**（形状 / 峰值 / 末值 / 总账）。")
    print("  ★ 禁区（一律回链，本程序不写）：优化器状态的**分片与通信量**在")
    print("    `100-parallelism/README.md:14`（`zero-fsdp`）；DDP / 梯度规约在 `:9` 与 `110-communication/`。")
    print("  ★ 回链（**只实现、不重定义**）：裁剪的判据与阈值在")
    print("    `autograd-and-backward/03-gradient-lifecycle.md` 的表 A ⑦（**含 `A.2b` 的两侧判据**，T26）；")
    print("    `unscale` 的时机在 `mixed-precision/02-amp-execution.md` 的 `A.2`/`A.2c`/`A.2d`。")
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

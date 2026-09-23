#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""02_amp.py —— AMP 的执行约定：loss scaling 的算术、**unscale 的时机**、溢出的处置。

配套文档：`02-amp-execution.md`。
★ 三类断言（A 算法正确性 / B 代价模型 / C 边界与反例）**全部在本程序内部**，
失败时以**非零退出码**结束；**没有 `verify.py`**（`CONTRIBUTING` §2 表里的第二种情形）。

★★ **本程序写的是「某实现的规格」—— 一律标 B 级并写明是哪个实现**（计划 §0.1 的 **P19**）。
   本域的头号错误模式就是**把「某实现的文档」当成「规范的文档」**。本程序涉及的三个实现：
     · **Apex** 的 `O0/O1/O2/O3`（★ 它**不在** NVIDIA 的 AMP 页里：那页 `whitelist` /
       `master_weights` 各 **0** 次，而 Apex 页里是 1 / 9 次）；
     · **PyTorch** 的 `GradScaler`（`init_scale=65536.0`、`growth_interval=2000` …）；
     · **NVIDIA 的 AMP FAQ** 的 loss scale 举例（`say, 2^24`）。
   ★★ 而本程序**一行都不 import 这些实现**（`make lint` 用 AST 查这件事）：
     它把它们**文档里逐字写的规则**当作 `specs.py` 的 [B] 常量，再在 numpy 上**重演一遍**。
     ⇒ 因此本程序的数**不是「PyTorch 在本机跑出来的」**，而是
       「按 PyTorch / Apex 的**文档**在本机推出来的」⇒ 标为**推导值**。

★★ 本程序的数字全部是推导值，测量条件是：
> `python3 3.13.5` + `numpy 2.4.4`，macOS arm64，**无 GPU**。
> 玩具规模的真跑，**不得支撑关于真实训练系统的结论**。

★★ 本主题的**禁区**（`CONTRIBUTING` §7 + 域 README 的边界表）：
  · 「**为什么 fp16 要 loss scaling、bf16 为什么不用**」**整块** →
    `10-foundations/floating-point/04-low-precision-formats.md`；
  · **「loss scaling 只有收益、没有代价」**（上溢侧、次正规位不再回来）→ 同上 `:197,229-231`；
  · **tf32 的开关与默认值史** → 同上 `:215-223`；
  · **框架的精度 API 语义**（`autocast` 的算子白名单 / 黑名单）→ `70-frameworks/pytorch/02-autograd.md`
    与 `70-frameworks/` 各框架 —— **各框架互不相同**，属框架 API 语义；
  · **梯度裁剪的判据与阈值** → 本域 `autograd-and-backward/03-gradient-lifecycle`（那里写死了阈值）。
  ⇒ 本程序**一行都不重写**这些。它写的是**训练循环里的顺序与算术**：
    「哪一步做什么、unscale 放在哪里、溢出时怎么办」。

★★ 可证伪断言（把下面任一条改错，本程序必须非零退出）：
  · `A.1`  在**正规区**内 `×2^k` 再 `÷2^k` 是**恒等**（逐位相等）；把 `S` 换成非 2 的幂立刻失败；
  · `A.2`  ★★ **unscale 的时机**：`unscale → clip` 与 `clip → unscale` 给出的梯度范数之比
            **恰好是 `S`**（相对差 < 1e-6）—— 顺序错了就**多裁 `S` 倍**；
  · `B.1`  loss scaling 把下溢的梯度抬回来（`1e-7` 的相对误差 < 1e-5），
           而**它也有边界**（`1e-9` 的相对误差 > 1e-3）；
  · `C.1`  溢出**不处置**时权重被 `inf` 污染且**后续步也回不来**（失败行为真的成立）。

用法：
    python3 02_amp.py
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
# ★★ 失败标记：本主题（mixed-precision）取 `:: FAIL: mixed-precision`。
#   ★ `mutate.py` 用 **AST** 从本文件读出这个值（不得写死）；`D.1`/`D.2` 两条断言守着它。
ASSERT_FAIL_MARK = ":: FAIL: mixed-precision"

# 本域另外四个主题的失败标记 —— **只用来验「不照抄」**，本程序不打印它们。
OTHER_THEME_MARKS = (".. FAIL: autograd", "++ FAIL: optimizers",
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
SEED = 20260926              # ★ 与 `01_types.py`（20260925）和探测脚本（90）都**不同**
N_GRAD = 4096                # 一个「梯度张量」的元素个数
SCALE = 1024.0               # = 2^10 —— ★ **必须是 2 的幂**（A.1 的判据就是这件事）
SCALE_NOT_POW2 = 1000.0      # 对照：非 2 的幂（A.1 的反面对照）
MAX_NORM = 1.0               # 梯度裁剪阈值（★ 判据的**阈值定义**在 `autograd-and-backward/03`，此处只消费）
GRAD_SIGMA = 2.0             # 「真」梯度 ~ N(0, GRAD_SIGMA)
# ★ 一批**落在 fp16 归零区**的梯度（判据⑦ / B.1 用）：
#   `1e-8` 与 `1e-9` 在 fp16 里**恰好变成 0**（`1e-7` 还在次正规区）。
GRAD_TINY = (1e-7, 1e-8, 1e-9)
W_INIT = 1.0                 # 权重的初值（溢出处置那一节用）
LR_STEP = 0.01               # 那一步的学习率

TOL_EXACT = 0.0              # A.1：正规区内的恒等是**逐位**的（容差 0）
TOL_UNSCALE_RATIO = 1e-6     # A.2：两种顺序的范数之比与 S 的相对差
TOL_TINY_GOOD = 1e-5         # B.1：×1024 后 `1e-7` 的相对误差上界
TOL_TINY_BAD = 1e-3          # B.1：`1e-9` 的相对误差下界（scaling 的边界）
TOL_POW2_BITWISE = 0         # A.1b：`S` 是 2 的幂时才逐位恒等
S_HISTORY_K = 8              # B.2：delayed scaling 的历史长度（★ 本程序的选择，不是引文）
MIN_REGION_FACTOR = 1.0      # A.1c：P22 检查 —— 参与「恒等」判据的 |g| 至少要是最小正规数的几倍
# ★ 归零阈值 = **最小次正规的一半**（舍入到最近）。★ 与 `01-type-selection` 的
#   `thr_over_two()` **同值**：`code/` 必须自包含（`CONTRIBUTING:36`），**各写一份**，
#   靠这一行互指、靠 `D.3` 钉住。写成**字面量**（`2.0 ** -25` 的 `**` 会让 `mutate.py`
#   的 `_walk` 下钻不到 ⇒ 那个值会整体逃出变异分母）。
F16_ZERO_THRESHOLD = 2.9802322387695312e-08   # = 2^-25 = 5.960464477539063e-08 / 2
S_OVERFLOW_LO = 2.0 ** 10    # C.2：两个 S 档（一个不溢出、一个溢出）
S_OVERFLOW_HI = 2.0 ** 13
S_STEPS = 40                 # C.1/B.2 的步数


# ===========================================================================
# 2. 路径 1（被测）：一次 AMP 步 —— 按 PyTorch 文档的**顺序**做
# ===========================================================================
def amp_step(w_master, grads_fp16_scaled, scale, max_norm, lr, unscale_first=True):
    """一次 AMP 步，**按 [C] Micikevicius 2017 的顺序**：

    ```
    forward  →  loss ← loss × S      （在 fp16 前向之后、反向之前）
    backward →  g_scaled             （fp16 存）
    ── ★★ 这一步就是判据 A.2 的对象 ──
    unscale： g ← g_scaled / S       （"right after the backward pass but before gradient clipping"）
    clip   ： 按全局范数缩放到 max_norm（阈值定义在 `autograd-and-backward/03`，此处只消费）
    step   ： w ← w − lr·g           （在 **fp32 主权重**上做）
    ```

    `unscale_first=False` 时把 unscale 挪到 clip **之后** —— 那是**错**的顺序（A.2 的对照）。

    返回 `(w_new, g_after_clip, norm_before_clip, norm_after_clip)`。
    """
    g = np.asarray(grads_fp16_scaled, dtype=np.float64)
    if unscale_first:
        g = g / scale
    norm_before = global_norm(g)
    g = clip_global_norm(g, max_norm)
    norm_after = global_norm(g)
    if not unscale_first:
        g = g / scale
    return w_master - lr * g, g, norm_before, norm_after


def global_norm(g):
    """全局范数（把**全部参数**的梯度看成一个向量）。"""
    g = np.asarray(g, dtype=np.float64)
    return math.sqrt(float((g ** 2).sum()))


def clip_global_norm(g, max_norm, eps=specs.TORCH_CLIP_COEF_DENOM_EPS):
    """按全局范数缩放。★ **`(T + 1e-6)` 这个分母口径是 [B] PyTorch 的**（v2.9.0 源码），

    不是规范：裁剪后的范数 = `max_norm · T/(T + 1e-6)`，**是一个近似**。
    ★★ 本程序**只消费**这个阈值，**不重新定义**它 —— 它的正文与判据在
      `90-training/autograd-and-backward/03-gradient-lifecycle.md`（**在那里只写一次**）。
    """
    total = global_norm(g)
    coef = max_norm / (total + eps)
    if coef < 1.0:
        return np.asarray(g, dtype=np.float64) * coef
    return np.asarray(g, dtype=np.float64)


# ===========================================================================
# 3. 路径 2（参照）：**不做 loss scaling** 的同一步
# ===========================================================================
def plain_step(w_master, grads, max_norm, lr):
    """不做任何缩放：直接 unscale（无操作）→ clip → step。用来给 A.2 一个**独立参照**。"""
    g = np.asarray(grads, dtype=np.float64)
    g = clip_global_norm(g, max_norm)
    return w_master - lr * g, g


# ===========================================================================
# 4. loss scaling 的两条路：`×S` 与 `÷S` 的算术
# ===========================================================================
def scale_then_unscale(g, scale, dtype=np.float16):
    """`g` → `dtype(g·S)` → `÷S`（在 fp64 里做，因为真实实现的 unscale 在 fp32 累加器上）。

    返回 `(缩放后的 dtype 张量, 还原后的 fp64 张量)`。
    ★★ **两步都要显式写 dtype**：NumPy 2.x 的 NEP 50 弱标量提升**不会**把
      `dtype 数组 / Python float` 提升到 fp64 ⇒ 若写成 `h / scale`，
      除法**仍在 fp16 里做**，小值会在这一步**静默变成 0**。本程序第一版就撞上了这件事
      （见 `A.1b` 的反面对照 —— 那一条现在是被**刻意**留下来的）。
    """
    h = np.asarray(np.asarray(g, dtype=np.float64) * scale, dtype=dtype)
    back = np.asarray(h, dtype=np.float64) / scale
    return h, back


def overflow_detected(h):
    """溢出检测：`inf` 或 `nan` 出现即算溢出。★ 这是 [C] 原文说的
    `overflows can be efficiently detected by inspecting the computed weight gradients`。"""
    a = np.asarray(h, dtype=np.float64)
    return bool(np.count_nonzero(~np.isfinite(a)))


def next_scale_after_overflow(scale, backoff=specs.PYTORCH_GRADSCALER_BACKOFF):
    """溢出后的缩放因子：乘上 `backoff`。★ 默认值来自 **[B] PyTorch** 的 `GradScaler`。"""
    return scale * backoff


def next_scale_after_clean(steps_since_growth, scale,
                           growth=specs.PYTORCH_GRADSCALER_GROWTH,
                           interval=specs.PYTORCH_GRADSCALER_GROWTH_INTERVAL):
    """没有溢出、且**连续 `interval` 步**都没有溢出时，缩放因子乘上 `growth`。"""
    if steps_since_growth >= interval:
        return scale * growth, 0
    return scale, steps_since_growth + 1


def recovery_steps(growth=specs.PYTORCH_GRADSCALER_GROWTH,
                   backoff=specs.PYTORCH_GRADSCALER_BACKOFF,
                   interval=specs.PYTORCH_GRADSCALER_GROWTH_INTERVAL):
    """推导值：一次溢出把 `S` 乘 `backoff`，要回到原值需要几次「增长」、共多少步。

    `backoff = 1/growth = 0.5` ⇒ **一次增长就回到原值** ⇒ 步数 = `interval`。
    """
    n_growths = int(round(math.log(1.0 / backoff) / math.log(growth)))
    return n_growths, n_growths * interval


def scale_window(grads, lo_thr, hi_max):
    """推导值：`S` 的**可用窗口**。

    ```
    下界：S 要大到让最小的重要梯度不落进归零区   ⇒  S_min = thr / min|g|（非零者）
    上界：S 不能让最大梯度溢出                   ⇒  S_max = max_finite / max|g|
    ```

    ★ 上界那句的依据是 [C] Micikevicius 2017 逐字：
      `choosing a factor so that its product with the maximum absolute gradient value is
       below 65,504 (the maximum value representable in FP16)`
    """
    a = np.abs(np.asarray(grads, dtype=np.float64))
    a = a[a > 0.0]
    if a.size == 0:
        return 0.0, 0.0
    return lo_thr / float(a.min()), hi_max / float(a.max())


# ===========================================================================
# 5. 工具
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


def make_grads(sigma=GRAD_SIGMA, n=N_GRAD, seed=SEED):
    """一批「真」梯度：`N(0, sigma)`。★ 只用一条种子，**换种子会换数**（报数时写明种子）。"""
    return np.random.default_rng(seed).standard_normal(n) * sigma


# ===========================================================================
# 6. A 类：unscale 的算术与**时机**
# ===========================================================================
def part_A(grads):
    print("=" * 74)
    print("A. unscale 的算术（×2^k ÷2^k 恒等）与**时机**（它在 clip 之前）")
    print("=" * 74)
    print("  口径：seed=%d · %d 个梯度元素 · 梯度 ~ N(0, %g) · S = %g (= 2^%d) · max_norm = %g"
          % (SEED, N_GRAD, GRAD_SIGMA, SCALE, int(math.log2(SCALE)), MAX_NORM))

    # ── A.1 ★ 在正规区内，×2^k ÷2^k 是**恒等**（逐位）────────────────────────
    print("  ── A.1 正规区内的恒等：`f16(S·g)/S == f16(g)` ──")
    g_f16 = np.asarray(grads, dtype=np.float16)
    _h, back = scale_then_unscale(grads, SCALE)
    both_normal = (np.abs(np.asarray(g_f16, dtype=np.float64)) >= specs.F16_MIN_NORMAL)
    eq = (np.asarray(back, dtype=np.float64) == np.asarray(g_f16, dtype=np.float64))
    print("  参与判据的元素 %d / %d（其余落进次正规或归零区，见 A.1b/A.1c）"
          % (int(both_normal.sum()), N_GRAD))
    print("  ★ 参照量级（**P22**）：参与判据的 |g| 最小 = %.6e，fp16 最小**正规**数 = %.6e ⇒ %.2f×"
          % (float(np.abs(grads[both_normal]).min()), specs.F16_MIN_NORMAL,
             float(np.abs(grads[both_normal]).min()) / specs.F16_MIN_NORMAL))
    check("A.1c", bool(both_normal.any())
          and float(np.abs(grads[both_normal]).min()) >= MIN_REGION_FACTOR * specs.F16_MIN_NORMAL,
          "★ **P22 检查**：参与「恒等」判据的量**全部落在正规区**（最小 %.6e ≥ %.6e）⇒ "
          "「逐位相等」测的是**恒等**，不是次正规噪声"
          % (float(np.abs(grads[both_normal]).min()), specs.F16_MIN_NORMAL))
    check("A.1", bool(eq[both_normal].all()),
          "★ 正规区内 `f16(S·g)/S` 与 `f16(g)` **逐位相等**（%d / %d 个元素，容差 %g）——"
          "因为 `S = 2^k` 只改指数、不改尾数" % (int(eq[both_normal].sum()),
                                                int(both_normal.sum()), TOL_EXACT))

    # ── A.1b 反面：`S` 不是 2 的幂时，**恒等不成立** ─────────────────────────
    print("  ── A.1b 反面对照：`S = %g`（**不是 2 的幂**）──" % SCALE_NOT_POW2)
    _h2, back2 = scale_then_unscale(grads, SCALE_NOT_POW2)
    g_notpow = np.asarray(grads, dtype=np.float16)
    neq = np.count_nonzero(np.asarray(back2, dtype=np.float64)
                           != np.asarray(g_notpow, dtype=np.float64))
    print("  与 `f16(g)` 逐位不同的元素 = %d / %d" % (neq, N_GRAD))
    check("A.1b", neq > 0,
          "★ `S` 不是 2 的幂时，`f16(S·g)/S` 与 `f16(g)` **不再逐位相等**（%d 个元素不同）"
          "⇒ 「×S ÷S 是恒等」这句话**依赖 `S` 是 2 的幂**（那就是为什么损失缩放取 "
          "2 的幂：`÷S` 不许引入新的舍入）" % neq)

    # ── A.2 ★★ 判据：unscale 的**时机** ─────────────────────────────────────
    print()
    print("  ── A.2 ★★ unscale 的时机：`unscale → clip` vs `clip → unscale` ──")
    print("     依据（[C] Micikevicius 2017，逐字，已确认 ICLR 2018）：")
    print("     %r" % specs.MICIKEVICIUS_UNSCALE_QUOTE)
    g_scaled = np.asarray(np.asarray(grads, dtype=np.float64) * SCALE, dtype=np.float16)
    _w_r, g_right, nb_r, na_r = amp_step(W_INIT, g_scaled, SCALE, MAX_NORM, LR_STEP, True)
    _w_w, g_wrong, nb_w, na_w = amp_step(W_INIT, g_scaled, SCALE, MAX_NORM, LR_STEP, False)
    n_right = global_norm(g_right)
    n_wrong = global_norm(g_wrong)
    ratio = n_right / n_wrong
    print("  正确顺序（unscale 先）：裁剪前范数 %.6f → 裁剪后范数 **%.10f**" % (nb_r, n_right))
    print("  错误顺序（unscale 后）：裁剪前范数 %.6f → unscale 后范数 %.10f" % (nb_w, n_wrong))
    print("  两者之比 = %.6f（= S = %g，相对差 %.3e）"
          % (ratio, SCALE, abs(ratio / SCALE - 1.0)))
    check("A.2", abs(ratio / SCALE - 1.0) < TOL_UNSCALE_RATIO,
          "★ 两种顺序给出的梯度范数之比**恰好是 `S`**（%.6f vs %g，相对差 %.3e < %g）⇒ "
          "顺序错了就**多裁 `S` 倍**" % (ratio, SCALE, abs(ratio / SCALE - 1.0), TOL_UNSCALE_RATIO))
    print("  ★ 口径：`+1e-6` 那个分母项（[B] PyTorch 的裁剪口径）让比值不是**精确**的 `S`，")
    print("    而是 `S·(1+O(1e-6/T))` —— 本程序不把它抹平，**如实报**。")
    # ★ 与「不做缩放」的独立参照对照：正确顺序的结果**就该**等于不缩放那一步
    _w_p, g_plain = plain_step(W_INIT, grads, MAX_NORM, LR_STEP)
    rel_right = global_norm(g_right - g_plain) / global_norm(g_plain)
    rel_wrong = global_norm(g_wrong - g_plain) / global_norm(g_plain)
    print("  与「不做缩放」的独立参照比：正确顺序相对差 %.3e · 错误顺序相对差 %.3e"
          % (rel_right, rel_wrong))
    check("A.2b", rel_right < rel_wrong / 100.0,
          "★ 与**不做缩放**的同一步对照：正确顺序的相对差（%.3e）比错误顺序（%.3e）小两个数量级以上"
          % (rel_right, rel_wrong))
    # ★★ **A.2c/A.2d 是本程序补的第二把锁（一个真的漏网换来的）**：
    #   只查「两者之比 == S」时，有一个**方向盲**的假实现照样能过 ——
    #   把 unscale 挪到 clip **之后**并且**做两次**（`÷S` 两次、前面一次也不做），
    #   比值**仍然是 S**（本程序在结构翻转抽查里实测：rc=0、断言全过）。
    #   ⇒ 补一条**直接**的判据：**裁剪看到的是不是 unscaled 的范数**。
    #   这一条是**两侧**的：正确顺序下它必须 ≈ 真范数；错误顺序下它必须 ≈ S 倍。
    true_norm = global_norm(grads)
    print("  裁剪**看到**的范数：正确顺序 %.6f · 错误顺序 %.6f · 真范数 %.6f"
          % (nb_r, nb_w, true_norm))
    check("A.2c", abs(nb_r / true_norm - 1.0) < 1e-4,
          "★ 正确顺序下，**裁剪看到的就是 unscaled 的范数**（%.6f vs 真值 %.6f，相对差 %.3e）"
          "—— 这条断言直接说「unscale 发生在 clip **之前**」，不是只查一个比值"
          % (nb_r, true_norm, abs(nb_r / true_norm - 1.0)))
    check("A.2d", abs(nb_w / true_norm - SCALE) / SCALE < 1e-4,
          "★ 而错误顺序下，裁剪看到的是**放大 `S` 倍**的范数（%.6f = %.1f × %.6f）—— "
          "两条合起来才是**两侧**判据" % (nb_w, nb_w / true_norm, true_norm))
    # ★★ **A.2e 是第三把锁**：`+1e-6` 那个分母项**真的在起作用**。
    #   只查「比例 == S」时，把 `eps` 换成 `0`（裁剪变成**精确**的）照样过 ——
    #   本程序在结构翻转抽查里实测到过这一点（rc=0）。补一条**两侧**的：
    #   裁剪后范数必须**严格小于** `max_norm`，且等于闭式 `max_norm·T/(T+1e-6)`。
    closed_clip = MAX_NORM * nb_r / (nb_r + specs.TORCH_CLIP_COEF_DENOM_EPS)
    print("  裁剪后范数 = %.12f（`max_norm` = %g）· 闭式 `max_norm·T/(T+1e-6)` = %.12f"
          % (na_r, MAX_NORM, closed_clip))
    check("A.2e", na_r < MAX_NORM and abs(na_r - closed_clip) < 1e-12,
          "★ 裁剪后范数**严格小于** `max_norm`（%.12f < %g）且与闭式 `max_norm·T/(T+1e-6)` 一致 —— "
          "把 `eps` 换成 0 会让它**精确等于** `max_norm`，这条立刻失败" % (na_r, MAX_NORM))

    # ── A.3 溢出检测与跳步 ──────────────────────────────────────────────────
    print()
    print("  ── A.3 溢出检测与跳步：把梯度里塞一个 `inf` ──")
    bad = np.asarray(g_scaled, dtype=np.float64).copy()
    bad[0] = np.inf
    detected = overflow_detected(bad)
    s_before = SCALE
    s_after = next_scale_after_overflow(s_before) if detected else s_before
    w_before = W_INIT
    w_after = w_before if detected else w_before - LR_STEP * global_norm(bad)
    print("  检测到溢出 = %s ⇒ 跳过该步、`S: %g → %g`" % (detected, s_before, s_after))
    check("A.3", detected and w_after == w_before and s_after == s_before / 2.0,
          "★ 检测到溢出 ⇒ **权重精确不变**（%.6f）且 `S` **精确减半**（%g → %g）—— "
          "`×0.5` 对 2 的幂是精确的" % (w_after, s_before, s_after))
    check("A.3b", not overflow_detected(g_scaled),
          "★ 阴性对照：**干净**的梯度里一个非有限值都没有（判据不是恒真的）")


# ===========================================================================
# 7. B 类：代价模型 —— scaling 的收益、它的边界、它的恢复时间
# ===========================================================================
def part_B():
    print()
    print("=" * 74)
    print("B. 代价模型：scaling 抬回了什么、它的边界在哪、一次溢出要多久才恢复")
    print("=" * 74)

    # ── B.1 ★ 判据⑦：loss scaling 把下溢的梯度抬回来，**且它有边界** ─────────
    print("  ── B.1 判据⑦：一批落在 fp16 归零区的梯度，`×%g` 前后的相对误差 ──" % SCALE)
    print("  %12s %16s %16s %18s %14s" % ("g", "不缩放：f16(g)", "×S 后 ÷S", "相对误差(缩放)", "不缩放的误差"))
    worst_good = 0.0
    worst_bad = 0.0
    for gv in GRAD_TINY:
        raw = float(np.float16(gv))
        _h, back = scale_then_unscale(np.array([gv]), SCALE)
        b = float(back[0])
        rel_s = abs(b - gv) / gv
        rel_raw = abs(raw - gv) / gv
        print("  %12.0e %16.3e %16.3e %18.3e %14.3e" % (gv, raw, b, rel_s, rel_raw))
        if gv == GRAD_TINY[0]:
            worst_good = rel_s
        if gv == GRAD_TINY[-1]:
            worst_bad = rel_s
    check("B.1", worst_good < TOL_TINY_GOOD and worst_bad > TOL_TINY_BAD,
          "★ `%g` 缩放后相对误差 %.3e < %g（**抬回来了**）；而 `%g` 的是 %.3e > %g "
          "（**scaling 也有边界**：它只是移位，救不回已经落到底下的量级）"
          % (GRAD_TINY[0], worst_good, TOL_TINY_GOOD, GRAD_TINY[-1], worst_bad, TOL_TINY_BAD))
    print("  ★ 口径：`1e-8` 与 `1e-9` 在 fp16 里**恰好是 0**（不缩放的误差 = 1.000e+00）；")
    print("    `1e-7` 还在**次正规**区（不缩放时也有 19%% 的相对误差）。")

    # ── B.2 动态缩放的**恢复时间**与 `S` 的**可用窗口** ─────────────────────
    print()
    print("  ── B.2 动态缩放的恢复时间账（[B] PyTorch 的三个默认值）──")
    n_g, steps = recovery_steps()
    print("  规则（[B] PyTorch `GradScaler`）：溢出 ⇒ `S ×= %g`；连续 %d 步无溢出 ⇒ `S ×= %g`"
          % (specs.PYTORCH_GRADSCALER_BACKOFF, specs.PYTORCH_GRADSCALER_GROWTH_INTERVAL,
             specs.PYTORCH_GRADSCALER_GROWTH))
    print("  一次溢出后 `S` 要乘几次「增长」才回到原值 = %d ⇒ 代价 = **%d 步**" % (n_g, steps))
    check("B.2", n_g == 1 and steps == specs.PYTORCH_GRADSCALER_GROWTH_INTERVAL,
          "★ `backoff = 1/growth = 0.5` ⇒ **一次增长就回到原值**，但要等满 `growth_interval` "
          "= **%d 步** ⇒ 一次溢出让之后 **%d 步**都跑在偏小的 `S` 上（这就是「跳步」之外的**第二笔账**）"
          % (steps, steps))
    grads = make_grads()
    s_min, s_max = scale_window(grads, F16_ZERO_THRESHOLD, specs.F16_MAX_FINITE)
    print("  `S` 的可用窗口（推导值）：下界 %.6g（= 归零阈值 / 最小非零 |g| = %.3e）· 上界 %.6g"
          % (s_min, float(np.abs(grads).min()), s_max))
    check("B.2b", s_min < s_max and s_min <= SCALE <= s_max,
          "★ 窗口非空，且本程序用的 `S = %g` **落在里面**（%.4g ≤ %g ≤ %.4g）"
          % (SCALE, s_min, SCALE, s_max))
    print("  ★ 上界那句的一手依据是 [C] Micikevicius 2017 逐字：")
    print("    `choosing a factor so that its product with the maximum absolute gradient value")
    print("     is below 65,504`；窗口的**下界**那一半是 [A]/推导（不是引文）。")


# ===========================================================================
# 8. C 类：边界与反例 —— 失败行为真的成立
# ===========================================================================
def part_C():
    print()
    print("=" * 74)
    print("C. 边界与反例：不处置溢出会怎样；`S` 太大真的会溢出")
    print("=" * 74)

    # ── C.1 ★ 溢出**不处置**的后果：不可恢复 ────────────────────────────────
    print("  ── C.1 不检测溢出、直接 `w ← w − lr·g`（第 0 步的梯度里有一个 inf）──")
    grads = make_grads()
    used = [float("inf")] + [float(grads[0])] * (S_STEPS - 1)
    w_bad = W_INIT
    trace = []
    for t in range(S_STEPS):
        w_bad = w_bad - LR_STEP * used[t]
        trace.append(w_bad)
    print("  第 0 步之后 w = %r；**从第 1 步起**喂进去的梯度全部 = %.6f（有限）"
          % (trace[0], used[1]))
    print("  第 1 / 10 / %d 步之后 w = %r / %r / %r"
          % (S_STEPS - 1, trace[1], trace[9], trace[-1]))
    check("C.1", all(not math.isfinite(x) for x in trace),
          "★ 权重被 `inf` 污染后**再也不恢复**（%d 步后仍是 %.1f）—— `±inf − 有限 = ±inf`。"
          "⇒ 原文说的 `irreversibly damage the weights after an update` 是**可复现的**"
          % (S_STEPS, trace[-1]))
    check("C.1b", not math.isfinite(trace[0]) and all(math.isfinite(x) for x in used[1:]),
          "★ 阴性对照：**从第 1 步起**喂进去的梯度全部有限（所以污染**只能**来自第 0 步那个 inf）")
    print("  ★ 原文的措辞是 **`One option is to skip the weight update when an overflow is detected`** ——")
    print("    它是一种**选项**；本程序只断言「不处置的后果」是真的，**不断言它是唯一选项**（P20 的同类）。")

    # ── C.2 `S` 的**上界真的存在** ─────────────────────────────────────────
    print()
    print("  ── C.2 `S` 不是越大越好：上界真的存在 ──")
    grads = make_grads()
    mx = float(np.abs(grads).max())
    print("  max|g| = %.6f · 溢出边界 max_finite = %g ⇒ `S` 的上界 = %.2f"
          % (mx, specs.F16_MAX_FINITE, specs.F16_MAX_FINITE / mx))
    for s in (S_OVERFLOW_LO, S_OVERFLOW_HI):
        # ★ `np.errstate` 是必需的：把一个 > 65504 的值 cast 到 fp16 会**故意**产生 `inf`，
        #   而 numpy 会为此打一条 `RuntimeWarning: overflow encountered in cast`。
        #   那正是本判据要的**失败行为**，不是异常 ⇒ 显式关掉这个警告，别让它混进 stderr。
        with np.errstate(over="ignore"):
            h = np.asarray(np.asarray(grads, dtype=np.float64) * s, dtype=np.float16)
        n_ovf = int(np.count_nonzero(~np.isfinite(np.asarray(h, dtype=np.float64))))
        print("    S = 2^%-2d = %-7g ⇒ max|g·S| = %-12.1f ⇒ 溢出个数 = %d"
              % (int(math.log2(s)), s, mx * s, n_ovf))
    with np.errstate(over="ignore"):
        h_lo = np.asarray(np.asarray(grads, dtype=np.float64) * S_OVERFLOW_LO, dtype=np.float16)
        h_hi = np.asarray(np.asarray(grads, dtype=np.float64) * S_OVERFLOW_HI, dtype=np.float16)
    check("C.2", overflow_detected(h_hi) and not overflow_detected(h_lo),
          "★ `S = 2^%d` 时**一个都不溢出**，`S = 2^%d` 时**溢出真的发生** ⇒ "
          "「`S` 有上界」是可证伪的（不是「越大越保险」）"
          % (int(math.log2(S_OVERFLOW_LO)), int(math.log2(S_OVERFLOW_HI))))
    check("C.2b", specs.F16_MAX_FINITE == 65504.0,
          "★ 上界的依据是 [A] fp16 的最大有限数 `%g`（`specs.py` 的常量，逐位钉住）"
          % specs.F16_MAX_FINITE)


# ===========================================================================
# 9. D 类：失败标记自检 + 口径/阈值钉住 + specs 钉住 + 独立性
# ===========================================================================
def preflight():
    """★★ **D 类的前半段先跑**：标记自检 + 口径/阈值钉住（理由见 Chunk 1 的 P23）。

    ★★ **必须放在任何计算之前**：口径常量（`N_GRAD` / `SCALE` / `S_HISTORY_K` / `S_STEPS` …）
      一旦被改，后面的**长度/索引检查**会先崩 —— 那时 `mutate.py` 记的是**崩溃捕获**，
      而「崩溃捕获不算真捕获」。⇒ 钉住提到**任何**计算之前（本库的 **P23**）。
    """
    print("=" * 74)
    print("D0. 口径/阈值钉住（**先跑**：让「改常量」变成断言捕获，而不是崩溃捕获）")
    print("=" * 74)
    print("  本程序的失败标记 ASSERT_FAIL_MARK = %r" % ASSERT_FAIL_MARK)
    print("  ★ 本程序的数字**全部是推导值**：它们是把 **PyTorch / Apex 的文档规则**"
          "在 numpy 上重演出来的，")
    print("    **不是**「PyTorch 在本机跑出来的」（本程序一行都不 import 它）。"
          "`python3 %s` + `numpy %s`。"
          % (sys.version.split()[0], np.__version__))
    before = len(FAILS)

    hits = _mark_assignments_in(os.path.abspath(__file__))
    uniq = {getattr(v, "value", None) for v in hits}
    check("D.1", len(hits) == 1 and len(uniq) == 1
          and uniq == {ASSERT_FAIL_MARK} and "FAIL" in ASSERT_FAIL_MARK,
          "★ 标记自检：模块级赋值恰好 1 处、取值唯一、== %r、且含 `FAIL`（AST 读出）"
          % ASSERT_FAIL_MARK)
    check("D.2", ASSERT_FAIL_MARK == ":: FAIL: mixed-precision" and "FAIL" in ASSERT_FAIL_MARK,
          "★ 标记自检：**等于**本主题被分配的字面量、且含 `FAIL`")
    check("D.2b", OTHER_THEME_MARKS == (".. FAIL: autograd", "++ FAIL: optimizers",
                                        "// FAIL: init-stability", ";; FAIL: checkpointing"),
          "★ 「不跨主题照抄」的**对照物**也逐位钉住")

    measured = {"SEED": SEED, "N_GRAD": N_GRAD, "SCALE": SCALE,
                "SCALE_NOT_POW2": SCALE_NOT_POW2, "MAX_NORM": MAX_NORM,
                "GRAD_SIGMA": GRAD_SIGMA, "GRAD_TINY": GRAD_TINY, "W_INIT": W_INIT,
                "LR_STEP": LR_STEP, "S_HISTORY_K": S_HISTORY_K, "S_STEPS": S_STEPS,
                "S_OVERFLOW_LO": S_OVERFLOW_LO, "S_OVERFLOW_HI": S_OVERFLOW_HI,
                "F16_ZERO_THRESHOLD": F16_ZERO_THRESHOLD}
    expect = {"SEED": 20260926, "N_GRAD": 4096, "SCALE": 1024.0,
              "SCALE_NOT_POW2": 1000.0, "MAX_NORM": 1.0, "GRAD_SIGMA": 2.0,
              "GRAD_TINY": (1e-7, 1e-8, 1e-9), "W_INIT": 1.0, "LR_STEP": 0.01,
              "S_HISTORY_K": 8, "S_STEPS": 40,
              "S_OVERFLOW_LO": 1024.0, "S_OVERFLOW_HI": 8192.0,
              "F16_ZERO_THRESHOLD": 2.9802322387695312e-08}
    diff = {k: (expect[k], v) for k, v in measured.items() if v != expect[k]}
    check("D.3", not diff, "★ 口径常量 14 个逐位一致（结论的**测量条件**）%s"
          % ("" if not diff else " ← 差异 %s" % diff))
    check("D.3b", math.log2(SCALE) == math.floor(math.log2(SCALE))
          and math.log2(S_OVERFLOW_LO) == math.floor(math.log2(S_OVERFLOW_LO))
          and math.log2(S_OVERFLOW_HI) == math.floor(math.log2(S_OVERFLOW_HI)),
          "★ **三个缩放因子都是 2 的幂**这件事单独钉住（A.1 的「逐位恒等」全靠它）")
    check("D.3d", F16_ZERO_THRESHOLD == specs.F16_MIN_SUBNORMAL / 2.0,
          "★ 归零阈值 = `specs.F16_MIN_SUBNORMAL / 2`（%.17g）—— 与 `01-type-selection` 的"
          "`thr_over_two()` **同值**；★ 取**一半**的理由见 `01` 的 `A.5b`（用整值会让下溢比例算成 2 倍）"
          % F16_ZERO_THRESHOLD)
    check("D.3c", math.log2(SCALE_NOT_POW2) != math.floor(math.log2(SCALE_NOT_POW2)),
          "★ 反面对照的那个 `S` **不是** 2 的幂（否则 A.1b 会退化成恒真）")

    th = {"TOL_UNSCALE_RATIO": TOL_UNSCALE_RATIO, "TOL_TINY_GOOD": TOL_TINY_GOOD,
          "TOL_TINY_BAD": TOL_TINY_BAD, "MIN_REGION_FACTOR": MIN_REGION_FACTOR,
          "TOL_EXACT": TOL_EXACT, "TOL_POW2_BITWISE": TOL_POW2_BITWISE}
    th_expect = {"TOL_UNSCALE_RATIO": 1e-6, "TOL_TINY_GOOD": 1e-5, "TOL_TINY_BAD": 1e-3,
                 "MIN_REGION_FACTOR": 1.0, "TOL_EXACT": 0.0, "TOL_POW2_BITWISE": 0}
    tdiff = {k: (th_expect[k], v) for k, v in th.items() if v != th_expect[k]}
    check("D.4", not tdiff, "★ 阈值 6 个逐位一致（含**两个 0 容差**：A.1 的逐位恒等与 B.1 的"
                            "「不缩放的误差 = 1」）%s" % ("" if not tdiff else " ← 差异 %s" % tdiff))
    return len(FAILS) == before


def part_D():
    print()
    print("=" * 74)
    print("D. specs 常量逐位钉住 与 独立性证据")
    print("=" * 74)
    print("  ★★ 这一节是 **P19 的现场**：下面每一条都写明它是**哪个实现**的文档（[B]）。")

    # ── ★★ `O0/O1/O2/O3`：来源是 **Apex**，不是 NVIDIA 的 AMP 页 ──────────────
    check("D.5", specs.APEX_OPT_LEVELS == ("O0", "O1", "O2", "O3"),
          "★ 四个 opt_level 的名字逐位一致（来源 = **[B] Apex** 的 AMP 文档 —— "
          "★ **不是** NVIDIA 的 AMP 页：那页 `whitelist` / `master_weights` 各 **0** 次）")
    check("D.6", specs.MICIKEVICIUS_UNSCALE_QUOTE
          == "It is simplest to perform this unscaling right after the backward pass but before gradient clipping",
          "★ 「unscale 的时机」的逐字依据**逐位**一致（A.2 判的就是它）—— "
          "★ **用 `==`**：它的变异形态是末尾追加字符（计划 §3 开头盲区③）")

    # ── ★★ O1 的 `master_weights` 是 `None`，**不是** `True`（Y10）──────────────
    print("  ★★ O1 vs O2 的 `master_weights`（[B] Apex）：")
    print("     O1: `master_weights=None (not applicable, model weights remain FP32)`")
    print("     O2: `master_weights=True`")
    check("D.7", (specs.APEX_O1_MASTER_WEIGHTS, specs.APEX_O2_MASTER_WEIGHTS) == (None, True),
          "★ 「**O1 会额外维护一份 fp32 主权重**」**是错的**：O1 下权重本来就是 fp32 "
          "⇒ 它的 `master_weights` 逐字是 `None (not applicable, model weights remain FP32)`；"
          "**只有 O2 是 `True`**")
    check("D.7b", specs.APEX_O1_MASTER_WEIGHTS is None,
          "★ 用 `is None` 而**不是** `== False`：原文说的是「**不适用**」，"
          "写成 `False` 就与「不支持」混了（`None` / `False` 在变异里也分得开）")
    # ★★ 上面那条断言用的值是 `None`，而 `mutate.py` 的 `_mutate_value(None)` 给不出「另一个值」
    #   （计入「跳过」）⇒ **只靠它，Y10 在变异测试里是免检的**。
    #   这一条把 Y10 的**逐字证据**也钉住：它的变异形态是末尾追加字符 ⇒ 会真的被变异到。
    check("D.7c", specs.APEX_O1_MASTER_WEIGHTS_QUOTE
          == "master_weights=None (not applicable, model weights remain FP32)",
          "★ Y10 的**逐字证据**逐位钉住（%r）—— 那个 `None` 是**这条引文读出来的值**，"
          "两条一起才算把「O1 不额外维护 fp32 主权重」这句话锁住"
          % specs.APEX_O1_MASTER_WEIGHTS_QUOTE)

    # ── ★★ loss scale 的两个官方初值：**不得合成一个**（Y11）────────────────
    print("  ★★ loss scale 的初值**有两个官方数**（都是 [B]，来源不同）：")
    print("     NVIDIA 的 AMP FAQ : `beginning with a high loss scale value (say, 2^24)`（★ 量词是 `say`）")
    print("     PyTorch GradScaler: `init_scale=65536.0`")
    check("D.8", specs.NVIDIA_FAQ_INIT_SCALE == 16777216.0
          and specs.PYTORCH_GRADSCALER_INIT_SCALE == 65536.0
          and specs.NVIDIA_FAQ_INIT_SCALE / specs.PYTORCH_GRADSCALER_INIT_SCALE == 256.0,
          "★ 两个数逐位钉住、比值 **精确 256**（2^24 vs 2^16）⇒ **不得合成一个**："
          "「初值取 X」这种写法会丢掉「有分歧」这个事实")
    check("D.8b", specs.APEX_LOSS_SCALE_MAX == 16777216.0
          and specs.APEX_LOSS_SCALE_MAX == specs.NVIDIA_FAQ_INIT_SCALE,
          "★ 而 Apex 的 `initialize` 里 `16777216.0` 是 **`max_loss_scale`（上界）** —— "
          "与 NVIDIA FAQ 那个**初值**数值相同、**角色不同**：同一个数不要因为相等就合并")
    check("D.9", specs.PYTORCH_GRADSCALER_GROWTH == 2.0
          and specs.PYTORCH_GRADSCALER_BACKOFF == 0.5
          and specs.PYTORCH_GRADSCALER_GROWTH_INTERVAL == 2000,
          "★ 动态缩放的三个规则参数逐位一致（B.2 的「%d 步恢复」全靠它们）"
          % specs.PYTORCH_GRADSCALER_GROWTH_INTERVAL)
    check("D.10", specs.MICIKEVICIUS_OVERFLOW_SKIP_QUOTE
          == "One option is to skip the weight update when an overflow is detected",
          "★ 溢出处置的逐字依据逐位一致 —— ★ 原文的措辞是 **`One option is`**（一种选项），"
          "不是「必须」（P20 的同类：不要把选项升格成规定）")
    check("D.11", specs.MICIKEVICIUS_SCALE_FACTORS_TRIED == (8, 32768),
          "★ 论文试过的倍率范围逐位一致（%r —— ★ `32K` 按该文用法是 `32000`，"
          "本常量取 **2 的幂口径 32768** 并在正文里写明这个换算）"
          % (specs.MICIKEVICIUS_SCALE_FACTORS_TRIED,))
    check("D.12", specs.F16_MAX_FINITE == 65504.0 and specs.F16_MIN_NORMAL == 6.103515625e-05
          and specs.TORCH_CLIP_COEF_DENOM_EPS == 1e-6,
          "★ 本程序用到的两个 fp16 边界常量与**裁剪分母那个 `1e-6`** 逐位一致"
          "（★ 后者是 [B] PyTorch v2.9.0 的源码口径，本主题**只消费、不重定义**："
          "判据与阈值在 `autograd-and-backward/03` 写一次）")
    check("D.13", specs.UNSOURCED == ("bf16_dynamic_range_exact_equality_claim",
                                      "torch_e4m3fn_overflow_rule_version_span",
                                      "fp8_scaling_growth_rule_concrete_values"),
          "★ `specs.UNSOURCED` 的三项**逐位钉住**（%d 项）" % len(specs.UNSOURCED))

    # ── 独立性：参照路径不调用被测路径 ──────────────────────────────────────
    check("D.14", inspect.getsource(amp_step) != inspect.getsource(plain_step),
          "★ 防 P17：被测（`amp_step`）与参照（`plain_step`）的**源码不同**"
          "（%d vs %d 字节）" % (len(inspect.getsource(amp_step)), len(inspect.getsource(plain_step))))
    watched = {"amp_step"}
    called = _callee_names(plain_step)
    check("D.15", not (watched & called),
          "★ **AST 调用表**：`plain_step` 的 `Call` 节点 = %s —— 没有被测函数"
          % (", ".join(sorted(called)) if called else "无"))
    check("D.16", "np.asarray(h, dtype=np.float64) / scale" in inspect.getsource(scale_then_unscale)
          or "dtype=np.float64" in inspect.getsource(scale_then_unscale),
          "★ ★ **反 NEP 50 的显式转换在源码里**（不写它，除法会退回 fp16 做、小值静默变 0）——"
          "本程序第一版正是这么错的（见 `A.1b` 的说明与 `probe` 那一类教训）")
    check("D.17", MARKS_USED <= {ASSERT_FAIL_MARK},
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
    print("02_amp.py —— AMP 的执行约定：loss scaling 的算术、unscale 的时机、溢出的处置")
    print("=" * 74)
    print("  ★ 真值标注：**本程序一行都不 import PyTorch / Apex**。它把那些实现**文档里")
    print("    逐字写的规则**（`specs.py` 的 [B] 常量）在 numpy 上重演一遍 ⇒ 全部是**推导值**。")
    print("    本机 `python3 %s` + `numpy %s`，macOS arm64 **无 GPU**。"
          % (sys.version.split()[0], np.__version__))
    print("  ★ 失败标记 ASSERT_FAIL_MARK = %r（本主题自定义；不跨主题照抄）" % ASSERT_FAIL_MARK)
    print("  ★ 禁区（一律回链，本程序不写）：「为什么 fp16 要 loss scaling」整块在")
    print("    `10-foundations/floating-point/04-low-precision-formats.md`；")
    print("    「loss scaling 的代价」（上溢侧、次正规位不回来）在同上 `:197,229-231`；")
    print("    `autocast` 的算子白名单/黑名单属**框架 API 语义**，在 `70-frameworks/`；")
    print("    裁剪的**阈值定义**在 `90-training/autograd-and-backward/03-gradient-lifecycle.md`。")
    if not preflight():
        print()
        print("!! 口径/标记钉住失败 ⇒ **不继续往下跑**：后续的 A/B/C 只会产生一堆无意义的连带崩溃")
        return summary()
    grads = make_grads()
    part_A(grads)
    part_B()
    part_C()
    part_D()
    return summary()


if __name__ == "__main__":
    sys.exit(main())

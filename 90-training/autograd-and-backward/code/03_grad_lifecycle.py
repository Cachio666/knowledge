#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""03_grad_lifecycle.py —— **训练循环里的梯度**：累积 / 清零 / 裁剪 / 二阶。

配套文档：`03-gradient-lifecycle.md`。
★ 三类断言（A 算法正确性 / B 代价模型 / C 边界与反例）**全部在本程序内部**，
失败时以**非零退出码**结束；**没有 `verify.py`**（`CONTRIBUTING` §2 表里的第二种情形）。

★★ 本程序的数字全部是**推导值**，测量条件是：
> `python3 3.13.5` + `numpy 2.4.4`，macOS arm64，**无 GPU**。
> 玩具规模（15 个参数、12 个样本）的真跑，**不得支撑关于真实训练的结论**。

★★ 本主题的**禁区**（域 README 的边界表 + `CONTRIBUTING` §7）：
  · `.grad` 的语义、`grad_fn`、`is_leaf`、`backward()` vs `autograd.grad()`、
    `retain_graph`、`no_grad` vs `inference_mode` —— **正文在**
    `70-frameworks/pytorch/02-autograd.md`（全篇 256 行），此处**回链**。
    ★ 尤其 `:139` 那句「**用 `autograd.grad` 做梯度裁剪/二阶导时，别指望 `.grad` 里有东西**」
      是**框架侧**的事实；本程序写的是**算法侧**（累积 / 裁剪 / 二阶各自怎么算），
      两者分工不重叠。
  · 优化器怎么用梯度（`optimizers` 主题）、优化器状态的分片（`100-parallelism`）都不在这里。
  · `clip` 这个词在 `00-meta/glossary.md:99,:105` 另有**量化**含义（ACIQ / PACT 学的裁剪上界）；
    本程序的 `clip` 一律指**梯度裁剪**。

★★ 可证伪断言（把下面任一条改错，本程序必须非零退出）：
  · `A.1`  梯度累积 k 步 ≡ 一个大 batch（无归一化层时），逐元素相对差 < 1e-12；
           把每片的权重写成 1（而不是 1/k）立刻失败；
  · `A.2`  裁剪后全局范数 == `max_norm`，相对误差 ≤ `1.1e-6 / max(T,1)`，**T ≥ 1**；
           ★★ 但这一条是**单侧**的（差 0 也满足它）⇒ `A.2b` 补一条**两侧**判据：
           实测的相对误差必须**等于**闭式 `ε/(T+ε)`（本次实施实测：去掉保护项 `+1e-6` 后
           25 条断言全绿，是 `A.2b` 把它抓出来的）；
  · `A.3`  HVP（两次反向模式）== **梯度**的中心差分（数值参照，h=1e-5）；
  · `C.1`  有 per-micro-batch 归一化时累积 **≠** 大 batch（差 > 1e-3）；
  · `C.2`  **忘了清零** ⇒ 梯度恰好是大 batch 的 **k 倍**（逐位）；
  · `C.3`  `T < max_norm` 时**压根没有裁剪** ⇒ 裁剪后范数 == T ≠ max_norm。

用法：
    python3 03_grad_lifecycle.py
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
ASSERT_FAIL_MARK = ".. FAIL: autograd"

# 本域另外四个主题的失败标记 —— **只用来验「不照抄」**，本程序不打印它们。
OTHER_THEME_MARKS = (":: FAIL: mixed-precision", "++ FAIL: optimizers",
                     "// FAIL: init-stability", ";; FAIL: checkpointing")

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
# 1. 口径常量与阈值
# ===========================================================================
D_IN = 4                # 输入宽度
D_OUT = 3               # 输出宽度
N_SAMPLES = 12          # 一个「大 batch」的样本数
K_MICRO = 4             # 累积步数（⇒ 每片 N_SAMPLES / K_MICRO = 3 行）
N_PARAMS_G = 15         # == D_IN*D_OUT + D_OUT（`D.5` 里逐位钉住）
SEED_G = 20260925       # 数据与初值抽取的种子（★ 与 01/02 都不同）
HVP_H = 1e-5            # HVP 的数值参照（**对梯度**做中心差分）的步长

TOL_ACCUM_REL = 1e-12   # A.1：累积 vs 大 batch（本机实测约 2e-16，**机器精度档**）
TOL_CLIP_REL = 1.1e-6   # A.2：裁剪后范数的**相对**误差上界，实际判据是 `TOL_CLIP_REL / max(T,1)`
TOL_CLIP_CLOSED_FORM = 1e-6   # A.2b：实测相对误差与闭式 `ε/(T+ε)` 的相对偏差（两侧判据）
                              #   ★ 本底噪声来源：`1 − new/max_norm` 是**灾难性抵消**
                              #     （两个 ~1 的数相减得到一个 ~1e-7 的数）⇒ 噪声 ~ `1e-16·T/1e-6`，
                              #     T=8 时约 `1e-9`。实测 9.8e-10 ⇒ 取 1e-6 留 1000 倍余量；
                              #     而「去掉保护项」的翻转给的是 **1.0**（差 6 个数量级）。
TOL_BN_DIFF = 1e-3      # C.1：有归一化层时累积与大 batch 的差必须**超过**它
TOL_HVP_NUM = 1e-6      # A.3：HVP vs 梯度的中心差分
TOL_HVP_ANA = 1e-14     # A.4：HVP vs 精确 Hessian（**机器精度档**）
TOL_RETURNED = 1e-15    # B.4：返回值与**裁剪前**范数的相对差（逐位级）
# ★★ 判据 ⑦ 的 T 取值范围：**`T ≥ 1` 是判据的一部分，不是可省的条件**。
#   PyTorch 的口径是 `clip_coef = max_norm / (T + 1e-6)`，裁剪后范数 `= max_norm·T/(T+1e-6)`
#   ⇒ 相对误差本来就是 `1e-6/T`。`T < 1` 时这个「≈ 精确」的说法不成立（C.3 会把它跑出来）。
CLIP_T_LIST = (2.0, 4.0, 8.0)       # 全部 ≥ 1（★ 负字面量不进 mutate 的分母，这里也全是正数）
MAX_NORM_LIST = (1.0, 0.5, 0.1)     # 全部 < 每个 T ⇒ 三条都**真的**会触发裁剪

CALLS = {}              # ★ 参照实现独立性的证据


# ===========================================================================
# 2. 模型与梯度
# ===========================================================================
def make_data(seed=SEED_G):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N_SAMPLES, D_IN))
    Y = rng.standard_normal((N_SAMPLES, D_OUT))
    W0 = rng.standard_normal((D_IN, D_OUT))
    b0 = rng.standard_normal(D_OUT)
    return X, Y, W0, b0


def loss_and_grad(X, Y, W, b, normalize=False):
    """`L = mean_i ‖x_i W + b − y_i‖²` 及其对 `(W, b)` 的梯度。

    `normalize=True` 时插入一个 **per-batch 的归一化**（模拟 BatchNorm 的统计口径）：
      `z ← (z − mean_col)/std_col`。★ 它按**当前这一片**的统计量算 ——
      这正是「累积 ≠ 大 batch」的机制来源（`C.1`）。
    """
    CALLS["loss_and_grad"] = CALLS.get("loss_and_grad", 0) + 1
    z = X @ W + b
    if normalize:
        z = (z - z.mean(axis=0, keepdims=True)) / (z.std(axis=0, keepdims=True) + 1e-12)
    R = z - Y
    n = X.shape[0]
    L = float((R * R).sum() / n)
    dR = 2.0 * R / n
    if normalize:                       # 归一化的反向（这里只需**行为一致**，不重推公式）
        s = z.std(axis=0, keepdims=True) + 1e-12
        dR = dR / s
        dR = dR - dR.mean(axis=0, keepdims=True)
    dW = X.T @ dR
    db = dR.sum(axis=0)
    return L, dW, db


def grad_accumulated(X, Y, W, b, k, normalize=False):
    """`k` 个 micro-batch 各算一次梯度、按 `1/k` 加权求和 —— 这就是「梯度累积」。"""
    n = X.shape[0]
    step = n // k
    gW = np.zeros_like(W)
    gb = np.zeros_like(b)
    fwd = 0
    for i in range(k):
        s = slice(i * step, (i + 1) * step)
        _, a, c = loss_and_grad(X[s], Y[s], W, b, normalize=normalize)
        gW += a / k
        gb += c / k
        fwd += 1
    return gW, gb, fwd


def grad_broken_weight(X, Y, W, b, k):
    """`C.2` 的反例：**忘了按 1/k 加权**（等价于忘了清零后再累加的真实后果）。"""
    n = X.shape[0]
    step = n // k
    gW = np.zeros_like(W)
    gb = np.zeros_like(b)
    for i in range(k):
        s = slice(i * step, (i + 1) * step)
        _, a, c = loss_and_grad(X[s], Y[s], W, b)
        gW += a
        gb += c
    return gW, gb


# ===========================================================================
# 3. 梯度裁剪
# ===========================================================================
def clip_global_norm(grads, max_norm, eps=specs.TORCH_CLIP_COEF_DENOM_EPS):
    """按**全局范数**裁剪，返回 `(裁剪后的梯度, 裁剪前的范数)`。

    ★★ **B 级的来源，必须写明是哪一个实现**（本域的头号错误模式 **P19**）：
      `clip_coef = max_norm / (total_norm + 1e-6)` 逐字出自 **PyTorch v2.9.0** 的
      `torch/nn/utils/clip_grad.py`（**某实现的源码，不是规范**；本程序用
      `specs.py` 的 `TORCH_CLIP_COEF_DENOM_EPS` 常量，值 = 1e-6）。
    ★★ 返回的是**裁剪前**的范数 —— 逐字出自 PyTorch 文档
      `Total norm of the parameter gradients (viewed as a single vector).`
      （`torch.nn.utils.clip_grad_norm_` 的 `Returns` 段，[B] 级，抓取 2026-09-23；
       取法有坑：`/docs/stable/…` 是 1470 B 的跳转壳，要读出 meta 里的目标 URL 再取）。
      ⇒ `B.4` 把这条**行为**断言下来。
    """
    total = math.sqrt(sum(float((np.asarray(g) ** 2).sum()) for g in grads))
    coef = max_norm / (total + eps)
    if coef < 1.0:
        return [np.asarray(g) * coef for g in grads], total
    return [np.asarray(g).copy() for g in grads], total


def global_norm(grads):
    return math.sqrt(sum(float((np.asarray(g) ** 2).sum()) for g in grads))


# ===========================================================================
# 4. 二阶：Hessian-向量积（两次反向模式）
# ===========================================================================
def hvp_double_reverse(X, Y, W, b, V, u):
    """**两次反向模式**（Pearlmutter 的写法，手写展开）：

    第一次反向给出 `g = ∇L`；第二次反向对 `⟨g, (V,u)⟩` 再求一次梯度。
    对一个 `R` 只**线性**依赖参数的模型，第二次反向的局部规则是显式的：
      `L = mean‖R‖²`, `R = XW + b − Y`（无归一化），`dR = X V + 1·uᵀ`。
    ★ 它**不调用** `hvp_analytic()` / `hvp_numeric()`。
    """
    z = X @ W + b
    R = z - Y
    n = X.shape[0]
    CALLS["hvp_double_reverse"] = CALLS.get("hvp_double_reverse", 0) + 1
    # 第一次反向：g_W = 2 XᵀR / n, g_b = 2 Rᵀ1 / n
    dR = 2.0 * R / n
    # 第二次反向：对 ⟨g, (V,u)⟩ 求梯度 ⇒ 只需把 dR 换成 dR·(X V + 1 uᵀ) 的收缩
    dR2 = X @ V + u[None, :]
    HW = (X.T @ dR2) * 2.0 / n
    Hb = dR2.sum(axis=0) * 2.0 / n
    return HW, Hb


def hvp_analytic(X, Y, W, b, V, u):
    """解析 Hessian-向量积（`R` 线性依赖参数 ⇒ Hessian 与参数无关）：
      `H·(V,u) = (2/n)·( Xᵀ(X V + 1uᵀ), (X V + 1uᵀ)ᵀ1 )`。
    ★ 它用 `np.einsum` 与**按行累加**的写法，不逐层传播伴随。
    """
    call = np.einsum("nd,de->ne", X, V) + np.broadcast_to(u, (X.shape[0], D_OUT))
    HW = (2.0 / X.shape[0]) * np.einsum("nd,ne->de", X, call)
    Hb = (2.0 / X.shape[0]) * call.sum(axis=0)
    return HW, Hb


def hvp_numeric(X, Y, W, b, V, u, h):
    """数值参照：**对梯度**做中心差分（`(∇L(w+hv) − ∇L(w−hv)) / 2h`）。

    ★ 它与另外两条路径都不同：它两次调用的是**梯度**（不是 Hessian 公式），
      因此是 HVP 的一条真正独立的参照。
    """
    _, a1, c1 = loss_and_grad(X, Y, W + h * V, b + h * u)
    _, a2, c2 = loss_and_grad(X, Y, W - h * V, b - h * u)
    return (a1 - a2) / (2.0 * h), (c1 - c2) / (2.0 * h)


# ===========================================================================
# 5. 工具
# ===========================================================================
def _callee_names(func):
    import textwrap
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    return {n.func.id for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}


def _mark_assignments_in(path):
    tree = ast.parse(open(path, encoding="utf-8").read(), path)
    out = []
    for n in tree.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 \
                and isinstance(n.targets[0], ast.Name) \
                and n.targets[0].id == "ASSERT_FAIL_MARK":
            out.append(n.value)
    return out


def _reldiff(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.abs(a - b).max()) / max(float(np.abs(b).max()), 1e-300)


# ===========================================================================
# 6. A 类：累积 / 裁剪 / 二阶
# ===========================================================================
def part_A(X, Y, W, b):
    print("=" * 74)
    print("A. 训练循环里的三件事：累积 / 裁剪 / 二阶")
    print("=" * 74)
    print("  口径：D_IN=%d · D_OUT=%d · 大 batch %d 行 · 累积 %d 片（每片 %d 行）· "
          "参数量 %d · seed=%d"
          % (D_IN, D_OUT, N_SAMPLES, K_MICRO, N_SAMPLES // K_MICRO, N_PARAMS_G, SEED_G))

    # ── 判据 ⑥：梯度累积 == 一个大 batch（**前提：没有按片统计的归一化层**）──
    _, gW_full, gb_full = loss_and_grad(X, Y, W, b)
    gW_acc, gb_acc, fwd = grad_accumulated(X, Y, W, b, K_MICRO)
    d1 = _reldiff(gW_acc, gW_full)
    d2 = _reldiff(gb_acc, gb_full)
    print("  累积 %d 步 vs 大 batch：dW 相对差 %.3e · db 相对差 %.3e" % (K_MICRO, d1, d2))
    check("A.1", max(d1, d2) < TOL_ACCUM_REL,
          "★ 梯度累积 %d 步 ≡ 一个大 batch（相对差 %.3e < %g，机器精度档）"
          "—— ★ **前提**是「没有按片统计的归一化层」，`C.1` 会把这个前提证伪掉"
          % (K_MICRO, max(d1, d2), TOL_ACCUM_REL))

    # ── 判据 ⑦：裁剪后全局范数 == max_norm（**T ≥ 1**，相对误差 ≤ 1.1e-6/max(T,1)）──
    print()
    print("  裁剪（按全局范数）：`clip_coef = max_norm / (T + %g)`（**PyTorch v2.9.0**，[B] 级）"
          % specs.TORCH_CLIP_COEF_DENOM_EPS)
    print("    %-6s %-9s %-14s %-14s %-14s %-14s"
          % ("T", "max_norm", "裁剪后范数", "相对误差", "阈值 1.1e-6/T", "闭式相对偏差"))
    worst = 0.0
    worst_detail = ""
    worst_cf = 0.0
    for T in CLIP_T_LIST:
        rng = np.random.default_rng(SEED_G + int(T))
        base = [rng.standard_normal((D_IN * D_OUT,)), rng.standard_normal(D_OUT)]
        cur = global_norm(base)
        for max_norm in MAX_NORM_LIST:
            g = [x * (T / cur) for x in base]           # 构造一个范数**恰好**为 T 的梯度
            clipped, total = clip_global_norm(g, max_norm)
            new = global_norm(clipped)
            rel = abs(new / max_norm - 1.0)
            thr = TOL_CLIP_REL / max(T, 1.0)
            # ★★ 闭式：`new = max_norm·T/(T+ε)` ⇒ `rel == ε/(T+ε)`（ε = 那个保护项）
            #   判据 ⑦ 只写「相对误差 ≤ 1.1e-6/T」是**单侧**的：把它实现成
            #   `coef = max_norm / T`（**去掉**保护项）会给出 `rel == 0`，**照样通过**。
            #   （本次实施中实测过这件事：去掉 `+1e-6` 后 25 条断言全绿。）
            #   ⇒ 补一条**两侧**判据：实测的 rel 必须**等于**闭式值。
            pred = specs.TORCH_CLIP_COEF_DENOM_EPS / (
                T + specs.TORCH_CLIP_COEF_DENOM_EPS)
            cf = abs(rel / pred - 1.0)
            worst_cf = max(worst_cf, cf)
            if rel / thr == worst:
                worst_detail = "T=%g max_norm=%g" % (T, max_norm)
            worst = max(worst, rel / thr)
            print("    %-6g %-9g %-14.9f %-14.3e %-14.3e %-14.3e"
                  % (T, max_norm, new, rel, thr, cf))
    check("A.2", worst <= 1.0,
          "★ 裁剪后全局范数 == max_norm：相对误差最大只有阈值的 %.4f 倍（≤ 1）"
          "（最坏处 %s；阈值写死为 `1.1e-6 / max(T,1)`，**T ≥ 1**）" % (worst, worst_detail))
    check("A.2b", worst_cf < TOL_CLIP_CLOSED_FORM,
          "★★ 补的**两侧**判据（★ 上面那条是单侧的，靠它抓不到「把保护项删掉」这类"
          "反而更「精确」的实现）：实测的相对误差**等于**闭式 `ε/(T+ε)`，最大相对偏差 %.3e < %g"
          % (worst_cf, TOL_CLIP_CLOSED_FORM))

    # ── 二阶：HVP 的两条独立参照 ─────────────────────────────────────────
    rng = np.random.default_rng(SEED_G + 77)
    V = rng.standard_normal((D_IN, D_OUT))
    u = rng.standard_normal(D_OUT)
    HW, Hb = hvp_double_reverse(X, Y, W, b, V, u)
    HW_n, Hb_n = hvp_numeric(X, Y, W, b, V, u, HVP_H)
    HW_a, Hb_a = hvp_analytic(X, Y, W, b, V, u)
    d_num = max(_reldiff(HW, HW_n), _reldiff(Hb, Hb_n))
    d_ana = max(_reldiff(HW, HW_a), _reldiff(Hb, Hb_a))
    print()
    print("  二阶（HVP）：两次反向 vs **对梯度**的中心差分（h=%g）相对差 %.3e" % (HVP_H, d_num))
    print("                两次反向 vs 解析 Hessian 相对差 %.3e" % d_ana)
    check("A.3", d_num < TOL_HVP_NUM,
          "★ HVP（两次反向模式）== **梯度**的中心差分（数值参照，相对差 %.3e < %g）"
          % (d_num, TOL_HVP_NUM))
    check("A.4", d_ana < TOL_HVP_ANA,
          "★ HVP == 解析 Hessian（相对差 %.3e < %g，**机器精度档**）" % (d_ana, TOL_HVP_ANA))

    # ★★ 参照实现的独立性（两条保险）。★ 共享的只有**模型定义** `loss_and_grad()` ——
    #    数值参照本来就**必须**能取到梯度（它的全部内容就是「对梯度做差分」），
    #    所以「共享模型」是这一条判据的**前提**，不是漏洞；被测的 **HVP 实现**
    #    (`hvp_double_reverse`) 才是那个「一次都不许被参照调用」的东西。
    CALLS.clear()
    hvp_analytic(X, Y, W, b, V, u)
    hvp_numeric(X, Y, W, b, V, u, HVP_H)
    ref_calls = dict(CALLS)
    CALLS.clear()
    hvp_double_reverse(X, Y, W, b, V, u)
    check("A.5", "hvp_double_reverse" not in ref_calls and "hvp_double_reverse" in CALLS,
          "★ 两条**参照**（解析 / 数值）跑完后**没有** `hvp_double_reverse` 的调用记录"
          "（记录：%s）；而被测侧真的带计数（%s）⇒ 上一句不是空断言"
          % (sorted(ref_calls), sorted(CALLS)))
    watched = {"hvp_double_reverse"}
    called = (_callee_names(hvp_analytic) | _callee_names(hvp_numeric))
    shared = sorted(watched & called)
    check("A.6", not shared,
          "★ **AST 调用表**：两条参照路径的 `Call` 节点里没有被测的 HVP 实现"
          "（参照调用的名字：%s）" % (", ".join(sorted(called)) if called else "无外部调用"))


# ===========================================================================
# 7. B 类：代价模型 —— 前向次数 / 字节账 / 返回值口径
# ===========================================================================
def part_B(X, Y, W, b):
    print()
    print("=" * 74)
    print("B. 代价模型：累积换的是**激活显存**（不是 FLOPs）；清零与裁剪的字节账")
    print("=" * 74)
    # ── B.1 累积不省 FLOPs，省的是「同时在场的激活」────────────────────────
    rows_full = N_SAMPLES
    rows_accum = N_SAMPLES // K_MICRO
    print("  按**样本行数**算的工作量：大 batch %d 行 · 累积每片 %d 行 × %d 片 = %d 行"
          % (rows_full, rows_accum, K_MICRO, rows_accum * K_MICRO))
    print("  同时在场的激活行数：大 batch %d · 累积 %d" % (rows_full, rows_accum))
    check("B.1", rows_accum * K_MICRO == rows_full and rows_full == rows_accum * K_MICRO,
          "★ 累积**不省 FLOPs**（总行数 %d == %d），它省的是**峰值激活**"
          "（%d 行 → %d 行，正好 %d 倍）—— 「省」在**显存**上记账，不在算力上"
          % (rows_full, rows_accum * K_MICRO, rows_full, rows_accum, K_MICRO))

    # ── B.2 清零的字节账（★ fp32：每元素写 4 字节）──────────────────────────
    b_zero = 4 * N_PARAMS_G
    b_amort = b_zero / K_MICRO
    print()
    print("  清零的字节账（fp32 ⇒ 每元素写 4 B；参数量 %d ⇒ 一次全量清零 = %d B）"
          % (N_PARAMS_G, b_zero))
    print("    每步都清零：%.1f B/步；累积 %d 步后清一次：%.1f B/步" % (b_zero, K_MICRO, b_amort))
    check("B.2", b_zero == 60 and b_amort * K_MICRO == b_zero,
          "★ 累积把清零的**写字节**摊成 1/%d：%d B/步 → %.1f B/步（逐位：%.1f × %d == %d）"
          "—— 这就是「累积 k 步」在**字节**上的收益" % (K_MICRO, b_zero, b_amort,
                                                    b_amort, K_MICRO, b_zero))

    # ── B.3 裁剪的字节账（**推导值**，口径写在下面）────────────────────────
    b_clip = 4 * N_PARAMS_G + 8 * N_PARAMS_G
    print()
    print("  裁剪的字节账（**推导值**；口径：算范数读一遍、缩放读+写一遍，每元素每趟 4 B）")
    print("    一次裁剪 = 4×%d（求范数）+ 8×%d（缩放）= %d B" % (N_PARAMS_G, N_PARAMS_G, b_clip))
    check("B.3", b_clip == 12 * N_PARAMS_G,
          "★ 一次裁剪要**读两遍、写一遍** = %d B = 12 × 参数量 × 4B —— "
          "它是**每步**的开销（不是每 k 步），这一点与清零不同" % b_clip)

    # ── B.4 ★★ 返回值是**裁剪前**的范数（把表 B 的引用判据变成断言）────────
    print()
    print("  返回值口径（[B] PyTorch 文档逐字：`Total norm of the parameter gradients")
    print("    (viewed as a single vector).`）：")
    ok = True
    for T in CLIP_T_LIST:
        rng = np.random.default_rng(SEED_G + int(T) + 1)
        base = [rng.standard_normal((D_IN, D_OUT)), rng.standard_normal(D_OUT)]
        cur = global_norm(base)
        g = [x * (T / cur) for x in base]
        clipped, total = clip_global_norm(g, MAX_NORM_LIST[0])
        new = global_norm(clipped)
        ok = ok and abs(total / T - 1.0) < TOL_RETURNED and total > new * 1.5
        print("    T=%-4g  返回值 %.9f · 裁剪前范数 %.9f · 裁剪后范数 %.9f"
              % (T, total, T, new))
    check("B.4", ok,
          "★ `clip_grad_norm_` 的返回值**就是裁剪前**的范数（相对差 < %g，逐位级）；"
          "它不是裁剪后的范数（后者 ≈ max_norm = %g）—— 把两者混用会在记录梯度范数时"
          "记成一个**恒定值**" % (TOL_RETURNED, MAX_NORM_LIST[0]))


# ===========================================================================
# 8. C 类：边界与反例
# ===========================================================================
def part_C(X, Y, W, b):
    print()
    print("=" * 74)
    print("C. 边界与反例")
    print("=" * 74)
    # ── C.1 ★ 判据 ⑥ 的前提：有按片归一化时，累积 ≠ 大 batch ────────────────
    _, gW_f, _ = loss_and_grad(X, Y, W, b, normalize=True)
    gW_a, _, _ = grad_accumulated(X, Y, W, b, K_MICRO, normalize=True)
    d = _reldiff(gW_a, gW_f)
    print("  C.1：有 **per-micro-batch 归一化** 时：累积 vs 大 batch 的相对差 %.3e" % d)
    check("C.1", d > TOL_BN_DIFF,
          "★ 有按片统计的归一化层时，累积 **≠** 大 batch（相对差 %.3e > %g）——"
          "这就是判据 ⑥ 必须写「**无归一化层**」这个前提的原因："
          "有 BN 时**必须**改用同步 BN 或放弃累积" % (d, TOL_BN_DIFF))

    # ── C.2 ★ 忘了清零 ⇒ 梯度恰好是大 batch 的 k 倍 ─────────────────────────
    gW_broken, gb_broken = grad_broken_weight(X, Y, W, b, K_MICRO)
    _, gW_full, gb_full = loss_and_grad(X, Y, W, b)
    r1 = gW_broken / gW_full
    r2 = gb_broken / gb_full
    print()
    print("  C.2：忘了按 1/k 加权（= 忘了清零再累加）：dW 的比值 %s"
          % np.round(r1.ravel()[:4], 6))
    check("C.2", float(np.abs(r1 - K_MICRO).max()) < 1e-9
          and float(np.abs(r2 - K_MICRO).max()) < 1e-9,
          "★ 忘了清零 ⇒ 梯度恰好是大 batch 的 **%d 倍**（最大偏离 %.3e）—— "
          "`B.2` 省下来的那点写字节，代价就是**必须**在 step 之后清零"
          % (K_MICRO, float(np.abs(r1 - K_MICRO).max())))

    # ── C.3 ★★ 判据 ⑦ 的边界：T < max_norm ⇒ 压根没裁剪（**P18 的现场**）──
    print()
    print("  C.3：`T < max_norm` 时 `clip_coef > 1` ⇒ **不发生裁剪**：")
    rng = np.random.default_rng(SEED_G + 5)
    base = [rng.standard_normal((D_IN, D_OUT)), rng.standard_normal(D_OUT)]
    cur = global_norm(base)
    T_small = 0.5
    g = [x * (T_small / cur) for x in base]
    clipped, total = clip_global_norm(g, 2.0)
    new = global_norm(clipped)
    print("    T=%g · max_norm=2.0 ⇒ 裁剪后范数 %.9f（== T，**没动**）· 与 max_norm 的相对偏差 %.3f"
          % (T_small, new, abs(new / 2.0 - 1.0)))
    check("C.3", abs(new / T_small - 1.0) < 1e-15 and abs(new / 2.0 - 1.0) > 0.5,
          "★ 范数没超限就**不会**被裁剪：裁剪后范数 == T == %g，与 max_norm 的偏差 %.3f —— "
          "判据「裁剪后 == max_norm」**只在 T > max_norm 时**成立（那份 `T ≥ 1` 的取值范围"
          "是判据的一部分）" % (T_small, abs(new / 2.0 - 1.0)))

    # ── C.4 ★ 裁剪与累积的**顺序**不是可交换的 ─────────────────────────────
    gW_acc, gb_acc, _ = grad_accumulated(X, Y, W, b, K_MICRO)
    # 顺序 A：先累积、再裁剪（**正确**的做法 —— 一步的梯度裁剪一次）
    a = clip_global_norm([gW_acc, gb_acc], 0.3)[0]
    # 顺序 B：先裁剪每一片、再累积（错的：每片各按自己的范数裁）
    n = X.shape[0]
    step = n // K_MICRO
    gW_b2 = np.zeros_like(W)
    gb_b2 = np.zeros_like(b)
    for i in range(K_MICRO):
        s = slice(i * step, (i + 1) * step)
        _, x_, y_ = loss_and_grad(X[s], Y[s], W, b)
        cx, cy = clip_global_norm([x_, y_], 0.3)[0]
        gW_b2 += cx / K_MICRO
        gb_b2 += cy / K_MICRO
    d = max(_reldiff(gW_b2, a[0]), _reldiff(gb_b2, a[1]))
    print()
    print("  C.4：先累积再裁剪 vs 先逐片裁剪再累积：相对差 %.3e" % d)
    check("C.4", d > 1e-2,
          "★ 两种顺序**不可交换**（相对差 %.3e > 1e-2）—— 裁剪必须作用在**一步的**梯度上；"
          "对 micro-batch 的梯度逐个裁剪会得到另一条轨迹" % d)

    # ── C.5 ★ 二阶在**线性**模型上的边界：Hessian 与参数无关 ───────────────
    rng = np.random.default_rng(SEED_G + 9)
    V = rng.standard_normal((D_IN, D_OUT))
    u = rng.standard_normal(D_OUT)
    h1 = hvp_analytic(X, Y, W, b, V, u)
    h2 = hvp_analytic(X, Y, W + 3.0, b - 2.0, V, u)
    print()
    print("  C.5：把参数挪到 `W+3, b−2` 后，解析 HVP 的相对变化 %.3e" % _reldiff(h2[0], h1[0]))
    check("C.5", _reldiff(h2[0], h1[0]) < 1e-15 and _reldiff(h2[1], h1[1]) < 1e-15,
          "★ 在**线性 + 平方损失**这个设定下 Hessian 与参数**无关**（相对变化 %.3e）—— "
          "所以「HVP 与解析一致」这条判据**没有**测到「Hessian 随参数变化」这件能力；"
          "它的作用域只到**实现正确性**" % _reldiff(h2[0], h1[0]))


# ===========================================================================
# 9. D 类：失败标记自检 + specs / 口径 / 阈值钉住
# ===========================================================================
def preflight():
    """★★ **D 类的前半段先跑**：标记自检 + 口径/阈值钉住（理由同 `01_backward.py`）。"""
    print("=" * 74)
    print("D0. 口径/阈值钉住（**先跑**：让「改常量」变成断言捕获，而不是崩溃捕获）")
    print("=" * 74)
    print("  本程序的失败标记 ASSERT_FAIL_MARK = %r" % ASSERT_FAIL_MARK)
    before = len(FAILS)

    hits = _mark_assignments_in(os.path.abspath(__file__))
    uniq = {getattr(v, "value", None) for v in hits}
    check("D.1", len(hits) == 1 and len(uniq) == 1
          and uniq == {ASSERT_FAIL_MARK} and "FAIL" in ASSERT_FAIL_MARK,
          "★ 标记自检：模块级赋值恰好 1 处、取值唯一、== %r、且含 `FAIL`（AST 读出）"
          % ASSERT_FAIL_MARK)
    check("D.2", ASSERT_FAIL_MARK == ".. FAIL: autograd" and "FAIL" in ASSERT_FAIL_MARK,
          "★ 标记自检：**等于**本主题被分配的字面量、且含 `FAIL`")
    check("D.2b", OTHER_THEME_MARKS == (":: FAIL: mixed-precision", "++ FAIL: optimizers",
                                        "// FAIL: init-stability", ";; FAIL: checkpointing"),
          "★ 「不跨主题照抄」的**对照物**逐位钉住 —— 写错一个字，这条检查就静默失效")

    measured = {"D_IN": D_IN, "D_OUT": D_OUT, "N_SAMPLES": N_SAMPLES, "K_MICRO": K_MICRO,
                "N_PARAMS_G": N_PARAMS_G, "SEED_G": SEED_G, "HVP_H": HVP_H,
                "CLIP_T_LIST": CLIP_T_LIST, "MAX_NORM_LIST": MAX_NORM_LIST}
    expect = {"D_IN": 4, "D_OUT": 3, "N_SAMPLES": 12, "K_MICRO": 4, "N_PARAMS_G": 15,
              "SEED_G": 20260925, "HVP_H": 1e-5,
              "CLIP_T_LIST": (2.0, 4.0, 8.0), "MAX_NORM_LIST": (1.0, 0.5, 0.1)}
    diff = {kk: (expect[kk], v) for kk, v in measured.items() if v != expect[kk]}
    check("D.3", not diff and N_PARAMS_G == D_IN * D_OUT + D_OUT,
          "★ 口径常量 9 个逐位一致，且 `N_PARAMS_G == D_IN·D_OUT + D_OUT`（%d = %d×%d + %d）%s"
          % (N_PARAMS_G, D_IN, D_OUT, D_OUT, "" if not diff else " ← 差异 %s" % diff))

    th = {"TOL_ACCUM_REL": TOL_ACCUM_REL, "TOL_CLIP_REL": TOL_CLIP_REL,
          "TOL_CLIP_CLOSED_FORM": TOL_CLIP_CLOSED_FORM,
          "TOL_BN_DIFF": TOL_BN_DIFF, "TOL_HVP_NUM": TOL_HVP_NUM,
          "TOL_HVP_ANA": TOL_HVP_ANA, "TOL_RETURNED": TOL_RETURNED}
    th_expect = {"TOL_ACCUM_REL": 1e-12, "TOL_CLIP_REL": 1.1e-6,
                 "TOL_CLIP_CLOSED_FORM": 1e-6, "TOL_BN_DIFF": 1e-3,
                 "TOL_HVP_NUM": 1e-6, "TOL_HVP_ANA": 1e-14, "TOL_RETURNED": 1e-15}
    tdiff = {kk: (th_expect[kk], v) for kk, v in th.items() if v != th_expect[kk]}
    check("D.4", not tdiff, "★ 阈值 7 个逐位一致（含 2 个**机器精度档**：TOL_ACCUM_REL / "
                            "TOL_HVP_ANA —— 放宽它们，A.1 / A.4 就退化成恒真）%s"
          % ("" if not tdiff else " ← 差异 %s" % tdiff))
    check("D.5", all(t >= 1 for t in CLIP_T_LIST)
          and all(mn < min(CLIP_T_LIST) for mn in MAX_NORM_LIST),
          "★ 矩阵的两条**取值范围**也钉住：所有 `T ≥ 1`（判据 ⑦ 的一部分），"
          "且所有 `max_norm < min(T)`（保证三条都**真的**触发裁剪，不是空跑）")
    return len(FAILS) == before


def part_D():
    print()
    print("=" * 74)
    print("D. specs 常量逐位钉住 与 独立性证据")
    print("=" * 74)
    print("  specs 常量逐位钉住（★ 没有消费者的常量只能是 `make mutate` 的漏网候选）：")
    check("D.6", specs.TORCH_CLIP_COEF_DENOM_EPS == 1e-6,
          "★ `TORCH_CLIP_COEF_DENOM_EPS` == 1e-6 —— ★ 它是 **B 级**（**PyTorch v2.9.0 的源码**，"
          "不是规范）；本程序把它写进 `clip_global_norm()` 的默认参数，"
          "并被 `A.2` / `C.3` 消费（计划 §3 表 A ⑦ **只在这一篇的常量里写一次**）")
    check("D.7", specs.UNSOURCED == ("chen2016_exact_segmented_forward_count",
                                    "torch_clip_coef_eps_version_span"),
          "★ `specs.UNSOURCED` 的两项**逐位钉住** —— 第 1 项（Chen 的分段计数是**本库推导**）"
          "属 `02`，第 2 项（`1e-6` 只在 v2.9.0 上核过）属**本篇**")

    check("D.8", inspect.getsource(hvp_double_reverse) != inspect.getsource(hvp_analytic)
          and inspect.getsource(hvp_analytic) != inspect.getsource(hvp_numeric),
          "★ 防 P17：HVP 的三条路径**源码两两不同**（%d / %d / %d 字节）"
          % (len(inspect.getsource(hvp_double_reverse)),
             len(inspect.getsource(hvp_analytic)),
             len(inspect.getsource(hvp_numeric))))
    check("D.9", MARKS_USED <= {ASSERT_FAIL_MARK},
          "★ 打印失败行时用的标记 ⊆ {%r}（本次实际用到 %d 个）—— 这条只能在最后查"
          % (ASSERT_FAIL_MARK, len(MARKS_USED)))
    print("     表 B 的**引用判据**已被断言化（`B.4`）：返回值 == 裁剪**前**的范数；")
    print("     逐字出处标 **[B] PyTorch 文档**（不是规范）—— 本域的头号错误模式是 **P19**。")


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
    print("03_grad_lifecycle.py —— 训练循环里的梯度：累积 / 清零 / 裁剪 / 二阶")
    print("=" * 74)
    print("  ★ 真值标注：本程序的数字**全部是推导值**（本机 `python3 %s` + `numpy %s`）。"
          % (sys.version.split()[0], np.__version__))
    print("    本机 macOS arm64 **无 GPU**：%d 个参数、%d 个样本的玩具真跑，"
          % (N_PARAMS_G, N_SAMPLES))
    print("    **不得支撑关于真实训练的结论**。")
    print("  ★ 失败标记 ASSERT_FAIL_MARK = %r（本主题自定义；不跨主题照抄）"
          % ASSERT_FAIL_MARK)
    print("  ★ 禁区（一律回链）：`.grad` 的语义 / `grad_fn` / `is_leaf` / `retain_graph` /")
    print("    `backward()` vs `autograd.grad()` —— 正文在 `70-frameworks/pytorch/02-autograd.md`")
    print("    （含 `:139` 那句「用 `autograd.grad` 做裁剪/二阶导时别指望 `.grad` 里有东西」，")
    print("      那是**框架侧**的事实；本程序写的是**算法侧**）。")
    if not preflight():
        print()
        print("!! 口径/标记钉住失败 ⇒ **不继续往下跑**：后续的 A/B/C 只会产生一堆"
              "无意义的连带崩溃")
        return summary()
    X, Y, W, b = make_data()
    part_A(X, Y, W, b)
    part_B(X, Y, W, b)
    part_C(X, Y, W, b)
    part_D()
    return summary()


if __name__ == "__main__":
    sys.exit(main())

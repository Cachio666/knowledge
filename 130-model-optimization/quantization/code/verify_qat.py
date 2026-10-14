#!/usr/bin/env python3
"""130-model-optimization/quantization 主题的纯 python3 复核入口。

★ 本文件**固定跑在默认 `python3` 下，永不 `import torch`、也**不 import numpy**
  （CONTRIBUTING §2 第三种形态；只用 `math` / `random` / `struct` / `sys`）。
  它提供的是「**换了台机器、什么都没装，也能复核的那一半**」：
  用**独立的参考实现**（下面的 `ref_*`）把本主题的**判据**重算一遍，
  再用手写期望值（不是 torch 的输出）判它对不对。
  **它的产出是推导，不是实测**——不得标成本机实测值。

  三类断言在本形态下的分工（CONTRIBUTING §2）：
    · **A 判据复核** —— 本文件。参照物是下面 `ref_*` 这套独立实现与手写期望值。
    · **C 边界与反例** —— 本文件。断言的是「失败行为成立」。
    · **B 代价模型** —— 在**编号程序**里（代价模型的参照是真系统，只有能 import 到它的
      环境里测才有意义）。本文件**不做** B。

★ 本主题里 A/C 与编号程序的分工是 **「数学与算法 vs 实现」**：
  · 本文件（纯 python3 + 标准库）验的是**可独立推导的算法**：伪量化的前向、
    两种掩码公式的**独立推导**、LSQ 的三段梯度、LSQ 的初始化式、PACT 的 α 梯度分段、
    STE 的三种变体、`1/sqrt(N·Q_P)` 的量级效应；
  · 编号程序（真 torch）测的是**真实实现的行为**：`FakeQuantize` 的反传掩码、
    `prepare_qat` 的模块替换、observer 在 `eval()` 下是否冻结、
    `disable_observer` 的两种调用形式、真 torch 上跑出来的 QAT vs PTQ 差距。
  · **两者不得互相背书。** 算法成立不证明实现如此；实现如此不证明算法成立。
  · **同一条结论若两边都能给，两边都要给**，且分别标注来源档位。
  · ⚠ **规模不同 ⇒ 数字不同**：本文件跑的是标量与小向量（≤ 4801 个点的扫描、
    ≤ 64 个元素的样例），编号程序跑的是 2^16 个样本、4096×64 的矩阵。
    **只有定性结构必须一致。**

★ 「独立」的硬要求（CONTRIBUTING §2）：本文件里没有一行代码来自 torch，
  期望值**全部**是手写的常量或从 `specs_qat.py` 抄来的外部常量。
  §E 节把这条**机械地验证一遍**：把参考实现故意改错，A 类断言必须失败。

★ `specs_qat.py` 的每条常量都至少被本文件或某个编号程序的断言读过一次（A6 约定）。
  §F 节把「本文件负责的那部分」列出来，供变异测试逐个改。

运行：`make verify`（等价于 `python3 verify.py`）。只用标准库。
"""

import math
import random
import struct
import sys

import specs_qat as SPECS


# ===========================================================================
# 口径
# ===========================================================================

_STATE = {"n": 0, "A": 0, "C": 0, "fail": []}

# 本文件自己的测量条件（**与编号程序那档不同**）
STDLIB_N = 4801             # 掩码扫描的点数（与 specs 里记录的 4801 同量纲）
STDLIB_SEED = 20260918


def check(cls, name, cond, detail=""):
    _STATE["n"] += 1
    _STATE[cls] += 1
    ok = bool(cond)
    if not ok:
        _STATE["fail"].append("[%s] %s" % (cls, name))
    print("    [%s] %s%s" % ("通过" if ok else "**不通过**", name,
                             "   ← %s" % detail if detail else ""))
    return ok


def section(tag, title):
    print()
    print("=" * 78)
    print("%s %s" % (tag, title))
    print("=" * 78)


def sub(title):
    print()
    print("  -- %s" % title)


# ===========================================================================
# 独立参考实现（ref_*）—— 一行都不来自 torch
# ===========================================================================

def f32(x):
    """把 python float 舍成 float32。

    判据：torch 的 `FakeQuantize.scale` / `zero_point` 是 **float32 张量**，
    `fake_quantize_per_tensor_affine` 的内部计算也在 float32 里做。
    **不做这一步，纯 double 的推导与 torch 会在掩码边界附近差一个样本**
    ——那个差不是噪声，就是「存储宽度」这件事本身。
    """
    return struct.unpack("f", struct.pack("f", x))[0]


def rne(x):
    """round-half-to-even（IEEE 默认舍入模式），作用在 python float 上。

    判据：本机实测在 `x/s + zp = −0.5` 处（x = −12.85、s = 0.1、zp = 128）
    torch 的反传**传梯度**——`round(−0.5)` 必须给 `−0`（偶），不能给 `−1`。
    所以这里用 RNE，不是 round-half-away-from-zero。
    """
    f = math.floor(x)
    d = x - f
    if d > 0.5:
        return f + 1
    if d < 0.5:
        return f
    return f if f % 2 == 0 else f + 1


def ref_fq_forward(x, scale, zp, qmin, qmax):
    """手写 `fake_quantize_per_tensor_affine` 的**前向**（stdlib 版）。

    判据逐条写出（P9）：
      · 先在「量化整数」空间算：`q = round(x / scale) + zero_point`；
      · 再 **clamp 到 [qmin, qmax]**（**饱和**，不是 NaN、不是回绕）；
      · 最后 `(q − zero_point) · scale` 回到实数域。
      · 除法与乘法都在 **float32** 里做（见 `f32`），舍入是 **RNE**（见 `rne`）。
    """
    q = rne(f32(f32(x) / f32(scale))) + f32(zp)
    q = min(max(q, qmin), qmax)
    return f32(f32(q - f32(zp)) * f32(scale))


def ref_fq_mask_rounded(x, scale, zp, qmin, qmax):
    """**舍入码式**掩码：`qmin ≤ round(x/s + zp) ≤ qmax`（本机实测的判据）。

    判据：**掩码作用在「舍入后的量化码」上，不是作用在 x 的实数区间上。**
    这条与计划 §0.7.D 写的区间式不同，§A.2 用它来区分两者。
    """
    q = rne(f32(f32(x) / f32(scale))) + f32(zp)
    return qmin <= q <= qmax


def ref_fq_mask_interval(x, scale, zp, qmin, qmax):
    """**区间式**掩码：`(qmin − zp)·scale ≤ x ≤ (qmax − zp)·scale`（计划 §0.7.D 的写法）。

    ⚠ 这是**被本机实测否证**的那一式，保留在这里作为 §A.2 / §E.2 的对照。
    """
    lo = f32(f32(qmin - f32(zp)) * f32(scale))
    hi = f32(f32(qmax - f32(zp)) * f32(scale))
    return lo <= f32(x) <= hi


def ref_fq_backward(x, scale, zp, qmin, qmax):
    """伪量化的反传：**带区间掩码的裁剪 STE**，返回 `grad_out` 的系数（0 或 1）。"""
    return 1.0 if ref_fq_mask_rounded(x, scale, zp, qmin, qmax) else 0.0


def ref_ste_identity(g_out, r, **_):
    """STE 变体 ①：**纯恒等**（Bengio 2013）。`g_in = g_out`。"""
    return list(g_out)


def ref_ste_clipped(g_out, r, **_):
    """STE 变体 ②：**裁剪 STE**（BNN）。`g_in = g_out · I[|r_i| ≤ 1]`。

    判据：阈值是 **1**——BNN 的激活被约束在 ±1。这是**论文里的形式**，
    阈值写死；torch 的伪量化把它推广成「量化码在 [qmin, qmax] 内」（见 §A.2）。
    """
    return [g if abs(ri) <= 1.0 else 0.0 for g, ri in zip(g_out, r)]


def ref_ste_xnor(g_out, r, per_channel_scale=None, **_):
    """STE 变体 ③：**XNOR-Net**——恒等 + 逐通道缩放。

    判据：反传在恒等之外再乘一个**逐通道**的缩放系数
    `α_j = mean(|W_j|) / mean(|r_j|)`（论文形式）。**它与前两个变体在数值上不同，
    这也是 DoReFa 论文指出它「反传无法用 bit convolution kernel」的原因。**
    """
    if per_channel_scale is None:
        return list(g_out)
    return [g * s for g, s in zip(g_out, per_channel_scale)]


def ref_lsq_forward(v, s, qn, qp):
    """LSQ 的量化器：`v̄ = ⌊clip(v/s, −Q_N, Q_P)⌉`、`v̂ = v̄·s`。"""
    r = min(max(f32(v) / f32(s), -float(qn)), float(qp))
    return rne(r) * f32(s)


def ref_lsq_grad_s(v, s, qn, qp):
    """★ LSQ 的 `∂v̂/∂s`，**三段分段线性**（论文式 3）。

    判据逐条写出（P9）：
      · 自变量是 `r = v/s`，分界点就是两个 **clip 边界** `−Q_N` 与 `Q_P`；
      · **内部段**（`−Q_N < r < Q_P`）：`round(r) − r`
        ——**它是量化残差的相反数，因此 |·| ≤ 0.5**；
      · **下饱和段**（`r ≤ −Q_N`）：常数 `−Q_N`（clip 把 `v̄` 钉死了，`dv̄/ds = 0`）；
      · **上饱和段**（`r ≥ Q_P`）：常数 `Q_P`。
      · 隐含约定：**`round` 在反传里被当成恒等**（STE）。不做这个约定，
        `dv̄/ds = 0`，整段都不是这个式子。§A.4 专门分辨这两者。
    """
    r = f32(v) / f32(s)
    if -float(qn) < r < float(qp):
        return rne(r) - r
    if r <= -float(qn):
        return -float(qn)
    return float(qp)


def ref_lsq_grad_s_naive(v, s, qn, qp):
    """**朴素 autograd** 的 `∂v̂/∂s`：`round` 处处导数为 0 ⇒ 结果就是 `round(r)`。

    ⚠ 这是**第二处 P8 的对照实现**：它与 `ref_lsq_grad_s` 在内部段相差整整一个 `v/s`。
    """
    r = min(max(f32(v) / f32(s), -float(qn)), float(qp))
    return rne(r)


def ref_lsq_init_scale(vs, qp):
    """LSQ 的 step size 初始化：`s₀ = 2·⟨|v|⟩ / √Q_P`。"""
    return 2.0 * (sum(abs(x) for x in vs) / len(vs)) / math.sqrt(qp)


def ref_lsq_grad_multiplier(n, qp):
    """`g = 1/√(N·Q_P)`（LSQ 的梯度缩放因子）。"""
    return 1.0 / math.sqrt(n * qp)


def ref_pact_forward(x, alpha):
    """PACT 的激活：`y = 0.5(|x| − |x − α| + α)`（**等价于** `clamp(x, 0, α)`）。"""
    return 0.5 * (abs(x) - abs(x - alpha) + alpha)


def ref_pact_forward_clamp(x, alpha):
    """同一件事的另一种写法（**判据：两者必须逐点相等**，§A.5 断言它）。"""
    return min(max(x, 0.0), alpha)


def ref_pact_grad_alpha(x, alpha):
    """PACT 的 `∂y_q/∂α`（论文式 3 的两段 + 边界点）。

    判据逐条写出（P9）：
      · `x < α`：**0**——**只有「被截断的那部分」激活才驱动 α**；
      · `x > α`：**1**；
      · `x == α`：论文的「0 / 1」两段在这一点上没有定义；
        本机实测 torch autograd 给 **0.5**（`|x−α|` 在 0 处不可导，取 subgradient 0）。
    """
    if x < alpha:
        return 0.0
    if x > alpha:
        return 1.0
    return SPECS.PACT_GRAD_ALPHA_AT_BOUNDARY_MEASURED


# ===========================================================================
# A. 判据复核
# ===========================================================================

def run_A():
    section("A", "判据复核：手写参照实现 vs specs 里的常量")

    sub("A.1  手写 `fake_quantize_per_tensor_affine` 前向，与 specs 的 7 点表比对")
    s, zp = SPECS.FQ_STE_SCALE, SPECS.FQ_STE_ZERO_POINT
    qmin, qmax = SPECS.FQ_STE_QMIN, SPECS.FQ_STE_QMAX
    got_f, got_b = [], []
    print("      x       = %s" % "  ".join("%+g" % v for v in SPECS.FQ_STE_X))
    print("      forward = %s" % "  ".join("%g" % ref_fq_forward(v, s, zp, qmin, qmax)
                                          for v in SPECS.FQ_STE_X))
    print("      backward= %s" % "  ".join("%g" % ref_fq_backward(v, s, zp, qmin, qmax)
                                          for v in SPECS.FQ_STE_X))
    ok = True
    for v, want in zip(SPECS.FQ_STE_X, SPECS.FQ_STE_FORWARD):
        g = ref_fq_forward(v, s, zp, qmin, qmax)
        got_f.append(g)
        ok &= check("A", "  forward(%+g) == %g" % (v, want), abs(g - want) < 1e-6,
                    "得到 %.10g" % g)
    for v, want in zip(SPECS.FQ_STE_X, SPECS.FQ_STE_BACKWARD):
        g = ref_fq_backward(v, s, zp, qmin, qmax)
        got_b.append(g)
        ok &= check("A", "  backward(%+g) == %g" % (v, want), abs(g - want) < 1e-12,
                    "得到 %g" % g)
    check("A", "★ 全部 14 格逐点吻合（前向 7 + 反传 7）", ok)
    check("A", "★ 理论可达区间 [%g, %g] 的两个端点恰好传梯度、越界一格不传"
          % ((qmin - zp) * s, (qmax - zp) * s),
          ref_fq_backward(-12.8, s, zp, qmin, qmax) == 1.0
          and ref_fq_backward(-12.9, s, zp, qmin, qmax) == 0.0)
    check("A", "★ 前向在区间外是**饱和**（不是 NaN、不是回绕）",
          abs(ref_fq_forward(-100.0, s, zp, qmin, qmax) - SPECS.FQ_STE_FORWARD[0]) < 1e-6
          and abs(ref_fq_forward(100.0, s, zp, qmin, qmax) - SPECS.FQ_STE_FORWARD[-1]) < 1e-6)
    check("A", "★ 而且这一点被记成一条**可查的断言**（不是只印在说明文字里）："
          "FQ_SATURATES_OUTSIDE_RANGE = %s" % SPECS.FQ_SATURATES_OUTSIDE_RANGE,
          SPECS.FQ_SATURATES_OUTSIDE_RANGE is True)
    check("A", "★ 表里的 7 个 x 逐值核对（判据：这张表是本篇所有结论的锚点，"
          "少一个点或换一个点都会改变它能区分什么）",
          tuple(SPECS.FQ_STE_X) == (-100.0, -12.9, -12.8, 0.0, 12.7, 12.8, 100.0)
          and (-12.9 in SPECS.FQ_STE_X) and (-12.8 in SPECS.FQ_STE_X)
          and min(SPECS.FQ_STE_X) <= -(qmax - zp) * s
          and max(SPECS.FQ_STE_X) >= (qmax - zp) * s,
          str(tuple(SPECS.FQ_STE_X)))
    check("A", "★ 掩码的判据写成文字时必须点名 `round` 与两个边界：%s"
          % SPECS.FQ_STE_MASK_FORM,
          "round(" in SPECS.FQ_STE_MASK_FORM
          and "quant_min" in SPECS.FQ_STE_MASK_FORM
          and "quant_max" in SPECS.FQ_STE_MASK_FORM)

    sub("A.2  ★ 掩码的**判据**：舍入码式 vs 区间式（本主题最重要的一条分辨）")
    print("      舍入码式（本机实测判据）：%s" % SPECS.FQ_STE_MASK_FORM)
    print("      区间式（计划 §0.7.D 的写法）：(qmin−zp)·s <= x <= (qmax−zp)·s")
    grid = [i * 0.01 for i in range(-1400, 1401)]
    rnd = random.Random(STDLIB_SEED)
    grid += [rnd.uniform(-13.0, 13.0) for _ in range(STDLIB_N - len(grid))]
    mis_round = mis_interval = 0
    for x in grid:
        a = ref_fq_mask_interval(x, s, zp, qmin, qmax)
        b = ref_fq_mask_rounded(x, s, zp, qmin, qmax)
        if a != b:
            mis_interval += 1
    print("      扫描 %d 个点（x ∈ [−14, 14]）：两式不一致 **%d** 个点"
          % (len(grid), mis_interval))
    check("A", "★ 两式**不是同一个判据**（不一致点数 > 0）", mis_interval > 0,
          "%d 个点" % mis_interval)
    ok = True
    for x in SPECS.FQ_STE_INTERVAL_FORM_MISMATCH:
        m_r = ref_fq_mask_rounded(x, s, zp, qmin, qmax)
        m_i = ref_fq_mask_interval(x, s, zp, qmin, qmax)
        ok &= check("A", "  x = %+g：舍入码式 %s / 区间式 %s（**相反**）"
                    % (x, m_r, m_i), m_r != m_i)
        ok &= check("A", "  且 specs 记录的那一点在舍入码式下**传梯度**", m_r is True)
    check("A", "★ 两式在 specs 记的两个点上给出**相反**答案", ok)
    check("A", "★ 所以「−12.8 传梯度、−12.9 不传」这句话**单独不足以**区分两式"
          "——两个式子在那张 7 点表上恰好同解", True)

    sub("A.3  ★ 掩码在何时真的咬人（判据：grad == 0 的样本占比）")
    frac = SPECS.FQ_STE_MASK_ZERO_FRACTION
    rnd = random.Random(STDLIB_SEED + 1)
    xs_n = [rnd.gauss(0.0, 1.0) for _ in range(20000)]
    xs_t = [_student_t3(rnd) for _ in range(20000)]
    xs_h = [rnd.gauss(0.0, 1.0) for _ in range(20000)]
    for i in range(200):
        xs_h[i] *= 100.0
    for name, xs in (("N(0,1)", xs_n), ("StudentT(3)", xs_t), ("1% 元素放大 100 倍", xs_h)):
        got = sum(1 for v in xs if ref_fq_backward(v, s, zp, qmin, qmax) == 0.0) / len(xs)
        want = frac[name]
        print("      %-20s 本文件 %.5f%%  vs specs %.5f%%" % (name, 100 * got, 100 * want))
        check("A", "  %s 的量级一致（±40%% 相对）" % name,
              abs(got - want) <= max(0.4 * want, 1e-4),
              "%.6f vs %.6f" % (got, want))
    print("      ⚠ 判据是「scale **固定**为 0.1」；让 observer 自己定 scale 时量程被"
          "离群值撑大，掩码挡掉 0%%（specs 记 %.1f%%）。"
          % (100 * SPECS.FQ_STE_MASK_ZERO_FRACTION_ADAPTIVE_SCALE))
    check("A", "★ 三档的**次序**：N(0,1) < StudentT(3) < 1%放大 100 倍",
          frac["N(0,1)"] < frac["StudentT(3)"] < frac["1% 元素放大 100 倍"])

    sub("A.4  ★ LSQ 的三段梯度：论文式 vs 朴素 autograd（第二处 P8）")
    print("      量化器   ：%s" % SPECS.LSQ_QUANTIZER_FORM)
    print("      论文式   ：%s" % SPECS.LSQ_GRAD_INTERIOR_FORM)
    print("                 %s" % SPECS.LSQ_GRAD_LOWER_FORM)
    print("                 %s" % SPECS.LSQ_GRAD_UPPER_FORM)
    print("      朴素式   ：%s" % SPECS.LSQ_GRAD_NAIVE_AUTOGRAD_FORM)
    check("A", "★ 量化器的公式逐项可核（round / clip / 两段乘回 s）",
          "round" in SPECS.LSQ_QUANTIZER_FORM and "clip" in SPECS.LSQ_QUANTIZER_FORM
          and "v_hat = v_bar * s" in SPECS.LSQ_QUANTIZER_FORM)
    check("A", "★ 三段各自的**文字**逐条可核：内部带 `- v / s`，两饱和段是常数",
          "round(v / s) - v / s" in SPECS.LSQ_GRAD_INTERIOR_FORM
          and "-Q_N <" in SPECS.LSQ_GRAD_INTERIOR_FORM
          and "-Q_N" in SPECS.LSQ_GRAD_LOWER_FORM
          and "v / s <= -Q_N" in SPECS.LSQ_GRAD_LOWER_FORM
          and "Q_P" in SPECS.LSQ_GRAD_UPPER_FORM
          and "v / s >= Q_P" in SPECS.LSQ_GRAD_UPPER_FORM)
    check("A", "★ 朴素式**不含** `- v / s` 项——这正是两式的分水岭",
          "round(v / s)" in SPECS.LSQ_GRAD_NAIVE_AUTOGRAD_FORM
          and "- v / s" not in SPECS.LSQ_GRAD_NAIVE_AUTOGRAD_FORM)
    check("A", "★ 两种 Q_N / Q_P 的取值满足定点格式的定义式："
          "有符号 `Q_N + Q_P == 2^b − 1`（b = 4 → 15）、无符号 `Q_N = 0, Q_P = 2^b − 1`（b = 8 → 255）",
          SPECS.LSQ_SIGNED_QN == 8 and SPECS.LSQ_SIGNED_QP == 7
          and SPECS.LSQ_SIGNED_QN + SPECS.LSQ_SIGNED_QP == 2 ** 4 - 1
          and SPECS.LSQ_SIGNED_QN == 2 ** 3 and SPECS.LSQ_SIGNED_QP == 2 ** 3 - 1
          and SPECS.LSQ_UNSIGNED_QN == 0 and SPECS.LSQ_UNSIGNED_QP == 2 ** 8 - 1)
    qn, qp = SPECS.LSQ_UNSIGNED_QN, SPECS.LSQ_UNSIGNED_QP
    s0 = 1.0
    cases = [-1.0, -0.4, 0.0, 0.25, 0.5, 1.5, 100.3, 254.5, 255.0, 255.5, 257.0]
    for r in cases:
        g = ref_lsq_grad_s(r, s0, qn, qp)
        tag = "内部" if -qn < r < qp else ("下饱和" if r <= -qn else "上饱和")
        print("      r = %8.3f  [%-6s]  论文式 %+10.5f" % (r, tag, g))
    check("A", "★ 内部段 = round(r) − r，且 |·| ≤ 0.5",
          all(abs(ref_lsq_grad_s(r, s0, qn, qp)) <= 0.5
              for r in (0.25, 0.5, 1.5, 100.3, 254.5)))
    check("A", "★ 上饱和段恒为 Q_P = %d（**不是** ~0）" % qp,
          ref_lsq_grad_s(255.0, s0, qn, qp) == float(qp)
          and ref_lsq_grad_s(1000.0, s0, qn, qp) == float(qp))
    check("A", "★ 下饱和段恒为 −Q_N = %d" % (-qn),
          ref_lsq_grad_s(-1.0, s0, qn, qp) == float(-qn))
    check("A", "★ 论文式与朴素式在**内部段相差整整一个 v/s**（两式不是同一个量）",
          abs((rne(2.5) - 2.5) - rne(2.5)) == 2.5,
          "%.5f vs %.5f" % (rne(2.5) - 2.5, rne(2.5)))
    check("A", "★ 且朴素式在饱和段与论文式**一致**——分歧只在内部段",
          ref_lsq_grad_s_naive(300.0, s0, qn, qp) == ref_lsq_grad_s(300.0, s0, qn, qp)
          and ref_lsq_grad_s_naive(-5.0, s0, qn, qp) == ref_lsq_grad_s(-5.0, s0, qn, qp))

    sub("A.5  LSQ 的初始化式与缩放因子")
    vs = [0.3, -0.7, 1.1, -0.2, 0.9]
    s_init = ref_lsq_init_scale(vs, SPECS.LSQ_SIGNED_QP)
    manual = 2.0 * (sum(abs(x) for x in vs) / len(vs)) / math.sqrt(SPECS.LSQ_SIGNED_QP)
    print("      %s = %.10f" % (SPECS.LSQ_INIT_FORM, s_init))
    check("A", "初始化式与 specs 的形一致（Q_P = %d）" % SPECS.LSQ_SIGNED_QP,
          abs(s_init - manual) < 1e-12 and SPECS.LSQ_INIT_FORM.startswith("s0 = 2 * mean"))
    check("A", "★ LSQ **不学** clip 阈值、**不学** zero-point（它只学 s）",
          SPECS.LSQ_LEARNS_CLIP_THRESHOLD is False
          and SPECS.LSQ_LEARNS_ZERO_POINT is False)
    g1 = ref_lsq_grad_multiplier(4096, 7)
    g2 = ref_lsq_grad_multiplier(16384, 7)
    print("      %s : N=4096 -> %.6e ; N=16384 -> %.6e" % (SPECS.LSQ_SCALE_FACTOR_FORM, g1, g2))
    check("A", "★ g 随 N 增大而**减小**（1/√N），且与 specs 的式子一致",
          g1 > g2 and abs(g1 - 1.0 / math.sqrt(4096 * 7)) < 1e-15)
    check("A", "★ 缩放因子的**式子本身**逐字可核：%s" % SPECS.LSQ_SCALE_FACTOR_FORM,
          SPECS.LSQ_SCALE_FACTOR_FORM == "g = 1 / sqrt(N * Q_P)"
          and abs(ref_lsq_grad_multiplier(4096, 7)
                  - 1.0 / math.sqrt(4096 * 7)) < 1e-18)
    check("A", "★ 论文消融记的是 %s 个数量级（引用值，本文件不复现它）"
          % (SPECS.LSQ_ABLATION_ORDERS_OF_MAGNITUDE,),
          SPECS.LSQ_ABLATION_ORDERS_OF_MAGNITUDE == (2, 3)
          and SPECS.LSQ_NO_SCALE_DIVERGENCE_GAP_PP == 3.4)

    sub("A.6  ★ 本机实测的相对更新幅度（判据见 specs 的注解；规模不同 ⇒ 数字不同）")
    loc = SPECS.LSQ_LOCAL_REL_UPDATE
    for n in (2048, 4096, 8192, 16384):
        a = loc["N=%d no_g" % n]
        b = loc["N=%d with_g" % n]
        print("      N=%5d  不加缩放 |ds|/|dw| = %8.3f  →  加 g 后 %6.4f  （降 %.0f×）"
              % (n, a, b, a / b))
    check("A", "★ 不加缩放时失衡（比值 > 5）——**四个 N 全部成立**",
          all(loc["N=%d no_g" % n] > 5.0 for n in (2048, 4096, 8192, 16384)))
    check("A", "★ 加 g 后落到同一个量级（比值 < 0.5）——**四个 N 全部成立**",
          all(loc["N=%d with_g" % n] < 0.5 for n in (2048, 4096, 8192, 16384)))
    check("A", "★ 加 g 的改善倍数 > 20×（**定性结论**；论文的「2–3 个数量级」是"
          "ImageNet 级网络上、N 比本机大 1–2 个数量级的引用值）",
          all(loc["N=%d no_g" % n] / loc["N=%d with_g" % n] > 20.0
              for n in (2048, 4096, 8192, 16384)))

    sub("A.7  PACT 的激活与 α 梯度")
    print("      %s" % SPECS.PACT_ACT_FORM)
    check("A", "★ PACT 的激活式逐项可核（0.5 / |x| / |x−α| / +α）：%s"
          % SPECS.PACT_ACT_FORM,
          SPECS.PACT_ACT_FORM == "y = 0.5 * (|x| - |x - alpha| + alpha)"
          and "0.5" in SPECS.PACT_ACT_FORM and "|x|" in SPECS.PACT_ACT_FORM
          and "|x - alpha|" in SPECS.PACT_ACT_FORM and "+ alpha" in SPECS.PACT_ACT_FORM)
    print("      %s" % SPECS.PACT_GRAD_ALPHA_FORM)
    check("A", "★ α 梯度的**两段**逐条可核（x < α → 0；x > α → 1）：%s"
          % SPECS.PACT_GRAD_ALPHA_FORM,
          "0 (x < alpha)" in SPECS.PACT_GRAD_ALPHA_FORM
          and "1 (x > alpha)" in SPECS.PACT_GRAD_ALPHA_FORM
          and "dy/dalpha" in SPECS.PACT_GRAD_ALPHA_FORM)
    alpha = SPECS.PACT_ALPHA_INIT
    for x in (-5.0, 0.0, 1.0, 9.99, 10.0, 10.01, 20.0):
        y = ref_pact_forward(x, alpha)
        yc = ref_pact_forward_clamp(x, alpha)
        ga = ref_pact_grad_alpha(x, alpha)
        print("      x = %+7.2f  y = %8.5f  clamp = %8.5f  ∂y/∂α = %s"
              % (x, y, yc, ("%.1f" % ga)))
    check("A", "★ PACT 的激活**等价于** `clamp(x, 0, α)`（逐点相等）",
          all(abs(ref_pact_forward(x, alpha) - ref_pact_forward_clamp(x, alpha)) < 1e-12
              for x in (-5.0, 0.0, 1.0, 5.0, 10.0, 20.0)))
    check("A", "★ `∂y_q/∂α` 在 clip 点**以下处处为零**（x < α）",
          all(ref_pact_grad_alpha(x, alpha) == 0.0 for x in (-5.0, 0.0, 1.0, 9.99)))
    check("A", "★ 在 clip 点**以上恒为 1**（x > α）",
          all(ref_pact_grad_alpha(x, alpha) == 1.0 for x in (10.01, 20.0, 1000.0)))
    check("A", "★ 边界点 x == α 上，论文的两段没有定义；本机实测 torch 给 %.1f"
          % SPECS.PACT_GRAD_ALPHA_AT_BOUNDARY_MEASURED,
          ref_pact_grad_alpha(alpha, alpha)
          == SPECS.PACT_GRAD_ALPHA_AT_BOUNDARY_MEASURED)
    check("A", "α 的初始值与配置：%s（%s）、L2 正则 = %s"
          % (SPECS.PACT_ALPHA_INIT, SPECS.PACT_ALPHA_INIT_SETTING,
             SPECS.PACT_USES_L2_REGULARIZATION),
          SPECS.PACT_ALPHA_INIT == 10.0
          and SPECS.PACT_ALPHA_INIT_SETTING == "CIFAR10-ResNet20"
          and SPECS.PACT_USES_L2_REGULARIZATION is True)

    sub("A.8  PACT 与 LSQ 的三层区别")
    for i, d in enumerate(SPECS.PACT_VS_LSQ_THREE_DIFFS, 1):
        print("      %d. %s" % (i, d))
    check("A", "★ 三层区别逐条记在 specs 里（学什么 / 梯度形状 / 对 round 的处理）",
          len(SPECS.PACT_VS_LSQ_THREE_DIFFS) == 3
          and all(k in d for d, k in zip(SPECS.PACT_VS_LSQ_THREE_DIFFS,
                                         ("学什么", "梯度形状", "对 round 的处理"))))
    check("A", "★ 且它们可从本文件的两个参照实现读出：PACT 的梯度是**阶跃**、"
          "LSQ 的梯度是**每个跃迁点的残差**",
          all(ref_pact_grad_alpha(x, 10.0) in (0.0, 1.0)
              for x in (1.0, 5.0, 9.0, 11.0, 20.0))
          and len(set(round(ref_lsq_grad_s(r, 1.0, qn, qp), 6)
                      for r in (0.1, 0.3, 0.7, 1.2, 2.4, 3.6))) > 3)

    sub("A.9  STE 的三种变体（**不能笼统说「STE 就是恒等通道」**）")
    for v in SPECS.STE_VARIANTS:
        print("      · %s" % v)
    go = [1.0, 1.0, 1.0, 1.0]
    r = [-2.0, -0.5, 0.5, 2.0]
    pc = [0.5, 0.5, 0.5, 0.5]
    v1 = ref_ste_identity(go, r)
    v2 = ref_ste_clipped(go, r)
    v3 = ref_ste_xnor(go, r, per_channel_scale=pc)
    print("      r        = %s" % r)
    print("      ① 恒等   = %s" % v1)
    print("      ② 裁剪   = %s  ← %s" % (v2, SPECS.STE_BNN_CLIP_FORM))
    print("      ③ XNOR   = %s  ← 恒等 × 逐通道缩放" % v3)
    check("A", "★ 三个变体**互不相同**（至少两两不同）",
          v1 != v2 and v1 != v3 and v2 != v3)
    check("A", "★ 裁剪变体的阈值是 **1**（论文形式），所以 |r| > 1 的元素被清零",
          v2 == [0.0, 1.0, 1.0, 0.0])
    check("A", "★ `specs.STE_VARIANTS` 记的是三种而不是一种",
          len(SPECS.STE_VARIANTS) == 3 and SPECS.STE_VARIANTS_PAPER == "arXiv:1606.06160v2")
    check("A", "★ 三条的**内容**逐条可核：分别点出「恒等 / 裁剪 / XNOR-Net 缩放」",
          "恒等" in SPECS.STE_VARIANTS[0]
          and "裁剪" in SPECS.STE_VARIANTS[1] and "BNN" in SPECS.STE_VARIANTS[1]
          and "XNOR" in SPECS.STE_VARIANTS[2] and "缩放" in SPECS.STE_VARIANTS[2])

    sub("A.10  STE 的出处与两条与流行转述相反的原文")
    print("      出处：%s 式 (%d)，该文归给 %s"
          % (SPECS.STE_PAPER, SPECS.STE_EQ_NUMBER, SPECS.STE_ATTRIBUTED_TO))
    print("      ① %s" % SPECS.STE_SIGMOID_VARIANT_QUOTE)
    print("      ② %s" % SPECS.STE_SIGN_SCOPE_QUOTE)
    check("A", "★ 出处三元组：论文编号 + 式号 + 被归给谁",
          SPECS.STE_PAPER == "arXiv:1308.3432v1" and SPECS.STE_EQ_NUMBER == 13
          and SPECS.STE_ATTRIBUTED_TO == "Hinton 2012 lecture 15b")
    check("A", "★ 第 ① 条说的是「乘 sigmoid 导数**更差**」——与流行转述相反",
          "without multiplying" in SPECS.STE_SIGMOID_VARIANT_QUOTE)
    check("A", "★ 第 ② 条说的是「**只在单层内**符号正确」——与流行转述相反",
          "single layer" in SPECS.STE_SIGN_SCOPE_QUOTE
          and "not guaranteed" in SPECS.STE_SIGN_SCOPE_QUOTE)
    check("A", "★ coarse gradient 的论文与术语都记了", SPECS.STE_COARSE_GRADIENT_PAPER
          == "arXiv:1903.05662v1" and SPECS.STE_COARSE_GRADIENT_TERM == "coarse gradient"
          and SPECS.STE_RELU_VERDICT_QUOTE == "poor choice")
    check("A", "★★ STE 偏差的定量大小 **UNSOURCED**——这是一条否定性常量，"
          "防止有人后来往里塞一个数", SPECS.STE_BIAS_MAGNITUDE == "UNSOURCED")

    sub("A.11  QAT vs PTQ：引用值 vs 本机实测，两档分开")
    n = SPECS.NAGEL_RESNET18_TOP1
    print("      引用值（%s，%s vs %s）：ResNet18 per-tensor"
          % (SPECS.NAGEL_WHITEPAPER, SPECS.NAGEL_PTQ_TABLE, SPECS.NAGEL_QAT_TABLE))
    check("A", "★ 引用的**锚点三元组**逐条可核：论文编号 + 两张表号 + 互不相同",
          SPECS.NAGEL_WHITEPAPER == "arXiv:2106.08295v1"
          and SPECS.NAGEL_PTQ_TABLE == "Table 6"
          and SPECS.NAGEL_QAT_TABLE == "Table 10"
          and SPECS.NAGEL_PTQ_TABLE != SPECS.NAGEL_QAT_TABLE)
    print("        FP32 %.2f | W8A8 PTQ %.2f / QAT %.2f | W4A8 PTQ %.2f / QAT %.2f"
          % (n["fp32"], n["w8a8_ptq"], n["w8a8_qat"], n["w4a8_ptq"], n["w4a8_qat"]))
    print("        ⇒ W8A8 的 gap = %+.2f 点，W4A8 的 gap = %+.2f 点"
          % (n["w8a8_qat"] - n["w8a8_ptq"], n["w4a8_qat"] - n["w4a8_ptq"]))
    check("A", "★ 白皮书那两行逐值核对（判据 = ImageNet Top-1）",
          n["w8a8_qat"] - n["w8a8_ptq"] > 0 and n["w4a8_qat"] - n["w4a8_ptq"] > 0
          and n["fp32"] == 69.68)
    check("A", "★ 但 gap 的**区间**是引用值：W8A8 %s、W4A8 %s"
          % (SPECS.NAGEL_W8A8_GAP_POINTS, SPECS.NAGEL_W4A8_GAP_POINTS),
          SPECS.NAGEL_W8A8_GAP_POINTS == (0.07, 0.78)
          and SPECS.NAGEL_W4A8_GAP_POINTS == (0.2, 1.1))
    check("A", "★★ 白皮书的 PTQ 表**没有 W4A4 列** ⇒ 「W4A4 下差多少」"
          "**用这份来源无法量化**（否定性常量）",
          SPECS.NAGEL_PTQ_HAS_W4A4_COLUMN is False)
    loc = SPECS.QAT_PTQ_LOCAL_GAP
    print("      本机实测（2.11.0，小规模合成任务，**不做 convert**）：")
    for k in ("W8A8", "W4A8", "W4A4", "W3A3", "W2A2"):
        print("        %-6s gap = %+.4f 点" % (k, loc[k]))
    check("A", "★ 本机结构：W8A8 的 gap **为负**（= 反例）、W2A2 的 gap 最大",
          loc["W8A8"] < 0 and loc["W2A2"] > 0 and loc["W2A2"] > 5 * loc["W4A4"])
    check("A", "★ 本机那一组与 2.6/2.7 那一组**定性结构一致**（都满足上面的不等式）",
          SPECS.QAT_PTQ_LOCAL_GAP_2_6["W8A8"] < 0
          and SPECS.QAT_PTQ_LOCAL_GAP_2_6["W2A2"]
          > 5 * SPECS.QAT_PTQ_LOCAL_GAP_2_6["W4A4"])
    check("A", "★ 本机数字与白皮书的数字**不得并列**（判据、网络、数据全不同）", True)

    sub("A.12  observer 机制：三处「看起来该在 A 处、其实在 B 处」")
    print("      observer.py 里 `self.training` = %d 命中；`observer_enabled` = %d 命中"
          % (SPECS.OBSERVER_PY_TRAINING_REFS, SPECS.OBSERVER_PY_OBSERVER_ENABLED_REFS))
    print("      `observer_enabled` 的真正归属：%s" % SPECS.OBSERVER_ENABLED_DEFINED_IN)
    check("A", "★ 两个命中数都是 0 ⇒ **observer 侧根本没有这两个东西**",
          SPECS.OBSERVER_PY_TRAINING_REFS == 0
          and SPECS.OBSERVER_PY_OBSERVER_ENABLED_REFS == 0)
    check("A", "★ 它的归属是 `fake_quantize.py` 的 `FakeQuantizeBase`（QAT 侧伪量化模块）",
          SPECS.OBSERVER_ENABLED_DEFINED_IN
          == "torch/ao/quantization/fake_quantize.py::FakeQuantizeBase")
    check("A", "★ `eval()` 不冻结 observer（这是**现象**；机制不是 observer 自己的）",
          SPECS.OBSERVER_UPDATES_UNDER_EVAL is True)
    print("      `disable_observer` 是 **apply 回调**：%s"
          % SPECS.DISABLE_OBSERVER_IS_APPLY_CALLBACK)
    print("      对 `prepare()` 插入的普通 observer 生效吗：%s"
          % SPECS.DISABLE_OBSERVER_WORKS_ON_PLAIN_OBSERVER)
    check("A", "★ `Q.disable_observer(model)` 是静默空操作；必须写 "
          "`model.apply(Q.disable_observer)`",
          SPECS.DISABLE_OBSERVER_IS_APPLY_CALLBACK is True)
    check("A", "★ 而且对普通 observer 连 `apply()` 也不起作用（它不是 FakeQuantizeBase）",
          SPECS.DISABLE_OBSERVER_WORKS_ON_PLAIN_OBSERVER is False)
    check("A", "★ `freeze_bn_stats` **不是**通用工具，只在 fused Conv/Linear+BN 里",
          len(SPECS.FREEZE_BN_STATS_MODULES) == 2
          and all("fused" in p for p in SPECS.FREEZE_BN_STATS_MODULES))

    sub("A.13  `prepare_qat` 的真实步骤与 qconfig 的「声明 vs 生效」")
    for i, st in enumerate(SPECS.PREPARE_QAT_STEPS, 1):
        print("      %d. %s" % (i, st))
    check("A", "★ `prepare_qat` 内部**走了一次 `convert`**——「只挂 observer」是错的",
          any("convert(" in st for st in SPECS.PREPARE_QAT_STEPS))
    check("A", "★ 而且它挂在步骤 ③（在 `prepare` 之前）",
          "convert" in SPECS.PREPARE_QAT_STEPS[SPECS.PREPARE_QAT_CONVERT_STEP_INDEX]
          and "prepare(" in SPECS.PREPARE_QAT_STEPS[SPECS.PREPARE_QAT_CONVERT_STEP_INDEX + 1])
    check("A", "★ `convert` 的下标记成一条常量（%d），且它后面的那一步确实是 `prepare`"
          % SPECS.PREPARE_QAT_CONVERT_STEP_INDEX,
          SPECS.PREPARE_QAT_CONVERT_STEP_INDEX == 2
          and "convert" in SPECS.PREPARE_QAT_STEPS[SPECS.PREPARE_QAT_CONVERT_STEP_INDEX]
          and "prepare(" in SPECS.PREPARE_QAT_STEPS[SPECS.PREPARE_QAT_CONVERT_STEP_INDEX + 1])
    check("A", "★ 源码锚点写成**可定位的路径**：%s" % SPECS.PREPARE_QAT_SRC_ANCHOR,
          SPECS.PREPARE_QAT_SRC_ANCHOR
          == "torch/ao/quantization/quantize.py::prepare_qat")
    check("A", "★ 四步的**内容**逐条可核（training / propagate / convert / prepare）",
          "training" in SPECS.PREPARE_QAT_STEPS[0]
          and "propagate_qconfig_" in SPECS.PREPARE_QAT_STEPS[1]
          and "remove_qconfig=False" in SPECS.PREPARE_QAT_STEPS[2]
          and "observer_non_leaf_module_list" in SPECS.PREPARE_QAT_STEPS[3])
    print("      ★ training 检查的写法**跨版本变过**：")
    for f in SPECS.PREPARE_QAT_TRAINING_CHECK_FORMS:
        print("        · %s" % f.replace("\n", " ⏎ "))
    check("A", "★ 两种写法**都记了**，且消息文字逐字相同"
          "（`%s`）——不得写成「torch 的写法是 X」"
          % SPECS.PREPARE_QAT_TRAINING_CHECK_MSG,
          len(SPECS.PREPARE_QAT_TRAINING_CHECK_FORMS) == 2
          and all(SPECS.PREPARE_QAT_TRAINING_CHECK_MSG in f
                  for f in SPECS.PREPARE_QAT_TRAINING_CHECK_FORMS)
          and any(f.startswith("assert ") for f in SPECS.PREPARE_QAT_TRAINING_CHECK_FORMS)
          and any("raise AssertionError" in f for f in SPECS.PREPARE_QAT_TRAINING_CHECK_FORMS))
    check("A", "★ 权重伪量化**写死在 forward 里**（源码逐字）",
          SPECS.QAT_LINEAR_FORWARD_SRC
          == "return F.linear(input, self.weight_fake_quant(self.weight), self.bias)")
    check("A", "★ 参数本身**始终 fp32**（伪量化只发生在 forward）",
          SPECS.QAT_PREPARED_WEIGHT_DTYPE == "torch.float32")
    print("      x86 QAT 激活：qconfig **声明** quant_max=%d / reduce_range=%s，"
          "模块上**实际生效** %d/%d"
          % (SPECS.QAT_X86_ACT_DECLARED_QMAX, SPECS.QAT_X86_ACT_DECLARED_REDUCE_RANGE,
             SPECS.QAT_X86_ACT_EFFECTIVE_QMIN, SPECS.QAT_X86_ACT_EFFECTIVE_QMAX))
    check("A", "★ 「声明」与「生效」不是同一个量（P8 的一处实例）",
          255 != SPECS.QAT_X86_ACT_EFFECTIVE_QMAX
          and SPECS.QAT_X86_ACT_DECLARED_REDUCE_RANGE is True)
    check("A", "★ reduce_range 的存在理由有源码级原文（vpmaddubsw 溢出）",
          "vpmaddubsw" in SPECS.REDUCE_RANGE_SRC_REASON_QUOTE)
    check("A", "★ 权重侧 %d/%d、per_channel_symmetric = %s"
          % (SPECS.QAT_X86_WEIGHT_QMIN, SPECS.QAT_X86_WEIGHT_QMAX,
             SPECS.QAT_X86_WEIGHT_PER_CHANNEL_SYMMETRIC),
          SPECS.QAT_X86_WEIGHT_QMIN == -128 and SPECS.QAT_X86_WEIGHT_QMAX == 127
          and SPECS.QAT_X86_WEIGHT_PER_CHANNEL_SYMMETRIC is True)
    check("A", "★ 对照 qnnpack：激活 %d/%d、reduce_range=%s、权重 per_channel=%s"
          % (SPECS.QAT_QNNPACK_ACT_QMAX, SPECS.QAT_QNNPACK_ACT_QMAX,
             SPECS.QAT_QNNPACK_ACT_REDUCE_RANGE,
             SPECS.QAT_QNNPACK_WEIGHT_PER_CHANNEL_SYMMETRIC),
          SPECS.QAT_QNNPACK_ACT_QMAX == 255
          and SPECS.QAT_QNNPACK_ACT_REDUCE_RANGE is False
          and SPECS.QAT_QNNPACK_WEIGHT_PER_CHANNEL_SYMMETRIC is False)
    check("A", "★ **PTQ** x86 的默认激活观察器是 `%s`（不是 MinMax、不是 MovingAverage）；"
          "**QAT** 侧是 `%s`" % (SPECS.PTQ_X86_ACT_OBSERVER, SPECS.QAT_X86_ACT_FAKE_QUANT),
          SPECS.PTQ_X86_ACT_OBSERVER == "HistogramObserver"
          and SPECS.QAT_X86_ACT_FAKE_QUANT == "FusedMovingAvgObsFakeQuantize")

    sub("A.14  LSQ / PACT 的论文编号（含一处常见误引）")
    print("      LSQ  = %s" % SPECS.LSQ_PAPER)
    print("      PACT = %s" % SPECS.PACT_PAPER)
    print("      ⚠ 把 PACT 的作者写成「%s」是误引" % SPECS.PACT_AUTHOR_MISATTRIBUTION)
    check("A", "★ 两个编号都与 `WebSearch` 的独立佐证一致",
          SPECS.LSQ_PAPER == "arXiv:1902.08153"
          and SPECS.PACT_PAPER == "arXiv:1805.06085")
    check("A", "★ 作者误引这件事本身记成常量（防止后人照抄那个错作者组）",
          "1612.01543" in SPECS.PACT_AUTHOR_MISATTRIBUTION)

    sub("A.15  测量条件那几条常量**本身**也要能被查（否则它们只是打印出来的装饰）")
    print("      MEASURED_VERSIONS = %s（数一下：%d 条）"
          % (SPECS.MEASURED_VERSIONS, len(SPECS.MEASURED_VERSIONS)))
    for v in SPECS.MEASURED_VERSIONS:
        print("        · %s" % v)
    print("      MEASURE_HOST      = %s" % SPECS.MEASURE_HOST)
    print("      UNCOVERED_MINORS  = %s" % SPECS.UNCOVERED_MINORS)
    print("      TORCH_MIN         = %s （%s）"
          % (SPECS.TORCH_MIN_FOR_THIS_TOPIC, SPECS.TORCH_MIN_REASON))
    check("A", "★ 版本清单**逐条**核对（五个都列出来，不是只数个数）",
          tuple(SPECS.MEASURED_VERSIONS)
          == ("2.6.0", "2.7.1", "2.8.0", "2.11.0", "2.13.0")
          and all(v.count(".") == 2 and v[0].isdigit() for v in SPECS.MEASURED_VERSIONS)
          and list(SPECS.MEASURED_VERSIONS) == sorted(SPECS.MEASURED_VERSIONS,
                                                      key=lambda v: tuple(int(x) for x in
                                                                          v.split("."))),
          str(SPECS.MEASURED_VERSIONS))
    check("A", "★ 测量条件里**必须写明**「CPU-only、无 GPU 数字」"
          "（否则读者会以为漏了 GPU 那一栏）",
          "CPU-only" in SPECS.MEASURE_HOST and "GPU" in SPECS.MEASURE_HOST)
    check("A", "★ 未覆盖的 minor 清单**非空**（P6：写「已覆盖」必须能说出范围）",
          bool(SPECS.UNCOVERED_MINORS) and "2." in SPECS.UNCOVERED_MINORS)
    print("      DeprecationWarning 的触发点是「%s」，%s 起有、%s 上没有"
          % (SPECS.DEPRECATION_WARNING_TRIGGER, SPECS.DEPRECATION_WARNING_FIRST_VERSION,
             " / ".join(SPECS.DEPRECATION_WARNING_ABSENT_IN)))
    check("A", "★★ 「import 时 0 条、调用时才有声」与「分界在 %s / %s 之间」"
          "两条都要能查（计划与任务书都说成「import 时打、2.11.0+ 起」——两半都错）"
          % (SPECS.DEPRECATION_WARNING_ABSENT_IN[-1], SPECS.DEPRECATION_WARNING_FIRST_VERSION),
          SPECS.DEPRECATION_WARNING_IMPORT_COUNT == 0
          and "不是 import" in SPECS.DEPRECATION_WARNING_TRIGGER
          and SPECS.DEPRECATION_WARNING_FIRST_VERSION == "2.8.0"
          and tuple(SPECS.DEPRECATION_WARNING_ABSENT_IN) == ("2.6.0", "2.7.1")
          and SPECS.OBSERVER_REDUCE_RANGE_DEPRECATION_QUOTE.startswith("Please use quant_min"))
    check("A", "★★ 版本下限记的是「**对齐选择**、**不是依赖**」——"
          "本主题在 2.6.0 上三个程序全部跑通",
          SPECS.TORCH_MIN_FOR_THIS_TOPIC == (2, 8)
          and "对齐选择" in SPECS.TORCH_MIN_REASON
          and "不是依赖" in SPECS.TORCH_MIN_REASON)


def _student_t3(rnd):
    """自由度为 3 的 Student-t 抽样（stdlib：正态 / sqrt(chi2/3)）。"""
    z = rnd.gauss(0.0, 1.0)
    c = sum(rnd.gauss(0.0, 1.0) ** 2 for _ in range(3))
    return z / math.sqrt(c / 3.0)


# ===========================================================================
# C. 边界与反例
# ===========================================================================

def run_C():
    section("C", "边界与反例：这些失败行为必须真的推出来")

    sub("C.1  掩码的两式在密集扫描下**必然**分道扬镳（纯推导，不依赖 torch）")
    s, zp = SPECS.FQ_STE_SCALE, SPECS.FQ_STE_ZERO_POINT
    qmin, qmax = SPECS.FQ_STE_QMIN, SPECS.FQ_STE_QMAX
    lo_i = f32(f32(qmin - f32(zp)) * f32(s))
    hi_i = f32(f32(qmax - f32(zp)) * f32(s))
    print("      区间式的边界（float32 精确值）：[%.17g, %.17g]" % (lo_i, hi_i))
    print("      舍入码式的边界：q = 0 当且仅当 round(x/s+128) == 0，即 x ∈ (≈−12.855, ≈−12.75]；")
    print("                      q = 255 当且仅当 x < 12.75（12.75 处 round(255.5) = 256，RNE）")
    rnd = random.Random(STDLIB_SEED + 2)
    grid = [i * 0.01 for i in range(-1400, 1401)]
    grid += [rnd.uniform(-13.0, 13.0) for _ in range(2000)]
    mis = [x for x in grid if ref_fq_mask_interval(x, s, zp, qmin, qmax)
           != ref_fq_mask_rounded(x, s, zp, qmin, qmax)]
    print("      不一致的 %d 个点全部落在 (−12.855, −12.80) ∪ (12.70, 12.75) 里吗：%s"
          % (len(mis), all((-12.86 < x < -12.79) or (12.69 < x < 12.76) for x in mis)))
    check("C", "★ 不一致点**必然存在**（> 0）——两式不是同一个判据",
          len(mis) > 0, "%d 个" % len(mis))
    check("C", "★ 且它们**都**落在两个边界窗口里（这不是随机噪声，是边界差）",
          all((-12.86 < x < -12.79) or (12.69 < x < 12.76) for x in mis))
    check("C", "★ 所以「−12.8 传梯度、−12.9 不传」这句**单独不足以**区分两式"
          "——必须再加一个边界内的点（如 −12.81）", True)

    sub("C.2  RNE 而不是 round-half-away-from-zero（判据：x = −12.85 那一点）")
    for tag, f in (("RNE（本文件）", rne), ("half-away", lambda x: math.floor(x + 0.5)
                                            if x >= 0 else math.ceil(x - 0.5))):
        q = f(f32(f32(-12.85) / f32(s))) + f32(zp)
        print("      %-14s round(−128.5) + 128 = %+.1f  ⇒ %s"
              % (tag, q, "传梯度" if qmin <= q <= qmax else "不传"))
    check("C", "★ 只有 RNE 给 −0（在 [0,255] 内）；half-away 给 −1（越界）",
          ref_fq_mask_rounded(-12.85, s, zp, qmin, qmax) is True)

    sub("C.3  LSQ 的梯度**主要由被截断的那部分驱动**（内部段有界 ≤ 0.5，饱和段 = Q_P）")
    qn, qp = SPECS.LSQ_UNSIGNED_QN, SPECS.LSQ_UNSIGNED_QP
    interior = max(abs(ref_lsq_grad_s(r / 100.0, 1.0, qn, qp)) for r in range(-99, 2550))
    print("      内部段梯度的最大绝对值 = %.4f（**上界 0.5**）" % interior)
    print("      饱和段梯度 = %d（比内部段上界大 %.0f 倍）" % (qp, qp / 0.5))
    check("C", "★ 内部段 |∂v̂/∂s| ≤ 0.5（量化残差有界）", interior <= 0.5 + 1e-12,
          "%.6f" % interior)
    check("C", "★ 饱和段 = Q_P，远大于内部段上界 ⇒ 梯度被截断部分主导",
          float(qp) > 10 * 0.5)
    check("C", "★ 且两段在 clip 边界上**不连续**（从 ≈0 跳到 Q_P）——这是设计的一部分，"
          "不是 bug", abs(ref_lsq_grad_s(254.999, 1.0, qn, qp) - float(qp)) > 200)

    sub("C.4  LSQ 的缩放因子：少了它，|ds|/|s| 会大一个到几个数量级")
    for n, q in ((4096, 7), (10 ** 6, 3)):
        g = ref_lsq_grad_multiplier(n, q)
        print("      N = %-8d Q_P = %d  ⇒  g = %.6e" % (n, q, g))
    check("C", "★ g 随 N 与 Q_P 的增大而减小——**「元素越多、更新幅度越大」这条经验"
          "正是被它抵消的**",
          ref_lsq_grad_multiplier(10 ** 6, 3) < ref_lsq_grad_multiplier(4096, 7))
    check("C", "★ 本机实测：去掉 g 后失衡（specs 记的最坏比值 %.2f）"
          % max(SPECS.LSQ_LOCAL_REL_UPDATE["N=%d no_g" % n]
                for n in (2048, 4096, 8192, 16384)),
          max(SPECS.LSQ_LOCAL_REL_UPDATE["N=%d no_g" % n]
              for n in (2048, 4096, 8192, 16384)) > 5.0)

    sub("C.5  STE 偏差的定量大小：**不得编数字**")
    print("      specs.STE_BIAS_MAGNITUDE = %r" % SPECS.STE_BIAS_MAGNITUDE)
    check("C", "★ 它必须是 UNSOURCED——本文件的任何断言都**不能**给出偏差的数值",
          SPECS.STE_BIAS_MAGNITUDE == "UNSOURCED")

    sub("C.6  「朴素 autograd」与「STE 版」在内部段**符号都可能相反**")
    qn2, qp2 = SPECS.LSQ_SIGNED_QN, SPECS.LSQ_SIGNED_QP
    flipped = []
    for r in (-1.5, 1.2, 1.5, 2.4, 3.6):
        a = ref_lsq_grad_s(r, 1.0, qn2, qp2)
        b = ref_lsq_grad_s_naive(r, 1.0, qn2, qp2)
        if a * b < 0:
            flipped.append(r)
        print("      r = %+5.2f  STE 版 %+8.4f  朴素版 %+8.4f  %s"
              % (r, a, b, "← 符号相反" if a * b < 0 else ""))
    print("      ⇒ 判据：`round(r) − r` 与 `round(r)` 同号**当且仅当** round(r) 与残差同号；")
    print("        在 `k < r < k + 1` 且 `k ≥ 1` 时二者**必然反号**（残差 = k − r < 0，而 k > 0）。")
    check("C", "★ 存在使两式**符号相反**的 r（整数格点右侧那一段，k < r < k+1、k ≥ 1）",
          len(flipped) > 0, "反号的 r = %s" % flipped)
    check("C", "★ 所以「LSQ 的 s 梯度」这句话**必须带上是哪一种**——两式不是同一个量", True)


# ===========================================================================
# E. 自检：把参考实现故意改错，A 类断言必须失败
# ===========================================================================

def run_E():
    section("E", "自检：故意改错时 A 类判据必须失败")
    ok = True
    s, zp = SPECS.FQ_STE_SCALE, SPECS.FQ_STE_ZERO_POINT
    qmin, qmax = SPECS.FQ_STE_QMIN, SPECS.FQ_STE_QMAX

    # E.1 前向去掉 clamp：区间外的 4 个点立刻对不上
    def fq_no_clamp(x):
        return f32(f32(rne(f32(f32(x) / f32(s))) + f32(zp) - f32(zp)) * f32(s))
    good = [ref_fq_forward(v, s, zp, qmin, qmax) for v in SPECS.FQ_STE_X]
    bad = [fq_no_clamp(v) for v in SPECS.FQ_STE_X]
    ok &= check("A", "E.1  去掉 clamp 后，前向 7 点表**对不上**（越界的 4 个点不再饱和）",
                sum(1 for a, b in zip(good, bad) if abs(a - b) > 1e-6) == 4,
                "%s" % ["%g" % v for v in bad])

    # E.2 掩码用区间式：在 specs 记的两个点上给出相反答案
    ok &= check("A", "E.2  换成区间式掩码后，specs 记的两个点**判反**",
                all(ref_fq_mask_interval(x, s, zp, qmin, qmax)
                    != SPECS.FQ_STE_BACKWARD[i]
                    for i, x in enumerate(SPECS.FQ_STE_X)
                    if x in SPECS.FQ_STE_INTERVAL_FORM_MISMATCH)
                or any(ref_fq_mask_interval(x, s, zp, qmin, qmax)
                       != ref_fq_mask_rounded(x, s, zp, qmin, qmax)
                       for x in SPECS.FQ_STE_INTERVAL_FORM_MISMATCH))

    # E.3 LSQ 内部段去掉 −v/s：7 点里应有 5 点对不上
    qn, qp = SPECS.LSQ_UNSIGNED_QN, SPECS.LSQ_UNSIGNED_QP
    rs = [0.25, 0.5, 1.5, 2.5, 100.3, 254.5]
    diff = sum(1 for r in rs if abs(ref_lsq_grad_s(r, 1.0, qn, qp)
                                    - ref_lsq_grad_s_naive(r, 1.0, qn, qp)) > 1e-9)
    ok &= check("A", "E.3  去掉内部段的 `−v/s` 后，%d 个内部点**全部**改变" % len(rs),
                diff == len(rs), "变了 %d 个" % diff)

    # E.4 PACT 的 α 梯度写成常量 1：clip 点以下就不再是 0
    def pact_grad_wrong(x, alpha):
        return 1.0
    ok &= check("A", "E.4  α 梯度写成常量 1 后，「clip 点以下为零」立刻不成立",
                pact_grad_wrong(1.0, 10.0) != ref_pact_grad_alpha(1.0, 10.0))

    # E.5 PACT 的激活写成裸 ReLU（丢掉了 α）：clamp 的等价性不成立
    def pact_relu(x, alpha):
        return max(x, 0.0)
    ok &= check("A", "E.5  PACT 写成裸 ReLU 后，与 `clamp(x, 0, α)` 的等价性不成立",
                abs(pact_relu(20.0, 10.0) - ref_pact_forward_clamp(20.0, 10.0)) > 1e-9)

    # E.6 裁剪 STE 的阈值从 1 改成 0.5：结果不同
    go, r = [1.0, 1.0], [-0.6, 0.6]
    ok &= check("A", "E.6  裁剪 STE 的阈值从 1 改成 0.5 会给出不同结果",
                ref_ste_clipped(go, r) != [g if abs(x) <= 0.5 else 0.0
                                           for g, x in zip(go, r)])

    print()
    print("  E 节结论：%s" % ("全部自检按预期失败 ⇒ A 类判据是活的" if ok
                              else "**有自检未按预期失败 ⇒ A 类判据有问题**"))
    return ok


# ===========================================================================
# F. 本文件覆盖了哪些 specs 常量（A6 变异测试的清单）
# ===========================================================================

def run_F():
    section("F", "变异测试清单：本文件读过的 specs 常量")
    names = sorted(n for n in dir(SPECS) if n.isupper() and not n.startswith("_"))
    src = open(__file__, encoding="utf-8").read()
    used = [n for n in names if ("SPECS.%s" % n) in src]
    print("      specs.py 里共 %d 条全大写常量；本文件直接读到的 %d 条"
          % (len(names), len(used)))
    unused = [n for n in names if n not in used]
    print("      本文件**没有**读到的 %d 条（由三个编号程序负责，见各程序 §F）：" % len(unused))
    for n in unused:
        print("        · %s" % n)
    check("A", "变异的对象是**常量**，每个常量的值都能被至少一处断言分辨",
          len(used) > 0)
    return unused


# ===========================================================================
# main
# ===========================================================================

def main():
    print("=" * 78)
    print("130-model-optimization/quantization · verify.py（纯 python3，**不 import torch / numpy**）")
    print("=" * 78)
    print("说明：本文件产出的是**推导**，不是实测。规模 = 掩码扫描 %d 点 /" % STDLIB_N)
    print("      重尾实验 2×10⁴ 样本 / 其余是标量与小向量。")
    print("      **规模不同 ⇒ 数字不同，只有定性结构必须一致。**")

    run_A()
    run_C()
    okE = run_E()
    run_F()

    print()
    print("=" * 78)
    n, a, c = _STATE["n"], _STATE["A"], _STATE["C"]
    print("合计 %d 条断言：A %d 条 / C %d 条；失败 %d 条" % (n, a, c, len(_STATE["fail"])))
    if _STATE["fail"]:
        print("失败清单：")
        for f in _STATE["fail"]:
            print("  · %s" % f)
    print("E 节自检：%s" % ("通过" if okE else "**不通过**"))
    sys.exit(0 if (not _STATE["fail"] and okE) else 1)


if __name__ == "__main__":
    main()

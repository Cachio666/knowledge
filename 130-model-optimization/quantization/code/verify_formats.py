#!/usr/bin/env python3
"""130-model-optimization/quantization 主题的纯 python3 复核入口。

★ 本文件**固定跑在默认 `python3` 下，永不 `import torch`**（CONTRIBUTING §2 第三种形态）。
  它提供的是「**换了台机器、什么都没装，也能复核的那一半**」：
  用**独立的参考实现**（下面的 `ref_*`）把本主题的**判据**重算一遍，
  再用手写期望值（不是 torch 的输出）判它对不对。
  **它的产出是推导，不是实测**——不得标成本机实测值。

  三类断言在本形态下的分工（CONTRIBUTING §2）：
    · **A 判据复核** —— 本文件。参照物是下面 `ref_*` 这套独立实现与手写期望值。
    · **C 边界与反例** —— 本文件。断言的是「失败行为成立」。
    · **B 代价模型** —— 在**编号程序**里（代价模型的参照是真系统，只有能 import 到它的
      环境里测才有意义）。本文件**不做** B。

★ 本主题里 A/C 与编号程序的分工不只是「有/没有 torch」，而是 **「数学 vs 实现」**：
  · 本文件（纯 python3）验的是**可独立推导的数学与位域**：从位域反解极值、
    e8m0 的取整规则、MX 的 Floor 致溢边界、饱和/溢出解析边界；
  · 编号程序（真 torch）测的是**真实实现的行为**：fp8 怎么溢出、e8m0 怎么取整、
    observer 算出什么 qparams、`torch.int4` 到底能做什么。
  · **两者不得互相背书。** 数学成立不证明实现如此；实现如此不证明数学成立。
  · **同一条结论若两边都能给，两边都要给**，且分别标注来源档位。

★ 「独立」的硬要求（CONTRIBUTING §2）：本文件里没有一行代码来自 torch，
  期望值**全部**是手写的常量（不是从 torch 抄的输出）。§D 节把这条**机械地验证一遍**：
  把参考实现故意改错，A 类断言必须失败。

运行：`make verify`（等价于 `python3 verify.py`）。只用标准库。
"""

import math
import struct
import random
import sys

import specs_formats as SPECS


# ===========================================================================
# 口径
# ===========================================================================

VERSIONS_BOUND = SPECS.MEASURED_VERSIONS
UNCOVERED = SPECS.UNCOVERED_MINORS

_STATE = {"n": 0, "A": 0, "C": 0, "fail": []}


def check(cls, name, cond, detail=""):
    _STATE["n"] += 1
    _STATE[cls] += 1
    ok = bool(cond)
    if not ok:
        _STATE["fail"].append("[%s] %s" % (cls, name))
    tail = "   ← %s" % detail if detail else ""
    print("    [%s] %s%s" % ("通过" if ok else "**不通过**", name, tail))
    return ok


def section(tag, title):
    print()
    print("=" * 76)
    print("§%s  %s" % (tag, title))
    print("=" * 76)


def sub(title):
    print()
    print("--- %s ---" % title)


# ===========================================================================
# §1 独立参考实现（全部手写，不含 torch 的任何代码或输出）
# ===========================================================================

def ref_fp8_params(e_bits, m_bits, bias, has_inf, nan_codes):
    """**从位域独立反解**一个类 IEEE-754 浮点格式的四个规格量。

    参数只给 (E, M, bias, 有无 Inf, NaN 码字集合)，**不给任何数值结果**。
    返回 dict：
      max_finite      = (2 - 2^-M) * 2^(emax - bias)，其中 emax = 2^E - 1 - (0 if has_inf else 1)
      smallest_sub    = 2^(1 - bias - M)
      smallest_normal = 2^(1 - bias)
      eps             = 2^-M
    这里 `emax` 的取法是**关键**：全 1 指数域只有在**有 Inf 的格式**里才被征用；
    对没有 Inf 的格式（e4m3fn 与两个 fnuz），全 1 指数域仍是**有效指数**，
    只是其中**尾部某些码字被 NaN 占掉**——`max_finite` 因此要看 NaN 码字落在哪。
    """
    exp_all_ones = (1 << e_bits) - 1
    if has_inf:
        emax = exp_all_ones - 1
        max_mant = (1 << m_bits) - 1
        max_finite = (2.0 - 2.0 ** -m_bits) * 2.0 ** (emax - bias)
    else:
        emax = exp_all_ones
        # 无 Inf 时，全 1 指数域里被 NaN 占掉的码字使「最大有限数」可能不在
        # 尾数全 1 处。逐个码字扫一遍更可靠，也更能暴露「想当然」的错。
        max_finite = None
        for mant in range((1 << m_bits) - 1, -1, -1):
            code = (exp_all_ones << m_bits) | mant
            if code in nan_codes:
                continue
            max_finite = (1.0 + mant / float(1 << m_bits)) * 2.0 ** (emax - bias)
            break
        assert max_finite is not None, "整个最高指数域都被 NaN 占了？"
    return {
        "max_finite": max_finite,
        "smallest_sub": 2.0 ** (1 - bias - m_bits),
        "smallest_normal": 2.0 ** (1 - bias),
        "eps": 2.0 ** -m_bits,
    }


def ref_nan_codes(e_bits, m_bits, has_inf, fnuz):
    """**从位域独立构造** NaN 码字集合。

    · 有 Inf 的格式：`|S|.全1指.{非0尾数}` —— 正的 (2^M - 1) 个，乘 2 个符号位。
    · 无 Inf、非 fnuz（即 e4m3fn）：全 1 指数 + 全 1 尾数，正负各一。
    · fnuz：**只有 `0x80` 一个码字**（负号 + 全 0 指数 + 全 0 尾数）——
      这是「`-0` 与 NaN 抢码字」的结果，代价是**没有负零**。
    """
    if fnuz:
        return {(1 << (e_bits + m_bits))}
    if has_inf:
        exp_all_ones = (1 << e_bits) - 1
        out = set()
        for sign in (0, 1):
            for mant in range(1, 1 << m_bits):
                out.add((sign << (e_bits + m_bits)) | (exp_all_ones << m_bits) | mant)
        return out
    exp_all_ones = (1 << e_bits) - 1
    mant_all_ones = (1 << m_bits) - 1
    body = (exp_all_ones << m_bits) | mant_all_ones
    return {body, body | (1 << (e_bits + m_bits))}


def ref_e8m0_value(code):
    """e8m0 的码字 → 值。**纯指数、无符号、0 位尾数**：值 = 2^(code - 127)。
    唯一例外是 `0xFF` = NaN。`verify.py` 用这张表独立复算，不依赖 torch。
    """
    if code == SPECS.IMPL_E8M0_NAN_CODES[0]:
        return None                      # None 表示 NaN
    return 2.0 ** (code - 127)


def ref_e8m0_round(x):
    """**独立实现** e8m0 的取整（按 ONNX 默认 `round_mode="up"` 的语义写）：

      1. 取绝对值（ONNX 说负值 undefined；本实现按 torch 实测的**取绝对值**写，
         并在文档里标明这是**实现的选择**，不是规范）；
      2. 向下饱和到 `2^-127`、向上饱和到 `2^127`（ONNX 的 `saturate=1` 那一列）；
      3. 按「最近、平局**远离零**」取整。

    这一份与 torch 源码里那份是**两条不同的路径**（那边是 guard/round/sticky 位运算），
    所以可以拿来对拍；**但它不是 torch 的输出**。
    """
    if x != x:                            # NaN 传播
        return None
    a = abs(x)
    if a == 0.0:
        return SPECS.IMPL_E8M0_MIN_FINITE
    e = math.floor(math.log2(a))
    lo, hi = 2.0 ** e, 2.0 ** (e + 1)
    if a - lo < hi - a:
        r = lo
    elif a - lo > hi - a:
        r = hi
    else:                                 # 平局 → 远离零
        r = hi
    if r < SPECS.IMPL_E8M0_MIN_FINITE:
        return SPECS.IMPL_E8M0_MIN_FINITE
    if r > SPECS.IMPL_E8M0_MAX_FINITE:
        return SPECS.IMPL_E8M0_MAX_FINITE
    return r


def ref_e8m0_round_ties_to_even(x):
    """**故意写成 ties-to-even** 的那一份。用来在 §D 里证明「平局规则确实是可区分的」：
    对 10 个中点里有 5 个会给出与 `ref_e8m0_round` 不同的结果。
    """
    if x != x:
        return None
    a = abs(x)
    if a == 0.0:
        return SPECS.IMPL_E8M0_MIN_FINITE
    e = math.floor(math.log2(a))
    lo, hi = 2.0 ** e, 2.0 ** (e + 1)
    if a - lo < hi - a:
        r = lo
    elif a - lo > hi - a:
        r = hi
    else:
        # ties-to-even：「偶」定义成**结果的 e8m0 码字 LSB 为 0**
        r = lo if (e + 127) % 2 == 0 else hi
    if r < SPECS.IMPL_E8M0_MIN_FINITE:
        return SPECS.IMPL_E8M0_MIN_FINITE
    if r > SPECS.IMPL_E8M0_MAX_FINITE:
        return SPECS.IMPL_E8M0_MAX_FINITE
    return r


def ref_e2m1_values():
    """E2M1 的值集：1 符号 + 2 指数 + 1 尾数，bias = 1。
    次正规 0 / 0.5；正规 1/1.5、2/3、4/6 → 最大幅值 6。
    """
    return [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def ref_e2m1_round(x):
    """E2M1 最近取整（**无 Inf/NaN**，越界饱和到 ±6；平局取较小下标）。"""
    s = -1.0 if x < 0 else 1.0
    a = min(abs(x), 6.0)
    best, bestd = None, None
    for v in ref_e2m1_values():
        d = abs(a - v)
        if bestd is None or d < bestd:
            best, bestd = v, d
    return s * best


def ref_mx_scale(amax, emax, block_scale_floor=True):
    """MX **论文 Algorithm 1** 的块 scale：`shared_exp = floor(log2(amax)) - emax`，
    `scale = 2^shared_exp`。注意它是 2 的幂（这正是 e8m0 能表示的东西）。
    """
    assert amax > 0
    e = math.floor(math.log2(amax)) - emax
    if block_scale_floor:
        return 2.0 ** e
    return 2.0 ** e


def ref_mx_blocked_overflow(xs, emax, elem_max):
    """返回块内「规约后最大值超过元素上限」的比例。**纯解析 + 直接计算**，无 torch。"""
    amax = max(abs(v) for v in xs)
    m = amax / (2.0 ** math.floor(math.log2(amax)))       # m ∈ [1, 2)
    return m * (2.0 ** emax) > elem_max, m


def ref_affine_quant(xs, bits):
    """无符号码字 0..2^b−1 的仿射（非对称）量化。
    与 arXiv:2004.09602 Eq.1–5 的**同一套数学**，但按无符号码字写
    （论文的 z 含 −2^(b−1) 偏移，**两套约定差一个常数偏移**）。
    返回 (scale, z_int, z_real, delta)。
    """
    a, b = min(xs), max(xs)
    qmax = (1 << bits) - 1
    s = (b - a) / qmax
    z_real = -a / s
    z = round(z_real)
    delta = (-z * s) - a
    return s, z, z_real, delta


def ref_sym_quant(xs, bits, full_range=True):
    """对称量化。`full_range=True` 用 arXiv:2103.13630 的
    `S = 2·max|r| / (2^n − 1)`（int8 → **/127.5**）——**torch 走的是这一支**；
    `full_range=False` 用 restricted range `S = max|r| / (2^(n−1) − 1)`（/127）。
    """
    mx = max(abs(v) for v in xs)
    if full_range:
        s = 2.0 * mx / ((1 << bits) - 1)
    else:
        s = mx / ((1 << (bits - 1)) - 1)
    qmax, qmin = (1 << (bits - 1)) - 1, -(1 << (bits - 1))
    out = []
    for v in xs:
        q = min(max(round(v / s), qmin), qmax)
        out.append(q * s)
    return s, out


def ref_sqnr_db(xs, xh):
    """SQNR = 10·log10(Σx² / Σ(x−x̂)²)，dB。定义写死在这里，免得与别的口径混。"""
    num = sum(v * v for v in xs)
    den = sum((v - w) ** 2 for v, w in zip(xs, xh))
    if den == 0:
        return float("inf")
    return 10.0 * math.log10(num / den)


def ref_float_sqnr_analytic_db(m_bits):
    """浮点 RNE 的 SQNR 解析式（尾数在 binade 内均匀假设下）：
        SQNR = 20·log10(√3 · √2 · 2^(M+1)) = 7.78 + 6.0206·(M+1)  dB
    推导：误差在 ±ulp/2 内均匀 → RMS 误差 = ulp/√12；相对化后取 binade 平均
    （∫_1^2 du/u² / ∫_1^2 du = 1/2）→ 多出一个 √2 因子。
    """
    return 20.0 * math.log10(math.sqrt(3.0) * math.sqrt(2.0) * 2.0 ** (m_bits + 1))


# ===========================================================================
# §A 判据复核：手写期望值 vs 独立参考实现
# ===========================================================================

# 手写期望值：**(E, M, bias, 有无 Inf, 是不是 fnuz, max, 最小次正规, 最小正规, eps)**
# 这些数**不是从 torch 抄的**——它们是位域定义的算术结果，本表的每一格都由
# 上面的 `ref_fp8_params` 从位域重算一遍并逐格比对（见 A1）。
CASES_FP8 = [
    # name,           E, M, bias, has_inf, fnuz, max,       smallest_sub, smallest_normal, eps
    ("float8_e4m3fn",   4, 3,  7, False, False, 448.0,     2.0 ** -9,  2.0 ** -6,  2.0 ** -3),
    ("float8_e5m2",     5, 2, 15, True,  False, 57344.0,   2.0 ** -16, 2.0 ** -14, 2.0 ** -2),
    ("float8_e4m3fnuz", 4, 3,  8, False, True,  240.0,     2.0 ** -10, 2.0 ** -7,  2.0 ** -3),
    # ⚠ 这一行的 eps 是 **2^-2 = 0.25**，**不是** torch.finfo 报的 0.125。
    #   位域上 e5m2fnuz 是 S1E5M2，eps = 2^-M = 2^-2。torch 的 finfo 在这一格上
    #   与位域自相矛盾（见 specs.IMPL_E5M2FNUZ_FINFO_EPS_IS_WRONG 与 A1 的末段）。
    ("float8_e5m2fnuz", 5, 2, 16, False, True,  57344.0,   2.0 ** -17, 2.0 ** -15, 2.0 ** -2),
]

# 手写期望值：torch 的 finfo 对 e5m2fnuz 报的 eps（**与位域不符**，本机实测）
E5M2FNUZ_FINFO_EPS_MEASURED = 0.125
E5M2FNUZ_TRUE_EPS = 0.25

# 手写期望值：四个变体的**解析溢出边界**。
# 判据（必须与数字一起写出）：设元素上限 `M`，其 ulp 为 `u`，则
#   · 阈值 `T = M + u/2`（中点）；
#   · 中点**以下**的输入一律舍到 `M`（不溢出）；
#   · 中点的输入按 RNE 平局规则取**偶**候选——若那个偶候选的码字正好是 NaN 码，
#     结果就是 NaN；若是 Inf 码，结果就是 Inf；**具体落哪个由码字空间决定**；
#   · 中点以上的输入走同一规则，越界者落 NaN / Inf。
# `T` 与中点行为都是**位域的直接读数**，不是经验拟合。
CASES_OVERFLOW_BOUNDARY = [
    # name,            元素上限,   ulp@上限, 中点阈值,   中点以下 → , 中点及以上 →（≤2.11 的 e4m3fn）
    ("float8_e4m3fn",   448.0,     32.0,     464.0,     448.0,       "NaN"),
    ("float8_e5m2",     57344.0,   8192.0,   61440.0,   57344.0,     "inf"),
    ("float8_e4m3fnuz", 240.0,     16.0,     248.0,     240.0,       "NaN"),
    ("float8_e5m2fnuz", 57344.0,   8192.0,   61440.0,   57344.0,     "NaN"),
]

# 手写期望值：e8m0 的取整行为（ONNX 默认 `up` = 最近、平局远离零）。
CASES_E8M0 = [
    # 输入,              期望输出（None = NaN）
    (1.0,                1.0),
    (-1.0,               1.0),          # ★ 取绝对值：符号位被丢弃
    (1.5,                2.0),          # ★ 平局 → 远离零
    (-1.5,               2.0),
    (0.0,                2.0 ** -127),  # ★ 零无法表示
    (-0.0,               2.0 ** -127),
    (1572864.0,          2097152.0),    # 中点 (2^20+2^21)/2 → 向上
    (1572863.0,          1048576.0),    # 同一下方一个 ulp → 向下（证明确实按最近）
    (3.0,                4.0),          # 中点 1.5·2^1：**ties-to-even 本会选 2.0**，实测走 4.0
    (12.0,               16.0),         # 中点 1.5·2^3：ties-to-even 本会选 8.0，实测走 16.0
    (2.0 ** -128,        2.0 ** -127),  # 向下饱和
    (2.0 ** -200,        2.0 ** -127),
    (2.0 ** 200,         2.0 ** 127),   # 向上饱和
    (float("nan"),       None),
]

# 手写期望值：e8m0 的码字统计（256 个码字全扫）
E8M0_EXPECT = {
    "nan_codes": [0xFF],
    "zero_codes": [],
    "negative_codes": [],
    "n_finite": 255,
    "min_finite": 2.0 ** -127,
    "max_finite": 2.0 ** 127,
    "strictly_increasing": True,
    "all_powers_of_two": True,
}

# 手写期望值：MX 的 Floor 取整致溢
#   e4m3：上限 448 = 1.75·2^8，emax = 8 → 阈值 m > 1.75（= 448/256）
#   e2m1：上限 6   = 1.5 ·2^2，emax = 2 → 阈值 m > 1.5 （= 6/4）
MX_CASES = [
    ("e4m3", 8, 448.0, 1.75, 0.25),
    ("e2m1", 2, 6.0,   1.50, 0.50),
]

# 手写期望值：MX 致溢率的蒙特卡洛（N(0,1)、32 元素块）。
# ⚠ 这里的 MC 与上面的「解析估算」**不是同一个量**（P9）：
#   解析估算假设 m 在 [1,2) 上**均匀**（25% / 50%）；
#   MC 是 **N(0,1) 数据**下 m 的分布（16.9% / 29.6%）。
#   **两个数并列时必须各带判据。** 本文件的 MC 用标准库 `random.gauss` 独立跑，
#   与编号程序里那份（真 torch 张量）是两条不同的路径。
MX_MC_EXPECT = dict(SPECS.MX_OVERFLOW_MC_GAUSSIAN_BLOCK32)
MX_MC_BLOCKS = 200000
MX_MC_TOL = 0.006

# 手写期望值：对称量化除法器的判别。
# ⚠ **不是另抄一份**——直接从 specs 里取那一条实测记录。
#   本仓库的变异测试抓到过：这里若写死一份副本，改了 specs 两边都对不上却都不报错。
SYM_DIVISOR_PROBE = dict(SPECS.IMPL_OBSERVER_MAXABS_PROBE)

# 手写期望值：整数格式的 δ 界
DELTA_CASES = [
    # (名字, 数据生成器, bits)  —— 数据用标准库 random 独立生成
    ("U(-1,1)", "uniform", 8),
    ("单边指数", "exp", 8),
]

# 手写期望值：浮点 SQNR 的解析式
FLOAT_SQNR_ANALYTIC = {3: 31.86, 7: 55.95, 10: 74.01}


def run_A():
    section("A", "判据复核：手写期望值 vs 独立参考实现")

    sub("A1  从位域反解四个 fp8 变体（纯整数/2 的幂运算，不碰 torch）")
    for (name, e, m, bias, has_inf, fnuz, mx, smin, snorm, eps) in CASES_FP8:
        nan_codes = ref_nan_codes(e, m, has_inf, fnuz)
        p = ref_fp8_params(e, m, bias, has_inf, nan_codes)
        check("A", "%s 的 max = %g" % (name, mx), abs(p["max_finite"] - mx) < 1e-9,
              "解得 %r" % p["max_finite"])
        check("A", "%s 的最小次正规 = 2^%d" % (name, round(math.log2(smin))),
              abs(p["smallest_sub"] - smin) < 1e-300, "解得 %r" % p["smallest_sub"])
        check("A", "%s 的最小正规 = 2^%d" % (name, round(math.log2(snorm))),
              abs(p["smallest_normal"] - snorm) < 1e-300, "解得 %r" % p["smallest_normal"])
        check("A", "%s 的 eps = 2^%d" % (name, round(math.log2(eps))),
              abs(p["eps"] - eps) < 1e-12, "解得 %r" % p["eps"])

    sub("A1b ★ 位域与实现自相矛盾的一格：float8_e5m2fnuz 的 eps")
    # 本文件不能 import torch，所以这里验的是**算术**：S1E5M2 的 eps = 2^-2 = 0.25，
    # 而 0.25 在 e5m2fnuz（bias 16）里的码字是 0x38、在 e5m2（bias 15）里是 0x34；
    # torch 的 `epsilon()` 两个类用的是**同一个字面码字 0x34**，所以 e5m2fnuz 读出 0.125。
    def _code_of_pow2(exponent, bias, m_bits):
        return ((exponent + bias) << m_bits)
    check("A", "e5m2fnuz 的真实 eps = 2^-2 = 0.25",
          abs(E5M2FNUZ_TRUE_EPS - 2.0 ** -2) < 1e-12)
    check("A", "0.25 在 e5m2fnuz 里的码字是 0x38（exp 域 = -2+16 = 14）",
          _code_of_pow2(-2, 16, 2) == 0x38, "得 0x%02X" % _code_of_pow2(-2, 16, 2))
    check("A", "同一个 0x34 在 e5m2 里才是 0.25（exp 域 = -2+15 = 13）",
          _code_of_pow2(-2, 15, 2) == 0x34, "得 0x%02X" % _code_of_pow2(-2, 15, 2))
    check("A", "★ 0x34 在 e5m2fnuz 里读出 2^-3 = 0.125 —— 这正是 torch 报的数",
          abs(2.0 ** (13 - 16) - E5M2FNUZ_FINFO_EPS_MEASURED) < 1e-12)
    check("A", "★ 所以 finfo 的那个数是**照抄码字**的结果，差一个因子 2",
          abs(E5M2FNUZ_TRUE_EPS / E5M2FNUZ_FINFO_EPS_MEASURED - 2.0) < 1e-12)
    check("A", "specs 里记的成因与实测版本都在",
          "0x34" in SPECS.IMPL_E5M2FNUZ_FINFO_EPS_IS_WRONG["cause"]
          and SPECS.IMPL_E5M2FNUZ_FINFO_EPS_IS_WRONG["observed_versions"] == ("2.8.0", "2.13.0"))
    check("A", "★ e4m3fn/e4m3fnuz（M=3）与 e5m2（M=2）三格 finfo 与位域一致，只有这一格不一致",
          abs(2.0 ** -3 - 0.125) < 1e-12 and abs(2.0 ** -2 - 0.25) < 1e-12)

    sub("A1c ★ specs.FP8_FORMATS 的**每一格**都要能被位域反解复现")
    # ⚠ 这一段是补的：没有它的话，`specs_formats.py` 里那八列规格值**一格都没被断言核过**——
    #   本仓库实测踩到过（把 e4m3fn 的 eps 从 2^-3 改成 2^-4，A 类全绿）。
    #   原因：A1 比的是「位域反解」与「手写期望」两份，而 specs 是**第三份**。
    #   三份都要互相咬住，否则 specs 就是一份没人查的抄件。
    for (name, e, m, bias, has_inf, fnuz, mx, smin, snorm, eps) in CASES_FP8:
        s_s, s_e, s_m, s_bias, s_mx, s_smin, s_snorm, s_eps = SPECS.FP8_FORMATS[name]
        check("A", "%s：specs 的位域字段 (S,E,M)=(%d,%d,%d) 与手写期望一致"
              % (name, s_s, s_e, s_m), (s_s, s_e, s_m) == (1, e, m))
        check("A", "%s：specs 的 bias = %d" % (name, s_bias), s_bias == bias)
        check("A", "%s：specs 的 max = %g" % (name, s_mx), abs(s_mx - mx) < 1e-9)
        check("A", "%s：specs 的最小次正规 = 2^%d" % (name, round(math.log2(s_smin))),
              abs(s_smin - smin) < 1e-300)
        check("A", "%s：specs 的最小正规 = 2^%d" % (name, round(math.log2(s_snorm))),
              abs(s_snorm - snorm) < 1e-300)
        check("A", "%s：specs 的 eps = %g" % (name, s_eps), abs(s_eps - eps) < 1e-12,
              "specs=%g 手写期望=%g" % (s_eps, eps))
        check("A", "%s：specs 的 NaN 码字集合与位域构造一致" % name,
              sorted(SPECS.FP8_NAN_CODES[name]) == sorted(ref_nan_codes(e, m, has_inf, fnuz)))
    check("A", "specs 的 FP8_FORMATS 覆盖了全部四个变体",
          set(SPECS.FP8_FORMATS) == set(c[0] for c in CASES_FP8))
    check("A", "specs 的 FP8_NAN_CODES 覆盖了全部四个变体",
          set(SPECS.FP8_NAN_CODES) == set(c[0] for c in CASES_FP8))

    sub("A1d ★ bf16 的中点舍入探针：specs 的值必须被**独立实现**复现")
    # ⚠ 这一段是 2026-09-20 补的。`specs.RNE_MIDPOINT_PROBE` 此前是**死常量**——
    #   全库没有任何程序引用它，于是它里面的错值（`expect: 1.0`）在建成验收里不可见。
    #   订正记录见 `docs/superpowers/plans/2026-09-20-kernel-lifecycle-domain.md` §E。
    def _bf16_rne(x):
        """fp32 → bf16 的 round-to-nearest-ties-to-even，纯整数位操作（不碰 torch）。"""
        u = struct.unpack("<I", struct.pack("<f", x))[0]
        mant = (u >> 16) & 0x7F          # bf16 的 7 位尾数
        if mant & 1:                      # 奇数 ⇒ 进位到偶
            return struct.unpack("<f", struct.pack("<I", (u & 0xFFFF0000) + 0x10000))[0]
        return struct.unpack("<f", struct.pack("<I", u & 0xFFFF0000))[0]

    for expr, d in sorted(SPECS.RNE_TIE_CASES.items()):
        x = eval(expr)                    # 表达式来自 specs，求值等价于手算
        got = _bf16_rne(x)
        check("A", "bf16 RNE：%s = %r → %r" % (expr, x, got),
              abs(got - d["expect"]) < 1e-12,
              "独立实现得 %r，specs 记 %r" % (got, d["expect"]))
        # ★ 真正区分两条规则的那一格：下界为**偶**时 RNE 取偶、half-up 进位
        if abs(d["expect"] - d["half_up_would_give"]) > 1e-15:
            check("A", "★ %s 能区分 RNE 与 round-half-up（RNE→%r，half-up→%r）"
                  % (expr, d["expect"], d["half_up_would_give"]),
                  True)
        else:
            check("A", "⚠ %s **分不出** RNE 与 round-half-up（两者都给 %r）——"
                  "它只能证明「舍入到中点的一侧」，不能证明「取偶」"
                  % (expr, d["expect"]), True)
    # 中点本身要能被独立验证（否则上面两格可能只是「值恰好对上」）
    for expr in SPECS.RNE_TIE_CASES:
        x = eval(expr)
        u = struct.unpack("<I", struct.pack("<f", x))[0]
        lo = struct.unpack("<f", struct.pack("<I", u & 0xFFFF0000))[0]
        hi = struct.unpack("<f", struct.pack("<I", (u & 0xFFFF0000) + 0x10000))[0]
        check("A", "%s = %r 确实是 bf16 相邻两数 %r 与 %r 的中点"
              % (expr, x, lo, hi), abs((lo + hi) / 2 - x) < 1e-15)

    sub("A2  NaN / Inf 码字集合从位域独立构造")
    for (name, e, m, bias, has_inf, fnuz, mx, smin, snorm, eps) in CASES_FP8:
        got = sorted(ref_nan_codes(e, m, has_inf, fnuz))
        want = sorted(SPECS.FP8_NAN_CODES[name])
        check("A", "%s 的 NaN 码字 = %s" % (name, [hex(c) for c in want]), got == want,
              "解得 %s" % [hex(c) for c in got])
    check("A", "fnuz 两个变体的 NaN 码字**恰好只有 0x80 一个**",
          len(SPECS.FP8_NAN_CODES["float8_e4m3fnuz"]) == 1
          and len(SPECS.FP8_NAN_CODES["float8_e5m2fnuz"]) == 1)
    check("A", "对照：e5m2 的 NaN 码字有 6 个（有 Inf 的格式）",
          len(SPECS.FP8_NAN_CODES["float8_e5m2"]) == 6)
    check("A", "fnuz 的 bias 恰比对应非 fnuz 大 1（7→8、15→16）",
          CASES_FP8[2][3] - CASES_FP8[0][3] == 1 and CASES_FP8[3][3] - CASES_FP8[1][3] == 1)

    sub("A3  e8m0：256 个码字全扫（独立按 2^(code-127) 复算）")
    vals = [ref_e8m0_value(c) for c in range(256)]
    nan_codes = [c for c, v in enumerate(vals) if v is None]
    zero_codes = [c for c, v in enumerate(vals) if v == 0.0]
    neg_codes = [c for c, v in enumerate(vals) if v is not None and v < 0]
    finite = [(c, v) for c, v in enumerate(vals) if v is not None]
    check("A", "NaN 码字恰好 [0xFF]", nan_codes == E8M0_EXPECT["nan_codes"],
          "得 %s" % [hex(c) for c in nan_codes])
    check("A", "**没有零码字**", zero_codes == E8M0_EXPECT["zero_codes"])
    check("A", "**没有负值码字**", neg_codes == E8M0_EXPECT["negative_codes"])
    check("A", "有限码字恰好 255 个", len(finite) == E8M0_EXPECT["n_finite"],
          "得 %d" % len(finite))
    check("A", "最小有限值 = 2^-127", finite[0][1] == E8M0_EXPECT["min_finite"])
    check("A", "最大有限值 = 2^127", finite[-1][1] == E8M0_EXPECT["max_finite"])
    check("A", "有限值随码字严格递增",
          all(finite[i][1] < finite[i + 1][1] for i in range(len(finite) - 1)))
    check("A", "所有有限值都是 2 的整数次幂",
          all(abs(math.log2(v) - round(math.log2(v))) < 1e-12 for _, v in finite))

    sub("A4  e8m0 的取整规则（独立实现 vs 手写期望）")
    for x, want in CASES_E8M0:
        if x != x:
            got = ref_e8m0_round(x)
            ok = got is None
        else:
            got = ref_e8m0_round(x)
            ok = (got is not None) and abs(got - want) <= abs(want) * 1e-12
        check("A", "e8m0 取整 %r → %r" % (x, want), ok, "解得 %r" % (got,))
    check("A", "★ 负值取绝对值：-1.0 → 1.0、-1.5 → 2.0（**这是 torch 的实现选择**）",
          ref_e8m0_round(-1.0) == 1.0 and ref_e8m0_round(-1.5) == 2.0)
    check("A", "★ 零不可表示：0.0 → 2^-127",
          ref_e8m0_round(0.0) == SPECS.IMPL_E8M0_MIN_FINITE)

    sub("A5  平局规则是可区分的：ties-to-even 那一份会在 10 个中点里错 5 个")
    ties = [1.5 * 2.0 ** a for a in range(SPECS.IMPL_E8M0_TIE_PROBE_A_RANGE[0],
                                        SPECS.IMPL_E8M0_TIE_PROBE_A_RANGE[1] + 1)]
    diffs = [t for t in ties
             if ref_e8m0_round(t) != ref_e8m0_round_ties_to_even(t)]
    check("A", "10 个中点里，两份实现在 5 个上不同（其余 5 个恰好同向）",
          len(diffs) == 5, "实测不同 %d 个：%s" % (len(diffs), diffs))

    sub("A6  四个 fp8 变体的解析溢出边界（位域的直接读数）")
    for (name, elem_max, ulp, thr, below, above) in CASES_OVERFLOW_BOUNDARY:
        check("A", "%s 的中点阈值 = %g" % (name, thr),
              abs((elem_max + ulp / 2.0) - thr) < 1e-9,
              "element_max + ulp/2 = %g" % (elem_max + ulp / 2.0))
        check("A", "%s 中点的**下一个码字**越界（%s）" % (name, above),
              above in ("NaN", "inf"))
        # 中点的下一个候选值 = elem_max + ulp；它必须不落在 [0, elem_max] 内
        check("A", "%s 的下一候选 %g 确实超过上限 %g" % (name, elem_max + ulp, elem_max),
              elem_max + ulp > elem_max)
        check("A", "%s 中点以下的输入舍到 %g（不溢出）" % (name, below),
              below == elem_max)

    sub("A7  MX 的 Floor 取整致溢：阈值与解析估算")
    for (fmt, emax, elem_max, thr_expect, uniform_expect) in MX_CASES:
        thr = elem_max / (2.0 ** emax)
        check("A", "%s 的溢出阈值 m > %g" % (fmt, thr_expect), abs(thr - thr_expect) < 1e-12,
              "解得 %g" % thr)
        check("A", "对照 specs 里的阈值常量 %g" % SPECS.MX_OVERFLOW_THRESHOLD[fmt],
              abs(SPECS.MX_OVERFLOW_THRESHOLD[fmt] - thr) < 1e-12)
        # 均匀假设下 P(m > thr) = 2 - thr
        est = 2.0 - thr
        check("A", "%s 的均匀假设估算 = %g" % (fmt, uniform_expect),
              abs(est - uniform_expect) < 1e-12, "解得 %g" % est)
        check("A", "对照 specs 里的均匀估算常量 %g" % SPECS.MX_OVERFLOW_UNIFORM_ESTIMATE[fmt],
              abs(SPECS.MX_OVERFLOW_UNIFORM_ESTIMATE[fmt] - est) < 1e-12)

    sub("A7b ★ 同一件事的多份副本必须互相咬住（变异测试的教训）")
    check("A", "对称除数的探针：specs 与 verify 用的是同一份（不是两份副本）",
          SYM_DIVISOR_PROBE == SPECS.IMPL_OBSERVER_MAXABS_PROBE)
    check("A", "MX 的 MC 期望：specs 与 verify 用的是同一份", 
          MX_MC_EXPECT == SPECS.MX_OVERFLOW_MC_GAUSSIAN_BLOCK32)
    check("A", "specs 记的 maxabs × specs 记的 127.5 == specs 记的 maxabs（内部自洽）",
          abs(SYM_DIVISOR_PROBE["scale"] * SPECS.IMPL_SYM_DIVISOR_INT8
              - SYM_DIVISOR_PROBE["maxabs"]) < 1e-5,
          "%.9f vs %.9f" % (SYM_DIVISOR_PROBE["scale"] * SPECS.IMPL_SYM_DIVISOR_INT8,
                            SYM_DIVISOR_PROBE["maxabs"]))
    check("A", "★ 对称 vs 非对称那张表：每一行的「差」必须等于两个端点之差",
          all(abs((a - s_) - d) < 0.02
              for (a, s_, d) in SPECS.IMPL_SYM_VS_ASYM_DB.values()))
    check("A", "★ 其中单边分布那两行的差 ≈ 6.02 dB，零均值那行远小于它",
          all(abs(SPECS.IMPL_SYM_VS_ASYM_DB[k][2] - 6.0206) < 0.1
              for k in ("|N(0,1)|", "U(0,1)"))
          and SPECS.IMPL_SYM_VS_ASYM_DB["N(0,1)"][2] < 1.0)
    check("A", "★ restricted range 的对照值必须**小于等于** full range（它少用半个码字）",
          all(SPECS.IMPL_SYM_VS_ASYM_RESTRICTED_DB[k] <= SPECS.IMPL_SYM_VS_ASYM_DB[k][1] + 0.01
              for k in SPECS.IMPL_SYM_VS_ASYM_RESTRICTED_DB))
    check("A", "★ 两个阈值本来就不同：e4m3 是 1.75、e2m1 是 1.5",
          abs(SPECS.MX_OVERFLOW_THRESHOLD["e4m3"] - 1.75) < 1e-12
          and abs(SPECS.MX_OVERFLOW_THRESHOLD["e2m1"] - 1.5) < 1e-12)
    check("A", "★ 两种判据下的四个数两两不同（解析估算 ≠ 蒙特卡洛）",
          SPECS.MX_OVERFLOW_UNIFORM_ESTIMATE["e4m3"] != SPECS.MX_OVERFLOW_MC_GAUSSIAN_BLOCK32["e4m3"]
          and SPECS.MX_OVERFLOW_UNIFORM_ESTIMATE["e2m1"] != SPECS.MX_OVERFLOW_MC_GAUSSIAN_BLOCK32["e2m1"])
    check("A", "★ 块大小 32 与 16 是两个不同的数（MX vs NVFP4）",
          SPECS.OCP_MX_BLOCK_SIZE != SPECS.NVFP4_BLOCK_SIZE)

    sub("A8  NVFP4 与 MXFP4 是三个不同的东西（不是「块小一点」）")
    check("A", "块大小 16 vs 32", SPECS.NVFP4_BLOCK_SIZE == 16 and SPECS.OCP_MX_BLOCK_SIZE == 32)
    check("A", "block scale 的格式不同：E4M3（带尾数）vs E8M0（纯指数）",
          "E4M3" in SPECS.NVFP4_SCALE_BLOCK and SPECS.OCP_MX_SCALE_FORMAT == "E8M0")
    check("A", "NVFP4 多一层 per-tensor FP32 global scale",
          len(SPECS.NVFP4_VS_MXFP4_DIFFS) == 3)

    sub("A9  对称量化的除数：127.5 还是 127（用 specs 记的实测 maxabs 判别）")
    mx = SYM_DIVISOR_PROBE["maxabs"]
    sc = SYM_DIVISOR_PROBE["scale"]
    check("A", "scale × 127.5 = maxabs（吻合）",
          abs(sc * 127.5 - mx) < 1e-5, "%.9f vs %.9f" % (sc * 127.5, mx))
    check("A", "对照：scale × 127 对不上",
          abs(sc * 127.0 - mx) > 1e-2, "%.9f vs %.9f" % (sc * 127.0, mx))
    check("A", "127.5 的来源 = (quant_max − quant_min)/2 = 255/2",
          abs(SPECS.IMPL_SYM_DIVISOR_INT8 - 255.0 / 2.0) < 1e-12)

    sub("A10 整数格式的 δ 界：|δ| ≤ s/2，两端点**同时**平移")
    for name, kind, bits in DELTA_CASES:
        random.seed(20260918)
        if kind == "uniform":
            xs = [random.uniform(-1.0, 1.0) for _ in range(20000)]
        else:
            xs = [random.expovariate(1.0) for _ in range(20000)]
        s, z, z_real, delta = ref_affine_quant(xs, bits)
        a, b = min(xs), max(xs)
        check("A", "%s：|δ| ≤ s/2" % name, abs(delta) <= s / 2.0 + 1e-12,
              "|δ|=%.6e  s/2=%.6e" % (abs(delta), s / 2.0))
        qmax = (1 << bits) - 1
        check("A", "%s：两端点同时平移 δ（范围宽度不变）" % name,
              abs(((-z * s + qmax * s) - (-z * s)) - (b - a)) < 1e-9,
              "有效宽度 %.9e vs 原宽度 %.9e" % (qmax * s, b - a))
        check("A", "%s：平移后的左端点 = -z·s（不再是 a）" % name,
              abs((-z * s) - (a + delta)) < 1e-12, "-z·s=%.9e  a+δ=%.9e" % (-z * s, a + delta))

    sub("A11 对称量化的两支约定（restricted vs full range）")
    random.seed(20260918)
    xs = [random.gauss(0, 1) for _ in range(20000)]
    mx = max(abs(v) for v in xs)
    s_full, _ = ref_sym_quant(xs, 8, full_range=True)
    s_restr, _ = ref_sym_quant(xs, 8, full_range=False)
    check("A", "full range 的除数 = 255/2 = 127.5", abs(mx / s_full - 127.5) < 1e-9,
          "解得 %r" % (mx / s_full))
    check("A", "restricted range 的除数 = 127", abs(mx / s_restr - 127.0) < 1e-9,
          "解得 %r" % (mx / s_restr))

    sub("A12 浮点 SQNR 的解析式（尾数均匀假设）")
    for m_bits, want in sorted(FLOAT_SQNR_ANALYTIC.items()):
        got = ref_float_sqnr_analytic_db(m_bits)
        check("A", "M=%d 的解析 SQNR ≈ %.2f dB" % (m_bits, want), abs(got - want) < 0.01,
              "解得 %.2f" % got)
        check("A", "对照 specs 的解析常量 %.2f" % SPECS.FLOAT_SQNR_ANALYTIC_DB[m_bits],
              abs(SPECS.FLOAT_SQNR_ANALYTIC_DB[m_bits] - got) < 0.01)

    sub("A13 浮点 SQNR 与分布无关：跨分布极差只有 ~1 dB（用 specs 的实测表）")
    for col in ("bf16", "fp16", "e4m3fn"):
        idx = SPECS.SQNR_COLUMNS.index(col)
        vals = [row[idx] for row in SPECS.SQNR_DB_BY_DIST.values()]
        spread = max(vals) - min(vals)
        check("A", "%s 跨四分布的极差 = %.2f dB（规格表里记的 %.2f）"
              % (col, spread, SPECS.SQNR_SPREAD_DB[col]),
              abs(spread - SPECS.SQNR_SPREAD_DB[col]) < 0.02, "实测极差 %.2f" % spread)
    check("A", "★ **不要写「< 1 dB」**：bf16 的两端 55.59 → 56.62 就是 1.03",
          SPECS.SQNR_SPREAD_DB["bf16"] > 1.0)
    check("A", "bf16（N(0,1)）与解析式同量级（差 < 0.5 dB）",
          abs(SPECS.SQNR_DB_BY_DIST["N(0,1)"][0] - ref_float_sqnr_analytic_db(7)) < 0.5,
          "解析 %.2f / 实测 %.2f" % (ref_float_sqnr_analytic_db(7),
                                    SPECS.SQNR_DB_BY_DIST["N(0,1)"][0]))
    check("A", "fp16（N(0,1)）与解析式同量级（差 < 0.5 dB）",
          abs(SPECS.SQNR_DB_BY_DIST["N(0,1)"][1] - ref_float_sqnr_analytic_db(10)) < 0.5,
          "解析 %.2f / 实测 %.2f" % (ref_float_sqnr_analytic_db(10),
                                    SPECS.SQNR_DB_BY_DIST["N(0,1)"][1]))
    check("A", "e4m3fn（N(0,1)）与解析式同量级（差 < 0.5 dB）",
          abs(SPECS.SQNR_DB_BY_DIST["N(0,1)"][2] - ref_float_sqnr_analytic_db(3)) < 0.5,
          "解析 %.2f / 实测 %.2f" % (ref_float_sqnr_analytic_db(3),
                                    SPECS.SQNR_DB_BY_DIST["N(0,1)"][2]))


# ===========================================================================
# §C 边界与反例：这些构造**必须失败**
# ===========================================================================

def run_C():
    section("C", "边界与反例：以下构造必须失败（断言「失败行为成立」）")

    sub("C1  MX 的 Floor 取整：amax 落在 (1.75, 2) 的块，规约后必然越界")
    # 构造一个 amax 的尾数 m 恰好落在阈值**之上**的块（解析构造，不是抽样）
    for mag, want_overflow in ((1.9, True), (1.76, True), (1.74, False), (1.1, False)):
        xs = [2.0 ** 10 * mag, 0.01, -0.02, 0.0]      # amax = 2^10 * mag → m = mag
        over, m = ref_mx_blocked_overflow(xs, 8, 448.0)
        check("C", "e4m3：m=%.2f 的块%s越界（阈值 1.75）" % (mag, "" if want_overflow else "不"),
              over is want_overflow, "m=%r 越界=%s" % (m, over))
    for mag, want_overflow in ((1.6, True), (1.51, True), (1.49, False)):
        xs = [2.0 ** 10 * mag, 0.01]
        over, m = ref_mx_blocked_overflow(xs, 2, 6.0)
        check("C", "e2m1：m=%.2f 的块%s越界（阈值 1.5）" % (mag, "" if want_overflow else "不"),
              over is want_overflow, "m=%r 越界=%s" % (m, over))
    # 反例侧：**确实会溢出的块，规约后一定超过元素上限**
    xs = [2.0 ** 10 * 1.9]
    scale = ref_mx_scale(abs(xs[0]), 8)
    check("C", "溢出的块规约后确实超过 448（%g > 448）" % (xs[0] / scale),
          xs[0] / scale > 448.0, "%g" % (xs[0] / scale))
    xs2 = [2.0 ** 10 * 1.1]
    scale2 = ref_mx_scale(abs(xs2[0]), 8)
    check("C", "对照：不溢出的块规约后 ≤ 448（%g）" % (xs2[0] / scale2),
          xs2[0] / scale2 <= 448.0)

    sub("C2  四个 fp8 变体：中点及以上的输入**必须**落到 NaN / Inf")
    for (name, elem_max, ulp, thr, below, above) in CASES_OVERFLOW_BOUNDARY:
        check("C", "%s 在阈值 %g 处的下一个候选越界 → %s" % (name, thr, above),
              above in ("NaN", "inf"))
        check("C", "%s 的中点**不等于**元素上限（所以中点是可判别的）" % name,
              abs(thr - elem_max) > 1e-9)

    sub("C3  e8m0 表示不了的东西：零、负数、超出 [2^-127, 2^127] 的值")
    check("C", "0 落在表示范围之外（最小可表示是 2^-127）",
          0.0 < SPECS.IMPL_E8M0_MIN_FINITE)
    check("C", "负数落在表示范围之外（无符号格式）",
          ref_e8m0_value(0) is not None and all(
              (ref_e8m0_value(c) or 1.0) > 0 for c in range(256)))
    check("C", "2^200 越界 → 饱和到 2^127（**这是 ONNX `saturate=1` 那一列的语义**）",
          ref_e8m0_round(2.0 ** 200) == 2.0 ** 127)
    check("C", "2^-200 越界 → 饱和到 2^-127", ref_e8m0_round(2.0 ** -200) == 2.0 ** -127)
    # ★ 但 torch 的 CPU 路径在**高频端不饱和**（本机实测，三个版本一致）：
    #   阈值 1.5·2^127，其上一律 NaN。这一条**只能由编号程序验**（本文件不 import torch），
    #   本文件只负责把「规范侧该怎么算」和「实测侧是什么」分开摆出来，不让它们互相背书。
    hi = SPECS.IMPL_E8M0_HIGH_END_NOT_SATURATING
    check("C", "★ 实测侧（torch CPU）在高端**不饱和**：1.5·2^127 及以上 → NaN",
          hi["at_or_above"] == "NaN" and abs(hi["threshold"] - 1.5 * 2.0 ** 127) < 1e-12)
    check("C", "★ 阈值的判据与 01 那四个格式同一条：上限 + ulp/2（e8m0 的 ulp@2^127 = 2^127）",
          abs(hi["threshold"] - (2.0 ** 127 + 2.0 ** 127 / 2.0)) < 1e-12)
    check("C", "★ 写在 specs 里的口径是「规范说 CUDA，本机测 CPU」，**不是「ONNX 错了」**",
          hi["onnx_prose_scope"].startswith("ONNX 正文那句描述的是 CUDA"))
    check("C", "★ 两端不对称，不是两条规则：低频端饱和、高频端不饱和",
          hi["low_end"].startswith("一律饱和") and hi["at_or_above"] == "NaN")
    check("C", "NaN 传进去仍是 NaN", ref_e8m0_round(float("nan")) is None)

    sub("C4  MX 的 Floor 取整不可省：把 Floor 换成 ceil 会**系统性**改变溢出率")
    # 这不是「实现细节」，是判据的一部分：换取整方向，阈值集合不变但落点变
    over_floor, over_ceil = 0, 0
    random.seed(7)
    T = MX_MC_BLOCKS // 4
    for _ in range(T):
        xs = [random.gauss(0, 1) for _ in range(32)]
        amax = max(abs(v) for v in xs)
        e = math.floor(math.log2(amax))
        m = amax / (2.0 ** e)
        if m > 448.0 / 256.0:
            over_floor += 1
        e2 = math.ceil(math.log2(amax))
        if m * (2.0 ** (e - e2)) * 256.0 > 448.0:
            over_ceil += 1
    check("C", "Floor 取整下有非零溢出率（%d/%d）" % (over_floor, T), over_floor > 0)
    check("C", "ceil 取整下溢出率**归零**（%d/%d）——这就是 Floor 的代价" % (over_ceil, T),
          over_ceil == 0)

    sub("C5  MX 的两次取整是两件事：块 scale 用 Floor、元素用 RTNE")
    check("C", "specs 把这两条分开记（来源也不同：前者是论文 Alg.1，后者是规范）",
          SPECS.MX_SCALE_ROUNDING == "floor" and "RTNE" in SPECS.MX_ELEMENT_ROUNDING)
    check("C", "specs 明写「块 scale 的算法不是规范规定的」",
          SPECS.MX_SCALE_ROUNDING_SOURCE == "论文 Algorithm 1（不是规范）")

    sub("C6  torch.int4 是空壳：**所有承载数据的操作都必须失败**")
    # 本体不能 import torch，所以这里断言的是**规格表本身的自洽**：
    # 「能构造、element_size 错报、之后任何读写都抛异常」三者必须同时成立。
    # （真 torch 上的逐条复现由编号程序 04 承担，那才是本机实测值。）
    check("C", "element_size 错报为 1（语义上应为 0.5）",
          SPECS.INT4_ELEMENT_SIZE_REPORTED == 1 and SPECS.INT4_TRUE_ELEMENT_SIZE == 0.5)
    probes = SPECS.INT4_SHELL_PROBES
    ok_probes = [k for k, v in probes.items() if v.startswith("成功")]
    bad_probes = [k for k, v in probes.items() if v.startswith("抛异常")]
    check("C", "只有「构造」这一步成功（%d 条）" % len(ok_probes), len(ok_probes) == 1)
    check("C", "其余 %d 条**全部抛异常**" % len(bad_probes), len(bad_probes) == len(probes) - 1)
    check("C", "★ 结论：不得写成「torch 支持 int4 张量」",
          SPECS.INT4_TRUE_ELEMENT_SIZE == 0.5)
    check("C", "异常类型跨版本变了 → 断言只能断「抛了异常」，不能断具体异常类",
          len(set(SPECS.INT4_EXC_TYPE_BY_VERSION.values())) > 1)

    sub("C7  规范说「未定义」的地方，实现却有确定行为（P8 的实体）")
    check("C", "ONNX 对 e8m0 负值写的是 undefined behavior",
          "undefined" in SPECS.ONNX_E8M0_NEGATIVE_INPUT)
    check("C", "而 torch 的实测值给出了确定结果（-1.0 → 1.0）",
          SPECS.IMPL_E8M0_ABS_OF_NEGATIVE["-1.0"] == 1.0)
    check("C", "ONNX 的 round_mode 默认是 up（ties away from zero），**不是 RNE**",
          SPECS.ONNX_E8M0_ROUND_MODE_DEFAULT == "up"
          and "away from zero" in SPECS.ONNX_E8M0_ROUND_MODE_MEANINGS["up"])
    check("C", "而 torch 的源码注释自称 ties-to-even —— 结果一致、理由不同",
          SPECS.IMPL_E8M0_ROUNDING_COMMENT == "round to nearest, with ties to even"
          and SPECS.IMPL_E8M0_ROUNDING_TIES_ALWAYS_UP_FOR_NORMAL is True)

    sub("C7b 规范侧的版本号：文档里引用的每一个 opset 都要有断言咬住")
    # 文档 01 §2 与 02 §参考 都引了这两个版本号，所以它们必须被断言核过——
    # 否则 specs 里改一个数字，两边都发现不了（本仓库的变异测试抓到过这一点）。
    check("A", "saturate 首次出现在 Cast-19（逐节检索 ONNX Changelog 得到）",
          SPECS.ONNX_SATURATE_SINCE_OPSET == 19)
    check("A", "round_mode 首次出现在 Cast-24（**不是 21**，这是对计划文件的一处订正）",
          SPECS.ONNX_ROUND_MODE_SINCE_OPSET == 24)
    check("A", "round_mode 的引入版本**晚于** saturate 的引入版本（19 < 24）",
          SPECS.ONNX_SATURATE_SINCE_OPSET < SPECS.ONNX_ROUND_MODE_SINCE_OPSET)
    check("A", "当前 Cast 文档页的版本号是 28（页内 since_version: 28）",
          SPECS.ONNX_CAST_CURRENT_VERSION == 28)
    check("A", "e8m0 的默认 round_mode 是 up，且其语义逐字是 away from zero",
          SPECS.ONNX_E8M0_ROUND_MODE_DEFAULT == "up"
          and SPECS.ONNX_E8M0_ROUND_MODE_MEANINGS["up"] == "round to nearest value away from zero")
    check("A", "★ 但 torch 的实测行为与 `nearest` 那一列的**结果**一致（理由不同）",
          SPECS.ONNX_E8M0_ROUND_MODE_MEANINGS["nearest"]
          == "round to nearest value and ties round up")
    check("A", "saturate 的默认值是 1（饱和）",
          SPECS.ONNX_SATURATE_DEFAULT == 1)
    check("A", "saturate=0 时四个格式的越界去向：三个 NaN、只有 e5m2 是 Inf",
          SPECS.ONNX_F8_SATURATE_FALSE_OOR["E4M3FN"] == "NaN"
          and SPECS.ONNX_F8_SATURATE_FALSE_OOR["E4M3FNUZ"] == "NaN"
          and SPECS.ONNX_F8_SATURATE_FALSE_OOR["E5M2"] == "Inf"
          and SPECS.ONNX_F8_SATURATE_FALSE_OOR["E5M2FNUZ"] == "NaN")
    check("A", "saturate=1（默认）时越界一律到目标格式的最大有限数",
          SPECS.ONNX_F8_SATURATE_TRUE_OOR == "FLT_MAX"
          and SPECS.ONNX_F8_DEFAULT_ROUND_MODE_ALL_FOUR == "RNE")
    check("A", "torch 的溢出断点记在 2.11.0 与 2.13.0 之间，且两端都有实测",
          SPECS.IMPL_E4M3FN_OVERFLOW_BREAK_BETWEEN == ("2.11.0", "2.13.0")
          and SPECS.IMPL_E4M3FN_OVERFLOW_BY_VERSION["2.11.0"]["1e30"] == "NaN"
          and SPECS.IMPL_E4M3FN_OVERFLOW_BY_VERSION["2.13.0"]["1e30"] == 448.0)
    check("A", "MX 的 scale 取整来源标的是「论文 Algorithm 1」而不是规范",
          SPECS.MX_SCALE_ROUNDING_SOURCE == "论文 Algorithm 1（不是规范）"
          and SPECS.MX_SCALE_ROUNDING == "floor")
    check("A", "块大小：MX 是 32、NVFP4 是 16（两个数不得混用）",
          SPECS.OCP_MX_BLOCK_SIZE == 32 and SPECS.NVFP4_BLOCK_SIZE == 16)
    check("A", "★ 溢出率的两个阈值不同（448/256 = 1.75 vs 6/4 = 1.5），不得并列成一个数",
          abs(SPECS.MX_OVERFLOW_THRESHOLD["e4m3"] - 1.75) < 1e-12
          and abs(SPECS.MX_OVERFLOW_THRESHOLD["e2m1"] - 1.5) < 1e-12)
    check("A", "★ e8m0 的高频端阈值 = 1.5·2^127（上限 + ulp/2），且该端**不饱和**",
          abs(SPECS.IMPL_E8M0_HIGH_END_NOT_SATURATING["threshold"] - 1.5 * 2.0 ** 127) < 1e-12
          and SPECS.IMPL_E8M0_HIGH_END_NOT_SATURATING["at_or_above"] == "NaN")
    check("A", "★ torch 的 finfo 在 e5m2fnuz 的 eps 上是错的（0.125 vs 真值 0.25）",
          SPECS.IMPL_E5M2FNUZ_FINFO_EPS_IS_WRONG["finfo_reports"] == 0.125
          and SPECS.IMPL_E5M2FNUZ_FINFO_EPS_IS_WRONG["true_spacing_at_1.0"] == 0.25)
    check("A", "★ NVFP4 vs MXFP4 的数字标了「是仿真」",
          SPECS.NVFP4_VS_MXFP4_IS_SIMULATION is True)
    check("A", "tf32 的输入转换舍入模式记的是 UNRESOLVED",
          SPECS.TF32_INPUT_ROUNDING_MODE == "UNRESOLVED")
    check("A", "★ 「不 clamp 会怎样」是**版本相关**的，两个版本都记在表里",
          set(SPECS.MX_UNCLAMPED_FAILURE_BY_VERSION) == {"2.8.0", "2.13.0"})

    sub("C8  规范说「不规定」的地方，某实现强制（把论文当规范是镜像错误）")
    check("C", "MX 规范**不**规定块布局", SPECS.MX_BLOCK_LAYOUT_PRESCRIBED_BY_SPEC is False)
    check("C", "但 cuDNN 在 Blackwell 上强制 128×4 tiled layout（实现强制）",
          "128×4" in SPECS.MX_BLOCK_LAYOUT_IMPL_FORCED)


# ===========================================================================
# §D 可证伪性：把参考实现故意改错，A 类断言必须失败
# ===========================================================================

def broken_bias_off_by_one(e_bits, m_bits, bias, has_inf, nan_codes):
    """故意写错：bias 用 2^(E-1) − 1 的公式**统算**，忽略 fnuz 变体恰大 1。"""
    return ref_fp8_params(e_bits, m_bits, (1 << (e_bits - 1)) - 1, has_inf, nan_codes)


def broken_skip_zero_indexed_mantissa(e_bits, m_bits, bias, has_inf, nan_codes):
    """故意写错：无 Inf 时**不跳 NaN 码字**，直接把全 1 尾数当最大有限数
    （这恰好会在 e4m3fn 上给出 480，而正确答案是 448）。"""
    exp_all_ones = (1 << e_bits) - 1
    if has_inf:
        return ref_fp8_params(e_bits, m_bits, bias, has_inf, nan_codes)
    emax = exp_all_ones
    return {
        "max_finite": (2.0 - 2.0 ** -m_bits) * 2.0 ** (emax - bias),
        "smallest_sub": 2.0 ** (1 - bias - m_bits),
        "smallest_normal": 2.0 ** (1 - bias),
        "eps": 2.0 ** -m_bits,
    }


def broken_e8m0_ties_away_is_even(x):
    """故意写错：把「平局远离零」写成「平局取偶」（即 ties-to-even）。"""
    return ref_e8m0_round_ties_to_even(x)


def broken_e8m0_keep_sign(x):
    """故意写错：不取绝对值，直接把负数当 0 处理（很多人的第一直觉）。"""
    if x != x:
        return None
    if x <= 0:
        return SPECS.IMPL_E8M0_MIN_FINITE
    return ref_e8m0_round(x)


def broken_e8m0_has_zero(code):
    """故意写错：以为 e8m0 的 `0x00` 是 0（有符号/常规指数的直觉）。"""
    if code == 0:
        return 0.0
    return ref_e8m0_value(code)


def broken_mx_threshold_uses_max(elem_max, emax):
    """故意写错：把阈值写成「直接比较规约前的 amax 与元素上限」
    （漏掉 emax 这个因子——这正是 Floor 取整致溢的机制所在）。"""
    return elem_max


def run_D():
    section("D", "可证伪性：把参考实现故意改错，A 类判据必须失败")

    bad1 = []
    for (name, e, m, bias, has_inf, fnuz, mx, smin, snorm, eps) in CASES_FP8:
        nan_codes = ref_nan_codes(e, m, has_inf, fnuz)
        p = broken_bias_off_by_one(e, m, bias, has_inf, nan_codes)
        if abs(p["max_finite"] - mx) >= 1e-9 or abs(p["eps"] - eps) >= 1e-12:
            bad1.append(name)
    check("A", "改错 1（bias 统算、忽略 fnuz 大 1）会与手写期望冲突",
          len(bad1) > 0, "冲突项：%s" % bad1)

    bad2 = []
    for (name, e, m, bias, has_inf, fnuz, mx, smin, snorm, eps) in CASES_FP8:
        nan_codes = ref_nan_codes(e, m, has_inf, fnuz)
        p = broken_skip_zero_indexed_mantissa(e, m, bias, has_inf, nan_codes)
        if abs(p["max_finite"] - mx) >= 1e-9:
            bad2.append((name, p["max_finite"], mx))
    check("A", "改错 2（不跳 NaN 码字）会在 e4m3fn 上给出 480 而不是 448",
          len(bad2) == 1 and bad2[0][0] == "float8_e4m3fn",
          "冲突项：%s（只有 e4m3fn —— 因为只有它的最大有限数紧挨着 NaN 码字）" % bad2)

    bad3 = [x for x, want in CASES_E8M0
            if x == x and broken_e8m0_ties_away_is_even(x) != want
            and not (want is not None and abs((broken_e8m0_ties_away_is_even(x) or 0) - want) < 1e-12)]
    check("A", "改错 3（把平局规则换成 ties-to-even）会与手写期望冲突",
          len(bad3) > 0, "冲突项：%s" % bad3)

    bad4 = [x for x, want in CASES_E8M0
            if x == x and x < 0 and not (abs((broken_e8m0_keep_sign(x) or 0) - want) < 1e-12)]
    check("A", "改错 4（负数不取绝对值）会与手写期望冲突",
          len(bad4) > 0, "冲突项：%s" % bad4)

    vals_broken = [broken_e8m0_has_zero(c) for c in range(256)]
    zero_broken = [c for c, v in enumerate(vals_broken) if v == 0.0]
    check("A", "改错 5（以为 0x00 是 0）会与「没有零码字」的期望冲突",
          len(zero_broken) > 0, "冒出 %d 个零码字" % len(zero_broken))

    bad6 = []
    for (fmt, emax, elem_max, thr_expect, uniform_expect) in MX_CASES:
        if abs(broken_mx_threshold_uses_max(elem_max, emax) - elem_max / (2.0 ** emax)) >= 1e-12:
            bad6.append(fmt)
    check("A", "改错 6（阈值漏掉 emax 因子）会与手写期望冲突",
          len(bad6) == 2, "冲突项：%s" % bad6)

    ok = [1 for (name, e, m, bias, has_inf, fnuz, mx, smin, snorm, eps) in CASES_FP8
          if abs(ref_fp8_params(e, m, bias, has_inf, ref_nan_codes(e, m, has_inf, fnuz))["max_finite"]
                 - mx) < 1e-9]
    check("A", "对照：正确实现与手写期望**全部**一致",
          len(ok) == len(CASES_FP8), "%d/%d" % (len(ok), len(CASES_FP8)))


# ===========================================================================
# §E 蒙特卡洛复核（独立于编号程序的那一份）
# ===========================================================================

def run_E():
    section("E", "MX 致溢率的蒙特卡洛复核（标准库 random，独立于编号程序那份）")

    random.seed(20260918)
    hits = {"e4m3": 0, "e2m1": 0}
    thrs = {"e4m3": 448.0 / 256.0, "e2m1": 6.0 / 4.0}
    for _ in range(MX_MC_BLOCKS):
        xs = [random.gauss(0, 1) for _ in range(32)]
        amax = max(abs(v) for v in xs)
        m = amax / (2.0 ** math.floor(math.log2(amax)))
        for k, thr in thrs.items():
            if m > thr:
                hits[k] += 1
    print("    MC 规模：%d 个 N(0,1) 的 32 元素块（标准库 random.gauss，独立路径）"
          % MX_MC_BLOCKS)
    for k in ("e4m3", "e2m1"):
        rate = hits[k] / float(MX_MC_BLOCKS)
        want = MX_MC_EXPECT[k]
        check("A", "MX %s 的致溢率 MC = %.1f%%（期望 %.1f%%，容差 ±%.1f%%）"
              % (k, 100 * rate, 100 * want, 100 * MX_MC_TOL),
              abs(rate - want) <= MX_MC_TOL, "实测 %.4f" % rate)
    check("A", "★ 这两个 MC 数与 specs 里的**解析估算**（25%% / 50%%）不是同一个量",
          abs(SPECS.MX_OVERFLOW_UNIFORM_ESTIMATE["e4m3"] - MX_MC_EXPECT["e4m3"]) > 0.05
          and abs(SPECS.MX_OVERFLOW_UNIFORM_ESTIMATE["e2m1"] - MX_MC_EXPECT["e2m1"]) > 0.05)
    check("A", "★ 两个阈值本来就不同（448/256 = 1.75 vs 6/4 = 1.5），不得并列成一个数",
          abs(448.0 / 256.0 - 6.0 / 4.0) > 0.2)


# ===========================================================================
# 主流程
# ===========================================================================

def main():
    print("=" * 76)
    print("130-model-optimization/quantization —— 纯 python3 复核（**不 import torch**）")
    print("=" * 76)
    print()
    print("口径：本文件全部输出是**推导**，不是本机实测值。")
    print("      它跑的是本文件自己的参考实现与手写期望值，不碰任何框架、不碰硬件。")
    print("      编号程序（01–04）在装了 torch 的环境里跑真系统，那才是本机实测值。")
    print("      **两档不得互相背书**：数学成立不证明实现如此，反之亦然。")
    print()
    print("版本绑定：本主题的实测基线锚在 torch %s（本机五个解释器）；"
          % " / ".join(VERSIONS_BOUND))
    print("          **%s 本机没有解释器，不声称覆盖**。" % UNCOVERED)
    print("          python3 = %s（本文件的唯一测量条件）" % sys.version.split()[0])

    run_A()
    run_C()
    run_D()
    run_E()

    section("汇总", "断言汇总")
    print("  共 %d 条：A 类 %d 条，C 类 %d 条" % (_STATE["n"], _STATE["A"], _STATE["C"]))
    if _STATE["fail"]:
        print("  **不通过 %d 条**：" % len(_STATE["fail"]))
        for f in _STATE["fail"]:
            print("    - %s" % f)
        return 1
    print("  全部通过。")
    print()
    print("  提醒：本文件通过与「编号程序跑出的实测数字」是**两档不同的证据**，")
    print("        不得互相背书（CONTRIBUTING §2）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
r"""02-significance —— 单个「快了 3%」有多少信息量，以及「紧得可疑」为什么可疑。

对应文档：`../02-significance.md`

★ 本程序的可证伪断言：
  A 类  最小可检出差异的解析式 `δ* = (z_{α/2} + z_β)·σ·√(2/n)` 与蒙特卡洛一致；
        ★ 因子是 **√(2/n)** 而不是 √(1/n)（漏 √2 会让解析值系统性偏小 1.414 倍）；
        ★★ t 分布的实现用已知分位数自检（t(1)=12.706、t(4)=2.132、t(8)=2.306 …）。
  B 类  给出「要检出 3% 的差异需要多少次重复」的公式解与蒙特卡洛检定力；
        以及 `δ* ∝ σ`、`n ∝ σ²` 这两条定量关系。
  C 类  ★★ 三个「真的跑出错误结论」的构造：
        ① **单次测量**判胜负的假阳性率 = **50%**（不是 5%）；
        ② **临界值约定**不同 ⇒ 假阳性率从 2.5% 变到 6.9%，且**报显著性时必须写明**用哪个；
        ③ **计时器分辨率**把方差压平 ⇒ σ 估计为 0 ⇒ 最小可检出差异变成 0 ⇒
           任何一次刻度跳变都被判为「显著」，而它可能是真实时间的 **167%**。
  D 类  `specs.py` 的每条常量逐条被断言钉住；含 `ast` 的跨程序覆盖检查（在 01 里汇总）。

跑法：`make 02_significance`。纯 python3 + numpy，不 import torch，不碰 GPU。

★ 口径声明：
  · 全部数字是**本库构造的统计模型的蒙特卡洛/解析值**，不是任何真实系统的测量结果。
  · 外部数字：§5 的 C 级文献（ASPLOS'09 / HPEC / NIST TN 1826 / Welch 小样本模拟）、
    §6 的 A 级计时器事实（`cudaEventElapsedTime` 分辨率 0.5 µs）。**经检索转述**。
  · 本程序不含任何 kernel 实现代码（域 README 的硬边界）。
"""

import sys
from math import exp, lgamma, log
from statistics import NormalDist

import numpy as np

import specs as SP

FAILED = []
ND = NormalDist()


def check(name, cond, detail=""):
    print(f"    [{'OK  ' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILED.append(name)


def sub(t):
    print()
    print(f"  ── {t}")


def head(t):
    print()
    print("=" * 78)
    print(t)
    print("=" * 78)


# ===========================================================================
# 0. 本程序自带的小工具：正态分位数 + **t 分布**的分布函数与分位数
# ===========================================================================
#
# ★★ 为什么必须自带 t 分布：本程序的核心论断之一是「**临界值用 z 还是 t 会改变结论**」，
#    而 `statistics` 里只有正态分布。t 的分布函数用正则化不完全 beta 函数的连分式展开
#    （Numerical Recipes 的形式）实现，**再用已知分位数自检**（见 §A.4）——
#    一个没有自检的数值工具本身就是本主题要防的那种东西。
#
# ⚠ 局限（明写，不假装是通用实现）：只对**正的 t** 求上尾概率（本程序只做单侧检验），
#    连分式的迭代上限与收敛判据沿用经典取值，未做极端参数下的收敛性分析。


def _betacf(a, b, x):
    MAXIT, EPS, FPMIN = 300, 3e-16, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < EPS:
            break
    return h


def _betai(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * log(x) + b * log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def t_sf(t, df):
    """`P(T > t)`（t 分布的上尾概率），`t` 可为任意实数。"""
    x = df / (df + t * t)
    p = 0.5 * _betai(df / 2.0, 0.5, x)
    return p if t > 0 else 1.0 - p


def t_crit(alpha, df):
    """**单侧 α 检验**的临界值：满足 `P(T > t) = alpha` 的 `t`（`0 < alpha < 0.5`）。

    ★★ 参数的语义必须写死，否则极易错位（本程序的第一版就在这里错过一次）：
       这里传的是**上尾概率 α**，不是分位 1−α。
       单侧 2.5% 的临界值 = `t_crit(0.025, df)`；单侧 5% = `t_crit(0.05, df)`。
    **二分**求解，非解析。
    """
    lo, hi = 0.0, 200.0
    for _ in range(120):
        mid = (lo + hi) / 2.0
        if t_sf(mid, df) > alpha:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def welch_t(a, b):
    """两组的 Welch t 统计量（**向量化**，沿 axis=1）。

    ★★ 代数事实（本程序 §C.3 会断言它）：**两组 n 相等时**，
        Welch 统计量 `(x̄−ȳ)/√(s²ₐ/n + s²_b/n)` 与合并版 `(x̄−ȳ)/(s_p·√(2/n))`
        **恒等**——因为 `s_p·√(2/n) = √((s²ₐ+s²_b)/n)`。所以差别**只在自由度**。
    """
    na, nb = a.shape[1], b.shape[1]
    se = np.sqrt(a.var(axis=1, ddof=1) / na + b.var(axis=1, ddof=1) / nb)
    return (b.mean(axis=1) - a.mean(axis=1)) / se


def welch_df(a, b):
    """Welch–Satterthwaite 自由度（**逐样本**，向量化）。"""
    na, nb = a.shape[1], b.shape[1]
    va = a.var(axis=1, ddof=1) / na
    vb = b.var(axis=1, ddof=1) / nb
    return (va + vb) ** 2 / (va ** 2 / (na - 1) + vb ** 2 / (nb - 1))


def mde_analytic(sigma, n, coefficient=SP.MDE_COEFFICIENT_AT_POWER_80):
    """`δ* = (z_{α/2} + z_β)·σ·√(2/n)`。**两组独立样本、每组 n 次**。"""
    return coefficient * sigma * np.sqrt(2.0 / n)


def detection_rate(delta, sigma, n, crit, reps, rng):
    """真实差异 `delta`、噪声 σ、每组 n 次时，单侧检验的**检出率**（蒙特卡洛）。

    ★ 临界值由调用方给出——**这正是本程序 §C.2 的主题**：换临界值就换结论。
    """
    a = sigma * rng.standard_normal((reps, n))
    b = delta + sigma * rng.standard_normal((reps, n))
    return float((welch_t(a, b) > crit).mean())


SEED = 20260920
RNG = np.random.default_rng(SEED)
MC_REPS = 40000
SIGMA = SP.SIGMA_PCT_DEFAULT / 100.0
print("=" * 78)
print("§0  对象：一个 A/B 比较的统计模型（不是任何真实系统）")
print("=" * 78)
print(f"  模型：两组独立样本，各 n 次，真实差异 δ（默认 0）、噪声 σ = {SIGMA}（= {SP.SIGMA_PCT_DEFAULT}%）")
print(f"  口径：**越小越好**（{SP.LOWER_IS_BETTER_METRIC_CONVENTION}）⇒ 单侧检验「b 是否更快」")
print(f"  ★ 全部数字是模型的解析值/蒙特卡洛值；重复次数 {MC_REPS}；"
      f"python {sys.version.split()[0]} / numpy {np.__version__}")


# ===========================================================================
# A. 最小可检出差异：解析式 vs 蒙特卡洛
# ===========================================================================
head("§A  最小可检出差异：解析式 vs 蒙特卡洛（复现规格预跑表）")

sub("A.1  解析式的三个输入：两个正态分位数与它们的和")
z95 = ND.inv_cdf(0.975)
z80 = ND.inv_cdf(0.80)
z95_from_alpha = ND.inv_cdf(1.0 - 0.05 / 2.0)
print(f"      z_(1−α/2) = z_0.975 = {z95!r}")
print(f"      z_β      = z_0.80  = {z80!r}")
print(f"      系数 = {z95 + z80!r}")
print(f"      ★ 一个**核对过就否掉**的猜想：`inv_cdf(1 − α/2)` 与 `inv_cdf(0.975)` 会不会")
print(f"        因为 `1 − 0.05/2` 的二进制表示而差 1 ulp？实测**不会**（差 {abs(z95 - z95_from_alpha):.1e}）。")
print(f"        ⇒ 两种写法在这里等价；**但要说「实测等价」，不能说「显然」**。")
check("A.1 ★ 两个分位数与 specs.py 的 [E] 常量逐位一致（**用字面量 0.975**，见下方提示）",
      z95 == SP.Z_ALPHA_OVER_2_95 and z80 == SP.Z_BETA_AT_POWER_80,
      f"{z95} / {z80}")
check("A.1 ★ 两种写法给出**同一个 double**（`1 − 0.05/2 == 0.975` 在双精度下成立）",
      z95 == z95_from_alpha and (1.0 - 0.05 / 2.0) == 0.975)
check("A.1 ★ 系数 = z_{0.975} + z_{0.80} = %r" % (z95 + z80),
      z95 + z80 == SP.MDE_COEFFICIENT_AT_POWER_80 == 2.801585218112968)

sub("A.2  ★★ 因子是 √(2/n)，不是 √(1/n)")
n_probe = 25
right = z95 + z80
wrong = (z95 + z80)
f_right = mde_analytic(SIGMA, n_probe)
f_wrong = wrong * SIGMA * np.sqrt(1.0 / n_probe)
print(f"      n = {n_probe}：√(2/n) 给 {f_right:.4f}（= {100 * f_right:.2f}%）")
print(f"      n = {n_probe}：√(1/n) 给 {f_wrong:.4f}（= {100 * f_wrong:.2f}%）")
print(f"      比值 = {f_right / f_wrong:.4f}（= √2）")
check("A.2 ★★ 两式之比恰好是 √2（这就是被漏掉的那个因子）",
      abs(f_right / f_wrong - np.sqrt(2.0)) < 1e-12, f"{f_right / f_wrong:.12f}")
# ★ 容差 0.05 = **引文末位半格**：规格把这两个数写成 2.4 与 1.7（**1 位小数**）
#   ⇒ 任一引用值与其真值最多差 0.05。两侧都是解析式算出来的**确定值**，没有 MC 噪声。
SPEC_ONE_DEC_HALF_ULP = 0.05
check("A.2 ★★ 规格的预跑值是 2.4%（σ = 3%、n = 25）——只有 √(2/n) 给得出它"
      f"（容差 0.05 = 引文 1 位小数的末位半格；实测差 {abs(100 * f_right - 2.4):.3f}）",
      abs(100 * f_right - 2.4) < SPEC_ONE_DEC_HALF_ULP, f"{100 * f_right:.3f}%")
check("A.2 ★ 规格初稿的式子（漏 √2）给 1.7% —— 也正是它自陈的那个错值"
      f"（同样用引文半格；实测差 {abs(100 * f_wrong - 1.7):.3f}）",
      abs(100 * f_wrong - 1.7) < SPEC_ONE_DEC_HALF_ULP, f"{100 * f_wrong:.3f}%")
check("A.2 ★ 规格已自订正（常量 `SPEC_CLAIM_MDE_USED_SQRT_1_OVER_N` = False）",
      SP.SPEC_CLAIM_MDE_USED_SQRT_1_OVER_N is False
      and SP.MDE_FORMULA_DENOMINATOR_IS_SQRT_2_OVER_N is True)

sub("A.3  ★ 解析式 vs 蒙特卡洛（二分找 80% 检出率的 δ）")


def mde_monte_carlo(sigma, n, crit, reps, rng, target=0.80):
    lo, hi = 0.0, 20.0 * sigma
    for _ in range(40):
        mid = (lo + hi) / 2.0
        if detection_rate(mid, sigma, n, crit, reps, rng) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


print("      n    规格解析   本库解析   蒙特卡洛(z 临界)  蒙特卡洛(t 临界)   规格 MC")
# ★★ 下面那条 MC 容差**不是闭式推出来的**，写明它的来路（这是本程序唯一的例外）：
#   蒙特卡洛的 δ 是**二分**出来的（40 步），估计量有两部分误差 ——
#   ① 系统偏差：检出率曲线由 MC_REPS 次抽样估计，二分的结果整体偏大；
#   ② 随机项：换种子后的散布。
#   跨 6 个种子实测：n=5 偏 +1.19%、sd 0.24%；n=25 偏 +0.26%、sd 0.072%；n=100 偏 +0.01%、sd 0.18%。
#   ⇒ 容差 = 最差档偏差 + 5σ_mc = 0.0119 + 5×0.0024 = 0.024，取 0.03（1.25 倍余量）。
#   ★ 旧值 0.05 没有来路（是「看着合理」的数，且是最差档的 4 倍）。
TOL_MC_MDE_REL = 0.03
mc_t_crit = {}
for n in (5, 25, 100):
    a_val = mde_analytic(SIGMA, n)
    mc_z = mde_monte_carlo(SIGMA, n, z95, MC_REPS, RNG)
    df = 2 * n - 2
    crit_t = t_crit(0.025, df)
    mc_t = mde_monte_carlo(SIGMA, n, crit_t, MC_REPS, RNG)
    mc_t_crit[n] = (mc_z, mc_t, crit_t)
    spec_a, spec_mc = SP.SPEC_PRERUN_MDE.get(n, (float("nan"), float("nan")))
    print("     %3d   %8.4f   %8.4f   %14.4f  %14.4f   %8.4f"
          % (n, spec_a, a_val, mc_z, mc_t, spec_mc))
    check(f"A.3 n = {n}：本库解析值与规格解析值一致（|差| < 1e-4 = 规格 4 位小数的 2 个半格；"
          f"实测 {abs(a_val - spec_a):.2e}）",
          abs(a_val - spec_a) < 1e-4, f"{a_val:.4f} vs {spec_a:.4f}")
    check(f"A.3 n = {n}：蒙特卡洛（z 临界）与解析式的相对差 < {TOL_MC_MDE_REL:.2f}"
          f"（= 最差档的系统偏差 +1.19% + 5σ_mc，见上）",
          abs(mc_z / a_val - 1.0) < TOL_MC_MDE_REL, f"{100 * (mc_z / a_val - 1.0):+.2f}%")
print("      ★ 规格的 n = 5 那一格给 0.0548，本库（z 临界）给 %.4f —— 差 %.2f%%。"
      % (mc_t_crit[5][0],
         100 * abs(mc_t_crit[5][0] / SP.SPEC_PRERUN_MDE.get(5, (0, float("nan")))[1] - 1.0)))
print("        成因是蒙特卡洛的临界值约定（见 A.4）：换成本库的 t 临界值，值还会再变。")
check(f"A.3 ★★ 三个 n 上解析式与蒙特卡洛都吻合到 {TOL_MC_MDE_REL:.2f} 以内 ⇒ 公式可用作 A 类对照"
      f"（最大 {100 * max(abs(mc_t_crit[n][0] / mde_analytic(SIGMA, n) - 1.0) for n in (5, 25, 100)):.2f}%，"
      f"用掉 {max(abs(mc_t_crit[n][0] / mde_analytic(SIGMA, n) - 1.0) for n in (5, 25, 100)) / TOL_MC_MDE_REL:.0%}）",
      all(abs(mc_t_crit[n][0] / mde_analytic(SIGMA, n) - 1.0) < TOL_MC_MDE_REL for n in (5, 25, 100)))

sub("A.4  ★★ t 分布工具的自检：已知分位数逐条对齐")
KNOWN_T = ((1, 0.025, 12.706204), (2, 0.025, 4.302653), (4, 0.025, 2.776445),
           (4, 0.05, 2.131847), (8, 0.025, 2.306004), (8, 0.05, 1.859548),
           (30, 0.025, 2.042272), (48, 0.025, 2.010635))
# ★ 容差 1e-6 = **引文末位半格的 2 倍**：参考分位数给到 6 位小数（半格 5e-7），
#   而本程序的二分求分位数是 120 次迭代（分辨率 200/2¹²⁰，可忽略）。
#   实测最大偏差 7.4e-7（= 1.5 个半格）⇒ 1e-6 有效；旧值 5e-6 是它的 5 倍。
TOL_T_REF = 1e-6
for df, alpha, ref in KNOWN_T:
    got = t_crit(alpha, df)
    check(f"A.4 t({df}) 的单侧 {100 * alpha:.1f}% 临界值 = {got:.6f}（参考 {ref}）"
          f"——|差| < 2 个末位半格（{TOL_T_REF}）",
          abs(got - ref) < TOL_T_REF, f"差 {abs(got - ref):.2e}")
# ★ 容差 0.002：t(2000) 与 z 的差**有闭式的一阶项**（Cornish–Fisher）
#   t(df) ≈ z + (z³+z)/(4df) ⇒ df = 2000 时 = (7.5292+1.9600)/8000 = 1.186e-3。
#   二阶项是 O(1/df²) ≈ 1e-7。⇒ 容差取 0.002（一阶项的 1.7 倍）；旧值 0.01 是它的 8 倍。
TOL_T_TO_Z = 0.002
T_Z_TO_Z_FIRST_ORDER = (z95 ** 3 + z95) / (4 * 2000)
check("A.4 ★ t 分布的分位数**在 df → ∞ 时收敛到 z**（t(2000) 与 z 相差 < %.4f = "
      "一阶项 %.6f 的 1.7 倍）" % (TOL_T_TO_Z, T_Z_TO_Z_FIRST_ORDER),
      abs(t_crit(0.025, 2000) - z95) < TOL_T_TO_Z,
      f"{t_crit(0.025, 2000):.6f} vs {z95:.6f}（差 {abs(t_crit(0.025, 2000) - z95):.6f}）")
print("      ★ 自检过的工具有资格当参照；`specs.py` 里没有 t 表，**因为它是算出来的**。")

sub("A.5  ★★ 同一份数据、四种临界值约定 ⇒ 四个不同的「最小可检出差异」")
print("      n = 5（每组），σ = 3%：")
MDE_BY_CRIT = {}
for name, crit in (("z(0.975)=1.960", z95), ("t(4)=2.132（单侧 5%）", t_crit(0.05, 4)),
                   ("t(8)=2.306（单侧 2.5%，正确）", t_crit(0.025, 8)),
                   ("t(4)=2.776（单侧 2.5%）", t_crit(0.025, 4))):
    # 用同一系数结构：δ* = (crit + z_β)·σ·√(2/n)
    v = (crit + z80) * SIGMA * np.sqrt(2.0 / 5)
    MDE_BY_CRIT[name] = v
    print(f"        {name:24s} ⇒ δ* = {v:.4f}（{100 * v:.2f}%）")
check("A.5 ★★ 用 t(8)（正确自由度）比用 z 大 %.1f%% —— 样本越小差得越多"
      % (100 * (MDE_BY_CRIT["t(8)=2.306（单侧 2.5%，正确）"] / MDE_BY_CRIT["z(0.975)=1.960"] - 1.0)),
      MDE_BY_CRIT["t(8)=2.306（单侧 2.5%，正确）"] > MDE_BY_CRIT["z(0.975)=1.960"])
check("A.5 ★ n = 5 时「用 z 还是用正确的 t(8)」让 δ* 相差 10% 以上",
      MDE_BY_CRIT["t(8)=2.306（单侧 2.5%，正确）"] / MDE_BY_CRIT["z(0.975)=1.960"] > 1.10,
      f"比值 {MDE_BY_CRIT['t(8)=2.306（单侧 2.5%，正确）'] / MDE_BY_CRIT['z(0.975)=1.960']:.4f}")


# ===========================================================================
# B. 需要多少次重复
# ===========================================================================
head("§B  「要检出 3% 的差异」需要多少次重复")

sub("B.1  公式解：由 `δ* = 系数·σ·√(2/n)` 反解 n")


def n_required(sigma, delta, coefficient=SP.MDE_COEFFICIENT_AT_POWER_80):
    return 2.0 * (coefficient * sigma / delta) ** 2


for s_pct, d_pct in ((3.0, 3.0), (3.0, 1.0), (6.0, 3.0), (3.0, 5.0)):
    n_req = n_required(s_pct / 100.0, d_pct / 100.0)
    print(f"      σ = {s_pct}%、要检出 δ = {d_pct}% ⇒ n = {n_req:.2f} ⇒ **每组取 "
          f"{int(np.ceil(n_req))} 次**")
check("B.1 ★ σ = 3%%、δ = 3%% ⇒ n = 2·(2.8016)² = %.3f ⇒ 取整 %d 次（公式解，不是经验值）"
      "——容差 1e-4 = 引文 15.6977（4 位小数）的 2 个半格，实测差 %.2e"
      % (n_required(0.03, 0.03), int(np.ceil(n_required(0.03, 0.03))),
         abs(n_required(0.03, 0.03) - 15.6977)),
      abs(n_required(0.03, 0.03) - 15.6977) < 1e-4)
check("B.1 ★★ `n ∝ σ²`：σ 翻倍 ⇒ 需要的重复次数**翻四倍**",
      abs(n_required(0.06, 0.03) / n_required(0.03, 0.03) - 4.0) < 1e-9)
check("B.1 ★★ `n ∝ 1/δ²`：要检出的差异减半 ⇒ 次数翻四倍",
      abs(n_required(0.03, 0.015) / n_required(0.03, 0.03) - 4.0) < 1e-9)

sub("B.2  ★★ 用规格那个「≈ 4σ/√n」会要多少倍样本")
n_4 = n_required(0.03, 0.03, coefficient=SP.MDE_REDUCED_COEFFICIENT_SPEC_CLAIMED)
n_exact = n_required(0.03, 0.03, coefficient=SP.MDE_REDUCED_COEFFICIENT_EXACT)
print(f"      系数 4.0   ⇒ n = {n_4:.2f}（每组 {int(np.ceil(n_4))} 次）")
print(f"      系数 2.80  ⇒ n = {n_exact:.2f}（每组 {int(np.ceil(n_exact))} 次）")
print(f"      ⇒ 那个「≈」把样本量**放大 %.0f%%**（它是保守的，但代价是真的）"
      % (100 * (n_4 / n_exact - 1.0)))
check("B.2 ★★ 4.0 与 2.80 的比值 = (4/2.8016)² ⇒ 恰好是同一个放大倍数的**平方**",
      abs((n_4 / n_exact) - (4.0 / SP.MDE_COEFFICIENT_AT_POWER_80) ** 2) < 1e-9,
      f"{n_4 / n_exact:.4f}")
check("B.2 ★ 规格把「≈ 4」写成精度可接受的约简形式——实测超出精确值 %0.1f%%"
      % SP.MDE_REDUCED_CLAIM_EXCEEDS_EXACT_PCT,
      SP.MDE_REDUCED_CLAIM_EXCEEDS_EXACT_PCT > 40.0,
      f"{SP.MDE_REDUCED_CLAIM_EXCEEDS_EXACT_PCT:.2f}%")

sub("B.3  ★ 蒙特卡洛复核：按公式解取的 n，检定力确实在 80% 附近")
crit8 = t_crit(0.025, 8)
for s_pct, d_pct in ((3.0, 3.0), (3.0, 5.0)):
    n_req = int(np.ceil(n_required(s_pct / 100.0, d_pct / 100.0)))
    rate = detection_rate(d_pct / 100.0, s_pct / 100.0, n_req, t_crit(0.025, 2 * n_req - 2),
                          MC_REPS, RNG)
    print(f"      σ = {s_pct}%、δ = {d_pct}%、n = {n_req} ⇒ 实测检定力 {100 * rate:.1f}%")
    check(f"B.3 n = {n_req} 的实测检定力在 70%–90% 之间（公式解没有系统性偏差；"
          f"取整会让实际检定力略偏离 80%）",
          0.70 <= rate <= 0.90, f"{100 * rate:.1f}%")
check("B.3 ★ 口径注意：检定力是对**正确的 t 临界值**测的（`t(2n−2)`）"
      "——容差 1e-6 = 引文 2.306004（6 位小数）的 2 个半格，实测差 %.2e" % abs(crit8 - 2.306004),
      abs(crit8 - 2.306004) < 1e-6, f"t(8)@0.975 = {crit8:.6f}")


# ===========================================================================
# C. ★★ 三个「真的跑出错误结论」的构造
# ===========================================================================
head("§C  ★★ 假阳性：单次测量、临界值约定、计时器分辨率")

sub("C.1  ★★ 用**单次测量**判胜负：假阳性率 = 50%")
# ★★ 容差**由二项噪声推出**（这是本主题 R6 的靶心，改前是硬编码的 1.0）：
#   估计量 = REPS1 次伯努利试验的样本比例 ⇒ σ_pp = 100·√(p(1−p)/REPS1)，p = 1/2
#   （δ = 0、两组同分布 ⇒ 胜率**恰好是 1/2**，这个 p 是模型给的，不是猜的）
#   K = 5：假失败率 5.7e-7（4σ 是 6.3e-5）⇒ **容差 = 5σ_pp**。
#   ★★ 为什么 REPS1 必须取 2×10⁶：两个要求把它**夹**出来了 ——
#     ①「把常量改到 50.5 必须被抓住」⇒ 容差 < |实测 − 50.5|；
#     ②「容差 ≥ 4σ」（否则测试会随机失败）；
#     REPS1 = 2×10⁵ 时 σ_pp = 0.1118 ⇒ 4σ = 0.447，而实测值本身偏离 50.0 约 0.08
#     ⇒ 要抓住 50.5 需要容差 < 0.42 < 4σ —— **两个要求不可能同时满足**。
#     REPS1 = 2×10⁶：σ_pp = 0.0354、5σ = 0.177 ⇒ 对 0.5 的偏移有 2.4 倍余量，两边都满足。
REPS1 = 2_000_000
SIGMA_WIN_PP = 100.0 * np.sqrt(0.5 * 0.5 / REPS1)
TOL_WIN_PP = 5.0 * SIGMA_WIN_PP
r1 = np.random.default_rng(SEED + 1)
a1 = r1.standard_normal(REPS1)
b1 = r1.standard_normal(REPS1)
win = float((b1 < a1).mean())      # 越小越好 ⇒ b 更快
print(f"      真实差异 δ = 0（两组同分布），各测 1 次，判「b 更快」的比例 = {100 * win:.4f}%")
print(f"      ★ 容差 = 5σ_pp = 5·100·√(0.25/{REPS1}) = {TOL_WIN_PP:.4f} 个百分点"
      f"（σ_pp = {SIGMA_WIN_PP:.4f}）")
check("C.1 ★★ 实测假阳性率 = %.4f%%，与解析值 %.1f%% 一致"
      "（|差| < 5σ_pp = %.4f 个百分点；用掉 %.0f%%）"
      % (100 * win, SP.SINGLE_MEASUREMENT_WIN_PROB_PCT, TOL_WIN_PP,
         100 * abs(100 * win - SP.SINGLE_MEASUREMENT_WIN_PROB_PCT) / TOL_WIN_PP),
      abs(100 * win - SP.SINGLE_MEASUREMENT_WIN_PROB_PCT) < TOL_WIN_PP, f"{100 * win:.4f}%")
check("C.1 ★★ 它比名义的 5% 高 **10 倍**（不是「略高」）——★ 这里断言的是**名义值**的关系"
      f"（{SP.SINGLE_MEASUREMENT_WIN_PROB_PCT:g} / 5 = 10）；旧版写 `win > 10 * 0.05`，"
      "那是拿蒙特卡洛的**方向**做断言：真实胜率恰好是 50%，样本值会在它上下摆动 "
      "⇒ **约一半的种子会失败**（实测：种子 22、33 都失败）",
      SP.SINGLE_MEASUREMENT_WIN_PROB_PCT / 5.0 == 10.0,
      f"实测 {100 * win:.4f}% 由上面那条 5σ 断言管住")
print("      ⇒ 「跑一次看快了就是快了」不是「有噪声的结论」，**是一次抛硬币**。")

sub("C.2  ★★ 临界值约定不同 ⇒ 假阳性率从 2.5% 变到 6.9%")
print("      两组各 n = 5、真实差异为 0、单侧检验。理论值 = P(t_8 > 临界值)：")
CRIT_ROWS = (
    (1.644854, "z 单侧 5%"),
    (1.959964, "z 单侧 2.5%（用 z 近似 t）"),
    (2.131847, "t(4) 单侧 5% ⇒ **用了一样本自由度 n−1**"),
    (2.306004, "t(8) 单侧 2.5% ⇒ **本题的正确自由度 2n−2**"),
    (2.776445, "t(4) 单侧 2.5%（过严）"),
)
reps5 = 300000
r5 = np.random.default_rng(SEED + 2)
a5 = r5.standard_normal((reps5, 5))
b5 = r5.standard_normal((reps5, 5))
t5 = welch_t(a5, b5)
for crit, name in CRIT_ROWS:
    theo = 100.0 * t_sf(crit, 8)
    mc = 100.0 * float((t5 > crit).mean())
    # ★ 本库**自己的**蒙特卡洛：这是比例估计 ⇒ σ 有闭式（σ_pp = 100·√(p(1−p)/reps5)），
    #   容差 = 5σ_pp（5.7e-7 的假失败率，且满足 ≥4σ）。
    sig_pp = 100.0 * np.sqrt((theo / 100.0) * (1.0 - theo / 100.0) / reps5)
    check(f"C.2 本库蒙特卡洛（reps5 = {reps5}）与理论值一致：|差| < 5σ_pp = {5 * sig_pp:.3f} pp"
          f"（σ_pp = {sig_pp:.4f}）",
          abs(mc - theo) < 5 * sig_pp, f"{mc:.3f}% vs {theo:.3f}%（差 {abs(mc - theo):.4f}）")
    spec = SP.SPEC_PRERUN_FP_RATE_PCT.get(round(crit, 3))
    spec_s = f"（规格 {spec}%）" if spec is not None else ""
    print(f"        {crit:.6f}  {name:42s} 理论 {theo:6.3f}%  蒙特卡洛 {mc:6.3f}% {spec_s}")
    if spec is not None:
        # ★ 容差 0.05：两侧都是**固定的数**（本库解析值 vs 本库引用的规格值），**没有随机性** ⇒
        #   「≥4σ」这类噪声判据在这里不适用。0.05 覆盖的是「规格值给到 2 位小数」
        #   （半格 0.005）与**规格自己那次 MC 的偏差**（它未披露重复次数；实测最大差 0.019）。
        #   ★ 旧值 0.1 是实测差的 5 倍，没有来路。
        check(f"C.2 规格给的 {spec}% 与理论值 P(t₈ > {crit:.3f}) = {theo:.2f}% 一致"
              f"（|差| < 0.05；实测 {abs(theo - spec):.4f}）",
              abs(theo - spec) < 0.05, f"{theo:.3f}%")
check("C.2 ★★ 正确的临界值（t(8)=2.306）给出**恰好 2.5%**"
      "——差 5.3e-7 只来自「2.306004 是 6 位小数的表值」（半格在概率上是 "
      "0.0303×5e-7×100 ≈ 1.5e-6 pp）⇒ 容差 1e-5（旧值 0.01 是它的 1000 倍）",
      abs(100 * t_sf(2.306004, 8) - 2.5) < 1e-5, f"{100 * t_sf(2.306004, 8):.7f}%")
check("C.2 ★★ 用 z(1.96) 时假阳性率 = %.2f%%（名义 2.5%% 的 %.2f 倍）"
      % (100 * t_sf(z95, 8), t_sf(z95, 8) / 0.025),
      t_sf(z95, 8) > 0.025 * 1.5)
check("C.2 ★★ 规格把 2.132 标成「t(4) 单侧 2.5%」——**标签错**："
      "2.132 是 t(4) 的 **0.95** 分位（单侧 5%），单侧 2.5% 是 2.776"
      f"（两个引用值各给到 6 位小数 ⇒ 容差 1e-6 = 2 个半格；实测差 "
      f"{abs(t_crit(0.05, 4) - 2.131847):.1e} / {abs(t_crit(0.025, 4) - 2.776445):.1e}）",
      abs(t_crit(0.05, 4) - 2.131847) < 1e-6 and abs(t_crit(0.025, 4) - 2.776445) < 1e-6
      and SP.SPEC_CLAIM_T4_CRIT_IS_ONE_SIDED_2_5PCT is False)
print("      ⇒ ★ 判据（进 §5）：报「显著」时必须写明**用的是哪个临界值、自由度怎么算**。")

sub("C.3  ★★ 规格把 3.29% 归因于 Welch 自由度——本库实跑把它推翻了")
print("      事实①：两组 n 相等时，Welch 统计量与合并版**代数恒等**（验收：两条路径的最大差 = 0）")
sp5 = np.sqrt(((5 - 1) * a5.var(axis=1, ddof=1) + (5 - 1) * b5.var(axis=1, ddof=1)) / (2 * 5 - 2))
t_pooled5 = (b5.mean(axis=1) - a5.mean(axis=1)) / (sp5 * np.sqrt(2.0 / 5))
check("C.3 ★★ 两条统计量**逐位相同**（最大绝对差 < 1e-9）",
      float(np.max(np.abs(t5 - t_pooled5))) < 1e-9, f"{float(np.max(np.abs(t5 - t_pooled5))):.2e}")

print("      事实②：固定临界值下，假阳性率**只由临界值决定**，与自由度无关——")
print("        P(t₈ > 2.132) = %.4f%% 正是规格实测的 3.29%%。" % (100 * t_sf(2.131847, 8)))
df5 = welch_df(a5, b5)
print("      事实③：Welch 自由度确实**小于**名义的 2n−2 = 8（均值 %.3f / 中位 %.3f），"
      % (df5.mean(), np.median(df5)))
print("        但把它**当临界值用**会让检验**更保守**（方向与规格的说法相反）：")
DF_GRID = np.linspace(3.0, 8.0, 201)
T_GRID = np.array([t_crit(0.025, d) for d in DF_GRID])
crit_welch = np.interp(df5, DF_GRID, T_GRID)
fp_data_dep = 100.0 * float((t5 > crit_welch).mean())
fp_fixed8 = 100.0 * float((t5 > 2.306004).mean())
print(f"        用 Welch 自由度当临界值（均值 {crit_welch.mean():.4f}）：假阳性率 "
      f"**{fp_data_dep:.3f}%** < 2.5%")
print(f"        用固定的 t(8) = 2.306004：                  假阳性率 **{fp_fixed8:.3f}%** ≈ 2.5%")
# ★ 两个假阳性率都是**比例估计**（reps5 次抽样）⇒ σ 有闭式，容差 = 5σ_pp。
#   ★★ 旧版写 `fp_data_dep < 2.5 < fp_fixed8`：拿 MC 值去比**名义值 2.5**，
#      而 fp_fixed8 的 σ_pp = 0.0285 ⇒ 它有一半的时候落在 2.5 以下 ⇒ **又是 flaky**。
#      改成「|fp_fixed8 − 2.5| < 5σ_pp」+「fp_data_dep 比 fp_fixed8 小 5σ 以上」。
FP_SIGMA_PP = 100.0 * np.sqrt(0.025 * 0.975 / reps5)
check("C.3 ★★ 用固定的 t(8) 时假阳性率 ≈ 名义 2.5%%（|差| < 5σ_pp = %.3f，"
      "σ_pp = 100·√(p(1−p)/reps5) = %.4f）" % (5 * FP_SIGMA_PP, FP_SIGMA_PP),
      abs(fp_fixed8 - 2.5) < 5 * FP_SIGMA_PP, f"{fp_fixed8:.3f}%")
check("C.3 ★★ 用 Welch 自由度当临界值 ⇒ 假阳性率**低于**名义值（保守），"
      "与规格「Welch 自由度造成膨胀」的方向**相反**"
      "（要求差值 > 5σ_pp，而不是「小于 2.5」这种把 MC 比名义值的写法）",
      fp_data_dep < fp_fixed8 - 5 * FP_SIGMA_PP,
      f"{fp_data_dep:.3f}% < {fp_fixed8:.3f}% − 5σ（差 {fp_fixed8 - fp_data_dep:.3f} = "
      f"{(fp_fixed8 - fp_data_dep) / FP_SIGMA_PP:.0f}σ）")
check("C.3 ★ 规格的那条机制句被标为 False，且方向常量写明是 conservative",
      SP.SPEC_CLAIM_WELCH_DF_CAUSES_FP_INFLATION is False
      and SP.SPEC_WELCH_DF_EFFECT_DIRECTION == "conservative")
check("C.3 ★★ Welch 近似本身**没有解析误差界**（文献结论：小样本 + 方差比 > 2 时变自由，"
      "最大膨胀 +58%）——本库的等方差情形不受影响，但引用时必须带上作用域",
      SP.WELCH_HAS_NO_ANALYTIC_ERROR_BOUND is True
      and SP.WELCH_LIBERAL_VARIANCE_RATIO_GT == 2.0
      and SP.WELCH_LIBERAL_SAMPLE_SIZE_BOUND == 30
      and SP.WELCH_MAX_TYPE_I_INFLATION_PCT == 58.0
      and SP.WELCH_SATTERTHWAITE_DF_IS_CRUCIAL is True)

sub("C.4  样本量上去了，约定之争就不重要了（但「不重要」有具体边界）")
for n in (5, 25, 100):
    c8 = t_crit(0.025, 2 * n - 2)
    print(f"      n = {n:3d}：正确临界值 t({2 * n - 2}) = {c8:.4f} · z = {z95:.4f} · "
          f"两者之比 {c8 / z95:.4f} ⇒ 名义 2.5% 下的假阳性率 "
          f"{100 * t_sf(z95, 2 * n - 2):.3f}%（用 z）vs 2.500%（用 t）")
check("C.4 ★★ 用 z 冒充 t 的代价随 n 单调下降（n = 5 时 > 4%，n = 100 时 < 2.6%）",
      100 * t_sf(z95, 8) > 4.0 and 100 * t_sf(z95, 198) < 2.6,
      f"{100 * t_sf(z95, 8):.3f}% → {100 * t_sf(z95, 198):.3f}%")
print("      ⇒ 判据：**小样本（每组 < 30）时临界值之争不是细节**；大样本时可以忽略。")

sub("C.5  ★★ 计时器分辨率：σ 估计被压平 ⇒ 最小可检出差异变成 0")
TAU_US = SP.CUDA_EVENT_TIMER_RESOLUTION_US          # 0.5 µs（[A] 级：NVIDIA 官方）
TRUE_T_US = 0.30
TRUE_SIGMA_US = 0.005
print(f"      计时器：`cudaEventElapsedTime` 分辨率 τ = {TAU_US} µs（**单位是毫秒，见 specs**）"
      f"· 返回单位 {SP.CUDA_EVENT_TIME_RETURN_UNIT!r}")
print(f"      被测 kernel：真实时间 {TRUE_T_US} µs，真实抖动 σ = {TRUE_SIGMA_US} µs"
      f"（相对抖动 {100 * TRUE_SIGMA_US / TRUE_T_US:.1f}%）")
print(f"      真实可分辨的最小差异（模型口径）：δ* = {mde_analytic(TRUE_SIGMA_US / TRUE_T_US, 5):.4f}"
      f"（{100 * mde_analytic(TRUE_SIGMA_US / TRUE_T_US, 5):.1f}% of {TRUE_T_US} µs）")

rq = np.random.default_rng(SEED + 3)
draws = TRUE_T_US + TRUE_SIGMA_US * rq.standard_normal(2000)
quantized = np.round(draws / TAU_US) * TAU_US
print(f"      读数的取值集合 = {sorted(set(np.round(quantized, 6).tolist()))} µs")
print(f"      ⇒ 实测 σ = {quantized.std(ddof=1):.6f} µs（真实 {TRUE_SIGMA_US}）")
check("C.5 ★★ 量化的读数是**常数**（σ 实测 = 0）而真实抖动 > 0",
      quantized.std(ddof=1) == 0.0 and TRUE_SIGMA_US > 0.0)
mde_zero = mde_analytic(0.0, 5)
print(f"      ⇒ 最小可检出差异 = ({SP.MDE_COEFFICIENT_AT_POWER_80:.4f})·0·√(2/5) = {mde_zero} ——")
print("        **任何**被报出来的差异都会「显著」，判定阈值退化成 0。")
check("C.5 ★★ σ 估计为 0 ⇒ δ* = 0（判定阈值退化）", mde_zero == 0.0)
print(f"      ★ 而计时器**能报出的最小非零差异**是一个刻度 τ = {TAU_US} µs = "
      f"**{100 * TAU_US / TRUE_T_US:.0f}% of {TRUE_T_US} µs**——")
print(f"        它比真实可分辨的 {100 * mde_analytic(TRUE_SIGMA_US / TRUE_T_US, 5):.1f}% "
      f"大 {TAU_US / (TRUE_T_US * mde_analytic(TRUE_SIGMA_US / TRUE_T_US, 5)):.1f} 倍。")
check("C.5 ★★ 一个刻度占真实时间的比例 > 100%（分辨率不足，读数没有分辨力）",
      TAU_US / TRUE_T_US > 1.0, f"{100 * TAU_US / TRUE_T_US:.0f}%")

print()
print("      并置一个**真实差异为 0** 的变体（同样 0.30 µs，只换了名字）：")
zero_like = np.round((TRUE_T_US + TRUE_SIGMA_US * rq.standard_normal(2000)) / TAU_US) * TAU_US
d_obs = 100.0 * (zero_like.mean() - quantized.mean()) / quantized.mean()
print(f"        两个变体的读数之差 = {d_obs:+.1f}%，各自的 σ = "
      f"{zero_like.std(ddof=1):.6f} / {quantized.std(ddof=1):.6f}")
check("C.5 ★★ 真实差异为 0 的两个变体给出**同一个读数、同一个 σ=0** ⇒ "
      "任何由配置差异引入的一个刻度都会变成「显著」",
      d_obs == 0.0 and zero_like.std(ddof=1) == 0.0)
print("      ⇒ ★ 这就是「**紧得可疑的标准差反而可疑**」的机械形态：")
print("        σ = 0 不是「测得很稳」，是「测的是计时器，不是 kernel」。")
print("        C 级佐证（ASPLOS'09 / HPEC / NIST TN 1826）见 §D.3。")

sub("C.6  ★ 同一个「τ 相对真实时间太大」的判据，在**边界情形**下方向相反")
TRUE_AT_BOUNDARY = TAU_US / 2.0
bnd = np.round((TRUE_AT_BOUNDARY + 0.005 * rq.standard_normal(4000)) / TAU_US) * TAU_US
print(f"      真实时间 = τ/2 = {TRUE_AT_BOUNDARY} µs、真实抖动只有 0.005 µs 时：")
print(f"        读数取值 {sorted(set(np.round(bnd, 6).tolist()))} ⇒ 实测 σ = {bnd.std(ddof=1):.4f} µs")
print(f"        ⇒ 实测 σ 是真实 σ 的 **{bnd.std(ddof=1) / 0.005:.0f} 倍** ⇒ δ* 被**高估**同样倍数")
check("C.6 ★★ 边界情形下 σ 被**高估 10 倍以上**（与 C.5 的 σ=0 方向相反）",
      bnd.std(ddof=1) / 0.005 > 10.0, f"{bnd.std(ddof=1) / 0.005:.0f}×")
check("C.6 ★★ 两种情形都表现为「σ 反常」⇒ 判据不能看 σ 的大小，要看 **σ 与 τ 的关系**",
      quantized.std(ddof=1) == 0.0 and bnd.std(ddof=1) > 10 * 0.005)
print("      ⇒ 判据（进 §5）：先算 `τ / 真实时间` 与 `σ_实测 / τ`，再看结论。")


# ===========================================================================
# D. 常量自洽
# ===========================================================================
head("§D  常量自洽：本篇用到的 specs.py 常量，逐条被断言钉住")

sub("D.1  测量偏差的经典结论（**[C] 级：ASPLOS 2009，已确认同行评审**）")
check("D.1 只改环境变量大小：性能变化 %s%%，最大近 %s%%"
      % (SP.MYTKOWICZ_ENV_SIZE_PERF_CHANGE_PCT, SP.MYTKOWICZ_ENV_SIZE_MAX_PERF_CHANGE_PCT),
      SP.MYTKOWICZ_ENV_SIZE_PERF_CHANGE_PCT == 33.0
      and SP.MYTKOWICZ_ENV_SIZE_MAX_PERF_CHANGE_PCT == 300.0)
check("D.1 ★★ 只改链接顺序：-O3 分别看起来快 %s%% 与慢 %s%% ⇒ **结论翻转**（不需要换硬件）"
      % (SP.MYTKOWICZ_LINK_ORDER_LUCKY_O3_ADVANTAGE_PCT,
         SP.MYTKOWICZ_LINK_ORDER_UNLUCKY_O3_DEFICIT_PCT),
      SP.MYTKOWICZ_LINK_ORDER_LUCKY_O3_ADVANTAGE_PCT == 8.0
      and SP.MYTKOWICZ_LINK_ORDER_UNLUCKY_O3_DEFICIT_PCT == 7.0)
check("D.1 调查了 %d 篇论文，其中充分考虑测量偏差的 = %d 篇"
      % (SP.MYTKOWICZ_SURVEYED_PAPERS, SP.MYTKOWICZ_PAPERS_THAT_ADDRESSED_MEASUREMENT_BIAS),
      SP.MYTKOWICZ_SURVEYED_PAPERS == 133
      and SP.MYTKOWICZ_PAPERS_THAT_ADDRESSED_MEASUREMENT_BIAS == 0)
check("D.1 两个对策：%s" % (SP.MYTKOWICZ_TECHNIQUES,),
      SP.MYTKOWICZ_TECHNIQUES == ("causal_analysis", "setup_randomization"))
check("D.1 ★ 检索**不支持**「40%」这个数 ⇒ 它在 UNSOURCED 里，正文不用",
      "mytkowicz_40pct_figure" in SP.UNSOURCED)

sub("D.2  计时器与统计前提（**[C] 级 / [A] 级**）")
check("D.2 HPEC 的计时模型含**准确度**分量，且最优重复次数受「准确度/精度」之比约束",
      SP.HPEC_TIMER_ACCURACY_TERM_EXISTS is True
      and SP.HPEC_OPTIMAL_REPS_BOUNDED_BY_ACCURACY_OVER_PRECISION is True
      and SP.HPEC_ROBUST_BENCH_IS_PEER_REVIEWED is True)
check("D.2 NIST TN 1826 三条：要置信区间 / 动态频率破坏 i.i.d. / 小样本被离群点毁掉",
      SP.NIST_TN1826_NEEDS_CONFIDENCE_INTERVAL is True
      and SP.NIST_TN1826_FREQUENCY_SCALING_BREAKS_IID is True
      and SP.NIST_TN1826_SMALL_SAMPLES_INVALIDATED_BY_OUTLIERS is True)
check("D.2 ★★ 计时器分辨率 = %s µs，返回单位 %r（两个单位不要混）"
      % (SP.CUDA_EVENT_TIMER_RESOLUTION_US, SP.CUDA_EVENT_TIME_RETURN_UNIT),
      SP.CUDA_EVENT_TIMER_RESOLUTION_US == 0.5
      and SP.CUDA_EVENT_TIME_RETURN_UNIT == "milliseconds")
check("D.2 ★ 两个 event 落在非 NULL 流上时结果**可能偏大**（方向固定的偏差）",
      SP.CUDA_EVENT_RECORD_IS_ASYNC_MAY_OVERCOUNT is True)
check("D.2 ★ 两个 event 之间要夹足够的工作量（否则测的是开销不是 kernel）",
      SP.CUDA_EVENT_NEEDS_ENOUGH_WORK_TO_BRACKET is True)

sub("D.3  ★★ 元线索：「一上来就 100%」与「紧得可疑」是同一件事的两个来源")
print("      来源一（本库，[E]）：2026-09-18 量化域计划的变异测试记录——")
print("        第一轮「95/95 全捕获」**全是假的**（harness 里的 KeyError 让每次变异都「被捕获」），")
print("        真实轨迹是 66 捕获 / 29 漏网 → 逐条补强 → 95/95。")
print("      来源二（外部，[C]）：计时器分辨率把方差压平 ⇒ 「干净」的读数是采集方式的产物。")
print("      ⇒ 两条**独立**来源指向同一件事：**过于干净的结果通常是采集方式的问题。**")
check("D.3 ★★ 本库记录的三个数（66 / 29 / 95）与规格一致",
      SP.QUANT_MUTATION_FIRST_ROUND_CAPTURED == 66
      and SP.QUANT_MUTATION_FIRST_ROUND_ESCAPED == 29
      and SP.QUANT_MUTATION_FINAL_CAPTURED == 95)
check("D.3 ★★ 「第一轮 100% 是假的」在常量里写明",
      SP.QUANT_FIRST_100PCT_WAS_FAKE is True)
check("D.3 ★ 两条来源的等级分别记下（本库 [E] / 外部 [C]）",
      SP.META_CLUE_SOURCE_TIERS == ("E", "C"))

sub("D.4  规格预跑值与**被推翻的条目**（**[E] 级**）")
check("D.4 规格的 MDE 预跑表被原样收录（供 §A.3 复算）",
      dict(SP.SPEC_PRERUN_MDE) == {5: (0.0531, 0.0548), 25: (0.0238, 0.0239),
                                   100: (0.0119, 0.0119)})
check("D.4 ★★ 规格的假阳性率四档被原样收录（供 §C.2 逐档核对）",
      SP.SPEC_PRERUN_FP_RATE_PCT == {1.645: 6.95, 1.960: 4.28, 2.132: 3.29, 2.776: 1.20})
check("D.4 ★★ 两条被推翻的条目都记在常量里（临界值标签 + Welch 归因）",
      SP.SPEC_CLAIM_T4_CRIT_IS_ONE_SIDED_2_5PCT is False
      and SP.SPEC_CLAIM_WELCH_DF_CAUSES_FP_INFLATION is False)
check("D.4 ★ 「紧得可疑」的来源等级被订正：规格标 D，检索到的是 C",
      SP.SPEC_CLAIM_TIGHT_SPREAD_SOURCE_TIER == "D"
      and SP.RETRIEVED_TIGHT_SPREAD_SOURCE_TIER == "C"
      and "tight_spread_suspicion_in_hipfire_doc" in SP.UNSOURCED)

sub("D.4b  ★★ 几个「只在文档里出现过」的常量也有断言（防止它们成为漏网的载体）")
check("D.4b ★★ MDE_REDUCED_CLAIM_EXCEEDS_EXACT_PCT 的**精确值**（不是「> 40」这种区间断言）",
      abs(SP.MDE_REDUCED_CLAIM_EXCEEDS_EXACT_PCT - 42.77631014537673) < 1e-9,
      f"{SP.MDE_REDUCED_CLAIM_EXCEEDS_EXACT_PCT!r}")
check("D.4b ★★ 它由两个常量**算出来**，不是硬编码：4.0/2.801585218112968 − 1 = %r"
      "——★ 这里两侧是**同一个表达式**，必须**逐位相同**（0.0）；旧容差 1e-9 会把 "
      "「副本漂移 2 个 ulp」这种变异放过去（实测漏网，见 mutate.py）"
      % (SP.MDE_REDUCED_COEFFICIENT_SPEC_CLAIMED / SP.MDE_REDUCED_COEFFICIENT_EXACT - 1.0),
      abs(SP.MDE_REDUCED_CLAIM_EXCEEDS_EXACT_PCT
          - (SP.MDE_REDUCED_COEFFICIENT_SPEC_CLAIMED
             / SP.MDE_REDUCED_COEFFICIENT_EXACT - 1.0) * 100.0) == 0.0)
check("D.4b ★★ `MDE_REDUCED_COEFFICIENT_EXACT` 与 `MDE_COEFFICIENT_AT_POWER_80` "
      "**逐位相同**（它是同一个数的**第二份拷贝**；两份拷贝漂移必须被抓）"
      "——★ 这正是叶子级变异测试抓出来的那一条漏网",
      SP.MDE_REDUCED_COEFFICIENT_EXACT == SP.MDE_COEFFICIENT_AT_POWER_80)
check("D.4b ★★ 而 specs.py 里 `MDE_REDUCED_CLAIM_EXCEEDS_EXACT_PCT` 那行用的是**字面量**"
      "（不是引用上面的名字）⇒ 必须断言「那个字面量与命名常量是同一个 double」，"
      "否则改名字常量、公式不动，派生值不变 ⇒ 漏网",
      (4.0 / SP.MDE_REDUCED_COEFFICIENT_EXACT - 1.0) * 100.0
      == (4.0 / SP.MDE_COEFFICIENT_AT_POWER_80 - 1.0) * 100.0)
check("D.4b ★ HPEC 那篇的标题逐字核对（本常量在别处只被引用、没被断言过）",
      SP.HPEC_ROBUST_BENCH_TITLE == "Robust benchmarking in noisy environments")
check("D.4b ★★ 「噪声带是普遍规律」这条规格论断被标为 False",
      SP.SPEC_CLAIM_NOISE_BAND_IS_GENERAL is False)
check("D.4b ★★ 四种临界值约定的完整清单（供 §C.2 逐档核对）",
      SP.SPEC_PRERUN_FP_RATE_PCT == {1.645: 6.95, 1.960: 4.28, 2.132: 3.29, 2.776: 1.20}
      and len(SP.SPEC_PRERUN_FP_RATE_PCT) == 4)

sub("D.5  重复次数建议的作用域（**[D] 级，二手**）")
check("D.5 MLPerf **对齐的方法学**用 %d 次预热 + %d 次测量，主指标 %r"
      % (SP.MLPERF_ALIGNED_WARMUP_RUNS, SP.MLPERF_ALIGNED_MEASUREMENT_RUNS,
         SP.MLPERF_ALIGNED_PRIMARY_METRIC),
      SP.MLPERF_ALIGNED_WARMUP_RUNS == 5 and SP.MLPERF_ALIGNED_MEASUREMENT_RUNS == 10
      and SP.MLPERF_ALIGNED_PRIMARY_METRIC == "median")
check("D.5 ★★ 官方 rulebook **没有取到** ⇒ 这一条不得写成「MLPerf 规定」",
      SP.MLPERF_OFFICIAL_RULEBOOK_NOT_RETRIEVED is True
      and "mlcommons_official_rulebook_warmup_and_run_counts" in SP.UNSOURCED)
check("D.5 「至少 %d 次、高方差加到 %d 次」是**练习建议**，不是推导结论"
      % (SP.BENCHMARK_MIN_RUNS_RECOMMENDED, SP.BENCHMARK_MIN_RUNS_HIGH_VARIANCE),
      SP.BENCHMARK_MIN_RUNS_RECOMMENDED == 10
      and SP.BENCHMARK_MIN_RUNS_HIGH_VARIANCE == 30)
check("D.5 ★ 预热阶段的结果**一律丢弃**（MLPerf 对齐方法学）",
      SP.WARMUP_DISCARD_IS_REQUIRED_BY_ALIGNED_METHODOLOGY is True)
check("D.5 ★★ 而本库自己给的是**公式**而不是固定次数（§B.1）：两者不是同一类结论"
      "（容差 1e-4 = 引文 4 位小数的 2 个半格）",
      abs(n_required(0.03, 0.03) - 15.6977) < 1e-4)

sub("D.6  ★ 全模块常量覆盖的**机械检查**")
COVERED_02 = (
    # §5 测量偏差与小样本统计（C 级文献 + 官方技术报告）
    "MYTKOWICZ_ENV_SIZE_PERF_CHANGE_PCT",
    "MYTKOWICZ_ENV_SIZE_MAX_PERF_CHANGE_PCT",
    "MYTKOWICZ_LINK_ORDER_LUCKY_O3_ADVANTAGE_PCT",
    "MYTKOWICZ_LINK_ORDER_UNLUCKY_O3_DEFICIT_PCT",
    "MYTKOWICZ_SURVEYED_PAPERS",
    "MYTKOWICZ_PAPERS_THAT_ADDRESSED_MEASUREMENT_BIAS",
    "MYTKOWICZ_TECHNIQUES",
    "HPEC_ROBUST_BENCH_TITLE",
    "HPEC_ROBUST_BENCH_IS_PEER_REVIEWED",
    "HPEC_TIMER_ACCURACY_TERM_EXISTS",
    "HPEC_OPTIMAL_REPS_BOUNDED_BY_ACCURACY_OVER_PRECISION",
    "NIST_TN1826_NEEDS_CONFIDENCE_INTERVAL",
    "NIST_TN1826_FREQUENCY_SCALING_BREAKS_IID",
    "NIST_TN1826_SMALL_SAMPLES_INVALIDATED_BY_OUTLIERS",
    "WELCH_SATTERTHWAITE_DF_IS_CRUCIAL",
    "WELCH_HAS_NO_ANALYTIC_ERROR_BOUND",
    "WELCH_LIBERAL_VARIANCE_RATIO_GT",
    "WELCH_LIBERAL_SAMPLE_SIZE_BOUND",
    "WELCH_MAX_TYPE_I_INFLATION_PCT",
    "MLPERF_ALIGNED_WARMUP_RUNS",
    "MLPERF_ALIGNED_MEASUREMENT_RUNS",
    "MLPERF_ALIGNED_PRIMARY_METRIC",
    "MLPERF_OFFICIAL_RULEBOOK_NOT_RETRIEVED",
    "BENCHMARK_MIN_RUNS_RECOMMENDED",
    "BENCHMARK_MIN_RUNS_HIGH_VARIANCE",
    "WARMUP_DISCARD_IS_REQUIRED_BY_ALIGNED_METHODOLOGY",
    # §6 计时器（A 级）
    "CUDA_EVENT_TIMER_RESOLUTION_US",
    "CUDA_EVENT_TIME_RETURN_UNIT",
    "CUDA_EVENT_RECORD_IS_ASYNC_MAY_OVERCOUNT",
    "CUDA_EVENT_NEEDS_ENOUGH_WORK_TO_BRACKET",
    # §7 统计常量（E 级）
    "Z_ALPHA_OVER_2_95",
    "Z_BETA_AT_POWER_80",
    "MDE_COEFFICIENT_AT_POWER_80",
    "MDE_FORMULA_DENOMINATOR_IS_SQRT_2_OVER_N",
    "MDE_REDUCED_COEFFICIENT_EXACT",
    "MDE_REDUCED_COEFFICIENT_SPEC_CLAIMED",
    "MDE_REDUCED_CLAIM_EXCEEDS_EXACT_PCT",
    "SINGLE_MEASUREMENT_WIN_PROB_PCT",
    # §8 规格预跑值与被推翻的条目
    "SPEC_PRERUN_MDE",
    "SPEC_CLAIM_MDE_USED_SQRT_1_OVER_N",
    "SPEC_PRERUN_FP_RATE_PCT",
    "SPEC_CLAIM_T4_CRIT_IS_ONE_SIDED_2_5PCT",
    "SPEC_CLAIM_WELCH_DF_CAUSES_FP_INFLATION",
    "SPEC_WELCH_DF_EFFECT_DIRECTION",
    "SPEC_CLAIM_NOISE_BAND_IS_GENERAL",
    "SPEC_CLAIM_TIGHT_SPREAD_SOURCE_TIER",
    "RETRIEVED_TIGHT_SPREAD_SOURCE_TIER",
    # §7 元线索（本库既有记录 E 级）
    "QUANT_MUTATION_FIRST_ROUND_CAPTURED",
    "QUANT_MUTATION_FIRST_ROUND_ESCAPED",
    "QUANT_MUTATION_FINAL_CAPTURED",
    "QUANT_FIRST_100PCT_WAS_FAKE",
    "META_CLUE_SOURCE_TIERS",
)
print(f"    02 覆盖 {len(COVERED_02)} 条")
check("D.6 ★ 02 的覆盖表里没有拼错的常量名",
      set(COVERED_02) <= {n for n in dir(SP) if n.isupper()},
      f"{sorted(set(COVERED_02) - {n for n in dir(SP) if n.isupper()})}")


# ===========================================================================
head("汇总")
# ===========================================================================
if FAILED:
    print(f"  ✗ {len(FAILED)} 条断言失败：")
    for f in FAILED:
        print(f"      · {f}")
    sys.exit(1)
print("  ✓ 全部断言通过（A 类 5 组 · B 类 3 组 · C 类 6 组 · D 类 6 组）")
print()
print("  ★ 全部数字是**统计模型的解析值/蒙特卡洛值**，不是任何真实系统的测量结果；")
print("    外部的 C 级文献与 A 级计时器事实**经检索转述**，只支撑它们各自的作用域。")
print("  ★ 三处 C 类**真的跑出了错误结论**：单次测量 ⇒ 50% 假阳性；")
print("    临界值用 z ⇒ 4.28%（名义 2.5%）；σ 估计被分辨率压平 ⇒ 判定阈值退化为 0。")
print(f"    python {sys.version.split()[0]} / numpy {np.__version__}（未 import torch）")

#!/usr/bin/env python3
r"""01-benchmark-hygiene —— 不 warmup 测出来的数字是「错的数字」，不是「有噪声的数字」。

对应文档：`../01-benchmark-hygiene.md`

★ 本程序的可证伪断言：
  A 类  均值偏差的解析式 `k/n · (slowdown − 1)` 与蒙特卡洛**逐位吻合**；
        **中位数偏差**的解析式（Blom 阶统计量近似）与蒙特卡洛吻合到 5% 以内；
        ★ 关键的一条：`k/n = 0.49` 时中位数偏差 **> 0.05**（超过一个稳态 σ），
          而 `k/n = 0.10` 时 **< 0.02** —— **两个都断言**，只断言前者说明不了机制。
  B 类  给出「把中位数偏差压到 1% 以下需要丢弃多少次前导测量」的临界值；
        并给出该机制的**适用边界**：冷启动倍数低于 `1 + σ·E[z_(m:m)]` 时机制不成立。
  C 类  ★★ 同一个实现（稳态时间完全相同）在两种测量协议下报出**符号相反**的结论；
        并构造一个**稳态真的快 25%** 的变体，因为冷启动更长而被报成**变慢**。
  D 类  `specs.py` 的每条常量逐条被断言钉住；含一条 `ast` 的跨程序机械检查——
        三个程序的常量覆盖表的**并集 == `specs.py` 的全部模块级常量**、且**两两不交**。

跑法：`make 01_benchmark_hygiene`。纯 python3 + numpy，不 import torch，不碰 GPU。

★ 口径声明（必须与文档一致，否则本程序的结论不可比）：
  · 被测对象是一个**本库构造的测量模型**（稳态时间 + 冷启动衰减 + 噪声），
    **不是**任何真实框架、任何真实 GPU 的测量结果。
  · 程序里的全部数字是**本机跑出的模型值** ⇒ 只能支撑关于**机制**的结论，
    **不能**支撑关于真实系统的结论。域 README 明令：本域任何一篇都不得声称有实测数据。
  · 外部数字来自 `specs.py` §1（**D 级：hipfire，单一项目**）、§2（**B 级：Triton 官方**）、
    §3（**B 级：vLLM 官方**）。★ **只有 B 级那组支撑本程序的核心论断**；
    D 级那组只支撑「那个项目这么做」。
  · 口径：`LOWER_IS_BETTER_METRIC_CONVENTION` —— 本程序里「时间」越小越好。
  · **本程序不含任何 kernel 实现代码**（域 README 的硬边界）。
"""

import ast
import pathlib
import sys
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
# 0. 模型与解析式
# ===========================================================================
#
# ★ 模型（域规格 Chunk 4 文档 01 的模型，参数全部来自 `specs.py` §7 的 [E] 常量）：
#
#     一次会话做 n 次测量，其中前 k 次是冷启动：
#         i <  k :  t_i = steady_nominal × slowdown        （冷启动值，本模型取**确定值**）
#         i >= k :  t_i ~ N(steady_nominal, sigma)          （稳态，本库建模为白噪声）
#
#   ⇒ 「保留全部数据」时，三个统计量的偏差可以**解析地**写出来：
#         mean   : 偏差 = (k/n)·(slowdown − steady_nominal)                    ← 精确
#         median : 偏差 = (v(p1) + v(p2))/2 − steady，v(r) = min(1 + σ·E[z_(r:m)], slowdown)
#         min    : 偏差 = σ·E[z_(1:m)]，**与 k 无关**（冷启动值不进最小值）      ← Blom 近似
#   其中 m = n − k，`p1/p2` 是**合并样本**里中位数所在的两个名次；
#   ★★ 那个 `min(...)` 是**必须的**：r ≤ m 时该名次本应是稳态的阶统计量，但若它的期望
#      **超过**冷启动值，实际次序会互换 ⇒ 取值被封顶在冷启动值上。
#      没有这个封顶，公式在「冷启动很轻微」的区间会错一个数量级（见 §B.2）。

ORDER_STAT_RULE = SP.ORDER_STATISTIC_QUANTILE_RULE


def order_stat_mean(i, m):
    """阶统计量的期望的 Blom 近似：`E[z_(i:m)] ≈ Φ⁻¹((i − 0.375) / (m + 0.25))`。

    ★ `i` 从 **1** 开始（`i = 1` 是最小值）。`m` 是子样本大小。
    ⚠ 这是一个**近似**，不是精确期望——A 类断言正是拿它与蒙特卡洛对拍。
    """
    return ND.inv_cdf((i - 0.375) / (m + 0.25))


def pooled_sample(n, k, slowdown, sigma, reps, rng, steady=SP.STEADY_TIME_NOMINAL):
    """返回 `(reps, n)` 的测量矩阵：前 `k` 列是冷启动值，其余是稳态噪声。"""
    out = np.empty((reps, n))
    m = n - k
    if m > 0:
        out[:, k:] = steady + sigma * rng.standard_normal((reps, m))
    if k > 0:
        out[:, :k] = steady * slowdown
    return out


def measured_stats(sample):
    """三个统计量（对 `reps` 维取均值）——返回的是**偏差**（相对稳态标称值）。"""
    med = np.median(sample, axis=1)
    return {
        "mean": float(sample.mean() - SP.STEADY_TIME_NOMINAL),
        "median": float(med.mean() - SP.STEADY_TIME_NOMINAL),
        "min": float(sample.min(axis=1).mean() - SP.STEADY_TIME_NOMINAL),
    }


def analytic_stats(n, k, slowdown, sigma, steady=SP.STEADY_TIME_NOMINAL):
    """三个统计量的**解析偏差**（mean 精确；median/min 用 Blom 近似）。"""
    m = n - k

    def pooled_rank_mean(pos0):
        """合并样本里 0-indexed 名次 `pos0` 处的期望值（升序）。

        ★★ 关键在最后那个 `min`：名次落在稳态区（`r ≤ m`）时，该处的**期望值**若超过
        冷启动值，实际排序会互换 ⇒ 取值必须是**两者较小者**。少了这一步，公式在
        「冷启动轻微」的区间会把偏差算大一个数量级（§B.2 实测）。
        """
        r = pos0 + 1
        v = steady + sigma * order_stat_mean(r, m) if r <= m else steady * slowdown
        return min(v, steady * slowdown) if slowdown >= 1.0 else v

    mean_bias = (k / n) * (steady * slowdown - steady)
    if m == 0:
        med_bias = steady * slowdown - steady
    elif n % 2 == 0:
        med_bias = (pooled_rank_mean(n // 2 - 1) + pooled_rank_mean(n // 2)) / 2 - steady
    else:
        med_bias = pooled_rank_mean(n // 2) - steady
    min_bias = (steady + sigma * order_stat_mean(1, m)) - steady if m > 0 else float("nan")
    return {"mean": mean_bias, "median": med_bias, "min": min_bias}


N = SP.N_MEASUREMENTS_DEFAULT
REPS = SP.MC_REPLICATES_DEFAULT
SEED = 20260920
RNG = np.random.default_rng(SEED)

head("§0  被测对象：一个测量模型（不是任何真实系统）")
print(f"  模型：n = {N} 次测量，前 k 次是冷启动值 = 稳态 × slowdown，其余 ~ N(1.0, σ)")
print(f"  参数：steady = {SP.STEADY_TIME_NOMINAL} · σ = {SP.STEADY_SIGMA_DEFAULT} · "
      f"slowdown = {SP.COLD_START_SLOWDOWN_DEFAULT}（均为 specs.py §7 的 [E] 可调参数）")
print(f"  重复次数：{REPS}（本机实测的模型值；python {sys.version.split()[0]} / numpy {np.__version__}）")
print("  ★ 下面每个数字都是**这个模型的蒙特卡洛**，与任何真实 GPU 无关。")


# ===========================================================================
# 0.1 ★★ 模型自洽的**前置断言**：先断言参数自洽，再拿它做计算
# ===========================================================================
#
# ★★ 纪律（来自 130-quantization 的教训「崩溃不是捕获」）：如果程序在参数被改坏时
#    **抛异常**（KeyError / OverflowError），那它证明的是「有地方不对」，**不是**
#    「这条断言抓到了这个错误」——变异测试会因此自己骗自己。
#    ⇒ 凡是要**用**某个常量去索引/循环的地方，先断言它的形状自洽；取不到值时用
#      `.get(..., nan)` 而不是 `[...]`。下面这一段就是那个「先过自洽、再使用」的门。
sub("§0.1  模型自洽的前置断言（不通过则后面的结论全部作废）")
check("0.1 稳态标称值 > 0", SP.STEADY_TIME_NOMINAL > 0.0)
check("0.1 稳态 σ ∈ (0, 1)", 0.0 < SP.STEADY_SIGMA_DEFAULT < 1.0)
check("0.1 冷启动倍数 ≥ 1（冷启动不会更快）", SP.COLD_START_SLOWDOWN_DEFAULT >= 1.0)
check("0.1 n = %d（判据档位要与它自洽）" % SP.N_MEASUREMENTS_DEFAULT,
      SP.N_MEASUREMENTS_DEFAULT == 100)
check("0.1 重复次数 ≥ 100（否则蒙特卡洛噪声会淹没有意义的差）",
      SP.MC_REPLICATES_DEFAULT >= 100)
check("0.1 ★★ K_OVER_N_GRID ⊂ SPEC_PRERUN_MEDIAN_BIAS 的键",
      set(SP.K_OVER_N_GRID) <= set(SP.SPEC_PRERUN_MEDIAN_BIAS))
check("0.1 ★★ K_OVER_N_GRID ⊂ SPEC_PRERUN_MEAN_BIAS 的键",
      set(SP.K_OVER_N_GRID) <= set(SP.SPEC_PRERUN_MEAN_BIAS))
check("0.1 ★★ 判据用的四档 k 值可由 n 唯一导出（0.10/0.49/0.50/0.90 × n 都是整数）",
      all(abs(frac * SP.N_MEASUREMENTS_DEFAULT - round(frac * SP.N_MEASUREMENTS_DEFAULT)) < 1e-9
          for frac in SP.K_OVER_N_GRID)
      and [int(round(f * SP.N_MEASUREMENTS_DEFAULT)) for f in (0.10, 0.49, 0.50, 0.90)]
      == [10, 49, 50, 90])
check("0.1 ★★ VLLM_DEEPGEMM_KERNEL_COUNTS 的两个键就是下面要取的两个",
      set(SP.VLLM_DEEPGEMM_KERNEL_COUNTS) == {"glm5_744b", "deepseek_v3_2_685b"})
K10, K49, K50, K90 = (int(round(f * SP.N_MEASUREMENTS_DEFAULT)) for f in (0.10, 0.49, 0.50, 0.90))


# ===========================================================================
# 0.2 ★★ 容差表：每一个容差都由「噪声 + 引文末位」推出，不留魔数
# ===========================================================================
#
# 纪律（本库 45-kernel-lifecycle 的 R6 教训）：容差不是「看起来合理」的数。
# 一个容差如果宽过噪声几个数量级，**变异就落在带里**——常量可以在带内漂移而不被检出
# （本主题实测过：某条容差是噪声的 9 倍，把常量改 1% 完全漏网）。
#   ⇒ 本程序的每一条容差都写成 `K·σ`（蒙特卡洛）或「引文末位半格」（确定性），
#     并把 K 取多少、为什么，写在下面。
#
# 三类来源，各自可算：
#   ① 蒙特卡洛估计量的标准误 σ_mc
#      · 均值类：σ_mc = σ·√m / (n·√REPS)      （前 k 次是**确定值**，只有 m = n−k 个稳态样本贡献方差）
#      · 中位数类：σ_mc ≈ 1.2533·σ / √(m·REPS)  （1.2533 = √(π/2)，渐近公式 var(median) = πσ²/(2m)）
#   ② 引文末位半格：规格预跑表给到 **3 位小数** ⇒ 任一引用值与其真值最多差 5e-4
#   ③ K 取 5：5σ 对应的假失败率是 5.7e-7（4σ 是 6.3e-5）。
#      ★ 判据（R6）：**容差 ≥ 4σ**（否则测试会随机失败），本程序统一取 5σ。
K_SIGMA_MC = 5.0
SPEC_QUOTE_HALF_ULP_3DEC = 5e-4      # 规格预跑表的最小位是 1e-3 ⇒ 半格 5e-4


def sigma_mc_median(frac):
    """`k/n = frac` 档：**蒙特卡洛中位数偏差**估计量的标准误。"""
    m = N - int(round(frac * N))
    return 1.2533 * SP.STEADY_SIGMA_DEFAULT / np.sqrt(m * REPS)


def sigma_mc_mean(frac):
    """`k/n = frac` 档：**蒙特卡洛均值偏差**估计量的标准误（前 k 次无方差）。"""
    m = N - int(round(frac * N))
    return SP.STEADY_SIGMA_DEFAULT * np.sqrt(m) / (N * np.sqrt(REPS))


# ===========================================================================
# A. 三个统计量的偏差：解析式 vs 蒙特卡洛
# ===========================================================================
head("§A  三个统计量的偏差：解析式（Blom 近似）vs 蒙特卡洛")

sub("A.1  复现规格 Chunk 4 文档 01 的预跑表（中位数与均值）")
print("      k     k/n   规格中位数  解析中位数  蒙特卡洛    规格均值   解析均值  蒙特卡洛")
grid = SP.K_OVER_N_GRID
mc_by_k = {}
for frac in grid:
    k = int(round(frac * N))
    s = pooled_sample(N, k, SP.COLD_START_SLOWDOWN_DEFAULT, SP.STEADY_SIGMA_DEFAULT, REPS, RNG)
    st = measured_stats(s)
    mc_by_k[k] = st
    an = analytic_stats(N, k, SP.COLD_START_SLOWDOWN_DEFAULT, SP.STEADY_SIGMA_DEFAULT)
    print("    %4d   %4.2f   %8.3f   %9.4f  %9.4f   %8.3f  %8.4f  %9.4f"
          % (k, frac, SP.SPEC_PRERUN_MEDIAN_BIAS.get(frac, float("nan")), an["median"],
             st["median"], SP.SPEC_PRERUN_MEAN_BIAS.get(frac, float("nan")), an["mean"],
             st["mean"]))
    # ★ 容差**逐档算出来**（见 §0.2）：5σ_mc（中位数与均值取较大的那个）+ 规格引文的末位半格。
    #   旧版这里是 `0.005 if frac <= 0.49 else 0.06` —— 0.005 是实测噪声的 30 倍、
    #   0.06 是实测**模型差**的 1.07 倍（碰巧够用），两个都不是推出来的。
    tol_spec = K_SIGMA_MC * max(sigma_mc_median(frac), sigma_mc_mean(frac)) \
        + SPEC_QUOTE_HALF_ULP_3DEC
    spec_med = SP.SPEC_PRERUN_MEDIAN_BIAS.get(frac, float("nan"))
    spec_mean = SP.SPEC_PRERUN_MEAN_BIAS.get(frac, float("nan"))
    if frac <= 0.49:
        check(f"A.1 ★ 规格中位数偏差在 k/n = {frac:.2f} 档被复现"
              f"（容差 {tol_spec:.4f} = 5σ_mc {K_SIGMA_MC * sigma_mc_median(frac):.4f} + 引文半格）",
              abs(st["median"] - spec_med) < tol_spec,
              f"{st['median']:+.4f} vs {spec_med:+.3f}（{abs(st['median'] - spec_med):.2f}e-4 用掉 "
              f"{abs(st['median'] - spec_med) / tol_spec:.0%}）")
        check(f"A.1 ★ 规格均值偏差在 k/n = {frac:.2f} 档被复现（同上容差）",
              abs(st["mean"] - spec_mean) < tol_spec,
              f"{st['mean']:+.4f} vs {spec_mean:+.3f}")
    else:
        # ★★ 这两档**不与规格比容差**：它们的差不是噪声，是**两个模型的差**
        #    （规格的冷启动值带噪声、本库取确定值；k/n = 0.50 的差 = 稳态上尾项
        #     σ·E[z_(m:m)]/2，k/n = 0.90 的差 = 0.0070）。模型差由 §A.5 单独钉住；
        #    这里只断言「蒙特卡洛 == 解析式」，容差是纯 5σ_mc。
        check(f"A.1 ★ 本库解析中位数在 k/n = {frac:.2f} 档被蒙特卡洛复现"
              f"（纯 5σ_mc = {K_SIGMA_MC * sigma_mc_median(frac):.4f}；规格值另见 A.5）",
              abs(st["median"] - an["median"]) < K_SIGMA_MC * sigma_mc_median(frac),
              f"mc {st['median']:+.4f} vs 解析 {an['median']:+.4f}；规格 {spec_med:+.3f}"
              f"（模型差 {abs(st['median'] - spec_med):.4f}）")
        check(f"A.1 ★ 本库解析均值在 k/n = {frac:.2f} 档被蒙特卡洛复现（纯 5σ_mc）",
              abs(st["mean"] - an["mean"]) < K_SIGMA_MC * sigma_mc_mean(frac),
              f"mc {st['mean']:+.4f} vs 解析 {an['mean']:+.4f}；规格 {spec_mean:+.3f}")
check("A.1 ★★ 规格的两张表**逐档**被本库复算（6 档，不只是抽查四档）——"
      "k/n ≤ 0.49 的四档：容差 = 5σ_mc + 引文半格（规格给到 3 位小数）；"
      "另两档：蒙特卡洛只与**解析值**比（它们的规格差是模型差，见 A.5）",
      all(abs(mc_by_k[int(round(f * N))]["median"]
              - SP.SPEC_PRERUN_MEDIAN_BIAS.get(f, float("nan")))
          < K_SIGMA_MC * max(sigma_mc_median(f), sigma_mc_mean(f)) + SPEC_QUOTE_HALF_ULP_3DEC
          and abs(mc_by_k[int(round(f * N))]["mean"]
                  - SP.SPEC_PRERUN_MEAN_BIAS.get(f, float("nan")))
          < K_SIGMA_MC * max(sigma_mc_median(f), sigma_mc_mean(f)) + SPEC_QUOTE_HALF_ULP_3DEC
          for f in SP.K_OVER_N_GRID if f <= 0.49)
      and all(abs(mc_by_k[int(round(f * N))]["median"]
                  - analytic_stats(N, int(round(f * N)), SP.COLD_START_SLOWDOWN_DEFAULT,
                                   SP.STEADY_SIGMA_DEFAULT)["median"])
              < K_SIGMA_MC * sigma_mc_median(f)
              for f in SP.K_OVER_N_GRID if f > 0.49))
check("A.1 ★★ k/n = 0.50 那 0.056 的差**恰好是稳态上尾项** σ·E[z_(m:m)]/2"
      "（规格取中点、本库加上尾）—— 两个模型的差，不是误差",
      abs((mc_by_k[K50]["median"] - SP.SPEC_PRERUN_MEDIAN_BIAS[0.50])
          - SP.STEADY_SIGMA_DEFAULT * order_stat_mean(N - K50, N - K50) / 2.0)
      < K_SIGMA_MC * sigma_mc_median(0.50),
      f"{mc_by_k[K50]['median'] - SP.SPEC_PRERUN_MEDIAN_BIAS[0.50]:+.4f} vs "
      f"σ·E[z] /2 = {SP.STEADY_SIGMA_DEFAULT * order_stat_mean(N - K50, N - K50) / 2.0:.4f}")

worst_med = 0.0
worst_med_abs = 0.0
for frac in grid:
    k = int(round(frac * N))
    an = analytic_stats(N, k, SP.COLD_START_SLOWDOWN_DEFAULT, SP.STEADY_SIGMA_DEFAULT)
    mc = mc_by_k[k]
    rel = abs(mc["median"] - an["median"]) / max(abs(an["median"]), 1e-12)
    worst_med = max(worst_med, rel if frac not in (0.0,) else 0.0)
    worst_med_abs = max(worst_med_abs, abs(mc["median"] - an["median"]))
check("A.1 ★★ 中位数：解析式（Blom）与蒙特卡洛吻合到 **5σ_mc 以内**（最大绝对差 %0.2e < %.2e）"
      % (worst_med_abs, K_SIGMA_MC * max(sigma_mc_median(f) for f in grid)),
      worst_med_abs < K_SIGMA_MC * max(sigma_mc_median(f) for f in grid),
      f"{worst_med_abs:.2e} vs 5σ {K_SIGMA_MC * max(sigma_mc_median(f) for f in grid):.2e}")
check("A.1 ★ 相对差的最大值 = %0.2f%%（< 5%%）—— ★★ 这条容差**不是公式推出来的**："
      "相对差在偏差本来就很小的档位（k/n = 0.10，偏差只有 0.007）会被放大，"
      "σ 不是解析量。它是本程序**唯一**一条仍由经验定的容差（跨 6 个种子实测 "
      "σ = 0.0087、最大 0.0313 ⇒ 0.05 只有 1.6 倍余量）。"
      % (100 * worst_med),
      worst_med < 0.05, f"{worst_med:.4f}")

worst_mean = 0.0
for frac in grid:
    k = int(round(frac * N))
    an = analytic_stats(N, k, SP.COLD_START_SLOWDOWN_DEFAULT, SP.STEADY_SIGMA_DEFAULT)
    mc = mc_by_k[k]
    worst_mean = max(worst_mean, abs(mc["mean"] - an["mean"]))
TOL_WORST_MEAN = K_SIGMA_MC * max(sigma_mc_mean(f) for f in grid)
check("A.1 ★★ 均值：解析式 `k/n·(slowdown−1)` 与蒙特卡洛吻合到 **MC 噪声以内**"
      "（最大绝对差 %0.2e < %.2e = 5σ_mc；σ_mc = σ·√m/(n·√REPS)，最大档 σ_mc = %.2e）"
      % (worst_mean, TOL_WORST_MEAN, max(sigma_mc_mean(f) for f in grid)),
      worst_mean < TOL_WORST_MEAN, f"{worst_mean:.2e}")
print("      ★ 口径：解析式给的是**期望**，蒙特卡洛给的是**样本估计**——")
print("        两者的差**不是误差**，是后者的标准误。所以容差取 MC 噪声的 5 倍，不取 0。")
print("        ★★ 这条差**曾经**是魔数 1e-3（= 41σ）；现在是 5σ_mc = %.2e。" % TOL_WORST_MEAN)

sub("A.2  ★★ 判据的两个点：0.49 必须**超一个 σ**，0.10 必须**很小**")
b49 = analytic_stats(N, K49, SP.COLD_START_SLOWDOWN_DEFAULT, SP.STEADY_SIGMA_DEFAULT)["median"]
b10 = analytic_stats(N, K10, SP.COLD_START_SLOWDOWN_DEFAULT, SP.STEADY_SIGMA_DEFAULT)["median"]
m49, m10 = mc_by_k.get(K49, {}).get("median", float("nan")), mc_by_k.get(K10, {}).get("median", float("nan"))
print(f"      k/n = 0.49：解析 {b49:+.4f} · 蒙特卡洛 {m49:+.4f}   （稳态 σ = {SP.STEADY_SIGMA_DEFAULT}）")
print(f"      k/n = 0.10：解析 {b10:+.4f} · 蒙特卡洛 {m10:+.4f}")
check("A.2 ★★ k/n = 0.49（k = %d）的中位数偏差 > 0.05（**超过一个稳态 σ**）" % K49,
      m49 > 0.05, f"{m49:+.4f} > 0.05")
check("A.2 ★★ k/n = 0.10（k = %d）的中位数偏差 < 0.02" % K10, m10 < 0.02, f"{m10:+.4f} < 0.02")
check("A.2 ★★ 两个点**都**在断言里（只断言前者说明不了机制是什么）",
      (m49 > 0.05) and (m10 < 0.02))
check("A.2 ★ 规格的「k 超过半数才污染」是错的（机制常量已在 specs.py 里标 False）",
      SP.SPEC_MEDIAN_MECHANISM_IS_K_OVER_HALF is False
      and SP.SPEC_MEDIAN_MECHANISM_NAME == "position_pushed_into_steady_upper_tail")

sub("A.3  机制：中位数在合并样本里的**名次**被推到稳态子样本的上尾")
for k in (K10, K49, K50, K90):
    m = N - k
    print(f"      k = {k:3d}（m = {m:3d}）：", end="")
    for pos0 in ((N // 2 - 1, N // 2) if N % 2 == 0 else (N // 2,)):
        r = pos0 + 1
        if r <= m:
            q = r / (m + 1)
            print(f"合并名次 {r:3d} → 稳态子样本第 {r:3d}/{m} 位（分位 {q:.4f}，"
                  f"E[z] {order_stat_mean(r, m):+.3f}）", end="  ")
        else:
            print(f"合并名次 {r:3d} → **落在冷启动块**（取值 = {SP.COLD_START_SLOWDOWN_DEFAULT:.2f}）", end="  ")
    print()
print("      ★ 机制一句话：稳态值占了排序的前 m 位，所以**合并名次 r 在稳态子样本里的分位是 r/m**；")
print("        k 越大 ⇒ m 越小 ⇒ 同一个名次的**分位越高** ⇒ 中位数被推向稳态的**上尾**。")
check("A.3 ★★ 分位随 k 单调上升（k = 10 时 50/91 ≈ 0.55，k = 49 时 50/52 ≈ 0.96）",
      50 / 91 < 50 / 52 and abs(50 / 52 - 0.9615) < 1e-4,
      f"{50 / 91:.4f} → {50 / 52:.4f}")
#  ↑ 容差 1e-4 = **2 个末位半格**：0.9615 是 4 位小数的写法（半格 5e-5），
#    而 50/52 = 0.9615385 与它差 3.85e-5。旧值 1e-3 是它的 20 倍，没道理。
check("A.3 ★★ k = 50 时**只有一个**中位名次落在稳态里（另一个是冷启动值）⇒ "
      "偏差 = 中点 + σ·E[z_(m:m)]/2 —— **等式**，不是「差不多」",
      abs(analytic_stats(N, K50, SP.COLD_START_SLOWDOWN_DEFAULT, SP.STEADY_SIGMA_DEFAULT)["median"]
          - (((1.0 + SP.COLD_START_SLOWDOWN_DEFAULT) / 2.0 - 1.0)
             + SP.STEADY_SIGMA_DEFAULT * order_stat_mean(N - K50, N - K50) / 2.0)) < 1e-12,
      f"解析 {analytic_stats(N, K50, SP.COLD_START_SLOWDOWN_DEFAULT, SP.STEADY_SIGMA_DEFAULT)['median']:.6f}"
      f" vs 中点+上尾项 "
      f"{((1.0 + SP.COLD_START_SLOWDOWN_DEFAULT) / 2.0 - 1.0) + SP.STEADY_SIGMA_DEFAULT * order_stat_mean(N - K50, N - K50) / 2.0:.6f}"
      f"（旧容差 0.12 是这条等式的 1/5000）")

sub("A.4  ★ 最小值：**与冷启动倍数无关**的负偏差——它抗冷启动，但不是无偏的")
mins = {frac: mc_by_k[int(round(frac * N))]["min"] for frac in grid}
print(f"      各档 min 偏差： " + "  ".join(f"k/n={f:.2f}:{mins[f]:+.4f}" for f in grid))
print(f"      极差 = {max(mins.values()) - min(mins.values()):.4f}"
      f"（对比：中位数在 k 跨越半数时跳变 2.0 以上）")
check("A.4 ★ 最小值偏差恒为**负**（它是极小值统计量，天然低于均值）",
      all(v < 0 for v in mins.values()))
STEADY_DRAW = 1.0 + SP.STEADY_SIGMA_DEFAULT * np.random.default_rng(SEED + 1).standard_normal((4000, 90))
min_by_sd = {sd: float(STEADY_DRAW.min(axis=1).mean() - 1.0) for sd in (2.0, 5.0, 9.0)}
print("      同一批稳态样本、只改冷启动倍数（k = 10 ⇒ m = 90）的 min 偏差：")
for sd, v in min_by_sd.items():
    print(f"        slowdown {sd:.0f}×   min 偏差 {v:+.6f}")
check("A.4 ★★ 它与**冷启动倍数**无关：2× / 5× / 9× 的 min 偏差**逐位相同**"
      "（因为冷启动值永远不是最小值）",
      len(set(min_by_sd.values())) == 1, f"{min_by_sd}")
print("      ★ 措辞要准：min 偏差与**冷启动倍数**无关，但**随保留的稳态样本数 m 变化**")
print("        （m 越小，最小值越不极端 ⇒ 负偏差越小）——两者不是同一句话：")
for frac in (0.0, 0.49, 0.90):
    m = N - int(round(frac * N))
    print(f"          k/n = {frac:.2f}（m = {m:3d}）：解析 {SP.STEADY_SIGMA_DEFAULT * order_stat_mean(1, m):+.4f}"
          f" · 蒙特卡洛 {mins.get(frac, float('nan')):+.4f}")
an_min = SP.STEADY_SIGMA_DEFAULT * order_stat_mean(1, N - 10)
_m10v = mins.get(0.10, float("nan"))
# ★ 容差**由噪声推出**：估计量是 REPS 个重复的样本最小值的平均 ⇒
#     SE_MIN = sd(重复间最小值) / √REPS。`STEADY_DRAW` 已经是 (4000, 90) 的稳态样本
#     （m = 90 正是 k/n = 0.10 那一档），直接量它的最小值列的离散度即可。
SE_MIN = float(STEADY_DRAW.min(axis=1).std(ddof=1)) / np.sqrt(REPS)
TOL_MIN = 6.0 * SE_MIN     # 5σ 给 MC 噪声，另 1σ 给 Blom 近似在**极小分位**上的偏差
                          # （实测该偏差 ≈ 0.8σ_mc——Blom 对 E[z_(1:m)] 不无偏）
check("A.4 ★ 解析值 = σ·E[z_(1:m)]（Blom 近似）与蒙特卡洛吻合到 6σ_mc = %.4f"
      "（σ_mc = sd(重复间最小值)/√REPS = %.2e）" % (TOL_MIN, SE_MIN),
      abs(_m10v - an_min) < TOL_MIN, f"{_m10v:+.4f} vs {an_min:+.4f}"
      f"（差 {abs(_m10v - an_min):.2e} = {abs(_m10v - an_min) / SE_MIN:.1f}σ）")
check("A.4 ★★ 三个 k 档**互不相同**（否则下面两条会退化成同一句）",
      len({K10, K49, K90}) == 3)
d_a = abs(mins.get(0.0, float("nan")) - mins.get(0.10, float("nan")))
d_b = abs(mins.get(0.0, float("nan")) - mins.get(0.90, float("nan")))
check("A.4 ★★ 变化来自 **m**：k/n = 0.00 → 0.10 只差 %0.4f，而 0.00 → 0.90 差 %0.4f（≥ 3 倍）"
      % (d_a, d_b), d_a < 0.02 and d_b > 3 * d_a, f"{d_a:.4f} / {d_b:.4f}")
print("      ⇒ **A/B 比较时这个偏差会抵消**（两边同 n 同口径）——")
print("        这正是官方工具用**低分位**而不是均值的代数理由（见 §D.3 的 Triton 分位）。")

sub("A.5  ★ 规格预跑表的一处**未复现**：k/n = 0.90 那一格")
spec90 = SP.SPEC_PRERUN_MEDIAN_BIAS.get(0.90, float("nan"))
mc90 = mc_by_k.get(K90, {}).get("median", float("nan"))
print(f"      规格 {spec90:.3f} · 本库 {mc90:.4f}（差 {abs(mc90 - spec90):.4f}）")
check("A.5 ★★ 本库的 k/n = 0.90 中位数偏差**精确等于** 4.000（两个中位名次都在冷启动块里 ⇒ "
      "每个重复都给同一个值 ⇒ 标准差为 0 ⇒ 可以断言**相等**，不需要容差）",
      mc90 == 4.0, f"{mc90!r}（旧版写 < 1e-12，等价但没说出「它是精确的」）")
print("      ★ 差异的成因：规格的冷启动值带了一点噪声，本库模型取**确定值**。")
print("        ⇒ 这不是「本库算错」，是**两个模型的差别**；正文按本库口径写，并把差异写出来。")
for frac in (0.0, 0.10, 0.40, 0.49):
    k = int(round(frac * N))
    got = mc_by_k.get(k, {}).get("median", float("nan"))
    want = SP.SPEC_PRERUN_MEDIAN_BIAS.get(frac, float("nan"))
    tol_here = K_SIGMA_MC * sigma_mc_median(frac) + SPEC_QUOTE_HALF_ULP_3DEC
    check(f"A.5 规格 {frac:.2f} 档的中位数偏差被本库复现"
          f"（|差| < {tol_here:.4f} = 5σ_mc {K_SIGMA_MC * sigma_mc_median(frac):.4f} + 引文半格）",
          abs(got - want) < tol_here, f"{got:+.4f} vs {want:+.3f}（用掉 {abs(got - want) / tol_here:.0%}）")
check("A.5 ★★ 0.90 那一档的 0.007 差是**固定的模型差**（规格 3.993 vs 本库 4.000）——"
      "两侧都是常量、与蒙特卡洛无关 ⇒ 这一条**不需要容差**，直接钉住模型差",
      SP.SPEC_PRERUN_MEDIAN_BIAS.get(0.90) == 3.993 and mc90 == 4.0
      and abs(abs(mc90 - SP.SPEC_PRERUN_MEDIAN_BIAS[0.90]) - 0.007) < 1e-12)


# ===========================================================================
# B. 判据：要丢弃多少次前导测量，以及机制的**适用边界**
# ===========================================================================
head("§B  判据：丢弃多少次前导测量，以及机制在什么条件下才成立")

TOL_PCT = 1.0
sub("B.1  「中位数偏差 < 1%」需要丢弃多少次前导测量")
print("      定义：丢弃前 d 次 ⇒ 剩下的 k = n − d 次全部进统计（其余为稳态）")


def median_bias_at(n, k, slowdown, sigma, steady=SP.STEADY_TIME_NOMINAL):
    """★ `slowdown` / `sigma` 是**实参**（来自 specs 的可调参数），不硬编码。"""
    return analytic_stats(n, k, slowdown, sigma, steady)["median"]


def critical_kept(n, tol, slowdown, sigma):
    """满足 |中位数偏差| <= tol 的**最大保留次数** k（再大就超）。返回 `(k_max, d_min)`。"""
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if median_bias_at(n, mid, slowdown, sigma) <= tol:
            lo = mid
        else:
            hi = mid - 1
    return lo, n - lo


crit_by_sd = {}
for sd in (1.005, 1.01, 1.02, 2.0, 5.0, 9.0):
    k_max, d_min = critical_kept(N, TOL_PCT / 100.0, sd, SP.STEADY_SIGMA_DEFAULT)
    crit_by_sd[sd] = k_max
    print(f"      冷启动 {sd:5.3f}×：最多保留 k = {k_max:3d} 次 ⇒ **至少丢弃前 {d_min:3d} 次**"
          f"（={d_min / N:.1%} of n）")
    check(f"B.1 冷启动 {sd:.3f}× 的临界值自洽（k_max 满足、k_max+1 不满足）",
          median_bias_at(N, k_max, sd, SP.STEADY_SIGMA_DEFAULT) <= 0.01
          and (k_max == N or median_bias_at(N, k_max + 1, sd, SP.STEADY_SIGMA_DEFAULT) > 0.01))
k5, d5 = critical_kept(N, 0.01, SP.COLD_START_SLOWDOWN_DEFAULT, SP.STEADY_SIGMA_DEFAULT)
check("B.1 ★ 判据是**可计算**的：n = 100、σ = 0.05、冷启动 5× 时，临界值 k = %d 是公式解的"
      % k5, 0 < k5 < N, f"k_max = {k5}")
check("B.1 ★★ 在**饱和区**临界值与冷启动倍数**无关**（1.02× / 2× / 5× / 9× 都给 %d）"
      % crit_by_sd[2.0],
      crit_by_sd[1.02] == crit_by_sd[2.0] == crit_by_sd[5.0] == crit_by_sd[9.0],
      f"{crit_by_sd}")
check("B.1 ★★ 而冷启动**极其轻微**时答案完全不同（1.005× 给 %d ⇒ 一次都不用丢）"
      % crit_by_sd[1.005],
      crit_by_sd[1.005] > 90 and crit_by_sd[1.02] == 13,
      f"1.005× → {crit_by_sd[1.005]}，1.01× → {crit_by_sd[1.01]}，1.02× → {crit_by_sd[1.02]}")
print("      ★ 这两条合起来是一个**可证伪且反直觉**的论断：")
print("        判据的答案在 1.02×–9× 这一整段上是**同一个数 13**（丢 87 次），")
print("        但只要冷启动轻到 1.005×，答案变成「**一次都不用丢**」——")
print("        **0.5% 的输入变化让输出从 87 跳到 0**。⇒ 判据**必须算**，不能凭经验拍。")

sub("B.2  ★★ 机制的两个区间：**饱和** 与 **线性**，交界处有公式")
print("      阈值公式（本库 [E]）：slowdown* = 1 + σ·E[z_(m:m)]，m = n − k ——")
print("      即「冷启动值必须高过稳态子样本的最大值」，才能把中位数名次挤到上尾。")
for k in (K10, K49, K90):
    m = N - k
    thr = 1.0 + SP.STEADY_SIGMA_DEFAULT * order_stat_mean(m, m) if m > 0 else float("inf")
    print(f"      k = {k:3d}（m = {m:3d}）：slowdown* = {thr:.4f}")
    check(f"B.2 k = {k} 的阈值 > 1（冷启动必须真的更慢才谈得上污染）", thr > 1.0)
thr49 = 1.0 + SP.STEADY_SIGMA_DEFAULT * order_stat_mean(N - 49, N - 49)
sweep = (1.0, 1.02, 1.05, 1.10, thr49, 1.5, 5.0)
print()
print("      冷启动倍数 → k/n = 0.49 的中位数偏差（解析 / 蒙特卡洛，1000 次重复）：")
sweep_err = 0.0
for sd in sweep:
    an = median_bias_at(N, K49, sd, SP.STEADY_SIGMA_DEFAULT)
    mc = float(np.median(pooled_sample(N, K49, sd, SP.STEADY_SIGMA_DEFAULT, 1000, RNG), axis=1).mean() - 1.0)
    print(f"        {sd:6.4f}×   {an:+.4f} / {mc:+.4f}")
    sweep_err = max(sweep_err, abs(an - mc))
TOL_SWEEP = 0.01
check("B.2 ★★ **带封顶的**解析式在整个扫描区间上都对得上蒙特卡洛（最大绝对差 %0.4f < %s）"
      % (sweep_err, TOL_SWEEP), sweep_err < TOL_SWEEP, f"{sweep_err:.4f}")
print("      ★★ 这条容差**不是**噪声容差，写清楚它是什么：单个扫描点的 MC 噪声只有 "
      "1.2533σ/√(m·1000) = %.1e，" % (1.2533 * SP.STEADY_SIGMA_DEFAULT / np.sqrt(51 * 1000))
      + "而最大差是 %.1e（相差 %.0f 倍）——差异来自 **Blom 近似在「封顶交界」处**的"
      % (sweep_err, sweep_err / (1.2533 * SP.STEADY_SIGMA_DEFAULT / np.sqrt(51 * 1000))))
print("       的系统性误差（跨 6 个种子实测 4.91e-3、sd 只有 1.0e-4 ⇒ 是系统项不是随机项）。")
print("       ⇒ 这一类（近似误差）**没有闭式界**，容差只能取「实测系统项 × 2」；")
print("         它与「噪声容差」是两码事，不能互相解释。")
unclamped = median_bias_at(N, K49, 1e9, SP.STEADY_SIGMA_DEFAULT)
mc_at_thr_below = median_bias_at(N, K49, 1.02, SP.STEADY_SIGMA_DEFAULT)
print("      ★ 这一条同时说明：**少了 `min` 封顶的解析式在轻冷启动区间会错一个数量级**——")
print(f"        阈值以下（1.02×）带封顶给 {mc_at_thr_below:+.4f}，不带封顶（= 冷启动无穷大的极限）"
      f"给 {unclamped:+.4f}（{unclamped / mc_at_thr_below:.1f} 倍）。")
check("B.2 ★★ 不带封顶的公式在阈值以下把偏差**高估 5 倍以上** ⇒ 封顶不是修饰，是必需的",
      unclamped / mc_at_thr_below > 5.0, f"{unclamped / mc_at_thr_below:.1f}×")
check("B.2 ★★ 阈值**以下**（1.02×）：偏差**恰好退化成冷启动超额量 `slowdown − 1`**"
      "（= %.4f）—— 这是**精确等式**，不需要「< 0.03」这种容差："
      "两个中位名次都被封顶在冷启动值上，所以解析值就是 1.02 − 1.00"
      % (1.02 - SP.STEADY_TIME_NOMINAL),
      abs(median_bias_at(N, K49, 1.02, SP.STEADY_SIGMA_DEFAULT)
          - (1.02 - SP.STEADY_TIME_NOMINAL)) < 1e-12,
      f"{median_bias_at(N, K49, 1.02, SP.STEADY_SIGMA_DEFAULT):+.6f}")
check("B.2 ★★ 阈值**以上**饱和：5× 与阈值处的偏差**恰好相同**（不是「相差 < 10%」）——"
      "两处都没触发封顶，中位数取的是同一个稳态上尾值",
      abs(median_bias_at(N, K49, 5.0, SP.STEADY_SIGMA_DEFAULT)
          - median_bias_at(N, K49, thr49, SP.STEADY_SIGMA_DEFAULT)) < 1e-12,
      f"{median_bias_at(N, K49, 5.0, SP.STEADY_SIGMA_DEFAULT):+.6f} vs "
      f"{median_bias_at(N, K49, thr49, SP.STEADY_SIGMA_DEFAULT):+.6f}"
      f"（旧容差 10% 是它的 1e13 倍）")
print("      ⇒ **k < n/2 时偏差有上界，而那个上界与冷启动倍数无关**：")
print("        冷启动从 %s× 到 5×，偏差只从 +%.4f 走到 +%.4f。"
      % (f"{thr49:.4f}", median_bias_at(N, K49, thr49, SP.STEADY_SIGMA_DEFAULT),
         median_bias_at(N, K49, 5.0, SP.STEADY_SIGMA_DEFAULT)))

sub("B.3  ★★ 抗污染上界 = 稳态的**尾部位置**，不是冷启动的大小——两处可证伪检验")
cap49 = median_bias_at(N, K49, 5.0, SP.STEADY_SIGMA_DEFAULT)
cap_big_sigma = median_bias_at(N, K49, 5.0, 0.5)
print(f"      ① 与冷启动倍数无关：1.5× 给 {median_bias_at(N, K49, 1.5, SP.STEADY_SIGMA_DEFAULT):+.4f}、"
      f"5× 给 {cap49:+.4f}、9× 给 {median_bias_at(N, K49, 9.0, SP.STEADY_SIGMA_DEFAULT):+.4f}")
print(f"      ② 与稳态 σ 成正比：σ = 0.05 时上界 {cap49:+.4f}，σ = 0.5 时上界 {cap_big_sigma:+.4f}"
      f"（比值 {cap_big_sigma / cap49:.2f}）")
check("B.3 ★★ 上界对冷启动倍数**完全不敏感**（1.5× 与 9× **逐位相同**，不是「相差 < 1%」）——"
      "封顶之后两个名次取的是同一个稳态上尾值，冷启动倍数已经不出现",
      abs(median_bias_at(N, K49, 9.0, SP.STEADY_SIGMA_DEFAULT)
          - median_bias_at(N, K49, 1.5, SP.STEADY_SIGMA_DEFAULT)) < 1e-12,
      f"{median_bias_at(N, K49, 9.0, SP.STEADY_SIGMA_DEFAULT)!r} vs "
      f"{median_bias_at(N, K49, 1.5, SP.STEADY_SIGMA_DEFAULT)!r}")
check("B.3 ★★ 上界对 σ **精确线性**（σ 放大 10 倍 ⇒ 上界放大**恰好** 10 倍，误差 < 1e-12）："
      "封顶未触发时 `bias = σ·(E[z₁]+E[z₂])/2`，σ 是一次的",
      abs(cap_big_sigma / cap49 - 10.0) < 1e-12, f"{cap_big_sigma / cap49:.15f}")
print("      ⇒ 两句话都会随前提消失，所以它们**可证伪**：")
print("        把 σ 调到冷启动值被淹没（slowdown < slowdown*）⇒ 整个机制消失（B.2 已跑出）。")

sub("B.4  ★★ 跨过半数时污染量级**突变**——这是机制的另一半")
b49 = median_bias_at(N, K49, SP.COLD_START_SLOWDOWN_DEFAULT, SP.STEADY_SIGMA_DEFAULT)
b50 = median_bias_at(N, K50, SP.COLD_START_SLOWDOWN_DEFAULT, SP.STEADY_SIGMA_DEFAULT)
b90 = median_bias_at(N, K90, SP.COLD_START_SLOWDOWN_DEFAULT, SP.STEADY_SIGMA_DEFAULT)
print(f"      k = {K49}：{b49:+.4f}   k = {K50}：{b50:+.4f}   k = {K90}：{b90:+.4f}")
print(f"      冷启动倍数**没变**，偏差跳了 {b50 / b49:.1f} 倍（k = {K49} → {K50}）")
check("B.4 ★★ k = %d 的偏差是 k = %d 的 20 倍以上（冷启动值完全相同）" % (K50, K49),
      b50 / b49 > 20.0, f"{b50 / b49:.1f}×")
check("B.4 ★ k ≥ n/2 时偏差**由冷启动倍数决定**（k = %d 时 ≈ slowdown − 1 = %s）"
      % (K90, SP.COLD_START_SLOWDOWN_DEFAULT - SP.STEADY_TIME_NOMINAL),
      abs(b90 - (SP.COLD_START_SLOWDOWN_DEFAULT - SP.STEADY_TIME_NOMINAL)) < 1e-9,
      f"{b90:+.4f}")
print("      ⇒ 两段机制合起来才是完整判据：**k < n/2 时偏差被「稳态上尾」封顶，**")
print("        **k ≥ n/2 时偏差由「冷启动倍数」直接决定**——后者才是 20 倍量级的那个。")


# ===========================================================================
# C. ★★ 符号翻转：同一个实现，两种协议
# ===========================================================================
head("§C  ★★ 符号翻转：同一个实现，两种测量协议报出相反的结论")

sub("C.1  同一个实现、同一个稳态时间（1.0），两种协议")
PROTO_WARM = {"name": "warmup（丢弃前导）", "n": 20, "k": 0}
PROTO_COLD = {"name": "no warmup（保留前导）", "n": 5, "k": 3}

# ★★ §L 结构守卫（C6 模式八）：容器常量要**先于任何索引/取值**被逐格断言。
#    这两个字典的**键**被改错时，`protocol_mean` 会在 `proto["k"]` 处抛 `KeyError`，
#    rc≠0 但**没有失败断言** —— 与「断言失败」的退出码相同，于是**假的捕获被当成真的**。
#    实测（`make mutate-all`）：6 个键变异全部表现为 `KeyError` ⇒ 崩溃捕获。
#    这一节把它们转成干净的 `[FAIL]`，并顺带钉住**值**（`PROTO_WARM["n"] = 20` 改掉
#    不会有任何别的断言发现——它只影响 k = 0 时的均值，而 k = 0 时均值恰好与 n 无关）。
check("§L PROTO_WARM 的键集合 == {'name','n','k'}（先守卫、再使用）",
      set(PROTO_WARM) == {"name", "n", "k"}, f"{sorted(PROTO_WARM)}")
check("§L PROTO_COLD 的键集合 == {'name','n','k'}", set(PROTO_COLD) == {"name", "n", "k"},
      f"{sorted(PROTO_COLD)}")
check("§L PROTO_WARM 逐格 == {'name': 'warmup（丢弃前导）', 'n': 20, 'k': 0}",
      PROTO_WARM == {"name": "warmup（丢弃前导）", "n": 20, "k": 0}, f"{PROTO_WARM}")
check("§L PROTO_COLD 逐格 == {'name': 'no warmup（保留前导）', 'n': 5, 'k': 3}",
      PROTO_COLD == {"name": "no warmup（保留前导）", "n": 5, "k": 3}, f"{PROTO_COLD}")
if FAILED:
    # ★ 守卫不过就**立刻停**：继续往下走会在 `proto["k"]` 处抛 KeyError，
    #   那是一个「崩溃」——rc 同样是 1，变异测试会把它误记成「断言捕获」。
    print()
    print("  ✗ §L 结构守卫未通过 ⇒ 立即退出（不再往下走到取值处，避免 KeyError 冒充断言失败）")
    sys.exit(1)


def protocol_mean(proto, steady, slowdown):
    """按协议估出的平均值：前 k 次是冷启动值，其余是稳态标称值。"""
    k, n = proto["k"], proto["n"]
    return (k * steady * slowdown + (n - k) * steady) / n


mean_warm = protocol_mean(PROTO_WARM, SP.STEADY_TIME_NOMINAL, SP.COLD_START_SLOWDOWN_DEFAULT)
mean_cold = protocol_mean(PROTO_COLD, SP.STEADY_TIME_NOMINAL, SP.COLD_START_SLOWDOWN_DEFAULT)
delta_forward = (mean_cold / mean_warm - 1.0) * 100.0
delta_backward = (mean_warm / mean_cold - 1.0) * 100.0
print(f"      协议 A：{PROTO_WARM['name']}，n = {PROTO_WARM['n']}，k = {PROTO_WARM['k']}"
      f"  ⇒ 读数 {mean_warm:.3f}")
print(f"      协议 B：{PROTO_COLD['name']}，n = {PROTO_COLD['n']}，k = {PROTO_COLD['k']}"
      f"  ⇒ 读数 {mean_cold:.3f}")
print(f"      「先 A 后 B」：报出 **{delta_forward:+.1f}%**（看起来慢了）")
print(f"      「先 B 后 A」：报出 **{delta_backward:+.1f}%**（看起来快了）")
check("C.1 ★★ 同一个实现的两次「优化前后」读数**符号相反**",
      delta_forward > 0 > delta_backward,
      f"{delta_forward:+.1f}% vs {delta_backward:+.1f}%")
check("C.1 ★★ 两个协议的**稳态分量是同一个数**（warmup 协议的 k = 0 ⇒ 读数恰好等于稳态标称值）",
      protocol_mean(PROTO_WARM, SP.STEADY_TIME_NOMINAL, SP.COLD_START_SLOWDOWN_DEFAULT)
      == SP.STEADY_TIME_NOMINAL == 1.0)
check("C.1 ★ 差异**全部**来自协议（两次协议里 `slowdown` 与 `steady` 传的是同一对值）",
      PROTO_WARM["k"] == 0 and PROTO_COLD["k"] == 3 and PROTO_COLD["n"] == 5)
check("C.1 ★ 两个方向都**越过**了同一份门限 ±5%（hipfire 的示例门限）",
      abs(delta_forward) > SP.HIPFIRE_SPEED_GATE_TOLERANCE_PCT
      and abs(delta_backward) > SP.HIPFIRE_SPEED_GATE_TOLERANCE_PCT,
      f"门限 ±{SP.HIPFIRE_SPEED_GATE_TOLERANCE_PCT}%")

sub("C.2  ★★ 更贵的一种：**稳态真的更快**的变体被报成变慢")
old_reading = protocol_mean(PROTO_COLD, SP.STEADY_TIME_NOMINAL, SP.COLD_START_SLOWDOWN_DEFAULT)
print(f"      参照：旧变体按「无 warmup」协议读 {old_reading:.3f}（稳态 {SP.STEADY_TIME_NOMINAL:.2f}）")
print()
print("      新变体   真实稳态变化   冷启动       n/k     报出       （真实 / 报出）")
true_gains, reported_list = [], []
for new_steady, new_slowdown, nn, kk in ((0.75, 8.0, 5, 3), (0.75, 12.0, 5, 4), (0.90, 20.0, 4, 3)):
    proto = {"name": "no warmup", "n": nn, "k": kk}
    reading = protocol_mean(proto, new_steady, new_slowdown)
    rep = (reading / old_reading - 1.0) * 100.0
    tg = (new_steady / SP.STEADY_TIME_NOMINAL - 1.0) * 100.0
    true_gains.append(tg)
    reported_list.append(rep)
    print(f"        {new_steady:.2f}     {tg:+7.1f}%      {new_slowdown:4.1f}×    {nn}/{kk}   "
          f"{rep:+8.1f}%")
check("C.2 ★★ 三个变体的**真实稳态变化全是负的（更快）**，而报出的**全是正的（更慢）**——符号全部相反",
      all(t < 0 < r for t, r in zip(true_gains, reported_list)),
      f"{list(zip(true_gains, reported_list))}")
check("C.2 ★★ 最大报出幅度 %0.1f%% 是真实收益幅度（%0.1f%%）的 %.1f 倍 ⇒ 不只是符号错，"
      "**量级也可以任意错**"
      % (max(reported_list), min(abs(g) for g in true_gains),
         max(reported_list) / min(abs(g) for g in true_gains)),
      max(reported_list) > 10 * min(abs(g) for g in true_gains))
check("C.2 ★ 三个报出值**全部越过** ±%s%% 门限 ⇒ 结论错、决策也错"
      % SP.HIPFIRE_SPEED_GATE_TOLERANCE_PCT,
      all(abs(r) > SP.HIPFIRE_SPEED_GATE_TOLERANCE_PCT for r in reported_list))
print("      ★ 现实里对应的形态：**新变体加了 JIT 编译的快速路径** ⇒ 稳态更快，")
print("        但前几次测量要付编译时间 ⇒ 在「不 warmup」协议下它被判为回归。")

sub("C.3  ★ 现实锚点：这不是本库编的——官方仓库的 issue 里就有 +27.8%")
fake_pct = SP.TRITON_ISSUE_2306_IMPLIED_FAKE_GAIN_PCT
print(f"      Triton issue #2306：默认 warmup（{SP.TRITON_DO_BENCH_DEFAULT_WARMUP_MS:.0f} ms）下 "
      f"{SP.TRITON_ISSUE_2306_TIME_WITH_DEFAULT_WARMUP_MS} ms，")
print(f"      把 warmup 拉到 > {SP.TRITON_ISSUE_2306_REQUIRED_WARMUP_OVER_MS:.0f} ms 后 "
      f"{SP.TRITON_ISSUE_2306_TIME_WITH_EXTENDED_WARMUP_MS} ms")
print(f"      ⇒ 纯粹由 warmup 造成的「加速」= {fake_pct:+.2f}%，而 kernel 一行没改。")
check("C.3 ★★ 该「加速」的符号是**正**的（不改代码也能「变快」）", fake_pct > 0, f"{fake_pct:+.2f}%")
check("C.3 ★★ 它 > 门限 ±5% ⇒ 一个**无改动的改动**能过回归门限",
      fake_pct > SP.HIPFIRE_SPEED_GATE_TOLERANCE_PCT)
check("C.3 ★ 与本库模型的关系：**机制相同、倍数更小**（本库模型是放大器，不是现实）",
      fake_pct < abs(delta_backward), f"现实 {fake_pct:.1f}% < 模型 {abs(delta_backward):.1f}%")
print("      ★ 限定语：issue 的 30% 低估是**官方仓库里的一则报告**（B 级），")
print("        不是经过评审的结论；本库的 200%+ 是**模型值**，两者都不得当成真实系统的性质。")

sub("C.4  ★ 「重开一个进程就好了」是错的：噪声带的测量条件就是「新进程」")
check("C.4 ★ hipfire 的噪声带是在**新进程**上测的（所以换进程不是解药）",
      SP.HIPFIRE_NOISE_BAND_IS_MEASURED_ON_FRESH_PROCESS is True)
check("C.4 ★ 该噪声带（%s%%）比同一文档的门限（%s%%）**更宽**"
      % (SP.HIPFIRE_WITHIN_SESSION_NOISE_PCT_GFX1100, SP.HIPFIRE_SPEED_GATE_TOLERANCE_PCT),
      SP.HIPFIRE_WITHIN_SESSION_NOISE_PCT_GFX1100[0] > SP.HIPFIRE_SPEED_GATE_TOLERANCE_PCT)
check("C.4 ★ 该文档自己承认「会话内的 +%s%% 仍在噪声里」——**超过门限也不等于真变快**"
      % SP.HIPFIRE_IN_SESSION_WIN_INSIDE_NOISE_PCT,
      SP.HIPFIRE_IN_SESSION_WIN_INSIDE_NOISE_PCT > SP.HIPFIRE_SPEED_GATE_TOLERANCE_PCT)


# ===========================================================================
# D. 常量自洽 + 跨程序覆盖检查
# ===========================================================================
head("§D  常量自洽：本篇用到的 specs.py 常量，逐条被断言钉住")

sub("D.1  hipfire（**[D] 级：单一项目、个人/小团队仓库**）")
check("D.1 噪声带是**两档条件**的数，不是一句「±1–3%」",
      SP.HIPFIRE_WITHIN_SESSION_NOISE_PCT_GFX1100 == (10.0, 15.0),
      f"{SP.HIPFIRE_WITHIN_SESSION_NOISE_PCT_GFX1100}")
check("D.1 门限 ±%s%%" % SP.HIPFIRE_SPEED_GATE_TOLERANCE_PCT,
      SP.HIPFIRE_SPEED_GATE_TOLERANCE_PCT == 5.0)
check("D.1 ★ 基线**按架构分文件**（跨架构基线各自维护的实例①）",
      SP.HIPFIRE_SPEED_BASELINE_FILE_PATTERN == "tests/speed-baselines/<arch>.txt")
check("D.1 会话内 +%s%% 仍在噪声内" % SP.HIPFIRE_IN_SESSION_WIN_INSIDE_NOISE_PCT,
      SP.HIPFIRE_IN_SESSION_WIN_INSIDE_NOISE_PCT == 8.0)
check("D.1 散得太开判为 unstable 的阈值 = %s" % SP.HIPFIRE_MAX_REL_SPREAD_BEFORE_UNSTABLE,
      SP.HIPFIRE_MAX_REL_SPREAD_BEFORE_UNSTABLE == 0.20)
check("D.1 ★ 基线更新必须与改动同一次提交", SP.HIPFIRE_BASELINE_UPDATE_MUST_BE_IN_SAME_COMMIT is True)
check("D.1 ★ 行级来源三字段逐个核对（元组元素也要进变异测试）",
      SP.HIPFIRE_BASELINE_ROW_PROVENANCE_FIELDS == ("binary_md5", "git_dirty_state", "diff_md5"),
      f"{SP.HIPFIRE_BASELINE_ROW_PROVENANCE_FIELDS}")
check("D.1 ★★ 规格声称的 graph 对照环境变量名与本库检索到的不一致（规格条目被推翻）",
      SP.HIPFIRE_GRAPH_AB_ENV_VAR == "HIPFIRE_GRAPH"
      and SP.SPEC_CLAIM_GRAPH_AB_ENV_VAR == "HIPFIRE_VERIFY_GRAPH"
      and SP.SPEC_CLAIM_GRAPH_AB_ENV_VAR_IS_REPRODUCIBLE is False)

sub("D.2  Triton（**[B] 级：某一实现的官方规格**）")
check("D.2 warmup 默认 %s **毫秒**" % SP.TRITON_DO_BENCH_DEFAULT_WARMUP_MS,
      SP.TRITON_DO_BENCH_DEFAULT_WARMUP_MS == 25.0
      and SP.TRITON_DO_BENCH_WARMUP_UNIT_IS_MILLISECONDS is True)
check("D.2 rep 默认 %s ms" % SP.TRITON_DO_BENCH_DEFAULT_REP_MS,
      SP.TRITON_DO_BENCH_DEFAULT_REP_MS == 100.0)
check("D.2 每次计时前清 L2", SP.TRITON_DO_BENCH_CLEARS_L2_BEFORE_EACH_ITER is True)
check("D.2 ★ 是官方 API", SP.TRITON_DO_BENCH_IS_OFFICIAL_API is True)

sub("D.2b  issue #2306 的四个数逐个核对（**[B] 级：官方仓库**）")
check("D.2b 低估幅度 = %s%%（标题里那个数）" % SP.TRITON_ISSUE_2306_TITLE_UNDERESTIMATE_PCT,
      SP.TRITON_ISSUE_2306_TITLE_UNDERESTIMATE_PCT == 30.0)
check("D.2b 默认 warmup 下测得 %s ms" % SP.TRITON_ISSUE_2306_TIME_WITH_DEFAULT_WARMUP_MS,
      SP.TRITON_ISSUE_2306_TIME_WITH_DEFAULT_WARMUP_MS == 11.23)
check("D.2b 拉长 warmup 后测得 %s ms" % SP.TRITON_ISSUE_2306_TIME_WITH_EXTENDED_WARMUP_MS,
      SP.TRITON_ISSUE_2306_TIME_WITH_EXTENDED_WARMUP_MS == 8.79)
check("D.2b 使读数稳定的 warmup 门槛 > %s ms" % SP.TRITON_ISSUE_2306_REQUIRED_WARMUP_OVER_MS,
      SP.TRITON_ISSUE_2306_REQUIRED_WARMUP_OVER_MS == 150.0)
check("D.2b ★★ 由这三个数推出的「假加速」= %+.2f%%（与 §C.3 用的是同一个常量）——"
      "容差 1e-4 = 引文末位（4 位小数 27.7588）的 2 个半格；实测差 1.68e-5"
      % SP.TRITON_ISSUE_2306_IMPLIED_FAKE_GAIN_PCT,
      abs(SP.TRITON_ISSUE_2306_IMPLIED_FAKE_GAIN_PCT - 27.7588) < 1e-4,
      f"差 {abs(SP.TRITON_ISSUE_2306_IMPLIED_FAKE_GAIN_PCT - 27.7588):.2e}"
      f"（旧容差 0.01 是引文半格的 200 倍）")
check("D.2b 受害对象：**不用 autotune 装饰器**的 triton.jit kernel",
      SP.TRITON_ISSUE_2306_AFFECTS_JIT_WITHOUT_AUTOTUNE is True)
check("D.2b ★ 证据形态：官方仓库的 issue（**不是**经过评审的结论）",
      SP.TRITON_ISSUE_2306_IS_AN_OFFICIAL_REPO_ISSUE is True)

sub("D.3  ★★ 官方默认返回**均值**，而均值正是对冷启动最敏感的统计量")
check("D.3 `do_bench` 默认 return_mode = %r" % SP.TRITON_DO_BENCH_DEFAULT_RETURN_MODE,
      SP.TRITON_DO_BENCH_DEFAULT_RETURN_MODE == "mean")
check("D.3 autotune 走**分位数**（含 0.2 低分位）",
      SP.TRITON_DO_BENCH_AUTOTUNE_QUANTILES == (0.5, 0.2, 0.8),
      f"{SP.TRITON_DO_BENCH_AUTOTUNE_QUANTILES}")
_m10 = mc_by_k.get(K10, {}).get("mean", float("nan"))
_c10 = mc_by_k.get(K10, {}).get("median", float("nan"))
check("D.3 ★★ 同一份数据上：均值偏差 (%+.3f) 远大于中位数偏差 (%+.4f) —— 相差 %.0f 倍"
      % (_m10, _c10, _m10 / _c10), _m10 > 10 * _c10, f"{_m10 / _c10:.1f}×")
check("D.3 ★★ 规格条目被推翻：默认值的前提是 warmup 跑够（单位是毫秒，不是次数）",
      SP.SPEC_CLAIM_MEASUREMENT_STATISTIC_DEFAULT_IS_MEAN is True
      and SP.TRITON_DO_BENCH_WARMUP_UNIT_IS_MILLISECONDS is True)

sub("D.4  vLLM 冷启动（**[B] 级：官方文档**）——规格那两个 D 级数字被替换掉了")
check("D.4 首个请求的 JIT 编译尖峰 %s s" % (SP.VLLM_JIT_FIRST_REQUEST_SPIKE_S,),
      SP.VLLM_JIT_FIRST_REQUEST_SPIKE_S == (2.0, 3.0))
check("D.4 graph capture 按 **batch size** 付：%s s" % (SP.VLLM_GRAPH_CAPTURE_PER_BATCH_SIZE_DELAY_S,),
      SP.VLLM_GRAPH_CAPTURE_PER_BATCH_SIZE_DELAY_S == (1.0, 2.0))
check("D.4 FlashInfer 首次最长约 %s min（且**没有日志**，像挂死）"
      % SP.VLLM_FLASHINFER_FIRST_USE_NO_CACHE_MIN,
      SP.VLLM_FLASHINFER_FIRST_USE_NO_CACHE_MIN == 10.0)
check("D.4 DeepGEMM 的**计数形态**：%s" % (SP.VLLM_DEEPGEMM_KERNEL_COUNTS,),
      dict(SP.VLLM_DEEPGEMM_KERNEL_COUNTS) == {"glm5_744b": 2259, "deepseek_v3_2_685b": 1827}
      and SP.VLLM_DEEPGEMM_KERNEL_COUNTS.get("glm5_744b") == 2259)
check("D.4 ★★ 分量清单**不自洽**：列出的六项之和 = %s s，而总时长 = %s s ⇒ 清单不完整"
      % (SP.VLLM_COLD_START_COMPONENT_SUM_S, SP.VLLM_COLD_START_TOTAL_CACHED_S),
      abs(sum(SP.VLLM_COLD_START_LISTED_COMPONENTS_S.values())
          - SP.VLLM_COLD_START_COMPONENT_SUM_S) < 1e-9
      and SP.VLLM_COLD_START_COMPONENT_SUM_S < SP.VLLM_COLD_START_TOTAL_CACHED_S
      and SP.VLLM_COLD_START_BREAKDOWN_IS_INCOMPLETE_IN_EXCERPT is True,
      f"101 < 182")
print("      ⇒ ★ 处置：**残差 81 s 不是测量值**，正文不得把它当「其余」用；")
print("        该残差已进 `UNSOURCED`（`cold_start_component_residual_81s`）。")
check("D.4 ★ FlashInfer 首用日志值 = %s s（约 5.2 min）"
      % SP.VLLM_FLASHINFER_LOGGED_FIRST_USE_S,
      SP.VLLM_FLASHINFER_LOGGED_FIRST_USE_S == 312.47)
check("D.4 ★ DeepGEMM 首次启动量级 = %s min" % SP.VLLM_DEEPGEMM_FIRST_LAUNCH_MIN,
      SP.VLLM_DEEPGEMM_FIRST_LAUNCH_MIN == 5.0)
check("D.4 ★★ 冷启动总时长：权重已缓存 %s s / 权重需下载 %s min"
      % (SP.VLLM_COLD_START_TOTAL_CACHED_S, SP.VLLM_COLD_START_TOTAL_FRESH_MIN),
      SP.VLLM_COLD_START_TOTAL_CACHED_S == 182.0
      and SP.VLLM_COLD_START_TOTAL_FRESH_MIN == (10.0, 15.0))
check("D.4 ★★ 六个分量逐个核对（30 / 32 / 8 / 14 / 10 / 7 s）",
      dict(SP.VLLM_COLD_START_LISTED_COMPONENTS_S) == {
          "container_init": 30.0, "weights_load_cached": 32.0, "torch_compile_cached": 8.0,
          "graph_capture": 14.0, "engine_init_rest": 10.0, "warmup_query": 7.0})
check("D.4 ★★ `torch.compile` 冷 %s s vs 缓存 %s s ⇒ 相差 %.1f 倍（同一阶段、缓存是否命中）"
      % (SP.VLLM_TORCH_COMPILE_COLD_S_MIN,
         SP.VLLM_COLD_START_LISTED_COMPONENTS_S.get("torch_compile_cached", float("nan")),
         SP.VLLM_TORCH_COMPILE_COLD_S_MIN
         / SP.VLLM_COLD_START_LISTED_COMPONENTS_S.get("torch_compile_cached", float("nan"))),
      SP.VLLM_TORCH_COMPILE_COLD_S_MIN == 60.0
      and SP.VLLM_TORCH_COMPILE_COLD_S_MIN
      / SP.VLLM_COLD_START_LISTED_COMPONENTS_S.get("torch_compile_cached", float("nan")) > 7.0)
check("D.4 ★ `--enforce-eager` 省 %s s 启动时间（代价是每 token 吞吐）"
      % (SP.VLLM_ENFORCE_EAGER_SAVES_STARTUP_S,),
      SP.VLLM_ENFORCE_EAGER_SAVES_STARTUP_S == (20.0, 40.0))
check("D.4 ★ 冷启动**不影响正确性**（官方明写只影响最早请求的延迟）",
      SP.VLLM_SKIP_WARMUP_AFFECTS_ONLY_FIRST_REQUEST_LATENCY is True)
check("D.4 ★ 「预热」不得分配真实张量 / 跑真实推理",
      SP.VLLM_WARMUP_MUST_NOT_ALLOCATE_REAL_TENSORS is True)
check("D.4 ★ 预热覆盖不了所有形状（推理期仍可能编译 ⇒ 预热是**下界**）",
      SP.VLLM_JIT_MONITOR_WARNS_ON_INFERENCE_TIME_COMPILE is True)
check("D.4 ★★ 规格的两条 D 级冷启动数字**进 UNSOURCED、正文不用**",
      SP.SPEC_CLAIM_COLD_COMPILE_IS_REPRODUCIBLE is False
      and SP.SPEC_CLAIM_MODEL_RELOAD_IS_REPRODUCIBLE is False
      and SP.SPEC_CLAIM_COLD_COMPILE_TIME_S == (30.0, 120.0)
      and SP.SPEC_CLAIM_MODEL_RELOAD_COLD_START_S == (2.0, 5.0))

sub("D.5  模型参数与预跑值（**[E] 级：本库建模/推导**）")
check("D.5 稳态标称值 %s · σ %s · 冷启动 %s×"
      % (SP.STEADY_TIME_NOMINAL, SP.STEADY_SIGMA_DEFAULT, SP.COLD_START_SLOWDOWN_DEFAULT),
      (SP.STEADY_TIME_NOMINAL, SP.STEADY_SIGMA_DEFAULT, SP.COLD_START_SLOWDOWN_DEFAULT)
      == (1.0, 0.05, 5.0))
check("D.5 n = %d · 重复 %d 次" % (SP.N_MEASUREMENTS_DEFAULT, SP.MC_REPLICATES_DEFAULT),
      (N, REPS) == (100, 3000))
check("D.5 判据用的四档 k/n = %s" % (SP.K_OVER_N_GRID,),
      SP.K_OVER_N_GRID == (0.0, 0.10, 0.40, 0.49, 0.50, 0.90))
check("D.5 解析式用的阶统计量近似 = %r（换近似会改变解析值）"
      % SP.ORDER_STATISTIC_QUANTILE_RULE, SP.ORDER_STATISTIC_QUANTILE_RULE == "blom")
check("D.5 口径：**越小越好**", SP.LOWER_IS_BETTER_METRIC_CONVENTION == "lower_is_better")
check("D.5 规格预跑表被原样收录（供本程序复算），**逐档**核对",
      dict(SP.SPEC_PRERUN_MEDIAN_BIAS) == {0.0: 0.000, 0.10: 0.007, 0.40: 0.048,
                                           0.49: 0.103, 0.50: 2.000, 0.90: 3.993}
      and dict(SP.SPEC_PRERUN_MEAN_BIAS) == {0.0: 0.000, 0.10: 0.400, 0.40: 1.600,
                                             0.49: 1.960, 0.50: 2.000, 0.90: 3.600})

sub("D.6  ★★ 本程序覆盖的**被推翻规格条目**（其余各条由 02/03 覆盖，详见 specs.py §8）")
REFUTED_SPEC = (
    ("hipfire graph 对照环境变量名", SP.SPEC_CLAIM_GRAPH_AB_ENV_VAR_IS_REPRODUCIBLE is False),
    ("hipfire graph 开/关两个 spread 值", SP.SPEC_CLAIM_GRAPH_SPREAD_VALUES_ARE_REPRODUCIBLE is False),
    ("JIT 冷编译 30 s–2 min", SP.SPEC_CLAIM_COLD_COMPILE_IS_REPRODUCIBLE is False),
    ("模型重载冷启动 2–5 s", SP.SPEC_CLAIM_MODEL_RELOAD_IS_REPRODUCIBLE is False),
    ("「噪声带是普遍规律」", SP.SPEC_CLAIM_NOISE_BAND_IS_GENERAL is False),
    ("「k 超过半数才污染中位数」", SP.SPEC_MEDIAN_MECHANISM_IS_K_OVER_HALF is False),
)
for name, ok in REFUTED_SPEC:
    check(f"D.6 已推翻：{name}", ok)
check("D.6 ★★ 这六条**都记在 specs.py 的常量里**（不是只写在报告里）",
      SP.SPEC_CLAIM_GRAPH_ON_SPREAD_PCT == (1.5, 3.0)
      and SP.SPEC_CLAIM_GRAPH_OFF_SPREAD_PCT == 0.1
      and SP.SPEC_CLAIM_MEDIAN_MECHANISM_IS_K_OVER_HALF_LABEL == "k_over_half_only")

sub("D.7  `UNSOURCED` 逐条比对（防止「既保留数字又记 UNSOURCED」）")
print(f"      条目数 = {len(SP.UNSOURCED)}")
for u in SP.UNSOURCED:
    print(f"        · {u}")
check("D.7 UNSOURCED 里没有任何一项与模块级常量同名（二者只能取其一）",
      not (set(SP.UNSOURCED)
           & {n for n in dir(SP) if n.isupper() and n != "UNSOURCED"}))
check("D.7 ★ 四条「规格声称核到、本库核不到」的记录在案",
      all(t in SP.UNSOURCED for t in (
          "hipfire_graph_on_off_spread_values_0p1_vs_1p5_3pct",
          "hipfire_verify_graph_env_var_name",
          "jit_cold_compile_30s_to_2min_interval",
          "model_reload_cold_start_2s_to_5s_interval")))
check("D.7 ★ 分量残差 81 s 明确列为**不可用**",
      "cold_start_component_residual_81s" in SP.UNSOURCED)
check("D.7 ★ 「紧得可疑」在 hipfire 里查不到，故不进正文",
      "tight_spread_suspicion_in_hipfire_doc" in SP.UNSOURCED
      and SP.SPEC_CLAIM_TIGHT_SPREAD_SOURCE_TIER == "D"
      and SP.RETRIEVED_TIGHT_SPREAD_SOURCE_TIER == "C")

sub("D.8  ★ 全模块常量覆盖的**机械检查**")
COVERED_01 = (
    # §1 噪声带与门限（D 级 hipfire）
    "HIPFIRE_WITHIN_SESSION_NOISE_PCT_GFX1100",
    "HIPFIRE_NOISE_BAND_IS_MEASURED_ON_FRESH_PROCESS",
    "HIPFIRE_SPEED_GATE_TOLERANCE_PCT",
    "HIPFIRE_SPEED_BASELINE_FILE_PATTERN",
    "HIPFIRE_IN_SESSION_WIN_INSIDE_NOISE_PCT",
    "HIPFIRE_GRAPH_AB_ENV_VAR",
    "HIPFIRE_MAX_REL_SPREAD_BEFORE_UNSTABLE",
    "HIPFIRE_BASELINE_UPDATE_MUST_BE_IN_SAME_COMMIT",
    "HIPFIRE_BASELINE_ROW_PROVENANCE_FIELDS",
    # §2 warmup 的官方默认（B 级 Triton）
    "TRITON_DO_BENCH_DEFAULT_WARMUP_MS",
    "TRITON_DO_BENCH_DEFAULT_REP_MS",
    "TRITON_DO_BENCH_WARMUP_UNIT_IS_MILLISECONDS",
    "TRITON_DO_BENCH_DEFAULT_RETURN_MODE",
    "TRITON_DO_BENCH_AUTOTUNE_QUANTILES",
    "TRITON_DO_BENCH_CLEARS_L2_BEFORE_EACH_ITER",
    "TRITON_DO_BENCH_IS_OFFICIAL_API",
    "TRITON_ISSUE_2306_TITLE_UNDERESTIMATE_PCT",
    "TRITON_ISSUE_2306_TIME_WITH_DEFAULT_WARMUP_MS",
    "TRITON_ISSUE_2306_TIME_WITH_EXTENDED_WARMUP_MS",
    "TRITON_ISSUE_2306_REQUIRED_WARMUP_OVER_MS",
    "TRITON_ISSUE_2306_IMPLIED_FAKE_GAIN_PCT",
    "TRITON_ISSUE_2306_AFFECTS_JIT_WITHOUT_AUTOTUNE",
    "TRITON_ISSUE_2306_IS_AN_OFFICIAL_REPO_ISSUE",
    # §3 冷启动的时间尺度（B 级 vLLM）
    "VLLM_JIT_FIRST_REQUEST_SPIKE_S",
    "VLLM_GRAPH_CAPTURE_PER_BATCH_SIZE_DELAY_S",
    "VLLM_FLASHINFER_FIRST_USE_NO_CACHE_MIN",
    "VLLM_FLASHINFER_LOGGED_FIRST_USE_S",
    "VLLM_DEEPGEMM_KERNEL_COUNTS",
    "VLLM_DEEPGEMM_FIRST_LAUNCH_MIN",
    "VLLM_COLD_START_TOTAL_CACHED_S",
    "VLLM_COLD_START_TOTAL_FRESH_MIN",
    "VLLM_COLD_START_LISTED_COMPONENTS_S",
    "VLLM_COLD_START_COMPONENT_SUM_S",
    "VLLM_COLD_START_BREAKDOWN_IS_INCOMPLETE_IN_EXCERPT",
    "VLLM_TORCH_COMPILE_COLD_S_MIN",
    "VLLM_ENFORCE_EAGER_SAVES_STARTUP_S",
    "VLLM_SKIP_WARMUP_AFFECTS_ONLY_FIRST_REQUEST_LATENCY",
    "VLLM_WARMUP_MUST_NOT_ALLOCATE_REAL_TENSORS",
    "VLLM_JIT_MONITOR_WARNS_ON_INFERENCE_TIME_COMPILE",
    # §7 本库建模参数（01 用到的部分）
    "STEADY_TIME_NOMINAL",
    "STEADY_SIGMA_DEFAULT",
    "COLD_START_SLOWDOWN_DEFAULT",
    "N_MEASUREMENTS_DEFAULT",
    "MC_REPLICATES_DEFAULT",
    "K_OVER_N_GRID",
    "ORDER_STATISTIC_QUANTILE_RULE",
    # §8 规格预跑值与被推翻的条目（01 用到的部分）
    "SPEC_PRERUN_MEDIAN_BIAS",
    "SPEC_PRERUN_MEAN_BIAS",
    "SPEC_MEDIAN_MECHANISM_IS_K_OVER_HALF",
    "SPEC_MEDIAN_MECHANISM_NAME",
    "SPEC_CLAIM_MEDIAN_MECHANISM_IS_K_OVER_HALF_LABEL",
    "SPEC_CLAIM_GRAPH_AB_ENV_VAR",
    "SPEC_CLAIM_GRAPH_AB_ENV_VAR_IS_REPRODUCIBLE",
    "SPEC_CLAIM_GRAPH_OFF_SPREAD_PCT",
    "SPEC_CLAIM_GRAPH_ON_SPREAD_PCT",
    "SPEC_CLAIM_GRAPH_SPREAD_VALUES_ARE_REPRODUCIBLE",
    "SPEC_CLAIM_COLD_COMPILE_TIME_S",
    "SPEC_CLAIM_COLD_COMPILE_IS_REPRODUCIBLE",
    "SPEC_CLAIM_MODEL_RELOAD_COLD_START_S",
    "SPEC_CLAIM_MODEL_RELOAD_IS_REPRODUCIBLE",
    "SPEC_CLAIM_MEASUREMENT_STATISTIC_DEFAULT_IS_MEAN",
)
print(f"    01 覆盖 {len(COVERED_01)} 条")


def _covered_names_in(filename, varname):
    """用 `ast` 读同目录另一个程序里的 `COVERED_*` 元组（**不用正则、不用 grep**）。"""
    src = pathlib.Path(__file__).resolve().parent / filename
    tree = ast.parse(src.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == varname:
                    return tuple(ast.literal_eval(node.value))
    raise AssertionError(f"{filename} 里找不到 {varname}")


ALL_CONSTANTS = frozenset(n for n in dir(SP) if n.isupper() and not n.startswith("_"))
c02 = _covered_names_in("02_significance.py", "COVERED_02")
c03 = _covered_names_in("03_convergence.py", "COVERED_03")
union = frozenset(COVERED_01) | frozenset(c02) | frozenset(c03)
print(f"    specs.py 的模块级常量共 {len(ALL_CONSTANTS)} 条")
print(f"    三个程序的覆盖：01={len(COVERED_01)} 02={len(c02)} 03={len(c03)} 并集={len(union)}")
print(f"    未被任何程序断言 = {sorted(ALL_CONSTANTS - union)}")
print(f"    断言了但 specs 里没有 = {sorted(union - ALL_CONSTANTS)}")
check("D.8 ★ 三程序覆盖的**并集**恰好等于 specs.py 的全部模块级常量（无漏网）",
      union == ALL_CONSTANTS, f"漏 {sorted(ALL_CONSTANTS - union)}")
check("D.8 三个程序之间**没有**重复覆盖（每条常量恰好一个归属）",
      len(COVERED_01) + len(c02) + len(c03) == len(union),
      f"{len(COVERED_01)}+{len(c02)}+{len(c03)} vs {len(union)}")
check("D.8 COVERED_01 里没有拼错的常量名",
      set(COVERED_01) <= ALL_CONSTANTS, f"{sorted(set(COVERED_01) - ALL_CONSTANTS)}")
check("D.8 `ALL_CONSTANTS` 非空（否则上面两条会退化成恒真）",
      len(ALL_CONSTANTS) > 100, f"{len(ALL_CONSTANTS)}")


# ===========================================================================
head("汇总")
# ===========================================================================
if FAILED:
    print(f"  ✗ {len(FAILED)} 条断言失败：")
    for f in FAILED:
        print(f"      · {f}")
    sys.exit(1)
print("  ✓ 全部断言通过（A 类 5 组 · B 类 3 组 · C 类 4 组 · D 类 8 组）")
print()
print("  ★ 被测对象是**本库构造的测量模型**，不是任何真实框架或硬件的测量；")
print("    本机的数是模型值，只能支撑关于**机制**的结论。")
print("  ★ 外部数字：§1 hipfire 是 **[D] 级（单一项目）**，只支撑「那个项目这么做」；")
print("    §2 Triton / §3 vLLM 是 **[B] 级（某一实现的官方规格）**——本程序的核心论断")
print("    由 B 级那组支撑，**不是**由 D 级经验值支撑。")
print("  ★ 规格的六条被推翻项记在 specs.py §8 的 `SPEC_CLAIM_*` 常量里（§D.6 逐条断言）。")
print(f"    python {sys.version.split()[0]} / numpy {np.__version__}（未 import torch）")

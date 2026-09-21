#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""mini_search.py —— **非编号共享模块**：人造代价地形 + 迷你搜索（独立实现）。

★ 它是「非编号支撑代码」，不计入「文档数 = 编号程序数」的核对（`CONTRIBUTING` §2）。

===============================================================================
★ 为什么是「独立实现，不 import `60` 域的任何代码」
===============================================================================

`60-compilers/auto-tuning` 的 `mini_autotune.py` 已经有一个迷你地形 + 四类搜索策略。
本模块**不 import 它**，理由有两条：

  1. **`CONTRIBUTING:36` 要求 `code/` 自包含**：跨域 import 会让本目录不再能单独拷贝运行。
  2. **本模块要回答的问题与它不同**：那一个讲「**无噪声**的搜索怎么被代价模型误导」，
     本模块讲「**把噪声加回来之后**搜索会怎么选错」。
     用同一个实现会让两件事的归因混在一起 —— 那正是 `60` 域在
     `mini_autotune.py:610-612` 主动冻结噪声时要避免的（见 `specs.MEASURE_NOISE_RATIONALE`）。

★ 常量名沿用 `MINI_` 前缀（与 `60` 的风格一致，便于读者对照），
  但**取值、地形与搜索都是本模块自己的**，与那边**没有数值上的对应关系**。

===============================================================================
★★ 本模块最关键的一条：噪声**不是造出来的**，是**量出来的**
===============================================================================

`60` 的 `measure()` 用一个开关 `MEASURE_NOISE` 乘一个 `rng.normal()` ——
**那是造出来的高斯噪声**。本模块**不这么做**，因为那会让「噪声有多重尾」这件事
由我而不是由机器决定（**P12：自造真值**）。

本模块的 `measure(cfg)` **真的把那份工作量跑一遍**，用 `time.perf_counter` 计时，
返回**真的秒数**：

    measured(cfg) = 真的计时( 跑 round(landscape(cfg)) 次原子操作 )

⇒ **噪声的全部性质（量级、重尾、时间相关性）都来自这台机器**，
  本模块一个字都没有规定它。这是与 `60` 的关键区别，也是本主题的立足点。

  · 代价：每次「测量」真的要花 `landscape(cfg) × t_op` 的墙钟时间
    （本机 `t_op ≈ 26 ns`，最优配置约 **0.5 ms** 一次）。
  · 「真值」`true_seconds(cfg)` 由**解析式**给出（`landscape(cfg) × UNIT_SECONDS_EST`），
    所以**我们知道正确答案**，而搜索只能看到带噪声的读数。
  · `UNIT_SECONDS_EST` 是**开局标定一次**的换算常数（把「迭代次数」换成「秒」），
    它只用来说明量级；正文里所有结论都建立在**比值**上。

===============================================================================
★ 地形：最优附近「分不开」
===============================================================================

`landscape(x)` 在 `x ∈ [0, MINI_N_CONFIGS)` 上是一个**二次碗**，最低点在
**非整数**位置 `MINI_OPT_X = 7.4` —— 所以**网格上根本不存在「恰好最优」的配置**，
本模块取 `MINI_BEST_X = round(MINI_OPT_X)` 为参考最优。

  · 顶部相邻配置的真实差 ≈ **0.36%**；
  · 而本机对一次 ~0.5 ms 的真实测量，**σ/中位 ≈ 2.5%**（见 `specs` 与 02 文档 §3）。
  ⇒ **真实差异比噪声小一个量级** —— 这正是「搜索会选错」的充分条件，
    也正是本主题要演示的东西。
"""

import math
import statistics
import time

# ===========================================================================
# 0. 常量（`MINI_` 前缀；**取值是本模块自己的**）
# ===========================================================================
MINI_N_CONFIGS = 20             # 配置数（一个 1-D 的「块大小」式搜索空间）
MINI_OPT_X = 7.4                # ★ **非整数**：网格上不存在恰好最优的配置
MINI_BEST_X = 7                 # 参考最优（网格上离 MINI_OPT_X 最近的那个）
# ★ 二次项系数：顶部相邻两个配置（x=7 与 x=8）的真实差只有 **0.12%**，
#   而本机一次 ~0.4 ms 真实测量的 σ/中位 ≈ **2.4%**（02 文档 §3 实测）。
#   ⇒ **真实差异比噪声小一个量级**，这是「搜索会选错」的充分条件。
MINI_CURVATURE = 6.0e-3
MINI_UNITS_AT_BEST = 5000       # 最优配置一次「测量」要跑的原子操作次数（本机 ≈ 0.40 ms）

# ★ 每「单位」地形值对应多少次原子操作（`MINI_UNITS_AT_BEST / 1.0` 的行文口径）。
MINI_UNITS_PER_COST = float(MINI_UNITS_AT_BEST)

# ★ 原子操作：**必须真的做一点工作、且结果必须回写模块级 sink** ——
#   否则整段会被解释器/编译器处理掉，测到的就不是「这份工作量」了
#   （`10-foundations/performance-models/03` §4.1 与计划 §0.7 D-7 第 2 条的教训：
#   被消掉的循环返回 0 秒，除以它得到 `inf`）。本机的**单次操作 ≈ 80 ns**。
MINI_ATOM_LO = 11
MINI_ATOM_HI = 29

_ATOM_SINK = 0


def atomic_batch(n):
    """跑 `n` 次原子操作。**这就是被计时的东西。**

    ★ 为什么写成一层平铺的循环：本模块只需要「一个代价稳定、不可被消除、
      且耗时与 `n` 成正比」的东西。`_ATOM_SINK` 是**模块级**变量，
      每次迭代都要读-改-写它，所以整段不可能被消掉。
    """
    global _ATOM_SINK
    for i in range(n):
        _ATOM_SINK ^= (i * MINI_ATOM_LO + MINI_ATOM_HI) & 0xFFFF


# ===========================================================================
# 1. 地形（**真值**，解析式）
# ===========================================================================
def landscape(x):
    """配置 `x` 的**真实**代价（单位：地形单位，越小越好）。

    ★ 最低点在 `MINI_OPT_X = 7.4`（非整数）—— 所以**没有任何一个网格配置恰好最优**。
      这是有意的：它让「网格上的参考最优」与「解析最优」分开，
      让读者不至于把「选错」误解成「实现 bug」。
    """
    return 1.0 + MINI_CURVATURE * (x - MINI_OPT_X) ** 2


CONFIGS = tuple(range(MINI_N_CONFIGS))
TRUE_UNITS = {x: landscape(x) * MINI_UNITS_PER_COST for x in CONFIGS}
BEST_X = min(CONFIGS, key=lambda x: landscape(x))
assert BEST_X == MINI_BEST_X, "网格参考最优与 MINI_BEST_X 不一致"


def calibrate_unit_seconds(reps=9, probe_units=MINI_UNITS_AT_BEST):
    """标定「一个地形单位 = 多少秒」：跑 `probe_units` 次原子操作，取**最小值**再折算。

    ★ **为什么用最小值**：最小值的物理含义是「在这台机器最不受打扰时跑出的那个数」，
      它最接近**这份工作量的固有成本**；中位数里混着抢占与调度抖动
      （本机实测：同规模的 σ/中位 ≈ 2.5%）。真值锚定在「固有成本」上，
      于是**噪声全部归给测量**，而不是被真值吸收掉一半。
    """
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        atomic_batch(probe_units)
        best = min(best, time.perf_counter() - t0)
    return best / probe_units


# ===========================================================================
# 2. ★ 测量：**真的跑一遍、真的计时**（不是 `random.gauss`）
# ===========================================================================
class Measurer:
    """把「一次测量」实现成**一次真实计时**。

    ★★ 本类没有任何噪声参数 —— 噪声全部来自 `time.perf_counter` 与本机负载。
      这正是与 `60/auto-tuning/code/mini_autotune.py` 的 `MEASURE_NOISE` 的分工：
      **那边把噪声显式冻结，这边把噪声如实放回来。**
    """

    def __init__(self, unit_seconds, noise=True):
        self.unit_seconds = unit_seconds
        self.noise = noise            # ★ 唯一的开关：关掉它 = 直接返回解析真值
        self.n_timed = 0              # 记账：真的计了几次时
        self.timed_seconds = 0.0

    def true_seconds(self, x):
        """配置 `x` 的**真值**（秒）—— 解析式，不含任何测量。"""
        return TRUE_UNITS[x] * self.unit_seconds

    def measure(self, x):
        """配置 `x` 的**一次测量**（秒）。

        · `noise=False` ⇒ 返回解析真值（**对照**：60 域的做法）；
        · `noise=True`  ⇒ **真的跑** `round(TRUE_UNITS[x])` 次原子操作并计时。
        """
        if not self.noise:
            return self.true_seconds(x)
        n = int(round(TRUE_UNITS[x]))
        t0 = time.perf_counter()
        atomic_batch(n)
        dt = time.perf_counter() - t0
        self.n_timed += 1
        self.timed_seconds += dt
        return dt

    def median_of(self, x, m):
        """对配置 `x` 测 `m` 次，返回**中位数**（秒）。

        ★ 用中位数而不是均值：本机噪声**重尾**（02 文档 §3 实测
          `max/中位` 可达百倍量级），均值会被尾部拖走。
        """
        if m <= 0:
            raise ValueError("m 必须 ≥ 1")
        if m == 1:
            return self.measure(x)
        return statistics.median([self.measure(x) for _ in range(m)])


# ===========================================================================
# 3. 迷你搜索：**穷举 + 每配置 m 次测量取中位**
# ===========================================================================
# ★ 为什么用穷举而不是「四类策略」：本模块的自变量是**噪声**，不是**搜索算法**。
#   搜索算法那一维是本主题**刻意固定住**的（四类策略、regret、预算公式的正文在
#   `60-compilers/auto-tuning/03-search-algorithms`，本域只回链）。
#   把搜索算法换掉，就把「噪声让搜索选错多少」这件事与「策略好坏」混在一起了。
class TrialResult:
    """一次搜索的记账。"""

    def __init__(self, pick, scores, regret, measured_gap):
        self.pick = pick                    # 选中的配置
        self.scores = scores                # {x: 每配置的测量中位（秒）}
        self.regret = regret                # 真实代价上的相对后悔（越小越好，0 = 选到参考最优）
        self.measured_gap = measured_gap    # ★ 报出来的「收益」：measured(pick) 比 measured(best) 好多少


def search(m, measurer, configs=CONFIGS):
    """穷举搜索：每个配置测 `m` 次取中位，选**读数最小**的那个。

    返回 `TrialResult`。★ `regret` 用**真值**算，`measured_gap` 用**读数**算 ——
    这两个数在噪声下会打架，那正是本模块要展示的东西。
    """
    scores = {x: measurer.median_of(x, m) for x in configs}
    pick = min(configs, key=lambda x: scores[x])
    t_pick = measurer.true_seconds(pick)
    t_best = measurer.true_seconds(BEST_X)
    # ★ 「报出来的收益」= 读数上 pick 比 best **好**多少（正数 = 报告有收益）
    gap = (scores[BEST_X] - scores[pick]) / scores[BEST_X]
    return TrialResult(pick, scores, t_pick / t_best - 1.0, gap)


def run_trials(m, n_trials, measurer, rng_configs=None):
    """跑 `n_trials` 次独立搜索，返回结果列表。

    ★ **每次试验都重新抽样**：`m` 次测量各自是一次真实计时，
      所以「重复试验」在这里就是「把同一件事在不同的机器状态下再做一遍」。
    """
    return [search(m, measurer) for _ in range(n_trials)]


def summarize(results):
    """把一批试验汇总成一张表用的字典。"""
    n = len(results)
    wrong = sum(1 for r in results if r.regret > 0)
    return {
        "n": n,
        "wrong": wrong,
        "wrong_frac": wrong / n if n else 0.0,
        "regret_median": statistics.median([r.regret for r in results]),
        "regret_mean": statistics.fmean([r.regret for r in results]),
        "regret_max": max(r.regret for r in results),
        "picks": sorted({r.pick for r in results}),
    }


def top_true_gaps(k=4):
    """打印用：最优附近前 `k` 个配置的**真实**相对差（证明顶部「分不开」）。"""
    order = sorted(CONFIGS, key=lambda x: landscape(x))[:k]
    base = landscape(BEST_X)
    return [(x, landscape(x) / base - 1.0) for x in order]


def describe_landscape():
    """一行摘要（给正文引用）。"""
    vals = [landscape(x) for x in CONFIGS]
    return ("配置 %d 个，真值范围 %.4f … %.4f（跨度 %.1f%%）；"
            "顶部相邻真实差 ≈ %.3f%%"
            % (MINI_N_CONFIGS, min(vals), max(vals),
               100.0 * (max(vals) / min(vals) - 1.0),
               100.0 * (landscape(BEST_X + 1) / landscape(BEST_X) - 1.0)))


def noise_probe(reps=400, x=BEST_X):
    """**本机测量噪声的实测**：对同一个配置连测 `reps` 次，返回统计量（秒）。

    ★ 这一半是给正文用的「σ 从哪里来」——它不依赖任何模型，就是本机的真实抖动。
    """
    ms = Measurer(calibrate_unit_seconds(), noise=True)
    vals = [ms.measure(x) for _ in range(reps)]
    med = statistics.median(vals)
    return {
        "n": reps,
        "median_s": med,
        "sigma_s": statistics.pstdev(vals),
        "sigma_over_median": statistics.pstdev(vals) / med if med else float("nan"),
        "p90_over_median": sorted(vals)[int(0.9 * len(vals))] / med,
        "max_over_median": max(vals) / med,
        "min_over_median": min(vals) / med,
    }


__all__ = [
    "MINI_N_CONFIGS", "MINI_OPT_X", "MINI_BEST_X", "MINI_CURVATURE", "MINI_UNITS_AT_BEST",
    "MINI_UNITS_PER_COST", "CONFIGS", "TRUE_UNITS", "BEST_X",
    "atomic_op", "atomic_batch", "landscape", "calibrate_unit_seconds",
    "Measurer", "TrialResult", "search", "run_trials", "summarize",
    "top_true_gaps", "describe_landscape", "noise_probe",
]

# ★ 防止「整个地形被优化掉」：模块级 sink 被 `atomic_batch` 写，且在这里被读一次。
_SINK_GUARD = _ATOM_SINK

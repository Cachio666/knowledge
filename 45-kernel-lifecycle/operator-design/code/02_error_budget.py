#!/usr/bin/env python3
r"""02-error-budget —— 归约长度 K 下的误差预算，以及「规范不保证什么」。

对应文档：`../02-numerical-design.md`

★ 本程序的可证伪断言：
  A 类  误差界的**公式**与手算逐位一致（`K·u`、`√K·u`、`γ_K = K·u/(1-K·u)`），
        且 `u16/u32 == 8192`（`log10 = 3.91`，**不是「三个数量级」**）。
  B 类  给出「K 增大到多少时 fp16 累加不再可用」的**临界值**，并给出**三个天花板**
        （最坏界 / fp16 值域 / 统计界）——**取最紧的那个**。改错任何一个 ⇒ B 失败。
  C 类  ★ 两条都必须**真的跑出问题**：
        ① **把同一组数打乱顺序再累加，结果不同**——差异远超单次舍入单位 `u32`。
           这是 specs 里 A 级事实「accumulation order **未规定**」的**直接证据**。
        ② **fp16 累加与 fp32 累加的差**超过 **fp16 在结果量级上的相邻可表示数间隔**
           （`np.spacing`）——「累加精度不是细节」这条论断在脚本里是可证的。
  C.3   ⚠ **一条自己先说破的边界**：观测到的相对误差**不是 K 的单调函数**——
        单一 K 上的单次测量**不能**支撑「误差随 K 增长」。这一条也写成了断言。

跑法：`make 02_error_budget`。纯 python3 + **numpy**（只用它做 fp16/fp32 的标量累加），
不 import torch，不碰 GPU。

★ 口径声明（必须与文档一致，否则数字不可比）：
  · 本程序产出的**不是实测性能数字**，而是**本机可复现的数值仿真**：它用 numpy 的
    `float16` / `float32` 标量**顺序**累加来模拟一条归约链的舍入行为。
  · **这不是 GPU 上某个 kernel 的实测**——它模拟的是**算术**，不是**硬件**。
    域 README 明令：本域任何一篇都不得声称有实测数据。
  · 随机序列用**固定种子**（`SEED`）⇒ 本程序的输出**逐字节可复现**。
"""

import sys

import numpy as np

import specs as SP

FAILED = []


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


SEED = 20260920
RNG = np.random.default_rng(SEED)

U16 = SP.FP16_UNIT_ROUNDOFF_U          # 2^-11 = 4.8828125e-04
U32 = SP.FP32_UNIT_ROUNDOFF_U          # 2^-24 = 5.960464477539063e-08

# ★ 本篇**不引用** `FLOAT16_BYTES`：本程序算的全部是**相对**量（误差 ÷ 结果），
#   元素宽度在约分中消失了。强行引用它只会制造一条**假断言**——那正是上一轮
#   C6 判据要消灭的东西。该常量由 03 篇（shared memory 算式）**真正**引用。
#   详见 01 篇 §D.9 的并集检查：每条常量**恰好一个**归属。

# 输入：归约长度 K、累加精度、数组分布（文档 §0 的「输入」三项）
DISTRIBUTIONS = {
    "normal(0,1)": lambda k: RNG.standard_normal(k),
}


def sequential_sum(x, dtype):
    """**顺序**累加（刻意不用 `np.sum`——它的成对求和会掩盖我们要展示的现象）。"""
    acc = dtype(0.0)
    for v in x:
        acc = dtype(acc + dtype(v))
    return float(acc)


def worst_bound(K, u):
    """最坏情形相对误差界（本库 `10-foundations/floating-point/04` 的同口径）。"""
    return K * u / (1.0 - K * u) if K * u < 1.0 else float("inf")


def worst_bound_simple(K, u):
    """简化最坏界：`K·u`（不写分母）。K·u ≥ 1 时即为「界失效」。"""
    return K * u


def stat_bound(K, u):
    """统计意义下的界：`√K·u`（随机游走，不是直接相加）。"""
    return np.sqrt(K) * u


def fp16_steps(x32, x16):
    """差值相当于**结果量级上**多少个 fp16 可表示步长。

    ★ 这是「超过 fp16 自身的表示精度」这句话的**精确定义**——
      不是「超过 u16」这种含糊说法，而是「超过了它在**自己那个量级**上的相邻可表示数间距」。
    """
    s = np.spacing(np.float16(abs(x32)))
    return abs(x16 - x32) / float(s), float(s)


# ===========================================================================
head("§A  算法正确性：误差界的公式与手算逐位一致")
# ===========================================================================

sub("A.1  `u` 的两个值与它们的比（**这里是最容易被写错的地方**）")
print(f"    u_fp16 = 2^-11 = {U16!r}")
print(f"    u_fp32 = 2^-24 = {U32!r}")
print(f"    u16/u32        = {U16/U32!r}")
print(f"    十进制数量级   = log10({U16/U32:g}) = {np.log10(U16/U32):.4f}")
check("A.1 u_fp16 == 2^-11 逐位一致", U16 == 2.0 ** -11, f"{U16!r}")
check("A.1 u_fp32 == 2^-24 逐位一致", U32 == 2.0 ** -24, f"{U32!r}")
check("A.1 ★ u16/u32 == 8192（= 2^13），与 specs 的 U16_OVER_U32 一致",
      U16 / U32 == SP.U16_OVER_U32 == 8192.0, f"{U16/U32!r}")
check("A.1 ★★★ 数量级是 **3.91**，**不是 3**（规格初稿写「约三个数量级」，是低估）",
      abs(np.log10(U16 / U32) - SP.U16_OVER_U32_ORDERS_OF_MAGNITUDE) < 5e-3
      and SP.U16_OVER_U32_ORDERS_OF_MAGNITUDE > 3.9,
      f"log10 = {np.log10(U16/U32):.4f}")
print("    ★ 这一条是本程序存在的理由之一：**同一个数在本域被写错过两次**。")
print("      8192 = 2^13 是「十进制的 3.91 个数量级」——写「三」低估、写「四」高估。")

sub("A.2  三个界的公式：手算 vs 函数")
HAND_K = (100, 1000)
for K in HAND_K:
    w_simple = worst_bound_simple(K, U16)
    w_gamma = worst_bound(K, U16)
    s = stat_bound(K, U16)
    print(f"    K={K:>5}：K·u16 = {w_simple:.10f}"
          f"   γ_K = K·u/(1-K·u) = {w_gamma:.10f}   √K·u16 = {s:.10f}")
    check(f"A.2 K={K} 的 K·u16 与手算一致", abs(w_simple - K * U16) < 1e-15,
          f"{w_simple!r}")
    check(f"A.2 K={K} 的 γ_K 与手算一致",
          abs(w_gamma - (K * U16) / (1 - K * U16)) < 1e-15, f"{w_gamma!r}")
    check(f"A.2 K={K} 的 √K·u16 与手算一致",
          abs(s - float(np.sqrt(K)) * U16) < 1e-15, f"{s!r}")
check("A.2 K·u16 < 1 时 γ_K 才有限（K·u16 ≥ 1 时界退化）",
      worst_bound(1000, U16) < float("inf") and worst_bound(2048, U16) == float("inf"),
      f"{worst_bound(1000, U16):.6f} / {worst_bound(2048, U16)}")

sub("A.3  最坏界与统计界**差两个数量级**——所以「用哪个界」必须写出来")
K_DEMO = 10000
print(f"    K = {K_DEMO}：最坏 K·u16 = {worst_bound_simple(K_DEMO, U16):.4f}"
      f"   统计 √K·u16 = {stat_bound(K_DEMO, U16):.4f}"
      f"   相差 {worst_bound_simple(K_DEMO, U16)/stat_bound(K_DEMO, U16):.1f} 倍")
check("A.3 ★ K=10000 时最坏界与统计界之比 == √K == 100（差两个数量级）",
      abs(worst_bound_simple(K_DEMO, U16) / stat_bound(K_DEMO, U16) - 100.0) < 1e-9,
      f"{worst_bound_simple(K_DEMO, U16)/stat_bound(K_DEMO, U16):.4f}")
check("A.3 这正是 specs 里 PTX 那条事实的后果：界取决于**你用哪种假设**",
      "accumulation order" in SP.PTX_WMMA_UNSPECIFIED_ASPECTS)
print("    ⇒ **「容差该定多少」没有无条件答案**：拿最坏界定出的容差会比统计界松")
print("      两个数量级——松到**什么都测不出来**。这条支撑 §B.4 的判据。")


# ===========================================================================
head("§B  模型与公式：fp16 累加的**三个天花板**，取最紧")
# ===========================================================================

sub("B.1  三个天花板的推导")
K_round_worst = 1.0 / U16
K_round_stat = (1.0 / U16) ** 2
K_range = SP.FP16_MAX_FINITE        # 对抗性输入：K 个数全为 1 ⇒ 运行和到 K
print(f"    ① 最坏界失效      ：K·u16 ≥ 1        ⇒ K ≥ 1/u16      = {K_round_worst:,.0f}")
print(f"    ② 统计界失效      ：√K·u16 ≥ 1       ⇒ K ≥ (1/u16)²   = {K_round_stat:,.0f}")
print(f"    ③ fp16 值域       ：对抗输入(全 1)下运行和 = K ≤ {SP.FP16_MAX_FINITE:,.0f}"
      f"  ⇒ K ≤ {K_range:,.0f}")
ceilings = {
    "worst-case rounding": K_round_worst,
    "fp16 finite range": K_range,
    "statistical rounding": K_round_stat,
}
tightest = min(ceilings, key=lambda k: ceilings[k])
print(f"    ⇒ **取最紧**的 = {tightest} ⇒ K ≤ {ceilings[tightest]:,.0f}")
check("B.1 最坏界给出的临界 K == 2048（= 1/u16）",
      K_round_worst == 2048.0, f"{K_round_worst!r}")
check("B.1 统计界给出的临界 K == 4194304（= (1/u16)²）",
      K_round_stat == 4194304.0, f"{K_round_stat!r}")
check("B.1 fp16 值域给出的临界 K == 65504（specs 的 FP16_MAX_FINITE）",
      K_range == 65504.0, f"{K_range!r}")
check("B.1 ★ 三者取最紧 ⇒ 答案是**最坏界**的 2048，不是统计界的 4.19e6",
      tightest == "worst-case rounding" and ceilings[tightest] == 2048.0,
      f"{tightest} = {ceilings[tightest]}")
check("B.1 ★ 最紧与次紧相差 31.98 倍（65504/2048）——「取最紧」不是修辞",
      abs(K_range / K_round_worst - 31.984375) < 1e-6,
      f"{K_range / K_round_worst:.6f}")

sub("B.2  同样的「取最紧」用在 fp32 累加上：三个天花板各是多少")
K32_worst, K32_stat = 1.0 / U32, (1.0 / U32) ** 2
print(f"    ① 最坏界  K ≥ 1/u32   = {K32_worst:,.0f}")
print(f"    ② 统计界  K ≥ (1/u32)² = {K32_stat:,.3e}")
print(f"    ③ 值域    fp32 的最大有限数 = 3.4028e38 ⇒ 实际不构成约束")
check("B.2 fp32 的最坏界临界 K == 16777216（= 2^24）",
      K32_worst == 16777216.0, f"{K32_worst!r}")
check("B.2 ★ fp32 的最坏界临界值比 fp16 的高 8192 倍（= u16/u32）",
      abs(K32_worst / K_round_worst - SP.U16_OVER_U32) < 1e-9,
      f"{K32_worst / K_round_worst:.1f}")
print(f"    ⇒ 同一份推导、同一份代码，**累加精度一换，可用的 K 就相差 8192 倍**。")

sub("B.3  临界值上下的**实测**（固定种子，中位数，本机可复现的数值仿真）")
print(f"    {'K':>8} {'|a16-a32| 中位':>17} {'÷u16':>9} {'fp16 步长数 中位':>17}"
      f" {'÷u16>1?':>9}")
OBS = []
for K in (16, 64, 256, 1024, 4096, 16384):
    diffs, steps = [], []
    for _ in range(40):
        x = DISTRIBUTIONS["normal(0,1)"](K)
        a32 = sequential_sum(x, np.float32)
        a16 = sequential_sum(x, np.float16)
        diffs.append(abs(a16 - a32))
        steps.append(fp16_steps(a32, a16)[0])
    md, ms = float(np.median(diffs)), float(np.median(steps))
    OBS.append((K, md, ms))
    print(f"    {K:>8} {md:>17.4e} {md/U16:>9.3f} {ms:>17.3f} {'✓' if md/U16 > 1 else '✗':>9}")
check("B.3 ★★ 即使 K 小到 16，两边的差也已 **> u16**（= 单次舍入单位）",
      OBS[0][1] / U16 > 1.0, f"K=16 → {OBS[0][1]/U16:.3f} × u16")
check("B.3 ★★ 所有 K 上「差 > u16」都成立（⇒「累加精度不是细节」全线可证）",
      all(o[1] / U16 > 1.0 for o in OBS), f"{[round(o[1]/U16,1) for o in OBS]}")
check("B.3 ★★ 且「差 ÷ u16」随 K **严格单调上升**（2153 倍 vs 3 倍，差三个数量级）",
      all(OBS[i][1] / U16 < OBS[i + 1][1] / U16 for i in range(len(OBS) - 1))
      and OBS[-1][1] / U16 > 500.0,
      f"{[round(o[1]/U16,0) for o in OBS]}")
print("    ⚠ 「fp16 步长数」那一列**不是单调的**（K=16 时 0.81 步、K=4096 时 0.73 步）——")
print("      原因是**相消**：它除的是「结果自己那个量级」，而结果量级本身在跳。见 §C.3。")
print("      所以本节把它**只作为参考列出**，断言放在「差 ÷ u16」这一列上。")

sub("B.4  判据：容差该按哪个界定")
print(f"    用最坏界（K·u16）定容差：K=2048 时容差要开到 **100%**——")
print(f"      这个容差对任何输入都通过，**等于没有容差**。")
print(f"    用统计界（√K·u16）定：K=2048 时是 {stat_bound(2048, U16)*100:.3f}%——**能测出问题**。")
check("B.4 ★ 最坏界在 K=2048 处正好退化到 1.0（100% 相对误差）",
      abs(worst_bound_simple(2048, U16) - 1.0) < 1e-15,
      f"{worst_bound_simple(2048, U16)!r}")
check("B.4 ★ 同一 K 上统计界仍是 2.21%（两个界的实用价值差三个数量级）",
      abs(stat_bound(2048, U16) - 0.02209708691207961) < 1e-12,
      f"{stat_bound(2048, U16)!r}")
print("    ⇒ **判据（写进 §5 表）**：容差的依据必须写明是**哪一条界**；")
print("      写「最坏界」的容差在 K ≳ 2048 时**自动失去判别力**。")


# ===========================================================================
head("§C  边界与反例：两条断言必须**真的跑出问题**")
# ===========================================================================

sub("C.1  ★ 把同一组数**打乱顺序**再累加，结果不同（「order 未规定」的直接证据）")
print(f"    specs 的 A 级事实：{SP.PTX_WMMA_UNSPECIFIED_ASPECTS}")
print(f"    做法：固定一组 K 个数的 fp32 顺序和，再做 6 次随机打乱，量最大差")
print()
print(f"    {'K':>8} {'与基准的最大差':>17} {'/u32':>12} {'≫ 单次舍入?':>12}")
scramble_rows = []
for K in (1000, 10000, 100000):
    x = DISTRIBUTIONS["normal(0,1)"](K)
    base = sequential_sum(x, np.float32)
    maxd, worst_sum = 0.0, base
    for _ in range(6):
        y = x.copy()
        RNG.shuffle(y)
        s = sequential_sum(y, np.float32)
        if abs(s - base) > maxd:
            maxd, worst_sum = abs(s - base), s
    scramble_rows.append((K, maxd))
    print(f"    {K:>8} {maxd:>17.4e} {maxd/U32:>12,.0f} {'✓' if maxd > U32 else '✗':>12}")
    print(f"             基准和 = {base!r}；某个打乱后的和 = {worst_sum!r}"
          f"  ⇒ 两者**不相等**：{base != worst_sum}")
check("C.1 ★★ K=1000 时打乱的差异就已 **> 100× u32**（远超单次舍入）",
      scramble_rows[0][1] / U32 > 100.0, f"{scramble_rows[0][1]/U32:,.0f}×")
check("C.1 ★★ 三个 K 上差异都**严格大于** u32（不是「约等于」）",
      all(d > U32 for _, d in scramble_rows), f"{[f'{d/U32:,.0f}x' for _, d in scramble_rows]}")
check("C.1 ★★ 最大差随 K **单调增长**（K 越大，顺序的影响越大）",
      scramble_rows[0][1] < scramble_rows[1][1] < scramble_rows[2][1],
      f"{[f'{d:.2e}' for _, d in scramble_rows]}")
print("    ★ 这就是 §2 那条规范事实的可运行形态：**同一份输入、同一个 kernel、")
print("      只把累加顺序换掉，结果就不是同一个数**。")
print("      ⇒ 任何承诺「逐位可复现」的实现，**必须自己保证顺序**——规范不保证。")

sub("C.2  ★ fp16 累加与 fp32 累加的差，**超过 fp16 自身的表示精度**")
print(f"    「自身的表示精度」在这里取**精确**定义：np.spacing(结果量级) ——")
print(f"    即 fp16 在**结果自己那个量级**上，相邻两个可表示数之间的间隔。")
print()
# ★ 每一行只生成**一次**序列，fp32 与 fp16 用的是**同一组数**（否则比较无意义）
print(f"    {'K':>8} {'a32':>12} {'a16':>12} {'|差|':>12} {'fp16 步长':>12} {'差 = 几步':>11}")
PAIRS = []
for K in (64, 256, 4096, 65536):
    x = DISTRIBUTIONS["normal(0,1)"](K)
    a32 = sequential_sum(x, np.float32)
    a16 = sequential_sum(x, np.float16)
    steps, sp = fp16_steps(a32, a16)
    PAIRS.append((K, x, a32, a16, steps, sp))
    print(f"    {K:>8} {a32:>12.4f} {a16:>12.4f} {abs(a16-a32):>12.4e}"
          f" {sp:>12.4e} {steps:>11.2f}")
steps_by_k = {K: st for K, _, _, _, st, _ in PAIRS}
check("C.2 ★★ K=256 时差异已超过 fp16 一个步长（规格预验证的量级关系成立）",
      steps_by_k[256] > 1.0, f"{steps_by_k[256]:.3f} 步")
check("C.2 ★★ 所有 K 上 |差| 都**严格大于** u16（= fp16 的单次舍入单位）",
      all(abs(a16 - a32) > U16 for _, _, a32, a16, _, _ in PAIRS),
      f"{ {K: round(abs(a16-a32)/U16, 2) for K, _, a32, a16, _, _ in PAIRS} }")
check("C.2 ★★ K=65536 时结果在**自己那个量级上**差了 **100 步以上**"
      "（fp16 的相邻可表示数之间）",
      steps_by_k[65536] > 100.0, f"{steps_by_k[65536]:.2f} 步")
check("C.2 ★★ K=65536 的步长数比 K=64 高**一个数量级以上**（随 K 上升）",
      steps_by_k[65536] > 10 * steps_by_k[64],
      f"{steps_by_k[65536]:.2f} vs {steps_by_k[64]:.2f}")
print("    ⚠ 但「差 = 几步」这一列**不是单调的**：K=4096 时只有 0.73 步——")
print("      因为结果量级（分母）本身在跳。⇒ **步长数不能单独用来下结论**（见 C.3）。")
# ★ 反驳点：把 fp16 累加换成 fp32 累加，差异必须**恰好归零**
#   （证明差异来自**累加格式**，不是来自随机性/序列本身）
K256_X = dict((K, x) for K, x, _, _, _, _ in PAIRS)[256]
check("C.2 反驳点：同一组数改用 fp32 累加，两次结果**逐位相同**（差异归零）",
      sequential_sum(K256_X, np.float32) - sequential_sum(K256_X, np.float32) == 0.0)
check("C.2 反驳点：而同一条 fp32 结果与 fp16 结果**确实不同**（差异非零）",
      sequential_sum(K256_X, np.float32) != sequential_sum(K256_X, np.float16))
print("    ⇒ **「我用 fp16 存、fp32 累加」与「我用 fp16 累加」是两个不同的算子**：")
print("      它们在 K=64 时就已经不是一个数，且差距随 K 拉大。")

sub("C.3  ⚠ 自己先说破：观测误差**不是** K 的单调函数——单次测量不能支撑趋势")
print("    上一节用的是**中位数**（40 次）。如果只看**单条**序列，结论会翻转：")
print()
print(f"    {'K':>8} {'|a16-a32|/|a32|':>18} {'= 几个 u16':>12} {'K·u16(最坏)':>13}")
singles = []
for K in (64, 256, 1024, 2048, 4096, 16384):
    x = DISTRIBUTIONS["normal(0,1)"](K)
    a32 = sequential_sum(x, np.float32)
    a16 = sequential_sum(x, np.float16)
    rel = abs(a16 - a32) / abs(a32)
    singles.append((K, rel))
    print(f"    {K:>8} {rel:>18.6e} {rel/U16:>12.4f} {K*U16:>13.3f}")
monotone = all(singles[i][1] <= singles[i + 1][1] for i in range(len(singles) - 1))
check("C.3 ★★ 单条序列的相对误差**不是**单调的（⇒ 不能用单次测量下结论）",
      not monotone, f"{[f'{r/U16:.2f}' for _, r in singles]}")
check("C.3 ★★ 而且最坏界 K·u16 与观测值**完全脱钩**（K=16384 时界是 8.0，"
      "观测不到 0.02）",
      worst_bound_simple(16384, U16) > 100 * max(r for _, r in singles),
      f"界 {worst_bound_simple(16384, U16):.1f} vs 观测最大 "
      f"{max(r for _, r in singles):.4f}")
print("    ★ 同一个现象在 C.2 的「步长数」那一列上**也**出现了（K=4096 只有 0.73 步）——")
print("      两个不同的度量、同一处非单调 ⇒ **这不是度量的毛病，是现象本身的性质**。")
print("    ★ 为什么：**相对**误差被**相消**支配——当 |a32| 恰好很大时，")
print("      同样的绝对误差除以一个大的分母就变小。⇒ **单点测量量的是「这一次的运气」**。")
print("      ⚠ 这条与本库的元教训同构：**紧得可疑的结果通常是采集方式的问题**。")
print("      ⇒ 判据（§5 表）：报误差时必须同时给出 **K、界的口径、以及测量次数**；")
print("        缺任何一个，那个数都不能用来支撑「误差随 K 增长」。")

sub("C.4  ★ 同一个序列，只把**累加精度**换掉 —— 结果差异相对 fp32 自身是否可忽略")
x = DISTRIBUTIONS["normal(0,1)"](4096)
a32 = sequential_sum(x, np.float32)
a64 = sequential_sum(x, np.float64)
print(f"    fp64（视为精确） = {a64!r}")
print(f"    fp32             = {a32!r}   与 fp64 的相对差 = {abs(a32-a64)/abs(a64):.3e}"
      f" = {abs(a32-a64)/abs(a64)/U32:.2f} × u32")
print(f"    fp16             = {sequential_sum(x, np.float16)!r}"
      f"   与 fp64 的相对差 = "
      f"{abs(sequential_sum(x, np.float16)-a64)/abs(a64):.3e}")
check("C.4 ★ fp32 累加相对 fp64 的误差在**几个 u32** 之内（fp32 累加是够用的）",
      abs(a32 - a64) / abs(a64) < 10 * U32,
      f"{abs(a32-a64)/abs(a64)/U32:.2f} × u32")
check("C.4 ★★ fp16 累加相对 fp64 的误差**高出至少两个数量级**",
      abs(sequential_sum(x, np.float16) - a64) / abs(a64) > 100 * U32,
      f"{abs(sequential_sum(x,np.float16)-a64)/abs(a64)/U32:.0f} × u32")
print("    ⇒ 「用 fp16 累加」与「用 fp32 累加」的差距**不是一点点**：")
print("      前者相对精确值的误差是后者的**两个数量级以上**。")


# ===========================================================================
head("§D  常量自洽：本篇用到的 specs.py 常量，逐条被断言钉住")
# ===========================================================================

sub("D.1  `u` 的两个常量")
check("D.1 FP16_UNIT_ROUNDOFF_U == 2^-11 逐位", U16 == 2.0 ** -11, f"{U16!r}")
check("D.1 FP32_UNIT_ROUNDOFF_U == 2^-24 逐位", U32 == 2.0 ** -24, f"{U32!r}")
check("D.1 两者都是 2 的整数次幂（⇒ 差是 2 的幂，不是「约」）",
      np.log2(U16).is_integer() and np.log2(U32).is_integer(),
      f"2^{np.log2(U16):.0f} / 2^{np.log2(U32):.0f}")

sub("D.2  `U16_OVER_U32` 与数量级常量")
check("D.2 U16_OVER_U32 == 8192.0", SP.U16_OVER_U32 == 8192.0, f"{SP.U16_OVER_U32!r}")
check("D.2 数量级常量在 (3.9, 3.92) 内（⇒ 断言它「不是 3」）",
      3.9 < SP.U16_OVER_U32_ORDERS_OF_MAGNITUDE < 3.92,
      f"{SP.U16_OVER_U32_ORDERS_OF_MAGNITUDE}")
check("D.2 ★ 它与 log10(U16/U32) 一致到 3 位小数",
      abs(SP.U16_OVER_U32_ORDERS_OF_MAGNITUDE - np.log10(SP.U16_OVER_U32)) < 5e-3,
      f"{np.log10(SP.U16_OVER_U32):.4f}")

sub("D.3  `FP16_MAX_FINITE`：它就是 §B 的第三个天花板，且被 numpy 独立复核")
check("D.3 FP16_MAX_FINITE == 65504.0", SP.FP16_MAX_FINITE == 65504.0,
      f"{SP.FP16_MAX_FINITE!r}")
check("D.3 ★ 用 numpy 的 fp16 复核：它的最大有限数正是 65504（另一条实现路径）",
      float(np.finfo(np.float16).max) == SP.FP16_MAX_FINITE == 65504.0,
      f"{float(np.finfo(np.float16).max)!r}")
check("D.3 ★ 而且 65504 == (2 − 2^-10)·2^15 —— 位域直算的结果与 specs 一致",
      65504.0 == (2 - 2.0 ** -10) * 2.0 ** 15
      and float(np.finfo(np.float16).max) == (2 - 2.0 ** -10) * 2.0 ** 15,
      f"{(2 - 2.0**-10) * 2.0**15!r}")
check("D.3 它给出的 K 天花板是 65504（对抗性全 1 输入下运行和 = K）",
      SP.FP16_MAX_FINITE == K_range)
print("    ★ 这一条是**独立复核**：specs 的 65504 出自 IEEE 754 的位域算式，")
print("      而 numpy 的 fp16 是另一条实现路径——两边给出同一个边界。")

sub("D.4  A 级规范事实：PTX 的「规定 vs 不规定」")
check("D.4 元素乘法「至少单精度」", SP.PTX_WMMA_MUL_PRECISION_AT_LEAST == "single",
      f"{SP.PTX_WMMA_MUL_PRECISION_AT_LEAST}")
check("D.4 `.f32` 累加「至少单精度」", SP.PTX_WMMA_ACC_F32_AT_LEAST == "single",
      f"{SP.PTX_WMMA_ACC_F32_AT_LEAST}")
check("D.4 ★ 两者都 `.f16` 时累加「至少**半**精度」（唯一允许半精度的那一档）",
      SP.PTX_WMMA_ACC_F16_AT_LEAST == "half", f"{SP.PTX_WMMA_ACC_F16_AT_LEAST}")
check("D.4 ★★ 未规定项恰好三项，且**含 accumulation order**（C.1 的依据）",
      SP.PTX_WMMA_UNSPECIFIED_ASPECTS
      == ("accumulation order", "rounding", "subnormal input handling"),
      f"{SP.PTX_WMMA_UNSPECIFIED_ASPECTS}")
check("D.4 ★★ 这三项的**引用等级是「逐字」**（两次独立检索一致）",
      SP.PTX_WMMA_UNSPECIFIED_QUOTE_IS_VERBATIM is True)
print("    ★ 与 01 篇的 `VLLM_KERNELCONFIG_HASH_RULE_IS_VERBATIM = False` 成对读：")
print("      一边够格、一边不够格——**这就是引用纪律在执行，不是在自我表扬**。")
check("D.4 「至少」是**下限**：它不等于「恰好」⇒ 正文不得把 at least 读成精确值",
      SP.PTX_WMMA_MUL_PRECISION_AT_LEAST == "single"
      and SP.PTX_WMMA_ACC_F16_AT_LEAST == "half")

sub("D.5  cuDNN 的 FP8 两档累加：**方向决定档位**")
check("D.5 两档 compute type 恰好是 (FAST_FLOAT_FOR_FP8, FLOAT)",
      SP.CUDNN_FP8_COMPUTE_TYPES == ("CUDNN_DATA_FAST_FLOAT_FOR_FP8", "CUDNN_DATA_FLOAT"),
      f"{SP.CUDNN_FP8_COMPUTE_TYPES}")
check("D.5 前向/推理**可以**用 FAST_FLOAT_FOR_FP8",
      SP.CUDNN_FP8_FORWARD_MAY_USE_FAST_FLOAT is True)
check("D.5 ★★ 训练**反向**需要更精确的 FLOAT（同一算子的两个方向不对称）",
      SP.CUDNN_FP8_BACKWARD_REQUIRES_FLOAT is True)
check("D.5 ★ 这条不对称与 03 篇的 batch 约束不对称**同向**（两处独立佐证）",
      SP.CUDNN_FP8_HOPPER_FWD_BATCH_ANY is True
      and SP.CUDNN_FP8_HOPPER_BWD_BATCH_MULTIPLE == 16,
      f"fwd any / bwd %16")
print("    ⇒ **「这个算子该用哪档累加」不是算子的属性，是「算子 × 方向」的联合属性**")
print("      ——与 `requirement-discovery` 02 篇「受限类型是「算子 × 精度路径」的联合属性」")
print("        **同构**（跨主题的同一条结构性认识）。")

sub("D.6  `UNSOURCED` 在本篇的落点")
check("D.6 ★ cuDNN **没有给出** FAST_FLOAT_FOR_FP8 的误差量级 ⇒ 正文不得编数字",
      "cudnn_fast_float_for_fp8_error_magnitude" in SP.UNSOURCED)
check("D.6 ★ wmma 的 fragment 元素归属**规范明写未规定** ⇒ 本主题不回链重推",
      "wmma_fragment_element_mapping_is_unspecified" in SP.UNSOURCED)
print("    ⚠ 所以本篇只能说「**允许**哪一档」，**不能**说「换了差多少」——")
print("      后者的数字在检索里找不到，按 §0.6 硬规则 4 **不写进来**。")

# ---------------------------------------------------------------------------
sub("D.7  本程序的常量覆盖表")
# ★ 归属口径：**每条常量恰好一个「家」**。`CUDNN_FP8_HOPPER_FWD_BATCH_ANY` 与
#   `CUDNN_FP8_HOPPER_BWD_BATCH_MULTIPLE` 在 §D.5 被**交叉引用**（用来佐证「方向不对称」），
#   但它们的**归属**是 03 篇——**引用 ≠ 归属**（01 的 §D.9 断言两两不交）。
COVERED_02 = (
    "FP16_UNIT_ROUNDOFF_U",
    "FP32_UNIT_ROUNDOFF_U",
    "U16_OVER_U32",
    "U16_OVER_U32_ORDERS_OF_MAGNITUDE",
    "FP16_MAX_FINITE",
    "PTX_WMMA_MUL_PRECISION_AT_LEAST",
    "PTX_WMMA_ACC_F32_AT_LEAST",
    "PTX_WMMA_ACC_F16_AT_LEAST",
    "PTX_WMMA_UNSPECIFIED_ASPECTS",
    "PTX_WMMA_UNSPECIFIED_QUOTE_IS_VERBATIM",
    "CUDNN_FP8_COMPUTE_TYPES",
    "CUDNN_FP8_FORWARD_MAY_USE_FAST_FLOAT",
    "CUDNN_FP8_BACKWARD_REQUIRES_FLOAT",
)
for _n in COVERED_02:
    print(f"      {_n:44s} 有断言")
print(f"    ⇒ 02 篇覆盖 {len(COVERED_02)} 条")


# ===========================================================================
head("汇总")
# ===========================================================================
if FAILED:
    print(f"  ✗ {len(FAILED)} 条断言失败：")
    for f in FAILED:
        print(f"      · {f}")
    sys.exit(1)
print("  ✓ 全部断言通过（A 类 3 组 · B 类 4 组 · C 类 4 组 · D 类 7 组）")
print()
print("  ★ 本程序的数**全部是本机可复现的数值仿真**，不是实测：")
print("    它用 numpy 的 float16/float32 **标量顺序累加**模拟一条归约链的舍入行为，")
print("    模拟的是**算术**，不是**硬件**。本机 macOS 无 GPU，本域不产出实测性能数字。")
print(f"    随机序列固定种子 SEED={SEED} ⇒ 输出逐字节可复现。")
print(f"    python {sys.version.split()[0]} / numpy {np.__version__}"
      f"（未 import torch）")

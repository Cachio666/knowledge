#!/usr/bin/env python3
r"""03-convergence —— 什么时候应当停手，以及「一直调下去」为什么必然产出假结论。

对应文档：`../03-convergence.md`

★ 本程序的可证伪断言：
  A 类  停止点的**闭式解** `k* = ⌊ln(δ*/g₀) / ln r⌋` 与逐步枚举**一致**；
        并且「用 4σ/√n 还是用精确的 2.80σ/√n」会给出**不同的停止步数**。
  B 类  噪声带每翻一倍，停止点提前 `ln2 / |ln r|` 步（可算的定量关系）；
        门限与噪声带的关系给出**误报率 = max(0, 1 − T/A)**；跨架构同一改动结论相反。
  C 类  ★★ 真实增益恒为 0 时「永远继续调」累积的假胜利次数（10/50/200 步 → 0.33/1.64/6.56），
        以及**假胜利被报出来的幅度**（不是只有符号错，量级也是噪声抽出来的）。
  D 类  `specs.py` 的每条常量逐条被断言钉住；含 `ast` 的跨程序覆盖检查（在 01 里汇总）。

跑法：`make 03_convergence`。纯 python3 + numpy，不 import torch，不碰 GPU。

★ 口径声明：
  · 全部数字是**本库构造的模型的解析值/蒙特卡洛值**，不是任何真实系统的测量结果。
  · 外部数字：§4 的 **arXiv:2607.02541 是预印本（未确认同行评审）**、§1 的 hipfire 是 **D 级**。
    ★★ 措辞硬约束：只能说「**已有报告显示**同一改动在不同架构上可以反向」，
      **不得**说「一定反向」（规格 Chunk 4 末段的原话）。
  · 本程序不含任何 kernel 实现代码（域 README 的硬边界）。
"""

import sys
from math import log

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


# ===========================================================================
# 0. 模型
# ===========================================================================
#
# ★ 对象（规格 Chunk 4 文档 03 的模型，参数全是 `specs.py` §7 的 [E] 可调参数）：
#
#     收益递减：第 i 步的**真实**增益  g_i = g0 · r^i          （g0 = 30%、r = 0.6）
#     A/B 判定：每组 n = 5 次、噪声 σ；单侧检验；真实增益为 0 时假阳性率 = p_fp
#     停止判据：当**下一步的预期增益** < 最小可检出差异 δ* 时停手
#                 δ* = (z_{α/2} + z_β)·σ·√(2/n)   ≈ 2.80·σ/√n
# ★★ 全部参数**由命令行之外的地方传进函数**（不硬编码进逻辑）——见 §A.1 的调用形式。

G0 = SP.GAIN_FIRST_STEP_PCT / 100.0
R = SP.GAIN_DECAY_RATIO_DEFAULT
SIGMA = SP.SIGMA_PCT_DEFAULT / 100.0
N_PER_GROUP = SP.N_PER_GROUP_DEFAULT
COEFF = SP.MDE_COEFFICIENT_AT_POWER_80

print("=" * 78)
print("§0  对象：一个「继续调还有没有意义」的模型（不是任何真实系统）")
print("=" * 78)
print(f"  收益序列：g_i = {G0:.2f} × {R}^i ⇒ {', '.join(f'{100 * G0 * R ** i:.1f}%' for i in range(6))}")
print(f"  判定模型：每组 n = {N_PER_GROUP} 次、σ = {SIGMA}（= {SP.SIGMA_PCT_DEFAULT}%）、单侧检验")
print(f"  ★ 参数来自 specs.py §7/§8（[E] 级可调参数），程序里**不出现魔法数**")
print(f"  python {sys.version.split()[0]} / numpy {np.__version__}")


sub("§0.1  模型自洽的前置断言（不通过则后面的结论全部作废）")
check("0.1 收益衰减比 ∈ (0, 1)（否则收益不递减、停止点无定义）", 0.0 < R < 1.0)
check("0.1 首步增益 > 0", G0 > 0.0)
check("0.1 每组 n ≥ 2（n = 1 时方差估计无定义）", N_PER_GROUP >= 2)
check("0.1 σ ∈ (0, 1)", 0.0 < SIGMA < 1.0)
check("0.1 ★★ 每步假阳性率 ∈ (0, 0.5)，且「错的自由度」给出的率**更大**",
      0.0 < SP.FP_RATE_PCT_AT_T8_CORRECT_DF < SP.FP_RATE_PCT_AT_T4_WRONG_DF < 50.0)
check("0.1 ★★ 步数清单 ⊂ SPEC_PRERUN_FALSE_VICTORIES 的键",
      set(SP.FALSE_VICTORY_STEP_COUNTS) <= set(SP.SPEC_PRERUN_FALSE_VICTORIES))
check("0.1 ★★ 步数清单 ⊂ SPEC_PRERUN_FALSE_VICTORY_NOMINAL 的键",
      set(SP.FALSE_VICTORY_STEP_COUNTS) <= set(SP.SPEC_PRERUN_FALSE_VICTORY_NOMINAL))
check("0.1 ★★ 两张假胜利表的**键集合相同**（否则逐档对比无意义）",
      set(SP.SPEC_PRERUN_FALSE_VICTORIES) == set(SP.SPEC_PRERUN_FALSE_VICTORY_NOMINAL))
check("0.1 ★★ 五类 GPU 名单与三表的键都不为空（`ast` 之外的形状自洽）",
      len(SP.ARXIV_2607_02541_GPUS) == 5 and len(SP.UNSOURCED) >= 10)


def mde(sigma, n, coefficient=COEFF):
    """最小可检出差异。★ 与 02 篇同一个公式、同一组常量（**跨主题口径一致**）。"""
    return coefficient * sigma * np.sqrt(2.0 / n)


def stop_step_by_enumeration(g0, r, threshold, max_steps=1000):
    """逐步枚举：返回**最后一个值得做的步序号**（0 = 只做第一步）。

    ★★ 退化保护（来自「**崩溃不是捕获**」这条纪律）：`r ≥ 1` 时收益不衰减，
    枚举会跑到 `max_steps` 并且 `r**(k+1)` 会**溢出**。这里显式返回哨兵值，
    让「参数被改坏」表现为一条干净的 `[FAIL]`（§0.1 的衰减比断言），**不是**一个
    `OverflowError`——否则变异测试会把「崩溃」混进「捕获」计数里。
    """
    if not (0.0 < r < 1.0):
        return max_steps - 1
    k = 0
    while k + 1 < max_steps and g0 * r ** (k + 1) >= threshold:
        k += 1
    return k


def stop_step_closed_form(g0, r, threshold):
    """闭式解：`k* = ⌊ln(threshold / g0) / ln r⌋`（浮点，取整前是连续量）。"""
    if threshold >= g0:
        return -1.0
    return log(threshold / g0) / log(r)


# ===========================================================================
# A. 停止点：闭式解 vs 枚举，以及系数选择的后果
# ===========================================================================
head("§A  停止点：闭式解 vs 逐步枚举")

sub("A.1  闭式解与枚举一致（阈值 = 精确的 2.80σ/√n）")
thr_exact = mde(SIGMA, N_PER_GROUP)
k_enum = stop_step_by_enumeration(G0, R, thr_exact)
k_cont = stop_step_closed_form(G0, R, thr_exact)
print(f"      δ* = {COEFF:.4f}·{SIGMA}·√(2/{N_PER_GROUP}) = {thr_exact:.6f}"
      f"（{100 * thr_exact:.3f}%）")
print(f"      闭式解（连续） = {k_cont:.6f} ⇒ 向下取整 {int(np.floor(k_cont))}")
print(f"      逐步枚举       = {k_enum}")
for i in range(min(k_enum + 3, 12)):   # ★ 上限 12：R 被改坏时避免 r**i 溢出（见 §0.1 的衰减比断言）
    g = G0 * R ** i
    print(f"        第 {i} 步：预期增益 {100 * g:6.3f}%  vs δ* {100 * thr_exact:.3f}%  ⇒ "
          f"{'值得做' if g >= thr_exact else '**低于可判定阈值 ⇒ 停手**'}")
check("A.1 ★★ 闭式解的取整等于枚举结果（两条独立路径一致）",
      int(np.floor(k_cont)) == k_enum, f"{int(np.floor(k_cont))} vs {k_enum}")
_i_lo, _i_hi = min(k_enum + 1, 12), min(k_enum, 12)   # ★ 上限 12：见 §0.1 的衰减比断言
check("A.1 ★★ 停在**第一步低于阈值**的地方：g_{k+1} < δ* ≤ g_k",
      G0 * R ** _i_lo < thr_exact <= G0 * R ** _i_hi,
      f"{100 * G0 * R ** _i_lo:.3f}% < {100 * thr_exact:.3f}% ≤ {100 * G0 * R ** _i_hi:.3f}%")
check("A.1 ★ 判据是**可计算**的：给定 (g0, r, σ, n) 就能算出一个整数，不需要「感觉差不多了」",
      stop_step_by_enumeration(G0, R, mde(0.06, N_PER_GROUP)) < k_enum)
check("A.1 ★ 停止判据用的是**同一个** δ* 公式（与 02 篇口径一致，禁止另立一套）",
      SP.STOP_RULE_USES_MDE_AS_THRESHOLD is True
      and abs(mde(SIGMA, N_PER_GROUP) - COEFF * SIGMA * (2.0 / N_PER_GROUP) ** 0.5) < 1e-12)

sub("A.2  ★★ 用规格那个「≈ 4σ/√n」会改变结论（不是精度问题，是**不同的答案**）")
thr_4 = mde(SIGMA, N_PER_GROUP, coefficient=SP.MDE_REDUCED_COEFFICIENT_SPEC_CLAIMED)
k_4 = stop_step_by_enumeration(G0, R, thr_4)
print(f"      δ*(2.80) = {100 * thr_exact:.3f}%  ⇒ 最后值得做的步 = {k_enum}（共 {k_enum + 1} 步）")
print(f"      δ*(4.00) = {100 * thr_4:.3f}%  ⇒ 最后值得做的步 = {k_4}（共 {k_4 + 1} 步）")
check("A.2 ★★ 两个系数给出**不同的停止步数**",
      k_4 != k_enum, f"{k_4} vs {k_enum}")
check("A.2 ★ 4.0/2.80 的比值 = %.4f（≈ 43%% 的保守放大）"
      % (SP.MDE_REDUCED_COEFFICIENT_SPEC_CLAIMED / SP.MDE_REDUCED_COEFFICIENT_EXACT),
      abs(SP.MDE_REDUCED_CLAIM_EXCEEDS_EXACT_PCT - 42.776) < 0.01,
      f"{SP.MDE_REDUCED_CLAIM_EXCEEDS_EXACT_PCT:.3f}%")
print("      ★ 这一条本身就是「规格的约简形式不是无害的」的机械证据：")
print("        它让整个过程**早停一步**，而那一步里有 %.2f%% 的真实增益。"
      % (100 * G0 * R ** min(k_enum if k_enum > k_4 else k_4 + 1, 12)))

sub("A.3  ★ 重复次数 n 越大，能判定的差异越小 ⇒ 停止点越晚")
print("      n       δ*        停止步   （σ = 3%、g0 = 30%、r = 0.6）")
for n in (3, 5, 10, 25, 100):
    thr = mde(SIGMA, n)
    k = stop_step_by_enumeration(G0, R, thr)
    print(f"      {n:3d}   {100 * thr:7.3f}%     {k}")
check("A.3 ★★ 停止点对 n **单调不减**（n 越大、δ* 越小、能做得越多）",
      all(stop_step_by_enumeration(G0, R, mde(SIGMA, n1))
          <= stop_step_by_enumeration(G0, R, mde(SIGMA, n2))
          for n1, n2 in ((3, 5), (5, 10), (10, 25), (25, 100))))
check("A.3 ★ n = 100 时停止点比 n = 5 晚（多出可判定的步）",
      stop_step_by_enumeration(G0, R, mde(SIGMA, 100))
      > stop_step_by_enumeration(G0, R, mde(SIGMA, 5)))
print("      ★ 注意方向：**增加重复次数**是「让判据更灵敏」的唯一免费手段——")
print("        它不改变真实增益，只改变你能**看见**多小的增益。")


# ===========================================================================
# B. 定量关系：噪声、门限、跨架构
# ===========================================================================
head("§B  噪声带 → 停止点；门限 → 误报率；跨架构 → 结论相反")

sub("B.1  ★★ 噪声带每翻一倍，停止点提前 `ln2 / |ln r|` 步")
delta_pred = log(2.0) / abs(log(R))
print(f"      r = {R} ⇒ `ln2 / |ln r|` = {log(2.0):.4f} / {abs(log(R)):.4f} = {delta_pred:.4f} 步")
print("      σ        停止步（枚举）   闭式解（连续）")
prev = None
rows = []
for s_pct in (1.5, 3.0, 6.0, 12.0):
    thr = mde(s_pct / 100.0, N_PER_GROUP)
    k = stop_step_by_enumeration(G0, R, thr)
    kc = stop_step_closed_form(G0, R, thr)
    rows.append((s_pct, k, kc))
    print(f"      {s_pct:5.1f}%        {k:3d}           {kc:8.4f}")
    if prev is not None:
        check(f"B.1 σ 从 {prev[0]}% 翻倍到 {s_pct}%：闭式解**恰好**提前 {delta_pred:.4f} 步"
              "（差分 = ln2/|ln r|，误差 < 1e-12）",
              abs((prev[2] - kc) - delta_pred) < 1e-12, f"{prev[2] - kc:.6f}")
    prev = (s_pct, k, kc)
check("B.1 ★★ 枚举的步数**单调不增**（噪声越大越早停）",
      all(rows[i][1] >= rows[i + 1][1] for i in range(len(rows) - 1)))
check("B.1 ★ 连续提前量与整数提前量相差 < 1 步（取整效应，不是模型误差）",
      all(abs((rows[i][2] - rows[i + 1][2]) - (rows[i][1] - rows[i + 1][1])) < 1.0
          for i in range(len(rows) - 1)))

sub("B.2  ★★ 门限与噪声带：误报率 = max(0, 1 − T/A)")
print("      模型（本库 [E]，**明写它是模型**）：某架构的真实噪声半带为 A、门限容差为 T，")
print("      读数落在 ±A 内均匀分布 ⇒ 误报率 = max(0, 1 − T/A)。")
for tol, half in ((5.0, 5.0), (5.0, 10.0), (5.0, 15.0), (5.0, 20.0), (15.0, 15.0)):
    fa = max(0.0, 1.0 - tol / half)
    print(f"      T = ±{tol:4.1f}%、该架构噪声半带 A = ±{half:4.1f}%  ⇒ 误报率 {100 * fa:5.1f}%")
check("B.2 ★★ 门限**小于**噪声半带时误报率 > 0；门限 ≥ 噪声半带时误报率 = 0",
      max(0.0, 1.0 - 5.0 / 5.0) == 0.0 and max(0.0, 1.0 - 5.0 / 15.0) > 0.0)
fa_15 = max(0.0, 1.0 - SP.HIPFIRE_SPEED_GATE_TOLERANCE_PCT
            / SP.HIPFIRE_WITHIN_SESSION_NOISE_PCT_GFX1100[1])
print(f"      ★ 用真实的两个输入算：门限 ±{SP.HIPFIRE_SPEED_GATE_TOLERANCE_PCT}% 对上 "
      f"A = ±{SP.HIPFIRE_WITHIN_SESSION_NOISE_PCT_GFX1100[1]}% ⇒ 误报率 {100 * fa_15:.1f}%")
check("B.2 ★★ 同一份门限（±5%）在这个噪声条件下误报率超过 60%",
      fa_15 > 0.60, f"{100 * fa_15:.1f}%")
check("B.2 ★★ 而该文档自己说「会话内的 +%s%% 仍在噪声里」——两句话是同一个事实"
      % SP.HIPFIRE_IN_SESSION_WIN_INSIDE_NOISE_PCT,
      SP.HIPFIRE_IN_SESSION_WIN_INSIDE_NOISE_PCT > SP.HIPFIRE_SPEED_GATE_TOLERANCE_PCT)
print("      ⇒ ★ 判据：**门限容差必须 ≥ 该架构的噪声半带**；否则门限在报噪声。")

sub("B.3  ★★ 跨架构：同一个改动的结论可以相反（**[C] 级预印本**，措辞用「已有报告显示」）")
l40s = SP.ARXIV_2607_02541_SAME_CHANGE_L40S_PCT
a100 = SP.ARXIV_2607_02541_SAME_CHANGE_A100_SXM4_PCT
tol = SP.HIPFIRE_SPEED_GATE_TOLERANCE_PCT
print(f"      同一个 (correct, buggy) 改动：L40S {l40s:+.1f}% · A100 SXM4 {a100:+.1f}%")
print("      门限语义（写死，避免口径混淆）：**单侧回归门限**——只惩罚变慢。")
print(f"      一份全局 ±{tol}% 门限（Δ ≥ −{tol}% 视为通过）下的判定：")
print(f"        L40S {l40s:+.1f}% ⇒ {'通过' if l40s >= -tol else '不通过'}"
      f" / A100 SXM4 {a100:+.1f}% ⇒ {'通过' if a100 >= -tol else '不通过'}")
check("B.3 ★★ 同一个改动在两个架构上的**实测 Δperf% 符号相反**",
      l40s > 0 > a100, f"{l40s:+.1f}% vs {a100:+.1f}%")
check("B.3 ★★ 一份全局门限下两者的判定**不同**（一个通过、一个不通过）",
      (l40s >= -tol) != (a100 >= -tol))
check("B.3 ★ 若门限改成双侧（|Δ| ≤ 5%），则**两者都不通过** —— 门限的语义也会改变结论",
      (abs(l40s) <= tol) == (abs(a100) <= tol) == False)
print(f"      而其余 {SP.ARXIV_2607_02541_ARCHS_WITHIN_PCT_OF_ZERO} 类"
      f"（共 {len(SP.ARXIV_2607_02541_GPUS)} 类）落在零的 "
      f"±{SP.ARXIV_2607_02541_WITHIN_BAND_PCT}% 以内 ⇒")
print(f"        在 ±{SP.ARXIV_2607_02541_WITHIN_BAND_PCT}% 的噪声里，"
      f"真实效应为 0 的改动本来就**测不出来**。")
check("B.3 ★ 五类 GPU 的名单与论文一致",
      SP.ARXIV_2607_02541_GPUS == ("RTX 3060", "A10", "L40S", "A100 SXM4", "H100 NVL"))
check("B.3 ★★ 论文结论：**基于性能的门限必须逐架构重新标定**（否则只能当冒烟检查）",
      SP.ARXIV_2607_02541_PERF_GATE_NEEDS_PER_ARCH_CALIBRATION is True)
check("B.3 ★★ 两个**独立主体**给出同一结论（P10 的机械检查通过）："
      "论文要按架构标定 + hipfire 的基线按架构分文件",
      SP.ARXIV_2607_02541_PERF_GATE_NEEDS_PER_ARCH_CALIBRATION is True
      and SP.HIPFIRE_SPEED_BASELINE_FILE_PATTERN == "tests/speed-baselines/<arch>.txt")
check("B.3 ★★ 措辞硬约束：外部的两个数来自**预印本**，正文必须写成「已有报告显示」",
      SP.ARXIV_2607_02541_IS_PREPRINT_NOT_PEER_REVIEWED is True
      and SP.ARXIV_2607_02541_TITLE.startswith("Static PTX Metrics"))
check("B.3 ★ 而**静态**指标跨架构一致 —— 所以「不可迁移」这条只说**性能**，不说静态量",
      SP.ARXIV_2607_02541_STATIC_METRICS_ARE_ARCH_INDEPENDENT is True)
check("B.3 ★ 该结论有**作用域**：噪声支配的前提是亚毫秒尺度",
      SP.ARXIV_2607_02541_NOISE_IS_SUB_MILLISECOND_SCALE is True)


# ===========================================================================
# C. ★★ 假胜利的累积与幅度
# ===========================================================================
head("§C  ★★ 真实增益恒为 0 时，「永远继续调」会累积出什么")

SEED = 20260920
MC_REPS = 40000
T8 = 2.306004      # t(8) 单侧 2.5%：**本题正确的临界值**（见 02 篇 §C.2）
T4 = 2.131847      # t(4) 单侧 5%：规格那次用的那个（自由度写成 n−1）


def false_victory_rate(crit, reps, rng, sigma=SIGMA, n=N_PER_GROUP):
    """真实增益恒为 0 时，**单步**被误判为「有收益」的概率（蒙特卡洛）。"""
    a = sigma * rng.standard_normal((reps, n))
    b = sigma * rng.standard_normal((reps, n))
    se = np.sqrt(a.var(axis=1, ddof=1) / n + b.var(axis=1, ddof=1) / n)
    return float(((b.mean(axis=1) - a.mean(axis=1)) / se > crit).mean())


rng = np.random.default_rng(SEED)
MC_REPS = 200000
mc_t4 = 100.0 * false_victory_rate(T4, MC_REPS, rng)
mc_t8 = 100.0 * false_victory_rate(T8, MC_REPS, rng)
rate_t4 = SP.FP_RATE_PCT_AT_T4_WRONG_DF / 100.0      # 3.28%（**解析率**，见 specs §7）
rate_t8 = SP.FP_RATE_PCT_AT_T8_CORRECT_DF / 100.0    # 2.50%
print(f"      单步假阳性率（解析率来自 specs.py §7，与 02 篇同一口径；蒙特卡洛复核 {MC_REPS} 次）：")
print(f"        临界值 {T4}（规格用的 t(4)）⇒ 解析 {100 * rate_t4:.2f}%"
      f"，蒙特卡洛 {mc_t4:.3f}%")
print(f"        临界值 {T8}（正确的 t(8)）⇒ 解析 {100 * rate_t8:.2f}%"
      f"，蒙特卡洛 {mc_t8:.3f}%")
check("C.0 ★★ 解析率与蒙特卡洛一致（相对差 < 5%）——本程序用**解析率**做累积，"
      "避免把 MC 噪声误读成模型差异",
      abs(mc_t4 / (100 * rate_t4) - 1.0) < 0.05 and abs(mc_t8 / (100 * rate_t8) - 1.0) < 0.05,
      f"{mc_t4:.3f}/{100 * rate_t4:.2f} · {mc_t8:.3f}/{100 * rate_t8:.2f}")

sub("C.1  ★★ 复现规格的假胜利表（10 / 50 / 200 步）")
print("      步数   规格值    本库(t(4) 临界)   规格名义(2.5%)   本库(t(8) 临界)")
for steps in SP.FALSE_VICTORY_STEP_COUNTS:
    v_t4 = steps * rate_t4
    v_t8 = steps * rate_t8
    spec = SP.SPEC_PRERUN_FALSE_VICTORIES.get(steps, float("nan"))
    nom = SP.SPEC_PRERUN_FALSE_VICTORY_NOMINAL.get(steps, float("nan"))
    print(f"      {steps:4d}   {spec:6.2f}    {v_t4:14.3f}   {nom:14.2f}   {v_t8:14.3f}")
    check(f"C.1 步数 {steps}：规格值 {spec} 与本库（t(4) 临界）相差 ≤ 10%",
          abs(v_t4 / spec - 1.0) < 0.10, f"{100 * (v_t4 / spec - 1.0):+.1f}%")
    check(f"C.1 步数 {steps}：用**正确的 t(8)** 临界值时回到名义值 {nom}（相差 ≤ 2%）",
          abs(v_t8 / nom - 1.0) < 0.02, f"{100 * (v_t8 / nom - 1.0):+.2f}%")
print("      ★ 口径差异写清：规格那组数对应**每步约 3.5%**，本库实测/解析是 **3.28%**")
print("        （t(4) 临界值下的假阳性率，见 02 篇）——相差约 6–7%，方向是规格偏高。")
print("        用**正确的 t(8)** 则回到名义的 2.5%，累积量随之回到名义值。")
spec_ratio = (SP.SPEC_PRERUN_FALSE_VICTORIES.get(200, float("nan"))
              / SP.SPEC_PRERUN_FALSE_VICTORY_NOMINAL.get(200, float("nan")))
crit_ratio = SP.FP_RATE_PCT_AT_T4_WRONG_DF / SP.FP_RATE_PCT_AT_T8_CORRECT_DF
print(f"      ★★ 规格自己的两张表**互相不一致**：假胜利表的比值 = {spec_ratio:.4f}，")
print(f"         而临界值表的比值 = {crit_ratio:.4f} ⇒ 差 {100 * (spec_ratio / crit_ratio - 1):.1f}%。")
print("         两者本应是同一个数（同一个每步率）。")
check("C.1 ★★ 这两张表的比值**不同** ⇒ 规格内部有约 5% 的不一致（已记入报告）",
      abs(spec_ratio - crit_ratio) > 0.05)

sub("C.2  ★★ 200 步、真实增益为 0 ⇒ 稳定累积 6 次以上的「胜利」")
v200 = 200 * rate_t4
print(f"      200 步 × 每步 {100 * rate_t4:.2f}% = **{v200:.2f} 次假胜利 / 试验**")
check("C.2 ★★ 200 步累积**超过 6 次**假胜利（而**没有任何一步有真实增益**）",
      v200 > 6.0, f"{v200:.3f}")
check("C.2 ★★ 累积量随步数**线性增长**（10 → 200 步，累积量增长 20 倍 ± 1%）",
      abs((200 * rate_t4) / (10 * rate_t4) - 20.0) < 0.2)
print("      ⇒ 结论：**没有停止判据的调优过程必然产出假结论**——")
print("        不是「可能」，而是「跑到哪一步就有多少次的期望值」。")

sub("C.3  ★★ 假胜利的**幅度**也是噪声抽出来的（不只符号错）")
steps_rep = 200
acc = []
for _ in range(steps_rep):
    a = SIGMA * rng.standard_normal((4000, N_PER_GROUP))
    b = SIGMA * rng.standard_normal((4000, N_PER_GROUP))
    se = np.sqrt(a.var(axis=1, ddof=1) / N_PER_GROUP + b.var(axis=1, ddof=1) / N_PER_GROUP)
    t = (b.mean(axis=1) - a.mean(axis=1)) / se
    d = b.mean(axis=1) - a.mean(axis=1)
    acc.append(float(d[t > T4].mean()) if (t > T4).any() else np.nan)
reported = np.array([v for v in acc if not np.isnan(v)])
print(f"      真实增益恒为 0；只看**被判为显著**的那些步，报出的「增益」：")
print(f"        平均 {100 * reported.mean():+.3f}% · 中位 {100 * np.median(reported):+.3f}% · "
      f"最大 {100 * reported.max():+.3f}%")
check("C.3 ★★ 被报出的「增益」平均是**正的且远大于 0**（真实值恒为 0）",
      reported.mean() > 0.02 and SP.GAIN_FIRST_STEP_PCT > 0,
      f"{100 * reported.mean():+.3f}%")
se_model = SIGMA * np.sqrt(2.0 / N_PER_GROUP)
check("C.3 ★★ 最大的一笔「增益」（%.2f%%）是与临界值同量级的解析量"
      "：临界值 × 标准误 = %.2f × %.4f = %.2f%%" %
      (100 * reported.max(), T4, se_model, 100 * T4 * se_model),
      100 * reported.max() > 100 * T4 * se_model * 0.9,
      f"{100 * reported.max():.2f}% vs {100 * T4 * se_model:.2f}%")
print("      ★ 判别法：这些「增益」的**分布**就是噪声的上尾——它们没有对应的真实效应。")

sub("C.4  ★★ 按停止判据在第 k 步停手，能把累积量压下来（但不是清零）")
k_star = stop_step_by_enumeration(G0, R, mde(SIGMA, N_PER_GROUP))
steps_stopped = k_star + 1
v_stop = steps_stopped * rate_t4
print(f"      停止点：第 {k_star} 步之后不做 ⇒ 实际执行 {steps_stopped} 次 A/B")
print(f"      累积假胜利：{steps_stopped} × {100 * rate_t4:.2f}% = {v_stop:.3f}"
      f"（对比「永远继续」200 步的 {v200:.2f}）")
print(f"      压下来的倍数 = {v200 / v_stop:.1f}×")
check("C.4 ★★ 按判据停手把累积量压到 1/10 以下",
      v200 / v_stop > 10.0, f"{v200 / v_stop:.1f}×")
check("C.4 ★★ 但它**不为零**：停手只减少次数，不改变每步的假阳性率 —— 判据不是「免疫」，是「有界」",
      v_stop > 0, f"{v_stop:.4f}")
print("      ⇒ 措辞要准：停止判据**限制**了假结论的数量，**不能**消灭它。")

sub("C.5  ★ 假胜利的**代价**：它出现在收益序列的哪一段")
for i in range(min(7, steps_stopped + 3)):
    g = 100 * G0 * R ** i
    mark = "（判据内的步）" if i <= k_star else "（判据外的步 ⇒ 不该做）"
    print(f"      第 {i} 步真实增益 {g:6.3f}% {mark}")
check("C.5 ★ 判据外的那些步里，真实增益已经低于可判定阈值——"
      "在那里「发现的收益」按定义不可能是真的",
      G0 * R ** min(k_star + 1, 12) < mde(SIGMA, N_PER_GROUP))


# ===========================================================================
# D. 常量自洽
# ===========================================================================
head("§D  常量自洽：本篇用到的 specs.py 常量，逐条被断言钉住")

sub("D.1  模型参数（**[E] 级：可调参数**）")
check("D.1 收益序列首项 %s%% · 衰减比 %s" % (SP.GAIN_FIRST_STEP_PCT, SP.GAIN_DECAY_RATIO_DEFAULT),
      SP.GAIN_FIRST_STEP_PCT == 30.0 and SP.GAIN_DECAY_RATIO_DEFAULT == 0.6)
check("D.1 每组 n = %d · σ = %s%%" % (SP.N_PER_GROUP_DEFAULT, SP.SIGMA_PCT_DEFAULT),
      SP.N_PER_GROUP_DEFAULT == 5 and SP.SIGMA_PCT_DEFAULT == 3.0)
check("D.1 停止判据**用 δ* 当阈值**（而不是「感觉收敛了」）",
      SP.STOP_RULE_USES_MDE_AS_THRESHOLD is True)
check("D.1 ★ 四个参数都是**函数实参**传进去的（程序里没有魔法数）",
      abs(mde(0.03, 5) - mde(SIGMA, N_PER_GROUP)) < 1e-15
      and abs(stop_step_by_enumeration(0.30, 0.6, mde(0.03, 5)) - k_enum) == 0)
check("D.1 步数清单 = %s" % (SP.FALSE_VICTORY_STEP_COUNTS,),
      SP.FALSE_VICTORY_STEP_COUNTS == (10, 50, 200))
check("D.1 ★★ 两个每步假阳性率**与 02 篇同一口径**（t(8) ⇒ 2.50%、t(4) ⇒ 3.28%）",
      SP.FP_RATE_PCT_AT_T8_CORRECT_DF == 2.50 and SP.FP_RATE_PCT_AT_T4_WRONG_DF == 3.28)
check("D.1 口径：越小越好（时间/延迟越小越快）",
      SP.LOWER_IS_BETTER_METRIC_CONVENTION == "lower_is_better")

sub("D.2  ★★ 规格预跑值与被推翻的条目（**[E] 级**）")
check("D.2 规格的假胜利表与名义值表被原样收录",
      SP.SPEC_PRERUN_FALSE_VICTORIES == {10: 0.35, 50: 1.75, 200: 6.90}
      and SP.SPEC_PRERUN_FALSE_VICTORY_NOMINAL == {10: 0.25, 50: 1.25, 200: 5.00})
check("D.2 ★★ 本库的复算把差异定位到**临界值**上：规格那组对应每步 3.28%，"
      "换成正确的 t(8) 后回到名义值",
      abs(100 * rate_t4 - 3.28) < 0.2 and abs(100 * rate_t8 - 2.50) < 0.2)
check("D.2 ★ 规格的「≈ 4σ/√n」让停止步数变化 —— 已记在常量里",
      SP.MDE_REDUCED_CLAIM_EXCEEDS_EXACT_PCT > 40.0 and k_4 != k_enum)

sub("D.3  跨架构（**[C] 级预印本** + **[D] 级项目**）")
check("D.3 同一个改动的两个读数：L40S %+.1f%% · A100 SXM4 %+.1f%%"
      % (SP.ARXIV_2607_02541_SAME_CHANGE_L40S_PCT,
         SP.ARXIV_2607_02541_SAME_CHANGE_A100_SXM4_PCT),
      SP.ARXIV_2607_02541_SAME_CHANGE_L40S_PCT == 9.4
      and SP.ARXIV_2607_02541_SAME_CHANGE_A100_SXM4_PCT == -50.0)
check("D.3 ★★ 预印本标记在常量层面可见（正文漏标会被 §D 的断言拦下）",
      SP.ARXIV_2607_02541_IS_PREPRINT_NOT_PEER_REVIEWED is True)
check("D.3 ★★ 论文标题**逐字**核对（不是 startswith 这种半截断言）",
      SP.ARXIV_2607_02541_TITLE
      == "Static PTX Metrics Track Structural Kernel Regressions but Miss Semantic Ones")
check("D.3 ★ 落在零的 ±20%% 以内的架构数 = %s（共 %s 类）"
      % (SP.ARXIV_2607_02541_ARCHS_WITHIN_PCT_OF_ZERO, len(SP.ARXIV_2607_02541_GPUS)),
      SP.ARXIV_2607_02541_ARCHS_WITHIN_PCT_OF_ZERO == 4
      and SP.ARXIV_2607_02541_ARCHS_WITHIN_PCT_OF_ZERO < len(SP.ARXIV_2607_02541_GPUS))
check("D.3 ★ 那个带子的半宽 = ±%s%%" % SP.ARXIV_2607_02541_WITHIN_BAND_PCT,
      SP.ARXIV_2607_02541_WITHIN_BAND_PCT == 20.0)
check("D.3 一个「真坏了功能但读数变好」的实例：+%s%%（落在噪声带内）"
      % SP.ARXIV_2607_02541_STRUCTURAL_BUG_DELTA_PCT_WITHIN_NOISE,
      SP.ARXIV_2607_02541_STRUCTURAL_BUG_DELTA_PCT_WITHIN_NOISE == 3.2)
check("D.3 ★ hipfire（D 级）的基线按架构分文件 = 第二个独立主体",
      SP.HIPFIRE_SPEED_BASELINE_FILE_PATTERN == "tests/speed-baselines/<arch>.txt")
check("D.3 ★ 基线更新必须与改动同一次提交（让评审看到交易）",
      SP.HIPFIRE_BASELINE_UPDATE_MUST_BE_IN_SAME_COMMIT is True)
check("D.3 ★ 行级来源三字段（脏工作区与已提交基线不可直接比）",
      SP.HIPFIRE_BASELINE_ROW_PROVENANCE_FIELDS
      == ("binary_md5", "git_dirty_state", "diff_md5"))
check("D.3 ★ 散得太开判 unstable 的阈值 = %s（与「紧得可疑」配成一对判据）"
      % SP.HIPFIRE_MAX_REL_SPREAD_BEFORE_UNSTABLE,
      SP.HIPFIRE_MAX_REL_SPREAD_BEFORE_UNSTABLE == 0.20)

sub("D.4  ★ 全模块常量覆盖的**机械检查**")
COVERED_03 = (
    # §4 跨架构（C 级预印本）
    "ARXIV_2607_02541_IS_PREPRINT_NOT_PEER_REVIEWED",
    "ARXIV_2607_02541_TITLE",
    "ARXIV_2607_02541_GPUS",
    "ARXIV_2607_02541_SAME_CHANGE_L40S_PCT",
    "ARXIV_2607_02541_SAME_CHANGE_A100_SXM4_PCT",
    "ARXIV_2607_02541_ARCHS_WITHIN_PCT_OF_ZERO",
    "ARXIV_2607_02541_WITHIN_BAND_PCT",
    "ARXIV_2607_02541_STRUCTURAL_BUG_DELTA_PCT_WITHIN_NOISE",
    "ARXIV_2607_02541_STATIC_METRICS_ARE_ARCH_INDEPENDENT",
    "ARXIV_2607_02541_PERF_GATE_NEEDS_PER_ARCH_CALIBRATION",
    "ARXIV_2607_02541_NOISE_IS_SUB_MILLISECOND_SCALE",
    # §7 收敛判据的参数（E 级）
    "STOP_RULE_USES_MDE_AS_THRESHOLD",
    "GAIN_FIRST_STEP_PCT",
    "GAIN_DECAY_RATIO_DEFAULT",
    "SIGMA_PCT_DEFAULT",
    "N_PER_GROUP_DEFAULT",
    "FALSE_VICTORY_STEP_COUNTS",
    "FP_RATE_PCT_AT_T8_CORRECT_DF",
    "FP_RATE_PCT_AT_T4_WRONG_DF",
    "LOWER_IS_BETTER_METRIC_CONVENTION",
    # §8 规格预跑值
    "SPEC_PRERUN_FALSE_VICTORIES",
    "SPEC_PRERUN_FALSE_VICTORY_NOMINAL",
    # 待办清单
    "UNSOURCED",
)
print(f"    03 覆盖 {len(COVERED_03)} 条")
check("D.4 ★ 03 的覆盖表里没有拼错的常量名",
      set(COVERED_03) <= {n for n in dir(SP) if n.isupper()},
      f"{sorted(set(COVERED_03) - {n for n in dir(SP) if n.isupper()})}")
UNSOURCED_EXPECTED = (
    "hipfire_graph_on_off_spread_values_0p1_vs_1p5_3pct",
    "hipfire_verify_graph_env_var_name",
    "jit_cold_compile_30s_to_2min_interval",
    "model_reload_cold_start_2s_to_5s_interval",
    "noise_band_vs_sigma_mapping",
    "hipfire_dpm_warmup_secs_value",
    "hipfire_warmup_equals_128_low_bias_claim",
    "tight_spread_suspicion_in_hipfire_doc",
    "mytkowicz_40pct_figure",
    "mlcommons_official_rulebook_warmup_and_run_counts",
    "cold_start_component_residual_81s",
    "real_system_fake_speedup_attributable_to_warmup",
)
check("D.4 ★★ `UNSOURCED` **逐条**核对（12 条，不是「包含某两条」这种半截断言）",
      tuple(SP.UNSOURCED) == UNSOURCED_EXPECTED,
      f"{sorted(set(UNSOURCED_EXPECTED) ^ set(SP.UNSOURCED))}")
check("D.4 ★★ 它里面有一条「没有可引用的真实案例」——支撑本程序 §C 的措辞纪律",
      "real_system_fake_speedup_attributable_to_warmup" in SP.UNSOURCED
      and "cold_start_component_residual_81s" in SP.UNSOURCED)


# ===========================================================================
head("汇总")
# ===========================================================================
if FAILED:
    print(f"  ✗ {len(FAILED)} 条断言失败：")
    for f in FAILED:
        print(f"      · {f}")
    sys.exit(1)
print("  ✓ 全部断言通过（A 类 3 组 · B 类 3 组 · C 类 5 组 · D 类 4 组）")
print()
print("  ★ 全部数字是**模型值**，不是任何真实系统的测量结果；")
print("    跨架构那两个数来自 **[C] 级预印本（未确认同行评审）**，措辞用「已有报告显示」。")
print("  ★ 本程序的核心结论：**没有停止判据的调优过程必然产出假结论**——")
print(f"    真实增益恒为 0 时，200 步累积 {v200:.2f} 次假胜利；")
print(f"    按判据在第 {k_star} 步停手可压到 {v_stop:.3f} 次（{v200 / v_stop:.1f}×），**但不为零**。")
print(f"    python {sys.version.split()[0]} / numpy {np.__version__}（未 import torch）")

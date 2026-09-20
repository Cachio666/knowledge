#!/usr/bin/env python3
r"""03-optimization-priority —— 优化价值的排序，与「停止条件」的第一道闸。

对应文档：`../03-optimization-priority.md`

★ 本程序的可证伪断言：
  A 类  headroom gate 的**边界**：距上界恰好 1.300× 时，声称 1.300× 通过、1.301× 被拒；
        并用 vllm.cpp 的实测值走一遍闸（1.090× ≤ 1.218× 通过）
  B 类  ★ 用 vllm.cpp 的实例数据**逐条检验三条排序规则**，结论是
        **三条规则都没有复现文档记录的实际顺序**——并给出**临界工作量比值**；
        另验证「kernel 未覆盖的份额不会被更快的 kernel 恢复」这条上界
  C 类  构造一组候选：占比最高的那个按**两种**规则都排第 1，然后改它的**工作量估计**，
        排序**真的翻转**——这就是「按占比排序会出错」的可复现形态

⚠ **本程序的全部外部数字都是 D 级**（一个个人仓库的 CPU 剖面）。
  每一个出现的地方都写了「D 级」与「CPU」，**不含任何 GPU 数字**。

跑法：`make 03_optimization_priority`。纯 python3 + 标准库，不 import torch，不碰 GPU。
"""

import sys

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
head("§A  算法正确性：headroom gate 的边界")
# ===========================================================================
#
# ★ 闸的来源（**B 级**）：NVIDIA/TensorRT-LLM PR #18805 的变更说明里写了
#   `headroom_verdict` 成为一道闸——「**一个 kernel 加速项声称的加速比不得超过
#   该算子到其约束资源的距离**」。本程序把它落成一个可判定的比较。

EPS = 1e-9


def gate(headroom, claimed):
    """声称的加速比 ≤ 到约束资源的距离 ⇒ 通过。

    `headroom` = 上界 ÷ 当前（≥ 1）；`claimed` = 声称的加速比（≥ 1）。
    ★ 边界是**闭**的：恰好相等通过（`≤`，不是 `<`）。
    """
    return claimed <= headroom + EPS


sub("A.1  恰好 1.300× 通过，1.301× 被拒")
h = 1.300
print(f"    距上界 = {h:.3f}×")
for claim in (1.0, 1.3, 1.3000000001, 1.301, 1.5, 2.0):
    print(f"      声称 {claim:<14.10g} ⇒ {'通过' if gate(h, claim) else '拒绝'}")
check("A.1 声称 1.300 通过（边界闭合）", gate(1.3, 1.3))
check("A.1 声称 1.301 被拒", not gate(1.3, 1.301))
check("A.1 声称 1.0 通过（下界平凡情形）", gate(1.3, 1.0))
check("A.1 声称 2.0 被拒", not gate(1.3, 2.0))

sub("A.2  用 vllm.cpp 的实测值走一遍闸（**D 级 · CPU 剖面**）")
# ★ 条件必须连在一起写（specs.py §5）：aarch64 DGX · 20 线程 · --input-len 128
#   --output-len 1 · q8_0 · warm page cache · 1070 ms TTFT。
ceil_km = SP.VLLMCPP_CEILING_DERIVED["kMatmul"]
real_km = SP.VLLMCPP_CHOSEN_MEASURED_SPEEDUP_LOW
print(f"    {SP.VLLMCPP_CHOSEN_OP}：距其约束资源的上界 = 1/(1−0.179) = {ceil_km:.4f}×")
print(f"    实测（同二进制 CPU prefill A/B）= {real_km:.3f}×")
print(f"    闸：{real_km:.3f} ≤ {ceil_km:.4f} ⇒ "
      f"{'通过' if gate(ceil_km, real_km) else '拒绝'}")
check("A.2 实测 1.090× 通过闸（未超其到上界的距离）", gate(ceil_km, real_km))
check("A.2 若有人声称 1.5× 则被拒（超过 1.2180 的上界）", not gate(ceil_km, 1.5))
print(f"    ★ 实测/上界 = 1.090/{ceil_km:.4f} = {real_km/ceil_km:.2%} ——"
      f" 这是一个**低工作量改动**（一个 loader flag，构造上逐字节相同）"
      f"拿到的上界兑现率。")

sub("A.3  top-3 Pareto：只看前 3 个桶覆盖多少（D 级 · CPU 剖面）")
shares = sorted(SP.VLLMCPP_PREFILL_SHARES_PCT.values(), reverse=True)
top3 = sum(shares[:3])
print(f"    占比降序：{shares}")
print(f"    前 3 个桶合计 = {shares[0]} + {shares[1]} + {shares[2]} = {top3:.1f}%")
print(f"    原文记的「GEMMs 占 prefill 的 83%」= {SP.VLLMCPP_PREFILL_GEMM_TOTAL_PCT}%"
      f"（与 {top3:.1f}% 自洽，原表四个数各留一位小数）")
check("A.3 前 3 个桶之和与原文的 83% 相差在 1 个百分点内",
      abs(top3 - SP.VLLMCPP_PREFILL_GEMM_TOTAL_PCT) < 1.0, f"{top3:.1f}% vs 83%")
check("A.3 前 3 个桶之和**小于 100%**（剩下的桶不是零）", top3 < 100.0, f"{top3:.1f}%")


# ===========================================================================
head("§B  模型与公式：三条排序规则对同一份实例数据给出什么顺序")
# ===========================================================================
#
# ★ 数据（**全部 D 级**，来源 `specs.py` §5；条件是 aarch64 DGX 的 **CPU** prefill，
#   20 线程、`--input-len 128 --output-len 1`、q8_0、warm page cache、总 1070 ms）：
#     算子 A = kMatmulBTQuant  占比 50.4%  工作量档 MEDIUM–HIGH
#     算子 B = kMatmul         占比 17.9%  工作量档 LOW
#   文档记录的实际决策：**先做 B**（17.9%），把 A（50.4%）写成「the next lever」。

A_NAME = SP.VLLMCPP_DEFERRED_OP
B_NAME = SP.VLLMCPP_CHOSEN_OP
SHR = SP.VLLMCPP_PREFILL_SHARES_PCT
EFF = SP.EFFORT_RANK_MAPPING
sh_a, sh_b = SHR[A_NAME], SHR[B_NAME]
ef_a = EFF[SP.VLLMCPP_DEFERRED_EFFORT_TIER]
ef_b = EFF[SP.VLLMCPP_CHOSEN_EFFORT_TIER]
d_a = SP.VLLMCPP_CEILING_DERIVED[A_NAME]
d_b = SP.VLLMCPP_CEILING_DERIVED[B_NAME]

print(f"    A = {A_NAME:16s} 占比 {sh_a:5.1f}%  工作量档 {SP.VLLMCPP_DEFERRED_EFFORT_TIER}"
      f"（秩 {ef_a}）  到上界的距离 {d_a:.4f}×")
print(f"    B = {B_NAME:16s} 占比 {sh_b:5.1f}%  工作量档 {SP.VLLMCPP_CHOSEN_EFFORT_TIER}"
      f"（秩 {ef_b}）  到上界的距离 {d_b:.4f}×")

sub("B.1  三条规则各自的第一名")
r1_a, r1_b = sh_a, sh_b
r2_a, r2_b = sh_a / ef_a, sh_b / ef_b
r3_a, r3_b = d_a * sh_a / ef_a, d_b * sh_b / ef_b
RULES = [
    ("① 按占比", r1_a, r1_b),
    ("② 占比 ÷ 工作量", r2_a, r2_b),
    ("③ 距离 × 占比 ÷ 工作量", r3_a, r3_b),
]
for label, va, vb in RULES:
    first = "A" if va > vb else "B"
    print(f"    {label:24s}  A = {va:8.4f}   B = {vb:8.4f}   ⇒ 第一名 **{first}**")
print(f"    {'文档记录的实际决策':24s} "
      f"{'':24s}  ⇒ 第一名 **B**（A 被写成「the next lever」）")

sub("B.2  ★ 结论：三条**数量**规则都没有复现实际决策；能复现的是一条**布尔**规则")
for label, va, vb in RULES:
    print(f"    {label:24s} ⇒ 第一名 {'A' if va > vb else 'B'}   "
          f"{'与实际相反' if va > vb else '与实际一致'}")
# 规则④：文档记录的实际判据——**要不要写新 kernel**。
# B 的修复是一个 loader flag（不写新 kernel）；A 的修复要新打包格式 + 新 kernel。
NEEDS_NEW_KERNEL = {"A": True, "B": False}
r4_first = "B" if not NEEDS_NEW_KERNEL["B"] else "A"
print(f"    {'④ 要不要写新 kernel':24s} A = {NEEDS_NEW_KERNEL['A']!s:5s}  "
      f"B = {NEEDS_NEW_KERNEL['B']!s:5s}   ⇒ 第一名 **{r4_first}**   "
      f"{'与实际一致' if r4_first == 'B' else '与实际相反'}")
check("B.2 规则④（要不要写新 kernel）复现实际决策：B 第一", r4_first == "B", r4_first)
print("    ⇒ ★ **能复现的不是公式，是一条布尔判据。** 这不是公式的失败——")
print("      它说明「工作量」在真实的排序里**不是一档名字，是有没有跨过一条线**。")
check("B.2 规则①（按占比）把 A 排在 B 之前",
      r1_a > r1_b, f"{r1_a} > {r1_b}")
check("B.2 规则②（占比÷工作量）仍把 A 排在 B 之前",
      r2_a > r2_b, f"{r2_a:.4f} > {r2_b:.4f}")
check("B.2 规则③（距离×占比÷工作量）仍把 A 排在 B 之前",
      r3_a > r3_b, f"{r3_a:.4f} > {r3_b:.4f}")
print()
print("    ⇒ 「17.9% 排在 50.4% 之前」**不是**从（占比、距离、工作量档名）算出来的。")
print("      文档记录的实际理由不是「算出来 B 更划算」，而是：")
print("      （下面两条是**文档逐字记录**的，不是本程序的推断）")
print("      · B 的修复**不需要写新 kernel**——它骑在既有的 `expand_nk` loader 策略上，"
      "`VT_GGUF_GDN_NK=0` 是同二进制的 A/B 退出开关，**构造上逐字节相同**；")
print("      · A 的修复需要一个**新的打包格式**（q8_0 repack-at-load）**加**一个新 kernel。")
print("      ⇒ 「工作量」在这条决策里**不是一档名字，是有没有跨过「要写新 kernel」这条线**。")

sub("B.3  临界工作量比值（读者可据此判断结论对定价方式有多敏感）")
be2 = sh_a / sh_b
be3 = (sh_a * d_a) / (sh_b * d_b)
print(f"    规则②下要 B 第一，需要 工作量_A / 工作量_B > 50.4/17.9 = **{be2:.4f}**")
print(f"    规则③下要 B 第一，需要 工作量_A / 工作量_B > "
      f"(50.4×{d_a:.4f})/(17.9×{d_b:.4f}) = **{be3:.4f}**")
print(f"    本库的秩映射给出 {ef_a}/{ef_b} = {ef_a/ef_b:.2f} ——"
      f"**{ef_a/ef_b:.2f} < {be3:.2f}**，所以规则③判 A 第一。")
check("B.3 临界比值严格大于本库秩映射给出的比值（这就是规则③判 A 第一的原因）",
      be3 > ef_a / ef_b, f"{be3:.4f} > {ef_a/ef_b:.2f}")
check("B.3 两个临界比值都 > 1（即 A 在两条规则下都占优）", be2 > 1 and be3 > 1,
      f"{be2:.4f}, {be3:.4f}")
print("    ★ 这条临界值是**唯一能让读者自己判断映射是否影响结论**的量——")
print("      文档里必须给出它，否则「按占比会错、按我们的公式就对」是句空话。")

sub("B.4  kernel 未覆盖的份额**不会被更快的 kernel 恢复**（这不是建议，是上界）")
print("    规则来源（**B 级**）：TRT-LLM PR #18805 ——「a launch-starved share is not"
      " recovered by a faster kernel」。")
print("    本库的算术（E 级）：设 kernel 覆盖份额为 p，把 kernel 加速到**无穷**，")
print("      剩余时间 = 1 − p，加速比上界 = 1/(1 − p)。")
for p in (0.30, 0.50, 0.80, 0.95):
    print(f"      p = {p:.2f} ⇒ 任何 kernel 优化的加速比上界 = 1/(1−{p:.2f}) = "
          f"{1/(1-p):.4f}×")
check("B.4 p 越小、上界越低（单调，且 p = 0.5 时上界恰为 2×）",
      abs(1 / (1 - 0.5) - 2.0) < 1e-12 and 1 / (1 - 0.3) < 1 / (1 - 0.8))
print("    ★ 对照实例（**B 级**）：Kimi K3 的专用 metadata builder 把 metadata 准备")
print(f"      从 {SP.KIMI_K3_METADATA_BUILDER_LATENCY_US[0]} µs 压到"
      f" {SP.KIMI_K3_METADATA_BUILDER_LATENCY_US[1]} µs（−96%），端到端只改善 **6%**。")
saved = SP.KIMI_K3_METADATA_BUILDER_LATENCY_US[0] - SP.KIMI_K3_METADATA_BUILDER_LATENCY_US[1]
iter_us = saved / 0.06                      # 反解：省下的 836 µs = 6% ⇒ 迭代 ≈ 13.9 ms
share = SP.KIMI_K3_METADATA_BUILDER_LATENCY_US[0] / iter_us
print(f"      本库反解（E 级）：省下 {saved} µs = 6% ⇒ 被改善的那段总时长 ≈ "
      f"{iter_us/1000:.1f} ms")
print(f"      该 op 的原始占比 = {SP.KIMI_K3_METADATA_BUILDER_LATENCY_US[0]}/{iter_us:.0f}"
      f" = {share:.2%}；兑现率 = 6.0%/{share:.2%} = {0.06/share:.1%}（接近 100%，"
      f"因为把它基本消除了）")
check("B.4 兑现的端到端收益不超过该 op 的占比（上界不可逾越）", 0.06 <= share,
      f"{0.06:.4f} ≤ {share:.4f}")


# ===========================================================================
head("§C  边界与反例：改变工作量估计，排序**真的翻转**")
# ===========================================================================
#
# ★ 这是「按占比排序会出错」的可复现形态。构造要点：
#   ① 占比最高的那个候选，在**两种**规则下都排第 1（所以「按占比」看起来没错）；
#   ② 只改它的**工作量估计**（别的什么都不改），它掉到第 2 名。
#   ⇒ 结论：**占比本身不足以定位第一目标**，工作量是同等权重的输入。

CANDS = [
    # (名字, 占比 %, 距上界的距离 ×, 工作量秩)
    ("op_hot",   48.0, 1.60, 1.0),
    ("op_mid",   22.0, 1.50, 1.0),
    ("op_small",  9.0, 1.30, 1.0),
    ("op_tiny",   3.2, 1.05, 1.0),
]
check("C.0 构造的候选占比合计 < 100%（它们之外的桶非零）",
      sum(c[1] for c in CANDS) < 100.0, f"{sum(c[1] for c in CANDS):.1f}%")


def rank(cands, with_distance=True):
    key = (lambda c: (c[2] if with_distance else 1.0) * c[1] / c[3])
    return sorted(cands, key=key, reverse=True)


sub("C.1  改动前：占比最高的 op_hot 在两种规则下都排第 1")
before_share = sorted(CANDS, key=lambda c: -c[1])
before_rank = rank(CANDS, with_distance=True)
print(f"    按**占比**      ：{[c[0] for c in before_share]}")
print(f"    按**排序量**    ：{[c[0] for c in before_rank]}")
check("C.1 占比排序的第一名是 op_hot", before_share[0][0] == "op_hot")
check("C.1 排序量的第一名也是 op_hot", before_rank[0][0] == "op_hot")

sub("C.2  只改 op_hot 的工作量估计（×4），排序翻转")
AFTER = [("op_hot", 48.0, 1.60, 4.0)] + CANDS[1:]
after_rank = rank(AFTER, with_distance=True)
after_share = sorted(AFTER, key=lambda c: -c[1])
print(f"    占比**没有变**  ：{[c[0] for c in after_share]}"
      f"   ⇒ 按占比仍把 op_hot 排第 1")
print(f"    排序量**变了**  ：{[c[0] for c in after_rank]}")
print(f"    ⇒ 第 1 名由 **{before_rank[0][0]}** 变成 **{after_rank[0][0]}**")
check("C.2 排序量下第 1 名**真的翻转**",
      before_rank[0][0] != after_rank[0][0], f"{before_rank[0][0]} → {after_rank[0][0]}")
check("C.2 占比没有变（翻转不是占比造成的）",
      [c[0] for c in before_share] == [c[0] for c in after_share])
print()
print("    ★ 「按占比排序会出错」这句话的**精确形态**：")
print("      占比是对**收益上界**的估计，而排序量还含**工作量**。")
print("      当工作量之比超过 §B.3 那个临界值时，两个排序必然相反。")
print("      ⚠ 反过来说：**若所有候选的工作量相同，按占比排序就是对的**——")
print("        所以「不要按占比排」这句话本身也是错的，错在把它当成无条件规则。")


# ===========================================================================
head("汇总")
# ===========================================================================
if FAILED:
    print(f"  ✗ {len(FAILED)} 条断言失败：")
    for f in FAILED:
        print(f"      · {f}")
    sys.exit(1)
print("  ✓ 全部断言通过（A 类 2 组 · B 类 4 组 · C 类 2 组）")
print()
print("  ★ 来源分级（正文必须逐处标注）：")
print(f"    · headroom gate 的规则  ：{SP.TENSORRT_LLM_TIMELINE_DECOMPOSITION['gate'][:24]}…"
      f"  [B]")
print("    · vllm.cpp 的 50.4/17.9 ：[D] · **CPU 剖面**（aarch64 DGX / 20 线程 / q8_0）")
print("    · 秩映射与 Amdahl 上界  ：[E] 本库推导")
print(f"    python {sys.version.split()[0]} / 纯标准库（未 import numpy、未 import torch）")

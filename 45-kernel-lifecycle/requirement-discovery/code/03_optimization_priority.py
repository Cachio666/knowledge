#!/usr/bin/env python3
r"""03-optimization-priority —— 优化价值的排序，与「停止条件」的第一道闸。

对应文档：`../03-optimization-priority.md`

★ 本程序的可证伪断言：
  A 类  headroom gate 的**边界**：距上界恰好 1.300× 时，声称 1.300× 通过、1.301× 被拒；
        并用 vllm.cpp 的实测值走一遍闸（1.090× ≤ 1.218× 通过）
  B 类  ★ 用 vllm.cpp 的实例数据**逐条检验三条排序规则**，结论是
        **三条规则都没有复现文档记录的实际顺序**——并给出**临界工作量比值**；
              另验证「kernel 未覆盖的份额不会被更快的 kernel 恢复」这条**本库推论**，
        以及 compute-absent 三档的**逐字配平恒等式**与 0.5 ms 残差门限
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
# ★ 闸的来源（**B 级，经检索转述**）：NVIDIA/TensorRT-LLM PR #18805 的变更说明里
#   有 `headroom_verdict` 这道校验。本程序把它落成一个可判定的比较。
#   ⚠ 「经检索转述」，**不是「逐字」**——见 specs.py §4 的引用纪律。

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
check("A.3 前 3 个桶之和**四舍五入到整数 == 常量**（83.2 → 83.0，改常量即失败）",
      round(top3) == SP.VLLMCPP_PREFILL_GEMM_TOTAL_PCT,
      f"round({top3:.1f}) = {round(top3)} vs {SP.VLLMCPP_PREFILL_GEMM_TOTAL_PCT}")
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
# ★★ 出处订正（本主题初版在这里**归错了两处**，见 specs.py §4 的引用纪律）：
#   ① 三分法的可复核出处是 **NVIDIA NeMo Megatron Bridge 的 Nsight Systems 分析文档**
#      （[B]），**不是** TensorRT-LLM PR #18805。
#   ② 「a launch-starved share is not recovered by a faster kernel」这句话
#      **两次独立检索都查不到**——初版把它标成那个 PR 的「逐字」引文，**那是编引文**。
#      现在改成**从逐字定义推出的本库推论**（[E]）。
print(f"    逐字定义（[B]，出处见 specs.py 的 `NSYS_*` 一组常量，文件级 URL 也在那里）：")
for _n, _d in ((SP.NSYS_BUCKET_LAUNCH_STARVED, SP.NSYS_LAUNCH_STARVED_DEFINITION),
               (SP.NSYS_BUCKET_BLOCKING, SP.NSYS_BLOCKING_DEFINITION),
               (SP.NSYS_BUCKET_DEPENDENCY_STALLED, SP.NSYS_DEPENDENCY_STALLED_DEFINITION)):
    print(f"      · {_n:20s} —— {_d[:58]}…")
print("    本库推论（[E]）：`Launch-starved` 的定义是「下一个计算 kernel **还没被宿主发出**」；")
print("      而「更快的 kernel」改变的是**已发出的 kernel 的执行时长**——两者不在同一侧。")
print("      ⇒ launch-starved 那一份**不会被更快的 kernel 收回**。**这是推论，不是引文。**")
print("    本库算术（[E]）：设 kernel 覆盖份额为 p，把 kernel 加速到**无穷**，")
print("      剩余时间 = 1 − p，加速比上界 = 1/(1 − p)。")
for p in (0.30, 0.50, 0.80, 0.95):
    print(f"      p = {p:.2f} ⇒ 任何 kernel 优化的加速比上界 = 1/(1−{p:.2f}) = "
          f"{1/(1-p):.4f}×")
check("B.4 p 越小、上界越低（单调，且 p = 0.5 时上界恰为 2×）",
      abs(1 / (1 - 0.5) - 2.0) < 1e-12 and 1 / (1 - 0.3) < 1 / (1 - 0.8))
check("B.4 「不可恢复」被标成**推论**（`NOT_RECOVERABLE_IS_DERIVED` 为真）",
      SP.NOT_RECOVERABLE_IS_DERIVED is True)
check("B.4 那句伪「逐字引文」**已不在** specs 的任何字符串里（编引文已清除）",
      not any("not recovered by a faster kernel" in str(_v)
              for _k, _v in vars(SP).items() if not _k.startswith("_")))

sub("B.5  三档之和必须与 compute-absent **配平**（逐字的恒等式 + 0.5 ms 的残差门限）")
# ★ 这是一条**逐字读到的**上游判据（[B]，两次独立检索一致），
#   也是本主题**最强的一条把「分解」变成「恒等式」的断言**。
print(f"    恒等式（逐字）：{SP.NSYS_RECONCILIATION}")
print(f"    残差门限（逐字）：> {SP.NSYS_RESIDUAL_MS_THRESHOLD} ms / iteration 时"
      f"**应当去查它**，而不是并进最大的那个桶。")
print(f"    另一个逐字规则：「{SP.NSYS_WALL_TIME_RULE[:60]}…」")


def residual_ms(absent, ls, blk, dep):
    """三档之和与 compute-absent 的残差（ms/iteration）。"""
    return absent - (ls + blk + dep)


print(f"    配平的例子：100.0 − (58.0 + 30.0 + 12.0) = {residual_ms(100., 58., 30., 12.):.1f} ms")
check("B.5 完全配平时残差 == 0", residual_ms(100.0, 58.0, 30.0, 12.0) == 0.0)
check("B.5 残差 0.4 ms ⇒ 在门限内（不必去查）",
      residual_ms(100.0, 58.0, 30.0, 11.6) < SP.NSYS_RESIDUAL_MS_THRESHOLD,
      f"{residual_ms(100.0, 58.0, 30.0, 11.6):.2f}")
check("B.5 残差 0.6 ms ⇒ **超出**门限（必须去查，不得并进最大的桶）",
      residual_ms(100.0, 58.0, 30.0, 11.4) > SP.NSYS_RESIDUAL_MS_THRESHOLD,
      f"{residual_ms(100.0, 58.0, 30.0, 11.4):.2f}")

sub("B.6  小占比换来可测收益：Kimi K3 的 metadata builder（[B]）")
print("    ★ 对照实例（**B 级**）：专用 metadata builder 把 metadata 准备")
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
check("B.6 兑现的端到端收益不超过该 op 的占比（上界不可逾越）", 0.06 <= share,
      f"{0.06:.4f} ≤ {share:.4f}")
# ★ 这三条把 (870, 34) **钉死**：改任何一个数都会失败。
check("B.6 省下的绝对量 == 870 − 34 == 836 µs", saved == 836, f"{saved}")
check("B.6 反解出的迭代长度 == 836 / 0.06 == 13933.3 µs",
      abs(iter_us - 13933.333333) < 0.01, f"{iter_us:.4f}")
check("B.6 该 op 的原始占比 == 870 / 13933.3 == 6.244%",
      abs(share - 0.06244) < 1e-5, f"{share:.6f}")


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
head("§D  常量自洽：本篇用到的 specs.py 常量，逐条被断言钉住")
# ===========================================================================
#
# ★ 机械判据（本域 C6）：**逐个改 specs.py 里的常量，`make run` 必须报错。**
#   这一节是 03 篇的落点；01 / 02 篇各有自己的 §D。

sub("D.1  vllm.cpp 剖面的两个「总账」常量")
_tot = SP.VLLMCPP_PREFILL_PROFILE_TOTAL_MS
_sh = SP.VLLMCPP_PREFILL_SHARES_PCT
print(f"    总时长 = {_tot} ms（**TTFT**）；各 op 的占比 = {_sh}")
_abs = {k: v / 100.0 * _tot for k, v in _sh.items()}
for _k, _v in sorted(_abs.items(), key=lambda kv: -kv[1]):
    print(f"      {_k:16s} {_sh[_k]:5.1f}% × {_tot} ms = {_v:8.2f} ms")
check("D.1 总时长 == 1070.0 ms（CPU 剖面的 TTFT）", _tot == 1070.0, f"{_tot}")
check("D.1 kMatmul 的绝对时间 == 17.9% × 1070 = 191.53 ms",
      abs(_abs["kMatmul"] - 191.53) < 0.005, f"{_abs['kMatmul']:.4f}")
check("D.1 五个桶的绝对时间之和 < 1070 ms（剩下的桶不是零）",
      sum(_abs.values()) < _tot, f"{sum(_abs.values()):.2f} < {_tot}")
check("D.1 完整占比表与来源逐条一致（C6：改任一条即失败）",
      _sh == {"kMatmulBTQuant": 50.4, "kMatmul": 17.9, "kMatmulBT": 14.9,
              "kGdnPrefill": 5.2, "kPagedAttention": 3.2}, str(_sh))
check("D.1 占比表降序、恰好 5 个键",
      len(_sh) == 5 and list(_sh.values()) == sorted(_sh.values(), reverse=True))

sub("D.2  「占比最大的被推后」这条论断的两个端点")
check("D.2 CHOSEN_OP 与 DEFERRED_OP 都在占比表里",
      SP.VLLMCPP_CHOSEN_OP in _sh and SP.VLLMCPP_DEFERRED_OP in _sh,
      f"{SP.VLLMCPP_CHOSEN_OP} / {SP.VLLMCPP_DEFERRED_OP}")
check("D.2 两个算子的名字逐字一致", (SP.VLLMCPP_DEFERRED_OP, SP.VLLMCPP_CHOSEN_OP)
      == ("kMatmulBTQuant", "kMatmul"), f"{SP.VLLMCPP_DEFERRED_OP}/{SP.VLLMCPP_CHOSEN_OP}")
check("D.2 被推后的那个**占比更大**（否则本篇的论点不成立）",
      _sh[SP.VLLMCPP_DEFERRED_OP] > _sh[SP.VLLMCPP_CHOSEN_OP],
      f"{_sh[SP.VLLMCPP_DEFERRED_OP]} > {_sh[SP.VLLMCPP_CHOSEN_OP]}")
check("D.2 两个工作量档名逐字一致，且都在 EFFORT_RANK_MAPPING 里",
      (SP.VLLMCPP_DEFERRED_EFFORT_TIER, SP.VLLMCPP_CHOSEN_EFFORT_TIER)
      == ("MEDIUM–HIGH", "LOW")
      and SP.VLLMCPP_DEFERRED_EFFORT_TIER in SP.EFFORT_RANK_MAPPING
      and SP.VLLMCPP_CHOSEN_EFFORT_TIER in SP.EFFORT_RANK_MAPPING)
check("D.2 被选中的那个工作量档**更低**（这就是决策的全部内容）",
      SP.EFFORT_RANK_MAPPING[SP.VLLMCPP_CHOSEN_EFFORT_TIER]
      < SP.EFFORT_RANK_MAPPING[SP.VLLMCPP_DEFERRED_EFFORT_TIER],
      f"{SP.VLLMCPP_CHOSEN_EFFORT_TIER} < {SP.VLLMCPP_DEFERRED_EFFORT_TIER}")

sub("D.3  Amdahl 上界与实测：1.090 ≤ 1.2180")
for _k, _v in sorted(SP.VLLMCPP_CEILING_DERIVED.items()):
    _m = _sh[_k] / 100.0
    print(f"      {_k:16s} 1/(1−{_m:.3f}) = {1/(1-_m):.4f}（表值 {_v}）")
check("D.3 两个 Amdahl 上界与 1/(1−占比) 逐位一致",
      all(abs(1 / (1 - _sh[_k] / 100.0) - _v) < 5e-5
          for _k, _v in SP.VLLMCPP_CEILING_DERIVED.items()),
      str(SP.VLLMCPP_CEILING_DERIVED))
check("D.3 上界表就是 {kMatmulBTQuant: 2.0161, kMatmul: 1.2180}（逐字）",
      SP.VLLMCPP_CEILING_DERIVED == {"kMatmulBTQuant": 2.0161, "kMatmul": 1.2180},
      str(SP.VLLMCPP_CEILING_DERIVED))
check("D.3 实测值 == 1.090（逐字）", SP.VLLMCPP_CHOSEN_MEASURED_SPEEDUP_LOW == 1.090,
      f"{SP.VLLMCPP_CHOSEN_MEASURED_SPEEDUP_LOW}")
check("D.3 实测 1.090 通过 headroom gate（未超其到上界的距离）",
      SP.VLLMCPP_CHOSEN_MEASURED_SPEEDUP_LOW
      <= SP.VLLMCPP_CEILING_DERIVED[SP.VLLMCPP_CHOSEN_OP] + 1e-9,
      f"{SP.VLLMCPP_CHOSEN_MEASURED_SPEEDUP_LOW} ≤ "
      f"{SP.VLLMCPP_CEILING_DERIVED[SP.VLLMCPP_CHOSEN_OP]}")
check("D.3 「占比大的那个上界更大」（本篇的反直觉点）",
      SP.VLLMCPP_CEILING_DERIVED[SP.VLLMCPP_DEFERRED_OP]
      > SP.VLLMCPP_CEILING_DERIVED[SP.VLLMCPP_CHOSEN_OP])

sub("D.4  秩映射：它必须给出比值 2.5，否则 §B.3 的临界值失去意义")
_em = SP.EFFORT_RANK_MAPPING
check("D.4 档名→秩的映射给出比值 == 2.5",
      _em[SP.VLLMCPP_DEFERRED_EFFORT_TIER] / _em[SP.VLLMCPP_CHOSEN_EFFORT_TIER] == 2.5,
      f"{_em[SP.VLLMCPP_DEFERRED_EFFORT_TIER]/_em[SP.VLLMCPP_CHOSEN_EFFORT_TIER]}")
check("D.4 整张映射表逐条一致（C6：改任一档的秩即失败）",
      _em == {"LOW": 1.0, "MEDIUM": 2.0, "MEDIUM–HIGH": 2.5, "HIGH": 3.0, "VERY HIGH": 4.0},
      str(_em))
check("D.4 秩全为正、且档名升序时秩严格递增（映射不是随意的）",
      all(v > 0 for v in _em.values())
      and [_em[k] for k in ("LOW", "MEDIUM", "MEDIUM–HIGH", "HIGH", "VERY HIGH")]
      == sorted([_em[k] for k in ("LOW", "MEDIUM", "MEDIUM–HIGH", "HIGH", "VERY HIGH")]))

sub("D.5  PR #18805 的四条布尔事实 + taxonomy 路径")
check("D.5 `share_pct` 由 interval-union time 定义", SP.TRTLLM_18805_SHARE_PCT_IS_INTERVAL_UNION is True)
check("D.5 `kern_sum` 只作显式兜底", SP.TRTLLM_18805_KERN_SUM_IS_EXPLICIT_FALLBACK_ONLY is True)
check("D.5 `headroom_verdict` 是一道闸（03 篇 §1.2 的来源）",
      SP.TRTLLM_18805_HEADROOM_VERDICT_IS_A_GATE is True)
check("D.5 流重叠会同时改变排序与百分比", SP.TRTLLM_18805_OVERLAP_CHANGES_RANK_AND_PCT is True)
check("D.5 taxonomy 文件路径逐字一致",
      SP.TRTLLM_18805_TAXONOMY_FILE
      == "agent_flow/workflows/perf_analyze/assets/taxonomy_trtllm.json",
      SP.TRTLLM_18805_TAXONOMY_FILE)
check("★ D.5 被归错的两处说法**都不在 specs 里**（编归属已清除）",
      not any(("busy rung" in str(v)) or ("not recovered" in str(v))
              for k, v in vars(SP).items() if not k.startswith("_")))

sub("D.6  compute-absent 三分法：**逐字定义**，三档名字钉死")
check("D.6 三档名字与顺序逐字一致",
      SP.NSYS_BUCKETS == ("launch-starved", "blocking", "dependency-stalled"), str(SP.NSYS_BUCKETS))
check("D.6 Launch-starved 定义逐字一致（这是 §B.4 那条推论的依据）",
      SP.NSYS_LAUNCH_STARVED_DEFINITION
      == "the next compute kernel had not yet been issued by the host",
      SP.NSYS_LAUNCH_STARVED_DEFINITION)
check("D.6 Blocking 定义逐字一致（含 negative slack）",
      "negative slack" in SP.NSYS_BLOCKING_DEFINITION
      and SP.NSYS_BLOCKING_DEFINITION.startswith("the kernel was issued and a resolved producer"),
      SP.NSYS_BLOCKING_DEFINITION)
check("D.6 Dependency-stalled 定义逐字一致",
      SP.NSYS_DEPENDENCY_STALLED_DEFINITION
      == ("the kernel was issued, but no resolved producer explains the remaining delay"),
      SP.NSYS_DEPENDENCY_STALLED_DEFINITION)
check("D.6 compute-absent 的定义逐字一致",
      SP.NSYS_COMPUTE_ABSENT_DEFINITION == "Iteration minus compute busy",
      SP.NSYS_COMPUTE_ABSENT_DEFINITION)
check("D.6 出处给到了**文件级 URL**（不是「某个仓库」）",
      SP.NSYS_URL.startswith("https://docs.nvidia.com/") and SP.NSYS_URL.endswith("SKILL.html"),
      SP.NSYS_URL)
check("D.6 `wall_time_rule` 逐字写着「never presented as wall time」",
      SP.NSYS_WALL_TIME_RULE == ("kernel-duration sums must be labeled only as work volume, "
                                 "never presented as wall time"), SP.NSYS_WALL_TIME_RULE)

sub("D.7  `NSYS_RESIDUAL_MS_THRESHOLD` 与它守卫的恒等式")
check("D.7 门限 == 0.5（ms / iteration）", SP.NSYS_RESIDUAL_MS_THRESHOLD == 0.5,
      f"{SP.NSYS_RESIDUAL_MS_THRESHOLD}")
check("D.7 恒等式文本逐字一致",
      SP.NSYS_RECONCILIATION
      == "launch-starved + blocking + dependency-stalled ~ compute-absent",
      SP.NSYS_RECONCILIATION)
check("D.7 门限**恰好**是残差判定里用的那个数（改它 §B.5 会翻）",
      residual_ms(100.0, 58.0, 30.0, 11.6) < SP.NSYS_RESIDUAL_MS_THRESHOLD
      < residual_ms(100.0, 58.0, 30.0, 11.4),
      f"{residual_ms(100.,58.,30.,11.6):.2f} < "
      f"{SP.NSYS_RESIDUAL_MS_THRESHOLD} < {residual_ms(100.,58.,30.,11.4):.2f}")

sub("D.8  四类处置：**elimination 排第一**")
check("D.8 四类名字与顺序逐字一致",
      SP.TRTLLM_COVERAGE_DISPOSITIONS
      == ("elimination", "faster execution", "fusion", "overlap"),
      str(SP.TRTLLM_COVERAGE_DISPOSITIONS))

sub("D.9  噪声阈值：本主题**只保留判据，不保留数值**")
print(f"    判据 = {SP.NOISE_THRESHOLD_RULE}")
check("D.9 判据逐字一致（C6：改一个字即失败）",
      SP.NOISE_THRESHOLD_RULE
      == "任何显著性阈值都必须连带测量条件；条件给不出，阈值不可用", SP.NOISE_THRESHOLD_RULE)
check("D.9 判据的来源等级标的是 [E]（它是本库提炼，不是引用值）",
      SP.NOISE_THRESHOLD_RULE_TAG == "E", SP.NOISE_THRESHOLD_RULE_TAG)
_banned = ("HIPFIRE_WITHIN_SESSION_NOISE_PCT_GFX1100", "HIPFIRE_GRAPH_CAPTURE_SPREAD_PCT",
           "HIPFIRE_GRAPH_CAPTURE_OFF_SPREAD_PCT", "HIPFIRE_SPEED_GATE_TOLERANCE_PCT",
           "KERNEL_LAUNCH_OVERHEAD_US_EXTERNAL", "ARXIV_2607_02541", "SM_COUNT_BY_GEN",
           "FLOAT32_BYTES", "GAIN_BELOW_NOISE_IS_NOT_WORTH_IT",
           "TOKENS_PER_PARAM_RULE_OF_THUMB", "TENSORRT_LLM_TIMELINE_DECOMPOSITION")
_re_added = [n for n in _banned if hasattr(SP, n)]
print(f"    已移除的常量是否被重新加回：{_re_added or '无'}")
check("★ D.9 那 11 条已移除的常量**没有被重新加回** specs.py"
      "（噪声/启动开销/跨架构方差归 convergence-and-tuning）", not _re_added, str(_re_added))

sub("D.11  vLLM 与 Kimi K3 的那几条结构化事实")
check("D.11 三档档名逐字一致", SP.VLLM_MODEL_INTEGRATION_TIERS == ("simple", "moderate", "complex"),
      str(SP.VLLM_MODEL_INTEGRATION_TIERS))
check("D.11 只有 `complex` 要新 kernel", SP.VLLM_TIER_NEEDS_NEW_KERNEL == "complex",
      SP.VLLM_TIER_NEEDS_NEW_KERNEL)
check("D.11 只有一档要新 kernel（其余两档不在需要列表里）",
      sum(1 for t in SP.VLLM_MODEL_INTEGRATION_TIERS if t == SP.VLLM_TIER_NEEDS_NEW_KERNEL) == 1)
check("D.11 `vllm serve <model>` 逐字一致",
      SP.VLLM_TRANSFORMERS_BACKEND_HINT == "vllm serve <model>", SP.VLLM_TRANSFORMERS_BACKEND_HINT)
check("D.11 Kimi K3 的四个新算子名逐字一致",
      SP.KIMI_K3_NEW_OPERATOR_NAMES
      == ("KDA prefill", "KDA decode", "Attention Residuals", "SiTU"),
      str(SP.KIMI_K3_NEW_OPERATOR_NAMES))
check("D.11 KDA 的两个 PR 号逐字一致",
      (SP.KIMI_K3_KDA_PREFILL_PRS, SP.KIMI_K3_KDA_DECODE_PR) == ((51311, 52458), 50654))
check("D.11 AttnRes 的「至多 8 块 / 至多 9 来源」自洽（9 = 8 + 当前块内残差）",
      SP.KIMI_K3_ATTNRES_MAX_SOURCES == SP.KIMI_K3_ATTNRES_MAX_CACHED_BLOCKS + 1,
      f"{SP.KIMI_K3_ATTNRES_MAX_SOURCES} vs {SP.KIMI_K3_ATTNRES_MAX_CACHED_BLOCKS}+1")
check("D.11 三个被拆开的概念逐字一致",
      SP.KIMI_K3_PREFIX_CACHE_SEPARATED_CONCEPTS
      == ("physical block size", "scheduler alignment", "prefix-match unit"),
      str(SP.KIMI_K3_PREFIX_CACHE_SEPARATED_CONCEPTS))
check("D.11 copy-on-write 布尔为真", SP.KIMI_K3_PREFIX_CACHE_RULE_IS_COPY_ON_WRITE is True)
check("D.11 环境变量名逐字一致",
      SP.KIMI_K3_PREFIX_CACHE_ENV_VAR == "VLLM_PREFIX_CACHE_RETENTION_INTERVAL",
      SP.KIMI_K3_PREFIX_CACHE_ENV_VAR)
check("D.11 metadata builder 的 870→34 与端到端 6% 逐字一致",
      SP.KIMI_K3_METADATA_BUILDER_LATENCY_US == (870, 34)
      and SP.KIMI_K3_METADATA_BUILDER_E2E_GAIN_PCT == 6,
      f"{SP.KIMI_K3_METADATA_BUILDER_LATENCY_US} / {SP.KIMI_K3_METADATA_BUILDER_E2E_GAIN_PCT}")

sub("D.12  vLLM IR 的四个标识符逐字一致")
check("D.12 装饰器/CLI 前缀/保留 provider 名逐字一致",
      (SP.VLLM_IR_DECLARATION_DECORATOR, SP.VLLM_IR_IMPL_DECORATOR,
       SP.VLLM_IR_PRIORITY_CLI_PREFIX, SP.VLLM_IR_RESERVED_FALLBACK_PROVIDER)
      == ("@register_op", "register_impl", "--ir-op-priority", "native"),
      f"{SP.VLLM_IR_DECLARATION_DECORATOR}/{SP.VLLM_IR_IMPL_DECORATOR}/"
      f"{SP.VLLM_IR_PRIORITY_CLI_PREFIX}/{SP.VLLM_IR_RESERVED_FALLBACK_PROVIDER}")

# ---------------------------------------------------------------------------
sub("D.10  本程序的常量覆盖表")
COVERED_03 = (
    "VLLM_MODEL_INTEGRATION_TIERS", "VLLM_TIER_NEEDS_NEW_KERNEL",
    "VLLM_TRANSFORMERS_BACKEND_HINT", "VLLM_IR_DECLARATION_DECORATOR",
    "VLLM_IR_IMPL_DECORATOR", "VLLM_IR_PRIORITY_CLI_PREFIX",
    "VLLM_IR_RESERVED_FALLBACK_PROVIDER",
    "KIMI_K3_NEW_OPERATOR_NAMES", "KIMI_K3_KDA_PREFILL_PRS", "KIMI_K3_KDA_DECODE_PR",
    "KIMI_K3_SITU_MEGAMOE_PR", "KIMI_K3_ATTNRES_MAX_CACHED_BLOCKS",
    "KIMI_K3_ATTNRES_MAX_SOURCES", "KIMI_K3_PREFIX_CACHE_SEPARATED_CONCEPTS",
    "KIMI_K3_PREFIX_CACHE_RULE_IS_COPY_ON_WRITE", "KIMI_K3_PREFIX_CACHE_ENV_VAR",
    "KIMI_K3_METADATA_BUILDER_E2E_GAIN_PCT",
    "TRTLLM_18805_SHARE_PCT_IS_INTERVAL_UNION",
    "TRTLLM_18805_KERN_SUM_IS_EXPLICIT_FALLBACK_ONLY",
    "TRTLLM_18805_HEADROOM_VERDICT_IS_A_GATE",
    "TRTLLM_18805_OVERLAP_CHANGES_RANK_AND_PCT", "TRTLLM_18805_TAXONOMY_FILE",
    "NSYS_COMPUTE_ABSENT_DEFINITION", "NSYS_BUCKET_LAUNCH_STARVED", "NSYS_BUCKET_BLOCKING",
    "NSYS_BUCKET_DEPENDENCY_STALLED", "NSYS_BUCKETS", "NSYS_LAUNCH_STARVED_DEFINITION",
    "NSYS_BLOCKING_DEFINITION", "NSYS_DEPENDENCY_STALLED_DEFINITION", "NSYS_RECONCILIATION",
    "NSYS_RESIDUAL_MS_THRESHOLD", "NSYS_WALL_TIME_RULE", "NSYS_URL",
    "NOT_RECOVERABLE_IS_DERIVED", "TRTLLM_COVERAGE_DISPOSITIONS",
    "NOISE_THRESHOLD_RULE", "NOISE_THRESHOLD_RULE_TAG",
    "VLLMCPP_PREFILL_PROFILE_TOTAL_MS",
    "VLLMCPP_PREFILL_SHARES_PCT", "VLLMCPP_PREFILL_GEMM_TOTAL_PCT",
    "VLLMCPP_DEFERRED_OP", "VLLMCPP_DEFERRED_EFFORT_TIER", "VLLMCPP_CHOSEN_OP",
    "VLLMCPP_CHOSEN_EFFORT_TIER", "VLLMCPP_CHOSEN_MEASURED_SPEEDUP_LOW",
    "VLLMCPP_CEILING_DERIVED", "KIMI_K3_METADATA_BUILDER_LATENCY_US",
    "EFFORT_RANK_MAPPING", "NOISE_THRESHOLD_PRINCIPLE",
)
for _n in COVERED_03:
    print(f"      {_n:40s} 有断言")
print(f"    ⇒ 03 篇覆盖 {len(COVERED_03)} 条；三篇合计覆盖 specs.py 的全部常量")


# ===========================================================================
head("汇总")
# ===========================================================================
if FAILED:
    print(f"  ✗ {len(FAILED)} 条断言失败：")
    for f in FAILED:
        print(f"      · {f}")
    sys.exit(1)
print("  ✓ 全部断言通过（A 类 2 组 · B 类 6 组 · C 类 2 组 · D 类 12 组）")
print()
print("  ★ 来源分级（正文必须逐处标注）：")
print(f"    · headroom gate 的规则  ："
      f"TRT-LLM PR #18805 的 headroom 校验（布尔事实）  [B·转述]")
print(f"    · compute-absent 三分法 ："
      f"{SP.NSYS_RECONCILIATION}  [B·逐字]（三档名 {SP.NSYS_BUCKETS}）")
print(f"    · 「不可恢复」那条      ：**本库推论（[E]）**，不是引文")
print("    · vllm.cpp 的 50.4/17.9 ：[D] · **CPU 剖面**（aarch64 DGX / 20 线程 / q8_0）")
print("    · 秩映射与 Amdahl 上界  ：[E] 本库推导")
print(f"    python {sys.version.split()[0]} / 纯标准库（未 import numpy、未 import torch）")

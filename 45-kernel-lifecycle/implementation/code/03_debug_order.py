#!/usr/bin/env python3
r"""03-debug-order —— 调试顺序的期望成本、它的**边界**、以及预算耗尽长什么样。

对应文档：`../03-debug-order.md`

★ 本程序的可证伪断言：
  A 类  两个模型（**步数模型**与**成本模型**）的期望值都与**解析解**一致，
        且解析解与**穷举 24 个排列**得到的 argmin 相同。
        ★ 把 `DEBUG_LAYER_SUBHYPOTHESES` 里任何一个成本改掉，A 类立刻给出不同的数。
  B 类  ★★ 本程序的核心诚实性断言：**依赖顺序在成本模型下不是最优的**。
        并给出**确切的门限**：把 `compute` 的成本从 6 降到 **3.38 以下**，
        依赖顺序才重新变成成本最优。⇒ 正文不得写成「按依赖顺序检查总是最好」。
  C 类  ★★ 缺陷位于 `layout` 层时：按「compute 优先」的直觉顺序在**预算内无法定位**
        （0 / N 次成功），而按依赖顺序 **N / N 定位**（2 步）。
        第二个 C 类：**非确定性的实现必须被拒**——同一份输入、200 个不同的累加顺序，
        `atomicAdd` 形态的实现给出多个不同结果，顺序累加只给出一个。
  D 类  `specs.py` §3 的每条常量逐条被断言钉住（01 号程序做跨程序并集检查）。

跑法：`make 03_debug_order`。纯 python3 + numpy，不 import torch，不碰 GPU。

★ 口径声明（必须与文档一致，否则本程序的结论不可比）：
  · 分层模型、各层的**先验概率**与**子假设个数**都是**本库的建模输入 [E]**——
    **没有来源，也测不了**（见 `specs.UNSOURCED` 第 5 项）。程序断言**结论随它们变化**，
    而不是把它们当成事实。
  · compute-sanitizer 的四档来自 NVIDIA 官方文档 **[A]**；PyTorch 的确定性与
    `atomicAdd` 那几条来自 PyTorch 官方 reproducibility 文档 **[B]**。
  · **本程序不含任何 kernel 实现代码**（域 README 的硬边界）。
"""

import random
import sys
from itertools import permutations

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
# 0. 模型：四层、每层的检查成本、每个层的先验概率
# ===========================================================================
#
# ★ 分层的**依据是依赖**（`DEBUG_LAYER_ORDER_IS_DEPENDENCY_ORDER`）：
#       layout → copy → compute → edges
#   后面的层依赖前面的层**正确**。这不是「重要性排序」，是可以被检验的结构：
#   把上一层的输出换成手算的正确值，若缺陷消失，则缺陷在上一层。

LAYERS = SP.DEBUG_LAYERS
PRIOR = SP.DEBUG_LAYER_PRIORS
COST = SP.DEBUG_LAYER_SUBHYPOTHESES
BUDGET = SP.DEBUG_CHECK_BUDGET_STEPS

# ★★ **先断言模型自洽，再使用它。** 否则 `DEBUG_LAYERS` 里改一个字会让下面的
#    `PRIOR[layer]` 抛 `KeyError` —— 那是 **harness 异常**（假捕获），
#    而不是「断言失败」。本主题的变异测试第一版正是在这里得到了一条 harness 异常：
#    崩溃**不是**捕获，它只证明「有地方不对」（01 篇 §1.5 的三态）。
if set(LAYERS) != set(PRIOR) or set(LAYERS) != set(COST) or len(set(LAYERS)) != len(LAYERS):
    print("[FAIL] 0 ★★ 模型自洽：层的集合 == 先验的键 == 成本的键，且层不重复")
    print(f"       LAYERS = {LAYERS}")
    print(f"       PRIOR  = {sorted(PRIOR)}")
    print(f"       COST   = {sorted(COST)}")
    print("  ✗ 1 条断言失败：模型自洽（后续分析全部依赖它，**不继续**）")
    sys.exit(1)
FIXED_ORDER = LAYERS                      # 依赖顺序（本程序要检验的那个「固定顺序」）
# ★ 「直觉顺序」= 先怀疑数学。重复三次是为了建模「同一层反复怀疑」这个真实行为
#   （文档 §4 坑 ①）：不是把 compute 查一遍就死心，而是回头再查。
INTUITIVE_ORDER = ("compute",) * 3 + ("edges", "copy", "layout")


def E_steps(order, prior=None):
    """**步数模型**：每检查一层算 1 步，期望步数 = Σ 先验 × 该层在顺序里的位次。"""
    prior = PRIOR if prior is None else prior
    return sum(prior[layer] * (i + 1) for i, layer in enumerate(order))


def E_cost(order, prior=None, cost=None):
    """**成本模型**：排除第 `i` 层要花 `cost[i]` 步，期望成本 = Σ 先验 × 累计成本。

    ★ 两个模型的差别只在「一层的检查成本是不是常数」——
      但这个差别会导致**不同的最优顺序**（这就是 §B 的全部内容）。
    """
    prior = PRIOR if prior is None else prior
    cost = COST if cost is None else cost
    total, run = 0.0, 0
    for layer in order:
        run += cost[layer]
        total += prior[layer] * run
    return total


def brute_force(best_of, **kw):
    """穷举全部 24 个排列，返回 (argmin, 最小值) —— 这是解析解的**对照**。"""
    scored = sorted(((best_of(p, **kw), p) for p in permutations(LAYERS)))
    return scored[0][1], scored[0][0]


head("§0  模型：四层与它们的成本 / 先验")
print(f"  层（按依赖顺序）：{LAYERS}")
print(f"  {'层':>9} | {'排除它要花的步数':>16} | {'先验概率':>9}")
for l in LAYERS:
    print(f"  {l:>9} | {COST[l]:>16} | {PRIOR[l]:>9.2f}")
print(f"  检查预算 = {BUDGET} 步；一层跑完一遍的总成本 = {sum(COST.values())} 步")
print(f"  固定顺序（依赖顺序）   = {FIXED_ORDER}")
print(f"  直觉顺序（compute 优先）= {INTUITIVE_ORDER}")
print("  ⚠ 成本与先验**都是本库的建模输入 [E]**，没有来源（specs.UNSOURCED 第 5 项）；")
print("    程序断言的是**结论随它们变化**，不是它们本身。")
check("0 先验概率之和为 1（否则期望值不是加权平均）",
      abs(sum(PRIOR.values()) - 1.0) < 1e-12, f"{sum(PRIOR.values()):.6f}")
check("0 四层的成本与先验都有定义（缺一个 KeyError 会立刻暴露）",
      set(LAYERS) == set(COST) == set(PRIOR))
check("0 ★ 成本与先验**不是**同一个字典（混淆它们会让两个模型等价，§B 就没了）",
      COST is not PRIOR)


# ===========================================================================
# A. 两种模型的期望值与解析解一致
# ===========================================================================
head("§A  期望值与解析解、穷举解三者一致")

sub("A.1  解析解：两个模型的最优顺序分别是什么")
by_prior_desc = tuple(sorted(LAYERS, key=lambda l: -PRIOR[l]))
smith = tuple(sorted(LAYERS, key=lambda l: COST[l] / PRIOR[l]))
print("    **步数模型**的解析最优顺序 = 按先验**降序**排列")
print(f"        {by_prior_desc}"
      f"   （先验 = {[PRIOR[l] for l in by_prior_desc]}）")
print("    **成本模型**的解析最优顺序 = 按 **成本/先验 升序**排列（史密斯规则）")
print(f"        {smith}")
print(f"        成本/先验 = {[round(COST[l] / PRIOR[l], 3) for l in smith]}")

sub("A.2  穷举 24 个排列，验证解析解就是 argmin")
bf_steps, best_steps = brute_force(E_steps)
bf_cost, best_cost = brute_force(E_cost)
print(f"    步数模型：穷举最优 = {bf_steps}（{best_steps:.6f}）")
print(f"              解析最优 = {by_prior_desc}（{E_steps(by_prior_desc):.6f}）")
print(f"    成本模型：穷举最优 = {bf_cost}（{best_cost:.6f}）")
print(f"              解析最优 = {smith}（{E_cost(smith):.6f}）")
check("A.2 ★ 步数模型：解析解 == 穷举解", by_prior_desc == bf_steps,
      f"{by_prior_desc} vs {bf_steps}")
check("A.2 ★ 成本模型：解析解 == 穷举解（史密斯规则）", smith == bf_cost,
      f"{smith} vs {bf_cost}")
check("A.2 ★ 两个模型的最优顺序**不相等**（否则 §B 的边界不存在）",
      by_prior_desc != smith)

sub("A.3  固定顺序（依赖顺序）在两种模型下的期望值")
e_steps_fixed = E_steps(FIXED_ORDER)
e_cost_fixed = E_cost(FIXED_ORDER)
print(f"    固定顺序的期望步数 = {e_steps_fixed:.6f}"
      f"（手算：{' + '.join(f'{PRIOR[l]:.2f}×{i + 1}' for i, l in enumerate(FIXED_ORDER))}）")
print(f"    固定顺序的期望成本 = {e_cost_fixed:.6f}")
check("A.3 ★ 固定顺序的期望步数 == 手算值 2.08", abs(e_steps_fixed - 2.08) < 1e-12,
      f"{e_steps_fixed:.6f}")
check("A.3 ★ 固定顺序的期望成本 == 手算值 6.16", abs(e_cost_fixed - 6.16) < 1e-12,
      f"{e_cost_fixed:.6f}")

sub("A.4  随机顺序下的分布（多次模拟）")
rng = random.Random(20260920)
N_SIM = 20000
sims = []
for _ in range(N_SIM):
    p = list(LAYERS)
    rng.shuffle(p)
    sims.append((E_steps(p), E_cost(p)))
steps_arr = np.array([s for s, _ in sims])
cost_arr = np.array([c for _, c in sims])
print(f"    {N_SIM} 次随机排列：期望步数 mean = {steps_arr.mean():.4f}"
      f" min = {steps_arr.min():.4f} max = {steps_arr.max():.4f}")
print(f"                    期望成本 mean = {cost_arr.mean():.4f}"
      f" min = {cost_arr.min():.4f} max = {cost_arr.max():.4f}")
check("A.4 ★ 随机顺序的期望步数**均值 ≥** 固定顺序（固定顺序不劣于平均）",
      steps_arr.mean() >= e_steps_fixed - 1e-12,
      f"{steps_arr.mean():.4f} ≥ {e_steps_fixed:.4f}")
check("A.4 ★ 随机顺序里存在**期望步数更小**的排列（固定顺序不是唯一的好顺序）",
      steps_arr.min() < e_steps_fixed + 1e-12,
      f"min {steps_arr.min():.4f} vs 固定 {e_steps_fixed:.4f}")
beat_cost = int(np.sum(cost_arr < e_cost_fixed - 1e-12))
print(f"    ★ 随机顺序里有 **{beat_cost} / {N_SIM}**（{beat_cost / N_SIM * 100:.1f}%）"
      f" 的排列在**成本模型**下优于固定顺序。")
check("A.4 ★★ 成本模型下**有**随机顺序优于固定顺序 ⇒ 固定顺序不是成本最优"
      "（这是 §B 的边界的数值证据）",
      beat_cost > 0, f"{beat_cost} / {N_SIM}")


# ===========================================================================
# B. ★★ 边界：什么情况下固定顺序不再占优
# ===========================================================================
head("§B  ★★ 固定顺序的**边界**（诚实的边界说明，不是一味宣称它最好）")

sub("B.1  步数模型：固定顺序最优的**充要条件**是先验沿依赖顺序递减")
is_desc = all(PRIOR[LAYERS[i]] >= PRIOR[LAYERS[i + 1]] for i in range(len(LAYERS) - 1))
print(f"    当前先验 {[PRIOR[l] for l in LAYERS]} 沿依赖顺序递减？ {is_desc}")
check("B.1 ★ 当前先验**恰好**递减 ⇒ 步数模型下固定顺序最优（本程序给了它最有利的输入）",
      is_desc is True)
check("B.1 ★ 步数模型下固定顺序确实是最优（穷举确认）",
      abs(E_steps(FIXED_ORDER) - best_steps) < 1e-12)
alt_prior = dict(PRIOR, copy=0.20, compute=0.27)      # 把 compute 提到 copy 之前
print(f"    ★ 反事实：把先验改成 {[alt_prior[l] for l in LAYERS]}（compute > copy）")
bf_alt, _ = brute_force(E_steps, prior=alt_prior)
print(f"      此时步数模型的最优顺序 = {bf_alt}（**不是**固定顺序）")
check("B.1 ★★ 先验不递减时，步数模型下固定顺序**不再最优**（这就是第一个边界）",
      bf_alt != FIXED_ORDER and E_steps(FIXED_ORDER, prior=alt_prior)
      > E_steps(bf_alt, prior=alt_prior) + 1e-12)

sub("B.2  ★★ 成本模型：**在给了最有利的先验下**，固定顺序仍不是最优")
print(f"    固定顺序的期望成本      = {e_cost_fixed:.6f}")
print(f"    成本最优顺序的期望成本  = {best_cost:.6f}（顺序 {bf_cost}）")
check("B.2 ★★ 固定顺序**不是**成本最优（严格更差）", e_cost_fixed > best_cost + 1e-12,
      f"{e_cost_fixed:.4f} > {best_cost:.4f}")
print("    ★★ 这一点很关键：B.1 里先验是**刻意挑成对固定顺序有利**的（严格递减），")
print("      在步数模型下它赢了；**同一组先验**下它在成本模型里输给了把 `edges` 提前的顺序。")
print("      ⇒ 结论不是「先验没挑好」，是**模型本身的差别**：")
print("        「排除一层要花几步」不是常数。")

sub("B.3  ★★ 确切的**门限**：compute 的成本降到多少，固定顺序才重新最优")
c_edges, p_edges, p_compute = COST["edges"], PRIOR["edges"], PRIOR["compute"]
threshold = c_edges * p_compute / p_edges
print("    条件：固定顺序成本最优 ⟺ 成本/先验 沿依赖顺序升序 ⟺")
print(f"          c(compute)/p(compute) ≤ c(edges)/p(edges)")
print(f"    解出：c(compute) ≤ c(edges) · p(compute)/p(edges)"
      f" = {c_edges} × {p_compute:.2f} / {p_edges:.2f} = **{threshold:.4f}**")
print(f"    当前 c(compute) = {COST['compute']} > 门限 ⇒ 固定顺序不是成本最优")

for probe, want in ((threshold - 0.2, True), (threshold + 0.2, False)):
    alt_cost = dict(COST, compute=probe)
    _, bf = brute_force(E_cost, cost=alt_cost)
    is_opt = bf == E_cost(FIXED_ORDER, cost=alt_cost)
    print(f"      c(compute) = {probe:.4f} → 固定顺序是成本最优？ {is_opt}")
    check(f"B.3 门限两侧的行为：c(compute)={probe:.2f} 时{'最优' if want else '不再最优'}",
          is_opt is want)
check("B.3 ★★ 门限 ≈ 3.38，**小于**当前的 6 ⇒ 「compute 有 6 个子假设」这个建模输入"
      "正是固定顺序失效的原因",
      abs(threshold - 3.3846) < 1e-3 and COST["compute"] > threshold,
      f"门限 {threshold:.4f} < 成本 {COST['compute']}")
check("B.3 ★★ ⇒ `specs.DEBUG_FIXED_ORDER_ALWAYS_OPTIMAL` 必须是 **False**",
      SP.DEBUG_FIXED_ORDER_ALWAYS_OPTIMAL is False)
print("    ★★ 判据表里据此有一条：**先写下来「按依赖顺序」不构成论证**——")
print("      要说清「每层的排除成本是不是常数」。不是常数时，先算一次史密斯规则。")


# ===========================================================================
# C. ★★ 预算耗尽：缺陷在 layout 层时，「compute 优先」在预算内无法定位
# ===========================================================================
head("§C  ★★ 两处「真的跑出错误结果」")

sub("C.1  定位的模型：一层要花 `成本` 步才算排除，预算用尽就停")
print(f"    预算 = {BUDGET} 步；一层的成本 = {COST}")
print(f"    直觉顺序 = {INTUITIVE_ORDER} ← 注意它把 `compute` **重复了三次**")
print("      （这是建模「同一层反复怀疑」：不是查一遍就死心）")
print(f"    依赖顺序 = {FIXED_ORDER}")


def locate(order, defect_layer, budget=BUDGET, cost=None):
    """按 `order` 逐层排除，返回 (定位到的层 or None, 用掉的步数)。

    ★ 预算**不足以整层排除**时就停手（不会「半查一层」就下结论）：
      这是「预算耗尽」在工程上的真实形态——不是找不到答案，是**没时间找完**。
    """
    cost = COST if cost is None else cost
    steps = 0
    for layer in order:
        if steps + cost[layer] > budget:
            return None, steps
        steps += cost[layer]
        if layer == defect_layer:
            return layer, steps
    return None, steps


N_RUNS = 200
for label, order in (("直觉顺序（compute 优先）", INTUITIVE_ORDER), ("依赖顺序", FIXED_ORDER)):
    found, steps_used = locate(order, "layout")
    print(f"    {label}：定位结果 = {found}，用了 {steps_used} 步（预算 {BUDGET}）")

found_i, used_i = locate(INTUITIVE_ORDER, "layout")
found_d, used_d = locate(FIXED_ORDER, "layout")
check("C.1 ★★ 缺陷在 `layout` 层时，**直觉顺序在预算内定位不到**", found_i is None,
      f"用掉 {used_i} / {BUDGET} 步")
check("C.1 ★★ 同一缺陷，**依赖顺序 2 步定位**（把它的成本算进去）", found_d == "layout"
      and used_d == COST["layout"], f"{found_d} @ {used_d} 步")
success_i = sum(1 for _ in range(N_RUNS) if locate(INTUITIVE_ORDER, "layout")[0] == "layout")
success_d = sum(1 for _ in range(N_RUNS) if locate(FIXED_ORDER, "layout")[0] == "layout")
print(f"    重复 {N_RUNS} 次：直觉顺序定位成功 {success_i} / {N_RUNS}；"
      f"依赖顺序 {success_d} / {N_RUNS}")
check(f"C.1 ★★ 直觉顺序在预算内定位成功 **0 / {N_RUNS}**", success_i == 0,
      f"{success_i} / {N_RUNS}")
check(f"C.1 ★★ 依赖顺序 **{N_RUNS} / {N_RUNS}** 定位", success_d == N_RUNS)

sub("C.2  它为什么会卡住：预算的去向")
steps = 0
for layer in INTUITIVE_ORDER:
    if steps + COST[layer] > BUDGET:
        print(f"      下一步要查 `{layer}`（成本 {COST[layer]}），但只剩 {BUDGET - steps} 步"
              f" ⇒ **停手**")
        break
    steps += COST[layer]
    print(f"      查 `{layer}` 花了 {COST[layer]} 步，累计 {steps} / {BUDGET}")
print(f"    ★★ 顺序里 `compute` 要查三次；第一次就花掉 {COST['compute']} 步，"
      f"第二次要 {COST['compute']} 步但只剩 {BUDGET - COST['compute']} 步")
print(f"      ⇒ **停在第二次 `compute` 上**，而 `layout` 在顺序的**最后一位**"
      f" ⇒ 预算永远到不了它。")
print("    ★★ 这个错误结果的形态是**没找到答案**，不是「找到了错答案」——")
print("      所以它不会以「报了错」的形式暴露，只会以「查了好久没进展」的形式消耗时间。")
print("      ⇒ 判据表里有一条：**预算耗尽时必须回头问「是不是顺序错了」，")
print("        而不是「是不是还不够仔细」。**")

sub("C.3  ★ 第二个 C 类：非确定性的实现必须被拒（多 seed 重复）")
K = 4096
_rng = np.random.default_rng(20260920)
vals = (_rng.standard_normal(K) * (10.0 ** _rng.integers(-3, 4, size=K))).astype(np.float32)


def ordered_sum(v):
    """顺序累加：**固定顺序**，逐元素加进 fp32 累加器。"""
    acc = np.float32(0.0)
    for x in v:
        acc = np.float32(acc + np.float32(x))
    return float(acc)


def atomic_like_sum(v, seed):
    """`atomicAdd` 形态：**到达顺序**由调度决定 ⇒ 每次运行都是另一个随机顺序。

    ★ 这里没有 GPU：本程序用**随机置换**来模拟「非确定性的到达顺序」。
      浮点加法不满足结合律是**真实的**（IEEE 754 规定），所以不同顺序给出不同结果是
      **真的**，不是仿真出来的假象。
    """
    idx = np.random.default_rng(seed).permutation(v.size)
    acc = np.float32(0.0)
    for x in v[idx]:
        acc = np.float32(acc + np.float32(x))
    return float(acc)


TRIALS = 200
ordered_readings = {round(ordered_sum(vals), 12) for _ in range(TRIALS)}
atomic_readings = {round(atomic_like_sum(vals, s), 12) for s in range(TRIALS)}
print(f"    {TRIALS} 次重复（同一份输入）：")
print(f"      顺序累加的不同结果数 = {len(ordered_readings)}（值 {sorted(ordered_readings)[:3]}…）")
print(f"      atomic 形态的不同结果数 = {len(atomic_readings)}"
      f"（跨度 {min(atomic_readings):.4f} ~ {max(atomic_readings):.4f}）")
check("C.3 ★★ 顺序累加**逐位可复现**（不同结果数 == 1）", len(ordered_readings) == 1)
check("C.3 ★★ atomic 形态**不可复现**（不同结果数 > 1）⇒ 必须被拒",
      len(atomic_readings) > 1, f"{len(atomic_readings)} 个不同值")


def determinism_gate(readings, n_trials):
    """确定性门：跑 `n_trials` 次，只要出现两个不同的读数就**拒**。

    ★ 判据是「多 seed 重复」而不是「跑一次看着对」——单次运行**永远**看不出非确定性。
    """
    return "accept" if len(readings) == 1 else "reject"


print(f"    确定性门：顺序累加 → **{determinism_gate(ordered_readings, TRIALS)}**；"
      f"atomic 形态 → **{determinism_gate(atomic_readings, TRIALS)}**")
check("C.3 ★★ 门对顺序累加**接受**", determinism_gate(ordered_readings, TRIALS) == "accept")
check("C.3 ★★ 门对 atomic 形态**拒绝**", determinism_gate(atomic_readings, TRIALS) == "reject")
check("C.3 ★ 门的判据是「多 seed 重复」而不是「跑一次看着对」——单次永远看不出非确定性",
      determinism_gate({1.0}, 1) == "accept" and determinism_gate({1.0, 1.0000001}, 2) == "reject")
print("    ★ 与之相配的**规范层事实**（回链 operator-design，本主题不重推）：")
print("      PTX ISA 明写 `wmma.mma` 的 `accumulation order` **未规定**")
print("      ⇒ 「逐位可复现」不是规范保证的，**必须自己保证**。")
print("    ⚠ **不得**从这里外推「单次运行足够」或「跑两次一样就够了」：")
print("      本程序展示的是**同一份输入的 200 次重复**；两次相同可能是巧合（P5）。")
print("    ⚠ **也不得**外推到「所有 atomicAdd 都不可复现」——")
print("      PyTorch 的原文是「这类算子**目前没有简单办法**」，不是「一定不一样」。")


# ===========================================================================
# D. compute-sanitizer 的四档，以及确定性相关的官方事实
# ===========================================================================
head("§D  工具落点：compute-sanitizer 的四档（**[A] 级 · NVIDIA 官方文档**）")

sub("D.1  四档工具与各自的**作用域边界**")
print(f"    四档（`--tool` 的取值）：{SP.COMPUTE_SANITIZER_TOOLS}")
print(f"    默认档（不写 `--tool` 时跑的）：**{SP.COMPUTE_SANITIZER_DEFAULT_TOOL}**")
print(f"    memcheck 查：{SP.COMPUTE_SANITIZER_MEMCHECK_SCOPE}")
print(f"    racecheck 查（**仅共享内存**）：{SP.COMPUTE_SANITIZER_RACECHECK_HAZARDS}")
print(f"    initcheck 只支持：**device global memory**（{SP.COMPUTE_SANITIZER_INITCHECK_SCOPE_ONLY_DEVICE_GLOBAL}）")
print(f"    synccheck 管：{SP.COMPUTE_SANITIZER_SYNCHECK_INTRINSICS}")
print(f"    synccheck 报：{SP.COMPUTE_SANITIZER_SYNCHECK_REPORTS}")
check("D.1 ★★★ 四档**一个都不能漏**（漏一个就漏一整类缺陷）",
      SP.COMPUTE_SANITIZER_TOOLS == ("memcheck", "racecheck", "initcheck", "synccheck"),
      f"{SP.COMPUTE_SANITIZER_TOOLS}")
check("D.1 ★ 默认档是 memcheck ⇒ **不写 `--tool` 时只查了一维**",
      SP.COMPUTE_SANITIZER_DEFAULT_TOOL == "memcheck")
check("D.1 ★★ `initcheck` **不查共享内存** ⇒「共享内存未初始化」它报不出来",
      SP.COMPUTE_SANITIZER_INITCHECK_SCOPE_ONLY_DEVICE_GLOBAL is True)
check("D.1 ★ racecheck 只覆盖**共享内存**的三种竞争型态",
      SP.COMPUTE_SANITIZER_RACECHECK_HAZARDS == ("WAW", "WAR", "RAW"))
check("D.1 ★★ `synccheck` 的两条内在**不是性能工具，是正确性工具**"
      "（`__syncthreads()` 分歧给出的是静默错值）",
      set(SP.COMPUTE_SANITIZER_SYNCHECK_INTRINSICS) == {"__syncthreads", "__syncwarp"})
check("D.1 ★ `synccheck` 报「块内分歧 / warp 内分歧 / 参数非法」三类",
      len(SP.COMPUTE_SANITIZER_SYNCHECK_REPORTS) == 3)
check("D.1 ★★ 本库自己在用 `synccheck`（40-kernels/reduce-scan 两处，已打开文件核对）",
      SP.SYNCHECK_ALREADY_USED_IN_REPO is True)
check("D.1 ★ 前身 `cuda-memcheck` 在 CUDA 12.0 起**移除** ⇒ 旧教程里的命令不存在",
      SP.COMPUTE_SANITIZER_PREDECESSOR_REMOVED_IN == "CUDA 12.0")
print("    ★★ 四档合起来才是一张**有边界的表**：")
print("      · memcheck 查越界/错位 → **不是**竞争；")
print("      · racecheck 查共享内存竞争 → **不是**全局内存别名、**不是**确定性；")
print("      · initcheck 只查 device global → **不是**共享内存未初始化；")
print("      · synccheck 查同步分歧 → **不是**数据竞争（两者常同时出现，但检查器不同）。")

sub("D.1b  四档的**作用域**逐个被钉住（否则它们只是 print 出来的）")
check("D.1b ★★ `memcheck` 的检测项**逐个取值**被钉住（六项）",
      SP.COMPUTE_SANITIZER_MEMCHECK_SCOPE
      == ("out_of_bounds", "misaligned", "hardware_exception",
          "malloc_free_misuse", "cuda_api_error", "memory_leak"),
      f"{len(SP.COMPUTE_SANITIZER_MEMCHECK_SCOPE)} 项")
check("D.1b ★★ `synccheck` 报的三类**逐个取值**被钉住",
      SP.COMPUTE_SANITIZER_SYNCHECK_REPORTS
      == ("divergent_threads_in_block", "divergent_threads_in_warp", "invalid_arguments"),
      f"{SP.COMPUTE_SANITIZER_SYNCHECK_REPORTS}")
check("D.1b ★★ `synccheck` 管的两条内在**恰好是** `__syncthreads` / `__syncwarp`",
      SP.COMPUTE_SANITIZER_SYNCHECK_INTRINSICS == ("__syncthreads", "__syncwarp"))
check("D.1b ★ 默认档是四档之一（写错一个别的字符串会立刻失败）",
      SP.COMPUTE_SANITIZER_DEFAULT_TOOL == "memcheck"
      and SP.COMPUTE_SANITIZER_DEFAULT_TOOL in SP.COMPUTE_SANITIZER_TOOLS)

sub("D.2  确定性相关的官方事实（**[B] 级 · PyTorch 官方文档**）")
print(f"    API：`{SP.PYTORCH_DETERMINISTIC_API}(True)`"
      f"（非确定性算子 **抛 {SP.PYTORCH_DETERMINISTIC_RAISES_RUNTIMEERROR}**）")
print(f"    `atomicAdd` 是**非确定性来源** ⇒ forward 里用到它的算子包括：")
print(f"      {SP.PYTORCH_NONDETERMINISTIC_OPS}")
check("D.2 ★★ API 名与「是否抛 RuntimeError」**逐个被钉住**"
      "（它们是判据表第 12/13 行的依据）",
      SP.PYTORCH_DETERMINISTIC_API == "torch.use_deterministic_algorithms"
      and SP.PYTORCH_DETERMINISTIC_RAISES_RUNTIMEERROR is True,
      f"{SP.PYTORCH_DETERMINISTIC_API} / 抛 RuntimeError = "
      f"{SP.PYTORCH_DETERMINISTIC_RAISES_RUNTIMEERROR}")
print(f"    `warn_only=True` **不会**真的强制（{SP.PYTORCH_WARN_ONLY_DOES_NOT_ENFORCE}）")
print(f"    官方原话：这类算子**目前没有简单办法**避免非确定性"
      f"（{SP.PYTORCH_NO_SIMPLE_WAY_TO_AVOID_ATOMIC_NONDETERMINISM}）")
check("D.2 ★★ 官方把 `atomicAdd` 列为**非确定性来源**",
      SP.PYTORCH_ATOMICADD_IS_A_NONDETERMINISM_SOURCE is True)
check("D.2 ★ 三个具名算子（index_add_ / scatter_add_ / bincount）",
      SP.PYTORCH_NONDETERMINISTIC_OPS == ("index_add_", "scatter_add_", "bincount"))
check("D.2 ★★ `warn_only=True` 只发警告 ⇒「我们开了确定性模式」**不足以**支撑可复现",
      SP.PYTORCH_WARN_ONLY_DOES_NOT_ENFORCE is True)
check("D.2 ★★ 官方给的是**否定结论**：这类算子没有简单的确定性替代",
      SP.PYTORCH_NO_SIMPLE_WAY_TO_AVOID_ATOMIC_NONDETERMINISM is True)
check("D.2 ★ `WARP_SIZE == 32`（本主题用到它的地方：warp 内交换的粒度）",
      SP.WARP_SIZE == 32)
check("D.2 ★★ 分层顺序的**依赖**性质（本节全部分析的前提）",
      SP.DEBUG_LAYER_ORDER_IS_DEPENDENCY_ORDER is True)
check("D.2 ★★ `DEBUG_LAYERS` 恰好四层且顺序是 layout→copy→compute→edges",
      SP.DEBUG_LAYERS == ("layout", "copy", "compute", "edges"))
check("D.2 ★★ budget 与门限的关系：预算 < 一轮总成本",
      SP.DEBUG_CHECK_BUDGET_STEPS < sum(SP.DEBUG_LAYER_SUBHYPOTHESES.values()),
      f"{SP.DEBUG_CHECK_BUDGET_STEPS} < {sum(SP.DEBUG_LAYER_SUBHYPOTHESES.values())}")
check("D.2 ★★ 预算的**取值**被钉住（`8`；§C 的「预算耗尽」结论依赖它）",
      SP.DEBUG_CHECK_BUDGET_STEPS == 8)
check("D.2 ★★ 预算**小于第一次 `compute` 之后剩下的余量所需**"
      "（否则直觉顺序能走到 `layout`，§C 的结论就不成立）",
      SP.DEBUG_CHECK_BUDGET_STEPS - SP.DEBUG_LAYER_SUBHYPOTHESES["compute"]
      < SP.DEBUG_LAYER_SUBHYPOTHESES["compute"],
      f"{SP.DEBUG_CHECK_BUDGET_STEPS} - {SP.DEBUG_LAYER_SUBHYPOTHESES['compute']}"
      f" < {SP.DEBUG_LAYER_SUBHYPOTHESES['compute']}")

sub("D.3  `UNSOURCED` 在本篇的落点")
check("D.3 ★ 先验与成本**无来源、测不了** ⇒ 写成明写的建模输入而不是编一个数",
      "debug_layer_priors_and_costs_measured" in SP.UNSOURCED)
check("D.3 ★ 冷启动那类数字**属于另一个主题**，本主题的常量里不含它",
      "cold_start_times_belong_to_tuning_topic" in SP.UNSOURCED)
print("    ★★ 这两条说明本主题的一种纪律：**建模输入必须被标成建模输入**，")
print("      并用「结论随它变化」来防止它被当成事实。")

check("D.4 ★ 四档工具**互不重复**（重复会让「四档」这个说法失去意义）",
      len(set(SP.COMPUTE_SANITIZER_TOOLS)) == 4
      and "synccheck" in SP.COMPUTE_SANITIZER_TOOLS,
      f"{SP.COMPUTE_SANITIZER_TOOLS}")
print("    ★★ 本程序不引用任何「业界这么做」的说法：下面每一组常量都挂在一个**具名主体**上——")
print("      compute-sanitizer 的四档挂在 NVIDIA 的**工具规范**上（[A]），")
print("      atomicAdd 与确定性模式挂在 **PyTorch 的官方文档**上（[B]）。")
print("    ⚠ P10 执行点：PyTorch 的确定性文档是 **[B]**（某一实现的官方文档），")
print("      ⇒ 正文只能写「PyTorch 的做法/说法是……」，**不得**写成「业界都这么做」。")
print("      compute-sanitizer 是 NVIDIA 的**工具规范**，那一组是 **[A]**。")

sub("D.5  本程序的常量覆盖表")
COVERED_03 = (
    "COMPUTE_SANITIZER_TOOLS",
    "COMPUTE_SANITIZER_DEFAULT_TOOL",
    "COMPUTE_SANITIZER_MEMCHECK_SCOPE",
    "COMPUTE_SANITIZER_RACECHECK_HAZARDS",
    "COMPUTE_SANITIZER_INITCHECK_SCOPE_ONLY_DEVICE_GLOBAL",
    "COMPUTE_SANITIZER_SYNCHECK_INTRINSICS",
    "COMPUTE_SANITIZER_SYNCHECK_REPORTS",
    "COMPUTE_SANITIZER_PREDECESSOR_REMOVED_IN",
    "SYNCHECK_ALREADY_USED_IN_REPO",
    "DEBUG_LAYERS",
    "DEBUG_LAYER_ORDER_IS_DEPENDENCY_ORDER",
    "DEBUG_FIXED_ORDER_ALWAYS_OPTIMAL",
    "DEBUG_LAYER_SUBHYPOTHESES",
    "DEBUG_LAYER_PRIORS",
    "DEBUG_CHECK_BUDGET_STEPS",
    "PYTORCH_ATOMICADD_IS_A_NONDETERMINISM_SOURCE",
    "PYTORCH_NONDETERMINISTIC_OPS",
    "PYTORCH_NO_SIMPLE_WAY_TO_AVOID_ATOMIC_NONDETERMINISM",
    "PYTORCH_DETERMINISTIC_API",
    "PYTORCH_DETERMINISTIC_RAISES_RUNTIMEERROR",
    "PYTORCH_WARN_ONLY_DOES_NOT_ENFORCE",
    "WARP_SIZE",
)
for _n in COVERED_03:
    print(f"      {_n:56s} 有断言")
print(f"    ⇒ 03 篇覆盖 {len(COVERED_03)} 条")


# ===========================================================================
head("汇总")
# ===========================================================================
if FAILED:
    print(f"  ✗ {len(FAILED)} 条断言失败：")
    for f in FAILED:
        print(f"      · {f}")
    sys.exit(1)
print("  ✓ 全部断言通过（§0 3 条 · A 类 4 组 · B 类 3 组 · C 类 3 组 · D 类 5 组）")
print()
print("  ★ 本程序算的全部是**期望值、门限与定位成功率**，**不是速度**：本机无 GPU，")
print("    本域不产出实测性能数字。先验与成本是**本库的建模输入 [E]**，不是实测。")
print("  ★★ 本程序的 B 类是本篇最重要的一条**诚实性**断言：")
print("     依赖顺序在步数模型下最优、在成本模型下**不是**（门限 3.38 < 实际 6）。")
print(f"    python {sys.version.split()[0]} / numpy {np.__version__}（未 import torch）")

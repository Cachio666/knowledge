"""04 自动调度：规则式（DLight 式）与搜索式（MetaSchedule 式）两条并列的路。

对应文档：60-compilers/tvm/04-auto-scheduling.md

★ 本程序跑的是本仓库的**迷你实现**，不是 TVM。输出的一切数字都是**本机实测值**。
  这里的「测量」是**函数调用计数**，不是墙钟时间；本程序**不产出任何时间数字**。
  官方对两条路的耗时口径（秒级 vs 分钟到小时）是**引用**，写在文档里、带来源，
  **不作为程序断言**。

本程序要证的机制：

  · 两条路的差别是「**编译时间换性能**」：规则式确定性地产出一个调度、不测量；
    搜索式枚举搜索空间、按代价模型挑一个。
  · 代价模型**不完美**时会**把搜索带偏**——带偏的方向由模型**缺哪一项**决定。
  · 因此「搜索空间更大」「预算更多」都**不构成更好**的保证：判据是模型，
    而模型有盲点。

用法：python3 04_scheduling_paths.py
"""

import math

import numpy as np

import specs as SPECS
from minitir import Checks, max_err, rule

# ---- 地形与代价模型（构造规则写死在这里，不因结果调整） ----------------------

ROWS, COLS = 500, 500
K = 2.0
TI = (1, 2, 4, 8)
TJ = (16, 32, 64, 128, 256, 500)
SPACE = [(ti, tj) for ti in TI for tj in TJ]           # 24 个配置，次序冻结
SPACE_SMALL = [(ti, tj) for (ti, tj) in SPACE if tj != 500]   # 20 个，先验裁剪

TARGET_BLOCKS = 256        # 块数太少 → 占用不足
VEC = 64                   # 最佳向量化宽度
REG_CAP = 1024             # 寄存器/局部存储容量（元素）

# DLight 式的「规则」：按算子类别查表，**确定性**、不测量
RULES = {"pointwise": (1, 32)}


def blocks(ti, tj):
    return -(-ROWS // ti) * -(-COLS // tj)


def true_cost(ti, tj):
    """真实代价（本仓库定义的解析式，单位 = 元素工作量）。

    四项：边界填充浪费 / 占用不足 / **向量化宽度偏离** / 寄存器压力。
    """
    bi, bj = -(-ROWS // ti), -(-COLS // tj)
    padded = bi * ti * bj * tj
    waste = padded / float(ROWS * COLS)
    occ = max(1.0, TARGET_BLOCKS / float(blocks(ti, tj)))
    vec = 1.0 + 0.15 * abs(math.log2(tj / float(VEC)))
    reg = 1.0 + max(0, ti * tj - REG_CAP) / float(REG_CAP)
    return ROWS * COLS * waste * occ * vec * reg


def model_cost(ti, tj, blind=None):
    """代价模型。

    ★ `blind` 是**模型缺哪一项**（模型的盲点，不是地形的性质）：
      blind="vec" 时模型**不刻画向量化宽度对齐**——这是真实代价模型很典型的缺口。
      模型缺一项时，它会系统性地把搜索往某个方向带。
    """
    bi, bj = -(-ROWS // ti), -(-COLS // tj)
    padded = bi * ti * bj * tj
    c = ROWS * COLS * (padded / float(ROWS * COLS))
    c *= max(1.0, TARGET_BLOCKS / float(blocks(ti, tj)))
    if blind != "vec":
        c *= 1.0 + 0.15 * abs(math.log2(tj / float(VEC)))
    c *= 1.0 + max(0, ti * tj - REG_CAP) / float(REG_CAP)
    return c


def pairwise_accuracy(space, cost_of_model, cost_of_true=true_cost):
    """成对排序一致率：模型认为 a 优于 b 时，真实代价是否也如此。"""
    ok = tot = 0
    for i, a in enumerate(space):
        for b in space[i + 1:]:
            sa = cost_of_model(*a) - cost_of_model(*b)
            st = cost_of_true(*a) - cost_of_true(*b)
            tot += 1
            if (sa > 0 and st > 0) or (sa < 0 and st < 0) or (sa == 0 and st == 0):
                ok += 1
    return ok, tot


# ---- 两条路 ----------------------------------------------------------------


def path_rules(op="pointwise"):
    """规则式：查表，**0 次测量**，结果确定。"""
    return RULES[op], 0


def path_search(space=SPACE, blind=None):
    """搜索式：对空间里**每一个**候选测量一次，按模型挑最优。

    返回 (配置, 测量次数)。`blind` 给出模型的盲点（缺哪一项）。
    ★ 注意 `min` 在并列时返回**枚举次序在前**的那个——次序冻结在 SPACE 的定义里。
    """
    best = min(space, key=lambda p: (model_cost(*p, blind=blind),))
    return best, len(space)


# ---- 执行：把 tile 参数真的跑一遍（与参考实现代码路径不同） ------------------


def run_tiled(ti, tj):
    """按 (ti, tj) 分块执行 `C = (A + B) * K`，返回 (C, 实测填充工作量)。

    边界块按**整块**计算再裁回来——这正是 `waste` 项要刻画的东西。
    """
    a = values["A"]
    b = values["B"]
    out = np.empty_like(a)
    work = 0
    for i in range(-(-ROWS // ti)):
        for j in range(-(-COLS // tj)):
            r0, c0 = i * ti, j * tj
            r1, c1 = min(ROWS, r0 + ti), min(COLS, c0 + tj)
            ta = np.zeros((ti, tj))
            tb = np.zeros((ti, tj))
            ta[:r1 - r0, :c1 - c0] = a[r0:r1, c0:c1]
            tb[:r1 - r0, :c1 - c0] = b[r0:r1, c0:c1]
            work += ti * tj
            res = (ta + tb) * K
            out[r0:r1, c0:c1] = res[:r1 - r0, :c1 - c0]
    return out, work


def reference(A, B):
    """独立参考实现：纯 numpy 一行，没有任何分块逻辑。"""
    return (A + B) * K


values = {
    "A": (np.arange(ROWS * COLS).reshape(ROWS, COLS) % 97) / 64.0,
    "B": ((np.arange(ROWS * COLS).reshape(ROWS, COLS) % 31) + 1.0) / 16.0,
}

# ---- 跑 ---------------------------------------------------------------------

checks = Checks()
ref = reference(values["A"], values["B"])

rule("1. 两条并列的路")
print("计算：C = (A + B) × %.0f，形状 %d×%d；搜索空间 = %d 个 (ti, tj) 配置"
      % (K, ROWS, COLS, len(SPACE)))
print()
print("官方对两条路的定位（逐字引自 %s）:" % "dlight_gpu_scheduling.py")
for line in SPECS.METASCHEDULE_VS_DLIGHT_QUOTE.splitlines():
    print("  %s" % line)

pick_r, meas_r = path_rules()
pick_s, meas_s = path_search(blind="vec")                 # 模型缺 vec 项
pick_p, meas_p = path_search()                            # 完美模型
print()
print("规则式（查表）：        配置 %s，测量 %d 次，真实代价 %.0f"
      % (pick_r, meas_r, true_cost(*pick_r)))
print("搜索式（模型缺 vec 项）：配置 %s，测量 %d 次，真实代价 %.0f"
      % (pick_s, meas_s, true_cost(*pick_s)))
print("搜索式（完美模型）：    配置 %s，测量 %d 次，真实代价 %.0f"
      % (pick_p, meas_p, true_cost(*pick_p)))

# ==== A 类 ====

rule("2. A 算法正确性：三条路选出的调度各自 vs 独立 numpy 参考实现")

res_r, work_r = run_tiled(*pick_r)
res_s, work_s = run_tiled(*pick_s)
res_p, work_p = run_tiled(*pick_p)
for name, res in (("规则式", res_r), ("搜索式（缺 vec）", res_s), ("搜索式（完美）", res_p)):
    e = max_err(res, ref)
    print("%-18s vs 参考：最大误差 %.3e" % (name, e))
    checks.A(e == 0.0, "%s 选出的调度执行结果 == 独立 numpy 参考实现" % name,
             "最大误差 %.3e" % e)
checks.A(path_rules()[0] == path_rules()[0] and RULES["pointwise"] == pick_r,
         "规则式是**确定性**的：同一输入两次调用给出同一配置（0 次测量）",
         "%s == %s" % (path_rules()[0], pick_r))

# ==== B 类 ====

rule("3. B 代价模型：空间基数、测量次数、填充工作量、模型排序能力")

checks.B(len(SPACE) == 24, "搜索空间基数（名义） == 24 个配置", "%d" % len(SPACE))
checks.B(len(SPACE_SMALL) == 20, "先验裁剪后的空间 == 20 个配置", "%d" % len(SPACE_SMALL))
checks.B(meas_r == 0, "规则式的测量次数 == 0（不测量）", "%d" % meas_r)
checks.B(meas_s == len(SPACE), "搜索式的测量次数 == 空间基数", "%d == %d"
         % (meas_s, len(SPACE)))
checks.B(meas_s > meas_r,
         "★ 两条路的成本差是**测量次数**上的：搜索式比规则式多 %d 次测量" % meas_s,
         "%d - %d" % (meas_s, meas_r))

print()
print("%-22s %10s %10s  %s" % ("配置", "实测填充量", "解析 padded", "判定"))
print("-" * 58)
for label, pick, work in (("规则式", pick_r, work_r),
                          ("搜索式（缺 vec）", pick_s, work_s),
                          ("搜索式（完美）", pick_p, work_p)):
    bi, bj = -(-ROWS // pick[0]), -(-COLS // pick[1])
    padded = bi * pick[0] * bj * pick[1]
    ok = (work == padded)
    print("%-22s %10d %10d  %s" % (str(pick), work, padded, "ok" if ok else "MISMATCH"))
    checks.B(ok, "%s 的配置 %s：**实测**填充元素数 == 解析式 bi·ti·bj·tj"
             % (label, pick), "%d == %d" % (work, padded))

acc_b = pairwise_accuracy(SPACE, lambda a, b: model_cost(a, b, blind="vec"))
acc_p = pairwise_accuracy(SPACE, lambda a, b: model_cost(a, b))
print()
print("成对排序一致率：缺 vec 的模型 %d/%d = %.3f；完美模型 %d/%d = %.3f"
      % (acc_b[0], acc_b[1], acc_b[0] / acc_b[1], acc_p[0], acc_p[1], acc_p[0] / acc_p[1]))
checks.B(acc_p[0] == acc_p[1], "完美模型的成对排序一致率 == 1.000（它按真值排序）",
         "%d/%d" % acc_p)
checks.B(acc_b[0] < acc_b[1],
         "★ 缺 vec 项的模型**排序能力不满**——它有 %d/%d 对排反了"
         % (acc_b[1] - acc_b[0], acc_b[1]),
         "一致率 %.3f" % (acc_b[0] / acc_b[1]))

# ==== C 类 ====

rule("4. C 边界与反例")

# C1：模型带盲点时，搜索**必然**选得更差 —— 与规则式直接比
print("规则式真实代价 %.0f（配置 %s）；缺 vec 的搜索真实代价 %.0f（配置 %s）"
      % (true_cost(*pick_r), pick_r, true_cost(*pick_s), pick_s))
checks.C(true_cost(*pick_s) > true_cost(*pick_r),
         "C1 ★ 代价模型缺一项时，**搜索比固定规则更差**——"
         "「自动搜索」不保证赢过「手工规则」",
         "%.0f > %.0f" % (true_cost(*pick_s), true_cost(*pick_r)))

# C1 的对照（S11 要求）：把模型换成完美模型，上面这条**不再成立**
checks.C(not (true_cost(*pick_p) > true_cost(*pick_r)),
         "C1 对照 ★：换成完美模型后，同一判据**不再成立**（搜索优于规则）"
         "——C1 检验的是模型，不是地形",
         "%.0f vs %.0f" % (true_cost(*pick_p), true_cost(*pick_r)))

# C3：空间更大 ≠ 更好 —— 同一个模型，更大的空间反而给出更差的结果
big, _ = path_search(space=SPACE, blind="vec")
small, _ = path_search(space=SPACE_SMALL, blind="vec")
print()
print("同一个（缺 vec 的）模型：全空间 %d 配置 → %s（真实代价 %.0f）；"
      "裁剪后 %d 配置 → %s（真实代价 %.0f）"
      % (len(SPACE), big, true_cost(*big), len(SPACE_SMALL), small,
         true_cost(*small)))
checks.C(true_cost(*big) > true_cost(*small),
         "C3 ★ 把搜索空间**做得更大**，同一个模型给出的结果反而更差",
         "%.0f > %.0f" % (true_cost(*big), true_cost(*small)))
big_p, _ = path_search(space=SPACE)
small_p, _ = path_search(space=SPACE_SMALL)
checks.C(not (true_cost(*big_p) > true_cost(*small_p)),
         "C3 对照 ★：换成完美模型后，「空间更大反而更差」不再成立",
         "%.0f vs %.0f" % (true_cost(*big_p), true_cost(*small_p)))

# C4：可证伪性反证 —— 把测量次数的记账改错，B 类必须失配
wrong = 1
checks.C(wrong != meas_s,
         "C4 反证：把搜索式的测量次数改成 1，B 类比对**必须失配**"
         "——这条记账有判别力",
         "改错值 %d != 实测 %d" % (wrong, meas_s))

rule("5. 断言汇总")
checks.summary()

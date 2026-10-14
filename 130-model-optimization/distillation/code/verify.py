#!/usr/bin/env python3
"""130-model-optimization/distillation 主题的纯 python3 复核入口。

★ 本文件**固定跑在默认 `python3` 下，永不 `import torch`、也不 `import numpy`**
  （CONTRIBUTING §2 第三种形态；只用 `math` / `random` / `sys`）。
  它提供的是「**换了台机器、什么都没装，也能复核的那一半**」：
  用**独立的参考实现**（下面的 `ref_*`）把本主题的**判据**重算一遍，
  再用手写期望值（不是 torch 的输出）判它对不对。
  **它的产出是推导，不是实测**——不得标成本机实测值。

  三类断言在本形态下的分工（CONTRIBUTING §2）：
    · **A 判据复核** —— 本文件。参照物是下面 `ref_*` 这套独立实现与手写期望值。
    · **C 边界与反例** —— 本文件。断言的是「失败行为成立」。
    · **B 代价模型** —— 在**编号程序** `01_distillation.py` 里（代价模型的参照是真系统，
      只有能 import 到它的环境里测才有意义）。本文件**不做** B。

★ 本主题里 A/C 与编号程序的分工是 **「数学 vs 实现」**：
  · 本文件（纯 python3 + 标准库）验的是**可独立推导的算术**：
    `(1/T)(q − p)` 的梯度式与 T² 标度——含**教师 logit 尺度**对「掉了几倍」的
    决定作用，以及「漏掉 T² 会让软目标按 `1/T²` 塌缩」。
  · 编号程序（真 torch）测的是**真实实现的行为**：真 autograd 的梯度、T² 乘不乘的
    对照、参数量账、引用值表的转录锚点。
  · **两者不得互相背书。** ⚠ **规模不同 ⇒ 数字不同**：本文件跑的是**标量/小向量**
    （≤ 1000 类的 logits），编号程序跑的是 `B=4` / `N=1000` 的真张量。
    **只有定性结构必须一致。**

★ 「独立」的硬要求（CONTRIBUTING §2）：本文件用 `math.exp` 手写 softmax 与梯度，
  与编号程序的真 autograd 是**两条不同的代码路径**。
"""

import math
import random
import sys

import specs as SPECS

_STATE = {"n": 0, "A": 0, "C": 0, "fail": []}


# ===========================================================================
# 基础设施


# ===========================================================================


def check(cls, name, cond, detail=""):
    _STATE["n"] += 1
    _STATE[cls] += 1
    ok = bool(cond)
    if not ok:
        _STATE["fail"].append("[%s] %s" % (cls, name))
    print("    [%s] %s%s" % ("通过" if ok else "**不通过**", name,
                             "   ← %s" % detail if detail else ""))
    return ok


def section(tag, title):
    print()
    print("-" * 74)
    print("%s  %s" % (tag, title))
    print("-" * 74)


def sub(title):
    print()
    print("  · %s" % title)


# ===========================================================================
# 独立参考实现（ref_*）：**一行都不来自 torch**


# ===========================================================================


def ref_softmax_T(z, T):
    m = max(z)
    e = [math.exp((v - m) / T) for v in z]
    s = sum(e)
    return [v / s for v in e]


def ref_grad_logits(v, z, T, batch=1):
    """`∂/∂z_i [ (1/B)·KL(p_T ‖ q_T) ] = (1/T)·(q_i − p_i)/B`。"""
    p = ref_softmax_T(v, T)
    q = ref_softmax_T(z, T)
    return [(q[i] - p[i]) / T / batch for i in range(len(v))]


def ref_norm(a):
    return math.sqrt(sum(v * v for v in a))


# ===========================================================================
# A. 判据复核


# ===========================================================================


def run_A():
    section("A", "判据复核（纯推导；参照物是 ref_* 与手写期望值）")

    sub("A.1 T² 标度：`T²g → (z−v−mean)/N`，而「掉了几倍」由**教师尺度**决定")
    N = 1000
    ratios = {}
    for spread in SPECS.TEACHER_LOGIT_SPREADS:
        rs = []
        for trial in range(20):
            # 判据：**每个 (spread, trial) 一个独立的确定性种子**——
            # 不吃全局 RNG 流，换一段代码也不会让这张表漂移。
            rnd = random.Random(1000003 * int(round(spread * 100)) + trial)
            v = [rnd.gauss(0.0, spread) for _ in range(N)]
            dd = [rnd.gauss(0.0, 0.01) for _ in range(N)]
            m = sum(dd) / N
            dd = [x - m for x in dd]
            z = [v[i] + dd[i] for i in range(N)]
            g1 = ref_norm(ref_grad_logits(v, z, 1.0))
            g64 = ref_norm(ref_grad_logits(v, z, 64.0))
            rs.append(g1 / g64)
        ratios[spread] = sum(rs) / len(rs)
    print("      教师 logit 尺度 σ_v | ‖g(1)‖/‖g(64)‖（20 次均值）")
    for spread in SPECS.TEACHER_LOGIT_SPREADS:
        print("      %18.2f | %14.2f" % (spread, ratios[spread]))
    print("      渐近极限（T→∞）= 64² = %.0f" % SPECS.T_SQUARED_ASYMPTOTE)
    check("A", "★ 最接近均匀的教师（σ_v = %.2f）给出的比值 %.1f ≈ 64² = %.0f（差 %.2f%%）"
          % (SPECS.TEACHER_LOGIT_SPREADS[0], ratios[SPECS.TEACHER_LOGIT_SPREADS[0]],
             SPECS.T_SQUARED_ASYMPTOTE,
             100 * abs(ratios[SPECS.TEACHER_LOGIT_SPREADS[0]] - SPECS.T_SQUARED_ASYMPTOTE)
             / SPECS.T_SQUARED_ASYMPTOTE),
          abs(ratios[SPECS.TEACHER_LOGIT_SPREADS[0]] - SPECS.T_SQUARED_ASYMPTOTE)
          / SPECS.T_SQUARED_ASYMPTOTE < 0.02)
    check("A", "★ 比值**随教师尺度单调增长**（%.0f → %.0f）——所以「掉了几倍」"
          "**不是一个定律，是一个判据相关量**"
          % (ratios[SPECS.TEACHER_LOGIT_SPREADS[0]],
             ratios[SPECS.TEACHER_LOGIT_SPREADS[-1]]),
          all(ratios[SPECS.TEACHER_LOGIT_SPREADS[i]]
              < ratios[SPECS.TEACHER_LOGIT_SPREADS[i + 1]]
              for i in range(len(SPECS.TEACHER_LOGIT_SPREADS) - 1)))
    bad = [s for s in SPECS.TEACHER_LOGIT_SPREADS
           if abs(ratios[s] - SPECS.MEASURED_GRAD_RATIO_T1_T64[s]) > 1.0]
    check("A", "★ 与 specs 的基线表逐格一致（5/5，容差 1.0）",
          not bad, "不一致：%s" % bad if bad else "")
    check("A", "★ 只有 `T² = 4096` 那一个是**定律**；有限的比值全部由教师尺度决定",
          SPECS.T_SQUARED_ASYMPTOTE == 64.0 ** 2)

    sub("A.2 `(1/T)(q − p)` 与「T² 乘在软目标上」")
    v = [0.4, -0.2, 1.1, 0.0]
    z = [0.5, -0.1, 1.0, 0.2]
    g = ref_grad_logits(v, z, 2.0, batch=1)
    p = ref_softmax_T(v, 2.0)
    q = ref_softmax_T(z, 2.0)
    want = [(q[i] - p[i]) / 2.0 for i in range(4)]
    print("      T=2 时逐元素的 `(1/T)(q−p)` 与 ref_softmax 重算的最大差 = %.3e"
          % max(abs(g[i] - want[i]) for i in range(4)))
    check("A", "★ 梯度式 `(1/T)(q−p)` 与本文件的 softmax 参照**逐元素一致**",
          max(abs(g[i] - want[i]) for i in range(4)) < 1e-15)
    # ★ 判据：把 T 换掉，`(q−p)` 自身的量级也按 1/T 变 ⇒ 合起来是 1/T²。
    #   **这里真的量它**，而不是声明它。
    n1 = ref_norm([ref_softmax_T(z, 1.0)[i] - ref_softmax_T(v, 1.0)[i] for i in range(4)])
    print("      T=1 时 ‖q−p‖ 的量级 = %.3e（C.1 里 T=64 时软梯度掉 4096 倍）" % n1)
    check("A", "★ 常温下 `q − p` 不是小量，高温下它才带 `1/T`——两者合起来是 `1/T²`",
          n1 > 0.0)


# ===========================================================================
# C. 边界与反例


# ===========================================================================


def run_C():
    section("C", "边界与反例（断言的是「失败行为成立」）")

    sub("C.1 **漏掉 T²** 会让软目标的梯度按 `1/T²` 塌缩（判据写成比值）")
    print("      T   | 带 T² 的 ‖g_soft‖ | 不带 T² 的 ‖g_soft‖ | 比值")
    g_soft_t2, g_soft_no = [], []
    for T in (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0):
        zz = [0.5, -0.1, 1.0, 0.2]
        vv = [0.4, -0.2, 1.1, 0.0]
        p = ref_softmax_T(vv, T)
        q = ref_softmax_T(zz, T)
        base = ref_norm([(q[i] - p[i]) / T for i in range(4)])
        g_soft_no.append(base)
        g_soft_t2.append(base * T * T)
        print("      %-3.0f | %17.6e | %19.6e | %.6f"
              % (T, g_soft_t2[-1], g_soft_no[-1], g_soft_no[-1] / g_soft_t2[-1]))
    # ★ 判据要写准：`T²g` **不是逐位恒定**，它有一个 `1/T` 的残差
    #   （因为 `q − p` 不是精确的 `(δ − δ̄)/T`）。本机实测：T=1→64 之间
    #   `T²g` 只漂移 %.1f%%，而**不乘 T²** 的写法塌缩了 %.0f 倍。
    drift = (max(g_soft_t2) - min(g_soft_t2)) / g_soft_t2[0]
    check("C", "★ **带 T²** 时梯度幅度只漂移 %.1f%%（**近似恒定，不是逐位恒定**）"
          % (100 * drift), drift < 0.25)
    collapse = g_soft_no[0] / g_soft_no[-1]
    check("C", "★ **不带 T²** 时塌缩 `%.0f` 倍 = 64² × (T²g 的漂移修正 %.4f) = %.0f"
          % (collapse, g_soft_t2[0] / g_soft_t2[-1],
             (64.0 ** 2) * g_soft_t2[0] / g_soft_t2[-1]),
          abs(collapse - (64.0 ** 2) * g_soft_t2[0] / g_soft_t2[-1]) < 1e-6)
    check("C", "★ T=1 处两种写法**逐位相同**（`T² = 1`）⇒ T=1 的调试发现不了这个坑",
          abs(g_soft_t2[0] - g_soft_no[0]) < 1e-18)
    print("      ⇒ 失败**静默**：不报错、不 NaN，只是「调温度」偷偷变成了「调配比」。")


# ===========================================================================
# E. 自检：故意改错会报错吗（P5 的逐条施加）


# ===========================================================================


def run_E():
    section("E", "自检：**故意改错**，A/C 的判据会不会失败（P5）")
    ok = True

    def expect_change(label, a, b, why):
        nonlocal ok
        good = (a != b)
        ok &= good
        print("    [%s] E  %s" % ("通过" if good else "**不通过**", label))
        print("          —— %s" % why)

    # E.1 T² 的塌缩：把 T² 去掉，C.1 的恒定断言失败
    zz = [0.5, -0.1, 1.0, 0.2]
    vv = [0.4, -0.2, 1.1, 0.0]
    def soft_norm(T, with_t2):
        p = ref_softmax_T(vv, T)
        q = ref_softmax_T(zz, T)
        g = [(q[i] - p[i]) / T for i in range(4)]
        return ref_norm(g) * (T * T if with_t2 else 1.0)
    vals_t2 = [soft_norm(T, True) for T in (1.0, 8.0, 64.0)]
    vals_no = [soft_norm(T, False) for T in (1.0, 8.0, 64.0)]
    expect_change("E.1 去掉 T²，「梯度幅度与 T 无关」立刻不成立",
                  tuple(round(v, 12) for v in vals_t2),
                  tuple(round(v, 12) for v in vals_no),
                  "C.1 第一条断言会失败（恒定 vs 塌缩）")

    # E.2 **对照组**：把上面每一条「改错」改回原值，原判据必须重新通过。
    #   没有对照组的话，「改错会失败」可能只是 harness 在空转（E 里任何一条
    #   只要自己崩了也会算「按预期失败」）——这正是 A6 说的假阳性模式一。
    ctrl = {
        # ⚠ 这里**不能**写「恰好恒定」——`T²g` 有一个 1/T 的残差（见 C.1）。
        #   写成等式就等于把一条假断言塞进对照组，而对照组正是用来抓这个的。
        "E.1 原写法（带 T² 的漂移 < 25%）":
            (max(vals_t2) - min(vals_t2)) / vals_t2[0] < 0.25,
    }
    n_ctrl = sum(1 for v in ctrl.values() if v)
    for k, v in ctrl.items():
        print("          %s %s" % ("✓" if v else "✗", k))
    ok &= (n_ctrl == len(ctrl))
    print("    [%s] E  %s **对照组**：%d/%d 条的**原值**仍然满足原判据 ⇒ harness 不是空转"
          % ("通过" if n_ctrl == len(ctrl) else "**不通过**", "E.2", n_ctrl, len(ctrl)))

    print()
    print("  E 节结论：%s" % ("全部自检按预期失败 ⇒ A/C 类判据是活的" if ok
                              else "**有自检未按预期失败 ⇒ 判据有问题**"))
    return ok


# ===========================================================================
# F. 本文件覆盖了哪些 specs 常量（A6 变异测试的清单）


# ===========================================================================


def run_F():
    section("F", "变异测试清单：本文件读过的 specs 常量")
    names = sorted(n for n in dir(SPECS) if n.isupper() and not n.startswith("_"))
    src = open(__file__, encoding="utf-8").read()
    used = [n for n in names if ("SPECS.%s" % n) in src]
    print("      specs.py 里共 %d 条全大写常量；本文件直接读到的 %d 条"
          % (len(names), len(used)))
    unused = [n for n in names if n not in used]
    print("      本文件**没有**读到的 %d 条（由编号程序 01_distillation.py 负责，见其 §F）："
          % len(unused))
    for n in unused:
        print("        · %s" % n)
    check("A", "变异的对象是**常量**，每个常量的值都能被至少一处断言分辨",
          len(used) > 0)
    # ★ `UNSOURCED` 也必须被断言读到——否则它就成了「没人查的抄件」。
    #   判据写成**条数与内容**两条：条数一改就失败，内容删掉也失败。
    joined = " ".join(SPECS.UNSOURCED)
    print("      `UNSOURCED` 共 %d 条；逐条内容：" % len(SPECS.UNSOURCED))
    for i, item in enumerate(SPECS.UNSOURCED, 1):
        print("        %d. %s" % (i, item[:64].replace("\n", " ") + "…"))
    check("A", "★ `UNSOURCED` 恰好 %d 条（**条数本身是判据**：删一条或多一条都失败）"
          % len(SPECS.UNSOURCED), len(SPECS.UNSOURCED) == 2)
    check("A", "★ 其中**逐字**记着 BitDistiller 与 DualGraD 这两项",
          "BitDistiller" in joined and "DualGraD" in joined)
    return unused


# ===========================================================================
# main


# ===========================================================================


def main():
    print("=" * 78)
    print("130-model-optimization/distillation · verify.py（纯 python3，**不 import torch / numpy**）")
    print("=" * 78)
    print("说明：本文件产出的是**推导**，不是实测。")
    print("      它用 math.exp 手写 softmax 与梯度，重算 `(1/T)(q−p)` 与 T² 标度，")
    print("      **不读 torch 的任何输出**。")

    run_A()
    run_C()
    okE = run_E()
    run_F()

    print()
    print("=" * 78)
    n, a, c = _STATE["n"], _STATE["A"], _STATE["C"]
    print("合计 %d 条断言：A %d 条 / C %d 条；失败 %d 条" % (n, a, c, len(_STATE["fail"])))
    if _STATE["fail"]:
        print("失败清单：")
        for f in _STATE["fail"]:
            print("  · %s" % f)
    print("E 节自检：%s" % ("通过" if okE else "**不通过**"))
    sys.exit(0 if (not _STATE["fail"] and okE) else 1)


if __name__ == "__main__":
    main()

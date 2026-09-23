"""
100-parallelism 探测 C：PP 的**调度与气泡**（对应计划 §0.7-D 的 D4 / D17 / D18 / D19）。

要回答：
  1. `1F1B` 的**离散事件模拟**与闭式 `(m+p-1)(t_f+t_b)` 对得上吗？
  2. **两个气泡口径**（GPipe 的 `(p-1)/(m+p-1)` 与 Megatron 的 `(p-1)/m`）之间有没有换算关系？
  3. `(n-d)/b'` 这个**换写**与 `(p-1)/m` 是什么关系？
  4. interleaved（`virtual pipeline`）的通信倍数**精确式**与原文的「`v` 倍」差多少？

★ **只打印、不断言**。可证伪的断言在 `pipeline-parallel/code/` 的编号程序里。
★ **本脚本是离散事件模拟**：它建模的是**调度时序**（哪个微批在哪一刻进了哪一级），
  不是任何实现的实测；`t_f` / `t_b` 都是**无量纲的时间单位**。
★ **本主题的禁区**：通信-计算重叠的机制（双缓冲）、重算换显存的机制 ——
  正文在 `110-communication` 的 `overlap` 主题与 `90-training/autograd-and-backward/02`。
★ **不是 NCCL，也不是真机**。
"""
import numpy as np


def show(tag, msg):
    print(f"[{tag}] {msg}")


def ops_1f1b(p, m, i):
    """1F1B 调度下，第 i 级（0 起）的操作序列。

    形状：前 `p-1-i` 个微批**只做前向**（warmup），此后**一前一后**交替（steady），
    最后把剩下的反向做完（cooldown）。★ 第 0 级的 warmup 最长、最后一级没有 warmup。
    """
    ops, f, b = [], 0, 0
    for _ in range(p - 1 - i):
        if f < m:
            ops.append(("F", f)); f += 1
    while b < m:
        if f < m:
            ops.append(("F", f)); f += 1
        ops.append(("B", b)); b += 1
    return ops


def sim_1f1b(p, m, tf=1.0, tb=1.0):
    """离散事件模拟：按上面的固定操作序列做**松弛迭代**，直到时间不再变。

    三条前驱约束：
      F(i,j) 要等 F(i-1,j) 与「第 i 级空出来」；
      B(i,j) 要等 F(i,j) 与 B(i+1,j) 与「第 i 级空出来」。
    """
    ops = [ops_1f1b(p, m, i) for i in range(p)]
    F_end = [[0.0] * m for _ in range(p)]
    B_end = [[0.0] * m for _ in range(p)]
    for _ in range(400):                       # 松弛；p、m 都是小整数，很快收敛
        for i in range(p):
            t = 0.0
            for kind, j in ops[i]:
                if kind == "F":
                    dep = F_end[i - 1][j] if i > 0 else 0.0
                    st = max(t, dep)
                    F_end[i][j] = st + tf
                    t = F_end[i][j]
                else:
                    dep = B_end[i + 1][j] if i < p - 1 else 0.0
                    st = max(t, F_end[i][j], dep)
                    B_end[i][j] = st + tb
                    t = B_end[i][j]
    return max(max(r) for r in B_end)          # 全部级别都做完才算完


print()
show("C0", "── C1：1F1B 的离散事件模拟 vs 闭式 ──")
show("C0", "口径：t_f = t_b = 1.0（无量纲时间单位）；闭式 = (m+p-1)(t_f+t_b)")
CASES = [(4, 4), (4, 8), (4, 16), (4, 32), (8, 16), (8, 64), (16, 64), (1, 8)]
print("  %6s %6s %14s %14s %12s" % ("p", "m", "模拟", "闭式", "差"))
for p, m in CASES:
    sim = sim_1f1b(p, m)
    closed = (m + p - 1) * 2.0
    print("  %6d %6d %14.6f %14.6f %12.3e" % (p, m, sim, closed, abs(sim - closed)))

show("C0", "★ p = 1（没有流水线）时：模拟与闭式相等，利用率应当是 1")

print()
show("C1", "── C2：1 − 利用率 与 (p-1)/(m+p-1) ──")
show("C1", "口径：利用率 = m(t_f+t_b) / makespan = m/(m+p-1)")
print("  %6s %6s %14s %14s %12s" % ("p", "m", "1-利用率", "(p-1)/(m+p-1)", "差"))
for p, m in CASES:
    sim = sim_1f1b(p, m)
    util = m * 2.0 / sim
    b_gpipe = (p - 1) / (m + p - 1)
    print("  %6d %6d %14.6f %14.6f %12.3e" % (p, m, 1 - util, b_gpipe, abs((1 - util) - b_gpipe)))

print()
show("C2", "── C3：**两个气泡口径的换算**（D17）──")
show("C2", "口径甲（GPipe）：b_g = (p-1)/(m+p-1)   口径乙（Megatron）：b_m = (p-1)/m")
show("C2", "★ 恒等式：b_m == b_g / (1 - b_g)")
print("  %6s %6s %14s %14s %14s %12s" % ("p", "m", "b_gpipe", "b_megatron", "b_g/(1-b_g)", "差"))
for p, m in CASES:
    b_g = (p - 1) / (m + p - 1)
    b_m = (p - 1) / m
    conv = b_g / (1 - b_g)
    print("  %6d %6d %14.6f %14.6f %14.6f %12.3e" % (p, m, b_g, b_m, conv, abs(b_m - conv)))

print()
show("C3", "── C4：`(n-d)/b'` 的换写（D18）──")
show("C3", "★ **本脚本对这一行的读法**：把 `(p-1)/m` 里的 `p-1` **换写**成 `n-d`、`m` **换写**成 `b'`")
show("C3", "  （即 p = n-d+1）⇒ 两边是同一个算术式；**这两个量各自的语义不在本脚本的范围内**")
show("C3", "  （语义归 `pipeline-parallel/01`，本探测只验算术）")
n, Bp = 64, 32
print("  %8s %18s %18s %10s" % ("d", "(n-d)/b'", "(p-1)/m", "相等?"))
for d in (2, 4, 8, 16, 32):
    lhs = (n - d) / Bp
    p = n - d + 1
    m = Bp
    rhs = (p - 1) / m
    print("  %8d %18.6f %18.6f %10s" % (d, lhs, rhs, "是" if lhs == rhs else "否"))

print()
show("C4", "── C5：interleaved 的通信倍数（D19）──")
show("C4", "★ 原文（Megatron-LM 的 interleaved 1F1B）说通信量是 **`v` 倍**（v = virtual stages）；")
show("C4", "  精确式是 `(p·v - 1)/(p - 1)` ⇒ 只有 p 很大时两者才接近")
print("  %8s %6s %20s %12s %12s" % ("p", "v", "(p*v-1)/(p-1)", "原文的 v", "相对误差"))
for p in (2, 4, 8, 16, 64, 256):
    for v in (2, 4):
        exact = (p * v - 1) / (p - 1)
        err = (exact - v) / v
        print("  %8d %6d %20.6f %12.6f %11.2f%%" % (p, v, exact, v, err * 100))

print()
show("C9", "★ 本脚本只打印、不断言；以上是**调度模型的结论**，不是任何实现的实测。")
show("C9", "★ p=1 那一行是**边界**：没有流水线时气泡为 0、利用率为 1。")
show("C9", "★ 不是 NCCL，也不是真机：这里连一次通信都没有发生。")

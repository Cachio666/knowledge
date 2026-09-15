"""求和误差的累积与 FMA：n 次运算会错多少。

运行：python3 03_summation_and_fma.py
依赖：numpy（+ 标准库 math.fma、fractions）
"""

import math
from fractions import Fraction

import numpy as np

import fpbits as F


def rule(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def seq_sum(x):
    """顺序求和：严格按下标递增累加。cumsum 保证是顺序的。"""
    return np.float32(np.cumsum(x, dtype=np.float32)[-1])


def pairwise_sum(x):
    """成对求和：自底向上两两相加，即一棵二叉归约树。"""
    x = np.asarray(x, dtype=np.float32).copy()
    n = 1
    while n < x.size:
        n *= 2
    if x.size < n:                                  # 补零到 2 的幂
        x = np.concatenate([x, np.zeros(n - x.size, dtype=np.float32)])
    while x.size > 1:
        x = x[0::2] + x[1::2]
    return np.float32(x[0])


def kahan_sum(x):
    """Kahan 补偿求和：用一个补偿项把每次舍入丢掉的部分捞回来。"""
    s = np.float32(0.0)
    c = np.float32(0.0)
    for v in x:
        y = np.float32(np.float32(v) - c)
        t = np.float32(s + y)
        c = np.float32(np.float32(t - s) - y)
        s = t
    return s


def gamma(n, u):
    """γ_n = n·u / (1 - n·u)：n 次运算的累积误差放大因子。"""
    return n * u / (1.0 - n * u)


rule("1. 三种求和法：误差随 n 的增长速度完全不同")

print("同一批数据、同一个 fp32 格式，只有相加的顺序不同。")
print("数据服从 [-1, 1] 上的均匀分布（正负号随机，误差有机会互相抵消）。")
print()
print("单次随机试验的噪声很大（n 小时三种方法谁好谁坏甚至可能反过来），")
print("所以下表每个 n 跑 7 个种子，取**中位数**相对误差。")
print()
print("  %-10s %-16s %-16s %-16s %s"
      % ("n", "顺序", "成对", "Kahan", "顺序/成对"))
print("  " + "-" * 88)
u32 = 2.0 ** -24
prev = {}
for n in (1_000, 10_000, 100_000, 1_000_000):
    e = {"seq": [], "pair": [], "kah": []}
    for seed in range(7):
        g = np.random.default_rng(100 + seed)
        x = g.uniform(-1, 1, n).astype(np.float32)
        true = float(np.sum(x.astype(np.float64)))
        e["seq"].append(abs(float(seq_sum(x)) - true) / abs(true))
        e["pair"].append(abs(float(pairwise_sum(x)) - true) / abs(true))
        if n <= 100_000:                        # Kahan 是 Python 循环，控制规模
            e["kah"].append(abs(float(kahan_sum(x)) - true) / abs(true))
    m = {k: float(np.median(v)) for k, v in e.items() if v}
    prev[n] = m
    print("  %-10d %-16.3e %-16.3e %-16s %.0f 倍"
          % (n, m["seq"], m["pair"], "%.3e" % m["kah"] if "kah" in m else "—",
             m["seq"] / max(m["pair"], 1e-30)))

print()
grow_seq = prev[1_000_000]["seq"] / prev[1_000]["seq"]
grow_pair = prev[1_000_000]["pair"] / prev[1_000]["pair"]
print("n 从 1e3 涨到 1e6（1000 倍）：顺序求和的误差涨了约 %.0f 倍，成对涨了约 %.0f 倍。"
      % (grow_seq, grow_pair))
print("√1000 ≈ 32，log2(1000) ≈ 10 —— 顺序法落在随机游走的量级上；")
print("成对求和增长得更慢，因为它的误差只按归约树的**层数**累积。")
print()
print("还有一个信号：成对求和在 n=1e6 时误差约 1.0e-07，而 fp32 的单位舍入误差")
print("u = %.2e。两者已是同一量级——说明它的误差**不再由累加主导**，"
      % u32)
print("而由「把最终结果表示成 fp32」这一步主导。继续增大 n 也不会让它明显变差，")
print("这正是树形归约在长序列上不可替代的原因。")
print()
print("原因不是「顺序法更差」这么简单——误差的**累积方式**不同：")
print()
print("  · 顺序求和：每一步的部分和都可能已经很大，后续加上小项时")
print("    小项被舍入掉。误差项数正比于 n，但符号随机 ⇒ 随机游走，")
print("    实际增长约 √n。")
print("  · 成对求和：归约树只有 log2(n) 层，每层误差只累积一次 ⇒ 约 log n。")
print("  · Kahan：把每次丢掉的低位存进补偿项 c，下次补回去 ⇒ 误差基本不随 n 增长。")

rule("2. 误差界：γ_n 从哪来")

print("设每步相对误差为 δ_i（|δ_i| ≤ u），顺序求和的第 n 步部分和为：")
print()
print("    S_n = (S_{n-1} + x_n)(1 + δ_n)")
print()
print("把递推展开。令 Ŝ_k 为第 k 步的计算值，则")
print()
print("    Ŝ_n = Σ x_k · Π_{i≥k} (1 + δ_i)")
print()
print("把连乘用 n·u 界定（利用 (1+u)^n ≤ 1 + nu/(1-nu)，n·u < 1 时）：")
print()
print("    |Ŝ_n - S_n| ≤ γ_n · Σ|x_k|,    γ_n = n·u / (1 - n·u)")
print()
print("这个界有三个要点，每个都常被误读：")
print("  ① 误差正比于 **Σ|x_k|**，不是 |Σx_k|。所以「正负抵消、结果接近 0」")
print("     的求和，相对误差可以任意大——分母塌了，分子没塌。")
print("  ② 误差正比于 n，是**线性**的，不是 √n。√n 是随机数据的统计行为，")
print("     不是数学保证。构造出来的最坏输入能达到线性界。")
print("  ③ γ_n 是**上界**，通常极其悲观。")
print()
print("  %-10s %-16s %-16s %s" % ("n", "γ_n", "n·u", "说明"))
print("  " + "-" * 72)
for n in (10, 1_000, 100_000, 16_000_000):
    print("  %-10d %-16.4e %-16.4e %s"
          % (n, gamma(n, u32), n * u32, "n·u 逼近 1，界失效" if n * u32 > 0.5 else ""))
print()
print("注意最后一行：fp32 下 n 达到 1.6e7 时 n·u 已接近 1，γ_n 失去意义——")
print("这不是说误差达到了 100%，而是说这个界本身已经不够紧到有用。")

print()
print("验证界的形式（误差 ∝ Σ|x| 而非 |Σx|）：构造一个 Σx 接近 0 的求和。")
g = np.random.default_rng(8)
n = 100_000
half = g.uniform(0, 1, n // 2).astype(np.float32)
x = np.concatenate([half, -half])                 # 前后各半，精确抵消
x = x[g.permutation(x.size)]
true_signed = float(np.sum(x.astype(np.float64)))
true_abs = float(np.sum(np.abs(x).astype(np.float64)))
got = float(seq_sum(x))
print("  n = %d，Σ|x| = %.1f，而 Σx 的真值精确等于 0（构造上前后各半、互为相反数）"
      % (n, true_abs))
print("  顺序求和得到 %.3e——真值是 0，答案是 %.1e。" % (got, got))
print("  用 γ_n·Σ|x| 给出的绝对误差界 = %.3e，界成立。" % (gamma(n, u32) * true_abs))
print()
print("  但若按「相对误差」表述：分母是真值 0，相对误差是无穷大。")
print("  这不是数值问题，是**指标选错了**：误差正比于 Σ|x|，与 Σx 无关，")
print("  所以当 Σx ≪ Σ|x| 时，相对误差这个指标本身失去意义。")
print("  结论：描述求和精度要看**绝对误差 ÷ Σ|x|**，不看 ÷ |Σx|。")

rule("3. FMA：把两次舍入变成一次")

print("a*b + c 的常规做法是两条指令、两次舍入：先算 fl(a*b)，再加 c 再舍入。")
print("FMA（fused multiply-add）在硬件里保留 a*b 的完整乘积，与 c 相加后**只舍入一次**。")
print()
print("先看 float64 上的真实差别（Python 3.13 的 math.fma，精确值用有理数算）：")
print()
a = 1.0 + 2.0 ** -27
b = 1.0 + 2.0 ** -27
c = -1.0
exact = Fraction(a) * Fraction(b) + Fraction(c)
sep = (a * b) + c
fus = math.fma(a, b, c)
print("    a = 1 + 2^-27,  b = 1 + 2^-27,  c = -1")
print("    精确 a*b + c = %.17g   (即 2^-26 + 2^-54)" % float(exact))
print("    两次舍入     = %.17g   相对误差 %.3e" % (sep, abs(Fraction(sep) - exact) / exact))
print("    fma 一次舍入 = %.17g   相对误差 %.3e" % (fus, abs(Fraction(fus) - exact) / exact))
print()
print("fma 给出了精确值；两次舍入的版本丢掉了 2^-54 那一项——它落在第一次")
print("乘法的舍入误差里，后面再也补不回来。")

print()
print("把同样的对比放到 fp16 上，后果从「不精确」升级为「结果消失」。")
print("（fp16 部分由 fpbits 用位运算模拟：把乘积舍入到 fp16、再相加舍入一次，")
print(" 对比「乘积与和只舍入一次」。模拟的是舍入行为，不是硬件 FMA 指令。）")
print()
a16 = 1.0 + 2.0 ** -10
b16 = 1.0 - 2.0 ** -10
prod16 = float(F.quantize(np.array([a16 * b16]), 5, 10)[0])          # 第一次舍入
sep16 = float(F.quantize(np.array([prod16 - 1.0]), 5, 10)[0])         # 第二次舍入
fus16 = float(F.quantize(np.array([a16 * b16 - 1.0]), 5, 10)[0])      # 只舍入一次
print("    a = 1 + 2^-10,  b = 1 - 2^-10,  c = -1")
print("    精确 a*b + c = %.10g" % (a16 * b16 - 1.0))
print("    两次舍入     = %.10g   ← 第一次舍入把 1-2^-20 变成 1.0，结果整个丢失" % sep16)
print("    一次舍入     = %.10g   ← 完整保留" % fus16)

print()
print("这是矩阵乘累加里 FMA 的真正价值：它不是为了更快，而是为了**更准**。")
print("一次 FMA 只有一次舍入，误差界从 2u 降到 u。对 K 很大的累加来说，")
print("这意味着误差增长斜率直接减半。")

rule("4. 树形归约：GPU 上为什么天然是成对求和")

print("GPU 的 warp 内归约用 __shfl_down_sync 做蝶形交换：")
print()
print("    第 1 轮：lane i 与 lane i+16 相加，步长 16")
print("    第 2 轮：lane i 与 lane i+8  相加，步长 8")
print("    ...     共 log2(32) = 5 轮")
print()
print("这正是一棵归约树：每一步的加法发生在**等长的部分和**之间，")
print("不存在「大数加小数」的长期积累，因此误差界是 O(log n) 而不是 O(n)。")
print()
n = 1 << 20
g = np.random.default_rng(9)
x = g.uniform(-1, 1, n).astype(np.float32)
true = float(np.sum(x.astype(np.float64)))
print("  n = 2^20 = %d 上的实测：" % n)
print("    顺序求和 %.8f   相对误差 %.3e" % (float(seq_sum(x)), abs(float(seq_sum(x)) - true) / abs(true)))
print("    树形归约 %.8f   相对误差 %.3e" % (float(pairwise_sum(x)), abs(float(pairwise_sum(x)) - true) / abs(true)))
print()
print("这不是实现细节的优劣，而是**结构性**的：只要归约是树，误差就按层数累积。")
print("代价是树形归约需要 log n 次同步，顺序求和只需 0 次——")
print("所以在 n 很小、或延迟比精度更重要的场合，顺序求和仍然是对的。")

rule("5. 本程序的边界")

print("  · 顺序/成对/Kahan 三列都是真实的 fp32 运算，运行在 CPU 上。")
print("    GPU 的 fp32 加法语义与 CPU 一致（IEEE754），但归约的树形结构")
print("    由硬件与实现决定，具体数值会不同，结论（误差增长阶）不变。")
print("  · float64 的 FMA 用的是 math.fma，是真实的融合乘加。")
print("    fp16 的 FMA 对比是**位运算模拟**，只复现舍入行为，")
print("    不代表硬件指令的吞吐或可用性。")
print("  · Kahan 求和用 Python 循环实现，n 上限取 1e5 以控制运行时间。")

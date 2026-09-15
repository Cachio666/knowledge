"""舍入模式与误差模型：一次运算会错多少。

运行：python3 02_rounding_and_error.py
依赖：numpy
"""

import numpy as np

MODE_NAMES = {
    "even": "就近舍入，取偶 (round-to-nearest-even)",
    "trunc": "向零舍入 (round-toward-zero)",
    "up": "向 +inf 舍入 (round-toward-+inf)",
    "down": "向 -inf 舍入 (round-toward--inf)",
}


def rule(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def round_to_p(x, p, mode="even"):
    """把标量 x 舍入到 p 位有效二进制位（p 含隐藏位）。

    x = m * 2^e，m ∈ [0.5, 1)。把 m 放大到 [2^(p-1), 2^p) 后取整，
    即得到尾数的整数值，再还原回浮点。
    """
    x = float(x)
    if x == 0.0 or not np.isfinite(x):
        return x
    sign = -1.0 if x < 0 else 1.0
    m, e = np.frexp(abs(x))
    scaled = m * 2.0 ** p                       # ∈ [2^(p-1), 2^p)
    # 定向舍入必须带符号判断：对负数，「向 +inf」是把绝对值往小取。
    if mode == "even":
        r = np.rint(scaled)                     # numpy 默认即 round-half-even
    elif mode == "trunc":
        r = np.floor(scaled)                    # 向零 = 截断绝对值
    elif mode == "up":                          # 向 +inf：结果不小于精确值
        r = np.ceil(scaled) if sign > 0 else np.floor(scaled)
    elif mode == "down":                        # 向 -inf：结果不大于精确值
        r = np.floor(scaled) if sign > 0 else np.ceil(scaled)
    else:
        raise ValueError(mode)
    return sign * float(r) * 2.0 ** (e - p)


rule("1. 四种舍入模式：同一个数，四个邻居")

print("待舍入的数是 1 + 2^-24。fp32 的尾数只有 23 位，这个数正好卡在")
print("1.0 与 1+2^-23 的正中间——两个候选邻居等距。")
print()
x = 1.0 + 2.0 ** -24
print("  x = 1 + 2^-24 = %.17f" % x)
print("  左邻居 1.0        = %.17f" % 1.0)
print("  右邻居 1+2^-23    = %.17f" % (1.0 + 2.0 ** -23))
print()
print("  %-42s %-22s %s" % ("舍入模式", "结果", "相对误差"))
print("  " + "-" * 78)
for mode in ("even", "trunc", "up", "down"):
    r = round_to_p(x, 24, mode)
    print("  %-42s %-22.17f %+.3e" % (MODE_NAMES[mode], r, (r - x) / x))

print()
print("「就近取偶」的规则：等距时选尾数最低位为 0 的那个。这里 1.0 的尾数是 0（偶），")
print("1+2^-23 的尾数是 1（奇），所以舍向 1.0。换个位置就舍向另一边：")
print()
for k in (1, 2, 3, 4):
    v = 1.0 + k * 2.0 ** -24
    r = round_to_p(v, 24, "even")
    tag = "等距，取偶" if k % 2 == 1 else "本身就精确"
    print("  1 + %d·2^-24 → %.17f   (%s)" % (k, r, tag))

print()
print("上面的正数例子里「向零」和「向 -inf」给出了相同结果——对正数两者本就等价。")
print("取负数才能把四种模式完全分开：")
print()
print("  %-42s %-22s %s" % ("舍入模式（作用于 -(1+2^-24)）", "结果", "偏离方向"))
print("  " + "-" * 78)
for mode in ("even", "trunc", "up", "down"):
    r = round_to_p(-x, 24, mode)
    print("  %-42s %-22.17f %s"
          % (MODE_NAMES[mode], r, "向 0" if abs(r) < x else "远离 0"))

print()
print("为什么不用「四舍五入」：它让 0.5 总是向上，长期累加会产生**系统性偏差**——")
print("大量舍入的期望不再等于真值，误差随 n 线性增长而不是随机抵消。")
print("就近取偶在统计上无偏，这是它成为 IEEE754 默认模式的原因。")
print()
print("（本程序用整数运算模拟舍入模式。真实硬件的舍入模式由指令的")
print(" 舍入修饰符决定，运行时切换需要显式设置，不在本程序范围内。）")

rule("2. 单位舍入误差 u：误差模型的地基")

print("任何一次「实数 → 浮点」的映射都能写成：")
print()
print("    fl(x) = x · (1 + δ),   |δ| ≤ u")
print()
print("其中 u = 2^-p，p 是有效精度位数。这个常数 u 叫**单位舍入误差**。")
print()
for name, p in (("fp16", 11), ("fp32", 24), ("fp64", 53)):
    u = 2.0 ** -p
    eps = 2.0 ** -(p - 1)
    print("  %-6s p=%-3d u = 2^-%-2d = %.4e      eps = 2^-%-2d = %.4e      eps/u = %.1f"
          % (name, p, p, u, p - 1, eps, eps / u))

print()
print("注意 u 与 eps 差 2 倍——上一节从「相对间距在 binade 内滑动 2 倍」解释过。")
print("误差模型用的是 **u**（误差上界），不是 eps。把两者混用会让误差估计错一倍，")
print("而这个错误在文献和代码注释里都极常见。")

print()
print("实测验证：对 200 万个随机 fp64 值舍入到 fp32，看 |δ| 的实际上确界。")
rng = np.random.default_rng(1)
mag = 10.0 ** rng.uniform(-20, 20, 2_000_000)
xr = (mag * rng.uniform(-1, 1, 2_000_000)).astype(np.float64)
xr = xr[xr != 0]
fl = xr.astype(np.float32).astype(np.float64)
delta = np.abs((fl - xr) / xr)
print("  实测 max|δ| = %.6e" % delta.max())
print("  fp32 的 u   = %.6e" % 2.0 ** -24)
print("  实测值 / u  = %.6f   （应 ≤ 1，且随机取样时非常接近 1）" % (delta.max() / 2.0 ** -24))

rule("3. 单次运算的误差模型")

print("IEEE754 规定基本运算的结果是**精确结果舍入一次**：")
print()
print("    fl(a ∘ b) = (a ∘ b)(1 + δ),  |δ| ≤ u    （∘ ∈ {+, -, ×, ÷}）")
print()
print("两个推论，后面每一节的误差界都从这里长出来：")
print("  · 计算值 = 精确值 × (1+δ)，即**相对误差有界**，且与运算本身无关。")
print("  · 但这是**单次**运算的界。n 次运算串起来，误差不必然线性累加，")
print("    也不必然互相抵消——取决于运算顺序（第三节展开）。")
print()
print("注意 a 与 b 本身可能已经是舍入过的。若 a、b 各自的输入误差为 δa、δb，")
print("则 fl(a+b) = (a+b)(1+δ) 中已经包含了它们——误差模型是**逐层复合**的。")
print("这正是浮点误差分析真正的难点所在。")

rule("4. 灾难性抵消：减法本身精确，但它放大已有误差")

print("先澄清一个常见误解：两个相近数相减，**不是减法产生了误差**。")
print("Sterbenz 引理：若 y/2 ≤ x ≤ 2y，则 x - y 在浮点里**精确无误差**。")
print()
rng = np.random.default_rng(2)
a = 1.0 + rng.random(1_000_000) * 0.5
b = 1.0 + rng.random(1_000_000) * 0.5
exact = a - b
got = (a.astype(np.float32) - b.astype(np.float32)).astype(np.float64)
exact_f32 = a.astype(np.float32).astype(np.float64) - b.astype(np.float32).astype(np.float64)
print("  Sterbenz 适用区间内 x-y 的浮点结果与「输入已舍入后的精确差」")
print("  之间的最大偏差：%.3e" % np.abs(got - exact_f32).max())
print("  即：减法这一步没有引入任何新误差。")

print()
print("那抵消为什么危险？因为它把**输入里已有的绝对误差**，除以一个很小的结果，")
print("放大成了巨大的**相对误差**。绝对误差不变，分母塌了。")
print()
print("经典例子：用 E[x²] - E[x]² 计算方差。这个公式在数学上恒等，")
print("但当均值远大于标准差时，它是浮点灾难。")
print()
print("下面三列用同一批样本、同一台机器，只有计算路径不同：")
print("  A 单趟 fp32   —— 真实生产中朴素写法的样子，累加也在 fp32")
print("  B 抵消隔离    —— 两个中间量先用 fp64 算到最准，只把结果舍入到 fp32 再相减")
print("  C 两趟 fp32   —— 先算均值，再算 Σ(x-μ)²")
print("B 列是关键：它把「累加误差」排除干净，剩下的误差**只能**来自抵消。")
print()
print("  %-8s %-14s %-14s %-14s %-12s %s"
      % ("μ", "A 单趟 fp32", "B 抵消隔离", "C 两趟 fp32", "真值 σ²", "A 的相对误差"))
print("  " + "-" * 88)
rows = []
for mu, sigma, n in ((1.0, 1.0, 100_000), (1.0e2, 1.0, 100_000),
                     (1.0e4, 1.0, 100_000), (1.0e6, 1.0, 100_000)):
    g = np.random.default_rng(3)
    xs = (mu + sigma * g.standard_normal(n)).astype(np.float32)

    # 顺序累加（cumsum 保证是顺序的），避免 numpy 的成对求和把问题掩盖掉
    s1 = np.float32(np.cumsum(xs, dtype=np.float32)[-1])
    s2 = np.float32(np.cumsum(xs * xs, dtype=np.float32)[-1])
    a_val = np.float32(np.float32(s2 / np.float32(n))
                       - np.float32(s1 / np.float32(n)) ** 2)

    xd = xs.astype(np.float64)                       # 用 fp64 把两个中间量算准
    ex2 = np.float32(np.mean(xd * xd))
    emu = np.float32(np.mean(xd))
    b_val = float(np.float32(ex2 - emu * emu))

    m32 = np.float32(np.mean(xd))
    c_val = float(np.mean((xs - m32) ** 2, dtype=np.float32))

    a_err = abs(float(a_val) - sigma ** 2) / sigma ** 2
    rows.append((mu, float(a_val), b_val, c_val, a_err))
    print("  %-8.0e %-14.5g %-14.5g %-14.5f %-12.1f %.3e"
          % (mu, float(a_val), b_val, c_val, sigma ** 2, a_err))

mu_last, a_last, _, _, a_last_err = rows[-1]
print()
print("三个读数，一个比一个更说明问题。")
print()
print("第一，A 列随 μ 增大迅速失真：μ=%.0e 时算出 %.4g，而真值是 %.1f——"
      % (mu_last, a_last, sigma ** 2))
print("差了 %.0f 倍。" % a_last_err)
print()
print("第二，**B 列在 μ≥1e4 时精确等于 0**。这一列的两个中间量是用 fp64 算到")
print("尽可能准之后才舍入到 fp32 的，也就是说：误差不是来自累加，不是来自")
print("精度不够——两个各自相当准确的数，只要在 fp32 里相减，方差就被整个吞掉。")
print("这是抵消最纯粹的形态：结果不是「不准」，而是**精确地等于错误的值**。")
print()
mu = 1.0e6
e2 = mu ** 2
print("第三，看数量级就明白为什么。μ=1e6 时 E[x²] ≈ %.3e，它在 fp32 里的" % e2)
print("ulp 是 %.3e；而待求的方差只有 σ² = 1——比一个 ulp 还小 %.0f 倍。"
      % (np.spacing(np.float32(e2)), np.spacing(np.float32(e2)) / 1.0))
print("真实信息全部落在最后一位以下，没有任何计算方法能把它捞回来。")
print()
print("唯一出路是**换公式**而不是换精度：C 列的两趟算法把减法提到前面——")
print("先减去均值再平方，被减的两个量都缩到了 σ 量级，抵消不再发生。")
print()
print("注：本表 A、B 两列的具体数值依赖随机样本，换种子会变；")
print("变的是具体数字，不变的是「随 μ 增大而崩坏」这个趋势。")

print()
print("对 LLM 的含义：LayerNorm / RMSNorm 就是在算方差。若用单趟公式，")
print("激活值均值一大（fp16/bf16 下很容易），方差会完全失真，")
print("归一化输出随之发散。这就是为什么生产级实现都用两趟或 Welford 算法。")

rule("5. 本程序的边界")

print("  · 舍入模式由整数运算模拟，不是硬件舍入模式的实测。")
print("  · 方差对比用的是 numpy 的 fp32 通路，未在 GPU 上运行；")
print("    GPU 的 fp32 语义与 CPU 一致（IEEE754），但归约顺序不同，")
print("    因此具体数值会随实现变化，结论（单趟公式失效）不变。")

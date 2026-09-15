"""IEEE754 格式探针：位宽如何同时决定精度与范围。

运行：python3 01_format_probe.py
依赖：numpy（bf16 / fp8 由 fpbits 用位运算模拟）
"""

import numpy as np

import fpbits as F

LOG10_2 = 0.30102999566398120


def rule(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


rule("1. 格式参数：位宽决定一切")

print("一个格式由两个数锁定：指数位宽 k、尾数位宽 m（另有 1 个符号位）。")
print("bias = 2^(k-1) - 1，规格化指数范围 [1-bias, emax]，有效精度 p = m+1。")
print("p 里那个 +1 是隐藏位：规格化后尾数首位恒为 1，不必存。")
print()
hdr = "%-10s %4s %4s %5s %11s %11s %12s %12s %10s"
print(hdr % ("格式", "指数", "尾数", "p", "eps", "u", "最小规格化数", "最大有限数", "十进制位"))
print("-" * 92)
for name, (eb, mb, var) in F.FORMATS.items():
    p = F.format_params(eb, mb, var)
    print(hdr % (name, eb, mb, p["p"], "%.3e" % p["eps"], "%.3e" % p["u"],
                 "%.3e" % p["tiny"], "%.3e" % p["max"], "%.2f" % (p["p"] * LOG10_2)))

print()
print("两条独立的因果链：")
print("  · 指数位宽 k 单独决定范围。emin = 1 - bias 决定最小规格化数，emax 决定最大有限数。")
print("  · 尾数位宽 m 单独决定精度。eps = 2^-m 是 1.0 处相邻两数之间的距离。")
print("两者互不干涉——这就是『指数管范围、尾数管精度』，也是 fp16 与 bf16 分野的根源。")

print()
print("范围跨度（用对数算，避免高动态范围格式在浮点里溢出）：")
for name, (eb, mb, var) in F.FORMATS.items():
    p = F.format_params(eb, mb, var)
    span = np.log10(p["max"]) - np.log10(p["tiny"])
    print("  %-10s %6.1f 个十进制数量级" % (name, span))

print()
print("注：fp8_e4m3 是 IEEE754 的**变体**，不遵守上面的指数规则。")
print("它把全 1 指数域征用为有效指数，于是 emax 比 bias 大 1、格式里没有无穷大；")
print("代价是全 1 尾数被征用为 NaN，最大有限数比公式少一档——")
print("若按标准规则算得 240，实际是 %.0f。" % F.format_params(4, 3, "e4m3")["max"])

rule("2. 浮点数是网格，不是数轴")

print("ulp(x) 定义为 x 与下一个可表示数之间的距离。它在**同一个 binade 内恒定**，")
print("（binade = [2^e, 2^(e+1))），跨过一个 2 的幂才翻倍。")
print()
print("fp32 在 [1, 2) 这一个 binade 内的实测：")
print()
print("  %-16s %-16s %-16s %s" % ("x", "ulp(x)", "ulp(x)/x", "对照"))
print("  " + "-" * 70)
below2 = float(np.nextafter(np.float32(2.0), np.float32(0)))
for v in (1.0, 1.25, 1.5, 1.75, below2):
    a = np.float32(v)
    u = np.nextafter(a, np.float32(np.inf)) - a
    note = ""
    if v == 1.0:
        note = "= eps = 2^-23（上确界）"
    elif v == below2:
        note = "= u = 2^-24（下确界）"
    print("  %-16.9g %-16.9g %-16.4e %s" % (v, u, u / v, note))

print()
print("**ulp 的绝对值只取决于 binade，与 x 在 binade 内的位置无关**；")
print("因此相对间距 ulp(x)/x 在同一 binade 内从 eps 单调滑到 u，比值恰好 2 倍。")
print("这正是 u = eps/2 的来历——u 是相对间距的下确界，eps 是上确界。")
print()
print("绝对误差与相对误差的走向相反：")
print("  1.0 量级的 fp32 数，ulp = %.3e" % np.spacing(np.float32(1.0)))
print("  1e6 量级的 fp32 数，ulp = %.3e —— 是前者的 2^19 倍"
      % np.spacing(np.float32(1e6)))
print("所以「先算差值再比较」的判断在大量级上会失效：两个 1e6 量级的 fp32 数")
print("只要相差不到 %.3e（半个 ulp），就会舍入成同一个数。"
      % (np.spacing(np.float32(1e6)) / 2))

rule("3. 次正规数：网格在 0 附近被迫变密")

eb, mb, _ = F.FORMATS["fp16"]
par = F.format_params(eb, mb)
print("fp16 的最小规格化数是 2^%d = %.3e。" % (par["emin"], par["tiny"]))
print("若无次正规数，0 与它之间就是一个 %.3e 宽的空洞——相对精度在这里断崖。" % par["tiny"])
print()
print("次正规数的做法：指数域锁定为全 0（代表 emin），隐藏位改为 0，")
print("于是尾数只剩 m 个有效位。空洞被切成 2^%d - 1 = %d 格。" % (mb, 2 ** mb - 1))
print("每格宽 2^(emin-m) = %.3e，是**恒定**的。" % par["min_sub"])
print()
print("恒定格宽 + 数值趋向 0 ⇒ 相对精度一路恶化：")
print()
print("  %-16s %-14s %-16s %s" % ("值 v", "格宽/ v", "单次舍入相对误差上界", "状态"))
print("  " + "-" * 74)
for shift in (0, 3, 6, 9, 10):
    v = 2.0 ** (par["emin"] - shift)
    spacing = par["min_sub"]
    rel = spacing / v
    tag = "与规格化数持平" if shift == 0 else ("只剩最低一位" if shift == 10 else "")
    print("  %-16.4e %-14.3e %-16.3e %s" % (v, rel, rel / 2, tag))

print()
print("规格化区间内 u = %.3e 恒定；跨入次正规后相对误差从 u 涨到 1/2，" % par["u"])
print("即**有效位从 %d 位一路掉到 1 位**。这就是次正规数的代价。" % par["p"])
print()
print("对 LLM 的含义：接近 0 的小量（激活的小分量、梯度尾部）一旦落入次正规区，")
print("相对误差就已经不可用；再往下越过 %.3e（0 与最小次正规数的中点），"
      % (par["min_sub"] / 2))
print("更是直接归零——梯度就此永久丢失，再也回不来。")
print("这就是 fp16 训练需要 loss scaling 的根源：不是「数值小所以不准」，")
print("而是**小数值的相对精度本就崩了，且存在一个硬归零下限**。")

rule("4. 特殊值")

print("指数域全 1 是保留组合，配合尾数域区分出三个非数：")
print()
print("  %-14s %-12s %-10s %s" % ("值", "fp32 位模式", "指数域", "回读结果"))
print("  " + "-" * 66)
for label, u, efield in [("+0", 0x00000000, "全 0"), ("-0", 0x80000000, "全 0"),
                         ("+inf", 0x7F800000, "全 1"), ("-inf", 0xFF800000, "全 1"),
                         ("quiet NaN", 0x7FC00000, "全 1"), ("signaling NaN", 0x7F800001, "全 1")]:
    v = float(np.array([u], dtype=np.uint32).view(np.float32)[0])
    shown = "%.1f" % v if np.isfinite(v) else ("nan" if np.isnan(v) else
                                               ("+inf" if v > 0 else "-inf"))
    print("  %-14s 0x%08X   %-10s %s" % (label, u, efield, shown))
print()
print("区分规则：指数域全 0 且尾数域非 0 → 次正规数；指数域全 1 时尾数域全 0")
print("→ 无穷大，尾数域非 0 → NaN。尾数域最高位为 1 的是 quiet NaN，为 0 的是")
print("signaling NaN——后者在参与运算时会触发无效操作异常。")

print()
print("-0.0 == 0.0 为 %s（IEEE754 规定它们相等），但 1/-0.0 = %s，"
      % (np.float32(-0.0) == np.float32(0.0), 1.0 / np.float32(-0.0)))
print("np.signbit 能区分二者。负数下溢到 0 时保留符号位，正是为了这个。")
print()
print("NaN 的唯一性质：NaN != NaN 为 %s。" % (np.float32(np.nan) != np.float32(np.nan)))
print("它是唯一不满足自反性的值，因此判空必须用 isnan。")
print("写 `if x != x` 依赖了这条性质，而它会被优化器当作恒假式消掉。")

rule("5. bf16 与 fp16：同一套规则下的两种取舍")

print("两个格式都占 16 位，但把位分给了不同的地方：")
print()
print("  %-8s %-6s %-6s %-4s %-14s %-10s %s" % ("格式", "指数", "尾数", "p", "最大有限数", "精度位数", "取向"))
print("  " + "-" * 70)
for name in ("fp16", "bf16"):
    e_, m_, var = F.FORMATS[name]
    p = F.format_params(e_, m_, var)
    print("  %-8s %-6d %-6d %-4d %-14.4e %-10.2f %s"
          % (name, e_, m_, p["p"], p["max"], p["p"] * LOG10_2,
             "要范围（指数对齐 fp32）" if e_ > 5 else "要精度"))

print()
print("bf16 的指数位宽与 fp32 相同 ⇒ **指数范围完全一致**；")
print("但尾数少 16 位 ⇒ 最大有限数比 fp32 小 %.1f%%。"
      % ((1 - F.format_params(8, 7)["max"] / F.format_params(8, 23)["max"]) * 100))
print("差别微小，工程上视作范围等价。真正付出的是精度：8 位 vs 24 位。")
print()
print("同一个数 1/3 在各格式下的样子：")
for name in ("fp16", "bf16", "fp32"):
    e_, m_, var = F.FORMATS[name]
    v = F.quantize(np.array([1.0 / 3.0], dtype=np.float32), e_, m_, var)[0]
    print("  %-6s %.10f   相对误差 %.3e" % (name, v, abs(float(v) - 1 / 3) / (1 / 3)))

print()
print("溢出边界的实测（fp16 最大有限数是 %.0f，向上第一个可表示值就是 inf）：" % 65504)
print()
print("  %-12s %-16s %s" % ("输入", "fp16", "bf16"))
print("  " + "-" * 48)
for v in (65504.0, 65519.0, 65520.0, 1.0e5, 1.0e30):
    a = F.quantize(np.array([v], dtype=np.float32), 5, 10)[0]
    b = F.quantize(np.array([v], dtype=np.float32), 8, 7)[0]
    sa = "%.1f" % a if np.isfinite(a) else repr(a)
    print("  %-12.6g %-16s %.4e" % (v, sa, b))

print()
print("65519 仍可表示（舍到 65504），65520 就溢出为 inf——中间只差 1。")
print("这正是「fp16 训练必须做 loss scaling」的直接原因：一旦激活或梯度超过")
print("65504，它不会饱和，而是变成 inf 并在后续运算中污染整条链路。")

rule("6. 本程序的边界")

print("bf16 与 fp8 的舍入由 fpbits.py 用位运算模拟，不是真实硬件：")
print("  · 复现的是「一个实数落到该格式网格上变成哪个数」。已用 fp16 的全部")
print("    65536 个位模式与 numpy 原生 float16 逐一比对，零差异。")
print("  · **不涉及**这些格式的吞吐、指令集、累加器实现或 fused 行为。")
print("  · tf32 无法这样模拟——它只在矩阵乘的输入端生效，没有存储格式。")

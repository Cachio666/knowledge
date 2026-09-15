"""低精度格式的工程后果：范围、精度、累加器宽度。

运行：python3 04_low_precision_formats.py
依赖：numpy（bf16 / fp8 / tf32 由 fpbits 用位运算模拟）
"""

import numpy as np

import fpbits as F


def rule(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def accumulate_rounded(x, ebits, mbits, variant="ieee"):
    """逐项累加，且**每次加法之后都把部分和舍入回目标格式**。

    这正是窄累加器的行为：部分和无法以高于累加器格式的精度暂存。
    """
    s = np.float32(0.0)
    for v in x:
        s = np.float32(F.quantize(np.array([np.float32(s) + np.float32(v)]),
                                  ebits, mbits, variant)[0])
    return float(s)


def dot_rounded(x, y, ebits, mbits):
    """模拟低精度点积：乘积舍入到目标格式，累加也在目标格式里。"""
    prods = F.quantize(x * y, ebits, mbits)
    return accumulate_rounded(prods, ebits, mbits)


rule("1. 低精度格式家族：指数与尾数之争")

print("下表把各格式放在同一把尺子上。『精度位数』= 有效二进制位数 p，")
print("『范围跨度』= log10(最大有限数 / 最小规格化数)。")
print()
print("  %-11s %-6s %-6s %-8s %-14s %-14s %s"
      % ("格式", "指数", "尾数", "精度位数", "最小规格化数", "最大有限数", "范围跨度"))
print("  " + "-" * 84)
for name, (eb, mb, var) in F.FORMATS.items():
    p = F.format_params(eb, mb, var)
    span = np.log10(p["max"]) - np.log10(p["tiny"])
    print("  %-11s %-6d %-6d %-8d %-14.3e %-14.3e %.1f 个数量级"
          % (name, eb, mb, p["p"], p["tiny"], p["max"], span))

print()
print("tf32 不在上表：它不是存储格式，只在矩阵乘的输入端把 fp32 的尾数截到 10 位，")
print("指数仍是 8 位。所以它的『精度』相当于 fp16，『范围』等同于 fp32。")
print()
print("选型的判据只有一条：**看数据的动态范围落在哪里**。")
print("  · 权重、激活：量级集中在 1e-3 ~ 1e2 —— fp16 的 9 个数量级足够，")
print("    且 fp16 多出的 3 位精度是真金白银。")
print("  · 梯度：量级可以到 1e-8 以下 —— 落在 fp16 的次正规区甚至归零，")
print("    这才是 fp16 训练需要 loss scaling 的原因。")
print("  · bf16 用 3 位精度换 67 个数量级，换来的是「不需要 loss scaling」。")

rule("2. 动态范围：溢出与下溢的实际后果")

print("真实网络里梯度跨越很多个数量级：靠近输出的层梯度大，靠近输入的层")
print("梯度可以小到 1e-10 以下。下面取量级 1e-10 ~ 1e0 共 20000 个样本。")
print()
g = np.random.default_rng(11)
grads = (10.0 ** g.uniform(-10, 0, 20_000) * g.choice([-1.0, 1.0], 20_000)).astype(np.float32)
print("  %-6s %-14s %-16s %-16s %s"
      % ("格式", "归零的个数", "落到次正规区", "中位相对误差（非零者）", "存活率"))
print("  " + "-" * 78)
for name, (eb, mb) in (("fp16", (5, 10)), ("bf16", (8, 7))):
    par = F.format_params(eb, mb)
    got = F.quantize(grads, eb, mb)
    nz = got != 0
    sub = (np.abs(got) < par["tiny"]) & nz
    rel = np.abs((got[nz] - grads[nz]) / grads[nz])
    print("  %-6s %-14d %-16d %-16.3e %.1f%%"
          % (name, int((~nz).sum()), int(sub.sum()), float(np.median(rel)),
             100.0 * nz.sum() / grads.size))

print()
p16 = F.format_params(5, 10)
pbf = F.format_params(8, 7)
print("fp16 的归零下限是 %.3e（0 与最小次正规数的中点）。低于它的梯度"
      % (p16["min_sub"] / 2))
print("不是「不准」，而是**直接变成 0**——这些权重从此再也收不到梯度信号，")
print("而且这个损失是不可逆的：0 乘任何数还是 0。")
print()
print("bf16 的归零下限是 %.3e，比 fp16 低 %.0f 个数量级，"
      % (pbf["min_sub"] / 2, np.log10(p16["min_sub"] / pbf["min_sub"])))
print("同一批数据里没有一个归零。这就是它不需要 loss scaling 的原因。")
print()
print("上溢方向的对称事实：fp16 的最大有限数是 %.0f，超过它不会饱和为最大值，"
      % p16["max"])
print("而是变成 inf，并在后续所有运算中污染整条链路。")

rule("3. 累加器宽度：为什么矩阵乘必须用 fp32 累加")

print("模拟一个线性层：权重 ~ N(0, 0.02)，输入 ~ N(0, 1)，K 为归约长度。")
print("乘积与每一步累加都舍入到目标格式，参考值用 fp64 精确计算。")
print()
print("先看**理论**怎么说。把每步的相对误差 δ 串起来：")
print()
print("    |误差| ≤ γ_K · Σ|x_k·y_k|,    γ_K = K·u / (1 - K·u)")
print()
print("这个界正比于 **Σ|x_k·y_k|**，不是正比于结果本身。而随机符号的数据里，")
print("部分和按 √K 增长、Σ|x·y| 按 K 增长——两者一比，相对误差应当按 **√K** 走。")
print("下表的实测验证了这个推论。每个 K 跑 21 个种子取中位数：")
print()
print("  %-8s %-14s %-14s %-14s %-12s %s"
      % ("K", "fp16 相对误差", "bf16 相对误差", "fp32 相对误差", "√K·u(fp16)", "实测/理论"))
print("  " + "-" * 84)
u16, ubf, u32 = 2.0 ** -11, 2.0 ** -8, 2.0 ** -24
k_ratios, ms = [], []
for K in (16, 64, 256, 1024, 4096, 16384):
    r16, rb, rf = [], [], []
    for seed in range(21):
        g = np.random.default_rng(9000 + seed)
        w = (0.02 * g.standard_normal(K)).astype(np.float32)
        a = g.standard_normal(K).astype(np.float32)
        true = float(np.dot(w.astype(np.float64), a.astype(np.float64)))
        if abs(true) < 1e-3:
            continue
        r16.append(abs(dot_rounded(w, a, 5, 10) - true) / abs(true))
        rb.append(abs(dot_rounded(w, a, 8, 7) - true) / abs(true))
        rf.append(abs(dot_rounded(w, a, 8, 23) - true) / abs(true))
    m16, mb, mf = float(np.median(r16)), float(np.median(rb)), float(np.median(rf))
    theory = np.sqrt(K) * u16
    k_ratios.append(m16 / theory)
    ms.append((K, m16, mb, mf))
    print("  %-8d %-14.3e %-14.3e %-14.3e %-12.2e %.2f"
          % (K, m16, mb, mf, theory, m16 / theory))

print()
print("三个读数：")
print()
k_lo, k_hi = ms[0][0], ms[-1][0]
print("第一，**相对误差按 √K 增长，不是按 K**。K 从 %d 涨到 %d（%.0f 倍），"
      % (k_lo, k_hi, k_hi / k_lo))
print("√K 涨 %.0f 倍，实测的 fp16 误差涨了 %.0f 倍——与 √K 一致，与 K 无关。"
      % (np.sqrt(k_hi / k_lo), ms[-1][1] / ms[0][1]))
print("最后一列的比值落在 %.2f ~ %.2f（中位数 %.2f），所以更严格的写法是"
      % (min(k_ratios), max(k_ratios), float(np.median(k_ratios))))
print("**误差 ≲ √K·u**——量级正确，常数因子在 1 以下。")
print()
print("第二，**fp16、bf16、fp32 三列遵循同一个规律，差别只在 u**。")
print("  u(fp16)/u(fp32) = %.0f 倍，而 K=%d 时实测误差之比是 %.0f 倍。"
      % (u16 / u32, ms[-1][0], ms[-1][1] / ms[-1][3]))
print("所以窄累加器的问题不是「有额外误差」，而是「误差的斜率被同一个倍数放大」。")
print()
print("第三，fp32 列停在 1e-7 ~ 1e-6，几乎不随 K 动。这不是因为它不受同一个规律")
print("支配，而是因为 u(fp32) 太小——√K·u(fp32) 在 K=%d 时才 %.1e，" % (ms[-1][0], np.sqrt(ms[-1][0]) * u32))
print("已经低于「把最终结果表示成 fp32」这一步自身的误差，累加不再是主导项。")
print()
print("结论仍然是「fp16/bf16 存、fp32 累加」，但理由要写对：")
print("**不是「K 一大就崩」，而是误差按 √K·u 增长，而 u 差 8192 倍。**")
print("K = 4096 时 fp16 累加的相对误差约 %.1f%%——单层尚可忍受，"
      % (100 * ms[4][1]))
print("但 Transformer 有几十上百层，误差逐层复合；而 attention 的 K 是序列长度，")
print("长上下文下可达 1e5 量级，√K·u 会冲到 15% 以上，那就真的不可用了。")
print()
print("现代 GPU 的 Tensor Core 正是按这个结论设计的——操作数可以是 fp16/bf16，")
print("累加器始终是 fp32，硬件层面就不给你用 fp16 累加的机会。")

rule("4. TF32：静默的精度截断")

print("tf32 的位宽是 19 位：符号 1 + 指数 8 + 尾数 10。它与 fp16 的尾数一样宽，")
print("但指数与 fp32 一样宽。因此在 fp32 代码里启用 tf32，**不需要改任何数据类型**，")
print("精度就已经掉了一半以上——没有类型签名会提示你。")
print()
print("  %-10s %-8s %-8s %-10s %s" % ("格式", "指数", "尾数", "精度位数", "相对 fp32 的精度"))
print("  " + "-" * 62)
for name, (eb, mb) in (("fp32", (8, 23)), ("tf32", (8, 10)),
                       ("bf16", (8, 7)), ("fp16", (5, 10))):
    p_ = F.format_params(eb, mb)
    print("  %-10s %-8d %-8d %-10d %.3f 倍" % (name, eb, mb, p_["p"], p_["p"] / 24))

print()
print("模拟 tf32 矩阵乘：输入截断到 tf32，累加仍在 fp32（这是硬件的实际做法）。")
print("注意此处的误差**只来自输入端截断**，与上一节的累加误差是两个独立来源。")
print("每个 K 跑 21 个种子取中位数：")
print()
print("  %-8s %-18s %-18s %s"
      % ("K", "fp32 相对误差（不截断）", "tf32 相对误差", "tf32 高出的数量级"))
print("  " + "-" * 74)
for K in (256, 1024, 4096):
    r32, rtf = [], []
    for seed in range(21):
        g = np.random.default_rng(7000 + seed)
        w = (0.02 * g.standard_normal(K)).astype(np.float32)
        a = g.standard_normal(K).astype(np.float32)
        true = float(np.dot(w.astype(np.float64), a.astype(np.float64)))
        if abs(true) < 1e-3:
            continue
        r32.append(abs(float(np.float32(np.dot(w, a))) - true) / abs(true))
        prod = F.quantize(w * a, 8, 10).astype(np.float64)   # tf32 截断乘积
        rtf.append(abs(float(np.sum(prod)) - true) / abs(true))
    m32, mtf = float(np.median(r32)), float(np.median(rtf))
    print("  %-8d %-18.3e %-18.3e %.1f"
          % (K, m32, mtf, np.log10(mtf / max(m32, 1e-30))))

print()
print("tf32 列比 fp32 列高约 3 个数量级，而代码里没有任何类型变化——")
print("`torch.backends.cuda.matmul.allow_tf32` 一开，所有 fp32 矩阵乘都受影响，")
print("包括那些你以为在跑「全精度」的路径。")
print()
print("对大多数推理负载这没关系：tf32 的精度仍高于 fp16 存储，")
print("而矩阵乘本身对精度不敏感。但对**精度敏感的累加**——")
print("如 long-context attention 的 softmax 分母、LayerNorm 的方差——")
print("必须显式关掉 tf32 或改走 fp32 通路。")

rule("5. 本程序的边界")

print("  · bf16 / fp8 / tf32 的运算全部由 fpbits 用位运算模拟，不是真实硬件。")
print("    模拟的是**舍入行为**，具体的误差量级可信；")
print("    但**不代表**这些格式的吞吐、指令集或 Tensor Core 的实际累加顺序。")
print("  · 真实 Tensor Core 的累加顺序是分块树形的，误差通常**优于**本程序")
print("    的逐项顺序累加。因此这里的数字应读作上限，不是实测值。")
print("  · fp32 累加一列是真实的 numpy fp32 运算，但它同样是顺序累加，")
print("    与 GPU 上的分块归约不同。")

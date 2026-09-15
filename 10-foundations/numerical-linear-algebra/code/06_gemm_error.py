"""点积与 GEMM 的误差：误差正比于 Σ|x_i·y_i|，不是正比于结果。

运行：python3 06_gemm_error.py
依赖：numpy
"""

import numpy as np


def rule(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def seq_dot32(x, y):
    """顺序点积，全程 fp32：乘一次、加一次，各舍入一次。"""
    s = np.float32(0.0)
    for a, b in zip(x, y):
        s = np.float32(s + np.float32(a) * np.float32(b))
    return float(s)


def seq_dot32_fast(x, y):
    """同上，但用 cumsum 做严格的顺序累加。数学上完全等价，只是快。"""
    z = (np.asarray(x, np.float32) * np.asarray(y, np.float32)).astype(np.float32)
    return float(np.cumsum(z, dtype=np.float32)[-1])


U32 = 2.0 ** -24


def gamma(n, u=U32):
    """γ_n = n·u / (1 - n·u)：n 次运算累积误差的放大因子。"""
    return n * u / (1.0 - n * u)


rule("1. 点积的误差界：γ_K 从哪来")

print("点积 s = Σ x_i·y_i 用浮点算，第 k 步的部分和满足：")
print()
print("    ŝ_k = fl(ŝ_{k-1} + fl(x_k·y_k))")
print()
print("把两个舍入都写成相对形式。乘积 fl(x_k·y_k) = x_k·y_k·(1+δ_k)，|δ_k| ≤ u；")
print("加法同理，再加上一个 (1+ε_k)。所以递推可以写成：")
print()
print("    ŝ_k = (ŝ_{k-1} + x_k·y_k·(1+δ_k))·(1+ε_k)")
print()
print("把 ŝ_K 按 k 展开。每一项 x_k·y_k 从第 k 步进来之后，要穿过后面每一步的")
print("加法舍入因子 (1+ε_j)，所以：")
print()
print("    ŝ_K = Σ_k x_k·y_k·(1+δ_k)·Π_{j≥k} (1+ε_j)")
print()
print("每项后面最多挂了 K 个 (1+u) 因子，用 (1+u)^K ≤ 1 + Ku/(1-Ku) 界定：")
print()
print("    |fl(xᵀy) - xᵀy| ≤ γ_K · Σ|x_k·y_k|,    γ_K = K·u/(1 - K·u)")
print()
print("三个要点，每一个都常被误读：")
print("  ① 界正比于 **Σ|x_k·y_k|**（绝对值之和），**不是** |xᵀy|。")
print("  ② 界随 K **线性**增长，是上界，通常极其悲观。")
print("  ③ 界与 x、y 的取值无关，只与 K 和 u 有关。")
print()
print("  %-12s %-16s %-16s %s" % ("K", "γ_K", "K·u", "说明"))
print("  " + "-" * 66)
for K in (64, 1024, 16384, 1_000_000, 16_000_000):
    note = "n·u 逼近 1，界失效" if K * U32 > 0.5 else ""
    print("  %-12d %-16.4e %-16.4e %s" % (K, gamma(K), K * U32, note))

rule("2. 核心结构：绝对误差 ∝ Σ|x·y|，与结果大小无关")

print("上一节的界说的是「绝对误差被 Σ|x·y| 卡住」。随机符号的数据上，这个界的")
print("量级是多少？下表每个 K 跑 21 个种子取中位数，测三件事：")
print()
print("    A = 绝对误差 / Σ|x·y|      （界的实际大小）")
print("    B = Σ|x·y| / |xᵀy|          （结果的「抵消程度」）")
print("    A·B = 相对误差              （两者相乘，是恒等式）")
print()
print("  %-8s %-12s %-12s %-12s %-12s %-12s %s"
      % ("K", "绝对误差", "Σ|x·y|", "A", "B", "A·B", "实测相对误差"))
print("  " + "-" * 90)
rows = []
for K in (256, 1024, 4096, 16384, 65536):
    ea, Sa, Ba, Ra = [], [], [], []
    for seed in range(21):
        g = np.random.default_rng(9000 + seed)
        w = (0.02 * g.standard_normal(K)).astype(np.float32)
        a = g.standard_normal(K).astype(np.float32)
        true = float(np.dot(w.astype(np.float64), a.astype(np.float64)))
        if abs(true) < 1e-3:
            continue
        S = float(np.sum(np.abs(w.astype(np.float64) * a.astype(np.float64))))
        d = abs(seq_dot32_fast(w, a) - true)
        ea.append(d)
        Sa.append(S)
        Ba.append(S / abs(true))
        Ra.append(d / abs(true))
    e, S, B, R = (float(np.median(ea)), float(np.median(Sa)),
                  float(np.median(Ba)), float(np.median(Ra)))
    rows.append({"K": K, "e": e, "S": S, "A": e / S, "B": B, "AB": (e / S) * B, "R": R})
    r = rows[-1]
    print("  %-8d %-12.3e %-12.3e %-12.3e %-12.3e %-12.3e %.3e"
          % (r["K"], r["e"], r["S"], r["A"], r["B"], r["AB"], r["R"]))

r0, r1 = rows[0], rows[-1]
print()
print("读这张表：")
print()
print("① **A 列几乎不随 K 变化**。K 涨了 %.0f 倍，绝对误差涨了 %.0f 倍，"
      % (r1["K"] / r0["K"], r1["e"] / r0["e"]))
print("   而 Σ|x·y| 也涨了 %.0f 倍——两者同步。于是 A = 误差/Σ|x·y| 只在"
      % (r1["S"] / r0["S"]))
print("   %.2e ~ %.2e 之间浮动（%.1f 倍），而 K 本身变了 %.0f 倍。"
      % (min(r["A"] for r in rows), max(r["A"] for r in rows),
         max(r["A"] for r in rows) / min(r["A"] for r in rows), r1["K"] / r0["K"]))
print("   绝对误差由 Σ|x·y| 和 u 决定，**K 不出现在里面**。")
print("   换算成 u：A/u = %s，即绝对误差约为 %.1f~%.1f 个 u 乘上 Σ|x·y|，"
      % (" / ".join("%.2f" % (r["A"] / U32) for r in rows),
         min(r["A"] / U32 for r in rows), max(r["A"] / U32 for r in rows)))
print("   中枢值 %.2f·u。" % np.median([r["A"] / U32 for r in rows]))
print()
print("② **B 列按 √K 增长，这是唯一的 K 依赖来源。**")
print()
print("   B 的量级可以解析算出来。记 X_i = x_i·y_i，取 x~N(0,1)、y~N(0,0.02²)：")
print("     E|X_i| = 0.02·(2/π) = %.5f   ⟹   Σ|X_i| ≈ %.5f·K"
      % (0.02 * 2 / np.pi, 0.02 * 2 / np.pi))
print("     Var(ΣX_i) = K·0.02²         ⟹   E|ΣX_i| ≈ 0.02·√(2/π)·√K = %.5f·√K"
      % (0.02 * np.sqrt(2 / np.pi)))
print("     相除：ρ = |ΣX|/Σ|X| ≈ %.4f/√K，取倒数得 B ≈ %.4f·√K"
      % (1.2533, 1.0 / 1.2533))
print()
print("   %-8s %-14s %-14s %s" % ("K", "B 实测", "B/√K", "理论 0.7980"))
print("   " + "-" * 52)
for r in rows:
    print("   %-8d %-14.3e %-14.3e %.4f" % (r["K"], r["B"], r["B"] / np.sqrt(r["K"]), 1 / 1.2533))
print()
print("   B/√K 落在 %.2f ~ %.2f，与理论值 %.4f 同量级。**B 就是 √K 的来源。**"
      % (min(r["B"] / np.sqrt(r["K"]) for r in rows),
         max(r["B"] / np.sqrt(r["K"]) for r in rows), 1 / 1.2533))
print()
print("③ 由此，`04_low_precision_formats.py` 里那条「累加相对误差 ∝ √K·u」的规律")
print("   可以拆成两个因子：")
print()
print("       相对误差 = A × B = (≈%.1f·u) × (≈%.2f·√K) ≈ %.2f·√K·u"
      % (np.median([r["A"] / U32 for r in rows]),
         np.median([r["B"] / np.sqrt(r["K"]) for r in rows]),
         np.median([r["A"] / U32 for r in rows]) * np.median([r["B"] / np.sqrt(r["K"]) for r in rows])))
print()
print("   实测的 A·B 列与「实测相对误差」列在每一行都只差 %.1f 倍以内——"
      % max(max(r["AB"] / r["R"], r["R"] / r["AB"]) for r in rows))
print("   两者本应严格相等（A·B 就是相对误差），差异只来自「中位数不可交换」：")
print("   表里报的是各自的中位数，乘积与中位数不是同一个运算。")
print()
print("   关键是因子 B：04 观察到的 √K **不是来自「累加次数变多」**，")
print("   而是来自 **分母**——随机符号的数据里，结果 |xᵀy| 自身只有")
print("   Σ|x·y| 的 1/(%.2f√K)。绝对误差其实几乎不随 K 变（A 列）。"
      % np.median([r["B"] / np.sqrt(r["K"]) for r in rows]))
print()
print("   这条拆分的用处是**定位**：要降低点积的相对误差，只有两条路——")
print("   压 A（换更宽的累加器，或做补偿求和），或压 B（改问题本身）。")

rule("3. 灾难性抵消：固定 Σ|x·y|，让结果缩到 0")

print("把第 2 节的结论推到极端：Σ|x·y| 不变而 xᵀy → 0 时，相对误差应当按 1/ρ 无界增长。")
print()
print("做一个受控实验。取 b_1…b_{K/2} 递减到很小，令")
print()
print("    x = [b, b]，y = [b, -b 的前 m 项取负再拼回剩余的 b]")
print()
print("于是 |x_i·y_i| = b_j² 与符号无关：**Σ|x·y| 在整个扫描中严格不变**。")
print("每改一次 m，只是把 U 形抵消的深度加深一点。K = %d，21 个种子取中位数。"
      % 1024)
print()
K = 1024
Hb = K // 2


def build(m, seed):
    g = np.random.default_rng(seed)
    b = 0.99 ** np.arange(Hb) * (1.0 + 0.05 * g.standard_normal(Hb))
    x = np.concatenate([b, b])
    y = np.concatenate([b, np.concatenate([-b[:m], b[m:]])])
    return x.astype(np.float32), y.astype(np.float32)


print("  %-6s %-12s %-13s %-13s %-13s %-12s %-12s %s"
      % ("m", "Σ|x·y|", "xᵀy 真值", "ρ", "相对误差", "A=err/Σ|xy|", "界", "界/实测"))
print("  " + "-" * 106)
frows = []
for m in (0, 64, 128, 256, 384, 448, 496, 511):
    rs, Ss, ts, bs = [], [], [], []
    for seed in range(21):
        x, y = build(m, 100 + seed)
        true = float(np.dot(x.astype(np.float64), y.astype(np.float64)))
        got = seq_dot32_fast(x, y)
        S = float(np.sum(np.abs(x.astype(np.float64) * y.astype(np.float64))))
        rs.append(abs(got - true) / abs(true))
        Ss.append(S)
        ts.append(true)
        bs.append(gamma(K) * S / abs(true))
    R, S, T, B = (float(np.median(rs)), float(np.median(Ss)),
                  float(np.median(ts)), float(np.median(bs)))
    rho = abs(T) / S
    frows.append({"m": m, "S": S, "T": T, "rho": rho, "R": R, "B": B})
    print("  %-6d %-12.4f %-13.4e %-13.4e %-13.4e %-12.3e %-12.3e %s"
          % (m, S, T, rho, R, R * rho, B,
             "松 %.0f 倍" % (B / R) if B > R else "★界被违反"))

f0, f1 = frows[0], frows[-1]
S_all = [r["S"] for r in frows]
print()
print("读这张表：")
print()
print("① **Σ|x·y| 一列严格不变**：8 行的值都是 %.4f，逐位相同。" % f0["S"])
print("   这是构造保证的——|x_i·y_i| = b_j² 与 y 的符号无关。")
print("   所以后面相对误差的变化，**只能**来自分母 |xᵀy| 在变小。")
print()
print("② **相对误差随 ρ 单调爆炸**。ρ 从 %.2f 降到 %.2e（跌了 %.0f 倍），"
      % (f0["rho"], f1["rho"], f0["rho"] / f1["rho"]))
print("   相对误差从 %.2e 涨到 %.2e（涨了 %.0f 倍）。两个倍数相差 %.1f 倍，"
      % (f0["R"], f1["R"], f1["R"] / f0["R"],
         (f0["rho"] / f1["rho"]) / (f1["R"] / f0["R"])))
print("   恰好就是下面 ③ 里 A 列的漂移量——误差的爆发几乎完全由分母驱动。")
print("   这就是「误差 ∝ Σ|x·y|，不 ∝ |xᵀy|」的直接后果：分子没动，分母塌了。")
print()
print("③ **A 列（误差/Σ|x·y|）几乎不动**：全文在 %.2e ~ %.2e 之间，"
      % (min(r["R"] * r["rho"] for r in frows), max(r["R"] * r["rho"] for r in frows)))
print("   只有 %.1f 倍的浮动，而 ρ 变了 %.0f 倍。"
      % (max(r["R"] * r["rho"] for r in frows) / min(r["R"] * r["rho"] for r in frows),
         f0["rho"] / f1["rho"]))
print("   这再次确认了第 2 节的结论：绝对误差只认 Σ|x·y| 和 u，不认结果大小。")
print()
print("④ **最后一列是界 γ_K·Σ|x·y|/|xᵀy| 与实测的比值**，%.0f ~ %.0f 倍。"
      % (min(r["B"] / r["R"] for r in frows), max(r["B"] / r["R"] for r in frows)))
print("   这个比值本身**也是稳定的**——因为 界/实测 = γ_K/A，")
print("   而 A 在整张表里几乎不动，γ_K 是个常数。")
print()
print("   所以对这个界要给出两句同样重要的评价：")
print("     · 它**从来没有被违反**（每一行都满足，最小余量 %.2e 倍）；"
      % min(r["B"] / r["R"] for r in frows))
print("     · 它**从来没有被逼近**（平均松 %.0f 倍，即 %.1f 个数量级）。"
      % (np.median([r["B"] / r["R"] for r in frows]),
         np.log10(np.median([r["B"] / r["R"] for r in frows]))))
print("   它正确预测了函数形式（∝ 1/ρ），但常数高估了 %.1f 个数量级。"
      % np.log10(np.median([r["B"] / r["R"] for r in frows])))
print("   这就是数值界的常态：**信它的形状，不要信它的数值**。")
print()
print("工程结论：点积的相对误差由「结果有多小」决定，而与「算得多辛苦」无关。")
print("任何形如 Σ a_i·b_i 且结果会抵消的路径——")
print("  · softmax 里减最大值之后的 exp 求和，")
print("  · LayerNorm 里 x - mean(x) 之后的平方和，")
print("  · attention score 的归一化、loss 的 baseline 相减——")
print("都会踩到这条。**指望「用 fp64 累加」来救它是错的**：")
print("fp64 只是把 A 从 0.2·u32 降到 0.2·u64，相对误差仍然正比于 1/ρ。")
print("真正的解法是改算法（减掉最大值、用 Welford 递推、做补偿求和），")
print("而不是换累加器。")

rule("4. GEMM：同一次矩阵乘里，不同元素的相对误差差很多")

print("GEMM C = A·B 的每一个元素都是一次长度为 K 的点积，所以界是**逐元素**的：")
print()
print("    |ΔC_ij| ≤ γ_K · (|A|·|B|)_ij")
print()
print("注意右边是 |A|·|B|（先取绝对值再相乘），不是 |A·B|。这意味着同一个 GEMM 的")
print("不同元素有各自的 ρ_ij = |C_ij| / (|A||B|)_ij，**它们可以差很多个数量级**。")
print()
m_, K_, n_ = 48, 512, 48
g = np.random.default_rng(4)
A = g.standard_normal((m_, K_)).astype(np.float32) * 0.5
B = (0.05 * g.standard_normal((K_, n_))).astype(np.float32)
C64 = A.astype(np.float64) @ B.astype(np.float64)
C32 = np.empty((m_, n_), np.float32)
for i in range(m_):
    for j in range(n_):
        C32[i, j] = seq_dot32(A[i], B[:, j])
AbsAB = np.abs(A.astype(np.float64)) @ np.abs(B.astype(np.float64))
dC = np.abs(C32.astype(np.float64) - C64)
bound = gamma(K_) * AbsAB
rel = dC / np.abs(C64)
rho = np.abs(C64) / AbsAB

print("  A: %d×%d，B: %d×%d，K = %d，共 %d 个元素。" % (m_, K_, K_, n_, K_, m_ * n_))
print("  γ_K = %.3e" % gamma(K_))
print()
print("  %-34s %s" % ("量", "数值"))
print("  " + "-" * 62)
print("  %-34s %.3e" % ("max |ΔC|（绝对误差）", dC.max()))
print("  %-34s %.3e" % ("max γ_K·(|A||B|)（界）", bound.max()))
print("  %-34s %.1f" % ("界 / 实际最大误差（松多少倍）", bound.max() / dC.max()))
print("  %-34s %d / %d" % ("界被违反的元素个数", int((dC > bound).sum()), m_ * n_))
print()
print("  %-12s %-16s %-16s %s" % ("", "相对误差", "同一元素的 ρ", "该元素 C_ij"))
print("  " + "-" * 62)
imin = np.unravel_index(np.argmin(rel), rel.shape)
imed = np.unravel_index(np.argmin(np.abs(rel - np.median(rel))), rel.shape)
imax = np.unravel_index(np.argmax(rel), rel.shape)
for name, idx in (("最小", imin), ("中位", imed), ("最大", imax)):
    print("  %-12s %-16.3e %-16.3e %.3e"
          % (name, rel[idx], rho[idx], C64[idx]))

print()
xr, yr = np.log(rho.ravel()), np.log(rel.ravel())
slope_rel = np.polyfit(xr, yr, 1)[0]
corr_rel = np.corrcoef(xr, yr)[0, 1]
Amp = dC / AbsAB
za = np.log(Amp.ravel())
slope_A = np.polyfit(xr, za, 1)[0]
corr_A = np.corrcoef(xr, za)[0, 1]
print()
print("同一份 A、B，同一个 K：最大与最小相对误差相差 %.0f 倍（%.1f 个数量级）。"
      % (rel.max() / rel.min(), np.log10(rel.max() / rel.min())))
print("  相对误差最大的元素 ρ = %.2e（抵消最狠，C = %.2e），" % (rho[imax], C64[imax]))
print("  相对误差最小的元素 ρ = %.2e（几乎不抵消，C = %.2e）。" % (rho[imin], C64[imin]))
print()
print("但**不要**把这条关系说成「相对误差 ∝ 1/ρ」。对 2304 个元素做 log-log 回归：")
print()
print("  %-52s %s" % ("回归 / 相关系数（对 log ρ）", "数值"))
print("  " + "-" * 66)
print("  %-52s %+.4f" % ("log(相对误差) 对 log ρ 的斜率", slope_rel))
print("  %-52s %+.4f" % ("log(相对误差) 与 log ρ 的相关系数", corr_rel))
print("  %-52s %+.4f" % ("log(A) 对 log ρ 的斜率", slope_A))
print("  %-52s %+.4f" % ("log(A) 与 log ρ 的相关系数", corr_A))
print()
print("斜率是 %+.2f 而不是 -1.0，相关系数是 %+.2f 而不是 -1。为什么？" % (slope_rel, corr_rel))
print()
print("因为 A_ij = |ΔC_ij| / (|A||B|)_ij 在这个矩阵里**不是常数**——它和 ρ 有 %+.2f 的"
      % corr_A)
print("弱正相关。原因是前面提过的：顺序求和的舍入误差 ∝ 每一步的**部分和**，")
print("而部分和同样会因为抵消而变小。抵消得越狠，分子也跟着小一点，")
print("所以相对误差的爆发比纯 1/ρ 略缓。")
print()
print("换句话说，「误差 ∝ Σ|x·y|」是**上界**意义上的严格陈述，")
print("不是等式。实测里 A 会随抵消程度漂移，抵消越深漂得越小。")
print("但漂移是 %.2f 次方量级的修正，而 ρ 本身跨了 %.1f 个数量级——"
      % (abs(slope_A), np.log10(rho.max() / rho.min())))
print("所以主导项仍然是 1/ρ。")
print()
print("相对误差中位数 %.2e 对应 A/u 中位数 %.2f，与第 2 节的 0.2~0.3 一致。"
      % (np.median(rel), np.median(Amp) / U32))
print()
print("这就是 GEMM 误差的完整图像：")
print("  · **绝对**误差的界对所有元素统一，只由 K、u、(|A||B|)_ij 决定；")
print("  · **相对**误差逐元素变化，因为分母 |C_ij| 逐元素变化，")
print("    实测 log-log 斜率 %+.2f，主导项就是 1/ρ；" % slope_rel)
print("  · 所以「这个 GEMM 的精度是多少」这个问题**没有全局答案**——")
print("    必须问「哪些输出元素是小量」。量化、稀疏化、注意力里真正吃亏的，")
print("    永远是那些输出接近 0 的位置，而不是平均值。")

rule("5. 本程序的边界")

print("  · 全部是 numpy 在 CPU 上的真实 fp32 / fp64 运算。")
print("  · 顺序点积由 Python 循环（小规模）或 np.cumsum（大规模）实现，")
print("    两者都是**严格按下标递增**的累加，与 GPU 上分块的树形归约不同。")
print("    真实 Tensor Core 的归约是分块的，误差通常**优于**本程序，")
print("    因此这里的绝对误差应读作顺序累加这一实现的上限。")
print("  · 第 2 节的理论系数 1.2533 = E|ΣX|/E Σ|X| 的推导假设 x、y 独立且")
print("    均值为 0（高斯或对称分布）。数据相关或均值非零时 B 的常数会变，")
print("    但 √K 这个增长阶来自中心极限定理，不依赖具体分布。")
print("  · 第 3 节的 Σ|x·y| 在每个种子内严格不变，跨种子因 b 的随机扰动而略有浮动，")
print("    表里报的是 21 个种子的中位数。")
print("  · 第 4 节的 C32 是逐元素 Python 循环算的（48×48×512），不是 BLAS。")

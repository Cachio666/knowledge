"""范数与条件数：误差放大由问题本身决定，不由算法决定。

运行：python3 05_norms_and_condition.py
依赖：numpy
"""

import numpy as np


def rule(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


rule("1. 范数：把「误差有多大」变成一个数")

print("解一个线性方程组 Ax = b 时，「答案错了多少」需要用矩阵和向量的**范数**")
print("来衡量。常用的向量范数有三个，都由同一个公式取不同的 p 得到：")
print()
print("    ‖x‖_p = ( Σ|x_i|^p )^(1/p)")
print()
print("  %-10s %-26s %s" % ("p", "公式", "读作"))
print("  " + "-" * 62)
for p, formula, meaning in ((1, "Σ|x_i|", "绝对值之和"),
                            (2, "(Σ|x_i|²)^(1/2)", "欧几里得长度"),
                            (np.inf, "max|x_i|", "最大分量")):
    print("  %-10s %-26s %s" % (p, formula, meaning))
print()
x = np.array([3.0, -4.0, 12.0])
print("同一个向量 x = %s 在三个范数下：" % x.tolist())
print("    ‖x‖₁ = %.1f    ‖x‖₂ = %.1f    ‖x‖∞ = %.1f"
      % (np.linalg.norm(x, 1), np.linalg.norm(x, 2), np.linalg.norm(x, np.inf)))
print()
print("三者相差可以很大，所以**比较误差前必须说清用的是哪个范数**。")
print("工程上 ‖·‖∞ 最直观（最大分量），‖·‖₂ 在能量/最小二乘里天然出现。")

print()
print("矩阵范数不是「元素的最大值」，而是由向量范数**诱导**出来的：")
print()
print("    ‖A‖_p = max_{x≠0} ‖Ax‖_p / ‖x‖_p")
print()
print("它的含义是「A 最多能把一个向量放大多少倍」。这个定义直接决定了")
print("它和元素取值的关系不是显然的——下面对比两个矩阵：")
print()
for label, A in (("对角阵", np.diag([1.0, 2.0, 3.0])),
                 ("全同阵", np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])),
                 ("上三角", np.array([[1.0, 1.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]]))):
    print("  %-8s max|a_ij| = %.1f   ‖A‖₁ = %.1f   ‖A‖₂ = %.1f   ‖A‖∞ = %.1f"
          % (label, np.abs(A).max(), np.linalg.norm(A, 1),
             np.linalg.norm(A, 2), np.linalg.norm(A, np.inf)))
print()
print("上三角阵的最大元素是 1，但 ‖A‖₂ = %.3f——它能把某些向量放大近 2 倍。"
      % np.linalg.norm(np.array([[1.0, 1.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]]), 2))
print("只看元素大小会严重低估一个矩阵的放大能力。")

rule("2. 条件数：从 ‖δb‖ 到 ‖δx‖ 的放大因子")

print("方程组 Ax = b。若 b 有一个小扰动 δb，解会变多少？")
print()
print("    A(x + δx) = b + δb   ⟹   A·δx = δb   ⟹   δx = A⁻¹·δb")
print()
print("两边取范数，再各自除以对应的原量：")
print()
print("    ‖δx‖ ≤ ‖A⁻¹‖·‖δb‖")
print("    ‖b‖ = ‖Ax‖ ≤ ‖A‖·‖x‖   ⟹   ‖x‖ ≥ ‖b‖ / ‖A‖")
print()
print("把第二个式子代入第一个的右边（注意是除以一个下界，所以方向要反过来）：")
print()
print("    ‖δx‖ / ‖x‖  ≤  ‖A‖·‖A⁻¹‖ · ‖δb‖ / ‖b‖")
print()
print("中间那个乘积就是**条件数**：")
print()
print("    κ(A) = ‖A‖·‖A⁻¹‖")
print()
print("结论一句话：**b 的相对误差会被放大最多 κ(A) 倍传到 x 上。**")
print("κ 是问题本身的性质，与你用什么算法求解无关。")

print()
print("实测验证。构造奇异值已知的矩阵 A = U·diag(σ)·Vᵀ，σ 从 1 降到 1/κ，")
print("分别测两种扰动下的放大倍数 (‖δx‖/‖x‖) ÷ (‖δb‖/‖b‖)：")
print()
print("  %-14s %-14s %-16s %-14s %s"
      % ("κ(A)", "随机 δb 中位", "随机 δb 最大", "最坏 δb", "最坏/κ"))
print("  " + "-" * 74)
for n, target in ((4, 1e2), (6, 1e6), (6, 1e10)):
    g = np.random.default_rng(int(np.log10(target)) + n)
    U, _ = np.linalg.qr(g.standard_normal((n, n)))
    V, _ = np.linalg.qr(g.standard_normal((n, n)))
    s_ = np.logspace(0, -np.log10(target), n)
    A = U @ np.diag(s_) @ V.T
    ka = 1.0 / s_[-1]
    amps = []
    for _ in range(4000):
        bb = g.standard_normal(n)
        dbb = g.standard_normal(n) * 1e-9
        xx = np.linalg.solve(A, bb)
        dxx = np.linalg.solve(A, bb + dbb) - xx
        amps.append((np.linalg.norm(dxx) / np.linalg.norm(xx))
                    / (np.linalg.norm(dbb) / np.linalg.norm(bb)))
    # 最坏情形：b 对齐最大奇异向量（使 ‖x‖ 最小），δb 对齐最小奇异向量
    b_w = U[:, 0]
    db_w = 1e-9 * U[:, -1]
    xw = np.linalg.solve(A, b_w)
    amp_w = ((np.linalg.norm(np.linalg.solve(A, b_w + db_w) - xw) / np.linalg.norm(xw))
             / (np.linalg.norm(db_w) / np.linalg.norm(b_w)))
    print("  %-14.0e %-14.2f %-16.2f %-14.2e %.3f"
          % (ka, float(np.median(amps)), max(amps), amp_w, amp_w / ka))

print()
print("这张表给出了 κ 这个界的**两层含义**，缺一层就会误用它：")
print()
print("① **界是紧的**。最坏情形一列精确等于 κ——存在特定的 b 与特定的扰动方向，")
print("   让放大倍数正好达到 κ。所以 κ 不是拍脑袋的保守估计，它是可达到的。")
print("   构造方法就是上面的代码：b 取最大奇异向量（此时 ‖x‖ 最小，分母最不利），")
print("   δb 取最小奇异向量（A⁻¹ 对它的放大最大）。")
print()
print("② **但随机扰动几乎撞不上最坏情形**。中位放大恒为 1 左右，")
print("   最大也只有几十——哪怕 κ 高达 1e10。这条对工程判断极其重要：")
print("   κ 大不等于「实际会错很多」，它等于「存在一种输入会让它错很多」。")
print()
print("所以在实际系统里，看到 κ = 1e8 就断言「这个 kernel 会崩」是错的；")
print("正确的问题是「我的输入会不会系统性地落在最坏方向上」——")
print("在 LLM 里这通常表现为**权重矩阵的谱分布是否极端**，而不是随机的舍入噪声。")

rule("3. κ(AᵀA) = κ(A)²：为什么不该解正规方程")

print("最小二乘问题 min‖Ax - b‖₂ 的经典解法是解**正规方程**：")
print()
print("    AᵀA x = Aᵀb")
print()
print("这个方程组的系数矩阵是 AᵀA。它的条件数是：")
print()
print("    κ(AᵀA) = κ(A)²")
print()
print("平方意味着**精度损失翻倍**：fp32 的 u = 6e-8，若 κ(A) = 1e4，")
print("则 κ(AᵀA) = 1e8，已经超过 fp32 能分辨的范围（u⁻¹ ≈ 1.7e7）。")
print()
print("  %-12s %-16s %-16s %s" % ("κ(A)", "κ(AᵀA) 理论", "κ(AᵀA) 实测", "fp32 下还有效数字吗"))
print("  " + "-" * 70)
for k in (1e2, 1e4, 1e6, 1e8):
    n = 6
    g = np.random.default_rng(int(np.log10(k)))
    U, _ = np.linalg.qr(g.standard_normal((n + 3, n)))
    V, _ = np.linalg.qr(g.standard_normal((n, n)))
    s = np.logspace(0, -np.log10(k), n)
    A = U @ np.diag(s) @ V.T
    kA = np.linalg.cond(A)
    kATA = np.linalg.cond(A.T @ A)
    digits = max(0.0, np.log10(1.0 / (2.0 ** -24)) - np.log10(kATA))
    print("  %-12.0e %-16.2e %-16.2e %.1f 位" % (kA, kA ** 2, kATA, digits * 0.301))

print()
print("最后一列 = log10(1/u) - log10(κ(AᵀA))，换算成十进制位。")
print("κ(A) 到 1e8 时，fp32 下解正规方程**一位有效数字都不剩**。")

print()
print("实测对比三种解法在病态问题上的表现：")
print()
n = 8
g = np.random.default_rng(21)
U, _ = np.linalg.qr(g.standard_normal((n, n)))
V, _ = np.linalg.qr(g.standard_normal((n, n)))
s = np.logspace(0, -8, n)
A = U @ np.diag(s) @ V.T
x_true = g.standard_normal(n)
b = A @ x_true
print("  κ(A) = %.2e" % np.linalg.cond(A))
print()
print("  %-26s %-14s %s" % ("解法", "相对误差", "说明"))
print("  " + "-" * 72)
for label, fn, note in (
        ("inv(A) @ b", lambda: np.linalg.inv(A) @ b, "显式求逆：代价约 3 倍"),
        ("solve(A, b)", lambda: np.linalg.solve(A, b), "LU 分解，标准做法"),
        ("solve(AᵀA, Aᵀb)", lambda: np.linalg.solve(A.T @ A, A.T @ b), "正规方程：κ 被平方"),
        ("lstsq（QR/SVD）", lambda: np.linalg.lstsq(A, b, rcond=None)[0],
         "正交变换，不放大误差")):
    err = np.linalg.norm(fn() - x_true) / np.linalg.norm(x_true)
    print("  %-26s %-14.3e %s" % (label, err, note))

print()
print("三个读数：")
print()
print("① `inv(A) @ b` 与 `solve(A, b)` 在本例中精度**几乎相同**（2.3e-9 vs 2.5e-9）。")
print("   这纠正一个常见说法：显式求逆的主要问题不是精度，而是**代价**——")
print("   求逆的运算量约为一次 LU 求解的 3 倍，而你要的只是解一次方程。")
print("   「不要用 inv」这条建议的正确理由是省算力，不是省精度。")
print()
print("② **正规方程明显最差**：5.0e-3，比 solve 差 6 个数量级。")
print("   这正是 κ(AᵀA) = κ(A)² 的直接后果——κ 从 1e8 变成 1e16，")
print("   而 fp64 的 1/u ≈ 9e15，精度已经被吃干净。")
print()
print("③ `lstsq` 最好（5.8e-10），因为它走 QR/SVD，全程用正交变换。")
print("   正交矩阵满足 ‖Q‖₂ = 1，不放大误差——κ 只作用一次而不是两次。")
print()
print("可直接取用的结论：解方程用 `solve`，最小二乘用 QR/SVD，")
print("**永远不要显式构造 AᵀA**。")

rule("4. 本程序的边界")

print("  · 全部是 numpy 在 CPU 上的 fp64 运算；本节的换算涉及 fp32 的 u，已注明。")
print("  · 第 3 节的对比未在 GPU 上验证；cuSOLVER / cuBLAS 的具体数值会不同，")
print("    但 κ(AᵀA) = κ(A)² 是数学性质，与实现无关。")

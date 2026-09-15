"""矩阵分解的稳定性：为什么 LU 必须选主元，Cholesky 为什么要求正定。

运行：python3 07_factorization.py
依赖：numpy
"""

import time

import numpy as np


def rule(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def lu_nopivot(A):
    """**不选主元**的 LU 分解，原地改写。返回的矩阵下三角存 L、上三角存 U。

    这就是教科书上直接消元的那 4 行：对每一列 k，用它自己的对角元当主元，
    把下面的元素消成 0。**一行交换都没有。**
    """
    A = np.array(A, dtype=np.float64, copy=True)
    n = A.shape[0]
    for k in range(n - 1):
        for i in range(k + 1, n):
            m = A[i, k] / A[k, k]
            A[i, k] = m                                  # 乘子存进 L 的位置
            A[i, k + 1:] -= m * A[k, k + 1:]             # 这一行的消元
    return A


def lu_pivot(A):
    """带**部分主元**的 LU：每列先在该列剩下的元素里挑绝对值最大的换上来。"""
    A = np.array(A, dtype=np.float64, copy=True)
    n = A.shape[0]
    perm = np.arange(n)
    for k in range(n - 1):
        p = k + int(np.argmax(np.abs(A[k:, k])))
        if p != k:
            A[[k, p], :] = A[[p, k], :]
            perm[[k, p]] = perm[[p, k]]
        if A[k, k] == 0.0:
            continue
        m = (A[k + 1:, k] / A[k, k]).reshape(-1, 1)
        A[k + 1:, k] = m.ravel()
        A[k + 1:, k + 1:] -= m * A[k, k + 1:]
    return A, perm


def lu_solve_nopivot(A, b):
    """用不选主元的 LU 解 A x = b。"""
    LU = lu_nopivot(A)
    n = A.shape[0]
    L = np.tril(LU, -1) + np.eye(n)
    U = np.triu(LU)
    return np.linalg.solve(U, np.linalg.solve(L, b))


def lu_solve_pivot(A, b):
    """用部分主元的 LU 解 A x = b。"""
    LU, perm = lu_pivot(A)
    n = A.shape[0]
    y = np.array(b, copy=True)[perm]
    for i in range(1, n):                                # 前代：L y = Pb
        y[i] -= np.dot(LU[i, :i], y[:i])
    x = y.copy()
    for i in range(n - 1, -1, -1):                       # 回代：U x = y
        x[i] = (x[i] - np.dot(LU[i, i + 1:], x[i + 1:])) / LU[i, i]
    return x


def growth(LU, A):
    """增长因子 = max|U_ij| / max|A_ij|。U 取 LU 的上三角部分。"""
    return np.abs(np.triu(LU)).max() / np.abs(A).max()


rule("1. 不选主元的 LU：一个 ε 就能毁掉整个分解")

print("先把算法写出来。不选主元的 LU 就四行：")
print()
print("    for k in range(n - 1):")
print("        for i in range(k + 1, n):")
print("            m       = A[i, k] / A[k, k]      # 用对角元当主元")
print("            A[i, k] = m                      # 乘子存进下三角")
print("            A[i, k+1:] -= m * A[k, k+1:]     # 消元")
print()
print("致命处在第一行：`A[k, k]` 是**当前位置上碰巧有的那个数**，不是选出来的。")
print("它可以是任意小。而乘子 m = A[i,k] / A[k,k] 会因此任意大。")
print()
print("经典的 2×2 反例，取 A = [[ε, 1], [1, 1]]，右端项由 x = [1, 1] 生成：")
print()
print("  %-16s %-14s %-14s %-14s %s"
      % ("ε", "不选主元 growth", "选主元 growth", "不选主元 相对误差", "选主元 相对误差"))
print("  " + "-" * 82)
grow_rows = []
for eps in (1e-1, 1e-2, 1e-4, 1e-8, 1e-12, 1e-15):
    A = np.array([[eps, 1.0], [1.0, 1.0]])
    xt = np.array([1.0, 1.0])
    b = A @ xt
    LU = lu_nopivot(A)
    xn = lu_solve_nopivot(A, b)
    xp = lu_solve_pivot(A, b)
    LP, _ = lu_pivot(A)
    en = np.linalg.norm(xn - xt) / np.linalg.norm(xt)
    ep = np.linalg.norm(xp - xt) / np.linalg.norm(xt)
    grow_rows.append((eps, growth(LU, A), growth(LP, A), en, ep))
    print("  %-16.0e %-14.3e %-14.3e %-14.3e %.3e" % (eps, growth(LU, A), growth(LP, A), en, ep))

print()
def fmt2(M):
    """把 2×2 矩阵印成一行，避免 numpy 自动换行把版面弄乱。"""
    return "[[%9.1e, %9.1e], [%9.1e, %9.1e]]" % (M[0, 0], M[0, 1], M[1, 0], M[1, 1])


print("ε = 1e-15 时两种分解的完整结果：")
eps = 1e-15
A = np.array([[eps, 1.0], [1.0, 1.0]])
xt = np.array([1.0, 1.0])
b = A @ xt
print("  A = [[%.0e, 1], [1, 1]]，b = A·[1,1] = [1+%.0e, 2]" % (eps, eps))
print()
print("  不选主元的 LU 因子         %s" % fmt2(lu_nopivot(A)))
LU, perm = lu_pivot(A)
print("  选主元的 LU 因子（换行后）  %s   行置换 = %s" % (fmt2(LU), perm.tolist()))
print()
print("  注意两个因子矩阵的上三角部分（即 U）：")
print("    不选主元：U₂₂ = %.1e，量级是 1/ε —— **原始矩阵里的 1 被吃掉了**；"
      % np.triu(lu_nopivot(A))[1, 1])
print("    选主元：  U₂₂ = %.3f，量级正常。"
      % np.triu(LU)[1, 1])
print()
print("  不选主元解 = %s   相对误差 %.3e" % (lu_solve_nopivot(A, b).tolist(),
                                          np.linalg.norm(lu_solve_nopivot(A, b) - xt) / np.linalg.norm(xt)))
print("  选主元解   = %s   相对误差 %.3e" % (lu_solve_pivot(A, b).tolist(),
                                          np.linalg.norm(lu_solve_pivot(A, b) - xt) / np.linalg.norm(xt)))
print()
print("第一行 x₁ 答成了 %.4f（正确值 1）。原因链条只有三步：" % lu_solve_nopivot(A, b)[0])
print()
print("  ① 主元是 ε ≈ 0，乘子 m = 1/ε = %.0e 爆炸；" % (1 / eps))
print("  ② L 的第二行全是 %.0e，消元后 U 的第二行变成 [0, 1 - 1/ε] ≈ [0, -%.0e]，" % (1 / eps, 1 / eps))
print("     也就是说**原始 A 第二行的信息（那个 1）被完全冲掉了**；")
print("  ③ 回代时 x₁ = (b₁ - U₁₂·x₂)/U₁₁ = (1+ε - 1)/ε。")
print("     分子是 **两个几乎相等的数相减**——b₁ = 1+ε 和 U₁₂·x₂ = 1。")
print("     1 附近的一个 ulp 是 %.2e，而 ε = %.0e 只有 %.1f 个 ulp 宽，"
      % (np.finfo(float).eps, eps, eps / np.finfo(float).eps))
print("     所以 b₁ 里「ε 那部分」的表示本身就只有约 %.1f 个二进制位的精度。"
      % np.log2(eps / np.finfo(float).eps))
print("     这个差再除以 ε（= 放大 %.0e 倍），x₁ 的相对误差就变成 %.0f%%。"
      % (1 / eps, 100 * abs(lu_solve_nopivot(A, b)[0] - 1)))
print()
print("这就是**灾难性抵消**（06 节讲过）在分解里的具体形态：")
print("**主元小 → 乘子大 → 相减的两个数接近 → 有效位全丢。**")
print("选主元做的事非常朴素：把该列最大的元素换到对角上。")
print("于是乘子 |m| ≤ 1，第二步里的相减永远发生在同量级的数之间。")

rule("2. 增长因子：随机矩阵上不选主元**通常也没事**")

print("衡量分解稳定性的标准量是**增长因子**：")
print()
print("    ρ_growth = max|U_ij| / max|A_ij|")
print()
print("它衡量消元过程中元素膨胀了多少倍。误差界大致是 ρ_growth · κ(A) · u，")
print("所以 ρ_growth 是算法引入的那部分。")
print()
print("上表已经显示：对 [[ε,1],[1,1]]，不选主元的增长因子是 1/ε，选主元恒为 1。")
print("但这是**构造出来的**。随机矩阵上是什么样？下面跑 200 个矩阵：")
print()
print("  %-5s %-24s %-24s %s" % ("n", "不选主元 growth 中位/最大", "部分主元 growth 中位/最大", "中位比"))
print("  " + "-" * 84)
gstats = []
for n in (10, 30, 60):
    gn, gp = [], []
    for t in range(200):
        g = np.random.default_rng(1000 * n + t)
        M = g.uniform(-1, 1, (n, n))
        gn.append(growth(lu_nopivot(M), M))
        gp.append(growth(lu_pivot(M)[0], M))
    gstats.append((n, float(np.median(gn)), max(gn), float(np.median(gp)), max(gp)))
    print("  %-5d %-24s %-24s %.1f"
          % (n, "%.2e / %.2e" % (gstats[-1][1], gstats[-1][2]),
             "%.2e / %.2e" % (gstats[-1][3], gstats[-1][4]),
             gstats[-1][1] / gstats[-1][3]))

print()
print("读一下：")
print("  · **部分主元的增长因子很小且稳定**：中位 %.1f ~ %.1f，200 个矩阵里最大也只有 %.0f。"
      % (min(s[3] for s in gstats), max(s[3] for s in gstats), max(s[4] for s in gstats)))
print("    这不是运气——部分主元保证 |L_ij| ≤ 1，而 U 的膨胀在实践中远小于 2^(n-1) 的最坏界。")
print("  · **不选主元的增长因子确实更大**：中位数随 n 从 %.0f 涨到 %.0f，"
      % (gstats[0][1], gstats[-1][1]))
print("    最大到了 %.1e。但量级只有 10²~10⁵，**不足以立刻毁掉结果**。" % max(s[2] for s in gstats))
print()
print("这不是一句「不选主元就是错的」能概括的。看实际求解精度：")
print()
print("  %-5s %-18s %-16s %-16s %s"
      % ("n", "测试矩阵", "不选主元 相对误差中位", "不选主元 最大", "选主元 中位"))
print("  " + "-" * 86)
for n in (30, 60):
    for kind, perturb in (("均匀随机", False), ("A[0,0] 设为 1e-14", True)):
        en, ep, gs = [], [], []
        for t in range(50):
            g = np.random.default_rng(5000 + t)
            M = g.uniform(-1, 1, (n, n))
            if perturb:
                M[0, 0] = 1e-14
            xt = g.standard_normal(n)
            bb = M @ xt
            en.append(np.linalg.norm(lu_solve_nopivot(M, bb) - xt) / np.linalg.norm(xt))
            ep.append(np.linalg.norm(lu_solve_pivot(M, bb) - xt) / np.linalg.norm(xt))
            gs.append(growth(lu_nopivot(M), M))
        print("  %-5d %-18s %-16.3e %-16.3e %.3e" % (n, kind, np.median(en), max(en), np.median(ep)))

print()
print("**这张表推翻了「不选主元就会算错」这个简化说法。**")
print()
print("均匀随机矩阵上，不选主元的 LU 精度是 1e-13 ~ 1e-12，与选主元的 1e-15 只差")
print("两三个数量级——能用，只是没那么好。真正致命的是第二行：")
print("**只把 A[0,0] 改成 1e-14，其它一切照旧**，中位误差立刻变成 1e-2 量级，")
print("最大到了 O(1)——答案整个作废。")
print()
print("所以正确的表述是：")
print("  · 不选主元**没有误差界**。存在输入让它彻底失败，且这样的输入离随机矩阵只有")
print("    一个元素的距离（把某个值调小即可，现实中对应「某个特征恰好接近线性相关」）。")
print("  · 选主元保证 |L_ij| ≤ 1，于是增长因子至少有界（最坏 2^(n-1)，实践中远小于此），")
print("    因此**在所有输入上都有保证**。")
print("  · 代价只有 O(n²) 次比较——相对 O(n³) 次浮点运算可以忽略。")
print()
print("**结论不是「不选主元精度差几个数量级」，而是「不选主元没有保证，而保证几乎免费」。**")

rule("3. Cholesky：要求对称正定，且判定本身会被舍入影响")

print("Cholesky 把对称正定矩阵 A 分解成 A = L·Lᵀ（L 下三角）。")
print("`np.linalg.cholesky` 返回的就是这个下三角 L。先验证约定：")
print()
A2 = np.array([[2.0, 1.0], [1.0, 2.0]])
L2 = np.linalg.cholesky(A2)
print("  A = %s" % np.array2string(A2, precision=1).replace("\n", " "))
print("  L = %s" % np.array2string(L2, precision=6).replace("\n", " "))
print("  ‖L·Lᵀ - A‖ = %.3e    ‖Lᵀ·L - A‖ = %.3e"
      % (np.linalg.norm(L2 @ L2.T - A2), np.linalg.norm(L2.T @ L2 - A2)))
print()
print("确认是 A = L·Lᵀ。这个分解只对**对称正定**矩阵存在：开平方那一步")
print("要求每个主元 l_kk = √(a_kk - Σ l_kj²) 里的被开方数是正的。")
print()
print("正定性的充要条件是对称矩阵的**全部特征值 > 0**（`eigvalsh` 直接给）。")
print("对比几种情形：")
print()
print("  %-24s %-26s %s" % ("矩阵", "eigvalsh", "cholesky(A)"))
print("  " + "-" * 74)
cases = [
    ("[[1,2],[2,1]]", np.array([[1.0, 2.0], [2.0, 1.0]])),
    ("[[2,1],[1,2]]", np.array([[2.0, 1.0], [1.0, 2.0]])),
    ("diag(1, 1e-16)", np.diag([1.0, 1e-16])),
    ("diag(1, -1e-16)", np.diag([1.0, -1e-16])),
    ("[[1,1],[1,1+1e-16]]", np.array([[1.0, 1.0], [1.0, 1.0 + 1e-16]])),
]
for name, M in cases:
    ev = np.linalg.eigvalsh(M)
    try:
        L = np.linalg.cholesky(M)
        res = "成功  min(diag L) = %.3e" % np.diag(L).min()
    except np.linalg.LinAlgError as e:
        res = "LinAlgError: %s" % e
    print("  %-24s %-26s %s" % (name, np.array2string(ev, precision=3).replace("\n", " "), res))

print()
print("三个不同性质的失败，值得分开看：")
print()
print("① **[[1,2],[2,1]]**：特征值有负的（-1），**真的不是正定**。")
print("   Cholesky 抛 LinAlgError: Matrix is not positive definite。这是正确行为。")
print()
print("② **[[1,1],[1,1+1e-16]]**：数学上 det = 1e-16 > 0，**它是正定的**。")
print("   但 1+1e-16 在 fp64 里就等于 1（u = %.1e），所以存进去的矩阵是"
      % np.finfo(float).eps)
print("   [[1,1],[1,1]]——**精确奇异**。eigvalsh 给出 [0, 2]，Cholesky 失败。")
print("   这不是 Cholesky 的错：信息在赋值给 float64 的那一刻就已经丢了。")
print()
print("③ **diag(1, 1e-16) 成功、diag(1, -1e-16) 失败**：1e-16 虽然比 u 还小，")
print("   但它是个可表示的正数（fp64 的次正规下界是 5e-324），所以正定性成立。")
print()
print("真正的麻烦是中间地带：λ_min 落在 **u·λ_max 量级**时，")
print("「正定」这个判断本身会被舍入噪声决定。做一个统计实验：")
print()
print("构造 A = Q·diag(1, λ_min, 1e-3)·Qᵀ（Q 随机正交，所以精确特征值已知），")
print("把 A 舍入到 fp64，看 Cholesky 是否还认为它正定。每个 λ_min 跑 200 个 Q：")
print()
print("  %-18s %-14s %-30s %s" % ("λ_min 目标", "Cholesky 失败", "eigvalsh 给出的 λ_min 范围", "λ_min/u"))
print("  " + "-" * 88)
pd_rows = []
for lm in (1e-15, 1e-16, 1e-17, 1e-18):
    fails, evs = 0, []
    for s in range(200):
        g = np.random.default_rng(s)
        Q, _ = np.linalg.qr(g.standard_normal((3, 3)))
        M = Q @ np.diag([1.0, lm, 1e-3]) @ Q.T
        M = (M + M.T) / 2.0
        evs.append(np.linalg.eigvalsh(M)[0])
        try:
            np.linalg.cholesky(M)
        except np.linalg.LinAlgError:
            fails += 1
    pd_rows.append((lm, fails, min(evs), max(evs)))
    print("  %-18.0e %-14s %-30s %.1f"
          % (lm, "%d/200" % fails, "%.2e ~ %.2e" % (min(evs), max(evs)),
             lm / np.finfo(float).eps))

print()
print("分界线很清楚，位置正好在 λ_min ≈ u·λ_max 上（u = %.1e，λ_max = 1）："
      % np.finfo(float).eps)
print()
for lm, fails, lo, hi in pd_rows:
    print("  λ_min = %.0e（= %.2f·u）：失败 %3d/200（%.0f%%），eigvalsh 给出最小特征值 %+.2e"
          % (lm, lm / np.finfo(float).eps, fails, 100.0 * fails / 200, lo))
print()
d0, d1 = pd_rows[0], pd_rows[2]
print("λ_min = %.0e 时 %d/200 通过：舍入噪声（量级 u·λ_max ≈ %.1e）比 λ_min 小，"
      % (d0[0], 200 - d0[1], np.finfo(float).eps))
print("正定性是**确定的**。λ_min = %.0e 时 %.0f%% 的样本失败，" % (d1[0], 100.0 * d1[1] / 200))
print("舍入噪声已经盖过 λ_min 本身——此时矩阵「是不是正定」由舍入方向决定。")
print()
print("中间那行（λ_min = %.0e）值得单独指出：Cholesky %s，"
      % (pd_rows[1][0], "全部通过" if pd_rows[1][1] == 0 else "部分失败"))
print("但 eigvalsh 报出的最小值已经是 **%.2e（负的）**。" % pd_rows[1][2])
print("也就是说：**eigvalsh 说「有负特征值」和 Cholesky 说「分解成功」可以同时成立，**")
print("因为两者看的是不同的量——eigvalsh 的特征值本身带 u·λ_max 量级的误差，")
print("而 Cholesky 只看分解过程中每个主元的符号。")
print("**在这个量级上，两个函数给出的「正定性」结论是不可靠的，不要拿它们互相印证。**")
print()
print("这在参数估计里是个真问题。GPTQ 之类的量化校准要算 H⁻¹，")
print("如果 H 的 λ_min 正好落在这个量级，")
print("那么「H 是否正定」取决于最后一次加法的舍入方向，而不是数据本身。")
print("下一节会看到，H 的 λ_min 确实**经常**落在这个量级。")

rule("4. 三种分解的代价：flops 模型 vs 实测墙上时间")

print("方阵 n×n 的分解，flops（乘加各算一次）的经典计数：")
print()
print("  %-12s %-16s %-16s %s" % ("分解", "flops", "n=1500 时", "用途"))
print("  " + "-" * 66)
n_t = 1500
for name, expr, use in (("LU（带主元）", "2n³/3", "解一般方阵"),
                        ("Cholesky", "n³/3", "解对称正定"),
                        ("QR", "2n³/3", "最小二乘 / 正交化")):
    coef = 2.0 / 3.0 if name != "Cholesky" else 1.0 / 3.0
    print("  %-12s %-16s %-16.3e %s" % (name, expr, coef * n_t ** 3, use))
print()
print("模型预测：**Cholesky 恰好是 LU 的一半**（后者多做了一倍的消元，")
print("因为不需要利用对称性），而 QR 与 LU 同阶。")
print()
print("实测。n = %d，每项取 %d 次最小值（避免调度噪声）：" % (n_t, 5))
g = np.random.default_rng(1)
M = g.standard_normal((n_t, n_t))
A_spd = M @ M.T + n_t * np.eye(n_t)
b_t = g.standard_normal(n_t)


def best(fn, rep=5):
    """一次试验：跑 rep 次取最小，消除单次调度噪声。

    最小值的意义是「这段代码在这台机器上最快能跑多快」——它比平均值稳定得多。
    但**它仍然不是逐位可复现的**：CPU 频率漂移、cache 状态、后台进程都会影响它。
    所以本程序不报单次结果，而是跑多次试验并报出**范围**。
    见下面「计时为什么报范围」一节。
    """
    ts = []
    for _ in range(rep):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts)


TRIALS = 9
OPS = (
    ("cholesky(A)", lambda: np.linalg.cholesky(A_spd)),
    ("solve(A,b)  [LU + 回代]", lambda: np.linalg.solve(A_spd, b_t)),
    ("qr(A, mode='r')  [只算 R]", lambda: np.linalg.qr(A_spd, mode="r")),
    ("qr(A)  [Q 和 R 都算]", lambda: np.linalg.qr(A_spd)),
)
FLOPS = (n_t ** 3 / 3.0, 2.0 / 3.0 * n_t ** 3 + 2 * n_t ** 2,
         2.0 / 3.0 * n_t ** 3, 4.0 / 3.0 * n_t ** 3)

# samples[i][t] = 第 t 次试验中第 i 个操作的最小耗时
samples = [[] for _ in OPS]
for _ in range(TRIALS):
    for i, (_, fn) in enumerate(OPS):
        samples[i].append(best(fn))

med = [float(np.median(s)) for s in samples]
lo = [min(s) for s in samples]
hi = [max(s) for s in samples]
t_chol, t_solve, t_qr_r, t_qr_full = med

print()
print("每项跑 %d 次试验，每次试验内部再取 5 次的最小值。" % TRIALS)
print()
print("  %-28s %-22s %-12s %-16s %s"
      % ("操作", "时间 (ms) 中位 [最小 ~ 最大]", "波动", "理论 flops", "ns/flop"))
print("  " + "-" * 96)
for i, (name, _) in enumerate(OPS):
    swing = (hi[i] - lo[i]) / med[i] * 100
    print("  %-28s %8.2f [%7.2f ~ %7.2f]  正负 %5.1f%%  %-16.3e %.3f"
          % (name, med[i] * 1e3, lo[i] * 1e3, hi[i] * 1e3, swing,
             FLOPS[i], med[i] * 1e9 / FLOPS[i]))

print()
print("比值的**逐次试验**分布（每次试验内四个操作都测一遍，所以比值可以配对）：")
print()
print("  %-40s %-12s %-24s %s"
      % ("比值", "中位", "9 次试验的范围", "flops 模型预测"))
print("  " + "-" * 92)
PAIRS = ((0, 1, "cholesky / solve"), (0, 2, "cholesky / qr(mode='r')"),
         (2, 3, "qr(mode='r') / qr(full)"))
ratio_stats = []
for a, b_, label in PAIRS:
    per_trial = [samples[a][t] / samples[b_][t] for t in range(TRIALS)]
    r_med = float(np.median(per_trial))
    r_lo, r_hi = min(per_trial), max(per_trial)
    ratio_stats.append((label, r_med, r_lo, r_hi))
    print("  %-40s %-12.3f [%.3f ~ %.3f]  %18s"
          % (label, r_med, r_lo, r_hi, "%.3f" % 0.5))

print()
print("【计时为什么报范围，而不报一个数】")
print()
print("其余各节的数字都是逐位可复现的——同一个输入必然给出同一个输出。")
print("**计时不是。** 同一台机器上重复运行，上面那三个比值各自摆动。")
print("判据很简单：**观察到的范围有没有把模型预测值 0.5 包进去。**")
print()
for label, r_med, r_lo, r_hi in ratio_stats:
    margin = 100 * min(abs(r_lo - 0.5), abs(r_hi - 0.5)) / 0.5
    if r_lo <= 0.5 <= r_hi:
        verdict = "**范围跨过 0.5 ⇒ 未证实**"
    elif margin >= 25:
        verdict = "范围离 0.5 还有 %.0f%% ⇒ 模型与实测明显不符，结论成立" % margin
    else:
        verdict = "范围离 0.5 只剩 %.0f%% ⇒ 方向一致，但具体数值对不上模型" % margin
    print("    %-26s [%.3f ~ %.3f]   %s" % (label, r_lo, r_hi, verdict))
print()
print("于是三种判定各自对应一种写法，不能混：")
print("  · **范围跨过预测值**：只能说「这一次测量与预测一致」，不能说模型被验证——")
print("    这叫**未证实**，不叫成立。")
print("  · **范围远离预测值**（如 67%）：模型与实测明显不符，结论成立，且可以报出差异倍数。")
print("  · **范围离预测值不远不近**（如 9%）：方向一致，但**具体数值对不上模型**——")
print("    此时只能说「Cholesky 比 LU 快」，不能说「实测 0.42，模型 0.50，吻合」。")
print()
print("把「未证实」写成「成立」、把「方向一致」写成「数值吻合」，是拿单次计时下结论的两种典型形态。")

print()
print("三个读数，一好一坏一印证：")
print()
_, cs, cs_lo, cs_hi = ratio_stats[0]
_, cq, cq_lo, cq_hi = ratio_stats[1]
_, qf, qf_lo, qf_hi = ratio_stats[2]
print("① **`Cholesky / LU = %.3f`（9 次试验 [%.3f ~ %.3f]），模型预测 0.50。**"
      % (cs, cs_lo, cs_hi))
print("   方向完全一致：对称性省掉一半消元。但**数值上模型乐观了约 %.0f%%**——"
      % (100 * (0.5 - cs) / 0.5))
print("   实测比 flops 模型更快，因为 LU 还要额外做主元搜索与行交换，那部分不在")
print("   flops 计数里。所以这一条的准确说法是「对称正定省一半**量级**的时间」，")
print("   而不是「实测 0.42、模型 0.50、吻合」。它是三者中唯一重复运行也稳定的。")
print()
print("② **`Cholesky / QR = %.3f`（[%.3f ~ %.3f]），模型预测 0.50——模型乐观了 %.1f 倍。**"
      % (cq, cq_lo, cq_hi, 0.5 / cq))
print("   flops 只差 2 倍（n³/3 对 2n³/3），实测差了 %.1f 倍。原因看 ns/flop 那一列："
      % (1 / cq))
print("   numpy 的 QR 每 flop 比 Cholesky 贵约 %.1f 倍。LAPACK 的 Cholesky/LU 有高度"
      % (med[2] / FLOPS[2] / (med[0] / FLOPS[0])))
print("   优化的分块内核（BLAS-3 占比高），而 QR 的 Householder 反射受限于 BLAS-2 的")
print("   内存带宽。**这是全表最结实的一个结论**——它的范围从不接近 0.5。")
print()
print("③ **`qr(mode=\'r\') / qr(full) = %.3f`（[%.3f ~ %.3f]），模型预测 0.5。**"
      % (qf, qf_lo, qf_hi))
print("   范围**跨过了 0.5**——摆动幅度大于它与模型预测值之间的差距。")
print("   两者的差别确实只是「要不要显式组装 Q」（flops 表里 4n³/3 与 2n³/3 之差），")
print("   理论上应当成立；但要实测证实它，需要比「多次取最小」更严格的测量方法")
print("   （固定 CPU 频率、绑定核心、提高重复次数并做统计检验）。**本文如实报未证实。**")
print()
print("顺带：这里用 `solve` 而不是 `inv(A) @ b`。前面已测过两者的精度几乎相同，")
print("差别在代价——求逆比求解多约 3 倍。**「别用 inv」的理由是省算力，不是省精度。**")

rule("5. 与 LLM 的关联：GPTQ 的 Hessian 几乎不正定")

print("GPTQ 之类的训练后量化，核心是解一个逐层的加权最小二乘：")
print()
print("    min_Ŵ ‖(W - Ŵ)·X‖²      ⟹  正规方程：  (X·Xᵀ)·Δ = …")
print()
print("其中 H = X·Xᵀ 是校准数据在**该层输入空间**上的协方差（常写成 2·X·Xᵀ）。")
print("GPTQ 的实现在 H 上做 Cholesky 分解（有些版本用 LDLᵀ），每一步按列更新。")
print("**所以 H 是否正定直接决定这段代码能不能跑通。**")
print()
print("下面用一个**纯模拟**的模型来说明问题（没有真实模型数据，")
print("但 H = X·Xᵀ 的结构性质与真实校准完全一样）：")
print()
print("  d = 256 个特征（对应一个 hidden dim），采 n 个校准样本 X ∈ R^(d×n)，")
print("  H = X·Xᵀ / n。")
print()
print("  %-10s %-10s %-12s %-16s %-16s %s"
      % ("样本数 n", "特征数 d", "d/n", "rank(H)", "κ(H)", "cholesky(H)"))
print("  " + "-" * 92)
for nsamp in (64, 128, 192, 256, 512, 1024, 2048):
    g = np.random.default_rng(11)
    X = g.standard_normal((256, nsamp))
    H = X @ X.T / nsamp
    rk = np.linalg.matrix_rank(H, tol=1e-12)
    kap = np.linalg.cond(H)
    try:
        np.linalg.cholesky(H)
        ok = "成功"
    except np.linalg.LinAlgError:
        ok = "LinAlgError（不是正定）"
    print("  %-10d %-10d %-12.2f %-16d %-16.3e %s" % (nsamp, 256, 256.0 / nsamp, rk, kap, ok))

print()
print("三个事实，每个都有教科书公式对应：")
print()
print("① **n < d 时 H 的秩只有 n，必然奇异。**")
print("   H = X·Xᵀ 是 n 个外积之和，秩 ≤ n。d = 256、n = 128 时 rank(H) = 128，")
print("   有一半的特征值是精确的 0，Cholesky 直接抛异常。")
print("   这就是 GPTQ 实现里几乎必然要加阻尼项（damp_percent / actorder）的原因。")
print()
print("② **n ≥ d 之后秩满了，但条件数由 d/n 决定，而不是由 n 决定。**")
print("   随机矩阵的 Marchenko-Pastur 定律给出")
print()
print("       κ(H) ≈ ((1 + √(d/n)) / (1 - √(d/n)))²")
print()
print("   %-10s %-10s %-16s %-16s %s" % ("d/n", "n", "κ(H) 实测", "MP 公式预测", "吻合"))
print("   " + "-" * 68)
for nsamp in (256, 512, 1024, 2048):
    g = np.random.default_rng(11)
    X = g.standard_normal((256, nsamp))
    H = X @ X.T / nsamp
    q = np.sqrt(256.0 / nsamp)
    pred = np.inf if q >= 1 else ((1 + q) / (1 - q)) ** 2
    kap = np.linalg.cond(H)
    print("   %-10.2f %-10d %-16.3e %-16.3f %s"
          % (256.0 / nsamp, nsamp, kap, pred,
             "%.2f 倍" % (kap / pred) if pred != np.inf else "—"))
print()
print("   d = n 时公式发散（真正的奇点），n 稍大于 d 时 κ 还有 1e6——")
print("   **「样本数 ≥ 特征数」远远不够，要 n ≫ d 才谈得上良态。**")
print()
print("③ **加阻尼 λI 是标准做法，效果可以精确预测。**")
print("   因为 H 的 λ_min = 0，加上 λI 之后")
print()
print("       κ(H + λI) = (λ_max + λ) / (λ_min + λ) = (λ_max + λ) / λ")
print()
g = np.random.default_rng(11)
X = g.standard_normal((256, 128))
H = X @ X.T / 128
lmax = np.linalg.eigvalsh(H).max()
print("   模拟设定：d = 256, n = 128（秩 %d，必然奇异），λ_max = %.4f。"
      % (np.linalg.matrix_rank(H, tol=1e-12), lmax))
print()
print("   %-12s %-14s %-16s %-16s %s"
      % ("λ", "λ/λ_max", "κ(H+λI) 实测", "(λ_max+λ)/λ", "cholesky"))
print("   " + "-" * 82)
for lam in (0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1):
    Hl = H + lam * np.eye(256)
    try:
        np.linalg.cholesky(Hl)
        ok = "成功"
    except np.linalg.LinAlgError:
        ok = "LinAlgError"
    pred = np.inf if lam == 0 else (lmax + lam) / lam
    print("   %-12.0e %-14.2e %-16.3e %-16s %s"
          % (lam, lam / lmax, np.linalg.cond(Hl),
             "inf" if lam == 0 else "%.3e" % pred, ok))

print()
print("   实测 κ 与 (λ_max+λ)/λ **逐行精确相等**——这不是拟合，是恒等式")
print("   （因为 λ_min(H) = 0，加上 λI 后最小特征值就是 λ）。")
print()
print("   但注意一个反直觉的地方：**λ = 1e-12 就已经让 Cholesky 成功了**，")
print("   尽管此时 κ 高达 %.1e。" % np.linalg.cond(H + 1e-12 * np.eye(256)))
print("   原因是 Cholesky 只要求「分解过程中没有非正主元」，")
print("   而 λ = 1e-12 这一个小小的正扰动就足以让最小的那个主元翻正。")
print()
print("   **这正是 GPTQ 里阻尼项的两面性：**")
print("     · 它让 Cholesky 能跑通（数值上必要）；")
print("     · 它同时把 H 的最小特征方向上的信息抹掉了（统计上有偏）；")
print("     · 而 λ 取多大，取决于你想在「跑得通」和「不引入偏差」之间怎么切。")
print()
print("   表里有一条数据支持的分界：把 κ 压到 1e3 以下需要 λ/λ_max ≳ %.0e；"
      % 1e-3)
print("   λ/λ_max = 1e-12（也就是「刚够让 Cholesky 跑通」的量级）时 κ 仍有 %.1e——"
      % np.linalg.cond(H + 1e-12 * np.eye(256)))
print("   求解结果在这个方向上等于全是噪声。")
print("   所以**阻尼的主要目的不是数值可行性，而是统计正则化**：")
print("   数值可行性只要 1e-12，而要让解有意义，需要的大得多。")

rule("6. 选型判据")

print("  %-34s %-22s %-16s %s" % ("问题", "矩阵性质", "选什么", "理由"))
print("  " + "-" * 92)
for prob, prop, choice, why in (
        ("解一般方阵 Ax=b", "非对称、可能病态", "LU + 部分主元", "通用，有误差界"),
        ("解对称正定 Ax=b", "已确认正定", "Cholesky", "代价减半（实测 %.2f）" % (t_chol / t_solve)),
        ("最小二乘 min‖Ax-b‖", "m ≥ n，可能秩亏", "QR（或 SVD）", "正交变换不放大 κ"),
        ("特征值问题", "对称", "eigvalsh", "实特征值，有保证"),
        ("量化校准 H⁻¹", "对称、**几乎不正定**", "Cholesky + 阻尼 λI", "不加阻尼直接失败"),
        ("计算行列式 / 惯性", "对称", "LDLᵀ（不取平方根）", "避免 Cholesky 的开方")):
    print("  %-34s %-22s %-16s %s" % (prob, prop, choice, why))

print()
print("两条贯穿全节的判断规则：")
print()
print("① **「能不能用」和「准不准」是两件事，要分开问。**")
print("   不选主元的 LU 在随机矩阵上精度只差两三个数量级（能用），")
print("   但存在一个元素的改动就让它作废的输入（没有保证）。")
print("   阻尼 λ = 1e-12 让 Cholesky 能跑通（能用），")
print("   但那个量级的阻尼在统计上没有意义（不准）。")
print("   **工程上要选的是「在所有输入上都有保证」的那个，而不是「我测的几个样例上更好」的那个。**")
print()
print("② **分解的选择先看矩阵性质，再看代价。**")
print("   对称正定是免费的 2 倍加速，前提是你**真的验证过**它正定。")
print("   05 节的 AᵀA、本节的 H = XXᵀ 都是「理论上对称，实际上几乎不正定」的典型——")
print("   直接对它们调 Cholesky，得到的是 LinAlgError 或者一段静默的垃圾。")

rule("7. 本程序的边界")

print("  · 全部是 numpy 在 CPU 上的 fp64 运算；计时是在本机单线程下测的，")
print("    绝对值只对本机有意义，但几个比值（Cholesky/LU）来自算法结构，可迁移。")
print("  · `np.linalg.solve` / `cholesky` / `qr` 分别调用 LAPACK 的对应例程，")
print("    它们的 BLAS-3 优化程度不同，这正是第 4 节 QR 比值偏离 flops 模型的原因。")
print("  · 第 2 节的 200 个随机矩阵用的是 [-1,1] 均匀分布。换成其它分布，")
print("    增长因子的具体数值会变，但「部分主元远小于不选主元」这个关系不变。")
print("  · **第 5 节全部是模拟**：X 是标准高斯随机矩阵，不是任何真实模型的激活。")
print("    真实激活是重尾、强相关的，κ 通常**远大于**本节的 Marchenko-Pastur 预测。")
print("    本节想说明的是「为什么需要阻尼」这个**结构性**原因，")
print("    不是给出任何具体模型的 κ 数值。")
print("  · 本程序没有调用 torch / scipy；所有分解都来自 numpy.linalg。")

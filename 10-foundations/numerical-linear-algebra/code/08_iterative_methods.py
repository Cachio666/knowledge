"""迭代法：用低精度分解 + 高精度残差把精度找回来，以及 CG 的 √κ。

运行：python3 08_iterative_methods.py
依赖：numpy
"""

import numpy as np


def rule(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


U32 = 2.0 ** -24
U64 = 2.0 ** -53


def lu_pivot32(A):
    """fp32 的 LU 分解，带部分主元。下三角存 L，上三角存 U，另返回行置换。

    这是真实的 LAPACK 做法在 fp32 上的缩微版：分解**只做一次**，
    之后每次精化都复用它，代价是 O(n²) 的回代而不是 O(n³) 的重新分解。
    """
    A = np.array(A, dtype=np.float32, copy=True)
    n = A.shape[0]
    perm = np.arange(n)
    for k in range(n):
        p = k + int(np.argmax(np.abs(A[k:, k])))
        if p != k:
            A[[k, p], :] = A[[p, k], :]
            perm[[k, p]] = perm[[p, k]]
        piv = A[k, k]
        if piv == 0.0:
            continue
        A[k + 1:, k] /= piv
        A[k + 1:, k + 1:] -= np.outer(A[k + 1:, k], A[k, k + 1:])
    return A, perm


def lu_solve32(LU, perm, b):
    """用上面得到的 fp32 因子解方程，全程 fp32。"""
    n = LU.shape[0]
    y = np.array(b, dtype=np.float32)[perm].copy()
    for i in range(1, n):                                # 前代 L y = P b
        y[i] = np.float32(y[i] - np.dot(LU[i, :i], y[:i]))
    x = y.copy()
    for i in range(n - 1, -1, -1):                       # 回代 U x = y
        x[i] = np.float32((x[i] - np.dot(LU[i, i + 1:], x[i + 1:])) / LU[i, i])
    return x


def make_spd(n, kappa, seed):
    """构造特征值对数均匀分布、条件数恰好为 kappa 的对称正定矩阵。"""
    g = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(g.standard_normal((n, n)))
    return Q @ np.diag(np.logspace(0, -np.log10(kappa), n)) @ Q.T


rule("1. 迭代精化：低精度分解 + 高精度残差")

print("算 Ax = b。fp32 的 LU 分解给出的解 x̂ 只有约 1e-7 的相对精度，")
print("但**残差** r = b - A·x̂ 可以在 fp64 里算得极准。于是：")
print()
print("    ① fp32 做一次 LU 分解，解出 x̂                （代价 O(n³)，只做一次）")
print("    ② fp64 算残差      r = b - A·x̂                （代价 O(n²)）")
print("    ③ 用**同一组 fp32 因子**解  A·d = r，得到修正量 d （代价 O(n²)）")
print("    ④ 在 fp64 里累积   x̂ ← x̂ + d                  （代价 O(n)）")
print("    ⑤ 回到 ②，重复")
print()
print("关键在 ③：修正量 d 自己也在 fp32 里算，所以 d 本身带误差。")
print("但 ④ 把 d 加进 x̂ 时是 fp64——**修正量不准没关系，只要它指向正确的方向**。")
print("迭代之所以收敛，是因为每一步都把残差按某个因子缩小。")
print()
print("下面实测。n = 200，A 对称正定、条件数可控，x_true 随机，b = A·x_true。")
print("表里每行是一个 κ（同一组随机数，只有 κ 在变），每列是精化到第 k 步的相对误差。")
print()
n = 200
ITERS = 5
KAPPAS = (1e2, 1e4, 1e6, 1e8)
results = {}
for kappa in KAPPAS:
    A = make_spd(n, kappa, 42)
    g = np.random.default_rng(7)
    xt = g.standard_normal(n)
    b = A @ xt
    LU, perm = lu_pivot32(A.astype(np.float32))
    x = lu_solve32(LU, perm, b.astype(np.float32)).astype(np.float64)
    errs = [np.linalg.norm(x - xt) / np.linalg.norm(xt)]
    for _ in range(ITERS):
        r = b - A @ x                                    # fp64 残差
        d = lu_solve32(LU, perm, r.astype(np.float32)).astype(np.float64)
        x = x + d                                        # fp64 累积
        errs.append(np.linalg.norm(x - xt) / np.linalg.norm(xt))
    results[kappa] = errs

print("  %-10s %-12s %-12s %-12s %-12s %-12s %s"
      % ("κ(A)", "不精化", "精化 1 次", "精化 2 次", "精化 3 次", "精化 5 次", "1 次的增益"))
print("  " + "-" * 92)
for kappa in KAPPAS:
    e = results[kappa]
    print("  %-10.0e %-12.3e %-12.3e %-12.3e %-12.3e %-12.3e %.0f 倍"
          % (kappa, e[0], e[1], e[2], e[3], e[5], e[0] / e[1]))

print()
print("三个读数：")
print()
e2, e4, e6, e8 = (results[k] for k in KAPPAS)
print("① **精化在 κ 小的时候几乎免费地拿到全部精度。**")
print("   κ = 1e2：不精化是 %.2e，一次精化到 %.2e（提升 %.0f 倍）；"
      % (e2[0], e2[1], e2[0] / e2[1]))
print("   两次之后是 %.2e。**再迭代也不动了**——后面几步是 %.2e / %.2e / %.2e，"
      % (e2[2], e2[3], e2[4], e2[5]))
print("   在地板附近随机跳动而不再下降。这个地板是 fp64 的舍入噪声（u64 = %.1e），"
      % U64)
print("   不是算法还能再改进的地方。")
print()
print("② **精化的增益随 κ 单调下降。**")
print("   %-10s %-16s %-16s %s" % ("κ", "不精化", "精化 1 次", "增益"))
print("   " + "-" * 58)
for kappa in KAPPAS:
    e = results[kappa]
    print("   %-10.0e %-16.3e %-16.3e %.0f 倍" % (kappa, e[0], e[1], e[0] / e[1]))
print()
print("   从 %.0f 倍一路掉到 %.1f 倍。κ = 1e8 时精化一次几乎没用（只从 %.2e 到 %.2e）。"
      % (e2[0] / e2[1], e8[0] / e8[1], e8[0], e8[1]))
print()
print("③ **原因正是「修正量本身也在 fp32 里算」。** 设 x̂ 的误差为 e，则")
print()
print("       d = Â⁻¹·r = Â⁻¹·A·e ≈ e + (Â⁻¹ - A⁻¹)·A·e")
print()
print("   新误差 e_new = e - d 就只剩下第二项，它的相对大小由 Â⁻¹ 的误差决定，")
print("   而那个误差是 κ·u32 量级。**κ 越大，d 越不准，e_new 缩得越慢。**")
print()
print("   把这句定量化。定义每一步的**收缩因子** r_k = err_k / err_{k-1}。")
print()
print("   注意不能简单地对所有步取几何平均：κ 小时迭代几步就撞上 **fp64 的地板**，")
print("   之后的步只是在地板附近随机跳动，会把平均值拉高。")
print("   所以下表只统计「误差确实下降了 1.5 倍以上」的那些步；")
print("   若一步都没下降，就报全部步的平均（这时它就是 > 1 的发散率）。")
print()
print("  %-9s %-12s %-14s %-12s %-14s %s"
      % ("κ(A)", "κ·u32", "有效收缩因子", "计入步数", "首步 err₁/err₀", "行为"))
print("  " + "-" * 88)
rate_rows = []
for kappa in (1e2, 1e4, 1e6, 1e7, 1e8, 1e9, 1e10):
    A = make_spd(n, kappa, 42)
    g = np.random.default_rng(7)
    xt = g.standard_normal(n)
    b = A @ xt
    LU, perm = lu_pivot32(A.astype(np.float32))
    x = lu_solve32(LU, perm, b.astype(np.float32)).astype(np.float64)
    errs = [np.linalg.norm(x - xt) / np.linalg.norm(xt)]
    for _ in range(8):
        r = b - A @ x
        d = lu_solve32(LU, perm, r.astype(np.float32)).astype(np.float64)
        x = x + d
        errs.append(np.linalg.norm(x - xt) / np.linalg.norm(xt))
    ratios = [errs[k + 1] / errs[k] for k in range(len(errs) - 1)]
    active = [q for q in ratios if q < 1.0 / 1.5]
    used = active if active else ratios
    geo = float(np.exp(np.mean(np.log(used))))
    behavior = "**发散**" if geo >= 1.0 else ("收敛（快）" if geo < 0.1 else "收敛（慢）")
    rate_rows.append((kappa, kappa * U32, geo, len(used), errs[1] / errs[0], behavior))
    print("  %-9.0e %-12.2e %-14.5f %-12s %-14.6f %s"
          % (kappa, kappa * U32, geo, "%d / 8" % len(used), errs[1] / errs[0], behavior))

print()
conv_rows = [r for r in rate_rows if r[2] < 1.0]
div_rows = [r for r in rate_rows if r[2] >= 1.0]
print("顺带看一眼「计入步数」这一列：κ = 1e2 时 8 步里只有 2 步是真的在收敛")
print("（其余 6 步已经在地板上），而 κ = 1e8 时 8 步全都在收敛——只是每步赚得少。")
print()
print("把「收缩因子」和「κ·u32」对照，只能说清两件事：")
print()
print("  · **收缩因子随 κ 单调上升**：%s。"
      % " → ".join("%.5f" % r[2] for r in conv_rows))
print("  · **它不是一个简单的 κ·u32 倍数**。算一下「收缩因子 / (κ·u32)」：")
print()
print("  %-9s %-14s %-14s %s" % ("κ", "κ·u32", "收缩因子", "收缩因子 / κ·u32"))
print("  " + "-" * 60)
for kappa, ku, geo, nb_, r1, behavior in rate_rows:
    print("  %-9.0e %-14.2e %-14.5f %.3f" % (kappa, ku, geo, geo / ku))
print()
print("这个比值从 %.1f 变到 %.3f，跨了约 %.0f 倍——**它不是常数**。"
      % (conv_rows[0][2] / conv_rows[0][1], conv_rows[-1][2] / conv_rows[-1][1],
         (conv_rows[0][2] / conv_rows[0][1]) / (conv_rows[-1][2] / conv_rows[-1][1])))
print()
print("所以不要写下「收缩因子 = κ·u32」这样的公式——**实测不支持它**。")
print("能站得住的陈述只有定性的两条：")
print()
print("  · 收缩因子**随 κ·u32 增长而增大**（因为修正量 d 自己就是在 fp32 里算的，")
print("    它的误差量级由 κ·u32 决定）；")
print("  · 当 κ·u32 大到一定程度，收缩因子**越过 1**，迭代从收敛转为发散。")
print()
print("判据因此也是定性的区间，而不是一个等式：")
print()
print("  · κ·u32 ≲ 0.1：收缩因子 < 0.01，每步赚 %.1f 位以上，**安全**；"
      % (-np.log10(conv_rows[1][2])))
print("  · κ·u32 ≈ 1 ~ 10：收缩因子 %.2f ~ %.2f，每步只赚 %.1f ~ %.1f 位，**勉强**；"
      % (conv_rows[-2][2], conv_rows[-1][2], -np.log10(conv_rows[-1][2]),
         -np.log10(conv_rows[-2][2])))
print("  · κ·u32 ≫ 10：收缩因子 %s > 1，**发散**。"
      % " / ".join("%.2f" % r[2] for r in div_rows))
print()
print("实测的临界点落在 κ·u32 的 %d（还收敛）到 %d（发散）之间。"
      % (conv_rows[-1][1], div_rows[0][1]))
print("工程上没人会贴着临界点用，安全区取 **κ·u32 ≲ 0.1**——")
print("对应 fp32 分解 + fp64 残差可靠处理到 κ ~ 1e6。")
print()
print("一句话总结这一节：**迭代精化把「精度」问题变成了「κ·u32 够不够小」的问题。**")
print("κ 小的时候它是白送的（每步赚 4 位以上），κ 大的时候它一点忙都帮不上。")

rule("2. 这是「低精度计算 + 高精度修正」的原型")

print("把第 1 节的四步和混合精度训练逐个对上：")
print()
print("  %-34s %-34s %s" % ("迭代精化（本程序）", "mixed-precision 训练", "角色"))
print("  " + "-" * 96)
for a, b_, c in (
        ("fp32 LU 分解（只做一次）", "fp16/bf16 的前向与反向", "廉价、量大、有误差"),
        ("fp32 因子解 A·d = r", "优化器步产生的更新量 Δw", "同样在低精度里算"),
        ("fp64 算残差 r = b - A·x̂", "fp32 主权重上算的 loss / 梯度", "高精度负责「发现误差」"),
        ("fp64 累积 x̂ ← x̂ + d", "fp32 master weights：w ← w + Δw", "高精度负责「记住结果」"),
        ("收缩因子随 κ·u32 上升", "训练能否收敛取决于 κ(H)·u_bf16", "同一个稳定性条件")):
    print("  %-34s %-34s %s" % (a, b_, c))

print()
print("对应关系的要点只有一条：")
print()
print("  **低精度用来产生修正量，高精度用来累积结果。**")
print()
print("这解释了 fp16/bf16 训练里那两个看起来多余的设计：")
print()
print("  · **fp32 主权重（master weights）**：权重本身用低精度存会因为累积而漂移，")
print("    但**更新量** Δw 在低精度里算没关系——正如本程序里 d 在 fp32 里算没关系。")
print("    只要 w 这个「累积器」是 fp32，方向正确的 Δw 就能一步步把误差修掉。")
print("  · **loss scaling**：这不是精度问题，是第 04 节的**动态范围**问题——")
print("    fp16 的次正规下界让小于 6e-8 的梯度直接归零，乘一个大系数再除回来即可。")
print("    它和精化是两件事，不要混为一谈。")
print()
print("同样的结构在别处也出现：Kahan 求和（低精度加、高精度存补偿项）、")
print("以及 GPU 上「bf16 输入 + fp32 累加器」的 Tensor Core。")
print("**它们的共同形状都是「窄精度的算子 + 宽精度的状态」。**")

rule("3. 共轭梯度：迭代次数由 √κ 控制")

print("前一节的迭代精化每一步都在解一个线性方程组。CG（共轭梯度）则完全不同：")
print("它**不分解矩阵**，只用矩阵-向量乘 A·p。")
print()
print("    r0 = b - A·x0,  p0 = r0")
print("    for k = 0, 1, 2, …")
print("        α  = (rᵏᵀrᵏ) / (pᵏᵀA pᵏ)")
print("        x  ← x + α·p")
print("        r  ← r - α·A·p")
print("        β  = (rᵏ⁺¹ᵀrᵏ⁺¹) / (rᵏᵀrᵏ)")
print("        p  ← r + β·p")
print()
print("对对称正定 A，CG 的收敛率由 √κ 而不是 κ 控制：")
print()
print("    ‖x_k - x*‖_A ≤ 2·((√κ-1)/(√κ+1))^k · ‖x_0 - x*‖_A")
print()
print("把右边的 2·ρ^k 压到 ε，解出 k ≈ ½·√κ·ln(2/ε)。")
print("**这就是「CG 用 √κ」的出处**——κ 被开了一次平方。")
print()
print("实测。n = 800，A 的特征值对数均匀分布在 [1/κ, 1] 上，")
print("终止条件是相对残差 ‖r‖/‖b‖ < 1e-6，3 个随机右端项取中位数：")


def cg(A, b, tol, maxit):
    """标准 CG。返回 (迭代次数, 是否达标)。"""
    x = np.zeros_like(b)
    r = b.copy()
    p = r.copy()
    rs = r @ r
    nb = np.linalg.norm(b)
    for i in range(maxit):
        if np.sqrt(rs) / nb < tol:
            return i, True
        Ap = A @ p
        a = rs / (p @ Ap)
        x = x + a * p
        r = r - a * Ap
        rs_new = r @ r
        p = r + (rs_new / rs) * p
        rs = rs_new
    return maxit, False


TOL = 1e-6
N_CG = 800
print()
print("  %-10s %-10s %-10s %-10s %-14s %s"
      % ("κ(A)", "√κ", "迭代次数", "达到 1e-6?", "½√κ·ln(2/ε)", "迭代次数/√κ"))
print("  " + "-" * 84)
cg_rows = []
for kappa in (1e1, 1e2, 1e3, 1e4, 1e5, 1e6):
    its = []
    for seed in range(3):
        A = make_spd(N_CG, kappa, 100 + seed)
        bb = np.random.default_rng(200 + seed).standard_normal(N_CG)
        its.append(cg(A, bb, TOL, N_CG))
    it_med = int(np.median([t[0] for t in its]))
    ok = all(t[1] for t in its)
    pred = 0.5 * np.sqrt(kappa) * np.log(2.0 / TOL)
    cg_rows.append((kappa, it_med, ok, pred))
    print("  %-10.0e %-10.1f %-10d %-10s %-14.0f %.2f"
          % (kappa, np.sqrt(kappa), it_med, "是" if ok else "否（撞上限）", pred,
             it_med / np.sqrt(kappa)))

print()
conv = [r for r in cg_rows if r[2]]
print("先看达标的那几行：")
print()
for kappa, it, ok, pred in conv:
    print("  κ = %-8.0e  迭代 %-5d  ½√κ·ln(2/ε) = %-7.0f  实测/理论 = %.2f"
          % (kappa, it, pred, it / pred))
ks = [r[0] for r in conv]
its = [r[1] for r in conv]
print()
print("**迭代次数与 √κ 成正比，与 κ 无关。** κ 从 %.0e 涨到 %.0e（%.0f 倍），"
      % (ks[0], ks[-1], ks[-1] / ks[0]))
print("√κ 涨 %.1f 倍，迭代次数涨 %.1f 倍，最后一列的比值稳定在 %.2f ~ %.2f。"
      % (np.sqrt(ks[-1] / ks[0]), its[-1] / its[0],
         min(r[1] / np.sqrt(r[0]) for r in conv), max(r[1] / np.sqrt(r[0]) for r in conv)))
print()
print("而且它和理论值 ½√κ·ln(2/ε) 的比是 %.2f ~ %.2f——**理论界在这里几乎是紧的**，"
      % (min(r[1] / r[3] for r in conv), max(r[1] / r[3] for r in conv)))
print("不是常见的那种「松好几个数量级」的界。原因是对数均匀的特征值分布")
print("恰好让 CG 的表现接近最坏情形。")
print()
print("**但要如实报告一个限制：κ ≥ %.0e 的行没有达标。**" % cg_rows[-2][0])
print()
print("  %-10s %-16s %s" % ("κ(A)", "迭代次数", "情况"))
print("  " + "-" * 56)
for kappa, it, ok, pred in cg_rows:
    if not ok:
        print("  %-10.0e %-16d 撞到 n = %d 的上限，未达 1e-6" % (kappa, it, N_CG))
print()
print("这不是实现 bug。CG 有一个精确算术下的定理：**最多 n 步收敛**。")
print("所以理论迭代数 ½√κ·ln(2/ε) 只有在**远小于 n** 时才有意义：")
print("κ = %.0e 时理论需要 %.0f 步，而 n 只有 %d——CG 来不及走到那里。"
      % (cg_rows[-1][0], cg_rows[-1][3], N_CG))
print()
print("  %-10s %-18s %-12s %s" % ("κ(A)", "理论迭代数", "n = 800", "√κ 规律是否可见"))
print("  " + "-" * 70)
for kappa, it, ok, pred in cg_rows:
    print("  %-10.0e %-18.0f %-12d %s"
          % (kappa, pred, N_CG, "是" if pred < N_CG else "**否——被 n 截断**"))
print()
print("实践含义：**CG 的 √κ 只在 n ≫ √κ 时兑现**。")
print("表里 κ = 1e4 时理论需要 725 步而 n = 800，已经只剩 %.0f%% 的余量；"
      % (100 * (1 - 725.0 / N_CG)))
print("κ 再大一点，所需步数就超过 n，未预条件 CG 直接失去意义。")
print()
print("这正是**预条件**（preconditioning）的意义：")
print("它不是「加速收敛」，而是把 κ 本身降下来，让 √κ 重新落进 n 允许的范围。")
print("换句话说，**预条件解决的是可行性问题，不是速度问题**。")

rule("4. CG 的迭代次数不只由 κ 决定")

print("上表用的是对数均匀分布的特征值。同样的 κ，换一种谱形状，迭代次数会**剧变**。")
print("n = 800，κ 固定为 1e4，只改特征值的分布：")
print()
g = np.random.default_rng(3)
Q, _ = np.linalg.qr(g.standard_normal((N_CG, N_CG)))
specs = (
    ("对数均匀", np.logspace(0, -4, N_CG)),
    ("两端聚集（各一半）", np.concatenate([np.ones(N_CG // 2), np.full(N_CG - N_CG // 2, 1e-4)])),
    ("单点小特征值", np.concatenate([np.ones(N_CG - 1), [1e-4]])),
)
print("  %-22s %-14s %-12s %-14s %s"
      % ("谱形状", "κ(A) 实测", "迭代次数", "迭代/√κ", "说明"))
print("  " + "-" * 84)
spec_its = []
for name, s in specs:
    A = Q @ np.diag(s) @ Q.T
    its = []
    for seed in range(3):
        bb = np.random.default_rng(200 + seed).standard_normal(N_CG)
        its.append(cg(A, bb, TOL, N_CG)[0])
    it_med = int(np.median(its))
    spec_its.append((name, float(np.linalg.cond(A)), it_med))
    note = "接近理论最坏情形" if name == "对数均匀" else "**几乎瞬间收敛**"
    print("  %-22s %-14.3e %-12d %-14.2f %s"
          % (name, np.linalg.cond(A), it_med, it_med / np.sqrt(1e4), note))

print()
print("三种谱的条件数**相同**（都是 %.1e），迭代次数却是 %s。"
      % (spec_its[0][1], " / ".join("%d" % s[2] for s in spec_its)))
print("相差 %.0f 倍。**κ 一样，工作量完全不一样。**"
      % (spec_its[0][2] / max(spec_its[1][2], 1)))
print()
print("原因在 CG 收敛定理的**推导方式**：它只用到了「特征值全部落在 [λ_min, λ_max] 里」")
print("这一条信息，然后用一个 Chebyshev 多项式去覆盖整个区间。")
print("如果特征值只聚集在两个点上，一个二次多项式就能把它压到 0——")
print("所以「两端聚集」两次迭代就收敛了，无论 κ 是 1e4 还是 1e12。")
print()
print("（「单点小特征值」也是 2 次：除去那一个离群点，其余特征值全相同，")
print("  CG 在第 1 步就把那部分解完了，第 2 步处理离群方向。）")
print()
print("所以正确的说法是：")
print()
print("  · **κ 决定的是最坏情况**，即「存在一种谱形状让 CG 需要 ½√κ·ln(2/ε) 步」；")
print("  · **实际迭代次数由谱的分布决定**，聚集的谱可以快几个数量级；")
print("  · 这和 05 节关于 κ 的结论是同一回事——κ 是界，不是预测值。")
print()
print("工程含义：报告「CG 跑了多少步」时必须带上谱的信息（或至少是预条件子的效果），")
print("只说 κ 是没有信息量的。")

rule("5. 本程序的边界")

print("  · 全部是 numpy 在 CPU 上的 fp32 / fp64 运算，没有 GPU、没有 torch。")
print("  · LU 分解与三角回代是**自己用 Python 循环写的**，不是 LAPACK。")
print("    精度行为正确（这一步是本节的论点所在），但速度远低于库实现，")
print("    所以本程序不适合用来做性能对比。")
print("  · A 是用 Q·diag(σ)·Qᵀ 构造出来的，特征值已知、条件数可控。")
print("    真实问题的谱形状不是对数均匀的，收缩因子的具体数值会不同；")
print("    但「收缩因子随 κ·u 上升、越 1 即发散」这个**定性**结论来自算法结构。")
print("    第 1 节已经测过：把它写成「收缩因子 = c·κ·u」这样的等式是不成立的。")
print("  · 第 3 节的 CG **没有用预条件**。加了预条件之后 κ 会变小，")
print("    所需迭代次数随之下降——本节的表格描述的是未预条件 CG 的行为。")
print("  · CG 的实测是在 fp64 下做的，因为这里要考察的是**算法**的收敛性，")
print("    而不是舍入误差。真实低精度 CG 还会额外受到舍入导致的停滞影响，")
print("    迭代次数会**多于**本节给出的数字。")

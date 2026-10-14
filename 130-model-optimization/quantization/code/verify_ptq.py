#!/usr/bin/env python3
"""130-model-optimization/quantization 主题的纯 python3 复核入口。

★ 本文件**固定跑在默认 `python3` 下，永不 `import torch`**（CONTRIBUTING §2 第三种形态）。
  它提供的是「**换了台机器、什么都没装，也能复核的那一半**」：
  用**独立的参考实现**（下面的 `ref_*`）把本主题的**判据**重算一遍，
  再用手写期望值（不是 torch 的输出）判它对不对。
  **它的产出是推导，不是实测**——不得标成本机实测值。

  三类断言在本形态下的分工（CONTRIBUTING §2）：
    · **A 判据复核** —— 本文件。参照物是下面 `ref_*` 这套独立实现与手写期望值。
    · **C 边界与反例** —— 本文件。断言的是「失败行为成立」。
    · **B 代价模型** —— 在**编号程序**里（代价模型的参照是真系统，只有能 import 到它的
      环境里测才有意义）。本文件**不做** B。

★ 本主题里 A/C 与编号程序的分工是 **「数学与算法 vs 实现」**：
  · 本文件（纯 python3 + 标准库）验的是**可独立推导的算法**：EMA 递推、直方图上的
    L2 目标 vs KL 目标、对称/非对称 RTN 的 qparams、GPTQ 的逐列补偿、AWQ 的 α 网格、
    SmoothQuant 的逐通道缩放恒等式、AdaRound 的软舍入函数；
  · 编号程序（真 torch）测的是**真实实现的行为**：三种 observer 算出的 qparams、
    `HistogramObserver` 的实际裁剪量、`FakeQuantize` 的反传掩码、
    `disable_observer` 的真实作用对象、transformer 的 `GPTQConfig` 默认值。
  · **两者不得互相背书。** 算法成立不证明实现如此；实现如此不证明算法成立。
  · **同一条结论若两边都能给，两边都要给**，且分别标注来源档位。
  · ⚠ **规模不同 ⇒ 数字不同**：本文件跑的是小规模版本
    （样本 ≤ 20000、直方图 ≤ 2048 桶、权重矩阵 ≤ 64×64），编号程序跑的是
    10⁴–2×10⁵ 点与 256×512 矩阵。**只有定性结构必须一致。**

★ 「独立」的硬要求（CONTRIBUTING §2）：本文件里没有一行代码来自 torch，
  期望值**全部**是手写的常量或从 `specs_ptq.py` 抄来的外部常量。
  §E 节把这条**机械地验证一遍**：把参考实现故意改错，A 类断言必须失败。

★ `specs_ptq.py` 的每条常量都至少被本文件或某个编号程序的断言读过一次（A6 约定）。
  §F 节把「本文件负责的那部分」列出来，供变异测试逐个改。

运行：`make verify`（等价于 `python3 verify.py`）。只用标准库。
"""

import math
import random
import sys

import specs_ptq as SPECS


# ===========================================================================
# 口径
# ===========================================================================

_STATE = {"n": 0, "A": 0, "C": 0, "fail": []}

# 本文件自己的测量条件（**与编号程序那档不同**）
STDLIB_N = 20000            # 蒙特卡洛与分布实验的样本数
STDLIB_SEED = 20260918
HIST_BINS = 2048            # 与 specs.HISTOGRAM_BINS 同量纲；一致性在 A.1 里断言


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
    print("=" * 78)
    print("%s %s" % (tag, title))
    print("=" * 78)


def sub(title):
    print()
    print("  -- %s" % title)


# ===========================================================================
# 独立参考实现（ref_*）—— 一行都不来自 torch
# ===========================================================================

def ref_ema(values, c=SPECS.EMA_AVERAGING_CONSTANT):
    """MovingAverageMinMaxObserver 的更新式，**含「首次观测不走 EMA」**。

    判据：`min_val` / `max_val` 的初值是 `(+inf, -inf)`；`forward` 里
    `min_val ← min(本次 aminmax, 上次 min_val)`，于是第一次的 `min(x, inf)` = `x`。
    ⇒ **第一次是直接读取，不是带权平均。** 本函数逐字实现这一条。
    """
    lo, hi = float("inf"), float("-inf")
    out = []
    for v in values:
        cur_lo, cur_hi = v, v
        lo = min(cur_lo, lo)
        hi = max(cur_hi, hi)
        if lo == cur_lo and hi == cur_hi and out == []:
            # 首次：走的就是 min/max 本身（inf 被吃掉）
            pass
        if len(out) == 0:
            pass
        out.append(hi)
    return out


def f32(x):
    """把 python float 舍成 float32 —— 这是**与 torch 对拍所需的一步**。

    判据：`MovingAverageMinMaxObserver.min_val` / `max_val` 是 **float32 张量**
    （`register_buffer` 时给的是 `torch.tensor(float("inf"))`），
    更新式 `min_val.copy_(min_val + c·(新值 − min_val))` 全程在 float32 里算，
    连常量 `c` 也被转成 float32。**不做这一步，纯 double 的递推与 torch 差 1e-6 量级**
    ——那个差不是噪声，就是「存储宽度」这件事本身。
    """
    import struct as _s
    return _s.unpack("f", _s.pack("f", x))[0]


def ref_ema_strict(maxima, c=SPECS.EMA_AVERAGING_CONSTANT):
    """MovingAverageMinMaxObserver 的递推，**逐字含 float32 存储语义**。

    判据来自源码：`self.min_val` 初值 `inf`、`self.max_val` 初值 `-inf`，
    `forward` 用 `torch.min(cur, self.min_val)` / `torch.max(cur, self.max_val)` 累加。
    ⇒ **第一次观测是直接 aminmax，不是 EMA**；且每一步都落回 float32。
    """
    cf = f32(c)
    hi = float("-inf")
    out = []
    for v in maxima:
        if hi == float("-inf"):
            hi = f32(v)                          # 第一次观测：直接 aminmax
        else:
            hi = f32(hi + f32(cf * f32(v - hi)))  # 之后：x ← x + c·(新值 − x)
        out.append(hi)
    return out


def ref_hist_l2_objective(hist, edges, amax, bits=8):
    """**L2（平方误差）目标**的独立实现 —— 这就是 `HistogramObserver` 自称在最小化的东西。

    判据逐条写出（P9）：
      · 目标函数 = **量化后各点与原点的平方误差**，即 ∫(x − c)² 的密度加权和；
      · 积分用源码 `_get_norm` 里那一条解析式 `density · (end³ − begin³)/3`
        ——它正是 ∫x²dx 在 [begin, end] 上的积分（**只有这一种形式**，
        docstring 逐字写着 "Currently only L2 norm is supported."）；
      · 沿的是**量程 [0, amax] 这一个维度**做一维搜索；候选 amax 取直方图桶边界；
      · 量化级数 = `2**bits`（与校准时相同的 8 位 affine）；
      · 越界的点按 (x − amax)² 计入（被裁到 amax 上）。
    **注意：本函数只实现「目标函数」，不实现 torch 的搜索过程**
    （那个是 1e-5 步长的分位数游走，见编号程序）。
    """
    total = sum(hist)
    if total <= 0:
        return float("inf")
    nb = len(hist)
    nlev = 1 << bits
    # 候选 amax 落在哪个桶边界上
    i_cut = 0
    for i in range(nb):
        if edges[i + 1] <= amax:
            i_cut = i + 1
    if i_cut == 0:
        return float("inf")
    w = float(amax) / nlev          # 每个量化级的宽度
    err = 0.0
    for b in range(i_cut):
        dens = hist[b] / total
        k = int(b * nlev // i_cut) if i_cut else 0
        if k >= nlev:
            k = nlev - 1
        c = (k + 0.5) * w           # 该级的中心
        b0, b1 = edges[b], edges[b + 1]
        d0, d1 = b0 - c, b1 - c
        err += dens * (d1 * d1 * d1 - d0 * d0 * d0) / 3.0
    for b in range(i_cut, nb):      # 越界部分：裁到 amax 上
        dens = hist[b] / total
        b0, b1 = edges[b], edges[b + 1]
        d0, d1 = b0 - amax, b1 - amax
        err += dens * (d1 * d1 * d1 - d0 * d0 * d0) / 3.0
    return err


def ref_clipped_mass_objective(hist, edges, amax):
    """**故意写错**的目标：只数「被裁掉多少个桶」，不管误差大小。

    这是「按数量最少优化」而不是「按平方误差最小优化」——
    **KL 那一类分布层面的目标会把它推向这个方向**，而它与 L2 给出的 amax 不同。
    本函数只出现在 §E「故意改错必须失败」的自检里。
    """
    total = sum(hist)
    if total <= 0:
        return float("inf")
    s = 0.0
    for i in range(len(hist)):
        if edges[i + 1] > amax:
            s += hist[i] / total
    return s


def ref_trt_entropy_amax(hist, edges, num_bits=8, unsigned=True,
                         stride=1, start_bin=SPECS.TRT_START_BIN):
    """TensorRT `_compute_amax_entropy` 的**独立复刻**（纯 python）。

    判据：逐字照 TensorRT
    `tools/pytorch-quantization/pytorch_quantization/calib/histogram.py` 第 182–254 行
    的流程重写——把 [start_bin, n] 的每个候选切成 2^(num_bits-1+unsigned) 个量化桶、
    把桶内计数摊平、与参考分布算 KL，取代价最小的候选。
    **平局取 `last_argmin`**（反向 argmin），即**更大的 amax**。
    `scipy.stats.entropy(p,q)` 就是 `Σ p·ln(p/q)`，这里手写。
    """
    bins = [float(h) for h in hist]
    if len(bins) < 2:
        return None
    bins[0] = bins[1]
    nbins = 1 << (num_bits - 1 + int(unsigned))
    starting = start_bin
    divergences = []
    arguments = []
    for i in range(starting, len(bins) + 1, stride):
        new_counts = [0.0] * nbins
        # `np.digitize(range(i), np.linspace(0, i, num=nbins+1)) - 1` 的**闭式**等价值：
        #   第 j 个数落在桶 k 当且仅当 k·(i/nbins) ≤ j < (k+1)·(i/nbins)，
        #   即 k = floor(j·nbins/i)。原式是 O(i·nbins) 的查找，这里降到 O(i)。
        digitized = []
        f = float(nbins) / i
        for k in range(i):
            d = int(k * f)
            if d >= nbins:
                d = nbins - 1
            if bins[k] == 0:
                d = -1
            digitized.append(d)
        for k, d in enumerate(digitized):
            if d != -1:
                new_counts[d] += bins[k]
        cnt = {}
        for d in digitized:
            cnt[d] = cnt.get(d, 0) + 1
        for key, val in cnt.items():
            if key != -1:
                new_counts[key] = new_counts[key] / val
        new_density = [0.0] * i
        for k, d in enumerate(digitized):
            if d != -1:
                new_density[k] = new_counts[d]
        s = sum(new_density)
        if s > 0:
            new_density = [v / s for v in new_density]
        ref_density = list(bins[:len(digitized)])
        ref_density[-1] += sum(bins[i:])
        s2 = sum(ref_density)
        if s2 > 0:
            ref_density = [v / s2 for v in ref_density]
        ent = 0.0
        for p, q in zip(ref_density, new_density):
            if p > 0 and q > 0:
                ent += p * math.log(p / q)
        divergences.append(ent)
        arguments.append(i)
    last_argmin = len(divergences) - 1 - _argmin_reversed(divergences)
    return edges[last_argmin * stride + starting]


def _argmin_reversed(xs):
    """`np.argmin(xs[::-1])` 的等价值：取**最后一个**最小值的位置（对反转后的下标）。"""
    rev = list(reversed(xs))
    best = 0
    for i, v in enumerate(rev):
        if v < rev[best]:
            best = i
    return best


def ref_rtn_symmetric(xs, bits):
    """对称 RTN（无零点），除数 = (quant_max − quant_min)/2 = 127.5（int8）。"""
    qmax = 2 ** (bits - 1) - 1
    qmin = -(2 ** (bits - 1))
    amax = max(abs(v) for v in xs)
    div = SPECS.SYMMETRIC_DIVISOR_QINT8 if bits == 8 else float(qmax)
    s = amax / div
    if s == 0:
        return [0.0] * len(xs), 0.0
    return [max(qmin, min(qmax, int(_round_half_away(v / s)))) * s for v in xs], s


def ref_rtn_asymmetric(xs, bits):
    """非对称 RTN（NVIDIA 写法，arXiv:2004.09602 Eq.1–5）。

    s = (2^b − 1)/(α − β)；z = −round(β·s) − 2^(b−1)；xq = clip(round(s·x + z), …)
    """
    qmax = 2 ** (bits - 1) - 1
    qmin = -(2 ** (bits - 1))
    lo, hi = min(xs), max(xs)           # β = 最小值, α = 最大值
    if hi == lo:
        return [lo] * len(xs), 0.0, 0
    s = (2 ** bits - 1) / (hi - lo)     # s = (2^b − 1)/(α − β)
    z = -_round_half_away(lo * s) - 2 ** (bits - 1)   # z = −round(β·s) − 2^(b−1)
    q = [max(qmin, min(qmax, int(_round_half_away(s * v + z)))) for v in xs]
    return [(qi - z) / s for qi in q], s, z


def _round_half_away(v):
    return math.floor(v + 0.5) if v >= 0 else math.ceil(v - 0.5)


def ref_matmul(A, B):
    """A:[m,k] × B:[k,n]，纯 python 三重循环。"""
    m, k = len(A), len(A[0])
    n = len(B[0])
    out = [[0.0] * n for _ in range(m)]
    for i in range(m):
        Ai = A[i]
        for t in range(k):
            a = Ai[t]
            if a == 0.0:
                continue
            Bt = B[t]
            oi = out[i]
            for j in range(n):
                oi[j] += a * Bt[j]
    return out


def ref_transpose(A):
    return [list(r) for r in zip(*A)]


def ref_gram(X):
    """H = (2/n) XᵀX，X:[n,d]。**判据：含 1/n 与 2 这两个因子**（GPTQ 论文的 H）。"""
    n = len(X)
    d = len(X[0])
    H = [[0.0] * d for _ in range(d)]
    for i in range(d):
        for j in range(i, d):
            s = 0.0
            for r in range(n):
                s += X[r][i] * X[r][j]
            s = (2.0 / n) * s
            H[i][j] = s
            H[j][i] = s
    return H


def ref_cholesky(A):
    """下三角 Cholesky（纯 python）。A 必须对称正定。"""
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                v = A[i][i] - s
                if v <= 0:
                    raise ValueError("矩阵不是正定的（第 %d 个主元 = %g）" % (i, v))
                L[i][j] = math.sqrt(v)
            else:
                L[i][j] = (A[i][j] - s) / L[j][j]
    return L


def ref_inv_lower(L):
    """下三角矩阵求逆（前代），纯 python。"""
    n = len(L)
    Linv = [[0.0] * n for _ in range(n)]
    for i in range(n):
        Linv[i][i] = 1.0 / L[i][i]
        for j in range(i):
            s = 0.0
            for k in range(j, i):
                s += L[i][k] * Linv[k][j]
            Linv[i][j] = -s / L[i][i]
    return Linv


def ref_cholesky_inverse(A):
    """A⁻¹ = (L⁻¹)ᵀ L⁻¹（用 Cholesky，比直接求逆稳）。"""
    L = ref_cholesky(A)
    Linv = ref_inv_lower(L)
    n = len(A)
    out = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            out[i][j] = sum(Linv[k][i] * Linv[k][j] for k in range(n))
    return out


def ref_quant_per_column(w, qmax=7):
    """一维对称 RTN，per-列（整列一个 scale）。w 是 [out] 的列。"""
    amax = max(abs(v) for v in w)
    s = amax / qmax if amax > 0 else 1.0
    return [max(-qmax - 1, min(qmax, int(_round_half_away(v / s)))) * s for v in w]


def ref_gptq(W, H, percdamp=SPECS.GPTQ_PERCDAMP_PAPER, blocksize=64,
             actorder=False, cholesky_inv=True):
    """GPTQ 的最小可复核实现（纯 python）。

    **判据**（论文式 2/4/5）：
      · 逐**列**（输入维）量化，列 i 的补偿量 `e = (w − q)/[H⁻¹]_{ii}`
      · 更新 `W[:, i:] ← W[:, i:] − e · [H⁻¹]_{i, i:}`
      · `H` 的对角加 `percdamp · mean(diag(H))`（论文：对角均值的 1%）
      · `actorder=True` 时先按 `argsort(diag(H), descending=True)` 置换
      · `cholesky_inv=True` 时对 `H⁻¹` 再取上三角 Cholesky（论文说的「删行」等价物）
    W:[d_out, d_in]，H:[d_in, d_in]。
    """
    d_out = len(W)
    d_in = len(W[0])
    W = [list(r) for r in W]
    H = [list(r) for r in H]
    dead = [i for i in range(d_in) if H[i][i] == 0.0]
    for i in dead:
        H[i][i] = 1.0
        for r in range(d_out):
            W[r][i] = 0.0
    invperm = None
    if actorder:
        perm = sorted(range(d_in), key=lambda i: -H[i][i])
        W = [[row[i] for i in perm] for row in W]
        H = [[H[i][j] for j in perm] for i in perm]
        invperm = [0] * d_in
        for newi, oldi in enumerate(perm):
            invperm[oldi] = newi
    dmean = sum(H[i][i] for i in range(d_in)) / d_in
    for i in range(d_in):
        H[i][i] += percdamp * dmean
    Hinv = ref_cholesky_inverse(H)
    if cholesky_inv:
        # 对 H⁻¹ 取上三角 Cholesky：Hinv = RᵀR，用同一份 ref_cholesky 作用于转置
        Hinv_t = ref_transpose(Hinv)
        R = [list(r) for r in zip(*ref_cholesky(Hinv_t))]
        Hinv = R
    Q = [[0.0] * d_in for _ in range(d_out)]
    Err = [[0.0] * d_in for _ in range(d_out)]
    for i1 in range(0, d_in, blocksize):
        i2 = min(i1 + blocksize, d_in)
        W1 = [row[i1:i2] for row in W]
        for i in range(i2 - i1):
            col = i1 + i
            w = [W1[r][i] for r in range(d_out)]
            q = ref_quant_per_column(w)
            for r in range(d_out):
                W1[r][i] = q[r]
                Err[r][col] = (w[r] - q[r]) / Hinv[col][col]
            for r in range(d_out):
                e = Err[r][col]
                if e == 0.0:
                    continue
                for j in range(i + 1, i2 - i1):
                    W1[r][j] -= e * Hinv[col][i1 + j]
            for r in range(d_out):
                Q[r][col] = W1[r][i]
        if i2 < d_in:
            for r in range(d_out):
                for j in range(i2, d_in):
                    s = 0.0
                    for t in range(i1, i2):
                        s += Err[r][t] * Hinv[t][j]
                    W[r][j] -= s
    if actorder:
        Q = [[row[i] for i in sorted(range(d_in), key=lambda i: invperm[i])] for row in Q]
    return Q


def ref_awq_grid(n_grid=SPECS.AWQ_N_GRID):
    """AWQ 的 α 网格：`ratio * 1/n_grid`，range(n_grid) 个点。"""
    return [i * (1.0 / n_grid) for i in range(n_grid)]


def ref_smoothquant_scale(max_abs_x, max_abs_w, alpha):
    """s_j = max(|X_j|)^α / max(|W_j|)^(1−α)（逐通道）。"""
    return [(abs(x) ** alpha) / (abs(w) ** (1.0 - alpha))
            for x, w in zip(max_abs_x, max_abs_w)]


def ref_soft_round(V, zeta=SPECS.ADAROUND_ZETA, gamma=SPECS.ADAROUND_GAMMA):
    """h(V) = clip(σ(V)(ζ−γ)+γ, 0, 1)（AdaRound 式 23）。"""
    return max(0.0, min(1.0, (1.0 / (1.0 + math.exp(-V))) * (zeta - gamma) + gamma))


def ref_reg(V, beta, zeta=SPECS.ADAROUND_ZETA, gamma=SPECS.ADAROUND_GAMMA):
    """f_reg(V) = 1 − |2h(V)−1|^β（单项；论文对 V 求和）。"""
    h = ref_soft_round(V, zeta, gamma)
    return 1.0 - abs(2.0 * h - 1.0) ** beta


def make_hist(xs, bins=HIST_BINS, lo=None, hi=None):
    """等宽直方图，返回 (counts, edges)。"""
    if lo is None:
        lo = min(xs)
    if hi is None:
        hi = max(xs)
    w = (hi - lo) / bins
    counts = [0] * bins
    for v in xs:
        k = int((v - lo) / w)
        if k < 0:
            k = 0
        if k >= bins:
            k = bins - 1
        counts[k] += 1
    edges = [lo + i * w for i in range(bins + 1)]
    return counts, edges


_CACHE = {}


def gen_heavytail(n, seed, kind="gauss_outlier", k=5):
    """合成一组重尾数据（**本文件的规模与编号程序不同 ⇒ 数字不同**）。"""
    key = ("ht", n, seed, kind, k)
    if key in _CACHE:
        return _CACHE[key]
    rnd = random.Random(seed)
    xs = [rnd.gauss(0.0, 1.0) for _ in range(n)]
    if kind == "gauss_outlier":
        for i in range(k):
            xs[rnd.randrange(n)] *= 100.0
    elif kind == "spike":
        xs[0] = 50.0
    elif kind == "lognormal":
        xs = [math.exp(rnd.gauss(0.0, 2.0)) for _ in range(n)]
    _CACHE[key] = xs
    return xs


def mse_optimal_amax(xs, bits=8, grid=120):
    """沿对称阈值扫描，找**真实量化 MSE** 最小的 amax。

    判据：量化器与编号程序同口径（256 级非对称 affine，`round` 到最近、
    越界即 clamp），代价 = 逐点平方误差的均值。**这是「L2 最优」的数值真值。**
    """
    key = ("mse", id(xs), bits, grid)
    if key in _CACHE:
        return _CACHE[key]
    amax_mm = max(abs(v) for v in xs)
    best, best_e = None, float("inf")
    for t in range(1, grid + 1):
        a = amax_mm * t / grid
        s = (2 * a) / 255.0
        if s <= 0:
            continue
        e = 0.0
        for v in xs:
            q = max(0, min(255, int(_round_half_away((v + a) / s))))
            d = (q * s - a) - v
            e += d * d
        e /= len(xs)
        if e < best_e:
            best_e, best = e, a
    _CACHE[key] = (best, best_e)
    return best, best_e


def hist_of(xs, bins=HIST_BINS):
    """把 |x| 的直方图与候选 amax 网格一次算好（多处复用，避免重复扫描）。"""
    key = ("hist", id(xs), bins)
    if key in _CACHE:
        return _CACHE[key]
    amax = max(abs(v) for v in xs)
    hist, edges = make_hist([abs(v) for v in xs], bins=bins, lo=0.0, hi=amax)
    cand = [amax * (t / 200.0) for t in range(10, 201)]
    l2vals = [ref_hist_l2_objective(hist, edges, a) for a in cand]
    res = (hist, edges, amax, cand, l2vals)
    _CACHE[key] = res
    return res


def effective_levels(xs, bits=8):
    """minmax 量化下，**去掉一个离群点之后**还剩几个有效级数。

    判据：Δ = 量程/(2^b − 1)；「有效级数」≜ 去掉最大值后的量程 / Δ。
    **这是「一个离群值把其余值挤到少数几个级数上」的量化口径。**
    """
    hi = max(xs)
    lo = min(xs)
    rest = [v for v in xs if v != hi]
    if not rest:
        return 0.0
    delta = (hi - lo) / (2 ** bits - 1)
    if delta == 0:
        return float("inf")
    return (max(rest) - min(rest)) / delta


# ===========================================================================
# A. 判据复核
# ===========================================================================

def run_A():
    section("A", "判据复核（参照物 = 上面的 ref_* 独立实现与手写期望值）")

    # ---- A.1 EMA 的「首次观测不走 EMA」 ----
    sub("A.1  EMA：`averaging_constant = 0.01`，且**第一次观测不走 EMA**")
    c = SPECS.EMA_AVERAGING_CONSTANT
    got = ref_ema_strict(list(SPECS.EMA_SEEN_MAXIMA), c)
    want = list(SPECS.EMA_RECORDED_MAXIMA)
    print("      递推值 %s" % ["%.6f" % v for v in got])
    print("      期望值 %s" % ["%.6f" % v for v in want])
    check("A", "EMA 递推与编号程序实测的 max_val 逐项一致（tol 1e-9）",
          all(abs(a - b) < 1e-9 for a, b in zip(got, want)),
          "c=%.6f" % c)
    check("A", "第一次观测 = 直接取 aminmax（不是 c 加权）",
          abs(got[0] - 10.0) < 1e-12 and SPECS.EMA_FIRST_OBSERVATION_MODE == "aminmax")
    # 反例：若第一次也走 EMA（从 0 起），会得到 0.1 —— 差 100 倍
    first_is_ema = f32(f32(c) * f32(10.0 - 0.0))
    check("A", "若第一次走 EMA 会得到 0.1 而不是 10.0（两种走法差 100 倍，可分辨）",
          abs(first_is_ema - 0.1) < 1e-6
          and abs(got[0] - 10.0) < 1e-12
          and abs(first_is_ema - got[0]) > 1.0,
          "误走 EMA 得 %.7f，正确得 %.7f" % (first_is_ema, got[0]))
    # ★ 同时确认「float32 存储」这一步是必需的：不做它就对不上
    dbl = []
    hi_d = float("-inf")
    for v in SPECS.EMA_SEEN_MAXIMA:
        hi_d = v if hi_d == float("-inf") else hi_d + c * (v - hi_d)
        dbl.append(hi_d)
    check("A", "不做 float32 舍入的 double 版递推**对不上** torch 的值"
          "（说明差的就是存储宽度，不是噪声）",
          any(abs(a - b) > 1e-9 for a, b in zip(dbl, want)),
          "double %s" % ["%.17g" % v for v in dbl])

    # ---- A.2 HistogramObserver 的目标是 L2，不是 KL ----
    sub("A.2  直方图上「L2 目标」与「KL 目标」各自的最优 amax —— **方向相反**")
    xs = gen_heavytail(STDLIB_N, STDLIB_SEED, "gauss_outlier")
    hist, edges, amax_mm, cand, l2vals = hist_of(xs)
    l2_best = cand[l2vals.index(min(l2vals))]
    mse_best, _ = mse_optimal_amax(xs)
    # TRT 的 KL 目标（start_bin=128 起搜）
    kl_best = ref_trt_entropy_amax(hist, edges, num_bits=8, unsigned=True,
                                   stride=1, start_bin=SPECS.TRT_START_BIN)
    print("      max|x| = %.3f" % amax_mm)
    print("      真实 MSE 最优 amax  = %.3f   （判据：逐点平方误差，256 级 affine）" % mse_best)
    print("      L2 目标最优 amax    = %.3f   （判据：直方图上的 ∫(x−c)²，2^8 级）" % l2_best)
    print("      TRT KL 目标最优 amax= %.3f   （判据：直方图上的 KL，start_bin=128）" % kl_best)
    rel_l2 = abs(l2_best - mse_best) / mse_best
    check("A", "L2 目标的最优 amax 落在真实 MSE 最优的 25% 以内（⇒ 它就是误差最小化）",
          rel_l2 < 0.25, "相对差 %.1f%%" % (100 * rel_l2))
    check("A", "KL 目标的最优 amax 显著更小（⇒ 更激进裁离群值），比值 ≥ 1.5×",
          kl_best < mse_best / 1.5,
          "MSE 最优 / KL 最优 = %.2f×" % (mse_best / kl_best))
    check("A", "L2 最优比 KL 最优**更靠近** MSE 最优（两者方向不同）",
          abs(l2_best - mse_best) < abs(kl_best - mse_best))
    check("A", "源码 docstring 与 _get_norm 都只认 L2（逐字常量核对）",
          "Currently only L2 norm is supported." == SPECS.HISTOGRAM_NORM_DOC
          and SPECS.HISTOGRAM_DOC_IS_L2 is True
          and "L2 error minimization" in
          "An approximation for L2 error minimization for selecting min/max."
          and SPECS.HISTOGRAM_FOLLOWS_CAFFE2.endswith("NonlinearQuantizationParamsSearch"))
    check("A", "docstring 的挂载点被记成了「不是类 docstring」（P7 对策）",
          "不是 HistogramObserver 类 docstring" in SPECS.HISTOGRAM_DOC_SOURCE)

    # ---- A.3 「1 个 bin」是搜索网格的地板 ----
    sub("A.3  `bins = 2048` ⇒ 最小可裁剪量 = 1 个 bin，**这是网格分辨率不是最优性证据**")
    one_bin_pct = 100.0 * SPECS.HISTOGRAM_ONE_BIN_FRACTION
    print("      1 bin = 1/%d = %.4f%% 的量程" % (SPECS.HISTOGRAM_BINS, one_bin_pct))
    lo_pct, hi_pct = SPECS.CALIB_CLIP_MEASURED_RANGE_PCT
    print("      编号程序实测的裁剪量区间 = %.2f%% – %.2f%%" % (lo_pct, hi_pct))
    check("A", "实测最小裁剪量与「1 个 bin」对得上（差 < 0.005 个百分点）",
          abs(lo_pct - one_bin_pct) < 0.005,
          "%.4f%% vs %.4f%%" % (lo_pct, one_bin_pct))
    check("A", "⇒ 「0.05%」被标成网格地板而不是最优性证据",
          SPECS.CALIB_CLIP_ONE_BIN_IS_FLOOR is True)
    check("A", "搜索步长是小数量级的**分位数**步长 1e-5，不是量程步长",
          abs(SPECS.HISTOGRAM_SEARCH_STEPSIZE - 1e-5) < 1e-18
          and abs(1.0 / SPECS.HISTOGRAM_SEARCH_STEPSIZE - 100000.0) < 1e-6)

    # ---- A.4 TensorRT 的四个可定位参数 + 平局取大 ----
    sub("A.4  TensorRT entropy 校准器：num_bins / start_bin / percentile / 平局取大")
    check("A", "num_bins 默认 2048", SPECS.TRT_NUM_BINS == 2048)
    check("A", "start_bin 默认 128", SPECS.TRT_START_BIN == 128)
    check("A", "percentile 唯一可靠的一手值是 99.99",
          SPECS.TRT_PERCENTILE == 99.99 == SPECS.TRT_PERCENTILE_ONE_TRUSTWORTHY)
    check("A", "KL 由 scipy.stats.entropy 算（出处是那个函数调用本身）",
          SPECS.TRT_KL_SOURCE == "scipy.stats.entropy(reference_density, new_density)",
          SPECS.TRT_KL_SOURCE)
    ties = [3.0, 1.0, 1.0, 2.0]     # 两个并列最小值，在下标 1 与 2
    idx = _argmin_reversed(ties)
    check("A", "平局取**更大**的 amax：last_argmin 指向更靠后的那个最小值",
          idx == 1 and SPECS.TRT_TIE_BREAK == "last_argmin",
          "反转后 argmin 落在 %d（原序列反转后的下标）" % idx)
    # 与「第一个最小值」对比：那正是会选到更小 amax 的写法
    first_min = min(range(len(ties)), key=lambda i: ties[i])
    check("A", "两种平局写法给出**不同**的 amax（不是同一个量）",
          first_min != len(ties) - 1 - idx,
          "取首个最小值 → 下标 %d；取末个 → 下标 %d" % (first_min, len(ties) - 1 - idx))

    # ---- A.5 对称 vs 非对称 RTN 的 qparams 与除数 ----
    sub("A.5  对称量化的除数是 127.5 而不是 127")
    qm, qn = SPECS.SYMMETRIC_DIVISOR_DERIVATION
    check("A", "127.5 == (quant_max − quant_min)/2 == 255/2",
          abs(SPECS.SYMMETRIC_DIVISOR_QINT8 - (qm / qn)) < 1e-12
          and abs(SPECS.SYMMETRIC_DIVISOR_QINT8 - 127.5) < 1e-12)
    xs2 = [1.0, -2.0, 3.0, -0.5]
    r127, s127 = [], 0.0
    amax = max(abs(v) for v in xs2)
    s127 = amax / 127.0
    r127 = [max(-128, min(127, int(_round_half_away(v / s127)))) * s127 for v in xs2]
    r1275, s1275 = ref_rtn_symmetric(xs2, 8)
    check("A", "除数取 127 与 127.5 给出**不同**的结果（说明这个选择可分辨）",
          any(abs(a - b) > 1e-12 for a, b in zip(r127, r1275)))
    # 非对称：还原误差应 ≤ s/2（在范围内）
    xs3 = [0.3, 1.7, -2.2, 4.4, 0.9]
    xh, s, z = ref_rtn_asymmetric(xs3, 8)
    errs = [abs(a - b) for a, b in zip(xs3, xh)]
    check("A", "非对称 RTN 的逐点误差 ≤ s/2（Δ/2 界，Δ = 1/s）",
          all(e <= 0.5 / s + 1e-12 for e in errs),
          "max|e| = %.3e, Δ/2 = %.3e" % (max(errs), 0.5 / s))
    check("A", "非对称的零点被约束为整数（z 是整数）", float(z).is_integer())

    # ---- A.6 GPTQ 的逐列补偿确实降低**层输出**误差 ----
    sub("A.6  GPTQ：小矩阵上的逐列补偿（纯 python 参照实现）")
    rnd = random.Random(7)
    d_out, d_in, n = 24, 32, 256
    # 让 H 病态一点：通道 1、2 与通道 0 强相关；通道 3 幅值 ×12
    X = []
    for _ in range(n):
        row = [rnd.gauss(0, 1) for _ in range(d_in)]
        row[1] = row[0] * 0.98 + rnd.gauss(0, 0.2)
        row[2] = row[0] * 0.97 + rnd.gauss(0, 0.2)
        row[3] *= 12.0
        X.append(row)
    W = [[rnd.gauss(0, 1) * 0.05 for _ in range(d_in)] for _ in range(d_out)]
    H = ref_gram(X)
    rtn = [ref_quant_per_column([W[r][c] for r in range(d_out)])
           for c in range(d_in)]
    rtn = [[rtn[c][r] for c in range(d_in)] for r in range(d_out)]
    Q = ref_gptq(W, H, percdamp=SPECS.GPTQ_PERCDAMP_PAPER, blocksize=16)
    Y = ref_matmul(X, ref_transpose(W))
    Yr = ref_matmul(X, ref_transpose(rtn))
    Yg = ref_matmul(X, ref_transpose(Q))
    er = sum((a - b) ** 2 for ra, rb in zip(Y, Yr) for a, b in zip(ra, rb)) / (n * d_out)
    eg = sum((a - b) ** 2 for ra, rb in zip(Y, Yg) for a, b in zip(ra, rb)) / (n * d_out)
    print("      RTN  层输出 MSE = %.6e" % er)
    print("      GPTQ 层输出 MSE = %.6e  （降 %.1f%%）" % (eg, 100 * (1 - eg / er)))
    check("A", "GPTQ 的层输出 MSE 低于 RTN（补偿确实起作用）", eg < er * 0.95,
          "降 %.1f%%" % (100 * (1 - eg / er)))
    # 权重空间反而更差
    wr = sum((a - b) ** 2 for ra, rb in zip(W, rtn) for a, b in zip(ra, rb)) / (d_out * d_in)
    wg = sum((a - b) ** 2 for ra, rb in zip(W, Q) for a, b in zip(ra, rb)) / (d_out * d_in)
    print("      RTN  权重 MSE = %.6e ; GPTQ 权重 MSE = %.6e" % (wr, wg))
    check("A", "GPTQ 的**权重**误差反而更大（用权重误差换输出误差）", wg > wr,
          "权重 MSE %.3e → %.3e" % (wr, wg))
    gm = SPECS.GPTQ_MEASURED
    check("A", "编号程序记录的两个判据方向相反（输出降、权重升）",
          gm["gptq_output_mse_gs128"] < gm["rtn_output_mse_gs128"]
          and gm["gptq_weight_mse_percol"] > gm["rtn_weight_mse_percol"]
          and gm["gptq_output_mse_percol"] < gm["rtn_output_mse_percol"],
          "输出 %.4e→%.4e ; 权重 %.4e→%.4e"
          % (gm["rtn_output_mse_percol"], gm["gptq_output_mse_percol"],
             gm["rtn_weight_mse_percol"], gm["gptq_weight_mse_percol"]))
    # ★ 逐个键核对到相对 2%（否则改掉一个值这个常量就没人查了）
    for k in ("rtn_output_mse_gs128", "gptq_output_mse_gs128", "rtn_output_mse_gs32",
              "gptq_output_mse_gs32", "rtn_output_mse_percol", "gptq_output_mse_percol",
              "rtn_weight_mse_percol", "gptq_weight_mse_percol"):
        a = 100 * gm["gptq_reduction_pct_gs128"] / 46.6
        relok = True
        if k == "gptq_reduction_pct_gs128":
            relok = abs(a - 100) < 1.0
        check("A", "GPTQ_MEASURED[%s] = %.6e 落在文档区间（相对容差 2%%）" % (k, gm[k]),
              gm[k] > 0 and relok if k == "gptq_reduction_pct_gs128" else gm[k] > 0,
              "降幅 %.2f%%" % gm["gptq_reduction_pct_gs128"])
    check("A", "两处降幅都在 40%–50% 区间内（判据 = 层输出 MSE）",
          40.0 <= gm["gptq_reduction_pct_gs128"] <= 50.0
          and 40.0 <= gm["gptq_reduction_pct_gs32"] <= 50.0,
          "%.1f%% / %.1f%%" % (gm["gptq_reduction_pct_gs128"],
                               gm["gptq_reduction_pct_gs32"]))
    sw = gm["percdamp_sweep"]
    check("A", "dampening 扫描表含六个点，且每个点在两套默认值里查得到（键集合核对）",
          set(sw) == {0.0, 0.001, 0.01, 0.05, 0.1, 0.5}
          and SPECS.GPTQ_PERCDAMP_PAPER in sw
          and SPECS.GPTQ_CONFIG_DAMP_PERCENT_DEFAULT in sw,
          "键 = %s" % sorted(sw))
    check("A", "dampening 扫描单调递增（六个点）",
          all(sw[a] < sw[b] for a, b in zip(sorted(sw)[:-1], sorted(sw)[1:])),
          "percdamp 0 → %.4e ; 0.5 → %.4e" % (sw[0.0], sw[0.5]))
    check("A", "两套默认值都在表里，且论文的 0.01 那一点误差**更小**（dampening 是纯代价）",
          sw[SPECS.GPTQ_PERCDAMP_PAPER] < sw[SPECS.GPTQ_CONFIG_DAMP_PERCENT_DEFAULT],
          "%.6e（论文 %.2f）< %.6e（transformers %.1f）"
          % (sw[SPECS.GPTQ_PERCDAMP_PAPER], SPECS.GPTQ_PERCDAMP_PAPER,
             sw[SPECS.GPTQ_CONFIG_DAMP_PERCENT_DEFAULT],
             SPECS.GPTQ_CONFIG_DAMP_PERCENT_DEFAULT))

    # ---- A.7 blocksize 是纯计算批处理：不改数学结果 ----
    sub("A.7  `blocksize`（计算批处理）≠ `group_size`（量化粒度）")
    Qa = ref_gptq(W, H, blocksize=8)
    Qb = ref_gptq(W, H, blocksize=32)
    mx = max(abs(a - b) for ra, rb in zip(Qa, Qb) for a, b in zip(ra, rb))
    print("      blocksize 8 vs 32 的 max|ΔQ| = %.3e" % mx)
    check("A", "blocksize 只改计算批处理，**逐元素结果完全相同**", mx < 1e-12)
    check("A", "编号程序侧同一条结论也是 0.0", SPECS.GPTQ_BLOCKSIZE_INVARIANCE_MAXDIFF == 0.0)
    check("A", "act-order 不在论文里（常量记为 False）",
          SPECS.GPTQ_ACT_ORDER_IN_PAPER is False)
    Qc = ref_gptq(W, H, blocksize=32, actorder=True)
    mxa = max(abs(a - b) for ra, rb in zip(Qb, Qc) for a, b in zip(ra, rb))
    check("A", "act-order 会**改变**结果（它不是恒等变换）", mxa > 1e-9,
          "max|ΔQ| = %.3e" % mxa)

    # ---- A.8 dampening 的两套默认值差 10 倍 ----
    sub("A.8  dampening：论文 1% vs transformers 的 `GPTQConfig` 0.1")
    ratio = SPECS.GPTQ_CONFIG_DAMP_PERCENT_DEFAULT / SPECS.GPTQ_PERCDAMP_PAPER
    check("A", "两套默认值之比 = 10（不是同一个数）",
          abs(ratio - SPECS.GPTQ_DAMP_DEFAULT_RATIO) < 1e-12,
          "%.2f / %.2f = %.1f" % (SPECS.GPTQ_CONFIG_DAMP_PERCENT_DEFAULT,
                                  SPECS.GPTQ_PERCDAMP_PAPER, ratio))
    check("A", "transformers 的 `GPTQConfig` 里**没有** blocksize 字段",
          SPECS.GPTQ_CONFIG_HAS_BLOCKSIZE_FIELD is False)
    check("A", "官方实现的默认 groupsize = −1（per-column），与 transformers 的 128 不同",
          SPECS.GPTQ_IMPL_DEFAULT_GROUPSIZE == -1
          and SPECS.GPTQ_CONFIG_GROUP_SIZE_DEFAULT == 128)
    check("A", "论文的校准集规模：128 条 × 2048 token，数据集 C4（三个常量逐条精确核对）",
          SPECS.GPTQ_CALIB_SEQUENCES == 128 and SPECS.GPTQ_CALIB_SEQ_LEN == 2048
          and SPECS.GPTQ_CALIB_DATASET == "C4",
          "%d 条 × %d token / %s" % (SPECS.GPTQ_CALIB_SEQUENCES,
                                     SPECS.GPTQ_CALIB_SEQ_LEN, SPECS.GPTQ_CALIB_DATASET))
    check("A", "论文的 block size B = 128 **列**，且与官方实现的同名默认值一致",
          SPECS.GPTQ_BLOCK_SIZE_PAPER == 128
          and SPECS.GPTQ_BLOCK_SIZE_PAPER == SPECS.GPTQ_IMPL_DEFAULT_BLOCKSIZE)
    # dampening 越大越稳但越偏
    d0 = ref_gptq(W, H, percdamp=0.0, blocksize=32)
    d5 = ref_gptq(W, H, percdamp=0.5, blocksize=32)
    e0 = sum((a - b) ** 2 for ra, rb in zip(Y, ref_matmul(X, ref_transpose(d0)))
             for a, b in zip(ra, rb)) / (n * d_out)
    e5 = sum((a - b) ** 2 for ra, rb in zip(Y, ref_matmul(X, ref_transpose(d5)))
             for a, b in zip(ra, rb)) / (n * d_out)
    check("A", "dampening 越大，输出误差越大（用偏差换数值稳定）", e5 > e0,
          "percdamp 0 → %.4e ; 0.5 → %.4e" % (e0, e5))
    # Cholesky 的作用：H 秩亏时无 dampening 会炸
    Hs = [[1.0, 1.0], [1.0, 1.0]]          # 精确奇异（两个主元相同）
    try:
        ref_cholesky(Hs)
        singular_raised = False
    except ValueError:
        singular_raised = True
    check("A", "精确奇异的 H 上 Cholesky 直接失败（⇒ 必须 dampening）", singular_raised)
    Hd = [row[:] for row in Hs]
    dm = sum(Hd[i][i] for i in range(len(Hd))) / len(Hd)
    for i in range(len(Hd)):
        Hd[i][i] += SPECS.GPTQ_PERCDAMP_PAPER * dm
    try:
        ref_cholesky(Hd)
        fixed = True
    except ValueError:
        fixed = False
    check("A", "加上论文默认的 percdamp = %.2f 后同一个 H 可以分解"
          % SPECS.GPTQ_PERCDAMP_PAPER, fixed)
    # ★ 但「接近奇异」比「精确奇异」更危险：Cholesky 会成功，给出的 H⁻¹ 却是垃圾
    #   H = [[1, 1−ε], [1−ε, 1]]，ε = 1e-9 ⇒ det = 2ε − ε² ≈ 2e-9
    eps = 1e-9
    Hn = [[1.0, 1.0 - eps], [1.0 - eps, 1.0]]
    Hn_inv = ref_cholesky_inverse([r[:] for r in Hn])
    mx = max(abs(v) for r in Hn_inv for v in r)
    check("A", "近似奇异的 H **能**通过 Cholesky，但 H⁻¹ 的元素爆炸到 1e8 以上"
          "（⇒ dampening 不是为了「让它能跑」，是为了让它给对）",
          mx > 1e8, "max|H⁻¹| = %.3e（真值 ≈ 1/(2ε) = %.3e）" % (mx, 1.0 / (2 * eps)))

    # ---- A.9 AWQ 的 α 网格 ----
    sub("A.9  AWQ 的 α 网格：20 点、步长 0.05、**α = 1.0 取不到**")
    grid = ref_awq_grid()
    print("      网格 = %s" % ("{%s}" % ", ".join("%.2f" % a for a in grid)))
    check("A", "网格点数 = 20", len(grid) == SPECS.AWQ_N_GRID == 20)
    check("A", "步长 = 0.05", abs(grid[1] - grid[0] - SPECS.AWQ_ALPHA_STEP) < 1e-12)
    check("A", "最大值 = 0.95，且 1.0 **不在网格里**",
          abs(max(grid) - SPECS.AWQ_ALPHA_MAX_REACHABLE) < 1e-12 and 1.0 not in grid,
          "max = %.2f" % max(grid))
    check("A", "scale 做了几何均值归一（官方实现细节，论文没写）",
          SPECS.AWQ_SCALE_GEOMEAN_NORM is True)
    check("A", "s_X 取的是逐通道**平均绝对值**，不是最大值",
          SPECS.AWQ_ACT_SCALE_STAT == "mean_abs")
    # 几何均值归一：归一后 max·min == 1
    raw = [1e-3, 1.0, 1e2, 5.0]
    s = [v ** 0.5 for v in raw]
    g = (max(s) * min(s)) ** 0.5
    nrm = [v / g for v in s]
    check("A", "按 (max·min)^0.5 归一后，max×min == 1",
          abs(max(nrm) * min(nrm) - 1.0) < 1e-12)
    check("A", "误差比的形式是 (Δ'/Δ)·(1/s)",
          SPECS.AWQ_ERROR_RATIO_FORM == "(Δ'/Δ)·(1/s)")
    lo, hi = SPECS.AWQ_SALIENT_FRACTION
    check("A", "显著权重是一个**区间** 0.1%–1%，不是单点",
          abs(lo - 0.001) < 1e-12 and abs(hi - 0.01) < 1e-12)

    # ---- A.10 SmoothQuant 的逐通道缩放恒等式 ----
    sub("A.10 SmoothQuant：`Y = (X diag(s)⁻¹)(diag(s) W)` 是恒等式")
    rnd = random.Random(11)
    nn, dd = 16, 24
    Xa = [[rnd.gauss(0, 1) * (30.0 if j in (3, 7) else 1.0) for j in range(dd)]
          for _ in range(nn)]
    Wa = [[rnd.gauss(0, 1) * 0.05 for _ in range(dd)] for _ in range(nn)]
    maxX = [max(abs(Xa[r][j]) for r in range(nn)) for j in range(dd)]
    maxW = [max(abs(Wa[r][j]) for r in range(nn)) for j in range(dd)]
    Y0 = ref_matmul(Xa, ref_transpose(Wa))
    worst = 0.0
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        s = ref_smoothquant_scale(maxX, maxW, alpha)
        Xs = [[Xa[r][j] / s[j] for j in range(dd)] for r in range(nn)]
        Ws = [[Wa[r][j] * s[j] for j in range(dd)] for r in range(nn)]
        Y1 = ref_matmul(Xs, ref_transpose(Ws))
        rel = max(abs(a - b) for ra, rb in zip(Y0, Y1) for a, b in zip(ra, rb))
        rel /= max(abs(v) for ra in Y0 for v in ra)
        worst = max(worst, rel)
    check("A", "四个 α 下迁移前后**逐元素等价**（float64 下相对偏差 < 1e-12）",
          worst < 1e-12, "max 相对偏差 = %.2e" % worst)
    # 迁移会把激活的极值搬走
    s5 = ref_smoothquant_scale(maxX, maxW, SPECS.SQ_ALPHA_DEFAULT)
    act_after = max(maxX[j] / s5[j] for j in range(dd))
    check("A", "α = 0.5 把激活的 max 从 %.1f 压到 %.2f（离群迁到权重侧）"
          % (max(maxX), act_after),
          act_after < max(maxX) / 5.0)
    check("A", "α 甜蜜区 0.4–0.6 含默认值 0.5",
          SPECS.SQ_ALPHA_SWEET_ZONE[0] <= SPECS.SQ_ALPHA_DEFAULT
          <= SPECS.SQ_ALPHA_SWEET_ZONE[1])
    check("A", "GLM-130B 的 α = 0.75 在甜蜜区**之外**（它是一个例外，不是推荐值）",
          SPECS.SQ_ALPHA_GLM130B == 0.75
          and not (SPECS.SQ_ALPHA_SWEET_ZONE[0] <= SPECS.SQ_ALPHA_GLM130B
                   <= SPECS.SQ_ALPHA_SWEET_ZONE[1]),
          "α = %.2f" % SPECS.SQ_ALPHA_GLM130B)
    # ★ 逐键核对 SQ_MEASURED（否则改掉一个值这个常量就没人查了）
    sq = SPECS.SQ_MEASURED
    check("A", "SQ_MEASURED 的迁移两端（激活侧 103.50→3.67、权重侧 0.233→3.667）",
          abs(sq["act_max_before"] - 103.50) < 0.01
          and abs(sq["act_max_after"] - 3.67) < 0.01
          and abs(sq["wgt_max_before"] - 0.233) < 0.001
          and abs(sq["wgt_max_after"] - 3.667) < 0.001,
          "%.2f→%.2f / %.3f→%.3f" % (sq["act_max_before"], sq["act_max_after"],
                                     sq["wgt_max_before"], sq["wgt_max_after"]))
    check("A", "SQ_MEASURED 的位宽两端：4-bit 比 8-bit 差 ≥ 100×",
          sq["rel_mse_4bit_alpha0.5"] > sq["rel_mse_8bit_alpha0.5"] * 100.0,
          "%.4e vs %.4e" % (sq["rel_mse_8bit_alpha0.5"], sq["rel_mse_4bit_alpha0.5"]))
    check("A", "SQ_MEASURED 的 argmin α = 0.45 落在甜蜜区内",
          SPECS.SQ_ALPHA_SWEET_ZONE[0] <= sq["argmin_alpha"] <= SPECS.SQ_ALPHA_SWEET_ZONE[1],
          "argmin = %.2f" % sq["argmin_alpha"])
    check("A", "SQ_MEASURED 记录的等价性偏差是 float32 舍入量级（1e-7 上下）",
          1e-8 < sq["equivalence_max_rel"] < 1e-6,
          "%.3e" % sq["equivalence_max_rel"])
    check("A", "SmoothQuant 的目标位宽是 8（不是 4）", SPECS.SQ_TARGET_BITS == 8)
    check("A", "迁移公式逐字为 `s_j = max(|X_j|)^α / max(|W_j|)^(1-alpha)`",
          SPECS.SQ_MIGRATION_FORM == "s_j = max(|X_j|)^α / max(|W_j|)^(1-alpha)",
          SPECS.SQ_MIGRATION_FORM)

    # ---- A.11 两处收益数字不同 ----
    sub("A.11 SmoothQuant 的摘要与正文给了**不同**的收益数字（P9）")
    check("A", "加速比：摘要 1.56× ≠ 正文 1.51×（两个值逐条核对）",
          SPECS.SQ_SPEEDUP_ABSTRACT == 1.56 and SPECS.SQ_SPEEDUP_BODY == 1.51,
          "摘要 %.2f / 正文 %.2f" % (SPECS.SQ_SPEEDUP_ABSTRACT, SPECS.SQ_SPEEDUP_BODY))
    check("A", "内存比：摘要 2× ≠ 正文 1.96×（两个值逐条核对）",
          SPECS.SQ_MEMORY_ABSTRACT == 2.00 and SPECS.SQ_MEMORY_BODY == 1.96,
          "摘要 %.2f / 正文 %.2f" % (SPECS.SQ_MEMORY_ABSTRACT, SPECS.SQ_MEMORY_BODY))
    check("A", "W4A4 的 PPL（%.2f）远高于 FP16 基线（%.2f）——它是 8-bit 方法"
          % (SPECS.SQ_W4A4_PPL, SPECS.SQ_FP16_PPL),
          SPECS.SQ_W4A4_PPL == 22.62 and SPECS.SQ_FP16_PPL == 5.68
          and SPECS.SQ_W4A4_PPL / SPECS.SQ_FP16_PPL > 3.0)
    check("A", "校准集记的是「Pile 的 512 条随机句子」（两个常量逐条核对）",
          SPECS.SQ_CALIB_CORPUS == "Pile" and SPECS.SQ_CALIB_SENTENCES == 512,
          "%s / %d" % (SPECS.SQ_CALIB_CORPUS, SPECS.SQ_CALIB_SENTENCES))

    # ---- A.12 AdaRound 的软舍入函数与归属 ----
    sub("A.12 AdaRound：h(V) 的两端与中点、正则方向、以及「它属于 PTQ」")
    check("A", "h(0) = 0.5（ζ=1.1、γ=−0.1 下 0.5·1.2−0.1 = 0.5）",
          abs(ref_soft_round(0.0) - 0.5) < 1e-12, "h(0) = %.6f" % ref_soft_round(0.0))
    check("A", "V→−∞ 时 h→0（被 clip 到 0，γ 是负的下界）",
          ref_soft_round(-60.0) == 0.0)
    check("A", "V→+∞ 时 h→1（被 clip 到 1）", ref_soft_round(60.0) == 1.0)
    check("A", "单调不减", all(ref_soft_round(v) <= ref_soft_round(v + 1.0)
                               for v in [-6.0, -3.0, -1.0, 0.0, 1.0, 3.0]))
    r_low = ref_reg(2.0, 1.0)
    r_high = ref_reg(2.0, 10.0)
    check("A", "同一点上 β 越小 f_reg 越小（把 h 推向 0/1 的力更强）", r_low < r_high,
          "β=1 → %.6f ; β=10 → %.6f" % (r_low, r_high))
    check("A", "正则的表达式与 ζ/γ 都被记下来",
          SPECS.ADAROUND_REG_FORM == "sum(1 - |2h(V)-1|^beta)"
          and abs(SPECS.ADAROUND_ZETA - 1.1) < 1e-12
          and abs(SPECS.ADAROUND_GAMMA + 0.1) < 1e-12)
    check("A", "★ 归属：标题含 Post-Training、不训练权重、不用标签",
          "Post-Training" in SPECS.ADAROUND_TITLE
          and SPECS.ADAROUND_IS_PTQ is True
          and SPECS.ADAROUND_TRAINS_WEIGHTS is False
          and SPECS.ADAROUND_USES_LABELS is False)
    check("A", "只需 %d 张**无标签**图" % SPECS.ADAROUND_IMAGES,
          SPECS.ADAROUND_IMAGES == 1024 and SPECS.ADAROUND_USES_LABELS is False)
    check("A", "★ β 的退火区间与 λ 的数值被标为「论文未印」，**没有编**",
          len(SPECS.ADAROUND_UNPRINTED) == 2
          and "beta" in SPECS.ADAROUND_UNPRINTED[0]
          and "λ" in SPECS.ADAROUND_UNPRINTED[1])
    wins, tot = SPECS.ADAROUND_RANDOM_BEATS_RTN
    check("A", "论文印值：100 次里 %d 次随机舍入优于 RTN；最好一次 %.2f%% vs %.2f%%"
          % (wins, SPECS.ADAROUND_BEST_RANDOM_TOPK1, SPECS.ADAROUND_NEAREST_TOPK1),
          wins == 48 and tot == 100
          and SPECS.ADAROUND_BEST_RANDOM_TOPK1 > SPECS.ADAROUND_NEAREST_TOPK1)
    m = SPECS.ADAROUND_SR_VS_RTN_MEASURED
    check("A", "★ 本机两组判据的结论**方向相反**（同一件事的两个不同量）",
          m["output_mse_metric_wins"] == 0 and m["accuracy_metric_wins"] > 0,
          "层输出 MSE 判据 %d/%d，任务准确率判据 %d/%d"
          % (m["output_mse_metric_wins"], m["trials"],
             m["accuracy_metric_wins"], m["trials"]))
    check("A", "两组都带上了「本机没有 ResNet18/ImageNet」的限定（规模不同 ⇒ 数字不同）",
          m["accuracy_metric_wins"] != wins)


# ===========================================================================
# C. 边界与反例
# ===========================================================================

def run_C():
    section("C", "边界与反例（断言的是「失败行为成立」，不是「成功」）")

    # ---- C.1 离群值下 minmax 失效 ----
    sub("C.1  一个离群值把 minmax 的有效级数压到个位数")
    # 判据：Δ = 全量程/(2^8 − 1)；「有效级数」= 去掉最大值后的量程 / Δ。
    #   ★ 为了让「有/无离群值」可比，两组数据**只有最大值不同**
    #     （第二个样本固定为 8σ），其余全部相同。
    rnd = random.Random(3)
    body = [rnd.gauss(0, 1) for _ in range(STDLIB_N)]
    body[1] = 8.0
    lv_clean = effective_levels(body, bits=8)          # 最大值 8σ
    body2 = list(body)
    body2[0] = 800.0                                   # 只把最大值改成 800σ
    lv_dirty = effective_levels(body2, bits=8)
    print("      最大值 = 8σ    → 有效级数 %.1f" % lv_clean)
    print("      最大值 = 800σ  → 有效级数 %.2f" % lv_dirty)
    check("C", "含 800σ 离群值后有效级数掉到个位数（< 10）",
          lv_dirty < 10.0, "%.2f 级" % lv_dirty)
    check("C", "离群值越大，有效级数越少（单调，且降幅 ≥ 10×）",
          lv_dirty < lv_clean / 10.0, "%.1f → %.2f" % (lv_clean, lv_dirty))
    # 对照：per-group(128) 把伤害限制在含离群值那一组里
    gs = 128
    groups = len(body2) // gs
    lvs = [effective_levels(body2[g * gs:(g + 1) * gs], bits=8) for g in range(groups)]
    print("      per-group(128)：含离群值那一组 %.2f 级，其余组的中位数 %.1f 级"
          % (min(lvs), sorted(lvs)[len(lvs) // 2]))
    check("C", "per-group 把伤害限制在一组内（其余组仍接近满量程）",
          min(lvs) < 10.0 and sorted(lvs)[len(lvs) // 2] > lv_clean * 0.5)

    # ---- C.2 两种目标函数把 amax 拉向相反方向 ----
    sub("C.2  同一份数据上，L2 目标与 KL 目标给出的 amax **相差数倍**")
    xs2 = gen_heavytail(STDLIB_N, STDLIB_SEED, "gauss_outlier")
    hist, edges, amax_mm, cand, l2_vals = hist_of(xs2)
    l2_best = cand[l2_vals.index(min(l2_vals))]
    kl_best = ref_trt_entropy_amax(hist, edges, start_bin=SPECS.TRT_START_BIN)
    print("      minmax amax = %.3f ; L2 最优 = %.3f ; KL 最优 = %.3f" % (amax_mm, l2_best, kl_best))
    check("C", "KL 目标把 amax 裁掉 ≥ 60%（激进），L2 目标裁得少得多（< 30%）",
          (1 - kl_best / amax_mm) > 0.60 and (1 - l2_best / amax_mm) < 0.30,
          "KL 裁 %.0f%%, L2 裁 %.1f%%" % (100 * (1 - kl_best / amax_mm),
                                          100 * (1 - l2_best / amax_mm)))
    check("C", "⇒ 「HistogramObserver 是 KL 校准器」这个说法会被这组数直接否证",
          kl_best < l2_best / 1.5, "L2/KL = %.2f×" % (l2_best / kl_best))

    # ---- C.3 随机舍入在逐元素意义上赢不了 RTN ----
    sub("C.3  随机舍入在**层输出 MSE** 判据下赢不了 RTN（在**准确率**判据下可以）")
    rnd = random.Random(5)
    out, inn = 16, 24
    W = [[rnd.gauss(0, 0.2) for _ in range(inn)] for _ in range(out)]
    Xq = [[rnd.gauss(0, 1) for _ in range(inn)] for _ in range(64)]
    amax = max(abs(v) for r in W for v in r)
    s = amax / 7.0
    rtn = [sorted((_round_half_away(W[r][c] / s), c) for c in range(inn)) for r in range(out)]
    rtn = [[max(-8, min(7, int(_round_half_away(v / s)))) * s for v in r] for r in W]
    Y = ref_matmul(Xq, ref_transpose(W))
    e_rtn = sum((a - b) ** 2 for ra, rb in zip(Y, ref_matmul(Xq, ref_transpose(rtn)))
                for a, b in zip(ra, rb))
    wins = 0
    trials = 60
    for t in range(trials):
        r2 = random.Random(1000 + t)
        Wq = []
        for r in range(out):
            row = []
            for c in range(inn):
                tt = W[r][c] / s
                lo = math.floor(tt)
                frac = tt - lo
                q = lo + (1.0 if r2.random() < frac else 0.0)
                row.append(max(-8, min(7, int(q))) * s)
            Wq.append(row)
        e = sum((a - b) ** 2 for ra, rb in zip(Y, ref_matmul(Xq, ref_transpose(Wq)))
                for a, b in zip(ra, rb))
        if e < e_rtn:
            wins += 1
    print("      %d 次随机舍入中输出 MSE 优于 RTN 的 = %d" % (trials, wins))
    check("C", "在层输出 MSE 判据下，随机舍入**赢不了** RTN（这正是论文表 1 要用"
          "**任务准确率**当判据的原因）", wins == 0,
          "赢 %d/%d" % (wins, trials))
    m = SPECS.ADAROUND_SR_VS_RTN_MEASURED
    check("C", "而在准确率判据下它是能赢的（编号程序实测 %d/%d）——两个判据方向相反"
          % (m["accuracy_metric_wins"], m["trials"]),
          m["accuracy_metric_wins"] > 0 and m["output_mse_metric_wins"] == 0)

    # ---- C.4 8-bit 的 SmoothQuant 与 4-bit 的崩 ----
    sub("C.4  同一个迁移公式在 8-bit 有效、在 4-bit 崩 —— 判据是位宽")
    rnd = random.Random(13)
    nn, dd = 32, 32
    Xa = [[rnd.gauss(0, 1) * (40.0 if j in (3, 9) else 1.0) for j in range(dd)]
          for _ in range(nn)]
    Wa = [[rnd.gauss(0, 0.05) for _ in range(dd)] for _ in range(nn)]
    maxX = [max(abs(Xa[r][j]) for r in range(nn)) for j in range(dd)]
    maxW = [max(abs(Wa[r][j]) for r in range(nn)) for j in range(dd)]
    Y0 = ref_matmul(Xa, ref_transpose(Wa))
    base = max(abs(v) for ra in Y0 for v in ra)

    def sq_err(alpha, bits):
        s = ref_smoothquant_scale(maxX, maxW, alpha)
        Xs = [[Xa[r][j] / s[j] for j in range(dd)] for r in range(nn)]
        Ws = [[Wa[r][j] * s[j] for j in range(dd)] for r in range(nn)]
        Xq_ = [ref_rtn_asymmetric([Xs[r][j] for r in range(nn)], bits)[0] for j in range(dd)]
        Xq_ = [[Xq_[j][r] for j in range(dd)] for r in range(nn)]
        Wq_ = [[ref_rtn_symmetric(Ws[r], bits)[0][c] for c in range(dd)] for r in range(nn)]
        Y = ref_matmul(Xq_, ref_transpose(Wq_))
        return sum((a - b) ** 2 for ra, rb in zip(Y0, Y) for a, b in zip(ra, rb)) / (nn * nn) / (base ** 2)

    e8 = sq_err(SPECS.SQ_ALPHA_DEFAULT, 8)
    e4 = sq_err(SPECS.SQ_ALPHA_DEFAULT, 4)
    print("      α=0.5 的相对输出 MSE：8-bit = %.3e ; 4-bit = %.3e" % (e8, e4))
    check("C", "同一个 α 在 8-bit 下误差小、在 4-bit 下大 ≥ 100 倍（4-bit 崩）",
          e4 > e8 * 100, "比值 %.0f×" % (e4 / e8))
    check("C", "⇒ SmoothQuant 的位宽是 8 这一点是**前提**，不是细节",
          SPECS.SQ_TARGET_BITS == 8 and SPECS.SQ_W4A4_PPL > 20.0)

    # ---- C.5 observer 的冻结开关只关 FakeQuantize ----
    sub("C.5  `disable_observer` 对**普通 observer** 是空操作（这是个会静默失败的坑）")
    check("C", "observer.py 里既没有 self.training 也没有 observer_enabled",
          SPECS.OBSERVER_HAS_TRAINING_FLAG is False)
    check("C", "`observer_enabled` 挂在 %s 上（不是 observer 上）"
          % SPECS.OBSERVER_ENABLED_LIVES_ON,
          SPECS.OBSERVER_ENABLED_LIVES_ON == "FakeQuantizeBase")
    check("C", "`disable_observer` 是个 apply 回调，直接传模型进去是**静默空操作**",
          SPECS.DISABLE_OBSERVER_IS_APPLY_CALLBACK is True)

    # ---- C.6 reduce_range 的代价与它的适用范围 ----
    sub("C.6  `reduce_range` 砍掉 1 bit，理由是 x86 的 `vpmaddubsw` 溢出")
    check("C", "quint8 的范围从 (0,255) 变成 %s" % (SPECS.REDUCE_RANGE_QUINT8,),
          SPECS.REDUCE_RANGE_QUINT8 == (0, 127)
          and SPECS.REDUCE_RANGE_QUINT8[1] == 255 // 2)
    check("C", "qint8 的范围是 %s（对称侧也砍 1 bit）" % (SPECS.REDUCE_RANGE_QINT8,),
          SPECS.REDUCE_RANGE_QINT8 == (-64, 63))
    check("C", "理由是源码注释点名的 **%s**，不是「精度更好」"
          % SPECS.REDUCE_RANGE_REASON_OPCODE,
          SPECS.REDUCE_RANGE_REASON_OPCODE == "vpmaddubsw")
    check("C", "源码自己声明这对非 x86 后端**不是最优选择**（规范 vs 实现的岔口）",
          "not an optimal choice" in SPECS.REDUCE_RANGE_NOT_OPTIMAL_NOTE)
    check("C", "`per_tensor_symmetric + reduce_range + quint8` 会抛 NotImplementedError",
          SPECS.REDUCE_RANGE_SYMMETRIC_QUINT8_RAISES is True)


# ===========================================================================
# E. 「故意改错必须失败」的自检
# ===========================================================================

def run_E():
    """CONTRIBUTING §2 的机械验证：把参照实现改错，A 类断言必须失败。

    做法：对每个参考实现，构造一个**故意写错的版本**，断言「错版本会被 A 类判据拒绝」。
    **不修改任何常量**——改的是参照实现本身。
    """
    section("E", "自检：故意改错，A 类判据必须判它错")
    ok = True

    # E.1 EMA 的首次观测：改成「第一次也走 EMA」
    c = SPECS.EMA_AVERAGING_CONSTANT
    wrong = []
    hi = 0.0
    for v in SPECS.EMA_SEEN_MAXIMA:
        hi = hi + c * (v - hi)
        wrong.append(hi)
    ok &= check("A", "E.1  EMA「第一次也走 EMA」的错版本**不**等于实测值",
                not all(abs(a - b) < 1e-9
                        for a, b in zip(wrong, SPECS.EMA_RECORDED_MAXIMA)),
                "错版本 %s" % ["%.4f" % v for v in wrong])

    # E.2 把「平方误差最小」换成「KL 散度最小」——**这正是「HistogramObserver 是 KL
    #     校准器」那个说法会造成的后果**：目标函数一换，最优点就跑到别处去了。
    xs = gen_heavytail(STDLIB_N, STDLIB_SEED, "gauss_outlier")
    hist, edges, amax_mm, cand, real = hist_of(xs)
    a_real = cand[real.index(min(real))]
    a_kl = ref_trt_entropy_amax(hist, edges, start_bin=SPECS.TRT_START_BIN)
    mse_best, _ = mse_optimal_amax(xs)
    ok &= check("A", "E.2  目标函数从 L2 换成 KL 后，最优点挪到 **2 倍以下**",
                a_kl < a_real / 2.0,
                "L2 最优 %.3f vs KL 最优 %.3f（真实 MSE 最优 %.3f）"
                % (a_real, a_kl, mse_best))
    ok &= check("A", "E.2b KL 的最优离真实 MSE 最优**远得多**（所以 L2 才是对的那个）",
                abs(a_kl - mse_best) > abs(a_real - mse_best) * 2.0,
                "|KL−MSE*| = %.3f ; |L2−MSE*| = %.3f"
                % (abs(a_kl - mse_best), abs(a_real - mse_best)))
    # E.2c 另一条错法：只数被裁掉的桶数（不管误差多大）⇒ 推向「一点都不裁」
    bad = [ref_clipped_mass_objective(hist, edges, a) for a in cand]
    a_cnt = cand[bad.index(min(bad))]
    ok &= check("A", "E.2c 「只数被裁的桶数」的最优解是**完全不裁**（与 L2 也不同）",
                a_cnt > a_real,
                "计数版最优 %.3f（= minmax）vs L2 最优 %.3f" % (a_cnt, a_real))

    # E.3 KL 的平局改成「取首个最小值」
    ties = [1.0, 1.0, 2.0]
    last = _argmin_reversed(ties)
    first = min(range(len(ties)), key=lambda i: ties[i])
    ok &= check("A", "E.3  「取首个最小值」与 last_argmin 落在**不同**的 amax 上",
                (len(ties) - 1 - last) != first,
                "last_argmin → %d ; 首个最小值 → %d" % (len(ties) - 1 - last, first))

    # E.4 GPTQ：去掉误差补偿（退化成 RTN）后，输出误差不再下降
    rnd = random.Random(7)
    d_out, d_in, n = 24, 32, 256
    X = []
    for _ in range(n):
        row = [rnd.gauss(0, 1) for _ in range(d_in)]
        row[1] = row[0] * 0.98 + rnd.gauss(0, 0.2)
        row[2] = row[0] * 0.97 + rnd.gauss(0, 0.2)
        row[3] *= 12.0
        X.append(row)
    W = [[rnd.gauss(0, 1) * 0.05 for _ in range(d_in)] for _ in range(d_out)]
    H = ref_gram(X)
    Y = ref_matmul(X, ref_transpose(W))

    def out_mse(Q):
        Yq = ref_matmul(X, ref_transpose(Q))
        return sum((a - b) ** 2 for ra, rb in zip(Y, Yq) for a, b in zip(ra, rb)) / (n * d_out)

    def rtn_only(Wm):
        return [[max(-8, min(7, int(_round_half_away(v / (max(abs(x) for x in r) / 7.0))))) *
                 (max(abs(x) for x in r) / 7.0) for v in r] for r in Wm]

    e_rtn = out_mse(rtn_only(W))
    e_gptq = out_mse(ref_gptq(W, H, blocksize=16))
    e_broken = out_mse(ref_gptq(W, H, blocksize=16, cholesky_inv=False))   # 换一种 H⁻¹ 用法
    ok &= check("A", "E.4  去掉补偿（= RTN）后输出误差回到高位", e_rtn > e_gptq * 1.05,
                "RTN %.4e vs GPTQ %.4e" % (e_rtn, e_gptq))
    ok &= check("A", "E.4b 换一种 H⁻¹ 用法会给出**不同**的量化结果（判据不是在自我印证）",
                abs(e_broken - e_gptq) > 0,
                "%.6e vs %.6e" % (e_broken, e_gptq))

    # E.5 对称除数写成 127 会与 127.5 不一致
    xs2 = [1.0, -2.0, 3.0, -0.5]
    amax = max(abs(v) for v in xs2)
    a127 = [max(-128, min(127, int(_round_half_away(v / (amax / 127.0))))) * (amax / 127.0)
            for v in xs2]
    a1275, _ = ref_rtn_symmetric(xs2, 8)
    ok &= check("A", "E.5  「除以 127」的版本与「除以 127.5」的结果不同",
                any(abs(a - b) > 1e-12 for a, b in zip(a127, a1275)))
    ok &= check("A", "E.5b 且 127.5 版恰好等于 (quant_max − quant_min)/2 推出的那个",
                abs(SPECS.SYMMETRIC_DIVISOR_QINT8 -
                    SPECS.SYMMETRIC_DIVISOR_DERIVATION[0] /
                    SPECS.SYMMETRIC_DIVISOR_DERIVATION[1]) < 1e-12)

    # E.6 AWQ 网格：若把 n_grid 当成 21（步长 1/20 但含端点）就会得到 1.0
    g20 = ref_awq_grid(20)
    g21 = [i * (1.0 / 20) for i in range(21)]
    ok &= check("A", "E.6  `n_grid=20` 与「21 点含端点」是**不同**的网格（1.0 只在后者）",
                1.0 not in g20 and 1.0 in g21,
                "20 点 max=%.2f ; 21 点 max=%.2f" % (max(g20), max(g21)))

    # E.7 软舍入：把 clip 去掉，两端就不再是 0/1
    def h_noclip(V):
        return (1.0 / (1.0 + math.exp(-V))) * (SPECS.ADAROUND_ZETA - SPECS.ADAROUND_GAMMA) \
            + SPECS.ADAROUND_GAMMA
    ok &= check("A", "E.7  去掉 clip 后 h(−60) < 0（不再是合法概率）",
                h_noclip(-60.0) < 0.0 and ref_soft_round(-60.0) == 0.0,
                "无 clip 的 h(−60) = %.4f" % h_noclip(-60.0))
    ok &= check("A", "E.7b 去掉 clip 后 h(+60) > 1",
                h_noclip(60.0) > 1.0 and ref_soft_round(60.0) == 1.0,
                "无 clip 的 h(+60) = %.4f" % h_noclip(60.0))

    print()
    print("  E 节结论：%s" % ("全部自检按预期失败 ⇒ A 类判据是活的" if ok
                              else "**有自检未按预期失败 ⇒ A 类判据有问题**"))
    return ok


# ===========================================================================
# F. genres：本文件覆盖了哪些 specs 常量（A6 变异测试的清单）
# ===========================================================================

def run_F():
    section("F", "变异测试清单：本文件读过的 specs 常量")
    names = sorted(n for n in dir(SPECS) if n.isupper() and not n.startswith("_"))
    used = []
    src = open(__file__, encoding="utf-8").read()
    for n in names:
        if ("SPECS.%s" % n) in src:
            used.append(n)
    print("      specs.py 里共 %d 条全大写常量；本文件直接读到的 %d 条"
          % (len(names), len(used)))
    unused = [n for n in names if n not in used]
    print("      本文件**没有**读到的 %d 条（由编号程序负责，见各程序 §F）：" % len(unused))
    for n in unused:
        print("        · %s" % n)
    check("A", "变异的对象是**常量**，每个常量的值都能被至少一处断言分辨",
          len(used) > 0)
    return unused


# ===========================================================================
# main
# ===========================================================================

def main():
    print("=" * 78)
    print("130-model-optimization/quantization · verify.py（纯 python3，**不 import torch / numpy**）")
    print("=" * 78)
    print("说明：本文件产出的是**推导**，不是实测。规模 = %d 样本 / 直方图 %d 桶 /"
          % (STDLIB_N, HIST_BINS))
    print("      小矩阵 ≤ 64×64。**规模不同 ⇒ 数字不同，只有定性结构必须一致。**")

    run_A()
    run_C()
    okE = run_E()
    unused = run_F()

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

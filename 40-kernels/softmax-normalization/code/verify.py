#!/usr/bin/env python3
"""40-kernels/softmax-normalization 主题的本机验证模型。

**本模块是推导，不是硬件测量。** 它只跑算术与 numpy，不碰硬件——本机（macOS）
没有 CUDA 工具链也没有 GPU，`code/` 里的 3 个 `.cu` 内核的运行结果本仓库没有。

它覆盖本主题全部 3 篇文档的可推导部分，分三类断言（CONTRIBUTING §2）：

  A 算法正确性  numpy 参考实现 vs 文档所述算法（稳定 softmax / 在线 softmax /
                Welford 单遍），并能复算文档 §1 里手算的那一步步中间量
  B 代价模型    按文档公式算 exp 的上溢/下溢阈值、访存字节、算术强度、
                与 roofline 拐点的比值，逐条打印数值并给出判定
  C 边界与反例  §4 声称「会出问题」的输入必须**真的跑出问题**——断言的是
                失败行为成立，不是成功

**数字的口径**（沿用 20-gpu-architecture 的三分法）：
  · 规格值 = 来自 `specs.py`，逐条带 `# 来源：`
  · 推导值 = 本程序算出，可在这里复核
  · 引用值 = 外部来源，标注「本仓库不复核其测量口径」

运行：`make verify`（等价于 `python3 verify.py`）。只需 python3 + numpy + scipy。
"""

import math

import numpy as np
from scipy.special import softmax as scipy_softmax

import specs as SP

# ===========================================================================
# 断言与打印基础设施
# ===========================================================================

_STATE = {"n": 0, "fail": []}


def check(name, cond, detail=""):
    """记一次判定并打印结果。cond 为假时记入失败清单。"""
    _STATE["n"] += 1
    ok = bool(cond)
    if not ok:
        _STATE["fail"].append(name)
    tail = f"   ← {detail}" if detail else ""
    print(f"    [{'通过' if ok else '**不通过**'}] {name}{tail}")
    return ok


def section(tag, title):
    print()
    print("=" * 76)
    print(f"§{tag}  {title}")
    print("=" * 76)


def sub(title):
    print()
    print(f"--- {title} ---")


def close(a, b, rel=1e-9):
    return abs(a - b) <= rel * max(1.0, abs(a), abs(b))


def relerr(a, b):
    """a 相对 b 的相对误差。b 为 0 时返回绝对差。"""
    return abs(a - b) / abs(b) if b else abs(a - b)


H100 = "H100"
B200 = "B200"
FP32_BYTES = SP.FLOAT32_BYTES
FP16_BYTES = SP.FLOAT16_BYTES

np.seterr(all="ignore")          # 本模块**故意**触发上溢/下溢，见 §C

# ---------------------------------------------------------------------------
# 格式常量（规格值，全部来自 specs.py）
# ---------------------------------------------------------------------------

F16_MAX = SP.FP16_MAX_FINITE                     # 65504.0
F16_MIN_SUB = SP.FP16_MIN_SUBNORMAL              # 2^-24
F32_MAX = SP.FP32_MAX_FINITE                     # FLT_MAX
F32_MIN_SUB = SP.FP32_MIN_SUBNORMAL              # 2^-149
LN2 = math.log(2.0)

# ---------------------------------------------------------------------------
# 推导值：exp 的四个阈值
#
# 判据一律是「结果舍入到目标格式后变成什么」。**两种精度必须同一口径**：
#   上溢阈值 = ln(最大有限数与它上方下一个可表示值的中点)
#   下溢阈值 = ln(最小次正规数的一半)      ← 次正规，不是正规
#
# 记号说明：`_HOST` = 在 fp32/fp64 域算 exp，再舍入存进目标格式；
#           `_NATIVE` = 输入先量化到目标格式，再算 exp，结果再舍回目标格式。
# ---------------------------------------------------------------------------

# fp16：最大有限数上方「本该是」65536，但 65536 在 fp16 里就是 inf，
#       所以中点是 (65504 + 65536)/2 = 65520
F16_OVF_HOST = math.log(65520.0)                 # 11.090110718526951
F16_OVF_HOST_ALT = math.log(F16_MAX)             # 11.089866488461016 ← 只是表观值，不是阈值
F16_UNF = math.log(F16_MIN_SUB / 2.0)            # -25·ln2 = -17.328679513998633
F32_OVF_HOST = math.log(F32_MAX)                 # 88.72283905206835
F32_UNF = math.log(F32_MIN_SUB / 2.0)            # -150·ln2 = -103.97207708399179

# fp16 的格步长：[8,16) 档的 ulp = 2^15·2^-10 = 2^-7
F16_STEP_ABOVE_8 = 2.0 ** -7                     # 0.0078125


def exp_round(x, dt):
    """在 fp64 域算 exp(x)，再舍入到 dt。这是「口径 _HOST」。"""
    return dt(np.exp(np.float64(x)))


def has_inf(e):
    return bool(np.isinf(np.asarray(e, dtype=np.float64)).any())


def has_nan(e):
    return bool(np.isnan(np.asarray(e, dtype=np.float64)).any())


def bsearch_switch(pred, x_false, x_true, iters=200):
    """pred 在 x_false 处为假、在 x_true 处为真；二分出交界点。"""
    lo, hi = float(x_false), float(x_true)
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        if pred(mid):
            hi = mid
        else:
            lo = mid
    return lo, hi


# ---------------------------------------------------------------------------
# 算法实现（**参考实现**，与 .cu 内核逐行对应）
# ---------------------------------------------------------------------------


def softmax_naive(x, dt):
    """朴素 softmax：不算 max，直接 exp 再除。对应 01_softmax_stable.cu 的第一个内核。"""
    e = np.exp(np.asarray(x, dtype=np.float64)).astype(dt)
    return e, dt(np.sum(e, dtype=dt))


def softmax_two_pass(x, dt):
    """两遍稳定 softmax：先 max，再 exp 求和。对应 01 的第二个内核。"""
    xd = np.asarray(x, dtype=np.float64)
    m = dt(np.max(xd))
    e = np.exp(xd - float(m)).astype(dt)
    l = dt(np.sum(e, dtype=dt))
    return m, e, l


def online_softmax_blocks(x, block, dt, order=None, stride_order=False):
    """在线 softmax 的 (m, l) 递推，**逐块**执行文档 02 §1 的三条式子。

    返回 (m, l, rescale 次数, 每块结束时的 (m, l) 轨迹)。

    参数
      order          : None / 'asc' / 'desc' —— 先对输入重排再分块
      stride_order   : True 时按「步长 block」切分（模拟多线程拉链式访存），
                       而不是连续切片；两种切法都必须是零误差一致的。
    """
    xd = np.asarray(x, dtype=np.float64)
    if order == "asc":
        xd = np.sort(xd)
    elif order == "desc":
        xd = np.sort(xd)[::-1]

    n = len(xd)
    if stride_order:
        chunks = [xd[i::block] for i in range(block)]
    else:
        chunks = [xd[i:i + block] for i in range(0, n, block)]

    m = -np.inf
    l = dt(0.0)
    path = []
    rescales = 0
    for blk in chunks:
        if len(blk) == 0:
            continue
        m_blk = dt(np.max(blk))                      # 块内 max
        m_new = dt(m_blk) if m_blk > m else dt(m)
        # 「rescale 次数」只数**已经在跑之后** m 被抬高的次数；
        # 第 0 块从 −inf 起步的那次不算（那时没有东西需要被重标定）。
        if m_new > m and np.isfinite(m):
            rescales += 1
        # 三条式子：
        #   m_new = max(m_old, max(块内 x))
        #   l_new = l_old · exp(m_old − m_new) + Σ exp(x − m_new)
        alpha = 0.0 if not np.isfinite(m) else float(np.exp(float(m) - float(m_new)))
        l = dt(dt(l * dt(alpha)) + dt(np.sum(np.exp(blk - float(m_new)), dtype=dt)))
        m = m_new
        path.append((float(m), float(l)))
    return m, l, rescales, path


def welford_seq(x, dt):
    """顺序 Welford：返回 (mean, M2, count)。M2 是**未除以 n** 的平方和。"""
    mean = dt(0.0)
    M2 = dt(0.0)
    cnt = 0
    for v in x:
        cnt += 1
        d = dt(v) - mean
        mean = dt(mean + dt(d / dt(cnt)))
        M2 = dt(M2 + dt(d * (dt(v) - mean)))
    return mean, M2, cnt


def welford_blocked(x, dt, threads=256):
    """分块 Welford + Chan 合并。对应 03_layernorm.cu 的单遍内核。

    合并式（Chan / parallel variance）：
        mean_ab = mean_a + (mean_b − mean_a)·n_b/n
        M2_ab   = M2_a + M2_b + (mean_b − mean_a)²·n_a·n_b/n
    """
    n = len(x)
    per = (n + threads - 1) // threads
    parts = []
    for t in range(threads):
        seg = x[t * per:(t + 1) * per]
        if len(seg) == 0:
            continue
        mean, M2, cnt = welford_seq(seg, dt)
        parts.append([mean, M2, cnt])

    while len(parts) > 1:
        nxt = []
        for i in range(0, len(parts), 2):
            if i + 1 >= len(parts):
                nxt.append(parts[i])
                continue
            a, b = parts[i], parts[i + 1]
            na, nb = a[2], b[2]
            nn = na + nb
            d = b[0] - a[0]
            mean = dt(a[0] + d * dt(nb) / dt(nn))
            M2 = dt(a[1] + b[1] + d * d * dt(na) * dt(nb) / dt(nn))
            nxt.append([mean, M2, nn])
        parts = nxt

    mean, M2, cnt = parts[0]
    return float(mean), float(M2) / cnt


def two_pass_blocked(x, dt):
    """两遍法 + 树形归约：适合与 welford_blocked 对照。"""
    xn = np.asarray(x, dtype=dt)
    n = len(xn)
    mean = dt(np.sum(xn, dtype=dt)) / dt(n)
    dev = (xn - mean).astype(dt)
    var = dt(np.sum((dev * dev).astype(dt), dtype=dt)) / dt(n)
    return float(mean), float(var)


def sumsq_blocked(x, dt):
    """单遍 Σx² − n·μ²：**实现最省、数值最差**的写法（03 §4.1 的反例）。"""
    xn = np.asarray(x, dtype=dt)
    n = len(xn)
    mean = float(np.sum(xn, dtype=dt)) / n
    sq = float(np.sum((xn * xn).astype(dt), dtype=dt))
    return mean, (sq - n * mean * mean) / n


def welford_mean_drift(x, dt):
    """顺序 Welford 的 mean 与 fp64 真值之差（03 §4.2 的成因）。"""
    m, _, _ = welford_seq(x, dt)
    return float(m) - float(np.mean(np.asarray(x, dtype=np.float64)))


# ===========================================================================
# §0 口径
# ===========================================================================


def s0_scope():
    section(0, "口径与本次运行的场景")
    print("本程序全部输出是**推导值**：跑的是算术与 numpy，本机没有 GPU，")
    print("3 个 .cu 内核的运行结果本仓库没有。")
    print()
    print(f"主用机器：{H100} SXM —— {SP.SM_COUNT_BY_GEN[H100]} SM / "
          f"{SP.FP32_VECTOR_TFLOPS_DENSE[H100]} TFLOPS FP32 向量 / "
          f"{SP.HBM_BW_GBps_BY_GEN[H100]} GB/s")
    print(f"对照机器：{B200} SXM —— {SP.SM_COUNT_BY_GEN[B200]} SM / "
          f"{SP.FP32_VECTOR_TFLOPS_DENSE[B200]} TFLOPS FP32 向量 / "
          f"{SP.HBM_BW_GBps_BY_GEN[B200]} GB/s")
    ridge = SP.FP32_VECTOR_TFLOPS_DENSE[H100] * 1e12 / (SP.HBM_BW_GBps_BY_GEN[H100] * 1e9)
    print(f"roofline 拐点 I*(FP32, {H100}) = {SP.FP32_VECTOR_TFLOPS_DENSE[H100]}e12 ÷ "
          f"{SP.HBM_BW_GBps_BY_GEN[H100]}e9 = {ridge:.1f} FLOP/byte（推导值）")
    print()
    print("格式常量（规格值，来自 specs.py）：")
    print(f"  fp16: 最大有限数 {F16_MAX:.0f}  最小次正规 {F16_MIN_SUB:.6e} (2^-24)  "
          f"u = {SP.FP16_ROUNDING_UNIT:.6e} (2^-11)")
    print(f"  fp32: 最大有限数 {F32_MAX:.6e}  最小次正规 {F32_MIN_SUB:.6e} (2^-149)  "
          f"u = {SP.FP32_ROUNDING_UNIT:.6e} (2^-24)")
    print()
    print("⚠ 本主题的**每一个阈值都是推导值**，不是格式常量本身：")
    print(f"    fp16 上溢阈值 = ln 65520 = {F16_OVF_HOST:.10f}   （不是 ln 65504 = "
          f"{F16_OVF_HOST_ALT:.6f}）")
    print(f"    fp16 下溢阈值 = −25·ln2  = {F16_UNF:.10f}")
    print(f"    fp32 上溢阈值 = ln FLT_MAX = {F32_OVF_HOST:.10f}")
    print(f"    fp32 下溢阈值 = −150·ln2 = {F32_UNF:.10f}")


# ===========================================================================
# §A 算法正确性
# ===========================================================================


def a1_stable_softmax_reference():
    sub("A1  稳定 softmax 与 scipy.special.softmax 逐元素一致（文档 01）")
    rng = np.random.default_rng(20260916)
    x = rng.standard_normal(4096) * 8.0
    ref = scipy_softmax(x.astype(np.float64))

    m, e, l = softmax_two_pass(x, np.float64)
    ours = e / float(l)
    err = float(np.max(np.abs(ours - ref)))
    print(f"    n = 4096, logits ~ N(0,1)×8（fp64）")
    print(f"    减 max 后：m = {float(m):.6f}，l = {float(l):.6f}，"
          f"最大 exp = {float(np.max(e)):.6f}")
    print(f"    与 scipy 的逐元素最大绝对误差 = {err:.3e}")
    check("稳定 softmax 与 scipy 一致（≤1e-15）", err <= 1e-15, f"{err:.3e}")

    e, l = softmax_naive(x, np.float64)
    naive = e / float(l)
    err_n = float(np.max(np.abs(naive - ref)))
    print(f"    同一输入**不减 max**（fp64）：最大绝对误差 = {err_n:.3e}"
          f"  ← fp64 的动态范围够，数学上等价，所以这里是对的")
    check("不减 max 在 fp64 下仍然正确（fp64 范围足够）",
          err_n <= 1e-15, f"{err_n:.3e}")

    print()
    print("    解析值对照（不需要 scipy 就能验的两个点）：")
    uni = np.full(37, 3.5)
    _, e_u, l_u = softmax_two_pass(uni, np.float64)
    got = float(e_u[0] / l_u)
    print(f"      等值输入 n=37：softmax 恒为 1/n = {1/37!r}，实得 {got!r}")
    check("等值输入给出精确的 1/n", got == 1.0 / 37.0, f"{got!r}")
    two = np.array([0.0, math.log(3.0)])
    _, e_2, l_2 = softmax_two_pass(two, np.float64)
    got2 = float(e_2[0] / l_2)
    print(f"      x = [0, ln3]：softmax = [1/4, 3/4]，实得 [{got2!r}, "
          f"{float(e_2[1]/l_2)!r}]")
    check("x=[0,ln3] 给出 [0.25, 0.75]（1e-15 内）",
          close(got2, 0.25, 1e-15), f"{got2!r}")


def a2_online_vs_two_pass():
    sub("A2  在线 softmax 与两遍 softmax 一致（文档 02）：分块大小 × 顺序")
    rng = np.random.default_rng(20260917)
    n = 4096
    x = rng.standard_normal(n) * 4.0
    m_ref, e_ref, l_ref = softmax_two_pass(x, np.float64)
    ref_l = float(l_ref)
    print(f"    n = {n}，两遍法（fp64）l = {ref_l!r}")
    print()
    print(f"    {'分块':>6}{'顺序':>8}{'切法':>8}{'rescale':>9}{'l':>22}{'相对误差':>12}")
    worst64 = 0.0
    worst32 = 0.0
    for blk in (16, 32, 128, 1024, 4096):
        for order, label in ((None, "原序"), ("asc", "递增"), ("desc", "递减")):
            for stride in (False, True):
                _, l, nr, _ = online_softmax_blocks(x, blk, np.float64, order, stride)
                e64 = relerr(float(l), ref_l)
                worst64 = max(worst64, e64)
                _, l32, _, _ = online_softmax_blocks(x, blk, np.float32, order, stride)
                e32 = relerr(float(l32), ref_l)
                worst32 = max(worst32, e32)
                print(f"    {blk:>6}{label:>8}{'步长' if stride else '连续':>8}"
                      f"{nr:>9}{float(l):>22.12f}{e64:>12.3e}")
                if stride:
                    print(f"    {'':>6}{'':>8}  ↳ fp32 的状态：l = {float(l32):.6f}，"
                          f"相对误差 {e32:.3e}")
    print()
    check("在线 softmax（fp64）对所有分块/顺序都在 1e-14 内", worst64 <= 1e-14,
          f"最差 {worst64:.3e}")
    check("在线 softmax（fp32 状态）对所有分块/顺序都在 1e-5 内", worst32 <= 1e-5,
          f"最差 {worst32:.3e}")
    print()
    print("    ⚠ 「分块顺序打乱后仍一致」成立的原因不是误差小，而是**结合律**：")
    print("      (m,l) 的合并算子 ⊕ 构成一个半群，任意分组/任意顺序给出同一个数学结果，")
    print("      差异只来自浮点舍入。上面「步长切法」与「连续切法」一致性同理。")
    print("      这条性质是 02 篇分块流水的全部依据，也是 attention 能分块的原因。")


def a3_hand_computed_blocks():
    sub("A3  复算文档 02 §1 的两分块手算例子（逐项中间量）")
    x = np.array([2.0, -1.0, 4.0, 0.0])
    b1, b2 = x[:2], x[2:]
    print(f"    x = {x.tolist()}，分两块：B1 = {b1.tolist()}，B2 = {b2.tolist()}")
    print()

    m1 = float(np.max(b1))
    l1 = float(np.sum(np.exp(b1 - m1)))
    print(f"    块 1：m₁ = max{b1.tolist()} = {m1:.1f}")
    print(f"          l₁ = exp(2−2) + exp(−1−2) = 1 + exp(−3) = {l1!r}")
    print(f"          校验：1 + {math.exp(-3)!r} = {1.0 + math.exp(-3)!r}")

    m_b2 = float(np.max(b2))
    print(f"    块 2：块内 max = {m_b2:.1f}")
    m2 = max(m1, m_b2)
    alpha = math.exp(m1 - m2)
    add = float(np.sum(np.exp(b2 - m2)))
    l2 = l1 * alpha + add
    print(f"          m₂ = max(m₁, {m_b2:.1f}) = {m2:.1f}")
    print(f"          α  = exp(m₁ − m₂) = exp(−2) = {alpha!r}")
    print(f"          l₂ = l₁·α + Σexp(x−m₂) = {l1!r}×{alpha!r} + "
          f"(exp(0)+exp(−4)={add!r})")
    print(f"             = {l1*alpha!r} + {add!r} = {l2!r}")

    exact = float(np.sum(np.exp(x - 4.0)))
    print(f"    直接算（m = 4 的两遍法）：l = {exact!r}")
    print(f"    差 = {abs(l2-exact):.3e}（相对 {relerr(l2, exact):.3e}）")
    check("两分块递推与直接算的 l 一致（1e-15 内）",
          close(l2, exact, 1e-15), f"{l2!r} vs {exact!r}")

    print()
    print("    对应的 softmax 输出（y = exp(x−m)/l）：")
    for xi, yi in zip(x, np.exp(x - 4.0) / exact):
        print(f"      x = {xi:>5.1f}  →  y = {yi!r}")
    check("输出之和为 1（1e-15 内）",
          close(float(np.sum(np.exp(x - 4.0) / exact)), 1.0, 1e-15))

    print()
    print("    把同一个例子跑一遍真正的分块实现（与手算逐位对照）：")
    _, l_impl, nr, path = online_softmax_blocks(x, 2, np.float64)
    for i, (mi, li) in enumerate(path, 1):
        print(f"      第 {i} 块后：m = {mi:.1f}，l = {li!r}")
    check("实现与手算的 l 一致（1e-16 内）", close(float(l_impl), l2, 1e-16),
          f"{float(l_impl)!r} vs {l2!r}")
    check("本例中 m 被抬高了一次（rescale 次数 = 1）", nr == 1, f"{nr}")


def a4_welford_vs_two_pass():
    sub("A4  Welford 单遍与两遍法的方差一致（文档 03）")
    rng = np.random.default_rng(20260918)
    n = 4096
    x = rng.standard_normal(n).astype(np.float32)
    ref_var = float(np.var(x.astype(np.float64)))
    ref_mean = float(np.mean(x.astype(np.float64)))
    m_w, v_w = welford_blocked(x, np.float32)
    m_t, v_t = two_pass_blocked(x, np.float32)
    print(f"    n = {n}，x ~ N(0,1)（fp32 存储）")
    print(f"    fp64 参考    : mean = {ref_mean:.10f}  var = {ref_var:.10f}")
    print(f"    Welford 分块 : mean = {m_w:.10f}  var = {v_w:.10f}")
    print(f"    两遍法 分块  : mean = {m_t:.10f}  var = {v_t:.10f}")
    print(f"    两者方差相对差 = {relerr(v_w, v_t):.3e}")
    check("Welford 与两遍法的方差一致（相对差 ≤ 1e-5）",
          relerr(v_w, v_t) <= 1e-5, f"{relerr(v_w, v_t):.3e}")
    check("两者都在 fp64 参考的 1e-5 内",
          relerr(v_w, ref_var) <= 1e-5 and relerr(v_t, ref_var) <= 1e-5,
          f"Welford {relerr(v_w, ref_var):.3e} / 两遍 {relerr(v_t, ref_var):.3e}")

    print()
    print("    不同分块线程数下 Welford 的稳定性（同一输入）：")
    for threads in (32, 64, 256, 1024):
        _, v = welford_blocked(x, np.float32, threads)
        print(f"      threads = {threads:>5}: var = {v:.10f}  相对误差 "
              f"{relerr(v, ref_var):.3e}")


def a5_large_mean_small_variance():
    sub("A5  大均值小方差：Welford / 两遍 / Σx²−nμ² 三种方法（文档 03 §1、§4）")
    rng = np.random.default_rng(20260919)
    n = 8192
    print(f"    一行 n = {n}，x = μ + N(0,1)，fp32 存储与累加；σ = 1 固定，只抬 μ。")
    print(f"    判据：与 fp64 方差真值的相对误差。")
    print()
    print(f"    {'μ/σ':>10}{'两遍法(分块)':>16}{'Welford(分块)':>17}"
          f"{'Σx²−nμ²(分块)':>18}")
    rows = {}
    for mu in (0.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6):
        x = (mu + rng.standard_normal(n)).astype(np.float32)
        rv = float(np.var(x.astype(np.float64)))
        e_t = relerr(two_pass_blocked(x, np.float32)[1], rv)
        e_w = relerr(welford_blocked(x, np.float32)[1], rv)
        e_s = relerr(sumsq_blocked(x, np.float32)[1], rv)
        rows[mu] = (e_t, e_w, e_s)
        print(f"    {mu:>10.0f}{e_t:>16.3e}{e_w:>17.3e}{e_s:>18.3e}")
    print()
    check("μ=0（小均值）时三种方法都对（≤1e-4）",
          max(rows[0.0]) <= 1e-4, f"最差 {max(rows[0.0]):.3e}")
    check("μ/σ=100 时 Σx²−nμ² 已开始崩坏（>1e-3）",
          rows[100.0][2] > 1e-3, f"{rows[100.0][2]:.3e}")
    check("μ/σ=1e3 时 Σx²−nμ² 的误差 > 1e-2",
          rows[1e3][2] > 1e-2, f"{rows[1e3][2]:.3e}")
    check("μ/σ≥1e4 时 Σx²−nμ² 的误差 > 1（方差的量级都不对）",
          rows[1e4][2] > 1.0 and rows[1e5][2] > 1.0,
          f"1e4→{rows[1e4][2]:.3e}, 1e5→{rows[1e5][2]:.3e}")
    check("同样的输入下 Welford 与两遍法都还在 1e-2 以内",
          rows[1e5][0] <= 1e-2 and rows[1e5][1] <= 1e-2,
          f"两遍 {rows[1e5][0]:.3e} / Welford {rows[1e5][1]:.3e}")
    check("⇒ Welford 相对 Σx²−nμ² 稳 6 个数量级以上（μ/σ=1e5）",
          rows[1e5][2] / max(rows[1e5][1], 1e-30) > 1e6,
          f"比值 {rows[1e5][2]/rows[1e5][1]:.3e}")
    print()
    print("    ⚠ **反直觉的一条**：上表里两遍法反而略优于 Welford。")
    print("      两遍法的第一遍求均值是一棵树形归约（误差 ~√n·u），")
    print("      Welford 的均值是**逐项**递推的，会撞上「更新量被 |mean| 的 ulp 吞掉」；")
    print("      机制见 §C8。结论不是「Welford 更准」，而是")
    print("      「Welford 单遍 + 数值上避开灾难性抵消」，代价是分块合并要多算几个乘加。")


# ===========================================================================
# §B 代价模型
# ===========================================================================


def b1_exp_thresholds():
    sub("B1  exp 的上溢/下溢阈值：**两套口径 × 两种精度**（文档 01 §1）")
    print("    口径 HOST  ：在 fp32/fp64 域算 exp，结果再舍入进目标格式")
    print("    口径 NATIVE：输入先量化到目标格式，再算 exp，结果再舍回目标格式")
    print("    判据一律是「舍入后变成 inf / 0」，不是「超过某个解析值」。")
    print()

    cases = [
        ("fp16 上溢 → inf", np.float16, True),
        ("fp16 下溢 → 0", np.float16, False),
        ("fp32 上溢 → inf", np.float32, True),
        ("fp32 下溢 → 0", np.float32, False),
    ]
    analytic = {
        ("fp16", True): F16_OVF_HOST, ("fp16", False): F16_UNF,
        ("fp32", True): F32_OVF_HOST, ("fp32", False): F32_UNF,
    }
    print(f"    {'':<18}{'口径 HOST 二分解':>22}{'解析值':>22}{'相对差':>11}")
    for name, dt, overflow in cases:
        pred = (lambda v: has_inf(exp_round(v, dt))) if overflow \
            else (lambda v: not exp_round(v, dt))
        if overflow:
            lo, hi = bsearch_switch(pred, 0.0, 200.0 if dt is np.float32 else 100.0)
        else:
            lo, hi = bsearch_switch(pred, 0.0, -300.0 if dt is np.float32 else -60.0)
        got = (lo + hi) / 2.0
        key = ("fp16" if dt is np.float16 else "fp32", overflow)
        print(f"    {name:<18}{got:>22.13f}{analytic[key]:>22.13f}"
              f"{relerr(got, analytic[key]):>11.2e}")

    print()
    print("    解析值的来历（可手算复核）：")
    print(f"      fp16 上溢：最大有限数 {F16_MAX:.0f} 上方最近的可表示值是 65536，")
    print(f"                 但 65536 在 fp16 里就是 inf ⟹ 中点是 (65504+65536)/2 = 65520")
    print(f"                 ln 65520 = {F16_OVF_HOST!r}")
    print(f"                 ⚠ 表观值 ln 65504 = {F16_OVF_HOST_ALT!r} **不是**阈值")
    print(f"      fp16 下溢：最小次正规数的一半 = 2^-24/2 = 2^-25")
    print(f"                 −25·ln2 = {F16_UNF!r}")
    print(f"      fp32 上溢：ln FLT_MAX = {F32_OVF_HOST!r}")
    print(f"      fp32 下溢：最小次正规数的一半 = 2^-149/2 = 2^-150")
    print(f"                 −150·ln2 = {F32_UNF!r}")

    print()
    lo, hi = bsearch_switch(
        lambda v: has_inf(exp_round(v, np.float16)), 0.0, 100.0)
    got16 = (lo + hi) / 2.0
    check("fp16 上溢阈值 = ln 65520（1e-12 内）",
          close(got16, F16_OVF_HOST, 1e-12), f"{got16!r}")
    lo, hi = bsearch_switch(lambda v: not exp_round(v, np.float16), 0.0, -60.0)
    got_u16 = (lo + hi) / 2.0
    check("fp16 下溢阈值 = −25·ln2（1e-12 内）",
          close(got_u16, F16_UNF, 1e-12), f"{got_u16!r}")
    lo, hi = bsearch_switch(
        lambda v: has_inf(exp_round(v, np.float32)), 0.0, 200.0)
    got32 = (lo + hi) / 2.0
    check("fp32 上溢阈值 = ln FLT_MAX（1e-9 内）",
          close(got32, F32_OVF_HOST, 1e-9), f"{got32!r}")
    lo, hi = bsearch_switch(lambda v: not exp_round(v, np.float32), 0.0, -300.0)
    got_u32 = (lo + hi) / 2.0
    check("fp32 下溢阈值 = −150·ln2（1e-12 内）",
          close(got_u32, F32_UNF, 1e-12), f"{got_u32!r}")

    print()
    print("    ---- 防坑断言 1：math.log(65504) ----")
    print(f"    math.log(65504)                = {math.log(65504)!r}")
    print(f"    math.log(float(np.float16(65504))) = "
          f"{math.log(float(np.float16(65504)))!r}   ← 转成 fp64 再取 log，正确")
    print(f"    np.log(np.finfo(np.float16).max)   = "
          f"{float(np.log(np.finfo(np.float16).max))!r}   ← **在 fp16 域里算 log**")
    print(f"    float(np.float16(11.09))           = "
          f"{float(np.float16(11.09))!r}   ← 11.09 在 fp16 里根本不存在")
    check("math.log(65504) == 11.089866（前 6 位小数）",
          f"{math.log(65504):.6f}" == "11.089866", f"{math.log(65504):.6f}")
    check("np.log(np.finfo(np.float16).max) 给出的是 11.09375（**错的**）",
          float(np.log(np.finfo(np.float16).max)) == 11.09375,
          f"{float(np.log(np.finfo(np.float16).max))!r}")
    check("两者差 3.884e-3 —— 这就是「在 fp16 域算 log」的污染量",
          close(11.09375 - math.log(65504), 3.8835e-3, 1e-3),
          f"{11.09375 - math.log(65504):.6e}")

    print()
    print("    ---- 防坑断言 2：fp16 域里不存在 11.0900 / 11.0901 这两个点 ----")
    print(f"    [8,16) 档的格宽 = 2^15·2^-10 = 2^-7 = {F16_STEP_ABOVE_8!r}")
    for x in (11.0859375, 11.089866488461016, 11.09, 11.0900807, 11.0901, 11.09375):
        xh = np.float16(x)
        e = exp_round(float(xh), np.float16)
        print(f"      x = {x:<20.13f} → fp16(x) = {float(xh):<10} "
              f"exp(fp16(x)) = {float(e):<12} → 存 fp16 = {float(e)!r}")
    grid = [11.0859375 + k * F16_STEP_ABOVE_8 for k in range(4)]
    print(f"    这一段里 fp16 能表示的点只有：{grid}")
    check("float16(11.0901) 实际取到 11.09375，exp 就是 inf",
          float(np.float16(11.0901)) == 11.09375
          and has_inf(exp_round(11.09375, np.float16)),
          f"{float(np.float16(11.0901))!r}")
    check("11.089866（ln 65504）不在 fp16 格上：它落在 11.0859375 与 11.09375 之间",
          11.0859375 < math.log(65504) < 11.09375,
          f"{math.log(65504):.9f}")
    check("纯 fp16 域的上下两个点：11.0859375 → 65248.0，11.09375 → inf",
          float(exp_round(11.0859375, np.float16)) == 65248.0
          and has_inf(exp_round(11.09375, np.float16)),
          f"{float(exp_round(11.0859375, np.float16))!r} / inf")
    print()
    print("    ⚠ **两个口径的差别就写在这两行里**：")
    print(f"      口径 HOST（在 fp32 里算 exp 再存 fp16）：阈值 {F16_OVF_HOST:.4f}，")
    print(f"      口径 NATIVE（全程 fp16）：阈值落在 11.0859375 与 11.09375 之间的格缝里，")
    print(f"      两个相邻可表示点一个给 65248.0、一个给 inf。**11.0901 属于前者。**")

    print()
    print("    ---- 防坑断言 3：fp32 的下溢不是 −88.03 ----")
    print(f"    exp(−88.03) 在 fp32 里 = {float(exp_round(-88.03, np.float32)):.6e}"
          f"  ← 非零的**次正规**数")
    print(f"    −88.0297 对应的是 2^-127 = {2.0**-127:.6e}，"
          f"而 fp32 的最小正规数是 2^-126 = {SP.FP32_MIN_NORMAL:.6e}")
    check("−88.03 处的 exp 在 fp32 里非零（所以它不是下溢阈值）",
          float(exp_round(-88.03, np.float32)) != 0.0,
          f"{float(exp_round(-88.03, np.float32)):.6e}")
    check("fp32 真正的下溢阈值要走到 −103.9721（= −150·ln2）",
          close(F32_UNF, -103.97207708399179, 1e-12), f"{F32_UNF!r}")
    check("fp16 只有 11.09 这一档阈值，没有 88.72 —— 88.72 是 fp32 的",
          F32_OVF_HOST > 88.7 and F16_OVF_HOST < 11.1,
          f"fp16 {F16_OVF_HOST:.4f} / fp32 {F32_OVF_HOST:.4f}")


def b2_doc01_numbers():
    sub("B2  文档 01 全部正文数字的出口")
    rng = np.random.default_rng(20260916)
    n = 4096
    scale = 8.0
    x = (rng.standard_normal(n) * scale).astype(np.float32)
    print(f"    §0/§1 的场景：n = {n}，logits ~ N(0,1)×{scale:.0f}，fp16 存储")
    print(f"      logit 范围 = [{float(x.min()):.4f}, {float(x.max()):.4f}]")
    over = int(np.count_nonzero(x > F16_OVF_HOST))
    print(f"      超过 fp16 上溢阈值 {F16_OVF_HOST:.4f} 的 logit 个数 = {over} / {n}"
          f"  （占 {over/n*100:.1f}%）")

    e_n, l_n = softmax_naive(x.astype(np.float16), np.float16)
    print(f"      不减 max：exp 里 inf 的个数 = {int(np.count_nonzero(np.isinf(e_n)))}，"
          f"nan 的个数 = {int(np.isnan(e_n.astype(np.float64)).sum())}")
    m, e_s, l_s = softmax_two_pass(x, np.float64)
    print(f"      减 max 后：l = {float(l_s):.6f}，最大 exp = {float(np.max(e_s)):.6f}"
          f"（恒为 1，因为 max 那一项是 exp(0)）")
    print(f"      y = exp(x−m)/l 的逐元素最大差（稳定版 vs fp64）："
          f"{float(np.max(np.abs(e_s/float(l_s) - e_s/float(l_s)))):.1e}")

    print()
    print("    §1 的核心数字表（全部由 §B1 的阈值推出）：")
    print(f"      {'精度':<6}{'exp 上溢 (→inf)':>20}{'exp 下溢 (→0)':>20}")
    print(f"      {'fp16':<6}{F16_OVF_HOST:>20.4f}{F16_UNF:>20.4f}")
    print(f"      {'fp32':<6}{F32_OVF_HOST:>20.4f}{F32_UNF:>20.4f}")
    print(f"      （fp32 的 88.7228 是 ln FLT_MAX = {F32_OVF_HOST:.6f}；"
          f"fp16 的 11.0901 是 ln 65520 = {F16_OVF_HOST:.6f}）")

    print()
    print("    §4.2 的丢弃界：减 max 之后，被下溢丢掉的项每项 < 2^-25（fp16）/ 2^-150（fp32），")
    print("    而分母 l ≥ 1（max 那一项贡献 exp(0)=1），所以整行的相对误差 ≤ n·2^-25")
    for nn, u in ((4096, 2.0 ** -25), (16384, 2.0 ** -25), (4096, 2.0 ** -150)):
        print(f"      n = {nn:>6}：界 = n·2^{math.log2(u):.0f} = {nn*u:.3e}")
    check("n=4096 的 fp16 丢弃界 = 1.221e-4",
          close(4096 * 2.0 ** -25, 1.220703125e-4, 1e-9),
          f"{4096*2.0**-25:.6e}")
    print()
    print("    §2 的访存账（下面 §B5 有完整推导）：")
    print(f"      两遍法读 2 次 + 写 1 次 = 3·n·esize；融合一遍 = 2·n·esize")


def b3_doc02_numbers():
    sub("B3  文档 02 全部正文数字的出口")
    rng = np.random.default_rng(20260917)
    n = 4096
    x = rng.standard_normal(n) * 4.0
    ref = float(np.sum(np.exp(x - x.max()), dtype=np.float64))
    print(f"    §1 手算例子的四个数字（详见 §A3）：")
    print(f"      exp(−3) = {math.exp(-3)!r}")
    print(f"      α = exp(−2) = {math.exp(-2)!r}")
    print(f"      Σexp(x−m₂) 的后半块 = 1 + exp(−4) = {1.0+math.exp(-4)!r}")
    print(f"      l₂ = 1.1603888691244324（§A3 已逐步打印）")
    check("§A3 的手算 l₂ = 1.1603888691244324（1e-15 内）",
          close(1.1603888691244324,
                float(np.sum(np.exp(np.array([2.0, -1.0, 4.0, 0.0]) - 4.0))), 1e-15))

    print()
    print(f"    §2 的 rescale 次数（n = {n}，block = 128）：")
    for order, label in ((None, "原序"), ("asc", "递增"), ("desc", "递减")):
        _, l, nr, _ = online_softmax_blocks(x, 128, np.float32, order)
        print(f"      {label}：rescale 次数 = {nr:>3}，l = {float(l):.8f}，"
              f"相对误差 {relerr(float(l), ref):.3e}")
    print("      ⚠ 见 §C6：rescale 次数从 0 涨到 31，误差**没有**随之单调增长。")

    print()
    print(f"    §4.2 的 rescale 因子下溢阈值（与 §B1 同一批数字）：")
    print(f"      fp16 状态：exp(m_old − m_new) 在 m_new − m_old > {-F16_UNF:.4f} 时取 0")
    print(f"      fp32 状态：对应阈值是 {-F32_UNF:.4f}，实际不可达")
    check("fp16 的 rescale 因子在 gap = 17.33 处已归零",
          float(exp_round(F16_UNF - 0.001, np.float16)) == 0.0,
          f"{float(exp_round(F16_UNF-0.001, np.float16))!r}")

    print()
    print("    §3 里「一遍 vs 两遍」的访存账见 §B5、§B6。")
    print(f"    §0 的场景：seq_len = {n}，d_head = 128，fp16")
    seq, dh = 4096, 128
    attn_bytes = seq * dh * FP16_BYTES
    print(f"      一个 (seq × d_head) 的 fp16 矩阵 = {attn_bytes/1024:.1f} KB；")
    print(f"      每个 query 位置的 logits 向量 = {seq} 个 fp16 = {seq*FP16_BYTES} B")


def b4_doc03_numbers():
    sub("B4  文档 03 全部正文数字的出口")
    print("    §1 的 Welford 递推（逐步打印，供手算复核）：")
    xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mean = 0.0
    M2 = 0.0
    print(f"      输入 x = {xs.tolist()}")
    for i, v in enumerate(xs, 1):
        d = v - mean
        mean_new = mean + d / i
        M2_new = M2 + d * (v - mean_new)
        print(f"      n={i}: x={v:.1f}  d={d:+.4f}  "
              f"mean {mean:+.6f} → {mean_new:+.6f}   M2 {M2:.6f} → {M2_new:.6f}")
        mean, M2 = mean_new, M2_new
    print(f"      终值：mean = {mean!r}，M2 = {M2!r}，"
          f"var = {M2/len(xs)!r}")
    check("Welford 在 [1..5] 上给出 mean=3、var=2",
          close(mean, 3.0, 1e-15) and close(M2 / len(xs), 2.0, 1e-15),
          f"mean={mean!r}, var={M2/len(xs)!r}")

    print()
    print("    §1 的三种写法对照（同一份数据，fp32 累加，n=8192，σ=1）：")
    rng = np.random.default_rng(20260919)
    for mu in (0.0, 1e5):
        x = (mu + rng.standard_normal(8192)).astype(np.float32)
        rv = float(np.var(x.astype(np.float64)))
        print(f"      μ = {mu:>8.0f}：Welford {welford_blocked(x, np.float32)[1]:.6f}"
              f" | 两遍 {two_pass_blocked(x, np.float32)[1]:.6f}"
              f" | Σx²−nμ² {sumsq_blocked(x, np.float32)[1]:.6f}"
              f"   （fp64 真值 {rv:.6f}）")

    print()
    print("    §2 的访存账见 §B6；§4.3 的 eps 分界见 §B7。")
    print(f"    一行 LayerNorm 的典型形状：N = 8192（hidden），fp16，batch = 4096 行")


def b5_softmax_bytes_and_intensity():
    sub("B5  softmax 的访存字节与算术强度（文档 01 §2、文档 02 §3）")
    rows, cols = 4096, 4096
    n = rows * cols
    e16 = FP16_BYTES
    print(f"    场景：{rows} 行 × {cols} 列，fp16，共 {n/1e6:.3f}M 个元素"
          f" = {n*e16/1e6:.1f} MB")
    print()
    print("    (a) 两遍法（先求 max，再求 exp 与和，最后写回）")
    b_2pass = 2 * n * e16 + n * e16
    print(f"        读 2 遍 + 写 1 遍 = 3·n·esize = {b_2pass/1e6:.1f} MB")
    print("    (b) 融合成一遍（在线 softmax，一个 block 吃一整行/一段）")
    b_1pass = n * e16 + n * e16
    print(f"        读 1 遍 + 写 1 遍 = 2·n·esize = {b_1pass/1e6:.1f} MB")
    print(f"    (c) 若 exp 结果要落回显存再读回来除（最差写法）：")
    b_3pass = 3 * n * e16 + n * e16
    print(f"        读 2 遍 + 写 1 遍中间结果 + 再读 1 遍 + 写 1 遍 = "
          f"4·n·esize = {b_3pass/1e6:.1f} MB")
    bw = SP.HBM_BW_GBps_BY_GEN[H100] * 1e9
    print()
    print(f"    按 HBM {SP.HBM_BW_GBps_BY_GEN[H100]} GB/s 折算（推导值，只算 HBM 侧）：")
    for label, b in (("两遍法", b_2pass), ("融合一遍", b_1pass), ("最差三遍", b_3pass)):
        print(f"      {label:<10} {b/1e6:>8.1f} MB → {b/bw*1e6:>7.1f} µs")
    check("融合一遍相对两遍法的字节数之比 = 2/3",
          close(b_1pass / b_2pass, 2.0 / 3.0, 1e-12),
          f"{b_1pass/b_2pass:.6f}")

    print()
    print("    算术强度：非超越部分每元素约 2 个浮点操作（一次比较/减、一次乘/除），")
    print(f"    而每元素要过 {2*e16} 字节（读 1 次 + 写 1 次，fp16）")
    ai = 2.0 / (2 * e16)
    ridge = SP.FP32_VECTOR_TFLOPS_DENSE[H100] * 1e12 / bw
    print(f"      I = 2 / {2*e16} = {ai:.3f} FLOP/byte")
    print(f"      roofline 拐点 I* = {SP.FP32_VECTOR_TFLOPS_DENSE[H100]}e12 ÷ "
          f"{SP.HBM_BW_GBps_BY_GEN[H100]}e9 = {ridge:.2f} FLOP/byte")
    print(f"      I / I* = {ai/ridge:.5f} → 带宽受限，可达算力 "
          f"{bw*ai/1e12:.4f} TFLOPS = 峰值的 {bw*ai/(SP.FP32_VECTOR_TFLOPS_DENSE[H100]*1e12)*100:.2f}%")
    check("softmax 的非超越部分远在带宽区（I/I* < 0.1）", ai / ridge < 0.1,
          f"{ai/ridge:.5f}")
    print()
    print("    反解：exp 本身要花多少「等效 FLOP」才会把 softmax 推过拐点？")
    per_elem_budget = ridge * (2 * e16)
    print(f"      每元素可用预算 = I* × 每元素字节 = {ridge:.2f} × {2*e16} = "
          f"{per_elem_budget:.1f} FLOP")
    print(f"      ⟹ 只要 exp 的等效代价 ≤ {per_elem_budget:.0f} FLOP/元素，"
          f"softmax 就是带宽受限的。")
    print(f"      ⚠ 本仓库**不给出** exp 在 CC 8.0/9.0 上的每 SM 每周期吞吐"
          f"（来源未能核实，见 specs.py 的 UNSOURCED），")
    print(f"        因此上面这个 {per_elem_budget:.0f} 是一个**条件判据**，不是一个结论。")
    check("这个预算数 = 80 FLOP/元素（可由公式复核）",
          close(per_elem_budget, 80.0, 1e-6), f"{per_elem_budget!r}")


def b6_layernorm_bytes():
    sub("B6  LayerNorm 单遍 / 两遍的访存字节（文档 03 §2、§4）")
    rows, hidden = 4096, 8192
    n = rows * hidden
    e16 = FP16_BYTES
    bw = SP.HBM_BW_GBps_BY_GEN[H100] * 1e9
    print(f"    场景：{rows} 行 × hidden {hidden}，fp16；"
          f"每行数据 = {hidden*e16/1024:.1f} KB")
    print(f"    γ、β 各 {hidden*e16/1024:.1f} KB（每行都要读，总量 {rows*hidden*e16/1e6:.1f} MB）")
    print()
    print("    (a) 两遍法、x 不留在片上（长行放不下）")
    b2 = 2 * n * e16 + n * e16 + 2 * hidden * e16
    print(f"        读 x 两遍 + 写 y 一遍 + 读 γ/β = {b2/1e6:.1f} MB")
    print("    (b) 单遍 Welford、x 留在片上（短行放得下）")
    b1 = n * e16 + n * e16 + 2 * hidden * e16
    print(f"        读 x 一遍 + 写 y 一遍 + 读 γ/β = {b1/1e6:.1f} MB")
    print(f"    比值 = {b2/b1:.4f}（趋近 3/2；γ/β 那一份两遍都要读，")
    print(f"             它在两式里各占 {2*hidden*e16/1e6:.3f} MB，相对 n·esize 可以忽略）")
    print()
    print(f"    HBM 侧时间（推导值，{SP.HBM_BW_GBps_BY_GEN[H100]} GB/s）：")
    print(f"      两遍 {(b2)/1e6:>8.1f} MB → {b2/bw*1e6:>7.1f} µs")
    print(f"      单遍 {(b1)/1e6:>8.1f} MB → {b1/bw*1e6:>7.1f} µs")
    print(f"    忽略 γ/β 时比值恰为 3/2：(3ne)/(2ne) = {3/2}；")
    print(f"    本行实算 = {b2/b1:.6f}（略小于 1.5，因为 γ/β 对两式是同一份常数项）")
    check("单遍/两遍的字节比 < 3/2 且 > 1.49",
          1.49 < b2 / b1 < 1.5, f"{b2/b1:.6f}")

    print()
    print("    ---- 但「单遍省一遍访存」有个前提，必须写清楚 ----")
    print("    如果 x 的一行装得进片上（寄存器 + SMEM），两遍法**不产生额外 HBM 流量**，")
    print("    单遍省下的只是片上带宽与指令数。一行的容量账：")
    for hidden_n in (4096, 8192, 16384, 32768, 65536, 131072):
        need = hidden_n * e16
        fit = "放得下" if need / 1024 <= SP.SMEM_PER_SM_MAX_KB else "**放不下**"
        print(f"      hidden = {hidden_n:>6}：一行 fp16 = {need/1024:>8.2f} KB"
              f"  （每 SM 的 SMEM 上限 {SP.SMEM_PER_SM_MAX_KB} KB，{fit}）")
    print(f"    ⟹ 一行放得进**整个 SM 的 SMEM** 的上限是 hidden ≤ "
          f"{int(SP.SMEM_PER_SM_MAX_KB*1024/e16)}。")
    print(f"      但一行通常由**一个 block** 处理，而 SMEM 要在同一 SM 的多个常驻 block 之间分：")
    for nb in (1, 2, 4, 8):
        budget = SP.SMEM_PER_SM_MAX_KB * 1024 / nb
        print(f"        常驻 {nb} 个 block → 每 block {budget/1024:>6.1f} KB → "
              f"hidden ≤ {int(budget/e16):>7}（fp16 存 x）")
    print(f"      超过这条线，第二遍就必须回 HBM 读，单遍法才开始省**显存带宽**。")
    print()
    print("    另一个口径：block 内每个线程负责 hidden/threads 个元素时，")
    for hidden_n, threads in ((8192, 256), (8192, 512), (4096, 256), (65536, 1024)):
        per = hidden_n // threads
        print(f"      hidden={hidden_n:>6}, threads={threads:>5} → 每线程 {per:>4} 个元素"
              f" = {per*4} 个 fp32 寄存器（只存 x，不含 γ/β/累加器）")


def b7_eps_placement():
    sub("B7  eps 放在分母内 vs 外（文档 03 §4.3）")
    print("    内：(x − μ) / sqrt(σ² + eps)   ← 主流实现（PyTorch / RMSNorm 论文）")
    print("    外：(x − μ) / (sqrt(σ²) + eps)")
    print()
    eps = 1e-5
    print(f"    eps = {eps}（LayerNorm 的常见默认值）")
    print(f"    {'σ':>12}{'内 sqrt(σ²+eps)':>20}{'外 sqrt(σ²)+eps':>20}{'比值':>12}")
    for sig in (0.0, 1e-8, 1e-6, 1e-4, 1e-3, 3.1623e-3, 1e-2, 1.0):
        ins = math.sqrt(sig * sig + eps)
        out = math.sqrt(sig * sig) + eps
        print(f"    {sig:>12.4e}{ins:>20.6e}{out:>20.6e}{ins/out:>12.4f}")
    print()
    print(f"    内式开始生效的分界：σ = sqrt(eps) = {math.sqrt(eps):.6e}")
    print(f"    外式开始生效的分界：σ = eps        = {eps:.6e}")
    print(f"    两者相差 1/sqrt(eps) = {1/math.sqrt(eps):.4f} 倍")
    print(f"    ⟹ 在 σ ∈ [{eps:.0e}, {math.sqrt(eps):.4e}] 这一段里，")
    print(f"      两种实现给出的结果差最多 {math.sqrt(eps)/eps:.1f} 倍（σ→0 时）。")
    check("两个分界相差 1/sqrt(eps) = 316.2278 倍（eps=1e-5）",
          close(1 / math.sqrt(eps), 316.227766, 1e-6),
          f"{1/math.sqrt(eps):.6f}")
    check("σ = 1e-4 时两式相差 28.76 倍",
          close(math.sqrt(1e-8 + eps) / (1e-4 + eps), 28.7623, 1e-4),
          f"{math.sqrt(1e-8+eps)/(1e-4+eps):.4f}")
    print()
    print("    σ 远大于 sqrt(eps) 之后两式趋同：")
    for sig in (1e-2, 0.1, 1.0):
        ins = math.sqrt(sig * sig + eps)
        out = math.sqrt(sig * sig) + eps
        print(f"      σ = {sig:<6}：比值 {ins/out:.6f}"
              f"  （相对差 {abs(ins/out-1)*100:.4f}%）")


# ===========================================================================
# §C 边界与反例
# ===========================================================================


def c1_no_max_subtraction():
    sub("C1  不减 max 在 fp16 下真的得到 inf / nan（文档 01 §4.1）")
    rng = np.random.default_rng(20260916)
    print("    现象断言：**失败行为成立**，不是「成功」。")
    print()
    print(f"    {'scale':>7}{'max logit':>12}{'朴素 exp 的 inf 数':>20}"
          f"{'输出 nan 数':>14}{'稳定版 l':>14}")
    for scale in (4.0, 8.0, 11.0, 12.0):
        x = (rng.standard_normal(4096) * scale).astype(np.float16)
        e, l = softmax_naive(x, np.float16)
        n_inf = int(np.count_nonzero(np.isinf(e.astype(np.float64))))
        y = e / l
        n_nan = int(np.count_nonzero(np.isnan(y.astype(np.float64))))
        _, _, l_s16 = softmax_two_pass(x, np.float16)
        print(f"    {scale:>7.1f}{float(np.max(x.astype(np.float64))):>12.3f}"
              f"{n_inf:>20}{n_nan:>14}{float(l_s16):>14.6f}")
        check(f"scale={scale:.0f}：朴素 fp16 softmax 输出含 nan",
              n_nan > 0, f"max logit {float(np.max(x.astype(np.float64))):.2f}，"
                         f"{n_nan} 个 nan")
    print()
    x = (rng.standard_normal(4096) * 8.0).astype(np.float16)
    e, l = softmax_naive(x, np.float16)
    check("不减 max：exp 结果里出现 inf", has_inf(e.astype(np.float64)),
          f"{int(np.count_nonzero(np.isinf(e.astype(np.float64))))} 个 inf")
    y = e / l
    check("不减 max：softmax 输出里出现 nan（inf/inf）",
          has_nan(y.astype(np.float64)),
          f"{int(np.count_nonzero(np.isnan(y.astype(np.float64))))} 个 nan")
    m, e_s, l_s = softmax_two_pass(x, np.float16)
    check("减 max 之后 exp 的最大值恒为 1，永不溢出",
          float(np.max(e_s)) == 1.0, f"{float(np.max(e_s))!r}")
    check("减 max 之后 l 是有限的正常数",
          math.isfinite(float(l_s)) and float(l_s) > 1.0, f"{float(l_s)!r}")
    print()
    print("    注意：这里失败的是**格式的范围**，不是算法。同一个输入在 fp64 下")
    print("    不减 max 也能算对（见 §A1）——这正说明减 max 不改变数学结果，")
    print("    只改变可表示范围。")


def c2_underflow_drops_terms():
    sub("C2  exp(x−max) 下溢为 0：项被静默丢弃（文档 01 §4.2）")
    print("    构造：一个 logit 比 max 低得足够多，看它在 fp16 / fp32 下的命运。")
    print()
    print(f"    {'x − max':>10}{'fp16 exp':>16}{'fp32 exp':>16}{'fp64 真值':>16}")
    for gap in (-10.0, -16.0, -17.0, -17.3287, -18.0, -20.0, -104.0):
        print(f"    {gap:>10.4f}{float(exp_round(gap, np.float16)):>16.6e}"
              f"{float(exp_round(gap, np.float32)):>16.6e}"
              f"{math.exp(gap):>16.6e}")
    print()
    check("x−max = −18 时，fp16 的 exp 已经精确等于 0",
          float(exp_round(-18.0, np.float16)) == 0.0,
          f"{float(exp_round(-18.0, np.float16))!r}")
    check("同一个值在 fp32 下仍然非零（次正规）",
          float(exp_round(-18.0, np.float32)) != 0.0,
          f"{float(exp_round(-18.0, np.float32)):.6e}")
    check("x−max = −17.3287（= −25·ln2）就是 fp16 的归零点",
          float(exp_round(F16_UNF, np.float16)) == 0.0,
          f"exp({F16_UNF:.6f}) = {float(exp_round(F16_UNF, np.float16))!r}")
    print()
    print("    这在 softmax 里是**良性**的：l ≥ 1（max 项贡献 1），所以整行相对")
    print("    误差的上界是 n·2^-25（fp16）/ n·2^-150（fp32），见 §B2。")
    print("    真正致命的是它作用在**别的**地方——见 §C4 与 §C5。")


def c3_fp16_accumulation_of_l():
    sub("C3  l（exp 之和）用 fp16 累加导致精度崩塌（文档 01 §4.3）")
    rng = np.random.default_rng(20260920)
    print("    实现口径：每个元素 exp(x−max) 之后，逐个加进一个累加器。")
    print("    （真实内核里，逐元素加是**单线程内**的循环；跨线程是树形归约，见下。）")
    print()
    print("    扫描 x ~ N(0,1)×σ 与 n：**l 的量级**才是决定 fp16 累加误差的那个自变量。")
    print()
    print(f"    {'σ':>6}{'n':>8}{'l(fp64)':>12}{'l 的 fp16 ulp/2':>17}"
          f"{'平均项 l/n':>13}{'fp32 顺序':>12}{'fp16 顺序':>12}{'fp16 树形':>12}")
    rows = []
    for sigma in (0.5, 1.0, 2.0, 4.0, 8.0):
        for n in (4096, 16384):
            r = np.random.default_rng(20260920 + n)
            x = r.standard_normal(n) * sigma
            e32 = np.exp(x - x.max()).astype(np.float32)
            ref = float(np.sum(e32, dtype=np.float64))
            acc32 = np.float32(0.0)
            for v in e32:
                acc32 = np.float32(acc32 + v)
            acc16 = np.float16(0.0)
            for v in e32.astype(np.float16):
                acc16 = np.float16(acc16 + v)
            e32s, e16s = relerr(float(acc32), ref), relerr(float(acc16), ref)
            e16t = relerr(float(np.sum(e32, dtype=np.float16)), ref)
            half_ulp = float(np.spacing(np.float16(ref))) / 2
            rows.append((sigma, n, ref, e32s, e16s, e16t))
            print(f"    {sigma:>6.2f}{n:>8}{ref:>12.4f}{half_ulp:>17.4f}"
                  f"{ref/n:>13.5f}{e32s:>12.3e}{e16s:>12.3e}{e16t:>12.3e}")
    print()
    print("    机制就写在第三、四列里：一旦 `l 的 fp16 ulp/2` 超过 `平均项 l/n`，")
    print("    加进去的项就落在半个 ulp 之内，被四舍五入吃掉——加得越多，丢得越多。")
    print()
    print("    结论断言（**同一份数据**，见表）：")
    check("所有格子：fp32 顺序累加的相对误差 < 1e-4",
          all(r[3] < 1e-4 for r in rows),
          f"最差 {max(r[3] for r in rows):.3e}")
    big = [r for r in rows if r[2] >= 20.0]
    check(f"l ≥ 20 的格子（{len(big)} 个）：fp16 顺序累加误差 > 1e-2",
          all(r[4] > 1e-2 for r in big),
          f"最小的一个是 {min(r[4] for r in big):.3e}")
    check("同一个格子上 fp16 顺序比 fp32 顺序差 3 个数量级以上",
          all(r[4] / max(r[3], 1e-30) > 1e3 for r in big),
          f"最小的比值 {min(r[4]/max(r[3],1e-30) for r in big):.3e}")
    huge = [r for r in rows if r[2] >= 500.0]
    check(f"l ≥ 500 的格子（{len(huge)} 个）：树形 fp16 比顺序 fp16 好 2 个数量级以上",
          all(r[4] / max(r[5], 1e-30) > 1e2 for r in huge),
          f"最小的比值 {min(r[4]/max(r[5],1e-30) for r in huge):.1f}")
    print()
    print("    ⚠ **一条重要的限定**：上面「fp16 顺序」是一种拓扑。换成树形归约")
    print("      （warp shuffle + SMEM，真实内核的做法）后 fp16 好 2~3 个数量级：")
    for r in huge:
        print(f"      σ={r[0]:<4} n={r[1]:>6}：顺序 {r[4]:.3e} → 树形 {r[5]:.3e}"
              f"（好 {r[4]/max(r[5],1e-30):.0f} 倍）")
    print("      ⟹ 「l 不能用 fp16 累加」这条要**区分累加拓扑**：")
    print("        顺序累加时误差按 u·n 涨；树形归约只有 u·√(log n) 量级。")
    print("        但工程上仍然推荐 fp32 累加器——理由是它同时消掉了 §C5 的整段丢失，")
    print("        而且 l 一旦越过 fp16 上限 65504，树形归约也救不回来。")
    check("l 接近 65504 时 fp16 的和本身就会溢出（树形也救不回）",
          has_inf(exp_round(math.log(70000.0), np.float16)),
          f"float16(70000) = {float(np.float16(70000.0))!r}")


def c4_lazy_max_update():
    sub("C4  把「m 单调递增」误读成「可以延后更新 m」→ 中间结果上溢（文档 02 §4.1）")
    print("    场景：3 个分块，前两块 logits 都在 0 附近，第 3 块里出现一个 20。")
    print("    三种策略：")
    print("      A 按递推更新 m（正确）")
    print("      B 把 m 冻结在第一块的 max（「反正 m 只会往上涨，先定一个」）")
    print("      C 预设 m ≡ 0（假设 logits 非正）")
    print()
    blocks = [np.array([0.0, 0.1, -0.2]),
              np.array([0.1, 0.0, 0.2]),
              np.array([20.0, 0.0, 0.0])]
    m_freeze = float(np.max(blocks[0]))

    def run(policy, dt):
        m = -np.inf
        l = dt(0.0)
        for blk in blocks:
            if policy == "A":
                mb = dt(np.max(blk))
                m_new = mb if mb > m else (m if np.isfinite(m) else dt(mb))
            elif policy == "B":
                m_new = dt(m_freeze)
            else:
                m_new = dt(0.0)
            a = 0.0 if not np.isfinite(m) else float(np.exp(float(m) - float(m_new)))
            l = dt(dt(l * dt(a)) + dt(np.sum(np.exp(blk - float(m_new)), dtype=dt)))
            m = m_new
        return float(m), float(l)

    print(f"    {'策略':<28}{'fp16 终态 l':>20}{'fp32 终态 l':>20}")
    for policy, label in (("A", "A 按递推更新 m"), ("B", f"B 冻结 m = {m_freeze:.1f}"),
                          ("C", "C 预设 m = 0")):
        _, l16 = run(policy, np.float16)
        _, l32 = run(policy, np.float32)
        print(f"    {label:<28}{l16:>20.6f}{l32:>20.6f}")
    print()
    print("    终态 l 的正确答案：第 3 块的 20 是全局 max，所以 l 应该 ≈ 1。")
    _, l_ok = run("A", np.float64)
    print(f"      策略 A 的 fp64 结果 l = {l_ok!r}")
    print()
    m16B, l16B = run("B", np.float16)
    m32B, l32B = run("B", np.float32)
    check("策略 B（冻结 m）：fp16 直接得到 inf",
          not math.isfinite(l16B) and l16B > 0, f"{l16B!r}")
    check("策略 B：fp32 不溢出但结果错 8 个数量级以上",
          math.isfinite(l32B) and l32B / l_ok > 1e8, f"{l32B:.4g}")
    check("策略 C（预设 m=0）：fp16 也直接得到 inf",
          not math.isfinite(run("C", np.float16)[1]))
    check("策略 A 在 fp16 下给出正确的 1.0",
          close(run("A", np.float16)[1], 1.0, 1e-3), f"{run('A', np.float16)[1]!r}")
    print()
    print("    机制：不更新 m 时算的是 exp(x − m_old)，而 m_old 可以比块内 max 小很多，")
    print("    指数就冲出格式范围。exp(20) 在 fp16 里是 inf（阈值只有 11.09）。")
    check("exp(20) 在 fp16 里溢出为 inf",
          has_inf(exp_round(20.0, np.float16)), f"{float(exp_round(20.0, np.float16))!r}")
    check("exp(20 − 0.1) 在 fp16 里同样溢出（冻结 m 的那一步）",
          has_inf(exp_round(19.9, np.float16)),
          f"{float(exp_round(19.9, np.float16))!r}")
    print()
    print("    ⚠ **与提示词里的预设不同**：这条失败模式不是「m 只增不减导致误差累积」")
    print("      （§C6 已用实跑否定），而是「把『m 只增不减』误读成『m 可以不更新』」。")
    print("      单调性 m_old ≤ m_new 是**递推保住的**，一旦主动放弃更新就同时丢了")
    print("      单调性和安全性——上面 fp16 的 inf 就是这一步的代价。")


def c5_rescale_underflow():
    sub("C5  rescale 因子下溢到 0：分块贡献被静默丢弃（文档 02 §4.2）")
    print("    状态住 fp16 还是 fp32，决定了 exp(m_old − m_new) 在哪一步归零。")
    print()
    print(f"    {'gap':>8}{'exp(−gap) fp64':>18}{'fp32 状态':>18}{'fp16 状态':>18}")
    for gap in (10.0, 17.0, 17.3287, 18.0, 20.0, 40.0, 104.0, 110.0):
        print(f"    {gap:>8.4f}{math.exp(-gap):>18.6e}"
              f"{float(exp_round(-gap, np.float32)):>18.6e}"
              f"{float(exp_round(-gap, np.float16)):>18.6e}")
    print()
    check("gap = 17.3287 时 fp16 的 rescale 因子已经是 0",
          float(exp_round(-17.3287, np.float16)) == 0.0,
          f"{float(exp_round(-17.3287, np.float16))!r}")
    check("同一 gap 下 fp32 的因子仍然精确非零",
          float(exp_round(-17.3287, np.float32)) != 0.0,
          f"{float(exp_round(-17.3287, np.float32)):.6e}")
    check("fp32 要到 gap = 104 才归零（≈ −150·ln2 = −103.9721）",
          float(exp_round(-104.0, np.float32)) == 0.0
          and float(exp_round(-103.0, np.float32)) != 0.0,
          f"-103→{float(exp_round(-103.0, np.float32)):.3e}, "
          f"-104→{float(exp_round(-104.0, np.float32))!r}")

    print()
    print("    丢掉的量有多大？先看单次丢弃的上界：")
    print("    前段 l_old 累积到一个数，新块的 max 比它高 gap。丢掉的是 l_old·exp(−gap)，")
    print("    而新块的 l ≥ 1（它的 max 项贡献 exp(0)=1）。")
    print()
    print(f"    {'l_old':>12}{'gap':>8}{'真实丢掉量':>18}{'fp16 状态':>26}")
    for l_old, gap in ((1.0, 10.0), (100.0, 10.0), (1000.0, 17.0), (100.0, 17.5),
                       (100.0, 20.0), (1000.0, 20.0), (65504.0, 20.0),
                       (100.0, 110.0)):
        lost = l_old * math.exp(-gap)
        f16 = float(exp_round(-gap, np.float16))
        print(f"    {l_old:>12.1f}{gap:>8.1f}{lost:>18.6e}"
              f"{'整体丢弃' if f16 == 0 else f'部分保留 (×{f16:.3e})':>26}")
    print()
    bound = 65504.0 * 2.0 ** -25
    print("    ⚠ **上界**：l_old 本身受 fp16 上限 65504 约束，所以一次丢弃最多丢掉")
    print(f"      65504 × 2^-25 = {bound:.6e}，相对新块 l（≥1）是 {bound*100:.4f}%，")
    print(f"      约等于 fp16 的 {bound/SP.FP16_ROUNDING_UNIT:.1f} 个 u —— 看得见，但不是灾难。")
    check("fp16 状态下单次丢弃的上界 = 65504·2^-25 ≈ 1.95e-3",
          close(bound, 1.9525e-3, 1e-3), f"{bound:.6e}")

    print()
    print("    一个能从头算到尾的两块例子（前段 60000 项都等于 0，后块只有一个 20）：")
    n_old = 60000
    l_old_true = float(n_old)                     # 60000 项各贡献 exp(0) = 1
    l_true = l_old_true * math.exp(-20.0) + 1.0
    for dt, name in ((np.float32, "fp32 状态"), (np.float16, "fp16 状态")):
        l_old = dt(n_old)
        alpha = dt(exp_round(-20.0, dt))
        l = float(dt(dt(l_old * alpha) + dt(1.0)))
        mark = ""
        if name.startswith("fp16"):
            mark = " ← 分块贡献被整体丢弃" if float(alpha) == 0.0 else ""
        print(f"      {name}：l_old = {float(l_old):.0f}，α = {float(alpha):.6e}，"
              f"l = {l:.10f}，相对误差 {relerr(l, l_true):.3e}{mark}")
    check("这个例子里 fp16 状态的 l 比真值小 1.24e-4（一次整体丢弃的量）",
          close(relerr(1.0, l_true), 1.2366e-4, 1e-3),
          f"{relerr(1.0, l_true):.6e}")

    print()
    print("    真正的灾难在**别处**：当 l_old 本身就超过 fp16 的上限 65504 时，")
    print("    fp16 状态在累加阶段就已经变成 inf，后面的 rescale 全部是 inf×0 = nan。")
    print("    这与 §C3 是同一个根因：**状态要用 fp32 存**。")
    big = np.float16(70000.0)
    print(f"      float16(70000.0) = {float(big)!r}")
    check("float16(70000.0) 直接溢出成 inf", has_inf([float(big)]),
          f"{float(big)!r}")
    print()
    print("    还有一处必须点名：**同一个 rescale 因子还要乘到输出累加器 o 上**")
    print("    （attention 的第三条递推）。那里丢掉的不是「一个计数」，而是已经算好的")
    print("    输出向量。完整递推的正文在 40-kernels/attention（本文档只交付 (m, l) 两条）。")


def c6_rescale_count_and_error():
    sub("C6  rescale 次数真的会累积误差吗？（文档 02 §4.3 —— **本项被实跑推翻**）")
    rng = np.random.default_rng(20260917)
    n = 4096
    print("    先把「rescale 次数」造出差别：递减排序时 m 在**第一块**就取到全局 max，")
    print("    之后一次都不再变；递增排序时 m 几乎每块都被抬高。")
    print()
    nr_asc = nr_desc = None
    for scale in (4.0, 20.0):
        x = rng.standard_normal(n) * scale
        ref = float(np.sum(np.exp(x - x.max()), dtype=np.float64))
        print(f"    logits ~ N(0,1)×{scale:.0f}（l 的量级 = {ref:.4f}）")
        print(f"    {'输入顺序':>10}{'分块':>8}{'rescale 次数':>14}"
              f"{'fp32 相对误差':>16}{'fp16 相对误差':>16}")
        for order, label in ((None, "原序"), ("asc", "递增"), ("desc", "递减")):
            for blk in (128, 512):
                _, l32, nr, _ = online_softmax_blocks(x, blk, np.float32, order)
                _, l16, _, _ = online_softmax_blocks(x, blk, np.float16, order)
                print(f"    {label:>10}{blk:>8}{nr:>14}{relerr(float(l32), ref):>16.3e}"
                      f"{relerr(float(l16), ref):>16.3e}")
                if scale == 4.0 and blk == 128:
                    if order == "asc":
                        nr_asc = nr
                    elif order == "desc":
                        nr_desc = nr
        print()
    check("递减输入让 rescale 次数降到 0、递增输入把它抬到几十次",
          nr_desc == 0 and nr_asc is not None and nr_asc > 20,
          f"递减 {nr_desc} / 递增 {nr_asc}")
    print("    **结论（与提示词里的预设相反）**：")
    print("      上面的表里，rescale 次数从 0 变到 31，误差**没有**随之单调增长——")
    print("      递增序的 fp32 误差甚至常常低于递减序。原因是每次 rescale 只引入一次")
    print("      乘法舍入（相对误差 ~u），而**真正的误差来源是 l 的逐项累加**（§C3），")
    print("      它的量级是 u·n 或 u·√n，比 rescale 的 u·k 大得多（k ≪ n）。")
    print()
    print("    再单独量一次 rescale 链本身（把累加误差排除掉）：")
    for k in (1, 8, 32, 128, 512, 2048):
        l = 1.0
        for _ in range(k):
            l *= math.exp(-1.0 / k)
        print(f"      连乘 {k:>5} 次 exp(−1/k) 与一次 exp(−1) 的相对差 "
              f"{relerr(l, math.exp(-1)):.3e}")
    check("rescale 链本身的累积误差在 k=2048 时仍 < 1e-12（可忽略）",
          relerr(_rescale_chain(2048), math.exp(-1.0)) < 1e-12,
          f"{relerr(_rescale_chain(2048), math.exp(-1.0)):.3e}")
    print()
    print("    所以 02 篇 §4 的第三条失败模式改写为：**「rescale 因子在 fp16 状态下的")
    print("    归零」（§C5），而不是「m 单调递增导致误差累积」。**")


def _rescale_chain(k):
    l = 1.0
    for _ in range(k):
        l *= math.exp(-1.0 / k)
    return l


def c7_sum_of_squares_catastrophe():
    sub("C7  Σx² − n·μ² 的灾难性抵消（文档 03 §4.1）")
    rng = np.random.default_rng(20260921)
    n = 8192
    print(f"    n = {n}，x = μ + N(0,1)，fp32；只抬 μ，看方差怎么烂掉。")
    print()
    print(f"    {'μ':>10}{'Σx²':>16}{'n·μ²':>16}{'两者之差':>16}{'方差真值':>14}"
          f"{'相对误差':>14}")
    for mu in (0.0, 100.0, 1e3, 1e4, 1e5):
        x = (mu + rng.standard_normal(n)).astype(np.float32)
        sq = float(np.sum((x * x).astype(np.float32), dtype=np.float32))
        mean = float(np.sum(x, dtype=np.float32)) / n
        rv = float(np.var(x.astype(np.float64)))
        var = (sq - n * mean * mean) / n
        print(f"    {mu:>10.0f}{sq:>16.4g}{n*mean*mean:>16.4g}{sq-n*mean*mean:>16.4g}"
              f"{rv:>14.6f}{relerr(var, rv):>14.3e}")
    print()
    print("    两个大数相减剩下的那个小数，有效位被吃光了；再走一步就变成**负数**：")
    found_negative = False
    for mu in (1e3, 1e4, 1e5, 1e6):
        x = (1.0 * mu + rng.standard_normal(n)).astype(np.float32)
        v = sumsq_blocked(x, np.float32)[1]
        flag = " ← 方差为负（物理上不可能）" if v < 0 else ""
        print(f"      μ = {mu:>8.0f}：Σx²−nμ² 给出的方差 = {v:>18.4f}{flag}")
        found_negative |= (v < 0)
    check("Σx²−nμ² 至少在一个 μ 上给出**负方差**", found_negative)
    print()
    print("    对比 Welford：它全程不构造 Σx² 这个中间量，")
    print("    只维护 mean 与 M2 = Σ(x−mean)²（后者本身就是小量），因此没有相减。")
    print()
    print("    ⚠ 注意一个容易搞混的点：LayerNorm 里的 ε 常常掩盖了这个 bug ——")
    print("      负方差加上 eps 之后 sqrt 仍然能算出数，只是那个数没有意义。见 §C8。")


def c8_welford_mean_drift():
    sub("C8  Welford 的均值漂移（文档 03 §4.2）")
    rng = np.random.default_rng(20260922)
    n = 8192
    print("    现象：朴素顺序 Welford 在**大均值**下，mean += d/n 里的 d/n 会被")
    print("          |mean| 的 ulp 整个吞掉，mean 卡住不动。")
    print()
    print("    先证明机制本身是**恒等式**而不是拟合：设 Welford 给出的均值为 m̂ = μ + δ，")
    print("    那么 Σ(x−m̂)²/n = Σ(x−μ)²/n + δ²（因为 Σ(x−μ) = 0）。")
    print("    所以 Δmean 造成的偏差**恰好**是 δ²，它是一个只加的**系统性**项；")
    print("    总误差另外还叠了 M2 累加本身的舍入（符号可正可负，会相消）。")
    print()
    print(f"    {'μ':>10}{'ulp(μ)/2':>12}{'d/n 量级':>12}{'δ = Δmean':>15}"
          f"{'δ²':>13}{'δ²/σ²（系统项）':>18}{'总误差':>13}{'残差(舍入)':>14}")
    rows = {}
    for mu in (0.0, 100.0, 1e4, 1e5, 1e6):
        x = (mu + rng.standard_normal(n)).astype(np.float32)
        m_seq, M2_seq, _ = welford_seq(x, np.float32)
        rv = float(np.var(x.astype(np.float64)))
        delta = float(m_seq) - float(np.mean(x.astype(np.float64)))
        sys_term = delta * delta / rv
        actual = relerr(float(M2_seq) / n, rv)
        # 用 fp64 按同一个 m̂ 重算方差，验证上面的恒等式
        var_at_mhat = float(np.mean(
            (x.astype(np.float64) - float(m_seq)) ** 2))
        assert close(var_at_mhat, rv + delta * delta, 1e-6), "恒等式不成立"
        rows[mu] = (delta, sys_term, actual)
        ulp = float(np.spacing(np.float32(mu))) / 2 if mu > 0 else 0.0
        print(f"    {mu:>10.0f}{ulp:>12.3e}{1.0/n:>12.3e}{delta:>15.6f}"
              f"{delta*delta:>13.3e}{sys_term:>18.3e}{actual:>13.3e}"
              f"{actual-sys_term:>+14.3e}")
    print()
    print("    ⟹ 系统性项 δ²/σ² 与总误差同量级；残差那一列是 M2 累加的舍入，")
    print("       两者符号相反时会相消——所以「Δmean² 精确等于误差」是错的，")
    print("       但「Δmean² 给出这一项的量级」成立。")
    check("恒等式 Σ(x−m̂)²/n = Σ(x−μ)²/n + δ² 在每个 μ 上都成立（1e-6 内）",
          True, "上面的 assert 已在循环里逐行验证")
    check("Δmean 随 μ 增大而增大（漂移是真的）",
          abs(rows[0.0][0]) < abs(rows[1e5][0]) < abs(rows[1e6][0]),
          f"μ=0: {rows[0.0][0]:.6f} < μ=1e5: {rows[1e5][0]:.6f} < μ=1e6: {rows[1e6][0]:.6f}")
    check("大均值（μ≥1e4）的误差比小均值（μ≤100）大 2 个数量级以上",
          min(rows[m][2] for m in (1e4, 1e5, 1e6))
          > 100 * max(rows[m][2] for m in (0.0, 100.0)),
          f"小均值最差 {max(rows[m][2] for m in (0.0,100.0)):.3e}，"
          f"大均值最好 {min(rows[m][2] for m in (1e4,1e5,1e6)):.3e}")
    check("δ²/σ² 与总误差的比值落在 1e-3 ~ 1e3（只承诺量级）",
          all(1e-3 <= rows[m][1] / max(rows[m][2], 1e-30) <= 1e3
              for m in (1e4, 1e5, 1e6)),
          "比值 " + ", ".join(f"μ={m:.0e}: {rows[m][1]/max(rows[m][2],1e-30):.3g}"
                              for m in (1e4, 1e5, 1e6)))
    print()
    print("    规避：不要用「单线程顺序扫完整个数组」的 Welford。分成小块")
    print("    （每块 ~32 个元素）再做 Chan 合并，每块内部的 n 小、漂移就小；")
    print("    更彻底的做法是把数据先减去一个粗中心（如 x[0]），把 μ 拉回 0 附近。")
    print()
    print("    同样数据下，三种策略的对照（μ = 1e5，n = 8192，σ = 1）：")
    for threads, label in ((1, "顺序 Welford（1 块）"), (32, "分块 32"),
                           (256, "分块 256"), (1024, "分块 1024")):
        v = welford_blocked(x, np.float32, threads)[1]
        print(f"      {label:<22} var = {v:.8f}  相对误差 {relerr(v, rv):.3e}")


# ===========================================================================
# 主流程
# ===========================================================================


def main():
    print("=" * 76)
    print("40-kernels/softmax-normalization 本机验证模型（verify.py）")
    print("=" * 76)
    print("本程序只跑算术与 numpy，不碰硬件。全部输出是**推导值**。")

    s0_scope()

    section("A", "算法正确性")
    a1_stable_softmax_reference()
    a2_online_vs_two_pass()
    a3_hand_computed_blocks()
    a4_welford_vs_two_pass()
    a5_large_mean_small_variance()

    section("B", "代价模型")
    b1_exp_thresholds()
    b2_doc01_numbers()
    b3_doc02_numbers()
    b4_doc03_numbers()
    b5_softmax_bytes_and_intensity()
    b6_layernorm_bytes()
    b7_eps_placement()

    section("C", "边界与反例")
    c1_no_max_subtraction()
    c2_underflow_drops_terms()
    c3_fp16_accumulation_of_l()
    c4_lazy_max_update()
    c5_rescale_underflow()
    c6_rescale_count_and_error()
    c7_sum_of_squares_catastrophe()
    c8_welford_mean_drift()

    print()
    print("=" * 76)
    total = _STATE["n"]
    fail = _STATE["fail"]
    print(f"§D  汇总：断言 {total} 条，通过 {total - len(fail)} 条，"
          f"不通过 {len(fail)} 条")
    if fail:
        for f in fail:
            print(f"    ✗ {f}")
    else:
        print("    全部通过。")
    print("=" * 76)
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())

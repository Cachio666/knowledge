"""
90-training 技术可行性探测（不改仓库，只验证「训练系统的账」能不能在本机测出来）。

要回答：
  A. Adam + L2 与 AdamW（decoupled weight decay）的差异 —— 真能测出来吗？量级多少？
  B. fp16 的溢出边界与 loss scaling —— 真能测出来吗？
  C. He vs Xavier 初始化 —— 前向激活的方差真能测出来吗？
  D. 反向传播 vs 数值梯度 —— 差多少？
  E. gradient checkpointing 的显存账 —— √n 那条成立吗？
  F. bf16 vs fp16 的数值行为（动态范围 vs 精度）—— 真能测出来吗？
"""
import numpy as np

rng = np.random.default_rng(90)


def show(tag, msg):
    print(f"[{tag}] {msg}")


# ============================================================
# A. Adam + L2  vs  AdamW
# ============================================================
def adam_l2(grads, lr=0.01, b1=0.9, b2=0.999, eps=1e-8, wd=0.1, w0=1.0):
    """Adam 的 L2：把 wd*w 加进梯度，**进 m 和 v**。"""
    w = np.full(1, w0)
    m = np.zeros(1); v = np.zeros(1)
    for t, g in enumerate(grads, 1):
        g = np.array([g]) + wd * w            # ← L2 项进梯度
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mh = m / (1 - b1 ** t); vh = v / (1 - b2 ** t)
        w = w - lr * mh / (np.sqrt(vh) + eps)
    return float(w[0])


def adamw(grads, lr=0.01, b1=0.9, b2=0.999, eps=1e-8, wd=0.1, w0=1.0):
    """AdamW：wd 不进 m/v，**直接乘在权重上**。"""
    w = np.full(1, w0)
    m = np.zeros(1); v = np.zeros(1)
    for t, g in enumerate(grads, 1):
        g = np.array([g])                        # ← 梯度只有真梯度
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mh = m / (1 - b1 ** t); vh = v / (1 - b2 ** t)
        w = w - lr * (mh / (np.sqrt(vh) + eps) + wd * w)   # ← 解耦
    return float(w[0])


# 用「梯度恒为 0」这个最干净的对照：没有自适应缩放干扰
g0 = [0.0] * 200
a_l2, a_w = adam_l2(g0), adamw(g0)
show("A1", f"★ 梯度恒为 0 时（wd=0.1）：Adam+L2 → w={a_l2:.6f}   AdamW → w={a_w:.6f}")
show("A1", f"     ⇒ 差的绝对值 = {abs(a_l2-a_w):.6f}；"
           f"「真解」应是 (1-lr*wd)^200 = {(1-0.01*0.1)**200:.6f}")

# 用真实梯度再看一次
g_real = list(rng.standard_normal(200) * 0.1)
b_l2, b_w = adam_l2(g_real), adamw(g_real)
show("A2", f"随机梯度 200 步：Adam+L2 → {b_l2:.6f}   AdamW → {b_w:.6f}   差 {abs(b_l2-b_w):.6f}")

# ============================================================
# B. fp16 的溢出边界与 loss scaling
# ============================================================
F16_MAX = np.finfo(np.float16).max
F16_TINY = np.nextafter(np.float16(0), np.float16(1))   # 最小次正规数
F16_NORM = np.finfo(np.float16).tiny                     # 最小**正规**数
F32_MAX = np.finfo(np.float32).max
F32_TINY = np.finfo(np.float32).tiny                     # fp32 最小正规数
F32_EPS = np.finfo(np.float32).eps
show("B1", f"fp16: max={F16_MAX:.0f}  最小次正规={float(F16_TINY):.3e}  "
           f"最小正规={float(F16_NORM):.3e}")
show("B1", f"fp32: eps={F32_EPS:.3e}")
# ★★ 入仓前的修正（2026-09-23）—— 这一行的除法**溢出**，而且是**两处**，不是一处。
#   本行原稿是：
#       f"{np.log10(F16_MAX/float(F16_TINY)):.1f} 个数量级（fp16）vs "
#       f"{np.log10(np.finfo(np.float32).max/np.finfo(np.float32).tiny):.1f} 个（fp32）"
#   实跑打印 `inf 个数量级（fp16）vs inf 个（fp32）`，并报两条
#   `RuntimeWarning: overflow encountered in scalar divide`。
#   ★ 原因**不是**「两个操作数都是 np.float16」——`float(F16_TINY)` 是 Python float。
#     真因是 **NumPy 2.x 的 NEP 50 弱标量提升**：Python 标量**不提升**数组/标量的 dtype，
#     于是 `np.float16 / float` 仍在 **fp16** 里做除法（`65504 / 5.96e-8 ≈ 1.10e12 > 65504`）⇒ inf。
#     证据（一行可复核）：`(np.finfo(np.float16).max / float(np.finfo(np.float16).tiny)).dtype`
#     → **`float16`**（★ 这一行本身也会溢出、也会报 warning，这里只用它的 **dtype**）。
#   ★ fp32 那一行**同类溢出**：两个操作数都是 `np.float32` 标量 ⇒ 除法在 fp32 里做
#     （`3.40e38 / 1.18e-38 ≈ 2.89e76 > 3.40e38`）⇒ 也是 inf。
#   ⇒ 修法：**显式转 float64** 再除（并把三个数的口径各自标出来）。
show("B1", "★ 动态范围（★ 三个数**各自标出口径**，不可两两相减）：")
show("B1", f"    fp16 正规区间  = log10(65504 / 2^-14)      = "
           f"{np.log10(float(F16_MAX) / float(F16_NORM)):.2f} 个数量级")
show("B1", f"    fp16 含次正规  = log10(65504 / 5.96e-8)     = "
           f"{np.log10(float(F16_MAX) / float(F16_TINY)):.2f} 个数量级")
show("B1", f"    fp32 正规区间  = log10(3.40e38 / 1.18e-38)  = "
           f"{np.log10(float(F32_MAX) / float(F32_TINY)):.2f} 个数量级")

# 梯度下溢：多小的梯度在 fp16 里会变成 0？
show("B2", "★ 一个梯度值在 fp16 下的命运：")
for g in (1e-3, 1e-5, 1e-7, 6.1e-8, 5.9e-8, 1e-8, 1e-10):
    h = np.float16(g)
    show("B2", f"    g={g:.2e} → fp16 {float(h):.3e}   {'★ 变成 0' if h == 0 else ''}")

# loss scale 的作用：把下溢的梯度抬起来
scale = 1024.0
show("B3", f"★ loss scaling ×{scale:.0f} 之后：")
for g in (1e-7, 1e-8, 1e-9):
    gs = g * scale
    h = np.float16(gs)
    back = float(h) / scale
    show("B3", f"    g={g:.2e} → 缩放后 {gs:.2e} → fp16 {float(h):.3e} → 还原 {back:.3e}"
               f"  相对误差 {abs(back-g)/g:.3e}")

# ============================================================
# C. He vs Xavier 初始化
# ============================================================
def init_std(fan_in, fan_out, kind):
    if kind == "he":
        return np.sqrt(2.0 / fan_in)
    if kind == "xavier":
        return np.sqrt(2.0 / (fan_in + fan_out))
    raise ValueError(kind)


show("C1", "★ 初始化后前向激活的方差（1000 次抽样，输入方差 1）")
show("C1", f"  {'fan_in':>8} {'fan_out':>8} {'He 实测':>12} {'He 理论':>10} "
           f"{'Xavier 实测':>14} {'Xavier 理论':>12}")
N = 1000
for fan_in, fan_out in ((256, 256), (1024, 1024), (256, 4096), (4096, 256)):
    row = []
    for kind in ("he", "xavier"):
        std = init_std(fan_in, fan_out, kind)
        outs = []
        for _ in range(N):
            W = rng.standard_normal((fan_in, fan_out)) * std
            x = rng.standard_normal(fan_in)
            outs.append((x @ W).var())
        row.append((np.mean(outs), kind))
    he_m, xa_m = row[0][0], row[1][0]
    he_t = 1.0 * (2.0 / fan_in) * fan_in      # Var = fan_in * std^2 * Var(x) = 2
    xa_t = 1.0 * (2.0 / (fan_in + fan_out)) * fan_in
    show("C1", f"  {fan_in:>8} {fan_out:>8} {he_m:>12.4f} {he_t:>10.4f} "
               f"{xa_m:>14.4f} {xa_t:>12.4f}")
show("C1", "  ⇒ He 让**前向**方差保持 2；Xavier 让 (fan_in+fan_out)/2 的折中 —— 后者在 "
           "fan_in≠fan_out 时既不保前向也不保反向")

# ============================================================
# D. 反向传播 vs 数值梯度
# ============================================================
def fwd(x, W1, W2):
    h = np.tanh(W1 @ x)
    return float(W2 @ h), (x, W1, W2, h)


def back(x, W1, W2, y):
    """手写反向。loss = y（对 y 求梯度 = 1）"""
    _, (x_, W1_, W2_, h) = fwd(x, W1, W2)
    dy = 1.0
    dW2 = dy * h
    dh = dy * W2_
    dz = dh * (1 - h * h)          # tanh'
    dW1 = np.outer(dz, x_)
    return dW1, dW2


def numgrad(x, W1, W2, idx, h=1e-6):
    def L():
        return fwd(x, W1, W2)[0]
    if idx[0] == 1:
        W1p, W1m = W1.copy(), W1.copy()
        W1p[idx[1], idx[2]] += h; W1m[idx[1], idx[2]] -= h
        return (fwd(x, W1p, W2)[0] - fwd(x, W1m, W2)[0]) / (2 * h)
    W2p, W2m = W2.copy(), W2.copy()
    W2p[idx[1]] += h; W2m[idx[1]] -= h
    return (fwd(x, W1, W2p)[0] - fwd(x, W1, W2m)[0]) / (2 * h)


d, m = 8, 6
x = rng.standard_normal(d)
W1 = rng.standard_normal((m, d)) * 0.5
W2 = rng.standard_normal(m) * 0.5
dW1, dW2 = back(x, W1, W2, None)
worst = 0.0
for i in range(m):
    for j in range(d):
        g_n = numgrad(x, W1, W2, (1, i, j))
        worst = max(worst, abs(g_n - dW1[i, j]) / (abs(g_n) + 1e-30))
for i in range(m):
    g_n = numgrad(x, W1, W2, (2, i, 0))
    worst = max(worst, abs(g_n - dW2[i]) / (abs(g_n) + 1e-30))
show("D1", f"★ 手写反向 vs 中心差分：最大相对差 = {worst:.3e}（{(m*d+m)} 个元素）")

# ============================================================
# E. gradient checkpointing 的账
# ============================================================
show("E1", "★ 存活的激活数：不 checkpoint vs 全 checkpoint（n 层）")
show("E1", f"  {'n':>6} {'不 ckpt':>10} {'全 ckpt':>10} {'√n 开方':>10} {'省了':>10}")
for n in (8, 32, 64, 128, 1024):
    show("E1", f"  {n:>6} {n:>10} {int(np.ceil(np.sqrt(n))):>10} "
               f"{np.sqrt(n):>10.2f} {n/np.ceil(np.sqrt(n)):>9.1f}×")
show("E1", "  ⇒ 「全 checkpoint 存 √n」是**分段策略**的结果（每 √n 层存一个边界，段内重算）")

# ============================================================
# F. bf16 vs fp16 的动态范围
# ============================================================
show("F1", "★ bf16 与 fp16 的对比（两者都是 16 bit，但分配方式不同）")
show("F1", f"  {'格式':>8} {'指数位':>8} {'尾数位':>8} {'最大':>14} {'最小正规':>14} {'eps':>12}")
for name, dt in (("fp16", np.float16), ("bf16", None)):
    if dt is not None:
        fi = np.finfo(dt)
        show("F1", f"  {name:>8} {fi.nexp:>8} {fi.nmant:>8} {fi.max:>14.4e} "
                   f"{fi.tiny:>14.4e} {fi.eps:>12.4e}")
    else:
        # numpy 未必有 bf16 的 finfo —— 用 ml_dtypes 或手工
        try:
            import ml_dtypes  # noqa
            fi = np.finfo(ml_dtypes.bfloat16)
            show("F1", f"  {name:>8} {fi.nexp:>8} {fi.nmant:>8} {fi.max:>14.4e} "
                       f"{fi.tiny:>14.4e} {fi.eps:>12.4e}")
        except Exception:
            show("F1", "  bf16: numpy 无 finfo ⇒ 按 IEEE 定义手工：指数 8 位、尾数 7 位，"
                       "max ≈ 3.39e38、eps = 2^-8 = 0.00390625")
            show("F1", f"  {'bf16':>8} {8:>8} {7:>8} {3.3895313892515355e38:>14.4e} "
                       f"{1.1754943508222875e-38:>14.4e} {2**-8:>12.4e}")
show("F1", "  ⇒ ★ fp16 精度高但范围窄（max 65504）；bf16 范围与 fp32 同但精度低（eps 0.0039）"
           " —— **这就是「为什么反向用 bf16 更省心」的全部理由**")

# ============================================================
# G. 一个可手算的闭式：fp16 下梯度下溢的比例
# ============================================================
show("G1", "★ 若梯度服从 N(0, σ)，多大比例的 |g| 在 fp16 下变成 0（下溢到次正规以下）？")
F16_SUB = float(F16_TINY)
for sigma in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
    from math import erf, sqrt
    p = erf(F16_SUB / (sigma * sqrt(2)))     # P(|g| < 下溢阈值) ≈ 2*Φ(x)-1
    show("G1", f"  σ={sigma:.0e}: 下溢比例 ≈ {p*100:.3f}%   "
               f"（loss scale ×1024 后 σ_eff={sigma*1024:.0e} ⇒ 下溢比例 "
               f"{erf(F16_SUB/(sigma*1024*sqrt(2)))*100:.3f}%）")

print("\n" + "=" * 74)
print(f"探测完成。numpy {np.__version__}")

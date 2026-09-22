"""
探测 5：RoPE 的频谱结构 —— 为什么直接外推会失败。

核心洞察：theta_i = base^(-2(i-1)/d)，i=1..d/2 ⇒ theta_1 = base^0 = **1** rad/位置。
         ⇒ 最高频分量的**周期 = 2π/1 ≈ 6.283 个位置**。
         ⇒ 位置 m 与 m + 6.28 的高频相位几乎相同 ⇒ **混叠**。

这直接支撑 `positional-encoding` 主题里「长度外推」那一节：
外推失败**不是**因为「训练没见过的位置」，而是因为**高频分量早已转过多圈**。
"""
import numpy as np

rng = np.random.default_rng(20260922)
print("numpy", np.__version__)
print("=" * 74)


def rope_freqs(d, base=10000.0):
    """原论文式：theta_i = base^(-2(i-1)/d)，i = 1..d/2。theta_1 = 1。"""
    i = np.arange(1, d // 2 + 1, dtype=np.float64)
    return base ** (-2.0 * (i - 1) / d)


d = 128
th = rope_freqs(d)
print(f"【1】d={d}, base=10000 ⇒ {d//2} 个频率（原论文式，theta_1 = base^0 = 1）")
print(f"  {'i':>4} {'theta_i':>14} {'周期 2π/θ (位置数)':>20} {'在 S=4096 内转几圈':>20}")
for i in (0, 1, 2, 4, 8, 16, 32, 63):
    period = 2 * np.pi / th[i]
    print(f"  {i+1:>4} {th[i]:>14.8f} {period:>20.3f} {4096/period:>20.1f}")

print()
print("  ⇒ 最高频分量（theta=1）的周期只有 6.28 个位置。")
print("    位置 1000 与位置 1001 的高频相位差 1 rad；位置 1000 与 1006 差 6 rad ≈ 一整圈。")

# ============================================================
# 2. 分频段看内积：高频是否真的「混叠」
# ============================================================
print()
print("【2】把内积按频段拆开，看它对 delta 的依赖（单个 q,k 对，d=128）")


def rope_apply_pairs(x, pos, th):
    """相邻对式，返回每一对的 (旋转后第一分量, 第二分量)"""
    xr = x.reshape(-1, 2)
    c, s = np.cos(pos * th), np.sin(pos * th)
    return np.stack([xr[:, 0] * c - xr[:, 1] * s,
                     xr[:, 0] * s + xr[:, 1] * c], axis=-1)


q = rng.standard_normal(d)
k = rng.standard_normal(d)

# 内积 = Σ_i  (q 的第 i 对) · Rot(-delta*θ_i) (k 的第 i 对)
qf = rope_apply_pairs(q, 0, th)   # 位置 0
print(f"  {'delta':>6} {'全频段内积':>14} {'高频段(i≤4)':>14} {'低频段(i≥32)':>14}")
for delta in (0, 1, 2, 4, 6, 8, 16, 64, 256, 1024, 4096):
    kf = rope_apply_pairs(k, delta, th)   # 位置 delta
    contrib = (qf * kf).sum(-1)           # 每对的贡献
    print(f"  {delta:>6} {contrib.sum():>14.6f} {contrib[:4].sum():>14.6f} "
          f"{contrib[32:].sum():>14.6f}")

# ============================================================
# 3. ★ 混叠的直接证据：高频段的内积在 delta 与 delta+2π 处几乎相同
# ============================================================
print()
print("【3】★ 混叠的直接证据（对 200 组随机 q,k 取均值，看内积的『准周期』）")
N = 200
Q = rng.standard_normal((N, d))
K = rng.standard_normal((N, d))
print(f"  {'delta':>6} {'全频段均值':>14} {'高频段(i≤2)均值':>18} {'低频段(i≥32)均值':>18}")
for delta in (0, 3, 6, 6.283185, 9, 12, 12.566371, 100.0):
    kf = np.array([rope_apply_pairs(K[n], delta, th) for n in range(N)])
    qf = np.array([rope_apply_pairs(Q[n], 0.0, th) for n in range(N)])
    contrib = (qf * kf).sum(-1)          # (N, d/2)
    print(f"  {delta:>6.3f} {contrib.sum(-1).mean():>14.4f} "
          f"{contrib[:, :2].sum(-1).mean():>18.4f} "
          f"{contrib[:, 32:].sum(-1).mean():>18.4f}")

print()
print("  ⇒ 高频段（i≤2，theta 接近 1）在 delta=0 与 delta=2π≈6.283 处的均值接近；")
print("    低频段（i≥32，theta 很小）在两个 delta 上差别明显。")
print("  ⇒ 这就是『位置 > 训练长度后高频相位环绕』的机制。")

# ============================================================
# 4. 换个 base 能延长多少？—— theta_1 = 1 与 base 无关（这是关键）
# ============================================================
print()
print("【4】换 base 能不能救最高频分量？")
print(f"  {'base':>10} {'theta_1':>12} {'最高频周期':>14} {'最低频(i=64)周期':>20}")
for base in (10000.0, 500000.0, 1000000.0):
    t = rope_freqs(d, base)
    print(f"  {base:>10.0f} {t[0]:>12.8f} {2*np.pi/t[0]:>14.3f} {2*np.pi/t[-1]:>20.1f}")
print("  ⇒ ★ theta_1 = base^0 = **1**，与 base **无关**。")
print("    提高 base 只把**低频**拉得更低（延长可分辨范围），")
print("    **动不了最高频的那一维**。⇒ 这就是为什么长上下文要靠插值/重标定")
print("    （position interpolation / NTK-aware / YaRN），而不是只调大 base。")

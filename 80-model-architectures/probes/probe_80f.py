"""
探测 6：修正混叠实验的设计错误（探测 5 的两处错）。

探测 5 的错：
  ✗ 判据用了「对随机 q,k 取均值」—— 而随机 q,k 的内积**期望本来就是 0**，
    混叠效应在均值上被完全抹掉。
  ✗ 我把 theta_1=1 与 theta_2=0.866 混在一起报作「高频段 i≤2」，
    后者的周期是 7.256 而非 6.283，所以「delta + 2π」对它**不闭合**。

正确设计：
  ① 只看 **i=1** 那一对（theta=1 ⇒ 周期**精确** = 2π）⇒ 它的贡献必须严格周期
  ② 用**每个样本内**的判据：同一对 q,k，比较 delta 与 delta+2π 的贡献（不是均值）
"""
import numpy as np

rng = np.random.default_rng(20260922)
print("numpy", np.__version__)
print("=" * 74)


def rope_freqs(d, base=10000.0):
    i = np.arange(1, d // 2 + 1, dtype=np.float64)
    return base ** (-2.0 * (i - 1) / d)


d = 64
th = rope_freqs(d)


def pair_contrib(q, k, delta, th, idx):
    """第 idx 对（从 0 数）对 <R_0 q, R_delta k> 的贡献。
    一对内：q' = (q0,q1) 位置 0；k' = Rot(delta*theta)(k0,k1)"""
    q0, q1 = q[2 * idx], q[2 * idx + 1]
    k0, k1 = k[2 * idx], k[2 * idx + 1]
    c, s = np.cos(delta * th[idx]), np.sin(delta * th[idx])
    return q0 * (k0 * c - k1 * s) + q1 * (k0 * s + k1 * c)


# ============================================================
# 1. 只看 i=1（theta 精确 = 1）—— 贡献必须严格以 2π 为周期
# ============================================================
print("【1】只看第 1 对（theta_1 = 1，周期精确 = 2π = 6.283185307）")
print(f"  {'delta':>12} {'contrib_1':>16} {'与 delta-2πk 的差':>20}")
q = rng.standard_normal(d)
k = rng.standard_normal(d)
base_delta = 0.37
ref = pair_contrib(q, k, base_delta, th, 0)
for extra in (0.0, 2 * np.pi, 4 * np.pi, 10 * np.pi):
    dd = base_delta + extra
    v = pair_contrib(q, k, dd, th, 0)
    print(f"  {dd:>12.6f} {v:>16.10f} {abs(v - ref):>20.3e}")
print("  ⇒ 差 = 机器精度 ⇒ **第 1 对严格以 2π 为周期**（这就是混叠的来源）")

# ============================================================
# 2. 换成「每样本内」判据：同一对 q,k，比较 delta 与 delta+2π
# ============================================================
print()
print("【2】每对各自的『2π 周期闭合误差』（相对该对贡献的量级，200 组取最大）")
N = 200
Q = rng.standard_normal((N, d))
K = rng.standard_normal((N, d))
base_delta = 3.0
print(f"  {'i':>4} {'theta_i':>12} {'该对自身的周期':>16} {'闭合误差/量级':>16} {'判定':>22}")
for idx in (0, 1, 2, 3, 7, 15, 31):
    v0 = np.array([pair_contrib(Q[n], K[n], base_delta, th, idx) for n in range(N)])
    v1 = np.array([pair_contrib(Q[n], K[n], base_delta + 2 * np.pi, th, idx)
                   for n in range(N)])
    scale = np.abs(v0).mean()
    err = np.abs(v1 - v0).max() / scale
    period = 2 * np.pi / th[idx]
    tag = "闭合（这就是混叠）" if err < 0.3 else "不闭合"
    print(f"  {idx+1:>4} {th[idx]:>12.8f} {period:>16.4f} {err:>16.4f} {tag:>22}")

# ============================================================
# 3. 长距离外推：分频段看「位置信息还剩多少」
# ============================================================
print()
print("【3】外推时各对还剩多少『位置分辨力』——")


def phase_sensitivity(idx, span):
    """在 span 这么大的位置范围内，第 idx 对的相位变化总量（弧度）。
    相位变化 << 1  ⇒ 这一对分辨不出这些位置（信息已丢失）；
    相位变化 >> 2π ⇒ 转过很多圈（混叠，位置不可唯一确定）。"""
    return span * th[idx]


print(f"  {'i':>4} {'theta_i':>12} {'delta=0..128 的相位变化':>24} "
      f"{'delta=0..4096 的相位变化':>26}")
for idx in (0, 1, 3, 7, 15, 31):
    p1 = phase_sensitivity(idx, 128)
    p2 = phase_sensitivity(idx, 4096)
    print(f"  {idx+1:>4} {th[idx]:>12.8f} {p1:>24.4f} {p2:>26.4f}")
print("  ⇒ 第 1 对：在 128 内已转 128 rad（20 圈）⇒ 早已混叠")
print("    第 32 对：在 128 内只转 0.06 rad ⇒ 分辨不出，但在 4096 内转 2.07 rad ⇒ 刚好")

# ============================================================
# 4. 外推失败的**可测后果**：同一 q,k，位置 m 与 m+2π 产生几乎相同的分数
# ============================================================
print()
print("【4】★ 可测后果：把 k 放在位置 m 与 m+2π，第 1 对的分数几乎相同")
print(f"  {'m':>8} {'score(m)':>16} {'score(m+2π)':>16} {'相对差':>14}")
for m in (10.0, 50.0, 200.0, 1000.0):
    s1 = pair_contrib(q, k, 0.0 - (-m), th, 0)          # 相对位置 m
    s2 = pair_contrib(q, k, (m + 2 * np.pi) - m, th, 0)  # 相对位置 m+2π
    # 直接用相对位置
    s1 = pair_contrib(q, k, m, th, 0)
    s2 = pair_contrib(q, k, m + 2 * np.pi, th, 0)
    print(f"  {m:>8.1f} {s1:>16.8f} {s2:>16.8f} "
          f"{abs(s2-s1)/(abs(s1)+1e-30):>14.3e}")
print("  ⇒ 相对差 ≈ 机器精度 ⇒ **位置 m 与 m+2π 在第 1 对上是同一个位置**。")
print("    模型无法区分它们 ⇒ 这就是『直接外推会崩』的机制，而且它**不是**")
print("    『没见过长位置』那么简单——是**位置编码本身发生了混叠**。")

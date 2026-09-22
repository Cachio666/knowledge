"""
探测 2：修正判据后的复验。

探测 1 暴露的两个控制者自己的错误：
  ✗ Q1 用了「相对极差 (max-min)/|mean|」——随机向量内积的 mean 本身接近 0 ⇒ 比值爆炸。
    ⇒ 改用**尺度无关的绝对判据**：max|v_m − v_0| / (‖q‖·‖k‖)。
  ✗ Q4 我预期「LayerNorm 不缩放等变」，实测 2.65e-07 ⇒ 它其实等变（eps 破坏它）。
    ⇒ 重新按「无 eps / 有 eps」两档分别测。
"""
import numpy as np

rng = np.random.default_rng(20260922)
print("numpy", np.__version__)
print("=" * 72)

# ============================================================
# RoPE
# ============================================================
def rope_freqs(d, base=10000.0):
    """原论文式 theta_i = base^(-2(i-1)/d), i = 1..d/2"""
    i = np.arange(1, d // 2 + 1, dtype=np.float64)
    return base ** (-2.0 * (i - 1) / d)


def rope_adj(x, pos, theta):
    """相邻对式：pair = (x[2i], x[2i+1])"""
    d = x.shape[-1]
    xr = x.reshape(d // 2, 2)
    c, s = np.cos(pos * theta), np.sin(pos * theta)
    x0, x1 = xr[:, 0], xr[:, 1]
    return np.stack([x0 * c - x1 * s, x0 * s + x1 * c], axis=-1).reshape(d)


def rope_half(x, pos, theta):
    """rotate_half 式（Llama/HF）：pair = (x[i], x[i+d/2])"""
    d = x.shape[-1]
    h = d // 2
    a, b = x[:h], x[h:]
    c, s = np.cos(pos * theta), np.sin(pos * theta)
    return np.concatenate([a * c - b * s, a * s + b * c])


d, S = 64, 1024
theta = rope_freqs(d)
q = rng.standard_normal(d)
k = rng.standard_normal(d)
SCALE = np.linalg.norm(q) * np.linalg.norm(k)   # 尺度无关判据的分母

print("【判据修正】RoPE 相对位置性质  ⟨R_m q, R_n k⟩ 只依赖 n−m")
print("  判据：max_m |v_m − v_0| / (‖q‖·‖k‖)   应 ≈ 机器精度")
for name, fn in [("相邻对", rope_adj), ("rotate_half", rope_half)]:
    for delta in (1, 7, 63, 255, 511):
        qr = {m: fn(q, m, theta) for m in range(delta, S)}
        vals = np.array([float(qr[m] @ fn(k, m - delta, theta)) for m in range(delta, S)])
        dev = np.abs(vals - vals[0]).max() / SCALE
        print(f"  {name:12s} delta={delta:4d}:  max 偏差/尺度 = {dev:.3e}   "
              f"内积量级 = {np.abs(vals).mean()/SCALE:.4f}")

print()
print("【复验 Q3】两种实现的数值是否相同？（不同 ⇒ 是一个真实的口径陷阱）")
for delta in (1, 63, 255):
    va = np.array([float(rope_adj(q, m, theta) @ rope_adj(k, m - delta, theta))
                   for m in range(delta, S)])
    vh = np.array([float(rope_half(q, m, theta) @ rope_half(k, m - delta, theta))
                   for m in range(delta, S)])
    print(f"  delta={delta:4d}:  相邻对均值={va.mean():+.6f}  rotate_half={vh.mean():+.6f}")

print()
print("【补充】两种实现各自是否『都是合法的旋转』？——范数保持")
for name, fn in [("相邻对", rope_adj), ("rotate_half", rope_half)]:
    err = max(abs(np.linalg.norm(fn(q, m, theta)) / np.linalg.norm(q) - 1.0)
              for m in range(S))
    print(f"  {name:12s}: max |‖R_m q‖/‖q‖ − 1| = {err:.3e}")

# ============================================================
# LayerNorm / RMSNorm 不变性（按 eps 分档）
# ============================================================
print()
print("=" * 72)


def rmsnorm(x, eps):
    return x / np.sqrt((x ** 2).mean(axis=-1, keepdims=True) + eps)


def layernorm(x, eps):
    mu = x.mean(axis=-1, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)


X = rng.standard_normal((64, 512)) * 3.0 + 5.0

print("【复验 Q4】不变性 —— 按 eps 分档（eps 是不变性的破坏者，这是关键）")
print("  RMSNorm 缩放等变：f(αx) == f(x) ?")
for eps in (0.0, 1e-6, 1e-3):
    for alpha in (2.0, 50.0, 1000.0):
        r = np.abs(rmsnorm(alpha * X, eps) - rmsnorm(X, eps)).max()
        print(f"    eps={eps:8.1e} α={alpha:8.1f}:  max|f(αx)−f(x)| = {r:.3e}")
print("  LayerNorm 缩放等变：f(αx) == f(x) ?   ← ★ 探测 1 我预期它【不】等变，实测说它等变")
for eps in (0.0, 1e-6, 1e-3):
    for alpha in (2.0, 50.0, 1000.0):
        r = np.abs(layernorm(alpha * X, eps) - layernorm(X, eps)).max()
        print(f"    eps={eps:8.1e} α={alpha:8.1f}:  max|f(αx)−f(x)| = {r:.3e}")

print("  反例对照（两个都应该是『显著非零』）：")
for eps in (1e-6,):
    print(f"    RMSNorm 平移 beta=100 : max|f(x+b)−f(x)| = "
          f"{np.abs(rmsnorm(X+100.0, eps)-rmsnorm(X, eps)).max():.3e}")
    print(f"    LayerNorm 平移 beta=100: max|f(x+b)−f(x)| = "
          f"{np.abs(layernorm(X+100.0, eps)-layernorm(X, eps)).max():.3e}")

# ============================================================
# pre-LN vs post-LN —— 复跑，看是否稳定
# ============================================================
print()
print("=" * 72)
print("【复验 Q6】pre-LN vs post-LN 残差流范数 —— 5 个随机种子，看结论稳不稳")


def run(seed, D=256, L=24, N=64):
    r = np.random.default_rng(seed)
    x0 = r.standard_normal((N, D)) * 0.5
    W = [r.standard_normal((D, D)) / np.sqrt(D) for _ in range(L)]
    b = [r.standard_normal(D) * 0.01 for _ in range(L)]

    def F(h, l):
        return np.tanh(h @ W[l]) @ W[l].T + b[l]

    hp = x0.copy()
    pre = []
    for l in range(L):
        hp = hp + F(layernorm(hp, 1e-6), l)
        pre.append(np.linalg.norm(hp, axis=-1).mean())

    hq = x0.copy()
    post = []
    for l in range(L):
        hq = layernorm(hq + F(hq, l), 1e-6)
        post.append(np.linalg.norm(hq, axis=-1).mean())
    return np.array(pre), np.array(post)


print(f"  {'seed':>6} {'pre: 首层':>10} {'末层':>10} {'增长':>8}   "
      f"{'post: 首层':>10} {'末层':>10} {'增长':>8}")
pres, posts = [], []
for s in range(5):
    pre, post = run(1000 + s)
    pres.append(pre); posts.append(post)
    print(f"  {1000+s:>6} {pre[0]:>10.3f} {pre[-1]:>10.3f} {pre[-1]/pre[0]:>7.2f}×   "
          f"{post[0]:>10.3f} {post[-1]:>10.3f} {post[-1]/post[0]:>7.2f}×")
pres = np.array(pres); posts = np.array(posts)
print(f"  {'均值':>6} {pres[:,0].mean():>10.3f} {pres[:,-1].mean():>10.3f} "
      f"{(pres[:,-1]/pres[:,0]).mean():>7.2f}×   "
      f"{posts[:,0].mean():>10.3f} {posts[:,-1].mean():>10.3f} "
      f"{(posts[:,-1]/posts[:,0]).mean():>7.2f}×")
print(f"  ⇒ pre-LN 增长 的跨种子极差 = "
      f"{(pres[:,-1]/pres[:,0]).max() - (pres[:,-1]/pres[:,0]).min():.2f}")
print(f"  ⇒ post-LN 增长 的跨种子极差 = "
      f"{(posts[:,-1]/posts[:,0]).max() - (posts[:,-1]/posts[:,0]).min():.2e}")
print(f"  ⇒ post-LN 末层 / √D (= {np.sqrt(256):.1f}) = "
      f"{posts[:,-1].mean()/np.sqrt(256):.6f}")

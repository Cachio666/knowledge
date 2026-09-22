"""
探测 3：attention logits 的尺度 —— `1/√d_h` 缩放的由来、QK-norm 的作用、深度的放大。

这是「现代架构为什么加 QK-norm」的定量根据，也是本域 `transformer`/`normalization`
两个主题交汇处最硬的内容。
"""
import numpy as np

rng = np.random.default_rng(20260922)
print("numpy", np.__version__)
print("=" * 74)

# ============================================================
# 1. `q·k` 的方差随 d_h 线性增长 ⇒ 除以 √d_h 使其与 d_h 无关
# ============================================================
print("【1】Var(q·k) 随 d_h 线性增长；除法之后与 d_h 无关")
print(f"  {'d_h':>5} {'Var(q·k)':>12} {'Var/ d_h':>10} {'Var((q·k)/√d_h)':>18}")
N = 40000
for dh in (32, 64, 128, 256, 512):
    q = rng.standard_normal((N, dh))
    k = rng.standard_normal((N, dh))
    dot = (q * k).sum(-1)
    print(f"  {dh:>5} {dot.var():>12.2f} {dot.var()/dh:>10.4f} "
          f"{(dot/np.sqrt(dh)).var():>18.4f}")

# ============================================================
# 2. QK-norm：把 Q、K 各自归一化后再点积 ⇒ logits 方差与 d_h 无关
# ============================================================
print()
print("【2】QK-norm（对 q、k 各做 RMSNorm）之后，logits 方差与 d_h 的关系")


def rmsnorm(x, eps=1e-6):
    return x / np.sqrt((x ** 2).mean(axis=-1, keepdims=True) + eps)


print(f"  {'d_h':>5} {'Var(q·k/√d_h)':>16} {'Var(RMS(q)·RMS(k))':>20} "
      f"{'后者/前者':>12}")
for dh in (32, 64, 128, 256, 512):
    q = rng.standard_normal((N, dh))
    k = rng.standard_normal((N, dh))
    plain = ((q * k).sum(-1) / np.sqrt(dh)).var()
    qn = rng.standard_normal((N, dh)) * 3.0 + 0.5   # 故意给它一个非单位尺度
    kn = rng.standard_normal((N, dh)) * 0.2 - 1.0
    qkn = (rmsnorm(qn) * rmsnorm(kn)).sum(-1).var()
    print(f"  {dh:>5} {plain:>16.4f} {qkn:>20.4f} {qkn/plain:>12.4f}")

print("  ⇒ QK-norm 把 logits 方差从『与 d_h 无关但由输入尺度决定』变成『干净地 ~1』")
print("    而且它**同时抹掉了输入尺度**（上表里 q/k 被乘了 3.0 和 0.2）")

# ============================================================
# 3. softmax 的熵 vs logits 尺度 —— 「注意力稀释 / 尖锐化」
# ============================================================
print()
print("【3】softmax 熵 vs logits 尺度（S=1024）——尺度错一个量级，注意力就废了")
S = 1024


def softmax_entropy(scale):
    lg = rng.standard_normal(S) * scale
    lg -= lg.max()
    p = np.exp(lg); p /= p.sum()
    return float(-(p * np.log(p + 1e-300)).sum()), float(p.max())


print(f"  {'logits std':>12} {'熵(归一化)':>14} {'最大概率':>12} {'判定':>22}")
for sc in (0.1, 0.5, 1.0, 3.0, 6.0, 12.0):
    ent, mx = softmax_entropy(sc)
    tag = "接近均匀" if ent / np.log(S) > 0.9 else \
          ("正常" if ent / np.log(S) > 0.3 else "几乎 one-hot")
    print(f"  {sc:>12.1f} {ent/np.log(S):>14.4f} {mx:>12.4f} {tag:>22}")

# ============================================================
# 4. 深度对 logits 尺度的放大（接 pre-LN 残差流范数增长）
# ============================================================
print()
print("【4】pre-LN 下，残差流范数随深度增长 ⇒ attention logits 也随之放大")
D, L, Nn = 256, 24, 64


def layernorm(x, eps=1e-6):
    mu = x.mean(axis=-1, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)


def run_pre(seed):
    r = np.random.default_rng(seed)
    h = r.standard_normal((Nn, D)) * 0.5
    W = [r.standard_normal((D, D)) / np.sqrt(D) for _ in range(L)]
    Wq = r.standard_normal((D, D)) / np.sqrt(D)
    b = [r.standard_normal(D) * 0.01 for _ in range(L)]
    logit_std = []
    for l in range(L):
        hn = layernorm(h)
        # 模拟这一层的 attention 用 Q/K 投影后的 logits 尺度
        qn = hn @ Wq
        logit_std.append(float(qn.std()))
        h = h + np.tanh(hn @ W[l]) @ W[l].T + b[l]
    return np.array(logit_std)


print(f"  {'seed':>6}  " + "  ".join(f"L{l:02d}" for l in range(0, L, 4)))
arrs = []
for s in range(3):
    a = run_pre(2000 + s)
    arrs.append(a)
    print(f"  {2000+s:>6}  " + "  ".join(f"{a[l]:.2f}" for l in range(0, L, 4)))
arrs = np.array(arrs)
print(f"  均值    " + "  ".join(f"{arrs[:,l].mean():.2f}" for l in range(0, L, 4)))
print(f"  ⇒ 首层 {arrs[:,0].mean():.3f} → 末层 {arrs[:,-1].mean():.3f}，"
      f"放大 {arrs[:,-1].mean()/arrs[:,0].mean():.2f}×")
print("  ⇒ 注意：LayerNorm 把**每层的输入**归一化了（这是 pre-LN 的作用），")
print("    但残差流本身在长 ⇒ 未归一化的量（如直接投影出的 logits）仍会放大。")

# ============================================================
# 5. 「直接对 q/k 做投影」时，深度的净效应
# ============================================================
print()
print("【5】对照：post-LN 下同样的量（因为每层都被归一化，不放大）")


def run_post(seed):
    r = np.random.default_rng(seed)
    h = r.standard_normal((Nn, D)) * 0.5
    W = [r.standard_normal((D, D)) / np.sqrt(D) for _ in range(L)]
    Wq = r.standard_normal((D, D)) / np.sqrt(D)
    b = [r.standard_normal(D) * 0.01 for _ in range(L)]
    logit_std = []
    for l in range(L):
        qn = h @ Wq
        logit_std.append(float(qn.std()))
        h = layernorm(h + np.tanh(h @ W[l]) @ W[l].T + b[l])
    return np.array(logit_std)


arrs2 = np.array([run_post(2000 + s) for s in range(3)])
print(f"  均值    " + "  ".join(f"{arrs2[:,l].mean():.2f}" for l in range(0, L, 4)))
print(f"  ⇒ 首层 {arrs2[:,0].mean():.3f} → 末层 {arrs2[:,-1].mean():.3f}，"
      f"放大 {arrs2[:,-1].mean()/arrs2[:,0].mean():.2f}×")

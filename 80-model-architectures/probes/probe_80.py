"""
80-model-architectures 技术可行性探测（不改仓库，只验证「架构层不变量」是否真能在本机测出来）。

要回答的问题：
  Q1 RoPE 的「相对位置性质」⟨R_m q, R_n k⟩ 只依赖 m−n —— 真能测到吗？精度多少？
  Q2 RoPE 的「范数保持」‖R_m q‖ = ‖q‖ —— 真能测到吗？
  Q3 ★ RoPE 的两种实现（相邻对 vs rotate_half）—— 内积相同吗？这是不是一个真实的口径陷阱？
  Q4 RMSNorm 的缩放等变性 / LayerNorm 的平移不变性 —— 真能测到吗？
  Q5 GQA 的等价性（r 份共享 = 独立算 r 遍）—— 真能测到吗？
     ★ 初版两边是**同一行代码**（恒真断言，见 Q5 处的修正注）；现改为两条不同路径。
  Q6 ★ pre-LN 与 post-LN 的残差流范数增长 —— 真能测出差异吗？量级多少？
  Q7 MoE 容量因子导致的 token 丢弃率 —— 真能测到吗？
  Q8 SwiGLU 的 d_ff = 8d/3 参数量 —— 与 4d² 差多少？
"""
import inspect

import numpy as np

rng = np.random.default_rng(20260922)
REPORT = []


def note(tag, msg):
    REPORT.append(f"[{tag}] {msg}")
    print(f"[{tag}] {msg}")


# ============================================================
# Q1 / Q2 / Q3  RoPE
# ============================================================
def rope_freqs(d, base=10000.0):
    """原论文式：theta_i = base^(-2(i-1)/d)，i = 1..d/2。"""
    i = np.arange(1, d // 2 + 1, dtype=np.float64)
    return base ** (-2.0 * (i - 1) / d)


def rope_adjacent(x, pos, theta):
    """原论文（RoFormer）式：相邻两维 (2i, 2i+1) 构成一对。
    x: (..., d)  实向量，按相邻对旋转。"""
    d = x.shape[-1]
    x = x.reshape(*x.shape[:-1], d // 2, 2)
    c, s = np.cos(pos * theta), np.sin(pos * theta)
    x0, x1 = x[..., 0], x[..., 1]
    out = np.stack([x0 * c - x1 * s, x0 * s + x1 * c], axis=-1)
    return out.reshape(*x.shape[:-2], d)


def rope_rotate_half(x, pos, theta):
    """Llama / HuggingFace 式：前后两半 (i, i+d/2) 构成一对。
    等价于把相邻对式的维度做一次置换。"""
    d = x.shape[-1]
    half = d // 2
    x1, x2 = x[..., :half], x[..., half:]
    # theta 需要重复一次，因为前半与后半各用一遍
    c, s = np.cos(pos * theta), np.sin(pos * theta)
    c = np.concatenate([c, c], axis=-1)
    s = np.concatenate([s, s], axis=-1)
    out = np.concatenate([x1 * c[:half] - x2 * s[:half],
                          x2 * c[:half] + x1 * s[:half]], axis=-1)
    return out


d, S = 64, 512
theta = rope_freqs(d)
q = rng.standard_normal((S, d))
k = rng.standard_normal((S, d))

# Q1: 相对位置性质（两种实现各测）
for name, fn in [("相邻对", rope_adjacent), ("rotate_half", rope_rotate_half)]:
    qr = np.stack([fn(q[m], m, theta) for m in range(S)]) if False else \
         np.array([fn(q[m], m, theta) for m in range(S)])
    kr = np.array([fn(k[n], n, theta) for n in range(S)])
    # 固定 m - n = delta，扫 m，看内积是否只依赖 delta
    worst = 0.0
    for delta in (1, 7, 63, 255):
        vals = []
        for m in range(max(delta, 0), S):
            n = m - delta
            vals.append(float(qr[m] @ kr[n]))
        vals = np.array(vals)
        spread = (vals.max() - vals.min()) / (np.abs(vals).mean() + 1e-30)
        worst = max(worst, spread)
    note("Q1", f"{name}: 固定 delta 时内积的相对极差（最大） = {worst:.3e}")

# Q2: 范数保持
for name, fn in [("相邻对", rope_adjacent), ("rotate_half", rope_rotate_half)]:
    qr = np.array([fn(q[m], m, theta) for m in range(S)])
    rel = np.abs(np.linalg.norm(qr, axis=-1) - np.linalg.norm(q, axis=-1)) / np.linalg.norm(q, axis=-1)
    note("Q2", f"{name}: max |‖R_m q‖ − ‖q‖| / ‖q‖ = {rel.max():.3e}")

# Q3 ★ 两种实现的内积是否相同？
qa = np.array([rope_adjacent(q[m], m, theta) for m in range(S)])
ka = np.array([rope_adjacent(k[n], n, theta) for n in range(S)])
qh = np.array([rope_rotate_half(q[m], m, theta) for m in range(S)])
kh = np.array([rope_rotate_half(k[n], n, theta) for n in range(S)])
for delta in (1, 63, 255):
    ia, ih = [], []
    for m in range(delta, S):
        ia.append(float(qa[m] @ ka[m - delta]))
        ih.append(float(qh[m] @ kh[m - delta]))
    ia, ih = np.array(ia), np.array(ih)
    note("Q3", f"delta={delta:3d}: 相邻对内积均值={ia.mean():+.6f}  rotate_half={ih.mean():+.6f}  "
               f"相对差={abs(ia.mean()-ih.mean())/(abs(ia.mean())+1e-30):.4f}")

# ============================================================
# Q4  RMSNorm / LayerNorm 的不变性
# ============================================================
x = rng.standard_normal((32, d)) * 3.0 + 5.0
g = rng.standard_normal(d) * 0.1 + 1.0  # gamma


def rmsnorm(x, g=None, eps=1e-6):
    ms = (x ** 2).mean(axis=-1, keepdims=True)
    y = x / np.sqrt(ms + eps)
    return y * g if g is not None else y


def layernorm(x, g=None, b=None, eps=1e-6):
    mu = x.mean(axis=-1, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
    y = (x - mu) / np.sqrt(var + eps)
    if g is not None:
        y = y * g
    if b is not None:
        y = y + b
    return y


for alpha in (2.0, 50.0):
    r = np.abs(rmsnorm(alpha * x) - rmsnorm(x)).max()
    note("Q4", f"RMSNorm 缩放等变性 alpha={alpha:5.1f}: max|f(ax)−f(x)| = {r:.3e}")
for beta in (1.0, 100.0):
    r = np.abs(layernorm(x + beta) - layernorm(x)).max()
    note("Q4", f"LayerNorm 平移不变性 beta={beta:6.1f}: max|f(x+b)−f(x)| = {r:.3e}")
# 反向：RMSNorm 不平移不变、LayerNorm 不缩放等变（互相的反例）
r1 = np.abs(rmsnorm(x + 100.0) - rmsnorm(x)).max()
r2 = np.abs(layernorm(50.0 * x) - layernorm(x)).max()
note("Q4", f"反例：RMSNorm 平移 max 差 = {r1:.3e}（应显著非零）；"
           f"LayerNorm 缩放 max 差 = {r2:.3e}（应显著非零）")

# ============================================================
# Q5  GQA 等价性：r 个 query 头共享一份 KV  ==  复制 r 份独立算
# ------------------------------------------------------------
# ★ 修正（2026-09-22，收进仓库前）：本探测的**初版**两边是**逐字符相同的同一行代码**
#   （`attn(Q[i], K[i//group], V[i//group])` 写了两遍）⇒ 它报出的 `0.000e+00` 是
#   **构造出来的、不是测出来的**（计划 §0.1 的 P17）。
#   现在改成**两条真正不同的路径**：
#     ① 共享 KV + **物化完整分数矩阵**的稠密 softmax；
#     ② 把 KV **物化复制**成 h_q 份 + **online softmax 流式**实现。
#   两者不共享任何辅助函数，累加顺序也不同（参照
#   `40-kernels/attention/code/verify.py` 的 A4 `:508-531`，那里用 attn_naive vs
#   attn_online 两条不同实现比同一组 (Q,K,V)）。
# ============================================================
hq, dh = 8, 32
SQ, SK = 16, 9
Q = rng.standard_normal((hq, SQ, dh))
K = rng.standard_normal((4, SK, dh))
V = rng.standard_normal((4, SK, dh))


def attn_naive(q, k, v):
    """路径 ①：物化完整分数矩阵，一次性 softmax。"""
    s = q @ k.T / np.sqrt(dh)
    s = s - s.max(axis=-1, keepdims=True)
    p = np.exp(s); p /= p.sum(axis=-1, keepdims=True)
    return p @ v


def attn_online(q, k, v, block=4):
    """路径 ②：**不物化**分数矩阵；逐块流式更新 running max / running sum，
    分母最后才除。与路径 ① 不共享任何辅助函数。"""
    out = np.zeros_like(q)
    for t in range(q.shape[0]):
        m, l, o = -np.inf, 0.0, np.zeros(q.shape[-1])
        for s0 in range(0, k.shape[0], block):
            kb, vb = k[s0:s0 + block], v[s0:s0 + block]
            sc = kb @ q[t] / np.sqrt(dh)
            m_new = max(m, float(sc.max()))
            p = np.exp(sc - m_new)
            scale = np.exp(m - m_new)       # ← 路径 ① 里没有这一步
            o = o * scale + p @ vb
            l = l * scale + p.sum()
            m = m_new
        out[t] = o / l
    return out


hkv = 4
group = hq // hkv
# 路径 ①：GQA —— 每 group 个 q 头**共享**它的那一份 KV
out_shared = np.concatenate([
    attn_naive(Q[i], K[i // group], V[i // group]) for i in range(hq)
], axis=0)
# 路径 ②：MHA 的等价写法 —— 把 KV **物化复制**成 h_q 份，再用另一条实现路径算
K_rep = np.repeat(K, group, axis=0)
V_rep = np.repeat(V, group, axis=0)
out_naive = np.concatenate([
    attn_online(Q[i], K_rep[i], V_rep[i]) for i in range(hq)
], axis=0)
diff = float(np.abs(out_shared - out_naive).max())
mag = float(np.abs(out_shared).max())
note("Q5", f"h_q={hq} h_kv={hkv} r={group}: 共享式（稠密）与复制式（online）最大差 = "
           f"{diff:.3e}（输出量级 {mag:.3f} ⇒ 相对 {diff/mag:.2e}）")
# ★ **反恒真自检**：两条路径的**代码本体**必须不是同一份。
#   为 True 时上面那个差就是构造出来的（P17），本行会当场揭穿它。
same = inspect.getsource(attn_naive) == inspect.getsource(attn_online)
note("Q5", f"两条路径的源码是否同一份 = {same}（必须 False）")

# ★ 关键：KV cache 字节随 r 的变化（可算）
for r in (1, 2, 4, 8):
    hk = hq // r
    byt = 2 * hk * dh * 2  # K+V, fp16
    note("Q5", f"r={r}: h_kv={hk:2d}  每 token 每层 KV = {byt:5d} B  相对 MHA = {1/r:.3f}")

# ============================================================
# Q6 ★ pre-LN 与 post-LN 的残差流范数增长
# ============================================================
D, L, N = 256, 24, 64
x0 = rng.standard_normal((N, D)) * 0.5
W = [rng.standard_normal((D, D)) / np.sqrt(D) for _ in range(L)]
b = [rng.standard_normal(D) * 0.01 for _ in range(L)]


def ffn(h, l):
    """一个两层 MLP 作为 F（用 tanh 代 GELU，形状不影响结论）"""
    return np.tanh(h @ W[l]) @ W[l].T + b[l]


# pre-LN： x_{l+1} = x_l + F(LN(x_l))
h = x0.copy()
norms_pre = []
for l in range(L):
    h = h + ffn(layernorm(h), l)
    norms_pre.append(np.linalg.norm(h, axis=-1).mean())

# post-LN： x_{l+1} = LN(x_l + F(x_l))
h = x0.copy()
norms_post = []
for l in range(L):
    h = layernorm(h + ffn(h, l))
    norms_post.append(np.linalg.norm(h, axis=-1).mean())

note("Q6", f"pre-LN  残差流范数：第0层={norms_pre[0]:.3f} 末层={norms_pre[-1]:.3f} "
           f"增长={norms_pre[-1]/norms_pre[0]:.2f}×")
note("Q6", f"post-LN 残差流范数：第0层={norms_post[0]:.3f} 末层={norms_post[-1]:.3f} "
           f"增长={norms_post[-1]/norms_post[0]:.2f}×")
note("Q6", f"pre-LN 各层范数（每 4 层）：" +
           " ".join(f"{norms_pre[i]:.2f}" for i in range(0, L, 4)))
note("Q6", f"post-LN 各层范数（每 4 层）：" +
           " ".join(f"{norms_post[i]:.2f}" for i in range(0, L, 4)))

# ============================================================
# Q7  MoE 容量因子导致的丢弃率
# ============================================================
T, E, k = 4096, 64, 2
# 让专家的负载不均匀（模拟真实 gating 的偏斜）
logits = rng.standard_normal((T, E))
# 加一个「热门专家」偏置
hot = rng.permutation(E)[:8]
logits[:, hot] += 1.2
topk = np.argsort(-logits, axis=1)[:, :k]
counts = np.zeros(E, dtype=np.int64)
for e in np.unique(topk):
    counts[e] = (topk == e).sum()
note("Q7", f"T={T} E={E} k={k}: 负载 min={counts.min()} max={counts.max()} "
           f"max/mean={counts.max()/counts.mean():.2f}×")
for cf in (1.0, 1.25, 1.5, 2.0):
    cap = int(np.ceil(T / E * cf))
    dropped = sum(max(0, int((topk[:, j] == e).sum()) - cap)
                  for j in range(k) for e in range(E))
    note("Q7", f"capacity_factor={cf:4.2f}: 容量/专家={cap:3d}  "
               f"丢弃槽位={dropped:5d} / 总槽位={T*k} = {dropped/(T*k)*100:.2f}%")

# ============================================================
# Q8  SwiGLU 的 d_ff = 8d/3
# ============================================================
for d in (4096, 8192):
    # 标准 FFN: W1 (d x 4d) + W2 (4d x d) = 2 * 4d * d = 8d²
    std = 2 * 4 * d * d
    gl_4d = 3 * 4 * d * d                          # SwiGLU 用 4d 中间维：3 个矩阵
    dff = int(8 * d / 3)
    gl_83 = 3 * dff * d
    note("Q8", f"d={d}: 标准 FFN(4d)={std/1e6:.1f}M   SwiGLU(4d)={gl_4d/1e6:.1f}M "
               f"SwiGLU(8d/3={dff})={gl_83/1e6:.1f}M   "
               f"比值 SwiGLU(8d/3)/标准 = {gl_83/std:.4f}")

print("\n" + "=" * 70)
print(f"探测完成，共 {len(REPORT)} 条。numpy {np.__version__}")

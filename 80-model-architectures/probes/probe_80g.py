"""
探测 7：三项不依赖外部来源的自构造验证。

  A. ★ RoPE 与低秩压缩的**不相容性** —— MLA 为什么必须把 RoPE 部分单独拎出来
  B. MoE 的 auxiliary load-balancing loss 对负载分布的作用
  C. 参数量分配：embedding / attention / FFN 的占比随 L、d 怎么变
"""
import numpy as np

rng = np.random.default_rng(20260922)
print("numpy", np.__version__)
print("=" * 74)

# ============================================================
# A. ★ RoPE 能不能「吸收」进权重矩阵？
# ============================================================
print("【A】★ RoPE 与低秩压缩的不相容性")
print()
print("  MLA 的做法：c = x·W_DKV（压缩到 d_c 维）；k = c·W_UK；v = c·W_UV")
print("  '矩阵吸收'：推理时把 W_UK 吸收进 W_q ⇒ q_eff = x·(W_q·W_UK^T)，")
print("             于是不需要显式算出 k，只需存 c。")
print()

d, dc, S = 32, 8, 6
X = rng.standard_normal((S, d))
Wq = rng.standard_normal((d, d)) / np.sqrt(d)
W_dkv = rng.standard_normal((d, dc)) / np.sqrt(d)
W_uk = rng.standard_normal((dc, d)) / np.sqrt(dc)

# 路径 1：显式算 k，再算 scores
C = X @ W_dkv                    # (S, dc)  ← 这就是 KV cache 里存的东西
K_explicit = C @ W_uk            # (S, d)
Q = X @ Wq
scores_explicit = Q @ K_explicit.T

# 路径 2：吸收（先算 W_q · W_uk^T，再作用到 X 上；且 K^T = W_uk^T C^T）
W_eff = Wq @ W_uk.T              # (d, dc)
scores_absorbed = (X @ W_eff) @ C.T

print(f"  吸收 vs 显式：max 差 = {np.abs(scores_explicit - scores_absorbed).max():.3e}")
print("  ⇒ 可以吸收（线性可交换）")
print()

# 现在加 RoPE
# ★ 修正（2026-09-22，收进仓库前）：本探测的**初版**「显式」路径**只旋转了 k、没旋转 q**
#   （写成 `scores_rope = Q @ K_rope.T`，用的是**未旋转的 Q**）⇒ 那个对象**不是任何
#   合法的注意力分数**，拿它去和「错误吸收」作差得到的是「**两个都不合法的对象之差**」
#   （计划 §0.7-D 的 D15 注；原稿报的 `18.90` 由此而来）。
#   **正确形式是 q 与 k 都被旋转**：scores[t,s] = (R_{pos_t} q_t)·(R_{pos_s} k_s)。
def rope_apply(x, pos, base=10000.0):
    dd = x.shape[-1]
    i = np.arange(1, dd // 2 + 1, dtype=np.float64)
    th = base ** (-2.0 * (i - 1) / dd)
    xr = x.reshape(*x.shape[:-1], dd // 2, 2)
    c, s = np.cos(pos * th), np.sin(pos * th)
    a, b = xr[..., 0], xr[..., 1]
    return np.stack([a * c - b * s, a * s + b * c], axis=-1).reshape(x.shape)


pos = np.arange(S, dtype=np.float64)
K_rope = np.stack([rope_apply(K_explicit[t], pos[t]) for t in range(S)])
Q_rope = np.stack([rope_apply(Q[t], pos[t]) for t in range(S)])
scores_rope_explicit = Q_rope @ K_rope.T     # ← 显式路径：(R_m q)·(R_n k)，两边都旋转

# 试图「吸收」：把 RoPE 也算进去
# K_rope^T 的第 t 列 = R_t · (W_uk^T C_t^T) —— R_t 在 C 的**外面**，且逐位置不同
# 于是 scores[t, s] = q_t^⊤ R_{pos_t} · R_{pos_s} W_uk^T c_s
# 无法写成 (x_t · 某个固定矩阵) · c_s，因为两个 R 夹在中间且互不相同
print("  加 RoPE 之后：scores[t,s] = (R_{pos_t} q_t) · (R_{pos_s} W_uk^T c_s)")
print("  ⇒ R 依赖位置，且夹在 q 与 C 之间 ⇒ **不能吸收到 W_q 里**")
print()
print("  数值验证：下面两条路径**不相等**，差异量级 = 一个旋转的量级")

# 路径 3（错误的「吸收」）：先吸收再套 RoPE（只把 R 套在 q 侧一次）
scores_wrong = np.stack([
    rope_apply(Q[t] @ W_eff, pos[t]) @ C.T for t in range(S)
])

print(f"  · 显式（不吸收，q、k 都旋转） vs 错误吸收（先吸收再旋转）："
      f"max 差 = {np.abs(scores_rope_explicit - scores_wrong).max():.3e}")
print(f"  · 参照量级：显式分数的 max|score| = {np.abs(scores_rope_explicit).max():.3f}")
print("  ⇒ ★ 这就是 MLA「把 RoPE 那部分单独留一份未压缩的 k」的**代数理由**：")
print("    可压缩的部分（与位置无关）能吸收；不可压缩的部分（RoPE）必须显式保留。")

# ============================================================
# B. MoE 的 auxiliary load-balancing loss
# ============================================================
print()
print("=" * 74)
print("【B】MoE 的 aux load-balancing loss 对专家负载的作用")
T, E, k = 4096, 64, 2


def load_stats(logits, E):
    topk = np.argsort(-logits, axis=1)[:, :k]
    cnt = np.bincount(topk.ravel(), minlength=E).astype(float)
    f = cnt / cnt.sum()                       # 每个专家的 token 数占比
    p = np.exp(logits - logits.max(-1, keepdims=True))
    p = p / p.sum(-1, keepdims=True)
    P = p.mean(0)
    aux = E * (f * P).sum()                   # Switch Transformer 的 aux loss
    return cnt, f, P, aux


print(f"  {'情形':>22} {'max/mean':>10} {'aux_loss':>10} {'最少专家':>10}")
for name, s in [("均匀 logits", 0.0), ("轻微偏斜", 0.3), ("中等偏斜", 0.8),
                ("严重偏斜", 1.5), ("极端偏斜", 3.0)]:
    lg = rng.standard_normal((T, E)) * 1.0
    lg[:, :8] += s                            # 前 8 个专家被偏好
    cnt, f, P, aux = load_stats(lg, E)
    print(f"  {name:>22} {cnt.max()/cnt.mean():>10.2f} {aux:>10.4f} "
          f"{cnt.min():>10.0f}")
print("  ⇒ aux_loss 的**下界是 1.0**（完全均匀时 E·Σ(1/E·1/E) = 1）")
lg = rng.standard_normal((T, E))
_, _, _, aux_uniform = load_stats(lg, E)
print(f"     均匀情形实测 aux_loss = {aux_uniform:.4f}（≈1，符合下界）")

# ============================================================
# C. 参数量分配：embedding / attention / FFN
# ============================================================
print()
print("=" * 74)
print("【C】参数量分配（GQA 的 attention，FFN 用 d_ff）")


def param_split(L, d, hq, hkv, dh, dff, V, tie_embed=True):
    emb = V * d
    attn = L * (d * hq * dh          # q
                + 2 * d * hkv * dh   # k, v
                + hq * dh * d)       # o
    ffn = L * 3 * d * dff            # gate, up, down
    lm = 0 if tie_embed else V * d
    return emb, attn, ffn, lm


print(f"  {'模型':>16} {'L':>4} {'d':>6} {'h_q/h_kv':>10} {'d_ff':>7} "
      f"{'emb%':>7} {'attn%':>7} {'ffn%':>7} {'总(B)':>9}")
for name, (L, d, hq, hkv, dh, dff, V) in [
    ("小玩具",      (6, 256, 8, 8, 32, 1024, 32000)),
    ("GPT-2 small", (12, 768, 12, 12, 64, 3072, 50257)),
    ("Llama-3-8B",  (32, 4096, 32, 8, 128, 14336, 128256)),
    ("Llama-2-70B", (80, 8192, 64, 8, 128, 28672, 32000)),
    ("Qwen3-30B-A3", (48, 2048, 32, 4, 128, 768, 151936)),
]:
    emb, attn, ffn, lm = param_split(L, d, hq, hkv, dh, dff, V)
    tot = emb + attn + ffn + lm
    print(f"  {name:>16} {L:>4} {d:>6} {f'{hq}/{hkv}':>10} {dff:>7} "
          f"{emb/tot*100:>6.1f}% {attn/tot*100:>6.1f}% {ffn/tot*100:>6.1f}% "
          f"{tot/1e9:>9.3f}")
print("  ⇒ 注意 Qwen3-30B-A3B 那一行是本探测**手填的结构参数**，")
print("    不是从官方 config 读的 —— 真实值待事实研究确认后替换。")

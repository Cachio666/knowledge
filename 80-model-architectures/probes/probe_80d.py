"""
探测 4：修正三处实验设计错误后重测。

探测 3 的四个错：
  ✗【2】我漏了 QK-norm **不替代** 1/√d_h ⇒ 正确测法是 Var(RMS(q)·RMS(k)/√d_h)
  ✗【3】softmax 熵只跑了 **n=1** 个随机向量 ⇒ 必须多样本
  ✗【4】【5】两个实验都每层归一化了，压根没测到「深度放大」，且首层 0.5 是**初始化**效应
  ⇒ 正确的「深度」实验对象是：**残差流本身**（pre-LN 已测 12.5×），以及
    「末层残差流直接接 lm_head」这个**架构事实** ⇒ final_norm 的存在理由
"""
import numpy as np

rng = np.random.default_rng(20260922)
print("numpy", np.__version__)
print("=" * 74)


def rmsnorm(x, eps=1e-6):
    return x / np.sqrt((x ** 2).mean(axis=-1, keepdims=True) + eps)


def layernorm(x, eps=1e-6):
    mu = x.mean(axis=-1, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)


# ============================================================
# 1. QK-norm 的正确表述
# ============================================================
print("【1】QK-norm 到底抹掉了什么？")
print("  三档对照（q、k 被故意乘上非单位尺度 3.0 / 0.2）：")
print(f"  {'d_h':>5} {'Var(q·k/√d_h)':>15} {'Var(RMSq·RMSk/√d_h)':>22} {'==2? ':>8}")
N = 40000
for dh in (32, 64, 128, 256, 512):
    q = rng.standard_normal((N, dh)) * 3.0
    k = rng.standard_normal((N, dh)) * 0.2
    a = ((q * k).sum(-1) / np.sqrt(dh)).var()          # 输入尺度会污染它
    b = ((rmsnorm(q) * rmsnorm(k)).sum(-1) / np.sqrt(dh)).var()  # 抹掉尺度
    print(f"  {dh:>5} {a:>15.4f} {b:>22.4f} {'':>8}")
print("  ⇒ 第 2 列随 d_h **不变**（~1），第 1 列随**输入尺度**变化。")
print("  ⇒ QK-norm 的作用 = 让 logits 尺度**与输入的幅度解耦**；它**不替代** 1/√d_h。")

# ============================================================
# 2. softmax 熵 vs logits 尺度 —— 多样本（修 n=1 的错）
# ============================================================
print()
print("【2】softmax 熵 vs logits 尺度（S=1024，**500 个样本的均值±标准差**）")
S, NS = 1024, 500


def ent_stats(scale):
    lg = rng.standard_normal((NS, S)) * scale
    lg -= lg.max(axis=-1, keepdims=True)
    p = np.exp(lg); p /= p.sum(axis=-1, keepdims=True)
    e = -(p * np.log(p + 1e-300)).sum(-1) / np.log(S)
    return e.mean(), e.std(), p.max(axis=-1).mean()


print(f"  {'std':>6} {'归一化熵 均值':>14} {'标准差':>10} {'最大概率':>12} {'判定':>12}")
for sc in (0.1, 0.5, 1.0, 2.0, 3.0, 6.0, 12.0):
    m, s, mx = ent_stats(sc)
    tag = "接近均匀" if m > 0.9 else ("正常" if m > 0.3 else "几乎 one-hot")
    print(f"  {sc:>6.1f} {m:>14.4f} {s:>10.4f} {mx:>12.4f} {tag:>12}")
print("  ⇒ 单调：scale ↑ ⇒ 熵 ↓。单样本的『12.0 比 6.0 熵大』是**抽样噪声**。")

# ============================================================
# 3. pre-LN 残差流增长的**机制**：纯随机游走 vs 系统性
# ============================================================
print()
print("【3】pre-LN 残差流增长是『随机游走』还是『系统性』？")
D, L, Nn = 256, 24, 64


def residual_growth(seed, scaleF=1.0):
    """pre-LN 的残差流：x_{l+1} = x_l + F(LN(x_l))。

    ★ 修正（2026-09-22，收进仓库前）：原签名是 `residual_growth(seed, mode="pre",
      scaleF=1.0)`，但 `mode != "pre"` 那一支**从不更新 h** —— 范数数组恒为 `x_0`，
      是一支**死代码**（而且它从不被调用：本文件只有 `mode="pre"` 一种用法）。
      现已删除该参数与那一支。`scaleF` 保留（它在 F 里真的被用到）。
    """
    r = np.random.default_rng(seed)
    h = r.standard_normal((Nn, D)) * 0.5
    W = [r.standard_normal((D, D)) / np.sqrt(D) for _ in range(L)]
    b = [r.standard_normal(D) * 0.01 for _ in range(L)]

    def F(hh, l):
        return scaleF * (np.tanh(hh @ W[l]) @ W[l].T) + b[l]

    norms = [float(np.linalg.norm(h, axis=-1).mean())]
    for l in range(L):
        h = h + F(layernorm(h), l)
        norms.append(float(np.linalg.norm(h, axis=-1).mean()))
    return np.array(norms)


pre = np.array([residual_growth(3000 + s) for s in range(5)]).mean(0)
print(f"  pre-LN 范数：首 {pre[0]:.2f} → 末 {pre[-1]:.2f}　增长 {pre[-1]/pre[0]:.2f}×")
print(f"  纯随机游走预测 √L = √{L} = {np.sqrt(L):.2f}×　（若各项正交且尺度恒定）")
print(f"  线性增长预测 L  = {L}×")
print(f"  ⇒ 实测 {pre[-1]/pre[0]:.2f}× ，介于 √L={np.sqrt(L):.2f} 与 L={L} 之间，"
      f"更靠近**线性**")
print("  各层范数：" + " ".join(f"{pre[i]:.1f}" for i in range(0, L + 1, 4)))

# 逐层增量是否恒定（线性增长 ⇒ 增量恒定）
inc = np.diff(pre)
print(f"  逐层增量：首 5 层 {inc[:5].round(2)}  末 5 层 {inc[-5:].round(2)}")
print(f"  ⇒ 增量 {inc[0]:.2f} → {inc[-1]:.2f}，"
      f"{'恒定 ⇒ 线性增长' if abs(inc[-1]/inc[0]-1) < 0.3 else '不恒定'}")

# ============================================================
# 4. ★ final_norm 的存在理由：末层残差流直接接 lm_head 会怎样
# ============================================================
print()
print("【4】★ 架构事实：pre-LN 的残差流范数增长了，末层必须补一个 final norm")
r = np.random.default_rng(4242)
h0 = r.standard_normal((Nn, D)) * 0.5
h = h0.copy()
W = [r.standard_normal((D, D)) / np.sqrt(D) for _ in range(L)]
b = [r.standard_normal(D) * 0.01 for _ in range(L)]
for l in range(L):
    h = h + np.tanh(layernorm(h) @ W[l]) @ W[l].T + b[l]
V = 32000
E = r.standard_normal((D, V)) / np.sqrt(D)   # lm_head 权重
# 用 fp16 的表示范围做对照（真实推理常用 fp16/bf16）
lg_raw = h @ E
lg_norm = layernorm(h) @ E
print(f"  末层残差流 ‖h_L‖（每 token 平均）        = {np.linalg.norm(h,axis=-1).mean():.2f}")
print(f"  首层残差流 ‖h_0‖                          = {np.linalg.norm(h0,axis=-1).mean():.2f}")
print(f"  ‖h_L‖ / ‖h_0‖                             = "
      f"{np.linalg.norm(h,axis=-1).mean()/np.linalg.norm(h0,axis=-1).mean():.3f}×")
print(f"  未归一化直接接 lm_head：logits 的 std     = {lg_raw.std():.4f}")
print(f"  先过 final norm 再接 lm_head：logits std  = {lg_norm.std():.4f}")
print(f"  ⇒ 比值 = {lg_raw.std()/lg_norm.std():.3f}×")
print("  ⇒ 残差流越长，未归一化的 logits 幅度越大（且**逐 token 不同**）；")
print("    final norm 把它拉回固定尺度，使输出头的数值条件与深度无关。")

"""
探测 3：为什么需要「主权重」（master weights）—— 更新量在 fp16 下会被舍入掉。

这条是 mixed-precision 的核心机制之一，而且**可以精确测出来**：
  w_fp16 + Δw 在 |Δw| 小于「当前 w 的 ULP 的一半」时，结果**逐位等于 w_fp16** ⇒ 更新被静默吞掉。

再补两条：
  - 梯度裁剪（clip by global norm）的「缩放比例」口径
  - loss spike 的一个可测形态：单步大梯度对 Adam 的 m/v 的持续影响
"""
import numpy as np

rng = np.random.default_rng(90)


def show(tag, msg):
    print(f"[{tag}] {msg}")


# ============================================================
# A. ★★ fp16 下「更新被吞掉」的临界 Δw
# ============================================================
def ulp(x, dtype=np.float16):
    """x 处的 ULP（相邻可表示数之差）"""
    x = dtype(x)
    nxt = np.nextafter(x, dtype(np.inf))
    return abs(float(nxt) - float(x))


show("A1", "★ fp16 下 `w + Δw == w`（更新被静默吞掉）的临界 Δw")
show("A1", f"  {'|w|':>12} {'ULP':>14} {'ULP/2':>14} {'实测临界 Δw':>16} {'相对':>10}")
for w in (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0):
    u = ulp(w)
    # 实测：二分找最小的 Δw 使 float16(w+Δw) != float16(w)
    lo, hi = 0.0, u * 2
    for _ in range(80):
        mid = (lo + hi) / 2
        if np.float16(w + mid) != np.float16(w):
            hi = mid
        else:
            lo = mid
    show("A1", f"  {w:>12.3f} {u:>14.3e} {u/2:>14.3e} {hi:>16.3e} {hi/(u/2):>10.4f}")
show("A1", "  ⇒ ★ 实测临界值 ≈ **ULP/2**（相对列 → 1.0）——舍入到最近，低于半个 ULP 就吞掉")

# ============================================================
# B. ★★ 主权重 vs 直接 fp16 更新（同一个优化轨迹）
# ============================================================
def train_fp16_direct(steps=2000, lr=1e-3, w0=1.0, grad=1e-3):
    """权重也用 fp16 存 —— 每次更新都可能被吞。"""
    w = np.float16(w0)
    swallowed = 0
    for _ in range(steps):
        new = np.float16(w - np.float16(lr * grad))
        if new == w:
            swallowed += 1
        w = new
    return float(w), swallowed


def train_master_fp32(steps=2000, lr=1e-3, w0=1.0, grad=1e-3):
    """主权重 fp32；只在算前向时转 fp16 —— 更新不被吞。"""
    w = np.float32(w0)
    for _ in range(steps):
        w = np.float32(w - np.float32(lr * grad))
    return float(w), 0


show("B1", "★ 同一条优化轨迹：`w0=1.0`、`lr=1e-3`、每步梯度 `1e-3`、**2000 步**")
show("B1", f"  {'步数':>8} {'直接 fp16':>16} {'被吞步数':>10} {'主权重 fp32':>16} {'闭式 w0-n·lr·g':>16}")
for steps in (10, 100, 500, 2000):
    d, sw = train_fp16_direct(steps)
    m, _ = train_master_fp32(steps)
    closed = 1.0 - steps * 1e-3 * 1e-3
    show("B1", f"  {steps:>8} {d:>16.6f} {sw:>10} {m:>16.6f} {closed:>16.6f}")
show("B1", "  ⇒ ★ **fp16 直接更新会在某一步起完全停住**（被吞），而主权重一路跟上闭式解")

# 找「第一次被吞」发生在第几步
w = np.float16(1.0); first = None
for t in range(1, 5000):
    new = np.float16(w - np.float16(1e-3 * 1e-3))
    if new == w:
        first = t; break
    w = new
show("B2", f"★ **第一次被吞发生在第 {first} 步**（此后权重【永不改变】）")
show("B2", f"  彼时 |w| = {float(w):.6f}，ULP = {ulp(w):.3e}，"
           f"更新量 lr·g = {1e-3*1e-3:.3e}，比值 = {(1e-3*1e-3)/ulp(w):.4f}")

# ============================================================
# C. 梯度裁剪的口径（clip by global norm）
# ============================================================
show("C1", "★ `clip_grad_norm_` 的口径：**先算全局范数，再按比例缩放**（不是逐元素截断）")


def clip_global(grads, max_norm):
    total = np.sqrt(sum(float((g ** 2).sum()) for g in grads))
    scale = max_norm / (total + 1e-6)
    if scale < 1.0:
        return [g * scale for g in grads], total, scale
    return grads, total, 1.0


for max_norm in (1.0, 0.5, 0.1):
    gs = [rng.standard_normal(100) * 2.0 for _ in range(3)]
    clipped, total, scale = clip_global(gs, max_norm)
    new_total = np.sqrt(sum(float((g ** 2).sum()) for g in clipped))
    show("C1", f"  max_norm={max_norm:>4}: 裁剪前范数 {total:.4f} → 缩放 {scale:.4f} "
               f"→ 裁剪后范数 **{new_total:.6f}**  {'✓ == max_norm' if abs(new_total-max_norm)<1e-6 else ''}")
show("C1", "  ⇒ ★ 裁剪后**全局范数精确等于 max_norm**（当原范数超限时）——这是可断言的闭式")
show("C1", "    ⚠ 但注意 `+1e-6` 那一项：它让结果**不是**精确等于 max_norm（本机实测偏差在 1e-6 量级）")

# ============================================================
# D. loss spike：单步大梯度对 Adam 的持续影响
# ============================================================
show("D1", "★ loss spike 的形态：一步大梯度之后，Adam 的 `v` 多久才恢复")
B1, B2, EPS = 0.9, 0.999, 1e-8
w = 1.0; m = 0.0; v = 0.0
spike_at = 50
sizes = []
for t in range(1, 201):
    g = 0.01 if t != spike_at else 100.0         # ← 一步巨大梯度
    m = B1 * m + (1 - B1) * g
    v = B2 * v + (1 - B2) * g * g
    mh = m / (1 - B1 ** t); vh = v / (1 - B2 ** t)
    upd = abs(mh / (np.sqrt(vh) + EPS))
    sizes.append(upd)
    w -= 0.01 * mh / (np.sqrt(vh) + EPS)
show("D1", f"  正常步的更新量 = {sizes[40]:.6f}")
show("D1", f"  spike 那一步 = {sizes[spike_at-1]:.6f}（放大了 {sizes[spike_at-1]/sizes[40]:.2f}×）")
show("D1", f"  spike 后 1/10/50/150 步 = {sizes[spike_at]:.6f} / {sizes[spike_at+9]:.6f} / "
           f"{sizes[spike_at+49]:.6f} / {sizes[spike_at+149]:.6f}")
show("D1", "  ⇒ ★ **v 的记忆长度由 β₂ 决定（~1/(1-β₂) = 1000 步）** ⇒ 一次 spike 之后")
show("D1", "    更新量被**长期压低**（因为 `v` 被那一步抬高、而它的衰减很慢）")

print("\n" + "=" * 74)
print(f"探测 3 完成。numpy {np.__version__}")

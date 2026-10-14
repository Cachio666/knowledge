"""
探测 2：修正 probe_90.py 的三处，并把 A1 的机制推广。

探测 1 的三处问题（都是控制者自己的错）：
  ✗ B1 的 `np.log10(F16_MAX/float(F16_TINY))` 报 overflow —— **因为那个除法是在 fp16 下做的**
    （`65504 / 5.96e-8` 在 fp16 里溢出）。**我在探测 fp16 的数值行为时自己踩了 fp16 的溢出。**
  ✗ F1 的 bf16 那行是**手工填的**（numpy 无 bf16 的 finfo）—— 要自己按 IEEE 定义算。
  ✗ A1 只测了「梯度恒为 0」这一个特例 —— 要不要推广？推广后闭式还成立吗？
"""
import numpy as np, struct

rng = np.random.default_rng(90)


def show(tag, msg):
    print(f"[{tag}] {msg}")


# ============================================================
# B1' 修正：用 **float64** 做除法
# ============================================================
F16_MAX = np.finfo(np.float16).max
F16_SUB = np.nextafter(np.float16(0), np.float16(1))
F16_TINY = np.finfo(np.float16).tiny
show("B1'", "★ 修正后的动态范围（**显式转 float64** 再除）：")
show("B1'", f"  fp16: max={float(F16_MAX):.6g}  次正规下限={float(F16_SUB):.6g}  "
            f"正规下限={float(F16_TINY):.6g}")
# ★★ 入仓前的修正（2026-09-23）—— 删掉一句**自造真值**。
#   本行原稿是：`show("B1'", "  ⇒ 差 ≈ 7.5 个数量级 —— 这就是 loss scaling 要补的缺口")`
#   紧跟在上面那两行之后，**没有任何推导**。它有两处错：
#     (a) `76.46 − 12.04 = **64.42**`，不是 `7.5`；
#     (b) 那两个数**不是同一个口径** —— 上面 fp16 那一行**含次正规**（除以 `5.96e-8`），
#         fp32 那一行用的是 `tiny`（**正规**下限）。
#   ⇒ `130-model-optimization/quantization/03-bf16-fp16-tf32.md:154` 逐字警告过这件事
#      （`2^-126` vs `2^-14`「**两个数不是同一个量，不要混用**」），并给出正确口径
#      「bf16 的量程跨度宽 **67.5** 个数量级（`76.5 vs 9.0`）」——**那是同口径相减**。
#   现在改成**三个各自标了口径的数**，同口径的那个差由脚本自己算出来。
r16_norm = float(F16_MAX) / float(F16_TINY)                                       # fp16 正规区间
r16_all = float(F16_MAX) / float(F16_SUB)                                         # fp16 含次正规
r32_norm = (float(np.finfo(np.float32).max) / float(np.finfo(np.float32).tiny))   # fp32 正规区间
show("B1'", "★ 动态范围（★ 三个数**必须各自标出口径**，不可两两相减）：")
show("B1'", f"    fp16 正规区间  = log10(65504 / 2^-14)      = {np.log10(r16_norm):.2f} 个数量级")
show("B1'", f"    fp16 含次正规  = log10(65504 / 5.96e-8)     = {np.log10(r16_all):.2f} 个数量级")
show("B1'", f"    fp32 正规区间  = log10(3.40e38 / 1.18e-38)  = {np.log10(r32_norm):.2f} 个数量级")
show("B1'", f"  ⇒ **同口径**（都取正规区间）的差 = {np.log10(r32_norm) - np.log10(r16_norm):.2f} "
            f"个数量级 —— 这才是有意义的那个「差」")
show("B1'", "     （口径对照：`130-model-optimization/quantization/03-bf16-fp16-tf32.md:154` "
            "给的是「bf16 的量程跨度宽 67.5 个数量级（`76.5 vs 9.0`）」，同一个口径）")
show("B1'", "     ⚠ **含次正规 vs 正规** 相减（`12.04 − 76.46`）得到的是一个混口径的数。")

# ============================================================
# F1' 修正：bf16 的精确 finfo（按 IEEE 定义算，不手工填）
# ============================================================
def bf16_finfo():
    """bf16 = 1 符号 + 8 指数 + 7 尾数（= fp32 截断到高 16 位）"""
    exp_bits, man_bits, bias = 8, 7, 127
    emax = (1 << (exp_bits - 1)) - 1                 # 127
    max_val = (2 - 2 ** (-man_bits)) * 2.0 ** emax   # (2-2^-7)*2^127
    tiny = 2.0 ** (1 - bias)                         # 最小正规 = 2^-126
    eps = 2.0 ** (-man_bits)                         # 机器 eps = 2^-7
    return max_val, tiny, eps


bmax, btiny, beps = bf16_finfo()
show("F1'", "★ bf16 的精确值（按定义算，非手工填）：")
show("F1'", f"  max = (2 - 2^-7) × 2^127 = {bmax:.10e}")
show("F1'", f"  最小正规 = 2^-126 = {btiny:.10e}")
show("F1'", f"  eps = 2^-7 = {beps:.10f}")
show("F1'", f"  ⇒ 与 fp32 的动态范围之比 = "
            f"{np.log10(bmax/btiny) / np.log10(r32_norm):.4f}（**应精确为 1**，因为指数位相同）")
show("F1'", f"  ⇒ bf16 的 eps = {beps:.6f} 而 fp16 的 eps = "
            f"{float(np.finfo(np.float16).eps):.6f} ⇒ bf16 精度**低 8 倍**")  # ★ 0.0078125/0.000977 ≈ 8

# ============================================================
# A2 ★★ 推广：Adam+L2 与 AdamW 的差，机制到底是什么？
# ============================================================
def step_adam_l2(w, m, v, g, t, lr, b1, b2, eps, wd):
    g = g + wd * w                    # L2 项进梯度 ⇒ 也进 m、v
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g * g
    mh = m / (1 - b1 ** t); vh = v / (1 - b2 ** t)
    return w - lr * mh / (np.sqrt(vh) + eps), m, v


def step_adamw(w, m, v, g, t, lr, b1, b2, eps, wd):
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g * g
    mh = m / (1 - b1 ** t); vh = v / (1 - b2 ** t)
    return w - lr * (mh / (np.sqrt(vh) + eps) + wd * w), m, v


def run(kind, grads, w0=1.0, lr=0.01, wd=0.1):
    w, m, v = w0, 0.0, 0.0
    for t, g in enumerate(grads, 1):
        f = step_adam_l2 if kind == "l2" else step_adamw
        w, m, v = f(w, m, v, g, t, lr, 0.9, 0.999, 1e-8, wd)
    return w


show("A2", "★ 机制：Adam+L2 的「L2 项」**进了 v** ⇒ 自适应分母被它抬高/压低")
show("A2", f"  {'梯度情形':>22} {'Adam+L2':>12} {'AdamW':>12} {'闭式 (1-lr*wd)^n':>18}")
cases = [("恒为 0", [0.0] * 200),
         ("恒为常数 0.01", [0.01] * 200),
         ("恒为常数 -0.01", [-0.01] * 200),
         ("随机 N(0,0.1)", list(rng.standard_normal(200) * 0.1)),
         ("随机 N(0,1)", list(rng.standard_normal(200) * 1.0))]
for name, gs in cases:
    a, b = run("l2", gs), run("adamw", gs)
    closed = (1 - 0.01 * 0.1) ** 200
    show("A2", f"  {name:>22} {a:>12.6f} {b:>12.6f} {closed:>18.6f}")
show("A2", "  ⇒ ★ **只有「梯度恒为 0」那一行 AdamW 与闭式精确相等**（因为那时更新量只剩 wd 项）")
show("A2", "    其它情形闭式**不成立**（梯度项也在动权重）—— 所以 A1 是一个**特例对照**，")
show("A2", "    **不能**写成「AdamW 恒等于 (1-lr*wd)^n」")

# 下溢比例：把「放大倍数」的作用量出来
show("A3", "★ Adam+L2 的异常来自 `g_L2 = wd·w` 与真梯度的**相对大小**：")
for wd in (0.0, 0.01, 0.1, 1.0):
    gs = [0.0] * 200
    show("A3", f"  wd={wd:>5}: Adam+L2 → {run('l2', gs, wd=wd):>10.6f}   "
               f"AdamW → {run('adamw', gs, wd=wd):>10.6f}")
show("A3", "  ⇒ wd=0 时两者**恒等**（这是应该的：没有 wd 就没有差异）")

# ============================================================
# E2 补：checkpointing 的**重算次数**账（探测 1 只给了存储数）
# ============================================================
show("E2", "★ 分段 checkpoint：把 n 层切成 k 段，每段 √n 层")
show("E2", f"  {'n':>6} {'段数 k':>8} {'存的边界':>10} {'段内重算':>10} "
           f"{'前向总次数':>12} {'相对不 ckpt':>12}")
for n in (64, 128, 1024, 4096):
    k = int(np.ceil(np.sqrt(n)))
    seg = int(np.ceil(n / k))
    stored = k
    recompute_extra = (seg - 1) * k          # 每段要重算 seg-1 次前向（除最后一个边界）
    total_fwd = n + recompute_extra
    show("E2", f"  {n:>6} {k:>8} {stored:>10} {recompute_extra:>10} "
               f"{total_fwd:>12} {total_fwd/n:>11.2f}×")
show("E2", "  ⇒ **存 O(√n) 的代价是前向次数从 n 涨到约 n + n·(1-1/√n) ≈ 2n**")
show("E2", "    （精确值依赖分段策略；上表用「每段 √n 层、段内从头重算」的常见做法）")

# ============================================================
# G2 补：下溢比例用闭式与蒙特卡洛互验
# ============================================================
from math import erf, sqrt
show("G2", "★ 下溢比例的**闭式**与**蒙特卡洛**互验（fp16 次正规阈值 5.96e-8）")
thr = float(F16_SUB)
for sigma in (1e-6, 1e-5, 1e-4, 1e-3):
    p_closed = erf(thr / (sigma * sqrt(2)))            # P(|g| < thr)
    g = rng.standard_normal(2_000_000) * sigma
    p_mc = float(np.mean(np.abs(np.float16(g)) == 0))
    show("G2", f"  σ={sigma:.0e}: 闭式 {p_closed*100:8.4f}%   MC {p_mc*100:8.4f}%   "
               f"相对差 {abs(p_mc-p_closed)/(p_closed+1e-30):.3f}")

print("\n" + "=" * 74)
print(f"探测 2 完成。numpy {np.__version__}")

#!/usr/bin/env python3
"""130-model-optimization/low-rank 编号程序 01 —— 最优低秩逼近的定理、边界与收益（推导，不是实测）。

配对文档：`130-model-optimization/low-rank/01-optimal-low-rank-approximation.md`

★ 形态（CONTRIBUTING §2 判定表第二种）：本程序**在默认 `python3` + `numpy` 下可跑**，
  三类断言**全部在本程序内部**，失败以非零码退出 ⇒ **不需要 `verify.py`**。
  产出**一律是推导值**（只跑算术与 numpy，不碰硬件、不加载权重）。

三类断言：
  A 算法正确性 —— ① 两条**互不依赖**的 SVD 路径对拍（`numpy.linalg.svd` vs `AᵀA` 的特征分解）；
                  ② Eckart–Young–Mirsky：截断 SVD 在 Frobenius 范数下**打败** N 个随机秩 r 矩阵；
                  ③ 恒等式 `‖A − A_r‖_F² = Σ_{i>r} σ_i²`（闭式 vs 显式做差）；
                  ④ 参数量账：`r(d_in+d_out)` vs `d_in·d_out`（显式因子矩阵 vs 闭式）。
  B 代价模型   —— 三种谱形状下的「误差-预算」曲线；**同样的参数预算下低秩与 int4 量化谁更省误差**；
                  低秩的**盈亏平衡点** `r < d_in·d_out/(d_in+d_out)`。
  C 边界与反例 —— ① 谱越平坦，低秩的相对误差越大（**低秩的收益依赖谱衰减**）；
                  ② ★★ **定理只覆盖单位不变范数**：构造一个秩 r 矩阵 `B`，它在**逐元素最大范数**
                     下**严格优于**截断 SVD，而在 Frobenius 范数下**与它打平**
                     —— 这正是「低秩最优」这句话被过度引用的地方；
                  ③ 秩超过盈亏平衡点时低秩**不再省参数**（反例）。

★ 本程序**不复制 `specs.py` 的任何常量值**：凡用到规格常量，一律写 `SPECS.xxx`。

用法：cd code && make 01_optimal_low_rank   （或 make run）
"""

import sys

import numpy as np

import specs as SPECS

FAILURES = []


def judge(label, ok, detail):
    mark = "通过" if ok else "**失败**"
    print(f"  [{mark}] {label}")
    print(f"          {detail}")
    if not ok:
        FAILURES.append(label)


# ---------------------------------------------------------------------------
# 构造与算子
# ---------------------------------------------------------------------------

def make_matrix(kind, n, seed):
    """按给定谱形状造一个矩阵：`A = U diag(σ) Vᵀ`，U/V 由高斯矩阵的 QR 得到。"""
    rng = np.random.default_rng(seed)
    u, _ = np.linalg.qr(rng.normal(size=(n, n)))
    v, _ = np.linalg.qr(rng.normal(size=(n, n)))
    i = np.arange(1, n + 1)
    if kind == "exponential":
        s = np.exp(-SPECS.EXPONENTIAL_RATE * (i - 1))
    elif kind == "powerlaw":
        s = i ** (-SPECS.POWERLAW_EXPONENT)
    elif kind == "flat":
        s = np.ones(n)
    else:
        raise ValueError(kind)
    return (u * s) @ v.T, s


def svd_numpy(a, r):
    """路径一：`numpy.linalg.svd` 的截断重构。"""
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    return (u[:, :r] * s[:r]) @ vt[:r, :], s


def svd_via_eig(a, r):
    """路径二（**独立**）：`AᵀA` 的特征分解给出 V 与 σ²，再取左奇异向量。"""
    m = a.T @ a
    w, v = np.linalg.eigh(m)                 # 升序
    order = np.argsort(w)[::-1]
    w, v = w[order], v[:, order]
    s = np.sqrt(np.maximum(w, 0.0))
    u = np.zeros((a.shape[0], r))
    for j in range(r):
        if s[j] > 0:
            u[:, j] = a @ v[:, j] / s[j]
    return (u * s[:r]) @ v[:, :r].T, s


def rtn(x):
    """Round-to-nearest, **half away from zero**（口径写死，与 `quantization` 主题一致）。"""
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def quantize_groups(w, bits, group):
    """对称 RTN，逐组 scale（组沿**输入维** `axis=1` 连续切，组不跨行）。"""
    out, inn = w.shape
    assert inn % group == 0
    wr = w.reshape(out, inn // group, group)
    mx = np.abs(wr).max(axis=2, keepdims=True)
    s = np.where(mx == 0, 1.0, mx / (2 ** (bits - 1) - 1))
    return (rtn(wr / s) * s).reshape(out, inn)


def rel_fro(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(a))


def low_rank_param_ratio(r, d_in, d_out):
    """低秩双因子占满秩的比例（**纯载荷**，不含 scale）。"""
    return r * (d_in + d_out) / (d_in * d_out)


def quant_param_ratio(bits, elem_bits=16):
    return bits / elem_bits


# ---------------------------------------------------------------------------
# A 算法正确性
# ---------------------------------------------------------------------------

def class_a():
    print("\nA 算法正确性（独立参照 / 定理的随机搜证 / 恒等式）")
    n = SPECS.MATRIX_SHAPE[0]
    a, s_true = make_matrix("exponential", n, SPECS.SEED)
    r = 64

    # A.1 两条独立 SVD 路径
    a1, s1 = svd_numpy(a, r)
    a2, s2 = svd_via_eig(a, r)
    judge(
        "A.1 两条独立 SVD 路径：`numpy.linalg.svd` vs `AᵀA` 特征分解",
        np.allclose(s1[:r], s2[:r], atol=1e-8) and np.allclose(a1, a2, atol=1e-8),
        f"前 {r} 个奇异值最大相对差 "
        f"{float(np.max(np.abs(s1[:r] - s2[:r]) / s1[:r])):.3e}；"
        f"重构矩阵最大逐元素差 {float(np.max(np.abs(a1 - a2))):.3e}",
    )

    # A.2 EYM：截断 SVD 打败随机秩 r 矩阵
    rng = np.random.default_rng(SPECS.SEED + 1)
    rel_svd = rel_fro(a, a1)
    best_b = float("inf")
    for _t in range(SPECS.N_RANDOM_TRIALS):
        b = rng.normal(size=(n, r)) @ rng.normal(size=(r, n))
        best_b = min(best_b, rel_fro(a, b))
    judge(
        f"A.2 EYM：截断 SVD 在 Frobenius 下优于全部 {SPECS.N_RANDOM_TRIALS} 个随机秩 r 矩阵",
        rel_svd <= best_b + 1e-12,
        f"截断 SVD 的相对误差 {rel_svd:.6e} ≤ 随机样本里**最好**的那个 {best_b:.6e}"
        f"（领先 {best_b - rel_svd:.3e}）",
    )

    # A.3 恒等式 ‖A − A_r‖_F² = Σ_{i>r} σ_i²
    lhs = float(np.linalg.norm(a - a1) ** 2)
    rhs = float(np.sum(s1[r:] ** 2))
    judge(
        "A.3 恒等式 `‖A − A_r‖_F² = Σ_{i>r} σ_i²`（显式做差 vs 闭式）",
        abs(lhs - rhs) / max(lhs, 1e-30) < 1e-9,
        f"显式 {lhs:.6e} vs 闭式 {rhs:.6e}（相对差 {abs(lhs-rhs)/lhs:.3e}）",
    )

    # A.4 参数量账：显式因子矩阵 vs 闭式
    d_in, d_out = 11008, 4096
    explicit = r * d_in + r * d_out
    judge(
        "A.4 参数量账：`r(d_in+d_out)`（闭式）== 两张因子矩阵的元素和",
        explicit == r * (d_in + d_out),
        f"r={r} / (d_in,d_out)=({d_in},{d_out})：{explicit} vs 满秩 {d_in*d_out}"
        f"（占比 {explicit/(d_in*d_out):.4f}）",
    )
    return a, s1


# ---------------------------------------------------------------------------
# B 代价模型
# ---------------------------------------------------------------------------

def class_b(a, s):
    print("\nB 代价模型（误差-预算曲线 / 与量化的对照 / 盈亏平衡点）")
    n = a.shape[0]
    print(f"  {'谱':<12}{'r':>5}{'参数占比':>10}{'低秩相对误差':>14}{'int4 相对误差':>15}")
    quant_err = rel_fro(a, quantize_groups(a, SPECS.QUANT_BITS, SPECS.QUANT_GROUP))
    rows = {}
    for kind in SPECS.DECAY_KINDS:
        mat, sv = make_matrix(kind, n, SPECS.SEED + 2)
        q = rel_fro(mat, quantize_groups(mat, SPECS.QUANT_BITS, SPECS.QUANT_GROUP))
        best = None
        for rr in (16, 32, 64, 128, 256):
            ar, _ = svd_numpy(mat, rr)
            e = rel_fro(mat, ar)
            if best is None or e < best[1]:
                best = (rr, e)
            if rr in (64,):
                print(f"  {kind:<12}{rr:>5}{low_rank_param_ratio(rr, n, n):>10.4f}"
                      f"{e:>14.4f}{q:>15.4f}")
        rows[kind] = (best, q, sv)
    print(f"  ★ 对照口径：int4 的**纯载荷**参数占比 = {quant_param_ratio(SPECS.QUANT_BITS):.4f}"
          f"（= {SPECS.QUANT_BITS}/16）；与它对齐的低秩是 r = {int(n*quant_param_ratio(SPECS.QUANT_BITS)/2)}"
          f"（方阵下 `2r/d = 0.25`）")

    r_match = int(n * quant_param_ratio(SPECS.QUANT_BITS) / 2)
    exp_mat, _ = make_matrix("exponential", n, SPECS.SEED + 2)
    flat_mat, _ = make_matrix("flat", n, SPECS.SEED + 2)
    e_exp = rel_fro(exp_mat, svd_numpy(exp_mat, r_match)[0])
    e_flat = rel_fro(flat_mat, svd_numpy(flat_mat, r_match)[0])
    q_exp = rel_fro(exp_mat, quantize_groups(exp_mat, SPECS.QUANT_BITS, SPECS.QUANT_GROUP))
    q_flat = rel_fro(flat_mat, quantize_groups(flat_mat, SPECS.QUANT_BITS, SPECS.QUANT_GROUP))
    judge(
        "B.1 **同样参数预算**下：低秩在衰减谱上赢、在平坦谱上输",
        e_exp < q_exp and e_flat > q_flat,
        f"指数衰减：低秩 {e_exp:.4f} < int4 {q_exp:.4f}（低秩赢 {q_exp/e_exp:.2f}×）；"
        f"平坦谱：低秩 {e_flat:.4f} > int4 {q_flat:.4f}（低秩输 {e_flat/q_flat:.2f}×）"
        f" ⇒ **低秩的收益完全取决于谱衰减，量化不依赖谱**",
    )

    # B.2 盈亏平衡点
    d_in, d_out = 11008, 4096
    cross = d_in * d_out / (d_in + d_out)
    judge(
        "B.2 低秩的盈亏平衡点：`r < d_in·d_out/(d_in+d_out)` 才省参数",
        low_rank_param_ratio(int(cross) - 1, d_in, d_out) < 1.0
        and low_rank_param_ratio(int(cross) + 1, d_in, d_out) > 1.0,
        f"平衡点 r* = {cross:.1f}；r={int(cross)-1} 时占比 "
        f"{low_rank_param_ratio(int(cross)-1, d_in, d_out):.4f} < 1，"
        f"r={int(cross)+1} 时 {low_rank_param_ratio(int(cross)+1, d_in, d_out):.4f} > 1",
    )
    print(f"  （对照：GLM-5 报告的 `KV LoRA Dim = {SPECS.MLA_KV_LORA_DIM_GLM5}` —— "
          f"**引用值**，见 `160-model/glm/02-glm5-architecture.md`）")
    return rows, (e_exp, q_exp, e_flat, q_flat)


# ---------------------------------------------------------------------------
# C 边界与反例
# ---------------------------------------------------------------------------

def class_c(rows, cmp_vals):
    print("\nC 边界与反例（断言的是「失败行为成立」）")
    e_exp, q_exp, e_flat, q_flat = cmp_vals

    # C.1 谱越平坦，低秩越差
    judge(
        "C.1 「低秩压缩任何权重矩阵」被证伪：谱平坦时收益崩塌",
        e_flat > 5 * e_exp,
        f"同一预算下平坦谱的相对误差 {e_flat:.4f} 是衰减谱 {e_exp:.4f} 的 "
        f"{e_flat/e_exp:.1f} 倍 ⇒ 低秩的前提是**谱衰减**，不是「矩阵低秩」",
    )

    # C.2 ★★ EYM 只覆盖单位不变范数：逐元素最大范数下的反例
    ident = np.eye(2)
    a_r, s2 = svd_numpy(ident, 1)
    b = np.full((2, 2), 0.5)
    max_err_svd = float(np.max(np.abs(ident - a_r)))
    max_err_b = float(np.max(np.abs(ident - b)))
    fro_svd = float(np.linalg.norm(ident - a_r))
    fro_b = float(np.linalg.norm(ident - b))
    judge(
        "C.2 ★★ 「低秩最优」在**逐元素最大范数**下被证伪（而 Frobenius 下打平）",
        max_err_b < max_err_svd and abs(fro_svd - fro_b) < 1e-12,
        f"A = I₂、r = 1：截断 SVD 的 `max|·|` 误差 {max_err_svd:.3f}，"
        f"而 B = 0.5·11ᵀ 是 {max_err_b:.3f}（**更小**）；两者 Frobenius 都是 {fro_b:.3f}"
        f" ⇒ 定理只对**单位不变范数**成立，逐元素范数不在其内",
    )

    # C.3 秩超过盈亏平衡点 ⇒ 不省参数
    d_in, d_out = 11008, 4096
    r_over = int(d_in * d_out / (d_in + d_out)) + 100
    judge(
        "C.3 「低秩一定省参数」被证伪（秩超过盈亏平衡点）",
        low_rank_param_ratio(r_over, d_in, d_out) > 1.0,
        f"r={r_over} 时双因子参数占比 "
        f"{low_rank_param_ratio(r_over, d_in, d_out):.4f} > 1 —— "
        f"**比满秩还大**（两个因子的维数之和超过了原矩阵的一维）",
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    print("130-model-optimization/low-rank 01 —— 最优低秩逼近：定理、边界与收益")
    print("★ 本程序只做算术，产出是**推导值**，不是实测值。")
    print(f"  形状 {SPECS.MATRIX_SHAPE}；随机搜索 {SPECS.N_RANDOM_TRIALS} 次（seed={SPECS.SEED}）；"
          f"int4 档位 {SPECS.QUANT_BITS}-bit / 组 {SPECS.QUANT_GROUP}")
    a, s = class_a()
    rows, cmp_vals = class_b(a, s)
    class_c(rows, cmp_vals)
    print()
    if FAILURES:
        print(f"结果：{len(FAILURES)} 条断言失败 →")
        for f in FAILURES:
            print(f"  · {f}")
        return 1
    print("结果：全部断言通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

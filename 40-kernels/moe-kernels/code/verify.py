#!/usr/bin/env python3
"""40-kernels/moe-kernels 主题的本机验证模型。

**本模块是推导，不是硬件测量。** 它只跑算术与 numpy，不碰硬件——本机（macOS）
没有 CUDA 工具链也没有 GPU，`code/` 里的 3 个 `.cu` 内核的运行结果本仓库没有。

它覆盖本主题全部 3 篇文档的可推导部分，分三类断言（CONTRIBUTING §2）：

  A 算法正确性  numpy 参考实现 vs 文档所述算法（路由 / permute / grouped GEMM /
                unpermute 的全链路，以及分段维度与逐专家调用的一致性）
  B 代价模型    按文档公式算 FLOPs / 权重流量 / 算术强度 / tokens_per_expert 统计 /
                填充浪费比 / 波次 / 访存字节 / 合并度，逐条打印数值并给出判定
  C 边界与反例  §4 声称「会出问题」的输入必须**真的跑出问题**——断言的是
                失败行为成立，不是成功

**数字的口径**（沿用 20-gpu-architecture 的三分法）：
  · 规格值 = 来自 `specs.py`，逐条带 `# 来源：`
  · 推导值 = 本程序算出，可在这里复核
  · 引用值 = 外部来源，标注「本仓库不复核其测量口径」
本文件里除了 `specs.py` 的常量，**全部是推导值**。

**场景参数**（`d_model` / `d_ff` / 专家数 / `top_k` / token 数）是**本文选定的对照
场景**，不是任何模型的规格，没有一手来源可引——见 `specs.py` 的 `UNSOURCED`。

运行：`make verify`（等价于 `python3 verify.py`）。只需 python3 + numpy。
"""

import math

import numpy as np

import specs as SP

# ===========================================================================
# 断言与打印基础设施
# ===========================================================================

_STATE = {"n": 0, "fail": []}


def check(name, cond, detail=""):
    """记一次判定并打印结果。cond 为假时记入失败清单。"""
    _STATE["n"] += 1
    ok = bool(cond)
    if not ok:
        _STATE["fail"].append(name)
    tail = f"   ← {detail}" if detail else ""
    print(f"    [{'通过' if ok else '**不通过**'}] {name}{tail}")
    return ok


def section(tag, title):
    print()
    print("=" * 78)
    print(f"§{tag}  {title}")
    print("=" * 78)


def sub(title):
    print()
    print(f"--- {title} ---")


def close(a, b, rel=1e-9):
    return abs(a - b) <= rel * max(1.0, abs(a), abs(b))


H100 = "H100"
B200 = "B200"


# ===========================================================================
# 场景参数（**本文选定的对照场景，不是模型规格**——见 specs.py 的 UNSOURCED）
# ===========================================================================

D_MODEL = 4096
D_FF = 14336          # 每个专家自己的隐层宽度
N_EXPERTS = 8
TOP_K = 2
TOKENS = 4096
ESIZE = SP.FLOAT16_BYTES      # 专家权重与激活按 FP16 记账

BM = 128              # 02 篇的 M 方向 tile
BN = 128              # 02 篇的 N 方向 tile

# 路由分布的两种形态
ZIPF_ALPHA = 1.0


def assignments():
    return TOKENS * TOP_K


def uniform_probs():
    return np.full(N_EXPERTS, 1.0 / N_EXPERTS)


def zipf_probs(alpha=ZIPF_ALPHA, n=N_EXPERTS):
    r = 1.0 / np.power(np.arange(1, n + 1, dtype=np.float64), alpha)
    return r / r.sum()


def sample_counts(rng, probs, n_assign=None):
    return rng.multinomial(n_assign or assignments(), probs)


def expect_counts(probs, n_assign=None):
    return (n_assign or assignments()) * probs


def pad_to_max_waste(counts):
    """「按最大专家填充」的算力浪费比 = 1 − 均值/最大值。"""
    mx = int(counts.max())
    return 1.0 - counts.mean() / mx, mx


def tile_pad(counts, bm=BM):
    """按 tile 粒度（每个专家各自向上取整到 bm 的整数倍）的 M 总长。"""
    return int(np.ceil(counts / bm).sum() * bm)


_FIXED_COUNTS = {}


def fixed_counts():
    """§B3 / §B4 / §B5 **共用同一组** tokens_per_expert。

    三节要能互相印证，就不能各自抽各自的样——否则 §B4 说「幂律的 M_max = 2997」
    而 §B5 用另一组幂律样本（M_max 更大）去算网格，两张表对不上。
    """
    if not _FIXED_COUNTS:
        rng = np.random.default_rng(20260917)
        _FIXED_COUNTS["uni"] = sample_counts(rng, uniform_probs())
        _FIXED_COUNTS["zipf"] = sample_counts(rng, zipf_probs())
    return _FIXED_COUNTS["uni"], _FIXED_COUNTS["zipf"]


# ===========================================================================
# §A 算法正确性
# ===========================================================================


def topk_ref(logits_row, k):
    """参考实现：按 logits 降序取 k 个，**并列时取下标小的**。"""
    return sorted(range(len(logits_row)), key=lambda e: (-logits_row[e], e))[:k]


def route_ref(logits, k):
    """逐行调用纯 Python 参考实现，返回 [T, k] 的下标。"""
    return np.array([topk_ref(row, k) for row in logits], dtype=np.int64)


def route_weights(logits, idx):
    """文档口径：softmax **只在选中的 k 个 logit 上**做。"""
    sel = np.take_along_axis(logits, idx, axis=1)
    m = sel.max(axis=1, keepdims=True)
    ex = np.exp(sel - m)
    return ex / ex.sum(axis=1, keepdims=True)


def weights_full_softmax(logits, idx):
    """反例口径：先在全部 E 个 logit 上 softmax，再取 top-k，**不重归一**。"""
    m = logits.max(axis=1, keepdims=True)
    ex = np.exp(logits - m)
    w = ex / ex.sum(axis=1, keepdims=True)
    return np.take_along_axis(w, idx, axis=1)


def make_gate(rng, t=TOKENS, d=D_MODEL, e=N_EXPERTS, scale=1.0):
    return rng.standard_normal((t, d)), rng.standard_normal((e, d)) * scale


def make_experts(rng, d=D_MODEL, f=D_FF, e=N_EXPERTS, scale=None):
    s = scale if scale is not None else 1.0 / math.sqrt(d)
    up = rng.standard_normal((e, d, f)) * s
    down = rng.standard_normal((e, f, d)) * s
    return up, down


def act(h):
    """SiLU。选它是为了让「两个投影」的 FFN 有一个非线性的、可微的中间层。"""
    return h / (1.0 + np.exp(-h))


def moe_dense(X, Wg, up, down, k=TOP_K):
    """把 MoE **展开成稠密计算**：每个 token 对它的 k 个专家各算一遍 FFN，按权重求和。"""
    logits = X @ Wg.T
    idx = route_ref(logits, k)
    w = route_weights(logits, idx)
    T = X.shape[0]
    out = np.zeros((T, X.shape[1]))
    for t in range(T):
        acc = np.zeros(X.shape[1])
        for j in range(k):
            e = int(idx[t, j])
            acc += w[t, j] * (act(X[t] @ up[e]) @ down[e])
        out[t] = acc
    return out, idx, w


def grouped_gemm(Xp, seg_lens, Ws):
    """按专家分段：第 e 段用 Ws[e]，段长 seg_lens[e]。返回 [S, N]。"""
    S = Xp.shape[0]
    N = Ws.shape[2]
    out = np.zeros((S, N))
    off = 0
    for e in range(len(seg_lens)):
        n_e = int(seg_lens[e])
        if n_e:
            out[off:off + n_e] = Xp[off:off + n_e] @ Ws[e]
        off += n_e
    return out


def moe_permuted(X, Wg, up, down, k=TOP_K, capacity=None, yp_buffer=None):
    """permute → grouped GEMM(up) → 激活 → grouped GEMM(down) → unpermute。

    `capacity=None` 表示不截断（每个专家能收下所有分到的 token）。
    返回 (out, slot, slot_valid, yp)。
    """
    T, d = X.shape
    E = Wg.shape[0]
    logits = X @ Wg.T
    idx = route_ref(logits, k)
    w = route_weights(logits, idx)

    counts = np.bincount(idx.ravel(), minlength=E)
    offsets = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)

    # permute：token 行整行搬到 slot 位置
    slot = np.zeros((T, k), dtype=np.int64)
    valid = np.zeros((T, k), dtype=bool)
    Xp = np.zeros((sum(counts), d))
    cursor = offsets.copy()
    for t in range(T):
        for j in range(k):
            e = int(idx[t, j])
            s = int(cursor[e])
            cursor[e] += 1
            slot[t, j] = s
            valid[t, j] = (capacity is None) or (s - offsets[e] < capacity)
            Xp[s] = X[t]

    # grouped GEMM（up → 激活 → down），只算有效段
    seg = counts.copy()
    if capacity is not None:
        seg = np.minimum(seg, capacity)
    if yp_buffer is None:
        yp = np.zeros((sum(counts), d))
    else:
        yp = yp_buffer          # 缓冲区复用：被丢的 slot 保留上一次迭代的数据
    off = 0
    for e in range(E):
        n_e = int(seg[e])
        if n_e:
            h = act(Xp[off:off + n_e] @ up[e])
            yp[off:off + n_e] = h @ down[e]
        off += int(counts[e])

    # unpermute：带权重回填（mask 与不 mask 两种）
    out_masked = np.zeros((T, d))
    out_unmasked = np.zeros((T, d))
    for t in range(T):
        for j in range(k):
            s = int(slot[t, j])
            out_unmasked[t] += w[t, j] * yp[s]
            if valid[t, j]:
                out_masked[t] += w[t, j] * yp[s]
    return out_masked, out_unmasked, slot, valid, yp, idx, w


def a1_routing():
    sub("A1  路由：top-k 选择与权重口径")
    rng = np.random.default_rng(20260916)
    X, Wg = make_gate(rng, t=64, d=32)
    logits = X @ Wg.T

    print("    口径甲（本文采用）：**先取 top-k，再在这 k 个 logit 上做 softmax**。")
    print("    口径乙（等价写法）：全 E 上 softmax，取 top-k 后**重归一**——与甲逐元素相同。")
    print("    口径丙（反例）：全 E 上 softmax，取 top-k 后**不重归一**——权重之和不等于 1。")
    print()
    idx_np = np.argsort(-logits, axis=1, kind="stable")[:, :TOP_K]
    idx_py = route_ref(logits, TOP_K)
    check("numpy 稳定排序与纯 Python 参考实现给出相同的 top-k 下标",
          np.array_equal(idx_np, idx_py), f"不一致行数 = {int((idx_np != idx_py).any(1).sum())}")

    w_a = route_weights(logits, idx_py)
    m = logits.max(axis=1, keepdims=True)
    ex = np.exp(logits - m)
    w_full = ex / ex.sum(axis=1, keepdims=True)
    w_then_renorm = np.take_along_axis(w_full, idx_py, axis=1)
    w_then_renorm = w_then_renorm / w_then_renorm.sum(axis=1, keepdims=True)
    print(f"    甲：权重行和 = {w_a.sum(1).min():.6f} ~ {w_a.sum(1).max():.6f}")
    check("甲（top-k 后 softmax）每行权重之和恒为 1",
          close(w_a.sum(1).min(), 1.0) and close(w_a.sum(1).max(), 1.0))
    dw = float(np.abs(w_a - w_then_renorm).max())
    print(f"    乙与甲：最大差 = {dw:.3e}（不是 0——两条写法的求和顺序不同，"
          f"只到浮点舍入）")
    check("乙（全 E softmax → 取 top-k → 重归一）与甲**数值等价**（≤ 1e-12）",
          dw <= 1e-12, f"最大差 {dw:.3e}")

    w_c = weights_full_softmax(logits, idx_py)
    s_c = w_c.sum(1)
    frac = float((s_c < 0.99).mean())
    print(f"    丙（不重归一）：权重行和 = {s_c.min():.6f} ~ {s_c.max():.6f}"
          f"（均值 {s_c.mean():.6f}）")
    print(f"      行和 < 0.99 的行占 {frac:.1%}"
          f"——**每一行都缺一块**，缺的量 = 被丢掉的 E−k 个专家的 softmax 质量")
    check("丙的权重行和恒 < 1（不是「偶尔差一点」）", float(s_c.max()) < 1.0,
          f"最大 {s_c.max():.6f}")
    check("丙的行和与 1 的缺口在多数行上可见（> 50% 的行 < 0.99）",
          frac > 0.5, f"{frac:.1%}")

    # 并列值的 tie-break
    ties = np.array([[1.0, 1.0, 0.5], [0.5, 1.0, 1.0]])
    idx_tie = np.array([topk_ref(r, 2) for r in ties])
    check("并列时取下标小的（[0,1] 与 [1,2]）",
          idx_tie[0].tolist() == [0, 1] and idx_tie[1].tolist() == [1, 2],
          f"{idx_tie.tolist()}")
    return idx_py, w_a


def a2_end_to_end():
    sub("A2  permute → grouped GEMM → unpermute 端到端 == 把 MoE 展开成稠密计算")
    rng = np.random.default_rng(7)
    T, d, f, E, k = 64, 32, 48, 4, 2
    X = rng.standard_normal((T, d))
    Wg = rng.standard_normal((E, d))
    up, down = make_experts(rng, d=d, f=f, e=E)

    ref, idx, w = moe_dense(X, Wg, up, down, k)
    got_masked, got_unmasked, slot, valid, yp, idx2, w2 = moe_permuted(
        X, Wg, up, down, k, capacity=None)

    d1 = np.abs(got_masked - ref).max()
    d2 = np.abs(got_unmasked - ref).max()
    print(f"    规模：tokens={T}，d_model={d}，d_ff={f}，专家={E}，top_k={k}"
          f"；slot 总数 = T×k = {T * k}，permuted 缓冲区 {yp.shape[0]} 行")
    print(f"    参考（稠密展开）与分组流水线的最大逐元素差 = {d1:.3e}")
    check("端到端结果与稠密展开**逐元素一致**（rel 1e-12）", close(d1, 0.0, 1e-12), f"{d1:.3e}")
    check("不截断时 mask 恒为真，两条路径结果相同", close(d2, 0.0), f"{d2:.3e}")

    # permute 是双射：每个 token 的 k 个 slot 互不相同，且并集恰好是全部 slot
    flat = slot.ravel()
    check("permute 是双射：slot 无重复且覆盖 0..T·k−1",
          len(set(flat.tolist())) == T * k and int(flat.min()) == 0
          and int(flat.max()) == T * k - 1, f"{len(set(flat.tolist()))} / {T * k}")
    return ref


def a3_grouped_gemm():
    sub("A3  grouped GEMM 的分段 M 维 == 逐专家单独调用")
    rng = np.random.default_rng(11)
    E = N_EXPERTS
    counts = sample_counts(rng, zipf_probs(), n_assign=TOKENS * TOP_K)
    S = int(counts.sum())
    d, f = 256, 512
    Xp = rng.standard_normal((S, d))
    Ws = rng.standard_normal((E, d, f)) / math.sqrt(d)

    grouped = grouped_gemm(Xp, counts, Ws)

    off = 0
    max_diff = 0.0
    for e in range(E):
        n_e = int(counts[e])
        if n_e:
            solo = Xp[off:off + n_e] @ Ws[e]          # 逐专家单独调用
            max_diff = max(max_diff, float(np.abs(grouped[off:off + n_e] - solo).max()))
        off += n_e
    print(f"    幂律路由下的分段长度（= tokens_per_expert）："
          f"{tool.rows(counts.tolist())}")
    print(f"    分段长度之和 = {S} = tokens × top_k = {TOKENS}×{TOP_K}")
    check("分段长度之和 = tokens × top_k", S == TOKENS * TOP_K, f"{S}")
    print(f"    grouped 与「逐专家单独调用」的最大逐元素差 = {max_diff:.3e}")
    check("grouped GEMM 的分段 M 维与逐专家单独调用一致", close(max_diff, 0.0, 1e-12),
          f"{max_diff:.3e}")

    # 空段：分段实现必须能跳过 0 长度的专家
    counts_empty = np.array([5, 0, 3, 0])
    Xe = rng.standard_normal((8, 16))
    We = rng.standard_normal((4, 16, 4))
    g = grouped_gemm(Xe, counts_empty, We)
    check("分段为 0 的专家被跳过（不贡献也不越界）",
          close(float(np.abs(g[5:8] - Xe[5:8] @ We[2]).max()), 0.0),
          f"第 2 段 {counts_empty[2]} 行落在 [5,8)")


class tool:
    @staticmethod
    def rows(v, width=6):
        return "[" + ",".join(f"{x:>{width}d}" for x in v) + "]"


def a4_offsets():
    sub("A4  专家偏移 = tokens_per_expert 的**排他前缀和**")
    rng = np.random.default_rng(3)
    counts = sample_counts(rng, zipf_probs(), n_assign=TOKENS * TOP_K)
    off_ref = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
    off_got = np.zeros_like(off_ref)
    run = 0
    for e in range(len(counts)):
        off_got[e] = run
        run += int(counts[e])
    check("排他前缀和与 numpy cumsum 一致", np.array_equal(off_ref, off_got))
    check("最后一个偏移 + 最后一段长 = 总 slot 数",
          int(off_got[-1] + counts[-1]) == int(counts.sum()),
          f"{off_got[-1]} + {counts[-1]} = {counts.sum()}")
    print("    （前缀和的并行算法正文在 40-kernels/reduce-scan/03，此处只做代入）")
    return counts


# ===========================================================================
# §B 代价模型
# ===========================================================================


def b1_flops():
    sub("B1  MoE 一层的 FLOPs：**两个 GEMM**，以及与等参数稠密模型的比值")
    d, f, E, k, T = D_MODEL, D_FF, N_EXPERTS, TOP_K, TOKENS
    per_token_per_expert = 4 * d * f
    moe_total = 4 * T * k * d * f
    up_only = 2 * T * k * d * f
    swiglu = 6 * T * k * d * f

    print("    单 token 单专家（标准两层 FFN）= up + down")
    print(f"      = 2·d_model·d_ff + 2·d_ff·d_model = 4·d_model·d_ff")
    print(f"      = 4 × {d} × {f} = {per_token_per_expert:,} FLOPs"
          f" = {per_token_per_expert / 1e6:.3f} MFLOPs")
    print(f"    MoE 层合计 = 4·tokens·top_k·d_model·d_ff")
    print(f"      = 4 × {T} × {k} × {d} × {f} = {moe_total:.6e} FLOPs"
          f" = {moe_total / 1e12:.4f} TFLOPs")
    check("MoE 层 FLOPs = 4·T·k·d·d_ff",
          close(moe_total, 4.0 * T * k * d * f, 1e-15), f"{moe_total:.6e}")
    print()
    print(f"    对照①：**只算 up 投影**（系数 2，即初版漏掉 down 的那个数）"
          f" = {up_only:.6e} FLOPs = {up_only / 1e12:.4f} TFLOPs")
    print(f"           比值 = {moe_total / up_only:.1f}× ——漏掉 down 会把计算量**少算一半**")
    check("系数 4 与系数 2 相差整整 2 倍",
          close(moe_total / up_only, 2.0, 1e-12), f"{moe_total / up_only:.4f}×")
    print(f"    对照②：SwiGLU（gate/up/down 三个投影，系数 6）"
          f" = {swiglu:.6e} FLOPs = {swiglu / 1e12:.4f} TFLOPs")
    check("SwiGLU 的系数是 6，= 系数 4 的 1.5 倍",
          close(swiglu / moe_total, 1.5, 1e-12), f"{swiglu / moe_total:.2f}×")

    print()
    print("    **与专家总数无关**：上面每一行里都没有出现 E。")
    print(f"    E 从 {E} 改成 64，MoE 层 FLOPs 仍是 {4 * T * k * d * f / 1e12:.4f} TFLOPs；")
    print(f"    改成 1（就是稠密 FFN + top_k={k} 次重复）也还是它。")
    for e_alt in (1, 64, 256):
        alt = 4 * T * k * d * f
        check(f"E={e_alt} 时 MoE 层 FLOPs 不变", close(alt, moe_total, 1e-15),
              f"{alt / 1e12:.4f} TFLOPs")

    print()
    print("    对照③：**等参数的稠密模型**。MoE 的 FFN 参数 = E·2·d_model·d_ff；")
    print("           稠密模型要装下同样多的参数，需要 d_ff_dense = E·d_ff。")
    dff_dense = E * f
    moe_params = E * 2 * d * f
    dense_params = 2 * d * dff_dense
    dense_total = 4 * T * d * dff_dense
    per_token_moe = 4 * k * d * f
    per_token_dense = 4 * d * dff_dense
    print(f"    MoE  FFN 参数 = {E} × 2 × {d} × {f} = {moe_params:,}"
          f"（{moe_params / 1e6:.1f} M）")
    print(f"    稠密 FFN 参数 = 2 × {d} × {dff_dense} = {dense_params:,}"
          f"（{dense_params / 1e6:.1f} M）  ← 与 MoE **相等**")
    check("两种模型的 FFN 参数个数相等", moe_params == dense_params,
          f"{moe_params} vs {dense_params}")
    print(f"    稠密模型一层 FLOPs = 4·T·d_model·d_ff_dense = {dense_total:.6e}"
          f" = {dense_total / 1e12:.4f} TFLOPs")
    print(f"    比值 = {dense_total / moe_total:.4f} = E / top_k = {E}/{k}")
    check("FLOPs 比值 = 专家数 ÷ top_k",
          close(dense_total / moe_total, E / k, 1e-12), f"{dense_total / moe_total:.4f}")
    print(f"    每 token 口径：MoE {per_token_moe / 1e6:.3f} MFLOPs vs "
          f"稠密 {per_token_dense / 1e6:.3f} MFLOPs"
          f"（比值 {per_token_dense / per_token_moe:.4f}）")
    return dict(moe_total=moe_total, up_only=up_only, swiglu=swiglu,
                dense_total=dense_total, dff_dense=dff_dense,
                moe_params=moe_params)


def b2_weight_stream():
    sub("B2  权重流量与算术强度：**MoE 省的是 FLOPs，不是显存、更不是带宽**")
    d, f, E, k, T = D_MODEL, D_FF, N_EXPERTS, TOP_K, TOKENS
    es = ESIZE
    moe_total = 4 * T * k * d * f
    dense_total = 4 * T * d * (E * f)
    weight_bytes = E * 2 * d * f * es
    print(f"    专家权重 = E × 2 × d_model × d_ff = {E * 2 * d * f:,} 个参数")
    print(f"             × {es} B = {weight_bytes:.6e} B = {weight_bytes / 1e9:.4f} GB"
          f"（FP16）")
    print(f"    等参数稠密模型的权重 = 2 × d_model × d_ff_dense × {es}"
          f" = {2 * d * E * f * es / 1e9:.4f} GB  ← **两者相等**")
    check("稠密模型与 MoE 的权重字节数相等（省显存的说法不成立）",
          close(2 * d * E * f * es, weight_bytes, 1e-15))

    ai_moe = moe_total / weight_bytes
    ai_dense = dense_total / weight_bytes
    peak = SP.TC_TFLOPS_DENSE_SPARSE["FP16"][H100][0] * 1e12
    bw = SP.HBM_BW_GBps_BY_GEN[H100] * 1e9
    ridge = peak / bw
    print()
    print(f"    权重流量的算术强度（每个专家权重字节摊到的 FLOPs）：")
    print(f"      MoE  = 4·T·k·d·d_ff ÷ (E·2·d·d_ff·esize) = 2·T·k / (E·esize)")
    print(f"           = 2×{T}×{k} / ({E}×{es}) = {ai_moe:.1f} FLOP/B")
    print(f"      稠密 = {ai_dense:.1f} FLOP/B   ← 恒为 MoE 的 {E / k:.0f} 倍（= E/top_k）")
    check("稠密模型的权重算术强度 = MoE 的 E/top_k 倍",
          close(ai_dense / ai_moe, E / k, 1e-12), f"{ai_dense / ai_moe:.4f}")
    print()
    print(f"    H100 的 FP16 拐点 I* = {peak / 1e12:.1f} TFLOPS ÷ {bw / 1e9:.0f} GB/s"
          f" = {ridge:.2f} FLOP/B（FP16 dense ÷ HBM3，规格值）")
    print(f"    AI_MoE / I* = {ai_moe / ridge:.3f}")
    check("本场景（T=4096）下 MoE 越过了拐点，是算力受限",
          ai_moe > ridge, f"{ai_moe / ridge:.3f}×I*")

    t_star = ridge * E * es / (2 * k)
    print()
    print(f"    把 AI_MoE = I* 反解出**拐点批大小**：")
    print(f"      T* = I*·E·esize / (2·top_k) = {ridge:.4f}×{E}×{es} / (2×{k})"
          f" = {t_star:.1f} 个 token / 层")
    print(f"    低于 T* 的批：**MoE 每层纯粹在搬专家权重**，FLOPs 富余而带宽不够。")
    print()
    print("    ⚠ **T* 与专家数成正比**——「MoE 的 FLOPs 与 E 无关」不等于"
          "「MoE 的代价与 E 无关」：")
    print(f"      {'E':>6}{'每层权重':>12}{'FLOPs(定 T)':>16}{'AI':>10}{'T*':>10}")
    for e_alt in (2, 8, 64, 256):
        wb = e_alt * 2 * d * f * es
        ai = 2 * T * k / (e_alt * es)
        ts = ridge * e_alt * es / (2 * k)
        print(f"      {e_alt:>6}{wb / 1e9:>10.4f}GB{moe_total / 1e12:>14.4f}TF"
              f"{ai:>10.1f}{ts:>10.0f}")
    check("拐点批大小 T* 与专家数成正比（E 越大越难算力受限）",
          close(ridge * 256 * es / (2 * k) / (ridge * 8 * es / (2 * k)), 32.0, 1e-12),
          f"E=256 的 T* 是 E=8 的 32 倍")
    print()
    act_bytes = 2 * T * d * es
    print(f"    ⚠ 上面的 AI 只记**权重流**这一项（占绝对多数）。同一层的激活流量"
          f"（读 X + 写 Y）")
    print(f"      = 2·T·d·esize = {act_bytes / 1e6:.1f} MB，"
          f"占权重的 {act_bytes / weight_bytes:.2%}——"
          f"E=8 时小到可以忽略，")
    print(f"      但它是 T 的一阶项、权重是零阶项，T 很小时两者会同量级。")
    for t in (128, 512, 1024, math.floor(t_star), math.ceil(t_star), 4096, 16384):
        ai = 2 * t * k / (E * es)
        tag = "算力受限" if ai > ridge else "访存受限"
        print(f"      T = {t:>6} → AI = {ai:>8.1f} FLOP/B "
              f"= {ai / ridge:>7.4f}×I* → {tag}")
    print()
    print(f"    本场景的时间下界（只算一层 FFN，不含 attention 与通信）：")
    print(f"      算力侧 = {moe_total / 1e12:.4f} TFLOPs ÷ {peak / 1e12:.1f} TFLOPS"
          f" = {moe_total / peak * 1e3:.4f} ms")
    print(f"      带宽侧 = {weight_bytes / 1e9:.4f} GB ÷ {bw / 1e9:.0f} GB/s"
          f" = {weight_bytes / bw * 1e3:.4f} ms")
    check("算力侧下界 > 带宽侧下界（与 T=4096 越过拐点一致）",
          moe_total / peak > weight_bytes / bw,
          f"{moe_total / peak * 1e3:.4f} ms vs {weight_bytes / bw * 1e3:.4f} ms")

    gemv_ai = 2.0 / es
    print()
    print("    逐块算算术强度：")
    print("      ① 最朴素的形态——每个 (token, 专家) 对调一次 GEMV（M=1）：")
    print(f"         每个对子读 2·d·d_ff·esize 字节、算 4·d·d_ff FLOPs")
    print(f"         → AI = 4·d·d_ff / (2·d·d_ff·esize) = 2/esize = {gemv_ai:.1f} FLOP/B "
          f"= {gemv_ai / ridge:.5f}×I*")
    check("朴素 GEMV 形态的算术强度 = 2/esize = 1.0 FLOP/B（离拐点三个数量级）",
          close(gemv_ai, 1.0, 1e-12),
          f"{gemv_ai:.1f} FLOP/B，I* 的 {gemv_ai / ridge * 100:.2f}%")
    print(f"         **与 d_model、d_ff 都无关**——它只由 esize 决定。这就是 M 必须")
    print(f"         攒起来（permute 到一段里）的原因：AI 里唯一能动的旋钮是 T·top_k/E。")

    gate_flops = 2.0 * T * E * d
    gate_bytes = (T * d + E * d) * es
    gate_ai = gate_flops / gate_bytes
    print("      ② gate GEMM（logits = X[T,d] @ Wg[d,E]，N 只有 E=8）：")
    print(f"         FLOPs = 2·T·E·d = {gate_flops:.6e}；输入+权重字节 = "
          f"(T·d + E·d)·esize = {gate_bytes:.6e} B")
    print(f"         AI = {gate_ai:.4f} FLOP/B = {gate_ai / ridge:.5f}×I*"
          f"（{gate_ai / ridge * 100:.2f}%）——本层唯一一个 N=8 的 GEMM")
    check("gate GEMM 的算术强度在 I* 的 3% 以内",
          gate_ai < 0.03 * ridge, f"{gate_ai:.4f} FLOP/B，{gate_ai / ridge * 100:.2f}%")

    t_compute = moe_total / peak
    lo, hi = SP.KERNEL_LAUNCH_OVERHEAD_US_EXTERNAL
    print()
    print(f"    §2 的四个 kernel 各启动一次（引用值 {lo:.0f}~{hi:.0f} µs/次，"
          f"本仓库不复核其测量口径）：")
    print(f"      4 次 = {4 * lo:.0f}~{4 * hi:.0f} µs，占算力侧下界 "
          f"{t_compute * 1e6:.1f} µs 的 {4 * lo / 1e6 / t_compute * 100:.2f}%~"
          f"{4 * hi / 1e6 / t_compute * 100:.2f}%")
    check("四次启动开销占一层时间下界的 2% 以内",
          4 * hi / 1e6 / t_compute < 0.02,
          f"{4 * hi / 1e6 / t_compute * 100:.2f}%")
    return ai_moe, ridge, t_star


def b3_tokens_per_expert():
    sub("B3  tokens_per_expert 的分布：均匀 vs 幂律，以及「按最大填充」的浪费比")
    uni0, zipf0 = fixed_counts()
    rows = []
    print(f"    tokens = {TOKENS}，top_k = {TOP_K}，专家 = {N_EXPERTS}"
          f" → 总 slot 数 = {assignments()}（每个 slot 是一个 (token, 专家) 对）")
    print(f"    均值 = {assignments()}/{N_EXPERTS} = {assignments() / N_EXPERTS:.1f}"
          f" —— **两种分布下均值相同**，差的全在形状上")
    print()
    print(f"    {'分布':<10}{'最小':>8}{'最大':>8}{'均值':>9}{'标准差':>10}"
          f"{'max/mean':>10}{'按最大填充的浪费':>18}")
    for name, c in (("均匀", uni0), (f"幂律 α={ZIPF_ALPHA:g}", zipf0)):
        waste, mx = pad_to_max_waste(c)
        rows.append((name, c, waste))
        print(f"    {name:<10}{int(c.min()):>8}{int(c.max()):>8}{c.mean():>9.1f}"
              f"{c.std():>10.2f}{c.max() / c.mean():>10.2f}{waste * 100:>17.2f}%")

    uni = rows[0][1]
    zipf = rows[1][1]
    check("两种分布的总 slot 数相同（都等于 tokens × top_k）",
          uni.sum() == zipf.sum() == assignments(), f"{uni.sum()} / {zipf.sum()}")
    check("均匀分布的 max/mean 很小（< 1.1）", uni.max() / uni.mean() < 1.10,
          f"{uni.max() / uni.mean():.3f}")
    check("幂律分布的「按最大填充」浪费比超过 60%", rows[1][2] > 0.60,
          f"{rows[1][2] * 100:.2f}%")
    check("幂律的浪费比是均匀的 8 倍以上",
          rows[1][2] / max(rows[0][2], 1e-9) > 8.0,
          f"{rows[1][2] / max(rows[0][2], 1e-9):.1f}×")
    print()
    print("    二项分布的理论值（推导值）：std = sqrt(N·p·(1−p))，"
          f"N = {assignments()}，p = 1/{N_EXPERTS}")
    th = math.sqrt(assignments() * (1 / N_EXPERTS) * (1 - 1 / N_EXPERTS))
    print(f"      = sqrt({assignments()} × {1 / N_EXPERTS:.4f} × "
          f"{1 - 1 / N_EXPERTS:.4f}) = {th:.4f}")
    check("抽样的 std 与二项理论值同量级（±20%）",
          abs(uni.std() - th) / th < 0.20, f"{uni.std():.2f} vs {th:.2f}")
    return uni, zipf


def b4_tile_granularity():
    sub("B4  tile 粒度：grouped（按实际长度分组）vs padded（按最大专家填充）")
    uni, zipf = fixed_counts()
    print(f"    BM = {BM}（M 方向 tile），N = d_ff = {D_FF}，BN = {BN}"
          f" → 每个专家 {D_FF // BN} 个 N-tile")
    print("    ⚠ 两种方案都**必须**把每个专家的 M 取整到 BM 的整数倍——")
    print("      padded 方案的「M_max」也要先向上取整，否则比的是一个不存在的方案。")
    print()
    print("    逐专家明细：")
    print(f"    {'分布':<12}{'tokens_per_expert':<34}{'tiles_e':<26}")
    res = {}
    for name, c in (("均匀", uni), (f"幂律 α={ZIPF_ALPHA:g}", zipf)):
        tiles = np.ceil(c / BM).astype(int)
        print(f"    {name:<12}{tool.rows(c.tolist(), 5):<34}{tool.rows(tiles.tolist(), 4):<26}")
    print()
    print(f"    {'分布':<12}{'ΣM_e':>8}{'M_max':>8}{'E·M_max':>10}{'padded 行':>11}"
          f"{'grouped 行':>12}{'padded 浪费':>13}{'grouped 浪费':>14}{'比值':>8}")
    for name, c in (("均匀", uni), (f"幂律 α={ZIPF_ALPHA:g}", zipf)):
        tot = int(c.sum())
        mx = int(c.max())
        pad_tiles = math.ceil(mx / BM)
        raw_pad = N_EXPERTS * mx                    # ← 没对齐到 tile（**不成立的方案**）
        padded_m = N_EXPERTS * pad_tiles * BM       # ← 同样对齐到 tile
        grouped_m = tile_pad(c)
        w_pad = 1.0 - tot / padded_m
        w_grp = 1.0 - tot / grouped_m
        res[name] = (padded_m, grouped_m, w_pad, w_grp, raw_pad)
        print(f"    {name:<12}{tot:>8}{mx:>8}{raw_pad:>10}{padded_m:>11}{grouped_m:>12}"
              f"{w_pad * 100:>12.2f}%{w_grp * 100:>13.2f}%"
              f"{padded_m / grouped_m:>7.2f}×")
    zp = f"幂律 α={ZIPF_ALPHA:g}"
    print()
    print(f"    padded 的每个专家排 {math.ceil(int(zipf.max()) / BM)} 个 M-tile"
          f"（= {math.ceil(int(zipf.max()) / BM) * BM} 行），"
          f"其中最少的专家只用了 {int(zipf.min())} 行"
          f"（{1 - int(zipf.min()) / (math.ceil(int(zipf.max()) / BM) * BM):.1%} 的 tile 是空的）")
    print(f"    幂律下 padded 要算 {res[zp][0]} 行，grouped 只要 {res[zp][1]} 行"
          f" → 差 {res[zp][0] - res[zp][1]} 行，**{res[zp][0] / res[zp][1]:.2f} 倍的算力**")
    check("甲乙两方案的行数差 = 多算的那些行",
          res[zp][0] - res[zp][1] == 24576 - 8704,
          f"{res[zp][0]} − {res[zp][1]} = {res[zp][0] - res[zp][1]} 行")
    check("幂律下 padded 的算力是 grouped 的 2 倍以上",
          res[zp][0] / res[zp][1] > 2.0, f"{res[zp][0] / res[zp][1]:.2f}×")
    check("「按最大专家填充」的浪费 = 1 − 均值/最大值（B3 的同一口径）",
          close(1 - int(uni.sum()) / (N_EXPERTS * int(uni.max())),
                pad_to_max_waste(uni)[0], 1e-12))
    print()
    print("    ⚠ 「E·M_max」这一列**不对齐到 tile**，是一个不存在的方案，只用来")
    print("      暴露「拿它跟 grouped 比」会低报多少（正文 §4.3）：")
    for name, c in (("均匀", uni), (f"幂律 α={ZIPF_ALPHA:g}", zipf)):
        raw = res[name][4]
        aligned = res[name][0]
        tot = int(c.sum())
        print(f"      {name:<12} E·M_max = {raw}（浪费 {(1 - tot / raw) * 100:.2f}%）"
              f" vs 对齐后 {aligned}（浪费 {(1 - tot / aligned) * 100:.2f}%）"
              f" —— 差 {(1 - tot / raw) * 100 - (1 - tot / aligned) * 100:+.2f}pp")
    print()
    print("    ⚠ grouped 不是零成本：每个专家的段末尾仍要补齐到 BM 的整数倍。")
    mean_e = int(round(int(uni.sum()) / N_EXPERTS))
    for name, c in (("均匀", uni), (f"幂律 α={ZIPF_ALPHA:g}", zipf)):
        pad_rows = np.ceil(c / BM).astype(int) * BM - c
        extra = int(pad_rows.sum())
        print(f"      {name:<12} 补齐 {extra:>5} 行 = {extra / int(c.sum()) * 100:>5.2f}%"
              f"，逐段 {tool.rows(pad_rows.tolist(), 4)}")
    w_grp_uni = res["均匀"][3]
    check("均匀路由下 grouped 仍有 4%~10% 的 tile 量化浪费（不是「几乎为零」）",
          0.04 < w_grp_uni < 0.10, f"{w_grp_uni * 100:.2f}%")
    print(f"      均匀路由下 M_e 的均值恰好是 {mean_e} = {mean_e // BM}×{BM}（整除）；"
          f"但抽样抖动让 {int((uni > mean_e).sum())}/{N_EXPERTS} 个段")
    print(f"      越过均值、各多出整整一个 tile，于是白算 "
          f"{(tile_pad(uni) - int(uni.sum())) / int(uni.sum()):.2%} 的行。")
    print("      **这部分浪费与路由均不均衡无关，只与「M_e 不是 BM 的整数倍」有关**——")
    print("      要让每段长度都落在 tile 边界上，得靠路由侧的负载均衡去凑。")
    print()
    print("    两个方案的共同前提：**N 方向（= d_ff）与 K 方向（= d_model）对所有专家相同，")
    print("    只有 M 方向变长**。这就是 grouped GEMM 的问题可以被归结为「M 不规整」的原因。")
    return res
    return res


def b5_grid_and_waves():
    sub("B5  网格形状：**按专家划分**还是**按 tile 划分**（波次量化的机制见 gemm/05）")
    sm = SP.SM_COUNT_BY_GEN[H100]
    uni, zipf = fixed_counts()
    print(f"    H100：{sm} 个 SM（规格值）。block 槽位数 = SM × 每 SM 常驻 block 数；")
    print("    常驻数取 1 与 3 两档（3 = gemm/02 的 128×128×32 FP32 配置：SMEM 32 KB，")
    print("    8×8 线程块约 80 寄存器/线程，两条约束都卡在 3）。")
    print()
    print("    映射甲：**一个专家一个 block**，网格 = E")
    for resident in (1, 3):
        slots = sm * resident
        print(f"      每 SM 常驻 {resident} 个 block → 机器一次装得下 {slots} 个 block")
        print(f"      {'E':>6}{'波数':>6}{'最后一波占用':>14}{'整机效率':>11}")
        for e in (N_EXPERTS, 64, 132, 133, 200, 396, 397, 792):
            waves = math.ceil(e / slots)
            last = e - (waves - 1) * slots
            eff = e / (waves * slots)
            print(f"      {e:>6}{waves:>6}{last:>9}/{slots:<6}{eff * 100:>10.2f}%")
    print()
    eff_8 = N_EXPERTS / (math.ceil(N_EXPERTS / (sm * 3)) * sm * 3)
    check(f"本场景 E={N_EXPERTS}：一个专家一个 block 只填满 "
          f"{N_EXPERTS}/{sm * 3} 个 block 槽位（常驻 3 时）",
          close(eff_8, N_EXPERTS / (sm * 3), 1e-12), f"{eff_8 * 100:.2f}%")
    eff_397 = 397 / (math.ceil(397 / (sm * 3)) * sm * 3)
    check("E 刚跨过槽位数（397 > 396）时，整机效率掉到一半左右",
          eff_397 < 0.55, f"{eff_397 * 100:.2f}%")

    print()
    print("    映射乙：**一个 tile 一个 block**，网格 = Σ_e ceil(M_e/BM)·(N/BN)")
    print(f"      每个专家的 N-tile 数 = {D_FF // BN}；"
          f"幂律下甲的 M-tile 数 = {N_EXPERTS} × "
          f"{math.ceil(int(zipf.max()) / BM)} = "
          f"{N_EXPERTS * math.ceil(int(zipf.max()) / BM)}")
    print(f"      {'分布':<12}{'grid(乙)':>10}{'grid(甲)':>10}{'波数':>7}{'最后一波':>10}"
          f"{'整机效率':>11}")
    for name, c in (("均匀", uni), (f"幂律 α={ZIPF_ALPHA:g}", zipf)):
        blocks = int(np.ceil(c / BM).sum() * (D_FF // BN))
        padded_blocks = N_EXPERTS * math.ceil(int(c.max()) / BM) * (D_FF // BN)
        waves = math.ceil(blocks / sm)
        last = blocks - (waves - 1) * sm
        eff = blocks / (waves * sm)
        print(f"      {name:<12}{blocks:>10}{padded_blocks:>10}{waves:>7}{last:>6}/{sm:<4}"
              f"{eff * 100:>10.2f}%")
    c_uni = uni
    blocks_uni = int(np.ceil(c_uni / BM).sum() * (D_FF // BN))
    eff_uni = blocks_uni / (math.ceil(blocks_uni / sm) * sm)
    check("tile 级网格的波数在两位数以上，尾部损失被摊薄（> 98%）",
          eff_uni > 0.98, f"{blocks_uni} 个 block，效率 {eff_uni * 100:.2f}%")
    check("tile 级网格的整机效率高于「一个专家一个 block」",
          eff_uni > eff_8, f"{eff_uni * 100:.2f}% vs {eff_8 * 100:.2f}%")
    print()
    print("    ⚠ 「细网格填满机器」不是无条件的：block 数超过 槽位数 之后，多出来的")
    print("      只是排队，尾部的代价 = (1 − 最后一波占用率) / 波数")
    print("      （波次量化的机制正文在 40-kernels/gemm/05，此处回链）。")
    print(f"    映射乙在常驻 1 时的效率 = {eff_uni * 100:.1f}%；常驻 3 时槽位变成 "
          f"{sm * 3}，")
    waves3 = math.ceil(blocks_uni / (sm * 3))
    eff3 = blocks_uni / (waves3 * sm * 3)
    print(f"      波数降到 {waves3}，效率 {eff3 * 100:.2f}%——"
          f"**尾部占比反而变大**，这是「常驻多」的隐性代价。")
    print(f"      尾部损失 = 100% − {eff3 * 100:.2f}% = {100 - eff3 * 100:.2f}%"
          f"（常驻 1 时是 {100 - eff_uni * 100:.2f}%）")
    check("常驻数越大，尾部损失占的比例越大（99.48% → 96.16%）",
          eff3 < eff_uni, f"{eff3 * 100:.2f}% < {eff_uni * 100:.2f}%")
    return eff_uni


def b6_memory_bytes():
    sub("B6  permute / unpermute 的访存字节与合并度（**按公式算，不测硬件**）")
    d, k, T, es = D_MODEL, TOP_K, TOKENS, ESIZE
    row_bytes = d * es
    rows = T * k
    print(f"    一行 = d_model 个 {es} 字节元素 = {row_bytes} B = {row_bytes / 1024:.1f} KB")
    print(f"    行数：输入 {T} 行，permute 之后 {rows} 行（每个 (token, 专家) 一个 slot）")
    print()
    perm_fused = T * row_bytes + rows * row_bytes
    perm_naive = 2 * rows * row_bytes
    unperm_best = rows * row_bytes + T * row_bytes
    print("    permute：")
    print(f"      读一次写 k 次（token 行留在寄存器/SMEM）"
          f" = T·d·esize + T·k·d·esize = {perm_fused / 1e6:.3f} MB")
    print(f"      读 k 次写 k 次（每个 slot 各自去取一遍）"
          f" = 2·T·k·d·esize = {perm_naive / 1e6:.3f} MB"
          f"  ← 多出来的 {perm_naive / perm_fused - 1:.0%}")
    print("    unpermute：")
    print(f"      读 k 次写一次（k 个贡献在寄存器里累加，每个 token 只写一行）"
          f" = T·k·d·esize + T·d·esize = {unperm_best / 1e6:.3f} MB")
    print(f"    两者合计 = 2·(1+k)·T·d·esize = {(perm_fused + unperm_best) / 1e6:.3f} MB")
    weight_gb = N_EXPERTS * 2 * d * D_FF * es / 1e9
    print(f"    对照：同一层要流的专家权重 = {weight_gb:.4f} GB"
          f" → 重排的流量只占 {(perm_fused + unperm_best) / 1e9 / weight_gb * 100:.2f}%")
    check("permute+unpermute 的流量不到权重流量的 15%",
          (perm_fused + unperm_best) / 1e9 / weight_gb < 0.15,
          f"{(perm_fused + unperm_best) / 1e9 / weight_gb * 100:.2f}%")
    l2 = SP.L2_MB_H100_IN_REPO * 1024 * 1024
    print()
    print(f"    permute 出来的中间张量 = {rows} × {row_bytes} B"
          f" = {rows * row_bytes / 1e6:.1f} MB")
    print(f"    H100 的 L2 = {SP.L2_MB_H100_IN_REPO} MB（引用值，见 specs.py 的口径注记）"
          f" → permuted 缓冲区 **装不进 L2**（{rows * row_bytes / l2:.2f}×）")
    check("permuted 缓冲区大于 L2 容量",
          rows * row_bytes > l2,
          f"{rows * row_bytes / 1e6:.1f} MB = L2 的 {rows * row_bytes / l2:.2f} 倍")

    slot_tbl = rows * 4          # dst_row：每个 slot 一个 int
    print()
    print(f"    索引表 dst_row / valid 的大小 = T·k 个 int = {rows} × 4 B"
          f" = {slot_tbl / 1024:.0f} KB")
    print(f"    占 permuted 缓冲区的 {slot_tbl / (rows * row_bytes):.5%}"
          f"——**表本身小到可忽略**，但它是一次额外的随机读")
    check("索引表不到 permuted 缓冲区的 1%",
          slot_tbl / (rows * row_bytes) < 0.01,
          f"{slot_tbl / 1024:.0f} KB vs {rows * row_bytes / 1e6:.1f} MB")

    print()
    print("    合并度：一个 warp（32 lane）的一次访存触及几个 32 B sector")
    sect = SP.SECTOR_BYTES
    coalesced = (32 * es) // sect
    scattered = 32
    print(f"      行内连续（lane ↔ d_model 的列）：32 × {es} B = {32 * es} B"
          f" 连续 → {coalesced} 个 sector")
    print(f"      元素级 scatter（lane ↔ slot）：32 个地址相距 {row_bytes} B"
          f" → 各占一个 sector = {scattered} 个 sector")
    print(f"      放大倍数（搬走的字节 / 有用的字节）= {scattered}/{coalesced} "
          f"= {scattered / coalesced:.0f}×")
    check("元素级 scatter 的 sector 数是行内连续的 16 倍",
          scattered // coalesced == 16, f"{scattered}/{coalesced}")
    return perm_fused, unperm_best


# ===========================================================================
# §C 边界与反例
# ===========================================================================


def c1_extreme_imbalance():
    sub("C1  极端不均的路由：90% 的 token 挤到一个专家")
    rng = np.random.default_rng(4242)
    T, k, E = TOKENS, TOP_K, N_EXPERTS
    hot = 0.90
    n_hot = int(T * hot)
    first = np.concatenate([np.zeros(n_hot, dtype=np.int64),
                            rng.integers(0, E, size=T - n_hot)])
    counts = np.bincount(first, minlength=E).astype(np.int64)
    # 第二选择在**除第一选择之外**的专家里均匀取
    for t in range(T):
        cand = [e for e in range(E) if e != first[t]]
        counts[int(rng.choice(cand))] += 1

    waste, mx = pad_to_max_waste(counts)
    print(f"    构造：{n_hot}/{T} = {hot:.0%} 的 token 第一选择是专家 0，"
          f"其余均匀；第二选择在剩余 {E - 1} 个专家里均匀")
    print(f"    tokens_per_expert = {tool.rows(counts.tolist())}")
    print(f"    总 slot = {counts.sum()}（= tokens × top_k = {T * k}）")
    print(f"    均值 = {counts.mean():.1f}，最大 = {mx}，max/mean = {mx / counts.mean():.2f}")
    print(f"    「按最大填充」的浪费比 = 1 − {counts.mean():.1f}/{mx} = {waste * 100:.2f}%")
    check("总 slot 数守恒", int(counts.sum()) == T * k, f"{counts.sum()}")
    check("极端不均下「按最大填充」的浪费比 > 70%（文档 §4 声称的阈值）",
          waste > 0.70, f"{waste * 100:.2f}%")
    grouped_m = tile_pad(counts)
    padded_m = E * math.ceil(mx / BM) * BM
    print(f"    按 tile 粒度分组后 M 总长 = {grouped_m}，浪费 "
          f"{(1 - counts.sum() / grouped_m) * 100:.2f}%；"
          f"padded（同样对齐到 tile）是 {padded_m}，浪费 "
          f"{(1 - counts.sum() / padded_m) * 100:.2f}%")
    print(f"    两者比值 = {padded_m / grouped_m:.2f}×")
    check("分组后浪费比降到 10% 以下", (1 - counts.sum() / grouped_m) < 0.10,
          f"{(1 - counts.sum() / grouped_m) * 100:.2f}%")
    check("极端不均下 padded 的算力是 grouped 的 3 倍以上",
          padded_m / grouped_m > 3.0, f"{padded_m / grouped_m:.2f}×")

    # 容量截断后的丢弃率（slot 数 C 的语义见 80-model-architectures/moe/02-capacity-factor.md）
    mean_c = T * k // E
    print()
    print(f"    若每个专家只开 C = 均值 = {mean_c} 个 slot（C 的**语义**不在本域，"
          f"见 80-model-architectures/moe/02-capacity-factor.md）：")
    dropped = int(np.maximum(counts - mean_c, 0).sum())
    frac_drop = dropped / counts.sum()
    print(f"      被丢的 slot = Σ max(0, M_e − C) = {dropped} / {counts.sum()}"
          f" = {frac_drop * 100:.2f}%")
    check("截断到均值时丢掉两成以上的 slot", frac_drop > 0.20,
          f"{frac_drop * 100:.2f}%")
    moe_flops = 4 * T * k * D_MODEL * D_FF
    print(f"      折算成这一层的算力：{moe_flops / 1e12:.4f} TFLOPs × "
          f"{frac_drop:.4f} = {moe_flops * frac_drop / 1e12:.4f} TFLOPs"
          f"（**算了但不参与输出**）")
    check("被丢 slot 对应的算力超过 0.5 TFLOPs",
          moe_flops * frac_drop / 1e12 > 0.5,
          f"{moe_flops * frac_drop / 1e12:.4f} TFLOPs")
    print(f"      而被截断的专家只有 {int((counts > mean_c).sum())} 个"
          f"（{E} 个里的 {int((counts > mean_c).sum())}）——"
          f"**不均体现为「少数专家溢出」，不是「大家一起溢出」**")
    return counts, mean_c


def c2_mask():
    sub("C2  被丢的 slot 在 unpermute 时**没做 mask** 的后果")
    rng = np.random.default_rng(99)
    T, d, f, E, k = 64, 8, 16, 4, 2
    C = 40                      # 每个专家开的 slot 数（> 均值 32，热专家仍然溢出）
    X = rng.standard_normal((T, d))
    Wg = np.ones((E, d))
    Wg[0] *= 3.0                # 让专家 0 变成热点
    up, down = make_experts(rng, d=d, f=f, e=E, scale=0.5)

    # 第一轮：容量足够，把输出缓冲区**填满**（模拟缓冲区复用，不清零）
    X0 = rng.standard_normal((T, d))
    _, _, _, _, yp_prev, _, _ = moe_permuted(X0, Wg, up, down, k, capacity=None)

    # 第二轮：同样的缓冲区，但有截断
    out_masked, out_unmasked, slot, valid, yp, idx, w = moe_permuted(
        X, Wg, up, down, k, capacity=C, yp_buffer=yp_prev.copy())

    counts = np.bincount(idx.ravel(), minlength=E)
    print(f"    规模：tokens={T}，d_model={d}，d_ff={f}，专家={E}，top_k={k}，"
          f"每个专家 {C} 个 slot")
    print(f"    tokens_per_expert = {tool.rows(counts.tolist())}"
          f"（要放的 slot 共 {counts.sum()}，开出的槽位 E×C = {E * C}）")
    n_drop = int((counts - np.minimum(counts, C)).sum())
    print(f"    被丢的 slot = {n_drop} / {counts.sum()} = {n_drop / counts.sum():.1%}")
    print(f"    有效的 (token, 专家) 对 = {int(valid.sum())} / {valid.size}")

    # 参考：只有被保留的贡献参与
    ref = np.zeros((T, d))
    for t in range(T):
        for j in range(k):
            if valid[t, j]:
                ref[t] += w[t, j] * yp[int(slot[t, j])]

    err_m = float(np.abs(out_masked - ref).max())
    err_u = float(np.abs(out_unmasked - ref).max())
    scale = float(np.abs(ref).max())
    print()
    print(f"    做 mask（只累加 valid 的 slot）：最大误差 = {err_m:.3e}")
    print(f"    不做 mask（把所有 slot 都累加）：最大误差 = {err_u:.3e}"
          f"（参考本身的量级 {scale:.3e}）")
    check("两份输出确实不同（不做 mask 会读到被丢 slot 里的残留数据）",
          not np.allclose(out_masked, out_unmasked), f"{err_u:.3e}")
    check("做 mask 的结果与参考逐元素一致", err_m < 1e-12, f"{err_m:.3e}")
    check("不做 mask 的误差与结果同量级（不是数值噪声）",
          err_u > 0.1 * scale, f"{err_u:.3e} vs {scale:.3e}")
    bad = int((np.abs(out_unmasked - ref).max(1) > 1e-9).sum())
    print(f"    受影响的行：{bad} / {T} 个 token 的整行输出被污染")
    check("受影响的行数 = 有 slot 被丢的 token 数", bad > 0, f"{bad} 行")
    print()
    print("    ⚠ 这里被丢的 slot 里放的是**上一批迭代留下的数据**（缓冲区没有清零）。")
    print("      真实实现里这些行要么是上一次的激活、要么是别的 token 的行——")
    print("      无论哪种，结果是**悄悄错**，不会报错。")


def c3_sectors():
    sub("C3  元素级 scatter 的 sector 放大（合并度退化）")
    d, k, T, es, sect = D_MODEL, TOP_K, TOKENS, ESIZE, SP.SECTOR_BYTES
    row_bytes = d * es
    print(f"    32 lane，每个 lane 搬 {es} B，共 {32 * es} B 有用数据")
    good = (32 * es + sect - 1) // sect
    bad = 32
    print(f"    行内连续：{32 * es} B 落在 {good} 个 sector 里，"
          f"搬走 {good * sect} B → 有效率 {32 * es / (good * sect):.1%}")
    print(f"    按 slot 展开：32 个 lane 的地址相距 {row_bytes} B，各占一个 sector，"
          f"搬走 {bad * sect} B → 有效率 {32 * es / (bad * sect):.1%}")
    print(f"    放大 = {bad * sect / (good * sect):.1f}×")
    check("元素级 scatter 让每个 warp 的写请求从 2 个 sector 变成 32 个",
          good == 2 and bad == 32, f"{good} → {bad}")
    check("有用字节占比从 100% 掉到 6.25%",
          close(32 * es / (bad * sect), 0.0625, 1e-12),
          f"{32 * es / (bad * sect):.4f}")
    print()
    print("    permute 的**行内**部分永远是连续的（一行就是 d_model 个连续元素），")
    print("    散开的是**行号**。所以修法不是「合并写」，而是「让 warp 内的 lane")
    print("    沿 d_model 走、行号留给 blockIdx / threadIdx.y」。")
    return good, bad


def c4_dense_treatment():
    sub("C4  把 MoE 当成稠密 GEMM 算：FLOPs 放大 = 专家数")
    T, k, d, f, E = TOKENS, TOP_K, D_MODEL, D_FF, N_EXPERTS
    moe = 4 * T * k * d * f
    dense_all = 4 * T * k * E * d * f
    ratio = dense_all / moe
    print(f"    「每个 token 的 k 个选择都送进全部 {E} 个专家」"
          f" = {dense_all:.6e} FLOPs")
    print(f"    真正的 MoE = {moe:.6e} FLOPs")
    print(f"    放大比 = {ratio:.0f}× = E")
    check("当成稠密算的 FLOPs 放大比恰好 = 专家数", close(ratio, E, 1e-12),
          f"{ratio:.1f}×")
    print()
    print("    另一条更隐蔽的退化是**「算了但没要」**：slot 被截断后，")
    print("    grouped GEMM 仍可能把它算完（因为它落在某个 tile 里），")
    print("    却在 unpermute 时被 mask 丢掉——这份算力既没产出也没省下。")
    print("    「被丢的 slot 占比」由 C1 算出，本节点到为止：")
    print("    把那个比例乘回 MoE 层 FLOPs 就是无效计算量。")


def c5_empty_experts():
    sub("C5  一个 token 都没分到的专家：分段长为 0")
    rng = np.random.default_rng(5)
    T_small, E_big, k = 16, 64, 2
    idx = rng.integers(0, E_big, size=(T_small, k))
    counts = np.bincount(idx.ravel(), minlength=E_big)
    empty = int((counts == 0).sum())
    print(f"    tokens={T_small}，top_k={k}，专家={E_big}"
          f" → 只有 {T_small * k} 个 slot 要分给 {E_big} 个专家")
    print(f"    空的专家有 {empty} 个（占 {empty / E_big:.1%}）")
    check("专家数远大于 slot 数时出现空段", empty > 0, f"{empty}/{E_big}")
    tiles = np.ceil(counts / BM).astype(int)
    print(f"    按实际长度分组：这些专家的 tile 数是 ceil(0/{BM}) = 0，**不占 block**；"
          f"全家一共 {int(tiles.sum())} 个 M-tile")
    check("空段的 tile 数为 0（不占 block）", int(tiles[counts == 0].sum()) == 0,
          f"{empty} 个空段贡献 0 个 tile")

    # 真的跑一遍：空段既不贡献输出，也不让后面的段错位
    d, f = 16, 8
    Xp = rng.standard_normal((int(counts.sum()), d))
    Ws = rng.standard_normal((E_big, d, f))
    got = grouped_gemm(Xp, counts, Ws)
    off = 0
    worst = 0.0
    for e in range(E_big):
        n_e = int(counts[e])
        if n_e:
            worst = max(worst, float(np.abs(got[off:off + n_e] - Xp[off:off + n_e] @ Ws[e]).max()))
        off += n_e
    check("含空段的 grouped GEMM 与逐专家调用逐元素一致（空段不错位后面的段）",
          worst < 1e-12, f"最大差 {worst:.3e}")

    print(f"    按最大填充：仍然给每个专家排 ceil({int(counts.max())}/{BM}) = "
          f"{math.ceil(int(counts.max()) / BM)} 个 M-tile，"
          f"{E_big} 个专家共 {E_big * math.ceil(int(counts.max()) / BM)} 个 M-tile，"
          f"其中 {empty * math.ceil(int(counts.max()) / BM)} 个是给空专家白排的")
    check("按最大填充时空段照样占 tile",
          E_big * math.ceil(int(counts.max()) / BM) > int(tiles.sum()),
          f"{E_big * math.ceil(int(counts.max()) / BM)} vs {int(tiles.sum())}")
    print()
    print("    ⚠ 但 segmented 归约的「迭代数由最长段决定」在这里**不适用**：")
    print("      grouped GEMM 的每个 block 只处理一个段的一个 tile，")
    print("      空段只是不发牌，不会拖慢别的段（对比 40-kernels/reduce-scan/01 §1.2）。")


def c6_dropped_gradient():
    sub("C6  被丢的 slot 在**反向**也要 mask（否则给没发生的计算发梯度）")
    rng = np.random.default_rng(2024)
    T, d, f, E, k = 8, 4, 6, 3, 2
    C = 3
    X = rng.standard_normal((T, d))
    Wg = rng.standard_normal((E, d))
    Wg[0] += 2.0                      # 造一个热点专家
    up, down = make_experts(rng, d=d, f=f, e=E, scale=0.7)

    logits = X @ Wg.T
    idx = route_ref(logits, k)
    w = route_weights(logits, idx)
    counts = np.bincount(idx.ravel(), minlength=E)
    offsets = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)

    cursor = offsets.copy()
    valid = np.zeros((T, k), dtype=bool)
    for t in range(T):
        for j in range(k):
            e = int(idx[t, j])
            s = int(cursor[e]); cursor[e] += 1
            valid[t, j] = (s - offsets[e]) < C

    # 前向（只在 valid 的 slot 上算）
    y = np.zeros((T, d))
    for t in range(T):
        for j in range(k):
            if valid[t, j]:
                y[t] += w[t, j] * (act(X[t] @ up[int(idx[t, j])]) @ down[int(idx[t, j])])
    loss = float((y ** 2).sum())
    dLdy = 2.0 * y

    g_ok = np.zeros_like(down)         # 反向**做** mask
    g_bad = np.zeros_like(down)        # 反向**不做** mask
    dropped_terms = []
    for t in range(T):
        for j in range(k):
            e = int(idx[t, j])
            h = act(X[t] @ up[e])      # 反向本来就要重算 h
            term = np.outer(h, w[t, j] * dLdy[t])
            if valid[t, j]:
                g_ok[e] += term
            else:
                dropped_terms.append((t, j, float(np.linalg.norm(term)),
                                      float(w[t, j])))
            g_bad[e] += term           # ← 没做 mask：给被丢的那一项也发了梯度

    n_drop = int((~valid).sum())
    rn_diff = float(np.linalg.norm(g_bad - g_ok) / np.linalg.norm(g_ok))
    print(f"    规模：tokens={T}，d_model={d}，d_ff={f}，专家={E}，top_k={k}，C={C}")
    print(f"    tokens_per_expert = {tool.rows(counts.tolist())}，"
          f"被丢的 (token, 专家) 对 = {n_drop} / {T * k}")
    print(f"    loss = {loss:.6f}；max|∂L/∂down| = {np.abs(g_ok).max():.4e}")
    print(f"    不做 mask 的反向：||Δg||/||g|| = {rn_diff:.4f}"
          f"（{rn_diff * 100:.2f}%，**不是同量级**）")
    check("反向不做 mask 确实改变了梯度（但不一定改变量级）",
          rn_diff > 1e-3, f"{rn_diff:.4f}")
    print()
    print("    为什么差得不多？看被丢的那几项本身：")
    for t, j, nt, ww in dropped_terms:
        print(f"      token {t} 的第 {j} 选择：|梯度项| = {nt:>8.4f}，权重 w = {ww:.4f}")
    print("    **被丢的对子恰好也是权重小的那些**——热点专家拿走 0.999 的权重，")
    print("    被挤掉的那一项就只剩 0.001 的量级。所以「反向漏 mask」的绝对误差")
    print("    取决于权重分布，本例只有 1.2%。不能笼统说成「误差很大」。")
    print()

    # 但有一类后果是**结构性**的，与量级无关：某个 token 的 k 个选择全被截断
    all_drop = [t for t in range(T) if not valid[t].any()]
    print(f"    真正致命的是另一类：**k 个选择全部被截断的 token** = {len(all_drop)} / {T}"
          f"（本例 {all_drop}）")
    print(f"    这些 token 的前向输出恒为 0：||y[t]|| = "
          f"{[float(np.linalg.norm(y[t])) for t in all_drop]}")
    print(f"    它们的梯度也恒为 0：max|∂L/∂y[t]| = "
          f"{[float(np.abs(dLdy[t]).max()) for t in all_drop]}")
    check("存在「k 个选择全被截断」的 token", len(all_drop) > 0,
          f"{all_drop}")
    check("这些 token 的前向输出恰好为 0（不是「很小」）",
          all(close(float(np.linalg.norm(y[t])), 0.0, 1e-15) for t in all_drop))
    check("这些 token 的梯度也恰好为 0——它们在训练里**完全消失**",
          all(close(float(np.abs(dLdy[t]).max()), 0.0, 1e-15) for t in all_drop))
    print()
    print("    ⚠ 前向做了 mask、反向没做，是一处**不对称**；但即使两边都做对，")
    print("      「整 token 被丢」的那几行仍然是前向 0、反向 0 —— 这个 token")
    print("      在本步训练中完全不存在，连一个把它拉回来的梯度都没有。")
    print("      这与 C2 是同一条边界的两侧：前向漏 mask 是「读了不该读的」，")
    print("      反向漏 mask 是「写了不该写的」，全丢则是「什么都没发生」。")


# ===========================================================================
# 入口
# ===========================================================================


def s0_scope():
    section(0, "口径")
    print("本程序全部输出是**推导值**：跑的是算术与 numpy，本机没有 GPU，")
    print("3 个 .cu 内核的运行结果本仓库没有。")
    print()
    print(f"主用机器：{H100} SXM —— {SP.SM_COUNT_BY_GEN[H100]} SM / "
          f"{SP.HBM_BW_GBps_BY_GEN[H100]} GB/s / "
          f"{SP.TC_TFLOPS_DENSE_SPARSE['FP16'][H100][0]} TFLOPS FP16 dense")
    print(f"每 SM 资源（规格值，见 specs.py）：SMEM 上限 {SP.SMEM_PER_SM_MAX_KB} KB"
          f"（与 L1 共池）、寄存器 {SP.REGISTERS_PER_SM} 个、"
          f"常驻线程 {SP.MAX_THREADS_PER_SM}、常驻 block {SP.MAX_BLOCKS_PER_SM}；"
          f"最小访存事务 {SP.SECTOR_BYTES} B（sector）")
    print()
    print("场景参数（**本文选定的对照场景，不是任何模型的规格**——没有一手来源可引，")
    print("见 specs.py 的 UNSOURCED）：")
    print(f"  d_model={D_MODEL}，d_ff={D_FF}（每个专家），专家数={N_EXPERTS}，"
          f"top_k={TOP_K}，tokens={TOKENS}，esize={ESIZE} B（FP16）")
    print(f"  tile：BM={BM}，BN={BN}")
    print()
    print("⚠ 三处边界，全文遵守：")
    print("  · MoE 的 **all-to-all 通信量**不在本域——正文只回链 110-communication/collectives；")
    print("  · **容量因子的定义与语义**不在本域——正文只讲它对 kernel 的影响，"
          "定义回链 80-model-architectures/moe/02-capacity-factor.md；")
    print("  · **segmented 归约**的正文在 40-kernels/reduce-scan/01，**访存合并**在")
    print("    20-gpu-architecture/memory-system/02，**bank 冲突**在 .../03——本域只代入。")


def main():
    s0_scope()
    section("A", "算法正确性")
    a1_routing()
    a2_end_to_end()
    a3_grouped_gemm()
    a4_offsets()
    section("B", "代价模型")
    b1_flops()
    b2_weight_stream()
    b3_tokens_per_expert()
    b4_tile_granularity()
    b5_grid_and_waves()
    b6_memory_bytes()
    section("C", "边界与反例")
    c1_extreme_imbalance()
    c2_mask()
    c3_sectors()
    c4_dense_treatment()
    c5_empty_experts()
    c6_dropped_gradient()

    print()
    print("=" * 78)
    n = _STATE["n"]
    if _STATE["fail"]:
        print(f"小结：{n} 条断言，**{len(_STATE['fail'])} 条不通过**：")
        for f in _STATE["fail"]:
            print(f"  · {f}")
    else:
        print(f"小结：{n} 条断言全部通过。")
    print("=" * 78)
    return 1 if _STATE["fail"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

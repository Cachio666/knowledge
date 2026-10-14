#!/usr/bin/env python3
"""130-model-optimization/overview 编号程序 01 —— 六族优化各自动的是哪个自由度（推导，不是实测）。

配对文档：`130-model-optimization/overview/01-optimization-taxonomy.md`

★ 形态（CONTRIBUTING §2 判定表第二种）：本程序**在默认 `python3` + `numpy` 下可跑**，
  三类断言**全部在本程序内部**，失败时以非零码退出 ⇒ **不需要 `verify.py`**。
  产出**一律是推导值**——只跑算术与 numpy，不碰硬件、不加载权重、不 import 任何框架。

三类断言：
  A 算法正确性 —— 四条**独立路径**的复核：
       ① 参数量：显式矩阵清单（数元素）vs 闭式公式；
       ② 权重字节：逐张量字节求和 vs `Ψ × 位宽/8`；
       ③ **跨域一致性**：用 KV 的六项乘积独立复算「每 token 512 KiB」，
          与 `specs.KV_BYTES_PER_TOKEN_FP16`（取自 `18-kv-cache-basics`）对拍；
       ④ 每 token FLOPs：逐矩阵 `2mn` 求和 vs `2 × 每 token 用到的参数`。
  B 代价模型   —— decode 一步搬多少字节、权重压 4× 换来几倍；训练显存里权重占几成。
  C 边界与反例 —— 文档声称**会出问题**的地方必须真的出问题：
       ① 「权重压 4× ⇒ 端到端至少 4×」在任何 `S > 0` 上都不成立；
       ② 上下文越长加速比越小，且**下界是 1**（趋于 1，不是趋于 0 或负）；
       ③ 算力侧同构的墙：把「int8 算力 2×」当成「prefill 快 2×」，只要矩阵乘
          不占满 100%，这条断言就**失败**（Amdahl 在算力侧的同一个形态）；
       ④ 「int4 权重 ⇒ 训练显存减半」被证伪。

★ 本程序**不复制 `specs.py` 的任何常量值**：凡用到规格常量，一律写 `SPECS.xxx`。

用法：cd code && make 01_taxonomy   （或 make run 全跑）
"""

import sys

import numpy as np

import specs as SPECS

FAILURES = []

# 示例假设：prefill 里「矩阵乘」所占的时间份额。**本文构造的假设**，
# 不是任何实测值——它只用来演示 Amdahl 的形式，结论对任何 f < 1 都成立。
MATMUL_TIME_FRACTION_EXAMPLE = 0.85
TENSOR_CORE_RATE_INT8_VS_FP16 = 2.0
# 来源：[A] 低精度算力倍率（int8 对 fp16 的峰值吞吐比）在厂商规格里是常见档位；
#   本程序**不引用任何具体 GPU 的数值**，只把它当作一个**示例倍率**，
#   结论是结构性的：`f < 1 ⇒ 加速 < 倍率`。


def judge(label, ok, detail):
    """统一的断言出口。ok=False 时记账，最终以非零码退出。"""
    mark = "通过" if ok else "**失败**"
    print(f"  [{mark}] {label}")
    print(f"          {detail}")
    if not ok:
        FAILURES.append(label)


# ---------------------------------------------------------------------------
# 被测路径：从配置推出四本账
# ---------------------------------------------------------------------------

def weight_matrices(cfg):
    """一层的矩阵清单 + embedding + LM head。**显式列出每张矩阵的 (out, in)**。

    这是 A 类里「数元素」那条路径的唯一来源——它**不经过**任何闭式公式。
    """
    d = cfg["d_model"]
    d_ff = cfg["d_ff"]
    heads = cfg["n_heads"]
    head_dim = cfg["head_dim"]
    kv_heads = cfg["n_kv_heads"]
    vocab = cfg["vocab"]
    per_layer = [
        ("Wq", heads * head_dim, d),
        ("Wk", kv_heads * head_dim, d),
        ("Wv", kv_heads * head_dim, d),
        ("Wo", d, heads * head_dim),
        ("W_gate", d_ff, d),
        ("W_up", d_ff, d),
        ("W_down", d, d_ff),
    ]
    return per_layer, ("embedding", vocab, d), ("lm_head", vocab, d)


def per_layer_params(cfg):
    per_layer, _emb, _lm = weight_matrices(cfg)
    return sum(int(np.prod(shape, dtype=np.int64)) for _n, *shape in per_layer)


def param_count_reference(cfg):
    """独立参照：把每张矩阵的元素个数数出来（`prod(out, in)`）再求和。"""
    _per_layer, emb, lm = weight_matrices(cfg)
    return (cfg["layers"] * per_layer_params(cfg)
            + int(np.prod(emb[1:], dtype=np.int64))
            + int(np.prod(lm[1:], dtype=np.int64)))


def param_count_formula(cfg):
    """闭式：`L·(4·d² + 3·d·d_ff) + 2·vocab·d`。**被测路径**。"""
    d, d_ff = cfg["d_model"], cfg["d_ff"]
    return cfg["layers"] * (4 * d * d + 3 * d * d_ff) + 2 * cfg["vocab"] * d


def params_used_per_token(cfg):
    """每生成一个 token 真正参与矩阵乘的参数：全部层 + LM head，**不含 embedding 查表**。"""
    _per_layer, _emb, lm = weight_matrices(cfg)
    return cfg["layers"] * per_layer_params(cfg) + int(np.prod(lm[1:], dtype=np.int64))


def weights_bytes(n_params, bits):
    """权重字节 = 参数个数 × 位宽 / 8。**纯载荷**口径（不含 scale/zero）。"""
    return n_params * bits / 8.0


def kv_bytes_per_token(bits):
    """每 token 的 KV 字节，按位宽等比缩放。基准值取自 `specs`（本库别处的正文）。"""
    return SPECS.KV_BYTES_PER_TOKEN_FP16 * bits / 16.0


def decode_bytes_per_step(cfg, w_bits, seq, batch, kv_bits):
    """decode 一步搬的字节 = 读一遍全部权重 + 读一遍全部 KV cache。

    ★ decode 是 **memory-bound**：约束是**字节**，不是算力。
    """
    return weights_bytes(params_used_per_token(cfg), w_bits) + batch * seq * kv_bytes_per_token(kv_bits)


def decode_speedup(cfg, seq, batch, w_bits_a, w_bits_b, kv_bits):
    """两个权重位宽之间的字节比。★ 带宽在比值里**约掉**，所以结论与 GPU 型号无关。"""
    return (decode_bytes_per_step(cfg, w_bits_a, seq, batch, kv_bits)
            / decode_bytes_per_step(cfg, w_bits_b, seq, batch, kv_bits))


def prefill_flops_per_token(cfg):
    """prefill 每 token 的 FLOPs = `2 × 每 token 用到的参数`。**与位宽无关**。"""
    return 2 * params_used_per_token(cfg)


def amdahl(rate, fraction):
    """`T' / T = (1 − f) + f / rate` 的倒数，即加速比 `1 / ((1 − f) + f/rate)`。"""
    return 1.0 / ((1.0 - fraction) + fraction / rate)


# ---------------------------------------------------------------------------
# A 算法正确性
# ---------------------------------------------------------------------------

def class_a():
    print("\nA 算法正确性（独立参照 vs 被测路径）")
    cfg = SPECS.EXAMPLE_MODEL
    per_layer, emb, lm = weight_matrices(cfg)

    # A.1 参数量：数元素 vs 闭式
    ref_total = param_count_reference(cfg)
    formula_total = param_count_formula(cfg)
    judge(
        "A.1 参数量：数元素 == 闭式",
        ref_total == formula_total,
        f"数元素 {ref_total/1e9:.6f}B == 闭式 {formula_total/1e9:.6f}B"
        f"（每层 {per_layer_params(cfg)/1e6:.3f}M / embedding {emb[1]*emb[2]/1e6:.3f}M"
        f" / LM head {lm[1]*lm[2]/1e6:.3f}M）",
    )

    # A.2 权重字节：逐张量字节求和 vs Ψ × 位宽/8（两条路径分开写）
    bits = 16
    by_tensor = 0.0
    for _n, out, inn in per_layer:
        by_tensor += cfg["layers"] * (out * inn * bits / 8.0)
    for _n, out, inn in (emb, lm):
        by_tensor += out * inn * bits / 8.0
    closed = weights_bytes(ref_total, bits)
    judge(
        "A.2 权重字节：逐张量求和 == Ψ×位宽/8",
        abs(by_tensor - closed) < 1e-6,
        f"逐张量 {by_tensor/2**30:.6f} GiB == 闭式 {closed/2**30:.6f} GiB"
        f"（fp16 总参数 {ref_total/1e9:.3f}B）",
    )

    # A.3 跨域一致性：KV 的六项乘积复算 == 18-kv-cache-basics 的值
    kv_recomputed = 2 * cfg["layers"] * cfg["n_kv_heads"] * cfg["head_dim"] * 2
    judge(
        "A.3 KV 每 token 字节：独立复算 == specs 的消费值（跨域一致性）",
        kv_recomputed == SPECS.KV_BYTES_PER_TOKEN_FP16,
        f"2×{cfg['layers']}×{cfg['n_kv_heads']}×{cfg['head_dim']}×2 = {kv_recomputed} B"
        f" == specs {SPECS.KV_BYTES_PER_TOKEN_FP16} B（{kv_recomputed/1024:.0f} KiB）",
    )

    # A.4 每 token FLOPs：逐矩阵 2mn 求和 vs 2×每 token 参数
    flops_ref = sum(cfg["layers"] * 2 * out * inn for _n, out, inn in per_layer)
    flops_ref += 2 * lm[1] * lm[2]
    judge(
        "A.4 prefill 每 token FLOPs：逐矩阵 2mn == 2×每 token 参数",
        flops_ref == prefill_flops_per_token(cfg),
        f"逐矩阵 {flops_ref/1e9:.3f} GFLOP == 2×{params_used_per_token(cfg)/1e9:.3f}B",
    )
    return ref_total


# ---------------------------------------------------------------------------
# B 代价模型
# ---------------------------------------------------------------------------

def class_b(n_params):
    print("\nB 代价模型（decode 的字节账 / 训练显存里权重的占比）")
    cfg = SPECS.EXAMPLE_MODEL
    # ★ 两个 Ψ 不同，必须分开：
    #   · decode 每步读的是「参与矩阵乘的权重」⇒ 不含 embedding 查表（`params_used_per_token`）；
    #   · 训练态的 16Ψ 是**全部**参数（含 embedding），因为梯度与优化器状态对它也算。
    psi_decode = params_used_per_token(cfg)
    w16 = weights_bytes(psi_decode, 16)
    w4 = weights_bytes(psi_decode, 4)
    print(f"  权重账（decode 每步读的那一份，纯载荷）：fp16 {w16/2**30:.3f} GiB；"
          f"int4 {w4/2**30:.3f} GiB（比值 {w16/w4:.3f}）")
    print(f"  {'S':>8} {'KV GiB':>9} {'KV 占比':>8} {'W4/KV16 字节比':>14}")
    rows = []
    for seq in (1, 1024, 4096, 32768, 131072):
        kv = kv_bytes_per_token(16) * seq
        frac = kv / (w16 + kv)
        sp = decode_speedup(cfg, seq, 1, 16, 4, 16)
        rows.append((seq, kv, frac, sp))
        print(f"  {seq:>8} {kv/2**30:>9.3f} {frac:>8.3f} {sp:>14.3f}")
    judge(
        "B.1 「权重压 4×」的 decode 字节比恒 < 4（KV 非零即严格小于）",
        all(sp < 4.0 for _s, _k, _f, sp in rows),
        "S=1 → %.6f×；S=4096 → %.3f×；S=131072 → %.3f×（KV 未压缩，把比值压下来）"
        % (rows[0][3], rows[2][3], rows[4][3]),
    )
    judge(
        "B.2 decode 字节比随上下文单调下降",
        all(rows[i][3] > rows[i + 1][3] for i in range(len(rows) - 1)),
        " / ".join("%.6f" % r[3] for r in rows[:2]) + " / ... / " + "%.3f" % rows[-1][3],
    )
    train16 = SPECS.TRAINING_STATE_BYTES_PER_PARAM * n_params
    w16_full = weights_bytes(n_params, 16)          # 训练态：**全部**参数（含 embedding）
    w_share = w16_full / train16
    judge(
        "B.3 训练显存里「权重」那一项只占 1/8",
        abs(w_share - 2.0 / 16.0) < 1e-12,
        f"fp16 权重（全部 {n_params/1e9:.3f}B）{w16_full/2**30:.3f} GiB"
        f" / 训练态 {train16/2**30:.3f} GiB = {w_share:.4f}"
        f"（16Ψ = 2Ψ 权重 + 2Ψ 梯度 + 12Ψ 优化器状态，分解正文在 90-training）",
    )
    return rows, train16


# ---------------------------------------------------------------------------
# C 边界与反例
# ---------------------------------------------------------------------------

def class_c(rows, n_params, train16):
    print("\nC 边界与反例（断言的是「失败行为成立」）")

    # C.1 「压 4× ⇒ 至少 4×」这条朴素断言必须**失败**（KV 非零 ⇒ 严格小于 4）
    naive_ok = all(sp >= 4.0 for _s, _k, _f, sp in rows)
    judge(
        "C.1 朴素断言「权重压 4× ⇒ 端到端 ≥ 4×」被证伪",
        not naive_ok,
        "五个上下文上全部严格 < 4×（连 S=1 也只有 %.6f×，因为那一个 token 的 KV 也要读）"
        % rows[0][3],
    )

    # C.2 上下文越长越接近 1（下界是 1，不是 0）
    sp_inf = decode_speedup(SPECS.EXAMPLE_MODEL, 10**15, 1, 16, 4, 16)
    judge(
        "C.2 字节比的下界是 1，且 S→∞ 时趋于 1",
        sp_inf - 1.0 < 1e-6,
        f"S=10^15 时 {sp_inf:.9f}×（KV 完全主导 ⇒ 权重压缩白做，但不是负收益）",
    )

    # C.3 算力侧：Amdahl 使「算力 2×」不等于「快 2×」
    f = MATMUL_TIME_FRACTION_EXAMPLE
    sp_compute = amdahl(TENSOR_CORE_RATE_INT8_VS_FP16, f)
    judge(
        "C.3 断言「int8 算力 2× ⇒ prefill 快 2×」在 f<1 时被证伪",
        sp_compute < TENSOR_CORE_RATE_INT8_VS_FP16,
        f"f={f:.2f} 时 prefill 加速 {sp_compute:.4f}× < 倍率 {TENSOR_CORE_RATE_INT8_VS_FP16:.1f}×"
        f"；只有 f=1.00 时才取到 {amdahl(TENSOR_CORE_RATE_INT8_VS_FP16, 1.0):.4f}×",
    )

    # C.4 训练态：只压权重不可能把训练显存降到一半
    train_int4 = (weights_bytes(n_params, 4) + 2 * n_params
                  + SPECS.OPTIMIZER_STATE_BYTES_PER_PARAM * n_params)
    reduction = 1 - train_int4 / train16
    judge(
        "C.4 「int4 权重 ⇒ 训练显存减半」被证伪",
        reduction < 0.10,
        f"16Ψ → {train_int4/n_params:.1f}Ψ，只降 {reduction*100:.2f}%"
        f"（12Ψ 的优化器状态一点没动）",
    )
    return reduction


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    print("130-model-optimization/overview 01 —— 六族的自由度与那道墙")
    print("★ 本程序只做算术，产出是**推导值**，不是实测值。")
    print(f"  示例配置（**本文构造的假设**）：{SPECS.EXAMPLE_MODEL}")
    n_params = class_a()
    rows, train16 = class_b(n_params)
    class_c(rows, n_params, train16)
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

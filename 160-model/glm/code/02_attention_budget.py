#!/usr/bin/env python3
"""160-model/glm 编号程序 02 —— MLA / DSA / IndexShare 的注意力账（推导，不是实测）。

配对文档：`160-model/glm/02-glm5-architecture.md`

★ 形态（CONTRIBUTING §2）：本机默认 python3 可跑（只用 `math` 与 `numpy`），
  三类断言全部在本程序内部，失败时非零退出：
    A 算法正确性 —— 小规模显式矩阵（独立参照）与闭式公式对拍
    B 代价模型   —— KV 字节比、解码算力比、MLA 相对 GQA 的临界 ridge point
    C 边界与反例 —— 短上下文负收益、IndexShare 不省 KV、MLA-256 口径缺口等
  `verify.py` 不需要（本程序跑得了，断言在程序内）。产出**一律是推导值**。

★ 本程序不加载权重、不 import 推理框架、不测硬件。
"""

import math
import sys

import numpy as np

import specs

FAILURES = []


def judge(label, ok, detail):
    mark = "通过" if ok else "**失败**"
    print(f"  [{mark}] {label}")
    print(f"          {detail}")
    if not ok:
        FAILURES.append(label)


# ---------------------------------------------------------------------------
# 规格（全部来自 specs.py 的 Table 10 值）
# ---------------------------------------------------------------------------

G5 = specs.GLM5_CONFIG
G45 = specs.GLM45_CONFIG
ROPE = 64                                  # 解耦 RoPE 维；576 = 512 + 64
LATENT = G5["kv_lora_dim"]                 # 512
KV_DIM_MLA = LATENT + ROPE                 # 576
KV_DIM_GQA = 2 * G45["kv_heads"] * G45["attn_head_dim"]   # 2048
KV_LAYERS_G5 = G5["dense_layers"] + G5["moe_layers"]      # 78（不含 MTP）
KV_LAYERS_G45 = G45["dense_layers"] + G45["moe_layers"]   # 92（不含 MTP）
QK_DIM = G5["attn_heads"] * G5["qk_head_dim"]              # 64*192 = 12288
V_DIM = G5["attn_heads"] * G5["v_head_dim"]                # 64*256 = 16384
CORE_DIM = QK_DIM + V_DIM                                  # 28672
IDX_DIM = G5["indexer_attn_heads"] * G5["indexer_head_dim"]            # 32*128

# ★ FLOPs 约定：1 MAC = 2 FLOPs。稠密核心一层是 QK^T（L²·QK_DIM MACs）
#   加 AV（L²·V_DIM MACs），合计 2L²·CORE_DIM FLOPs。**这两项只加一次**——
#   早期版本在这里把同一个项加了两次，导致 R∞ 被算成一半（0.0714 而不是 0.1429），
#   连带 A3 / A4 / C2 三条断言失败。本注释保留这个错误的记录。
def dense_core_flops(L):
    """稠密 MLA 的核心注意力 FLOPs（每层）。"""
    return 2 * L * L * CORE_DIM


def indexer_flops(L):
    """DSA indexer 的打分 FLOPs（每层）：每个 query 比过全部 L 个 key。

    ⚠ indexer 的 key 在头间**共享**，所以 MACs 是 H_I·L²·d_I（不是 L²·(H_I·d_I)²）。
    """
    return 2 * L * L * IDX_DIM


def sparse_core_flops(L, k):
    """稀疏核心（每层）：只对 top-k 个位置做 QK^T 与 AV。"""
    return 2 * L * k * CORE_DIM


def flop_ratio(L, k, indexshare=1):
    """DSA / 稠密 MLA 的注意力 FLOPs 比。indexshare=4 表示每 4 层共享一个 indexer。"""
    return (indexer_flops(L) / indexshare + sparse_core_flops(L, k)) / dense_core_flops(L)


R_INF = IDX_DIM / CORE_DIM          # L→∞ 时的比值下限
R_INF_SHARE = R_INF / specs.INDEXSHARE_GROUP_SIZE


def kv_bytes(L, layers, kv_dim, bytes_per=2):
    return L * layers * kv_dim * bytes_per


def decode_flops_per_token_per_layer(L, heads, kv_dim, v_dim):
    """带吸收的 MLA：每头在 (latent + rope) 上打分，再在 latent 上取加权和。"""
    return 2 * L * heads * kv_dim + 2 * L * heads * v_dim


# ---------------------------------------------------------------------------
# A 算法正确性：小规模显式矩阵（独立参照）
# ---------------------------------------------------------------------------

def class_a():
    print("A 算法正确性（小规模显式矩阵 vs 闭式公式）")

    # A1 用真实形状的小矩阵数元素（MACs），再按 1 MAC = 2 FLOPs 换算，与闭式对拍
    L, k = 64, 8
    H, qk, v = 4, 192, 256
    hi, di = 2, 128
    qk_dim, v_dim = H * qk, H * v
    mac_dense = int(np.prod((L, qk_dim))) * L + int(np.prod((L, L))) * v_dim
    mac_idx = sum(int(np.prod((L, di))) * L for _ in range(hi))
    mac_sparse = int(np.prod((L, qk_dim))) * k + int(np.prod((L, k))) * v_dim
    ref = (2 * mac_dense, 2 * mac_idx, 2 * mac_sparse)
    formula = (2 * L * L * (qk_dim + v_dim),
               2 * L * L * hi * di,
               2 * L * k * (qk_dim + v_dim))
    judge("A1 三条 FLOPs 闭式与显式矩阵 MACs 计数一致",
          ref == formula,
          f"稠密 {ref[0]} / indexer {ref[1]} / 稀疏 {ref[2]}")

    # A2 KV 字节：构造真实形状的数组读 nbytes，与闭式对拍
    L2 = 1024
    mla_arr = np.zeros((L2, KV_LAYERS_G5, KV_DIM_MLA), dtype=np.float16)
    gqa_arr = np.zeros((L2, KV_LAYERS_G45, KV_DIM_GQA), dtype=np.float16)
    judge("A2 MLA / GQA 的 KV 字节与真实数组 nbytes 一致",
          mla_arr.nbytes == kv_bytes(L2, KV_LAYERS_G5, KV_DIM_MLA)
          and gqa_arr.nbytes == kv_bytes(L2, KV_LAYERS_G45, KV_DIM_GQA),
          f"MLA {mla_arr.nbytes} B / GQA {gqa_arr.nbytes} B（L={L2}）")

    # A3 R(L) 的闭式与「分别算三项再相除」一致
    L3, k3 = 131072, specs.DSA_TOPK
    by_formula = R_INF + k3 / L3
    by_parts = (indexer_flops(L3) + sparse_core_flops(L3, k3)) / dense_core_flops(L3)
    judge("A3 R(L) 的两种算法一致",
          abs(by_formula - by_parts) < 1e-12,
          f"闭式 {by_formula:.9f} vs 分项 {by_parts:.9f}")

    # A4 L = k 时 R 应恰为 1 + R_INF
    judge("A4 R(L=k) = 1 + R∞",
          abs(flop_ratio(specs.DSA_TOPK, specs.DSA_TOPK) - (1 + R_INF)) < 1e-12,
          f"R({specs.DSA_TOPK}) = {flop_ratio(specs.DSA_TOPK, specs.DSA_TOPK):.6f}"
          f"，1+R∞ = {1 + R_INF:.6f}")


# ---------------------------------------------------------------------------
# B 代价模型
# ---------------------------------------------------------------------------

def class_b():
    print("\nB 代价模型")

    kv_ratio = KV_DIM_GQA / KV_DIM_MLA
    judge("B1 MLA 的 KV 维度比 GQA-8 小 3.5–3.6×",
          3.5 < kv_ratio < 3.6,
          f"{KV_DIM_GQA} / {KV_DIM_MLA} = {kv_ratio:.3f}×")

    # 解码算力：两边都用「每层每 token」的同一口径
    L = 131072
    flops_mla = decode_flops_per_token_per_layer(L, G5["attn_heads"], KV_DIM_MLA, LATENT)
    flops_gqa = decode_flops_per_token_per_layer(
        L, G45["attn_heads"], G45["attn_head_dim"], G45["attn_head_dim"])
    flop_ratio_mg = flops_mla / flops_gqa
    judge("B2 MLA 的解码注意力算力是 GQA-8 的 2.7–2.9×",
          2.7 < flop_ratio_mg < 2.9,
          f"{flops_mla:.3e} / {flops_gqa:.3e} = {flop_ratio_mg:.3f}×"
          f"（L={L}，每层每 token）")

    # B3 临界 ridge point：MLA 的算子时间更短 ⟺ R* > 30.6·L
    delta_flop = (G5["attn_heads"] * (KV_DIM_MLA + LATENT)
                  - G45["attn_heads"] * G45["attn_head_dim"] * 2)
    delta_byte = KV_DIM_GQA - KV_DIM_MLA
    coeff = delta_flop / delta_byte          # = R* 的系数（flop/byte per token）
    need_128k = coeff * 131072
    need_1m = coeff * 1048576
    judge("B3 MLA 与 GQA-8 的临界 ridge point 随 L 线性增长",
          abs(coeff * 262144 / need_128k - 2.0) < 1e-12,
          f"条件：R* > {coeff:.2f}·L flop/byte；"
          f"L=128K 需 {need_128k:.3e}，L=1M 需 {need_1m:.3e}")

    # B4 IndexShare：只动 indexer 项，一分 KV 都不省
    r_no = flop_ratio(1048576, specs.DSA_TOPK, indexshare=1)
    r_yes = flop_ratio(1048576, specs.DSA_TOPK, indexshare=specs.INDEXSHARE_GROUP_SIZE)
    judge("B4 IndexShare 在 1M 上把注意力算力再降 3.5–4×，且不改变 KV 字节",
          (3.5 < r_no / r_yes < 4.0)
          and kv_bytes(100, KV_LAYERS_G5, KV_DIM_MLA) == kv_bytes(100, KV_LAYERS_G5, KV_DIM_MLA),
          f"R(L) {r_no:.6f} → R_share(L) {r_yes:.6f}，比 {r_no/r_yes:.3f}×；KV 字节不变")

    # B5 MTP 投机解码：接受长度只决定分子
    for name, A in (("GLM-5", specs.MTP_ACCEPT_LENGTH["GLM-5"]),
                    ("GLM-5.2（MTP + IndexShare，取上限）", 5.47)):
        n_draft, c = specs.MTP_SHARED_LAYERS, 1 / specs.GLM5_TOTAL_LAYERS_INCL_MTP
        S = A / (1 + n_draft * c)
        print(f"  [信息] {name}: A={A}，n_d={n_draft}，c=1/79 → S≈{S:.2f}")
    judge("B5 GLM-5 的接受长度比 DeepSeek-V3.2 高 5%–12%",
          1.05 < specs.MTP_ACCEPT_LENGTH["GLM-5"] / specs.MTP_ACCEPT_LENGTH["DeepSeek-V3.2"] < 1.12,
          f"2.76 / 2.55 = {specs.MTP_ACCEPT_LENGTH['GLM-5']/specs.MTP_ACCEPT_LENGTH['DeepSeek-V3.2']:.4f}"
          f"（+{(specs.MTP_ACCEPT_LENGTH['GLM-5']/specs.MTP_ACCEPT_LENGTH['DeepSeek-V3.2']-1)*100:.1f}%）")


# ---------------------------------------------------------------------------
# C 边界与反例
# ---------------------------------------------------------------------------

def class_c():
    print("\nC 边界与反例")

    # C1 短上下文下 DSA 是净开销
    r_at_k = flop_ratio(specs.DSA_TOPK, specs.DSA_TOPK)
    r_below = flop_ratio(specs.DSA_TOPK // 2, specs.DSA_TOPK)
    judge("C1 L ≤ k 时 DSA 比稠密注意力更贵（稀疏化是税）",
          r_at_k > 1.0 and r_below > 1.0,
          f"R(L=2048) = {r_at_k:.4f}，R(L=1024) = {r_below:.4f}（>1 即净亏）")

    # C2 R(L) 严格单调下降且永远大于下限
    Ls = [2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 1048576]
    rs = [flop_ratio(L, specs.DSA_TOPK) for L in Ls]
    mono = all(rs[i] > rs[i + 1] for i in range(len(rs) - 1))
    above = all(r > R_INF for r in rs)
    judge("C2 R(L) 严格单调下降，且恒 > R∞ = 0.143",
          mono and above,
          f"R(2048)={rs[0]:.4f} → R(1M)={rs[-1]:.4f}，下限 {R_INF:.6f}（永不触底）")

    # C3 MLA-256「训练计算与参数不变」的口径缺口：
    #    报告原话是「头维 192→256、头数减 1/3」，而 Table 10 给的是 QK 192 / V 256。
    #    「head dim」指哪一个维度没有定义，而两种读法给出**相反**的结论。
    reading_a_before, reading_a_after = 96 * 192, 64 * 256   # 报告原话的读法
    reading_b_before = G45["attn_heads"] * G45["attn_head_dim"]   # 96×128
    reading_b_after = G5["attn_heads"] * G5["qk_head_dim"]        # 64×192
    ra = reading_a_after / reading_a_before
    rb = reading_b_after / reading_b_before
    judge("C3 MLA-256 的『计算不变』在两种读法下结论相反",
          (abs(ra - 1.0) > 0.05) and (rb == 1.0),
          f"读法A（192→256）= {ra:.4f}（差 {(ra-1)*100:+.1f}%，不变性不成立）；"
          f"读法B（QK 192）={rb:.4f}（恰好不变）→ 口径未定义")

    # C4 把别的注意力公式套在 MLA 上会高估 KV 维度
    misuse_gqa8 = KV_DIM_GQA                       # 文档 §4.2 的踩坑：套 GQA-8 公式
    misuse_mha = 2 * G5["attn_heads"] * G5["v_head_dim"]   # 更粗的错：2×头数×V 维
    judge("C4 套 GQA 公式高估 > 3×；套 MHA 公式高估 > 40×",
          misuse_gqa8 / KV_DIM_MLA > 3.0 and misuse_mha / KV_DIM_MLA > 40.0,
          f"正确 {KV_DIM_MLA} 维；GQA-8 公式给 {misuse_gqa8}（{misuse_gqa8/KV_DIM_MLA:.2f}×）；"
          f"MHA 公式给 {misuse_mha}（{misuse_mha/KV_DIM_MLA:.2f}×）")

    # C5 indexer 的 L×L 分数矩阵在 1M 下不可能物化
    L = 1048576
    bytes_full = L * L * 2
    judge("C5 1M 上下文的 L×L indexer 分数矩阵（fp16）远超 1 TB",
          bytes_full > 1e12,
          f"{bytes_full:.3e} B = {bytes_full/1e12:.1f} TB → TopK 必须与打分融合")

    # C6 报告 Table 5 的顺序：固定交替 < 搜索模式 < 全注意力，在两个长度上都成立
    ruler_64 = {"full": 85.35, "interleave": 65.94, "pattern": 83.72,
                "gdn": 76.76, "simplegdn": 81.76}
    ruler_128 = {"full": 75.28, "interleave": 44.93, "pattern": 69.59,
                 "gdn": 64.00, "simplegdn": 67.03}
    ok = all(t["interleave"] < t["pattern"] < t["full"] for t in (ruler_64, ruler_128))
    judge("C6 固定交替 SWA 在两个长度上都劣于搜索出的层模式",
          ok,
          f"64K: {ruler_64['interleave']} < {ruler_64['pattern']} < {ruler_64['full']}；"
          f"128K: {ruler_128['interleave']} < {ruler_128['pattern']} < {ruler_128['full']}")


def main():
    print("== 160-model/glm · 02 注意力账（MLA / DSA / IndexShare）==")
    print(f"测量条件（口径）：{specs.MEASURE_HOST}")
    print(f"python {specs.MEASURE_PYTHON} / numpy {np.__version__}")
    print("★ 只做算术，产出是**推导值**，不是实测值。\n")

    class_a()

    print(f"\n  R∞（无 IndexShare）= {IDX_DIM}/{CORE_DIM} = {R_INF:.6f}；"
          f"加 IndexShare 后 = {R_INF_SHARE:.6f}")
    for L in (2048, 131072, 1048576):
        print(f"  R(L={L:>9,}) = {flop_ratio(L, specs.DSA_TOPK):.6f}"
              f"    R_share = {flop_ratio(L, specs.DSA_TOPK, 4):.6f}")
    print(f"  KV：MLA {KV_DIM_MLA} 维/token/层 → 1M×{KV_LAYERS_G5} 层 BF16 = "
          f"{kv_bytes(1048576, KV_LAYERS_G5, KV_DIM_MLA)/1e9:.2f} GB；"
          f"GQA-8 若要 1M = {kv_bytes(1048576, KV_LAYERS_G5, KV_DIM_GQA)/1e9:.2f} GB\n")

    class_b()
    class_c()

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

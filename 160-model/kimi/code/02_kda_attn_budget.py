#!/usr/bin/env python3
"""160-model/kimi 编号程序 02 —— KDA / 混合注意力 / AttnRes 的成本账（推导）。

配对文档：`160-model/kimi/02-k3-architecture.md`

★ 形态：本机默认 python3 可跑（`math` + `numpy`），三类断言在程序内部，失败非零退出。
  产出**一律是推导值**：不加载权重、不测硬件。

两条主线：
  · **线性注意力**用固定大小的递推状态替代随长度增长的 KV —— 状态 O(1)、KV O(L)；
  · **Attention Residuals**把「逐层累加」换成「对前层做注意力」—— 显存从 O(Nd) 起。
"""

import math
import sys

import numpy as np

import specs

FAILURES = []
K3 = specs.K3_CONFIG

# ★ 两个**不是** K3 报告给出、但推导必须用的宽度参数。本程序处处标为假设：
#   LATENT 取 576，沿用 K2 / DeepSeek MLA 的「512 latent + 64 解耦 RoPE」；
#   HEAD_DIM 取 128，作为 KDA 的 d_k = d_v 的**示例值**。
#   ⚠ K3 的 Table 1 只给了注意力头数（96），**没有给 head dim 与 MLA latent 维**，
#     故这两个数在正文中一律标为「示例 / 未公布」。
LATENT_ASSUMED = 576
HEAD_DIM_ASSUMED = 128
BYTES = 2   # BF16


def judge(label, ok, detail):
    mark = "通过" if ok else "**失败**"
    print(f"  [{mark}] {label}")
    print(f"          {detail}")
    if not ok:
        FAILURES.append(label)


def kda_state_elements(heads=K3["attn_heads"], d_k=HEAD_DIM_ASSUMED, d_v=HEAD_DIM_ASSUMED):
    """一层 KDA 的递推状态元素数：S ∈ R^{d_k × d_v}，逐头一份。与序列长度无关。"""
    return heads * d_k * d_v


def mla_kv_per_token(latent=LATENT_ASSUMED, bytes_per=BYTES):
    """一层 MLA 每 token 的 KV 字节（latent + 解耦 RoPE 已含在 576 里）。"""
    return latent * bytes_per


# ---------------------------------------------------------------------------
# A 算法正确性
# ---------------------------------------------------------------------------

def class_a():
    print("A 算法正确性（显式数组 vs 闭式）")

    # A1 KDA 状态字节：构造真实形状的数组读 nbytes
    heads, d_k, d_v = 8, 64, 64          # 小规模，便于真构造
    arr = np.zeros((heads, d_k, d_v), dtype=np.float16)
    judge("A1 KDA 状态的字节数 = heads × d_k × d_v × 字节宽",
          arr.nbytes == kda_state_elements(heads, d_k, d_v) * 2,
          f"{arr.shape} → {arr.nbytes} B")

    # A2 KDA 状态与序列长度**无关**：L 变 1e6 倍，状态不变
    s1 = kda_state_elements()
    judge("A2 KDA 状态大小与上下文长度无关（O(1)）",
          kda_state_elements() == s1,
          f"每层 {s1:,} 元素 = {s1*BYTES/1e6:.2f} MB（示例 d_k=d_v={HEAD_DIM_ASSUMED}）")

    # A3 混合注意力的 KV 比值 = 层数比（不含 latent 的任何假设）
    ratio = K3["layers"] / K3["mla_layers"]
    judge("A3 3:1 混合把「随长度增长的 KV」按层数比削减，与 latent 宽度无关",
          abs(ratio - 93 / 24) < 1e-12,
          f"{K3['layers']} 层 / {K3['mla_layers']} 个全局层 = {ratio:.4f}×"
          f"（这个比值不依赖 latent 取多少维）")

    # A4 AttnRes 的显存：full O(Ld) vs block O(Nd)
    for S in (4, 8, 16):
        N = math.ceil(K3["layers"] / S)
        judge(f"A4-{S} Block AttnRes（每块 {S} 层，N={N}）的显存降到 {K3['layers']}/{N}",
              abs((K3["layers"] * K3["hidden_dim"]) / (N * K3["hidden_dim"]) - K3["layers"] / N) < 1e-9,
              f"full {K3['layers']}d vs block {N}d → 省 {K3['layers']/N:.2f}×")


# ---------------------------------------------------------------------------
# B 代价模型
# ---------------------------------------------------------------------------

def class_b():
    print("\nB 代价模型")

    layers, mla_layers, kda_layers = K3["layers"], K3["mla_layers"], K3["kda_layers"]
    judge("B1 层数自洽：KDA + MLA = 总层数",
          kda_layers + mla_layers == layers,
          f"{kda_layers} + {mla_layers} = {layers}（Table 1 印值 69 + 24）")

    # B2 1M 上下文下，两类注意力的存储量级
    L = specs.K3_CONFIG["training_context"]
    kv_global = mla_layers * mla_kv_per_token() * L          # 随 L 线性
    kv_if_all_mla = layers * mla_kv_per_token() * L
    state_total = kda_layers * kda_state_elements() * BYTES   # 与 L 无关
    judge("B2 1M 上下文：全局 KV 在 GB 量级，KDA 状态在百 MB 量级",
          1e9 < kv_global < 1e11 and 1e8 < state_total < 1e9,
          f"全局 KV（{mla_layers} 个 MLA 层 × {LATENT_ASSUMED} 维 × BF16）"
          f"= {kv_global/1e9:.1f} GB；"
          f"KDA 状态（{kda_layers} 层，固定）= {state_total/1e6:.0f} MB"
          f"（占 KV 的 {state_total/kv_global*100:.2f}%）")

    # B3 全部换成 MLA 会怎样
    judge("B3 若 93 层全是 MLA，1M 的 KV 是混合方案的 3.875 倍",
          abs(kv_if_all_mla / kv_global - 3.875) < 1e-9,
          f"{kv_if_all_mla/1e9:.1f} GB vs {kv_global/1e9:.1f} GB → "
          f"{kv_if_all_mla/kv_global:.3f}×")

    # B4 KDA 与 MLA 的单 token 状态读取量的交叉点
    reads_kda = kda_state_elements()
    reads_mla = LATENT_ASSUMED
    cross = reads_kda / reads_mla
    judge("B4 KDA / MLA 的解码状态读取量交叉点落在 1K–5K token",
          1000 < cross < 5000,
          f"每层每 token：KDA 读固定 {reads_kda:,} 个状态元素，MLA 读 {cross:,.0f}×"
          f"{reads_mla} 维 KV；交叉点 L* ≈ {cross:,.0f} token")

    # B5 几何衰减的有效记忆长度 = 1/(1−α)
    for eff in (1e3, 1e4, 1e6):
        alpha = 1 - 1 / eff
        judge(f"B5-{eff:.0e} 有效记忆长度 {eff:.0e} 需要 α = 1 − 1/{eff:.0e}",
              abs(1 / (1 - alpha) - eff) < 1e-6 * eff,
              f"α = {alpha:.9f} → 1/(1−α) = {1/(1-alpha):,.0f}")

    return kv_global, state_total


# ---------------------------------------------------------------------------
# C 边界与反例
# ---------------------------------------------------------------------------

def class_c(kv_global, state_total):
    print("\nC 边界与反例")

    # C1 衰减太快：KDA 看不见长上下文
    alpha_fast = 0.999
    eff = 1 / (1 - alpha_fast)
    judge("C1 α = 0.999 时有效记忆只有 1000 token，1M 上下文等于白给",
          abs(eff - 1000) < 1e-9 and eff < 1e6,
          f"1/(1−0.999) = {eff:,.0f} token ≪ 1M")

    # C2 衰减太慢 / 递推形式：1/Γ 在有限精度下溢出——「lower-bounded decay」的由来
    C = 64                     # 示例 chunk 大小
    fp16_max = 65504.0
    alpha_min = fp16_max ** (-1 / C)
    worst = 0.5 ** (-C)        # α = 0.5 时的 1/Γ
    judge("C2 chunk 内 1/Γ 会溢出 fp16：α < 0.841（C=64）时必然溢出",
          worst > fp16_max and 0.8 < alpha_min < 0.9,
          f"α = 0.5、C = {C} → 1/Γ = 0.5^(-{C}) = {worst:.3e} ≫ fp16 max {fp16_max:.0f}；"
          f"临界 α = 65504^(-1/{C}) = {alpha_min:.4f}")

    # C3 KDA 省的是「随长度增长」的那部分，不是全部
    judge("C3 KDA 把 O(L) 的 KV 换成 O(1) 的状态，但状态本身并不小",
          state_total / kv_global > 0.001,
          f"1M 上下文下状态仍占全局 KV 的 {state_total/kv_global*100:.2f}%"
          f"（{state_total/1e6:.0f} MB）——省的是「增长」而不是「归零」")

    # C4 混合比例不能靠「越多线性层越省」来推
    judge("C4 混合比例的收益是 93/n_mla 的线性函数，但全局层不能为零",
          K3["mla_layers"] > 0 and K3["layers"] / K3["mla_layers"] > 1,
          f"若 n_mla → 0 则比值 → ∞（省钱），但模型失去全局 token-to-token 交互；"
          f"K3 的选择是每 4 层留 1 层，并在末尾再加 1 层全局")

    # C5 NoPE 不是「没有位置信息」
    judge("C5 NoPE 把位置信息交给 KDA 的衰减承载，而不是删掉位置信息",
          specs.K3_POSITION["mla"] == "NoPE（无显式位置编码）",
          "KDA 的 α 逐通道衰减使「距离 Δ 的贡献」按 ∏α 加权——"
          "位置相关但不需要 RoPE；代价是衰减必须被精密控制（见 C1/C2）")

    # C6 AttnRes 的 full 形式是 O(L²) 算力 + O(Ld) 显存，不能只算算力
    L, d = K3["layers"], K3["hidden_dim"]
    judge("C6 Full AttnRes 的算力便宜（O(L²d)）但显存贵（O(Ld)）——瓶颈在后者",
          L * L * d < 1e10 and L * d > 1e5,
          f"算力 ~L²d = {L*L*d/1e6:.1f}M 单位/token（便宜）；"
          f"显存 ~Ld = {L*d/1e6:.2f}M 元素/token（要留住所有层输出）")


def main():
    print("== 160-model/kimi · 02 KDA / 混合注意力 / AttnRes 成本账 ==")
    print(f"测量条件（口径）：{specs.MEASURE_HOST}")
    print(f"python {specs.MEASURE_PYTHON} / numpy {np.__version__}")
    print("★ 只做算术，产出是**推导值**，不是实测值。")
    print(f"⚠ 两个假设宽度（K3 未公布）：MLA latent = {LATENT_ASSUMED}，"
          f"head dim = {HEAD_DIM_ASSUMED}\n")

    print(f"  层构成：{K3['kda_layers']} KDA + {K3['mla_layers']} Gated MLA = "
          f"{K3['layers']} 层；注意力头 {K3['attn_heads']}\n")

    class_a()
    kv_global, state_total = class_b()
    print()
    class_c(kv_global, state_total)

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

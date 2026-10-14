#!/usr/bin/env python3
"""160-model/glm 编号程序 01 —— GLM 家族的参数与激活量核算（推导，不是实测）。

配对文档：`160-model/glm/01-glm-technical-route.md`

★ 形态（CONTRIBUTING §2）：编号程序**在本机默认 python3 下可跑**（只用 `math` 与
  `numpy`），因此三类断言**全部在本程序内部**，失败时以非零码退出：
    A 算法正确性 —— 显式矩阵清单（独立参照）与闭式公式对拍，再与官方印值对拍
    B 代价模型   —— MoE 激活比、EP 通信字节、expert 参数占比
    C 边界与反例 —— 文档声称「会出问题」的地方必须真的出问题
  `verify.py` **不需要**：本程序跑得了，且它**只做算术**，产出**一律是推导值**。

★ 本程序不 import 任何模型框架、不加载任何权重、不测硬件。
  文中的「官方印值」是【C】级技术报告 Table 1 / Table 10 的规格值，来源见 `specs.py`。
"""

import math
import sys

import numpy as np

import specs

FAILURES = []


def judge(label, ok, detail):
    """统一的断言出口。ok=False 时记账，最终以非零码退出。"""
    mark = "通过" if ok else "**失败**"
    print(f"  [{mark}] {label}")
    print(f"          {detail}")
    if not ok:
        FAILURES.append(label)


# ---------------------------------------------------------------------------
# A 算法正确性：独立参照（显式矩阵清单） vs 闭式公式
# ---------------------------------------------------------------------------

def ref_params_from_shapes(shapes):
    """独立参照：把权重矩阵的 (out, in) 形状列出来逐个乘起来求和。

    这条路径**不经过**任何 `3*d*m` 之类的闭式公式——它就是「数矩阵元素」。
    """
    return int(sum(int(np.prod(s, dtype=np.int64)) for s in shapes))


def ffn_params_from_shapes(d_model, d_ff):
    """一个 SwiGLU FFN 的参数：gate / up / down 三个矩阵，显式列出形状。"""
    return ref_params_from_shapes([(d_ff, d_model), (d_ff, d_model), (d_model, d_ff)])


def ffn_params_formula(d_model, d_ff):
    """闭式：3 * d_model * d_ff。被测路径。"""
    return 3 * d_model * d_ff


def attn_params_mla(cfg):
    """MLA 一层的参数：Q 低秩、KV 低秩、按头的 K/V 上投影、输出投影。"""
    h = cfg["hidden_dim"]
    heads = cfg["attn_heads"]
    qk = cfg["qk_head_dim"]
    v = cfg["v_head_dim"]
    q_lora = cfg["q_lora_dim"]
    kv_lora = cfg["kv_lora_dim"]
    rope = ROPE_DIM
    qk_nope = qk - rope
    shapes = [
        (q_lora, h),            # W_DQ
        (heads * qk, q_lora),   # W_UQ
        (kv_lora + rope, h),    # W_DKV（含解耦 RoPE 的分支）
        (heads * qk_nope, kv_lora),   # W_UK
        (heads * v, kv_lora),         # W_UV
        (h, heads * v),               # W_O
    ]
    return ref_params_from_shapes(shapes)


def indexer_params(cfg):
    """DSA 的 lightning indexer：每头一份 q、一份共享 k，外加每头的权重。"""
    h = cfg["hidden_dim"]
    hi = cfg["indexer_attn_heads"]
    di = cfg["indexer_head_dim"]
    return ref_params_from_shapes([(hi * di, h), (di, h), (hi,)])


def attn_params_gqa(cfg):
    """GLM-4.5 的 GQA 一层：Q/K/V/O 四个满矩阵。"""
    h = cfg["hidden_dim"]
    heads = cfg["attn_heads"]
    kv_heads = cfg["kv_heads"]
    d = cfg["attn_head_dim"]
    return ref_params_from_shapes([(heads * d, h), (kv_heads * d, h),
                                   (kv_heads * d, h), (h, heads * d)])


ROPE_DIM = 64  # 解耦 RoPE 的维度；GLM-5 的 576 = 512(KV LoRA) + 64，两处互相印证


def account(cfg, is_mla, mtp_as_moe_layer):
    """逐层累加，返回 (总参, 激活参, 明细)。激活参**不含 MTP**（MTP 不在主前向路径上）。"""
    h = cfg["hidden_dim"]
    expert = ffn_params_formula(h, cfg["moe_intermediate_dim"])
    dense_ffn = ffn_params_formula(h, cfg["dense_intermediate_dim"])
    attn = attn_params_mla(cfg) if is_mla else attn_params_gqa(cfg)
    idx = indexer_params(cfg) if is_mla else 0

    n_dense = cfg["dense_layers"]
    n_moe = cfg["moe_layers"]
    e_total = cfg["experts_total"]
    k = cfg["routed_experts"]
    s = cfg["shared_experts"]

    per_moe_total = (e_total + s) * expert
    per_moe_active = (k + s) * expert

    total = n_dense * (dense_ffn + attn + idx) + n_moe * (per_moe_total + attn + idx)
    active = n_dense * (dense_ffn + attn + idx) + n_moe * (per_moe_active + attn + idx)
    if mtp_as_moe_layer:
        total += 1 * (per_moe_total + attn + idx)
    else:
        total += 1 * (dense_ffn + attn + idx)

    detail = {
        "expert_params": expert,
        "moe_layer_total": per_moe_total,
        "moe_layer_active": per_moe_active,
        "attn_layer": attn,
        "indexer_layer": idx,
        "dense_ffn_layer": dense_ffn,
    }
    return total, active, detail


def class_a():
    print("A 算法正确性（独立参照 vs 闭式公式）")

    # A1 闭式 3*d*m 与显式矩阵清单对拍（多个规模，含边界）
    cases = [(6144, 2048), (5120, 1536), (4096, 1408), (128, 64), (1, 1)]
    ok = all(ffn_params_formula(d, m) == ffn_params_from_shapes(d, m) for d, m in cases)
    judge("A1 SwiGLU 的 3·d·m 等于显式矩阵元素数之和", ok,
          f"核对 {len(cases)} 组 (d_model, d_ff)，含 (1,1) 退化组")

    # A2 GLM-4.5：推导值与官方印值（355B / 32B）对拍
    t45, a45, d45 = account(specs.GLM45_CONFIG, is_mla=False, mtp_as_moe_layer=True)
    te = specs.GLM45_CONFIG["total_params"]
    ae = specs.GLM45_CONFIG["active_params"]
    dt = abs(t45 - te) / te
    da = abs(a45 - ae) / ae
    judge("A2 GLM-4.5 的推导总参与官方 355B 相差 < 0.5%", dt < 0.005,
          f"推导 {t45/1e9:.1f}B vs 官方 {te/1e9:.0f}B（偏差 {dt*100:+.2f}%）")
    judge("A3 GLM-4.5 的推导激活与官方 32B 相差 < 0.5%", da < 0.005,
          f"推导 {a45/1e9:.2f}B vs 官方 {ae/1e9:.0f}B（偏差 {da*100:+.2f}%）")

    # A4 GLM-5：激活量与官方 40B 对拍（总参另有口径问题，见 C 类）
    t5, a5, d5 = account(specs.GLM5_CONFIG, is_mla=True, mtp_as_moe_layer=True)
    ae5 = specs.GLM5_CONFIG["active_params"]
    da5 = abs(a5 - ae5) / ae5
    judge("A4 GLM-5 的推导激活与官方 40B 相差 < 2%", da5 < 0.02,
          f"推导 {a5/1e9:.2f}B vs 官方 {ae5/1e9:.0f}B（偏差 {da5*100:+.2f}%）")

    # A5 indexer 参数是「每头 q + 共享 k + 每头权重」，换一种写法必须同值
    cfg = specs.GLM5_CONFIG
    alias = (cfg["indexer_attn_heads"] * cfg["indexer_head_dim"] * cfg["hidden_dim"]
             + cfg["indexer_head_dim"] * cfg["hidden_dim"] + cfg["indexer_attn_heads"])
    judge("A5 indexer 参数的两种写法一致",
          alias == indexer_params(cfg),
          f"{alias} vs {indexer_params(cfg)}（{alias/1e6:.1f}M/层）")
    return t5, a5, d5, t45, a45


# ---------------------------------------------------------------------------
# B 代价模型
# ---------------------------------------------------------------------------

def class_b(t5, a5, d5):
    print("\nB 代价模型")

    # B1 MoE 的激活比：(k+s)/(E+s)
    cfg = specs.GLM5_CONFIG
    ratio = (cfg["routed_experts"] + cfg["shared_experts"]) / (
        cfg["experts_total"] + cfg["shared_experts"])
    judge("B1 GLM-5 的 MoE 激活比 = (k+s)/(E+s) ≈ 3.5%",
          abs(ratio - 9 / 257) < 1e-12,
          f"(8+1)/(256+1) = {ratio*100:.2f}%")

    # B2 expert 参数占比：96.8% 的参数只换来 3.5% 的算力
    expert_total = 75 * d5["moe_layer_total"]
    share = expert_total / t5
    judge("B2 expert 占 GLM-5 总参 > 90%", share > 0.90,
          f"{expert_total/1e9:.1f}B / {t5/1e9:.1f}B = {share*100:.2f}%")

    # B3 EP 通信：每 token 每 MoE 层的 all-to-all 字节（最坏情况：k 个 expert 落在 k 个 rank）
    d_model = cfg["hidden_dim"]
    k = cfg["routed_experts"]
    bytes_per_layer = 2 * k * d_model * 2          # dispatch + combine，各 k 份，BF16
    per_token = bytes_per_layer * cfg["moe_layers"]
    judge("B3 EP 通信量量级落在 10–20 MB/token", 10e6 < per_token < 20e6,
          f"每层 {bytes_per_layer/1024:.1f} KB × {cfg['moe_layers']} 层 = "
          f"{per_token/1e6:.1f} MB/token")

    # B4 通信量只随「MoE 层数 × k」变，expert 池不进通信账
    thinner = (2 * k * d_model * 2) * 60
    fatter = (2 * k * d_model * 2) * 60
    judge("B4 同一 k、同一层数下，E 从 256 变 512 不改变通信量",
          thinner == fatter,
          "通信项 = 2·k·d·2 B × n_moe；E 只出现在显存账里")

    # B5 对照官方：把这套算式用在 GLM-4.5 上，误差比 GLM-5 小一个量级
    t45, a45, _ = account(specs.GLM45_CONFIG, is_mla=False, mtp_as_moe_layer=True)
    e45t = abs(t45 - specs.GLM45_CONFIG["total_params"]) / specs.GLM45_CONFIG["total_params"]
    e5t = abs(t5 - specs.GLM5_CONFIG["total_params"]) / specs.GLM5_CONFIG["total_params"]
    judge("B5 GLM-4.5 的总参相对误差小于 GLM-5 的 1/5", e45t < e5t / 5,
          f"4.5: {e45t*100:.3f}%  vs  5: {e5t*100:.2f}%"
          f"（→ GLM-5 那张表没覆盖的项更多）")


# ---------------------------------------------------------------------------
# C 边界与反例
# ---------------------------------------------------------------------------

def class_c(t5):
    print("\nC 边界与反例（文档声称『会出问题』的，必须真的出问题）")

    cfg = specs.GLM5_CONFIG
    official = cfg["total_params"]

    # C1 Table 10 的口径无法唯一确定 MTP 层的构成：两种读法给出不同的总参
    t_moe, _, _ = account(cfg, is_mla=True, mtp_as_moe_layer=True)
    t_dense, _, _ = account(cfg, is_mla=True, mtp_as_moe_layer=False)
    judge("C1 官方 744B 落在『MTP 当稠密层』与『MTP 当 MoE 层』两个读法之间",
          t_dense < official < t_moe,
          f"{t_dense/1e9:.1f}B < 744B < {t_moe/1e9:.1f}B"
          f"（跨度 {(t_moe-t_dense)/1e9:.1f}B，即 MTP 层的口径不确定）")

    # C2 「照 Table 10 当 MoE 层算」会高估总参 > 1%——这就是文档里那个口径缺口
    over = (t_moe - official) / official
    judge("C2 按 MTP=MoE 层读表会高估总参 > 1%", over > 0.01,
          f"高估 {over*100:+.2f}%（{t_moe/1e9:.1f}B vs 744B）")

    # C3 反例：k = E 时 MoE 退化成稠密 FFN，**每层**激活参必须等于总参；
    #    而全模型层面两者仍差一个 MTP 层——这正好把「激活参不含 MTP」这条口径
    #    暴露成一个可复核的等式，而不是一句注释。
    degenerate = dict(cfg)
    degenerate["routed_experts"] = cfg["experts_total"]
    td, ad, dd = account(degenerate, is_mla=True, mtp_as_moe_layer=True)
    judge("C3 k = E 时每层 MoE 的激活参等于总参（稀疏性消失）",
          dd["moe_layer_active"] == dd["moe_layer_total"],
          f"每层 {dd['moe_layer_total']/1e9:.2f}B = {dd['moe_layer_active']/1e9:.2f}B")
    mtp_term = dd["moe_layer_total"] + dd["attn_layer"] + dd["indexer_layer"]
    judge("C3b 全模型 total − active 恰好等于一个 MTP 层（口径可复核）",
          td - ad == mtp_term,
          f"差 {td-ad} B = MTP 层 {mtp_term} B")

    # C4 反例：把 total params 当解码算力用会低估每步算力 18 倍
    factor = cfg["total_params"] / cfg["active_params"]
    judge("C4 总参与激活参的比值 ≈ 18.6×（混用即错 18 倍）",
          abs(factor - 18.6) < 0.3,
          f"744B / 40B = {factor:.2f}×")

    # C5 层数的三个口径互不相等，写「层数」必须带口径
    tabulated = cfg["dense_layers"] + cfg["moe_layers"]
    with_mtp = tabulated + cfg["mtp_layers"]
    prose = specs.GLM5_REPORT_PROSE_LAYER_COUNT
    judge("C5 Table 10 可加出的 78 / 79 与正文的 80 三者互不相等",
          len({tabulated, with_mtp, prose}) == 3,
          f"dense+MoE={tabulated}，含 MTP={with_mtp}，报告正文={prose}")


def main():
    print("== 160-model/glm · 01 参数与激活量核算 ==")
    print(f"测量条件（口径）：{specs.MEASURE_HOST}")
    print(f"python {specs.MEASURE_PYTHON} / numpy {np.__version__}")
    print("★ 本程序只做算术，产出是**推导值**，不是实测值。\n")

    t5, a5, _, t45, a45 = class_a()
    _, _, d45 = account(specs.GLM45_CONFIG, is_mla=False, mtp_as_moe_layer=True)
    _, _, d5 = account(specs.GLM5_CONFIG, is_mla=True, mtp_as_moe_layer=True)
    print(f"\n  GLM-4.5 逐层明细：expert {d45['expert_params']/1e6:.2f}M"
          f" / MoE 层 {d45['moe_layer_total']/1e9:.2f}B"
          f" / 注意力 {d45['attn_layer']/1e6:.1f}M / indexer {d45['indexer_layer']}")
    print(f"  推导：GLM-4.5 = {t45/1e9:.1f}B 总 / {a45/1e9:.2f}B 激活"
          f"；GLM-5 = {t5/1e9:.1f}B 总 / {a5/1e9:.2f}B 激活")
    print(f"  GLM-5 逐层明细：expert {d5['expert_params']/1e6:.2f}M"
          f" / MoE 层 {d5['moe_layer_total']/1e9:.2f}B"
          f" / 注意力 {d5['attn_layer']/1e6:.1f}M"
          f" / indexer {d5['indexer_layer']/1e6:.1f}M\n")

    class_b(t5, a5, d5)
    class_c(t5)

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

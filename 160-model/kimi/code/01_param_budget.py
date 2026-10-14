#!/usr/bin/env python3
"""160-model/kimi 编号程序 01 —— Kimi K2 / K3 的参数与激活量核算（推导，不是实测）。

配对文档：`160-model/kimi/01-technical-route.md`

★ 形态（CONTRIBUTING §2 判定表第二种）：本机默认 python3 可跑（`math` + `numpy`），
  三类断言在程序内部，失败非零退出：
    A 算法正确性 —— 显式矩阵清单（独立参照）与闭式公式对拍，再与官方印值对拍
    B 代价模型   —— LatentMoE 的「宽度分离」账、激活量分解、MXFP4 的字节账
    C 边界与反例 —— 「不用 latent 直接放大专家池」「忽略 scale 开销」都会算错
  产出**一律是推导值**：不加载权重、不 import 推理框架、不测硬件。
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
# A 算法正确性
# ---------------------------------------------------------------------------

def ref_params_from_shapes(shapes):
    """独立参照：显式列出权重矩阵形状再数元素，不经过任何闭式公式。"""
    return int(sum(int(np.prod(s, dtype=np.int64)) for s in shapes))


def ffn_params(d_model, d_ff):
    """一个 SwiGLU / SiTU-GLU FFN 的参数（gate / up / down）。"""
    return 3 * d_model * d_ff


def moe_layer_total(d_model, d_ff, routed, shared):
    return (routed + shared) * ffn_params(d_model, d_ff)


def latent_moe_layer_total(d_model, latent, d_expert, routed, shared):
    """LatentMoE 一层：shared 走满宽；routed 在 latent 上算；另有共享的 W↓ / W↑。"""
    shared_params = shared * ffn_params(d_model, d_expert)
    routed_params = routed * ffn_params(latent, d_expert)
    down_up = ref_params_from_shapes([(latent, d_model), (d_model, latent)])
    return shared_params + routed_params + down_up


def latent_moe_layer_active(d_model, latent, d_expert, k, shared):
    return (shared * ffn_params(d_model, d_expert)
            + k * ffn_params(latent, d_expert)
            + ref_params_from_shapes([(latent, d_model), (d_model, latent)]))


def class_a():
    print("A 算法正确性（独立参照 vs 闭式公式）")

    cases = [(7168, 2048), (7168, 3072), (3584, 3072), (64, 32), (1, 1)]
    ok = all(ffn_params(d, m) == ref_params_from_shapes([(m, d), (m, d), (d, m)])
             for d, m in cases)
    judge("A1 FFN 的 3·d·m 等于显式矩阵元素数之和", ok,
          f"核对 {len(cases)} 组 (d_model, d_ff)，含 (1,1) 退化组")

    # ---- K3 ----
    c3 = specs.K3_CONFIG
    per_layer3 = latent_moe_layer_total(c3["hidden_dim"], c3["latent_moe_dim"],
                                        c3["expert_intermediate"],
                                        c3["experts_total"], c3["shared_experts"])
    act_layer3 = latent_moe_layer_active(c3["hidden_dim"], c3["latent_moe_dim"],
                                         c3["expert_intermediate"],
                                         c3["routed_experts"], c3["shared_experts"])
    ffn3 = c3["layers"] * per_layer3
    rel3 = abs(ffn3 - c3["total_params"]) / c3["total_params"]
    judge("A2 K3 的 MoE FFN 单独就能把总参算到 2.78T（差 < 1%）", rel3 < 0.01,
          f"每层 {per_layer3/1e9:.3f}B × {c3['layers']} 层 = {ffn3/1e12:.4f}T vs 官方 "
          f"{c3['total_params']/1e12:.2f}T（偏差 {rel3*100:+.2f}%）")

    judge("A3 K3 的 latent 宽度恰是 hidden 的 0.5×",
          c3["latent_moe_dim"] * 2 == c3["hidden_dim"],
          f"{c3['latent_moe_dim']} × 2 = {c3['hidden_dim']}")

    # ---- K2 ----
    c2 = specs.K2_CONFIG
    per_layer2 = moe_layer_total(c2["hidden_dim"], c2["expert_intermediate"],
                                 c2["experts_total"], c2["shared_experts"])
    act_layer2 = (c2["routed_experts"] + c2["shared_experts"]) * ffn_params(
        c2["hidden_dim"], c2["expert_intermediate"])
    dense_ffn2 = ffn_params(c2["hidden_dim"], c2["expert_intermediate"])
    ffn2_all_moe = c2["layers"] * per_layer2
    ffn2_with_dense = (c2["layers"] - c2["dense_layers"]) * per_layer2 + dense_ffn2
    # ★ 两种读法都要再加一层「注意力 + embedding」的余量才能夹住官方 1.04T。
    #   本仓库**不猜** K2 的 MLA 具体维度，只给一个宽松上界 200M/层作为余量，
    #   并把这个上界明确写进判据里（它是假设，不是规格）。
    attn_allowance = 200e6 * c2["layers"]
    upper = ffn2_all_moe + attn_allowance
    judge("A4 K2 的两种读法 + 宽松注意力余量共同夹住官方 1.04T",
          ffn2_with_dense < c2["total_params"] < upper,
          f"{ffn2_with_dense/1e12:.4f}T（60 MoE + 1 dense）< 1.04T < "
          f"{upper/1e12:.4f}T（61 层全 MoE + 每层 ≤200M 注意力余量）；"
          f"跨度 {(upper-ffn2_with_dense)/1e9:.1f}B")
    return per_layer3, act_layer3, per_layer2, act_layer2, ffn3, ffn2_all_moe, ffn2_with_dense


# ---------------------------------------------------------------------------
# B 代价模型
# ---------------------------------------------------------------------------

def class_b(act_layer3, act_layer2, per_layer3, ffn3):
    print("\nB 代价模型")
    c2, c3 = specs.K2_CONFIG, specs.K3_CONFIG

    # B1 官方口径的稀疏度 = 总专家 / 每 token 激活
    judge("B1 官方『sparsity』定义复算：K3 = 56、K2 = 48",
          c3["experts_total"] / c3["routed_experts"] == 56
          and c2["experts_total"] / c2["routed_experts"] == 48,
          f"K3 {c3['experts_total']}/{c3['routed_experts']} = {c3['experts_total']/c3['routed_experts']:.0f}；"
          f"K2 {c2['experts_total']}/{c2['routed_experts']} = {c2['experts_total']/c2['routed_experts']:.0f}")

    # B2 激活量分解：FFN 占多少，注意力是余量
    for name, cfg, act_layer, ffn in (("K2", c2, act_layer2, None),
                                      ("K3", c3, act_layer3, ffn3)):
        ffn_active = cfg["layers"] * act_layer
        residual = cfg["active_params"] - ffn_active
        per_layer_attn = residual / cfg["layers"]
        print(f"  [信息] {name}: FFN 激活 {ffn_active/1e9:.2f}B（占官方激活 "
              f"{ffn_active/cfg['active_params']*100:.1f}%）→ 反解注意力余量 "
              f"{residual/1e9:.2f}B（每层 {per_layer_attn/1e6:.0f}M）")
    ffn_act2 = c2["layers"] * act_layer2
    ffn_act3 = c3["layers"] * act_layer3
    judge("B2 两个模型的反解注意力余量都落在 100–500M/层（MLA / KDA 的量级）",
          all(100e6 < (cfg["active_params"] - la) / cfg["layers"] < 500e6
              for cfg, la in ((c2, ffn_act2), (c3, ffn_act3))),
          f"K2 {(c2['active_params']-ffn_act2)/c2['layers']/1e6:.0f}M/层；"
          f"K3 {(c3['active_params']-ffn_act3)/c3['layers']/1e6:.0f}M/层")

    # B3 MXFP4 的字节账：4 bit 元素 + 每 32 个元素一个 8 bit 的 E8M0 scale
    eff_bits = 4 + 8 / 32
    routed_params3 = c3["layers"] * c3["experts_total"] * ffn_params(
        c3["latent_moe_dim"], c3["expert_intermediate"])
    expert_bytes = routed_params3 * eff_bits / 8
    rest_params = c3["total_params"] - routed_params3
    rest_bytes = rest_params * 2                      # 非 expert 按 BF16 估
    total_bytes = expert_bytes + rest_bytes
    bf16_bytes = c3["total_params"] * 2
    judge("B3 MXFP4（4.25 bit/参数）+ 非 expert BF16 把 K3 压到 1.6 TB 以内",
          total_bytes < 1.6e12,
          f"expert {expert_bytes/1e12:.3f} TB（有效 {eff_bits} bit = 4 + 8/32）+ 其余 "
          f"{rest_bytes/1e9:.0f} GB = {total_bytes/1e12:.3f} TB"
          f"（全 BF16 需 {bf16_bytes/1e12:.2f} TB，{bf16_bytes/total_bytes:.2f}×）")

    # B4 LatentMoE 的真实收益：routed 路径的「载荷宽度」减半
    judge("B4 LatentMoE 把 routed 路径的宽度减半（省的是通信与 expert 权重流量）",
          c3["latent_ratio"] == 0.5,
          f"ℓ/d = {c3['latent_moe_dim']}/{c3['hidden_dim']} = {c3['latent_ratio']}；"
          f"每 token 每个 routed expert 的载荷从 d 降到 ℓ")
    return eff_bits, routed_params3


# ---------------------------------------------------------------------------
# C 边界与反例
# ---------------------------------------------------------------------------

def class_c(eff_bits, routed_params3):
    print("\nC 边界与反例")
    c3 = specs.K3_CONFIG

    # C1 不用 latent 直接放大专家池：参数量几乎翻倍
    full_width = c3["layers"] * c3["experts_total"] * ffn_params(
        c3["hidden_dim"], c3["expert_intermediate"])
    judge("C1 routed expert 走满宽 d 会让专家参数变成 ~2×（896 专家不可承受）",
          full_width / routed_params3 > 1.9,
          f"满宽 {full_width/1e12:.2f}T vs latent {routed_params3/1e12:.2f}T → "
          f"{full_width/routed_params3:.2f}×（而总参官方只有 2.78T）")

    # C2 忽略 E8M0 scale 的开销会低估多少
    naive = routed_params3 * 0.5
    real = routed_params3 * eff_bits / 8
    judge("C2 把 MXFP4 当成「正好 0.5 字节/参数」会低估 1.5% 以上",
          (real - naive) / naive > 0.015,
          f"0.5 B/参数 → {naive/1e12:.4f} TB；含 scale → {real/1e12:.4f} TB"
          f"（低估 {(real-naive)/naive*100:.2f}%，即 {(real-naive)/1e9:.1f} GB）")

    # C3 latent 不是免费的：ℓ 变窄意味着 routed expert 表达力下降
    judge("C3 LatentMoE 把 routed 宽度砍半，代价是每个 expert 的表达力下降",
          c3["latent_moe_dim"] < c3["hidden_dim"],
          f"ℓ = {c3['latent_moe_dim']} < d = {c3['hidden_dim']}；"
          f"官方为此加了 RMSNorm + SiTU-GLU + Quantile Balancing 三个稳定器")

    # C4 激活量不能只算 routed expert
    routed_active = c3["routed_experts"] * ffn_params(c3["latent_moe_dim"],
                                                      c3["expert_intermediate"])
    shared_active = c3["shared_experts"] * ffn_params(c3["hidden_dim"],
                                                      c3["expert_intermediate"])
    judge("C4 每层激活里 shared expert 与 W↓/W↑ 合计占 20% 以上，不能只算 routed",
          (shared_active + 2 * c3["latent_moe_dim"] * c3["hidden_dim"]) / (
              routed_active + shared_active + 2 * c3["latent_moe_dim"] * c3["hidden_dim"]) > 0.20,
          f"routed {routed_active/1e6:.1f}M / shared {shared_active/1e6:.1f}M / "
          f"W↓W↑ {2*c3['latent_moe_dim']*c3['hidden_dim']/1e6:.1f}M")

    # C5 官方「2.5× scaling 效率」不可由架构复算
    judge("C5 官方 2.5× scaling 效率是经验结果，本仓库不复核、也不反推",
          specs.K3_TRAINING["scaling_efficiency_vs_k2"] == 2.5
          and "不复核" in specs.K3_TRAINING["scaling_efficiency_note"],
          "记为引用值；架构账只能给出参数量与激活量，给不出 held-out loss 曲线")


def main():
    print("== 160-model/kimi · 01 参数与激活量核算 ==")
    print(f"测量条件（口径）：{specs.MEASURE_HOST}")
    print(f"python {specs.MEASURE_PYTHON} / numpy {np.__version__}")
    print("★ 只做算术，产出是**推导值**，不是实测值。\n")

    per3, act3, per2, act2, ffn3, allmoe, withdense = class_a()
    print(f"\n  K3 每层 MoE：总参 {per3/1e9:.3f}B / 激活 {act3/1e6:.1f}M"
          f"；MoE FFN 合计 {ffn3/1e12:.4f}T")
    print(f"  K2 每层 MoE：总参 {per2/1e9:.3f}B / 激活 {act2/1e6:.1f}M"
          f"；FFN 两种读法 {withdense/1e12:.4f}T 与 {allmoe/1e12:.4f}T\n")

    eff_bits, routed_params3 = class_b(act3, act2, per3, ffn3)
    print()
    class_c(eff_bits, routed_params3)

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

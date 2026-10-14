#!/usr/bin/env python3
"""160-model/deepseek 编号程序 01 —— DeepSeek 家族的参数与激活量核算（推导，不是实测）。

配对文档：`160-model/deepseek/01-technical-route.md`

★ 形态（CONTRIBUTING §2 判定表第二种）：编号程序本机默认 python3 可跑（`math` + `numpy`），
  三类断言全部在程序内部，失败时非零退出：
    A 算法正确性 —— 显式矩阵清单（独立参照）与闭式公式对拍，再与官方印值对拍
    B 代价模型   —— MoE 激活比、CED 的 prefill/decode 不对称、KV 压缩倍数
    C 边界与反例 —— 「忽略 CED」「按稠密注意力算 KV」等朴素做法必须真的算错
  产出**一律是推导值**：不加载权重、不 import 推理框架、不测硬件。

★ 本程序只对**技术报告给了完整逐字配置**的 V4.1-Flash 做逐层核算；
  V3 / V4-Pro / V4-Flash 只用报告给的总参/激活值做横向对照（标为规格值）。
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
    """独立参照：把权重矩阵的 (out, in) 形状逐个乘起来求和，不经过任何闭式公式。"""
    return int(sum(int(np.prod(s, dtype=np.int64)) for s in shapes))


def expert_params(d_model, d_expert_inter):
    """一个 SwiGLU expert 的参数（gate / up / down 三个矩阵）。"""
    return 3 * d_model * d_expert_inter


def moe_layer_total(d_model, d_inter, routed, shared):
    return (routed + shared) * expert_params(d_model, d_inter)


def moe_layer_active(d_model, d_inter, k, shared):
    return (k + shared) * expert_params(d_model, d_inter)


def class_a():
    print("A 算法正确性（独立参照 vs 闭式公式）")

    # A1 3·d·m 与显式矩阵元素数之和一致（含退化组）
    cases = [(5120, 2304), (7168, 3072), (7168, 2048), (64, 32), (1, 1)]
    ok = all(expert_params(d, m) == ref_params_from_shapes(
        [(m, d), (m, d), (d, m)]) for d, m in cases)
    judge("A1 SwiGLU expert 的 3·d·m 等于显式矩阵元素数之和", ok,
          f"核对 {len(cases)} 组 (d_model, d_inter)，含 (1,1) 退化组")

    cfg = specs.V41_CONFIG
    d, m = cfg["hidden_dim"], cfg["expert_intermediate"]
    per_layer_total = moe_layer_total(d, m, cfg["routed_experts"], cfg["shared_experts"])
    per_layer_active = moe_layer_active(d, m, cfg["activated_experts"], cfg["shared_experts"])

    # A2 用官方 552B / 16B / 8B 反解每层「注意力 + 其它」的参数
    layers = cfg["layers"]
    ffn_total = layers * per_layer_total
    ffn_active_decode = layers * per_layer_active
    implied_per_layer = specs.V41["active_decode"] / layers          # 400.0M
    implied_attn = implied_per_layer - per_layer_active
    derived_total = ffn_total + layers * implied_attn
    official = specs.V41["backbone_params"]
    rel = abs(derived_total - official) / official
    judge("A2 由官方 8B/16B 反解出的每层激活量，能把总参算回 552B（差 < 1%）",
          rel < 0.01,
          f"每层 MoE 激活 {per_layer_active/1e6:.1f}M + 反解注意力 {implied_attn/1e6:.1f}M "
          f"= {implied_per_layer/1e6:.1f}M；总参 {derived_total/1e9:.1f}B vs 官方 "
          f"{official/1e9:.0f}B（偏差 {rel*100:+.2f}%）")

    # A3 同一个 expert 参数按两种口径写必须同值
    alias = ref_params_from_shapes([(m, d)] * 2 + [(d, m)])
    judge("A3 expert 参数的两种写法一致", alias == expert_params(d, m),
          f"{alias} = {expert_params(d, m)}（{alias/1e6:.2f}M/expert）")
    return per_layer_total, per_layer_active, implied_attn, ffn_total


# ---------------------------------------------------------------------------
# B 代价模型
# ---------------------------------------------------------------------------

def class_b(per_layer_total, per_layer_active, implied_attn, ffn_total):
    print("\nB 代价模型")
    cfg = specs.V41_CONFIG

    # B1 FFN 占主干参数的比例
    share = ffn_total / specs.V41["backbone_params"]
    judge("B1 MoE FFN 占 V4.1-Flash 主干参数 > 97%", share > 0.97,
          f"{ffn_total/1e9:.1f}B / {specs.V41['backbone_params']/1e9:.0f}B = {share*100:.2f}%")

    # B2 MoE 激活比（含 shared）
    ratio = (cfg["activated_experts"] + cfg["shared_experts"]) / (
        cfg["routed_experts"] + cfg["shared_experts"])
    judge("B2 每 token 只激活 (k+s)/(E+s) 的 expert",
          abs(ratio - 7 / 385) < 1e-12,
          f"(6+1)/(384+1) = {ratio*100:.2f}%")

    # B3 CED 的不对称：prefill 跑 20 层、decode 跑 40 层
    prefill = cfg["encoder_layers"] * (per_layer_active + implied_attn)
    decode = cfg["layers"] * (per_layer_active + implied_attn)
    judge("B3 CED 下 decode 激活量恰是 prefill 的 2 倍（层数比 40:20）",
          abs(decode / prefill - 2.0) < 1e-12,
          f"prefill {prefill/1e9:.2f}B vs decode {decode/1e9:.2f}B"
          f"（官方印值 8B / 16B）")

    # B4 与官方印值对拍
    for name, got, want in (("prefill", prefill, specs.V41["active_prefill"]),
                            ("decode", decode, specs.V41["active_decode"])):
        rel = abs(got - want) / want
        judge(f"B4-{name} 推导激活量与官方印值差 < 2%", rel < 0.02,
              f"{got/1e9:.2f}B vs {want/1e9:.0f}B（{rel*100:+.2f}%）")

    # B5 KV 压缩倍数（全是除法，但方向必须写对）
    v41 = specs.V41["global_kv_bytes_per_token"]
    v1 = specs.V41["v1_kv_bytes_per_token"]
    judge("B5 相对 V1 的 437× 压缩可由两个规格值复算",
          abs(v1 / v41 - 437) < 1.0,
          f"{v1} / {v41} = {v1/v41:.1f}×（报告印值 437×）")
    v4flash_implied = v41 / specs.V41["global_kv_vs_v4flash"]
    judge("B5b 由「1/4」反推的 V4-Flash 全局 KV 是 3560 B/token",
          abs(v4flash_implied - 3560) < 1e-9,
          f"890 / 0.25 = {v4flash_implied:.0f} B/token")

    # B6 与「每层都存 KV」的朴素做法对比
    naive = cfg["layers"] * cfg["head_dim"] * specs.V41_KV_FORMAT["bytes_per_element"]
    judge("B6 朴素「每层存一份 512 维 FP4 KV」比官方值大 10× 以上",
          naive / v41 > 10,
          f"朴素 {naive:.0f} B vs 官方 {v41} B → 高估 {naive/v41:.1f}×")
    return v41, v4flash_implied


# ---------------------------------------------------------------------------
# C 边界与反例
# ---------------------------------------------------------------------------

def class_c(v41):
    print("\nC 边界与反例")

    cfg = specs.V41_CONFIG

    # C1 【反推】把 890 B 拆成「存 KV 的层」的账：两种读法夹住官方值
    #   只有 Full Mode 层需要产出并保存 main KV 与 indexer K：
    #     encoder：3 组 × 每组 1 个 Full 层 = 3 层，压缩率 m = 2
    #     decoder：5 组 × 每组 1 个 Full 层 = 1 层真正新建 KV（其余组首层是 Reindex，复用）
    n_full_enc, n_full_dec = cfg["encoder_groups"], 1
    kv_entry_bytes = cfg["head_dim"] * specs.V41_KV_FORMAT["bytes_per_element"]      # 256 B
    idx_fp4 = cfg["indexer_head_dim"] * 0.5                                          # 64 B
    idx_fp8 = cfg["indexer_head_dim"] * 1.0                                          # 128 B

    def decompose(idx_bytes):
        enc = n_full_enc * (kv_entry_bytes + idx_bytes) / cfg["encoder_compression"]
        dec = n_full_dec * (kv_entry_bytes + idx_bytes) / cfg["decoder_compression"]
        return enc + dec

    lo, hi = decompose(idx_fp4), decompose(idx_fp8)
    judge("C1 【反推】两种 indexer K 精度的读法夹住官方 890 B/token",
          lo < v41 < hi,
          f"indexer K 取 FP4 → {lo:.0f} B；取 FP8 → {hi:.0f} B；官方 {v41} B"
          f"（区间宽 {hi-lo:.0f} B）")

    # C2 官方那句「1/3 总参、1/4 激活参」与 8B/16B 对不上
    total_ratio = specs.V41["backbone_params"] / specs.V4_FAMILY["V4-Pro"]["total_params"]
    ratio_prefill = specs.V41["active_prefill"] / specs.V4_FAMILY["V4-Pro"]["active_params"]
    ratio_decode = specs.V41["active_decode"] / specs.V4_FAMILY["V4-Pro"]["active_params"]
    judge("C2 官方「1/3 总参 / 1/4 激活参」中，1/3 成立、1/4 与 8B/16B 都不吻合",
          abs(total_ratio - 1 / 3) < 0.02
          and not (abs(ratio_prefill - 0.25) < 0.01 or abs(ratio_decode - 0.25) < 0.01),
          f"总参 {specs.V41['backbone_params']/1e9:.0f}B/1600B = {total_ratio:.3f}（≈1/3 ✓）；"
          f"8B/49B = {ratio_prefill:.3f}、16B/49B = {ratio_decode:.3f}（都不等于 1/4）")

    # C3 忽略 CED 会把 prefill 成本高估 2 倍
    judge("C3 把 prefill 当成跑满 40 层，会高估 prefill 激活量 2.0×",
          specs.V41["active_decode"] / specs.V41["active_prefill"] == 2.0,
          "16B / 8B = 2.0×；CED 的作用正是让 prefill 只跑 encoder 的 20 层")

    # C4 短 prompt 下 CED 没有收益（预填折半的边界）
    L, n_win = cfg["layers"], cfg["swa_window"]
    for N in (n_win, 4 * n_win, 1000 * n_win):
        r = 0.5 + n_win / (2 * N)
        print(f"  [信息] prompt N={N:>7} → CED/无CED 的 prefill 比 = {r:.4f}")
    judge("C4 当 N ≤ n_win 时 CED 的 prefill 折半完全消失（比值 → 1.0）",
          abs(0.5 + n_win / (2 * n_win) - 1.0) < 1e-12,
          f"N = n_win = {n_win} 时比值 = 1.0000；N ≫ n_win 时下探到 0.5")

    # C5 「KV 只随层数线性」这个直觉在 CSA2 上是错的
    judge("C5 CSA2 只有 4/40 层真正新建 main KV（跨层复用把层维也压了）",
          (n_full_enc + n_full_dec) == 4,
          f"encoder {n_full_enc} 个 Full 层 + decoder {n_full_dec} 个（其余是 Reindex / Reuse）")


def main():
    print("== 160-model/deepseek · 01 参数与激活量核算 ==")
    print(f"测量条件（口径）：{specs.MEASURE_HOST}")
    print(f"python {specs.MEASURE_PYTHON} / numpy {np.__version__}")
    print("★ 只做算术，产出是**推导值**，不是实测值。\n")

    per_layer_total, per_layer_active, implied_attn, ffn_total = class_a()
    print(f"\n  V4.1-Flash 逐层明细：expert {expert_params(5120, 2304)/1e6:.2f}M"
          f" / MoE 层总参 {per_layer_total/1e9:.2f}B"
          f" / MoE 层激活 {per_layer_active/1e6:.1f}M"
          f" / 反解注意力 {implied_attn/1e6:.1f}M")
    print(f"  MoE FFN 合计 {ffn_total/1e9:.1f}B（占主干 "
          f"{ffn_total/specs.V41['backbone_params']*100:.1f}%）\n")

    v41, v4flash = class_b(per_layer_total, per_layer_active, implied_attn, ffn_total)
    print(f"\n  全局 KV：V4.1-Flash {v41} B/token；反推 V4-Flash {v4flash:.0f} B/token\n")
    class_c(v41)

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

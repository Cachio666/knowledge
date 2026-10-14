#!/usr/bin/env python3
"""160-model/glm 编号程序 03 —— 部署账：权重比特、KV 容量、EP 通信、批处理（推导）。

配对文档：`160-model/glm/03-serving-efficiency-and-deployment.md`

★ 形态（CONTRIBUTING §2）：本机默认 python3 可跑（只用 `math` 与 `numpy`），
  三类断言全部在本程序内部，失败时非零退出：
    A 算法正确性 —— KV / 权重的字节账用显式数组与逐组件求和两条独立路径对拍
    B 代价模型   —— 三种精度的显存、decode 每步字节、吞吐上限、KV/权重交叉点、
                    EP 通信量、批处理的次线性
    C 边界与反例 —— 「全模型 4bit」「套 GQA 公式」「1M 能靠单序列扛」都会失败
  `verify.py` 不需要。产出**一律是推导值**：本程序不加载权重、不测硬件。

★ 带宽是**引用值/量级参照**，不是本机实测：
  `BW_REFERENCE_TBps = 3.35` 取 H100 SXM 量级，用来把字节账换成「每秒多少 token」的
  量级判断。换机器请改这个常量；本仓库**不声称**核对过该白皮书原文。
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
# 规格与口径
# ---------------------------------------------------------------------------

G5 = specs.GLM5_CONFIG
ROPE = 64
LATENT = G5["kv_lora_dim"]
KV_DIM_MLA = LATENT + ROPE                                  # 576
KV_DIM_GQA8 = 2 * specs.GLM45_CONFIG["kv_heads"] * specs.GLM45_CONFIG["attn_head_dim"]  # 2048
KV_LAYERS = G5["dense_layers"] + G5["moe_layers"]           # 78（不含 MTP）
CTX_1M = specs.GLM52_CONTEXT                                # 1 048 576（config 口径，不是 10^6）
TOTAL_PARAMS = G5["total_params"]                           # 744B（官方口径，不含 embedding/输出层）
ACTIVE_PARAMS = G5["active_params"]                         # 40B

# expert 参数占比：由 01 的逐层核算得出（727.6B / 752.2B）。
# 把它当**比例**用在官方 744B 上，而不是直接用 01 的绝对值——这样三档精度
# 的字节账共用同一个总参口径，避免「内部推导值 × 官方便宜」的混用。
EXPERT_PARAM_SHARE = 75 * (256 + 1) * 3 * 6144 * 2048 / 752_246_194_176
NON_EXPERT_SHARE = 1.0 - EXPERT_PARAM_SHARE

# expert 在**激活**参数里的占比：75 层 × (8+1) 个 expert / 主路径激活量
EXPERT_ACT_SHARE = (75 * 9 * 3 * 6144 * 2048) / 40_237_793_280

# 来源：【引用值/量级参照】H100 SXM 级 HBM 带宽。本仓库未逐字核对白皮书。
BW_REFERENCE_TBps = 3.35


def kv_bytes(L, kv_dim, layers=KV_LAYERS, bytes_per=2):
    """KV 字节。注意 MLA 的 kv_dim 是「latent + RoPE」而**不乘 2**（K/V 共享 latent）。"""
    return L * layers * kv_dim * bytes_per


def weights_bytes(total, b_per_param, expert_share=EXPERT_PARAM_SHARE):
    """按组件分配比特：expert 用 b_expert，其余用 b_rest。"""
    b_expert, b_rest = b_per_param
    return total * expert_share * b_expert + total * (1 - expert_share) * b_rest


PRECISIONS = {
    "BF16": (2.0, 2.0),
    "FP8/INT8（全部）": (1.0, 1.0),
    "混合：expert W4 + 其余 W8": (0.5, 1.0),
}


def active_weight_bytes(b_per_param):
    b_expert, b_rest = b_per_param
    return (ACTIVE_PARAMS * EXPERT_ACT_SHARE * b_expert
            + ACTIVE_PARAMS * (1 - EXPERT_ACT_SHARE) * b_rest)


def step_bytes(L, b_per_param, kv_bytes_per, batch=1):
    """一个 decode 步要搬的字节：权重一份 + batch × 每条序列的 KV。"""
    return active_weight_bytes(b_per_param) + batch * kv_bytes(L, KV_DIM_MLA, bytes_per=kv_bytes_per)


def tokens_per_sec(L, b_per_param, kv_bytes_per, batch=1, bw_tbps=BW_REFERENCE_TBps):
    """batch=1 时直接是单流吞吐；batch>1 时分母是「每步总字节」，再乘 batch 得聚合吞吐。"""
    per_step = step_bytes(L, b_per_param, kv_bytes_per, batch)
    steps_per_sec = bw_tbps * 1e12 / per_step
    return steps_per_sec * batch


# ---------------------------------------------------------------------------
# A 算法正确性
# ---------------------------------------------------------------------------

def class_a():
    print("A 算法正确性")

    # A1 KV 字节：显式数组 nbytes 与闭式对拍（MLA 与 GQA 各一条）
    L = 512
    mla = np.zeros((L, KV_LAYERS, KV_DIM_MLA), dtype=np.float16)
    gqa = np.zeros((L, KV_LAYERS, KV_DIM_GQA8), dtype=np.float16)
    judge("A1 KV 字节：显式数组 nbytes == 闭式（MLA 不乘 2）",
          mla.nbytes == kv_bytes(L, KV_DIM_MLA)
          and gqa.nbytes == kv_bytes(L, KV_DIM_GQA8),
          f"MLA {mla.nbytes} B（{KV_DIM_MLA} 维） / GQA {gqa.nbytes} B（{KV_DIM_GQA8} 维）")

    # A2 权重字节：逐组件求和 vs 占比闭式（两条独立路径）
    total = 1000  # 用一个 1000 参数的玩具模型，保证是整数
    n_expert, n_rest = 967, 33
    componentwise = n_expert * 0.5 + n_rest * 1.0
    by_share = weights_bytes(total, (0.5, 1.0), expert_share=n_expert / total)
    judge("A2 混合精度权重字节：逐组件求和 == 占比闭式",
          abs(componentwise - by_share) < 1e-9,
          f"{componentwise} vs {by_share}")

    # A3 激活权重字节：expert 走 4bit、其余 8bit，且两项都不为零
    a_mixed = active_weight_bytes((0.5, 1.0))
    a_bf16 = active_weight_bytes((2.0, 2.0))
    judge("A3 激活权重字节随精度单调：混合 < FP8 < BF16",
          a_mixed < active_weight_bytes((1.0, 1.0)) < a_bf16,
          f"混合 {a_mixed/1e9:.2f} GB < FP8 {ACTIVE_PARAMS/1e9:.1f} GB < BF16 {a_bf16/1e9:.1f} GB")


# ---------------------------------------------------------------------------
# B 代价模型
# ---------------------------------------------------------------------------

def class_b():
    print("\nB 代价模型")

    # B1 三档精度的权重字节
    rows = {name: weights_bytes(TOTAL_PARAMS, bp) for name, bp in PRECISIONS.items()}
    judge("B1 混合精度权重落在 350–400 GB（官方 744B 口径）",
          350e9 < rows["混合：expert W4 + 其余 W8"] < 400e9,
          " / ".join(f"{k.split('：')[0]} {v/1e9:.1f} GB" for k, v in rows.items()))

    # 需要几张 80GB 卡才装得下权重
    cards = {k: math.ceil(v / 80e9) for k, v in rows.items()}
    print(f"  [信息] 按 80 GB/卡只放权重：{cards}")

    # B2 1M 上下文的每步字节与吞吐
    kv_bf16_1m = kv_bytes(CTX_1M, KV_DIM_MLA, bytes_per=2)
    kv_int8_1m = kv_bytes(CTX_1M, KV_DIM_MLA, bytes_per=1)
    bf16_step = step_bytes(CTX_1M, (2.0, 2.0), 2)
    mixed_step = step_bytes(CTX_1M, (0.5, 1.0), 1)
    tps_bf16 = tokens_per_sec(CTX_1M, (2.0, 2.0), 2)
    tps_mixed = tokens_per_sec(CTX_1M, (0.5, 1.0), 1)
    judge("B2 1M 上下文下混合精度把单流吞吐提到 BF16 的 2.2–2.5×",
          2.2 < tps_mixed / tps_bf16 < 2.5,
          f"BF16 {bf16_step/1e9:.1f} GB/步 → {tps_bf16:.1f} tok/s；"
          f"混合+INT8KV {mixed_step/1e9:.1f} GB/步 → {tps_mixed:.1f} tok/s")

    # B3 KV 与权重的交叉点
    w_mixed = active_weight_bytes((0.5, 1.0))
    per_token_kv_int8 = KV_LAYERS * KV_DIM_MLA * 1
    cross = w_mixed / per_token_kv_int8
    judge("B3 交叉点落在 500K–700K token（超过它，KV 搬运量 > 权重）",
          500e3 < cross < 700e3,
          f"L* = {w_mixed/1e9:.2f} GB / {per_token_kv_int8/1024:.1f} KiB每token "
          f"= {cross/1e3:.0f}K；1M 处 KV {kv_int8_1m/1e9:.1f} GB > 权重 {w_mixed/1e9:.1f} GB")

    # B4 EP 通信：每 token 每 MoE 层（最坏情况 k 个 expert 落在 k 个 rank）
    per_layer = 2 * G5["routed_experts"] * G5["hidden_dim"] * 2
    per_token = per_layer * G5["moe_layers"]
    judge("B4 EP 通信量：每层约 192 KiB、每 token 约 14.7 MB",
          abs(per_layer / 1024 - 192.0) < 0.1,
          f"{per_layer} B = {per_layer/1024:.1f} KiB/层 × {G5['moe_layers']} = "
          f"{per_token/1e6:.1f} MB/token")

    # B5 与第三方吞吐的交叉验证：短上下文上限应高于 115 TPS，长上下文低于它
    short_ctx = tokens_per_sec(0, (0.5, 1.0), 1)
    judge("B5 推导的吞吐区间包住第三方给的 115 TPS（量级交叉验证）",
          tps_mixed < 115 < short_ctx,
          f"1M 推导 {tps_mixed:.1f} < 115 < 短上下文推导 {short_ctx:.1f} tok/s")

    # B6 批处理的次线性：KV 随 batch 线性涨，权重被摊薄
    t1 = tokens_per_sec(CTX_1M, (0.5, 1.0), 1, batch=1)
    t8 = tokens_per_sec(CTX_1M, (0.5, 1.0), 1, batch=8)
    kv_share8 = 8 * kv_int8_1m / (w_mixed + 8 * kv_int8_1m)
    judge("B6 1M 上下文 batch=8 的聚合吞吐不足 batch=1 的 2×（KV 主导）",
          t8 / t1 < 2.0,
          f"batch1 {t1:.1f} → batch8 {t8:.1f} tok/s（{t8/t1:.2f}×）；"
          f"batch8 时 KV 占总字节 {kv_share8*100:.1f}%")


# ---------------------------------------------------------------------------
# C 边界与反例
# ---------------------------------------------------------------------------

def class_c():
    print("\nC 边界与反例")

    # C1 「全模型 4bit」会低估真实的混合精度字节
    all_w4 = TOTAL_PARAMS * 0.5
    mixed = weights_bytes(TOTAL_PARAMS, (0.5, 1.0))
    judge("C1 按「全模型 4bit」算会低估混合精度权重 > 3%",
          (mixed - all_w4) / all_w4 > 0.03,
          f"全 W4 {all_w4/1e9:.1f} GB vs 真实混合 {mixed/1e9:.1f} GB"
          f"（低估 {(mixed-all_w4)/all_w4*100:.1f}%，即 "
          f"{(mixed-all_w4)/1e9:.1f} GB）")

    # C2 套 GQA 公式算 1M KV：高估 3.5× 以上，且绝对值超过 300 GB
    wrong = kv_bytes(CTX_1M, KV_DIM_GQA8)
    right = kv_bytes(CTX_1M, KV_DIM_MLA)
    judge("C2 套 GQA 公式算 1M KV 会高估 > 3×（且 > 300 GB）",
          wrong / right > 3.0 and wrong > 300e9,
          f"错算 {wrong/1e9:.1f} GB vs 正确 {right/1e9:.1f} GB（{wrong/right:.2f}×）")

    # C3 1M 上下文下 KV 本身就超过权重——「只优化权重」救不了长上下文
    w_mixed = active_weight_bytes((0.5, 1.0))
    kv_int8 = kv_bytes(CTX_1M, KV_DIM_MLA, bytes_per=1)
    judge("C3 1M 上下文下 KV 搬运量超过权重搬运量",
          kv_int8 > w_mixed,
          f"KV(INT8) {kv_int8/1e9:.1f} GB > 权重 {w_mixed/1e9:.1f} GB"
          f"（{kv_int8/w_mixed:.2f}×）")

    # C4 1M 上下文下，聚合吞吐有一个与 batch 无关的硬上限 BW / KV_per_sequence，
    #    因为每生成一个 token（无论 batch 多大）都要把该序列的 KV 读一遍。
    kv_int8 = kv_bytes(CTX_1M, KV_DIM_MLA, bytes_per=1)
    ceiling = BW_REFERENCE_TBps * 1e12 / kv_int8
    t1 = tokens_per_sec(CTX_1M, (0.5, 1.0), 1, batch=1)
    t1024 = tokens_per_sec(CTX_1M, (0.5, 1.0), 1, batch=1024)
    judge("C4 batch 无法突破 1M 的 KV 上限（BW / 单序列KV ≈ 71 tok/s）",
          abs(t1024 - ceiling) / ceiling < 0.05 and t1024 < t1 * 2,
          f"batch1 {t1:.1f} → batch1024 {t1024:.1f} tok/s，"
          f"上限 {ceiling:.1f}（batch→∞）；远低于短上下文上限 "
          f"{tokens_per_sec(0, (0.5, 1.0), 1):.1f}")

    # C5 1M 的 L×L indexer 分数矩阵不可物化（与 02 的 C5 同源，这里是部署视角）
    bytes_full = CTX_1M * CTX_1M * 2
    judge("C5 1M 的 indexer 分数矩阵（fp16）需要 TB 级，必须融合 TopK",
          bytes_full > 1e12,
          f"{bytes_full/1e12:.1f} TB")

    # C6 「3× 提升」是相对同硬件自有基线的口径，不是跨平台倍数
    speedup = specs.GLM53_FLASH_SERVING["speedup_vs_own_baseline"]
    judge("C6 3× 的官方口径是『相对自有基线』（不是相对 NVIDIA）",
          speedup == 3.0 and "own_baseline" in "speedup_vs_own_baseline",
          f"GLM53_FLASH_SERVING['speedup_vs_own_baseline'] = {speedup}×；"
          f"原文另有『comparable to mainstream NVIDIA GPUs』的量级表述")


def main():
    print("== 160-model/glm · 03 部署账（权重 / KV / EP / 批处理）==")
    print(f"测量条件（口径）：{specs.MEASURE_HOST}")
    print(f"python {specs.MEASURE_PYTHON} / numpy {np.__version__}")
    print(f"带宽参照（引用值）：{BW_REFERENCE_TBps} TB/s")
    print("★ 只做算术，产出是**推导值**，不是实测值。\n")

    print(f"  expert 参数占比（按 01 的逐层核算）= {EXPERT_PARAM_SHARE*100:.2f}%")
    print(f"  expert 在激活参数里的占比          = {EXPERT_ACT_SHARE*100:.2f}%")
    print(f"  1M = {CTX_1M:,} token（config 口径）；KV/token/层 = {KV_DIM_MLA} 维\n")

    class_a()
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

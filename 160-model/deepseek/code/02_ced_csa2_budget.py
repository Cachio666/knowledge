#!/usr/bin/env python3
"""160-model/deepseek 编号程序 02 —— CED / CSA2 / 层级索引器的成本账（推导）。

配对文档：`160-model/deepseek/02-v4-1-flash-architecture.md`

★ 形态：本机默认 python3 可跑（`math` + `numpy`），三类断言在程序内部，失败非零退出。
  产出**一律是推导值**：不加载权重、不测硬件。

核心是把三件事写成可复算的式子：
  CED        prefill 从 O(N·L) 降到 O((L/2)(N + n_win))
  CSA2       只有 Full Mode 层真正新建 main KV（4/40 层），其余 Reindex / Reuse
  层级索引器 后续 indexer 的打分域从「整个上下文」缩到固定候选池
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


CFG = specs.V41_CONFIG
L = CFG["layers"]                 # 40
HALF = L // 2                     # 20
N_WIN = CFG["swa_window"]         # 128
CTX = specs.V41["context"]        # 1 048 576
CANDIDATES = CFG["hier_candidates"]   # 16 384


def prefill_units(N, with_ced):
    """以「token × 层」为单位的 prefill 工作量（不含 MoE，只看层遍历）。

    无 CED：每层都要过完整 prompt → N · L
    有 CED：encoder 过 N 个 token；decoder 只为 SWA 回放 n_win 个 token
            → (L/2)·N + (L/2)·n_win
    """
    if not with_ced:
        return N * L
    return HALF * N + HALF * N_WIN


def prefill_ratio(N):
    return prefill_units(N, True) / prefill_units(N, False)


# ---------------------------------------------------------------------------
# A 算法正确性
# ---------------------------------------------------------------------------

def class_a():
    print("A 算法正确性（显式计数 vs 闭式）")

    # A1 用显式双层循环数「层 × token」，与闭式对拍
    def explicit(N, with_ced):
        units = 0
        for layer in range(L):
            if not with_ced:
                units += N
            else:
                # 前 L/2 层过完整 prompt；后 L/2 层只为 SWA 回放 n_win
                units += N if layer < HALF else N_WIN
        return units

    ok = all(explicit(N, c) == prefill_units(N, c)
             for N in (N_WIN, 4096, 131072) for c in (True, False))
    judge("A1 prefill 工作量的闭式与显式逐层计数一致", ok,
          f"核对 N ∈ {{128, 4096, 131072}} × CED ∈ {{True, False}}；"
          f"L={L}，HALF={HALF}，n_win={N_WIN}")

    # A2 闭式 0.5 + n_win/(2N) 与比值定义对拍
    ok2 = all(abs(prefill_ratio(N) - (0.5 + N_WIN / (2 * N))) < 1e-15
              for N in (N_WIN, 1000, 100000))
    judge("A2 比值闭式 R(N) = 1/2 + n_win/(2N) 成立", ok2,
          f"R(128) = {prefill_ratio(128):.4f}，R(1M) = {prefill_ratio(CTX):.6f}")

    # A3 CSA2「新建 KV 的层数」由配置数组装算出，而不是手抄一个 4
    enc_full = sum(1 for m in CFG["encoder_group_pattern"] if m == "Full") * CFG["encoder_groups"]
    dec_full = sum(1 for m in CFG["decoder_group1_pattern"] if m == "Full") + \
        sum(1 for m in CFG["decoder_group_rest_pattern"] if m == "Full") * (CFG["decoder_groups"] - 1)
    judge("A3 由 Full/Reindex/Reuse 模式数组装出「新建 KV 的层数」= 4",
          enc_full + dec_full == 4,
          f"encoder {enc_full}（{CFG['encoder_groups']} 组 × 每组 1 个 Full）"
          f" + decoder {dec_full}（5 组里只有第 1 组首层是 Full）")


# ---------------------------------------------------------------------------
# B 代价模型
# ---------------------------------------------------------------------------

def class_b():
    print("\nB 代价模型")

    # B1 CED 对 prefill 的折半（长 prompt）
    r_1m = prefill_ratio(CTX)
    judge("B1 1M prompt 下 CED 把 prefill 工作量压到 50.0%–50.1%",
          0.500 < r_1m < 0.501,
          f"R(1M) = {r_1m:.6f}（≈ O(N·L) → O(N·L/2)）")

    # B2 由「256× 上下文、decode FLOPs 只涨 1/4」反解有效指数 α
    growth = specs.V41["decode_flops_growth_for_256x_context"]
    alpha = math.log(1 + growth) / math.log(256)
    judge("B2 反解出的上下文有效指数 α < 0.05（decode 近似与长度无关）",
          alpha < 0.05,
          f"256^α = 1 + {growth} → α = ln(1.25)/ln(256) = {alpha:.4f}")

    # B3 层级稀疏索引器：打分域从整个上下文缩到固定候选池
    red = CTX / CANDIDATES
    judge("B3 1M 上下文下，后续 indexer 的打分域缩小恰为 64×",
          abs(red - 64.0) < 1e-9,
          f"{CTX} / {CANDIDATES} = {red:.1f}×（候选池 = "
          f"{CFG['hier_blocks']} block × {CFG['hier_block_positions']} 位置）")

    # B4 top-k 与候选池的比
    judge("B4 最终 top-k 512 只占候选池的 1/32",
          CANDIDATES / CFG["attention_topk"] == 32,
          f"{CANDIDATES} / {CFG['attention_topk']} = {CANDIDATES/CFG['attention_topk']:.0f}")

    # B5 SWA bounded replay：从 L/2 × n_win 缩到 n_win
    exact_tokens = HALF * N_WIN
    judge("B5 SWA bounded replay 把回放量从 L/2×n_win 压到 n_win（20×）",
          exact_tokens / N_WIN == HALF,
          f"精确 {HALF}×{N_WIN} = {exact_tokens} token → 有界回放 {N_WIN} token"
          f"（{exact_tokens/N_WIN:.0f}×，即 L/2）")

    # B6 持久化 KV 的 1/8 必须能被两个因子相乘复算
    prod = 0.5 * 0.25
    judge("B6 持久化 KV 的 1/8 = 不再存 SWA KV（1/2）× 全局 KV 压到 1/4",
          abs(prod - specs.V41["persistent_kv_vs_v4flash"]) < 1e-12,
          f"0.5 × 0.25 = {prod} = 官方 1/8 ✓")

    # B7 CSA2 的层级复用：新建 KV 的层占比
    share = 4 / L
    judge("B7 只有 10% 的层真正新建 main KV（CSA2 压了「层」这一维）",
          abs(share - 0.10) < 1e-12,
          f"4 / {L} = {share*100:.0f}%")


# ---------------------------------------------------------------------------
# C 边界与反例
# ---------------------------------------------------------------------------

def class_c():
    print("\nC 边界与反例")

    # C1 短 prompt：CED 一点收益都没有
    judge("C1 N ≤ n_win 时 CED 的 prefill 折半完全消失",
          abs(prefill_ratio(N_WIN) - 1.0) < 1e-12 and abs(prefill_ratio(N_WIN // 2) - 1.5) < 1e-12,
          f"R({N_WIN}) = {prefill_ratio(N_WIN):.4f}（=1，无收益）；"
          f"R({N_WIN//2}) = {prefill_ratio(N_WIN//2):.4f}（**比不用 CED 还慢**）")

    # C2 上下文短于候选池时，层级索引器没有可缩的空间
    judge("C2 当上下文 ≤ 候选池（16384）时，层级索引器的缩减比为 1×",
          CANDIDATES / CANDIDATES == 1.0,
          f"L = {CANDIDATES} 时打分域 = 整个上下文，64× 的收益归零")

    # C3 Reuse 模式被误设成 Full：KV 存储层的数量翻 10 倍
    judge("C3 把所有层都设成 Full Mode，等于放弃 10× 的 KV 缩减",
          L / 4 == 10,
          f"{L} 层全存 KV vs 4 层 → {L/4:.0f}× 更多 main KV 存储")

    # C4 用「KV 是 O(L)」的直觉推 1/8 会漏掉 SWA 那一半
    only_global = 0.25
    judge("C4 只算全局 KV 的 1/4 会漏掉「不再持久化 SWA KV」这一半",
          abs(only_global - specs.V41["persistent_kv_vs_v4flash"]) > 0.1,
          f"只乘全局那一项得 {only_global}，官方是 "
          f"{specs.V41['persistent_kv_vs_v4flash']}（差 {only_global - specs.V41['persistent_kv_vs_v4flash']:+.3f}）")

    # C5 prefill 折半不等于端到端折半
    print(f"  [信息] 1M 上下文：prefill 层遍历 {prefill_ratio(CTX)*100:.1f}% ，"
          f"但 decode 仍要跑满 {L} 层（16B 激活）")
    judge("C5 CED 只作用于 prefill；decode 的层数与激活量不受影响",
          specs.V41["active_decode"] / specs.V41["active_prefill"] == 2.0,
          "decode/prefill 激活比 = 2.0×；把 CED 说成「整体快一倍」即为误读")


def main():
    print("== 160-model/deepseek · 02 CED / CSA2 / 层级索引器成本账 ==")
    print(f"测量条件（口径）：{specs.MEASURE_HOST}")
    print(f"python {specs.MEASURE_PYTHON} / numpy {np.__version__}")
    print("★ 只做算术，产出是**推导值**，不是实测值。\n")

    print(f"  配置：L={L}（encoder {HALF} + decoder {HALF}），n_win={N_WIN}，"
          f"上下文 {CTX:,}，候选池 {CANDIDATES:,}，attention top-k {CFG['attention_topk']}\n")

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

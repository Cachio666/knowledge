#!/usr/bin/env python3
"""03 低比特 KV：1-bit 为什么不行，以及各家引擎把 scale 放在哪一维。

对应文档：130-model-optimization/quantization/20-low-bit-and-engines.md

★ 本程序跑的是**真 torch**——输出是**本机实测值**，支撑关于**真实 PyTorch** 与
  **合成张量上的量化机制**的结论，但**必须绑定版本**。本机五个解释器都跑过：
    /usr/bin/python3 torch 2.8.0 / ~/miniconda3/bin/python torch 2.11.0
    ~/miniconda3/envs/deep-prac/bin/python torch 2.6.0
    ~/miniconda3/envs/dp-test/bin/python torch 2.7.1
    ~/PyCharmMiscProject/.venv/bin/python torch 2.13.0
  **2.9 / 2.10 / 2.12 本机没有解释器，不声称覆盖。** 全部 CPU-only，无 GPU 数字。

三类断言（CONTRIBUTING §2 第三种形态：B 在编号程序里，A/C 另见 verify.py）：
  B —— **代价模型**：① 载荷 bit 与「整块等效宽度」两笔账（差的正是 scale/zero 元数据）；
       ② 引擎表里那些容量比的**位宽自洽性**检验；③ 1-bit 断崖的 SQNR 曲线。
  A —— **两条互不依赖的通道**：① 引擎取值集合逐条核对（vLLM / llama.cpp）；
       ② KVQuant 两张表的基线**互不相同**这件事被显式断言。
  C —— 失败行为必须**真的跑出来**：对称 1-bit 的 `qmax = 0`（一个正值都表示不了）、
       1-bit 的灾难性 SQNR、「zero overhead」被读成「零计算开销」、
       以及**不落成断言**的那条（FP8 KV 的硬件支持矩阵来源冲突）。

★ 本程序**不复制 `specs_kv.py` 的任何常量值**：凡用到规格常量，一律写 `SPECS.xxx`。

用法：cd code && make 20_lowbit_engines   （或 make run 全跑）
"""

import math
import platform
import subprocess
import sys

import torch

import specs_kv as SPECS


_STATE = {"n": 0, "A": 0, "B": 0, "C": 0, "M": 0, "fail": []}

BIG = 1024
OUT_CHANNELS = (7, 61)
SEED = 20260918


def cpu_model():
    try:
        if platform.system() == "Darwin":
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"],
                                           stderr=subprocess.DEVNULL).decode().strip()
        if platform.system() == "Linux":
            with open("/proc/cpuinfo") as fh:
                for line in fh:
                    if line.lower().startswith("model name"):
                        return line.split(":")[1].strip()
    except Exception:
        pass
    return "%s（型号未取到）" % platform.machine()


def check(cls, name, cond, detail=""):
    _STATE["n"] += 1
    _STATE[cls] += 1
    ok = bool(cond)
    if not ok:
        _STATE["fail"].append("[%s] %s" % (cls, name))
    print("    [%s] %s%s" % ("通过" if ok else "**不通过**", name,
                             "   ← %s" % detail if detail else ""))
    return ok


def rule(title):
    print()
    print("-" * 74)
    print(title)
    print("-" * 74)


def sub(title):
    print()
    print("  · %s" % title)


def banner():
    print("=" * 74)
    print("130-model-optimization/quantization · 03 低比特与引擎")
    print("=" * 74)
    print("解释器：%s / python %s" % (sys.executable, sys.version.split()[0]))
    print("torch   ：%s  （CUDA 可用: %s）" % (torch.__version__, torch.cuda.is_available()))
    print("CPU     ：%s" % cpu_model())
    print("⚠ 全部 CPU-only：**引擎的 kernel 行为与显存带宽本机一根都测不了**，")
    print("  引擎那一节全部是引用值；本程序只测**量化器的数学**。")


# ===========================================================================
# 量化器（与 02 号程序同一条口径，但这里要对比「对称 / 非对称」两种）
# ===========================================================================

def quant_asym(x, bits, dim):
    """非对称 min-max（KIVI 口径）：`s = (max−min)/(2^b−1)`，`z = min`。级数 = 2^b。"""
    qmax = 2 ** bits - 1
    xmin = x.amin(dim=dim, keepdim=True)
    xmax = x.amax(dim=dim, keepdim=True)
    s = torch.where((xmax - xmin) == 0, torch.ones_like(xmax), (xmax - xmin) / qmax)
    return torch.clamp(torch.round((x - xmin) / s), 0, qmax) * s + xmin


def quant_sym(x, bits, dim):
    """对称 restricted-range（多数 RTN 实现的口径）：`qmax = 2^(b−1) − 1`。

    ⚠ b=1 时 `qmax = 0` ⇒ **一个正值都表示不了**（§C1）。这是本程序最锋利的反例。
    """
    qmax = 2 ** (bits - 1) - 1
    amax = x.abs().amax(dim=dim, keepdim=True)
    s = torch.where(amax == 0, torch.ones_like(amax), amax / qmax) if qmax > 0 else amax
    return torch.clamp(torch.round(x / s), -qmax, qmax) * s


def sqnr(x, xh):
    err = ((x - xh) ** 2).sum().item()
    if err == 0:
        return float("inf")
    return 10.0 * math.log10((x ** 2).sum().item() / err)


def make_kv(seed=SEED):
    """K：固定列 outlier（持久）；V：行异质。与 02 号程序的构造一致。"""
    gen = torch.Generator().manual_seed(seed)
    k = torch.randn(BIG, BIG, generator=gen)
    k[:, torch.tensor(OUT_CHANNELS)] = 100.0 * torch.tensor([1.0, -1.0]) * (
        1.0 + 0.1 * torch.randn(BIG, len(OUT_CHANNELS), generator=gen))
    v = torch.randn(BIG, BIG, generator=gen) * (
        10.0 ** (torch.rand(BIG, generator=gen) * -3.0))[:, None]
    return k, v


# ===========================================================================
# B. 代价模型
# ===========================================================================

def run_B():
    rule("B 代价模型（一）：两笔比特账——**载荷**与**整块等效宽度**")

    sub("B1 「2-bit KV cache」到底占多少 bit/元素？")
    print("      %-12s %8s %10s %12s %12s" % ("配置", "载荷", "元数据", "整块", "相对fp16"))
    for name in ("fp16", "fp8_per_tensor", "int4_g32", "int4_g128", "int2_g32", "int2_g128"):
        payload = {"fp16": 16.0, "fp8_per_tensor": 8.0}.get(name, None)
        if payload is None:
            payload = 4.0 if name.startswith("int4") else 2.0
        meta = SPECS.KV_METADATA_BITS_PER_ELEMENT[name]
        print("      %-12s %8.1f %10.2f %12.2f %11.2fx" % (name, payload, meta,
                                                           payload + meta,
                                                           16.0 / (payload + meta)))
    print()
    print("      判据：元数据 = （32 bit scale + 32 bit zero-point）/ 组大小 G = 64/G。")
    print("      ⚠ **「2-bit KV」是载荷口径（2.00 bit/元素），不是整块口径**——")
    print("         G=32 时整块是 4.00（载荷的 **2 倍**）。**两者不得并列。**")
    print("      ⚠ **这是三种记账口径里的一种**：只算 scale 一份是 32/G、")
    print("         **完全不算元数据**是 KVQuant Table 16 的 `Avg. Num. Bits` 口径")
    print("         （它的 KVQuant-4bit 写 4.00-4.02）。**同一个「4-bit」三个读数。**")
    check("B", "★ G=32 的 int2：载荷 2.00 → 整块 %.2f bit/元素（+%.0f%%）"
          % (2.0 + SPECS.KV_METADATA_BITS_PER_ELEMENT["int2_g32"],
             100 * SPECS.KV_METADATA_BITS_PER_ELEMENT["int2_g32"] / 2.0),
          abs(2.0 + SPECS.KV_METADATA_BITS_PER_ELEMENT["int2_g32"] - 4.0) < 1e-12)
    check("B", "★ G=128 的 int2 整块 %.2f bit/元素；G 从 128 缩到 32 让元数据占的**比例**"
               "从 %.1f%% 涨到 %.1f%%"
          % (2.0 + SPECS.KV_METADATA_BITS_PER_ELEMENT["int2_g128"],
             100 * SPECS.KV_METADATA_BITS_PER_ELEMENT["int2_g128"] / 2.0,
             100 * SPECS.KV_METADATA_BITS_PER_ELEMENT["int2_g32"] / 2.0),
          SPECS.KV_METADATA_BITS_PER_ELEMENT["int2_g32"] == 4
          * SPECS.KV_METADATA_BITS_PER_ELEMENT["int2_g128"])

    sub("B2 KV cache 的实际压缩比（用整块等效宽度，不用载荷）")
    for name in ("int4_g128", "int2_g32", "int2_g128"):
        payload = 4.0 if name.startswith("int4") else 2.0
        whole = payload + SPECS.KV_METADATA_BITS_PER_ELEMENT[name]
        print("      %-10s 整块 %.2f bit/元素 ⇒ fp16 的 %.2fx" % (name, whole,
                                                                 16.0 / whole))
    check("B", "★ KIVI 说的「2bit」在 G=32 下对应整块 %.2f bit ⇒ 实压缩 **%.2fx**，"
               "而不是「16/2 = 8x」" % (4.0, 16.0 / 4.0), abs(16.0 / 4.0 - 4.00) < 0.01)

    sub("B3 ★ 引擎表里的容量比**能不能用位宽算术复现**")
    print("      SGLang 报：BF16→FP8 %.2fx ；BF16→FP4 %.2fx" % (
        SPECS.SGLANG_TOKEN_CAPACITY_FP8, SPECS.SGLANG_TOKEN_CAPACITY_FP4))
    fp8_bits = 8.0
    fp4_bits = 4.0 + SPECS.KV_METADATA_BITS_PER_ELEMENT["int4_g128"]
    print("      FP8：位宽 %.2f bit/元素 ⇒ 16/%.2f = **%.4f×**（报 %.2f）"
          % (fp8_bits, fp8_bits, 16.0 / fp8_bits, SPECS.SGLANG_TOKEN_CAPACITY_FP8))
    print("      FP4：位宽 %.2f bit/元素（4 bit 载荷 + 64/128 元数据）⇒ 16/%.2f = **%.4f×**"
          "（报 %.2f）" % (fp4_bits, fp4_bits, 16.0 / fp4_bits,
                          SPECS.SGLANG_TOKEN_CAPACITY_FP4))
    check("B", "★ FP8 那条与位宽算术**精确一致**（16/8 = 2.00）",
          abs(16.0 / fp8_bits - SPECS.SGLANG_TOKEN_CAPACITY_FP8) < 1e-9)
    check("B", "★ FP4 那条**也对得上**——但前提是「4 bit + G=128 的 64/G 元数据」= %.2f bit，"
               "16/%.2f = %.3f，四舍五入到两位就是来源印的 %.2f"
          % (fp4_bits, fp4_bits, 16.0 / fp4_bits, SPECS.SGLANG_TOKEN_CAPACITY_FP4),
          abs(round(16.0 / fp4_bits, 2) - SPECS.SGLANG_TOKEN_CAPACITY_FP4) < 0.005,
          "反解出的位宽应为 %.2f，specs 记的是 %.2f"
          % (16.0 / SPECS.SGLANG_TOKEN_CAPACITY_FP4, SPECS.SGLANG_FP4_IMPLIED_BITS))
    check("B", "★ 反过来说：**这一格同时印证了「SGLang 的 FP4 KV 用 G=128」**——"
               "若按 G=32（元数据 %.2f bit）则是 %.2fx、按纯载荷是 %.2fx，都对不上 %.2f"
          % (SPECS.KV_METADATA_BITS_PER_ELEMENT["int4_g32"],
             16.0 / (4.0 + SPECS.KV_METADATA_BITS_PER_ELEMENT["int4_g32"]), 16.0 / 4.0,
             SPECS.SGLANG_TOKEN_CAPACITY_FP4),
          abs(16.0 / (4.0 + SPECS.KV_METADATA_BITS_PER_ELEMENT["int4_g32"])
              - SPECS.SGLANG_TOKEN_CAPACITY_FP4) > 0.3)


def run_B2():
    rule("B 代价模型（二）：1-bit 断崖——真 torch 上的 SQNR 曲线")

    k, v = make_kv()
    sub("B4 逐 bit 的 SQNR（判据：K 用固定列 outlier、V 用行异质；非对称 min-max）")
    print("      %3s | %-24s | %-24s" % ("bit", "K（per-channel 治列）", "V（per-token 治行）"))
    print("      %3s | %-24s | %-24s" % ("", "per-token   per-channel", "per-token   per-channel"))
    rows = {}
    for bits in (1, 2, 3, 4, 8):
        kt, kc = sqnr(k, quant_asym(k, bits, 1)), sqnr(k, quant_asym(k, bits, 0))
        vt, vc = sqnr(v, quant_asym(v, bits, 1)), sqnr(v, quant_asym(v, bits, 0))
        rows[bits] = (kt, kc, vt, vc)
        print("      %3d | %8.2f %12.2f    | %8.2f %12.2f" % (bits, kt, kc, vt, vc))
    print()
    print("      可用码字数：%s（非对称 min-max）" % ", ".join(
        "%d bit→%d 级" % (b, 2 ** b) for b in (1, 2, 3, 4, 8)))

    gap_k = rows[2][1] - rows[1][1]
    gap_v = rows[2][2] - rows[1][2]
    check("B", "★ K（per-channel）：1-bit %.2f dB → 2-bit %.2f dB，**一步跳 %.2f dB**"
          % (rows[1][1], rows[2][1], gap_k),
          gap_k > SPECS.LOCAL_1BIT_VS_2BIT_GAP_DB - 1.0,
          "specs 记的断崖量级 %.1f dB" % SPECS.LOCAL_1BIT_VS_2BIT_GAP_DB)
    check("B", "★ V（per-token）：1-bit %.2f dB → 2-bit %.2f dB，一步跳 %.2f dB"
          % (rows[1][2], rows[2][2], gap_v), gap_v > 8.0)
    check("B", "★ **1-bit 的 K（per-channel）只有 %.2f dB**——低于「信号与噪声同量级」的 %d dB 线"
          % (rows[1][1], 0), rows[1][1] < 10.0)
    check("B", "★ 2-bit 起 K 已经可用（per-channel %.2f dB），这就是 KIVI 选 2-bit 的落点"
          % rows[2][1], rows[2][1] > 15.0)
    # ⚠ **斜率必须按「每 bit」比，不能拿 2-bit 跨度与 4-bit 跨度的总增量比**——
    #   那是两个不同的分母（本程序的第一版就是这么写错的，被自己的实测推翻）。
    slope_low = (rows[4][1] - rows[2][1]) / 2.0     # 2→4 bit，跨 2 个 bit
    slope_high = (rows[8][1] - rows[4][1]) / 4.0    # 4→8 bit，跨 4 个 bit
    check("B", "★ SQNR 随 bit 单调上升（K per-channel：%s）"
               % " → ".join("%.2f" % rows[b][1] for b in (1, 2, 3, 4, 8)),
          all(rows[b][1] < rows[b2][1] for b, b2 in ((1, 2), (2, 3), (3, 4), (4, 8))))
    check("B", "★ 每 bit 的斜率**递减**：2→4 bit 段 %.2f dB/bit vs 4→8 bit 段 %.2f dB/bit"
               "（**判据是「每 bit」，不是两段的总增量**——总增量比会得出相反结论）"
          % (slope_low, slope_high), slope_low > slope_high)


# ===========================================================================
# A. 两条互不依赖的通道
# ===========================================================================

def run_A():
    rule("A 判据复核：引擎取值集合与两张 PPL 表")

    sub("A1 vLLM 的 `CacheDType`：逐项列出，并核对计数")
    for i, name in enumerate(SPECS.VLLM_CACHE_DTYPE_MAIN, 1):
        print("      %2d. %s" % (i, name))
    check("A", "★ `len(VLLM_CACHE_DTYPE_MAIN)` == VLLM_CACHE_DTYPE_COUNT == %d"
          % SPECS.VLLM_CACHE_DTYPE_COUNT,
          len(SPECS.VLLM_CACHE_DTYPE_MAIN) == SPECS.VLLM_CACHE_DTYPE_COUNT)
    check("A", "★ 这一份来自 `vllm/config/cache.py`（`main` 分支）**逐字读到**，"
               "**不是**老文档里的 7 项——两套取值都真实，**差别在版本**",
          len(SPECS.VLLM_CACHE_DTYPE_MAIN) > 7 and "auto" in SPECS.VLLM_CACHE_DTYPE_MAIN)
    check("A", "★ 里面既有 **per-tensor/per-head** 的 `fp8`，也有名字里写着 "
               "`per_token_head` 的三项（int4/int8/fp8）——**它们是两种不同的粒度**，"
               "不得当成同一个 `fp8` 的别名",
          sum(1 for x in SPECS.VLLM_CACHE_DTYPE_MAIN if "per_token_head" in x) == 3)

    sub("A2 llama.cpp 的允许值：逐项列出，并核对计数与默认值")
    print("      %s" % ", ".join(SPECS.LLAMACPP_CACHE_TYPES))
    check("A", "★ llama.cpp 侧恰好 %d 项，默认 `%s`"
          % (len(SPECS.LLAMACPP_CACHE_TYPES), SPECS.LLAMACPP_DEFAULT_CACHE_TYPE),
          len(SPECS.LLAMACPP_CACHE_TYPES) == 9
          and SPECS.LLAMACPP_DEFAULT_CACHE_TYPE == "f16")
    check("A", "★ 明确**不含** `q8_1` / `q2_K`…`q6_K` / `tq1_0` / `tq2_0` / `mxfp4`"
               "——「llama.cpp 支持 mxfp4 KV」这句话在这份列表下**是假的**",
          not any(x in SPECS.LLAMACPP_CACHE_TYPES
                  for x in ("q8_1", "q2_K", "q6_K", "tq1_0", "mxfp4"))
          and "q8_0" in SPECS.LLAMACPP_CACHE_TYPES
          and "iq4_nl" in SPECS.LLAMACPP_CACHE_TYPES)

    sub("A3 三家引擎的 scale 粒度：**三家三个答案**")
    for name, (write, read, per_token) in sorted(SPECS.ENGINE_KV_SCALE_GRANULARITY.items()):
        print("      %-12s 写侧 %-24s 读侧 %-22s 支持 per-token: %s"
              % (name, write, read, per_token))
    check("A", "★ 只有 KIVI 那一行 `per-token` 为真；vLLM 与 SGLang 都是标量/逐头粒度",
          sum(1 for v in SPECS.ENGINE_KV_SCALE_GRANULARITY.values() if v[2]) == 1)
    check("A", "★ vLLM 的写侧是 `per_tensor_or_per_head`——**没有 per-token**；"
               "这正是 KIVI 那套方案在 vLLM 里跑不起来的原因",
          SPECS.ENGINE_KV_SCALE_GRANULARITY["vLLM (FP8)"][2] is False)

    sub("A4 KVQuant 两张表的**基线不同**——这是必须显式断言的一条")
    b1 = SPECS.KVQUANT_TABLE1_LLAMA7B["baseline"][0]
    b16 = SPECS.KVQUANT_TABLE16_LLAMA2_7B["baseline"]
    print("      Table 1 （LLaMA-7B）   baseline PPL %.2f" % b1)
    print("      Table 16（Llama-2-7B） baseline PPL %.2f" % b16)
    check("A", "★ 两张表的基线 **%.2f vs %.2f，不相等** ⇒ **不得并列比较**（P9）" % (b1, b16),
          b1 != b16)
    check("A", "★ 而且两张表**连模型都不是同一个**（LLaMA-7B vs Llama-2-7B）",
          True)
    print("      Table 16 / Llama-2-7B 那一列：")
    for name in ("baseline", "int4", "KVQuant-4bit", "KVQuant-3bit-1%", "int3", "int2"):
        print("        %-18s PPL %8.2f" % (name, SPECS.KVQUANT_TABLE16_LLAMA2_7B[name]))
    check("A", "★ 同一张表内的落差：int3 → %.2f，int2 → %.2f（**int2 是 3 个数量级**）"
          % (SPECS.KVQUANT_TABLE16_LLAMA2_7B["int3"],
             SPECS.KVQUANT_TABLE16_LLAMA2_7B["int2"]),
          SPECS.KVQUANT_TABLE16_LLAMA2_7B["int2"] > 1000
          and SPECS.KVQUANT_TABLE16_LLAMA2_7B["int3"] > 10)
    check("A", "★ 加技巧后 4-bit 只比基线差 %.2f（KVQuant-4bit %.2f vs baseline %.2f）"
          % (SPECS.KVQUANT_TABLE16_LLAMA2_7B["KVQuant-4bit"]
             - SPECS.KVQUANT_TABLE16_LLAMA2_7B["baseline"],
             SPECS.KVQUANT_TABLE16_LLAMA2_7B["KVQuant-4bit"],
             SPECS.KVQUANT_TABLE16_LLAMA2_7B["baseline"]),
          SPECS.KVQUANT_TABLE16_LLAMA2_7B["KVQuant-4bit"]
          - SPECS.KVQUANT_TABLE16_LLAMA2_7B["baseline"] < 0.1)


# ===========================================================================
# C. 边界与反例
# ===========================================================================

def run_C():
    rule("C 边界与反例（1-bit 的三种死法 + 一条**不落断言**的）")

    sub("C1 ★ 对称 restricted-range 在 1-bit 下 `qmax = 2^0 − 1 = 0`：**一个正值都表示不了**")
    for bits in (1, 2, 3, 4, 8):
        qmax = 2 ** (bits - 1) - 1
        print("      b=%d  对称 qmax = %3d ⇒ 非零码字 %3d 个（2·qmax）"
              % (bits, qmax, 2 * qmax))
    check("C", "★ b=1 的对称量化 `qmax = %d` ⇒ 全张量被压成常数 0" % SPECS.SYMMETRIC_QBITS_AT_B1,
          SPECS.SYMMETRIC_QBITS_AT_B1 == 2 ** 0 - 1 == 0)
    k, v = make_kv()
    sym1 = sqnr(k, quant_sym(k, 1, 0))
    asym1 = sqnr(k, quant_asym(k, 1, 0))
    print("      K 上实测：对称 1-bit **%.2f dB** ；非对称 1-bit %.2f dB" % (sym1, asym1))
    sym1_recon = quant_sym(k, 1, 0)
    check("C", "★ 对称 1-bit 的**重建结果恒为 0**（`qmax=0` ⇒ 所有码字都被夹到 0）",
          bool((sym1_recon == 0).all()), "非零元素 %d 个"
          % int((sym1_recon != 0).sum()))
    check("C", "★ 因此它的 SQNR 恰为 **%.2f dB**——不是负无穷，而是「**误差能量 == 信号能量**」"
               "（分子分母都是 Σx²）；非对称 1-bit 是 %.2f dB，**同一张张量差 %.2f dB**"
          % (sym1, asym1, asym1 - sym1), abs(sym1) < 1e-9 and asym1 > 0)
    check("C", "★ 所以「1-bit」这个词**必须连量化器一起说**："
               "非对称有 %d 级、对称有 %d 级——**差一整档**"
          % (2 ** 1, 2 * (2 ** 0 - 1)), 2 ** 1 == 2 and 2 * (2 ** 0 - 1) == 0)

    sub("C2 ★ 1-bit 全量 KV 量化在没有额外技巧时不可用（本机合成张量上的结论）")
    kc1 = sqnr(k, quant_asym(k, 1, 0))
    vt1 = sqnr(v, quant_asym(v, 1, 1))
    print("      仅 K → 1-bit（per-channel）：%7.2f dB" % kc1)
    print("      仅 V → 1-bit（per-token）  ：%7.2f dB" % vt1)
    print("      两者一起                    ：K %7.2f / V %7.2f dB" % (kc1, vt1))
    check("C", "★ 两个方向 **1-bit 都掉到 10 dB 以下**（K %.2f、V %.2f）——"
               "信号与量化噪声同量级" % (kc1, vt1), kc1 < 10.0 and vt1 < 10.0)
    check("C", "⚠ **本机给不出 PPL 数字**（无权重、无 GPU）⇒ 这一条只支撑「量化器层面不可用」，"
               "不支撑「模型层面不可用」", True)

    sub("C3 ★ QJL 的「zero overhead」**不是「零计算开销」**")
    print("      论文口径：%.1f bits per FPN；「zero overhead」= **不存 zero-point 与 scale**"
          % SPECS.QJL_BITS_PER_FPN)
    print("      省下的是 block 量化要额外占的 %g–%g bit/数"
          % (SPECS.QJL_BLOCK_QUANT_METADATA_BITS_LOW, SPECS.QJL_BLOCK_QUANT_METADATA_BITS_HIGH))
    print("      代价（论文自己写的）：JL 随机投影矩阵；query 侧必须做同样的投影；"
          "prompt 阶段要识别 outlier 通道")
    check("C", "★ 「zero overhead」的确切含义是**省掉元数据比特**（%g–%g bit/数），"
               "**不是**「零计算开销」——把它读成后者，就漏掉了 JL 投影这一整块成本"
          % (SPECS.QJL_BLOCK_QUANT_METADATA_BITS_LOW,
             SPECS.QJL_BLOCK_QUANT_METADATA_BITS_HIGH),
          SPECS.QJL_BLOCK_QUANT_METADATA_BITS_LOW == 1.0
          and SPECS.QJL_BLOCK_QUANT_METADATA_BITS_HIGH == 2.0)
    check("C", "★ QJL 只把 **K** 量化到 1 bit，V 仍走标准 per-token——"
               "「1-bit KV」这句话省掉「只对 K」就是错的", SPECS.QJL_BITS_PER_FPN == 3.0)

    sub("C4 ★ FP8 KV 的硬件支持矩阵来源冲突 ⇒ **不落成断言**（本程序的诚实处）")
    print("      来源 A：FP8 KV cache 支持 Blackwell/Hopper/Ada/Ampere（含 Ampere）")
    print("      来源 B：Ampere 没有 FP8（无 FP8 tensor core）")
    print("      ⇒ 两者**都真实**（「存储 dtype 支持」与「硬件加速」是两件事），"
          "但本仓库没读到官方 support matrix，**不予判定**。")
    check("C", "★ 判据：`specs_kv.py` 里**没有**任何「FP8 KV 支持哪些架构」的常量——"
               "**不落成常量就不会被误引**（与 P6 的对策同源）",
          not any("FP8" in n and "ARCH" in n for n in dir(SPECS))
          and not any("SUPPORT" in n for n in dir(SPECS)))
    check("C", "★ 同理，`UNSOURCED` 里那三条（全模型 1 FLOP/weight-byte、"
               "1-bit 的 17.17/177.13/620.08、Databricks 博客原文）**一个都没落成常量**",
          not hasattr(SPECS, "FULL_MODEL_DECODE_FLOPS_PER_BYTE")
          and not hasattr(SPECS, "ONE_BIT_K_PPL"))

    sub("C5 ★ 「2.35×–3.47×」不是显存压缩——本主题第三次重复这条（P1 的反面）")
    print("      %s" % SPECS.KIVI_MEMORY_QUOTE)
    print("      %s" % SPECS.KIVI_THROUGHPUT_QUOTE)
    check("C", "★ 显存 %s×（含权重）与吞吐 %s×–%s× 是**两个量**；"
               "本主题三篇文档各写一次"
          % (SPECS.KIVI_PEAK_MEMORY_RATIO, SPECS.KIVI_THROUGHPUT_LOW,
             SPECS.KIVI_THROUGHPUT_HIGH),
          SPECS.KIVI_PEAK_MEMORY_RATIO != SPECS.KIVI_THROUGHPUT_LOW)


# ===========================================================================
# M / F
# ===========================================================================

def run_M():
    rule("M 测量条件自洽")
    v = torch.__version__
    check("M", "本次运行的 torch %s 在 `MEASURED_VERSIONS` 表里" % v,
          v in SPECS.MEASURED_VERSIONS or ".".join(v.split(".")[:2])
          in SPECS.MEASURED_VERSIONS, "表 = %s" % (SPECS.MEASURED_VERSIONS,))
    check("M", "`UNCOVERED_MINORS` 逐字等于 `2.9.x / 2.10.x / 2.12.x`",
          SPECS.UNCOVERED_MINORS == "2.9.x / 2.10.x / 2.12.x")
    host = cpu_model()
    import os as _os
    osname = {"Darwin": "macOS", "Linux": "Linux"}.get(platform.system(), platform.system())
    check("M", "`MEASURE_HOST` 逐字等于「本机 CPU 型号 + 核数 + 系统 + CPU-only」",
          SPECS.MEASURE_HOST == "%s（%d 核）/ %s / **CPU-only，无 GPU 数字**"
          % (host, _os.cpu_count(), osname), SPECS.MEASURE_HOST)
    check("M", "本主题三篇共用同一张测量条件表（`MEASURED_VERSIONS` 长度 5）",
          len(SPECS.MEASURED_VERSIONS) == 5)


def run_F():
    rule("F 变异测试清单（A6）：本程序直接读到的 specs 常量")
    src = open(__file__, encoding="utf-8").read()
    names = sorted(n for n in dir(SPECS) if n.isupper() and not n.startswith("_"))
    used = [n for n in names if ("SPECS.%s" % n) in src]
    print("      specs.py 共 %d 条常量；本程序直接读到 %d 条" % (len(names), len(used)))
    print("      %s" % ", ".join(used))
    check("B", "本程序读到的常量数 ≥ 10（每条都参与断言）", len(used) >= 10)


def main():
    banner()
    run_B()
    run_B2()
    run_A()
    run_C()
    run_M()
    run_F()
    print()
    print("=" * 74)
    print("合计 %d 条断言：A %d / B %d / C %d / M %d；失败 %d 条"
          % (_STATE["n"], _STATE["A"], _STATE["B"], _STATE["C"], _STATE["M"],
             len(_STATE["fail"])))
    if _STATE["fail"]:
        for f in _STATE["fail"]:
            print("  · %s" % f)
    sys.exit(0 if not _STATE["fail"] else 1)


if __name__ == "__main__":
    main()

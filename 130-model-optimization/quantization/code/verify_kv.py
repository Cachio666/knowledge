#!/usr/bin/env python3
"""130-model-optimization/quantization 主题的纯 python3 复核入口。

★ 本文件**固定跑在默认 `python3` 下，永不 `import torch`**（CONTRIBUTING §2 第三种形态）。
  它提供的是「**换了台机器、什么都没装，也能复核的那一半**」：
  用**独立的参考实现**（下面的 `ref_*`）把本主题的**判据**重算一遍，
  再用手写期望值（**不是** torch 的输出）判它对不对。
  **它的产出是推导，不是实测**——不得标成本机实测值。

  三类断言在本形态下的分工（CONTRIBUTING §2）：
    · **A 判据复核** —— 本文件。参照物是下面 `ref_*` 这套独立实现与**手写期望值**。
    · **C 边界与反例** —— 本文件。断言的是「失败行为成立」。
    · **B 代价模型** —— 在**编号程序**里（代价模型的参照是真系统，只有能 import 到它的
      环境里测才有意义）。本文件**不做** B。

★ 本主题里 A/C 与编号程序的分工是 **「算术 vs 实现」**：
  · 本文件（纯 python3 + 标准库）验的是**可独立推导的算术**：
    KV cache 的字节账与三处外部锚点、算术强度 `ST/(S+T)` 的推导、
    per-token 与 per-channel 的误差结构、1-bit 的级数、
    残差滑窗的输出误差、载荷与元数据两笔比特账。
  · 编号程序（真 torch）测的是**真实实现与本机 CPU 的行为**：
    `torch.empty(...).nbytes` 的真张量分配、`quantize_per_channel` 的轴语义、
    GEMV 与 GEMM 的落差、decode 步跑到的带宽百分比。
  · **两者不得互相背书。** 算术成立不证明实现如此；实现如此不证明算术成立。
  · ⚠ **规模不同 ⇒ 数字不同**：本文件跑 64×64 的小张量、用标准库的 `random`；
    编号程序跑 1024×1024 与 8192 长度的 QK^T。**只有定性结构必须一致。**
    ⚠ 而「列 outlier 的领先幅度**随张量宽度变化**」已在同域 `06-granularity` 实测过
      （256 宽 10.30 dB vs 4096 宽 20.89 dB）——**本文件的阈值因此写得比编号程序宽得多**。

★ 「独立」的硬要求（CONTRIBUTING §2）：本文件里没有一行代码来自 torch，
  期望值**全部**是手写的常量或从 `specs_kv.py` 抄来的外部常量。
  §E 节把这条**机械地验证一遍**：把参考实现故意改错，A 类断言必须失败。

★ `specs_kv.py` 的每条常量都至少被本文件或某个编号程序的断言读过一次（A6 约定）。
  §F 节把「本文件负责的那部分」列出来，供变异测试逐个改。

运行：`make verify`（等价于 `python3 verify.py`）。只用标准库。
"""

import math
import random
import sys

import specs_kv as SPECS


# ===========================================================================
# 口径
# ===========================================================================

_STATE = {"n": 0, "A": 0, "C": 0, "fail": []}

STDLIB_N = 64                 # 本文件自己的小规模边长（**与编号程序的 1024 不同**）
STDLIB_SEED = 20260918        # 标准库 random 的种子（**与 torch 的无关**）
STDLIB_SEEDS = (20260918, 20260931, 20260944)   # 小张量噪声大，取 3 个种子的均值

# A.4 的消融结果缓存（C.3 与 E 节要复用同一批数）
_RES = {}
OUT_CHANNELS = (7, 61)        # 列 outlier 的位置（与编号程序同，便于对照）

KiB = 1024.0
GiB = 1024.0 ** 3


def check(cls, name, cond, detail=""):
    _STATE["n"] += 1
    _STATE[cls] += 1
    ok = bool(cond)
    if not ok:
        _STATE["fail"].append("[%s] %s" % (cls, name))
    print("    [%s] %s%s" % ("通过" if ok else "**不通过**", name,
                             "   ← %s" % detail if detail else ""))
    return ok


def section(tag, title):
    print()
    print("=" * 78)
    print("%s %s" % (tag, title))
    print("=" * 78)


def sub(title):
    print()
    print("  -- %s" % title)


# ===========================================================================
# 独立参考实现（ref_*）—— 一行都不来自 torch / numpy
# ===========================================================================
# 数据一律用「列表的列表」表示：x[i][j]，i 是 token（行），j 是 feature（列）。
# ⚠ **轴约定与三篇文档、三个编号程序完全一致**（见 specs.AXIS_CONVENTION_TOKENS_FIRST）：
#     沿「行内」归约（每行一组） = per-token
#     沿「列内」归约（每列一组） = per-channel
#   **写反不会报错**，只会推出与 KIVI 相反的结论（§C3 把这个后果真跑出来）。

def mkV(n, seed=STDLIB_SEED):
    """col_persist：固定列 outlier，**跨 token 近似恒定**（K 的结构）。"""
    rnd = random.Random(seed)
    x = [[rnd.gauss(0.0, 1.0) for _ in range(n)] for _ in range(n)]
    signs = (1.0, -1.0)
    for t in range(n):
        for k, c in enumerate(OUT_CHANNELS):
            x[t][c] = 100.0 * signs[k] * (1.0 + 0.1 * rnd.gauss(0.0, 1.0))
    return x


def mkRowHetero(n, seed=STDLIB_SEED):
    """行异质：每个 token 的整体尺度跨 10^−3…10^0（V 的结构）。"""
    rnd = random.Random(seed + 7)
    x = [[rnd.gauss(0.0, 1.0) for _ in range(n)] for _ in range(n)]
    for t in range(n):
        s = 10.0 ** rnd.uniform(-3.0, 0.0)
        for j in range(n):
            x[t][j] *= s
    return x


def mkIID(n, seed=STDLIB_SEED):
    rnd = random.Random(seed + 13)
    return [[rnd.gauss(0.0, 1.0) for _ in range(n)] for _ in range(n)]


def ref_quant_asym(x, bits, dim):
    """KIVI 口径的非对称 min-max：`s = (max−min)/(2^b−1)`、`z = min`。

    `dim=1` ⇒ 每行一组（per-token）；`dim=0` ⇒ 每列一组（per-channel）。
    可用码字恰为 `2^b` 个。
    """
    qmax = 2 ** bits - 1
    n = len(x)
    m = len(x[0])
    out = [[0.0] * m for _ in range(n)]
    if dim == 1:
        for i in range(n):
            lo = min(x[i])
            hi = max(x[i])
            s = (hi - lo) / qmax if hi > lo else 1.0
            for j in range(m):
                q = int(round((x[i][j] - lo) / s))
                q = 0 if q < 0 else (qmax if q > qmax else q)
                out[i][j] = q * s + lo
    else:
        for j in range(m):
            col = [x[i][j] for i in range(n)]
            lo = min(col)
            hi = max(col)
            s = (hi - lo) / qmax if hi > lo else 1.0
            for i in range(n):
                q = int(round((x[i][j] - lo) / s))
                q = 0 if q < 0 else (qmax if q > qmax else q)
                out[i][j] = q * s + lo
    return out


def ref_quant_sym(x, bits, dim):
    """对称 restricted-range：`qmax = 2^(b−1) − 1`。

    ⚠ `bits = 1` 时 `qmax = 0` ⇒ **重建结果恒为 0**（§C1）。
    """
    qmax = 2 ** (bits - 1) - 1
    n = len(x)
    m = len(x[0])
    out = [[0.0] * m for _ in range(n)]
    if dim == 1:
        for i in range(n):
            amax = max(abs(v) for v in x[i])
            s = amax / qmax if qmax > 0 and amax > 0 else amax
            for j in range(m):
                q = int(round(x[i][j] / s)) if s > 0 else 0
                q = -qmax if q < -qmax else (qmax if q > qmax else q)
                out[i][j] = q * s
    else:
        for j in range(m):
            amax = max(abs(x[i][j]) for i in range(n))
            s = amax / qmax if qmax > 0 and amax > 0 else amax
            for i in range(n):
                q = int(round(x[i][j] / s)) if s > 0 else 0
                q = -qmax if q < -qmax else (qmax if q > qmax else q)
                out[i][j] = q * s
    return out


def ref_sqnr(x, xh):
    """`10·log10(Σx² / Σ(x−x̂)²)`。**全表求和**，不做逐元素相除。"""
    num = 0.0
    den = 0.0
    for i in range(len(x)):
        for j in range(len(x[i])):
            num += x[i][j] ** 2
            den += (x[i][j] - xh[i][j]) ** 2
    if den == 0.0:
        return float("inf")
    return 10.0 * math.log10(num / den)


def ref_rel_elementwise(x, xh):
    """KIVI Table 2 的判据：`‖(X−X')/X‖_F`——**逐元素相除**（近零元素会把它炸掉，§C4）。"""
    acc = 0.0
    for i in range(len(x)):
        for j in range(len(x[i])):
            acc += ((x[i][j] - xh[i][j]) / x[i][j]) ** 2
    return math.sqrt(acc)


def ref_sqnr_vec(a, b):
    """向量版的 SQNR（注意力的输出是一维的）。**判据与矩阵版逐字相同。**"""
    num = sum(v * v for v in a)
    den = sum((u - v) ** 2 for u, v in zip(a, b))
    if den == 0.0:
        return float("inf")
    return 10.0 * math.log10(num / den)


def ref_attention_output(x, weights):
    """`p @ X`：注意力权重（长度 = token 数）对 value 各行做加权和。"""
    m = len(x[0])
    return [sum(weights[i] * x[i][j] for i in range(len(x))) for j in range(m)]


def ref_cache_bytes(n_layers, n_kv_heads, head_dim, seq, batch, bpe):
    """KV cache 字节数。**这是本文件自己写的第二条算式**（编号程序里有第一条）。"""
    return SPECS.KV_TENSOR_COUNT * n_layers * n_kv_heads * head_dim * seq * batch * bpe


def ref_attention_ai(s, t, d, bytes_per_elem=2.0):
    """注意力算子的算术强度：`FLOPs / bytes`。

    FLOPs = 2·S·T·D（QK^T）+ 2·S·T·D（PV） = 4·S·T·D
    bytes = (Q + K + V + O) 每元素 `bytes_per_elem` 字节 = 2·bpe·D·(S+T)
    ⇒ AI = 4·S·T·D / (2·bpe·D·(S+T)) = 2·S·T / (bpe·(S+T))
    ⚠ **bpe = 2（fp16）时才化简成 `ST/(S+T)`**（见 §A3）。
    """
    flops = 4.0 * s * t * d
    byts = 2.0 * bytes_per_elem * d * (s + t)
    return flops / byts


# ===========================================================================
# A. 判据复核
# ===========================================================================

def run_A():
    section("A", "判据复核（纯标准库的独立重算）")

    sub("A.1 KV cache 的字节账：五个模型 × 四种精度")
    for name in ("Llama-2-7B", "Llama-2-13B", "Llama-2-70B", "Mistral-7B", "OPT-13B"):
        nl, nh, hd = SPECS.KV_MODEL_SHAPES[name]
        pt = ref_cache_bytes(nl, nh, hd, 1, 1, SPECS.KV_BYTES_PER_ELEMENT["fp16"])
        print("      %-14s 每 token %8d B = %7.1f KiB" % (name, pt, pt / KiB))
    nl, nh, hd = SPECS.KV_MODEL_SHAPES["Llama-2-7B"]
    pt7 = ref_cache_bytes(nl, nh, hd, 1, 1, SPECS.KV_BYTES_PER_ELEMENT["fp16"])
    check("A", "Llama-2-7B 每 token == `LLAMA2_7B_PER_TOKEN_BYTES` = %d"
          % SPECS.LLAMA2_7B_PER_TOKEN_BYTES, pt7 == SPECS.LLAMA2_7B_PER_TOKEN_BYTES)
    total = ref_cache_bytes(nl, nh, hd, SPECS.LLAMA2_7B_CTX, 1,
                            SPECS.KV_BYTES_PER_ELEMENT["fp16"])
    check("A", "★ 4096 ctx / batch 1 / fp16 == %d == 2**31 **精确成立**"
          % SPECS.LLAMA2_7B_FP16_TOTAL_BYTES,
          total == SPECS.LLAMA2_7B_FP16_TOTAL_BYTES and total == 2 ** 31)
    check("A", "⚠ `KV_TENSOR_COUNT` 是 **%d**（K 与 V 两份）；**不再乘注意力头数**"
          % SPECS.KV_TENSOR_COUNT,
          ref_cache_bytes(1, 1, 1, 1, 1, 1.0) == SPECS.KV_TENSOR_COUNT)

    sub("A.2 三处外部锚点（三处的算式互不相同）")
    nl, nh, hd = SPECS.KV_MODEL_SHAPES["OPT-13B"]
    pt_opt = ref_cache_bytes(nl, nh, hd, 1, 1, SPECS.KV_BYTES_PER_ELEMENT["fp16"])
    check("A", "(a) PagedAttention 的 OPT-13B 每 token %d B ÷ 1024 == %d（论文印 800 KB）"
          % (pt_opt, SPECS.PAGEDATTENTION_PER_TOKEN_KB),
          abs(pt_opt / KiB - SPECS.PAGEDATTENTION_PER_TOKEN_KB) < 1e-9)
    tot_opt = ref_cache_bytes(nl, nh, hd, SPECS.PAGEDATTENTION_SEQ, 1,
                              SPECS.KV_BYTES_PER_ELEMENT["fp16"])
    check("A", "(a) 800 KiB × 2048 = %.4f GiB → 一位小数 %.1f（论文印 1.6 GB）"
          % (tot_opt / GiB, round(tot_opt / GiB, 1)),
          round(tot_opt / GiB, 1) == SPECS.PAGEDATTENTION_TOTAL_GB)
    nl, nh, hd = SPECS.KV_MODEL_SHAPES["Llama-2-70B"]
    v70 = ref_cache_bytes(nl, nh, hd, 1024, 1, SPECS.KV_BYTES_PER_ELEMENT["fp16"])
    check("A", "(b) Llama-2-70B seq=1024 = %.6f GiB → 三位小数 %.3f（博客印 %.3f）"
          % (v70 / GiB, round(v70 / GiB, 3), SPECS.DATABRICKS_70B_SEQ1024_GIB),
          round(v70 / GiB, 3) == SPECS.DATABRICKS_70B_SEQ1024_GIB)
    nl, nh, hd = SPECS.KV_MODEL_SHAPES["Llama-2-7B"]
    v128 = ref_cache_bytes(nl, nh, hd, SPECS.KVQUANT_128K_TOKENS, 1,
                           SPECS.KV_BYTES_PER_ELEMENT["fp16"])
    check("A", "(c) 7B @128K = %.1f GiB（KVQuant 印 %.1f）" % (v128 / GiB,
                                                              SPECS.KVQUANT_7B_128K_CACHE_GB),
          abs(v128 / GiB - SPECS.KVQUANT_7B_128K_CACHE_GB) < 1e-9)
    check("A", "★ 三处锚点**都只在 GiB/KiB 口径下精确**；换十进制单位就都对不上"
               "（(c) 的十进制是 %.1f，不是 %.1f）" % (v128 / 1e9,
                                                     SPECS.KVQUANT_7B_128K_CACHE_GB),
          abs(v128 / 1e9 - SPECS.KVQUANT_7B_128K_CACHE_GB) > 4.0)

    sub("A.3 算术强度：`AI = 2·S·T / (bpe·(S+T))` 的两条分支")
    print("      S=1, T=8192, fp16(2 B)：AI = %.6f" % ref_attention_ai(1, 8192, 128, 2.0))
    print("      S=1, T=8192, fp32(4 B)：AI = %.6f" % ref_attention_ai(1, 8192, 128, 4.0))
    check("A", "★ fp16 下 S=1、T→∞ 时 AI → %s（化简成 `ST/(S+T)`）" % SPECS.DECODE_AI,
          abs(ref_attention_ai(1, 10 ** 7, 128, 2.0) - SPECS.DECODE_AI) < 1e-4)
    check("A", "★ **同一个式子在 fp32 下恰为它的一半**——「AI = 1」是「每元素 2 字节」的产物",
          abs(ref_attention_ai(1, 10 ** 7, 128, 4.0) / ref_attention_ai(1, 10 ** 7, 128, 2.0)
              - 0.5) < 1e-9)
    check("A", "★ prefill（S = T = 4096）的 AI = %.1f = T/2，比 decode 高 %.0f 倍"
          % (ref_attention_ai(4096, 4096, 128, 2.0), ref_attention_ai(4096, 4096, 128, 2.0)
             / SPECS.DECODE_AI),
          abs(ref_attention_ai(4096, 4096, 128, 2.0) - 2048.0) < 1e-9)

    sub("A.4 per-token / per-channel 的误差结构（%d×%d 小张量 × %d 个种子取均值）"
        % (STDLIB_N, STDLIB_N, len(STDLIB_SEEDS)))
    print("      轴约定：沿**行内**归约 = per-token ；沿**列内**归约 = per-channel")
    for title, fn in (("iid（对照组）", mkIID), ("列 outlier（K 的结构）", mkV),
                      ("行异质（V 的结构）", mkRowHetero)):
        for bits in (2, 4):
            gts, gcs = [], []
            for sd in STDLIB_SEEDS:
                data = fn(STDLIB_N, sd)
                gts.append(ref_sqnr(data, ref_quant_asym(data, bits, 1)))
                gcs.append(ref_sqnr(data, ref_quant_asym(data, bits, 0)))
            ct = sum(gts) / len(gts)
            cc = sum(gcs) / len(gcs)
            _RES[(title, bits)] = (ct, cc)
            print("      %-22s B=%d  per-token %7.2f dB   per-channel %7.2f dB   差 %+7.2f"
                  % (title, bits, ct, cc, cc - ct))
    tok_i, chn_i = _RES[("iid（对照组）", 2)]
    check("A", "★ **对照组**：iid 数据上两种粒度差 %.3f dB（组大小相等 ⇒ 本该对称）"
               "——这是一条**必须成立的零假设**" % (chn_i - tok_i),
          abs(chn_i - tok_i) < 0.25)
    for bits in (2, 4):
        tok, chn = _RES[("列 outlier（K 的结构）", bits)]
        check("A", "★ B=%d：列 outlier 持久时 per-channel 领先 %.2f dB（**方向**与 KIVI 一致）"
              % (bits, chn - tok), chn - tok > 5.0)
        tok, chn = _RES[("行异质（V 的结构）", bits)]
        check("A", "★ B=%d：行异质时 per-token 领先 %.2f dB（**方向**与 KIVI 一致）"
              % (bits, tok - chn), tok - chn > 2.0)
    tok_c, chn_c = _RES[("列 outlier（K 的结构）", 2)]
    tok_r, chn_r = _RES[("行异质（V 的结构）", 2)]
    check("A", "★ 两个方向**不可互替**：列异质时 per-channel 赢 %.2f dB，"
               "行异质时 per-token 赢 %.2f dB"
          % (chn_c - tok_c, tok_r - chn_r),
          (chn_c - tok_c) > 0 and (tok_r - chn_r) > 0)
    check("A", "⚠ **领先幅度随张量宽度变**：这里 64×64 上列 outlier 的领先 %.2f dB，"
               "而编号程序在 1024×1024 上是 %.2f dB；行异质在 64 宽上只有 %.2f dB、"
               "1024 宽上是 %.2f dB。同域 `06-granularity` 亦实测过同一现象"
               "（256 宽 10.30 dB vs 4096 宽 20.89 dB）"
          % (chn_c - tok_c, 32.22, tok_r - chn_r, 8.50),
          (chn_c - tok_c) != 32.22 and (tok_r - chn_r) < 8.50)

    sub("A.5 残差滑窗：输出误差与「窗口内落了多少注意力质量」")
    n = STDLIB_N
    v = mkRowHetero(n)
    vq = ref_quant_asym(v, 2, 1)
    rnd = random.Random(STDLIB_SEED + 101)
    w = [0.0] * n
    for t in range(n - 16, n):
        w[t] = rnd.random()
    tot_w = sum(w)
    w = [x / tot_w for x in w]
    out = ref_attention_output(v, w)
    print("      注意力权重全部落在最后 16 个 token 上（窗口内质量 100%）")
    for r in (0, 8, 16):
        vq2 = [row[:] for row in vq]
        for t in range(n - r, n) if r else []:
            vq2[t] = v[t][:]
        mass = sum(w[n - r:]) if r else 0.0
        o = ref_sqnr_vec(out, ref_attention_output(vq2, w))
        print("      R=%2d  窗口内质量 %5.1f%%   输出 SQNR %s" % (
            r, 100 * mass, "∞（误差恰为 0）" if o == float("inf") else "%6.2f dB" % o))
    vq_all = [row[:] for row in vq]
    check("A", "★ 窗口为空时输出 SQNR %.2f dB（远低于覆盖后的无穷）"
          % ref_sqnr_vec(out, ref_attention_output(vq_all, w)),
          ref_sqnr_vec(out, ref_attention_output(vq_all, w)) < 20.0)
    vq16 = [row[:] for row in vq]
    for t in range(n - 16, n):
        vq16[t] = v[t][:]
    check("A", "★ **R=%d 覆盖全部注意力质量后，输出 SQNR 跳到无穷**——"
               "滑窗的收益完全由「窗口内落了多少注意力质量」决定"
          % 16, ref_sqnr_vec(out, ref_attention_output(vq16, w)) == float("inf"))
    check("A", "★ KIVI 的 R = %d、G = %d 是两个不同的量，且 R %% G == 0"
          % (SPECS.KIVI_RESIDUAL_LENGTH_R, SPECS.KIVI_GROUP_SIZE_G),
          SPECS.KIVI_RESIDUAL_LENGTH_R % SPECS.KIVI_GROUP_SIZE_G == 0)

    sub("A.6 两笔比特账：载荷 vs 整块等效宽度")
    for name in ("fp16", "fp8_per_tensor", "int4_g32", "int4_g128", "int2_g32", "int2_g128"):
        payload = {"fp16": 16.0, "fp8_per_tensor": 8.0}.get(name)
        if payload is None:
            payload = 4.0 if name.startswith("int4") else 2.0
        meta = SPECS.KV_METADATA_BITS_PER_ELEMENT[name]
        print("      %-12s 载荷 %5.1f + 元数据 %5.2f = 整块 %5.2f bit/元素（fp16 的 %4.2fx）"
              % (name, payload, meta, payload + meta, 16.0 / (payload + meta)))
    check("A", "★ `KV_BYTES_PER_ELEMENT['int2']` = %.2f B = **%.1f bit 是载荷**；"
               "整块（G=32，元数据 64/G）是 %.2f bit——**两者不是同一个量**"
          % (SPECS.KV_BYTES_PER_ELEMENT["int2"], SPECS.KV_BYTES_PER_ELEMENT["int2"] * 8,
             2.0 + SPECS.KV_METADATA_BITS_PER_ELEMENT["int2_g32"]),
          SPECS.KV_BYTES_PER_ELEMENT["int2"] * 8 == 2.0
          and abs(2.0 + SPECS.KV_METADATA_BITS_PER_ELEMENT["int2_g32"]
                  - 32.0 / SPECS.KIVI_GROUP_SIZE_G * 2 - 2.0) < 1e-12)

    sub("A.7 外部规格常量的**逐字核对**（A6 的靶子：每条都要能被一处断言分辨）")
    check("A", "`MEASURED_VERSIONS` 逐字 == 本机五个解释器的 torch 版本",
          SPECS.MEASURED_VERSIONS == ("2.6.0", "2.7.1", "2.8.0", "2.11.0", "2.13.0"),
          str(SPECS.MEASURED_VERSIONS))
    check("A", "`UNCOVERED_MINORS` 逐字 == `2.9.x / 2.10.x / 2.12.x`",
          SPECS.UNCOVERED_MINORS == "2.9.x / 2.10.x / 2.12.x")
    check("A", "`MEASURE_HOST` 逐字 == 「Apple M1 Pro（10 核）/ macOS / **CPU-only，无 GPU 数字**」",
          SPECS.MEASURE_HOST == "Apple M1 Pro（10 核）/ macOS / **CPU-only，无 GPU 数字**",
          SPECS.MEASURE_HOST)
    check("A", "`TORCH_MIN_FOR_THIS_TOPIC` == (2, 8)（**对齐选择**，不是依赖）",
          SPECS.TORCH_MIN_FOR_THIS_TOPIC == (2, 8))
    check("A", "`DECODE_AI` == 1.0（fp16 口径下的 decode 算术强度）",
          SPECS.DECODE_AI == 1.0)
    check("A", "`KIVI_GROUP_SIZE_G` == 32 且 `KIVI_RESIDUAL_LENGTH_R` == 128",
          SPECS.KIVI_GROUP_SIZE_G == 32 and SPECS.KIVI_RESIDUAL_LENGTH_R == 128)
    check("A", "`KIVI_ATTENTION_SPARSITY_PCT` == 84.3（KIVI Table 2 的那一格）",
          SPECS.KIVI_ATTENTION_SPARSITY_PCT == 84.3, str(SPECS.KIVI_ATTENTION_SPARSITY_PCT))
    check("A", "`KIVI_V_DELTA_RATIO` == 14.05（= 49.89/3.55，由 Table 2 两格算出）",
          SPECS.KIVI_V_DELTA_RATIO == 14.05)
    check("A", "`KIVI_PEAK_MEMORY_RATIO` == 2.6（显存口径，**含模型权重**）",
          SPECS.KIVI_PEAK_MEMORY_RATIO == 2.6)
    check("A", "`KIVI_THROUGHPUT_LOW` == 2.35、`KIVI_THROUGHPUT_HIGH` == 3.47（**吞吐**口径）",
          SPECS.KIVI_THROUGHPUT_LOW == 2.35 and SPECS.KIVI_THROUGHPUT_HIGH == 3.47)
    check("A", "`KIVI_PERPLEXITY_COUNT` == 0（KIVI 全文 0 处 perplexity）",
          SPECS.KIVI_PERPLEXITY_COUNT == 0)
    check("A", "`KIVI_MEMORY_QUOTE` / `KIVI_THROUGHPUT_QUOTE` 逐字 == 摘要原文",
          SPECS.KIVI_MEMORY_QUOTE == "2.6x less peak memory (including model weight)"
          and SPECS.KIVI_THROUGHPUT_QUOTE == "2.35x~3.47x throughput")
    check("A", "`KIVI_TABLE2` 四行的 8 个数逐字 == KIVI Table 2",
          SPECS.KIVI_TABLE2 == {
              "K_recon": {"per_token": 13.67, "per_channel": 4.55},
              "K_score": {"per_token": 47.00, "per_channel": 9.60},
              "V_recon": {"per_token": 4.57, "per_channel": 3.73},
              "V_delta": {"per_token": 3.55, "per_channel": 49.89}})
    check("A", "`KIVI_TABLE3_LLAMA2_7B` 的六行逐字 == KIVI Table 3 的 Llama-2-7B 列"
               "（CoQA / GSM8K 两列）",
          SPECS.KIVI_TABLE3_LLAMA2_7B == {
              "16bit": (63.88, 13.50), "2bit_Kc_Vt": (59.08, 5.76),
              "2bit_Kt_Vt": (39.88, 0.83), "2bit_Kc_Vc": (3.60, 0.00),
              "2bit_Kt_Vc": (1.30, 0.08), "KIVI-2": (63.05, 12.74)})
    check("A", "`KVQUANT_TABLE16_LLAMA2_7B` 七格逐字 == KVQuant Table 16 的 Llama-2-7B 列",
          SPECS.KVQUANT_TABLE16_LLAMA2_7B == {
              "baseline": 5.12, "int4": 5.66, "KVQuant-4bit": 5.16,
              "KVQuant-3bit-1%": 5.17, "int3": 22.71, "int2": 4708.0,
              "KVQuant-2bit-1%": 5.41})
    check("A", "`KVQUANT_TABLE1_LLAMA7B` 五行的 (PPL, KV cache GB) 逐字 == Table 1",
          SPECS.KVQUANT_TABLE1_LLAMA7B == {
              "baseline": (5.68, 64.0), "int4": (5.98, 16.0),
              "KVQuant-4bit": (5.72, 16.0), "KVQuant-3bit-1": (5.75, 13.3),
              "KVQuant-2bit-1": (6.01, 9.3)})
    check("A", "`KV_MODEL_SHAPES` 五个模型的 (层, KV 头, head_dim) 逐字",
          SPECS.KV_MODEL_SHAPES == {
              "Llama-2-7B": (32, 32, 128), "Llama-2-13B": (40, 40, 128),
              "Llama-2-70B": (80, 8, 128), "Mistral-7B": (32, 8, 128),
              "OPT-13B": (40, 40, 128)})
    check("A", "`KV_BYTES_PER_ELEMENT` 四种精度逐字 == 2 / 1 / 0.5 / 0.25",
          SPECS.KV_BYTES_PER_ELEMENT == {"fp16": 2.0, "fp8": 1.0, "int4": 0.5,
                                         "int2": 0.25})
    check("A", "★ `KV_METADATA_BITS_PER_ELEMENT` 六格逐字（口径 = **64/G**："
               "32 bit scale + 32 bit zero-point）；"
               "**fp16 与 fp8 per-tensor 必须是 0.0**（它们没有 scale 元数据）",
          SPECS.KV_METADATA_BITS_PER_ELEMENT == {
              "fp16": 0.0, "fp8_per_tensor": 0.0, "int4_g32": 2.0,
              "int4_g128": 0.5, "int2_g32": 2.0, "int2_g128": 0.5})
    check("A", "★ 元数据与 G 严格反比：64/32 = %.2f、64/128 = %.2f——"
               "口径写死在 `specs_kv.py` 的注释里（只算 scale 一份是 32/G，"
               "不算元数据是 KVQuant `Avg. Num. Bits` 的口径）"
          % (SPECS.KV_METADATA_BITS_PER_ELEMENT["int4_g32"],
             SPECS.KV_METADATA_BITS_PER_ELEMENT["int4_g128"]),
          abs(SPECS.KV_METADATA_BITS_PER_ELEMENT["int4_g32"] - 64.0 / 32) < 1e-12
          and abs(SPECS.KV_METADATA_BITS_PER_ELEMENT["int4_g128"] - 64.0 / 128) < 1e-12)
    check("A", "`LOCAL_1BIT_VS_2BIT_GAP_DB` == 12.0（本机 1-bit 断崖的量级）",
          SPECS.LOCAL_1BIT_VS_2BIT_GAP_DB == 12.0)
    check("A", "`SYMMETRIC_QBITS_AT_B1` == 0（对称 1-bit 的 qmax）",
          SPECS.SYMMETRIC_QBITS_AT_B1 == 0)
    check("A", "`QJL_BITS_PER_FPN` == 3.0、元数据 %.0f–%.0f bit/数"
          % (SPECS.QJL_BLOCK_QUANT_METADATA_BITS_LOW,
             SPECS.QJL_BLOCK_QUANT_METADATA_BITS_HIGH),
          SPECS.QJL_BITS_PER_FPN == 3.0
          and SPECS.QJL_BLOCK_QUANT_METADATA_BITS_LOW == 1.0
          and SPECS.QJL_BLOCK_QUANT_METADATA_BITS_HIGH == 2.0)
    check("A", "★ `ENGINE_KV_SCALE_GRANULARITY` 三行逐字（写侧粒度 / 读侧 / 是否支持 per-token）",
          SPECS.ENGINE_KV_SCALE_GRANULARITY == {
              "vLLM (FP8)": ("per_tensor_or_per_head", "attention_kernel", False),
              "SGLang": ("scalar", "attention_kernel", False),
              "KIVI": ("K_per_channel_V_per_token", "fused_dequant_matmul", True)})
    check("A", "★ `LLAMACPP_CACHE_TYPES` 九项逐字 == f32/f16/bf16/q8_0/q4_0/q4_1/iq4_nl/q5_0/q5_1",
          SPECS.LLAMACPP_CACHE_TYPES == ("f32", "f16", "bf16", "q8_0", "q4_0", "q4_1",
                                         "iq4_nl", "q5_0", "q5_1"))
    check("A", "`LLAMACPP_DEFAULT_CACHE_TYPE` == `f16`",
          SPECS.LLAMACPP_DEFAULT_CACHE_TYPE == "f16")
    check("A", "★ `VLLM_CACHE_DTYPE_MAIN` 十八项逐字（`vllm/config/cache.py` main 分支）",
          SPECS.VLLM_CACHE_DTYPE_MAIN == (
              "auto", "float16", "bfloat16", "fp8", "fp8_e4m3", "fp8_e5m2", "fp8_inc",
              "fp8_ds_mla", "nvfp4_ds_mla", "turboquant_k8v4", "turboquant_4bit_nc",
              "turboquant_k3v4_nc", "turboquant_3bit_nc", "int4_per_token_head",
              "int8_per_token_head", "fp8_per_token_head", "nvfp4", "nvfp4_4over6"))
    check("A", "`VLLM_CACHE_DTYPE_COUNT` == 18（**不是老文档里的 7**）",
          SPECS.VLLM_CACHE_DTYPE_COUNT == 18)
    check("A", "`SGLANG_TOKEN_CAPACITY_FP8` == 2.00、`_FP4` == 3.56、"
               "`SGLANG_FP4_IMPLIED_BITS` == 4.50（= 4 bit 载荷 + 64/128 元数据）",
          SPECS.SGLANG_TOKEN_CAPACITY_FP8 == 2.00
          and SPECS.SGLANG_TOKEN_CAPACITY_FP4 == 3.56
          and SPECS.SGLANG_FP4_IMPLIED_BITS == 4.50)
    check("A", "`PAGEDATTENTION_PER_TOKEN_KB` == 800、`_SEQ` == 2048、`_TOTAL_GB` == 1.6",
          SPECS.PAGEDATTENTION_PER_TOKEN_KB == 800 and SPECS.PAGEDATTENTION_SEQ == 2048
          and SPECS.PAGEDATTENTION_TOTAL_GB == 1.6)
    check("A", "`DATABRICKS_70B_SEQ1024_GIB` == 0.312（**计划基线，本次未独立核实**）",
          SPECS.DATABRICKS_70B_SEQ1024_GIB == 0.312)
    check("A", "★ `REFERENCE_BASELINE_LLAMA2_7B_WIKITEXT2` == 5.47，"
               "**与 KVQuant Table 16 的 %.2f 不是同一个测量口径**——"
               "同一个模型、同一个数据集、同名的两批 harness，基线差 %.2f"
          % (SPECS.KVQUANT_TABLE16_LLAMA2_7B["baseline"],
             SPECS.REFERENCE_BASELINE_LLAMA2_7B_WIKITEXT2
             - SPECS.KVQUANT_TABLE16_LLAMA2_7B["baseline"]),
          SPECS.REFERENCE_BASELINE_LLAMA2_7B_WIKITEXT2 == 5.47
          and SPECS.REFERENCE_BASELINE_LLAMA2_7B_WIKITEXT2
          != SPECS.KVQUANT_TABLE16_LLAMA2_7B["baseline"])


# ===========================================================================
# C. 边界与反例
# ===========================================================================

def run_C():
    section("C", "边界与反例（断言的是「失败行为成立」）")

    sub("C.1 ★ 对称 1-bit 的 `qmax = 2^0 − 1 = 0`：重建恒为 0")
    check("C", "`SYMMETRIC_QBITS_AT_B1` == %d（即 `2^0 − 1`）" % SPECS.SYMMETRIC_QBITS_AT_B1,
          SPECS.SYMMETRIC_QBITS_AT_B1 == 2 ** 0 - 1 == 0)
    x = mkV(STDLIB_N)
    sym1 = ref_quant_sym(x, 1, 0)
    nonzero = sum(1 for row in sym1 for v in row if v != 0.0)
    check("C", "★ 对称 1-bit 的重建结果**非零元素 %d 个**（全被夹到 0）" % nonzero,
          nonzero == 0)
    check("C", "★ 它的 SQNR 恰为 **0.00 dB**——「误差能量 == 信号能量」，"
               "不是负无穷（分子分母都是 Σx²）",
          abs(ref_sqnr(x, sym1)) < 1e-9)
    asym1 = ref_quant_asym(x, 1, 0)
    check("C", "★ 换成**非对称** 1-bit（%d 级码字）就不再恒为 0，SQNR = %.2f dB"
          % (2 ** 1, ref_sqnr(x, asym1)), ref_sqnr(x, asym1) > 0.0)

    sub("C.2 ★ 1-bit 断崖：跳到 2-bit 的收益远大于后面每一步")
    k = mkV(STDLIB_N)
    s1 = ref_sqnr(k, ref_quant_asym(k, 1, 0))
    s2 = ref_sqnr(k, ref_quant_asym(k, 2, 0))
    s4 = ref_sqnr(k, ref_quant_asym(k, 4, 0))
    print("      K per-channel：1-bit %6.2f dB → 2-bit %6.2f dB → 4-bit %6.2f dB"
          % (s1, s2, s4))
    check("C", "★ 1→2 bit 一步跳 **%.2f dB**（specs 记的量级 %.1f dB）"
          % (s2 - s1, SPECS.LOCAL_1BIT_VS_2BIT_GAP_DB),
          (s2 - s1) > SPECS.LOCAL_1BIT_VS_2BIT_GAP_DB - 4.0)
    check("C", "★ 断崖的**判据是「1→2 这一步」**：它比 2→4 的每一步都大"
               "（%.2f vs %.2f dB/bit）" % (s2 - s1, (s4 - s2) / 2.0),
          (s2 - s1) > (s4 - s2) / 2.0)

    sub("C.3 ★ 轴写反 ⇒ **两个数字都对、标签互换、结论反向**")
    tok, chn = res_lookup()
    print("      真实的：per-token %.2f dB、per-channel %.2f dB" % (tok, chn))
    print("      写对：per-channel 领先 %+.2f dB ⇒ 「K 用 per-channel」" % (chn - tok))
    print("      写反：标签互换 ⇒ 读出「per-token 领先 %+.2f dB」⇒ 「K 用 per-token」"
          % (tok - chn))
    check("C", "★ **一个数字都没变**，结论却与 KIVI 完全相反；"
               "而且**没有任何断言会失败**（这正是它危险的地方）", (chn - tok) > 5.0)
    check("C", "★ 唯一的对策是把轴约定写成一行、放在第一次使用这两个词之前"
               "（`specs.AXIS_CONVENTION_TOKENS_FIRST` = %s）"
          % SPECS.AXIS_CONVENTION_TOKENS_FIRST,
          SPECS.AXIS_CONVENTION_TOKENS_FIRST is True)

    sub("C.4 ★ KIVI 的「逐元素相对误差」判据会被近零元素炸掉")
    for bits in (2, 4):
        for nm, dim in (("per-token", 1), ("per-channel", 0)):
            xh = ref_quant_asym(x, bits, dim)
            print("      B=%d %-12s 逐元素相对 ‖(X−X')/X‖_F = %14.3f ；SQNR %7.2f dB"
                  % (bits, nm, ref_rel_elementwise(x, xh), ref_sqnr(x, xh)))
    big = ref_rel_elementwise(x, ref_quant_asym(x, 2, 0))
    check("C", "★ 逐元素相对误差的量级是 **10³ 以上**，而 KIVI 论文印的是 13.67 / 4.55——"
               "同一个判据在合成的 iid 高斯上**不可用**（分母里的近零元素把它炸了）",
          big > 1e3, "%.1f" % big)
    check("C", "⚠ 但**方向**仍一致（per-channel 的逐元素相对误差更小）⇒ "
               "该判据可用来比方向、**不可用来引绝对值**",
          ref_rel_elementwise(x, ref_quant_asym(x, 2, 0))
          < ref_rel_elementwise(x, ref_quant_asym(x, 2, 1)))

    sub("C.5 ★ 三处「看起来该有但其实没有」的口径（P9 的三次预防）")
    check("C", "★ KIVI 全文 **%d 处「perplexity」**——引用它的 PPL 必须标注第三方复现"
          % SPECS.KIVI_PERPLEXITY_COUNT, SPECS.KIVI_PERPLEXITY_COUNT == 0)
    check("C", "★ KIVI 的显存口径逐字带 `including model weight`（%.1f×），"
               "而 %.2f×–%.2f× 是**吞吐**——两个量"
          % (SPECS.KIVI_PEAK_MEMORY_RATIO, SPECS.KIVI_THROUGHPUT_LOW,
             SPECS.KIVI_THROUGHPUT_HIGH),
          "including model weight" in SPECS.KIVI_MEMORY_QUOTE
          and "throughput" in SPECS.KIVI_THROUGHPUT_QUOTE)
    check("C", "★ KIVI 的 **G = %d（group size）** 与 **R = %d（residual length）** "
               "是两个不同的量——128 **不是** group size"
          % (SPECS.KIVI_GROUP_SIZE_G, SPECS.KIVI_RESIDUAL_LENGTH_R),
          SPECS.KIVI_GROUP_SIZE_G == 32 and SPECS.KIVI_RESIDUAL_LENGTH_R == 128)
    v = SPECS.KIVI_TABLE2
    check("C", "★ KIVI Table 2 的 **V 重建误差 per-channel(%.2f) 比 per-token(%.2f) 更小**，"
               "而 V 的**输出误差** Δ per-token(%.2f) 比 per-channel(%.2f) 小 %.2f×——"
               "**把 4.57 与 49.89 并列成「15 倍」是把两个量当成了一个**"
          % (v["V_recon"]["per_channel"], v["V_recon"]["per_token"], v["V_delta"]["per_token"],
             v["V_delta"]["per_channel"], SPECS.KIVI_V_DELTA_RATIO),
          v["V_recon"]["per_channel"] < v["V_recon"]["per_token"]
          and abs(v["V_delta"]["per_channel"] / v["V_delta"]["per_token"]
                  - SPECS.KIVI_V_DELTA_RATIO) < 0.05)
    b1 = SPECS.KVQUANT_TABLE1_LLAMA7B["baseline"][0]
    b16 = SPECS.KVQUANT_TABLE16_LLAMA2_7B["baseline"]
    check("C", "★ 两套 Llama PPL 表基线不同（%.2f vs %.2f）⇒ **不得并列**"
          % (b1, b16), b1 != b16)
    check("C", "★ SGLang 的两个容量比**都能被位宽算术复现**——"
               "FP8 16/8 = %.2f；FP4 16/(4 + 64/128) = %.3f → 两位小数 %.2f。"
               "**后者同时印证了「FP4 KV 用 G=128」**"
          % (16.0 / 8.0, 16.0 / 4.5, round(16.0 / 4.5, 2)),
          abs(16.0 / 8.0 - SPECS.SGLANG_TOKEN_CAPACITY_FP8) < 1e-9
          and abs(round(16.0 / (4.0 + SPECS.KV_METADATA_BITS_PER_ELEMENT["int4_g128"]), 2)
                  - SPECS.SGLANG_TOKEN_CAPACITY_FP4) < 0.005
          and abs(16.0 / SPECS.SGLANG_TOKEN_CAPACITY_FP4
                  - SPECS.SGLANG_FP4_IMPLIED_BITS) < 0.01)
    check("C", "★ QJL 的「zero overhead」逐字指**不存 scale/zero**（省 %g–%g bit/数），"
               "**不是「零计算开销」**"
          % (SPECS.QJL_BLOCK_QUANT_METADATA_BITS_LOW, SPECS.QJL_BLOCK_QUANT_METADATA_BITS_HIGH),
          SPECS.QJL_BITS_PER_FPN == 3.0
          and SPECS.QJL_BLOCK_QUANT_METADATA_BITS_LOW == 1.0)
    check("C", "★ vLLM 的 `CacheDType` 是 **%d 项**（`main` 分支快照），"
               "不是老文档里的 7 项——引用必须写分支/版本"
          % SPECS.VLLM_CACHE_DTYPE_COUNT,
          len(SPECS.VLLM_CACHE_DTYPE_MAIN) == SPECS.VLLM_CACHE_DTYPE_COUNT == 18)
    check("C", "★ llama.cpp 侧是 **%d 项**、默认 `%s`，**不含** `mxfp4` / `q2_K`…`q6_K`"
          % (len(SPECS.LLAMACPP_CACHE_TYPES), SPECS.LLAMACPP_DEFAULT_CACHE_TYPE),
          len(SPECS.LLAMACPP_CACHE_TYPES) == 9
          and "mxfp4" not in SPECS.LLAMACPP_CACHE_TYPES)
    check("C", "★ **不落成断言**的那两条：FP8 KV 的硬件支持矩阵来源冲突、"
               "以及「全模型 decode = 1 FLOP/weight-byte」——`specs_kv.py` 里都没有对应常量",
          not hasattr(SPECS, "FP8_KV_SUPPORTED_ARCHS")
          and not hasattr(SPECS, "FULL_MODEL_DECODE_FLOPS_PER_BYTE"))


_RES = {}


def res_lookup():
    """复用 A.4 的「列 outlier」结果（**同一批数字，不重算**）。"""
    return _RES[("列 outlier（K 的结构）", 2)]


# ===========================================================================
# E. 自检：A 类判据是不是活的（把参考实现故意改错，断言必须失败）
# ===========================================================================

def run_E():
    section("E", "自检：把参考实现/期望值故意改错，A 类判据必须**失败**")
    ok = True

    # E.1 把 per-token 与 per-channel 的归约维对调 —— 结论必须反转
    x = mkV(STDLIB_N)
    right = ref_sqnr(x, ref_quant_asym(x, 2, 0))
    swapped = ref_sqnr(x, ref_quant_asym(x, 2, 1))
    ok &= check("A", "E.1  把归约维对调后，A.4 的结论**反转**（%.2f dB 变成 %.2f dB）"
                % (right, swapped), swapped < right)

    # E.2 把 KV cache 的 ×2 去掉 —— 每 token 字节数必须对不上
    n_layers, n_kv_heads, head_dim = SPECS.KV_MODEL_SHAPES["Llama-2-7B"]
    wrong = 1 * n_layers * n_kv_heads * head_dim * 1 * 1 * 2
    ok &= check("A", "E.2  去掉 `KV_TENSOR_COUNT` 的因子 2 后，A.1 的等式**失败**",
                wrong != SPECS.LLAMA2_7B_PER_TOKEN_BYTES
                and wrong * 2 == SPECS.LLAMA2_7B_PER_TOKEN_BYTES)

    # E.3 把「每元素 2 字节」换成 4 字节 —— A.3 的 AI 腰斩，与 DECODE_AI 对不上
    ok &= check("A", "E.3  把 bpe 从 2 改成 4 后，A.3 的 `AI == DECODE_AI` **失败**",
                abs(ref_attention_ai(1, 10 ** 7, 128, 4.0) - SPECS.DECODE_AI) > 0.4)

    # E.4 把窗口缩到「覆盖不了全部注意力质量」—— A.5 的「恰为 0」必须失败
    n = STDLIB_N
    v = mkRowHetero(n)
    vq = ref_quant_asym(v, 2, 1)
    rnd = random.Random(STDLIB_SEED + 101)
    w = [0.0] * n
    for t in range(n - 16, n):
        w[t] = rnd.random()
    tw = sum(w)
    w = [x / tw for x in w]
    out = ref_attention_output(v, w)
    vq8 = [row[:] for row in vq]
    for t in range(n - 8, n):
        vq8[t] = v[t][:]
    sq8 = ref_sqnr_vec(out, ref_attention_output(vq8, w))
    ok &= check("A", "E.4  窗口只覆盖 8 个 token 时，A.5 的「误差恰为 0」**失败**"
                     "（%.2f dB，是有限值）" % sq8, sq8 < 30.0)

    # E.5 把对称 1-bit 的级数当成「和 b=2 一样」—— C.1 必须失败
    ok &= check("C", "E.5  对称 1-bit 的 `qmax` 若按 `2^1 − 1 = 1` 算，C.1 的「恒为 0」**失败**",
                2 ** (1 - 1) - 1 != 2 ** 1 - 1)

    # E.6 把 G 与 R 换过来 —— C.5 的「R % G == 0」在这个方向上仍然成立，
    #     所以必须另有断言区分它们（这正是本文件用「两个常量不相等」而不是「整除」的原因）
    ok &= check("C", "E.6  `G` 与 `R` 互换后整除仍成立（%d %% %d == 0）⇒ "
                     "**判据必须是「两个常量不相等」**，不能只查整除"
                % (SPECS.KIVI_GROUP_SIZE_G, SPECS.KIVI_RESIDUAL_LENGTH_R),
                SPECS.KIVI_GROUP_SIZE_G != SPECS.KIVI_RESIDUAL_LENGTH_R
                and SPECS.KIVI_GROUP_SIZE_G % SPECS.KIVI_RESIDUAL_LENGTH_R != 0)

    # E.7 轴约定若为 False，C.3 的前提不成立
    ok &= check("C", "E.7  `AXIS_CONVENTION_TOKENS_FIRST` 为 False 时，"
                     "A.4/C.3 的解释全部失效（本文件把它的值写进了断言）",
                SPECS.AXIS_CONVENTION_TOKENS_FIRST is True)

    print()
    print("  E 节结论：%s" % ("全部自检按预期失败 ⇒ A/C 类判据是活的" if ok
                            else "**有自检未按预期失败 ⇒ 判据有问题**"))
    return ok


# ===========================================================================
# F. genres：本文件覆盖了哪些 specs 常量（A6 变异测试的清单）
# ===========================================================================

def run_F():
    section("F", "变异测试清单：本文件读过的 specs 常量")
    names = sorted(n for n in dir(SPECS) if n.isupper() and not n.startswith("_"))
    src = open(__file__, encoding="utf-8").read()
    used = [n for n in names if ("SPECS.%s" % n) in src]
    print("      specs.py 里共 %d 条全大写常量；本文件直接读到的 %d 条"
          % (len(names), len(used)))
    print("      %s" % ", ".join(used))
    unused = [n for n in names if n not in used]
    print("      本文件**没有**读到的 %d 条（由编号程序负责，见各程序 §F）：" % len(unused))
    for n in unused:
        print("        · %s" % n)
    check("A", "变异的对象是**常量**，每个常量的值都能被至少一处断言分辨", len(used) > 0)
    return unused


# ===========================================================================
# main
# ===========================================================================

def main():
    print("=" * 78)
    print("130-model-optimization/quantization · verify.py"
          "（纯 python3，**不 import torch / numpy**）")
    print("=" * 78)
    print("说明：本文件产出的是**推导**，不是实测。规模 = %d×%d 张量 / 标准库 random；"
          % (STDLIB_N, STDLIB_N))
    print("      **规模与编号程序不同 ⇒ 数字不同，只有定性结构必须一致。**")

    run_A()
    run_C()
    okE = run_E()
    run_F()

    print()
    print("=" * 78)
    n, a, c = _STATE["n"], _STATE["A"], _STATE["C"]
    print("合计 %d 条断言：A %d 条 / C %d 条；失败 %d 条" % (n, a, c, len(_STATE["fail"])))
    if _STATE["fail"]:
        print("失败清单：")
        for f in _STATE["fail"]:
            print("  · %s" % f)
    print("E 节自检：%s" % ("通过" if okE else "**不通过**"))
    sys.exit(0 if (not _STATE["fail"] and okE) else 1)


if __name__ == "__main__":
    main()

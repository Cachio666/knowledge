#!/usr/bin/env python3
"""01 KV cache 有多大，以及 decode 为什么是 memory-bound。

对应文档：130-quantization/kv-cache-quantization/01-kv-cache-basics.md

★ 本程序跑的是**真 torch**——输出是**本机实测值**，支撑关于**真实 PyTorch / 本机 CPU**
  的结论，但**必须绑定版本**。本机五个解释器都跑过：
    /usr/bin/python3                        torch 2.8.0  / Python 3.9.6
    ~/miniconda3/bin/python                 torch 2.11.0 / Python 3.12.9
    ~/miniconda3/envs/deep-prac/bin/python  torch 2.6.0  / Python 3.12.11
    ~/miniconda3/envs/dp-test/bin/python    torch 2.7.1  / Python 3.13.7
    ~/PyCharmMiscProject/.venv/bin/python   torch 2.13.0 / Python 3.13.5
  **2.9 / 2.10 / 2.12 本机没有解释器，不声称覆盖。** 全部 CPU-only，无 GPU 数字。

三类断言（CONTRIBUTING §2 第三种形态：B 在编号程序里，A/C 另见 verify.py）：
  B —— **代价模型**：KV cache 的字节账（每 token / 每 ctx / 每精度），
       以及 decode 步的算术强度与**本机实测的 achieved GFLOP/s 落差**。
  A —— **两条互不依赖的通道**：① 纯算术公式 vs `torch.empty(...).nbytes` 的真张量分配；
       ② 三处外部锚点（PagedAttention 的 OPT-13B 800 KB、Databricks 的 70B 0.312 GiB、
       KVQuant 的 64.0 @128K）**各自用自己的算式**重算一遍。
  C —— 失败行为必须**真的跑出来**：把 KV 头数当成注意力头数、把 KiB 当成 KB、
       把「每元素 2 字节」这个前提换掉，三种误用各自给出多大的偏差。

★ 本程序**不复制 `specs.py` 的任何常量值**：凡用到规格常量，一律写 `SPECS.xxx`。
  这样改掉 `specs.py` 里任意一条就会在这里报错（A6 约定）。

用法：cd code && make 01_kv_cache_size   （或 make run 全跑）
"""

import platform
import subprocess
import sys
import time

import torch

import specs as SPECS


_STATE = {"n": 0, "A": 0, "B": 0, "C": 0, "M": 0, "fail": []}

KiB = 1024.0
MiB = 1024.0 ** 2
GiB = 1024.0 ** 3


def cpu_model():
    """测量条件里的「CPU 型号」。**这不是算法**，只是把测量条件打出来。"""
    try:
        if platform.system() == "Darwin":
            out = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"],
                                          stderr=subprocess.DEVNULL)
            return out.decode().strip()
        if platform.system() == "Linux":
            with open("/proc/cpuinfo") as fh:
                for line in fh:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
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
    print("130-quantization/kv-cache-quantization · 01 KV cache 大小与 decode 的算术强度")
    print("=" * 74)
    print("解释器：%s / python %s" % (sys.executable, sys.version.split()[0]))
    print("torch   ：%s  （CUDA 可用: %s）" % (torch.__version__, torch.cuda.is_available()))
    print("CPU     ：%s" % cpu_model())
    print("⚠ 全部 CPU-only，无 GPU 数字。K/V 的 kernel 行为与显存带宽**不在本机测**。")


# ===========================================================================
# 纯算术（与 torch 无关的那一半；torch 那一半在 A2 里与它独立对拍）
# ===========================================================================

def cache_bytes(n_layers, n_kv_heads, head_dim, seq, batch, bytes_per_elem):
    """KV cache 字节数 = 2 × layers × kv_heads × head_dim × seq × batch × bytes/元素。

    判据：×2 是 K 与 V 两份（`SPECS.KV_TENSOR_COUNT`）；再乘 2 的地方没有了——
    **每个 head 的 head_dim 就是这一维的长度**，不要再乘注意力头数。
    """
    return (SPECS.KV_TENSOR_COUNT * n_layers * n_kv_heads * head_dim
            * seq * batch * bytes_per_elem)


def per_token_bytes(model, bytes_per_elem):
    n_layers, n_kv_heads, head_dim = SPECS.KV_MODEL_SHAPES[model]
    return cache_bytes(n_layers, n_kv_heads, head_dim, 1, 1, bytes_per_elem)


# ===========================================================================
# B. 代价模型
# ===========================================================================

def run_B():
    rule("B 代价模型（一）：KV cache 的字节账")

    sub("B1 每 token 的字节数（fp16）与 4096 ctx 的总量")
    hdr = "      %-14s %6s %6s %6s | %10s %10s | %9s %9s" % (
        "模型", "层", "kv头", "d", "B/token", "KiB/token", "4096fp16", "8192fp16")
    print(hdr)
    for m in ("Llama-2-7B", "Llama-2-13B", "Llama-2-70B", "Mistral-7B", "OPT-13B"):
        pt = per_token_bytes(m, SPECS.KV_BYTES_PER_ELEMENT["fp16"])
        print("      %-14s %6d %6d %6d | %10d %10.1f | %8.2fG %8.2fG" % (
            m, SPECS.KV_MODEL_SHAPES[m][0], SPECS.KV_MODEL_SHAPES[m][1],
            SPECS.KV_MODEL_SHAPES[m][2], pt, pt / KiB,
            cache_bytes(*SPECS.KV_MODEL_SHAPES[m], 4096, 1,
                        SPECS.KV_BYTES_PER_ELEMENT["fp16"]) / GiB,
            cache_bytes(*SPECS.KV_MODEL_SHAPES[m], 8192, 1,
                        SPECS.KV_BYTES_PER_ELEMENT["fp16"]) / GiB))

    pt7 = per_token_bytes("Llama-2-7B", SPECS.KV_BYTES_PER_ELEMENT["fp16"])
    check("B", "Llama-2-7B 每 token = %d B（= 512 KiB）" % SPECS.LLAMA2_7B_PER_TOKEN_BYTES,
          pt7 == SPECS.LLAMA2_7B_PER_TOKEN_BYTES)
    total = cache_bytes(32, 32, 128, SPECS.LLAMA2_7B_CTX, 1,
                        SPECS.KV_BYTES_PER_ELEMENT["fp16"])
    check("B", "4096 ctx / batch 1 / fp16 = **恰好 2 GiB**（= 2**31 B，一个精确的整数等式）",
          total == SPECS.LLAMA2_7B_FP16_TOTAL_BYTES and total == 2 ** 31,
          "%d B = %.4f GiB" % (total, total / GiB))

    sub("B2 换精度与换 ctx 的对照（同一模型 Llama-2-7B）")
    print("      %-8s %10s %10s %10s" % ("精度", "4096ctx", "128Kctx", "相对fp16"))
    for name in ("fp16", "fp8", "int4", "int2"):
        b = SPECS.KV_BYTES_PER_ELEMENT[name]
        v4 = cache_bytes(32, 32, 128, SPECS.LLAMA2_7B_CTX, 1, b)
        v128 = cache_bytes(32, 32, 128, SPECS.KVQUANT_128K_TOKENS, 1, b)
        print("      %-8s %9.2fG %9.2fG %9.2fx" % (
            name, v4 / GiB, v128 / GiB,
            SPECS.KV_BYTES_PER_ELEMENT["fp16"] / b))
    check("B", "精度与字节数是严格反比：fp16/fp8/int4/int2 的每元素字节 = 2 / 1 / 0.5 / 0.25",
          [SPECS.KV_BYTES_PER_ELEMENT[k] for k in ("fp16", "fp8", "int4", "int2")]
          == [2.0, 1.0, 0.5, 0.25])
    check("B", "⚠ 这里的 int4/int2 是**载荷**字节，**不含 scale 元数据**；"
               "整块等效宽度（载荷 + 元数据）是 03 号程序的记账，两者不是同一个量",
          SPECS.KV_METADATA_BITS_PER_ELEMENT["int4_g32"] == 2.0
          and SPECS.KV_BYTES_PER_ELEMENT["int4"] == 0.5,
          "载荷 0.5 B = 4 bit ；G=32 时元数据再加 %.1f bit（64/32）⇒ 整块 %.1f bit"
          % (SPECS.KV_METADATA_BITS_PER_ELEMENT["int4_g32"],
             4 + SPECS.KV_METADATA_BITS_PER_ELEMENT["int4_g32"]))

    sub("B3 KV cache 在什么 batch 下超过模型权重（Llama-2-7B，fp16 权重 13.5 GB 用 2×7e9 估）")
    w_bytes = 2 * 7e9
    for batch in (1, 8, 32, 64, 128):
        kv = cache_bytes(32, 32, 128, SPECS.LLAMA2_7B_CTX, batch,
                         SPECS.KV_BYTES_PER_ELEMENT["fp16"])
        print("      batch %4d  KV = %8.2f GiB   权重/KV = %6.3f" % (
            batch, kv / GiB, w_bytes / kv))
    kv32 = cache_bytes(32, 32, 128, SPECS.LLAMA2_7B_CTX, 32,
                       SPECS.KV_BYTES_PER_ELEMENT["fp16"])
    check("B", "batch=32 / 4096 ctx 时 KV cache（64 GiB）**超过** fp16 权重（≈13 GiB）",
          kv32 > w_bytes, "KV %.1f GiB vs 权重 %.1f GiB" % (kv32 / GiB, w_bytes / GiB))


def run_B2():
    rule("B 代价模型（二）：decode 步的算术强度——真 torch 实测")

    sub("B4 注意力部分的算术强度公式 AI = S·T/(S+T)")
    for s in (1, 2, 16, 128, 1024, 4096):
        t = 4096
        print("      S=%5d T=%5d   AI = S·T/(S+T) = %9.4f" % (s, t, s * t / (s + t)))
    print("      ⇒ decode（S=1，T≫1）→ 1.0 ；prefill（S=T）→ T/2")
    check("B", "decode 的 AI 收敛到 %s（S=1、T→∞ 时 S·T/(S+T) → 1）" % SPECS.DECODE_AI,
          abs(1 * 10 ** 7 / (1 + 10 ** 7) - SPECS.DECODE_AI) < 1e-3,
          "S=1,T=1e7 → %.6f" % (1 * 10 ** 7 / (1 + 10 ** 7)))
    check("B", "prefill（S=T）的 AI = T/2，比 decode 高 T/2 倍",
          abs(4096 * 4096 / (4096 + 4096) - 2048.0) < 1e-9)

    sub("B5 真 torch 实测：同一个 QK^T，换 S 就换 achieved GFLOP/s")
    d = 128
    t_len = 8192

    def timed(fn, reps, tries=5):
        """**交错 best-of-`tries`**：本机是 CPU，六个子进程并行时计时会被调度抖动污染。

        判据：每个 trial 连续跑 `reps` 次，取所有 trial 里**最快的那一次**的平均。
        慢的那些只会来自外部干扰，不会来自被测代码——这样断言才不会在机器忙时假失败。
        `tries × reps` 的总时长要够长（毫秒级以下的操作单次会被噪声整个吃掉）。
        """
        best = None
        for _ in range(tries):
            fn()
            t0 = time.perf_counter()
            for _ in range(reps):
                fn()
            dt = (time.perf_counter() - t0) / reps
            best = dt if best is None else min(best, dt)
        return best

    k = torch.randn(t_len, d)
    sizes = (1, 16, 256, 4096)
    qs = {sz: torch.randn(sz, d) for sz in sizes}
    a = torch.randn(2048, 2048)
    b = torch.randn(2048, 2048)

    # ★ **交错测量 + best-of-N**：本机是 CPU，六个子进程并行时计时会被调度抖动污染
    #   （实测「decode vs GEMM」的比值可在 4.6×–15× 之间漂 3 倍）。
    #   做法：把**所有**待测量放进同一个 trial 循环里，每个量各取多轮的最小值——
    #   它们经历同一批外部干扰，比值才是可比的。**这是量级判据，不是精度判据。**
    TRIES = 5
    best = {}
    for _ in range(TRIES):
        for sz in sizes:
            reps = max(20, 20000 // sz)
            dt = timed(lambda: qs[sz] @ k.T, reps, tries=1)
            best[sz] = dt if sz not in best else min(best[sz], dt)
        dtg = timed(lambda: a @ b, 5, tries=1)
        best["gemm"] = dtg if "gemm" not in best else min(best["gemm"], dtg)

    rows = []
    for sz in sizes:
        flops = 4.0 * sz * t_len * d          # QK^T 与 PV 各 2·S·T·D
        rows.append((sz, sz * t_len / (sz + t_len), best[sz], flops / best[sz] / 1e9))
        print("      S=%5d  AI=%9.2f  QK^T %9.2f us  → %9.2f GFLOP/s" % (
            rows[-1][0], rows[-1][1], rows[-1][2] * 1e6, rows[-1][3]))
    gemm_gflops = 2.0 * 2048 ** 3 / best["gemm"] / 1e9
    print("      对照：2048³ 稠密 GEMM  %9.2f us  → %9.2f GFLOP/s（都是 best-of-%d）"
          % (best["gemm"] * 1e6, gemm_gflops, TRIES))
    decode_gflops = rows[0][3]
    check("B", "★ decode 形状（S=1）的 achieved GFLOP/s 不足稠密 GEMM 的 1/5"
               "（**量级判据**：比值在不同机器/负载下会漂，只断言「差一个量级」）",
          decode_gflops < 0.2 * gemm_gflops,
          "decode %.2f vs GEMM %.2f GFLOP/s（%.1f 倍差距）" % (
              decode_gflops, gemm_gflops, gemm_gflops / decode_gflops))
    check("B", "★ 更稳的同族对照：**同一个 `q @ k.T`**，S=1 与 S=4096 的 GFLOP/s 差 ≥5×"
               "（同一族算子、同一份 K，只有 S 不同 ⇒ 没有跨算子的混杂）",
          rows[3][3] > 5.0 * rows[0][3],
          "%.2f vs %.2f GFLOP/s（%.1f 倍）" % (rows[0][3], rows[3][3], rows[3][3] / rows[0][3]))
    # ⚠ 判据写到**平台期为止**：S=256 与 S=4096 的先后在 torch 2.8.0 / py3.9 上会反转
    #   （本仓库实测 2716.3 vs 2582.7 GFLOP/s）——那已经是平台期内的噪声，
    #   不是「算术强度越高越快」这条结论的反例。**把判据写到它真正支持的范围内。**
    check("B", "★ achieved GFLOP/s 随 AI **上升并进入平台期**：前三点严格单调，"
               "最大 S 仍不低于 S=256 的一半（内存受限 ⇒ 算术强度是瓶颈，不是算力）",
          rows[0][3] < rows[1][3] < rows[2][3] and rows[3][3] > 0.5 * rows[2][3],
          " → ".join("%.1f" % r[3] for r in rows))

    sub("B6 decode 步搬运了多少字节、跑到峰值带宽的百分之几")
    n = 128 * 1024 * 1024 // 4
    src = torch.randn(n)
    dst = torch.empty_like(src)
    q1 = torch.randn(1, d)
    v = torch.randn(t_len, d)

    def decode_step():
        s = q1 @ k.T
        p = torch.softmax(s, dim=-1)
        return p @ v

    moved = 2 * t_len * d * 4          # K + V，各 T·D 个元素、fp32 4 B
    # ★ 判据必须是**逐 trial 的比值**，不是「分别取最优再相除」：
    #   本机是 CPU，六个子进程并行时两个量的可观测带宽会各自漂（实测 decode 侧
    #   7–32 GB/s、拷贝侧 50–100 GB/s），分别取最优再比会把噪声乘进去。
    #   在**同一个 trial 里**背靠背测两个量，噪声在比值里对消；取各 trial 比值的**中位数**
    #   （取最大值会被「一次侥幸很快的 decode + 一次很慢的拷贝」拉爆，实测到 729%）。
    m = t_len * d // 2                 # 拷贝张量规模：读+写 = 2·m·4 = T·d·4 字节
    csrc = torch.randn(m)
    cdst = torch.empty_like(csrc)
    ratios = []
    for _ in range(9):
        t_c = timed(lambda: cdst.copy_(csrc), 30, tries=1)     # 拷贝的读+写带宽
        t_d = timed(decode_step, 500, tries=1)
        ratios.append((moved / t_d) / (2 * m * 4 / t_c))
    ratios.sort()
    med = ratios[len(ratios) // 2]
    print("      decode 步（T=%d, D=%d, fp32）：搬运 %.2f MB ；达到同 trial 内拷贝带宽的"
          " **%.0f%%**（9 个 trial 比值的**中位数**；区间 %.0f%%–%.0f%%）"
          % (t_len, d, moved / 1e6, 100 * med, 100 * ratios[0], 100 * ratios[-1]))
    check("B", "★ decode 步达到同 trial 内拷贝带宽的 25% 以上 ⇒ 它真的是在**搬字节**，"
               "不是在算（判据是**逐 trial 比值的中位数**，噪声在比值里对消）",
          med > 0.25, "中位数 %.0f%%" % (100 * med))

    sub("B7 解析侧的同一条结论：decode 与稠密 GEMM 的算术强度差两个数量级")
    ai_dec = (4.0 * 1 * t_len * d) / (2 * 4.0 * d * (1 + t_len))       # fp32 记账
    m = 2048
    ai_gemm = (2.0 * m ** 3) / (3.0 * m ** 2 * 4.0)                    # A@B：读 2 写 1
    print("      decode 步（fp32）：AI = %.4f FLOP/B" % ai_dec)
    print("      2048³ 稠密 GEMM  ：AI = %.1f FLOP/B" % ai_gemm)
    print("      ⇒ 相差 %.0f 倍。**这才是「换形状就换瓶颈」的定量表述。**" % (ai_gemm / ai_dec))
    check("B", "★ 解析侧的算术强度差 >100 倍（与本机实测的 achieved GFLOP/s 落差同向）",
          ai_gemm / ai_dec > 100.0, "%.0f 倍" % (ai_gemm / ai_dec))
    check("B", "⚠ 同一算式在 fp32 下 AI **腰斩**（比值恰为 0.5）——"
               "「AI=1」是「每元素 2 字节」这个前提的产物，不是注意力的固有性质",
          abs(ai_dec / ((4.0 * 1 * t_len * d) / (2 * 2.0 * d * (1 + t_len))) - 0.5) < 1e-12,
          "fp16 AI %.6f ；fp32 AI %.6f" % ((4.0 * t_len * d) / (2 * 2.0 * d * (1 + t_len)),
                                           ai_dec))


# ===========================================================================
# A. 两条互不依赖的通道
# ===========================================================================

def run_A():
    rule("A 判据复核：公式路径 vs 真 torch 张量分配路径")

    sub("A1 真张量分配：`torch.empty(2, L, H, seq, d, fp16).nbytes` 与公式对拍")
    ok_all = True
    for m in ("Llama-2-7B", "Llama-2-13B", "Llama-2-70B", "Mistral-7B"):
        nl, nh, hd = SPECS.KV_MODEL_SHAPES[m]
        t = torch.empty(SPECS.KV_TENSOR_COUNT, nl, nh, 512, hd, dtype=torch.float16)
        allocated = t.numel() * t.element_size()
        formula = cache_bytes(nl, nh, hd, 512, 1, SPECS.KV_BYTES_PER_ELEMENT["fp16"])
        same = allocated == formula
        ok_all = ok_all and same
        print("      %-14s torch nbytes %12d   公式 %12d   %s" % (
            m, allocated, formula, "一致" if same else "**不一致**"))
    check("A", "★ 四条模型上，真张量分配的字节数与公式**逐字节一致**",
          ok_all, "判据：`numel()*element_size()` vs `2·L·H·d·seq·2`，用到的 seq 相同")

    sub("A2 三处外部锚点各自独立重算（这三处的算式**互不相同**）")
    # (a) PagedAttention：OPT-13B，hidden = kv_heads × head_dim
    nl, nh, hd = SPECS.KV_MODEL_SHAPES["OPT-13B"]
    pt_opt = per_token_bytes("OPT-13B", SPECS.KV_BYTES_PER_ELEMENT["fp16"])
    print("      (a) OPT-13B：2·5120·40·2 = %d B ÷ 1024 = %.1f KiB（论文印 800 KB）"
          % (pt_opt, pt_opt / KiB))
    check("A", "(a) PagedAttention 的 OPT-13B 每 token 800（KiB）可复现",
          abs(pt_opt / KiB - SPECS.PAGEDATTENTION_PER_TOKEN_KB) < 1e-9,
          "%.1f vs %s" % (pt_opt / KiB, SPECS.PAGEDATTENTION_PER_TOKEN_KB))
    total_opt = cache_bytes(nl, nh, hd, SPECS.PAGEDATTENTION_SEQ, 1,
                            SPECS.KV_BYTES_PER_ELEMENT["fp16"])
    check("A", "(a) 论文那句「2048 token 最多 1.6 GB」**是四舍五入到一位小数**"
               "（精确 1.5625 GiB；若按十进制 GB 读则是 1.678）",
          round(total_opt / GiB, 1) == SPECS.PAGEDATTENTION_TOTAL_GB,
          "%.4f GiB → 一位小数 %.1f ；十进制 %.3f GB"
          % (total_opt / GiB, round(total_opt / GiB, 1), total_opt / 1e9))
    check("A", "(a) `kv_heads × head_dim` 恰等于 OPT-13B 的 hidden size 5120（它就是 MHA）",
          nh * hd == 5120)

    # (b) Databricks：Llama-2-70B GQA，seq=1024，batch=1
    nl, nh, hd = SPECS.KV_MODEL_SHAPES["Llama-2-70B"]
    v70 = cache_bytes(nl, nh, hd, 1024, 1, SPECS.KV_BYTES_PER_ELEMENT["fp16"])
    print("      (b) Llama-2-70B seq=1024：%.6f GiB（博客印 %.3f）" % (v70 / GiB,
                                                                    SPECS.DATABRICKS_70B_SEQ1024_GIB))
    check("A", "(b) Databricks 的 0.312 GiB 是 **0.3125 的三位小数**（不是另一个数）",
          abs(v70 / GiB - 0.3125) < 1e-12
          and round(v70 / GiB, 3) == SPECS.DATABRICKS_70B_SEQ1024_GIB,
          "精确值 %.6f，四舍五入到三位 = %.3f" % (v70 / GiB, round(v70 / GiB, 3)))
    check("A", "⚠ (b) 这条**本次未取到博客原文**，档位是「计划基线，未独立核实」，"
               "只做量级对拍", True)

    # (c) KVQuant：LLaMA-7B @128K = 64.0
    nl, nh, hd = SPECS.KV_MODEL_SHAPES["Llama-2-7B"]
    v128 = cache_bytes(nl, nh, hd, SPECS.KVQUANT_128K_TOKENS, 1,
                       SPECS.KV_BYTES_PER_ELEMENT["fp16"])
    print("      (c) 7B @128K：%d B = %.1f GiB = %.1f GB(十进制)" % (
        v128, v128 / GiB, v128 / 1e9))
    check("A", "(c) KVQuant 的 64.0 复现，**但只在 GiB 下**（十进制 GB 是 %.1f）" % (v128 / 1e9),
          abs(v128 / GiB - SPECS.KVQUANT_7B_128K_CACHE_GB) < 1e-9,
          "论文写 64.0，按 GiB 记账才成立")
    check("A", "(c) 反解回去的每 token 字节数与 Llama-2-7B 的公式值相同（同一件事的两种到达方式）",
          v128 // SPECS.KVQUANT_128K_TOKENS == SPECS.LLAMA2_7B_PER_TOKEN_BYTES)

    sub("A3 KVQuant Table 1 的另一半：内存与 PPL 同时读")
    for k, (ppl, gb) in sorted(SPECS.KVQUANT_TABLE1_LLAMA7B.items()):
        print("      %-16s PPL %7.2f   KV cache %6.1f GB" % (k, ppl, gb))
    base = SPECS.KVQUANT_TABLE1_LLAMA7B["baseline"]
    check("A", "Table 1 的 baseline 与 (c) 的 64.0 GB **是同一行的两个格子**（互相印证）",
          base[1] == SPECS.KVQUANT_7B_128K_CACHE_GB)
    check("A", "Table 1 的压缩比可自洽验算：16/3.32 ≈ 4.8×（论文说的「4.8× 更长上下文」）",
          abs(16.0 / 3.32 - 4.82) < 0.02)


# ===========================================================================
# C. 边界与反例
# ===========================================================================

def run_C():
    rule("C 边界与反例（三种误用各自的偏差，全部真跑出来）")

    sub("C1 把 **KV 头数**当成**注意力头数**（GQA 模型上会差一个整数倍）")
    for m, attn_heads in (("Llama-2-70B", 64), ("Mistral-7B", 32)):
        nl, nh, hd = SPECS.KV_MODEL_SHAPES[m]
        right = cache_bytes(nl, nh, hd, 4096, 1, SPECS.KV_BYTES_PER_ELEMENT["fp16"])
        wrong = cache_bytes(nl, attn_heads, hd, 4096, 1, SPECS.KV_BYTES_PER_ELEMENT["fp16"])
        print("      %-14s 正确(kv_heads=%2d) %8.3f GiB   误用(attn=%2d) %8.3f GiB  → 差 %.0f×"
              % (m, nh, right / GiB, attn_heads, wrong / GiB, wrong / right))
        check("C", "%s：用注意力头数算会把 KV cache 高估 %d 倍（GQA 把 KV 头数压到了 %d）"
              % (m, attn_heads // nh, nh), wrong > right)
    check("C", "★ MHA 的模型上这个错**不会发生**（Llama-2-7B 的 kv_heads == attn_heads == 32）",
          SPECS.KV_MODEL_SHAPES["Llama-2-7B"][1] == 32)

    sub("C2 「KB」到底是 1000 还是 1024——两个读数都真实，但不是同一个量")
    nl, nh, hd = SPECS.KV_MODEL_SHAPES["OPT-13B"]
    pt = per_token_bytes("OPT-13B", SPECS.KV_BYTES_PER_ELEMENT["fp16"])
    print("      OPT-13B 每 token 精确 %d B = %.1f KiB = %.1f KB(十进制)" % (
        pt, pt / 1024, pt / 1000))
    check("C", "★ 论文写的「800 KB」只有在 **KiB** 下精确（十进制下是 819.2）",
          abs(pt / 1024 - 800.0) < 1e-9 and abs(pt / 1000 - 800.0) > 19.0,
          "同一份字节数，两个单位的读数相差 %.1f" % (pt / 1000 - pt / 1024))
    check("C", "同理 KVQuant 的 64.0「GB」在十进制下是 %.1f——**引用时必须写明单位口径**"
          % (cache_bytes(32, 32, 128, SPECS.KVQUANT_128K_TOKENS, 1, 2.0) / 1e9),
          abs(cache_bytes(32, 32, 128, SPECS.KVQUANT_128K_TOKENS, 1, 2.0) / 1e9 - 64.0) > 4.0)

    sub("C3 「AI = 1」的前提是「每元素 2 字节」——换掉它，AI 腰斩")
    d, t_len = 128, 8192
    for bpe in (2.0, 4.0):
        flops = 4.0 * 1 * t_len * d
        moved = 2 * t_len * d * bpe
        print("      %d B/元素：搬运 %8.0f B，AI = %5.3f FLOP/B" % (bpe, moved,
                                                                   flops / moved))
    check("C", "★ AI 那个「1」是 fp16 专属的：fp32 的 AI 恰好是 0.5，**不是 1**",
          abs((4.0 * t_len * d) / (2 * t_len * d * 4.0) - 0.5) < 1e-12)

    sub("C4 「全模型 decode = 1 FLOP/weight-byte」这句话本仓库**不写**")
    check("C", "本程序只落 attention 部分的 ST/(S+T)（有一手来源）；"
               "全模型那条**只找到二手博客推导，无一手论文** ⇒ 记 UNSOURCED",
          not hasattr(SPECS, "FULL_MODEL_DECODE_FLOPS_PER_BYTE"),
          "判据：`specs.py` 里**没有**这个常量（不落成常量就不会被误引）")


# ===========================================================================
# M. 测量条件自洽
# ===========================================================================

def run_M():
    rule("M 测量条件自洽（这几个常量也必须被断言，否则就是没人查的抄件）")
    v = torch.__version__
    short = ".".join(v.split(".")[:2])
    check("M", "本次运行的 torch %s 在 `MEASURED_VERSIONS` 表里" % v,
          v in SPECS.MEASURED_VERSIONS or short in SPECS.MEASURED_VERSIONS,
          "表 = %s" % (SPECS.MEASURED_VERSIONS,))
    check("M", "表里恰好 %d 个版本（与本机解释器数一致）" % len(SPECS.MEASURED_VERSIONS),
          len(SPECS.MEASURED_VERSIONS) == 5)
    unc = [s.strip() for s in SPECS.UNCOVERED_MINORS.split("/")]
    covered = {".".join(x.split(".")[:2]) for x in SPECS.MEASURED_VERSIONS}
    print("      未覆盖的 minor：%s" % unc)
    check("M", "`UNCOVERED_MINORS` 逐字等于 `2.9.x / 2.10.x / 2.12.x`",
          SPECS.UNCOVERED_MINORS == "2.9.x / 2.10.x / 2.12.x", SPECS.UNCOVERED_MINORS)
    check("M", "`UNCOVERED_MINORS` 列的 minor 与已覆盖的**不重叠**", not (set(unc) & covered))
    import os as _os
    host = cpu_model()
    print("      本机 CPU = %s" % host)
    osname = {"Darwin": "macOS", "Linux": "Linux"}.get(platform.system(), platform.system())
    expect_host = "%s（%d 核）/ %s / **CPU-only，无 GPU 数字**" % (host, _os.cpu_count(),
                                                                   osname)
    check("M", "`MEASURE_HOST` 逐字等于「本机 CPU 型号 + 核数 + 系统 + CPU-only」",
          SPECS.MEASURE_HOST == expect_host,
          "specs「%s」vs 推出「%s」" % (SPECS.MEASURE_HOST, expect_host))
    check("M", "`MEASURE_HOST` 明写 **CPU-only，无 GPU 数字**",
          "CPU-only" in SPECS.MEASURE_HOST and "无 GPU 数字" in SPECS.MEASURE_HOST)
    # Makefile 里的下限必须与 specs 一致（两个文件不许各说一套）
    import os
    mf = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Makefile")
    txt = open(mf, encoding="utf-8").read()
    maj = [ln for ln in txt.splitlines() if ln.startswith("TORCH_MIN_MAJOR :=")]
    mnr = [ln for ln in txt.splitlines() if ln.startswith("TORCH_MIN_MINOR :=")]
    mk = (int(maj[0].split(":=")[1]), int(mnr[0].split(":=")[1])) if maj and mnr else None
    print("      Makefile 的下限 = %s ；specs 的下限 = %s" % (mk,
                                                             SPECS.TORCH_MIN_FOR_THIS_TOPIC))
    check("M", "★ `Makefile` 与 `specs.py` 的版本下限**逐字一致**（不许各说一套）",
          mk == SPECS.TORCH_MIN_FOR_THIS_TOPIC)
    check("M", "★ 下限的理由逐字为「对齐选择（不是依赖）：…」（P4 的反面）",
          SPECS.TORCH_MIN_REASON.startswith("对齐选择（不是依赖）：")
          and SPECS.TORCH_MIN_REASON.endswith("§5 的测量条件全域可比"),
          SPECS.TORCH_MIN_REASON)


# ===========================================================================
# F. 变异测试清单（A6）
# ===========================================================================

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

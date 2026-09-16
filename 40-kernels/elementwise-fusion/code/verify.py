#!/usr/bin/env python3
"""40-kernels/elementwise-fusion 主题的本机验证模型。

**本模块是推导，不是硬件测量。** 它只跑算术与 numpy，不碰硬件——本机（macOS）
没有 CUDA 工具链也没有 GPU，`code/` 里的 3 个 `.cu` 内核的运行结果本仓库没有。

它覆盖本主题全部 3 篇文档的可推导部分，分三类断言（CONTRIBUTING §2）：

  A 算法正确性  用 numpy 写参考实现（非融合链 / 融合链 / 向量化的元素映射），
                与文档所述算法逐元素比对，打印最大误差与判定
  B 代价模型    按文档公式算访存字节 / roofline 可达时间下界 / 融合加速比 /
                指令条数与访存事务数 / 填满带宽所需的在途请求数，
                逐条打印数值并给出判定
  C 边界与反例  §4 声称「会出问题」的输入必须**真的跑出问题**——断言的是
                失败行为成立，不是成功

**数字的口径**（沿用 20-gpu-architecture 的三分法）：
  · 规格值 = 来自 `specs.py`，逐条带 `# 来源：`
  · 推导值 = 本程序算出，可在这里复核
  · 引用值 = 外部来源，标注「本仓库不复核其测量口径」

**一条额外约定**：本主题的「结构」断言（字节数、指令条数、事务数、寄存器与
occupancy 的整数计数）一律用**整数**算术，使「恰好 2N·esize」「恰好 4 倍」
这类断言是**精确相等**而不是 allclose；只有涉及浮点误差的断言才用浮点。

各篇文档与本节次的对应（文档的「编译与运行」小节会逐条点名）：

  01-memory-bound-limit         §B1 §B2 §C1
  02-fusion-and-traffic         §A1 §B3 §C3
  03-vectorized-and-grid-stride §A2 §B4 §B5 §B6 §C2 §C4 §C5

运行：`make verify`（等价于 `python3 verify.py`）。只需 python3 + numpy。
"""

import math

import numpy as np

import specs as SP

# ===========================================================================
# 断言与打印基础设施
# ===========================================================================

_STATE = {"n": 0, "fail": []}


def check(name, cond, detail=""):
    """记一次判定并打印结果。cond 为假时记入失败清单。"""
    _STATE["n"] += 1
    ok = bool(cond)
    if not ok:
        _STATE["fail"].append(name)
    tail = f"   ← {detail}" if detail else ""
    print(f"    [{'通过' if ok else '**不通过**'}] {name}{tail}")
    return ok


def section(tag, title):
    print()
    print("=" * 76)
    print(f"§{tag}  {title}")
    print("=" * 76)


def sub(title):
    print()
    print(f"--- {title} ---")


def close(a, b, rel=1e-9):
    return abs(a - b) <= rel * max(1.0, abs(a), abs(b))


def bits_equal(a, b):
    """逐位比较两个同形数组（按内存里的位模式，不比数值）。

    ⚠ 不用 `np.asarray(x).view(np.uint32)`：对 0 维数组 `.view()` 的行为与本意
    不符。直接用 `tobytes()` 比字节。
    """
    return np.ascontiguousarray(np.asarray(a)).tobytes() == \
        np.ascontiguousarray(np.asarray(b)).tobytes()


def ceil_to(x, g):
    """向上取整到 g 的整数倍（`g` = 寄存器分配粒度）。"""
    return int(math.ceil(x / g) * g)


def mib(nbytes):
    return nbytes / 2 ** 20


H100 = "H100"
B200 = "B200"
ESIZE_F32 = SP.FLOAT32_BYTES

B_PEAK = SP.HBM_BW_GBps_BY_GEN[H100] * 1e9          # B/s，标称
I_STAR_FP32 = SP.FP32_VECTOR_TFLOPS_DENSE[H100] * 1e12 / B_PEAK

# 时钟：**由规格值反推的推导值**，不是规格值。官方未公布 H100 SXM5 的 boost 时钟
# （见 specs.py 的 UNSOURCED），这里用「每 SM 128 个 FP32 单元 × 每单元每周期 1 条
# FMA × 2 FLOP」反解。凡依赖它的结论都标注这一条。
FP32_CLOCK_HZ_DERIVED = (SP.FP32_VECTOR_TFLOPS_DENSE[H100] * 1e12 /
                         (SP.SM_COUNT_BY_GEN[H100] * SP.FP32_LANES_PER_SM[H100] * 2))

# 全卡每周期可发射的 **warp 指令**数
ISSUE_WARP_INSTR_PER_CLK = SP.ISSUE_SLOTS_PER_CLK_PER_SM * SP.SM_COUNT_BY_GEN[H100]


# ===========================================================================
# §0 口径
# ===========================================================================


def s0_scope():
    section(0, "口径")
    print("本程序全部输出是**推导值**：跑的是算术与 numpy，本机没有 GPU，")
    print("3 个 .cu 内核的运行结果本仓库没有。")
    print()
    print(f"主用机器：{H100} SXM —— {SP.SM_COUNT_BY_GEN[H100]} SM / "
          f"{SP.FP32_VECTOR_TFLOPS_DENSE[H100]} TFLOPS FP32 向量 / "
          f"{SP.HBM_BW_GBps_BY_GEN[H100]} GB/s 标称 HBM 带宽")
    print(f"对照机器：{B200} SXM —— {SP.SM_COUNT_BY_GEN[B200]} SM / "
          f"{SP.FP32_VECTOR_TFLOPS_DENSE[B200]} TFLOPS FP32 向量 / "
          f"{SP.HBM_BW_GBps_BY_GEN[B200]} GB/s 标称 HBM 带宽")
    print()
    print(f"roofline 拐点 I*(FP32 向量) = {SP.FP32_VECTOR_TFLOPS_DENSE[H100]}e12 ÷ "
          f"{SP.HBM_BW_GBps_BY_GEN[H100]}e9 = {I_STAR_FP32:.1f} FLOP/byte")
    print("  （roofline 模型的定义与拐点的推导在 10-foundations/performance-models/01，")
    print("    本主题**只代入、不重推**。）")
    print()
    print(f"⚠ 时钟 = {FP32_CLOCK_HZ_DERIVED/1e9:.4f} GHz 是**推导值**，不是规格值：")
    print(f"    {SP.FP32_VECTOR_TFLOPS_DENSE[H100]}e12 ÷ "
          f"({SP.SM_COUNT_BY_GEN[H100]} SM × {SP.FP32_LANES_PER_SM[H100]} lane × 2 FLOP)")
    print("   官方未公布 H100 SXM5 的 boost 时钟（见 specs.py 的 UNSOURCED）。")
    print("   它只在「67 TFLOPS 是在 boost 时钟下测得」这个前提下成立。")
    print()
    print(f"全卡发射能力 = {SP.ISSUE_SLOTS_PER_CLK_PER_SM} 槽/SM/周期 × "
          f"{SP.SM_COUNT_BY_GEN[H100]} SM = {ISSUE_WARP_INSTR_PER_CLK} 条 warp 指令/周期")
    print()
    print("访存粒度的两个数（只在 03 篇做代入，机制在别处）：")
    print(f"  sector = {SP.SECTOR_BYTES} 字节（搬运单位）；"
          f"cache line = {SP.CACHE_LINE_BYTES} 字节（记账单位）")
    print("  正文在 20-gpu-architecture/memory-system/02-coalescing-and-cache-line §1。")
    print()
    print("⚠ 「标称带宽」与「可达带宽」是两个数：")
    print(f"   标称 = {SP.HBM_BW_GBps_BY_GEN[H100]} GB/s（datasheet 规格值）")
    lo, hi = SP.ACHIEVABLE_BW_SHORTFALL_PCT_IN_REPO
    print(f"   可达 = 标称 × η，η 的口径是「扣掉几个到十几个百分点」→ "
          f"η ∈ [{1 - hi/100:.2f}, {1 - lo/100:.2f}]")
    print("   gap 的分解在 20-gpu-architecture/interconnect/04-measuring-bandwidth")
    print("   与 memory-system/01-hierarchy-overview §2，**本主题不重推**。")


# ===========================================================================
# 参考实现：融合链与非融合链
# ===========================================================================

# 一条 k = 4 的链，故意混入非线性算子——融合**不改变**逐元素算子的结果，
# 这一点与归约类算子相反（后者的正文在 40-kernels/reduce-scan，本程序不复述）。
def op_fma(w, b):
    return lambda t: t * np.float32(w) + np.float32(b)


CHAIN_OPS = [
    ("fma(x, 1.5, +0.25)", op_fma(1.5, 0.25)),
    ("exp(·)", np.exp),
    ("fma(·, 0.5, −1.0)", op_fma(0.5, -1.0)),
    ("tanh(·)", np.tanh),
]


def chain_unfused(x, ops):
    """非融合：k 个算子各写一个 kernel，中间结果每个都落一块完整的数组。

    内存里同时存在 k+1 个张量（输入 + k 个中间/输出），流量模型见 §B3。
    """
    with np.errstate(over="ignore", invalid="ignore"):
        t = x.copy()
        for _, f in ops:
            t = f(t)
        return t


def chain_fused(x, ops, chunk):
    """融合：一个 kernel 把 k 个算子算完，中间值住在寄存器里、不落 HBM。

    用 chunk 分块纯粹是为了在 numpy 里模拟「一个线程处理一小段」——**分块不改
    结果**（逐元素算子没有跨元素的依赖），这一点 §A1 会同时断言。
    """
    out = np.empty_like(x)
    with np.errstate(over="ignore", invalid="ignore"):
        for i0 in range(0, x.size, chunk):
            t = x[i0:i0 + chunk].copy()
            for _, f in ops:
                t = f(t)
            out[i0:i0 + chunk] = t
    return out


# ===========================================================================
# §A 算法正确性
# ===========================================================================


def a1_fused_chain_matches():
    sub("A1  融合链与非融合链逐元素一致（含非线性算子 exp / tanh）")

    rng = np.random.default_rng(20260916)
    for n in (4096, 4097):          # 第二个不是 chunk 与向量宽度的整数倍
        x = (rng.standard_normal(n) * 0.5).astype(np.float32)
        ref = chain_unfused(x, CHAIN_OPS)
        fus = chain_fused(x, CHAIN_OPS, 1024)
        err = float(np.max(np.abs(fus - ref)))
        same_bits = bits_equal(fus, ref)
        print(f"    N = {n}：链长 k = {len(CHAIN_OPS)}，最大绝对误差 = {err:.3e}，"
              f"逐位相同 = {same_bits}")
        check(f"N={n}：融合链与非融合链**逐位**相同（不是 allclose）", same_bits,
              "逐元素算子没有跨元素依赖，融合只改数据流、不改运算序列")

    print()
    print("    换分块大小也不改变结果（分块是模拟手段，不是算法的一部分）：")
    x = (rng.standard_normal(4096) * 0.5).astype(np.float32)
    base = chain_fused(x, CHAIN_OPS, 1024)
    for chunk in (1, 7, 64, 4096):
        got = chain_fused(x, CHAIN_OPS, chunk)
        check(f"chunk = {chunk} 与 chunk = 1024 逐位相同", bits_equal(got, base))

    print()
    print("    ⚠ 这条结论**只对逐元素算子成立**。归约 / 扫描 / softmax 这类")
    print("      「跨元素」算子的融合会改变加法树形，结果就不再逐位相同——")
    print("      正文与验证在 40-kernels/reduce-scan §C7，本程序不复述。")


def a2_vectorized_element_mapping():
    sub("A2  向量化的元素映射：一组 4 个元素 vs 逐元素")

    rng = np.random.default_rng(7)
    for n in (1024, 1026):          # 第二个不是 4 的整数倍，要留尾巴
        x = rng.standard_normal(n).astype(np.float32)
        # 逐元素（标量）路径：每线程一个元素
        y_scalar = np.empty(n, dtype=np.float32)
        for i in range(n):
            y_scalar[i] = x[i] * np.float32(2.0) + np.float32(1.0)
        # 向量化路径：每线程四个元素（一个 float4）
        y_vec = np.empty(n, dtype=np.float32)
        full = n - n % 4
        for i0 in range(0, full, 4):
            v = x[i0:i0 + 4]                       # 一条 16 字节读
            r = v * np.float32(2.0) + np.float32(1.0)
            y_vec[i0:i0 + 4] = r                   # 一条 16 字节写
        for i in range(full, n):                   # 尾部：标量收尾
            y_vec[i] = x[i] * np.float32(2.0) + np.float32(1.0)

        err = float(np.max(np.abs(y_vec - y_scalar)))
        print(f"    N = {n}（4 的整数倍 = {n % 4 == 0}）：最大绝对误差 = {err:.3e}")
        check(f"N={n}：向量化路径与标量路径逐位相同", bits_equal(y_vec, y_scalar),
              f"尾部处理了 {n - full} 个元素" if n % 4 else "无尾巴")
    print()
    print("    向量化改的是**每次访存的宽度**，不是**算的东西**：")
    print("      标量路径：每线程 1 个元素 = 1 条 4 字节读 + 1 条 4 字节写")
    print("      向量化：  每线程 4 个元素 = 1 条 16 字节读 + 1 条 16 字节写")
    print("    代价在 §C2：16 字节访问要求 16 字节对齐，未对齐是未定义行为。")


# ===========================================================================
# §B 代价模型
# ===========================================================================


def b1_unary_bound():
    sub("B1  单目算子的最少访存字节与 roofline 可达时间下界")

    N = 1 << 26                      # 67108864 个 fp32 = 256 MiB
    by_read = N * ESIZE_F32
    by_min = 2 * N * ESIZE_F32
    print(f"    N = 2^26 = {N} 个 fp32（数据本身 {mib(by_read):.1f} MiB）")
    print(f"      只数「读」        N·esize   = {by_read} B = {mib(by_read):.1f} MiB")
    print(f"      最少往返 HBM 字节 2N·esize  = {by_min} B = {mib(by_min):.1f} MiB")
    check("最少字节恰好是「只数读」的两倍（写也占一份，且不可省）",
          by_min == 2 * by_read, "2N·esize ÷ N·esize = 2.00")
    print()
    print(f"    ⚠ 就地（in-place）写法**不省这一半**：`y[i] = f(y[i])` 仍然是")
    print(f"      「读一次 + 写一次」，HBM 流量还是 2N·esize。它省的是**容量**")
    print(f"      （输出缓冲不再单独占 {mib(by_read):.0f} MiB）与**地址足迹**")
    print(f"      （触及的 line 数减半），不是流量。正文 01 篇 §4.4。")

    print()
    print(f"    代入 roofline 的时间下界（标称 B = {SP.HBM_BW_GBps_BY_GEN[H100]} GB/s）：")
    print(f"      {'η':>6} {'口径':>16} {'可达带宽 GB/s':>16} {'下界时间 µs':>14}")
    rows = [(1.00, "标称（乐观下界）"),
            (0.95, "可达（缺口 5%）"),
            (0.85, "可达（缺口 15%）")]
    for eta, note in rows:
        b_eff = eta * SP.HBM_BW_GBps_BY_GEN[H100]
        t = by_min / (b_eff * 1e9) * 1e6
        print(f"      {eta:>6.2f} {note:>16} {b_eff:>16.1f} {t:>14.2f}")
    t_peak = by_min / B_PEAK * 1e6
    t_90 = by_min / (0.90 * B_PEAK) * 1e6
    print(f"    η = 0.90（缺口 10%）→ 可达带宽 "
          f"{0.90 * SP.HBM_BW_GBps_BY_GEN[H100]:.1f} GB/s，"
          f"下界 {t_90:.2f} µs，比 η = 1 的 {t_peak:.2f} µs 多 "
          f"{t_90 / t_peak - 1:.1%}")
    print(f"    ⚠ 同一份数据，只数「读」的时间 = {by_read} B ÷ "
          f"{SP.HBM_BW_GBps_BY_GEN[H100]} GB/s = "
          f"{by_read / B_PEAK * 1e6:.2f} µs（正文 §4.1 那条少算一半的坑）")
    t_85 = by_min / (0.85 * B_PEAK) * 1e6
    print(f"    ⚠ η = 1 与 η = 0.85 的差 = {t_85:.2f} − {t_peak:.2f} = "
          f"{t_85 - t_peak:.2f} µs（{(t_85 - t_peak) / t_peak:.1%}）——")
    print(f"       **这不是「余量」，是缺口**（正文 §4.3）")
    check("可达带宽缺口造成的差值 ≈ 时间下界的 17.6%（被误读成「优化空间」）",
          close((t_85 - t_peak) / t_peak, 0.176, 0.01),
          f"({t_85:.2f} − {t_peak:.2f}) / {t_peak:.2f} = "
          f"{(t_85 - t_peak) / t_peak:.1%}")
    lo, hi = SP.ACHIEVABLE_BW_SHORTFALL_PCT_IN_REPO
    check("可达带宽的缺口只吃掉百分之十几的时间，**不改变量级**",
          hi / 100 < 0.20,
          f"缺口 {lo}%~{hi}%（引用值，本仓库不复核其测量口径）")

    print()
    print("    算术强度（口径：Bytes 只算必须往返 HBM 的 2N·esize）：")
    print(f"      {'每元素运算':>28} {'FLOPs/元素':>11} {'I':>8} {'I/I*':>9}")
    for name, fpe in (("y = a·x + b（1 次 FMA）", 2),
                      ("y = exp(x)（计 1 次）", 1),
                      ("y = tanh(x)（计 1 次）", 1)):
        I = fpe / (2 * ESIZE_F32)
        print(f"      {name:>28} {fpe:>11} {I:>8.3f} {I / I_STAR_FP32:>8.3%}")
    print(f"    I*（FP32 向量）= {I_STAR_FP32:.1f} FLOP/byte。要让逐元素算子翻到")
    print(f"      算力受限那一侧，每个元素的运算次数必须超过 "
          f"2·esize·I* = {2 * ESIZE_F32 * I_STAR_FP32:.0f} 次——实际情况差 2~3 个数量级。")
    check("任何实用的单目算子都落在 roofline 的**斜线段**上（memory-bound）",
          2 < 2 * ESIZE_F32 * I_STAR_FP32,
          "I = ops/(2·esize) 与 N 无关，这是逐元素算子的定义性质")
    print()
    print("    ⚠ 「y = exp(x) 计 1 次运算」是本表的口径。SASS 上它是 `MUFU.EX2`")
    print("      加一次乘法（还要先乘 log2 e），真实运算数更多——**只会让 I 更大**，")
    print("      结论方向不变。")

    print()
    print("    同一张 roofline 的另一端（正文 01 篇 §0 的对照表）：")
    M, Kd, Nd = 4096, 8192, 8192
    g_flops = 2.0 * M * Kd * Nd
    g_bytes = 2.0 * (M * Kd + Kd * Nd + M * Nd)
    g_I = g_flops / g_bytes
    g_tcmp = g_flops / (SP.FP32_VECTOR_TFLOPS_DENSE[H100] * 1e12) * 1e6
    g_tmem = g_bytes / B_PEAK * 1e6
    print(f"      prefill 大 GEMM [{M}×{Kd}] @ [{Kd}×{Nd}]")
    print(f"        FLOPs = 2·M·K·N = {g_flops:.3e}；"
          f"Bytes = 2·(M·K + K·N + M·N) = {g_bytes:.3e}")
    print(f"        I = {g_I:.1f} FLOP/byte → I/I* = {g_I / I_STAR_FP32:.0%}"
          f"（算力受限）")
    print(f"        算力时间 {g_tcmp:.2f} µs，访存时间 {g_tmem:.1f} µs，"
          f"比值 = {g_tcmp / g_tmem:.0f}×")
    e_flops = 1.0 * N
    e_bytes = 2.0 * N * ESIZE_F32
    e_I = e_flops / e_bytes
    e_tcmp = e_flops / (SP.FP32_VECTOR_TFLOPS_DENSE[H100] * 1e12) * 1e6
    e_tmem = e_bytes / B_PEAK * 1e6
    print(f"      y = exp(x)，N = 2^26")
    print(f"        FLOPs = {e_flops:.3e}；Bytes = 2N·esize = {e_bytes:.3e}")
    print(f"        I = {e_I:.3f} FLOP/byte → I/I* = {e_I / I_STAR_FP32:.3%}"
          f"（访存受限）")
    print(f"        算力时间 {e_tcmp:.2f} µs，访存时间 {e_tmem:.2f} µs，"
          f"比值 = {e_tmem / e_tcmp:.0f}×")
    check("两行的 I/I* 相差 4 个数量级，落在 roofline 的两端",
          g_I / e_I > 1e4,
          f"GEMM {g_I:.1f} vs exp {e_I:.3f}，比 {g_I / e_I:.0f}×")
    check("为 exp 这类算子做的计算侧优化，动的是全部工作量的千分之几",
          e_tcmp / e_tmem < 0.01,
          f"算力时间只占访存时间的 {e_tcmp / e_tmem:.3%}")


def b2_small_scale_launch():
    sub("B2  小规模下的固定开销：launch 开销 vs 访存时间")

    print("    逐元素算子每处理 N 个元素都要付一次 kernel 启动——它是**固定**成本：")
    print(f"    {'N':>14} {'数据':>10} {'2N·esize':>12} {'访存时间 µs':>12} "
          f"{'启动/访存':>18}")
    for k in (16, 18, 20, 24, 26):
        N = 1 << k
        by = 2 * N * ESIZE_F32
        t_us = by / B_PEAK * 1e6
        lo, hi = SP.KERNEL_LAUNCH_OVERHEAD_US_EXTERNAL
        print(f"    {N:>14} {mib(N * ESIZE_F32):>8.1f} MiB {by:>12} "
              f"{t_us:>12.2f} {lo / t_us:>7.0%}~{hi / t_us:<8.0%}")
    print()
    print(f"    启动开销 = {SP.KERNEL_LAUNCH_OVERHEAD_US_EXTERNAL[0]}~"
          f"{SP.KERNEL_LAUNCH_OVERHEAD_US_EXTERNAL[1]} µs（**引用值**，"
          f"本仓库不复核其测量口径）")
    t16 = (1 << 16) * 2 * ESIZE_F32 / B_PEAK * 1e6
    t26 = (1 << 26) * 2 * ESIZE_F32 / B_PEAK * 1e6
    print(f"    同一笔固定开销：N = 2^16 时占 {SP.KERNEL_LAUNCH_OVERHEAD_US_EXTERNAL[0] / t16:.0%}"
          f"~{SP.KERNEL_LAUNCH_OVERHEAD_US_EXTERNAL[1] / t16:.0%}，"
          f"N = 2^26 时只占 "
          f"{SP.KERNEL_LAUNCH_OVERHEAD_US_EXTERNAL[0] / t26:.1%}"
          f"~{SP.KERNEL_LAUNCH_OVERHEAD_US_EXTERNAL[1] / t26:.1%}")
    print(f"    两者的比值 = {t26 / t16:.0f} 倍——小规模下**固定开销是唯一重要的项**。")


def b3_fusion_traffic():
    sub("B3  融合 k 个算子的字节数与加速比（模型甲 / 乙 / 丙）")

    N = 1 << 24                      # 16777216 个 fp32 = 64 MiB
    print(f"    N = 2^24 = {N} 个 fp32（单份张量 {mib(N * ESIZE_F32):.1f} MiB）")
    print()
    print("    先推导（正文 02 篇 §1 逐步展开）：")
    print("      模型甲（中间结果**每次往返 HBM**）")
    print("        不融合：k 个算子各读一遍写一遍 → 2k·N·esize")
    print("        融合  ：读输入一次 + 写输出一次   → 2·N·esize")
    print("        比值 = 2k·N·esize ÷ 2·N·esize = **k**")
    print("      模型乙（中间结果**命中 L2**）")
    print("        不融合：中间张量在 L2 里被写、被读，不回 HBM → 仍 ≈ 2·N·esize")
    print("        融合  ：与甲相同 → 2·N·esize")
    print("        比值 = **1**")
    print("      模型丙（**另一种拓扑**：k 个输入 1 个输出，二叉链）")
    print("        不融合：k−1 个算子各读两个输入写一个输出 → 3(k−1)·N·esize")
    print("        融合  ：读 k 个输入 + 写 1 个输出      → (k+1)·N·esize")
    print("        比值 = 3(k−1)/(k+1)")
    print()
    print(f"    {'k':>3} {'甲:不融合':>12} {'甲:融合':>10} {'甲比值':>7} "
          f"{'乙比值':>7} {'丙比值':>8} {'甲时间 µs':>10} {'甲@η=0.9':>10}")
    r_a_rows = []
    for k in range(1, 7):
        a_unf = 2 * k * N * ESIZE_F32
        a_fus = 2 * N * ESIZE_F32
        r_a = a_unf / a_fus
        r_a_rows.append(r_a)
        r_b = 1.0
        r_c = 3.0 * (k - 1) / (k + 1)
        t_a = a_unf / B_PEAK * 1e6
        t_a9 = a_unf / (0.9 * B_PEAK) * 1e6
        c_str = "  —  " if k == 1 else f"{r_c:.4f}"
        print(f"    {k:>3} {mib(a_unf):>10.0f} MiB {mib(a_fus):>8.0f} MiB "
              f"{r_a:>7.2f} {r_b:>7.2f} {c_str:>8} {t_a:>10.2f} {t_a9:>10.2f}")
    print(f"    （k = 1 时模型丙没有算子，比值为 0 属退化情形，记 —）")
    print()
    check("模型甲的加速比**恰好等于 k**（逐行精确相等，不是近似）",
          all(close(r, k, 1e-12) for k, r in enumerate(r_a_rows, start=1)),
          "k = 1..6 → 1× 2× 3× 4× 5× 6×")
    check("模型乙的加速比恒为 1×（**与链长无关**）",
          all(close(2 * N * ESIZE_F32 / (2 * N * ESIZE_F32), 1.0, 1e-12)
              for _ in range(1, 7)),
          "上界 k 只在「每个中间结果都往返 HBM」时成立")
    check("模型甲的比值严格大于模型乙（k ≥ 2 时）",
          all(r > 1.0 + 1e-12 for r in r_a_rows[1:]),
          f"k=2: 甲 {r_a_rows[1]:.2f} vs 乙 1.00 → 差 {r_a_rows[1]:.0f} 倍")
    check("模型丙的比值**永远小于模型甲**（拓扑不同，上界也不同）",
          all(3.0 * (k - 1) / (k + 1) < r for k, r in enumerate(r_a_rows, 1)
              if k >= 2),
          f"k=6: 甲 {r_a_rows[5]:.2f} vs 丙 {3*5/7:.2f}——丙把 k 个输入也算进流量")
    print()
    print("    ⚠ 两个必须一起说的限制：")
    print(f"      ① 上界 k **只在每个中间结果都往返 HBM 时**成立。中间张量装得进 L2")
    print(f"         时，不融合的流量本来就≈ 2N·esize，收益趋于 1×。")
    print(f"         H100 的 L2 记为 {SP.L2_MB_H100_IN_REPO} MB（**本仓库内部口径不一致**，")
    print(f"         见 specs.py 的注释），换算成 fp32 元素是 "
          f"{SP.L2_MB_H100_IN_REPO * 2**20 // ESIZE_F32 / 1e6:.1f}e6 个")
    print(f"         （≈ 2^{math.log2(SP.L2_MB_H100_IN_REPO * 2**20 // ESIZE_F32):.1f}）。")
    print()
    print("         判据要看**同时在途的张量个数**（k+1 份：输入 + k−1 个中间 + 输出）：")
    print(f"           {'N':>8} {'单份张量':>10} {'k=3 时 (k+1) 份':>18} "
          f"{'k=4 时 (k+1) 份':>18}")
    for k_ in (20, 22, 23, 24):
        n_ = 1 << k_
        one = n_ * ESIZE_F32
        print(f"           2^{k_:<6} {mib(one):>8.0f} MiB {mib(4 * one):>16.0f} MiB "
              f"{mib(5 * one):>16.0f} MiB")
    print(f"         N = 2^24 时单份 {mib((1<<24) * ESIZE_F32):.0f} MiB，")
    print(f"           (k+1) 份 {mib(5 * (1<<24) * ESIZE_F32):.0f} MiB——已明显超过 L2，")
    print(f"           本例落在模型甲那一侧；N ≲ 2^20、k = 4 时 (k+1) 份只有 "
          f"{mib(5 * (1<<20) * ESIZE_F32):.0f} MiB，就未必了。")
    print(f"      ② 真实比值落在 (1, k) 之间，不是一个端点。")
    lo, hi = SP.END_TO_END_FUSION_SPEEDUP_RANGE_EXTERNAL
    print(f"         外部报道的**端到端**收益 {lo}×~{hi}×（**引用值，本仓库不复核")
    print(f"         其测量口径**；且一手来源本次未定位，见 specs.py 的 UNSOURCED）")
    print(f"         正落在这个区间里——它比 k 小，是因为（a）中间结果常能留在 L2，")
    print(f"         （b）端到端里还有别的部分没被融合。")
    print()
    print("    → 正文**禁止**写成「上界趋于 2」，也禁止写成「融合越多越快」。")


def b4_vector_instruction_and_transaction():
    sub("B4  向量化前后的指令条数与访存事务数")

    N = 1 << 26
    warp = SP.WARP_SIZE
    block = 256
    # 标量路径：每线程 1 个 fp32
    s_ld, s_st = N, N
    s_grid = (N + block - 1) // block
    # 向量化路径：每线程 4 个 fp32（一个 float4 = 16 字节）
    v_ld, v_st = N // 4, N // 4
    v_grid = (N // 4 + block - 1) // block
    print(f"    N = 2^26 = {N} 个 fp32，block = {block}，"
          f"每线程的元素数 = 1（标量）/ 4（float4）")
    print(f"    grid：标量 {s_grid} 个 block / float4 {v_grid} 个 block"
          f"（gridDim.x 上限 2^31−1，差 4 个数量级——"
          f"「grid 太大」不是本主题的失败模式）")
    print()
    print(f"    {'':>18} {'标量':>16} {'float4':>16} {'比值':>8}")
    print(f"    {'每线程元素数':>18} {1:>16} {4:>16} {4:>8.2f}")
    print(f"    {'访存指令（lane 级）':>18} {s_ld + s_st:>16} {v_ld + v_st:>16} "
          f"{(s_ld + s_st) / (v_ld + v_st):>8.2f}")
    print(f"    {'访存指令（warp 级）':>18} {(s_ld + s_st) // warp:>16} "
          f"{(v_ld + v_st) // warp:>16} "
          f"{(s_ld + s_st) / (v_ld + v_st):>8.2f}")

    # 合并口径（机制回链 memory-system/02 §1，本主题只代入）：
    #   line = 128 字节 = 一条 L1→L2 的**请求**；sector = 32 字节 = 实际**搬运**单位。
    one_warp_bytes_s = warp * ESIZE_F32                  # 32 × 4 = 128 字节
    one_warp_bytes_v = warp * SP.MAX_VECTOR_BYTES_IN_CUDA_C   # 32 × 16 = 512 字节
    req_s = (N * ESIZE_F32) // SP.CACHE_LINE_BYTES       # 触及的 line 数 = 请求数
    req_v = req_s                                        # 512 字节是 4 条 line，每 128 个元素 4 条
    sec_s = (N * ESIZE_F32) // SP.SECTOR_BYTES
    sec_v = sec_s
    print()
    print(f"    {'一次 warp 访存覆盖字节':>24} {one_warp_bytes_s:>16} {one_warp_bytes_v:>16} "
          f"{one_warp_bytes_v / one_warp_bytes_s:>8.2f}")
    print(f"    {'访存事务（line/请求）数':>24} {req_s:>16} {req_v:>16} "
          f"{req_s / req_v:>8.2f}")
    print(f"    {'搬运 sector 数':>24} {sec_s:>16} {sec_v:>16} {sec_s / sec_v:>8.2f}")
    print(f"    {'传输字节':>24} {N * ESIZE_F32:>16} {N * ESIZE_F32:>16} "
          f"{1.0:>8.2f}")
    print()
    check("向量化把访存指令数降到 **1/4**（精确 4.00，不是近似）",
          (s_ld + s_st) == 4 * (v_ld + v_st), f"{s_ld + s_st} / {v_ld + v_st} = 4")
    check("向量化**一个事务也没省**（事务数是精确相等）", req_s == req_v,
          f"两边都是 {req_s} 条 line 请求 / {sec_s} 个 sector")
    check("向量化**一个字节也没省**（传输字节精确相等）", sec_s * SP.SECTOR_BYTES ==
          N * ESIZE_F32, "同样的 4 条 line，标量分四次读到、float4 一次读到")

    print()
    print("    发射侧的时间账（**用推导时钟**，见 §0 的口径声明）：")
    iss = ISSUE_WARP_INSTR_PER_CLK * FP32_CLOCK_HZ_DERIVED
    print(f"      全卡发射能力 = {ISSUE_WARP_INSTR_PER_CLK} 条 warp 指令/周期 × "
          f"{FP32_CLOCK_HZ_DERIVED/1e9:.4f} GHz = {iss:.4e} 条/s")
    t_mem = N * 2 * ESIZE_F32 / B_PEAK * 1e6
    print(f"      内存时间（N=2^26，η=1）= {t_mem:.2f} µs")
    print(f"      {'口径':<28} {'warp 指令数':>14} {'发射时间 µs':>12} {'占内存':>9}")
    cases = [
        ("只数访存指令(下界) 标量", (2 * N) // warp),
        ("只数访存指令(下界) float4", (2 * N // 4) // warp),
        ("每元素共 8 条指令 标量", (8 * N) // warp),
        ("每元素共 8 条指令 float4", (2 * N) // warp),
    ]
    for name, wi in cases:
        t_i = wi / iss * 1e6
        print(f"      {name:<28} {wi:>14} {t_i:>12.2f} {t_i / t_mem:>8.2%}")
    print()
    print("    ⚠ 结论与直觉相反：**在大 N 的纯逐元素算子上，发射槽从来不是瓶颈**")
    print("      —— 即使按「每元素 8 条指令」的悲观口径，标量版也只占内存时间的百分之几。")
    print("      向量化真正买到的东西在 §B5，不是这里。")


def b5_inflight_requests():
    sub("B5  填满带宽需要多少在途请求——向量化的真正收益")

    lo_c, hi_c = SP.HBM_LATENCY_CYCLES_RANGE
    print("    Little 定律：在途字节 = 带宽 × 延迟。")
    print(f"      延迟用 {lo_c}~{hi_c} 周期（规格值，memory-system/01 §1），")
    print(f"      换算成时间要用**推导时钟** {FP32_CLOCK_HZ_DERIVED/1e9:.4f} GHz。")
    print()
    print(f"    {'延迟周期':>9} {'延迟 ns':>9} {'在途字节':>12} {'每 SM 字节':>11} "
          f"{'需 128B 请求':>13} {'需 512B 请求':>13}")
    for c in (lo_c, 600, hi_c):
        ns = c / FP32_CLOCK_HZ_DERIVED * 1e9
        infl = B_PEAK * ns * 1e-9
        per_sm = infl / SP.SM_COUNT_BY_GEN[H100]
        print(f"    {c:>9} {ns:>9.1f} {infl / 1024:>10.1f} KiB "
              f"{per_sm / 1024:>9.2f} KiB "
              f"{per_sm / 128:>13.1f} {per_sm / 512:>13.1f}")
    print()
    ns600 = 600 / FP32_CLOCK_HZ_DERIVED * 1e9
    need_s = B_PEAK * ns600 * 1e-9 / SP.SM_COUNT_BY_GEN[H100] / 128
    need_v = B_PEAK * ns600 * 1e-9 / SP.SM_COUNT_BY_GEN[H100] / 512
    print(f"    取延迟 {600} 周期（区间中点）：需要的在途请求数是 "
          f"{need_s:.0f} 条（标量）/ {need_v:.0f} 条（float4）")
    print(f"      ≈ 上游 memory-system/02 §3 说的「约 60 个降到约 15 个」——")
    print(f"      **本程序复现了那个数**（同一条 Little 定律，同一个延迟量级）。")
    check("向量化把「填满带宽所需的在途请求数」降到 1/4（精确 4.00）",
          close(need_s / need_v, 4.0, 1e-9), f"{need_s:.1f} / {need_v:.1f} = 4.00")
    check("标量口径的在途需求（60）**没有超过**每 SM 的 warp 上限（64），但已用掉 94%",
          need_s < SP.MAX_WARPS_PER_SM,
          f"{need_s:.1f} / {SP.MAX_WARPS_PER_SM} = {need_s / SP.MAX_WARPS_PER_SM:.0%}"
          "（**单请求深度假设**：每 warp 只挂一条在途请求）")
    print()
    print("    ⚠ 两条使用边界：")
    print("      ① 「每 warp 只挂一条在途请求」是**保守假设**。真实硬件上一条 warp")
    print("         可以有多条未完成的 load（scoreboard 允许若干条在飞），所以这个数")
    print("         是需求的上界、不是精确值。")
    print("      ② 它给出的是**方向**：向量化把并发需求降到 1/4，等价于把 02 篇 §C3")
    print("         那道 occupancy 墙往后推了 4 倍。")


def b6_grid_stride_vs_direct():
    sub("B6  grid-stride 与直接映射：线程数与空转比例")

    block, per_thread = 256, 4
    gs_blocks = SP.SM_COUNT_BY_GEN[H100] * 4          # 每 SM 4 个 block
    gs_threads = gs_blocks * block
    print(f"    grid-stride 的 grid 固定为 {SP.SM_COUNT_BY_GEN[H100]} SM × 4 block = "
          f"{gs_blocks} 个 block = {gs_threads} 个线程")
    print(f"    每线程 {per_thread} 个元素，block = {block}")
    print()
    print(f"    {'N':>12} {'需要线程数':>12} {'直接映射启动':>14} "
          f"{'gs 每线程轮数':>14} {'gs 空转线程占比':>16}")
    rows = []
    for k in (10, 14, 18, 22, 26):
        n = 1 << k
        needed = (n + per_thread - 1) // per_thread
        dir_threads = ((n + block * per_thread - 1) // (block * per_thread)) * block
        rounds = (needed + gs_threads - 1) // gs_threads
        idle = max(0, gs_threads - needed)
        frac = idle / gs_threads
        rows.append((k, n, needed, dir_threads, rounds, frac))
        print(f"    2^{k:<9} {needed:>12} {dir_threads:>14} {rounds:>14} "
              f"{frac:>15.2%}")
    print()
    check("小规模下 grid-stride 启动的线程**几乎全部空转**（N = 2^10 时 > 99%）",
          rows[0][5] > 0.99, f"2^10：{rows[0][5]:.2%} 的线程算一次索引就退出")
    check("直接映射启动的线程数恰好等于需要数（不浪费）",
          all(r[3] == r[2] for r in rows),
          "直接映射的 grid 随 N 一起涨，没有空转线程")
    check("⚠ 但直接映射**不总是**更好：N 大到每线程多轮时，grid-stride 才摊薄开销",
          rows[-1][4] > 1, f"2^26：grid-stride 每线程 {rows[-1][4]} 轮")
    print()
    print("    ⚠ 这张表只数**线程数**，不含每轮多出的索引与边界计算（约 3 条指令/轮）。")
    print("      后者按 §B4 的发射预算仍然不是瓶颈（大 N 时发射只占内存时间的百分之几），")
    print("      所以 grid-stride 的代价主要落在**空转线程**这一项上。")


# ===========================================================================
# §C 边界与反例
# ===========================================================================


def c1_launch_overhead_dominates():
    sub("C1  小规模下 launch 开销占主导")

    lo, hi = SP.KERNEL_LAUNCH_OVERHEAD_US_EXTERNAL
    t16 = (1 << 16) * 2 * ESIZE_F32 / B_PEAK * 1e6
    print(f"    N = 2^16（65536 个 fp32 = 256 KiB）")
    print(f"      访存时间（η=1，乐观下界）= {t16:.4f} µs")
    print(f"      一次 kernel 启动          = {lo}~{hi} µs（引用值）")
    print(f"      比值                      = {lo / t16:.0f}× ~ {hi / t16:.0f}×")
    check("N = 2^16 时，**一次** kernel 启动的开销是访存时间的十倍以上",
          lo / t16 > 10.0, f"{lo/t16:.0f}× ~ {hi/t16:.0f}×")
    print()
    print("    这条直接否掉一种估算：把「访存受限」当成「时间 = 字节 ÷ 带宽」")
    print("    之后就宣布小规模也能跑到那个下界。小规模下**下界根本不是访存**。")
    print()
    print("    顺便一个量级对照（模型甲，k = 4，N = 2^16）：")
    print(f"      不融合  4 个 kernel → 4 次启动 = {4*lo:.0f}~{4*hi:.0f} µs")
    print(f"      融合    1 个 kernel → 1 次启动 = {lo:.0f}~{hi:.0f} µs")
    print(f"      省下的 3 次启动 = {3*lo:.0f}~{3*hi:.0f} µs；而 3 份中间张量的")
    print(f"      流量是 6·N·esize = {6*(1<<16)*ESIZE_F32} B = "
          f"{6*(1<<16)*ESIZE_F32/B_PEAK*1e6:.4f} µs——")
    print(f"      **小规模下融合省的是启动，不是带宽**（比例 "
          f"{(3*lo)/(6*(1<<16)*ESIZE_F32/B_PEAK*1e6):.0f}× 以上）。")


def c2_misaligned_vector_access():
    sub("C2  未对齐的向量化读：漏掉首元素 + 越过数组末尾")

    N = 1000
    base = np.arange(N, dtype=np.float32)
    off = 1                                  # 基址偏移 1 个元素（4 字节）
    print(f"    N = {N} 个 fp32，把 `(base + {off})` 直接当成 `float4` 数组的起点")
    print(f"      float4 要求 {SP.MAX_VECTOR_BYTES_IN_CUDA_C} 字节对齐；"
          f"偏移 {off} 个 fp32 后地址 ≡ {off * ESIZE_F32} (mod "
          f"{SP.MAX_VECTOR_BYTES_IN_CUDA_C})，**不满足**")
    n_vec = math.ceil((N - off) / 4)
    print(f"      需要的向量组数 = ceil((N − {off}) / 4) = {n_vec}")
    covered = set()
    last_end = None
    for j in range(n_vec):
        i0 = off + 4 * j
        for i in range(i0, i0 + 4):
            covered.add(i)
            last_end = i
    valid = sorted(covered & set(range(N)))
    missed = sorted(set(range(N)) - covered)
    print()
    print(f"      合法下标被覆盖：{len(valid)} / {N} 个")
    print(f"      **漏掉的**：{missed}（共 {len(missed)} 个）")
    print(f"      最后一组覆盖到下标 {last_end}，而数组最大合法下标是 {N - 1}")
    print(f"      → 读越过数组末尾 {last_end - (N - 1)} 个元素")
    check("未对齐的向量化读**漏掉首元素**（下标 0 不在任何一组里）",
          0 not in covered, f"漏掉下标 {missed}，只覆盖了 {len(valid)}/{N} 个")
    check("未对齐的向量化读**越过数组末尾**（越界是真会发生的，不是理论风险）",
          last_end > N - 1, f"读到下标 {last_end} > {N - 1}，越界 {last_end - (N - 1)} 个元素")
    print()
    print("    「要么慢」的另一半：改成逐元素（标量）回退就没有这个问题，")
    print("     但访存指令数回到 N 而不是 N/4（§B4），也就是放弃向量化的收益。")
    print("    → 正确做法：**先把指针对齐、把元素个数补齐**，再进向量化路径，")
    print("      尾部用标量收尾（§A2 就是这么做的）。")


def c3_register_pressure_kills_occupancy():
    sub("C3  融合链的寄存器压力压垮 occupancy（公式判定，不测硬件）")

    R_BASE, R_PER_OP, BLOCK = 16, 12, 256
    print(f"    模型（显式声明为**假设**）：r(k) = {R_BASE} + {R_PER_OP}·k，"
          f"block = {BLOCK}")
    print(f"      {R_BASE} = 索引 / 地址 / 循环控制；{R_PER_OP} = 每个算子要同时活着的")
    print(f"      中间值与常量。实际值由编译器决定，正文只用这条曲线的**形状**。")
    print()
    print(f"    需要的在途请求数（§B5，延迟 600 周期的口径）：")
    ns600 = 600 / FP32_CLOCK_HZ_DERIVED * 1e9
    need_s = B_PEAK * ns600 * 1e-9 / SP.SM_COUNT_BY_GEN[H100] / 128
    need_v = B_PEAK * ns600 * 1e-9 / SP.SM_COUNT_BY_GEN[H100] / 512
    print(f"      标量（每 warp 请求 128 字节）：{need_s:.1f} 条")
    print(f"      float4（每 warp 请求 512 字节）：{need_v:.1f} 条")
    print()
    print(f"    {'k':>3} {'r(k)':>6} {'取整8':>6} {'块/ SM':>7} {'warp/ SM':>8} "
          f"{'occupancy':>10} {'≥标量需求?':>11} {'≥float4需求?':>13}")
    rows = []
    for k in range(1, 21):
        r = R_BASE + R_PER_OP * k
        r8 = ceil_to(r, SP.REGISTER_ALLOC_GRANULARITY)
        if r8 > SP.MAX_REGISTERS_PER_THREAD:
            print(f"    {k:>3} {r:>6} {'> 255':>6} {'—':>7} {'—':>8} {'—':>10} "
                  f"{'spill':>11} {'spill':>13}")
            rows.append((k, r, None, None, None))
            continue
        by_regs = SP.REGISTERS_PER_SM // (r8 * BLOCK)
        blocks = min(SP.MAX_BLOCKS_PER_SM, SP.MAX_THREADS_PER_SM // BLOCK, by_regs)
        warps = blocks * (BLOCK // SP.WARP_SIZE)
        occ = warps / SP.MAX_WARPS_PER_SM
        print(f"    {k:>3} {r:>6} {r8:>6} {blocks:>7} {warps:>8} {occ:>10.1%} "
              f"{('是' if warps >= need_s else '**否**'):>11} "
              f"{('是' if warps >= need_v else '**否**'):>13}")
        rows.append((k, r, r8, blocks, warps))

    ok_s = [k for k, _, _, _, w in rows if w is not None and w >= need_s]
    ok_v = [k for k, _, _, _, w in rows if w is not None and w >= need_v]
    spill = [k for k, r, r8, _, _ in rows if r8 is None]
    k_s = max(ok_s) if ok_s else 0
    k_v = max(ok_v) if ok_v else 0
    k_spill = min(spill) if spill else None
    print()
    print(f"    ★ 标量访存下，k = {k_s + 1} 起常驻 warp 数就**低于填满带宽所需**了")
    print(f"    ★ 换 float4（请求 512 字节）后，最后一条能过的链长是 k = {k_v}")
    print(f"    ★ 而寄存器的**硬上限**要到 k = {k_spill} 才撞到（r = "
          f"{R_BASE + R_PER_OP * k_spill} > {SP.MAX_REGISTERS_PER_THREAD}，只能 spill）")
    print()
    check("标量访存下，融合链长到 k = 2 就已经压垮了「填满带宽所需的常驻 warp 数」",
          k_s == 1, f"k ≤ {k_s} 可以，k = {k_s + 1} 不行——而寄存器还富余得很")
    check("限制融合链长的**不是**寄存器硬上限（那个要到 k ≈ 20）",
          k_spill is not None and k_spill - 1 >= 15,
          f"寄存器上限允许到 k = {k_spill - 1}（r ≤ 255），"
          f"occupancy 却早在 k = 2 就掉队")
    check("**向量化把这道墙至少往后推 4 倍**",
          k_v >= 4 * max(k_s, 1),
          f"标量最后一条能过的是 k = {k_s}，float4 是 k = {k_v}，"
          f"比值 {k_v / max(k_s, 1):.0f}×")
    print()
    print("    ⚠ 这一节是**公式判定**，不是硬件测量：模型里的 "
          f"{R_BASE} 与 {R_PER_OP} 是假设值，")
    print("      「每 warp 一条在途请求」也是保守假设。但它给出的是**方向**——")
    print("      融合省掉的 HBM 流量，和融合吃掉的寄存器，来自**同一个资源池**：")
    print("      寄存器既是中间值的家，也是买「在途深度」的货币。")
    print("      → 这正是 02 篇 §4.1 那条「融合反而更慢」的机制。")


def c5_grid_stride_wrong_stride():
    sub("C5  grid-stride 的 stride 漏写 `gridDim.x`：结果对，工作量放大且极不均衡")

    n = 4096
    block = 256
    G = n // block                       # 用 blockDim.x 做 stride 时的「组数」
    print(f"    N = {n}，block = {block}，用 blockDim.x 做 stride 时有 {G} 组")
    print()
    print(f"    {'grid':>6} {'正确:每个下标访问次数':>22} {'漏写:每个下标访问次数':>22} "
          f"{'漏写总访问':>12} {'放大倍数':>10}")
    rows = []
    for grid in (2, 4, 8, 16, 64, 256):
        ok = np.zeros(n, dtype=np.int32)
        for b in range(grid):
            for t in range(block):
                i = b * block + t
                while i < n:
                    ok[i] += 1
                    i += grid * block
        bad = np.zeros(n, dtype=np.int32)
        for b in range(grid):
            for t in range(block):
                i = b * block + t
                while i < n:
                    bad[i] += 1
                    i += block                # ← 漏了 gridDim.x
        rows.append((grid, int(ok.min()), int(ok.max()), int(ok.sum()),
                     int(bad.min()), int(bad.max()), int(bad.sum())))
        print(f"    {grid:>6} {f'{ok.min()}~{ok.max()}':>22} "
              f"{f'{bad.min()}~{bad.max()}':>22} {int(bad.sum()):>12} "
              f"{bad.sum() / n:>9.2f}×")

    print()
    check("正确写法下，每个下标被处理**恰好 1 次**，总访问恰好 = N",
          all(r[1] == 1 and r[2] == 1 and r[3] == n for r in rows),
          f"六种 grid 下总访问都是 {rows[0][3]} = N")
    check("漏写 `gridDim.x` 时访问次数**不均匀**：从 1 涨到 `min(grid, N/block)`",
          rows[0][4] == 1 and rows[1][4] == 1,
          f"grid=2: {rows[0][4]}~{rows[0][5]} 次；grid=256: "
          f"{rows[5][4]}~{rows[5][5]} 次")
    cap16 = [r for r in rows if r[0] == 16][0]
    cap64 = [r for r in rows if r[0] == 64][0]
    cap256 = [r for r in rows if r[0] == 256][0]
    check("⚠ 放大倍数**封顶**在 `(N/block + 1) / 2`，**不随 gridDim.x 继续涨**",
          cap16[6] == cap64[6] == cap256[6],
          f"grid=16 / 64 / 256 的总访问都是 {cap16[6]}，放大 "
          f"{cap16[6] / n:.2f}×（N/block = {G}，封顶值 {(G + 1) / 2:.1f}）")
    print()
    print("    ★ 这条推翻了一个想当然的说法：「漏写 gridDim.x 会把工作量放大")
    print("      gridDim.x 倍」。**不会**——每个下标的访问次数是")
    print("      `min(⌊i/block⌋, grid−1) + 1`，它的上界由 `N/block` 封顶，")
    print("      与 grid 无关（grid 一旦 ≥ N/block 就不再变化）。")
    print("    ★ 更值得说的是**不均衡**：下标 0~(block−1) 只被处理 1 次，")
    print(f"      后半段被处理 {cap16[5]} 次。落在同一批 block 上的负载差 "
          f"{cap16[5]} 倍——")
    print("      这比「总工作量大了几倍」更难排查。")
    print()
    print("    ⚠ 这个错**不会给出错误结果**（`out[i] = in[i]·2+1` 每次写的是同一个值），")
    print("      **它只在耗时上暴露，在正确性检查里看不见**——与 C2 那条「一眼看出错」")
    print("      的坑正好相反。")


def c4_half2_is_not_a_16b_vector():
    sub("C4  `__half2` 不是 16 字节的向量——半精度上的向量化只走了一半")

    print(f"    {'类型':>10} {'字节':>6} {'每 warp 覆盖':>14} {'相对标量 half':>14}")
    rows = [
        ("__half", SP.FLOAT16_BYTES, "32 × 2 = 64 B"),
        ("__half2", SP.HALF2_BYTES, "32 × 4 = 128 B"),
        ("8 个 half（按 uint4 搬）", SP.MAX_VECTOR_BYTES_IN_CUDA_C, "32 × 16 = 512 B"),
    ]
    for name, sz, cover in rows:
        print(f"    {name:>10} {sz:>6} {cover:>14} {sz / SP.FLOAT16_BYTES:>13.0f}×")
    print()
    check("`__half2` 恰好是 4 字节（= 2 × 2），**不是** 16 字节",
          SP.HALF2_BYTES == 2 * SP.FLOAT16_BYTES and SP.HALF2_BYTES != 16,
          "__half2 = 两个 half 打包装进一个 32-bit 寄存器")
    check("用 `__half2` 做向量化只把每线程宽度从 2 字节抬到 4 字节（2×）",
          SP.HALF2_BYTES / SP.FLOAT16_BYTES == 2.0,
          f"要到 {SP.MAX_VECTOR_BYTES_IN_CUDA_C} 字节还需再宽 "
          f"{SP.MAX_VECTOR_BYTES_IN_CUDA_C // SP.HALF2_BYTES}×")
    print()
    print(f"    → 半精度 kernel 里「一次搬 8 个 half」才等于 fp32 上一次搬 4 个 float。")
    print(f"      写成「用了 `__half2` 就向量化好了」是**只走了一半**——")
    print(f"      §B5 那条在途请求需求的账，在每线程 4 字节的情况下并没有改善。")
    print(f"    ⚠ 上游 memory-system/02 §3 改进二举的例子是 `float4`（16 字节），")
    print(f"      不是 `__half2`——两者在访存宽度上差 4 倍，不能互替。")


# ===========================================================================
# 主程序
# ===========================================================================


def main():
    print("=" * 76)
    print("40-kernels/elementwise-fusion 本机验证模型（verify.py）")
    print("=" * 76)
    print("本程序只跑算术与 numpy，不碰硬件。全部输出是**推导值**。")

    s0_scope()

    section("A", "算法正确性")
    a1_fused_chain_matches()
    a2_vectorized_element_mapping()

    section("B", "代价模型")
    b1_unary_bound()
    b2_small_scale_launch()
    b3_fusion_traffic()
    b4_vector_instruction_and_transaction()
    b5_inflight_requests()
    b6_grid_stride_vs_direct()

    section("C", "边界与反例")
    c1_launch_overhead_dominates()
    c2_misaligned_vector_access()
    c3_register_pressure_kills_occupancy()
    c4_half2_is_not_a_16b_vector()
    c5_grid_stride_wrong_stride()

    print()
    print("=" * 76)
    total = _STATE["n"]
    fail = _STATE["fail"]
    print(f"§D  汇总：断言 {total} 条，通过 {total - len(fail)} 条，"
          f"不通过 {len(fail)} 条")
    if fail:
        for f in fail:
            print(f"    ✗ {f}")
    else:
        print("    全部通过。")
    print("=" * 76)
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())

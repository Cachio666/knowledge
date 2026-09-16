#!/usr/bin/env python3
"""40-kernels/gemm 主题的本机验证模型。

**本模块是推导，不是硬件测量。** 它只跑算术与 numpy，不碰硬件——本机（macOS）
没有 CUDA 工具链也没有 GPU，`code/` 里的 6 个 `.cu` 内核的运行结果本仓库没有。

它覆盖本主题全部 6 篇文档的可推导部分，分三类断言（CONTRIBUTING §2）：

  A 算法正确性  numpy 参考实现 vs 文档所述算法（分块 / split-K / Stream-K 工作划分）
  B 代价模型    按文档公式算 SMEM 占用 / 算术强度 / 寄存器估算 / 访存字节 / 与
                roofline 拐点的比值，逐条打印数值并给出判定
  C 边界与反例  §4 声称「会出问题」的输入必须**真的跑出问题**——断言的是
                失败行为成立，不是成功

**数字的口径**（沿用 20-gpu-architecture 的三分法）：
  · 规格值 = 来自 `specs.py`，逐条带 `# 来源：`
  · 推导值 = 本程序算出，可在这里复核
  · 引用值 = 外部来源，标注「本仓库不复核其测量口径」

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


def kb(nbytes):
    return nbytes / 1024.0


def close(a, b, rel=1e-9):
    return abs(a - b) <= rel * max(1.0, abs(a), abs(b))


H100 = "H100"
B200 = "B200"

# 每 SM 的 FP32 向量速率（推导值：整卡峰值 ÷ SM 数）
PER_SM_FP32_FLOPS = SP.FP32_VECTOR_TFLOPS_DENSE[H100] * 1e12 / SP.SM_COUNT_BY_GEN[H100]


# ===========================================================================
# §0 口径
# ===========================================================================


def s0_scope():
    section(0, "口径")
    print("本程序全部输出是**推导值**：跑的是算术与 numpy，本机没有 GPU，")
    print("6 个 .cu 内核的运行结果本仓库没有。")
    print()
    print(f"主用机器：{H100} SXM —— "
          f"{SP.SM_COUNT_BY_GEN[H100]} SM / "
          f"{SP.FP32_VECTOR_TFLOPS_DENSE[H100]} TFLOPS FP32 向量 / "
          f"{SP.TC_TFLOPS_DENSE_SPARSE['FP16'][H100][0]} TFLOPS FP16 dense / "
          f"{SP.HBM_BW_GBps_BY_GEN[H100]} GB/s")
    print(f"对照机器：{B200} SXM —— "
          f"{SP.SM_COUNT_BY_GEN[B200]} SM / "
          f"{SP.TC_TFLOPS_DENSE_SPARSE['FP16'][B200][0]} TFLOPS FP16 dense / "
          f"{SP.HBM_BW_GBps_BY_GEN[B200]} GB/s")
    print(f"每 SM 的 FP32 向量速率（推导值）= {SP.FP32_VECTOR_TFLOPS_DENSE[H100]}e12 ÷ "
          f"{SP.SM_COUNT_BY_GEN[H100]} = {PER_SM_FP32_FLOPS:.4e} FLOP/s")
    print(f"每 SM 共享内存上限 = {SP.SMEM_PER_SM_MAX_KB} KB；"
          f"每 SM 寄存器文件 = {SP.REGISTERS_PER_SM} 个 32-bit 寄存器（"
          f"{SP.REGISTER_FILE_PER_SM_KB} KB）")
    print()
    print("⚠ 全文出现的「算术强度」有两套口径，正文每次使用都会注明是哪一套：")
    print("   · **block 级（L2 零命中）**：只算这一个 block 的 A/B tile 从 HBM 取一次；")
    print("   · **整题级（每个元素只从 HBM 取一次）**：roofline 原文的口径。")


# ===========================================================================
# §A 算法正确性
# ===========================================================================


def a1_tiled_gemm():
    sub("A1  分块 GEMM 与 numpy 参考实现一致")
    rng = np.random.default_rng(20260916)
    M, N, K = 96, 80, 48
    A = rng.standard_normal((M, K))
    B = rng.standard_normal((K, N))
    C_ref = A @ B

    BM, BN, BK = 32, 16, 16

    def tiled(A, B, BM, BN, BK):
        M, K = A.shape
        _, N = B.shape
        C = np.zeros((M, N))
        for i0 in range(0, M, BM):
            for j0 in range(0, N, BN):
                acc = np.zeros((min(BM, M - i0), min(BN, N - j0)))
                for k0 in range(0, K, BK):
                    a = A[i0:i0 + BM, k0:k0 + BK]
                    b = B[k0:k0 + BK, j0:j0 + BN]
                    acc += a @ b
                C[i0:i0 + BM, j0:j0 + BN] = acc
        return C

    C_t = tiled(A, B, BM, BN, BK)
    err = float(np.max(np.abs(C_t - C_ref)))
    print(f"    BM/BN/BK = {BM}/{BN}/{BK}，形状 {M}×{N}×{K}，最大绝对误差 = {err:.3e}")
    check("分块实现与 numpy 参考 GEMM 一致（float64，误差 < 1e-9）", err < 1e-9,
          f"max|Δ| = {err:.3e}")

    # 分块结果不应依赖 BK 的选择（数学上恒等，浮点上只差舍入）
    C_other = tiled(A, B, BM, BN, 8)
    err2 = float(np.max(np.abs(C_other - C_t)))
    print(f"    换 BK=8 后与 BK=16 的差 = {err2:.3e}")
    check("分块结果不依赖 BK（只差浮点舍入，< 1e-12）", err2 < 1e-12,
          f"max|Δ| = {err2:.3e}")

    # 非整除边界：M/N/K 不是 BM/BN/BK 的整数倍时也要对
    M2, N2, K2 = 100, 70, 50
    A2 = rng.standard_normal((M2, K2))
    B2 = rng.standard_normal((K2, N2))
    err3 = float(np.max(np.abs(A2 @ B2 - tiled(A2, B2, 32, 16, 16))))
    print(f"    非整除形状 {M2}×{N2}×{K2} 的最大绝对误差 = {err3:.3e}")
    check("非整除形状下分块实现仍正确（< 1e-9）", err3 < 1e-9, f"max|Δ| = {err3:.3e}")


def a2_splitk_decomposition():
    sub("A2  split-K 的部分和拼回原结果（fp32）")
    rng = np.random.default_rng(4096)
    M = N = 256
    K = 65536
    S = 32               # split 段数
    KC = K // S
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    C_ref = (A.astype(np.float64) @ B.astype(np.float64))

    partials = []
    for s in range(S):
        a = A[:, s * KC:(s + 1) * KC]
        b = B[s * KC:(s + 1) * KC, :]
        partials.append((a @ b).astype(np.float32))

    C_seq = np.zeros((M, N), dtype=np.float32)
    for p in partials:
        C_seq = (C_seq + p).astype(np.float32)
    C_rev = np.zeros((M, N), dtype=np.float32)
    for p in reversed(partials):
        C_rev = (C_rev + p).astype(np.float32)
    # 树形归约
    level = [p.copy() for p in partials]
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level) - 1, 2):
            nxt.append((level[i] + level[i + 1]).astype(np.float32))
        if len(level) % 2:
            nxt.append(level[-1])
        level = nxt
    C_tree = level[0]

    scale = float(np.abs(C_ref).max())        # 结果的量级
    print(f"    形状 {M}×{N}×{K}，S={S} 段，每段 K={KC}（fp32 累加）")
    print(f"    参考结果的量级 max|C| = {scale:.2f}")
    print()
    print("    两种误差口径都要看（回链 10-foundations/numerical-linear-algebra/02）：")
    print("      · 相对**量级**误差 = max|Δ| / max|C|  ← 衡量整体精度")
    print("      · 相对**元素**误差 = max(|Δ|/|C_ij|)  ← 被接近 0 的元素支配，会大得吓人")
    print()
    print(f"    {'归约顺序':<10}{'max|Δ|':>12}{'相对量级':>12}{'相对元素':>12}{'中位相对元素':>14}")
    metrics = {}
    for name, Cx in (("顺序", C_seq), ("逆序", C_rev), ("树形", C_tree)):
        d = np.abs(Cx - C_ref)
        rel_elem = d / np.maximum(np.abs(C_ref), 1e-30)
        metrics[name] = (float(d.max()), float(d.max() / scale),
                         float(rel_elem.max()), float(np.median(rel_elem)))
        print(f"    {name:<10}{metrics[name][0]:>12.3e}{metrics[name][1]:>12.3e}"
              f"{metrics[name][2]:>12.3e}{metrics[name][3]:>14.3e}")
    print()
    check("三种归约的「相对量级」误差都 < 1e-4（整体精度没问题）",
          all(m[1] < 1e-4 for m in metrics.values()),
          f"最大 {max(m[1] for m in metrics.values()):.3e}")
    check("三种归约的「相对元素」误差都远大于「相对量级」误差（被小元素支配）",
          all(m[2] > 100 * m[1] for m in metrics.values()))
    same = np.array_equal(C_seq, C_rev) and np.array_equal(C_seq, C_tree)
    print(f"    顺序 / 逆序 / 树形三种结果的位模式是否完全相同？ {same}")
    check("三种归约顺序**不是**逐位相同（顺序确实改变结果）", not same)
    return C_seq, C_rev, C_tree


def a3_streamk_partition():
    sub("A3  split-K 与 Stream-K 的工作对划分都是完备划分")
    T = 4            # (M/BM)·(N/BN)
    U_per_tile = 64  # K / BK_chunk，Stream-K 的工作对粒度
    U = T * U_per_tile

    # --- split-K：网格 = T·S，block b → tile b//S, split b%S ---
    #     每个 split 拥有 U_per_tile/S 个连续的工作对（要求整除）
    S = 16
    grid_sk = T * S
    per_split = U_per_tile // S
    owners_sk = {}
    for b in range(grid_sk):
        t, c = b // S, b % S
        for j in range(per_split):
            owners_sk.setdefault(t, []).append(c * per_split + j)
    covered = sum(len(v) for v in owners_sk.values())
    print(f"    split-K: 网格 = T·S = {T}·{S} = {grid_sk} 个 block，"
          f"每个 tile 切成 {S} 段、每段 {per_split} 个工作对，"
          f"覆盖工作对 {covered}/{U}")
    check("split-K 覆盖每个 (tile, chunk) 恰好一次",
          covered == U and all(len(v) == U_per_tile for v in owners_sk.values()) and
          all(sorted(v) == list(range(U_per_tile)) for v in owners_sk.values()))
    print(f"      ⚠ split-K 的网格数只能取 T·S（S 为整数段数）——它是**量化**的：")
    for s in (16, 32, 64):
        print(f"        S={s:>3} → 网格 {T*s:>4} 个 block"
              f"（机器有 {SP.SM_COUNT_BY_GEN[H100]} 个 SM，"
              f"{'填满' if T*s >= SP.SM_COUNT_BY_GEN[H100] else '**填不满**'}，"
              f"且 {T*s} 不是 SM 数的倍数 → 有余数波）")

    # --- Stream-K：网格 = P，block b 取连续的 U/P 个全局工作对 ---
    for P in (128, 132, 256):
        lo = [(b * U) // P for b in range(P)]
        hi = [((b + 1) * U) // P for b in range(P)]
        units = [hi[b] - lo[b] for b in range(P)]
        seen = set()
        owners = {}          # tile → 拥有它的**不同 block** 的集合
        for b in range(P):
            for u in range(lo[b], hi[b]):
                seen.add(u)
                owners.setdefault(u // U_per_tile, set()).add(b)
        n_own = [len(v) for v in owners.values()]
        print(f"    Stream-K: 网格 = P = {P}，覆盖 {len(seen)}/{U} 个工作对，"
              f"每 block 拿 {min(units)}~{max(units)} 个，"
              f"单 tile 的 owner block 数 {min(n_own)}~{max(n_own)}"
              f"（split-K 口径下这一列恒为 S=16）")
        check(f"Stream-K(P={P}) 覆盖每个工作对恰好一次", len(seen) == U)
        check(f"Stream-K(P={P}) 的负载不均 ≤ 1 个工作对",
              max(units) - min(units) <= 1)


# ===========================================================================
# §B 代价模型
# ===========================================================================


def roofline_ridge(peak_tflops, bw_gbps):
    return peak_tflops * 1e12 / (bw_gbps * 1e9)


def b1_roofline_ridge():
    sub("B1  roofline 拐点（规格值代入，非本主题新推）")
    i_fp16 = roofline_ridge(SP.TC_TFLOPS_DENSE_SPARSE["FP16"][H100][0], SP.HBM_BW_GBps_BY_GEN[H100])
    i_fp32 = roofline_ridge(SP.FP32_VECTOR_TFLOPS_DENSE[H100], SP.HBM_BW_GBps_BY_GEN[H100])
    print(f"    I*(FP16, H100) = 989.5e12 ÷ 3350e9 = {i_fp16:.4f} FLOP/byte")
    print(f"    I*(FP32 向量, H100) = 67.0e12 ÷ 3350e9 = {i_fp32:.4f} FLOP/byte")
    check("I*(FP16) ≈ 295.4", close(i_fp16, 295.4, 1e-3), f"{i_fp16:.4f}")
    check("I*(FP32) ≈ 20.0", close(i_fp32, 20.0, 1e-3), f"{i_fp32:.4f}")
    return i_fp16, i_fp32


def b2_naive_ai(i_fp32, i_fp16):
    sub("B2  朴素 GEMM 的算术强度（两套口径）")
    print("    口径一：累加器在寄存器里，每次迭代读 A、B 各一次")
    print("            FLOP = 2·K，字节 = 2·K·esize  ⟹  I = 1/esize")
    print("    口径二：K 循环在 kernel 外，C 每一步都要读写回来（beta·C）")
    print("            FLOP = 2·K，字节 = 4·K·esize  ⟹  I = 1/(2·esize)")
    print()
    print("    口径三（roofline 原文，作为对照）：每个元素只从 HBM 取一次")
    print("            I = 2MNK / ((MK+KN+MN)·esize)，方阵 n 时 = n/esize/3")
    print()
    for name, peak, ridge, esize in (
            ("FP32 向量", SP.FP32_VECTOR_TFLOPS_DENSE[H100], i_fp32, SP.FLOAT32_BYTES),
            ("FP16 TC", SP.TC_TFLOPS_DENSE_SPARSE["FP16"][H100][0], i_fp16, SP.FLOAT16_BYTES)):
        n = 4096
        i1 = 1.0 / esize
        i2 = 1.0 / (2 * esize)
        i3 = 2.0 * n ** 3 / ((3.0 * n ** 2) * esize)
        print(f"    {name}（峰值 {peak} TFLOPS，I* = {ridge:.1f}）：")
        print(f"      口径一 I = {i1:.4f}   → 可达 = min(峰值, B·I) = "
              f"{min(peak, SP.HBM_BW_GBps_BY_GEN[H100]*i1/1e3):.4f} TFLOPS "
              f"= 峰值的 {min(peak, SP.HBM_BW_GBps_BY_GEN[H100]*i1/1e3)/peak*100:.2f}%")
        print(f"      口径二 I = {i2:.4f}   → 可达 = "
              f"{min(peak, SP.HBM_BW_GBps_BY_GEN[H100]*i2/1e3):.4f} TFLOPS "
              f"= 峰值的 {min(peak, SP.HBM_BW_GBps_BY_GEN[H100]*i2/1e3)/peak*100:.2f}%")
        print(f"      口径三 I = {i3:.1f}（n={n}）→ 越过拐点，算力受限")
        check(f"{name}：口径一判为 bandwidth-bound（I < I*）", i1 < ridge)
        check(f"{name}：口径三判为 compute-bound（I > I*）", i3 > ridge)
    print()
    print("    ⚠ 同一道 GEMM，两套口径给出的瓶颈类型相反——这正是本主题的全部张力：")
    print("      朴素内核写出来的**实际访存**把它钉在口径一，而 tiling 才是把它搬向口径三的手段。")
    i_cube_fp32 = 2.0 * 4096 ** 3 / ((3.0 * 4096 ** 2) * SP.FLOAT32_BYTES)
    i_cube_fp16 = 2.0 * 4096 ** 3 / ((3.0 * 4096 ** 2) * SP.FLOAT16_BYTES)
    print(f"    方阵 n=4096 的整题级 AI：FP32 = n/esize/3 = {i_cube_fp32:.1f}，"
          f"FP16 = {i_cube_fp16:.1f}")

    print()
    print("    --- 三道天花板（文档 01 §2；都假设内存子系统「理想」）---")
    lanes = SP.FP32_LANES_PER_SM[H100]
    peak = SP.FP32_VECTOR_TFLOPS_DENSE[H100]
    # 发射槽：每 SM 4 个 scheduler；每个 FMA 要 2 条 LDG + 1 条 FFMA
    slots = SP.ISSUE_SLOTS_PER_CLK_PER_SM
    warps_full = lanes / SP.WARP_SIZE                     # 满速所需的 warp-FFMA/周期
    warps_issue = slots / 3.0                             # 发射槽允许的 warp-FFMA/周期
    cap_issue = warps_issue / warps_full
    # L1/SMEM 交付率：一次 FMA 要 2 个操作数（8 字节）
    cap_l1 = SP.SMEM_DELIVERY_BYTES_PER_CLK / (lanes * 2 * SP.FLOAT32_BYTES)
    # HBM
    cap_hbm = min(1.0, SP.HBM_BW_GBps_BY_GEN[H100] * i1_for_fp32 / 1e3 / peak) \
        if (i1_for_fp32 := 1.0 / SP.FLOAT32_BYTES) else 0.0
    print(f"    {'约束':<22}{'算式':<44}{'上限':>9}")
    print(f"    {'发射槽':<22}{f'{slots} 槽 ÷ 3 条指令 = {warps_issue:.2f} warp-FFMA/周期，满速 {warps_full:.0f}':<44}"
          f"{cap_issue*100:>8.1f}%")
    print(f"    {'L1 / SMEM 交付率':<22}"
          f"{f'{lanes}×8 B = {lanes*8} B/周期，而交付 {SP.SMEM_DELIVERY_BYTES_PER_CLK} B/周期':<44}"
          f"{cap_l1*100:>8.1f}%")
    print(f"    {'HBM':<22}"
          f"{f'I = 1/esize = {i1_for_fp32}, 3350 GB/s × I':<44}{cap_hbm*100:>8.2f}%")
    check("发射槽上限 ≈ 33.3%", close(cap_issue, 1.0 / 3.0, 1e-9),
          f"{cap_issue*100:.1f}%")
    check("L1 交付率上限 = 12.5%", close(cap_l1, 0.125, 1e-9), f"{cap_l1*100:.1f}%")
    check("HBM 上限 = 1.25%（三道里最紧的）", close(cap_hbm, 0.0125, 1e-9),
          f"{cap_hbm*100:.2f}%")
    check("三道天花板依次收紧：发射槽 > L1 > HBM",
          cap_issue > cap_l1 > cap_hbm)


BK_TILE = 32
# 与 02 号内核的 BK 一致（文档 02 §1 表格里 128×128×32 的那一档）


TILING_CONFIGS = (
    # (BM, BN, BK)
    (32, 32, 32),
    (64, 64, 16),
    (128, 128, 16),
    (128, 128, 32),
    (256, 128, 32),
)
# 与文档 02 §1 的表格逐行对应（FP32）


def b3_tiling_table():
    sub("B3  分块参数表（FP32，esize=4）")
    print("    smem = (BM·BK + BK·BN)·esize")
    print("    block 级 AI（**L2 零命中口径**）= 2·BM·BN·BK / smem = 2·BM·BN/((BM+BN)·esize)")
    print("      ↑ 注意 BK 被约掉了：这套口径下 AI 只由 BM、BN 与 esize 决定")
    print()
    print(f"    {'BM×BN×BK':<14}{'smem':>10}{'块级 AI':>10}{'≤228 KB':>10}{'泄漏 L2 后可达':>18}")
    rows = []
    bw = SP.HBM_BW_GBps_BY_GEN[H100]
    peak = SP.FP32_VECTOR_TFLOPS_DENSE[H100]
    for BM, BN, BK in TILING_CONFIGS:
        smem = (BM * BK + BK * BN) * SP.FLOAT32_BYTES
        ai = 2.0 * BM * BN * BK / smem
        ok = smem <= SP.SMEM_PER_SM_MAX_KB * 1024
        reach = min(peak, bw * ai / 1e3)
        rows.append((BM, BN, BK, smem, ai, ok))
        print(f"    {f'{BM}×{BN}×{BK}':<14}{kb(smem):>8.1f}KB{ai:>10.1f}"
              f"{('是' if ok else '否'):>10}{reach:>15.1f} TF")
    print()
    expect = {  # 与文档 02 §1 表格逐行核对
        (32, 32, 32): (8.0, 8.0),
        (64, 64, 16): (8.0, 16.0),
        (128, 128, 16): (16.0, 32.0),
        (128, 128, 32): (32.0, 32.0),
        (256, 128, 32): (48.0, 42.6667),
    }
    for BM, BN, BK, smem, ai, ok in rows:
        e_kb, e_ai = expect[(BM, BN, BK)]
        check(f"{BM}×{BN}×{BK}: smem = {e_kb} KB", close(kb(smem), e_kb, 1e-6),
              f"{kb(smem):.2f} KB")
        check(f"{BM}×{BN}×{BK}: 块级 AI = {e_ai:.4f}", close(ai, e_ai, 1e-4), f"{ai:.4f}")
        check(f"{BM}×{BN}×{BK}: SMEM ≤ 228 KB → 可行", ok,
              f"{kb(smem):.1f} KB")
    print()
    print("    关键读数：128×128×16 与 128×128×32 的块级 AI 完全相同（都是 32.0）。")
    print("    BK 不改变这套口径下的 AI——它只改变 SMEM 占用与 K-loop 迭代数。")


def b4_traffic_reduction():
    sub("B4  分块把 HBM 读流量降下来的倍数")
    print("    朴素（L1/L2 零命中）读字节 = 2·M·N·K·esize")
    print("    分块后读字节 = (M/BM)(N/BN)·(BM·K + K·BN)·esize = M·N·K·(1/BM + 1/BN)·esize")
    print("    比值 = 2·BM·BN/(BM+BN)   ← **它就是块级 AI × esize**")
    print()
    M = N = K = 4096
    esize = SP.FLOAT32_BYTES
    naive = 2.0 * M * N * K * esize
    print(f"    形状 {M}×{N}×{K} FP32：朴素读 {naive/1e9:.3f} GB")
    for BM, BN, _ in ((32, 32, 0), (64, 64, 0), (128, 128, 0), (256, 128, 0)):
        tiled = M * N * K * (1.0 / BM + 1.0 / BN) * esize
        ratio = naive / tiled
        ai = 2.0 * BM * BN / ((BM + BN) * esize)
        print(f"      BM×BN = {BM}×{BN}: 读 {tiled/1e9:.3f} GB，降低 {ratio:.1f} 倍"
              f"（= 块级 AI {ai:.1f} × esize {esize}）")
        check(f"{BM}×{BN}: 降幅 = 块级 AI × esize", close(ratio, ai * esize, 1e-9))
    print()
    print("    ⚠ 这是**零缓存**口径。真实 GEMM 的整题级读字节是 (MK+KN)·esize，")
    print("      与 BM/BN 无关——tiling 不减少总字节，它减少的是**重复搬运的次数**。")
    print("      两者不可混谈：分块不改变 roofline 的 I，它提高的是**可达**的那一侧。")


THREAD_TILE_CONFIGS = (
    # (BM, BN, TM, TN)
    (128, 128, 8, 8),
    (128, 128, 8, 4),
    (256, 128, 8, 8),
)
# 与文档 03 §1 的表格逐行对应（FP32）


def regs_est(tm, tn):
    """每线程寄存器的**公式估算**（不含寻址与索引寄存器）。"""
    return tm * tn + (tm + tn)


def round8(r):
    g = SP.REGISTER_ALLOC_GRANULARITY
    return int(math.ceil(r / g) * g)


def occupancy(threads, regs_per_thread, smem_bytes):
    """返回 (block 数, 常驻线程数, warp 数, 占用率)。约束取最小。"""
    r = round8(regs_per_thread)
    by_regs = SP.REGISTERS_PER_SM // (r * threads)
    by_smem = int((SP.SMEM_PER_SM_MAX_KB * 1024) // smem_bytes) if smem_bytes else 10 ** 6
    by_threads = SP.MAX_THREADS_PER_SM // threads
    n_blocks = max(0, min(by_regs, by_smem, by_threads, SP.MAX_BLOCKS_PER_SM))
    t = n_blocks * threads
    return n_blocks, t, t // SP.WARP_SIZE, t / SP.MAX_THREADS_PER_SM


def b5_thread_tile_and_registers():
    sub("B5  线程级寄存器分块：线程数、寄存器估算、占用率（FP32）")
    print("    线程数 = (BM/TM)·(BN/TN)")
    print("    每线程寄存器估算 = TM·TN + (TM+TN)   ← **不含**寻址与索引寄存器")
    print("      真实编译时还要高 30~60%（寻址、循环变量、SMEM 指针、谓词）")
    print()
    print(f"    {'BM×BN / TM×TN':<18}{'线程':>6}{'C 个数':>8}{'公式估算':>10}"
          f"{'上取整8':>9}{'blocks/SM':>11}{'常驻线程':>10}{'占用率':>9}")
    expected = {
        (128, 128, 8, 8): (256, 80, 64),
        (128, 128, 8, 4): (512, 44, 32),
        (256, 128, 8, 8): (512, 80, 64),
    }
    for BM, BN, TM, TN in THREAD_TILE_CONFIGS:
        threads = (BM // TM) * (BN // TN)
        est = regs_est(TM, TN)
        c_regs = TM * TN
        # 用该配置的 smem（BK=32, FP32）算占用率
        smem = (BM * 32 + 32 * BN) * SP.FLOAT32_BYTES
        nb, nt, nw, occ = occupancy(threads, est, smem)
        tag = f"{BM}×{BN}/{TM}×{TN}"
        print(f"    {tag:<18}{threads:>6}{c_regs:>8}{est:>10}{round8(est):>9}"
              f"{nb:>11}{nt:>10}{occ*100:>8.1f}%")
        e_threads, e_est, e_c = expected[(BM, BN, TM, TN)]
        check(f"{tag}: 线程数 = {e_threads}", threads == e_threads, f"{threads}")
        check(f"{tag}: 公式估算 = {e_est}（C 占 {e_c}）", est == e_est, f"{est}")
    print()
    print("    加上 30~60% 的真实开销后（按 +50% 估）：")
    for BM, BN, TM, TN in THREAD_TILE_CONFIGS:
        threads = (BM // TM) * (BN // TN)
        est = regs_est(TM, TN)
        real = int(round(est * 1.5))
        smem = (BM * 32 + 32 * BN) * SP.FLOAT32_BYTES
        nb, nt, nw, occ = occupancy(threads, real, smem)
        print(f"      {f'{BM}×{BN}/{TM}×{TN}':<18} 估算 {est} → 实际 ~{real} "
              f"→ {nb} blocks/SM，{nw} warp，占用率 {occ*100:.1f}%")
    print()
    print("    ⚠ 占用率对 tile 大小**不是单调的**：8×8 的线程块更小、能多塞几个 block，")
    print("      但每个 block 的线程数也少；8×4 反而拿到了最高的常驻线程数。")


def b6_split_k():
    sub("B6  split-K：收益与代价（文档 06 的瘦高场景）")
    M = N = 256
    K = 65536
    BM = BN = 128
    esize = SP.FLOAT32_BYTES
    tiles = (M // BM) * (N // BN)
    flops = 2.0 * M * N * K
    bw = SP.HBM_BW_GBps_BY_GEN[H100]
    sm = SP.SM_COUNT_BY_GEN[H100]
    print(f"    场景：M=N={M}，K={K}（瘦高），BM=BN={BM}，FP32")
    print(f"    FLOPs = 2MNK = {flops:.4e}")
    print(f"    未切分时网格 = (M/BM)(N/BN) = {tiles} 个 block，"
          f"机器有 {sm} 个 SM → 只有 {tiles}/{sm} = {tiles/sm*100:.2f}% 的 SM 有活干")
    print(f"    每 SM 的 FP32 速率 = {PER_SM_FP32_FLOPS:.4e} FLOP/s（推导值）")
    print()
    print(f"    {'S':>5}{'blocks':>9}{'占用SM':>8}{'计算时间':>12}{'加速比':>9}"
          f"{'归约字节':>11}{'归约时间':>11}{'归约/计算':>11}")
    t_base = flops / (tiles * PER_SM_FP32_FLOPS)
    rows = []
    for S in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
        blocks = tiles * S
        busy = min(blocks, sm)
        t_cmp = flops / (busy * PER_SM_FP32_FLOPS)
        red_bytes = 2 * S * M * N * esize      # atomicAdd 口径：读改写各一次
        red_t = red_bytes / (bw * 1e9)
        rows.append((S, blocks, busy, t_cmp, t_base / t_cmp, red_bytes, red_t,
                     red_t / t_cmp))
        print(f"    {S:>5}{blocks:>9}{busy:>8}{t_cmp*1e6:>10.1f}us"
              f"{t_base/t_cmp:>8.1f}x{red_bytes/1e6:>9.1f}MB"
              f"{red_t*1e6:>9.2f}us{red_t/t_cmp*100:>10.2f}%")
    print()
    check("S=1（未切分）只有个位数个 block",
          1 <= tiles < 10, f"{tiles} 个")
    check("S=1 时 SM 占用率 < 5%", tiles / sm < 0.05, f"{tiles/sm*100:.2f}%")
    check("S=32 的加速比 ≈ 32×", close(rows[5][4], 32.0, 0.02), f"{rows[5][4]:.2f}×")
    check("S 大到 512 时归约开销吃掉一半以上的时间",
          rows[-1][7] > 0.5, f"{rows[-1][7]*100:.1f}%")
    print()
    print("    把「填满机器」与「归约不超过计算时间 10%」两个条件并起来解：")
    s_lo = math.ceil(sm / tiles)
    s_hi = int(0.10 * (flops / (sm * PER_SM_FP32_FLOPS)) * bw * 1e9 / (2 * M * N * esize))
    print(f"      填满机器需要 S ≥ ceil({sm}/{tiles}) = {s_lo}")
    print(f"      归约 ≤ 10% 计算时间需要 S ≤ {s_hi}")
    print(f"      ⟹ 有用的 S 窗口 = [{s_lo}, {s_hi}]，"
          f"宽度只有 {s_hi - s_lo + 1} 档（而且它随 K 变小而收窄）")
    check("有用的 S 窗口非空", s_lo <= s_hi, f"[{s_lo}, {s_hi}]")
    check("有用窗口的上界 = 81（被归约开销卡住，不是被 SM 数卡住）",
          s_hi == 81, f"[{s_lo}, {s_hi}]")
    return s_lo, s_hi


def b8_bank_conflict():
    sub("B8  GEMM 里 SMEM 读的行距与 bank 冲突路数（**按映射规则算出，不是测硬件**）")
    print("    规则：32 个 bank，每个 4 字节宽，地址按 4 字节为单位轮流分配")
    print("          （规则本身正文在 20-gpu-architecture/memory-system/03，此处只做代入）")
    print()
    print("    考察本主题 02 号内核的 B 读：B 以 [n][k] 转置存放，行距 = BK + PAD。")
    print("    warp 内 16 个 tx（同一 tx 的两个 ty 读同一地址 → 广播），")
    print("    地址（以 4 字节字为单位）= (2·tx + j)·(BK + PAD) + k。")
    print()
    print(f"    {'行距':>6}{'PAD':>6}{'一个 warp 的不同地址数':>24}{'冲突路数':>10}{'判定':>14}")
    for pad in (0, 1, 2, 4, 8):
        stride = BK_TILE + pad
        ways = 1
        for j in range(2):
            per_bank = [0] * 32
            for tx in range(16):
                per_bank[((2 * tx + j) * stride + 7) % 32] += 1
            ways = max(ways, max(per_bank))
        print(f"    {stride:>6}{pad:>6}{16:>24}{ways:>10}"
              f"{('无冲突' if ways == 1 else f'{ways} 遍'):>14}")
        if pad == 0:
            check("行距 = BK（32）时 16 个地址全挤在同一个 bank → 16 路冲突", ways == 16,
                  f"{ways} 路")
        if pad == 1:
            check("行距 = BK+1（33）时 16 个地址落在 16 个不同 bank → 无冲突", ways == 1)
        if pad == 2:
            check("偶数行距的冲突路数 = PAD 本身（PAD=2 → 2 路）", ways == 2, f"{ways} 路")
        if pad == 8:
            check("PAD=8 时是 8 路，不是 2 路", ways == 8, f"{ways} 路")
    print()
    print("    ⚠ 三条必须一起说的：")
    print("      · A 不转置时读的是**广播**（同一 warp 内地址相同），本来就不冲突，")
    print("        给它 padding 是白花的（本主题 02 号内核因此只给 B 加 padding）。")
    print("      · PAD=1 完全消除冲突，但行距 33 个 float 破坏了 16 字节对齐，")
    print("        用不了 LDS.128。保持对齐又要打散 bank 要用 **swizzle**（XOR 行号到列地址）。")
    print("      · 冲突的代价是「串行遍数」：16 路冲突让这条 LDS 的交付率降到 1/16。")


def b9_epilogue_cost():
    sub("B9  epilogue：beta ≠ 0 的读改写代价（文档 02 §3）")
    n = 4096
    esize = SP.FLOAT32_BYTES
    flops = 2.0 * n ** 3
    bytes_beta0 = (n * n * 3) * esize                # 理想：A、B、C 各过一遍
    extra_read = n * n * esize                       # beta·C 的那一次读
    print(f"    方阵 n={n}，FP32。理想总字节 = (MK+KN+MN)·4 = {bytes_beta0/1e6:.1f} MB")
    print(f"    其中 C 的写 = MN·esize = {n*n*esize/1e6:.1f} MB")
    print(f"    beta ≠ 0 时多一次 C 的读 = {extra_read/1e6:.1f} MB "
          f"= 理想总字节的 {extra_read/bytes_beta0*100:.2f}%")
    print()
    print("    alpha/beta 的取舍：")
    print("      · beta = 0（覆盖写）→ 省掉整次读，但调用者必须保证 C 的内容可以丢；")
    print("      · beta ≠ 0（累加）→ 多 33% 的 HBM 流量，换来「不重置累加器」这个语义；")
    print("      · 两者的算术都在**寄存器**里做（alpha·acc + beta·C 在 epilogue 一次完成），")
    print("        所以 alpha 的乘法是免费的，beta 的读不是——**代价全在那一次读上**。")
    check("beta≠0 的额外读 = 理想总字节的 1/3（方阵）",
          close(extra_read / bytes_beta0, 1.0 / 3.0, 1e-9),
          f"{extra_read/bytes_beta0*100:.2f}%")
    print()
    print("    ⚠ 这个 33% 只在「C 必须从 HBM 读」时成立。若 C 的分块能被 L2 兜住")
    print("      （小 M/N、或恰好复用），额外读会被缓存吃掉，代价随之下降。")
    print("      这与 §B3 的块级 AI 是同一类口径问题：先问「这字节真的走 HBM 吗」。")


def b11_pipeline_timing():
    sub("B11 多级流水的 SMEM 预算与每段时间（文档 04 §1 第三、四步）")
    peak = SP.TC_TFLOPS_DENSE_SPARSE["FP16"][H100][0]
    bw = SP.HBM_BW_GBps_BY_GEN[H100]
    sm = SP.SM_COUNT_BY_GEN[H100]
    per_sm_flops = peak * 1e12 / sm
    per_sm_bw = bw * 1e9 / sm
    BM = BN = 128
    esize = SP.FLOAT16_BYTES
    print(f"    每 SM 分到的 FP16 算力 = {peak}e12 / {sm} = {per_sm_flops:.4e} FLOP/s")
    print(f"    每 SM 分到的带宽       = {bw}e9 / {sm} = {per_sm_bw:.4e} B/s")
    print()
    print(f"    {'BK':>5}{'每段字节':>12}{'228KB 允许级数':>16}{'t_compute':>12}"
          f"{'t_memory':>11}{'t_mem/t_cmp':>13}{'AI/I*':>9}")
    for BK in (32, 64, 128):
        stage = (BM * BK + BK * BN) * esize
        stages = int(SP.SMEM_PER_SM_MAX_KB * 1024 // stage)
        t_c = 2.0 * BM * BN * BK / per_sm_flops
        t_m = stage / per_sm_bw
        ai = 2.0 * BM * BN / ((BM + BN) * esize)
        i_star = roofline_ridge(peak, bw)
        print(f"    {BK:>5}{stage:>11}B{stages:>16}{t_c*1e6:>10.2f}us"
              f"{t_m*1e6:>9.2f}us{t_m/t_c:>12.2f}x{ai/i_star:>8.3f}"
              f"   （t_c/t_m = {t_c/t_m:.3f}）")
        if BK == 64:
            check("BK=64：每段 32 KB，228 KB 允许 7 级",
                  stage == 32768 and stages == 7, f"{stage} B, {stages} 级")
            check("BK=64：t_compute ≈ 0.28 us", close(t_c * 1e6, 0.28, 0.02),
                  f"{t_c*1e6:.2f} us")
            check("BK=64：t_memory ≈ 1.29 us", close(t_m * 1e6, 1.29, 0.02),
                  f"{t_m*1e6:.2f} us")
            check("BK=64：搬是算的 4.6 倍（1.29/0.28）", close(t_m / t_c, 4.6, 0.05),
                  f"{t_m/t_c:.2f}x")
            check("t_compute/t_memory = AI_block/I*（两条算法互为校验；注意方向）",
                  close(t_c / t_m, ai / i_star, 1e-6),
                  f"{t_c/t_m:.4f} vs {ai/i_star:.4f}")
            check("反过来写（t_memory/t_compute = AI_block/I*）是错的",
                  not close(t_m / t_c, ai / i_star, 0.01),
                  f"{t_m/t_c:.4f} vs {ai/i_star:.4f}")
    print()
    print("    ⚠ 级数 × 每段字节 = SMEM 总量，是个常数——BK 大 + 级数少 与")
    print("      BK 小 + 级数多 是在同一个预算里选点，不是两个独立旋钮。")
    print("    ⚠ 流水线只能把「搬」与「算」重叠，不能把 4.6 倍的关系变成 1。")
    print("      重叠之后时间由较长的那一侧（t_memory）决定。")


def b10_persistent_and_tmem():
    sub("B10 persistent kernel 与 TMEM（文档 05）")
    peak = SP.TC_TFLOPS_DENSE_SPARSE["FP16"][B200][0]
    bw = SP.HBM_BW_GBps_BY_GEN[B200]
    sm = SP.SM_COUNT_BY_GEN[B200]
    i_b200 = roofline_ridge(peak, bw)
    print(f"    B200 SXM：{sm} SM，FP16 dense {peak} TFLOPS，HBM {bw} GB/s")
    print(f"    I*(B200, FP16) = {peak}e12 ÷ {bw}e9 = {i_b200:.4f} FLOP/byte"
          f"（对照 H100 的 295.4）")
    check("I*(B200, FP16) ≈ 292.2", close(i_b200, 292.2, 1e-3), f"{i_b200:.4f}")
    print()
    print("    --- 网格 1024 个 tile 时的波次量化（静态分牌）---")
    n, bm = 4096, 128
    tiles = (n // bm) ** 2
    flops = 2.0 * n ** 3
    t_ideal = flops / (peak * 1e12)
    print(f"    {n}³ FP16，BM=BN={bm} → {tiles} 个 tile；理想时间 = "
          f"{flops:.4e} / {peak}e12 = {t_ideal*1e6:.1f} us")
    print(f"    {'blocks/SM':>10}{'槽位':>8}{'波数':>6}{'最后一波占用':>14}"
          f"{'总时间损失':>12}{'实际时间':>12}")
    for bps in (1, 2, 4):
        slots = sm * bps
        waves = (tiles + slots - 1) // slots
        last = tiles - (waves - 1) * slots
        util = last / slots
        loss = (1.0 - util) / waves
        print(f"    {bps:>10}{slots:>8}{waves:>6}{util*100:>13.1f}%"
              f"{loss*100:>11.2f}%{t_ideal/(1-loss)*1e6:>11.1f}us")
        if bps == 1:
            check("blocks/SM = 1 时波次量化的损失 < 2%", loss < 0.02,
                  f"{loss*100:.2f}%")
        if bps == 2:
            check("blocks/SM = 2 的损失反而更大（槽位变多 → 波数变少 → 尾波占比变高）",
                  loss > 0.10, f"{loss*100:.2f}%")
    print()
    print("    ⚠ 这条反直觉：**每个 SM 能放更多 block 会让波次量化更严重**，")
    print("      因为「槽位数」相对 tile 数变大了，最后一波填不满的比例变高。")
    print("      persistent 发牌把尾部压到「每槽位最多 1 个 tile」的量级，")
    print(f"      即总时间的 1/{tiles} = {1.0/tiles*100:.4f}%——但代价见下。")
    print()
    print("    --- 启动开销（**引用值，本仓库不复核其测量口径**）---")
    lo, hi = SP.KERNEL_LAUNCH_OVERHEAD_US_EXTERNAL
    print(f"    一次 kernel 启动 ≈ {lo}~{hi} us（外部测量，非 NVIDIA 官方规格）")
    print(f"    理想时间 {t_ideal*1e6:.1f} us 下它占 "
          f"{lo/t_ideal/1e6*100:.1f}%~{hi/t_ideal/1e6*100:.1f}%")
    print("    ⚠ 但它**只付一次，与网格大小无关**——persistent 与否都要付。")
    print("      persistent 省掉的是**每个 block 的启动项**（SMEM 分配、流水线填充的")
    print("      首次延迟、`__grid_constant__` 参数装载），这些在 K 大时占比很小，")
    print("      在 K 小、块数多时才显著。**不要把启动开销当成 persistent 的主要理由。**")
    print()
    print("    --- TMEM 容量约束（推导值）---")
    tmem_bytes = SP.TMEM_LANES_PER_CTA * SP.TMEM_MAX_COLUMNS_PER_CTA * (SP.TMEM_CELL_BITS // 8)
    print(f"    TMEM = {SP.TMEM_LANES_PER_CTA} lane × {SP.TMEM_MAX_COLUMNS_PER_CTA} 列 × "
          f"32 bit = {tmem_bytes/1024:.0f} KB，**per CTA 口径**")
    print(f"    （数值恰好等于每 SM 的寄存器文件 256 KB，但它是**另一份独立的存储**）")
    print(f"    分配粒度 {SP.TMEM_ALLOC_GRANULARITY_COLUMNS} 列；合法取值 "
          f"{SP.TMEM_ALLOC_LEGAL_COLUMNS}")
    print(f"    {'累加器形状':>14}{'占列数':>8}{'512 列能容纳几个':>18}")
    for m, ncol in ((128, 64), (128, 128), (128, 256), (128, 512)):
        cols = ((ncol + 31) // 32) * 32
        print(f"    {f'{m}×{ncol}':>14}{cols:>8}{SP.TMEM_MAX_COLUMNS_PER_CTA//cols:>18}")
    check("128×128 的 fp32 累加器占 128 列，每 CTA 最多 4 个",
          SP.TMEM_MAX_COLUMNS_PER_CTA // 128 == 4)
    check("128×512 的累加器占满 512 列 → 每 CTA 只能有 1 个",
          SP.TMEM_MAX_COLUMNS_PER_CTA // 512 == 1)
    print()
    print("    ⚠ 这张表是 **per CTA** 的：TMEM 允许一个 CTA 同时持有多少个累加器。")
    print("      一个 SM 上多个 CTA 如何**仲裁**这 512 列，本次采用的来源没有说明——")
    print("      正文只做 per CTA 的算术，不把它当成 per SM 的硬上限。")


def b7_ridge_tile_size(i_fp16, i_fp32):
    sub("B7  想靠「一个 block 自己就到达 roofline 拐点」，tile 要多大")
    print("    块级 AI = BM·BN/((BM+BN)·esize)，方阵 BM=BN 时 = BM/esize")
    print("    令块级 AI = I* ⟹ BM = I*·esize")
    print()
    print(f"    {'精度':<8}{'I*':>8}{'esize':>7}{'临界 BM':>10}{'累加器':>12}"
          f"{'占寄存器文件':>14}{'可行':>7}")
    row = {}
    for name, i_star, esize, expect_ok in (("FP32", i_fp32, SP.FLOAT32_BYTES, True),
                                           ("FP16", i_fp16, SP.FLOAT16_BYTES, False)):
        bm = i_star * esize
        acc = bm * bm * SP.FLOAT32_BYTES     # fp32 累加器
        frac = acc / (SP.REGISTER_FILE_PER_SM_KB * 1024)
        ok = frac <= 1.0
        row[name] = (bm, acc, frac)
        print(f"    {name:<8}{i_star:>8.1f}{esize:>7}{bm:>10.1f}{acc/1024:>10.1f}KB"
              f"{frac:>13.2f}x{('是' if ok else '否'):>7}")
        check(f"{name}：临界 tile 的 fp32 累加器"
              f"{'装得下' if expect_ok else '装不下'} 256 KB 寄存器文件（公式判定 = "
              f"{'可行' if ok else '不可行'}）",
              ok == expect_ok, f"{acc/1024:.1f} KB = {frac:.2f}×")
    print()
    bm32 = i_fp32 * SP.FLOAT32_BYTES
    bm16 = i_fp16 * SP.FLOAT16_BYTES
    acc32 = bm32 * bm32 * 4
    acc16 = bm16 * bm16 * 4
    print(f"    ⟹ FP32 的临界 tile 是 {bm32:.0f}×{bm32:.0f}，累加器 {acc32/1024:.1f} KB"
          f"（{acc32/(SP.REGISTER_FILE_PER_SM_KB*1024):.2f}× 寄存器文件）—— 可行；")
    print(f"      FP16 的临界 tile 是 {bm16:.0f}×{bm16:.0f}，累加器 {acc16/1024/1024:.2f} MB"
          f"（{acc16/(SP.REGISTER_FILE_PER_SM_KB*1024):.2f}× 寄存器文件）—— 不可行。")
    print()
    print("    ⚠ 这条是 04 篇的支点：**精度越快，拐点越靠右（I* 越大）。**")
    print(f"      I* 涨 {i_fp16/i_fp32:.1f} 倍、esize 减半 ⟹ 临界 BM 涨 "
          f"{row['FP16'][0]/row['FP32'][0]:.1f} 倍（{row['FP32'][0]:.0f} → "
          f"{row['FP16'][0]:.0f}），")
    print(f"      而累加器按 BM² 涨 ⟹ 涨 {row['FP16'][1]/row['FP32'][1]:.1f} 倍"
          f"（{row['FP32'][1]/1024:.0f} KB → {row['FP16'][1]/1024/1024:.2f} MB）。")
    print("      所以 FP16/FP8 的 GEMM **不可能靠单块 tiling 到达拐点**，")
    print("      只能靠 L2 在 tile 之间提供复用（也就是整题级 AI 那一套）。")


# ===========================================================================
# §C 边界与反例
# ===========================================================================


def c1_register_spill():
    sub("C1  TM=TN=16 会 spill，TM=TN=8 不会（**公式判定，不是测硬件**）")
    print(f"    判定式：TM·TN + (TM+TN) > {SP.MAX_REGISTERS_PER_THREAD} ⟹ 编译器必须 spill")
    print()
    for TM, TN in ((8, 8), (8, 4), (16, 16), (16, 8)):
        est = regs_est(TM, TN)
        c_regs = TM * TN
        # 光累加器就超限的情况
        acc_alone_over = c_regs > SP.MAX_REGISTERS_PER_THREAD
        over = est > SP.MAX_REGISTERS_PER_THREAD
        print(f"    TM×TN = {TM}×{TN}: C 占 {c_regs}，公式估算 {est}"
              f"{'  ← 光 C 就超过 255' if acc_alone_over else ''}"
              f"{'  → 判定 spill' if over else '  → 判定不 spill'}")
        if (TM, TN) == (8, 8):
            check("TM=TN=8 的估算约 80，且 < 255 → **不会 spill**",
                  close(float(est), 80.0) and not over, f"{est}")
            check("TM=TN=8 正是经典 GEMM 教程推荐的配置，不能拿它当 spill 反例", est < 255)
        if (TM, TN) == (16, 16):
            check("TM=TN=16 的估算 = 288 > 255 → **会 spill**", over and est == 288, f"{est}")
            check("TM=TN=16 的光 C 就 256 > 255（与寻址寄存器无关，必然 spill）",
                  acc_alone_over, f"C = {c_regs}")
        if (TM, TN) == (8, 4):
            check("TM=TN=8 与 8×4 都远低于上限（8×4 更小）", est < 255, f"{est}")
    print()
    print("    spill 的机制与后果回链 20-gpu-architecture/execution-model/01：")
    print("    溢出的活跃变量落到 local memory，物理上在片外，每次 spill 是一次真实的")
    print(f"    全局访存——延迟比寄存器高两个数量级（HBM 往返 "
          f"{SP.HBM_LATENCY_CYCLES_RANGE[0]}~{SP.HBM_LATENCY_CYCLES_RANGE[1]} 周期）。")
    print()
    print("    16×16 的线程分块还有第二道墙：线程数 = (BM/16)·(BN/16)。")
    for BM, BN in ((128, 128), (256, 256)):
        print(f"      block {BM}×{BN} → 线程数 {(BM//16)*(BN//16)}"
              f"（{((BM//16)*(BN//16))//SP.WARP_SIZE} 个 warp）")


def c2_smem_over_limit():
    sub("C2  BK 过大使 SMEM 超过 228 KB → 判定为不可行")
    limit = SP.SMEM_PER_SM_MAX_KB * 1024
    print(f"    smem = (BM·BK + BK·BN)·esize，上限 = {SP.SMEM_PER_SM_MAX_KB} KB")
    print()
    cases = (
        (128, 128, 3, 224, SP.FLOAT32_BYTES, True),
        (128, 128, 4, 232, SP.FLOAT32_BYTES, False),
        (128, 128, 4, 256, SP.FLOAT32_BYTES, False),
        (256, 128, 4, 144, SP.FLOAT32_BYTES, True),
        (256, 128, 4, 160, SP.FLOAT32_BYTES, False),
        (512, 512, 2, 64, SP.FLOAT16_BYTES, True),
        (512, 512, 2, 128, SP.FLOAT16_BYTES, False),
    )
    for BM, BN, esize, BK, esz, expect_ok in cases:
        smem = (BM * BK + BK * BN) * esz
        ok = smem <= limit
        print(f"    {BM}×{BN}×{BK}，esize={esz}: smem = {smem} B = {kb(smem):.1f} KB "
              f"→ {'可行' if ok else '**不可行**（超上限）'}")
        check(f"{BM}×{BN}×{BK} esize={esz}: 判定 = {'可行' if expect_ok else '不可行'}",
              ok == expect_ok, f"{kb(smem):.1f} KB")
    print()
    BM = BN = 128
    esz = SP.FLOAT32_BYTES
    max_bk = limit // ((BM + BN) * esz)
    print(f"    反解：BM=BN=128、FP32 时 BK 的最大可行值 = "
          f"{limit} ÷ (({BM}+{BN})×{esz}) = {max_bk}（取 8 的倍数 → "
          f"{(max_bk//8)*8}）")
    check("BK=224 可行而 BK=232 不可行（边界就在 228 上）",
          (128 * 224 + 224 * 128) * 4 <= limit and (128 * 232 + 232 * 128) * 4 > limit)
    print()
    print("    ⚠ 注意门槛不是「SMEM 不够」这一件事：tile 变大时**同时**会撞上")
    print("      寄存器（累加器 BM·BN·4 B）与 32 blocks/SM 上限。反过来 BK 变小则")
    print("      K-loop 迭代数 = K/BK 上涨，每轮的同步与地址计算开销占比升高。")


def c3_reduction_nondeterminism():
    sub("C3  归约顺序不确定 → split-K 结果逐位不可复现")
    rng = np.random.default_rng(7)

    def reduce_with(parts, order):
        acc = np.zeros(parts.shape[1], dtype=np.float32)
        for idx in order:
            acc = (acc + parts[idx]).astype(np.float32)
        return acc

    # (i) 真实量级：32 段正态分布的部分和
    S = 32
    parts = (rng.standard_normal((S, 4096)) * 45.0).astype(np.float32)
    ref = parts.astype(np.float64).sum(axis=0)
    fwd = reduce_with(parts, list(range(S)))
    rev = reduce_with(parts, list(range(S - 1, -1, -1)))
    rnd = reduce_with(parts, [int(i) for i in rng.permutation(S)])
    print(f"    (i) {S} 段、每段 4096 个 fp32 部分和，量级相当（各段 ~45，总和 ~255）：")
    print(f"        正序 vs 参考 最大相对差 = "
          f"{float(np.max(np.abs(fwd - ref)/np.maximum(np.abs(ref),1e-30))):.3e}")
    print(f"        逆序 vs 正序 的位模式相同？ {np.array_equal(fwd, rev)}")
    print(f"        随机序 vs 正序 的位模式相同？ {np.array_equal(fwd, rnd)}")
    print(f"        逆序 vs 正序 的不同的元素个数 = "
          f"{int(np.count_nonzero(fwd != rev))}/{fwd.size}")
    check("相同量级的 32 段，换序后位模式已经不同",
          not (np.array_equal(fwd, rev) and np.array_equal(fwd, rnd)))
    print()
    # (ii) 量级悬殊：差异会被放大到肉眼可见
    #     1e8 在 fp32 下的 ulp = 8.0，所以 1.0 加上去会被整个吞掉。
    big = (rng.standard_normal((128, 4)) * 1.0).astype(np.float32)
    big[0] = 1.0e8
    big[1:] = 1.0
    a1 = reduce_with(big, list(range(128)))              # 先加 1e8
    a2 = reduce_with(big, list(range(1, 128)) + [0])     # 先加 127 个 1.0
    print("    (ii) 第 0 段 = 1e8，其余 127 段 = 1.0（fp32 在 1e8 处的 ulp = 8.0）：")
    print(f"        先加 1e8 再加小量：{[f'{v:.1f}' for v in a1]}")
    print(f"        先加小量再加 1e8：{[f'{v:.1f}' for v in a2]}")
    print(f"        两者之差的最大绝对值 = {float(np.max(np.abs(a1 - a2))):.6e}")
    check("量级悬殊时，换序的差值肉眼可见（不是「只差最后一位」）",
          float(np.max(np.abs(a1 - a2))) > 1.0)
    print()
    print("    ⚠ 这条是 06 篇的核心：`atomicAdd` 的**执行顺序由硬件决定**，")
    print("      所以「同一个输入、同一份代码、同一个 GPU」两次运行会给出不同的 bit。")
    print("      上面的构造完全在 numpy 里做，它演示的是**浮点加法不满足结合律**，")
    print("      而不是 GPU 上跑出来的——本仓库没有跑过。")


def c4_sm_idle():
    sub("C4  瘦高场景的 SM 空闲（文档 06 §0 的对照场景）")
    M = N = 256
    K = 65536
    sm = SP.SM_COUNT_BY_GEN[H100]
    for BM, BN in ((128, 128), (64, 64), (32, 32)):
        tiles = (M // BM) * (N // BN)
        idle = sm - tiles
        print(f"    BM=BN={BM}: block 数 = {tiles}，空闲 SM = {idle}，"
              f"利用率 = {tiles/sm*100:.2f}%")
        if (BM, BN) == (128, 128):
            check("BM=BN=128 时只有个位数个 block", tiles < 10, f"{tiles}")
            check("空闲 SM 数 > 120", idle > 120, f"{idle}")
    print()
    print("    ⚠ 而算力需求一点没变：FLOPs = 2·256·256·65536 = "
          f"{2*256*256*65536:.4e}，与 M=N=4096,K=256 的方阵题完全相同，")
    print("      但后者有 (4096/128)² = 1024 个 block 去分。")
    print()
    print("    减小 tile 也不行——它把每个 block 的活变少，但块级 AI 同步下降：")
    for BM in (32, 64, 128, 256):
        ai = BM / SP.FLOAT32_BYTES
        print(f"      BM=BN={BM}: 块级 AI = {ai:.1f}，"
              f"{'越过 I*=20' if ai > 20 else '**低于 I*=20**，越切越memory-bound'}")


def c5_l1_balance_point():
    sub("C5  8×8 正是「L1 交付率」与「FP32 lane 需求」的平衡点")
    lanes = SP.FP32_LANES_PER_SM[H100]
    need_per_fma_bytes = 2 * SP.FLOAT32_BYTES      # 一次 FMA 要从 SMEM 取两个操作数
    need = lanes * need_per_fma_bytes              # 若每次都从 SMEM 取
    delivered = SP.SMEM_DELIVERY_BYTES_PER_CLK
    print(f"    每 SM 的 FP32 lane 数 = {lanes}，一次 FMA 需要 {need_per_fma_bytes} B 操作数")
    print(f"    ⟹ 若每个操作数都从 SMEM 取，需要 {lanes}×{need_per_fma_bytes} "
          f"= {need} B/周期")
    print(f"    而共享内存无冲突时只能交付 {delivered} B/周期")
    print(f"    ⟹ 缺口 = {need/delivered:.0f} 倍。**寄存器复用不是优化，是必需品。**")
    check("L1/SMEM 交付率与 lane 需求差 8 倍", close(need / delivered, 8.0), "8×")
    print()
    print("    线程级寄存器分块 TM×TN 让每个操作数被用 TM（或 TN）次：")
    print("      每线程每 k 步：TM·TN 次 FMA，TM+TN 次 SMEM 读")
    print("      每个 FMA 的 SMEM 字节 = (TM+TN)/(TM·TN) × esize")
    print(f"    {'TM×TN':>8}{'每 FMA 的 SMEM 字节':>22}{'全集需求 B/周期':>18}{'交付':>8}{'判定':>10}")
    for TM, TN in ((8, 8), (8, 4), (16, 16), (4, 4)):
        per_fma = (TM + TN) / (TM * TN) * SP.FLOAT32_BYTES
        total = lanes * per_fma
        ok = total <= delivered
        print(f"    {f'{TM}×{TN}':>8}{per_fma:>22.4f}{total:>18.1f}{delivered:>8}"
              f"{('够' if ok else '**不够**'):>10}")
        if (TM, TN) == (8, 8):
            check("TM=TN=8 时需求恰好 = 128 B/周期，与交付率**精确相等**",
                  close(total, float(delivered)), f"{total:.1f} vs {delivered}")
        if (TM, TN) == (4, 4):
            check("TM=TN=4 时需要 256 B/周期 > 128，**L1 带宽不够**", total > delivered,
                  f"{total:.1f} > {delivered}")
    print()
    print("    ⚠ 这解释了为什么 8×8 是经典推荐值：它是**在不超过 255 寄存器约束下**")
    print("      刚好把 L1 交付率用满的最小线程分块。再小（4×4）L1 带宽成为瓶颈，")
    print("      再大（16×16）累加器本身超限。")


def c6_accumulator_wall():
    sub("C6  累加器容量墙：为什么大 tile 走不到底")
    sm = SP.SM_COUNT_BY_GEN[H100]
    rf = SP.REGISTER_FILE_PER_SM_KB * 1024
    print(f"    一个 BM×BN 的 fp32 累加器占 BM·BN·4 字节（跨整个 block 的寄存器）")
    print(f"    每 SM 寄存器文件 = {rf} B")
    print()
    print(f"    {'tile':>12}{'累加器':>12}{'占寄存器文件':>14}{'块内线程数(8×8)':>17}{'判定':>12}")
    for BM, BN in ((32, 32), (64, 64), (128, 128), (256, 128), (256, 256), (591, 591)):
        acc = BM * BN * SP.FLOAT32_BYTES
        frac = acc / rf
        threads = (BM // 8) * (BN // 8) if BM % 8 == 0 and BN % 8 == 0 else None
        tstr = f"{threads}" if threads else "—"
        verdict = "可行" if frac <= 1.0 else "**不可行**"
        print(f"    {f'{BM}×{BN}':>12}{acc/1024:>10.1f}KB{frac:>13.2f}x{tstr:>17}"
              f"{verdict:>12}")
    print()
    check("128×128 的累加器 = 64 KB = 寄存器文件的 1/4（可行）",
          close(128 * 128 * 4 / rf, 0.25, 1e-9))
    check("591×591（FP16 的临界 tile）的累加器 > 寄存器文件 → **不可行**",
          591 * 591 * 4 > rf, f"{591*591*4/1024:.0f} KB = {591*591*4/rf:.2f}×")
    print()
    print("    TMEM 一侧的天花板（回链 20-gpu-architecture/tensor-core/04）：")
    tmem_bytes = SP.TMEM_LANES_PER_CTA * SP.TMEM_MAX_COLUMNS_PER_CTA * (SP.TMEM_CELL_BITS // 8)
    print(f"      TMEM = {SP.TMEM_LANES_PER_CTA} lane × {SP.TMEM_MAX_COLUMNS_PER_CTA} 列 × "
          f"{SP.TMEM_CELL_BITS} bit = {tmem_bytes/1024:.0f} KB（**per CTA 口径**）")
    for n_cols in (128, 256):
        used = n_cols
        n_acc = SP.TMEM_MAX_COLUMNS_PER_CTA // used
        print(f"      M=128、N={n_cols} 的 fp32 累加器占 {used} 列 → "
              f"每 CTA 的 {SP.TMEM_MAX_COLUMNS_PER_CTA} 列最多容纳 {n_acc} 个这样的累加器")
    check("TMEM 每 CTA 512 列，128×128 的累加器只能用 4 个",
          SP.TMEM_MAX_COLUMNS_PER_CTA // 128 == 4)


# ===========================================================================
# 主流程
# ===========================================================================


def main():
    print("=" * 76)
    print("40-kernels/gemm 本机验证模型（verify.py）")
    print("=" * 76)
    print("本程序只跑算术与 numpy，不碰硬件。全部输出是**推导值**。")

    s0_scope()

    section("A", "算法正确性")
    a1_tiled_gemm()
    a2_splitk_decomposition()
    a3_streamk_partition()

    section("B", "代价模型")
    i_fp16, i_fp32 = b1_roofline_ridge()
    b2_naive_ai(i_fp32, i_fp16)
    b3_tiling_table()
    b4_traffic_reduction()
    b5_thread_tile_and_registers()
    b6_split_k()
    b7_ridge_tile_size(i_fp16, i_fp32)
    b8_bank_conflict()
    b9_epilogue_cost()
    b10_persistent_and_tmem()
    b11_pipeline_timing()

    section("C", "边界与反例")
    c1_register_spill()
    c2_smem_over_limit()
    c3_reduction_nondeterminism()
    c4_sm_idle()
    c5_l1_balance_point()
    c6_accumulator_wall()

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

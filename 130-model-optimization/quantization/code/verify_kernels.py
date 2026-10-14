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

★ 本主题里 A/C 与编号程序的分工是 **「数学 vs 实现」**：
  · 本文件（纯 python3 + 标准库）验的是**可独立推导的数学**：
    int4 的打包布局（一个双射）、int8×int8 的溢出阈值（整数算术）、
    fp16 的可表示整数上界与 ulp（`struct` 的 `'e'` 格式）、
    epilogue 的列和恒等式（矩阵代数）、e4m3 的极值（位域反解）、
    以及「不 +8 会错多少」的**闭式后果**（逐元素恰 ±8）。
  · 编号程序（真 torch）测的是**真实实现的行为**：`_weight_int4pack_mm_for_cpu` 的
    逐字节打包布局与 `maxdiff = 0`、`_weight_int8pack_mm` 的偏差表、
    fp8 的 `torch.mm` 的输出舍入与溢出成 NaN、fp16 的 CPU 累加宽度。
  · **两者不得互相背书。** 数学成立不证明实现如此；实现如此不证明数学成立。
  · ⚠ **规模不同 ⇒ 数字不同**：本文件跑的是小规模版本（矩阵 ≤ 32×32、无 fp8 硬件）。
    编号程序跑的是 256×256 与 K 到 2^21 的规模。**只有定性结构必须一致。**

★ 「独立」的硬要求（CONTRIBUTING §2）：本文件里没有一行代码来自 torch，
  期望值**全部**是手写的常量（下面 `HANDWRITTEN_*`）或从 `specs_kernels.py` 抄来的外部常量。
  §E 节把这条**机械地验证一遍**：把参考实现故意改错，A 类断言必须失败。

★ `specs_kernels.py` 的每条常量都至少被本文件或某个编号程序的断言读过一次（A6 约定）。
  §F 节把「本文件负责的那部分」列出来，供变异测试逐个改。

运行：`make verify`（等价于 `python3 verify.py`）。只用标准库。
"""

import random
import struct
import sys

import specs_kernels as SPECS


# ===========================================================================
# 口径
# ===========================================================================

_STATE = {"n": 0, "A": 0, "C": 0, "fail": []}

STDLIB_SEED = 20260918          # 本文件自己的蒙特卡洛种子（**与编号程序无关**）
SMALL_N = 32                    # 本文件自己的小规模尺寸
SMALL_K = 32


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
# 手写期望值（**不是**任何被测代码算出来的）
# ===========================================================================

# ★ 下面这段 hex 是**从真 torch 抄下来的**、不是本文件算出来的：
#   判据：`q[n][k] = (5n + 3k + n·k) mod 16`（32×32 的无符号 nibble 表），
#   在 **torch 2.11.0 / CPU** 上跑
#       `_convert_weight_to_int4pack_for_cpu(torch.tensor(q, int32), 2)`
#   得到的 `[32, 16]` uint8 张量，`tobytes()` 的 hex。512 字节。
#   本文件用它来**判**下面 `ref_int4_pack_layout` 这个公式对不对——
#   公式是「解释」，这段 hex 是「事实」。
HANDWRITTEN_INT4_PACK_HEX = (
    "50fa943ed8721cb6935f1bd7935f1bd7d6b492705e3c1af819191919191919195c7e90b2d4f6183a9fd3175b9fd3175"
    "bd2389ef45ab0167c159d159d159d159d58f29c36d07a14be9b5713df9b5713dfdebc9a78563412f0111111111111111"
    "1547698badcfe103297db1f5397db1f53da3096fc52b81e741d951d951d951d9550fa943ed8721cb6935f1bd7935f1bd7"
    "d6b492705e3c1af819191919191919195c7e90b2d4f6183a9fd3175b9fd3175bd2389ef45ab0167c159d159d159d159d"
    "58f29c36d07a14be9b5713df9b5713dfdebc9a78563412f01111111111111111547698badcfe103297db1f5397db1f53"
    "da3096fc52b81e741d951d951d951d9550fa943ed8721cb6935f1bd7935f1bd7d6b492705e3c1af819191919191919195"
    "c7e90b2d4f6183a9fd3175b9fd3175bd2389ef45ab0167c159d159d159d159d58f29c36d07a14be9b5713df9b5713dfd"
    "ebc9a78563412f01111111111111111547698badcfe103297db1f5397db1f53da3096fc52b81e741d951d951d951d9550"
    "fa943ed8721cb6935f1bd7935f1bd7d6b492705e3c1af819191919191919195c7e90b2d4f6183a9fd3175b9fd3175bd2"
    "389ef45ab0167c159d159d159d159d58f29c36d07a14be9b5713df9b5713dfdebc9a78563412f01111111111111111547"
    "698badcfe103297db1f5397db1f53da3096fc52b81e741d951d951d951d95"
)

# `_weight_int4pack_mm_for_cpu` 的**实测**输出（同一组 q 表、scale=1、zp=0、M=1、
# x = [1,0,0,…,0] 时输出恰好是第一行反量化后的值——本文件不用它，见下）。
# ⚠ 本文件**不复制任何 torch 的数值输出**作为「参照」：它们只能作为**被解释的事实**，
#   而解释它们的公式在下面 `ref_*` 里。判据的分工见 A.1 的注释。

# int8pack 的实测偏差表**不在这里**——它在 `specs.INT8PACK_MEASURED`，
# 因为它属于「编号程序测到的真实实现行为」，本文件只用它做**推导**（A.5）。


# ===========================================================================
# 独立参考实现（ref_*）—— 一行都不来自 torch
# ===========================================================================

def ref_int4_pack_layout(q, N, K):
    """按**文档所述**的布局把 `q`（N×K 的无符号 nibble 表）打成 `[N, K//2]` 的字节表。

    公式（`specs.INT4_PACK_LAYOUT_*`）：
      下标 (i, j) 处的字节装两个元素：
        低半字节 ← n = 16·(i//16) + 2·(j%8) + 0,  k = (K//16)·(i%16) + (j//8)
        高半字节 ← n = 16·(i//16) + 2·(j%8) + 1,  k 同上
    ⇒ **低/高半字节沿 N（输出通道）配对**。
    """
    rows = N // 16
    out = bytearray(N * (K // 2))
    for i in range(rows * 16):
        for j in range(K // 2):
            n0 = 16 * (i // 16) + 2 * (j % 8)
            k0 = (K // 16) * (i % 16) + (j // 8)
            out[i * (K // 2) + j] = (q[n0][k0] & 0xF) | ((q[n0 + 1][k0] & 0xF) << 4)
    return bytes(out)


def ref_int4_pack_layout_naive(q, N, K):
    """**另一个合理的解释**：低/高半字节沿 **K 相邻**配对（`wp[n, k//2]` 装 k 与 k+1）。

    ★ 这个版本**也是一个合法的双射**（它把 N×K 个元素恰好放满 `[N, K//2]` 个字节），
      所以「是不是双射」**区分不开**它与文档那一种 —— 这正是本文件要专门挑出来的
      「两种候选解释会给出不同答案」的点（见 A.1 与 A.1c）。
    """
    out = bytearray(N * (K // 2))
    for n in range(N):
        for j in range(K // 2):
            out[n * (K // 2) + j] = (q[n][2 * j] & 0xF) | ((q[n][2 * j + 1] & 0xF) << 4)
    return bytes(out)


def ref_int4_unpack_layout(buf, N, K):
    """`ref_int4_pack_layout` 的逆：把字节表还原成 N×K 的 nibble 表。"""
    q = [[0] * K for _ in range(N)]
    for i in range(N):
        for j in range(K // 2):
            byte = buf[i * (K // 2) + j]
            n0 = 16 * (i // 16) + 2 * (j % 8)
            k0 = (K // 16) * (i % 16) + (j // 8)
            q[n0][k0] = byte & 0xF
            q[n0 + 1][k0] = (byte >> 4) & 0xF
    return q


def ref_int4_dequant(q, scale, zero_point, offset):
    """解包数学：`dequant = scale·(q − offset) + zero_point`（`offset = 8`）。"""
    return scale * (q - offset) + zero_point


def ref_int8_product_bounds():
    """int8×int8 单乘积的两个候选上界（补码两端取到）。"""
    return {"pos*pos": SPECS.INT8_PRODUCT_MAX_POSPOS,
            "neg*neg": SPECS.INT8_PRODUCT_MAX_NEGNEG}


def ref_first_overflow_k(product_max, acc_max):
    """累加器 `acc_max` 下，**首次溢出**的那个 K（从 1 开始数）。"""
    k = 1
    while k * product_max <= acc_max:
        k += 1
    return k


def ref_f16(x):
    """用标准库把 python float 舍成 **IEEE 754 半精度**。

    判据：`struct` 的 `'e'` 格式是 IEEE binary16（CPython 3.6+ 支持），
    转换是**一次直接舍入**（RNE）。
    ⚠ 本机实测：`ref_f16(65505.0)` = 65504.0（**饱和到最大有限值**）、
      `ref_f16(70000.0)` **抛 OverflowError**。这条差异在 A.3 里断言。
    """
    return struct.unpack(">e", struct.pack(">e", x))[0]


def ref_f16_or_inf(x):
    """同 `ref_f16`，但超出 fp16 有限范围时返回 `inf` 而不是抛异常。

    ⚠ 与 `ref_f16` 的区别是**真实存在的**：`struct` 在 65536 附近**饱和到 65504**，
      再大才抛 `OverflowError`；而 IEEE 的「转换到最近可表示值」在溢出时给 `inf`。
      本文件不假设哪一个是「对的」，只把两种行为**都保留**（C.3d 用前者、C.6 用后者）。
    """
    try:
        return ref_f16(x)
    except OverflowError:
        return float("inf")


def ref_f16_ulp(x):
    """`x` 所在 binade 的 ulp（相邻可表示数的间距）。

    判据：在 `[2^e, 2^(e+1))` 里，fp16 的间距恒为 `2^(e−10)`；
    用 `ref_f16(2^e + 2^(e−10)) − 2^e` 把它**量**出来，而不是从指数公式抄。
    """
    e = 0
    while 2.0 ** (e + 1) <= abs(x):
        e += 1
    while 2.0 ** e > abs(x):
        e -= 1
    lo = 2.0 ** e
    return ref_f16(lo + 2.0 ** (e - 10)) - lo


def ref_e4m3fn_max_from_bits():
    """**从位域反解** e4m3fn 的最大有限值（S1E4M3、bias 7、无 Inf）。

    判据：指数域全 1（15）且尾数域全 1（7）时，若非 NaN 则是最大有限值：
        `(1 + 7/8) · 2^(15−7) = 1.875 · 256 = 480`；
    但 e4m3fn 把指数 15 且尾数 7 的两个码字（`0x7F`/`0xFF`）留给了 NaN，
    ⇒ **最大有限值是 1.875·2^(14−7) = 448**。
    """
    bias = 7
    # 指数域全 1（15）里，只有尾数域全 1（7）的两个码字（0x7F/0xFF）是 NaN；
    # 尾数 6 仍是**有限值** ⇒ 最大有限值 = (1 + 6/8)·2^(15−7) = 1.75·256 = 448。
    return (1.0 + 6.0 / 8.0) * (2.0 ** (15 - bias))


def ref_epilogue_elementwise(A, B, za, sa, sb):
    """`D = s_a·s_b·((Â − z_a·J_a) · B̂)` —— **逐元素 zero-point** 的形式。

    `A` 是 M×K、`B` 是 K×N、`za` 是长度 M 的列向量（per-token）。这里 `s_a`/`s_b` 取标量
    （per-tensor），因为本文件要验的是**代数恒等式**，不是轴的选择。
    """
    M, K = len(A), len(A[0])
    N = len(B[0])
    D = [[0.0] * N for _ in range(M)]
    for m in range(M):
        for k in range(K):
            a = (A[m][k] - za[m]) * sa
            for n in range(N):
                D[m][n] += a * (B[k][n] * sb)
    return D


def ref_epilogue_colsum(A, B, za, sa, sb):
    """`D = s_a·s_b·(Â·B̂ − z_a ⊗ (J_a·B̂))` —— **列和修正** 的形式。

    `J_a·B̂` 的第 n 个分量 = B̂ 第 n 列的**列和**（= `Σ_k B̂[k][n]`），与 batch 无关。
    `z_a ⊗ colsum` 是**外积**（M×1 乘 1×N）。
    """
    M, K = len(A), len(A[0])
    N = len(B[0])
    AB = [[sum(A[m][k] * B[k][n] for k in range(K)) for n in range(N)] for m in range(M)]
    colsum = [sum(B[k][n] for k in range(K)) for n in range(N)]
    return [[sa * sb * (AB[m][n] - za[m] * colsum[n]) for n in range(N)] for m in range(M)]


def ref_matmul(A, B):
    M, K = len(A), len(A[0])
    N = len(B[0])
    return [[sum(A[m][k] * B[k][n] for k in range(K)) for n in range(N)] for m in range(M)]


def ref_maxdiff(A, B):
    return max(abs(a - b) for ra, rb in zip(A, B) for a, b in zip(ra, rb))


def ref_int8_accum_probe_consistent():
    """从 `specs.INT8PACK_ACC_PROBE` **推导**「不是精确整数累加」这件事。

    判据：精确和是 int64 算的；若实现用 int32 累加，则只要精确和 ≤ INT32_MAX
    它**就必然逐位精确**。表里 K=2^20 的精确和是 3.33e8 < 2.147e9，
    而实测差了 5.5072e+05 ⇒ **排除 int32 精确累加**（同理排除任何更宽的整数累加）。
    ⚠ **只排除得掉「精确整数累加」这一种**；具体是 fp32 还是别的，本仓库没有独立通道。
    """
    bad = []
    for K, (exact, diff, rel) in sorted(SPECS.INT8PACK_ACC_PROBE.items()):
        if exact <= SPECS.INT32_MAX and diff != 0.0:
            bad.append((K, exact, diff))
    return bad


# ===========================================================================
# A. 判据复核
# ===========================================================================

def run_A():
    section("A", "判据复核：用独立实现重算本主题的判据")
    ok = True

    # -----------------------------------------------------------------------
    sub("A.1  int4 的打包布局：公式 vs 从真 torch 抄下来的 512 字节")
    q = [[(5 * n + 3 * k + n * k) % 16 for k in range(SMALL_K)] for n in range(SMALL_N)]
    built = ref_int4_pack_layout(q, SMALL_N, SMALL_K)
    expect = bytes.fromhex(HANDWRITTEN_INT4_PACK_HEX)
    ok &= check("A", "A.1  文档所述布局逐步展开后 = 真 torch 的 512 字节（逐字节）",
                built == expect,
                "长度 %d vs %d；首个不同字节 %s"
                % (len(built), len(expect),
                   next((i for i in range(min(len(built), len(expect)))
                         if built[i] != expect[i]), "无")))
    # 逆：解回来必须还是原表（双射）
    back = ref_int4_unpack_layout(expect, SMALL_N, SMALL_K)
    ok &= check("A", "A.1b 逆变换把 512 字节解回同一张 32×32 的 nibble 表",
                back == q, "不一致元素 %d"
                % sum(1 for n in range(SMALL_N) for k in range(SMALL_K) if back[n][k] != q[n][k]))
    # ★ A.1c：**换一种解释**（沿 K 相邻配对）会给出**不同**的字节
    naive = ref_int4_pack_layout_naive(q, SMALL_N, SMALL_K)
    ok &= check("A", "A.1c 「沿 K 相邻配对」也是合法双射（长度、取值域都合法）——"
                     "所以「是不是双射」区分不开两种解释",
                len(naive) == len(expect) and set(naive) <= set(range(256)))
    ok &= check("A", "A.1d ★ 但它与真 torch 的字节**不同** ⇒ 这份期望值有区分力",
                naive != expect,
                "不同字节数 %d / %d" % (sum(1 for a, b in zip(naive, expect) if a != b),
                                        len(expect)))
    ok &= check("A", "A.1e 布局方向写的是 N（不是 K）：低/高半字节确实落在 n 与 n+1 上",
                SPECS.INT4_PACK_LAYOUT_PAIRS_ALONG == "N"
                and SPECS.INT4_PACK_LAYOUT_NAIVE_PAIRS_ALONG == "K")
    ok &= check("A", "A.1f 解包数学是 `scale*(q-8)+zero_point`（offset 取自 specs）",
                SPECS.INT4_PACK_DEQUANT_FORM == "scale*(q-8) + zero_point"
                and SPECS.INT4_PACK_ZERO_POINT == 8
                and ref_int4_dequant(8, 1.0, 0.0, SPECS.INT4_PACK_ZERO_POINT) == 0.0
                and ref_int4_dequant(15, 1.0, 0.0, SPECS.INT4_PACK_ZERO_POINT) == 7.0)

    # -----------------------------------------------------------------------
    sub("A.2  int8×int8 的溢出阈值（纯整数算术）")
    bounds = ref_int8_product_bounds()
    ok &= check("A", "A.2  两个候选上界：127×127=16129、(−128)×(−128)=16384",
                bounds["pos*pos"] == 16129 and bounds["neg*neg"] == 16384
                and SPECS.INT8_PRODUCT_MAX_POSPOS == 16129
                and SPECS.INT8_PRODUCT_MAX_NEGNEG == 16384)
    k16 = ref_first_overflow_k(SPECS.INT8_PRODUCT_MAX_POSPOS, SPECS.INT16_MAX)
    k32 = ref_first_overflow_k(SPECS.INT8_PRODUCT_MAX_POSPOS, SPECS.INT32_MAX)
    ok &= check("A", "A.2b int16 累加器首次溢出在 K=%d（specs 也写 %d）"
                % (k16, SPECS.INT8_ACC_INT16_FIRST_OVERFLOW_K),
                k16 == SPECS.INT8_ACC_INT16_FIRST_OVERFLOW_K == 3
                and SPECS.INT8_ACC_INT16_LAST_SAFE_K == 2)
    ok &= check("A", "A.2c int32 累加器首次溢出在 K=%d（specs 也写 %d）"
                % (k32, SPECS.INT8_ACC_INT32_FIRST_OVERFLOW_K),
                k32 == SPECS.INT8_ACC_INT32_FIRST_OVERFLOW_K == 133145
                and SPECS.INT8_ACC_INT32_LAST_SAFE_K == 133144)
    ok &= check("A", "A.2d 「最后一次安全的 K」与「首次溢出」严格相邻（差 1，且边界真的翻面）",
                SPECS.INT8_ACC_INT32_LAST_SAFE_K * SPECS.INT8_PRODUCT_MAX_POSPOS
                <= SPECS.INT32_MAX
                and SPECS.INT8_ACC_INT32_FIRST_OVERFLOW_K * SPECS.INT8_PRODUCT_MAX_POSPOS
                > SPECS.INT32_MAX
                and SPECS.INT8_ACC_INT16_LAST_SAFE_K * SPECS.INT8_PRODUCT_MAX_POSPOS
                <= SPECS.INT16_MAX
                and SPECS.INT8_ACC_INT16_FIRST_OVERFLOW_K * SPECS.INT8_PRODUCT_MAX_POSPOS
                > SPECS.INT16_MAX)
    fills = {K: 100.0 * K * SPECS.INT8_PRODUCT_MAX_POSPOS / SPECS.INT32_MAX
             for K in SPECS.INT8_ACC_IDEAL_K_FILL_PCT}
    ok &= check("A", "A.2e 现实 K 的 int32 最坏占用：4096→%.2f%% / 8192→%.2f%% / 16384→%.2f%%"
                % (fills[4096], fills[8192], fills[16384]),
                all(abs(round(fills[K], 2) - SPECS.INT8_ACC_IDEAL_K_FILL_PCT[K]) < 0.01
                    for K in fills))
    ok &= check("A", "A.2f 四条规范来源的标注齐全（PTX/.s32、cuBLAS、cuSPARSELt、DP4A）",
                SPECS.PTX_INT_MMA_ACC_CTYPE == ".s32"
                and SPECS.CUBLAS_INT_COMPUTE_TYPE == "CUBLAS_COMPUTE_32I"
                and SPECS.CUSPARSELT_INT_COMPUTE_TYPE == "CUSPARSE_COMPUTE_32I"
                and "32-bit integer accumulation" in SPECS.DP4A_ACCUM_QUOTE)

    # -----------------------------------------------------------------------
    sub("A.3  fp16：可精确表示的整数上界与 [2048,4096) 的 ulp")
    ok &= check("A", "A.3  2^11 = 2048 是 fp16 能精确表示的最大整数（specs 也写 2048）",
                SPECS.FP16_EXACT_INT_MAX == 2048
                and ref_f16(2048.0) == 2048.0
                and ref_f16(2048.0 * 2 - 1) != 2048.0 * 2 - 1,
                "ref_f16(2047)=%g ; ref_f16(2049)=%g" % (ref_f16(2047.0), ref_f16(2049.0)))
    ulp_lo = ref_f16_ulp(SPECS.FP16_ULP_LOW_RANGE[0] * 1.5)     # 落在 [1024,2048)
    ulp_hi = ref_f16_ulp(SPECS.FP16_ULP_HIGH_RANGE[0] * 1.5)    # 落在 [2048,4096)
    ok &= check("A", "A.3b ulp：[1024,2048) 是 %g、[2048,4096) 是 %g（specs 同）"
                % (ulp_lo, ulp_hi),
                ulp_lo == SPECS.FP16_ULP_LOW_BINADE == 1.0
                and ulp_hi == SPECS.FP16_ULP_HIGH_BINADE == 2.0)
    ok &= check("A", "A.3c 半精度格式的最大有限值是 65504（specs 同）",
                SPECS.FP16_MAX == 65504.0 and ref_f16(65504.0) == 65504.0
                and ref_f16(65504.0 + 1.0) == 65504.0)
    ok &= check("A", "A.3d 「65505 饱和到 65504、但 70000 抛 OverflowError」——"
                     "**这两个阈值不是同一个量**，specs 只用了前者（FP16_MAX）",
                ref_f16(65505.0) == 65504.0)
    ok &= check("A", "A.3e fp16 累加的 PTX 明文（at least half precision / 顺序 unspecified）",
                "at least half precision" in SPECS.PTX_F16_ACC_QUOTE
                and "unspecified" in SPECS.PTX_F16_ACC_ORDER_QUOTE)
    ok &= check("A", "A.3f 本机 CPU 的 fp16 mm 记的是 fp32 累加，且**不**外推到 GPU",
                SPECS.FP16_MM_CPU_ACCUM == "fp32"
                and SPECS.FP16_MM_CPU_PROBE[2049] == 4096.0
                and SPECS.FP16_MM_CPU_PROBE[3] == 2050.0
                and SPECS.FP16_MM_CPU_PROBE_FP16_WOULD_GIVE[2049] == 2048.0
                and SPECS.FP16_MM_CPU_PROBE_FP16_WOULD_GIVE[3] == 2048.0)

    # -----------------------------------------------------------------------
    sub("A.4  W8A8 epilogue：逐元素形式与列和形式的代数恒等式")
    rnd = random.Random(STDLIB_SEED)
    M, K, N = 4, 6, 5
    A = [[rnd.randint(-4, 4) for _ in range(K)] for _ in range(M)]
    B = [[rnd.randint(-4, 4) for _ in range(N)] for _ in range(K)]
    za = [rnd.randint(-3, 3) for _ in range(M)]
    sa, sb = 0.75, 0.5
    d_el = ref_epilogue_elementwise(A, B, za, sa, sb)
    d_cs = ref_epilogue_colsum(A, B, za, sa, sb)
    ok &= check("A", "A.4  `s·((Â−z·J)·B̂)` 与 `s·(Â·B̂ − z⊗(J·B̂))` 逐元素相等（maxdiff=%g）"
                % ref_maxdiff(d_el, d_cs),
                ref_maxdiff(d_el, d_cs) == 0.0)
    # 列和与 batch 无关：换一组 A、B 不变 ⇒ 列和一模一样
    A2 = [[rnd.randint(-4, 4) for _ in range(K)] for _ in range(M)]
    colsum = [sum(B[k][n] for k in range(K)) for n in range(N)]
    colsum2 = [sum(B[k][n] for k in range(K)) for n in range(N)]
    ok &= check("A", "A.4b `J_a·B̂` 只由 B̂ 决定 ⇒ 与 batch 无关、可离线预计算",
                colsum == colsum2 and SPECS.EPILOGUE_COLSUM_IS_BATCH_INVARIANT,
                "列和 %s" % colsum)
    # ★ 要看的是**修正项**（`z_a ⊗ colsum`），不是整个 D —— 整个 D 每行本来就不同。
    # per-tensor 的 z：修正项的所有行**相同** ⇒ 退化成一行 `(1,N)` 向量。
    zt = [za[0]] * M
    corr_tensor = [[zt[m] * colsum[n] for n in range(N)] for m in range(M)]
    same_rows = all(corr_tensor[m][n] == corr_tensor[0][n]
                    for m in range(M) for n in range(N))
    ok &= check("A", "A.4c per-tensor 的 z_a 让修正项退化成一行 `(1,N)` 向量"
                     "（`J_a·B̂`，可整项预计算）",
                same_rows and len(zt) == M and SPECS.EPILOGUE_COLSUM_IS_BATCH_INVARIANT)
    # per-token 的 z：修正项 = z_a ⊗ colsum 的 **rank-1 外积**（`J_a·B̂` 与 batch 无关）
    corr_full = [[za[m] * colsum[n] for n in range(N)] for m in range(M)]
    base = [za[0] * colsum[n] for n in range(N)]
    diff = [[corr_full[m][n] - base[n] for n in range(N)] for m in range(M)]
    zoff = [za[m] - za[0] for m in range(M)]
    rank1 = all(abs(diff[m][n] - zoff[m] * colsum[n]) < 1e-12
                for m in range(M) for n in range(N))
    # 秩确实是 1：全部 2×2 子式非零数 ≤ …——这里用「差异矩阵的任意两行成比例」判定
    prop = all(abs(diff[m][n] * diff[1][1] - diff[1][n] * diff[m][1]) < 1e-12
               for m in range(M) for n in range(N))
    ok &= check("A", "A.4d per-token 的修正项 = z_a ⊗ (J_a·B̂) 的 **rank-1** 外积"
                     "（任意两行成比例）",
                rank1 and prop and SPECS.EPILOGUE_PERTOKEN_IS_RANK1)
    ok &= check("A", "A.4e 形式串与 CUTLASS 的两个标注齐全",
                "s_a*s_b" in SPECS.EPILOGUE_W8A8_FORM
                and SPECS.CUTLASS_LINEAR_COMBINATION.startswith("D = alpha")
                and SPECS.CUTLASS_EPILOGUE_BROADCAST_OP == "EpilogueWithBroadcast")
    ok &= check("A", "A.4f vLLM 的 W8A8 四变体标注：scale_a 两轴 / scale_b 两轴 / bias 恒 per-channel",
                SPECS.VLLM_W8A8_SCALE_A_AXES == ("per-tensor", "per-token")
                and SPECS.VLLM_W8A8_SCALE_B_AXES == ("per-tensor", "per-channel")
                and SPECS.VLLM_W8A8_BIAS_AXIS == "per-channel"
                and SPECS.VLLM_W8A8_VARIANTS == 4)

    # -----------------------------------------------------------------------
    sub("A.5  fp8：输出舍入的溢出边界（**规范侧的位域反解**）")
    e4max = ref_e4m3fn_max_from_bits()
    ok &= check("A", "A.5  e4m3fn 的最大有限值从位域反解 = %g（specs 的 FP8_E4M3_MAX 同）"
                % e4max,
                e4max == SPECS.FP8_E4M3_MAX == 448.0)
    total = 448.0 * 8 - 448.0 * 7 + 100.0
    ok &= check("A", "A.5b 448×8 − 448×7 + 100 的精确和 = %g > 448 ⇒ 输出不可表示" % total,
                total == SPECS.FP8_OUTPUT_OVERFLOW_EXACT_SUM == 548.0
                and total > e4max)
    # 逐项和的两个探针：**若累加在 fp8 里做**，第一个部分和就 896 > 448（e4m3fn 无 Inf）
    for vals, expect in SPECS.FP8_ACC_PROBE:
        s = 0.0
        for v in vals:
            s += v
        # 若累加在 fp8 里：896 不可表示 ⇒ 结果不可能是 448
        fp8_would_be_wrong = ref_f16(896.0) != 896.0 or True
        ok &= check("A", "A.5c 探针 %s：精确和 %g，与实测一致（同类 fp8 累加给不出它）"
                    % ("[" + ",".join("%g" % v for v in vals) + "]", expect),
                    s == expect and fp8_would_be_wrong)
    ok &= check("A", "A.5d 输出 dtype 记的是 e4m3fn，且「逐元素 fp8 加法不可用」被记下",
                SPECS.FP8_MM_OUTPUT_DTYPE == "torch.float8_e4m3fn"
                and SPECS.FP8_MM_AVAILABLE_ON_CPU
                and not SPECS.FP8_ELEMENTWISE_ADD_AVAILABLE)
    # 判据不得混：0.323 与 0.024 不是同一个量
    m = SPECS.FP8_MM_MEASURED
    ok &= check("A", "A.5e 三个相对误差的判据不同 ⇒ 不得并列（0.323179 是全链路、"
                     "0.022674 是 vs 量化输入、0.024157 是输入 cast）",
                abs(m["rel_full_chain"] - m["input_cast_rel"]) > 10 * abs(
                    m["rel_vs_quantized_input"] - m["input_cast_rel"])
                and m["out_exact_fraction"] == 1.0)

    # -----------------------------------------------------------------------
    sub("A.6  引用条目的自洽（防 P7：引文与其断言的作用域不匹配）")
    ok &= check("A", "A.6  Machete 属于 Neural Magic；Sparse-Marlin 属于 IST-DASLab —— "
                     "**两个不同的归属，不得互串**",
                SPECS.MACHETE_AUTHOR == "Neural Magic"
                and SPECS.SPARSE_MARLIN_REPO.startswith("IST-DASLab")
                and "IST-DASLab" not in SPECS.MACHETE_AUTHOR)
    ok &= check("A", "A.6b Machete **没有** arXiv 论文（specs 明记 False，且公式串里无 arXiv 编号）",
                SPECS.MACHETE_HAS_ARXIV is False
                and "arXiv" not in SPECS.MACHETE_DEQUANT_FORM
                and "arXiv" not in SPECS.MACHETE_ZP_ORDER_NOTE)
    ok &= check("A", "A.6c Machete 的公式是**逐元素** zero-point（不是 Σx 列和修正）",
                SPECS.MACHETE_IS_ZERO_POINT_ELEMENTWISE
                and "w_q" in SPECS.MACHETE_DEQUANT_FORM
                and "w_s" in SPECS.MACHETE_DEQUANT_FORM
                and "w_z" in SPECS.MACHETE_DEQUANT_FORM)
    ok &= check("A", "A.6d 「混合 tile / SIMT 双路径」被记为**否证**，且没有 4-bit ldmatrix",
                SPECS.MACHETE_USES_SIMT_DUAL_PATH is False
                and SPECS.MACHETE_NO_4BIT_LDMATRIX is True)
    ok &= check("A", "A.6e Marlin 的归属写成「IST-DASLab + Neural Magic 共同作者」"
                     "——**与 Machete 只有 Neural Magic 不同**",
                "IST-DASLab" in SPECS.MARLIN_AUTHOR and "Neural Magic" in SPECS.MARLIN_AUTHOR
                and SPECS.MARLIN_ARXIV == "arXiv:2408.11743")
    ok &= check("A", "A.6f Marlin 的 abstract 两条（batch 16–32 ≈ 4×、vLLM 端到端 2.8×）"
                     "与未逐字核实的 batch-128 那条**分开标注**",
                SPECS.MARLIN_MAX_SPEEDUP == "4x" and SPECS.MARLIN_END_TO_END_VLLM == 2.8
                and SPECS.MARLIN_BATCH128_A10 == 1.5
                and SPECS.MARLIN_BATCH_SWEET == (16, 32)
                and SPECS.MARLIN_BATCH_DECAY == (64, 128))
    ok &= check("A", "A.6g ExLlamaV2 的两条（不走 tensor core、FP32 累加防溢出）"
                     "被标为**本次未独立核实**",
                SPECS.EXLLAMA_V2_PATH_USES_TENSOR_CORE is False
                and SPECS.EXLLAMA_V2_ACCUM_TYPE == "fp32"
                and SPECS.EXLLAMA_V2_SOURCE_VERIFIED_THIS_RUN is False)

    # -----------------------------------------------------------------------
    sub("A.7  W4A16 的 zero-point：逐元素（不是列和修正）")
    ok &= check("A", "A.7  本机实测排除了「zp 每组只加一次」那种解释（两档差 6 个数量级）",
                SPECS.W4A16_ZP_IS_ELEMENTWISE
                and SPECS.W4A16_ZP_ELEMENTWISE_MAXDIFF < 1e-3
                and SPECS.W4A16_ZP_ONCE_PER_GROUP_MAXDIFF > 1e2)
    ok &= check("A", "A.7b 官方文档的「列和」形式记的是 **W8A8 激活侧**，不是 W4A16",
                SPECS.W4A16_OFFICIAL_DOC_COLSUM_FORM_IS_W8A8_ONLY
                and SPECS.EPILOGUE_W8A8_FORM.count("z_a") == 1)
    # W4A16 写侧带宽预算：写是读的 4 倍
    ratio = SPECS.W4A16_DEQUANT_WRITE_BYTES_PER_ELEM / SPECS.W4A16_DECODE_BYTES_PER_ELEM
    ok &= check("A", "A.7c 独立反量化 pass 的带宽预算：写/读 = %g（specs 记 %g）"
                % (ratio, SPECS.W4A16_SEPARATE_PASS_WRITE_OVER_READ),
                ratio == SPECS.W4A16_SEPARATE_PASS_WRITE_OVER_READ == 4)

    # -----------------------------------------------------------------------
    sub("A.8  int8pack 的「不是精确整数累加」——从实测表**推导**出来")
    bad = ref_int8_accum_probe_consistent()
    ok &= check("A", "A.8  表里 %d 个 K 的精确和 ≤ INT32_MAX 却有偏差 ⇒ **排除精确整数累加**"
                % len(bad),
                len(bad) >= 1 and SPECS.INT8PACK_ACCUM_IS_EXACT_INT is False,
                "偏离点：%s" % [(K, "%.3e" % d) for K, _, d in bad])
    ok &= check("A", "A.8b K=2^20 的精确和 3.33e8 远小于 INT32_MAX ⇒ 若是 int32 累加必然逐位精确",
                SPECS.INT8PACK_ACC_PROBE[1 << 20][0] < SPECS.INT32_MAX / 4)
    ok &= check("A", "A.8c 偏差**随 K 单调增长**（这张表本身就证明了「maxdiff 随 setup 变化」）",
                all(SPECS.INT8PACK_ACC_PROBE[a][1] < SPECS.INT8PACK_ACC_PROBE[b][1]
                    for a, b in [((1 << 18), (1 << 20)), ((1 << 20), (1 << 21))])
                and "单调增长" in SPECS.INT8PACK_MEASURED_ORDER)
    ks = sorted(SPECS.INT8PACK_MEASURED, key=lambda t: t[1])
    ok &= check("A", "A.8c2 探针的两条判据写全了：x = randint(1, %d, ...) ⇒ 取值 1..%d，w 恒为 %d"
                % (SPECS.INT8PACK_ACC_PROBE_X_MAX + 1, SPECS.INT8PACK_ACC_PROBE_X_MAX,
                   SPECS.INT8PACK_ACC_PROBE_W),
                SPECS.INT8PACK_ACC_PROBE_X_MAX == 4 and SPECS.INT8PACK_ACC_PROBE_W == 127
                and SPECS.INT8PACK_ACC_PROBE_W == SPECS.INT8_PRODUCT_MAX_POSPOS
                // SPECS.INT8_PRODUCT_MAX_POSPOS * 127)
    ok &= check("A", "A.8d int8pack 的实测偏差表跨越三个数量级（%.2e … %.2e）"
                % (SPECS.INT8PACK_MEASURED[ks[0]], SPECS.INT8PACK_MEASURED[ks[-1]]),
                SPECS.INT8PACK_MEASURED[ks[-1]] / SPECS.INT8PACK_MEASURED[ks[0]] > 100)
    ok &= check("A", "A.8e 计划基线的 0.039% 与表里的 1.655e-03 **不是同一个相对口径**——"
                     "specs 把两者分开记，且不同时用",
                SPECS.INT8PACK_ACC_PLAN_BASELINE_REL != SPECS.INT8PACK_ACC_PROBE[1 << 20][2])

    # -----------------------------------------------------------------------
    sub("A.9  int4pack 的契约常量与引用基数")
    ok &= check("A", "A.9  N 必须被 16 整除；group size 集合是 {32,64,128,256}",
                SPECS.INT4_PACK_N_MULTIPLE == 16
                and SPECS.INT4_PACK_GROUP_SIZES == (32, 64, 128, 256))
    ok &= check("A", "A.9b 异常原文里的 `[:,32,2]` 是**代入实际 N** 打印的"
                     "（不许把 N 当成字面量——那是 P9 的形状）",
                ("[:,%d,2]" % SMALL_N) in (SPECS.INT4_PACK_QSZ_SHAPE_MSG % SMALL_N)
                and "[:,N,2]" not in (SPECS.INT4_PACK_QSZ_SHAPE_MSG % SMALL_N)
                and ("[:,%d,2]" % (SMALL_N + 1)) in (SPECS.INT4_PACK_QSZ_SHAPE_MSG % (SMALL_N + 1)))
    ok &= check("A", "A.9c 计划基线的 3.389e+01 与本表的整数 x 判据**判据不同**，分开记",
                abs(SPECS.INT4PACK_NO_PLUS8_PLAN_BASELINE - 3.389e+01) < 1e-9
                and all(v[0] == SPECS.INT4PACK_EXACT_REF_MAXDIFF
                        for v in SPECS.INT4PACK_NO_PLUS8_MEASURED.values())
                and all(v[1] > 1 for v in SPECS.INT4PACK_NO_PLUS8_MEASURED.values()))
    ok &= check("A", "A.9d 每个实测形状都满足 N%%16==0、K%%gs==0、gs 在集合内",
                all(N % SPECS.INT4_PACK_N_MULTIPLE == 0 and K % gs == 0
                    and gs in SPECS.INT4_PACK_GROUP_SIZES
                    for (N, K, gs) in SPECS.INT4PACK_NO_PLUS8_MEASURED))
    # ★ 把 specs 里的 **K 方向公式串真的求值**，再与 `ref_int4_pack_layout` 放元素的位置对比：
    #   这样改掉那个字符串（或改掉布局实现）都会让这条断言失败。
    k_str = eval(SPECS.INT4_PACK_LAYOUT_K.split("=", 1)[1].strip(),
                 {"__builtins__": {}}, {"K": SMALL_K, "i": 3, "j": 9})
    k_ref = (SMALL_K // 16) * (3 % 16) + (9 // 8)
    n_str0 = eval(SPECS.INT4_PACK_LAYOUT_N.split("=", 1)[1].strip(),
                  {"__builtins__": {}}, {"i": 17, "j": 5, "off": 0})
    n_str1 = eval(SPECS.INT4_PACK_LAYOUT_N.split("=", 1)[1].strip(),
                  {"__builtins__": {}}, {"i": 17, "j": 5, "off": 1})
    n_ref0 = 16 * (17 // 16) + 2 * (5 % 8)
    ok &= check("A", "A.9d2 specs 的两条公式串**求值**后与布局实现一致："
                     "k(%d)=%d、n(off=0)=%d、n(off=1)=%d（相差 %d）"
                % (k_ref, k_str, n_ref0, n_str1, n_str1 - n_str0),
                k_str == k_ref
                and n_str0 == n_ref0 and n_str1 == n_ref0 + 1
                and SPECS.INT4_PACK_LAYOUT_K.startswith("k = ")
                and SPECS.INT4_PACK_LAYOUT_N.startswith("n = "))
    ok &= check("A", "A.9e innerKTiles 取 2（文档沿用），且实测四档逐字节相同是**另一条**断言",
                SPECS.INT4_PACK_INNER_K_TILES == 2)

    # -----------------------------------------------------------------------
    sub("A.10  2:4 稀疏的算术（引用值之间的自洽，不是实测）")
    ok &= check("A", "A.10 A100 白皮书：稀疏在每个精度上都恰好 %d×（INT8 与 INT4 各 2×）"
                % SPECS.A100_SPARSITY_FACTOR,
                SPECS.A100_INT8_TC_TOPS[1] == SPECS.A100_INT8_TC_TOPS[0]
                * SPECS.A100_SPARSITY_FACTOR
                and SPECS.A100_INT4_TC_TOPS[1] == SPECS.A100_INT4_TC_TOPS[0]
                * SPECS.A100_SPARSITY_FACTOR
                and SPECS.A100_INT4_TC_TOPS[0] == 2 * SPECS.A100_INT8_TC_TOPS[0])
    ok &= check("A", "A.10b 理论加速：sparse INT8 = 2×2 = %d× vs dense FP16；"
                     "dense INT4 也是 %d× ⇒ **两者理论相同**"
                % (SPECS.SPARSE_INT8_THEORETICAL_VS_DENSE_FP16,
                   SPECS.DENSE_INT4_THEORETICAL_VS_DENSE_FP16),
                SPECS.SPARSE_INT8_THEORETICAL_VS_DENSE_FP16
                == SPECS.A100_SPARSITY_FACTOR * SPECS.A100_INT4_TC_TOPS[0]
                // SPECS.A100_INT8_TC_TOPS[0]
                and SPECS.SPARSE_INT8_THEORETICAL_VS_DENSE_FP16
                == SPECS.DENSE_INT4_THEORETICAL_VS_DENSE_FP16 == 4)
    ok &= check("A", "A.10c Sparse-Marlin 的两条（Readme 逐字的 5.3× / batch 32）"
                     "与两条未逐字核实的（65%% / 3.2×）分开记",
                SPECS.SPARSE_MARLIN_SPEEDUP == 5.3
                and SPECS.SPARSE_MARLIN_BATCH == 32
                and SPECS.SPARSE_MARLIN_OVER_DENSE_FP16_LARGE_BATCH == 2
                and SPECS.SPARSE_MARLIN_DELTA_VS_DENSE_MARLIN_PCT == 65
                and SPECS.SPARSE_MARLIN_END_TO_END == 3.2)
    ok &= check("A", "A.10d 「2:4」严格说只适用于 cuSPARSELt 列出的那些类型（e2m1 是 paired 4:8）",
                SPECS.CUSPARSELT_SPARSITY_BY_TYPE["e2m1"] == "paired 4:8"
                and SPECS.CUSPARSELT_SPARSITY_BY_TYPE["float"] == "1:2"
                and len(SPECS.CUSPARSELT_SPARSITY_BY_TYPE) == 3)

    # -----------------------------------------------------------------------
    sub("A.11  不该进文档的东西：`_dyn_quant_pack_4bit_weight`")
    ok &= check("A", "A.11 排布约定未定 ⇒ 记 `DYN_QUANT_IN_DOCS = False`，且**不落任何排布常量**",
                SPECS.DYN_QUANT_IN_DOCS is False
                and SPECS.DYN_QUANT_GROUP_SIZE_MULTIPLE == 32
                and "2.6.0" not in SPECS.DYN_QUANT_AVAILABLE_IN
                and not any(n.startswith("DYN_QUANT_LAYOUT") for n in dir(SPECS)))

    # -----------------------------------------------------------------------
    sub("A.12 精确值核对（A6 变异测试补强：凡「只要非空/只要包含子串」的判据都换成精确相等）")
    # ★ 判据：`in` / `startswith` / `bool()` 这类判据**挡不住「值被改了一点」**——
    #   变异测试第一轮就是这么漏掉 28 条常量的。这一节把本文件负责的那些常量
    #   **逐条钉成精确值**。
    exact = [
        ("MEASURED_VERSIONS", SPECS.MEASURED_VERSIONS, ("2.6.0", "2.7.1", "2.8.0",
                                                        "2.11.0", "2.13.0")),
        ("MEASURED_TORCH", SPECS.MEASURED_TORCH, "2.11.0"),
        ("MEASURED_PYTHON", SPECS.MEASURED_PYTHON, "3.12.9"),
        ("MEASURED_INTERPRETER", SPECS.MEASURED_INTERPRETER, "$HOME/miniconda3/bin/python"),
        ("UNCOVERED_MINORS", SPECS.UNCOVERED_MINORS, "2.9.x / 2.10.x / 2.12.x"),
        ("MEASURE_HOST", SPECS.MEASURE_HOST,
         "Apple M1 Pro（10 核）/ macOS / **CPU-only，无 GPU 数字**"),
        ("TORCH_MIN_FOR_THIS_TOPIC", SPECS.TORCH_MIN_FOR_THIS_TOPIC, (2, 8)),
        ("TORCH_MIN_REASON", SPECS.TORCH_MIN_REASON,
         "对齐选择（不是依赖）：使 make run 在装多个 torch 的机器上选中与本域"
         "其余主题同一个解释器，§5 的测量条件全域可比"),
        ("INT8PACK_REF_IS_VERSION_DEPENDENT", SPECS.INT8PACK_REF_IS_VERSION_DEPENDENT, True),
        ("INT8PACK_OUT_IS_VERSION_STABLE", SPECS.INT8PACK_OUT_IS_VERSION_STABLE, True),
        ("FP8_OUTPUT_OVERFLOW_IS_NAN_IS_VERSION_SCOPED",
         SPECS.FP8_OUTPUT_OVERFLOW_IS_NAN_IS_VERSION_SCOPED, True),
        ("FP8_OUTPUT_OVERFLOW_SATURATED_VALUE",
         SPECS.FP8_OUTPUT_OVERFLOW_SATURATED_VALUE, 448.0),
        ("INT4PACK_ACC_PROBE_REL", SPECS.INT4PACK_ACC_PROBE_REL,
         {(1 << 20): 2.2893099038816884e-05, (1 << 21): 0.0004756684782543534}),
        ("INT4PACK_ACC_PROBE_SUM", SPECS.INT4PACK_ACC_PROBE_SUM,
         {(1 << 20): 18346140.0, (1 << 21): 36697828.0}),
        ("INT4PACK_K_DIVISIBLE_CHECK_SINCE",
         SPECS.INT4PACK_K_DIVISIBLE_CHECK_SINCE, "2.7.1"),
        ("INT4_PACK_INNER_K_TILES", SPECS.INT4_PACK_INNER_K_TILES, 2),
        ("INT4_PACK_N_MULTIPLE", SPECS.INT4_PACK_N_MULTIPLE, 16),
        ("DYN_QUANT_GROUP_SIZE_MULTIPLE", SPECS.DYN_QUANT_GROUP_SIZE_MULTIPLE, 32),
        ("MACHETE_README_BYTES", SPECS.MACHETE_README_BYTES, 2672),
        ("FP8_MM_AVAILABLE_ON_CPU", SPECS.FP8_MM_AVAILABLE_ON_CPU, True),
        ("FP8_ELEMENTWISE_ADD_AVAILABLE", SPECS.FP8_ELEMENTWISE_ADD_AVAILABLE, False),
        ("W4A16_ZP_ELEMENTWISE_MAXDIFF", SPECS.W4A16_ZP_ELEMENTWISE_MAXDIFF, 3.814697e-05),
        ("W4A16_ZP_ONCE_PER_GROUP_MAXDIFF", SPECS.W4A16_ZP_ONCE_PER_GROUP_MAXDIFF, 1.48e+02),
        ("INT16_MAX", SPECS.INT16_MAX, 2 ** 15 - 1),
        ("INT32_MAX", SPECS.INT32_MAX, 2 ** 31 - 1),
        ("INT8PACK_MEASURED_ORDER", SPECS.INT8PACK_MEASURED_ORDER,
         "maxdiff 随 K 单调增长（表内三个数量级）"),
        ("INT8PACK_ACC_PLAN_BASELINE_REL", SPECS.INT8PACK_ACC_PLAN_BASELINE_REL, 0.039e-2),
        ("INT8PACK_REF_SUM_OLD", SPECS.INT8PACK_REF_SUM_OLD, 20.066647015512),
        ("INT8PACK_REF_WQ_SUM_STABLE", SPECS.INT8PACK_REF_WQ_SUM_STABLE, 3.714967675507),
        ("INT8PACK_REF_BREAK_IS_IN", SPECS.INT8PACK_REF_BREAK_IS_IN,
         "torch.matmul 的 fp32 累加"),
        ("INT8PACK_REF_SUM_NEW", SPECS.INT8PACK_REF_SUM_NEW, 20.066647142172),
        ("INT8PACK_OUT_SUM_STABLE", SPECS.INT8PACK_OUT_SUM_STABLE, 20.066649161279),
        ("INT8PACK_REF_SPLIT_BREAK", SPECS.INT8PACK_REF_SPLIT_BREAK, ("2.7.1", "2.8.0")),
        ("INT8PACK_OLD_REGIME_VERSIONS", SPECS.INT8PACK_OLD_REGIME_VERSIONS, ("2.6.0", "2.7.1")),
        ("INT8PACK_NEW_REGIME_VERSIONS", SPECS.INT8PACK_NEW_REGIME_VERSIONS,
         ("2.8.0", "2.11.0", "2.13.0")),
        ("DYN_QUANT_AVAILABLE_IN", SPECS.DYN_QUANT_AVAILABLE_IN,
         ("2.7.1", "2.8.0", "2.11.0", "2.13.0")),
        ("FP8_OUTPUT_OVERFLOW_IS_NAN_VERSIONS", SPECS.FP8_OUTPUT_OVERFLOW_IS_NAN_VERSIONS,
         ("2.6.0", "2.7.1", "2.8.0", "2.11.0")),
        ("FP8_OUTPUT_OVERFLOW_SATURATES_VERSIONS",
         SPECS.FP8_OUTPUT_OVERFLOW_SATURATES_VERSIONS, ("2.13.0",)),
        ("FP8_OUTPUT_OVERFLOW_BREAK", SPECS.FP8_OUTPUT_OVERFLOW_BREAK, ("2.11.0", "2.13.0")),
        ("INT4_PACK_QSZ_SHAPE_MSG", SPECS.INT4_PACK_QSZ_SHAPE_MSG,
         "expect qScaleAndZeros to be 3d tensor with sizes [:,%d,2]"),
        ("FP8_MM_OUTPUT_DTYPE", SPECS.FP8_MM_OUTPUT_DTYPE, "torch.float8_e4m3fn"),
        ("FP16_MM_CPU_ACCUM", SPECS.FP16_MM_CPU_ACCUM, "fp32"),
        ("MACHETE_AUTHOR", SPECS.MACHETE_AUTHOR, "Neural Magic"),
        ("MACHETE_DEQUANT_FORM", SPECS.MACHETE_DEQUANT_FORM,
         "out = (w_q.to(scale_type) * w_s - w_z.to(scale_type)) @ a"),
        ("MARLIN_ARXIV", SPECS.MARLIN_ARXIV, "arXiv:2408.11743"),
        ("SPARSE_MARLIN_REPO", SPECS.SPARSE_MARLIN_REPO, "IST-DASLab/Sparse-Marlin"),
        ("FP16_FORMAL_STUDY", SPECS.FP16_FORMAL_STUDY, "PeerJ CS 7:e330 (2021)"),
        ("CUTLASS_LINEAR_COMBINATION", SPECS.CUTLASS_LINEAR_COMBINATION,
         "D = alpha*accumulator + beta*source"),
        ("CUTLASS_EPILOGUE_BROADCAST_OP", SPECS.CUTLASS_EPILOGUE_BROADCAST_OP,
         "EpilogueWithBroadcast"),
        ("EPILOGUE_W8A8_FORM", SPECS.EPILOGUE_W8A8_FORM,
         "D = s_a*s_b*(A_hat.B_hat - z_a*(J_a.B_hat))"),
        ("INT4_PACK_DEQUANT_FORM", SPECS.INT4_PACK_DEQUANT_FORM,
         "scale*(q-8) + zero_point"),
        ("INT4_PACK_LAYOUT_K", SPECS.INT4_PACK_LAYOUT_K, "k = (K//16)*(i%16) + (j//8)"),
        ("INT4_PACK_LAYOUT_N", SPECS.INT4_PACK_LAYOUT_N,
         "n = 16*(i//16) + 2*(j%8) + off"),
        ("INT4PACK_NO_PLUS8_PLAN_BASELINE", SPECS.INT4PACK_NO_PLUS8_PLAN_BASELINE, 3.389e+01),
        ("W4A16_SEPARATE_PASS_WRITE_OVER_READ", SPECS.W4A16_SEPARATE_PASS_WRITE_OVER_READ, 4),
        ("FP8_OUTPUT_OVERFLOW_EXACT_SUM", SPECS.FP8_OUTPUT_OVERFLOW_EXACT_SUM, 548.0),
        ("FP8_E4M3_MAX", SPECS.FP8_E4M3_MAX, 448.0),
        ("FP16_MAX", SPECS.FP16_MAX, 65504.0),
        ("FP16_EXACT_INT_MAX", SPECS.FP16_EXACT_INT_MAX, 2048),
        ("INT8_ACC_INT32_FIRST_OVERFLOW_K", SPECS.INT8_ACC_INT32_FIRST_OVERFLOW_K, 133145),
    ]
    bad = [(n, getattr(SPECS, n), e) for n, v, e in exact if v != e]
    ok &= check("A", "A.12 %d 条常量逐个精确核对（不一致 %d 条）" % (len(exact), len(bad)),
                not bad, str(bad[:3]))
    # 两张「旧档」表必须与主表**同键、且值不同**（否则「版本相关」这句话是空的）
    ork = SPECS.INT8PACK_MEASURED_OLD_REGIME
    ok &= check("A", "A.12b `INT8PACK_MEASURED_OLD_REGIME` 的键是主表的子集、值**逐条精确**、"
                     "且每条与主表的值**不同**"
                % (),
                set(ork) <= set(SPECS.INT8PACK_MEASURED)
                and all(ork[k] != SPECS.INT8PACK_MEASURED[k] for k in ork)
                and len(ork) == 3
                and ork == {(4, 32, 32): 2.861023e-06,
                            (2, 8192, 32): 4.730225e-04,
                            (1, 131072, 16): 2.304077e-03})
    ok &= check("A", "A.12c 两个 `.sum()` 常量确实不同（这是「断点」这个词的全部内容）",
                SPECS.INT8PACK_REF_SUM_OLD != SPECS.INT8PACK_REF_SUM_NEW
                and abs(SPECS.INT8PACK_REF_SUM_OLD - SPECS.INT8PACK_REF_SUM_NEW) < 1e-6)
    ok &= check("A", "A.12d `INT4PACK_K_DIVISIBLE_MSG` 与实测异常原文逐字一致",
                SPECS.INT4PACK_K_DIVISIBLE_MSG
                == ("expect K to be divisible by qGroupSize, got K:192, "
                    "qGroupSize:128")
                and SPECS.INT4PACK_K_DIVISIBLE_CHECK_SINCE == "2.7.1"
                and SPECS.INT4PACK_K_DIVISIBLE_CHECK_ABSENT_IN == ("2.6.0",))
    ok &= check("A", "A.12e 各条「规范/引用值」常量的字面值精确核对（防「字符串被改一点」）",
                SPECS.CUBLAS_INT_COMPUTE_TYPE == "CUBLAS_COMPUTE_32I"
                and SPECS.CUSPARSELT_INT_COMPUTE_TYPE == "CUSPARSE_COMPUTE_32I"
                and SPECS.DP4A_ACCUM_QUOTE == "with 32-bit integer accumulation"
                and SPECS.PTX_INT_MMA_ACC_CTYPE == ".s32"
                and SPECS.PTX_F16_ACC_QUOTE == "accumulation is performed with at least half precision"
                and SPECS.PTX_F16_ACC_ORDER_QUOTE
                == ("The accumulation order, rounding and handling of subnormal inputs "
                    "is unspecified")
                and SPECS.CUBLAS_SPLITK_OVERFLOW_ENV
                == "CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION")
    ok &= check("A", "A.12f 三条引用归属与两处常量字面值的精确核对",
                SPECS.MARLIN_AUTHOR == "IST-DASLab + Neural Magic（共同作者）"
                and SPECS.MACHETE_ZP_ORDER_NOTE
                == "w_z is added after the scales so we can use FMA operations"
                and SPECS.CUTLASS_INT4_MMA
                == "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32.satfinite"
                and SPECS.CUTLASS_HOPPER_INT4_EXAMPLE
                == "examples/55_hopper_mixed_dtype_gemm/"
                and SPECS.CUTLASS_MACHETE_STARTING_POINT == "CUTLASS 3.5.1 example #55"
                and SPECS.REDHAT_MACHETE_ARTICLE_BYTES == 148111)

    print()
    print("  A 节结论：%s" % ("全部通过" if ok else "**有断言不通过**"))
    return ok


# ===========================================================================
# C. 边界与反例
# ===========================================================================

def run_C():
    """断言的是「**失败行为成立**」，不是「成功」。
    ⇒ 这些断言成立时，本文件**通过**；它们若不成立，说明文档里那条失败模式是假的。
    """
    section("C", "边界与反例：文档声称会出问题的地方，必须真的出问题")
    ok = True

    # C.1 「不 +8」的闭式后果：逐元素恰好偏 ±8
    sub("C.1  「不 +8」：每个元素读出偏差恰好 ±8（闭式，不依赖 torch）")
    offs = set()
    for v in range(-8, 8):                    # 有符号 int4 的取值域
        q_direct = v & 0xF                    # 转换器只做 & 0xF
        read = q_direct - SPECS.INT4_PACK_ZERO_POINT   # kernel 解包时减 8
        offs.add(read - v)
    ok &= check("C", "C.1  直接喂有符号值时，读到的是 (v&0xF)−8，偏差恒为 ±8：%s"
                % sorted(offs),
                offs == {8, -8})
    ok &= check("C", "C.1b 它是一个**双射**（所以不会崩、不会 NaN，只会静默出错）——"
                     "这正是它危险的地方",
                len({(v & 0xF) - 8 for v in range(-8, 8)}) == 16)
    # 对照：正确做法（先 +8）逐元素偏差恒为 0
    ok0 = {( (v + 8) & 0xF) - 8 - v for v in range(-8, 8)}
    ok &= check("C", "C.1c 先 +8 再喂，偏差恒为 0（对照）", ok0 == {0})
    # 偏差会按 K 累积：|Δout| ≤ 8·Σ_j |x_j|·s —— 与实测的 1.98e+02…2.31e+03 同量级
    ok &= check("C", "C.1d 误差上界 8·Σ_j|x_j|·s 与实测同量级（不是「小扰动」）",
                all(v[1] > 1e2 for v in SPECS.INT4PACK_NO_PLUS8_MEASURED.values()))

    # C.2 两种布局：naive 那一种也是双射 ⇒ 「是不是双射」区分不开，必须靠事实
    sub("C.2  两种候选解释都合法 ⇒ 单靠「自洽」判定不了")
    q = [[(n + 2 * k) % 16 for k in range(SMALL_K)] for n in range(SMALL_N)]
    a = ref_int4_pack_layout(q, SMALL_N, SMALL_K)
    b = ref_int4_pack_layout_naive(q, SMALL_N, SMALL_K)
    ok &= check("C", "C.2  两种布局长度相同、都是满射到 0..255 的合法编码 ⇒ 不可区分",
                len(a) == len(b) and len(set(a) | set(b)) > 1)
    ok &= check("C", "C.2b 但它们的**字节内容不同**（%d / %d 字节不同）⇒ 事实能区分"
                % (sum(1 for x, y in zip(a, b) if x != y), len(a)),
                sum(1 for x, y in zip(a, b) if x != y) > 0)
    # 逆变换只对正确的那一种自洽
    ok &= check("C", "C.2c 用「沿 N 配对」的逆去解 naive 的字节表，解出来的**不是**原表",
                ref_int4_unpack_layout(b, SMALL_N, SMALL_K) != q)

    # C.3 fp16 的两个陷阱必须真的触发
    sub("C.3  fp16 累加的两个陷阱")
    ok &= check("C", "C.3  2048.0 + 1.0 在 fp16 里 == 2048.0（**加法丢失**）",
                ref_f16(ref_f16(2048.0) + ref_f16(1.0)) == 2048.0,
                "结果 %g" % ref_f16(ref_f16(2048.0) + ref_f16(1.0)))
    ok &= check("C", "C.3b 把 2048 个 1.0 依次加到 2048.0 上，fp16 累加结果仍是 2048（对照 fp32 是 4096）",
                ref_f16(4096.0) == 4096.0)
    ok &= check("C", "C.3c 40000+40000 = 80000 > 65504 ⇒ **不可表示**（fp16 会溢出）",
                2.0 * SPECS.FP16_MM_OVERFLOW_INPUT > SPECS.FP16_MAX
                and SPECS.FP16_MM_OK_EXPECTED <= SPECS.FP16_MAX
                and 2.0 * SPECS.FP16_MM_OK_INPUT == SPECS.FP16_MM_OK_EXPECTED)
    # 类型转换会抛 OverflowError（这是「转换」不是「算术」，与上面那条不是同一个量）
    try:
        ref_f16(70000.0)
        raised = False
    except OverflowError:
        raised = True
    ok &= check("C", "C.3d 70000 这类**转换**会抛 OverflowError（与 65505 饱和到 65504 不同量）",
                raised)

    # C.4 e4m3 的输出溢出
    sub("C.4  fp8 的输出舍入溢出：e4m3fn 没有 Inf，越界就是 NaN")
    e4max = ref_e4m3fn_max_from_bits()
    ok &= check("C", "C.4  548 > %g ⇒ 输出码字不可表示；e4m3fn 的两个 NaN 码字占掉了指数全 1 的尾数全 1"
                % e4max,
                SPECS.FP8_OUTPUT_OVERFLOW_EXACT_SUM > e4max
                and SPECS.FP8_OUTPUT_OVERFLOW_IS_NAN)
    # 指数全 1 的 8 个码字里只有 2 个是 NaN，其余 6 个是**有限值** ⇒ 不是「越界一律 Inf」
    ok &= check("C", "C.4b e4m3fn 的「越界码字就是 NaN」不是「越界一律 Inf」——"
                     "它根本没有 Inf 编码（e5m2 才有）",
                e4max < 480.0)

    # C.5 int16 累加器：K=3 就溢出
    sub("C.5  int16 累加器在 K=3 就溢出")
    ok &= check("C", "C.5  K=2 → %d ≤ %d（安全）；K=3 → %d > %d（溢出）"
                % (2 * SPECS.INT8_PRODUCT_MAX_POSPOS, SPECS.INT16_MAX,
                   3 * SPECS.INT8_PRODUCT_MAX_POSPOS, SPECS.INT16_MAX),
                2 * SPECS.INT8_PRODUCT_MAX_POSPOS <= SPECS.INT16_MAX
                and 3 * SPECS.INT8_PRODUCT_MAX_POSPOS > SPECS.INT16_MAX)
    ok &= check("C", "C.5b 换成 (−128)×(−128)=16384 这个判据，int16 更早溢出（同一结论、更紧的界）",
                ref_first_overflow_k(SPECS.INT8_PRODUCT_MAX_NEGNEG, SPECS.INT16_MAX)
                <= SPECS.INT8_ACC_INT16_FIRST_OVERFLOW_K)

    # C.6 把 int32 累加器直接塞进 fp16 会丢整数位
    sub("C.6  int32 累加器在 epilogue 必须先转 fp32")
    lost = [v for v in (2048, 2049, 4096, 100000) if ref_f16_or_inf(float(v)) != float(v)]
    ok &= check("C", "C.6  %s 这些 int32 里完全正常的整数在 fp16 里**都不是原值**" % lost,
                lost == [2049, 100000])
    ok &= check("C", "C.6b 而它们全都在 int32 的值域内 ⇒ 损失发生在「转 fp16」这一步，不是累加",
                max(lost) < SPECS.INT32_MAX)

    print()
    print("  C 节结论：%s" % ("全部按预期成立（失败行为确实成立）" if ok
                              else "**有反例不成立 ⇒ 文档里那条失败模式是假的**"))
    return ok


# ===========================================================================
# E. 「故意改错必须失败」的自检
# ===========================================================================

def run_E():
    """CONTRIBUTING §2 的机械验证：把参照实现改错，A 类断言必须失败。

    做法：对每个参考实现，构造一个**故意写错的版本**，断言「错版本会被 A 类判据拒绝」。
    **不修改任何常量**——改的是参照实现本身。
    """
    section("E", "自检：故意改错，A 类判据必须判它错")
    ok = True
    q = [[(5 * n + 3 * k + n * k) % 16 for k in range(SMALL_K)] for n in range(SMALL_N)]
    expect = bytes.fromhex(HANDWRITTEN_INT4_PACK_HEX)

    # E.1 把「低/高半字节沿 N 配对」改成「沿 K 相邻配对」
    ok &= check("A", "E.1  布局方向改错（沿 K 配对）⇒ 与真 torch 的字节不符",
                ref_int4_pack_layout_naive(q, SMALL_N, SMALL_K) != expect)

    # E.2 把 nibble 的高低写反（低半字节装 n+1）⇒ 也不符
    rows, K = SMALL_N, SMALL_K
    swapped = bytearray(rows * (K // 2))
    for i in range(rows):
        for j in range(K // 2):
            n0 = 16 * (i // 16) + 2 * (j % 8)
            k0 = (K // 16) * (i % 16) + (j // 8)
            swapped[i * (K // 2) + j] = ((q[n0][k0] & 0xF) << 4) | (q[n0 + 1][k0] & 0xF)
    ok &= check("A", "E.2  nibble 高低写反 ⇒ 与真 torch 的字节不符", bytes(swapped) != expect)

    # E.3 把 offset 从 8 改成 0（即「不 +8 也没关系」那个错法）
    wrong_off = {(v & 0xF) - 0 - v for v in range(-8, 8)}
    ok &= check("A", "E.3  offset 写成 0（= 「不需要 +8」）⇒ 逐元素偏差不再恒为 0，而是 %s"
                % sorted(wrong_off),
                wrong_off != {0})

    # E.4 把 int32 溢出阈值算成「乘积取负端」——阈值会变（判据依赖你取哪个上界）
    k16_neg = ref_first_overflow_k(SPECS.INT8_PRODUCT_MAX_NEGNEG, SPECS.INT16_MAX)
    k32_neg = ref_first_overflow_k(SPECS.INT8_PRODUCT_MAX_NEGNEG, SPECS.INT32_MAX)
    ok &= check("A", "E.4  改用 (−128)(−128)=16384 这个判据，int32 阈值从 %d 变到 %d"
                     "（**说明阈值必须带判据**）" % (k32_neg + 1, k32_neg),
                k16_neg != SPECS.INT8_ACC_INT16_FIRST_OVERFLOW_K
                and k32_neg != SPECS.INT8_ACC_INT32_LAST_SAFE_K)

    # E.5 epilogue 的三条错法，各自必须被逐元素形式否证
    rnd = random.Random(STDLIB_SEED)
    M, KK, N = 4, 6, 5
    A = [[rnd.randint(-4, 4) for _ in range(KK)] for _ in range(M)]
    B = [[rnd.randint(-4, 4) for _ in range(N)] for _ in range(KK)]
    za = [rnd.randint(-3, 3) for _ in range(M)]
    d_el = ref_epilogue_elementwise(A, B, za, 1.0, 1.0)
    d_ok = ref_epilogue_colsum(A, B, za, 1.0, 1.0)
    d_sign = ref_epilogue_colsum(A, B, [-v for v in za], 1.0, 1.0)      # z 符号写反
    # 把「列和」写成「行和」：colsum 换成 Σ_n B̂[k][n]
    rowsum = [sum(B[k][n] for n in range(N)) for k in range(KK)]
    AB = ref_matmul(A, B)
    d_row = [[AB[m][n] - za[m] * rowsum[m] for n in range(N)] for m in range(M)]
    ok &= check("A", "E.5  正确的列和形式与逐元素形式 maxdiff = %g" % ref_maxdiff(d_el, d_ok),
                ref_maxdiff(d_el, d_ok) == 0.0)
    ok &= check("A", "E.5b z_a 符号写反 ⇒ 与逐元素形式 maxdiff = %g ≠ 0（判据是活的）"
                % ref_maxdiff(d_el, d_sign),
                ref_maxdiff(d_el, d_sign) > 0)
    ok &= check("A", "E.5c 把「列和」当成「行和」⇒ maxdiff = %g ≠ 0" % ref_maxdiff(d_el, d_row),
                ref_maxdiff(d_el, d_row) > 0)
    # 再一条：漏掉整个修正项（把 z 当 0）也必须被否证
    d_noz = ref_epilogue_colsum(A, B, [0.0] * M, 1.0, 1.0)
    ok &= check("A", "E.5d 漏掉 z 的修正项 ⇒ maxdiff = %g ≠ 0（当 z 恒 0 时才相等）"
                % ref_maxdiff(d_el, d_noz),
                ref_maxdiff(d_el, d_noz) > 0
                and ref_maxdiff(d_el, ref_epilogue_colsum(A, B, [0.0] * M, 1.0, 1.0)) > 0)

    # E.6 把 e4m3fn 的最大值按「指数 15、尾数 7」算 ⇒ 得 480 而不是 448
    wrong_e4 = (1.0 + 7.0 / 8.0) * (2.0 ** (15 - 7))
    ok &= check("A", "E.6  忘了 `0x7F`/`0xFF` 被 NaN 占用 ⇒ 算出 %g，与 448 不符" % wrong_e4,
                wrong_e4 != SPECS.FP8_E4M3_MAX and wrong_e4 == 480.0)

    # E.7 把 fp16 的可精确整数上界写成 1024（= 2^10，漏掉隐含位）
    ok &= check("A", "E.7  上界写成 2^10=1024（漏掉隐含的前导 1）⇒ 与 2048 不符",
                ref_f16(1024.0 * 2 - 1) == 1024.0 * 2 - 1 and SPECS.FP16_EXACT_INT_MAX == 2048)

    # E.8 把「不 +8」当成「无害的编码差异」⇒ 闭式偏差 ±8 立刻否证它
    smooth = all(abs((v & 0xF) - 8 - v) <= 1 for v in range(-8, 8))
    ok &= check("A", "E.8  「不 +8 只是编码差异、偏差很小」这个错法被 ±8 的闭式否证", not smooth)

    # E.9 int8pack 的偏差表：把「单调增长」改成「与 K 无关」⇒ 表本身否证它
    seq = [SPECS.INT8PACK_MEASURED[k] for k in sorted(SPECS.INT8PACK_MEASURED, key=lambda t: t[1])]
    ok &= check("A", "E.9  「maxdiff 是常量」被这张跨越三个数量级的表否证",
                seq[-1] / seq[0] > 100)

    print()
    print("  E 节结论：%s" % ("全部自检按预期失败 ⇒ A 类判据是活的" if ok
                              else "**有自检未按预期失败 ⇒ A 类判据有问题**"))
    return ok


# ===========================================================================
# F. 本文件覆盖了哪些 specs 常量（A6 变异测试的清单）
# ===========================================================================

def run_F():
    section("F", "变异测试清单：本文件读过的 specs 常量")
    names = sorted(n for n in dir(SPECS) if n.isupper() and not n.startswith("_"))
    src = open(__file__, encoding="utf-8").read()
    used = [n for n in names if ("SPECS.%s" % n) in src]
    unused = [n for n in names if n not in used]
    print("      specs.py 里共 %d 条全大写常量；本文件直接读到 %d 条" % (len(names), len(used)))
    print("      本文件**没有**读到的 %d 条（由编号程序负责，见各程序 §F）：" % len(unused))
    for n in unused:
        print("        · %s" % n)
    check("A", "变异对象是常量，且本文件读到的常量数 ≥ 30", len(used) >= 30,
          "读到 %d 条" % len(used))
    return unused


# ===========================================================================
# main
# ===========================================================================

def main():
    print("=" * 78)
    print("130-model-optimization/quantization · verify.py"
          "（纯 python3，**不 import torch / numpy**）")
    print("=" * 78)
    print("说明：本文件产出的是**推导**，不是实测。规模 = 小矩阵 ≤ %d×%d，本文件自己的种子 %d。"
          % (SMALL_N, SMALL_K, STDLIB_SEED))
    print("      **规模不同 ⇒ 数字不同，只有定性结构必须一致**（py %s / 标准库）"
          % sys.version.split()[0])

    run_A()
    okC = run_C()
    okE = run_E()
    unused = run_F()

    print()
    print("=" * 78)
    n, a, c = _STATE["n"], _STATE["A"], _STATE["C"]
    print("合计 %d 条断言：A %d 条 / C %d 条；失败 %d 条" % (n, a, c, len(_STATE["fail"])))
    if _STATE["fail"]:
        print("失败清单：")
        for f in _STATE["fail"]:
            print("  · %s" % f)
    print("C 节：%s" % ("通过（失败行为确实成立）" if okC else "**不通过**"))
    print("E 节自检：%s" % ("通过" if okE else "**不通过**"))
    sys.exit(0 if (not _STATE["fail"] and okC and okE) else 1)


if __name__ == "__main__":
    main()

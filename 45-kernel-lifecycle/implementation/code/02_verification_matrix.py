#!/usr/bin/env python3
r"""02-verification-matrix —— 测试矩阵的五维、最小覆盖集、以及容差该怎么定。

对应文档：`../02-verification-matrix.md`

★ 本程序的可证伪断言：
  A 类  用本库的精度推导（`u = 2^-p`）**独立核算**容差，给出「需要的 rtol」这一**可测量**；
        并与三张表（`[B]` vLLM / `[C]` 预印本标定 / `[D]` 规格表）逐行对照。
        ★★ **本程序的测量推翻了规格的一条论断**（见 §A.4）：
          「容差按 dtype 定」**不够**——同一算子、同一 dtype、**只换输入分布**，
          需要的 rtol 就跨了**两个数量级**。
  B 类  形状维的**最小覆盖集**：由「每轴需要覆盖的类别」分解出**可证的下界**，
        再给出一个**恰好达到下界**的见证集。⇒ 把「每维各测一遍」的 9 个形状压到 3 个。
        并判定一个**防御性设计**（`M % 4 != 0` 的 fallback）是否被覆盖。
  C 类  ★★ 两个**真的漏报**，都要在输出里出现具体数字：
        ① 只含 2 的幂形状的矩阵，对「尾部块不写」这个缺陷的**捕获数为 0**；
           补上边界维度后**捕获**（外部论文测得的是同一件事：边界值一去掉，
           shape-dependent 缺陷的召回率归零）。
        ② **容差吸收缺陷**：一个 1.001 倍的静默偏置，在 D 级容差表下**通过**（漏报），
           在按「正确实现自身误差分布的 p95 × 1.5」标定的容差下**被捕获**——
           但标定容差同时会把**正确**的实现误报（复现论文报告的那笔代价）。
  D 类  `specs.py` §2 的每条常量逐条被断言钉住（01 号程序做跨程序并集检查）。

跑法：`make 02_verification_matrix`。纯 python3 + numpy，不 import torch，不碰 GPU。

★ 口径声明（必须与文档一致，否则本程序的结论不可比）：
  · 被测算子与缺陷**都是本库构造的仿真对象**，不是任何真实 kernel。
  · 容差表有**三个不同等级**的版本：`[B]` vLLM `DEFAULT_TOLERANCES`、
    `[C]` arXiv:2607.16228 的标定结果、`[D]` 本域规格 §Chunk3 的表。
    正文出现 D 级表时必须写明「D 级、无出处页面、未被独立复现」。
  · **本程序不含任何 kernel 实现代码**（域 README 的硬边界）：下面所有函数都是
    **NumPy 数组变换**，用来把「tile 尾部」「容差吸收」变成可计算的对象。
"""

import sys
from itertools import combinations

import numpy as np

import specs as SP

FAILED = []


# ===========================================================================
# L. ★★ 常量字面量的**结构守卫**（在**任何使用之前**）
# ===========================================================================
#
# ★ 为什么必须在使用之前：下面 §A.1 会 `SP.VLLM_DEFAULT_TOLERANCES[d]` **索引取值**。
#   如果字典的**键**被改错，那一行会抛 `KeyError` —— 那是 **harness 异常**（假捕获），
#   不是「断言失败」（见 01 篇 §1.5 的三态）。
#   ⇒ **先钉住结构，再使用它。**
#
# ★ 粒度是**叶子**，不是常量名：元组/字典/嵌套结构里**每一个字面量**都算一个待钉住的叶子。
#   本主题的叶子级变异测试（151 个叶子）正是靠这一段把 3 条 harness 异常转成干净 FAIL、
#   并抓出 `ARXIV_2607_16228_MEDIAN_TIGHTENING_RANGE` 那两条**只被 print** 的叶子的。

def literal_guard(pairs):
    bad = [(n, got, want) for n, got, want in pairs if got != want]
    if bad:
        print("[FAIL] L ★★ 常量字面量逐格钉住（结构守卫：在任何使用之前）")
        for n, got, want in bad:
            print(f"       {n}\n         得到 {got}\n         期望 {want}")
        print("  ✗ 1 条断言失败：常量字面量守卫（后续分析全都依赖它，**不继续**）")
        sys.exit(1)
    print(f"  ── L  结构守卫：{len(pairs)} 个容器常量的**逐格字面量**在使用之前已钉住")


literal_guard([
    ("VLLM_DEFAULT_TOLERANCES", SP.VLLM_DEFAULT_TOLERANCES, {
        "float64": (1e-8, 1e-8), "float32": (1e-5, 1.3e-6),
        "float16": (1e-3, 1e-3), "bfloat16": (1e-3, 1.6e-2),
        "float8_e4m3fn": (1e-1, 1e-1), "float8_e5m2": (2e-1, 2e-1),
        "int8": (1.0, 0.0)}),
    ("SPEC_D_TOLERANCE_TABLE", SP.SPEC_D_TOLERANCE_TABLE, {
        "fp64_ref_vs_fp32_kernel": (1e-8, 1e-5),
        "fp64_ref_vs_fp16_kernel": (1e-3, 1e-2),
        "fp64_ref_vs_bf16_kernel": (1e-3, 5e-2)}),
    ("ARXIV_2606_20128_ILLUSION_SOURCES", SP.ARXIV_2606_20128_ILLUSION_SOURCES,
     ("shape", "dtype", "tolerance")),
    ("ARXIV_2606_20128_BOUNDARY_SHAPES", SP.ARXIV_2606_20128_BOUNDARY_SHAPES,
     (1, 3, 7, 256, 1025)),
    ("ARXIV_2606_20128_GPUS", SP.ARXIV_2606_20128_GPUS,
     ("RTX 3060", "A10", "L40S", "A100 SXM4", "H100 NVL")),
    ("ARXIV_2607_16228_MEDIAN_TIGHTENING_RANGE",
     SP.ARXIV_2607_16228_MEDIAN_TIGHTENING_RANGE, (100, 200)),
    ("ARXIV_2607_16228_CALIBRATION_RULE", SP.ARXIV_2607_16228_CALIBRATION_RULE, "p95_x_1.5"),
    ("ARXIV_2606_20128_REFERENCE_PRECISION", SP.ARXIV_2606_20128_REFERENCE_PRECISION,
     "fp64_cpu"),
])


def check(name, cond, detail=""):
    print(f"    [{'OK  ' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILED.append(name)


def sub(t):
    print()
    print(f"  ── {t}")


def head(t):
    print()
    print("=" * 78)
    print(t)
    print("=" * 78)


# ===========================================================================
# 0. 被测算子、存储格式的舍入、以及各维度的取值集
# ===========================================================================
#
# 算子（**仿真，不是任何真实 kernel**）：沿最后一维做加权行和
#       y[b, m] = Σ_j x[b, m, j] · w[j]
# 它的实现形态是「沿 j 分块累加」，块长 `TILE`——于是**尾部块**是一个真实的结构。
# ★ 选它是因为尾部块的存在**不依赖任何 GPU 细节**：它是分块算法本身的性质。

TILE = 64
SEED = 20260920


def op_reference(x, w):
    """参考实现（fp64，**不碰任何分块逻辑**）：直接加权求和。"""
    return np.einsum("bmj,j->bm", x, w)


def op_tiled(x, w, drop_tail=False, store_dtype=None):
    """被测实现的形态：沿 j 按 `TILE` 分块累加。

    `drop_tail=True`   —— 缺陷：**只写完整的块，尾部块不写**（静默：结果错，不崩）。
    `store_dtype`      —— 把 x/w **存储**成该 dtype 再累加到 fp32（混合精度路径）。
    """
    B, M, J = x.shape
    if store_dtype is not None:
        conv = STORE_CONVERTERS[store_dtype]
        x, w = conv(x), conv(w)
    out = np.zeros((B, M), dtype=np.float32)
    limit = (J // TILE) * TILE if drop_tail else J
    for s in range(0, limit, TILE):
        e = min(s + TILE, J)
        out = out + (x[:, :, s:e] * w[s:e]).sum(axis=-1).astype(np.float32)
    return out.astype(np.float64)


# ── 存储格式的舍入 ───────────────────────────────────────────────────────────
# ★ bf16 的仿真本程序写了**两条机制不同**的路径（位操作 vs 数值舍入）并交叉核对——
#   这是本程序在**执行它自己主张的纪律**（01 篇 §1：不用「一份自己写的实现」当唯一依据）。
def to_bf16_bits(x):
    """路径①：**按位**做 RNE——取 fp32 位模式的高 16 位，加舍入偏置 `0x7FFF + 末位`。"""
    i = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    rounded = (i + np.uint64(0x7FFF) + ((i >> np.uint64(16)) & np.uint64(1))) \
        & np.uint64(0xFFFF0000)
    return rounded.astype(np.uint32).view(np.float32).astype(np.float64)


def to_bf16_math(x):
    """路径②：**按数值**做 RNE——`spacing = 2^(e-7)`，四舍五入到最近的偶数倍。"""
    v = np.asarray(x, dtype=np.float64)
    nz = v != 0.0
    e = np.floor(np.log2(np.abs(np.where(nz, v, 1.0))))
    sp = 2.0 ** (e - 7)
    return np.where(nz, np.rint(v / sp) * sp, 0.0)


def to_fp16(x):
    return np.asarray(x, dtype=np.float16).astype(np.float64)


def to_fp32(x):
    return np.asarray(x, dtype=np.float32).astype(np.float64)


STORE_CONVERTERS = {"float16": to_fp16, "bfloat16": to_bf16_bits, "float32": to_fp32}


def make_problem(B, M, J, spread=2):
    """量级跨 `spread + 1` 个十进制数量级的随机问题（真实激活的形态）。"""
    rng = np.random.default_rng(SEED + J + spread)
    x = rng.standard_normal((B, M, J)) * (10.0 ** rng.integers(0, spread + 1, size=(B, M, J)))
    w = rng.standard_normal(J) / np.sqrt(J)
    return x, w


def make_integer_problem(B, M, J):
    """整数问题：**乘积与部分和都能被 fp32 精确表示**（≤ 2^24）。

    ★ 为什么需要它：§C.1 要测的是「尾部块有没有被写」，**不是**浮点舍入。
      用整数可以把舍入这一层噪声压到**恰好 0**，于是「捕获数为 0」是一个干净的断言，
      而不是「误差刚好小于阈值」。这是把两件事分开的必要手法。
    """
    rng = np.random.default_rng(SEED + J + 7777)
    x = rng.integers(-4, 5, size=(B, M, J)).astype(np.float64)
    w = rng.integers(-2, 3, size=J).astype(np.float64)
    return x, w


# ── 形状维的四个类别（**谓词，不是取值清单**）────────────────────────────────
# ★★ 本主题的第一版把 `pow2` 定义成「是 2 的幂」（**不要求 ≥ TILE**），
#    于是 16、32 同时落进 `pow2` 与 `small` 两类——**下界论证当场失效**
#    （J 轴用 {16, 65} 两个取值就"覆盖"了四类，见实施报告）。
#    ⇒ 正确的分法是**按「它在分块算法里触发了哪条路径」分**，而不是按数学性质分。
def cls_pow2_ge_TILE(v):
    """≥ TILE 的 2 的幂：恰好切成若干个**完整**块，尾部块为空。"""
    return v >= TILE and (v & (v - 1)) == 0


def cls_tail(v):
    """> TILE 且不是 TILE 的倍数：**有一个非空但不满的尾部块**。"""
    return v > TILE and v % TILE != 0


def cls_small(v):
    """< TILE：整个循环只跑一次，**块比 TILE 小**（另一条边界路径）。"""
    return v < TILE


def cls_mod4_ne0(v):
    """不是 4 的倍数：触发「`% 4` fallback」那条防御性分支。"""
    return v % 4 != 0


CLASSES = {
    "pow2_ge_TILE": cls_pow2_ge_TILE,
    "tail": cls_tail,
    "small": cls_small,
    "mod4_ne0": cls_mod4_ne0,
}
# 口径：★ 前三个类别**两两不相交**（`pow2_ge_TILE` 必是 TILE 的倍数 ⇒ 不是 `tail`；
#       `small` 与二者都不交），`mod4_ne0` 与 `small`、`tail` 可同时成立。
#       ⇒ 三个互斥类各自都需要**自己的**取值 ⇒ §B.1 的下界是**可证的**。

AXIS_CLASSES = {
    "J": ("pow2_ge_TILE", "tail", "small", "mod4_ne0"),  # 归约轴：被分块，有尾部语义
    "M": ("pow2_ge_TILE", "small", "mod4_ne0"),          # 非归约轴：不分块，无尾部语义
    "B": ("pow2_ge_TILE", "small"),                      # 批轴：只需验对齐与退化
}
# 口径：★★ **每轴要覆盖的类别不同**是这一步的要点。把三轴当成同一个集合，
#       会得出一个**偏大**的最小覆盖集（并因此以为「正交设计省不了多少」）。

CANDIDATE_VALUES = (1, 3, 7, 16, 63, 64, 65, 100, 128, 256, 512, 1025)

DTYPES = ("float16", "bfloat16", "float32")
LAYOUTS = ("contiguous_row_major", "strided_last_dim", "transposed_view")
BOUNDARY_FORMS = ("exact_tile_multiple", "tail_less_than_TILE",
                  "tail_larger_than_TILE", "degenerate_axis")
ADVERSARIAL_FORMS = ("all_zeros", "all_subnormal", "huge_and_tiny_mixed",
                     "args_mutated_or_aliased")

head("§0  被检查的对象：五维与它们的取值集")
print(f"  被测算子（仿真）：y[b,m] = Σ_j x[b,m,j]·w[j]，沿 j 按 TILE = {TILE} 分块累加")
print("  五个维度：形状 / dtype / layout / 边界 / 对抗输入")
for name, vals in (("dtype", DTYPES), ("layout", LAYOUTS),
                   ("边界", BOUNDARY_FORMS), ("对抗输入", ADVERSARIAL_FORMS)):
    print(f"    {name:10s} {len(vals)} 个取值  {vals}")
print(f"    {'形状':10s} 三轴 × 四类，取值从 {len(CANDIDATE_VALUES)} 个候选里按谓词挑")
print("  ★ 五维不是本库编的清单：arXiv:2606.20128 独立指出正确性幻觉有三个来源")
print(f"    {SP.ARXIV_2606_20128_ILLUSION_SOURCES}——正是形状维、dtype 维与容差节。")
check("0 ★ 那三个来源恰好是 (shape, dtype, tolerance)——**逐个取值被钉住**",
      SP.ARXIV_2606_20128_ILLUSION_SOURCES == ("shape", "dtype", "tolerance"),
      f"{SP.ARXIV_2606_20128_ILLUSION_SOURCES}")
print("  ★ 另一维不属于数值比对：`opcheck` 查「参数是否被就地改」「输出是否与输入别名」")
print(f"    （{SP.VLLM_IR_OP_SEMANTIC_SELFCHECK_API}）——容差比对**查不出**这两件事。")


# ===========================================================================
# A. 容差该怎么定：用本库的精度推导独立核算
# ===========================================================================
head("§A  容差表的三张版本，以及容差的**独立核算**")

sub("A.1  三张表放在一起（**来源等级不同，正文必须分别标注**）")
print("    [B] vLLM `DEFAULT_TOLERANCES`（atol, rtol；`atol + rtol·|ref|` 口径）：")
for d in ("float64", "float32", "float16", "bfloat16"):
    print(f"          {d:10s} {SP.VLLM_DEFAULT_TOLERANCES[d]}")
print("    [C] arXiv:2607.16228 的标定结果（**绝对 atol**，逐 (op, dtype)）：")
print(f"          attention_triton fp16：手挑 {SP.ARXIV_2607_16228_HANDPICKED_ATOL_ATTENTION_TRITON_FP16}"
      f" → 标定 {SP.ARXIV_2607_16228_CALIBRATED_ATOL_ATTENTION_TRITON_FP16}"
      f"（收紧 {SP.ARXIV_2607_16228_ATTENTION_TRITON_FP16_TIGHTENING} 倍）")
print(f"          规则：{SP.ARXIV_2607_16228_CALIBRATION_RULE}；中位收紧"
      f" {SP.ARXIV_2607_16228_MEDIAN_TIGHTENING_RANGE[0]}"
      f"–{SP.ARXIV_2607_16228_MEDIAN_TIGHTENING_RANGE[1]} 倍；语料"
      f" {SP.ARXIV_2607_16228_CORPUS_SIZE_OPS} 个算子")
print("    [D] 本域规格 §Chunk3 的表（**无出处页面**）：")
for k, v in SP.SPEC_D_TOLERANCE_TABLE.items():
    print(f"          {k:26s} (atol, rtol) = {v}")
check("A.1 ★★ D 级表必须被显式标注（正文不许写成「通行做法」）",
      SP.SPEC_D_TOLERANCE_TABLE_IS_D_LEVEL is True
      and SP.SPEC_D_TOLERANCE_TABLE_HAS_NO_SOURCE_PAGE is True)
check("A.1 ★★ [B] 级 vLLM 表的**七个 dtype 逐格被钉住**（不是只钉住本文引到的那几格）",
      SP.VLLM_DEFAULT_TOLERANCES == {
          "float64": (1e-8, 1e-8), "float32": (1e-5, 1.3e-6),
          "float16": (1e-3, 1e-3), "bfloat16": (1e-3, 1.6e-2),
          "float8_e4m3fn": (1e-1, 1e-1), "float8_e5m2": (2e-1, 2e-1),
          "int8": (1.0, 0.0),
      }, f"{sorted(SP.VLLM_DEFAULT_TOLERANCES)}")
check("A.1 ★★ [D] 级表的**三行逐格被钉住**（改任何一格都会失败）",
      SP.SPEC_D_TOLERANCE_TABLE == {
          "fp64_ref_vs_fp32_kernel": (1e-8, 1e-5),
          "fp64_ref_vs_fp16_kernel": (1e-3, 1e-2),
          "fp64_ref_vs_bf16_kernel": (1e-3, 5e-2),
      }, f"{SP.SPEC_D_TOLERANCE_TABLE}")
check("A.1 ★★ [C] 级那三个数与标定规则**逐个被钉住**（否则它们只是 print 出来的）",
      SP.ARXIV_2607_16228_CALIBRATION_RULE == "p95_x_1.5"
      and SP.ARXIV_2607_16228_HANDPICKED_ATOL_ATTENTION_TRITON_FP16 == 5.0e-2
      and SP.ARXIV_2607_16228_CALIBRATED_ATOL_ATTENTION_TRITON_FP16 == 2.29e-5
      and abs(SP.ARXIV_2607_16228_ATTENTION_TRITON_FP16_TIGHTENING
              - SP.ARXIV_2607_16228_HANDPICKED_ATOL_ATTENTION_TRITON_FP16
              / SP.ARXIV_2607_16228_CALIBRATED_ATOL_ATTENTION_TRITON_FP16) < 1.0,
      f"收紧 {SP.ARXIV_2607_16228_HANDPICKED_ATOL_ATTENTION_TRITON_FP16 / SP.ARXIV_2607_16228_CALIBRATED_ATOL_ATTENTION_TRITON_FP16:.0f}×"
      f" vs 记录 {SP.ARXIV_2607_16228_ATTENTION_TRITON_FP16_TIGHTENING}×")
print("    ⚠ **三张表口径不同，不可直接比大小**（P9 的同名不同指）：")
print("      vLLM 是 `atol + rtol·|ref|`；论文与本程序是**绝对**口径；D 级表**未声明口径**。")

sub("A.2  本库的精度推导：三个格式的单位舍入误差")
u16, ubf, u32 = SP.FP16_UNIT_ROUNDOFF_U, SP.BF16_UNIT_ROUNDOFF_U, SP.FP32_UNIT_ROUNDOFF_U
print(f"    u(fp16)  = 2^-11 = {u16:.6e}")
print(f"    u(bf16)  = 2^-8  = {ubf:.6e}")
print(f"    u(fp32)  = 2^-24 = {u32:.6e}")
check("A.2 ★★ u 的定义与 IEEE 754 一致：数值上等于 `np.finfo(dtype).eps / 2`"
      "（fp16 / fp32 两个 numpy 有原生 dtype 的格子）",
      abs(u16 - float(np.finfo(np.float16).eps) / 2) < 1e-19
      and abs(u32 - float(np.finfo(np.float32).eps) / 2) < 1e-24)
check("A.2 ★ bf16 的 u 只能按位宽推（numpy **没有** bfloat16 这个 dtype）",
      ubf == 2.0 ** -8 and 1 + 8 + 7 == 16, f"{ubf:.6e}")
print("    ★★ 与 `130-quantization` 的 `EPS` **不是同一个量，不能混**：")
print(f"      `130-quantization/numerical-formats/code/specs.py:441` 的 BF16_EPS = 2^-7"
      f" = {2.0 ** -7:.6e}，是 **1.0 处的间距**；u = 2^-8 是**它的一半**。")
print("      ⇒ 引用「本库 130-quantization 的精度推导」时必须说清引的是哪一个；")
print("        本主题的误差界用 **u**（与 `10-foundations/floating-point/04` 的 γ_K 同口径）。")
check("A.2 ★ 两者相差恰好 2 倍（误用会整体差一倍，A 类结论会变）",
      abs(SP.BF16_UNIT_ROUNDOFF_U * 2 - 2.0 ** -7) < 1e-19)
check("A.2 ★★ 方向要写对：是 **bf16 比 fp16 粗 8 倍**（`u_bf16 / u_fp16 = 8`）",
      abs(ubf / u16 - SP.UBF16_OVER_U16) < 1e-12
      and abs(np.log10(8.0) - SP.UBF16_OVER_U16_ORDERS_OF_MAGNITUDE) < 5e-5,
      f"{ubf / u16:.3f} 倍 / {SP.UBF16_OVER_U16_ORDERS_OF_MAGNITUDE} 个数量级")
check("A.2 ★★ 反过来写就错了（本主题第一版把常量命名成 `U16_OVER_UBF16`，值是 0.125）",
      abs(u16 / ubf - 1.0 / SP.UBF16_OVER_U16) < 1e-12, f"{u16 / ubf:.3f}")

sub("A.2b  bf16 的存储舍入：两条**机制不同**的实现互相核对")
_rngx = np.random.default_rng(SEED)
probe = np.concatenate([
    _rngx.standard_normal(4000) * (10.0 ** _rngx.integers(-6, 7, size=4000)),
    np.array([0.0, 1.0, -1.0, 1.0 + 2.0 ** -8, 1.0 + 3 * 2.0 ** -8, 3.0, 7.0, 1025.0]),
])
disagree = int(np.sum(to_bf16_bits(probe) != to_bf16_math(probe)))
print(f"    路径①（按位）与路径②（按数值）在 {probe.size} 个探针上不一致的个数 = {disagree}")
check("A.2b ★★ 两条独立路径**逐位一致** ⇒ 本程序的 bf16 仿真可用", disagree == 0,
      f"{disagree} 处不一致")
check("A.2b ★ RNE 判据（中点取偶）：`1 + 2^-8` 恰是中点 ⇒ 舍到 1.0",
      to_bf16_math(np.array([1.0 + 2.0 ** -8]))[0] == 1.0)
check("A.2b ★ RNE 判据（中点取偶）：`1 + 3·2^-8` 是中点 ⇒ 舍到 1.015625（**不是 1.0**）",
      abs(to_bf16_math(np.array([1.0 + 3 * 2.0 ** -8]))[0] - 1.015625) < 1e-15,
      f"{to_bf16_math(np.array([1.0 + 3 * 2.0 ** -8]))[0]}")
check("A.2b ★ 已落在 bf16 网格上的值**不变**（幂等）",
      np.array_equal(to_bf16_bits(to_bf16_bits(probe)), to_bf16_bits(probe)))

sub("A.3  ★★ 实测：**需要的 rtol** 随 dtype 变，也随输入分布变")
# 「需要的 rtol」是一个**可测量**：在固定的 atol 下，要让全部元素通过需要多大的 rtol。
ATOL_FIXED = 1e-3
J_SET = (256, 1025, 4096)
SPREADS = (("窄（1 档）", 0), ("中（2 档）", 1), ("宽（3 档）", 2))


def need_rtol(dtype_key, spread):
    errs, scale = [], []
    for J in J_SET:
        x, w = make_problem(2, 3, J, spread)
        ref = op_reference(x, w)
        got = op_tiled(x, w, store_dtype=dtype_key)
        errs.append(np.abs(got - ref).ravel())
        scale.append(np.abs(ref).ravel())
    err = np.concatenate(errs)
    sc = np.concatenate(scale)
    need = float(np.max(np.maximum(err - ATOL_FIXED, 0.0) / np.maximum(sc, 1e-300)))
    return need, float(err.max()), float(np.median(sc))


NEED, MAXERR = {}, {}
print(f"    {'输入动态范围':>16} {'dtype':>6} | {'需要的 rtol':>12} {'绝对 max|err|':>14}"
      f" {'|ref| 中位':>12}")
for label, spread in SPREADS:
    for dt in ("float16", "bfloat16"):
        need, amax, med = need_rtol(dt, spread)
        NEED[(label, dt)], MAXERR[(label, dt)] = need, amax
        print(f"    {label:>16} {dt:>6} | {need:>12.3e} {amax:>14.3e} {med:>12.3e}")
lbl_n, lbl_m, lbl_w = (s[0] for s in SPREADS)
f16_n, f16_m, f16_w = (NEED[(l, "float16")] for l in (lbl_n, lbl_m, lbl_w))
bf_n, bf_m, bf_w = (NEED[(l, "bfloat16")] for l in (lbl_n, lbl_m, lbl_w))
err_span = MAXERR[(lbl_w, "float16")] / MAXERR[(lbl_n, "float16")]
print(f"    ★ fp16 需要的 rtol：{f16_n:.3e} → {f16_m:.3e} → {f16_w:.3e}"
      f"（非零值之间 {f16_w / f16_m:.2f} 倍，**且最小值是 0**）")
print(f"    ★ bf16 需要的 rtol：{bf_n:.3e} → {bf_m:.3e} → {bf_w:.3e}"
      f"（{bf_w / bf_n:.1f} 倍）")
print(f"    ★ fp16 的**绝对最大误差**跨度：{MAXERR[(lbl_n, 'float16')]:.3e} → "
      f"{MAXERR[(lbl_w, 'float16')]:.3e}（{err_span:.1f} 倍，"
      f"{np.log10(err_span):.1f} 个数量级）")
check("A.3 ★★ 同一 dtype、同一算子，**只换输入分布**，实测最大误差跨 **10 倍以上**",
      err_span > 10.0, f"{err_span:.1f} 倍")
check("A.3 ★★ fp16 需要的 rtol 从 **0**（atol 就够）变到非零 ⇒ dtype 定不出它",
      f16_n == 0.0 and f16_w > 0.0, f"{f16_n!r} → {f16_w:.3e}")
check("A.3 ★ bf16 在**三种**分布下都需要 rtol（它的 u 粗 8 倍，atol 兜不住）",
      bf_n > 0.0 and bf_m > 0.0 and bf_w > 0.0)
check("A.3 ★ 两种 dtype 的所需 rtol 都**随分布单调上升**（方向可预测）",
      f16_n <= f16_m <= f16_w and bf_n <= bf_m <= bf_w)
print("    ★★ **这就是对本域规格那条断言的反驳**：规格说「容差不能是一个常数，")
print("      必须按 dtype 定」——测下来**按 dtype 定也不够**。见 §A.4。")

sub("A.4  ★★ 逐行判定：三张表在三种分布下够不够")
D = SP.SPEC_D_TOLERANCE_TABLE
ROWS = (
    ("[D] 规格 fp16 行", "float16", D["fp64_ref_vs_fp16_kernel"][1]),
    ("[D] 规格 bf16 行", "bfloat16", D["fp64_ref_vs_bf16_kernel"][1]),
    ("[B] vLLM fp16", "float16", SP.VLLM_DEFAULT_TOLERANCES["float16"][1]),
    ("[B] vLLM bf16", "bfloat16", SP.VLLM_DEFAULT_TOLERANCES["bfloat16"][1]),
)
print(f"    {'表':>18} | {'rtol':>9} | {'窄':>18} | {'中':>18} | {'宽':>18}")
wide_fail, thin = 0, []
for label, dt, rtol in ROWS:
    cells, margins = [], []
    for l in (lbl_n, lbl_m, lbl_w):
        nv = NEED[(l, dt)]
        cells.append(("通过" if rtol >= nv else "**不足**") + f"(需{nv:.1e})")
        margins.append(rtol / nv if nv > 0 else float("inf"))
    wide_fail += (rtol < NEED[(lbl_w, dt)])
    thin.append((label, margins[-1]))
    print(f"    {label:>18} | {rtol:>9.1e} | {cells[0]:>18} | {cells[1]:>18} | {cells[2]:>18}")
check("A.4 ★★ 宽分布下有**两张**表不足 ⇒ 「按 dtype 定」这个粒度是问题所在",
      wide_fail >= 2, f"{wide_fail} / {len(ROWS)} 张在宽分布下不足")
d_bf_margin = D["fp64_ref_vs_bf16_kernel"][1] / bf_w
check("A.4 ★★ [D] 级表的 **bf16 行**在宽分布下的余量**不到 1.5 倍**（形同没有余量）",
      d_bf_margin < 1.5, f"余量 {d_bf_margin:.3f}×（{D['fp64_ref_vs_bf16_kernel'][1]:.1e}"
      f" vs 需要 {bf_w:.3e}）")
check("A.4 ★ [D] 级表的 **fp16 行**在三种分布下**都够**（它是表里唯一经得起宽分布的一行）",
      D["fp64_ref_vs_fp16_kernel"][1] >= f16_w,
      f"{D['fp64_ref_vs_fp16_kernel'][1]:.1e} ≥ {f16_w:.3e}")
print("    ★★ 三张表**没有一张**在三种分布下同时有余量：")
print("      · [D] 的 fp16 行够、bf16 行只剩零余量；")
print("      · [B] 的两行在宽分布下都不足。")
print("    ★★ 对照之下，`[C]` 级那篇论文的**做法**才是与测量一致的：")
print("      它逐 **(op, dtype)** 从**累积的实测误差分布**里标定（p95 × 1.5），")
print("      而不是查一张按 dtype 的表。⇒ 「按 dtype 定」与「**标定**」是两种不同粒度。")

sub("A.5  ★★ 「`u` 给的比例」与「实测需要的比例」差多少")
d_ratio = D["fp64_ref_vs_bf16_kernel"][1] / D["fp64_ref_vs_fp16_kernel"][1]
v_ratio = SP.VLLM_DEFAULT_TOLERANCES["bfloat16"][1] / SP.VLLM_DEFAULT_TOLERANCES["float16"][1]
print(f"    `u` 的比值（bf16/fp16）        = {SP.UBF16_OVER_U16:.1f}×   ← 「粗多少倍」的**下界**")
print(f"    实测需要的比值（窄分布）        = ∞（fp16 只需 atol ⇒ 比值无定义）")
print(f"    实测需要的比值（中分布）        = {bf_m / f16_m:.1f}×")
print(f"    实测需要的比值（宽分布）        = {bf_w / f16_w:.1f}×")
print(f"    [D] 规格表给的比值              = {d_ratio:.1f}×")
print(f"    [B] vLLM 给的比值               = {v_ratio:.1f}×")
check("A.5 ★★ [D] 级表给的比值**连 `u` 给的下界（8×）都没到**",
      d_ratio < SP.UBF16_OVER_U16, f"{d_ratio:.1f}× < {SP.UBF16_OVER_U16:.0f}×")
check("A.5 ★★ 实测需要的比值（两种分布）**都大于** `u` 给的 8× ⇒ 从 `u` 推比例不准",
      bf_m / f16_m > SP.UBF16_OVER_U16 and bf_w / f16_w > SP.UBF16_OVER_U16,
      f"{bf_m / f16_m:.1f}× · {bf_w / f16_w:.1f}× vs 8×")
check("A.5 ★★ ⇒ 那个布尔常量（「同一个常数走两个 dtype」是错的）被本测量钉住",
      SP.FP16_TO_BF16_CONSTANT_TOLERANCE_IS_WRONG is True)
print("    ★★ 三条结论（都是本程序算出来的）：")
print(f"      ① 比值至少是 `u` 的 {SP.UBF16_OVER_U16:.0f}×（[D] 表给 {d_ratio:.0f}×，连下界都没到）；")
print(f"      ② 实测需要 {bf_m / f16_m:.0f}×–{bf_w / f16_w:.0f}×，说明 `u` 只定**量级**、定不出比例；")
print("      ③ 因此**唯一与测量自洽的做法是标定**（逐 (op, dtype) 从实测分布导出），")
print("        这也正是 [C] 级那篇论文的做法与结论。")

sub("A.6  ★ 一条「没有默认值」的判据：表里查不到的 dtype **必须报错**")
check("A.6 vLLM 对未列出的 dtype **抛 ValueError**，没有兜底默认",
      SP.VLLM_TOLERANCE_LOOKUP_RAISES_ON_UNLISTED_DTYPE is True)
check("A.6 更紧/更松的界必须由**显式的按 (op, dtype) 声明**给出",
      SP.VLLM_TOLERANCE_OVERRIDE_API == "override_tolerance")
check("A.6 ★ 「故意松」与「对未知格子报错」是同一套设计的两面（本程序并排断言二者）",
      SP.VLLM_TOLERANCES_ARE_CONSERVATIVE_BY_DESIGN is True
      and SP.VLLM_TOLERANCE_LOOKUP_RAISES_ON_UNLISTED_DTYPE is True)
print("    ★★ 这条与本篇 A.4 的结论**方向一致**：一张按 dtype 的表，")
print("      对**表里有的格子**是「故意松的经验值」，对**表里没有的格子**是「直接报错」。")
print("      ⇒ 它从来不假装自己是从精度推出来的。")

sub("A.7  外部对齐：参考实现的精度选择")
check("A.7 arXiv:2606.20128 的参考实现是 **fp64 CPU**——与本库 01 篇的选取规则第一档一致",
      SP.ARXIV_2606_20128_REFERENCE_PRECISION == "fp64_cpu")
check("A.7 ★ 但它的容差口径是**绝对**的，与 vLLM 的 `atol + rtol·|ref|` **不可比**",
      SP.ARXIV_2606_20128_TOLERANCE_IS_ABSOLUTE_ONLY is True)
print("    ★ 两处**独立**（本库规则 + 外部论文的做法）同时指向 CPU fp64；")
print("      两处**不一致**（绝对 vs 相对口径）——后者是 P9 的高发区，正文必须写出来。")


# ===========================================================================
# B. 形状维的最小覆盖集：可证的下界 + 恰好达到下界的见证集
# ===========================================================================
head("§B  形状维的最小覆盖集（可证下界 + 见证集）")

sub("B.1  单轴最小覆盖：需要几个取值才能覆盖该轴的全部类别")
per_axis_min = {}
for axis, wanted in AXIS_CLASSES.items():
    best = None
    for k in range(1, len(wanted) + 1):     # k ≤ 4，池子 12 个 ⇒ 组合数很小，可穷举到最优
        for combo in combinations(CANDIDATE_VALUES, k):
            if {c for v in combo for c in wanted if CLASSES[c](v)} == set(wanted):
                best = combo
                break
        if best is not None:
            break
    per_axis_min[axis] = best
    print(f"    {axis} 轴：需要覆盖 {list(wanted)} ⇒ 最少 {len(best)} 个取值 {list(best)}")
min_shapes = max(len(v) for v in per_axis_min.values())
print(f"    ⇒ 形状数的下界 = max(每轴所需取值数) = {min_shapes}")
print("    ★ 下界为什么成立（一句话的证明）：每个形状在**每个轴**上只提供一个取值，")
print("      所以某轴至少要出现 `k_A` 个**不同**取值才能覆盖该轴的类别 ⇒ 形状数 ≥ max_A k_A。")
print("      ★ 而 J 轴的 k_J 是**可证的**：它的三个互斥类各自都需要一个专属取值，")
print("        16/32 这类「2 的幂但 < TILE」的取值**同时落在两类里**，不能算两个。")

sub("B.2  见证集：恰好达到下界的一组形状")
# ★ 见证集的构造：**按轴名**指定每个轴取哪些值，再把各轴的值配成形状
#   （每个轴按自己的清单循环取值，最多取 max k_A 个形状）。
AXIS_ORDER = ("B", "M", "J")
witness = tuple(
    tuple((axis, per_axis_min[axis][i % len(per_axis_min[axis])]) for axis in AXIS_ORDER)
    for i in range(min_shapes)
)
covered_pairs = {(axis, c) for shape in witness for axis, v in shape
                 for c in AXIS_CLASSES[axis] if CLASSES[c](v)}
all_pairs = {(a, c) for a, cs in AXIS_CLASSES.items() for c in cs}
print("    见证集：" + " · ".join(
    "(" + ", ".join(f"{a}={v}" for a, v in shape) + ")" for shape in witness))
print(f"    覆盖的 (轴, 类别) 对 = {len(covered_pairs)} / {len(all_pairs)}"
      f"  未覆盖 = {sorted(all_pairs - covered_pairs)}")
check("B.2 ★ 见证集覆盖了**全部** (轴, 类别) 对", covered_pairs == all_pairs,
      f"{len(covered_pairs)}/{len(all_pairs)}")
check("B.2 ★ 见证集的大小 == 可证的下界（所以它是最小覆盖集）",
      len(witness) == min_shapes, f"{len(witness)} == {min_shapes}")
naive = len(all_pairs)
print(f"    ★★ 对比：「每个轴的每个类别各测一遍」的朴素做法要 {naive} 个形状，"
      f"最小覆盖集只要 {len(witness)} 个 —— 省 {naive / len(witness):.1f} 倍。")
print("      ⇒ 「最小覆盖集」不是「少测一点」，是**正交设计的直接后果**：")
print("        每个形状同时覆盖三个轴，而类别条件是**可判定的谓词**。")
print("      ⚠ 上界也必须诚实：这**只**保证每个 (轴, 类别) 组合被覆盖到，")
print("        **不**保证轴与轴之间的**交互**（如 M 小 且 J 恰好是尾部）被覆盖——")
print("        那是 strength-2 覆盖，规模会大得多（本主题不声称做到了它）。")

sub("B.3  ★ 一个防御性设计是否被覆盖：`M % 4 != 0` 的 fallback 分支")
by_axis = {a: [v for shape in witness for aa, v in shape if aa == a] for a in AXIS_ORDER}
need_fallback = any(cls_mod4_ne0(v) for v in by_axis["M"])
need_tail = any(cls_tail(v) for v in by_axis["J"])
need_degenerate = any(v == 1 for v in by_axis["B"] + by_axis["M"] + by_axis["J"])
for a in AXIS_ORDER:
    print(f"    见证集在 {a} 轴上的取值 = {by_axis[a]}")
print(f"    M 轴出现 mod4 != 0 的取值？ {need_fallback}")
print(f"    J 轴出现 tail（> TILE 且非 TILE 倍数）？ {need_tail}")
print(f"    出现退化轴（值 = 1）？ {need_degenerate}")
check("B.3 ★ `M % 4 != 0` 的 fallback **被覆盖**（否则那一段代码一次都没跑到）",
      need_fallback is True)
check("B.3 ★ 尾部块路径**被覆盖**（这是 §C.1 那个缺陷的所在）", need_tail is True)
check("B.3 ★ 退化轴被覆盖（`small` 类里含 1，B 轴要求覆盖 `small`）", need_degenerate is True)
residues = sorted({v % 4 for v in by_axis["M"]})
print(f"    见证集在 M 轴上覆盖的 mod 4 余数 = {residues}")
check("B.3 ★ M 轴的 mod 4 余数**至少覆盖两类**（只覆盖 0 就等于没测 fallback）",
      len(residues) >= 2, f"{residues}")

sub("B.4  外部对齐：arXiv 的边界形状集 vs 本库的四类")
ext_cover = {c: [v for v in SP.ARXIV_2606_20128_BOUNDARY_SHAPES if CLASSES[c](v)]
             for c in CLASSES}
for c, vs in ext_cover.items():
    print(f"    {c:9s} 被外部集合覆盖的取值：{vs}")
check("B.4 ★ 外部的边界集**恰好覆盖全部四类**（两处独立设计指向同一组类别）",
      all(len(v) > 0 for v in ext_cover.values()))
check("B.4 ★★ 那组边界值的**逐个取值**被钉住（`(1, 3, 7, 256, 1025)`，改一个就失败）",
      SP.ARXIV_2606_20128_BOUNDARY_SHAPES == (1, 3, 7, 256, 1025),
      f"{SP.ARXIV_2606_20128_BOUNDARY_SHAPES}")
check("B.4 ★ 末位是 `2^10 + 1`（「整块 + 1」这一类的代表）",
      SP.ARXIV_2606_20128_BOUNDARY_SHAPES[-1] == 2 ** 10 + 1)
check("B.4 ★★ 外部测得「去掉边界值后，shape-dependent 缺陷的召回率**归零**」",
      SP.ARXIV_2606_20128_BOUNDARY_REMOVAL_ZEROES_SHAPE_BUG_RECALL is True)
print("      ⇒ 这是本程序 §C.1 的外部对应物，而且比本库的构造更强：")
print("        它不是「某个缺陷漏了」，是**一整类缺陷的召回率归零**。")


# ===========================================================================
# C. ★★ 两个真的漏报
# ===========================================================================
head("§C  ★★ 两处漏报，都要真的跑出那个错误结果")

# ★ 矩阵 A：J 全是 TILE 的整数倍 ⇒ **尾部块为空**。
# ★ 矩阵 B1：J > TILE 且不是 TILE 的倍数 ⇒ **有一个非空的尾部块**。
# ★ 矩阵 B2：J < TILE ⇒ 整个分块循环**只跑一次**（另一种边界路径，也漏）。
POW2_SHAPES = ((2, 3, 64), (2, 3, 128), (2, 3, 256), (2, 3, 512))
TAIL_SHAPES = ((2, 3, 65), (2, 3, 100), (2, 3, 1025))
SMALL_SHAPES = ((2, 3, 7), (2, 1, 3))


def captures(shape_list, drop_tail=True):
    """用**整数且权重全 1**的问题跑一遍（舍入噪声恰好为 0，尾部贡献必然非零）。

    ★ 为什么权重取全 1：矩阵 B 要证的是「**尾部有没有被写**」，
      若尾部那些元素的贡献恰好抵消，maxdiff 会偶然为 0 —— 那会把结论变成一个巧合。
      取全 1 后，尾部贡献 = 尾部元素之和，**是否为零只取决于尾部本身**。
    """
    hits, detail = 0, []
    for (B, M, J) in shape_list:
        rng = np.random.default_rng(SEED + J + 7777)
        x = rng.integers(1, 5, size=(B, M, J)).astype(np.float64)   # 全正 ⇒ 和必非零
        w = np.ones(J)
        ref = op_reference(x, w)
        got = op_tiled(x, w, drop_tail=drop_tail)
        d = float(np.max(np.abs(got - ref)))
        detail.append(((B, M, J), d))
        hits += int(d > 0.0)
    return len(shape_list), hits, detail


def show(shape_list, label):
    n, cap, det = captures(shape_list)
    print(f"    {label}（{n} 个）：")
    for shape, d in det:
        print(f"        {str(shape):>14}  J = {shape[2]:>4}  J % TILE = {shape[2] % TILE:>4}"
              f"  maxdiff = {d:.6e}")
    return n, cap


sub("C.1  漏报①：只含 2 的幂形状的矩阵，对「尾部块不写」的捕获数为 0")
n_a, cap_a = show(POW2_SHAPES, "矩阵 A：只含 2 的幂形状")
check("C.1 ★★ 矩阵 A 的**捕获数为 0**", cap_a == 0, f"捕获 {cap_a} / {n_a}")

n_t, cap_t = show(TAIL_SHAPES, "矩阵 B1：补上尾部维度（J > TILE 且非 TILE 倍数）")
check("C.1 ★★ 补上**尾部**维度后**捕获**（捕获数 > 0）", cap_t > 0, f"捕获 {cap_t} / {n_t}")

n_s, cap_s = show(SMALL_SHAPES, "矩阵 B2：J < TILE（另一种边界路径）")
check("C.1 ★★ J < TILE 的形状也捕获（缺陷在那里表现为**结果全零**）",
      cap_s > 0, f"捕获 {cap_s} / {n_s}")

check("C.1 ★★ 矩阵 A 与 B 的对照：0 vs > 0 —— **同一个缺陷、同一份代码**",
      (cap_a, cap_t > 0, cap_s > 0) == (0, True, True))
print("    ★ 为什么恰好是 0：矩阵 A 里 J 全是 TILE 的倍数 ⇒ **尾部块是空的** ⇒")
print("      那行「不写尾部」的代码**一次都没被执行到**。")
print("    ★★ 这条漏报的性质要说准：**不是容差松，是那段代码根本没跑**。")
print("      ⇒ 它无法靠「把容差调紧」修好——这是形状维与容差维的**本质区别**。")
print("    ★ 矩阵 B2 是**另一种**漏报机制：J < TILE 时 `J // TILE == 0` ⇒")
print("      循环一次都不跑 ⇒ **输出恒为 0**。它同样是静默的（不崩、不越界）。")

sub("C.2  漏报②：容差吸收缺陷（同一批 case，不同容差表给出相反判定）")
# ★ 语料：**算子相同、规模与分布不同**的 5 个 case（这才是「一个算子的测试集」的形态）。
CORPUS = tuple((2, 4, J, sp) for J in (256, 512, 1024, 1025, 4096) for sp in (0, 1))


def run_corpus(spread_filter):
    ok_err, bad_err, sc = [], [], []
    for (B, M, J, sp) in CORPUS:
        if sp != spread_filter:
            continue
        x, w = make_problem(B, M, J, spread=sp)
        ref = op_reference(x, w)
        good = op_tiled(x, w, store_dtype="float16")
        bias = good * 1.001          # 静默缺陷：1.001 倍增益（只改量级，不改控制流）
        ok_err.append(np.abs(good - ref).ravel())
        bad_err.append(np.abs(bias - ref).ravel())
        sc.append(np.abs(ref).ravel())
    return ok_err, bad_err, sc


for spread_filter in (0, 1):
    ok_err, bad_err, sc = run_corpus(spread_filter)
    pooled = np.concatenate(ok_err)
    calibrated_atol = float(np.percentile(pooled, 95)) * 1.5   # [C] 级论文的标定规则
    d_atol, d_rtol = D["fp64_ref_vs_fp16_kernel"]
    v_atol, v_rtol = SP.VLLM_DEFAULT_TOLERANCES["float16"]

    def verdict(errs, atol, rtol):
        return sum(1 for e, s in zip(errs, sc) if not np.all(e <= atol + rtol * s))

    specs = (
        ("[D] 规格 fp16 (atol=1e-3, rtol=1e-2)", d_atol, d_rtol),
        ("[B] vLLM fp16 (atol=1e-3, rtol=1e-3)", v_atol, v_rtol),
        (f"[C] 标定 (atol={calibrated_atol:.2e}, rtol=0)", calibrated_atol, 0.0),
    )
    print(f"    语料：{len(ok_err)} 个 case（spread={spread_filter}），"
          f"正确实现的误差 p95 = {np.percentile(pooled, 95):.3e}"
          f" → 标定 atol = p95×1.5 = {calibrated_atol:.3e}")
    print(f"      {'容差表':>38} | {'捕获（缺陷）':>12} | {'误报（正确）':>12}")
    for label, atol, rtol in specs:
        print(f"      {label:>38} | {verdict(bad_err, atol, rtol):>7} / {len(ok_err):<4}"
              f" | {verdict(ok_err, atol, rtol):>7} / {len(ok_err):<4}")
    if spread_filter == 0:
        d_catch = verdict(bad_err, d_atol, d_rtol)
        c_catch = verdict(bad_err, calibrated_atol, 0.0)
        check("C.2 ★★ [D] 级容差在这批 case 上**捕获数为 0**（全部漏报）",
              d_catch == 0, f"捕获 {d_catch} / {len(ok_err)}")
        check("C.2 ★★ 标定容差在**同一批** case 上**全部捕获**",
              c_catch == len(ok_err), f"捕获 {c_catch} / {len(ok_err)}")
        check("C.2 ★★ 同一批 case、同一个缺陷，两张容差表给出**相反**判定",
              d_catch != c_catch)
    else:
        v_catch = verdict(bad_err, v_atol, v_rtol)
        v_fp = verdict(ok_err, v_atol, v_rtol)
        c_catch = verdict(bad_err, calibrated_atol, 0.0)
        c_fp = verdict(ok_err, calibrated_atol, 0.0)
        check("C.3 ★★ 标定容差也有**误报** ⇒ 它是**一笔交易**，不是免费的改进",
              c_fp > 0, f"误报 {c_fp} / {len(ok_err)}")
        check("C.3 ★★ 在这批 case 上标定的捕获率 ≥ vLLM 的，且误报 ≤ vLLM 的（**占优**）",
              c_catch >= v_catch and c_fp <= v_fp,
              f"标定 {c_catch}/{c_fp} vs vLLM {v_catch}/{v_fp}")
        print("    ★★ 这就是论文报告的同一笔交易：召回 73.2% → 82.4%（+9.3 个百分点），")
        print("      代价是控制组误报 0.0% → 1.1%。**两个方向都要写进正文**，")
        print("      只写收益就是选择性引用；只写代价就是否定一个已在用的做法。")

sub("C.4  五类 GPU 上判定一致 ⇒ 「换块卡再测一遍」**不构成**补充证据")
print(f"    论文用的五类 GPU：{SP.ARXIV_2606_20128_GPUS}")
check("C.4 ★★ 五类 GPU 的**逐个名字**被钉住（含 3 个消费级/工作站 + 2 个数据中心卡）",
      SP.ARXIV_2606_20128_GPUS
      == ("RTX 3060", "A10", "L40S", "A100 SXM4", "H100 NVL"),
      f"{SP.ARXIV_2606_20128_GPUS}")
check("C.4 ★ 判定在五类 GPU 上完全一致",
      SP.ARXIV_2606_20128_VERDICTS_IDENTICAL_ACROSS_GPUS is True)
print("    ⚠ **不得**读成「换卡结果逐位相同」——论文比的是**判定**（flagged / clean），")
print("      不是逐位数值。这两件事的差别正是 01 篇 §1 的定理所关心的。")
check("C.4 ★ 论文自陈它的 buggy 变体是**作者播种的**（与本主题 §C 的「本库构造」同一件事）",
      SP.ARXIV_2606_20128_BUGGY_VARIANTS_ARE_AUTHOR_SEEDED is True)
check("C.4 ★ 语料规模与采集条件由外部给出（26 个算子）",
      SP.ARXIV_2607_16228_CORPUS_SIZE_OPS == 26)


# ===========================================================================
# D. 常量自洽
# ===========================================================================
head("§D  常量自洽：本篇用到的 specs.py 常量，逐条被断言钉住")

sub("D.1  vLLM 容差表的内部一致性与那处**未决**")
for d, (a, r) in SP.VLLM_DEFAULT_TOLERANCES.items():
    print(f"      {d:16s} atol = {a:<8g} rtol = {r:<8g}")
check("D.1 ★ fp16 的 atol 与 rtol **相等**（都是 1e-3）",
      SP.VLLM_DEFAULT_TOLERANCES["float16"] == (1e-3, 1e-3))
check("D.1 ★ bf16 的 rtol **宽于** fp16（因为尾数少 3 位）",
      SP.VLLM_DEFAULT_TOLERANCES["bfloat16"][1] > SP.VLLM_DEFAULT_TOLERANCES["float16"][1])
check("D.1 ★★ bf16 那一格的 **atol 存在未决**：两次独立检索读到两个不同的值",
      "vllm_bf16_atol_two_readings_disagree" in SP.UNSOURCED)
check("D.1 ★ int8 用 rtol = 0（相对误差对整数无意义）—— 与 D 级表的 int8 行**不可比**",
      SP.VLLM_DEFAULT_TOLERANCES["int8"][1] == 0.0
      and "spec_d_int8_tolerance_row" in SP.UNSOURCED)
check("D.1 表里至少 7 个 dtype（少一个就说明被裁剪过）",
      len(SP.VLLM_DEFAULT_TOLERANCES) >= 7, f"{len(SP.VLLM_DEFAULT_TOLERANCES)}")

sub("D.2  D 级表的处置：只当靶子，且必须显式标注")
check("D.2 D 级表有三行（fp32 / fp16 / bf16）",
      set(SP.SPEC_D_TOLERANCE_TABLE) == {"fp64_ref_vs_fp32_kernel", "fp64_ref_vs_fp16_kernel",
                                        "fp64_ref_vs_bf16_kernel"})
check("D.2 ★ fp32 行比 fp16 行**紧 1000 倍**（表里唯一的数量级信息）",
      abs(SP.SPEC_D_TOLERANCE_TABLE["fp64_ref_vs_fp16_kernel"][1]
          / SP.SPEC_D_TOLERANCE_TABLE["fp64_ref_vs_fp32_kernel"][1] - 1000.0) < 1e-6)
check("D.2 ★★ 那个 1000× 与 `u16/u32 = 8192` 同量级但**不等**（十进制取整，不是从 `u` 推的）",
      abs(np.log10(SP.SPEC_D_TOLERANCE_TABLE["fp64_ref_vs_fp16_kernel"][1]
                   / SP.SPEC_D_TOLERANCE_TABLE["fp64_ref_vs_fp32_kernel"][1])
          - np.log10(8192.0)) < 1.0)
check("D.2 ★ 表里三行的 rtol **单调递增**（fp32 < fp16 < bf16）——方向上是对的",
      SP.SPEC_D_TOLERANCE_TABLE["fp64_ref_vs_fp32_kernel"][1]
      < SP.SPEC_D_TOLERANCE_TABLE["fp64_ref_vs_fp16_kernel"][1]
      < SP.SPEC_D_TOLERANCE_TABLE["fp64_ref_vs_bf16_kernel"][1])

sub("D.3  A 级：u 与 J 的关系（本行决定「累加口径」变了结论就变）")
for J in (64, 1024, 4096, 16384):
    print(f"      J = {J:>6}：√J·u16 = {np.sqrt(J) * u16:.3e}（fp16 累加）"
          f" · γ_J(u32) = {J * u32:.3e}（fp32 累加）")
check("D.3 ★ fp32 累加的界在 J = 16384 时仍 < 1e-3，而 √J·u16 已到 6e-2",
      np.sqrt(16384) * u16 > 1e-3 and 16384 * u32 < 1e-3,
      f"{np.sqrt(16384) * u16:.2e} vs {16384 * u32:.2e}")
check("D.3 ★ 三个 u 都从 IEEE 754 的 p 推出，不是独立来源",
      u16 == 2.0 ** -11 and ubf == 2.0 ** -8 and u32 == 2.0 ** -24)
check("D.3 ★★ 累加口径决定了容差该按哪个 u 推 ⇒ 本库的既有事实必须一起读",
      SP.FP16_ACCUMULATOR_IS_FP32_IN_REPO is True)

sub("D.4  `UNSOURCED` 在本篇的落点")
check("D.4 ★ D 级表的**原始出处页面找不到** ⇒ 保留表但只当靶子",
      "spec_d_tolerance_table_source_page" in SP.UNSOURCED)
check("D.4 ★ D 级表的 int8 行**不收录**（「通常 < 0.5×scale」无可核实条件）",
      "spec_d_int8_tolerance_row" in SP.UNSOURCED)
print("    ★★ 这两条合起来是本篇的来源纪律：**D 级材料可以出现，但只能作为被检验的对象，")
print("      不能作为结论的依据**；找不到出处的行**直接不写**，而不是「大概差不多」。")

sub("D.5  本程序的常量覆盖表")
COVERED_02 = (
    "VLLM_DEFAULT_TOLERANCES",
    "VLLM_TOLERANCES_ARE_CONSERVATIVE_BY_DESIGN",
    "VLLM_TOLERANCE_LOOKUP_RAISES_ON_UNLISTED_DTYPE",
    "VLLM_TOLERANCE_OVERRIDE_API",
    "SPEC_D_TOLERANCE_TABLE",
    "SPEC_D_TOLERANCE_TABLE_IS_D_LEVEL",
    "SPEC_D_TOLERANCE_TABLE_HAS_NO_SOURCE_PAGE",
    "FP16_UNIT_ROUNDOFF_U",
    "BF16_UNIT_ROUNDOFF_U",
    "FP32_UNIT_ROUNDOFF_U",
    "UBF16_OVER_U16",
    "UBF16_OVER_U16_ORDERS_OF_MAGNITUDE",
    "ARXIV_2606_20128_ILLUSION_SOURCES",
    "ARXIV_2606_20128_BOUNDARY_SHAPES",
    "ARXIV_2606_20128_BOUNDARY_REMOVAL_ZEROES_SHAPE_BUG_RECALL",
    "ARXIV_2606_20128_REFERENCE_PRECISION",
    "ARXIV_2606_20128_TOLERANCE_IS_ABSOLUTE_ONLY",
    "ARXIV_2606_20128_GPUS",
    "ARXIV_2606_20128_VERDICTS_IDENTICAL_ACROSS_GPUS",
    "ARXIV_2606_20128_BUGGY_VARIANTS_ARE_AUTHOR_SEEDED",
    "ARXIV_2607_16228_CALIBRATION_RULE",
    "ARXIV_2607_16228_CORPUS_SIZE_OPS",
    "ARXIV_2607_16228_MEDIAN_TIGHTENING_RANGE",
    "ARXIV_2607_16228_ATTENTION_TRITON_FP16_TIGHTENING",
    "ARXIV_2607_16228_HANDPICKED_ATOL_ATTENTION_TRITON_FP16",
    "ARXIV_2607_16228_CALIBRATED_ATOL_ATTENTION_TRITON_FP16",
    "FP16_TO_BF16_CONSTANT_TOLERANCE_IS_WRONG",
    "FP16_ACCUMULATOR_IS_FP32_IN_REPO",
)
for _n in COVERED_02:
    print(f"      {_n:56s} 有断言")
print(f"    ⇒ 02 篇覆盖 {len(COVERED_02)} 条")


# ===========================================================================
head("汇总")
# ===========================================================================
if FAILED:
    print(f"  ✗ {len(FAILED)} 条断言失败：")
    for f in FAILED:
        print(f"      · {f}")
    sys.exit(1)
print("  ✓ 全部断言通过（A 类 7 组 · B 类 4 组 · C 类 4 组 · D 类 5 组）")
print()
print("  ★ 本程序算的全部是**误差界、覆盖率与判定**，**不是速度**：本机无 GPU，")
print("    本域不产出实测性能数字。被测算子与缺陷都是**本库构造的仿真**。")
print("  ★★ 本程序**推翻了规格的一条论断**（「容差按 dtype 定」不够，见 §A.3/A.4），")
print("     并用 [C] 级预印本的标定做法给出了与测量自洽的替代方案。")
print(f"    python {sys.version.split()[0]} / numpy {np.__version__}（未 import torch）")

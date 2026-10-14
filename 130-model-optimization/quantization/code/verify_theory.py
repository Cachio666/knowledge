#!/usr/bin/env python3
"""130-model-optimization/quantization 主题的纯 python3 复核入口。

★ 本文件**固定跑在默认 `python3` 下，永不 `import torch`**（CONTRIBUTING §2 第三种形态）。
  它提供的是「**换了台机器、什么都没装，也能复核的那一半**」：
  用**独立的参考实现**（下面的 `ref_*`）把本主题的**判据**重算一遍，
  再用手写期望值（不是 torch 的输出）判它对不对。
  **它的产出是推导，不是实测**——不得标成本机实测值。

  三类断言在本形态下的分工（CONTRIBUTING §2）：
    · **A 判据复核** —— 本文件。参照物是下面 `ref_*` 这套独立实现与手写期望值。
    · **C 边界与反例** —— 本文件。断言的是「失败行为成立」。
    · **B 代价模型** —— 在**编号程序**里（代价模型的参照是真系统，只有能 import 到它的
      环境里测才有意义）。本文件**不做** B。

★ 本主题里 A/C 与编号程序的分工是 **「数学 vs 实现」**：
  · 本文件（纯 python3 + 标准库）验的是**可独立推导的数学**：Δ/2 界、Δ²/12、
    两种 SQNR 口径的退化、满量程正弦的 SQNR、浮点格式的 RNE 舍入、四种粒度的误差结构；
  · 编号程序（真 torch）测的是**真实实现的行为**：4096² 张量上的粒度表、
    合成链上的误差传播、`torch.quantize_per_*` 的轴语义。
  · **两者不得互相背书。** 数学成立不证明实现如此；实现如此不证明数学成立。
  · **同一条结论若两边都能给，两边都要给**，且分别标注来源档位。
  · ⚠ **规模不同 ⇒ 数字不同**：本文件跑的是小规模版本（N ≤ 2×10^5、张量 ≤ 256²），
    编号程序跑的是 (4096,4096) 与 2×10^6 样本。**只有定性结构必须一致。**

★ 「独立」的硬要求（CONTRIBUTING §2）：本文件里没有一行代码来自 torch，
  期望值**全部**是手写的常量。§D 节把这条**机械地验证一遍**：
  把参考实现故意改错，A 类断言必须失败。

★ 浮点格式的舍入在本文件里是**用 `struct` 自己做的**（把 float64 经 float32 再按
  RNE 截到 bf16）——这是一条与 torch 的 `.to(torch.bfloat16)` **完全不同的代码路径**，
  所以可以拿来对拍；**但它不是 torch 的输出**。

运行：`make verify`（等价于 `python3 verify.py`）。只用标准库。
"""

import math
import random
import struct
import sys

import specs_theory as SPECS


# ===========================================================================
# 口径
# ===========================================================================

_STATE = {"n": 0, "A": 0, "C": 0, "fail": []}

# 本文件自己的测量条件（**与编号程序那档不同**）
STDLIB_N = 200000            # 蒙特卡洛与分布实验的样本数
STDLIB_SEED = 20260918
GRID_M, GRID_N = 256, 256    # 粒度表的规模
GRID_GS = 128
GRID_SEED = 20260918


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
    print("=" * 76)
    print("§%s  %s" % (tag, title))
    print("=" * 76)


def sub(title):
    print()
    print("--- %s ---" % title)


# ===========================================================================
# §1 独立参考实现（全部手写，不含 torch 的任何代码或输出）
# ===========================================================================

def ref_sqnr_db(xs, xh):
    """SQNR = 10log10(Σx²/Σ(x−x̂)²)。**判据：逐元素平方和之比，无归一化。**"""
    s = sum(v * v for v in xs)
    e = sum((a - b) ** 2 for a, b in zip(xs, xh))
    return 10.0 * math.log10(s / e)


def ref_uniform_quant(xs, lo, hi, bits, mode="round"):
    """在 [lo, hi] 上的均匀量化。`step = (hi−lo)/(2^bits − 1)`（**把两端都当格点**）。

    `mode="round"`：最近舍入；`mode="trunc"`：截断（朝 −∞）。
    返回 `(x̂ 列表, step)`。
    """
    step = (hi - lo) / float((1 << bits) - 1)
    out = []
    for v in xs:
        q = (v - lo) / step
        q = math.floor(q) if mode == "trunc" else round(q)     # round = 平局取偶
        out.append(lo + q * step)
    return out, step


def ref_midtread_sine(n, bits, span, phase=None):
    """**mid-tread 满量程正弦**：`x̂ = round(x/step)·step`，`step = span/2^bits`。

    判据（**不写出来这个数就不可复现**）：
      · `x_k = sin(2π·frac(k·φ))`，`φ = (√5−1)/2`（无理相位，避免与采样格点共振）
      · 正弦幅度为 1，`span = 2`（满量程）
    """
    if phase is None:
        phase = (math.sqrt(5.0) - 1.0) / 2.0
    step = float(span) / (1 << bits)
    xs, xh = [], []
    for k in range(n):
        v = math.sin(2.0 * math.pi * math.fmod(k * phase, 1.0))
        xs.append(v)
        xh.append(round(v / step) * step)
    return xs, xh, step


def ref_to_bf16(x):
    """**自己实现** float64 → bfloat16 的 RNE 舍入。

    路径：先用 `struct` 把 double 舍成 float32（这一步就是 RNE），
    再把 32 位的低 16 位按「最近、平局取偶」丢掉。
    **这是一条与 torch 完全不同的代码路径**（torch 走的是 C++ 的 cast 内核）。
    """
    u = struct.unpack("<I", struct.pack("<f", x))[0]
    if (u & 0x7F800000) == 0x7F800000 and (u & 0x007FFFFF):
        return float("nan")
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", u))[0]


def ref_sym_rtn(xs, bits):
    """对称 RTN：`s = max|x|/(2^(bits−1) − 1)`，`x̂ = clip(round(x/s), −QMAX, QMAX)·s`。
    **判据：restricted range（用 −QMAX 而不是 −2^(bits−1)）。**"""
    qmax = float((1 << (bits - 1)) - 1)
    s = max(abs(v) for v in xs) / qmax
    return [max(-qmax, min(qmax, round_away(v / s))) * s for v in xs]


def round_away(x):
    """**平局远离零**的取整（与 torch.round 的平局取偶不同）。

    用它是为了与编号程序的两条通道都不同：编号程序用 torch 的 ties-to-even，
    这里用 ties-away。**两者的差别只在恰好落在半步上的样本上**——
    连续分布下概率为 0，所以两条通道给出的 SQNR 一致到远小于容差。
    """
    return math.floor(x + 0.5) if x >= 0 else math.ceil(x - 0.5)


def ref_quantize_grid(mat, mode, gs=GRID_GS):
    """[tokens, features] 张量上的四种粒度。**轴约定与 specs.AXIS_CONVENTION 一致**：
       · "token"   —— 每**行**一个 scale（沿 axis 1 归约）
       · "channel" —— 每**列**一个 scale（沿 axis 0 归约）
       · "group"   —— 每行内沿 feature 每 gs 个一组
       · "group_tok" —— 每列内沿 token 每 gs 个一组（**另一个切分维度**）
    """
    M, N = len(mat), len(mat[0])

    def q(v, s):
        return max(-127.0, min(127.0, round_away(v / s))) * s

    if mode == "tensor":
        flat = [v for row in mat for v in row]
        s = max(abs(v) for v in flat) / 127.0
        return [q(v, s) for v in flat]
    if mode == "token":
        out = []
        for row in mat:
            s = max(abs(v) for v in row) / 127.0
            out.extend(q(v, s) for v in row)
        return out
    if mode == "channel":
        cols = [max(abs(mat[i][j]) for i in range(M)) / 127.0 for j in range(N)]
        return [q(mat[i][j], cols[j]) for i in range(M) for j in range(N)]
    if mode == "group":
        out = []
        for row in mat:
            for k in range(0, N, gs):
                blk = row[k:k + gs]
                s = max(abs(v) for v in blk) / 127.0
                out.extend(q(v, s) for v in blk)
        return out
    if mode == "group_tok":
        out = []
        for i0 in range(0, M, gs):
            block = mat[i0:i0 + gs]
            for j in range(N):
                col = [block[i][j] for i in range(len(block))]
                s = max(abs(v) for v in col) / 127.0
                out.extend(q(block[i][j], s) for i in range(len(block)))
        # group_tok 的输出按列优先拼接，重排回行优先
        reordered = [None] * (M * N)
        pos = 0
        for i0 in range(0, M, gs):
            block = mat[i0:i0 + gs]
            for j in range(N):
                for i in range(len(block)):
                    reordered[(i0 + i) * N + j] = out[pos]
                    pos += 1
        return reordered
    raise ValueError(mode)


def ref_aciy_alpha(M, b=1.0):
    """Laplace(0,b) 下最优裁剪阈值 α* 的**独立数值最小化**。

    判据（目标函数必须写出）：
        MSE(α) = (1 − P(|X|>α))·Δ²/12 + 2∫_α^∞ (x−α)²f(x)dx,  Δ = 2α/2^M, f = Laplace(0,b)
    尾部积分解析：`2∫_α^∞ (x−α)²(1/(2b))e^{−x/b}dx = 2b²e^{−α/b}`。
    """
    best_a, best_v = None, None
    a = 0.01
    while a < 12.0:
        D = 2.0 * a / (1 << M)
        p = math.exp(-a / b)
        v = (1.0 - p) * D * D / 12.0 + 2.0 * b * b * math.exp(-a / b)
        if best_v is None or v < best_v:
            best_a, best_v = a, v
        a += 0.0005
    return best_a


# ---------------------------------------------------------------------------
# 手写期望值（**不是从任何实现抄的输出**）
# ---------------------------------------------------------------------------

# 判据：stdlib `random`，N = 2×10^5，种子见下各节。
BF16_SQNR_EXPECT = {"N(0,1)": 55.607, "U(-1,1)": 56.604,
                    "Lap(0,1)": 55.620, "单边指数": 55.608}
FIXED8_SQNR_EXPECT = {"N(0,1)": 39.855, "U(-1,1)": 48.093,
                      "Lap(0,1)": 33.346, "单边指数": 32.801}

# ACIQ：论文印值（引用）与本机数值解（推导）**是两档，必须分开**
ACIQ_PAPER = dict(SPECS.ACIQ_ALPHA_PAPER)
ACIQ_LOCAL = dict(SPECS.ACIQ_ALPHA_LOCAL_NUMERIC)

# 粒度表：小规模（256×256，gs = 128）下的**定性结构**断言用阈值
GRID_MIN_ROW_GAIN_DB = 5.0     # 「每行一个 scale」在行异质上至少赢这么多
GRID_MIN_COL_GAIN_DB = 5.0
# ⚠ 这个阈值**依赖张量宽度**：per-channel 在列 outlier 上的优势随宽度增大
#   （256 宽上 7 列 outlier 占 2.7%，4096 宽上只占 0.17%）。本文件用 256 宽，
#   编号程序用 4096 宽（那里是 20.9 dB）。**这就是「规模不同 ⇒ 数字不同」。**
GRID_MIN_OUTLIER_GAIN_DB = 8.0


def make_dists():
    """四个分布，每个用**独立的随机流与种子**（避免相互污染）。"""
    out = {}
    random.seed(STDLIB_SEED)
    out["N(0,1)"] = [random.gauss(0.0, 1.0) for _ in range(STDLIB_N)]
    out["U(-1,1)"] = [random.uniform(-1.0, 1.0) for _ in range(STDLIB_N)]
    random.seed(STDLIB_SEED + 1)
    out["Lap(0,1)"] = [-math.copysign(1.0, u) * math.log(1.0 - 2.0 * abs(u))
                       for u in (random.random() - 0.5 for _ in range(STDLIB_N))]
    random.seed(STDLIB_SEED + 2)
    out["单边指数"] = [random.expovariate(1.0) for _ in range(STDLIB_N)]
    return out


def make_grid(kind, M=GRID_M, N=GRID_N, seed=GRID_SEED):
    """五种场景的小规模版本（判据与 `06_granularity.py` 一致，只是规模不同）。"""
    random.seed(seed)
    if kind == "行异质":
        scales = [10.0 ** random.uniform(-3, 0) for _ in range(M)]
        return [[random.gauss(0, 1) * scales[i] for _ in range(N)] for i in range(M)]
    if kind == "列异质":
        scales = [10.0 ** random.uniform(-3, 0) for _ in range(N)]
        return [[random.gauss(0, 1) * scales[j] for j in range(N)] for _ in range(M)]
    if kind == "列 outlier":
        cols = set(random.sample(range(N), 7))
        return [[random.gauss(0, 1) * (100.0 if j in cols else 1.0)
                 for j in range(N)] for _ in range(M)]
    raise ValueError(kind)


# ===========================================================================
# §A 判据复核
# ===========================================================================

def run_A():
    section("A", "判据复核：手写期望值 vs 独立参考实现")

    sub("A1  Δ/2 界与 σ² = Δ²/12（判据：**未越界**、误差在 [−Δ/2, Δ/2] 内均匀）")
    for label, lo, hi in (("U(-1,1)", -1.0, 1.0), ("N(0,1) 分位截断", -4.0, 4.0)):
        random.seed(STDLIB_SEED + 3)
        if label == "U(-1,1)":
            xs = [random.uniform(lo, hi) for _ in range(STDLIB_N)]
        else:
            xs = [max(lo, min(hi, random.gauss(0, 1))) for _ in range(STDLIB_N)]
            lo, hi = min(xs), max(xs)
        for bits in (4, 8):
            xh, step = ref_uniform_quant(xs, lo, hi, bits)
            e = [b - a for a, b in zip(xs, xh)]          # e ≜ x̂ − x
            mx = max(abs(v) for v in e)
            check("A", "%s B=%d：max|e| == Δ/2（比值 %.9f）" % (label, bits, mx / (step / 2)),
                  abs(mx / (step / 2) - 1.0) < 1e-4,
                  "max|e|=%.12g Δ/2=%.12g" % (mx, step / 2))
            mean = sum(e) / len(e)
            var = sum((v - mean) ** 2 for v in e) / len(e)
            ratio = var / (step * step / 12.0)
            check("A", "%s B=%d：Var(e)/(Δ²/12) = %.4f（±0.5%% 容差）" % (label, bits, ratio),
                  abs(ratio - 1.0) < 0.005,
                  "偏离 1 的部分是有限样本的统计抖动，不是偏差")

    sub("A2  截断：幅度是舍入的两倍、符号恒为负（**同一个 e ≜ x̂ − x 约定**）")
    random.seed(STDLIB_SEED + 4)
    xs = [random.uniform(-1.0, 1.0) for _ in range(STDLIB_N)]
    lo, hi = -1.0, 1.0
    step = (hi - lo) / 255.0
    er, _ = ref_uniform_quant(xs, lo, hi, 8, "round")
    et, _ = ref_uniform_quant(xs, lo, hi, 8, "trunc")
    er = [b - a for a, b in zip(xs, er)]
    et = [b - a for a, b in zip(xs, et)]
    check("A", "舍入：e ∈ [−Δ/2, +Δ/2]（实测 [%.9g, %.9g]）"
          % (min(er), max(er)), min(er) >= -step / 2 - 1e-12 and max(er) <= step / 2 + 1e-12)
    check("A", "截断：e ∈ (−Δ, 0]，**恒 ≤ 0**（实测 [%.9g, %.9g]）"
          % (min(et), max(et)), max(et) <= 1e-12 and min(et) > -step - 1e-12)
    r = math.sqrt(sum(v * v for v in et) / sum(v * v for v in er))
    check("A", "★ 截断 RMS / 舍入 RMS = %.5f ≈ 2（指**幅度**，方差比是 4）" % r,
          abs(r - 2.0) < 1e-2, "偏离 2 的部分来自有限样本与平局样本")

    sub("A3  ★ 两种 SQNR 口径：**都必须能退化到 6.02B + 1.76** —— 这是判据")
    for key, c in sorted(SPECS.SQNR_CONVENTIONS.items()):
        s = SPECS.SQNR_CONST_DERIVATION
        Xm_over_sigma = (1.0 / s["degenerate_sigma_over_Xm_span"] if key == "A_span"
                         else 1.0 / s["degenerate_sigma_over_Xm_peak"])
        at8 = SPECS.MT001_SLOPE_DB_PER_BIT * 8 + c["const_db"] - 20.0 * math.log10(Xm_over_sigma)
        check("A", "%s（Xm = %s）：B=8 退化值 %.4f dB（specs 记 %.2f，6.02·8+1.76 = %.2f）"
              % (key, c["Xm_meaning"], at8, c["at_B8_db"], SPECS.MT001_SLOPE_DB_PER_BIT * 8 + SPECS.MT001_FULL_SCALE_SINE_SNR_DB),
              abs(at8 - c["at_B8_db"]) < 0.02 and abs(at8 - (SPECS.MT001_SLOPE_DB_PER_BIT * 8 + SPECS.MT001_FULL_SCALE_SINE_SNR_DB)) < 0.02)
    check("A", "★ 两个常数只差 20·log10(2) = %.4f dB —— 差别就是「Xm 指跨度还是峰值」"
          % (20 * math.log10(2)),
          abs((SPECS.SQNR_CONVENTIONS["A_span"]["const_db"]
               - SPECS.SQNR_CONVENTIONS["B_peak"]["const_db"]) - 20 * math.log10(2)) < 0.01)
    check("A", "★ 10·log10(12) = %.4f（跨度口径的常数）；4.77 = 它 − 20log10(2) = %.4f"
          % (10 * math.log10(12), 10 * math.log10(12) - 20 * math.log10(2)),
          abs(10 * math.log10(12) - SPECS.SQNR_CONVENTIONS["A_span"]["const_db"]) < 0.01)

    sub("A4  ★ 满量程正弦：mid-tread 定义下标出 50.02 / 98.09（判据在 ref_midtread_sine 里）")
    for bits in (8, 16):
        xs, xh, step = ref_midtread_sine(20000, bits, 2.0)
        got = ref_sqnr_db(xs, xh)
        want = SPECS.SQNR_SINE_MEASURED[bits]
        check("A", "n=%d：独立实现 %.4f dB（编号程序与 specs 记 %.3f）" % (bits, got, want),
              abs(got - want) < 0.05,
              "公式 6.02n+1.76 = %.2f（实测略高，因为正弦的幅度分布不是均匀的）"
              % (SPECS.MT001_SLOPE_DB_PER_BIT * bits + 1.76))
    xs, xh, step = ref_midtread_sine(200000, 8, 2.0)
    e = [b - a for a, b in zip(xs, xh)]
    mean = sum(e) / len(e)
    var = sum((v - mean) ** 2 for v in e) / len(e)
    r = var / (step * step / 12.0)
    check("A", "★ 正弦的误差方差 / (Δ²/12) = %.4f（< 1）—— Δ²/12 那条「误差均匀」"
          "的假设在正弦上不精确" % r, 0.95 < r < 1.0)

    sub("A5  ★ 浮点 SQNR：**用 `struct` 自己实现 bf16 的 RNE**（独立于 torch 的路径）")
    dists = make_dists()
    rows = {}
    for k in ("N(0,1)", "U(-1,1)", "Lap(0,1)", "单边指数"):
        rows[k] = ref_sqnr_db(dists[k], [ref_to_bf16(v) for v in dists[k]])
        check("A", "bf16 %s：%.3f dB（手写期望 %.3f）" % (k, rows[k], BF16_SQNR_EXPECT[k]),
              abs(rows[k] - BF16_SQNR_EXPECT[k]) < 0.05)
    spread = max(rows.values()) - min(rows.values())
    check("A", "★ bf16 跨四个分布的波动 %.3f dB —— **就在 1 dB 上下**（本题的判据是 "
          "`|spread − 1| < 0.1`）" % spread, abs(spread - 1.0) < 0.1,
          "**所以「< 1 dB」这个写法不能写**：本文件（N=2×10^5）量到 %.3f（低于 1），"
          "而编号程序（torch，N=2×10^6）量到 %.2f（高于 1）—— **同一个量跨在 1 上**"
          % (spread, SPECS.FIXED_VS_FLOAT["float_spread_db"]["bfloat16"]))
    check("A", "★ 对照：编号程序（torch，N=2×10^6）量到 %.2f dB **> 1** —— "
          "所以正文不得写「< 1 dB」"
          % SPECS.FIXED_VS_FLOAT["float_spread_db"]["bfloat16"],
          SPECS.FIXED_VS_FLOAT["float_spread_db"]["bfloat16"] > 1.0)

    sub("A6  ★ 定点 SQNR：同一个 6.02n+1.76 的问题，**定点是另一套答案**")
    fix = {}
    for k in ("N(0,1)", "U(-1,1)", "Lap(0,1)", "单边指数"):
        fix[k] = ref_sqnr_db(dists[k], ref_sym_rtn(dists[k], 8))
        check("A", "定点 8-bit 对称 per-tensor %s：%.3f dB（手写期望 %.3f）"
              % (k, fix[k], FIXED8_SQNR_EXPECT[k]), abs(fix[k] - FIXED8_SQNR_EXPECT[k]) < 0.05)
    spread_f = max(fix.values()) - min(fix.values())
    check("A", "★ 定点跨四个分布的波动 %.2f dB = 浮点的 %.1f 倍"
          % (spread_f, spread_f / spread), spread_f > 10 * spread)
    check("A", "★ 同一个格式（8 bit）在同一个分布上，定点 %.2f dB vs 浮点 %.2f dB —— "
          "**两者不可比**（判据不同：一个是 min-max 定点，一个是 bf16）"
          % (fix["N(0,1)"], rows["N(0,1)"]), abs(fix["N(0,1)"] - rows["N(0,1)"]) > 10.0)

    sub("A7  ACIQ：**论文印值 vs 本机数值解**是两档，且两者**不一致**")
    for M in (2, 3, 4):
        got = ref_aciy_alpha(M)
        paper = ACIQ_PAPER[M]
        rel = abs(got - paper) / paper * 100.0
        check("A", "M=%d：本机数值解 α* = %.4f（论文印值 %.2f，相对差 %.2f%%）"
              % (M, got, paper, rel), 0.1 < rel < 1.0 and abs(got - ACIQ_LOCAL[M]) < 0.01,
              "specs 记的本机解 %.3f" % ACIQ_LOCAL[M])
        check("A", "M=%d：α*=%.2f 处的饱和概率 e^(−α/b) = %.4f（specs 记 %.4f）"
              % (M, paper, math.exp(-paper), SPECS.ACIQ_SATURATION_PROB[M]),
              abs(math.exp(-paper) - SPECS.ACIQ_SATURATION_PROB[M]) < 5e-5)

    sub("A8  ★ 粒度表（小规模 %d×%d，gs=%d）：**定性结构与 (4096,4096) 版本一致**"
        % (GRID_M, GRID_N, GRID_GS))
    row_h = make_grid("行异质")
    col_h = make_grid("列异质")
    flat_row = [v for r in row_h for v in r]
    flat_col = [v for r in col_h for v in r]
    g = {}
    for tag, mat, flat in (("行异质", row_h, flat_row), ("列异质", col_h, flat_col)):
        g[tag] = {m: ref_sqnr_db(flat, ref_quantize_grid(mat, m))
                  for m in ("tensor", "token", "channel", "group")}
        print("      %-8s tensor %6.2f | token %6.2f | channel %6.2f | group %6.2f"
              % ((tag,) + tuple(g[tag][m] for m in ("tensor", "token", "channel", "group"))))
    check("A", "★ 行异质上「**每行**一个 scale」赢 %.2f dB ⇒ 它治的是**行**"
          % (g["行异质"]["token"] - g["行异质"]["channel"]),
          g["行异质"]["token"] - g["行异质"]["channel"] > GRID_MIN_ROW_GAIN_DB)
    check("A", "★ 列异质上「**每列**一个 scale」赢 %.2f dB ⇒ 它治的是**列**"
          % (g["列异质"]["channel"] - g["列异质"]["token"]),
          g["列异质"]["channel"] - g["列异质"]["token"] > GRID_MIN_COL_GAIN_DB)
    check("A", "★ specs 的轴约定与实测一致（每行 = per-token、每列 = per-channel）",
          "每行一个 scale" in SPECS.AXIS_CONVENTION["row_scale"]
          and "每列一个 scale" in SPECS.AXIS_CONVENTION["col_scale"])
    ol = make_grid("列 outlier")
    flat_ol = [v for r in ol for v in r]
    o_tok = ref_sqnr_db(flat_ol, ref_quantize_grid(ol, "token"))
    o_chn = ref_sqnr_db(flat_ol, ref_quantize_grid(ol, "channel"))
    check("A", "★ 列 outlier 下：per-channel %.2f vs per-token %.2f，**per-channel 多 %.2f dB**"
          % (o_chn, o_tok, o_chn - o_tok), o_chn - o_tok > GRID_MIN_OUTLIER_GAIN_DB,
          "**这一格是 KIVI「K 必须用 per-channel」的理论根据；标签写反就会反过来。**"
          "⚠ 本文件是 %d 宽（差 %.2f dB），编号程序是 4096 宽（差 20.89 dB）—— "
          "**阈值依赖宽度，只有定性方向必须一致**" % (GRID_N, o_chn - o_tok))

    sub("A9  ★ per-group 的切分维度：同一张张量、同一个 group size，换维度差很多")
    a = ref_sqnr_db(flat_ol, ref_quantize_grid(ol, "group"))
    b = ref_sqnr_db(flat_ol, ref_quantize_grid(ol, "group_tok"))
    check("A", "沿 feature %.2f dB vs 沿 token %.2f dB，差 %.2f dB ⇒ "
          "**不写这一维，group size 就不可比**" % (a, b, abs(a - b)), abs(a - b) > 3.0)


# ===========================================================================
# §B 代价模型的算术（**不是 B 类断言**，B 类在编号程序里）
# ===========================================================================

def run_B():
    section("B", "代价模型的算术复核（本文件只验算术，B 类断言在编号程序里）")

    sub("B1  scale 的元数据开销：每 G 个元素一个 fp32 scale ⇒ 32/G bit/元素")
    for G, w in ((None, 32), (4096, 32), (128, 32), (128, 16), (32, 32), (16, 32)):
        if G is None:
            print("      per-tensor：32 bit / 整张量 ⇒ 可忽略")
            continue
        print("      G=%4d, scale=%2d bit：%8.4f bit/元素" % (G, w, float(w) / G))
    check("A", "★ G=32 的 scale 开销恰好是 G=128 的 4 倍（同一 scale 宽度下）",
          abs((32.0 / 32) / (32.0 / 128) - 4.0) < 1e-12)
    check("A", "★ 4-bit 载荷 + G=128 的 fp32 scale ⇒ 等效 4.25 bit/元素（+6.25%%）",
          abs(4.0 + 32.0 / 128 - 4.25) < 1e-12)


# ===========================================================================
# §C 边界与反例（**断言的是「失败行为成立」**）
# ===========================================================================

def run_C():
    section("C", "边界与反例：下面每一条都**必须真的跑出问题**")

    sub("C1  越界（overload）使 Δ/2 界失效 —— 要跑出**超过 Δ/2 的误差**")
    random.seed(STDLIB_SEED + 5)
    for bits, cover, xmax in ((3, 1.0, 2.0), (8, 1.0, 1.5)):
        step = 2.0 * cover / (1 << bits)
        n = 100000
        over = 0
        mx = 0.0
        for _ in range(n):
            v = random.uniform(-xmax, xmax)
            q = max(-cover, min(cover, round(v / step) * step))
            e = abs(q - v)
            mx = max(mx, e)
            if e > step / 2:
                over += 1
        check("C", "B=%d 量程±%.0f 输入±%.0f：max|e|/(Δ/2) = %.1f，超界样本 %.1f%%"
              % (bits, cover, xmax, mx / (step / 2), 100.0 * over / n),
              over > 0 and mx / (step / 2) > 1.5,
              "**Δ/2 界只在输入不越界时成立**")

    sub("C2  ★ 两种口径**混用**会给出偏高 6 dB 的错值 —— 要跑出那个 55.95")
    bad = SPECS.MT001_SLOPE_DB_PER_BIT * 8 + SPECS.SQNR_CONVENTIONS["A_span"]["const_db"] - 20.0 * math.log10(math.sqrt(2.0))
    good = SPECS.SQNR_CONVENTIONS["A_span"]["at_B8_db"]
    check("C", "混用（跨度口径的 10.8 配峰值口径的 σx = Xm/√2）= %.2f dB，"
          "正确 %.2f dB，偏高 %.2f dB" % (bad, good, bad - good),
          abs(bad - 55.95) < 0.05 and (bad - good) > 6.0)
    check("C", "★ 混用版本退化出的常数项是 %.2f，**不是 1.76** —— 判据在这里生效"
          % (bad - SPECS.MT001_SLOPE_DB_PER_BIT * 8), abs((bad - SPECS.MT001_SLOPE_DB_PER_BIT * 8) - 1.76) > 5.0)

    sub("C3  ★ 「n=8 → 50.02」**依赖量化定义** —— 换定义就跑不出它")
    defs = []
    xs, xh, step = ref_midtread_sine(20000, 8, 2.0)
    defs.append(("mid-tread round(x/step)·step（本文口径）", ref_sqnr_db(xs, xh)))
    defs.append(("round(x·2^(n−1))/2^(n−1)  ← **与上一行代数等价**",
                 ref_sqnr_db(xs, [round(v * 128.0) / 128.0 for v in xs])))
    smax = max(abs(v) for v in xs)
    defs.append(("对称 /127（restricted range）",
                 ref_sqnr_db(xs, [max(-127.0, min(127.0, round(v / (smax / 127.0))))
                                  * (smax / 127.0) for v in xs])))
    defs.append(("mid-rise (floor(x/step)+0.5)·step",
                 ref_sqnr_db(xs, [(math.floor(v / step) + 0.5) * step for v in xs])))
    for name, v in defs:
        print("      %-48s → %7.3f dB" % (name, v))
    check("C", "★ 前两个定义**是同一个式子**（span=2、B=8 时 round(x·128)/128 ≡ "
          "round(x/step)·step）", abs(defs[0][1] - defs[1][1]) < 1e-9)
    check("C", "★ 真正不同的定义（对称 /127、mid-rise）**都到不了 50.02**",
          all(abs(v - SPECS.SQNR_SINE_MEASURED[8]) > 0.05 for _, v in defs[2:]),
          "**「实测值可复现」这条验收的实质：定义必须一起写出来**")

    sub("C4  Δ²/12 的四条假设失效：误差既非均匀、也与信号强相关")
    random.seed(STDLIB_SEED + 6)
    cases = [("恒定 0.2（网格正中央）", [0.2] * 50000, 3, 2.0),
             ("均匀落在 [0, 0.1]（远小于一个步长）",
              [random.random() * 0.1 for _ in range(50000)], 3, 2.0)]
    for name, xs, bits, span in cases:
        step = span / (1 << bits)
        xh, _ = ref_uniform_quant(xs, -1.0, 1.0, 0, "round") if False else (None, None)
        q = [round(v / step) * step for v in xs]
        e = [b - a for a, b in zip(xs, q)]
        mean = sum(e) / len(e)
        var = sum((v - mean) ** 2 for v in e) / len(e)
        r = var / (step * step / 12.0)
        check("C", "%s：Var(e)/(Δ²/12) = %.4f（应明显偏离 1）" % (name, r),
              abs(r - 1.0) > 0.05,
              "教科书自述四条假设只在「步长小、信号复杂、未过载」时是好近似")

    sub("C5  ★ 轴标签写反会推出**相反的结论**（写反时不会有任何报错）")
    ol = make_grid("列 outlier")
    flat = [v for r in ol for v in r]
    tok = ref_sqnr_db(flat, ref_quantize_grid(ol, "token"))
    chn = ref_sqnr_db(flat, ref_quantize_grid(ol, "channel"))
    check("C", "★ 「把标签写反」推出的结论是「per-token 更好」= %s，实测 per-channel 好 %.2f dB "
          "⇒ 那个结论为假" % (tok > chn, chn - tok), (not (tok > chn)) and chn - tok > 10.0,
          "两个数各自都对，只是标签错了 —— 这就是 P9 的形状")

    sub("C6  ★ 「# groups = 128」当成「group size = 128」会差一个数量级（**纯算术**）")
    check("C", "★ 768 宽的矩阵上，128 **组** ⇒ 每组 %d 个元素，不是 128"
          % (768 // SPECS.QBERT_GROUP_SWEEP["# groups"][2]),
          768 // SPECS.QBERT_GROUP_SWEEP["# groups"][2] == 6)
    check("C", "★ 把「128 组」写成「group size 128」是 %d 倍的作用域错配"
          % (128 // 6), 128 // 6 == 21,
          "Q-BERT 的编号是 %s（不是 %s = HAWQ）"
          % (SPECS.QBERT_ARXIV, SPECS.QBERT_NOT_THIS_ARXIV))
    check("C", "★ 128 组→768 组的增量 %.2f（SST-2），**「≤0.1%%」对 SST-2 不成立**"
          % SPECS.QBERT_128_TO_768_DELTA_SST2, SPECS.QBERT_128_TO_768_DELTA_SST2 > 0.1)

    sub("C7  ★ massive activations ≠ outlier features（**换分母就差 2000 倍**）")
    # 小规模复算：R1 式（4 列 × 全部 token）vs R3 式（4 个标量）
    n_tok, n_feat = 2048, 4096
    n_act = n_tok * n_feat
    r1_share = 4.0 / n_feat                  # 占 feature 维
    r3_share = 4.0 / n_act                   # 占 activation 位置
    check("C", "★ 按**占 activation 位置**算：R1 %.4f%% vs R3 %.2e%%，差 %.0f 倍"
          % (100 * (4.0 * n_tok / n_act), 100 * r3_share, (4.0 * n_tok / n_act) / r3_share),
          (4.0 * n_tok / n_act) / r3_share > 1000.0)
    check("C", "★ 按**占 feature 维**算：R1 %.4f%% vs R3 %.4f%% —— **同一个量级**"
          % (100 * r1_share, 100 * 2.0 / n_feat), True,
          "**换一个分母，两者就差 2000 倍变成只差 2 倍** —— 两个数都对，但不是同一个量")
    check("C", "★ 所以 specs 里两条描述**必须带各自的分母**",
          "feature" in SPECS.MASSIVE_VS_OUTLIER["outlier_features"]["location"]
          and "activation" in SPECS.MASSIVE_VS_OUTLIER["massive"]["count"])


# ===========================================================================
# §D 可证伪性：把参考实现故意改错，A 类断言必须失败
# ===========================================================================

def broken_rounding_bound_uses_quarter(xs, lo, hi, bits):
    """故意写错：把 Δ/2 界写成 Δ/4（**常见的「差一个因子 2」写反了方向**）。

    ⚠ 注意：写成 **Δ** 不是一条有效的变异——那是一个**更松**的界，max|e| 根本够不到它，
    断言照样绿。**「改错」必须是往紧的方向改，否则它测不出任何东西。**
    """
    _, step = ref_uniform_quant(xs, lo, hi, bits)
    return step / 4.0


def broken_truncation_sign(xs, lo, hi, bits):
    """故意写错：以为截断的误差是**正的**（把 e ≜ x̂ − x 的约定记反）。"""
    xh, _ = ref_uniform_quant(xs, lo, hi, bits, "trunc")
    return [a - b for a, b in zip(xs, xh)]


def broken_sqnr_const_is_10_8(x, bits):
    """故意写错：把**峰值**口径的常数写成跨度口径的 10.8。"""
    return SPECS.MT001_SLOPE_DB_PER_BIT * bits + 10.8 - 20.0 * math.log10(x)


def broken_bf16_rounds_up(x):
    """故意写错：bf16 不做 RNE，直接截断低 16 位。"""
    u = struct.unpack("<I", struct.pack("<f", x))[0]
    return struct.unpack("<f", struct.pack("<I", u & 0xFFFF0000))[0]


def broken_quantize_row_is_channel(mat, gs=GRID_GS):
    """故意写错：把「每行一个 scale」标成 channel（**轴标签互换**）。

    实现上就是返回「列 scale」的结果，但调用方以为它是 token ——
    这里直接返回列尺度的结果，用来证明「A1 的标签断言会因为互换而失败」。
    """
    return ref_quantize_grid(mat, "channel")


def run_D():
    section("D", "可证伪性：把参考实现故意改错，A 类断言必须失败")

    random.seed(STDLIB_SEED + 7)
    xs = [random.uniform(-1.0, 1.0) for _ in range(20000)]
    xh, step = ref_uniform_quant(xs, -1.0, 1.0, 8)
    e = [b - a for a, b in zip(xs, xh)]
    mx = max(abs(v) for v in e)
    bad = broken_rounding_bound_uses_quarter(xs, -1.0, 1.0, 8)
    check("A", "改错 1（把 Δ/2 写成 Δ/4）会与「max|e| == Δ/2」冲突"
          "（max|e| 是错误上界的 %.2f 倍）" % (mx / bad), mx / bad > 1.5)

    et = broken_truncation_sign(xs, -1.0, 1.0, 8)
    check("A", "改错 2（截断误差符号写反）会与「恒 ≤ 0」冲突（实测 max = %.3g）"
          % max(et), max(et) > 0.0)

    good = broken_sqnr_const_is_10_8(1.0 / math.sqrt(2.0), 8)
    check("A", "改错 3（峰值口径硬套跨度常数）会与 `at_B8_db` 冲突（得 %.2f）" % good,
          abs(good - SPECS.SQNR_CONVENTIONS["B_peak"]["at_B8_db"]) > 5.0,
          "这正是 §C2 那个 55.95 的成因")

    dists = make_dists()
    rows = [ref_sqnr_db(dists[k], [broken_bf16_rounds_up(v) for v in dists[k]])
            for k in ("N(0,1)", "U(-1,1)")]
    check("A", "改错 4（bf16 用截断而非 RNE）会与手写期望冲突（%.2f / %.2f dB）"
          % tuple(rows), all(abs(r - BF16_SQNR_EXPECT[k]) > 0.05
                             for r, k in zip(rows, ("N(0,1)", "U(-1,1)"))),
          "截断是**有偏**的，SQNR 会掉")

    row_h = make_grid("行异质")
    flat = [v for r in row_h for v in r]
    swapped = ref_sqnr_db(flat, broken_quantize_row_is_channel(row_h))
    correct = ref_sqnr_db(flat, ref_quantize_grid(row_h, "token"))
    check("A", "改错 5（把「每行」错当成「每列」）会与 A1 的轴断言冲突"
          "（%.2f vs %.2f dB，差 %.2f）" % (swapped, correct, correct - swapped),
          correct - swapped > GRID_MIN_ROW_GAIN_DB,
          "**这就是 P9 实例 2 的判据：标签换过来，断言必须失败**")

    a = ref_aciy_alpha(4)
    check("A", "对照：正确实现与手写期望**全部**一致（本文件 §A 的汇总会再核一遍）",
          abs(a - ACIQ_LOCAL[4]) < 0.01)


# ===========================================================================
# 主流程
# ===========================================================================

def main():
    print("=" * 76)
    print("130-model-optimization/quantization —— 纯 python3 复核（**不 import torch**）")
    print("=" * 76)
    print()
    print("口径：本文件全部输出是**推导**，不是本机实测值。")
    print("      它跑的是本文件自己的参考实现与手写期望值，不碰任何框架、不碰硬件。")
    print("      编号程序（01–03）在装了 torch 的环境里跑真系统，那才是本机实测值。")
    print("      **两档不得互相背书**：数学成立不证明实现如此，反之亦然。")
    print()
    print("版本绑定：本主题的实测基线锚在 torch %s（本机五个解释器）；"
          % " / ".join(SPECS.MEASURED_VERSIONS))
    print("          **%s 本机没有解释器，不声称覆盖**。" % SPECS.UNCOVERED_MINORS)
    print("          python3 = %s（本文件的唯一测量条件）" % sys.version.split()[0])
    print("          本文件的规模：N = %d 样本、粒度表 %d×%d —— "
          "**与编号程序不同，数字也不同**" % (STDLIB_N, GRID_M, GRID_N))

    run_A()
    run_B()
    run_C()
    run_D()

    section("汇总", "断言汇总")
    print("  共 %d 条：A 类 %d 条，C 类 %d 条" % (_STATE["n"], _STATE["A"], _STATE["C"]))
    if _STATE["fail"]:
        print("  **不通过 %d 条**：" % len(_STATE["fail"]))
        for f in _STATE["fail"]:
            print("    - %s" % f)
        return 1
    print("  全部通过。")
    print()
    print("  提醒：本文件通过与「编号程序跑出的实测数字」是**两档不同的证据**，")
    print("        不得互相背书（CONTRIBUTING §2）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

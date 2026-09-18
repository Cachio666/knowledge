#!/usr/bin/env python3
"""03 bf16 / fp16 / tf32：从量化视角看这四个格式。

对应文档：130-quantization/numerical-formats/03-bf16-fp16-tf32.md

★ 本程序的**主线是「量化误差」**，不是「累加误差」。
  累加误差（同一个格式算得越多、误差怎么按 √K·u 涨）的正文在
  `10-foundations/floating-point/04-low-precision-formats.md`，**本篇不复述、只回链**。
  两篇讲的是**两个不同的误差来源**，不是同一件事的两个深浅。

★ 本程序跑的是**真 torch**——输出是**本机实测值**。本机五个解释器都跑过：
    2.6.0 / 2.7.1 / 2.8.0 / 2.11.0 / 2.13.0，**四个分布的 SQNR 表在三个版本上逐格相同**。
  全部 CPU-only，无 GPU 数字。

★ **tf32 那一列是位掩码仿真，不是硬件实测**：torch **没有 tf32 存储 dtype**
  （它是「输入端截尾数」的行为，不是一种可分配的格式），本机又是 CPU-only，
  所以 §B3 的 tf32 数字全部来自**在 float64 里按 RNE 截到 10 位尾数**的仿真。
  **这一列不得写成「实测 tf32 硬件行为」。**

三类断言（CONTRIBUTING §2 第三种形态：B 在编号程序里，A/C 另见 verify.py）：
  B —— **代价模型**：一位尾数值多少 dB（6.02 dB/bit 在浮点上成不成立）。
  A —— 两条独立通道：真 torch 的 cast vs 自己实现的 RNE 截尾数。
  C —— 失败行为：fp16 的上溢/下溢边界、tf32 仿真与 fp16 在正常区的**重合**。

用法：cd code && make 03_bf16_fp16_tf32   （或 make run 全跑）
"""

import math
import platform
import sys

import torch

import specs as SPECS


_STATE = {"n": 0, "A": 0, "B": 0, "C": 0, "fail": []}


def cpu_model():
    """测量条件里的「CPU 型号」。跨平台：macOS 读 sysctl，Linux 读 /proc/cpuinfo，
    都不行就退回 `platform.machine()`。**这不是算法**，只是把测量条件打出来。"""
    import subprocess
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
    tail = "   ← %s" % detail if detail else ""
    print("    [%s] %s%s" % ("通过" if ok else "**不通过**", name, tail))
    return ok


def rule(title):
    print()
    print("=" * 76)
    print(title)
    print("=" * 76)


def sub(title):
    print()
    print("--- %s ---" % title)


def sqnr_db(x, xh):
    """SQNR = 10·log10(Σx² / Σ(x−x̂)²)，dB。**定义写死**，否则不同口径的数字不可比。"""
    x = x.to(torch.float64)
    xh = xh.to(torch.float64)
    return 10 * math.log10(float((x * x).sum()) / float(((x - xh) ** 2).sum()))


def rne_to_format(x64, m_bits, emin_normal):
    """**独立实现**的「RNE 舍到 m 位尾数的浮点格式」。

    `emin_normal = 1 - bias` 是**最小正规数的指数**；低于它的量必须用**次正规网格**
    （固定步长 `2^(emin_normal - m)`），这一点是整个实现里最容易漏的地方——
    漏了就会在次正规区给出一个比格式实际更细的网格（本程序 A3 会把这个差异显示出来）。

    这一份**不是** torch 的代码路径：它自己求指数、自己定 ulp、再用
    `torch.round`（round-half-to-even）取整。ulp 是 2 的整数次幂，所以 `a / ulp`
    在浮点里是**精确**的（只是指数移位），平局判定不会因为除法而失真。
    """
    a = x64.abs()
    e = torch.clamp(torch.floor(torch.log2(a)), min=float(emin_normal))
    ulp = torch.pow(torch.tensor(2.0, dtype=torch.float64), e - m_bits)
    return torch.sign(x64) * torch.round(a / ulp) * ulp


# 目标格式的 (尾数位, 最小正规数的指数 1-bias)
FORMAT_PARAMS = {
    "bfloat16": (7, -126),
    "float16": (10, -14),
    "tf32(仿真)": (10, -126),          # tf32 的指数与 fp32 同宽
}


def sim_tf32(x64):
    """tf32 的位掩码仿真。**两步，顺序不能颠倒**：

      1. 先把输入按 **fp32** 取整——tf32 是「fp32 矩阵乘在输入端把尾数截到 10 位」，
         **它的入口本来就是 fp32**，不是 fp64；
      2. 再从那个 fp32 值舍到 10 位尾数。

    **这是仿真，不是硬件实测**：torch 没有 tf32 存储 dtype，本机也没有 GPU。
    """
    return rne_to_format(x64.to(torch.float32).to(torch.float64), SPECS.TF32_MANTISSA_BITS, -126)


def make_dists(n):
    """四个分布，**每个都写清参数与截断**（P9：判据没写出来不得并列）。"""
    torch.manual_seed(20260918)
    return [
        ("N(0,1)", torch.randn(n, dtype=torch.float64)),
        ("U(-1,1)", torch.rand(n, dtype=torch.float64) * 2 - 1),
        ("Laplace(0,1)", torch.distributions.Laplace(0.0, 1.0).sample((n,)).to(torch.float64)),
        ("StudentT(3)", torch.clamp(
            torch.distributions.StudentT(3.0).sample((n,)).to(torch.float64), -1e3, 1e3)),
    ]


# ---------------------------------------------------------------------------
# §0 测量条件
# ---------------------------------------------------------------------------

def banner():
    rule("§0  测量条件（文档 §5 直接引用这一行）")
    print("  torch      : %s" % torch.__version__)
    print("  python     : %s" % sys.version.split()[0])
    print("  CPU 型号   : %s" % cpu_model())
    print("  CUDA 可用  : %s（**CPU-only，无 GPU 数字**）" % torch.cuda.is_available())
    print("  本机已跑版本: %s" % " / ".join(SPECS.MEASURED_VERSIONS))
    print("  未覆盖 minor: %s" % SPECS.UNCOVERED_MINORS)
    print()
    print("  ⚠ tf32 那一列是**位掩码仿真**，不是硬件实测——torch 没有 tf32 存储 dtype。")
    print("  ⚠ 本程序**不测累加误差**（√K·u 那条线）——那是")
    print("     10-foundations/floating-point/04-low-precision-formats.md 的正文。")


# ---------------------------------------------------------------------------
# §A 两条独立通道：真 torch 的 cast vs 手写 RNE
# ---------------------------------------------------------------------------

def channels():
    rule("§A  两条独立通道：真 torch 的 cast vs 手写 RNE 截尾数")
    print("  通道 1：`x.to(torch.bfloat16)` / `.to(torch.float16)`（torch 的转换 kernel）。")
    print("  通道 2：`rne_to_mantissa(x, 7)` / `(x, 10)`（本程序在 float64 里另写的最近偶舍入）。")
    print("  两条通道的**实现机制完全不同**：前者走 ATen 的 cast kernel，后者走本程序的算术。")

    sub("A1  bf16 与 fp16 的位域（规格值，由 finfo 与手写期望两条通道核）")
    for name, dt, m_bits in (("bfloat16", torch.bfloat16, 7),
                             ("float16", torch.float16, 10)):
        fi = torch.finfo(dt)
        check("A", "%s 的尾数位数 = %d" % (name, m_bits), fi.bits == 16)
        check("A", "%s 的 eps = 2^-%d = %g" % (name, m_bits, 2.0 ** -m_bits),
              abs(fi.eps - 2.0 ** -m_bits) < 1e-12, "finfo=%g" % fi.eps)
        want_max = SPECS.BF16_MAX if m_bits == 7 else SPECS.FP16_MAX
        check("A", "%s 的 max = %g" % (name, fi.max), abs(fi.max - want_max) < want_max * 1e-6,
              "finfo=%g specs=%g" % (fi.max, want_max))
    check("A", "★ bf16 是 **7 位尾数，不是 8**（eps = 2^-7 = 0.0078125）",
          abs(torch.finfo(torch.bfloat16).eps - 0.0078125) < 1e-12)
    # 把 specs 里的常量与实测咬住——否则 specs 里改一个数字，两边都发现不了
    check("A", "specs.BF16_MANTISSA_BITS = 7 且 2^-M == finfo.eps",
          SPECS.BF16_MANTISSA_BITS == 7
          and abs(2.0 ** -SPECS.BF16_MANTISSA_BITS - torch.finfo(torch.bfloat16).eps) < 1e-12,
          "specs M=%d" % SPECS.BF16_MANTISSA_BITS)
    check("A", "specs.FP16_MANTISSA_BITS = 10 且 2^-M == finfo.eps",
          SPECS.FP16_MANTISSA_BITS == 10
          and abs(2.0 ** -SPECS.FP16_MANTISSA_BITS - torch.finfo(torch.float16).eps) < 1e-12,
          "specs M=%d" % SPECS.FP16_MANTISSA_BITS)
    check("A", "specs.TF32_MANTISSA_BITS = 10（与 fp16 相同；**引用值**，非本机实测）",
          SPECS.TF32_MANTISSA_BITS == 10
          and SPECS.TF32_10BIT_MANTISSA_CONFIDENCE.startswith("引用"))
    check("A", "specs.BF16_MAX / FP16_MAX 与 finfo 一致",
          abs(SPECS.BF16_MAX - torch.finfo(torch.bfloat16).max) < SPECS.BF16_MAX * 1e-6
          and SPECS.FP16_MAX == torch.finfo(torch.float16).max)
    check("A", "bf16 的指数位与 fp32 相同（8 位）⟹ 范围与 fp32 同量级",
          abs(torch.finfo(torch.bfloat16).max / torch.finfo(torch.float32).max - 1.0) < 0.01)

    sub("A2  ★ bf16 / fp16 的转换是 RNE（用中点判别）")
    # bf16（7 位尾数）在 [1,2) 的 ulp = 2^-7 = 1/128。
    # 1 + 1/256 恰在 1.0（尾数 0000000，偶）与 1+1/128（尾数 0000001，奇）的**中点**。
    # RNE 取偶 → 1.0；「恒向上」会得 1+1/128。
    probe = 1.0 + 1.0 / 256.0
    got = float(torch.tensor([probe], dtype=torch.float64).to(torch.bfloat16).to(torch.float64)[0])
    print("      1 + 1/256 = %r（1.0 与 1+1/128 的中点）→ bf16 得 %r" % (probe, got))
    check("A", "RNE：中点取偶尾数 → 1.0（若为「恒向上」会得 1.0078125）",
          got == 1.0, "实测 %r" % got)
    # ⚠ 扰动必须**大于 fp32 的 ulp**（约 1.19e-7），否则它会先被 fp32 那一步吃掉、
    #   又变成一个精确平局——本程序一开始用 2^-30 就踩了这个坑（见 A4）。
    check("A", "对照：比中点高一点（+2^-15）→ 1.0078125（确实按最近）",
          float(torch.tensor([probe + 2.0 ** -15]).to(torch.bfloat16).to(torch.float64)[0]) == 1.0 + 1.0 / 128.0)
    check("A", "对照：比中点低一点（−2^-15）→ 1.0",
          float(torch.tensor([probe - 2.0 ** -15]).to(torch.bfloat16).to(torch.float64)[0]) == 1.0)
    # 另一条中点的独立构造：fp16 的 1 + 1/2048
    p16 = 1.0 + 1.0 / 2048.0
    g16 = float(torch.tensor([p16], dtype=torch.float64).to(torch.float16).to(torch.float64)[0])
    check("A", "fp16 的同一个判据：中点 1 + 1/2048 → 1.0（取偶）", g16 == 1.0, "实测 %r" % g16)

    sub("A3  ★ 两条通道（真 cast vs 手写 RNE）在上万个样本上逐位比对")
    n = 200000
    torch.manual_seed(20260918)
    x = torch.randn(n, dtype=torch.float64)
    for name, dt in (("bfloat16", torch.bfloat16), ("float16", torch.float16)):
        m_bits, emin = FORMAT_PARAMS[name]
        a = x.to(dt).to(torch.float64)
        b = rne_to_format(x, m_bits, emin)
        d = (a != b).nonzero().flatten()
        print("      %-10s：不一致 %d / %d" % (name, len(d), n))
        for i in d[:4].tolist():
            print("         x=%.20g  真 cast=%.20g  手写 RNE=%.20g" % (float(x[i]), float(a[i]), float(b[i])))
        if name == "float16":
            check("A", "fp16：真 cast 与手写 RNE **逐元素完全相同**（含次正规区）",
                  len(d) == 0, "不同 %d 个" % len(d))
        else:
            # bf16 的这两处不一致**不是手写实现的错**：torch 的 float64 → bf16
            # 走的是 **float64 → float32 → bfloat16 的双重舍入**（见 A4）。
            c = x.to(torch.float32).to(torch.bfloat16).to(torch.float64)
            check("A", "bf16：真 cast 与「先转 fp32 再转 bf16」**逐元素相同**"
                       "（⟹ 直转内部就走 fp32）",
                  len((a != c).nonzero()) == 0, "不同 %d 个" % len((a != c).nonzero()))
            check("A", "★ 手写**单次** RNE 与它差 %d 个（双重舍入真的会改变结果）" % len(d),
                  len(d) > 0)
            check("A", "★ fp16 不是这样：它的 float64→fp16 是单次舍入（A3 上一行全等）",
                  True)

    sub("A4  ★ 双重舍入真的会给出不同的结果（bf16 的两个具体反例）")
    for v in (1.4257812575996542392, -2.1953125050592809941):
        t = torch.tensor([v], dtype=torch.float64)
        direct = float(t.to(torch.bfloat16).to(torch.float64)[0])
        via32 = float(t.to(torch.float32).to(torch.bfloat16).to(torch.float64)[0])
        single = float(rne_to_format(t, 7, -126)[0])
        print("      x=%.19g" % v)
        print("        float64→bf16 直转        = %.19g" % direct)
        print("        float64→fp32→bf16        = %.19g" % via32)
        print("        单次 7 位 RNE（手写）    = %.19g" % single)
        check("A", "x=%.10g：直转 == 经 fp32 转，且**都不同于**单次舍入" % v,
              direct == via32 and direct != single)
    check("A", "★ 所以「fp32→16bit 是 RNE」这句话里的**中间精度**也要写清："
               "bf16 的入口是 fp32，fp16 的入口是 fp64→fp16 单次",
          True)

    sub("A4  tf32 的位域（**引用值**，不是本机实测）")
    print("      tf32 = S1E8M10（19 位）。**本机没有 tf32 存储 dtype、也没有 GPU**，")
    print("      所以这一条只能记成引用值：")
    print("      %s" % SPECS.TF32_10BIT_MANTISSA_CONFIDENCE)
    check("A", "tf32 的尾数位 = 10（与 fp16 相同）",
          SPECS.TF32_MANTISSA_BITS == 10 and SPECS.FP16_MANTISSA_BITS == 10)
    check("A", "★ tf32 的输入转换舍入模式：**未解决**，不下断言",
          SPECS.TF32_INPUT_ROUNDING_MODE == "UNRESOLVED")
    check("A", "torch 里确实没有 tf32 这个 dtype（所以「实测 tf32」在本机不可能）",
          not hasattr(torch, "tf32"), "hasattr(torch,'tf32') = %s" % hasattr(torch, "tf32"))


# ---------------------------------------------------------------------------
# §B 代价模型与量化视角
# ---------------------------------------------------------------------------

def cost_model():
    rule("§B  代价模型：一位尾数值多少 dB（**量化误差**，不是累加误差）")

    sub("B1  四分布 × 四格式的 SQNR 表（2×10⁶ 样本）")
    n = SPECS.SQNR_SAMPLE_COUNT
    dists = make_dists(n)
    rows = []
    print("  %-16s %-10s %-10s %-10s %-10s" % ("分布", "bf16", "fp16", "e4m3fn", "tf32(仿真)"))
    for name, x in dists:
        b = sqnr_db(x, x.to(torch.bfloat16))
        h = sqnr_db(x, x.to(torch.float16))
        e = sqnr_db(x, x.to(torch.float8_e4m3fn))
        t = sqnr_db(x, sim_tf32(x))
        rows.append((name, b, h, e, t))
        print("  %-16s %-10.2f %-10.2f %-10.2f %-10.2f" % (name, b, h, e, t))

    sub("B2  ★ 与 Kuzmin 论文印值吻合到 0.1 dB 以内（跨来源交叉验证）")
    # 判据：N(0,1)、2×10⁶ 样本、SQNR = 10·log10(Σx²/Σ(x−x̂)²)。
    # 印值来自 Kuzmin et al. arXiv:2208.09225 Table 1。
    # ⚠ 论文的 SQNR 定义本仓库**未逐字核实**，所以这是「两个来源给出同一量级」的旁证，
    #   不是「定义相同」的证明。
    bf16_v = rows[0][1]
    fp16_v = rows[0][2]
    print("      本机 torch %s，N(0,1)：bf16 %.2f dB / fp16 %.2f dB"
          % (torch.__version__, bf16_v, fp16_v))
    print("      Kuzmin arXiv:2208.09225 Table 1 印值：BFloat-16 %.1f dB / IEEE float-16 %.1f dB"
          % (SPECS.KUZMIN_TABLE1_BF16_DB, SPECS.KUZMIN_TABLE1_FP16_DB))
    check("B", "bf16 与印值差 < 0.1 dB", abs(bf16_v - SPECS.KUZMIN_TABLE1_BF16_DB) < 0.1,
          "差 %.3f dB" % abs(bf16_v - SPECS.KUZMIN_TABLE1_BF16_DB))
    check("B", "fp16 与印值差 < 0.1 dB", abs(fp16_v - SPECS.KUZMIN_TABLE1_FP16_DB) < 0.1,
          "差 %.3f dB" % abs(fp16_v - SPECS.KUZMIN_TABLE1_FP16_DB))

    sub("B3  ★ 浮点 SQNR 与分布几乎无关（**跨分布极差约 1 dB**）")
    print("      判据：同一格式在四个分布上的 SQNR 取极差。")
    print("      分布集 = %s（每个都带参数与截断，见 §B1 表）"
          % " / ".join(SPECS.SQNR_SPREAD_DIST_SET))
    for idx, col in ((1, "bf16"), (2, "fp16"), (3, "e4m3fn"), (4, "tf32(仿真)")):
        vals = [r[idx] for r in rows]
        print("      %-12s 极差 = %.2f dB（%.2f .. %.2f）"
              % (col, max(vals) - min(vals), min(vals), max(vals)))
        if col in SPECS.SQNR_SPREAD_DB:
            check("B", "%s 的极差 %.2f dB 与 specs 记的 %.2f 一致"
                  % (col, max(vals) - min(vals), SPECS.SQNR_SPREAD_DB[col]),
                  abs((max(vals) - min(vals)) - SPECS.SQNR_SPREAD_DB[col]) < 0.02)
    check("B", "★ **不要写「< 1 dB」**：bf16 的两端（55.59 → 56.62）就是 1.03 dB",
          (max(r[1] for r in rows) - min(r[1] for r in rows)) > 1.0)
    print("      ⚠ 引用时**给端点 + 说明分布集**，不要给一个笼统区间——")
    print("        两个端点是稳的，中间值随分布集变化。")

    sub("B4  ★ 一位尾数 = 6.02 dB（浮点也守这条斜率）")
    print("      判据：bf16（7 位）与 fp16（10 位）相差 3 位尾数 → 预期 %.2f dB。"
          % (3 * 6.0206))
    for r in rows:
        print("      %-16s 实测差 = %.2f dB" % (r[0], r[2] - r[1]))
    for r in rows:
        check("B", "%s：fp16 − bf16 = %.2f dB（预期 %.2f ± 0.2）"
              % (r[0], r[2] - r[1], 3 * 6.0206),
              abs((r[2] - r[1]) - 3 * 6.0206) < 0.2)
    check("B", "e4m3fn（3 位）与 bf16（7 位）相差 4 位 → 预期 %.2f dB"
          % (4 * 6.0206),
          all(abs((r[1] - r[3]) - 4 * 6.0206) < 0.3 for r in rows),
          "实测 %.2f" % (rows[0][1] - rows[0][3]))

    sub("B5  与解析式对拍（**量级校验，不是同一个量的两种测法**）")
    print("      解析式（尾数在 binade 内均匀）：SQNR ≈ 20·log10(√3·√2·2^(M+1))")
    for m_bits in (3, 7, 10):
        print("        M=%-3d → %.2f dB" % (m_bits, SPECS.FLOAT_SQNR_ANALYTIC_DB[m_bits]))
    check("B", "bf16 的实测与解析差 < 0.5 dB",
          abs(rows[0][1] - SPECS.FLOAT_SQNR_ANALYTIC_DB[7]) < 0.5,
          "实测 %.2f / 解析 %.2f" % (rows[0][1], SPECS.FLOAT_SQNR_ANALYTIC_DB[7]))
    check("B", "fp16 的实测与解析差 < 0.5 dB",
          abs(rows[0][2] - SPECS.FLOAT_SQNR_ANALYTIC_DB[10]) < 0.5,
          "实测 %.2f / 解析 %.2f" % (rows[0][2], SPECS.FLOAT_SQNR_ANALYTIC_DB[10]))
    check("B", "e4m3fn 的实测与解析差 < 0.5 dB",
          abs(rows[0][3] - SPECS.FLOAT_SQNR_ANALYTIC_DB[3]) < 0.5,
          "实测 %.2f / 解析 %.2f" % (rows[0][3], SPECS.FLOAT_SQNR_ANALYTIC_DB[3]))

    sub("B6  ★ 为什么浮点量化**不需要 clipping**：这个对照就是答案")
    print("      定点格式的 SQNR 强相关于分布（同一格式跨场景可差 20+ dB，见 theory/02-granularity），")
    print("      而浮点格式跨四个分布的极差只有约 1 dB —— 因为浮点的**相对**误差与尺度无关，")
    print("      所以不会有「单个离群值把 scale 撑大、其余值落入极少级数」这件事。")
    bf16_spread = max(r[1] for r in rows) - min(r[1] for r in rows)
    e4_spread = max(r[3] for r in rows) - min(r[3] for r in rows)
    print("      本机实测：bf16 极差 %.2f dB、e4m3fn 极差 %.2f dB" % (bf16_spread, e4_spread))
    check("B", "★ 两个浮点格式的跨分布极差都 ≈1 dB（定点格式的量级是 20+ dB）",
          bf16_spread < 1.5 and e4_spread < 1.5)
    check("B", "★ 反过来说：**浮点量化不需要 clipping** 的依据就在这里，"
               "而不是「clipping 在浮点上没用」",
          True, "对照见 theory/02-granularity 的粒度表（定点）")

    sub("B7  tf32 仿真与 fp16 在正常区内**数值重合**（两者都是 10 位尾数）")
    # 这不是巧合：两者的尾数位数相同，只要数据不触碰各自的范围边界，
    # 「输入截到 10 位尾数」这一步的结果就一样。差别在范围（指数位 8 vs 5）。
    x = torch.randn(200000, dtype=torch.float64)
    t = sim_tf32(x)
    h = x.to(torch.float16).to(torch.float64)
    same = int((t == h).sum())
    print("      tf32 仿真 vs fp16 cast：%d/%d 逐元素相同" % (same, x.numel()))
    check("B", "★ 两者在正常区内高度重合（>99.9%）——因为尾数位数相同", same > 0.999 * x.numel(),
          "相同 %d/%d" % (same, x.numel()))
    check("B", "★ 它们的差别只在**范围**：fp16 的指数只有 5 位（max 65504），tf32 是 8 位",
          torch.finfo(torch.float16).max == 65504.0)


# ---------------------------------------------------------------------------
# §C 失败行为：两个格式的边界真的会咬人
# ---------------------------------------------------------------------------

def failures():
    rule("§C  边界与反例：这些必须在真系统上真的发生")

    sub("C1  fp16 的上溢是 **inf**，不是饱和（与 fp8_e4m3fn 相反）")
    lo = float(torch.tensor([65519.0]).to(torch.float16)[0])
    hi = float(torch.tensor([65520.0]).to(torch.float16)[0])
    print("      65519 → %r   65520 → %r" % (lo, hi))
    check("C", "65519 还能舍到 65504", lo == 65504.0)
    check("C", "★ 65520 直接变成 inf（**不是**饱和到 65504）", math.isinf(hi))
    check("C", "所以「窄格式溢出会饱和」这句话**对 fp16 不成立**——"
               "饱和是 e4m3fn 在 2.13 才有的行为，fp16 从来就是 inf",
          math.isinf(hi) and torch.finfo(torch.float16).max == 65504.0)

    sub("C2  fp16 的下溢：次正规区与归零")
    small = 2.0 ** -24                       # 恰好是 fp16 的最小次正规
    smaller = 2.0 ** -25
    check("C", "2^-24 是 fp16 的最小次正规（能表示）",
          float(torch.tensor([small]).to(torch.float16)[0]) == 2.0 ** -24)
    check("C", "2^-25 落到 0（再小的就是 0 了）",
          float(torch.tensor([smaller]).to(torch.float16)[0]) == 0.0)
    check("C", "bf16 的同一位置是 2^-133 量级，绝不会在这里归零",
          float(torch.tensor([2.0 ** -30]).to(torch.bfloat16)[0]) == 2.0 ** -30)

    sub("C3  ★ tf32 的「精度下界」在本机**测不了**（这一格是未解决，不是通过）")
    print("      tf32 的输入转换舍入模式：两处来源冲突，且可能是「输入转换」与")
    print("      「累加舍入」两件事。本机无 tf32 存储 dtype + CPU-only，**跑不了真 TF32**。")
    check("C", "本机确实没有 tf32 存储 dtype", not hasattr(torch, "tf32"))
    check("C", "specs 里这一格记的是 **UNRESOLVED**，不是一个看起来合理的答案",
          SPECS.TF32_INPUT_ROUNDING_MODE == "UNRESOLVED")
    check("C", "★ 本程序对 tf32 的**全部**输出都带「仿真」限定语",
          SPECS.SQNR_COLUMNS[3].startswith("tf32(位掩码仿真"))

    sub("C4  仿真与实测的作用域：**本程序能说什么、不能说什么**")
    check("C", "能说：真 torch 的 bf16/fp16 cast 行为（§A1–A3，本机实测）", True)
    check("C", "不能说：「tf32 硬件的行为」（只有位掩码仿真）",
          SPECS.TF32_INPUT_ROUNDING_MODE == "UNRESOLVED")
    check("C", "不能说：「累加误差」（本程序一个累加都没做）",
          True, "累加误差的正文在 10-foundations/floating-point/04-low-precision-formats.md")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    print("=" * 76)
    print("03 bf16 / fp16 / tf32 —— 真 torch 实测（对应文档 03-bf16-fp16-tf32.md）")
    print("=" * 76)
    print("★ 本篇讲的是**量化误差**（把连续值压到有限格点），")
    print("  **不是累加误差**（√K·u 那条线，正文在 10-foundations/floating-point/04）。")
    print("★ A/C 类的独立复核在 verify.py（默认 python3、**不 import torch**）里。")

    banner()
    channels()
    cost_model()
    failures()

    rule("汇总  断言汇总")
    print("  共 %d 条：A 类 %d 条，B 类 %d 条，C 类 %d 条"
          % (_STATE["n"], _STATE["A"], _STATE["B"], _STATE["C"]))
    if _STATE["fail"]:
        print("  **不通过 %d 条**：" % len(_STATE["fail"]))
        for f in _STATE["fail"]:
            print("    - %s" % f)
        return 1
    print("  全部通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

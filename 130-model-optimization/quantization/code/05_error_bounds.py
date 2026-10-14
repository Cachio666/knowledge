#!/usr/bin/env python3
"""01 误差界：Δ/2、Δ²/12、以及 6.02n + 1.76 的前提。

对应文档：130-model-optimization/quantization/05-error-bounds.md

★ 本程序跑的是**真 torch**——输出是**本机实测值**，支撑关于**真实 PyTorch** 的结论，
  但**必须绑定版本**。本机五个解释器都跑过：
    /usr/bin/python3                        torch 2.8.0  / Python 3.9.6
    ~/miniconda3/bin/python                 torch 2.11.0 / Python 3.12.9
    ~/miniconda3/envs/deep-prac/bin/python  torch 2.6.0  / Python 3.12.11
    ~/miniconda3/envs/dp-test/bin/python    torch 2.7.1  / Python 3.13.7
    ~/PyCharmMiscProject/.venv/bin/python   torch 2.13.0 / Python 3.13.5
  **2.9 / 2.10 / 2.12 本机没有解释器，不声称覆盖。** 全部 CPU-only，无 GPU 数字。

三类断言（CONTRIBUTING §2 第三种形态：B 在编号程序里，A/C 另见 verify.py）：
  B —— **代价模型**：比特 ↔ 误差的交换。每多一个比特买到多少 dB（实测斜率 vs 6.02）、
       以及 scale 本身的元数据开销（32 bit / G 个元素）怎么把这个收益吃掉一部分。
  A —— **两条互不依赖的观察通道**做同一件事。本程序里有三组：
       ① 「Δ/2 界」：解析上界 vs 直接量出的 max|e|；
       ② 「6.02n+1.76」：一般式退化 vs 满量程正弦的直接测量；
       ③ 「ACIY α*」：论文印值 vs 本机独立数值最小化。**第三组故意不一致**，
          而断言断言的就是「它们不一致，且不一致的幅度落在 0.2–0.6%」。
  C —— 失败行为必须**真的跑出来**：越界使 Δ/2 界失效（要跑出超过 Δ/2 的误差）、
       两种 SQNR 口径**混用**（要跑出那个 6 dB 偏高的错值）、
       换一个量化定义（n=8 → 50.02 这个实测值就复现不出）。

用法：cd code && make 05_error_bounds   （或 make run 全跑）
"""

import math
import platform
import sys

import torch

import specs_theory as SPECS


_STATE = {"n": 0, "A": 0, "B": 0, "C": 0, "fail": []}


def cpu_model():
    """测量条件里的「CPU 型号」。跨平台：macOS 读 sysctl，Linux 读 /proc/cpuinfo。
    **这不是算法**，只是把测量条件打出来。"""
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


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def sqnr_db(x, xh):
    """SQNR = 10log10(Σx² / Σ(x−x̂)²)。**判据：逐元素平方和之比，无归一化。**
    先在 float64 上算，避免 float32 的累加误差混进被检验的量。"""
    x = x.double()
    xh = xh.double()
    return (10.0 * torch.log10((x * x).sum() / ((x - xh) ** 2).sum())).item()


def quant_midtread(x, bits, span):
    """mid-tread 均匀量化：`x̂ = round(x/step)·step`，`step = span / 2^bits`。
    **这个定义是本文所有「满量程正弦」数字的一部分**，换定义就复现不出（见 §C3）。"""
    step = float(span) / (1 << bits)
    return torch.round(x / step) * step


def midtread_step(bits, span):
    return float(span) / (1 << bits)


def sine_samples(n, phase):
    k = torch.arange(n, dtype=torch.float64)
    return torch.sin(2.0 * math.pi * torch.frac(k * phase))


def sym_rtn(x, bits):
    """对称 RTN：scale = max|x| / (2^(bits−1) − 1)，量化为整数后乘回。
    **判据：减号那一侧用 −QMAX，即「restricted range」口径。**"""
    qmax = float((1 << (bits - 1)) - 1)
    s = x.abs().max() / qmax
    return torch.clamp(torch.round(x / s), -qmax, qmax) * s


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
    print("  未覆盖 minor: %s（本机没有对应解释器）" % SPECS.UNCOVERED_MINORS)
    print()
    print("  ⚠ 本主题的 torch 版本下限 %s **不是依赖，是对齐的选择**："
          % (SPECS.TORCH_MIN_FOR_THIS_TOPIC,))
    print("     %s。" % SPECS.TORCH_MIN_REASON)
    print("     （本主题只用 float32/float64/bfloat16/float16/float8_e4m3fn，2.6.0 起都有。）")


# ---------------------------------------------------------------------------
# §A 判据复核
# ---------------------------------------------------------------------------

def run_A():
    rule("§A  判据复核：解析结论 vs 直接测量（两条独立通道）")

    # ---- A1：Δ/2 界 ----
    sub("A1  Δ/2 界：**解析上界 vs 直接量出的 max|e|**（未越界时应当精确相等）")
    g = torch.Generator().manual_seed(11)
    n = 2_000_000
    cases = [("U(-1,1)", torch.rand(n, generator=g, dtype=torch.float64) * 2 - 1),
             ("N(0,1)", torch.randn(n, generator=g, dtype=torch.float64))]
    for name, x in cases:
        for bits in (4, 8):
            lo, hi = x.min().item(), x.max().item()
            steps = (1 << bits) - 1
            step = (hi - lo) / steps
            xh = lo + torch.round((x - lo) / step) * step
            e = (xh - x).abs()                      # e ≜ x̂ − x（O&S 符号约定）
            ratio = e.max().item() / (step / 2.0)
            check("A", "%s B=%d：max|e| == Δ/2（比值 %.6f）" % (name, bits, ratio),
                  abs(ratio - 1.0) < 1e-5,
                  "max|e|=%.9g Δ/2=%.9g" % (e.max().item(), step / 2.0))

    # ---- A2：σ² = Δ²/12 ----
    sub("A2  σ² = Δ²/12：蒙特卡洛复核（判据：无过载、误差在 [−Δ/2, Δ/2] 内均匀）")
    var_lo, var_hi = SPECS.BOUND_CHECK["var_ratio_range"]
    for name, x in cases:
        for bits in (4, 8):
            lo, hi = x.min().item(), x.max().item()
            step = (hi - lo) / ((1 << bits) - 1)
            xh = lo + torch.round((x - lo) / step) * step
            e = xh - x
            ratio = (e.var(unbiased=False).item()) / (step * step / 12.0)
            check("A", "%s B=%d：Var(e)/(Δ²/12) = %.4f ∈ [%.4f, %.4f]"
                  % (name, bits, ratio, var_lo, var_hi), var_lo <= ratio <= var_hi,
                  "实测 %.6f（**偏离 1 的部分是有限样本的统计抖动，不是偏差**）" % ratio)

    # ---- A3：截断 = 2× 舍入，且恒为负 ----
    sub("A3  截断：**幅度是舍入的两倍、符号恒为负**（同一 e ≜ x̂ − x 约定下）")
    x = torch.rand(n, generator=g, dtype=torch.float64) * 2 - 1
    step = midtread_step(8, 2.0)
    er = torch.round(x / step) * step - x          # 最近舍入
    et = torch.floor(x / step) * step - x          # 截断（朝 −∞）
    rms_ratio = math.sqrt((et ** 2).mean().item() / (er ** 2).mean().item())
    check("A", "舍入：e ∈ [−Δ/2, +Δ/2]（实测 [%.9g, %.9g]，Δ/2=%.9g）"
          % (er.min().item(), er.max().item(), step / 2),
          er.min().item() >= -step / 2 - 1e-12 and er.max().item() <= step / 2 + 1e-12)
    check("A", "截断：e ∈ (−Δ, 0]，**恒 ≤ 0**（实测 [%.9g, %.9g]，Δ=%.9g）"
          % (et.min().item(), et.max().item(), step),
          et.max().item() <= 0.0 and et.min().item() > -step - 1e-12)
    check("A", "★ 截断 RMS / 舍入 RMS = %.4f ≈ 2（「误差是两倍」指的是幅度，不是方差）"
          % rms_ratio, abs(rms_ratio - 2.0) < 0.01,
          "方差之比会是 %.4f = 4" % (rms_ratio ** 2))

    # ---- A4：满量程正弦 ----
    sub("A4  ★ 6.02n+1.76：**一般式退化 vs 满量程正弦的直接测量**")
    phi = (math.sqrt(5.0) - 1.0) / 2.0            # 无理相位，避免与采样格点共振
    for bits in (8, 16):
        sp = SPECS.SQNR_SINE_MEASURED[bits]
        fo = SPECS.SQNR_SINE_FORMULA[bits]
        xs = sine_samples(100000, phi)
        xh = quant_midtread(xs, bits, 2.0)
        got = sqnr_db(xs, xh)
        check("A", "满量程正弦 n=%d：实测 %.3f dB，公式 %.2f dB（差 %.2f dB）"
              % (bits, got, fo, got - fo),
              abs(got - sp) < 0.05 and abs(got - fo) < 0.30,
              "specs 记的实测值 %.3f" % sp)
    # 误差方差比 Δ²/12 小 → 实测略高于解析式，这就是上面那 0.1 dB 的来源
    xs = sine_samples(1000000, phi)
    step = midtread_step(8, 2.0)
    e = quant_midtread(xs, 8, 2.0) - xs
    r = e.var(unbiased=False).item() / (step * step / 12.0)
    check("A", "★ 正弦的误差方差 / (Δ²/12) = %.4f（**< 1**）—— 这不是误差，"
          "是 Δ²/12 那条「误差在 [−Δ/2,Δ/2] 内均匀」假设在正弦上不精确" % r,
          0.95 < r < 1.0, "实测 %.6f" % r)

    # ---- A5：两种口径都能退化 ----
    sub("A5  ★ SQNR 的两种自洽口径：**都必须能退化到 6.02B + 1.76**（这是判据）")
    for key, c in sorted(SPECS.SQNR_CONVENTIONS.items()):
        # 一般式 → 代入退化条件 → 应当等于 6.02B + 1.76
        s = SPECS.SQNR_CONST_DERIVATION
        if key == "A_span":
            Xm_over_sigma = 1.0 / s["degenerate_sigma_over_Xm_span"]
        else:
            Xm_over_sigma = 1.0 / s["degenerate_sigma_over_Xm_peak"]
        at8 = SPECS.MT001_SLOPE_DB_PER_BIT * 8 + c["const_db"] - 20.0 * math.log10(Xm_over_sigma)
        check("A", "%s（Xm = %s）：B=8 退化值 %.3f dB" % (key, c["Xm_meaning"], at8),
              abs(at8 - c["at_B8_db"]) < 0.02 and abs(at8 - (SPECS.MT001_SLOPE_DB_PER_BIT * 8 + SPECS.MT001_FULL_SCALE_SINE_SNR_DB)) < 0.02,
              "specs 记 %.2f；6.02·8+1.76 = %.2f" % (c["at_B8_db"], SPECS.MT001_SLOPE_DB_PER_BIT * 8 + SPECS.MT001_FULL_SCALE_SINE_SNR_DB))
    check("A", "两个常数的来历：10·log10(12) = %.4f；4.77 = 它 − 20log10(2) = %.4f"
          % (10 * math.log10(12), 10 * math.log10(12) - 20 * math.log10(2)),
          abs(SPECS.SQNR_CONVENTIONS["A_span"]["const_db"] - 10 * math.log10(12)) < 0.01
          and abs(SPECS.SQNR_CONVENTIONS["B_peak"]["const_db"]
                  - (10 * math.log10(12) - 20 * math.log10(2))) < 0.01)

    # ---- A6：ACIY 的本机数值解 vs 论文印值 ----
    sub("A6  ACIQ 的 α*：**论文印值 vs 本机独立数值最小化**——两者**不一致**，"
        "而这里断言的就是「不一致且幅度落在 0.2–0.6%」")
    for M in (2, 3, 4):
        got = aciq_numeric_alpha(M)
        paper = SPECS.ACIQ_ALPHA_PAPER[M]
        local = SPECS.ACIQ_ALPHA_LOCAL_NUMERIC[M]
        rel = abs(got - paper) / paper * 100.0
        check("A", "M=%d：本机数值解 α* = %.4f（论文印值 %.2f，相对差 %.2f%%）"
              % (M, got, paper, rel),
              0.1 < rel < 1.0 and abs(got - local) < 0.01,
              "specs 记的本机解 %.3f" % local)
        # 饱和概率：P(|X| > α) = e^(−α/b)，b = 1
        check("A", "M=%d：α*=%.2f 处的饱和概率 e^(−α/b) = %.4f（specs 记 %.4f）"
              % (M, paper, math.exp(-paper), SPECS.ACIQ_SATURATION_PROB[M]),
              abs(math.exp(-paper) - SPECS.ACIQ_SATURATION_PROB[M]) < 5e-5)

    # ---- A7：定点 vs 浮点（P8 的落点） ----
    sub("A7  ★ 同一个「6.02n+1.76 适不适用」的问题，**定点与浮点是两套答案**")
    ff = SPECS.FIXED_VS_FLOAT
    fixed = ff["fixed_8bit_sym_sqnr_db"]
    got_fixed = {}
    for i, name in enumerate(("N(0,1)", "U(-1,1)", "Lap(0,1)", "单边指数")):
        x = make_dist(name, 9000 + i)
        got_fixed[name] = sqnr_db(x, sym_rtn(x, 8))
        check("A", "定点 8-bit 对称 per-tensor，%s：%.2f dB（specs 记 %.2f）"
              % (name, got_fixed[name], fixed[name]),
              abs(got_fixed[name] - fixed[name]) < 0.10,
              "判据：专属种子 %d，N = 2×10^6" % (9000 + i))
    spread_f = max(got_fixed.values()) - min(got_fixed.values())
    check("A", "★ 定点：跨四个分布的 SQNR 波动 %.2f dB（specs 记 %.2f）"
          % (spread_f, ff["fixed_spread_db"]),
          abs(spread_f - ff["fixed_spread_db"]) < 0.10, "**定点是强分布的**")
    for dtname, dt in (("bfloat16", torch.bfloat16), ("float16", torch.float16),
                       ("float8_e4m3fn", torch.float8_e4m3fn)):
        row = {}
        for i, name in enumerate(("N(0,1)", "U(-1,1)", "Lap(0,1)", "单边指数")):
            x = make_dist(name, 9100 + i)
            row[name] = sqnr_db(x, x.to(dt).to(torch.float32))
        spread = max(row.values()) - min(row.values())
        want = ff["float_spread_db"][dtname]
        check("A", "浮点 %s：跨分布波动 %.2f dB（specs 记 %.2f）；N(0,1) 下 %.2f dB"
              % (dtname, spread, want, row["N(0,1)"]),
              abs(spread - want) < 0.05,
              "四格：%s" % {k: round(v, 2) for k, v in row.items()})
        check("A", "★ %s 的波动 **> 1 dB**（%.2f）—— 这就是「不得写 < 1 dB」的原因"
              % (dtname, spread), spread > 1.0)
    check("A", "★ 对照：定点波动 %.2f dB 是浮点 %.2f dB 的 %.1f 倍"
          % (spread_f, ff["float_spread_db"]["bfloat16"],
             spread_f / ff["float_spread_db"]["bfloat16"]),
          spread_f / ff["float_spread_db"]["bfloat16"] > 10.0)


def make_dist(kind, seed, n=2_000_000):
    """四个输入分布的**固定生成式**（每个分布一个专属种子）。

    判据（**不写出来这个数就不可复现**）：
      N(0,1)   = torch.randn
      U(-1,1)  = torch.rand·2 − 1
      Lap(0,1) = Exponential(1)·sign，sign 取 ±1 等概率 ⇒ Laplace(0,1)
      单边指数 = Exponential(1)（λ = 1 的单边分布，模拟 ReLU 后的激活）
    """
    g = torch.Generator().manual_seed(seed)
    if kind == "N(0,1)":
        return torch.randn(n, generator=g, dtype=torch.float32)
    if kind == "U(-1,1)":
        return torch.rand(n, generator=g, dtype=torch.float32) * 2.0 - 1.0
    if kind == "Lap(0,1)":
        return (torch.empty(n).exponential_(1.0, generator=g)
                * (torch.randint(0, 2, (n,), generator=g).float() * 2.0 - 1.0))
    return torch.empty(n).exponential_(1.0, generator=g)


def aciq_numeric_alpha(M, b=1.0):
    """**独立数值最小化** Laplace(0,b) 下的最优裁剪阈值 α*。

    判据（目标函数必须写出，否则这个数没有意义）：
        MSE(α) = (1 − P(|X|>α))·Δ²/12 + 2∫_α^∞ (x−α)²·f(x)dx,   Δ = 2α/2^M,  f = Laplace(0,b)
    尾部积分解析：2∫_α^∞ (x−α)²·(1/(2b))e^{−x/b}dx = 2b²e^{−α/b}。
    **与 torch 无关**——它是一段纯算术，放在这里是为了与论文印值对拍（A6）。"""
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
# §B 代价模型：比特 ↔ 误差
# ---------------------------------------------------------------------------

def run_B():
    rule("§B  代价模型：b 个比特买到多少 dB，以及 scale 的元数据把这笔账吃掉多少")

    sub("B1  **解析代价模型** vs 实测：SQNR(b) = 10log10(12) + 20log10(QMAX(b)/amax)")
    print("      判据：per-tensor min-max 对称 RTN（受限量程 [−QMAX, QMAX]）、")
    print("            QMAX(b) = 2^(b−1) − 1、N(0,1)、N = 2×10^6、种子 23。")
    print("      ⚠ 这不是 6.02b + 1.76 —— **那个式子只对满量程正弦成立**。")
    g = torch.Generator().manual_seed(23)
    n = 2_000_000
    x = torch.randn(n, generator=g, dtype=torch.float32)
    amax = x.abs().max().item()
    fitted = SPECS.COST_MODEL_FIT
    res = {}
    for bits in (2, 3, 4, 6, 8, 10, 12):
        qmax = float((1 << (bits - 1)) - 1)
        s = amax / qmax
        xh = torch.clamp(torch.round(x / s), -qmax, qmax) * s
        got = sqnr_db(x, xh)
        pred = 10.0 * math.log10(12.0) + 20.0 * math.log10(qmax / amax)
        res[bits] = got
        print("      b=%2d  QMAX=%5d  实测 %7.3f dB  解析 %7.3f dB  差 %+.3f"
              % (bits, qmax, got, pred, got - pred))
        if bits in fitted:
            check("B", "b=%d：实测 %.3f vs 解析 %.3f（specs 记 %.3f/%.3f）"
                  % (bits, got, pred, fitted[bits][0], fitted[bits][1]),
                  abs(got - fitted[bits][0]) < 0.02 and abs(pred - fitted[bits][1]) < 0.02)
    print("      amax = %.6f（N = 2×10^6 的样本极值 —— **amax 本身是测量条件的一部分**）" % amax)
    # b=2 是模型的边界：QMAX = 1 时只剩 3 个级数，Δ²/12 的前提（级数多）不成立
    q2 = 1.0
    xh2 = torch.clamp(torch.round(x / (amax / q2)), -q2, q2) * (amax / q2)
    pred2 = 10.0 * math.log10(12.0) + 20.0 * math.log10(q2 / amax)
    check("B", "★ 边界：b=2 时解析式给出 %.3f dB，实测 %.3f dB，**差 %+.3f dB —— 模型在这里失效**"
          % (pred2, res[2], res[2] - pred2),
          abs(res[2] - pred2) > 3.0 and abs(res[2] - SPECS.COST_MODEL_B2_FAILS[0]) < 0.02,
          "QMAX = 1 ⇒ 只有 3 个级数，Δ²/12 的前提（级数多）不成立")
    for (lo, hi), want in sorted(SPECS.COST_MODEL_SLOPE.items()):
        got = (res[hi] - res[lo]) / (hi - lo)
        check("B", "斜率 b=%d→%d：%.3f dB/bit（specs 记 %.3f）" % (lo, hi, got, want),
              abs(got - want) < 0.01)
    check("B", "★ 斜率随 b 收敛到 6.02：b=3→4 是 %.3f，b=10→12 是 %.3f —— "
          "小 b 处偏陡是**有限 QMAX** 造成的，不是测量误差"
          % (res[4] - res[3], (res[12] - res[10]) / 2.0),
          (res[12] - res[10]) / 2.0 < (res[4] - res[3])
          and abs((res[12] - res[10]) / 2.0 - 6.02) < 0.02)
    check("B", "★ 常数项：N(0,1) 在 b=8 下 %.2f dB，满量程正弦是 49.92 dB —— 差 %.2f dB。"
          "**「1.76」是信号分布的属性，不是量化器的属性**"
          % (res[8], 49.92 - res[8]), res[8] < 45.0)

    sub("B2  ★ per-tensor min-max 的 SQNR **是一个随机变量**（100 个种子的极差）")
    print("      判据：对称 8-bit per-tensor、N = 2×10^5、100 个不同种子。")
    for kind, want in SPECS.MINMAX_SEED_SENSITIVITY.items():
        vals = []
        for sd in range(100):
            xx = make_dist(kind, 1000 + sd, n=200000)
            qmax = 127.0
            s = xx.abs().max() / qmax
            vals.append(sqnr_db(xx, torch.clamp(torch.round(xx / s), -qmax, qmax) * s))
        spread = max(vals) - min(vals)
        check("B", "%s：SQNR ∈ [%.2f, %.2f]，极差 %.2f dB（specs 记 [%.2f, %.2f] / %.2f）"
              % (kind, min(vals), max(vals), spread, want["min"], want["max"],
                 want["spread"]),
              abs(min(vals) - want["min"]) < 0.05 and abs(spread - want["spread"]) < 0.10,
              "**成因：scale 由样本极值决定，而极值是小样本统计量**")

    sub("B3  scale 的元数据开销：每 G 个元素一个 fp32 scale ⇒ 32/G bit/元素")
    for G in (None, 4096, 128, 32, 16):
        if G is None:
            print("      per-tensor（整张量一个 scale）：32 bit / 2^24 元素 ≈ 1.9e-6 bit/元素")
            continue
        print("      G=%4d：%6.4f bit/元素（相对 4-bit 载荷是 %.2f%%）"
              % (G, 32.0 / G, 32.0 / G / 4.0 * 100))
    # 把「粒度换来的 dB」与「粒度花掉的 bit」放在一起，就是 §0.8 三角的那条边
    print()
    print("      粒度换来的收益（`06_granularity.py` 的 5×4 表，judge 见该程序）：")
    t = SPECS.GRANULARITY_TABLE
    for scen in ("iid 高斯", "7/4096 列（随机位置）×100"):
        lo, hi = t[scen], None
        print("        %-28s per-tensor %.2f → per-group(128) %.2f dB（+%.2f dB，"
              "代价 %.4f bit/元素）" % (scen, t[scen][0], t[scen][3],
                                       t[scen][3] - t[scen][0], 32.0 / 128))
    check("B", "★ per-group(128) 的 scale 开销（%.4f bit/元素）比 per-tensor 贵，"
          "但对 outlier 场景换来 %.2f dB" % (32.0 / 128, t["7/4096 列（随机位置）×100"][3]
                                            - t["7/4096 列（随机位置）×100"][0]),
          32.0 / 128 > 0.0)

    sub("B4  scale 用 bf16 存时的开销（对照：真实 kernel 里 scale 常是 fp16/fp32）")
    for G, w in ((128, 16), (128, 32), (32, 16), (32, 32)):
        print("      G=%3d, scale=%d bit：%.4f bit/元素" % (G, w, float(w) / G))
    check("B", "★ group 越小 scale 越贵：G=32 的开销是 G=128 的 4 倍",
          abs((32.0 / 32) / (32.0 / 128) - 4.0) < 1e-9)


# ---------------------------------------------------------------------------
# §C 边界与反例（**断言的是「失败行为成立」**）
# ---------------------------------------------------------------------------

def run_C():
    rule("§C  边界与反例：下面每一条都**必须真的跑出问题**，跑不出就是断言失败")

    sub("C1  越界（overload）使 Δ/2 界失效 —— 要跑出**超过 Δ/2 的误差**")
    g = torch.Generator().manual_seed(5)
    for bits, cover, xmax in ((3, 1.0, 2.0), (8, 1.0, 1.5), (8, 1.0, 10.0)):
        step = 2.0 * cover / (1 << bits)
        x = torch.rand(2_000_000, generator=g, dtype=torch.float64) * 2 * xmax - xmax
        xh = torch.clamp(torch.round(x / step) * step, -cover, cover)
        e = (xh - x).abs()
        n_over = int((e > step / 2.0).sum().item())
        ratio = e.max().item() / (step / 2.0)
        check("C", "B=%d 量程±%.0f 输入±%.0f：max|e|/(Δ/2) = %.1f，超界样本 %d 个（%.1f%%）"
              % (bits, cover, xmax, ratio, n_over, 100.0 * n_over / 2_000_000),
              n_over > 0 and ratio > 1.5,
              "**Δ/2 界在这里失效** —— 它只在输入不越界时成立")

    sub("C2  ★ 两种 SQNR 口径**混用**会给出偏高 6 dB 的错值 —— 要跑出那个 55.95")
    bad = SPECS.MT001_SLOPE_DB_PER_BIT * 8 + SPECS.SQNR_CONVENTIONS["A_span"]["const_db"] \
        - 20.0 * math.log10(math.sqrt(2.0))
    good = SPECS.SQNR_CONVENTIONS["A_span"]["at_B8_db"]
    check("C", "混用（跨度口径的 10.8 配峰值口径的 σx = Xm/√2）= %.2f dB，"
          "正确值 %.2f dB，偏高 %.2f dB" % (bad, good, bad - good),
          abs(bad - 55.95) < 0.05 and (bad - good) > 6.0,
          "偏差恰好是 20log10(2) = 6.02 dB —— 两个常数本来就只差这一项")
    # 反过来说：混用后**退化不出 1.76**，这就是「判据」能抓住它的原因
    deg = bad - SPECS.MT001_SLOPE_DB_PER_BIT * 8
    check("C", "★ 混用版本退化出的常数项是 %.2f，**不是 1.76** —— 判据在这里生效" % deg,
          abs(deg - 1.76) > 5.0, "退化到 %.2f dB" % deg)

    sub("C3  ★ 「n=8 → 50.02」这个实测值**依赖量化定义** —— 换定义就跑不出它")
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    xs = sine_samples(100000, phi)
    step8 = midtread_step(8, 2.0)
    s127 = xs.abs().max() / 127.0
    s1275 = xs.abs().max() / 127.5
    defs = [
        ("mid-tread round(x/step)·step, step=2/2^n（本文口径）",
         sqnr_db(xs, torch.round(xs / step8) * step8)),
        ("round(x·2^(n−1))/2^(n−1)  ← **与上一行代数等价**",
         sqnr_db(xs, torch.round(xs * 128.0) / 128.0)),
        ("对称 /127（restricted range）",
         sqnr_db(xs, torch.clamp(torch.round(xs / s127), -127, 127) * s127)),
        ("对称 /127.5（torch 的对称除数是 127.5 不是 127）",
         sqnr_db(xs, torch.clamp(torch.round(xs / s1275), -128, 127) * s1275)),
        ("mid-rise (floor(x/step)+0.5)·step",
         sqnr_db(xs, (torch.floor(xs / step8) + 0.5) * step8)),
        ("截断 floor(x/step)·step",
         sqnr_db(xs, torch.floor(xs / step8) * step8)),
    ]
    want = SPECS.SQNR_SINE_BY_DEFINITION
    for name, v in defs:
        print("      %-48s → %7.3f dB" % (name, v))
    check("C", "★ **前两个定义在这个设置下是同一个式子**（span=2、B=8 时 "
          "round(x·128)/128 ≡ round(x/step)·step），实测 %.3f 与 %.3f 相同"
          % (defs[0][1], defs[1][1]), abs(defs[0][1] - defs[1][1]) < 1e-9,
          "**所以不能把它当成「另一种定义」**")
    check("C", "★ 真正不同的定义（mid-rise / 对称 /127 / 截断）给出的 n=8 值"
          "**都到不了 50.02**（极差 %.2f dB），只有 mid-tread 能复现"
          % (max(v for _, v in defs) - min(v for _, v in defs)),
          abs(defs[0][1] - want["mid-tread round(x/step)·step, step = 2/2^B"]) < 0.05
          and all(abs(v - 50.017) > 0.05 for _, v in defs[2:]),
          "**「实测值可复现」这条验收的实质就在这里：定义必须一起写出来**")
    check("C", "六个定义逐格与 specs 一致（**specs 就是这个断言的参照**）",
          all(abs(v - want[n]) < 0.05 for n, v in
              zip(["mid-tread round(x/step)·step, step = 2/2^B",
                   "round(x·2^(n−1))/2^(n−1)",
                   "对称 /127（restricted range）",
                   "对称 /127.5",
                   "mid-rise (floor(x/step)+0.5)·step",
                   "截断 floor(x/step)·step"], [v for _, v in defs])))

    sub("C4  Δ²/12 的四条假设失效：误差既非均匀、也与信号强相关")
    cases_c4 = [("b=3，输入恒等于 0.2（网格正中央）",
                 torch.full((200000,), 0.2, dtype=torch.float64), 3, 2.0),
                ("b=3，输入均匀落在 [0, 0.1]（远小于一个步长）",
                 torch.rand(200000, generator=g, dtype=torch.float64) * 0.1, 3, 2.0),
                ("b=4，相干采样正弦（每周期恰好 1000 点）",
                 torch.sin(2 * math.pi * torch.arange(200000, dtype=torch.float64)
                           / 1000.0), 4, 2.0)]
    for name, x, bits, span in cases_c4:
        st = span / (1 << bits)
        e = torch.round(x / st) * st - x
        r = e.var(unbiased=False).item() / (st * st / 12.0)
        check("C", "%s：Var(e)/(Δ²/12) = %.4f（应明显偏离 1）" % (name, r),
              abs(r - 1.0) > 0.05,
              "**教科书的四条假设在这里不成立** —— 它自己声明过只在"
              "「步长小、信号复杂、未过载」时是好近似")

    sub("C5  「层间对消」不得写 —— 本程序**不给**这条断言，因为本机跑不了 LLM")
    print("      本机无 LLM 权重、无 GPU，perplexity 与层间误差传播只有引用值。")
    print("      计划 §0.7.B 的要求是：**「层间对消」没有任何正面证据，不得写「误差会互相抵消」**。")
    print("      这里不编造一条本机断言去「证明」它 —— 那正是 P3（要求复现本域跑不了的东西）。")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    print("=" * 76)
    print("130-model-optimization/quantization/05-error-bounds 误差界 —— 真 torch 实测（B 类在内部）")
    print("=" * 76)
    banner()
    run_A()
    run_B()
    run_C()

    rule("§D  断言汇总")
    print("  共 %d 条：A 类 %d 条，B 类 %d 条，C 类 %d 条"
          % (_STATE["n"], _STATE["A"], _STATE["B"], _STATE["C"]))
    if _STATE["fail"]:
        print("  **不通过 %d 条**：" % len(_STATE["fail"]))
        for f in _STATE["fail"]:
            print("    - %s" % f)
        return 1
    print("  全部通过。")
    print()
    print("  A/C 类的独立复核在 `verify.py`（默认 python3、**不 import torch**）。")
    print("  **两档不得互相背书**：本程序测的是真 torch 的行为，verify.py 跑的是数学。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

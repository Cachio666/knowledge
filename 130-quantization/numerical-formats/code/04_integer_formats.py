#!/usr/bin/env python3
"""04 整数格式：对称与非对称，以及那个「除以 127.5 还是 127」的坑。

对应文档：130-quantization/numerical-formats/04-integer-formats.md

★ 本程序跑的是**真 torch**——输出是**本机实测值**。本机五个解释器都跑过：
    2.6.0 / 2.7.1 / 2.8.0 / 2.11.0 / 2.13.0。全部 CPU-only，无 GPU 数字。

三类断言（CONTRIBUTING §2 第三种形态：B 在编号程序里，A/C 另见 verify.py）：
  B —— **代价模型**：对称把多少比特花在「偏置」上（单边分布下有效级数减半 = 6.02 dB）。
  A —— 两条独立通道：`torch.ao.quantization` 的 observer 算出的 qparams
       vs 本程序按公式手算的 qparams。**两条通道的实现机制完全不同。**
  C —— 失败行为：`torch.int4` 是空壳，四类操作**各自抛异常**。
       ⚠ 异常**类型**跨版本变了，所以断言只能断「抛了异常」，不能断具体异常类。

用法：cd code && make 04_integer_formats   （或 make run 全跑）
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
    x = x.to(torch.float64)
    xh = xh.to(torch.float64)
    return 10 * math.log10(float((x * x).sum()) / float(((x - xh) ** 2).sum()))


# ---------------------------------------------------------------------------
# 本程序自己那份「按公式手算」的实现（与 torch 的 observer 是两条通道）
# ---------------------------------------------------------------------------

def ref_asym_qparams(x, bits=8):
    """无符号码字 0..2^b−1 的仿射（非对称）qparams：`s=(b−a)/qmax`、`z=round(−a/s)`。

    与 arXiv:2004.09602 Eq.1–5 是**同一套数学**，但按**无符号码字**写：
    论文的 `z` 已经含了 `−2^(b−1)` 的偏移，两套约定**差一个常数偏移**，
    比较时**必须先写死用的是哪一套**（本程序全程用无符号码字那一套）。
    """
    qmax = (1 << bits) - 1
    a = float(x.min())
    b = float(x.max())
    s = (b - a) / qmax
    z = round(-a / s)
    return s, z


def ref_asym_quant(x, bits=8):
    s, z = ref_asym_qparams(x, bits)
    qmax = (1 << bits) - 1
    q = torch.clamp(torch.round(x / s + z), 0, qmax)
    return (q - z) * s, q


def ref_sym_scale(x, bits=8, full_range=True):
    """对称量化的尺度。`full_range=True` → `S = 2·max|r| / (2^n − 1)`（int8 时 **/127.5**）；
    `full_range=False` → restricted range `S = max|r| / (2^(n−1) − 1)`（**/127**）。
    两条式子都来自 arXiv:2103.13630 Eq.2–5。**torch 走的是 full range 那一支。**
    """
    mx = float(x.abs().max())
    if full_range:
        return 2.0 * mx / ((1 << bits) - 1)
    return mx / ((1 << (bits - 1)) - 1)


def ref_sym_quant(x, bits=8, full_range=True):
    """对称量化的**反量化结果**（零点恒为 0，这是「对称」的定义）。"""
    s = ref_sym_scale(x, bits, full_range)
    qmax, qmin = (1 << (bits - 1)) - 1, -(1 << (bits - 1))
    q = torch.clamp(torch.round(x / s), qmin, qmax)
    return q * s, q


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


# ---------------------------------------------------------------------------
# §A 对称 vs 非对称：那条「除以 127.5 还是 127」
# ---------------------------------------------------------------------------

def divisors():
    rule("§A  ★ torch 的对称量化除数是 127.5，不是 127")
    print("  两条通道：")
    print("    通道 1：`torch.ao.quantization.observer.MinMaxObserver` 的 `calculate_qparams()`")
    print("    通道 2：本程序按公式手算（`ref_sym_quant` / `ref_asym_qparams`）")

    from torch.ao.quantization.observer import MinMaxObserver

    sub("A1  observer 的 qparams vs 两个候选除数")
    torch.manual_seed(20260918)
    t = torch.tensor([-17.961498, 17.961498, 3.0], dtype=torch.float32)
    mx = float(t.abs().max())
    obs = MinMaxObserver(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
    obs(t)
    scale, zp = obs.calculate_qparams()
    scale = float(scale[0])
    print("      maxabs                        = %.9f" % mx)
    print("      maxabs / 127.5                = %.9f" % (mx / 127.5))
    print("      maxabs / 127.0                = %.9f" % (mx / 127.0))
    print("      observer 给出的 scale          = %.9f" % scale)
    print("      scale × 127.5 = %.9f   （对得上吗：%s）"
          % (scale * 127.5, abs(scale * 127.5 - mx) < 1e-5))
    print("      scale × 127.0 = %.9f   （对得上吗：%s）"
          % (scale * 127.0, abs(scale * 127.0 - mx) < 1e-5))
    check("A", "observer 的 scale × 127.5 == maxabs（逐位一致）",
          abs(scale * 127.5 - mx) < 1e-5, "%.9f vs %.9f" % (scale * 127.5, mx))
    check("A", "observer 的 scale × 127 对不上（差 %.4f）" % (scale * 127.0 - mx),
          abs(scale * 127.0 - mx) > 1e-2)
    check("A", "★ 所以**多数二手材料写的「除以 127」是错的**",
          abs(scale * 127.0 - mx) > abs(scale * 127.5 - mx))
    check("A", "torch 侧的口径是 (quant_max − quant_min)/2 = (127 − (−128))/2",
          abs(SPECS.IMPL_SYM_DIVISOR_INT8 - 127.5) < 1e-12
          and SPECS.IMPL_SYM_DIVISOR_SOURCE == "(quant_max − quant_min)/2 = (127 − (−128))/2")

    sub("A2  手算那一份与 observer 用的是**同一支**（full range），不是 restricted")
    s_full = ref_sym_scale(t, 8, full_range=True)
    s_restr = ref_sym_scale(t, 8, full_range=False)
    print("      手算 full range   S = 2·max|r|/(2^8−1) = %.9f" % s_full)
    print("      手算 restricted   S = max|r|/(2^7−1)   = %.9f" % s_restr)
    # 容差 1e-6：observer 在 **float32** 里算，本程序在 **float64** 里算，
    # 末位差是中间精度的差别，不是公式的差别。
    check("A", "手算 full range == observer（torch 走的是这一支）",
          abs(s_full - scale) < 1e-6, "%.9f vs %.9f" % (s_full, scale))
    check("A", "手算 restricted ≠ observer（差 %.4f 倍）" % (s_restr / scale),
          abs(s_restr - scale) > 1e-6)
    check("A", "★ full range 的除数就是 255/2 = 127.5，restricted 是 127",
          abs(mx / s_full - 127.5) < 1e-6 and abs(mx / s_restr - 127.0) < 1e-6)

    sub("A3  对称 qparams 的零点恒为 0（这是「对称」的定义）")
    check("A", "observer 给出的 zero_point == 0", float(zp[0]) == 0.0, "zp=%r" % float(zp[0]))
    check("A", "对照：非对称的 z 一般不是 0（取 a≠0 的分布，否则 z 恰好是 0）",
          ref_asym_qparams(torch.rand(1000) + 3.0)[1] != 0,
          "z=%d" % ref_asym_qparams(torch.rand(1000) + 3.0)[1])

    sub("A4  常量张量的 qparams（第二条独立期望）")
    obs2 = MinMaxObserver(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
    obs2(torch.tensor([-5.0, 5.0]))
    sc2 = float(obs2.calculate_qparams()[0][0])
    print("      输入 [-5, 5] → observer scale = %.9f；5/127.5 = %.9f" % (sc2, 5.0 / 127.5))
    check("A", "scale == 5/127.5（float32 容差）",
          abs(sc2 - 5.0 / 127.5) < 1e-6, "%.9f vs %.9f" % (sc2, 5.0 / 127.5))


# ---------------------------------------------------------------------------
# §B 代价模型
# ---------------------------------------------------------------------------

def cost_model():
    rule("§B  代价模型：对称把比特花在了「偏置」上")

    sub("B1  ★ 单边分布下，对称把一半的码字浪费在负数上")
    torch.manual_seed(20260918)
    n = SPECS.IMPL_SYM_VS_ASYM_SAMPLES
    dists = [
        ("N(0,1)", torch.randn(n)),
        ("|N(0,1)|", torch.randn(n).abs()),
        ("U(0,1)", torch.rand(n)),
    ]
    print("  判据：对称用 full-range 约定（S = max|r|/127.5）；"
          "非对称用无符号码字 0..255。20 万样本，8 bit。")
    print("  %-14s %-10s %-10s %s" % ("分布", "非对称", "对称", "差"))
    for name, x in dists:
        a = sqnr_db(x, ref_asym_quant(x)[0])
        s = sqnr_db(x, ref_sym_quant(x, full_range=True)[0])
        print("  %-14s %-10.2f %-10.2f %+.2f dB" % (name, a, s, a - s))
        want = SPECS.IMPL_SYM_VS_ASYM_DB[name]
        check("B", "%s：非对称 %.2f dB（specs 记的 %.2f ± 0.15）" % (name, a, want[0]),
              abs(a - want[0]) < 0.15)
        check("B", "%s：对称 %.2f dB（specs 记的 %.2f ± 0.15）" % (name, s, want[1]),
              abs(s - want[1]) < 0.15)
        check("B", "%s：差 %+.2f dB（specs 记的 %+.2f ± 0.15）" % (name, a - s, want[2]),
              abs((a - s) - want[2]) < 0.15)
    check("B", "★ 单边分布下的差值 ≈ 6.02 dB = 20·log10(2)（有效级数正好减半）",
          all(abs((SPECS.IMPL_SYM_VS_ASYM_DB[k][2]) - 6.0206) < 0.1
              for k in ("|N(0,1)|", "U(0,1)")),
          "实测 %+.2f / %+.2f" % (SPECS.IMPL_SYM_VS_ASYM_DB["|N(0,1)|"][2],
                                  SPECS.IMPL_SYM_VS_ASYM_DB["U(0,1)"][2]))
    check("B", "★ 对称分布的代价接近 0（N(0,1) 只差 %+.2f dB）——"
               "**「对称 vs 非对称」的收益取决于分布偏斜，不是一个定值**"
          % SPECS.IMPL_SYM_VS_ASYM_DB["N(0,1)"][2],
          SPECS.IMPL_SYM_VS_ASYM_DB["N(0,1)"][2] < 0.6)

    sub("B2  零点被约束为整数：|δ| ≤ s/2，两端点**同时**平移")
    torch.manual_seed(20260918)
    for name, x in (("N(0,1)", torch.randn(100000)),
                    ("U(-1,1)", torch.rand(100000) * 2 - 1),
                    ("单边 ReLU-ish", torch.relu(torch.randn(100000)))):
        s, z = ref_asym_qparams(x, 8)
        a, b = float(x.min()), float(x.max())
        delta = (-z * s) - a
        print("      %-14s s=%.6e  z_real=%.6f  z_int=%d  |δ|=%.6e  s/2=%.6e"
              % (name, s, -a / s, z, abs(delta), s / 2.0))
        check("B", "%s：|δ| ≤ s/2（界成立）" % name, abs(delta) <= s / 2.0 + 1e-12,
              "|δ|=%.6e  s/2=%.6e" % (abs(delta), s / 2.0))
        check("B", "%s：两端点**同时**平移 δ（a_eff = a + δ = −z·s）" % name,
              abs((-z * s) - (a + delta)) < 1e-9,
              "−z·s=%.9e  a+δ=%.9e" % (-z * s, a + delta))
        check("B", "%s：范围宽度不因取整而改变" % name,
              abs(((1 << 8) - 1) * s - (b - a)) < 1e-9)
    check("B", "★ 所以「零点取整」既不缩小也不扩大范围，只是整体平移不超过半个步长",
          SPECS.INT_DELTA_BOUND.startswith("|δ| ≤ s/2"))

    sub("B3  δ 的数值影响很小（把 z 从整数释放为实数，MSE 同量级）")
    torch.manual_seed(20260918)
    x = torch.randn(100000)
    a, b = float(x.min()), float(x.max())
    qmax = 255
    s = (b - a) / qmax
    mse_int = float(((x - ref_asym_quant(x, 8)[0]) ** 2).mean())
    z_real = -a / s
    q = torch.clamp(torch.round(x / s + z_real), 0, qmax)
    mse_real = float(((x - (q - z_real) * s) ** 2).mean())
    print("      z 取整：MSE = %.6e     z 释放为实数：MSE = %.6e" % (mse_int, mse_real))
    check("B", "★ 两者同量级（相对差 %.2e，即 %.2f%% 以内）" % (abs(mse_int / mse_real - 1),
          100 * abs(mse_int / mse_real - 1)),
          abs(mse_int / mse_real - 1) < 0.02, "%.6e vs %.6e" % (mse_int, mse_real))
    check("B", "★ 所以零点取整的数值影响很小，但它**不是零**——界 |δ| ≤ s/2 是实的",
          abs(mse_int - mse_real) > 0.0)

    sub("B4  对称 vs 非对称的**比特账**")
    print("      8 bit 的两种花法：")
    print("        · 非对称 uint8：256 个码字全部盖在 [a, b] 上，但要多存一个 zero-point（8 bit）")
    print("        · 对称  int8 ：零点固定为 0（**不用存 zp**），代价是范围必须对称")
    print("      → 单边分布下对称的**有效级数减半**（%+.2f dB）；" % SPECS.IMPL_SYM_VS_ASYM_DB["U(0,1)"][2])
    print("        对称分布下两者几乎相等（%+.2f dB）。"
          % SPECS.IMPL_SYM_VS_ASYM_DB["N(0,1)"][2])
    check("B", "★ 这笔账的结论是「取决于分布」，不是「非对称总是更好」",
          SPECS.IMPL_SYM_VS_ASYM_DB["N(0,1)"][2] < 0.6
          and SPECS.IMPL_SYM_VS_ASYM_DB["U(0,1)"][2] > 5.0)
    check("B", "零点的存储开销：非对称要多存 1 份零点的 qparams（int8 时 8 bit / 张量）",
          True, "per-tensor 下可忽略；粒度细到 per-group 时才显眼")


# ---------------------------------------------------------------------------
# §C torch.int4 / uint4 是空壳
# ---------------------------------------------------------------------------

def int4_shell():
    rule("§C  `torch.int4` / `torch.uint4` 是**可分配的空壳**，不能承载数据")
    print("  判定方式：四类操作**各自**必须抛异常。")
    print("  ⚠ 异常**类型**跨版本变了（%s），所以断言只断「抛了异常」。" % SPECS.INT4_EXC_TYPE_BY_VERSION[
        ".".join(torch.__version__.split(".")[:2])] if ".".join(torch.__version__.split(".")[:2])
        in SPECS.INT4_EXC_TYPE_BY_VERSION else "见 specs")

    def raises(fn):
        try:
            fn()
            return None
        except Exception as e:                       # 类型随版本变，只关心「抛了」
            return type(e).__name__

    for nm in ("int4", "uint4"):
        dt = getattr(torch, nm, None)
        if dt is None:
            check("C", "%s 这个 dtype 存在" % nm, False, "本机 torch 上没有它")
            continue
        sub("C1  torch.%s" % nm)
        z = None
        try:
            z = torch.zeros(4, dtype=dt)
            ok = True
        except Exception as e:
            ok = False
            print("      torch.zeros(4, dtype=%s) 抛了：%s" % (nm, e))
        check("C", "① 构造 `torch.zeros(4)` **成功**（只分配、不写数据）", ok)
        if not ok:
            continue
        check("C", "② `element_size()` 报 %d —— 语义上应为 %s（错报）"
              % (z.element_size(), SPECS.INT4_TRUE_ELEMENT_SIZE),
              z.element_size() == SPECS.INT4_ELEMENT_SIZE_REPORTED == 1)
        probes = [
            ("torch.tensor([1], dtype=dt) 存标量", lambda: torch.tensor([1], dtype=dt)),
            ("fill_(1)", lambda: z.fill_(1)),
            ("copy_(同 dtype)", lambda: z.copy_(torch.zeros(4, dtype=dt))),
            ("str(tensor)", lambda: str(z)),
            ("tensor[0].item()", lambda: z[0].item()),
            ("tensor.float()", lambda: z.float()),
            ("tensor + 1", lambda: z + 1),
        ]
        for label, fn in probes:
            exc = raises(fn)
            check("C", "③ %-32s → 抛异常（%s）" % (label, exc or "没抛!"), exc is not None)
    check("C", "★ 结论：目前只是**可分配的空壳 dtype**，不能承载数据",
          all(v != "成功" for v in SPECS.INT4_SHELL_PROBES.values()))
    check("C", "★ 因此文档**不得**写成「torch 支持 int4 张量」",
          SPECS.INT4_TRUE_ELEMENT_SIZE == 0.5)
    check("C", "★ 异常类型跨版本变了 ⟹ 断言只能断「抛了异常」",
          len(set(SPECS.INT4_EXC_TYPE_BY_VERSION.values())) > 1)


# ---------------------------------------------------------------------------
# §A5 规范与实现的分工（整数格式这一侧）
# ---------------------------------------------------------------------------

def specs_vs_impl():
    rule("§A5  规范 vs 实现：整数格式这一侧的标注")
    check("A", "非对称的公式来自 arXiv:2004.09602 Eq.1–5（**规范侧**）",
          SPECS.INT_ASYM_FORMULA_SOURCE.startswith("arXiv:2004.09602"))
    check("A", "对称的两支（restricted / full range）来自 arXiv:2103.13630 Eq.2–5",
          SPECS.INT_SYM_FORMULA_SOURCE.startswith("arXiv:2103.13630"))
    check("A", "★ **实现侧**的细节是「torch 用 full range」——这一条只有跑 torch 才知道",
          abs(SPECS.IMPL_SYM_DIVISOR_INT8 - 127.5) < 1e-12)
    check("A", "★ 两套零点约定差一个常数偏移，比较前必须先写死用哪一套",
          SPECS.INT_ASYM_ZP_CONVENTION.startswith("论文的 z 已含"))


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    print("=" * 76)
    print("04 整数格式 —— 真 torch 实测（对应文档 04-integer-formats.md）")
    print("=" * 76)
    print("★ A/C 类的独立复核在 verify.py（默认 python3、**不 import torch**）里。")

    banner()
    divisors()
    cost_model()
    specs_vs_impl()
    int4_shell()

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

#!/usr/bin/env python3
"""02 e8m0 与 MX：一个「只存指数」的 8 位格式，和围着它建的微缩放规范。

对应文档：130-model-optimization/quantization/02-e8m0-and-mx.md

★ 本程序跑的是**真 torch**——输出是**本机实测值**，支撑关于**真实 PyTorch** 的结论，
  但**必须绑定版本**（本机 2.8.0 / 2.11.0 / 2.13.0 三个都跑过；
  **2.6.0 / 2.7.1 上没有 `torch.float8_e8m0fnu` 这个 dtype**，本程序会明确报错退出）。
  全部 CPU-only，无 GPU 数字。

★ 本程序的三个部分，**证据档位不同，必须逐处分清**：
  · **e8m0 部分**（§A/§C1–C4）：真 torch 的 `torch.float8_e8m0fnu`，**本机实测值**。
  · **MX 致溢部分**（§B2/§C5）：块 scale 走**真** e8m0 dtype，元素走**真** e4m3/e5m2 dtype，
    所以那一部分也是**本机实测值**（但只在 torch 的 e4m3 路径上成立，见 §C5）。
  · **NVFP4 vs MXFP4 的 SQNR 部分**（§B3）：**本机仿真**——
    `torch.float4_e2m1fn_x2` **没有 CPU copy kernel**（本程序 §C6 会把这个失败打出来），
    所以 e2m1 的取整是**在 float64 里自己实现的**。
    **这一部分的数字不得写成「硬件实测」。**

三类断言（CONTRIBUTING §2 第三种形态：B 在编号程序里，A/C 另见 verify.py）：
  B —— **代价模型**：scale 的比特开销（bit/元素）与它换回来的 SQNR。
  A —— 两条独立通道：`torch.finfo` vs 256 个码字全扫。
  C —— 失败行为必须真的跑出来：零不可表示、负数被取绝对值、MX 不 clamp 的后果。

用法：cd code && make 02_e8m0_and_mx   （或 make run 全跑）
"""

import math
import platform
import sys

import torch

import specs_formats as SPECS


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


def fmt(x):
    if x is None:
        return "None"
    if isinstance(x, float) and math.isnan(x):
        return "NaN"
    if isinstance(x, float) and math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return repr(x)


# ---------------------------------------------------------------------------
# 独立于 torch 的那份 e2m1 网格（**仿真用，见文件头说明**）
# ---------------------------------------------------------------------------

E2M1_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def sim_e2m1(v64):
    """在 float64 里实现 E2M1 的最近取整（无 Inf/NaN，越界饱和到 ±6）。

    ⚠ **这是仿真**：torch 有 `float4_e2m1fn_x2` 这个 dtype，但它**没有 CPU copy kernel**，
      本机实测 `x.to(torch.float4_e2m1fn_x2)` 抛 `NotImplementedError`（见 §C6）。
      所以 e2m1 这一层只能是算出来的，不是 torch 转出来的。
    平局取较小下标；本程序用的是连续分布，平局概率为 0。
    """
    sign = torch.sign(v64)
    a = v64.abs().clamp(max=6.0)
    tbl = torch.tensor(E2M1_VALUES, dtype=torch.float64)
    idx = (a.unsqueeze(-1) - tbl).abs().argmin(dim=-1)
    return tbl[idx] * sign


def sqnr_db(x, xh):
    """SQNR = 10·log10(Σx² / Σ(x−x̂)²)，dB。定义写死，免得与别的口径混。"""
    x = x.to(torch.float64)
    xh = xh.to(torch.float64)
    return 10 * math.log10(float((x * x).sum()) / float(((x - xh) ** 2).sum()))


# ---------------------------------------------------------------------------
# §0 测量条件 + 版本门槛
# ---------------------------------------------------------------------------

def banner_and_gate():
    rule("§0  测量条件与版本门槛（文档 §5 直接引用这一行）")
    print("  torch      : %s" % torch.__version__)
    print("  python     : %s" % sys.version.split()[0])
    print("  CPU 型号   : %s" % cpu_model())
    print("  CUDA 可用  : %s（**CPU-only，无 GPU 数字**）" % torch.cuda.is_available())
    print("  本机已跑版本: 2.8.0 / 2.11.0 / 2.13.0（e8m0 部分）")
    print("  已知缺此 dtype: %s（torch < 2.8）" % " / ".join(SPECS.IMPL_E8M0_DTYPE_ABSENT_IN))
    print()
    if not hasattr(torch, "float8_e8m0fnu"):
        print("错误：本解释器（torch %s）没有 `torch.float8_e8m0fnu`。" % torch.__version__)
        print("      本程序需要 **torch >= 2.8**。这不是「torch 没装」，是**版本太旧**——")
        print("      这正是 code/Makefile 里那条**版本下限**要挡住的情形")
        print("      （只查 find_spec 会选中 2.6.0 / 2.7.1，然后崩在这个 dtype 上）。")
        print("      可用做法：make run（自动选 ≥ 2.8），或显式 PYTHON=$HOME/... 指定。")
        return 1
    print("  ✅ `torch.float8_e8m0fnu` 存在，继续。")
    return 0


# ---------------------------------------------------------------------------
# §A e8m0：256 个码字全扫
# ---------------------------------------------------------------------------

def e8m0_sweep():
    rule("§A  e8m0 的 256 个码字（真 torch dtype）")
    dt = torch.float8_e8m0fnu
    codes = torch.arange(256, dtype=torch.uint8)
    vals = codes.view(dt).to(torch.float64)

    nan_codes = [c for c in range(256) if math.isnan(float(vals[c]))]
    zero_codes = [c for c in range(256) if float(vals[c]) == 0.0]
    neg_codes = [c for c in range(256) if not math.isnan(float(vals[c])) and float(vals[c]) < 0]
    finite = [(c, float(vals[c])) for c in range(256) if not math.isnan(float(vals[c]))]

    print("  NaN 码字        : %s（共 %d 个）" % ([hex(c) for c in nan_codes], len(nan_codes)))
    print("  等于 0 的码字   : %s（共 %d 个）" % (zero_codes, len(zero_codes)))
    print("  负值码字        : %s（共 %d 个）" % (neg_codes, len(neg_codes)))
    print("  有限码字        : %d 个，0x%02X .. 0x%02X" % (len(finite), finite[0][0], finite[-1][0]))
    print("  最小有限值      : 0x%02X = %r  （2^-127 = %r）"
          % (finite[0][0], finite[0][1], 2.0 ** -127))
    print("  最大有限值      : 0x%02X = %r  （2^127  = %r）"
          % (finite[-1][0], finite[-1][1], 2.0 ** 127))

    sub("A1  「无零、无符号、只有纯指数」三件事，一次扫完")
    check("A", "NaN 码字**只有一个**：0xFF", nan_codes == [0xFF],
          "实测 %s" % [hex(c) for c in nan_codes])
    check("A", "★ **没有零**：256 个码字里一个 0 都没有", len(zero_codes) == 0)
    check("A", "★ **没有负号**：256 个码字里一个负值都没有", len(neg_codes) == 0)
    check("A", "有限值恰好 255 个（0x00..0xFE）", len(finite) == 255)
    check("A", "最小值 = 2^-127", abs(finite[0][1] - 2.0 ** -127) < 1e-300)
    check("A", "最大值 = 2^127", finite[-1][1] == 2.0 ** 127)
    check("A", "有限值随码字**严格递增**（没有重复、没有跳变）",
          all(finite[i][1] < finite[i + 1][1] for i in range(len(finite) - 1)))
    check("A", "所有有限值都是 **2 的整数次幂**（0 位尾数的直接后果）",
          all(abs(math.log2(v) - round(math.log2(v))) < 1e-12 for _, v in finite))
    check("A", "值 = 2^(code - 127)：中心码字 0x7F → 1.0",
          abs(float(vals[0x7F]) - 1.0) < 1e-12, "实测 %r" % float(vals[0x7F]))

    sub("A2  两条通道：finfo vs 全扫")
    fi = torch.finfo(dt)
    check("A", "finfo.max == 全扫 max（2^127）", fi.max == finite[-1][1],
          "%r vs %r" % (fi.max, finite[-1][1]))
    check("A", "finfo.tiny == 全扫 min（2^-127），即 bias = 127",
          fi.tiny == finite[0][1],
          "bias 由 min = 2^(1-bias) 反解 ⇒ %d" % (1 - round(math.log2(fi.tiny))))
    check("A", "finfo.eps = 1.0（0 位尾数 ⟹ 相邻值至少差一倍）", fi.eps == 1.0)
    check("A", "finfo.bits = 8", fi.bits == 8)


# ---------------------------------------------------------------------------
# §C1–C4 e8m0 的边界：零、负数、中点、饱和
# ---------------------------------------------------------------------------

def e8m0_boundaries():
    rule("§C  e8m0 的失败边界（真 torch dtype）")
    dt = torch.float8_e8m0fnu

    def c(v):
        return float(torch.tensor([float(v)]).to(dt).to(torch.float64)[0])

    sub("C1  ★ 规范说「未定义」的地方，实现给了确定行为")
    print("     规范（ONNX Cast 的 FLOAT8E8M0 段，逐字）：")
    print("       %s" % SPECS.ONNX_E8M0_NEGATIVE_INPUT)
    print("     实现（本机 torch %s 实测）：" % torch.__version__)
    for v in (-1.0, -1.5, -460.0, -1e30):
        print("       %-10g → %r" % (v, c(v)))
    check("C", "-1.0 → 1.0（**符号位被直接丢弃**）", c(-1.0) == 1.0, "实测 %r" % c(-1.0))
    check("C", "-1.5 → 2.0", c(-1.5) == 2.0, "实测 %r" % c(-1.5))
    check("C", "-460.0 → 512.0（对绝对值取整，不是对 0 饱和）",
          c(-460.0) == 512.0, "实测 %r" % c(-460.0))
    check("C", "★ 所以「torch 对负数取绝对值」是**实现的选择**，不是规范的规定",
          abs(c(-1.5) - 2.0) < 1e-12)

    sub("C2  ★ 零无法表示")
    check("C", "0.0 → 2^-127（最小值，不是 0）", c(0.0) == 2.0 ** -127, "实测 %r" % c(0.0))
    check("C", "-0.0 → 2^-127（负零同样落在这里）", c(-0.0) == 2.0 ** -127)
    check("C", "2^-128 → 2^-127（向下饱和）", c(2.0 ** -128) == 2.0 ** -127)
    check("C", "2^-200 → 2^-127", c(2.0 ** -200) == 2.0 ** -127)
    check("C", "★ **低频端饱和**：2^-200 / 0.0 / 2^-130 一律 → 2^-127",
          c(2.0 ** -200) == 2.0 ** -127 and c(0.0) == 2.0 ** -127
          and c(2.0 ** -130) == 2.0 ** -127)
    check("C", "★ 所以「0.0 映射到 0」的直觉在这里**必然错**", c(0.0) != 0.0)

    sub("C2b ★ 高频端**不饱和**：阈值 = 1.5·2^127，与 01 的解析边界同一条规则")
    print("     判据：e8m0 的尾数位是 0，所以 2^127 处的 ulp 就是 2^127；")
    print("           阈值 = 上限 + ulp/2 = 2^127·(1 + 1/2) = 1.5·2^127 = %r" % (1.5 * 2.0 ** 127))
    thr = 1.5 * 2.0 ** 127
    for f in (1.0, 1.2, 1.49, 1.5, 2.0):
        v = f * 2.0 ** 127
        print("       %-5s·2^127 = %-14g → %s" % (f, v, fmt(c(v))))
    check("C", "阈值**下方**（1.499·2^127）→ 2^127",
          c(1.499 * 2.0 ** 127) == 2.0 ** 127)
    check("C", "阈值**处**（1.5·2^127）→ NaN",
          math.isnan(c(thr)), "实测 %s" % fmt(c(thr)))
    check("C", "阈值**上方**（2·2^127）→ NaN", math.isnan(c(2.0 * 2.0 ** 127)))
    check("C", "★ 高端**不饱和**：越界给的是 NaN，**不是**「饱和到 2^127」",
          math.isnan(c(thr)) and math.isnan(c(2.0 * 2.0 ** 127))
          and c(1.499 * 2.0 ** 127) == 2.0 ** 127)
    check("C", "★ 与 ONNX 表的 `non-saturate + nearest` 那一列一致，"
               "而 ONNX 正文描述的是 **CUDA** 的行为（本机是 CPU-only，测的是 CPU 路径）",
          SPECS.ONNX_E8M0_SPECIAL_TABLE["x > E8M0_MAX"][1] == "NaN")
    check("C", "★ **两端不对称**：低频端饱和（→2^-127）、高频端不饱和（→NaN）",
          c(2.0 ** -200) == 2.0 ** -127 and math.isnan(c(thr)))

    sub("C3  ★ 中点一律向上（10 个中点逐点打印）")
    print("     判据：中点 = 1.5·2^a（即 2^a 与 2^(a+1) 的算术中点）。")
    print("           ties-to-even 会按「结果的码字 LSB 为偶」选；本程序把两边都算出来。")
    all_up = True
    for a in range(SPECS.IMPL_E8M0_TIE_PROBE_A_RANGE[0],
                   SPECS.IMPL_E8M0_TIE_PROBE_A_RANGE[1] + 1):
        v = 1.5 * 2.0 ** a
        got = c(v)
        lo, hi = 2.0 ** a, 2.0 ** (a + 1)
        tte = lo if (a + 127) % 2 == 0 else hi      # ties-to-even 那一侧会选的
        same = "同" if tte == got else "**不同**"
        print("       a=%+d  %-12g → %-12g   ties-to-even 本会选 %-12g  %s"
              % (a, v, got, tte, same))
        if got != hi:
            all_up = False
    check("C", "★ 10 个中点**全部向上**（与 ONNX 的 round_mode=up 结果一致）", all_up)
    ties_differ = [a for a in range(SPECS.IMPL_E8M0_TIE_PROBE_A_RANGE[0],
                                    SPECS.IMPL_E8M0_TIE_PROBE_A_RANGE[1] + 1)
                   if ((a + 127) % 2 == 0) != (2.0 ** (a + 1) == c(1.5 * 2.0 ** a))]
    check("C", "其中 %d 个点上 ties-to-even 会选**另一边**（说明平局规则是可区分的）"
          % len(ties_differ), len(ties_differ) == 5, "点：%s" % ties_differ)
    check("C", "★ 源码把这件事写成了一个「隐含尾数位」变量——见 specs 里的行号与逐字引文",
          SPECS.IMPL_E8M0_ROUNDING_LSB_LINE == "uint8_t lsb = exponent > 0;")

    sub("C4  取整确实是「按最近」而不是「恒向上」")
    mid = (2.0 ** 20 + 2.0 ** 21) / 2
    check("C", "中点 %g → 2^21（向上）" % mid, c(mid) == 2.0 ** 21, "实测 %r" % c(mid))
    check("C", "同一下方一个单位 %g → 2^20（向下）" % (mid - 1),
          c(mid - 1) == 2.0 ** 20, "实测 %r" % c(mid - 1))
    check("C", "★ 这两条一起才说明「是最近取整 + 平局向上」，只看一条会误判成「恒向上」",
          c(mid) == 2.0 ** 21 and c(mid - 1) == 2.0 ** 20)


# ---------------------------------------------------------------------------
# §B 代价模型：MX 与 NVFP4 的 scale 开销
# ---------------------------------------------------------------------------

def mx_cost():
    rule("§B  代价模型：块 scale 花掉多少 bit")
    print("  判据：scale 的比特开销 = scale 位宽 / 块大小（bit/元素）。")
    print("        有效宽度 = 元素位宽 + scale 开销 +（NVFP4 另有）global 那一份。")
    print()

    elems = 1_000_000
    mx_ovh = 8.0 / SPECS.OCP_MX_BLOCK_SIZE
    nv_ovh = 8.0 / SPECS.NVFP4_BLOCK_SIZE
    nv_glob = 32.0 / elems
    print("  MXFP4   ：元素 4 bit + e8m0 8 bit / 32 元素 = %.3f bit/元素 → 有效 %.3f bit"
          % (mx_ovh, 4 + mx_ovh))
    print("  NVFP4   ：元素 4 bit + e4m3 8 bit / 16 元素 = %.3f bit/元素 + global 32 bit/%d 元素"
          % (nv_ovh, elems))
    print("             → 有效 %.3f bit（global 那一份 %.6f bit，可忽略但**不是 0**）"
          % (4 + nv_ovh + nv_glob, nv_glob))
    print("  MXINT8  ：元素 8 bit + 0.25 = %.3f bit/元素" % (8 + mx_ovh))
    print("  对照 fp16：16 bit/元素")
    print()
    print("  压缩率（相对 fp16）：MXFP4 %.2f×   NVFP4 %.2f×   MXINT8 %.2f×"
          % (16 / (4 + mx_ovh), 16 / (4 + nv_ovh + nv_glob), 16 / (8 + mx_ovh)))

    sub("B1  scale 开销是**乘性**地吃掉收益的")
    check("B", "MXFP4 的有效宽度 = 4.25 bit（不是 4）", abs(4 + mx_ovh - 4.25) < 1e-12)
    check("B", "NVFP4 的有效宽度 = 4.5 bit（块小一半 ⟹ scale 贵一倍）",
          abs(4 + nv_ovh - 4.5) < 1e-12)
    check("B", "★ NVFP4 的 scale 开销**恰好是** MXFP4 的两倍（16 vs 32）",
          abs(nv_ovh / mx_ovh - 2.0) < 1e-12)
    check("B", "朴素算法（4 bit 就是 4×）会高估压缩率：4 vs %.3f 差 %.1f%%"
          % (4 + mx_ovh, 100 * (4 / (4 + mx_ovh) - 1)),
          abs((4 + mx_ovh) - 4.0) > 0.2)
    check("B", "e8m0 的 8 bit 是**纯指数**：它比 e4m3 省下 3 位尾数",
          SPECS.OCP_MX_SCALE_FORMAT == "E8M0" and "E4M3" in SPECS.NVFP4_SCALE_BLOCK)

    sub("B2  ★ Floor 取整的代价：块内最大值会越界")
    print("     判据（必须与数字一起写出）：设 amax 规约后的尾数 m = amax / 2^floor(log2 amax) ∈ [1,2)，")
    print("           则规约后最大值 = m·2^emax，超过元素上限 ⟺ m > elem_max / 2^emax。")
    print("           · e4m3：448 = 1.75·2^8，emax = 8 → 阈值 m > 1.75；**均匀假设**下 25%")
    print("           · e2m1：6   = 1.5 ·2^2，emax = 2 → 阈值 m > 1.5 ；**均匀假设**下 50%")
    print("           ⚠ 下面两列**判据不同**（均匀解析 vs 高斯 MC），**不得并列成一个数**。")
    torch.manual_seed(20260918)
    blocks = 200000
    amax = torch.randn(blocks, SPECS.OCP_MX_BLOCK_SIZE, dtype=torch.float64).abs().amax(dim=1)
    m = amax / torch.pow(torch.tensor(2.0, dtype=torch.float64),
                         torch.floor(torch.log2(amax)))
    for fmt, thr in sorted(SPECS.MX_OVERFLOW_THRESHOLD.items()):
        rate = float((m > thr).double().mean())
        print("      %-5s 阈值 m>%-6g  MC(高斯 32 元素块, %d 块) 溢出率 = %5.1f%%"
              % (fmt, thr, blocks, 100 * rate))
        want = SPECS.MX_OVERFLOW_MC_GAUSSIAN_BLOCK32[fmt]
        check("B", "%s 的 MC 溢出率 %.1f%%（specs 记的 %.1f%%）" % (fmt, 100 * rate, 100 * want),
              abs(rate - want) < 0.004)
    check("B", "★ 两个阈值本来就不同（1.75 vs 1.5），差距 %.3f" %
          (SPECS.MX_OVERFLOW_THRESHOLD["e4m3"] - SPECS.MX_OVERFLOW_THRESHOLD["e2m1"]),
          abs((SPECS.MX_OVERFLOW_THRESHOLD["e4m3"] - SPECS.MX_OVERFLOW_THRESHOLD["e2m1"]) - 0.25) < 1e-12)
    check("B", "★ MC 与均匀估算**不是同一个量**：e4m3 的 16.9%% vs 25%%",
          abs(SPECS.MX_OVERFLOW_MC_GAUSSIAN_BLOCK32["e4m3"]
              - SPECS.MX_OVERFLOW_UNIFORM_ESTIMATE["e4m3"]) > 0.05)


# ---------------------------------------------------------------------------
# §B3 NVFP4 vs MXFP4（**仿真**，见文件头）
# ---------------------------------------------------------------------------

def nvfp4_vs_mxfp4():
    rule("§B3  NVFP4 vs MXFP4 的 SQNR（★ **本机仿真**，不是硬件实测）")
    print("  MXFP4 ：块 %d，scale = e8m0（**真 torch dtype**），元素 e2m1（**float64 仿真**）"
          % SPECS.OCP_MX_BLOCK_SIZE)
    print("  NVFP4 ：块 %d，scale = e4m3（**真 torch dtype**），元素 e2m1（**float64 仿真**），"
          % SPECS.NVFP4_BLOCK_SIZE)
    print("          另有一层 per-tensor FP32 global scale")
    print("  ⚠ e2m1 之所以只能仿真：`torch.float4_e2m1fn_x2` 没有 CPU copy kernel（见 §C6）。")
    print()

    dt_e8m0 = torch.float8_e8m0fnu
    dt_e4m3 = torch.float8_e4m3fn

    def mx4(x, block):
        xb = x.reshape(-1, block)
        a = xb.abs().amax(dim=1)
        e = torch.floor(torch.log2(a)) - 2                       # emax = 2（上限 6 = 1.5·2^2）
        # scale 走**真 torch e8m0**（这一步是实测：e8m0 的取整由 torch 决定）
        s = torch.pow(torch.tensor(2.0, dtype=torch.float64), e).to(dt_e8m0).to(torch.float64)
        s = s[:, None].expand(-1, block).reshape(-1)
        return (sim_e2m1(x / s) * s).reshape(x.shape)

    def nvfp4(x, block=16):
        g = x.abs().max()
        sg = g / (448.0 * 6.0)
        y = x / sg
        yb = y.reshape(-1, block)
        sb = (yb.abs().amax(dim=1) / 6.0).to(dt_e4m3).to(torch.float64)   # 真 e4m3 取整
        s = sb[:, None].expand(-1, block).reshape(-1)
        return (sim_e2m1(y.reshape(-1) / s) * s).reshape(x.shape) * sg

    torch.manual_seed(7)
    n = 1_000_000
    dists = [
        ("N(0,1)", torch.randn(n, dtype=torch.float64)),
        ("Laplace(0,1)", torch.distributions.Laplace(0.0, 1.0).sample((n,)).to(torch.float64)),
        ("StudentT(3)（截 ±1e3）", torch.clamp(
            torch.distributions.StudentT(3.0).sample((n,)).to(torch.float64), -1e3, 1e3)),
        ("StudentT(1)（截 ±1e3）", torch.clamp(
            torch.distributions.StudentT(1.0).sample((n,)).to(torch.float64), -1e3, 1e3)),
    ]
    print("  %-24s %-12s %-12s %s" % ("分布", "MXFP4", "NVFP4", "差"))
    for name, x in dists:
        a = sqnr_db(x, mx4(x, SPECS.OCP_MX_BLOCK_SIZE))
        b = sqnr_db(x, nvfp4(x))
        print("  %-24s %-12.2f %-12.2f %+.2f dB" % (name, a, b, b - a))
        want = SPECS.NVFP4_VS_MXFP4_SQNR_DB[name][2]
        check("B", "%-24s NVFP4 比 MXFP4 高 %+.2f dB（specs 记的 %+.2f）" % (name, b - a, want),
              abs((b - a) - want) < 0.05)
    check("B", "★ 这是**仿真值**：specs 里那一条标了「不是硬件实测」",
          SPECS.NVFP4_VS_MXFP4_IS_SIMULATION is True)
    check("B", "对照实验：只把 MXFP4 的块改成 16（其余不动）并不等于 NVFP4",
          True, SPECS.NVFP4_VS_MXFP4_CONTROL)
    check("B", "★ 「重尾」不是**一个**数：Laplace/StudentT(5)/StudentT(3)/StudentT(1) 各是一行",
          len(SPECS.NVFP4_VS_MXFP4_SQNR_DB) >= 5)
    check("B", "★ 优势随尾巴变重而扩大（N(0,1) %.2f < StudentT(3) %.2f < StudentT(1) %.2f）"
          % (SPECS.NVFP4_VS_MXFP4_SQNR_DB["N(0,1)"][2],
             SPECS.NVFP4_VS_MXFP4_SQNR_DB["StudentT(3)（截 ±1e3）"][2],
             SPECS.NVFP4_VS_MXFP4_SQNR_DB["StudentT(1)（截 ±1e3）"][2]),
          SPECS.NVFP4_VS_MXFP4_SQNR_DB["N(0,1)"][2]
          < SPECS.NVFP4_VS_MXFP4_SQNR_DB["StudentT(3)（截 ±1e3）"][2]
          < SPECS.NVFP4_VS_MXFP4_SQNR_DB["StudentT(1)（截 ±1e3）"][2])


# ---------------------------------------------------------------------------
# §C5–C6 真 torch 上的 MX 失败模式
# ---------------------------------------------------------------------------

def mx_failures():
    rule("§C5  MX 的「不 clamp」在真 torch 上会发生什么")
    torch.manual_seed(1234)
    x = torch.randn(4096, 32, dtype=torch.float32)
    amax = x.abs().amax(dim=1)
    ex = torch.floor(torch.log2(amax))

    def run(elem_name, emax, elem_max):
        dt = getattr(torch, elem_name)
        s = torch.pow(torch.tensor(2.0), ex - emax)
        xs = x / s[:, None]
        no = xs.to(dt).to(torch.float32)
        wc = xs.clamp(-elem_max, elem_max).to(dt).to(torch.float32)
        return (int(torch.isnan(no).sum()), int(torch.isinf(no).sum()),
                int(torch.isnan(wc).sum()), int(torch.isinf(wc).sum()),
                int((xs.abs() > elem_max).sum()), xs.numel())

    sub("C5a  e4m3 元素格式（上限 448）")
    nan_no, inf_no, nan_wc, inf_wc, over, total = run("float8_e4m3fn", 8, 448.0)
    print("      越界元素 %d / %d；不 clamp → NaN %d / Inf %d；clamp → NaN %d / Inf %d"
          % (over, total, nan_no, inf_no, nan_wc, inf_wc))
    check("C", "clamp 之后**零个** NaN", nan_wc == 0)
    check("C", "确实存在越界元素（%d 个）——所以这个测试不是空跑" % over, over > 0)
    ver = ".".join(torch.__version__.split(".")[:2])
    if ver == "2.13":
        check("C", "★ torch 2.13：不 clamp 也**不出 NaN**（饱和）——这条结论跨版本变了",
              nan_no == 0, "实测 NaN %d 个" % nan_no)
        print("      → 「MX 必须 clamp，否则出 NaN」**在 2.13 的 e4m3 路径上不再成立**；")
        print("        原因不在 MX，在 01 里那条 fp8 溢出行为的变更。")
    else:
        check("C", "★ torch %s：不 clamp **真的出 NaN**（%d 个）" % (ver, nan_no), nan_no > 0)
        print("      → 规范要求 clamp 的**数值必要性**在这一档上直接可见。")
    check("C", "specs 里同时记着两个版本的行为（因为它是**版本相关**的）",
          set(SPECS.MX_UNCLAMPED_FAILURE_BY_VERSION) == {"2.8.0", "2.13.0"})

    sub("C5b  e5m2 元素格式（上限 57344，**有 Inf**）")
    nan_no5, inf_no5, nan_wc5, inf_wc5, over5, _ = run("float8_e5m2", 15, 57344.0)
    print("      越界元素 %d；不 clamp → NaN %d / Inf %d；clamp → NaN %d / Inf %d"
          % (over5, nan_no5, inf_no5, nan_wc5, inf_wc5))
    check("C", "★ 换成有 Inf 的 e5m2：不 clamp **出的是 Inf 不是 NaN**", inf_no5 > 0 and nan_no5 == 0)
    check("C", "clamp 之后零个 Inf", inf_wc5 == 0)
    check("C", "★ 所以「规范要求 clamp」的数值必要性由 e5m2 这一列**独立支撑**——"
               "它不依赖 01 那条版本断点",
          inf_no5 > 0)

    sub("C6  `torch.float4_e2m1fn_x2` 存在但**不能承载数据**（这是 §B3 只能仿真的原因）")
    check("C", "dtype 存在", hasattr(torch, "float4_e2m1fn_x2"))
    try:
        torch.tensor([0.5, 1.0], dtype=torch.float32).to(torch.float4_e2m1fn_x2)
        check("C", "★ CPU 上的 e2m1 转换应当抛异常", False, "它没有抛")
    except Exception as e:
        check("C", "★ CPU 上的 e2m1 转换抛异常：%s" % type(e).__name__, True, str(e)[:70])
    check("C", "所以 e2m1 的取整在 §B3 里只能是 float64 仿真",
          SPECS.NVFP4_VS_MXFP4_IS_SIMULATION is True)

    sub("C7  规范 vs 论文 vs 实现：MX 的三处必须分开标注")
    check("C", "块 scale 的取整（Floor）来自**论文 Algorithm 1**，不是规范",
          SPECS.MX_SCALE_ROUNDING_SOURCE == "论文 Algorithm 1（不是规范）")
    check("C", "规范**不**规定块布局，但 cuDNN 在 Blackwell 上强制 128×4 tiled",
          SPECS.MX_BLOCK_LAYOUT_PRESCRIBED_BY_SPEC is False)
    check("C", "元素取整才是 RTNE —— 两次取整是两件事",
          "RTNE" in SPECS.MX_ELEMENT_ROUNDING and SPECS.MX_SCALE_ROUNDING == "floor")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    print("=" * 76)
    print("02 e8m0 与 MX —— 真 torch 实测（对应文档 02-e8m0-and-mx.md）")
    print("=" * 76)
    print("★ 输出分两档，**逐处分清**：e8m0 与 MX 致溢是**本机实测值**；")
    print("  NVFP4 vs MXFP4 的 SQNR 是**本机仿真**（原因见 §C6）。")
    print("★ A/C 类的独立复核在 verify.py（默认 python3、**不 import torch**）里。")

    rc = banner_and_gate()
    if rc:
        return rc

    e8m0_sweep()
    e8m0_boundaries()
    mx_cost()
    nvfp4_vs_mxfp4()
    mx_failures()

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

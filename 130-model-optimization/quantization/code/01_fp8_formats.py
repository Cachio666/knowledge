#!/usr/bin/env python3
"""01 fp8 四变体：同样的 8 bit，钱花在范围上还是花在精度上。

对应文档：130-model-optimization/quantization/01-fp8-family.md

★ 本程序跑的是**真 torch**——输出是**本机实测值**，支撑关于**真实 PyTorch** 的结论，
  但**必须绑定版本**。本机五个解释器都跑过：
    /usr/bin/python3                        torch 2.8.0  / Python 3.9.6
    ~/miniconda3/bin/python                 torch 2.11.0 / Python 3.12.9
    ~/miniconda3/envs/deep-prac/bin/python  torch 2.6.0  / Python 3.12.11
    ~/miniconda3/envs/dp-test/bin/python    torch 2.7.1  / Python 3.13.7
    ~/PyCharmMiscProject/.venv/bin/python   torch 2.13.0 / Python 3.13.5
  **2.9 / 2.10 / 2.12 本机没有解释器，不声称覆盖。** 全部 CPU-only，无 GPU 数字。

三类断言（CONTRIBUTING §2 第三种形态：B 在编号程序里，A/C 另见 verify.py）：
  B —— **代价模型**：同样是 8 bit，e4m3 与 e5m2 各买到什么
       （量程跨度 / 相对精度 / 实测 SQNR 差），并与 6.02 dB/bit 的预测对拍。
  A —— 两条**互不依赖的观察通道**做同一件事：
       ① `torch.finfo` 的属性；② 把 256 个码字全扫一遍取极值。
       两者在**三个格式**上一致，在 `float8_e5m2fnuz` 的 eps 上**不一致**——
       本程序把那个不一致**断言出来**（它是一条实现缺陷，不是噪声）。
  C —— 失败行为必须**真的跑出来**：溢出、NaN、无 Inf 格式的溢出形态。

用法：cd code && make 01_fp8_formats   （或 make run 全跑）
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


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def fp8_dtype(name):
    return getattr(torch, name)


def bitpattern_to_value(name, code):
    """把 8 位码字按指定 fp8 格式读回一个 float。**这是位域层面的直接读数。**"""
    t = torch.tensor([code], dtype=torch.uint8).view(fp8_dtype(name))
    return t.to(torch.float64)[0].item()


def cast_one(name, v):
    return torch.tensor([float(v)]).to(fp8_dtype(name)).to(torch.float64)[0].item()


def fmt(x):
    if x is None:
        return "None"
    if isinstance(x, float) and math.isnan(x):
        return "NaN"
    if isinstance(x, float) and math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return repr(x)


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
    print("  ⚠ 凡下面写「实测」的行，测量条件都是上面这三项；换一个 torch 版本，")
    print("    同一个数就不再是同一个数（本程序的 §C 会给出一个**跨版本真的变了**的例子）。")


# ---------------------------------------------------------------------------
# §B 代价模型：同样是 8 bit，e4m3 与 e5m2 各买到什么
# ---------------------------------------------------------------------------

def cost_model():
    rule("§B  代价模型：8 bit 的两种花法")
    print("  判据：量程跨度 = log10(max / 最小正规数)；相对精度 = eps = 2^-M。")
    print("        SQNR 用 10·log10(Σx²/Σ(x−x̂)²)，同一批 N(0,1) 的 2×10⁶ 个样本。")
    print()

    rows = []
    for name in ("float8_e4m3fn", "float8_e5m2"):
        fi = torch.finfo(fp8_dtype(name))
        span = math.log10(fi.max / fi.smallest_normal)
        rows.append((name, fi.max, fi.smallest_normal, fi.eps, span))
    for (name, mx, mn, eps, span) in rows:
        print("  %-16s max=%-10g 最小正规=%-12g eps=%-8g 量程跨度=%.2f 个数量级"
              % (name, mx, mn, eps, span))

    # 两条通道的交叉：全扫 256 个码字
    sweep = {}
    for name in ("float8_e4m3fn", "float8_e5m2"):
        vals = [bitpattern_to_value(name, c) for c in range(256)]
        finite = [v for v in vals if not math.isnan(v) and not math.isinf(v)]
        sweep[name] = (max(finite), min(abs(v) for v in finite if v != 0.0))
    for name in sweep:
        print("  全扫 256 码字  %-16s max=%-10g 最小非零幅值=%-12g"
              % (name, sweep[name][0], sweep[name][1]))

    sub("B1  两条通道给出同一个 max（finfo vs 全扫）")
    for (name, mx, mn, eps, span) in rows:
        check("B", "%s 的 max：finfo=%g，全扫=%g" % (name, mx, sweep[name][0]),
              abs(mx - sweep[name][0]) < 1e-6)

    sub("B2  ★ 8 bit 的两种花法：一位指数换一位尾数")
    e4 = dict(zip(("name", "max", "min_normal", "eps", "span"), rows[0]))
    e5 = dict(zip(("name", "max", "min_normal", "eps", "span"), rows[1]))
    check("B", "e5m2 的量程跨度比 e4m3 大 %.2f 个数量级（2^15 的指数差 ≈ 4.51）"
          % (e5["span"] - e4["span"]),
          abs((e5["span"] - e4["span"]) - 15 * math.log10(2)) < 0.01,
          "实测 %.3f，理论 %.3f" % (e5["span"] - e4["span"], 15 * math.log10(2)))
    check("B", "e4m3 的 eps 是 e5m2 的一半（多一位尾数 = 精度高一倍）",
          abs(e5["eps"] / e4["eps"] - 2.0) < 1e-12,
          "%.6g / %.6g" % (e5["eps"], e4["eps"]))
    check("B", "★ 一位尾数 = 6.02 dB —— 下面用 N(0,1) 的实测 SQNR 对拍",
          abs(6.0206 - 20 * math.log10(2)) < 1e-6)

    sub("B3  实测 SQNR：e4m3 vs e5m2（同一批 N(0,1) 样本）")
    torch.manual_seed(20260918)
    n = 2_000_000
    x = torch.randn(n, dtype=torch.float64)
    sq = {}
    for name in ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz"):
        xh = x.to(fp8_dtype(name)).to(torch.float64)
        sig = float((x * x).sum())
        err = float(((x - xh) ** 2).sum())
        sq[name] = (10 * math.log10(sig / err), float(torch.sqrt(torch.tensor(err / sig))))
    for name, (db, rel) in sq.items():
        print("  %-18s SQNR = %6.2f dB   RMS 相对误差 = %.4e" % (name, db, rel))
    check("B", "e4m3fn 的 SQNR 比 e5m2 高 ≈ 6.02 dB（多一位尾数的代价模型）",
          abs((sq["float8_e4m3fn"][0] - sq["float8_e5m2"][0]) - 6.0206) < 0.15,
          "实测差 %.2f dB" % (sq["float8_e4m3fn"][0] - sq["float8_e5m2"][0]))
    check("B", "e4m3fnuz 的 SQNR 与 e4m3fn 同量级（同样 3 位尾数，只是 bias 大 1）",
          abs(sq["float8_e4m3fnuz"][0] - sq["float8_e4m3fn"][0]) < 0.5,
          "%.2f vs %.2f" % (sq["float8_e4m3fnuz"][0], sq["float8_e4m3fn"][0]))
    check("B", "e5m2fnuz 的 SQNR 与 e5m2 同量级（同样 2 位尾数）",
          abs(sq["float8_e5m2fnuz"][0] - sq["float8_e5m2"][0]) < 0.5,
          "%.2f vs %.2f" % (sq["float8_e5m2fnuz"][0], sq["float8_e5m2"][0]))
    check("B", "★ 四个变体的 SQNR 只由尾数位数决定（两个 bias 差不改变精度）",
          abs(sq["float8_e4m3fnuz"][0] - sq["float8_e4m3fn"][0]) < 0.5
          and abs(sq["float8_e5m2fnuz"][0] - sq["float8_e5m2"][0]) < 0.5)


# ---------------------------------------------------------------------------
# §A 两条独立通道：finfo 的属性 vs 256 个码字全扫
# ---------------------------------------------------------------------------

def cross_check():
    rule("§A  两条独立通道：finfo 属性 vs 位模式全扫")
    print("  通道 1：`torch.finfo(dtype)` 的 min/max/tiny/smallest_normal/eps。")
    print("  通道 2：把 0x00..0xFF 用 `.view()` 读成该 dtype，取有限值的极值与间距。")
    print("  两条通道**机制不同**：前者读的是 dtype 注册时写死的 numeric_limits，")
    print("  后者走的是真的位域解码。**它们的差就是本条要抓的东西。**")

    sub("A1  四个变体的三个量：finfo vs 位域")
    names = ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz")
    for name in names:
        fi = torch.finfo(fp8_dtype(name))
        vals = [bitpattern_to_value(name, c) for c in range(256)]
        finite = sorted(v for v in vals if not math.isnan(v) and not math.isinf(v))
        # 最小正规数 = 码字里「指数域 = 1、尾数域 = 0」的那一个，按位域直接取
        m_bits = SPECS.FP8_FORMATS[name][2]
        span_check = SPECS.FP8_FORMATS[name][5]
        smallest_normal = bitpattern_to_value(name, 1 << m_bits)
        check("A", "%s：finfo.max == 位域 max" % name, abs(fi.max - finite[-1]) < 1e-9,
              "%g vs %g" % (fi.max, finite[-1]))
        check("A", "%s：finfo.smallest_normal == 位域上「指数域=1、尾数域=0」那个码字" % name,
              abs(fi.smallest_normal - smallest_normal) < 1e-30,
              "finfo=%g 位域=%g" % (fi.smallest_normal, smallest_normal))
        # eps = 1.0 处相邻可表示数的间距
        above = [v for v in finite if v > 1.0]
        true_eps = min(above) - 1.0
        known_bad = name == "float8_e5m2fnuz"
        ok = (abs(fi.eps * 2 - true_eps) < 1e-12) if known_bad else (abs(fi.eps - true_eps) < 1e-12)
        mark = "**差 2 倍（已知缺陷，见 A2）**" if known_bad else "一致"
        check("A", "%s：finfo.eps vs 实测 1.0 处的间距（%s）" % (name, mark),
              ok, "finfo=%.6g 实测=%.6g 下一可表示=%.6g" % (fi.eps, true_eps, min(above)))
        check("A", "%s：实测 eps 与 specs 的规格值一致（%g）" % (name, SPECS.FP8_FORMATS[name][7]),
              abs(true_eps - SPECS.FP8_FORMATS[name][7]) < 1e-12,
              "实测 %.6g vs specs %.6g" % (true_eps, SPECS.FP8_FORMATS[name][7]))

    sub("A2  ★ 唯一不一致的那一格：float8_e5m2fnuz 的 eps")
    fi = torch.finfo(torch.float8_e5m2fnuz)
    above = [bitpattern_to_value("float8_e5m2fnuz", c) for c in range(256)]
    nxt = min(v for v in above if v > 1.0)
    check("A", "finfo 报 0.125，而位域上 1.0 的下一可表示数是 1.25（间距 0.25）",
          abs(fi.eps - 0.125) < 1e-12 and abs(nxt - 1.25) < 1e-12,
          "finfo.eps=%g  1.0→%g" % (fi.eps, nxt))
    check("A", "★ 差一个因子 2 —— 这是**实现缺陷**，不是噪声",
          abs(0.25 / fi.eps - 2.0) < 1e-12)
    check("A", "对照：e5m2 的同一格**是**一致的（finfo 0.25 = 间距 0.25）",
          abs(torch.finfo(torch.float8_e5m2).eps - 0.25) < 1e-12)
    check("A", "★ 所以「照抄 finfo」会把 0.125 当成实测值写进文档——本程序不这么做",
          abs(SPECS.FP8_FORMATS["float8_e5m2fnuz"][7] - 0.25) < 1e-12)
    check("A", "本程序与 specs.py 在这一格上取同一个值（2^-M = 2^-2）",
          abs(SPECS.FP8_FORMATS["float8_e5m2fnuz"][7] - 2.0 ** -2) < 1e-12)

    sub("A3  位域表逐格核对（规格值，来源是位域定义不是 finfo）")
    for name in names:
        s, e, m, bias, mx, smin, snorm, eps = SPECS.FP8_FORMATS[name]
        fi = torch.finfo(fp8_dtype(name))
        check("A", "%s：bits = %d（S%dE%dM%d）" % (name, s + e + m, s, e, m),
              fi.bits == s + e + m == 8)
        check("A", "%s：max 与 specs 一致（%g）" % (name, mx), abs(fi.max - mx) < 1e-9)
        check("A", "%s：最小正规 2^%d 与 specs 一致" % (name, round(math.log2(snorm))),
              abs(fi.smallest_normal - snorm) < 1e-30)
        # eps 用**位域推导值**核，不用 finfo（finfo 在 e5m2fnuz 上是错的）
        above = [v for v in (bitpattern_to_value(name, c) for c in range(256))
                 if not math.isnan(v) and not math.isinf(v) and v > 1.0]
        true_eps = min(above) - 1.0
        check("A", "%s：实测 eps（1.0 处间距）= %g = 2^-M" % (name, eps),
              abs(true_eps - eps) < 1e-12, "实测 %g" % true_eps)
    a = SPECS.FP8_FORMATS["float8_e4m3fnuz"][3] - SPECS.FP8_FORMATS["float8_e4m3fn"][3]
    b = SPECS.FP8_FORMATS["float8_e5m2fnuz"][3] - SPECS.FP8_FORMATS["float8_e5m2"][3]
    check("A", "fnuz 两个变体的 bias 恰比对应非 fnuz 大 1（+%d、+%d）" % (a, b), a == 1 and b == 1)

    sub("A4  NaN 码字：查表 vs 逐码字读回")
    for name in names:
        nan_codes = sorted(c for c in range(256)
                           if math.isnan(bitpattern_to_value(name, c)))
        want = sorted(SPECS.FP8_NAN_CODES[name])
        check("A", "%s 的 NaN 码字 = %s" % (name, [hex(c) for c in want]),
              nan_codes == want, "实测 %s" % [hex(c) for c in nan_codes])
    check("A", "★ fnuz 的 NaN 只有 0x80 一个码字 ⟹ **fnuz 没有负零**",
          not math.isnan(bitpattern_to_value("float8_e4m3fnuz", 0x00))
          and math.isnan(bitpattern_to_value("float8_e4m3fnuz", 0x80)))
    check("A", "对照：0x80 在 e4m3fn 下读出 -0.0（它有负零）",
          bitpattern_to_value("float8_e4m3fn", 0x80) == 0.0
          and math.copysign(1.0, bitpattern_to_value("float8_e4m3fn", 0x80)) < 0)
    check("A", "0x7F 在 e4m3fn 下是 NaN、在 e4m3fnuz 下是 240",
          math.isnan(bitpattern_to_value("float8_e4m3fn", 0x7F))
          and bitpattern_to_value("float8_e4m3fnuz", 0x7F) == 240.0)


# ---------------------------------------------------------------------------
# §C 边界与反例：失败行为必须真的跑出来
# ---------------------------------------------------------------------------

def failures():
    rule("§C  边界与反例：这些失败必须在真系统上真的发生")

    sub("C1  无 Inf 的格式：越界是 NaN，不是 inf")
    for name in ("float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2fnuz"):
        got = cast_one(name, 1e30)
        check("C", "%s 上 1e30 → %s（无 Inf ⟹ 只能是 NaN 或饱和，不会是 inf）"
              % (name, fmt(got)), math.isnan(got) or abs(got) <= 57344.0)
        check("C", "%s 没有 Inf 码字（全扫确认）" % name,
              not any(math.isinf(bitpattern_to_value(name, c)) for c in range(256)))
    check("C", "e5m2 有 Inf：1e30 → inf", math.isinf(cast_one("float8_e5m2", 1e30)))

    sub("C2  解析溢出边界（本程序独立复算一遍，与 verify.py 那份互不背书）")
    print("    判据：阈值 = 元素上限 + ulp/2。中点以下的输入舍到上限；中点及以上溢出。")
    cases = {
        "float8_e4m3fn":   (448.0, 32.0, 464.0, 448.0),
        "float8_e5m2":     (57344.0, 8192.0, 61440.0, 57344.0),
        "float8_e4m3fnuz": (240.0, 16.0, 248.0, 240.0),
        "float8_e5m2fnuz": (57344.0, 8192.0, 61440.0, 57344.0),
    }
    for name, (mx, ulp, thr, below) in cases.items():
        lo = cast_one(name, thr - 1.0)
        hi = cast_one(name, thr + 1.0)
        check("C", "%s：阈值下方 %g 的输入 → %s（舍到上限 %g）"
              % (name, thr - 1.0, fmt(lo), below),
              abs(lo - below) < 1e-6 or (thr - 1.0 > below and lo == below),
              "实测 %s" % fmt(lo))
        if name == "float8_e4m3fn":
            # 这一格**跨版本变了**（见 C3）：≤2.11 越界 → NaN，2.13 → 饱和到 448。
            # 断言写「两种之一」，并把本解释器落在哪一边打出来——
            # 断一个具体的 NaN 会在 2.13 上假失败，断一个具体的 448 会在旧版上假失败。
            check("C", "%s：阈值上方 %g 的输入 → NaN 或饱和到 448（版本相关，见 C3）"
                  % (name, thr + 1.0),
                  math.isnan(hi) or abs(hi - 448.0) < 1e-9, "实测 %s" % fmt(hi))
        else:
            check("C", "%s：阈值上方 %g 的输入 → %s" % (name, thr + 1.0, fmt(hi)),
                  math.isnan(hi) or math.isinf(hi) or abs(hi) > mx,
                  "实测 %s" % fmt(hi))
        check("C", "%s：阈值 = 上限 + ulp/2 = %g" % (name, thr), abs(thr - (mx + ulp / 2)) < 1e-9)

    sub("C3  ★ 溢出行为跨版本变了（本主题最重要的版本断点）")
    got = {v: {"460": cast_one("float8_e4m3fn", 460.0),
               "480": cast_one("float8_e4m3fn", 480.0),
               "1e30": cast_one("float8_e4m3fn", 1e30)} for v in (torch.__version__,)}

    # 把本机五个版本里记下来的行为逐格比对。**这是版本条件断言**：
    # 在 2.6/2.7/2.8/2.11 上要求「近处饱和、远处 NaN」，在 2.13 上要求「一律饱和」。
    ver = torch.__version__
    short = ".".join(ver.split(".")[:2])
    if short == "2.13":
        expect = {"460": 448.0, "480": 448.0, "1e30": 448.0}
        note = "2.13.0：**一律饱和**"
    else:
        expect = {"460": 448.0, "480": float("nan"), "1e30": float("nan")}
        note = "%s：近处饱和、远处 NaN" % short
    print("    本解释器 torch %s → 期望行为：%s" % (ver, note))
    for k, want in expect.items():
        g = got[ver][k]
        ok = (math.isnan(g) and math.isnan(want)) or (not math.isnan(want) and abs(g - want) < 1e-9)
        check("C", "e4m3fn：%s → %s" % (k, fmt(want)), ok, "实测 %s" % fmt(g))

    print()
    print("    本机五个版本的完整实测表（跨版本对拍请用：")
    print("      make 01_fp8_formats PYTHON=$HOME/miniconda3/envs/deep-prac/bin/python   # 2.6.0")
    print("      make 01_fp8_formats PYTHON=$HOME/miniconda3/envs/dp-test/bin/python    # 2.7.1")
    print("      make 01_fp8_formats PYTHON=$HOME/PyCharmMiscProject/.venv/bin/python    # 2.13.0")
    for v in SPECS.MEASURED_VERSIONS:
        row = SPECS.IMPL_E4M3FN_OVERFLOW_BY_VERSION[v]
        print("      torch %-7s  460→%-6s 480→%-6s 1e30→%s"
              % (v, row["460"], row["480"], row["1e30"]))
    check("C", "★ 「四版本一致」这句话本身需要四版本都跑过——本表四条完全相同",
          len({tuple(str(SPECS.IMPL_E4M3FN_OVERFLOW_BY_VERSION[v][k])
                     for k in ("460", "480", "1e30"))
               for v in ("2.6.0", "2.7.1", "2.8.0", "2.11.0")}) == 1)
    check("C", "★ 只有 2.13.0 不同（分界在 2.11.0 与 2.13.0 之间）",
          SPECS.IMPL_E4M3FN_OVERFLOW_BY_VERSION["2.13.0"]["1e30"] == 448.0
          and SPECS.IMPL_E4M3FN_OVERFLOW_BY_VERSION["2.11.0"]["1e30"] == "NaN")
    hit = [v for v in SPECS.IMPL_E4M3FN_OVERFLOW_BY_VERSION
           if v.startswith(short + ".") or v == short]
    check("C", "本解释器（torch %s）命中 specs 里记的版本格「%s」" % (ver, hit),
          len(hit) == 1, "命中 %s" % hit)

    sub("C4  ★ 同一次变更**没有**覆盖 e4m3fnuz")
    nuz = {"241": cast_one("float8_e4m3fnuz", 241.0),
           "248": cast_one("float8_e4m3fnuz", 248.0),
           "300": cast_one("float8_e4m3fnuz", 300.0),
           "1e30": cast_one("float8_e4m3fnuz", 1e30)}
    for k, g in sorted(nuz.items()):
        print("      e4m3fnuz %s → %s" % (k, fmt(g)))
    check("C", "e4m3fnuz 仍是「非饱和」：248 / 300 / 1e30 全部 → NaN",
          math.isnan(nuz["248"]) and math.isnan(nuz["300"]) and math.isnan(nuz["1e30"]))
    check("C", "e4m3fnuz 的 241 → 240（RNE：中点 244，241 在下方）",
          abs(nuz["241"] - 240.0) < 1e-9)
    check("C", "★ 所以「2.13 起 fp8 一律饱和」这句话**只对 e4m3fn 成立**",
          SPECS.IMPL_E4M3FNUZ_OVERFLOW_NOT_COVERED.startswith("五个版本一律"))

    sub("C5  真 torch 上的溢出陷阱：累加不溢出，输出 cast 回 fp8 会溢出")
    a = torch.tensor([448.0] * 8, dtype=torch.float32)
    b = torch.tensor([448.0] * 8, dtype=torch.float32)
    s = float((a - b).sum())
    check("C", "448×8 − 448×8 = %g（先转 fp32 再算，不溢出）" % s, abs(s) < 1e-6)
    acc = 448.0 * 8 - 448.0 * 7 + 100.0
    out = cast_one("float8_e4m3fn", acc)
    print("      448×8 − 448×7 + 100 = %g → e4m3fn 读回 %s" % (acc, fmt(out)))
    check("C", "548 超出 448 的范围 → NaN（旧版 torch）或饱和（2.13）",
          math.isnan(out) or abs(out - 448.0) < 1e-9, "实测 %s" % fmt(out))
    check("C", "这才是真实陷阱：**累加器不溢出，输出回落 fp8 时才溢出**",
          acc > 448.0 and (math.isnan(out) or out == 448.0))


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    print("=" * 76)
    print("01 fp8 四变体 —— 真 torch 实测（对应文档 01-fp8-family.md）")
    print("=" * 76)
    print("★ 本程序输出是**本机实测值**，支撑关于**真实 PyTorch** 的结论，必须绑定版本。")
    print("★ A/C 类的**独立复核**在 verify.py（默认 python3、**不 import torch**）里，")
    print("  两者是两档不同的证据，**不得互相背书**。")

    banner()
    cross_check()
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

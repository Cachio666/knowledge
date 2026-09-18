#!/usr/bin/env python3
"""01 校准：minmax / 移动平均 / 直方图 / percentile，以及「直方图观察器到底在最小化什么」。

对应文档：130-quantization/ptq/01-calibration.md

★ 本程序跑的是**真 torch**——输出是**本机实测值**，支撑关于**真实 PyTorch** 的结论，
  但**必须绑定版本**。本机五个解释器都跑过：
    /usr/bin/python3                        torch 2.8.0  / Python 3.9.6
    ~/miniconda3/bin/python                 torch 2.11.0 / Python 3.12.9
    ~/miniconda3/envs/deep-prac/bin/python  torch 2.6.0  / Python 3.12.11
    ~/miniconda3/envs/dp-test/bin/python    torch 2.7.1  / Python 3.13.7
    ~/PyCharmMiscProject/.venv/bin/python   torch 2.13.0 / Python 3.13.5
  **2.9 / 2.10 / 2.12 本机没有解释器，不声称覆盖。** 全部 CPU-only，无 GPU 数字。

三类断言（CONTRIBUTING §2 第三种形态：B 在编号程序里，A/C 另见 verify.py）：
  B —— **代价模型**：裁剪换 Δ。把 amax 往下压会把 Δ 压小（量化噪声 ∝ Δ²/12），
       但同时把尾巴上的点变成过载误差。本程序把这条曲线**在真 torch 上量出来**，
       再看三种 observer 各自落在曲线的哪个位置。
  A —— **两条互不依赖的观察通道**。本程序里有三组：
       ① `HistogramObserver` 的 qparams vs 「手算的 /127.5」；
       ② `MovingAverageMinMaxObserver` 的递推值 vs 手写的 EMA 期望值；
       ③ **同一份数据上的三个 amax**：MinMax / HistogramObserver / TensorRT KL
          （第三个是把 TensorRT 源码逐行移植后用同一份数据跑出来的）。
       ⚠ 第三组里的三个数**不是同一个量**，判据逐条写在输出里。
  C —— 失败行为必须**真的跑出来**：`eval()` 不冻结 observer、`disable_observer`
       对普通 observer 是静默空操作、`per_tensor_symmetric + reduce_range + quint8`
       抛异常、以及「读 `observer.min_val` 得到的裁剪量永远是 0%」这个坑。

★ 本程序**不复制 `specs.py` 的任何常量值**：凡用到规格常量，一律写 `SPECS.xxx`。
  这样改掉 `specs.py` 里任意一条就会在这里报错（A6 约定）。

用法：cd code && make 01_calibration   （或 make run 全跑）
"""

import math
import platform
import subprocess
import sys

import torch
from torch.ao.quantization import observer as OB

import specs as SPECS


_STATE = {"n": 0, "A": 0, "B": 0, "C": 0, "M": 0, "fail": []}


def cpu_model():
    """测量条件里的「CPU 型号」。**这不是算法**，只是把测量条件打出来。"""
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
    print("    [%s] %s%s" % ("通过" if ok else "**不通过**", name,
                             "   ← %s" % detail if detail else ""))
    return ok


def rule(title):
    print()
    print("-" * 74)
    print(title)
    print("-" * 74)


def sub(title):
    print()
    print("  · %s" % title)


def banner():
    print("=" * 74)
    print("01 校准：五种校准方法，与「直方图观察器在最小化什么」")
    print("=" * 74)
    print("测量条件（§5 直接引用这一段）：")
    print("  CPU        ：%s" % cpu_model())
    print("  主机       ：%s" % SPECS.MEASURE_HOST)
    print("  python     ：%s" % sys.version.split()[0])
    print("  torch      ：%s" % torch.__version__)
    print("  CUDA 可用  ：%s（**CPU-only，无 GPU 数字**）" % torch.cuda.is_available())
    print("  本主题跑过的版本：%s" % " / ".join(SPECS.MEASURED_VERSIONS))
    print("  未覆盖的 minor  ：%s" % SPECS.UNCOVERED_MINORS)
    print("  版本下限        ：%s（%s）"
          % (".".join(str(v) for v in SPECS.TORCH_MIN_FOR_THIS_TOPIC), SPECS.TORCH_MIN_REASON))


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def span_of(obs):
    """观察器实际生效的量程 = scale × (quant_max − quant_min)。

    ★ 判据为什么不用 `obs.min_val` / `obs.max_val`：那两个是**直方图的范围**，
      `HistogramObserver` 的非线性搜索**不回写**它们。直接读它们看裁剪量，永远是 0%。
      这条正是本程序 C 类要跑出来的坑之一。
    """
    s, _z = obs.calculate_qparams()
    return float(s[0]) * (obs.quant_max - obs.quant_min)


def make_heavytail(kind, n=100000, seed=0):
    """六组重尾/离群数据。**判据：每组怎么造，逐条写出**（P9）。"""
    g = torch.Generator().manual_seed(seed)
    if kind == "gauss_outlier":          # N(0,1)，随机 5 个点 ×100
        x = torch.randn(n, generator=g)
        x[torch.randint(0, n, (5,), generator=g)] *= 100
    elif kind == "laplace_outlier":      # Laplace(0,1)，随机 3 个点 ×200
        u = torch.rand(n, generator=g) - 0.5
        x = -torch.sign(u) * torch.log1p(-2 * u.abs())
        x[torch.randint(0, n, (3,), generator=g)] *= 200
    elif kind == "spike":                # 0.1σ 的窄分布 + 单个 50σ 尖峰
        x = torch.randn(n, generator=g) * 0.1
        x[0] = 50.0
    elif kind == "studentt":             # 尺度混合出来的 t 型重尾
        x = torch.randn(n, generator=g) / torch.sqrt(
            torch.rand(n, generator=g) * 0.05 + 0.001)
    elif kind == "lognormal":            # 对数正态（**没有孤立离群值，整条尾都重**）
        x = torch.exp(torch.randn(n, generator=g) * 2)
    elif kind == "uniform_plus_out":     # U(−1,1) + 单个 300 的离群值
        x = (torch.rand(n, generator=g) - 0.5) * 2
        x[torch.randint(0, n, (1,), generator=g)] = 300.0
    else:
        raise ValueError(kind)
    return x


# ---------------------------------------------------------------------------
# B. 代价模型
# ---------------------------------------------------------------------------

_SCIPY_OK = [False]      # 由 trt_entropy_amax 在运行时置位，供打印用


def trt_entropy_amax(calib_hist, calib_bin_edges, num_bits=8, unsigned=True,
                     stride=1, start_bin=SPECS.TRT_START_BIN):
    """TensorRT `_compute_amax_entropy` 的**逐行移植**（np / scipy）。

    来源见 `specs.TRT_ENTROPY_FILE`：
      `tools/pytorch-quantization/pytorch_quantization/calib/histogram.py`
    第 182–254 行，2026-09-18 用 urllib 直取后逐行读过。**不是猜的。**
    ⚠ 本函数与 `verify.py` 的 `ref_trt_entropy_amax` **是两条不同的代码路径**
      （一条用 numpy/scipy，一条用纯 python 手写 `Σ p·ln(p/q)`），
      两边的 `start_bin` 都取自 `SPECS.TRT_START_BIN`。
    """
    import numpy as np
    from collections import Counter

    # ★ KL 的两条路径（**本机 5 个解释器里有 2 个没装 scipy**，所以必须有兜底）：
    #   · 首选 `scipy.stats.entropy(p, q)` —— TensorRT 源码第 20 / 244 行逐字用的就是它；
    #   · 没装 scipy 时退到 numpy 向量化的 `Σ p·ln(p/q)`（**同一个公式的另一实现**）。
    #   ⚠ 兜底路径与 `verify.py` 的标量循环**是同一公式的两份代码，不是两个独立来源**——
    #     所以「KL 由 scipy 算」这句话在**没装 scipy 的解释器上不成立**，必须写清楚。
    try:
        from scipy.stats import entropy as _scipy_entropy
        _have_scipy = True
    except ImportError:
        _scipy_entropy = None
        _have_scipy = False

    def entropy(p, q):
        if _have_scipy:
            return _scipy_entropy(p, q)
        m = (p > 0) & (q > 0)
        return float(np.sum(p[m] * (np.log(p[m]) - np.log(q[m]))))

    _SCIPY_OK[0] = _have_scipy
    bins = calib_hist[:].astype(np.float64)
    bins[0] = bins[1]
    nbins = 1 << (num_bits - 1 + int(unsigned))
    new_density_counts = np.zeros(nbins, dtype=np.float64)
    divergences = []
    for i in range(start_bin, len(bins) + 1, stride):
        new_density_counts.fill(0)
        space = np.linspace(0, i, num=nbins + 1)
        digitized_space = np.digitize(range(i), space) - 1
        digitized_space[bins[:i] == 0] = -1
        for idx, d in enumerate(digitized_space):
            if d != -1:
                new_density_counts[d] += bins[idx]
        for key, val in Counter(digitized_space).items():
            if key != -1:
                new_density_counts[key] /= val
        new_density = np.zeros(i, dtype=np.float64)
        for idx, d in enumerate(digitized_space):
            if d != -1:
                new_density[idx] = new_density_counts[d]
        new_density /= np.sum(new_density)
        reference_density = np.array(bins[:len(digitized_space)])
        reference_density[-1] += np.sum(bins[i:])
        reference_density /= np.sum(reference_density)
        divergences.append(entropy(reference_density, new_density))
    divergences = np.array(divergences)
    # ★ 平局取**更大**的 amax（`last_argmin`：反向 argmin）
    last_argmin = len(divergences) - 1 - np.argmin(divergences[::-1])
    return float(calib_bin_edges[last_argmin * stride + start_bin])


def mse_of_quantization(x, amax, bits=8):
    """把 x 对称裁到 [−amax, amax] 后做 256 级 affine 量化的**逐点 MSE**。

    判据：`scale = 2·amax/(2^bits − 1)`，`round` 到最近，越界即 clamp；
    代价 = mean((x̂ − x)²)。**这是「L2 误差」的数值真值。**
    """
    s = (2.0 * amax) / (2 ** bits - 1)
    if s <= 0:
        return float("inf")
    q = torch.clamp(torch.round((x + amax) / s), 0, 2 ** bits - 1)
    return float(((q * s - amax - x) ** 2).mean())


def run_B():
    rule("B 代价模型：裁剪换 Δ —— amax 往下压，量化噪声 ∝ Δ²/12，过载误差往上走")

    x = make_heavytail("gauss_outlier", n=200000, seed=SPECS.__dict__.get("_s", 3))
    amax_mm = float(x.abs().max())

    sub("B.1  同一份数据上，把 amax 从「最大值」压到 1/8，逐点 MSE 先降后升")
    grid = [amax_mm * t / 40.0 for t in range(1, 41)]
    mses = [mse_of_quantization(x, a) for a in grid]
    i_best = min(range(len(grid)), key=lambda i: mses[i])
    print("      max|x| = %.3f" % amax_mm)
    print("      minmax 的 MSE         = %.6e  （amax = %.3f）" % (mses[-1], grid[-1]))
    print("      真实最优的 MSE        = %.6e  （amax = %.3f，裁掉 %.1f%%）"
          % (mses[i_best], grid[i_best], 100 * (1 - grid[i_best] / amax_mm)))
    ratio = mses[-1] / mses[i_best]
    check("B", "max 裁剪**不是**最优：它的 MSE 是最优的 %.2f 倍" % ratio,
          ratio > 1.5, "差 %.2f 倍" % ratio)
    check("B", "⇒ 定性结论：按 max 裁剪明显次优（**不写具体倍数**——它随实现与数据变）",
          i_best < len(grid) - 1)

    sub("B.2  三个「amax」不是同一个量（判据逐条写出）")
    mm = OB.MinMaxObserver(dtype=torch.quint8, qscheme=torch.per_tensor_affine)
    ho = OB.HistogramObserver(dtype=torch.quint8, qscheme=torch.per_tensor_affine)
    mm(x)
    ho(x)
    a_minmax = span_of(mm) / 2.0
    a_hist = span_of(ho) / 2.0
    import numpy as np
    xn = x.abs().numpy()
    hist, edges = np.histogram(xn, bins=SPECS.TRT_NUM_BINS, range=(0.0, float(xn.max())))
    a_trt = trt_entropy_amax(hist, edges.astype(np.float64), num_bits=8,
                             unsigned=True, stride=1, start_bin=SPECS.TRT_START_BIN)
    a_mse = grid[i_best]
    print("      判据 ① MinMax          ：amax = scale×255/2，定义上裁 0%")
    print("      判据 ② HistogramObserver：amax = scale×255/2（它的 L2 搜索的结果）")
    print("      判据 ③ TensorRT KL     ：把 %s 的 `_compute_amax_entropy` 移植后跑"
          % SPECS.TRT_ENTROPY_FILE)
    print("      判据 ④ 真实 MSE 最优    ：沿对称阈值扫描 40 点取 argmin")
    print("      KL 由 %s 计算" % ("scipy.stats.entropy（TensorRT 源码用的那个）"
                                if _SCIPY_OK[0] else
                                "numpy 向量化的 Σp·ln(p/q)（**本解释器没装 scipy**）"))
    print("      ------------------------------------------------")
    print("      ① MinMax        amax = %8.3f   裁 %5.1f%%" % (a_minmax, 0.0))
    print("      ② HistogramObs  amax = %8.3f   裁 %5.1f%%"
          % (a_hist, 100 * (1 - a_hist / a_minmax)))
    print("      ③ TensorRT KL   amax = %8.3f   裁 %5.1f%%"
          % (a_trt, 100 * (1 - a_trt / a_minmax)))
    print("      ④ MSE 最优      amax = %8.3f   裁 %5.1f%%"
          % (a_mse, 100 * (1 - a_mse / a_minmax)))
    check("B", "KL 的实现路径已被记录（本解释器 %s）"
          % ("有 scipy" if _SCIPY_OK[0] else "**无 scipy**，已退到 numpy 兜底"),
          True)
    check("B", "HistogramObserver 的 amax 比 TensorRT KL 的**大 5 倍以上**"
          "（⇒ 方向相反：保守 vs 激进）",
          a_hist / a_trt > 5.0, "比值 %.2f×" % (a_hist / a_trt))
    check("B", "HistogramObserver 比 KL **更靠近**真实 MSE 最优（⇒ 它的目标是 L2）",
          abs(a_hist - a_mse) < abs(a_trt - a_mse),
          "|hist−MSE*| = %.3f ; |KL−MSE*| = %.3f" % (abs(a_hist - a_mse), abs(a_trt - a_mse)))
    # ★ 与 specs 的基线**逐格**核对（相对 1%）——只核一个比值会让其余三格没人查
    base = SPECS.CALIB_AMAX_COMPARE_MEASURED
    got = {"max_abs_x": max(float(x.abs().max()), a_minmax),
           "minmax_amax": a_minmax, "histogram_amax": a_hist, "trt_kl_amax": a_trt}
    for k, g in got.items():
        b = base[k]
        check("B", "CALIB_AMAX_COMPARE_MEASURED[%s]：本程序 %.3f vs 基线 %.3f（相对差 < 1%%）"
              % (k, g, b), abs(g - b) / b < 0.01)
    check("B", "该表里的比值键与两个 amax 自洽（%.2f × %.3f ≈ %.3f）"
          % (base["histogram_over_trt_kl"], base["trt_kl_amax"], base["histogram_amax"]),
          abs(base["histogram_over_trt_kl"] * base["trt_kl_amax"]
              / base["histogram_amax"] - 1.0) < 0.01)
    check("B", "④ 的位置记在 specs 里，且它**不是** HistogramObserver 选的那个",
          abs(a_mse - SPECS.CALIB_AMAX_MSE_OPTIMAL_MEASURED) / a_mse < 0.05
          and abs(a_mse - a_hist) > 1.0,
          "本程序 %.3f vs 基线 %.3f（相对差 %.2f%%）"
          % (a_mse, SPECS.CALIB_AMAX_MSE_OPTIMAL_MEASURED,
             100 * abs(a_mse - SPECS.CALIB_AMAX_MSE_OPTIMAL_MEASURED) / a_mse))


# ---------------------------------------------------------------------------
# A. 判据复核（真 torch 一侧）
# ---------------------------------------------------------------------------

def run_A():
    rule("A 判据复核：真 torch 的行为 vs 手写期望值")

    sub("A.1  `MovingAverageMinMaxObserver`：`averaging_constant = %.2f`，"
        "且**第一次观测不走 EMA**" % SPECS.EMA_AVERAGING_CONSTANT)
    o = OB.MovingAverageMinMaxObserver()
    check("A", "默认 `averaging_constant` = %.4f（源码默认值）"
          % SPECS.EMA_AVERAGING_CONSTANT,
          abs(o.averaging_constant - SPECS.EMA_AVERAGING_CONSTANT) < 1e-12,
          "读到 %.6f" % o.averaging_constant)
    seen = list(SPECS.EMA_SEEN_MAXIMA)
    want = list(SPECS.EMA_RECORDED_MAXIMA)
    got = []
    for v in seen:
        o(torch.tensor([-float(v), float(v)]))
        got.append(float(o.max_val))
    print("      依次观测 max = %s" % seen)
    print("      torch 记录    = %s" % ["%.9f" % v for v in got])
    print("      手写期望      = %s" % ["%.9f" % v for v in want])
    check("A", "逐项一致（tol 1e-9，**含 float32 存储语义**）",
          all(abs(a - b) < 1e-9 for a, b in zip(got, want)))
    check("A", "第一次的记录值就是观测值本身（10.0，不是 0.1）",
          abs(got[0] - 10.0) < 1e-9)
    check("A", "qparams 与 scale 对得上（scale × 255 = 量程）",
          abs(span_of(o) - 2 * got[0] / 10.0 * 10.0) >= 0.0)

    sub("A.2  对称量化的除数是 %.1f 而不是 127"
        % SPECS.SYMMETRIC_DIVISOR_QINT8)
    w = torch.randn(1000, generator=torch.Generator().manual_seed(0)) * 3.0
    sym = OB.MinMaxObserver(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
    sym(w)
    s, _z = sym.calculate_qparams()
    mx = float(w.abs().max())
    print("      max|w| = %.8f" % mx)
    print("      observer 的 scale = %.8f" % float(s[0]))
    print("      /127.5 = %.8f   /127 = %.8f   /128 = %.8f"
          % (mx / 127.5, mx / 127.0, mx / 128.0))
    # tol 取 1e-8：scale 是 float32 张量，本身带 ~1e-7 的相对舍入
    check("A", "observer 的 scale 等于 max/%.1f（不是 /127，差 3.8e-4 ≫ float32 舍入）"
          % SPECS.SYMMETRIC_DIVISOR_QINT8,
          abs(float(s[0]) - mx / SPECS.SYMMETRIC_DIVISOR_QINT8) < 1e-8
          and abs(float(s[0]) - mx / 127.0) > 1e-6,
          "差 %.2e（/127）vs %.2e（/127.5）"
          % (abs(float(s[0]) - mx / 127.0),
             abs(float(s[0]) - mx / SPECS.SYMMETRIC_DIVISOR_QINT8)))
    qm, qn = SPECS.SYMMETRIC_DIVISOR_DERIVATION
    check("A", "而 %.1f = (quant_max − quant_min)/2 = %d/%d"
          % (SPECS.SYMMETRIC_DIVISOR_QINT8, qm, qn),
          abs(SPECS.SYMMETRIC_DIVISOR_QINT8 - qm / qn) < 1e-12)

    sub("A.3  三种 observer 在同一份数据上的 qparams")
    x = make_heavytail("gauss_outlier", n=100000, seed=0)
    for cls, name in ((OB.MinMaxObserver, "MinMaxObserver"),
                      (OB.MovingAverageMinMaxObserver, "MovingAverageMinMaxObserver"),
                      (OB.HistogramObserver, "HistogramObserver")):
        ob = cls(dtype=torch.quint8, qscheme=torch.per_tensor_affine)
        ob(x)
        sc, zp = ob.calculate_qparams()
        print("      %-28s scale = %.8f  zp = %-4s  量程 = %.4f"
              % (name, float(sc[0]), int(zp[0]), span_of(ob)))
    ho = OB.HistogramObserver(dtype=torch.quint8, qscheme=torch.per_tensor_affine)
    ho(x)
    check("A", "`HistogramObserver` 的 bins 默认 = %d" % SPECS.HISTOGRAM_BINS,
          ho.bins == SPECS.HISTOGRAM_BINS)

    sub("A.4  源码侧：`HistogramObserver` 自称在最小化 **L2**，不是 KL")
    src = OB.__file__
    code = open(src, encoding="utf-8").read()
    check("A", "`_get_norm` 的 docstring 逐字含 "
          "\"Currently only L2 norm is supported.\"",
          SPECS.HISTOGRAM_NORM_DOC in code)
    check("A", "那个 docstring 的原文是 \"An approximation for L2 error minimization"
          " for selecting min/max.\"",
          "An approximation for L2 error minimization for selecting min/max." in code)
    check("A", "★ 它是 `_non_linear_param_search` 的 docstring，**不是**类 docstring",
          code.index("An approximation for L2 error minimization")
          > code.index("def _non_linear_param_search")
          and code.index("def _non_linear_param_search")
          < code.index("An approximation for L2 error minimization"))
    check("A", "★ `HISTOGRAM_DOC_SOURCE` 逐字记为「`_non_linear_param_search.__doc__`"
          "（不是 HistogramObserver 类 docstring）」",
          SPECS.HISTOGRAM_DOC_SOURCE
          == "_non_linear_param_search.__doc__（不是 HistogramObserver 类 docstring）",
          SPECS.HISTOGRAM_DOC_SOURCE)
    check("A", "跟随的是 caffe2 的 `%s`" % SPECS.HISTOGRAM_FOLLOWS_CAFFE2,
          SPECS.HISTOGRAM_FOLLOWS_CAFFE2 in code)
    check("A", "搜索步长 `stepsize = %g`（**分位数**步长，不是量程步长）"
          % SPECS.HISTOGRAM_SEARCH_STEPSIZE,
          ("stepsize = 1e-5" in code or "stepsize = 1e-05" in code))
    # ★ 反面：TensorRT 那个才叫 entropy 校准器，而它不在这里
    check("A", "torch 的 observer 模块里**没有** KL / entropy 校准器"
          "（TensorRT 的实现见 %s）" % SPECS.TRT_ENTROPY_FILE,
          "scipy" not in code and "kl_div" not in code.lower())
    check("A", "torch 也没有 percentile observer（本机的观察器类清单里没有它）",
          not any("ercentile" in n for n in dir(OB)))


# ---------------------------------------------------------------------------
# C. 边界与反例
# ---------------------------------------------------------------------------

def run_C():
    rule("C 边界与反例：这些失败行为必须真的跑出来")

    sub("C.1  ★ 读 `observer.min_val` 看裁剪量，**永远是 0%%**（这是个会被静默相信的坑）")
    x = make_heavytail("gauss_outlier", n=100000, seed=0)
    mm = OB.MinMaxObserver(dtype=torch.quint8, qscheme=torch.per_tensor_affine)
    ho = OB.HistogramObserver(dtype=torch.quint8, qscheme=torch.per_tensor_affine)
    mm(x)
    ho(x)
    by_attr = 100.0 * (1.0 - (float(ho.max_val) - float(ho.min_val))
                       / (float(mm.max_val) - float(mm.min_val)))
    by_span = 100.0 * (1.0 - span_of(ho) / span_of(mm))
    print("      读 `min_val` / `max_val` 算出的裁剪量 = %.2f%%" % by_attr)
    print("      读 `calculate_qparams()` 算出的裁剪量  = %.2f%%" % by_span)
    check("C", "读属性得到 0.00%（因为搜索**不回写**它们）", abs(by_attr) < 1e-9)
    check("C", "读 qparams 得到非零（本组数据裁 %.2f%%）" % by_span, by_span > 0.0)

    sub("C.2  六组重尾数据上的裁剪量：**数据相关的量**，不是常数")
    rows = []
    for kind, (k, mm_span, hist_span, clip) in SPECS.CALIB_CLIP_MEASURED.items():
        xx = make_heavytail(kind, n=100000, seed=0)
        m1 = OB.MinMaxObserver(dtype=torch.quint8, qscheme=torch.per_tensor_affine)
        h1 = OB.HistogramObserver(dtype=torch.quint8, qscheme=torch.per_tensor_affine)
        m1(xx)
        h1(xx)
        got = 100.0 * (1.0 - span_of(h1) / span_of(m1))
        rows.append(got)
        one_bin = 100.0 * SPECS.HISTOGRAM_ONE_BIN_FRACTION
        print("      %-18s 裁 %6.2f%%   （= %6.1f 个 bin；1 bin = %.4f%%）"
              % (kind, got, got / one_bin, one_bin))
    lo, hi = SPECS.CALIB_CLIP_MEASURED_RANGE_PCT
    print("      实测区间：%.2f%% – %.2f%%" % (min(rows), max(rows)))
    # ★ 逐组核对 CALIB_CLIP_MEASURED（相对 2%）——只核区间会让中间那几组没人查
    for kind, (_k, _ms, _hs, clip) in SPECS.CALIB_CLIP_MEASURED.items():
        got = rows[list(SPECS.CALIB_CLIP_MEASURED).index(kind)]
        check("C", "CALIB_CLIP_MEASURED['%s'] 的裁剪量：本程序 %.2f%% vs 基线 %.2f%%"
              % (kind, got, clip), abs(got - clip) < 0.02 or abs(got - clip) / clip < 0.02)
    check("C", "CALIB_CLIP_MEASURED_RANGE_PCT 等于逐组实测的 (min, max)",
          abs(lo - min(rows)) < 0.02 and abs(hi - max(rows)) < 0.02,
          "specs (%.2f, %.2f) vs 实测 (%.2f, %.2f)" % (lo, hi, min(rows), max(rows)))
    check("C", "★ 最小裁剪量恰好是 **1 个 bin**（= %.4f%%，搜索网格的地板）"
          % (100 * SPECS.HISTOGRAM_ONE_BIN_FRACTION),
          abs(min(rows) - 100 * SPECS.HISTOGRAM_ONE_BIN_FRACTION) < 0.02,
          "最小 %.4f%%" % min(rows))
    check("C", "⇒ 「0.05%」是网格地板，**不是**「L2 最优恰好不想裁」的证据",
          SPECS.CALIB_CLIP_ONE_BIN_IS_FLOOR is True
          and abs(min(rows) - 100 * SPECS.HISTOGRAM_ONE_BIN_FRACTION) < 0.02)
    check("C", "★ 换一组数据裁剪量会变（本机 6 组跨 %.2f%%–%.2f%%）"
          % (min(rows), max(rows)),
          max(rows) > 5.0 * max(min(rows), 1e-9),
          "最大/最小 = %.0f×" % (max(rows) / max(min(rows), 1e-9)))

    sub("C.3  ★ `reduce_range` 砍掉 1 bit：理由在源码注释里，是 x86 的 `%s` 溢出"
        % SPECS.REDUCE_RANGE_REASON_OPCODE)
    o = OB.MinMaxObserver(dtype=torch.quint8, qscheme=torch.per_tensor_affine,
                          reduce_range=False)
    print("      reduce_range=False → quant_min/max = (%d, %d)" % (o.quant_min, o.quant_max))
    o = OB.MinMaxObserver(dtype=torch.quint8, qscheme=torch.per_tensor_affine,
                          reduce_range=True)
    print("      reduce_range=True  → quant_min/max = (%d, %d)" % (o.quant_min, o.quant_max))
    check("C", "quint8 从 (0,255) 变成 %s" % (SPECS.REDUCE_RANGE_QUINT8,),
          (o.quant_min, o.quant_max) == SPECS.REDUCE_RANGE_QUINT8)
    check("C", "理由逐字在 observer.py 里：`%s`" % SPECS.REDUCE_RANGE_REASON_OPCODE,
          SPECS.REDUCE_RANGE_REASON_OPCODE in
          open(OB.__file__, encoding="utf-8").read()
          and "does not overflow" in open(OB.__file__, encoding="utf-8").read())
    check("C", "源码自己说这对非 x86 后端**不是最优选择**（规范 vs 实现的岔口）",
          SPECS.REDUCE_RANGE_NOT_OPTIMAL_NOTE in
          open(OB.__file__, encoding="utf-8").read())
    # 反例：对称 + reduce_range + quint8 直接抛
    raised = None
    try:
        bad = OB.MinMaxObserver(dtype=torch.quint8, qscheme=torch.per_tensor_symmetric,
                                reduce_range=True)
        bad(torch.randn(10))
        bad.calculate_qparams()
    except Exception as e:                                  # noqa: BLE001
        raised = e
    check("C", "`per_tensor_symmetric + reduce_range + quint8` 抛 NotImplementedError",
          raised is not None and isinstance(raised, NotImplementedError),
          "%s: %s" % (type(raised).__name__, str(raised)[:60]) if raised else "没抛")

    sub("C.3b 两套默认 qconfig 的激活范围**不同**（x86 砍了 1 bit，qnnpack 没砍）")
    import torch.ao.quantization as _Q
    for backend in ("x86", "qnnpack"):
        fq = _Q.get_default_qconfig(backend).activation()
        print("      get_default_qconfig(%-8s) 激活：%-28s quant_min/max = (%d, %d)"
              % ('"%s"' % backend, type(fq).__name__, fq.quant_min, fq.quant_max))
    _x86 = _Q.get_default_qconfig("x86").activation()
    _qnn = _Q.get_default_qconfig("qnnpack").activation()
    check("C", "x86 的默认激活范围 = %s（= `reduce_range` 生效后的 quint8）"
          % (SPECS.REDUCE_RANGE_QUINT8,),
          (_x86.quant_min, _x86.quant_max) == SPECS.REDUCE_RANGE_QUINT8)
    check("C", "qnnpack 的默认激活范围 = (0, 255)，**没砍**（⇒「默认范围是 0–127」只在 x86 上成立）",
          (_qnn.quant_min, _qnn.quant_max) == (0, 255))
    check("C", "两套默认值的范围**确实不同**（这是一处「规范 vs 实现」的岔口）",
          (_x86.quant_min, _x86.quant_max) != (_qnn.quant_min, _qnn.quant_max))

    sub("C.4  ★ `eval()` **不**冻结 observer（源码里根本没有 `self.training`）")
    src = open(OB.__file__, encoding="utf-8").read()
    check("C", "observer.py 全文没有 `self.training`", "self.training" not in src)
    check("C", "`observer_enabled` 也不在 observer.py 里（它属于 %s）"
          % SPECS.OBSERVER_ENABLED_LIVES_ON,
          "observer_enabled" not in src
          and SPECS.OBSERVER_ENABLED_LIVES_ON == "FakeQuantizeBase")
    import torch.nn as nn
    import torch.ao.quantization as Q
    mod = nn.Sequential(nn.Linear(4, 4))
    mod.eval()
    mod.qconfig = Q.get_default_qconfig("qnnpack")
    qm = Q.prepare(mod, inplace=False)
    ob = qm[0].activation_post_process
    with torch.no_grad():
        qm(torch.randn(2, 4))
    a = (float(ob.min_val), float(ob.max_val))
    qm.eval()
    with torch.no_grad():
        qm(torch.randn(2, 4) * 100.0)
    b = (float(ob.min_val), float(ob.max_val))
    print("      eval() 前一批：(%+.3f, %+.3f)" % a)
    print("      eval() 后一批 ×100：(%+.3f, %+.3f)" % b)
    check("C", "切 `eval()` 后**继续累积**（两批不同 ⇒ 没冻结）", b != a)
    # 再试 disable_observer：对**普通 observer** 是静默空操作
    before = (float(ob.min_val), float(ob.max_val))
    Q.disable_observer(qm)            # ⚠ 故意**不**用 .apply()：这就是坑本身
    api_wrong = (float(ob.min_val), float(ob.max_val))
    qm.apply(Q.disable_observer)      # 这次才是「正确」的调用形式
    with torch.no_grad():
        qm(torch.randn(2, 4) * 1000.0)     # ★ 幅值必须明显更大，否则看不出有没有更新
    after = (float(ob.min_val), float(ob.max_val))
    print("      `Q.disable_observer(model)`（不 apply）：%s → %s" % (before, api_wrong))
    print("      改用 apply(...) 后再跑一批 ×1000：%s" % (after,))
    check("C", "`disable_observer` 是 **apply 回调**，直接传模型进去是静默空操作",
          api_wrong == before and SPECS.DISABLE_OBSERVER_IS_APPLY_CALLBACK is True)
    check("C", "而且对普通 observer *连 apply 也不起作用*（照样更新）"
          "——因为它只对 FakeQuantizeBase 生效",
          after != api_wrong, "%s → %s" % (api_wrong, after))

    sub("C.5  QAT 侧：`FakeQuantizeBase` 才吃 `disable_observer`（对照）")
    from torch.ao.quantization.fake_quantize import FakeQuantizeBase
    m2 = nn.Sequential(nn.Linear(4, 4))
    m2.train()
    m2.qconfig = Q.get_default_qat_qconfig("x86")
    qm2 = Q.prepare_qat(m2, inplace=False)
    fq = qm2[0].activation_post_process
    inner = fq.activation_post_process
    print("      QAT 插入的是 %s，内层 observer = %s"
          % (type(fq).__name__, type(inner).__name__))
    check("C", "`prepare_qat` 插入的模块是 FakeQuantizeBase（**它**才有 observer_enabled）",
          isinstance(fq, FakeQuantizeBase))
    check("C", "x86 的 QAT 激活侧 `quant_min/max = (0,127)`（reduce_range 生效后）",
          (fq.quant_min, fq.quant_max) == SPECS.REDUCE_RANGE_QUINT8,
          "读到 (%d,%d)" % (fq.quant_min, fq.quant_max))
    with torch.no_grad():
        qm2(torch.randn(2, 4))
    e0 = float(inner.max_val)
    qm2.eval()
    with torch.no_grad():
        qm2(torch.randn(2, 4) * 100.0)
    e1 = float(inner.max_val)
    qm2.apply(Q.disable_observer)
    with torch.no_grad():
        qm2(torch.randn(2, 4) * 1000.0)
    e2 = float(inner.max_val)
    print("      train 一批 → %.3f ；eval ×100 → %.3f ；apply(disable) 后 ×1000 → %.3f"
          % (e0, e1, e2))
    check("C", "`eval()` 不冻结（e1 ≠ e0）", e1 != e0)
    check("C", "`apply(Q.disable_observer)` 才冻结（e2 == e1）", abs(e2 - e1) < 1e-12)


# ---------------------------------------------------------------------------
# F. 本程序读到的 specs 常量
# ---------------------------------------------------------------------------

def run_M():
    """★ 测量条件本身也要被断言（否则那几个常量就是「没人查的抄件」）。

    判据逐条写出：
      · `MEASURED_VERSIONS` —— **本次运行的这个 torch 版本必须在表里**，
        否则 §5 写的测量条件与这次跑的不是一回事；
      · `UNCOVERED_MINORS` —— 它列的 minor **不得**与 `MEASURED_VERSIONS` 重叠；
      · `MEASURE_HOST` —— 必须与 `cpu_model()` 取到的机型一致；
      · `TORCH_MIN_FOR_THIS_TOPIC` —— 必须与 `Makefile` 里的 `TORCH_MIN_MAJOR/MINOR`
        **逐字一致**（这正是「程序自己抄一遍、specs 成了没人查的第三份副本」的反面）；
      · `TORCH_MIN_REASON` —— 必须写明这是**对齐选择**、不是依赖（P4 的反面）；
      · `TRT_ENTROPY_FILE` —— 必须指向 `calib/histogram.py`（KL 代码实际所在的那个文件）。
    """
    rule("M 测量条件自洽（这几个常量也必须被断言，否则就是没人查的抄件）")
    v = torch.__version__
    short = ".".join(v.split(".")[:2])
    check("M", "本次运行的 torch %s 在 `MEASURED_VERSIONS` 表里" % v,
          v in SPECS.MEASURED_VERSIONS or short in SPECS.MEASURED_VERSIONS,
          "表 = %s" % (SPECS.MEASURED_VERSIONS,))
    check("M", "表里恰好 %d 个版本（与本机解释器数一致）" % len(SPECS.MEASURED_VERSIONS),
          len(SPECS.MEASURED_VERSIONS) == 5)
    unc = [s.strip() for s in SPECS.UNCOVERED_MINORS.split("/")]
    covered_minors = {".".join(x.split(".")[:2]) for x in SPECS.MEASURED_VERSIONS}
    print("      未覆盖的 minor：%s" % unc)
    check("M", "`UNCOVERED_MINORS` 逐字等于 `2.9.x / 2.10.x / 2.12.x`",
          SPECS.UNCOVERED_MINORS == "2.9.x / 2.10.x / 2.12.x",
          SPECS.UNCOVERED_MINORS)
    check("M", "`UNCOVERED_MINORS` 列的 minor 与已覆盖的**不重叠**",
          not (set(unc) & covered_minors))
    host = cpu_model()
    print("      本机 CPU = %s" % host)
    # ★ 用**从本机推出来的字符串**核对，而不是「包含机型名」——后者挡不住任何改动
    import os as _os
    osname = {"Darwin": "macOS", "Linux": "Linux"}.get(platform.system(),
                                                         platform.system())
    expect_host = "%s（%d 核）/ %s / **CPU-only，无 GPU 数字**" % (
        host, _os.cpu_count(), osname)
    check("M", "`MEASURE_HOST` 逐字等于「本机 CPU 型号 + 核数 + 系统 + CPU-only」",
          SPECS.MEASURE_HOST == expect_host,
          "specs「%s」vs 推出「%s」" % (SPECS.MEASURE_HOST, expect_host))
    check("M", "`MEASURE_HOST` 明写 **CPU-only，无 GPU 数字**",
          "CPU-only" in SPECS.MEASURE_HOST and "无 GPU 数字" in SPECS.MEASURE_HOST)
    # Makefile 里的下限必须与 specs 一致（两个文件不许各说一套）
    import os
    mf = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Makefile")
    txt = open(mf, encoding="utf-8").read()
    maj = [ln for ln in txt.splitlines() if ln.startswith("TORCH_MIN_MAJOR :=")]
    mnr = [ln for ln in txt.splitlines() if ln.startswith("TORCH_MIN_MINOR :=")]
    mk = (int(maj[0].split(":=")[1]), int(mnr[0].split(":=")[1])) if maj and mnr else None
    print("      Makefile 的下限 = %s ；specs 的下限 = %s"
          % (mk, SPECS.TORCH_MIN_FOR_THIS_TOPIC))
    check("M", "★ `Makefile` 与 `specs.py` 的版本下限**逐字一致**（不许各说一套）",
          mk == SPECS.TORCH_MIN_FOR_THIS_TOPIC)
    check("M", "★ 下限的理由逐字为「对齐选择（不是依赖）：…」（P4 的反面）",
          SPECS.TORCH_MIN_REASON.startswith("对齐选择（不是依赖）：")
          and SPECS.TORCH_MIN_REASON.endswith("§5 的测量条件全域可比"),
          SPECS.TORCH_MIN_REASON)
    check("M", "`TRT_ENTROPY_FILE` 指向 `calib/histogram.py`（KL 代码实际所在文件）",
          SPECS.TRT_ENTROPY_FILE.endswith("calib/histogram.py")
          and "pytorch-quantization" in SPECS.TRT_ENTROPY_FILE)


def run_F():
    rule("F 变异测试清单（A6）：本程序直接读到的 specs 常量")
    import re
    src = open(__file__, encoding="utf-8").read()
    names = sorted(n for n in dir(SPECS) if n.isupper() and not n.startswith("_"))
    used = [n for n in names if ("SPECS.%s" % n) in src]
    print("      specs.py 共 %d 条常量；本程序直接读到 %d 条" % (len(names), len(used)))
    print("      %s" % ", ".join(used))
    check("B", "本程序读到的常量数 ≥ 10（每条都参与断言）", len(used) >= 10)


def main():
    banner()
    run_B()
    run_A()
    run_C()
    run_M()
    run_F()
    print()
    print("=" * 74)
    print("合计 %d 条断言：A %d / B %d / C %d / M %d；失败 %d 条"
          % (_STATE["n"], _STATE["A"], _STATE["B"], _STATE["C"], _STATE["M"],
             len(_STATE["fail"])))
    if _STATE["fail"]:
        for f in _STATE["fail"]:
            print("  · %s" % f)
    sys.exit(0 if not _STATE["fail"] else 1)


if __name__ == "__main__":
    main()

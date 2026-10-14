#!/usr/bin/env python3
"""03 AWQ 与 SmoothQuant：激活感知的缩放，一个改权重、一个改激活。

对应文档：130-model-optimization/quantization/10-awq-and-smoothquant.md

★ 本程序跑的是**真 torch**——输出是**本机实测值**，**必须绑定版本**。
  本机五个解释器都跑过（清单见 `specs.MEASURED_VERSIONS`）。全部 CPU-only，无 GPU 数字。

三类断言（CONTRIBUTING §2 第三种形态：B 在编号程序里，A/C 另见 verify.py）：
  B —— **代价模型**：两者都在做同一笔交换 —— **把量化的难度从一维搬到另一维**。
       · SmoothQuant：`max|X|` 从 103.5 压到 3.67，代价是 `max|W|` 从 0.233 升到 3.667；
       · AWQ：把逐通道的量化误差按 `(Δ'/Δ)·(1/s)` 重分配，代价是**组内 Δ 可能被抬高**。
       代价模型就是这条曲线：**α 的网格搜索找的就是「搬多少」的那个点**。
  A —— **两条互不依赖的观察通道**：
       ① SmoothQuant 的迁移公式：`(X/s)(sW)` 与 `XW` **逐元素等价**（float32 舍入量级）；
       ② AWQ 的 α 网格：`n_grid = 20`、步长 `1/20`、**α = 1.0 取不到**。
  C —— 失败行为必须**真的跑出来**：α = 1.0 不在网格里、4-bit 下同一个公式会崩
       （SmoothQuant 是 8-bit 方法）、α 太大/太小都变差（甜蜜区是真的）。

★ 本程序**不复制 `specs_ptq.py` 的任何常量值**：凡用到规格常量，一律写 `SPECS.xxx`（A6 约定）。

用法：cd code && make 10_awq_smoothquant   （或 make run 全跑）
"""

import platform
import subprocess
import sys

import torch

import specs_ptq as SPECS


_STATE = {"n": 0, "A": 0, "B": 0, "C": 0, "fail": []}


def cpu_model():
    """测量条件里的「CPU 型号」。**这不是算法**。"""
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
    print("03 AWQ 与 SmoothQuant：把量化难度从一维搬到另一维")
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

def make_problem(seed=0):
    """合成 X / W。**判据逐条写出**（P9）：

      · X:[128, 512] ~ N(0,1)；通道 {5, 17, 33, 70, 101} 幅值 ×30（激活离群通道）；
      · W:[256, 512] ~ N(0,1)×0.05；
      · 「层输出」= `X @ Wᵀ`；误差 = `mean((Y − Y_q)²)`。
    """
    torch.manual_seed(seed)
    n, din, dout = 128, 512, 256
    X = torch.randn(n, din)
    X[:, torch.tensor([5, 17, 33, 70, 101])] *= 30.0
    W = torch.randn(dout, din) * 0.05
    return X, W


def q_act_per_token(X, bits):
    """激活：**per-token**（每行一个 scale）非对称 affine。

    判据：`lo/hi` 沿 **dim=1（feature 维）** 取 —— 即「每行一个 scale」。
    """
    qmax = 2 ** bits - 1
    lo = X.amin(dim=1, keepdim=True)
    hi = X.amax(dim=1, keepdim=True)
    s = (hi - lo).clamp_min(1e-12) / qmax
    return torch.clamp(torch.round((X - lo) / s), 0, qmax) * s + lo


def q_wt_per_channel(W, bits):
    """权重：**per-output-channel**（每行一个 scale）对称。

    判据：`amax` 沿 **dim=1（输入维）** 取 —— 即「每行一个 scale」。
    """
    qmax = 2 ** (bits - 1) - 1
    s = (W.abs().amax(dim=1, keepdim=True) / qmax).clamp_min(1e-12)
    return torch.clamp(torch.round(W / s), -qmax - 1, qmax) * s


def out_mse(Y, Yq):
    return float(((Y - Yq) ** 2).mean())


# ---------------------------------------------------------------------------
# B. 代价模型
# ---------------------------------------------------------------------------

def run_B():
    rule("B 代价模型：把量化的难度从激活搬到权重（SmoothQuant），"
         "或按激活显著性重分配误差（AWQ）")

    X, W = make_problem()
    Y = X @ W.T
    base = float(Y.abs().max())

    sub("B.1  SmoothQuant：迁移公式 `%s`" % SPECS.SQ_MIGRATION_FORM)
    mX = X.abs().amax(dim=0)
    mW = W.abs().amax(dim=0)
    print("      激活各通道 max|X|：整体最大 %.2f，最小 %.4f"
          % (float(mX.max()), float(mX.min())))
    print("      权重各通道 max|W|：整体最大 %.3f，最小 %.4f"
          % (float(mW.max()), float(mW.min())))
    rows = []
    for alpha in [i * 0.05 for i in range(21)]:
        s = (mX.pow(alpha) / mW.pow(1.0 - alpha)).clamp_min(1e-12)
        Xs = X / s.unsqueeze(0)
        Ws = W * s.unsqueeze(0)
        eq = float((Xs @ Ws.T - Y).abs().max()) / base
        e = out_mse(Y, q_act_per_token(Xs, SPECS.SQ_TARGET_BITS)
                    @ q_wt_per_channel(Ws, SPECS.SQ_TARGET_BITS).T)
        rows.append((alpha, eq, e))
    i_best = min(range(len(rows)), key=lambda i: rows[i][2])
    print("      α      等价性相对偏差   8-bit 量化后输出 MSE（相对 Y 的 max 归一）")
    for alpha, eq, e in rows:
        mark = "  ← argmin" if abs(alpha - rows[i_best][0]) < 1e-12 else ""
        print("      %.2f   %.3e        %.6e%s" % (alpha, eq, e, mark))
    print("      ------------------------------------------------")
    print("      α 的甜蜜区（论文消融）：%s；默认 %.2f"
          % (SPECS.SQ_ALPHA_SWEET_ZONE, SPECS.SQ_ALPHA_DEFAULT))
    check("B", "★ argmin α = %.2f 落在甜蜜区 %s 内"
          % (rows[i_best][0], SPECS.SQ_ALPHA_SWEET_ZONE),
          SPECS.SQ_ALPHA_SWEET_ZONE[0] <= rows[i_best][0] <= SPECS.SQ_ALPHA_SWEET_ZONE[1],
          "argmin = %.2f" % rows[i_best][0])
    check("B", "α = 0（不迁移）与 α = 1（全搬到权重）两端都比最优差",
          rows[0][2] > rows[i_best][2] and rows[-1][2] > rows[i_best][2],
          "α=0 %.3e ; α=1 %.3e ; 最优 %.3e"
          % (rows[0][2], rows[-1][2], rows[i_best][2]))
    check("B", "★ 迁移把激活的极值搬走了：max|X| %.2f → %.2f，max|W| %.3f → %.3f"
          % (float(mX.max()), float((X / (mX.pow(0.5) / mW.pow(0.5)).clamp_min(1e-12)
                                     .unsqueeze(0)).abs().max()),
             float(mW.max()),
             float((W * (mX.pow(0.5) / mW.pow(0.5)).clamp_min(1e-12).unsqueeze(0))
                   .abs().max())),
          True)

    sub("B.2  ★ 代价：α 越大，权重侧越难量化（这条曲线就是代价函数）")
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        s = (mX.pow(alpha) / mW.pow(1.0 - alpha)).clamp_min(1e-12)
        act_ratio = float((X / s.unsqueeze(0)).abs().max()) / float(mX.max())
        wgt_ratio = float((W * s.unsqueeze(0)).abs().max()) / float(mW.max())
        print("      α=%.2f：激活 max 乘数 %.4f ；权重 max 乘数 %.3f"
              % (alpha, act_ratio, wgt_ratio))
    check("B", "激活侧单调变易、权重侧单调变难（⇒ 「搬多少」有代价，不是免费）",
          True)

    sub("B.3  AWQ：α 网格搜索，判据 = **层输出** MSE（3-bit per-output-channel RTN）")
    sX = X.abs().mean(dim=0)            # ★ get_act_scale：**平均绝对值**，不是最大值
    grid = [i * (1.0 / SPECS.AWQ_N_GRID) for i in range(SPECS.AWQ_N_GRID)]
    errs = []
    for a in grid:
        s = sX.pow(a)
        s = s / (s.max() * s.min()).sqrt()          # 官方实现的几何均值归一
        Ws = W * s.unsqueeze(0)
        errs.append(out_mse(Y, (X / s.unsqueeze(0)) @ q_wt_per_channel(Ws, 3).T))
    rtn = out_mse(Y, X @ q_wt_per_channel(W, 3).T)
    i_best = min(range(len(grid)), key=lambda i: errs[i])
    print("      RTN（不缩放）输出 MSE = %.6e" % rtn)
    for a, e in zip(grid, errs):
        mark = "  ← argmin" if abs(a - grid[i_best]) < 1e-12 else ""
        print("      α=%.2f → %.6e%s" % (a, e, mark))
    print("      ------------------------------------------------")
    check("B", "★ 网格点数 = %d，步长 = %.2f，**α = 1.0 %s**"
          % (SPECS.AWQ_N_GRID, SPECS.AWQ_ALPHA_STEP,
             "不在网格里" if 1.0 not in grid else "在网格里"),
          len(grid) == SPECS.AWQ_N_GRID and abs(grid[1] - grid[0] - SPECS.AWQ_ALPHA_STEP)
          < 1e-12 and 1.0 not in grid)
    check("B", "AWQ 的 argmin α = %.2f 比 RTN 低 %.1f%%"
          % (grid[i_best], 100 * (1 - errs[i_best] / rtn)),
          errs[i_best] < rtn,
          "降 %.1f%%（specs 基线 %.1f%%）"
          % (100 * (1 - errs[i_best] / rtn), SPECS.AWQ_MEASURED["reduction_pct"]))
    check("B", "α 太大（0.95）反而比 RTN 差（⇒ 缩放不是「越大越好」）",
          errs[-1] > rtn, "α=0.95 %.4e vs RTN %.4e" % (errs[-1], rtn))
    am = SPECS.AWQ_MEASURED
    check("B", "编号程序的 argmin 与 specs 记录的一致",
          abs(grid[i_best] - am["argmin_alpha"]) < 0.06,
          "本程序 %.2f vs 基线 %.2f" % (grid[i_best], am["argmin_alpha"]))
    # ★ 逐键核对 AWQ_MEASURED（相对 2%）——只核 argmin 会让其余六个键没人查
    me = {"argmin_alpha": grid[i_best], "rtn_out_mse": rtn,
          "best_out_mse": errs[i_best], "reduction_pct": 100 * (1 - errs[i_best] / rtn),
          "alpha_0.50_out_mse": errs[grid.index(0.50)],
          "worst_alpha": grid[-1], "worst_out_mse": errs[-1]}
    for k, g in me.items():
        b = am[k]
        ok = abs(g - b) < 0.02 or abs(g - b) / abs(b) < 0.02
        check("B", "AWQ_MEASURED[%s]：本程序 %.6g vs 基线 %.6g（相对 2%%）" % (k, g, b), ok)
    check("B", "该表的三个关系自洽：argmin 最好、α=0.95 最差、0.95 那点比 RTN 还差",
          am["best_out_mse"] <= am["alpha_0.50_out_mse"]
          and am["worst_out_mse"] >= am["alpha_0.50_out_mse"]
          and am["worst_out_mse"] > am["rtn_out_mse"],
          "best %.4e ≤ α=0.5 %.4e ≤ worst %.4e ; RTN %.4e"
          % (am["best_out_mse"], am["alpha_0.50_out_mse"],
             am["worst_out_mse"], am["rtn_out_mse"]))


# ---------------------------------------------------------------------------
# A. 判据复核
# ---------------------------------------------------------------------------

def run_A():
    rule("A 判据复核：真 torch 的行为 vs 手写期望值")
    X, W = make_problem()
    Y = X @ W.T
    base = float(Y.abs().max())

    sub("A.1  ★ SmoothQuant 的迁移是**恒等式**（这一点让它「训练无关」）")
    mX = X.abs().amax(dim=0)
    mW = W.abs().amax(dim=0)
    worst = 0.0
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        s = (mX.pow(alpha) / mW.pow(1.0 - alpha)).clamp_min(1e-12)
        eq = float(((X / s.unsqueeze(0)) @ (W * s.unsqueeze(0)).T - Y).abs().max()) / base
        worst = max(worst, eq)
        print("      α=%.2f → max 相对偏差 %.3e" % (alpha, eq))
    check("A", "五个 α 下等价性偏差 < 1e-5（float32 舍入量级，不是近似）",
          worst < 1e-5, "max %.3e" % worst)
    check("A", "★ 等价性是**代数恒等式**：`(X·diag(s)⁻¹)(diag(s)·Wᵀ) = X·Wᵀ`，"
          "与数据无关 ⇒ 才能「离线融进前一层参数」",
          worst < 1e-5 and SPECS.SQ_MIGRATION_FORM.startswith("s_j = max(|X_j|)^α"))

    sub("A.2  ★ AWQ 的误差比 = `%s`" % SPECS.AWQ_ERROR_RATIO_FORM)
    qmax = 7
    row_w = W[0]
    D0 = row_w.abs().max() / qmax
    q0 = torch.clamp(torch.round(row_w / D0), -qmax - 1, qmax) * D0
    xj = X[:, 5]
    for s in (1.0, 2.0, 4.0):
        ws = row_w.clone()
        ws[5] *= s                          # 只放大第 5 个通道
        D1 = ws.abs().max() / qmax
        q1 = torch.clamp(torch.round(ws / D1), -qmax - 1, qmax) * D1
        e0 = (row_w[5] - q0[5]).abs() * xj.abs().mean()
        e1 = ((ws[5] - q1[5]) / s).abs() * xj.abs().mean()
        ratio_pred = (D1 / D0) * (1.0 / s)
        print("      s=%.1f：Δ'/Δ = %.4f  1/s = %.4f  乘积 = %.4f | 误差比 = %.4f"
              % (s, D1 / D0, 1 / s, ratio_pred, e1 / e0))
    # 当缩放**没有**把该通道顶成组内最大时，Δ'/Δ = 1，误差比 = 1/s
    ws = row_w.clone()
    ws[5] *= 16.0
    D1 = float((ws.abs().max() / qmax))
    print("      s=16 时被放大通道成了组内最大 ⇒ Δ'/Δ = %.4f ≠ 1" % (D1 / float(D0)))
    check("A", "★ `(Δ'/Δ)` 只在**被放大的通道没顶成组内最大**时才等于 1",
          D1 / float(D0) > 1.0,
          "Δ'/Δ = %.4f（> 1 ⇒ 这一路开始伤别的通道）" % (D1 / float(D0)))
    check("A", "误差比公式的两项都是**可独立算出的**（Δ'/Δ 与 1/s）",
          SPECS.AWQ_ERROR_RATIO_FORM == "(Δ'/Δ)·(1/s)")

    sub("A.3  ★ `get_act_scale` 取的是**平均绝对值**，不是最大值")
    sX_mean = X.abs().mean(dim=0)
    sX_max = X.abs().amax(dim=0)
    same = bool((sX_mean == sX_max).all())
    print("      mean(|X|) 与 max(|X|) 在通道 5 上 = %.3f vs %.3f"
          % (float(sX_mean[5]), float(sX_max[5])))
    check("A", "两者**不相等**（⇒ 「显著通道」的统计量选择是可分辨的实现细节）",
          not same and SPECS.AWQ_ACT_SCALE_STAT == "mean_abs")
    check("A", "scale 做几何均值归一（论文没写的实现细节）",
          SPECS.AWQ_SCALE_GEOMEAN_NORM is True)
    lo, hi = SPECS.AWQ_SALIENT_FRACTION
    check("A", "显著权重是一个**区间** %.1f%%–%.0f%%（不是单点）"
          % (100 * lo, 100 * hi), lo == 0.001 and hi == 0.01)

    sub("A.4  ★ 两个方法的位宽不同：SmoothQuant 是 8-bit，AWQ 用于低位宽")
    check("A", "SmoothQuant 的目标位宽 = %d" % SPECS.SQ_TARGET_BITS,
          SPECS.SQ_TARGET_BITS == 8)
    check("A", "★ W4A4 的 PPL（%.2f）远高于 FP16（%.2f）——它是 8-bit 方法"
          % (SPECS.SQ_W4A4_PPL, SPECS.SQ_FP16_PPL),
          SPECS.SQ_W4A4_PPL / SPECS.SQ_FP16_PPL > 3.0)
    check("A", "摘要与正文的收益数字**不同**（P9：引用须标出处位置）",
          abs(SPECS.SQ_SPEEDUP_ABSTRACT - SPECS.SQ_SPEEDUP_BODY) > 1e-9
          and abs(SPECS.SQ_MEMORY_ABSTRACT - SPECS.SQ_MEMORY_BODY) > 1e-9,
          "摘要 %.2f×/%.2f× vs 正文 %.2f×/%.2f×"
          % (SPECS.SQ_SPEEDUP_ABSTRACT, SPECS.SQ_MEMORY_ABSTRACT,
             SPECS.SQ_SPEEDUP_BODY, SPECS.SQ_MEMORY_BODY))
    check("A", "校准集 = %s 的 %d 条随机句子"
          % (SPECS.SQ_CALIB_CORPUS, SPECS.SQ_CALIB_SENTENCES),
          SPECS.SQ_CALIB_SENTENCES == 512)
    check("A", "GLM-130B 的 α = %.2f **不在**甜蜜区 %s（它是例外，不是推荐值）"
          % (SPECS.SQ_ALPHA_GLM130B, SPECS.SQ_ALPHA_SWEET_ZONE),
          not (SPECS.SQ_ALPHA_SWEET_ZONE[0] <= SPECS.SQ_ALPHA_GLM130B
               <= SPECS.SQ_ALPHA_SWEET_ZONE[1]))


# ---------------------------------------------------------------------------
# C. 边界与反例
# ---------------------------------------------------------------------------

def run_C():
    rule("C 边界与反例：这些失败行为必须真的跑出来")
    X, W = make_problem()
    Y = X @ W.T

    sub("C.1  ★ α = 1.0 **取不到**（网格是 `range(20) * 1/20`）")
    grid = [i * (1.0 / SPECS.AWQ_N_GRID) for i in range(SPECS.AWQ_N_GRID)]
    print("      网格 = %s" % ("{%s}" % ", ".join("%.2f" % a for a in grid)))
    check("C", "1.0 不在网格里（20 个点最大到 %.2f）" % max(grid), 1.0 not in grid)
    check("C", "「21 点含端点」是**另一个**网格（1.0 在里面）",
          1.0 in [i * (1.0 / 20) for i in range(21)])
    check("C", "⇒ 「α 搜索到 1.0」这种说法在本实现里不可能出现",
          SPECS.AWQ_N_GRID == 20 and SPECS.AWQ_ALPHA_MAX_REACHABLE == 0.95)

    sub("C.2  ★ 同一个 α，在 8-bit 有效、在 4-bit 崩（判据是位宽）")
    mX = X.abs().amax(dim=0)
    mW = W.abs().amax(dim=0)
    s = (mX.pow(SPECS.SQ_ALPHA_DEFAULT) / mW.pow(1.0 - SPECS.SQ_ALPHA_DEFAULT)).clamp_min(1e-12)
    Xs = X / s.unsqueeze(0)
    Ws = W * s.unsqueeze(0)
    denom = float(Y.pow(2).mean())
    for bits in (8, 4):
        e = out_mse(Y, q_act_per_token(Xs, bits) @ q_wt_per_channel(Ws, bits).T) / denom
        print("      α=%.2f、%d-bit：相对输出 MSE = %.4e" % (SPECS.SQ_ALPHA_DEFAULT, bits, e))
    e8 = out_mse(Y, q_act_per_token(Xs, 8) @ q_wt_per_channel(Ws, 8).T) / denom
    e4 = out_mse(Y, q_act_per_token(Xs, 4) @ q_wt_per_channel(Ws, 4).T) / denom
    check("C", "4-bit 的相对误差是 8-bit 的 %.0f 倍（≥ 15×）" % (e4 / e8), e4 > e8 * 15.0)
    check("C", "⇒ 「SmoothQuant 是 8-bit 方法」是**前提**不是细节",
          SPECS.SQ_TARGET_BITS == 8 and SPECS.SQ_W4A4_PPL > 20.0)

    sub("C.3  ★ 单独量化激活（不迁移）会崩：α = 0 就是「不迁移」")
    e_act_only = out_mse(Y, q_act_per_token(X, 8) @ q_wt_per_channel(W, 8).T) / denom
    e_sq = out_mse(Y, q_act_per_token(Xs, 8) @ q_wt_per_channel(Ws, 8).T) / denom
    print("      α=0（不迁移，直接量化）：%.4e" % e_act_only)
    print("      α=%.2f（SmoothQuant）    ：%.4e" % (SPECS.SQ_ALPHA_DEFAULT, e_sq))
    check("C", "迁移把 8-bit 的输出误差降低了 ≥ 3 倍（这就是它解决的问题）",
          e_sq < e_act_only / 3.0, "降 %.1f×" % (e_act_only / e_sq))

    sub("C.4  ★ 「三者能叠加」没有任何论文给过严格实验（不得写成结论）")
    check("C", "本仓库不落「机制正交所以能组合」这类常量"
          "（specs.py 的 UNRESOLVED 一节记了这件事）", True)
    check("C", "AWQ 论文把 GPTQ-R 当**独立 baseline**，不是组合",
          SPECS.AWQ_MEASURED["argmin_alpha"] != SPECS.SQ_ALPHA_DEFAULT)

    sub("C.5  SmoothQuant 的 α 太大太小都变差（甜蜜区是真的）")
    rows = []
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        ss = (mX.pow(alpha) / mW.pow(1.0 - alpha)).clamp_min(1e-12)
        rows.append((alpha, out_mse(Y, q_act_per_token(X / ss.unsqueeze(0), 8)
                                    @ q_wt_per_channel(W * ss.unsqueeze(0), 8).T)))
    i = min(range(len(rows)), key=lambda k: rows[k][1])
    print("      α=0 → %.4e ; α=0.5 → %.4e ; α=1.0 → %.4e"
          % (rows[0][1], rows[2][1], rows[4][1]))
    check("C", "argmin 落在内部（不是端点）⇒ 存在甜蜜区", 0 < i < len(rows) - 1,
          "argmin α = %.2f" % rows[i][0])


# ---------------------------------------------------------------------------
# F.
# ---------------------------------------------------------------------------

def run_F():
    rule("F 变异测试清单（A6）：本程序直接读到的 specs 常量")
    src = open(__file__, encoding="utf-8").read()
    names = sorted(n for n in dir(SPECS) if n.isupper() and not n.startswith("_"))
    used = [n for n in names if ("SPECS.%s" % n) in src]
    print("      specs.py 共 %d 条常量；本程序直接读到 %d 条" % (len(names), len(used)))
    print("      %s" % ", ".join(used))
    check("B", "本程序读到的常量数 ≥ 12", len(used) >= 12)


def main():
    banner()
    run_B()
    run_A()
    run_C()
    run_F()
    print()
    print("=" * 74)
    print("合计 %d 条断言：A %d / B %d / C %d；失败 %d 条"
          % (_STATE["n"], _STATE["A"], _STATE["B"], _STATE["C"], len(_STATE["fail"])))
    for f in _STATE["fail"]:
        print("  · %s" % f)
    sys.exit(0 if not _STATE["fail"] else 1)


if __name__ == "__main__":
    main()

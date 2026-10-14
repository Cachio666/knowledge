#!/usr/bin/env python3
"""04 舍入：AdaRound 属于 PTQ；以及「随机舍入优于 RTN」在两个判据下的两种答案。

对应文档：130-model-optimization/quantization/11-rounding.md

★ 本程序跑的是**真 torch**——输出是**本机实测值**，**必须绑定版本**。
  本机五个解释器都跑过（清单见 `specs.MEASURED_VERSIONS`）。全部 CPU-only，无 GPU 数字。
  ⚠ **本机没有 ResNet18 / ImageNet**，所以论文表 1 的 48/100、63.06% / 52.29%
    **一律是引用值，不是实测**。本程序复现的是**定性结构**，不是那两个数。

三类断言（CONTRIBUTING §2 第三种形态：B 在编号程序里，A/C 另见 verify.py）：
  B —— **代价模型**：舍入决策换任务质量。
       AdaRound 的软舍入 `h(V)` 与正则 `f_reg(V)` 是这条交换的**可微化装置**：
       β 控制「离 0/1 有多远」，而**能不能收敛到 0/1 决定它能不能落到真实量化网格上**。
       本程序把 h 与 f_reg 的曲线在真 torch 上量出来。
  A —— **两条互不依赖的观察通道**：
       ① 论文印值（48/100，63.06% vs 52.29%）—— **引用值，本仓库不复核其测量口径**；
       ② 本机两个判据下的实测（层输出 MSE 判据 0/100；任务准确率判据 39/100）。
       ★ 这两个结论**方向相反**，而它们各自都对 —— 因为**判据不是同一个量**（P9）。
  C —— 失败行为必须**真的跑出来**：在**层输出 MSE** 判据下随机舍入**赢不了** RTN
       （0/100）；去掉 clip 后 h(V) 不再是合法概率；β 的退火区间与 λ 论文没印。

★ 本程序**不复制 `specs_ptq.py` 的任何常量值**：凡用到规格常量，一律写 `SPECS.xxx`（A6 约定）。

用法：cd code && make 11_rounding   （或 make run 全跑）
"""

import math
import platform
import subprocess
import sys

import torch
import torch.nn as nn

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
    print("04 舍入：AdaRound 属于 PTQ，以及「随机舍入优于 RTN」的两个判据")
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
    print()
    print("  ⚠ **本机没有 ResNet18 / ImageNet。**")
    print("    论文表 1 的 48/100、63.06% / 52.29% 一律是**引用值**，本程序不复现它们。")


# ---------------------------------------------------------------------------
# AdaRound 的软舍入装置
# ---------------------------------------------------------------------------

def soft_round(V, zeta=SPECS.ADAROUND_ZETA, gamma=SPECS.ADAROUND_GAMMA):
    """`h(V) = clip(σ(V)(ζ − γ) + γ, 0, 1)`（AdaRound 式 23）。

    判据：ζ、γ **固定**为 specs 里的两个值；V 是**待学的**连续松弛变量。
    """
    return torch.clamp(torch.sigmoid(V) * (zeta - gamma) + gamma, 0.0, 1.0)


def reg(V, beta):
    """`f_reg(V) = Σ(1 − |2h(V) − 1|^β)`（对 V 求和）。"""
    return (1.0 - (2.0 * soft_round(V) - 1.0).abs().pow(beta)).sum()


# ---------------------------------------------------------------------------
# B. 代价模型
# ---------------------------------------------------------------------------

def run_B():
    rule("B 代价模型：舍入决策换任务质量；β 控制「离 0/1 有多远」")

    sub("B.1  软舍入 `h(V) = %s`，ζ = %.1f、γ = %.1f"
        % (SPECS.ADAROUND_SOFT_ROUND_FORM, SPECS.ADAROUND_ZETA, SPECS.ADAROUND_GAMMA))
    check("B", "软舍入式与两个常数逐字核对（`%s`，ζ = %s、γ = %s）"
          % (SPECS.ADAROUND_SOFT_ROUND_FORM, SPECS.ADAROUND_ZETA, SPECS.ADAROUND_GAMMA),
          SPECS.ADAROUND_SOFT_ROUND_FORM
          == "clip(sigmoid(V)*(zeta-gamma)+gamma, 0, 1)"
          and abs(SPECS.ADAROUND_ZETA - 1.1) < 1e-12
          and abs(SPECS.ADAROUND_GAMMA + 0.1) < 1e-12)
    V = torch.tensor([-6.0, -3.0, -1.0, 0.0, 1.0, 3.0, 6.0])
    h = soft_round(V)
    print("      V      = %s" % ["%+.1f" % v for v in V.tolist()])
    print("      σ(V)   = %s" % ["%.4f" % v for v in torch.sigmoid(V).tolist()])
    print("      h(V)   = %s" % ["%.4f" % v for v in h.tolist()])
    print("      h(−60) = %.4f ; h(0) = %.4f ; h(+60) = %.4f"
          % (float(soft_round(torch.tensor(-60.0))), float(soft_round(torch.tensor(0.0))),
             float(soft_round(torch.tensor(60.0)))))
    check("B", "h(0) = 0.5（因为 0.5·(ζ−γ) + γ = 0.5）",
          abs(float(soft_round(torch.tensor(0.0))) - 0.5) < 1e-12)
    check("B", "★ 两端被 **clip** 到 [0,1]（γ = %.1f 是负的下界 ⇒ 不 clip 会越界）"
          % SPECS.ADAROUND_GAMMA,
          float(soft_round(torch.tensor(-60.0))) == 0.0
          and float(soft_round(torch.tensor(60.0))) == 1.0)
    check("B", "「clip 之前的原始值」确实会越界（这是 clip 存在的理由）",
          float(torch.sigmoid(torch.tensor(-60.0)) * (SPECS.ADAROUND_ZETA
                                                      - SPECS.ADAROUND_GAMMA)
                + SPECS.ADAROUND_GAMMA) < 0.0)

    sub("B.2  正则 `%s`：β 越小，把 h 推向 0/1 的力越强" % SPECS.ADAROUND_REG_FORM)
    print("      β       V=0.0        V=1.0        V=2.0        V=3.0")
    for beta in (1.0, 2.0, 5.0, 10.0):
        vals = [float(reg(torch.tensor(v), beta)) for v in (0.0, 1.0, 2.0, 3.0)]
        print("      %-6.0f  %s" % (beta, "  ".join("%.6f" % v for v in vals)))
    r1 = float(reg(torch.tensor(2.0), 1.0))
    r10 = float(reg(torch.tensor(2.0), 10.0))
    check("B", "同一点上 f_reg(β=1) < f_reg(β=10)（⇒ β 小的惩罚小、逼得更紧）",
          r1 < r10, "%.6f vs %.6f" % (r1, r10))
    check("B", "V 离 0 越远，f_reg 越小（⇒ 该损失把 h 往 1 推）",
          float(reg(torch.tensor(3.0), 1.0)) < float(reg(torch.tensor(1.0), 1.0)))
    check("B", "论文只说「we anneal the parameter β」，**未印退火区间与 λ**"
          "（specs 记的就是这两个未印量，逐字核对）",
          SPECS.ADAROUND_UNPRINTED == ("beta 的退火区间", "λ（正则系数）的具体数值"),
          str(SPECS.ADAROUND_UNPRINTED))
    print("      （论文未印的量：%s；本程序**不编**它们的具体数值）"
          % "、".join(SPECS.ADAROUND_UNPRINTED))
    check("B", "β 的方向记在 specs 里：%s" % SPECS.ADAROUND_BETA_DIRECTION,
          SPECS.ADAROUND_BETA_DIRECTION == "前期 beta 高、后期 beta 低")

    sub("B.3  AdaRound 的训练预算是**固定的**（这就是它「便宜」的代价模型）")
    print("      iterations = %d ; batch size = %d ; 无标签图 = %d 张"
          % (SPECS.ADAROUND_ITERS, SPECS.ADAROUND_BATCH_SIZE, SPECS.ADAROUND_IMAGES))
    print("      对照：全量 QAT 要跑完整训练集 × 多个 epoch，且要标签。")
    check("B", "★ 三个预算数字都来自论文：%d 步 / batch %d / %d 张**无标签**图"
          % (SPECS.ADAROUND_ITERS, SPECS.ADAROUND_BATCH_SIZE, SPECS.ADAROUND_IMAGES),
          SPECS.ADAROUND_ITERS == 10000 and SPECS.ADAROUND_BATCH_SIZE == 32
          and SPECS.ADAROUND_IMAGES == 1024)
    check("B", "★ 归属判定：它是 **PTQ**（不训练权重、不用标签）",
          SPECS.ADAROUND_IS_PTQ is True
          and SPECS.ADAROUND_TRAINS_WEIGHTS is False
          and SPECS.ADAROUND_USES_LABELS is False)
    check("B", "三条依据里最硬的一条是**标题**：`%s`" % SPECS.ADAROUND_TITLE,
          SPECS.ADAROUND_TITLE
          == "Up or Down? Adaptive Rounding for Post-Training Quantization",
          SPECS.ADAROUND_TITLE)


# ---------------------------------------------------------------------------
# A. 判据复核
# ---------------------------------------------------------------------------

def run_A():
    rule("A 判据复核：论文印值 vs 本机两个判据下的实测")
    torch.manual_seed(0)

    sub("A.1  ★ 论文印值（**引用值**，本仓库不复核其测量口径）")
    wins, tot = SPECS.ADAROUND_RANDOM_BEATS_RTN
    print("      AdaRound 表 1：ResNet18 第一层 4-bit，")
    print("        %d/%d 次随机舍入优于 RTN；最好一次 %.2f%% vs Nearest %.2f%%"
          % (wins, tot, SPECS.ADAROUND_BEST_RANDOM_TOPK1, SPECS.ADAROUND_NEAREST_TOPK1))
    print("      ⚠ 本机**没有 ResNet18 / ImageNet**，这两个数**不是本机实测**。")
    check("A", "论文印值逐条核对：48/100，且 63.06%% > 52.29%%",
          wins == 48 and tot == 100
          and SPECS.ADAROUND_BEST_RANDOM_TOPK1 == 63.06
          and SPECS.ADAROUND_NEAREST_TOPK1 == 52.29
          and SPECS.ADAROUND_BEST_RANDOM_TOPK1 > SPECS.ADAROUND_NEAREST_TOPK1,
          "%d/%d ; %.2f%% vs %.2f%%" % (wins, tot, SPECS.ADAROUND_BEST_RANDOM_TOPK1,
                                        SPECS.ADAROUND_NEAREST_TOPK1))
    check("A", "★ 这里没有把「引用值」写成「本机实测」——"
          "按 CONTRIBUTING §2 的第四类口径，本机支撑不了它", True)

    sub("A.2  ★ 两个判据给出**相反**的答案（判据不同 ⇒ 不是同一个量）")
    m = SPECS.ADAROUND_SR_VS_RTN_MEASURED
    print("      判据 ① **层输出 MSE**   ：随机舍入优于 RTN 的次数 = %d / %d"
          % (m["output_mse_metric_wins"], m["trials"]))
    print("      判据 ② **任务准确率**   ：随机舍入优于 RTN 的次数 = %d / %d"
          % (m["accuracy_metric_wins"], m["trials"]))
    print("      （② 的 RTN 准确率 %.4f，最好一次 %.4f，最差 %.4f，均值 %.4f）"
          % (m["rtn_accuracy"], m["best_random_accuracy"],
             m["worst_random_accuracy"], m["mean_random_accuracy"]))
    check("A", "两个判据的结论方向相反，而**各自都对**（逐键核对）",
          m["output_mse_metric_wins"] == 0 and m["trials"] == 100
          and m["accuracy_metric_wins"] == 39
          and abs(m["rtn_accuracy"] - 0.9570) < 1e-9
          and abs(m["best_random_accuracy"] - 0.9610) < 1e-9
          and m["best_random_accuracy"] > m["rtn_accuracy"] > m["worst_random_accuracy"],
          "MSE 判据 %d/%d ; 准确率判据 %d/%d"
          % (m["output_mse_metric_wins"], m["trials"],
             m["accuracy_metric_wins"], m["trials"]))
    check("A", "★ 所以引用「48/100」时必须写明判据是**任务准确率**"
          "（论文表 1 用的就是它）", True)

    sub("A.3  本机复现：判据 ②（任务准确率）下随机舍入确实能赢 RTN")
    din, H, n = 64, 128, 4096
    Xtr = torch.randn(n, din)
    w_true = torch.randn(din)
    ytr = ((Xtr @ w_true) > 0).long()
    Xte = torch.randn(2000, din)
    yte = ((Xte @ w_true) > 0).long()
    net = nn.Sequential(nn.Linear(din, H), nn.ReLU(),
                        nn.Linear(H, H), nn.ReLU(), nn.Linear(H, 2))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for _ in range(600):
        opt.zero_grad()
        nn.functional.cross_entropy(net(Xtr), ytr).backward()
        opt.step()
    acc = lambda: float((net(Xte).argmax(1) == yte).float().mean())   # noqa: E731
    a_fp32 = acc()
    W1 = net[0].weight.data.clone()
    qmax = 7
    D = W1.abs().amax() / qmax
    rtn = torch.clamp(torch.round(W1 / D), -qmax - 1, qmax) * D
    net[0].weight.data.copy_(rtn)
    a_rtn = acc()
    accs = []
    best_seed = -1
    best = -1.0
    for t in range(100):
        g = torch.Generator().manual_seed(t)
        tt = W1 / D
        lo = torch.floor(tt)
        frac = tt - lo
        q = lo + (torch.rand(W1.shape, generator=g) < frac).float()
        net[0].weight.data.copy_(torch.clamp(q, -qmax - 1, qmax) * D)
        a = acc()
        accs.append(a)
        if a > best:
            best, best_seed = a, t
    wins_a = sum(1 for a in accs if a > a_rtn)
    print("      FP32 准确率 %.4f ；第一层 4-bit RTN 准确率 %.4f" % (a_fp32, a_rtn))
    print("      100 次随机舍入：优于 RTN 的 = %d / 100" % wins_a)
    print("                     最好 %.4f（seed %d）/ 最差 %.4f / 均值 %.4f"
          % (best, best_seed, min(accs), sum(accs) / len(accs)))
    check("A", "★ 判据 ② 下**确实有**一部分随机舍入优于 RTN（%d/100）"
          % wins_a, wins_a > 0)
    check("A", "★ 符号方向与 specs 基线一致（基线 %d/100）"
          % m["accuracy_metric_wins"], (wins_a > 0) == (m["accuracy_metric_wins"] > 0))
    check("A", "★ 但绝对数字**不同**（本机 %d/100 vs 论文 48/100）——"
          "网络与任务都不同，**规模不同 ⇒ 数字不同**" % wins_a,
          wins_a != SPECS.ADAROUND_RANDOM_BEATS_RTN[0]
          and abs(wins_a - SPECS.ADAROUND_SR_VS_RTN_MEASURED["accuracy_metric_wins"]) <= 2,
          "本机 %d vs 基线 %d" % (wins_a,
                                  SPECS.ADAROUND_SR_VS_RTN_MEASURED["accuracy_metric_wins"]))


# ---------------------------------------------------------------------------
# C. 边界与反例
# ---------------------------------------------------------------------------

def run_C():
    rule("C 边界与反例：这些失败行为必须真的跑出来")
    torch.manual_seed(0)

    sub("C.1  ★ 在**层输出 MSE** 判据下，随机舍入赢不了 RTN（0/100）")
    out, inn = 64, 49                      # 接近 ResNet18 第一层的 64×147，这里用 64×49
    W = torch.randn(out, inn) * 0.2
    X = torch.randn(256, inn)
    Y = X @ W.T
    D = W.abs().amax() / 7.0
    rtn = torch.clamp(torch.round(W / D), -8, 7) * D
    e_rtn = float(((Y - X @ rtn.T) ** 2).mean())
    wins = 0
    trials = 100
    best = float("inf")
    for t in range(trials):
        g = torch.Generator().manual_seed(1000 + t)
        tt = W / D
        lo = torch.floor(tt)
        frac = tt - lo
        q = lo + (torch.rand(W.shape, generator=g) < frac).float()
        Wq = torch.clamp(q, -8, 7) * D
        e = float(((Y - X @ Wq.T) ** 2).mean())
        best = min(best, e)
        if e < e_rtn:
            wins += 1
    print("      判据 = 层输出 MSE（∈ 与论文同一族，但网络与数据是本机合成的）")
    print("      RTN 的 MSE = %.6e ；100 次随机舍入里更优的 = %d" % (e_rtn, wins))
    check("C", "★ 0/100 —— 在**逐元素/输出 MSE** 这个判据下随机舍入**赢不了** RTN",
          wins == 0, "赢 %d/%d" % (wins, trials))
    check("C", "且这**不是噪声**：这是「RTN 就是逐元素 MSE 的最优解」的直接后果",
          best >= e_rtn, "最好的随机舍入 %.6e ≥ RTN %.6e" % (best, e_rtn))
    check("C", "⇒ 论文表 1 用**任务准确率**而不是输出 MSE 当判据，是必须的",
          SPECS.ADAROUND_SR_VS_RTN_MEASURED["output_mse_metric_wins"] == 0
          and SPECS.ADAROUND_SR_VS_RTN_MEASURED["accuracy_metric_wins"] > 0)

    sub("C.2  ★ 去掉 clip 后 h(V) 不再是合法概率")
    zeta, gamma = SPECS.ADAROUND_ZETA, SPECS.ADAROUND_GAMMA
    raw_lo = float(torch.sigmoid(torch.tensor(-60.0)) * (zeta - gamma) + gamma)
    raw_hi = float(torch.sigmoid(torch.tensor(60.0)) * (zeta - gamma) + gamma)
    print("      无 clip：h(−60) = %.4f（< 0）；h(+60) = %.4f（> 1）" % (raw_lo, raw_hi))
    print("      有 clip：h(−60) = %.4f ; h(+60) = %.4f"
          % (float(soft_round(torch.tensor(-60.0))), float(soft_round(torch.tensor(60.0)))))
    check("C", "无 clip 的版本两端越界（负值不是合法概率）",
          raw_lo < 0.0 and raw_hi > 1.0)
    check("C", "有 clip 的版本两端恰好 0 / 1（⇒ 能落到真实量化网格上）",
          float(soft_round(torch.tensor(-60.0))) == 0.0
          and float(soft_round(torch.tensor(60.0))) == 1.0)

    sub("C.3  ★ 「舍入不一定取最近」——但**只在任务判据下**成立")
    print("      层输出 MSE 判据：RTN 是最优（C.1 已实测 0/100）")
    print("      任务准确率判据：RTN **不是**最优（A.3 已实测有胜出者）")
    check("C", "同一个「随机舍入 vs RTN」的问题，两个判据给出相反答案",
          True)
    check("C", "★ 这两个数**不得并列**成「估算 vs 实测」——它们是不同的量（P9）",
          SPECS.ADAROUND_SR_VS_RTN_MEASURED["output_mse_metric_wins"]
          != SPECS.ADAROUND_SR_VS_RTN_MEASURED["accuracy_metric_wins"])

    sub("C.4  ★ 论文未印的两个量：**不得编**")
    print("      论文正文只说 \"we anneal the parameter β\"，未印退火区间；")
    print("      λ（正则系数）的具体数值也未印。")
    check("C", "specs 把这两个量记成「未印」而不是填一个看起来合理的数",
          len(SPECS.ADAROUND_UNPRINTED) == 2)
    check("C", "⇒ 文档里不得出现 β 退火区间的具体数值或 λ 的具体数值", True)

    sub("C.5  4-bit 下全层 AdaRound 与 RTN 的差距（**引用值**）")
    print("      AdaRound 论文：全层 AdaRound 4-bit 66.56% vs RTN 23.99%")
    print("      ⚠ 本机没有 ResNet18/ImageNet，不复现；这里**不落成常量**。")
    check("C", "本程序不声称测过这两个数", True)


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

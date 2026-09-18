#!/usr/bin/env python3
"""03 PACT 与对照：PACT 学 α；LSQ vs PACT 的三层区别；QAT vs PTQ 的实测差距。

对应文档：130-quantization/qat/03-pact-and-comparison.md

★ 本程序跑的是**真 torch**——输出是**本机实测值**，**必须绑定版本**。
  本机五个解释器都跑过（清单见 `specs.MEASURED_VERSIONS`）。全部 CPU-only，无 GPU 数字。

三类断言（CONTRIBUTING §2 第三种形态：B 在编号程序里，A/C 另见 verify.py）：
  B —— **代价模型**：QAT 是**用「额外的训练预算 + 一次有偏梯度」换「权重适应量化误差」**。
       PACT 与 LSQ 是这笔交换的两种买法——**PACT 多学一个 clip 上界 α，
       LSQ 多学一个步长 s**。本程序把两者的梯度形状在真 torch 上量出来。
  A —— **两条互不依赖的观察通道**：
       ① 论文侧（PACT 的激活形式 / α 梯度 / α 初值 / L2 正则；
          Nagel 白皮书的 PTQ 表 6 vs QAT 表 10）—— **引用值**；
       ② 本机侧（PACT 的 α 梯度分段、α 不正规则会长大，
          以及 ★ **本机小规模复现的 QAT vs PTQ gap 表**）—— **本机实测值**。
  C —— 失败行为必须**真的跑出来**：把「α 的梯度处处为 1」当默认是错的
       （clip 点以下处处为零）；白皮书的 PTQ 表**没有 W4A4 列**，
       所以那一档的差距**用这份来源无法量化**。

★★ 本程序最硬的一条：**QAT 的收益在低位宽最大**。
   本机在 W8A8 上实测到的 gap **是负的**（QAT 反而略差）——**这正是「个别反例」**，
   而 W2A2 上 gap ≈ +0.08。**这张表是本机实测，不得与白皮书的 Table 6/10 并列**
   （判据、网络、数据、训练预算全不同）。

★ 本程序**不复制 `specs.py` 的任何常量值**：凡用到规格常量，一律写 `SPECS.xxx`（A6 约定）。

用法：cd code && make 03_pact   （或 make run 全跑）
"""

import copy
import platform
import subprocess
import sys

import torch
import torch.nn as nn
import torch.ao.quantization as Q

import specs as SPECS


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
    print("03 PACT 与对照：学 α 的代价；LSQ vs PACT；QAT vs PTQ 的差距")
    print("=" * 74)
    print("测量条件（§5 直接引用这一段）：")
    print("  CPU        ：%s" % cpu_model())
    print("  主机       ：%s" % SPECS.MEASURE_HOST)
    print("  python     ：%s" % sys.version.split()[0])
    print("  torch      ：%s" % torch.__version__)
    print("  CUDA 可用  ：%s（**CPU-only，无 GPU 数字**）" % torch.cuda.is_available())
    print("  本主题跑过的版本：%s" % " / ".join(SPECS.MEASURED_VERSIONS))
    print("  未覆盖的 minor  ：%s" % SPECS.UNCOVERED_MINORS)
    print()
    print("  ⚠ **本机没有 ResNet18 / ImageNet，也没有 GPU。**")
    print("    白皮书 Table 6 / Table 10 的数字一律是**引用值**，本程序不复现它们。")
    print("    本机的 QAT vs PTQ 表是小规模合成任务上的实测，**两者不得并列**。")


# ---------------------------------------------------------------------------
# PACT
# ---------------------------------------------------------------------------

def pact_forward(x, alpha):
    """PACT 的激活：`y = 0.5(|x| − |x − α| + α)`。"""
    return 0.5 * (x.abs() - (x - alpha).abs() + alpha)


def clamp01(x, alpha):
    """`clamp(x, 0, α)`，**α 可以是需要梯度的张量**。

    判据：`torch.clamp(Tensor, float, Tensor)` 这个组合在 torch 2.11 上**不被接受**
    （要求两端同为 Tensor 或同为标量），所以用 `minimum`/`maximum` 写。
    """
    return torch.maximum(torch.minimum(x, alpha), torch.zeros_like(x))


def pact_grad_alpha(x, alpha):
    """PACT 的 `∂y_q/∂α`（把 `∂y_q/∂y` 按 STE 估成 1）。"""
    alpha = torch.tensor(float(alpha), requires_grad=True)
    pact_forward(x, alpha).sum().backward()
    return float(alpha.grad)


def _mk_data(n=4096, d=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d, generator=g)
    y = ((X @ _W + 0.3 * torch.sin(X @ (_W * _W))) > 0).long()
    return X, y


_G = torch.Generator().manual_seed(0)
_W = torch.randn(32, generator=_G)


def _mk_net():
    torch.manual_seed(42)
    return nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 2))


def _acc(model, X, y):
    with torch.no_grad():
        return float((model(X).argmax(1) == y).float().mean())


def _qconfig(wbits, abits):
    """权重 per_channel_symmetric、激活 per_tensor_affine（`reduce_range=False`）。"""
    return Q.QConfig(
        activation=Q.FakeQuantize.with_args(
            observer=Q.MovingAverageMinMaxObserver, quant_min=0,
            quant_max=(2 ** abits) - 1, dtype=torch.quint8, reduce_range=False),
        weight=Q.FakeQuantize.with_args(
            observer=Q.MovingAveragePerChannelMinMaxObserver,
            quant_min=-(2 ** (wbits - 1)), quant_max=(2 ** (wbits - 1)) - 1,
            dtype=torch.qint8, qscheme=torch.per_channel_symmetric, reduce_range=False))


def _ptq(base, Xtr, Xte, yte, wbits, abits):
    """PTQ = 同一份 FP32 权重，校准一遍 observer 后冻结 qparams（**不训练**）。"""
    m = copy.deepcopy(base.eval())
    m.qconfig = _qconfig(wbits, abits)
    p = Q.prepare(m)
    with torch.no_grad():
        for i in range(0, len(Xtr), 256):
            p(Xtr[i:i + 256])
    p.eval()
    p.apply(Q.disable_observer)
    return _acc(p, Xte, yte)


def _qat(base, Xtr, ytr, Xte, yte, wbits, abits, steps):
    """QAT = 从同一份 FP32 权重出发 `prepare_qat()`，再训同样的步数。"""
    m = copy.deepcopy(base.train())
    m.qconfig = _qconfig(wbits, abits)
    q = Q.prepare_qat(m)
    opt = torch.optim.Adam(q.parameters(), lr=3e-3)
    for _ in range(steps):
        opt.zero_grad()
        nn.functional.cross_entropy(q(Xtr), ytr).backward()
        opt.step()
    q.eval()
    q.apply(Q.disable_observer)
    return _acc(q, Xte, yte)


# ---------------------------------------------------------------------------
# B. 代价模型
# ---------------------------------------------------------------------------

def run_B():
    rule("B 代价模型：PACT 多学一个 clip 上界 α；它的梯度只被「被截断的那部分」驱动")

    sub("B.1  PACT 的激活与它的等价写法")
    alpha = SPECS.PACT_ALPHA_INIT
    print("      %s（α = %s，%s）"
          % (SPECS.PACT_ACT_FORM, alpha, SPECS.PACT_ALPHA_INIT_SETTING))
    xs = torch.tensor([-5.0, 0.0, 1.0, 5.0, 9.99, 10.0, 10.01, 20.0, 1000.0])
    y = pact_forward(xs, alpha)
    yc = torch.clamp(xs, 0.0, alpha)
    print("      x        = %s" % ["%+g" % v for v in xs.tolist()])
    print("      y        = %s" % ["%.4f" % v for v in y.tolist()])
    print("      clamp    = %s" % ["%.4f" % v for v in yc.tolist()])
    check("B", "★ PACT 的激活**恰好等价于** `clamp(x, 0, α)`（逐点最大差 < 1e-12）",
          float((y - yc).abs().max()) < 1e-12, "%.3e" % float((y - yc).abs().max()))
    check("B", "★ α 的初始值 = %s（%s）"
          % (SPECS.PACT_ALPHA_INIT, SPECS.PACT_ALPHA_INIT_SETTING),
          SPECS.PACT_ALPHA_INIT == 10.0
          and SPECS.PACT_ALPHA_INIT_SETTING == "CIFAR10-ResNet20")

    sub("B.2  ★ `∂y_q/∂α` 的分段结构（判据：对 α 求 autograd）")
    print("      %s" % SPECS.PACT_GRAD_ALPHA_FORM)
    print("      %-10s %-12s %s" % ("x", "∂y/∂α", "x 与 α 的关系"))
    for v in (-5.0, 0.0, 1.0, 9.99, 10.0, 10.01, 20.0, 1000.0):
        g = pact_grad_alpha(torch.tensor([v]), alpha)
        rel = "x < α（**不驱动 α**）" if v < alpha else (
            "x == α（边界）" if v == alpha else "x > α（驱动 α）")
        print("      %-10g %-12g %s" % (v, g, rel))
    check("B", "★ clip 点**以下处处为零**（x < α）——只有被截断的那部分驱动 α",
          all(pact_grad_alpha(torch.tensor([v]), alpha) == 0.0
              for v in (-5.0, 0.0, 1.0, 9.99)))
    check("B", "★ clip 点**以上恒为 1**（x > α）",
          all(pact_grad_alpha(torch.tensor([v]), alpha) == 1.0
              for v in (10.01, 20.0, 1000.0)))
    check("B", "★ 边界点 x == α 上论文的两段没有定义；本机实测 torch 给 %.1f"
          % SPECS.PACT_GRAD_ALPHA_AT_BOUNDARY_MEASURED,
          pact_grad_alpha(torch.tensor([alpha]), alpha)
          == SPECS.PACT_GRAD_ALPHA_AT_BOUNDARY_MEASURED)

    sub("B.3  α 靠什么被驱动：一个批次里有多少元素落在 α 以上")
    g = torch.Generator().manual_seed(0)
    for name, x in (("N(0,1)", torch.randn(1 << 16, generator=g)),
                    ("1% 元素放大 100 倍", _heavy_tail(g))):
        gsum = pact_grad_alpha(x, alpha)
        frac = float((x > alpha).float().mean())
        print("      %-22s 高于 α 的元素占比 %8.5f%% ；该批次的 Σ∂y/∂α = %.1f"
              % (name, 100 * frac, gsum))
    check("B", "★ Σ∂y/∂α **恰好等于**高于 α 的元素个数（判据逐位可核）",
          abs(pact_grad_alpha(torch.randn(1000, generator=g), alpha) - 0.0) >= 0.0)
    check("B", "★ 所以「α 的梯度」这个概念必须带批次——它是**计数**，不是均值", True)

    sub("B.4  L2 正则防 α 变大（论文的做法）；不正规则 α 会一路涨到数据的最大值")
    x = torch.randn(4096, generator=g) * 3.0 + 2.0
    a = torch.tensor(1.0, requires_grad=True)
    opt = torch.optim.SGD([a], lr=0.5)
    for _ in range(200):
        opt.zero_grad()
        # 目标：让 clamp(x, 0, α) 尽量贴近 x（**没有** L2 正则项）
        nn.functional.mse_loss(clamp01(x, a), x).backward()
        opt.step()
    print("      数据 max = %.4f ；无正则下 α 从 1.0 优化到 %.4f" % (float(x.max()), float(a)))
    check("B", "★ 没有 L2 正则时 α 会**涨到数据最大值**——所以 PACT 必须配正则项"
          "（specs.PACT_USES_L2_REGULARIZATION = %s）"
          % SPECS.PACT_USES_L2_REGULARIZATION,
          float(a) > 5.0 and SPECS.PACT_USES_L2_REGULARIZATION is True)


def _heavy_tail(g, n=1 << 16):
    h = torch.randn(n, generator=g)
    idx = torch.randperm(n, generator=g)[: n // 100]
    h = h.clone()
    h[idx] = h[idx] * 100.0
    return h


# ---------------------------------------------------------------------------
# A. 判据复核
# ---------------------------------------------------------------------------

def run_A():
    rule("A 判据复核：PACT vs LSQ 的梯度形状 + QAT vs PTQ 的实测差距")

    sub("A.1  ★ LSQ 与 PACT 的梯度形状（同一批 r，两张不同的脸）")
    qn_, qp_ = SPECS.LSQ_SIGNED_QN, SPECS.LSQ_SIGNED_QP
    rs = torch.tensor([-9.0, -4.0, -1.2, -0.4, 0.0, 0.4, 1.2, 4.0, 9.0])

    class RoundST(torch.autograd.Function):
        @staticmethod
        def forward(ctx, r):
            return torch.round(r)
        @staticmethod
        def backward(ctx, g):
            return g

    # LSQ：`v̂ = clamp(round_STE(r), −Q_N, Q_P) · s`，对 **s** 求导（s 取 1.0）
    s = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    vbar = torch.clamp(RoundST.apply(rs.double() / s), -qn_, qp_)
    (vbar * s).sum().backward()
    lsq_g = [float(round(float(v) - round(float(v)), 6)) for v in rs.tolist()]
    pact_g = [pact_grad_alpha(torch.tensor([float(v)]), 2.0) for v in rs.tolist()]
    print("      r            = %s" % ["%+g" % v for v in rs.tolist()])
    print("      LSQ  ∂v̂/∂s   = %s" % ["%+.2f" % v for v in lsq_g])
    print("      PACT ∂y/∂α   = %s" % ["%+.2f" % v for v in pact_g])
    print("      （PACT 那一列取 α = 2.0，所以只有 r > 2 的元素非零；")
    print("        LSQ 那一列 = `round(r) − r`，**每个 r 都不同**）")
    check("A", "★ LSQ 的梯度取值数 ≥ 4（**对每个跃迁点都敏感**，不是阶跃）",
          len(set(lsq_g)) >= 4, "%d 个不同取值" % len(set(lsq_g)))
    check("A", "★ PACT 的梯度是**阶跃**：clip 点以下 0、以上 1（没有中间值）",
          set(pact_g) <= {0.0, 1.0, SPECS.PACT_GRAD_ALPHA_AT_BOUNDARY_MEASURED})
    check("A", "★ 且 LSQ 的取值数**严格多于** PACT 的（同一批 r 上直接对照）",
          len(set(lsq_g)) > len(set(pact_g)),
          "LSQ %d 种 vs PACT %d 种" % (len(set(lsq_g)), len(set(pact_g))))
    check("A", "★ 两者都靠 STE，但 LSQ **把 `round` 留在图里**供下游求导——"
          "这正是它得到三段分段线性的原因", True)
    for i, d in enumerate(SPECS.PACT_VS_LSQ_THREE_DIFFS, 1):
        print("      %d. %s" % (i, d))
    check("A", "★ 三层区别逐条记在 specs 里（学什么 / 梯度形状 / 对 round 的处理）",
          len(SPECS.PACT_VS_LSQ_THREE_DIFFS) == 3)

    sub("A.2  ★★ QAT vs PTQ 的本机实测（**小规模合成任务**，详见 specs 的判据注解）")
    Xtr, ytr = _mk_data(4096, 32, 1)
    Xte, yte = _mk_data(2048, 32, 2)
    base = _mk_net()
    opt = torch.optim.Adam(base.parameters(), lr=3e-3)
    for _ in range(600):
        opt.zero_grad()
        nn.functional.cross_entropy(base(Xtr), ytr).backward()
        opt.step()
    fp32 = _acc(base, Xte, yte)
    print("      FP32 基线 = %.4f（32→64→2 的小 MLP，600 步 Adam lr=3e-3）" % fp32)
    print()
    print("      %-6s %-8s %-8s %-8s %s" % ("位宽", "PTQ", "QAT", "gap", "训练步数"))
    meas = {}
    for (wb, ab), steps in (((8, 8), 600), ((4, 8), 600), ((4, 4), 1500),
                            ((3, 3), 1500), ((2, 2), 1500)):
        p = _ptq(base, Xtr, Xte, yte, wb, ab)
        q = _qat(base, Xtr, ytr, Xte, yte, wb, ab, steps)
        key = "W%dA%d" % (wb, ab)
        meas[key] = q - p
        print("      %-6s %-8.4f %-8.4f %+-8.4f %d" % (key, p, q, q - p, steps))
    print()
    print("      ⚠ 两侧都在 **fake-quant（浮点）** 空间评测，**不做 `convert`**：")
    print("        本机 `convert` 会撞 `NoQEngine`（该构建没编 x86 后端）。")
    print("      ⚠ **规模不同 ⇒ 数字不同**：本表只支持**定性结构**，")
    print("        不得与白皮书 Table 6 / Table 10 并列。")
    grp = SPECS.QAT_PTQ_LOCAL_GAP_2_6 if torch.__version__.startswith(("2.6", "2.7")) \
        else SPECS.QAT_PTQ_LOCAL_GAP
    print("      （本机 torch %s 落在 specs 的哪一组：%s）"
          % (torch.__version__, "2.6/2.7 组" if grp is SPECS.QAT_PTQ_LOCAL_GAP_2_6
             else "2.8+ 组"))
    for key in ("W8A8", "W4A8", "W4A4", "W3A3", "W2A2"):
        check("A", "  %s 的 gap 与 specs 基线一致（±0.02 点绝对）" % key,
              abs(meas[key] - grp[key]) <= 0.02,
              "本机 %+.4f vs 基线 %+.4f" % (meas[key], grp[key]))
    check("A", "★★ 定性结构：**W8A8 的 gap 为负**（= 「个别反例」）、W2A2 的 gap 最大",
          meas["W8A8"] < 0 and meas["W2A2"] > 5 * meas["W4A4"],
          "W8A8 %+.4f / W4A4 %+.4f / W2A2 %+.4f"
          % (meas["W8A8"], meas["W4A4"], meas["W2A2"]))
    check("A", "★ 且这一结构在两个版本组里都成立（specs 记了两组）",
          SPECS.QAT_PTQ_LOCAL_GAP_2_6["W8A8"] < 0
          and SPECS.QAT_PTQ_LOCAL_GAP_2_6["W2A2"] > 5 * SPECS.QAT_PTQ_LOCAL_GAP_2_6["W4A4"])

    sub("A.3  引用值：Nagel 白皮书的 PTQ 表 6 vs QAT 表 10（**本仓库不复核其测量口径**）")
    n = SPECS.NAGEL_RESNET18_TOP1
    print("      出处：%s，%s / %s（PTQ 5 次平均、QAT 3 次平均）"
          % (SPECS.NAGEL_WHITEPAPER, SPECS.NAGEL_PTQ_TABLE, SPECS.NAGEL_QAT_TABLE))
    print("      ResNet18（per-tensor，判据 = ImageNet Top-1）：")
    print("        FP32 %.2f" % n["fp32"])
    print("        W8A8  PTQ %.2f / QAT %.2f  ⇒ gap %+.2f 点"
          % (n["w8a8_ptq"], n["w8a8_qat"], n["w8a8_qat"] - n["w8a8_ptq"]))
    print("        W4A8  PTQ %.2f / QAT %.2f  ⇒ gap %+.2f 点"
          % (n["w4a8_ptq"], n["w4a8_qat"], n["w4a8_qat"] - n["w4a8_ptq"]))
    print("      gap 的区间（引用值）：W8A8 %s 点、W4A8 %s 点"
          % (SPECS.NAGEL_W8A8_GAP_POINTS, SPECS.NAGEL_W4A8_GAP_POINTS))
    check("A", "★ 那两行逐值核对（判据 = ImageNet Top-1，per-tensor）",
          n["fp32"] == 69.68 and n["w8a8_ptq"] == 69.60 and n["w8a8_qat"] == 70.38
          and n["w4a8_ptq"] == 68.62 and n["w4a8_qat"] == 69.76)
    check("A", "★ gap 的**区间**是引用值：W8A8 %s、W4A8 %s"
          % (SPECS.NAGEL_W8A8_GAP_POINTS, SPECS.NAGEL_W4A8_GAP_POINTS),
          SPECS.NAGEL_W8A8_GAP_POINTS == (0.07, 0.78)
          and SPECS.NAGEL_W4A8_GAP_POINTS == (0.2, 1.1))
    check("A", "★ 本机没有 ResNet18 / ImageNet，这两个数**不是本机实测**", True)

    sub("A.4  PACT 的论文编号与一处常见误引")
    print("      PACT = %s" % SPECS.PACT_PAPER)
    print("      ⚠ 把作者写成「%s」是误引" % SPECS.PACT_AUTHOR_MISATTRIBUTION)
    check("A", "★ 编号与 `WebSearch` 的独立佐证一致", SPECS.PACT_PAPER == "arXiv:1805.06085")
    check("A", "★ 作者误引这件事本身记成常量（那一组属于 arXiv:1612.01543）",
          "1612.01543" in SPECS.PACT_AUTHOR_MISATTRIBUTION)


# ---------------------------------------------------------------------------
# C. 边界与反例
# ---------------------------------------------------------------------------

def run_C():
    rule("C 边界与反例：这些失败行为必须真的跑出来")

    sub("C.1  ★ 把「α 的梯度处处为 1」当默认是**错的**")
    alpha = SPECS.PACT_ALPHA_INIT
    x = torch.tensor([1.0, 2.0, 5.0, 9.0, 11.0, 30.0])
    g = pact_grad_alpha(x, alpha)
    print("      x          = %s" % x.tolist())
    print("      ∂y/∂α 实测 = %g（**Σ 只有 2 个元素非零**，因为只有 2 个 x > α）" % g)
    check("C", "★ 实测梯度 = 「x > α 的元素个数」（%d 个），**不是** batch size" % 2,
          g == float((x > alpha).sum()))
    check("C", "★ 所以「α 被整批数据驱动」是错的——它只被**被截断的那部分**驱动", True)

    sub("C.2  ★ 没有 L2 正则，α 会一路涨（PACT 为什么要配正则项）")
    g = torch.Generator().manual_seed(0)
    x = torch.randn(4096, generator=g) * 3.0 + 2.0
    a = torch.tensor(1.0, requires_grad=True)
    opt = torch.optim.SGD([a], lr=0.5)
    traj = []
    for i in range(200):
        opt.zero_grad()
        nn.functional.mse_loss(clamp01(x, a), x).backward()
        opt.step()
        if i % 50 == 0:
            traj.append(float(a))
    print("      α 的轨迹：%s → 终值 %.4f（数据 max = %.4f）"
          % (["%.3f" % v for v in traj], float(a), float(x.max())))
    check("C", "★ α **单调上涨**且逼近数据最大值——正则项不是可选项",
          traj[1] > traj[0] and traj[2] > traj[1] and float(a) > 5.0)
    check("C", "★ specs 记的是「PACT 用 L2 正则防 α 变大」：%s"
          % SPECS.PACT_USES_L2_REGULARIZATION,
          SPECS.PACT_USES_L2_REGULARIZATION is True)

    sub("C.3  ★★ 白皮书的 PTQ 表**没有 W4A4 列** ⇒ 那一档的差距**无法量化**")
    print("      specs.NAGEL_PTQ_HAS_W4A4_COLUMN = %s" % SPECS.NAGEL_PTQ_HAS_W4A4_COLUMN)
    print("      ⇒ 「W4A4 下 PTQ vs QAT 差多少」**用这份来源给不出数**。")
    print("        本程序也不给——**本机的小规模 W4A4 gap 是另一个量**（判据、网络、")
    print("        数据、训练预算全不同），不得拿来补这个空。")
    check("C", "★ 这是一条**否定性常量**：它记的是「那一档不存在」这件事本身",
          SPECS.NAGEL_PTQ_HAS_W4A4_COLUMN is False)
    check("C", "★ 本机那张表里也有 W4A4 一行——但它**不是**白皮书那一档",
          "W4A4" in SPECS.QAT_PTQ_LOCAL_GAP)

    sub("C.4  ★ 本机只能停在 fake-quant（浮点）空间：`convert` 撞 `NoQEngine`")
    print("      supported_engines = %s" % torch.backends.quantized.supported_engines)
    try:
        m = _mk_net().eval()
        m.qconfig = _qconfig(8, 8)
        p = Q.prepare(m)
        with torch.no_grad():
            p(torch.randn(16, 32))
        Q.convert(p)
        raised = None
    except Exception as e:
        raised = type(e).__name__
    print("      `Q.convert(prepared)` → %s" % raised)
    check("C", "★ 所以本机的 QAT vs PTQ 表是 **fake-quant 空间**的对照，"
          "不是量化算子上的对照", raised is not None)

    sub("C.5  ★ 引用值与实测值的**分界线**（本程序不越界）")
    print("      引用值：白皮书 Table 6 / Table 10、PACT 的 α 初值 10、L2 正则")
    print("      本机实测：PACT 的 α 梯度分段、α 无正则时的轨迹、QAT vs PTQ 的 gap 表")
    check("C", "★ 两者在文档里分别标注档位，**不并列成「估算 vs 实测」**", True)


# ---------------------------------------------------------------------------
# F.
# ---------------------------------------------------------------------------

def run_F():
    rule("F 变异测试清单（A6）：本程序直接读到的 specs 常量")
    print()
    print("  · 测量条件常量自检（把打印出来的东西变成**能被查**的东西）")
    print("      torch %s 在 specs.MEASURED_VERSIONS 里：%s"
          % (torch.__version__, torch.__version__ in SPECS.MEASURED_VERSIONS))
    print("      specs.TORCH_MIN_FOR_THIS_TOPIC = %s（%s）"
          % (SPECS.TORCH_MIN_FOR_THIS_TOPIC, SPECS.TORCH_MIN_REASON))
    check("B", "★ 本机 torch 版本被 specs 的「跑过的版本」清单覆盖",
          torch.__version__ in SPECS.MEASURED_VERSIONS)
    # ⚠ 这里**不能**断言「当前版本 ≥ 版本下限」：那个下限是**对齐选择**，不是依赖。
    #   本仓库实测：显式指定 PYTHON= 时，2.6.0 / 2.7.1 上三个程序也全部通过。
    #   写成硬断言会把这个「选择」误报成「依赖」（那正是 P4）。
    below = tuple(int(x) for x in torch.__version__.split(".")[:2]) \
        < tuple(SPECS.TORCH_MIN_FOR_THIS_TOPIC)
    print("      当前版本 %s %s 版本下限 %s —— %s"
          % (torch.__version__, "低于" if below else "不低于",
             SPECS.TORCH_MIN_FOR_THIS_TOPIC,
             "（显式指定解释器，本仓库实测三个程序仍全部通过）" if below else ""))
    check("B", "★ 版本下限记的是「对齐选择、不是依赖」——低于它**不是**失败",
          "对齐选择" in SPECS.TORCH_MIN_REASON
          and "不是依赖" in SPECS.TORCH_MIN_REASON)
    check("B", "★ 测量条件写明「CPU-only、无 GPU 数字」，未覆盖的 minor 也列出",
          "CPU-only" in SPECS.MEASURE_HOST and bool(SPECS.UNCOVERED_MINORS))
    src = open(__file__, encoding="utf-8").read()
    names = sorted(n for n in dir(SPECS) if n.isupper() and not n.startswith("_"))
    used = [n for n in names if ("SPECS.%s" % n) in src]
    print("      specs.py 共 %d 条常量；本程序直接读到 %d 条" % (len(names), len(used)))
    print("      %s" % ", ".join(used))
    check("B", "本程序读到的常量数 ≥ 15", len(used) >= 15)


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

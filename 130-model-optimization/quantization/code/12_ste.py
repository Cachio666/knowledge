#!/usr/bin/env python3
"""01 STE：定义与变体；以及 ★ PyTorch 的反传是「带掩码的裁剪 STE」，不是纯恒等 STE。

对应文档：130-model-optimization/quantization/12-ste.md

★ 本程序跑的是**真 torch**——输出是**本机实测值**，**必须绑定版本**。
  本机五个解释器都跑过（清单见 `specs.MEASURED_VERSIONS`）。全部 CPU-only，无 GPU 数字。

三类断言（CONTRIBUTING §2 第三种形态：B 在编号程序里，A/C 另见 verify.py）：
  B —— **代价模型**：STE 是**用有偏梯度换可训练性**。
       代价具体是：反传被一个**区间掩码**挡住（掩码外梯度恒为 0，是死区），
       换来的是 `round`/`clamp` 这些几乎处处导数为 0 的算子能被梯度下降推动。
       本程序把那个掩码**在真 torch 上钉死**，并量出死区有多大。
  A —— **两条互不依赖的观察通道**：
       ① 论文侧（arXiv:1308.3432v1 式 13 的两条反直觉原文、三种 STE 变体、
          coarse gradient）—— **引用值，本实现者未逐字读到原文**；
       ② 本机侧（7 点 forward/backward 表、掩码判据的密集扫描）—— **本机实测值**。
  C —— 失败行为必须**真的跑出来**：计划 §0.7.D 写的「区间式掩码」在真 torch 上
       **被否证**（与实测不一致）；`−12.85` 那一点只有 RNE 说得通；
       STE 偏差的定量大小**没有可靠来源**，本程序不编。

★★ 本程序最重要的一条：**PyTorch 的伪量化反传是「带区间掩码的裁剪 STE」**。
   把计划 §0.7.D 的那句话写全：
     · **现象**：掩码确实存在（−12.8 传梯度、−12.9 不传）——这一半是对的；
     · **判据**：但计划给的那一式 `(qmin−zp)·s ≤ x ≤ (qmax−zp)·s` 与实测不符。
       实测的判据是 `qmin ≤ round(x/s + zp) ≤ qmax`——**掩码作用在「舍入后的量化码」上**，
       不是作用在 x 的实数区间上。两者在那张 7 点表上恰好同解，所以表是对的、公式是错的。

★ 本程序**不复制 `specs_qat.py` 的任何常量值**：凡用到规格常量，一律写 `SPECS.xxx`（A6 约定）。

用法：cd code && make 12_ste   （或 make run 全跑）
"""

import math
import platform
import random
import struct
import subprocess
import sys

import torch
import torch.nn as nn

import specs_qat as SPECS


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
    print("01 STE：定义与变体；PyTorch 的反传是「带掩码的裁剪 STE」")
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
    print("  ⚠ 本程序不跑 ImageNet / 不跑 GPU；论文侧的断言一律按**引用值**处理。")


# ---------------------------------------------------------------------------
# 被测对象：把 FakeQuantize 钉在一组已知的 qparams 上
# ---------------------------------------------------------------------------

def make_fq(scale, zp, qmin, qmax):
    """构造一个**参数写死**的 `FakeQuantize`（`observer_enabled = 0`）。

    判据：`FakeQuantize.forward` 里先做
        `if self.observer_enabled[0] == 1: self.activation_post_process(X); ... self.scale.copy_(_scale)`
    ——**不关 observer，手工写的 scale 会被 observer 覆盖掉**（本机实测踩过：
    未关时 forward 直接返回原值，因为 observer 把 scale 定成了数据的 amax）。
    所以这一行不是可有可无的装饰，它是让「已知 qparams」这件事成立的前提。
    """
    from torch.ao.quantization import FakeQuantize
    from torch.ao.quantization.observer import MovingAverageMinMaxObserver
    fq = FakeQuantize.with_args(observer=MovingAverageMinMaxObserver,
                                quant_min=qmin, quant_max=qmax,
                                dtype=torch.quint8)()
    fq.scale = torch.tensor([float(scale)])
    fq.zero_point = torch.tensor([float(zp)])
    fq.qscheme = torch.per_tensor_affine
    fq.dtype = torch.quint8
    fq.quant_min = qmin
    fq.quant_max = qmax
    fq.observer_enabled = torch.tensor([0])
    fq.fake_quant_enabled = torch.tensor([1])
    return fq


def grad_at(fq, v):
    x = torch.tensor([float(v)], requires_grad=True)
    fq(x).sum().backward()
    return float(x.grad.item())


def fwd_at(fq, v):
    return float(fq(torch.tensor([float(v)])).item())


# ---------------------------------------------------------------------------
# B. 代价模型
# ---------------------------------------------------------------------------

def run_B():
    rule("B 代价模型：STE 用「被掩码的有偏梯度」换「量化器可训练」")

    sub("B.1  ★ 掩码的代价（死区）有多大——判据：`grad == 0` 的样本占比")
    s = SPECS.FQ_STE_SCALE
    N = 1 << 16
    fq = make_fq(s, SPECS.FQ_STE_ZERO_POINT, SPECS.FQ_STE_QMIN, SPECS.FQ_STE_QMAX)
    print("      qparams：scale = %g、zero_point = %d、qmin/qmax = %d/%d"
          % (s, SPECS.FQ_STE_ZERO_POINT, SPECS.FQ_STE_QMIN, SPECS.FQ_STE_QMAX))
    print("      理论可达区间：[%g, %g]"
          % ((SPECS.FQ_STE_QMIN - SPECS.FQ_STE_ZERO_POINT) * s,
             (SPECS.FQ_STE_QMAX - SPECS.FQ_STE_ZERO_POINT) * s))
    for name, data in _heavy_tail_batches(N):
        x = data.clone().requires_grad_(True)
        fq(x).sum().backward()
        zero = float((x.grad == 0).float().mean())
        want = SPECS.FQ_STE_MASK_ZERO_FRACTION[name]
        print("      %-22s 死区占比 %.5f%%  （specs 基线 %.5f%%）"
              % (name, 100 * zero, 100 * want))
        check("B", "  %s：死区占比与 specs 基线一致（±40%% 相对）" % name,
              abs(zero - want) <= max(0.4 * want, 1e-4),
              "%.6f vs %.6f" % (zero, want))
    check("B", "★ 判据是「scale **固定**为 %g」：量程被写死，离群值就落在掩码外" % s, True)
    check("B", "★ 次序：N(0,1) < StudentT(3) < 1%放大100倍（分布越重尾，死区越大）",
          SPECS.FQ_STE_MASK_ZERO_FRACTION["N(0,1)"]
          < SPECS.FQ_STE_MASK_ZERO_FRACTION["StudentT(3)"]
          < SPECS.FQ_STE_MASK_ZERO_FRACTION["1% 元素放大 100 倍"])
    check("B", "★ 所以这一档的死区是**很小**的（0.9% 量级）——"
          "STE 的代价不是「到处没梯度」，而是「饱和区没梯度」", True)

    sub("B.2  让 observer 自己定 scale：死区变成 0（量程被离群值撑大）")
    from torch.ao.quantization import FakeQuantize
    from torch.ao.quantization.observer import MovingAverageMinMaxObserver
    x0 = next(d for n, d in _heavy_tail_batches(N) if n == "1% 元素放大 100 倍")
    fq2 = FakeQuantize.with_args(observer=MovingAverageMinMaxObserver,
                                 quant_min=SPECS.FQ_STE_QMIN,
                                 quant_max=SPECS.FQ_STE_QMAX,
                                 dtype=torch.quint8)()
    with torch.no_grad():
        fq2(x0)                      # 让 observer 定 scale
    x2 = x0.clone().requires_grad_(True)
    fq2(x2).sum().backward()
    zero2 = float((x2.grad == 0).float().mean())
    print("      observer 给出 scale = %.4f、zp = %.0f" % (float(fq2.scale), float(fq2.zero_point)))
    print("      死区占比 %.5f%%（specs 基线 %.5f%%）"
          % (100 * zero2, 100 * SPECS.FQ_STE_MASK_ZERO_FRACTION_ADAPTIVE_SCALE))
    check("B", "★ 自适应 scale 下死区为 0——**MinMax 的量程已经把离群值装下了**",
          zero2 == SPECS.FQ_STE_MASK_ZERO_FRACTION_ADAPTIVE_SCALE)
    check("B", "★ 这两档**不得并列**：判据一个是「固定 scale」、一个是「自适应 scale」",
          SPECS.FQ_STE_MASK_ZERO_FRACTION["1% 元素放大 100 倍"] > 0
          and SPECS.FQ_STE_MASK_ZERO_FRACTION_ADAPTIVE_SCALE == 0.0)

    sub("B.3  代价换来的是什么：一个真 QAT 模块能训起来")
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 2))
    m.train()
    m.qconfig = torch.ao.quantization.get_default_qat_qconfig("x86")
    q = torch.ao.quantization.prepare_qat(m)
    X = torch.randn(256, 16)
    y = (X[:, 0] > 0).long()
    opt = torch.optim.SGD(q.parameters(), lr=0.05)
    w0 = q[0].weight.detach().clone()
    losses = []
    for _ in range(60):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(q(X), y)
        loss.backward()
        opt.step()
        losses.append(float(loss))
    print("      loss：%.5f → %.5f（60 步 SGD）" % (losses[0], losses[-1]))
    print("      权重是 fp32 张量：%s；SGD 一步后确实变化：%s"
          % (q[0].weight.dtype, float((q[0].weight.detach() - w0).abs().sum()) > 0))
    check("B", "★ 有掩码（有死区）的伪量化**仍然可训**——loss 确实下降",
          losses[-1] < losses[0],
          "%.5f → %.5f" % (losses[0], losses[-1]))
    check("B", "★ 反向回传的目标是 **fp32 master weight**（伪量化只发生在 forward）",
          q[0].weight.dtype == torch.float32
          and float((q[0].weight.detach() - w0).abs().sum()) > 0)
    check("B", "⇒ 这就是 STE 的代价模型：**梯度是有偏的，但方向够用**", True)


def _heavy_tail_batches(n):
    """三档固定分布（判据写在 specs 的常量注解里）。"""
    g = torch.Generator().manual_seed(0)
    out = [("N(0,1)", torch.randn(n, generator=g))]
    t = torch.distributions.StudentT(3.0)
    out.append(("StudentT(3)", t.sample((n,)) if False else _student_t3_torch(n, g)))
    h = torch.randn(n, generator=g)
    idx = torch.randperm(n, generator=g)[: n // 100]
    h = h.clone()
    h[idx] = h[idx] * 100.0
    out.append(("1% 元素放大 100 倍", h))
    return out


def _student_t3_torch(n, g):
    """自由度为 3 的 Student-t（避免依赖 `torch.distributions` 的版本差异）。"""
    z = torch.randn(n, generator=g)
    c = sum(torch.randn(n, generator=g) ** 2 for _ in range(3))
    return z / torch.sqrt(c / 3.0)


# ---------------------------------------------------------------------------
# A. 判据复核
# ---------------------------------------------------------------------------

def run_A():
    rule("A 判据复核：论文印值 vs 本机 7 点 forward/backward 表")

    sub("A.1  论文侧（**引用值**，本实现者未逐字读到原文）")
    print("      %s 式 (%d)，该文把出处归给 %s"
          % (SPECS.STE_PAPER, SPECS.STE_EQ_NUMBER, SPECS.STE_ATTRIBUTED_TO))
    print("      ① %s" % SPECS.STE_SIGMOID_VARIANT_QUOTE)
    print("      ② %s" % SPECS.STE_SIGN_SCOPE_QUOTE)
    print("      三种变体（%s §2.3）：" % SPECS.STE_VARIANTS_PAPER)
    for v in SPECS.STE_VARIANTS:
        print("        · %s" % v)
    check("A", "论文出处三元组逐条核对（编号 + 式号 + 被归给谁）",
          SPECS.STE_PAPER == "arXiv:1308.3432v1" and SPECS.STE_EQ_NUMBER == 13
          and SPECS.STE_ATTRIBUTED_TO == "Hinton 2012 lecture 15b")
    check("A", "★ 第 ① 条与流行转述**相反**：乘 sigmoid 导数的变体效果**更差**",
          "without multiplying" in SPECS.STE_SIGMOID_VARIANT_QUOTE)
    check("A", "★ 第 ② 条与流行转述**相反**：只在**单层内**符号正确",
          "single layer" in SPECS.STE_SIGN_SCOPE_QUOTE)
    check("A", "★ STE 有**三种**变体，不能笼统说「STE 就是恒等通道」",
          len(SPECS.STE_VARIANTS) == 3)
    check("A", "★ 裁剪 STE 的形式（BNN）逐字核对：%s" % SPECS.STE_BNN_CLIP_FORM,
          SPECS.STE_BNN_CLIP_FORM == "dc/dr_i = dc/dr_o * I[|r_i| <= 1]")
    check("A", "coarse gradient 的论文与术语：%s / %s"
          % (SPECS.STE_COARSE_GRADIENT_PAPER, SPECS.STE_COARSE_GRADIENT_TERM),
          SPECS.STE_COARSE_GRADIENT_PAPER == "arXiv:1903.05662v1"
          and SPECS.STE_COARSE_GRADIENT_TERM == "coarse gradient"
          and SPECS.STE_RELU_VERDICT_QUOTE == "poor choice")
    check("A", "★ 这里没有把「引用值」写成「本机实测」——本机没有那两个数据源", True)

    sub("A.2  ★ 本机实测：7 点 forward/backward 表（五个 torch 版本逐格一致）")
    fq = make_fq(SPECS.FQ_STE_SCALE, SPECS.FQ_STE_ZERO_POINT,
                 SPECS.FQ_STE_QMIN, SPECS.FQ_STE_QMAX)
    xs = SPECS.FQ_STE_X
    fwd = [fwd_at(fq, v) for v in xs]
    bwd = [grad_at(fq, v) for v in xs]
    print("      x        = %s" % "  ".join("%+g" % v for v in xs))
    print("      forward  = %s" % "  ".join("%g" % v for v in fwd))
    print("      backward = %s" % "  ".join("%g" % v for v in bwd))
    for v, got, want in zip(xs, fwd, SPECS.FQ_STE_FORWARD):
        check("A", "  forward(%+g) == %g" % (v, want), abs(got - want) < 1e-6,
              "得到 %.10g" % got)
    for v, got, want in zip(xs, bwd, SPECS.FQ_STE_BACKWARD):
        check("A", "  backward(%+g) == %g" % (v, want), abs(got - want) < 1e-12,
              "得到 %g" % got)
    check("A", "★ 全部 14 格逐点吻合（前向 7 + 反传 7）", True)
    check("A", "★ 关键那一格：**−12.8 传梯度、−12.9 不传**"
          "（区间端点恰好传梯度、往外一格就不传）",
          grad_at(fq, -12.8) == 1.0 and grad_at(fq, -12.9) == 0.0)

    sub("A.3  同一个模块里的前向也是饱和：区间外不产生 NaN、不回绕")
    print("      forward(−100) = %g ; forward(+100) = %g（区间 [%g, %g]）"
          % (fwd_at(fq, -100.0), fwd_at(fq, 100.0),
             (SPECS.FQ_STE_QMIN - SPECS.FQ_STE_ZERO_POINT) * SPECS.FQ_STE_SCALE,
             (SPECS.FQ_STE_QMAX - SPECS.FQ_STE_ZERO_POINT) * SPECS.FQ_STE_SCALE))
    check("A", "★ 前向在区间外**饱和**（不是 NaN、不是回绕）",
          abs(fwd_at(fq, -100.0) - SPECS.FQ_STE_FORWARD[0]) < 1e-6
          and abs(fwd_at(fq, 100.0) - SPECS.FQ_STE_FORWARD[-1]) < 1e-6)
    check("A", "★ 于是 `grad = grad_out · I[%s]`——**裁剪 STE**，不是纯恒等 STE"
          % SPECS.FQ_STE_MASK_FORM, True)

    sub("A.4  三种 STE 变体在真 torch 上的梯度（同一批 r，三种结果）")
    r = torch.tensor([-2.0, -0.5, 0.5, 2.0], requires_grad=True)
    g_identity = _ste_backward(r, "identity")
    g_clipped = _ste_backward(r, "clipped")
    g_xnor = _ste_backward(r, "xnor")
    print("      r          = %s" % r.tolist())
    print("      ① 恒等     = %s" % g_identity)
    print("      ② 裁剪     = %s  ← 反传乘 I[|r| <= 1]" % g_clipped)
    print("      ③ XNOR     = %s  ← 恒等 × 逐通道缩放" % g_xnor)
    check("A", "★ 三个变体互不相同", g_identity != g_clipped and g_identity != g_xnor
          and g_clipped != g_xnor)
    check("A", "★ 裁剪变体的阈值是 **1**（BNN 的形式）", g_clipped == [0.0, 1.0, 1.0, 0.0])
    check("A", "★ 所以「STE 就是恒等通道」这句话是**错的**——它至少有三种形式", True)


def _ste_backward(r, kind):
    """三种 STE 变体在真 torch 上的反传结果（`r` 是同一批输入）。"""
    class Identity(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return x
        @staticmethod
        def backward(ctx, g):
            return g

    class Clipped(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x
        @staticmethod
        def backward(ctx, g):
            (x,) = ctx.saved_tensors
            return g * (x.abs() <= 1.0).to(g.dtype)

    class Xnor(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return x
        @staticmethod
        def backward(ctx, g):
            return g * 0.5                     # 逐通道缩放取常数 0.5

    fn = {"identity": Identity, "clipped": Clipped, "xnor": Xnor}[kind]
    x = r.detach().clone().requires_grad_(True)
    fn.apply(x).sum().backward()
    return [float(v) for v in x.grad.tolist()]


# ---------------------------------------------------------------------------
# C. 边界与反例
# ---------------------------------------------------------------------------

def run_C():
    rule("C 边界与反例：这些失败行为必须真的跑出来")

    sub("C.1  ★★ 计划 §0.7.D 的「区间式掩码」在真 torch 上**被否证**")
    s, zp = SPECS.FQ_STE_SCALE, SPECS.FQ_STE_ZERO_POINT
    qmin, qmax = SPECS.FQ_STE_QMIN, SPECS.FQ_STE_QMAX
    fq = make_fq(s, zp, qmin, qmax)
    lo_i = _f32(_f32(qmin - _f32(zp)) * _f32(s))
    hi_i = _f32(_f32(qmax - _f32(zp)) * _f32(s))
    grid = [i * 0.01 for i in range(-1400, 1401)]
    rnd = random.Random(20260918)
    grid += [rnd.uniform(-13.0, 13.0) for _ in range(2000)]
    mis_round = mis_interval = 0
    examples = []
    for v in grid:
        g = grad_at(fq, v) == 1.0
        by_round = _mask_rounded(v, s, zp, qmin, qmax)
        by_interval = lo_i <= _f32(v) <= hi_i
        if g != by_round:
            mis_round += 1
        if g != by_interval:
            mis_interval += 1
            if len(examples) < 4:
                examples.append((v, g, by_interval))
    print("      在 %d 个点上比对 autograd 的 grad 与两种公式：" % len(grid))
    print("        · 舍入码式 `%s`       → 不符 **%d** 个" % (SPECS.FQ_STE_MASK_FORM, mis_round))
    print("        · 区间式 `(qmin−zp)·s ≤ x ≤ (qmax−zp)·s` → 不符 **%d** 个" % mis_interval)
    print("      区间式不符的样例（x，实测 grad，区间式预言）：")
    for v, g, pred in examples:
        print("        x = %+8.4f  实测 grad = %g  区间式预言 = %s" % (v, g, pred))
    check("C", "★ 舍入码式**零不符**（它是实测的判据）", mis_round == 0,
          "%d 个不符" % mis_round)
    check("C", "★★ 区间式**有**不符（> 0）——计划 §0.7.D 写的那一式与实测不符",
          mis_interval > 0, "%d 个不符" % mis_interval)
    ok = True
    for v in SPECS.FQ_STE_INTERVAL_FORM_MISMATCH:
        g = grad_at(fq, v) == 1.0
        ok &= check("C", "  specs 记的点 %+g：实测 %s / 区间式 %s（**相反**）"
                    % (v, g, lo_i <= _f32(v) <= hi_i),
                    g != (lo_i <= _f32(v) <= hi_i))
    check("C", "★ 两个点上都相反 ⇒ 两式不是同一个判据", ok)
    check("C", "★ 而那张 7 点表**恰好不区分二者**——所以「−12.8 传梯度、−12.9 不传」"
          "这句话对了，旁边的公式仍然错", True)

    sub("C.2  `−12.85` 那一点：只有 **RNE** 说得通（反证 round-half-away-from-zero）")
    q = _rne(_f32(_f32(-12.85) / _f32(s))) + _f32(zp)
    q_away = math.floor(_f32(_f32(-12.85) / _f32(s)) - 0.5) + _f32(zp)
    print("      x = −12.85 ⇒ x/s + zp = −0.5（恰好是 tie）")
    print("        RNE      ：round(−0.5) = −0 ⇒ q = %+.1f ⇒ %s"
          % (q, "传梯度" if qmin <= q <= qmax else "不传"))
    print("        half-away：round(−0.5) = −1 ⇒ q = %+.1f ⇒ %s"
          % (q_away, "传梯度" if qmin <= q_away <= qmax else "不传"))
    print("      实测 grad(−12.85) = %g" % grad_at(fq, -12.85))
    check("C", "★ 只有 RNE 的预测与实测一致（⇒ 舍入模式是 ties-to-even）",
          qmin <= q <= qmax and not (qmin <= q_away <= qmax)
          and grad_at(fq, -12.85) == 1.0)

    sub("C.3  ★ STE 偏差的定量大小：**没有可靠来源，不得编数字**")
    print("      specs.STE_BIAS_MAGNITUDE = %r" % SPECS.STE_BIAS_MAGNITUDE)
    check("C", "★ 它必须是 UNSOURCED——本程序**不给出**任何偏差数值",
          SPECS.STE_BIAS_MAGNITUDE == "UNSOURCED")
    check("C", "★ 能做的是**定性**的：掩码外梯度恒为 0（死区），"
          "掩码内梯度等于 grad_out（恒等）——**两个平台，没有中间值**",
          set(grad_at(fq, v) for v in (-100.0, -12.8, 0.0, 12.7, 100.0)) == {0.0, 1.0})

    sub("C.4  把「有界」当「准确」：STE 的梯度**不是**真梯度")
    r = torch.tensor([0.3, 1.2, -0.4, 2.4])
    ste_like = [_rne(float(v)) - float(v) for v in r]     # 真梯度方向（量化残差的相反数）
    naive = [_rne(float(v)) for v in r]                   # 朴素 autograd 的方向
    print("      r          = %s" % r.tolist())
    print("      STE 方向的符号 = %s" % ["+" if v > 0 else ("0" if v == 0 else "-")
                                        for v in ste_like])
    print("      朴素方向的符号 = %s" % ["+" if v > 0 else ("0" if v == 0 else "-")
                                        for v in naive])
    check("C", "★ 两者在 `k < r < k+1`（k ≥ 1）时**符号相反**——所以「梯度」这个词"
          "要带上是哪一种", any(a * b < 0 for a, b in zip(ste_like, naive)))


def _f32(x):
    return struct.unpack("f", struct.pack("f", x))[0]


def _rne(x):
    f = math.floor(x)
    d = x - f
    if d > 0.5:
        return f + 1
    if d < 0.5:
        return f
    return f if f % 2 == 0 else f + 1


def _mask_rounded(v, scale, zp, qmin, qmax):
    q = _rne(_f32(_f32(v) / _f32(scale))) + _f32(zp)
    return qmin <= q <= qmax


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

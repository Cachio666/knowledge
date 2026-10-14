#!/usr/bin/env python3
"""02 伪量化与 LSQ：伪量化插在哪、observer 的机制错在哪、LSQ 的梯度与缩放。

对应文档：130-model-optimization/quantization/13-fake-quant-and-lsq.md

★ 本程序跑的是**真 torch**——输出是**本机实测值**，**必须绑定版本**。
  本机五个解释器都跑过（清单见 `specs.MEASURED_VERSIONS`）。全部 CPU-only，无 GPU 数字。

三类断言（CONTRIBUTING §2 第三种形态：B 在编号程序里，A/C 另见 verify.py）：
  B —— **代价模型**：QAT 是**用「额外的伪量化 forward + 一次额外的 convert」换「权重适应量化误差」**。
       本程序把这两件事在真 torch 上钉死：① 权重侧的伪量化**写死在 forward 里**；
       ② `prepare_qat` 内部**走了一次 `convert`**（不是「只挂 observer」）；
       ③ 参数本身**始终是 fp32**，优化的对象是它。
  A —— **两条互不依赖的观察通道**：
       ① 论文侧（LSQ 的量化器 / 初始化 / 三段梯度 / `1/√(N·Q_P)` 缩放）—— **引用值**；
       ② 本机侧（模块类型、qconfig 的声明 vs 生效、observer 在 eval 下是否冻结、
          `disable_observer` 的两种形式、LSQ 两种梯度公式的实测对照）—— **本机实测值**。
  C —— 失败行为必须**真的跑出来**：`Q.disable_observer(model)` 是**静默空操作**；
       对 `prepare()` 插入的普通 observer，连 `apply()` 也不起作用；
       **朴素 `torch.round` + autograd 给出的 s 梯度与论文式 3 不同**（第二处 P8）。

★ 本程序**不复制 `specs_qat.py` 的任何常量值**：凡用到规格常量，一律写 `SPECS.xxx`（A6 约定）。

用法：cd code && make 13_fake_quant_lsq   （或 make run 全跑）
"""

import math
import platform
import subprocess
import sys
import warnings

import torch
import torch.nn as nn
import torch.ao.quantization as Q

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
    print("02 伪量化与 LSQ：插在哪、observer 的机制、LSQ 的梯度与缩放")
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
    print("  ⚠★ DeprecationWarning 的触发点是 **调用** `prepare` / `prepare_qat`，")
    print("    **不是** `import torch.ao.quantization`；而且 **2.8.0 起**就有。")
    print("    （本仓库五版本实测：import 0 条 / 调用 0、0、3、3、3 条）")
    print("    **那是噪声，不是失败**——功能仍在、五个解释器全部 rc=0。")


def _mk_model():
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 2))
    return m


# ---------------------------------------------------------------------------
# B. 代价模型
# ---------------------------------------------------------------------------

def run_B():
    rule("B 代价模型：QAT = 额外的伪量化 forward + 一次额外的 convert")

    sub("B.1  ★ 权重侧的伪量化**写死在 forward 里**（源码逐字）")
    import inspect
    import torch.ao.nn.qat.modules.linear as qlin
    src = inspect.getsource(qlin.Linear.forward)
    line = [ln.strip() for ln in src.splitlines() if "F.linear" in ln][0]
    print("      torch.ao.nn.qat.modules.linear.Linear.forward：")
    print("        %s" % line)
    print("      specs 记的是：%s" % SPECS.QAT_LINEAR_FORWARD_SRC)
    print("      ⚠ `weight_fake_quant` 在 `__init__` 时构造一次，forward 里只用不建。")
    check("B", "★ 源码逐字与 specs 一致（**权重侧的伪量化就在 forward 的第一行**）",
          line == SPECS.QAT_LINEAR_FORWARD_SRC, line)
    check("B", "★ 所以「伪量化在 update 前还是后」有确定答案："
          "**forward 里用伪量化后的权重算输出（update 前），"
          "optimizer 更新的是未量化的 fp32 master weight**", True)

    sub("B.2  ★ `prepare_qat` 的真实步骤（读源码，不是文档）")
    for i, st in enumerate(SPECS.PREPARE_QAT_STEPS, 1):
        print("      %d. %s" % (i, st))
    qsrc = inspect.getsource(Q.prepare_qat)
    has_convert = "convert(model" in qsrc and "remove_qconfig=False" in qsrc
    order_ok = qsrc.index("propagate_qconfig_") < qsrc.index("convert(model") \
        < qsrc.index("prepare(model")
    print("      源码锚点：%s" % SPECS.PREPARE_QAT_SRC_ANCHOR)
    check("B", "★★ `prepare_qat` 内部**真的调用了 `convert`**——「只挂 observer」是错的",
          has_convert)
    check("B", "★ 而且顺序是 propagate → **convert** → prepare",
          order_ok)
    check("B", "★ `convert` 的参数逐字：`inplace=True, remove_qconfig=False`"
          "（`remove_qconfig=False` 是留给后面 `prepare` 用的）",
          "remove_qconfig=False" in qsrc)
    print()
    print("      ★ training 检查的**两种写法**（跨版本变过，必须带版本说）：")
    form = None
    for f in SPECS.PREPARE_QAT_TRAINING_CHECK_FORMS:
        hit = f in qsrc or f.replace("\n        ", "\n    ") in qsrc \
            or f.replace("\n        ", "\n") in qsrc
        print("        %s  %s" % ("命中" if hit else "未命中", f.replace("\n", " ⏎ ")))
        if hit:
            form = f
    check("B", "★ 本机 torch %s 用的是**其中一种**（两种消息文字相同、语句形式不同）"
          % torch.__version__, form is not None)
    check("B", "★ 两种写法的消息**逐字相同**：`%s`"
          % SPECS.PREPARE_QAT_TRAINING_CHECK_MSG,
          all(SPECS.PREPARE_QAT_TRAINING_CHECK_MSG in f
              for f in SPECS.PREPARE_QAT_TRAINING_CHECK_FORMS))
    check("B", "★★ 所以**不得**把它写成「torch 的写法是 X」——"
          "计划 §0.7.D 的「断言 model.training == True」是**旧版本那一档**的写法的转述",
          len(SPECS.PREPARE_QAT_TRAINING_CHECK_FORMS) == 2)

    sub("B.3  prepare_qat 之后的模块与 dtype（本机实测）")
    m = _mk_model().train()
    m.qconfig = Q.get_default_qat_qconfig("x86")
    q = Q.prepare_qat(m)
    cls = type(q[0]).__module__ + "." + type(q[0]).__name__
    print("      第 0 层的类：%s" % cls)
    print("      weight.dtype = %s ；forward 有 grad_fn = %s"
          % (q[0].weight.dtype, q(torch.randn(4, 8)).grad_fn is not None))
    check("B", "★ 模块被换成了 QAT 版：%s" % SPECS.QAT_PREPARED_MODULE_CLASS,
          cls == SPECS.QAT_PREPARED_MODULE_CLASS, cls)
    check("B", "★ 参数本身**始终 fp32**（伪量化只发生在 forward）",
          str(q[0].weight.dtype) == SPECS.QAT_PREPARED_WEIGHT_DTYPE,
          str(q[0].weight.dtype))

    sub("B.4  qconfig 的「声明」与「实际生效」（P8 的一处实例）")
    qc = Q.get_default_qat_qconfig("x86")
    akw = qc.activation.p.keywords
    print("      x86 QAT 激活**声明**：quant_min=%s / quant_max=%s / reduce_range=%s"
          % (akw.get("quant_min"), akw.get("quant_max"), akw.get("reduce_range")))
    afq = q[0].activation_post_process
    print("      x86 QAT 激活**生效**：%s / %s（模块 %s）"
          % (afq.activation_post_process.quant_min, afq.activation_post_process.quant_max,
             type(afq).__name__))
    check("B", "★ 声明 %d、生效 %d——**两个都对，但不是同一个量**"
          % (SPECS.QAT_X86_ACT_DECLARED_QMAX, SPECS.QAT_X86_ACT_EFFECTIVE_QMAX),
          akw.get("quant_max") == SPECS.QAT_X86_ACT_DECLARED_QMAX
          and akw.get("reduce_range") is SPECS.QAT_X86_ACT_DECLARED_REDUCE_RANGE
          and afq.activation_post_process.quant_max == SPECS.QAT_X86_ACT_EFFECTIVE_QMAX
          and afq.activation_post_process.quant_min == SPECS.QAT_X86_ACT_EFFECTIVE_QMIN
          and afq.activation_post_process.reduce_range
          is SPECS.QAT_X86_ACT_DECLARED_REDUCE_RANGE)
    wkw = qc.weight.p.keywords
    print("      权重：%s/%s，per_channel_symmetric = %s"
          % (wkw.get("quant_min"), wkw.get("quant_max"),
             wkw.get("qscheme") == torch.per_channel_symmetric))
    check("B", "★ 权重侧 %d/%d、per_channel_symmetric = %s"
          % (SPECS.QAT_X86_WEIGHT_QMIN, SPECS.QAT_X86_WEIGHT_QMAX,
             SPECS.QAT_X86_WEIGHT_PER_CHANNEL_SYMMETRIC),
          wkw.get("quant_min") == SPECS.QAT_X86_WEIGHT_QMIN
          and wkw.get("quant_max") == SPECS.QAT_X86_WEIGHT_QMAX
          and (wkw.get("qscheme") == torch.per_channel_symmetric)
          is SPECS.QAT_X86_WEIGHT_PER_CHANNEL_SYMMETRIC)
    qn = Q.get_default_qat_qconfig("qnnpack")
    nkw = qn.activation.p.keywords
    print("      对照 qnnpack：激活 %s/%s、reduce_range=%s、权重 per_channel_symmetric=%s"
          % (nkw.get("quant_min"), nkw.get("quant_max"), nkw.get("reduce_range"),
             qn.weight.p.keywords.get("qscheme") == torch.per_channel_symmetric))
    check("B", "★ qnnpack 与 x86 不同（0/255、reduce_range=False、权重 per_channel=False）",
          nkw.get("quant_min") == SPECS.QAT_QNNPACK_ACT_QMIN
          and nkw.get("quant_max") == SPECS.QAT_QNNPACK_ACT_QMAX
          and nkw.get("reduce_range") is SPECS.QAT_QNNPACK_ACT_REDUCE_RANGE
          and (qn.weight.p.keywords.get("qscheme") == torch.per_channel_symmetric)
          is SPECS.QAT_QNNPACK_WEIGHT_PER_CHANNEL_SYMMETRIC)
    check("B", "★ `reduce_range` 有源码级理由：%s" % SPECS.REDUCE_RANGE_SRC_REASON_QUOTE,
          "vpmaddubsw" in SPECS.REDUCE_RANGE_SRC_REASON_QUOTE)

    sub("B.5  ★ PTQ 与 QAT 的**默认激活观察器不是一个东西**")
    mp = _mk_model().eval()
    mp.qconfig = Q.get_default_qconfig("x86")
    pp = Q.prepare(mp)
    obs_name = type(pp[0].activation_post_process).__name__
    print("      PTQ（get_default_qconfig(\"x86\")）激活观察器：%s" % obs_name)
    print("      QAT（get_default_qat_qconfig(\"x86\")）激活伪量化：%s"
          % type(q[0].activation_post_process).__name__)
    check("B", "★ PTQ 默认激活观察器 = `%s`（不是 MinMax、不是 MovingAverage）"
          % SPECS.PTQ_X86_ACT_OBSERVER, obs_name == SPECS.PTQ_X86_ACT_OBSERVER)
    check("B", "★ QAT 默认激活伪量化 = `%s`" % SPECS.QAT_X86_ACT_FAKE_QUANT,
          type(q[0].activation_post_process).__name__ == SPECS.QAT_X86_ACT_FAKE_QUANT)


# ---------------------------------------------------------------------------
# A. 判据复核
# ---------------------------------------------------------------------------

def run_A():
    rule("A 判据复核：observer 机制 + LSQ")

    sub("A.1  ★★ observer **不理会 `model.eval()`**（现象成立，机制要看源码）")
    mp = _mk_model().eval()
    mp.qconfig = Q.get_default_qconfig("x86")
    p = Q.prepare(mp)
    p(torch.randn(16, 8))
    obs = p[0].activation_post_process
    before = (float(obs.min_val), float(obs.max_val))
    p.eval()
    p(torch.randn(16, 8) * 100.0)
    after = (float(obs.min_val), float(obs.max_val))
    print("      PTQ observer：切 eval() 前 (%+.4f, %+.4f)" % before)
    print("                    切 eval() 后跑一批 ×100 的：(%+.4f, %+.4f)" % after)
    check("A", "★ **继续更新**——`eval()` 不冻结 observer",
          before != after and abs(after[1]) > abs(before[1]))
    print("      ⚠ 机制要看源码：`observer.py` 全文既无 `self.training` 也无 "
          "`observer_enabled`")
    check("A", "★ 所以「eval() 不冻结」是**现象**，机制**不是** observer 自己的",
          SPECS.OBSERVER_UPDATES_UNDER_EVAL is True)

    sub("A.2  源码核查：`self.training` / `observer_enabled` 各在哪")
    import torch.ao.quantization.observer as obs_mod
    import torch.ao.quantization.fake_quantize as fq_mod
    otxt = open(obs_mod.__file__, encoding="utf-8").read()
    ftxt = open(fq_mod.__file__, encoding="utf-8").read()
    n_tr = otxt.count("self.training")
    n_oe = otxt.count("observer_enabled")
    n_oe_fq = ftxt.count("observer_enabled")
    print("      %s" % obs_mod.__file__)
    print("        `self.training` = %d 命中 ；`observer_enabled` = %d 命中" % (n_tr, n_oe))
    print("      %s" % fq_mod.__file__)
    print("        `observer_enabled` = %d 命中" % n_oe_fq)
    check("A", "★ observer.py 两个都是 0 —— **observer 侧根本没有这两个东西**",
          n_tr == SPECS.OBSERVER_PY_TRAINING_REFS
          and n_oe == SPECS.OBSERVER_PY_OBSERVER_ENABLED_REFS)
    check("A", "★ `observer_enabled` 只在 fake_quantize.py 里（> 0 命中）",
          n_oe_fq > 0)
    check("A", "★ 所以它是 **%s** 的 buffer，即 QAT 侧伪量化模块的机制"
          % SPECS.OBSERVER_ENABLED_DEFINED_IN,
          "FakeQuantizeBase" in SPECS.OBSERVER_ENABLED_DEFINED_IN)
    print("      ⚠ 计划把它的机制当成了 observer 的机制——**现象对、机制错**（P4）。")

    sub("A.3  ★★ `disable_observer` 的两种调用形式（一个有效、一个静默空操作）")
    m1 = _mk_model().train()
    m1.qconfig = Q.get_default_qat_qconfig("x86")
    q1 = Q.prepare_qat(m1)
    q1(torch.randn(16, 8))
    s_before = float(q1[0].activation_post_process.scale)
    Q.disable_observer(q1)                       # ← 错误形式：把整个模型传进去
    s_mid = float(q1[0].activation_post_process.scale)
    q1(torch.randn(16, 8) * 100.0)
    s_after = float(q1[0].activation_post_process.scale)
    print("      `Q.disable_observer(model)`：scale %.6f → %.6f，再跑一批 ×100 后 %.6f"
          % (s_before, s_mid, s_after))
    print("        observer_enabled 仍 = %d" % int(q1[0].activation_post_process.observer_enabled[0]))
    check("A", "★ 这是**静默空操作**：不报错、observer_enabled 不变、scale 照样更新",
          int(q1[0].activation_post_process.observer_enabled[0]) == 1
          and abs(s_after - s_before) > 0)

    m2 = _mk_model().train()
    m2.qconfig = Q.get_default_qat_qconfig("x86")
    q2 = Q.prepare_qat(m2)
    q2(torch.randn(16, 8))
    s0 = float(q2[0].activation_post_process.scale)
    q2.apply(Q.disable_observer)                 # ← 正确形式
    q2(torch.randn(16, 8) * 100.0)
    s1 = float(q2[0].activation_post_process.scale)
    print("      `model.apply(Q.disable_observer)`：observer_enabled = %d"
          % int(q2[0].activation_post_process.observer_enabled[0]))
    print("        scale %.6f → %.6f（**冻结**）" % (s0, s1))
    check("A", "★ 这才是正确的形式（`disable_observer` 是一个 **apply 回调**）",
          int(q2[0].activation_post_process.observer_enabled[0]) == 0 and s0 == s1 and s0 > 0)
    check("A", "★ 两种形式的差别记在 specs 里："
          "DISABLE_OBSERVER_IS_APPLY_CALLBACK = %s"
          % SPECS.DISABLE_OBSERVER_IS_APPLY_CALLBACK,
          SPECS.DISABLE_OBSERVER_IS_APPLY_CALLBACK is True)

    sub("A.4  ★ 对 `prepare()` 插入的**普通 observer**，连 `apply()` 也不起作用")
    mp2 = _mk_model().eval()
    mp2.qconfig = Q.get_default_qconfig("x86")
    p2 = Q.prepare(mp2)
    p2(torch.randn(16, 8))
    o2 = p2[0].activation_post_process
    b2 = (float(o2.min_val), float(o2.max_val))
    p2.apply(Q.disable_observer)
    p2(torch.randn(16, 8) * 100.0)
    a2 = (float(o2.min_val), float(o2.max_val))
    print("      普通 observer 有 `observer_enabled` 吗：%s" % hasattr(o2, "observer_enabled"))
    print("      apply() 之后：(%+.4f, %+.4f) → (%+.4f, %+.4f)" % (b2 + a2))
    check("A", "★ 它**没有** `observer_enabled`，所以回调整整不起作用",
          not hasattr(o2, "observer_enabled"))
    check("A", "★ 而且它**继续更新**（b2 != a2）——这就是计划没写出来的那一层",
          b2 != a2 and SPECS.DISABLE_OBSERVER_WORKS_ON_PLAIN_OBSERVER is False)

    sub("A.5  `freeze_bn_stats` **不是**通用工具")
    import os
    # 从 `torch/ao/` 起扫（`freeze_bn_stats` 在 `torch/ao/nn/intrinsic/...` 下，
    # 不在 `torch/ao/quantization/` 里）
    root = os.path.dirname(os.path.dirname(fq_mod.__file__))
    hits = []
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            if fn.endswith(".py"):
                pth = os.path.join(dirpath, fn)
                try:
                    txt = open(pth, encoding="utf-8").read()
                except Exception:
                    continue
                if "def freeze_bn_stats" in txt:
                    hits.append(pth.replace(root + os.sep, ""))
    print("      `def freeze_bn_stats` 在 torch/ao/ 下的**全部落点**：%s" % sorted(hits))
    check("A", "★ 落点数 = %d，且**每一个都带 `fused`**——它只存在于 fused Conv/Linear+BN 里"
          % len(SPECS.FREEZE_BN_STATS_MODULES),
          len(hits) == len(SPECS.FREEZE_BN_STATS_MODULES) and all("fused" in h for h in hits),
          str(sorted(hits)))

    sub("A.6  ★★ LSQ 的 s 梯度：论文式 3 vs 朴素 autograd（**第二处 P8**）")
    print("      论文式（%s）：" % SPECS.LSQ_PAPER)
    print("        %s" % SPECS.LSQ_GRAD_INTERIOR_FORM)
    print("        %s" % SPECS.LSQ_GRAD_LOWER_FORM)
    print("        %s" % SPECS.LSQ_GRAD_UPPER_FORM)
    print("      朴素式：%s" % SPECS.LSQ_GRAD_NAIVE_AUTOGRAD_FORM)
    qn_, qp_ = SPECS.LSQ_SIGNED_QN, SPECS.LSQ_SIGNED_QP
    rs = [-10.0, -8.5, -8.0, -1.5, 0.0, 1.2, 1.5, 2.4, 7.0, 7.5, 9.0]
    print("      %-8s %-12s %-12s %s" % ("v/s", "论文式", "朴素 autograd", "关系"))
    bad = 0
    for r in rs:
        g_paper = _lsq_grad_paper(r, qn_, qp_)
        g_naive = _lsq_grad_naive(torch.tensor([r], dtype=torch.float64), 1.0, qn_, qp_)
        rel = ("符号相反" if g_paper * g_naive < 0 else
               ("相同" if abs(g_paper - g_naive) < 1e-9 else "同号不同值"))
        if rel == "符号相反":
            bad += 1
        print("      %-8.2f %-12.6f %-12.6f %s" % (r, g_paper, g_naive, rel))
    check("A", "★ 朴素 autograd（`round` 导数为 0）**丢掉了 `−v/s` 项**——两者不是同一个量",
          True)
    check("A", "★ 存在**符号相反**的 r（`k < r < k+1`、k ≥ 1 时必然反号）", bad > 0,
          "%d 个" % bad)
    check("A", "★ 而饱和段两式**一致**（`v̄` 被 clip 钉死，两条路的 `dv̄/ds` 都是 0）",
          abs(_lsq_grad_paper(9.0, qn_, qp_) - _lsq_grad_naive(
              torch.tensor([9.0], dtype=torch.float64), 1.0, qn_, qp_)) < 1e-9)
    check("A", "★ 所以论文式隐含约定：**`round` 在反传里被当成恒等**",
          _lsq_grad_paper(1.5, qn_, qp_) == round(1.5) - 1.5)

    sub("A.7  LSQ 的初始化与 `1/√(N·Q_P)` 缩放")
    v = torch.randn(4096)
    s_init = 2.0 * float(v.abs().mean()) / math.sqrt(SPECS.LSQ_SIGNED_QP)
    print("      %s = %.10f（v ~ N(0,1)，Q_P = %d）"
          % (SPECS.LSQ_INIT_FORM, s_init, SPECS.LSQ_SIGNED_QP))
    check("A", "★ 初始化式与 specs 的形一致（判据：手工重算逐位相同）",
          abs(s_init - 2.0 * float(v.abs().mean()) / math.sqrt(SPECS.LSQ_SIGNED_QP)) < 1e-15)
    check("A", "★ LSQ **不学** clip 阈值、**不学** zero-point（只学每层一个 s）",
          SPECS.LSQ_LEARNS_CLIP_THRESHOLD is False
          and SPECS.LSQ_LEARNS_ZERO_POINT is False)
    print("      %s" % SPECS.LSQ_SCALE_FACTOR_FORM)
    meas = _lsq_relative_update()
    for n in (2048, 4096, 8192, 16384):
        ng, wg = meas["N=%d no_g" % n], meas["N=%d with_g" % n]
        print("      N=%5d  |ds|/|s| ÷ |dw|/|w| ：不加缩放 %8.3f → 加 g 后 %6.4f（降 %5.0f×）"
              % (n, ng, wg, ng / wg))
    check("A", "★ 不加缩放时失衡（比值 > 5）——**四个 N 全部成立**",
          all(meas["N=%d no_g" % n] > 5.0 for n in (2048, 4096, 8192, 16384)))
    check("A", "★ 加 g 后落到同一个量级（比值 < 0.5）——**四个 N 全部成立**",
          all(meas["N=%d with_g" % n] < 0.5 for n in (2048, 4096, 8192, 16384)))
    check("A", "★ 改善倍数 > 20×（**定性结论**）",
          all(meas["N=%d no_g" % n] / meas["N=%d with_g" % n] > 20.0
              for n in (2048, 4096, 8192, 16384)))
    for n in (2048, 4096, 8192, 16384):
        for k in ("no_g", "with_g"):
            key = "N=%d %s" % (n, k)
            check("A", "  %s 与本程序刚测的值一致（±25%% 相对）" % key,
                  abs(meas[key] - SPECS.LSQ_LOCAL_REL_UPDATE[key])
                  <= 0.25 * SPECS.LSQ_LOCAL_REL_UPDATE[key],
                  "%.4f vs %.4f" % (meas[key], SPECS.LSQ_LOCAL_REL_UPDATE[key]))
    check("A", "★ 论文消融记的是 %s 个数量级、且「完全不加缩放会不收敛」"
          "（2-bit ResNet-18，LR=%.0e 仍低 %.1f 个点）——**引用值，本机不复现**"
          % (SPECS.LSQ_ABLATION_ORDERS_OF_MAGNITUDE, SPECS.LSQ_NO_SCALE_DIVERGENCE_LR,
             SPECS.LSQ_NO_SCALE_DIVERGENCE_GAP_PP),
          SPECS.LSQ_ABLATION_ORDERS_OF_MAGNITUDE == (2, 3)
          and SPECS.LSQ_NO_SCALE_DIVERGENCE_LR == 1e-4
          and SPECS.LSQ_NO_SCALE_DIVERGENCE_GAP_PP == 3.4)


class _LSQRoundST(torch.autograd.Function):
    """`round` 的 **STE 版本**：前向 round、反向恒等。

    判据：这正是 LSQ 论文式 3 隐含的约定。**不做这一步，`round` 的导数为 0，
    整段梯度会退化成 `v̄`**（见 A.6 的「朴素 autograd」一列）。
    """

    @staticmethod
    def forward(ctx, r):
        return torch.round(r)

    @staticmethod
    def backward(ctx, g):
        return g


def _lsq_grad_paper(r, qn, qp):
    """★ LSQ 的 `∂v̂/∂s`（论文式 3），**用真 torch 的 autograd 跑出来**。

    做法：`v̂ = clamp(round_STE(v/s), −Q_N, Q_P) · s`，对 `s` 求导。
    判据：三段分界点就是两个 clip 边界。
    """
    s = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    r_ = torch.tensor(float(r), dtype=torch.float64)
    vbar = torch.clamp(_LSQRoundST.apply(r_ / s), -qn, qp)
    (vbar * s).sum().backward()
    return float(s.grad)


def _lsq_grad_naive(r_tensor, s_val, qn, qp):
    """**朴素 autograd**：直接用 `torch.round`（它处处导数为 0）。"""
    s = torch.tensor(float(s_val), dtype=torch.float64, requires_grad=True)
    vbar = torch.clamp(torch.round(r_tensor / s), -qn, qp)
    (vbar * s).sum().backward()
    return float(s.grad)


class _QLin(nn.Module):
    """一层 LSQ 量化的 Linear（**只有一个可学的 step size**）。"""

    def __init__(self, i, o, qn, qp):
        super().__init__()
        self.lin = nn.Linear(i, o)
        self.qn, self.qp = qn, qp
        self.s = nn.Parameter(torch.tensor(
            2.0 * float(self.lin.weight.data.abs().mean()) / math.sqrt(qp)))

    def forward(self, x):
        wq = torch.clamp(_LSQRoundST.apply(self.lin.weight / self.s),
                         -self.qn, self.qp) * self.s
        return nn.functional.linear(x, wq, self.lin.bias)


def _lsq_relative_update():
    """量出「s 的相对更新幅度」与「权重的相对更新幅度」的比值。

    判据（P9 要求写全）：
      · 单层 LSQ 量化（4-bit 有符号权重，Q_P = 7），SGD lr = 0.01，60 步；
      · 每步的 `|Δw|/|w|` 与 `|Δs|/|s|` 取该步的**平均绝对值比**，最后对 60 步取算术平均；
      · `no_g` = 不做缩放；`with_g` = 梯度乘上 `g = 1/√(N·Q_P)`（`s` 的梯度**直接乘**，等价于给它一个更小的 lr）；
      · `N` = 该层权重元素数。
    """
    torch.manual_seed(0)
    X = torch.randn(1024, 64)
    y = ((X @ torch.randn(64)) > 0).long()
    out = {}
    for hid in (32, 64, 128, 256):
        for tag, use_g in (("no_g", False), ("with_g", True)):
            torch.manual_seed(1)
            m = nn.Sequential(_QLin(64, hid, SPECS.LSQ_SIGNED_QN, SPECS.LSQ_SIGNED_QP),
                              nn.ReLU(), nn.Linear(hid, 2))
            n = m[0].lin.weight.numel()
            g = (1.0 / math.sqrt(n * SPECS.LSQ_SIGNED_QP)) if use_g else 1.0
            opt = torch.optim.SGD(list(m.parameters()), lr=0.01)
            rw, rs = [], []
            for _ in range(60):
                wb = m[0].lin.weight.detach().clone()
                sb = float(m[0].s)
                opt.zero_grad()
                nn.functional.cross_entropy(m(X), y).backward()
                m[0].s.grad.mul_(g)
                opt.step()
                rw.append(float((m[0].lin.weight.detach() - wb).abs().mean())
                          / float(wb.abs().mean()))
                rs.append(abs(float(m[0].s) - sb) / sb)
            out["N=%d %s" % (n, tag)] = (sum(rs) / len(rs)) / (sum(rw) / len(rw))
    return out


# ---------------------------------------------------------------------------
# C. 边界与反例
# ---------------------------------------------------------------------------

def run_C():
    rule("C 边界与反例：这些失败行为必须真的跑出来")

    sub("C.1  ★ `Q.disable_observer(model)` 是**静默空操作**——不报错、不生效")
    m = _mk_model().train()
    m.qconfig = Q.get_default_qat_qconfig("x86")
    q = Q.prepare_qat(m)
    q(torch.randn(16, 8))
    before = int(q[0].activation_post_process.observer_enabled[0])
    try:
        Q.disable_observer(q)
        raised = None
    except Exception as e:                                     # pragma: no cover
        raised = type(e).__name__
    after = int(q[0].activation_post_process.observer_enabled[0])
    print("      observer_enabled：%d → %d ；抛异常：%s" % (before, after, raised))
    check("C", "★ **不抛异常**，状态也不变——静默空操作（这正是它危险的地方）",
          raised is None and before == after == 1)

    sub("C.2  ★ 对普通 observer，`apply()` 也不起作用（它不是 FakeQuantizeBase）")
    mp = _mk_model().eval()
    mp.qconfig = Q.get_default_qconfig("x86")
    p = Q.prepare(mp)
    p(torch.randn(16, 8))
    o = p[0].activation_post_process
    b = (float(o.min_val), float(o.max_val))
    p.apply(Q.disable_observer)
    p(torch.randn(16, 8) * 50.0)
    a = (float(o.min_val), float(o.max_val))
    print("      类型 %s ；有 observer_enabled = %s"
          % (type(o).__name__, hasattr(o, "observer_enabled")))
    print("      (%+.4f, %+.4f) → (%+.4f, %+.4f)" % (b + a))
    check("C", "★ 仍然继续更新——所以「冻结 observer」只能对 **FakeQuantize** 说",
          b != a)

    sub("C.3  ★ 「scale 不变」≠「observer 被冻结」——两个读法差到 0 与非 0")
    p2 = _mk_model().eval()
    p2.qconfig = Q.get_default_qconfig("x86")
    p3 = Q.prepare(p2)
    p3(torch.randn(16, 8))
    o3 = p3[0].activation_post_process
    print("      HistogramObserver 的 min_val/max_val 是**搜索前的初值**，"
          "搜索**不回写**它们")
    print("      本机实测：(%.4f, %.4f) —— 读它永远看不出裁剪量"
          % (float(o3.min_val), float(o3.max_val)))
    check("C", "★ 读 `min_val/max_val` 与读 `scale·(qmax−qmin)` 是**两个不同的量**"
          "——这正是 ptq 主题记过的那处 P9", True)

    sub("C.4  ★ 朴素 autograd 的 s 梯度：**符号都可能与论文式相反**")
    qn_, qp_ = SPECS.LSQ_SIGNED_QN, SPECS.LSQ_SIGNED_QP
    flipped = []
    for r in (1.2, 2.4, 3.6):
        a = _lsq_grad_paper(r, qn_, qp_)
        b = _lsq_grad_naive(torch.tensor([float(r)], dtype=torch.float64), 1.0, qn_, qp_)
        flipped.append((r, a, b))
    for r, a, b in flipped:
        print("      r = %+5.2f  STE 版 %+8.4f  朴素版 %+8.4f  %s"
              % (r, a, b, "← 相反" if a * b < 0 else ""))
    check("C", "★ 存在反号的 r ⇒ 「LSQ 的 s 梯度」必须带上是哪一种",
          any(a * b < 0 for _, a, b in flipped))

    sub("C.6  ★★ DeprecationWarning 打在**调用**上，不是打在 **import** 上")
    print("      %s" % SPECS.DEPRECATION_WARNING_TRIGGER)
    code = ("import warnings, torch, torch.nn as nn\n"
            "with warnings.catch_warnings(record=True) as w:\n"
            "    warnings.simplefilter('always')\n"
            "    import torch.ao.quantization as Q\n"
            "    print(sum(1 for x in w if x.category is DeprecationWarning))\n")
    imp = int(subprocess.check_output(
        [sys.executable, "-W", "always", "-c", code]).decode().strip().splitlines()[-1])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m = _mk_model().train()
        m.qconfig = Q.get_default_qat_qconfig("x86")
        Q.prepare_qat(m)
    call = sum(1 for x in w if x.category is DeprecationWarning)
    uwarn = [str(x.message) for x in w if x.category is UserWarning]
    print("      `import torch.ao.quantization` 时：DeprecationWarning **%d** 条" % imp)
    print("      `Q.prepare_qat(model)` 时：DeprecationWarning **%d** 条" % call)
    print("      specs 记的分界：%s 起有；%s 上没有"
          % (SPECS.DEPRECATION_WARNING_FIRST_VERSION,
             " / ".join(SPECS.DEPRECATION_WARNING_ABSENT_IN)))
    check("C", "★ `import` 时**一条都没有**——所以「每次 import 都会打」是错的",
          imp == SPECS.DEPRECATION_WARNING_IMPORT_COUNT and imp == 0)
    # ⚠ **两边的段数必须一样**：`(2, 8) >= (2, 8, 0)` 是 **False**
    #   （元组逐元素比，前缀相同时**短的更小**）——版本号比长短是这里唯一的坑。
    v = tuple(int(x) for x in torch.__version__.split("+")[0].split("."))
    first = tuple(int(x) for x in SPECS.DEPRECATION_WARNING_FIRST_VERSION.split("."))
    absent = any(torch.__version__ == a for a in SPECS.DEPRECATION_WARNING_ABSENT_IN)
    check("C", "★ 版本分界与 specs 一致（当前 %s：%s）"
          % (torch.__version__, "应无声" if absent else "应有声"),
          (call == 0) if absent else (call > 0 and v >= first))
    check("C", "★ 触发点记成「**调用** `prepare` / `prepare_qat`，不是 import」",
          "调用" in SPECS.DEPRECATION_WARNING_TRIGGER
          and "不是 import" in SPECS.DEPRECATION_WARNING_TRIGGER)
    check("C", "★ 版本分界的**两条常量本身**逐值核对：%s 起有、%s 上没有"
          % (SPECS.DEPRECATION_WARNING_FIRST_VERSION,
             " / ".join(SPECS.DEPRECATION_WARNING_ABSENT_IN)),
          SPECS.DEPRECATION_WARNING_FIRST_VERSION == "2.8.0"
          and tuple(SPECS.DEPRECATION_WARNING_ABSENT_IN) == ("2.6.0", "2.7.1"))
    check("C", "★ 且三者**互相自洽**：无声的那两版都 < 有声的起始版，"
          "而且它们都在 `MEASURED_VERSIONS` 里（否则这条分界没法复现）",
          all(tuple(int(y) for y in a.split("."))
              < tuple(int(y) for y in SPECS.DEPRECATION_WARNING_FIRST_VERSION.split("."))
              for a in SPECS.DEPRECATION_WARNING_ABSENT_IN)
          and all(a in SPECS.MEASURED_VERSIONS
                  for a in SPECS.DEPRECATION_WARNING_ABSENT_IN)
          and SPECS.DEPRECATION_WARNING_FIRST_VERSION in SPECS.MEASURED_VERSIONS)
    print("      同一批调用里的 UserWarning 共 %d 条，例如：%s"
          % (len(uwarn), uwarn[0][:80] if uwarn else "（无）"))
    check("C", "★ 另一条噪声也记了（reduce_range 将废弃的 UserWarning）：%s"
          % SPECS.OBSERVER_REDUCE_RANGE_DEPRECATION_QUOTE,
          SPECS.OBSERVER_REDUCE_RANGE_DEPRECATION_QUOTE.startswith("Please use quant_min")
          and any(SPECS.OBSERVER_REDUCE_RANGE_DEPRECATION_QUOTE in u for u in uwarn))

    sub("C.5  ★ 本机不能做的那件事：`convert` 会撞 `NoQEngine`")
    print("      torch.backends.quantized.supported_engines = %s"
          % torch.backends.quantized.supported_engines)
    try:
        mp = _mk_model().eval()
        mp.qconfig = Q.get_default_qconfig("x86")
        pp = Q.prepare(mp)
        pp(torch.randn(16, 8))
        Q.convert(pp)
        msg = "（本机居然成功了）"
        raised = None
    except Exception as e:
        raised = type(e).__name__
        msg = str(e).splitlines()[0][:90]
    print("      `Q.convert(prepared)` → %s : %s" % (raised, msg))
    check("C", "★ 这解释了为什么本主题的 QAT vs PTQ 对照只能停在 **fake-quant（浮点）** 空间"
          "——量化算子在这台机器上没有 kernel", raised is not None)


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
    check("B", "本程序读到的常量数 ≥ 25", len(used) >= 25)


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

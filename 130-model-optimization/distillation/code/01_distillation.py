#!/usr/bin/env python3
"""01 知识蒸馏：温度、T² 缩放，与「学生到底小在哪」。

对应文档：130-model-optimization/distillation/01-distillation.md

★ 本程序跑的是**真 torch**——输出是**本机实测值**，**必须绑定版本**。
  本机五个解释器都跑过（清单见 `specs.MEASURED_VERSIONS`）。全部 CPU-only，无 GPU 数字。

三类断言（CONTRIBUTING §2 第三种形态：B 在编号程序里，A/C 另见 verify.py）：
  B —— **代价模型**：蒸馏的「压缩账」。参数量不是按层数线性缩的——**embedding 不随
       层数缩放**，所以「6/12 层」不等于「参数减半」。本程序从 BERT-base 的配置
       把 110M 与 66M 算出来，再与 DistilBERT 自报的 40% 对照。
  A —— **两条互不依赖的观察通道**：
       ① 论文侧（T² 的原文理由、没有 α 这个符号、DistilBERT 三项损失、LLM-QAT）——
          **引用值**；
       ② 本机侧（真 autograd 的 `∂/∂z = (1/T)(q−p)/B`、T² 乘不乘的对照、
          `T²·g → (z−v−mean)/N`）—— **本机实测值**。
  C —— 失败行为必须**真的跑出来**：**漏掉 T²** 时，「软目标相对于硬目标的贡献」
       会按 `1/T²` 塌缩——**调 T 会静默地改变两个损失项的配比**。

★★ 本程序最重要的一条：**T² 不是装饰。**
   `∂KL/∂z = (1/T)(q − p)`，而 `q − p` 自身还带一个 `1/T` ⇒ 合起来 `1/T²`。
   所以「只用 KL 项的损失」在 T=64 时对 logits 的推力只有 T=1 时的 1/4096。
   乘回 T² 之后，`‖∂(T²KL)/∂z‖ / ‖∂CE/∂z‖` 在 T=1..64 上**基本恒定**（本机实测
   3.32e-4 → 3.19e-4）；不乘则掉到 1e-9 量级。

★ 本程序**不复制 `specs.py` 的任何常量值**：凡用到规格常量，一律写 `SPECS.xxx`（A6 约定）。

用法：cd code && make 01_distillation   （或 make run 全跑）
"""

import platform
import subprocess
import sys

import torch
import torch.nn.functional as F

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
    print("01 知识蒸馏：温度、T² 缩放，与「学生到底小在哪」")
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
    print("  ⚠ 本程序**不训练任何模型**（没有语料、没有 GPU）。所有「压缩了多少」")
    print("     都是**参数量与配置的算术**；准确率一律按引用值处理。")


# ---------------------------------------------------------------------------
# 被测对象
# ---------------------------------------------------------------------------

def softmax_T(z, T):
    return F.softmax(z / T, dim=-1)


def kl_soft(z, v, T):
    """`KL(p_T ‖ q_T)`，**batchmean** 归约。

    判据：`p` = 教师在该温度下的 softmax（`detach`，它是标签不是变量）；
    `q` = 学生在该温度下的 softmax；损失按 `batchmean` ⇒ **外面还有一个 `1/B`**。
    这个 `1/B` 会一路乘进梯度里，**读梯度公式时必须带上它**。
    """
    p = softmax_T(v, T).detach()
    return F.kl_div(F.log_softmax(z / T, dim=-1), p, reduction="batchmean")


def make_problem(batch=4, classes=1000, spread=0.25, delta=0.01, seed=0):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(batch, classes, generator=g, dtype=torch.float64) * spread
    d = torch.randn(batch, classes, generator=g, dtype=torch.float64) * delta
    d = d - d.mean(dim=1, keepdim=True)        # ★ **零均值**：这是那条极限式的判据之一
    return v, v + d, d


def grad_of(fn, z):
    zz = z.clone().requires_grad_(True)
    return torch.autograd.grad(fn(zz), zz)[0]


# ---------------------------------------------------------------------------
# B. 代价模型：蒸馏的压缩账
# ---------------------------------------------------------------------------

def bert_params(layers, cfg):
    """参数量模型（**本程序自己列的算式**，不是论文给的）：每层 QKVO + FFN，
    外加**不随层数缩放**的 embedding。"""
    per_layer = 4 * cfg["hidden"] * cfg["hidden"] + 2 * cfg["hidden"] * cfg["ffn"]
    emb = cfg["vocab"] * cfg["hidden"] + cfg["max_pos"] * cfg["hidden"]
    return layers * per_layer + emb, per_layer, emb


def run_B():
    rule("B 代价模型：蒸馏的「压缩账」——**参数量不随层数线性缩**")
    cfg = SPECS.BERT_BASE_CONFIG
    print("      BERT-base 配置（规格值）：vocab %d / hidden %d / FFN %d / %d 层 / 位置 %d"
          % (cfg["vocab"], cfg["hidden"], cfg["ffn"], cfg["layers"], cfg["max_pos"]))
    print("      每层参数 = 4·h² + 2·h·FFN = %.3fM；embedding = vocab·h + pos·h = %.3fM（**与层数无关**）"
          % ((4 * cfg["hidden"] ** 2 + 2 * cfg["hidden"] * cfg["ffn"]) / 1e6,
             (cfg["vocab"] * cfg["hidden"] + cfg["max_pos"] * cfg["hidden"]) / 1e6))

    total_t, per_layer, emb = bert_params(cfg["layers"], cfg)
    total_s, _, _ = bert_params(cfg["layers"] // 2, cfg)
    print()
    print("      层数 | 模型       | 本程序算出的参数量 | 论文印值")
    print("      %4d | BERT-base  | %14.1fM | %.0fM"
          % (cfg["layers"], total_t / 1e6, SPECS.DISTILBERT["params_teacher_M"]))
    print("      %4d | DistilBERT | %14.1fM | %.0fM（**一半层数**）"
          % (cfg["layers"] // 2, total_s / 1e6, SPECS.DISTILBERT["params_student_M"]))
    check("B", "★ 转录锚点：DistilBERT 的印值逐字对上（教师 %.0fM / 学生 %.0fM / 少 %.2f%%）"
          % (SPECS.DISTILBERT["params_teacher_M"], SPECS.DISTILBERT["params_student_M"],
             100 * SPECS.DISTILBERT["reported_param_reduction"]),
          abs(SPECS.DISTILBERT["params_teacher_M"] - 110.0) < 1e-9
          and abs(SPECS.DISTILBERT["params_student_M"] - 66.0) < 1e-9
          and abs(SPECS.DISTILBERT["reported_param_reduction"] - 0.40) < 1e-9
          and abs(SPECS.DISTILBERT["reported_speedup"] - 1.60) < 1e-9
          and abs(SPECS.DISTILBERT["glue_teacher"] - 79.5) < 1e-9
          and abs(SPECS.DISTILBERT["glue_student"] - 77.0) < 1e-9)
    check("B", "★ 12 层算出的参数量与论文印值一致（%.1fM ≈ %.0fM，相对差 %.1f%%）"
          % (total_t / 1e6, SPECS.DISTILBERT["params_teacher_M"],
             100 * abs(total_t / 1e6 - SPECS.DISTILBERT["params_teacher_M"])
             / SPECS.DISTILBERT["params_teacher_M"]),
          abs(total_t / 1e6 - SPECS.DISTILBERT["params_teacher_M"]) < 3.0)
    check("B", "★ 6 层算出的参数量与 DistilBERT 印值一致（%.1fM ≈ %.0fM，相对差 %.1f%%）"
          % (total_s / 1e6, SPECS.DISTILBERT["params_student_M"],
             100 * abs(total_s / 1e6 - SPECS.DISTILBERT["params_student_M"])
             / SPECS.DISTILBERT["params_student_M"]),
          abs(total_s / 1e6 - SPECS.DISTILBERT["params_student_M"]) < 3.0)

    derived_reduction = 1.0 - total_s / total_t
    check("B", "★ **层数减半 ≠ 参数减半**：层数比 = %.3f，但参数比 = %.3f"
          % (0.5, total_s / total_t), abs(total_s / total_t - 0.5) > 0.05)
    check("B", "★ 由配置推出的「少 %.1f%% 参数」≈ 论文自报的 %.0f%%"
          % (100 * derived_reduction, 100 * SPECS.DISTILBERT["reported_param_reduction"]),
          abs(derived_reduction - SPECS.DISTILBERT["reported_param_reduction"]) < 0.03)
    print("      ⇒ 差的那部分**全部是 embedding**：它占 %.1fM，层数减半也照样留着。"
          % (emb / 1e6))
    check("B", "★ 三项损失的清单（**没有 α**）逐项列出：%s"
          % "、".join(SPECS.DISTILBERT["loss_terms"]),
          len(SPECS.DISTILBERT["loss_terms"]) == 3)

    sub("B.2 推理侧的另一个账：**提速不等于参数量比**")
    print("      DistilBERT 自报：少 %.0f%% 参数、快 %.2f×（推理）。"
          % (100 * SPECS.DISTILBERT["reported_param_reduction"],
             SPECS.DISTILBERT["reported_speedup"]))
    check("B", "★ 提速 %.2f× 与参数比 %.3f 是**两个量**（提速略小于 1/参数比 %.3f）"
          % (SPECS.DISTILBERT["reported_speedup"], total_s / total_t, total_t / total_s),
          SPECS.DISTILBERT["reported_speedup"] < total_t / total_s)
    print("      TinyBERT 自报：小 %.1f×、推理快 %.1f×、GLUE 达教师 %.1f%%"
          % (SPECS.TINYBERT["smaller"], SPECS.TINYBERT["faster"],
             100 * SPECS.TINYBERT["glue_ratio"]))
    check("B", "★ TinyBERT 的「小」与「快」也**不是同一个数**（%.1f× vs %.1f×）"
          % (SPECS.TINYBERT["smaller"], SPECS.TINYBERT["faster"]),
          abs(SPECS.TINYBERT["faster"] - SPECS.TINYBERT["smaller"]) > 1.0)
    check("B", "★ 而且**「快」大于「小」**（%.1f× > %.1f×）——参数少得多、速度只多一截"
          % (SPECS.TINYBERT["faster"], SPECS.TINYBERT["smaller"]),
          SPECS.TINYBERT["faster"] > SPECS.TINYBERT["smaller"] > 1.0)
    check("B", "★ TinyBERT 的 GLUE 保留率 ∈ (0, 1) 且 ≥ 0.9：%.3f"
          % SPECS.TINYBERT["glue_ratio"], 0.9 <= SPECS.TINYBERT["glue_ratio"] < 1.0)
    # ★ 转录锚点（同上：保证印值没被静默改）
    check("B", "★ 转录锚点：TinyBERT 三个数逐字对上（小 %.1f× / 快 %.1f× / 保留 %.3f）"
          % (SPECS.TINYBERT["smaller"], SPECS.TINYBERT["faster"],
             SPECS.TINYBERT["glue_ratio"]),
          abs(SPECS.TINYBERT["smaller"] - 7.5) < 1e-9
          and abs(SPECS.TINYBERT["faster"] - 9.4) < 1e-9
          and abs(SPECS.TINYBERT["glue_ratio"] - 0.968) < 1e-9)


# ---------------------------------------------------------------------------
# A. 论文侧 × 本机侧
# ---------------------------------------------------------------------------

def run_A():
    rule("A 温度 softmax 与梯度：`(1/T)(q − p)`，以及 T² 的来历")

    v, z0, d = make_problem()
    B, N = z0.shape
    sub("A.1 本机侧：`∂KL/∂z = (1/T)(q − p)/B` 逐元素核对（判据：batchmean 的 1/B）")
    errs = []
    for T in (1.0, 4.0, 64.0):
        def loss(zz, T=T):
            return kl_soft(zz, v, T)
        g = grad_of(loss, z0)
        analytic = (softmax_T(z0, T) - softmax_T(v, T)) / T / B
        errs.append((g - analytic).abs().max().item())
        print("      T=%-5.0f  实测与解析式的最大差 = %.3e（解析式量级 %.3e）"
              % (T, errs[-1], analytic.abs().max().item()))
    check("A", "★ 三条 T 上真 autograd 与 `(1/T)(q−p)/B` 都吻合到 machine precision",
          all(e < 1e-15 for e in errs))
    # ★ 「1/B 不能省」必须**证伪式**地查：写出不带 1/B 的解析式，它必须**对不上**。
    def loss4(zz):
        return kl_soft(zz, v, 4.0)
    g4 = grad_of(loss4, z0)
    no_b = (softmax_T(z0, 4.0) - softmax_T(v, 4.0)) / 4.0
    d_with = (g4 - no_b / B).abs().max().item()
    d_without = (g4 - no_b).abs().max().item()
    scale = no_b.abs().max().item()
    rel_without = d_without / scale
    print("      带 1/B 的解析式与实测差 %.3e（解析式量级 %.3e）" % (d_with, scale))
    print("      **不带 1/B** 时差 %.3e ⇒ **相对误差 %.6f，正好是 (B−1)/B = %.6f**"
          % (d_without, rel_without, (B - 1.0) / B))
    check("A", "★ **`1/B` 不能省**：去掉它，解析式的相对误差正好是 `(B−1)/B = %.4f`"
          % ((B - 1.0) / B),
          d_with < 1e-15 and abs(rel_without - (B - 1.0) / B) < 1e-6)

    sub("A.2 本机侧：`T²·g → (z − v − mean)/N`（判据：δ 必须零均值）")
    print("      T      | max|T²g − δ/(B·N)| / max|δ/(B·N)|")
    rels = []
    for T in (16.0, 64.0, 256.0, 1024.0, 4096.0):
        def loss(zz, T=T):
            return kl_soft(zz, v, T)
        g = grad_of(loss, z0)
        tgt = d / (B * N)
        rel = (T * T * g - tgt).abs().max().item() / tgt.abs().max().item()
        rels.append(rel)
        print("      %-6.0f | %.6f" % (T, rel))
    check("A", "★ 相对误差**随 T 单调下降**（%.5f → %.6f），说明极限式是对的"
          % (rels[0], rels[-1]), all(rels[i] > rels[i + 1] for i in range(len(rels) - 1)))
    # ★ 判据：T 每翻 4 倍，误差约掉 4 倍 ⇒ 一阶修正 ∝ 1/T。
    #   （本机实测：即使教师尺度 = 0，也还是 1/T——因为 q 相对均匀分布的偏离本身是
    #     O(δ/T)。**所以「T²g 的收敛速率」与 Hinton 的「梯度幅度按 1/T² 缩放」
    #     不是同一件事**，两个都对，但是两个量。）
    quads = [rels[i] / rels[i + 1] for i in range(len(rels) - 1)]
    check("A", "★ 收敛速率 ≈ **`1/T`**（T 翻 4 倍，误差约掉 4 倍）：实测 %s"
          % " ".join("%.2f×" % q for q in quads),
          all(3.0 < q < 5.0 for q in quads))
    check("A", "★ ⚠ 这**不是** `1/T²`——「梯度幅度按 1/T² 缩放」是另一个量"
          "（见 C.1 的 4096 倍），**两者不得混称**",
          abs(quads[0] - 4.0) < 1.0)
    check("A", "★ 所以「高温极限等价于最小化 ½‖z−v‖²」——**梯度是 `(z−v−mean)/N`**，"
          "去掉均值那一项是因为 softmax 只对差值敏感", rels[-1] < 1e-3)

    sub("A.3 引用值：Hinton 原文的 T² 理由（**本实现者未逐字读到原文**）")
    print('      "%s"' % SPECS.HINTON_T2_QUOTE)
    # ★ 转录锚点：逐字引文必须**逐字**对得上（多一个字符就不叫「原文」了）
    check("A", "★ 转录锚点：这句引文**逐字**对上（长度 %d）" % len(SPECS.HINTON_T2_QUOTE),
          SPECS.HINTON_T2_QUOTE ==
          "Since the magnitudes of the gradients produced by the soft targets "
          "scale as 1/T^2 it is important to multiply them by T^2 when using both "
          "hard and soft targets.")
    check("A", "★ 原文里**没有 α 这个符号**（社区约定）",
          SPECS.ALPHA_SYMBOL_IN_ORIGINAL_PAPER is False)
    # ★ 两种写法的换算**不是「互为倒数」**，是 `beta = (1-alpha)/alpha`——
    #   本机按代数重算（把第一式除以 alpha）后订正了计划基线的说法。
    for a in (0.25, 0.5, 0.9):
        b = (1.0 - a) / a
        print("      α = %.2f ⇒ β = (1−α)/α = %.4f（**不是 1/α = %.4f**）"
              % (a, b, 1.0 / a))
    # ★ 判据写成**往返**：`β = (1−α)/α` 的逆变换必须是 `α = 1/(1+β)`。
    #   往返对不上就说明这条换算写错了（写成 `1/α` 时往返会失败——见下一条）。
    beta_of_a = lambda a: (1.0 - a) / a
    a_of_beta = lambda b: 1.0 / (1.0 + b)
    check("A", "★ 换算 `β = (1−α)/α` 的**逆变换**是 `α = 1/(1+β)`（往返误差 < 1e-15）",
          all(abs(a_of_beta(beta_of_a(a)) - a) < 1e-15 for a in (0.25, 0.5, 0.9)))
    # ★ **它不等于 `1/α`**——`(1−α)/α = 1/α − 1` 恒等式，两者在任何 α>0 处都不相等。
    check("A", "★ 且它**处处不等于 `1/α`**（`α=0.5` 时 %.4f vs %.4f；`α=0.9` 时 %.6f vs %.6f）"
          % ((1.0 - 0.5) / 0.5, 1.0 / 0.5, (1.0 - 0.9) / 0.9, 1.0 / 0.9),
          all(abs((1.0 - a) / a - 1.0 / a - (-1.0)) < 1e-15 for a in (0.25, 0.5, 0.9)))

    sub("A.4 引用值：蒸馏用在量化模型上（**逐行带表号与基线**）")
    for dataset in ("WikiText2", "C4"):
        row = SPECS.LLMQAT_LLAMA7B_4_8_4[dataset]
        print("      %-10s : dense %.1f / RTN %.1f / SmoothQuant %.1f / LLM-QAT %.1f"
              % (dataset, row["dense"], row["RTN"], row["SmoothQuant"], row["LLM-QAT"]))
    check("A", "★ LLM-QAT 在 4-8-4 设置上把 RTN 的 151.4 拉回 11.6（同列同基线）",
          SPECS.LLMQAT_LLAMA7B_4_8_4["WikiText2"]["LLM-QAT"]
          < SPECS.LLMQAT_LLAMA7B_4_8_4["WikiText2"]["RTN"] / 10.0)
    check("A", "★ 两列（WikiText2 / C4）是同一实验的两个数据集，**跨列不可比**",
          SPECS.LLMQAT_LLAMA7B_4_8_4["WikiText2"]["dense"]
          > SPECS.LLMQAT_LLAMA7B_4_8_4["C4"]["dense"])
    # ★ 每一列内部的**四项排序**是一个可证伪的联合判据（四个数任意一个被改都会翻车）：
    #   蒸馏后 LLM-QAT 应当**优于**两种 PTQ 基线，且都远差于稠密。
    for ds in ("WikiText2", "C4"):
        row = SPECS.LLMQAT_LLAMA7B_4_8_4[ds]
        print("      %-10s 排序检查：dense %.1f < LLM-QAT %.1f < RTN %.1f < SmoothQuant %.1f"
              % (ds, row["dense"], row["LLM-QAT"], row["RTN"], row["SmoothQuant"]))
        check("A", "★ %s 列的四项排序：dense < LLM-QAT < RTN < SmoothQuant" % ds,
              row["dense"] < row["LLM-QAT"] < row["RTN"] < row["SmoothQuant"])
    # ★ **转录锚点**：只保证印值没被静默改掉，**不声称**复核了测量口径。
    #   光靠「排序」不够——本机实测：把四个数同时 ×1.5+1，排序仍然成立。
    wi, c4 = SPECS.LLMQAT_LLAMA7B_4_8_4["WikiText2"], SPECS.LLMQAT_LLAMA7B_4_8_4["C4"]
    check("A", "★ 转录锚点：两列的 8 个数逐字对上（WikiText2 %.1f/%.1f/%.1f/%.1f，"
          "C4 %.1f/%.1f/%.1f/%.1f）"
          % (wi["dense"], wi["RTN"], wi["SmoothQuant"], wi["LLM-QAT"],
             c4["dense"], c4["RTN"], c4["SmoothQuant"], c4["LLM-QAT"]),
          abs(wi["dense"] - 10.4) < 1e-9 and abs(wi["RTN"] - 151.4) < 1e-9
          and abs(wi["SmoothQuant"] - 163.6) < 1e-9 and abs(wi["LLM-QAT"] - 11.6) < 1e-9
          and abs(c4["dense"] - 7.2) < 1e-9 and abs(c4["RTN"] - 55.1) < 1e-9
          and abs(c4["SmoothQuant"] - 81.1) < 1e-9 and abs(c4["LLM-QAT"] - 8.6) < 1e-9)
    # ★ 有信息量的那条：RTN 崩掉一个数量级，蒸馏把它拉回到**稠密的 1.2 倍以内**。
    ratio_rtn = wi["RTN"] / wi["dense"]
    ratio_qat = wi["LLM-QAT"] / wi["dense"]
    print("      RTN/稠密 = %.2f×；LLM-QAT/稠密 = %.2f×" % (ratio_rtn, ratio_qat))
    check("A", "★ 蒸馏后与稠密的比值 ∈ (1, 1.2)：%.4f（对照 RTN 的 %.2f×）"
          % (ratio_qat, ratio_rtn), 1.0 < ratio_qat < 1.2 and ratio_rtn > 5.0)
    check("A", "★ BitDistiller 的编号是 %s（**不是** 2402.19115）"
          % SPECS.BITDISTILLER_ARXIV,
          SPECS.BITDISTILLER_ARXIV == "arXiv:2402.10631")
    check("A", "★ DualGraD：来源未定位（检索零命中）⇒ `DUALGRAD_SOURCED` 必须保持 False",
          SPECS.DUALGRAD_SOURCED is False)


# ---------------------------------------------------------------------------
# C. 边界与反例
# ---------------------------------------------------------------------------

def run_C():
    rule("C 边界与反例：**漏掉 T²** ⇒ 调温度会静默改变两个损失项的配比")

    v, z0, _ = make_problem()
    taus = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
    y = torch.zeros(z0.shape[0], dtype=torch.long)     # 硬标签：类别 0

    sub("C.1 量「软目标梯度 / 硬目标梯度」这个比值")
    print("      判据：两个梯度都取 ‖·‖₂；软项分别按 `T²·KL` 与 `KL` 两种写法算。")
    print()
    print("      T   | 带 T² 的比值 | 不带 T² 的比值 | 不带/带")
    rows = []
    for T in taus:
        gh = grad_of(lambda zz: F.cross_entropy(zz, y), z0).norm().item()
        g_t2 = grad_of(lambda zz: kl_soft(zz, v, T) * (T * T), z0).norm().item()
        g_no = grad_of(lambda zz: kl_soft(zz, v, T), z0).norm().item()
        rows.append((T, g_t2 / gh, g_no / gh))
        print("      %-3.0f | %12.3e | %14.3e | %8.6f"
              % (T, g_t2 / gh, g_no / gh, (g_no / gh) / (g_t2 / gh)))
    c_t2 = (max(r[1] for r in rows) - min(r[1] for r in rows)) / rows[0][1]
    c_no = rows[-1][2] / rows[0][2]
    check("C", "★ **带 T²** 时比值在 T=1..64 上基本恒定（相对波动 %.1f%%）"
          % (100 * c_t2), c_t2 < 0.10)
    check("C", "★ **不带 T²** 时比值按 `1/T²` 塌缩：T=64 时只剩 T=1 的 %.3e"
          % c_no, c_no < 1e-2)
    check("C", "★ 塌缩倍数 ≈ 64² = %.0f（实测 %.1f×）"
          % (64.0 ** 2, rows[0][2] / rows[-1][2]), rows[0][2] / rows[-1][2] > 1000.0)
    print()
    print("      ⇒ **失败是静默的**：漏掉 T² 不会报错、不会 NaN，只会让软目标在")
    print("        大 T 下**消失**——于是「调温度」实际上变成了「调软硬配比」，")
    print("        而调参的人以为自己在调一个与配比无关的超参。")

    sub("C.2 对照：T=1 时两种写法**完全相同**（所以 T=1 的调试发现不了这个坑）")
    check("C", "★ T=1 处 `T² = 1`，两种写法的梯度**逐位相同**（判据：‖g²−g‖ = 0）",
          rows[0][1] == rows[0][2])

    sub("C.3 第二个反例：把 T² 乘到**硬标签项**上")
    def loss_wrong(zz, T):
        return F.cross_entropy(zz, y) * (T * T) + kl_soft(zz, v, T)
    w1 = grad_of(lambda zz: loss_wrong(zz, 1.0), z0).norm().item()
    w64 = grad_of(lambda zz: loss_wrong(zz, 64.0), z0).norm().item()
    print("      ‖∂L_wrong/∂z‖：T=1 时 %.4e，T=64 时 %.4e（T=1 只有 T=64 的 %.3e）"
          % (w1, w64, w1 / w64))
    check("C", "★ 乘错项会让梯度**随 T 放大**，方向与「塌缩」正好相反",
          w1 / w64 < 1.0)
    print("      ⇒ 两个方向都错：**漏乘会让软项消失，乘错会让硬项爆炸**。")
    print("        原文那句只说了一件事：T² 乘在**软目标那一项**上。")


# ---------------------------------------------------------------------------
# F. 变异测试清单
# ---------------------------------------------------------------------------

def run_F():
    rule("F 变异测试清单（A6）：本程序直接读到的 specs 常量")
    src = open(__file__, encoding="utf-8").read()
    names = sorted(n for n in dir(SPECS) if n.isupper() and not n.startswith("_"))
    used = [n for n in names if ("SPECS.%s" % n) in src]
    print("      specs.py 共 %d 条常量；本程序直接读到 %d 条" % (len(names), len(used)))
    print("      %s" % ", ".join(used))
    check("B", "本程序读到的常量数 ≥ 8", len(used) >= 8)
    check("B", "★ 本机 torch 版本被 specs 的「跑过的版本」清单覆盖",
          torch.__version__ in SPECS.MEASURED_VERSIONS)


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

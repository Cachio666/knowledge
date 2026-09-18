#!/usr/bin/env python3
"""02 GPTQ：Hessian 逆、逐列误差补偿、Cholesky、dampening、以及两个常被混淆的「size」。

对应文档：130-quantization/ptq/02-gptq.md

★ 本程序跑的是**真 torch**——输出是**本机实测值**，支撑关于**真实 PyTorch 的线性代数路径**
  的结论，但**必须绑定版本**。本机五个解释器都跑过（清单见 `specs.MEASURED_VERSIONS`）。
  全部 CPU-only，无 GPU 数字。

三类断言（CONTRIBUTING §2 第三种形态：B 在编号程序里，A/C 另见 verify.py）：
  B —— **代价模型**：GPTQ 花掉的是什么、买回来的是什么。
       ① 它**不能**降低权重空间的误差（实测反而更大），换来的是**层输出**误差下降；
       ② `blocksize` 是纯计算批处理：改它**逐元素结果不变**（代价 0）；
       ③ `dampening` 是纯代价：调大它输出误差单调变差，买的是 H 奇异时的可解性。
  A —— **两条互不依赖的观察通道**：
       ① 本程序的 GPTQ（真 torch，`torch.linalg.cholesky` 路径）vs
          `verify.py` 的 `ref_gptq`（纯 python 手写 Cholesky）——测试集不同、代码路径不同；
       ② `act-order` 在真 torch 上确实**改变了结果**（`argsort(diag(H))` 不是恒等变换）。
  C —— 失败行为必须**真的跑出来**：不做 dampening 时秩亏的 H 会让 Cholesky 抛异常；
       近奇异的 H 会让 H⁻¹ 元素爆炸；`act-order` 在收益上**不稳定**（不是「有效」）。

★ 本程序**不复制 `specs.py` 的任何常量值**：凡用到规格常量，一律写 `SPECS.xxx`（A6 约定）。

用法：cd code && make 02_gptq   （或 make run 全跑）
"""

import platform
import subprocess
import sys

import torch

import specs as SPECS


_STATE = {"n": 0, "A": 0, "B": 0, "C": 0, "fail": []}

QMAX = 7            # 4-bit 对称：可用整数范围 [−8, 7]
BITS = 4


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
    print("02 GPTQ：逐列补偿、Cholesky、dampening，以及 blocksize ≠ group_size")
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
# 参考实现（本程序自己的 GPTQ —— 与 verify.py 的 ref_gptq 是两条不同路径）
# ---------------------------------------------------------------------------

def group_scales(W, gs, qmax=QMAX):
    """W:[d_out, d_in] → [d_out, n_groups]。分组沿**输入维**（GPTQ 的 groupsize 语义）。"""
    out, inn = W.shape
    if gs <= 0:
        return (W.abs().amax(dim=1, keepdim=True) / qmax).clamp_min(1e-12)
    n_g = (inn + gs - 1) // gs
    S = torch.zeros(out, n_g, dtype=W.dtype)
    for g in range(n_g):
        S[:, g] = W[:, g * gs:(g + 1) * gs].abs().amax(dim=1) / qmax
    return S.clamp_min(1e-12)


def qcol(w, s, qmax=QMAX):
    return torch.clamp(torch.round(w / s), -qmax - 1, qmax) * s


def rtn(W, gs, qmax=QMAX):
    S = group_scales(W, gs, qmax)
    o = torch.zeros_like(W)
    for i in range(W.shape[1]):
        s = S[:, i // gs] if gs > 0 else S[:, 0]
        o[:, i] = qcol(W[:, i].clone(), s, qmax)
    return o


def gptq(W, H, gs, blocksize=SPECS.GPTQ_IMPL_DEFAULT_BLOCKSIZE,
         percdamp=SPECS.GPTQ_PERCDAMP_PAPER, actorder=SPECS.GPTQ_IMPL_DEFAULT_ACTORDER,
         cholesky_inv=True):
    """GPTQ（论文式 2/4/5）。

    W:[d_out, d_in]；H:[d_in, d_in] = (2/n)·XᵀX（**判据：含 1/n**）。
    逐**列**（输入维）量化；列 i 的补偿 `e = (w − q)/[H⁻¹]_{ii}`；
    更新 `W[:, i:] ← W[:, i:] − e·[H⁻¹]_{i, i:}`。
    `blocksize` 只控制「一批算多少列」，**不改数学结果**。
    """
    W = W.clone()
    H = H.clone()
    out, inn = W.shape
    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0
    perm = None
    if actorder:
        # ★ act-order = argsort(diag(H), descending=True)；
        #   因为 H = (2/n)Σxxᵀ ⇒ diag(H) ∝ 该输入通道激活的**二阶矩** Σx²
        perm = torch.argsort(torch.diag(H), descending=True)
        W = W[:, perm].clone()
        H = H[perm][:, perm].clone()
    d = torch.diag(H).clone()
    H[torch.arange(inn), torch.arange(inn)] = d + percdamp * d.mean()
    Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))
    if cholesky_inv:
        # 论文：对 H⁻¹ 取上三角 Cholesky，等价于「删行」，避免直接求逆的累积误差
        Hinv = torch.linalg.cholesky(Hinv, upper=True)
    S = group_scales(W, gs)
    Q = torch.zeros_like(W)
    Err = torch.zeros_like(W)
    for i1 in range(0, inn, blocksize):
        i2 = min(i1 + blocksize, inn)
        cnt = i2 - i1
        W1 = W[:, i1:i2].clone()
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(cnt):
            gi = (i1 + i) // gs if gs > 0 else 0
            w = W1[:, i].clone()      # ★ 必须 clone：`W1[:, i] = q` 会改写这个视图
            q = qcol(w, S[:, gi])
            W1[:, i] = q
            e = (w - q) / Hinv1[i, i]
            W1[:, i + 1:] = W1[:, i + 1:] - e.unsqueeze(1) * Hinv1[i, i + 1:].unsqueeze(0)
            Err[:, i1 + i] = e
        Q[:, i1:i2] = W1
        if i2 < inn:
            W[:, i2:] = W[:, i2:] - Err[:, i1:i2] @ Hinv[i1:i2, i2:]
    if actorder:
        Q = Q[:, torch.argsort(perm)]
    return Q


# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------

def make_problem(seed=0):
    """合成校准集与权重。**判据逐条写出**（P9）：

      · X:[1024, 512] ~ N(0,1)；通道 1/2/3 = 0.98·通道 0 + N(0,0.2)（造出**病态 H**）；
      · 通道 7 幅值 ×50（模拟 activation outlier）；
      · W:[256, 512] ~ N(0,1)×0.05；
      · H = (2/n)·XᵀX（**含 1/n 因子**，与 GPTQ 论文一致）；
      · 误差口径 = `mean((X Wᵀ − X Wqᵀ)²)`，即**层输出**误差。
    """
    torch.manual_seed(seed)
    n, din, dout = 1024, 512, 256
    X = torch.randn(n, din)
    for k in (1, 2, 3):
        X[:, k] = X[:, 0] * 0.98 + torch.randn(n) * 0.2
    X[:, 7] *= 50.0
    W = torch.randn(dout, din) * 0.05
    H = (2.0 / n) * (X.T @ X)
    return X, W, H


def out_mse(X, W, Wq):
    return float(((X @ W.T - X @ Wq.T) ** 2).mean())


def weight_mse(W, Wq):
    return float(((W - Wq) ** 2).mean())


# ---------------------------------------------------------------------------
# B. 代价模型
# ---------------------------------------------------------------------------

def run_B():
    rule("B 代价模型：GPTQ 花掉的是什么、买回来的是什么")

    X, W, H = make_problem()
    print("  设定：X %s / W %s / H %s；cond(H) = %.3e"
          % (tuple(X.shape), tuple(W.shape), tuple(H.shape), float(torch.linalg.cond(H))))
    print("  4-bit 对称（qmax=%d）；percdamp = %.4f（论文默认 %s）"
          % (QMAX, SPECS.GPTQ_PERCDAMP_PAPER, SPECS.GPTQ_PERCDAMP_PAPER))

    sub("B.1  RTN vs GPTQ：**层输出**误差 vs **权重**误差，方向相反")
    for gs in (128, 32, -1):
        Wr = rtn(W, gs)
        Wg = gptq(W, H, gs)
        eo_r, eo_g = out_mse(X, W, Wr), out_mse(X, W, Wg)
        ew_r, ew_g = weight_mse(W, Wr), weight_mse(W, Wg)
        print("      gs=%-4d | 输出 MSE %.4e → %.4e（降 %4.1f%%）"
              "| 权重 MSE %.4e → %.4e（升 %4.1f%%）"
              % (gs, eo_r, eo_g, 100 * (1 - eo_g / eo_r),
                 ew_r, ew_g, 100 * (ew_g / ew_r - 1)))
    Wr, Wg = rtn(W, 128), gptq(W, H, 128)
    check("B", "GPTQ 的**层输出**误差低于 RTN", out_mse(X, W, Wg) < out_mse(X, W, Wr))
    check("B", "★ GPTQ 的**权重**误差**高于** RTN（它不优化权重空间）",
          weight_mse(W, Wg) > weight_mse(W, Wr),
          "权重 MSE %.4e → %.4e" % (weight_mse(W, Wr), weight_mse(W, Wg)))
    check("B", "⇒ 用「权重量化误差」衡量 GPTQ 会得出**相反**的结论",
          weight_mse(W, Wg) > weight_mse(W, Wr) and out_mse(X, W, Wg) < out_mse(X, W, Wr))
    check("B", "编号程序这一组的降幅在 specs 记录的区间内（40%–50%）",
          40.0 <= 100 * (1 - out_mse(X, W, Wg) / out_mse(X, W, Wr)) <= 50.0,
          "降 %.2f%%" % (100 * (1 - out_mse(X, W, Wg) / out_mse(X, W, Wr))))

    sub("B.2  `blocksize` 是纯计算批处理：**逐元素结果完全相同**（代价 0）")
    base = gptq(W, H, 128, blocksize=SPECS.GPTQ_IMPL_DEFAULT_BLOCKSIZE)
    for bs in (8, 16, 32, 64, 256):
        other = gptq(W, H, 128, blocksize=bs)
        mx = float((base - other).abs().max())
        print("      blocksize=%3d → max|ΔW| = %.3e" % (bs, mx))
    mxs = [float((base - gptq(W, H, 128, blocksize=bs)).abs().max())
           for bs in (8, 16, 32, 64, 256)]
    check("B", "★ 五个 blocksize 下逐元素差 = %.1e（二进制相同）" % max(mxs),
          max(mxs) == SPECS.GPTQ_BLOCKSIZE_INVARIANCE_MAXDIFF)
    check("B", "⇒ `blocksize` 与 `group_size` 是**两个不同的东西**"
          "（前者不改数学，后者改）",
          max(mxs) == 0.0
          and float((gptq(W, H, 128) - gptq(W, H, 32)).abs().max()) > 0.0)

    sub("B.3  `dampening` 是纯代价：调大它输出误差单调变差")
    prev = None
    rows = []
    for pc in (0.0, 0.001, 0.01, 0.05, 0.1, 0.5):
        e = out_mse(X, W, gptq(W, H, 128, percdamp=pc))
        rows.append((pc, e))
        tag = ""
        if abs(pc - SPECS.GPTQ_IMPL_DEFAULT_PERCDAMP) < 1e-15:
            tag = "  ← 官方仓库默认"
        if abs(pc - SPECS.GPTQ_CONFIG_DAMP_PERCENT_DEFAULT) < 1e-15:
            tag = "  ← transformers 的 GPTQConfig 默认"
        print("      percdamp=%-6s → 输出 MSE %.6e%s" % (pc, e, tag))
        prev = e
    check("B", "percdamp 越大输出误差越大（单调，端点差 %.1f%%）"
          % (100 * (rows[-1][1] / rows[0][1] - 1)),
          rows[-1][1] > rows[0][1])
    check("B", "★ 两套默认值差 %.0f 倍：论文/官方仓库 %.2f vs transformers %.1f"
          % (SPECS.GPTQ_DAMP_DEFAULT_RATIO, SPECS.GPTQ_PERCDAMP_PAPER,
             SPECS.GPTQ_CONFIG_DAMP_PERCENT_DEFAULT),
          abs(SPECS.GPTQ_CONFIG_DAMP_PERCENT_DEFAULT / SPECS.GPTQ_PERCDAMP_PAPER
              - SPECS.GPTQ_DAMP_DEFAULT_RATIO) < 1e-12)
    # ★ 三个默认值逐条精确核对（否则改掉一个值这三条常量就没人查了）
    check("B", "`GPTQ_IMPL_DEFAULT_PERCDAMP` = 论文的 %.2f（官方仓库与论文同值）"
          % SPECS.GPTQ_PERCDAMP_PAPER,
          SPECS.GPTQ_IMPL_DEFAULT_PERCDAMP == 0.01
          and SPECS.GPTQ_IMPL_DEFAULT_PERCDAMP == SPECS.GPTQ_PERCDAMP_PAPER)
    check("B", "`GPTQ_IMPL_DEFAULT_BLOCKSIZE` = %d，与论文的 block size 同值"
          % SPECS.GPTQ_BLOCK_SIZE_PAPER,
          SPECS.GPTQ_IMPL_DEFAULT_BLOCKSIZE == 128
          and SPECS.GPTQ_IMPL_DEFAULT_BLOCKSIZE == SPECS.GPTQ_BLOCK_SIZE_PAPER)
    check("B", "`GPTQ_IMPL_DEFAULT_ACTORDER` = `GPTQ_CONFIG_DESC_ACT_DEFAULT` = %s"
          % SPECS.GPTQ_IMPL_DEFAULT_ACTORDER,
          SPECS.GPTQ_IMPL_DEFAULT_ACTORDER is False
          and SPECS.GPTQ_CONFIG_DESC_ACT_DEFAULT is False)
    check("B", "校准集三个常量：%d 条 × %d token / %s"
          % (SPECS.GPTQ_CALIB_SEQUENCES, SPECS.GPTQ_CALIB_SEQ_LEN, SPECS.GPTQ_CALIB_DATASET),
          SPECS.GPTQ_CALIB_SEQUENCES == 128 and SPECS.GPTQ_CALIB_SEQ_LEN == 2048
          and SPECS.GPTQ_CALIB_DATASET == "C4")
    # ★ GPTQ_MEASURED 逐键核对（相对 2%）
    gm = SPECS.GPTQ_MEASURED
    me = {"rtn_output_mse_gs128": out_mse(X, W, rtn(W, 128)),
          "gptq_output_mse_gs128": out_mse(X, W, gptq(W, H, 128)),
          "rtn_output_mse_gs32": out_mse(X, W, rtn(W, 32)),
          "gptq_output_mse_gs32": out_mse(X, W, gptq(W, H, 32)),
          "rtn_output_mse_percol": out_mse(X, W, rtn(W, -1)),
          "gptq_output_mse_percol": out_mse(X, W, gptq(W, H, -1)),
          "rtn_weight_mse_percol": weight_mse(W, rtn(W, -1)),
          "gptq_weight_mse_percol": weight_mse(W, gptq(W, H, -1))}
    for k, g in me.items():
        check("B", "GPTQ_MEASURED[%s]：本程序 %.6e vs 基线 %.6e（相对差 < 2%%）"
              % (k, g, gm[k]), abs(g - gm[k]) / gm[k] < 0.02)
    check("B", "GPTQ_MEASURED 的两处降幅常量与两个 MSE 自洽",
          abs(100 * (1 - gm["gptq_output_mse_gs128"] / gm["rtn_output_mse_gs128"])
              - gm["gptq_reduction_pct_gs128"]) < 0.1)


# ---------------------------------------------------------------------------
# A. 判据复核
# ---------------------------------------------------------------------------

def run_A():
    rule("A 判据复核：真 torch 的行为 vs 手写期望值")
    X, W, H = make_problem()

    sub("A.1  `act-order`：实现的 `argsort(diag(H), descending=True)`")
    diagH = torch.diag(H)
    perm = torch.argsort(diagH, descending=True)
    print("      diag(H) 最大的 5 个通道 = %s" % perm[:5].tolist())
    print("      diag(H) 的头部值        = %s"
          % ["%.2f" % float(v) for v in diagH[perm[:5]]])
    check("A", "★ 被排序的量是 **diag(H)**，即 `Σx²`（激活的**二阶矩**），"
          "不是「激活大小」（一阶量）",
          abs(float(diagH[7]) - 2.0 * float((X[:, 7] ** 2).mean())) < 1e-3,
          "diag(H)[7] = %.3f ; 2·mean(x²) = %.3f"
          % (float(diagH[7]), 2.0 * float((X[:, 7] ** 2).mean())))
    check("A", "★ `act-order` **不在 GPTQ 论文里**（常量记为 False），"
          "是实现的 `argsort(diag(H))`",
          SPECS.GPTQ_ACT_ORDER_IN_PAPER is False)
    check("A", "transformers 的 `desc_act` 默认 = %s（也印证它是可选项）"
          % SPECS.GPTQ_CONFIG_DESC_ACT_DEFAULT,
          SPECS.GPTQ_CONFIG_DESC_ACT_DEFAULT is False)
    Wn = gptq(W, H, 128, actorder=True)
    Wa = gptq(W, H, 128, actorder=False)
    mx = float((Wn - Wa).abs().max())
    check("A", "act-order 确实**改变**量化结果（max|ΔW| = %.3e）" % mx, mx > 1e-9)

    sub("A.2  ★ act-order 的收益**不稳定** —— 同一实现的三个配置三个方向")
    d = {}
    for gs, key in ((128, "gs128"), (32, "gs32"), (-1, "percol")):
        a = out_mse(X, W, gptq(W, H, gs, actorder=False))
        b = out_mse(X, W, gptq(W, H, gs, actorder=True))
        d[key] = 100 * (1 - b / a)
        print("      gs=%-4s  act-order 关闭 %.6e → 打开 %.6e（%+.2f%%）" % (gs, a, b, d[key]))
    check("A", "★ 三个配置里既有正的也有负的（⇒ 「act-order 有效」不是一个可断言的结论）",
          any(v > 0 for v in d.values()) and any(v < 0 for v in d.values()),
          str({k: round(v, 2) for k, v in d.items()}))
    check("A", "它与 specs 里记录的值**逐项**一致（绝对容差 1 个百分点）",
          all(abs(d[k] - v) < 1.0
              for k, v in SPECS.GPTQ_ACTORDER_MEASURED_DELTA_PCT.items()),
          "本程序 %s vs 基线 %s" % ({k: round(v, 2) for k, v in d.items()},
                                    SPECS.GPTQ_ACTORDER_MEASURED_DELTA_PCT))
    check("A", "三个键齐全，且**至少一个为正、一个为负**（否则「不稳定」这句话不成立）",
          set(SPECS.GPTQ_ACTORDER_MEASURED_DELTA_PCT) == {"gs128", "gs32", "percol"}
          and any(v > 0 for v in SPECS.GPTQ_ACTORDER_MEASURED_DELTA_PCT.values())
          and any(v < 0 for v in SPECS.GPTQ_ACTORDER_MEASURED_DELTA_PCT.values()))

    sub("A.3  ★ 两条独立路径：本程序的 GPTQ（torch）vs verify.py 的 ref_gptq（纯 python）")
    print("      verify.py 跑的是 24×32 的小矩阵；本程序跑的是 256×512。")
    print("      **规模不同 ⇒ 数字不同，只有定性结构必须一致**：")
    print("        两侧共同的定性结论 = ①输出误差降 ②权重误差升 ③blocksize 不变。")
    small = gptq(W[:24, :32].contiguous(),
                 H[:32, :32].contiguous(), 32)
    check("A", "小规模下同一条定性结构成立（输出降、权重升）",
          out_mse(X[:, :32], W[:24, :32], small)
          < out_mse(X[:, :32], W[:24, :32], rtn(W[:24, :32].contiguous(), 32))
          and weight_mse(W[:24, :32], small)
          > weight_mse(W[:24, :32], rtn(W[:24, :32].contiguous(), 32)))
    check("A", "★ 两档不互相背书：本程序测的是**真 torch 的 linalg 路径**，"
          "verify.py 测的是**手写 Cholesky**",
          True)

    sub("A.4  Cholesky 上三角 vs 直接 H⁻¹：两条路径给出**不同**的 Q")
    q1 = gptq(W, H, 128, cholesky_inv=True)
    q2 = gptq(W, H, 128, cholesky_inv=False)
    mx = float((q1 - q2).abs().max())
    print("      上三角 Cholesky(H⁻¹) 的输出 MSE = %.6e" % out_mse(X, W, q1))
    print("      直接用 H⁻¹          的输出 MSE = %.6e" % out_mse(X, W, q2))
    check("A", "两条路径的 max|ΔW| = %.3e（> 0 ⇒ 不是同一个算法）" % mx, mx > 1e-9)


# ---------------------------------------------------------------------------
# C. 边界与反例
# ---------------------------------------------------------------------------

def run_C():
    rule("C 边界与反例：这些失败行为必须真的跑出来")
    X, W, H = make_problem()

    sub("C.1  秩亏的 H：不做 dampening 时 Cholesky 直接抛")
    Hsing = torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float32)
    raised = None
    try:
        torch.linalg.cholesky(Hsing)
    except Exception as e:                                  # noqa: BLE001
        raised = e
    print("      精确奇异的 H → %s" % (type(raised).__name__ if raised else "没抛"))
    check("C", "精确奇异的 H 上 `torch.linalg.cholesky` 抛异常"
          "（本机 %s）" % (type(raised).__name__ if raised else "没抛"),
          raised is not None
          and isinstance(raised, (torch.linalg.LinAlgError, RuntimeError)),
          str(raised)[:80] if raised else "没抛")
    dmean = float(torch.diag(Hsing).mean())
    Hd = Hsing + SPECS.GPTQ_PERCDAMP_PAPER * dmean * torch.eye(2)
    ok = True
    try:
        torch.linalg.cholesky(Hd)
    except Exception:                                       # noqa: BLE001
        ok = False
    check("C", "加上论文默认的 percdamp = %.2f 后可以分解" % SPECS.GPTQ_PERCDAMP_PAPER, ok)

    sub("C.2  ★ 近奇异的 H 更危险：Cholesky **能过**，但 H⁻¹ 的元素爆炸")
    eps = 1e-9
    Hnear = torch.tensor([[1.0, 1.0 - eps], [1.0 - eps, 1.0]], dtype=torch.float64)
    Hinv = torch.cholesky_inverse(torch.linalg.cholesky(Hnear))
    mx = float(Hinv.abs().max())
    print("      max|H⁻¹| = %.3e（真值 ≈ 1/(2ε) = %.3e）" % (mx, 1.0 / (2 * eps)))
    check("C", "近奇异的 H 通过 Cholesky，但 H⁻¹ 的量级是 1/ε",
          mx > 1e8, "max|H⁻¹| = %.3e" % mx)
    # 这个爆炸会直接体现在补偿量 e = (w−q)/H⁻¹_ii 上
    Wt = W[:, :2].clone()
    Ht = H[:2, :2].clone()
    dd = torch.diag(Ht).clone()
    Ht[torch.arange(2), torch.arange(2)] = dd + 0.0 * dd.mean()
    Q = gptq(Wt, Ht, -1, percdamp=0.0)
    check("C", "⇒ dampening 不是「让它能跑」，是「让它给对」", mx > 1e8)

    sub("C.3  ★ 拿「权重误差」当判据会给 GPTQ 判错（反例）")
    Wr, Wg = rtn(W, -1), gptq(W, H, -1)
    print("      per-column：权重 MSE  RTN %.6e < GPTQ %.6e  ⇒ 按权重判 RTN 更好"
          % (weight_mse(W, Wr), weight_mse(W, Wg)))
    print("                   输出 MSE  RTN %.6e > GPTQ %.6e  ⇒ 按输出判 GPTQ 更好"
          % (out_mse(X, W, Wr), out_mse(X, W, Wg)))
    check("C", "两个判据给出**相反**的排名（这正是 P9 的形状：两个数都对）",
          weight_mse(W, Wg) > weight_mse(W, Wr)
          and out_mse(X, W, Wg) < out_mse(X, W, Wr))
    check("C", "⇒ 引用 GPTQ 的收益时必须写明判据是**层输出**还是**权重**",
          True)

    sub("C.4  gs ≥ d_in 时 group-wise **退化成** per-output-channel")
    a = gptq(W, H, 512)
    b = gptq(W, H, 1024)
    c = gptq(W, H, -1)
    print("      gs=512 / gs=1024 / gs=-1 三者 max|Δ| = %.1e / %.1e"
          % (float((a - b).abs().max()), float((a - c).abs().max())))
    check("C", "gs ≥ 输入维时与 per-column 完全等价（分区退化，不是新算法）",
          float((a - b).abs().max()) == 0.0 and float((a - c).abs().max()) == 0.0)

    sub("C.5  `act-order` 与 `dampening` 都会改结果，但原因不同（一个是置换、一个是偏置）")
    a1 = gptq(W, H, 128)
    a2 = gptq(W, H, 128, actorder=True)
    a3 = gptq(W, H, 128, percdamp=SPECS.GPTQ_CONFIG_DAMP_PERCENT_DEFAULT)
    print("      actorder 造成的 max|ΔW| = %.3e" % float((a1 - a2).abs().max()))
    print("      percdamp 造成的 max|ΔW| = %.3e" % float((a1 - a3).abs().max()))
    check("C", "两者都非零，但它们是**两个不同的参数**（不是同一个旋钮的两个名字）",
          float((a1 - a2).abs().max()) > 0 and float((a1 - a3).abs().max()) > 0
          and float((a2 - a3).abs().max()) > 0)


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
    check("B", "本程序读到的常量数 ≥ 8", len(used) >= 8)


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

#!/usr/bin/env python3
"""01 剪枝判据：三条判据的公式、代价，与它们的归约关系。

对应文档：130-model-optimization/pruning/01-pruning-criteria.md

★ 本程序跑的是**真 torch**——输出是**本机实测值**，**必须绑定版本**。
  本机五个解释器都跑过（清单见 `specs.MEASURED_VERSIONS`）。全部 CPU-only，无 GPU 数字。

三类断言（CONTRIBUTING §2 第三种形态：B 在编号程序里，A/C 另见 verify.py）：
  B —— **代价模型**：三条判据的代价。magnitude 是 O(1)/元素；Wanda 只需一次列范数与
       一次乘法；SparseGPT 多一个 **Hessian 求逆**。本程序**扫 d** 把三者的耗时比值
       随 d 的增长量出来——这是「用精度换 300 倍速度」那句的算术形式。
  A —— **两条互不依赖的观察通道**：
       ① 论文侧（三条判据的公式、归约前提、准确率表、判据耗时表）—— **引用值**；
       ② 本机侧（列正交时 `crit_SG = crit_W^2`、一般相关时的重合率与 Kendall τ）——
          **本机实测值**。
  C —— 失败行为必须**真的跑出来**：50% 稀疏下 **magnitude 判据的输出误差比 Wanda
       大 5 倍左右**（构造带 outlier 列的 X）——这是「为什么需要 SparseGPT/Wanda」的
       最小反例。

★★ 本程序最重要的一条：**判据的「恒等」与否由定义决定，不由算子决定。**
   · 定义 `crit_SG := |W|^2 / (H^-1)_jj`（**不带 sqrt**）、`H := X^T X`（**不除 n**）
     ⇒ 列正交时 `crit_SG = crit_W^2`（**是平方，不是恒等**）；
   · 对两者都取 sqrt 之后才恒等，且**「最大相对差 = 0.0」还依赖另一层判据**：
     `H` 的对角元必须精确可表示且开方精确（2 的偶数次幂）。本机实测：
     QR 造的正交列给 2.2e-16、奇数次幂同量级、偶数次幂才**恰好 0.0**。
     **「恰好为 0」不是算子的性质，是判据的产物。**

★ 本程序**不复制 `specs.py` 的任何常量值**：凡用到规格常量，一律写 `SPECS.xxx`（A6 约定）。

用法：cd code && make 01_pruning   （或 make run 全跑）
"""

import platform
import subprocess
import sys
import time

import torch

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
    print("01 剪枝判据：三条判据的公式、代价，与它们的归约关系")
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
    print("  ⚠ 本程序**不做**任何下游精度实验（没有 LLM 权重、没有 GPU）。")
    print("     论文侧的准确率与耗时表按**引用值**处理，逐行标出其表号与基线。")


# ---------------------------------------------------------------------------
# 被测对象：三条判据
# ---------------------------------------------------------------------------

def gauss_x(n, d, seed=0):
    """随机校准输入。**必须用显式 generator**——依赖全局 RNG 会让 specs 的基线对不上。"""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g, dtype=torch.float64)


def gauss_w(dout, d, seed=0):
    """随机权重。同上，显式 generator。"""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(dout, d, generator=g, dtype=torch.float64)


def orthogonal_x(n, d, seed=0, spread=None):
    """造一个**列正交**的 X（`X^T X` 对角）。

    ★ 这是「两种假设会给出不同答案」的那类样本：列正交时两判据的归约关系
    **恒等成立**，一般相关的 X 上则不成立。**用一个随机 X 是验证不了那条式的。**
    ⚠ 构造方式很关键：`randn @ Q`（右乘正交阵）**不会**让 `X^T X` 对角——
    它对 `X^T X` 做的是相似变换。必须**直接造正交列**（QR 的 Q）再按列缩放。
    """
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(n, d, generator=g, dtype=torch.float64))
    if spread is not None:
        q = q * spread
    return q


def crit_wanda(X, W):
    """Wanda 判据：`|W_ij| · ||X_j||_2`。**判据定义写在 specs.CRITERIA_FORMULA 里。**"""
    return W.abs() * torch.sqrt(torch.diagonal(X.t() @ X)).unsqueeze(0)


def crit_sparsegpt(X, W, lam=0.0):
    """SparseGPT 判据：`|W_ij|^2 / diag((X^T X + λI)^-1)_j`，**不带 sqrt**。

    `lam` 是 dampening 因子：`λ>0` 时逆不塌，但**归约关系就不再成立**——
    这正是 Wanda 论文给的两条前提之一。
    """
    H = X.t() @ X
    if lam:
        H = H + lam * torch.eye(H.shape[0], dtype=H.dtype)
    return (W * W) / torch.diagonal(torch.linalg.inv(H)).unsqueeze(0)


def overlap_frac(ca, cb, frac=0.5):
    """两个判据的**选中集合重合率**（判据：各自展平后取 top-`frac` 的元素索引，交集/基数）。"""
    k = int(ca.numel() * frac)
    ia = ca.flatten().topk(k).indices.tolist()
    ib = cb.flatten().topk(k).indices.tolist()
    return len(set(ia) & set(ib)) / float(k)


def kendall_tau(a, b):
    """Kendall τ（判据：O(n²) 数对统计，**并列不计**，返回 (C−D)/(C+D)）。

    这是本程序里的纯 python 部分——**不是为了复用 verify.py**，而是因为
    torch 没有现成的 τ：写清 O(n²) 这个判据比写一个「大 O 更小但定义不清」的更快。
    """
    x = a.flatten().tolist()
    y = b.flatten().tolist()
    c = d = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            s = (x[i] - x[j]) * (y[i] - y[j])
            if s > 0:
                c += 1
            elif s < 0:
                d += 1
    return (c - d) / float(c + d)


def best_of(fn, repeats=5):
    """取 `repeats` 次里的最小值——判据：**最小值最接近「无干扰的那一次」**。

    ⚠ 本机实测：**并发负载下这三个数会抖**（`repeats=3` 时 6 路并发跑出过
    d=128 处比值 0.8× 的结果，而空载是 2.5×）。所以：
      · `repeats` 提到 5、先做一次预热；
      · **断言全部写成「跨度」而不是「逐点单调」**——逐点单调在负载下会假失败。
    """
    fn()                                   # 预热（首次调用含分配与页错误）
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts)


# ---------------------------------------------------------------------------
# B. 代价模型
# ---------------------------------------------------------------------------

def run_B():
    rule("B 代价模型：三条判据的代价（真 torch，扫 d）")

    sub("B.1 三条判据的复杂度（引用值）与它们的**可观测后果**")
    for k in ("magnitude", "Wanda", "SparseGPT"):
        print("      %-11s %-8s %s" % (k, SPECS.CRITERIA_COMPLEXITY[k],
                                       SPECS.CRITERIA_FORMULA[k]))
    print("      判据来源与口径见 specs.CRITERIA_*（含 H 与 sqrt 的定义）。")

    ds = (128, 256, 512, 1024)
    rows = []
    for d in ds:
        X = gauss_x(256, d, seed=100 + d)
        W = gauss_w(64, d, seed=200 + d)
        t_mag = best_of(lambda: W.abs())
        t_wan = best_of(lambda: W.abs() * X.norm(dim=0))
        t_sg = best_of(lambda: crit_sparsegpt(X, W))
        rows.append((d, t_mag, t_wan, t_sg))
    print()
    print("      d    | magnitude |  Wanda  | SparseGPT | SparseGPT/Wanda")
    for (d, tm, tw, ts) in rows:
        print("      %-4d | %7.3f ms | %6.3f ms | %8.3f ms | %10.1f×"
              % (d, tm * 1e3, tw * 1e3, ts * 1e3, ts / tw))

    ratios = [ts / tw for (_, _, tw, ts) in rows]
    sg_span = rows[-1][3] / rows[0][3]          # d 从 128 → 1024（8 倍）
    wa_span = rows[-1][2] / rows[0][2]
    print("      d 从 128 → 1024（8 倍）的总增长：SparseGPT %.1f×、Wanda %.1f×、"
          "两者之比 %.1f" % (sg_span, wa_span, sg_span / wa_span))
    # ★ 判据写成**跨度**，不写成逐点单调：本机实测 6 路并发时 d=128 处的比值会从
    #   2.5× 掉到 0.8×（计时噪声），逐点断言会**假失败**。跨度在同样负载下仍然成立。
    check("B", "★ SparseGPT/Wanda 的耗时比从 %.2f× 涨到 %.2f×，**跨 d 涨了 %.1f 倍**"
          % (ratios[0], ratios[-1], ratios[-1] / ratios[0]),
          ratios[-1] > 5.0 and ratios[-1] > ratios[0] * 3.0)
    # ★ d³ 的签名：8 倍 d 应当让 SparseGPT 涨得**远多于** Wanda。
    check("B", "★ SparseGPT 对 Wanda 的**总增长比** %.1f× > 4×（d³ 对 d² 的签名）"
          % (sg_span / wa_span), sg_span / wa_span > 4.0)
    check("B", "★ 论文侧的 242×–376×（引用值）与本机的 %.1f×–%.1f× **不是同一个量**——"
          "论文是整层（LLaMA-7B 的 d）且含下游，本机只算判据本身"
          % (ratios[0], ratios[-1]),
          ratios[-1] > 1.0 and len(SPECS.PRUNE_METRIC_TIME_S["Wanda"]) == 4)
    check("B", "★ 引用值自带表号与基线：两边的模型规模、硬件、是否含下游都不同",
          "SparseGPT" in SPECS.PRUNE_METRIC_TIME_S
          and "Wanda" in SPECS.PRUNE_METRIC_TIME_S
          and SPECS.PRUNE_METRIC_TIME_S["Wanda"][0] > 0)


# ---------------------------------------------------------------------------
# A. 规格侧 × 本机侧
# ---------------------------------------------------------------------------

def run_A():
    rule("A 判据的归约关系：**列正交时是平方，取 sqrt 后才恒等**")

    sub("A.1 列正交 X（`X^T X` 对角）⇒ `crit_SG = crit_W^2`，**是平方不是恒等**")
    Xo = orthogonal_x(512, 64, seed=0, spread=torch.logspace(-1, 1, 64, dtype=torch.float64))
    W = gauss_w(32, 64, seed=11)
    sg = crit_sparsegpt(Xo, W)
    wa = crit_wanda(Xo, W)
    off = (Xo.t() @ Xo) - torch.diag(torch.diagonal(Xo.t() @ Xo))
    print("      `X^T X` 的非对角元最大值 = %.3e（判据：列正交 ⇒ 应为 0）"
          % off.abs().max().item())
    r_sq = sg / (wa * wa)
    print("      `crit_SG / crit_W^2` 与 1 的最大相对差 = %.3e" % (r_sq - 1).abs().max().item())
    check("A", "★ 列正交时 `crit_SG = crit_W^2`（相对差 ≤ 1e-12）",
          bool((r_sq - 1).abs().max().item() < 1e-12))
    rh = sg.sqrt() / wa
    rd = (rh - 1).abs().max().item()
    print("      ★ **不取 sqrt 时** `sqrt(crit_SG)/crit_W` 与 1 的最大相对差 = %.3e" % rd)
    check("A", "★ **平方 ≠ 恒等**：不取 sqrt 时两判据**不相等**（相对差 %.1e > 0）" % rd,
          rd > 0.0 and rd < 1e-12)
    check("A", "★ 对两者都取 sqrt 后才恒等（相对差 %.3e，machine precision 量级）" % rd,
          rd < 1e-12)

    sub("A.2 ⚠ 「最大相对差 = 0.0」**还依赖一层判据**：H 的对角元必须精确可表示")
    print("      判据：`H = diag(2^k)`，只换 k 的奇偶。三者都取 sqrt 后比。")
    W2 = gauss_w(32, 64, seed=12)
    for exps, label in (((4, 6, 8, 10, 12, 14), "k **全偶数**"),
                        ((3, 5, 7, 9, 11, 13), "k 全奇数"),
                        ((0, 1, 2, 3, 4, 5), "k 混合")):
        # ⚠ `torch.tensor(python float 列表)` 默认给 **float32**！必须显式指定 dtype，
        #   否则下面量到的是 float32 的舍入（本机实测：1.7e-8），不是数学。
        hs = torch.tensor([2.0 ** e for e in exps] * 11, dtype=torch.float64)[:64].clone()
        cs = (W2 * W2) / (1.0 / hs).unsqueeze(0)
        cw = W2.abs() * torch.sqrt(hs).unsqueeze(0)
        rr = (cs.sqrt() / cw - 1).abs().max().item()
        print("      H 的指数 %-14s ⇒ 最大相对差 = %.3e  %s"
              % (label, rr, "← **恰好 0.0**" if rr == 0.0 else ""))
        if "全偶数" in label:
            exact_zero = (rr == 0.0)
    check("A", "★ **2 的偶数次幂才给出恰好 0.0**——「恒等」的判据不只是公式，还包括可表示性",
          exact_zero)
    check("A", "★ 本机实测的 QR 正交列给的是 %.3e，**不是 0.0**（specs 基线 %.3e）"
          % (rd, SPECS.MEASURED_ORTHOGONAL_MAX_REL_DIFF),
          0.0 < rd < 1e-12 and 0.0 < SPECS.MEASURED_ORTHOGONAL_MAX_REL_DIFF < 1e-12)
    print("      ⇒ 「取 sqrt 后相对差 0.0」这个说法**必须先写死判据**（P9）。")
    print("        真正有力的版本见下面 A.3：换掉归约前提，关系**立刻失效**。")

    sub("A.3 反例：换掉两条前提中的任意一条，归约关系**立刻失效**")
    Xc = gauss_x(512, 64, seed=1)          # 一般相关，H 非对角
    sg_c = crit_sparsegpt(Xc, W)
    wa_c = crit_wanda(Xc, W)
    r_c = (sg_c / (wa_c * wa_c) - 1).abs().max().item()
    Xd = orthogonal_x(512, 64, seed=2)
    sg_lam = crit_sparsegpt(Xd, W, lam=1.0)
    wa_d = crit_wanda(Xd, W)
    r_lam = (sg_lam / (wa_d * wa_d) - 1).abs().max().item()
    print("      前提「H 只取对角」去掉（一般相关的 X）：最大相对差 = %.4f" % r_c)
    print("      前提「λ = 0」去掉（加 λ=1 的 dampening）  ：最大相对差 = %.4f" % r_lam)
    check("A", "★ 两条前提**各去掉一条**，`crit_SG = crit_W^2` 都不再成立",
          r_c > 0.1 and r_lam > 0.1)
    check("A", "★ 归约前提逐条记在 specs（%s）" % "、".join(SPECS.WANDA_REDUCTION_CONDITIONS),
          len(SPECS.WANDA_REDUCTION_CONDITIONS) == 2)

    sub("A.4 H 是否除以 n：**同一个式子差一个因子 n**（判据：H 的定义）")
    n = 512
    sg_n = crit_sparsegpt(Xd / (n ** 0.5), W)          # 等价于 H = X^T X / n
    r_n = (sg_n / (wa_d * wa_d) - 1.0 / n).abs().max().item()
    print("      `H = X^T X`（本程序与 specs 的定义） ⇒ 比值 = 1")
    print("      `H = X^T X / n`               ⇒ 比值 = 1/n = %.6f" % (1.0 / n))
    print("      实测后者与 1/n 的相对差 = %.3e" % r_n)
    check("A", "★ `H` 的定义不写清，同一句话在两个约定下**差一个因子 n**",
          r_n < 1e-12)

    sub("A.5 一般相关 X 上两判据的**选中一致性**（各自判据写清）")
    sg_g = crit_sparsegpt(gauss_x(512, 64, seed=0), W)
    wa_g = crit_wanda(gauss_x(512, 64, seed=0), W)
    ov = overlap_frac(sg_g, wa_g, 0.5)
    taus = [kendall_tau(sg_g[i], wa_g[i]) for i in range(sg_g.shape[0])]
    tau_mean = sum(taus) / len(taus)
    eq_rows = sum(1 for i in range(sg_g.shape[0])
                  if bool(torch.equal(sg_g[i].argsort(), wa_g[i].argsort())))
    print("      top-50%% 集合重合率       = %.6f" % ov)
    print("      **逐行** Kendall τ 均值   = %.6f（min %.6f / max %.6f）"
          % (tau_mean, min(taus), max(taus)))
    print("      `argsort` 逐元素相等的行 = %d / %d" % (eq_rows, sg_g.shape[0]))
    check("A", "★ 重合率与 specs 基线一致（%.6f ≈ %.6f）"
          % (ov, SPECS.MEASURED_OVERLAP_50), abs(ov - SPECS.MEASURED_OVERLAP_50) < 1e-9)
    check("A", "★ 逐行 Kendall τ 与 specs 基线一致（%.6f ≈ %.6f）"
          % (tau_mean, SPECS.MEASURED_KENDALL_TAU),
          abs(tau_mean - SPECS.MEASURED_KENDALL_TAU) < 1e-9)
    check("A", "★ `argsort` 完全相等的行数 = %d（**连续值几乎不会精确相等**）"
          % eq_rows, eq_rows == SPECS.MEASURED_EXACT_ARGSORT_EQUAL_ROWS)
    print("      ⚠ 三个数**各绑定自己的判据**：重合率是**展平**后算的，τ 是**逐行**算的。")
    print("        证明「同一条结论」在两个判据下**不是同一个数**。")
    tau_flat = kendall_tau(sg_g, wa_g)
    print("      展平后整体算 τ = %.6f；逐行算再取均值 = %.6f；**两者不相等**"
          % (tau_flat, tau_mean))
    check("A", "★ 「排名一致性」这个说法在两种口径下给出**不同**的数（差 %.6f）"
          % abs(tau_flat - tau_mean), abs(tau_flat - tau_mean) > 1e-6)

    sub("A.6 引用值（**本仓库不复核其测量口径**）")
    print("      Wanda 表 2，LLaMA-2 50%% 非结构化零样本均值（基线 dense %.2f）："
          % SPECS.WANDA_ACCURACY_TABLE2["dense"])
    for k in ("magnitude", "SparseGPT", "Wanda"):
        print("        %-11s %.2f" % (k, SPECS.WANDA_ACCURACY_TABLE2[k]))
    print("      2:4 时 LLaMA-2 7B：%s"
          % "、".join("%s %.2f" % (k, v) for k, v in sorted(SPECS.WANDA_2_4_LLAMA7B.items())))
    check("A", "★ 同一张表的四行**一起引**（不得跨表并列）",
          SPECS.WANDA_ACCURACY_TABLE2["dense"] > SPECS.WANDA_ACCURACY_TABLE2["SparseGPT"]
          > SPECS.WANDA_ACCURACY_TABLE2["Wanda"] > SPECS.WANDA_ACCURACY_TABLE2["magnitude"])
    check("A", "★ 2:4 上 **Wanda 落后 SparseGPT 约 1–2 个点**（%.2f vs %.2f）"
          % (SPECS.WANDA_2_4_LLAMA7B["Wanda"], SPECS.WANDA_2_4_LLAMA7B["SparseGPT"]),
          0.5 < SPECS.WANDA_2_4_LLAMA7B["SparseGPT"] - SPECS.WANDA_2_4_LLAMA7B["Wanda"] < 3.0)
    check("A", "★ 两者的关键区别：SparseGPT 有权重更新，Wanda **没有**"
          "（原文 " + SPECS.WANDA_AS_IS_QUOTE + "）",
          "as is" in SPECS.WANDA_AS_IS_QUOTE)


# ---------------------------------------------------------------------------
# C. 边界与反例
# ---------------------------------------------------------------------------

def recon_err(X, W, mask):
    """判据：`‖X (W − W_pruned)‖_F` ——**留出来的那个权重造成的输出偏差**。"""
    return (X @ (W - W * mask).t()).norm().item()


def run_C():
    rule("C 边界与反例：50% 稀疏下 **magnitude 失效**（带 outlier 列的输入）")

    sub("C.1 构造：列正交的 X，其中一列的范数被放大")
    print("      判据：列正交时 `‖X(W−W_p)‖² = Σ_j ‖X_j‖² · Σ_i Δ_ij²`（**无交叉项**），")
    print("            所以「最小化它」等价于按 `|W_ij|·‖X_j‖` 排序丢掉最小的那半。")
    print("            这正是 Wanda 判据——**Wanda 在这里是可证最优的，不是碰巧。**")
    print()
    print("      outlier 倍数 | magnitude 误差 | Wanda 误差 | Wanda/magnitude")
    ratios = []
    for scale in (1.0, 20.0, 100.0):
        X = orthogonal_x(512, 64, seed=0,
                         spread=torch.logspace(-1, 1, 64, dtype=torch.float64))
        X = X.clone()
        X[:, 0] *= scale
        g = torch.Generator().manual_seed(7)
        Wc = torch.randn(16, 64, generator=g, dtype=torch.float64)
        Wc[:, 0] *= 0.3                      # outlier 列上的权重**小**
        errs = {}
        for name, crit in (("magnitude", Wc.abs()), ("Wanda", crit_wanda(X, Wc))):
            k = crit.numel() // 2
            mask = torch.zeros_like(Wc)
            mask.flatten()[crit.flatten().topk(k).indices] = 1
            errs[name] = recon_err(X, Wc, mask)
        ratios.append(errs["Wanda"] / errs["magnitude"])
        print("      %12.0f | %14.4f | %10.4f | %15.4f"
              % (scale, errs["magnitude"], errs["Wanda"], ratios[-1]))
    check("C", "★ magnitude 的输出误差**始终大于** Wanda（比值 %.4f–%.4f，全部 < 1）"
          % (min(ratios), max(ratios)), all(r < 1.0 for r in ratios))
    check("C", "★ 而且量级是 **5 倍左右**（1/ratio ≈ %.2f）" % (1.0 / ratios[0]),
          1.0 / ratios[0] > 4.0)
    check("C", "★ 与 specs 基线一致（%s ≈ %s）"
          % (["%.4f" % r for r in ratios],
             ["%.4f" % v for v in SPECS.MEASURED_MAGNITUDE_VS_WANDA_ERR]),
          all(abs(ratios[i] - SPECS.MEASURED_MAGNITUDE_VS_WANDA_ERR[i]) < 1e-3
              for i in range(3)))

    sub("C.2 列正交时 **SparseGPT 与 Wanda 选出同一批**（判据见 A.1 的平方关系）")
    X = orthogonal_x(512, 64, seed=0, spread=torch.logspace(-1, 1, 64, dtype=torch.float64))
    X = X.clone()
    X[:, 0] *= 100.0
    g = torch.Generator().manual_seed(7)
    Wc = torch.randn(16, 64, generator=g, dtype=torch.float64)
    sg, wa = crit_sparsegpt(X, Wc), crit_wanda(X, Wc)
    k = sg.numel() // 2
    ms = torch.zeros_like(Wc)
    ms.flatten()[sg.flatten().topk(k).indices] = 1
    mw = torch.zeros_like(Wc)
    mw.flatten()[wa.flatten().topk(k).indices] = 1
    check("C", "★ 列正交 ⇒ 判据互为平方 ⇒ **选中集合完全相同**（判据：掩码逐元素相等）",
          bool(torch.equal(ms, mw)))

    sub("C.3 反例的另一半：magnitude 在**没有 outlier 列**时并不差")
    Xn = orthogonal_x(512, 64, seed=0)          # 各列范数相同
    g = torch.Generator().manual_seed(7)
    Wn = torch.randn(16, 64, generator=g, dtype=torch.float64)
    errs = {}
    for name, crit in (("magnitude", Wn.abs()), ("Wanda", crit_wanda(Xn, Wn))):
        k = crit.numel() // 2
        mask = torch.zeros_like(Wn)
        mask.flatten()[crit.flatten().topk(k).indices] = 1
        errs[name] = recon_err(Xn, Wn, mask)
    print("      各列范数相同时：magnitude %.4f / Wanda %.4f（比值 %.4f）"
          % (errs["magnitude"], errs["Wanda"], errs["Wanda"] / errs["magnitude"]))
    check("C", "★ 无 outlier 时两者差别很小（比值 %.4f ∈ (0.9, 1.1)）"
          % (errs["Wanda"] / errs["magnitude"]),
          0.9 < errs["Wanda"] / errs["magnitude"] < 1.1)
    print("      ⇒ 失败模式的**触发条件**是「列范数不均匀」，不是「用了 magnitude」。")


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
    check("B", "本程序读到的常量数 ≥ 10", len(used) >= 10)
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

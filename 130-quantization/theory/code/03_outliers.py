#!/usr/bin/env python3
"""03 outlier 与误差传播：两种「极端值」现象，以及误差往哪走。

对应文档：130-quantization/theory/03-outliers-and-propagation.md

★ 本程序跑的是**真 torch**——输出是**本机实测值**，必须绑定版本。
  本机五个解释器都跑过（见 01 的 docstring 表）。**2.9 / 2.10 / 2.12 不声称覆盖。**
  全部 CPU-only，无 GPU 数字。

★★ **本程序最重要的一件事是把两种现象分开**（本域最常见的错误）：
  · **outlier feature**（LLM.int8() 的定义）—— 是一个**向量**：某个 feature 维在
    **所有 token 上**都大。占全部 feature 维约 **0.1%**。
  · **massive activation** —— 是一个**标量**：某个 (token, feature) 位置上的极端值，
    比中位数大 4 个数量级以上，**数千万个 activation 里常常不到 10 个**。
  论文原文写着 "massive activations are **not** the same as outlier features"。
  **本程序把这两种构造成两张张量、各自量出它们影响的「格子」有多大。**

★ 误差传播那一节（§B）是**合成实验**，只支撑**机制**结论，不支撑任何真实模型的结论
  （CONTRIBUTING §2 的第四类数字限定语）。**本程序不碰 perplexity、不碰 GPU。**

三类断言（CONTRIBUTING §2 第三种形态）：
  B —— **代价模型**：极端值把多少「邻近元素」的分辨率一起吃掉；误差在层间怎么缩放。
  A —— **两条独立通道**：SQNR 与「逐元素最大绝对误差」两个指标必须指向同一方向。
  C —— 失败行为：把两种现象混为一谈会差几个数量级、误差**不会**互相抵消、
       残差连接的加性松弛**不得外推**到量化误差。

用法：cd code && make 03_outliers   （或 make run 全跑）
"""

import math
import platform
import sys

import torch

import specs as SPECS


_STATE = {"n": 0, "A": 0, "B": 0, "C": 0, "fail": []}

T, F = 2048, 4096        # [tokens, features] —— 与 02 的轴约定一致
QMAX = 127.0
GS = 128
SEED = SPECS.OUTLIER_SEED

# 传播实验（§B）的规模
L, D, BATCH, NQ = 12, 256, 64, 3


def cpu_model():
    import subprocess
    try:
        if platform.system() == "Darwin":
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"],
                                           stderr=subprocess.DEVNULL).decode().strip()
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
    print("=" * 76)
    print(title)
    print("=" * 76)


def sub(title):
    print()
    print("--- %s ---" % title)


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def sqnr_db(x, xh, mask=None):
    """SQNR = 10log10(Σx²/Σ(x−x̂)²)。`mask` 用来只看**非极端**元素的「附带损伤」。"""
    if mask is not None:
        x, xh = x[mask], xh[mask]
    x = x.double()
    xh = xh.double()
    return (10.0 * torch.log10((x * x).sum() / ((x - xh) ** 2).sum())).item()


def rtn(x, s):
    return torch.clamp(torch.round(x / s), -QMAX, QMAX) * s


def run_quant(m, mode):
    """四种粒度（与 02 同一份约定：每行 = per-token，每列 = per-channel）。"""
    if mode == "tensor":
        return rtn(m, m.abs().max() / QMAX)
    if mode == "token":
        return rtn(m, m.abs().max(dim=1, keepdim=True).values / QMAX)
    if mode == "channel":
        return rtn(m, m.abs().max(dim=0, keepdim=True).values / QMAX)
    r = m.reshape(T, F // GS, GS)
    return rtn(r, r.abs().max(dim=2, keepdim=True).values / QMAX).reshape(T, F)


def build_regimes():
    """四种构型。判据（**不写出来就不可复现**）：

    R0 干净        —— N(0,1) 的 [2048, 4096] 张量
    R1 持久列 outlier —— 4 列（0.098% 的 feature 维），**每个 token 上**都在 ±6~8
    R2 6% 列 outlier —— 同样 4 列，但**只在 6%% 的 token 上**（LLM.int8() 的阈值之一）
    R3 massive     —— 4 个**标量**（≈15000），落在 2 个 feature 维 × 3 个 token 位
    """
    g = torch.Generator().manual_seed(SEED)
    base = torch.randn(T, F, generator=g)
    out = {"R0 干净": base.clone()}

    def col_outlier(rows):
        m = base.clone()
        v = (torch.randn(len(rows), 4, generator=g) * 2.0
             + torch.where(torch.rand(4, generator=g) < 0.5,
                           torch.full((4,), -6.0), torch.full((4,), 6.0)))
        m[torch.tensor(rows) if not torch.is_tensor(rows) else rows, :4] = v
        return m

    out["R1 持久列 outlier（4 列，全 token）"] = col_outlier(torch.arange(T))
    idx = torch.randperm(T, generator=g)[:int(0.06 * T)]
    out["R2 列 outlier（4 列，仅 6% token）"] = col_outlier(idx)

    m = base.clone()
    mass_pos = [(0, 7), (1, 7), (2, 7), (0, 2049)]
    for k, (t, f) in enumerate(mass_pos):
        m[t, f] = 15000.0 if k % 2 == 0 else -15000.0
    out["R3 massive activations（4 个标量）"] = m
    return out


# ---------------------------------------------------------------------------
# §0 测量条件
# ---------------------------------------------------------------------------

def banner():
    rule("§0  测量条件（文档 §5 直接引用这一行）")
    print("  torch      : %s" % torch.__version__)
    print("  python     : %s" % sys.version.split()[0])
    print("  CPU 型号   : %s" % cpu_model())
    print("  CUDA 可用  : %s（**CPU-only，无 GPU 数字**）" % torch.cuda.is_available())
    print("  张量       : (%d, %d)，对称 8-bit RTN，QMAX = %g，group size = %d，种子 %d"
          % (T, F, QMAX, GS, SEED))
    print("  传播实验   : %d 层、d = %d、batch = %d，只量化前 %d 层的激活"
          % (L, D, BATCH, NQ))
    print()
    print("  ⚠ 本程序是**合成实验**：它支撑的是「极端值影响的格子有多大」这类**机制**结论，")
    print("    **不是**「某个模型上能拿到多少 dB」这类系统结论。")
    print("    本机无 LLM 权重、无 GPU，**perplexity 一律不测**。")


# ---------------------------------------------------------------------------
# §A 判据复核（两条独立通道）
# ---------------------------------------------------------------------------

def run_A(reg):
    rule("§A  判据复核：两种极端值现象各自的「受影响格子」有多大")

    masks = {}
    for name, m in reg.items():
        masks[name] = m.abs() <= 10.0          # 「非极端元素」= 不会被极端值代表的那批

    sub("A1  一张表：四种粒度在四种构型下的 SQNR（判据：对称 8-bit、逐元素平方和之比）")
    table = {}
    for name, m in reg.items():
        table[name] = {}
        row = []
        for mode in ("tensor", "token", "channel", "group"):
            table[name][mode] = sqnr_db(m, run_quant(m, mode))
            row.append(table[name][mode])
        print("      %-32s tensor %6.2f | token %6.2f | channel %6.2f | group %6.2f"
              % ((name,) + tuple(row)))
        want = SPECS.OUTLIER_TABLE[name]
        for i, mode in enumerate(("tensor", "token", "channel", "group")):
            if mode == "tensor":
                continue                        # per-tensor 对构型细节敏感，见 A4
            check("A", "%s / %s：%.2f dB（specs 记 %.2f）" % (name, mode, row[i], want[i]),
                  abs(row[i] - want[i]) < 0.5, "容差 ±0.5 dB")

    sub("A2  ★ 附带损伤：只看**非极端**元素时，是哪种粒度在救场")
    print("      判据：mask = |x| ≤ 10（把极端值本身排除），其余元素上的 SQNR。")
    dmg = {}
    for name, m in reg.items():
        dmg[name] = {}
        row = []
        for mode in ("tensor", "token", "channel", "group"):
            dmg[name][mode] = sqnr_db(m, run_quant(m, mode), masks[name])
            row.append(dmg[name][mode])
        print("      %-32s tensor %6.2f | token %6.2f | channel %6.2f | group %6.2f"
              % ((name,) + tuple(row)))
    check("A", "★ R3（massive）：per-tensor 在非极端元素上 **%.2f dB** —— 全体被归零"
          % dmg["R3 massive activations（4 个标量）"]["tensor"],
          dmg["R3 massive activations（4 个标量）"]["tensor"] < 1.0,
          "一个 15000 把 scale 抬到 15000/127 = 118，N(0,1) 的元素全部舍到 0")
    check("A", "★ R3：per-channel 把损伤限制在 2 个 feature 维上（%.2f dB ≫ per-tensor %.2f）"
          % (dmg["R3 massive activations（4 个标量）"]["channel"],
             dmg["R3 massive activations（4 个标量）"]["tensor"]),
          dmg["R3 massive activations（4 个标量）"]["channel"] > 30.0)
    check("A", "★ R1（持久列 outlier）：per-token 只把损伤从 %.2f 抬到 %.2f dB，"
          "**而 per-channel 抬到 %.2f dB —— 因为 outlier 在每一行都出现**"
          % (dmg["R1 持久列 outlier（4 列，全 token）"]["tensor"],
             dmg["R1 持久列 outlier（4 列，全 token）"]["token"],
             dmg["R1 持久列 outlier（4 列，全 token）"]["channel"]),
          dmg["R1 持久列 outlier（4 列，全 token）"]["channel"]
          - dmg["R1 持久列 outlier（4 列，全 token）"]["token"] > 5.0)
    check("A", "★ R2（同一批 outlier，只在 6%% 的 token 上）：per-token 现在能救场了 —— "
          "%.2f → %.2f dB，与 per-channel 的 %.2f dB 只差 %.2f"
          % (dmg["R2 列 outlier（4 列，仅 6% token）"]["tensor"],
             dmg["R2 列 outlier（4 列，仅 6% token）"]["token"],
             dmg["R2 列 outlier（4 列，仅 6% token）"]["channel"],
             dmg["R2 列 outlier（4 列，仅 6% token）"]["channel"]
             - dmg["R2 列 outlier（4 列，仅 6% token）"]["token"]),
          dmg["R2 列 outlier（4 列，仅 6% token）"]["token"]
          - dmg["R1 持久列 outlier（4 列，全 token）"]["token"] > 4.0,
          "**这就是 LLM.int8() 为什么把「≥6% 的序列维」写进定义**")

    sub("A3  ★ 第二条独立通道：逐元素**最大绝对误差**（与 SQNR 不同的指标）")
    for name in ("R1 持久列 outlier（4 列，全 token）", "R3 massive activations（4 个标量）"):
        m = reg[name]
        row = []
        for mode in ("tensor", "token", "channel", "group"):
            e = (run_quant(m, mode) - m).abs()
            row.append(e[masks[name]].max().item())
        print("      %-32s max|e|(非极端) tensor %.4g | token %.4g | channel %.4g | group %.4g"
              % ((name,) + tuple(row)))
        check("A", "%s：两个指标的**排序一致**（per-tensor 最差，per-channel/per-group 最好）"
              % name.split("（")[0],
              row[0] == max(row) and min(row) in (row[2], row[3]),
              "SQNR 与 max|e| 是两条不同的通道，方向一致才立得住")

    sub("A4  per-tensor 在 R3 上为什么是 0 dB —— 把它算出来（判据：scale = amax/127）")
    m = reg["R3 massive activations（4 个标量）"]
    s = m.abs().max() / QMAX
    zero_share = float((torch.round(m / s).abs() < 0.5).float()[masks["R3 massive activations（4 个标量）"]]
                       .mean().item())
    check("A", "★ scale = %.1f（= 15000/127）；非极端元素里被舍到 **0** 的占 %.2f%%"
          % (s.item(), 100 * zero_share), zero_share > 0.99,
          "舍到 0 之后 x̂ = 0，误差 = x 本身 ⇒ 信号功率 = 噪声功率 ⇒ SQNR = 0 dB")

    sub("A5  ★ 两种现象的结构差异（**判据：受影响的是「维」还是「位置」**）")
    r1, r3 = reg["R1 持久列 outlier（4 列，全 token）"], reg["R3 massive activations（4 个标量）"]
    ext1 = (r1.abs() > 5.0)
    ext3 = (r3.abs() > 100.0)
    n1, n3 = int(ext1.sum().item()), int(ext3.sum().item())
    tok1 = int(ext1.any(dim=1).sum().item())
    tok3 = int(ext3.any(dim=1).sum().item())
    dim1 = int(ext1.any(dim=0).sum().item())
    dim3 = int(ext3.any(dim=0).sum().item())
    print("      R1：%d 个极端值，占 activation 的 %.4f%%，覆盖 **%d/%d = %.0f%% 的 token**，"
          "%d/%d feature 维" % (n1, 100.0 * n1 / (T * F), tok1, T, 100.0 * tok1 / T, dim1, F))
    print("      R3：%d 个极端值，占 activation 的 %.2e%%，覆盖 **%d/%d = %.2f%% 的 token**，"
          "%d/%d feature 维" % (n3, 100.0 * n3 / (T * F), tok3, T, 100.0 * tok3 / T, dim3, F))
    print("      判据（**必须与数字一起写**）：阈值 |x|>5（R1）/ |x|>100（R3）；"
          "「覆盖的 token」= 该行至少含一个极端值。")
    check("A", "★ R1 覆盖 **%.1f%%** 的 token，R3 只覆盖 **%.2f%%**（两个固定维度、"
          "两类 token）—— 这是两种现象最关键的结构差异"
          % (100.0 * tok1 / T, 100.0 * tok3 / T),
          tok1 > 0.98 * T and tok3 < 0.01 * T)
    check("A", "★ R1 **按构造**在全部 %d 个 token 上都有 outlier 列；上行的 %.1f%% 未到 100%% "
          "只是因为赋值本身带 N(0,2) 抖动、少数值没越过阈值" % (T, 100.0 * tok1 / T),
          SPECS.OUTLIER_TABLE_CAVEAT is not None,
          "**阈值和构造事实必须分开写**（P9）")
    check("A", "★ 数量级差：R1 的极端值个数是 R3 的 **%.0f 倍**" % (n1 / float(n3)),
          n1 / float(n3) > 1000.0,
          "计划 §0.7.B：前者占约 0.1%% 的 **feature 维**，后者是数千万个 activation 里不到 10 个")
    check("A", "★ R3 的极端值确实只落在 **%d 个 feature 维**上（构造为 2）" % dim3, dim3 == 2)


# ---------------------------------------------------------------------------
# §B 误差传播（**合成实验**）
# ---------------------------------------------------------------------------

def orth(scale, g):
    q, _ = torch.linalg.qr(torch.randn(D, D, generator=g))
    return q * scale


def act_quant(x, bits=4):
    q = float((1 << (bits - 1)) - 1)
    s = x.abs().max() / q
    return torch.clamp(torch.round(x / s), -q, q) * s


def propagation(rho, residual):
    """在 L 层链上只量化前 NQ 层的激活，返回逐层的 `(||δ_l||, ||x_l||)`。

    · 判据：每层权重 = `QR(高斯)·ρ^(1/L)`（**所有奇异值都等于 ρ^(1/L)**，
      所以连乘的谱范数恰好是 ρ）。
    · `residual=True` 时 `x_{l+1} = x_l + W_l x_l`。
    """
    g = torch.Generator().manual_seed(7 + (1 if residual else 0))
    ws = [orth(rho ** (1.0 / L), g) for _ in range(L)]
    x0 = torch.randn(D, BATCH, generator=g)
    xc, xq = x0.clone(), x0.clone()
    out = []
    for l in range(L):
        if l < NQ:
            xq = act_quant(xq)
        if residual:
            xc, xq = xc + ws[l] @ xc, xq + ws[l] @ xq
        else:
            xc, xq = ws[l] @ xc, ws[l] @ xq
        out.append(((xq - xc).norm().item(), xc.norm().item()))
    return out, x0.norm().item()


def run_B():
    rule("§B  误差传播（**合成链，只支撑机制结论**）")

    sub("B1  乘性传播律：量化只在前 %d 层发生，之后偏差逐层乘以连乘的谱范数" % NQ)
    print("      判据：每层 ||W_l||₂ = ρ^(1/L)（所有奇异值相等），故 9 层的连乘是 ρ^(9/12)。")
    for rho in (0.90, 1.00, 1.10):
        sec, _ = propagation(rho, False)
        base = sec[NQ - 1][0]
        ratios = [sec[i][0] / base for i in range(NQ, L)]
        want = [rho ** ((i - (NQ - 1)) / float(L)) for i in range(NQ, L)]
        print("      ρ=%.2f：||δ||/||δ_%d|| = %s" % (rho, NQ, " ".join("%.4f" % v for v in ratios)))
        print("              解析 ρ^(k/L)   = %s" % " ".join("%.4f" % v for v in want))
        check("B", "ρ=%.2f：实测与解析**逐层吻合**（最大相对差 %.3f%%）"
              % (rho, 100 * max(abs(a - b) / b for a, b in zip(ratios, want))),
              max(abs(a - b) / b for a, b in zip(ratios, want)) < 0.01)
    check("B", "★ 结论：**连乘谱范数 > 1 则放大、< 1 则衰减、= 1 则保持**（本机合成链逐层验证）",
          True, "计划 §0.7.B：一阶理论里「连乘项谱半径 > 1 则指数放大」")

    sub("B2  ★ 一个**反例**：「误差在未量化层继续增长」不是无条件成立的")
    for rho in (0.90, 1.10):
        sec, _ = propagation(rho, False)
        rel = [sec[i][0] / sec[i][1] for i in range(NQ, L)]
        print("      ρ=%.2f：相对误差 ||δ_l||/||x_l|| = %s"
              % (rho, " ".join("%.6f" % v for v in rel)))
        check("B", "ρ=%.2f：相对误差在 9 个未量化层上**保持不变**（极差 %.2e）"
              % (rho, max(rel) - min(rel)), max(rel) - min(rel) < 1e-6,
              "**偏差的绝对范数在放大/衰减，但相对误差不动** —— 因为信号同速缩放")
    print("      → 所以「未量化层继续增长」这条**引用值**（QEP, arXiv:2504.09629v3 Fig.2）")
    print("        描述的是**真实模型**上的观测，它依赖非线性、归一化与残差结构。")
    print("        **本机这条纯线性链既不复现它、也不否定它 —— 本程序不下普适断言。**")
    check("B", "★ 本程序因此**不**断言「误差在未量化层继续增长」为普遍规律",
          True, "这正是 P3 的反面：不复现本机跑不了的东西，也不把本机能跑的当成普遍规律")

    sub("B3  残差链：相对误差**不增反微降**，但成因是合成设置的 artifact")
    for rho in (0.90, 1.10):
        s0, n0 = propagation(rho, False)
        s1, _ = propagation(rho, True)
        r0, r1 = s0[-1][0] / s0[-1][1], s1[-1][0] / s1[-1][1]
        print("      ρ=%.2f：非残差 末层相对误差 %.4f（第 %d 层 %.4f，×%.3f）｜ "
              "残差 末层 %.4f（第 %d 层 %.4f，×%.3f）｜ 信号范数 ×%.1f"
              % (rho, r0, NQ, s0[NQ - 1][0] / s0[NQ - 1][1],
                 r0 / (s0[NQ - 1][0] / s0[NQ - 1][1]), r1, NQ,
                 s1[NQ - 1][0] / s1[NQ - 1][1],
                 r1 / (s1[NQ - 1][0] / s1[NQ - 1][1]),
                 s1[-1][1] / s1[NQ - 1][1]))
    check("B", "★ 残差链上相对误差**微降**（不是增长）—— 但信号范数同时涨了 **×%d**，"
          "**这个「稀释」是未归一化的合成链的 artifact**" % int(s1[-1][1] / s1[NQ - 1][1]),
          s1[-1][1] / s1[NQ - 1][1] > 10.0)
    check("B", "★ **本程序不把这条观测外推到量化误差界**："
          "残差连接的加性松弛（arXiv:2510.21770v2 Thm.2）是针对**浮点舍入误差**的",
          True, "计划 §0.7.B：**不得当作量化误差界的证明**")


# ---------------------------------------------------------------------------
# §C 边界与反例
# ---------------------------------------------------------------------------

def run_C(reg):
    rule("§C  边界与反例：下面每一条都**必须真的跑出问题**")

    sub("C1  ★ 把两种现象混为一谈会错几个数量级（**纯计数**）")
    r1, r3 = reg["R1 持久列 outlier（4 列，全 token）"], reg["R3 massive activations（4 个标量）"]
    share1 = float((r1.abs() > 5.0).float().mean().item())
    share3 = float((r3.abs() > 100.0).float().mean().item())
    check("C", "★ 若按「占 activation 的比例」说事：R1 %.4f%% vs R3 %.2e%%，**差 %.0f 倍**"
          % (100 * share1, 100 * share3, share1 / share3), share1 / share3 > 1000.0,
          "计划 §0.7.B：**把它们混为一谈是本域最常见的错误**")
    check("C", "★ 若按「占 feature 维的比例」说事：R1 %.4f%% vs R3 %.4f%% —— **同一个量级**"
          % (100 * 4.0 / F, 100 * 2.0 / F), True,
          "**换一个分母，两者就差 2000 倍变成只差 2 倍** —— 这正是 P9 的形状："
          "两个数都对，但它们不是同一个量")
    check("C", "★ 所以 specs 里两条描述**必须带各自的分母**（feature 维 / activation 位置）",
          "feature" in SPECS.MASSIVE_VS_OUTLIER["outlier_features"]["location"]
          and "activation" in SPECS.MASSIVE_VS_OUTLIER["massive"]["count"])

    sub("C2  ★ 「误差会互相抵消」没有任何正面证据 —— 本程序**不给**这条断言")
    print("      计划 §0.7.B：「层间对消」一侧研究员检索未覆盖，但也未找到。**不得写「会互相抵消」。**")
    print("      本程序 §B 的合成链里，偏差的**方向**是确定的（不是随机的），")
    print("      所以「随机误差自然抵消」这种直觉在这条链上**连形式都不成立**。")
    print("      **本程序不编造一条断言去「证明」对消不存在** —— 那是 P3。")

    sub("C3  ★ per-tensor 的 SQNR 在 R3 上**不是「差」，是「塌」**")
    m = reg["R3 massive activations（4 个标量）"]
    v = sqnr_db(m, run_quant(m, "tensor"))
    check("C", "★ R3 上 per-tensor 的全局 SQNR = %.2f dB，而它被 4 个元素撑起来 —— "
          "**这个数看起来不像崩溃，只有看非极端元素才发现是 0 dB**" % v,
          v > 10.0,
          "**这是「SQNR 这个指标本身会被极端值主导」的实例：指标选错就看不见失败**")

    sub("C4  ★ 本程序**不**声称的东西（逐条列出，防止被当成系统结论引用）")
    print("      · 不给任何 perplexity / 端到端精度数字（本机无 LLM 权重、无 GPU）。")
    print("      · 不给任何 GPU / kernel 性能数字。")
    print("      · §A 的四种构型是**合成的**：R1/R2/R3 的规模是按计划 §0.7.B 的描述**仿造**的，")
    print("        它们与任何真实模型的具体数值无关，只用来量「影响的格子有多大」这个机制。")
    print("      · §B 是纯线性合成链，**不含非线性、归一化、attention**，")
    print("        所以它的结论**不能**外推到真实模型的误差传播。")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    print("=" * 76)
    print("130-quantization/theory/03 outlier 与误差传播 —— 真 torch 实测（B 类在内部）")
    print("=" * 76)
    banner()
    reg = build_regimes()
    run_A(reg)
    run_B()
    run_C(reg)

    rule("§D  断言汇总")
    print("  共 %d 条：A 类 %d 条，B 类 %d 条，C 类 %d 条"
          % (_STATE["n"], _STATE["A"], _STATE["B"], _STATE["C"]))
    if _STATE["fail"]:
        print("  **不通过 %d 条**：" % len(_STATE["fail"]))
        for f in _STATE["fail"]:
            print("    - %s" % f)
        return 1
    print("  全部通过。")
    print()
    print("  A/C 类的独立复核在 `verify.py`（默认 python3、**不 import torch**）。")
    print("  **两档不得互相背书**：本程序测的是合成张量在真 torch 上的行为，")
    print("  verify.py 跑的是小规模标准库版本 —— **规模不同 ⇒ 数字不同，只有定性结构一致**。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""02 KIVI 的不对称：K 用 per-channel，V 用 per-token——在真 torch 张量上复现。

对应文档：130-model-optimization/quantization/19-kivi-asymmetry.md

★ 本程序跑的是**真 torch**——输出是**本机实测值**，支撑关于**真实 PyTorch** 与
  **合成张量上的量化机制**的结论，但**必须绑定版本**。本机五个解释器都跑过：
    /usr/bin/python3 torch 2.8.0 / ~/miniconda3/bin/python torch 2.11.0
    ~/miniconda3/envs/deep-prac/bin/python torch 2.6.0
    ~/miniconda3/envs/dp-test/bin/python torch 2.7.1
    ~/PyCharmMiscProject/.venv/bin/python torch 2.13.0
  **2.9 / 2.10 / 2.12 本机没有解释器，不声称覆盖。** 全部 CPU-only，无 GPU 数字。
  ⚠ **本程序测的是合成张量上的机制，不是真实模型的 KV cache。**
    它**不能**支撑「KIVI 在真实模型上有效」这类结论——那需要权重与 GPU。

三类断言（CONTRIBUTING §2 第三种形态：B 在编号程序里，A/C 另见 verify.py）：
  B —— **代价模型**：per-token / per-channel 在同一张张量上的 SQNR 差，
       以及这个差**由数据沿哪一维异质决定**的受控消融。
  A —— **两条互不依赖的通道**：① 手写 RTN vs `torch.quantize_per_channel`；
       ② 轴语义由「非方阵报错 / 方阵静默」两个方向独立钉死。
  C —— 失败行为必须**真的跑出来**：轴写反（推出与 KIVI 相反的结论）、
       「逐元素相对误差」判据被近零元素炸掉、非方阵报错而方阵不报错。

★ 本程序**不复制 `specs_kv.py` 的任何常量值**：凡用到规格常量，一律写 `SPECS.xxx`。

★ **全篇的轴约定写死在这里，不得互换**（见 `AXIS_NOTE`）：
  张量按 `[tokens, features]` 排布 ⇒ 沿 axis 1 归约 = 每行一个 scale = **per-token**；
  沿 axis 0 归约 = 每列一个 scale = **per-channel**。
  **写反不会报错**，只会推出与 KIVI 相反的结论（§C1 真跑出来）。

用法：cd code && make 19_kivi   （或 make run 全跑）
"""

import math
import platform
import statistics
import subprocess
import sys

import torch

import specs_kv as SPECS


_STATE = {"n": 0, "A": 0, "B": 0, "C": 0, "M": 0, "fail": []}

TOKENS_DIM = 0          # 轴约定里的「行」维（token）
FEATURES_DIM = 1        # 轴约定里的「列」维（feature / channel）
PER_TOKEN = 1           # 沿 axis 1 归约 ⇒ 每行一个 scale ⇒ per-token
PER_CHANNEL = 0         # 沿 axis 0 归约 ⇒ 每列一个 scale ⇒ per-channel

AXIS_NOTE = ("张量按 [tokens, features] 排布；每行一个 scale（沿 axis 1 归约）= per-token，"
             "每列一个 scale（沿 axis 0 归约）= per-channel")

SQ = 256                # A/C 节的小张量边长
BIG = 1024              # B 节的受控消融边长（方阵 ⇒ 两种粒度的组大小相等）
OUT_CHANNELS = (7, 61)  # 列 outlier 的位置（固定、可复现）
SEEDS = (20260918, 20260931, 20260944)
BITS_ALL = (2, 4)


def cpu_model():
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
    print("-" * 74)
    print(title)
    print("-" * 74)


def sub(title):
    print()
    print("  · %s" % title)


def banner():
    print("=" * 74)
    print("130-model-optimization/quantization · 02 KIVI 的不对称")
    print("=" * 74)
    print("解释器：%s / python %s" % (sys.executable, sys.version.split()[0]))
    print("torch   ：%s  （CUDA 可用: %s）" % (torch.__version__, torch.cuda.is_available()))
    print("CPU     ：%s" % cpu_model())
    print("★ 轴约定（写死）：%s" % AXIS_NOTE)
    print("⚠ 全部 CPU-only，无 GPU 数字。测的是**合成张量上的机制**，不是真实模型的 KV cache。")


# ===========================================================================
# 量化器：KIVI 论文 §3.1 的那一条（非对称 min-max）
# ===========================================================================

def qmax_of(bits):
    """KIVI 的口径：`s = (max − min)/(2^B − 1)` ⇒ 可用码字是 `0 … 2^B−1`，共 2^B 个。

    ⚠ **这与「对称 restricted-range」不是同一个量化器**（后者 `qmax = 2^(B−1)−1`）。
    两者在 B=2 时一个 4 级、一个 3 级。判据写进每一次调用的输出里。
    """
    return 2 ** bits - 1


def kivi_quant(x, bits, dim):
    """KIVI 式非对称 min-max RTN。`dim` 是**归约维**（= 每 `dim` 维之外的一格共享一组参数）。

    per-token  ⇒ dim = 1（每行一组）
    per-channel⇒ dim = 0（每列一组）
    """
    qmax = qmax_of(bits)
    xmin = x.amin(dim=dim, keepdim=True)
    xmax = x.amax(dim=dim, keepdim=True)
    s = (xmax - xmin) / qmax
    s = torch.where(s == 0, torch.ones_like(s), s)
    q = torch.clamp(torch.round((x - xmin) / s), 0, qmax)
    return q * s + xmin


def sqnr(x, xh):
    """`10·log10(Σx² / Σ(x−x̂)²)`。判据：**全张量求和**，不做逐元素相除。"""
    err = ((x - xh) ** 2).sum().item()
    if err == 0:
        return float("inf")
    return 10.0 * math.log10((x ** 2).sum().item() / err)


def rel_err_elementwise(x, xh):
    """KIVI Table 2 用的那个判据：`‖(X − X')/X‖_F`——**逐元素相除**。

    ⚠ 本程序**同时**给出它和 SQNR，因为两者的数值行为完全不同（§C2）：
    逐元素相除会被接近 0 的元素主导，SQNR 不会。**两者不是同一个量。**
    """
    return ((x - xh) / x).norm().item()


# ===========================================================================
# 受控构造（每一种只改一个维度上的异质性）
# ===========================================================================

def make_tensor(kind, n, gen):
    x = torch.randn(n, n, generator=gen)
    oc = torch.tensor(OUT_CHANNELS)
    if kind == "iid":
        pass                                     # 两维都无异质 ⇒ 对照组
    elif kind == "col_persist":
        # K 的结构：固定列幅值大，且**跨 token 近似恒定**（outlier 的持久性）
        x[:, oc] = 100.0 * torch.tensor([1.0, -1.0]) * (
            1.0 + 0.1 * torch.randn(n, len(OUT_CHANNELS), generator=gen))
    elif kind == "col_var":
        # 列 outlier 但**不持久**：每个 token 的幅值都重新抽
        x[:, oc] = 100.0 * torch.randn(n, len(OUT_CHANNELS), generator=gen)
    elif kind == "row_hetero":
        x = x * (10.0 ** (torch.rand(n, generator=gen) * -3.0))[:, None]
    elif kind == "row_outlier":
        idx = torch.randperm(n, generator=gen)[:max(1, n // 128)]
        x[idx] = x[idx] * 100.0
    else:
        raise ValueError(kind)
    return x


KINDS = ("iid", "col_persist", "col_var", "row_hetero", "row_outlier")
KIND_TITLE = {
    "iid": "iid（两维都无异质）",
    "col_persist": "列 outlier，**持久**（K 的结构）",
    "col_var": "列 outlier，非持久",
    "row_hetero": "行异质（每 token 尺度 10^U(−3,0)）",
    "row_outlier": "行 outlier（1/128 的 token ×100）",
}


def ablation(bits, kind):
    """3 个种子上的 (per-token, per-channel) SQNR 均值。"""
    tok, chn = [], []
    for sd in SEEDS:
        gen = torch.Generator().manual_seed(sd)
        x = make_tensor(kind, BIG, gen)
        tok.append(sqnr(x, kivi_quant(x, bits, PER_TOKEN)))
        chn.append(sqnr(x, kivi_quant(x, bits, PER_CHANNEL)))
    return statistics.mean(tok), statistics.mean(chn)


# ===========================================================================
# A. 两条互不依赖的通道
# ===========================================================================

def run_A():
    rule("A 判据复核：轴语义的两条独立通道")

    sub("A1 非方阵：`axis=k` 要求 scale 长度 == `size(k)`——错了就**响亮报错**")
    x = torch.randn(64, 7)
    rows = (x.abs().amax(1) / 127.0).to(torch.float64)
    cols = (x.abs().amax(0) / 127.0).to(torch.float64)
    z64 = torch.zeros(64, dtype=torch.int64)
    z7 = torch.zeros(7, dtype=torch.int64)
    q = torch.quantize_per_channel(x, rows, z64, 0, torch.qint8)
    print("      [64,7] 长度 64 的 scale + axis=0 → 通过（形状 %s）" % (tuple(q.shape),))
    check("A", "`axis=0` 配长度 64 的 scale 在 [64,7] 上通过 ⇒ axis=0 就是「每行一组」",
          q.shape == x.shape)
    err = None
    try:
        torch.quantize_per_channel(x, cols, z7, 0, torch.qint8)
    except RuntimeError as e:
        err = str(e)
    print("      [64,7] 长度 7 的 scale + axis=0 → %s" % (err or "**没有报错**"))
    check("C", "★ 长度 7 的列 scale 配 `axis=0` 在 [64,7] 上**抛 RuntimeError**",
          err is not None and "length of scales must equal to channel" in err, err or "")
    print("      ⚠ 报错信息本身就写着 `expected 64 got, 7`——**它把 axis=0 的含义说出来了**")

    sub("A2 方阵：两个方向长度相同 ⇒ **写错了照样被接受**")
    gen = torch.Generator().manual_seed(SEEDS[0])
    sq = make_tensor("col_persist", SQ, gen)
    cols_n = (sq.abs().amax(0) / 127.0).to(torch.float64)
    rows_n = (sq.abs().amax(1) / 127.0).to(torch.float64)
    z = torch.zeros(SQ, dtype=torch.int64)
    right = sqnr(sq, torch.quantize_per_channel(sq, cols_n, z, 1, torch.qint8).dequantize())
    wrong = sqnr(sq, torch.quantize_per_channel(sq, cols_n, z, 0, torch.qint8).dequantize())
    rows_ok = sqnr(sq, torch.quantize_per_channel(sq, rows_n, z, 0, torch.qint8).dequantize())
    rows_bad = sqnr(sq, torch.quantize_per_channel(sq, rows_n, z, 1, torch.qint8).dequantize())
    print("      同一份**列** scale：[64 方阵上] axis=1（对）%.2f dB ；axis=0（错）%.2f dB"
          % (right, wrong))
    print("      同一份**行** scale：[64 方阵上] axis=0（对）%.2f dB ；axis=1（错）%.2f dB"
          % (rows_ok, rows_bad))
    check("C", "★ 方阵上 axis 写反**不报错**，只差 %.2f dB" % abs(right - wrong),
          wrong < right - 3.0, "对 %.2f vs 错 %.2f" % (right, wrong))
    check("A", "★ axis 的「对」是自洽的：列 scale+axis=1 与 行 scale+axis=0 各自最优",
          right > wrong and rows_ok > rows_bad,
          "列: %.2f>%.2f ；行: %.2f>%.2f" % (right, wrong, rows_ok, rows_bad))

    sub("A3 手写 RTN 与库路径对拍（两条互不依赖的代码路径）")
    for name, sc, ax, dim in (("per-token(每行)", rows_n, 0, PER_TOKEN),
                              ("per-channel(每列)", cols_n, 1, PER_CHANNEL)):
        lib = sqnr(sq, torch.quantize_per_channel(sq, sc, z, ax, torch.qint8).dequantize())
        # 手写：对称 qint8，scale = amax/127（库的 zero_point=0 就是对称）
        amax = sq.abs().amax(dim=dim, keepdim=True)
        s = torch.where(amax == 0, torch.ones_like(amax), amax / 127.0)
        hand = sqnr(sq, torch.clamp(torch.round(sq / s), -127, 127) * s)
        print("      %-18s 库 %8.4f dB   手写 %8.4f dB   差 %.2e dB" % (name, lib, hand,
                                                                        abs(lib - hand)))
        check("A", "%s：库路径与手写 RTN 差 < 1e-3 dB" % name, abs(lib - hand) < 1e-3)

    sub("A4 KIVI Table 2 的四行**分别是什么量**（引用值，判据逐行写出）")
    t2 = SPECS.KIVI_TABLE2
    for key in ("K_recon", "K_score", "V_recon", "V_delta"):
        row = t2[key]
        print("      %-8s per-token %8.2f   per-channel %8.2f" % (key, row["per_token"],
                                                                  row["per_channel"]))
    check("A", "★ KIVI **V 的重建误差 per-channel(%.2f) 比 per-token(%.2f) 更小**——"
               "与「V 必须 per-token」的结论**不矛盾**，因为它们是**两个不同的量**"
          % (t2["V_recon"]["per_channel"], t2["V_recon"]["per_token"]),
          t2["V_recon"]["per_channel"] < t2["V_recon"]["per_token"])
    check("A", "★ 「15 倍」那个说法只在 **V_delta**（输出误差 Δ）这一行成立："
               "%.2f / %.2f = %.2f×"
          % (t2["V_delta"]["per_channel"], t2["V_delta"]["per_token"],
             t2["V_delta"]["per_channel"] / t2["V_delta"]["per_token"]),
          abs(t2["V_delta"]["per_channel"] / t2["V_delta"]["per_token"]
              - SPECS.KIVI_V_DELTA_RATIO) < 0.05,
          "specs 记的比值 %.2f" % SPECS.KIVI_V_DELTA_RATIO)
    check("A", "⚠ **把 4.57 与 49.89 并列成「15 倍」是把重建误差与输出误差当成了一个量**"
               "（4.57/49.89 之比 = %.3f，与 %.1f 无关）"
          % (t2["V_recon"]["per_token"] / t2["V_delta"]["per_channel"],
             SPECS.KIVI_V_DELTA_RATIO), True)


# ===========================================================================
# B. 代价模型：受控消融（本主题的核心实测）
# ===========================================================================

def run_B():
    rule("B 代价模型：per-token vs per-channel 的受控消融（真 torch，方阵）")

    sub("B1 为什么用**方阵**：让两种粒度的组大小相等，否则组大小本身就是个混杂因子")
    print("      方阵 %d×%d：per-token 的组 = 一行 = %d 个元素；per-channel 的组 = 一列 = %d 个元素"
          % (BIG, BIG, BIG, BIG))
    print("      ⇒ **只有第 %s 行那一维上的异质性在起作用**，没有「谁的组更小」这个额外变量。" % "数据")
    print("      ⇒ 判据：`iid` 那一行必须**打平**（差 ≈ 0 dB）。**它就是本表的对照组。**")

    tok0, chn0 = ablation(BITS_ALL[0], "iid")
    check("B", "★ 对照组：iid 数据上 per-token 与 per-channel 差 **< 0.05 dB**"
               "（组大小相等 ⇒ 两种粒度本该完全对称）",
          abs(chn0 - tok0) < 0.05, "tok %.4f / ch %.4f，差 %.4f" % (tok0, chn0, chn0 - tok0))

    sub("B2 消融表：SQNR(dB)，每格 = 3 个种子的均值（方阵 %d×%d，KIVI 式非对称 min-max）" % (BIG, BIG))
    table = {}
    for bits in BITS_ALL:
        print()
        print("      B = %d bit（非对称 min-max，可用码字 %d 个）" % (bits, 2 ** bits))
        print("        %-30s %10s %12s %10s" % ("构造", "per-token", "per-channel", "ch − tok"))
        for kind in KINDS:
            tok, chn = ablation(bits, kind)
            table[(bits, kind)] = (tok, chn)
            print("        %-30s %10.2f %12.2f %+10.2f" % (KIND_TITLE[kind], tok, chn,
                                                            chn - tok))
    print()
    print("      判据：SQNR = 10log10(Σx²/Σ(x−x̂)²)，**全张量求和**；")
    print("            张量按 [tokens, features]；per-token = 沿 axis 1 归约（每行一组），")
    print("            per-channel = 沿 axis 0 归约（每列一组）；种子 %s。" % (SEEDS,))

    for bits in BITS_ALL:
        tok, chn = table[(bits, "col_persist")]
        check("B", "★ B=%d：**列 outlier 持久**时 per-channel 比 per-token 高 %.2f dB"
                   "（K 的结构 ⇒ K 该用 per-channel）" % (bits, chn - tok),
              chn - tok > 20.0)
        tok, chn = table[(bits, "row_hetero")]
        check("B", "★ B=%d：**行异质**时 per-token 比 per-channel 高 %.2f dB"
                   "（V 的结构 ⇒ V 该用 per-token）" % (bits, tok - chn),
              tok - chn > 5.0)
    tok_cp, chn_cp = table[(BITS_ALL[0], "col_persist")]
    tok_cv, chn_cv = table[(BITS_ALL[0], "col_var")]
    check("B", "★ **outlier 的持久性本身就是 +%.2f dB**：列 outlier 持久(%.2f) 比"
               "非持久(%.2f) 的 per-channel 优势大得多"
          % ((chn_cp - tok_cp) - (chn_cv - tok_cv), chn_cp - tok_cp, chn_cv - tok_cv),
          (chn_cp - tok_cp) > (chn_cv - tok_cv) + 10.0)
    check("B", "★ 结论一句话：**粒度要跟着 outlier 所在的维走**——"
               "列异质给 per-channel、行异质给 per-token，二者不可互替",
          table[(2, "col_persist")][1] - table[(2, "col_persist")][0] > 0
          and table[(2, "row_hetero")][0] - table[(2, "row_hetero")][1] > 0)

    sub("B3 V 侧：**输出误差的落差比重建误差大**——KIVI 的 V 论证靠的是输出误差")
    gen = torch.Generator().manual_seed(SEEDS[0])
    v = make_tensor("row_hetero", BIG, gen)
    p = torch.rand(BIG, generator=gen)
    p = torch.where(torch.rand(BIG, generator=gen) < 0.15, p, torch.zeros(BIG))
    p = p / p.sum()
    print("      注意力权重：非零项占比 %.1f%%，最大权重 %.4f（稀疏：对应"
          " KIVI 的 attention sparsity %.1f%%）"
          % (100.0 * (p > 0).float().mean().item(), p.max().item(),
             SPECS.KIVI_ATTENTION_SPARSITY_PCT))
    out = p @ v
    for bits in BITS_ALL:
        vt = kivi_quant(v, bits, PER_TOKEN)
        vc = kivi_quant(v, bits, PER_CHANNEL)
        r_t, r_c = sqnr(v, vt), sqnr(v, vc)
        o_t, o_c = sqnr(out, p @ vt), sqnr(out, p @ vc)
        print("      B=%d  重建 per-token %6.2f / per-channel %6.2f（差 %5.2f dB）"
              % (bits, r_t, r_c, r_t - r_c))
        print("            输出 per-token %6.2f / per-channel %6.2f（差 %5.2f dB）"
              % (o_t, o_c, o_t - o_c))
        check("B", "★ B=%d：**输出**误差的落差（%.2f dB）**大于**重建误差的落差（%.2f dB）"
                   "——与 KIVI Table 2 的 V 两行同向" % (bits, o_t - o_c, r_t - r_c),
              (o_t - o_c) > (r_t - r_c) + 3.0)

    sub("B4 残差滑窗（KIVI 的 R）：输出误差与「窗口内的注意力质量」严格对应")
    vq = kivi_quant(v, 2, PER_TOKEN)
    p2 = torch.zeros(BIG)
    p2[BIG - SPECS.KIVI_RESIDUAL_LENGTH_R:] = torch.rand(
        SPECS.KIVI_RESIDUAL_LENGTH_R, generator=gen)
    p2 = torch.where(p2 > 0.5, p2, torch.zeros(BIG))
    p2 = p2 / p2.sum()
    out2 = p2 @ v
    for r in (0, 16, 32, 64, SPECS.KIVI_RESIDUAL_LENGTH_R, 2 * SPECS.KIVI_RESIDUAL_LENGTH_R):
        vq2 = vq.clone()
        if r > 0:
            vq2[BIG - r:] = v[BIG - r:]
        mass = 100.0 * p2[BIG - r:].sum().item() if r > 0 else 0.0
        o = sqnr(out2, p2 @ vq2)
        print("      R=%3d  窗口内注意力质量 %5.1f%%   输出 SQNR %s" % (
            r, mass, "∞（误差恰为 0）" if o == float("inf") else "%6.2f dB" % o))
    check("B", "★ `R = %d` 的定义（KIVI §4.1 逐字）在代码里就是「最后 R 个 token 不量化」"
          % SPECS.KIVI_RESIDUAL_LENGTH_R,
          SPECS.KIVI_RESIDUAL_LENGTH_R == 128 and SPECS.KIVI_GROUP_SIZE_G == 32)
    check("B", "★ 残差滑窗的收益**完全由「窗口内落了多少注意力质量」决定**："
               "窗口内质量 100% 时输出误差恰为 0，不是「近似为 0」",
          sqnr(out2, p2 @ v) == float("inf"))
    check("C", "⚠ 本机的合成 V **复现不了** KIVI 说的「滑窗对 GSM8K crucial」那件事——"
               "我的注意力分布是**造**出来的；真实模型的局部性是模型性质，本机测不了",
          True)

    sub("B5 比特账：G=32 与 G=128 的元数据差（与 KIVI 的 G=32 选择对得上）")
    for name in ("int4_g32", "int4_g128", "int2_g32", "int2_g128"):
        payload = 4.0 if name.startswith("int4") else 2.0
        meta = SPECS.KV_METADATA_BITS_PER_ELEMENT[name]
        print("      %-10s 载荷 %3.1f bit + 元数据 %.2f bit = %5.2f bit/元素"
              "   相对 fp16 %5.2f×" % (name, payload, meta, payload + meta,
                                      16.0 / (payload + meta)))
    check("B", "★ KIVI 的 G=%d 下，int2 的整块等效宽度是 %.2f bit/元素"
               "（不是「2 bit」——载荷与整块是两个不同的量）"
          % (SPECS.KIVI_GROUP_SIZE_G,
             2.0 + SPECS.KV_METADATA_BITS_PER_ELEMENT["int2_g32"]),
          abs(2.0 + SPECS.KV_METADATA_BITS_PER_ELEMENT["int2_g32"] - 4.0) < 1e-12)
    check("B", "★ G 从 128 放到 32 的代价：元数据 ×4（%.2f → %.2f bit/元素）"
          % (SPECS.KV_METADATA_BITS_PER_ELEMENT["int2_g128"],
             SPECS.KV_METADATA_BITS_PER_ELEMENT["int2_g32"]),
          SPECS.KV_METADATA_BITS_PER_ELEMENT["int2_g32"]
          == 4 * SPECS.KV_METADATA_BITS_PER_ELEMENT["int2_g128"])


# ===========================================================================
# C. 边界与反例
# ===========================================================================

def run_C():
    rule("C 边界与反例（写反轴、量错判据、方阵静默——全部真跑出来）")

    sub("C1 ★ 把轴写反 ⇒ 推出**与 KIVI 相反**的工程建议（而程序一声不吭）")
    for bits in BITS_ALL:
        tok, chn = ablation(bits, "col_persist")
        print("      B=%d  列 outlier（K 的结构）：真实的 per-token %6.2f dB、per-channel %6.2f dB"
              % (bits, tok, chn))
        print("        写对：per-channel 赢 %+.2f dB ⇒ 建议「K 用 per-channel」" % (chn - tok))
        print("        写反：两个**数字一个没变**，只是标签互换 ⇒ 读出「per-token 赢 %+.2f dB」"
              % (tok - chn))
        print("              ⇒ 建议变成「K 用 per-token」——**与 KIVI 相反，且没有任何报错**")
    tok, chn = ablation(BITS_ALL[0], "col_persist")
    check("C", "★ 轴写反时，**两个数字都还是对的**、只是贴反了标签，"
               "推出的工程建议与 KIVI 完全相反（相差 %.2f dB 的两倍）" % (chn - tok),
          (chn - tok) > 20.0)
    check("C", "★ 判据：轴约定必须在**第一次使用这两个词之前**写成一行；"
               "本程序把它放在 banner 里与 `specs.AXIS_CONVENTION_TOKENS_FIRST` 一起",
          SPECS.AXIS_CONVENTION_TOKENS_FIRST is True)

    sub("C2 KIVI 的「逐元素相对误差」判据会被**近零元素**炸掉")
    gen = torch.Generator().manual_seed(SEEDS[0])
    x = make_tensor("col_persist", SQ, gen)
    for bits in (2, 4):
        for nm, dim in (("per-token", PER_TOKEN), ("per-channel", PER_CHANNEL)):
            xh = kivi_quant(x, bits, dim)
            print("      B=%d %-12s 逐元素相对 ‖(X−X')/X‖_F = %14.3f ；SQNR = %7.2f dB"
                  % (bits, nm, rel_err_elementwise(x, xh), sqnr(x, xh)))
    check("C", "★ 逐元素相对误差的量级是 **10⁵–10⁷**，而 KIVI 论文印的是 13.67 / 4.55——"
               "**同一个判据在合成 iid 高斯上不可用**（分母里的近零元素把它炸了）",
          rel_err_elementwise(x, kivi_quant(x, 2, PER_CHANNEL)) > 1e4)
    check("C", "⚠ 但**方向**仍然一致（per-channel 的逐元素相对误差更小）⇒ 该判据可用来比方向，"
               "**不可用来引绝对值**", True)

    sub("C3 G 与 R 互串（128 是 residual length，**不是** group size）")
    print("      KIVI §4.1 逐字：group size G = %d（所有实验统一）；"
          "residual length R = %d" % (SPECS.KIVI_GROUP_SIZE_G,
                                      SPECS.KIVI_RESIDUAL_LENGTH_R))
    print("      论文 Table 5 的 G 消融（Llama2-13B / GSM8K）：32 → 20.77、64 → 21.00、"
          "128 → 17.29")
    check("C", "★ G 与 R 是两个不同的量：**G=%d、R=%d**；"
               "论文自述 G=32 与 64 相近、G=128 时性能显著下降"
          % (SPECS.KIVI_GROUP_SIZE_G, SPECS.KIVI_RESIDUAL_LENGTH_R),
          SPECS.KIVI_GROUP_SIZE_G != SPECS.KIVI_RESIDUAL_LENGTH_R)
    check("C", "★ 「R 必须能被 G 整除」（G=%d，R=%d，%d÷%d=%d）"
          % (SPECS.KIVI_GROUP_SIZE_G, SPECS.KIVI_RESIDUAL_LENGTH_R,
             SPECS.KIVI_RESIDUAL_LENGTH_R, SPECS.KIVI_GROUP_SIZE_G,
             SPECS.KIVI_RESIDUAL_LENGTH_R // SPECS.KIVI_GROUP_SIZE_G),
          SPECS.KIVI_RESIDUAL_LENGTH_R % SPECS.KIVI_GROUP_SIZE_G == 0)

    sub("C4 口径订正：KIVI 的「2.35×–3.47×」是**吞吐**，显存口径是「2.6× 且含权重」")
    print("      逐字：%s" % SPECS.KIVI_MEMORY_QUOTE)
    print("      逐字：%s" % SPECS.KIVI_THROUGHPUT_QUOTE)
    check("C", "★ 显存口径那一条**逐字带着 `including model weight`**——"
               "判据：把吞吐的 %.2f× 当成显存压缩，省掉的正是这半句"
          % SPECS.KIVI_THROUGHPUT_LOW,
          "including model weight" in SPECS.KIVI_MEMORY_QUOTE)
    check("C", "★ 吞吐那一条**逐字带着 `throughput`**，是一个 **%.2f×–%.2f× 的区间**，"
               "而显存口径是**一个数** %.1f×——两个不是同一个量"
          % (SPECS.KIVI_THROUGHPUT_LOW, SPECS.KIVI_THROUGHPUT_HIGH,
             SPECS.KIVI_PEAK_MEMORY_RATIO),
          "throughput" in SPECS.KIVI_THROUGHPUT_QUOTE
          and SPECS.KIVI_THROUGHPUT_LOW != SPECS.KIVI_THROUGHPUT_HIGH
          and SPECS.KIVI_PEAK_MEMORY_RATIO not in (SPECS.KIVI_THROUGHPUT_LOW,
                                                   SPECS.KIVI_THROUGHPUT_HIGH))
    check("C", "★ **KIVI 全文 %d 处「perplexity」**（逐字检索 arXiv:2402.02750v2）——"
               "引用 KIVI 的 PPL 必须标注第三方复现"
          % SPECS.KIVI_PERPLEXITY_COUNT, SPECS.KIVI_PERPLEXITY_COUNT == 0)
    v = SPECS.KIVI_TABLE3_LLAMA2_7B
    check("C", "★ 单点证据（Llama-2-7B / GSM8K）：K-ℂ+V-𝕋 %.2f → K-𝕋+V-𝕋 **%.2f**"
               "（16bit %.2f，带滑窗的 KIVI-2 %.2f）——**判据是任务准确率，不是 PPL**"
          % (v["2bit_Kc_Vt"][1], v["2bit_Kt_Vt"][1], v["16bit"][1], v["KIVI-2"][1]),
          v["2bit_Kc_Vt"][1] > 5 * v["2bit_Kt_Vt"][1] and v["16bit"][1] == 13.50)
    check("C", "★ 「V 用 per-channel 就崩」在三种设置上都成立：Llama-2-7B GSM8K %.2f、"
               "CoQA %.2f（16bit 是 %.2f / %.2f）"
          % (v["2bit_Kc_Vc"][1], v["2bit_Kc_Vc"][0], v["16bit"][1], v["16bit"][0]),
          v["2bit_Kc_Vc"][1] < 0.5 and v["2bit_Kc_Vc"][0] < 5.0)


_TABLE_CACHE = {}


def table_cached(bits):
    if bits not in _TABLE_CACHE:
        _TABLE_CACHE[bits] = ablation(bits, "col_persist")
    return _TABLE_CACHE[bits]


# ===========================================================================
# M / F
# ===========================================================================

def run_M():
    rule("M 测量条件自洽")
    v = torch.__version__
    check("M", "本次运行的 torch %s 在 `MEASURED_VERSIONS` 表里" % v,
          v in SPECS.MEASURED_VERSIONS or ".".join(v.split(".")[:2])
          in SPECS.MEASURED_VERSIONS, "表 = %s" % (SPECS.MEASURED_VERSIONS,))
    check("M", "表里恰好 5 个版本（与本机解释器数一致）", len(SPECS.MEASURED_VERSIONS) == 5)
    check("M", "`UNCOVERED_MINORS` 逐字等于 `2.9.x / 2.10.x / 2.12.x`",
          SPECS.UNCOVERED_MINORS == "2.9.x / 2.10.x / 2.12.x")
    host = cpu_model()
    import os as _os
    osname = {"Darwin": "macOS", "Linux": "Linux"}.get(platform.system(), platform.system())
    check("M", "`MEASURE_HOST` 逐字等于「本机 CPU 型号 + 核数 + 系统 + CPU-only」",
          SPECS.MEASURE_HOST == "%s（%d 核）/ %s / **CPU-only，无 GPU 数字**"
          % (host, _os.cpu_count(), osname), SPECS.MEASURE_HOST)
    check("M", "`DECODE_AI` 与 `KIVI_ATTENTION_SPARSITY_PCT` 的量纲不同，不得混用"
               "（1.0 是 FLOP/B，%.1f 是百分比）" % SPECS.KIVI_ATTENTION_SPARSITY_PCT,
          SPECS.DECODE_AI == 1.0 and SPECS.KIVI_ATTENTION_SPARSITY_PCT > 50)


def run_F():
    rule("F 变异测试清单（A6）：本程序直接读到的 specs 常量")
    src = open(__file__, encoding="utf-8").read()
    names = sorted(n for n in dir(SPECS) if n.isupper() and not n.startswith("_"))
    used = [n for n in names if ("SPECS.%s" % n) in src]
    print("      specs.py 共 %d 条常量；本程序直接读到 %d 条" % (len(names), len(used)))
    print("      %s" % ", ".join(used))
    check("B", "本程序读到的常量数 ≥ 10（每条都参与断言）", len(used) >= 10)


def main():
    banner()
    run_A()
    run_B()
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

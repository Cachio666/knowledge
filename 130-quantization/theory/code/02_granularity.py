#!/usr/bin/env python3
"""02 粒度：per-token 只治「行」，per-channel 只治「列」，二者不可互替。

对应文档：130-quantization/theory/02-granularity.md

★ 本程序跑的是**真 torch**——输出是**本机实测值**，必须绑定版本。
  本机五个解释器都跑过（见 01 的 docstring 表）。**2.9 / 2.10 / 2.12 不声称覆盖。**
  全部 CPU-only，无 GPU 数字。

★★ **本程序最重要的一件事是把轴约定写死**（P9 实例 2）：
  张量按 **`[tokens, features]`** 排布（transformers 的通行布局）。
    · 「**每行**一个 scale」 = 沿 **axis 1** 归约 = **per-token**
    · 「**每列**一个 scale」 = 沿 **axis 0** 归约 = **per-channel**
  计划 §0.7.B 的初版把这两个标签**写反了**。写反**不会有任何报错**，
  只会推出「K 应该用 per-token」——与 KIVI 正好相反。
  所以本程序 §A1 把「标签与归约轴的对应」单独立成一条断言：
  **把标签换过来，断言就会失败。**

三类断言（CONTRIBUTING §2 第三种形态）：
  B —— **代价模型**：粒度换来的 dB vs scale 元数据花掉的 bit。
  A —— 两条独立通道：(a) 手写 RTN；(b) torch 的 `quantize_per_tensor` /
       `quantize_per_channel` 库路径。两者必须给出同一个结论。
  C —— 失败行为：轴写反会推出相反的结论、per-group 的切分维度不写明就不可比、
       「# groups = 128」当成「group size = 128」会差一个数量级。

用法：cd code && make 02_granularity   （或 make run 全跑）
"""

import math
import platform
import sys

import torch

import specs as SPECS


_STATE = {"n": 0, "A": 0, "B": 0, "C": 0, "fail": []}

M = N = 4096          # 张量规模（列异质的 10^U(-3,0) 尺度因子在 4096 宽上才铺得开）
QMAX = 127.0          # 对称 8-bit 的正侧上限
GS = 128              # per-group 的组大小
SEED = SPECS.GRANULARITY_TABLE_SEED


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

def sqnr_db(x, xh):
    """SQNR = 10log10(Σx²/Σ(x−x̂)²)。**判据：全局逐元素平方和之比，无归一化。**"""
    x = x.double()
    xh = xh.double()
    return (10.0 * torch.log10((x * x).sum() / ((x - xh) ** 2).sum())).item()


def rtn(x, s):
    """手写对称 RTN：`clip(round(x/s), −127, 127)·s`。**判据：restricted range（用 −127 而非 −128）。**"""
    return torch.clamp(torch.round(x / s), -QMAX, QMAX) * s


def scale_of(x, dim):
    """沿 `dim` 归约出 max|x| / 127。**`dim=None` 表示 per-tensor。**"""
    if dim is None:
        return x.abs().max() / QMAX
    return x.abs().max(dim=dim, keepdim=True).values / QMAX


def quantize(m, mode, gs=GS):
    """四种粒度的**手写参考实现**（第一通道）。

    · "token"   —— 沿 axis 1 归约（**每行一个 scale**）
    · "channel" —— 沿 axis 0 归约（**每列一个 scale**）
    · "group"   —— 每行内沿 **feature（最后一维）** 每 gs 个一组
    · "group_tok" —— 每列内沿 **token（第 0 维）** 每 gs 个一组（**另一个切分维度**）
    """
    if mode == "tensor":
        return rtn(m, scale_of(m, None))
    if mode == "token":
        return rtn(m, scale_of(m, 1))
    if mode == "channel":
        return rtn(m, scale_of(m, 0))
    if mode == "group":
        r = m.reshape(m.shape[0], m.shape[1] // gs, gs)
        return rtn(r, scale_of(r, 2)).reshape(m.shape)
    if mode == "group_tok":
        r = m.reshape(m.shape[0] // gs, gs, m.shape[1])
        return rtn(r, scale_of(r, 1)).reshape(m.shape)
    if mode == "flat":
        r = m.reshape(-1, gs)
        return rtn(r, scale_of(r, 1)).reshape(m.shape)
    raise ValueError(mode)


def make_scenarios():
    """五个场景。判据（**不写出来这些数就不可复现**）：
       · iid 高斯 / 拉普拉斯：直接采样
       · 行异质：每**行**乘一个 10^U(−3,0) 的尺度因子
       · 列异质：每**列**乘一个 10^U(−3,0) 的尺度因子
       · outlier：**随机 7 列 × 100**（⚠ 位置随机 —— 取「前 7 列」会得到完全不同的 per-group 数）"""
    g = torch.Generator().manual_seed(SEED)
    out = {}
    out["iid 高斯"] = torch.randn(M, N, generator=g)
    out["iid 拉普拉斯"] = (torch.empty(M, N).exponential_(1.0, generator=g)
                          * (torch.randint(0, 2, (M, N), generator=g).float() * 2 - 1))
    out["行异质（每 token 尺度跨 1e-3..1e0）"] = (
        torch.randn(M, N, generator=g) * torch.pow(10.0, torch.rand(M, 1, generator=g) * 3 - 3))
    out["列异质（每 channel 尺度跨 1e-3..1e0）"] = (
        torch.randn(M, N, generator=g) * torch.pow(10.0, torch.rand(1, N, generator=g) * 3 - 3))
    sc = torch.ones(N)
    sc[torch.randperm(N, generator=g)[:7]] = 100.0     # 随机 7 列
    out["7/4096 列（随机位置）×100"] = torch.randn(M, N, generator=g) * sc
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
    print("  张量规模   : (%d, %d)，对称 8-bit RTN，QMAX = %g，group size = %d"
          % (M, N, QMAX, GS))
    print("  随机种子   : %d（**per-tensor 那一列对种子敏感，见 §B3**）" % SEED)
    print()
    print("  ★ 轴约定（**必须先写死**，否则 per-token / per-channel 两个标签会互换）：")
    print("      %s" % SPECS.AXIS_CONVENTION["layout"])
    print("      %s" % SPECS.AXIS_CONVENTION["row_scale"])
    print("      %s" % SPECS.AXIS_CONVENTION["col_scale"])


# ---------------------------------------------------------------------------
# §A 判据复核（两条独立通道）
# ---------------------------------------------------------------------------

def run_A(scen):
    rule("§A  判据复核：两条独立通道（手写 RTN vs torch 的库路径）")

    sub("A1  ★ 轴约定：**标签与归约轴的对应**本身就是一条可失败的断言")
    # 行异质数据上，「每行一个 scale」必须赢；列异质上，「每列一个 scale」必须赢。
    row_h = scen["行异质（每 token 尺度跨 1e-3..1e0）"]
    col_h = scen["列异质（每 channel 尺度跨 1e-3..1e0）"]
    row_axis1 = sqnr_db(row_h, quantize(row_h, "token"))      # 每行（axis 1）
    row_axis0 = sqnr_db(row_h, quantize(row_h, "channel"))    # 每列（axis 0）
    col_axis1 = sqnr_db(col_h, quantize(col_h, "token"))
    col_axis0 = sqnr_db(col_h, quantize(col_h, "channel"))
    print("      行异质数据：每行一个 scale %.2f dB ｜ 每列一个 scale %.2f dB"
          % (row_axis1, row_axis0))
    print("      列异质数据：每行一个 scale %.2f dB ｜ 每列一个 scale %.2f dB"
          % (col_axis1, col_axis0))
    check("A", "★ 行异质数据上「**每行**一个 scale」赢 %.2f dB ⇒ 它治的是**行**"
          % (row_axis1 - row_axis0), row_axis1 - row_axis0 > 5.0)
    check("A", "★ 列异质数据上「**每列**一个 scale」赢 %.2f dB ⇒ 它治的是**列**"
          % (col_axis0 - col_axis1), col_axis0 - col_axis1 > 5.0)
    # specs 里的标签定义：每行 = per-token、每列 = per-channel。若把 specs 改反，下面两条失败。
    check("A", "★ specs 的轴约定说「每行 = per-token」，而实测的赢家是**行** ⇒ "
          "「per-token 治行异质」为真，其否定为假",
          "每行一个 scale" in SPECS.AXIS_CONVENTION["row_scale"]
          and "per-token" in SPECS.AXIS_CONVENTION["row_scale"]
          and row_axis1 > row_axis0)
    check("A", "★ 同理「per-channel 治列异质」为真",
          "每列一个 scale" in SPECS.AXIS_CONVENTION["col_scale"]
          and "per-channel" in SPECS.AXIS_CONVENTION["col_scale"]
          and col_axis0 > col_axis1)

    sub("A2  5×4 的误差表：手写 RTN vs specs（specs 即本断言的参照）")
    got = {}
    for name, m in scen.items():
        got[name] = tuple(sqnr_db(m, quantize(m, k))
                          for k in ("tensor", "token", "channel", "group"))
        want = SPECS.GRANULARITY_TABLE[name]
        print("      %-30s %7.2f %7.2f %7.2f %7.2f" % ((name,) + got[name]))
        for i, col in enumerate(("per-tensor", "per-token", "per-channel", "per-group")):
            if col == "per-tensor":
                continue    # 单独在 A3/B3 里处理（它对种子敏感）
            check("A", "%s / %s：%.2f dB（specs 记 %.2f）" % (name, col, got[name][i], want[i]),
                  abs(got[name][i] - want[i]) <= SPECS.GRANULARITY_REPRO_TOL_DB,
                  "容差 ±%.2f dB" % SPECS.GRANULARITY_REPRO_TOL_DB)

    sub("A3  per-tensor 那一列：**它是随机变量，只做宽容差断言与定性断言**")
    for name, m in scen.items():
        want = SPECS.GRANULARITY_TABLE[name][0]
        check("B", "%s / per-tensor：%.2f dB（specs 记 %.2f，**宽容差 ±2.5 dB**）"
              % (name, got[name][0], want), abs(got[name][0] - want) < 2.5,
              "计划 §0.7.B 明确要求：**以「per-tensor 最差」的定性结论为准，不逐格引这三个数**")
    check("A", "★ 定性：**每一个场景里 per-tensor 都是最差（或并列最差）的那一列**",
          all(got[n][0] <= min(got[n][1], got[n][2], got[n][3]) + 0.01 for n in scen),
          "逐场景：%s" % {n: round(got[n][0] - min(got[n][1:]), 2) for n in scen})

    sub("A4  ★ 第二条独立通道：torch 的 `quantize_per_tensor` / `quantize_per_channel`")
    m = scen["iid 高斯"]
    s = m.abs().max() / QMAX
    lib_t = torch.quantize_per_tensor(m, float(s), 0, torch.qint8).dequantize()
    mine_t = rtn(m, s)
    check("A", "per-tensor：库路径 %.2f dB vs 手写 RTN %.2f dB（差 %.3f）"
          % (sqnr_db(m, lib_t), sqnr_db(m, mine_t), sqnr_db(m, lib_t) - sqnr_db(m, mine_t)),
          abs(sqnr_db(m, lib_t) - sqnr_db(m, mine_t)) < 0.02,
          "两条**不同的代码路径**（ATen 的 quantizer vs 本程序的 round/clip）")
    sc = m.abs().max(dim=0, keepdim=True).values / QMAX
    # ★ 库 API 的 axis 语义：`axis=k` 的 scales 长度必须是 `size(k)`，且**沿 k 维索引取 scale**。
    #   所以「每列一个 scale」（长度 = size(1) = N）对应的是 **axis=1**，不是 axis=0。
    lib_c = torch.quantize_per_channel(
        m, sc.squeeze(0), torch.zeros(N, dtype=torch.long), 1, torch.qint8).dequantize()
    mine_c = rtn(m, sc)
    check("A", "per-channel：库路径 %.2f dB vs 手写 RTN %.2f dB（差 %.3f）"
          % (sqnr_db(m, lib_c), sqnr_db(m, mine_c), sqnr_db(m, lib_c) - sqnr_db(m, mine_c)),
          abs(sqnr_db(m, lib_c) - sqnr_db(m, mine_c)) < 0.02,
          "scale 长度 = size(1) ⇒ **库的 `axis=1`** 才等于「每列一个 scale」")
    check("A", "★ 库路径也复现「per-channel 在列异质数据上赢」这一结论",
          sqnr_db(scen["列异质（每 channel 尺度跨 1e-3..1e0）"],
                  torch.quantize_per_channel(
                      scen["列异质（每 channel 尺度跨 1e-3..1e0）"],
                      (scen["列异质（每 channel 尺度跨 1e-3..1e0）"].abs().max(dim=0).values
                       / QMAX), torch.zeros(N, dtype=torch.long), 1,
                      torch.qint8).dequantize()) > col_axis1)
    # 第二条通道也覆盖 per-token（库的 axis=0）
    sc_row = m.abs().max(dim=1).values / QMAX
    lib_r = torch.quantize_per_channel(
        m, sc_row, torch.zeros(M, dtype=torch.long), 0, torch.qint8).dequantize()
    mine_r = rtn(m, sc_row.unsqueeze(1))
    check("A", "per-token：库路径 %.2f dB vs 手写 RTN %.2f dB（差 %.3f）"
          % (sqnr_db(m, lib_r), sqnr_db(m, mine_r), sqnr_db(m, lib_r) - sqnr_db(m, mine_r)),
          abs(sqnr_db(m, lib_r) - sqnr_db(m, mine_r)) < 0.02,
          "scale 长度 = size(0) ⇒ 库的 `axis=0` = 「每行一个 scale」")

    sub("A5  ★ outlier 在哪一维，粒度就该给哪一维（判据：**列** outlier）")
    m = scen["7/4096 列（随机位置）×100"]
    tok = sqnr_db(m, quantize(m, "token"))
    chn = sqnr_db(m, quantize(m, "channel"))
    check("A", "★ 列（feature 维）outlier 下：per-channel %.2f dB vs per-token %.2f dB，"
          "**per-channel 多 %.2f dB**" % (chn, tok, chn - tok), chn - tok > 20.0,
          "**这一格是 KIVI「K 必须用 per-channel」的理论根据**")


# ---------------------------------------------------------------------------
# §B 代价模型
# ---------------------------------------------------------------------------

def run_B(scen):
    rule("§B  代价模型：粒度换来的 dB vs scale 元数据花掉的 bit")

    sub("B1  每种粒度的 scale 个数与元数据开销（判据：fp32 scale = 32 bit）")
    kinds = [("per-tensor", 1), ("per-token", M), ("per-channel", N),
             ("per-group(%d)" % GS, M * (N // GS))]
    for name, cnt in kinds:
        print("      %-16s scale 个数 %9d，共 %8.4f MB，摊到每元素 %.4f bit"
              % (name, cnt, cnt * 4 / 1024.0 ** 2, 32.0 * cnt / (M * N)))
    check("B", "★ per-token 与 per-channel 的 scale 数**相同**（都是 4096 个 ⇒ 0.25 bit/元素），"
          "差别只在**沿哪一维**——这正是 §A5 那 21 dB 的来源",
          M == N)

    sub("B2  每花 1 bit 的元数据买到多少 dB（判据：同一场景内，相对 per-tensor）")
    base = {n: 32.0 / (M * N) for n in scen}
    for name, m in scen.items():
        t = sqnr_db(m, quantize(m, "tensor"))
        g = sqnr_db(m, quantize(m, "group"))
        extra = 32.0 * (M * (N // GS)) / (M * N)
        print("      %-30s per-tensor→per-group(%d)：+%5.2f dB，代价 %.4f bit/元素，"
              "**%.1f dB / bit**" % (name, GS, g - t, extra, (g - t) / extra))
    check("B", "★ 同样的 0.25 bit/元素，在 outlier 场景买到的 dB 远多于 iid 场景 "
          "—— **这条边（粒度 vs scale 开销）的回报取决于数据，不取决于粒度本身**",
          (got_b(scen, "7/4096 列（随机位置）×100")) > 3 * (got_b(scen, "iid 高斯")),
          "见上表最后两列")

    sub("B3  per-tensor 的 SQNR 是随机变量（判据：换 5 个种子看 per-tensor 那一格）")
    for seed in (1, 2, 3, 4, 5):
        mm = make_scenarios_with_seed(seed)["iid 高斯"]
        print("      seed=%d：per-tensor %.2f dB ｜ per-token %.2f ｜ per-channel %.2f ｜ "
              "per-group %.2f" % ((seed, sqnr_db(mm, quantize(mm, "tensor")),
                                   sqnr_db(mm, quantize(mm, "token")),
                                   sqnr_db(mm, quantize(mm, "channel")),
                                   sqnr_db(mm, quantize(mm, "group")))))
    vals = [sqnr_db(make_scenarios_with_seed(s)["iid 高斯"], quantize(
        make_scenarios_with_seed(s)["iid 高斯"], "tensor")) for s in (1, 2, 3)]
    check("B", "★ per-tensor 三格极差 %.2f dB —— **而 per-group 那一列在 specs 里是逐格固定的**"
          % (max(vals) - min(vals)), max(vals) - min(vals) > 0.1,
          "成因：per-tensor 的 scale 由**整张量的极值**决定")


def got_b(scen, name):
    m = scen[name]
    return sqnr_db(m, quantize(m, "group")) - sqnr_db(m, quantize(m, "tensor"))


def make_scenarios_with_seed(seed):
    global SEED
    old = SEED
    SEED = seed
    try:
        return make_scenarios()
    finally:
        SEED = old


# ---------------------------------------------------------------------------
# §C 边界与反例
# ---------------------------------------------------------------------------

def run_C(scen):
    rule("§C  边界与反例：下面每一条都**必须真的跑出问题**")

    sub("C1  ★ 轴写反**不会有任何报错**，只会推出相反的结论 —— 这里把它跑出来")
    m = scen["7/4096 列（随机位置）×100"]
    tok = sqnr_db(m, quantize(m, "token"))
    chn = sqnr_db(m, quantize(m, "channel"))
    # 「K 应该用 per-token」这个（错的）结论，正是把标签写反后必然得到的
    wrong_conclusion = tok > chn
    check("C", "★ 「把标签写反」推出的结论是「per-token 更好」= %s，"
          "而实测 per-channel 好 %.2f dB ⇒ **那个结论为假**" % (wrong_conclusion, chn - tok),
          not wrong_conclusion and chn - tok > 20.0,
          "写反时**没有任何报错**：两个数各自都对，只是标签错了——这就是 P9 的形状")

    sub("C2  ★ per-group 的**切分维度**不写明，group size 就不可比")
    m = scen["7/4096 列（随机位置）×100"]
    along_feat = sqnr_db(m, quantize(m, "group"))
    along_tok = sqnr_db(m, quantize(m, "group_tok"))
    flat = sqnr_db(m, quantize(m, "flat"))
    print("      沿 feature（GPTQ 口径） %.2f dB ｜ 沿 token（KIVI 口径） %.2f dB ｜ "
          "拉平（Dettmers 口径） %.2f dB" % (along_feat, along_tok, flat))
    check("C", "★ 同一个张量、同一个 group size = %d，只换切分维度：差 %.2f dB"
          % (GS, abs(along_tok - along_feat)), abs(along_tok - along_feat) > 5.0,
          "**不写这一维，group 相关数字就没有可比性**（计划 §0.7.B 的警告）")
    check("C", "★ specs 里三个来源的切分维度确实不同",
          len(set(SPECS.PER_GROUP_SPLIT_DIM.values())) == 3,
          "%s" % SPECS.PER_GROUP_SPLIT_DIM)

    sub("C3  ★ 「# groups = 128」当成「group size = 128」会差一个数量级（**纯算术**）")
    width = 768
    group_size_if_128_groups = width // SPECS.QBERT_GROUP_SWEEP["# groups"][2]
    check("C", "★ 768 宽的矩阵上，128 **组** ⇒ 每组的元素数是 **%d**，不是 128"
          % group_size_if_128_groups, group_size_if_128_groups == 6,
          "Q-BERT %s 的那张表的列名是 \"# groups\"（**组数**）" % SPECS.QBERT_ARXIV)
    check("C", "★ 把「128 组」写成「group size 128」是 **%d 倍**的作用域错配"
          % (GS // group_size_if_128_groups), GS // group_size_if_128_groups == 21,
          "**P9 实例 3**：grep 命中、来源真实、数字正确，错在它不是同一个量")
    check("C", "★ Q-BERT 的编号是 %s，**不是 %s**（后者是 HAWQ）"
          % (SPECS.QBERT_ARXIV, SPECS.QBERT_NOT_THIS_ARXIV),
          SPECS.QBERT_ARXIV == "arXiv:1909.05840"
          and SPECS.QBERT_NOT_THIS_ARXIV == "arXiv:1905.03696")
    check("C", "★ 128 组→768 组的增量是 %.2f（SST-2），**「≤0.1%%」对 SST-2 不成立**"
          % SPECS.QBERT_128_TO_768_DELTA_SST2, SPECS.QBERT_128_TO_768_DELTA_SST2 > 0.1,
          "所以引用时写「组数超过 128 后收益饱和」，不写「group size = 128 就够」")

    sub("C4  ★ 两个标签**不能互替**：per-token 在列 outlier 上救不了场")
    m = scen["7/4096 列（随机位置）×100"]
    print("      同一张张量：per-tensor %.2f ｜ per-token %.2f ｜ per-channel %.2f ｜ "
          "per-group(128) %.2f"
          % (sqnr_db(m, quantize(m, "tensor")), sqnr_db(m, quantize(m, "token")),
             sqnr_db(m, quantize(m, "channel")), sqnr_db(m, quantize(m, "group"))))
    check("C", "★ per-token 比 per-tensor 只好 %.2f dB，而 per-channel 好 %.2f dB "
          "—— **粒度必须给到 outlier 所在的那一维**"
          % (sqnr_db(m, quantize(m, "token")) - sqnr_db(m, quantize(m, "tensor")),
             sqnr_db(m, quantize(m, "channel")) - sqnr_db(m, quantize(m, "tensor"))),
          (sqnr_db(m, quantize(m, "channel")) - sqnr_db(m, quantize(m, "tensor")))
          > 2 * (sqnr_db(m, quantize(m, "token")) - sqnr_db(m, quantize(m, "tensor"))))

    sub("C5  ★ 库 API 的 `axis` 语义：**方阵上写错轴不会报错，只会静默换一个东西**")
    g2 = torch.Generator().manual_seed(31)
    # (a) 非方阵：轴写错会**响亮地失败**
    rect = torch.randn(64, 7, generator=g2)
    raised = False
    try:
        torch.quantize_per_channel(rect, rect.abs().max(dim=0).values / QMAX,
                                   torch.zeros(7, dtype=torch.long), 0, torch.qint8)
    except RuntimeError as exc:
        raised = True
        msg = str(exc)
    check("C", "★ 非方阵 [64, 7] 上传「长度 7 的 scale + axis=0」**真的抛异常**：%s"
          % ("yes" if raised else "no"), raised, msg[:70] if raised else "")
    # (b) 方阵：同一份 scale、只换 axis，两者**都接受**但量的是两件不同的事
    sq = scen["列异质（每 channel 尺度跨 1e-3..1e0）"]
    s_col = sq.abs().max(dim=0).values / QMAX          # 每列一个 scale（长度 N）
    a0 = sqnr_db(sq, torch.quantize_per_channel(sq, s_col,
                                                torch.zeros(N, dtype=torch.long),
                                                0, torch.qint8).dequantize())
    a1 = sqnr_db(sq, torch.quantize_per_channel(sq, s_col,
                                                torch.zeros(N, dtype=torch.long),
                                                1, torch.qint8).dequantize())
    check("C", "★ 方阵上同一份 scale 只换 axis：axis=0 给 %.2f dB，axis=1 给 %.2f dB，"
          "差 %.2f dB —— **两个都被接受，没有任何报错**" % (a0, a1, abs(a0 - a1)),
          abs(a0 - a1) > 3.0,
          "**axis 的语义是「scale 长度 = size(axis)、沿 axis 索引取 scale」**；"
          "方阵上两个方向长度相同，错的那个不会报错")
    col_tok_ref = sqnr_db(sq, quantize(sq, "token"))     # 每行一个 scale
    col_chn_ref = sqnr_db(sq, quantize(sq, "channel"))   # 每列一个 scale
    check("C", "★ `axis=1` 给出的 %.2f dB 与「每列一个 scale」那一格（%.2f）**逐位吻合**"
          % (a1, col_chn_ref), abs(a1 - col_chn_ref) < 0.05)
    check("C", "★★ 而 `axis=0` 给出的 %.2f dB **既不是每列（%.2f）也不是每行（%.2f）** —— "
          "它把**列 j 的 scale 用在了行 j 上**，得到一个看起来像结果的数"
          % (a0, col_chn_ref, col_tok_ref),
          abs(a0 - col_chn_ref) > 3.0 and abs(a0 - col_tok_ref) > 3.0,
          "**这才是最危险的形态**：写错轴不只是「换了另一种粒度」，而是给出一个无意义的数，"
          "而它照样会打印出一行漂亮的小数")

    sub("C6  单边分布上对称量化**必然差**（outlier「要么全正要么全负」的另一面）")
    g = torch.Generator().manual_seed(77)
    x = torch.nn.functional.relu(torch.randn(1 << 20, generator=g))     # ReLU 后的激活
    s = x.abs().max() / QMAX
    span = float((x.max() - x.min()).item())
    asym = torch.quantize_per_tensor(x, span / 255.0,
                                     int(round(-x.min().item() / (span / 255.0))),
                                     torch.quint8).dequantize()
    check("C", "★ ReLU 后的激活：**对称 %.2f dB vs 非对称 %.2f dB，非对称多 %.2f dB**"
          % (sqnr_db(x, rtn(x, s)), sqnr_db(x, asym), sqnr_db(x, asym) - sqnr_db(x, rtn(x, s))),
          sqnr_db(x, asym) - sqnr_db(x, rtn(x, s)) > 3.0,
          "**对称量化的收益取决于分布偏斜**；outlier「要么全正要么全负」正是它特别有效的理由")

    sub("C7  本程序**不**声称的东西")
    print("      · 本机无 GPU、无 LLM 权重：**不给任何 perplexity / 端到端数字**。")
    print("      · 这张表是**合成实验**，支撑的是「粒度给到 outlier 那一维」这个**机制**结论，")
    print("        不是「某个模型上能拿到多少 dB」这个系统结论（CONTRIBUTING §2 的第四类数字限定语）。")
    print("      · KIVI 的 V 必须 per-token 有**另一个**理由（attention 分数稀疏），")
    print("        **本程序没有复现它** —— 见文档 §3 的说明。")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    print("=" * 76)
    print("130-quantization/theory/02 粒度 —— 真 torch 实测（B 类在内部）")
    print("=" * 76)
    banner()
    scen = make_scenarios()
    run_A(scen)
    run_B(scen)
    run_C(scen)

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
    print("  ⚠ 本程序的 A/C 类与 `verify.py` 的那一份是**两档不同的证据**：")
    print("     本程序跑真 torch 张量（4096²，五个版本逐格一致）；")
    print("     `verify.py` 在默认 python3 下用**标准库**跑一个小规模版本（不 import torch）。")
    print("     **两者不得互相背书**，且**规模不同 ⇒ 数字不同，只有定性结构必须一致**。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

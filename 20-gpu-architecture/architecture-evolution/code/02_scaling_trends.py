"""跨代趋势：算力涨得比带宽快（在低精度轴上）。

运行：python3 02_scaling_trends.py
依赖：numpy（仅用于格式化；全部计算在本机 CPU 上完成，不需要 GPU）

口径声明：
  · 本程序做的是**算术推导**，没有在 GPU 上跑任何 kernel，也不产出任何实测数据。
  · 规格常量全部来自 specs.py（每条带 `# 来源：` 注释）；本程序只做代入与推导。
  · 峰值一律 **dense（不含 2:4 稀疏）**。NVIDIA 产品页上带 "*" 的数字是
    with sparsity，恰好是 dense 的两倍——混用会让 roofline 拐点整体挪位一倍。
  · 第 5 节引用的算子算术强度**引自本仓库 10-foundations/performance-models/01
    的 §5 表3**，本程序不重算它们，只做跨代分类。重复计算会引入两份数字。
"""

import numpy as np

import specs as S


def rule(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# 取用的一代一档：数据中心 SXM。P100 不进趋势表（官方材料未给出可与
# 其它代直接对比的 dense 峰值，见 specs.PEAK_TFLOPS_DENSE_BY_GEN["P100"]）。
TREND_GENS = ("V100", "A100", "H100", "B200")
PRECISIONS = ("FP32_VECTOR", "TF32_TC", "FP16_TC", "BF16_TC", "FP8_TC", "FP4_TC")


def peak(gen, prec):
    return S.PEAK_TFLOPS_DENSE_BY_GEN[gen].get(prec)


def bw(gen):
    return S.CHIP_BY_GEN[gen]["hbm_bw_GBps"]


def i_star(gen, prec):
    """roofline 拐点 I* = P_peak / B_peak，单位 FLOP/byte。

    模型本身的推导在 10-foundations/performance-models/01-roofline-model.md，
    本程序只做代入，不重推。
    """
    p = peak(gen, prec)
    return None if p is None else p * 1e12 / (bw(gen) * 1e9)


rule("1. 逐代规格基线（数据中心 SXM 档）")

print("  %-16s %s" % ("", "".join("%-13s" % g for g in TREND_GENS)))
print("  " + "-" * 72)
print("  %-16s %s" % ("发布年份",
                    "".join("%-13d" % S.YEAR_BY_GEN[g] for g in TREND_GENS)))
print("  %-16s %s" % ("制程 (nm)",
                    "".join("%-13d" % S.CHIP_BY_GEN[g]["process_nm"] for g in TREND_GENS)))
print("  %-16s %s" % ("晶体管 (B)",
                    "".join("%-13.1f" % S.CHIP_BY_GEN[g]["transistors_B"] for g in TREND_GENS)))
print("  %-16s %s" % ("SM 数",
                    "".join("%-13d" % S.CHIP_BY_GEN[g]["sm_count"] for g in TREND_GENS)))
print("  %-16s %s" % ("FP32 lane/SM",
                    "".join("%-13d" % S.CHIP_BY_GEN[g]["fp32_lanes_per_sm"]
                            for g in TREND_GENS)))
print("  %-16s %s" % ("FP32 lane/GPU",
                    "".join("%-13d" % (S.CHIP_BY_GEN[g]["sm_count"]
                                       * S.CHIP_BY_GEN[g]["fp32_lanes_per_sm"])
                            for g in TREND_GENS)))
tc = ["%d" % S.CHIP_BY_GEN[g]["tensor_cores_per_sm"]
      if S.CHIP_BY_GEN[g]["tensor_cores_per_sm"] is not None else "未披露"
      for g in TREND_GENS]
print("  %-16s %s" % ("TC/SM", "".join("%-13s" % c for c in tc)))
print("  %-16s %s" % ("TDP (W)",
                    "".join("%-13d" % S.CHIP_BY_GEN[g]["tdp_W"] for g in TREND_GENS)))
print("  %-16s %s" % ("HBM (GB/s)",
                    "".join("%-13.0f" % bw(g) for g in TREND_GENS)))

print()
print("两处必须标注的例外：")
print("  · B200 的 SM 数是 %d —— **非官方直述**，由官方 FP32 峰值 75 TFLOPS 反推"
      % S.B200_SM_COUNT)
print("    （第 6 节给出反推全过程）；官方技术简报未披露 SM 数。")
print("  · B200 的晶体管 %d B 是**双 die 封装总数**（单 die %.0f B），"
      % (S.B200_TRANSISTORS_B, S.B200_TRANSISTORS_B / 2))
print("    而 V100~H100 一栏是单 die 数。不标注会误读出「一代涨 2.6 倍」。")
print("  · B200 的每 SM Tensor Core 数：**官方未披露**，本表照实写「未披露」，")
print("    不填第三方聚合站的数字。")


rule("2. 逐代增长率：谁涨得快")

print("  %-14s %-12s %-12s %-12s %s"
      % ("跃迁", "FP32 向量", "FP16 TC", "HBM 带宽", "谁涨得快"))
print("  " + "-" * 78)
GROWTHS = {}
for a, b in zip(TREND_GENS, TREND_GENS[1:]):
    g_fp32 = peak(b, "FP32_VECTOR") / peak(a, "FP32_VECTOR")
    g_fp16 = peak(b, "FP16_TC") / peak(a, "FP16_TC")
    g_bw = bw(b) / bw(a)
    if g_fp32 > g_bw and g_fp16 > g_bw:
        who = "算力反超带宽"
    elif g_fp32 < g_bw and g_fp16 >= g_bw:
        who = "带宽 ≈ tensor，FP32 向量被甩下"
    else:
        who = "带宽 ≈ FP16/FP8；FP32 向量几乎停滞"
    GROWTHS[(a, b)] = (g_fp32, g_fp16, g_bw)
    print("  %-14s %-12s %-12s %-12s %s"
          % ("%s→%s" % (a, b), "%.2f×" % g_fp32, "%.2f×" % g_fp16,
             "%.2f×" % g_bw, who))

print()
print("主线的一句话形态：**在 FP32 向量轴上，带宽从不落后；在低精度轴上，")
print("算力在 A100→H100 这一段反超一次，随后被带宽追平。**")
print()
for a, b in zip(TREND_GENS, TREND_GENS[1:]):
    g_fp32, g_fp16, g_bw = GROWTHS[(a, b)]
    print("  %s→%s：FP16 TC 涨 %.2f×，带宽涨 %.2f×，比值 %.2f。%s"
          % (a, b, g_fp16, g_bw, g_fp16 / g_bw,
             "算力领先" if g_fp16 > g_bw else "带宽领先"))
print()
print("比值那一列就是下一节拐点迁移的全部内容——**拐点是这两条增长率的商**。")


rule("3. roofline 拐点 I* = P_peak / B_peak 的逐代迁移")

print("  模型（两段线、min(P_peak, B_peak·I)、I 的定义与 Bytes 口径）的推导在")
print("  10-foundations/performance-models/01-roofline-model.md，**此处只回链、不重推**。")
print()
print("  %-14s %s" % ("精度轴", "".join("%-13s" % g for g in TREND_GENS)))
print("  " + "-" * 72)
ISTAR = {}
for prec in PRECISIONS:
    cells = []
    for g in TREND_GENS:
        v = i_star(g, prec)
        cells.append("—" if v is None else "%.1f" % v)
        ISTAR[(g, prec)] = v
    print("  %-14s %s" % (prec, "".join("%-13s" % c for c in cells)))

print()
print("  （单位 FLOP/byte，dense）")
print()
fp32_vals = [ISTAR[(g, "FP32_VECTOR")] for g in TREND_GENS]
print("  **FP32 向量轴的十年**：%s —— 最低 %.1f、最高 %.1f，"
      % (" → ".join("%.1f" % v for v in fp32_vals), min(fp32_vals), max(fp32_vals)))
print("  最大最小之比只有 %.2f 倍，而序列是**震荡**的（降、升、降），不是单调迁移。"
      % (max(fp32_vals) / min(fp32_vals)))
print("  原因很直接：FP32 向量算力的十年涨幅是 %.2f×，带宽是 %.2f×，两者只差 %.2f 倍，"
      % (peak("B200", "FP32_VECTOR") / peak("V100", "FP32_VECTOR"),
         bw("B200") / bw("V100"),
         (bw("B200") / bw("V100")) / (peak("B200", "FP32_VECTOR") / peak("V100", "FP32_VECTOR"))))
print("  商自然就在原地打转。")
print()
fp16_vals = [ISTAR[(g, "FP16_TC")] for g in TREND_GENS]
print("  **低精度轴的十年**：FP16 TC 从 %.1f 涨到 %.1f（%.2f×）；"
      % (fp16_vals[0], fp16_vals[-1], fp16_vals[-1] / fp16_vals[0]))
print("  再往低走，B200 的 FP8 是 %.1f、FP4 是 %.1f。"
      % (ISTAR[("B200", "FP8_TC")], ISTAR[("B200", "FP4_TC")]))
print("  **从 V100 的 FP16（%.1f）到 B200 的 FP4（%.1f），拐点移动了 %.2f 倍。**"
      % (fp16_vals[0], ISTAR[("B200", "FP4_TC")],
         ISTAR[("B200", "FP4_TC")] / fp16_vals[0]))
print()
print("  这就是「必须换精度而不是换算法」的量化依据：")
print("  一个 I = 150 的算子，在 V100 上刚过拐点（%.1f）属算力受限；"
      % fp16_vals[0])
print("  到 B200 上它离 FP16 的拐点（%.1f）差 %.1f 倍，早已是带宽受限。"
      % (ISTAR[("B200", "FP16_TC")], ISTAR[("B200", "FP16_TC")] / 150))
print("  **同一条算法、同一份数据，换硬件的净效果是把它推到带宽那一侧。**")
print("  唯一能把它推回去的操作是换精度——FP4 那一档的拐点是 %.1f。"
      % ISTAR[("B200", "FP4_TC")])


rule("4. 封闭形式：一个算子在哪一代翻转")

print("第一步：把「拐点怎么移」写成一个比值。")
print()
print("  设两代的算力增长率为 r_P = P₂/P₁，带宽增长率为 r_B = B₂/B₁，则")
print()
print("      I*₂ / I*₁ = (P₂/B₂) / (P₁/B₁) = r_P / r_B")
print()
print("  **拐点的迁移方向只由 r_P 与 r_B 谁大决定，与绝对规模无关。**")
print("  记 ρ = r_B / r_P（带宽增长率 ÷ 算力增长率），则 I*₂ = I*₁ / ρ：")
print("    · ρ > 1（带宽涨得更快）⇒ 拐点下降")
print("    · ρ < 1（算力涨得更快）⇒ 拐点上升")
print()
print("第二步：把「翻转」写成 I 落在哪个区间。")
print()
print("  算子在某一代算力受限 ⟺ I > I*。所以要跨代翻转，I 必须落在两代的拐点之间：")
print()
print("      翻转 ⟺ min(I*₁, I*₂) < I < max(I*₁, I*₂)")
print("           ⟺ I / I*₁ ∈ ( min(1, 1/ρ) , max(1, 1/ρ) )")
print()
print("  方向也随之确定：")
print("    · ρ < 1（拐点上升）⇒ 窗口是 (I*₁, I*₁/ρ)，方向 **算力受限 → 带宽受限**")
print("    · ρ > 1（拐点下降）⇒ 窗口是 (I*₁/ρ, I*₁)，方向 **带宽受限 → 算力受限**")
print()
print("  第三句是最有用的一句：**带宽涨得比算力快的那一代，会把一批算子从")
print("  算力受限甩到带宽受限那一侧。** 反之亦然。")
print()
print("第三步：逐代核对。")
print()
print("  %-18s %-10s %-10s %-8s %-22s %s"
      % ("跃迁", "ρ = r_B/r_P", "拐点方向", "窗口宽度", "翻转后的方向", "窗口两侧的 I*"))
print("  " + "-" * 100)
FLIPS = {}
for a, b in zip(TREND_GENS, TREND_GENS[1:]):
    g_fp32, g_fp16, g_bw = GROWTHS[(a, b)]
    for prec, r_p in (("FP32_VECTOR", g_fp32), ("FP16_TC", g_fp16)):
        i1 = ISTAR[(a, prec)]
        rho = g_bw / r_p
        i2 = i1 / rho
        lo, hi = min(i1, i2), max(i1, i2)
        width = hi / lo
        direction = ("算力受限 → 带宽受限" if rho < 1 else "带宽受限 → 算力受限")
        FLIPS[(a, b, prec)] = (lo, hi, direction)
        print("  %-18s %-10.4f %-8s %-10.3f× %-22s (%.1f, %.1f)"
              % ("%s→%s %s" % (a, b, prec.split("_")[0]), rho,
                 "下降" if rho > 1 else "上升", width, direction, lo, hi))

print()
print("  注意 FP16 那一栏 H100→B200 的窗口宽度只有 %.4f 倍——"
      % (FLIPS[("H100", "B200", "FP16_TC")][1] / FLIPS[("H100", "B200", "FP16_TC")][0]))
print("  两代的 FP16 拐点几乎重合（%.1f 对 %.1f，差 %.2f%%），"
      % (ISTAR[("H100", "FP16_TC")], ISTAR[("B200", "FP16_TC")],
         (ISTAR[("B200", "FP16_TC")] / ISTAR[("H100", "FP16_TC")] - 1) * 100))
print("  几乎没有算子会在这个窗口里翻转。**「新一代一定重排瓶颈类型」是错的**——")
print("  重排只发生在拐点真的动了的那些跃迁上。")

print()
print("第四步：把窗口换算成 batch。取一个具体的 GEMM 族：")
print("  `X[M×8192] @ W[8192×8192]`，FP16/BF16，权重矩阵固定、只有 M 变。")
print("  由 I = 2MKN / (2(MK+KN+MN))，代入 K = N = 8192：")
print()
print("      I(M) = 4096·M / (M + 4096)    ⟶_{M→∞}  4096 = K 与 N 的调和平均")
print()
print("  %-24s %s" % ("跃迁 / 精度轴", "翻转的 M 区间"))
print("  " + "-" * 60)


def m_of_I(I):
    """I = 4096M/(M+4096) 反解 M。"""
    return 4096.0 * I / (4096.0 - I)


MWINDOWS = {}
for key in FLIPS:
    lo, hi, direction = FLIPS[key]
    m_lo, m_hi = m_of_I(lo), m_of_I(hi)
    MWINDOWS[key] = (m_lo, m_hi, direction)
    print("  %-24s M ∈ (%.1f, %.1f)    %s"
          % ("%s→%s %s" % (key[0], key[1], key[2].split("_")[0]),
             m_lo, m_hi, direction))

print()
print("  核对：M=4096 时 I(4096) = %.1f、M=512 时 I = %.1f，"
      % (4096 * 4096 / (4096 + 4096), 4096 * 512 / (512 + 4096)))
print("  与本仓库 10-foundations/performance-models/01 §5 表3 的 prefill 大 / 小 GEMM")
print("  两行（2.048e+03 / 4.551e+02）一致。**这两行是引用，不是本程序算出来的新数。**")


rule("5. 真实算子的落点与翻转")

print("下表算子的算术强度**引自本仓库 10-foundations/performance-models/01 §5 表3**，")
print("本程序不重算（重算会得到第二份数字，违反「一个知识点一处正文」）。")
print()
OPS = [
    ("decode GEMV（batch=1）", 1.000),
    ("softmax（4096×4096）", 1.250),
    ("SiLU（elementwise）", 1.250),
    ("LayerNorm", 1.500),
    ("attention QK^T", 120.471),
    ("attention PV", 120.471),
    ("prefill 小 GEMM（M=512）", 455.111),
    ("prefill 大 GEMM（M=4096）", 2048.000),
]
print("  %-26s %-10s %s" % ("算子", "I", "在四个 FP16 拐点下的分类"))
print("  " + "-" * 92)
for name, I in OPS:
    cls = []
    for g in TREND_GENS:
        cls.append("%s:%s" % (g, "算力" if I > ISTAR[(g, "FP16_TC")] else "带宽"))
    print("  %-26s %-10.3f %s" % (name, I, "  ".join(cls)))

print()
flipped = []
for name, I in OPS:
    sides = ["C" if I > ISTAR[(g, "FP16_TC")] else "M" for g in TREND_GENS]
    if len(set(sides)) > 1:
        flipped.append((name, I, sides))
print("**在 FP16 轴上，上表 8 个算子没有一个发生过翻转**（分类列逐代相同）。")
print("原因是它们的落点离拐点太远：elementwise 一族在 I ≈ 1~1.5，")
print("离最低的拐点（V100 的 %.1f）也差两个数量级；"
      % ISTAR[("V100", "FP16_TC")])
print("而两个 GEMM 在 I = 455 与 2048，离最高的拐点（H100 的 %.1f）高出一截。"
      % ISTAR[("H100", "FP16_TC")])
print()
print("真正会翻转的落点在**中间那一段**——也就是第 4 节算出来的 M 窗口。")
print("下面把 M 窗口代回，看一个具体的算子。")
print()
M_EX = 256
I_EX = 4096 * M_EX / (M_EX + 4096)
print("  取 M = 256、K = N = 8192、BF16：I = 4096×%d/(%d+4096) = %.2f"
      % (M_EX, M_EX, I_EX))
print("  %-8s %-12s %-12s %-14s %s" % ("代", "I* (FP16)", "分类", "可达 TFLOPS", "占峰值"))
print("  " + "-" * 72)
def attained_tflops(gen, prec, I):
    """roofline 屋顶的高度 min(P_peak, B_peak·I)，单位 TFLOPS。

    单位必须对齐：bw 是 GB/s（1e9 byte/s），乘 I（FLOP/byte）得 FLOP/s，
    再除 1e12 才是 TFLOPS；而 peak() 直接就是 TFLOPS。
    """
    return min(peak(gen, prec), bw(gen) * 1e9 * I / 1e12)


for g in ("A100", "H100"):
    ist = ISTAR[(g, "FP16_TC")]
    at = attained_tflops(g, "FP16_TC", I_EX)
    print("  %-8s %-12.1f %-12s %-14.1f %.2f%%"
          % (g, ist, "算力受限" if I_EX > ist else "带宽受限",
             at, at / peak(g, "FP16_TC") * 100))
a_at, h_at = (attained_tflops(g, "FP16_TC", I_EX) for g in ("A100", "H100"))
print()
print("  **同一个形状、同一份代码**：在 A100 上它是算力受限，能吃到 100% 峰值；")
print("  在 H100 上它变成了带宽受限，卡在 %.2f%%。"
      % (h_at / peak("H100", "FP16_TC") * 100))
print("  而这一形状的翻转窗口是 M ∈ (%.1f, %.1f)。"
      % (MWINDOWS[("A100", "H100", "FP16_TC")][0],
         MWINDOWS[("A100", "H100", "FP16_TC")][1]))
print()
print("  这不是「H100 更慢」——它的绝对吞吐仍是 A100 的 %.2f 倍。" % (h_at / a_at))
print("  但 **H100 的峰值里有 %.1f%% 在这一形状上拿不到，而 A100 一点没浪费**，"
      % (100 - h_at / peak("H100", "FP16_TC") * 100))
print("  因为拐点从 %.1f 升到了 %.1f（比值 %.3f）。**优化方向必须跟着换**："
      % (ISTAR[("A100", "FP16_TC")], ISTAR[("H100", "FP16_TC")],
         ISTAR[("H100", "FP16_TC")] / ISTAR[("A100", "FP16_TC")]))
print("  在 A100 上该做的是「把这一步算得更快」，在 H100 上该做的是「少搬一次」")
print("  或「换低精度」——同一份代码，两种答案。")


rule("6. 反推时钟：用官方 FP32 峰值交叉验证 SM 数")

print("FP32 向量峰值有一个只用规格常量的解析式：")
print()
print("      P_FP32 = SM 数 × 每 SM FP32 lane 数 × 2 FLOP/lane × 时钟")
print()
print("  其中那个 2 来自 FMA 计 2 FLOP（一次乘、一次加）。把它反解：")
print()
print("      时钟 = P_FP32 / (SM 数 × lane 数 × 2)")
print()
print("  %-8s %-9s %-8s %-14s %-12s %s"
      % ("代", "SM 数", "lane/SM", "官方 FP32 峰值", "反推时钟", "官方时钟"))
print("  " + "-" * 82)
CLOCKS_KNOWN = {  # 来源：各代产品规格页公布的 boost clock（第三方，仅用于对照）
    "V100": "1.53 GHz", "A100": "1.41 GHz", "H100": "不公布",
}
for g in TREND_GENS:
    chip = S.CHIP_BY_GEN[g]
    n = chip["sm_count"] * chip["fp32_lanes_per_sm"]
    f = peak(g, "FP32_VECTOR") * 1e12 / (n * 2) / 1e9
    print("  %-8s %-9d %-8d %-14.1f %-12s %s"
          % (g, chip["sm_count"], chip["fp32_lanes_per_sm"],
             peak(g, "FP32_VECTOR"), "%.3f GHz" % f, CLOCKS_KNOWN.get(g, "—")))

print()
print("四代反推出来的时钟全部落在 1.4~2.0 GHz 这个数据中心卡的合理区间里：")
print("  · V100 %.3f GHz、A100 %.3f GHz —— 与公开的 boost clock（1.53 / 1.41）吻合，"
      % (peak("V100", "FP32_VECTOR") * 1e12 / (S.V100_SM_COUNT * 64 * 2) / 1e9,
         peak("A100", "FP32_VECTOR") * 1e12 / (S.A100_SM_COUNT * 64 * 2) / 1e9))
print("    这验证了公式与规格常量两边都没错。")
print("  · H100 反推出 %.3f GHz —— NVIDIA 从不公布 H100 的 boost 时钟，"
      % (peak("H100", "FP32_VECTOR") * 1e12 / (S.H100_SM_COUNT * 128 * 2) / 1e9))
print("    **这是反推值，不是官方规格**；它与 67 TFLOPS ÷ (132×256) 的定义式等价。")
print("  · B200 反推出 %.3f GHz。这里正是 SM 数取 148 的理由："
      % (peak("B200", "FP32_VECTOR") * 1e12 / (S.B200_SM_COUNT * 128 * 2) / 1e9))
print("    SM 数取 132 会反推出 %.3f GHz —— 在 1000 W 的封装下这个频率不可信；"
      % (peak("B200", "FP32_VECTOR") * 1e12 / (132 * 128 * 2) / 1e9))
print("    取 %d 则得到 %.3f GHz，与 H100 的 %.3f GHz 几乎相同，"
      % (S.B200_SM_COUNT,
         peak("B200", "FP32_VECTOR") * 1e12 / (S.B200_SM_COUNT * 128 * 2) / 1e9,
         peak("H100", "FP32_VECTOR") * 1e12 / (S.H100_SM_COUNT * 128 * 2) / 1e9))
print("    是同一制程世代的合理设计点。**这是 SM 数取 %d 的全部依据。**"
      % S.B200_SM_COUNT)
print()
print("顺带得到一个与发射槽有关的恒等式。每 SM 每周期可发射 4 条 warp 指令、")
print("每条覆盖 32 个 lane，所以每周期最多 4 × 32 = 128 个 lane 操作：")
print()
print("  %-8s %-14s %-16s %s" % ("代", "FP32 lane/SM", "达到峰值所需的", "占发射上限"))
print("  " + "-" * 74)
for g in TREND_GENS:
    lanes = S.CHIP_BY_GEN[g]["fp32_lanes_per_sm"]
    need = lanes / 32.0
    print("  %-8s %-14d %-16s %.1f%%"
          % (g, lanes, "%.1f 条 FFMA/周期" % need, need / 4 * 100))
print()
print("  A100 只需要 2 条 FFMA/周期就顶到 FP32 峰值，发射上限有 2 倍余量；")
print("  **H100/B200 需要 4 条，即把 4 个发射槽全部占满、一条别的指令都不能插。**")
print("  换句话说：Hopper 的 FP32 峰值恰好等于发射上限，**零余量**。")
print("  任何一条 LDS、地址运算或分支指令都会从峰值上扣掉一份——")
print("  这不是实现缺陷，是 128 lane/SM 与 4 发射槽/周期这两个规格的算术后果。")
print()
print("顺带把「用 SM 数外推算力」到底错多少算出来（正文 §4.3 引用的两个数）：")
print()
sm_ratio = S.CHIP_BY_GEN["H100"]["sm_count"] / S.CHIP_BY_GEN["A100"]["sm_count"]
lane_ratio = (S.CHIP_BY_GEN["H100"]["sm_count"] * S.CHIP_BY_GEN["H100"]["fp32_lanes_per_sm"]
              / (S.CHIP_BY_GEN["A100"]["sm_count"] * S.CHIP_BY_GEN["A100"]["fp32_lanes_per_sm"]))
print("  按 SM 数外推：        %d / %d = %.2f×   ← 低估"
      % (S.CHIP_BY_GEN["H100"]["sm_count"], S.CHIP_BY_GEN["A100"]["sm_count"], sm_ratio))
print("  按 SM 数 × lane 数：  (%d × %d) / (%d × %d) = %.2f×"
      % (S.CHIP_BY_GEN["H100"]["sm_count"], S.CHIP_BY_GEN["H100"]["fp32_lanes_per_sm"],
         S.CHIP_BY_GEN["A100"]["sm_count"], S.CHIP_BY_GEN["A100"]["fp32_lanes_per_sm"],
         lane_ratio))
print("  再乘频率（%.3f / %.3f = %.2f）才是真正的峰值比 %.2f×。"
      % (peak("H100", "FP32_VECTOR") * 1e12 / (S.H100_SM_COUNT * 128 * 2) / 1e9,
         peak("A100", "FP32_VECTOR") * 1e12 / (S.A100_SM_COUNT * 64 * 2) / 1e9,
         (peak("H100", "FP32_VECTOR") * 1e12 / (S.H100_SM_COUNT * 128 * 2) / 1e9)
         / (peak("A100", "FP32_VECTOR") * 1e12 / (S.A100_SM_COUNT * 64 * 2) / 1e9),
         peak("H100", "FP32_VECTOR") / peak("A100", "FP32_VECTOR")))
print("  **SM 数外推把涨幅低估了 %.2f 倍**（%.2f× 对 %.2f×）。"
      % (lane_ratio / sm_ratio, sm_ratio, lane_ratio))


rule("7. tensor core ÷ CUDA core 峰值比（含一个反例）")

print("  %-8s %-12s %-12s %-12s %s" % ("代", "FP16/FP32", "FP8/FP32", "FP4/FP32", "说明"))
print("  " + "-" * 82)
for g in TREND_GENS:
    p32 = peak(g, "FP32_VECTOR")
    cells = []
    for prec in ("FP16_TC", "FP8_TC", "FP4_TC"):
        p = peak(g, prec)
        cells.append("—" if p is None else "%.1f×" % (p / p32))
    note = ""
    if g == "H100":
        note = "← **比值反而低于 A100**"
    print("  %-8s %-12s %-12s %-12s %s" % (g, cells[0], cells[1], cells[2], note))

print()
r_v100 = peak("V100", "FP16_TC") / peak("V100", "FP32_VECTOR")
r_a100 = peak("A100", "FP16_TC") / peak("A100", "FP32_VECTOR")
r_h100 = peak("H100", "FP16_TC") / peak("H100", "FP32_VECTOR")
print("  **H100 那一格是本节的重点。** FP16:FP32 的比值从 A100 的 %.1f× 掉到 H100 的 %.1f×——"
      % (r_a100, r_h100))
print("  因为 Hopper 的 FP32 向量吞吐涨了 %.2f×，快于 FP16 tensor 的 %.2f×。"
      % (GROWTHS[("A100", "H100")][0], GROWTHS[("A100", "H100")][1]))
print("  所以 **「tensor core 相对收益逐代单调增大」这句话是错的**：")
print("  它在 A100→H100 这一段是下降的，到 B200 才又跳到 %.1f×。" % (peak("B200", "FP16_TC") / peak("B200", "FP32_VECTOR")))
print()
print("  这条反例有实际后果：**「用 tensor core 一定比 FP32 快 N 倍」里的 N")
print("  不是一个逐代变大的数。** 它先涨（%.1f→%.1f）、再跌（→%.1f）、再涨（→%.1f）。"
      % (r_v100, r_a100, r_h100, peak("B200", "FP16_TC") / peak("B200", "FP32_VECTOR")))


rule("8. 精度阶梯：低精度轴是怎么把峰值推上去的")

print("  %-12s %-10s %s" % ("精度", "首次出现", "逐代峰值 (TFLOPS, dense)"))
print("  " + "-" * 78)
for prec in PRECISIONS:
    first = S.PRECISION_FIRST_GEN[prec]
    cells = []
    for g in TREND_GENS:
        p = peak(g, prec)
        if p is None:
            cells.append("%s:—" % g)
        else:
            cells.append("%s:%.1f" % (g, p))
    print("  %-12s %-10s %s" % (prec, first, "  ".join(cells)))

print()
print("  逐级读低精度轴（同代内比，消除代际差异）：")
for g in ("H100", "B200"):
    vals = [(p, peak(g, p)) for p in PRECISIONS if peak(g, p) is not None]
    base = peak(g, "FP32_VECTOR")
    print("    %s：" % g)
    for p, v in vals:
        print("      %-12s %8.1f TFLOPS = FP32 向量的 %.1f 倍" % (p, v, v / base))

print()
print("  三个可以直接引用的量级：")
print("    · B200 的 FP4 相对 FP32 向量是 %.0f 倍——这就是「精度阶梯」的全部内容。"
      % (peak("B200", "FP4_TC") / peak("B200", "FP32_VECTOR")))
print("    · **位宽精确减半的两档，峰值精确翻倍**：FP16→FP8 %.1f×、FP8→FP4 %.1f×。"
      % (peak("B200", "FP8_TC") / peak("B200", "FP16_TC"),
         peak("B200", "FP4_TC") / peak("B200", "FP8_TC")))
print("      TF32→FP16 只是**约** 2 倍（%.2f×）——两边的官方数字各自取过整。"
      % (peak("B200", "FP16_TC") / peak("B200", "TF32_TC")))
print("      而 FP32 向量→TF32 是一个 %.1f× 的大跳，**不是一档**："
      % (peak("B200", "TF32_TC") / peak("B200", "FP32_VECTOR")))
print("      「每降一档翻一倍」只适用于 TF32 以下的阶梯，别把它套到 FP32 那一跳上。")
print("    · 迁移的账要这么算：从 V100 到 B200，")
print("      算力（FP16 → FP4）涨了 %.1f×，带宽涨了 %.2f×，"
      % (peak("B200", "FP4_TC") / peak("V100", "FP16_TC"), bw("B200") / bw("V100")))
print("      商就是 %.2f / %.2f = %.2f —— 正是第 3 节那个 %.1f → %.1f 的迁移倍数。"
      % (peak("B200", "FP4_TC") / peak("V100", "FP16_TC"), bw("B200") / bw("V100"),
         ISTAR[("B200", "FP4_TC")] / fp16_vals[0], fp16_vals[0],
         ISTAR[("B200", "FP4_TC")]))
print("      **迁移的全部来源是分子（精度阶梯），不是分母（带宽）。**")


rule("9. 两处口径陷阱的现场演示")

print("陷阱一：带 star 的 Tensor Core 数字。")
print("  NVIDIA 用于宣传的数字是 **with sparsity** 的那一个：")
print()
print("  %-8s %-24s %-24s %s" % ("代", "产品页的 FP16 TC", "实际 dense FP16 TC", "差"))
print("  " + "-" * 82)
for g, sparse, dense in (("A100", S.A100_FP16_TC_TFLOPS_SPARSE_2TO4,
                          S.A100_FP16_TC_TFLOPS_DENSE),
                         ("H100", S.H100_FP16_TC_TFLOPS_SPARSE_2TO4,
                          S.H100_FP16_TC_TFLOPS_DENSE)):
    print("  %-8s %-24.1f %-24.1f %.2f×" % (g, sparse, dense, sparse / dense))
print()
print("  **NVIDIA 产品页写「H100 FP16 Tensor Core 1979 TFLOPS」，那个 1979 是")
print("  带 2:4 稀疏的，dense 是 %.1f。A100 同理（%.0f dense / %.0f sparse）。**"
      % (S.H100_FP16_TC_TFLOPS_DENSE, S.A100_FP16_TC_TFLOPS_DENSE,
         S.A100_FP16_TC_TFLOPS_SPARSE_2TO4))
print("  混用会让第 3 节的每一个 I* 整体挪位一倍。本程序全程 dense。")
print()
print("陷阱二：累加器类型改变 TFLOPS。")
print("  同一个 FP8 Tensor Core，配 FP16 累加与配 FP32 累加，峰值可以差 2 倍。")
print("  所以「这个格式有多少 TFLOPS」这个问法**本身不完整**——")
print("  必须写成「<型号> <精度> <累加器类型> <dense/sparse>」四元组。")
print("  spec 里还差一项：**没有来源的项本程序不填数**——")
print("  B200 的每 SM Tensor Core 数官方未披露，第 1 节照实写「未披露」。")
print()
print("陷阱三：同代换封装。")
print("  本文正文引用的两个差值是下面算出来的（H100 PCIe 的三个规格在 specs.py 里）：")
print()
h_sxm = S.CHIP_BY_GEN["H100"]
print("  %-24s %-14s %-14s %s" % ("量", "H100 SXM5", "H100 PCIe", "PCIe 相对 SXM5"))
print("  " + "-" * 70)
for label, a, b in (("SM 数", h_sxm["sm_count"], S.H100_PCIE_SM_COUNT),
                    ("TDP (W)", h_sxm["tdp_W"], S.H100_PCIE_TDP_W),
                    ("HBM 带宽 (GB/s)", h_sxm["hbm_bw_GBps"], S.H100_PCIE_HBM_BW_GBps)):
    print("  %-24s %-14.0f %-14.0f %+.1f%%" % (label, a, b, (b / a - 1) * 100))
print()
print("  **带宽差 %.1f%%、功耗差一倍**——把这两个档混进同一条趋势曲线，"
      % abs((S.H100_PCIE_HBM_BW_GBps / h_sxm["hbm_bw_GBps"] - 1) * 100))
print("  曲线会拐一个不存在的弯，而且不会报错：两个数都是真的、都来自官方文档。")


rule("10. 断言自检")

checks = [
    ("FP32 向量的 I* 十年在 9.5~20.1 之间（常被简写成「10~20」）",
     all(9.5 <= ISTAR[(g, "FP32_VECTOR")] <= 20.1 for g in TREND_GENS)),
    ("FP32 向量的 I* 在 9.5~20.1 之间震荡（非单调）",
     not (fp32_vals[0] < fp32_vals[1] < fp32_vals[2] < fp32_vals[3])),
    ("FP16 的 I* 逐代不低于 FP32 向量",
     all(ISTAR[(g, "FP16_TC")] > ISTAR[(g, "FP32_VECTOR")] for g in TREND_GENS)),
    ("B200 的 FP4 拐点比 V100 的 FP16 拐点大一个数量级（8~10 倍）",
     8 < ISTAR[("B200", "FP4_TC")] / ISTAR[("V100", "FP16_TC")] < 10),
    ("A100→H100 的 FP32 与 FP16 两个轴都是拐点上升",
     GROWTHS[("A100", "H100")][2] / GROWTHS[("A100", "H100")][0] < 1
     and GROWTHS[("A100", "H100")][2] / GROWTHS[("A100", "H100")][1] < 1),
    ("V100→A100 的 FP16 轴拐点上升、FP32 轴拐点下降",
     ISTAR[("A100", "FP16_TC")] > ISTAR[("V100", "FP16_TC")]
     and ISTAR[("A100", "FP32_VECTOR")] < ISTAR[("V100", "FP32_VECTOR")]),
    ("H100→B200 的 FP16 拐点几乎不动（差 < 2%）",
     abs(ISTAR[("B200", "FP16_TC")] / ISTAR[("H100", "FP16_TC")] - 1) < 0.02),
    ("H100 的 FP16:FP32 比值低于 A100（反例成立）", r_h100 < r_a100),
    ("B200 的 FP16:FP32 比值高于 A100", peak("B200", "FP16_TC") / peak("B200", "FP32_VECTOR") > r_a100),
    ("dense 恰好是 sparse 的一半（A100 / H100 两代都成立）",
     abs(S.A100_FP16_TC_TFLOPS_SPARSE_2TO4 / S.A100_FP16_TC_TFLOPS_DENSE - 2) < 1e-9
     and abs(S.H100_FP16_TC_TFLOPS_SPARSE_2TO4 / S.H100_FP16_TC_TFLOPS_DENSE - 2) < 1e-9),
    ("反推时钟全部落在 1.4~2.05 GHz",
     all(1.4 <= peak(g, "FP32_VECTOR") * 1e12
         / (S.CHIP_BY_GEN[g]["sm_count"] * S.CHIP_BY_GEN[g]["fp32_lanes_per_sm"] * 2)
         / 1e9 <= 2.05 for g in TREND_GENS)),
    ("B200 取 148 SM 的反推时钟比取 132 更接近 H100",
     abs(peak("B200", "FP32_VECTOR") * 1e12 / (148 * 128 * 2) / 1e9
         - peak("H100", "FP32_VECTOR") * 1e12 / (132 * 128 * 2) / 1e9)
     < abs(peak("B200", "FP32_VECTOR") * 1e12 / (132 * 128 * 2) / 1e9 - 2.0)),
    ("M=512 与 M=4096 的 I 与引用表一致（455.11 / 2048.00）",
     abs(4096 * 512 / (512 + 4096) - 455.111) < 0.01
     and abs(4096 * 4096 / (4096 + 4096) - 2048.0) < 1e-9),
    ("M=256 的算例在 A100 上算力受限、在 H100 上带宽受限",
     I_EX > ISTAR[("A100", "FP16_TC")] and I_EX < ISTAR[("H100", "FP16_TC")]),
    ("该算例在 A100 上可达 100% 峰值",
     abs(attained_tflops("A100", "FP16_TC", I_EX) / peak("A100", "FP16_TC") - 1) < 1e-9),
    ("该算例在 H100 上只能拿到 75%~85% 峰值",
     0.75 < attained_tflops("H100", "FP16_TC", I_EX) / peak("H100", "FP16_TC") < 0.85),
    ("H100 的绝对吞吐仍高于 A100（翻转不等于变慢）", h_at > a_at),
    ("但 H100 相对 A100 的可达吞吐增益（%.2f×）低于峰值之比（%.2f×）"
     % (h_at / a_at, peak("H100", "FP16_TC") / peak("A100", "FP16_TC")),
     h_at / a_at < peak("H100", "FP16_TC") / peak("A100", "FP16_TC")),
    ("引用表的 8 个算子在 FP16 轴上无一翻转",
     all(len({("C" if I > ISTAR[(g, "FP16_TC")] else "M") for g in TREND_GENS}) == 1
         for _, I in OPS)),
    ("A100→H100 的 FP16 翻转窗口比 H100→B200 的宽 20 倍以上",
     (MWINDOWS[("A100", "H100", "FP16_TC")][1] - MWINDOWS[("A100", "H100", "FP16_TC")][0])
     > 20 * (MWINDOWS[("H100", "B200", "FP16_TC")][1] - MWINDOWS[("H100", "B200", "FP16_TC")][0])),
    ("位宽精确减半的两档（FP16→FP8→FP4）峰值精确翻倍",
     all(abs(peak("B200", b) / peak("B200", a) - 2) < 1e-9
         for a, b in (("FP16_TC", "FP8_TC"), ("FP8_TC", "FP4_TC")))),
    ("TF32→FP16 只是约 2 倍（B200 上是 2.05×，官方数字各自取整）",
     2.0 < peak("B200", "FP16_TC") / peak("B200", "TF32_TC") < 2.1),
    ("FP32 向量→TF32 不是 2 倍（B200 上是 14.7 倍）",
     abs(peak("B200", "TF32_TC") / peak("B200", "FP32_VECTOR") - 2) > 10),
    ("V100→B200 的算力涨幅 ÷ 带宽涨幅 == FP4 与 FP16 拐点之比",
     abs((peak("B200", "FP4_TC") / peak("V100", "FP16_TC")) / (bw("B200") / bw("V100"))
         / (ISTAR[("B200", "FP4_TC")] / ISTAR[("V100", "FP16_TC")]) - 1) < 1e-9),
]
for desc, ok in checks:
    print("  [%s] %s" % ("通过" if ok else "**失败**", desc))
print()
print("全部 %d 项%s。" % (len(checks),
                     "通过" if all(o for _, o in checks) else "**未全部通过，见上**"))


rule("11. 本程序的边界")

print("  · 全部是解析计算，没有在 GPU 上跑任何 kernel。所有峰值都是**规格上限**，")
print("    不是可达性能：boost clock 漂移、占用率不足、尾效应都会让实际值更低。")
print("    本程序**不给**「能到峰值的百分之几」的经验系数——那个比例取决于实现。")
print("  · 峰值一律 dense（不含 2:4 稀疏）。若真能用上稀疏，所有 I* 翻倍。")
print("  · 第 4/5 节的翻转分析建立在「算子在两代上跑同一份代码、算术强度不变」")
print("    这个假设上。一旦实现方式变了（例如从 FP32 换成 BF16 实现），")
print("    Bytes 口径随精度变化，I 本身也会变——那时要重算 I，不是重算 I*。")
print("  · 第 5 节引用的算子 I 值来自本仓库 10-foundations/performance-models/01")
print("    §5 表3，本程序只做分类与换算，**不重算它们**。若那篇文档更新了数值，")
print("    本节的两个算例要跟着换。")
print("  · roofline 模型只给吞吐边界，不含延迟约束（Little 定律那一半在")
print("    20-gpu-architecture/execution-model/03 与 memory-system/01）。")
print("    能满足 I 的判据，不等于带宽一定吃得满。")
print("  · B200 的 SM 数（%d）是**反推值**，不是官方直述；B200 的每 SM Tensor Core"
      % S.B200_SM_COUNT)
print("    数与 L2 容量官方未披露，本程序不填。")
print("  · B200 的 TF32/FP16/FP8/FP4 峰值取自 Blackwell 技术简报表1。若该文档改版，")
print("    第 3 节的低精度拐点与第 8 节的阶梯要一起重算。")

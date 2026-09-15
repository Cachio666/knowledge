"""标称 vs 可达：gap 从哪来，以及该怎么比。

运行：python3 04_measuring_bandwidth.py
依赖：numpy（只用于表格；全部计算在本机 CPU 上完成，不需要 GPU）

本程序做的是**口径推导与模型估算**，不是硬件测量。数字分三类：
  · 规格值：来自 specs.py，每条带来源注释；
  · 外部引用值：来自 10-foundations/performance-models/02（每步 5 μs 的延迟量级）；
  · 推导值：上面两类代入公式的结果。
本程序**产不出实测值**——它一行硬件都没碰。

★ 边界声明：延迟-带宽模型（t = 延迟 + 数据/带宽）的定义在
  10-foundations/performance-models 已经给过，本文**引用不重推**；
  本文只用它解出「多小的消息算小」。

复核的五个结论：
  1. **只有 busbw 能与硬件峰值对比**——但 busbw 的上限是**单向**链路带宽
     （H100 = 450 GB/s），不是 900。拿 900 当分母会把上限砍半。
  2. busbw 在 NVLS / SHARP / COLLNET 下会**高估**：修正系数是按 ring 的
     搬运量定的，而归约在交换机里完成后，点到点搬运量变了。
  3. 小消息的分界点由 `α·BW` 解出：8 卡 NVLink all-reduce 是 **18 MB**。
  4. PCIe 的 gap 是三级的：线速率 → 载荷（−1.54%）→ 官方样例（−11.7%）。
  5. pinned vs pageable 差 2.5 倍，而**它不是 GPU 或 PCIe 的属性**。
"""

import specs


def rule(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


BW_NV_UNIDIR = specs.NVLINK_PER_GPU_BIDIR_GBps_BY_GEN[4] / 2      # 450 GB/s
BW_NV_BIDIR = specs.NVLINK_PER_GPU_BIDIR_GBps_BY_GEN[4]           # 900 GB/s


# ===========================================================================
rule("1. algbw 与 busbw：NCCL 的两个指标，只有一个是「链路口径」")

print("nccl-tests 对一次操作测出耗时 t 与消息大小 S，然后给两个数：")
print()
print("    algbw = S / t                    ← 算法带宽：只跟「用户看到的字节数」有关")
print("    busbw = algbw × 修正系数          ← 总线带宽：想把「链路真实搬运量」还原出来")
print()
print("修正系数不是拍脑袋定的，它是**该算法每个 rank 的搬运量 ÷ S**。")
print("以 ring all-reduce 为例推导（推导，不是引用）：")
print()
print("  已知（引用 performance-models/02，不重推）：每卡搬运量 V = 2(n-1)/n × S")
print("  若链路恰好跑在它的**单向**容量 BW 上，则 t = V / BW")
print()
print("    algbw = S / t = S × BW / V = BW × n / (2(n-1))")
print("    busbw = algbw × 2(n-1)/n = **BW**")
print()
print("**这是一个恒等式：完美实现的 ring all-reduce，其 busbw 恰好等于单向链路带宽。**")
print("所以 busbw 的物理上限是 %.0f GB/s（H100 的单向），**不是 %.0f GB/s（双向）**。"
      % (BW_NV_UNIDIR, BW_NV_BIDIR))
print()
n = 8
algbw_at_peak = BW_NV_UNIDIR * n / (2 * (n - 1))
print("  反解：busbw 打到 %.0f GB/s 时，algbw 只有 %.1f GB/s（n = %d）"
      % (BW_NV_UNIDIR, algbw_at_peak, n))
print("  修正系数 = %.2f / %.2f = %.4f ✓ 与 NCCL 文档给 AllReduce 的 2(n-1)/n 一致"
      % (BW_NV_UNIDIR, algbw_at_peak, BW_NV_UNIDIR / algbw_at_peak))
print()
print("  这就是为什么**用 algbw 与硬件峰值比是错的**：拿 %.1f 去比 %.0f，"
      % (algbw_at_peak, BW_NV_BIDIR))
print("  会得出「只跑到 29%」的假结论，而链路其实已经满了。")
print("  但**拿 busbw 去比 %.0f 同样是错的**——它比的是单向那一列。" % BW_NV_BIDIR)


# ===========================================================================
rule("2. NCCL 官方的修正系数表（规格值）")

print("来源：NCCL 官方文档。这里把公式本身当规格存进 specs.py，")
print("      n 代进去得到的系数是**推导值**（见下表最后一列）。")
print()
print("%-34s %-16s %-14s" % ("操作", "NCCL 的修正系数", "n = 8 时"))
print("  " + "-" * 66)
for op, formula in specs.NCCL_BUSBW_CORRECTION.items():
    if formula == "1":
        val = "1.0000"
    elif formula == "2(n-1)/n":
        val = "%.4f" % (2 * (n - 1) / n)
    else:
        val = "%.4f" % ((n - 1) / n)
    print("%-34s %-16s %-14s" % (op, formula, val))
print()
print("同一批数据配不同的系数，读数会差多少？以 n = 8、algbw = 257.1 GB/s 为例：")
print("  AllReduce 的 busbw  = 257.1 × 1.7500 = %.1f GB/s（= 单向上限 ✓）"
      % (algbw_at_peak * 1.75))
print("  若误用 AllGather 的系数：257.1 × 0.8750 = %.1f GB/s（腰斩）"
      % (algbw_at_peak * 0.875))
print()
print("⚠ **报数字必须连操作名与系数一起报**：「busbw 400 GB/s」这句话本身不完整，")
print("  因为没有系数就没有口径。")


# ===========================================================================
rule("3. busbw 在 NVLS / SHARP / COLLNET 下会高估：一个可以算出来的例子")

print("上面那个恒等式 busbw = BW 的前提是「每卡搬运 2(n-1)/n × S」——")
print("**这是 ring 算法的搬运量**。当归约发生在交换机里（NVLS / SHARP / COLLNET），")
print("点到点的搬运量不再服从这个式子，而修正系数**仍然是 ring 的那个**。")
print()
print("用一个显式的模型算一遍（**下面是假设的模型，不是 NCCL 的实现细节**）：")
print("  假设：每卡只把 S 字节发到交换机一次、再从交换机收回 S 字节（归约在交换机内完成）")
print("        ⟹ 每卡搬运量 ≈ S，而不是 %.2f × S" % (2 * (n - 1) / n))
print("  这个模型下 algbw = S / (S / BW) = BW")
print("  报出来的 busbw = algbw × %.2f = %.2f × BW —— **超出单向链路带宽 %.0f%%**"
      % (2 * (n - 1) / n, 2 * (n - 1) / n, (2 * (n - 1) / n - 1) * 100))
print()
print("  具体到 H100：algbw ≈ %.0f GB/s，报出来的 busbw ≈ %.1f GB/s，"
      % (BW_NV_UNIDIR, BW_NV_UNIDIR * 1.75))
print("  而任何一条链路的单向容量都只有 %.0f GB/s。" % BW_NV_UNIDIR)
print()
print("**结论：busbw 超过单向链路带宽不一定是错，但它意味着「这个数不再是链路口径」。**")
print("  NVIDIA 维护者也已经表示 busbw 现在只是「algbw 乘一个比例」。")
print("  所以：拿 busbw 与硬件峰值对比时，必须同时说明算法（ring 还是 NVLS/SHARP），")
print("  否则这个对比没有意义。")


# ===========================================================================
rule("4. 多小的消息算小：由 α·BW 解出来")

print("延迟-带宽模型（定义在 10-foundations/performance-models，本文引用不重推）：")
print()
print("    t = α + 数据量 / BW")
print()
print("两层含义，都是解同一个方程得到：")
print()
print("【(a) 单次点对点传输】两边相等时 α = size/BW，解出")
print("      size* = α × BW          ← 小于它的消息，延迟项比数据项还大")
print()
ALPHA_US = specs.EXTERNAL_RING_STEP_LATENCY_US       # 5 μs，外部引用值
cases = [
    ("NVLink 4 单向", BW_NV_UNIDIR),
    ("IB NDR 单向", specs.IB_PER_PORT_GBps["NDR"]),
    ("PCIe Gen5 x16 载荷", specs.PCIE_LANE_GTps_BY_GEN["5"] * 16 / 8 * (128 / 130)),
]
print("    取 α = %.0f μs（外部引用值：performance-models/02 的每步延迟）：" % ALPHA_US)
print()
print("    %-26s %-14s %-16s" % ("链路", "带宽(GB/s)", "交叉点 size*"))
print("    " + "-" * 58)
for label, bw in cases:
    s_star = ALPHA_US * 1e-6 * bw * 1e9
    print("    %-26s %-14.2f %-16s" % (label, bw, "%.2f MB" % (s_star / 1e6)))

print()
print("【(b) 一次 n 卡 ring all-reduce】延迟项是 2(n-1)·α，数据项是 2(n-1)/n·S/BW，")
print("      两边相等解出（2(n-1) 直接约掉）：")
print()
print("      S* = n × α × BW          ← 比单次点对点多出 n 倍，因为每个 chunk 只有 S/n")
print()
print("    %-26s %-8s %-16s" % ("链路", "n", "S*"))
print("    " + "-" * 54)
for label, bw, nn in (("NVLink 4 单向（单机）", BW_NV_UNIDIR, 8),
                      ("IB NDR 单向（跨机）", specs.IB_PER_PORT_GBps["NDR"], 64)):
    s_star = nn * ALPHA_US * 1e-6 * bw * 1e9
    print("    %-26s %-8d %-16s" % (label, nn, "%.1f MB" % (s_star / 1e6)))
print()
print("【效率曲线】把效率定义成「数据项占的比例」：")
print("      η = (size/BW) / (α + size/BW) = size / (size + α·BW)")
print()
print("  于是各效率点整齐地落在 α·BW 的整数倍上：")
print("    %-12s %-18s %-18s" % ("η", "size / (α·BW)", "NVLink 4 上是多少"))
print("    " + "-" * 50)
for eta in (0.5, 0.9, 0.99):
    mult = eta / (1 - eta)
    print("    %-12s %-18s %-18s" % ("%.0f%%" % (eta * 100), "%.0f" % mult,
                                     "%.2f MB" % (mult * ALPHA_US * 1e-6 * BW_NV_UNIDIR * 1e9 / 1e6)))
print()
print("⚠ 两条使用边界：")
print("  1. α 取的是 %.0f μs 这个**量级**（外部引用值），节点间实际更大（要过 NIC）。"
      % ALPHA_US)
print("     结论的形状不变，具体数字请把 α 换成你系统上的值重算。")
print("  2. η 到 50% 的那个点常被当作「小消息/大消息」的分界。它**不是**硬件属性，")
print("     它同时取决于 α 与 BW——换一张网卡就换一个分界点。")


# ===========================================================================
rule("5. gap 分解：PCIe 从线速率到可达，中间掉了多少")

print("以 PCIe Gen5 x16 单向为例，逐级往下掉（每一级都是规格值之间的换算）：")
print()
gt = specs.PCIE_LANE_GTps_BY_GEN["5"]
raw = gt * 16 / 8
pay = raw * (128 / 130)
sample = specs.PCIE_GEN5_X16_H2D_PINNED_SAMPLE_GBps
print("  %-40s %-12s %-14s" % ("口径", "GB/s", "相对上一级"))
print("  " + "-" * 68)
print("  %-40s %-12.2f %-14s" % ("① 线速率（GT/s × 16 ÷ 8）", raw, "—"))
print("  %-40s %-12.2f %-14s" % ("② 载荷（扣 128b/130b = 2/130）", pay,
                                    "-%.2f%%" % ((1 - pay / raw) * 100)))
print("  %-40s %-12.2f %-14s" % ("③ 官方 H2D 样例（pinned）", sample,
                                    "-%.2f%%" % ((1 - sample / pay) * 100)))
print("  %-40s %-12.2f %-14s" % ("① → ③ 合计", sample,
                                    "-%.2f%%" % ((1 - sample / raw) * 100)))
print()
print("  ③②的差（%.2f GB/s，占载荷的 %.1f%%）不是编码开销，是**协议与实现开销**："
      % (pay - sample, (1 - sample / pay) * 100))
print("  TLP 头、DLLP、流控、以及 GPU 侧拷贝引擎的实际效率。")
print("  ⚠ 这个 ③ 是**官方样例输出**，转引自本主题实施规格 §0.8.1；")
print("     本次未能定位一手页面（见 specs.UNSOURCED）。它只能当「官方口径的量级」看。")
print()
print("同一个分解在 PCIe Gen2 x16 上更极端（这一代的载荷是 %.2f GB/s）："
      % (specs.PCIE_LANE_GTps_BY_GEN["2"] * 16 / 8 * 0.8))
pay2 = specs.PCIE_LANE_GTps_BY_GEN["2"] * 16 / 8 * 0.8
pin2 = specs.PCIE_GEN2_X16_H2D_PINNED_SAMPLE_GBps
pag2 = specs.PCIE_GEN2_X16_H2D_PAGEABLE_SAMPLE_GBps
print("  %-40s %-12.2f %-14s" % ("② 载荷", pay2, "—"))
print("  %-40s %-12.2f %-14s" % ("③ pinned 内存", pin2, "%.0f%% of ②" % (pin2 / pay2 * 100)))
print("  %-40s %-12.2f %-14s" % ("④ pageable 内存", pag2, "%.0f%% of ②" % (pag2 / pay2 * 100)))
print()
print("  **pinned ÷ pageable = %.2f 倍**——这是整条链路上最大的一档。" % (pin2 / pag2))
print()
print("  ⚠⚠ 但必须说清楚：**pageable 的那一档不是 GPU 或 PCIe 的属性。**")
print("     pageable 拷贝要先经驱动 staging buffer，它的速度随 CPU、主板、")
print("     内存带宽剧烈变化，换一台机器就换一个数。报测量数字时必须写明用了哪种内存，")
print("     否则这个数不可比。")
print()
print("【NVLink 侧的可达率】规格 900 GB/s 双向，第三方给出的可达约 85% ——")
print("  **这是单一第三方来源，可信度低（specs.UNSOURCED 里记了这条），本文只提其存在，")
print("  不用它做任何推导。** 这也正好说明为什么「官方样例」比「第三方百分比」可靠。")


# ===========================================================================
rule("6. 该怎么比：一页对照表")

print("%-34s %-22s %-26s" % ("你要回答的问题", "可以用的分子", "可以比的分母"))
print("  " + "-" * 84)
rows = [
    ("这次 all-reduce 快不快", "busbw", "单向链路带宽（H100: 450 GB/s）"),
    ("这个 kernel/拷贝快不快", "自己测得的字节/秒", "同一口径的峰值（线速率或载荷，写明）"),
    ("这块 GPU 的互联能力如何", "每 GPU 双向带宽", "别的平台的每 GPU 双向带宽"),
    ("跨机扩展划不划算", "节点间每 GPU 带宽", "节点内每 GPU 带宽（**同方向**）"),
]
for a, b, c in rows:
    print("%-34s %-22s %-26s" % (a, b, c))
print()
print("三条最容易犯的错，按「错得多远」排序：")
print("  1. 用 algbw 比 900 GB/s          ⟹ 假效率 %.0f%%（真值 100%%）"
      % (algbw_at_peak / BW_NV_BIDIR * 100))
print("  2. 用 busbw 比 900 GB/s          ⟹ 假效率 50%（真值 100%）")
print("  3. 用双向 900 比单向 50 GB/s 的 IB ⟹ 假倍数 18 倍（真值 9 倍）")
print()
print("  真正的硬件上限：**H100 单卡单向 450 GB/s，双向 900 GB/s**；")
print("  一档不落地说清「哪个方向、哪条链路、哪个口径」，这三条都不会犯。")
print()
print("本程序**没有**产出任何实测值：它算的是修正系数、交叉点与口径换算。")

"""PCIe / NVLink 的带宽口径换算：同一个数字的三种含义。

运行：python3 01_bandwidth_conventions.py
依赖：numpy（只用于格式化与表格；全部计算在本机 CPU 上完成，不需要 GPU）

本程序做的是**单位换算与口径推导**，不是硬件测量。所有数字分两类：
  · 规格值：从 specs.py 抄来，每条带来源注释；
  · 推导值：规格值代入下面这些公式的结果。
本程序**产不出实测值**，正文里也不许把它写成实测。

本程序复核的四个结论：
  1. PCIe 的「带宽」至少有三种口径：线速率 / 编码后载荷 / 营销取整。
     Gen1/2 的编码开销是 20%，Gen3~5 是 1.54%，Gen6 是 5.47%。
  2. NVIDIA 自己的倍数表述用的是**线速率口径**（PCIe Gen5 = 128 而非 126.03），
     所以凡是拿它的倍数去反推有效带宽，都会系统性偏 1.5%（Gen6 是 5.5%）。
  3. NVLink 的每 GPU 数字全是**双向合计**；做「搬 X 字节要多久」必须用一半。
  4. NVIDIA 全线用十进制 GB/s（10^9），不是 GiB/s（2^30），两者差 7.4%。
"""

import specs


def rule(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# ===========================================================================
rule("1. PCIe：一个「带宽」有至少三种口径")

print("先固定三条公式。它们是本文全部 PCIe 数字的唯一来源，没有例外：")
print()
print("  线速率 x16 单向 GB/s = GT/s × 16 ÷ 8      ← GT/s 是每 lane 的原始传输率，")
print("                                             这一步还没有扣任何编码开销")
print("  载荷   x16 单向 GB/s = 线速率 × (编码分子 ÷ 编码分母)")
print("  双向                 = 单向 × 2            ← PCIe 每方向独立，全双工")
print()

GEN_ORDER = ["1", "2", "3", "4", "5", "6"]

print("%-6s %-11s %-13s %-11s %-12s %-11s %-11s %-10s" % (
    "代", "每lane GT/s", "编码", "编码效率", "线速率单向", "载荷单向", "载荷双向", "编码开销"))
print("  " + "-" * 88)

rows = []
for g in GEN_ORDER:
    gt = specs.PCIE_LANE_GTps_BY_GEN[g]
    name, num, den = specs.PCIE_ENCODING_BY_GEN[g]
    eff = num / den
    raw_unidir = gt * 16 / 8
    pay_unidir = raw_unidir * eff
    pay_bidir = pay_unidir * 2
    overhead = 1 - eff
    rows.append((g, gt, name, eff, raw_unidir, pay_unidir, pay_bidir, overhead))
    print("%-6s %-11.1f %-13s %-11.4f %-12.2f %-11.2f %-11.2f %-10s" % (
        "Gen" + g, gt, name, eff, raw_unidir, pay_unidir, pay_bidir,
        "%.2f%%" % (overhead * 100)))

print()
print("三种口径分别是：")
print("  · **线速率口径**：`GT/s × lane 数 ÷ 8`。从不含编码开销，硬件上不可达。")
print("  · **载荷口径**：扣掉编码开销之后真正能承载 TLP 的字节率。这是标准表格里的")
print("    精确值（Gen1 4.00、Gen3 15.75、Gen5 63.02）。NVIDIA 官方 Ampere 博客那句")
print("    *\"PCIe Gen 4 … providing 31.5 GB/sec vs. 15.75 GB/sec for x16 connections\"*")
print("    用的就是这一列（本程序复算：%.2f 与 %.2f）。" % (rows[3][5], rows[2][5]))
print("  · **营销取整口径**：市面上流传的整数。注意它**在两代上是不同东西**——")
print("    Gen1/Gen2 流传的 4 / 8 GB/s 是**载荷值**（因为 8b/10b 的 20% 太显眼，")
print("    没人取整得下去）；Gen3 起的 16 / 32 / 64 / 128 GB/s 是**线速率值**")
print("    （1.5% 的差在营销里被抹掉）。**同一个词在两代指不同的量，这就是本篇的题眼。**")

print()
print("⚠ 一个非常容易写错的推论：")
print("    「Gen1 标称 5 GB/s、有效 4 GB/s，差 20%」——**这句话对，但把它说成")
print("    『PCI-SIG 的 Gen1 营销值比有效值高 20%』就错了**：PCI-SIG 印出来的")
print("    Gen1 x16 就是 4 GB/s（每 lane 250 MB/s），它本来就是编码后的数。")
print("    20% 是「线速率 vs 载荷」的差，不是「营销 vs 载荷」的差。")
print("    真正的「营销 vs 载荷」差额是：Gen1/2 = 0%，Gen3~5 = 1.54%，Gen6 = 5.47%。")

print()
print("Gen6 多出来的一层：242/256 不是线编码，而是 FLIT 的字段开销。")
tot = sum(specs.PCIE6_FLIT_BYTES.values())
payload = specs.PCIE6_FLIT_BYTES["TLP"] + specs.PCIE6_FLIT_BYTES["DLP"]
print("    每个 256 B FLIT = %d B TLP + %d B DLP + %d B CRC + %d B FEC = %d B"
      % (specs.PCIE6_FLIT_BYTES["TLP"], specs.PCIE6_FLIT_BYTES["DLP"],
         specs.PCIE6_FLIT_BYTES["CRC"], specs.PCIE6_FLIT_BYTES["FEC"], tot))
print("    可用载荷 = %d + %d = %d B，占 %d/%d = %.4f → 开销 %.2f%%"
      % (specs.PCIE6_FLIT_BYTES["TLP"], specs.PCIE6_FLIT_BYTES["DLP"], payload,
         payload, tot, payload / tot, (1 - payload / tot) * 100))
print("    Gen6 是 PAM4，一个传输带 2 bit，所以「GT/s 与 Gb/s 相等」这条在 Gen6 失效。")


# ===========================================================================
rule("2. NVIDIA 自己用的是哪一种口径？——从它的倍数表述反推")

print("NVIDIA 在三处拿 NVLink 的每 GPU 带宽去除 PCIe，得到三个倍数：")
print("  Hopper 架构页：900 GB/s 是 \"over 7X the bandwidth of PCIe Gen5\"")
print("  HGX H100 博客：900 GB/s 是 \"more than 14x the bandwidth of PCIe Gen4 x16\"")
print("  NVLink 规格页：3600 GB/s 是 \"over 14x the bandwidth of PCIe Gen6\"")
print()
print("拿两种口径各算一遍，就能看出它用的是哪一个：")
print()
print("%-24s %-16s %-14s %-12s %-12s" % (
    "对照", "NVIDIA 的除数", "NVIDIA 倍数", "载荷口径倍数", "倍数偏差"))
print("  " + "-" * 82)

cases = [
    ("NVLink 4 vs PCIe Gen5", 900.0, "5"),
    ("NVLink 3 vs PCIe Gen4", 600.0, "4"),
    ("NVLink 6 vs PCIe Gen6", 3600.0, "6"),
]
for label, nvlink, g in cases:
    raw_bidir = specs.PCIE_LANE_GTps_BY_GEN[g] * 16 / 8 * 2
    _, num, den = specs.PCIE_ENCODING_BY_GEN[g]
    pay_bidir = raw_bidir * num / den
    ratio_nv = nvlink / raw_bidir
    ratio_pay = nvlink / pay_bidir
    # 偏差 = 线速率口径倍数相对载荷口径倍数低多少（因为除数偏大）
    print("%-24s %-16s %-14.2f %-12.2f %-12s" % (
        label, "%.0f（线速率）" % raw_bidir, ratio_nv, ratio_pay,
        "-%.2f%%" % ((1 - ratio_nv / ratio_pay) * 100)))

print()
print("结论：**NVIDIA 的倍数是拿「线速率口径」的 PCIe 值算的**——它的除数是")
print("128 / 64 / 256（线速率），而不是 126.03 / 63.02 / 242.00（载荷）。")
print("这不是造假，是取整习惯。但方向是**单边**的：除数偏大 ⟹ 倍数偏低 ⟹")
print("**用它的倍数去反推 PCIe 侧的字节数，你会系统性地少算 1.5%**（Gen6 是 5.5%）。")
print("多卡规划里这个偏差会被继承到每一层推导。")


# ===========================================================================
rule("3. NVLink：每 link / 每 GPU / 整机，三级放大，且全是双向合计")

print("NVIDIA 的 NVLink 规格页给两列：每 GPU 的双向带宽，和每 GPU 的最大链路数。")
print("其余四个数都是这两列相除或折半得到的——**这一步是推导，不是直述**：")
print()
print("  每 link 双向     = 每 GPU 双向 ÷ 链路数")
print("  每 GPU 单向     = 每 GPU 双向 ÷ 2")
print("  每 link 每方向   = 每 link 双向 ÷ 2")
print()

print("%-11s %-8s %-6s %-11s %-12s %-13s %-12s %-13s" % (
    "代", "GPU", "年", "链路数/GPU", "每GPU双向", "每GPU单向", "每link双向", "每link每方向"))
print("  " + "-" * 92)
for g in sorted(specs.NVLINK_PER_GPU_BIDIR_GBps_BY_GEN):
    bidi = specs.NVLINK_PER_GPU_BIDIR_GBps_BY_GEN[g]
    links = specs.NVLINK_LINKS_PER_GPU_BY_GEN[g]
    print("%-11s %-8s %-6d %-11d %-12.0f %-13.0f %-12.2f %-13.2f" % (
        "NVLink %d" % g, specs.NVLINK_GPU_BY_GEN[g], specs.NVLINK_YEAR_BY_GEN[g],
        links, bidi, bidi / 2, bidi / links, bidi / links / 2))

print()
print("★ 本主题最危险的一条：**上表「每GPU双向」整列都是双向合计**。")
print("  H100 的 900 GB/s = %d GB/s 发送 + %d GB/s 接收。" % (900 // 2, 900 // 2))
print("  **任何单向搬运的天花板是每 GPU 单向那一列（H100 = %.0f GB/s），不是 900。**"
      % (specs.NVLINK_PER_GPU_BIDIR_GBps_BY_GEN[4] / 2))
print()
print("  官方原文（两条独立来源，可直接引用）：")
print("    Hopper 架构页：\"Fourth-generation NVLink can scale multi-GPU input and")
print("      output (IO) with NVIDIA DGX and HGX servers at 900 gigabytes per second")
print("      (GB/s) bidirectional per GPU, over 7X the bandwidth of PCIe Gen5.\"")
print("    HGX H100 博客：\"this communication runs at the NVLink bidirectional speed")
print("      of 900 gigabytes per second (GB/s), which is more than 14x the bandwidth")
print("      of the current PCIe Gen4 x16 bus.\"")
print()
print("  另一条只靠算术就能查的旁证：每 link 双向那一列若换成「每 link 每方向」，")
print("  序列应当是 20/25/25/25/50/50——干净。若把 NVLink 6 写成 3000 GB/s，")
print("  每 link 会变成 %.1f，而这个序列里从来没有出现过非整数。" % (3000 / 36))
print("  （NVLink 6 的每 GPU 带宽取值见 §6 的说明。）")


# ===========================================================================
rule("4. 口径混淆的代价：搬 140 GB 究竟要多久")

S_GB = 140.0        # 70B 参数 BF16 的梯度大小（本仓库 performance-models/02 用的同一个量）
S_BYTES = S_GB * 1e9

print("场景：把一份 %.0f GB 的梯度（70B 参数、BF16）单向搬过 H100 的 NVLink。" % S_GB)
print("  公式：t = 字节数 ÷ 每 GPU 单向带宽")
print()
for label, bw in (("用每 GPU 单向 450 GB/s（正确）", 450.0),
                  ("用每 GPU 双向 900 GB/s（错，少算一倍）", 900.0)):
    t = S_BYTES / (bw * 1e9)
    print("  %-38s t = %.0f GB ÷ %.0f GB/s = %.4f s = %.1f ms" % (label, S_GB, bw, t, t * 1000))
print()
print("两个结果差 **2 倍**。这正是本篇标题里那个「弄错一个就得到 2 倍的错误结论」：")
print("900 是双向合计，它描述的是「同时 450 发 + 450 收」，不是「能单向搬 900」。")
print()
print("同理，把 NVLink 的三级放大搞混也会错：")
print("  每 link 双向 50 → 每 GPU 双向 900 → 8 卡聚合 7200 GB/s")
print("**7200 GB/s（7.2 TB/s）是 8 张卡的收发容量之和，不是任何一个 GPU 能用的带宽。**")
print("把它写成「每 GPU 带宽」，会让单卡推导偏 8 倍。")


# ===========================================================================
rule("5. 十进制与二进制：7.4% 的常数偏差")

print("NVIDIA 全线用十进制：900 GB/s 意思是 900 × 10^9 B/s。")
for label, bw in (("十进制 GB/s（10^9）", 1e9), ("二进制 GiB/s（2^30）", 2 ** 30)):
    print("  %-20s：900e9 B/s ÷ %-12s = %8.2f" % (label, "%.0f" % bw, 900e9 / bw))
print()
print("两个数的比 = 2^30 / 10^9 = %.4f，即 GiB/s 的数值比 GB/s 小 %.2f%%。"
      % (2 ** 30 / 1e9, (2 ** 30 / 1e9 - 1) * 100))
print("**差 7.4%**（不是 6.9%——那是把方向算反了：要问「同一个字节数在两种单位下")
print("的数值差多少」，就该用 2^30/10^9，而不是 10^9/2^30 的补数。）")
print()
print("按 2^30 反推 H100 会得到 %.2f GiB/s —— 它是对的，只是单位不同。" % (900e9 / 2 ** 30))
print("把 838 与 900 并列比较，就会凭空多出 7.4% 的差。")
print()
print("同一件事也发生在「搬 140 GB 要多久」上：140 GB = %.1f GiB。" % (140e9 / 2 ** 30))


# ===========================================================================
rule("6. 一个被广泛写错的说法：NVLink 5「每 link 100 GB/s 单向」")

print("网上大量文章写「NVLink 5 每 link 100 GB/s 单向」。用 NVIDIA 自己的算术验一下：")
print()
links5 = specs.NVLINK_LINKS_PER_GPU_BY_GEN[5]
claim_bidi = links5 * 100 * 2
real_bidi = specs.NVLINK_PER_GPU_BIDIR_GBps_BY_GEN[5]
print("  若每 link 每方向 100 GB/s，则每 GPU 双向 = %d 条 × 100 × 2 = %.0f GB/s"
      % (links5, claim_bidi))
print("  但 NVIDIA 规格页给每 GPU 双向 = %.0f GB/s" % real_bidi)
print("  差 %.1f 倍 —— **该说法与 NVIDIA 自己的算术矛盾**。" % (claim_bidi / real_bidi))
print()
print("正确读法：每 link 100 GB/s 是**双向**（50 GB/s 每方向），18 条合计 %.0f GB/s 双向。"
      % real_bidi)
print()
print("溯源：这条错误多半来自把「per-link 双向」与「per-direction」混写。")
print("H100 一侧的同一处最容易写成「每 link 50 GB/s 单向」——其实 50 是双向，25 才是单向。")
print("（且 25 这个数 NVIDIA 从未直接写过，它等于 900 ÷ 18 ÷ 2，是推导值。）")


# ===========================================================================
rule("7. 本程序引用的每一个规格值（全部来自 specs.py，带来源注释）")

print("来源清单见 specs.py。下面只列本程序实际用到的：")
print()
print("  PCIE_LANE_GTps_BY_GEN          PCI-SIG（规范/博客/Wikipedia 交叉一致）")
print("  PCIE_ENCODING_BY_GEN           同上；128b/130b 自 PCIe 3.0 起，8b/10b 只用于 1/2")
print("  PCIE6_FLIT_BYTES               PCI-SIG 博客 + Cadence 对 FLIT 的解析")
print("  NVLINK_PER_GPU_BIDIR_GBps_BY_GEN  NVIDIA NVLink 规格页（口径：每 GPU 双向合计）")
print("  NVLINK_LINKS_PER_GPU_BY_GEN       NVIDIA NVLink 规格页（口径：每 GPU 条数）")
print()
print("本程序**没有**产出任何实测值：它算的是规格值之间的换算。")

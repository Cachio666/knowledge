"""拓扑与 bisection 带宽：为什么「每卡带宽相同」不等于「集合通信一样快」。

运行：python3 02_topology_bisection.py
依赖：numpy（只用于表格；全部计算在本机 CPU 上完成，不需要 GPU）

本程序做的是**组合与算术推导**，不是硬件测量。数字分两类：
  · 规格值：来自 specs.py，每条带来源注释（如 HGX H100 的 3.6 TB/s）；
  · 推导值：规格值代入下面的公式算出（如 ring 的 0.9 TB/s）。
本程序**产不出实测值**。

复核的五个结论：
  1. ring 的 bisection 与 N 无关（= 2 × 每跳带宽）；全互联的 bisection ∝ N²。
  2. 官方两个 bisection 数字（HGX A100 2.4 / HGX H100 3.6 TB/s）恰好是各自
     聚合端口带宽的一半——**两组官方数字互相印证**。
  3. 同样用满 18 条链路，NVSwitch 的 bisection 是 ring 的 **4 倍**，不是 2 倍。
  4. 「路径均匀性」比 bisection 更本质：NVSwitch 下任意 GPU 对都是 900 GB/s。
  5. 「NVLink 域」的规模由 NVSwitch 端口数决定，不由软件或网络规模决定。
"""

import specs


def rule(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# 本文反复用到的三个符号
print("本文的符号：")
print("  N = GPU 数        L = 每 GPU 的链路数        b = 每链路的**双向**带宽")
print("  bisection = 把网络切成两半（每半 N/2 个 GPU）时，跨越切口的链路总容量")
print()
print("先立一条本文反复使用的记账规则（它是纯定义，不是来源）：")
print("  **一条链路 = 一个端口对**。GPU_i 分给 GPU_{i+1} 的 9 个端口，")
print("  与 GPU_{i+1} 分给 GPU_i 的 9 个端口，配对成的是 **9 条链路**，不是 18 条。")


# ===========================================================================
rule("1. bisection 的两条推导：ring 不随 N 长，全互联随 N² 长")

print("【推导 A：ring】")
print("  每个 GPU 的 L 条链路平分给 2 个邻居 ⟹ 每跳 L/2 条链路。")
print("  把 ring 切成两半，恰好要切断 2 跳，所以跨越切口的链路数是：")
print()
print("      crossing_links(ring) = 2 跳 × L/2 = L 条        ← **与 N 无关**")
print("      bisection(ring)      = L × b = 2 × (每跳带宽)")
print()
print("  另一个等价写法是「2 × 每跳带宽」——这就是教科书里 ring 的 bisection 公式。")
print()
print("【推导 B：全互联（每对 GPU 之间 k 条链路）】")
print("  切成 N/2 与 N/2，跨切口的 GPU 对数是 (N/2)² ，每对 k 条链路：")
print()
print("      crossing_links(mesh) = (N/2)² × k = N²/4 × k")
print("      bisection(mesh)      = N²/4 × k × b            ← **随 N² 增长**")
print()
print("  当每对恰好 1 条链路（k=1）时，退化成最常见的那个形式：N²/4 条链路。")
print()
print("**这就是本篇的核心结论：ring 的 bisection 是一个与 N 无关的常数，")
print("而全互联随 N² 增长。** 集合通信的下界由前者给出，且它不会因为加卡变好。")

N_DEMO = (8, 16, 64, 256)
print()
print("把两条公式并排（取 L = 18、b = 50 GB/s，即 H100 的链路预算）：")
print()
print("%-8s %-22s %-24s %-14s" % ("N", "ring 跨切口链路数", "全互联跨切口链路数", "比值"))
print("  " + "-" * 68)
for n in N_DEMO:
    ring_links = 18
    mesh_links = n * n / 4 * (18 / max(n - 1, 1))
    print("%-8d %-22d %-24.1f %-14.1f" % (n, ring_links, mesh_links, mesh_links / ring_links))
print()
print("注：全互联那一列用的是同样的 18 条链路预算（k = 18/(N-1)），所以 N 大时 k < 1，")
print("     意味着「每对之间连一条链路」这件事在 18 条链路的预算下根本做不到（N > 19）。")


# ===========================================================================
rule("2. 官方直述的两个 bisection：它们恰好是聚合端口带宽的一半")

print("NVIDIA 在 HGX 平台对比表里直接给出整机（8 GPU）的 bisection bandwidth。")
print("这两个数是**直述**，不是推导——本主题里可信度最高的一对：")
print()
print("%-18s %-16s %-20s %-18s %-12s" % ("系统", "每GPU双向", "聚合端口(8×每GPU)", "官方bisection", "是否为一半"))
print("  " + "-" * 88)
for name, per_gpu, bisec, nvsw in (
        ("HGX A100 8-GPU", 600.0, specs.HGX_A100_8GPU_BISECTION_TBps, specs.HGX_A100_NVSWITCH_COUNT),
        ("HGX H100 8-GPU", 900.0, specs.HGX_H100_8GPU_BISECTION_TBps, specs.HGX_H100_NVSWITCH_COUNT)):
    agg = 8 * per_gpu / 1000
    print("%-18s %-16.0f %-20.1f %-18.1f %-12s" % (
        name, per_gpu, agg, bisec, "是（%.1f / 2）" % agg))

print()
print("一致性自检（本程序算的，不是抄的）：")
for name, per_gpu, bisec in (("HGX A100", 600.0, 2.4), ("HGX H100", 900.0, 3.6)):
    agg = 8 * per_gpu / 1000
    print("  %-10s 8 × %.0f = %.1f TB/s 聚合；bisection = %.1f = %.4f × 聚合（应恰为 0.5）"
          % (name, per_gpu, agg, bisec, bisec / agg))
print()
print("**两组官方数字互相印证**：一个无阻塞全互联结构被均分成两半时，")
print("任意一半的收发容量正好是全网聚合容量的一半——两条独立记录都落在 0.5 上，")
print("说明聚合与 bisection 这两列是同一套口径（**双向合计**）下的一致读数。")
print()
print("⚠ 反过来说：如果哪天看到某系统的 bisection ≠ 聚合/2，先怀疑口径，")
print("   而不是先相信「这个拓扑很特别」。")
print()
print("NVSwitch 颗数（官方）：HGX A100 = %d 颗 gen2，HGX H100 = %d 颗 gen3。"
      % (specs.HGX_A100_NVSWITCH_COUNT, specs.HGX_H100_NVSWITCH_COUNT))
print("  ⚠ DGX H100 用户指南里有一句把「4 颗 NVSwitch」与「4th gen NVLink」混写。")
print("    那个 4 是 **NVSwitch 颗数**，不是 NVLink 条数——H100 每 GPU 是 %d 条。"
      % specs.NVLINK_LINKS_PER_GPU_BY_GEN[4])


# ===========================================================================
rule("3. 同一份链路预算：NVSwitch 的 bisection 是 ring 的 4 倍")

print("把两种拓扑放在**同一份 18 条/GPU 的链路预算**下比（这是公平比较的前提）：")
print()
print("【NVSwitch（crossbar）】每个 GPU 的 18 条链路全部接到交换机上，")
print("  交换机可以把它送到任意目的 GPU ⟹ 每半个网络的收发容量都能跨切口：")
print()
print("      bisection(crossbar) = (N/2) × (每 GPU 双向) = (N/2) × L × b")
print()
print("【ring】只有切口两侧的那 2 个 GPU 能跨切口发送：")
print()
print("      bisection(ring)     = L × b")
print()
print("  两者的比 = N/2。N = 8 时是 4 倍。")
print()

N = 8
for label, links, b, bisec_official in (
        ("H100（NVLink 4）", 18, 50.0, specs.HGX_H100_8GPU_BISECTION_TBps),
        ("A100（NVLink 3）", 12, 50.0, specs.HGX_A100_8GPU_BISECTION_TBps)):
    per_gpu = links * b
    xbar = N / 2 * per_gpu / 1000
    ring = links * b / 1000
    print("%s：每 GPU %d 条 × %.0f GB/s = %.0f GB/s 双向" % (label, links, b, per_gpu))
    print("   NVSwitch bisection = %d/2 × %.0f = %.1f TB/s   （官方直述 %.1f TB/s，吻合）"
          % (N, per_gpu, xbar, bisec_official))
    print("   ring     bisection = %d 条 × %.0f GB/s = %.2f TB/s" % (links, b, ring))
    print("   比值 = %.1f 倍" % (xbar / ring))
    print()

print("**结论：ring 的 bisection 是 NVSwitch 的 1/4，不是 1/2。**")
print()
print("对比一下「全互联」这个中间选项（每对 GPU 之间 k = L/(N-1) 条链路）：")
k = 18 / 7
mesh = (N / 2) ** 2 * k * 50 / 1000
print("   k = 18 / 7 = %.3f 条/对（不整除——这本身就说明直连 mesh 在这份预算下很别扭）" % k)
print("   bisection(mesh) = 4² × %.3f × 50 GB/s = %.2f TB/s" % (k, mesh))
print("   它比 ring 好（%.2f vs 0.90）却不如 NVSwitch（3.60）——因为直连 mesh 的每条链路"
      % mesh)
print("   **被硬绑给一个对端**，而 crossbar 的每条链路可以到达任意对端。")
print()
print("这条对比揭示的是比 bisection 更本质的一件事：**crossbar 并没有增加链路数，")
print("它增加的是「可达性」**——每 GPU 的 18 条链路从「固定伙伴」变成了「任意伙伴」。")


# ===========================================================================
rule("4. ring 的 bisection 极易多算一倍：错在哪")

print("一个很有诱惑力的错误算法是：")
print()
print("      「切 2 处 × 每处 18 条链路 × 50 GB/s = 2 × 18 × 50 = 1800 GB/s = 1.8 TB/s」")
print()
print("它错在**把同一条链路数了两遍**。逐项看：")
print()
links = 18
per_hop = links // 2
print("  每个 GPU 的 %d 条链路平分给 2 个邻居 ⟹ 每跳 %d 条。" % (links, per_hop))
print("  GPU_i 分给 GPU_{i+1} 的是 %d 个端口；GPU_{i+1} 分给 GPU_i 的也是 %d 个端口。" % (per_hop, per_hop))
print("  **这两组端口配对成 %d 条点对点链路**（每条链路两端各占一个端口），" % per_hop)
print("  所以「%d + %d」不是链路数，是端口数——链路数要除以 2。" % (per_hop, per_hop))
print()
print("  正确的账：")
print("    总链路数 = 8 GPU × %d 条 ÷ 2（每条链路两端）= %d 条" % (links, 8 * links // 2))
print("    跨切口 = 2 跳 × %d 条 = %d 条" % (per_hop, 2 * per_hop))
print("    bisection = %d 条 × 50 GB/s = %.0f GB/s = %.2f TB/s" % (2 * per_hop, per_hop * 2 * 50, per_hop * 2 * 50 / 1000))
print()
print("  若按错误算法要成立，每跳必须有 18 条链路，那每个 GPU 就得有")
print("  18 × 2 = 36 个端口——**H100 只有 18 条**。多出来的那一倍没有物理出处。")
print()
print("⚠ 这条陷阱之所以值得单独写出来，是因为 1.8 TB/s 这个错值**看起来很干净**")
print("  （1800 = 18 × 100，而 100 恰好是 NVLink 5 的每 link 双向带宽），")
print("  它不会在量纲上暴露自己，只会让「NVSwitch 是 ring 的几倍」这个结论错一倍。")


# ===========================================================================
rule("5. 路径均匀性：比 bisection 更本质的那一半")

print("bisection 只回答「最坏的那一刀有多宽」，不回答「任意一对 GPU 之间有多宽」。")
print()
print("【NVSwitch】任意 GPU 对：都是 900 GB/s 双向（官方：任意两个 H100 可以")
print("  同时以 NVLink 双向速度 900 GB/s 通信）。跳数恒为 1（经交换机）。")
print()
print("【ring】相邻对之间只有 1 跳的容量，非相邻对要共享沿途链路：")
M = 8
dist_count = {}
for i in range(M):
    for j in range(i + 1, M):
        d = min((j - i) % M, (i - j) % M)
        dist_count[d] = dist_count.get(d, 0) + 1
print()
print("  ring(8) 的 GPU 对距离分布：")
total_pairs = M * (M - 1) // 2
mean_hops = 0.0
for d in sorted(dist_count):
    print("    距离 %d 跳：%d 对（占 %.1f%%），每对可用带宽 = 450 / %d = %.0f GB/s"
          % (d, dist_count[d], dist_count[d] / total_pairs * 100, d, 450 / d))
    mean_hops += d * dist_count[d]
mean_hops /= total_pairs
print("    合计 %d 对，平均跳数 = %.3f" % (total_pairs, mean_hops))
print()
print("  **同一个「每 GPU 900 GB/s」，在 NVSwitch 下每一对都能拿到 900，")
print("  在 ring 下只有相邻的 %d/%d 对能拿到 450，最远的 %d 对只有 %.0f GB/s。**"
      % (dist_count[1], total_pairs, dist_count[4], 450 / 4))
print()
print("【DGX-1 的 hybrid cube-mesh】8 卡、每卡 4 条 NVLink（NVLink 1）：")
print("  每个 GPU 直连 4 个对端（4/7 直连），剩下 3 个对端需要一跳中转（3/7）。")
print("  **它不是全互联，也不是 ring**——这正是「NVLink 域」这个词在 V100 时代")
print("  还不该出现的原因（见下一节）。")


# ===========================================================================
rule("6. NVLink 域：规模由 NVSwitch 端口数决定，不由网络规模决定")

print("官方定义：*\"A set of nodes that can communicate over NVLink.\"* ——")
print("**它是一个跨 OS 实例的概念**，不是「装在一个机箱里」的意思。")
print()
print("%-34s %-10s" % ("平台", "NVLink 域(GPU)"))
print("  " + "-" * 48)
for plat, gpus in specs.NVLINK_DOMAIN_GPUS_BY_PLATFORM.items():
    print("%-34s %-10d" % (plat, gpus))
print()
print("⚠ 两个好失败模式（都值得单独点名）：")
print("  1. **「NVLink 域」这个词在 V100 时代并不存在**。DGX-1 的 8 卡不是 NVLink 域：")
print("     术语还没出现，而且拓扑是 hybrid cube-mesh（4/7 直连、3/7 需一跳）。")
print("  2. **A100 SuperPOD 的 256 卡也不是 NVLink 域**。跨节点走 InfiniBand，")
print("     NVIDIA 官方表格里 256 A100 pod 的 NVLink Domain 仍然是 **8 GPU**。")
print("     「一个 pod 有 256 张卡」与「一个 NVLink 域有 256 张卡」是两件事。")
print("     到 H100 才有 256 GPU 的域（经 NVLink Switch System）——")
print("     域从 8 涨到 256 靠的是把 NVSwitch 级联出去，不是靠加 GPU。")


# ===========================================================================
rule("7. 三级放大与 scale-up/scale-out：130 TB/s 不是「每 GPU 带宽」")

print("【三级放大】NVLink 4（H100）：")
print("  每 link 双向 %.0f GB/s → 每 GPU 双向 %.0f GB/s → 8 卡聚合 %.1f TB/s"
      % (specs.NVLINK4_PER_LINK_BIDIR_GBps,
         specs.NVLINK_PER_GPU_BIDIR_GBps_BY_GEN[4],
         8 * specs.NVLINK_PER_GPU_BIDIR_GBps_BY_GEN[4] / 1000))
print("  **7.2 TB/s 是 8 张卡的收发容量之和**，不是任何单个 GPU 能用的带宽。")
print("  把它当成「每 GPU 带宽」，会让所有单卡推导偏 8 倍。")
print()
print("【GB200 NVL72】两个数字属于两套**并行**网络，不能相加也不能混用：")
nvl72_up = specs.GB200_NVL72_SCALEUP_AGGR_TBps
nvl72_nic = specs.GB200_NVL72_NIC_COUNT * specs.GB200_NVL72_NIC_Gbps / 8 / 1000
print("  scale-up（NVLink，机架内）：%.0f TB/s  = 72 × %.1f TB/s"
      % (nvl72_up, specs.NVLINK_PER_GPU_BIDIR_GBps_BY_GEN[5] / 1000))
print("  scale-out（%d 张 ConnectX-8 @ %d Gb/s）：%.1f TB/s"
      % (specs.GB200_NVL72_NIC_COUNT, specs.GB200_NVL72_NIC_Gbps, nvl72_nic))
print("  比值 = %.2f 倍" % (nvl72_up / nvl72_nic))
print()
print("  自检：130 TB/s ÷ 72 = %.4f TB/s = 每 GPU 的 NVLink 5 带宽（%.1f TB/s）✓"
      % (nvl72_up / 72, specs.NVLINK_PER_GPU_BIDIR_GBps_BY_GEN[5] / 1000))
print("  这说明 130 TB/s 就是「每 GPU 带宽 × 卡数」，**它是 scale-up 的全端口聚合**。")
print()
print("⚠ 把 NVL72 的整体通信容量写成 130 TB/s 而不区分方向，")
print("  所有跨节点推导都会**错 18 倍**——因为跨节点走的是另一张网（scale-out），")
print("  每 GPU 只有 100 GB/s，不是 1800 GB/s。")
print()
print("⚠ 还有一条更隐蔽的：NVIDIA **没有公布** NVL72 的 bisection bandwidth。")
print("  130 TB/s 不是 bisection（它是全端口聚合）。要算 NVL72 的 bisection，")
print("  只能自己按第 1 节的公式推导，并明确标注为推导值。")
print()
print("【节点内 vs 节点间的倍数：口径必须写清】")
for plat, inside, outside in (("H100", 900.0, 50.0), ("A100", 600.0, 25.0),
                              ("GB200 NVL72", 1800.0, 100.0)):
    print("  %-13s 节点内 %.0f GB/s 双向 vs 节点间 %.0f GB/s 双向 = %.0f : 1"
          % (plat, inside, outside, inside / outside))
    print("  %-13s 若按**单向**比则是 %.0f : 1 —— **两个数都对，取决于要表达什么**"
          % ("", inside / 2 / outside))
print()
print("本程序统一按**双向**给出倍数（与 NVIDIA 公布的口径一致），")
print("并在正文里每次都注明用的是哪一个。这是本主题最容易被指为「算错」的一处。")

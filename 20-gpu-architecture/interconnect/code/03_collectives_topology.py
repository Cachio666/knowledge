"""拓扑受限的集合通信：bisection 下界、all-to-all、分层。

运行：python3 03_collectives_topology.py
依赖：numpy（只用于表格；全部计算在本机 CPU 上完成，不需要 GPU）

本程序做的是**下界推导与串行相加**，不是硬件测量。数字分三类：
  · 规格值：来自 specs.py（如 3.6 TB/s、900 GB/s），每条带来源注释；
  · 外部引用值：来自本仓库 10-foundations/performance-models/02（如每步 5 μs）；
  · 推导值：上面两类代入公式的结果。
本程序**产不出实测值**。

★ 边界声明（CONTRIBUTING §7）：**ring all-reduce 的每卡通信量 2(N-1)/N·S
  与它的带宽饱和，已在 10-foundations/performance-models/02-amdahl-and-scaling.md
  §1 第四步推导过。本程序不重推那个式子，只把它的结果当成输入。**
  本篇要推的是**拓扑受限**的那部分：
  1. bisection 下界：一次 all-reduce 至少要把多少字节搬过切口；
  2. all-to-all 的跨切口量随 N 线性增长（all-reduce 的不增长）；
  3. 分层（节点内 NVLink + 节点间 IB）的两段串行相加，以及配比决定哪段是瓶颈。
"""

import specs


def rule(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# --- 场景输入：全部显式写出，且明确标注它们是「场景假设」而不是硬件规格 ---
S_GB = 140.0          # 场景假设：70B 参数、BF16 的梯度大小（与 performance-models/02 同）
S = S_GB * 1e9        # 字节
N_H100 = 8            # 单机 8 卡
BW_H100_UNIDIR = specs.NVLINK_PER_GPU_BIDIR_GBps_BY_GEN[4] / 2      # 450 GB/s，单向
B_H100 = specs.HGX_H100_8GPU_BISECTION_TBps * 1e12                  # 3.6 TB/s，双向合计
BW_IB_NDR_UNIDIR = specs.IB_PER_PORT_GBps["NDR"]                    # 50 GB/s，单向
G_PER_NODE = 8        # 场景假设：每节点 8 GPU（DGX/HGX 的常规形状）

print("场景输入（**这些是假设，不是规格**——规格部分见 specs.py）：")
print("  S = %.0f GB           梯度大小（70B 参数 BF16）" % S_GB)
print("  单机 N = %d 卡        H100 SXM，NVLink 4" % N_H100)
print("  每 GPU 单向带宽 = %.0f GB/s（= 双向 900 ÷ 2）" % BW_H100_UNIDIR)
print("  HGX H100 bisection = %.1f TB/s（规格值，官方直述）"
      % specs.HGX_H100_8GPU_BISECTION_TBps)
print("  每 GPU 网卡 = %s 张 @ %s，即节点间每 GPU %.0f GB/s 单向"
      % (specs.DGX_H100_NIC_PER_GPU, "400 Gb/s IB (NDR)", BW_IB_NDR_UNIDIR))


# ===========================================================================
rule("1. bisection 下界：一次 all-reduce 至少要搬 2S 过切口")

print("★ 本节不重推 ring all-reduce 的每卡通信量。那个式子（2(N-1)/N·S）与它的")
print("  带宽饱和在 10-foundations/performance-models/02-amdahl-and-scaling.md §1")
print("  第四步已经推导过（正文在该域，此处回链）。本节要推的是**另一个**量：")
print("  在 bisection 为 B 的网络上，一次 all-reduce 至少要搬多少字节过切口。")
print()
print("推导：把 N 个 rank 按任意方式切成两半 A 与 B，切口带宽为 B。")
print("  1) 最终每个 rank 都要拿到全部 rank 的和。")
print("  2) A 里的 rank 需要「B 的贡献」；B 里的 rank 需要「A 的贡献」。")
print("  3) 每一半都可以先在**本地**归约（本地归约不跨切口），于是跨切口的最小")
print("     载荷是：A 的 S 字节部分和 与 B 的 S 字节部分和，各跨一次。")
print()
print("      crossing(all-reduce) = 2S     ← 一个与 N 无关的常数")
print()
print("  注意第 3 步是关键：**先在本地归约**把「B 里每个 rank 各 S 字节」压缩成")
print("  「B 整体的 S 字节部分和」。不做这一步，跨界量会是 N·S/2 而不是 S。")
print()
for label, b in (("HGX H100 8-GPU（NVSwitch）", B_H100),
                 ("物理 ring（18 条链路/GPU）", 0.9e12)):
    t = 2 * S / b
    print("  %-28s t ≥ 2S / B = %.0f GB ÷ %.2f TB/s = %.4f s = %.1f ms"
          % (label, 2 * S_GB, b / 1e12, t, t * 1000))
print()
print("这些是**下界**，不是可达值。它们回答的是「无论如何都不可能比这更快」。")
print("（对照：10-foundations 那边给出的 NVLink 饱和通信时间是 0.622 s——")
print(" 那是「ring 算法 + 每 GPU 单向 450 GB/s」的结果，比下界大一个量级。）")


# ===========================================================================
rule("2. 把下界和「每 GPU 带宽上界」摆在一起：谁才是瓶颈")

print("同一次 all-reduce，还有**另一个**上界：每个 rank 自己的链路带宽。")
print("ring 算法的每卡搬运量是 2(N-1)/N·S（引用 performance-models/02，不重推）：")
print()
V = 2 * (N_H100 - 1) / N_H100 * S
print("  V = 2 × (%d-1)/%d × %.0f GB = %.1f GB   （占 2S 的 %.1f%%）"
      % (N_H100, N_H100, S_GB, V / 1e9, (2 - 2 / N_H100) / 2 * 100))
t_pergpu = V / (BW_H100_UNIDIR * 1e9)
t_bisec = 2 * S / B_H100
print()
print("  %-34s %-12s" % ("上界", "时间"))
print("  " + "-" * 48)
print("  %-34s %.4f s (%.1f ms)" % ("每 GPU 带宽上界（450 GB/s 单向）", t_pergpu, t_pergpu * 1000))
print("  %-34s %.4f s (%.1f ms)" % ("bisection 下界（3.6 TB/s）", t_bisec, t_bisec * 1000))
print("  %-34s %.1f 倍" % ("两者之比", t_pergpu / t_bisec))
print()
print("**结论：在 8 卡 NVSwitch 系统上，bisection 不是瓶颈，每 GPU 链路带宽才是。**")
print("  这条很容易被说反——因为 NVSwitch 的 bisection 恰好等于 8 张卡的单向容量之和")
print("  （3600 GB/s = 8 × 450），所以真正卡住 ring all-reduce 的是每卡那一份 245 GB。")
print()
print("换个角度看得更清楚——**算法的全网搬运量**：")
print("  ring all-reduce 全网实际搬运 = %d 卡 × %.1f GB = %.0f GB = %.1f × S"
      % (N_H100, V / 1e9, N_H100 * V / 1e9, N_H100 * V / 1e9 / S_GB))
print("  而任何算法跨切口的**下界**只需要 2S = %.0f GB" % (2 * S_GB))
print()
print("  这两个量不是同一类东西（一个是「全网总搬运」，一个是「必须跨过切口的最小量」），")
print("  但它们的比 %.1f 倍说明了一件事：**链路并没有跑不满，"
      % (N_H100 * V / (2 * S)))
print("  是算法把同一份数据搬了很多遍。** 这就是「算法效率」这四个字在本例中的含义，")
print("  也是 04 篇讲 busbw 时要盯住的量。")
print()
print("顺带一个常被忽略的对照——同样是「ring all-reduce」，如果跑在**物理**的")
print("cable ring 上（每跳 9 条链路 × 25 GB/s 单向 = 225 GB/s），每卡那 245 GB 要花：")
t_physical_ring = V / (225e9)
print("  %.1f GB ÷ 225 GB/s = %.4f s = %.1f ms" % (V / 1e9, t_physical_ring, t_physical_ring * 1000))
print("  是走 NVSwitch 的逻辑环（%.1f ms）的 %.2f 倍。" % (t_pergpu * 1000, t_physical_ring / t_pergpu))
print("  **同一个算法，同一批链路，拓扑不同就差 2 倍**——这就是本篇的主线。")


# ===========================================================================
rule("3. all-to-all：跨切口量随 N 线性增长，而不是常数")

print("all-to-all 里每个 rank 把数据**分给所有其他 rank**，没有归约可做：")
print("  记 V = 每个 rank 发出的总字节数（收发均衡时也是它收到的总字节数），")
print("  则它发给每个对端 V/(N-1) 字节。")
print()
print("推导跨界量：切成 N/2 与 N/2，每个 rank 有 N/2 个对端在另一半：")
print("  每 rank 跨界 = (N/2) × V/(N-1) ≈ V/2")
print("  全网跨界     = (N/2) × (V/2) = **N·V/4**")
print()
print("对照 all-reduce 的跨界量 2S（常数）：**all-to-all 的跨界量 ∝ N。**")
print()
print("%-8s %-22s %-22s %-16s" % ("N", "all-reduce 跨界量", "all-to-all 跨界量", "比值"))
print("  " + "-" * 72)
for n in (8, 16, 64, 256, 576):
    ar = 2.0
    a2a = n * 1.0 / 4          # V = S 归一化
    print("%-8d %-22s %-22.2f %-16.1f" % (n, "2S = 2.00（常数）", a2a, a2a / ar))
print()
print("（V 用 S 归一化。这个对比与 V 的具体值无关，所以它是一条结构性的结论。）")
print()
print("换一个更锋利的表述——**跨界量占自己搬运总量的比例**：")
print("  all-reduce：全网搬运 2NS，跨界 2S  ⟹ 跨界占 2/N（N=8 时 25%，N=64 时 3.1%）")
print("  all-to-all：全网搬运 NV，跨界 NV/4 ⟹ 跨界恒占 **1/4**")
print()
for n in (8, 16, 64, 256, 576):
    load = (1 / 4) / (2 / n)
    print("  N = %-4d all-to-all 的 bisection 压力是 all-reduce 的 %.1f 倍" % (n, load))
print()
print("**all-to-all 是 bisection-bound 的，ring all-reduce 不是。** 这是两者最本质的")
print("量级差别：不在于「每卡搬多少字节」（那个量级相当），而在于")
print("**「每卡搬的字节里有多大比例必须跨过那条最窄的切口」**，以及这个比例")
print("会不会随 N 增长。all-reduce 的会随 N 下降，all-to-all 的不会。")
print()
print("★ 一个具体的 MoE 场景（**下面几个是场景假设，不是规格**）：")
tokens_per_rank, hidden, topk, dtype_bytes = 4096, 8192, 2, 2
per_rank_layer = tokens_per_rank * topk * hidden * dtype_bytes
print("  每 rank %d token、hidden %d、top-%d、BF16（%d B）"
      % (tokens_per_rank, hidden, topk, dtype_bytes))
print("  单层单次 all-to-all（dispatch）每 rank ≈ %d × %d × %d × %d = %.1f MB"
      % (tokens_per_rank, topk, hidden, dtype_bytes, per_rank_layer / 1e6))
print("  一层有 dispatch + combine 两次 ⟹ 每 rank 每层 %.1f MB" % (per_rank_layer * 2 / 1e6))
print("  按 80 层、前后向各一次算：每 rank 每步 %.1f GB" % (per_rank_layer * 2 * 80 * 2 / 1e9))
print("  对照：同一步里梯度的 ring all-reduce 每 rank 搬 %.1f GB" % (V / 1e9))
print("  ⟹ 在这个场景下 all-reduce 的**搬运量**反而更大（%.1f 倍），"
      % (V / (per_rank_layer * 2 * 80 * 2)))
print("     但 all-to-all 的**跨切口比例**是它的 %.1f 倍（N=8 时）。"
      % ((1 / 4) / (2 / N_H100)))
print()
print("  （MoE 的 all-to-all 细节正文在 40-kernels/moe-kernels，此处回链。）")


# ===========================================================================
rule("4. 分层：节点内 + 节点间两段串行相加，配比决定哪段是瓶颈")

print("分层 all-reduce（G = 每节点 GPU 数，M = 节点数，N = G × M）分三段：")
print("  (a) 节点内 reduce-scatter：每卡搬 (G-1)/G × S")
print("  (b) 跨节点 all-reduce：每卡只负责自己那 1/G 的分片 ⟹ 搬 2(M-1)/M × S/G")
print("  (c) 节点内 all-gather：每卡搬 (G-1)/G × S")
print("  三段串行相加（(a) 与 (c) 之间隔着 (b)，不能重叠）。")
print()
M_NODES = 8
N_TOTAL = G_PER_NODE * M_NODES
intra_bytes = 2 * (G_PER_NODE - 1) / G_PER_NODE * S
inter_bytes = 2 * (M_NODES - 1) / M_NODES * S / G_PER_NODE
t_intra = intra_bytes / (BW_H100_UNIDIR * 1e9)
t_inter = inter_bytes / (BW_IB_NDR_UNIDIR * 1e9)
print("场景：N = %d 卡 = %d 节点 × %d 卡/节点" % (N_TOTAL, M_NODES, G_PER_NODE))
print()
print("  (a)+(c) 节点内：2 × %d/%d × %.0f GB = %.1f GB   @ %.0f GB/s ⟹ %.4f s"
      % (G_PER_NODE - 1, G_PER_NODE, S_GB, intra_bytes / 1e9, BW_H100_UNIDIR, t_intra))
print("  (b)     节点间：2 × %d/%d × %.0f/%d GB = %.2f GB @ %.0f GB/s ⟹ %.4f s"
      % (M_NODES - 1, M_NODES, S_GB, G_PER_NODE, inter_bytes / 1e9,
         BW_IB_NDR_UNIDIR, t_inter))
print("  合计 = %.4f s" % (t_intra + t_inter))
print()
print("对照：**不用分层**，把 %d 张卡当成一个扁平 ring，全部走 IB：" % N_TOTAL)
V_flat = 2 * (N_TOTAL - 1) / N_TOTAL * S
t_flat = V_flat / (BW_IB_NDR_UNIDIR * 1e9)
print("  每卡搬 2 × %d/%d × %.0f GB = %.1f GB @ %.0f GB/s ⟹ %.4f s"
      % (N_TOTAL - 1, N_TOTAL, S_GB, V_flat / 1e9, BW_IB_NDR_UNIDIR, t_flat))
print()
print("  **分层快 %.2f 倍**。" % (t_flat / (t_intra + t_inter)))
print("  原因不是「NVLink 更快」这么笼统，而是：**同样多的字节被搬到了更宽的链路上**。")
print("  两种做法搬的总字节数几乎一样（分层 %.1f GB vs 扁平 %.1f GB），"
      % (intra_bytes / 1e9 + inter_bytes / 1e9, V_flat / 1e9))
print("  但分层把其中 %.1f GB（%.0f%%）留在了节点内的 450 GB/s 链路上，"
      % (intra_bytes / 1e9, intra_bytes / (intra_bytes + inter_bytes) * 100))
print("  只有 %.2f GB（%.0f%%）走节点间的 50 GB/s。"
      % (inter_bytes / 1e9, inter_bytes / (intra_bytes + inter_bytes) * 100))
print()
print("【哪一段是瓶颈？——这是一条可以解出来的判据】")
print("  两段之比 t_intra / t_inter = G × (M-1)/M × BW_ib / BW_nv（G = 每节点卡数）")
print("  M 较大时它趋于 G × BW_ib / BW_nv，于是：")
ratio_bw = BW_H100_UNIDIR / BW_IB_NDR_UNIDIR     # 9
print("    平衡条件 = **G = 节点内单向带宽 ÷ 节点间单向带宽 = %.0f / %.0f = %.0f**"
      % (BW_H100_UNIDIR, BW_IB_NDR_UNIDIR, ratio_bw))
print("    本例 G = %d < %.0f ⟹ t_intra/t_inter = %.3f ⟹ **节点间那一腿是瓶颈**（略）"
      % (G_PER_NODE, ratio_bw, t_intra / t_inter))
print()
print("  配比（GPU:网卡）正是从这里进来的：每节点网卡少一半，节点聚合 IB 带宽就少一半，")
print("  t_inter 翻倍，比值掉到 %.3f——节点间那一腿更吃紧。" % (t_intra / (t_inter * 2)))
print("  **节点内带宽是 GPU 的属性，节点间带宽是「每卡配几张网卡」的属性**，")
print("  所以后者是可以在系统设计里调的旋钮。")
print()
print("  自检（换一个等价的算法，看是否得到同一个数）：")
print("    节点聚合 IB 带宽 = %d 卡 × %.0f GB/s = %.0f GB/s（1:1 配比）" %
      (G_PER_NODE, BW_IB_NDR_UNIDIR, G_PER_NODE * BW_IB_NDR_UNIDIR))
print("    节点跨节点总搬运 = %d × %.2f GB = %.1f GB ⟹ %.4f s （与上面一致 ✓）"
      % (G_PER_NODE, inter_bytes / 1e9, G_PER_NODE * inter_bytes / 1e9,
         G_PER_NODE * inter_bytes / (G_PER_NODE * BW_IB_NDR_UNIDIR * 1e9)))
print()
print("  ⚠ 两条口径警告（正文必须写清）：")
print("    1. 节点内 vs 节点间 = %.0f : 1 是**双向**口径（900 vs 50）；"
      % (900 / 50))
print("       按**单向**比是 %.0f : 1（450 vs 50）。**两个数都对**，取决于表达什么。"
      % (BW_H100_UNIDIR / BW_IB_NDR_UNIDIR))
print("    2. 本节的 50 GB/s 是「400 Gb/s ÷ 8」的上界，**未扣 IB 链路层开销**；")
print("       IB 与 RoCE 的头开销不同，不要混用（见 specs.UNSOURCED）。")


# ===========================================================================
rule("5. 本程序引用的规格值（全部来自 specs.py，带来源注释）")

print("  NVLINK_PER_GPU_BIDIR_GBps_BY_GEN[4] = 900 GB/s    NVIDIA NVLink 规格页（双向）")
print("  HGX_H100_8GPU_BISECTION_TBps       = 3.6 TB/s    NVIDIA HGX 平台对比表（直述）")
print("  IB_PER_PORT_GBps['NDR']            = 50 GB/s     NVIDIA 文档（400 Gb/s ÷ 8）")
print("  DGX_H100_NIC_PER_GPU               = 1           DGX H100 用户指南端口映射表")
print("  EXTERNAL_RING_STEP_LATENCY_US      = 5 μs        performance-models/02（引用）")
print()
print("本程序**没有**产出任何实测值：它算的是下界与串行相加。")

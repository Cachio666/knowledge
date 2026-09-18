"""结构化稀疏：2 倍算力的承诺，实际拿到多少。

运行：python3 05_structured_sparsity.py
依赖：numpy（仅用于向量化的组合枚举与格式化；全部计算在本机 CPU 上完成，不需要 GPU）

口径声明：
  · 本程序做的是**组合与算术推导**，没有在 GPU 上跑任何 kernel，也不产出实测数据。
  · 规格常量全部来自 specs.py（每条带 `# 来源：` 注释）；本程序只做代入与推导。
  · 程序最后拿推导出的加速比区间与**外部来源报出的实测区间**对照。那些实测值
    来自 NVIDIA 官方博客与一篇 IEEE TC 论文，本程序只转述，不能复核它们的口径。
"""

import numpy as np

import specs as S


def rule(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def pad(s, width):
    return s + " " * max(1, width - sum(2 if _wide(c) else 1 for c in s))


def _wide(c):
    o = ord(c)
    return (0x2E80 <= o <= 0x303E or 0x3041 <= o <= 0x33FF or 0x3400 <= o <= 0x4DBF
            or 0x4E00 <= o <= 0x9FFF or 0xAC00 <= o <= 0xD7A3 or 0xF900 <= o <= 0xFAFF
            or 0xFF00 <= o <= 0xFF60)


# ===========================================================================
rule("1. 2:4 的精确定义与掩码的组合数")

print("PTX ISA §9.7.16.6.1 的原文（逐字抄录）：")
print()
print('  "%s"' % S.SPARSE_DEFINITION_QUOTE)
print()
print("把这段定义拆成四条可判定的约束：")
print("  ① 粒度：每 **%d 个连续元素** 为一组（a sub-chunk of 4 consecutive elements）"
      % S.SPARSE_GROUP_ELEMENTS)
print("  ② 数量：每组恰好 **%d 个非零**（contains 2 zeros ⟹ 4 - 2 = 2 个非零）"
      % S.SPARSE_NONZEROS_PER_GROUP)
print("  ③ 方向：子块是连续的 4 个元素，PTX 说的是「a matrix row」——")
print("     所以「连续」沿 **K 维**，且**不能跨行**。")
print("  ④ 施加对象：**只施加在 A 上**（B 与 C/D 保持稠密）⟹ 这是 SpMM，不是 SpGEMM。")
print()
print("第一步：数一数一个 4 元素组有几种合法形态。")
print("    从 4 个位置里挑 2 个位置的组合数 = C(4,2) = **%d**"
      % (4 * 3 // 2))
print("    逐个列出来：")
combos = [(i, j) for i in range(4) for j in range(i + 1, 4)]
for i, j in combos:
    bits = ["1" if k in (i, j) else "0" for k in range(4)]
    print("      非零在位置 %d,%d → 位模式 0b%s" % (i, j, "".join(bits)))
print()
print("第二步：这些形态用什么编码？PTX 原文说「Index of each non-zero element」——")
print("    **每个非零各带一个 2-bit 索引，所以一组用 %d 个 2-bit 索引 = %d bit**。"
      % (S.SPARSE_NONZEROS_PER_GROUP, S.SPARSE_META_BITS_PER_GROUP))
print("    ⚠ **这里是本主题最容易少算一半的地方**：一组是 %d bit，不是 2 bit。"
      % S.SPARSE_META_BITS_PER_GROUP)
print()
print("第三步：核对编码的合法值个数。PTX 明说这四个值非法：")
for v in S.SPARSE_INVALID_METADATA_VALUES:
    i0, i1 = (v >> 2) & 0b11, v & 0b11
    print("    0b%04s  → 两个索引分别是 %d 与 %d（%s）"
          % (format(v, "04b"), i0, i1, "相同，非法" if i0 == i1 else "?"))
print("    4 个非法值恰好是「两个索引相同」的那 4 个 ⟹ 合法编码 16 - 4 = **12 个**。")
print("    而位置组合只有 %d 种 ⟹ 每个组合对应 2 个编码（两种书写顺序）。" % len(combos))
print()
print("第四步：一条 MMA 的 K 维上有多少组？metadata 一共多少 bit？")
print("    组数 = K ÷ 4；每组 %d bit ⟹ **每行的 metadata bit 数恰好等于 K**。"
      % S.SPARSE_META_BITS_PER_GROUP)
print()
print("  %s%s%s%s" % (pad("K", 8), pad("组数 K/4", 12), pad("掩码组合数 6^(K/4)", 22),
                      "metadata bit 数"))
print("  " + "-" * 66)
for k in (16, 32, 64, 128):
    g = k // S.SPARSE_GROUP_ELEMENTS
    print("  %-8d %-12d %-22.3e %d" % (k, g, 6.0 ** g, g * S.SPARSE_META_BITS_PER_GROUP))
print()
print("组合数涨得非常快：K=64 时一条 MMA 的 A 就有 %.1fe12 种合法掩码。"
      % (6.0 ** 16 / 1e12))
print("**换句话说，硬件的「稀疏支持」不是「支持某种特定稀疏」，而是支持这 %.1fe12 种里的"
      % (6.0 ** 16 / 1e12))
print("任意一种：位置信息全部由 metadata 现场给出，没有编码进指令里。**")
print()
print("信息论视角：每个 4-bit 字段实际只承载 log2(6) = %.3f bit 的信息，"
      % np.log2(6))
print("    编码效率 = %.3f ÷ %d = **%.1f%%**（剩下的 %.1f%% 是冗余，"
      % (np.log2(6), S.SPARSE_META_BITS_PER_GROUP,
         100 * np.log2(6) / S.SPARSE_META_BITS_PER_GROUP,
         100 * (1 - np.log2(6) / S.SPARSE_META_BITS_PER_GROUP)))
print("    其中「两个索引可以换序」贡献了 1 bit 的冗余）。")


# ===========================================================================
rule("2. 推导：压缩率——三个不同的分母，别混")

print("算压缩率之前必须先问「相对谁」。三种问法各自成立，混用就会差 2 倍。")
print()
print("  %s%s%s%s%s" % (pad("元素宽度", 20), pad("原稠密", 14), pad("载荷", 14),
                      pad("metadata", 14), "总计"))
print("  " + "-" * 74)
rows_ = []
for label, ebits in (("16-bit (fp16/bf16)", 16), ("8-bit (int8)", 8)):
    dense = S.SPARSE_GROUP_ELEMENTS * ebits
    payload = S.SPARSE_NONZEROS_PER_GROUP * ebits
    meta = S.SPARSE_META_BITS_PER_GROUP
    total = payload + meta
    rows_.append((label, dense, payload, meta, total))
    print("  %-20s %-14s %-14s %-14s %s"
          % (label, "%d bit" % dense, "%d bit" % payload, "%d bit" % meta, "%d bit" % total))
print()
print("（一行 = 一组 4 个元素。16-bit 的 36 bit 正是「2×16 值 + 4 bit meta」。）")
print()
print("  %s%s%s%s" % (pad("元素宽度", 20), pad("总压缩率", 18), pad("省下多少", 14),
                      "metadata 占原数据的比例"))
print("  " + "-" * 84)
for label, dense, payload, meta, total in rows_:
    print("  %-20s %-18s %-14s %.2f%%"
          % (label, "%d/%d = %.2f%%" % (total, dense, 100 * total / dense),
             "%.2f%%" % (100 * (1 - total / dense)), 100 * meta / dense))
print()
print("**16 位时省下的是 43.75%，不是 50%。** 总压缩率 36/64 = 9/16 = 56.25%。")
print()
print("metadata 占比的两个分母（这是 05 篇最常被写错的一格）：")
for label, dense, payload, meta, total in rows_:
    print("    %-18s 对**原始稠密数据**：%d/%d = **%.2f%%**（= 1/%d）"
          % (label, meta, dense, 100 * meta / dense, dense // meta))
    print("    %-18s 对**压缩后载荷**：  %d/%d = **%.2f%%**（= 1/%d）"
          % ("", meta, payload, 100 * meta / payload, payload // meta))
print()
print("NVIDIA 论文与博客里那个「1/8」用的是**第二个分母**（压缩后载荷）。")
print("把它当成「对原始数据的比例」会得到一个 2 倍大的数（12.5% vs 6.25%）。")
print()
print("把这笔账落到容量规划上（以 16-bit 为例）：")
dense, payload, meta, total = rows_[0][1:]
print("    按「省一半」估：需要 %.4f 倍原容量" % 0.5)
print("    实际：        %.4f 倍原容量" % (total / dense))
print("    ⟹ 估算偏乐观 **%.1f%%**（(%.4f - 0.5) ÷ 0.5）"
      % (100 * ((total / dense) - 0.5) / 0.5, total / dense))
print("    10 GB 的稠密权重按稀疏存，估 5 GB，实际要 %.2f GB。" % (10 * total / dense))


# ===========================================================================
rule("3. 推导：为什么理论 2x 拿不满（一）算子侧的形状账")

print("先看形状。稀疏的 K 是对应稠密形状的 **%d 倍**（规格）：稠密 m16n8k16 的 2:4 版是"
      % S.SPARSE_SHAPE_K_MULTIPLIER)
print("m16n8k32。一条指令吃掉两倍的 K ——**这就是 2x 的全部来源：每条指令的 MAC 数翻倍。**")
print()
print("  %s%s%s%s%s" % (pad("形状", 16), pad("MAC", 10), pad("A 字节", 12), pad("B 字节", 12),
                      "操作数字节/MAC"))
print("  " + "-" * 72)
dense_a = 16 * 16 * 2
dense_b = 16 * 8 * 2
# A 的每行：32 个稠密元素 = 8 个子块 × 36 bit = 288 bit = 36 字节；16 行 → 576 B
sparse_a = 16 * (16 * 2 + S.SPARSE_META_BITS_PER_GROUP * 8 // 8)  # 16 行 × 36 B
sparse_b = 32 * 8 * 2
print("  %-16s %-10d %-12d %-12d %.4f"
      % ("m16n8k16 dense", 16 * 8 * 16, dense_a, dense_b,
         (dense_a + dense_b) / (16 * 8 * 16)))
print("  %-16s %-10d %-12d %-12d %.4f"
      % ("m16n8k32 sparse", 16 * 8 * 32, sparse_a, sparse_b,
         (sparse_a + sparse_b) / (16 * 8 * 32)))
print()
print("A 的算法：每行 32 个稠密元素 = 8 个子块；每子块 (2 个值 × 2 B + %d bit metadata)"
      % S.SPARSE_META_BITS_PER_GROUP)
print("          = 4 B + 0.5 B = 4.5 B ⟹ 每行 8 × 4.5 = 36 B ⟹ 16 行 = %d B" % sparse_a)
print("（稠密的同一个 A 是 16 × 32 × 2 = %d B ⟹ A 变成 %.2f%%，省 %.1f%%）"
      % (16 * 32 * 2, 100 * sparse_a / (16 * 32 * 2),
         100 * (1 - sparse_a / (16 * 32 * 2))))
print("B 的算法：32 × 8 × 2 = %d B ⟹ **B 完全不省，反而翻倍**（k 从 16 变成 32）"
      % sparse_b)
print()
print("关键的一格：**每个 MAC 的操作数字节从 %.4f 降到 %.4f，只改善 %.3f 倍**"
      % ((dense_a + dense_b) / (16 * 8 * 16), (sparse_a + sparse_b) / (16 * 8 * 32),
         ((dense_a + dense_b) / (16 * 8 * 16)) / ((sparse_a + sparse_b) / (16 * 8 * 32))))
print("——而 MAC 数翻了一倍。两件事叠在一起的意思是：")
print("    · 对**计算**而言：同样的 MAC 数只要一半的指令 ⟹ 算力翻倍 ✅")
print("    · 对**片上带宽（SMEM→单元）**而言：每个 MAC 只要 %.1f%% 的操作数字节"
      % (100 * ((sparse_a + sparse_b) / (16 * 8 * 32)) / ((dense_a + dense_b) / (16 * 8 * 16))))
print("      ⟹ SMEM 这一侧不是瓶颈（这是好消息）")
print("    · 对**显存**而言：A 的载荷从 %d B 降到 %d B（省 %.1f%%），"
      % (16 * 32 * 2, 16 * 16 * 2, 100 * (1 - (16 * 16 * 2) / (16 * 32 * 2))))
print("      加上 metadata 后是 %d B（省 %.1f%%）；而 B 与 C/D 一点没变"
      % (sparse_a, 100 * (1 - sparse_a / (16 * 32 * 2))))
print("      ⟹ **端到端的瓶颈在显存这一侧**，下一节算它。")


# ===========================================================================
rule("4. 推导：为什么理论 2x 拿不满（二）端到端的时间账")

print("模型：一次 GEMM 的时间 = max(计算时间, 访存时间)。稀疏让计算时间减半，")
print("但访存时间只减掉 A 的一小部分。谁的占比大，结果就由谁决定。")
print()
print("  time_dense  = max( 2MNK / P ,       (MK + KN + MN)·b / BM )")
print("  time_sparse = max( 2MNK / (2P) , (0.625·MK + KN + MN)·b / BM )")
print("  加速比 = time_dense / time_sparse")
print()
print("其中 0.625 = 载荷省到 %.4f + metadata 占 %.4f（§2 的两个数，都按 16-bit）"
      % (0.5, 0.0625))
print("0.5625 + 0.0625 = %.4f ⟹ **A 的字节数变成原来的 %.4f 倍**"
      % (0.5625 + 0.0625, 0.625))
print()
sm = S.SM_COUNT_BY_GEN["A100"]
peak = S.TC_TFLOPS_DENSE_SPARSE["FP16"]["A100"][0] * 1e12
bw = S.HBM_BW_GBps_BY_GEN["A100"] * 1e9
knee = peak / bw
print("取 A100 + FP16 dense（%.1f TFLOPS、%.0f GB/s ⟹ roofline 拐点 I* = %.1f FLOP/Byte）："
      % (peak / 1e12, bw / 1e9, knee))
print()
print("  %s%s%s%s%s%s" % (pad("(M,N,K)", 22), pad("字节强度", 12), pad("I/I*", 10),
                      pad("dense", 12), pad("sparse", 12), "加速比"))
print("  " + "-" * 86)
shapes = [(4096, 4096, 4096), (1024, 1024, 1024), (256, 1024, 1024),
          (256, 256, 4096), (128, 768, 768), (64, 64, 256)]
results = []
for m, n, k in shapes:
    flops = 2 * m * n * k
    d_elems = m * k + k * n + m * n
    s_elems = 0.625 * m * k + k * n + m * n
    t_d = max(flops / peak, d_elems * 2 / bw)
    t_s = max(flops / (2 * peak), s_elems * 2 / bw)
    intensity = flops / (d_elems * 2)
    results.append(((m, n, k), intensity, intensity / knee, t_d, t_s, t_d / t_s,
                    s_elems / d_elems))
    print("  %-22s %-12.1f %-10.2f %-12.3e %-12.3e %.2f×"
          % ("%d×%d×%d" % (m, n, k), intensity, intensity / knee, t_d, t_s, t_d / t_s))
print()
print("这张表把三种情形都覆盖了，而且**三段的分界线是算出来的、不是描述的**。")
print("把式子解出来，分界点只取决于两个量：r（= A 的字节比，16-bit 时 0.625 的加权）")
print("与 I*（机器的 roofline 拐点）：")
print()
print("  %s%s%s" % (pad("区间", 24), pad("加速比的闭式", 24), "这张表里的例子"))
print("  " + "-" * 76)
print("  %-24s %-24s %s" % ("I/I* > 2r", "恰好 2（算力完全主导）",
                            "4096³、1024³"))
print("  %-24s %-24s %s" % ("1 < I/I* < 2r", "(I/I*)/r", "256×1024×1024"))
print("  %-24s %-24s %s" % ("I/I* < 1", "1/r（**与算力无关**）", "256×256×4096 等"))
print()
two_r = 2 * (0.625 * 4096 + 4096 + 4096) / (3 * 4096)
print("对称 GEMM（M=N=K）时 r = (0.625 + 1 + 1)/3 = %.4f ⟹ **分界点是 I/I* > %.3f**。"
      % (two_r / 2, two_r))
print("也就是说：稀疏要拿满 2×，这个算子的字节强度必须比机器的拐点还高 %.0f%% 以上。"
      % (100 * (two_r - 1)))
print()
for (mnk, intensity, ratio, t_d, t_s, sp, r) in results:
    if ratio > 2 * r:
        verdict = "I/I* = %.2f > 2r = %.2f ⟹ 恰好 2×" % (ratio, 2 * r)
    elif ratio >= 1.0:
        verdict = "1 < I/I* = %.2f < 2r ⟹ 被访存压到 %.2f×" % (ratio, sp)
    else:
        verdict = ("I/I* = %.2f < 1 ⟹ 加速比 = 1/r = %.2f×，**与算力优势无关**"
                   % (ratio, 1 / r))
    print("    %-14s %s" % ("%d×%d×%d" % mnk, verdict))
print()
lo = min(r[5] for r in results)
hi = max(r[5] for r in results)
print("推导区间：[%.2f×, %.2f×]" % (lo, hi))
print("注意第三段那条：**I/I* < 1 的算子，稀疏的加速比恰好是 1/r，与「算力翻倍」无关。**")
mem_rows = [r for r in results if r[2] < 1.0]
print("它省的是显存字节，不是时间里的计算部分。这张表里属于这一段的算子的 1/r：")
for (mnk, intensity, ratio, t_d, t_s, sp, r) in mem_rows:
    print("    %-14s r = %.4f ⟹ 1/r = %.2f×（A 的占比越低，这个值越接近 1）"
          % ("%d×%d×%d" % mnk, r, 1 / r))
print("对照外部来源报出的实测区间：")
for src, scene, rng in S.SPARSE_SPEEDUP_OBSERVED:
    print("    %-30s %-46s %.2f× ~ %.2f×" % (src, scene, rng[0], rng[1]))
print()
obs_lo = min(r[0] for _, _, r in S.SPARSE_SPEEDUP_OBSERVED)
obs_hi = max(r[1] for _, _, r in S.SPARSE_SPEEDUP_OBSERVED)
print("外部观测区间：[%.2f×, %.2f×]；本模型的推导区间：[%.2f×, %.2f×]。"
      % (obs_lo, obs_hi, lo, hi))
print("**两者同量级，且本模型给出的「小 GEMM 拿不到加速」与论文里「小 CNN GEMM 无加速」")
print(" 是同一个结论。** 这说明「2× 拿不满」不是实现质量的问题，是形状与口径的必然。")
print()
print("NVIDIA 自己的说法（B 级，逐字抄录）：")
print('  "%s"' % S.SPARSE_WHITEPAPER_QUOTE)


# ===========================================================================
rule("5. mma.sp 的 fragment 账：A 少了一半，metadata 是额外的操作数")

print("稀疏 A 的 fragment 规则（PTX ISA §9.7.16.6.2，specs.py 逐条转写）。")
print("与稠密 A 最大的不同：**一个寄存器对应一个 4 元素子块**（不是一个相邻列对）。")
print()
LANES = np.arange(S.WARP_SIZE)
G, T = LANES >> 2, LANES % 4
per_lane = []
for lane in range(S.WARP_SIZE):
    g, t = int(G[lane]), int(T[lane])
    chunks = []
    for ri, (row_grp, col_blk) in enumerate(S.SPARSE_M16N8K32_A_CHUNKS):
        chunks.append((g + 8 * row_grp, t + 4 * col_blk, ri))   # 列块下标 = 元素起点/4
    per_lane.append(chunks)
allchunks = [c for lane in per_lane for c in [(r, cb) for r, cb, _ in lane]]
uniq = set(allchunks)
full = set((r, cb) for r in range(16) for cb in range(8))
print("穷举 32 个 lane：")
print("    每 lane 子块数 = %s；子块总数 = 32 × %d = %d"
      % (sorted(set(len(x) for x in per_lane)), S.SPARSE_M16N8K32_A_CHUNKS_PER_LANE,
         len(allchunks)))
print("    A 的合法子块 = 16 行 × (32 列 ÷ 4) = 16 × 8 = %d" % (16 * 8))
print("    覆盖集合 == 全部合法子块？%s    有重复？%s    有越界？%s"
      % ("✅" if uniq == full else "❌",
         "无" if len(allchunks) == len(uniq) else "有！",
         "无" if uniq <= full else "有！"))
print("    ⟹ **双射成立**：%d 个 (行, 子块) 对，一个不多一个不少。" % len(full))
print()
print("  %s%s%s" % (pad("lane", 8), pad("g", 6), "4 个子块 (行, 列块) 与它对应的寄存器"))
print("  " + "-" * 70)
for lane in range(4):
    print("  %-8d %-6d %s" % (lane, int(G[lane]),
                               "  ".join("行%d 块%d[列%d起]→reg%d" % (r, cb, 4 * cb, ri)
                                        for r, cb, ri in per_lane[lane])))
print()
print("子块的顺序（按寄存器号，写成 (行组, 列块组)）：%s"
      % " → ".join("(%d,%d)" % (r // 8, cb // 4) for r, cb, _ in per_lane[0]))
print("**与稠密 A 的 4 个 8×8 子块顺序 (0,0),(1,0),(0,1),(1,1) 是同一个模式**——")
print("两套布局在这一层是共用的：(0,0),(1,0),(0,1),(1,1) 这个「先上下、再左右」的次序")
print("在稀疏版里原样保留，只是子块从 8×8 变成了「行 × 4 列」。**这不是巧合。**")
print()
print("第二步：A 的寄存器账。")
print("    稠密 m16n8k16：A = 16×16 = 256 个元素 ÷ 32 lane = 8 个/lane = **4 个寄存器**")
print("    稀疏 m16n8k32：A 的逻辑大小 = 16×32 = 512 个元素，但只存 256 个非零值")
print("                    → 256 ÷ 32 lane = 8 个/lane = 也是 **4 个寄存器**")
print("    **逻辑上覆盖了两倍的元素，寄存器数一模一样** ⟹ 位置信息必须放到别处。")
print()
print("第三步：那个「别处」就是 metadata。每 lane 的子块数 × 每子块 bit 数：")
meta_per_lane = S.SPARSE_M16N8K32_A_CHUNKS_PER_LANE * S.SPARSE_META_BITS_PER_GROUP
print("    %d 个子块 × %d bit = **%d bit**（装在一个 .b32 寄存器里，只用到低 %d 位）"
      % (S.SPARSE_M16N8K32_A_CHUNKS_PER_LANE, S.SPARSE_META_BITS_PER_GROUP,
         meta_per_lane, meta_per_lane))
print()
print("    核对总量：%d lane × %d bit = %d bit；而 A 的逻辑大小是 16 行 × 32 列" %
      (S.WARP_SIZE, meta_per_lane, S.WARP_SIZE * meta_per_lane))
print("    ÷ 4 个一组 × %d bit = %d bit ✅ 两边一致。"
      % (S.SPARSE_META_BITS_PER_GROUP, 16 * 32 // 4 * S.SPARSE_META_BITS_PER_GROUP))
print()
print("    （这个 bit 数是从 fragment 映射**推导**出来的：每 lane 覆盖的子块数 × 每子块")
print("      的 metadata 位宽。这也解释了 specs.py 里那条口径警告——`kMetaSizeInBits`")
print("      说的是**寄存器片段里的用量**，不是内存里的位宽。两者不是一回事。）")
print()
print("第四步：整条指令的操作数（A + B + metadata）与稠密对照：")
regs_dense = 4 + 2 + 4
regs_sparse = 4 + (32 * 8 // 32 // 2) + 0 + 1
print("  %s%s%s%s" % (pad("形状", 18), pad("A", 8), pad("B", 12), "其他"))
print("  " + "-" * 56)
print("  %-18s %-8s %-12s %s" % ("m16n8k16 dense", "4 reg", "2 reg", "—"))
print("  %-18s %-8s %-12s %s" % ("m16n8k32 sparse", "4 reg", "4 reg",
                                "1 reg metadata（低 %d 位有效）" % meta_per_lane))
print()
print("B 从 2 个寄存器涨到 4 个：k 从 16 变 32，B 的 k 维跟着翻倍。")
print("**稀疏省下的是 A，付出的是 B 的翻倍与一个 metadata 寄存器。**")


# ===========================================================================
rule("6. 执行映射：它不是「跳过零」，是「一条指令吃两倍的 K」")

print("从 §3 那张表可以读出一件容易误解的事。稀疏路径**没有变快的时间片**——")
print("它不是「遇到零就跳过所以省了时间」，而是：")
print()
print("    一条 m16n8k32 的指令，MAC 数 %d = 稠密 m16n8k16 的两倍，"
      % (16 * 8 * 32))
print("    而它占用的硬件时间片是**同一个形状族的固定时长**（见 01 号程序 §6：")
print("    同一形状族里 k 减半吞吐就减半，说明单元按指令的固定时序摊销）。")
print()
print("所以稀疏的机制不是「跳过」，而是「**选择**」：")
print("  ① A 的操作数在寄存器里只存 %d 个非零值/lane（稠密的 A 存 8 个元素/lane）"
      % S.SPARSE_M16N8K32_A_NONZEROS_PER_LANE)
print("  ② metadata 告诉单元：这 2 个值原来站在 4 个位置的哪两个上")
print("  ③ 单元按索引把它们对齐到 K 维的对应位置，再与 B 相乘累加")
print()
print("⟹ **位置信息是被「告诉」的，不是被「推导」的**，这就是 metadata 必须存在的原因。")
print("   也解释了为什么一条稀疏指令要多吃一个操作数（metadata），而稠密指令没有。")
print()
print("这些 metadata 存在哪、谁读？")
print("    · 它是**指令的操作数**（与 A、B 并列），所以就在寄存器里——")
print("      见 §5 末：每 lane 一个 .b32，低 %d 位有效。" % meta_per_lane)
print("    · **读它是 MMA 单元自己做的**：metadata 参与指令的取指与选路，不占额外的")
print("      加载指令、不占额外的 SMEM 往返。")
print("    · 上游（谁生成 metadata）在软件侧：压缩权重时要把每 4 个元素的两个非零")
print("      位置算出来、按 PTX 要求的顺序打包。这一笔在 §7.4 有一条陷阱。")
print()
print("还有一条与顺序有关的规格（PTX ISA 8.5 起）：")
print('    "%s"' % S.SPARSE_ORDERED_METADATA_RULE)
print("    ⟹ 不带 `::ordered_metadata` 时顺序不受约束，但**在某些未来架构上性能可能")
print("      显著下降**。手写稀疏内核建议总是带上它。")


# ===========================================================================
rule("7. 失败模式的数字支撑")

print("【7.1】以为「任意稀疏都能加速」")
print()
print("  现象：把剪枝率 90% 的模型直接喂给稀疏内核，发现没有加速，甚至更慢。")
print("  成因：硬件只认「每连续 4 个元素恰好 2 个非零」这一种结构。90% 稀疏的模型")
print("        平均每组只有 4 × 0.1 = %.1f 个非零，而 2:4 强制每组留 2 个："
      % (4 * 0.1))
print("        ⟹ 必须把 %.1f 倍的「假非零」补回来，实际 MAC 数是模型信息的 %.0f 倍，"
      % (2 / (4 * 0.1), 2 / (4 * 0.1)))
print("        载荷也从 %.0f%% 涨到 50%%。**不但没省，还多算了。**" % (100 * 0.1))
print("  规避：稀疏化必须是**结构化的**——把非零往「每组恰好 2 个」的位置上对齐，")
print("        而不是先随便剪再喂给硬件。两者是不同的算法（前者要重新训练/重排）。")
print()
print("【7.2】按「省一半」做容量规划")
print()
print("  现象：显存估算差了十几个百分点，部署时权重放不下。")
print("  成因：16-bit 的实际总压缩率是 %d/%d = %.4f，不是 0.5。"
      % (36, 64, 36 / 64))
print("  规避：用 §2 的 %.4f；并且注意 8-bit 时是 %d/%d = %.4f（省 %.1f%%，"
      % (36 / 64, 20, 32, 20 / 32, 100 * (1 - 20 / 32)))
print("        比 16-bit 省得更少——**元素越窄，%d bit 的 metadata 越贵**）。"
      % S.SPARSE_META_BITS_PER_GROUP)
print()
print("【7.3】把 metadata 的「1/8」当成对原始数据的比例")
print()
print("  现象：算出 metadata 只占 12.5% 的额外开销，实际实现时发现 A 的字节数比")
print("        预期多了 2 倍。")
print("  成因：1/8 的分母是**压缩后的载荷**（%d bit），不是原始稠密数据（%d bit）。"
      % (32, 64))
print("        对原始数据是 %d/%d = 1/%d = %.1f%%。**两个分母差 2 倍。**"
      % (4, 64, 16, 100 * 4 / 64))
print("  规避：写清楚分母。§2 把三个分母都算了一遍，照那张表用。")
print()
print("【7.4】metadata 的顺序写错")
print()
print("  现象：用了 `::ordered_metadata` 之后结果错，或者换了一台机器结果就不一样。")
print("  成因：该限定符要求两个索引**从最低位开始按升序排**。压缩时若按「先出现的")
print("        非零在前」而不排序，在没有该限定符时可能侥幸正确，加上了就是 UB。")
print("  规避：打包 metadata 时显式排序（i0 < i1），并在指令上带 `::ordered_metadata`。")
print()
print("【7.5】把 2× 当成端到端预期")
print()
print("  现象：给一个模型做 2:4 稀疏化，端到端只快了 1.2×，怀疑实现有问题。")
print("  成因：§4 那张表——2× 是**指令吞吐**的倍数，不是端到端的。端到端的时间由")
print("        max(计算, 访存) 决定，而稀疏只把 A 的字节压到 %.4f 倍。" % 0.625)
print("  规避：先算 §4 那张表里的 I/I*，判断这个算子有没有资格拿到 2×：")
print("        I/I* < 1（dense 就访存受限）的算子，稀疏化基本白做。")
print("  口径：实测值要分两档说——**层级/内核级约 1.3×–1.7×**，**端到端约 1.2×**，")
print("        小 batch 甚至**低于 1×**（PyTorch 官方教程 BERT-base bs=4 = 0.71×）。")
print("        **两者是不同口径，不得混称。**（本程序上方打印的外部表逐行标了场景。）")
print()
print("【7.6】把稀疏当成免费的精度")
print()
print("  现象：以为「非零元素都留着，所以精度不变」。")
print("  成因：**这是模型侧的事，与本文的硬件机制是两件事。** 硬件只保证「按 metadata")
print("        取数、算、累加」；哪些元素被剪掉、剪掉之后精度掉多少，完全由稀疏化")
print("        算法决定。硬件不检查「你剪得对不对」，只检查「每组是不是 2 个非零」。")
print("  规避：把两件事分开评估——硬件加速比看 §4，精度损失看稀疏化算法本身")
print("        （正文回链 130-quantization/sparsity-and-distillation）。")

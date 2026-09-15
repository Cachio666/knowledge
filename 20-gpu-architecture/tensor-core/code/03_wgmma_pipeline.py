"""wgmma 与流水线：操作数从寄存器搬到 SMEM，以及异步省下的等待。

运行：python3 03_wgmma_pipeline.py
依赖：numpy（仅用于向量化的算术与格式化；全部计算在本机 CPU 上完成，不需要 GPU）

口径声明：
  · 本程序做的是**算术推导**，没有在 GPU 上跑任何 kernel，也不产出实测数据。
  · 规格常量全部来自 specs.py（每条带 `# 来源：` 注释）；本程序只做代入与推导。
  · 涉及的 SMEM 交付率、HBM 延迟、寄存器文件三笔账，出处分别是
    memory-system/03（128 B/周期）、memory-system/01（400~800 周期）与
    CUDA C++ Programming Guide（65536 个寄存器/SM）。
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


H100_SM = S.SM_COUNT_BY_GEN["H100"]
H100_PEAK = S.TC_TFLOPS_DENSE_SPARSE["FP16"]["H100"][0]            # 989.5 TFLOPS
H100_CLK = 1.830                                                   # GHz（反推值，见 01 号程序 §5）
H100_PER_SM_TFLOPS = H100_PEAK / H100_SM
H100_PER_SM_FLOPS = H100_PER_SM_TFLOPS * 1e12
SMEM_BW = S.SMEM_DELIVERY_BYTES_PER_CLK                            # 128 B/周期


# ===========================================================================
rule("1. warpgroup 为什么是 4 个 warp")

print("wgmma 的 M 恒为 %d（规格锁死，不由调用者选）。问题是：为什么执行单元是"
      % S.WGMMA_M)
print("%d 个 warp，而不是 1 个、8 个？" % S.WARPGROUP_WARPS)
print()
print("第一步：算一个 m64×k16 的 A 有多少元素、摊到多少线程上。")
a_elems = 64 * 16
print("    A = m64 × k16 = %d 个 f16 元素" % a_elems)
print("    摊到 %d 个 lane 上：%d ÷ %d = %d 个元素/lane"
      % (S.WARPGROUP_THREADS, a_elems, S.WARPGROUP_THREADS, a_elems // S.WARPGROUP_THREADS))
print("    = %d 个 .f16x2 寄存器/lane（与 Ampere m16n8k16 的 A 完全一样的每 lane 开销）"
      % (a_elems // S.WARPGROUP_THREADS // 2))
print()
print("第二步：把 m64 按「一个 warp 管 16 行」切。16 这个数不是新规定——")
print("    Ampere 的 mma.m16n8k16 里，一个 warp 的 A fragment 就是 16 行。")
print("    于是：64 行 ÷ 16 行/warp = **%d 个 warp**" % (64 // 16))
print()
print("第三步：核对 16 行 × k16 能不能被一个 warp 的 32 个 lane 铺满。")
print("    每个 warp 负责 16 行 × 16 列 = %d 个元素（= Ampere m16n8k16 的 A）"
      % (16 * 16))
print("    而 Ampere 的 A fragment 每 lane 正好 %d 个元素 ⟹ 32 × %d = %d ✅"
      % (8, 8, 32 * 8))
print("    于是 4 个 warp × %d 个元素 = %d = 整个 m64×k16 的 A。**不重不漏。**"
      % (16 * 16, 4 * 16 * 16))
print()
print("  %s%s%s%s" % (pad("量", 22), pad("m64n8k16", 14), pad("m64n64k16", 14), "m64n256k16"))
print("  " + "-" * 62)
for label, f in (("D 元素总数 m×N", lambda n: 64 * n),
                 ("D 每 lane 元素数", lambda n: 64 * n // S.WARPGROUP_THREADS),
                 ("D 每 lane f32 寄存器", lambda n: 64 * n // S.WARPGROUP_THREADS)):
    print("  %-22s %-14d %-14d %d" % (label, f(8), f(64), f(256)))
print("  %-22s %-14s %-14s %s" % ("A 每 lane 寄存器", "4", "4", "4（与 N 无关）"))
print()
print("⟹ **warpgroup（128 线程）是「一个 m64 tile 的最小线程集合」**：")
print("   m64 = 4 个 warp × 16 行。换成 8 个 warp 就要把 N 或 K 再切一刀，")
print("   而 A 的 4 个寄存器/lane 正好是 Ampere 一个 warp 的量——多出来的 warp 拿不到活干。")
print()
print("N 可以在 [%d, %d] 里以 %d 为步长任取，代价完全落在累加器寄存器上："
      % (S.WGMMA_N_MIN, S.WGMMA_N_MAX, S.WGMMA_N_STEP))
print("N=256 时 D = 64 × 256 ÷ 128 = **128 个 f32 寄存器/lane**，正好是每线程上限的")
print("一半（上限 %d）。" % S.MAX_REGISTERS_PER_THREAD)


# ===========================================================================
rule("2. 核心推导：操作数若留在寄存器，要付多少寄存器")

print("规格（PTX ISA §9.7.15）：**B 必须来自 shared memory；A 可以是寄存器也可以是 SMEM。**")
print("为什么 ISA 要这么规定？把「B 放寄存器」这个反事实算一遍。")
print()
print("第一步：先算 A 与 B 的唯一数据量（不重复的寄存器数）。")
print("    A 的唯一数据：64 × 16 = %d 个 f16 元素 = %d 个 .f16x2 寄存器，"
      % (a_elems, a_elems // 2))
print("    摊到 %d 个 lane = **%d 个寄存器/lane**（无复制）"
      % (S.WARPGROUP_THREADS, a_elems // 2 // S.WARPGROUP_THREADS))
print("    B 的唯一数据：16 × N 个 f16 = %d × N ÷ 2 个寄存器" % 16)
print()
print("第二步：但**B 会在这 4 个 warp 之间被复制**。4 个 warp 各算 16 行、覆盖全部 N 列，")
print("    所以每个 warp 都需要完整的 k16×N 那一份 B。用 Ampere 的 fragment 分法，")
print("    一个 warp 需要的 B 每 lane 元素数 = k16 × N ÷ 32 = N/2 个 f16 = N/4 个寄存器。")
print()
print("  %s%s%s%s%s" % (pad("N", 8), pad("A 寄存器/lane", 16), pad("D 寄存器/lane", 16),
                      pad("B 寄存器/lane", 16), "B 的复制倍数"))
print("  " + "-" * 74)
for n in (8, 64, 128, 256):
    b_regs = n // 4
    b_unique = 16 * n
    b_stored = S.WARPGROUP_WARPS * S.WARP_SIZE * (n // 2)
    print("  %-8d %-16d %-16d %-16d %.0f×（%d ÷ %d）"
          % (n, a_elems // S.WARPGROUP_THREADS // 2, 64 * n // S.WARPGROUP_THREADS,
             b_regs, b_stored / b_unique, b_stored, b_unique))
print()
print("**B 一旦进寄存器，就会被复制 4 份。** 这是 ISA 强制 B 走 SMEM 的第一条理由：")
print("SMEM 里的一份数据可以被 4 个 warp 同时读（多播），寄存器里的 4 份不能共享。")
print()
print("第三步：把三笔账合起来，算常驻 warp 数。寄存器按 %d 个一档上取整，"
      % S.REGISTER_ALLOC_GRANULARITY)
print("每 SM 的寄存器文件是 %d 个（%d KB）。" % (S.REGISTERS_PER_SM, S.REGISTER_FILE_PER_SM_KB))
print()
print("  %s%s%s%s%s" % (pad("N", 6), pad("B 在 SMEM", 14), pad("常驻 warp", 14),
                      pad("B 在寄存器", 16), "常驻 warp"))
print("  " + "-" * 70)


def warps_for(raw_regs):
    r = -(-raw_regs // S.REGISTER_ALLOC_GRANULARITY) * S.REGISTER_ALLOC_GRANULARITY
    return r, min(64, int(S.REGISTERS_PER_SM // (S.WARP_SIZE * r)))


CAP_WARPS = 64
for n in (8, 64, 128, 256):
    a_regs = 4
    d_regs = 64 * n // S.WARPGROUP_THREADS
    r_smem, w_smem = warps_for(a_regs + d_regs)
    r_reg, w_reg = warps_for(a_regs + d_regs + n // 4)
    print("  %-6d %-14s %-14s %-16s %s"
          % (n, "%d reg" % r_smem, "%d（%d wg）" % (w_smem, w_smem // S.WARPGROUP_WARPS),
             "%d reg" % r_reg, "%d（%d wg）" % (w_reg, w_reg // S.WARPGROUP_WARPS)))
print()
print("读法：N 越大，D 越占寄存器，B 的复制代价也越大。到 N=256 时：")
print("    B 在 SMEM  → %d 个寄存器/lane → 每 SM %d 个常驻 warp"
      % (warps_for(4 + 128)[0], warps_for(4 + 128)[1]))
print("    B 在寄存器 → %d 个寄存器/lane → 每 SM %d 个常驻 warp"
      % (warps_for(4 + 128 + 64)[0], warps_for(4 + 128 + 64)[1]))
print("**并发度从 %d 掉到 %d（%.0f%%），而多花的这 %d 个寄存器装的是同一份数据的副本。**"
      % (warps_for(4 + 128)[1], warps_for(4 + 128 + 64)[1],
         100 * (warps_for(4 + 128 + 64)[1] / warps_for(4 + 128)[1] - 1), 64))
print()
print("（A 那一侧没有这个问题：64 行 ÷ 4 个 warp = 每个 warp 16 行，互不重叠，")
print("  复制倍数 = 1。所以 A 允许放寄存器，B 不允许——**这个不对称是有账可算的**，")
print("  不是历史包袱。）")


# ===========================================================================
rule("3. 推导：操作数带宽——为什么必须多播")

print("取 m64n256k16 这个最大的形状当样本。一次 wgmma 的操作数字节：")
a_bytes = 64 * 16 * 2
b_bytes = 16 * 256 * 2
flops = 2 * 64 * 256 * 16
print("    A = 64 × 16 × 2 B = %d B" % a_bytes)
print("    B = 16 × 256 × 2 B = %d B" % b_bytes)
print("    合计 = %d B，而工作量 = 2·64·256·16 = %d FLOP" % (a_bytes + b_bytes, flops))
print("    ⟹ 字节强度 = %d ÷ %d = **%.1f FLOP/Byte**"
      % (flops, a_bytes + b_bytes, flops / (a_bytes + b_bytes)))
print()
print("第一步：要打满 H100 的 FP16 dense 峰值，每秒要发多少条 wgmma？")
per_s = H100_PEAK * 1e12 / flops
print("    %.1fe12 ÷ %d = **%.4fe9 条/秒**" % (H100_PEAK, flops, per_s / 1e9))
print()
print("第二步：若 A、B 各读一次（多播），操作数需要多少带宽？")
bw_multi = per_s * (a_bytes + b_bytes)
bw_multi_sm = bw_multi / H100_SM
print("    %.4fe9 × %d B = %.1fe12 B/s = **%.1f TB/s（整卡）**"
      % (per_s / 1e9, a_bytes + b_bytes, bw_multi / 1e12, bw_multi / 1e12))
print("    每 SM：%.1fe12 ÷ %d = %.1f GB/s = **%.1f B/周期**（按 %.3f GHz）"
      % (bw_multi / 1e12, H100_SM, bw_multi_sm / 1e9,
         bw_multi_sm / (H100_CLK * 1e9), H100_CLK))
print("    共享内存无冲突时的交付率 = %d B/周期 ⟹ **占用 %.1f%%**"
      % (SMEM_BW, 100 * bw_multi_sm / (H100_CLK * 1e9) / SMEM_BW))
print()
print("第三步：若硬件对 4 个 warp 各交付一份 B（不复制就做不到），会怎样？")
bw_copy = per_s * (a_bytes + 4 * b_bytes)
bw_copy_sm = bw_copy / H100_SM
print("    %.4fe9 × (%d + 4×%d) B = %.1fe12 B/s = **%.1f TB/s（整卡）**"
      % (per_s / 1e9, a_bytes, b_bytes, bw_copy / 1e12, bw_copy / 1e12))
print("    每 SM：%.1f B/周期 = **%.0f%% 的交付率**"
      % (bw_copy_sm / (H100_CLK * 1e9), 100 * bw_copy_sm / (H100_CLK * 1e9) / SMEM_BW))
print()
print("结论有两条，都很硬：")
print("  ① **按唯一字节算，SMEM 带宽占 %.1f%%**——够用，但有 4 成余量已经被吃掉。"
      % (100 * bw_multi_sm / (H100_CLK * 1e9) / SMEM_BW))
print("  ② **若每个 warp 各拿一份 B，需要 %.0f%% 的交付率——超过物理上限**。"
      % (100 * bw_copy_sm / (H100_CLK * 1e9) / SMEM_BW))
print("     ⟹ 硬件**必须**把同一份 B 多播给 4 个 warp，这不是优化而是能不能跑起来的问题。")
print("     这也解释了为什么 B 走 SMEM 描述符、而描述符里没有「哪一份」这种信息：")
print("     读的是同一块地址。")
print()
print("一个待确认项：上面算的是**读**这一侧。流水线里 TMA 还要往 SMEM **写**同样多的字节")
print("（每级 %.1f KB / %d 周期 ≈ 同样 %.1f B/周期）。读写两条加起来是 %.0f B/周期，"
      % ((a_bytes + b_bytes) / 1024, 128, bw_multi_sm / (H100_CLK * 1e9),
         2 * bw_multi_sm / (H100_CLK * 1e9)))
print("是否超出一条 %d B/周期的通道，取决于该阵列的读端口与写端口是否独立——"
      % SMEM_BW)
print("本次采用的来源只给了「无冲突时的交付率」，没有区分读写。**这一条记为待确认，")
print("本文不用它下结论**（要么读写独立，要么实际可达率低于标称）。")
print()
print("顺带算清另一笔账：A 若也走 SMEM，要多付多少？")
bw_a_smem = per_s * a_bytes / H100_SM / (H100_CLK * 1e9)
print("    单独读 A 需要 %.1f B/周期，占交付率的 %.1f%%"
      % (bw_a_smem, 100 * bw_a_smem / SMEM_BW))
print("    ⟹ 把 A 放寄存器（4 个寄存器/lane）能省下这 %.1f B/周期，"
      % bw_a_smem)
print("      这就是 §2 里「A 可以放寄存器」这条规格的收益侧。")


# ===========================================================================
rule("4. 推导：流水深度要几级才够掩盖延迟")

print("先算一级（一个 k 步）的计算时间。k 步的步长是 16，所以一级的数据是")
print("A 的 64×16 与 B 的 16×N 两块。")
print()
print("  %s%s%s%s%s" % (pad("N", 6), pad("每级字节", 12), pad("每级 FLOP", 14),
                      pad("每级计算(周期)", 18), "SMEM 最多放几级"))
print("  " + "-" * 74)
for n in (8, 64, 128, 256):
    stage_bytes = a_bytes + 16 * n * 2
    stage_flops = 2 * 64 * n * 16
    cycles = stage_flops / H100_PER_SM_FLOPS * H100_CLK * 1e9
    capacity = int(S.SMEM_PER_SM_MAX_KB * 1024 / stage_bytes)
    print("  %-6d %-12s %-14d %-18.1f %d 级"
          % (n, "%d B" % stage_bytes, stage_flops, cycles, capacity))
print()
stage_bytes = a_bytes + b_bytes
stage_flops = flops
cycles_per_stage = stage_flops / H100_PER_SM_FLOPS * H100_CLK * 1e9
print("以 N=256 为例（每级 %d B、每级 %.1f 个 SM 周期）：" % (stage_bytes, cycles_per_stage))
print("    每 SM 的算力 = %.1f TFLOPS ÷ %d SM = %.3f TFLOPS"
      % (H100_PEAK, H100_SM, H100_PER_SM_TFLOPS))
print("    每级计算 = %d FLOP ÷ %.3fe12 = %.2fe-8 s = %.1f ns = **%.1f 个周期**"
      % (stage_flops, H100_PER_SM_TFLOPS, stage_flops / H100_PER_SM_FLOPS * 1e8,
         stage_flops / H100_PER_SM_FLOPS * 1e9, cycles_per_stage))
print()
print("用「延迟 = 级数 × 每级计算时间」定深度下界（这是 Little 定律的标准用法，")
print("模型本身的推导在 10-foundations/performance-models/01，本文不重推）：")
lo, hi = S.HBM_LATENCY_CYCLES_RANGE
print("    需要的级数 D ≥ HBM 延迟 ÷ 每级周期")
for lat in (lo, hi, (lo + hi) // 2):
    print("      延迟 %d 周期 → D ≥ %d ÷ %.1f = **%.1f 级** → 取整 %d 级"
          % (lat, lat, cycles_per_stage, lat / cycles_per_stage,
             int(np.ceil(lat / cycles_per_stage))))
print()
print("对上一步的容量账：")
cap = int(S.SMEM_PER_SM_MAX_KB * 1024 / stage_bytes)
print("    SMEM 每 SM 上限 %d KB ÷ 每级 %d B = **%d 级**"
      % (S.SMEM_PER_SM_MAX_KB, stage_bytes, cap))
print("    而掩盖延迟只要 %d~%d 级 ⟹ **容量不是这个形状下的约束**。"
      % (int(np.ceil(lo / cycles_per_stage)), int(np.ceil(hi / cycles_per_stage))))
print()
print("真正卡住深度的不是容量，是另外两件事：")
print("  ① **k 循环有多长**。每级只推进 k 的 16，K=4096 时有 256 步；K=64 时只有 4 步——")
print("     4 步的循环根本没有空间做 7 级流水。")
print("  ② **SMEM 还要给别的东西用**：epilogue 的 staging、A 若走 SMEM 的那一份、")
print("     以及多个 CTA 共存时每 CTA 分到的额度。")
print("     若每 CTA 只分到 %d KB，这个形状只剩 %.0f 级。"
      % (S.SMEM_PER_SM_MAX_KB // 2, S.SMEM_PER_SM_MAX_KB * 1024 / 2 / stage_bytes))


# ===========================================================================
rule("5. 异步：省下的等待，以及不遵守它就无声出错")

print("mma.sync 是同步的：发完那条指令，warp 的寄存器 D 就成了「等结果」的现场，")
print("warp 要么停在这条指令上，要么把 D 让出去。wgmma 不是——它是异步的，")
print("D 的累加在后台继续，warp 可以去干别的（算下一轮的地址、做 epilogue）。")
print()
print("那「异步」在账上值多少？把「等待」换算成周期数：")
print("    一次 wgmma 的 D 是 64×N，累加要 %d 个 SM 周期的张量单元时间（N=256 时）"
      % int(round(cycles_per_stage)))
print("    掩盖 HBM 延迟需要 %d~%d 级流水（§4）"
      % (int(np.ceil(lo / cycles_per_stage)), int(np.ceil(hi / cycles_per_stage))))
print("    ⟹ 在 D 累加的同时，warp 手上空出来的时间可以用来发下一级的拷贝与算地址。")
print("    如果 wgmma 是同步的，这些时间只能拿来等。")
print()
print("实现这套异步要两条指令族（PTX ISA §9.7.15）：")
print("  · `wgmma.commit_group`：把此前发出的 wgmma 归成一个组")
print("  · `wgmma.wait_group N`：等到「在途组数 ≤ N」再继续")
print()
print("配套的两个 fence 要求（同样是规格，不是建议）：")
print("  · `wgmma.fence` 必须在 warpgroup 里**每一条** wgmma 之前的位置发出：")
print("    第一次 wgmma 之前，以及「某线程读写寄存器」与「wgmma 访问同一批寄存器」之间。")
print("  · `wgmma.wait_group` / `commit_group` / `fence` 都带 `.sync` 与 `.aligned`：")
print("    **warpgroup 内所有线程必须执行同一条指令**，条件分支里若不能保证条件一致，")
print("    行为未定义。")
print()
print("最后一条是最容易踩的，逐字抄 PTX：")
print('    "Accessing the accumulator register or the input register containing fragments')
print('     of matrix A of a wgmma.mma_async instruction without first performing a')
print('     wgmma.wait_group that waits on a wgmma-group including that instruction is')
print('     undefined behavior."')
print("⟹ **不等 `wait_group` 就去读 D，不是「结果慢一点」，是未定义。** §7.1 展开。")


# ===========================================================================
rule("6. 顺带算一下：发射槽省了多少，以及它是不是瓶颈")

print("先统一比较口径：同一个 128×256 的输出 tile，k 走一步 16。")
print()
print("  %s%s%s%s" % (pad("代", 10), pad("指令", 20), pad("要几条指令", 14), "共几个 warp 发射槽"))
print("  " + "-" * 70)
tile = (128, 256, 16)
tile_flops = 2 * tile[0] * tile[1] * tile[2]
n_ampere = (tile[0] // 16) * (tile[1] // 8)
print("  %-10s %-20s %-14d %d" % ("Ampere", "mma.m16n8k16", n_ampere, n_ampere))
n_hopper = tile[0] // 64
print("  %-10s %-20s %-14d %d（每条由 %d 个 warp 各发一次）"
      % ("Hopper", "wgmma.m64n256k16", n_hopper, n_hopper * S.WARPGROUP_WARPS,
         S.WARPGROUP_WARPS))
print()
print("同一个 tile 的工作量 = %d FLOP。换成「每 FLOP 几个发射槽」：" % tile_flops)
for label, issues in (("Ampere mma.sync", n_ampere), ("Hopper wgmma", n_hopper * S.WARPGROUP_WARPS)):
    print("    %-18s %d ÷ %d = %.3e 个发射槽/FLOP" % (label, issues, tile_flops, issues / tile_flops))
print("    ⟹ Hopper 比 Ampere 省的倍数是 %.0f 倍"
      % ((n_ampere / tile_flops) / (n_hopper * S.WARPGROUP_WARPS / tile_flops)))
print()
print("但**发射槽本身是不是瓶颈**？把「打满峰值时每秒需要多少发射槽」算出来：")
print()
print("  %s%s%s%s" % (pad("代", 8), pad("峰值 dense", 14), pad("发射槽/秒", 16), "占 4 个槽的百分比"))
print("  " + "-" * 66)
for gen, peak, n_ins, sm, clk in (("A100", 312.0, n_ampere, 108, 1.41),
                                  ("H100", 989.5, n_hopper * S.WARPGROUP_WARPS, 132, H100_CLK)):
    per_s_ins = peak * 1e12 / tile_flops * n_ins
    per_sm_clk = per_s_ins / sm / (clk * 1e9)
    print("  %-8s %-14.1f %-16.3e %.2f%%"
          % (gen, peak, per_s_ins, 100 * per_sm_clk / S.ISSUE_SLOTS_PER_CLK_PER_SM))
print()
print("**两条都在 1%~13% 之间：发射槽远远不是瓶颈。**")
print("⟹ 「wgmma 省发射槽」这个说法在数量级上是对的（省 %.0f 倍），"
      % ((n_ampere / tile_flops) / (n_hopper * S.WARPGROUP_WARPS / tile_flops)))
print("   但它**不是**这一代把执行单元从 warp 改成 warpgroup 的理由——省下来的那 1%")
print("   换不来任何东西。真正的理由是前面三节算的那些：**形状更大、操作数离开寄存器、")
print("   以及异步**。这条结论在 04 号程序里还要用一次：那一代把发射粒度从 128 线程")
print("   缩到 1 个线程，如果动机是省发射槽，那省下来的就更没有意义了。")


# ===========================================================================
rule("7. 失败模式的数字支撑")

print("【7.1】把 wgmma 当同步指令用：不等 wait_group 就读 D")
print()
print("  现象：结果**偶尔**错，错的位置不固定；加了 printf 或者换个规模就正常了。")
print("  成因：§5 那条官方原文——D 的累加在异步进行，不等 `wgmma.wait_group` 就读它")
print("        是 undefined behavior。它不是「读到旧值」这种温和的错，硬件可以给出")
print("        任意结果。而「偶尔」来自它依赖于时序：编译器重排、寄存器分配、")
print("        循环展开都会改变窗口。")
print("  规避：读写 D 之前必须有 `wgmma.wait_group 0`（或一个足够小的 N）；")
print("        跨 wgmma 与寄存器访问之间插 `wgmma.fence`。")
print()
print("【7.2】以为 A、B 都必须在 SMEM")
print()
print("  现象：把一个本来能省 SMEM 带宽的 A 也塞进描述符，SMEM 容量与带宽都被占掉。")
print("  成因：规格只强制了 B。A 可以是寄存器（每 lane %d 个寄存器），"
      % (a_elems // S.WARPGROUP_THREADS // 2))
print("        而这一份数据本来就分布在各 lane 手上、复制倍数是 1（§2 末）。")
print("  规避：按 §3 的账决定——A 走寄存器省 %.1f B/周期，代价是 %d 寄存器/lane；"
      % (bw_a_smem, a_elems // S.WARPGROUP_THREADS // 2))
print("        在 SMEM 带宽吃紧（§3 已占 %.1f%%）而寄存器有余量时选寄存器。"
      % (100 * bw_multi_sm / (H100_CLK * 1e9) / SMEM_BW))
print()
print("【7.3】流水深度设成「越深越好」")
print()
print("  现象：SMEM 申请到上限，常驻 CTA 数掉到 1，占用率崩了，比浅流水还慢。")
print("  成因：深度是从 SMEM 里扣的。每级 %d B，%d 级要 %d KB；而上限是 %d KB/CTA，"
      % (stage_bytes, cap, stage_bytes * cap / 1024, S.SMEM_PER_SM_MAX_KB))
print("        申请得越多，每 SM 能常驻的 CTA 越少。")
print("  规避：深度按「掩盖延迟所需」定，不按「容量允许多少」定。§4 算出只需 %d~%d 级。"
      % (int(np.ceil(lo / cycles_per_stage)), int(np.ceil(hi / cycles_per_stage))))
print("        多出来的 SMEM 应该留给 epilogue staging 与并发 CTA。")
print()
print("【7.4】在条件分支里发 wgmma")
print()
print("  现象：只在 warpgroup 的部分线程上执行到 wgmma，或在一个线程间条件不一致的")
print("        `if` 里发它——结果是 UB，且常常看起来「只是慢」。")
print("  成因：`.aligned` 限定符要求 **warpgroup 内所有线程执行同一条指令**。")
print("        128 个线程的条件必须完全一致。")
print("  规避：把条件提到 warpgroup 之外（用整个 warpgroup 一致的条件），")
print("        或者让所有线程都执行、把「不做的那部分」做成形状上的空操作。")
print()
print("【7.5】把 N 拉大只为了「更大的 tile」")
print()
print("  现象：N=256 时寄存器爆掉，编译器开始 spill，性能反而下降。")
print("  成因：D = 64×N÷128 = N/2 个 f32 寄存器/lane。N=256 就是 %d 个，"
      % (64 * 256 // S.WARPGROUP_THREADS))
print("        加上 A 的 %d 个与地址/循环变量，已经贴近 %d 的上限。"
      % (4, S.MAX_REGISTERS_PER_THREAD))
print("  规避：按 §2 那张表选 N。**N 的代价全在累加器上，与操作数无关。**")

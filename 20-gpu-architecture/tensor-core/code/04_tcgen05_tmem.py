"""tcgen05 与 TMEM：累加器从寄存器里搬出去以后。

运行：python3 04_tcgen05_tmem.py
依赖：numpy（仅用于向量化的算术与格式化；全部计算在本机 CPU 上完成，不需要 GPU）

口径声明：
  · 本程序做的是**算术推导**，没有在 GPU 上跑任何 kernel，也不产出实测数据。
  · 规格常量全部来自 specs.py（每条带 `# 来源：` 注释）；本程序只做代入与推导。
  · **TMEM 的官方口径是 per CTA，不是 per SM。** 正文所有算术都按 per CTA 说。
  · Blackwell 每 SM 的 tensor core 数量**官方未披露**，本程序不使用也不推测它。
  · B200 的 SM 数（148）与 boost 时钟（1.98 GHz）都是**反推值**，用到它们的地方
    会显式标注，且只用于量级判断。
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


TMEM_CELLS = S.TMEM_LANES_PER_CTA * S.TMEM_MAX_COLUMNS_PER_CTA
TMEM_BYTES = TMEM_CELLS * S.TMEM_CELL_BITS // 8
TMEM_KB = TMEM_BYTES / 1024


# ===========================================================================
rule("1. TMEM 的规格：一块与寄存器文件和 SMEM 并列的存储")

print("PTX ISA §9.7.18 给的口径：")
print("    结构       = %d lane × 至多 %d 列，每 cell %d bit"
      % (S.TMEM_LANES_PER_CTA, S.TMEM_MAX_COLUMNS_PER_CTA, S.TMEM_CELL_BITS))
print("    容量       = %d × %d × %d B = %d B = **%.0f KB**"
      % (S.TMEM_LANES_PER_CTA, S.TMEM_MAX_COLUMNS_PER_CTA, S.TMEM_CELL_BITS // 8,
         TMEM_BYTES, TMEM_KB))
print("    分配粒度   = %d 列；合法列数 = %s"
      % (S.TMEM_ALLOC_GRANULARITY_COLUMNS, list(S.TMEM_ALLOC_LEGAL_COLUMNS)))
print("    地址编码   = 32 位：bit[31:16] = lane，bit[15:0] = column")
print("    访问限制   = 一个 warp 只能访问本 warpgroup 里 1/%d 的 lane 位置"
      % S.TMEM_WARP_ACCESS_FRACTION)
print()
print("**第一条口径警告：PTX 说的是 per CTA，不是 per SM。**")
print("    %d × %d × %d B = %.0f KB，这个数恰好等于每 SM 的寄存器文件大小（%d KB）。"
      % (S.TMEM_LANES_PER_CTA, S.TMEM_MAX_COLUMNS_PER_CTA, S.TMEM_CELL_BITS // 8,
         TMEM_KB, S.REGISTER_FILE_PER_SM_KB))
print("    一个 SM 上驻留多个 CTA 时它们如何分 TMEM，本次采用的来源没有说明——")
print("    **本文只做 per CTA 的算术，不对多 CTA 的仲裁下结论。**")
print()
print("**第二条口径警告：Blackwell 每 SM 的 tensor core 数量官方未披露。**")
print("    本程序不使用该数字，也不使用第三方的「4/SM」——第三方数字与「官方每代都在")
print("    改 tensor core 的定义」这件事放在一起，不构成可引用的规格（见 01 号程序 §8.2）。")
print()
print("把三种存储摆在一起看，重点是它们各自「谁在管」：")
print()
print("  %s%s%s%s" % (pad("存储", 16), pad("容量/SM", 14), pad("谁分配", 20), "存放什么"))
print("  " + "-" * 70)
print("  %-16s %-14s %-20s %s" % ("寄存器文件", "%d KB" % S.REGISTER_FILE_PER_SM_KB,
                                  "编译器（每线程粒度）", "操作数、地址、临时值"))
print("  %-16s %-14s %-20s %s" % ("共享内存", "%d KB" % S.SMEM_PER_SM_MAX_KB,
                                  "程序员（每 CTA 申请）", "复用中的操作数 tile"))
print("  %-16s %-14s %-20s %s" % ("TMEM", "%.0f KB" % TMEM_KB,
                                  "程序员（tcgen05.alloc）", "MMA 的累加器 D"))
print()
print("⟹ 三者容量同量级，但**粒度不同**：寄存器是每线程的、SMEM 是每 CTA 的、")
print("   TMEM 是每 CTA 且坐标是 (lane, column) 的。§2 会说清这个粒度差异为什么")
print("   直接把「把 D 留在寄存器」这条路堵死了。")


# ===========================================================================
rule("2. 核心推导：为什么 D 不能继续留在寄存器")

print("先看一个具体形状。tcgen05 的 M ∈ %s、N 步长 %d，取最大的组合算："
      % (list(S.TCGEN05_M_VALUES), S.TCGEN05_N_STEP))
print()
print("  %s%s%s%s%s" % (pad("tile (m×n)", 16), pad("元素数", 14), pad("每 lane 元素数", 20),
                      pad("f32 寄存器/lane", 20), "能不能放进寄存器"))
print("  " + "-" * 80)
tiles = [(64, 64), (64, 128), (64, 256), (128, 64), (128, 128), (128, 256)]
for m, n in tiles:
    cells = m * n
    per_lane = cells / S.WARPGROUP_THREADS
    ok = per_lane <= S.MAX_REGISTERS_PER_THREAD
    print("  %-16s %-14d %-20.1f %-20.1f %s"
          % ("%d×%d" % (m, n), cells, per_lane, per_lane,
             "可以（剩 %d 个给别的）" % (S.MAX_REGISTERS_PER_THREAD - per_lane) if ok
             else "**不可能**：超过每线程上限 %d" % S.MAX_REGISTERS_PER_THREAD))
print()
print("**128×256 的 f32 累加器，光它自己就要 %d 个寄存器/lane，而架构上限是 %d。**"
      % (128 * 256 // S.WARPGROUP_THREADS, S.MAX_REGISTERS_PER_THREAD))
print("这不是「挤一挤还能放」的问题——它超过了硬件的寄存器编号空间。")
print()
print("还要注意的是：这里的「128 个 lane」是**一个 warpgroup**。若把它摊到更多线程上，")
print("每 lane 的负担会下降，但那个 tile 就不再是「一个执行单元持有的对象」了：")
for threads in (S.WARPGROUP_THREADS, 256, 512):
    print("    摊到 %-4d 个线程 → %6.1f 个寄存器/lane（%s）"
          % (threads, 128 * 256 / threads,
             "可行" if 128 * 256 / threads <= S.MAX_REGISTERS_PER_THREAD else "仍不可行"))
print()
print("第二步：即便放得下，粒度也不对。两边的「一格」不是同一个东西：")
print()
print("  %-16s %-30s %s" % ("", "寄存器", "TMEM"))
print("  " + "-" * 76)
print("  %-16s %-30s %s" % ("分配单位", "每线程若干个（%d 个一档）"
                            % S.REGISTER_ALLOC_GRANULARITY, "每 CTA 若干列（%d 列一档）"
                            % S.TMEM_ALLOC_GRANULARITY_COLUMNS))
print("  %-16s %-30s %s" % ("坐标", "线程号（隐式）", "(lane, column) 显式编码"))
print("  %-16s %-30s %s" % ("名字空间", "每线程私有，编号从 0 开始", "整块共享，地址含 lane"))
print()
print("累加器是**一个 CTA 级对象**：它的 (行, 列) 语义要能被 MMA 单元按坐标寻址，")
print("而寄存器的编号空间里根本没有「第 128 行」这个概念——只有 32 个线程号。")
print("**把 D 放进 TMEM，是让累加器第一次有了自己的地址空间。**")


# ===========================================================================
rule("3. 推导：容量、分配粒度、一个 tile 占多少")

print("分配规则（规格，不是建议）：")
print("  · 粒度 %d 列；合法列数 %s"
      % (S.TMEM_ALLOC_GRANULARITY_COLUMNS, list(S.TMEM_ALLOC_LEGAL_COLUMNS)))
print("  · **分配一列 = 占满全部 %d 个 lane 位置**" % S.TMEM_LANES_PER_CTA)
print("  · 所以容量账要按「列」算，不是按「cell」算")
print()
print("最小的一次分配 = %d 列 × %d lane × %d B = **%d B = %d KB**"
      % (S.TMEM_ALLOC_GRANULARITY_COLUMNS, S.TMEM_LANES_PER_CTA, S.TMEM_CELL_BITS // 8,
         S.TMEM_ALLOC_GRANULARITY_COLUMNS * S.TMEM_LANES_PER_CTA * 4,
         S.TMEM_ALLOC_GRANULARITY_COLUMNS * S.TMEM_LANES_PER_CTA * 4 // 1024))
print()
print("一个 m×n 的 f32 累加器需要 n 列（每 cell 一个 f32），且 m 行占 m 个 lane 位置：")
print()
print("  %s%s%s%s%s%s" % (pad("tile (m×n)", 16), pad("需要列数", 14), pad("按 32 取整", 14),
                      pad("占用", 14), pad("占 TMEM", 14), "能同时放几个"))
print("  " + "-" * 86)
for m, n in tiles:
    alloc = -(-n // S.TMEM_ALLOC_GRANULARITY_COLUMNS) * S.TMEM_ALLOC_GRANULARITY_COLUMNS
    kb = alloc * S.TMEM_LANES_PER_CTA * 4 / 1024
    fits = S.TMEM_MAX_COLUMNS_PER_CTA // alloc
    print("  %-16s %-14d %-14d %-14s %-14s %d 个"
          % ("%d×%d" % (m, n), n, alloc, "%.0f KB" % kb,
             "%.1f%%" % (100 * kb / TMEM_KB), fits))
print()
print("注意 m=64 的那几行：**一个 m64 的 tile 只用到一半的 lane 位置。**")
print("  %d 行 ÷ %d lane = %.0f%% 的 lane 被用上，另外 %.0f%% 是空的——"
      % (64, S.TMEM_LANES_PER_CTA, 100 * 64 / S.TMEM_LANES_PER_CTA,
         100 * (1 - 64 / S.TMEM_LANES_PER_CTA)))
print("  而分配是按「列」走的，一列就占满 128 个 lane 位置。")
print("  ⟹ 要么把两个 m64 的 tile 塞进同一列的上下两半（推导：PTX 的地址含 lane 字段，")
print("     所以按 lane 偏移就能把两个 64 行的 tile 叠在一列里），要么白扔一半容量。")
print("     **本文不对「怎么叠」给出结论**——PTX 的 tcgen05 形状表里有 M=64 与 M=128")
print("     两种，它们与 allocation 的对应关系本次来源未展开，这里只指出这笔账存在。")
print()
print("再看 N 的粒度与容量的配合：")
for n in (32, 64, 128, 256, 512):
    print("    n=%-4d → 一个 tile 占 %2d 列 → 512 列里能放 %d 个"
          % (n, n, S.TMEM_MAX_COLUMNS_PER_CTA // n))
print("    ⟹ 列数必须是 2 的幂，所以 n=192 这样的 tile 要么按 256 申请，要么拆成 128+64。")
print("      **按 256 申请就白扔 64 列（12.5% 的 TMEM）。**")


# ===========================================================================
rule("4. 地址编码：一个坐标，不是一个下标")

addr_bits = 32
lane_bits = addr_bits - S.TMEM_ADDRESS_LANE_SHIFT
col_bits = S.TMEM_ADDRESS_LANE_SHIFT
addr_cells = 2 ** addr_bits
print("地址是 %d 位：lane 字段 %d 位、column 字段 %d 位。于是可寻址的 cell 数："
      % (addr_bits, lane_bits, col_bits))
print("    lane 字段可表示 %d 个值，但实际只有 %d 个 lane"
      % (2 ** lane_bits, S.TMEM_LANES_PER_CTA))
print("    column 字段可表示 %d 个值，但实际最多 %d 列"
      % (2 ** col_bits, S.TMEM_MAX_COLUMNS_PER_CTA))
print("    可寻址 cell 总数 = 2^%d = %.3e" % (addr_bits, addr_cells))
print("    实际存在的 cell  = %d × %d = %d" % (S.TMEM_LANES_PER_CTA,
                                              S.TMEM_MAX_COLUMNS_PER_CTA, TMEM_CELLS))
print("    ⟹ 地址空间是实际容量的 **%.0f 倍**（%.3e ÷ %d）"
      % (addr_cells / TMEM_CELLS, addr_cells, TMEM_CELLS))
print()
print("这个比例说明 TMEM 的「地址」不是内存下标，而是**一组坐标**：")
print("    address = (lane << %d) | column" % S.TMEM_ADDRESS_LANE_SHIFT)
print("字段留得比实际需要宽，是为了让 (lane, column) 这对坐标能被一条 32 位地址直接")
print("表达、在指令里当成一个操作数传——**这正是 §2 说的「累加器第一次有了自己的")
print("地址空间」的具体形态**。")
print()
print("顺带算一处容易踩的地方：按字段宽度算容量会算错。")
print("    按字段宽：2^%d × 2^%d = %.3e 个 cell —— 比实际大 %s 倍"
      % (lane_bits, col_bits, addr_cells, "{:,.0f}".format(addr_cells / TMEM_CELLS)))
print("    按规格：  %d × %d = %d 个 cell = %.0f KB"
      % (S.TMEM_LANES_PER_CTA, S.TMEM_MAX_COLUMNS_PER_CTA, TMEM_CELLS, TMEM_KB))


# ===========================================================================
rule("5. 单线程发射：省下的发射槽，以及它为什么不是重点")

print("规格（PTX ISA §9.7.18，逐字抄录）：")
print('    "%s"' % S.TCGEN05_ISSUE_QUOTE)
print()
print("把三代放在同一个口径上比：**同一个 128×256 的输出 tile，k 走一步 16。**")
print()
tile_flops = 2 * 128 * 256 * 16
n_ampere = (128 // 16) * (256 // 8)
n_hopper = (128 // 64) * S.WARPGROUP_WARPS
print("  %s%s%s%s" % (pad("代", 12), pad("指令", 40), pad("发射次数", 14),
                      "每 FLOP 的发射次数"))
print("  " + "-" * 84)
for label, ins, issues in (("Ampere", "mma.m16n8k16（warp 级）", n_ampere),
                           ("Hopper", "wgmma.m64n256k16（warpgroup 级）", n_hopper),
                           ("Blackwell", "tcgen05.mma 128×256×16（CTA 级）", 1)):
    print("  %-12s %-40s %-14d %.3e" % (label, ins, issues, issues / tile_flops))
print()
print("逐级省的倍数：")
print("    Hopper ÷ Blackwell = %d ÷ 1 = **%d 倍**" % (n_hopper, n_hopper))
print("    Ampere ÷ Blackwell = %d ÷ 1 = **%d 倍**" % (n_ampere, n_ampere))
print()
print("但**发射槽从来不是瓶颈**。打满峰值时每秒需要多少次发射：")
print()
print("  %s%s%s%s" % (pad("代", 12), pad("峰值 dense", 16), pad("发射次数/秒", 18),
                      "占 4 个发射槽的百分比"))
print("  " + "-" * 76)
rows_issue = [
    ("A100", 312.0, n_ampere, 108, 1.41),
    ("H100", 989.5, n_hopper, 132, 1.830),
    ("B200", 2250.0, 1, 148, 1.98),
]
for label, peak, issues, sm, clk in rows_issue:
    per_s = peak * 1e12 / tile_flops * issues
    frac = per_s / sm / (clk * 1e9) / S.ISSUE_SLOTS_PER_CLK_PER_SM
    note = "（用的是两个反推值：148 SM 与 1.98 GHz）" if label == "B200" else ""
    print("  %-12s %-16.1f %-18.3e %.3f%%  %s" % (label, peak, per_s, 100 * frac, note))
print()
print("**三代都在 0.01%~13% 之间。** 发射槽不是瓶颈，而且离瓶颈很远。")
print("⟹ 所以「单线程发射省发射槽」这句话**成立但不能当理由**——省下来的那 0.01%")
print("   在 4 个发射槽里看不见。真正的变化是另外三条：")
print()
print("  ① **形状放大到 CTA 级**。128×256 的 tile 在 warp 级指令里根本表达不出来：")
print("     warp 级最大的 m 是 16，要 8 个 warp 协作才能覆盖 128 行——而 warp 之间")
print("     没有一条指令能把它们的寄存器当作一个对象。**这是「必须换发射模型」的根因。**")
print("  ② **摆脱 `.aligned`**。wgmma 要求 warpgroup 内 128 个线程执行同一条指令")
print("     （见 03 号程序 §5）；tcgen05 只要求 1 个线程执行。")
print("     差别不是「少发 127 次」，而是**控制流解耦**：其余 127 个线程可以去跑")
print("     load、做上一个 tile 的 epilogue、或者干脆在做别的事。")
print("  ③ **D 不再占寄存器**（§2）。省下的不是「几个寄存器」，而是整个 %d KB"
      % S.REGISTER_FILE_PER_SM_KB)
print("     里原本要留给累加器的那一大块。128×256 的 D 本来要 %d 个寄存器/lane。"
      % (128 * 256 // S.WARPGROUP_THREADS))


# ===========================================================================
rule("6. 取回结果：一个 warp 只能碰 1/4 的 TMEM")

print("访问限制（规格）：TMEM 的 %d 个 lane 按 32 个一组分成 %d 块，"
      % (S.TMEM_LANES_PER_CTA, S.TMEM_WARP_ACCESS_FRACTION))
print("一个 warp 只能访问**本 warpgroup 内由 warp id 决定的那 32 个 lane**。")
print("⟹ **要覆盖全部 %d 个 lane，必须整个 warpgroup 参与。**"
      % S.TMEM_LANES_PER_CTA)
print()
print("这条限制在 epilogue 上直接变成一笔搬运账。以 128×256 的 f32 D 为例：")
d_bytes = 128 * 256 * 4
print("    D 的总字节 = 128 × 256 × 4 = **%d B = %d KB**" % (d_bytes, d_bytes // 1024))
print("    每 lane 能读自己那 32 行：一个 warp 负责 32 行 × 256 列 × 4 B = %d B = %d KB"
      % (32 * 256 * 4, 32 * 256 * 4 // 1024))
print("    %d 个 warp 各读一份 = %d KB ✅（正好是 D 的总量，无重复）"
      % (S.TMEM_WARP_ACCESS_FRACTION, d_bytes // 1024))
print()
print("第二步：把这笔搬运摊到计算上。D 是**每个 (m,n) tile 读一次**，不是每个 k 步读一次：")
print()
print("  %s%s%s%s" % (pad("K", 10), pad("k 步数", 12), pad("该 tile 的 FLOP", 20),
                      "D 的字节/FLOP"))
print("  " + "-" * 62)
for k in (64, 512, 4096, 16384):
    steps = k // 16
    f = 2 * 128 * 256 * k
    print("  %-10d %-10d %-18d %.3e" % (k, steps, f, d_bytes / f))
print()
print("第三步：把「字节/FLOP」还原成带宽需求（B200 的 FP16 dense 峰值 2250 TFLOPS）：")
peak_b200 = S.TC_TFLOPS_DENSE_SPARSE["FP16"]["B200"][0]
hbm_b200 = S.HBM_BW_GBps_BY_GEN["B200"]
print("  %s%s%s%s" % (pad("K", 10), pad("需要的读出带宽", 20), pad("对照 HBM", 14), "结论"))
print("  " + "-" * 70)
for k in (64, 512, 4096, 16384):
    f = 2 * 128 * 256 * k
    bw = peak_b200 * 1e12 * (d_bytes / f)
    ratio = bw / (hbm_b200 * 1e9)
    print("  %-10d %-18s %-12s %s"
          % (k, "%.1f TB/s" % (bw / 1e12), "%.0f%%" % (100 * ratio),
             "必须与 epilogue 融合，不能落到 HBM" if ratio > 1
             else "落回 HBM 也可以（占带宽 %.0f%%）" % (100 * ratio)))
print()
print("⟹ **K 短的时候，累加器的取回就是主瓶颈；K 长的时候它几乎免费。**")
print("   这就是为什么「把 epilogue 融合进 GEMM」在 K 小的算子上（如 attention 的")
print("   QK^T、逐层的 FC）比在 K 大的算子上更关键——它是把上表那个 %.0f%% 压下去的唯一手段。"
      % (100 * (peak_b200 * 1e12 * d_bytes / (2 * 128 * 256 * 64)) / (hbm_b200 * 1e9)))


# ===========================================================================
rule("7. 失败模式的数字支撑")

print("【7.1】沿用 Hopper 的心智模型：以为结果在寄存器里能直接后处理")
print()
print("  现象：把 Hopper 的 epilogue 原样搬过来（对 D 的寄存器做 bias 加、量化、转 fp16），")
print("        编译不过或结果全错。")
print("  成因：tcgen05 的 D **只能在 TMEM**（规格：D 必须在 tensor memory；A 可在 TMEM")
print("        或 SMEM；B 必须来自 SMEM）。要用 D，必须先 `tcgen05.ld` 把它读回寄存器。")
print("  规避：把 epilogue 显式拆成「TMEM → 寄存器 → 处理 → 写出」四步；")
print("        并注意 §6 的分块限制：读回必须由整个 warpgroup 完成。")
print()
print("【7.2】把 512 列当成「每 SM」的额度")
print()
print("  现象：按「每 SM 512 列」规划 2 个 CTA × 256 列，实际跑起来第二个 CTA 拿不到。")
print("  成因：PTX 给的口径是 **per CTA**。一个 SM 上多个 CTA 如何仲裁 TMEM，")
print("        本次采用的来源没有说明——**任何「一 SM 能放几个 CTA 的 TMEM」的规划")
print("        都是没有来源支撑的**。")
print("  规避：按 per CTA 的容量（%.0f KB）规划单 CTA 的 tile 数，把多个 CTA 的情况"
      % TMEM_KB)
print("        当作未知量处理（用运行时实测的占用率而不是纸上推算）。")
print()
print("【7.3】m64 的 tile 白扔一半 lane")
print()
print("  现象：用 M=64 的形状做小 GEMM，发现 TMEM 只能放下一半的 tile。")
print("  成因：分配按列走，一列占满 %d 个 lane；而 m64 只用 %d 个。"
      % (S.TMEM_LANES_PER_CTA, 64))
print("        M=64 时 lane 利用率 = %.0f%%。" % (100 * 64 / S.TMEM_LANES_PER_CTA))
print("  规避：要么把两个 m64 tile 按 lane 偏移叠进同一列，要么直接选 M=128 的形状。")
print("        （怎么叠：本文只在 §3 指出这笔账，不对 PTX 的 M 与 allocation 的对应")
print("         关系下结论。）")
print()
print("【7.4】以为「不需要 warpgroup 对齐」等于「可以随便调度」")
print()
print("  现象：把 epilogue 交给一个 warp 单独做，其余 warp 去跑别的活——结果只处理了")
print("        四分之一的行。")
print("  成因：发射是单线程的，但**访问不是**：一个 warp 只能碰 1/%d 的 lane。"
      % S.TMEM_WARP_ACCESS_FRACTION)
print("        发射粒度和访问粒度是两回事，规格把它们分开写了。")
print("  规避：把「谁发射」与「谁读回」分开设计。读回必须覆盖整个 warpgroup，")
print("        且每个 warp 认领自己的那 32 行。")
print()
print("【7.5】把 D 的取回开销当成常数")
print()
print("  现象：一个 K=64 的小 GEMM，算得很快，整体吞吐却很差。")
print("  成因：§6 那张表——K=64 时 D 的取回是 %.3e B/FLOP，折合 %.1f TB/s，"
      % (d_bytes / (2 * 128 * 256 * 64),
         peak_b200 * 1e12 * d_bytes / (2 * 128 * 256 * 64) / 1e12))
print("        是 HBM 带宽（%.1f TB/s）的 %.0f 倍。"
      % (hbm_b200 / 1000, 100 * (peak_b200 * 1e12 * d_bytes
                                 / (2 * 128 * 256 * 64)) / (hbm_b200 * 1e9)))
print("  规避：对 K 小的算子，epilogue 必须融合（在片上就完成缩放、转格式、写回），")
print("        而不是把 f32 的 D 写出去再读回来。")
print()
print("（本节所有以 B200 峰值换算的数字，用的是官方 FP16 dense 峰值 2250 TFLOPS；")
print("  K 那一栏的 FLOP 是 2·m·n·K，与 tile 形状无关，只用来摊销 D 的字节。）")

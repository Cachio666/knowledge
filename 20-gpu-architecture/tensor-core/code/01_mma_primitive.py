"""MMA 原语：一条指令换一个 tile，代价是形状被固定。

运行：python3 01_mma_primitive.py
依赖：numpy（仅用于向量化的组合枚举与格式化；全部计算在本机 CPU 上完成，不需要 GPU）

口径声明：
  · 本程序做的是**算术与组合推导**，没有在 GPU 上跑任何 kernel，也不产出实测数据。
  · 规格常量全部来自 specs.py（每条带 `# 来源：` 注释）；本程序只做代入与推导。
  · 程序打印出的每一个数都是文档里引用的数；文档不引用本程序没打印的数。
"""

import numpy as np

import specs as S


def rule(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def pair(p):
    """把 (dense, sparse) 二元组打成一个短串。"""
    d, s = p
    return "%.1f / %s" % (d, "—" if s is None else "%.1f" % s)


def pad(s, width):
    """按终端显示宽度补空格：东亚宽字符占 2 列，其余占 1 列。"""
    w = sum(2 if east_asian_wide(c) else 1 for c in s)
    return s + " " * max(1, width - w)


def east_asian_wide(c):
    o = ord(c)
    return (0x1100 <= o <= 0x115F or 0x2E80 <= o <= 0x303E or 0x3041 <= o <= 0x33FF
            or 0x3400 <= o <= 0x4DBF or 0x4E00 <= o <= 0x9FFF or 0xA000 <= o <= 0xA4CF
            or 0xAC00 <= o <= 0xD7A3 or 0xF900 <= o <= 0xFAFF or 0xFE30 <= o <= 0xFE6F
            or 0xFF00 <= o <= 0xFF60 or 0xFFE0 <= o <= 0xFFE6 or 0x20000 <= o)


# ===========================================================================
rule("1. MMA 形状表：形状是硬件的固定形状，不是可选参数")

print("判定依据：PTX ISA §9.7.16.1 的 Shape 表 + 同节各指令的 Target ISA Notes。")
print("**Shape 表是按 PTX ISA 版本组织的，不是按架构组织的**——要判断某形状属于")
print("哪一代，必须与 Target ISA Notes 交叉比对。只抄 Shape 表会得出「Volta 就支持")
print("m16n8k4」这类错误结论。")
print()
print("  %s%s%s" % (pad("形状", 22), pad("首次可用", 14), "最低 target"))
print("  " + "-" * 60)
for shape, (gen, target) in S.MMA_SHAPES.items():
    print("  %-22s %-14s %s" % (shape, gen, target))

print()
print("三条由「形状表 + Target ISA Notes」直接读出的修正：")
print("  · **m16n8k4 没有 fp16 版本**。k=4 的 m16n8 只有 .tf32（sm_80+）与 .f64（sm_90+）。")
print("    「Turing 支持 m16n8k4 fp16」是错的。")
print("  · **A100 没有 FP8 MMA**：e4m3/e5m2 的 mma 要 sm_89+。A100 只有 f16/bf16/tf32/int8。")
print("  · **A100 的 f64 mma 只有 m8n8k4**。另外三个 f64 形状的最低 target 是：")
for sh in S.MMA_F64_SHAPES:
    print("      %-10s %s" % (sh, S.MMA_F64_MIN_TARGET[sh]))
print()
print("wgmma 与 tcgen05 的形状是参数化的：")
print("  wgmma   : M 恒为 %d（不由调用者选），K 由精度定（f16/bf16 → 16，tf32 → 8），"
      % S.WGMMA_M)
print("            N ∈ [%d, %d] 步长 %d" % (S.WGMMA_N_MIN, S.WGMMA_N_MAX, S.WGMMA_N_STEP))
print("  tcgen05 : M ∈ %s，N 步长 %d"
      % (str(list(S.TCGEN05_M_VALUES)), S.TCGEN05_N_STEP))
print()
print("「加速变体」target（只在该精确架构上可用，不向前兼容家族成员）：")
print("  %s" % " / ".join(S.ACCELERATED_TARGETS))


# ===========================================================================
rule("2. 推导一：一条 MMA 的算术强度，以及 k 为什么不在式子里")

M, N, K = 16, 8, 16          # mma.m16n8k16 的实际形状
BYTES = 2                    # fp16 / bf16，每元素 2 字节

flops = 2 * M * N * K
elems = M * K + K * N        # A 是 m×k，B 是 k×n，操作数按元素计
print("以 mma.m16n8k16 为例（A=m×k 的 fp16，B=k×n 的 fp16，C/D=m×n 的 f32）：")
print()
print("第一步：算工作量。一次乘加 = 2 FLOP，tile 里有 m·n·k 个乘加：")
print("    FLOP = 2·m·n·k = 2·%d·%d·%d = %d" % (M, N, K, flops))
print()
print("第二步：算操作数。A 有 m·k 个元素、B 有 k·n 个元素（C/D 是累加器，不算输入）：")
print("    操作数元素数 = m·k + k·n = %d·%d + %d·%d = %d"
      % (M, K, K, N, elems))
print()
print("第三步：相除，得到**每个操作数元素支撑的 FLOP 数**：")
print("    I = 2mnk / (mk + nk) = 2mnk / (k(m+n)) = 2mn / (m+n)")
print("    I = 2·%d·%d / (%d + %d) = %.3f FLOP/元素"
      % (M, N, M, N, 2 * M * N / (M + N)))
print()
print("**第五步：k 消掉了。** 这是本节唯一反直觉的地方——「k 越大复用越多」这句话")
print("在算术强度上是错的：k 在分子分母里各出现一次，强度与 k 无关。")
print("    m16n8k16（fp16）: I = %.3f FLOP/元素" % (2 * M * N / (M + N)))
print("    m16n8k8 （tf32）: I = %.3f FLOP/元素   ← 同一个值"
      % (2 * 16 * 8 / (16 + 8)))
print()
print("k 在什么地方起作用？在**累加器的摊销**上。C/D 是 m×n 个 f32，与 k 无关，")
print("而一次指令的工作量是 2mnk——所以每个 FLOP 分摊到的累加器寄存器数是：")
print("    mn / (2mnk · 32) = 1 / (64k)   个寄存器/FLOP")
for k in (8, 16, 32, 64):
    print("      k=%-3d → %.6f 个 32-bit 寄存器/FLOP" % (k, 1.0 / (64 * k)))
print("    k 翻倍，累加器的单位开销就减半。**k 是唯一能扩的维度，因为 m 与 n 被")
print("    指令的固定形状锁死了**（§1 那张表里每个形状的 m、n 都是常量）。")
print()
print("换到字节口径（fp16 每元素 2 字节），强度是上面的一半：")
print("    I_bytes = %.3f FLOP/Byte" % (flops / (elems * BYTES)))
print("    单条指令搬运的操作数字节 = %d 元素 × %d B = %d B"
      % (elems, BYTES, elems * BYTES))


# ===========================================================================
rule("3. 推导二：与逐元素执行比，复用能力提升了多少")

print("对照场景：同一个矩阵乘，不用 tensor core。每个输出元素要读 A 的一行与 B 的一列：")
print()
print("    逐元素路径：一个输出元素 = 2k 个 FLOP，要读 2k 个操作数元素")
print("                I_scalar = 2k / 2k = **1.000 FLOP/元素**，与 k 无关")
print()
print("    MMA 路径   ：I_mma = 2mn/(m+n)")
print()
print("两者相除，就是「同一份数据被多用了几次」：")
gain = (2 * M * N / (M + N)) / 1.0
print("    增益 = I_mma / I_scalar = %.3f / 1.000 = **%.3f 倍**" % (2 * M * N / (M + N), gain))
print()
print("再把 m16n8k16 折算成一个等价的方形 tile：")
meq = 2 * M * N / (M + N)
print("    2mn/(m+n) = %.3f ⟹ 等价方形边长 m_eq = %.3f" % (meq, meq))
print("    方形 tile（m=n）的强度就是它的边长：2m²/(2m) = m")
print("    ⟹ **m16n8k16 的复用能力，相当于一个 %.2f×%.2f 的方形 tile**" % (meq, meq))
print()
print("每 lane 的元素数与寄存器数（这是「固定形状」落到 32 个 lane 上的样子）：")
print()
print("  %s%s%s%s%s"
      % (pad("形状", 16), pad("A 元素/reg", 12), pad("B 元素/reg", 12),
         pad("C 元素/reg", 12), "MAC/寄存器"))
print("  " + "-" * 74)
rows = [
    ("m16n8k16 f16",  S.M16N8K16_F16_A_ELEMS_PER_LANE, S.M16N8K16_F16_B_ELEMS_PER_LANE,
     S.M16N8K16_F32_C_ELEMS_PER_LANE, S.M16N8K16_F16_A_REGS_PER_LANE,
     S.M16N8K16_F16_B_REGS_PER_LANE, S.M16N8K16_F32_C_REGS_PER_LANE, M, N, K),
    ("m16n8k8 tf32",  S.M16N8K8_TF32_A_ELEMS_PER_LANE, S.M16N8K8_TF32_B_ELEMS_PER_LANE,
     S.M16N8K16_F32_C_ELEMS_PER_LANE, S.M16N8K8_TF32_A_REGS_PER_LANE,
     S.M16N8K8_TF32_B_REGS_PER_LANE, S.M16N8K16_F32_C_REGS_PER_LANE, 16, 8, 8),
]
for name, ae, be, ce, ar, br, cr, m_, n_, k_ in rows:
    macs = m_ * n_ * k_
    regs = ar + br + cr
    print("  %-14s %-10s %-10s %-10s %.1f MAC/reg"
          % (name, "%d / %d" % (ae, ar), "%d / %d" % (be, br), "%d / %d" % (ce, cr),
             macs / regs))
    print("     A=%d 元素(%d reg) + B=%d 元素(%d reg) + C=%d 元素(%d reg) = 每 lane %d 个寄存器"
          % (ae, ar, be, br, ce, cr, regs))
print()
print("注意两行的工作量差一倍（%d vs %d 个 MAC），而寄存器数一样（都是 10 个）。"
      % (M * N * K, 16 * 8 * 8))
print("**这就是「TF32 峰值恰好是 FP16 的一半」的形状层面原因**：同一形状族里 k 减半，")
print("寄存器占用不变、工作量减半。§6 会从时钟那一侧再推一遍，两处结论必须一致。")


# ===========================================================================
rule("4. 推导三：为什么形状是「小 m、小 n」，而 k 可以大")

print("把「每寄存器的 FLOP 数」写成 k 的函数。固定 m=16、n=8（指令锁死的固定形状），")
print("令 k 可变。每 lane 的寄存器数随 k 线性涨：")
print("    A = m·k/32 个 f16 元素，每寄存器放 2 个 → mk/64 个寄存器 = k/4")
print("    B = k·n/32 个 f16 元素 → kn/64 个寄存器 = k/8")
print("    C = m·n/32 个 f32 元素，每寄存器放 1 个 → 4 个寄存器（与 k 无关）")
print("    R(k) = k/4 + k/8 + 4 = 3k/8 + 4")
print("    FLOP(k) = 2·m·n·k = 256k")
print("    f(k) = 256k / (3k/8 + 4) = 2048k / (3k + 32)")
print()
asym = 2048.0 / 3.0
half_k = 32.0 / 3.0
print("    渐近值 f(∞) = 2048/3 = %.1f FLOP/寄存器" % asym)
print("    取到渐近值一半的 k = 32/3 = %.1f" % half_k)
print()
print("  %-8s %-14s %-16s %s" % ("k", "寄存器/lane", "FLOP/寄存器", "占渐近值"))
print("  " + "-" * 56)
for k in (4, 8, 16, 32, 64, 128, 256):
    r = 3 * k / 8.0 + 4
    f = 256.0 * k / r
    print("  %-8d %-14.1f %-16.1f %.1f%%" % (k, r, f, 100 * f / asym))
r16 = 3 * 16 / 8.0 + 4
r64 = 3 * 64 / 8.0 + 4
print()
print("读法：**k 从 16 涨到 64，收益从 %.1f%% 只涨到 %.1f%%，而寄存器占用从 %.0f 涨到 %.0f。**"
      % (100 * (256 * 16 / r16) / asym, 100 * (256 * 64 / r64) / asym, r16, r64))
print("实际发货的形状取 k=16（fp16 族），正好落在边际收益开始钝化的位置——")
print("k=16 处每寄存器 %.1f FLOP，已经是渐近值 %.1f 的 %.1f%%。"
      % (256 * 16 / r16, asym, 100 * (256 * 16 / r16) / asym))
print()
print("「小 m、小 n」的那一侧是另一笔账：m、n 决定 C 的大小（mn/32 个 f32 寄存器/lane），")
print("而 C 是**每个 warp 必须常驻**的东西。一次 k 循环里，一个 warp 会同时开几个 tile")
print("来掩盖 MMA 的延迟。把 m16n8k16 的每 lane 寄存器数记下来：")
per_tile = S.M16N8K16_F16_A_REGS_PER_LANE + S.M16N8K16_F16_B_REGS_PER_LANE \
    + S.M16N8K16_F32_C_REGS_PER_LANE
print("    每 tile %d 个寄存器/lane（A %d + B %d + C %d）"
      % (per_tile, S.M16N8K16_F16_A_REGS_PER_LANE, S.M16N8K16_F16_B_REGS_PER_LANE,
         S.M16N8K16_F32_C_REGS_PER_LANE))
print()
print("  %-10s %-12s %-16s %-16s %s"
      % ("tile 数", "寄存器/lane", "按 8 个一档", "寄存器上限的 warp", "实际常驻"))
CAP_WARPS = 64
print("  " + "-" * 76)
for ntile in (1, 2, 4, 8):
    raw = ntile * per_tile
    r = ((raw + S.REGISTER_ALLOC_GRANULARITY - 1) // S.REGISTER_ALLOC_GRANULARITY) \
        * S.REGISTER_ALLOC_GRANULARITY
    reg_limited = int(S.REGISTERS_PER_SM // (S.WARP_SIZE * r))
    actual = min(CAP_WARPS, reg_limited)
    print("  %-10d %-12d %-16d %-16d %d%s"
          % (ntile, raw, r, reg_limited, actual,
             "（撞到 %d warp/SM 的硬件上限）" % CAP_WARPS if reg_limited > CAP_WARPS
             else "（寄存器是先撞到的那条约束）"))
print()
print("**一个 warp 同时开 4 个 tile 的时候，寄存器文件开始成为真正起作用的约束。**")
print("若把形状放大到 m=n=64 的假想 tile：C 一侧就要 64·64/32 = %d 个寄存器/lane，"
      % (64 * 64 // 32))
print("连一个 tile 都开不动，更不用说「几个 tile 同时在飞」。")
print()
print("所以形状小的收益不是「省寄存器」，而是**能在一个 warp 里塞下多个独立 tile**，")
print("而多条独立的工作流正是掩盖 MMA 延迟的手段。**形状小是特性，不是妥协。**")


# ===========================================================================
rule("5. 推导四：从每 SM 的 FMA/clock 推到峰值，再反解时钟")

print("峰值公式：P = SM 数 × (每 SM 的 FP16 FMA/clock) × 2 FLOP × 时钟频率")
print()
print("官方给出的是 P 与「每 SM 的 FMA/clock」两头，中间那个时钟官方不给。")
print("正推一遍、反解一遍，看两者对不对得上。")
print()
print("  %s%s%s%s%s%s"
      % (pad("卡", 8), pad("SM", 8), pad("FMA/clk/SM", 14), pad("官方 dense 峰值", 18),
         pad("用官方时钟正推", 18), "反解出的时钟"))
print("  " + "-" * 88)
CLOCKS = {"V100": 1.530, "A100": 1.410}     # 官方 spec 页给出的 boost 时钟
for gen in ("V100", "A100"):
    sm = S.SM_COUNT_BY_GEN[gen]
    fma = S.FP16_FMA_PER_CLK_PER_SM_DENSE[gen]
    peak = S.TC_TFLOPS_DENSE_SPARSE["FP16"][gen][0]
    clk = CLOCKS[gen]
    forward = sm * fma * 2 * clk * 1e9 / 1e12
    reclock = peak * 1e12 / (sm * fma * 2) / 1e9
    print("  %-8s %-8d %-14d %-18.1f %-18.1f %.3f GHz"
          % (gen, sm, fma, peak, forward, reclock))
v100_forward = 80 * 512 * 2 * 1.53e9 / 1e12
a100_forward = 108 * 1024 * 2 * 1.41e9 / 1e12
print()
print("V100：640 个 TC × 64 FMA/clk = %d FMA/clk/SM；80 × %d × 2 × 1.53 GHz = %.1f TFLOPS，"
      % (S.FP16_FMA_PER_CLK_PER_SM_DENSE["V100"], S.FP16_FMA_PER_CLK_PER_SM_DENSE["V100"],
         v100_forward))
print("      与官方 125 吻合；反解时钟 1.526 GHz 与 1.53 吻合到 %.1f%%。"
      % (100 * abs(1.526 - 1.53) / 1.53))
print("A100：108 × 1024 × 2 × 1.41 GHz = %.1f TFLOPS，与官方 312 吻合；" % a100_forward)
print("      反解时钟 1.411 GHz 与 1.41 吻合到 %.2f%%。**两条路径在这一代是自洽的。**"
      % (100 * abs(1.411 - 1.41) / 1.41))
print()
print("H100 就不自洽了。同一个 SM 数（%d）、同一份官方材料，三条路径给出三个时钟："
      % S.SM_COUNT_BY_GEN["H100"])
sm_h = S.SM_COUNT_BY_GEN["H100"]
h_paths = [
    ("FP32 向量峰值 67 TFLOPS", 67.0, sm_h * S.FP32_LANES_PER_SM["H100"] * 2),
    ("FP64 非 TC 峰值 34 TFLOPS", 34.0, sm_h * S.FP64_LANES_PER_SM_H100 * 2),
    ("FP16 TC dense 989.5 TFLOPS", 989.5, sm_h * S.FP16_FMA_PER_CLK_PER_SM_DENSE["H100"] * 2),
]
print()
print("  %s%s%s" % (pad("口径", 34), pad("每 SM 每时钟的操作数", 24), "反解出的时钟"))
print("  " + "-" * 76)
clks = []
for label, peak, perclk in h_paths:
    clk = peak * 1e12 / perclk / 1e9
    clks.append(clk)
    print("  %-34s %-24d %.3f GHz" % (label, perclk, clk))
print()
lo, hi = min(clks), max(clks)
print("三条路径的极差：%.3f GHz ~ %.3f GHz" % (lo, hi))
print("    相对差（除以最小值）= %.2f%%   除以最大值 = %.2f%%   除以中值 = %.2f%%"
      % (100 * (hi - lo) / lo, 100 * (hi - lo) / hi, 100 * (hi - lo) / ((hi + lo) / 2)))
print()
print("含义：**张量峰值与向量峰值不是同一口径下算出来的**。真实时钟只有一个，")
print("所以至少有一条路径的「每 SM 每时钟的操作数」写的是四舍五入后的整数，")
print("或者那两条通路并非每时钟都能打满。")
print()
print("⟹ **不要用「峰值 ÷ 时钟」反推每 SM 的 FMA 数，也不要反过来。**")
print("   本主题只用 FMA/clk/SM 做量级判断（§6），不用它算时钟。")
print()
print("把这件事倒过来看还有一条更有用的推论：若两者的时钟真的相同，")
print("则由 FP32 向量反推的时钟会要求 FP16 张量通路达到")
fma_implied = 989.5e12 / (sm_h * max(clks) * 1e9 * 2)
print("    989.5e12 / (%d × %.4f GHz × 2) = %.0f FMA/clk/SM" % (sm_h, max(clks), fma_implied))
print("而官方材料给出的形状推出的是 %d FMA/clk/SM。两者差 %.1f%%。"
      % (S.FP16_FMA_PER_CLK_PER_SM_DENSE["H100"],
         100 * (2048 - fma_implied) / fma_implied))
print("这两个数**不能同时成立**——这正是上面那段话的定量版本。")


# ===========================================================================
rule("6. 推导五：一条 MMA 占 SM 多少个时钟，等价于多少条 FFMA")

print("每 SM 的 FP16 FMA/clock 是「整个 SM 的」张量吞吐。用它除一条指令的 MAC 数：")
print()
print("  %s%s%s%s%s"
      % (pad("卡", 8), pad("指令", 18), pad("MAC/指令", 12), pad("FMA/clk/SM", 14),
         "占用 SM 时钟数"))
print("  " + "-" * 74)
jobs = [
    ("V100", "mma.m8n8k4",    8 * 8 * 4,   S.FP16_FMA_PER_CLK_PER_SM_DENSE["V100"]),
    ("A100", "mma.m16n8k16",  16 * 8 * 16, S.FP16_FMA_PER_CLK_PER_SM_DENSE["A100"]),
    ("A100", "mma.m16n8k8",   16 * 8 * 8,  S.FP16_FMA_PER_CLK_PER_SM_DENSE["A100"]),
    ("H100", "mma.m16n8k16",  16 * 8 * 16, S.FP16_FMA_PER_CLK_PER_SM_DENSE["H100"]),
]
for gen, ins, macs, fma in jobs:
    print("  %-6s %-16s %-10d %-12d %.1f" % (gen, ins, macs, fma, macs / fma))
print()
print("第二、三行是关键：**m16n8k8 与 m16n8k16 占用同样的时钟数**（都是 2 个），")
print("因为 k 减半的同时官方吞吐也减半。换句话说，张量单元**不是按 MAC 数摊销的，")
print("而是按指令的固定时序摊销的**——同一形状族里 k 减半，吞吐就减半。")
print("这就是「TF32 峰值（156）恰好是 FP16（312）的一半」在时钟一侧的同一件事。")
print()
print("再换成「同一条 MMA 换成 FFMA 要做多久」。一条 FFMA 是每 lane 一个乘加，")
print("一条 warp 指令覆盖 32 个 lane：")
m_macs = 16 * 8 * 16
ffma_per_lane = m_macs // S.WARP_SIZE
print("    m16n8k16 的 %d 个 MAC ÷ 32 lane = **每 lane %d 条 FFMA**"
      % (m_macs, ffma_per_lane))
print()
print("  %s%s%s%s%s"
      % (pad("卡", 8), pad("FP32 lane/SM", 16), pad("FFMA warp 指令/clk", 22),
         pad("64 条 FFMA 要几时钟", 24), "换 MMA 后"))
print("  " + "-" * 88)
for gen in ("A100", "H100"):
    lanes = S.FP32_LANES_PER_SM[gen]
    ffma_clk = lanes / S.WARP_SIZE
    cost = ffma_per_lane / ffma_clk
    mma_cost = m_macs / S.FP16_FMA_PER_CLK_PER_SM_DENSE[gen]
    print("  %-6s %-14d %-20.1f %-22.1f %.1f（%.1f 倍）"
          % (gen, lanes, ffma_clk, cost, mma_cost, cost / mma_cost))
print()
print("倍数与官方「张量峰值 ÷ FP32 向量峰值」应当一致，验算：")
for gen in ("A100", "H100"):
    r = S.TC_TFLOPS_DENSE_SPARSE["FP16"][gen][0] / S.TFLOPS_FP32_VECTOR_DENSE[gen]
    print("    %-5s %.1f / %.1f = **%.2f 倍**"
          % (gen, S.TC_TFLOPS_DENSE_SPARSE["FP16"][gen][0],
             S.TFLOPS_FP32_VECTOR_DENSE[gen], r))
print()
print("A100：形状侧算出 %.2f 倍，官方峰值侧算出 %.2f 倍，**逐位吻合**。"
      % ((64 / 2.0) / (m_macs / 1024), 312.0 / 19.5))
print("H100：形状侧算出 %.2f 倍，峰值侧算出 %.2f 倍，差 %.1f%%。"
      % ((64 / 4.0) / (m_macs / 2048), 989.5 / 67.0,
         100 * ((64 / 4.0) / (m_macs / 2048) / (989.5 / 67.0) - 1)))
print("差额的来源就是 §5 那两条反推时钟之比：%.4f / %.4f = %.4f（即 %.1f%%）。"
      % (989.5e12 / (sm_h * 2048 * 2) / 1e9, 67.0e12 / (sm_h * 128 * 2) / 1e9,
         (989.5e12 / (sm_h * 2048 * 2)) / (67.0e12 / (sm_h * 128 * 2)),
         100 * ((989.5e12 / (sm_h * 2048 * 2)) / (67.0e12 / (sm_h * 128 * 2)) - 1)))
print("**同一个「16 倍」在两个口径下差 %.1f%%——因为分子分母来自两份不同的官方数字。**"
      % (100 * ((64 / 4.0) / (m_macs / 2048) / (989.5 / 67.0) - 1)))
print()
print("最后看发射槽。每 SM 有 %d 个发射槽/时钟（%d 个 warp scheduler 各一条）："
      % (S.ISSUE_SLOTS_PER_CLK_PER_SM, S.ISSUE_SLOTS_PER_CLK_PER_SM))
for gen in ("A100", "H100"):
    sm = S.SM_COUNT_BY_GEN[gen]
    clk = {"A100": 1.41, "H100": 1.83}[gen]
    peak = S.TC_TFLOPS_DENSE_SPARSE["FP16"][gen][0]
    mma_per_clk = peak * 1e12 / (2 * m_macs) / sm / (clk * 1e9)
    print("    %-5s 要打满是 %.3f 条 MMA/clk/SM = 占 %.1f%% 的发射槽"
          % (gen, mma_per_clk, 100 * mma_per_clk / S.ISSUE_SLOTS_PER_CLK_PER_SM))
print("**发射槽不是瓶颈**——这条结论在 03 与 04 会各用一次，所以在这里先算出来。")


# ===========================================================================
rule("7. 推导六：单条 MMA 的强度，离机器的 roofline 拐点差多远")

print("机器的 roofline 拐点 I* = 峰值算力 ÷ 峰值带宽（FLOP/Byte）。")
print("（roofline 模型本身的定义在 10-foundations/performance-models/01，此处只做拐点迁移的计算。）")
print()
print("  %s%s%s%s" % (pad("卡", 8), pad("精度", 22), pad("峰值 dense", 16), "I* = P / 带宽"))
print("  " + "-" * 68)
knees = {}
for gen in ("V100", "A100", "H100", "B200"):
    bw = S.HBM_BW_GBps_BY_GEN[gen]
    for prec, label in (("FP32_VECTOR", "FP32 向量"), ("FP16", "FP16 TC")):
        if prec == "FP32_VECTOR":
            peak = S.TFLOPS_FP32_VECTOR_DENSE[gen]
        else:
            peak = S.TC_TFLOPS_DENSE_SPARSE["FP16"].get(gen, (None, None))[0]
        if peak is None:
            continue
        knee = peak * 1e12 / (bw * 1e9)
        knees[(gen, prec)] = knee
        print("  %-8s %-22s %-16.1f %.1f FLOP/Byte" % (gen, label, peak, knee))
print()
print("一条 m16n8k16 的字节强度是 §2 算出的 %.2f FLOP/Byte。两者的比值："
      % (flops / (elems * BYTES)))
print()
print("  %s%s%s%s" % (pad("卡", 8), pad("I*(FP16)", 16), pad("I*/I_tile", 18), "含义"))
print("  " + "-" * 70)
for gen in ("A100", "H100", "B200"):
    knee = knees[(gen, "FP16")]
    ratio = knee / (flops / (elems * BYTES))
    print("  %-6s %-14.1f %-16.1f 单条指令的操作数比拐点所需的复用少 %.1f 倍"
          % (gen, knee, ratio, ratio))
print()
print("结论：**单条 MMA 的字节强度比机器的 roofline 拐点低 1~2 个数量级。**")
print("⟹ 一条 MMA 的操作数**不可能**来自 HBM：要打到拐点，每个操作数字节平均要被复用")
print("%.1f 次（H100）到 %.1f 次（A100）。这就是分块（tiling）存在的全部理由——"
      % (knees[("H100", "FP16")] / (flops / (elems * BYTES)),
         knees[("A100", "FP16")] / (flops / (elems * BYTES))))
print("块级 tiling 提供的复用倍数是接下来的两篇（02 fragment 与 03 流水线）要算的东西。")


# ===========================================================================
rule("8. 失败模式的数字支撑")

print("【8.1】把标称 TFLOPS 当成 dense 用")
print()
print("  %s%s%s%s" % (pad("卡", 10), pad("精度", 16), pad("dense", 16), "sparse(2:4)"))
print("  " + "-" * 56)
for gen in ("A100", "H100", "B200"):
    for prec in ("FP16", "FP8"):
        p = S.TC_TFLOPS_DENSE_SPARSE[prec].get(gen)
        if p:
            print("  %-10s %-16s %-16.1f %s" % (gen, prec, p[0], pair(p)))
print()
print("NVIDIA 产品页写「H100 FP16 Tensor Core 1,979 TFLOPS」——那是**带 2:4 稀疏**的值。")
print("用 1979 做容量或时间估算，算出的时间会比 dense 实际能做到的乐观 %.0f%%。"
      % (100 * (1979.0 / 989.5 - 1)))
print()
print("【8.2】用「每 SM 的 tensor core 个数」衡量算力")
print()
print("  %s%s%s%s%s%s"
      % (pad("卡", 8), pad("TC/SM", 12), pad("TC/GPU", 14), pad("FP16 FMA/clk/SM", 18),
         pad("整卡 FP16 dense", 18), "相对 V100"))
print("  " + "-" * 86)
base = None
for gen in ("V100", "A100", "H100"):
    tc = S.TENSOR_CORES_PER_SM[gen]
    sm = S.SM_COUNT_BY_GEN[gen]
    fma = S.FP16_FMA_PER_CLK_PER_SM_DENSE[gen]
    peak = S.TC_TFLOPS_DENSE_SPARSE["FP16"][gen][0]
    if base is None:
        base = peak
    print("  %-8s %-12d %-14d %-18d %-18.1f %.2f×"
          % (gen, tc, tc * sm, fma, peak, peak / base))
tc_drop = 100 * (432 / 640.0 - 1)
peak_up = 100 * (312 / 125.0 - 1)
print()
print("从 V100 到 A100：**tensor core 个数从 640 降到 432（%.1f%%），" % tc_drop)
print("而整卡 FP16 峰值从 125 涨到 312（+%.1f%%）**。用「数 tensor core」这个指标，"
      % peak_up)
print("连符号都是反的。官方原文（A100 白皮书，逐字抄录）：")
print()
print('  "%s"' % S.A100_WHITEPAPER_TENSOR_CORE_QUOTE)
print()
print("两个因子要成对读：每 SM 的 TC 数减半（8 → 4），每个 TC 的 FMA/clk 翻两番（64 → 256），")
print("乘积从 %d 涨到 %d。**稳定可比的是 FMA/clock/SM，不是 TC 个数。**"
      % (S.FP16_FMA_PER_CLK_PER_SM_DENSE["V100"], S.FP16_FMA_PER_CLK_PER_SM_DENSE["A100"]))
print()
print("【8.3】用「峰值 ÷ 时钟」反推每 SM 的 FMA 数")
print()
print("§5 已算出 H100 三条路径给出的时钟是 %.3f / %.3f / %.3f GHz。"
      % (clks[0], clks[1], clks[2]))
print("把任一个当成「真的时钟」，都会让另一条通路的每 SM 操作数算错 %.1f%% 左右。"
      % (100 * (hi / lo - 1)))
print()
print("【8.4】把「16 位」当成一个精度")
print()
print("fp16 与 bf16 的**形状、寄存器数、FMA/clock 完全相同**（见 §3 的表与 §5 的 FMA 栏），")
print("差别只在数值格式（范围与精度）。所以「换个 16 位格式会不会更快」这个问题")
print("在本主题的范围内答案是**不会**——快慢由形状与指令决定，不由尾数位宽决定。")
print("（格式本身的差异见 10-foundations/floating-point/01-ieee754-format。）")

"""不变量：哪些数字十几年没动，以及它们为什么不动。

运行：python3 03_invariants.py
依赖：numpy（仅用于格式化；全部计算在本机 CPU 上完成，不需要 GPU）

口径声明：
  · 本程序做的是**算术推导**，没有在 GPU 上跑任何 kernel，也不产出任何实测数据。
  · 规格常量全部来自 specs.py（每条带 `# 来源：` 注释）；本程序只做代入与推导。
  · 第 7 节的访存延迟是**外部引用值**，本程序只转述、不产出、也无法复核。
"""

import numpy as np

import specs as S


def rule(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


rule("1. 真不变量：CC 3.5 → 12.x 一路没动（唯一的例外见下）")

print("  %-26s %-12s %s" % ("量", "值", "依据"))
print("  " + "-" * 92)
INVARIANTS = [
    ("warp 大小", str(S.WARP_SIZE),
     "CUDA 13.4 §5.1 表 与 CUDA 11.8 归档版 §5.1 表，**两代一致**"),
    ("每 block 最大线程数", str(S.MAX_THREADS_PER_BLOCK),
     "CUDA 13.4 §5.1（CC 2.0 起即为 1024）"),
    ("每线程最大寄存器数", str(S.MAX_REGISTERS_PER_THREAD),
     "CUDA 13.4 §5.1（CC 3.0 起即为 255）"),
    ("共享内存 bank 数", str(S.SHARED_MEMORY_BANKS),
     "CUDA 13.4 §5.1，各 CC 恒为 32"),
    ("每 bank 宽度", "%d B" % S.BANK_WIDTH_BYTES,
     "CUDA 13.4 §5.1（32-bit）"),
    ("每 SM 寄存器文件", "%.0f KB" % S.REGISTER_FILE_PER_SM_KB,
     "CUDA 13.4 §5.1（CC 3.5 起 65536 个 32-bit 寄存器）"),
]
for a, b, c in INVARIANTS:
    print("  %-26s %-12s %s" % (a, b, c))

print()
print("  **唯一的例外**：CC 3.7（GK210）的每 SM 寄存器文件是 %.0f KB，是标准值的 %.0f 倍。"
      % (S.REGISTER_FILE_PER_SM_KB_GK210, S.REGISTER_FILE_PER_SM_KB_GK210 / S.REGISTER_FILE_PER_SM_KB))
print("  这一格必须跟「256 KB 是不变量」这句话一起写出，否则它是个假陈述。")
print("  不过它是**孤例**：此后到 CC 12.x 再没有出现过 512 KB 的 SM。")
print()
print("  时间跨度：CC 3.5（Kepler GK110，2012）到 CC 12.x，**十几年**。")
print("  **而同样跨度的两个增长量是：")
print("    · 每 SM 的 FP32 向量峰值涨了 %.2f 倍（V100 15.7 → B200 75，只算数据中心线）；"
      % (S.B200_FP32_VECTOR_TFLOPS_DENSE / S.V100_FP32_VECTOR_TFLOPS_DENSE))
print("    · 每 SM 的 FP16 tensor 峰值涨了 %.1f 倍（V100 125 → B200 2250）。**"
      % (S.B200_FP16_TC_TFLOPS_DENSE / S.V100_FP16_TC_TFLOPS_DENSE))
print("  增长全部来自「每 lane 每周期做多少事」，不来自「能装多少东西」。")
print("  这条分工是本文与前两篇的连接点：前两篇讲的那些「变」都发生在这条线的")
print("  上面那一侧，而本节的这些数在下面那一侧一动不动。")


rule("2. 推导一：warp 大小为什么钉死在 32")

print("第一步：warp 大小不是孤立参数，它同时是四个东西。")
print()
print("  %-4s %-30s %s" % ("#", "它同时是", "改它会动到什么"))
print("  " + "-" * 92)
ROLES = [
    ("SIMT 的发射宽度", "一条指令覆盖 32 个 lane；改了就改了数据通路宽度"),
    ("寄存器分配的单位", "每 warp 占用 = warp 大小 × 每线程寄存器数"),
    ("调度器的记账单位", "active mask、PC、记分板位都是每 warp 一份"),
    ("共享内存的冲突单位", "32 个 bank 恰好让 stride-1 访问无冲突（见第 4 节）"),
]
for i, (a, b) in enumerate(ROLES, 1):
    print("  %-4d %-30s %s" % (i, a, b))

print()
print("第二步：四条独立约束都从不同方向给出 32，改一处就要同时改四处。")
print()
print("  约束一（发射宽度）。每 SM 有 4 个 warp scheduler，每周期各发射 1 条")
print("  warp 指令。若 warp 大小变成 64，同样的 4 条指令就要覆盖 256 个 lane，")
print("  而每 SM 的 FP32 数据通路只有 %d 个 lane——"
      % S.CHIP_BY_GEN["H100"]["fp32_lanes_per_sm"])
print("  要么把数据通路加宽到 256 lane，要么让一条指令占 2 个周期发射。")
print("  两条路的代价都落在同一处：**每 SM 每周期能发射的 warp 指令数**。")
print()
print("  约束二（寄存器读端口）。寄存器文件被 4 个处理块均分，每块服务本块的")
print("  32 个 lane，每周期读出**一个 warp 的全部操作数**。warp 变成 64 意味着")
print("  每块的读端口数翻倍——而 [20-gpu-architecture/memory-system/01 §1 第一步]")
print("  已经论证过：**寄存器堆做不大的主因正是端口数与面积**。")
print("  于是「warp 加倍」会反过来压小寄存器文件，直接打破 256 KB 那条不变量。")
print()
print("  约束三（分配粒度）。每线程寄存器按 8 个一档分配，所以每 warp 的分配")
print("  粒度是 8 × warp 大小。warp=32 时是 %d 个寄存器一档，"
      % (8 * S.WARP_SIZE))
print("  寄存器文件 %d ÷ %d = %d 个 warp 槽位——这个整除是硬件按整档记账的前提。"
      % (S.REGISTERS_PER_SM, 8 * S.WARP_SIZE, S.REGISTERS_PER_SM // (8 * S.WARP_SIZE)))
print()
print("  同一份预算的三种切法（第 2 节部件表引用的三个数）：")
print("    · 每 SM 总量：   %d 个 32-bit 寄存器 = %.0f KB"
      % (S.REGISTERS_PER_SM, S.REGISTERS_PER_SM * 4 / 1024))
print("    · 每个处理块：   %d ÷ 4 = %d 个 = %.0f KB"
      % (S.REGISTERS_PER_SM, S.REGISTERS_PER_SM // 4,
         S.REGISTERS_PER_SM // 4 * 4 / 1024))
print("    · 每个 warp 一档：8 × %d = %d 个；%d ÷ %d = %d 个槽位"
      % (S.WARP_SIZE, 8 * S.WARP_SIZE, S.REGISTERS_PER_SM, 8 * S.WARP_SIZE,
         S.REGISTERS_PER_SM // (8 * S.WARP_SIZE)))

print()
print("  约束四（整除恒等式）。反解一次，看 warp 大小与线程上限的关系：")
print()
print("      寄存器文件 ÷ warp 大小 = %d ÷ %d = %d"
      % (S.REGISTERS_PER_SM, S.WARP_SIZE, S.REGISTERS_PER_SM // S.WARP_SIZE))
print()
print("      而数据中心线的每 SM 最大常驻线程数**恰好也是 %d**。"
      % S.RESIDENT_LIMITS_BY_CC["9.0"][1])
print("      也就是说：`寄存器文件 = warp 大小 × 常驻线程上限`，在该产品线上、")
print("      二者都取满时成立。换个 warp 大小，这个等式要么不成立，")
print("      要么需要同时改寄存器文件容量——而后者是第 1 节那条不变量。")
print("      **但要注意这条恒等式用到了一个产品量**（数据中心线的 %d）。"
      % S.RESIDENT_LIMITS_BY_CC["9.0"][1])
print("      它解释的是「32 为什么与这条产品线配得这么整齐」，")
print("      不是「32 是 32 的唯一原因」。把产品线决策当论据用，")
print("      正是第 6 节那条坑的雏形——**只有三个量同时摆出来时，这个结论才成立。**")

print()
print("第三步：还有一条与性能无关、但有决定性的约束——**二进制兼容**。")
print()
print("  warp 大小是写进 ISA 与 ABI 的：")
print("    · `__ballot_sync` / `__activemask` 返回的是 32 位掩码；")
print("    · 所有 `*_sync` 原语的 mask 参数是 32 位；")
print("    · 每 block 线程数按完整 warp 切分，1024 = %d 个 warp 恰好整除；"
      % (S.MAX_THREADS_PER_BLOCK // S.WARP_SIZE))
print("    · 已经编译出的每一份 cubin 都按 32 编码。")
print()
print("  warp 变成 64，上面四项要同时改成 64 位掩码与新编码——")
print("  **这不是「新硬件加个特性」，是「让过去十几年的二进制全部作废」。**")
print("  这是 32 之所以能钉住的最强理由：它的代价不是性能，是生态。")


rule("3. 推导二：寄存器文件为什么是架构量，线程上限为什么不是")

print("这两个数看起来是一对：寄存器文件 %d 个、常驻线程上限 2048 个。"
      % S.REGISTERS_PER_SM)
print("但它们的性质完全不同。判别方法只有一个：**在规格变化时，哪个动了？**")
print()
print("  先看三个真实存在过的线程上限，各自反解出满额常驻时的每线程寄存器预算：")
print()
print("  %-10s %-14s %-22s %-12s %s"
      % ("CC", "线程上限", "寄存器 ÷ 线程上限", "是否整除", "按 8 一档可取"))
print("  " + "-" * 88)
for cc, label in (("7.5", "Turing"), ("8.9", "Ada"), ("9.0", "H100")):
    thr = S.RESIDENT_LIMITS_BY_CC[cc][1]
    raw = S.REGISTERS_PER_SM / thr
    exact = abs(raw - round(raw)) < 1e-9
    r_alloc = int(raw // 8) * 8
    used = r_alloc * thr
    print("  %-10s %-14d %-22.2f %-12s r = %d（用掉 %d / %d，余 %.2f%%）"
          % ("%s (%s)" % (cc, label), thr, raw,
             "是" if exact else "**否**", r_alloc, used, S.REGISTERS_PER_SM,
             (1 - used / S.REGISTERS_PER_SM) * 100))

print()
print("  三行的**分母变化**（1024 / 1536 / 2048），**分子始终是 %d**。"
      % S.REGISTERS_PER_SM)
print("  这就是判别结果：**线程上限在动，寄存器文件没动。**")
print()
print("  而且第三列暴露了一件事：")
print("    · 2048 与 1024 都是 %d 的因数，整除后得到 32 与 64，**零浪费**；"
      % S.REGISTERS_PER_SM)
print("    · 1536 **不是** %d 的因数，反解出 %.2f 这个非整数，"
      % (S.REGISTERS_PER_SM, S.REGISTERS_PER_SM / 1536))
print("      按 8 一档只能取到 r = 40，于是 65536 里有 4096 个寄存器用不上，")
print("      占整个寄存器文件的 %.2f%%——**一份预算里白白扔掉 %.1f 分之一。**"
      % ((1 - 40 * 1536 / S.REGISTERS_PER_SM) * 100,
         1 / (1 - 40 * 1536 / S.REGISTERS_PER_SM)))
print()
print("  换句话说：**2048 与 1024 是与 %d 共同设计过的线程上限，1536 不是。**"
      % S.REGISTERS_PER_SM)
print("  1536 是把一个产品决策（「这个 SKU 给多少线程」）叠在一个架构常量上，")
print("  叠的时候没有对齐——**这个不整齐本身就是它有别于寄存器文件的证据。**")

print()
print("再看另一头：如果寄存器文件才是那个可动的量，我们会看到什么？")
print("  我们会看到 1536 线程的产品配一份能让 1536 整除的寄存器文件")
print("  （例如 61440 或 73728 个寄存器），而 2048 线程的产品配另一份。")
print("  **实际没有发生——所有 CC 都报 %d。** 规格表上写的是同一个数，"
      % S.REGISTERS_PER_SM)
print("  一写就是十年。（唯一的例外是 CC 3.7 的 512 KB，见第 1 节。）")
print()
print("结论：")
print("  · **%d 个寄存器 / SM（256 KB）** —— 架构常量。"
      % S.REGISTERS_PER_SM)
print("  · **每 SM 最大常驻线程数** —— 产品线决策。数据中心 SXM 线恰好")
print("    稳定在 2048，但那不是「架构规定的」。**这是本文最需要记住的一条。**")


rule("4. 推导三：bank 数为什么也等于 32")

print("共享内存的 bank 编号规则是")
print()
print("      bank = (字节地址 ÷ %d) mod %d" % (S.BANK_WIDTH_BYTES, S.SHARED_MEMORY_BANKS))
print()
print("第一步：一个 warp 的 32 个 lane 在同一周期各发一个地址。硬件按 bank 分拣，")
print("  一个 bank 一个周期只服务一个地址。所以这次访问的周期数 =")
print("  **这个 warp 触及的不同 bank 数**。")
print()
print("第二步：取最典型的访问——stride-1，lane i 访问 `base + 4i`：")
print()
print("      bank(lane i) = ((base + 4i) ÷ 4) mod 32 = (base/4 + i) mod 32")
print()
print("  当 i 从 0 走到 31 时，`(base/4 + i) mod 32` 恰好取遍 0..31 各一次。")
print("  **32 个 lane 落在 32 个不同 bank 上，一个周期读完。**")
print()
print("第三步：反解——如果 bank 数不是 32 会怎样？")
print()
print("  %-14s %-26s %-16s %s" % ("bank 数", "stride-1 触及的 bank 数", "周期数", "代价"))
print("  " + "-" * 78)
for nbank in (8, 16, 32, 64):
    touched = min(nbank, S.WARP_SIZE)
    cycles = int(np.ceil(S.WARP_SIZE / touched))
    cost = "无冲突" if cycles == 1 else "%d 路冲突" % cycles
    note = "**加倍硬件，收益为零**" if nbank == 64 else ""
    print("  %-14d %-26d %-16d %s %s" % (nbank, touched, cycles, cost, note))

print()
print("  16 个 bank 会让**每一次** stride-1 访问都吃 2 个周期；")
print("  64 个 bank 则是一个字节都不多读、却要付双倍面积。")
print("  **32 是让规范访问恰好无冲突的最小值**——所以它也被钉住了。")
print()
print("第四步：把这条与 warp 大小放在一起，得到一个更整齐的恒等式：")
print()
print("      bank 数 × bank 宽度 = %d × %d B = %d B"
      % (S.SHARED_MEMORY_BANKS, S.BANK_WIDTH_BYTES,
         S.SHARED_MEMORY_BANKS * S.BANK_WIDTH_BYTES))
print("      warp 大小 × 每 lane 字宽 = %d × %d B = %d B"
      % (S.WARP_SIZE, S.BANK_WIDTH_BYTES, S.WARP_SIZE * S.BANK_WIDTH_BYTES))
print()
print("      **两边都是 %d 字节** —— 一次 stride-1 访问正好把一个 warp 的数据"
      % (S.SHARED_MEMORY_BANKS * S.BANK_WIDTH_BYTES))
print("      铺满全部 bank，一个字节不剩、一个周期不等。")
print()
print("  这就是「bank 数 = warp 大小」不是巧合同一个数字，而是**同一个约束的两种写法**：")
print("  32 是那条「一次访问服务一个 warp」的宽度，写在调度器上叫 warp 大小，")
print("  写在 SRAM 上叫 bank 数。**改其中一个，另一个必须跟着改。**")


rule("5. 不是不变量：每 SM 最大常驻 warp / 线程 / block 数")

print("  %-12s %-26s %-12s %-10s %s"
      % ("CC", "代表（产品线）", "常驻 warp", "常驻线程", "常驻 block"))
print("  " + "-" * 88)
NAMES = {"6.0": "P100（数据中心）", "7.0": "V100（数据中心）",
         "7.5": "Turing（**消费级**）", "8.0": "A100（数据中心）",
         "8.6": "GA10x（消费级）", "8.9": "Ada（消费级）",
         "9.0": "H100（数据中心）", "10.0": "B200（数据中心）",
         "11.0": "—", "12.x": "—"}
for cc, (w, t, b) in S.RESIDENT_LIMITS_BY_CC.items():
    mark = ""
    if cc in ("7.5", "8.9", "11.0", "12.x"):
        mark = "  ← **反例**"
    print("  %-12s %-26s %-12d %-10d %-10s%s"
          % (cc, NAMES.get(cc, "—"), w, t, "—" if b is None else str(b), mark))

print()
dc = [cc for cc in ("6.0", "7.0", "8.0", "9.0", "10.0")]
print("  数据中心 SXM 线（%s）五代的取值**完全一致**："
      % " / ".join(dc))
print("  %d warp / %d 线程 / %d block。"
      % tuple(S.RESIDENT_LIMITS_BY_CC["9.0"][:2] + (S.RESIDENT_LIMITS_BY_CC["9.0"][2],)))
print()
print("  **但这不代表它是架构不变量。** 两个反例在同一张官方规格表里：")
print("    · CC 7.5（Turing）：%d warp / %d 线程 / %d block —— 三条全不一样；"
      % S.RESIDENT_LIMITS_BY_CC["7.5"])
print("    · CC 8.9（Ada）与 CC 11.0 / 12.x：%d warp / %d 线程。"
      % S.RESIDENT_LIMITS_BY_CC["8.9"][:2])
print()
print("  所以正确的表述是：**「本产品线稳定在 2048 线程」，不是「架构规定 2048 线程」。**")
print("  这两句话的差别有实际后果，见第 6 节。")

print()
print("  三个数字里还有一个容易混进来的量。常驻 warp 数与常驻线程数只差一个因数：")
print("  %d × %d = %d —— 它俩是同一件事的两种写法（后者是前者的乘 32 倍）。"
      % (S.RESIDENT_LIMITS_BY_CC["7.5"][0], S.WARP_SIZE, S.RESIDENT_LIMITS_BY_CC["7.5"][1]))
print("  真正独立的是**每 SM 最大常驻 block 数**那一栏：它决定的是「并发粒度」")
print("  而不是「并发总量」，所以它可以单独变（7.5 的 16 对 9.0 的 32）。")
print()
print("  把这一节的结论与第 3 节拼起来，得到一张三档对照：")
print()
print("  %-34s %-14s %s" % ("量", "性质", "变化史"))
print("  " + "-" * 84)
for a, b, c in (
    ("每 SM 寄存器文件（%d 个）" % S.REGISTERS_PER_SM, "架构不变量",
     "CC 3.5 起 %d 个，十年未变（CC 3.7 是孤例）" % S.REGISTERS_PER_SM),
    ("warp 大小（%d）" % S.WARP_SIZE, "架构不变量", "两代官方规格表一致，从未变"),
    ("共享内存 bank 数（%d）" % S.SHARED_MEMORY_BANKS, "架构不变量", "各 CC 恒为 32"),
    ("每 SM 常驻线程数", "**产品线决策**",
     "1024 / 1536 / 2048 三档并存；数据中心线恰好一路是 2048"),
):
    print("  %-34s %-14s %s" % (a, b, c))


rule("6. 这条区别的实际后果：什么代码能活得久")

print("不变量与产品线决策的差别，落到代码上是两种命运。")
print()
print("  按不变量写的代码：")
print("    · warp 内规约、`__shfl_sync` 的 32 路、1024 线程/block 的 block 形状、")
print("      `#pragma unroll` 后的寄存器预算、按 32 bank 排布的共享内存 padding ——")
print("      这些写法在 CC 3.5 与 CC 12.x 上**语义与资源占用都不变**。")
print("    · 一份 2012 年为 Kepler 写的、只用这些量的 kernel，**今天仍然能编译、")
print("      能跑、资源账本一分不差**。这是「不变量」这三个字的兑现方式。")
print()
print("  按产品线决策写的代码：")
print("    · 「每 SM 挂 64 个 warp 就能占满」——在 Turing（%d warp）与 Ada（%d warp）"
      % (S.RESIDENT_LIMITS_BY_CC["7.5"][0], S.RESIDENT_LIMITS_BY_CC["8.9"][0]))
print("      上直接错 %.2f 倍与 %.2f 倍。"
      % (64 / S.RESIDENT_LIMITS_BY_CC["7.5"][0], 64 / S.RESIDENT_LIMITS_BY_CC["8.9"][0]))
print("    · 「按 2048 线程算占用率分母」——同一个 kernel 换到 Ada 上，")
print("      占用率的定义本身变了（分母从 2048 变 %d），**优化目标跟着挪**。"
      % S.RESIDENT_LIMITS_BY_CC["8.9"][1])
print("    · 「共享内存给到 228 KB 就对了」——228 KB 是 Hopper/Blackwell 的每 SM 上限，")
print("      不是架构规定的。")
print()
print("  判据很简单，问一句就够：")
print()
print("      **这个数在 CUDA Programming Guide 的哪一栏？**")
print()
print("    · 在「Compute Capabilities」表的**同一行跨 CC 不变**的那几栏 → 不变量；")
print("    · **同一列跨 CC 会变**的那几栏 → 产品线决策，必须查表，不能凭记忆。")
print()
print("  §1 那张表的每一行都是前者；§5 那张表的每一行都是后者。")
print("  **两张表在官方文档里挨着，这正是它们最容易混为一谈的原因。**")

print()
print("把 §4.2 与 §4.3 引用的几个数算出来。")
print()
print("  其一：block 形状对占用率的影响（线程上限是产品量，所以结论随 CC 变）。")
print()
print("  %-10s %-12s %-14s %-14s %s"
      % ("线程上限", "block 大小", "能放几个 block", "常驻线程", "占该 CC 上限"))
print("  " + "-" * 78)
for thr_cap, cc in ((2048, "数据中心线"), (1536, "Ada / 11.0 / 12.x")):
    for bs in (1024, 512):
        blocks = thr_cap // bs
        resident = blocks * bs
        pct = resident / thr_cap * 100
        tag = "  ← **剩下 512 装不下**" if resident < thr_cap else ""
        print("  %-10s %-12d %-14d %-14d %.1f%%%s"
              % ("%d (%s)" % (thr_cap, cc), bs, blocks, resident, pct, tag))
print()
print("  同一个 1024 线程的 block：在 %d 上限下是 %.1f%%，在 %d 上限下掉到 %.1f%%。"
      % (2048, 2048 / 2048 * 100, 1536, 1024 / 1536 * 100))
print("  **1024 整除 2048、不整除 1536 —— 这就是 §4.2 那条坑的全部算术。**")
print()
print("  其二：255 与 256 之间的那条缝。")
print()
print("  %-34s %-8s %s" % ("量", "值", "算式"))
print("  " + "-" * 74)
for label, val, expr in (
    ("每线程可用寄存器（按 256 线程反解）", 65536 // 256, "65536 ÷ 256"),
    ("每线程硬上限", S.MAX_REGISTERS_PER_THREAD, "8 位寄存器号字段 2^8 - 1"),
    ("256 线程实际能用掉的寄存器", 256 * S.MAX_REGISTERS_PER_THREAD,
     "256 × 255"),
    ("永远用不上的寄存器", S.REGISTERS_PER_SM - 256 * S.MAX_REGISTERS_PER_THREAD,
     "65536 − 65280"),
    ("用不上的比例", None,
     "%.2f%%" % ((S.REGISTERS_PER_SM - 256 * S.MAX_REGISTERS_PER_THREAD)
                 / S.REGISTERS_PER_SM * 100)),
):
    print("  %-34s %-8s %s" % (label, "—" if val is None else str(val), expr))
print()
print("  反解出来的是 %d，硬件只给到 %d——**差 1，而 1 个寄存器都拿不到。**"
      % (65536 // 256, S.MAX_REGISTERS_PER_THREAD))
print("  这正是 §4.3 那条坑：`每 SM 容量 ÷ 线程数` 与 `每线程上限` 是两个不同层级的约束，")
print("  不能用一个反解出另一个。")

print()
print("  其三：各代增长量（§3 那张对照表引用）。")
print()
for label, a, b, fmt in (("每 SM 的 FP32 向量峰值（V100 → B200）",
                          S.V100_FP32_VECTOR_TFLOPS_DENSE, S.B200_FP32_VECTOR_TFLOPS_DENSE,
                          "%.2f×"),
                         ("每 SM 的 FP16 tensor 峰值（V100 → B200）",
                          S.V100_FP16_TC_TFLOPS_DENSE, S.B200_FP16_TC_TFLOPS_DENSE,
                          "%.1f×")):
    print("  %-38s %7.1f → %-8.1f = %s" % (label, a, b, fmt % (b / a)))
print("  %-38s %7.0f → %-8.0f = %.2f×"
      % ("HBM 带宽（GB/s，V100 → B200）",
         S.V100_HBM_BW_GBps, S.B200_HBM_BW_GBps,
         S.B200_HBM_BW_GBps / S.V100_HBM_BW_GBps))
print("  %-38s %7s   %8s   %.2f×"
      % ("访存延迟（外部引用值，见第 7 节）", "460", "620",
         S.MEMORY_LATENCY_CYCLES_RANGE_EXTERNAL[1]
         / S.MEMORY_LATENCY_CYCLES_RANGE_EXTERNAL[0]))


rule("7. 访存延迟：跨代基本持平的另类不变量")

print("  > 本节的数字是**外部引用值**：外部来源在同一套方法下的跨代对照。")
print("  > **本主题的配套程序不产出它们**，也无法复核。Pascal / Volta / Turing")
print("  > 的数值来自另一套方法，只能比量级、不能逐位对齐。")
print()
print("  %-14s %-22s %s" % ("型号", "HBM 往返延迟（周期）", "备注"))
print("  " + "-" * 76)
for k, v in S.MEMORY_LATENCY_CYCLES_EXTERNAL.items():
    print("  %-14s %-22d %s" % (k, v, "同一套方法" if k != "Ada" else "另一产品线，同套方法"))
lo, hi = S.MEMORY_LATENCY_CYCLES_RANGE_EXTERNAL
print()
print("  量级区间：**%d ~ %d 个时钟周期**。最慢与最快相差 %.2f 倍——"
      % (lo, hi, hi / lo))
print("  与同期的算力（%.2f×）和带宽（%.2f×）涨幅相比，"
      % (S.B200_FP32_VECTOR_TFLOPS_DENSE / S.V100_FP32_VECTOR_TFLOPS_DENSE,
         S.CHIP_BY_GEN["B200"]["hbm_bw_GBps"] / S.CHIP_BY_GEN["V100"]["hbm_bw_GBps"]))
print("  **延迟这一项几乎原地不动。**")
print()
print("  为什么它不动：延迟由**路径长度**决定（见 [20-gpu-architecture/memory-system/01")
print("  §1 第一步]），而带宽由**并行通道数**决定。多堆几个 HBM stack 能涨带宽，")
print("  但每一条请求走的还是那条路。**这两个量由不同的物理量决定，所以可以独立变化。**")
print()
print("  这条「不动」的后果比任何一条不变量都直接：")
print("    · 隐藏延迟所需的**在途数据量 = 带宽 × 延迟**，带宽涨了 %.2f×，"
      % (S.CHIP_BY_GEN["B200"]["hbm_bw_GBps"] / S.CHIP_BY_GEN["V100"]["hbm_bw_GBps"]))
print("      延迟没动，于是这个需求也涨了 %.2f×。"
      % (S.CHIP_BY_GEN["B200"]["hbm_bw_GBps"] / S.CHIP_BY_GEN["V100"]["hbm_bw_GBps"]))
print("    · 而承载在途数据的地方是寄存器文件——**每 SM 十年一直是 256 KB。**")
print("    · 两个数相除就是常量占用率上的压力：**带宽涨了 8.56 倍，寄存器没涨。**")
print()
print("  这就是为什么「占用率」这个指标十几年一直是性能讨论的中心：")
print("  **延迟不降、带宽在涨、承载在途数据的存储不涨 —— 三者合起来只留一个出口：")
print("  让同样的常驻 warp 数承担更多的在途字节，也就是每个 warp 自己要更「能装」。**")


rule("8. 断言自检")

checks = [
    ("warp 大小两代官方规格表一致（%d）" % S.WARP_SIZE, S.WARP_SIZE == 32),
    ("每 block 最大线程数 %d 是 warp 大小的整数倍（%d 个 warp）"
     % (S.MAX_THREADS_PER_BLOCK, S.MAX_THREADS_PER_BLOCK // S.WARP_SIZE),
     S.MAX_THREADS_PER_BLOCK % S.WARP_SIZE == 0),
    ("每线程最大寄存器数 255 = 2^8 - 1（8 位寄存器号留一格）",
     S.MAX_REGISTERS_PER_THREAD == 2 ** 8 - 1),
    ("寄存器文件能被 warp 大小整除，且商恰等于数据中心线的常驻线程上限",
     S.REGISTERS_PER_SM // S.WARP_SIZE == S.RESIDENT_LIMITS_BY_CC["9.0"][1]),
    ("65536 ÷ 2048 与 65536 ÷ 1024 都是整数（32 与 64）",
     S.REGISTERS_PER_SM % 2048 == 0 and S.REGISTERS_PER_SM % 1024 == 0),
    ("65536 ÷ 1536 不是整数", S.REGISTERS_PER_SM % 1536 != 0),
    ("1536 线程上限下按 8 一档只能取 r=40，余 4096 个寄存器",
     int((S.REGISTERS_PER_SM / 1536) // 8) * 8 == 40
     and S.REGISTERS_PER_SM - 40 * 1536 == 4096),
    ("bank 数 × bank 宽度 == warp 大小 × 每 lane 字宽（都是 128 B）",
     S.SHARED_MEMORY_BANKS * S.BANK_WIDTH_BYTES == S.WARP_SIZE * 4
     and S.SHARED_MEMORY_BANKS * S.BANK_WIDTH_BYTES == 128),
    ("stride-1 访问在 32 bank 下一次读完（触及 32 个不同 bank）",
     min(S.SHARED_MEMORY_BANKS, S.WARP_SIZE) == S.WARP_SIZE),
    ("16 bank 会让 stride-1 变成 2 路冲突（周期数翻倍）",
     int(np.ceil(S.WARP_SIZE / min(16, S.WARP_SIZE))) == 2),
    ("数据中心线五代的常驻 warp / 线程 / block 完全一致",
     len({S.RESIDENT_LIMITS_BY_CC[cc] for cc in
          ("6.0", "7.0", "8.0", "9.0", "10.0")}) == 1),
    ("Turing (7.5) 的常驻 warp 只有数据中心线的一半",
     S.RESIDENT_LIMITS_BY_CC["7.5"][0] * 2 == S.RESIDENT_LIMITS_BY_CC["9.0"][0]),
    ("Ada (8.9) 与 11.0 / 12.x 都不等于数据中心线的 2048",
     S.RESIDENT_LIMITS_BY_CC["8.9"][1] != 2048
     and S.RESIDENT_LIMITS_BY_CC["12.x"][1] != 2048),
    ("Turing 的常驻线程数是 Ada 的 2/3", 1024 * 3 == 1536 * 2),
    ("CC 3.7 的寄存器文件是标准值的 2 倍（唯一的反例）",
     S.REGISTER_FILE_PER_SM_KB_GK210 == 2 * S.REGISTER_FILE_PER_SM_KB),
    ("每 SM 寄存器文件 %d × 4 B = %.0f KB"
     % (S.REGISTERS_PER_SM, S.REGISTERS_PER_SM * 4 / 1024),
     S.REGISTERS_PER_SM * 4 // 1024 == S.REGISTER_FILE_PER_SM_KB),
    ("同代内每线程寄存器分配粒度（8）× warp 大小 = 每 warp 256 个",
     8 * S.WARP_SIZE == 256),
    ("寄存器文件 ÷ 每 warp 分配粒度 = 256 个 warp 槽位",
     S.REGISTERS_PER_SM // (8 * S.WARP_SIZE) == 256),
    ("访存延迟区间宽度不到 1.35 倍（跨代基本持平）",
     S.MEMORY_LATENCY_CYCLES_RANGE_EXTERNAL[1]
     / S.MEMORY_LATENCY_CYCLES_RANGE_EXTERNAL[0] < 1.35),
    ("V100→B200 的带宽涨幅（%.2f×）远大于延迟区间宽度（%.2f×）"
     % (S.CHIP_BY_GEN["B200"]["hbm_bw_GBps"] / S.CHIP_BY_GEN["V100"]["hbm_bw_GBps"],
        S.MEMORY_LATENCY_CYCLES_RANGE_EXTERNAL[1]
        / S.MEMORY_LATENCY_CYCLES_RANGE_EXTERNAL[0]),
     (S.CHIP_BY_GEN["B200"]["hbm_bw_GBps"] / S.CHIP_BY_GEN["V100"]["hbm_bw_GBps"])
     / (S.MEMORY_LATENCY_CYCLES_RANGE_EXTERNAL[1]
        / S.MEMORY_LATENCY_CYCLES_RANGE_EXTERNAL[0]) > 5),
]
for desc, ok in checks:
    print("  [%s] %s" % ("通过" if ok else "**失败**", desc))
print()
print("全部 %d 项%s。" % (len(checks),
                     "通过" if all(o for _, o in checks) else "**未全部通过，见上**"))


rule("9. 本程序的边界")

print("  · 全部是算术推导，没有在 GPU 上跑任何 kernel，也没有编译过任何 CUDA 代码。")
print("  · 第 2 节的四条约束是**结构性论证**，不是官方文档的表述。它解释「为什么")
print("    改不动」，不构成「不可能改」的证明——真要改，改的是整个 ISA + ABI + 生态。")
print("  · 第 3 节的判别标准（「哪个动了」）只用了一个证据面：官方规格表在")
print("    CC 7.5 / 8.9 / 11.0 / 12.x 上给出的线程上限都不一样，而寄存器文件一样。")
print("    这是**归纳证据**，不是设计文档的直接引述。")
print("  · 第 5 节表里每 SM 最大常驻 block 数一栏在有来源的 CC 上给出，其余留空——")
print("    **不填一个猜的值。** 本节的结论只用 warp / 线程两栏。")
print("  · 第 7 节的访存延迟是外部引用值，本程序只转述。它的**口径与规格值不同源**，")
print("    正文已单独标注，不得与第 1 节的规格值混在一起引用。")
print("  · 「不变量」这个词在本程序里的定义是「官方规格表跨 CC 取值相同」，")
print("    不是「永远不会变」。第 1 节那个 CC 3.7 = 512 KB 的孤例正是提醒：")
print("    **本文能担保的只是「过去十年没变」。**")

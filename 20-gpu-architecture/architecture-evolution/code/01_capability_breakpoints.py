"""能力断点：同一份 kernel 为什么在这一代能编译、下一代不能。

运行：python3 01_capability_breakpoints.py
依赖：numpy（仅用于格式化；全部计算在本机 CPU 上完成，不需要 GPU）

口径声明：
  · 本程序做的是**算术推导**，没有在 GPU 上跑任何 kernel，也不产出任何实测数据。
  · 规格常量全部来自 specs.py（每条带 `# 来源：` 注释）；本程序只做代入与推导。
  · 规格里的例外与争议（P100 带宽两个来源、B200 SM 数反推、官方未披露项）
    在 specs.py 中已标注，本程序不重复裁决。
"""

import numpy as np

import specs as S


def rule(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def kr(k):
    """KB 的整数写法（避免 255.99 这种浮点尾巴）。"""
    return "%.0f KB" % k


rule("1. 能力断点表：特性在哪个 target 上第一次可用")

print("判定依据统一用 PTX ISA 9.4 的 Target ISA Notes——每条指令的")
print("「Target ISA Notes」小节写明它从哪个 sm_ 版本起可用。")
print()
print("  %-34s %-11s %-9s %s" % ("特性", "首次引入", "target", "改变了什么"))
print("  " + "-" * 100)
CHANGED = {
    "tensor_core_mma": ("矩阵乘变成一条指令，fragment 布局成为程序员的显式责任"),
    "independent_thread_scheduling": ("warp 内不再 lockstep，无同步的 warp 内规约从此是 UB"),
    "tf32": ("FP32 数据零改代码走 tensor 路径（代价是尾数 23→10 位）"),
    "bf16": ("与 FP32 同指数范围，消除 loss scaling"),
    "sparse_mma_2to4": ("稀疏成为显式一等公民（矩阵 + metadata 两个操作数）"),
    "cp_async": ("global→shared 不经寄存器，双缓冲流水线的前提"),
    "fp8": ("量化从离线步骤变成运行时机制"),
    "tma_cp_async_bulk": ("一个线程可发起整块多维搬运"),
    "wgmma": ("MMA 执行单元从 warp 变成 warpgroup（128 线程）且异步"),
    "cluster_dsmem": ("共享内存首次可跨 SM 直接寻址"),
    "tmem": ("累加器不再占寄存器（每 SM 256 KB 专用存储）"),
    "tcgen05": ("MMA 改为 CTA 级、单线程发起"),
    "fp4": ("必须配 block scaling，否则动态范围不够"),
}
for feat, (gen, target) in S.FEATURE_FIRST_TARGET.items():
    print("  %-34s %-11s %-9s %s" % (feat, gen, target, CHANGED[feat]))

print()
print("两条必须记清的修正：")
print("  · BF16 第一次出现在 **Ampere (sm_80)**，不是 Turing。Turing 的 Tensor Core")
print("    只加了 %s——把 BF16 归给 Turing 是本主题最常见的记错。"
      % " / ".join(S.TURING_TENSOR_ADDITIONS))
print("  · FP8 在**数据中心线**是 Hopper (sm_90)；消费级线（Ada，CC 8.9）更早具备。")
print("    凡说「某代 FA8」都要带上线。")
print()
print("还有一类容易漏看的：sm_90a / sm_100a 是 **\"accelerated\" 变体**。")
ACCEL = [f for f, (g, t) in S.FEATURE_FIRST_TARGET.items() if t.endswith("a")]
print("  属于加速变体的特性：%s" % "、".join(sorted(ACCEL)))
print("  它们只在**该精确架构**上可用，不像 sm_90 那样向前兼容同一家族。")
print("  后果是：为 sm_90a 编出的 cubin 跑不了 sm_90，反之亦然。")


rule("2. 推导一：从 cp.async 的有无，推出流水线深度与寄存器预算的耦合")

print("问题：Ampere 之前能不能做「访存与计算重叠」的双缓冲流水线？")
print("答案是**能，但深度只能在浅档位上停留**——下面把这句话算成一个数。")
print()
print("第一步：global→shared 的搬运有两条路径。")
print("  · 寄存器中转（sm_80 之前唯一的选择）：LDG 把数据写进寄存器，")
print("    再由 STS 从寄存器写进共享内存。数据在「LDG 完成」到「STS 发射」")
print("    之间必须**住在寄存器里**，而且这个寄存器的生命周期跨过整段全局延迟。")
print("  · cp.async（sm_80 起）：数据从 global 直达 shared，不经过寄存器回写。")
print("    中转的落脚点从寄存器文件变成共享内存。")
print()
print("第二步：给两条路径各建一个账本。取每线程每级的 tile 为 16 字节")
print("（一个 float4，也正是 cp.async 的规范粒度），它折合 4 个 32-bit 寄存器。")
print("记流水线深度为 D 级，计算本身需要的寄存器为 R0。")
print()
R0 = 40                    # 计算侧基础寄存器用量（经验区间的中值，见 01-sm-internals §1）
STAGE_REGS = 4             # 每级 16 B 的 tile = 4 个 32-bit 寄存器
TPB = 256                  # 每 block 线程数
WARPS_PER_BLOCK = TPB // 32
SMEM_KB = 164.0            # A100 SXM4 每 SM 的共享内存上限
CAP_WARPS = 64             # 数据中心线的每 SM 常驻 warp 上限

print("第三步：两栏算式。")
print("  寄存器路径：r_reg(D) = R0 + %d·D  → 按 8 个一档上取整 → "
      "warp 数 = floor(65536 / (32·r))" % STAGE_REGS)
print("  cp.async 路径：r_async = R0（中转不进寄存器）→ warp 数恒定在 "
      "floor(65536/(32·%d))" % R0)
print("     但它的共享内存账本是 D·%d B × %d 线程 = D·%d KB 每 block，"
      % (STAGE_REGS * 4, TPB, STAGE_REGS * 4 * TPB // 1024))
print("     而 A100 只有 %.0f KB 共享内存 / SM，于是"
      % SMEM_KB)
print("     warp 数 = min(寄存器上限, floor(%.0f / (D·%d)) × %d, %d)"
      % (SMEM_KB, STAGE_REGS * 4 * TPB // 1024, WARPS_PER_BLOCK, CAP_WARPS))
print()


def reg_warps(r):
    r_alloc = -(-r // 8) * 8                      # 8 个寄存器一档，向上取整
    return min(CAP_WARPS, S.REGISTERS_PER_SM // (32 * r_alloc)), r_alloc


def async_warps(D):
    r_alloc = -(-R0 // 8) * 8
    w_reg = min(CAP_WARPS, S.REGISTERS_PER_SM // (32 * r_alloc))
    stage_kb = D * STAGE_REGS * 4 * TPB / 1024.0
    w_smem = int(SMEM_KB // stage_kb) * WARPS_PER_BLOCK if stage_kb > 0 else CAP_WARPS
    return min(w_reg, w_smem), w_smem


print("  %-4s %-10s %-9s %-10s %-12s %-9s %-22s %s"
      % ("D", "r_reg", "上取整后", "寄存器路径", "cp.async r", "共享内存", "cp.async 路径", "谁赢"))
print("  " + "-" * 100)
rows = []
for D in (0, 1, 2, 3, 4, 6, 8, 12, 16):
    w_reg, r_alloc = reg_warps(R0 + STAGE_REGS * D)
    if D == 0:
        print("  %-4d %-10d %-9d %-10d %-12d %-9s %-22d %s"
              % (0, R0, r_alloc, w_reg, R0, "0 KB", w_reg, "持平（无流水线）"))
        rows.append((D, w_reg, w_reg))
        continue
    w_async, w_smem = async_warps(D)
    stage_kb = D * STAGE_REGS * 4 * TPB / 1024
    who = ("cp.async" if w_async > w_reg else
           ("寄存器" if w_async < w_reg else "持平"))
    print("  %-4d %-10d %-9d %-10d %-12d %-9s %-22d %s"
          % (D, R0 + STAGE_REGS * D, r_alloc, w_reg, R0,
             "%.0f KB" % stage_kb, w_async, who))
    rows.append((D, w_reg, w_async))

print()
print("看出两件事。")
print()
print("**其一，寄存器路径的深度是占用率的减函数。** 每加深一级流水线，就固定扣掉")
print("每线程 4 个寄存器，在 r 跨过 8 的档位边界时整档掉 warp 数。D=0 → D=16，")
print("常驻 warp 从 %d 掉到 %d（%.2f 倍）。而**占用率正是用来隐藏延迟的那个资源**——"
      % (rows[0][1], rows[-1][1], rows[0][1] / rows[-1][1]))
print("于是出现一个循环依赖：**加深流水线的目的是藏住延迟，可它消耗的恰好是藏延迟")
print("用的那份预算。** 这是「Ampere 之前能重叠、但只能在浅流水线上重叠」的量化形态。")
print()
print("**其二，cp.async 把这条耦合搬到了共享内存，而不是取消它。** 在 D ≤ 8 时它")
print("白赚：寄存器侧恒定 %d 个 warp，共享内存还没成为约束。但 D 继续加深，"
      % async_warps(1)[0])
print("共享内存接替寄存器成为新的瓶颈——D=16 时共享内存只允许 %d 个 warp，"
      % async_warps(16)[0])
print("**反而低于寄存器路径的 %d 个**。" % reg_warps(R0 + STAGE_REGS * 16)[0])
cross = None
for i in range(len(rows) - 1):
    if (rows[i][1] - rows[i][2]) * (rows[i + 1][1] - rows[i + 1][2]) < 0:
        cross = (rows[i][0], rows[i + 1][0])
print("两条曲线在 D ∈ [%s, %s] 之间交叉。" % cross if cross else "（无交叉）")
print()
print("所以 cp.async 兑现的不是「从不能重叠变成能重叠」，而是")
print("**「把流水线深度的代价从一个预算挪到另一个预算」**——从每线程的寄存器")
print("（A100 每 SM 65536 个，但按线程切分、直接决定 warp 数）挪到每 block 的")
print("共享内存（每 SM %.0f KB，按 block 切分）。**两个预算都被解耦了，但没有被取消。**"
      % SMEM_KB)


rule("3. 推导二：从 Independent Thread Scheduling 推出 __syncwarp 的必要性")

print("问题：为什么 7.0 之前的「无同步 warp 内规约」能用，之后必须加 __syncwarp()？")
print()
print("第一步：7.0 之前，一个 warp 的 32 个 lane 共享**一个 PC**。")
print("  硬件按 warp 取指、按 warp 发射，32 个 lane 在同一周期执行同一条指令的")
print("  各自数据。只要没有分支发散，它们必然在同一时刻处于同一个程序点。")
print()
print("第二步：这个「必然」正是无同步规约所依赖的前提。")
print("  形如 `v += __shfl_down(v, 1)` 的规约，每一步都在读**邻居 lane 的寄存器**。")
print("  在共享 PC 的模型下，邻居一定已经把上一步写完了——因为大家同步推进。")
print("  于是这段代码在 7.0 之前是**可用的**，虽然它从未被 ISA 明文保证过。")
print()
print("第三步：Volta 给每个线程配了**独立的 PC 与调用栈**。")
print("  硬件仍然按 warp 发射，但一条发射出去的指令携带一个**任意的 active mask**——")
print("  32 个 lane 里可以只有一部分处在同一个程序点。")
print("  后果直接命中第二步：邻居 lane 可能**还没走到**那条写寄存器的指令，")
print("  也可能**已经越过去了**。读到的值不再有定义。")
print()
print("第四步：因此程序必须显式声明「哪些 lane 要在这里会合」。")
print("  这就是 __syncwarp(mask)：它在硬件里插入一个**重汇聚点**，")
print("  强制 mask 指定的 lane 走到同一程序点再继续。mask 参数不是装饰。")
print()
print("第五步：证据在 API 的演化里——**同步版本是后来补上的，而且带 mask**。")
print()
print("  %-28s %-32s %s" % ("7.0 之前的形式", "Volta 起的形式", "新增的那个参数"))
print("  " + "-" * 92)
for old, new, why in [
    ("__shfl(v, srcLane)", "__shfl_sync(mask, v, srcLane)", "mask：哪些 lane 参与"),
    ("__ballot(pred)", "__ballot_sync(mask, pred)", "mask：哪些 lane 参与"),
    ("__any(pred) / __all(pred)", "__any_sync / __all_sync(mask, pred)", "mask：哪些 lane 参与"),
    ("（无对应物）", "__syncwarp(mask)", "显式的会合点"),
    ("（无对应物）", "__activemask()", "读出当前活跃的 lane 集合"),
]:
    print("  %-28s %-32s %s" % (old, new, why))
print()
print("  注意最后两行：**7.0 之前根本没有查询活跃 mask 的手段**，")
print("  因为那时「活跃集合」是一个不需要问的问题。")
print("  这条 API 缺口本身就是断点的直接证据：**新硬件引入了一个旧程序无法表达的状态。**")
print()
print("结论：这不是「性能变了」，是「同一段源码在 7.0 之前有定义、之后无定义」。")
print("这正是本文 §0 那句话的技术内容——**改的是能不能表达，不是快不快。**")


rule("4. 执行映射：每个断点落在 SM 的哪个部件上")

print("  %-20s %-34s %s" % ("断点", "落在哪个部件", "代价/数量级"))
print("  " + "-" * 104)
# 来源：NVIDIA Hopper Architecture In-Depth 官方博客 / CUDA C++ Programming Guide 13.4
# 「Thread Block Clusters」章节——可移植簇大小上限 8 个 block，非可移植 16。
CLUSTER_MAX_BLOCKS_PORTABLE = 8
CLUSTER_MAX_BLOCKS_NONPORTABLE = 16
PARTS = [
    ("Tensor Core / mma", "每 SM 的 Tensor Core 单元 + 寄存器文件里的 fragment",
     "Volta 8 个/SM；Ampere 起 4 个/SM；B200 官方未披露"),
    ("Independent Thread Scheduling", "每线程 PC + 重汇聚栈 + 发射指令的 active mask 字段",
     "active mask 是每条发射指令携带的 32 位状态"),
    ("TF32 / BF16", "Tensor Core 的输入端口模式（不是新部件）",
     "同一端口，只换解码；精度 23→10 位 / 指数对齐 FP32"),
    ("mma.sp（2:4 稀疏）", "Tensor Core 的操作数取回路径 + metadata 操作数",
     "peak 翻倍；代价是权重需重排 + 一份 metadata"),
    ("cp.async", "LSU + 共享内存写端口（绕过寄存器回写）+ 异步组计数器",
     "搬运指令数减半；中转从每线程寄存器改到每 block 共享内存"),
    ("TMA / cp.async.bulk", "每 SM 的 TMA 单元 + 共享内存端口；描述符住在 global",
     "一个线程发起整块多维搬运，寄存器占用 ≈ 常数"),
    ("wgmma", "Tensor Core 的发射路径（warp → warpgroup 128 线程）+ 异步完成",
     "累加器仍在寄存器：m64n256 每线程占 128 个"),
    ("Cluster / DSMEM", "GPC 内的 SM 间互联 + 共享内存地址映射（mapa）",
     "可移植簇大小上限 %d 个 block；非可移植 %d"
     % (CLUSTER_MAX_BLOCKS_PORTABLE, CLUSTER_MAX_BLOCKS_NONPORTABLE)),
    ("TMEM / tcgen05", "TMEM（512 列 × 128 行 × 32-bit）+ 单线程发射路径",
     "TMEM 每 SM %s，与寄存器文件同容量" % kr(S.REGISTER_FILE_PER_SM_KB)),
]
for a, b, c in PARTS:
    print("  %-32s %-46s %s" % (a, b, c))

TMEM_BYTES = 512 * 128 * 4
print()
print("  核对 TMEM 那一格：512 × 128 × 4 B = %d B = %.0f KB，"
      % (TMEM_BYTES, TMEM_BYTES / 1024))
print("  与每 SM 的寄存器文件 %s **恰好同容量**。这不是巧合——它回答的正是"
      % kr(S.REGISTER_FILE_PER_SM_KB))
print("  「累加器挤爆寄存器预算」那个问题：**不加宽寄存器，另开一份。**")
print()
print("  再核对发射槽那一格：每 SM 每周期最多 4 条 warp 指令（4 个 warp scheduler），")
print("  H100 有 %d 个 SM ⇒ 全卡每周期 %d 个发射槽。**断点改变的是这些槽能装什么**："
      % (S.H100_SM_COUNT, S.H100_SM_COUNT * 4))
print("  一条 mma 顶掉几十条 FFMA，这是「同样 4 个槽、算力差一个数量级」的来源。")


rule("5. 逐代能力年表")

feats = list(S.FEATURE_FIRST_TARGET.keys())
gens = S.GENERATIONS
print("  %-32s %s" % ("特性", "".join("%-12s" % g for g in gens)))
print("  " + "-" * 92)
for f in feats:
    _, target = S.FEATURE_FIRST_TARGET[f]
    cells = []
    for g in gens:
        chip = S.CHIP_BY_GEN[g]
        if g == "P100":
            cells.append("有" if f == "tensor_core_mma" and chip["tensor_cores_per_sm"] else "—")
        elif f == "tensor_core_mma":
            cells.append("有")
        else:
            # 用「首次引入代际」的位置作为分界
            first = S.FEATURE_FIRST_TARGET[f][0]
            cells.append("有" if gens.index(g) >= gens.index(
                {"Volta": "V100", "Ampere": "A100",
                 "Hopper": "H100", "Blackwell": "B200"}[first]) else "—")
    print("  %-32s %s" % (f, "".join("%-12s" % c for c in cells)))

print()
print("  逐年读这张表可以看出断点的**节奏**（年份取自 specs.py）：")
prev = None
for g in gens:
    fs = [f for f in feats
          if S.FEATURE_FIRST_TARGET[f][0] ==
          {"P100": "Pascal", "V100": "Volta", "A100": "Ampere",
           "H100": "Hopper", "B200": "Blackwell"}[g]]
    if fs:
        print("    %s（%d）新增 %d 项：%s"
              % (g, S.YEAR_BY_GEN[g], len(fs), "、".join(fs)))
    prev = g
print()
print("  Volta 一次引入 2 项、Ampere 4 项、Hopper 4 项、Blackwell 3 项。")
print("  **断点不是均匀分布的**：Volta 是把 GPU 从「一堆标量核」变成")
print("  「一台矩阵机」的那一代，Ampere 是把「流水线」变成可编程的那一代，")
print("  Hopper 是把「SM 之间的墙」拆掉的那一代。")
print("  P100 一栏全空不是遗漏：**Pascal 没有任何一项本文定义的断点**——")
print("  它是这套能力阶梯的零级。")


rule("6. 断言自检")

checks = [
    ("Tensor Core 与 Independent Thread Scheduling 同在 Volta (sm_70) 引入",
     S.FEATURE_FIRST_TARGET["tensor_core_mma"][1] == "sm_70"
     and S.FEATURE_FIRST_TARGET["independent_thread_scheduling"][1] == "sm_70"),
    ("BF16 的首次引入是 sm_80，不是 Turing 的 sm_75",
     S.FEATURE_FIRST_TARGET["bf16"][1] == "sm_80"),
    ("Turing 的 Tensor Core 新增项是 INT8/INT4，不含 BF16",
     "BF16" not in S.TURING_TENSOR_ADDITIONS),
    ("Pascal 的每 SM Tensor Core 数为 0",
     S.P100_TENSOR_CORES_PER_SM == 0),
    ("wgmma 与 tcgen05 都属于 \"a\"（加速）变体",
     S.FEATURE_FIRST_TARGET["wgmma"][1].endswith("a")
     and S.FEATURE_FIRST_TARGET["tcgen05"][1].endswith("a")),
    ("cp.async 在 Ampere (sm_80) 引入，早于 TMA (sm_90)",
     int(S.FEATURE_FIRST_TARGET["cp_async"][1][-2:])
     < int(S.FEATURE_FIRST_TARGET["tma_cp_async_bulk"][1][-2:])),
    ("TMEM 容量与每 SM 寄存器文件相等（都是 %s）" % kr(S.REGISTER_FILE_PER_SM_KB),
     TMEM_BYTES == S.REGISTER_FILE_PER_SM_KB * 1024),
    ("R0=40 时寄存器路径的常驻 warp 数为 51（与 01-sm-internals 的算例一致）",
     reg_warps(40)[0] == 51),
    ("寄存器路径的常驻 warp 数随深度 D 单调不增",
     all(rows[i][1] >= rows[i + 1][1] for i in range(len(rows) - 1))),
    ("cp.async 路径在共享内存成为约束前恒定，之后随 D 下降",
     async_warps(1)[0] == async_warps(4)[0] > async_warps(16)[0]),
    ("两条路径在 D ∈ [12, 16] 之间交叉（浅流水线上 cp.async 白赚，深流水线上被共享内存接替）",
     cross is not None and cross == (12, 16)),
    ("cp.async 路径在 D=16 时反而不如寄存器路径（共享内存成为新瓶颈）",
     rows[-1][2] < rows[-1][1]),
    ("H100 全卡每周期发射槽为 132 × 4 = 528",
     S.H100_SM_COUNT * 4 == 528),
    ("可移植簇上限 8 严格小于非可移植上限 16",
     CLUSTER_MAX_BLOCKS_PORTABLE == 8 and CLUSTER_MAX_BLOCKS_NONPORTABLE == 16),
]
for desc, ok in checks:
    print("  [%s] %s" % ("通过" if ok else "**失败**", desc))
print()
print("全部 %d 项%s。" % (len(checks),
                     "通过" if all(o for _, o in checks) else "**未全部通过，见上**"))


rule("7. 本程序的边界")

print("  · 全部是算术推导，没有在 GPU 上跑任何 kernel，也没有编译过任何 CUDA 代码。")
print("    本程序**不产出**「某 kernel 在某代上快多少」这类数字——那要跑过才知道。")
print("  · 第 2 节的账本是**单一变量模型**：只用每线程寄存器数与每 block 共享内存")
print("    两个约束，忽略了每 SM 最大常驻 block 数（32）与其它约束。因此表中的")
print("    warp 数是**上界**，真实 kernel 还要再取一次较小值。")
print("  · R0 = 40 是经验区间的中值，不是规格。改用别的 R0，第 2 节的绝对值会平移，")
print("    两条曲线的交叉位置也会跟着挪——**「深度与占用率耦合」这个结论不随 R0 变，")
print("    但「交叉在 D≈13」这个具体位置会变。**")
print("  · 第 4 节的部件表是结构性的（谁被谁取代），不含吞吐数字。")
print("  · 断点的 target 判定依据是 PTX ISA 的 Target ISA Notes，不是「哪一代产品")
print("    先搭载」。两者偶尔会差一代（例如某个 target 先出现在工程样片上）。")

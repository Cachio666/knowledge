"""40-kernels/reduce-scan 主题的共享事实源。

本模块**只放从官方来源抄来的规格常量与语义事实**，不放任何推导值。
代价模型的全部推导在 `verify.py` 里做，本模块不重复。

三条硬性约定（见 CONTRIBUTING §2 与 40-kernels/README.md）：

  1. **没有 `# 来源：` 注释的常量不许写进本模块。**
  2. **常量名必须编码口径**，使误用可见：带宽一律带 `_GBps`，线程数一律带 `_PER_SM`，
     语义事实一律用一个**断言式短句**做名字（`SHFL_DOWN_..._KEEPS_OWN_VALUE`），
     使「反过来说」在命名上就说不通。
  3. **找不到可靠来源的项不写进来**，改记为 `UNSOURCED`。

**常量与来源沿用 `20-gpu-architecture/*/code/specs.py` 已有的口径**：
`code/` 必须自包含所以重新抄一份，但同名的常量同值同来源，不得重新发明。
（本模块的 `WARP_SIZE` / `REGISTERS_PER_SM` / `SMEM_PER_SM_MAX_KB` /
`SMEM_DELIVERY_BYTES_PER_CLK` / `SMEM_LATENCY_CYCLES_RANGE` /
`HBM_LATENCY_CYCLES_RANGE` / `ISSUE_SLOTS_PER_CLK_PER_SM` / `MAX_THREADS_PER_SM` /
`MAX_BLOCKS_PER_SM` / `FLOAT32_BYTES` / `SM_COUNT_BY_GEN` / `HBM_BW_GBps_BY_GEN` /
`FP32_VECTOR_TFLOPS_DENSE` 与 `40-kernels/gemm/code/specs.py` 逐字一致。）

来源等级沿用本仓库约定：**A** = NVIDIA 官方规范/白皮书/datasheet；
**B** = NVIDIA 官方代码或作者为 NVIDIA 的论文；**C** = 第三方逆向/学术。
**只有 A 级可以写进本模块的来源注释。**

**本主题用到的规格分三类**：硬件资源（§0）、指令语义（§1）、算法参数（**不在这里**）。
归约/扫描的**算法参数**（每线程处理几个元素、分几级、`width` 取多少）
不是规格，它们住在 `verify.py`。
"""

# ===========================================================================
# 0. 基础口径与每 SM 资源
# ===========================================================================

WARP_SIZE = 32
# 来源：CUDA C++ Programming Guide 13.4 §5.1（各 compute capability 的
#       "warp size" 一栏恒为 32）；与本仓库 20-gpu-architecture/execution-model/01 一致

MAX_WARPS_PER_SM = 64
# 来源：CUDA C++ Programming Guide 13.4 §5.1 表（CC 7.0/8.0/9.0/10.0 的
#       "Maximum number of resident warps per SM" = 64）
# 口径：64 × 32 = 2048，与 MAX_THREADS_PER_SM 是**同一条约束的两种写法**。
#       本主题讲多级归约的级数时要数「每 SM 最多几个 warp 的局部和」，
#       所以显式写出这个数，而不是每次从 2048/32 反推。

REGISTERS_PER_SM = 65536
# 来源：CUDA C++ Programming Guide 13.4 §5.1（CC 3.5 及以后各代
#       "Maximum number of 32-bit registers per SM" = 65536）
REGISTER_ALLOC_GRANULARITY = 8
# 来源：CUDA C++ Programming Guide 13.4 §5.1 的寄存器分配粒度说明
#       （每线程寄存器数按 8 个一档向上取整）；与 execution-model/01 §1 同源

SMEM_PER_SM_MAX_KB = 228
# 来源：NVIDIA H100 Tensor Core GPU Architecture Whitepaper（每 SM 统一池
#       256 KB，其中至多 228 KB 可切给 shared memory）
SMEM_DELIVERY_BYTES_PER_CLK = 128
# 来源：CUDA C++ Programming Guide 13.4 §5.1（32 个 bank × 4 字节，
#       无冲突时每周期交付 128 字节）
SMEM_LATENCY_CYCLES_RANGE = (20, 30)
# 来源：本仓库 20-gpu-architecture/memory-system/01-hierarchy-overview §1（20~30 周期）
HBM_LATENCY_CYCLES_RANGE = (400, 800)
# 来源：同上（一次片外访存的往返延迟）
ISSUE_SLOTS_PER_CLK_PER_SM = 4
# 来源：NVIDIA Hopper Architecture In-Depth（每 SM 4 个 warp scheduler，各每周期
#       发射一条指令）；Pascal~Blackwell 数据中心线均为 4

MAX_THREADS_PER_SM = 2048
MAX_BLOCKS_PER_SM = 32
# 来源：CUDA C++ Programming Guide 13.4 §5.1 表（CC 9.0 / 10.0 的
#       "Maximum number of resident threads per SM" = 2048 与
#       "Maximum number of resident blocks per SM" = 32）；
#       与 20-gpu-architecture/architecture-evolution/code/specs.py 的
#       RESIDENT_LIMITS_BY_CC[("9.0"|"10.0")] 一致

FLOAT32_BYTES = 4
FLOAT16_BYTES = 2
# 来源：IEEE 754 binary32 / binary16 的位宽（32 / 16 bit）

MAX_GRID_DIM = {"x": 2147483647, "y": 65535, "z": 65535}
# 来源：CUDA C++ Programming Guide 13.4 §5.1 表（"Maximum x-dimension of a grid
#       of thread blocks" = 2^31−1；y 与 z 维均为 65535）
# 口径：本主题 04 篇用它说明「把第二级归约塞进 gridDim.y 是一个**有上限**的选择」。
#       ⚠ 注意 x 维与 y/z 维**差 5 个数量级**，把大维度放错轴会直接被 launch 拒绝。


# ===========================================================================
# 1. 指令语义（**这些是事实，不是性能数字**）
# ===========================================================================
#
# 本主题的主线是「归约/扫描的结果由什么决定」，其中一半的坑来自指令语义
# 而不是性能。所以这里显式记录语义事实，让 `verify.py` 能针对它们建模。

SHUFFLE_LEGAL_WIDTHS = (2, 4, 8, 16, 32)
# 来源：CUDA C++ Programming Guide 13.4 §"Warp Shuffle Functions"
#       （`width` 必须是 2 的幂且不大于 warpSize）；同见 CUDA Math API 的
#       `__shfl_down_sync` 条目（"width must be a power of 2 ... results are
#       undefined otherwise"）
# 口径：非 2 的幂或 > 32 是**未定义**，不是「取模」。

SHFL_NEVER_CROSSES_WARP = True
# 来源：同上（`__shfl_sync` 系列「permit exchanging of a variable between
#       threads within a warp」——交换范围是一个 warp 的 32 个 lane）
# 口径：**与 `width` 无关**。把 `width` 写错不会导致「跨 warp 读」，
#       只会改变**同一个 warp 内**的段划分。这句话本主题 02 篇要反复用到。

SHFL_DOWN_OUT_OF_RANGE_KEEPS_OWN_VALUE = True
# 来源：同上（"If the source lane ID is out of range or the source thread has
#       exited, the calling thread's own var is returned"；且 `__shfl_down_sync`
#       的源 lane **不按 width 回绕**，于是段内最高的 `delta` 条 lane 拿到自己）
# 口径：这是本主题 02 篇那条失败模式（"`width` 用错 → 结果偏大"）的**机制**：
#       `v += __shfl_down_sync(m, v, o, w)` 在越界时变成 `v += v`，
#       于是这条 lane 的值被**重复计入**。⚠ 与 `__shfl_sync` 相反——
#       后者越界时按 `srcLane % width` **回绕**。

SHFL_XOR_SOURCE_STAYS_IN_SEGMENT = True
# 来源：同上（`__shfl_xor_sync` 的源 lane 是 `laneid ^ laneMask`；
#       `laneMask < width` 时异或的结果必然仍落在同一段内）
# 口径：**`xor` 变体没有「越界保留自身值」这一说**（只要 `laneMask < width`）。
#       这是它与 `down` 变体的关键差别，也是蝴蝶型归约更抗误用的原因。

BAR_SYNC_ARRIVAL_COUNT_IS_MULTIPLE_OF_WARP = True
BAR_SYNC_EXECUTES_AS_IF_ALL_LANES_ACTIVE = True
# 来源：PTX ISA §"Parallel Synchronization and Communication Instructions: bar"
#       ——"Barriers are executed on a per-warp basis as if all the threads in a
#       warp are active"；`bar.sync a{, b}` 的 `b` **必须是 warp size 的整数倍**，
#       「到达计数按 warp 计，而不是按活跃 lane 数计」
# 口径：**这是 PTX 层面的记账号径，Volta 之前也是硬件实际行为**（一条 lane 到达
#       即整个 warp 记到）。它正是「`__syncthreads()` 不能强制发散 lane 收敛」
#       的原因，**同时也正是「warp 是同步单位」那个过时假设的来源**——
#       两条话其实说的是同一件事，区别只在把它当「现在的保证」还是「过去的实现」。
#       本主题 01 篇 §4 按此口径写。

INDEPENDENT_THREAD_SCHEDULING_FIRST_CC = "7.0"
# 来源：CUDA C++ Programming Guide 13.4 §"Independent Thread Scheduling"
#       （Volta / CC 7.0 起每个线程有自己的 PC 与调用栈；CUDA 9 相应引入
#        `__syncwarp()` 与各 `_sync` 后缀的 warp 原语）
# 口径：**它同时收紧了两件事**——(a) 锁步不再是保证，warp 内 lane 间传值必须
#       显式同步；(b) `__syncthreads()` / `bar.sync` 改为**按线程**强制，
#       部分 lane 到达的屏障**不会**释放。所以「一条 lane 到达即整个 warp 记到」
#       是 **Volta 之前**的记账号径，不能在讲 Volta 及以后时当作现状。

FMAXF_IS_IEEE_MAXNUM = True
# 来源：CUDA C++ Programming Guide 附录 "Mathematical Functions" 与 CUDA Math API
#       的 `fmaxf` 条目（"Returns the maximum numeric value of the two
#       arguments"；**一个参数是 NaN 时返回另一个**）
# 口径：这对应 **IEEE 754-2008 的 maxNum**（忽略 NaN），
#       而不是 **IEEE 754-2019 的 maximum**（传播 NaN）。
#       C 的 `fmaxf` 与 numpy 的 `np.fmax` 是前者，`np.maximum` / `std::max` 是后者。
#       ⚠ 写 argmax 归约时这两者**给出不同的结果**，必须显式选一个并写进文档。

IEEE_NAN_MAX_FAMILY = ("maxNum (IEEE 754-2008, 忽略 NaN)  → fmaxf / np.fmax",
                       "maximum (IEEE 754-2019, 传播 NaN) → std::max? / np.maximum")
# 来源：IEEE 754-2008 §5.3.1 的 maxNum/minNum 与 IEEE 754-2019 的 maximum/minimum
# 口径：这里**不是**说 C++ 的 `std::max` 实现了 IEEE 的 `maximum`——`std::max(a,b)`
#       是 `a<b ? b : a`，对 NaN 的行为由**比较结果**决定（`NaN<x` 与 `x<NaN` 皆为假），
#       于是它既不完全是 maxNum 也不完全是 maximum。**本主题不依赖 `std::max`**，
#       只用 `fmaxf` 与显式比较两种写法，并分别标出它们的 NaN 口径。


# ===========================================================================
# 2. 逐代芯片规格（数据中心 SXM 档，本主题只用 H100 / B200）
# ===========================================================================

SM_COUNT_BY_GEN = {
    "V100": 80,     # 来源：NVIDIA Volta V100 Whitepaper（80 SM / 5120 FP32 core）
    "A100": 108,    # 来源：NVIDIA A100 Whitepaper 表1（108 SM / 6912 FP32 core）
    "H100": 132,    # 来源：NVIDIA Hopper Architecture In-Depth 表3（132 SM / 16896 FP32 core）
    "B200": 148,    # 来源：**非官方直述**——官方未披露 B200 的 SM 数，此值由官方
                    #       FP32 峰值 75 TFLOPS 反推 + Hot Chips 2024 交叉验证得到
}
FP32_LANES_PER_SM = {
    "V100": 64,     # 来源：NVIDIA Volta V100 Whitepaper（5120 ÷ 80）
    "A100": 64,     # 来源：NVIDIA A100 Whitepaper（6912 ÷ 108）
                    # 口径警告：**128 lane/SM 是消费级 GA10x 的事实**，GA100 是 64
    "H100": 128,    # 来源：NVIDIA Hopper Architecture In-Depth（16896 ÷ 132）
    "B200": 128,    # 来源：NVIDIA Blackwell Architecture Technical Brief V2.1
}
HBM_BW_GBps_BY_GEN = {
    "V100": 900.0,      # 来源：NVIDIA Volta V100 Whitepaper（HBM2，900 GB/s）
    "A100": 2039.0,     # 来源：NVIDIA A100 Whitepaper 表4（80GB HBM2e，2039 GB/s）
    "H100": 3350.0,     # 来源：NVIDIA Hopper Architecture In-Depth 表3（HBM3，3.35 TB/s）
    "B200": 7700.0,     # 来源：NVIDIA Blackwell Architecture Technical Brief V2.1（HBM3e）
}
# 口径：十进制 10^9 B/s，SXM 档。**不得混入 PCIe 档**。

FP32_VECTOR_TFLOPS_DENSE = {
    "V100": 15.7,   # 来源：NVIDIA Volta V100 Whitepaper
    "A100": 19.5,   # 来源：NVIDIA A100 Whitepaper 表1
    "H100": 67.0,   # 来源：NVIDIA Hopper Architecture In-Depth 表3 / H100 datasheet
    "B200": 75.0,   # 来源：NVIDIA Blackwell Architecture Technical Brief V2.1
}


# ===========================================================================
# 3. 外部引用值（**不是 NVIDIA 官方规格，本主题的程序不产出它们**）
# ===========================================================================

KERNEL_LAUNCH_OVERHEAD_US_EXTERNAL = (3.0, 5.0)
# 来源：外部对 CUDA 空 kernel 启动开销的对照测量（**非 NVIDIA 官方**）
# 口径：**引用值，本仓库不复核其测量口径**。量级是「一次 kernel 启动数微秒」，
#       随驱动版本、是否用 CUDA Graph、是否走 MPS 而变。正文只用作量级对照。
#       本主题 04 篇用它对比「atomic 单 kernel」与「两级 kernel」的固定开销。


# ===========================================================================
# 4. 明确未找到可靠来源的项——**留空，不填**
# ===========================================================================

UNSOURCED = (
    "`__shfl_sync` / `__shfl_down_sync` 单条指令的延迟（cycles）：本次来源未给出"
    "确定的数，本主题不做逐指令的延迟建模，只数指令条数与访存字节",
    "`bar.sync` 屏障本身的到达/释放延迟：同上，正文只用「一次块级屏障」做计数",
    "`atomicAdd(float)` 在 L2 上的吞吐（每周期几次）：官方未公布，"
    "本主题只算字节数，不把原子操作的吞吐折算成时间",
    "不同实现（cuBLAS / CUB / Thrust）归约的逐位结果：本仓库不复核其实现口径，"
    "正文不引用具体库的实际输出位模式",
)

# 处理方式：正文写「官方未披露」或直接回避，而不是填第三方聚合站的数字。

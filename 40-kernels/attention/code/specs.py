"""40-kernels/attention 主题的共享事实源。

本模块**只放从官方来源抄来的规格常量**，不放任何推导值。
代价模型的全部推导在 `verify.py` 里做，本模块不重复。

三条硬性约定（见 CONTRIBUTING §2 与 40-kernels/README.md）：

  1. **没有 `# 来源：` 注释的常量不许写进本模块。**
  2. **常量名必须编码口径**，使误用可见：带宽一律带 `_GBps`，字节宽度一律带 `_BYTES`，
     算力一律带 `_DENSE_SPARSE` 后缀并存成 `(dense, sparse_2to4)` 二元组；
     **指数/下溢的阈值一律不写进来**——那些是推导值。
  3. **找不到可靠来源的项不写进来**，改记为 `UNSOURCED`。

**常量与来源沿用 `20-gpu-architecture/*/code/specs.py` 已有的口径**：
`code/` 必须自包含所以重新抄一份，但同名的常量同值同来源，不得重新发明。
（本模块的 `WARP_SIZE` / `REGISTERS_PER_SM` / `SMEM_PER_SM_MAX_KB` /
`SMEM_DELIVERY_BYTES_PER_CLK` / `SMEM_LATENCY_CYCLES_RANGE` /
`HBM_LATENCY_CYCLES_RANGE` / `ISSUE_SLOTS_PER_CLK_PER_SM` / `MAX_THREADS_PER_SM` /
`MAX_BLOCKS_PER_SM` / `FLOAT32_BYTES` / `FLOAT16_BYTES` / `SM_COUNT_BY_GEN` /
`HBM_BW_GBps_BY_GEN` / `FP32_VECTOR_TFLOPS_DENSE` / `TC_TFLOPS_DENSE_SPARSE` 与
`40-kernels/gemm/code/specs.py` 逐字一致；`WARPGROUP_*` / `WGMMA_*` / `MMA_SHAPES` /
`ACCELERATED_TARGETS` 与 `20-gpu-architecture/tensor-core/code/specs.py` 逐字一致；
`FP16_*` / `FP32_*` 格式常量与 `40-kernels/softmax-normalization/code/specs.py` 一致。）

来源等级沿用本仓库约定：**A** = NVIDIA/IEEE 官方规范、白皮书、datasheet；
**B** = NVIDIA 官方代码或作者为 NVIDIA 的论文；**C** = 第三方逆向/学术。
**只有 A 级可以写进本模块的来源注释。**

**本主题用到的规格分三类**：硬件资源与带宽（§0 / §1）、IEEE 754 的格式常量（§2）、
指令与形状的 ISA 下限（§3）。attention 的**算法参数**（分块大小 `Br`/`Bc`、
paged KV 的页大小、`heads_kv`/`heads_q` 的比例）**不是规格**，它们住在 `verify.py`。
"""

# ===========================================================================
# 0. 基础口径与每 SM 资源
# ===========================================================================

WARP_SIZE = 32
# 来源：CUDA C++ Programming Guide 13.4 §5.1（各 compute capability 的
#       "warp size" 一栏恒为 32）；与本仓库 20-gpu-architecture/execution-model/01 一致

WARPGROUP_WARPS = 4
# 来源：PTX ISA §9.7.15.1（wgmma 的 "warpgroup" 定义为 4 个连续 warp，
#       它们必须一起执行同一条 wgmma 指令）

REGISTERS_PER_SM = 65536
# 来源：CUDA C++ Programming Guide 13.4 §5.1（CC 3.5 及以后各代
#       "Maximum number of 32-bit registers per SM" = 65536）
REGISTER_FILE_PER_SM_KB = 256
# 来源：同上（65536 × 4 B = 256 KB）
MAX_REGISTERS_PER_THREAD = 255
# 来源：CUDA C++ Programming Guide 13.4 §5.1（"Maximum number of 32-bit
#       registers per thread" 一栏，CC 3.0 起为 255）
REGISTER_ALLOC_GRANULARITY = 8
# 来源：CUDA C++ Programming Guide 13.4 §5.1 的寄存器分配粒度说明
#       （每线程寄存器数按 8 个一档向上取整）；与 execution-model/01 §1 同源

SMEM_PER_SM_MAX_KB = 228
# 来源：NVIDIA H100 Tensor Core GPU Architecture Whitepaper（每 SM 统一池
#       256 KB，其中至多 228 KB 可切给 shared memory）
SMEM_DELIVERY_BYTES_PER_CLK = 128
# 来源：NVIDIA CUDA C++ Programming Guide 13.4 §5.1（32 个 bank × 4 字节，
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
#       "Maximum number of resident blocks per SM" = 32）

FLOAT32_BYTES = 4
FLOAT16_BYTES = 2
FLOAT8_BYTES = 1
# 来源：IEEE 754 binary32 / binary16 的位宽（32 / 16 bit）；FP8 取 e4m3/e5m2 的 8 bit。
# 口径：这是**数据格式的位宽**，不代表访存粒度——一次合并访存的粒度是 32 字节 sector。

CACHE_LINE_BYTES = 128
SECTOR_BYTES = 32
# 来源：CUDA C++ Programming Guide 13.4 §5.1 的 "Global memory" 小节
#       （L1 cache line 128 字节 = 4 个 32 字节 sector，sector 是访存事务的最小单位）；
#       与 20-gpu-architecture/memory-system/02-coalescing-and-cache-line 同源
# 口径：**paged KV 的碎片化代价要按 sector 记，不是按页记**——页是软件的分配单位，
#       硬件的最小事务粒度仍是 32 字节 sector。


# ===========================================================================
# 1. 逐代芯片规格（**数据中心 SXM 档**）
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
# 口径：十进制 10^9 B/s，SXM 档。**不得混入 PCIe 档**（同代 PCIe 的带宽不同）。
HBM_CAPACITY_GB_BY_GEN = {
    "A100": 80.0,   # 来源：NVIDIA A100 Whitepaper 表4（80GB HBM2e 版本）
    "H100": 80.0,   # 来源：NVIDIA H100 datasheet（80GB HBM3 版本）
    "B200": 180.0,  # 来源：NVIDIA Blackwell Architecture Technical Brief V2.1（180GB HBM3e）
}
# 口径：**单卡容量**，用于「S 矩阵物化放不放得下」这一类对照，不含多卡聚合。

FP32_VECTOR_TFLOPS_DENSE = {
    "V100": 15.7,   # 来源：NVIDIA Volta V100 Whitepaper
    "A100": 19.5,   # 来源：NVIDIA A100 Whitepaper 表1
    "H100": 67.0,   # 来源：NVIDIA Hopper Architecture In-Depth 表3 / H100 datasheet
    "B200": 75.0,   # 来源：NVIDIA Blackwell Architecture Technical Brief V2.1
}
# 口径：**非 tensor core** 的 FP32 向量峰值。attention 的在线 softmax 那一步
#       （max / exp / 求和 / 重标定）走的正是这条通路；QKᵀ 与 PV 走 tensor core。

TC_TFLOPS_DENSE_SPARSE = {
    "TF32": {"A100": (156.0, 312.0), "H100": (494.5, 989.0), "B200": (1100.0, 2200.0)},
    "FP16": {"V100": (125.0, None), "A100": (312.0, 624.0),
             "H100": (989.5, 1979.0), "B200": (2250.0, 4500.0)},
    "BF16": {"A100": (312.0, 624.0), "H100": (989.5, 1979.0), "B200": (2250.0, 4500.0)},
    "FP8":  {"H100": (1979.0, 3958.0), "B200": (4500.0, 9000.0)},
    "FP4":  {"B200": (9000.0, 18000.0)},
}
# 来源：A100 与 V100 = NVIDIA A100 Whitepaper 表1/表4；H100 = Hopper Architecture
#       In-Depth 表3 与 H100 产品页；B200 = Blackwell Architecture Technical Brief V2.1
# 口径（沿用 tensor-core/code/specs.py，逐字一致）：
#   · **表中的 dense 值一律不含 2:4 稀疏**。NVIDIA 产品页写「H100 FP16 Tensor Core
#     1,979 TFLOPS」——那个 1,979 是**带 2:4 稀疏**的，dense 只有 989.5。
#   · **跨精度撞车**：A100 的 FP16 dense(312) == TF32 sparse(312)；H100 的
#     FP16 dense(989.5) == TF32 sparse(989)；B200 的 FP4 dense(9000) == FP8 sparse(9000)。
#   · attention 的 QKᵀ 与 PV 是**稠密**的：全文按 dense 一列记账，绝不引用 1,979。


# ===========================================================================
# 2. IEEE 754 的格式常量（阈值是推导值，不在这里）
# ===========================================================================

FP16_EXP_BITS = 5
FP16_MANT_BITS = 10
FP16_MAX_FINITE = 65504.0
FP16_MIN_SUBNORMAL = 5.960464477539063e-08
FP16_MIN_NORMAL = 6.103515625e-05
FP16_ROUNDING_UNIT = 4.8828125e-04
# 来源：IEEE Std 754-2019 binary16 的参数（p=11、emax=15、emin=-14），与
#       NVIDIA CUDA C++ Programming Guide 13.4 §5.1 的 fp16 范围一致
# 口径（逐项）：
#   · 最大有限数 = (2 - 2^-10)·2^15           = 65504
#     —— 它**不是** 65536：尾数只有 10 位，指数 15 档最大的就是它。
#   · 最小次正规数 = 2^(emin - m) = 2^-24     = 5.960464477539063e-08
#   · 最小正规数   = 2^emin = 2^-14           = 6.103515625e-05
#   · 舍入单位 u   = 2^-11                    = 4.8828125e-04（规格化区恒定）
# ⚠ 这三个数**都不是** exp 的阈值。阈值是 `ln(最大有限数与它上方相邻可表示值的中点)`
#   与 `ln(最小次正规数的一半)`，是**推导值**，见 verify.py 的 §A0。

FP32_EXP_BITS = 8
FP32_MANT_BITS = 23
FP32_MAX_FINITE = 3.4028234663852886e38
FP32_MIN_SUBNORMAL = 1.401298464324817e-45
FP32_MIN_NORMAL = 1.1754943508222875e-38
FP32_ROUNDING_UNIT = 5.960464477539063e-08
# 来源：IEEE Std 754-2019 binary32 的参数（p=24、emax=127、emin=-126）；
#       `FP32_MAX_FINITE` 与 C 的 `FLT_MAX` 同值
# 口径：最小次正规数 = 2^-149，最小正规数 = 2^-126，舍入单位 u = 2^-24。
# ⚠ **fp32 的下溢边界是「最小次正规数的一半」= 2^-150，不是 2^-127**
#   （2^-127 是「最小正规数的一半」，对应 `ln` 约 -88.03——那里 exp 仍然是一个
#    非零的次正规数，用它当阈值会把边界写早 16 个数量级）。完整推导见 verify.py §A0。


# ===========================================================================
# 3. 指令与形状的 ISA 下限（**照抄已入库的口径，不重推**）
# ===========================================================================

FEATURE_FIRST_TARGET = {
    "mma_sync":          ("Volta",     "sm_70"),
    "cp_async":          ("Ampere",    "sm_80"),
    "tma_cp_async_bulk": ("Hopper",    "sm_90"),
    "wgmma":             ("Hopper",    "sm_90a"),
    "tmem":              ("Blackwell", "sm_100a"),
    "tcgen05":           ("Blackwell", "sm_100a"),
}
# 来源：PTX ISA 9.4 各指令的 "Target ISA Notes"；与
#       20-gpu-architecture/architecture-evolution/code/specs.py 的
#       FEATURE_FIRST_TARGET 逐条一致
# ⚠ 两条最常写错的：
#   · **TMA（`cp.async.bulk`）是 `sm_90`，不是 `sm_90a`**——只有 multicast 等
#     子变体才被收到 `sm_90a`。
#   · `wgmma` 才是 `sm_90a`（它的 `.aligned` 语义要求整个 warpgroup 一起发射）。

ACCELERATED_TARGETS = ("sm_90a", "sm_100a", "sm_100f")
# 来源：PTX ISA 9.4 开头对 "a"/"f" 后缀的定义——**只在该精确架构上可用**，
#       不像 `sm_90` 那样兼容 sm_90 家族的其它成员。
# 口径：**这条是单向的**——为 `sm_90` 编出的 cubin 能在 `sm_90a` 硬件（H100）上
#       正常加载运行，反过来不行。写成「两者不兼容」是错的。

MMA_SHAPES = {
    # 结构：形状名 → (首次可用代际, 最低 target)
    # 来源：PTX ISA 9.4 §9.7.16.1 "Matrix Shape" 表 + 同节各指令的 Target ISA Notes
    # 口径：**PTX 的 Shape 表是按 PTX ISA 版本组织的，不是按架构组织的**；
    #       要判断某形状属于哪一代，必须与 Target ISA Notes 交叉比对。
    "m16n8k16.f16":  ("Ampere",  "sm_80"),
    "m16n8k8.bf16":  ("Ampere",  "sm_80"),
    "m16n8k8.tf32":  ("Ampere",  "sm_80"),
    "m64nNk16.f16":  ("Hopper",  "sm_90a"),
    "m64nNk16.bf16": ("Hopper",  "sm_90a"),
    "m64nNk8.tf32":  ("Hopper",  "sm_90a"),
}
# 本主题只用这张表的五行（其余形状见 20-gpu-architecture/tensor-core/01-02 与
# 本模块的完整版出处）。attention 的 QKᵀ 与 PV 都落在
# m16n8k16（sm_80 档）与 m64nNk16（sm_90a 档）上。

WGMMA_M = 64
WGMMA_N_MIN = 8
WGMMA_N_MAX = 256
WGMMA_N_STEP = 8
# 来源：PTX ISA §9.7.15.4.1 wgmma 形状枚举（.m64n8k16 … .m64n256k16，
#       N 从 8 到 256、步长 8；f16/bf16 的 K=16，tf32 的 K=8）
# 口径：**wgmma 的 M 恒为 64**，不由调用者选择——这是它与 mma.sync 最直观的区别。

WGMMA_B_MUST_BE_SMEM = True
WGMMA_A_MAY_BE_REG_OR_SMEM = True
# 来源：PTX ISA §9.7.15（wgmma 的操作数：B 必须来自 shared memory 描述符；
#       A 可以是寄存器也可以是 shared memory 描述符）
# ⚠ 常见错误：「wgmma 的 A、B 都必须在 SMEM」是错的。

WGMMA_REQUIRES_WHOLE_WARPGROUP = True
WGMMA_ALIGNED_QUALIFIER_MANDATORY = True
# 来源：PTX ISA §9.7.15 的 `.aligned` 限定符（"all threads in the warpgroup must
#       execute the same instruction"；条件分支里若不能保证 warpgroup 内条件一致，
#       行为未定义）
# 口径：这就是「wgmma 的发射粒度是 128 线程」的来源。**warp specialization 里
#       分工出去做访存/softmax 的 warp 不能与做 wgmma 的 warp 混在同一个 warpgroup 里。**


# ===========================================================================
# 4. 外部引用值（**不是 NVIDIA 官方规格，本主题的程序不产出它们**）
# ===========================================================================

KERNEL_LAUNCH_OVERHEAD_US_EXTERNAL = (3.0, 5.0)
# 来源：外部对 CUDA 空 kernel 启动开销的对照测量（**非 NVIDIA 官方**）
# 口径：**引用值，本仓库不复核其测量口径**。量级是「一次 kernel 启动数微秒」，
#       随驱动版本、是否用 CUDA Graph、是否走 MPS 而变。正文只用作量级对照：
#       「反向多一个归约 kernel」与「多一次 launch」要放在一起比。


# ===========================================================================
# 5. 明确未找到可靠来源的项——**留空，不填**
# ===========================================================================

UNSOURCED = (
    "`atomicAdd` 在 L2 上的实际吞吐（以「每周期几次」为单位）：本次采用的来源"
    "只给出 HBM/L2 带宽这一类聚合口径，没有给出原子操作的单位吞吐，"
    "因此**不写进本模块**。正文只做「原子版与 workspace 版的**访存字节数**相当、"
    "差别在确定性与一次额外的 launch」这一层讨论。",
    "注意力里 softmax 那一步（max / exp / 求和 / 重标定）在 H100 上的实际耗时："
    "本仓库不复核其测量口径，正文不引用。",
    "H100 SXM5 的官方 boost 时钟：官方未公布，只能用 67 TFLOPS ÷ (132×128×2) 反推。",
    "给定 kernel 在某台机器上的实际耗时与 MFU：本仓库不复核其测量口径，正文不引用。",
)
# 处理方式：正文写「官方未披露」或直接回避，而不是填第三方聚合站的数字。

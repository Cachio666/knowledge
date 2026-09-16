"""40-kernels/gemm 主题的共享事实源。

本模块**只放从官方来源抄来的硬件规格常量**，不放任何推导值。
代价模型的全部推导在 `verify.py` 里做，本模块不重复。

三条硬性约定（见 CONTRIBUTING §2 与 40-kernels/README.md）：

  1. **没有 `# 来源：` 注释的常量不许写进本模块。**
  2. **常量名必须编码口径**，使误用可见：算力一律带 `_DENSE_SPARSE` 后缀并
     **存成 `(dense, sparse_2to4)` 二元组**，带宽一律带 `_GBps`，容量一律带 `_KB`。
  3. **找不到可靠来源的项不写进来**，改记为 `UNSOURCED`。

**常量与来源沿用 `20-gpu-architecture/*/code/specs.py` 已有的口径**：
`code/` 必须自包含所以重新抄一份，但同名的常量同值同来源，不得重新发明。
（本模块的 `SMEM_PER_SM_MAX_KB` / `REGISTERS_PER_SM` / `HBM_BW_GBps_BY_GEN` /
`TC_TFLOPS_DENSE_SPARSE` 与 `20-gpu-architecture/tensor-core/code/specs.py` 逐字一致。）

来源等级沿用本仓库约定：**A** = NVIDIA 官方规范/白皮书/datasheet；
**B** = NVIDIA 官方代码或作者为 NVIDIA 的论文；**C** = 第三方逆向/学术。
**只有 A 级可以写进本模块的来源注释。**
"""

# ===========================================================================
# 0. 基础口径与每 SM 资源
# ===========================================================================

WARP_SIZE = 32
# 来源：CUDA C++ Programming Guide 13.4 §5.1（各 compute capability 的
#       "warp size" 一栏恒为 32）；与本仓库 20-gpu-architecture/execution-model/01 一致

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

# 每 SM 的常驻上限——**不是不变量**，只对数据中心 SXM 线成立
MAX_THREADS_PER_SM = 2048
MAX_BLOCKS_PER_SM = 32
# 来源：CUDA C++ Programming Guide 13.4 §5.1 表（CC 9.0 / 10.0 的
#       "Maximum number of resident threads per SM" = 2048 与
#       "Maximum number of resident blocks per SM" = 32）；
#       与 20-gpu-architecture/architecture-evolution/code/specs.py 的
#       RESIDENT_LIMITS_BY_CC[("9.0"|"10.0")] 一致（64 warp / 2048 线程 / 32 block）

FLOAT32_BYTES = 4
FLOAT16_BYTES = 2
FLOAT8_BYTES = 1
# 来源：IEEE 754 binary32 / binary16 的位宽（32 / 16 bit）；FP8 取 e4m3/e5m2 的 8 bit。
# 口径：这是**数据格式的位宽**，不代表访存粒度——一次合并访存的粒度是 32 字节 sector。


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

FP32_VECTOR_TFLOPS_DENSE = {
    "V100": 15.7,   # 来源：NVIDIA Volta V100 Whitepaper
    "A100": 19.5,   # 来源：NVIDIA A100 Whitepaper 表1
    "H100": 67.0,   # 来源：NVIDIA Hopper Architecture In-Depth 表3 / H100 datasheet
    "B200": 75.0,   # 来源：NVIDIA Blackwell Architecture Technical Brief V2.1
}
# 口径：**非 tensor core** 的 FP32 向量峰值（GEMM 的 FP32 内核走的就是这条通路）。

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

FP32_LANES_FOR_PEAK = {"V100": 64, "A100": 64, "H100": 128, "B200": 128}
# 来源：同 FP32_LANES_PER_SM（用于把向量峰值反解成时钟）
# 口径：本表与 FP32_LANES_PER_SM 同值，保留独立名字是为了让「用哪个数反推时钟」
#       在程序里显式可见——这两个口径一旦分叉，反推出的时钟就会变。


# ===========================================================================
# 2. 指令的 ISA 下限（**照抄已入库的口径，不重推**）
# ===========================================================================

FEATURE_FIRST_TARGET = {
    "mma_sync":        ("Volta",     "sm_70"),
    "cp_async":        ("Ampere",    "sm_80"),
    "tma_cp_async_bulk": ("Hopper",  "sm_90"),
    "wgmma":           ("Hopper",    "sm_90a"),
    "tmem":            ("Blackwell", "sm_100a"),
    "tcgen05":         ("Blackwell", "sm_100a"),
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

CP_ASYNC_LEGAL_CP_SIZES = (4, 8, 16)
# 来源：PTX ISA §9.7.9 `cp.async` 小节（`cp-size` 的可取值为 4 / 8 / 16 字节）
# 口径：`cp.async` 单条指令每线程搬运 4/8/16 字节；**16 字节（一个 float4）是它的
#       规范粒度**，本仓库 architecture-evolution/01 §3 与
#       execution-model/01 §3 已按此口径记账，本主题沿用。

TMA_SINGLE_THREAD_ISSUE = True
# 来源：NVIDIA Hopper Architecture In-Depth 与 PTX ISA §9.7.9 `cp.async.bulk` 小节
#       （**一个线程即可发起整块多维搬运**；地址由 tensor map / 描述符给出，
#        不占参与线程的寄存器）；与 architecture-evolution/01 §3 的表一致
TMA_DESCRIPTOR_IN_GLOBAL = True
# 来源：同上（描述符住在 global memory，由 host 端 API 构造，设备侧只传指针）

WGMMA_B_MUST_BE_SMEM = True
WGMMA_A_MAY_BE_REG_OR_SMEM = True
# 来源：PTX ISA §9.7.15（wgmma 的操作数：B 必须来自 shared memory 描述符；
#       A 可以是寄存器也可以是 shared memory 描述符）
# ⚠ 常见错误：「wgmma 的 A、B 都必须在 SMEM」是错的。

TCGEN05_ISSUE_BY_SINGLE_THREAD = True
TCGEN05_D_IN_TMEM_ONLY = True
TCGEN05_B_IN_SMEM_ONLY = True
# 来源：PTX ISA §9.7.18（tcgen05.mma：**单线程发射**；D 必须在 tensor memory；
#       A 可在 tensor memory 或 shared memory；B 必须来自 shared memory）
# 口径：与 wgmma 的差异就在 D——wgmma 的 D 在寄存器，tcgen05 的 D 在 TMEM。

TMEM_LANES_PER_CTA = 128
TMEM_MAX_COLUMNS_PER_CTA = 512
TMEM_CELL_BITS = 32
TMEM_ALLOC_GRANULARITY_COLUMNS = 32
TMEM_ALLOC_LEGAL_COLUMNS = (32, 64, 128, 256, 512)
# 来源：PTX ISA §9.7.18 的 tensor memory 小节（TMEM 是 128 lane × 至多 512 列的
#       二维地址空间，每 cell 32 bit；列是动态分配的，分配粒度 32 列，合法的
#       n_cols 取值是 32/64/128/256/512）
# 口径：**PTX 给的口径是 per CTA，不是 per SM**。128 × 512 × 4 B = 256 KB，
#       恰好等于每 SM 的容量；一个 SM 上多个 CTA 如何仲裁 TMEM，本次采用的
#       来源没有说明——正文回避这一点，只做 per CTA 的算术。


# ===========================================================================
# 3. 外部引用值（**不是 NVIDIA 官方规格，本主题的程序不产出它们**）
# ===========================================================================

KERNEL_LAUNCH_OVERHEAD_US_EXTERNAL = (3.0, 5.0)
# 来源：外部对 CUDA 空 kernel 启动开销的对照测量（**非 NVIDIA 官方**）
# 口径：**引用值，本仓库不复核其测量口径**。量级是「一次 kernel 启动数微秒」，
#       随驱动版本、是否用 CUDA Graph、是否走 MPS 而变。正文只用作量级对照。


# ===========================================================================
# 4. 明确未找到可靠来源的项——**留空，不填**
# ===========================================================================

UNSOURCED = (
    "H100 SXM5 的官方 boost 时钟：官方未公布，只能用 67 TFLOPS ÷ (132×128×2) 反推",
    "TMA 的 `cp.async.bulk` 单条指令的字节数上限：本次来源未给出，"
    "本主题不做单条指令粒度的算术，正文回避",
    "一个 SM 上多个 CTA 如何仲裁 TMEM：PTX 只给 per CTA 口径，来源未说明仲裁规则",
    "cuBLAS / CUTLASS 在给定形状上的 TFLOPS 观测值：本仓库不复核其测量口径，正文不引用",
)
# 处理方式：正文写「官方未披露」或直接回避，而不是填第三方聚合站的数字。

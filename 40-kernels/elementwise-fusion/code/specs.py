"""40-kernels/elementwise-fusion 主题的共享事实源。

本模块**只放从官方来源抄来的规格常量与语义事实**，不放任何推导值。
代价模型的全部推导在 `verify.py` 里做，本模块不重复。

三条硬性约定（见 CONTRIBUTING §2 与 40-kernels/README.md）：

  1. **没有 `# 来源：` 注释的常量不许写进本模块。**
  2. **常量名必须编码口径**，使误用可见：带宽一律带 `_GBps`，容量一律带 `_KB`/`_MB`，
     字节数一律带 `_BYTES`，语义事实一律用一个**断言式短句**做名字
     （`HALF2_IS_NOT_A_16B_VECTOR`），使「反过来说」在命名上就说不通。
  3. **找不到可靠来源的项不写进来**，改记为 `UNSOURCED`。

**常量与来源沿用 `20-gpu-architecture/*/code/specs.py` 与
`40-kernels/reduce-scan/code/specs.py` 已有的口径**：`code/` 必须自包含所以重新抄一份，
但同名的常量同值同来源，不得重新发明。
（本模块的 `WARP_SIZE` / `MAX_WARPS_PER_SM` / `REGISTERS_PER_SM` /
`REGISTER_FILE_PER_SM_KB` / `MAX_REGISTERS_PER_THREAD` / `REGISTER_ALLOC_GRANULARITY` /
`SMEM_PER_SM_MAX_KB` / `HBM_LATENCY_CYCLES_RANGE` / `ISSUE_SLOTS_PER_CLK_PER_SM` /
`MAX_THREADS_PER_SM` / `MAX_BLOCKS_PER_SM` / `FLOAT32_BYTES` / `FLOAT16_BYTES` /
`MAX_GRID_DIM` / `SM_COUNT_BY_GEN` / `HBM_BW_GBps_BY_GEN` /
`FP32_VECTOR_TFLOPS_DENSE` / `KERNEL_LAUNCH_OVERHEAD_US_EXTERNAL` 与
`40-kernels/reduce-scan/code/specs.py` 逐字一致。）

来源等级沿用本仓库约定：**A** = NVIDIA 官方规范/白皮书/datasheet；
**B** = NVIDIA 官方代码或作者为 NVIDIA 的论文；**C** = 第三方逆向/学术。
**只有 A 级可以写进本模块的来源注释。**
"""

# ===========================================================================
# 0. 基础口径与每 SM 资源
# ===========================================================================

WARP_SIZE = 32
# 来源：CUDA C++ Programming Guide 13.4 §5.1（各 compute capability 的
#       "warp size" 一栏恒为 32）；与本仓库 40-kernels/reduce-scan/code/specs.py 一致

MAX_WARPS_PER_SM = 64
# 来源：CUDA C++ Programming Guide 13.4 §5.1 表（CC 7.0/8.0/9.0/10.0 的
#       "Maximum number of resident warps per SM" = 64）
# 口径：64 × 32 = 2048，与 MAX_THREADS_PER_SM 是**同一条约束的两种写法**。

REGISTERS_PER_SM = 65536
# 来源：CUDA C++ Programming Guide 13.4 §5.1（CC 3.5 及以后各代
#       "Maximum number of 32-bit registers per SM" = 65536）
REGISTER_FILE_PER_SM_KB = 256
# 来源：同上（65536 × 4 B = 256 KB）
MAX_REGISTERS_PER_THREAD = 255
# 来源：CUDA C++ Programming Guide 13.4 §5.1（"Maximum number of 32-bit
#       registers per thread"，CC 3.0 起为 255）
# 口径：这是**硬上限**。融合链把每线程的活跃值累加到 255 以上时，编译器
#       没有别的选择——只能把多出来的值放进 local memory（spill）。
REGISTER_ALLOC_GRANULARITY = 8
# 来源：CUDA C++ Programming Guide 13.4 §5.1 的寄存器分配粒度说明
#       （每线程寄存器数按 8 个一档向上取整）；与 execution-model/01 §1 同源

MAX_THREADS_PER_SM = 2048
MAX_BLOCKS_PER_SM = 32
# 来源：CUDA C++ Programming Guide 13.4 §5.1 表（CC 9.0 / 10.0 的
#       "Maximum number of resident threads per SM" = 2048 与
#       "Maximum number of resident blocks per SM" = 32）

SMEM_PER_SM_MAX_KB = 228
# 来源：NVIDIA H100 Tensor Core GPU Architecture Whitepaper（每 SM 统一池
#       256 KB，其中至多 228 KB 可切给 shared memory）
# 口径：本主题的三个内核**都不用共享内存**。这一条只用来在 §4 说明
#       「显式占用会挤压 occupancy」的路不止寄存器一条，不做定量结论。

HBM_LATENCY_CYCLES_RANGE = (400, 800)
# 来源：本仓库 20-gpu-architecture/memory-system/01-hierarchy-overview §1
#       （一次片外访存的往返延迟 400~800 周期）
ISSUE_SLOTS_PER_CLK_PER_SM = 4
# 来源：NVIDIA Hopper Architecture In-Depth（每 SM 4 个 warp scheduler，各每周期
#       发射一条指令）；Pascal~Blackwell 数据中心线均为 4

FLOAT32_BYTES = 4
FLOAT16_BYTES = 2
FLOAT8_BYTES = 1
# 来源：IEEE 754 binary32 / binary16 的位宽（32 / 16 bit）；FP8 取 e4m3/e5m2 的 8 bit。
# 口径：这是**数据格式的位宽**，不代表访存粒度——访存粒度见下面的 SECTOR_BYTES。

MAX_GRID_DIM = {"x": 2147483647, "y": 65535, "z": 65535}
# 来源：CUDA C++ Programming Guide 13.4 §5.1 表（"Maximum x-dimension of a grid
#       of thread blocks" = 2^31−1；y 与 z 维均为 65535）
# 口径：逐元素算子的 grid 通常是一维的。这条在本主题只用于排除一种可能性——
#       N 再大也撞不到 x 维上限（2^31−1 个 block 要 N ≥ 2^31·1024 个元素），
#       所以「grid 太大」不是本主题的失败模式，「**grid 太小**」才是（03 篇 §2）。


# ===========================================================================
# 1. 访存粒度（本主题全部推导的地基）
# ===========================================================================

SECTOR_BYTES = 32
CACHE_LINE_BYTES = 128
# 来源：CUDA C++ Programming Guide ——「Device Memory Accesses」章节
#       （合并成服务完所有线程所需的 32 字节事务；cache line 为 128 字节并映射到
#        显存中 128 字节对齐的一段；只有被触及的 sector 会被取回）
# 口径：**两个数的分工不是「两级缓存」**——line 是**记账单位**（一条 tag 覆盖 128 字节，
#       一次 warp 访存落在几条 line 上就是几条 L1→L2 请求），sector 是**搬运单位**
#       （实际搬动的字节数恒为 32 的整数倍）。完整推导在
#       20-gpu-architecture/memory-system/02-coalescing-and-cache-line §1，
#       **本主题只代入、不重推**。

MAX_VECTOR_BYTES_IN_CUDA_C = 16
# 来源：CUDA C++ Programming Guide 的内建向量类型表（`float4` / `int4` / `uint4` /
#       `double2` / `longlong2` 均为 16 字节，是 C++ 层可用的最宽向量类型）
# 口径：**这是 CUDA C++ 的向量类型上限，不是硬件的访存宽度上限**。
#       PTX 在 Blackwell 上引入了更宽的形式（见 UNSOURCED）。

HALF2_BYTES = 4
HALF2_IS_NOT_A_16B_VECTOR = True
# 来源：CUDA Math API —— `__half2` 是**两个 `__half` 打包进一个 32-bit 寄存器**
#       （`__half` = 16 bit，`2 × 16 = 32 bit = 4 字节`）
# 口径：⚠ **半精度 kernel 里 `__half2` 只把每次访存从 2 字节抬到 4 字节**，
#       离 16 字节的向量上限还差 4 倍。要拿满宽度必须一次搬 8 个 half
#       （4 个 `__half2`，或按 `uint4` / `float4` 重新解释）。本主题 03 篇 §4 用。


# ===========================================================================
# 2. 逐代芯片规格（**数据中心 SXM 档**）
# ===========================================================================

SM_COUNT_BY_GEN = {
    "V100": 80,     # 来源：NVIDIA Volta V100 Whitepaper（80 SM / 5120 FP32 core）
    "A100": 108,    # 来源：NVIDIA A100 Whitepaper 表1（108 SM / 6912 FP32 core）
    "H100": 132,    # 来源：NVIDIA Hopper Architecture In-Depth 表3（132 SM / 16896 FP32 core）
    "B200": 148,    # 来源：**非官方直述**——官方未披露 B200 的 SM 数，此值由官方
                    #       FP32 峰值 75 TFLOPS 反推 + Hot Chips 2024 交叉验证得到
}
HBM_BW_GBps_BY_GEN = {
    "V100": 900.0,      # 来源：NVIDIA Volta V100 Whitepaper（HBM2，900 GB/s）
    "A100": 2039.0,     # 来源：NVIDIA A100 Whitepaper 表4（80GB HBM2e，2039 GB/s）
    "H100": 3350.0,     # 来源：NVIDIA Hopper Architecture In-Depth 表3（HBM3，3.35 TB/s）
    "B200": 7700.0,     # 来源：NVIDIA Blackwell Architecture Technical Brief V2.1（HBM3e）
}
# 口径：十进制 10^9 B/s，SXM 档。**不得混入 PCIe 档**（同代 PCIe 的带宽不同）。
#       ⚠ 这是**标称**带宽（datasheet 值）。**可达**带宽低于它——这个 gap 的分解在
#       20-gpu-architecture/interconnect/04-measuring-bandwidth 与
#       memory-system/01-hierarchy-overview §2，**本主题不重推**，只做代入（01 篇 §1）。

FP32_VECTOR_TFLOPS_DENSE = {
    "V100": 15.7,   # 来源：NVIDIA Volta V100 Whitepaper
    "A100": 19.5,   # 来源：NVIDIA A100 Whitepaper 表1
    "H100": 67.0,   # 来源：NVIDIA Hopper Architecture In-Depth 表3 / H100 datasheet
    "B200": 75.0,   # 来源：NVIDIA Blackwell Architecture Technical Brief V2.1
}
# 口径：**非 tensor core** 的 FP32 向量峰值。逐元素算子走的就是这条通路。
FP32_LANES_PER_SM = {
    "V100": 64,     # 来源：NVIDIA Volta V100 Whitepaper（5120 ÷ 80）
    "A100": 64,     # 来源：NVIDIA A100 Whitepaper（6912 ÷ 108）
    "H100": 128,    # 来源：NVIDIA Hopper Architecture In-Depth（16896 ÷ 132）
    "B200": 128,    # 来源：NVIDIA Blackwell Architecture Technical Brief V2.1
}
# 口径：本主题只用它把 FP32 峰值反解成时钟（`verify.py` 的推导，不写进本模块）。


# ===========================================================================
# 3. 本仓库上游文档已入库的口径（**转引，本模块不重新溯源**）
# ===========================================================================
#
# 本主题是**下游性极强**的一篇：几乎每条机制都在别处。这一节把「本主题要代入、
# 但正文在别处」的量集中放在这里，让 `verify.py` 能引用它们而**不重新推导**。

ACHIEVABLE_BW_SHORTFALL_PCT_IN_REPO = (5, 15)
# 来源：本仓库 10-foundations/performance-models/03-measuring-roofline §0——
#       「真实的**可达**峰值低于规格值——boost clock 漂移、占用率不足、ECC 开销
#        都会扣掉几个到十几个百分点」
# 口径：**这是「几个到十几个百分点」这句话的区间化**，本仓库对它的原始出处
#       没有给出一手页面。**引用值，本仓库不复核其测量口径**。
#       它对应 01 篇 §1 里的「可达带宽系数 η」：η = 1 − 缺口，即 η ∈ [0.85, 0.95]。
#       η = 1（标称）只作为**乐观下界**出现，不是可达值。

L2_MB_H100_IN_REPO = 50
# 来源：本仓库 10-foundations/performance-models/09_roofline_model（「H100 有 50 MB L2」）
#       与 performance-models/03-measuring-roofline §2 的规格表
# 口径：⚠ **本仓库对 L2 容量这一项内部不一致**——`20-gpu-architecture/
#       architecture-evolution/code/specs.py` 只把 **B200** 的 L2 列为 UNSOURCED，
#       并未给 H100 的值。本条只用于 02 篇 §1 的量级判断（「中间张量装不装得进
#       L2」），**不做精确结论**，正文每次使用都注明这一点。

KERNEL_LAUNCH_OVERHEAD_US_EXTERNAL = (3.0, 5.0)
# 来源：外部对 CUDA 空 kernel 启动开销的对照测量（**非 NVIDIA 官方**）
# 口径：**引用值，本仓库不复核其测量口径**。量级是「一次 kernel 启动数微秒」，
#       随驱动版本、是否用 CUDA Graph、是否走 MPS 而变。正文只用作量级对照。
#       （与 40-kernels/gemm/code/specs.py 的常量同名同值同来源。）

END_TO_END_FUSION_SPEEDUP_RANGE_EXTERNAL = (1.2, 1.5)
# 来源：外部对**端到端融合收益**的报道区间（**本次未能定位一手页面**，见 §5 UNSOURCED）
# 口径：**引用值，本仓库不复核其测量口径**。两点使用边界：
#       ① 它描述的是**整个模型/整个网络**的端到端收益，不是「那一串逐元素算子自己」
#          的收益——后者按 02 篇 §1 的模型甲可以到 k，按模型乙只有 1；
#       ② 它与 `20-gpu-architecture/tensor-core/05-structured-sparsity` 里的
#          「1.2×~1.6×」**不是同一个数**（那个有可定位的来源：NVIDIA TensorRT 官方
#          博客的 A100 / ResNeXt-101 端到端观测），**两者机制不同，不可互换引用**。
#          正文只用它做「量级」对照，并同时给出它自己的成因解释。


# ===========================================================================
# 4. 指令的 ISA 下限（**照抄已入库的口径，不重推**）
# ===========================================================================

FEATURE_FIRST_TARGET = {
    "ld_global_v4_b32": ("Kepler", "sm_30"),
    "ld_global_v8_b32": ("Blackwell", "sm_100a"),
}
# 来源：PTX ISA 各版 `ld` 指令的 "Target ISA Notes"；`ld.global.v4.b32`（即
#       `float4` 的 PTX 形态）自 Kepler 起可用；更宽的 `.v8.b32` 形式在
#       Blackwell 的 accelerated 档引入
# ⚠ **`.v8` 这一行本次未能逐字核对 PTX ISA 原文**（见 §5 UNSOURCED），
#   所以本模块把它只当「16 字节不是硬件的上限」这一句话的依据，
#   **不据此做任何定量结论**，正文 03 篇 §3 也不引用它的具体档位。


# ===========================================================================
# 5. 明确未找到可靠来源的项——**留空，不填**
# ===========================================================================

UNSOURCED = (
    "H100 SXM5 的官方 boost 时钟：官方未公布，只能用 67 TFLOPS ÷ (132×128×2) 反推。"
    "本主题 03 篇 §1 会用到这个反推值，正文与 verify.py 都显式标注为"
    "「由规格值反推的推导值」，并说明它只在该卡跑在 boost 时钟下时成立",

    "「端到端融合收益 1.2×~1.5×」的一手页面：本次未定位。"
    "本仓库已入库的同类引用值（结构化稀疏的 1.2×~1.6×）有一手来源，"
    "但那是另一个机制。本条只保留量级，正文注明「一手来源未定位」",

    "`ld.global.v8.b32` / `.v8.f32` 的确切 PTX 版本与目标档位："
    "本次未逐字核对。本主题不做 256 位访存的定量结论，正文只提一句「16 字节是"
    "CUDA C++ 向量类型的上限，不是硬件访存宽度的上限」",

    "各级缓存的**命中率**与 **L2 的替换策略**：官方未公布，"
    "本主题只做「装得下/装不下」的量级判断，不建模命中率",

    "一次向量化访存在 SASS 上是否真的合成一条 128 位指令："
    "取决于编译器的向量化与合并，本主题不假设它一定合并",
)
# 处理方式：正文写「官方未披露」或直接回避，而不是填第三方聚合站的数字。

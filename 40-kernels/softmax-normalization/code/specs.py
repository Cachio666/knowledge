"""40-kernels/softmax-normalization 主题的共享事实源。

本模块**只放从官方来源抄来的规格常量**，不放任何推导值。
代价模型的全部推导在 `verify.py` 里做，本模块不重复。

三条硬性约定（见 CONTRIBUTING §2 与 40-kernels/README.md）：

  1. **没有 `# 来源：` 注释的常量不许写进本模块。**
  2. **常量名必须编码口径**，使误用可见：带宽一律带 `_GBps`，字节宽度一律带 `_BYTES`，
     格的宽度一律带 `_UNIT`；**指数/下溢的阈值一律不写进来**——那些是推导值。
  3. **找不到可靠来源的项不写进来**，改记为 `UNSOURCED`。

**常量与来源沿用 `20-gpu-architecture/*/code/specs.py` 已有的口径**：
`code/` 必须自包含所以重新抄一份，但同名的常量同值同来源，不得重新发明。
（本模块的 `WARP_SIZE` / `REGISTERS_PER_SM` / `SMEM_PER_SM_MAX_KB` /
`SMEM_DELIVERY_BYTES_PER_CLK` / `SMEM_LATENCY_CYCLES_RANGE` /
`HBM_LATENCY_CYCLES_RANGE` / `ISSUE_SLOTS_PER_CLK_PER_SM` / `MAX_THREADS_PER_SM` /
`MAX_BLOCKS_PER_SM` / `FLOAT32_BYTES` / `FLOAT16_BYTES` / `SM_COUNT_BY_GEN` /
`HBM_BW_GBps_BY_GEN` / `FP32_VECTOR_TFLOPS_DENSE` 与
`40-kernels/gemm/code/specs.py` 逐字一致，后者又逐字来自
`20-gpu-architecture/tensor-core/code/specs.py`。）

来源等级沿用本仓库约定：**A** = NVIDIA/IEEE 官方规范、白皮书、datasheet；
**B** = NVIDIA 官方代码或作者为 NVIDIA 的论文；**C** = 第三方逆向/学术。
**只有 A 级可以写进本模块的来源注释。**

**本主题只用到「格式常量」这一类规格。** exp 的上溢/下溢阈值（`ln 65520`、
`-25·ln 2` 等）是**推导值**，一律住在 `verify.py`，不在本模块。
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
# 来源：IEEE 754 binary32 / binary16 的位宽（32 / 16 bit）。
# 口径：这是**数据格式的位宽**，不代表访存粒度——一次合并访存的粒度是 32 字节 sector。


# ===========================================================================
# 1. 逐代芯片规格（**数据中心 SXM 档**）—— 本主题只用 H100 与 B200 两列
# ===========================================================================

SM_COUNT_BY_GEN = {
    "A100": 108,    # 来源：NVIDIA A100 Whitepaper 表1（108 SM / 6912 FP32 core）
    "H100": 132,    # 来源：NVIDIA Hopper Architecture In-Depth 表3（132 SM / 16896 FP32 core）
    "B200": 148,    # 来源：**非官方直述**——官方未披露 B200 的 SM 数，此值由官方
                    #       FP32 峰值 75 TFLOPS 反推 + Hot Chips 2024 交叉验证得到
}
HBM_BW_GBps_BY_GEN = {
    "A100": 2039.0,     # 来源：NVIDIA A100 Whitepaper 表4（80GB HBM2e，2039 GB/s）
    "H100": 3350.0,     # 来源：NVIDIA Hopper Architecture In-Depth 表3（HBM3，3.35 TB/s）
    "B200": 7700.0,     # 来源：NVIDIA Blackwell Architecture Technical Brief V2.1（HBM3e）
}
# 口径：十进制 10^9 B/s，SXM 档。**不得混入 PCIe 档**（同代 PCIe 的带宽不同）。

FP32_VECTOR_TFLOPS_DENSE = {
    "A100": 19.5,   # 来源：NVIDIA A100 Whitepaper 表1
    "H100": 67.0,   # 来源：NVIDIA Hopper Architecture In-Depth 表3 / H100 datasheet
    "B200": 75.0,   # 来源：NVIDIA Blackwell Architecture Technical Brief V2.1
}
# 口径：**非 tensor core** 的 FP32 向量峰值。softmax / LayerNorm 走的正是这条通路
#       ——它们没有矩阵乘，tensor core 帮不上忙。


# ===========================================================================
# 2. IEEE 754 的格式常量（本主题的推导全部从这里出发）
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
# ⚠ 这三个数**都不是** exp 的阈值。exp 的阈值是 `ln(最大有限数与它上方
#   相邻可表示值的中点)`、`ln(最小次正规数的一半)`，是**推导值**，见 verify.py。

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
#   （2^-127 是「最小正规数的一半」，对应 `ln` 约 -88.03——那里 exp 仍然是
#    一个**非零**的次正规数，用它当阈值会把边界写早 16 个数量级）。
#    两个精度的下溢判据必须用同一口径。完整推导见 verify.py 的 §B1。


# ===========================================================================
# 3. 外部引用值（**不是 NVIDIA 官方规格，本主题的程序不产出它们**）
# ===========================================================================

UNSOURCED_LAUNCH_OVERHEAD_US = (3.0, 5.0)
# 来源：外部对 CUDA 空 kernel 启动开销的对照测量（**非 NVIDIA 官方**）
# 口径：**引用值，本仓库不复核其测量口径**。量级是「一次 kernel 启动数微秒」，
#       随驱动版本、是否用 CUDA Graph、是否走 MPS 而变。正文只用作量级对照：
#       「把 softmax 的三遍合成一遍省下的时间」与「多一次 launch」要放在一起比。


# ===========================================================================
# 4. 明确未找到可靠来源的项——**留空，不填**
# ===========================================================================

UNSOURCED = (
    "超越函数（`exp2f` / `__log2f` / 倒数 / 平方根）在 CC 8.0 / 9.0 上每 SM 每周期的"
    "吞吐：CUDA C++ Programming Guide 的 \"Throughput of Native Arithmetic "
    "Instructions\" 表在各版本间口径差异很大（CC 2.x 档与 CC 3.5+ 档给出的量级不同），"
    "本次未能核实到当前版本的确切取值，因此**不写进本模块**。"
    "正文因此不用「exp 的吞吐是 FMA 的几分之一」这类数字，"
    "softmax 的瓶颈判定改用算术强度（只用 HBM 带宽与 FP32 峰值两个已入库的规格值）。",
    "H100 SXM5 的官方 boost 时钟：官方未公布，只能用 67 TFLOPS ÷ (132×128×2) 反推",
    "给定 kernel 在某台机器上的实际耗时：本仓库不复核其测量口径，正文不引用",
)
# 处理方式：正文写「官方未披露」或直接回避，而不是填第三方聚合站的数字。

"""tensor-core 主题的共享事实源。

本模块**只放从官方来源抄来的硬件规格常量与规格表格**，不放任何推导值。
所有推导都在 01~05 五个编号程序里做，本模块不重复。

四条硬性约定（见 CONTRIBUTING §2 与本主题的实现规格）：

  1. **没有 `# 来源：` 注释的常量不许写进本模块。**
  2. **常量名必须编码口径**，使误用可见。算力一律带 `_DENSE_SPARSE` 后缀并
     **存成 `(dense, sparse_2to4)` 二元组**，不存裸数字——官方材料的书写顺序
     自己都不统一（Blackwell 技术简报用 `Dense/Sparse`，DGX B200/B300 datasheet
     用 `sparse|dense`），存成裸数字必然被读错一次。
  3. **找不到可靠来源的项不写进来**（如 Blackwell 每 SM 的 tensor core 数、
     tf32 稀疏的 metadata 尺寸）。该写「官方未披露」的地方正文写「官方未披露」。
  4. **规格表格原样转写**。fragment 布局、形状表这类「规格是一张表」的东西，
     本模块存的是 PTX 原文表格的逐条转写（表达式原样保留在注释里），不是转写
     之后的结论——否则程序里验证的就成了笔者自己的再编码，而不是规格本身。

来源等级（沿用本域约定）：**A** = NVIDIA 官方规范/白皮书/datasheet；
**B** = NVIDIA 官方代码或作者为 NVIDIA 的论文；**C** = 第三方逆向/学术。
**只有 A 级可以写进本模块的来源注释。**
"""

# ===========================================================================
# 0. 基础口径
# ===========================================================================

WARP_SIZE = 32
# 来源：CUDA C++ Programming Guide 13.4 §5.1（各 compute capability 的
#       "warp size" 一栏恒为 32）；与本仓库 20-gpu-architecture/execution-model/01 一致

WARPGROUP_WARPS = 4
# 来源：PTX ISA §9.7.15.1（wgmma 的 "warpgroup" 定义为 4 个连续 warp，
#       即 128 个线程；`wgmma.mma_async` 是 warpgroup 级指令）
WARPGROUP_THREADS = WARPGROUP_WARPS * WARP_SIZE
# 来源：**这一条不是抄来的数，是上面两条恒等式的乘积**（4 × 32 = 128）。
#       放在这里是因为「warpgroup = 128 线程」这个说法在规格与文档里都直接用，
#       单独命名比每次写乘法更不容易写错；它的两个输入各自都有来源。
#       除此之外本模块不含任何推导值。

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

# 逐代 HBM 带宽（GB/s，十进制 10^9 B/s，SXM 档）
HBM_BW_GBps_BY_GEN = {
    "V100": 900.0,      # 来源：NVIDIA Volta V100 Whitepaper（HBM2，900 GB/s）
    "A100": 2039.0,     # 来源：NVIDIA A100 Whitepaper 表4（80GB HBM2e，2039 GB/s）
    "H100": 3350.0,     # 来源：NVIDIA Hopper Architecture In-Depth 表3（HBM3，3.35 TB/s）
    "B200": 7700.0,     # 来源：NVIDIA Blackwell Architecture Technical Brief V2.1（HBM3e）
}

# 逐代 SM 数与每 SM 的 FP32 lane 数（**数据中心 SXM 档**）
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
                    # 数据中心线到 Hopper 才第一次有 128 FP32 lane/SM
    "B200": 128,    # 来源：NVIDIA Blackwell Architecture Technical Brief V2.1
}
FP64_LANES_PER_SM_H100 = 64
# 来源：**本次采用的来源里没有直述**。取 64 是因为官方 FP64 非 TC 峰值 34 TFLOPS
#       反解出的时钟（2.012 GHz）与 FP32 向量峰值反解出的时钟（1.983 GHz）接近——
#       这个假设是那条反推的前提，正文必须一起写出来，否则等于偷偷塞了一个数。
FP32_LANES_FOR_PEAK = {"V100": 64, "A100": 64, "H100": 128, "B200": 128}
# 来源：同 FP32_LANES_PER_SM（用于把向量峰值反解成时钟）
# 说明：本表与 FP32_LANES_PER_SM 同值，保留独立名字是为了让「用哪个数反推时钟」
#       在程序里显式可见——这两个口径一旦分叉，反推出的时钟就会变。


# ===========================================================================
# 1. MMA 形状表（规格是一张「形状 × 最低 target」的表）
# ===========================================================================

# 结构：形状名 → (首次可用代际, 最低 target)
# 来源：PTX ISA 9.4 §9.7.16.1 "Matrix Shape" 表 + 同节各指令的 Target ISA Notes
# 口径：**PTX 的 Shape 表是按 PTX ISA 版本组织的，不是按架构组织的**。
#       要判断某形状属于哪一代，必须与 Target ISA Notes（"requires sm_XX or higher"）
#       交叉比对——只抄 Shape 表会得出「Volta 支持 m16n8k4」这类错误结论。
MMA_SHAPES = {
    # --- Volta (sm_70)：只有一种形状 ---
    "m8n8k4.f16":        ("Volta", "sm_70"),
    # --- Turing (sm_75)：加了整数形状，f16 首次到 k8 ---
    "m16n8k8.f16":       ("Turing", "sm_75"),
    "m8n8k16.s8":        ("Turing", "sm_75"),
    "m8n8k32.s4":        ("Turing", "sm_75"),
    "m8n8k128.b1":       ("Turing", "sm_75"),
    # --- Ampere (sm_80)：f16/bf16 到 k16，tf32 首次出现 ---
    "m16n8k8.bf16":      ("Ampere", "sm_80"),
    "m16n8k16.f16":      ("Ampere", "sm_80"),
    "m16n8k16.bf16":     ("Ampere", "sm_80"),
    "m16n8k4.tf32":      ("Ampere", "sm_80"),
    "m16n8k8.tf32":      ("Ampere", "sm_80"),
    "m16n8k16.s8":       ("Ampere", "sm_80"),
    "m16n8k32.s8":       ("Ampere", "sm_80"),
    # --- Hopper (sm_90a)：warpgroup 级、m 固定 64、N 可变 ---
    "m64nNk16.f16":      ("Hopper", "sm_90a"),
    "m64nNk16.bf16":     ("Hopper", "sm_90a"),
    "m64nNk8.tf32":      ("Hopper", "sm_90a"),
    "m64nNk32.e4m3":     ("Hopper", "sm_90a"),
    "m64nNk32.s8":       ("Hopper", "sm_90a"),
    # --- Blackwell (sm_100a/f)：tcgen05，M 可取 64 / 128 / 256 ---
    "tcgen05.mNk.f16":   ("Blackwell", "sm_100a"),
}
# 三条必须写进正文的修正（均由上表 + Target ISA Notes 直接读出）：
#   · `m16n8k4` **没有 fp16 版本**。k4 的 m16n8 只有 .tf32（sm_80+）与 .f64（sm_90+）。
#   · A100 **没有 FP8 MMA**：e4m3/e5m2 的 mma 需 sm_89+。A100 只有 f16/bf16/tf32/int8。
#   · A100 的 f64 mma 只有 m8n8k4；m16n8k4/k8/k16 的 f64 需 sm_90+。

MMA_F64_SHAPES = ("m8n8k4", "m16n8k4", "m16n8k8", "m16n8k16")
# 来源：PTX ISA 9.4 §9.7.16.1 与各形状的 Target ISA Notes
MMA_F64_MIN_TARGET = {"m8n8k4": "sm_80", "m16n8k4": "sm_90",
                      "m16n8k8": "sm_90", "m16n8k16": "sm_90"}
# 口径：A100 只能做 m8n8k4 的 f64 mma；其余三个要 sm_90+。

# wgmma 的 N 取值范围
WGMMA_N_MIN = 8
WGMMA_N_MAX = 256
WGMMA_N_STEP = 8
# 来源：PTX ISA §9.7.15.4.1 wgmma 形状枚举（.m64n8k16 … .m64n256k16，
#       N 从 8 到 256、步长 8；f16/bf16 的 K=16，tf32 的 K=8）
WGMMA_M = 64
# 来源：同上（wgmma 的 M 恒为 64，不由调用者选择）

TCGEN05_M_VALUES = (64, 128, 256)
TCGEN05_N_STEP = 8
# 来源：PTX ISA §9.7.18 tcgen05.mma 形状小节（M ∈ {64, 128, 256}，N 步长 8）
# 口径警告：**M=256 的形状与 TMEM 每个分配只有 128 lane 这两件事如何对应，
#           本次采用的来源里没有说明，正文回避，只做 M=64 / 128 的算术。**

# 「加速变体」target：只在该精确架构上可用，不像 sm_90 那样向前兼容家族成员
ACCELERATED_TARGETS = ("sm_90a", "sm_100a", "sm_100f")
# 来源：PTX ISA 9.4 §9.7.15 / §9.7.18 的 Target ISA Notes 与 PTX ISA 开头
#       对 "a"/"f" 后缀的定义


# ===========================================================================
# 2. fragment 布局（A 级，PTX ISA §9.7.16.5.8 与 §9.7.16.5.3）
# ===========================================================================
#
# 通用约定（PTX 在几乎每个 fragment 小节逐字重复）：
#
#     groupID           = %laneid >> 2
#     threadID_in_group = %laneid % 4
#
# 下面每个 fragment 存成一张 (行参数, 列参数) 的**逐条转写表**，
# 表里的每个元组对应 PTX 原文表格里的一行。行/列的表达式写在紧邻的注释里。

# --- mma.m16n8k16 .f16/.bf16 的 A：16×16，每 lane 4 个 .f16x2 寄存器 ---
# 来源：PTX ISA §9.7.16.5.8 "Matrix Fragments for mma.m16n8k16 with floating point type"
# 口径：f16 或 bf16 输入、f32 累加；A 为 row-major；sm_80+（f16 形状自 Turing 起）
# 转写：row = groupID + 8*[i in {2,3,6,7}]，col = 2*threadID_in_group + (i&1) + 8*[i>=4]
#       表里存 (行参数, 列偏移)：行参数 0 → groupID，1 → groupID+8
M16N8K16_F16_A_ELEMENTS = ((0, 0), (0, 1), (1, 0), (1, 1),
                           (0, 8), (0, 9), (1, 8), (1, 9))
# 打包规则：每 2 个相邻列的元素进一个 .f16x2 寄存器 → reg0=a0a1, reg1=a2a3, reg2=a4a5, reg3=a6a7
M16N8K16_F16_A_REGS_PER_LANE = 4
M16N8K16_F16_A_ELEMS_PER_LANE = 8
# 来源：同上（"A vector expression containing four .f16x2 registers"）

# --- mma.m16n8k16 .f16/.bf16 的 B：16×8（k×n），每 lane 2 个 .f16x2 寄存器 ---
# 来源：PTX ISA §9.7.16.5.8
# 转写：col = groupID；row = 2*threadID_in_group + (i&1) + 8*[i>=2]
#       表里存 (行偏移, 寄存器号)：b0→(0,reg0)，b1→(1,reg0)，b2→(8,reg1)，b3→(9,reg1)
M16N8K16_F16_B_ELEMENTS = ((0, 0), (1, 0), (8, 1), (9, 1))
M16N8K16_F16_B_REGS_PER_LANE = 2
M16N8K16_F16_B_ELEMS_PER_LANE = 4
# 来源：同上（"A vector expression containing two .f16x2 registers"）

# --- mma.m16n8k16 .f16/.bf16 的 C/D：16×8，每 lane 4 个 .f32 寄存器 ---
# 来源：PTX ISA §9.7.16.5.8（C/D 表）
# 转写：row = groupID + 8*[i in {2,3}]，col = 2*threadID_in_group + (i&1)
#       表里存 (行偏移, 列偏移)：c0→(0,0)，c1→(0,1)，c2→(8,0)，c3→(8,1)
M16N8K16_F32_C_ELEMENTS = ((0, 0), (0, 1), (8, 0), (8, 1))
M16N8K16_F32_C_REGS_PER_LANE = 4
M16N8K16_F32_C_ELEMS_PER_LANE = 4
# 来源：同上（"A vector expression containing four .f32 registers"）

# --- mma.m16n8k8 .tf32 的 A：16×8，每 lane 4 个 .b32 寄存器（每个 1 个 tf32）---
# 来源：PTX ISA §9.7.16.5.3 "Matrix Fragments for mma.m16n8k8 with .tf32 floating point type"
# 转写：row = groupID + 8*[i in {1,3}]，col = threadID_in_group + 4*[i in {2,3}]
#       表里存 (行参数, 列参数)：行参数 0 → groupID，1 → groupID+8；列参数 0 → t，1 → t+4
# ⚠ 形状是 **m16n8k8**，不是 m16n8k16：PTX 的 tf32 形状只有 m16n8k4 与 m16n8k8，
#   **没有 k16 的 tf32**。「照 fp16 的 k16 类推」在这里第一步就错了。
M16N8K8_TF32_A_ELEMENTS = ((0, 0), (1, 0), (0, 1), (1, 1))
M16N8K8_TF32_A_REGS_PER_LANE = 4
M16N8K8_TF32_A_ELEMS_PER_LANE = 4
# 来源：同上（"A vector expression containing four .b32 registers, each containing
#       one .tf32 element"）

# --- mma.m16n8k8 .tf32 的 B：8×8（k×n），每 lane 2 个 .b32 寄存器 ---
# 来源：PTX ISA §9.7.16.5.3
# 转写：col = groupID；row = threadID_in_group + 4*[i==1]
#       表里存 (行偏移, 0)：b0→(0,0)，b1→(4,0)
M16N8K8_TF32_B_ELEMENTS = ((0, 0), (4, 0))
M16N8K8_TF32_B_REGS_PER_LANE = 2
M16N8K8_TF32_B_ELEMS_PER_LANE = 2
# 来源：同上（"A vector expression containing two .b32 registers"）

# --- ldmatrix 的物理约束：fragment 规则的来历 ---
LDMATRIX_ELEMENT_BITS = 16
LDMATRIX_BYTES_PER_ROW = 16
# 来源：PTX ISA §9.7.14.4.15 的 ldmatrix 小节（该指令只吃 .b16 元素，
#       一个 8×8 矩阵的一行是 8 个元素 × 2 字节 = 16 字节）
LDMATRIX_MATRIX_ROWS = 8
# 来源：同上（.m8n8 形状，即每个矩阵是 8×8）
LDMATRIX_X4_MATRICES = 4
# 来源：同上（.x4 变体一次加载 4 个 8×8 矩阵）
LDMATRIX_BYTES_PER_LANE = 4
# 来源：同上（每条 lane 拿到 4 字节 = 2 个 .b16；一个 8×8 矩阵共 128 字节，
#       32 lane × 4 B = 128 B，恰好整除）

# --- wmma API 的两个约束（A 级，CUDA C++ Programming Guide §7.24）---
WMMA_FRAGMENT_ALIGNMENT_BYTES = 32
# 来源：CUDA C++ Programming Guide §7.24（wmma 的 mptr "必须按 256 bit
#       （32 字节）对齐"）
WMMA_LDM_MULTIPLE_HALF = 8
WMMA_LDM_MULTIPLE_FLOAT = 4
# 来源：同上（ldm 必须是 8（__half）/ 4（float）的倍数——两者都是 16 字节的倍数）
# ⚠ 两套口径别混：CUDA C++ API 层是「mptr 32 字节对齐 + ldm 是 8/4 的倍数」；
#   PTX `wmma.load` 层是「起始地址按对应 fragment 的字节大小对齐」。

WMMA_MAPPING_IS_UNSPECIFIED = True
# 来源：PTX ISA 的 wmma 小节与 CUDA C++ Programming Guide §7.24 均明确写
#       fragment 的「元素 → lane」映射是 **"unspecified and target architecture
#       dependent"**。这是本主题最重要的一条否定性规格：
#       **只有 mma.sync 的 fragment 规则可推导可验证；用 wmma 做这类推导会全错。**

WMMA_SUPPORTED_COMBINATIONS = (
    ("__half -> __half -> float/__half", "16x16x16 / 32x8x16 / 8x32x16"),
    ("char -> int", "8x8x32 等整数形状"),
    ("__nv_bfloat16 -> float", "16x16x16 等"),
    ("precision::tf32 -> float", "**仅 16x16x8**"),
    ("double -> double", "**仅 8x8x4**"),
)
# 来源：CUDA C++ Programming Guide §7.24（wmma 的类型组合与形状表）


# ===========================================================================
# 3. 每 SM 的 tensor core 数与 FP16 FMA/clock/SM
# ===========================================================================

TENSOR_CORES_PER_SM = {"V100": 8, "A100": 4, "H100": 4, "B200": None}
# 来源：V100 每 SM 8 个（NVIDIA Volta V100 Whitepaper）；A100 每 SM 4 个
#       （NVIDIA A100 Whitepaper）；H100 每 SM 4 个（Hopper Architecture In-Depth）
# 口径：这是**厂商定义里的「第 N 代 Tensor Core」个数**，不是物理上可数的单元——
#       NVIDIA 每代都在改这个定义。B200 官方未披露，置 None 而不是填第三方数字。

FP16_FMA_PER_CLK_PER_SM_DENSE = {
    "V100": 512,    # 8 个 TC × 64 FMA/clk/TC（A100 Whitepaper 的 Volta 栏原文）
    "A100": 1024,   # 4 个 TC × 256 FMA/clk/TC（A100 Whitepaper 原文）
    "H100": 2048,   # A100 Whitepaper 的 Volta 栏原文
    "B200": None,   # 官方未披露
}
# 来源：NVIDIA A100 Tensor Core GPU Architecture Whitepaper 的原文段落
#       （同段给出「Volta/Turing 每 SM 8 个 TC，每个 64 FP16/FP32 FMA per clock；
#        A100 每 SM 4 个 TC，每个 256 FMA per clock，合计 1024 dense FP16/FP32
#        FMA per clock，per-SM 算力是 Volta/Turing 的 2 倍」）
# 口径：**这是「每 SM 合计」的 FP16 FMA/clock，不是每个 tensor core 的**。
#       H100 的值由官方 FP16 dense 峰值反推校验，见 01 号程序第 5 节——
#       **反推出的时钟与 FP32 向量峰值反推出的时钟不一致（差约 10%）**，
#       所以正文只用它做「量级」判断，不用它反推每 SM 的 FMA 数。

A100_WHITEPAPER_TENSOR_CORE_QUOTE = (
    "Volta and Turing have eight Tensor Cores per SM, with each Tensor Core "
    "performing 64 FP16/FP32 mixed-precision fused multiply-add (FMA) operations "
    "per clock. The A100 SM includes new third-generation Tensor Cores that each "
    "perform 256 FP16/FP32 FMA operations per clock. A100 has four Tensor Cores "
    "per SM, which together deliver 1024 dense FP16/FP32 FMA operations per clock, "
    "a 2x increase in computation horsepower per SM compared to Volta and Turing."
)
# 来源：NVIDIA A100 Tensor Core GPU Architecture Whitepaper（原文逐字抄录）
# 用途：同时说明「数量从 8 降到 4 而每 SM 算力翻倍」——这是本主题最好的失败模式素材


# ===========================================================================
# 4. 峰值算力（**一律 dense（不含 2:4 稀疏）**，SXM 档）
# ===========================================================================
#
# 每个常量是 (dense, sparse_2to4) 二元组。稀疏值没有官方来源的写 None。
# 存成二元组而不是裸数字的理由：官方材料的书写顺序自己都不统一，
# 且**同一代里不同精度会撞车**（A100 的 FP16 dense 312 与 TF32 sparse 312 同值）。

TFLOPS_FP32_VECTOR_DENSE = {
    "V100": 15.7,   # 来源：NVIDIA Volta V100 Whitepaper
    "A100": 19.5,   # 来源：NVIDIA A100 Whitepaper 表1
    "H100": 67.0,   # 来源：NVIDIA Hopper Architecture In-Depth 表3 / H100 datasheet
    "B200": 75.0,   # 来源：NVIDIA Blackwell Architecture Technical Brief V2.1
}
# 口径：非 tensor 的 FP32 向量峰值。H100 起数据中心线每 SM 128 个 FP32 lane（A100 是 64），
#       这一栏的 3.44× 跃升就来自这里。

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
# 口径（本模块最关键的一栏）：
#   · **表中的 dense 值一律不含 2:4 稀疏**。NVIDIA 产品页写「H100 FP16 Tensor Core
#     1,979 TFLOPS」——那个 1,979 是**带 2:4 稀疏**的，dense 只有 989.5。
#   · **跨精度撞车**：A100 的 FP16 dense(312) == TF32 sparse(312)；H100 的
#     FP16 dense(989.5) == TF32 sparse(989)；B200 的 FP4 dense(9000) == FP8 sparse(9000)。
#     看到 312 / 989 / 9000 必须先问「哪个精度、dense 还是 sparse」。
#   · H100 的 TF32 sparse 在产品页写作 989，与 FP16 dense 的 989.5 不是同一个数。

FP64_TFLOPS_DENSE = {"V100": 7.8, "A100": 19.5, "H100": (34.0, 67.0), "B200": 37.0}
# 来源：同上各代官方材料。H100 存成二元组是因为官方给了两栏
#       （非 TC 34 / TC 67，相差 2 倍）——引用时必须指明哪一栏。
# 口径：**H100 的 FP64 高于 B200**（67 vs 37），「新一代所有精度都更快」是错的。


# ===========================================================================
# 5. wgmma 与 tcgen05 的规格（A 级）
# ===========================================================================

WGMMA_B_MUST_BE_SMEM = True
WGMMA_A_MAY_BE_REG_OR_SMEM = True
# 来源：PTX ISA §9.7.15（wgmma 的操作数：B 必须来自 shared memory 描述符；
#       A 可以是寄存器也可以是 shared memory 描述符）
# ⚠ 常见错误：「wgmma 的 A、B 都必须在 SMEM」是错的。

WGMMA_REQUIRES_WHOLE_WARPGROUP = True
WGMMA_ALIGNED_QUALIFIER_MANDATORY = True
# 来源：PTX ISA §9.7.15 的 `.aligned` 限定符（"all threads in the warpgroup must
#       execute the same instruction"；条件分支里若不能保证 warpgroup 内条件一致，
#       行为未定义）与 `.sync` 限定符
# 口径：这就是「wgmma 的发射粒度是 128 线程」的来源——4 个 warp 各发一次。

TCGEN05_ISSUE_BY_SINGLE_THREAD = True
TCGEN05_ISSUE_QUOTE = (
    "An issue from a single thread in the current CTA would initiate the base operation."
)
# 来源：PTX ISA §9.7.18（tcgen05.mma 的发射模式，原文逐字抄录）
# 口径：与 wgmma 的「128 线程都要执行」相反——**发射粒度 1 个线程**。

TCGEN05_D_IN_TMEM_ONLY = True
TCGEN05_A_IN_TMEM_OR_SMEM = True
TCGEN05_B_IN_SMEM_ONLY = True
# 来源：PTX ISA §9.7.18（tcgen05.mma 的操作数位置：D 必须在 tensor memory；
#       A 可在 tensor memory 或 shared memory；B 必须来自 shared memory）
# 口径：与 wgmma 的差异就在 D——wgmma 的 D 在寄存器，tcgen05 的 D 在 TMEM。

TMEM_LANES_PER_CTA = 128
TMEM_MAX_COLUMNS_PER_CTA = 512
TMEM_CELL_BITS = 32
TMEM_ALLOC_GRANULARITY_COLUMNS = 32
TMEM_ALLOC_LEGAL_COLUMNS = (32, 64, 128, 256, 512)
TMEM_ADDRESS_LANE_SHIFT = 16
# 来源：PTX ISA §9.7.18 的 tensor memory 小节（TMEM 是 128 lane × 至多 512 列的
#       二维地址空间，每 cell 32 bit；列是动态分配的，分配粒度 32 列，合法的
#       n_cols 取值是 32/64/128/256/512；分配一列即占满全部 128 个 lane 位置；
#       32 位地址的 bit[31:16] 是 lane、bit[15:0] 是 column）
# 口径：**PTX 给的口径是 per CTA，不是 per SM**。128 × 512 × 4 B = 256 KB，
#       恰好等于每 SM 的容量；一个 SM 上多个 CTA 如何仲裁 TMEM，本次采用的
#       来源没有说明——正文回避这一点，只做 per CTA 的算术。

TMEM_WARP_ACCESS_FRACTION = 4
# 来源：PTX ISA §9.7.18（TMEM 的 128 个 lane 按 32 个一组分成 4 块，一个 warp
#       只能访问本 warpgroup 内由 warp id 决定的那 32 个 lane；**要覆盖全部
#       128 lane 必须有整个 warpgroup**）


# ===========================================================================
# 6. 结构化稀疏（A 级，PTX ISA §9.7.16.6.1）
# ===========================================================================

SPARSE_DEFINITION_QUOTE = (
    "sparsity is expected to be at 2:4 granularity, i.e. each 4-element vector "
    "(i.e. a sub-chunk of 4 consecutive elements) of a matrix row contains 2 zeros. "
    "Index of each non-zero element in a sub-chunk is stored in the metadata operand. "
    "Values 0b0000, 0b0101, 0b1010, 0b1111 are invalid."
)
# 来源：PTX ISA §9.7.16.6.1（原文逐字抄录）

SPARSE_GROUP_ELEMENTS = 4
SPARSE_NONZEROS_PER_GROUP = 2
SPARSE_INDEX_BITS = 2
SPARSE_META_BITS_PER_GROUP = 4
SPARSE_INVALID_METADATA_VALUES = (0b0000, 0b0101, 0b1010, 0b1111)
# 来源：同上。**每个 4 元素组用两个 2-bit 索引 = 共 4 bit，不是 2 bit**——
#       这是本主题最容易少算一半的一条。

SPARSE_APPLIES_TO_A_ONLY = True
# 来源：PTX ISA §9.7.16.6.1 与 mma.sp 的操作数定义（稀疏只施加在 A 上，
#       B 与 C/D 保持稠密 → 它是 SpMM，不是 SpGEMM）
SPARSE_GRANULARITY_AXIS = "K 维（PTX 原文说的是 a matrix row，且子块是连续的 4 个元素）"
# 来源：同上
SPARSE_CANNOT_CROSS_ROW = True
# 来源：同上（约束逐行施加，不同行的非零位置可以完全不同）

SPARSE_VARIANTS = {
    "f16/bf16/int8": "2:4",
    "tf32":          "1:2",
    "u4/s4 与 fp4":  "4:8 paired",
}
# 来源：PTX ISA §9.7.16.6.1 的稀疏变体表
# 口径：tf32 的「1:2」是**表述**上的不同（每 2 个元素 1 个非零），
#       数值上仍是每 4 个元素 2 个非零；但这个变体的 metadata 尺寸本次来源未给出，
#       正文不对它做压缩率算术。

SPARSE_SHAPE_K_MULTIPLIER = 2
# 来源：PTX ISA §9.7.16.6.1（稀疏形状的 K 一律是对应稠密形状的 2 倍：
#       m16n8k16 + 2:4 → m16n8k32）

SPARSE_M16N8K32_A_CHUNKS_PER_LANE = 4
# 来源：PTX ISA §9.7.16.6.2 "Matrix Fragments for sparse mma.m16n8k32" 的 A 映射
#       （row = groupID + 8*[i in {2,3,6,7}]；col 的起点 = 4*threadID_in_group
#        + 16*[i>=4]，每个 ai 覆盖 **4 个连续列** = 一个 4 元素子块）
# 转写：每 lane 覆盖 4 个子块，表里存 (行参数, 列块参数)：
#       row = groupID + 8*行参数；列块起点 = 4*threadID_in_group + 16*列块参数
#       一个寄存器对应一个子块（该子块里的 2 个非零值都装在这一个寄存器里）
SPARSE_M16N8K32_A_CHUNKS = ((0, 0), (1, 0), (0, 1), (1, 1))
SPARSE_M16N8K32_A_REGS_PER_LANE = 4
SPARSE_M16N8K32_A_NONZEROS_PER_LANE = 8
# 来源：同上（"A vector expression containing four .b32 registers, each holding
#       two non-zero .f16/.bf16 elements"）

SPARSE_ORDERED_METADATA_REQUIRED_SINCE = "PTX ISA 8.5"
SPARSE_ORDERED_METADATA_RULE = (
    "mma.sp::ordered_metadata requires the indices in the metadata to be sorted "
    "in increasing order starting from the LSB; otherwise behavior is undefined."
)
# 来源：PTX ISA §9.7.16.6 的 `::ordered_metadata` 限定符说明
# 口径：**没有这个限定符时 metadata 的顺序不受约束，但在某些未来架构上
#       性能可能显著下降**。做手写稀疏内核时建议总是带上它。


# ===========================================================================
# 7. 外部引用值（**不是本次推导的产物，本主题的程序不产出它们**）
# ===========================================================================
#
# 稀疏 GEMM 对稠密 GEMM 的**实测**加速比。这些是外部来源报出的观测区间，
# 本文只转述、无法复核，也不对它们的测量口径负责。它们的作用只有一个：
# 给 05 号程序推导出的区间提供一个对照。

SPARSE_SPEEDUP_OBSERVED = (
    ("NVIDIA cuSPARSELt 官方博客", "A100 / BERT-Large FP16 各层", (1.3, 1.6)),
    ("NVIDIA TensorRT 官方博客", "A100 / ResNeXt-101 端到端", (1.2, 1.2)),
    ("arXiv:2402.10876 (IEEE TC)", "大 GEMM 1.67x；BERT/NMT 平均 1.25x；"
                                   "小 CNN GEMM 无加速", (1.0, 1.67)),
)
# 来源：三条独立来源，见正文「参考」小节
# 口径：**三行分属两种口径，不得混称**——第 1 行是「各层」（层级）、第 3 行是「内核级」、
#       第 2 行是「端到端」。它们都不是算力倍数，是实测倍数。
#       ⚠ 理论 2x 在真实模型上要分两档说：**层级/内核级约 1.3x–1.7x**；
#         **端到端约 1.2x**，且**小 batch 可能低于 1x**（PyTorch 官方教程
#         BERT-base/SQuAD bs=4 实测 0.71x，更慢）。
#       把「层级/内核级」的区间说成「端到端」是**不同口径混称**——这是本条注释
#       此前犯过的错（见 130-quantization/sparsity-and-distillation 的跨域订正）。

SPARSE_WHITEPAPER_QUOTE = (
    "Speedups that 2:4 sparse matrix multiplications achieve over dense "
    "multiplications depend on several factors, such as arithmetic intensity "
    "and GEMM dimensions. ... As larger GEMMs tend to have higher arithmetic "
    "intensity, they get closer to the 2x speedup."
)
# 来源：NVIDIA 稀疏 GEMM 论文（作者为 NVIDIA，B 级）


# ===========================================================================
# 8. 明确未找到可靠来源的项——**留空，不填**
# ===========================================================================

UNSOURCED = (
    "Blackwell 每 SM 的 tensor core 数量：官方未披露（第三方说 4/SM，不可作官方引用）",
    "B200 的 SM 总数 148：由官方四种精度峰值反推同时吻合，但**反推不是来源**",
    "H100 SXM5 与 B200 的官方 boost 时钟：官方未公布，只能用峰值反推并标注",
    "稀疏 wgmma.sp 的 SASS 助记符：未找到任何真实转发储，正文回避",
    "HMMA.884（Volta SASS opcode）命名规则：NVIDIA 无官方 Volta SASS opcode 表",
    "Ampere 稀疏 MMA 的 SASS 助记符：仅第三方逆向库有，标注「社区实测，非官方」",
    "tf32 稀疏变体（1:2）的 metadata 尺寸：本次来源未给出，正文不做它的压缩率算术",
    "tcgen05 的 M=256 形状与「TMEM 每个分配只有 128 lane」如何对应：来源未说明",
    "「cuSPARSELt 对 FP16 2:4 用 4-bit 索引」：该说法无可靠来源，实测源码是 2 bit",
)
# 处理方式：正文写「官方未披露」或直接回避，而不是填第三方聚合站
# （topcpu.net、cpudb.cc、technical.city、fpsbench.com 等）的数字。

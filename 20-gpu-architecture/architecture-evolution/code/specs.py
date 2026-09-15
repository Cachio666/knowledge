"""architecture-evolution 主题的共享事实源。

本模块**只放从官方来源抄来的硬件规格常量**，不放任何推导值。
所有推导都在 01 / 02 / 03 三个编号程序里做，本文档不重复。

三条硬性约定（见 CONTRIBUTING §2 与本主题的实现规格）：

  1. **没有 `# 来源：` 注释的常量不许写进本模块。**
  2. **常量名必须编码口径**，使误用可见。用 `H100_HBM_BW_GBps`，不用 `H100_BW`。
     精度一律带 `_TFLOPS_DENSE` / `_TFLOPS_SPARSE_2TO4` 后缀，档位一律带 `_SXM`。
  3. **找不到可靠来源的项不写进来**（如 B200 的每 SM tensor core 数、B200 的 L2 容量、
     B200 的 die 面积与 boost 时钟）。该写「官方未披露」的地方，正文写「官方未披露」，
     而不是在这里塞一个第三方聚合站的数。

逐代数据一律为**数据中心 SXM 档**：P100 SXM2 / V100 SXM2 / A100 SXM4 80GB /
H100 SXM5 / B200 SXM。**不得混入 PCIe 档**（同代 PCIe 档的 SM 数、TDP、带宽都不同）。

`GENERATIONS` 之后的所有 `_BY_GEN` 字典都只是上面那些具名常量的**聚合视图**，
不含任何新数——它们存在的唯一理由是让三个编号程序能按代迭代。
"""

# ===========================================================================
# 1. 代际标识
# ===========================================================================

# 来源：本模块开头的口径约定（数据中心 SXM 档）。
# 注意 Turing 不在其中：Turing 没有数据中心 SXM 档（只有 PCIe 70 W 的 T4），
# 进趋势表会引入不同口径的封装，故本主题的所有逐代表都不含 Turing。
GENERATIONS = ("P100", "V100", "A100", "H100", "B200")

YEAR_BY_GEN = {"P100": 2016, "V100": 2017, "A100": 2020, "H100": 2022, "B200": 2024}
# 来源：NVIDIA Tesla P100 / V100 / A100 / H100 / Blackwell 各代官方发布年。


# ===========================================================================
# 2. 芯片级规格
# ===========================================================================

# --- P100 SXM2（Pascal，CC 6.0）---
P100_SM_COUNT = 56
# 来源：NVIDIA Tesla P100 Whitepaper，GP100 规格页（56 SM / 3584 FP32 core）
P100_FP32_LANES_PER_SM = 64
# 来源：同上（3584 ÷ 56 = 64，白皮书按整卡给出 3584 个 FP32 core）
P100_TENSOR_CORES_PER_SM = 0
# 来源：Pascal 架构无 Tensor Core（首次引入见 §4 VOLTA 段）。
P100_TDP_W = 300
# 来源：NVIDIA Tesla P100 Whitepaper，SXM2 板卡规格页（300 W）
P100_TRANSISTORS_B = 15.3
# 来源：NVIDIA Tesla P100 Whitepaper（153 亿个晶体管）
P100_PROCESS_NM = 16
# 来源：NVIDIA Tesla P100 Whitepaper（TSMC 16 nm FinFET）
# P100 的带宽有两个官方来源互相矛盾，两个都记，取用时必须注明用的是哪一个。
P100_HBM_BW_GBps_A100WP = 720.0
# 来源：NVIDIA A100 Whitepaper 表4「Comparison of NVIDIA Data Center GPUs」
P100_HBM_BW_GBps_VOLTAWP = 732.0
# 来源：NVIDIA Volta V100 Whitepaper 正文（P100 规格小节）
# 口径分歧：同一块 P100 SXM2，两份官方文档差 12 GB/s（1.7%）。正文并列、不做裁决。

# --- V100 SXM2（Volta，CC 7.0）---
V100_SM_COUNT = 80
V100_FP32_LANES_PER_SM = 64
# 来源：NVIDIA Volta V100 Whitepaper，GV100 规格页（80 SM / 5120 FP32 core）
V100_TENSOR_CORES_PER_SM = 8
# 来源：NVIDIA Volta V100 Whitepaper（每 SM 8 个第二代 Tensor Core）
V100_TDP_W = 300
V100_TRANSISTORS_B = 21.1
V100_PROCESS_NM = 12
# 来源：NVIDIA Volta V100 Whitepaper，SXM2 板卡规格页（300 W / 211 亿晶体管 / TSMC 12 nm）
V100_HBM_BW_GBps = 900.0
# 来源：NVIDIA Volta V100 Whitepaper（HBM2，900 GB/s）；与 A100 Whitepaper 表4 一致

# --- A100 SXM4 80GB（Ampere，CC 8.0）---
A100_SM_COUNT = 108
A100_FP32_LANES_PER_SM = 64
# 来源：NVIDIA A100 Whitepaper 表1 / 表4（108 SM / 6912 FP32 core）
# 注意：GA100 是 64 lane/SM；**128 lane/SM 是消费级 GA10x 的事实**，
# 见 NVIDIA Ada GPU Architecture Whitepaper 对 GA10x 与 GA100 的对照。
A100_TENSOR_CORES_PER_SM = 4
# 来源：NVIDIA A100 Whitepaper（每 SM 4 个第三代 Tensor Core）
A100_TDP_W = 400
A100_TRANSISTORS_B = 54.2
A100_PROCESS_NM = 7
# 来源：NVIDIA A100 Whitepaper，SXM4 板卡规格页（400 W / 542 亿晶体管 / TSMC 7 nm）
A100_HBM_BW_GBps = 2039.0
# 来源：NVIDIA A100 Whitepaper 表4（80GB HBM2e 档，2039 GB/s）

# --- H100 SXM5（Hopper，CC 9.0）---
H100_SM_COUNT = 132
H100_FP32_LANES_PER_SM = 128
# 来源：NVIDIA Hopper Architecture In-Depth 官方博客表3（132 SM / 16896 FP32 core）
# 关键：数据中心线到 Hopper 才第一次有 128 FP32 lane/SM（A100 是 64）。
H100_TENSOR_CORES_PER_SM = 4
# 来源：同上（每 SM 4 个第四代 Tensor Core）
H100_TDP_W = 700
H100_TRANSISTORS_B = 80.0
H100_PROCESS_NM = 4
# 来源：NVIDIA Hopper Architecture In-Depth 官方博客（700 W / 800 亿晶体管 / TSMC 4N）
H100_HBM_BW_GBps = 3350.0
# 来源：NVIDIA Hopper Architecture In-Depth 官方博客表3（HBM3，3.35 TB/s）
# 版本分歧：Hopper 发布时的官方博客同时给出 3000 GB/s 并明写 "Not finalized"；
# 最终产品页与上表是 3350 GB/s。本模块取最终值，正文引用时带版本说明。

# --- B200 SXM（Blackwell，CC 10.0）---
B200_SM_COUNT = 148
# 来源：**非官方直述**——官方未披露 B200 的 SM 数。本值由官方 FP32 峰值 75 TFLOPS
# 反推 + Hot Chips 2024 披露交叉验证得到（148 × 128 × 2 × 1.98 GHz ≈ 75 TFLOPS）。
# 第三方聚合站给的 132 与官方 FP32 峰值反推出的 2.22 GHz 时钟矛盾，不采信。
B200_FP32_LANES_PER_SM = 128
# 来源：NVIDIA Blackwell Architecture Technical Brief V2.1（每 SM 128 个 FP32 lane）
B200_TENSOR_CORES_PER_SM = None
# 来源：**官方未披露**。置 None 而不是填一个第三方数字——正文写「官方未披露」。
B200_TDP_W = 1000
B200_TRANSISTORS_B = 208.0
# 来源：NVIDIA Blackwell Architecture Technical Brief V2.1（1000 W）
# 口径警告：2080 亿是**双 die 封装的总数**（单 die 1040 亿），而 P100~H100 一栏
# 是单 die 数。混进来直接读会误判成「一代涨 2.6 倍」。
B200_PROCESS_NM = 4
# 来源：NVIDIA Blackwell Architecture Technical Brief V2.1（TSMC 4NP）
B200_HBM_BW_GBps = 7700.0
# 来源：NVIDIA Blackwell Architecture Technical Brief V2.1（HBM3e，7.7 TB/s）
B200_CLOCK_GHZ_REVERSE_DERIVED = 1.98
# 来源：**反推值，非官方**——官方未披露 B200 的 boost 时钟。由官方 FP32 峰值
# 75 TFLOPS ÷ (148 SM × 128 lane × 2 FLOP) = 1.9797 GHz 得到。
# 本常量放在此处是为了让 02 号程序能复核这个反推，不代表它是规格值。

# --- H100 PCIe（**对照组，不进任何趋势表**；用途见 02 号程序 §10）---
H100_PCIE_SM_COUNT = 114
H100_PCIE_TDP_W = 350
H100_PCIE_HBM_BW_GBps = 2000.0
# 来源：NVIDIA H100 PCIe 产品页规格表（114 SM / 350 W / HBM2e 2.0 TB/s，80 GB）
# 口径：这是**同代另一个封装档**，与 SXM5 不是同一台机器。放在这里的唯一理由是
# 让「SXM 与 PCIe 不能混」这条陷阱的差值可以被程序算出来，而不是写死在正文里。

CHIP_BY_GEN = {
    "P100": {"sm_count": P100_SM_COUNT, "fp32_lanes_per_sm": P100_FP32_LANES_PER_SM,
             "tensor_cores_per_sm": P100_TENSOR_CORES_PER_SM, "tdp_W": P100_TDP_W,
             "transistors_B": P100_TRANSISTORS_B, "process_nm": P100_PROCESS_NM,
             "hbm_bw_GBps": P100_HBM_BW_GBps_A100WP},
    "V100": {"sm_count": V100_SM_COUNT, "fp32_lanes_per_sm": V100_FP32_LANES_PER_SM,
             "tensor_cores_per_sm": V100_TENSOR_CORES_PER_SM, "tdp_W": V100_TDP_W,
             "transistors_B": V100_TRANSISTORS_B, "process_nm": V100_PROCESS_NM,
             "hbm_bw_GBps": V100_HBM_BW_GBps},
    "A100": {"sm_count": A100_SM_COUNT, "fp32_lanes_per_sm": A100_FP32_LANES_PER_SM,
             "tensor_cores_per_sm": A100_TENSOR_CORES_PER_SM, "tdp_W": A100_TDP_W,
             "transistors_B": A100_TRANSISTORS_B, "process_nm": A100_PROCESS_NM,
             "hbm_bw_GBps": A100_HBM_BW_GBps},
    "H100": {"sm_count": H100_SM_COUNT, "fp32_lanes_per_sm": H100_FP32_LANES_PER_SM,
             "tensor_cores_per_sm": H100_TENSOR_CORES_PER_SM, "tdp_W": H100_TDP_W,
             "transistors_B": H100_TRANSISTORS_B, "process_nm": H100_PROCESS_NM,
             "hbm_bw_GBps": H100_HBM_BW_GBps},
    "B200": {"sm_count": B200_SM_COUNT, "fp32_lanes_per_sm": B200_FP32_LANES_PER_SM,
             "tensor_cores_per_sm": B200_TENSOR_CORES_PER_SM, "tdp_W": B200_TDP_W,
             "transistors_B": B200_TRANSISTORS_B, "process_nm": B200_PROCESS_NM,
             "hbm_bw_GBps": B200_HBM_BW_GBps},
}
# 口径提醒：上表中 P100~H100 的 transistors_B 是单 die 数，B200 是双 die 总数。
# 这是本主题最容易误读的一格，正文 §4 专门有一条讲它。


# ===========================================================================
# 3. 逐代峰值算力（一律 dense，不含 2:4 结构化稀疏）
# ===========================================================================

# --- V100 SXM2 ---
V100_FP32_VECTOR_TFLOPS_DENSE = 15.7
V100_FP16_TC_TFLOPS_DENSE = 125.0
# 来源：NVIDIA Volta V100 Whitepaper；与 A100 Whitepaper 表4 一致
# 口径：非 tensor 的 FP32 向量峰值；FP16 为 Tensor Core dense（V100 无稀疏支持）

# --- A100 SXM4 80GB ---
A100_FP32_VECTOR_TFLOPS_DENSE = 19.5
A100_TF32_TC_TFLOPS_DENSE = 156.0
A100_FP16_TC_TFLOPS_DENSE = 312.0
A100_BF16_TC_TFLOPS_DENSE = 312.0
# 来源：NVIDIA A100 Whitepaper 表1 / 表4（FP32 19.5 / TF32 156 / FP16 312 / BF16 312）
A100_FP16_TC_TFLOPS_SPARSE_2TO4 = 624.0
# 来源：NVIDIA A100 产品页（FP16 Tensor Core with sparsity = 624）
# 口径：star 陷阱的对照值——同一台 A100，稀疏口径下的 FP16 是 dense 的两倍。
A100_FP64_TC_TFLOPS_DENSE = 19.5
# 来源：NVIDIA A100 Whitepaper 表1（FP64 Tensor Core 19.5，恰与 FP32 向量同值）

# --- H100 SXM5 ---
H100_FP32_VECTOR_TFLOPS_DENSE = 67.0
H100_TF32_TC_TFLOPS_DENSE = 494.5
H100_FP16_TC_TFLOPS_DENSE = 989.5
H100_BF16_TC_TFLOPS_DENSE = 989.5
H100_FP8_TC_TFLOPS_DENSE = 1979.0
# 来源：NVIDIA Hopper Architecture In-Depth 官方博客表3 / H100 datasheet
# 口径：全部 dense。H100 产品页的 FP16 1979 是**带 2:4 稀疏**的值，dense 是 989.5。
H100_FP16_TC_TFLOPS_SPARSE_2TO4 = 1979.0
# 来源：NVIDIA H100 产品页（FP16 Tensor Core with sparsity = 1979）
H100_FP64_VECTOR_TFLOPS_DENSE = 34.0
H100_FP64_TC_TFLOPS_DENSE = 67.0
# 来源：NVIDIA H100 datasheet 规格表（与本仓库 10-foundations/performance-models/01 一致）
# 口径：H100 的 FP64 有两栏（非 TC 34 / TC 67），相差 2 倍——引用时必须指明哪一栏。

# --- B200 SXM ---
B200_FP32_VECTOR_TFLOPS_DENSE = 75.0
B200_TF32_TC_TFLOPS_DENSE = 1100.0
B200_FP16_TC_TFLOPS_DENSE = 2250.0
B200_BF16_TC_TFLOPS_DENSE = 2250.0
B200_FP8_TC_TFLOPS_DENSE = 4500.0
B200_FP4_TC_TFLOPS_DENSE = 9000.0
# 来源：NVIDIA Blackwell Architecture Technical Brief V2.1 表1（全部 dense）
B200_FP64_TFLOPS_DENSE = 37.0
# 来源：NVIDIA Blackwell Architecture Technical Brief V2.1（FP64 dense 37）
# 口径警告：Blackwell 的官方材料只有一个 FP64 数字，H100 有两个（34 / 67）。
# 「谁的 FP64 更高」取决于拿哪一栏对哪一栏，见 02 号程序第 8 节。

PEAK_TFLOPS_DENSE_BY_GEN = {
    "P100": {},                      # 官方材料未给出与其他代可比的 dense 峰值，留空
    "V100": {"FP32_VECTOR": V100_FP32_VECTOR_TFLOPS_DENSE,
             "FP16_TC": V100_FP16_TC_TFLOPS_DENSE},
    "A100": {"FP32_VECTOR": A100_FP32_VECTOR_TFLOPS_DENSE,
             "TF32_TC": A100_TF32_TC_TFLOPS_DENSE,
             "FP16_TC": A100_FP16_TC_TFLOPS_DENSE,
             "BF16_TC": A100_BF16_TC_TFLOPS_DENSE},
    "H100": {"FP32_VECTOR": H100_FP32_VECTOR_TFLOPS_DENSE,
             "TF32_TC": H100_TF32_TC_TFLOPS_DENSE,
             "FP16_TC": H100_FP16_TC_TFLOPS_DENSE,
             "BF16_TC": H100_BF16_TC_TFLOPS_DENSE,
             "FP8_TC": H100_FP8_TC_TFLOPS_DENSE},
    "B200": {"FP32_VECTOR": B200_FP32_VECTOR_TFLOPS_DENSE,
             "TF32_TC": B200_TF32_TC_TFLOPS_DENSE,
             "FP16_TC": B200_FP16_TC_TFLOPS_DENSE,
             "BF16_TC": B200_BF16_TC_TFLOPS_DENSE,
             "FP8_TC": B200_FP8_TC_TFLOPS_DENSE,
             "FP4_TC": B200_FP4_TC_TFLOPS_DENSE},
}

# 精度阶梯的**首次出现代际**（用于 02 号程序 §3 的阶梯表）
# 来源：PTX ISA 9.4 各指令的 Target ISA Notes
PRECISION_FIRST_GEN = {
    "FP32_VECTOR": "P100",
    "FP16_TC": "V100",
    "TF32_TC": "A100",
    "BF16_TC": "A100",
    "FP8_TC": "H100",
    "FP4_TC": "B200",
}


# ===========================================================================
# 4. 能力断点的首次引入代际
# ===========================================================================

# 判定依据统一用 **PTX ISA 9.4 的 Target ISA Notes**（每条指令的 "Target ISA Notes"
# 小节写明它从哪个 sm_ 版本可用）。下表是「特性 → 首个可用的 target」的抄录。
FEATURE_FIRST_TARGET = {
    "tensor_core_mma":       ("Volta",   "sm_70"),
    "independent_thread_scheduling": ("Volta", "sm_70"),
    "tf32":                  ("Ampere",  "sm_80"),
    "bf16":                  ("Ampere",  "sm_80"),
    "sparse_mma_2to4":       ("Ampere",  "sm_80"),
    "cp_async":              ("Ampere",  "sm_80"),
    "fp8":                   ("Hopper",  "sm_90"),
    "tma_cp_async_bulk":     ("Hopper",  "sm_90"),
    "wgmma":                 ("Hopper",  "sm_90a"),
    "cluster_dsmem":         ("Hopper",  "sm_90"),
    "tmem":                  ("Blackwell", "sm_100a"),
    "tcgen05":               ("Blackwell", "sm_100a"),
    "fp4":                   ("Blackwell", "sm_100"),
}
# 来源：PTX ISA 9.4，各指令 "Target ISA Notes"
# 两条必须写清的修正：
#   · BF16 首次出现在 **Ampere (sm_80)**，不是 Turing。Turing 只加了 INT8/INT4。
#   · FP8 在数据中心线是 **Hopper (sm_90)**；消费级线（Ada，CC 8.9）更早具备。
#   · sm_90a / sm_100a 是 "accelerated" 变体：只在**该精确架构**上可用，
#     不像 sm_90 那样向前兼容到 sm_90 家族的其它成员。

# Turing 引入但常被误记成 "BF16" 的那一项
TURING_TENSOR_ADDITIONS = ("INT8", "INT4")
# 来源：NVIDIA Turing 架构白皮书（Tensor Core 新增 INT8 / INT4 精度）
# 注意：Turing 没有 BF16。把它归给 Turing 是本主题最常见的记错。


# ===========================================================================
# 5. 架构不变量（本主题 03 篇的事实源）
# ===========================================================================

WARP_SIZE = 32
# 来源：CUDA C++ Programming Guide 13.4 §5.1 表30，与 CUDA C++ Programming Guide
#       11.8 归档版 §5.1 的表**两代一致**。11.8 是唯一仍含 CC 6.0 / 7.0 规格表的
#       官方文档，两代对照是「它十几年没动」的直接证据。
MAX_THREADS_PER_BLOCK = 1024
# 来源：CUDA C++ Programming Guide 13.4 §5.1（各 CC 的 "Maximum number of threads
#       per block" 一栏，自 CC 2.0 起为 1024）
MAX_REGISTERS_PER_THREAD = 255
# 来源：CUDA C++ Programming Guide 13.4 §5.1（"Maximum number of 32-bit registers
#       per thread" 一栏，CC 3.0 起为 255）
SHARED_MEMORY_BANKS = 32
# 来源：CUDA C++ Programming Guide 13.4 §5.1（各 CC 的 "Number of 32-bit banks in
#       shared memory" 一栏恒为 32）
BANK_WIDTH_BYTES = 4
# 来源：同上一行（每 bank 的宽度为 32 bit = 4 byte）

REGISTER_FILE_PER_SM_KB = 256
# 来源：CUDA C++ Programming Guide 13.4 §5.1（CC 3.5 及以后各代 "Maximum number of
#       32-bit registers per SM" = 65536，即 65536 × 4 B = 256 KB）
REGISTER_FILE_PER_SM_KB_GK210 = 512
# 来源：CUDA C++ Programming Guide 11.8 归档版 §5.1（CC 3.7 = GK210 为 131072 个
#       32-bit 寄存器 = 512 KB）
# 口径：**这是「256 KB 是不变量」这条论断唯一的例外**，正文必须一起写出。

REGISTERS_PER_SM = 65536
# 来源：同 REGISTER_FILE_PER_SM_KB（65536 个 32-bit 寄存器）

# --- 每 SM 最大常驻 warp / 线程 / block：**不是不变量** ---
# 结构：CC → (每 SM 最大常驻 warp 数, 每 SM 最大常驻线程数, 每 SM 最大常驻 block 数)
# 来源：CUDA C++ Programming Guide 13.4 §5.1 表（CC 7.5 / 8.0 / 8.6 / 8.9 / 9.0 /
#       10.0 / 11.0 / 12.x）与 CUDA C++ Programming Guide 11.8 归档版 §5.1 表（CC 6.0 / 7.0）
RESIDENT_LIMITS_BY_CC = {
    "6.0": (64, 2048, 32),      # P100（数据中心线）
    "7.0": (64, 2048, 32),      # V100（数据中心线）
    "7.5": (32, 1024, 16),      # Turing —— **反例：不是 64/2048/32**
    "8.0": (64, 2048, 32),      # A100（数据中心线）
    "8.6": (48, 1536, None),    # GA10x 消费级
    "8.9": (48, 1536, None),    # Ada —— **反例**
    "9.0": (64, 2048, 32),      # H100（数据中心线）
    "10.0": (64, 2048, 32),     # B200（数据中心线）
    "11.0": (48, 1536, None),   # **反例**
    "12.x": (48, 1536, None),   # **反例**
}
# 口径：None 表示该 CC 的每 SM 最大常驻 block 数未在本次采用的来源中逐条核对，
# 不填一个猜的值。本表要说明的是 warp / 线程两栏，block 一栏只在能核对的 CC 上给出。
# 关键结论：**数据中心 SXM 线（6.0 / 7.0 / 8.0 / 9.0 / 10.0）恰好稳定在 64 / 2048 / 32**，
# 而 7.5 与 8.6 / 8.9 / 11.0 / 12.x 不是。**这是产品线的巧合稳定，不是架构不变量。**


# ===========================================================================
# 6. 外部引用值（**不是 NVIDIA 官方规格，本主题的程序不产出它们**）
# ===========================================================================

# 访存延迟的跨代对照。这一组**不是本文产出的**：它们来自外部在同一套方法下的
# 跨代对照，本文只转述，无法复核，也不对其口径负责。
# 来源：外部跨代对照（同一套方法）—— A100 466 / Ada 542 / H800 479，单位：时钟周期。
MEMORY_LATENCY_CYCLES_EXTERNAL = {
    "A100": 466, "Ada": 542, "H800": 479,
}
# 量级区间：HBM 往返跨代基本持平，落在 460~620 个时钟周期。
MEMORY_LATENCY_CYCLES_RANGE_EXTERNAL = (460, 620)
# 口径：**Pascal / Volta / Turing 的数值来自另一套方法**，与上表不同源，
# 只能比量级、不能逐位对齐。正文只给区间，不给逐代逐位数。


# ===========================================================================
# 7. 明确未找到可靠来源的项——**留空，不填**
# ===========================================================================

UNSOURCED = (
    "B200 每 SM 的 tensor core 数量：官方未披露",
    "B200 的 L2 容量：第三方给出 126 MB 与 50 MB 两个互斥值，均不可采信",
    "B200 的 die 面积与 boost 时钟：官方未披露（时钟只能用 1.98 GHz 的反推值并标注）",
    "H100 SXM5 的官方 boost 时钟：NVIDIA 从不公布，只能用 67 TFLOPS 反推",
    "B200 的 INT4 tensor 支持：CUDA 13.4 表33 与 Blackwell 技术简报表1 冲突，回避",
)
# 处理方式：正文写「官方未披露」，而不是填第三方聚合站（topcpu.net、cpudb.cc、
# technical.city、fpsbench.com 等）的数字。这是本模块存在的理由之一。

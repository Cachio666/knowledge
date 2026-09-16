"""40-kernels/moe-kernels 主题的共享事实源。

本模块**只放从官方来源抄来的硬件规格常量**，不放任何推导值。
代价模型的全部推导在 `verify.py` 里做，本模块不重复。

三条硬性约定（见 CONTRIBUTING §2 与 40-kernels/README.md）：

  1. **没有 `# 来源：` 注释的常量不许写进本模块。**
  2. **常量名必须编码口径**，使误用可见：算力一律带 `_DENSE_SPARSE` 后缀并
     **存成 `(dense, sparse_2to4)` 二元组**，带宽一律带 `_GBps`，容量一律带 `_KB`。
  3. **找不到可靠来源的项不写进来**，改记为 `UNSOURCED`。

**常量与来源沿用 `20-gpu-architecture/*/code/specs.py` 已有的口径**：
`code/` 必须自包含所以重新抄一份，但同名的常量同值同来源，不得重新发明。
本模块里 `SM_COUNT_BY_GEN` / `HBM_BW_GBps_BY_GEN` / `TC_TFLOPS_DENSE_SPARSE` /
`SMEM_PER_SM_MAX_KB` / `FLOAT16_BYTES` / `KERNEL_LAUNCH_OVERHEAD_US_EXTERNAL` /
`L2_MB_H100_IN_REPO` 与 `40-kernels/gemm/code/specs.py` 逐字一致（同值同来源）。

来源等级沿用本仓库约定：**A** = NVIDIA 官方规范/白皮书/datasheet；
**B** = NVIDIA 官方代码或作者为 NVIDIA 的论文；**C** = 第三方逆向/学术。
**只有 A 级可以写进本模块的来源注释。**

⚠ **本模块不含「模型规格」。** MoE 的 `d_model` / `d_ff` / 专家数 / `top_k` / token 数
都是**本文选定的对照场景**，不是从任何一手来源抄来的模型配置——它们住在
`verify.py` 与各 `.cu` 内核里，并在正文中标注为「场景参数」。本模块只管硬件。
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

SMEM_PER_SM_MAX_KB = 228
# 来源：NVIDIA H100 Tensor Core GPU Architecture Whitepaper（每 SM 统一池
#       256 KB，其中至多 228 KB 可切给 shared memory）
SMEM_DELIVERY_BYTES_PER_CLK = 128
# 来源：NVIDIA CUDA C++ Programming Guide 13.4 §5.1（32 个 bank × 4 字节，
#       无冲突时每周期交付 128 字节）
SECTOR_BYTES = 32
# 来源：NVIDIA CUDA C++ Programming Guide 13.4 §5.1 与 *CUDA C++ Best Practices
#       Guide*「Coalesced Access to Global Memory」（**L1 与 DRAM 之间的最小
#       事务粒度是 32 字节 sector**，不足 32 字节的访问同样整块取回）
# 口径：**这是「合并度」度量的单位**——本主题 03 篇算「一个 warp 的访存触及几个
#       sector」，而不是算「搬了几字节」。两者在未合并时会差一个放大倍数。

MAX_THREADS_PER_SM = 2048
MAX_BLOCKS_PER_SM = 32
# 来源：CUDA C++ Programming Guide 13.4 §5.1 表（CC 9.0 / 10.0 的
#       "Maximum number of resident threads per SM" = 2048 与
#       "Maximum number of resident blocks per SM" = 32）；
#       与 20-gpu-architecture/architecture-evolution/code/specs.py 的
#       RESIDENT_LIMITS_BY_CC[("9.0"|"10.0")] 一致（64 warp / 2048 线程 / 32 block）

FLOAT32_BYTES = 4
FLOAT16_BYTES = 2
BFLOAT16_BYTES = 2
# 来源：IEEE 754 binary32 / binary16 的位宽（32 / 16 bit）；BF16 为 16 bit
#       （1 符号 + 8 指数 + 7 尾数，见 *BFloat16: The secret to high performance on
#        Cloud TPUs* 与 NVIDIA A100 Whitepaper 对 BF16 格式的描述）。
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
# 口径：**非 tensor core** 的 FP32 向量峰值（本主题三个内核走的就是这条通路，
#       它们都不发射 `mma` —— 本主题考的是**调度与访存**，不是 tensor core 吞吐）。

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
#   · 本主题只用它算**专家权重流量的算术强度拐点** `I* = 峰值 ÷ 带宽`
#     （正文 01 篇 §1），这个口径要求 dense 且同精度：FP16 dense ÷ HBM3。


# ===========================================================================
# 2. 指令的 ISA 下限（**照抄已入库的口径，不重推**）
# ===========================================================================

FEATURE_FIRST_TARGET = {
    "atomic_add_float": ("Kepler", "sm_20"),
    "shfl_sync":        ("Volta",  "sm_70"),
    "cp_async":         ("Ampere", "sm_80"),
    "tma_cp_async_bulk": ("Hopper", "sm_90"),
    "wgmma":            ("Hopper", "sm_90a"),
    "tcgen05":          ("Blackwell", "sm_100a"),
}
# 来源：PTX ISA 9.4 各指令的 "Target ISA Notes"（`red.global.add.f32` /
#       `atom.global.add.f32` 自 sm_20 起可用；`shfl.sync` 自 sm_70 起需显式
#       mask）；与 20-gpu-architecture/architecture-evolution/code/specs.py 的
#       FEATURE_FIRST_TARGET 逐条一致
# 口径：**本主题三个内核全部落在 `cp_async` 一档及以下**（sm_80 可移植），
#       所以它们全进默认 TARGETS，不需要 gemm 主题那样的 hopper / blackwell 目标。
#       `tma_cp_async_bulk` 是 `sm_90` **不是** `sm_90a`；`wgmma` 才是 `sm_90a`。


# ===========================================================================
# 3. 外部引用值（**不是 NVIDIA 官方规格，本主题的程序不产出它们**）
# ===========================================================================

L2_MB_H100_IN_REPO = 50
# 来源：本仓库 10-foundations/performance-models/09_roofline_model（「H100 有 50 MB L2」）
#       与 performance-models/03-measuring-roofline §2 的规格表
# 口径：⚠ **本仓库对 L2 容量这一项内部不一致**——`20-gpu-architecture/
#       architecture-evolution/code/specs.py` 只把 **B200** 的 L2 列为 UNSOURCED，
#       并未给 H100 的值。本条只用于 03 篇 §1 的量级判断（「permute 出来的
#       中间张量装不装得进 L2」），**不做精确结论**，正文每次使用都注明这一点。
#       （与 40-kernels/elementwise-fusion/code/specs.py 的常量同名同值同来源。）

KERNEL_LAUNCH_OVERHEAD_US_EXTERNAL = (3.0, 5.0)
# 来源：外部对 CUDA 空 kernel 启动开销的对照测量（**非 NVIDIA 官方**）
# 口径：**引用值，本仓库不复核其测量口径**。量级是「一次 kernel 启动数微秒」，
#       随驱动版本、是否用 CUDA Graph、是否走 MPS 而变。
#       本主题 01 篇用它做「四个 kernel 各自启动一次」的量级对照。
#       （与 40-kernels/gemm/code/specs.py 的常量同名同值同来源。）


# ===========================================================================
# 4. 明确未找到可靠来源的项——**留空，不填**
# ===========================================================================

UNSOURCED = (
    "H100 SXM5 的官方 boost 时钟：官方未公布，只能用 67 TFLOPS ÷ (132×128×2) 反推。"
    "本主题不做需要时钟的算术（不推 cycles），正文回避",

    "本主题「场景参数」（d_model / d_ff / 专家数 / top_k / token 数）的**出处**："
    "它们不是**模型规格**，是本文选定的对照场景，**没有一手来源可引**，"
    "所以不进 specs.py，只住在 verify.py 与 .cu 里，正文一律标注为「场景参数」，"
    "**不得**写成「某模型的规格」",

    "「MoE 端到端比等参数稠密模型快多少倍」这类区间：**本次检索不到可引的一手页面**。"
    "按本模块的规则（找不到可靠来源的不写进来），**正文不引用任何具体区间**，"
    "只给可判定的判据（FLOPs 比值 = 专家数 ÷ top_k；权重流量与专家数成正比，"
    "与批大小无关）。⚠ 不要与任何「加速比」报道互换引用",

    "`atomicAdd` 直方图在「哪个 SM 先到」这一层面的非确定性程度："
    "CUDA 文档只承诺**计数结果**确定，不承诺**到达顺序**。"
    "本主题 03 篇只把「顺序不确定」作为一条失败模式的成因，**不给概率或量级**",
)
# 处理方式：正文写「官方未披露」或直接回避，而不是填第三方聚合站的数字。

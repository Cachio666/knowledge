"""interconnect 主题的共享事实源。

本模块**只放从官方来源抄来的硬件规格常量**，不放任何推导值。
所有推导都在 01 / 02 / 03 / 04 四个编号程序里做。

四条硬性约定：

  1. **没有 `# 来源：` 注释的常量不许写进本模块。**
  2. **常量名必须编码口径**，使误用可见。本主题最容易被张冠李戴的量是带宽，
     所以命名里的 `_PER_LINK` / `_PER_GPU` / `_AGG` 与 `_UNIDIR` / `_BIDIR`
     一个都不能省。用 `H100_NVLINK4_PER_GPU_BIDIR_GBps`，不用 `H100_NVLINK_BW`。
  3. **单位一律写明**：`_GBps` 是十进制 10^9 B/s（NVIDIA 全线用十进制），
     `_Gbps` 是 10^9 bit/s，`_GTps` 是每秒十亿次传输（**不是** Gb/s）。
     `_GB` 是 10^9 字节。二进制口径另带 `_GiB`。
  4. **找不到可靠来源的项不写进来**，只在末尾的 `UNSOURCED` 里留一条说明。

本模块的常量分两组：
  · 前六节是**硬件规格**（有官方来源）。
  · 第七节是**外部引用值**（不是规格，是别处已推导过的量级，本文只转述）。

口径警告：本主题里「谁比谁」比「谁是多少」更容易错。三条最容易踩的：
  · NVLink 的每 GPU 数字全是**双向合计**（H100 900 = 450 发 + 450 收）；
  · PCIe 的标称值**不含编码开销**，Gen1/2 差 20%、Gen3 起只差 1.5%；
  · scale-up（NVLink）与 scale-out（IB/RoCE）是**两套并行网络**，不能相加。
"""

# ===========================================================================
# 1. PCIe 逐代
# ===========================================================================

# 来源：PCI-SIG PCIe 规范与官方博客；PCI-SIG 官网受 Cloudflare 拦截，本次未能
#       取到一手页面，下列数值在 PCI-SIG 博客、Wikipedia「PCI Express」条目与
#       NVIDIA 官方博客三处交叉一致（见 UNSOURCED 里的说明）。
# 口径：**每 lane 的原始线速率**，单位 GT/s（每秒十亿次传输）。Gen1/2/3/4/5 是
#       NRZ 编码，一个传输 = 一个比特，所以 GT/s 与 Gb/s 数值相同；Gen6 改为
#       PAM4，一个传输 = 两个比特，此时 GT/s 与 Gb/s 不再相等（这条差异是
#       「标称带宽」口径混乱的一个来源）。
PCIE_LANE_GTps_BY_GEN = {
    "1": 2.5,
    "2": 5.0,
    "3": 8.0,
    "4": 16.0,
    "5": 32.0,
    "6": 64.0,
}

# 来源：同 PCIE_LANE_GTps_BY_GEN（PCI-SIG 博客与 Wikipedia PCI Express 条目）。
#       128b/130b 由 PCIe 3.0 引入；8b/10b 只用于 Gen1/Gen2。
# 口径：`(分子, 分母)` 表示「每分母个传输单位里，有分子个承载载荷」。这是**编码
#       效率**，不是协议开销的全部（Gen6 还另有 FLIT 层面的 CRC/FEC 开销，见下）。
PCIE_ENCODING_BY_GEN = {
    "1": ("8b/10b", 8, 10),
    "2": ("8b/10b", 8, 10),
    "3": ("128b/130b", 128, 130),
    "4": ("128b/130b", 128, 130),
    "5": ("128b/130b", 128, 130),
    "6": ("PAM4 + FLIT", 242, 256),
}
# Gen6 的 242/256 口径说明：一个 256 B FLIT = 236 B TLP + 6 B DLP（= 242 B 载荷）
# + 8 B CRC + 6 B FEC。所以「载荷/线速」是 242/256，而 Gen1~5 那一列是比特级的
# 线编码效率，两者不是同一层的东西——但都表现为「有效带宽 < 线速率」，本文按同一
# 列处理并在正文里点名。

# 来源：PCI-SIG 博客《PCIe 6.0 Specification Webinar Q&A: Error Detection and
#       Correction with FEC》与 Cadence 对 PCIe 6.0 FLIT 模式的解析（两处一致）。
# 口径：一个 256 B FLIT 的字段拆分（字节）。
PCIE6_FLIT_BYTES = {
    "TLP": 236,          # 事务层包（真正的载荷）
    "DLP": 6,            # 数据链路层包（含 2 B 的序号/ACK/NAK）
    "CRC": 8,            # 保护前 242 B
    "FEC": 6,            # 保护前 250 B（载荷 + CRC），三路交织
}
# 口径：242 = 236 + 6 是「可用载荷」，14 = 8 + 6 是「为可靠传输付出的开销」。

# 来源：NVIDIA 官方文档给出的 H2D（host→device）样例输出，转引自本主题实施规格
#       §0.8.1。**本次未能定位一手页面**（见 UNSOURCED），只作为「官方口径的样例
#       值」使用，正文必须标注这一点。
# 口径：**单向**（host→device），十进制 GB/s，pinned（页锁定）内存，PCIe Gen5 x16。
PCIE_GEN5_X16_H2D_PINNED_SAMPLE_GBps = 55.63
# 口径：同上，但 PCIe Gen2 x16 + pinned 内存。
PCIE_GEN2_X16_H2D_PINNED_SAMPLE_GBps = 5.77
# 口径：同上，但 PCIe Gen2 x16 + **pageable（可分页）**内存。**这不是 GPU 或 PCIe
#       的属性**：它随 CPU、主板、内存带宽剧烈变化，只能当量级看。
PCIE_GEN2_X16_H2D_PAGEABLE_SAMPLE_GBps = 2.31


# ===========================================================================
# 2. NVLink 逐代
# ===========================================================================

# 来源：NVIDIA NVLink 官方规格页 https://www.nvidia.com/en-us/data-center/nvlink/
#       （逐代的 per-GPU 带宽与 max links per GPU 两列）；NVLink 1 的 160 GB/s
#       另见 Pascal 架构白皮书；NVLink 6 的 3600 GB/s 见 NVLink 规格页与
#       Vera Rubin NVL72 的发布材料（72 × 3.6 TB/s = 259.2 ≈ 官方 260 TB/s）。
# 口径：**每 GPU 的双向合计**（发送 + 接收），十进制 GB/s。
#       ★ 这是本主题最危险的一条：H100 的 900 = 450 发送 + 450 接收。
#       做「搬运 X 字节要多久」的推导必须用 450，不是 900。
NVLINK_PER_GPU_BIDIR_GBps_BY_GEN = {
    1: 160.0,    # P100
    2: 300.0,    # V100
    3: 600.0,    # A100
    4: 900.0,    # H100
    5: 1800.0,   # B200
    6: 3600.0,   # Rubin（NVIDIA 标注 preliminary）
}

# 来源：同 NVLINK_PER_GPU_BIDIR_GBps_BY_GEN（NVLink 规格页的 "Max links per GPU"）。
# 口径：**每 GPU 的 NVLink 条数**，不是整机条数，也不是 NVSwitch 颗数。
NVLINK_LINKS_PER_GPU_BY_GEN = {
    1: 4,     # P100
    2: 6,     # V100
    3: 12,    # A100
    4: 18,    # H100
    5: 18,    # B200
    6: 36,    # Rubin
}

# 来源：同 NVLINK_PER_GPU_BIDIR_GBps_BY_GEN。用于表格的对照列。
NVLINK_GPU_BY_GEN = {
    1: "P100", 2: "V100", 3: "A100", 4: "H100", 5: "B200", 6: "Rubin",
}
NVLINK_YEAR_BY_GEN = {
    1: 2016, 2: 2017, 3: 2020, 4: 2022, 5: 2024, 6: 2026,
}

# 来源：NVIDIA NVLink 规格页与 Hopper 架构页原文
#       *"Fourth-generation NVLink can scale multi-GPU input and output (IO) with
#       NVIDIA DGX and HGX servers at 900 gigabytes per second (GB/s)
#       bidirectional per GPU, over 7X the bandwidth of PCIe Gen5."*
# 口径：**每 link 的双向合计**。50 = 900 / 18，25 = 50 / 2 是每方向。
#       注意「每 link 每方向 25 GB/s」是**由官方总数除官方链路数得到**，
#       NVIDIA 从未直接写过 "25 GB/s per direction"（属推导，非直述）。
NVLINK4_PER_LINK_BIDIR_GBps = 50.0
NVLINK5_PER_LINK_BIDIR_GBps = 100.0


# ===========================================================================
# 3. NVSwitch 与机内拓扑
# ===========================================================================

# 来源：NVIDIA NVSwitch 官方材料（gen1 的 18 端口 18×18 全交叉在 DGX-2 架构
#       材料里直述）；HGX H100 博客 *"Each NVSwitch is a fully non-blocking
#       switch that fully connects all eight H100 Tensor Core GPU."*
# 口径：**每颗 NVSwitch 的 NVLink 端口数与交换容量**。
NVSWITCH_GEN1_PORTS = 18
NVSWITCH_GEN1_CAPACITY_BIDIR_GBps = 900.0   # 18 × 50，双向合计

# 来源：NVIDIA HGX H100 / HGX A100 官方材料。
# 口径：**整机的 NVSwitch 颗数**。
#       ⚠ DGX H100 用户指南有一句把「4 颗 NVSwitch」与「4th gen NVLink」混写；
#       那个 4 指 NVSwitch **颗数**，不是 NVLink 条数（H100 是 18 条）。
HGX_H100_NVSWITCH_COUNT = 4
HGX_A100_NVSWITCH_COUNT = 6

# 来源：NVIDIA HGX 平台对比表（HGX A100 8-GPU 与 HGX H100 8-GPU 两行，
#       "Bisection Bandwidth" 一行分别为 2.4 TB/s 与 3.6 TB/s）。
# 口径：**整机（8 GPU）的 bisection bandwidth，双向合计口径**，十进制 TB/s。
#       这是本主题里可信度最高的两个拓扑数字——它们是官方**直述**，不是推导。
HGX_A100_8GPU_BISECTION_TBps = 2.4
HGX_H100_8GPU_BISECTION_TBps = 3.6

# 来源：NVIDIA NVLink 规格页 / NVL72 发布材料。
# 口径：**scale-up 网络（NVLink）的整机聚合带宽**，双向合计，十进制 TB/s。
#       ⚠ 它与 scale-out 不是一回事（见第 4 节），两者相差 18 倍。
GB200_NVL72_SCALEUP_AGGR_TBps = 130.0     # 72 × 1.8
GB200_NVL72_SWITCH_TRAYS = 9

# 来源：NVIDIA 官方对 "NVLink Domain" 的定义（*"A set of nodes that can
#       communicate over NVLink."*，**是跨 OS 实例的概念**）与逐平台材料。
# 口径：**一个 NVLink 域里的 GPU 数**。注意它 ≠ 物理连接方式：
#       DGX-1 的 8 卡是 hybrid cube-mesh（4/7 直连、3/7 需一跳），不是全互联；
#       A100 SuperPOD 的 256 卡不是 NVLink 域（跨节点走 InfiniBand，官方表格里
#       256 A100 pod 的 NVLink Domain 仍是 8 GPU）。
NVLINK_DOMAIN_GPUS_BY_PLATFORM = {
    "DGX-1 (P100/V100)": 8,          # hybrid cube-mesh，非全互联
    "DGX-2 (V100 + NVSwitch gen1)": 16,
    "HGX A100 8-GPU": 8,
    "HGX H100 8-GPU": 8,
    "HGX H100 + NVLink Switch System": 256,
    "GH200 NVL32": 32,
    "GB200 NVL72": 72,
    "GB200 多机架": 576,
}
# A100 SuperPOD 的 256 卡**不在上表里**——它的 NVLink 域仍是 8 GPU。


# ===========================================================================
# 4. 机间 scale-out（InfiniBand / RoCE 这一侧）
# ===========================================================================

# 来源：InfiniBand 官方代际命名与 NVIDIA 文档。
# 口径：**每端口标称速率**，单位 Gb/s（10^9 bit/s）。按 `Gb/s ÷ 8` 换算成 GB/s
#       是**上界**（未扣链路层开销）；IB 走原生链路层，头开销比 RoCE 小得多，
#       「400 Gb/s IB」与「400GbE」的载荷吞吐并不相同（见 UNSOURCED）。
IB_PER_PORT_GBps = {
    "EDR": 12.5,   # 100 Gb/s
    "HDR": 25.0,   # 200 Gb/s
    "NDR": 50.0,   # 400 Gb/s
    "XDR": 100.0,  # 800 Gb/s
}

# 来源：NVIDIA DGX H100 用户指南的端口映射表（逐行把 CX0↔GPU7、CX1↔GPU4…），
#       与 DGX A100 的同类材料。
# 口径：**每 GPU 配了几张网卡**。1 表示 1:1。
#       这是「GPU:网卡配比」这条判据的来源：1:1 时节点聚合 IB 带宽 = 每卡 IB × 卡数。
DGX_H100_NIC_PER_GPU = 1        # 8 张 ConnectX-7 @ 400 Gb/s
DGX_A100_NIC_PER_GPU = 1        # 8 张 ConnectX-6 @ 200 Gb/s（HDR）
GB200_NVL72_NIC_COUNT = 72      # 72 张 ConnectX-8 @ 800 Gb/s，1:1
GB200_NVL72_NIC_Gbps = 800


# ===========================================================================
# 5. 测量口径（NCCL / nvbandwidth）
# ===========================================================================

# 来源：NCCL 官方文档（nccl-tests 的 bandwidth 定义一节）。
# 口径：`busbw = algbw × 修正系数`，系数随算子不同而不同。这里存的是**NCCL 文档
#       写明的公式本身**（字符串），数值化的部分由 04 号程序按 n 现场算——
#       把 n 代进去得到的不是规格，是推导值。
NCCL_BUSBW_CORRECTION = {
    "AllReduce": "2(n-1)/n",
    "ReduceScatter": "(n-1)/n",
    "AllGather": "(n-1)/n",
    "AllToAll": "(n-1)/n",
    "Broadcast": "1",
    "Reduce": "1",
}

# 来源：GPUDirect Storage Best Practices（NVIDIA 官方）。
# 口径：P2P 可用性的**前置条件**，不是性能数字。
P2P_REQUIRES_ACS_DISABLED = True          # 否则事务被强制上行到 Root Complex
P2P_REQUIRES_IOMMU_DISABLED_ON_X86 = True  # 例外：Grace Hopper / DGX 平台应保持开启


# ===========================================================================
# 6. 明确未找到可靠来源的项——**留空，不填**
# ===========================================================================

UNSOURCED = (
    "NVSwitch gen2 的端口数与交换容量：官方只写「6 颗 gen2，比 gen1 快 2 倍」",
    "NVSwitch gen3 的 64 端口 / 3.2 TB/s：仅二级来源，未见 NVIDIA 一手规格页确认",
    "NVLink 6 每 link 速率：官方未逐条列出，3000/36=83.3 与 3600/36=100 两种口径"
    "都在流传；本主题采用 NVLink 规格页的 3600 GB/s（见下）并标注 preliminary",
    "DGX H100 中 18 条 NVLink 到 4 颗 NVSwitch 的分配比（二级来源称 5/4/4/5）",
    "InfiniBand 每端口 Gb/s 是否已扣 FEC/线路编码：未见一手确认，"
    "且 NDR 的线路编码说法不一（64b/66b vs 256b/257b）",
    "GDR 1.6 Tb/s：仅路线图，NVIDIA 官方产品页未列",
    "GB200 NVL72 的 bisection bandwidth：NVIDIA 未公布，要算须自行推导并标注",
    "覆盖 PCIe Gen3/4/5 的官方 H2D 达成率基准表：不存在",
    "PCI-SIG 一手页面：Cloudflare 拦截，引文来自官方博客与 Wikipedia 的交叉一致",
    "PCIe Gen5 x16 / Gen2 x16 的 H2D 样例值（55.63 / 5.77 / 2.31 GB/s）："
    "转引自本主题实施规格 §0.8.1，本次未定位到一手页面",
)


# ===========================================================================
# 7. 外部引用值（**不是硬件规格，本主题的程序不产出它们**）
# ===========================================================================

# 来源：本仓库 10-foundations/performance-models/02-amdahl-and-scaling.md
#       §1 第六步（原文：「每步 5 μs」）。
# 口径：**ring all-reduce 每走一步的固定延迟**，单位 μs。它不是 NVLink 的规格，
#       也不是本主题测出来的量；本文引用它，不重推它。
EXTERNAL_RING_STEP_LATENCY_US = 5.0

# 口径：本文所有「小消息/大消息分界」的推导都用这个量级，不另找来源。
EXTERNAL_RING_STEP_LATENCY_RANGE_US = (2.0, 10.0)

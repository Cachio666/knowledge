---
title: 带宽口径：同一个数字的三种含义
slug: interconnect-bandwidth-conventions
domain: 20-gpu-architecture
topic: interconnect
level: 2
difficulty: beginner
prerequisites: []
tags: [pcie, nvlink, bandwidth, encoding, unit-conversion, bidirectional]
related:
  - 20-gpu-architecture/interconnect/02-topology-and-bisection
  - 20-gpu-architecture/interconnect/04-measuring-bandwidth
  - 110-communication/collectives
  - 10-foundations/performance-models/02-amdahl-and-scaling
code:
  - code/01_bandwidth_conventions.py
status: stable
updated: 2026-09-15
---

# 带宽口径：同一个数字的三种含义

> 互联的所有「2 倍错误」都来自同一个动作：把一个**双向合计**的数字当成单向可用带宽，
> 或者把一个**含编码开销之前的线速率**当成能搬数据的载荷。本文只做一件事——
> 给每个数字贴上它属于哪种口径的标签。
>
> **全文的数字口径**：本文的数字分两类——**规格值**（引自 NVIDIA / PCI-SIG 官方来源，
> 见脚注与 §参考）与**推导值**（由 `code/specs.py` 中的规格常量代入 §1 的公式算出）。
> **本文无实测数据。** 要复核，运行 `code/01_bandwidth_conventions.py`，
> 它会打印出本文引用的每一个数。

## 0. 背景与动机

两份资料都写「PCIe Gen5 x16 是 64 GB/s」，但它们说的**不是同一件事**：

- 第一份说的是**线速率口径**：`32 GT/s × 16 lane ÷ 8 = 64 GB/s`，这一步还没有扣编码开销；
- 第二份说的是**载荷口径**：真正能承载数据的字节率是 **63.02 GB/s**（扣掉 128b/130b 的 2/130）。

两份资料又都写「H100 的 NVLink 是 900 GB/s」，这一次分歧更大：900 是**双向合计**
（450 发 + 450 收）。如果拿它去算「搬一份 140 GB 的梯度要多久」，会得到 **155.6 ms**；
用正确的单向 450 GB/s 算是 **311.1 ms**——**差整整 2 倍**（推导值，§1 第三步）。

这不是学究式的较真。一个 64 卡的训练任务，扩展效率的估算、checkpoint 的落盘时间
（**训练状态由什么组成、一共多少字节**的正文在 [90-training/checkpointing/01-state-and-bytes.md](../../90-training/checkpointing/01-state-and-bytes.md)，此处回链）、
MoE 的 all-to-all 预算，全都建立在这几个数字上。而在这一层**弄错一个口径，
得到的不是小数点后的偏差，是一个 2 倍的错误结论**——并且它会自洽地传染到后面每一层
推导：因为推导过程本身是对的，错的只是喂进去的那个常量。

所以本文的定位是给后三篇打地基：把「带宽」这个词拆成几个**互不通用**的口径，
每个都给出换算公式和逐代的数，并且指出 NVIDIA 自己在哪一处用了哪一种。

## 1. 原理

### 第一步：PCIe 的一个「带宽」至少是三个数

先固定三条公式。本文全部 PCIe 数字都由它们产生，没有例外：

```text
线速率 x16 单向 GB/s = GT/s × 16 ÷ 8            ← GT/s 是每 lane 的原始传输率
载荷   x16 单向 GB/s = 线速率 × (编码分子 ÷ 编码分母)
双向                 = 单向 × 2                  ← PCIe 每方向独立，全双工
```

`GT/s`（每秒十亿次传输）不是 `Gb/s`：Gen1~Gen5 用 NRZ，一个传输带 1 bit，两者数值相等；
**Gen6 改成 PAM4，一个传输带 2 bit，两者不再相等**——这条差异本身就是标称口径混乱的一个来源。

逐代列全表（推导值，`code/01_bandwidth_conventions.py` 第 1 节）：

| 代 | 每 lane GT/s | 编码 | 编码效率 | 线速率 x16 单向 | 载荷 x16 单向 | 载荷 x16 双向 | 编码开销 |
|---|---|---|---|---|---|---|---|
| Gen1 | 2.5 | 8b/10b | 80.00% | 5.00 | **4.00** | 8.00 | 20.00% |
| Gen2 | 5.0 | 8b/10b | 80.00% | 10.00 | **8.00** | 16.00 | 20.00% |
| Gen3 | 8.0 | **128b/130b** | 98.46% | 16.00 | **15.75** | 31.51 | 1.54% |
| Gen4 | 16.0 | 128b/130b | 98.46% | 32.00 | **31.51** | 63.02 | 1.54% |
| Gen5 | 32.0 | 128b/130b | 98.46% | 64.00 | **63.02** | 126.03 | 1.54% |
| Gen6 | 64.0 | PAM4 + FLIT | 94.53% | 128.00 | **121.00** | 242.00 | 5.47% |

**8b/10b 只用于 Gen1/Gen2；128b/130b 是 PCIe 3.0 引入的。** 这一条值得单独写出来，
因为把它记错会连带把 Gen3 的载荷算成 12.8 GB/s（把 20% 的编码开销套在了只有 1.54% 的那一代上），
而正确值是 **15.75 GB/s**。Gen3 起编码开销只剩 1.54%，Gen6 因为 PAM4 需要 FEC 才回升到 5.47%。

Gen6 的 242/256 不是线编码，是 FLIT 的字段开销：一个 256 B FLIT =
236 B TLP + 6 B DLP + 8 B CRC + 6 B FEC，可用载荷 242 B（推导值，规格值来自 PCI-SIG）。
口径提醒：PCI-SIG 另有一句「FLIT 模式的开销低于 2%」，那个 2% 与这里的 5.47% **不是同一个口径**
（前者是相对 Gen5 逐 TLP 的 CRC 开销而言的净改善，后者是相对原始线速率的绝对开销）。

**标准表格里的精确值是「载荷口径」**，NVIDIA 自己的博客也用这一列：
*"PCIe Gen 4 … providing 31.5 GB/sec vs. 15.75 GB/sec for x16 connections"*——
本程序复算的两个数是 31.51 与 15.75，逐位吻合（来源见参考）。

**但市面上流传的整数在两代上是不同的东西**，这是最容易踩的地方：

```text
Gen1 / Gen2 流传的 4 / 8 GB/s  →  载荷口径（8b/10b 的 20% 太显眼，没人取整得下去）
Gen3 起流传的 16 / 32 / 64 / 128 GB/s  →  线速率口径（1.5% 的差在营销里被抹掉）
```

所以「Gen1 标称比有效高 20%」这句话**是错的**（Gen1 流传的 4 GB/s 本来就是编码后的数），
正确的是「**编码开销**是 20%」——差在「线速率 vs 载荷」，不在「营销 vs 载荷」。
真正的「取整 vs 载荷」差额是：Gen1/2 = 0%，Gen3~5 = 1.54%，Gen6 = 5.47%。

### 第二步：NVIDIA 用的是哪一种口径？从它的倍数反推

这是一个可以用算术查证的推断。NVIDIA 在三处拿 NVLink 的每 GPU 带宽去除 PCIe：

| 出处 | 原文 | 除数 | NVIDIA 倍数 | 换成载荷口径 | 偏差 |
|---|---|---|---|---|---|
| Hopper 架构页 | "over 7X the bandwidth of PCIe Gen5" | 128（线速率） | 7.03 | 7.14 | −1.54% |
| HGX H100 博客 | "more than 14x the bandwidth of PCIe Gen4 x16" | 64（线速率） | 9.38 | 9.52 | −1.54% |
| NVLink 规格页 | "over 14x the bandwidth of PCIe Gen6" | 256（线速率） | 14.06 | 14.88 | −5.47% |

（推导值。900 ÷ 128 = 7.03；若除数换成载荷口径的 126.03，倍数是 7.14。）

结论：**NVIDIA 的除数用的是线速率**（Gen3 起的取整值），不是载荷。
这不是造假，是取整习惯——但偏差方向是**单边**的：除数偏大 ⟹ 倍数偏低 ⟹
**用它的倍数去反推 PCIe 侧的字节数，你会系统性地少算 1.5%**（Gen6 是 5.5%）。
1.5% 本身无害，但它是**同一个推导里混用两套口径**的证据，
而混用一旦被继承到下一层，就会和本文第三步的 2 倍错误叠在一起。

### 第三步：NVLink 的三级放大，且三级全是双向合计

NVIDIA 的 NVLink 规格页只给两列：**每 GPU 的双向带宽**与**每 GPU 的最大链路数**。
下表其余四个数都是这两列相除或折半得到的，**是推导，不是直述**：

```text
每 link 双向  = 每 GPU 双向 ÷ 链路数
每 GPU 单向   = 每 GPU 双向 ÷ 2
每 link 每方向 = 每 link 双向 ÷ 2
```

| 代 | GPU | 年 | 链路数/GPU | 每 GPU 双向 | 每 GPU 单向 | 每 link 双向 | 每 link 每方向 |
|---|---|---|---|---|---|---|---|
| NVLink 1 | P100 | 2016 | 4 | 160 | 80 | 40.00 | 20.00 |
| NVLink 2 | V100 | 2017 | 6 | 300 | 150 | 50.00 | 25.00 |
| NVLink 3 | A100 | 2020 | 12 | 600 | 300 | 50.00 | 25.00 |
| NVLink 4 | H100 | 2022 | 18 | **900** | **450** | 50.00 | 25.00 |
| NVLink 5 | B200 | 2024 | 18 | 1800 | 900 | 100.00 | 50.00 |
| NVLink 6 | Rubin | 2026 | 36 | 3600 | 1800 | 100.00 | 50.00 |

**★ 本主题最危险的一条：上表「每 GPU 双向」整列都是双向合计。**
H100 的 900 GB/s = **450 GB/s 发送 + 450 GB/s 接收**。
**任何单向搬运的天花板是 450 GB/s，不是 900。** 官方原文有两条独立来源可直接引用：

- Hopper 架构页：*"Fourth-generation NVLink can scale multi-GPU input and output (IO)
  with NVIDIA DGX and HGX servers at **900 gigabytes per second (GB/s) bidirectional
  per GPU**, over 7X the bandwidth of PCIe Gen5."*
- HGX H100 博客：*"this communication runs at the **NVLink bidirectional speed of 900
  gigabytes per second (GB/s)**, which is more than 14x the bandwidth of the current
  PCIe Gen4 x16 bus."*

量化的后果（推导值，`code/01_bandwidth_conventions.py` 第 4 节）：搬一份 **140 GB** 的梯度
（70B 参数、BF16）单向过 H100 的 NVLink——

```text
用每 GPU 单向 450 GB/s（正确）      t = 140 GB ÷ 450 GB/s = 0.3111 s = 311.1 ms
用每 GPU 双向 900 GB/s（错，少算一倍） t = 140 GB ÷ 900 GB/s = 0.1556 s = 155.6 ms
```

**差 2 倍。** 900 描述的是「同时 450 发 + 450 收」，不是「能单向搬 900」。

这里还有两层幅度更大的放大，同样容易张冠李戴：

```text
每 link 双向 50  →  每 GPU 双向 900  →  8 卡聚合 7200 GB/s（7.2 TB/s）
```

**7.2 TB/s 是 8 张卡的收发容量之和，不是任何一个 GPU 能用的带宽。**
把它写成「每 GPU 带宽」，会让单卡推导偏 **8 倍**。

**关于 NVLink 6 的取值**：有些资料写 3000 GB/s。本主题采用 NVIDIA NVLink 规格页与
Vera Rubin NVL72 发布材料里的 **3600 GB/s**，理由是三条独立的算术都能对上：
① 它恰好是 NVLink 5 的 2 倍；② 72 × 3.6 TB/s = 259.2 ≈ 官方公布的 NVL72 scale-up 260 TB/s；
③ 3600 ÷ 36 = **100 GB/s 每 link**，与「每 link 双向」那一列 40/50/50/50/100/100 的干净序列一致
（3000 ÷ 36 = 83.3 会在这一列里唯一地破掉整数性）。NVIDIA 对 NVLink 6 标注了 preliminary。

### 第四步：十进制与二进制差 7.4%

NVIDIA 全线用**十进制**：900 GB/s 意思是 `900 × 10⁹ B/s`。换成 GiB/s 是 **838.19 GiB/s**
（推导值）。两个数的比是 `2³⁰ / 10⁹ = 1.0737`，即**同一个字节数在两种单位下的数值差 7.4%**。

（注意别算反：要问的是「同一字节数在两种单位下的数值差多少」，就该用 `2³⁰/10⁹`，
而不是 `1 − 10⁹/2³⁰ = 6.9%` 这个补数。六点几和七点几都有人在写。）

把 838 与 900 并列比较，就会凭空多出 7.4% 的差；而「140 GB」在二进制口径下是 130.4 GiB——
同一份数据，两个数都对，混用就错。

### 第五步：Gb/s 与 GB/s 之间还有一次 8 倍

跨机那一侧的数字常用 `Gb/s` 表述：EDR 100、HDR 200、NDR 400、XDR 800 Gb/s。
换算成字节要除以 8：

| 代 | 每端口 | GB/s |
|---|---|---|
| EDR | 100 Gb/s | 12.5 |
| HDR | 200 Gb/s | 25 |
| NDR | 400 Gb/s | **50** |
| XDR | 800 Gb/s | 100 |

**这个除法给出的是上界**：它没有扣链路层开销，而 IB 与 RoCE 的固定头开销**不同**
（IB 走原生链路层，开销小得多）。所以不要假设「400 Gb/s 的 IB」与「400GbE」有相同的
载荷吞吐；也不要把这个 50 与 NVLink 的 450（或 900）直接比——它们一个是单向、一个是双向。

## 2. 执行映射

数据从一张 GPU 到另一张 GPU，**经过的物理部件决定了它受哪个口径约束**。按距离分三种：

**① 节点内直连（NVLink 直连）。** GPU 的 NVLink 端口经板级线缆直连到对端 GPU 的端口。
这段链路上没有交换部件，瓶颈就是那 **25 GB/s 每方向每链路**（H100）。DGX-1 时代的
hybrid cube-mesh 就是这种形态。

**② 节点内交换（NVLink + NVSwitch）。** 每张 GPU 的 18 条链路接到 NVSwitch 上，
NVSwitch 是**全非阻塞**的。这时每张 GPU 能用满自己的 **450 GB/s 单向**，
但代价是链路不再独占——它变成共享资源（这一层在
[02-topology-and-bisection](02-topology-and-bisection.md) 展开）。

**③ 跨节点（PCIe / NIC / InfiniBand）。** 数据要先经 PCIe 到网卡，再走 IB。
这一段**同时受三个口径约束**：PCIe 侧是 `63.02 GB/s` 载荷（Gen5 x16），
网卡侧是 `50 GB/s` 单向（400 Gb/s NDR），而这两者之间还有 **GPU Direct RDMA
是否生效** 的问题——不生效就要多走一次 host 内存往返。

**P2P 什么时候不可用**（官方，源自 GPUDirect Storage Best Practices）：PCIe 的对等传输
要求 **ACS 被禁用**（否则事务被强制上行到 Root Complex）、**x86 裸机上 IOMMU 被禁用**
（例外：Grace Hopper / DGX 平台应保持开启）。没有 PCIe switch 时 `nvidia-smi topo -m`
显示 `PHB`，**那是正常结果而不是配置错误**；跨 CPU socket（`SYS`）则 P2P 不可用。
诊断命令是 `nvidia-smi topo -m` 与 `nvidia-smi topo -p2p rw`。

**数值量级**（三个可用于心算的锚点，全部是推导值）：

| 量 | 数值 | 来历 |
|---|---|---|
| H100 每 GPU 单向 NVLink | **450 GB/s** | 900 ÷ 2（§1 第三步） |
| 搬 140 GB 单向所需时间 | **311 ms** | 140 GB ÷ 450 GB/s |
| PCIe Gen5 x16 载荷 | **63.02 GB/s** | 32 GT/s × 16 × 128/130 ÷ 8 |

## 3. 实现与演进

每一步都是对上一代瓶颈的回应，而**每一次回应都换了一种口径**——这正是混乱的来源。

**PCIe Gen1/Gen2（8b/10b）。** 编码开销 20%。这一代的「带宽」在营销与规格里都是载荷口径
（Gen1 x16 = 4 GB/s），因为 20% 太显眼，取整也取不掉。

**PCIe Gen3（128b/130b）。** 编码开销降到 1.54%，于是「取整」变得廉价：**从这一代起，
市面上的整数（16/32/64/128 GB/s）开始指线速率**。口径分叉就是从这一代开始的——
同一篇文档里 Gen2 的 8 GB/s 是载荷、Gen3 的 16 GB/s 是线速率，两者相差 1.5%。

**PCIe Gen4/Gen5。** 线速率翻倍，口径延续上一代。NVIDIA 的倍数表述（"over 7X"、
"more than 14x"）用的都是线速率那一列（§1 第二步）——**这是口径污染的活标本**。

**PCIe Gen6（PAM4 + FLIT）。** PAM4 把误码率从 10⁻¹² 抬到 10⁻⁶，必须上 FEC，于是带宽
重新被切成 256 B 的 FLIT，载荷回到 242/256，编码开销回升到 5.47%。**这一代的口径偏差
比前三代大 3.5 倍**，凡是把 Gen3~5 的 1.5% 直觉套到 Gen6 上的推导都会偏。

**NVLink 1 → 6。** 主线是链路数与每链路速率的交替提升：4×40 → 6×50 → 12×50 → 18×50 →
18×100 → 36×100。注意 H100 → B200 是**每 link 翻倍**（50 → 100），B200 → Rubin 是
**链路数翻倍**（18 → 36）。两条路线的工程含义不同：前者要求 PHY 支持更高频率，
后者要求封装上多引出 18 对差分信号。**但对推导来说，每 GPU 双向那一列都翻了倍，
而单向那一列也同步翻倍——口径没变，只是数变了。**

## 4. 踩坑与失败模式

### 4.1 把 NVLink 的 900 GB/s 当成单向可用带宽

**现象**：规划一个 TP=8 的张量并行方案，按「每卡 900 GB/s」估算单层 all-reduce 只要 3 ms，
但预算表上写的是 6 ms——差一倍，而且找不到原因，最后往往归咎于「NCCL 没调好」。

**成因**：NVIDIA 公布的 every-GPU 数字**全部是双向合计**（§1 第三步，两条官方原文）。
900 GB/s 描述的是「同时 450 发 + 450 收」，而一次 all-reduce 的每一步里，
每个 rank 的瓶颈在**发送**或**接收**的单向链路上。把双向当单向，会让所有
「搬 X 字节要多久」的推导统一偏 2 倍——**方向一致、倍数固定**，所以它不会在自洽性检查里暴露自己。

**规避**：凡是要算时间的公式，一律用**每 GPU 单向**。把这句话写进推导的模板里：
`t = 字节数 ÷ 单向带宽`。要复核就运行 `code/01_bandwidth_conventions.py` §4，
它会把同一份 140 GB 在两个口径下的时间并排打印出来（311.1 ms vs 155.6 ms）。

### 4.2 把 7.2 TB/s 说成「每 GPU 带宽」

**现象**：文档里写「HGX H100 提供 7.2 TB/s 的 GPU 间带宽」，读者按单卡理解，
于是推算出「搬 140 GB 只要 19.4 ms」——比真实值快 16 倍。

**成因**：7.2 TB/s = 8 × 900 GB/s，是**整机 8 张卡的收发容量之和**（§1 第三步的三级放大）。
更糟的是它是**双向**口径，所以错得是 8 倍与 2 倍的乘积。而这两个数在文档里长得一模一样
（都是「900 GB/s」的整数倍），肉眼扫过去分不出来。

**规避**：任何带宽数字都要带主体与方向两个后缀。本主题的写法是
`每 GPU 双向 900 GB/s`、`整机聚合 7.2 TB/s`——**「整机聚合」四个字必须写出来**。
这也是 `specs.py` 里常量名一律带 `_PER_GPU_BIDIR` / `_AGG` 的原因：
命名里省掉口径，就等于把错误藏进了不可见的假设。

### 4.3 在同一篇推导里混用线速率与载荷

**现象**：用 NVIDIA 的倍数（"over 7X PCIe Gen5"）反推 PCIe 侧的字节数，
算出来的传输时间总是比预期短 1.5%；到 Gen6 上变成短 5.5%，于是误判成「PCIe Gen6 足够快」。

**成因**：NVIDIA 的除数是**线速率**（128），而你能观测到的传输速度受**载荷**（126.03）约束。
两者相差 1.54%（Gen6 是 5.47%）。这个偏差不大，但它是**单边**的：
永远朝「高估 PCIe 能力」的方向偏（§1 第二步）。

**规避**：把「线速率 / 载荷」的选择固定成一条规则——**凡是算字节数用载荷，凡是引用厂商倍数
先问它用的是哪一个**。Gen6 上尤其要小心，因为 5.47% 已经不能当噪声忽略。

### 4.4 用 2³⁰ 处理 NVIDIA 的带宽数字

**现象**：一份容量规划里，NVLink 带宽按 838 GB/s 算，显存放按 80 GiB 算，
最后「通信/计算比」偏了 7.4%，但每一步都算对了。

**成因**：GB/s 与 GiB/s 是两个单位，比值 `2³⁰/10⁹ = 1.0737`（§1 第四步）。
NVIDIA 的带宽、显存、消息大小全线十进制；而很多软件栈（以及人的直觉）用二进制。

**规避**：全篇统一十进制，并在第一次出现处标一句「本文 GB = 10⁹ 字节」。
要写成二进制就必须**全篇**都写二进制——混用的代价不是一个常数偏差，
而是每一处换算都要单独判断用哪个，最终一定会有一处漏掉。

### 4.5 PCIe P2P 静默失败：算得好好的带宽一个字节也没走

**现象**：拓扑推导显示两条链路之间应该有 63 GB/s 的 P2P 带宽，程序也能启动，
但 GPU 侧看到的是 hang 或者（更隐蔽的）数据错。

**成因**：P2P 的可用性取决于两个**平台配置**而不是链路能力——ACS 未禁用时事务被强制
上行到 Root Complex，x86 裸机上 IOMMU 未禁用时地址翻译会拦掉对等访问（§2）。
**而且失败是静默的**：PCIe 的 peer 写是无 ACK 的 posted transaction，
丢包只会体现在 host kernel log 里，GPU 侧没有任何直接反馈。

**规避**：上机第一件事是 `nvidia-smi topo -m` 看拓扑类型（`PHB` 是正常的，
`SYS` 表示跨 socket、P2P 不可用），再用 `nvidia-smi topo -p2p rw` 查一遍可达性；
在 x86 上确认 ACS 与 IOMMU 的状态。**不要用「程序没报错」当作 P2P 生效的证据。**

## 编译与运行

进入 `code/` 执行 `make run` 即可复现本文全部数字：

```bash
cd code
make run                      # 依次运行 01~04 四个程序
make 01_bandwidth_conventions # 只跑本文对应的程序（省略 .py 后缀）
```

配套程序是 **`code/01_bandwidth_conventions.py`**（Python ≥ 3.10 + numpy ≥ 2.0，
纯数值换算，不需要 GPU）。它的第 1、3 节打印本文的两张主表，第 2 节打印 NVIDIA 倍数的
口径反推，第 4 节打印 140 GB 在两个口径下的耗时差。全部规格常量集中在
`code/specs.py`，每条带 `# 来源：` 与口径注释。

## 6. 小结与延伸

本文只有一条主线：**「带宽」不是标量，是「数字 + 主体 + 方向 + 编码口径」四元组。**
围绕它有三个可独立复用的中间结论：

1. **PCIe 有三个口径**：线速率、载荷、营销取整。编码开销是 Gen1/2 的 20%、
   Gen3~5 的 1.54%、Gen6 的 5.47%；而「取整 vs 载荷」的差额是 0% / 1.54% / 5.47%。
   **这两组数不是同一个东西**，混用会把 20% 套到 Gen3 上，得到 12.8 GB/s 这个错值。
2. **NVLink 的官方数字全是双向合计**，且有三层放大（每 link → 每 GPU → 整机聚合）。
   做时间推导用单向 450 GB/s，做能力对比用双向 900 GB/s，**两者都对，不能互换**。
3. **NVIDIA 的倍数用线速率口径**，所以引用它的倍数会系统性地少算 1.5%（Gen6 是 5.5%）。

下一篇 [02-topology-and-bisection](02-topology-and-bisection.md) 承接这里的「每 GPU 双向 900 GB/s」，
回答一个口径本身回答不了的问题：**8 张卡都标 900 GB/s，为什么接成 ring 和接成 NVSwitch
全互联，集合通信能差 4 倍？** 答案是 bisection，而不是链路带宽。
[03-collectives-on-topology](03-collectives-on-topology.md) 把 bisection 变成集合通信的时间下界，
[04-measuring-bandwidth](04-measuring-bandwidth.md) 则回答这些规格值离可达值还有多远。

## 参考

- NVIDIA, *NVLink and NVLink Switch*（官方规格页）—— 逐代的「每 GPU 双向带宽」与
  「每 GPU 最大链路数」两列，以及 "over 14x the bandwidth of PCIe Gen6" 一句。
- NVIDIA, *NVIDIA Hopper GPU Architecture*（官方架构页）—— "900 gigabytes per second
  (GB/s) bidirectional per GPU, over 7X the bandwidth of PCIe Gen5" 原文。
- NVIDIA, *Introducing NVIDIA HGX H100*（官方博客）—— 900 GB/s 与 PCIe Gen4 x16 的倍数原文。
- NVIDIA, *NVIDIA Ampere Architecture In-Depth*（官方博客）—— "PCIe Gen 4 … providing
  31.5 GB/sec vs. 15.75 GB/sec for x16 connections" 原文，本文 §1 第一步的强交叉验证。
- PCI-SIG, *PCIe 6.0 Specification Webinar Q&A: Error Detection and Correction with FEC*
  与 PCIe 规范 —— 128b/130b 自 Gen3 引入、8b/10b 用于 Gen1/2、FLIT 的
  236 + 6 + 8 + 6 = 256 B 字段拆分。（PCI-SIG 官网受 Cloudflare 拦截，
  本次未取到一手页面，引文来自 PCI-SIG 博客与 Wikipedia「PCI Express」条目的交叉一致。）
- NVIDIA, *GPUDirect Storage Best Practices* —— P2P 对 ACS / IOMMU 的要求与
  `nvidia-smi topo` 的诊断用法。
- 本仓库 [10-foundations/performance-models/02-amdahl-and-scaling](../../10-foundations/performance-models/02-amdahl-and-scaling.md)
  —— 140 GB 这个场景量的出处，以及 NVLink 与 IB 两侧的饱和通信时间。

---
title: 测带宽：标称与可达之间的 gap 怎么分解
slug: interconnect-measuring-bandwidth
domain: 20-gpu-architecture
topic: interconnect
level: 3
difficulty: intermediate
prerequisites:
  - 20-gpu-architecture/interconnect/01-bandwidth-conventions
  - 10-foundations/performance-models/02-amdahl-and-scaling
tags: [nccl, nccl-tests, nvbandwidth, busbw, algbw, latency, pinned-memory]
related:
  - 20-gpu-architecture/interconnect/03-collectives-on-topology
  - 110-communication/collectives
  - 110-communication/overlap
code:
  - code/04_measuring_bandwidth.py
status: stable
updated: 2026-09-15
---

# 测带宽：标称与可达之间的 gap 怎么分解

> 「规格 900 GB/s，跑出来多少」这个问题没有单一答案，因为缺口来自四个互不相干的地方：
> **编码/协议开销**、**方向口径**、**算法搬运了多少遍**、以及**消息太小所以延迟说了算**。
> 本文把这四处分开，并指出每一处该用哪个指标去看。
>
> **全文的数字口径**：本文的数字分三类——**规格值**（引自 NVIDIA / PCI-SIG 官方来源，
> 见参考）、**外部引用值**（引自本仓库 `10-foundations/performance-models`，本文不重推）、
> 与**推导值**（由 `code/specs.py` 的规格常量代入 §1 的公式算出）。**本文无实测数据**——
> 本文一行硬件都没碰，讲的是「该怎么比」。运行 `code/04_measuring_bandwidth.py`
> 可复核本文引用的每一个数。

## 0. 背景与动机

一个示意的场景（**下面的数字是为了说明口径而设的，不是本文或本仓库测出来的**）：
某次 8 卡 H100 上的 all-reduce，`nccl-tests` 报出 **busbw ≈ 390 GB/s**。
旁边的人说「才 43%，NVLink 明明有 900 GB/s」。这个结论有**两处**错，而且方向相反：

- 用错指标：应该看 busbw 而不是 algbw（这一处如果错了，误差是 1.75 倍）；
- 用错分母：busbw 的口径是**单向**，上限是 **450 GB/s** 而不是 900（这一处错了，误差是 2 倍）。

**390 ÷ 450 = 87%，这其实是一个相当好的数字。** 两处错误叠在一起，
把一个 87% 的健康结果读成了 43% 的糟糕结果——而两处错误都「有据可依」：
algbw 确实是 nccl-tests 的输出字段，900 GB/s 确实是 NVIDIA 规格页上的数字。

本文要给出的是**判据**而不是数字：每个指标对应哪个口径、每一项 gap 该归因到哪里、
以及「多小的消息算小」这个分界点怎么解出来。

## 1. 原理

### 第一步：algbw 与 busbw——只有一个是「链路口径」

`nccl-tests` 对一次操作测出耗时 t 与消息大小 S，输出两个数：

```text
algbw = S / t                    ← 算法带宽：只跟「用户看到的字节数」有关
busbw = algbw × 修正系数          ← 总线带宽：想把「链路真实搬运量」还原出来
```

修正系数不是拍脑袋定的，它等于**该算法每个 rank 的搬运量 ÷ S**。以 ring all-reduce 为例
（推导，不是引用）：

```text
已知（引用 performance-models/02，不重推）：每卡搬运量 V = 2(n-1)/n × S
若链路恰好跑在它的**单向**容量 BW 上：t = V / BW

algbw = S / t = S × BW / V = BW × n / (2(n-1))
busbw = algbw × 2(n-1)/n = **BW**
```

**这是一个恒等式：完美实现的 ring all-reduce，其 busbw 恰好等于单向链路带宽。**
所以 busbw 的物理上限是 **450 GB/s**（H100 的单向），**不是 900 GB/s**（双向）。
反解一遍就知道两个指标差多远：busbw 打到 450 GB/s 时，**algbw 只有 257.1 GB/s**（n = 8），
比值 450.00 / 257.14 = **1.7500**——正好是 NCCL 文档给 AllReduce 的 `2(n-1)/n`，
两边对得上。

这就把 §0 那两处错误定量化了：

| 读法 | 分母 | 结果 |
|---|---|---|
| 用 algbw 比 900 GB/s | 900 | 假效率 **29%**（真值 100%） |
| 用 busbw 比 900 GB/s | 900 | 假效率 **50%**（真值 100%） |

### 第二步：修正系数随算子变，所以报数字必须连算子一起报

NCCL 官方给的系数（规格值）：

| 操作 | 修正系数 | n = 8 时 |
|---|---|---|
| AllReduce | `2(n-1)/n` | 1.7500 |
| ReduceScatter | `(n-1)/n` | 0.8750 |
| AllGather | `(n-1)/n` | 0.8750 |
| AlltoAll | `(n-1)/n` | 0.8750 |
| Broadcast | `1` | 1.0000 |
| Reduce | `1` | 1.0000 |

同一批原始数据配错系数，读数会**腰斩**：algbw = 257.1 GB/s 时，用 AllReduce 的系数得
450.0 GB/s，用 AllGather 的系数只有 225.0 GB/s。**所以「busbw 400 GB/s」这句话本身
是不完整的**——没有系数就没有口径，没有算子就没有系数。

### 第三步：busbw 在 NVLS / SHARP / COLLNET 下会高估

第一步那个恒等式 `busbw = BW` 的前提是「每卡搬运 `2(n-1)/n × S`」——
**这是 ring 算法的搬运量**。当归约发生在交换机里（NVLS / SHARP / collnet），
点到点的搬运量不再服从这个式子，而修正系数**仍然是 ring 的那个**。

用一个显式模型算一遍（**下面是假设的模型，不是 NCCL 的实现细节**）：

```text
假设：每卡只把 S 字节发到交换机一次、再从交换机收回 S 字节（归约在交换机内完成）
      ⟹ 每卡搬运量 ≈ S，而不是 1.75 × S
这个模型下 algbw   = S / (S / BW) = BW
报出来的 busbw = algbw × 1.75 = 1.75 × BW     ⟹ 超出单向链路带宽 75%
```

具体到 H100：algbw ≈ 450 GB/s，报出来的 busbw ≈ **787.5 GB/s**，
而任何一条链路的单向容量都只有 450 GB/s。**busbw 超过单向链路带宽不一定是错**，
但它意味着**这个数不再是链路口径**——它是「算法把搬运量压下去之后，那个旧系数放大出来的
一个虚数」。NVIDIA 维护者也已经表示 busbw 现在只是「algbw 乘一个比例」。

所以：拿 busbw 与硬件峰值对比时，**必须同时说明算法**（ring 还是 NVLS/SHARP），
否则这个对比没有意义。这也是 §0 那个 87% 的解读能成立的前提——它得是 ring 或 tree。

### 第四步：多小的消息算小——由 `α·BW` 解出来

延迟-带宽模型（定义在 [10-foundations/performance-models](../../10-foundations/performance-models/01-roofline-model.md)，
本文**引用不重推**）：

```text
t = α + 数据量 / BW
```

两层含义，都是解同一个方程。

**(a) 单次点对点传输**：两边相等时 `α = size/BW`，解出

```text
size* = α × BW        ← 小于它的消息，延迟项比数据项还大
```

**(b) 一次 n 卡 ring all-reduce**：延迟项是 `2(n-1)·α`，数据项是 `2(n-1)/n · S/BW`，
两边相等时 `2(n-1)` 直接约掉：

```text
S* = n × α × BW       ← 比单次点对点多出 n 倍，因为每个 chunk 只有 S/n
```

取 α = **5 μs**（外部引用值：performance-models/02 的每步延迟），代入（推导值）：

| 链路 | (a) 点对点 size* | (b) ring all-reduce S* |
|---|---|---|
| NVLink 4 单向（450 GB/s） | **2.25 MB** | **18.0 MB**（n=8） |
| IB NDR 单向（50 GB/s） | **0.25 MB** | **16.0 MB**（n=64） |
| PCIe Gen5 x16 载荷（63.02 GB/s） | **0.32 MB** | — |

**效率曲线**把「数据项占的比例」记成 η：

```text
η = (size/BW) / (α + size/BW) = size / (size + α·BW)
```

于是各效率点整齐地落在 `α·BW` 的整数倍上（NVLink 4 上）：

| η | size / (α·BW) | 在 NVLink 4 上 |
|---|---|---|
| 50% | 1 | 2.25 MB |
| 90% | 9 | 20.25 MB |
| 99% | 99 | 222.75 MB |

⚠ 两条使用边界：α 取的是 **5 μs 这个量级**（外部引用值），节点间实际更大（要过 NIC）；
**η 到 50% 的那个点不是硬件属性**，它同时取决于 α 与 BW——换一张网卡就换一个分界点。

### 第五步：把 PCIe 的 gap 逐级分解

以 PCIe Gen5 x16 单向为例，逐级往下掉（每一级都是规格值之间的换算，推导值）：

| 口径 | GB/s | 相对上一级 |
|---|---|---|
| ① 线速率（GT/s × 16 ÷ 8） | 64.00 | — |
| ② 载荷（扣 128b/130b = 2/130） | 63.02 | −1.54% |
| ③ 官方 H2D 样例（pinned） | 55.63 | −11.72% |
| ① → ③ 合计 | 55.63 | −13.08% |

**②→③ 那一档（7.39 GB/s，占载荷的 11.7%）不是编码开销**，而是协议与实现开销：
TLP 头、DLLP、流控，以及 GPU 侧拷贝引擎的实际效率。**把它和编码开销混为一谈，
就会得出「PCIe 编码效率很差」的错误归因**，进而去优化错误的地方。

⚠ ③ 是**官方样例输出**，转引自本主题的实施规格；本次未能定位一手页面（见
`code/specs.py` 的 `UNSOURCED`）。它只能当「官方口径的量级」看，不能当基准表用——
**覆盖 Gen3/4/5 的官方 H2D 达成率基准表并不存在**。

同一套分解在 PCIe Gen2 x16 上更极端（这一代的载荷是 8.00 GB/s）：

| 口径 | GB/s | 占载荷 |
|---|---|---|
| ② 载荷 | 8.00 | 100% |
| ③ pinned 内存 | 5.77 | 72% |
| ④ pageable 内存 | 2.31 | 29% |

**pinned ÷ pageable = 2.50 倍**——这是整条链路上最大的一档。但必须说清楚：
**pageable 的那一档不是 GPU 或 PCIe 的属性。** pageable 拷贝要先经驱动的 staging buffer，
它的速度随 CPU、主板、内存带宽剧烈变化，换一台机器就换一个数。

## 2. 执行映射

**两个工具测的是不同的东西，不能互相替代。**

| | `nvbandwidth` | `nccl-tests` |
|---|---|---|
| 测什么 | 点到点内存拷贝可达带宽 + 延迟（H2D / D2H / D2D、All-to-One 等） | **集合通信操作**性能 |
| 测不到什么 | 集合通信的算法效率 | **纯硬件上限**——受 NCCL 的算法、拓扑感知与调优影响 |
| 拷贝方式 | **CE（copy engine）与 SM 两种，结果可差很多** | 由 NCCL 决定 |

**CE 与 SM 是两条不同的搬运路径**，官方把这两种模式分开列出，理由是结果会不一样：
CE（copy engine）走复制引擎，不占用 SM，但受引擎数量限制；
SM 模式让 kernel 亲自搬数据，可以占满更多带宽，代价是吃掉计算资源。
**报数字时必须说明用了哪一种**，否则两个数不可比——这和 §1 第五步里
「pinned 还是 pageable 必须说明」是同一类要求。

**`nvidia-smi topo -m` 是这两类工具的输入。** 它给出的 `NV#` / `PHB` / `SYS`
决定了 NCCL 选 ring 还是 tree、走不走 NVLS；也决定了 `nvbandwidth` 的 P2P 用例会
拿到什么结果（`PHB` 是正常的，`SYS` 表示 P2P 不可用——见
[01-bandwidth-conventions](01-bandwidth-conventions.md) §2）。

**数值量级**（三个可用于心算的锚点，全部推导值）：

| 量 | 数值 | 来历 |
|---|---|---|
| H100 上 ring all-reduce 的 busbw 上限 | **450 GB/s** | = 单向链路带宽（§1 第一步的恒等式） |
| 同一次操作对应的 algbw | **257.1 GB/s** | 450 × n/(2(n-1))，n=8 |
| 8 卡 NVLink all-reduce 的「小消息」分界 | **18.0 MB** | n × α × BW（§1 第四步） |

## 3. 实现与演进

**测点对点（`bandwidthTest` / `nvbandwidth`）。** 起点是「这条链路能搬多少」。
这一层的关键是把口径钉死：方向（H2D / D2H / D2D）、内存类型（pinned / pageable）、
拷贝方式（CE / SM）。**三个都不说，数字就没有意义。**

**测集合通信（`nccl-tests`）。** 第二步把「算法」放进来了：这时你测的不再是链路，
而是「算法 × 拓扑 × 调优」。algbw 与 busbw 的分工就出现在这里——
**busbw 想做的是把算法因子除掉，还原回链路口径**（§1 第一步）。

**测拓扑感知（NVLS / SHARP / collnet）。** 再进一步，归约下沉到交换机，
`2(n-1)/n` 这个因子被抹掉了。**代价是 busbw 这个指标失去了它的还原对象**（§1 第三步）。
走到这一步，「测带宽」这件事本身要重新定义：**当搬运量不再是 `1.75S` 时，
「带宽」这个说法必须换成「每秒完成的归约元素数」之类的操作性定义。**

**贯穿三个阶段的一条线**：**指标的口径必须与被测对象的搬运量一致。**
点对点阶段是「链路搬了多少」，集合通信阶段是「算法搬了多少」，
拓扑感知阶段是「交换机替算法省了多少」。**每换一个阶段，旧指标就退化成
「algbw 乘一个比例」**——它还能用，但不能再拿去比硬件峰值。

## 4. 踩坑与失败模式

### 4.1 用 algbw 与硬件峰值对比

**现象**：报告里写「all-reduce 只跑到 257 GB/s，NVLink 有 900 GB/s，利用率 29%，
说明 NCCL 有严重问题」，于是花两周去调 NCCL 环境变量，没有任何改善。

**成因**：algbw 是「用户看到的字节数 ÷ 时间」，而 ring all-reduce 每个 rank 实际搬运
`2(n-1)/n × S`——8 卡时是 **1.75 倍**。algbw 天然比链路口径小 1.75 倍（§1 第一步）。
再叠上「900 是双向」这一层，29% 这个数字里没有一个因子是真的。

**规避**：与硬件峰值对比只用 **busbw**，且分母取**单向**（H100 = 450 GB/s）。
本节的例子里 algbw = 257.1 GB/s 对应的 busbw 是 450.0 GB/s——**满速**。
要复核就跑 `code/04_measuring_bandwidth.py` §1，它把这两个数并排打印出来。

### 4.2 用 busbw 与 900 GB/s 对比

**现象**：看到 busbw = 430 GB/s，评价「只有 48%，离 900 还差一半」，
于是把优化目标定成「翻倍」。

**成因**：busbw 的口径是**单向**——它等于「每卡搬运量 ÷ 时间」，而搬运发生在单向链路上。
900 GB/s 是**双向合计**（450 发 + 450 收）。430 对 450 已经是 **96%**，
**优化目标本身设错了**（§1 第一步）。

**规避**：把「900 只用来做能力对比，450 才用来算时间与利用率」写成一条硬规则。
这条与 [01-bandwidth-conventions](01-bandwidth-conventions.md) §4.1 是同一个坑的
两种表现——一个是算时间，一个是算利用率。

### 4.3 报了数字却没说是哪种内存/哪种拷贝

**现象**：两份报告都写「PCIe Gen5 H2D 55 GB/s」与「PCIe Gen2 H2D 2.3 GB/s」，
看起来是同一套方法测的，实际前者是 pinned + CE，后者是 pageable + SM，
**两个数不能放在一起比较**，更不能用来推导「Gen5 是 Gen2 的 24 倍」。

**成因**：pinned ÷ pageable 的差是 **2.50 倍**（§1 第五步），CE 与 SM 的差同样是数倍。
这两组差异与 PCIe 代数**无关**，却和代数差异（Gen2 → Gen5 是 8 倍）
叠在同一个数上，让人误以为是链路能力的变化。

**规避**：报带宽数字时按固定格式写全三要素：**方向 / 内存类型 / 拷贝方式**，
缺一项就标「未说明」。**尤其不要用 pageable 的数字去推断硬件能力**——
它不是 GPU 或 PCIe 的属性。

### 4.4 把小消息的性能当成链路的性能

**现象**：为了快速验证，用 1 MB 的消息测 all-reduce，得到 120 GB/s，
判断「这条链路不行」——而实际在大消息上它是满速的。

**成因**：8 卡 NVLink 上的小消息分界点是 **18.0 MB**（§1 第四步）。
1 MB 远在分界点以下，此时耗时由**每步 5 μs 的固定延迟**主导（2(n-1) 步累计），
数据项根本不是瓶颈。**分界点不是硬件属性**——它同时取决于 α 与 BW，
IB 上（n=64）是 16.0 MB，量级相近但成因不同。

**规避**：测吞吐要用**远大于分界点**的消息（10 倍以上，即 NVLink 上 ≥ 180 MB），
测延迟才用小消息。**并且要意识到分界点会随 NIC、随 CPU 平台移动**——
换一台机器，分界点就不是这个数了。

## 编译与运行

进入 `code/` 执行 `make run` 即可复现本文全部数字：

```bash
cd code
make run                      # 依次运行 01~04 四个程序
make 04_measuring_bandwidth   # 只跑本文对应的程序（省略 .py 后缀）
```

配套程序是 **`code/04_measuring_bandwidth.py`**（Python ≥ 3.10 + numpy ≥ 2.0，
纯口径推导，不需要 GPU——**它一行硬件都没碰**）。第 1 节推 `busbw = 单向带宽` 这个恒等式
并反解 algbw；第 2 节打印 NCCL 的修正系数表；第 3 节算 NVLS 下的高估幅度；
第 4 节解出小消息分界点与效率曲线；第 5 节做 PCIe 的逐级 gap 分解；
第 6 节给出一页「该怎么比」的对照表。全部规格常量集中在 `code/specs.py`，
每条带 `# 来源：` 与口径注释。

## 6. 小结与延伸

本文的主线是：**「跑出来多少」这个问题必须先问「分子是哪个口径、分母是哪个口径」。**
三个可独立复用的中间结论：

1. **`busbw = algbw × 该算子的修正系数`，而完美 ring 的 busbw 恰好等于单向链路带宽。**
   所以：与硬件峰值对比只用 busbw，分母取**单向**（H100 = 450 GB/s）；
   用 algbw 或拿 900 当分母，都会得到**看起来像问题**的假结论。
2. **小消息的分界点由 `α·BW` 解出**，而 ring all-reduce 的对应量是 `n·α·BW`：
   8 卡 NVLink 上是 **18.0 MB**。它不是硬件属性，换网卡就换分界点。
3. **gap 要按来源分开**：PCIe Gen5 上编码开销 1.54%、协议与实现开销 11.72%、
   pinned↔pageable 是 2.50 倍。**三者的优化手段完全不同**，混为一谈就会去优化错的地方。

本文是 interconnect 主题的最后一篇。回到第 01 篇的那句话：互联的所有「2 倍错误」
都来自口径——**到这里它有了一个测量版本：所有「利用率只有一半」的结论，
先检查分子与分母是不是同一口径，再检查分母是单向还是双向。**
[03-collectives-on-topology](03-collectives-on-topology.md) 给出了分母侧最硬的那个数
（bisection 下界），[01-bandwidth-conventions](01-bandwidth-conventions.md)
给出了所有分子侧的口径定义。

## 参考

- NVIDIA, *NCCL / nccl-tests 官方文档* —— `algbw` / `busbw` 的定义与逐算子的修正系数表
  （AllReduce 为 `2(n-1)/n`，ReduceScatter / AllGather / AlltoAll 为 `(n-1)/n`，
  Broadcast / Reduce 为 `1`），以及「busbw 可与硬件带宽比较」的原始表述。
- NVIDIA, *nvbandwidth 文档与 GPUDirect 相关材料* —— CE 与 SM 两种拷贝方式的区分、
  H2D/D2H/D2D 用例的划分。
- NVIDIA, *GPUDirect Storage Best Practices* —— pinned / pageable 的差异来源，
  以及 P2P 对 ACS / IOMMU 的要求。
- NVIDIA 官方样例输出（PCIe Gen5 x16 / Gen2 x16 的 H2D 达成率）—— 转引自本主题实施规格，
  **本次未定位一手页面**，见 `code/specs.py` 的 `UNSOURCED`。
- 本仓库 [10-foundations/performance-models/01-roofline-model](../../10-foundations/performance-models/01-roofline-model.md)
  与 [02-amdahl-and-scaling](../../10-foundations/performance-models/02-amdahl-and-scaling.md)
  —— 延迟-带宽模型的定义与 5 μs 每步延迟这个量级（**本文引用，不重推**）。

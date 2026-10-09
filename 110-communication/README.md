# 110-communication 通信与网络

分布式系统的物理约束在这里。通信算法的复杂度与网络拓扑决定了并行的可扩展性上界。

## 本域覆盖什么

| 主题 | 内容 |
|---|---|
| collectives | ★★ **集合通信的算法族**（ring / recursive doubling / halving-doubling / Bruck / binomial tree / Van de Geijn）与各自的**步数与每卡搬运量** · reduce-scatter / all-gather 的**阶段划分** · **all-to-all** 的载荷口径 · ★ **P2P 原语**（`send` / `recv` / `sendrecv`） |
| nccl | NCCL 的**算法取值**（`NCCL_ALGO`）与**协议**（`NCCL_PROTO`）· **拓扑感知**（`NCCL_TOPO_FILE` / `NCCL_DEBUG_SUBSYS=GRAPH`）· **报错归因**与错误码 —— ★★ **本主题【全靠 `[B]` 级】**（本机无 GPU / 无 NCCL） |
| network-topology | ★ **胖树 / 三层 Clos / rail-optimized / 多轨道**的**组合结构**（`k³/4` 主机、`(k/2)²` 核心、等价路径数）· **RoCE v2 / PFC / ECN / DCQCN** · InfiniBand 的 **SHARP**（在网计算） |
| overlap | 通信-计算重叠的**机制**（双缓冲 / 分块 / 异步发起）· ★★★ **「能藏多少」的判据** · **异步落盘**（与计算重叠的那一半） |

★ **NVLink 与 PCIe 的带宽 / 拓扑【不在本域】** —— 正文在 `20-gpu-architecture/interconnect/01,02`，此处只回链。

## 主线

> »**通信有四层，每一层有自己的「不可能」 —— 把它们混为一谈，就会写出比物理下界还快的「时间」。**«

| 层 | 它的约束 | 本机可验性 | 本库的正文在哪 |
|---|---|---|---|
| **算法** | `V(N) = 2(N−1)/N × S`（饱和到 `2S`） | ★ **可算**（算法可逐步模拟） | `10-foundations/performance-models/02:147-155,209` —— ★★ **本域不重推，只回链** |
| **实现** | NCCL 选哪个算法 / 协议、怎么感知拓扑 | ✗ **不可验**（无 GPU / 无 NCCL） | ★★ **本域 `nccl`（只有 `[B]` 级）** |
| **物理** | bisection 下界 `t ≥ 2S/B` | ★★ **组合那一半【全可算】**（路径数 / `k³/4` / rail 划分 **可与外部论文值对拍**）；★ 带宽是规格值 | ★★ **本域 `network-topology`**（`01`/`02` **有真 A 类**，锚点 = SIGCOMM 2008 / NVIDIA）+ `20-gpu-architecture/interconnect/02,03`（★ **NVLink / bisection 的部分只回链**） |
| **时间** | 首项不可重叠 ⇒ 总时间 **`≥ max(C, T)`** | ▲ 可模拟（离散事件） | ★ **本域 `overlap`（`[E]` 级）** |

★★★ **推论（本域最要紧的一条）**：**四层之间不能互相推导** ——
「算法层的代数」推不出「实现层的性能」，「本机 socket 的实测」推不出「物理层的带宽」。
★ **每篇文档都在抬头写明它写的是哪一层。**

## 与其他域的关系

- **上游**：`20-gpu-architecture/interconnect`（NVLink / PCIe 的物理带宽与拓扑）· `10-foundations/performance-models`（`V(N)` 的推导与扩展效率模型）。
- **下游**：`100-parallelism`（并行策略的可行性由通信决定）。
- **横切**：`00-meta/glossary.md` 的「互联与多卡」节 —— ★ **那 10 条词条的回链在本域**（收尾时补）。

★★ **上游契约的分布【极不均等】**（★ **口径**：`grep -rn '110-communication/<主题>'`，排除 `docs/` 与 `.git`
⇒ **即【别的域】指向本域的引用**；★ 用别的 pattern 会得到别的数，**所以口径必须一起写**）：

| 主题 | 承接的让渡 | 结论 |
|---|---|---|
| `collectives` | ★ **84 处**（横跨 `10` / `20` / `40` / `100` 四个域） | 最重的一章 |
| `overlap` | ★ **23 处**（`100-parallelism` 的五个主题 + `90-training/checkpointing`） | ★ 含核心判据让渡 |
| `nccl` | ★ **0 处** | ★★ **自建**（库内先例：`90-training` 的 `initialization-and-stability`） |
| `network-topology` | ★ **0 处**（且上游**反向声明** NVLink 拓扑归它） | ★★ **自建** |

## 本域不做什么（硬边界）

下表是**硬边界**：本域的主题**不写**「本域不写」那一列的内容，只在正文里**回链**。
回链的注记方向按 `CONTRIBUTING` §7（「正文在 X，此处回链」与「正文在本域，回链 Y」两种，**不可混用**）。

| 主题 | 本域**不写** | 那里的正文 |
|---|---|---|
| **全主题** | ★★★ **集合通信原语的算法与复杂度**（ring / tree allreduce、`V(N) = 2(N−1)/N × S` 的**推导**、`S` 的饱和、reduce-scatter / all-gather 的阶段划分的**代数**） | `10-foundations/performance-models/02-amdahl-and-scaling.md:147-155,209` —— ★★ **本域只写「算法族」本身**（有哪几种、各自步数与搬运量、怎么选） |
| **全主题** | ★★ **拓扑受限的集合通信**（**bisection 下界** `t ≥ 2S/B`、**all-to-all 的跨切口量** `N·V/4`、分层 all-reduce 的三段式与平衡条件） | `20-gpu-architecture/interconnect/03-collectives-on-topology.md` —— ★★ **本域只写 all-to-all 的【总搬运量 / 载荷口径】** |
| **全主题** | ★★ **链路带宽的三种口径**（线速率 / 载荷 / 营销取整）与 **NVLink / PCIe / IB 的规格值**、**NVLink 域逐代**、**NVSwitch** | `20-gpu-architecture/interconnect/01-bandwidth-conventions.md` 与 `02-topology-and-bisection.md` |
| **全主题** | **测量口径**（`algbw` vs `busbw` 与逐算子修正系数表、小消息分界点） | `20-gpu-architecture/interconnect/04-measuring-bandwidth.md` |
| **全主题** | **Amdahl / Gustafson / `N₅₀` / 扩展效率的模型** | `10-foundations/performance-models/02-amdahl-and-scaling.md`（**全篇**） |
| **全主题** | ★★ **通信压缩 / 低精度 all-reduce / PowerSGD** | ★ **本域不做** —— 它是「**通信量**」侧的优化，不是本域四层里的任何一层；★ 见 `docs/superpowers` 计划 `§4` 的 `T65`/`T66` |
| **全主题** | **框架的分布式 API 语义**（`torch.distributed` / `DDP` / `FSDP` 的类与参数、`ncclGroupStart` 的**用法**） | ★ **全库当前没有正文** ⇒ 本域**只引 NCCL/NCCL 官方文档的取值与语义**（标 `[B]` + 版本），**不写框架的使用教程** |
| **全主题** | **推理侧的通信**（KV cache 的传输、PD 分离、推理的 TP/PP/EP） | ★ **已建成**：`120-inference/serving-infra/02-inference-parallelism-and-pd`（**推理侧的通信【需求】**）；★ **通信的机制**仍在本域（本域只被回链，不反向让渡） |
| `overlap` | **异步落盘的【存储侧】机制**（写盘流水、元数据、重分片） | `90-training/checkpointing/01,02,03` —— ★★ **本域只写「通信 / 计算与落盘重叠」的那一半** |
| `network-topology` | ★★ **NVLink 域 / NVSwitch / PCIe 的带宽与拓扑** | `20-gpu-architecture/interconnect/02`（★ 上游已逐代写好） |
| **全主题** | ★ **`00-meta/glossary.md` 已有的相关词条**（`all-reduce` / `all-to-all` / `bisection bandwidth` / `NVLink Domain` / `algbw`-`busbw` / `P2P` / `InfiniBand` / `scale-up`-`scale-out`） | `00-meta/glossary.md` ⇒ ★★ **本域不另立同名词条**；建成后**补回链**（收尾办） |

## 四层与「同名不同指」

★★ 本域要用的词里，**六个在库内已有别的含义**（详见计划的 `§2.2`）：

| 词 | 库内已有的含义 | 本域的处置 |
|---|---|---|
| **`stage` / 阶段** | TVM 计算阶段 · cutlass 流水级数 · 流水线并行的 stage · ZeRO 阶段（**四个**） | ★★ 一律写「**通信阶段**」或「reduce-scatter / all-gather **两段**」，**不得裸写 `stage`** |
| **`ring`** | ring 拓扑（`interconnect/02` 的 `bisection(ring)`）· ring attention（`100-parallelism`） | ★★ 一律写「**ring allreduce**」全称 |
| **`tree`** | AST / 计算图的 tree（`60-compilers`） | ★ 一律写「**tree allreduce**」全称 |
| **`rank`** | **张量秩**（`60-compilers/xla` 的 `rank-0`） | ★ 首次出现处注明；**尽量写「设备」** |
| **`bucket`** | **梯度分桶**（DDP）· **形状分桶**（JAX） | ★★ 本域一律写「**通信分块**」（chunking） |
| **`overlap`** | ★ 与 `90-training` 的「**异步落盘**」**是同一件事的另一面** | ★ 首次出现处**点明两者是同一件事**（**不是**同名不同指） |

## 代码约定

- **编号程序**：`.py`，**默认 `python3` 可跑**；**不 import 任何第三方框架**，`numpy` 与标准库（`socket` / `multiprocessing`）为限。
- ★★ **多进程只进 `probes/`**（★ 实测：ring allreduce 的原型**连跑通都不稳**，约 80% 概率挂）——
  编号程序一律**单进程**，保证 `make run` 快且稳。
- **三类断言（A / B / C）缺一不可**，全部**在程序内部**，失败以非零退出码结束。**不设 `verify.py`**。
- ★★★ **本域最危险的判据是「把上游已经推导好的东西再算一遍」** —— 那是 `P12`（自证）。
  ★ 本域的 A 类钉在**算法族的步数与搬运量**上（参照是**消息序列模拟**，与闭式是**两条独立路径**）。
- ★★ **`nccl` 与 `network-topology/03` 没有 A 类**（本机无 NCCL / 无网卡）⇒ **写明不可测，不得把「常量钉住」冒充 A 类**；
  ★★★ **而 `collectives`、`network-topology/01,02` 有真 A 类**（本机可算，锚点是**外部**的：MPICH 论文 / SIGCOMM 2008）。
- ★ **限定语**（本域最强的）：**「不是 NCCL，也不是真机」** —— `probes/` 走的是 **loopback**，
  ★ 它的**任何数字都不得**用作带宽 / 延迟 / 吞吐的依据。

## 计划主题清单见 [roadmap.md](../00-meta/roadmap.md)

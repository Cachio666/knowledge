---
title: 搜索算法与预算
slug: auto-tuning-search-algorithms
domain: 60-compilers
topic: auto-tuning
level: 3
difficulty: advanced
prerequisites:
  - 60-compilers/auto-tuning/02-cost-models
tags: [auto-tuning, ansor, triton, autotuner, search-strategy, budget, regret, pruned-configs]
related:
  - 60-compilers/tvm/04-auto-scheduling
  - 50-performance/tuning-practice/01-measuring-sigma-and-tau
  - 40-kernels/gemm/02-shared-memory-tiling
code:
  - code/03_search_trace.py
status: stable
updated: 2026-09-16
---

# 搜索算法与预算

> 「自动调优」听起来像「有个聪明的搜索算法」，但工程上最常见的那个 autotuner **根本不搜索**——Triton 的默认路径是把配置穷举一遍、各测一次、取中位数最小的那个，**默认连剪枝都没有**。而预算这件事也有个反直觉的结论：**「搜得多就更好」不是单调关系**——方向错了，加预算买不到东西。

## 0. 背景与动机

### 本文与主线的关系：**② 本文是主线的代价项**

本域的主线是**物化（materialize）还是重算（recompute）**（口径与完整展开见 [01-search-space](01-search-space.md) §0）。

**本文为什么是代价项**：主线的决策点在 [01](01-search-space.md) 里已经落到了搜索空间的**骨架层**——选 `FuseConsumer` 还是 `AddCacheStage`，就是选「重算」还是「物化」。而**做出这个选择要付一笔账**：搜索本身（编译 + 测量 + 训练模型）的机器时间。

这笔账有明确的量纲和明确的替代品：你可以不搜（用默认调度，性能差），也可以全搜（性能好，但那篇论文的原话是「typically, it takes **several hours**」，**引用值**）。所以搜索策略与预算讨论的不是「哪个算法更聪明」，而是**「这笔账付到哪里为止」**。本文把两种极端（穷举与随机）和两种中间形态（模型引导与演化）在同一份预算下量出来。

### 一个具体的对照场景

同一个 4596 个配置的迷你空间、同一个起点，两种花法：

- **穷举**：把 4596 个全测一遍——必然拿到最优，账单是 4596 次测量。
- **预算 48 的随机搜索**：从 4596 个里随机抽 48 个测——本机实测的最差 regret 是 **1.65**（即最差的那个种子找到的解比最优差 65%）。

中间那条路（用模型挑、用演化爬）才是「搜索算法」。本机实测里，**同样 48 次测量**，演化搜索的最差 regret 是 **1.33**，比随机好一截；而**模型引导的搜索在 12 次测量上**的 regret 是 **2.54**——**比「只花 6 次测量的随机搜索」（2.65）只好一点点，远不如花 12 次的随机搜索（2.08）**。这就是本文的 §4。

## 1. 原理

### 第一步：四类策略，各自的前提不同

| 策略 | 它假设什么 | 它的代价 | 谁在用 |
|---|---|---|---|
| **穷举** | 什么都不假设 | **空间基数 × 一次测量** | Triton 的默认路径（§2） |
| **随机** | 什么都不假设 | 预算那么多 | 基线；也常做暖机 |
| **模型引导** | **有一个在新图上仍然排得对的模型** | 训练 + 每次打分（近乎免费） | Ansor 的取样器（[02](02-cost-models.md)） |
| **演化** | 有一个能产生「邻近好解」的变异算子 | 预算那么多（不训练模型） | Ansor 的搜索内核、MetaSchedule 的 `EvolutionarySearch` |

**为什么「穷举」也在表里**：因为它是**这个领域的默认做法**，不是边角情形。§2 会看到 Triton 的 autotuner 在默认配置下就是这么干的。把一个主题讲成「搜索算法有很多种」而略过「最常用的那个不搜索」，会让读者在实际项目里看不懂调优日志。

**四者的前提是互斥的**：模型引导需要模型排得对；其他三条不需要。这就是为什么「先上模型」不是无脑更优——加上一个前提就多了一个失效点。

### 第二步：预算的账怎么算

设有效空间基数为 `N`，预算是 `B` 次测量。三条简单的观察：

1. **`B ≥ N` 时，穷举是严格最优的**（它能测完所有点），而且**没有任何搜索算法能超过它**——本仓库的程序把这条做成了 A 类断言：任何搜索找到的 best 都不可能低于暴力枚举的最优（本机实测：随机搜索 75784 ≥ 最优 64519；演化搜索 64519 == 最优）。
2. **`B < N` 时，问题变成「怎么用 `B` 次测量覆盖一个大空间」**。这时候三种策略的差别才出现。
3. **regret 的口径**：`regret = 搜索到的最好真实代价 / 暴力枚举最优`。它 ≥ 1，1 表示命中最优。

**★ 一条必须空着的地方**：`B` 应该取多少？**这个问题没有可引用的通用答案。** 见 §5 ——本文只能给出单篇论文在其基准上的设置，**并明确标注那不是定律**。

### 第三步：regret 只能作为本机实测值

`regret` 是个很有诱惑力的量：它看起来像「有文献给出上界」的东西。**本文不给任何界**，理由在 §5 第一条（无定论）。本文的做法是：**把首次实跑得到的 regret 写死成一个常数阈值**，然后断言搜索不超过它——这是计划里给的两个可证伪写法之一，另一个是「对照比较」。**本仓库选的是前一个（冻结阈值）**，理由在后一个上：对照比较（「模型引导优于随机，在 k 个地形上至少 m 次」）与 02 号文档的 C1 互为反面，两条都成立会让 C1 变成没说清的事；而冻结阈值直接、可证伪、且**反例能一眼看出它有牙**（§5 里那个被模型带偏的搜索就越过了阈值）。

## 2. 执行映射：Triton autotuner 的真实默认行为

这一节全部钉在 **Triton tag `v3.6.0`** 的 `python/triton/runtime/autotuner.py`（见 `code/specs.py`；最新的 tag 是 **v3.8.0**，差异见下）。

**默认路径是穷举，不是搜索**：对（剪枝后的）**全部** config 逐个 `_bench`，再

```text
min(timings, key=timings.get)
```

——取**中位数**最小的那个。**默认没有任何剪枝**：`perf_model=None`、`configs_top_k=1.0`（注意属性名是 `configs_top_k`，`top_k` 只是 `prune_configs_by` 字典的键）、`early_config_prune=None`、`prune_configs_by=None`。

**其他六条默认行为**（逐条都是本仓库程序里可观察的）：

| 行为 | 事实 |
|---|---|
| 只有一个 config 时 | **不走调优路径**（`if len(self.configs) > 1:`）；configs 为空时默认 `Config({}, num_warps=4, num_stages=3, num_ctas=1)` |
| key | 参数名列表；运行时 key = 这些参数本次调用的取值元组，**且每个带 dtype 的参数会把 dtype 字符串追加进 key**；key 不在 cache 里才重新调优 |
| 内存缓存 | 一直有（`self.cache: Dict[Tuple, Config]`） |
| **磁盘缓存** | **默认关闭**（`cache_results=False`）；开启方式 `cache_results=True` 或 `knobs.autotuning.cache`；落盘名 `{kernel_name}.autotune.json`；**带 `pre_hook` 的 config 一律不落盘** |
| 计时 | `do_bench`，`quantiles=(0.5, 0.2, 0.8)`，用**中位数**比较 |
| **编译失败** | `OutOfResources` / `CompileTimeAssertionFailure` / `PTXASError` 返回 **`[inf, inf, inf]` 三元组**，**不抛异常**——等价于自动出局 |

> ⚠ **两处最容易写错的地方**：
> 1. **不得写「Triton 会把 autotune 结果缓存到磁盘」**——默认配置下不成立（需要显式 `cache_results=True`）。内存缓存是另一回事，它一直有。
> 2. **编译失败返回的是 `[inf, inf, inf]`（三元组），不是单个 `inf`**——这一点在 [01](01-search-space.md) 的坑 3 里讲过它为什么重要：`min` 会跳过 `inf`，但会选中一个「很大的有限哨兵」。

**版本注记**：「磁盘缓存默认关闭」在 **v3.4.0–v3.8.0 全部成立**（3.3.0 及更早**根本没有**磁盘缓存）；**v3.8.0 新增了「剪枝后只剩一个 config 时的短路」**——v3.6.0 没有这一步，所以「先剪枝再短路」这个组合行为在两个 tag 上不一样。**凡写 Triton 的默认行为必须带版本。**

**为什么这个设计在工程上是合理的**：autotuner 面对的是**一个 kernel 的几个到几十个 config**（`num_warps` / `num_stages` 这种），不是 Ansor 那种几千个程序的搜索空间。在这个量级上，「穷举 + 取中位数」比「引入一个模型」更简单、更可预测，而且**没有模型前提**。**规模不同，策略不同**——这正是 §1 那张表的读法。

## 3. 实现与演进

**第一代：人肉调参。** 手工试 `num_warps` / `num_stages`，写死在代码里。

**第二代：穷举 autotuner。** 把候选配置列出来，逐个测，取最好——Triton 的默认路径就是它的成熟形态。它的**边界由规模决定**：几十个 config 可行，几千个程序就是「数小时」。

**第三代：搜索式。** 空间大到穷举不划算时，才有必要引入策略。Ansor 的两级空间（[01](01-search-space.md)）+ GBDT 代价模型（[02](02-cost-models.md)）+ 演化搜索，是这条线的代表。它的搜索在**注释层**做：论文逐字（见 `code/specs.py`）——

> we define a hierarchical search space with **two levels: sketch and annotation**. At the top level, we generate sketches by recursively applying a few derivation rules. At the bottom level, we **randomly annotate** these sketches to get complete programs.

**第四代：把搜索组件化、可替换。** MetaSchedule 把空间生成、搜索、模型、数据库拆成可替换组件（`PostOrderApply` / `EvolutionarySearch` / `XGBModel` / `JSONDatabase`）。

> ⚠ **「MetaSchedule 是 Ansor 的工程化」是推断，不是引文**（TVM 源码无明文）。Ansor 在 TVM 里的位置与 DLight 的并行关系，正文在 [tvm/04-auto-scheduling](../tvm/04-auto-scheduling.md)（回链）。

## 4. 踩坑与失败模式

### 坑 1：以为「autotune」就等于「有个聪明的搜索算法」

- **现象**：读 Triton 的调优日志，期待看到「搜索收敛」的过程，结果发现它是把 `num_warps × num_stages` 全部测一遍。于是怀疑配置写错了。
- **成因**：**默认路径就是穷举**（§2），而且**默认连剪枝都没有**——所以 bench 次数**等于** config 个数。这不是退化，这是设计：在几十个 config 的规模上，穷举的账单本来就付得起，而引入模型就多一个失效点。
- **规避**：看调优日志先问**规模**。config 是个位数到两位数，穷举是对的；要搜的是几千个程序，才轮到搜索算法。本机实测（§5）：12 个 config、无剪枝 → **12 次 bench**；显式开启一个剪枝条件之后 → **8 次**；只给一个 config → **0 次**（根本不走调优路径）。

### 坑 2：★ 把「搜得多就更好」当成单调关系

- **现象**：调优效果不好，第一反应是「预算加一倍」。加了之后没变化，再加一倍，还是没有。
- **成因**：**预算只是乘数，方向由策略决定。** 一个策略如果系统性地测错区域，加预算只是**在错区域里多采几个点**。本机实测里这条非常干净：把预算从 12 翻到 24，**演化搜索**的 regret 从 1.30 降到 1.11（改善 14%），而**模型引导的搜索**从 2.54 到 2.54（**改善 0%**）——它的偏好区域没变，多测的那些点全落在同一个区域里。
- **规避**：**先看 best-so-far 轨迹的斜率，再决定要不要加预算。** 本机实测的两条轨迹（预算 48，同一个种子，每 8 个试次取一点）：
  - 随机搜索：`167765 → 98927 → 98927 → 98927 → 75784 → 75784`——**中段完全平掉了**，后 8 个试次一点没赚；它最后能到 1.65 是靠运气抽中了好的点，不是靠策略。
  - 演化搜索：`167765 → 86659 → 71899 → 64519 → 64519 → 64519`——**单调且有斜率**，第 32 个试次就命中了暴力枚举最优。
  
  两条轨迹的对比给出的判据很实用：**如果 best-so-far 已经平了十几轮，加预算不会救你，换策略才会。**

### 坑 3：把 regret 当成有文献界的量

- **现象**：写「调优的 regret 一般在 1.2 以内」之类的话。
- **成因**：没有这样的文献结论。**「搜到多少个配置之后收益饱和」无定论**（§5 第一条）。能引的只有绑定具体 workload / 硬件 / 实现的经验曲线。
- **规避**：regret 要么报成本仓库在自己的地形上实测的值（并写清是哪个空间、哪个策略、哪个预算），要么不报。**本文的 C3 断言用的是「冻结阈值」**：把首次实跑得到的值写死成常数，之后改坏搜索就会失败——它检验的是**本仓库的实现有没有退化**，不是「业界应该做到多少」。

## 5. 实测数据

**测量条件**：`code/03_search_trace.py` 在本机 `python3 3.13.5` + `numpy 2.4.4` 上跑出的结果，`cd code && make run` 全绿（31 条断言：A 10 / B 14 / C 7）。

### 5.1 ★ 引用值（Ansor 论文）

> **★ 全部标注：引用值，本仓库不复核其测量口径。** 下面每个数字都来自 **Ansor 论文**（*Ansor: Generating High-Performance Tensor Programs for Deep Learning*，14th USENIX OSDI，2020-11-04/06，`https://www.usenix.org/system/files/osdi20-zheng.pdf`），**本仓库没有复现它们的实验**。

**搜索成本（§7.4 Table 3）**。表标题逐字是：「The number of measurements and wall-clock time used for Ansor to **match the performance of AutoTVM** on the Intel CPU (batch size=1)」。

> ⚠ **口径必须写明**：Table 3 是「**追平 AutoTVM** 所需的成本」，**不是「达到最优」所需的成本**。不写明就会把一个上界说成下界——「Ansor 只要 6403 次测量就能达到最优」是**错的读法**。

| 网络 | AutoTVM 测量次数 | Ansor 测量次数 | 次数节省 | AutoTVM 墙钟(秒) | Ansor 墙钟(秒) | 墙钟节省 |
|---|---|---|---|---|---|---|
| ResNet-50 | 21,220 | 6,403 | 3.3× | 39,250 | 4,540 | 8.6× |
| Mobilenet-V2 | 31,272 | 1,892 | 16.5× | 58,468 | 660 | 88.6× |
| 3D-ResNet | 5,158 | 1,927 | 2.7× | 7,594 | 2,296 | 3.3× |
| DCGAN | 3,003 | 298 | 10.1× | 4,914 | 420 | 11.7× |
| BERT | 6,220 | 496 | 12.5× | 12,007 | 266 | 45.1× |

**加速比（§7.3 正文原句）**：

> Compared with search-based AutoTVM, Ansor matches or outperforms it in all cases with **1.0–21.8×** speedup. **Compared with the best alternative**, Ansor improves the execution performance of DNNs on the Intel CPU, ARM CPU, and NVIDIA GPU by **up to 3.8×, 2.6×, and 1.7×**.

> ⚠ **口径必须用正文版，不要用 abstract 版**：abstract 写的是 "relative to the state-of-the-art"，而正文限定更严——「**the best alternative**」且是「**up to**」。三处限定（对比对象是「最好的替代方案」、是上限、分设备给出）少一个都会把结论说过头。

**搜索成本的文字结论（§7.4 正文）**：「Ansor can match the performance of AutoTVM with **an order of magnitude less search time**」；「Typically, it **takes several hours** for Ansor to generate fully-optimized programs for a DNN on a single machine」。

### 5.2 ★★「搜多少配置后收益饱和」：**无定论**

**这个问题本文给不出结论，也不给任何区间或拐点数字。** 研究代理的结论是：找不到任何官方文档或论文给出**通用**的「搜到多少个配置之后收益饱和」；能查到的只有绑定具体 workload / 硬件 / 实现的经验曲线。**拿 GCC flag 调优的收敛结论类推张量编译器是无效的**（机制不同：flag 调优的空间是离散的编译选项，张量编译器的空间是程序结构）。

**可引的只有一条，且必须写明它是单篇论文的实验设置**（§7.4 逐字）：

> We let both AutoTVM and Ansor run auto-tuning until they use to **1000 × n measurement trials** on each DNN, where n is the number of subgraphs in the DNN. **This is typically enough for them to converge.**

读法（三条限定，缺一不可）：

1. 这是**该论文在其基准上**的设置，**不是普适定律**；
2. `n` 是该 DNN 的**子图个数**，所以它是**按图缩放**的一个数，不是一个常数——这也解释了为什么它不能写成「搜 N 个就够」；
3. 「typically enough for **them** to converge」说的是**那两次实验里的收敛**，不是「任何空间搜到这个数都会饱和」。

**本文对 regret 的判据因此只能建在本机实测上**（§5.3 的冻结阈值）。

### 5.3 本机实测值（迷你实现）

> **★ 限定语（本域强制）**：下面每一个数字都是**本仓库的迷你搜索算法与迷你 autotuner** 跑出来的，**不是 Triton**、**不是 Ansor**、**不是硬件**，也**不产生任何时间数字**（量纲是代价，不是秒）。迷你 autotuner 只复现 **v3.6.0 默认路径的分支与计数**；代价地形、两个 DAG 的尺寸、预算、种子、阈值全是本仓库自己定的。这些数字支持的是「搜索策略与预算的权衡可以怎样被观察」这类**机制**结论。

**迷你 autotuner（复现 Triton v3.6.0 默认路径的计数）**：

| 项 | 本机值 |
|---|---|
| config 数（`num_warps ∈ {2,4,8,16}` × `num_stages ∈ {2,3,4}`） | **12** |
| 默认（无剪枝）一次 run 的 bench 次数 | **12**（= config 个数） |
| 其中编译失败的 config 数 | **5** |
| 只给 1 个 config 时的 bench 次数 | **0**（不走调优路径） |
| 显式开启一个剪枝条件后 bench 次数 | **8**（剪掉 4 个） |
| 内存缓存命中（同 key 再调） | **0** 次 bench；换 key → **12** 次 |
| `cache_results=False` / `True` 落盘 | **不存在 / 存在** |
| 带 `pre_hook` 且 `cache_results=True` | **不落盘** |
| 编译失败的 config 的 bench 返回 | **`[inf, inf, inf]`**（3 个分量），且不被选中 |

**搜索策略 × 预算的 regret**（`small` 图，有效空间 4596，暴力枚举最优 **64519**；取 3 个冻结种子的**最差** regret）：

| 预算 | 随机 | 演化 | 模型引导（仅 12） |
|---|---|---|---|
| 6 | 2.65 | 3.83 | — |
| 12 | 2.08 | 2.09 | **2.54** |
| 24 | 2.08 | 1.80 | — |
| 48 | **1.65** | **1.33** | — |

**best-so-far 轨迹**（预算 48，种子 0，第 8/16/24/32/40/48 个试次）：

| 策略 | 8 | 16 | 24 | 32 | 40 | 48 |
|---|---|---|---|---|---|---|
| 随机 | 167765 | 98927 | 98927 | 98927 | 75784 | 75784 |
| 演化 | 167765 | 86659 | 71899 | **64519** | 64519 | 64519 |

**C3（冻结阈值，本仓库选的就是这个写法）**：

- 阈值 `REGRET_MAX = 1.40`，来源是首次实跑：演化搜索在 3 个冻结种子上 budget 48 的 regret 是 **1.00 / 1.33 / 1.11**（最差 **1.33**），阈值取在它之上约 5%。
- 断言：演化搜索的最差 regret **1.33 < 1.40** ✔
- **反例（这条阈值有牙的证据）**：同一个空间、被代价模型带偏的搜索 regret = **2.54 > 1.40** ✘ —— 一个把搜索改坏的改动会**越过**这个阈值。

**A 类（独立参照）**：随机搜索的 best **75784 ≥ 64519**（暴力枚举最优），演化搜索的 best **64519 == 64519**——**没有任何搜索能赢过全枚举**；且两条搜索的记账都自洽（返回的 best_cfg 真的被 bench 过，代价与 best 一致，轨迹长度 = 试次数）。

**§4 坑 2 的那组数（预算 12 → 24，种子 0）**：

| 策略 | budget 12 | budget 24 | 改善 |
|---|---|---|---|
| 演化搜索 | 1.30 | 1.11 | **14%** |
| 模型引导 | 2.54 | 2.54 | **0%** |

**第一版被实跑推翻的两项**：

1. **★ 演化搜索的第一版是个静默退化的随机搜索**。第一版把种群大小写成 `pop = 24` 而预算也是 24——**整个预算被初始种群吃光，一次变异都没发生**。症状是它的结果与随机搜索在三个种子上**逐位相同**（2.65 / 2.08 / 2.08 / 1.65），这正是「一个看起来像模像样的算法其实什么都没做」的典型。修法是把种群夹到 `budget // 3`；修完之后预算 6 那一档的结果立刻变了（演化 **3.83** vs 随机 **2.65**——预算太小的时候演化反而是浪费，因为它把一部分预算用来探索邻域）。**这条只能靠「两个策略的结果一模一样」这个可疑信号发现**，光看单个策略的输出看不出来。
2. **`MiniAutotuner` 第一次跑就抛 `AttributeError`**：落盘时用了 `self.name`，而构造函数没把它存下来。这条本身是低级错误，但它暴露的是一件真事——**「默认关闭」这条路径在写对之前是测不到的**：`cache_results=False` 时 `_dump_cache` 根本不会被调用，所以只有把 `cache_results=True` 那条路真的跑一遍，才会碰到这个 bug。**断言「默认关闭」和断言「打开能工作」是两条不同的断言，缺一条就会漏掉一个死代码路径。**

## 6. 小结与延伸

- **四类策略的前提不同**：穷举与随机不假设任何东西；模型引导假设「模型在新图上仍排得对」；演化假设「有个能产生邻近好解的变异算子」。**加上一个前提就多一个失效点**——这是「先上模型」不是无脑更优的原因。
- **★ Triton 的 autotuner 默认不是搜索，是穷举**：对（剪枝后的）全部 config 逐个 bench 再 `min(timings, key=timings.get)`；**默认无任何剪枝**（`perf_model=None` / `configs_top_k=1.0` / `early_config_prune=None`）；**只有一个 config 时不调优**；**磁盘缓存默认关闭**（`cache_results=False`，带 `pre_hook` 的 config 一律不落盘）；**编译失败返回 `[inf, inf, inf]` 三元组**（不是单个 `inf`）。**全部钉在 v3.6.0**；最新是 v3.8.0，且 v3.8.0 另有「剪枝后单 config 短路」。
- **★「搜得多就更好」不是单调关系**：预算只是乘数，方向由策略决定。本机实测：预算翻倍，演化搜索改善 14%，被模型带偏的搜索改善 **0%**。
- **实用的判据是 best-so-far 轨迹的斜率**，不是剩余预算。轨迹平了十几轮就该换策略，不是加预算。
- **★「搜多少配置后收益饱和」无定论**，本文不给任何区间或拐点。唯一可引的是 Ansor 的单篇实验设置 `1000 × n measurement trials`（**n 是子图个数**，所以它本身也不是一个常数），且必须写明那是该论文在其基准上的设置。
- **引用值全部标了口径**：Table 3 是「**追平 AutoTVM** 所需成本」而非「达到最优」；加速比用**正文口径**（「Compared with **the best alternative**」且 **up to**），不用 abstract 口径。

**延伸**：

- 搜索空间怎么被构造、基数有多大 → [01-search-space](01-search-space.md)
- 代价模型：为什么需要它、它什么时候会把搜索带偏 → [02-cost-models](02-cost-models.md)
- 调度原语与物化位置 → [tvm/02-schedule-primitives](../tvm/02-schedule-primitives.md)
- DLight 的确定性路径与 MetaSchedule 的组件 → [tvm/04-auto-scheduling](../tvm/04-auto-scheduling.md)
- 怎么测量、怎么定位瓶颈（测量方法论在上游） → [50-performance/tuning-practice/01-measuring-sigma-and-tau](../../50-performance/tuning-practice/01-measuring-sigma-and-tau.md)
- tile 大小与共享内存的取舍（上游，本文只代入） → [40-kernels/gemm/02-shared-memory-tiling](../../40-kernels/gemm/02-shared-memory-tiling.md)

## 编译与运行

```bash
cd 60-compilers/auto-tuning/code
make 03_search_trace      # 只跑本篇（约 20 秒：模型引导的搜索每轮都要重训一次模型）
make run                  # 跑全部三篇（约 55 秒）
```

程序：`code/03_search_trace.py`（配套 `code/mini_autotune.py`、外部事实源 `code/specs.py`）。

## 参考

- **Ansor 论文**（*Ansor: Generating High-Performance Tensor Programs for Deep Learning*，14th USENIX OSDI，2020-11-04/06）：`https://www.usenix.org/system/files/osdi20-zheng.pdf`
  - 两级结构逐字（"a hierarchical search space with two levels: sketch and annotation"）：「§3」
  - §7.4 Table 3（标题逐字 "used for Ansor to **match the performance of AutoTVM**"）+ 全文表格：本文 §5.1
  - §7.3 加速比正文原句（"Compared with **the best alternative**… **up to** 3.8×, 2.6×, and 1.7×"）：本文 §5.1
  - §7.4 正文（"an order of magnitude less search time" / "takes several hours"）：本文 §5.1
  - §7.4 实验设置逐字（"**1000 × n measurement trials**… typically enough **for them** to converge"）：本文 §5.2
- **Triton autotuner**，tag **`v3.6.0`**，`python/triton/runtime/autotuner.py`：`https://github.com/triton-lang/triton/blob/v3.6.0/python/triton/runtime/autotuner.py`
  - 默认值（`cache_results=False` / `perf_model=None` / `configs_top_k=1.0` / `early_config_prune=None` / `prune_configs_by=None`）、选择式 `min(timings, key=timings.get)`、单 config 守卫 `if len(self.configs) > 1:`、`do_bench` 的 `quantiles=(0.5, 0.2, 0.8)`、编译失败返回 `[inf, inf, inf]`、落盘名 `{kernel_name}.autotune.json` 与 `pre_hook` 规则、已废弃参数 `warmup` / `rep` / `use_cuda_graph`：都在 §2 逐条对应
  - 版本区间（「磁盘缓存默认关闭」在 v3.4.0–v3.8.0 全部成立；v3.8.0 新增剪枝后单 config 短路）：见 `code/specs.py` 的 `TRITON_VERSION_NOTES`
- 上述每条引文与默认值都在 `code/specs.py` 里逐条对应。**本主题执行时本仓库所在环境没有出网**，引文与默认值照抄自控制者基线（计划 §0.7.E，由两路独立研究对照一手来源核过），**本仓库不复核其内容与测量口径**——这一点在 `specs.py` 的文件头如实标注了。

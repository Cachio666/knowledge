---
title: 融合与调度
slug: inductor-fusion-scheduling
domain: 60-compilers
topic: inductor
level: 3
difficulty: advanced
prerequisites:
  - 60-compilers/inductor/01-graph-capture
tags: [inductor, fusion, scheduler, can-fuse, horizontal-fusion, mix-order-reduction]
related:
  - 60-compilers/inductor/03-codegen
  - 40-kernels/elementwise-fusion/02-fusion-and-traffic
  - 60-compilers/xla
code:
  - code/02_fusion_groups.py
status: stable
updated: 2026-09-16
---

# 融合与调度

> Inductor 决定「融不融」靠的**不是一条规则，而是一串门槛**：垂直方向要求消费者的读与生产者的写**索引逐字匹配**；水平方向的共享数据得分要跨过一个**以字节计**的阈值；全局还有节点数上限、峰值显存门、读缓冲上限。**这一串门槛的存在本身就是答案**——编译器经常选择不融，因为每一条门槛都对应一种「融了更慢」的情形。

## 0. 背景与动机

`40-kernels/elementwise-fusion/02` 已经把融合的收益算清楚了：一条链 `k` 个逐元素算子，**收益上界是链长 `k`**，而且只在中间结果本来要往返 HBM 时才成立。那篇文章留下的是**上游问题**：编译器靠什么判据决定融不融，以及**为什么它经常选择不融**。

对照场景（本仓库程序里真的跑的那一个）：

```python
def model(x, w):
    a = x * 2.0        # 读 x（4×3）
    b = a + w          # 读 a 与 w（w 是 (3,) 的 bias，会广播）
    c = b.relu()
    return c
```

不融：3 个核，中间结果 `a`、`b` 各写一次读一次。全融：1 个核，`x` 与 `w` 读一遍、`c` 写一遍。**本仓库的迷你实现测到 156 字节 → 60 字节**（见 §5；这是本仓库迷你实现的实测值，不是 Inductor 的数字）。

那为什么不无条件全融？因为下面这些情形里融合是**负收益**的：

- 两个消费者没有共享输入，融在一起只是让核变大、寄存器压力上升；
- 中间结果要被别处用，融了也得物化，白融；
- 中间结果是**归约**（读得多、写得少），把它内联进消费者意味着**重算**，读放大比省下的写还多；
- 融合后 kernel 的寄存器溢出，占用率掉下来——这一条有官方实锤（§4）。

本文与主线的关系：**本文是主线的判据来源**。fusion group 怎么划，就是「哪些中间结果不落内存」这个决策本身。

§1 推两阶段的划分与门槛的确切形式。§2 讲这些门槛在真实系统里落在哪几个函数上。§4 集中讲「融了更慢」。

## 1. 原理

### 第一步：融合发生在两个阶段，别把它们混起来

Inductor 的融合不是一个 pass，是**两个阶段各做一半**：

1. **lowering 阶段的 inlining。** FX 图被 lower 成 Inductor 自己的 IR（`ComputedBuffer` 之类）时，一些算子会被**直接内联**进消费者的表达式——这一步决定「这个中间值有没有一个属于自己的缓冲」。**没有缓冲 = 没有物化**。
2. **scheduling 阶段的 fusion。** 剩下的节点被划进 `SchedulerNode` / `FusedSchedulerNode`，由 `Scheduler.can_fuse` 与 `V.choices.can_fuse` 逐对判定能不能合，`score_fusion` 给候选打分，贪心挑分高的先合。

**本文主要讲第二阶段的判据**，因为它才是「什么时候选择不融」的所在。第一阶段的 inlining 在 [03-codegen](03-codegen.md) 里作为 codegen 的输入出现。

### 第二步：垂直融合——索引必须逐字匹配

垂直 = 生产者 → 消费者。表面上「消费者读了生产者的输出」就够了，实际上要求更强：

> 消费者的每一次读，都要与生产者的写**是同一个内存依赖**（同一缓冲 + 同一索引）。

索引对不上时源码报的理由串是 **`memory deps did not match`**。源码注释给了两个逐字的反例：

```text
MemoryDep("foo", x) != MemoryDep("foo", x + 1)
MemoryDep("foo", x) != StarDep("foo")
```

第一条是「错位读」，第二条是「整块依赖」。为什么这么严？因为融合之后生产者与消费者**共享同一套循环**。生产者在循环 `(i,j)` 上写 `foo[i,j]`，消费者在循环 `(i,j)` 上读 `foo[i,j]`——对齐，融。消费者如果在 `(i,j)` 上读 `foo[i+1,j]`，那它需要的是生产者**下一个迭代**才写的值，两个循环嵌套无法用一次遍历同时满足。

**这条判据解释了本仓库程序里的那个现象**：一个点式链 `x*2 → +w → relu` 全部融合（每一步的读索引 `@0,@1` 与上一步的写索引逐字相等）；而中间插一个**转置**之后，转置节点读上游的索引变成 `@1,@0`，与写的 `@0,@1` 不相等 → **拒绝**，报的就是 `memory deps did not match`。

### 第三步：水平融合——共享数据得分要跨过一个**以字节计**的阈值

水平 = 消费者 ↔ 消费者（两个兄弟节点，互不依赖）。判据是**共享数据的量**：

```text
shared_data_score = Σ （两个节点的公共依赖的字节数）
```

- 得分为 **0** 时：若 `aggressive_fusion = False`（默认），直接拒，理由串是 **`no shared data`**；若两个节点里有任何一个缓冲区名相同（读的是同一块但索引不同），另有理由串 **`no shared data due to indexing mismatch`**。
- 得分 **> 0 但低于** `score_fusion_memory_threshold`（默认 **10**）：拒，理由串就是配置名 `score_fusion_memory_threshold`。

**这里最反直觉的一点是单位**：`dep_size_hint` 在 `count_bytes=True`（默认）下取的是 `dep.numbytes_hint()`，所以这个 **10 是 10 字节**。一个 float32 的 2 元素张量共享出去只有 8 字节——不够。

还有一个**条件写反就错**的地方：

```python
if shared_data_score == 0 and (
    not config.aggressive_fusion or node1.is_reduction() or node2.is_reduction()
):
    ...  # 拒
```

把它读准：`aggressive_fusion=True` **并不能**放开带归约的那一对——**只要有一侧是归约，score 为 0 仍然拒**。

### 第四步：reduction ↔ reduction 不是通用能力

两个归约是**两个**归约——读者很容易以为「都在读同一张输入，当然能融」。**默认路径上不能融。** 唯一成体系的特例叫 **mix-order reduction**：融合**同一输入张量不同维度**上的两个规约。它的条件是一串，缺一不可：

| 条件 | 具体形式 |
|---|---|
| **兄弟关系** | 不能有祖先关系（不是生产者-消费者）——源码注释逐字写着 "sibling reductions, not producer-consumer" |
| **循环序镜像** | 两个节点的 `(numel, rnumel)` 必须**互为镜像**（`tuple(g1) == tuple(reversed(g2))`） |
| **必须有公共读** | 公共读为空直接拒 |
| **规模下限** | `nrow * ncol >= 5 * 2**20`（即 5,242,880 个元素） |
| **行数下限** | `nrow >= 4096` |
| **列数上限** | `ncol <= 1024 * 16` |
| **读缓冲上限** | 融合对的唯一读缓冲数 ≤ `mix_order_reduction_max_reads`（默认 **10**） |
| 其他 | `reduction_hint ∈ {INNER, DEFAULT}`；不是 split reduction；max/min 这类归约不支持 |

（`nrow = numel`、`ncol = rnumel` 取自那一对里被称为「连续节点」的一侧；真实实现里由 `is_contiguous_node` 判定，本仓库的迷你实现把**归约循环在内**的那一侧当作连续节点，这是简化，写在 `code/minifuse.py` 的注释里。）

**规模下限的注释就是这条判据的动机**（源码逐字）：数据小到能被 L2 装下时，融合带来的收益抵不过开销。也就是说：**小张量上的「当然能融」在真实系统里是被一条数值门槛拦住的。**

### 第五步：全局门槛

| 配置 | 默认值 | 作用 |
|---|---|---|
| `max_fusion_size` | **64** | 一次融合最多几个节点，理由串 `exceeds max fusion` |
| `max_fusion_unique_io_buffers` | `None`（检查关闭） | 融合对的唯一读/写缓冲上限，理由串 `fusion_prevent_too_many_reads_and_writes` |
| 峰值显存门 | 无开关 | `can_fusion_increase_peak_memory` 返回真则拒，理由串 `Fusion will increase peak memory` |
| 距离门 | 无开关 | 两个节点在图里离得太远则拒，理由串 `Nodes are too far away. Fusing them may increase peak memory.` |

**这一节的总述是 5 步；§2、§3 只讲它们落在哪些函数上、以及这些门槛怎么长出来的，不再新增判据。**

## 2. 执行映射

**组件一：`V.choices.can_fuse`——统一的门槛链。** 这是上面那些判据的**逐行落点**（torch 2.13.0 的 `torch/_inductor/choices.py:568-636`），顺序就是源码顺序：`no shared data` → `exceeds max fusion` → 峰值显存 → 读缓冲上限。它的 docstring 自己声明这些是 **heuristics**、`not needed for correctness`——也就是**它们全都只影响性能**。这个定位很重要：一条门槛去掉，程序照样正确，只是可能更慢；反过来，加一条门槛也不会让程序变错。

**组件二：`Scheduler.can_fuse_vertical` / `Scheduler.can_fuse_horizontal`。** 两个钩子分别管两个方向，源码里 `can_fuse_vertical` 的默认实现直接 `return True`——把**合法性**（索引匹配、依赖关系）留给 `Scheduler._can_fuse` 这条主路径，把**启发式**（分数、峰值显存）留给 `V.choices`。**这条分工是理解整节的钥匙**：合法性错 = 算错；启发式错 = 变慢。

**组件三：`Scheduler.score_fusion_memory`。** 它算的就是上面那个以字节计的分数：`count_bytes=True`（默认）时取 `dep.numbytes_hint()`，否则取 `dep.numel_hint()`（`torch/_inductor/graph.py:690-710`）。**量级**：默认阈值 **10 字节**——在 float32 上相当于 2.5 个元素。所以「有没有共享」与「共享够不够多」是两道不同的门。

**组件四：`MixOrderReduction`。** 归约↔归约的唯一通路，判据全在 `torch/_inductor/scheduler.py:204-450`。**量级**：`5 * 2**20 = 5,242,880` 是规模下限，`4096` 是行数下限，`1024 * 16 = 16384` 是列数上限。三个数字都写死在 `can_fuse` 里，不是配置项。

**组件五：`FusedMixOrderReductions` 的读缓冲上限。** 混序归约融合之后，源码还会在**调度期**统计它读了多少个唯一缓冲，超过 `mix_order_reduction_max_reads`（默认 **10**）就拒绝——`torch/_inductor/scheduler.py:2922`。这条上限是**事后补上的**，来历见 §4。

**量级（本机实测）**：本仓库迷你调度器在点式链地形上做了 2 次判定、全部通过；在转置地形上做了 4 次判定，其中 3 次拒绝（1 次索引不匹配、1 次无共享、1 次没有可垂直融合的生产者）。水平融合的字节门槛在 **8 字节（拒）** 与 **16 字节（融）** 两侧各验了一次。

## 3. 实现与演进

**第一代：只有「有共享就读得少」这一条直觉。** 早期判据接近「两个节点共享输入就融」。它的问题在 §4 第 1 条：共享让核变大，寄存器压力上去之后收益被吃掉。

**第二代：阈值化的共享数据得分。** 把「有没有共享」升级成「共享了多少字节」，并给了一个可调的阈值（`score_fusion_memory_threshold = 10`）。这一代同时引入了 `aggressive_fusion` 作为逃生门——但源码的条件写明了这个逃生门**对带归约的一侧无效**。

**第三代：按融合类型分别加门槛。** 归约↔归约单独开了 `MixOrderReduction` 这条路，并带上一组**数值门槛**（规模下限、行数下限、列数上限、读缓冲上限）。**这一代的特征是：判据开始依赖形状的绝对大小**，而不再只是「有没有依赖」。

**版本边界（引用时必须带版本）**：

| 项 | 版本边界 |
|---|---|
| `score_fusion_memory_threshold = 10`、`max_fusion_size = 64`、`aggressive_fusion = False` | torch **2.6.0 / 2.7.1 / 2.13.0** 三个版本同名同值 |
| `max_fusion_unique_io_buffers` | **2.13.0 才有**（`torch/_inductor/config.py:975`）；2.6.0 与 2.7.1 的 `config.py` 里没有这个名字 |
| `mix_order_reduction_max_reads = 10` | 2.13.0（`config.py:2137`）；2.6.0 与 2.7.1 的 `torch/_inductor/` 子树里 0 命中 |
| epilogue 的 spill / occupancy 判据（`fused_n_spills >= 8`、`MIN_ACCEPTED_OCCUPANCY = 4`、`REGRESSED_OCCUPANCY_RATIO = 0.5`） | 2.13.0（`scheduler.py:3713,3745,3746`）；2.6.0 与 2.7.1 的 `torch/_inductor/` 子树里 0 命中 |
| `max_horizontal_fusion_size = 16` | 三版都有，但它在 **`config.cpp`**（**CPU 的 C++ 后端**）里，不是 GPU/Triton 的开关 |

**一处必须标明是推论的地方**：`score_fusion_memory_threshold` 的默认值 10 与它「以字节计」的单位，合起来意味着**任何小于 10 字节的共享都不算共享**（float32 下 2.5 个元素）。这是一个从源码读出的**数值边界**，不是官方文档里写的设计意图——源码里没有解释为什么是 10。

## 4. 踩坑与失败模式

**失败模式 1（本篇核心）：融合越界反而更慢，而且官方自己踩过。**

- **现象**：把该融的都融上之后，每步的耗时**反而涨了**。官方记录在案的实锤是 PR #179494：过度融合的 mix-order reduction 在 **H100 上 +6.3 ms/step**，profiling 显示 **11 个读**因为 **register spill** 回退。（**引用值**：这是外部来源的硬件数字，**本仓库不复核其测量口径**。）
- **成因**：融合把多个消费者的循环合成一个，每个消费者都要在寄存器里留住自己的中间值；寄存器分配不上就 spill 到本地内存——而 spill 的访问代价可能比省下的那次 HBM 往返更贵。**前文论证的是「融合省字节」，这里是「字节省了却输在寄存器压力上」，两件事不矛盾。**
- **规避**：给融合加**资源上限**，而不是只看字节。PR #179494 的修法就是在**调度期**统计唯一读缓冲数并设上限——这就是 `mix_order_reduction_max_reads = 10` 的来历。另一处同类判据是 epilogue 融进模板时的 occupancy/spill 检查（`fused_n_spills >= 8` 直接拒，另有 `MIN_ACCEPTED_OCCUPANCY = 4`、`REGRESSED_OCCUPANCY_RATIO = 0.5`）。

  **本仓库能把这条机制跑出来**：同一份代价模型，在两个地形上给出**相反**的结论。中间结果是**归约**（读 (4,32) 只写 (4,)，读写比 32）时，把它内联进两个消费者要 **2616 字节**，物化只要 **2152 字节** → 判定「不该融」；中间结果是**点式**（读写比 2）时反过来，内联 **120 字节** vs 物化 **136 字节** → 判定「该融」。**融合不是越多越好，好坏由「重算要重读多少」决定。**

**失败模式 2：把阈值当成「有没有共享」。**

- **现象**：两个消费者明明共享一个输入，却没有融在一起；输出的 kernel 数比预期多。
- **成因**：判据不是「有没有共享」，而是「共享的**字节数**是否 ≥ `score_fusion_memory_threshold`（默认 10）」。本仓库程序里，共享 8 字节（2 个 float32）被拒，共享 16 字节被放行——**中间那条线就是 10**。共享存在但很小的时候，融合省下的访存还抵不上一趟核启动的开销。
- **规避**：先确认共享的**字节数**，再确认阈值是否被改过；不要只看依赖图。

**失败模式 3：以为两个归约「当然能融」。**

- **现象**：对一个张量的不同维度各做一次规约（`sum(axis=1)` 与 `sum(axis=0)`），期望它们合成一个核，结果是两个。
- **成因**：**reduction ↔ reduction 不是通用能力**。通用路径直接拒；唯一的特例 `MixOrderReduction` 要求兄弟关系、循环序镜像、有公共读，还要同时满足 `nrow*ncol >= 5*2**20`、`nrow >= 4096`、`ncol <= 1024*16`。**小张量上的镜像归约对会被规模下限拦下**——这是设计意图（数据小到进得了 L2 时，融合不值得）。
- **规避**：知道这是一条**特例**，不是通用路径。`aggressive_fusion=True` 也救不了它（§1 第三步的条件写明了）。

**失败模式 4：只看「融合了几个」，不看「融合之后寄存器怎么样」。**

- **现象**：调大 `max_fusion_size`、打开 `aggressive_fusion`，融合组变大了，性能没变好甚至更差。
- **成因**：这些开关放开的是**启发式**的门，而启发式按源码自己的话说 `should not be needed for correctness`——它们的存在就是为了拦住负收益的融合。把它们全放开等于把「什么时候不融」的知识扔掉。
- **规避**：调参之前先确认瓶颈在访存而不是在占用率；`max_horizontal_fusion_size = 16` 这类开关**在 CPU 的 C++ 后端配置段（`config.cpp`）里**，在 GPU 上调它不会有任何效果——这是 §3 表格里那条容易被误读的版本注记。

## 5. 实测数据

**测量条件**：`code/02_fusion_groups.py` 在本机 `python3 3.13.5` + `numpy 2.4.4` 上跑出的结果，`cd code && make run` 全绿（本篇 25 条断言：A 10 / B 8 / C 7）。

> **★ 限定语（本域强制）**：下面的数字全部来自**本仓库几百行的迷你调度器与代价模型**，**不是 Inductor 的调度器**。判据与默认值逐条抄自固定版本的源码（见 `code/specs.py`），但**实现形态是重写的**：真实的调度器工作在 `BaseSchedulerNode` 与 sympy 索引表达式上，本仓库工作在字符串索引元组上。这些数字支持的是「哪条判据在什么条件下拦住了什么」这类**机制**结论，**不支持任何关于 Inductor 实际融合结果的结论**。

**判据演示表**（全部是本机实测值）：

| 地形 | 判定 | 拒绝理由串（源码里实际存在的字符串） |
|---|---|---|
| 点式链 `x*2 → +w → relu` | 全部融合（1 组） | — |
| 中间插一个转置 | 拒绝 | `memory deps did not match（('@1','@0') != ('@0','@1')）` |
| 水平：共享 8 字节 | 拒绝 | `score_fusion_memory_threshold（score=8 < 10）` |
| 水平：共享 16 字节 | 融合 | — |
| mix-order：(32,64) 的镜像对 | 拒绝 | `nrow*ncol < 5*2**20（2048）` |
| mix-order：(4096,1280) 的镜像对 | **放行** | — |
| mix-order：nrow=2048 | 拒绝 | `nrow < 4096（2048）` |
| mix-order：唯一读缓冲 10 个 | 放行（边界内） | — |
| mix-order：唯一读缓冲 11 个 | 拒绝 | `唯一读缓冲数 11 > mix_order_reduction_max_reads=10` |
| mix-order：改成生产者-消费者 | 拒绝 | `祖先关系（mix-order 要求兄弟关系）` |
| 两个循环序不镜像的归约 | 拒绝 | `循环序不是镜像 ((8,8) vs (8,8))` |

**「融合越界反而更慢」的两个地形**（本仓库模型，字节数）：

| 中间结果 | 读写比 | 计划「物化」 | 计划「内联」 | 结论 |
|---|---|---|---|---|
| 归约 `(x*w).sum(axis=1)`（读 4×32，写 4） | 32 | **2152 字节**（重算 0 个节点） | 2616 字节（重算 1 个） | **不该融** |
| 点式 `x*w`（读 4，写 4；两个输入） | 2 | 136 字节 | **120 字节**（重算 1 个） | **该融** |

**A 类的独立参照路径**：四个计划（两个地形 × 物化/内联）都走「生成 Python 源码 → `exec` → 与纯 numpy 参考实现对拍」，最大绝对误差全部 **0.000e+00**；另有三个地形（垂直 / 转置 / 水平）走「图解释器 vs 纯 numpy 参考实现」，最大绝对误差同样全部 **0.000e+00**。生成物的核数分别是 4 / 3 / 3 / 2。

**C 类的失败行为**（每条都真的发生）：

| 构造 | 结果 |
|---|---|
| 转置后的读索引与写索引 | 拒绝，理由串 `memory deps did not match` |
| 共享 8 字节 | 拒绝，理由串 `score_fusion_memory_threshold` |
| 循环序不镜像的归约对 | 拒绝 |
| 可证伪性反证（S11）：写死「永远内联」的常数策略 | 在两个地形上都成立吗？**False**（在地形 R 上失败） |
| 可证伪性反证（S11）：写死「永远物化」的常数策略 | 在两个地形上都成立吗？**False**（在地形 P 上失败） |

**关于上面那两条反证的设计**（计划 §0.4 第 5 条要求「自造地形」类断言必须能因为「把被检验的那一方换成正确的」而失败）：地形常量是**写死的**、先于结果冻结（`terrain_reduction` 用 (4,32)、`terrain_pointwise` 用 (4,) 的向量），不看到结果再调；断言不是「模型在地形 X 上说 Y」——那只是在检验地形——而是「**模型的判定与实测更便宜的那个计划一致，且在两个地形上都一致**」。**任何常数策略（永远融 / 永远不融）都会在其中一个地形上失败**，所以这条断言检验的是机制而不是地形。

**第一版被实跑推翻的三项**：

1. **「N 个不同 shape → 重编译 N 次」的直觉在融合这里也有同胞。** 我第一版把 mix-order 的规模地形写成 `(32, 32)`（方阵）。结果它**不是**被规模下限拒的，而是被 `g1 == g2` 这条更早的判据拒的——源码在比较镜像之前先排除了 `g1 == g2`。改成 `(32, 64)` 之后才真的验到规模下限。
2. **「唯一读缓冲 11 个」这条一开始根本没被构造出来。** 第一版让两个归约读同一批额外缓冲，于是唯一读缓冲数是 2 而不是 11，断言失败。修法是把额外缓冲**逐个分开**给两个节点。
3. **A 类的参照实现一开始返回的是单个数组而不是列表**，`zip(g.outputs, want)` 于是把数组的**行**当成了输出，误差报成 `inf`。这个 bug 只有在真的跑起来之后才看得见。

## 6. 小结与延伸

- **两个阶段**：lowering 阶段的 inlining（决定有没有缓冲）+ scheduling 阶段的 fusion（决定缓冲要不要落内存）。
- **垂直融合要索引逐字匹配**，否则理由串是 `memory deps did not match`。
- **水平融合的门槛以字节计**：`shared_data_score >= score_fusion_memory_threshold`（默认 **10 字节**）；`aggressive_fusion=True` **也救不了带归约的一侧**。
- **reduction ↔ reduction 不是通用能力**：只有 mix-order 特例，且要同时满足兄弟关系、循环序镜像、有公共读、`nrow*ncol >= 5*2**20`、`nrow >= 4096`、`ncol <= 1024*16`、读缓冲 ≤ 10。
- **全局门槛**：`max_fusion_size = 64`、峰值显存门、距离门、读缓冲上限。
- **融合越界反而更慢有官方实锤**（PR #179494，H100 上 +6.3 ms/step，11 个读因 register spill 回退），修法是在调度期给读缓冲数设上限。

**延伸**：

- 划好的融合组怎么变成代码、走到哪条后端上 → [03-codegen](03-codegen.md)
- 同一张图要编几次、什么时候不再编 → [04-dynamic-shapes](04-dynamic-shapes.md)
- 融合的收益上界（链长）与字节推导 → [40-kernels/elementwise-fusion/02](../../40-kernels/elementwise-fusion/02-fusion-and-traffic.md)
- 另一套融合判据（预算 + 合法性 + 代价模型三层） → [60-compilers/xla](../xla/)（前向声明）

## 编译与运行

```bash
cd 60-compilers/inductor/code
make 02_fusion_groups      # 只跑本篇
make run                   # 跑全部四篇
```

程序：`code/02_fusion_groups.py`（配套 `code/minifuse.py` 迷你调度器与代价模型，外部事实源 `code/specs.py`）。

## 参考

- 门槛链的逐行落点：`torch/_inductor/choices.py:568-636`（torch 2.13.0）
- 垂直融合的索引判据与 `memory deps did not match`：`torch/_inductor/scheduler.py:7734-7766, 7813-7866`
- 分数以字节计：`torch/_inductor/scheduler.py:8058-8193` + `torch/_inductor/graph.py:690-710`
- `MixOrderReduction` 的全部条件：`torch/_inductor/scheduler.py:204-450`；读缓冲上限在 `scheduler.py:2922`
- epilogue 的 spill / occupancy 判据：`torch/_inductor/scheduler.py:3713,3745,3746`
- 默认值（含版本边界）：`code/specs.py` 的 `INDUCTOR_DEFAULTS` / `MAX_FUSION_UNIQUE_IO_BUFFERS` / `MAX_HORIZONTAL_FUSION_SIZE` / `EPILOGUE_SPILL`
- **引用值**：pytorch/pytorch PR #179494（H100 上 +6.3 ms/step，11 个读因 register spill 回退）——**本仓库不复核其测量口径**
- 以上每条都在 `code/specs.py` 里逐条对应，取用日期 2026-09-16

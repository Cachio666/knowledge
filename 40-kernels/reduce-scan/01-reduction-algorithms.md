---
title: 归约的并行算法：树形、segmented 与 argmax
slug: reduction-algorithms
domain: 40-kernels
topic: reduce-scan
level: 2
difficulty: intermediate
prerequisites:
  - 20-gpu-architecture/execution-model/01-sm-internals
  - 20-gpu-architecture/execution-model/02-simt-and-divergence
tags: [reduction, tree-reduction, work-depth, segmented-reduction, argmax, nan, tie-break, brent]
related:
  - 20-gpu-architecture/memory-system/02-coalescing-and-cache-line
  - 20-gpu-architecture/execution-model/03-occupancy-and-latency-hiding
  - 40-kernels/reduce-scan/02-warp-shuffle-and-hierarchy
  - 40-kernels/attention/02-flash-attention
  - 40-kernels/softmax-normalization/02-online-softmax
  - 40-kernels/moe-kernels/03-permute-and-unpermute
code:
  - code/01_reduce_basics.cu
status: stable
updated: 2026-09-16
---

# 归约的并行算法：树形、segmented 与 argmax

> 把 2^20 个数加起来，顺序做法要 1048575 次加法、依赖链长 1048575；树形做法**加法次数一模一样**、依赖链只有 20。并行归约没有减少任何工作量，它减少的只有**依赖链**——这是理解归约的全部起点。而真实算子要的不是「一个和」，是**每行一个和**（变长、边界不对齐 warp），或者**一个带下标的极值**（NaN 与并列值的口径必须写死）。

## 0. 背景与动机

先看对照场景：**把 2^20 = 1048576 个 fp32 归约成一个标量**。

```text
work（加法次数）  = n − 1 = 1048575
```

一个线程顺序做：`acc = x[0]; for i in 1..n-1: acc += x[i];`。每次加法依赖上一次的结果，所以**依赖链长度（depth）= 1048575**。这条链跑不出并行度——把它放在 132 个 SM 的机器上，用的是 1 个线程。

换树形归约（两两配对折半），**加法次数仍然是 1048575**：

| 形态 | work | depth |
|---|---|---|
| 顺序（1 条线程） | 1048575 | **1048575** |
| 树形（两两配对） | 1048575 | **20** |

（推导值，`make verify` §B4 逐行打印。）**work 一项一个字节都没少。** 变的只有 depth：从 1048575 降到 20，比值 **52429×**。

这决定了后面所有内容的方向：**并行归约能做的唯一一件事，是把顺序依赖拆短。** 所以任何「归约的优化」都要先问一句：我减少的是 work，还是 depth？

用 Brent 模型 `T = work/p + depth` 可以把这句话变成数字（推导值，`make verify` §B4）：

| p（并行度） | 树形 T | 相对顺序版的加速比 |
|---|---|---|
| 1 | 1048595.0 | 1.0× |
| 32 | 32788.0 | 32.0× |
| 1024 | 1044.0 | 1004.4× |
| 1048576 | 21.0 | **49932.1×** |

注意最后一行：**p 已经等于元素个数了，加速比仍然没到 work/depth = 52429× 的上界**。`p → ∞` 时 `T → depth`，所以加速比的上界是 `work/depth`，**不是「有多少并行度就快多少」**。把 p 从 1024 加到 1048576（×1024）只把加速比从 1004× 抬到 49932×（×49.7）——因为这一步之后 depth 项（20）开始和 work 项（1）同量级了。

### 真实算子要的不是「一个和」

上面那套账只是一个标量的归约。生产代码里要归约的东西几乎都不是这个形状，本文第二节的两种形态**不是同一条公式换个参数**：

- **segmented（grouped）归约**：softmax 的每一行、MoE 的每个专家，各要一个和。行的长度是**变的**，而且**边界通常不对齐 warp**。这里没有「一个 n」，有 `nrows` 个不同的 `n_r`。
- **argmax / argmin 归约**：要的不只是最大值，还有**它在哪**。值和索引必须**配对**归约，而 NaN 与并列值（tie）的处理在不同实现里**语义不同**——不写清口径，读者无法判断实现对不对。

本文只讲这两件事：§1 推导树形归约的 work/depth 并给出上面两种变体，§2 说明块内归约在硬件上发生了什么、`__syncthreads()` 为什么是必需的，§3 给出实现演进，§4 收五条「读完全文仍会踩」的坑。

## 1. 原理

### 1.1 树形归约：work 不变，depth 变

**第一步：把顺序归约的依赖链画出来。**

```text
acc0 = x[0]
acc1 = acc0 + x[1]     ← 依赖 acc0
acc2 = acc1 + x[2]     ← 依赖 acc1
...
```

第 `i` 次加法依赖第 `i−1` 次。这是**一条链**，depth = n−1。work = n−1。

**第二步：换成两两配对，前一半的加法互相独立。**

```text
第 1 层： y[i] = x[2i] + x[2i+1]        i = 0 .. n/2−1     （n/2 次加法，互相独立）
第 2 层： z[i] = y[2i] + y[2i+1]        i = 0 .. n/4−1     （n/4 次加法，互相独立）
...
第 k 层： 只剩 1 个数                                      （1 次加法）
```

**第三步：数两笔账。**

```text
work  = n/2 + n/4 + … + 1 = n − 1        ← 与顺序形态**完全相同**
depth = 层数 = ⌈log2 n⌉                  ← n = 2^20 时是 20
```

（`make verify` §A1 用整数输入断言了树形归约的加法次数恰好是 n−1、depth 恰好是 ⌈log2 n⌉；§B4 打印 n = 2^20 的对照表。）

**第四步：把「每线程一个元素」放松成「每线程 k 个元素」。**

GPU 上一个 block 只有 T 条线程，要归约 n 个元素（T ≪ n）。做法是两段拼接：

```text
第 1 段（每线程自己吃 k = n/T 个）：work = k−1，depth = k−1
第 2 段（T 条线程走树形）        ：work = T−1，depth = log2(T)

合计 work  = (k−1)·T + (T−1) = k·T − 1 = n − 1     ← 还是 n−1
合计 depth = (k−1) + log2(T)
```

（`make verify` §A1：p = 32/128/1024 三档的总加法次数都等于 4095 = n−1。）n = 2^20、p = 1024 时 depth = 1023 + 10 = **1033**；p = 2^20 时 depth = 0 + 20 = **20**。

**第五步：结论。**

> **work 与并行度无关，恒为 n−1。并行归约减少的只有 depth。**
> 所以「顺序依赖是并行归约的唯一敌人」不是一个比喻：work 那一项在并行前后**一个字都没变**，唯一变化的是那条链有多长。

### 1.2 变体一：segmented / grouped 归约

现在把「一个 n」换成 `nrows` 个长度不等的段。这个形态在两类算子里反复出现：

- **per-row softmax**：`S×S` 的注意力矩阵按**行**归约求 `rowmax` 与行和（本文只讲归约侧，softmax 的数值稳定性正文在 [40-kernels/softmax-normalization/01](../softmax-normalization/01-softmax-stability.md)，此处回链）；
- **MoE 的 `tokens_per_expert`**：每个专家收到的 token 数不同，要对每个专家的 token 段做归约（正文在 [40-kernels/moe-kernels/01](../moe-kernels/01-moe-compute-pattern.md)，此处回链）。

**第一步：把段表示成「起点 + 长度」两个数组。**

```text
row_start[r], row_len[r]      r = 0 .. nrows−1
```

这样段是**变长**的，而且相邻段的边界**与 warp 边界无关**。这正是本节要处理的核心困难。

**第二步：先看段长与 warp 尺寸的关系。**

用「一个 warp 负责一行」的形态（warp 内 shuffle 折半，不需要块级屏障——机制见 [02 篇](02-warp-shuffle-and-hierarchy.md)）：

| L | 每线程吃的元素数 | 每 warp 一行的 lane 利用率 |
|---|---|---|
| 8 | 1 | **25.0%**（空转 24/32） |
| 20 | 1 | **62.5%**（空转 12/32） |
| 1024 | 32 | 100%（但每线程串行 31 次加法） |
| 4096 | 128 | 100%（串行部分 127 次） |

（推导值，`make verify` §B5。）`L < 32` 时**浪费的比例是 `1 − L/32`**：L = 8 时四分之三的 lane 在空转。

**第三步：段不等长时的代价由谁决定。**

这是 segmented 归约与普通归约最不一样的地方。假设 `nrows = 4096`、行长在 `[12, 200]` 之间均匀分布（平均 106），用「每 warp 一行」：

```text
迭代数由**最长行**决定   → 每 lane 要吃 ceil(200/32) = 7 个元素
每行占用的 lane 槽位      = 7 × 32 = 224
有效利用率                = 106 / 224 = 47.3%
```

（推导值，`make verify` §B5。）**空转不是因为某一行短，是因为同一个 kernel 要覆盖所有行，迭代数只能按最长的行取。** 行长差得越远，空转越多——`Lmin/Lmax = 12/200` 时过半的槽位是空的。

**第四步：段边界不对齐 warp 时会发生什么。**

这是本节的**唯一**结论，也是 §4.3 那条失败模式的来源。若直接用「线程编号 ÷ 32」当段号：

```text
段长 lens = [20, 12, 50, 6]（总长 88）
按段归约的正确结果 = [332, 176, 763, 104]
按 warp 边界切（每 32 个连续元素当一段）= [508, 493]
```

（推导值，`make verify` §C6。）第一个 warp 拿到的是第 0 段的 20 个元素**加上第 1 段的前 12 个元素**——`508 = 332 + 176`，两行被串成一行。

> **正确做法：先定段，再归约。** 每线程先算出「我属于哪一段」（段起点数组做查表或对 `row_start` 二分），再在段内归约。**绝不能拿 `threadIdx.x / 32` 当段号。**

**第五步：「补齐」这一步是与算子绑定的。**

段内归约通常要把长度补到 warp 的整数倍。补什么值取决于算子（推导值，`make verify` §A4）：

| 算子 | 幺元 |
|---|---|
| sum | `0.0` |
| max | `−∞`（fp32 用 `-CUDART_INF_F`，或一个足够小的有限值） |
| min | `+∞` |

同一段变长数据、同一个 warp-per-row 形态，换成 max 归约后（段长 `[17, 64, 5, 128, 33]`）：

```text
补齐用 −∞ ： [-2, -2, -6, -2, -4]   ← 正确（= 各段真值）
补齐照抄 0： [ 0, -2,  0, -2,  0]   ← 短段的真实最大值被 0 顶替
```

（推导值，`make verify` §A4。）**这不是精度问题，是「结果变成另一个值」。** 抄上一个 kernel 的补齐值，是这类 bug 最常见的来源。

### 1.3 变体二：argmax / argmin 归约

**第一步：为什么它不能拆成「先 max、再找索引」。**

看起来两步很自然：先归约出最大值，再回头在数组里搜一遍哪个位置等于它。但这个绕法**隐式选定了一个 tie-break 规则**——它总是给出**最小索引**：

```text
输入 [1.0, nan, 5.0, 3.0, 5.0, 2.0, nan, 4.0]
「只归约值、再回头找索引」→ 索引 2（最小索引）
用户声明的 tie='last'      → 索引 4
```

（推导值，`make verify` §A5。）所谓「值对了」只是巧合：**索引不是你归约出来的，是你事后猜出来的。**

正确做法是把 `(value, index)` 当成**一个整体**一起归约：一起 shuffle、用同一个比较函数。比较函数先比值、值相等时比索引。

**第二步：NaN 的口径必须显式选一个。**

IEEE 754 定义了两种 max，它们给出的结果不同：

| 口径 | 标准 | CUDA / numpy | `f(1.0, NaN)` |
|---|---|---|---|
| **忽略 NaN** | IEEE 754-2008 的 `maxNum` | `fmaxf` / `np.fmax` | `1.0` |
| **传播 NaN** | IEEE 754-2019 的 `maximum` | `np.maximum` | `NaN` |

（规格值 + 推导值，来源见 `code/specs.py` 的 `FMAXF_IS_IEEE_MAXNUM`；`make verify` §A5 断言了这两个函数在同一输入上给出不同的答案。）

这条差异会一路传导到索引上：

```text
输入 [3.0, 5.0, nan, 5.0, 1.0, nan]
忽略 NaN   + 最小索引 → 值 5.0，索引 1
传播 NaN   + 最小索引 → 值 nan，索引 2
```

（推导值，`make verify` §A5。）**两条口径都「自洽」，所以文档不写口径，读者就无法判断实现对不对。**

> **本文的口径（下文与配套程序都用这一条）：值按 `fmaxf` 语义——忽略 NaN；并列时取最小索引。** 换口径必须写明。

**第三步：并列值（tie）也必须写死。**

```text
输入 [1.0, 7.0, 7.0, 7.0, 2.0]
tie='first'（最小索引）→ 索引 1
tie='last' （最大索引）→ 索引 3
```

（推导值，`make verify` §A5。）NaN 口径 2 种 × tie 口径 2 种 = **4 种组合**，两两之间都可能给出不同的索引。本文只声明其中一种。

**第四步：NaN 口径在归一化里的后果。**

同一个归约用在 per-row softmax 的分母上时，口径的选择会决定**整行**的输出（推导值，`make verify` §A5）：

```text
分母 = 5.0（忽略 NaN）    → 该行输出正常
分母 = nan（传播 NaN）    → 该行输出**全部变成 NaN**
```

所以 per-row 归约的 NaN 口径不是一个「风格问题」：它决定了这一行是不是整行报废。这也解释了为什么 `np.fmax.reduce` 与 `np.maximum.reduce` 在数值代码里**不可互换**。

## 2. 执行映射

### 2.1 块内归约：`__syncthreads()` 是必需的，但它等的是块

一个 block 内做树形归约时，`n/T` 那一段是每线程私有的（无同步），但**折半那一段跨 warp**——线程 `i` 要读线程 `i+off` 写进共享内存的值。写者与读者在**不同 warp**，两条指令流之间没有任何顺序保证，所以必须用**块级**屏障 `__syncthreads()` 划界。

| 部件 | 在归约里做什么 | 量级 |
|---|---|---|
| warp scheduler | 发射折半那几步的 `SHFL` / `FADD` | 每 SM 4 个，各每周期 1 条 |
| 共享内存 | 落每 warp 的局部和 | 20~30 周期延迟、128 B/周期、228 KB/SM 上限 |
| `bar.sync`（块级屏障） | 让局部和跨 warp 可见 | 每次把整个块卡住 |
| HBM | 第一趟读 n 个元素 | 3350 GB/s（H100）、400~800 周期延迟 |

（规格值，来源见 `code/specs.py`。）

**具体数字**：T = 1024（32 个 warp）、N = 2^20 时，两种块内形态的代价（推导值，`make verify` §B3）：

| 形态 | 块级屏障次数 | 共享内存字节数 | 全局读字节数 |
|---|---|---|---|
| 共享内存树（经典写法） | **11** | 12280 B | 4194304 B |
| warp shuffle + 每 warp 一个局部和 | **1** | 256 B | 4194304 B |

第三列两者**完全相同**——归约的全局流量由问题规模决定，与块内怎么做无关。差别全在前两列：**屏障从 11 次降到 1 次，共享内存流量从 12 KB 降到 256 B**。（屏障次数为什么会差这么多，[02 篇](02-warp-shuffle-and-hierarchy.md) §1 会逐条推导；这里只用它的结论。）

### 2.2 屏障「等」的粒度：不能指望它收敛

这是本文最容易写错的一处机制，值得单独说清。

`__syncthreads()` 是**块级**屏障。它在 PTX 层面的**记账号径是按 warp 计**的：`bar.sync` "executes on a per-warp basis as if all the threads in a warp are active"，到达计数按 **warp size 递增**（不是按实际活跃的 lane 数），而 `bar.sync` 的 thread-count 操作数**必须是 warp size 的整数倍**。换句话说：**一条 lane 到达，整个 warp 就记为到达。**（来源见 `code/specs.py` 的 `BAR_SYNC_*` 两条。）

由这条能推出两件事：

1. **屏障不能强制已经发散的 lane 收敛。** 屏障能做的只有「等待」，它**不改变任何 lane 的 PC**。要收敛必须靠控制流自己走到合流点，或者用 `__syncwarp()` 在 warp 内划界。
2. **它也不提供 warp 内 lane 之间的顺序保证。** 「warp 内所有 lane 隐含地停在同一处」这条保证在 Volta 就被取消了（[20-gpu-architecture/execution-model/02](../../20-gpu-architecture/execution-model/02-simt-and-divergence.md) §3「演进」，此处回链）——所以 **warp 内交换必须走 `__syncwarp()`**，见 §4.1。

> **⚠ 不要说「warp 是同步单位」。** 那句话在两种口径下都是错的：**Volta 之前**，「按 warp 记账」是硬件的**实现方式**，它不构成任何对程序的保证（一条 lane 到达即整 warp 记到，恰恰意味着另外 31 条 lane 可能还没到）；**Volta（CC 7.0）起**，`bar.sync` 改成**按线程**强制——块内所有未退出的线程都必须到达，否则屏障**不会释放**。规格上从未有过「warp 是同步单位」这一条。

§4.2 会把这两代硬件的差别推成一个具体的失败模式。

### 2.3 argmax 归约的部件开销

argmax 与 sum 归约的**结构完全相同**（同一棵树、同一组屏障），但每一步要搬两样东西：

| 操作 | sum 归约 | argmax 归约 |
|---|---|---|
| shuffle 指令数/步 | 1（值） | **2**（值 + 索引） |
| 比较 | 1 次浮点加法 | 一次比较 + 一次索引比较（并列时） |
| 共享内存/步 | 4 B | **8 B**（`float` + `int`） |

（推导值。）所以「带索引的归约」在换 shuffle 的时候**指令数是 sum 的两倍**，而这一点在算 roofline 时常常被漏掉——它不是带宽问题，是发射量问题。

## 3. 实现与演进

按「用什么手段把依赖链拆短」这条线：

1. **一个线程顺序归约**。work = n−1、depth = n−1。除极小的 n 之外没有可用性（§0 的 1048575 步）。
2. **共享内存树**。把折半那几步搬到共享内存里，用 `__syncthreads()` 划界。depth 降到 `n/T + log2(T)`，work 不变。T = 1024 时要 **11 次块级屏障**（§2.1）。这是 NVIDIA 早期 reduction 样例的形态。
3. **warp shuffle + 每 warp 一个局部和（本文的主形态）**。把「warp 内折半」整个换成`__shfl_down_sync`——**不过共享内存、不需要屏障**。屏障从 11 次降到 **1** 次，共享内存流量从 12280 B 降到 **256 B**（§2.1）。机制与 `width`/`mask` 的细节正文在 [02 篇](02-warp-shuffle-and-hierarchy.md)，此处回链。
4. **segmented 化**。段号不再等于线程编号，要先查表/二分定段（§1.2 第四步）。代价是每行一个偏移的额外访存，以及「迭代数由最长段决定」的空转（§1.2 第三步）。
5. **带索引（argmax/argmin）**。值与索引配对 shuffle（§1.3），并把 NaN 与 tie 的口径写进接口约定。
6. **跨 block**。单块装不下时，第一级各块出局部和，第二级归约局部和。两条路线的取舍与**确定性**正文在 [04 篇](04-multi-block-and-determinism.md)，此处回链。

每一步都在回答同一个问题：**这一步能不能把剩下那条链再拆短一点，而不增加新的同步点。** 第 3 步是这条线上收益最大的一次，因为它是**唯一一步同时减少了屏障次数和共享内存流量、而全局流量一个字节没变**的改动。

## 4. 踩坑与失败模式

### 4.1 warp 内交换用 `__syncthreads()` → 随机出错

**现象**：一段 warp 内 lane 之间通过共享内存传值的代码，**有时对有时错**。换个 block 大小、换张卡、或者同一份二进制跑两次，行为都可能不同；`compute-sanitizer` 也未必稳定复现。

**成因**：`__syncthreads()` 是**块级**屏障，且（PTX 口径）其到达计数**按 warp 计**——一条 lane 到达即整个 warp 记到。于是它**既不保证 warp 内 lane 之间的顺序**（lane A 写、lane B 读，两者的先后没有任何约定），**也不能强制已经发散的 lane 收敛**（屏障只「等待」，不改变任何 lane 的 PC）。Volta 之前，硬件按 warp 记账会让屏障在「有 31 条 lane 尚未到达」时就释放，那 31 条 lane 随后读到的是未写完的共享内存；Volta（CC 7.0）起改为按线程强制，同一段代码变成**永不释放**（挂死）。**同一段代码，两代硬件的失败模式相反：静默出错 vs 挂死。**

**规避**：warp 内交换用 `__syncwarp()` 划界，或干脆用 `__shfl_*_sync` 不经共享内存。跨 warp 的交换才用 `__syncthreads()`。规则始终是「`__syncthreads()` 必须被块内**所有**线程在同一点执行」。真机自检用 `compute-sanitizer --tool synccheck`。配套程序 `code/01_reduce_basics.cu` 里的 `wrong_warp_sync_demo` 就是这个错法的形态（**本仓库没有运行过它**）。

### 4.2 条件分支里的 `__syncthreads()` → UB / 死锁

**现象**：`if (tid < 32) { ...; __syncthreads(); }` 这类写法有时「看着能跑」，有时挂死。

**成因**：`__syncthreads()` 的语义是「块内所有线程都到达」。放进条件分支后，块内只有一部分线程会执行到它，屏障永不满足。**在 Volta 之前**，因为到达计数按 warp 计，屏障可能**提前释放**——那 31 条没到达的 lane 与已过屏障的 lane 之间就没有任何顺序保证，于是它表现为「静默出错」而不是挂死；**Volta 起**改为按线程强制，同样的代码**直接死锁**。两种表现都不是「正确」，只是错法不同。

**规避**：把 `__syncthreads()` 放在**所有线程都必然经过**的位置。如果确实只想让部分线程同步，那要的是 `__syncwarp()`（warp 内）或具名屏障 `bar.sync` 带线程数（必须同时是 warp size 的整数倍），而不是 `__syncthreads()`。**不存在「小心使用」的余地。**

### 4.3 拿 `threadIdx.x / 32` 当 segment 号

**现象**：per-row 归约的结果「大体对」，但某些行的值明显偏大或偏小，且换成不同的行长分布后错的行也跟着变。

**成因**：段边界与 warp 边界**不对齐**。`threadIdx.x / 32` 给出的是「第几个 warp 的 32 个连续元素」，不是「第几行」。段长 `[20, 12, 50, 6]` 时，第一个 warp 会把第 0 段的 20 个元素加上第 1 段的前 12 个元素一起归约：结果是 **508**，而第 0 段的真值是 **332**（推导值，`make verify` §C6）。注意 508 = 332 + 176，**它恰好等于两个相邻段的和**——所以这个错在「各行量级接近」时看起来很像一个偏大的正常值。

**规避**：**先定段、再归约。** 每线程用段起点数组查表（或对 `row_start` 二分）算出自己的段号，段内用 `__match_any_sync` 之类的原语或直接在共享内存里按段聚合。另外，**不要忘了补齐值是与算子绑定的**（§1.2 第五步）：加法补 `0`、max 补 `−∞`。这一条对 MoE 的 `tokens_per_expert` 尤其要紧——专家之间的 token 数差异可以很大，而补齐值写错时**只有短段会出错**，长段完全正常。

### 4.4 把 argmax 拆成「先 max、再搜索引」

**现象**：索引偶尔与期望差一个位置；把 tie 规则从「最小索引」改成「最大索引」后，索引**完全没变**——参数像是没接上。

**成因**：值归约和索引搜索是两个独立的动作。事后搜索天然从前往后找，于是永远给出**最小索引**，与声明的 tie 规则无关（推导值，`make verify` §A5：naive 找到索引 2，而 `tie='last'` 要求 4）。更隐蔽的形态是「只 shuffle 值、不 shuffle 索引」：值是对的、索引也是「一个真实的索引」，所以单看输出看不出问题。

**规避**：把 `(value, index)` 打包成一个结构体一起 shuffle、用同一个比较函数（配套程序的 `pair_max`）。同时把 NaN 与 tie 两条口径**写进文档和接口注释**——它们加起来有 4 种组合，两两之间都可能给出不同索引（§1.3）。

### 4.5 用「跑两遍结果一样」证明归约是确定性的

**现象**：为了证明某个跨 block 归约「可复现」，把同一个输入跑两遍、比较输出，发现完全一致，于是判定它确定性。上线后换一个数据分布或换一张卡，两次运行开始不一致。

**成因**：**浮点加法不满足结合律，但顺序改变并不保证改变位模式。** 132 个量级相当的局部和，正序与逆序累加在本例里**恰好得到完全相同的位**（推导值，`make verify` §C7）——因为两者的舍入路径落在了同一个可表示数上。同一批数据上 50 个随机排列才给出 12 种不同的位模式。所以「跑两遍一样」既不能证明确定性，也不能证明不确定性。

**规避**：判据看**代码里的加法树形是否固定**，不看一次观察——`atomicAdd` 的到达顺序由硬件决定，无论观察多少次都不能保证；workspace + 固定下标归约的顺序写死在代码里，与运行时无关。这条判据的完整形态（连同「哪些归约是确定的」分类）正文在 [04 篇](04-multi-block-and-determinism.md)，此处回链。要构造一个**必定**能暴露顺序敏感性的输入，用量级悬殊的数据（例如第 0 段 `1e8`、其余 127 段 `1.0`：fp32 在 `1e8` 处的 ulp 是 **8.0**，先加哪个会给出 `100000000` 与 `100000128` 两个不同的结果）。

## 编译与运行

配套程序是 **`code/01_reduce_basics.cu`**（顺序/树形归约、segmented 归约、配对 argmax、以及一段故意写错的 warp 同步）：

```bash
cd code
make 01_reduce_basics
./01_reduce_basics
```

程序会打印：§1 的 work/depth 与 Brent 模型表、segmented 归约的映射与利用率、argmax 的两条口径声明、以及 §4 那两条失败模式的说明。**两类数字要分开看**：

- work / depth / 屏障数 / 利用率 / 分段结果——全部是**推导值**，可用 `make verify` 的 §A1 / §A4 / §A5 / §B4 / §B5 / §C6 复核；
- 真机上的耗时与寄存器用量本仓库**没有**，程序里对应位置打印的是占位说明，本仓库对其数值不负责任。

本机（macOS）只能跑：

```bash
make verify
```

对应本文的是 §A1（树形归约的 work/depth 与 lane→结果映射）、§A4（segmented 归约与补值）、§A5（argmax 的 NaN 与 tie 口径）、§B4（work/depth 对照表）、§B5（segmented 的映射与利用率）、§C3（发散条件下 `__syncthreads()` 的规则模型）、§C5（NaN 与并列值的口径差异）、§C6（段边界切开 warp）。

## 6. 小结与延伸

本文一条主线：**并行归约不减少 work，只减少 depth——所以「顺序依赖」是它唯一能拆的东西。** 围绕它有三个可独立复用的中间结论：

1. **work 与并行度无关，恒为 n−1；depth 从 n−1 降到 log2 n。** 用 Brent 模型 `T = work/p + depth` 看，加速比的上界是 `work/depth`（n = 2^20 时 52429×），不是「有多少并行度就快多少」——p 从 1024 加到 2^20 只把加速比从 1004× 抬到 49932×。
2. **segmented 归约的困难不在「批量小」，在「边界不对齐」。** 用 `threadIdx/32` 当段号会把相邻两行串起来（508 = 332 + 176）；而正确做法（先定段再归约）的代价是**迭代数由最长段决定**——行长在 `[12, 200]` 之间时利用率只有 **47.3%**。另外记住**补齐值是与算子绑定的**：加法补 `0`、max 补 `−∞`。
3. **argmax 必须配对归约，且 NaN 与 tie 的口径要显式声明。** 值用 `fmaxf` 语义（IEEE 754-2008 `maxNum`，忽略 NaN），索引用最小索引——这是本文的口径。换成「传播 NaN」（IEEE 754-2019 `maximum` / `np.maximum`）在同一输入上会给出**不同的索引**，而在归一化里它会直接把整行的输出变成 NaN。

本文留下的问题是：§2.1 说「warh shuffle 把屏障从 11 次降到 1 次」——那 10 次屏障到底差在哪、`__shfl_*_sync` 的 `mask` 与 `width` 各自在管什么、写错了会怎样，是 **[02 篇](02-warp-shuffle-and-hierarchy.md)** 的内容。而 §1.2 的段长与 §0 的 n 都还只是「一个 block 装得下」的规模：装不下时第一级怎么切、第二级怎么收、以及「结果能不能复现」，是 [04 篇](04-multi-block-and-determinism.md) 的内容。

## 参考

- NVIDIA, *CUDA C++ Programming Guide* ——「SIMT Architecture」「Warp Vote Functions」「Warp Shuffle Functions」「Independent Thread Scheduling」各节；`__syncthreads()` 与 `bar.sync` 的到达计数口径、`__shfl_*_sync` 的 `mask`/`width` 语义。
- NVIDIA, *PTX ISA* —— `bar` 指令小节（「barriers are executed on a per-warp basis as if all the threads in a warp are active」、到达计数按 warp size 递增、thread-count 操作数必须是 warp size 的整数倍）。
- NVIDIA, *CUDA Math API* —— `fmaxf` 的 NaN 语义（一个参数为 NaN 时返回另一个），对应 IEEE 754-2008 的 `maxNum`。
- IEEE 754-2008 §5.3.1（`maxNum`/`minNum`，忽略 NaN）与 IEEE 754-2019（`maximum`/`minimum`，传播 NaN）—— 两代标准的差异是 §1.3 两条口径的来源。
- Mark Harris, *Optimizing Parallel Reduction in CUDA* —— 共享内存树形态与「每线程先吃多个元素」的经典分解。
- 本仓库 [20-gpu-architecture/execution-model/02](../../20-gpu-architecture/execution-model/02-simt-and-divergence.md) —— 发散与重汇聚的机制、`__syncwarp()` 与 `__activemask()` 的用法（本文只回链，不复述）。
- 本仓库 [20-gpu-architecture/memory-system/02](../../20-gpu-architecture/memory-system/02-coalescing-and-cache-line.md) —— 第一趟读 n 个元素的合并访存（正文在那里，本文只回链）。
- 本仓库 [20-gpu-architecture/execution-model/03](../../20-gpu-architecture/execution-model/03-occupancy-and-latency-hiding.md) —— 占用率约束与延迟隐藏（正文在那里，本文只回链）。
- 本仓库 `code/verify.py` 的 §A1、§A4、§A5、§B4、§B5、§C3、§C5、§C6 —— 本文章节里的数字来源。

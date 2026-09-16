---
title: warp shuffle 与多级归约
slug: warp-shuffle-and-hierarchy
domain: 40-kernels
topic: reduce-scan
level: 3
difficulty: intermediate
prerequisites:
  - 40-kernels/reduce-scan/01-reduction-algorithms
  - 20-gpu-architecture/execution-model/02-simt-and-divergence
tags: [warp-shuffle, shfl-down, shfl-xor, mask, width, syncthreads, barrier, multi-level-reduction]
related:
  - 20-gpu-architecture/execution-model/03-occupancy-and-latency-hiding
  - 20-gpu-architecture/memory-system/03-shared-memory-bank-conflict
  - 40-kernels/attention/02-flash-attention
  - 40-kernels/softmax-normalization/03-normalization-fusion
code:
  - code/02_warp_reduce.cu
status: stable
updated: 2026-09-16
---

# warp shuffle 与多级归约

> 同一个块内归约，用共享内存树要做 **11 次块级屏障、搬 12280 B 共享内存**；换成 warp shuffle 之后是 **1 次屏障、256 B**——而全局访存一个字节没变。代价是三条必须写对的语义：`mask`（收敛要求）、`width`（段划分）、以及「越界 lane 静默保留自己的值」。

## 0. 背景与动机

先看对照场景：**T = 1024 条线程（32 个 warp）的块内归约，N = 2^20 个 fp32**。

两种实现形态做的是同一件事：

```text
形态 A：共享内存树（经典写法）
    s[tid] = 局部和;
    for (off = T/2; off > 0; off >>= 1) {
        __syncthreads();                       // ← 每轮一次块级屏障
        if (tid < off) s[tid] += s[tid + off];
    }

形态 B：warp shuffle + 每 warp 一个局部和
    for (off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, off);   // ← 不过共享内存、不要屏障
    if (lane == 0) warp_sums[wid] = v;
    __syncthreads();                           // ← 唯一的块级屏障
```

代价对照（推导值，`make verify` §B3）：

| 形态 | 块级屏障次数 | 共享内存字节数 | 全局读字节数 |
|---|---|---|---|
| A 共享内存树 | **11** | 12280 B | 4194304 B |
| B warp shuffle + 每 warp 一个和 | **1** | 256 B | 4194304 B |

T = 128 / 256 / 1024 三档都是同一个形状（推导值，`make verify` §B3）：

| T | A 的屏障数 | B 的屏障数 | A 的 smem | B 的 smem |
|---|---|---|---|---|
| 128（4 warp） | 8 | 1 | 1528 B | 32 B |
| 256（8 warp） | 9 | 1 | 3064 B | 64 B |
| 1024（32 warp） | 11 | 1 | 12280 B | 256 B |

**第三列（全局读字节数）两者完全相同**——这一点值得先说清楚：**归约的全局流量由问题规模决定，与块内怎么归约无关。** 形态 B 换来的东西全在「屏障」和「共享内存」这两栏上。

本文要回答三个问题：

1. **`__shfl_*_sync` 到底做了什么？** 它的 `mask` 与 `width` 两个参数各自在管什么，谁改了语义、谁只改段划分？
2. **多级归约（warp → block → grid）的字节数与同步次数怎么算？** 第二级为什么不能省掉那次块级屏障？
3. **写错会怎样？** 特别是 `width` 写错——它的后果与直觉相反，而且**不是**很多人以为的「跨 warp 读」。

## 1. 原理

### 1.1 `__shfl_down_sync` 的语义：三个事实

**第一步：交换的范围是「调用 warp 的 32 条 lane」。**

`__shfl_*_sync` 系列允许**一个 warp 内部**的线程直接交换寄存器值，不经过共享内存。这个范围**与 `width` 无关**：`__shfl_*` **永不跨 warp**。

（`make verify` §C4 对**所有**合法的 `width` 与 `delta` 组合做了枚举，断言源 lane 恒落在调用者所在的 warp 内——越 warp 的组合数为 **0**。）

**第二步：`__shfl_down_sync(mask, var, delta, width)` 的源 lane 是「段内相对下标 + delta」。**

`width` 把 warp 切成**等长**的若干段（每段 `width` 条 lane），每段有自己的「逻辑 lane 0」。**段内**的下标加上 `delta` 就是源 lane。

```text
width = 8, delta = 4 时的源 lane（32 条 lane）：
  段 0： 4  5  6  7 | 4  5  6  7     ← 后四条越界
  段 1：12 13 14 15 |12 13 14 15
  段 2：20 21 22 23 |20 21 22 23
  段 3：28 29 30 31 |28 29 30 31
```

（推导值，`make verify` §A3。）竖线右边那 4 条 lane 的「段内下标 + 4」越出了本段。

**第三步：越界时**不回绕**，返回调用者自己的值。**

规约里写的是：「If the source lane ID is out of range or the source thread has exited, the calling thread's own `var` is returned.」而且 `__shfl_down_sync` 的源 lane **不按 `width` 回绕**——所以段内最高的 `delta` 条 lane 拿到的就是**自己的值**。

> **⚠ 这一点是 §4.1 那条失败模式的全部机制。** 在归约循环 `v += __shfl_down_sync(m, v, off, width)` 里，越界的 lane 会执行 `v += v`——**自己的值被计入了两次**。
>
> **⚠ 对照 `__shfl_sync`（不带方向的那个）**：它的越界源 lane **会**按 `srcLane % width` 回绕。两个变体的越界行为相反，不要混。

**第四步：`__shfl_xor_sync` 没有这个问题。**

`__shfl_xor_sync(mask, var, laneMask, width)` 的源 lane 是 `laneid ^ laneMask`。只要 `laneMask < width`，异或的结果必然仍是同一段内的 lane——**这个变体不存在「越界保留自身值」的情况**（`make verify` §A3 对每个段验证了这一点）。

所以蝴蝶型（butterfly）归约比 down 型更抗 `width` 误用：

```text
down 型：v += __shfl_down_sync(m, v, off, width);    off = width/2 … 1
xor  型：v += __shfl_xor_sync (m, v, off, width);    off = width/2 … 1
```

两者算出的**结果相同**（都是段内和），但 xor 型的源 lane 永远在段内，不会触发越界保留。**在只需要「段内和」时，xor 型是更安全的选择。**

**第五步：`mask` 参数意味着「收敛要求」。**

`mask` 是一张位掩码，含义是「**哪些 lane 参与这次交换**」。两条约束：

1. **出现在 `mask` 里的每一条 lane 都必须执行同一条 `__shfl_*_sync`。** 如果有 lane 在掩码里、却没有走到这条指令（例如它还在另一个分支里），行为未定义。
2. **调用者自己必须在 `mask` 里。** 全 warp 参与时习惯写 `0xffffffffu`；只让部分 lane 参与时用 `__activemask()` —— 但必须清楚 `__activemask()` 返回的是**此刻**活跃的 lane，它随发散状态变化，**不是一个稳定的常量**。

> **⚠ 把 `__activemask()` 的返回值存进变量、在分支合流之后再拿来当 `mask` 用**，是这类 bug 里最隐蔽的形态：掩码看着是「当时那个」掩码，实际上所有 lane 早已收敛，掩码与当前状态对不上。这条的机制与诊断在 [20-gpu-architecture/execution-model/02](../../20-gpu-architecture/execution-model/02-simt-and-divergence.md) §4.2，此处回链。

### 1.2 多级归约：lane → warp → block → grid

**第一步：先确定「谁持有结果」。**

warp 内折半（offset = 16, 8, 4, 2, 1）跑完之后，**只有 lane 0 持有整个 warp 的和**——其余 31 条 lane 停在自己那一层的部分和上（推导值，`make verify` §A3）：

```text
32 个输入值 → 5 步 shfl_down 之后每个 lane 持有什么：
  lane 0  = 1478   ← 恰好等于 32 个值的和
  lane 1  = 1512, lane 2 = 1519, … lane 31 = 352   ← 都是「部分和」，不是任何一行的答案
```

**第二步：段级同理——每个段的首 lane 持有该段的和。**

```text
width = 8 时 4 段的真值       = [387, 578, 509, 378]
归约后各段首 lane（0,8,16,24）= [387, 578, 509, 378]
```

（推导值，`make verify` §A3。）这就是「lane → 结果」的映射：**只有 `seg_start` 这些位置的 lane 持有效结果**，别的 lane 的值是中间过程。

**第三步：块级两级归约的字节与屏障。**

```text
第 1 级（warp 内）  5 步 shuffle                → 0 次屏障、0 字节共享内存
第 1.5 级           每 warp 的 lane 0 写 smem    → 32 次写 = 128 B（T=1024）
第 2 级（块内）     warp 0 读 32 个局部和再折半   → 32 次读 = 128 B
```

合计 **256 B 共享内存 + 1 次块级屏障**。

**第四步：那 1 次屏障为什么省不掉。**

第 2 级的写者是 **32 个不同 warp** 的 lane 0。`__syncwarp()` 只能同步**一个 warp 内部**的 lane，对「warp 3 写、warp 0 读」这种跨 warp 的依赖完全无效——所以这里**必须**用块级屏障。

```text
写者：32 个不同 warp 的 lane 0      → 跨 warp 依赖
读者：warp 0 的全部 lane            → 必须有块级屏障
```

**第五步：第二级只有 W 个元素，但 shuffle 需要 32 条 lane。**

`__shfl_down_sync` 在 32 条 lane 上工作，而第 2 级只有 W（= warp 数）个局部和。做法是**补零到 32**：

```text
acc = (lane < nwarps) ? warp_sums[lane] : 0.0f;   // ← 补 0
```

（推导值，`make verify` §A3 断言了三级归约的结果与 `np.sum` 一致，并验证了「不补零在模型里直接越界」。）

> **⚠ 补的值同样是与算子绑定的**（同 [01 篇](01-reduction-algorithms.md) §1.2 第五步）：加法补 `0`，max 归约要补 `−∞`。写 max 归约时照抄 `: 0.0f` 会把结果变成一个非负的值。

**第六步：三级（grid）的账。**

| 级别 | 并行单元 | 通信方式 | 屏障/同步 |
|---|---|---|---|
| warp 内 | 32 lane | `__shfl_down_sync` | 0（`__syncwarp()` 语义由 `_sync` 变体覆盖） |
| block 内 | 4~32 warp | 共享内存 | 1 次 `__syncthreads()` |
| grid 内 | G 个 block | 全局内存 / 原子操作 | 0 次屏障，代价是**一次额外启动**或**不确定性** |

第三行的两条路线与它们的确定性差异，正文在 [04 篇](04-multi-block-and-determinism.md)，此处回链。

### 1.3 归约用 down、扫描用 up

这是本主题里最容易搞混的一处对称性，先在这里立好：

| 用途 | 原语 | offset 序列 | 守卫 |
|---|---|---|---|
| **归约**（warp 内求和） | `__shfl_down_sync` | `width/2, …, 2, 1`（**递减**） | 不需要（lane 0 恰好拿到全和） |
| **扫描**（warp 内前缀和） | `__shfl_up_sync` | `1, 2, 4, …, width/2`（**递增**） | **必需**：`if (rel >= off)` |

（扫描的守卫为什么必需、漏掉会怎样，正文在 [03 篇](03-scan-algorithms.md) §4；此处只留下这张对照表。）

## 2. 执行映射

### 2.1 一次 `__shfl_*_sync` 在硬件上走什么

| 部件 | 在 shuffle 里做什么 | 量级 |
|---|---|---|
| warp scheduler | 发射 `SHFL` / `FADD` | 每 SM 4 个，各每周期 1 条 |
| 寄存器文件与 crossbar | `SHFL` 在 warp 内**搬运寄存器值** | 不经过共享内存，**不占 LSU** |
| 共享内存 | 只落每 warp 的**一个**局部和 | 20~30 周期延迟、128 B/周期、228 KB/SM 上限 |
| `bar.sync`（块级屏障） | 让 32 个 warp 的局部和互相可见 | 每块**仅 1 次** |
| HBM | 第一趟读 N 个元素 | 3350 GB/s（H100）、400~800 周期延迟 |
| 每 SM 常驻 | warp 上限 | 64 warp / 2048 线程 |

（规格值，来源见 `code/specs.py`。）

**关键点**：`SHFL` 走的是寄存器交叉开关，**不占共享内存的带宽、也不占 LSU 的发射槽**。这就是形态 B 能把共享内存流量从 12280 B 压到 256 B 的原因——它不是「把共享内存用得更省」，而是**根本不用**。

### 2.2 屏障为什么这么贵

形态 A 的 11 次屏障不是「多执行了 11 条指令」的问题：

- **屏障让整个块停下来等**。同一个 block 的 32 个 warp 必须全部到达才能继续，跑得快的 warp 全部空转。这与 [20-gpu-architecture/execution-model/03](../../20-gpu-architecture/execution-model/03-occupancy-and-latency-hiding.md) 讲的「靠其他 warp 填发射槽」正好相反——屏障是唯一一种**连其他 warp 也填不上**的空等。
- **屏障之间的依赖链无法被别的 warp 掩盖**。形态 A 的 10 轮折半是**串行**的（第 i 轮依赖第 i−1 轮），所以 10 次屏障就是 10 段串行区间。
- 形态 B 的 5 步 `__shfl_down_sync` 虽然也是串行依赖（每步的 `v` 依赖上一步），但**不需要把整个块拉进来**——只有 32 条 lane 的一条链，而且没有屏障开销。

所以「11 → 1」不是一个 11 倍的倍数，而是**把 10 段全块串行区间换成了 1 段**。块越大，这个差越明显（T = 128 时是 8 → 1，T = 1024 时是 11 → 1）。

### 2.3 段划分的代价：`width` 只在一种情况下改变结果

这一节是 §4.1 的定量铺垫。

`width` 改变的是**段划分**。直觉上「段划分变了，结果当然变」——但在 numpy 模型里逐例算出来（推导值，`make verify` §C2）不是这样：

```text
offset 序列 = [4, 2, 1]（固定），只改 width：
  width = 8 ：段首 lane 0 的值 = 166（= 8 条 lane 的和）
  width = 16：段首 lane 0 的值 = 166
  width = 32：段首 lane 0 的值 = 166
```

**三个 width 给出一模一样的结果。** 原因在依赖锥：lane 0 的依赖锥是

```text
{0, 4} ∪ {2, 6} ∪ {1, 5, 3, 7} = 8 条 lane
```

**恰好就是「每 8 条一段」那一段。** 越界只发生在段内**高位**的 lane 上（例如 lane 7 在 `delta=4` 时越界），而那些 lane **不在结果 lane 的依赖锥里**——它们保留自身值这件事传不到结果。

> **所以「`width` 写错 → 结果偏大」这个说法不够准确。** 准确的触发条件是：**循环里的 `offset` 出现了 ≥ `width` 的步**。那时该步**所有** lane 都越界（`rel + offset ≥ width` 对每条 lane 都成立），全部执行 `v += v`，结果被乘上 `2^k`（k = 超界的步数）。§4.1 会把这个倍数算准。

## 3. 实现与演进

按「用什么手段在 warp 内传递数据」这条线：

1. **共享内存 + `__syncthreads()`**。能用，但每次交换都要一次块级屏障，而且共享内存的读改写要走 LSU。T = 1024 时 11 次屏障、12280 B。
2. **`volatile` + 双缓冲的「warp 同步」写法**。这是 Volta 之前的标准技巧：用 `volatile` 防止编译器缓存、靠**锁步执行**隐含地保证 lane A 写完之后 lane B 才读。它在 Volta 上**失效**了，原因分两层（§4.2）。
3. **`__shfl_*_sync` + `mask`**（本文主形态）。不走共享内存，用寄存器交叉开关直接搬值。屏障降到 1 次，共享内存降到 256 B。**这是收益最大的一步，因为它是唯一一步同时减少屏障与共享内存、而全局流量一个字节没变。**
4. **`__shfl_xor_sync` 的蝴蝶型**。语义上更安全（`laneMask < width` 时源 lane 必在段内），代价是段外的 lane 也会跟着算一遍——在「只关心段首」的归约里这没有额外成本。
5. **多级分解（warp → block → grid）**。第 2 级用共享内存 + 1 次屏障；第 3 级交给跨块方案（[04 篇](04-multi-block-and-determinism.md)）。
6. **`__reduce_add_sync` 之类的硬件归约指令**（CC 8.0 起的 `redux.sync`）。把「5 步 shuffle + 5 次加法」压成一条指令。适用范围限于整数与 `min`/`max` 等少数算子，本主题不展开。

## 4. 踩坑与失败模式

### 4.1 `width` 写错的三种后果（只有一种是「偏大」）

**现象**：归约结果不对，但看不出明显规律。有时整段偏大一个整数倍，有时数值「看着完全合理」但分组错了。

**成因**：`width` 改变的是**段划分**，越界的 lane 静默保留自己的值。三种情形（全部用整数计算，所以倍数是精确值；推导值，`make verify` §C2）：

| 情形 | 循环 offset | 结果 |
|---|---|---|
| **offset 全部 < width** | 与段匹配 | **与 width 无关**（依赖锥 = 本段） |
| **offset 里出现 ≥ width 的步** | `[16,8,4,2,1]` 而 `width=8` | **恰好 4 倍**（偏大） |
| **width 写小、循环跟着 width 走** | `[2,1]` 而 `width=4` | **不偏大**，只是换了分组 |

第二行的机制：`offset=16` 与 `offset=8` 都 ≥ `width=8`，这两步里**所有** lane 都越界，每条 lane 执行 `v += v`，于是每步翻一倍、两步翻四倍：

```text
各段真值        = [166, 162, 189, 194]
误用后的段首    = [664, 648, 756, 776]     ← 比值恰好 [4.0, 4.0, 4.0, 4.0]
```

第三行是最难发现的一档：`width=4`、循环自动跟成 `[2,1]` 时，得到的是「每 4 条 lane 一段」的和——**数值合理、量级合理，只是回答了另一个问题**：

```text
每 4 条 lane 一段的结果 = [38, 128, 81, 81, 91, 98, 77, 117]
而「每 8 条 lane」的真值 = [166, 162, 189, 194]
```

**规避**：先确定「一段应该是多少条 lane」，然后让 `offset` 序列与 `width` **同时**从它推出来，不要一个写死一个跟着改。用 `__shfl_xor_sync` 可以彻底避免第二行那种情形（`laneMask < width` 时源 lane 必在段内）。另外记住 `width` 必须是 2 的幂且 ≤ 32——**其它取值是未定义，不是取模**（例如 `width=3` 不会退化成「3 条 lane 一段」）。

> **⚠ 不要把这个现象写成「跨 warp 读」。** `__shfl_*` **永不跨 warp**，与 `width` 无关：把 `width` 写错只会改变**同一个 warp 内**的段划分（推导值，`make verify` §C4 对全部合法 `width`/`delta` 组合枚举，越 warp 组合数为 0）。写成「跨 warp」会让排错方向整个跑偏——读者会去查 block 维度与网格配置，而真正的错在 `width` 的取值上。

### 4.2 `volatile` + 双缓冲的旧写法为什么在现代编译器下失效

**现象**：一段 Volta 之前的 warp 内归约代码，在 Pascal 上通过，在 Volta 及以后**随机出错**——同一份二进制有时对有时错，换个 block 大小或换张卡可能就好了。

**成因**：两层，缺一不可。

- **硬件层**：Volta 的 independent thread scheduling（CC 7.0）取消了「warp 内所有 lane 隐含地停在同一处」这条保证。lane A 写共享内存、lane B 读，两者之间**不再有任何顺序**。`volatile` 只约束编译器「每次访问都真的发出去」，它**不是内存屏障、也不产生 lane 间的顺序**——**它从来就不是同步手段**，只是锁步时代硬件替它把顺序补齐了。
- **编译器层**：`volatile` 保证的是「不做冗余访问」，不是「在正确的时刻访问」。锁步前提消失之后，`volatile` 剩下的那点保证不足以补上顺序。

**规避**：warp 内交换用 `__shfl_*_sync`（根本不用共享内存），或者用 `__syncwarp()` 显式划界。**带 `_sync` 后缀的原语必须传正确的 `mask`**，且不要缓存 `__activemask()` 的返回值（§1.1 第五步、[execution-model/02](../../20-gpu-architecture/execution-model/02-simt-and-divergence.md) §4.2）。

### 4.3 以为「1 次屏障」就意味着第二级也可以省

**现象**：把块内归约改成 shuffle 版本之后，为了「再省一次屏障」，把 `__syncthreads()` 删掉、改成 `__syncwarp()`——小规模测试全过，规模一大就偶发错误。

**成因**：第二级的写者是 **32 个不同 warp** 的 lane 0。`__syncwarp()` 的作用域**只有调用它的那一个 warp**，它无法让 warp 0 看到 warp 3 的写。屏障次数从 11 降到 1，靠的是把「warp 内的 10 轮折半」整个搬出了共享内存；剩下那 1 次**恰好是不可省的那一次**——它保护的是唯一的跨 warp 依赖。

**规避**：数一遍「这次同步保护的写者和读者分别是谁」。**同一个 warp 内 → `__syncwarp()`；跨 warp → `__syncthreads()`。** 照这个判据数，第二级那次永远是后者。另外**扫描**比归约多一次：扫描的第二级产出的偏移要**广播回所有 warp**，所以是 2 次块级屏障（正文在 [03 篇](03-scan-algorithms.md) §2，此处回链）。

### 4.4 用 `mask` 表达「部分 lane 参与」时忘了收敛要求

**现象**：`__shfl_*_sync` 在某些数据分布下返回垃圾值；`compute-sanitizer` 报 sync 类错误。

**成因**：`mask` 的语义是「这些 lane 参与这次交换」，它**不只是一个过滤器**——列在 `mask` 里的每一条 lane 都必须**执行到同一条指令**。如果某条 lane 在掩码里、却因为分支还在别处，这次交换的行为未定义。最隐蔽的形态是用 `0xffffffffu` 但代码实际处于发散状态（全掩码断言了「32 条 lane 都在这里」，而实际上不在）。

**规避**：只在**确定收敛**的位置用 `0xffffffffu`；真需要部分 lane 参与时，用当时真实活跃的掩码，并且**当场用**（不缓存）。写完一段 shuffle 代码后，问一句「这条指令执行时，掩码里的 lane 是不是都在这里」。

## 编译与运行

配套程序是 **`code/02_warp_reduce.cu`**（三种块内形态、`width` 误用、以及 `volatile` 旧写法的形态）：

```bash
cd code
make 02_warp_reduce
./02_warp_reduce
```

程序会打印：屏障次数与共享内存字节数的三档对照（§0 的两张表）、`__shfl_*_sync` 的语义要点、`width` 用错的三种后果、以及 §4.2 那条的说明。**两类数字要分开看**：

- 屏障数、字节数、段划分与倍数——全部是**推导值**，可用 `make verify` 的 §A3 / §B3 / §C2 / §C4 复核；
- 真机上的耗时与寄存器用量本仓库**没有**，程序里对应位置打印的是占位说明。

本机（macOS）只能跑：

```bash
make verify
```

对应本文的是 §A3（shuffle 语义的 numpy 模型与 lane→结果映射）、§B3（屏障次数与字节数）、§B8（扫描比归约**多一次**块级屏障的原因）、§C2（`width` 用错的三种后果）、§C4（shuffle 永不跨 warp）。

## 6. 小结与延伸

本文一条主线：**warp shuffle 把「块内归约」从共享内存搬到寄存器交叉开关上，换来的屏障从 log2(T) 次降到 1 次、共享内存流量从 O(T) 降到 O(warp 数)——而全局访存一个字节没变。** 围绕它有三个可独立复用的中间结论：

1. **三条必须写对的语义**：`mask` 表达「哪些 lane 参与」并隐含**收敛要求**（掩码里的 lane 必须都执行到这条指令，且不要缓存 `__activemask()`）；`width` 改变的是**段划分**；**越界的源 lane 静默返回调用者自己的值**（而 `__shfl_sync` 会回绕，两者相反）。`__shfl_xor_sync` 在 `laneMask < width` 时**没有**越界问题，更抗误用。
2. **多级归约的账**：warp 内 5 步 shuffle（0 屏障）→ 每 warp 一个局部和（32 次写）→ **1 次块级屏障** → warp 0 再折半。第 2 级那次屏障**省不掉**，因为写者是 32 个不同 warp 的 lane 0。而**扫描**要多一次（偏移要广播回所有 warp，共 2 次；`make verify` §B8 逐条列出两次同步各自保护的写者与读者）。
3. **`width` 写错的结果与直觉相反**：`offset` 全部小于 `width` 时**结果与 `width` 无关**（依赖锥恰好是本段）；只有 `offset ≥ width` 的步才会让**全部** lane 越界、执行 `v += v`、把结果乘上 `2^k`（例如 4 倍）；而 `width` 写小、循环跟着走的那一档**根本不偏大**，只是静默地换了分组。「偏大」的触发条件是 `offset ≥ width`，不是「width 写错」泛泛而论——**也绝不是「跨 warp 读」**。

本文留下的问题是：§1.3 那张「归约用 down、扫描用 up」的对照表，反过来写会怎样；以及 Blelloch 的两阶段之间漏掉屏障会怎样——那是 **[03 篇](03-scan-algorithms.md)** 的内容。

## 参考

- NVIDIA, *CUDA C++ Programming Guide* ——「Warp Shuffle Functions」一节（`__shfl_*_sync` 的源 lane 计算、`width` 必须是 2 的幂且 ≤ warpSize、越界源 lane 返回调用者自身值与 `__shfl_sync` 的回绕差别、`mask` 的收敛要求）。
- NVIDIA, *CUDA Math API* —— `__shfl_down_sync` / `__shfl_xor_sync` / `__shfl_up_sync` 各变体的返回值规则与 `width` 的合法取值。
- NVIDIA, *PTX ISA* —— `shfl.sync` 指令小节（段划分与越界行为）。
- NVIDIA, *CUDA C++ Programming Guide* ——「Independent Thread Scheduling」与 Volta 兼容性说明：`volatile` 不是同步手段、`__syncwarp()` 与各 `_sync` 后缀原语的引入原因。
- Mark Harris, *Optimizing Parallel Reduction in CUDA* 与 CUB 的 `BlockReduce` 文档 —— 共享内存树与「warp shuffle + 每 warp 一个局部和」两种形态的工程取舍。
- 本仓库 [20-gpu-architecture/execution-model/02](../../20-gpu-architecture/execution-model/02-simt-and-divergence.md) —— 发散与重汇聚、`__activemask()` 的时效性、`__syncwarp()` 语义（本文只回链，不复述）。
- 本仓库 [20-gpu-architecture/memory-system/03](../../20-gpu-architecture/memory-system/03-shared-memory-bank-conflict.md) —— 共享内存的 bank 冲突与交付带宽（正文在那里，本文只回链）。
- 本仓库 `code/verify.py` 的 §A3、§B3、§C2、§C4 —— 本文章节里的数字来源。

---
title: 扫描（前缀和）：Hillis-Steele 与 Blelloch
slug: scan-algorithms
domain: 40-kernels
topic: reduce-scan
level: 2
difficulty: intermediate
prerequisites:
  - 40-kernels/reduce-scan/01-reduction-algorithms
  - 40-kernels/reduce-scan/02-warp-shuffle-and-hierarchy
tags: [scan, prefix-sum, hillis-steele, kogge-stone, blelloch, brent-kung, work-depth, brent-model]
related:
  - 20-gpu-architecture/execution-model/02-simt-and-divergence
  - 20-gpu-architecture/memory-system/03-shared-memory-bank-conflict
  - 40-kernels/softmax-normalization/02-online-softmax
code:
  - code/03_scan.cu
status: stable
updated: 2026-09-16
---

# 扫描（前缀和）：Hillis-Steele 与 Blelloch

> n = 2^10 时，Hillis-Steele 做 10240 次加法、**10 步**；Blelloch 只做 2046 次加法，却要 **20 步**。**两者的 depth 差一倍——10 步 vs 20 步，不是「相同」。** 在 GPU 上「步数」就是屏障次数，所以这场比较没有胜者：**Blelloch 省 work，Hillis-Steele 省步数。**

## 0. 背景与动机

先看对照场景：**对 n = 2^10 = 1024 个元素做前缀和**。

| 算法 | work（加法次数） | depth（步数） |
|---|---|---|
| Hillis-Steele（Kogge-Stone） | **10240** | **10** |
| Blelloch（Brent-Kung） | **2046** | **20** |

（推导值，`make verify` §B1。）

两个数都值得盯一眼：

- **work 差 5 倍**（10240 / 2046 = 5.005）。Blelloch 的 work 是 `2n−2`，与 `log n` 无关；Hillis-Steele 的 work 是 `n·log n`，随 `n` 长。n 从 2^10 长到 2^20 时，比值从 5.005 涨到 **10.000**（推导值，`make verify` §B1）。
- **depth 差一倍**（10 vs 20）。这一条很容易被写错：**两者都在 Θ(log n) 里，但常数不同**——Hillis-Steele 是 `log n`，Blelloch 是 `2·log n`。说「两者 depth 相同、只差 work」是**错的**，n = 2^16 时打印出来是 16 vs 32，n = 2^20 时是 20 vs 40。

为什么 depth 的差别在 GPU 上特别要紧？因为**在单块（共享内存）实现里，每一步都要一次块级屏障**（[02 篇](02-warp-shuffle-and-hierarchy.md) §2.2）。n = 2^10 时 Hillis-Steele 要 **10 次**屏障、Blelloch 要 **20 次**；而屏障是**全块串行**的区间，且这 10 / 20 步之间是**严格串行依赖**（第 i 步读第 i−1 步的写），无法被其他 warp 掩盖。

于是「省 work」和「省步数」是两笔**不同方向**的收益。把它们合成一个数需要 Brent 模型：

```text
T = work/p + depth        p = 并行度
```

代入两个算法的 work/depth 解出临界 p（推导值，`make verify` §B2）：

| n | 临界 p | 占 n 的比例 |
|---|---|---|
| 2^10 | **819.4** | 80% |
| 2^16 | **57344.1** | 88% |
| 2^20 | **943718.5** | 90% |

```text
p 小  → work/p 项主导 → **Blelloch 赢**（它的 work 是 2n，不是 n·log n）
p 大  → depth 项主导  → **Hillis-Steele 赢**（它的 depth 只有一半）
```

**注意方向**：work 项是 `work/p`，p 越大它**越不重要**。所以「p 大时 Blelloch 省 work 赢」是把方向搞反了——p 大的时候 depth 才是决定项。具体数字（推导值，`make verify` §B2）：

| n | p ≈ n/100（Blelloch 的领先） | p = n（Hillis-Steele 的领先） |
|---|---|---|
| 2^10 | **4.604×**（p = 10） | +10.0% |
| 2^16 | **6.966×**（p = 655） | +6.2% |
| 2^20 | **8.417×**（p = 10485） | +5.0% |

**临界 p 逼近元素个数 n 本身**（80%~90% 的 n）。换句话说：只有在并行度**几乎等于元素数**（一条 lane 一个元素）时 Hillis-Steele 才反超，而且优势只有 5%~10%。**这解释了为什么 GPU 上的块内扫描不用纯 Blelloch**：单块里 `n ≈ T`，正好落在 Hillis-Steele 占优的那一侧（§2 会给混合形态）。

## 1. 原理

### 1.1 两种算法

**Hillis-Steele（inclusive）**：第 k 步（k = 0, 1, …）里，位置 `i` 读位置 `i − 2^k` 并相加。

```text
k=0: x[i] += x[i-1]
k=1: x[i] += x[i-2]
k=2: x[i] += x[i-4]
...
```

**Blelloch（exclusive）**：两阶段。

```text
up-sweep  ：自底向上两两求和，把和放在右端点
根置幺元  ：最右端置 0
down-sweep：自顶向下分发
```

两者都是**前缀和**（同一个数学函数），但用的加法树不同。`make verify` §A2 在 n = 1024 与 n = 4096 上用整数输入断言了：

- Hillis-Steele 的 inclusive 结果与 `np.cumsum` **逐元素精确相等**；
- Blelloch 的 exclusive 结果与 `np.cumsum` 右移一位精确相等；
- **两者的 exclusive 结果逐元素相同**（它们算的是同一个函数，尽管树形不同）。

### 1.2 重推 work：`n·log n` 与 `2n−2`

**第一步：数 Hillis-Steele 的加法次数。**

第 k 步里，位置 `i < 2^k` 的 lane 没有可以读的源（`i − 2^k < 0`），这一步它**不做加法**。所以第 k 步的有效加法次数是 `n − 2^k`：

```text
work_valid = Σ_{k=0}^{log2(n)−1} (n − 2^k)
           = n·log2(n) − (2^log2(n) − 1)
           = n·log2(n) − (n − 1)
```

n = 2^10：`1024×10 − 1023 = 9217`。（推导值，`make verify` §B1 用实跑计数断言了这个公式：**实跑 9217，公式 9217**。）

**第二步：区分「发射口径」与「有效口径」。**

在 GPU 上，每条 lane 每一步都会**执行**那条加法（越界的用谓词屏蔽，或者根本就是白算）。所以从**发射量**看是 `n·log n`：

```text
work_发射 = n·log2(n) = 10240（n = 2^10）
work_有效 = n·log2(n) − (n−1) = 9217
```

> **本文的表格与比值统一用「发射口径」`n·log n`**，因为 GPU 上真正付出的是发射量。有效口径同时给出，两者都可核对。注意用发射口径时，n = 2^10 的比值是 **5.005**（10240/2046），不是整 5。

**第三步：数 Blelloch 的加法次数。**

```text
up-sweep  ：第 1 层 n/2 次、第 2 层 n/4 次 … 最后一层 1 次
            → n/2 + n/4 + … + 1 = n − 1
down-sweep：每层 n/(2d) 个结点，每结点 **1 次加法**（另有一次移位，移位不是加法）
            → n/2 + n/4 + … + 1 = n − 1
```

```text
work = (n−1) + (n−1) = 2n − 2
```

n = 2^10：`1023 + 1023 = 2046`。（推导值，`make verify` §B1 断言实跑计数 up = down = 1023、合计 2046。文档与表格常按习惯记作 `2n`。）

**第四步：work 比值。**

```text
比值 = n·log2(n) / (2n−2)  →  log2(n)/2   （n 大时）
```

| n | 发射口径的 work 比 | 有效加法的 work 比 |
|---|---|---|
| 2^10 | 5.005 | 9217 / 2046 = 4.505 |
| 2^16 | 8.000 | 983041 / 131070 = 7.500 |
| 2^20 | 10.000 | 19922945 / 2097150 = 9.500 |

（推导值，`make verify` §B1。）**比值随 n 增长**——这是 `log n` 出现在分子上的直接后果。

### 1.3 重推 depth：`log n` 与 `2·log n`

**第一步：Hillis-Steele 的步数。**

第 k 步的偏移是 `2^k`，从 `1` 涨到 `n/2`：

```text
1, 2, 4, …, n/2     → 共 log2(n) 步
depth = log2(n)
```

n = 2^10 → **10 步**，n = 2^16 → **16 步**，n = 2^20 → **20 步**。

**第二步：Blelloch 的步数。**

```text
up-sweep  ：d = 1, 2, 4, …, n/2        → log2(n) 层
down-sweep：d = n/2, …, 2, 1           → log2(n) 层
depth = 2·log2(n)
```

n = 2^10 → **20 步**，n = 2^16 → **32 步**，n = 2^20 → **40 步**。

**第三步：两者不相等。**

```text
Hillis-Steele: 10   Blelloch: 20      （n = 2^10）
Hillis-Steele: 16   Blelloch: 32      （n = 2^16）
Hillis-Steele: 20   Blelloch: 40      （n = 2^20）
```

（`make verify` §B1 逐行断言 `depth_HS ≠ depth_Blelloch`，并断言 `depth_Blelloch == 2 × depth_HS`。）

> **⚠ 「两者都是 Θ(log n)、所以 depth 相同」是错的。** `Θ` 抹掉的是常数，而这里被抹掉的恰好是**两倍**——在 GPU 上那两倍就是**两倍的屏障次数**。这正是「Blelloch 省 work、Hillis-Steele 省步数」这句话成立的**前提**：如果 depth 真的相同，那么 Brent 模型里两项会被同时优化，就不存在「两种赢法」了。

**第四步：临界 p。**

```text
T_HS(p) = n·log2(n)/p + log2(n)
T_B (p) = (2n−2)/p    + 2·log2(n)

T_HS < T_B  ⟺  (n·log2(n) − 2n + 2)/p < log2(n)
            ⟺  p > (n·log2(n) − 2n + 2)/log2(n)        ← 两边同乘 p、再除以 log n
            =  n(1 − 2/log2(n)) + 2/log2(n)
            ≈  n(1 − 2/log2(n))
```

**注意不等号方向**：左边那个系数 `n·log2(n) − 2n + 2` 是**正的**（n ≥ 4 时成立），
所以两边同乘 `p` 之后是 `p >`，不是 `p <`。**即：p 大于临界值时 Hillis-Steele 赢。**

| n | 临界 p（精确） | `n(1 − 2/log n)` |
|---|---|---|
| 2^10 | 819.4 | 819.2 |
| 2^16 | 57344.1 | 57344.0 |
| 2^20 | 943718.5 | 943718.4 |

（推导值，`make verify` §B2。）**临界 p 是 n 的 80%~90%**——这就是 §0 那个「只有在并行度几乎等于元素数时才反超」的定量形态。

### 1.4 exclusive 与 inclusive 的换算

两者只差一位位移：

```text
exclusive[i] = inclusive[i-1],  exclusive[0] = 0（幺元）
inclusive[i] = exclusive[i] + x[i]
```

（`make verify` §A2 断言了这条换算在两种算法上都成立。）**注意 `0` 是加法的幺元**——换成 max 扫描，`exclusive[0]` 要置 `−∞`（同 [01 篇](01-reduction-algorithms.md) §1.2 第五步）。

### 1.5 非 2 的幂：补齐

Blelloch 的树形要求 `n` 是 2 的幂。n = 1000 时补到 1024（**补在尾部**）：

```text
补零到 1024 后，Blelloch 仍给出正确的前 1000 项前缀
```

（推导值，`make verify` §A2。）**补在尾部不改变任何已有项的前缀和**——因为加法的幺元 `0` 加在末尾，前面每一项的 `inclusive` 值都不变。同一个技巧在 [04 篇](04-multi-block-and-determinism.md) 的块偏移数组上还会再用一次（块数往往不是 2 的幂）。

## 2. 执行映射

### 2.1 warp 内扫描：up shuffle + 守卫

warp 内做 Hillis-Steele 扫描：

```text
for (off = 1; off < 32; off <<= 1) {
    float t = __shfl_up_sync(0xffffffffu, v, off);
    if (lane >= off) v += t;        // ← 守卫：必需
}
```

两个细节都必须写对：

- **扫描用 `up`、offset 从 1 递增**（归约用 `down`、offset 从 `width/2` 递减，见 [02 篇](02-warp-shuffle-and-hierarchy.md) §1.3）。
- **守卫 `if (lane >= off)` 是正确性的一部分，不是性能优化。** `__shfl_up_sync` 在段内越下界时返回**调用者自己的值**，于是 lane 0 在每个 step 都会执行 `v += v`——5 个 step 下来它把 `x[0]` 计入了 **32 次**（推导值，`make verify` §C2）：输入 `x[0] = 14` 时，正确的 lane 0 是 **14**，漏守卫的 lane 0 是 **448 = 32 × 14**。倍数恰好是 `2^log2(32) = 32`。

> **守卫生效的方式是「不加」，不是「加自己」。** 写成 `if (lane < off) t = v;` 会差一整个 2 的幂——这个错法在代码里看起来同样「很整齐」。

### 2.2 块内扫描：warp 内扫描 + 跨 warp 前缀

单块内 n ≈ T 时，Brent 模型里的 depth 项主导（§1.3），所以工程实现**不用纯 Blelloch**，而是三级拼接：

```text
第 1 级：每个 warp 对自己那 32 个元素做 warp 内 inclusive 扫描（5 步 shuffle）
第 2 级：每 warp 的总和（lane 31 持有）落共享内存；
        warp 0 对这 W 个总和再做一次 warp 内扫描；
        用 __shfl_up_sync 取「前一条 lane 的 inclusive 值」得到 exclusive 偏移
第 3 级：各 warp 把自己的结果加上本 warp 的块内偏移
```

**屏障数是 2**，比归约的 1 多一次：

| | 屏障数 | 原因 |
|---|---|---|
| **归约**（warp shuffle + 每 warp 一个和） | **1** | 第二级只产出一个标量，warp 0 直接写全局就结束了 |
| **扫描**（warp 内扫描 + 跨 warp 前缀） | **2** | 第二级产出的**偏移必须广播回所有 warp**，所以还要一次屏障 |

（推导值，`make verify` §B8 逐条列出两次同步各自保护的写者与读者；配套程序 `code/03_scan.cu` 的 `block_scan_hybrid` 就是这个形态。）这是「扫描比归约贵」的一个常被忽略的来源——另一处是下面的共享内存流量。

### 2.3 共享内存流量：HS 每步都动 n 个元素

| n | HS 步数 | HS smem 访问 | Blelloch 步数 | Blelloch smem 访问 | 比值 |
|---|---|---|---|---|---|
| 2^5 | 5 | 320 | 10 | 124 | 2.58 |
| 2^8 | 8 | 4096 | 16 | 1020 | 4.02 |
| 2^10 | 10 | **20480** | 20 | **4092** | **5.005** |

（推导值，`make verify` §B7。数法：HS 每步 n 读 + n 写 = `2n`，共 `log n` 步；Blelloch 的 up/down 各自 `n−1` 次读改写 = `2(2n−2)`。）

**这正是 depth 与 work 的取舍在硬件上的直接形态**：

- Blelloch 每一步只动**一半**的线程（树形，第 k 层只有 `n/2^k` 个活跃结点），所以它的**访问总量小**；
- 但它的**步数是两倍**，而 GPU 上每一步都要一次块级屏障，**屏障是整个块串行的**。

| 部件 | 在扫描里做什么 | 量级 |
|---|---|---|
| warp scheduler | 发射 `SHFL` / `FADD` / `LDS` / `STS` | 每 SM 4 个，各每周期 1 条 |
| 共享内存 | HS 的每步读改写；Blelloch 的上/下扫 | 20~30 周期延迟、128 B/周期 |
| `bar.sync`（块级屏障） | 每步之间划界 | HS 10 次 vs Blelloch 20 次（n=2^10、T=1024） |
| HBM | 读 n 个元素、写 n 个结果 | 3350 GB/s（H100）、400~800 周期延迟 |

（规格值，来源见 `code/specs.py`。）

**结论**：在单块内 `n ≈ T` 的那一侧，**步数决定成败**，所以是「warp 内 shuffle 扫描（HS 形态）+ 跨 warp 前缀」的混合形态胜出——它把 log2(T) 步压进了 warp 内部（**不需要屏障**），只在跨 warp 处留 2 次屏障。而 Blelloch 的优势（work 小）只有在并行度低得多、或者元素数远大于线程数时才兑现。

## 3. 实现与演进

按「把 depth 压在哪里」这条线：

1. **单线程顺序扫描**。work = n−1，depth = n−1。除极小 n 外不可用。
2. **共享内存 Hillis-Steele**。depth = log2(n) 步，每步一次块级屏障。n = 2^10 时 10 步、20480 次共享内存访问（§2.3）。这是最直白的实现。
3. **共享内存 Blelloch**。work 降到 `2n−2`，但 depth 涨到 `2·log2(n)`——**步数翻倍**。在单块的尺度上通常是净亏（§1.3 的临界 p ≈ 0.8n），只有在元素数远大于线程数时才划算。
4. **warp 内 shuffle 扫描 + 跨 warp 前缀**（本文主形态）。把最下面 log2(32) = 5 步压进 warp 内部，**那 5 步不要屏障**；跨 warp 处只剩 2 次（§2.2）。这是 CUB `BlockScan` 的形态。
5. **多 block 三段式扫描**。块内扫描 → 收集块总和 → exclusive 扫描块总和 → 加回偏移。正文在 [04 篇](04-multi-block-and-determinism.md)，此处回链。
6. **单遍（decoupled look-back）扫描**。把第 4 步的「收集—扫描—加回」三个阶段合并进一个 kernel，用全局标志与前缀做 look-back。代价是引入了**顺序依赖的运行时时序**，确定性需要额外小心（[04 篇](04-multi-block-and-determinism.md) §3）。

这条线的方向很清楚：**每一步都在把「需要块级屏障的步」往「不需要屏障的 warp 内部」或「更粗的粒度」搬。**

## 4. 踩坑与失败模式

### 4.1 Blelloch 的 up-sweep 与 down-sweep 之间漏掉 `__syncthreads()`

**现象**：前缀和的结果在**后半段**出错，前半段完全正确；元素数越大错得越多。改小 n 有时会「碰巧对」。

**成因**：up-sweep 的最后一个写者与 down-sweep 的第一个读者**跨 warp**。中间那次「根置幺元」`if (tid == 0) s[n-1] = 0;` 之后必须有一次块级屏障，让全块看到这个写。漏掉它，某些 warp 会在根还没置零时就开始 down-sweep——读到的根是 up-sweep 留下的**总和**。这个错**不会报错**，只是给出另一个答案：它把整棵树的下发基准值变得很大，而且**首元素（`exclusive[0]`）仍然是 0**（它是硬写的），所以「查首项」这种自检抓不到。

**规避**：把 Blelloch 的两阶段之间的那次屏障当成**结构的一部分**（配套程序 `code/03_scan.cu` 的 `blelloch_scan` 里专门写了注释标记它）。更稳妥的做法是用「每一层结尾一次屏障」的统一写法——多一次屏障，但不会漏。诊断手段：把 kernel 的输出与 CPU 上的 `np.cumsum` 逐元素比对（而不是只看首尾两项）。

### 4.2 把 inclusive / exclusive 的位移方向搞反

**现象**：结果整体偏一格；或者块内正确、块间偏移全错（每个块都多算了自己）。

**成因**：inclusive 与 exclusive 只差一位位移（§1.4），**在代码里体现为一行的有无**。最容易出错的形态是**块偏移**：第 `b` 块的偏移应该是「前面所有块的总和」，用 inclusive 会把它**自己**的总和也算进去，于是第 0 块之后的每一块都整体偏了一个块的量。

（推导值，`make verify` §A6：三段式扫描的块偏移改用 inclusive 后，**首个不符处就是索引 0**，差值等于第 0 块的总和——这类错是「整体平移」而不是「小误差」，所以自检抓得到。）

**规避**：把 `exclusive` 的构造写成一个固定套路：`exclusive[i] = inclusive[i-1]`、`exclusive[0] = 幺元`。用 `__shfl_up_sync` 取前一条 lane 的 inclusive 值，并且 `(lane == 0) ? 0.0f : prev`——**`lane 0` 那一支不能漏**。测试时**逐元素**与参考实现比对，不要只看首尾。

### 4.3 扫描用了 `down` shuffle（把归约的写法抄过来）

**现象**：扫描结果不对，但 **lane 0 恰好等于总和**——也就是**恰好等于 inclusive 扫描的最后一项**。所以「out[0] 对不对」这种自检会给出误导性的结论。

**成因**：归约用 `down`（offset 从 `width/2` 递减）、扫描用 `up`（offset 从 1 递增），两个循环长得几乎一样。抄成 down 之后，每条 lane 拿到的既不是前缀和也不是后缀和，而是**方向相反的树的部分和**（推导值，`make verify` §C8）：

```text
正确（shfl_up + 守卫）前 6 项 = [10, 13, 27, 31, 45, 55]    ← = np.cumsum
down 版前 6 项                = [320, 328, 337, 347, 341, 349]
后缀和（反向 cumsum）前 6 项   = [320, 310, 307, 293, 289, 275]
```

lane 0 = **320 = 全和**（巧合：它的依赖锥恰好覆盖全体），而 lane 31 = **32 × x[31]**（末位 lane 被自己重复计入了 32 次，机制同 [02 篇](02-warp-shuffle-and-hierarchy.md) §4.1）。

**规避**：记住那张对照表（[02 篇](02-warp-shuffle-and-hierarchy.md) §1.3）——**归约 down、扫描 up**。真正的防线是逐元素比对参考实现：单点自检（查首项）在 lane 0 恰好正确时抓不到它，但**逐元素比对一定抓得到**。

### 4.4 漏掉扫描的守卫，以为它只是「省一条加法」

**现象**：前缀和整体偏大，且**偏得很有规律**——首元素是 `x[0]` 的 32 倍。

**成因**：守卫 `if (lane >= off)` 不是性能优化。`__shfl_up_sync` 在段内越下界时返回调用者自己的值，于是段内前 `off` 条 lane 执行 `v += v`；5 个 step 各翻一倍 → **恰好 32 倍**（§2.1）。

**规避**：守卫写成「不加」而不是「加自己」。这条与 §4.3、与 [02 篇](02-warp-shuffle-and-hierarchy.md) §4.1 是**同一族**错误：根源都是「越界 lane 静默保留自身值」这一条语义。写任何 shuffle 循环时先问一句：**这一步里哪些 lane 会越界，它们会拿到什么？**

### 4.5 对非 2 的幂的 n 直接用 Blelloch

**现象**：树形索引算错，结果莫名其妙；n 恰好是 2 的幂时一切正常。

**成因**：Blelloch 的 `d = 1, 2, 4, …` 与结点下标 `k = 2·d·(i+1) − 1` 都假设了 `n` 是 2 的幂。n = 1000 时，最后一层会有一批结点落空，down-sweep 的下发基准对不上。

**规避**：补到下一个 2 的幂，**补在尾部**（加法的幺元加在末尾不改变前面任何一项的前缀，§1.5）。注意补充的值同样与算子绑定：加法补 `0`、max 扫描补 `−∞`。如果补零的代价不可接受（n 刚好比 2 的幂大一点，例如 1025），改用 Hillis-Steele 形态或「warp 内 shuffle 扫描 + 跨 warp 前缀」的混合形态——它们不要求 `n` 是 2 的幂。

## 编译与运行

配套程序是 **`code/03_scan.cu`**（Hillis-Steele、Blelloch、块内混合扫描，以及两个故意写错的版本）：

```bash
cd code
make 03_scan
./03_scan
```

程序会打印：§1 的 work/depth 表与 Brent 临界 p、§2 的共享内存流量表、§4 的失败模式清单。**两类数字要分开看**：

- work / depth / 步数 / smem 访问次数 / 临界 p / 屏障数——全部是**推导值**，可用 `make verify` 的 §A2 / §B1 / §B2 / §B7 / §B8 / §C8 复核；
- 真机上的耗时本仓库**没有**，程序里对应位置打印的是占位说明。

本机（macOS）只能跑：

```bash
make verify
```

对应本文的是 §A2（两种扫描 vs `np.cumsum`、非 2 的幂的补齐）、§B1（work/depth 表，含「两者 depth 不相等」的断言）、§B2（Brent 模型的临界 p）、§B7（共享内存流量）、§B8（扫描比归约多一次块级屏障）、§C8（方向搞反的结果既不是前缀和也不是后缀和）。

## 6. 小结与延伸

本文一条主线：**Hillis-Steele 与 Blelloch 的 work 与 depth 都不同——Blelloch 省 work（`2n−2` vs `n·log n`），Hillis-Steele 省步数（`log n` vs `2·log n`）。** 围绕它有三个可独立复用的中间结论：

1. **两笔账要一起算。** n = 2^10 时 work 是 10240 vs 2046（发射口径比值 **5.005**），depth 是 **10 vs 20**——**不是「depth 相同」**。用 Brent 模型 `T = work/p + depth` 解出临界 p = `n(1 − 2/log n)`，**逼近元素个数 n 本身**（80%~90%）。p 小（work 主导）→ Blelloch 赢；p 大（depth 主导）→ Hillis-Steele 赢，且优势只有 5%~10%。
2. **GPU 上「步数」就是屏障次数。** 单块内 `n ≈ T` 落在 Hillis-Steele 占优的那一侧，所以工程实现是「warp 内 shuffle 扫描 + 跨 warp 前缀」的混合形态：把 5 步压进 warp 内部（**不要屏障**），跨 warp 处留 **2 次**——比归约多一次，因为扫描的偏移必须**广播回所有 warp**。
3. **共享内存流量的方向与步数相反。** n = 2^10 时 HS 是 20480 次访问、Blelloch 只有 4092 次（比值 5.005），因为 Blelloch 树形每层只动一半的线程。**省 work 与省步数是两笔方向不同的收益**，这就是为什么没有单一胜者。

本文留下的问题是：单块装不下时，第 4 步「块偏移」的那次扫描放在哪里、以及跨块方案能不能复现——那是 **[04 篇](04-multi-block-and-determinism.md)** 的内容（本文 §3 第 5、6 步都已指向它）。

## 参考

- Hillis & Steele, *Data parallel algorithms*, Communications of the ACM 1986 —— Hillis-Steele（Kogge-Stone）扫描的原始文献。
- Blelloch, *Prefix Sums and Their Applications*, 1990 —— Brent-Kung / Blelloch 扫描的 up-sweep + down-sweep 两阶段形态、`2n−2` 的 work 与 `2·log n` 的 depth。
- Brent, *The parallel evaluation of general arithmetic expressions*, JACM 1974 —— Brent 模型 `T = work/p + depth` 的来源（本文 §0 与 §1.3 用它比较两种赢法）。
- NVIDIA, *CUDA C++ Programming Guide* —— `__shfl_up_sync` 的段内越界语义（§2.1 守卫的必要性）。
- CUB 文档的 `BlockScan` —— 「warp 内 shuffle 扫描 + 跨 warp 前缀」的工程形态与它对 `n` 非 2 的幂的处理。
- 本仓库 [40-kernels/reduce-scan/02](02-warp-shuffle-and-hierarchy.md) —— shuffle 的 `mask`/`width` 语义、归约与扫描的 down/up 对照表（本文只回链，不复述）。
- 本仓库 [20-gpu-architecture/memory-system/03](../../20-gpu-architecture/memory-system/03-shared-memory-bank-conflict.md) —— 共享内存的 bank 冲突（本文 §2.3 的总量口径不含冲突，正文在那里）。
- 本仓库 `code/verify.py` 的 §A2、§B1、§B2、§B7、§C8 —— 本文章节里的数字来源。

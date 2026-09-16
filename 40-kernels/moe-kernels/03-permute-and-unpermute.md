---
title: 路由的重排与回收
slug: permute-and-unpermute
domain: 40-kernels
topic: moe-kernels
level: 3
difficulty: intermediate
prerequisites:
  - 40-kernels/moe-kernels/01-moe-compute-pattern
  - 20-gpu-architecture/memory-system/02-coalescing-and-cache-line
  - 40-kernels/reduce-scan/01-reduction-algorithms
tags: [moe, permute, unpermute, scatter, gather, coalescing, masking, capacity-factor]
related:
  - 40-kernels/moe-kernels/02-grouped-gemm
  - 40-kernels/reduce-scan/03-scan-algorithms
  - 80-model-architectures/moe
  - 20-gpu-architecture/memory-system/03-shared-memory-bank-conflict
code:
  - code/03_permute_unpermute.cu
status: stable
updated: 2026-09-16
---

# 路由的重排与回收

> permute 是**行级的 scatter**，unpermute 是**行级的 gather + 加权求和**。散开的是**行号**，一行内部的 `d_model` 个元素永远是连续的——所以「合不合并」只取决于一件事：**一个 warp 的 32 个 lane 是沿着 `d_model` 走，还是沿着 slot 走**。这两种映射相差 **16 倍**的访存字节。

## 0. 背景与动机

[01 篇](01-moe-compute-pattern.md)把 MoE 一层的四个 kernel 分了工，[02 篇](02-grouped-gemm.md)解决了中间那个 `grouped GEMM` 的 tile 调度。剩下的两个——把 token 排成专家序、算完再排回来——看上去只是「数据搬运」，是四步里最没有技术含量的两步。

但它的形状很特别。以本主题的场景（`tokens=4096, top_k=2, d_model=4096, FP16`，**选定值，见 `code/specs.py` 的 `UNSOURCED`**）为例：

```text
一行 = d_model 个 FP16 元素 = 8192 B = 8.0 KB
输入 4096 行，permute 之后 8192 行（每个 (token, 专家) 对一行）
```

（推导值，`make verify` §B6。）**每次搬运的粒度是 8 KB，而它要去的位置由路由决定。**

这篇文档要回答三个问题：

1. 这个 scatter/gather 该怎么映射到线程上？为什么同一个算法能写出 16 倍的访存差距？
2. 被容量截断丢掉的 slot，在 unpermute 时该怎么处理？
3. 这一步值不值得优化？（先给答案：它的流量是同层专家权重的 **10.71%**——不是瓶颈，但也不小到可以乱写。）

**容量因子本身的定义与语义不在本域**，正文在 `80-model-architectures/moe`（正文在彼，此处回链）。本文只讲它落到 kernel 上的那部分：**哪些 slot 要 mask、被丢的 token 在 unpermute 时怎么处理**。

## 1. 原理

按六步推导。

**第一步：permute 是什么。**

三个数组把 token 序映射到专家序：

```text
counts[e]      = tokens_per_expert，第 e 个专家收到多少个 (token, 专家) 对
offsets[e]     = counts 的排他前缀和，第 e 段在 permuted 缓冲区里的起始行
dst_row[t, j]  = offsets[idx[t, j]] + (该 (t, j) 在专家内的排位)
```

于是 `Xp[dst_row[t,j]] = X[t]`。`make verify` §A2 断言了这条映射是**双射**：`T·top_k` 个 slot 号互不重复、且恰好覆盖 `0 .. T·top_k−1`。

排位由一次 `atomicAdd`（**计数结果确定，到达顺序不确定**）配合一次排他前缀和得到。前缀和的并行算法正文在 [reduce-scan/03](../reduce-scan/03-scan-algorithms.md)，此处只做代入；`make verify` §A4 断言它与 `numpy.cumsum` 逐元素相等。

**第二步：一行是连续的，散开的是行号。**

这是全篇的关键，值得单独拎出来：

```text
X[t] 这一行 = X[t·d_model .. t·d_model + d_model)，d_model 个**连续**元素
permute 把它整行搬到 Xp[dst_row[t,j]]
```

所以**行内没有任何散开**。把它和「逐元素 scatter」对比一下：如果按元素搬，那么每个元素的去向是 `dst_row·d_model + c`，`c` 从 0 变到 `d_model−1` 时地址连续——**仍然只有行号是散的**。

> **permute 是「行的排列」，不是「元素的排列」。** 把它当成后者写，就丢掉了全部合并的机会。

**第三步：字节账。**

按行搬运，一个 `(token, 专家)` 对搬 `d_model·esize` 字节。两种写法：

```text
甲（读一次，写 k 次）：token 行留在寄存器/SMEM，写进 k 个 slot
  读 = T·d_model·esize        写 = T·top_k·d_model·esize
  合计 = (1 + top_k)·T·d_model·esize = 100.663 MB

乙（读 k 次，写 k 次）：每个 slot 各自回源取一遍
  合计 = 2·top_k·T·d_model·esize = 134.218 MB（比甲多 33%）
```

（推导值，`make verify` §B6。）

unpermute 是它的镜像，同样能把 k 个贡献在寄存器里累加、每个 token 只写一行：

```text
读 = T·top_k·d_model·esize        写 = T·d_model·esize
合计 = (1 + top_k)·T·d_model·esize = 100.663 MB
```

（推导值，`make verify` §B6。）两者加起来 `2·(1+top_k)·T·d_model·esize = 201.327 MB`，**是同层专家权重（1.8790 GB，见 [01 篇 §1](01-moe-compute-pattern.md)）的 10.71%**。

**第四步：合并度账——两种 lane 映射差 16 倍。**

字节数相同不代表代价相同。看一个 warp 的一次访存落在几个 32 字节 sector 上（sector 是 L1 与 DRAM 之间的最小事务粒度，机制正文在 [memory-system/02](../../20-gpu-architecture/memory-system/02-coalescing-and-cache-line.md)，此处回链）：

```text
甲 lane 沿 d_model 走：一个 warp 的 32 个 lane 覆盖 32×2 B = 64 B 连续地址
                      → 2 个 sector，有用字节占比 100%

乙 lane 沿 slot 走：  一个 warp 的 32 个 lane 各写一行的同一个列，
                      地址相距 d_model·esize = 8192 B
                      → 32 个 sector，有用字节占比 64 / 1024 = 6.25%
```

（推导值，`make verify` §B6 与 §C3。）**16 倍**的放大，纯粹来自「lane 沿哪个方向走」。

注意这个账的一个特点：**两种映射写出的内容逐元素完全相同**，所以任何正确性测试都测不出来——只有访存计数器（或者一个 profiler）能区分。`code/03_permute_unpermute.cu` 里两个版本的 permute 内核就是这样并列摆着的。

**第五步：缓冲区放得下吗。**

permuted 缓冲区的大小是 `T·top_k·d_model·esize = 67.1 MB`。

H100 的 L2 是 50 MB（引用值，见 `code/specs.py` 的口径注记：**本仓库对 L2 容量这一项内部不一致**，此处只做量级判断）。`67.1 / 50 = 1.28`——**装不下**（推导值，`make verify` §B6）。所以 permute 写出去的每一行、unpermute 读回来的每一行，都得从 HBM 走一遍。这也是第三步那 201.327 MB 必须按 HBM 流量记账、而不是按「反正会被 L2 接住」记账的原因。

**第六步：unpermute 的逆映射带一个 mask。**

`out[t] = Σ_j w[t,j] · Yp[dst_row[t,j]]`。如果 `(t, j)` 这一对被容量截断了，它**仍然有 `dst_row`**（缓冲区里那个位置仍然存在、可能还留着上一次迭代的数据），只是**不该被累加**。所以 `dst_row` 与 `valid` 是**两个数组**，不是一个：

```text
dst_row[t, j]  = 真实位置（offsets[e] + 排位）——**无效也记**
valid[t, j]    = 是否参与本次输出
```

把它们合成一个「无效就填 −1」的数组是个常见的偷懒写法，代价是：unpermute 没有任何办法表达「读这个位置但不用它」和「这个位置根本不存在」的区别。`make verify` §C2 就是在这个结构上跑出「不做 mask」的后果的。

## 2. 执行映射

### 2.1 三个 kernel 的分工

| kernel | 干什么 | 关键点 |
|---|---|---|
| `histogram_kernel` | `counts[idx[i]] += 1` | `atomicAdd` 到全局计数数组；**结果确定、顺序不定** |
| `scan_offsets_kernel` | 把 `counts` 变成 `offsets` | 单 block 内 `__syncthreads()` 同步的扫描；`E` 小的时候一个 block 够 |
| `build_slot_map_kernel` | 给每个 `(t, j)` 分配 `dst_row` 与 `valid` | `cursor[e]` 是 `offsets` 的副本，原地累加 |
| `permute_rows_kernel` | 按 `dst_row` 搬行 | **`threadIdx.x` 沿 `d_model` 走** |
| `unpermute_kernel` | 按权重回收 + mask | k 个贡献在寄存器里累加，每 token 只写一行 |

`atomicAdd` 那一栏值得说透：**它的计数结果是确定的**（`counts[e]` 与执行顺序无关），但**哪个 `(t, j)` 拿到哪个 slot 号是不确定的**。这不是 bug，但它让中间缓冲区不可复现——见 §4.4。

### 2.2 线程映射：把 `d_model` 放在 `threadIdx.x` 上

`permute_rows_kernel` 的写法只有一行是关键：

```cuda
// 一个 block 搬一行；lane 沿 d_model 走
const int s = blockIdx.x;                       // 第 s 个 slot
const float* src = X  + (size_t)src_of[s] * d;  // 源行：任意位置
float*       dst = Xp + (size_t)s * d;          // 目标行：连续
for (int c = threadIdx.x; c < d; c += blockDim.x) dst[c] = src[c];
```

`src_of[s]` 决定了源行在哪、`s` 决定了目标行在哪——**两个行号都是「随便」的**。但只要 `threadIdx.x` 沿 `c` 走，同一个 warp 的 32 个 lane 就在**同一行内**相邻，读写都落在 64 B（FP16）的连续地址上。

对照 §1 第四步的反例：把 `threadIdx.x` 换成 slot 号、把 `d_model` 放进循环里，写法同样自然、结果同样正确，访存却是 16 倍。**这两段代码的差别不在算法，只在线程映射——而它测不出来。**

### 2.3 unpermute 里的 mask

```cuda
template <int K, int MASK>
__global__ void unpermute_kernel(..., const int* dst_row,
                                 const unsigned char* valid, ...) {
    const int t = blockIdx.x;                   // 一个 block 回收一个 token
    for (int c = threadIdx.x; c < d; c += blockDim.x) {
        float acc = 0.0f;
        for (int j = 0; j < K; ++j) {
            const int i = t * K + j;
            if (MASK && !valid[i]) continue;     // ← 全部差别就在这一行
            acc += w[i] * Yp[(size_t)dst_row[i] * d + c];
        }
        out[(size_t)t * d + c] = acc;
    }
}
```

`MASK` 做成模板参数（而不是运行时分支），是为了让「做」与「不做」是两个真正的内核、而不是一个内核里的一个 `if`。**把 mask 放进分支里而不放进模板里，是让这个 bug 活下来的最常见方式**：性能测试时它被跳过了，正确性测试时它又恰好总是真。

### 2.4 被丢的 token：三种处理，kernel 侧都要能表达

容量截断之后，一个 `(t, j)` 对的状态有三种，而不是两种：

1. **有效**：`valid = 1`，正常累加；
2. **无效，但同 token 还有别的有效选择**：跳过这一项，其余照算——输出是 `X[t] × 幸存权重之和`（见 §4.1）；
3. **无效，且该 token 的 k 个选择全被截断**：前向输出恒为 0，**反向梯度也恒为 0**（`make verify` §C6：8 个 token 里有 2 个全丢，它们的 `‖y‖` 与 `max|∂L/∂y|` 都**恰好**是 0）。

第 3 种是「这个 token 在本步训练里完全不存在」。它不是一个数值问题，是一个**语义**问题：丢掉之后要不要把幸存权重重新归一化、要不要给这类 token 补一条旁路，属于架构层的决定（正文在 `80-model-architectures/moe`，此处回链）；**kernel 侧的责任是让这个决定成为可能**——先得有 `valid` 数组，才谈得上怎么用它。

### 2.5 跨设备：permute 之后还有一步

上面全是**单卡内的本地重排**。一旦专家分布在不同设备上，permute 之后还要一次 all-to-all——**它的通信量分析不在本域**，见 `110-communication/collectives`（正文在彼，此处回链）。本文与 [01](01-moe-compute-pattern.md)、[02](02-grouped-gemm.md) 都只讲 kernel 侧的本地重排。

## 3. 实现与演进

**甲：按元素 scatter。**

最直觉的写法：每个线程负责一个 `(token, 列)` 元素，算出目标行号，写一个元素。`make verify` §C3 算过这条路：一个 warp 的写请求触及 **32 个 sector**，有用字节占比 **6.25%**。它在大 `d_model` 下几乎是纯粹的浪费——因为**行内的连续性被 lane 映射丢掉了**。

**乙：按行搬运（本文的做法）。**

把 `threadIdx.x` 让给 `d_model`，行号交给 `blockIdx` / `threadIdx.y`。字节数不变，sector 数从 32 降到 2。这是当前实现的主流形态，也是 `code/03_permute_unpermute.cu` 的 `permute_rows_kernel`。

**丙：让 slot 表信息更少。**

`dst_row` 是 `[T, top_k]` 的 int 数组，读写它本身也是一笔流量（本场景 `8192 × 4 B = 32 KB`，占 permuted 缓冲区的 0.04883%——**小到可忽略**，但它是一次额外的随机读，`T` 很大时要注意；推导值，`make verify` §B6）。进一步的做法是**不存 `dst_row`，改用「按专家分块」的网格**：`blockIdx.x` 直接编码「专家 + 该专家内的行号」，`src_of` 查表一次即可。这样每个 block 处理的段是**连续**的，只是源行散——而源行的读本来就已经合并了。

**丁：与 grouped GEMM 融合。**

permute、grouped GEMM、unpermute 之间是三个全局同步点。`unpermute` 恰好是 grouped GEMM 的 epilogue 能顺手做的事（加权 + 回填），省掉一整次 `T·top_k·d_model·esize` 的往返。代价是 epilogue 要持有「这个 slot 属于哪个 token、权重是多少」的信息——也就是把 `dst_row` 从全局内存搬进 GEMM 内核的 SMEM。这是当前 MoE 内核优化的主要方向之一。

## 4. 踩坑与失败模式

### 4.1 unpermute 漏了 mask：被丢 slot 里的残留数据被当成了激活

**现象**：容量截断上线之后，验证集指标看起来还行（略微变差），但个别的输出**整行是错的**，而且错得毫无规律。

**成因**：permute 时对无效 slot **不写**，于是 `Xp` 的那个位置保留着**上一次迭代的数据**（缓冲区复用）或者别人的行。unpermute 不做 mask 就把这份残留按权重加了进去。`make verify` §C2 在 `tokens=64, d_model=8, d_ff=16, E=4, top_k=2, C=40` 上跑出来：

```text
tokens_per_expert = [37, 64, 27, 0]        被丢的 slot = 24 / 128（18.8%）
做 mask  （只累加 valid 的 slot）：最大误差 = 0.000e+00
不做 mask（把所有 slot 都累加）  ：最大误差 = 3.490e+00（参考本身的量级 5.748e+00）
受影响的 token = 24 / 64 行
```

（推导值，`make verify` §C2。）**误差是整行的量级，不是数值噪声**；受影响的行数恰好等于「有 slot 被丢的 token 数」。而且它**不会报错、不会 NaN**——只是悄悄错。

**规避**：把 `valid` 做成独立的数组（§1 第六步），让「读这个位置」与「用这个位置」成为两件事。**并注意做 mask 之后输出并不是 `X[t]` 本身**：它是 `X[t] × 幸存权重之和`（本场景用 §C2 的构造可以验证这一点）。要不要重新归一化是架构层的选择；kernel 侧只要保证这个选择是可表达的。

### 4.2 反例 kernel：scatter 写导致写不合并

**现象**：内核完全正确，逐元素对比一个数都不差，但耗时是「同一个算法」的另一版的两三倍。profiler 里 `l1tex__t_sectors` 高得离谱。

**成因**：线程映射反了——`threadIdx.x` 沿 slot 走、`d_model` 进了循环。`make verify` §C3 把这个代价算了出来：

```text
甲 lane 沿 d_model 走：  有用 64 B 落在 2 个 sector 里 → 搬走 64 B，占比 100%
乙 lane 沿 slot 走：    32 个地址相距 8192 B，各占一个 sector → 搬走 1024 B，占比 6.25%
放大 = 16×
```

（推导值，`make verify` §C3。）

**规避**：先问「这个 warp 的 32 个 lane 覆盖的是哪一段地址」。**判据是「一个 warp 内这条 `STG` 有几个不同的 128 字节段」，而不是「有几个不同的地址」**——permute 里目标行号千差万别，但行内地址连续，所以答案仍然是 1。合并的机制正文在 [memory-system/02](../../20-gpu-architecture/memory-system/02-coalescing-and-cache-line.md)，此处回链。

### 4.3 反向漏了 mask：给没发生的计算发梯度

**现象**：前向的输出是对的（输出侧验证不出问题），训练却收敛得比预期差；把容量开大就好了——于是问题被「调参」掩盖过去。

**成因**：前向做了 mask、反向没做，是最容易漏的一种**不对称**。反向会为每一个 `(t, j)` 对重算激活并累加梯度，包括那些前向**根本没算过**的对子。

但这里有一个反直觉的细节，值得单独记一笔：**这个误差不一定大**。`make verify` §C6 把每个被丢对子的梯度项打了出来：

```text
token 3 的第 0 选择：|梯度项| = 0.0034，权重 w = 0.9989
token 4 的第 1 选择：|梯度项| = 0.1854，权重 w = 0.0264
token 5 的第 1 选择：|梯度项| = 0.6764，权重 w = 0.0695
                    ...
||Δg|| / ||g|| = 0.0121（1.21%）
```

（推导值，`make verify` §C6。）**被挤掉的对子恰好也是权重小的那些**——热点专家的 softmax 权重接近 0.999，被它挤下去的那一项就只剩 0.001 的量级。所以「反向漏 mask 的误差很大」这句话**不成立**；正确的说法是「误差量级由权重分布决定，本例 1.21%」。

真正**结构性**的后果是另一类：**`k` 个选择全部被截断的 token**。`make verify` §C6 的例子里 8 个 token 中有 2 个（`[6, 7]`）**全部**被丢，它们的 `‖y‖` 与 `max|∂L/∂y|` 都**恰好是 0**。这些 token 在本步训练中完全不存在——**连一个把它们拉回来的梯度都没有**。

**规避**：mask 必须在**前向与反向的同一个位置**生效。实现上最稳的做法是让反向读同一份 `valid` 数组，而不是「前向丢掉之后反向也不会碰」——后者依赖两个 kernel 的网格恰好一致，一改就断。

### 4.4 slot 分配顺序不确定，导致中间缓冲区不可复现

**现象**：同一个输入跑两次，**最终输出**一致（浮点求和顺序的差异在最后一位），但**中间的 permuted 缓冲区**逐字节不同；基于它做的回归测试或差分调试因此失效。

**成因**：slot 号来自 `cursor[e]` 的一次 `atomicAdd`——**计数结果确定，但「哪个 SM 先到」不确定**，所以同一个专家内部哪个 `(token, j)` 拿到更小的 slot 号是不确定的。这不是精度问题，是**并行归约的顺序不确定性**在重排这一步的具体形态（同类问题见 [reduce-scan/04](../reduce-scan/04-multi-block-and-determinism.md)）。

**规避**：把「不确定」限制在**不参与输出**的地方。两条路：(a) 确定性的 slot 分配——按 `(专家, token 序号)` 排序而不是按到达顺序，代价是排序本身；(b) 承认缓冲区顺序不定，但保证 **unpermute 的求和顺序与 slot 顺序无关**（加权和本身是可交换的），让不确定性止于中间缓冲区。选 (b) 时要明白：**中间缓冲区不能再被当成「可复现的中间态」**，任何以它为基准的回归测试都不可靠。

## 编译与运行

配套程序是 **`code/03_permute_unpermute.cu`**（直方图 → 前缀和 → slot 映射 → 合并版/反例版 permute → 做 mask / 不做 mask 的 unpermute，外加访存字节与 sector 账）：

```bash
cd code
make 03_permute_unpermute
./03_permute_unpermute
```

程序会把两条 permute 路径都跑一遍、把 unpermute 的两个模板实例都跑一遍，与 fp64 参考逐元素比，然后打印 §1 的字节账与 §2 的 sector 账。**注意其中两类数字的性质不同**：

- 字节数、sector 数、放大倍数、占权重流量的比例——都是**推导值**（纯算术），可用 `make verify` 的 §B6 / §C3 复核；
- `max|Δ|` 那几行是 **kernel 的运行结果**，本仓库没有 GPU，**没有跑过**，不对其数值负责。

本机（macOS，无 CUDA 工具链）只能跑：

```bash
make verify
```

对应本文的是 §A2（permute→GEMM→unpermute 端到端与稠密展开一致、permute 是双射）、§A4（偏移的前缀和）、§B6（字节与 sector 账）、§C2（不做 mask 的后果）、§C3（元素级 scatter 的 sector 放大）、§C6（反向 mask 与整 token 被丢）。

## 6. 小结与延伸

本文一条主线：**permute/unpermute 是行级的 scatter/gather，`d_model` 那一维永远连续——所以合并度完全由「lane 沿哪个方向走」决定，而这个选择在正确性上完全不可见。** 围绕它有三个可独立复用的中间结论：

1. **一行 8 KB，散开的是行号。** 把 `threadIdx.x` 让给 `d_model`、行号交给 `blockIdx`，一个 warp 的访存从 32 个 sector 降到 2 个（FP16）——**16 倍**，而两种写法的输出逐元素相同。
2. **这一步不贵，但不能乱写。** permute + unpermute 的流量是 `2·(1+top_k)·T·d_model·esize = 201.327 MB`，占同层专家权重的 **10.71%**；permuted 缓冲区 `67.1 MB` 装不进 50 MB 的 L2，所以这笔流量要老老实实按 HBM 记。
3. **`dst_row` 与 `valid` 必须是两个数组。** 「位置存在但这次不用」和「位置根本不存在」是两件事；合成一个数组就没有办法表达 mask，而 mask 漏在前向是「读了不该读的」（误差整行量级），漏在反向是「写了不该写的」，两边都做对时还有第三类 token——**k 个选择全被丢，前向 0、反向 0，在本步训练里完全消失**。

三篇的分工到此完整了：[01](01-moe-compute-pattern.md) 算清了 MoE 一层值多少 FLOPs、瓶颈在哪一栏；[02](02-grouped-gemm.md) 解决了「M 是分段的」这件事怎么排 tile；本文解决了「这些段怎么攒起来、算完怎么还回去」。三者合起来就是一次完整的 MoE 前向：`routing → permute → grouped GEMM → unpermute`。

## 参考

- NVIDIA, *CUDA C++ Best Practices Guide* —— 「Coalesced Access to Global Memory」：32 字节 sector 与合并访存的判据。
- NVIDIA, *CUDA C++ Programming Guide* —— §5.1「Compute Capabilities」；`atomicAdd` 的内存序与原子性说明。
- 本仓库 [20-gpu-architecture/memory-system/02-coalescing-and-cache-line](../../20-gpu-architecture/memory-system/02-coalescing-and-cache-line.md) —— 合并与 cache line 的机制（本文 §1 第四步、§4.2 只做代入）。
- 本仓库 [20-gpu-architecture/memory-system/03-shared-memory-bank-conflict](../../20-gpu-architecture/memory-system/03-shared-memory-bank-conflict.md) —— SMEM 行距与 bank 冲突。
- 本仓库 [40-kernels/reduce-scan/01-reduction-algorithms](../reduce-scan/01-reduction-algorithms.md) —— segmented 归约的正文；`tokens_per_expert` 是它的一个实例。
- 本仓库 [40-kernels/reduce-scan/03-scan-algorithms](../reduce-scan/03-scan-algorithms.md) —— 排他前缀和的并行算法（本文 §1 第一步只做代入）。
- 本仓库 [40-kernels/reduce-scan/04-multi-block-and-determinism](../reduce-scan/04-multi-block-and-determinism.md) —— 并行归约的顺序不确定性与可复现性（本文 §4.4）。
- 本仓库 [40-kernels/moe-kernels/02-grouped-gemm](02-grouped-gemm.md) —— 本文搬出来的那些行，下一步是被谁消费的。
- 本仓库 `code/verify.py` 的 §A2、§A4、§B6、§C2、§C3、§C6 —— 本文全部数字的来源。

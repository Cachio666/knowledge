---
title: attention 的计算形态与两个瓶颈
slug: attention-as-a-kernel
domain: 40-kernels
topic: attention
level: 3
difficulty: intermediate
prerequisites:
  - 10-foundations/performance-models/01-roofline-model
  - 20-gpu-architecture/tensor-core/01-mma-primitive
tags: [attention, arithmetic-intensity, roofline, memory-bound, softmax, intermediate-matrix]
related:
  - 40-kernels/gemm/01-naive-gemm-and-roofline
  - 40-kernels/softmax-normalization/01-softmax-stability
  - 40-kernels/attention/02-flash-attention
  - 20-gpu-architecture/memory-system/01-hierarchy-overview
code:
  - code/01_naive_attention.cu
status: stable
updated: 2026-09-16
---

# attention 的计算形态与两个瓶颈

> 一句话：attention 是「两段矩阵乘夹一个 softmax」，FLOPs 是 `4S²d`、访存是 `O(S²)`。
> 朴素实现把中间矩阵 S 物化到 HBM，于是被钉死在 memory-bound；更关键的是它的算术强度
> **有上界 `d`**，永远越不过 roofline 拐点——所以真正的出路不是「提高 AI 的比值」
> （那条路最多给 2.31×），而是把**绝对字节数**从 `O(S²)` 降到 `O(S·d)`。

## 0. 背景与动机

attention 的全部计算是两段矩阵乘夹一个逐行 softmax：

```text
S = softmax(Q·Kᵀ/√d)          （S 是 S×S 的中间矩阵，S 这里指序列长度）
O = S·V
```

朴素实现把它直译成「先算出整个 S，存下来，再乘 V」。**代价的具体对照场景**：
序列长度 16384、head 维度 128、32 个 head、batch 8、fp16。

- **S 本身要占多少显存**：`S²·esize = 16384² × 2 B = 536.9 MB` —— 这是**单 head 单 batch**。
  乘上 32 个 head 是 17.18 GB，再乘 8 个 batch 是 **137.44 GB**，
  是 H100 单卡 80 GB 的 **1.72 倍**（推导值，`make verify` §B3）。
  也就是说：**这一档配置下不是「慢」的问题，是根本放不下**。
- **放得下的那一档也一样亏**：序列长度 4096 时单 head 单 batch 的 S 只有 33.6 MB，
  可放得下。但朴素的 HBM 流量是 71.3 MB，flash（S 不落 HBM）是 4.19 MB —— **17 倍**。
- **账算下来最刺眼的一条**：序列长度 16384、单 head 单 batch，朴素**光是 S 的写 + 读**
  就要 `2S²·esize = 1.074 GB → 320.5 µs`，而整个前向的全部矩阵乘只需要
  `4S²d = 1.374e11 FLOP → 138.9 µs`。**搬运中间结果的代价是全部计算的 2.308 倍**
  （推导值，`make verify` §B2）。这条比值**与 S 无关**：S=1024 时是 0.5 µs 对 1.3 µs，
  S=65536 时是 2222.4 µs 对 5128.3 µs，都是 2.308。

所以有两个瓶颈，而且它们是**两个不同的问题**，本篇要做的第一件事就是不让它们混在一起：

| 瓶颈 | 一句话 | 出路 |
|---|---|---|
| ① 绝对字节数 | 中间矩阵是 `O(S²)`，写 + 读都要过 HBM | 让 S 不落 HBM（02 篇） |
| ② 算术强度有上界 | 朴素 AI 渐近于 `d`（= 128），`I*`(FP16) = 295.4，越不过 | 同上；但**收益的量级完全不同** |

第 ② 条的出路和第 ① 条是同一个动作，可**收益不是一回事**：AI 的比值最多给出 2.31× 的
性能上界（而且这个上界与 S 无关），字节数的降幅却随 S 线性增长（S=16384 时 65 倍，
S=65536 时 257 倍）。§1 会把这两个数分开推。

## 1. 原理

### 1.1 计算量：`4S²d`

设 Q 是 `S×d`、K/V 是 `S×d`、输出 O 是 `S×d`。两段矩阵乘各贡献 `2S²d`：

```text
QKᵀ : (S×d)·(d×S) → S×S    乘加次数 = S·S·d，每次乘加 = 2 FLOP ⟹ 2S²d
PV  : (S×S)·(S×d) → S×d    乘加次数 = S·S·d        ⟹ 2S²d
                                                   合计 4S²d
```

中间那个 softmax 本身的算术量也是 `O(S²)`（每行一次 max、`S` 次 `exp`、一次求和），
但它走的是 **FP32 向量通路与超越函数**，不是 tensor core 的 FMA。
本仓库没有超越函数吞吐的可靠规格（见 `code/specs.py` 的 `UNSOURCED`），
**所以本文的算力账一律只算 `4S²d`**，并把 softmax 那一步记在「带宽受限、多算不花钱」这一侧——
这个判断本身由下面的 AI 给出。

### 1.2 两套口径的访存

**口径必须写在每个数字旁边**，因为本文的结论对口径敏感。统一取 `d = 128`、fp16、`esize = 2 B`：

| 写法 | 字节（单 head 单 batch） | 逐项 |
|---|---|---|
| 朴素（两遍、物化 S） | `(4·S·d + 2·S²)·esize` | 读 Q/K/V（`3Sd`）+ 写 O（`Sd`）+ 写 S（`S²`）+ 读 S（`S²`） |
| flash（S 不落 HBM） | `4·S·d·esize` | 读 Q/K/V（`3Sd`）+ 写 O（`Sd`） |

两者的差额**恰好是那 `2S²·esize`**——S 的写与读。下面这张表就是全文的起点：

| S | 朴素字节 | 朴素 AI | flash 字节 | flash AI | 字节比 |
|---|---|---|---|---|---|
| 1024 | 5.2 MB | 102.4 | 1.05 MB | 512.0 | 5× |
| 4096 | 71.3 MB | 120.5 | 4.19 MB | 2048.0 | 17× |
| 16384 | 1090.5 MB | 126.0 | 16.78 MB | 8192.0 | 65× |
| 65536 | 17247.0 MB | 127.5 | 67.11 MB | 32768.0 | 257× |

（全部为推导值，`make verify` §B1 逐组打印并与本表逐项核对。
注意 flash 那一列的 AI `= S/esize = S/2` 是可以随便涨的——**它是被算力顶住的，不是被带宽**。）

### 1.3 渐近值：`d`，以及口径怎么改变它

朴素这一列的 AI 不随 S 发散，值得把极限算出来：

```text
AI_朴素(S) = 4S²d / ((4Sd + 2S²)·esize)

分子分母同除以 2S²：  = 2d / (esize·(1 + 2d/S))

S → ∞ 时 (1 + 2d/S) → 1：  AI_朴素 → 2d/esize
```

代入 `d = 128`、`esize = 2`，**`2d/esize = d = 128`**。（fp16 下 `2d/esize` 恰好等于 `d`，
这是 esize=2 的巧合，不是普遍规律——换成 fp32（`esize=4`）时它是 64。）

**这一步就是口径的入口**：分子里的 `4S²d` 与分母里的 `2S²` 是同一个阶，
所以 S 被约掉了，极限值只剩 `d`。如果把口径改成「S 只算写回、不算读回」，
分母变成 `S²·esize`，极限就变成

```text
AI → 4S²d / (S²·esize) = 4d/esize = 2d = 256
```

**同一个核，渐近值从 128 变成 256。** 这不是吹毛求疵：§4.1 会把这条口径错误的后果算给你看
（字节比从 65.00× 变成 33.00×）。`make verify` §B1 把两条渐近线都打印了出来。

### 1.4 与 roofline 对照：朴素永远越不过拐点

`make verify` §B2 的两个规格值代入：

```text
I*(FP16 tensor, H100) = 989.5e12 ÷ 3350e9 = 295.4 FLOP/byte
I*(FP32 向量,  H100)  =  67.0e12 ÷ 3350e9 =  20.0 FLOP/byte
```

| S | 朴素 AI | 占 `I*` 比例 | 朴素可达 TFLOPS | 占峰值 | flash AI | flash 可达 |
|---|---|---|---|---|---|---|
| 1024 | 102.4 | 34.7% | 343.0 | 34.7% | 512.0 | 989.5（峰值） |
| 4096 | 120.5 | 40.8% | 403.6 | 40.8% | 2048.0 | 989.5 |
| 16384 | 126.0 | **42.7%** | 422.2 | 42.7% | 8192.0 | 989.5 |
| 65536 | 127.5 | 43.2% | 427.1 | 43.2% | 32768.0 | 989.5 |
| 渐近 | 128.0 | **43.3%** | 428.8 | 43.3% | ∞ | 989.5 |

**三行读数**（`make verify` §B2 逐条断言）：

1. 朴素 AI 到 S=16384 时只有 `I*` 的 **42.7%**，S 再大也就爬到 43.3%——
   **它确实是 memory-bound，而且 S 再大也救不了**。
2. flash 的 AI 是 `S/2`，S=1024 时就已经是 `I*` 的 1.73 倍——**大 S 下是 compute-bound**。
3. 于是「flash 提升的是 AI 的比值」这句话只对了一半，另一半是错的：见下。

### 1.5 分开讲：AI 之比有上界，字节比没有

把两列的「可达算力」相除：

```text
flash 可达 ÷ 朴素可达 = 峰值 ÷ (带宽 × d) = 989.5e12 ÷ (3350e9 × 128) = 2.31
```

**这个 2.31 与 S 无关**：因为朴素 AI 有上界 `d`，flash 那一侧早就顶到峰值了，
所以「靠提高 AI 比值」能拿到的性能上界是**封死的 2.31×**。

而同一组数里，字节数的降幅是：S=16384 时 65×、S=65536 时 257×——**随 S 线性增长**。

这就是本篇最想留下的一句话：

> **flash 的收益是「绝对字节数从 O(S²) 降到 O(S·d)」，不是「把 AI 的比值提高了」。
> 前者的上界与 S 无关（2.31×），后者随 S 增长。**

两个说法都「对」，但它们解释了不同的现象：只有后者能说明**为什么长序列的收益更大**。
S=16384 时的两笔账很直白（单 head 单 batch，推导值，`make verify` §B2）：

```text
前向的纯算力时间 = 4S²d  ÷ 989.5e12 = 138.9 µs
S 的写 + 读      = 2S²·esize ÷ 3350 = 320.5 µs     比值 2.308
```

那个 2.308 与上面的 2.31 是同一个数（都是「峰值 ÷ (带宽 × d)」）——**它们是同一件事的两种说法**。

## 2. 执行映射

### 一个 block 在干什么

朴素的 `code/01_naive_attention.cu` 里，一个 block 处理一行 query，三遍：

```text
① 算 S = scale·q·Kᵀ，**写回 HBM**（同时归约出行 max）
② 从 HBM **读回** S，算 exp(x − max) 与它们的和 l，把 exp **写回 HBM**
③ 从 HBM **再读回** exp，乘 V
```

三遍里有**两次**跨 HBM 的往返，这就是 §1.2 那 `2S²·esize` 的来源。

在硬件上，这三遍落在不同的通路上：

| 步骤 | 通路 | 具体组件 |
|---|---|---|
| ① QKᵀ | tensor core（`mma.sync` / `wgmma`） | 见 [tensor-core/01](../../20-gpu-architecture/tensor-core/01-mma-primitive.md) |
| ② softmax | FP32 向量 + 超越函数 | warp scheduler 发射 `exp`，走 SFU/MUFU |
| ③ PV | tensor core | 同上 |

FP32 向量那条通路的峰值是 67.0 TFLOPS（H100，非 tensor core），
`I*`(FP32 向量) 只有 **20.0** —— 也就是说 softmax 这一步**比矩阵乘更容易变成带宽受限**。
好在它每元素只做常数次操作，而 S 一旦被物化，`S²` 个元素的每一次触碰都要过 HBM。

### 数量级

取 S=16384、单 head 单 batch（推导值，`make verify` §B2）：

- 全部矩阵乘：`4S²d = 1.374e11 FLOP`，按 989.5 TFLOPS 折算是 **138.9 µs**；
- S 的写 + 读：`2S²·esize = 1.074 GB`，按 3350 GB/s 折算是 **320.5 µs**；
- 再乘上 32 head × 8 batch（256 份）：算力 35.6 ms、流量 **274.9 GB → 82.1 ms**。

参考量级（规格值）：一次 shared memory 访问 20~30 周期，一次片外访存往返 **400~800 周期**，
每 SM 每周期最多发射 4 条指令。softmax 这一步的每一次 `exp` 都是一条 MUFU 指令，
**即使全部命中 L2，也换不回那 320.5 µs**。

### 为什么 L2 帮不上忙

S 是**每个 head 私有**的：32 个 head 各自有一个 `S×S` 的矩阵，彼此没有复用。
单 head 的 S 在 S=16384 时是 536.9 MB，远超片上能留住的容量;
而且它的生命周期是「写完 → 立刻读一次 → 再读一次 → 丢掉」，
时间局部性只覆盖「同一行被连续处理」这一小段。

> 口径说明：这里按「S 的写与读各记一次 HBM 流量」记账。
> 若某一行刚写完就被读回、正好命中 L2，那次读可以不算 HBM——但那只对很小的 S 成立。
> 本文全篇用「读 + 写各一次」这条口径，并**在每一个 AI 旁边都写清用的是哪一条**。

## 3. 实现与演进

按「S 有没有落 HBM」这条线：

1. **朴素三遍（本文）**。S 落 HBM。AI 渐近 `d`，永远 memory-bound；显存是 `O(S²)`
   且要乘 heads × batch。它的价值是**基准**：后面每一步都要与它比。
2. **分块 + 在线 softmax（[02 篇](02-flash-attention.md)）**。S 在片上算出来、当场用掉、
   当场丢掉，AI 变成 `S/2`。代价是状态 `(m, l, O)` 必须一直活着，而 `O` 的大小与
   `Br·d` 成正比——**这正是 flash 换来的东西和它付出的东西**。
3. **反向（[03 篇](03-flash-attention-backward.md)）**。前向把 S 丢了，反向就得重算它。
   重算一次 S 多花 `2S²d` 的 FLOPs（前向的 0.5 倍），省下的是 P 的写 + 读
   = `2S²·esize` 的流量——重算便宜 4.62 倍（`make verify` §B5）。
4. **工程形态与变体（[04 篇](04-warp-specialization-and-variants.md)）**。
   把「搬数据」与「算」拆到不同的 warpgroup，让 wgmma 连续发射；
   变体（滑窗 / ALiBi / softcap）只改分块范围或 exp 的输入，不改结构。
5. **decode（[05 篇](05-decoding-attention.md)）**。每 token 只算一行 query，
   形态完全变了：AI `= 1/r`（与 S 和 d 都无关），瓶颈从「S 矩阵」变成「KV cache 的读取」。

每一步的必要性都能用同一把尺子量：**这一版的绝对字节数是多少，AI 是多少**。

## 4. 踩坑与失败模式

### 4.1 算术强度表的两列用了不一致的矩阵集合

**现象**：读起来完全合理的一张表，两列各自都对，合起来就错。本仓库写这篇时的第一版
就是这么错的，而**错误的形式不是算错数，是两列用了不同的「读什么、写什么」**。

`make verify` §B1 把这件事直接算了出来（S=16384，统一 `d=128`、`esize=2`）：

| 口径 | 朴素 AI | flash AI | 字节比 |
|---|---|---|---|
| 正经（朴素 `4Sd + 2S²`，flash `4Sd`） | 126.0 | 8192.0 | **65.00×** |
| 错法 A：朴素漏掉 S 的**读回** | 248.2 | 8192.0 | **33.00×** |
| 错法 B：flash 漏掉**写 O** | 126.0 | 10922.7 | **86.67×** |
| 错法 C：A + B 同时犯 | 248.2 | 10922.7 | **44.00×** |

同一组形状，结论从 65× 变成 33× 或 87×——而两者的**渐近 AI 也从 128 变成 256**
（错法 A 这一列）。

**成因**：AI 是「FLOP ÷ 字节」，它的值完全由「你把哪些矩阵算进了字节」决定。
S 矩阵的「读」在朴素那一侧是理所当然的（两遍法必须读回来），但在心算时很容易漏；
而 flash 那一侧「写 O」是最容易漏的一项，因为它不是「中间结果」。
两列各漏一项，比值的分子分母就同时变了。

**规避**：把两列的**逐项分解**写在同一张表里（本文 §1.2 的那张表），
并且在写每一个 AI 时都问一句「这份字节里装的是哪几个矩阵」。
`code/specs.py` 的命名约定（`HBM_BW_GBps_BY_GEN` 而不是 `HBM_BW`）是同一个思路：
**让口径出现在名字里，误用才会可见**。

### 4.2 把 S 的显存按「单 head 单 batch」记账

**现象**：「S=16384 时中间矩阵只有 536.9 MB，怎么会放不下？」
按字面核对会觉得这句话有理，于是整段论证的说服力反而被削弱。

**成因**：attention 的中间矩阵是 `heads × batch` 份的。真实的账是
`536.9 MB × 32 head × 8 batch = 137.44 GB`，是 H100 单卡 80 GB 的 **1.72 倍**
（推导值，`make verify` §B3）。S=65536 时更夸张：单 head 8.59 GB、
×256 之后 **2199.02 GB**。

**规避**：**报 S 的显存时一律带上 `× heads × batch`**，像这样：

```text
S=16384: 单 head·单 batch = 536.9 MB；×32 head = 17.18 GB；×32 head × 8 batch = 137.44 GB
```

注意这条错误与 §4.1 是同一类毛病的两个面：**都是漏掉了某一个维度上的倍数**。
一个是漏掉「读还是写」，一个是漏掉「有几份」。

## 编译与运行

配套程序是 **`code/01_naive_attention.cu`**。它把 S 完整物化到 global memory，
用三遍结构跑完 attention，并打印本文 §1 的访存账；另外给了一个 `mode` 开关做
**不减 max** 的对照（fp16 上溢，见 §C1）。

它需要 `nvcc` 与 GPU：

```bash
cd code
make 01_naive_attention
./01_naive_attention            # 默认 S=1024，d=128
./01_naive_attention 4096 128   # S=4096，d=128
```

程序分四段打印：访存账（与本文 §1.2 / §1.3 的表逐项对照，全部是表达式直接算出的推导值）、
与 CPU double 参考的逐元素最大差、不减 max 时的 inf/nan 计数、以及 §0 那两笔账。

**两类数字必须分开看**：

- 本文所有表格里的数、`2.31×`、`137.44 GB`、`320.5 µs` —— **推导值**，
  可用 `make verify` 的 §A1 / §B1 / §B2 / §B3 / §C1 复核；
- 各内核的逐元素差与耗时 —— **那台机器上的运行结果**。本仓库没有跑过，也不对其数值负责。

本机（macOS，无 CUDA 工具链）只能跑：

```bash
make verify
```

它对应本文的是 §A1（在线分块与朴素两遍的结果一致，四种分块的最大按范数误差 `7.951e-16`）、
§B1（算术强度表的四组逐组核对 + 两条渐近线 + 三种口径错法的对照）、
§B2（`I*` 与两列的占比、S=16384 的两笔账）、§B3（S 的显存 × heads × batch）、
§C1（fp16 不减 max 得到 inf/nan）。

## 6. 小结与延伸

全文一条主线：**attention 的瓶颈不是一个，是两个；而这两个瓶颈虽然被同一个动作解决，
收益的量级却完全不同。**

三条可独立复用的结论：

1. **FLOPs 是 `4S²d`，朴素字节是 `(4Sd + 2S²)·esize`**——差额就是 S 的写 + 读。
2. **朴素 AI 渐近于 `2d/esize`（fp16 下 = `d` = 128），与 S 无关**，
   所以它永远只到 `I*` 的 43% 上下，**S 再大也越不过拐点**。
   *这个渐近值强依赖口径*：把 S 的「读」去掉，它变成 `4d/esize` = 256。
3. **flash 的收益是绝对字节数从 `O(S²)` 降到 `O(S·d)`**，而不是「AI 的比值」——
   后者有封死的 2.31× 上界（`= 峰值 ÷ (带宽 × d)`），前者随 S 线性增长。
   长序列收益更大，讲的正是后者。

下一站：

- [02-flash-attention](02-flash-attention.md) —— 把 S 从 HBM 里拿掉。
  它的正文是**输出累加器 `O` 的重标定**；`(m, l)` 那两条递推的正文在
  [softmax-normalization/02](../softmax-normalization/02-online-softmax.md)，本文只回链。
- [03-flash-attention-backward](03-flash-attention-backward.md) —— 前向省下的，反向要还回来：
  S 不存就得重算，`dQ` 的累加方式决定结果能不能复现。
- [05-decoding-attention](05-decoding-attention.md) —— 推理侧的形态：
  每 token 只算一行 query，瓶颈从「S 矩阵」变成「KV cache 的读取字节数」。

## 参考

- Dao, Fu, Ermon, Rudra, Ré, *FlashAttention: Fast and Memory-Efficient Exact Attention
  with IO-Awareness*, NeurIPS 2022 —— 本文 §1 的访存口径（S 不落 HBM）与 §3 的演进路线均出自此文；
  论文原文的 AI 分析用的是同一套 IO 记账。
- Williams, Waterman, Patterson, *Roofline: An Insightful Visual Performance Model for
  Multicore Architectures*, CACM 2009 —— 算术强度与拐点的定义，本文 §1.4 的对照表用它。
- 本仓库 [10-foundations/performance-models/01-roofline-model](../../10-foundations/performance-models/01-roofline-model.md)
  —— `I*` 与「可达算力 `min(峰值, 带宽×AI)`」的推导，本文直接沿用其结论。
- 本仓库 [20-gpu-architecture/tensor-core/01-mma-primitive](../../20-gpu-architecture/tensor-core/01-mma-primitive.md)
  —— 两段矩阵乘能落到哪些形状上。
- 本仓库 [40-kernels/softmax-normalization/01-softmax-stability](../softmax-normalization/01-softmax-stability.md)
  —— 「softmax 是带宽受限算子（`I = 0.5 FLOP/byte`）」与四个 exp 阈值，本文 §C1 的溢出判据出自此处。
- 本仓库 `code/verify.py` 的 §A1、§B1、§B2、§B3、§C1 —— 本文全部表格的数字来源。
- 规格值出处见 `code/specs.py`（`HBM_BW_GBps_BY_GEN`、`TC_TFLOPS_DENSE_SPARSE`、
  `HBM_CAPACITY_GB_BY_GEN`、`SMEM_PER_SM_MAX_KB` 等，逐条带 `# 来源：`）。

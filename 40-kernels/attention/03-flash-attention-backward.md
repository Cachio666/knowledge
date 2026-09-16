---
title: flash attention 反向：重算 S 与 dQ 的累加
slug: flash-attention-backward
domain: 40-kernels
topic: attention
level: 4
difficulty: advanced
prerequisites:
  - 40-kernels/attention/02-flash-attention
  - 40-kernels/gemm/06-split-k-and-stream-k
tags: [flash-attention, backward, recompute, atomic-add, determinism, split-k, gradient]
related:
  - 40-kernels/softmax-normalization/02-online-softmax
  - 40-kernels/attention/01-attention-as-a-kernel
code:
  - code/03_flash_attention_backward.cu
status: stable
updated: 2026-09-16
---

# flash attention 反向：重算 S 与 dQ 的累加

> 一句话：前向把 `S` 丢掉了，反向就得**重算**它——多花 `2S²d` 的 FLOPs（前向的 0.5 倍），
> 换掉 `2S²·esize` 的流量，这笔交易便宜 **4.62 倍**。
> 而反向真正的难点不在重算，在两件很容易弄错的事上：
> **`D = rowsum(dO ∘ O)` 是整行的量**（按块算会安静地毁掉 `dQ` 与 `dK`，而 `dV` 纹丝不动），
> 以及**三个张量的切分维度不同**（`dQ` 按 Q 的行、`dK`/`dV` 按 K 的行）。

## 0. 背景与动机

[02 篇](02-flash-attention.md) 的收益建立在「`S` 不落 HBM」上。反向立刻面对一个后果：
**链式法则需要 `P = softmax(S)`，而 `P` 已经被丢掉了。**

三条路，只有一条能走：

| 做法 | 代价 | 判定 |
|---|---|---|
| 前向把 `P` 写回 HBM | 写 `S²·esize` + 反向读 `S²·esize`（单 head 单 batch，S=16384 时 **1.074 GB**，折合 320.5 µs） | 把 02 篇省下的又还回去了 |
| 前向把 `P` 留在片上 | `S²` 个 fp32 —— 不可能，比寄存器文件大几个数量级 | 直接排除 |
| **反向重算 `S`** | 多算一次 `QKᵀ`：`2S²d` 的 FLOPs（69.4 µs） | **便宜 4.62 倍** |

具体场景：S=16384、d=128、单 head 单 batch（推导值，`make verify` §B5）：

```text
前向 FLOPs            = 4·S²·d = 1.374e+11
反向的四个矩阵乘       = 8·S²·d = 2.749e+11   （= 前向的 2 倍）
重算 S 的额外开销      = 2·S²·d = 6.872e+10   （= 前向的 0.5 倍）
⟹ 反向总 FLOPs ≈ 10·S²·d = 3.436e+11，是前向的 2.5 倍

省下的：P 的写 + 读 = 2·S²·esize = 1.074 GB
  重算 = 6.872e+10 ÷ 989.5e12 =  69.4 µs
  搬运 = 1.074 GB  ÷ 3350 GB/s = 320.5 µs      ⟹ 重算便宜 4.62 倍
```

乘上 32 个 head：重算多花 **2.20 TFLOP**，搬运要 **34.36 GB**。
**重算把「时间」换成了「算力」——而算力恰好是 flash 前向省下来的那一侧。**

但重算不是本文最难的部分。真正的坑有两个，都是本仓库在写这篇时踩到的：

1. 反向要的 `D = rowsum(dP ∘ P)`（等价于 `rowsum(dO ∘ O)`，§1.2）
   是**整行**的和，而分块之后每一块只看得到自己那一段；
2. `dQ`、`dK`、`dV` 三个张量**按不同的维度切分**，于是「谁需要原子累加」的答案对三者不一样。

## 1. 原理

### 1.1 链式法则：`D` 是从哪来的

记号：`S = scale·QKᵀ`（`S×S`），`P = softmax_rows(S)`（行和 1），`O = P·V`。
给定上游梯度 `dO`（`S×d`），要求 `dQ`、`dK`、`dV`。

**第一步：`dV` 直接从 `O = P·V` 来。**

```text
O = P·V  ⟹  dV = Pᵀ·dO
```

**第二步：`dP` 也从定义来。**

```text
dP = dO·Vᵀ
```

**第三步：`dS` 需要 softmax 的雅可比。** 逐行看，`p = softmax(s)`，
其雅可比是 `∂p_i/∂s_j = p_i(δ_ij − p_j)`。于是

```text
dS_j = Σ_i dP_i · p_i(δ_ij − p_j)
     = dP_j·p_j − p_j · Σ_i dP_i·p_i
     = p_j · (dP_j − D),        D ≡ Σ_i dP_i·p_i
```

**第四步：`dQ`、`dK` 由两段矩阵乘反向。**

```text
S = scale·QKᵀ  ⟹  dQ = (dS·scale)·K,   dK = (dS·scale)ᵀ·Q
```

**`D` 是这里唯一的非局部量**：它是对**整行** `S` 求和的结果。分块之后，
每一块只有自己那一段的部分和——§4.2 会说明按块用它有多致命。

### 1.2 关键一步：`D` 不需要 `P` 就能算出来

如果 `D` 必须用 `P` 算，那「重算 S」就白搭了（`D` 仍然要求全部 `P`）。
好在它有一个等价的算法：

```text
D = Σ_j dP_j · p_j
  = Σ_j (Σ_c dO_c·v_jc) · p_j                   （代入 dP = dO·Vᵀ）
  = Σ_c dO_c · (Σ_j p_j·v_jc)                   （交换求和次序）
  = Σ_c dO_c · O_c                              （括号里正是 O 的定义）
```

**于是 `D = rowsum(dO ∘ O)`——只需要前向的输出 `O`，不需要 `P`。**

这一步是 flash 反向能成立的**前提**，而不是一个优化：
正因为 `D` 可以不依赖 `P` 算出来，`dS = P∘(dP − D)` 里所有的非局部依赖都消失了，
于是「丢掉 `P`、分块重算」才成为可能。

> 顺带解释了一个常见疑问：「前向要存 `O` 才能反向」——对，但 `O` 是 `S×d`，
> 比 `P` 的 `S×S` 小一整个序列长度倍。**这正是 02 篇那笔账的闭环。**

### 1.3 重算的代价（把 4.62 倍推出来）

重算一次 `S` 要再做一遍 `QKᵀ`，即 `2S²d` 的 FLOPs，走 tensor core：

```text
t_recompute = 2S²d / 峰值 = 2·16384²·128 / 989.5e12 = 69.4 µs
```

而不重算就要把 `P` 搬两趟 HBM（前向写、反向读）：

```text
t_traffic = 2S²·esize / 带宽 = 2·16384²·2 / 3350e9 = 320.5 µs
```

比值 `320.5/69.4 = 4.62`。**把两者相除可以看清它依赖什么**：

```text
t_traffic / t_recompute = (2S²·esize/BW) / (2S²d/峰值) = (esize/d)·(峰值/BW)
                        = (esize/d)·I* = (2/128)·295.4 = 4.62
```

即：**这个比值与 S 无关**，只取决于「`esize/d`」与 roofline 拐点 `I*`。
`d` 越大（head 维度越宽）、`esize` 越小（精度越低），重算越划算——
这是一个可以直接搬到别的算子上的结论。

**别忘了重算的隐形成本**：重算还要再做一遍 `exp`（`S²` 次超越函数）。
它不占 tensor core，走 FP32 向量通路；本仓库没有超越函数吞吐的可靠规格
（见 `code/specs.py` 的 `UNSOURCED`），**所以本文不做这一步的算术**，只点名它的存在。

### 1.4 三个张量的切分维度不同

这是反向实现最容易绕晕的地方。看三个梯度的定义式：

```text
dQ = (dS·scale)·K      每一行只依赖一个 Q 行
dK = (dS·scale)ᵀ·Q     每一行只依赖一个 K 行
dV = Pᵀ·dO            每一行只依赖一个 K 行
```

于是：

| 梯度 | 切分维度 | 一个 block 循环另一维时 | 原子累加落在谁头上 |
|---|---|---|---|
| `dQ` | 按 **Q 的行** | 循环全部 KV 块，块内累加 | **无**（自己的行只写一次） |
| `dK`/`dV` | 按 **K 的行** | 跨全部 Q 块累加 | **有**（多个 Q 块写同一行） |

两种切法**不可能同时省掉原子**。于是实现上有两个 kernel：

- **`dQ` kernel**：按 Q 的行切，一个 block 循环全部 KV 块 → `dQ` 无原子，
  `dK`/`dV` 用 `atomicAdd` 汇总。这是 flash-attention 官方实现的形态。
- **`dK`/`dV` kernel**：按 K 的行切，一个 block 循环全部 Q 块 → `dK`/`dV` 无原子，
  但它需要把 `dQ` 也重算一遍（或者接受一个额外的原子）。

`code/03_flash_attention_backward.cu` 里三条路径都给了：
`backward_seq_dq_kernel`（按 Q 行切、`dQ` 顺序累加）、
`backward_atomic_dq_kernel`（按 KV 段切、`dQ` 用原子）、
`backward_workspace_kernel` + `dq_reduce_kernel`（按 KV 段切、`dQ` 写 workspace 再固定顺序归约）。

## 2. 执行映射

### 三个 kernel 的分工

```text
kernel 1（按 Q 行切）：grid = ceil(S/Br)
    每个 block：循环全部 KV 块，逐块重算 S、算 P、算 dS
        → dQ 在自己的 Br 行上顺序累加（寄存器里）
        → dK/dV 的贡献 atomicAdd 到全局

kernel 2（按 K 行切，可选）：grid = ceil(S/Bkc)
    每个 block：循环全部 Q 块，同样重算
        → dK/dV 在自己的 Bkc 行上顺序累加（无原子）
        → dQ 需要另想办法（原子或 workspace）
```

### 重算块的资源

一个 `Br×Bc` 的 tile 在反向要多留三样东西（相对前向）：

| 数据 | 大小 | 说明 |
|---|---|---|
| `P` 的 fragment | `Br·Bc` 个 fp32 | 前向用完就丢，反向要一直用到算完 `dS` |
| `dS` | 同 `P` | 中间量 |
| `dQ` 累加器 | `Br·d` 个 fp32 | 与前向的 `O` 同量级 |

> **反向的寄存器压力 ≥ 前向**：前向只有 `O`（`Br·d`），反向有 `O`（`D` 要用）、
> `P`/`dS`（`Br·Bc`）与 `dQ`（`Br·d`）。这也是 §1.3 那笔「算力换时间」的交易里
> 没有被计入的一侧。

### `dQ` 的两种累加：流量几乎一样，差别在确定性

`make verify` §B6 给出的模型（一个 Q 块 `Sq = 2048` 行、`d = 128`、fp32 累加器，
KV 切成 `n_split` 段）：

| `n_split` | atomic（读-改-写） | workspace（多写一份 + 归约） | 差 | atomic 侧时间 @3350 GB/s |
|---|---|---|---|---|
| 1 | 2.1 MB | 3.1 MB | 1.05 MB | 0.6 µs |
| 2 | 4.2 MB | 5.2 MB | 1.05 MB | 1.3 µs |
| 4 | 8.4 MB | 9.4 MB | 1.05 MB | 2.5 µs |
| 8 | 16.8 MB | 17.8 MB | 1.05 MB | 5.0 µs |
| 16 | 33.6 MB | 34.6 MB | 1.05 MB | 10.0 µs |

两者**只差一份 `dQ`**（`Sq·d·4 = 1.05 MB`）。而 workspace 那一侧还要多一次归约 kernel 的
launch：`3.0~5.0 µs`（**引用值，本仓库不复核其测量口径**）——
`make verify` §B6 断言了它与一次 `dQ` 的搬运时间**同量级**（0.60×）。

> **所以真正的差别不是流量，是确定性。** `atomicAdd` 的加法顺序由硬件在运行时决定：
> 换一个 grid 划分、换一张卡、甚至同一份输入跑两次，都可能得到不同的位模式。
> 这对训练意味着「同一份数据、同一份代码，两次跑出来的梯度不一样」——
> 复现实验、对拍、调试都会变难。
> 想可复现，就必须走 **workspace + 固定顺序归约** 那条路。

## 3. 实现与演进

按「`P` 存在哪里」这条线：

1. **朴素反向（物化 `P`）**。前向存、反向读。`O(S²)` 的显存与流量——
   01 篇的账在这里再付一次。
2. **重算 `S` 的反向（本文）**。前向只存 `O`（`S×d`）与 `(m, l)`（`S` 个标量），
   反向逐块重算 `S` 与 `P`。**关键在于 `D` 用 `rowsum(dO ∘ O)` 算**，
   于是重算不需要任何跨块的 `P`。（§1.2）
3. **拆分 `dQ` 与 `dK`/`dV` 的 kernel**。三个梯度按不同维度切分（§1.4），
   拆开之后每个 kernel 内部的原子需求最小化。
4. **并行度不够时切 KV**。当 `S/Br` 太小时（长序列 + 大 `Br`，或小 batch），
   按 KV 段切分把 grid 撑大——代价是 `dQ` 必须原子或 workspace（§2），
   这与 [gemm/06](../gemm/06-split-k-and-stream-k.md) 里 split-K 的取舍是同一个问题：
   **切分换来并行度，代价是归约。**
5. **`dQ` 的累加方式选择**（确定性 vs 流量）——与「跨 block 归约的确定性」是同一族问题，
   规划中的 `40-kernels/reduce-scan` 会正面处理它。本文只给结论：**换顺序 ⟹ 换位模式**。

## 4. 踩坑与失败模式

### 4.1 `dQ` 用 `atomicAdd` 导致结果不可复现

**现象**：同一份输入、同一份代码跑两次，`dQ` 有若干元素**位模式不同**。
`make verify` §C3 在 `24×16` 的 `dQ`（384 个元素）上：

```text
顺序累加 vs 反序（模拟 atomic 的不确定顺序）：265 个元素的位模式不同
顺序累加 vs 两两归约（workspace + 固定顺序）：220 个元素不同
但按范数归一的**数值**差只有 1.252e-07
```

**成因**：浮点加法**不满足结合律**。`(a+b)+c ≠ a+(b+c)` 在整数域成立，
在浮点域不成立——每次加法都要把结果舍入到 24 位有效数字，
不同的分组方式经过不同的舍入路径。

> **口径说明**：本程序用「换一个累加顺序」模拟 `atomicAdd` 的不确定顺序。
> 它证明的是「不同顺序 ⟹ 位模式不同」这一事实，**不是 atomic 硬件本身**。

**规避**：要可复现就走 workspace + 固定顺序归约；要极致性能就接受 `atomicAdd` 的非确定性。
**两条路都合理，但不能既不知道自己在哪条路上、又指望结果可复现。**
另外注意这条错误**不会被容差检查发现**（数值差 `1.25e-07`，任何 `allclose` 都过），
它只在「严格复现」这个要求下才是错误。

### 4.2 把 `D` 按块算

**现象**：`dQ` 与 `dK` 明显错掉，**而 `dV` 完全正确**——只检查 `dV` 的话，这个 bug 会整个溜过去。
`make verify` §C6（`24×16`，KV 分 8 块）：

```text
                       dQ          dK          dV
rowsum(dO ∘ O)（整行，正确）  3.05e-16    2.18e-16    2.45e-16
每块各自的 rowsum(dP_b∘P_b)   2.75e-01    2.92e-01    2.45e-16     ← 错 2.75e-01
```

**成因**：`dS = P∘(dP − D)` 里，只有 `D` 是跨块求和的量。分块之后每一块只知道
自己在那一行上的部分和，用它去减，等于给每一块单独做了一次「去掉自己的基准」，
块与块之间不再共享同一个尺度。

**规避**：`D` 从**最终的 `O`** 一次算出（§1.2）。这不是技巧，而是 flash 反向能成立的前提。
**本仓库写这篇时的第一版就是按块算的**，症状与 §C6 那张表完全一样：
`dQ` 与 `dK` 大错（§C6 的量级是 `2.75e-01` 与 `2.92e-01`），而 `dV` 一直精确——
正因为 `dV` 是对的，最初还以为是别的地方错了。
**`dV` 不检查 `D`**（`dV = Pᵀ·dO` 里根本没这一项），所以它不能当哨兵。

### 4.3 重算 `P` 时 `(m, l)` 配对不一致

**现象**：反向重算 `P` 时用错的 `(m, l)`，三个梯度**全都错**，而且错得很大。
`make verify` §C5（`8×16`，KV 分 8 块，参照 = 物化 `P` 的朴素反向）：

```text
重算 P 时用的 (m, l)                      dQ          dK          dV
前向最终的 (m_final, l_final)        4.19e-16    2.94e-16    1.48e-16   ← 正确
该块当时的 (m_b, l_b)                2.41e+00    3.85e+00    4.12e+00
该块当时的 m_b + 最终 l_final          1.11e+00    1.51e+00    9.15e-01
```

**这里要纠正一个流传很广的说法**：错的不在「用了**最终值**」。
`P = exp(S − m)/l` 要求 `(m, l)` 是**同一时刻的同一对**；
而前向收敛后的 `(m_final, l_final)` 恰好就是每行真 softmax 的分母，
所以它**是对的**（也正是 flash-attention 存下来交给反向的那两个量，
工程上进一步压成 `lse = m + ln l` 一个数）。

错的两种是：

- **`(m_b, l_b)`**：它描述的是「前 `b` 块的部分 softmax」。拿它当分母，
  等于把每一块都按自己的基准归一化，块与块之间没有共同的尺度。
- **`m_b` 配 `l_final`**：配对不一致，`exp(S − m_b)/l_final` 既不是部分 softmax
  也不是真 softmax。

**成因**：`m` 与 `l` 是一对**共同**描述同一个基准的量（[softmax/02](../softmax-normalization/02-online-softmax.md)
的 `⊕` 把它们当**一个**对象处理，道理就在这里）。取到不同时刻的两个值，就得到了一个
不描述任何东西的比值。**规避**：存 `lse = m + ln l` 一个数，从根上消除「配错」的可能。

## 编译与运行

配套程序是 **`code/03_flash_attention_backward.cu`**，含三个反向 kernel：

- `backward_seq_dq_kernel` —— 按 Q 的行切，`dQ` 顺序累加（确定），`dK`/`dV` 原子；
- `backward_atomic_dq_kernel` —— 按 KV 段切，`dQ` 用 `atomicAdd`；
- `backward_workspace_kernel` + `dq_reduce_kernel` —— 按 KV 段切，`dQ` 走 workspace。

共用 `recompute_block`：逐块重算 `S`、算 `P`、算 `D = rowsum(dO ∘ O)`、给出三个梯度。

它需要 `nvcc` 与 GPU：

```bash
cd code
make 03_flash_attention_backward
./03_flash_attention_backward             # S=512，d=64，Bc=64
./03_flash_attention_backward 1024 64 64
```

程序会打印：重算 vs 存储的两笔账（与 §0 / §1.3 的算式逐项对照）、
`dQ` 两种累加的流量对照表（§2）、三条路径与 CPU double 参考的逐元素差、
**三条路径两两的位模式不同元素个数**，以及本文 §4 的三条纪律。

**两类数字必须分开看**：

- 本文表格里的数、`4.62`、`1.074 GB`、`69.4 µs`、`16.8 MB`——**推导值**，
  可用 `make verify` 的 §B5 / §B6 复核；
- 各内核的逐元素差、位模式差异与耗时 —— **那台机器上的运行结果**。本仓库没有跑过。

本机（macOS，无 CUDA 工具链）只能跑：

```bash
make verify
```

它对应本文的是 §A2（用**数值梯度**独立校验反向：`dQ`/`dK`/`dV` 与有限差分的最大相对误差
分别为 `5.080e-09`、`1.908e-08`、`4.341e-08`；分块版与朴素反向的差 `< 1e-12`）、
§A3（三条累加路径数值等价）、§B5（重算 vs 存储）、§B6（`dQ` 两种累加的流量）、
§C3（位模式不同）、§C5（`(m, l)` 配对的三种取法）、§C6（`D` 按块算的后果）。

## 6. 小结与延伸

全文一条主线：**反向的每一个量都要么能重算、要么能从「整行」的信息里恢复；
一旦某个量既不能重算、又只有局部信息，分块就垮了。**

三条可独立复用的结论：

1. **重算划算，而且划算程度与 S 无关**：
   `t_traffic/t_recompute = (esize/d)·I*`，S=16384、d=128、fp16 时是 4.62 倍。
2. **`D = rowsum(dO ∘ O)` 必须整行算**。它是 `dS` 里唯一的非局部量，
   也正是「能扔掉 `P`」的前提。按块算会毁掉 `dQ` 与 `dK`，
   而 **`dV` 完全不受影响**——它不能当哨兵。
3. **三个张量按不同维度切分**（`dQ` 按 Q 的行、`dK`/`dV` 按 K 的行），
   两个方向的原子不可能同时省掉；`dQ` 的累加方式（原子 vs workspace）
   在流量上几乎一样（差一份 `dQ`），**差别在确定性**。

下一站：

- [04-warp-specialization-and-variants](04-warp-specialization-and-variants.md) ——
  把这些 kernel 放到 Hopper 上，用 warpgroup 分工让 wgmma 连续发射。
- [gemm/06-split-k-and-stream-k](../gemm/06-split-k-and-stream-k.md) ——
  「切分换并行度、代价是归约」这件事在 GEMM 里的正面处理，本文 §3 第 4 条的同一个问题。
- 规划中的 `40-kernels/reduce-scan`（跨 block 与确定性）—— 本文 §4.1 那族问题的正文。

## 参考

- Dao, Fu, Ermon, Rudra, Ré, *FlashAttention: Fast and Memory-Efficient Exact Attention
  with IO-Awareness*, NeurIPS 2022 —— 反向重算与 `D = rowsum(dO ∘ O)` 的方案出处。
- Dao, *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*,
  2023 —— 「按 Q 的行切 `dQ`、按 K 的行切 `dK`/`dV`」的 kernel 拆分与并行度讨论。
- 本仓库 [40-kernels/gemm/06-split-k-and-stream-k](../gemm/06-split-k-and-stream-k.md) ——
  split-K 的部分和归约与确定性，本文 §2 的 `dQ` 累加是同一族问题。
- 本仓库 [40-kernels/attention/02-flash-attention](02-flash-attention.md) ——
  前向省下的 `2S²·esize` 是本文要还的账；`(m, l, O)` 的状态格式也是这里定下的。
- 本仓库 [40-kernels/softmax-normalization/02-online-softmax](../softmax-normalization/02-online-softmax.md)
  —— `(m, l)` 必须成对处理的理由（`⊕` 把两者当一个对象）。
- 本仓库 `code/verify.py` 的 §A2、§A3、§B5、§B6、§C3、§C5、§C6 —— 本文全部数字的来源。

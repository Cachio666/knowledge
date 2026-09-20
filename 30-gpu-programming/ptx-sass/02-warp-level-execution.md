---
title: warp 级执行：同一个结果，两种写法，两笔代价
slug: warp-level-execution
domain: 30-gpu-programming
topic: ptx-sass
level: 2
difficulty: intermediate
prerequisites:
  - 20-gpu-architecture/execution-model/02-simt-and-divergence
  - 30-gpu-programming/ptx-sass/01-ptx-as-virtual-isa
tags: [ptx, simt, warp-divergence, shfl, bar-sync, predication, reconvergence]
related:
  - 30-gpu-programming/ptx-sass/03-memory-and-sass-gap
  - 30-gpu-programming/cuda/01-thread-hierarchy-and-sync
  - 40-kernels/reduce-scan/01-reduction-algorithms
  - 20-gpu-architecture/memory-system/03-shared-memory-bank-conflict
code:
  - code/02_warp_execution.py
status: stable
updated: 2026-09-20
---

# warp 级执行：同一个结果，两种写法，两笔代价

> **同一段 `if`，写成真分支与写成谓词，结果逐元素相同，代价不同。** 本篇用自己写的 PTX 子集解释器
> （`code/ptxsub.py`）把「代价」做成**逐发射的活跃 lane 数**，并给出四条带数字的结论：
>
> 1. ★★ **同一份输入、同一个结果，`BRANCH` 与 `PREDICATED` 在模型里的发射数是 11 vs 9、活跃 lane-指令数是 304 vs 288**
>    —— 差 **2 条发射 / 16 个活跃 lane-指令**。发散调度步数 **3 vs 0**。
>    **这两组数都是本解释器的模型值**（它是逐 warp 推进到阻塞点的），不是硬件上的数。
> 2. ★★ **重聚是 PC 相等自然发生的**：`bra L_END` 让两条路径走到**同一个 pc**，
>    于是下一步两个 lane 组自动合流 —— **不需要任何显式的 re-convergence 栈**。
>    `BRANCH` 的 11 步 pc 序列**严格单调递增**（`0,1,…,10`），程序 `B.4c` 钉住这一点。
> 3. ★ **`bar.sync` 的到达计数按 warp 增量，不按活跃 lane 数**：ISA `9.7.15.1` 原文说
>    「**marks warps' arrival** at barrier」。程序 `B.6` 把这件事跑成一个**极端例子**：
>    **只剩 1 个活跃 lane** 执行 `bar.sync 0, 32` 时，屏障**仍然放行** ——
>    若改成按活跃 lane 计数，那个例子会**死锁**（`meta_falsify.py` 的 `D3` 就是这个缺陷）。
> 4. ★★ **`shfl.sync.up` 的 `pval` 判据【未定案】**：ISA 原文写 `maxLane`，**本库的实现取 `minLane`**
>    —— 这是**有意偏离**，而 `cval = 0`（ISA 自己的例子用的就是这个值）时两式**恰好重合**，
>    所以**无法判定谁对**。§4.1 逐字引原文并列出三种可能。

> ★ **本域的三条强制限定语，在本篇的落点**（域 README 规定逐篇写明，不得省略）：
>
> 1. **「不是硬件」**——`code/ptxsub.py` **逐 warp 推进到阻塞点**：一个 warp 一路跑到
>    `ret` / 阻塞在 `bar.sync` / 结束，才轮到下一个 warp。**真实 SM 上多个 warp 是指令级交错的**
>    （多个 scheduler、双发射、延迟隐藏），所以本篇的发射数 / 活跃 lane-指令数 / 发散步数
>    **全是这个模型的产物**。尤其「`BRANCH` 比 `PREDICATED` 多 2 条发射」这件事，
>    在真实硬件上还会被 **SASS 层的调度与谓词化**改写 —— **不能**读作硬件行为。
> 2. **「按 PTX ISA `9.7.10.6`（`shfl.sync`）/ `9.7.15.1`（`bar, barrier`）实现」**——
>    引文与章节号逐字保存在 `code/specs.py` 的 `ISA` 表。★ 且 `shfl.sync.up` 处
>    **本库的实现是对 ISA 文本的有意偏离**，§4.1 逐字引原文。
> 3. **「手写 PTX 的语法未经过任何工具验证」**——本机**无 `ptxas`**；本篇的每一段 PTX 都只被 `ptxsub` 解释过。

## 0. 背景与动机

### 0.1 两个具体的对照场景

**场景 A（同一个 `if`，两种写法）**。对一个数据相关分支，PTX 里至少有两条路：

```ptx
// BRANCH：真分支
    setp.lt.s32 %p0, %r1, 0;        // p0 = (data < 0)
    @%p0 bra L_NEG;
    add.u32  %r2, %r1, 1000;        // then
    bra      L_END;
L_NEG:
    sub.u32  %r2, 0, %r1;           // else
L_END:
    st.shared.u32 [%r7], %r2;
```

```ptx
// PREDICATED：谓词化（if-conversion 风格）
    setp.lt.s32 %p0, %r1, 0;
    @%p0  sub.u32 %r2, 0, %r1;      // 谓词执行，**无跳转**
    @!%p0 add.u32 %r2, %r1, 1000;
    st.shared.u32 [%r7], %r2;
```

**同一份输入**（一个 warp 里前 16 个 lane 的数据为负、后 16 个为正），两种写法**输出逐元素相同**。
代价：`BRANCH` **11 次发射 / 304 活跃 lane-指令 / 3 步发散**，
`PREDICATED` **9 / 288 / 0** —— **差 2 条发射、16 个活跃 lane-指令**。

**场景 B（同一个「按列取用」，两种 lane 模式）** —— 那是 `03-memory-and-sass-gap` 的题材；
这里只提一句它为什么属于同一个主题：**warp 的 32 个 lane 怎么落到数据上**，
在两篇里都是那个决定代价的东西（一个是控制流，一个是访存）。

### 0.2 本篇覆盖什么、不覆盖什么

本篇落在 `code/02_warp_execution.py` 的 **34 条断言**里（A 7 / B 17 / C 10）：

| 组 | 内容 | 数量 |
|---|---|---|
| **A** | bfly 归约的**三条互不重叠**参照（numpy 分段和 / 纯 Python 位运算 / 闭式 clamp 公式）、三种 shuffle 模式的逐 lane 映射、位搬运 | 7 |
| **B** | 发射三元组、逐发射 trace、pc 序列、`bar.sync` 的 warp 粒度到达、ISA `.up` 例子的 `cval` 解析 | 17 |
| **C** | 缺 `bar.sync` 的 32/64 指纹、**可证伪性**（含一个**等价变异**）、`.up` 两式的对照与 `cval=0` 的重合 | 10 |

**边界**：warp 与独立线程调度的**硬件原理**（为什么 Volta 之后逐线程 PC 改变了屏障语义）
回链 `20-gpu-architecture/execution-model/02-simt-and-divergence`；
`__syncthreads()` 与 `bar.sync` 在 **CUDA 源语层**的契约（含 Pascal/Volta 的架构分界）
在 `30-gpu-programming/cuda/01-thread-hierarchy-and-sync`，本篇**不重复**。
归约算法本身在 `40-kernels/reduce-scan/01-reduction-algorithms`。

### 0.3 统一主线：那个「别处」在哪里

本域主线是「**你写的程序、编译器生成的东西、硬件执行的东西，是三个不同的程序**」。
本篇守的是其中一个很具体的形态：**同一个语义，PTX 允许你写两种控制流，
而两者的代价在 SASS 上还会再变一次**。代价被藏到了哪里？

> **它被藏进了「if-conversion 是不是免费的」这个默认假设里。**
> 「谓词化比分支快」是一条**在 PTX 层的近似**：它在**本模型**里成立（9 < 11），
> 而在真实硬件上，这个判断是**编译器**做的、而且**依赖它知道分支代价**——
> `ptxas` 会把 PTX 的 `bra` 与 `@%p` **重新选择一遍**（它可能把 `bra` 谓词化，也可能反过来
> 把谓词变成分支）。⇒ **你在 PTX 层数出来的发射数，不是 SASS 层的发射数**（§2.3）。
> 那个「别处」就是**下一个编译器**：本篇的 11 vs 9 到了 SASS 层会被重新洗牌，
> 而**洗牌的依据（分支的可能性、发散的真实频率）恰恰是 PTX 文本里没有的**。

## 1. 原理

### 1.1 `setp` 与 `@%p`：谓词是**每个 lane 各自**的值

PTX 的谓词寄存器是**每线程**的（`.reg .pred %p<4>` 声明 4 个谓词槽）。
`setp.lt.s32 %p0, %r1, 0` 在每个 lane 上各自算出一个布尔值；
`@%p0 <指令>` 让**谓词为真的那些 lane** 执行这条指令，其余 lane **不执行**（但仍要等这条指令发射）。

⇒ 一条被谓词守卫的指令，它的一次发射里**活跃 lane 数可能小于 32**。
**这就是本篇「代价」的全部量纲**：不是「几条指令」，而是「**几次发射 × 每次多少活跃 lane**」。
程序 `B.1`–`B.3` 把它拆成两个数：

- **发射数（issue）**：一个 warp 发一条指令算 1（**不管它覆盖几个 lane**）；
- **活跃 lane-指令数（lane_issue）**：该次发射命中的 lane 数之和（**这才是「工作量」**）。

### 1.2 ★ 多步推导：`BRANCH` 的 11 步是怎么来的

设一个 warp，lanes `0..15` 的数据为负（`p0` 真）、`16..31` 为正。指令编号（`pc`）按源码顺序：

| pc | 指令 | **执行**它的 lane（≠ 计入 `lane_issue` 的那个组） | 发射后各 lane 的 pc |
|---:|---|---|---|
| 0–4 | `mov`/`mov`/`mul`/`ld`/`setp` | 全部 32（无谓词） | 全体 → 5 |
| 5 | `@%p0 bra L_NEG` | `p0` 为真的 16 个 | **分裂**：0–15 → 8；16–31 → 6 |
| 6 | `add.u32 %r2, %r1, 1000` | 16–31（16 个） | 16–31 → 7 |
| 7 | `bra L_END` | 16–31 | 16–31 → 9 |
| 8 | `sub.u32 %r2, 0, %r1` | 0–15（16 个） | 0–15 → 9 |
| 9 | `st.shared.u32` | **全部 32**（两组已重聚） | 全体 → 10 |
| 10 | `ret` | 全部 32 | 退出 |

**逐步算两个计数器**（本解释器对**每一个调度步**都记一次）：

$$\text{issue} = 11 \quad (\text{pc}=0\ldots10)$$
$$\text{lane\_issue} = 5\times 32 + 32 + 16 + 16 + 16 + 32 + 32 = 160 + 144 = 304$$

★ **注意 pc=5 那一步**：`@%p0 bra` 的**活跃 lane 只有 16 个**（真的是 16），
但**发射数是 1**、而这一步在 `lane_issue` 里记的是**该 pc 上的 lane 组大小 32**
（因为「谁在 pc=5」是全部 32 个 lane）—— 这个口径就是 `ptxsub` 的实现，
程序 `B.4` 的逐发射 trace 把它逐条打出来：`[32,32,32,32,32,16,16,16,16,32,32]`。

**`PREDICATED` 的 9 步**：`pc=0..4` 五次满 32 lane，`pc=5`（`@%p0 sub`）与 `pc=6`（`@!%p0 add`）
各命中 16 lane，`pc=7`（`st`）与 `pc=8`（`ret`）满 32 ⇒

$$\text{issue} = 9,\qquad \text{lane\_issue} = 5\times 32 + 16 + 16 + 32 + 32 = 160 + 96 = 256\ \text{(✗ 见下)}$$

★ **上面这个手算的 256 是错的，而程序给出 288。** 差在哪？——在**记的是哪个量**：
`lane_issue` 记的是**该 pc 上的 lane 组大小**，而不是「谓词为真的 lane 数」。
`PREDICATED` 里 `pc=5` 与 `pc=6` 上**全部 32 个 lane 都在这个 pc**（谓词只决定谁**执行**，
不改变 PC） ⇒ 那两步各记 32，于是

$$\text{lane\_issue} = 5\times32 + 32 + 32 + 32 + 32 = 160 + 128 = 288$$

**这个「手算错了、程序对了」的插曲本身就是本篇的一条结论**：
「活跃 lane 数」有两个自然定义 ——
**(a) 谓词为真的 lane 数**（`BRANCH` 在 pc=5 上真的是 16），
**(b) 落在同一个 pc 上的 lane 数**（`PREDICATED` 在 pc=5 上仍是 32）。
本库用的是 **(b)**，因为 **(a) 在 `BRANCH` 的分裂点之后根本无从定义**（两组 lane 在不同的 pc 上，
「这条指令的活跃 lane 数」在 `pc=6` 上只可能是 16）—— **(b) 才是两种写法共用的口径**。
**换一个口径，11/304 与 9/288 这两组数都会变** ⇒ 引用它们时必须写明口径（本域 P9 的又一实例）。

### 1.3 `shfl.sync`：warp 内的寄存器交换，以及两个打包在 `c` 里的参数

`shfl.sync.<mode>.b32 d, a, b, c, membermask` 的 `c` 操作数**打包了两个 5 位字段**
（ISA `9.7.10.6` 原文逐字，见 §4.1）：

```
cval[4:0]    = c[4:0];        // clamp 值
segmask[4:0] = c[12:8];       // 把 warp 逻辑切成子段的掩码
maxLane = (lane[4:0] & segmask[4:0]) | (cval[4:0] & ~segmask[4:0]);
minLane = (lane[4:0] & segmask[4:0]);
```

**`segmask` 的作用**是把 warp 切成子段：`segmask = 0` 时 `maxLane = cval`（全 warp 一档）；
`segmask = 0x08` 时 `maxLane` 会**逐段**变化（0–7 段与 8–15 段各一个上界）。
程序 `B.7` 从 **ISA 的引文里解析**出它自己那个例子的参数：

```
shfl.sync.up.b32 Ry|p, Rx, 0x1, 0x0, 0xffffffff;     ← bval = 0x1, cval = 0x0
```

**注意 `cval = 0x0`** —— 这一点是 §4.1 的关键（`cval = 0` 时 `maxLane` 与 `minLane` **恰好相等**）。

**clamp 生效时会发生什么**（程序 `A.3` 在本机跑出来的三档，`shfl.sync.bfly.b32`、`bval = 8`）：

| `cval` | `maxLane` | 返回**自己**的 lane 数 | 说明 |
|---:|---:|---:|---|
| 31 | 31 | **0** | clamp 不生效：全部 32 个 lane 都拿到 `lane ^ 8` |
| 15 | 15 | **16** | lane 16–23 的 `lane^8 = 24..31 > 15` ⇒ 越界，拷自己 |
| 7 | 7 | **24** | 只剩 0–7 与 8–15 两段内互换，其余拷自己 |

★ 这三档是**同一段伪代码**在不同参数下的行为，不是三条规则（程序 `A.3` 用闭式公式独立算了第二遍）。

### 1.4 `bar.sync`：到达的**单位是 warp**

ISA `9.7.15.1` 原文（逐字，见 `specs.ISA['B1']`/`B2`/`B3`）：

- 「either specified number of threads (**in multiple of warp size**) or all threads in the CTA participate」
  ⇒ 参与单位是 **warp size 的整数倍**；
- 「**marks warps' arrival** at barrier」⇒ 到达是**按 warp 记的**；
- 「All threads in warp (except for those have exited) must execute `barrier{.cta}` instruction **in convergence**」。

本库据此把到达量记成 **`+= 32`**（每个 warp 到达一次），而**不是** `+= 活跃 lane 数`。
程序 `B.6` 把这个选择跑成一个**极端例子**：只有一个 lane 的谓词为真、只有它执行
`bar.sync 0, 32` —— **屏障仍然放行**（到达量 32 ≥ count 32）。
若按活跃 lane 计数（`+= 1`），这个例子会**死锁** ⇒ 这条断言是**可证伪**的。

## 2. 执行映射

### 2.1 逐发射的 trace（程序 `B.4` 钉住的 11 步）

| step | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **pc（BRANCH）** | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| 活跃 lane 数 | 32 | 32 | 32 | 32 | 32 | **16** | **16** | **16** | **16** | 32 | 32 |
| **pc（PREDICATED）** | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | — | — |
| 活跃 lane 数 | 32 | 32 | 32 | 32 | 32 | 16 | 16 | 32 | 32 | — | — |

**两张表的读法**：
- `BRANCH` 多出的两步是 `pc=6/7` 与 `pc=8` —— 两条路径**各自**要走一遍；
- `PREDICATED` 的 `pc=5/6` 是**两条谓词指令**（`@%p0` 与 `@!%p0`），
  它们各只命中 16 lane，但**都只发一次**（不是两次）；
- **重聚点是 `pc=9`**（`st.shared`）：两条路径的 `bra` 都把目标指向同一个标签
  ⇒ **PC 相等即重聚**，没有别的机制。程序 `B.4c` 断言 `BRANCH` 的 pc 序列**严格单调递增**。

### 2.2 参与的对象与量

| 对象 | 本模型里的角色 |
|---|---|
| warp | 调度单位：**一次发射覆盖整个 warp**（活跃与否由谓词/PC 决定） |
| lane 的 PC | **每个 lane 各有一个 PC**；PC 不同即发散（这是模型的发散定义） |
| `setp` / `.pred` | 每 lane 的布尔值，只影响「谁执行」，**不影响 PC** |
| `bra`（含 `@%p`） | **唯一**会改变 PC 的指令（⇒ 发散的来源） |
| `shfl.sync` | warp 内寄存器交换，`membermask` 决定谁参与 |
| `bar.sync` | 阻塞点：到达量按 **warp** 记 |
| **真实的 warp scheduler / re-convergence 硬件** | **本模型没有**它们（§4.4） |

### 2.3 ★ 本模型的数字到了 SASS 层会再变一次（只做描述，本机无法观测）

| 层 | 谁在做选择 | 本库能观测吗 |
|---|---|---|
| CUDA C++ | `if` 写成什么样 | ✗（本主题不写 `.cu`） |
| PTX | `bra` 还是 `@%p`（**本篇的题材**） | ✅ 手写片段 + `ptxsub` |
| **SASS** | `ptxas` **可能把 `bra` 谓词化，也可能把谓词变成分支**；并重排指令以掩盖延迟 | ✗（无 `ptxas` / `nvdisasm`） |
| 硬件 | 多 scheduler 交错、双发射、发散多久 | ✗（无 GPU） |

⇒ **「`BRANCH` 比 `PREDICATED` 多 2 条发射」这句话的作用域到此为止**：
它是**PTX 层、本模型内**的结论。`ptxas` 完全可以把上面那个 `BRANCH` 谓词化成 9 步、
也可以把 `PREDICATED` 变成带跳转的版本 —— **那一步本库看不到**。

## 3. 实现与演进：为什么有 `.sync` 后缀

### 3.1 ISA 原文：`shfl` 的引入与弃用（锚点 `P3`，章节 `9.7.10.5`）

```
Introduced in PTX ISA version 3.0. Deprecated in PTX ISA version 6.0 in favor of shfl.sync.
```

⇒ **两节并存**：`9.7.10.5` 是无后缀的旧 `shfl`（deprecated），`9.7.10.6` 是 `shfl.sync`。
**这不是风格问题，是跨了一个语义断点**：`.sync` 形式**带 `membermask`**，
并且规定「`shfl.sync` 的语义在 diverged 控制流下是 undefined 的，如果执行线程不在 `membermask` 内」
（`specs.ISA['P4']` 逐字）—— 无后缀的那个版本**没有这个参数**，也就没有这个契约。

程序 `B.6` 把这条写成一条**可机械判定的**判据：扫本主题**五个源文件**里出现的每一个
`shfl.<…>` token，断言**全部**是 `shfl.sync.…` 形式（弃用形式出现 **0** 次）。
★ 这条断言第一版**误报**了：正则的字符类漏了数字，`shfl.sync.bfly.b32` 被截成
`shfl.sync.bfly.b`，于是「不以 `shfl.sync` 开头的 token」凭空多出几个。
**修法是把数字加进字符类**，而教训是**扫描范围包含注释** ⇒ 注释里也不能写「像 token 一样的字面量」。

### 3.2 演进：从「能用」到「能证伪」

| 版本 | 做法 | 被什么逼着改 |
|---|---|---|
| 探针 | 归约与 numpy 对拍（`array_equal` = True） | 只能证明「算对了」，不能证明「断言抓得住错」 |
| 探针 v2 | 加发散计数（issue / lane_issue / divergent） | 需要「同一结果、两笔代价」的载体 |
| **本篇** | ① 逐发射 `trace_hook`；② **可证伪性**实验（含一个**等价变异**）；③ `.up` 的分歧被写成**可检测的**（`meta_falsify` 的 `D7`） | 计划 §1 C3 要求「至少一个可证伪的断言」；而 `.up` 的「有意偏离」如果不被断言钉住，就只是一句注释 |

## 4. 踩坑与失败模式

### 4.1 ★★ `shfl.sync.up` 的判据【未定案】

**（a）ISA 原文（逐字，章节 `9.7.10.6` Data Movement and Conversion Instructions: `shfl.sync`）**：

```
maxLane = (lane[4:0] & segmask[4:0]) | (cval[4:0] & ~segmask[4:0]);
minLane = (lane[4:0] & segmask[4:0]);
switch (.mode) {
  case .up:   j = lane - bval;  pval = (j >= maxLane);  break;
  case .down: j = lane + bval;  pval = (j <= maxLane);  break;
  case .bfly: j = lane ^ bval;  pval = (j <= maxLane);  break;
  case .idx:  j = minLane | (bval[4:0] & ~segmask[4:0]);  pval = (j <= maxLane); break;
}
if (!pval) j = lane;   // copy from own lane
```

★ **`.up` 那一行写的是 `maxLane`**（`j >= maxLane`）。`minLane` 在这段伪代码里**只出现两次**：
定义行与 `.idx` 分支。原文与计数一起保存在 `code/specs.py`（`ISA['P1']` 与 `RAW_COUNTS`），
并由程序 `A.4` 与 `make anchors` 分头钉住（在**原始字节**上数：`j &gt;= maxLane` = 2、`j &gt;= minLane` = 0）。

**（b）本库的实现是对 ISA 文本的有意偏离**：`code/ptxsub.py` 的 `.up` 分支取的是 **`minLane`**：

```python
elif mode == 'up':
    j = l - (src & 0x1F)
    pval = (j >= minLane)          # ★ 有意偏离 ISA 文本（原文是 maxLane）
```

**理由是 `cval = 0` 时两式等价** —— 而 **ISA 自己给 `.up` 的例子正是 `cval = 0`**
（`shfl.sync.up.b32 Ry|p, Rx, 0x1, 0x0, 0xffffffff`，程序 `B.7c` 从引文里解析出 `0x0`）。
程序 `C.4` 把「等价」这件事**穷举**了一遍：`clamp = 0` 时，
对**全部 32 个 `segmask` × 32 个 lane**（1024 组），两式**逐 lane 相等**。

**（c）为什么按 `maxLane` 那式会让 `.up` 退化**（程序 `C.3` 的那组参数）：

```
lane = 16, bval = 1, cval = 31（segmask = 0 ⇒ maxLane = 31, minLane = 0）
j = lane - bval = 15
ISA 原文：pval = (15 >= 31) = False  ⇒ j = lane = 16   （**返回自己**，`.up` 成了空操作）
本库实现：pval = (15 >=  0) = True   ⇒ j = 15          （取 laneid 更小的那个）
```

⇒ **两式在这个参数下结论相反**。程序 `C.4b` 还给出规模：换到 `clamp = 31, segmask = 0` 时，
32 个 lane 里有 **31** 个两式分道（只有 lane 0 例外，因为 `j = -1 < 0`）。

**（d）★ 这一点【未定案】，本库无法判定**。三种可能，**一条也不声称**：

1. **(a)** ISA 文本在此处有**笔误**（`.up` 本该写 `minLane`）；
2. **(b)** `cval` 对 `.up` **本就该传 0**（ISA 自己的例子就是 0），
   于是两式**等价**、**根本不构成矛盾**；
3. **(c)** 对 **`cval` / `segmask` 的语义理解不对**（例如 `maxLane` 在这里的含义，
   与「源 lane 的上界」不是一回事）。

**为什么本库判不出**：**本机没有 `ptxas`、没有 GPU**（`nvcc` / `cuobjdump` / `nvdisasm` / `ncu` 全无）。
⇒ 本篇**不得**写「规范规定 `minLane`」「二手转述是错的」「已定案」—— 那是**伪造引文**。

**（e）但「本库偏离了原文」这件事本身是**可检测的**：`meta_falsify.py` 的 `D7`
把 `.up` 的实现**改成照抄原文**（`maxLane`），断言 `02` 必须**失败**（它确实失败在 `A.4`）。
⇒ 本库没有偷偷偏离：**偏离被写在源码里、被断言钉着、改回去会被抓住**。
（`D7` 是七条缺陷里**唯一一条方向相反**的：它不是「改错」，是「改成另一个候选语义」。）

★ **这一节与另一件事必须分清**：本主题的规划阶段还犯过一次**转写错误** ——
抽取脚本打印的是 `maxLane`，写进规格时被抄成了 `minLane`，并据此编出「ISA 用 `minLane`、
二手转述是错的」（详见 `01-ptx-as-virtual-isa.md` §4.3）。
**那是抄写失误，不是判断失误**；而本节是**明知原文、仍按语义取舍**的实现选择。
**两件事长得很像，结论完全不要混。**

### 4.2 构造「能暴露同步缺失」的例子**需要设计**

- **现象**：随手写一个「两个阶段、阶段间有依赖」的 kernel，**缺 `bar.sync` 也不出错**。
- **成因**：如果邻居关系是 `tid ^ 1`（或任何只在**同一个 warp 内**的配对），
  那么该 warp 内部的顺序由**同一个 lane 组**自己保证 —— **缺屏障也不会错**。
  必须让依赖**跨 warp** 才敏感：本例用 `(t + 32) % 64`（64 线程 = 2 个 warp），
  于是 `t < 32` 的线程要从 warp1 的写入里读数据。
- **规避**：设计例子时先问「**这个依赖跨不跨 warp**」。程序 `C.1` 把它跑出来：
  缺 `bar.sync` 时**错误元素 32/64**，错的**恰好是 `t < 32` 的那一半**，
  而且错值**全是初始哨兵 `-1`** —— 这是「读到了尚未写入的共享内存」的**指纹**，
  不是随机数值错误。
- **★ 必须带限定语**：这是**本模型**的产物（逐 warp 推进到阻塞点 ⇒ warp0 必然先跑完）。
  真实 SM 上 warp0 未必能一路跑完阶段 2，**「错 32 个」不能读作硬件行为**。

### 4.3 ★ 「改了它断言没叫」有两种原因：**断言弱** 或 **变异等价**

- **现象**：把 bfly 归约的某一轮**异或步长从 8 改成 9**，numpy 对拍**仍然全等**（不匹配 0/128）。
- **成因**：这是一个**等价变异**，不是断言漏了。代数：前 3 轮（步长 1/2/4）之后，
  每个 lane 持有「与它同处一个 8-lane 块」的局部和 $S[l \gg 3]$；
  步长 $9 = 0b1001$ 翻转 bit3 与 bit0，于是 $l \oplus 9$ 的**块号**正好是 $(l \gg 3) \oplus 1$
  ⇒ 该轮把相邻两块配对成 $S[q] + S[q \oplus 1]$；最后一步（16）再把 $q$ 与 $q \oplus 2$ 配对
  ⇒ **恰好还是总和**。
- **规避**：可证伪性实验要选一个**真的会改变结果**的变异：程序 `C.2` 用的是
  **8 → 3**（同样的位置），它让 128 个元素**全部**不匹配。
  **两个变异体都留在程序里**（`C.2` 与 `C.2b`），并断言它们的结论**相反**
  （128 vs 0）—— 「分不清等价变异与漏网」是 P5 的反面，这条断言把它变成了可判定的。
- **一般化**：`make mutate` 报出的**漏网**里，**有一部分是等价变异**。
  本库不把「漏网」直接读成「断言无效」——两者的区别是**可判定的**（把变异体换一个再试）。

### 4.4 ★ 那个「别处」在哪里

本域主线说：**抽象层的价值与它的泄漏是同一件事的两面**。那么本篇的代价被藏到了哪里？

> **它被藏进了「下一个编译器」。**
>
> 你在 PTX 里写 `bra` 或 `@%p`，看起来是在**做选择**；但 `ptxas` 会**再选一次**——
> 它可以把 `bra` 谓词化，也可以把谓词变成分支，还可以为掩盖延迟把你的指令重排。
> ⇒ **本篇的 11 vs 9 是 PTX 层的数，SASS 层是另一个数**（§2.3）。
> 而 `ptxas` 做这个选择所依据的信息（**分支实际发散的概率**、它估的代价模型），
> **恰恰不在 PTX 文本里** —— 你在源码层写的「这个分支多半一致」它**收不到**。
>
> 那个「别处」就是**编译器之间的这一层**：PTX 承诺的是「一种可移植的中间表示」，
> 它的代价是**把代价决策让渡给了下游**。于是同一段 PTX 在不同架构、不同 `ptxas` 版本上
> 会被编译成不同的 SASS —— **而本篇与 `01` 篇一样，连看一眼 SASS 的工具都没有**。

同一条主线的第二种形态也在本篇里：`bar.sync` 的契约（「warp 内所有未退出线程必须
**收敛地**执行」）**本解释器不检查** —— 它只按 warp 粒度记到达。
**一个发散地执行的 `bar.sync`，在本模型里照样放行**（程序 `B.6` 的那个例子正是这样）。
真实硬件上这属于未定义行为；**本库不声称覆盖 UB**。

## 5. 判据表

> ★ 按本域约定（域 README），§5 的位置与编号保留，内容**不是实测数据**而是**判据表**。
> 每一行是「**可机械判定的条件 → 结论**」。前提：本机无 GPU、无 PTX 工具链，
> 本主题**没有任何 GPU 数字**；所有数字都是**模型值**（`ptxsub` 跑出来的）或**引用值**（ISA 原文，带章节号）。

| # | 可机械判定的条件 | 结论 |
|---|---|---|
| 1 | 同一份输入，`BRANCH` 与 `PREDICATED` 的输出 | **逐元素相同**（程序 `B.1c`） |
| 2 | `BRANCH` 的 (issue, lane_issue, divergent) | **(11, 304, 3)**——本模型的值（程序 `B.2`） |
| 3 | `PREDICATED` 的同一三元组 | **(9, 288, 0)**（程序 `B.2b`） |
| 4 | 两者的差 | **2 条发射 / 16 个活跃 lane-指令**（程序 `B.3`） |
| 5 | 「活跃 lane 数」的口径 | 本库用「**落在同一个 pc 上的 lane 数**」；换成「谓词为真的 lane 数」两组数都会变（§1.2） |
| 6 | 两条路径的 `bra` 指向同一个标签 | **重聚自然发生**，无需 re-convergence 栈（程序 `B.4c`：pc 序列严格单调递增） |
| 7 | `bar.sync` 的到达量 | 每个到达的 warp **+32**（不按活跃 lane 数；ISA `9.7.15.1` 原文） |
| 8 | 只有 1 个活跃 lane 执行 `bar.sync 0, 32` | **放行**——到达按 warp 记 ⇒ 32 ≥ 32（程序 `B.6`）。改成按 lane 记会**死锁** |
| 9 | `bar.sync 0`（省略 count）与 `bar.sync 0, 64` | **结果相同**（前者默认全 CTA；程序 `B.5b`） |
| 10 | 缺 `bar.sync`，且依赖**跨 warp** | **错误元素 32/64**，错的恰是 `t < 32` 的一半，错值**全是哨兵 `-1`**（程序 `C.1`） |
| 11 | 缺 `bar.sync`，但依赖只在**warp 内**（如 `tid ^ 1`） | **不出错**——随手写的例子大概率不敏感（§4.2） |
| 12 | `shfl.sync.bfly.b32` 的 clamp 取 31 / 15 / 7 | 返回自己的 lane 数 **0 / 16 / 24**（程序 `A.3`） |
| 13 | ISA `.up` 例子的 `cval` | **0x0**（从引文里解析出来；程序 `B.7c`） |
| 14 | `clamp = 0` 时 `maxLane` 与 `minLane` | **逐 lane 相等**（1024 组穷举；程序 `C.4`） |
| 15 | `lane=16, bval=1, cval=31` 下两式 | **结论相反**：原文给 16（返回自己）、本库给 15（程序 `C.3`） |
| 16 | 那一个「谁对」 | **本库不判定**——三种可能，见 §4.1(d)（程序 `C.3b`） |
| 17 | 把 `.up` 的实现改成照抄原文的 `maxLane` | **程序必须失败**（`make falsify` 的 `D7` 在本机跑出：`A.4` 报错） |
| 18 | bfly 的某一轮步长 8 → **3** | 128/128 元素不匹配 ⇒ 对拍**抓得住**（程序 `C.2`） |
| 19 | bfly 的某一轮步长 8 → **9** | **不匹配 0/128** ⇒ **等价变异**，不是断言漏了（程序 `C.2b`） |
| 20 | 本主题五个源文件里出现非 `.sync` 形式的 `shfl` | **ERROR**（程序 `B.6`） |
| 21 | 想知道 SASS 层的发射数 | **本库不知道**——`ptxas` 会重新选择（§2.3） |

**闭合性检查**：读完这张表能判完「一段**本子集内**的 warp 级 PTX 在本模型里的代价、
以及哪几类本库**判不了**（第 11、16、21 条）」。
**唯一的表外输入**是「这段 PTX 想算什么」——它必须**读文本**。

## 6. 小结与延伸

- ★★ **同一个结果，两笔代价**：`BRANCH` `(11, 304, 3)` vs `PREDICATED` `(9, 288, 0)` ——
  **差 2 条发射、16 个活跃 lane-指令**。**这是本模型的值**，不是硬件数字。
- ★★ **重聚不需要机制**：`bra` 把两条路径送到**同一个 pc**，下一步就自动合流。
  程序 `B.4c` 钉住 `BRANCH` 的 pc 序列严格单调递增。
- ★★ **「活跃 lane 数」有两个自然口径**（谓词为真的 lane 数 / 落在同一 pc 的 lane 数），
  **换口径两组数都会变** ⇒ 引用时必须写明口径。这也是为什么本库选了 (b)：它在两种写法下都良定义。
- ★ **`bar.sync` 的到达按 warp 增量**（ISA `9.7.15.1` 原文），
  极端例子：**只剩 1 个活跃 lane 也照样放行**（程序 `B.6`，`make falsify` 的 `D3` 是它的反证）。
- ★ **缺 `bar.sync` 的指纹是「错的那一半全是哨兵 `-1`」**，而构造这样的例子**必须让依赖跨 warp**。
- ★★ **`shfl.sync.up` 的 `pval` 判据【未定案】**：原文写 `maxLane`、本库有意取 `minLane`；
  `cval = 0`（ISA 自己的例子）时两式**恰好重合** ⇒ **判不出谁对**。三种可能，**一条也不声称**。
  ★ 且「本库偏离了原文」这件事**是可检测的**（`D7`）。
- ★ **「改了它断言没叫」有两种原因**：断言弱，或**变异等价**（8 → 9 就是一例）。
  程序 `C.2` / `C.2b` 把这两个变异体都留着，并断言它们**结论相反**。
- ★ **残余（如实列出）**：本模型**不检查** `bar.sync` 的收敛性要求（ISA 原文要求
  「in convergence」），也**不覆盖任何 UB**（`membermask` 违规只报错，不等于复现了 UB）。

**延伸**：

- PTX 的虚拟性与覆盖范围表（本篇不讲）：`30-gpu-programming/ptx-sass/01-ptx-as-virtual-isa`。
- 共享内存访存计数与 PTX → SASS 的两大落差：`30-gpu-programming/ptx-sass/03-memory-and-sass-gap`。
- `__syncthreads()` 在 CUDA 源语层的契约（含 Pascal / Volta 的架构分界）：`30-gpu-programming/cuda/01-thread-hierarchy-and-sync`。
- 归约算法本身（本篇只讲 warp 内 bfly 的**代价**）：`40-kernels/reduce-scan/01-reduction-algorithms`。
- SIMT 与发散的硬件原理：`20-gpu-architecture/execution-model/02-simt-and-divergence`。

## 编译与运行

```bash
cd 30-gpu-programming/ptx-sass/code
make run                      # 跑本主题全部三个编号程序；本主题这一篇是第二个
make 02_warp_execution        # 只跑本篇对应的那一个
make falsify                  # ★ 元检验：往 ptxsub.py 注入 7 种缺陷，断言程序必须失败
make mutate                   # 叶子级变异测试（非编号工具，**不进 `make run`**）
```

本篇对应的编号程序是 **`code/02_warp_execution.py`**（639 行）。它只依赖 `code/` 内的
`ptxsub.py` 与 `specs.py`，**不 import 任何 GPU / 框架依赖**，**不执行任何 PTX**（它**解释**手写的片段），
也不 spawn 任何进程。

**本机跑出的结果**（本仓库程序在本机跑出；测量条件是 `python3 3.13.5` + `numpy 2.4.4`，
Apple M1 Pro，**无 GPU**）：

```
汇总: A 7/7 · B 17/17 · C 10/10 ⇒ 断言 34 条，通过 34 条，失败 0 条
```

元检验（`make falsify`）里与本篇直接相关的四条缺陷：

```
[D1 bfly 的异或方向被改成加一]       02_warp_execution.py 退出 1  ← A.1 报「不匹配 124 个」
[D2 bar.sync 不阻塞（直接放行）]     02_warp_execution.py 退出 1  ← B.5c 报「有 bar.sync 时 0 个错误元素」不成立
[D3 到达计数按活跃 lane 记]          02_warp_execution.py 退出 1
[D7 .up 改成照抄 ISA 原文的 maxLane] 02_warp_execution.py 退出 1  ← A.4 报「up(...) 不一致」
```

## 参考

**本仓库程序（可复现）**

- `code/02_warp_execution.py` —— 本篇的编号程序（A/B/C 三类断言在程序内部，共 34 条）。
- `code/ptxsub.py` —— **非编号引擎**：`shfl.sync.{bfly,idx,up,down}` 的逐 lane 实现、
  `bar.sync` 的 warp 粒度到达计数、逐发射记账（`issue` / `lane_issue` / `divergent`）与 `trace_hook`。
- `code/specs.py` —— 9 条 A 级 ISA 锚点（`P1`/`P1b`/`P2`/`P3`/`P4`/`B1`/`B2`/`B3`/`B4`，**每条带章节号**）
  + `UP_MODE_UNDECIDED`（`.up` 的三条**不可判定**的可能，与八条**可判定**的量）。
- `code/meta_falsify.py` —— `make falsify`：7 种缺陷（含 `D7` 这条方向相反的）。
- `code/mutate.py` —— `make mutate`：叶子级变异测试 + 对照组。

**外部规范（A 级：厂商官方规范原文，逐字且机器抽取）**

- PTX ISA（本机取到 **3928316** 字节）：
  - **`9.7.10.6`** Data Movement and Conversion Instructions: `shfl.sync`
    （锚点 `P1`：语义伪代码；`P2`：`.up` 的例子 `cval = 0x0`；`P4`：`membermask` 的 UB 说明）
  - **`9.7.10.5`** 同章：`shfl`（**deprecated**）（锚点 `P1b` 伪代码、`P3` 引入/弃用版本）
  - **`9.7.15.1`** Parallel Synchronization and Communication Instructions: `bar, barrier`
    （锚点 `B1`：warp 粒度；`B2`：`marks warps' arrival`；`B3`：`in convergence`）
  - **`9.7.15.3`** 同章：`barrier.cluster`（锚点 `B4` —— ★ 它**不属于** `9.7.15.1`）

> 九条锚点的**逐字引文**保存在 `code/specs.py`，并由 `make anchors` 从**实际下载的官方 HTML** 里
> 再抽取一遍并逐字比对（本机跑出：**9/9 条一致**，含每条引文上面最近的 `<h4>` 标题）。
> ★ 其中 `P1` 与 `P1b` 是**两节**（`9.7.10.6` 用 `segmask`、`9.7.10.5` 用 `mask`）——
> 只引一节无法排除另一节写法不同。

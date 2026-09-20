---
title: 流与事件：默认流的栅栏是**星形**，不是全连接
slug: cuda-streams-and-events
domain: 30-gpu-programming
topic: cuda
level: 2
difficulty: intermediate
prerequisites:
  - 30-gpu-programming/cuda/01-thread-hierarchy-and-sync
tags: [cuda, streams, events, default-stream, legacy-barrier, dependency-dag, partial-order]
related:
  - 40-kernels/gemm/06-split-k-and-stream-k
  - 20-gpu-architecture/execution-model/03-occupancy-and-latency-hiding
  - 30-gpu-programming/cuda/03-memory-and-graphs
  - 60-compilers/inductor/01-graph-capture
code:
  - code/02_stream_dag.py
status: stable
updated: 2026-09-20
---

# 流与事件：默认流的栅栏是**星形**，不是全连接

> **你写两条流是为了让它们并发，而默认流会悄悄给它们之间插进栅栏。** 本篇把「一段多流程序里到底有哪些依赖」做成一个可执行的**依赖 DAG 模型**（`code/02_stream_dag.py`），它的输入是**真实文件**（`cuda-training-series` 的 hw10 / hw13），输出是逐条带出处标注的边。四条带数字的结论：
>
> 1. ★★ **默认流的栅栏是星形（各流 ↔ legacy 流），不是全连接。** `cuda-training-series` 的 hw10 里，一个落在 NULL 流上的 `cudaMemset`（第 95 行）的后继是两条阻塞流上**全部 12 个动作**，而 **`streams[0]` 与 `streams[1]` 之间的直接边是 0 条**。⇒ 「同步所有流」不等于「所有流互相同步」，这两个模型在全规模下相差 **767.5 倍**（3072 条边 vs 2357760 条边）。
> 2. ★★ **栅栏的出边是 O(后续动作数)，入边只有 O(流数)——它不对称。** 同一份程序里那个 `cudaMemset`：**出度 12，入度 1**。把它挪到分块循环之后（一个很容易记错的位置），**出度变成 0、入度变成 3**。同一个 `cudaMemset`，两个相反的位置，两个完全相反的图。
> 3. ★★ **`cudaStreamNonBlocking` 只在程序同时用到 legacy 流时才改变依赖图。** hw13 那段程序里一条 NULL 流动作都没有，所以把它换成 `cudaStreamCreate`（blocking）后 **DAG 逐条边完全相同**。⇒ 「用了 NonBlocking ⇒ 图一定不同」是错的；这个标志的作用域**由程序里有没有 legacy 动作决定**。
> 4. ★ **「可达」不是「直接边」，而 NVIDIA 那张图画的是传递归约（4 条边），不是传递闭包（5 条）。** 两者恰好差一条：`kernel_a → kernel_d` —— 它**可达**（`a→b→d`），但既不是直接边、也不在归约里。
>
> ★ **本域的三条限定语，在本篇的落点**（域 README 规定逐篇写明，不得省略）：
> 1. **「不是硬件」**——本模型只建模命令之间的**偏序依赖**，不建模执行时间、SM 占用、并发度、硬件队列深度。本程序**没有发射过一个 thread**。
> 2. **「分层输出只能读作『允许并发』，不能读作『实际并发』」**——§2.2 那张 6 层的表是**图论意义上的拓扑层**。出处（Guide §3.2.8.5）：「this behavior is **not guaranteed** and should therefore **not be relied upon for correctness**」。
> 3. **「所有时间数字本机不可得」**——hw10 的 readme 第 70 行说 `about 2X faster`、hw13 README 第 74 行说 `a small preformance increase`，**两个都没有倍数、都没有验证**，也不出现在本程序的任何输出里。

## 0. 背景与动机

### 0.1 一个具体的对照场景：两条流为什么没有并发

你在两条流上各发了一串工作，期望它们重叠。代码是这样的（`cuda-training-series` hw10 的 `streams_solution.cu`）：

```c
const int chunks = 64;                                    // 第 9 行
const int num_streams = 8;                                // 第 12 行
for (int i = 0; i < num_streams; i++) cudaStreamCreate(&streams[i]);   // 第 72–74 行，★ 没有 NonBlocking
...
cudaMemcpy(d_x, h_x, ds * sizeof(ft), cudaMemcpyHostToDevice);   // 第 86 行，NULL 流
gaussian_pdf<<<(ds + 255) / 256, 256>>>(d_x, d_y, 0.0, 1.0, ds); // 第 87 行，默认参数流 = NULL 流
cudaMemcpy(h_y1, d_y, ds * sizeof(ft), cudaMemcpyDeviceToHost);  // 第 88 行，NULL 流
cudaMemset(d_y, 0, ds * sizeof(ft));                             // 第 95 行，★ NULL 流
#pragma omp parallel for
for (int i = 0; i < chunks; i++) {                               // 第 100 行，depth-first
    cudaMemcpyAsync(... cudaMemcpyHostToDevice, streams[i % num_streams]);   // 第 101 行
    gaussian_pdf<<<...>>>(...);                                              // 第 102 行
    cudaMemcpyAsync(... cudaMemcpyDeviceToHost, streams[i % num_streams]);   // 第 103 行
}
cudaDeviceSynchronize();                                         // 第 105 行
```

**第 95 行的 `cudaMemset` 落在 NULL 流上**，而第 72–74 行的流是 `cudaStreamCreate` 建的（**blocking**）。于是按默认流的同步语义，那 64 × 3 = 192 个异步动作**全都要等这个 `cudaMemset` 完成**——它们本来是想重叠的。

**这就是那个「悄悄插进来的栅栏」**：你在两条流上排了工作，而一条**你没有显式创建**的流（默认流）把它们串起来了。

第二份素材是 hw13 的 fork/join：

```c
cudaStreamBeginCapture(streams[0], cudaStreamCaptureModeGlobal);   // 第 111 行
kernel_a<<<blocks, threads, 0, streams[0]>>>(d_x, d_y);            // 第 114 行
cudaEventRecord(event1, streams[0]);                                // 第 117 行
kernel_b<<<blocks, threads, 0, streams[0]>>>(d_x, d_y);            // 第 120 行
cudaStreamWaitEvent(streams[1], event1);                            // 第 123 行  ← fork
kernel_c<<<blocks, threads, 0, streams[1]>>>(d_x, d_y);            // 第 126 行
cudaEventRecord(event2, streams[1]);                                // 第 129 行
cudaStreamWaitEvent(streams[0], event2);                            // 第 132 行  ← join
kernel_d<<<blocks, threads, 0, streams[0]>>>(d_x, d_y);            // 第 135 行
cudaStreamEndCapture(streams[0], &graph);                           // 第 138 行
```

**这一段里一个 NULL 流动作都没有**——它全靠 event 显式建依赖。两份素材合起来正好覆盖了本篇的两条主线：**隐式栅栏的范围**（hw10）与**显式依赖的构造**（hw13）。

### 0.2 本篇讲什么、不讲什么

**讲**：把一段多流程序翻译成一个**带出处标注的依赖 DAG**，并把每条边的依据落到**一个可定位的外部来源**上（锚点 A1–A7，全部逐字取自 Driver API Reference 与 Programming Guide）。**本文只推导这 7 条锚点支撑的 5 条建边规则。**

**不讲**：硬件**怎么**调度这些队列（那是 `20-gpu-architecture`），以及 `cudaGraphLaunch` 之后会发生什么（那是本主题的 `03-memory-and-graphs`）。**本篇不出现任何算子的实现**。

### 0.3 本篇在本域主线上的位置

主线是「**你写的程序、编译器生成的东西、硬件执行的东西，是三个不同的程序**」。本篇守的是第一条断层里**第二个**泄漏点：

> 你写的**线程组织**（上一篇）能不能满足约束 → 编译器不查；
> 你写的**命令顺序**（本篇）到底变成什么依赖 → **运行时给你一个你没写过的图**。

## 1. 原理

### 1.1 七条锚点，五条规则

| 规则 | 依据的锚点 | 建立的边 |
|---|---|---|
| `RULE_STREAM_ORDER` | A5, A7 | 同一条流内的先后发射顺序（**程序序**） |
| `RULE_EVENT` | A5, A6, A7 | `cudaEventRecord` / `cudaStreamWaitEvent` 的**跨流边** |
| `RULE_LEGACY_IN` | A1, A2 | legacy 流上的动作**先等**所有 blocking 流（A2 **前半句**） |
| `RULE_LEGACY_OUT` | A2, A3 | **然后**所有 blocking 流等该 legacy 动作（A2 **后半句**） |
| `RULE_PERTHREAD` | A4 | per-thread default stream **不产生**任何隐式栅栏 |

**A2 是三步，不是一步**（逐字）：

> "When an action is taken in the legacy stream such as a kernel launch or cudaStreamWaitEvent(), **the legacy stream first waits on all blocking streams**, the action is queued in the legacy stream, **and then all blocking streams wait on the legacy stream**."

**「第一步」与「第三步」是两个不同的边集**：前者是 `legacy-barrier-in`（入边，数量 = 当时每条阻塞流上最后一个动作，**O(流数)**），后者是 `legacy-barrier-out`（出边，数量 = **之后**该 legacy 动作要挡住的全部动作，**O(后续动作数)**）。只读前半句会把「barrier」这个词最要紧的那一半丢掉——程序里有一条断言专门钉这件事（§5 的 #6）。

### 1.2 ★ 多步推导：为什么栅栏是**星形**，以及它与全连接差多少

**第一步（同步的双方是谁）**。锚点 A1 逐字：

> "The legacy default stream is an implicit stream which synchronizes with **all other streams in the same CUcontext except for non-blocking streams**, described below."

主语是 **legacy default stream**，宾语是 **所有其它流**。⇒ 每条边的一端**必定**是 legacy 流上的一次动作，另一端是另一条流上的一次动作。**「两条阻塞流之间」从来不是这条规则的双方**。

**第二步（一张图上的后果）**。设一段程序里有 `n` 个设备侧动作、`k` 条阻塞流。按 A2：

```text
legacy-barrier-in  的边数 ≤ k          （每条阻塞流最多贡献一条）
legacy-barrier-out 的边数 ≤ n          （每个后续动作最多收到一条）
⇒ 星形模型的边数 ≤ k + n = O(n)   （k ≤ n）
```

而如果（**错误地**）把「同步所有流」理解成「所有阻塞流互相同步」，边数是：

```text
全连接模型的边数 = n·(n − 1)      （任意两个动作之间都可能有一条边）
⇒ 星形 O(n) vs 全连接 O(n²) —— 阶就不同，不是一个常数因子的差别
```

**第三步（代进 hw10 的实际规模）**。hw10 全规模：`num_streams = 8`、`chunks = 64`、每个 chunk 3 个异步动作（第 101–103 行）。

```text
n = 8 × 64 × 3 = 1536   个设备侧动作
星形上界 2n = 3072
全连接   n(n−1) = 1536 × 1535 = 2357760
比值     2357760 / 3072 = 767.5
```

⇒ **两个模型在同一段程序上相差 767.5 倍。** 这不是一个抽象的差别：它决定了「你以为这两条流会重叠，还是不会」。

**第四步（本仓库程序算出来的那个具体数）**。为了能逐行复核，程序把 hw10 缩成一个**同构切片**（`chunks = 4`、`num_streams = 2`，每个流上的语句种类与顺序、以及 NULL 流动作的插入位置完全一样），得到 16 个设备侧动作、**25 条边**：

| 边类 | 条数 | 构成 |
|---|---|---|
| `program-order` | **13** | NULL 流 4 个动作 → 3 条；`streams[0]` 6 个动作 → 5 条；`streams[1]` 6 个动作 → 5 条 |
| `legacy-barrier-out` | **12** | 第 95 行的 `cudaMemset` 指向它**之后**两条流上**全部 12 个动作** |
| `legacy-barrier-in` | **0** | ★ 见下 |
| 两条阻塞流之间的直接边 | **0** | ★ 星形，不是全连接 |

**`legacy-barrier-in = 0` 不是 bug**：NULL 流那 4 个动作（第 86/87/88/95 行）全部出现在两条阻塞流**发射任何东西之前**，所以它们等不到任何东西。这是「入边 O(流数)」在**这个特定程序**上的取值——在这个程序里恰好是 0。

### 1.3 per-thread default stream：同一段程序的另一种解释

锚点 A4 逐字：

> "The per-thread default stream is an implicit stream local to both the thread and the CUcontext, and which **does not synchronize with other streams (just like explicitly created streams)**."

括号里那半句是本条的要害。⇒ 把 hw10 切片的 `mode:` 从 `legacy` 改成 `per_thread`：

```text
隐式栅栏边：12 条 → 0 条
同流程序序与 event 边：一条不少（13 条）
```

**同一段程序、同一个源码、只换一个编译标志**（`--default-stream per-thread`），依赖图的隐式部分整个消失。这就是编译标志与语义之间的耦合：**`--default-stream` 不是优化开关，它改变程序的语义。**

### 1.4 `cudaStreamNonBlocking` 的作用域**由程序决定**

锚点 A3 逐字：

> "Non-blocking streams which do not synchronize with the legacy stream can be created using the `cudaStreamNonBlocking` flag with the stream creation APIs."

注意它的条件：**这些流不与 legacy 流同步**。而栅栏（A2）**由 legacy 流上的动作触发**。⇒ 一个程序如果**从来不碰 legacy 流**，那么「哪些流是 NonBlocking」就**不改变任何一条边**。

**这是可证伪的**：把 hw13 那段（它一个 NULL 流动作都没有）的两条流从 `cudaStreamNonBlocking` 改成 `cudaStreamCreate`（blocking），DAG **逐条边完全相同**。程序里有一条断言钉住这一点（§5 的 #7），并且**故意构造了一个把 `is_blocking` 判定删掉的错误版本**，证明交叉校验能发现它（§5 的 #8）。

## 2. 执行映射

### 2.1 依赖图与「硬件队列」的关系

**参与的部件**：

| 层 | 部件 | 它做什么 |
|---|---|---|
| 宿主 | CUDA runtime 的**命令缓冲** | 把 `<<<>>>` / `cudaMemcpyAsync` / `cudaEventRecord` 追加到**某条流的队列** |
| 设备 | 每条流的**硬件命令队列** | 队列内**按序**执行；不同队列**可以**并行 |
| 设备 | **work distributor / GigaThread 引擎** | 把 grid 按 block 派给 SM |
| 设备 | SM 的 **block 槽位** | 决定同一时刻能驻留多少 block |

**★ 最关键的一句限定语**（锚点之外、但决定本模型能主张什么）——CUDA C++ Programming Guide **12.6.1 §3.2.8.5** Overlapping Behavior 原文：

> "this behavior is **not guaranteed** and should therefore **not be relied upon for correctness**"

⇒ **同一条流内是「按序」**（这是 A5/A7 给的硬保证）；**不同流之间「可以重叠」但不是承诺**。所以本模型的层次输出（§3.2 的 6 层）只能读作「**允许**并发」——它是一张**偏序图**，不是一个时间线。硬件是否真的让第 L 层的两个节点并行，本模型**一概没有验证**。

### 2.2 一个具体的数：hw13 的图有几层

hw13 那段（8 个设备侧动作、8 条直接边）的**拓扑层数 = 6**：

| 层 | 节点 |
|---|---|
| L0 | `n000` kernel_a |
| L1 | `n001` rec(event1) |
| L2 | `n002` kernel_b、`n003` wait(event1) |
| L3 | `n004` kernel_c |
| L4 | `n005` rec(event2)、`n007` kernel_d |
| L5 | `n006` wait(event2) |

**L2 的两个节点「允许」并发**（`kernel_b` 在 `streams[0]` 上、`wait(event1)` 在 `streams[1]` 上，两者之间没有边）。**L4 的两个节点也「允许」并发**（`rec(event2)` 与 `kernel_d`——它们分属两条流，且互不等待）。

⇒ 层数 6 是**图的性质**，不是**时间**。如果硬件把某个 L2 节点排到 L3 之后才跑，图不会变。**「层」这个名字容易让人以为是时间片**——它不是。

### 2.3 一个量化的对比：星形与全连接

| 模型 | 边数（hw10 全规模） | 隐含的判断 |
|---|---|---|
| **星形**（A1/A2 的原文） | `2n = 3072` | 各流**只**与 legacy 流同步 ⇒ 两条阻塞流**可以**重叠 |
| **全连接**（把「同步所有流」读成「互相同步」） | `n(n−1) = 2357760` | 两条阻塞流**必须**互等 ⇒ 重叠不可能 |

**比值 767.5**。这两个模型在同一段源码上给出**关于「能不能重叠」的相反判断**，而源码在这两种读法下**一个字都不变**。

## 3. 实现与演进

### 3.1 两条独立的实现路径（本篇的 A 类断言靠它）

模型不是只有一份实现——**两份**，而且是**结构不同**的两份：

| | `build()`（被检实现） | `build_reference()`（参照实现） |
|---|---|---|
| 复杂度 | **O(n)** 单趟 | **O(n²)** 逐对 |
| 做法 | 按发射顺序扫一遍，维护 `last_in_stream` / `last_record` / `last_legacy` 三个状态 | 对每一对 `(a, b)`（`a` 在 `b` 之前发射）**直接问**「按文档，这条边必须存在吗」——判据是「中间有没有人插队」的**全量扫描** |
| 出错时会怎样 | 状态更新漏一处 ⇒ 边集错 | —— |

**两者必须逐边逐类一致**（程序 A.5 断言，5 个 fixture 共 45 条边）。这是 `CONTRIBUTING` 第 214–217 行硬要求的落地：**参照实现必须走与被检代码不同的代码路径**。

还有**第三条**路径（A.5b）：用 numpy 的 **bool 矩阵幂** `R = R @ R` 再算一遍传递闭包，与集合式的 `reachability()`（邻接表上的不动点迭代）比对。

### 3.2 转写必须钉回源文件（否则模型只是「我自己写的程序」）

模型要读真实文件，但**它不能直接编译 `.cu`**（本机无工具链）。所以做法是：把真实文件的语句序列**转写**成一种小的文本格式（`<流> : <kind> <name>`），然后**用源文件本身把转写钉住**：

| 检查 | 做法 |
|---|---|
| `A.6a` | `specs.HW13_TASK1_OPS` 记了 10 行**「行号 + 逐字片段」**（第 111/114/117/120/123/126/129/132/135/138 行），逐条核对 |
| `A.6b` | 转写出的 **op 序列**必须等于源文件里这些行的**出现顺序**（**不经过模拟器**，只比「文本 → op」这一层） |
| `A.6c` | hw10 切片的循环部分由**源文件的行内容机械推导**出来（3 条调用的种类 + 流向 `streams[i % num_streams]`，把 64/8 切成 4/2），再与内嵌的转写逐项比对 |
| `A.6d` | hw10 的 NULL 流前缀 4 行的**语义**逐条命中（第 86/87/88/95 行） |

⇒ **转写漂了任何一行，程序就报错。** 这就是「真值来自外部工件」在本篇的具体执行方式——不是照抄工件的输出，而是**让工件反过来检查我的转写**。

### 3.3 演进：从「手写 event 依赖」到「把图交给 capture」

hw10 用的是**隐式栅栏**（完全没有 event），hw13 用的是**显式 event + capture**。这条路线的走向很清楚：

| 阶段 | 依赖怎么表达 | 问题 |
|---|---|---|
| 只用默认流 | 全靠隐式栅栏 | 栅栏范围**不由你控制**（hw10 的 192 个动作被串起来） |
| 显式 event | `cudaEventRecord` + `cudaStreamWaitEvent` | 依赖**显式**了，但每一次 fork/join 都要一对 API 调用（hw13 每迭代 8 个宿主动作） |
| stream capture | 写一遍普通的多流代码，让运行时**录下**依赖图 | 依赖图自动生成，replay 时宿主侧只剩 1 个动作（本主题 `03-memory-and-graphs`） |

**这条演进的方向是「减少宿主侧动作」**，不是「减少依赖」——依赖一条都没少，只是从「每次都重新建立」变成「建立一次、录下来」。**代价被搬到了别处**（capture 的合法性约束，见 `03-memory-and-graphs`）。

## 4. 踩坑与失败模式

### 4.1 把「可达」当成「直接边」

- **现象**：说「hw13 的图里 `kernel_a` 指向 `kernel_d`」——**错的**。`kernel_a → kernel_d` **可达**（`a→b→d` 与 `a→c→d` 都成立），但**不是直接边**。
- **成因**：图上「有箭头」这句话有三个不同的意思：**直接边**（8 条）、**传递闭包**（多出来的都是冗余的）、**传递归约 / cover**（去掉可由更短路径蕴含的边）。NVIDIA 在 `exercises/hw13/graph_stream_capture.png` 右侧画的是 **cover**——A/B/C/D 四个节点、**四条边**：`{a→b, a→c, b→d, c→d}`。
- **规避**：说「画了几个箭头」之前先问清是**归约**还是**闭包**。程序里有一条断言（§5 的 #5）钉住「`a→d` 可达、不是直接边、也不在 cover 里」这三件事**同时**为真。
- **★ 这条是真实的教训**：本仓库的探针 v1 在期望值里写了「`a→d` 是直接边」，断言失败后核对发现——**错的不是模拟器，是期望值**。

### 4.2 凭印象记错了发射顺序（`cudaMemset` 在循环**之前**）

- **现象**：以为 hw10 的 `cudaMemset` 在分块循环**之后**。按这个记忆建出来的 DAG 是**反的**。
- **成因**：一个「清理缓冲区」的 `cudaMemset` 在直觉上应该发生在工作之后。而源文件第 95 行明确在循环（第 100–104 行）**之前**。
- **规避**：**回到源文件数行**（程序 `A.6d` 做的正是这件事）。
- **★ 量化的后果**（程序 C.2 把它做成了可运行的反例）：

| `cudaMemset` 的位置 | 出度 | 入度 |
|---|---|---|
| 真实位置（第 95 行，循环**之前**） | **12** | **1** |
| 记错的位置（循环**之后**） | **0** | **3** |

**同一个 API 调用、两个相反的位置，出度从 12 变成 0。** ⇒ 只凭「legacy 流会同步」这句话是**推不出发射顺序**的。

### 4.3 把 A2 读成两句（丢掉「然后」那一步）

- **现象**：以为 legacy 栅栏只有「legacy 流等所有阻塞流」这一个方向。于是期望 hw13 以外的例子只有 2 条边。
- **成因**：A2 是一句**三步**的话。只抓前半句是「摘要式阅读」的典型产物——**前半句独立读起来是通顺的**，这正是它危险的地方。
- **规避**：把锚点的 `must_contain` **拆成两半**（`specs.ANCHORS['A2']['must_contain']` 就是这么写的），让机器能发现「只抄了前半句」这种漏写。程序里有一条断言（§5 的 #6）检查 `k_3` 的入边里**同时**有 `program-order` 与 `legacy-barrier-out` 两条——其中 `k_2 → k_3` 那一条**只**来自 A2 的第三步。

### 4.4 把 `cudaStreamNonBlocking` 与 per-thread default stream 混为一谈

- **现象**：以为「用了 NonBlocking 的流就完全不参与任何隐式同步」，或者以为「per-thread default stream 是一种 non-blocking 流」。
- **成因**：两个概念都含「不与 legacy 流同步」这层意思，但来源不同——**A3** 讲的是**你创建的流**（`cudaStreamNonBlocking`），**A4** 讲的是**默认流本身**（`--default-stream per-thread`）。A4 的原句里甚至给了反证：per-thread default stream **不是** non-blocking stream，**它会与 legacy default stream 同步**（如果两者同时被使用）。
- **规避**：`Program.is_blocking()` 里对三条路径**分开写**：legacy 流**自己不是** blocking stream（它是栅栏的发起者）、per-thread 默认流**是** blocking 的、显式创建的流按建流标志判定。★ 这是 P9「同名不同指」的高危点，程序里把这三条并列写成注释。

### 4.5 归属错误：把 legacy 栅栏规则当成 Programming Guide 的内容

- **现象**：正文写「CUDA 文档说默认流会同步所有流」，读者去 Programming Guide 里找**找不到**。
- **成因**：那条栅栏规则（A1/A2/A3/A4）**只在 CUDA Driver API Reference**（页名：`3. Stream synchronization behavior`）。Programming Guide §3.2.8.5.2 Default Stream 全文只讲 `--default-stream legacy/per-thread` 与 NULL stream 的存在，**没有那句栅栏语**。网上二手资料常把两者混起来。
- **规避**：**机械化**——`specs.LEGACY_FENCE_ANCHORS` 声明了这 4 条必须归 `driver`，程序有一条**归属断言**（A.8）同时检查**正向**（这 4 条的 `doc` 必须是 `driver`）与**反向**（driver 页的锚点**只允许是**这 4 条）。把任一 `doc` 改成 `guide`，`make run` 立刻失败。
- **一般化**：这是 P7「引文与断言作用域不匹配」的实例。**引 CUDA 语义时写文档名与章节号，不写「CUDA 文档说」。**

### 4.6 ★ 那个「别处」在哪里

本域主线说：**抽象层的价值与它的泄漏是同一件事的两面**。这一篇里，代价被藏到了哪里？

> **它被藏进了「我不需要显式同步」这句话里。**
>
> 用 `cudaStreamCreate` 建一条流、往里发工作、然后 `cudaDeviceSynchronize()`——**看起来你什么都没说**。没有 event，没有 wait，没有任何一行代码在表达「谁等谁」。这份「干净」是真实的：你确实不需要手写同步。
>
> 但依赖**没有消失**。它由**默认流的语义**替你说出来了，而那句话的双方、顺序、以及「哪些流参与」全都藏在 Driver API Reference 的一页里（不在 Guide 里，§4.5）。于是那个「别处」是：**你把依赖的表达权交给了默认流，而默认流给出的依赖比你想要的密得多**——hw10 那 192 个想重叠的动作，被一个落在 NULL 流上的 `cudaMemset`（第 95 行，在循环**之前**）串了起来。
>
> 同一条主线的第二种形态：**依赖图这个抽象本身也有泄漏**——「图里有多少箭头」有**三个**意思（直接边 / 闭包 / 归约，§4.1），而 NVIDIA 自己画的那张图用的是第三个。**抽象替你选了一个，且没告诉你是哪一个。**

## 5. 判据表

> ★ 按本域约定（域 README），§5 的位置与编号保留，内容**不是实测数据**而是**判据表**。每一行是「**可机械判定的条件 → 结论**」。前提是本机无 GPU，本主题**没有任何 GPU 实测数字**。

| # | 可机械判定的条件 | 结论 |
|---|---|---|
| 1 | 一段程序里有落在 legacy（NULL）流上的动作，且某条流是 blocking | 该 legacy 动作**先等**所有 blocking 流上各自的最后一个动作（A2 前半），**之后**所有 blocking 流上的动作又都要等它（A2 后半） |
| 2 | 程序里**一条 NULL 流动作都没有** | 建流标志（`cudaStreamCreate` vs `cudaStreamNonBlocking`）**不改变任何一条边**（§1.4，程序 `C.7`） |
| 3 | `--default-stream` 切到 `per-thread` | 隐式栅栏边**全部消失**（12 → 0），同流与 event 边**一条不少**（§1.3，程序 `C.4`） |
| 4 | 两条 blocking 流之间要判断「有没有互相等待」 | **0 条直接边**——它们只通过 legacy 流**间接**同步（程序 `C.3`） |
| 5 | 想知道 hw13 的 kernel 层面「画了几个箭头」 | **4 条**（`a→b, a→c, b→d, c→d`）——这是**传递归约（cover）**；**传递闭包** 5 条，两者恰好差 `a→d` 这一条（`a→b→d` 已蕴含它）。★ `b` 与 `c` 之间**两个方向都没有边**，闭包里也没有（程序 `C.5`） |
| 6 | `k_3`（在 blocking 流上、发射于 `k_2` 之后）的入边 | **两条**：`k_1 → k_3`（program-order）与 `k_2 → k_3`（**legacy-barrier-out**）。只读 A2 前半句会丢掉后者（§4.3，程序 `C.6`） |
| 7 | 一条 `wait(ev)` 之前没有同事件的 `record` | **无效依赖**——程序以 `ValueError` 退出（`C.1`）；真实 CUDA 里这个 event 没有任何 captured work（A7） |
| 8 | 同一段程序的两个 DAG 要比较 | 必须比**边的集合（带原因分类）**，不能只比节点数——hw13 的 blocking 与 nonblocking 版**节点与边都相同**（程序 `C.7`） |
| 9 | 想引用 legacy 栅栏规则的出处 | **CUDA Driver API Reference，页名 `3. Stream synchronization behavior`**——**不是** Programming Guide（§4.5，程序 `A.8` 正反双向钉住） |
| 10 | 想知道 per-thread default stream 是不是 non-blocking 流 | **不是**——A4 原句说它「is not a non-blocking stream and **will synchronize with the legacy default stream** if both are used」 |
| 11 | 把模型的层次输出读成「时间」 | **不允许**——Guide §3.2.8.5 原文：this behavior is **not guaranteed** and should therefore **not be relied upon for correctness**。只能读作「**允许**并发」 |
| 12 | 想引用 hw10 / hw13 的加速倍数 | **本机不可得**——hw10 readme 第 70 行只说 `about 2X faster`、hw13 README 第 74 行只说 `a small preformance increase`，**两个都没有倍数、都没有验证** |
| 13 | 把 `cudaStreamNonBlocking` 的效果推广到「任何用到多流的程序」 | **不允许**——它的作用域**由程序里有没有 legacy 动作决定**（§1.4） |
| 14 | 想用 hw10 的 `chunks = 64` / `num_streams = 8` 直接建图 | 可以，但**必须写明口径**：本篇为了逐行复核用的是**同构切片** `chunks = 4` / `num_streams = 2`，两者的每条流语句种类与顺序、NULL 动作插入位置完全一致 |

**闭合性检查**：读完这张表能判完「一段多流程序建出的依赖图对不对、层次输出能主张什么、出处该引哪里」。**唯一的表外输入**是「程序用了哪种 `--default-stream` 编译标志」——它不在源码里，必须**从构建配置查**。

## 6. 小结与延伸

- ★★ **legacy 栅栏是星形**：同步的双方是「legacy 流」与「其它流」，**不是**「任意两条阻塞流」（A1 原句）。hw10 切片里 `streams[0]` 与 `streams[1]` 之间**直接边 0 条**。
- ★★ **星形与全连接在同一段程序上相差 767.5 倍**（3072 vs 2357760 条边，hw10 全规模 `n = 1536`）。两个模型给出**相反的**「能不能重叠」判断，而源码一个字都不变。
- ★★ **栅栏不对称**：出边 O(后续动作数)、入边 O(流数)。hw10 里那个 `cudaMemset` **出度 12、入度 1**；把它记到循环之后，**出度 0、入度 3**。
- ★★ **`cudaStreamNonBlocking` 的作用域由程序决定**：没有任何 NULL 流动作时，换建流标志**不改变任何一条边**（程序 `C.7`）。
- ★ **`--default-stream per-thread` 让隐式栅栏边从 12 条变成 0 条**，而同流与 event 边一条不少（A4）。
- ★ **hw13 的 kernel 层归约是 4 条边**（`a→b, a→c, b→d, c→d`）、闭包 5 条，差的正是 `a→d`（**可达但不在归约里**）；拓扑层数 **6**。
- ★ **出处必须写 Driver API Reference 的 `3. Stream synchronization behavior` 页**——那条栅栏语**不在** Programming Guide 里（§4.5，程序 `A.8` 正反双向钉住）。
- ★ **模型的产出是推导，不是实测**：本机无 GPU，一个 thread 都没发射过。所有结论都是「按 `<出处>` 的规定」，不是「硬件实测如此」。

**延伸**：

- 多流切分在 GEMM 上的真实用法（本篇只讲依赖语义，不写算子）：`40-kernels/gemm/06-split-k-and-stream-k`。
- 占用率与延迟隐藏（「允许并发」在硬件上受什么限制）：`20-gpu-architecture/execution-model/03-occupancy-and-latency-hiding`。
- capture 如何把这张图录下来、以及 capture 的合法边界：`30-gpu-programming/cuda/03-memory-and-graphs`。
- 框架侧怎么捕获这张图（Inductor 的 graph capture）：`60-compilers/inductor/01-graph-capture`。
- 本域上一个主题（线程层次与同步契约）：`30-gpu-programming/cuda/01-thread-hierarchy-and-sync`。

## 编译与运行

```bash
cd 30-gpu-programming/cuda/code
make run                     # 跑本主题三个编号程序
make 02_stream_dag           # 只跑本篇对应的那一个
make show                    # 打印两个 DAG 的可读形式（★ 不是时间线）
make anchors                 # ★ 联网：从官方文档再抽取全部 22 条锚点并逐字比对
make mutate                  # 叶子级变异测试（非编号工具，**不进 `make run`**）
```

本篇对应的编号程序是 **`code/02_stream_dag.py`**。它**不 import** 任何 GPU / 框架依赖（只用 `numpy` 做一条独立的闭包复核），**不执行任何 CUDA**。

**它需要一棵外部源码树**（本域代码形态的「真值来自外部真实工件」）：`CUDA_TRAINING_SERIES` = NVIDIA 官方训练系列（**B 级**：厂商材料，不是规范），commit `9cd77d5`（2021-10-13，**无 release tag**）+ 每个文件的 sha256 前 16 位。程序按候选列表探测并**打印它选中的路径**；找不到时以**可操作的错误信息**退出，**不静默回落**：

```bash
make run CUDA_TRAINING_SERIES=/path/to/cuda-training-series
```

> ★ 注：`make` **不做** `~` 展开；请写 `$HOME/...` 而不是 `~/...`。

**三类断言**（全部在程序内部，失败即以非零码退出）：

- **A 类（15 条）**——`build()` 与 `build_reference()` **逐边逐类一致**（5 个 fixture、45 条边）；**numpy bool 矩阵幂**与集合式可达关系一致；hw13 的 10 行源文件逐字命中；hw13 的 op 序列与源文件出现顺序逐项相同；hw10 切片的循环部分由源文件**机械推导**并按 4 个 NULL 流源行核对语义；本程序覆盖的 7 条锚点的 `must_contain` 逐字命中；**归属断言**（A1–A4 必须归 `driver` 页，且该页只有这 4 条）；5 条建边规则引用的锚点都存在且在覆盖范围内；★ `A.10`/`A.10b` 把本程序 7 条锚点的 `covered_by` 与 `note` 里出现的标签**双向对齐**（改 `covered_by` 里任何一个标签，这一条立刻失败）。
- **B 类（4 条）**——星形 / 全连接的边数（3072 / 2357760，比值 767.5）；hw13 的拓扑层数（6）；memset 出度（12）；边构成（13 条 program-order + 12 条 legacy-barrier-out，**两条独立实现各算一遍**）。
- **C 类（9 条，★ 真的跑出问题）**——① `wait` 无 `record` ⇒ `ValueError`；② **把 `memset` 挪到循环之后** ⇒ 出度 12→0、入度 1→3；③ 两条阻塞流之间 0 条直接边；④ `per_thread` ⇒ 隐式边 0 条；⑤ **可达 ≠ 直接边 ≠ 归约**三件事同时为真；⑥ `k_3` 的入边里**同时**有 program-order 与 legacy-barrier-out；⑦ hw13 的 blocking/nonblocking **边集完全相同**；⑧ **故意删掉 `is_blocking` 判定** ⇒ 交叉校验必须发现（1 条边变 3 条）；⑨ 本程序声明的锚点不与上一篇重叠。
- ★ **程序不修改磁盘上的任何文件**：所有反例都在**内存里的文本**上构造。

**本机跑出的结果**（本仓库程序在本机跑出；测量条件是 `python3 3.13.5` + `numpy 2.4.4`，Apple M1 Pro，**无 GPU**）：

```
汇总: A 15/15 · B 4/4 · C 9/9 ⇒ 断言 28 条，通过 28 条，失败 0 条
```

`make anchors` 在本机跑出：**抓到 22 / 22 条锚点；不一致 0 条**（三条 A 族锚点逐字一致、归属为 Driver API Reference）。

## 参考

**本仓库程序（可复现）**

- `code/02_stream_dag.py` —— 本篇的依赖 DAG 模型（A/B/C 三类断言在程序内部；`--show` 打印可读形式；`--anchors-net` 联网再抽取锚点）。
- `code/specs.py` —— 22 条语义锚点（含 A1–A7）+ hw10/hw13 的行号 / 逐字内容 / sha256。

**外部工件（B 级：厂商训练材料，不是规范）**

- `cuda-training-series`，commit `9cd77d539d85ee3457e09a083ba299b787b7a16e`（2021-10-13，**无 release tag**）：
  - `exercises/hw10/streams_solution.cu` 第 9/12/72–74/86/87/88/95/100–104/105 行（**纯 legacy 栅栏、一个 event 都没有**）
  - `exercises/hw13/axpy_stream_capture_with_fixme.cu` 第 111/114/117/120/123/126/129/132/135/138 行（fork/join 的显式 event）
  - `exercises/hw13/graph_stream_capture.png` —— NVIDIA 自己画的 capture 图，**右侧四个节点、四条边**（★ **人眼读过**；本仓库程序读不了 PNG。hw13 README 第 15 行的 `This is the same example from the slides` 只说明它与课件同源，**不**支持「四条边」这个断言——两者必须分开写）
  - `exercises/hw10/readme.md` 第 70 行（`about 2X faster`，**无倍数、本机未验证**）

**外部规范（A 级：厂商官方文档原文，逐字且机器抽取）**

- CUDA **Driver API Reference** —— `3. Stream synchronization behavior`：锚点 `A1`（legacy default stream 的同步范围）、`A2`（**三步**栅栏）、`A3`（`cudaStreamNonBlocking`）、`A4`（per-thread default stream）。
  > ★★ **这一页是那 4 条锚点的唯一出处**——**不在** Programming Guide（§4.5）。程序 `A.8` 正反双向钉住这条归属。
- CUDA C++ Programming Guide **12.6.1** §**3.2.8.5.3** Explicit Synchronization（锚点 `A5`：`cudaStreamWaitEvent()` 的语义）；§**3.2.8.8** Events（锚点 `A6`：event 何时算完成）。
- CUDA Runtime API Reference —— `cudaStreamWaitEvent`（锚点 `A7`：`Makes all future work submitted to stream wait for all work captured in event.`）。
- CUDA C++ Programming Guide **12.6.1** §**3.2.8.5** Overlapping Behavior（§2.1 引的 `not guaranteed` / `not be relied upon for correctness`）。

> 锚点的**逐字引文**保存在 `code/specs.py`，并由 `make anchors` 从**实际下载的官方 HTML** 里正则**再抽取**一遍、逐字比对。抽取器**不手打任何一句引文**。

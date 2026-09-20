---
title: 内存管理与 CUDA Graph：宿主侧动作的账，与一个建流标志的边界
slug: cuda-memory-and-graphs
domain: 30-gpu-programming
topic: cuda
level: 2
difficulty: intermediate
prerequisites:
  - 30-gpu-programming/cuda/02-streams-and-events
tags: [cuda, cuda-graph, stream-capture, graph-replay, shared-memory, dynamic-shared-memory, host-api-overhead]
related:
  - 60-compilers/inductor/01-graph-capture
  - 20-gpu-architecture/memory-system/01-hierarchy-overview
  - 40-kernels/attention/04-warp-specialization-and-variants
  - 45-kernel-lifecycle/requirement-discovery/03-optimization-priority
code:
  - code/03_graph_capture.py
status: stable
updated: 2026-09-20
---

# 内存管理与 CUDA Graph：宿主侧动作的账，与一个建流标志的边界

> **CUDA Graph 省下来的是「宿主侧动作」，不是「kernel 执行」。** 本篇把 `cuda-training-series` hw13 的两个**真实文件**按行数出这笔账，并用一个显式的 capture 状态机复现 capture 的**合法边界**。四条带数字的结论：
>
> 1. ★★ **定时循环内的宿主侧动作：无 graph `8` 个/迭代 → 有 graph `1` 个/迭代（比值 8:1）；而 kernel 执行次数两边**都是 4 个/迭代**。** 1000 次迭代下来，省掉的是 **7000 个宿主侧动作**——**不是**「快了 8 倍」。hw13 的 README 第 74 行只肯说 `a small preformance increase`，**没有给倍数**，本机也没有 GPU 去测。
> 2. ★★ **一个建流标志决定 capture 是否合法。** 在 capture 区内插入一个落在 legacy 流上的动作：如果被捕获的流是 `cudaStreamNonBlocking` 建的 ⇒ `cudaSuccess`；如果不是 ⇒ `cudaErrorStreamCaptureInvalidated`。**唯一差别是那一个标志。** ⇒ hw13 第 **67** 行的 `cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking)` **不是随手写的**，它是这段 capture 合法性的前提。
> 3. ★ **一次违规废掉的是「所有关联的图」，不是「一部分」。** 违规时 `cudaStreamEndCapture` 返回 **NULL graph**（不是一张已经录了一部分的图）——出处原文用的是复数：`any associated capture graphs are invalidated`。
> 4. ★ **capture 图的 node 数是 4 还是 8，本机无法判定。** 只有三份互相吻合的**间接证据**指向 4。**证据吻合 ≠ 执行结果。**
>
> ⚠ **本机不执行任何 CUDA**（无 GPU、无 `nvcc`、无 libcublas）。上面的每一个数字都是**从源文件的行数出来的**（可对着 hw13 的文件逐行复核），或者**按官方原文判定的**——没有一个是硬件实测。

> ★ **本域的三条限定语，在本篇的落点**（域 README 规定逐篇写明，不得省略）：
> 1. **「不是硬件」**——本程序**不执行任何 CUDA**、不执行任何 kernel，也不碰 libcublas。它只做两件事：按行数出宿主侧动作、按官方原文判定 capture 期间的合法性。capture 出来的图能不能 replay、libcublas 是否支持被 capture，**本仓库一概没有验证**。
> 2. **「分层输出只能读作『允许并发』，不能读作『实际并发』」**——本篇**没有**分层输出（那是 `02-streams-and-events` 的事），所以这一条在本篇**自动成立**。同源的结论在这里表现为：**「宿主动作少了 8 倍」不能推出「时间少了 8 倍」**——本机没有 GPU，执行时间与并发度一概没测。
> 3. **「所有时间数字本机不可得」**——hw13 README 第 74 行只说 `a small preformance increase`，**没有倍数**；hw13 的 readme 里也没有任何时间数字。**本篇的全部数字都是「动作次数」，不是时间。**

## 0. 背景与动机

### 0.1 一个具体的对照场景：同一段工作，逐个 launch 与 capture 后 replay

`cuda-training-series` hw13 提供了**同一段工作**的两个版本。**逐个 launch 版**（`axpy_stream_capture_timer.cu`）的定时循环是第 134–158 行：

```c
for (int i = 0; i < 1000; ++i) {                                    // 第 134 行
    kernel_a<<<blocks, threads, 0, streams[0]>>>(d_x, d_y);          // 第 135 行
    cudaEventRecord(event1, streams[0]);                             // 第 138 行
    kernel_b<<<blocks, threads, 0, streams[0]>>>(d_x, d_y);          // 第 141 行
    cudaStreamWaitEvent(streams[1], event1);                         // 第 144 行
    kernel_c<<<blocks, threads, 0, streams[1]>>>(d_x, d_y);          // 第 147 行
    cudaEventRecord(event2, streams[1]);                             // 第 150 行
    cudaStreamWaitEvent(streams[0], event2);                         // 第 153 行
    kernel_d<<<blocks, threads, 0, streams[0]>>>(d_x, d_y);          // 第 156 行
}
```

**每一次迭代都要重新说一遍这个依赖图**——8 个宿主侧动作（4 次 kernel 发射 + 4 次 event/wait 调用）。

**capture 后 replay 版**（`Solutions/axpy_stream_capture_solution.cu`）的定时循环是第 166–170 行：

```c
for (int i = 0; i < 1000; ++i) {                                    // 第 166 行
    cudaGraphLaunch(instance, streams[0]);                           // 第 167 行
}
```

**一次迭代只有 1 个宿主侧动作。** 而依赖图本身是在**第一趟**录下来的（第 107–145 行的一个 `if (graphCreated == false)` 块里，第 110 行 `cudaStreamBeginCapture` … 第 137 行 `cudaStreamEndCapture` … 第 141 行 `cudaGraphInstantiate`）。

**这就是本篇要算的那笔账**：**8 : 1**。而它的分母与分子都是**宿主侧动作**——两边的 **kernel 执行次数完全一样**。

### 0.2 本篇讲什么、不讲什么

**讲**两件事：① **内存管理层次**里那条会咬人的规则（每 block 超过 48 KB 的静态共享内存必须改动态 + 显式 opt-in），以及它在**哪一代架构**上成立；② **CUDA Graph** 的收益从哪来（宿主侧动作的账）、以及 capture 的**合法边界**（哪些调用会毁掉整张图）。

**本文只推导这两条**（`CONTRIBUTING` 反模式 1 的判据）：① §1.1 的 48 KB 规则与它的作用域；② §1.2–§1.3 的收益账与 capture 边界清单。**其余都不在本文的推导范围内**——特别是下面这几件，本文**明确不做**：

| 不做的 | 去哪 |
|---|---|
| 具体算子的写法 | `40-kernels` |
| 框架怎么用这些 API 捕获计算图 | `60-compilers/inductor/01-graph-capture` |
| **node 数到底是几** | **本机判不了**（§4.2）——hw13 自己会在运行时 `printf` 它，本仓库跑不了 |
| `cudaGraphInstantiate` 花了多少时间 | **本机无 GPU**，测不了（§1.2 第四步） |
| libcublas 是否支持被 capture | **本机无 libcublas**，验证不了（§2.3） |
| `cudaGraphGetNodes` 是否把 child graph 展开计数 | 同上 |

### 0.3 本篇在本域主线上的位置

主线是「**你写的程序、编译器生成的东西、硬件执行的东西，是三个不同的程序**」。本篇守的是第一条断层里**第三个**泄漏点，而且它**与前两篇的方向相反**：

> 前两篇：**你写的**东西被**运行时**改写了（约束不查、依赖图不由你定）。
> 本篇：**你把描述权交给了运行时**（让 capture 录下依赖图），于是**换来一次新的、更严格的契约**——capture 区里有一整张「不许做的事」清单。

**「抽象层的价值与它的泄漏是同一件事的两面」**在本篇里是最清楚的一次：graph 省掉了逐次建立依赖的宿主开销，而代价是**你必须在 capture 期间放弃几乎所有的同步与查询**（§1.3）。

## 1. 原理

### 1.1 内存管理层次与那条 48 KB 规则

| 层 | 分配方式 | 谁来管 |
|---|---|---|
| 寄存器 | 编译器分配 | 编译器（超出会 spill 到 local memory） |
| **静态共享内存** | `__shared__ float tile[96][128];` | 编译期定尺寸，**编译进 kernel 的元数据** |
| **动态共享内存** | `extern __shared__ float tile[];` + launch 的第三个参数 | 运行期给定尺寸，**可以超过静态上限** |
| 全局内存 | `cudaMalloc`（设备）/ `cudaMallocHost`（锁页主机） | 你 |
| 统一/受管内存 | `cudaMallocManaged` | 驱动按需迁移 |

**那条会咬人的规则**（锚点 `C2`，逐字）：

> "**Compute capability 7.x devices** allow a single thread block to address the full capacity of shared memory: 96 KB on Volta, 64 KB on Turing. **Kernels relying on shared memory allocations over 48 KB per block are architecture-specific**, as such they **must use dynamic shared memory** (rather than statically sized arrays) and **require an explicit opt-in using `cudaFuncSetAttribute()`**"

**★ 引用这条时必须把第一句一起引**——它的作用域是 **7.x 设备**（Volta 96 KB / Turing 64 KB）。抽掉第一句，读者会以为「48 KB 是所有架构的通行上限」（这是 P7「引文与断言作用域不匹配」）。

**量级**（算出来的，可复核）：

```text
48 KB = 48 × 1024 = 49152 字节 = 49152 / 4 = 12288 个 float
96 × 128 × 4 = 49152    ⇒ 正好 48 KB，**不违规**（判据是 over 48 KB）
97 × 128 × 4 = 49664    ⇒ 违规
```

⇒ **判据的分界点落在一个 `float` 上**。这不是「大致超了就不行」，而是一条有精确边界的规则。

### 1.2 ★ 多步推导：graph 省掉的是什么，省不掉的是什么

**第一步（把「宿主侧动作」拆开）**。定时循环里的一次迭代，宿主线程要做的事有两类：

| 类 | 无 graph 版（第 134–158 行） | 有 graph 版（第 166–170 行） |
|---|---|---|
| CUDA runtime API 调用 | `cudaEventRecord` × 2 + `cudaStreamWaitEvent` × 2 = **4** | `cudaGraphLaunch` × 1 = **1** |
| kernel 发射（`<<<>>>`） | **4** | **0**（那 4 个 kernel 在图里） |
| **合计** | **8** | **1** |

**★ 口径必须写明**：`cudaCheckErrors(...)` 展开出的 `cudaGetLastError()` **不计**——它是诊断宏，每个调用后面都跟一个，计入会让两边**同时**翻倍（比值不变，绝对数失去意义）。而 **kernel 发射计入**（它是货真价实的宿主侧动作）。

> ⚠ **P9 提示**：计划与常见说法把 8 这个数叫「宿主 API 调用次数」，但它**含 kernel 发射**。严格说该叫「宿主侧动作」。本篇统一用后者，并在需要处把两类分开给出。

**第二步（比值与总量）**：

```text
无 graph：8 × 1000 = 8000 个宿主侧动作
有 graph：1 × 1000 = 1000 个宿主侧动作
比值 8000 / 1000 = 8 : 1
省下 8000 − 1000 = 7000 个宿主侧动作
```

**第三步（kernel 一侧的账）**：

```text
无 graph：每迭代 4 次 <<<>>> ⇒ 1000 迭代 = 4000 次 kernel 执行
有 graph：每迭代 4 个 kernel **在图里**，由 1 次 cudaGraphLaunch 一起发射 ⇒ 1000 迭代 = 4000 次 kernel 执行
⇒ 两边完全相同
```

**第四步（结论的边界）**。⇒ **收益完全是宿主侧的**。而「宿主侧动作少了 8 倍」**不蕴含**「时间少了 8 倍」：

- 省下的是**发起**的开销，不是**执行**的开销；
- 一次 `cudaGraphLaunch` 的 CPU 开销比一次 `<<<>>>` 大还是小、以及 4000 次 kernel 的执行时间，**本机都测不了**（无 GPU）；
- hw13 的 README 第 74 行自己就写了这个边界（逐字）："Given this, you should still be able to see **a small preformance increase** using the graph from the **launch overhead savings**. The **instantiation phase is not included in the timing** however, so it is not exactly a apple-to-apples comparison."

**★ 最后一句很重要**：`cudaGraphInstantiate`（第 141 行）的开销**不在计时区间内**（它在 `if (graphCreated == false)` 块里，而计时从第 164 行才起）。⇒ **这份比较本身就不是完全对等的**——README 自己说了「not exactly a apple-to-apples comparison」。**引用「8:1」时不能把它说成「加速 8 倍」，也不能说成「一次对等的基准测试」。**

### 1.3 capture 的合法边界：一整张「不许做的事」清单

capture 期间，被捕获的流**不是一条能执行的队列**——它正在被**录成一张图**。锚点 `B5a` 逐字给了理由：

> "It is invalid to synchronize or query the execution status of a stream which is being captured or a captured event, **because they do not represent items scheduled for execution**."

**★ 注意理由是句子自带的**（`because` 后面那半句），不是本篇编的解释。它**不能**改写成「因为会破坏图的完整性」之类——那是编机制从句。

由这条理由出发，五条边界**都能推出来**：

| 边界 | 依据锚点 | 为什么 |
|---|---|---|
| capture 不能**起始于** legacy 流 | `B3` | `Stream capture can be used on any CUDA stream except cudaStreamLegacy (the “NULL stream”).` |
| capture 期间**同步**一条被捕获的流 / 一个被捕获的 event **非法** | `B5a` | 它们那时不表示「已被排入执行」的东西 |
| capture 期间**查询**同上，**非法** | `B5a` | 同上 |
| capture 期间**同步 API**（`cudaMemcpy` / `cudaMemset`）**非法** | `B5c` | 它**归约到**前一条：`Synchronous APIs ... enqueue work to the legacy stream and synchronize it before returning.` |
| 违规 ⇒ **所有**关联的 capture 图**作废** | `B6` | `any associated capture graphs are invalidated`（复数） |

**还有两条「必须做」的**：

| 义务 | 依据锚点 | 内容 |
|---|---|---|
| 被 wait 的流**自动进入** capture，之后**必须 join 回 origin** | `B4a` / `B4b` / `B4c` | `Failure to rejoin to the origin stream will result in failure of the overall capture operation.` |
| 跨流依赖的 event 必须**录在同一张 capture 图**里 | `B4a` | `provided the event being waited upon was recorded into the same capture graph` |

**hw13 第 132 行 `cudaStreamWaitEvent(streams[0], event2)` 就是那个 join**——它看起来像一条普通的 event 边，实际是 capture 合法的**必需动作**。把它删掉，capture 整体失败（程序里有一条断言：`cudaErrorStreamCaptureUnjoined`）。

### 1.4 一句话总结 capture 的悖论

> **你让运行时替你把依赖录下来，代价是：在录的过程中，你不能对它做任何需要它「真的在跑」的事。**

这就是 §0.3 说的「方向相反的泄漏」。graph 抽象承诺「像调用一个函数一样发射一整段工作」，而它把代价藏在了一个**只有 capture 期间才会触发的慢路径**上——一张清单，清单上的每一条都会让**整张图**作废。

## 2. 执行映射

### 2.1 参与的部件与一次 replay 的路径

| 阶段 | 参与的部件 | 发生了什么 |
|---|---|---|
| 录制 | 宿主线程、被捕获流的**捕获状态** | 每个 API 调用被记成一个 node，而不是被排入执行队列 |
| 结束 | `cudaStreamEndCapture` | 返回一个 `cudaGraph_t`（**图的形状与参数**，还不能发射） |
| 实例化 | `cudaGraphInstantiate` | 把 `cudaGraph_t` 变成一个 `cudaGraphExec_t`（**可发射的实例**）——hw13 README 第 21 行明确区分这两个类型 |
| 发射 | `cudaGraphLaunch` | 把整张图交给设备；一次宿主调用对应 4 个 kernel |
| 执行 | work distributor / GigaThread、SM block 槽位、warp scheduler | 与逐 launch 版**完全相同**——图只是把「怎么发」提前算好了 |

**两个类型的区别值得单独记住**：`cudaGraph_t` 是**定义**（形状 + 参数），`cudaGraphExec_t` 是**可调用的实例**。README 第 21 行：

> "`cudaGraph_t` is used to define the shape and the arguments of the overall graph and `cudaGraphExec_t` is a callable instance of the graph, which has gone through the instantiate step."

⇒ **instantiate 是那条「藏在别处」的代价**：它不在定时区间里（第 141 行在 `if` 块内，计时从第 164 行起），但它确实要花时间。**README 自己承认这不是对等比较。**

### 2.2 一个具体的量级：同一段程序的两种 node 计数

capture 期间产生的**设备侧动作**有 **8 个**（4 个 kernel + 2 个 `record` + 2 个 `wait`）。但「图里有几个 node」有**两种口径**：

| 口径 | 数 | 依据 |
|---|---|---|
| **A**：`record` / `wait` 各算一个 node | **8** | §3.2.8.7.1.1 Node Types 里确实列了 `waiting on an event` 与 `recording an event` |
| **B**：`record` / `wait` **折叠成依赖边**、不产生 node | **4** | `B4b` 的后半句：`the next item in the stream will have **additional dependencies** on the nodes in the captured event`——**只加依赖、没加节点**；`B8` 说只有 external event node 才是显式节点 |

**三份间接证据都指向口径 B**：

1. `B4b` 的措辞（「additional dependencies」，不是「additional nodes」）；
2. `B8` 专门为 `external event node` 单列一条——如果普通 `record` 也成节点，这一条就没有存在意义；
3. 仓库里 NVIDIA 自己画的 `exercises/hw13/graph_stream_capture.png`，**右侧画出 A/B/C/D 四个节点、四条边**。★ 这是**人眼读过**的（本仓库程序读不了 PNG）。

**⇒ 但这是「证据吻合」，不是「执行结果」。** 本机无 GPU，`cudaGraphGetNodes` 返回几**判不了**。hw13 的 solution 文件第 155 行**真的会打印**这个数（`printf("Number of the nodes in the graph = %zu\n", numNodes);`）——**但本仓库跑不了它**。正文只能写「证据指向 4」，不能写「是 4」。

### 2.3 库调用也要被 capture：child graph

hw13 的 task2（`axpy_cublas_with_fixme.cu`）演示了另一条路：**把一个库调用录成一张子图**。它的结构是：

```c
cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);   // 第 38 行
cudaGraphCreate(FIXME, 0);                                    // 第 85 行，显式建图
cudaGraphAddKernelNode(&kernelNode1, graph, ...);             // 第 102 行，kernel_a
cudaStreamBeginCapture(stream1, cudaStreamCaptureModeGlobal);  // 第 109 行
cublasSaxpy(cublas_handle, N, &d_a, d_x, 1, d_y, 1);           // 第 113 行  ← 被 capture 的库调用
cudaStreamEndCapture(stream1, &libraryGraph);                  // 第 116 行
cudaGraphAddChildGraphNode(FIXME, graph, FIXME, 1, libraryGraph); // 第 120 行，把子图挂进主图
cudaGraphAddKernelNode(&kernelNode2, graph, ...);              // 第 135 行，kernel_c
cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);         // 第 149 行
```

**两种建图方式混用**：显式 API（第 102/120/135 行）与 stream capture（第 109–116 行）。README 第 79 行（逐字）解释了它们的关系：

> "A key to this example is remembering while we are using both explicit graph creation and stream capture, **both are just ways of defining to a `cudaGraph_t`** which we then instantiate into a `cudaGraphExec_t`."

README 第 81 行给了期望的 node 数：

> "We are creating **2 kernel nodes and a child graph** derived from a cuBLAS axpy function call."

⇒ 顶层应是 **3 个 node**。**★ 但 `cudaGraphGetNodes` 会不会把 child graph 展开计数、以及 libcublas 是否真的支持被 capture，本机都验证不了**（无 GPU、无 libcublas）。本仓库只复现「按文档，这个调用在 capture 区里是合法的」这一步。

## 3. 实现与演进

### 3.1 两条独立的计数路径（本篇的 A 类断言靠它）

「8 个/迭代」这个数从**两条完全不同的路径**得到，两者必须一致：

| 路径 | 做法 |
|---|---|
| **被检实现** | `Recorder` 状态机回放**转写出来的调用序列**，按相位计数（`timed_loop` / `launch_loop` / …） |
| **参照实现** | `count_calls_in_range()` 直接**正则扫源文件的行范围**：数 `cuda[A-Z]\w*(`（**排除**诊断宏 `cudaCheckErrors`）与 `<<<` |

两条路径的一致性**只能**由「转写没漂」解释。而且路径 2 还带来一个额外的检查（程序 `A.1c`）：**有 graph 的定时循环里，`cuda*` 调用只有 `cudaGraphLaunch` 一种，且一个 `<<<` 都没有**。

> ★ **测量条件必须写明**：源文件里的 `for` 循环体只写了一次，所以正则扫出来的是**「一次迭代」的量**；`Recorder` 回放 `iter=1000` 遍，得到 8000。两者相除才得到「每迭代」。

### 3.2 七种变体的判定表

capture 的边界不是靠读文字记住的——本篇把它做成一张**判定表**，每一次只改一处：

| # | 在 capture 区内插入 | 期望的 `cudaStreamEndCapture` |
|---|---|---|
| 1 | 什么都不插 | `cudaSuccess` |
| 2 | `cudaStreamSynchronize(streams[0])` | `cudaErrorStreamCaptureInvalidated` |
| 3 | `cudaDeviceSynchronize()` | `cudaErrorStreamCaptureInvalidated` |
| 4 | `cudaEventSynchronize(event1)` | `cudaErrorStreamCaptureInvalidated` |
| 5 | 同步 `cudaMemcpy`（D2H） | `cudaErrorStreamCaptureInvalidated` |
| 6 | 一个 legacy 流动作，**被捕获的流是 NonBlocking 的** | **`cudaSuccess`** |
| 7 | 一个 legacy 流动作，**被捕获的流是 blocking 的** | `cudaErrorStreamCaptureInvalidated` |

**第 6 与第 7 行只差一个建流标志**（A3 与 B5b）。这就是本篇的反直觉量化结论：**读源码时你会看到一个「多余的」`cudaStreamNonBlocking`，而它是这段 capture 合法性的唯一前提。**

### 3.3 演进：从「每次重新说一遍」到「说一遍、录下来」

| 阶段 | 依赖怎么建立 | 每次迭代的宿主侧动作 |
|---|---|---|
| 只用默认流（hw10） | 全靠隐式栅栏 | 无 event，但栅栏**不由你控制** |
| 显式 event（hw13 timer，第 134–158 行） | 每次迭代重发一遍 event/wait | **8** |
| capture + replay（hw13 solution，第 166–170 行） | 第一趟录一次，之后只 launch | **1** |
| 显式 graph API + child graph（hw13 cublas） | 手搭节点、把库调用录成子图 | **2**（launch + synchronize，100 次迭代） |

**方向是「把建立依赖的次数从 N 次降到 1 次」**，而代价是 §1.3 那张清单。**依赖一条都没少**——只是从「每次都重新建立」变成「建立一次、录下来」。

## 4. 踩坑与失败模式

### 4.1 把「宿主侧动作少了 8 倍」读成「快了 8 倍」

- **现象**：看到 8:1 就写成「graph 带来 8 倍加速」。
- **成因**：省下的是**发起**的开销。**kernel 执行次数两边完全相同（4 : 4）**；而 `cudaGraphInstantiate` 的开销**不在计时区间内**（第 141 行在 `if` 块里，计时从第 164 行起）——README 自己说 `not exactly a apple-to-apples comparison`。
- **规避**：把它说成「**宿主侧动作** 8 : 1」，并同时写出 hw13 README 第 74 行那句（`a small preformance increase`，**没有倍数**）。**任何时间数字本机都不可得。**
- **★ 这是「把预测写成实测」的典型入口**（`CONTRIBUTING` 反模式 4）：8:1 是**数出来的**，加速倍数是**没测过的**。两者在文字上只差几个字。

### 4.2 把「node 数是 4」说成已确定

- **现象**：正文写「这张图有 4 个 node」。
- **成因**：4 来自**三份间接证据**（§2.2），不是执行结果。而 hw13 的 solution 文件第 155 行**真的会打印**这个数——**但本仓库跑不了它**（无 GPU）。
- **规避**：写成「三份证据都指向 4；本机无法判定」。★ 这一条是 P12（自造真值）最容易命中的地方：程序自己算出一个数，就以为它是真值——**而真相是 hw13 自己会在运行时打印它，我们只是看不见**。
- **量化的反差**：两种口径给 **8** 与 **4** ——**差 2 倍**，而这个差别完全来自「`record`/`wait` 算不算节点」这**一个约定**。

### 4.3 忽略第 67 行那个建流标志

- **现象**：读 hw13 时把那行 `cudaStreamCreateWithFlags(..., cudaStreamNonBlocking)` 当成「随手写的」；或者自己写 capture 时用 `cudaStreamCreate` 建流，然后在 capture 区里插了一个 legacy 流动作，得到一个 `cudaErrorStreamCaptureInvalidated` 而**找不到原因**。
- **成因**：`B5b` 的判定条件是「**and it was not created with cudaStreamNonBlocking**」。`not` 在中间——把 `not` 读掉，结论整个反过来。
- **规避**：捕获区里**只用 NonBlocking 流**（或干脆不碰 legacy 流）。程序把这一条做成**正反一对**断言（§5 的 #6/#7）。
- **★ 为什么它值一条失败模式**：这是本篇唯一一条「**一个字符的差别决定程序是否合法**」的规则，而它在源码里表现为一个**看起来多余的标志**。

### 4.4 把 task2 的发射循环当成「定时循环」

- **现象**：把 task1 的「8 : 1」推广到 task2，或者把 task2 的 100 次循环当成 1000 次。
- **成因**：hw13 的两个 task **看起来同构**（都是「循环里发一串工作」），但 task2 的第 153 行是 `for (int i = 0; i < **100**; ++i)`，而且 **task2 里根本没有 `std::chrono`**（没有计时）。它的循环体是 `cudaGraphLaunch` + **`cudaStreamSynchronize`**（每趟同步！），共 **2 个宿主侧动作/迭代**。
- **规避**：**回到源文件数行**。程序里有一条断言把这件事钉住（§5 的 #5、#10）。★ 这条是实施时**从源文件读出来的**——计划与探针把 task2 的循环也叫「定时循环」，**那是错的**。
- **一般化**：这是 P10（单一实践升格为通用规则）的入口——**「8 : 1」只对 task1 成立**。

### 4.5 忘掉 join，capture 整体失败

- **现象**：`cudaStreamWaitEvent(streams[0], event2)`（第 132 行）被当成「一条多余的 event 边」删掉，`cudaStreamEndCapture` 返回 `cudaErrorStreamCaptureUnjoined`。
- **成因**：`B4b` 前半句说「被 wait 的流**自动进入** capture」，`B4c` 说「**必须 join 回 origin**，否则整体失败」。⇒ 一旦你用 event 把第二条流拉进 capture，你就**欠了一个 join**。
- **规避**：capture 区结束时，每条被拉进来的流都要有一条「回到 origin 流上等它」的 wait。hw13 第 131–132 行正是这个动作（`cudaStreamWaitEvent(streams[0], event2)`——注意它等的 event 是从 `streams[1]` 上 record 的）。

### 4.6 把 `cudaStreamCaptureModeRelaxed` 当成「安全总开关」

- **现象**：以为 `Relaxed` 模式下什么都能做，于是在 capture 期间 `cudaEventQuery` 一个被捕获的 event。
- **成因**：锚点 `B9c` 逐字：`The local thread is **not** prohibited from potentially unsafe API calls. Note that **the thread is still prohibited** from API calls which necessarily conflict with stream capture, **for example, attempting cudaEventQuery on an event that was last recorded inside a capture sequence**.`
- **规避**：读全句。**它举的例子恰好就是 `cudaEventQuery`** ⇒ 这一条**不看 mode**，一律违规（程序 `C.5`）。
- **★ 为什么它藏得住**：`Relaxed` 这个名字本身就暗示「更宽松」，而原文的第二个句子用 `still prohibited` 把这条豁免**又收了回去**。**只读第一个句子就会得到相反的结论。**

### 4.7 ★ 那个「别处」在哪里

本域主线说：**抽象层的价值与它的泄漏是同一件事的两面**。这一篇里，代价被藏到了哪里？

> **它被藏进了「capture 是一段普通代码」这句话里。**
>
> 你写的那段 capture 区，**和 segment 之外的多流代码长得一模一样**——同样的 `<<<>>>`、同样的 `cudaEventRecord`、同样的 `cudaStreamWaitEvent`。没有一行代码在说「现在进入了一个特殊区域」。而恰恰在这段「看起来一样」的代码里：
>
> - **`cudaStreamBeginCapture` 改变了每一个后续调用的含义**——它们从「排入执行队列」变成「被记成图的 node」；
> - **一整类调用从合法变成非法**（§1.3 的清单），而**编译器一条都不查**（这与 `01-thread-hierarchy-and-sync` 的结论同源：这些约束在一般情况下不可判定）；
> - **一个建流标志**（`cudaStreamNonBlocking`）成了整段代码合法性的唯一前提，而它在源码里离 capture 区有 40 多行（第 67 行 vs 第 110 行）。
>
> ⇒ **那个「别处」是「capture 区的边界」**——它不由语法标记，由**一次 API 调用与一次 API 调用之间**的隐式状态决定。你的程序里没有任何一个字符在提示你「这里开始是特殊区域」。
>
> 同一条主线的第二种形态：**`cudaGraph_t` 与 `cudaGraphExec_t` 是两个不同的东西**（定义 vs 可调用实例），而 `cudaGraphInstantiate` 这个「把定义变成实例」的代价**被排除在了计时之外**。README 自己承认这不是对等比较（§1.2 第四步）。

## 5. 判据表

> ★ 按本域约定（域 README），§5 的位置与编号保留，内容**不是实测数据**而是**判据表**。每一行是「**可机械判定的条件 → 结论**」。前提是本机无 GPU、无 `nvcc`、无 libcublas，本主题**没有任何 GPU 实测数字**。

| # | 可机械判定的条件 | 结论 |
|---|---|---|
| 1 | 定时循环内数宿主侧动作（含 kernel 发射，不含 `cudaCheckErrors`） | 无 graph **8**/迭代、有 graph **1**/迭代 ⇒ **8 : 1**；**kernel 执行次数两边都是 4/迭代** |
| 2 | 1000 次迭代的总账 | 8000 → 1000，省 **7000** 个宿主侧动作。★ 这是**调用次数**，**不是时间** |
| 3 | 想把这个 8 : 1 说成「加速倍数」 | **不允许**——README 第 74 行只说 `a small preformance increase`，**无倍数**；本机无 GPU，时间数字不可得 |
| 4 | 想把这个结论用到 task2 上 | **不允许**——task2 的循环是 `for (i < 100)`（第 153 行）、每迭代 **2** 个动作（launch + synchronize）、且**没有计时**（无 `std::chrono`） |
| 5 | 想知道 `cudaGraphInstantiate` 的开销在不在计时里 | **不在**——它在第 141 行、位于 `if (graphCreated == false)` 块内；计时从第 164 行起。README 自己说 `not exactly a apple-to-apples comparison` |
| 6 | capture 区内插入同步 / 查询（stream、device、event） | **`cudaErrorStreamCaptureInvalidated`**（`B5a`，理由是 `because they do not represent items scheduled for execution`） |
| 7 | capture 区内插入同步 API（`cudaMemcpy` / `cudaMemset`） | **`cudaErrorStreamCaptureInvalidated`**（`B5c`——它**归约到** `B5b`，不是一条独立规则） |
| 8 | capture 区内插入 legacy 流动作，**被捕获的流是 NonBlocking 的** | **`cudaSuccess`**（`B5b` 的 `not` 条件不成立） |
| 9 | 同上，但那条流是 blocking 的 | **`cudaErrorStreamCaptureInvalidated`**。★ #8 与 #9 **只差一个建流标志** |
| 10 | capture **起始于** legacy 流 | **`cudaErrorStreamCaptureInvalidated`**（`B3`）——与 #8/#9 是**两条不同的规则** |
| 11 | 被 wait 的流没有 join 回 origin | **`cudaErrorStreamCaptureUnjoined`**（`B4c`） |
| 12 | 一次违规之后 `cudaStreamEndCapture` 返回什么 | **NULL graph**，不是「部分图」（`B6`：`any associated capture graphs are invalidated`，复数） |
| 13 | capture 期间 `cudaEventQuery` 一个被捕获的 event，且 mode 是 **Relaxed** | **仍然违规**（`B9c` 的第二个句子，且它举的例子**恰好**是 `cudaEventQuery`） |
| 14 | capture 期间 `cudaEventQuery`，mode 是 Global | **违规**（`B9b`：`this thread is prohibited from potentially unsafe API calls`） |
| 15 | 数 capture 图的 node 数 | 口径 A（`record`/`wait` 各算一个）= **8**；口径 B（折叠成依赖边）= **4**。**本机无法判定运行时给哪个**，三份间接证据指向 B |
| 16 | 静态 `__shared__` 合计 **over 48 KB** | 必须改**动态共享内存** + `cudaFuncSetAttribute()` 显式 opt-in（锚点 `C2`）。★ 该锚点原句的作用域是 **7.x 设备**（Volta 96 KB / Turing 64 KB），引用时必须一起引 |
| 17 | 静态 `__shared__` 合计**正好等于** 48 KB（49152 字节 = 96×128×4） | **不违规**——判据是 **over** 48 KB；边界落在**一个 `float`** 上 |
| 18 | 引用 `cuda-learning` 的代码 | **必须同时说明它的目标环境是 Tesla T4 / sm_75（Turing）**，**没有** bf16 / TF32 / `cp.async` / TMA（其 README 第 16–17 / 22 行） |
| 19 | 引用 CUDA 的 capture 限制 | 写**文档名与章节号**（Programming Guide **§3.2.8.7.3** / **§3.2.8.7.3.1** / **§3.2.8.7.3.2** / **§3.2.8.7.3.3**），**不写「CUDA 文档说」** |
| 20 | libcublas 是否真的支持被 capture、`cudaGraphGetNodes` 是否把 child graph 展开计数 | **本机无法验证**（无 GPU、无 libcublas）——hw13 task2 的 README 描述是**成功案例**，仅此而已 |

**闭合性检查**：读完这张表能判完「这段 capture 代码合不合法、graph 的收益怎么描述、哪些数是本机判不了的」。**唯一的表外输入**是「本机装的是哪一代 GPU」——它不进源码，必须**从运行环境查**（而本仓库**没有** GPU）。

## 6. 小结与延伸

- ★★ **8 : 1 是宿主侧动作的比值**：无 graph 每迭代 8 个（4 次 kernel 发射 + 4 次 event/wait），有 graph 每迭代 **1** 个（`cudaGraphLaunch`）。**kernel 执行次数两边都是 4/迭代。** 1000 次迭代省 **7000** 个宿主侧动作。
- ★★ **一个建流标志决定 capture 是否合法**：`cudaStreamNonBlocking` ⇒ legacy 流动作给 `cudaSuccess`；`cudaStreamCreate`（blocking）⇒ `cudaErrorStreamCaptureInvalidated`。**唯一差别就是那个标志**（锚点 `B5b` 的 `not` 条件）。
- ★ **一次违规废掉所有关联的图**：返回 **NULL graph**（`B6` 原文用复数 `any associated capture graphs`）。
- ★ **capture 的完整清单**：不能起始于 legacy 流（`B3`）；不能同步/查询被捕获的流或 event（`B5a`）；不能调同步 API（`B5c`，它**归约到** `B5b`）；被 wait 的流**自动进入** capture 且**必须 join 回 origin**（`B4b`/`B4c`）；`Relaxed` 模式**不是豁免**（`B9c` 举的例子恰好是 `cudaEventQuery`）。
- ★ **node 数是 4 还是 8 本机判不了**：两种口径差 2 倍，三份**间接证据**指向 4。**证据吻合 ≠ 执行结果。**
- ★ **`cudaGraphInstantiate` 的开销不在计时区间内**（第 141 行 vs 第 164 行）——README 自己承认 `not exactly a apple-to-apples comparison`。
- ★ **48 KB 那条规则的作用域是 7.x 设备**（锚点 `C2` 原句以 `Compute capability 7.x devices …` 开头）；判据是 **over** 48 KB，边界落在**一个 `float`** 上（49152 不报、49664 报）。
- ★ **hw13 task2 不是定时循环**：`for (i < 100)`、每迭代 2 个动作、**没有计时**。★ 这条是从源文件读出来的——计划与探针把它也叫「定时循环」，**那是错的**。
- ★ **`cuda-learning` 的目标环境是 T4 / sm_75（Turing）**——引用它的代码时必须同时说明。

**延伸**：

- 框架侧怎么用这些 API 捕获计算图（Inductor 的 graph capture）：`60-compilers/inductor/01-graph-capture`。
- 存储层次本身（本篇只讲那条 48 KB 规则，不重推层次）：`20-gpu-architecture/memory-system/01-hierarchy-overview`。
- 什么时候值得为「宿主侧开销」做优化（收益判断的方法论）：`45-kernel-lifecycle/requirement-discovery/03-optimization-priority`。
- warp specialization 等把「让渡控制权」推到更远的方案：`40-kernels/attention/04-warp-specialization-and-variants`。
- 本域上一个主题（依赖图与默认流栅栏）：`30-gpu-programming/cuda/02-streams-and-events`。

## 编译与运行

```bash
cd 30-gpu-programming/cuda/code
make run                     # 跑本主题三个编号程序
make 03_graph_capture        # 只跑本篇对应的那一个
make mutate                  # 叶子级变异测试（非编号工具，**不进 `make run`**）
```

本篇对应的编号程序是 **`code/03_graph_capture.py`**。它**不 import** 任何 GPU / 框架依赖（只用 `numpy` 做变体码表与比值复核），**不执行任何 CUDA、不执行任何 kernel**。

**它需要一棵外部源码树**（本域代码形态的「真值来自外部真实工件」）：`CUDA_TRAINING_SERIES` = NVIDIA 官方训练系列（**B 级**：厂商材料，不是规范），commit `9cd77d5`（2021-10-13，**无 release tag**）+ 每个文件的 sha256 前 16 位。程序按候选列表探测并**打印它选中的路径**；找不到时以**可操作的错误信息**退出：

```bash
make run CUDA_TRAINING_SERIES=/path/to/cuda-training-series
```

> ★ 注：`make` **不做** `~` 展开；请写 `$HOME/...` 而不是 `~/...`。

**三类断言**（全部在程序内部，失败即以非零码退出）：

- **A 类（16 条）**——**两条独立路径**给出同一个数：`Recorder` 回放（内存里的转写）vs `count_calls_in_range()`（**正则扫源文件第 134–158 / 166–170 行**）；有 graph 的循环里 `cuda*` 调用**只有一种**、且 `<<<` 计数为 0；两个版本的 kernel 发射次数/迭代相同；solution 的 **6 处**与 cublas 的 **9 处**「行号 + 逐字内容」命中；本程序覆盖的 **12 条**锚点的 `must_contain` 逐字命中；**七种变体**逐行与规格表一致；**最后两行只差一个建流标志**，**且那个标志在源文件第 67 行逐字命中**；capture 区只对第一趟生效（`graphCreated` 标志）；task2 的库调用被 capture；★ 三个循环的**行范围与迭代数**（timer / graph / task2 共 9 个格子）由源文件**机械推导**（找 `for` 那一行 + 手写括号配平找闭合行 + 从 `i < N` 取迭代数）后与 `specs.py` 的常量逐项比对；每迭代的动作数四个常量与实测一致；★ `A.9`/`A.9b` 把本程序 12 条锚点的 `covered_by` 与 `note` 里出现的标签**双向对齐**（改 `covered_by` 里任何一个标签，这一条立刻失败）。
- **B 类（5 条）**——8 : 1 与 8.0/1.0 次每迭代；kernel 执行次数相同（4 : 4）；省下 **7000**；task2 的循环是 **100** 次、每次 **2** 个动作；node 数的两种口径（**8** 与 **4**）。
- **C 类（9 条，★ 真的跑出问题）**——① 变体表用 `int8` 码表**逐格**比对；② 只有 2 种变体合法、5 种失效；③ 去掉 `cudaStreamNonBlocking` ⇒ 变体 6 从合法变失效；④ capture 起始于 legacy 流 ⇒ 违规（`B3`）；⑤ 不 join 回 origin ⇒ `cudaErrorStreamCaptureUnjoined`（`B4c`）；⑥ **Relaxed** 模式下 `cudaEventQuery` **仍然违规**（`B9c`）；⑦ 违规 ⇒ 返回 **NULL graph**（`B6`）；⑧ 被 wait 的流**自动进入** capture（`B4b` 前半句）；⑨ task1 的定时循环**无同步**、task2 的发射循环**每趟都同步**（§4.4 的可运行证据）。
- ★ **程序不修改磁盘上的任何文件**：所有数字都从**读**源文件得到。

**本机跑出的结果**（本仓库程序在本机跑出；测量条件是 `python3 3.13.5` + `numpy 2.4.4`，Apple M1 Pro，**无 GPU**）：

```
汇总: A 16/16 · B 5/5 · C 9/9 ⇒ 断言 30 条，通过 30 条，失败 0 条
```

## 参考

**本仓库程序（可复现）**

- `code/03_graph_capture.py` —— 本篇的 capture 状态机与宿主侧动作计数（A/B/C 三类断言在程序内部）。
- `code/specs.py` —— 22 条语义锚点（含 `B3`–`B9c` 的 12 条）+ hw13 的文件行号 / 逐字内容 / sha256 + 七种变体表。

**外部工件（B 级：厂商训练材料，不是规范）**

- `cuda-training-series`，commit `9cd77d539d85ee3457e09a083ba299b787b7a16e`（2021-10-13，**无 release tag**）：
  - `exercises/hw13/axpy_stream_capture_timer.cu` 第 **134–158** 行（逐 launch 的定时循环，1000 次）+ 第 97 行（`blocks` 表达式）
  - `exercises/hw13/Solutions/axpy_stream_capture_solution.cu` 第 **110/137/141/147/149/167** 行（capture 区边界、instantiate、定时循环里的唯一一条 launch）+ 第 104 行
  - `exercises/hw13/axpy_cublas_with_fixme.cu` 第 **38/85/109/113/116/120/149/155/156** 行（task2：库调用 capture + child graph；★ 循环是 `i < 100`，**没有计时**）
  - `exercises/hw13/README.md` 第 **15** / **19** / **21** / **74** / **79** / **81** 行（逐字引用；★ 第 74 行的拼写 `preformance` 是**原文如此**）
  - `exercises/hw13/graph_stream_capture.png` —— NVIDIA 自己画的 capture 图（★ **人眼读过**：右侧四个节点、四条边）
- `cuda-learning`（**不是 git 仓库**，只能用文件 sha256）：`README.md` 第 **16–17** 行（目标环境 **Tesla T4 / sm_75（Turing）**）、第 **22** 行（**没有** bf16 / TF32 / `cp.async` / TMA / warp-specialization）

**外部规范（A 级：厂商官方文档原文，逐字且机器抽取）**

- CUDA C++ Programming Guide **12.6.1** ——
  - §**3.2.8.7.3** Creating a Graph Using Stream Capture：锚点 `B3`、`B4a`、`B4b`
  - §**3.2.8.7.3.1** Cross-stream Dependencies and Events：锚点 `B4c`
  - §**3.2.8.7.3.2** Prohibited and Unhandled Operations：锚点 `B5a`、`B5b`、`B5c`
  - §**3.2.8.7.3.3** Invalidation：锚点 `B6`
  - §**16.6.4** Shared Memory：锚点 `C2`（48 KB 规则；★ 作用域是 **7.x 设备**）
- CUDA Runtime API Reference —— `cudaEventRecord`（锚点 `B8`：external event node）；`cudaStreamCaptureMode` 相关条目（锚点 `B9a` / `B9b` / `B9c`）。

> 锚点的**逐字引文**保存在 `code/specs.py`，并由 `make anchors` 从**实际下载的官方 HTML** 里正则**再抽取**一遍、逐字比对（本机跑出：**22/22 条逐字一致**）。抽取器**不手打任何一句引文**。

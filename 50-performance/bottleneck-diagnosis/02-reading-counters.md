---
title: 读计数器：三种系统性误读，而且都不报错
slug: reading-counters
domain: 50-performance
topic: bottleneck-diagnosis
level: 3
difficulty: advanced
prerequisites:
  - 50-performance/bottleneck-diagnosis/01-latency-and-bandwidth
tags: [counters, metrics, out-of-range, same-name-different-referent, attribution, warp-stall]
related:
  - 20-gpu-architecture/execution-model/01-sm-internals
  - 30-gpu-programming/ptx-sass
  - 50-performance/profiling-tools/03-defaults-and-attribution
code:
  - code/02_reading_counters.py
status: stable
updated: 2026-09-21
---

# 读计数器：三种系统性误读，而且都不报错

> **一个计数器的值由三段定义共同决定：采样条件（何时计数）、归因对象（算给谁）、
> 取值范围（超范围怎么办）。任一段没读，那个数就不是你以为的那个数。**
> 三种系统性误读是：**超范围值**（工具故意不 clamp）、**同名不同指**（两套
> `Warp Stall Reasons`）、**把调度器状态当 stall**（官方反模式）。
> **它们都不会让工具报错——它们只是让那个数变成另一个数。**

## 0. 背景与动机

三个具体场景。

**场景一：读到一个 > 100% 的命中率。** 工具不报错、不警告、不 clamp，
它**照原样打印**——`code/specs.py` 的 **F10 逐字**：「超出度量逻辑范围的值，
工具**故意不 clamp**，以保证报告的其余部分保持自洽」。
**它不是 bug，它是为了不掩盖上一个错误。**

**场景二：同一个指标名，两套定义。** Profiling Guide 的 **§2.4.7** 与 **§2.4.8**
都叫 `Warp Stall Reasons`，第一个句子的**逐字相同**
（`Collected using warp scheduler state sampling.`），**第二个句子方向相反**：
一个「无论调度器是否发射指令都计数」，一个「**只在调度器未发射指令的周期**计数」（**F14**）。
**两个数都对，回答的是不同问题。**

**场景三：报告里最热的符号，你根本没写过。** 本机实测（§3）：
程序自己定义的两个函数在采样报告里**各出现 0 次**，最热的是
`mach_absolute_time` 与 `_PyEval_EvalFrameDefault`。

**不用它的代价**：拿场景一那个 > 100% 的比值去驱动优化；
拿场景二里 2.4.7 的数去「降 stall」；拿场景三的报告断定「我那个函数不耗时」。

> ★ **限定语（本篇强制）**：本篇凡出现 `ncu` 的 metric 名、章节号、定义与行为，
> **全部是规格引用**（`code/specs.py` 的 F10 / F11 / F12 / F14，A 级，
> 带 URL + 章节 + 抓取日）。**本仓库没有 GPU，从未运行 `ncu`/`nsys`。**
> 本机实测的部分**只有 §3**，它是 **CPU 上的测量**，**不是 GPU、不是 Nsight Compute**；
> 两者**不得互证**。

## 1. 原理：一个计数器的语义由三段定义共同决定

读一个计数器之前，先把这三段补齐。**缺哪一段，那个数就归到「不知道是什么」里。**

| 段 | 问题 | 缺了它会怎样 | 规格依据 |
|---|---|---|---|
| **采样条件** | **什么时候计一次数？** | 同一个名字下会有两个不同的量（§2） | **F14**（§2.4.7 vs §2.4.8 的两套增量规则） |
| **归因对象** | **这一次算给谁？** | 你会去优化「生产者」，而样本记在「消费者」名下 | **F11**：`Warp stall samples are attributed to the consumer instruction that is waiting on a dependency, not the producer` |
| **取值范围** | **超出物理范围时怎么办？** | 你会以为工具坏了，而其实它**故意不 clamp** | **F10**：`they are not clamped by the tool to their expected value on purpose` |

### 1.1 归因为什么会失效（本机可复现的那一段）

**归因的前提是「符号在栈上」**（`50-performance/profiling-tools/01-collection-mechanisms.md` §1.3）。
这条前提一旦破掉，报告**没有出错**——它只是**在回答另一个问题**：
「这台机器此刻在哪个符号里」，而不是「你写的哪个函数最耗时」。

三种破法，本主题管后两种：

1. **解释型语言的帧不出现在机器栈上** —— 正文在同域 `profiling-tools/01`；
2. **内联让函数边界消失** —— 正文在同域 `profiling-tools/01` §4.3；
3. **机器栈上的名字是别人的名字** —— **本篇 §3 的本机实测**。

### 1.2 一条可判定的形式

设某次采集里，报告**提到**的符号集合为 $M$，程序**自己定义**的函数集合为 $O$。
「归因失效」的判据是：

$$M \cap O = \varnothing$$

**★ 这个式子的口径必须写死**（不写口径的集合运算就是 `P13`）：
本机程序里 $M$ 是在**一个有限候选集合**上取的（程序自己定义的函数 ∪ 机器栈上那几个固定符号），
$O$ 就是程序自己定义的那两个函数名。**它不是「把报告里所有符号解析出来」**——
后者要解析 `sample` 的文本格式，格式一变判据就失效，而且解析本身是新错源。
有限候选让判据**稳定且可复核**，代价是只覆盖这两个集合。

**★ 还要加一条反向条件**：$M \ne \varnothing$。
只断言 $M \cap O = \varnothing$ 是不够的——**采样窗口太短、或者 pid 打错了，
报告是空的，这条断言照样成立**。所以必须同时要求「机器栈符号在场」。

## 2. 执行映射：把 F14 的两套定义并置

这是 `P9`（同名不同指）的教科书实例。**并置**，不要只看其中一个：

| metric 名 | 采样条件（增量规则，逐字） | 逐字依据 |
|---|---|---|
| `Warp Stall Reasons`（**§2.4.7**） | `They are incremented regardless if the scheduler issued an instruction in the same cycle or not.` | **F14** |
| `Warp Stall Reasons (Not Issued)`（**§2.4.8**） | `They are incremented only on cycles in which the warp scheduler issued no instruction.` | **F14** |

**两节共享的第一句完全相同**（`Collected using warp scheduler state sampling.`）——
这就是「同名」那一半；**第二个句子方向相反**——这就是「不同指」那一半。

**为什么这不是文字游戏**：§2.4.7 那一版的计数**包含调度器正在正常发射指令的周期**，
所以它**天然更大**；拿它当「stall 有多少」，等于把「忙着的周期」也算进了「停顿」。
**两个数都是工具按设计给出的**，问题只在于你读的是哪一个。

| 组件 | 它在这里的角色 | 数值量级 |
|---|---|---|
| **warp scheduler** | §2.4.7 与 §2.4.8 两套定义的**唯一差别**就在「采样时调度器有没有发射指令」 | 两套定义的第一句**逐字相同**（同名），第二句**取向相反** |
| **SM 的 issue 槽** | F12 说真正的目标是**提高 issue-slot 利用率**，不是把 stall 计数器压到 0 | 官方原文：`increasing issue-slot utilization, not driving every stall counter to zero` |
| **消费者的指令** | F11 规定 stall 样本**归因给等在依赖上的那条指令**，不是生产者 | `the consumer instruction that is waiting on a dependency` |
| **跨 pass 的分子分母** | 超范围的另一个常见来源：比值型指标的分子分母来自**不同的 pass** | 稳态阈值一般 **> 20 µs**（F3 逐字）；详见 `01-latency-and-bandwidth.md` §2 与同域 `profiling-tools/02` |

## 3. 实现与演进

**朴素：把工具的输出当程序的属性。** 「stall 高 ⇒ 我的 kernel 有 stall 问题」——
三段定义一段都没读。这一阶段的典型症状是**把口径当结论**。

**第一步：把「采样条件」读出来。** 同一个 metric 名在不同章节下有不同定义（§2）。
**为什么需要下一步**：条件读懂了，**归因对象**还没读——样本算给谁，决定你去改哪一行。

**第二步：把归因显式化。** F11 把 stall 归给**消费者指令**。
**为什么需要下一步**：归因规则再清楚，也管不到「符号根本不在栈上」的那些情况（§1.1）——
那要靠**另一条采集路线**而不是另一条规则。

**第三步：承认归因会失效，并把它做成可跑的对照（本机实测）。**
这就是下面这一段。

### 3.1 本机实测：归因失效的现场

被测程序是**解释型语言**的程序，只有两个函数：

- `slow_spin_probe(0.30)` —— 用 `time.perf_counter` 自旋；
- `busy_accumulate()` —— 纯计算。

**先用显式计时确立「谁最耗时」**（不靠采样器，`A.2`）：
本机实测 `slow_spin_probe` 是 `busy_accumulate` 的 **6 倍**
（★ 两个函数**都按墙钟定界**，所以这个比值**不随机器负载漂** ——
实施者第一版让其中一个按迭代次数定界，它在忙机器上从 0.08 s 涨到 0.30 s，
`A.2` 直接失败）。

**再让 `sample` 去采它**（`A.1`–`A.5`）：

| 类别 | 符号 | 出现次数 |
|---|---|---|
| **程序自己定义**的 | `busy_accumulate` | **0** |
| | `slow_spin_probe` | **0** |
| **机器栈**上的 | `_PyEval_EvalFrameDefault` | > 0 |
| | `PyObject_Vectorcall` | > 0 |
| | `mach_absolute_time` | > 0 |
| | `cfunction_vectorcall_NOARGS` | > 0 |
| | `_tlv_get_addr` | > 0 |

> ⚠ 上表的「出现次数」是**一次** `make run` 的读数；程序**只对「0 次 / > 0 次」下断言**。

**同一批数据、两个方向**：`A.3` 断言自己的函数 **0 次**，
`A.4` 断言机器栈符号 **> 0 次**——只有后者成立，前者才不是「采样器什么都没采到」。
`A.5` 是 §1.2 那条式子的机器化：$M \cap O = \varnothing$ **且** $M \ne \varnothing$。

**结论（正文只写这一句）**：**耗时最长的那个函数，采样器一次都没归给它。**
`time.perf_counter()` 调 `mach_absolute_time`，于是「自旋等待」被归给了**内核计时调用**。

> ⚠ **绝对计数不可照抄**：上面的具体次数**随程序与采样窗口变化**
> （同域 `profiling-tools` 的 §0.7 **D-13**：控制者与事实层审查者复跑同一实验，
> `mach_absolute_time` 分别是 136 与 468）。
> ⇒ **正文只写「0 次 / > 0 次」这个结论**，程序里也**只对它下断言**。

**跨语言同构**（本域的核心论点之一）：这与 GPU 上「内联使函数边界消失」
「JIT 代码没有符号」是**同一类问题**——采样器不是「没看见」，
而是**看见了一个没有你名字的栈**。

> ★ **一处必须写清的口径**：程序里那两个函数名**刻意避开了** `main` / `run` / `start`
> 这类会出现在机器栈上的子串。同域 `profiling-tools` 的 `03` 记着这个教训：
> 采样图里的 `pymain_main` / `Py_BytesMain` / `pymain_run_file` 都含 `main`，
> `str.count("main")` 会**假阳性** —— 那是 `P13` 的一个微型实例：**我量的不是「我写的那个函数」。**

### 3.2 本机实测：同一个名字下的**两个计数器**

F14 讲的是「同一个 metric 名、两套采样条件」。**本机的对应物是「耗时」这个词本身**：

| 计数器 | 它是什么 | 谁维护 |
|---|---|---|
| `time.perf_counter()` | **挂钟**：现在离开始过了多久 | 单调时钟（本机是 `mach_absolute_time`） |
| `resource.getrusage()` 的 `ru_utime + ru_stime` | **CPU 时间**：这个进程真的占了 CPU 多久 | **内核**维护的 per-process 计数器 |

本机实测（`B.1`–`B.3`）：

| 负载 | 挂钟 | CPU 时间 | 挂钟 / CPU |
|---|---|---|---|
| 纯计算 `burn(0.25 s)` | 0.2504 s | 0.2243 s | **1.01–1.47**（静息 1.005；机器被打满时会被抬高） |
| 睡眠 `sleep(0.30 s)` | 0.3000 s | 0.0000 s | **5000–8600** |

**两个数都对**，它们量的**不是同一件事**。
「耗时」这个词在中文里可以指其中任何一个 —— 这就是本机版本的**同名不同指**。

★ **口径**：`B.1` 的窗是 `< 20`（静息实测 1.005），因为机器被打满时挂在墙钟上的时间被拉长、
而 CPU 时间不会 —— 这一条的区分力**不在那个绝对值**，在 `B.2` 的**对照**（几千倍 vs 1 倍）。

> ★ **与 `45` 的分工**：`45-kernel-lifecycle` 讲的是**怎么选统计量**、MDE 与噪声带；
> 本篇不讲那些（正文在 `45`，此处回链）。本篇讲的是**读一个数之前先把它的定义补齐**。

## 4. 踩坑与失败模式

### 4.1 现象：拿 §2.4.7 的那一版 `Warp Stall Reasons` 去优化「stall」

- **成因**：**F14**——同一节名下有两套增量规则，§2.4.7 那一版
  「无论调度器是否发射指令都计数」，**它天然更大**，而且它不是「停顿」的意思。
- **规避**：**先读两处定义**，按你要回答的问题挑一个。
  问「有多少个周期被浪费在依赖上」用 §2.4.8 那一版；问「调度器看到了多少 stall 状态」用 §2.4.7。

### 4.2 现象：读到一个 > 100% 的比值，以为工具坏了

- **成因**：**F10 逐字**——超出逻辑范围时工具「**故意不 clamp**，以保证报告的其余部分自洽」。
  它不是 bug，**是为了不掩盖上一个错误**。
- **规避**：**先怀疑 multi-pass（分子分母来自不同的 pass，F3），再怀疑工作量可变，
  最后才怀疑硬件**。同域 `profiling-tools/02-replay-and-multipass.md` 有本机的最小复现
  （构造出 **400%** 的命中率，程序照原样打印、不做 clamp）。

### 4.3 现象：把 stall 归给了生产者

- **成因**：**F11 逐字**——`attributed to the consumer instruction that is waiting on a
  dependency, not the producer`。你去改生产者，而样本记在**等在依赖上的那条指令**名下。
- **规避**：**去 SASS 里找「等在依赖上的那条指令」**。
  读 SASS 的方法在 `30-gpu-programming/ptx-sass`（正文在那边，此处回链）。
  ⚠ 本仓库**没有** `ncu`、**没有** GPU，这条是**规格引用**，不是本机实测。

### 4.4 现象：采样报告里最热的符号是 `mach_absolute_time` 一类「你没写过的函数」

- **成因**：归因（§1.1、§3.1）。采样看到的是**机器栈**。
- **规避**：**结合另一条采集路线**（拿到精确的调用次数与函数级 exclusive 时间，
  见同域 `profiling-tools/01-collection-mechanisms.md`），或**下钻到 SASS**。
  **不要据此断定「这个函数不耗时」**——本机实测里，那个出现 0 次的函数正是**最耗时的那个**。

### 4.5 现象：同一个程序，「耗时」有两个数

- **成因**：**同名不同指**（§3.2）。挂钟与 CPU 时间是两个计数器。
  本机实测：睡眠 0.30 s 的那一段，CPU 时间只有 `0.0000 s`。
- **规避**：**报「耗时」时必须说明是哪一个**。性能报告里这一条最容易被省掉，
  而它决定了你接下来要去优化什么。

## 5. 判据表

| 条件 | 结论 |
|---|---|
| metric 名相同，但来自不同章节 / 带不同后缀 | **先读两处定义**——可能是**同名不同指**（F14：§2.4.7 与 §2.4.8 的增量规则取向相反） |
| 值超出物理范围（> 100%、> 1 的比值） | **不 clamp 是有意的**（F10）；**先查 multi-pass（F3），不要先怀疑硬件** |
| 想知道「谁该为这次 stall 负责」 | 归因给**消费者指令**（F11）——去 SASS 里找**等在依赖上的那条**，不是生产者 |
| 目标是「把 stall 降到 0」 | **错的目标**（F12 逐字）；目标是**提高 issue-slot 利用率**。且 stall 减少**只在 kernel 受延迟限时**才转化为加速 |
| 报告里**你自己的函数一次都没出现** | 先排除「打错了 pid / 采样窗口太短」（要求 `M ≠ ∅`），再按归因失效处理。**不要**据此断定该函数不耗时 |
| 「报告提到的符号集合 ∩ 程序自己定义的函数集合 = 空集」 | 归因失效已被证实——但这条**只在 `M ≠ ∅` 时**才有意义（空报告也满足它） |
| 同一个程序，「耗时」有两个不同的数 | 先问是**挂钟**还是 **CPU 时间**（本机实测两者可差 **5000–8600 倍**） |
| 一个指标你只读了名字，没读它的采样条件 | 那个数**归到「不知道是什么」**里。补齐三段（采样条件 / 归因对象 / 取值范围）再用 |

## 6. 小结与延伸

**一句话**：**读计数器有三种系统性误读，而且都不报错**——
超范围的值（F10 故意不 clamp）、同名的两个指标（F14）、
以及把归因当成「时间算给了谁写的代码」（F11/F12 说算给了**消费者指令**）。

本域主线在这里的形态：**你看到的不是你的程序，是关于你的程序的一次测量**——
而这次测量里，**那个数属于谁、它是什么时候被采到的、它超范围时会怎样**，
三件事都得你自己读出来。**工具不会替你报错。**

- **姊妹篇**：`01-latency-and-bandwidth.md`（先判 bound 类型，再谈怎么改 ——
  本篇 §4.1 那个「降 stall」的误区，根子在那里）、
  `03-diagnosis-walkthrough.md`（把这四步串成一次完整诊断）。
- **往上游回链**：`20-gpu-architecture/execution-model/01-sm-internals`（warp scheduler 与
  issue 槽的物理背景）、`50-performance/profiling-tools/03-defaults-and-attribution.md`
  （F11/F12 那两条规格在那里首次展开；**本篇不重复它的默认值部分**）。
- **往下游**：`30-gpu-programming/ptx-sass`（正文在那边，此处回链）
  ——归因对不上时要下钻到 SASS，读 SASS 的方法在那一篇。

## 编译与运行

```bash
cd code
make run                              # 三个编号程序依次跑；本程序是 02_reading_counters.py
python3 02_reading_counters.py        # 只跑本程序
python3 02_reading_counters.py --lint # AST 扫描：不 import torch/triton/numba/cuda
```

- 配套程序：**`code/02_reading_counters.py`**（编号与外键严格对应）。
- 需要的外部工具：**`sample`**（macOS 自带）。**不需要 `clang`、不需要网络。**
  `sample` 不在时 A 类断言**整体 `[SKIP]` 且单独计数**（不计入通过），程序仍以 0 退出。
- **本程序从不运行 `ncu`/`nsys`** —— 凡出现它们的 metric 名与定义，
  全部是 `code/specs.py` 里 `FACTS` 的规格引用（F10 / F11 / F12 / F14，
  带 URL + 章节 + 抓取日）。
- 可选：设置 `PERF_DOC_DIR` 指向两份 NVIDIA 文档的下载目录，
  程序会额外核 sha256 并逐字核对 F1–F15 的引文。

## 参考

**本机实测值**

- `code/02_reading_counters.py` 的 `make run` 输出（Apple M1 Pro / macOS / `python3 3.13.5`）。
  **全部是 CPU 上的测量**，不是 GPU、不是 Nsight Compute。
- 计划 `docs/superpowers/plans/2026-09-21-performance-domain.md` **§0.7 D-5**（归因失效）、
  **D-13**（`sample` 的绝对计数不可照抄）、**D-8 / D-10**（噪声是重尾的）。

**规格值（A 级，厂商官方文档）**

- NVIDIA, *Nsight Compute — Profiling Guide* —
  `https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html`
  - **§2.3.8 Range and Precision**（F3 / F10 的出处）
  - **§2.4.7 Warp Stall Reasons**（F11 / F12 的出处；F14 的第一半）
  - **§2.4.8 Warp Stall Reasons (Not Issued)**（F14 的第二半：
    与 §2.4.7 **同名不同指**的另一个 metric）
  - 抓取日 **2026-09-21**；页脚 last-updated **Aug 27, 2026**；
    原始字节 381812 B，sha256（前 16）`6a84b367be4dd99a…`。
- ⚠ **本仓库从未运行 `ncu`/`nsys`**：F10–F14 是**规格值**，不是本机实测，
  **也不得**与本机的 CPU 实测互证。
- 逐字引文与 `must_contain` 锚点全部记在 **`code/specs.py`** 的 `FACTS` 里；
  F14 的两套增量规则另见 `F14_DEFINITIONS`。

**上游契约**

- `45-kernel-lifecycle/convergence-and-tuning/03-convergence.md:256`
  「在 kernel 上继续调**不改变总时间** ⇒ 换分支（本域只给判据，工具属 `50-performance`）」
  —— 判据在 `45`，**定位的工具与方法**在本域。
- `10-foundations/performance-models/03-measuring-roofline.md:187`
  「微基准与 profiler 的分工是：**前者给出边界，后者给出当前位置**。」

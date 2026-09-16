---
title: codegen 与后端
slug: inductor-codegen
domain: 60-compilers
topic: inductor
level: 3
difficulty: advanced
prerequisites:
  - 60-compilers/inductor/02-fusion-scheduling
tags: [inductor, codegen, triton, cpp-backend, cutlass, autotune, max-autotune]
related:
  - 30-gpu-programming/triton
  - 60-compilers/auto-tuning
  - 60-compilers/inductor/04-dynamic-shapes
code:
  - code/03_codegen_and_exec.py
status: stable
updated: 2026-09-16
---

# codegen 与后端

> Inductor 自己不发明 kernel 语言：GPU 上它产出**标准 Triton 内核**，CPU 上是 **C++/OpenMP 模板**。后端按 **device 字符串**注册——所以「换一个后端」在这套系统里是「换一个 device 绑定的 codegen」，不是一个编译选项。而「默认到底搜不搜」的答案**分两类算子**：GEMM/conv 默认走外部库核，而 **pointwise/reduction 默认就在搜**。

## 0. 背景与动机

[02-fusion-scheduling](02-fusion-scheduling.md) 把图划成了若干**融合组**。本文接着往下走一步：**每个融合组变成什么**。

对照场景（本仓库程序里真的跑的那一个）：

```python
def model(x, w, m):
    a = (x * w).relu()      # pointwise ×2
    s = a.sum(axis=1)       # reduction
    e = a.matmul(m)         # GEMM → 外部库
    c = s + 1.0             # pointwise
    return [e, c]
```

四个计算节点，但它们在**两条不同的路**上：

- `mul` / `relu` / `sum` / `add` → 被划进一个融合组，生成**一个模板核**（GPU 上是 Triton 内核）；
- `matmul` → 默认**不参与模板搜索**，交给外部库（cuBLAS 这一类）。

问题就在最后这一句上，因为**它很容易被读成「默认什么都不搜」**。事实不是这样：默认不搜的是 GEMM/conv；**pointwise/reduction 默认就走搜索路径**（`triton.autotune_pointwise` 默认 `True`，源码注释写着 "should only be disabled for debugging/testing"）。把这两件事混起来，会对「`max-autotune` 这个模式到底改变了什么」给出完全相反的答案。

本文与主线的关系：**本文是主线的代价项**。融合组划得好不好，最终体现在生成的核上——生成了几个核、每个核读了多少字节、中间结果有几个真的落了内存。§1 讲后端怎么按 device 分发，§2 讲默认搜什么不搜什么，§3 讲官方给的定位入口，§4 讲两个会读错的默认值。

## 1. 原理

### 第一步：后端是按 **device** 注册的

Inductor 的 codegen 不是「一个后端 + 一堆特例」，而是一张 **device → codegen** 的表。注册 API 叫 `register_backend_for_device`，签名里的第一个参数就是设备字符串：

```text
register_backend_for_device(device, device_scheduling,
                            device_wrapper_codegen, device_cpp_wrapper_codegen, …)
```

注册表本体（torch 2.13.0 的 `torch/_inductor/codegen/common.py:519-586`）：

| device | 注册的 codegen |
|---|---|
| `cpu` | `config.cpu_backend`，默认 **`cpp`**（另有 triton / halide / pallas） |
| `cuda` | `config.cuda_backend`，默认 **`triton`**（走 `CUDACombinedScheduling`） |
| `xpu` | `XPUCombinedScheduling` |
| `tpu` | `PallasScheduling` |
| `mps` | `MetalScheduling` |
| `mtia` | `TritonScheduling` |

两个默认值的确切位置是 `config.py:2666`（`cpu_backend = "cpp"`）与 `config.py:2670`（`cuda_backend = "triton"`）。

**这张表有三个后果**：

1. **「换后端」= 换一个 device 的绑定**，而不是给编译器加一个 flag。ROCm 走的是 `cuda` 这一行（HIP 在 torch 里是一个 cuda device）。
2. **`cuda` 那一行不是「纯 Triton」**：它的 `Scheduling` 叫 `CUDACombinedScheduling`，由它把 Triton 与 CUDA C++ 两条调度**委托**起来。
3. **没有注册的 device 会在查表处失败**，不会静默退化到某个默认后端。本仓库的迷你注册表把这条做成了可跑的对照（§5）。

### 第二步：Inductor 产出的是**标准** Triton 内核

设计原则是「复用 SOTA 语言，不发明 DSL」：GPU 侧的产物是 Triton（一种已经存在的、有自己编译器的语言），CPU 侧是 C++ 模板。**这一点决定了它和上一层的边界**：Triton 语言本身属于 [30-gpu-programming/triton](../../30-gpu-programming/triton/)（前向声明），本域只讲「编译器生成什么形状的 Triton」，不讲 Triton 语法。

### 第三步：默认搜什么、不搜什么

这是本篇最要紧的一节，因为默认值的**方向**很容易记反。

| 算子类别 | 默认行为 | 依据 |
|---|---|---|
| **GEMM / conv** | **不搜**：走外部库核（cuBLAS / cuDNN 这一类） | `max_autotune_gemm_backends` 默认 `"ATEN,TRITON,CPP"`；不开 `max_autotune` 时 GEMM 走 ATen |
| **pointwise / reduction** | **默认就在搜** | `triton.autotune_pointwise = True` |

`autotune_pointwise` 的源码注释逐字是：

> use triton.autotune for pointwise ops with complex layouts
> this should only be disabled for debugging/testing

**「默认不搜，开 `max_autotune` 才搜」这句话是错的**——按这句去读，会以为默认状态下 pointwise 用的是某个固定启发式配置。实际不是：那条路径本来就是 `triton.autotune`。

`max_autotune` 打开之后变的是什么？**变的是 GEMM/conv 那一侧**：它们从「交给外部库」变成「纳入模板 / CUTLASS 这些候选」。

**一处不得给数字的地方**：默认 pointwise autotune 到底搜多少个 config、`max_autotune` 定量上省多少编译时间 / 换多少性能，**官方没有数字**。本文只写定性边界，**不给区间**。

### 第四步：两个 `mode` 的实际映射

`torch.compile(mode=...)` 有几个取值：`"default"` / `"reduce-overhead"` / `"max-autotune"` / `"max-autotune-no-cudagraphs"`。它们是若干配置项的**预设组合**，而不是三套不同的编译器。要看某个 mode 到底设了什么，官方给的入口是 `torch._inductor.list_mode_options()`。

**引用时以目标版本的源码为准，不要抄博客**——这类「mode 等于哪几个开关」的对应关系跨版本会变。

## 2. 执行映射

**组件一：设备注册表（`codegen/common.py`）。** §1 那张表的落点。**量级**：注册表里有 6 个 device 行（cpu / cuda / xpu / tpu / mps / mtia），外加一个 private-use 的扩展口。往里加一个后端就是调一次 `register_backend_for_device`——外部项目（如 Intel 的 XPU 后端）就是这么接进来的。

**组件二：`max_autotune_gemm_backends` 的取值集合。** 默认值是字符串 `"ATEN,TRITON,CPP"`。**这三个名字不是同一层的三个后端**：

- `ATEN`：外部库核（cuBLAS/cuDNN）。
- `TRITON`：Triton 模板。
- `CPP`：**CPU 的 C++ 模板**——同一段源码的注释逐字写着 `CPP: CPP templates and kernels for CPU.`；旁边列出的其它名字是 `NVGEMM` / `CK` / `CKTILE`（GPU / AMD 侧）。

**同一个字符串里出现 `CPP` 与 GPU 侧的名字，是这条默认值最容易被误读的地方**（§4 第 1 条）。GPU 侧的 C++ 模板叫 **CUTLASS**，不叫 CPP。

**组件三：wrapper 与调度是分开注册的。** `register_backend_for_device` 的参数里有 `device_scheduling` 与 `device_wrapper_codegen` 两类——**谁负责调度、谁负责拼出调用代码**是两组对象。这个拆分解释了为什么 `cuda` 能同时是「Triton 调度」与「C++ wrapper」（`CppWrapperGpu`）：它们不是同一件事。

**组件四：定位入口。** 生成物在磁盘上的位置与调试开关见 §3。**量级**：本仓库程序里那张 4 计算节点的图，最终生成 **2 个核**、物化 **3 个缓冲**、字节流量 读 112 / 写 56 / 合计 168（见 §5）。

## 3. 实现与演进

**第一代：只生成 Triton。** 早期路径非常直白——一个融合组一个 Triton 内核。问题在 CPU 上：没有 Triton，于是 CPU 只能退回 ATen。

**第二代：按 device 注册，把 C++ 变成一等公民。** `register_backend_for_device` 与 `CppWrapperCpu` / `CppWrapperGpu` 这一组 wrapper 让 CPU 有了自己的模板核，让 GPU 有了 C++ 调用封装。**这一代确立了本文的核心结构**：后端不是编在代码里的分支，而是注册表里的一行。

**第三代：模板与搜索。** `max_autotune` 把 GEMM/conv 从「交给外部库」变成「在模板候选里搜」。**这一代引入了新的失败面**：搜索是**非确定性**的——AOTInductor + `max_autotune` 会对小 GEMM 非确定性地选 cuBLAS 而不是融合的 Triton 模板，从而**打破原本已经形成的融合**造成回退（issue #180811）。也就是说：打开搜索**可能让另一处已经做好的优化失效**。

**官方调优工作流：支持的是「读 + 定位」，不是「改」。**

| 入口 | 它给什么 |
|---|---|
| `TORCH_COMPILE_DEBUG=1` | 打开调试目录（同时自动开 trace 目录） |
| `TORCH_TRACE=<dir>` | 指定 trace 输出目录 |
| IR 转储 `ir_pre_fusion` / `ir_post_fusion` | 融合前 / 融合后的 IR |
| `INDUCTOR_PROVENANCE` | 打开 provenance（溯源）信息 |
| `torch._dynamo.utils.compile_times()` | 各阶段编译耗时 |
| `TORCH_LOGS=output_code\|kernel_code\|fusion\|schedule` | 按模块打开日志 |

**这一组名字都能在源码树里定位**（逐条的文件与行号见 `code/specs.py` 的 `DEBUG_ENTRY_POINTS`），但**本仓库没有跑过它们**，因此本文不描述它们的输出长什么样。

**一条边界必须写清楚**：官方支持的是**读**生成的代码并**定位**到是哪一段 IR 产生了它。**没有任何官方入口是「手改生成的 kernel 然后让它生效」**。社区工具（如 `depyf`）不是官方的一部分。

## 4. 踩坑与失败模式

**失败模式 1：把 `CPP` 当成 GPU 后端。**

- **现象**：看到 `max_autotune_gemm_backends` 默认值是 `"ATEN,TRITON,CPP"`，于是认为「GPU 上还有一条 C++ 模板的路」，进而去调 `config.cpp.*` 里的一堆开关（比如 `max_horizontal_fusion_size`），**在 GPU 上没有任何效果**。
- **成因**：这个默认值把三类名字放在同一个字符串里。源码注释里 `CPP` 那一行的定位是 `CPP: CPP templates and kernels for **CPU**`；GPU 侧的 C++ 模板叫 **CUTLASS**。更隐蔽的是 `max_horizontal_fusion_size`（默认 16）——它**不在** `triton` 配置段里，而在 **`config.cpp`** 段内，即**CPU 的 C++ 后端**配置。
- **规避**：读默认值时**连带读它所在的那一段配置的注释**。本仓库把这条做成了可跑的对照：请求「用 cpu 的 codegen 去编 cuda」会直接报错，代码名与设备名对不上就是错，不会静默退化（见 §5）。

**失败模式 2：以为可以手改生成的 kernel。**

- **现象**：为了让某个核更快，直接改生成出来的源码，期望它被下游复用。
- **成因**：官方提供的全部入口都是**读**——调试目录、IR 转储、日志、provenance——**没有一条是「改」。** 生成物是编译流程的产物，不是源码。
- **规避**：要改变生成结果，就改变**输入**（图、调度决策、配置项），而不是改产物。定位手段用 §3 那张表。
  > **⚠ 一条不得写进来的机制从句**：曾经流行的一种说法是「手改的 kernel 会在编译缓存里被覆盖」。**这条没有任何一手来源**，本仓库在 torch 2.6.0 / 2.7.1 / 2.13.0 里都没有找到对应的表述（记录在 `code/specs.py` 的 `UNSOURCED`）。**正确的表述只有一句：官方支持的是「读 + 定位」。**

**失败模式 3：把「默认不搜」推广到所有算子。**

- **现象**：以为不开 `max_autotune` 就等于「用固定启发式配置」，于是抱怨 pointwise 的配置差；或者反过来，以为开了 `max_autotune` 才能让 pointwise 变快。
- **成因**：默认边界**分两类**——GEMM/conv 默认不搜（走外部库），pointwise/reduction **默认就在搜**（`triton.autotune_pointwise = True`，注释明说只有调试/测试才该关掉它）。把 GEMM 那一侧的默认推广到全部算子，方向就反了。
- **规避**：记住判据的**粒度是算子类别**，不是全局开关。

**失败模式 4：打开搜索，反而丢掉已经做好的融合。**

- **现象**：开 `max_autotune` 之后，某些小 GEMM 的耗时反而变大。
- **成因**：搜索是**非确定性**的。issue #180811 记录的情形是：AOTInductor + `max_autotune` 会对小 GEMM 非确定性地**选 cuBLAS 而不是融合的 Triton 模板**——而那个 Triton 模板正是先前融合的产物。选回去等于把融合取消。
- **规避**：把「搜索出来的最优」与「融合形成的结构」当成**两个会互相干扰的决策**来看，而不是「搜索只会更好」。（**引用值**：issue #180811 是外部来源，**本仓库不复核其测量口径**。）

> **一条不得写进本文的内容**：「reduction 有时候用 C++ 后端更快」。这条**没有任何一手来源**，且属于概念混淆——`CPP` 是 CPU 的 C++ 模板（见失败模式 1）。记录在 `code/specs.py` 的 `UNSOURCED`。

## 5. 实测数据

**测量条件**：`code/03_codegen_and_exec.py` 在本机 `python3 3.13.5` + `numpy 2.4.4` 上跑出的结果，`cd code && make run` 全绿（本篇 18 条断言：A 5 / B 9 / C 4）。

> **★ 限定语（本域强制）**：下面的数字全部来自**本仓库几百行的迷你 codegen 与迷你设备注册表**，**不是 Inductor 的 Triton/C++ 后端**。设备注册表的内容、默认值、路径选择的**边界**逐条抄自固定版本的源码（见 `code/specs.py`），但「哪个节点走哪条路」的判定是在**本仓库的迷你模型**上跑的——它演示的是判据的形状，**不是 Inductor 的实际决策**。这些数字支持「边界长什么样」这类**机制**结论，**不支持任何关于 Inductor 生成代码的结论**。

**地形**：5 个计算节点（`mul` / `relu` / `sum` / `mm` / `add`），调度的结果是 **2 个融合组**（`[mul, relu, sum, add]` 与 `[mm]`）。

| 项 | 本机实测值 |
|---|---|
| 图里的计算节点数 | 5 |
| 融合组数（= 生成几个核） | 2 |
| **默认**下走外部库核的算子数 | **1**（`mm`） |
| **默认**下走搜索路径的算子数 | **4**（`mul` / `relu` / `sum` / `add`） |
| 开 `max_autotune` 后走外部库核的算子数 | 0 |
| 开 `max_autotune` 后走搜索路径的算子数 | 5 |
| 物化的缓冲个数 | 3 |
| 字节流量（读 / 写 / 合计） | 112 / 56 / 168 |

**生成物（本仓库迷你 codegen 的输出，**不是 Triton**）**：

```python
def kern0(in0, in1):
    v_buf0 = (in0 * in1)
    v_buf1 = np.maximum(v_buf0, 0.0)
    v_buf2 = np.sum(v_buf1, axis=(1,))
    v_buf4 = (v_buf2 + 1.0)
    return [v_buf1, v_buf4]

def kern1(in0, in1):
    v_buf3 = np.matmul(in0, in1)
    return [v_buf3]
```

第二个核 `kern1` 读的 `in0` 就是第一个核返回的 `v_buf1`——**这条边就是「物化」**：`relu` 的输出同时被组内的 `sum` 和组外的 `mm` 使用，所以它必须落一次存储。如果 `mm` 也在这个组里，它就会留在寄存器里、不出现在返回值里。

**A 类的独立参照路径**：上面两个核**真的被 `exec` 执行**，输出与纯 numpy 参考实现逐元素比对，最大绝对误差 **0.000e+00**。这是计划里 C4 那条要求的落点——**它属 A 类，不属 C 类**（断言的是成功行为）。

**C 类的失败行为**（每条都真的抛了异常，不是"理论上会"）：

| 构造 | 结果 |
|---|---|
| 用 `cpp` 这个 codegen 去编 `cuda` | 抛出 `BackendDeviceMismatch: codegen 'cpp' 注册在 ['cpu'] 上，**不在** cuda 上 |
| 请求一个未注册的 device（`npu`） | 抛出 `UnregisteredDevice`（**不会静默退化**） |
| 把一个 extern 核与一个模板核分到同一组 | 拒绝（extern 核不参与模板融合） |
| 可证伪性反证：把 codegen 的一条表达式（`mul`）故意写成 `div` | 最大绝对误差从 0 变成非零，A 类判据立刻失败 |

**第一版被实跑推翻的两项**：

1. **「搜索路径上的算子个数」两个期望值我一开始都写错了**（写成 3 / 4，实际是 4 / 5）。这是先跑再写的直接产物：如果先写表格再补程序，程序会被改成迎合表格。
2. **`emit_python` 一开始把「组内被读过」的缓冲排除出返回值**，于是下游的 extern 核拿不到它的输入，`exec` 时直接 `KeyError`。修法是把判据从「组内有没有人读」换成「**跨组有没有人读**」——即 `materialized_nodes`，代价模型与 codegen 现在共用同一个函数，避免两处漂移。

## 6. 小结与延伸

- **后端按 device 注册**（`register_backend_for_device`）：cpu → 默认 `cpp`，cuda → 默认 `triton`（`CUDACombinedScheduling`），另有 xpu / tpu / mps / mtia。**未注册的 device 会报错，不会静默退化。**
- **Inductor 产出标准 Triton 内核**（GPU 侧）与 C++ 模板（CPU 侧），不发明 DSL。Triton 语言本身属于上游。
- **默认 autotune 的边界分两类**：GEMM/conv **默认不搜**（走外部库核，`max_autotune` 才纳入模板候选）；**pointwise/reduction 默认就搜**（`triton.autotune_pointwise = True`）。
- **`CPP` 是 CPU 的 C++ 模板**，GPU 侧的 C++ 模板是 `CUTLASS`；`max_horizontal_fusion_size` 在 `config.cpp` 段里。
- **官方支持读 + 定位，不支持手改**：`TORCH_COMPILE_DEBUG` / `TORCH_TRACE` / `ir_pre_fusion` / `ir_post_fusion` / `INDUCTOR_PROVENANCE` / `compile_times()`。
- **搜索是非确定性的，可能打破已经形成的融合**（issue #180811）。

**延伸**：

- 生成物里那些形状常数什么时候会失效、失效之后怎么办 → [04-dynamic-shapes](04-dynamic-shapes.md)
- 搜索空间、代价模型与搜索策略（编译时间换性能） → [60-compilers/auto-tuning](../auto-tuning/)（前向声明）
- 生成的 Triton 内核长什么样、Triton 语言本身 → [30-gpu-programming/triton](../../30-gpu-programming/triton/)（前向声明）

## 编译与运行

```bash
cd 60-compilers/inductor/code
make 03_codegen_and_exec    # 只跑本篇
make run                    # 跑全部四篇
```

程序：`code/03_codegen_and_exec.py`（配套 `code/minifuse.py` 的 codegen 与解释器，外部事实源 `code/specs.py`）。

## 参考

- 设备注册表与注册 API：`torch/_inductor/codegen/common.py:407-421, 519-586`（torch 2.13.0）
- 后端默认值：`torch/_inductor/config.py:2666`（`cpu_backend = "cpp"`）、`:2670`（`cuda_backend = "triton"`）
- `max_autotune_gemm_backends` 与 `CPP` 的注释：`torch/_inductor/config.py:617-620`
- `autotune_pointwise` 的默认值与注释：`torch/_inductor/config.py:1915-1917`
- 调试入口的可定位位置：`code/specs.py` 的 `DEBUG_ENTRY_POINTS`（逐条给出文件与行号）
- **引用值**：pytorch/pytorch issue #180811（AOTInductor + `max_autotune` 对小 GEMM 非确定性选 cuBLAS，打破融合）——**本仓库不复核其测量口径**
- 以上每条都在 `code/specs.py` 里逐条对应，取用日期 2026-09-16

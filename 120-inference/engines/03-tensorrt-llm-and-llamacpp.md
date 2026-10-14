---
title: TensorRT-LLM 的编译式路线与 llama.cpp 的边界
slug: inference-engines-tensorrt-llamacpp
domain: 120-inference
topic: engines
level: 3
difficulty: intermediate
prerequisites:
  - 120-inference/engines/01-vllm
tags: [tensorrt-llm, in-flight-batching, compiled-engine, llamacpp, boundary]
related:
  - 120-inference/engines/02-sglang-and-radix
  - 120-inference/batching-and-scheduling/01-iteration-level-scheduling
  - 45-kernel-lifecycle/requirement-discovery/03-optimization-priority
code:
  - code/03_tensorrt_and_llamacpp.py
status: stable
updated: 2026-10-09
---

# TensorRT-LLM 的编译式路线与 llama.cpp 的边界

> **本文只写【选择与差异】**：TensorRT-LLM 走的是**编译式**路线 ——
> 它把模型编成引擎，并给 `in-flight batching` 一个 **`[A]` 级**的术语（`= continuous batching =
> iteration-level batching`）；★★★ 而该术语下**有一条顺序约束，是【该实现当前版本】的**。
> ★★★ **llama.cpp 本文【不展开】** —— **本次取数没有拿到它的一手来源**（见 §0.5 与 §2.4）。
>
> ★★★ **本主题的硬约束**：**「引擎的机制」一律回链** ——
> **请求调度器** → [01-iteration-level-scheduling](../batching-and-scheduling/01-iteration-level-scheduling.md)；
> **分页 / 前缀树** → 本域 `kv-cache/02,03`；**PagedAttention 的算式** →
> `130-model-optimization/quantization/18-kv-cache-basics:68-84`；
> ★★ **TensorRT-LLM 的 per-iteration 时间预算** → `45-kernel-lifecycle/requirement-discovery/03`。
> ★★★ **本文【没有 A 类】** —— 本机**无 GPU、无任何引擎** ⇒
> TensorRT-LLM 的**【行为与性能】无处可验**。★★ 「不是任何真实引擎，也不是真机」。
> ★ 本文的 `[A]` 是**外部来源的级别**（NVIDIA 官方文档），**不是**本库的 A 类判据。

## 0. 背景与动机

同一个「让请求不断进出」的目标，不同引擎**在什么时候付代价**上做了不同选择：

- **运行期决定形态**（vLLM 的路线，见 [01-vllm](01-vllm.md)）：调度参数是一张**按设备/场景**给的表，
  ★ **代价付在运行期**（受显存约束）；
- **编译期决定形态**（TensorRT-LLM 的路线）：把模型**编成引擎**，★ **代价付在编译期**。

★★★ **本文要讲的就是这第二种选择的【结构】**：**它的代价为什么是【乘性】的**。

★ **一个具体的对照场景**（本文程序里的真算例，`code/03_tensorrt_and_llamacpp.py` 的 `B.1`）：
若每个 `(max_batch_size, max_input_len, max_output_len)` 组合**各编一份**，
而三个轴各取 `3 / 3 / 2` 档 ⇒ **份数 = `3 × 3 × 2 = 18`**。
★★★ **加一档的代价取决于【加在哪条轴】**：把 **`3` 档轴**加一档 ⇒ **`18 → 24`**；
把 **`2` 档轴**加一档 ⇒ **`18 → 27`** —— ★★ **每加一档配置，份数就【乘】一次**（乘的是**其它轴**的乘积）。

### 0.5 ★★★ llama.cpp：本文为什么不展开（★ 请先读这一条）

★ 本主题的定位（见域 README）里，`engines` 要写「vLLM · SGLang · TensorRT-LLM · **llama.cpp**」
四家的**选择与差异**。★★★ **但本次取数【没有拿到 llama.cpp 的一手来源】**：
外部事实研究（`docs/superpowers/research/2026-10-09-inference-external-facts.md`）的 **13 条里，
没有一条是关于 llama.cpp 的**；唯一与它相关的记录是「**本机没有它**」。

⇒ ★★★ **处置（本库纪律：没有来源的数字比不写更糟）**：
- **不得**凭印象写它的实现细节（那会是一条**看起来像事实的社区转述**）；
- **不得**把「CPU-first 取向」这类说法**当成已核事实**写进来；
- ★★ **只把「未取到」本身写成一条边界**（本文程序 `C.3` 用一条 `[E]` 记录 + 一个机械检查钉住它）。

★★ **一条纪律**：**「数不到」≠「不存在」**。本文**不说**「llama.cpp 没有某性质」，
★ 只说**「本次取数没有取到它的一手来源，因此不展开」**。
★★ **要核它需要什么**：它的**官方仓库 / 官方文档**里关于「默认后端 / 量化格式 / 是否依赖 GPU」
的**逐字**表述 —— ★ 那是**下一次取数**的事，不是本文可以补的。

## 1. 原理

### 第一步：`in-flight batching` —— 一个 **[A] 级**的术语

NVIDIA 官方文档逐字（**`[A]` 级**，厂商第一方）：

> 「TensorRT-LLM supports **in-flight batching** of requests (**also known as continuous batching or
> iteration-level batching**) for higher serving throughput.」

★★★ **三点必须写清**：

1. ★ **这是 `[A]` 级**的术语同义声明 —— 它的**主体是 NVIDIA 对 TensorRT-LLM 自己**的表述；
2. ★★ **它把三个词声明为同义**：`in-flight batching` = `continuous batching` = `iteration-level batching`；
3. ★★★ **但「原始术语」归 ORCA（OSDI '22，`[C]`）** —— 那篇论文的词是
   **`iteration-level scheduling`**（逐字「schedules execution at the granularity of iteration
   (instead of request)」）。⇒ ★ **不得把 `in-flight batching` 当成 ORCA 的原始术语**（反之亦然）。

### 第二步：★ 该术语下的一条【顺序约束】

同一页逐字（**`[A]` 级**）：

> 「**In the current implementation**, the sequences that are going through the context phase
> **must be before** the sequences in the generation phase in the input tensor.」

★★★ **两个词决定了它的作用域**：**`In the current implementation`** —— 它是**该实现当前版本**的约束。
★ 同一页又逐字写着：

> 「The constraint **may or may not be relaxed in a future version**.」

⇒ ★★★ **不得把它说成「continuous batching 的通用约束」**（`§0.7 C` 事实 8）。

### 第三步：这条约束的【前提】是「张量必须紧凑」

同一页逐字（**`[A]` 级**）：

> 「the support for inflight batching requires the input tensors to be **packed (no padding)**」

★ 脚注给出了理由：把 generation 段（只含一个 token 的序列）**padding 到最长输入长度**，
是**浪费资源**。

### 第四步：★★ 顺序约束的【结构后果】—— 布局被唯一确定

★★★ **前提（紧凑、无 padding）＋ 要求（context 段在 generation 段之前）⇒ 布局【没有自由度】。**

因为张量是紧凑的（`packed`），一旦要求 context 段在前，**第一条 generation 段的下标就必须
恰好等于 `Σ(context 段长度)`** —— ★ 不是「大概排一下」，而是**被唯一确定**。
★ 本文程序的算例（`B.2`）：`context 段长度 = (4, 6, 5)`（共 `15`）+ `generation 段 2 条`（各 `1` token）
⇒ **紧凑张量总长 `17`**，**第一条 generation 段的下标 = `15`**。

| 布局 | 是否满足该实现当前版本的顺序约束 |
|---|---|
| `[c, c, c, g, g]` | ★ **满足** |
| `[g, c, …]`（generation 在前） | ★★ **违反** |
| 带 padding 的等长张量 | ★★ **前提不成立**（本来就要求 `packed`） |

## 2. 执行映射

引擎层。★ **本节的「量」全部来自本机程序，验证的是【结构账】，不是任何引擎的性能。**

**（1）★★ 编译式路线的代价结构：乘性增长。**
`(max_batch_size, max_input_len, max_output_len)` 三轴各 `3 / 3 / 2` 档 ⇒ 份数 `18`；
★★ 加一档的代价**看加在哪条轴**：`3` 档轴加一档 ⇒ `24`、`2` 档轴加一档 ⇒ **`27`**。★★ **对照**：运行期决定形态的引擎**只有一组参数**，代价随**显存**约束，不随**配置组合数**放大。

**（2）★★ 顺序约束 + packed ⇒ 布局唯一**：见 §1 第四步（算例 `17 / 15`）。

**（3）★ 该实现的顺序约束有一个「可放宽」的自述。** ⇒ ★★ **引用它必须连着这条自述一起引**。

**（4）★★★ llama.cpp 的边界**：见 §0.5 —— **未取到一手来源 ⇒ 本文不展开它的实现细节**。
★ 本文程序 `C.3` 把这条变成**可机械检查**的：`specs` 里**没有**任何 llama.cpp 的引文常量，
只有一个 `[E]` 边界记录（`LLAMACPP_FIRST_SOURCE_OBTAINED = False`）。

**（5）★★★ 这条链的产物是「结构账」，不是「真实性能」。**
本机**无 GPU、无任何引擎** ⇒ 上面的数**没有一个**来自真实系统。
★★★ **限定语**：「不是任何真实引擎，也不是真机」。

## 3. 实现与演进

**第一步：运行期决定形态（解释式）。** 引擎在启动时读配置、运行时挑请求 ——
★ **代价可控**（一组参数），★ 但**每步的 kernel 组合不固定**（vLLM 的路线，见 [01](01-vllm.md)）。

**第二步：编译式。** 把模型**编成引擎**（图固定、kernel 融合在编译期做）。
★★ **收益**：单次前向的执行更可预测；★★★ **代价**：**配置组合数【乘性】放大**（§1 第四步的 `18`）。

**第三步：`in-flight batching`。** 让编好的引擎也能处理**不断进出的请求** ——
★★ **它给这套机制一个 `[A]` 级的名字**（NVIDIA 的术语声明），
★ **代价是引入一条顺序约束**（§1 第二步），而该约束**可能在未来版本放宽**。

**第四步：把「优化优先级」变成可测的东西。** ★ 相关的一处正文（**本主题只回链**）：
TensorRT-LLM 的 **per-iteration 时间预算**（`45-kernel-lifecycle/requirement-discovery/03`）——
它把 `nsys` 的「哪个 kernel 最耗时」换成了「**这次迭代的墙钟时间去哪了**」。
★ 那是「编译式路线怎么被观测」的问题，**正文在彼处**。

**第五步（未做）：llama.cpp。** ★ 见 §0.5 —— **本文不展开**。

## 4. 踩坑与失败模式

### 4.1 把【该实现当前版本】的顺序约束当成通用约束

- **现象**：写「continuous batching 要求 context 段排在 generation 段之前」；
- **成因**：★★★ 原文写的是「**In the current implementation**」，且自述「**may or may not be
  relaxed in a future version**」—— ★ **限定语是句子的一部分**；
- **本文实测**（`code/03_tensorrt_and_llamacpp.py` 的 `C.1`）：「通用约束」这个论断**判为不成立**
  （因为限定语在）；★ 约束引文与限定语在**同一条断言**里钉住；
- **规避**：★★★ **引这条约束时连它的自述一起引**；★ 去掉限定就把它升格成了别的引擎未必有的规则。

### 4.2 把 `in-flight batching` 当成 ORCA 的原始术语

- **现象**：写「ORCA 提出了 in-flight batching」；
- **成因**：★ `in-flight batching` 是 **NVIDIA 对 TensorRT-LLM 自己**的术语；
  ORCA 的**原始术语**是 `iteration-level scheduling`；
- **本文实测**（`C.2`）：两条引文**逐字钉住**，且判定它们**不相等**；
- **规避**：★★ **术语三件套要一起写**：谁说的、什么级别、原始出处是谁。

### 4.3 把「编译式」读成「只要编一次」

- **现象**：以为编一个引擎就够；
- **成因**：★ 引擎是按配置编的 —— **三个配置轴各取 N 档 ⇒ 份数 = 乘积**；
- **本库算例**（`B.1`，★ 乘法推导、非实测）：`3 × 3 × 2 = 18`；
  ★ 把 `3` 档轴加一档 ⇒ `24`、把 `2` 档轴加一档 ⇒ **`27`**；
- **★★★ 一条反直觉的后果**：**加一档的边际代价随【其它轴】的大小放大** —— 同样只加一档，
  加在 `2` 档轴上（`18 → 27`，多 `9` 份）**比**加在 `3` 档轴上（`18 → 24`，多 `6` 份）**更贵**。
  ⇒ ★★ **「去收窄档位最多的那条轴」未必是省得最多的改法** —— 乘积结构下，
  **省下来的成本 = 少的档数 × 其它轴的乘积**；
- **规避**：★★ **把配置轴当成【乘性】成本来规划**（★ 且**看着乘积、不是看着单条轴**）；
  ★ 这与运行期决定形态的引擎是**两种代价结构**。

### 4.4 拿「未取到一手来源」的东西当事实写

- **现象**：写「llama.cpp 是 CPU-first 的，所以……」；
- **成因**：★★★ **本次取数没有 llama.cpp 的一手来源** —— 这类句子会变成**看起来像事实的社区转述**；
- **本文实测**（`C.3`）：`specs` 里**没有**任何 llama.cpp 的引文常量，只有一个 `[E]` 边界记录；
- **规避**：★★★ **只写「未取到」，不写「它是什么」**；★ **「数不到」≠「不存在」**。

## 5. 判据表

| 可机械判定的条件 | 结论 |
|---|---|
| 见到「`in-flight batching`」 | ★★ 是 **NVIDIA 对 TensorRT-LLM 自己**的术语（`[A]`），且**自己声明** `= continuous batching = iteration-level batching`（`D.5`/`C.2`） |
| 见到「`iteration-level scheduling` 的出处」 | ★★★ **ORCA（OSDI '22，`[C]`）** —— **不是** NVIDIA，也**不是** vLLM（`C.2`） |
| 见到「context 段必须在 generation 段之前」 | ★★★ **是【该实现当前版本】的约束** —— 原文写 `In the current implementation`，且自述**可能放宽** ⇒ **不得说成通用约束**（`C.1`） |
| 见到「in-flight batching 的输入」 | ★★ **必须 `packed`（无 padding）**（`D.5`） |
| 见到「顺序约束带来的布局问题」 | ★★ **紧凑 + 顺序 ⇒ 布局被唯一确定**（第一条 generation 段下标 = `Σ context 长度`，`B.2`） |
| 见到「编译式路线编几份」 | ★★ **三轴乘积**（`3×3×2 = 18`；★ `3` 档轴加一档 `24`、`2` 档轴加一档 `27`）—— ★ 是**本主题的模型假设**，不是任何引擎的实测（`B.1`） |
| 见到「llama.cpp 的实现细节」 | ★★★ **本文不展开** —— **本次取数未取到它的一手来源**（`C.3`）；★ **不得凭印象写** |
| 见到「本库的 A 类判据」 | ★★★ **本文【没有】** —— 本机无引擎 ⇒ `D.0` 用 AST 证明 `check("A…")` == `0`；★ 钉住归 `D` 类 |
| 见到「请求调度器的机制」 | ★★★ **不在本文** —— 本域 [01-iteration-level-scheduling](../batching-and-scheduling/01-iteration-level-scheduling.md) |
| 见到「per-iteration 时间预算」 | ★★★ **不在本文** —— `45-kernel-lifecycle/requirement-discovery/03`（**正文在彼处**） |
| 见到「TensorRT-LLM 的真实吞吐 / 延迟」 | ★★★ **本机测不了**（无 GPU、无引擎）⇒ 只能引 `[A]`/`[B]` 级值，**必须带条件** |

## 6. 小结与延伸

**一句话**：**TensorRT-LLM 把「什么时候付代价」放在编译期 —— 它的成本随配置轴【乘性】放大；
它给 `in-flight batching` 一个 `[A]` 级的名字，但那条顺序约束是【该实现当前版本】的。**

**三条可独立复用的结论**：

1. ★★★ **限定语是句子的一部分**：`In the current implementation` 一丢，一条实现约束就被升格成通用规律；
2. ★★ **编译式 vs 运行期的代价结构不同**：一个是配置轴的**乘积**，一个是受显存约束的**一组参数**；
3. ★★★ **「未取到一手来源」要写成边界，不要写成事实**（llama.cpp 的处置）。

**延伸**：

- ★★ **vLLM 的架构选择**（引擎循环 / 参数面）→ [01-vllm](01-vllm.md)；
- ★★ **SGLang 的 RadixAttention**（前缀树 / LRU / 两处口径差）→ [02-sglang-and-radix](02-sglang-and-radix.md)；
- ★ **请求调度器的机制** → 本域
  [01-iteration-level-scheduling](../batching-and-scheduling/01-iteration-level-scheduling.md)；
- ★★ **TensorRT-LLM 的 per-iteration 时间预算** → `45-kernel-lifecycle/requirement-discovery/03`；
- ★★★ **PagedAttention 的定义与算式** → `130-model-optimization/quantization/18-kv-cache-basics:68-84`。

## 参考

- **`[A]` NVIDIA 官方文档** —— `https://nvidia.github.io/TensorRT-LLM/advanced/gpt-attention.html`
  （厂商第一方；抓取日 **2026-10-09**）。★ 注：`.../performance/optimization.html` 实测 **HTTP 404**，
  有效来源改用 `advanced/gpt-attention.html`。逐字依据：`in-flight batching` 的术语同义声明 ·
  `packed (no padding)` 要求 · **顺序约束** · **可能放宽**的自述。
- **`[C]` ORCA（OSDI '22）** —— `https://www.usenix.org/conference/osdi22/presentation/yu`；
  原始术语 `iteration-level scheduling`。
- **`[E]` 本主题的实现** —— 配置轴的**乘积**模型是**本主题的假设**（`TRTLLM_CONFIG_AXES`），
  ★ **不是**任何引擎的实测编译次数（本机无引擎）。
- ★ **一条术语纪律**：本文一律写「**KV 缓存**」「**请求调度器**」全称；
  ★ 引用 `in-flight batching` 时**必须写明是 NVIDIA 对 TensorRT-LLM 的表述**。
- ★ **跨域回链**（**正文在彼处，本文只回链**）：
  · 请求调度器 → 本域 `batching-and-scheduling/01`；
  · 分页 / 前缀树 → 本域 `kv-cache/02,03`；
  · per-iteration 时间预算 → `45-kernel-lifecycle/requirement-discovery/03`。

## 编译与运行

```sh
cd code && make run                       # 三个编号程序，任一断言失败即非零退出
python3 03_tensorrt_and_llamacpp.py        # 只跑本文的配套程序
make 03_tensorrt_and_llamacpp             # ★★ 与上一行**不等价**：它先过 check-env（查解释器 / numpy）再跑
```

★ 本机的运行条件：`python3 3.13.5`、`numpy 2.4.4`、macOS arm64，**无 GPU、无任何引擎**。
★★★ **本文的程序【没有 A 类】**（本机无引擎 ⇒ 引擎的行为与性能无处可验）——
`D.0` 用 **AST 扫自己的源码**证明 `check("A…")` 的调用数 == `0`，**一切钉住归 `D` 类**。
★ `B.1`/`B.2` 是**结构账**（编译式份数 / 顺序约束的后果）；`C.1`–`C.3` 是**三条禁写与一条边界**
（顺序约束不得升格 / 术语不得误归 / llama.cpp 不得凭印象写）；★ `D.9` 是**术语纪律**。
★ **本文对应 `code/03_tensorrt_and_llamacpp.py`**（三类断言全在程序内部，失败即非零退出）。
